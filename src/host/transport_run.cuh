// The transport engine: uploads a flattened scene and runs events on it.
//
// This is src/host/b1_gpu_sched.cu's loop, generalised - any number of volumes, materials and
// scorers, and primaries from a Source description rather than a hard-coded beam. The
// scheduling is unchanged and deliberately so: tracks live in structure-of-arrays buffers, one
// per species for warp coherence, advanced one step per kernel launch, with the output buffer
// compacting implicitly as survivors and secondaries are appended.
//
// Scoring note. A scorer slot is chosen from the volume the track occupied at the *start* of
// the step, which is where every deposit in the stepper is attributed: a step ends either at
// an interaction (deposit first, no crossing) or at a boundary (crossing last, after the
// continuous loss has been attributed). Using the post-step volume would move the continuous
// loss of a boundary-terminated step into the next volume.
#pragma once
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "core/step_hook.cuh"
#include "g4/G4Flatten.hh"
#include "host/hadronic_upload.cuh"
#include "host/level_upload.cuh"
#include "host/pe_upload.cuh"
#include "physics/source.cuh"
#include "physics/stepper.cuh"
#include "render/trajectory.cuh"

namespace g4gpu::host {

#define G4GPU_CUDA_CHECK(call)                                                             \
  do {                                                                                     \
    const cudaError_t err__ = (call);                                                      \
    if (err__ != cudaSuccess) {                                                            \
      std::printf("CUDA error %s at %s:%d\n", cudaGetErrorString(err__), __FILE__,         \
                  __LINE__);                                                               \
      std::exit(1);                                                                        \
    }                                                                                      \
  } while (0)

template <typename real_t>
struct DeviceTracks {
  TrackBuffer<real_t> view{};
  bool slab_backed_ = false;

  /// With `slab`, the arrays are carved from a shared allocation and this object owns none of
  /// them; free_all() then only clears the view. Without one it allocates as it always did,
  /// which is what b1_gpu_sched.cu still does.
  void alloc(int capacity, TrackSlab* slab = nullptr) {
    G4GPU_CUDA_CHECK(allocate_track_buffer(view, capacity, nullptr, slab));
    slab_backed_ = (slab != nullptr);
    if (slab_backed_) {
      // The two counters get their own allocations and never live in the slab.
      //
      // Kept that way on the strength of what it cost when they did not. The arena used to be
      // re-divided between the species mid-run, and a re-carve lays the arrays out from the
      // slab's offset zero again - so the counters, carved after the arrays, were written
      // straight over by track data on the next division while the pointers to them survived.
      // The count read back was garbage, the next launch sized itself from it, and the kernel
      // walked off the end of the buffer. Found by forcing an overflow on purpose; it could
      // not happen while the pool was generous, which was every run that did not go looking.
      //
      // There is no re-division any more - one pool, carved once per side - so the hazard is
      // gone with it. Two eight-byte allocations outside the slab is still the right answer
      // for a value whose whole job is to survive whatever happens to the arrays.
      G4GPU_CUDA_CHECK(cudaMalloc(&view.count, sizeof(int)));
      G4GPU_CUDA_CHECK(cudaMalloc(&view.overflow, sizeof(int)));
    }
    reset();
    G4GPU_CUDA_CHECK(cudaMemset(view.overflow, 0, sizeof(int)));
  }
  void reset() { G4GPU_CUDA_CHECK(cudaMemset(view.count, 0, sizeof(int))); }
  /// Clears the dropped-track counter. Called before each attempt at a batch, so that what is
  /// read afterwards belongs to that attempt and not to one that was already retried.
  void reset_overflow() { G4GPU_CUDA_CHECK(cudaMemset(view.overflow, 0, sizeof(int))); }
  /// Tracks in the buffer, never more than it can hold.
  ///
  /// The clamp is load-bearing. append() bumps the cursor with an unconditional atomicAdd
  /// and only then checks whether the slot exists, so after an overflow the raw cursor is
  /// larger than the capacity - it counts what was OFFERED. The drain loop sizes the next
  /// kernel launch from this, one thread per track, and each thread loads its own index, so
  /// an unclamped cursor would put threads to work reading past the end of every array in
  /// the buffer.
  ///
  /// That was unreachable while the species capacities were four times what B1 needed. The
  /// adaptive pool makes them tight, which makes this reachable, which is why it is here.
  /// The dropped tracks are still counted - see overflow() - so nothing is hidden by it.
  int count() const {
    int c = 0;
    G4GPU_CUDA_CHECK(cudaMemcpy(&c, view.count, sizeof(int), cudaMemcpyDeviceToHost));
    return (c > view.capacity) ? view.capacity : c;
  }
  int overflow() const {
    int c = 0;
    G4GPU_CUDA_CHECK(cudaMemcpy(&c, view.overflow, sizeof(int), cudaMemcpyDeviceToHost));
    return c;
  }
  void free_all() {
    if (slab_backed_) {
      // The arrays were carved from the engine's arena, which the engine frees in one
      // piece; the two counters are ours.
      cudaFree(view.count);
      cudaFree(view.overflow);
      view = TrackBuffer<real_t>{};
      return;
    }
    free_track_buffer(view);
  }
};

// ---------------------------------------------------------------- engine

struct RunStats {
  /// GPU time for the batch loop: primary upload, the stepping kernels, the score readback
  /// and the per-event sink callbacks. Measured with CUDA events, so it excludes everything
  /// the host did before the first kernel.
  double milliseconds = 0;
  /// Host wall clock for the whole event loop - the primaries generated one event at a time
  /// through the user's GeneratePrimaries, then the above. This is the number that compares
  /// against Geant4's, whose G4Timer brackets InitializeEventLoop to TerminateEventLoop and so
  /// also covers primary generation while excluding physics-table building. `milliseconds` is
  /// a strictly smaller number measuring a strictly smaller thing, and quoting it against
  /// Geant4's would be flattering rather than wrong.
  double event_loop_ms = 0;
  long long track_steps = 0;
  int max_iterations = 0;
  /// Times a stepping action asked for a track status this transport cannot do what Geant4
  /// would do with: fSuspend, fPostponeToNextEvent (nowhere to put a track aside - every track
  /// in flight is in a species buffer being stepped in lockstep) or fKillTrackAndSecondaries
  /// (whose secondaries are already in their buffers by the time a hook runs, so only the
  /// track itself dies).
  ///
  /// Those requests are honoured as far as they can be - the track is killed - and counted
  /// here, because a run that asked for something it did not get should say so rather than
  /// leave someone to find out from a dose that is quietly wrong.
  /// Tracks that were alive but not stepped this iteration, because the output buffer had no
  /// room for what they would have produced. They were carried forward untouched and stepped
  /// later, so nothing is lost - this is the cost of a tight pool, measured in deferred steps
  /// rather than in a wrong answer.
  long long throttled = 0;
  /// Events that lost a track and were transported again on a repair pass. Not an error - the
  /// answer is the one a run with room to spare would have given - but every one of them cost
  /// its shower being computed twice. If this is large, raise SetLiveTracksPerEvent.
  long long events_repaired = 0;
  /// Batches that ran out of buffer space and were re-run with a smaller batch. Not an error -
  /// every event is still transported, and the answer is the same one a correctly sized run
  /// would have produced - but it costs the work already done on the attempt that overflowed.
  /// If this is large, raise SetLiveTracksPerEvent so the first attempt fits.
  long long batch_retries = 0;
  long long unsupported_track_status = 0;
  /// Secondaries that did not fit the arena behind GetSecondaryInCurrentStep(), and so are
  /// missing from the lists a stepping action saw. GetNumberOfSecondariesInCurrentStep() stays
  /// exact regardless. Zero in every run of this project's pipeline; if it is not zero, raise
  /// the arena with SetSecondaryArenaCapacity.
  long long secondary_overflow = 0;
  /// Peak live tracks of each species, indexed by TrackSpeciesIndex.
  ///
  /// Five named fields before - peak_gamma through peak_alpha - which is the shape that has to
  /// be edited every time a species is added, in a struct whose whole job is to report on a
  /// run whose species set is not fixed. An array and species_of_index() name themselves.
  int peak_live[kNumTrackSpecies] = {};
  /// Energy carried out of each event by particles this transport books rather than steps -
  /// the neutrinos - in MeV, one entry per event of the whole run. Empty when nothing made one.
  ///
  /// Per event because that is the only form in which it is checkable: a shower's balance is
  /// `deposited + escaped + carried away = primary energy`, and a run total cannot be held
  /// against a single primary. See BufferEmitter::carried_away.
  std::vector<double> carried_away;
  /// How many were made of each flavour, and the total energy, so a run that made none says so
  /// rather than reporting a zero that could equally mean the accounting is off. Indexed by
  /// ParticleType.
  long long carried_by_species[static_cast<int>(ParticleType::kNumTypes)] = {};
  long long carried_away_n = 0;
  double carried_away_total = 0;
  /// Secondaries this port refuses to transport, counted at the point of emission and indexed
  /// by ParticleType so the report names each one. Hyperons, K0L/K0S, anti-nuclei, b/c hadrons.
  ///
  /// Non-zero is not an error in the sense that the run failed - it is an error in the sense
  /// that the answer is missing those particles' energy, and it says which particles. A primary
  /// of a refused species never gets this far: BeamOn ends the run instead.
  long long refused_by_species[static_cast<int>(ParticleType::kNumTypes)] = {};
  long long refused_total = 0;
  /// The kinetic energy of those refused secondaries, MeV, by species and in total. See
  /// EmitterBooks::refused_energy: a count is not a size, and an elastic recoil heavier than an
  /// alpha is refused for want of ion transport rather than for want of physics.
  double refused_energy_by_species[static_cast<int>(ParticleType::kNumTypes)] = {};
  double refused_energy_total = 0;
  /// Hadronic processes this transport reached and could not apply, by
  /// `had::HadronicRefusal`, with the energy each one cost. See physics/hadronic/wiring.cuh.
  long long had_refused_count[static_cast<int>(had::HadronicRefusal::kNumHadronicRefusals)] =
      {};
  double had_refused_energy[static_cast<int>(had::HadronicRefusal::kNumHadronicRefusals)] = {};
  long long had_refused_total = 0;
  /// Energy discarded by the neutron time cut, MeV, and how many neutrons it killed.
  ///
  /// Separate from `carried_away` because it is a different kind of loss and conflating them
  /// would hide both. A neutrino's energy leaves the event because it physically leaves; a
  /// neutron aged past 10 us has its energy DELETED, by a process whose stated purpose is to
  /// save CPU (`G4NeutronKiller.cc`: "The process to kill particles to save CPU"). Geant4 does
  /// the same thing and also does not conserve energy across it - see step_neutral - so
  /// reproducing it is right and reporting it is the only way anyone finds out.
  long long neutron_killed_n = 0;
  double neutron_killed_energy = 0;
  long long abandoned = 0;
  int overflow = 0;
};

/// A per-event callback, so that a G4UserEventAction and a G4UserSteppingAction can be
/// called the way Geant4 calls them.
///
/// The scores are already per event on the device - d_score_ is n_scorers by batch, indexed
/// by the event's slot in the batch - so this costs a host virtual call per event and no
/// extra device work. At 2.7 million events per second that is a few per cent of the run,
/// which is why it is only installed when a user action asks for it.
struct EventSink {
  virtual ~EventSink() = default;
  /// @p scores has one entry per scorer, in device_index order, for event @p event_id.
  virtual void Event(int event_id, const double* scores, int n_scorers) = 0;
};

/// Owns the device-side scene and the track buffers, and runs events on them.
///
/// @tparam StepHook a device functor called once per step of every track - the general
///         step-level customisation point, and the only one that sees a real step rather than
///         an event aggregate. See core/step_hook.cuh, which is where the important part is:
///         a hook must REDUCE on the device, because anything that stores per step scales
///         with the amount of transport and so fails on exactly the runs worth doing.
///
///         The default is StepTap, which materialises steps into a capped buffer. That is the
///         right default only because it is the one hook that answers an arbitrary question
///         without being recompiled, and it is gated at runtime by a null pointer, so the
///         stock build pays one predicated load per step and nothing else. It is for
///         debugging and small runs. For production, instantiate this class on a StepTally
///         (or any other reducer) whose footprint is set by the question rather than by the
///         number of steps:
///
///             using QHook = StepTally<double, BinByEvent, QWeight>;
///             TransportEngine<double, QHook> engine;
///
///         and add `template class TransportEngine<double, QHook>;` next to the existing
///         explicit instantiation at the bottom of transport_run.cu. A custom hook means
///         rebuilding that translation unit either way - the kernels live there.
template <typename real_t, typename StepHook = StepTap<real_t>>
class TransportEngine {
 public:
  /// Uploads @p scene. Materials were built with their production cuts already applied.
  ///
  /// @p batch_size 0 - the default - sizes the batch from the memory the device actually has
  /// free, which is what you want unless you have a reason not to. Pass a positive number to
  /// choose it yourself; it is honoured even if it does not fit, with a warning saying by how
  /// much, because a caller who asked for a specific batch usually has a reason and would
  /// rather see the allocation fail than be silently given a different run.
  ///
  /// Automatic sizing matters more than it used to. A track carries the G4Track block now, so
  /// a slot is 232 bytes rather than 88, and the batch that used to be the fixed default -
  /// 1048576 - needs 4.1 GB of track buffers where it once needed 1.6. That is more than an
  /// 8 GB card has free with a desktop running, which is not a hypothetical: it is how this
  /// was found.
  void Upload(const g4::FlatScene& scene, int batch_size = 0, int threads = 128);

  /// What Upload actually chose. Only meaningful after Upload.
  int ChosenBatch() const { return batch_; }

  /// Let a run finish even though particles were dropped for want of buffer space.
  ///
  /// Off by default, and it should stay off for anything whose answer matters: a dropped track
  /// is a particle that stopped being transported, so the dose comes out low and the run still
  /// looks like it worked. With this off, such a run ends instead of reporting.
  void AllowDroppedTracks(bool yes) { allow_dropped_ = yes; }
  bool DroppedTracksAllowed() const { return allow_dropped_; }

  /// Live track slots reserved per event, shared across all species. Default 4.
  ///
  /// This replaces five fixed per-species capacities that summed to 8.5 slots per event. They
  /// were guesses, and measurement showed them wrong in shape as well as size: B1 peaks at 1.00
  /// live gammas per event against 2 reserved, and 0.90 electrons against 4. A single pool
  /// divided by what is actually alive needs far less, because memory reserved for electrons is
  /// now available to gammas when the shower does not go that way.
  ///
  /// 4 is measured B1 peak (2.0) doubled. What this number bounds is CONCURRENCY - tracks alive
  /// at one instant, summed over the batch - not how many secondaries a step may create or how
  /// many tracks an event may produce. Raise it for a problem with higher multiplicity than a
  /// 6 MeV gamma beam; if it is too low the engine says so, loudly, because a dropped track is
  /// a dose that is too low rather than a slow run.
  ///
  /// Set before Upload.
  void SetLiveTracksPerEvent(double n) { live_per_event_ = (n < 1.0) ? 1.0 : n; }
  double GetLiveTracksPerEvent() const { return live_per_event_; }

  /// The share of FREE device memory the track buffers may occupy. Default 0.55.
  ///
  /// It governs both halves of the batch decision, deliberately: an automatic batch is the
  /// largest that fits inside this share, and a batch you set yourself is refused if it does
  /// not. One number, so raising it raises both - and so that "how much of the card may this
  /// run take" has a single answer rather than two that could disagree.
  ///
  /// It is a share of what is FREE rather than of what the card has, because a desktop is
  /// usually holding a gigabyte or two and that memory is not available whatever the card's
  /// specification says.
  ///
  /// Why not all of it: the track buffers are the biggest allocation but not the only one.
  /// The scene follows them - and a voxel phantom or a triangle mesh is not small - along with
  /// the per-event score array, the secondary arena, and whatever the driver keeps back.
  /// Leaving room means being wrong about the estimate costs throughput instead of a run.
  /// Raise it if you know what else is on the card; on a headless compute GPU 0.8 is
  /// reasonable, and on a desktop card driving monitors it is not.
  ///
  /// Set before Upload. Clamped to [0.05, 0.95] - 1.0 is not offered, because the allocations
  /// that follow the track buffers are not optional.
  void SetMemoryFraction(double f) {
    mem_fraction_ = (f < 0.05) ? 0.05 : ((f > 0.95) ? 0.95 : f);
  }
  double GetMemoryFraction() const { return mem_fraction_; }

  /// Runs @p n_events from primaries the host generated, filling `score_sum` and
  /// `score_sum_sq` with one entry per scorer.
  ///
  /// @p primaries points at n_events records; the engine uploads them a batch at a time.
  /// They come from G4VUserPrimaryGeneratorAction::GeneratePrimaries, called once per event,
  /// so a primary can be anything a user's code can compute. The device used to sample a
  /// Source description instead, which was faster and closed the set of possible primaries to
  /// whatever distributions the gun offered - a Gaussian beam spot could not be written.
  ///
  /// @p seed keys the per-track RNG, so a given seed reproduces a run's showers.
  /// A non-empty @p traj captures trajectory segments for the first `traj.max_event` events.
  /// A non-null @p sink is called once per event with that event's scores.
  /// @param stream_pos where in the seed's random stream this run's first primary sits. Zero
  ///        replays the start of the stream, which is what a freshly started process wants;
  ///        G4RunManager advances it by one per primary so that a second BeamOn in the same
  ///        process is an independent sample rather than a repeat of the first.
  RunStats BeamOn(int n_events, const Primary<real_t>* primaries, unsigned int seed,
                  std::vector<double>& score_sum, std::vector<double>& score_sum_sq,
                  vis::TrajectoryBuffer traj = vis::TrajectoryBuffer{},
                  EventSink* sink = nullptr, long long stream_pos = 0);

  void Free();

  const Scene<real_t>& scene() const { return scene_; }
  const geom::Geometry<real_t>& geometry() const { return geom_; }
  const std::vector<data::Material<real_t>>& materials() const { return h_mats_; }
  int batch() const { return batch_; }

  /// Energy deposit per voxel cell after the last BeamOn, in MeV, indexed by cell. Empty when
  /// no scorer asked for per-voxel scoring. Accumulated over the whole run rather than per
  /// event: a per-event copy of a 512-cubed grid is 130 million doubles a batch.
  const std::vector<double>& voxel_scores() const { return h_voxel_score_; }

  /// Which processes to run. Must be set before Upload; all on is the validated default.
  void SetProcesses(const ProcessFlags& f) { processes_ = f; }
  const ProcessFlags& GetProcesses() const { return processes_; }

  /// Which Geant4 configuration this run is the like-for-like partner of, and which of P8's
  /// hadronic processes are active. See physics/hadronic/wiring.cuh.
  ///
  /// A RUN-TIME setting, so that both columns of the comparison table come out of one binary -
  /// docs/RISK.md V45 is what a second binary costs. Set before BeamOn; the wiring struct is
  /// built at each launch, so this can change between runs in one process.
  void SetHadronicStage(had::HadronicStage s) { had_stage_ = s; }
  had::HadronicStage GetHadronicStage() const { return had_stage_; }
  void SetHadronicProcesses(bool decay, bool elastic, bool capture) {
    had_decay_ = decay;
    had_elastic_ = elastic;
    had_capture_ = capture;
  }

  /// Read PhotonEvaporation5.7 and upload it, so a capture cascade on the device has a level
  /// scheme to walk. Set before Upload. **Off by default, and that is a statement about the
  /// port and not a preference.**
  ///
  /// Geant4 does this unconditionally - `G4ExcitationHandler::SetParameters` calls
  /// `G4NuclearLevelData::UploadNuclearLevelData(Zmax+1)` at initialisation whether a neutron
  /// ever arrives or not - and this port will too, the day the neutron general process is
  /// wired. Today it is not (`physics/hadronic/neutron_general_xs.cuh`, and `Upload`'s refusal
  /// below), so the only consumer of the table is unreachable, and the table costs
  /// `read_all_level_data` opening **3110 files** against a B1 run whose whole transport is
  /// 750 ms. Paying that in every gamma run for something nothing reads is the wrong default.
  ///
  /// What it is NOT is a switch on the physics: the table is checked against the host copy
  /// level by level and the cascade on top of it by `tests/test_capture_device.cu`, which calls
  /// `host/upload_level_data` directly, so the path is exercised whatever this is set to.
  void SetNuclearLevelData(bool on) { load_level_data_ = on; }
  bool GetNuclearLevelData() const { return load_level_data_; }

  /// How many secondaries the arena behind GetSecondaryInCurrentStep() can hold in one kernel
  /// launch, across every track in flight. Not a per-step limit - a step may create as many
  /// secondaries as physics makes - and the default is sized against the batch. Set before
  /// Upload.
  void SetSecondaryArenaCapacity(int n) { sec_capacity_ = n; }

  /// The per-step hook, copied by value into every kernel launch. Set it before BeamOn; the
  /// copy the kernels get is taken at launch, so any device pointers inside it must already
  /// be allocated.
  void SetStepHook(const StepHook& h) { hook_ = h; }
  StepHook& GetStepHook() { return hook_; }
  const StepHook& GetStepHook() const { return hook_; }

 private:
  static std::vector<int> DistinctZ(const g4::FlatScene& scene) {
    std::vector<int> zs;
    for (int i = 0; i < scene.materials.count; ++i) {
      const auto& m = scene.materials.m[i];
      for (int e = 0; e < m.n_elements; ++e) {
        const int z = static_cast<int>(m.z[e] + 0.5);
        bool seen = false;
        for (int q : zs) {
          if (q == z) { seen = true; break; }
        }
        if (!seen) { zs.push_back(z); }
      }
    }
    std::sort(zs.begin(), zs.end());
    return zs;
  }

  int batch_ = 0, threads_ = 128, n_volumes_ = 0, n_materials_ = 0, n_scorers_ = 1;
  ProcessFlags processes_{};
  StepHook hook_{};
  int sec_capacity_ = 0;
  double mem_fraction_ = 0.55;
  double live_per_event_ = 4.0;
  bool allow_dropped_ = false;
  geom::Volume<real_t>* d_vols_ = nullptr;
  data::Material<real_t>* d_mats_ = nullptr;
  em::RangeTable<real_t>* d_rt_ = nullptr;
  geom::Solid<real_t>* d_pool_solids_ = nullptr;
  geom::Transform<real_t>* d_pool_xforms_ = nullptr;
  real_t* d_pool_aux_ = nullptr;
  short* d_voxels_ = nullptr;
  /// Per-cell class index and per-class layer, uploaded only when some voxel class was given
  /// a layer of its own. Null otherwise, which is what switches the navigator's per-point
  /// layer path off. See geom::VoxelStore.
  short* d_voxel_class_ = nullptr;
  int* d_class_layer_ = nullptr;
  unsigned char* d_class_absent_ = nullptr;
  real_t* d_tri_ = nullptr;
  real_t* d_bvh_ = nullptr;
  double* d_score_ = nullptr;
  /// Device counter behind RunStats::unsupported_track_status.
  int* d_status_warn_ = nullptr;
  /// The one allocation every species buffer is carved out of. See TrackSlab.
  char* d_track_arena_ = nullptr;
  /// Event ids that lost a track, so a repair pass can transport just those events again
  /// instead of the whole batch. See the repair block in BeamOn.
  int* d_dropped_ = nullptr;
  int* d_dropped_count_ = nullptr;
  int dropped_capacity_ = 0;
  /// Bytes in each half of the arena; the two halves ping-pong.
  size_t arena_half_ = 0;
  /// Live-track slots each half may hold, shared across the species. See
  /// SetLiveTracksPerEvent.
  long long pool_ = 0;
  /// The arena behind G4Step::GetSecondaryInCurrentStep. Reset once per kernel launch, so it
  /// only ever has to hold the secondaries made by one launch rather than by a whole run.
  SecondaryArena sec_{};
  /// Energy deposit per voxel cell, summed over the whole run, or null when no scorer asks
  /// for it. One entry per cell in the scene's voxel pool, so a scene with no voxel volume
  /// allocates nothing.
  /// One batch of primaries, uploaded per batch. See BeamOn.
  Primary<real_t>* d_primaries_ = nullptr;
  double* d_voxel_score_ = nullptr;
  int n_voxel_cells_ = 0;
  em::RangeTable<real_t> h_rt_{};
  /// Built only when the scene can actually see a hadron - it is 256 bins x 2 species x
  /// every material, and a photon run has no use for it. Null in Scene when not built.
  em::HadronRangeTable<real_t>* d_hrt_ = nullptr;
  /// `hadElastic`'s device tables and the allocations behind them. See
  /// host/hadronic_upload.cuh; the view inside it is copied into every kernel launch as part
  /// of `had::HadronicWiring`.
  ElasticTableOwner<real_t> elastic_tables_{};
  /// PhotonEvaporation5.7 on the device, and the host copy it was uploaded from. Empty unless
  /// `SetNuclearLevelData(true)` was called before Upload; see that setter for why.
  LevelTableOwner level_tables_{};
  data::LevelTableStorage level_storage_{};
  bool load_level_data_ = false;
  std::vector<data::Material<real_t>> h_mats_;
  std::vector<double> h_voxel_score_;
  geom::Geometry<real_t> geom_{};
  Scene<real_t> scene_{};
  /// Every track in flight, of every species, on each side of the ping-pong.
  ///
  /// There used to be five of these per side, one per species, and their capacities had to be
  /// guessed before the run - which is what tied the memory layout to the physics list and made
  /// "how many electrons might there be" a question somebody had to answer in advance. There is
  /// one number now: how many tracks may be alive. Whether they are gammas or alphas is the
  /// pool's business only in so far as each carries its own species.
  DeviceTracks<real_t> tracks_[2];

  /// ONE index list per side, bucketed by species. Rebuilt every iteration.
  ///
  /// Dispatch needs each species contiguous - a warp whose threads take different physics paths
  /// serialises through all of them - and this is what provides it. It used to be one ARRAY per
  /// species, each sized at the whole pool because any one species may in principle be all of
  /// it, and that made the memory cost of naming a species `4 * pool` bytes a side. With five
  /// species that was 40 bytes on top of a 236-byte slot, which was worth not thinking about.
  /// With sixteen it is 128, a 54% surcharge on the track arena and therefore a 35% smaller
  /// batch - paid by a gamma run that will never see a kaon.
  ///
  /// A counting sort removes the dependence entirely: one array of `pool_` ints, a histogram
  /// pass, a host-side prefix sum, and a scatter pass that writes each species into its own
  /// contiguous range. Four bytes a slot for any number of species. The cost is a second kernel
  /// launch per iteration - B1 drains in about 110 of them, at a few microseconds each, against
  /// a batch that takes some hundreds of milliseconds.
  int* idx_[2] = {};
  /// kNumTrackSpecies ints per side, used twice per iteration: as the histogram, read back to
  /// the host to build the offsets, then zeroed and reused as the scatter cursors. One
  /// allocation, and the offsets travel to the scatter kernel as a by-value argument rather
  /// than through a third buffer.
  int* idx_n_[2] = {};
  /// Tracks whose species no kernel steps. A tripwire rather than an accounting line: the
  /// emitter refuses such a species before it can be appended and BeamOn refuses it as a
  /// primary, so anything counted here got past both guards.
  int* d_unknown_ = nullptr;
  /// Energy carried out of each event by a booked-not-stepped species. batch_ doubles.
  double* d_carried_away_ = nullptr;
  /// Counts of booked-not-stepped secondaries by ParticleType, and refused ones. Both written
  /// at emission by BufferEmitter::push through EmitterBooks.
  int* d_carried_n_ = nullptr;
  int* d_refused_ = nullptr;
  double* d_refused_e_ = nullptr;
  /// P8's per-refusal ledgers: one count and one energy per `had::HadronicRefusal`.
  int* d_had_refused_n_ = nullptr;
  double* d_had_refused_e_ = nullptr;
  had::HadronicStage had_stage_ = had::HadronicStage::kStage1;
  bool had_decay_ = true;
  bool had_elastic_ = true;
  bool had_capture_ = true;
  /// Energy and count discarded by the neutron time cut. Two words, written by run_step_neutral.
  double* d_killed_energy_ = nullptr;
  int* d_killed_n_ = nullptr;
  /// The neutron's combined cross-section table, or null.
  ///
  /// Null in every run today and that is the state the port is in, not a switch: P2 produces
  /// the four cross sections and P8 sums them onto G4NeutronGeneralProcess's grid and writes
  /// the final states. See physics/hadronic/neutron_general_xs.cuh for the contract, and the
  /// refusal in Upload() for why a table without final states is not an allowed state.
  had::NeutronGeneralXs<real_t>* d_neutron_xs_ = nullptr;
};


// The kernels this engine launches are __global__ templates. A __global__ template defined in
// a header gets a device stub emitted into every translation unit that instantiates it, and
// those stubs collide at link time ("is not a specialization of a function template"). So the
// kernels and these method bodies live in transport_run.cu, and the instantiation for double
// is declared here and defined there.
extern template class TransportEngine<double, StepTap<double>>;

}  // namespace g4gpu::host
