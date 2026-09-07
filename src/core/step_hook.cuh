// Step-level customisation: what a G4UserSteppingAction would be if it could run on a GPU.
//
// THE PROBLEM
//
// Geant4 calls UserSteppingAction once per step of every track, with arbitrary host C++ in the
// callback. This transport runs ~3e7 track-steps per second with tens of thousands of tracks
// in flight; a host virtual call per step would cost more than the physics and would serialise
// the device. So the G4UserSteppingAction this port offers is handed an *aggregate* - one
// pseudo-step per volume per event - and g4/G4Step.hh is blunt about what that costs you: a
// stepping action that sums gets the right answer, and one that does anything else does not.
//
// That is fine for dose. It is useless for anything that needs a step to be a step: the
// kinetic energy of a particle as it crosses a boundary, the LET of the step that deposited
// the energy, a spectrum of anything, a threshold per step, a coincidence between volumes.
//
// THE MECHANISM: a device functor called once per real step
//
// A StepHook is a __device__ callable invoked once per step of every track, holding the real
// pre- and post-step state:
//
//     struct MyHook {
//       __device__ void operator()(const DeviceStep<double>& s) const {
//         if (s.entered(kDetector)) {
//           // s.ekin_pre is the energy it crossed the boundary with
//         }
//         const double let = s.let();   // MeV/mm, this step
//       }
//     };
//
// The mechanism itself stores nothing. DeviceStep is a view assembled in registers from state
// the kernel already holds, and NoStepHook compiles to nothing at all. Whatever a step-level
// analysis costs in memory is decided by what the hook *does*, which is the subject of the
// next section and is the most important thing on this page.
//
// MEMORY: reduce on the device; do not materialise steps
//
// The tempting hook is one that copies each step into a buffer for the host to read. Do the
// arithmetic before reaching for it. A 2M-event B1 run is ~2.6e7 track-steps; at 104 bytes a
// record - sizeof(StepRecord<double>) - that is 2.7 GB, and B1 is four volumes with one
// pencil beam. A patient CT, a
// shielding problem, or a detector with 1e5 volumes is worse by orders of magnitude, and worse
// in the direction that matters: the cost grows with the amount of transport, which is the one
// quantity a Monte Carlo exists to increase. A design whose memory scales with steps fails
// exactly when the simulation gets interesting, so it is not the design here.
//
// The primitive is therefore a *reducer*, StepTally: a fixed device array, atomicAdd-ed into,
// with the bin and the weight both supplied as device functors of the step. Its footprint is
// chosen at setup and never moves - a 200-bin LET spectrum is 1.6 kB whether it sees a
// thousand steps or a trillion. This is not a new idea in this codebase; it is exactly what
// `score` and `voxel_score` in host/transport_run.cu already are, and voxel_score already runs
// at ~1e6 cells. StepTally is those arrays with the bin and the weight opened up.
//
// It is also what Geant4 users do, for the same reason. Nobody push_backs every G4Step into a
// vector; they fill a G4AnalysisManager histogram from UserSteppingAction and keep the
// histogram. The GPU changes the arithmetic, not the conclusion.
//
// StepTap - the materialising hook - is kept at the bottom of this file, because a bounded
// dump of real steps is the right tool for debugging and for small runs. It is capped, it
// counts what it drops, and it is not the answer to "how do I customise at step level".
//
// WHAT A HOOK CANNOT DO, and this is a property of the machine rather than a design choice
//
//   * It is __device__ code. It cannot call host functions, allocate, do I/O, or throw. A user
//     who needs those reduces on the device and does the rest on the host afterwards.
//   * It runs in parallel over tracks. Anything shared between tracks needs an atomic, which
//     is why StepTally is written in terms of one. That atomic is atomicAdd on a double, so
//     a tally needs sm_60 or better - compile anything that instantiates one with -arch, as
//     build_engine.bat and the TESTS_GPU list in build_all.bat do.
//   * A custom hook type is a new template instantiation, so it means rebuilding the engine.
//     The stock build carries StepTally and StepTap, both gated at runtime by a null pointer,
//     so the common cases need no rebuild and cost the stock kernels one predicated load.
//
// WHY A RUNTIME GATE RATHER THAN TWO INSTANTIATIONS. run_step_hadron is already templated on
// the species, and ptxas has died twice on this kernel's size in the life of this project
// (docs/RISK.md S12). Doubling the instantiation count to make the inactive case free would
// trade a predictable null check for an unpredictable compiler failure. The cost of the check
// is measured rather than assumed - see tests/test_step_hook.cu.
#pragma once
#include "core/particle.cuh"
#include "core/step_report.cuh"
#include "core/track_buffer.cuh"
#include "core/vec3.cuh"

namespace g4gpu {

/// One end of a step - the device answer to G4StepPoint.
///
/// Returned BY VALUE from DeviceStep::GetPreStepPoint(), where Geant4 returns a G4StepPoint*.
/// A pointer would have to point at something, and there is no per-step object on the device
/// to point at; the values live in the kernel's registers. Building this on demand costs
/// nothing once the hook is inlined - the compiler folds it back into the same registers, and
/// drops whatever the hook does not read.
///
/// Geant4 methods that are NOT here, and why:
///   GetGlobalTime, GetLocalTime, GetProperTime  the transport carries no time at all. Adding
///                                               it is a field on every track, paid on every
///                                               load and store, not an accessor.
///   GetTouchable, GetPhysicalVolume             volumes are indices here, not objects.
///                                               GetVolume() returns the index.
///   GetSensitiveDetector                        GetScoreSlot() on the step is the analogue.
///   GetPolarization                             not tracked.
///   GetWeight                                   there is no variance reduction; it is 1.
template <typename real_t>
struct DeviceStepPoint {
  Vec3<real_t> position;
  Vec3<real_t> direction;
  real_t ekin;
  real_t mass;    ///< MeV, of the particle taking the step
  real_t charge;  ///< in units of the positron charge
  int volume;     ///< volume index, or geom::kOutsideWorld
  int material;   ///< material index, or -1
  real_t safety;  ///< mm; negative when the stepper had no reason to compute one
  StepStatus status;
  ProcessId process;

  /// So that both Geant4 spellings compile. Geant4 hands back a G4StepPoint* and every
  /// ported line says `GetPreStepPoint()->GetKineticEnergy()`; this is returned by value,
  /// because there is no per-step object on the device to point at. Returning `this` from
  /// operator-> keeps the arrow working on the temporary, so ported code needs no edit -
  /// and the dot works too.
  __host__ __device__ const DeviceStepPoint* operator->() const { return this; }

  __host__ __device__ Vec3<real_t> GetPosition() const { return position; }
  __host__ __device__ Vec3<real_t> GetMomentumDirection() const { return direction; }
  __host__ __device__ real_t GetKineticEnergy() const { return ekin; }
  __host__ __device__ real_t GetTotalEnergy() const { return ekin + mass; }
  __host__ __device__ real_t GetMass() const { return mass; }
  __host__ __device__ real_t GetCharge() const { return charge; }
  /// |p|, MeV/c. sqrt(T(T+2m)) - exact for a photon too, where it reduces to T.
  __host__ __device__ real_t GetMomentumMagnitude() const {
    return sqrt(ekin * (ekin + real_t(2) * mass));
  }
  __host__ __device__ Vec3<real_t> GetMomentum() const {
    return GetMomentumMagnitude() * direction;
  }
  /// v/c. Exactly 1 for a massless particle rather than 0/0.
  __host__ __device__ real_t GetBeta() const {
    const real_t e = GetTotalEnergy();
    return (e > real_t(0)) ? GetMomentumMagnitude() / e : real_t(0);
  }
  /// E/m. Geant4 returns DBL_MAX for a massless particle; this returns 0 to say "not a
  /// meaningful quantity here" rather than a number that will silently poison an average.
  __host__ __device__ real_t GetGamma() const {
    return (mass > real_t(0)) ? GetTotalEnergy() / mass : real_t(0);
  }
  __host__ __device__ int GetVolume() const { return volume; }
  __host__ __device__ int GetMaterial() const { return material; }
  __host__ __device__ real_t GetSafety() const { return safety; }
  __host__ __device__ StepStatus GetStepStatus() const { return status; }
  __host__ __device__ ProcessId GetProcessDefinedStep() const { return process; }
};

/// The track taking the step - the device answer to G4Track.
///
/// A HANDLE, not a copy. It holds a pointer to the live TrackState the kernel is stepping, so
/// the setters are real: SetTrackStatus(fStopAndKill) kills the track, SetPolarization stores
/// something the next step will read back, SetKineticEnergy changes what gets transported. That
/// is the same relationship Geant4 has, where GetTrack() hands back a mutable G4Track*.
///
/// A handle also means nothing is duplicated: GetKineticEnergy() reads the track's own field
/// rather than a snapshot that could drift out of step with it.
///
/// Methods deliberately NOT here, each for a reason that is not "no time to do it":
///   GetTouchable, GetNextTouchable, GetOriginTouchable, GetVolume, GetNextVolume,
///   GetMaterial, GetLogicalVolumeAtVertex   Geant4 returns pointers into a hierarchy of host
///                                           objects. Volumes and materials are INDICES here,
///                                           so the index-returning forms are provided instead.
///   GetDynamicParticle, GetStep             those objects do not exist; their contents are on
///                                           this handle and on the step that owns it.
///   CalculateVelocityForOpticalPhoton,      no optical photons.
///   UseGivenVelocity, SetVelocity
///   GetCreatorModelID, GetCreatorModelName  Geant4's model registry has no analogue here.
template <typename real_t>
struct DeviceTrack {
  /// The live track. Writes through this reach the transport.
  TrackState<real_t>* t;
  ParticleType species;
  /// Whether the transport intends to requeue this track, before any SetTrackStatus.
  bool alive_;

  /// So that `GetTrack()->GetTrackID()` - Geant4's spelling - compiles on the returned value.
  __host__ __device__ DeviceTrack* operator->() { return this; }
  __host__ __device__ const DeviceTrack* operator->() const { return this; }

  __host__ __device__ ParticleDef<real_t> Def() const { return particle_def<real_t>(species); }

  // ---- identity and provenance
  __host__ __device__ ParticleType GetDefinition() const { return species; }
  __host__ __device__ ParticleType GetParticleDefinition() const { return species; }
  /// Stable across a run, but a key rather than Geant4's small dense counter.
  __host__ __device__ unsigned int GetTrackID() const { return t->rng_key; }
  /// The parent's key. 0 for a primary, as Geant4 reports 0 for a primary.
  __host__ __device__ unsigned int GetParentID() const {
    static_assert(kFullTrackState, "This G4Track accessor needs the per-track state block, "
                  "which this build turned off with -DG4GPU_FULL_TRACK_STATE=0. It is on by "
                  "default and costs 8% of throughput. See track_buffer.cuh.");
    return t->parent_key;
  }
  __host__ __device__ ProcessId GetCreatorProcess() const {
    static_assert(kFullTrackState, "This G4Track accessor needs the per-track state block, "
                  "which this build turned off with -DG4GPU_FULL_TRACK_STATE=0. It is on by "
                  "default and costs 8% of throughput. See track_buffer.cuh.");
    return t->creator_process;
  }
  __host__ __device__ void SetCreatorProcess(ProcessId p) { t->creator_process = p; }
  __host__ __device__ unsigned int GetCurrentStepNumber() const { return t->step; }

  // ---- kinematics
  __host__ __device__ real_t GetKineticEnergy() const { return t->ekin; }
  __host__ __device__ void SetKineticEnergy(real_t v) { t->ekin = v; }
  __host__ __device__ real_t GetMass() const { return Def().mass; }
  __host__ __device__ real_t GetCharge() const { return Def().charge; }
  __host__ __device__ real_t GetTotalEnergy() const { return t->ekin + GetMass(); }
  __host__ __device__ real_t GetMomentumMagnitude() const {
    const real_t m = GetMass();
    return sqrt(t->ekin * (t->ekin + real_t(2) * m));
  }
  __host__ __device__ Vec3<real_t> GetMomentum() const { return GetMomentumMagnitude() * t->dir; }
  __host__ __device__ Vec3<real_t> GetPosition() const { return t->pos; }
  __host__ __device__ void SetPosition(const Vec3<real_t>& p) { t->pos = p; }
  __host__ __device__ Vec3<real_t> GetMomentumDirection() const { return t->dir; }
  __host__ __device__ void SetMomentumDirection(const Vec3<real_t>& d) { t->dir = d; }
  /// mm/ns. G4Track::GetVelocity - c_light * GetBeta().
  __host__ __device__ real_t GetVelocity() const {
    return units::c_light<real_t>() * dynamic_particle_beta(t->ekin, GetMass());
  }
  __host__ __device__ real_t CalculateVelocity() const { return GetVelocity(); }
  __host__ __device__ real_t GetBeta() const {
    return dynamic_particle_beta(t->ekin, GetMass());
  }

  // ---- clocks and path
  __host__ __device__ real_t GetGlobalTime() const {
    static_assert(kFullTrackState, "This G4Track accessor needs the per-track state block, "
                  "which this build turned off with -DG4GPU_FULL_TRACK_STATE=0. It is on by "
                  "default and costs 8% of throughput. See track_buffer.cuh.");
    return t->global_time;
  }
  __host__ __device__ void SetGlobalTime(real_t v) { t->global_time = v; }
  __host__ __device__ real_t GetLocalTime() const {
    static_assert(kFullTrackState, "This G4Track accessor needs the per-track state block, "
                  "which this build turned off with -DG4GPU_FULL_TRACK_STATE=0. It is on by "
                  "default and costs 8% of throughput. See track_buffer.cuh.");
    return t->local_time;
  }
  __host__ __device__ void SetLocalTime(real_t v) { t->local_time = v; }
  __host__ __device__ real_t GetProperTime() const {
    static_assert(kFullTrackState, "This G4Track accessor needs the per-track state block, "
                  "which this build turned off with -DG4GPU_FULL_TRACK_STATE=0. It is on by "
                  "default and costs 8% of throughput. See track_buffer.cuh.");
    return t->proper_time;
  }
  __host__ __device__ void SetProperTime(real_t v) { t->proper_time = v; }
  __host__ __device__ real_t GetTrackLength() const {
    static_assert(kFullTrackState, "This G4Track accessor needs the per-track state block, "
                  "which this build turned off with -DG4GPU_FULL_TRACK_STATE=0. It is on by "
                  "default and costs 8% of throughput. See track_buffer.cuh.");
    return t->track_length;
  }
  __host__ __device__ void AddTrackLength(real_t v) { t->track_length += v; }

  // ---- vertex
  __host__ __device__ Vec3<real_t> GetVertexPosition() const {
    static_assert(kFullTrackState, "This G4Track accessor needs the per-track state block, "
                  "which this build turned off with -DG4GPU_FULL_TRACK_STATE=0. It is on by "
                  "default and costs 8% of throughput. See track_buffer.cuh.");
    return t->vertex_pos;
  }
  __host__ __device__ void SetVertexPosition(const Vec3<real_t>& p) { t->vertex_pos = p; }
  __host__ __device__ Vec3<real_t> GetVertexMomentumDirection() const {
    static_assert(kFullTrackState, "This G4Track accessor needs the per-track state block, "
                  "which this build turned off with -DG4GPU_FULL_TRACK_STATE=0. It is on by "
                  "default and costs 8% of throughput. See track_buffer.cuh.");
    return t->vertex_dir;
  }
  __host__ __device__ void SetVertexMomentumDirection(const Vec3<real_t>& d) {
    t->vertex_dir = d;
  }
  __host__ __device__ real_t GetVertexKineticEnergy() const {
    static_assert(kFullTrackState, "This G4Track accessor needs the per-track state block, "
                  "which this build turned off with -DG4GPU_FULL_TRACK_STATE=0. It is on by "
                  "default and costs 8% of throughput. See track_buffer.cuh.");
    return t->vertex_ekin;
  }
  __host__ __device__ void SetVertexKineticEnergy(real_t v) { t->vertex_ekin = v; }
  /// The index of the volume the track was created in - the index form of
  /// GetLogicalVolumeAtVertex().
  __host__ __device__ int GetVolumeAtVertex() const {
    static_assert(kFullTrackState, "This G4Track accessor needs the per-track state block, "
                  "which this build turned off with -DG4GPU_FULL_TRACK_STATE=0. It is on by "
                  "default and costs 8% of throughput. See track_buffer.cuh.");
    return t->vertex_volume;
  }
  /// The index of the volume the track is in now.
  __host__ __device__ int GetVolume() const { return t->volume; }

  // ---- status and flags
  __host__ __device__ TrackStatus GetTrackStatus() const { return t->status; }
  /// Kill a track, suspend it, or let it live. The kernel reads this back the instant the hook
  /// returns; see G4VUserDeviceSteppingAction for which values this transport can honour and
  /// what happens to the two it cannot.
  __host__ __device__ void SetTrackStatus(TrackStatus st) { t->status = st; }
  __host__ __device__ bool IsAlive() const { return alive_; }
  __host__ __device__ bool IsBelowThreshold() const { return (t->flags & kBelowThreshold) != 0u; }
  __host__ __device__ void SetBelowThresholdFlag(bool v = true) {
    t->flags = v ? (t->flags | kBelowThreshold) : (t->flags & ~kBelowThreshold);
  }
  __host__ __device__ bool IsGoodForTracking() const {
    return (t->flags & kGoodForTracking) != 0u;
  }
  __host__ __device__ void SetGoodForTrackingFlag(bool v = true) {
    t->flags = v ? (t->flags | kGoodForTracking) : (t->flags & ~kGoodForTracking);
  }

  // ---- weight, polarization, user data
  __host__ __device__ real_t GetWeight() const {
    static_assert(kFullTrackState, "This G4Track accessor needs the per-track state block, "
                  "which this build turned off with -DG4GPU_FULL_TRACK_STATE=0. It is on by "
                  "default and costs 8% of throughput. See track_buffer.cuh.");
    return t->weight;
  }
  __host__ __device__ void SetWeight(real_t w) { t->weight = w; }
  __host__ __device__ Vec3<real_t> GetPolarization() const {
    static_assert(kFullTrackState, "This G4Track accessor needs the per-track state block, "
                  "which this build turned off with -DG4GPU_FULL_TRACK_STATE=0. It is on by "
                  "default and costs 8% of throughput. See track_buffer.cuh.");
    return t->polarization;
  }
  __host__ __device__ void SetPolarization(const Vec3<real_t>& p) { t->polarization = p; }
  /// The device answer to G4VUserTrackInformation - see TrackState::user_data. A pointer to a
  /// host object cannot exist here; a POD slot that survives from one step to the next can.
  __host__ __device__ unsigned int GetUserInformation() const {
    static_assert(kFullTrackState, "This G4Track accessor needs the per-track state block, "
                  "which this build turned off with -DG4GPU_FULL_TRACK_STATE=0. It is on by "
                  "default and costs 8% of throughput. See track_buffer.cuh.");
    return t->user_data;
  }
  __host__ __device__ void SetUserInformation(unsigned int v) { t->user_data = v; }
};

/// The secondaries one step created - the device answer to
/// G4Step::GetSecondaryInCurrentStep(), which returns a vector of `const G4Track*`.
///
/// Each element is a real track read out of the species buffer it was appended to, not a copy
/// made for the occasion, so what you get is the same object the transport will step next
/// iteration. Reading one costs a scattered load; a hook that only wants the count uses
/// GetNumberOfSecondariesInCurrentStep() and touches none of this.
///
/// ORDER IS NEWEST FIRST. The chain is built by pointing each secondary at the previous one,
/// which is what makes a step's list unbounded - no array, no capacity, no number chosen in
/// advance about how many secondaries a process may make. Walking it therefore yields the last
/// secondary first, where Geant4's vector is in creation order. That is a real difference and
/// it is named here rather than hidden behind a reversal that would need somewhere to put the
/// reversed copy - which would be an array with a capacity again.
template <typename real_t>
struct SecondaryList {
  SecondaryArena arena;
  /// The one pool every track lives in. A secondary's species comes from the track itself now,
  /// rather than from which of three buffers it happened to land in.
  const TrackBuffer<real_t>* pool;
  int last;   ///< head of the chain; -1 when the step made nothing
  int count;  ///< how many the step made, whether or not they all fit the arena

  /// How many are actually reachable through the chain. Equal to size() unless the arena ran
  /// out, in which case the arena's own overflow counter has already recorded it.
  __host__ __device__ int size() const { return count; }
  __host__ __device__ bool empty() const { return last < 0; }

  /// An iterator over the chain. `*it` is a DeviceTrack handle onto the real secondary.
  struct Iter {
    const SecondaryList* list;
    int at;
    TrackState<real_t> loaded;

    __host__ __device__ bool valid() const { return at >= 0; }
    __host__ __device__ void advance() {
      at = (at >= 0) ? list->arena.prev[at] : -1;
    }
    /// Loads the secondary this iterator is on. The TrackState lives in the iterator, so the
    /// handle stays valid exactly as long as the iterator does.
    __host__ __device__ DeviceTrack<real_t> get() {
      const unsigned int e = list->arena.entry[at];
      list->pool->load(SecondaryArena::slot_of(e), loaded);
      return DeviceTrack<real_t>{&loaded, loaded.species, true};
    }
  };

  __host__ __device__ Iter begin() const { return Iter{this, last, TrackState<real_t>{}}; }
};

/// One real step of one track - the device answer to G4Step.
///
/// A view, not a record: every field is already in registers when the hook is called, and
/// nothing here is stored unless a hook chooses to store it. Every value is that step's own,
/// not an event total. `volume_pre != volume_post` is a boundary crossing; `!alive` is the
/// step the track died on.
///
/// Two spellings of everything, on purpose. The flat fields are what the reducers in this file
/// read and are the shortest thing to write in a hook. The Get* methods are Geant4's names, so
/// that a stepping action ported from a Geant4 project reads the way it did there. They are the
/// same values; nothing is stored twice.
///
/// The one structural difference from G4Step: GetPreStepPoint() and GetTrack() return objects
/// BY VALUE, not pointers. See DeviceStepPoint for why.
template <typename real_t>
struct DeviceStep {
  ParticleType species;
  real_t ekin_pre;   ///< MeV, before the step
  real_t ekin_post;  ///< MeV, after
  real_t edep;       ///< MeV deposited by this step in this volume
  real_t length;     ///< mm actually travelled - the TRUE path, not |pos_post - pos_pre|
  Vec3<real_t> pos_pre;
  Vec3<real_t> pos_post;
  Vec3<real_t> dir_pre;
  Vec3<real_t> dir_post;
  int volume_pre;
  int volume_post;
  int score_slot;           ///< scoring index of the pre-step volume, or -1; indexes `score`
  int event;                ///< index within the batch, matching the score array
  bool alive;               ///< false if the track ended on this step

  /// The live track this step belongs to. GetTrack() is a handle onto it, so a stepping
  /// action's setters reach the transport rather than a copy. Null only in a synthetic step
  /// built by a test.
  TrackState<real_t>* track_ptr;
  /// G4Step::IsFirstStepInVolume - true if the PREVIOUS step of this track ended on a
  /// boundary. Captured before the kernel updates the flag for the next step.
  bool first_in_volume;

  int material;            ///< pre-step material index, or -1
  real_t safety;           ///< pre-step isotropic safety, mm; negative when not computed
  real_t non_ionizing;     ///< MeV of `edep` that went into nuclear recoils
  int n_secondaries;       ///< tracks this step handed to the emitter
  /// Head of this step's secondary chain, and where the chain lives. See SecondaryList.
  SecondaryArena sec_arena;
  const TrackBuffer<real_t>* sec_pool;
  int sec_last;
  StepStatus status;       ///< how the step ended
  ProcessId process;       ///< what ended it

  // ---------------------------------------------------------------- short spellings

  /// Unrestricted linear energy transfer of this step, MeV/mm. Zero-length steps give 0.
  __host__ __device__ real_t let() const { return (length > real_t(0)) ? edep / length : real_t(0); }
  /// True if this step crossed into volume `v` from somewhere else.
  __host__ __device__ bool entered(int v) const { return volume_post == v && volume_pre != v; }
  /// True if this step left volume `v`.
  __host__ __device__ bool left(int v) const { return volume_pre == v && volume_post != v; }

  // ---------------------------------------------------------------- G4Step spellings

  __host__ __device__ real_t GetTotalEnergyDeposit() const { return edep; }
  __host__ __device__ real_t GetNonIonizingEnergyDeposit() const { return non_ionizing; }
  /// The TRUE path length, as G4Step::GetStepLength returns. Not the displacement: multiple
  /// scattering deflects inside a step, so the two differ and only one of them is the distance
  /// the particle actually ran.
  __host__ __device__ real_t GetStepLength() const { return length; }
  __host__ __device__ Vec3<real_t> GetDeltaPosition() const { return pos_post - pos_pre; }
  __host__ __device__ real_t GetDeltaEnergy() const { return ekin_post - ekin_pre; }
  /// G4Step::GetDeltaTime - ns this step advanced the clock by.
  ///
  /// Derived, not stored. It is the true path length over the PRE-step velocity, which is
  /// exactly what TrackState::advance() adds to the global and local clocks, so no field has
  /// to ride the track for a hook to ask. The two must not drift apart, which is why this
  /// calls dynamic_particle_beta rather than DeviceStepPoint::GetBeta: the same beta by the
  /// same route, not the same beta by algebra that happens to agree to 1e-16.
  __host__ __device__ real_t GetDeltaTime() const {
    const real_t v =
        units::c_light<real_t>() * dynamic_particle_beta(ekin_pre, GetPreStepPoint().GetMass());
    return (v > real_t(0)) ? length / v : real_t(0);
  }
  /// How many secondaries this step created. Always exact, whatever the arena did.
  __host__ __device__ int GetNumberOfSecondariesInCurrentStep() const { return n_secondaries; }

  /// The secondaries themselves, as real tracks. See SecondaryList - in particular that the
  /// order is newest first, and why.
  __host__ __device__ SecondaryList<real_t> GetSecondaryInCurrentStep() const {
    SecondaryList<real_t> l;
    l.arena = sec_arena;
    l.pool = sec_pool;
    l.last = sec_last;
    l.count = n_secondaries;
    return l;
  }
  __host__ __device__ bool IsLastStepInVolume() const {
    return status == StepStatus::fGeomBoundary || status == StepStatus::fWorldBoundary;
  }
  __host__ __device__ bool IsFirstStepInVolume() const { return first_in_volume; }
  /// Which scorer the pre-step volume belongs to, or -1. The analogue of
  /// G4StepPoint::GetSensitiveDetector().
  __host__ __device__ int GetScoreSlot() const { return score_slot; }
  /// Index of the event within the batch, matching the scorer's array.
  __host__ __device__ int GetEventID() const { return event; }

  /// Identifies the track. Read from the track itself rather than copied onto the step: two
  /// fields that must agree are two fields that can disagree, and this pair already did once -
  /// a synthetic step in a test set one and not the other, and GetTrackID() answered with the
  /// stale one. One source of truth removes the whole class of that mistake.
  __host__ __device__ unsigned int GetTrackID() const { return track_ptr->rng_key; }
  /// G4Track::GetCurrentStepNumber - 1 for a track's first step, as in Geant4.
  __host__ __device__ unsigned int GetCurrentStepNumber() const { return track_ptr->step; }

  __host__ __device__ DeviceStepPoint<real_t> GetPreStepPoint() const {
    const ParticleDef<real_t> pd = particle_def<real_t>(species);
    DeviceStepPoint<real_t> q;
    q.position = pos_pre;
    q.direction = dir_pre;
    q.ekin = ekin_pre;
    q.mass = pd.mass;
    q.charge = pd.charge;
    q.volume = volume_pre;
    q.material = material;
    q.safety = safety;
    // Geant4 puts the status of the step on its POST point; the pre point carries the status
    // the previous step ended with, which is not available here. fUndefined says so.
    q.status = StepStatus::fUndefined;
    q.process = ProcessId::fNotDefined;
    return q;
  }

  __host__ __device__ DeviceStepPoint<real_t> GetPostStepPoint() const {
    const ParticleDef<real_t> pd = particle_def<real_t>(species);
    DeviceStepPoint<real_t> q;
    q.position = pos_post;
    q.direction = dir_post;
    q.ekin = ekin_post;
    q.mass = pd.mass;
    q.charge = pd.charge;
    q.volume = volume_post;
    // The material of the volume the step ENDED in is not looked up by the stepper - it would
    // be an extra navigation query on every step for the benefit of hooks that may not want
    // it. What is reported is the material the step happened in, on the pre point.
    q.material = -1;
    q.safety = real_t(-1);
    q.status = status;
    q.process = process;
    return q;
  }

  /// A handle onto the live track, so the setters on it are real. Const on this step does not
  /// propagate through the pointer, deliberately: Geant4's UserSteppingAction is handed a
  /// `const G4Step*` and can still call `step->GetTrack()->SetTrackStatus(fStopAndKill)`, and
  /// that line is the single most common reason to write a stepping action at all.
  __host__ __device__ DeviceTrack<real_t> GetTrack() const {
    return DeviceTrack<real_t>{track_ptr, species, alive};
  }
};

/// Whether the transport should requeue a track, after a stepping action has had its say.
///
/// `proposed` is what the physics decided. `asked` is what the hook set through
/// G4Track::SetTrackStatus - fAlive unless it touched it. The two statuses that cannot be
/// honoured here are reported by the caller rather than resolved silently; see
/// kUnsupportedTrackStatus.
template <typename real_t>
__host__ __device__ inline bool resolve_track_status(bool proposed, TrackStatus asked) {
  switch (asked) {
    case TrackStatus::fAlive:
    case TrackStatus::fStopButAlive:
      // fStopButAlive means "stop moving but stay in the world" - Geant4 keeps the track for
      // at-rest processes. There are none here, so a track that stops is a track that is done;
      // whatever the physics decided stands.
      return proposed;
    case TrackStatus::fStopAndKill:
    case TrackStatus::fKillTrackAndSecondaries:
      // The secondaries of this step have already been appended to their buffers by the time a
      // hook runs, so fKillTrackAndSecondaries kills the track and NOT its secondaries. That is
      // a real difference from Geant4 and it is counted, not glossed - see the counter below.
      return false;
    case TrackStatus::fSuspend:
    case TrackStatus::fPostponeToNextEvent:
      // Both need somewhere to put a track aside, and there is nowhere: every track in flight
      // is in a species buffer being stepped in lockstep. Killing is the least wrong thing to
      // do with it, and the count makes sure nobody finds out by noticing missing dose.
      return false;
  }
  return proposed;
}

/// True for a status this transport cannot do what Geant4 would do with.
__host__ __device__ inline bool is_unsupported_track_status(TrackStatus asked) {
  return asked == TrackStatus::fSuspend || asked == TrackStatus::fPostponeToNextEvent
         || asked == TrackStatus::fKillTrackAndSecondaries;
}

/// The default: costs nothing and generates nothing.
struct NoStepHook {
  template <typename real_t>
  __device__ void operator()(const DeviceStep<real_t>&) const {}
};

// ---------------------------------------------------------------- bounded reduction

/// A fixed-size device histogram filled from steps. The general step-level scorer.
///
/// `Bin` maps a step to an index in [0, nbins), or to any negative value to drop it. `Weight`
/// maps a step to the quantity added there. Both are device functors, so the pair is what the
/// tally *means* - and no question below needed a special case in this file:
///
///   LET spectrum of the dose in a volume
///     Bin    = bin s.let() with LogBins{1e-3, 1e3, 200}, gated on s.score_slot == slot
///     Weight = s.edep                                  -> dose per LET decade
///
///   Dose-averaged quality factor per event (the case this file was written for)
///     two tallies, both Bin = BinByEvent{}
///     Weight = Q(s.let()) * s.edep, and Weight = s.edep
///     Qbar[event] = weighted[event] / plain[event]
///
///   Fluence spectrum entering a detector
///     Bin    = bin s.ekin_pre with LogBins{...}, gated on s.entered(kDetector)
///     Weight = 1
///
///   Dose from one species only
///     Bin = BinByScoreSlot{}, Weight = (s.species == kAlpha) ? s.edep : 0
///
/// Memory is `nbins * 8` bytes, fixed at setup and independent of the number of steps, tracks,
/// events or volumes in the problem. That is the whole point: tallying a 1e12-step run costs
/// exactly what tallying a 1e3-step run costs.
///
/// `bins == nullptr` disables it.
template <typename real_t, typename Bin, typename Weight>
struct StepTally {
  double* bins = nullptr;
  int nbins = 0;
  Bin bin{};
  Weight weight{};

  __device__ void operator()(const DeviceStep<real_t>& s) const {
    if (bins == nullptr) { return; }
    const int b = bin(s);
    if (b < 0 || b >= nbins) { return; }
    const real_t w = weight(s);
    // Skipping zero is worth a branch: in most tallies most steps contribute nothing, and an
    // atomicAdd of 0.0 costs a serialised round trip to make no difference. It is also exact -
    // adding zero cannot change a sum - so this is speed with no effect on the answer.
    if (w == real_t(0)) { return; }
    atomicAdd(&bins[b], static_cast<double>(w));
  }
};

/// Bin by the event's index within the batch. One number per event, in the same layout and the
/// same footprint as the existing per-event `score` array.
struct BinByEvent {
  template <typename real_t>
  __device__ int operator()(const DeviceStep<real_t>& s) const {
    return s.event;
  }
};

/// Bin by the scoring index of the volume the step started in. One double per scored volume -
/// 800 kB even for a geometry with 1e5 of them.
struct BinByScoreSlot {
  template <typename real_t>
  __device__ int operator()(const DeviceStep<real_t>& s) const {
    return s.score_slot;
  }
};

/// Uniform bins over [lo, hi). Out-of-range values are dropped rather than piled into the end
/// bins, so an overflow shows up as a missing integral instead of a spurious spike at an edge.
struct LinBins {
  double lo = 0, hi = 1;
  int n = 100;
  __host__ __device__ int operator()(double x) const {
    if (!(x >= lo) || x >= hi) { return -1; }  // the !(x >= lo) form also rejects NaN
    return static_cast<int>((x - lo) / (hi - lo) * n);
  }
};

/// Logarithmic bins over [lo, hi), the natural axis for an energy or a LET. Both bounds must
/// be positive; non-positive and out-of-range values are dropped.
struct LogBins {
  double lo = 1e-3, hi = 1e3;
  int n = 100;
  __host__ __device__ int operator()(double x) const {
    if (!(x > 0) || x < lo || x >= hi) { return -1; }
    return static_cast<int>(log(x / lo) / log(hi / lo) * n);
  }
};

// ---------------------------------------------------------------- composition

/// Run two hooks on every step. Nests, so `StepHooks<A, StepHooks<B, C>>` runs three.
template <typename A, typename B>
struct StepHooks {
  A a{};
  B b{};
  template <typename real_t>
  __device__ void operator()(const DeviceStep<real_t>& s) const {
    a(s);
    b(s);
  }
};

// ---------------------------------------------------------------- materialisation

/// A record written by StepTap. Flat and trivially copyable so the readback is one memcpy.
///
/// Not everything a DeviceStep carries - a record is paid for per step, and this is the one
/// hook whose cost grows with the run, so what goes in it is chosen rather than copied
/// wholesale. Momentum direction and safety are left out because a host-side analysis that
/// wants them is usually better written as a device reducer anyway.
template <typename real_t>
struct StepRecord {
  real_t ekin_pre, ekin_post, edep, length;
  real_t x, y, z;
  int score_slot, volume_pre, volume_post, event;
  unsigned int track_key, step_index;
  int species;
  int alive;
  int material;
  int n_secondaries;
  int status;   ///< StepStatus
  int process;  ///< ProcessId
};

/// Copies whole steps out to be read on the host.
///
/// THIS ONE SCALES WITH THE SIMULATION AND THE OTHERS DO NOT. A record is 104 bytes in the
/// default double build - sizeof(StepRecord<double>), 76 in a float one - so a capacity of
/// 1e6 costs 104 MB and holds about 4% of the steps in a 2M-event B1 run. It is the right
/// tool for a debugging run of a few thousand events, for checking a
/// tally's arithmetic against real numbers, and for exploratory work where you do not yet know
/// what to reduce. It is the wrong tool for production, where StepTally costs the same whether
/// the run is small or enormous.
///
/// What it does guarantee is that it never lies. Steps beyond `capacity` are counted in
/// `overflow` rather than dropped silently, so a truncated dump is visibly truncated - and a
/// nonzero overflow means the sample is biased toward whatever ran first, not merely
/// incomplete, which is the more dangerous of the two failures.
///
/// `records == nullptr` disables it.
template <typename real_t>
struct StepTap {
  StepRecord<real_t>* records = nullptr;
  int* count = nullptr;     ///< device-resident write cursor
  int* overflow = nullptr;  ///< steps that did not fit; never silent
  int capacity = 0;
  /// Which scoring slot to record. -1 takes every volume, which is only sane for a small run.
  int score_slot = -1;
  /// Skip steps that deposited nothing. A boundary crossing with no deposit is still a step,
  /// so this is off by default - a fluence or a spectrum at entry needs exactly those.
  bool only_deposits = false;

  __device__ void operator()(const DeviceStep<real_t>& s) const {
    if (records == nullptr) { return; }
    if (score_slot >= 0 && s.score_slot != score_slot) { return; }
    if (only_deposits && s.edep == real_t(0)) { return; }
    const int i = atomicAdd(count, 1);
    if (i >= capacity) {
      atomicAdd(overflow, 1);
      return;
    }
    StepRecord<real_t>& r = records[i];
    r.ekin_pre = s.ekin_pre;
    r.ekin_post = s.ekin_post;
    r.edep = s.edep;
    r.length = s.length;
    r.x = s.pos_post.x;
    r.y = s.pos_post.y;
    r.z = s.pos_post.z;
    r.score_slot = s.score_slot;
    r.volume_pre = s.volume_pre;
    r.volume_post = s.volume_post;
    r.event = s.event;
    r.track_key = s.GetTrackID();
    r.step_index = s.GetCurrentStepNumber();
    r.species = static_cast<int>(s.species);
    r.alive = s.alive ? 1 : 0;
    r.material = s.material;
    r.n_secondaries = s.n_secondaries;
    r.status = static_cast<int>(s.status);
    r.process = static_cast<int>(s.process);
  }
};

}  // namespace g4gpu
