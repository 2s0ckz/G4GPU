// P15's interaction queue: how an inelastic final state is applied WITHOUT putting the models
// inside the stepping kernels.
//
// ---------------------------------------------------------------------------------------------
// THE DESIGN QUESTION, AND THE MEASUREMENT THAT SETTLED IT
//
// The four entry points a wired inelastic process has to reach are the largest functions in this
// project, and their own probes say so (docs/PORTED.md 2.1.10-2.1.13):
//
//     bert::apply_yourself   255 registers, 11,872-byte frame   (10,160 of it P6's de-excitation)
//     bic::apply_yourself    255 registers, about 10 kB
//     ftf::entry::apply      255 registers, 1,376-byte frame  + 458,096 B of workspace IN FLIGHT
//     stopping::at_rest      255 registers, 13,456-byte frame
//
// against stepping kernels whose frames, measured on this branch's baseline build with
// `-Xptxas -v`, are 3,936 B (`run_step_hadron<kProton>`), 4,592 B (`<kPionMinus>`) and 2,352 B
// (`run_step_neutral<kNeutron>`) - all at 255 registers already, all spilling. `Upload` fails a
// kernel whose frame passes 16,384 B.
//
// TWO ARGUMENTS DECIDED IT, and the second is the one that is not about compile times.
//
//  1. **The frame.** A stepping kernel's stack frame is the maximum over its whole call tree,
//     so calling the four entry points from inside `step_hadron` puts a ~13 kB frame on top of
//     a ~4 kB one in every one of the thirteen `run_step_hadron` instantiations, `__noinline__`
//     or not. docs/RISK.md V55 is the entry recording that inlining the far smaller ELASTIC
//     package into `run_step_hadron` killed ptxas with an access violation, and V63/V65 are the
//     entries recording that this file's translation units are one per kernel precisely because
//     ptxas falls over on less than this. The measurement is in docs/RISK.md V188.
//
//  2. **The workspace, and this one is arithmetic rather than a compiler's opinion.** FTFP
//     needs 458,096 bytes per thread IN FLIGHT and the whole set of arms needs 1.41 MB
//     (`tests/test_inelastic_transport.cu` prints the table from the types themselves). A
//     stepping kernel runs over a 65,536-track batch, so an inline model would need either
//     65,536 slots - 92 GB, past every card - or a slot pool indexed by the THREAD, with the
//     threads past the last slot refused. And that refusal is the difference:
//
//         inline, slot = tid      a shortage of slots costs INTERACTIONS: the tracks past slot
//                                 N never react, and the hole is proportional to the batch.
//         queued, drained in      a shortage of slots costs TIME: the queue is drained in
//         chunks of n_slots       chunks and every entry is processed.
//
//     A capacity whose shortage costs physics has to be sized for the worst case of a
//     distribution nobody controls; a capacity whose shortage costs latency can be sized by
//     what fits. That is the whole of it.
//
// ---------------------------------------------------------------------------------------------
// WHAT THE STEPPER DOES, AND WHERE THE STEP ENDS
//
// The stepper does the whole step except the model call. It draws the inelastic interaction
// length as a fourth (charged) or existing (neutron) discrete competitor, and when that
// competitor wins it: applies the continuous loss and the multiple scattering, moves the track
// to the interaction point, emits whatever the along-step part of the step emitted (a delta
// ray), fills in the step report with `fHadronInelastic`, and then **enqueues the track instead
// of killing it**. It does NOT append the track to the output pool - the queue entry IS the
// track - and it does NOT call the step hook. Both are the interaction kernel's, so that one
// step produces one hook call with the right secondary count on it.
//
// ---------------------------------------------------------------------------------------------
// THE QUEUE CANNOT OVERFLOW, AND THAT IS A PROOF RATHER THAN A SIZING
//
// One track can queue at most one interaction per launch - the enqueue is on the one PostStep
// branch that won - and a launch steps at most `n_step <= n_live <= pool` tracks. So a queue of
// `pool` entries is a bound and not an estimate, which is the same argument `SecondaryArena`'s
// own comment makes for sizing the arena at the pool. It costs `sizeof(PendingInteraction)` a
// slot, which the engine prints.
//
// `kInteractionQueueFull` therefore exists as a TRIPWIRE, not as an expected count - the shape
// `HadronicRefusal::kCaptureOverflow` already has. If it ever moves, either the bound above is
// wrong or a caller set a smaller capacity by hand, and the run says so rather than losing an
// interaction quietly. A queued track that cannot be queued is killed with its kinetic energy
// deposited locally, which is the conservative disposal `kNeutronInelastic` has used since P8d
// and is NOT what Geant4 does with it.
#pragma once

#include "core/particle.cuh"
#include "core/step_report.cuh"
#include "core/track_buffer.cuh"
#include "physics/hadronic/inelastic_wiring.cuh"

namespace g4gpu::had {

/// Which arm of the interaction kernel a queue entry asks for.
enum class InteractionKind : int {
  /// An in-flight `*Inelastic` process: `G4HadronicProcess::PostStepDoIt`.
  kInelastic = 0,
  /// A stopped negative hadron's at-rest capture: `G4HadronStoppingProcess::AtRestDoIt`.
  /// Queued from the DYING branch of `step_hadron`, where `G4Decay`'s at-rest competitor would
  /// otherwise have run - `AtRestGetPhysicalInteractionLength` returns 0.0 and pre-empts it.
  kAtRest = 1,
};

/// WHICH KERNEL a queue entry is run by, and therefore which translation unit carries the model
/// it needs.
///
/// ONE KERNEL PER MODEL, AND IT IS FORCED. `transport_run_interaction.cu` began as ONE kernel
/// with all four in-flight arms behind a switch, and ptxas went past **22 GB of working set in
/// 200 seconds and was still climbing** when it was stopped. Compiled one arm at a time into a
/// kernel of its own, the same code takes:
///
///     FTFP             8,377 MB, 130 s        Binary cascade   9,838 MB, 365 s
///     Bertini          9,319 MB, 210 s        light ion        (see docs/RISK.md V189)
///
/// so the models ARE compilable as device code - which nothing in this project had ever tried,
/// because every one of their tests is host-only - and what is not compilable is four of them
/// in one module. That is docs/RISK.md V65's rule ("one translation unit per kernel") arriving
/// one level down, and it buys the thing V65 bought as well: a warp whose threads take
/// different physics paths serialises through all of them, so binning the queue by model is
/// better for throughput than branching inside one kernel would have been.
///
/// The bucket is decided by the STEPPER, at enqueue, which is why `PendingInteraction` carries
/// it: for an in-flight interaction it is `choose_inelastic_model`, which needs one uniform and
/// no cross-section table; for an at-rest capture it is a pure function of the PDG code.
enum class InteractionBucket : int {
  kFtfp = 0,
  kBertini,
  kBinary,
  kLightIon,
  /// `G4HadronStoppingProcess::AtRestDoIt`, all three arms.
  ///
  /// ONE BUCKET AND NOT TWO, AND THE REASON IS MEASURED. Splitting the at-rest entries into a
  /// Bertini arm and a Fritiof arm would put Bertini in one kernel and FTFP in the other, which
  /// is what the in-flight split buys - except that `stopping::at_rest` is ONE function that
  /// calls Bertini unconditionally and FTFP through an invoke, so instantiating it instantiates
  /// Bertini whichever arm a kernel is built for. Two kernels would each be Bertini + FTFP and
  /// cost twice what one does. Measured: the at-rest kernel is **19,770 MB of ptxas and 660
  /// seconds**, against 9,319 MB for Bertini alone and 8,377 MB for FTFP alone. docs/RISK.md
  /// V189.
  kAtRest,
  /// The species has a process and no model this port can run. Booked by name without a
  /// kernel launch, which is what keeps a refusal from costing a 1.6 MB slot.
  kNone,
  kNumInteractionBuckets,
};

__host__ __device__ inline const char* interaction_bucket_name(InteractionBucket b) {
  switch (b) {
    case InteractionBucket::kFtfp:           return "FTFP";
    case InteractionBucket::kBertini:        return "Bertini";
    case InteractionBucket::kBinary:         return "BinaryCascade";
    case InteractionBucket::kLightIon:       return "BinaryLightIonReaction";
    case InteractionBucket::kAtRest:         return "at rest (G4HadronStoppingProcess)";
    case InteractionBucket::kNone:           return "(no model)";
    case InteractionBucket::kNumInteractionBuckets: break;
  }
  return "unknown";
}

__host__ __device__ inline InteractionBucket bucket_of_model(InelasticModel m) {
  switch (m) {
    case InelasticModel::kFtfp:     return InteractionBucket::kFtfp;
    case InelasticModel::kBertini:  return InteractionBucket::kBertini;
    case InelasticModel::kBinary:   return InteractionBucket::kBinary;
    case InelasticModel::kLightIon: return InteractionBucket::kLightIon;
    case InelasticModel::kNone:     break;
  }
  return InteractionBucket::kNone;
}

/// One pending interaction: the track, the step that produced it, and the choice already made.
///
/// IT CARRIES THE TRACK BY VALUE AND THE STEPPER DOES NOT APPEND IT. The alternative - append
/// the primary to the output pool and queue its slot index - would need the interaction kernel
/// to be able to KILL a track already in the pool, and `TrackBuffer` has no tombstone. Copying
/// 248 bytes once per interaction is the cheaper of the two, and it keeps `core/track_buffer.cuh`
/// (P1's) untouched.
///
/// THE PRE-STEP SNAPSHOT IS HERE BECAUSE THE HOOK IS. `run_interaction` calls the step hook for
/// this step, so it needs what `run_step_hadron` would have put in `DeviceStep`: the pre-step
/// energy, position, direction and volume, the true length, the safety, the material and the
/// score slot. The VOXEL cell is not among them and is recomputed in `run_interaction` instead -
/// three integer divisions off `(volume_pre, pos_pre)`, against four more bytes on every entry.
/// Eleven scalars against one hook call per step, which is the property a stepping action
/// depends on - `GetNumberOfSecondariesInCurrentStep()` on a step whose secondaries are made in
/// a later kernel is otherwise zero.
template <typename real_t>
struct PendingInteraction {
  TrackState<real_t> track{};   ///< at the interaction point, after the continuous loss

  // ---- what the step hook needs, captured before the step ran
  real_t ekin_pre = 0;
  Vec3<real_t> pos_pre{real_t(0), real_t(0), real_t(0)};
  Vec3<real_t> dir_pre{real_t(0), real_t(0), real_t(1)};
  real_t edep = 0;           ///< deposited by the step BEFORE the interaction
  real_t true_length = 0;
  real_t safety = real_t(-1);
  real_t non_ionizing = 0;
  int volume_pre = 0;
  int material = -1;
  int score_slot = -1;
  /// Secondaries this step has already emitted (a delta ray), so the interaction kernel's
  /// emitter continues the child index rather than restarting it. Restarting would give two
  /// secondaries of one step the same `child_rng_key` and therefore the same random stream.
  unsigned int child_count = 0;
  int sec_last = -1;         ///< the arena chain this step has built so far
  bool first_in_volume = false;
  StepStatus status = StepStatus::fUndefined;

  // ---- the interaction itself
  ParticleType species = ParticleType::kNumTypes;
  InteractionKind kind = InteractionKind::kInelastic;
  /// Which kernel runs this entry, and therefore which model's code it needs. Decided by the
  /// STEPPER, because the decision costs one uniform and no table.
  InteractionBucket bucket = InteractionBucket::kNone;
  /// The model that bucket corresponds to, so the interaction kernel does not re-derive it.
  InelasticModel model = InelasticModel::kNone;
  /// The macroscopic cross section the interaction length was drawn with, for
  /// `G4HadronicProcess`'s integral-approach rejection. Not used on the at-rest arm.
  real_t xs_at_step_start = 0;
};

/// The device queue: an array, an atomic cursor and an overflow count.
///
/// A null `items` disables it, and `push` then reports overflow - which is the state a run whose
/// interaction pool could not be allocated is in, and it behaves as the refusal it is rather
/// than as silence.
template <typename real_t>
struct InteractionQueue {
  PendingInteraction<real_t>* items = nullptr;
  int* cursor = nullptr;   ///< one int, reset per launch by the engine
  int capacity = 0;

  /// @return the slot taken, or -1 when the queue is full (or absent). The caller must then
  ///         refuse by name; see the file header for the disposal.
  ///
  /// `__host__ __device__` for the reason `book_refusal` gives: `step_hadron` and `step_neutral`
  /// are run on the HOST and compared against the device bit for bit
  /// (`tests/test_step_hadron.cu`, `tests/test_neutron_general.cu`), and a device-only push
  /// would have made the enqueue the one branch those comparisons could not reach. A host
  /// caller is single-threaded, so the host arm is a plain increment and not a serialised
  /// atomic.
  __host__ __device__ int push(const PendingInteraction<real_t>& q) const {
    if (items == nullptr || cursor == nullptr || capacity <= 0) { return -1; }
#ifdef __CUDA_ARCH__
    const int i = atomicAdd(cursor, 1);
#else
    const int i = (*cursor)++;
#endif
    if (i >= capacity) { return -1; }
    items[i] = q;
    return i;
  }
};

/// Builds a queue entry out of a step and pushes it.
///
/// `__noinline__`, AND THAT IS NOT AN OPTIMISATION. Written inline in `step_hadron`, the two
/// enqueue blocks put a 384-byte `PendingInteraction` and its eighteen assignments into the
/// kernel body twice, and `nvcc error : 'ptxas' died with status 0xC0000005
/// (ACCESS_VIOLATION)` on `tests/test_step_hadron.cu` - measured, at a 3.7 GB working set after
/// 120 seconds. That is docs/RISK.md **V55** exactly (inlining the elastic package into
/// `run_step_hadron` killed ptxas the same way, and four functions are `__noinline__` there
/// because of it), **V63** and **V65** for a fifth time, and the fix is the same one: give the
/// compiler a call boundary to put the frame behind.
///
/// @return true when the entry was queued. False means the queue was full or absent, and the
///         caller must refuse by name - `has_at_rest_arm`'s file header says what with.
template <typename real_t>
__host__ __device__ __noinline__ bool enqueue_interaction(
    const InteractionQueue<real_t>& queue, const TrackState<real_t>& track,
    InteractionKind kind, ParticleType species, const StepReport<real_t>& rep, real_t edep,
    real_t ekin_pre, const Vec3<real_t>& pos_pre, const Vec3<real_t>& dir_pre, int volume_pre,
    int score_slot, unsigned int child_count, int sec_last, StepStatus status,
    real_t xs_at_step_start, InteractionBucket bucket, InelasticModel model) {
  PendingInteraction<real_t> q;
  q.bucket = bucket;
  q.model = model;
  q.track = track;
  q.ekin_pre = ekin_pre;
  q.pos_pre = pos_pre;
  q.dir_pre = dir_pre;
  q.edep = edep;
  q.true_length = rep.true_length;
  q.safety = rep.safety;
  q.non_ionizing = rep.non_ionizing;
  q.volume_pre = volume_pre;
  q.material = rep.material;
  q.score_slot = score_slot;
  q.child_count = child_count;
  q.sec_last = sec_last;
  q.status = status;
  q.species = species;
  q.kind = kind;
  q.xs_at_step_start = xs_at_step_start;
  // What the step should REPORT, which is the flag as the track arrived with it. The stepping
  // kernel recomputes `p.flags` from the step's own status AFTER the step; a queued step ended
  // on the interaction rather than on a boundary, so the track still carries the pre-step
  // value here and `run_interaction` does that recompute instead.
  q.first_in_volume = (track.flags & kFirstStepInVolume) != 0u;
  return queue.push(q) >= 0;
}

/// WHICH at-rest nuclear arm QBBC gives this species, or `kNone`.
///
/// `stopping::stopping_arm` collapsed onto the two BUCKETS a kernel can be built for, which is
/// the distinction that matters here and not the one that matters there:
/// `G4MuonMinusCapture` and `G4HadronicAbsorptionBertini` are different processes with different
/// de-excitation choices, and `stopping::at_rest` tells them apart itself off the PDG code -
/// but both end in BERTINI, so both go in the kernel that carries Bertini. The anti-baryons end
/// in FTFP and go in the other one. A kernel that carried both would be the 22 GB ptxas run
/// `InteractionBucket`'s own comment records.
__host__ __device__ inline InteractionBucket at_rest_bucket(ParticleType t) {
  const int pdg = pdg_code(t);
  if (pdg == 13) { return InteractionBucket::kAtRest; }  // mu-, G4MuonMinusCapture
  switch (pdg) {
    case -211:    // pi-            G4HadronicAbsorptionBertini
    case -321:    // K-
    case 3112:    // Sigma-
    case 3312:    // Xi-
    case 3334:    // Omega-
      return InteractionBucket::kAtRest;
    case -2212:   // anti-proton    G4HadronicAbsorptionFritiof
    case -2112:   // anti-neutron   (neutral, and in the list)
    case -3122:   // anti-Lambda    (neutral)
    case -3212:   // anti-Sigma0    (neutral)
    case -3222:   // anti-Sigma+    (charge -1)
    case -3322:   // anti-Xi0       (neutral)
      return InteractionBucket::kAtRest;
    default:
      break;
  }
  // Anti-nuclei: `particle->GetBaryonNumber() < -1`, which is the Fritiof arm.
  return (pdg <= -1000000000) ? InteractionBucket::kAtRest : InteractionBucket::kNone;
}

/// Does QBBC give this species an at-rest nuclear process?
///
/// `G4StoppingPhysics::ConstructProcess`'s two explicit lists plus the muon, as a function of
/// the PDG code - which is exactly what `stopping::stopping_arm` is, and this is a SECOND COPY
/// of it. The duplication is deliberate and is bounded by a test rather than by care:
/// `physics/stepper.cuh` cannot include `stopping/stopping_process.cuh`, because that header
/// pulls in Bertini and FTFP and would put both inside every stepping kernel (docs/RISK.md
/// V188) - and the stepper has to know, before it enqueues, whether there is anything to
/// enqueue.
///
/// `tests/test_inelastic_transport.cu` asserts this function against `stopping::stopping_arm`
/// for every `ParticleType` in the enum, so a species added to one and not the other is a test
/// failure and not a silent divergence. That assertion is the reason this is a copy and not a
/// guess.
///
/// It answers about the SPECIES and not about the stage: whether the capture actually runs is
/// `HadronicWiring::hadron_at_rest` and `HadronicStage`, which the caller tests.
__host__ __device__ inline bool has_at_rest_arm(ParticleType t) {
  return at_rest_bucket(t) != InteractionBucket::kNone;
}

/// `G4ParticleDefinition::GetBaryonNumber()` for a transported track.
///
/// `ParticleDef` does not carry one - it is an EM struct, built for dE/dx and delta rays - and
/// two things here need it: `G4EnergyRangeManager` divides the energy by |B| for any |B| > 1,
/// which is what makes an alpha's model choice per nucleon, and `G4StoppingPhysics`'s
/// anti-nucleus arm tests `B < -1`.
///
/// Off the PDG code, as Geant4 does. A nuclear code is 10LZZZAAAI, so A is `(|pdg|/10) % 1000`
/// and the sign is the code's; for `kGenericIon` the code is G4GenericIon's placeholder and the
/// mass number travels on the track instead, which is what `ion_a` is for.
__host__ __device__ inline int baryon_number_of(ParticleType t, int ion_a) {
  if (t == ParticleType::kGenericIon) { return (ion_a > 0) ? ion_a : 1; }
  const int pdg = pdg_code(t);
  const int a = (pdg < 0) ? -pdg : pdg;
  if (a >= 1000000000) {
    const int mass_number = (a / 10) % 1000;
    return (pdg < 0) ? -mass_number : mass_number;
  }
  switch (pdg) {
    case 2212: case 2112: return 1;     // p, n
    case -2212: case -2112: return -1;  // pbar, nbar
    // The hyperons and anti-hyperons, which this port refuses at emission but which a queue
    // entry could carry if that ever changes. Listed rather than defaulted, so adding one to
    // the transported set does not silently give it baryon number 0.
    case 3122: case 3222: case 3212: case 3112: case 3322: case 3312: case 3334: return 1;
    case -3122: case -3222: case -3212: case -3112: case -3322: case -3312: case -3334:
      return -1;
    default: return 0;                  // the mesons and the leptons
  }
}

/// The RNG purpose the interaction kernel draws on.
///
/// A FOURTH PURPOSE, and it is what makes the split possible at all. `Philox` is counter-based
/// on `(rng_key, step, purpose)` and exposes no way to resume a partly consumed stream, so the
/// interaction kernel cannot continue the stepper's `0xB19Du` stream from where the interaction
/// length left it. It opens its own instead, keyed by the same `(rng_key, step)` - deterministic,
/// reproducible across batch sizes and block sizes, and independent of every other stream a
/// track of that key draws, which is the property `core/rng.cuh` is built to give and the one
/// the gamma, lepton and hadron purposes already rely on.
///
/// What this costs in faithfulness is nothing that was not already spent: this transport's
/// random stream differs from Geant4's by construction (it re-draws every interaction length
/// each step rather than carrying `theNumberOfInteractionLengthLeft` - see
/// `decay_in_flight_length`'s header), so what a comparison rests on is that every draw has the
/// right distribution and that the run is reproducible. Both hold.
inline constexpr unsigned int kInteractionRngPurpose = 0x1E7Au;

// =============================================================================================
// The workspace pool
// =============================================================================================

/// Every per-thread buffer ONE interaction needs, as a view onto the pool.
///
/// Each member is an array of `n_slots`, and thread `i` takes element `i`. The struct is copied
/// by value into the launch, like `had::HadronicWiring` and `ftf::entry::Handle` are.
///
/// WHAT A SLOT COSTS, and every number is `sizeof` rather than an estimate (the test prints the
/// table from the types themselves):
///
///     ftf::entry::HadronWorkspace          458,096 B   the FTFP arm
///     Bertini (workspace, nuclei model,    210,552 B   the Bertini arm and the at-rest one
///       four CollisionOutput, ColliderOutput)
///     Binary cascade (nucleons, scratch,   234,384 B   the BIC and light-ion arms
///       fields, track pool, collisions, buffers, products)
///     PreCompound + de-excitation arrays   536,576 B   the tail of ALL FOUR
///     two HadFinalState<real_t, 256> and
///       one HadronicStepResult              ~74,000 B
///                                        -----------
///                                        about 1.48 MB per slot
///
/// The de-excitation arrays are the largest item and they are shared by every arm, which is why
/// the pool is one pool and not four: a slot that could run only FTFP would still carry them.
/// Partitioning the queue by model would let the rarer arms have fewer slots, and is not done
/// here because the measurement says the slot count is not the binding constraint - see
/// docs/RISK.md V189.
template <typename real_t>
struct InteractionSlotCounts {
  int n_slots = 0;
  /// `kMaxSecondaries` of the final states in the pool. Not a template parameter, so that the
  /// host allocator and the kernel cannot disagree about it.
  static constexpr int kSecondaryCap = 256;
};

}  // namespace g4gpu::had
