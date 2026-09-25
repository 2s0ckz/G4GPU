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
/// energy, position, direction and volume, the true length, the safety, the material, the score
/// slot and the voxel cell. Eleven scalars against one hook call per step, which is the property
/// a stepping action depends on - `GetNumberOfSecondariesInCurrentStep()` on a step whose
/// secondaries are made in a later kernel is otherwise zero.
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
  int voxel_cell = -1;
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
  __device__ int push(const PendingInteraction<real_t>& q) const {
    if (items == nullptr || cursor == nullptr || capacity <= 0) { return -1; }
    const int i = atomicAdd(cursor, 1);
    if (i >= capacity) { return -1; }
    items[i] = q;
    return i;
  }
};

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
