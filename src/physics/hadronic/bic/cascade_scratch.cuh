// The Binary cascade's fixed-size working arrays, held in caller storage and not on the stack.
//
// WHY THIS FILE EXISTS. Every function of `Propagate`'s call tree declared its scratch lists as
// locals - the indices of the tracks that crossed the surface in one time step, the target list
// `FindAndFillCluster` walks, the final state `G4DecayKineticTracks` decays - each 256 entries
// long, and ptxas lays out one frame for the whole call tree of `run_interaction<kBinary>`:
// 69,920 bytes at sm_86, 69,904 at sm_70, and `run_interaction<kLightIon>`, which runs the same
// `propagate`, 61,728 (docs/RISK.md V237's addendum, V239). The device stack raise
// (`RaiseStackForInteractions`, V190, V196) reserves the largest frame for EVERY thread the card
// can hold resident, not for the threads a launch has: 77,824 bytes a thread over the Titan V's
// 163,840 resident threads is 11.9 GB of its 12, and the proton depth-dose gate paged through
// system memory there (V239). docs/RISK.md V256 measures what each array below cost.
//
// So the arrays live here, ONE `BicScratch` per interaction slot (`had::InteractionSlot`), reached
// through `CascadeWorkspace::scratch` the way the pool, the collision list and `CascadeBuffers`
// already are. It is NOT a kernel local (that is the frame this removes), NOT `__shared__` (a
// block's worth of these is far past the 48 kB a block may hold) and NOT malloc (the device heap
// is a second reservation of its own). The slot is uploaded from one constructed host image
// (`build_interaction_pool`), so this costs no allocation of its own.
//
// EVERY MEMBER IS ITS OWN ARRAY, AND THAT IS DELIBERATE. Several of these are live at once - the
// `secondaries` list `step_particles_out` hands to `FindCollisions` is still being read while
// `schedule_absorption` fills the cluster arrays, and `do_time_step` runs inside
// `step_particles_out` - and a shared buffer would be a correct answer only for as long as
// nobody reorders a call. None of these functions recurses or is re-entered while it is active,
// so one array per declaration is exactly the stack's semantics with none of its cost. Nothing
// is read before the function that owns it has written it in the same call: each list is filled
// from index 0 and read only below the count it was filled to, which is what the locals did -
// they were never initialised either, apart from `DecayTrack`'s default member initialisers,
// which `propagate` overwrites entry by entry (`dl[n] = DecayTrack{}`) before it reads one.
//
// The bound is `kCascadeMaxScratch` = 256 for every list, which is what the locals were. An
// overflow is the refusal or the clamp each function already had; nothing here changes a
// capacity.
#ifndef G4GPU_BIC_CASCADE_SCRATCH_CUH
#define G4GPU_BIC_CASCADE_SCRATCH_CUH

#include "physics/hadronic/bic/cascade_state.cuh"

namespace g4gpu::bic {

/// The most tracks one of the cascade's scratch index lists will hold. These index the SECONDARY,
/// CAPTURED and FINAL lists, or the target list for `FindAndFillCluster`, and 256 is the bound
/// every file in this package used for them; the overflow is a refusal or a clamp, never a silent
/// truncation of physics.
inline constexpr int kCascadeMaxScratch = 256;

/// One thread's scratch lists for the Binary cascade's `Propagate` and everything it calls - the
/// arrays that were stack locals until docs/RISK.md V256. Members are grouped by the function
/// that owns them; within a function, by the order it declared them.
struct BicScratch {
  // ---- `propagate` (cascade_propagate.cuh), `G4BinaryCascade::Propagate`.
  /// `FindCollisions(&theSecondaryList)`, "if called stand alone find first collisions": the
  /// pool indices of the secondary list, in pool order. 1,024 bytes.
  int initial[kCascadeMaxScratch];
  /// `G4DecayKineticTracks decay(&theFinalState)` after `DeExcite`: theFinalState in its own
  /// order, which the shared decay engine grows in place with each short-lived track's
  /// daughters. 88 bytes an entry, 22,528 - the largest single array the frame held.
  DecayTrack decay_list[kCascadeMaxScratch];

  // ---- `capture_secondaries` (cascade_propagate.cuh), `G4BinaryCascade::Capture`.
  /// `captured`: the secondaries `Capture` moves to theCapturedList, which
  /// `UpdateTracksAndCollisions(&captured, NULL, NULL)` then strips of their collisions.
  int capture_captured[kCascadeMaxScratch];

  // ---- `decay_void_nucleus` (cascade_deexcite.cuh), `G4BinaryCascade::DecayVoidNucleus`.
  /// The pool index of each nucleon left, theTargetList then theCapturedList.
  int void_index[kCascadeMaxScratch];
  /// Their PDG masses, `G4KineticTrack::Get4Momentum` of the nucleons the Kopylov decay shares
  /// the remnant's energy over.
  double void_mass[kCascadeMaxScratch];
  /// `fermi_phase_space_decay`'s answer, one four-momentum per nucleon. 8,192 bytes.
  imr::LorentzVector void_momenta[kCascadeMaxScratch];

  // ---- `schedule_absorption` (cascade_find.cuh), `G4Absorption`'s `FindAndFillCluster`.
  /// The target list as `FindAndFillCluster` walks it for one candidate: charge, position and pool
  /// index of each target nucleon. 1,024 + 6,144 + 1,024 bytes.
  int cluster_charges[kCascadeMaxScratch];
  deex::Vec3d cluster_positions[kCascadeMaxScratch];
  int cluster_ids[kCascadeMaxScratch];

  // ---- `step_particles_out` (cascade_finish.cuh), `G4BinaryCascade::StepParticlesOut`.
  /// The secondary list `FindCollisions` is re-run on after a hundred steps with no collision.
  int stepout_secondaries[kCascadeMaxScratch];
  /// `Capture(true)`'s `captured`, as in `capture_secondaries`.
  int stepout_captured[kCascadeMaxScratch];

  // ---- `do_time_step` (cascade_step.cuh), `G4BinaryCascade::DoTimeStep`.
  /// `kt_outside` and `kt_inside`: the secondaries that were outside and inside the nucleus
  /// BEFORE the step - taken before transporting, which is the order that decides who crossed.
  int step_was_outside[kCascadeMaxScratch];
  int step_was_inside[kCascadeMaxScratch];
  /// `kt_gone_in` and `kt_gone_out`, which `CorrectBarionsOnBoundary` corrects.
  int step_gone_in[kCascadeMaxScratch];
  int step_gone_out[kCascadeMaxScratch];
  /// `kt_captured`: the secondaries the step left `captured`.
  int step_captured[kCascadeMaxScratch];
};

}  // namespace g4gpu::bic

#endif
