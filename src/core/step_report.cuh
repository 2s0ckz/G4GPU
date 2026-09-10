// What a stepper reports about the step it just took, beyond the energy it deposited.
//
// This exists so that the Geant4-shaped G4Step accessors in g4/G4DeviceStep.hh have something
// to read. Almost everything on it is a value the stepper already computed and then threw
// away: which process won the competition for the step, whether geometry cut it short, the
// pre-step safety, the material. None of it is new physics and none of it changes a random
// draw - tests/test_step_hook.cu and `g4dose -verify-step-hook` hold the dose to the last
// digit across the change.
//
// It is a struct rather than a growing list of out-parameters because there are now seven of
// them and stepper.cuh has three entry points that all have to agree.
#pragma once
#include "core/particle.cuh"

namespace g4gpu {

/// G4StepStatus. Same names and same meanings as Geant4's, minus the ones that cannot arise
/// in this transport - there is no at-rest process table here, and nothing is forced.
enum class StepStatus : int {
  fUndefined = 0,
  /// The step ended on a volume boundary. G4Step::IsLastStepInVolume is this.
  fGeomBoundary = 1,
  /// The step ended at the world boundary and the track left.
  fWorldBoundary = 2,
  /// A discrete process fired: Compton, photoelectric, pair, Rayleigh, brems, a delta ray,
  /// annihilation in flight.
  fPostStepDoItProc = 3,
  /// Nothing discrete happened. The step was limited by the continuous-loss step limit or by
  /// multiple scattering, and the track simply lost energy along it.
  fAlongStepDoItProc = 4,
  /// The track fell below its tracking cut and was stopped and killed where it stood.
  fStopAndKill = 5,
};

/// Which process defined the step - the device answer to
/// G4StepPoint::GetProcessDefinedStep(), which returns a G4VProcess* that cannot exist here.
///
/// An enum rather than a pointer because a process object is a host object; the identity is
/// what a stepping action actually reads, and the identity is all this carries.
enum class ProcessId : int {
  fNotDefined = 0,
  fTransportation = 1,   ///< the step ended on geometry, not on physics
  fCompton = 2,
  fPhotoelectric = 3,
  fGammaConversion = 4,  ///< pair production
  fRayleigh = 5,
  fIonisation = 6,       ///< the continuous loss, or the delta ray that ended the step
  fBremsstrahlung = 7,
  fAnnihilation = 8,
  fMultipleScattering = 9,
  fNuclearStopping = 10,
  /// Below the tracking cut - Geant4 would call this the tracking manager rather than a
  /// process, and reports fStopAndKill on the track instead.
  fBelowTrackingCut = 11,

  // Nothing below is produced by any stepper in this port yet. They are here so that the
  // process which lands tomorrow has a value to report on the day it is written, instead of
  // an enum that has to be widened - and widened in lockstep with every switch over it -
  // before it can say what it did. The set is the one this port is being built toward; a
  // process outside it should be ADDED here rather than folded into a nearby name, because
  // fNotDefined is reserved for exactly one meaning: a branch nobody annotated. That meaning
  // is load-bearing - `g4dose -verify-step-hook` fails the build on it.
  fHadronElastic = 12,
  fHadronInelastic = 13,
  fNeutronCapture = 14,
  fPhotoNuclear = 15,
  fElectroNuclear = 16,
  fMuonNuclear = 17,
  fDecay = 18,
  fRadioactiveDecay = 19,
  fCoulombScattering = 20,
  fPairProdByCharged = 21,   ///< mu+/mu- -> e+e- pair production
  fAnnihilationToMuMu = 22,
  fAnnihilationToHadrons = 23,
  fCerenkov = 24,
  fScintillation = 25,
  fTransitionRadiation = 26,
  fSynchrotronRadiation = 27,
  fAtomicDeexcitation = 28,
  fUserDefined = 29,         ///< a process a project added that this enum does not name
  /// The neutron time cut, which is a PROCESS and not the tracking manager's energy cut.
  /// `G4NeutronKiller` when the physics list registers one; in 11.1.1
  /// `EnableNeutronGeneralProcess` is 1, so it is the identical two lines inside
  /// `G4NeutronGeneralProcess::PostStepDoIt` instead. Distinct from fBelowTrackingCut on
  /// purpose: this one kills on the CLOCK and deposits nothing, where fBelowTrackingCut kills
  /// on energy and deposits what is left. Folding them together would have made a step that
  /// loses energy indistinguishable from one that gives it to a volume, which is the whole
  /// difference between the two.
  fNeutronKiller = 30,
};

/// G4TrackStatus. The complete Geant4 set, not the subset this transport can act on.
///
/// A stepping action may set it - see G4VUserDeviceSteppingAction - and the kernel reads it
/// back after the hook returns to decide whether the track is requeued. What CAN be honoured
/// here is fAlive and fStopButAlive (requeue) against fStopAndKill and
/// fKillTrackAndSecondaries (do not).
///
/// fSuspend and fPostponeToNextEvent need a track stack that this transport does not have -
/// every track in flight is in a species buffer being stepped in lockstep, and there is
/// nowhere to put one aside. They are in the enum anyway, because an action written against
/// a Geant4 project should COMPILE, and because silently treating a suspend as a kill is the
/// failure this project keeps writing up. Instead the engine counts them and RunStats
/// reports the count, so a run that asked for something it did not get says so.
enum class TrackStatus : int {
  fAlive = 0,
  fStopButAlive = 1,
  fStopAndKill = 2,
  fKillTrackAndSecondaries = 3,
  fSuspend = 4,
  fPostponeToNextEvent = 5,
};

/// Filled by a stepper for the step it just took.
///
/// Every field is that step's own. Defaults are what a step that did nothing would report, so
/// a path that returns early leaves something meaningful rather than stale.
template <typename real_t>
struct StepReport {
  /// mm actually travelled - the TRUE path length, not |post - pre|. Multiple scattering
  /// deflects inside a step, so the displacement is shorter than the distance run and a LET
  /// taken from it is biased high. This is what G4Step::GetStepLength returns.
  real_t true_length = 0;
  /// Isotropic safety at the pre-step point, mm. Negative when the stepper did not need it -
  /// only the lepton path computes one, because only it asks multiple scattering for a limit.
  /// Geant4 has the same property for the same reason and returns whatever the navigator last
  /// computed; a negative value here says "not computed" rather than pretending to a number.
  real_t safety = real_t(-1);
  /// Energy deposited by nuclear recoils, MeV - G4Step::GetNonIonizingEnergyDeposit. Only the
  /// hadron path produces any; it is part of `edep`, not additional to it.
  real_t non_ionizing = 0;
  /// Material index at the pre-step point, or -1 outside the world.
  int material = -1;
  /// How the step ended.
  StepStatus status = StepStatus::fUndefined;
  /// What ended it.
  ProcessId process = ProcessId::fNotDefined;
  /// Secondaries this step handed to the emitter. Geant4 lets a stepping action walk the
  /// actual secondary vector; the tracks here have already gone into their species buffers by
  /// the time the hook is called, so what is reported is how many, not which.
  int n_secondaries = 0;
  // NOT here yet: IsFirstStepInVolume. A step knows it ENDED on a boundary (status ==
  // fGeomBoundary, which is IsLastStepInVolume), but knowing it STARTED on one means carrying
  // a bit across the step boundary, which is a field on every track rather than a value the
  // kernel already holds. It is grouped with the other per-track costs - time, track length,
  // parent id, vertex - and is not free, so it is not smuggled in here as a field that would
  // always read false.
};

}  // namespace g4gpu
