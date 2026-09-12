/// \file QualityFactorScoring.hh
///
/// A project's own step-level stepping action, in the project's own header.
///
/// IT IS A HEADER, AND SINCE P8e IT HAS TO BE. The class used to sit in test_custom_hook.cu,
/// which was the honest shape while the project compiled its own kernels in that same file. It
/// no longer does: `build_hook_engine.bat` generates one translation unit per stepping kernel
/// for this hook type, and every one of those units has to see the class. docs/RISK.md V65.
///
/// Nothing about the class changed in the move, which is the point - the arrangement a project
/// uses to get a device stepping action into the transport is still "write an ordinary class",
/// and the only new requirement is that it lives where more than one .cu can include it, which
/// is where a Geant4 project's action class lives anyway.
#ifndef QualityFactorScoring_h
#define QualityFactorScoring_h 1

#include "g4/G4VUserDeviceSteppingAction.hh"

// In a real project this is include/QualityFactorScoring.hh, which is exactly where it is. It
// is an ordinary class: base class, member data, a named method. The two differences from
// Geant4 are that the base takes the derived type (CRTP - so the call inlines instead of going
// through a vtable, which at 3e7 steps a second is worth the odd-looking declaration) and that
// the method is __device__.
class QualityFactorScoring : public G4VUserDeviceSteppingAction<QualityFactorScoring> {
 public:
  QualityFactorScoring() = default;
  QualityFactorScoring(G4double* weighted, G4double* plain, G4double* sec_walked,
                       int n_events, int slot)
      : weighted_(weighted), plain_(plain), sec_walked_(sec_walked), n_events_(n_events),
        slot_(slot) {}

  /// Called once per real step of every track.
  ///
  /// Written in Geant4's spellings throughout - GetTotalEnergyDeposit, GetStepLength,
  /// GetPreStepPoint()->GetKineticEnergy() - because the point of the exercise is that a
  /// stepping action ported from a Geant4 project reads the way it did there. The two
  /// differences are visible in the signature, not the body: the method is __device__, and the
  /// step arrives by reference rather than as a G4Step*.
  __device__ void UserSteppingAction(const G4DeviceStep& step) const {
    if (weighted_ == nullptr) { return; }
    if (step.GetScoreSlot() != slot_) { return; }
    if (step.GetTotalEnergyDeposit() == 0) { return; }
    if (step.GetEventID() < 0 || step.GetEventID() >= n_events_) { return; }

    // The quantity that needs a real step: energy deposited per unit of the path actually
    // travelled. A deliberately blocky Q(LET) so the expected answer can be checked by hand.
    const G4double edep = step.GetTotalEnergyDeposit();
    const G4double len = step.GetStepLength();
    const G4double let = (len > 0) ? edep / len : 0.0;
    const G4double q = (let < 1.0) ? 1.0 : (let < 10.0 ? 5.0 : 20.0);

    // Nothing below is used by the arithmetic; it is here so that the test fails to compile if
    // any of these stops being reachable the way a Geant4 stepping action reaches it.
    const G4double ke_in = step.GetPreStepPoint()->GetKineticEnergy();
    const G4double ke_out = step.GetPostStepPoint()->GetKineticEnergy();
    const G4StepStatus st = step.GetPostStepPoint()->GetStepStatus();
    const G4ProcessId pr = step.GetPostStepPoint()->GetProcessDefinedStep();
    const G4double beta = step.GetPreStepPoint()->GetBeta();
    const int mat = step.GetPreStepPoint()->GetMaterial();
    const unsigned int tid = step.GetTrack()->GetTrackID();
    (void)ke_in; (void)ke_out; (void)st; (void)pr; (void)beta; (void)mat; (void)tid;

    // The secondaries this step made, walked as real tracks. Two things are asserted on the
    // host afterwards from what this accumulates: that the chain is exactly as long as
    // GetNumberOfSecondariesInCurrentStep() says, and that every secondary carries less
    // energy than the step that made it had - which is the cheapest statement that would
    // fail if the chain were walking into the wrong buffer or the wrong slot.
    int walked = 0;
    G4double worst_excess = 0;
    for (auto it = step.GetSecondaryInCurrentStep().begin(); it.valid(); it.advance()) {
      const auto sec = it.get();
      ++walked;
      // The most energy a secondary can carry is the parent's TOTAL energy plus one
      // target electron's rest mass. Kinetic energy alone is the wrong bound and this test
      // found that out: annihilation turns rest mass into two 511 keV photons, so a
      // positron that has all but stopped produces secondaries far above its own kinetic
      // energy. The bound below is exactly tight for that case - T + m_e + m_e - and holds
      // for every other process here, where a secondary cannot exceed the parent's T.
      const G4double ceiling = ke_in + step.GetPreStepPoint()->GetMass()
                               + G4double(0.510998910);
      const G4double excess = sec.GetKineticEnergy() - ceiling;
      if (excess > worst_excess) { worst_excess = excess; }
    }
    atomicAdd(&sec_walked_[0], static_cast<G4double>(walked));
    atomicAdd(&sec_walked_[1],
              static_cast<G4double>(step.GetNumberOfSecondariesInCurrentStep()));
    if (worst_excess > 0) { atomicAdd(&sec_walked_[2], worst_excess); }

    // Two running sums per event. Their ratio is the dose-averaged Q. Bounded by the number of
    // events, not by the number of steps - which is the whole discipline of core/step_hook.cuh.
    atomicAdd(&weighted_[step.GetEventID()], q * edep);
    atomicAdd(&plain_[step.GetEventID()], edep);
  }

 private:
  G4double* weighted_ = nullptr;
  G4double* plain_ = nullptr;
  /// [0] secondaries walked, [1] secondaries reported, [2] total energy excess.
  G4double* sec_walked_ = nullptr;
  int n_events_ = 0;
  int slot_ = 0;
};

#endif  // QualityFactorScoring_h
