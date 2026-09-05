/// \file B1/src/SteppingAction.cc
/// \brief Implementation of the B1::SteppingAction class
///
/// Character for character what Geant4's B1 SteppingAction does: look up the scoring volume
/// once, ignore steps that are not in it, and add the energy deposit to the event.
///
/// One thing about it is worth knowing, and it is a property of this transport rather than of
/// this file. The step handed to UserSteppingAction here is an aggregate: one per event and
/// per volume with a sensitive detector, carrying that event's total in that volume, because
/// the individual steps happen inside a device kernel millions of times a second. This action
/// adds what it is given, and the sum over an event of the per-step deposits is the event's
/// deposit, so the number it produces is exactly the number Geant4 produces - which is what
/// the 0.09-sigma agreement in docs/RESULT.md is measured on.
///
/// An action that did something non-additive with a step - took a maximum, tested a threshold
/// per step, cared where inside the volume the step was - would not be so lucky. G4Step's
/// IsAggregated() returns true so that such an action can detect this and say so. The whole
/// story is at the top of g4/G4Step.hh.

#include "SteppingAction.hh"

#include "DetectorConstruction.hh"
#include "EventAction.hh"

#include "G4Event.hh"
#include "G4LogicalVolume.hh"
#include "G4RunManager.hh"
#include "G4Step.hh"

namespace B1 {

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

SteppingAction::SteppingAction(EventAction* eventAction) : fEventAction(eventAction) {}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void SteppingAction::UserSteppingAction(const G4Step* step) {
  if (!fScoringVolume) {
    const auto detConstruction = static_cast<const DetectorConstruction*>(
        G4RunManager::GetRunManager()->GetUserDetectorConstruction());
    fScoringVolume = detConstruction->GetScoringVolume();
  }

  // get volume of the current step
  G4LogicalVolume* volume =
      step->GetPreStepPoint()->GetTouchableHandle()->GetVolume()->GetLogicalVolume();

  // check if we are in scoring volume
  if (volume != fScoringVolume) return;

  // collect energy deposited in this step
  G4double edepStep = step->GetTotalEnergyDeposit();
  fEventAction->AddEdep(edepStep);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

}  // namespace B1
