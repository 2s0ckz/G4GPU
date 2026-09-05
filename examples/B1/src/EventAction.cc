/// \file B1/src/EventAction.cc
/// \brief Implementation of the B1::EventAction class
///
/// Unchanged from Geant4's. The energy this accumulates arrives from the device rather than
/// from a CPU stepping loop, but it arrives per event, so resetting at begin of event and
/// handing the total to the RunAction at end of event means exactly what it means in Geant4 -
/// including for the second moment, which is what the run's rms is computed from.

#include "EventAction.hh"

#include "G4Event.hh"
#include "G4RunManager.hh"
#include "RunAction.hh"

namespace B1 {

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

EventAction::EventAction(RunAction* runAction) : fRunAction(runAction) {}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void EventAction::BeginOfEventAction(const G4Event*) { fEdep = 0.; }

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void EventAction::EndOfEventAction(const G4Event*) {
  // accumulate statistics in run action
  fRunAction->AddEdep(fEdep);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

}  // namespace B1
