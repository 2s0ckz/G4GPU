/// \file B1/src/RunAction.cc
/// \brief Implementation of the B1::RunAction class
///
/// The first half is Geant4's B1 RunAction: register the dose units, accumulate the energy
/// deposit and its square, and print the dose with its rms at end of run. It reads the mass
/// off the scoring volume and the run conditions off the gun, as Geant4's does.
///
/// The second half, after the marked line, is this project's: the comparison against
/// Geant4 11.1.1's own answer for this example, and the throughput. Neither belongs in an
/// example a user is meant to copy, and both are why this example exists here at all.

#include "RunAction.hh"

#include <cmath>

#include "DetectorConstruction.hh"
#include "PrimaryGeneratorAction.hh"

#include "G4AccumulableManager.hh"
#include "G4LogicalVolume.hh"
#include "G4Run.hh"
#include "G4RunManager.hh"
#include "G4SystemOfUnits.hh"
#include "G4UnitsTable.hh"
#include "G4ios.hh"

namespace B1 {

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

RunAction::RunAction() {
  // add new units for dose
  //
  const G4double milligray = 1.e-3 * gray;
  const G4double microgray = 1.e-6 * gray;
  const G4double nanogray = 1.e-9 * gray;
  const G4double picogray = 1.e-12 * gray;

  new G4UnitDefinition("milligray", "milliGy", "Dose", milligray);
  new G4UnitDefinition("microgray", "microGy", "Dose", microgray);
  new G4UnitDefinition("nanogray", "nanoGy", "Dose", nanogray);
  new G4UnitDefinition("picogray", "picoGy", "Dose", picogray);

  // Register accumulable to the accumulable manager
  G4AccumulableManager* accumulableManager = G4AccumulableManager::Instance();
  accumulableManager->RegisterAccumulable(fEdep);
  accumulableManager->RegisterAccumulable(fEdep2);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void RunAction::BeginOfRunAction(const G4Run*) {
  // inform the runManager to save random number seed
  G4RunManager::GetRunManager()->SetRandomNumberStore(false);

  // reset accumulables to their initial values
  G4AccumulableManager* accumulableManager = G4AccumulableManager::Instance();
  accumulableManager->Reset();
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void RunAction::EndOfRunAction(const G4Run* run) {
  G4int nofEvents = run->GetNumberOfEvent();
  if (nofEvents == 0) return;

  // Merge accumulables
  G4AccumulableManager* accumulableManager = G4AccumulableManager::Instance();
  accumulableManager->Merge();

  // Compute dose = total energy deposit in a run and its variance
  //
  G4double edep = fEdep.GetValue();
  G4double edep2 = fEdep2.GetValue();

  G4double rms = edep2 - edep * edep / nofEvents;
  if (rms > 0.)
    rms = std::sqrt(rms);
  else
    rms = 0.;

  const auto detConstruction = static_cast<const DetectorConstruction*>(
      G4RunManager::GetRunManager()->GetUserDetectorConstruction());
  G4double mass = detConstruction->GetScoringVolume()->GetMass();
  G4double dose = edep / mass;
  G4double rmsDose = rms / mass;

  // Run conditions
  //  note: There is no primary generator action object for "master"
  //        run manager for multi-threaded mode.
  const auto generatorAction = static_cast<const PrimaryGeneratorAction*>(
      G4RunManager::GetRunManager()->GetUserPrimaryGeneratorAction());
  G4String runCondition;
  G4String beamParticle;
  G4double beamEnergy = 0;
  if (generatorAction) {
    const G4ParticleGun* particleGun = generatorAction->GetParticleGun();
    beamParticle = particleGun->GetParticleDefinition()->GetParticleName();
    runCondition += beamParticle;
    runCondition += " of ";
    G4double particleEnergy = particleGun->GetParticleEnergy();
    beamEnergy = particleEnergy;
    runCondition += G4BestUnit(particleEnergy, "Energy");
  }

  // Print
  //
  if (IsMaster()) {
    G4cout << G4endl << "--------------------End of Global Run-----------------------";
  } else {
    G4cout << G4endl << "--------------------End of Local Run------------------------";
  }

  G4cout << G4endl << " The run consists of " << nofEvents << " " << runCondition << G4endl
         << " Cumulated dose per run, in scoring volume : " << G4BestUnit(dose, "Dose")
         << " rms = " << G4BestUnit(rmsDose, "Dose") << G4endl
         << "------------------------------------------------------------" << G4endl << G4endl;

  // ---------------------------------------------------------------------------------------
  // Everything below here is this project's, not example B1's.
  // ---------------------------------------------------------------------------------------

  // Geant4 11.1.1's own answer for this example, from 2,000,000 events run by
  // ref/run/runb1.bat. Quoted scaled to 10,000 events so that any event count compares
  // against it, and with its own uncertainty, because the whole point of S1 in docs/RISK.md
  // is that a comparison without both uncertainties is not a comparison.
  // The quoted reference is for B1's *own* beam - 6 MeV gammas - and for nothing else. B1 can
  // now be run with `/gun/particle proton` (examples/B1/proton.mac) and with an alpha, and
  // printing a gamma's dose next to a proton's under the heading "Geant4 11.1.1" would be a
  // comparison in every visible respect and a comparison of nothing at all. So the block is
  // gated on the beam actually being B1's, and says so when it is not.
  const G4double kG4Dose10k = 427.385;  // picoGy
  const G4double kG4Sigma10k = 0.870;
  const G4bool isB1Beam =
      (beamParticle == "gamma" && std::fabs(beamEnergy / MeV - 6.0) < 1e-9);
  const G4double scaled = dose / picogray * 10000.0 / nofEvents;
  const G4double sigma = (dose > 0) ? scaled * (rmsDose / dose) : 0.0;

  G4cout << "mass of scoring volume = " << G4BestUnit(mass, "Mass")
         << ", edep = " << G4BestUnit(edep, "Energy") << G4endl;
  G4cout << "scaled to 10k events: " << scaled << " pGy  +/- " << sigma << " (this run)"
         << G4endl;
  if (isB1Beam) {
    G4cout << "Geant4 11.1.1        : " << kG4Dose10k << " pGy  +/- " << kG4Sigma10k
           << "    ratio = " << (scaled / kG4Dose10k) << G4endl;
    const G4double combined = std::sqrt(sigma * sigma + kG4Sigma10k * kG4Sigma10k);
    if (combined > 0) {
      G4cout << "difference           : " << (scaled - kG4Dose10k) << " pGy = "
             << (std::fabs(scaled - kG4Dose10k) / combined) << " sigma" << G4endl;
    }
  } else {
    G4cout << "Geant4 11.1.1        : no reference quoted - the stored number is for B1's own"
           << " 6 MeV gamma beam." << G4endl;
    G4cout << "                       For this beam, run the same macro against a real Geant4"
           << " build of B1" << G4endl;
    G4cout << "                       (ref/run/runb1.bat) and compare; note that B1's physics"
           << " list is QBBC," << G4endl;
    G4cout << "                       whose hadronic processes this transport does not have."
           << " See ref/b1hadron/." << G4endl;
  }

  const auto& st = G4RunManager::GetRunManager()->GetLastRunStats();
  // Two times, because they measure two different things and only one of them is comparable
  // to Geant4's. `event loop` is host wall clock over generating every primary and running
  // every batch - the counterpart of Geant4's own "Run Summary ... Real=", whose G4Timer
  // brackets InitializeEventLoop to TerminateEventLoop. `gpu` is CUDA-event time around the
  // batch loop alone, which is the smaller and more flattering number.
  G4cout << G4endl << "event loop " << st.event_loop_ms << " ms for " << nofEvents
         << " events = " << (nofEvents / (st.event_loop_ms * 1e-3)) << " events/s" << G4endl;
  G4cout << "time " << st.milliseconds << " ms for " << nofEvents << " events = "
         << (nofEvents / (st.milliseconds * 1e-3)) << " events/s   ("
         << (st.track_steps / (st.milliseconds * 1e-3)) << " track-steps/s)" << G4endl;
  G4cout << "total track-steps " << st.track_steps << " = "
         << (static_cast<G4double>(st.track_steps) / nofEvents)
         << " per event; max iterations per batch " << st.max_iterations << G4endl;
  if (st.abandoned > 0) {
    G4cout << "WARNING: " << st.abandoned << " tracks abandoned at the iteration limit"
           << G4endl;
  }
  if (st.overflow > 0) {
    G4cout << "WARNING: track buffer overflowed " << st.overflow
           << " times; raise the batch multiplier" << G4endl;
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void RunAction::AddEdep(G4double edep) {
  fEdep += edep;
  fEdep2 += edep * edep;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

}  // namespace B1
