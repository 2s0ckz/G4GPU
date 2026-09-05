/// \file exampleB1.cc
/// \brief Main program of the B1 example
///
/// The same shape as Geant4's exampleB1.cc: a run manager from the factory, a detector, a
/// physics list, an action initialisation, a visualisation manager, and either a macro or an
/// interactive session.
///
///   exampleB1                     interactive: runs init_vis.mac, opens the viewer
///   exampleB1 run1.mac            batch: the macro drives everything
///   exampleB1 -n 2000000          batch: this many events, no macro
///
/// -n is the one addition. It exists because the dose check in build_all.bat wants an event
/// count on the command line and nothing else.

#include "ActionInitialization.hh"
#include "DetectorConstruction.hh"

#include "G4RunManagerFactory.hh"
#include "G4SteppingVerbose.hh"
#include "G4UImanager.hh"
#include "QBBC.hh"

#include "G4UIExecutive.hh"
#include "G4VisExecutive.hh"

#include "G4ios.hh"
#include "Randomize.hh"

#include <cstdlib>
#include <cstring>
#include <string>

using namespace B1;

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

int main(int argc, char** argv) {
  int n_events = 0;
  std::string macro;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
      n_events = std::atoi(argv[++i]);
    } else if (argv[i][0] != '-') {
      macro = argv[i];
    }
  }

  // Detect interactive mode (no arguments) and define UI session
  //
  G4UIExecutive* ui = nullptr;
  if (argc == 1) { ui = new G4UIExecutive(argc, argv); }

  // Optionally: choose a different Random engine... (host side only; see g4/Randomize.hh)
  // G4Random::setTheEngine(new CLHEP::MTwistEngine);

  // use G4SteppingVerboseWithUnits
  G4int precision = 4;
  G4SteppingVerbose::UseBestUnit(precision);

  // Construct the default run manager
  //
  auto* runManager = G4RunManagerFactory::CreateRunManager(G4RunManagerType::Default);

  // Set mandatory initialization classes
  //
  // Detector construction
  runManager->SetUserInitialization(new DetectorConstruction());

  // Physics list
  G4VModularPhysicsList* physicsList = new QBBC;
  physicsList->SetVerboseLevel(1);
  runManager->SetUserInitialization(physicsList);

  // User action initialization
  runManager->SetUserInitialization(new ActionInitialization());

  // The production range cut. Geant4's QBBC sets 0.7 mm through its own default; setting it
  // here is what makes this run comparable to the reference in docs/RESULT.md.
  runManager->SetCutValue(0.7 * mm);

  // Initialize visualization
  //
  G4VisManager* visManager = new G4VisExecutive;
  visManager->Initialize();

  // Get the pointer to the User Interface manager
  G4UImanager* UImanager = G4UImanager::GetUIpointer();

  // Process macro or start UI session
  //
  if (!macro.empty()) {
    // batch mode
    //
    // Geant4's exampleB1 discards ApplyCommand's return value. This does not: a macro whose
    // commands failed - a misspelled command, a /vis/ command before /run/initialize - has
    // not done what the file said, and a program that reports success anyway is how a run
    // gets published with the physics it was supposed to have switched on still off.
    G4String command = "/control/execute ";
    const G4int errors = UImanager->ApplyCommand(command + macro);
    if (errors != 0) {
      G4cout << macro << ": " << errors << " command(s) failed" << G4endl;
      delete visManager;
      delete runManager;
      return 1;
    }
  } else if (n_events > 0) {
    // batch mode, event count on the command line
    runManager->Initialize();
    runManager->BeamOn(n_events);
  } else {
    // interactive mode
    UImanager->ApplyCommand("/control/execute init_vis.mac");
    ui->SessionStart();
    delete ui;
  }

  // Job termination
  // Free the store: user actions, physics_list and detector_description are
  // owned and deleted by the run manager, so they should not be deleted
  // in the main() program !

  delete visManager;
  delete runManager;
  return 0;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
