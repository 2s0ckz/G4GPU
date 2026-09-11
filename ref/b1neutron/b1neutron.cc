// Example B1, unchanged, with G4NeutronGeneralProcess turned OFF - the stage-1 reference for a
// neutron, and the only way there is to produce one.
//
// WHY THIS PROGRAM EXISTS AND ref/B1build's exampleB1.exe WILL NOT DO
//
// docs/RISK.md V53: with `EnableNeutronGeneralProcess` set, the neutron's process manager holds
// `Transportation`, `Decay` and `NeutronGeneralProc` and nothing else, so
// `/process/inactivate neutronInelastic` answers "illegal process (or type) name" and there is
// no configuration in which the neutron's elastic and capture are active and its inelastic is
// not. That is the whole reason the stage-1 table's neutron row has been `0.0000 +/- 0.0000`
// against `0.0000 +/- 0.0000` since P8.
//
// **And the flag is settable, from C++, in exactly one window.** P8d's finding:
//
//   * `G4HadronInelasticQBBC::ConstructProcess` does NOT set it. Its CONSTRUCTOR does
//     (G4HadronInelasticQBBC.cc, the `G4VHadronPhysics("hInelasticQBBC")` initialiser body:
//     `param->SetEnableBCParticles(true); param->SetEnableNeutronGeneralProcess(true);`).
//     V53 and docs/PORTED.md 2.1.2 both name `ConstructProcess`, and the difference matters:
//     the constructor runs when `new QBBC` registers its physics constructors, and
//     `ConstructProcess` runs at `/run/initialize`.
//   * `G4HadronicParameters::SetEnableNeutronGeneralProcess` is public and guarded by
//     `if (!IsLocked())`, and `IsLocked()` is
//     `!G4Threading::IsMasterThread() || GetCurrentState() != G4State_PreInit`.
//
// So a master thread that calls the setter after constructing QBBC and before initialising the
// run manager is still in `G4State_PreInit` and the value takes. There is NO UI command for it:
// `G4HadronicParametersMessenger` builds exactly three - `/process/had/verbose`,
// `/process/had/maxEnergy` and `/process/had/enableCRCoalescence`. One line of C++ is the whole
// difference between this program and Geant4's own exampleB1, and it cannot be a macro.
//
// WHAT THE RESULTING PROCESS LIST IS, and the run prints it rather than claiming it.
// `G4HadProcesses::BuildNeutronElastic` and `BuildNeutronInelasticAndCapture` take their
// `else` branches and `ph->RegisterProcess` each process onto the neutron, and
// `G4NeutronTrackingCut::ConstructProcess` no longer finds a general process so it creates a
// real `G4NeutronKiller` (10 us, 0 MeV - the same two numbers the general process carries).
// The neutron then has:
//
//     Transportation, Decay, hadElastic, neutronInelastic, nCapture, nKiller
//
// and `/process/inactivate neutronInelastic` reaches one of them. THAT is the stage-1
// configuration: elastic and capture on their own data stores with their own interaction
// lengths, which is what `had::HadronicStage::kStage1` runs in the port.
//
// EVERYTHING ELSE IS GEANT4's OWN B1. The geometry, the scoring volume, the dose accumulation
// and the printed "Cumulated dose per run" line come from
// `$G4SRC/examples/basic/B1/{src,include}`, compiled from the Geant4 source tree by the
// CMakeLists beside this file - so the comparison is against the same B1 every other row of
// `ref/b1hadron/stage1_README.md` was measured with, and not a re-description of it. Only the
// main is ours, and only by the one line and the absence of the visualisation manager (this is
// a batch program).
//
// Usage:  b1neutron.exe <macro>

#include "ActionInitialization.hh"
#include "DetectorConstruction.hh"

#include "G4HadronicParameters.hh"
#include "G4RunManagerFactory.hh"
#include "G4SteppingVerbose.hh"
#include "G4UImanager.hh"
#include "QBBC.hh"
#include "Randomize.hh"

#include <cstdio>

using namespace B1;

int main(int argc, char** argv) {
  if (argc < 2) {
    std::printf("usage: b1neutron.exe <macro>\n");
    return 1;
  }

  G4int precision = 4;
  G4SteppingVerbose::UseBestUnit(precision);

  auto* runManager = G4RunManagerFactory::CreateRunManager(G4RunManagerType::Default);
  runManager->SetUserInitialization(new DetectorConstruction());

  // QBBC's own constructor chain sets the flag to true, so the order of these two statements is
  // the whole program.
  G4VModularPhysicsList* physicsList = new QBBC;
  physicsList->SetVerboseLevel(1);
  runManager->SetUserInitialization(physicsList);

  // THE ONE LINE. Still `G4State_PreInit` here, so `IsLocked()` is false and the setter takes;
  // `ConstructProcess` reads it at `/run/initialize`, which the macro does.
  G4HadronicParameters::Instance()->SetEnableNeutronGeneralProcess(false);
  std::printf("b1neutron: EnableNeutronGeneralProcess = %d after the setter (QBBC's "
              "constructor had set it to 1)\n",
              static_cast<int>(G4HadronicParameters::Instance()->EnableNeutronGeneralProcess()));

  runManager->SetUserInitialization(new ActionInitialization());

  G4UImanager* UImanager = G4UImanager::GetUIpointer();
  UImanager->ApplyCommand(G4String("/control/execute ") + argv[1]);

  delete runManager;
  return 0;
}
