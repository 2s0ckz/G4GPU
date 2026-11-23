// Example B1 with ONE macro command added - /B1/det/material <NIST name> - for P27's acceptance
// sweep (docs/RISK.md V336): the Geant4 11.1.1 side of every row, QBBC as it ships.
//
// WHY A BUILD OF ITS OWN. ref/B1build, the Geant4 B1 that ref/run/runb1.bat runs, compiles the
// files of the Geant4 SOURCE TREE's examples/basic/B1 - the user's install, which nothing in this
// repository edits. So the command is added here the way ref/b1ion adds its counters: Geant4's own
// B1 sources compiled unchanged, and what is ours is this file. The sweep needs the scoring
// trapezoid (Shape2) in six materials; Geant4's B1 hard-codes G4_BONE_COMPACT_ICRU, and the port's
// examples/B1 got the same command in its DetectorConstruction.
//
// HOW. `MaterialDetectorConstruction` IS Geant4's B1::DetectorConstruction: its Construct() calls
// B1's, then - only if the command named a material - sets that material on the logical volume B1
// made the scoring volume (`fScoringVolume`, protected in B1's header for exactly this kind of
// derived class). Everything else B1 builds, places and scores is B1's own code. The command is a
// G4UIcmdWithAString available in G4State_PreInit only, so it must come before /run/initialize, as
// on the port's side; it is not broadcast, because only the master constructs the geometry (the
// workers share the master's logical volumes).
//
// WITH THE COMMAND UNSET THIS IS GEANT4'S exampleB1 TO THE DIGIT: the main below is
// exampleB1.cc's batch path - G4RunManagerFactory's default run manager (so G4FORCE_RUN_MANAGER_
// TYPE and G4FORCENUMBEROFTHREADS act as they do on ref/B1build), QBBC at verbose 1, B1's own
// ActionInitialization - less the visualisation manager, which draws no random number. V336
// records the shipped macro's dose from both executables.
//
// One difference the command does make, and it is not physics: B1's Construct() still builds
// G4_BONE_COMPACT_ICRU before this class replaces it, so the material table carries bone (and
// its elements Mg, P, S, Ca) on every run. A material no volume uses has no couple, no table and
// no step; Geant4's element-wise cross-section loaders may read its elements' data at
// initialisation, which costs start-up time and moves no dose.
//
// Usage:  run.bat <macro>      (MT as the install ships it; the caller sets the environment)
#include "ActionInitialization.hh"
#include "DetectorConstruction.hh"

#include "G4Exception.hh"
#include "G4LogicalVolume.hh"
#include "G4Material.hh"
#include "G4NistManager.hh"
#include "G4RunManagerFactory.hh"
#include "G4SteppingVerbose.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIdirectory.hh"
#include "G4UImanager.hh"
#include "G4UImessenger.hh"
#include "G4ios.hh"
#include "QBBC.hh"

#include <cstdio>

namespace {

class MaterialDetectorConstruction : public B1::DetectorConstruction {
 public:
  void SetShape2Material(const G4String& name) { fShape2Material = name; }

  G4VPhysicalVolume* Construct() override {
    G4VPhysicalVolume* world = B1::DetectorConstruction::Construct();
    if (!fShape2Material.empty()) {
      G4Material* mat = G4NistManager::Instance()->FindOrBuildMaterial(fShape2Material);
      if (mat == nullptr) {
        G4ExceptionDescription ed;
        ed << "/B1/det/material " << fShape2Material << ": no such NIST material";
        G4Exception("MaterialDetectorConstruction::Construct", "B1MAT001", FatalException, ed);
      }
      fScoringVolume->SetMaterial(mat);
      G4cout << "/B1/det/material: Shape2 (the scoring volume) is " << mat->GetName() << G4endl;
    }
    return world;
  }

 private:
  G4String fShape2Material;  // empty: Shape2 stays B1's G4_BONE_COMPACT_ICRU
};

class MaterialMessenger : public G4UImessenger {
 public:
  explicit MaterialMessenger(MaterialDetectorConstruction* det) : fDet(det) {
    fDir = new G4UIdirectory("/B1/det/");
    fDir->SetGuidance("Example B1's detector (P27's one added command).");
    fCmd = new G4UIcmdWithAString("/B1/det/material", this);
    fCmd->SetGuidance("The scoring trapezoid's (Shape2's) NIST material, before /run/initialize.");
    fCmd->SetParameterName("material", false);
    fCmd->AvailableForStates(G4State_PreInit);
    fCmd->SetToBeBroadcasted(false);
  }
  ~MaterialMessenger() override {
    delete fCmd;
    delete fDir;
  }
  void SetNewValue(G4UIcommand* cmd, G4String value) override {
    if (cmd == fCmd) { fDet->SetShape2Material(value); }
  }

 private:
  MaterialDetectorConstruction* fDet;
  G4UIdirectory* fDir = nullptr;
  G4UIcmdWithAString* fCmd = nullptr;
};

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::printf("usage: b1mat.exe <macro>\n");
    return 1;
  }
  G4int precision = 4;
  G4SteppingVerbose::UseBestUnit(precision);
  auto* runManager = G4RunManagerFactory::CreateRunManager(G4RunManagerType::Default);
  auto* detector = new MaterialDetectorConstruction();
  // Deleted before the run manager, whose kernel takes the UI manager the command lives in.
  auto* messenger = new MaterialMessenger(detector);
  runManager->SetUserInitialization(detector);
  G4VModularPhysicsList* physicsList = new QBBC;
  physicsList->SetVerboseLevel(1);
  runManager->SetUserInitialization(physicsList);
  runManager->SetUserInitialization(new B1::ActionInitialization());
  G4UImanager::GetUIpointer()->ApplyCommand(G4String("/control/execute ") + argv[1]);
  delete messenger;
  delete runManager;
  return 0;
}
