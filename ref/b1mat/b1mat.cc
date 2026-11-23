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
#include "EventAction.hh"
#include "PrimaryGeneratorAction.hh"
#include "RunAction.hh"
#include "SteppingAction.hh"

#include "G4Box.hh"
#include "G4Exception.hh"
#include "G4LogicalVolumeStore.hh"
#include "G4ParticleGun.hh"
#include "G4SystemOfUnits.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4VUserActionInitialization.hh"
#include "Randomize.hh"
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
#include <iomanip>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "G4EmCalculator.hh"
#include "G4Event.hh"
#include "G4EventManager.hh"
#include "G4ParticleTable.hh"
#include "G4ProductionCutsTable.hh"
#include "G4Run.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4UIcmdWithoutParameter.hh"
#include "G4UserTrackingAction.hh"
#include "G4VProcess.hh"

namespace {

// ---- /B1/tape/on (P46, docs/RISK.md V346): the per-process deposit tape.
//
// WHY. P27 found the muon 0.6% low at 10 GeV in every material and for both charges (V337,
// V338). B1 scores one number per event; which process the missing energy belongs to is not in
// it. The tape splits the scoring trapezoid's deposit by the FIRST-GENERATION process the
// depositing track descends from - the primary's own steps are "primary", a delta ray and its
// whole shower are "muIoni", a bremsstrahlung photon's shower is "muBrems", and so on - and
// counts the primary's own discrete interactions anywhere in the world. It is an observer:
// B1's stepping action still runs first, unchanged, so the dose B1 prints is the dose B1
// prints without the tape (it draws no random number). Off unless the command is given, which
// keeps every P27 row's executable exactly what V336 measured.
G4bool gTape = false;
std::mutex gTapeMutex;
struct TapeSum {
  double sum = 0, sum2 = 0;  // per-event deposit in the trapezoid: sum and sum of squares
};
std::map<std::string, TapeSum> gTapeEdep;
std::map<std::string, double> gTapeFirstGenN, gTapeFirstGenE, gTapePrimaryHits;
long long gTapeEvents = 0;

struct TapeThread {
  G4int event = -1;
  std::unordered_map<G4int, std::string> root;
  std::map<std::string, double> ev;  // this event's deposit per root process
  std::map<std::string, TapeSum> edep;
  std::map<std::string, double> fgn, fge, hits;
  long long events = 0;
  G4LogicalVolume* shape2 = nullptr;

  void flush_event() {
    if (event < 0) { return; }
    for (auto& kv : ev) {
      edep[kv.first].sum += kv.second;
      edep[kv.first].sum2 += kv.second * kv.second;
    }
    ev.clear();
    ++events;
  }
  void merge() {
    flush_event();
    event = -1;
    std::lock_guard<std::mutex> lock(gTapeMutex);
    for (auto& kv : edep) {
      gTapeEdep[kv.first].sum += kv.second.sum;
      gTapeEdep[kv.first].sum2 += kv.second.sum2;
    }
    for (auto& kv : fgn) { gTapeFirstGenN[kv.first] += kv.second; }
    for (auto& kv : fge) { gTapeFirstGenE[kv.first] += kv.second; }
    for (auto& kv : hits) { gTapePrimaryHits[kv.first] += kv.second; }
    gTapeEvents += events;
    edep.clear();
    fgn.clear();
    fge.clear();
    hits.clear();
    events = 0;
  }
};
G4ThreadLocal TapeThread* tTape = nullptr;
TapeThread& tape() {
  if (tTape == nullptr) { tTape = new TapeThread; }
  return *tTape;
}

class TapeTrackingAction : public G4UserTrackingAction {
 public:
  void PreUserTrackingAction(const G4Track* t) override {
    TapeThread& tp = tape();
    const G4int ev = G4EventManager::GetEventManager()->GetConstCurrentEvent()->GetEventID();
    if (ev != tp.event) {
      tp.flush_event();
      tp.event = ev;
      tp.root.clear();
    }
    std::string r;
    if (t->GetParentID() == 0) {
      r = "primary";
    } else if (t->GetParentID() == 1) {
      r = (t->GetCreatorProcess() != nullptr) ? std::string(t->GetCreatorProcess()->GetProcessName())
                                              : std::string("none");
      tp.fgn[r] += 1;
      tp.fge[r] += t->GetKineticEnergy();
    } else {
      auto it = tp.root.find(t->GetParentID());
      r = (it != tp.root.end()) ? it->second : std::string("orphan");
    }
    tp.root[t->GetTrackID()] = r;
  }
};

class TapeSteppingAction : public B1::SteppingAction {
 public:
  using B1::SteppingAction::SteppingAction;
  void UserSteppingAction(const G4Step* step) override {
    B1::SteppingAction::UserSteppingAction(step);
    TapeThread& tp = tape();
    const G4Track* t = step->GetTrack();
    if (t->GetTrackID() == 1) {
      const G4VProcess* pr = step->GetPostStepPoint()->GetProcessDefinedStep();
      if (pr != nullptr) { tp.hits[std::string(pr->GetProcessName())] += 1; }
    }
    if (tp.shape2 == nullptr) {
      tp.shape2 = G4LogicalVolumeStore::GetInstance()->GetVolume("Shape2");
    }
    if (step->GetPreStepPoint()->GetTouchableHandle()->GetVolume()->GetLogicalVolume()
        != tp.shape2) {
      return;
    }
    const G4double e = step->GetTotalEnergyDeposit();
    if (e == 0.) { return; }
    auto it = tp.root.find(t->GetTrackID());
    tp.ev[(it != tp.root.end()) ? it->second : std::string("untracked")] += e;
  }
};

// Each worker's run action merges its thread's tape at the end of its run. The master's run
// action is B1's own: G4MTRunManager calls BuildForMaster when the action initialisation is SET,
// before any macro command, so the master never sees /B1/tape/on - main prints the merged tape
// after the macro instead (PrintTape). Serial mode has one run action, which merges here.
class TapeRunAction : public B1::RunAction {
 public:
  void EndOfRunAction(const G4Run* run) override {
    B1::RunAction::EndOfRunAction(run);
    if (!IsMaster() || !G4Threading::IsMultithreadedApplication()) { tape().merge(); }
  }
};

void PrintTape() {
  {
    std::lock_guard<std::mutex> lock(gTapeMutex);
    const double n = static_cast<double>(gTapeEvents);
    G4cout << "\nTAPE events " << gTapeEvents << "\n"
           << "TAPE edep,root,MeV_per_event,sigma_MeV_per_event\n";
    double tot = 0;
    for (auto& kv : gTapeEdep) {
      const double m = kv.second.sum / n;
      const double var = (kv.second.sum2 / n - m * m) / n;
      tot += m;
      G4cout << "TAPE edep," << kv.first << "," << std::setprecision(9) << m / MeV << ","
             << std::sqrt(var > 0 ? var : 0) / MeV << "\n";
    }
    G4cout << "TAPE edep,TOTAL," << tot / MeV << ",\n";
    G4cout << "TAPE firstgen,process,per_event,MeV_per_event_at_birth\n";
    for (auto& kv : gTapeFirstGenN) {
      G4cout << "TAPE firstgen," << kv.first << "," << kv.second / n << ","
             << gTapeFirstGenE[kv.first] / n / MeV << "\n";
    }
    G4cout << "TAPE primary_step_end,process,per_event\n";
    for (auto& kv : gTapePrimaryHits) {
      G4cout << "TAPE primary_step_end," << kv.first << "," << kv.second / n << "\n";
    }
    G4cout << G4endl;
    gTapeEdep.clear();
    gTapeFirstGenN.clear();
    gTapeFirstGenE.clear();
    gTapePrimaryHits.clear();
    gTapeEvents = 0;
  }
}

// /B1/tape/tables: G4EmCalculator's answers for mu- and mu+ in every material with a couple, at
// P46's five energies - the restricted dE/dx the transport reads (GetDEDX, every energy-loss
// process summed), each process's restricted dE/dx at the couple's own cut (ComputeDEDX), the
// range, and each discrete process's cross section from its table (GetCrossSectionPerVolume) and
// from the model (ComputeCrossSectionPerVolume). After a /run/beamOn, so the tables exist.
void PrintMuonTables() {
  G4EmCalculator calc;
  const G4ProductionCutsTable* ct = G4ProductionCutsTable::GetProductionCutsTable();
  const char* parts[2] = {"mu-", "mu+"};
  const double es[] = {100., 300., 1000., 3000., 10000.};
  const char* procs[] = {"muIoni", "muBrems", "muPairProd", "muonNuclear", "CoulombScat"};
  G4cout << "TABLE particle,material,E_MeV,ecut_MeV,gcut_MeV,dedx_total,range_mm,"
            "dedx_muIoni,dedx_muBrems,dedx_muPairProd,"
            "xs_tab_muIoni,xs_tab_muBrems,xs_tab_muPairProd,xs_tab_muonNuclear,"
            "xs_tab_CoulombScat,xs_mod_muIoni,xs_mod_muBrems,xs_mod_muPairProd,"
            "xs_mod_muonNuclear\n";
  for (const char* pn : parts) {
    const G4ParticleDefinition* p = G4ParticleTable::GetParticleTable()->FindParticle(pn);
    for (std::size_t i = 0; i < ct->GetTableSize(); ++i) {
      const G4MaterialCutsCouple* c = ct->GetMaterialCutsCouple(static_cast<G4int>(i));
      const G4Material* m = c->GetMaterial();
      const double ecut = (*ct->GetEnergyCutsVector(1))[c->GetIndex()];
      const double gcut = (*ct->GetEnergyCutsVector(0))[c->GetIndex()];
      for (double e : es) {
        G4cout << std::setprecision(10) << "TABLE " << pn << "," << m->GetName() << "," << e
               << "," << ecut << "," << gcut << "," << calc.GetDEDX(e, p, m) * mm / MeV << ","
               << calc.GetRangeFromRestricteDEDX(e, p, m) / mm;
        G4cout << "," << calc.ComputeDEDX(e, p, "muIoni", m, ecut) * mm / MeV << ","
               << calc.ComputeDEDX(e, p, "muBrems", m, gcut) * mm / MeV << ","
               << calc.ComputeDEDX(e, p, "muPairProd", m, ecut) * mm / MeV;
        for (const char* pr : procs) {
          G4cout << "," << calc.GetCrossSectionPerVolume(e, p, pr, m) * mm;
        }
        G4cout << "," << calc.ComputeCrossSectionPerVolume(e, p, "muIoni", m, ecut) * mm << ","
               << calc.ComputeCrossSectionPerVolume(e, p, "muBrems", m, gcut) * mm << ","
               << calc.ComputeCrossSectionPerVolume(e, p, "muPairProd", m, ecut) * mm << ","
               << calc.ComputeCrossSectionPerVolume(e, p, "muonNuclear", m, ecut) * mm << "\n";
      }
    }
  }
  G4cout << G4endl;
}

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

// /B1/gun/z (P27's tier 4, V336): the plane B1's gun fires from. Set on the master before the
// workers' first event and only read after, so a plain global is enough; unset, the generator is
// B1's own GeneratePrimaries, call for call.
G4bool gHasGunZ = false;
G4double gGunZ = 0.;

// B1::PrimaryGeneratorAction with z0 replaced when the command was given: the same envelope
// lookup, the same two G4UniformRand draws in the same order for x0 and y0, the same gun.
class GunZPrimaryGeneratorAction : public B1::PrimaryGeneratorAction {
 public:
  void GeneratePrimaries(G4Event* event) override {
    if (!gHasGunZ) {
      B1::PrimaryGeneratorAction::GeneratePrimaries(event);
      return;
    }
    if (fEnvelope == nullptr) {
      G4LogicalVolume* lv = G4LogicalVolumeStore::GetInstance()->GetVolume("Envelope");
      if (lv != nullptr) { fEnvelope = dynamic_cast<G4Box*>(lv->GetSolid()); }
    }
    const G4double envSizeXY = (fEnvelope != nullptr) ? fEnvelope->GetXHalfLength() * 2. : 0.;
    const G4double size = 0.8;
    const G4double x0 = size * envSizeXY * (G4UniformRand() - 0.5);
    const G4double y0 = size * envSizeXY * (G4UniformRand() - 0.5);
    auto* gun = const_cast<G4ParticleGun*>(GetParticleGun());
    gun->SetParticlePosition(G4ThreeVector(x0, y0, gGunZ));
    gun->GeneratePrimaryVertex(event);
  }

 private:
  G4Box* fEnvelope = nullptr;
};

// B1::ActionInitialization's two methods with the generator above in Build().
class GunZActionInitialization : public G4VUserActionInitialization {
 public:
  void BuildForMaster() const override {
    SetUserAction(gTape ? new TapeRunAction : new B1::RunAction);
  }
  void Build() const override {
    SetUserAction(new GunZPrimaryGeneratorAction);
    auto* runAction = gTape ? new TapeRunAction : new B1::RunAction;
    SetUserAction(runAction);
    auto* eventAction = new B1::EventAction(runAction);
    SetUserAction(eventAction);
    if (gTape) {
      // P46's tape: B1's stepping action runs first inside TapeSteppingAction, unchanged.
      SetUserAction(new TapeSteppingAction(eventAction));
      SetUserAction(new TapeTrackingAction);
    } else {
      SetUserAction(new B1::SteppingAction(eventAction));
    }
  }
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
    fGunDir = new G4UIdirectory("/B1/gun/");
    fGunDir->SetGuidance("Example B1's gun (P27's tier 4).");
    fGunZ = new G4UIcmdWithADoubleAndUnit("/B1/gun/z", this);
    fGunZ->SetGuidance("The z plane the gun fires from (world coordinates); x-y and direction B1's.");
    fGunZ->SetParameterName("z", false);
    fGunZ->SetUnitCategory("Length");
    fGunZ->SetDefaultUnit("mm");
    fGunZ->AvailableForStates(G4State_PreInit, G4State_Idle);
    fGunZ->SetToBeBroadcasted(false);
    fTapeDir = new G4UIdirectory("/B1/tape/");
    fTapeDir->SetGuidance("P46's per-process deposit tape (docs/RISK.md V346); off by default.");
    fTapeOn = new G4UIcmdWithoutParameter("/B1/tape/on", this);
    fTapeOn->SetGuidance("Split Shape2's deposit by first-generation process; before init.");
    fTapeOn->AvailableForStates(G4State_PreInit);
    fTapeOn->SetToBeBroadcasted(false);
    fTables = new G4UIcmdWithoutParameter("/B1/tape/tables", this);
    fTables->SetGuidance("Print G4EmCalculator's muon tables (after a /run/beamOn).");
    fTables->AvailableForStates(G4State_Idle);
    fTables->SetToBeBroadcasted(false);
  }
  ~MaterialMessenger() override {
    delete fTables;
    delete fTapeOn;
    delete fTapeDir;
    delete fGunZ;
    delete fGunDir;
    delete fCmd;
    delete fDir;
  }
  void SetNewValue(G4UIcommand* cmd, G4String value) override {
    if (cmd == fCmd) { fDet->SetShape2Material(value); }
    if (cmd == fGunZ) {
      gGunZ = fGunZ->GetNewDoubleValue(value);
      gHasGunZ = true;
      G4cout << "/B1/gun/z: the gun fires from z = " << gGunZ / mm << " mm" << G4endl;
    }
    if (cmd == fTapeOn) { gTape = true; }
    if (cmd == fTables) { PrintMuonTables(); }
  }

 private:
  MaterialDetectorConstruction* fDet;
  G4UIdirectory* fDir = nullptr;
  G4UIcmdWithAString* fCmd = nullptr;
  G4UIdirectory* fGunDir = nullptr;
  G4UIcmdWithADoubleAndUnit* fGunZ = nullptr;
  G4UIdirectory* fTapeDir = nullptr;
  G4UIcmdWithoutParameter* fTapeOn = nullptr;
  G4UIcmdWithoutParameter* fTables = nullptr;
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
  // B1's ActionInitialization's code, with GunZPrimaryGeneratorAction - which IS B1's generator
  // until /B1/gun/z is given (V336's proof covers the default).
  runManager->SetUserInitialization(new GunZActionInitialization());
  G4UImanager::GetUIpointer()->ApplyCommand(G4String("/control/execute ") + argv[1]);
  if (gTape) { PrintTape(); }
  delete messenger;
  delete runManager;
  return 0;
}
