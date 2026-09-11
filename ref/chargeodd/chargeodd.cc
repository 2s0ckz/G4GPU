// Geant4 measured against itself: where does a negative hadron's missing dose go?
//
// docs/RISK.md V44 records the observation and what it excludes. Example B1, 200 MeV,
// 500,000 events a side: Geant4's own pi+ and pi- doses in the scored bone differ by 5.43%,
// mu+/mu- by 4.36% and K+/K- by 9.62%, while the restricted dE/dx table the same install
// dumps has each charge pair 0.023% apart in water and 0.233% apart in bone. Multiple
// scattering, the at-rest and hadronic processes, decay, a Bragg peak and the range table are
// all excluded there by measurement. B1's dose is one number per run, so it cannot say WHICH
// of the terms behind it moved. That is all this program does.
//
// The step limitation, the continuous loss, the delta-ray channel and the secondaries'
// transport all feed one scalar in B1. Here every one of them is tallied separately for the
// same beam, so the 5% lands on exactly one of:
//
//   * the primary's own energy loss across the slab   -> ke_in - ke_out
//   * what the primary deposits locally               -> edep_primary
//   * what its secondaries deposit inside the slab    -> edep_secondary
//   * what its secondaries carry out of the slab      -> esec_created - edep_secondary
//   * how far the primary travels inside the slab     -> len_primary
//   * how many primaries arrive at all                -> n_entered
//
// If ke_in - ke_out splits by 5% the mechanism is in the loss; if it does not and edep does,
// the energy is leaving the slab. Geometry is a water block with one slab of a second
// material in it, not B1: the point is to reproduce the SPLIT, and if the split is a property
// of the transport it does not need B1's trapezoid to appear.
//
// Usage: chargeodd.exe <particle> <energy_MeV> <events> [cut_mm] [target] [seed] [mode]
//
// One species per invocation on purpose. Two beams in one process share a random sequence, so
// the second one starts from wherever the first ended - V44 records three diagnostics where
// that cost a sigma of interpretation. Every run here starts from the same seed.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "G4Box.hh"
#include "G4EmParameters.hh"
#include "G4Event.hh"
#include "G4LogicalVolume.hh"
#include "G4NistManager.hh"
#include "G4PVPlacement.hh"
#include "G4ParticleGun.hh"
#include "G4ParticleTable.hh"
#include "G4PhysicalConstants.hh"
#include "G4RunManagerFactory.hh"
#include "G4Step.hh"
#include "G4StepPoint.hh"
#include "G4StepStatus.hh"
#include "G4SystemOfUnits.hh"
#include "G4Track.hh"
#include "G4UImanager.hh"
#include "G4UserSteppingAction.hh"
#include "G4VPhysicalVolume.hh"
#include "G4VUserActionInitialization.hh"
#include "G4VUserDetectorConstruction.hh"
#include "G4VUserPrimaryGeneratorAction.hh"
#include "QBBC.hh"
#include "Randomize.hh"

namespace {

// ------------------------------------------------------------------ configuration
struct Cfg {
  std::string particle = "pi-";
  double energy = 200.0;         // MeV
  long events = 20000;
  double cut = 0.7;              // mm, the production range cut, on every species at once
  std::string target = "G4_BONE_COMPACT_ICRU";
  long seed = 12345;
  // "emonly" is the seven inactivations ref/b1hadron/pion_minus_emonly.mac applies - every
  // process QBBC gives a charged hadron that this port does not have. "nomsc" adds msc, which
  // is pion_nomsc.mac. "full" inactivates nothing. The V44 table was measured at "emonly", so
  // that is the default here: a diagnostic run against a different process list than the
  // observation it is explaining is not a diagnostic.
  std::string mode = "emonly";
  // Two knobs that pick which branch of G4VEnergyLossProcess::AlongStepDoIt computes the
  // continuous loss, and therefore which TABLE the transport reads:
  //
  //   eloss = length*GetDEDXForScaledEnergy(E)                        // "short step"
  //   if(eloss > preStepKinEnergy*linLossLimit) {                     // "long step"
  //     eloss = preStepKinEnergy - ScaledKinEnergyForLoss((fRange-length)/reduceFactor)...
  //
  // The first reads theDEDXTable; the second reads theRangeTableForLoss and
  // theInverseRangeTable. linLossLimit defaults to 0.01, and a 200 MeV pion in bone takes
  // ~20 mm steps losing ~5% of its energy, so the long branch fires on every step. Setting
  // linLossLimit to 1.0 disables the long branch outright; shrinking dRoverRange makes the
  // steps short enough that it would not fire anyway. Either isolates one table.
  double lin_loss_limit = -1.0;   // < 0: leave G4EmParameters alone
  double drover_range = -1.0;     // < 0: leave the mu/hadron step function alone
};
Cfg g_cfg;

// The slab the tally is over, and the water in front of it. 200 mm of water upstream is
// roughly B1's path from the world edge to the scored trapezoid; 60 mm of target is its
// thickness along the beam. Neither number is critical - what is measured is a ratio between
// two charges through the same geometry.
constexpr double kUpstream = 200.0;   // mm of water before the slab
constexpr double kSlab = 60.0;        // mm of target material

G4LogicalVolume* g_target_lv = nullptr;

// ------------------------------------------------------------------ the tally
//
// Doubles, summed over events. Each is per-beam, so a ratio between two runs is the
// measurement; absolute values are only used to check that the terms add up.
struct Tally {
  double edep_primary = 0;      // deposit by track 1 inside the slab (MeV)
  double edep_secondary = 0;    // deposit by everything else inside the slab
  double ke_in = 0;             // primary kinetic energy entering the slab
  double ke_out = 0;            // ... and leaving it (0 if it stopped inside)
  double len_primary = 0;       // primary path length inside the slab (mm)
  double esec_created = 0;      // initial kinetic energy of secondaries born in the slab
  double edep_world = 0;        // deposit everywhere, by anything
  long n_entered = 0;           // events whose primary entered the slab
  long n_exited = 0;            // ... and left it again
  long n_steps_primary = 0;     // primary steps inside the slab
  long n_sec_created = 0;       // secondaries born inside the slab
};
Tally g_t;

class Tallier : public G4UserSteppingAction {
 public:
  void UserSteppingAction(const G4Step* step) override {
    const G4Track* tr = step->GetTrack();
    const G4double e = step->GetTotalEnergyDeposit();
    g_t.edep_world += e / MeV;

    // The volume the step STARTED in: a step that ends on the slab boundary belongs to the
    // volume it crossed, which is the pre-step one. G4Step::GetPreStepPoint's touchable is
    // what proton_depth.cc bins on for the same reason.
    const auto* touch = step->GetPreStepPoint()->GetTouchableHandle()->GetVolume();
    if (nullptr == touch || touch->GetLogicalVolume() != g_target_lv) { return; }

    const bool primary = (tr->GetTrackID() == 1);
    if (primary) {
      g_t.edep_primary += e / MeV;
      g_t.len_primary += step->GetStepLength() / mm;
      ++g_t.n_steps_primary;
      // Entry and exit are read off the step points rather than counted from a flag, so a
      // primary that re-enters the slab after leaving it is not double counted: the first
      // pre-step point inside the slab is the entry and the last post-step point is the exit.
      if (step->GetPreStepPoint()->GetStepStatus() == fGeomBoundary ||
          step->GetPreStepPoint()->GetStepStatus() == fUndefined) {
        if (!entered_) {
          entered_ = true;
          ++g_t.n_entered;
          g_t.ke_in += step->GetPreStepPoint()->GetKineticEnergy() / MeV;
        }
      }
      ke_last_ = step->GetPostStepPoint()->GetKineticEnergy() / MeV;
      exited_ = (step->GetPostStepPoint()->GetStepStatus() == fGeomBoundary);
    } else {
      g_t.edep_secondary += e / MeV;
    }

    // Secondaries created in this step, whatever made them. fSecondary is the step's own
    // list, so this counts each secondary once, at its birth, and reads its birth energy
    // before any transport has touched it.
    const auto* sec = step->GetSecondaryInCurrentStep();
    if (nullptr != sec) {
      for (const auto* s : *sec) {
        g_t.esec_created += s->GetKineticEnergy() / MeV;
        ++g_t.n_sec_created;
      }
    }
  }

  // The primary's exit energy is only known once its last step in the slab has been taken,
  // and a stepping action cannot tell that a step is the last one. So the per-event state is
  // closed out from the NEXT event's primary generator and once more after the run - which is
  // why ke_out is accumulated here rather than in UserSteppingAction.
  void Flush() {
    if (entered_) {
      if (exited_) { ++g_t.n_exited; }
      g_t.ke_out += ke_last_;
    }
    entered_ = exited_ = false;
    ke_last_ = 0.0;
  }
  bool Entered() const { return entered_; }

 private:
  bool entered_ = false;
  bool exited_ = false;
  double ke_last_ = 0.0;
};
Tallier* g_tallier = nullptr;

// ------------------------------------------------------------------ geometry
class Slab : public G4VUserDetectorConstruction {
 public:
  G4VPhysicalVolume* Construct() override {
    auto* nist = G4NistManager::Instance();
    auto* water = nist->FindOrBuildMaterial("G4_WATER");
    auto* tgt = nist->FindOrBuildMaterial(g_cfg.target);
    if (nullptr == tgt) {
      std::fprintf(stderr, "chargeodd: no NIST material '%s'\n", g_cfg.target.c_str());
      std::exit(2);
    }
    // The world is water and 600 mm deep so the beam never leaves it before the slab: a
    // charged hadron scattering out the side would be a difference between the two charges
    // that has nothing to do with energy loss.
    const double halfXY = 300.0 * mm;
    const double halfZ = 300.0 * mm;
    auto* ws = new G4Box("World", halfXY, halfXY, halfZ);
    auto* wl = new G4LogicalVolume(ws, water, "World");
    auto* wp = new G4PVPlacement(nullptr, {}, wl, "World", nullptr, false, 0);

    auto* ts = new G4Box("Target", 200.0 * mm, 200.0 * mm, 0.5 * kSlab * mm);
    g_target_lv = new G4LogicalVolume(ts, tgt, "Target");
    // Beam starts at -halfZ; the slab's front face sits kUpstream downstream of it.
    const double zc = -halfZ + (kUpstream + 0.5 * kSlab) * mm;
    new G4PVPlacement(nullptr, {0, 0, zc}, g_target_lv, "Target", wl, false, 0);
    return wp;
  }
};

class Gun : public G4VUserPrimaryGeneratorAction {
 public:
  Gun() : gun_(new G4ParticleGun(1)) {
    auto* def = G4ParticleTable::GetParticleTable()->FindParticle(g_cfg.particle);
    if (nullptr == def) {
      std::fprintf(stderr, "chargeodd: no particle '%s'\n", g_cfg.particle.c_str());
      std::exit(2);
    }
    gun_->SetParticleDefinition(def);
    gun_->SetParticleEnergy(g_cfg.energy * MeV);
    gun_->SetParticleMomentumDirection({0, 0, 1});
    gun_->SetParticlePosition({0, 0, -300.0 * mm + 1.0e-3 * mm});
  }
  ~Gun() override { delete gun_; }
  void GeneratePrimaries(G4Event* e) override {
    if (nullptr != g_tallier) { g_tallier->Flush(); }
    gun_->GeneratePrimaryVertex(e);
  }

 private:
  G4ParticleGun* gun_;
};

class Actions : public G4VUserActionInitialization {
 public:
  void Build() const override {
    SetUserAction(new Gun());
    g_tallier = new Tallier();
    SetUserAction(g_tallier);
  }
};

}  // namespace

int main(int argc, char** argv) {
  if (argc > 1) { g_cfg.particle = argv[1]; }
  if (argc > 2) { g_cfg.energy = std::atof(argv[2]); }
  if (argc > 3) { g_cfg.events = std::atol(argv[3]); }
  if (argc > 4) { g_cfg.cut = std::atof(argv[4]); }
  if (argc > 5) { g_cfg.target = argv[5]; }
  if (argc > 6) { g_cfg.seed = std::atol(argv[6]); }
  if (argc > 7) { g_cfg.mode = argv[7]; }
  if (argc > 8) { g_cfg.lin_loss_limit = std::atof(argv[8]); }
  if (argc > 9) { g_cfg.drover_range = std::atof(argv[9]); }

  auto* rm = G4RunManagerFactory::CreateRunManager(G4RunManagerType::Serial);
  rm->SetUserInitialization(new Slab());
  auto* phys = new QBBC(0);
  phys->SetDefaultCutValue(g_cfg.cut * mm);
  rm->SetUserInitialization(phys);
  rm->SetUserInitialization(new Actions());

  // Set through the API and BEFORE Initialize(), not through /process/eLoss/... afterwards.
  // G4EmParameters::SetLinearLossLimit and SetStepFunctionMuHad both begin `if(IsLocked())
  // return;`, and the messenger route silently does nothing outside PreInit/Init/Idle - the
  // first attempt at this diagnostic issued the commands after Initialize() and got numbers
  // BIT-IDENTICAL to the default run, which is the only reason the rejection was noticed.
  // G4EmParameters exists as soon as the physics list constructor has run.
  {
    G4EmParameters* p = G4EmParameters::Instance();
    if (g_cfg.lin_loss_limit > 0.0) { p->SetLinearLossLimit(g_cfg.lin_loss_limit); }
    if (g_cfg.drover_range > 0.0) { p->SetStepFunctionMuHad(g_cfg.drover_range, 0.01 * mm); }
    std::printf("linLossLimit=%.9g  dRoverRange(MuHad)=%.9g\n", p->LinearLossLimit(),
                g_cfg.drover_range);
  }
  rm->Initialize();

  // The same UI commands ref/b1hadron/*_emonly.mac applies, issued here rather than through a
  // macro so that the process list is part of the program and cannot drift from the tally.
  // The inelastic process is named after the particle, so it is built from the beam species.
  if (g_cfg.mode != "full") {
    G4UImanager* ui = G4UImanager::GetUIpointer();
    const char* names[] = {"hadElastic", "Decay", "hBertiniCaptureAtRest",
                           "hFritiofCaptureAtRest", "muMinusCaptureAtRest",
                           "hBrems", "hPairProd", "CoulombScat"};
    for (const char* n : names) { ui->ApplyCommand(std::string("/process/inactivate ") + n); }
    ui->ApplyCommand("/process/inactivate " + g_cfg.particle + "Inelastic");
    ui->ApplyCommand("/process/inactivate anti_protonInelastic");
    if (g_cfg.mode == "nomsc") { ui->ApplyCommand("/process/inactivate msc"); }
    // What actually took effect, printed once, because V44's own exclusions rest on this dump
    // and not on a command name being right.
    ui->ApplyCommand("/particle/select " + g_cfg.particle);
    ui->ApplyCommand("/particle/process/dump");
  }
  G4Random::setTheSeed(g_cfg.seed);
  rm->BeamOn(g_cfg.events);
  if (nullptr != g_tallier) { g_tallier->Flush(); }

  const double n = double(g_cfg.events);
  const Tally& t = g_t;
  // One CSV row per invocation, appended by run.bat, so two charges are two lines that can be
  // divided. Every column is a sum over events divided by the event count.
  std::printf("CSV,%s,%.9g,%ld,%.9g,%s,%ld,%s", g_cfg.particle.c_str(), g_cfg.energy,
              g_cfg.events, g_cfg.cut, g_cfg.target.c_str(), g_cfg.seed, g_cfg.mode.c_str());
  std::printf(",%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g\n",
              t.edep_primary / n, t.edep_secondary / n, (t.edep_primary + t.edep_secondary) / n,
              t.ke_in / n, t.ke_out / n, (t.ke_in - t.ke_out) / n, t.len_primary / n,
              t.esec_created / n, t.edep_world / n, double(t.n_entered) / n,
              double(t.n_steps_primary) / n);
  std::printf("particle=%s  E=%g MeV  N=%ld  cut=%g mm  target=%s\n", g_cfg.particle.c_str(),
              g_cfg.energy, g_cfg.events, g_cfg.cut, g_cfg.target.c_str());
  std::printf("  entered/exited slab      %ld / %ld\n", t.n_entered, t.n_exited);
  std::printf("  primary KE in  (MeV)     %.6f\n", t.ke_in / n);
  std::printf("  primary KE out (MeV)     %.6f\n", t.ke_out / n);
  std::printf("  primary loss   (MeV)     %.6f\n", (t.ke_in - t.ke_out) / n);
  std::printf("  primary path   (mm)      %.6f\n", t.len_primary / n);
  std::printf("  primary steps            %.4f\n", double(t.n_steps_primary) / n);
  std::printf("  <dE/dx> loss/path        %.6f MeV/mm\n",
              (t.ke_in - t.ke_out) / (t.len_primary > 0 ? t.len_primary : 1));
  std::printf("  edep primary   (MeV)     %.6f\n", t.edep_primary / n);
  std::printf("  edep secondary (MeV)     %.6f\n", t.edep_secondary / n);
  std::printf("  edep in slab   (MeV)     %.6f\n", (t.edep_primary + t.edep_secondary) / n);
  std::printf("  sec born in slab (MeV)   %.6f  (%ld tracks)\n", t.esec_created / n,
              t.n_sec_created);
  std::printf("  edep whole world (MeV)   %.6f\n", t.edep_world / n);
  delete rm;
  return 0;
}
