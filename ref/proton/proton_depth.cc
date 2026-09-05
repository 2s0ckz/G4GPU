// Proton depth-dose in water: the transport-level check for the hadron stepper.
//
// **One source file, two builds.** This compiles against the real Geant4 11.1.1 install and
// against this port's Geant4-shaped headers, and both write depth_dose.csv in the same format.
// That is the whole point: every other transport comparison in this repository (docs/RESULT.md)
// runs *different* code on the two sides and trusts that the two descriptions of the geometry
// and the beam agree. Here they cannot disagree, because there is one description.
//
// What it measures is the thing a proton calculation exists to produce: where the Bragg peak
// is and what the dose looks like on the way to it. Every piece of the hadron physics feeds
// into that, and each in a way the depth-dose curve distinguishes:
//
//   the range table          -> where the peak sits
//   the restricted dE/dx     -> how high the plateau is
//   delta-ray production     -> how much of the plateau dose is carried by electrons, and so
//                               how far from the track it lands
//   nuclear stopping         -> the last few microns, worth ~0.1% of the total
//   fluctuations             -> the width of the distal falloff, and *only* that. It is the
//                               one metric here that no other part of the physics touches.
//
// Physics list: G4EmStandardPhysics and nothing else. NOT QBBC. A proton in QBBC undergoes
// inelastic nuclear reactions - about 1% per centimetre of water - which remove primaries from
// the beam and redistribute their energy. This port has no hadronic physics at all, so
// comparing against QBBC would measure that absence rather than the stepper. The EM-only list
// is the physics this port actually implements, and it is a legitimate Geant4 configuration
// rather than a contrivance: it is what G4EmStandardPhysics alone gives you.
//
// **Known omissions, stated up front.** No hadronic interactions - see the physics-list note
// above, which is why the reference is run without them either. And the port's Bragg peak sits
// about 0.23 mm proximal of Geant4's, a 0.3% range difference that docs/RISK.md V5 records in
// detail, including the seven things it has been measured *not* to be.
//
// Read the four metrics separately, which is what tools/compare_depth.ps1 does. A stopping
// power 1% high and a range table 1% long cancel in the plateau and add in R80, so a single
// aggregate agreement figure would pass a calculation that is wrong twice.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "G4Box.hh"
#include "G4EmStandardPhysics.hh"
#include "G4LogicalVolume.hh"
#include "G4NistManager.hh"
#include "G4PVPlacement.hh"
#include "G4ParticleGun.hh"
#include "G4ParticleTable.hh"
#include "G4RunManagerFactory.hh"
#include "G4Step.hh"
#include "G4SystemOfUnits.hh"
#include "G4VModularPhysicsList.hh"
#include "G4VUserActionInitialization.hh"
#include "G4VUserDetectorConstruction.hh"
#include "G4VUserPrimaryGeneratorAction.hh"
#include "G4UserSteppingAction.hh"
#include "G4MultiFunctionalDetector.hh"
#include "G4PSEnergyDeposit.hh"
#include "G4SDManager.hh"
#include "Randomize.hh"

// ---------------------------------------------------------------- the phantom, in one place
//
// Both builds read these, so the two geometries cannot drift apart.
namespace cfg {
constexpr int kMaxBins = 400;          ///< storage bound; the run picks how many it uses
constexpr double kHalfXY = 50.0;       ///< mm, half-width of a slab transverse to the beam
constexpr double kWorldHalf = 200.0;   ///< mm

// Slab thickness and count are runtime, not compile-time, and that is worth the small
// awkwardness: the phantom's own slab boundaries chop every step, so the binning is not just
// a readout - it is part of the transport. Being able to re-bin the same phantom depth without
// recompiling is how "does the answer depend on the step size?" gets asked of *both* sides at
// once, which is a question no amount of reading either stepper answers.
inline int& bins() { static int v = 200; return v; }
inline double& slab() { static double v = 0.5; return v; }
inline double depth_lo(int i) { return i * slab(); }
inline double depth_hi(int i) { return (i + 1) * slab(); }
}  // namespace cfg

// The bins. A global because the stepping action, the detector construction and main all need
// it and threading it through three Geant4 base classes would add nothing.
static double g_edep[cfg::kMaxBins];
static std::vector<G4LogicalVolume*> g_slab_lv;

// ---------------------------------------------------------------- geometry
class Phantom : public G4VUserDetectorConstruction {
 public:
  G4VPhysicalVolume* Construct() override {
    auto* nist = G4NistManager::Instance();
    G4Material* vac = nist->FindOrBuildMaterial("G4_Galactic");
    G4Material* water = nist->FindOrBuildMaterial("G4_WATER");

    auto* world_box =
        new G4Box("World", cfg::kWorldHalf * mm, cfg::kWorldHalf * mm, cfg::kWorldHalf * mm);
    auto* world_lv = new G4LogicalVolume(world_box, vac, "World");
    auto* world = new G4PVPlacement(nullptr, G4ThreeVector(), world_lv, "World", nullptr, false,
                                    0, false);

    // One logical volume per slab rather than one replicated: the port keys its per-volume
    // aggregate step on the physical volume's own logical volume, and this way the stepping
    // action below can find the bin by pointer identity in both builds.
    auto* slab_box = new G4Box("Slab", cfg::kHalfXY * mm, cfg::kHalfXY * mm,
                               0.5 * cfg::slab() * mm);
    g_slab_lv.resize(cfg::bins());
    for (int i = 0; i < cfg::bins(); ++i) {
      char name[32];
      std::snprintf(name, sizeof name, "Slab%03d", i);
      auto* lv = new G4LogicalVolume(slab_box, water, name);
      g_slab_lv[i] = lv;
      const double zc = 0.5 * (cfg::depth_lo(i) + cfg::depth_hi(i));
      new G4PVPlacement(nullptr, G4ThreeVector(0, 0, zc * mm), lv, name, world_lv, false, i,
                        false);
    }
    return world;
  }

  void ConstructSDandField() override {
    // A sensitive detector per slab. On the real Geant4 side this is what makes the slab's
    // hits real; on the port's side it is what makes the volume *scored*, and only a scored
    // volume produces the per-event aggregate step the action below reads.
    for (int i = 0; i < cfg::bins(); ++i) {
      char name[48];
      std::snprintf(name, sizeof name, "slabSD%03d", i);
      auto* det = new G4MultiFunctionalDetector(name);
      G4SDManager::GetSDMpointer()->AddNewDetector(det);
      det->RegisterPrimitive(new G4PSEnergyDeposit("edep"));
      SetSensitiveDetector(g_slab_lv[i], det);
    }
  }
};

// ---------------------------------------------------------------- the beam
class Gun : public G4VUserPrimaryGeneratorAction {
 public:
  Gun() : gun_(new G4ParticleGun(1)) {
    gun_->SetParticleDefinition(G4ParticleTable::GetParticleTable()->FindParticle("proton"));
    gun_->SetParticleEnergy(energy_ * MeV);
    gun_->SetParticleMomentumDirection(G4ThreeVector(0, 0, 1));
    // Just outside the first slab, in vacuum, so the full energy enters the water.
    gun_->SetParticlePosition(G4ThreeVector(0, 0, -1.0 * mm));
  }
  ~Gun() override { delete gun_; }
  void GeneratePrimaries(G4Event* evt) override { gun_->GeneratePrimaryVertex(evt); }
  static void SetEnergy(double e) { energy_ = e; }

 private:
  G4ParticleGun* gun_;
  static double energy_;
};
double Gun::energy_ = 100.0;

// ---------------------------------------------------------------- the binning
//
// Identical on both sides, and deliberately written to use only what the port's aggregate step
// carries: the deposit and which volume it happened in. It must NOT use the step's position -
// the port hands out one step per volume per event whose position is not a point on any track.
// See the note at the top of g4/G4Step.hh.
class Binner : public G4UserSteppingAction {
 public:
  void UserSteppingAction(const G4Step* step) override {
    const G4double e = step->GetTotalEnergyDeposit();
    if (e <= 0) { return; }
    const auto* touch = step->GetPreStepPoint()->GetTouchableHandle()->GetVolume();
    if (touch == nullptr) { return; }
    G4LogicalVolume* lv = touch->GetLogicalVolume();
    for (int i = 0; i < cfg::bins(); ++i) {
      if (g_slab_lv[i] == lv) {
        g_edep[i] += e / MeV;
        return;
      }
    }
  }
};

class Actions : public G4VUserActionInitialization {
 public:
  void Build() const override {
    SetUserAction(new Gun());
    SetUserAction(new Binner());
  }
};

// ---------------------------------------------------------------- main
int main(int argc, char** argv) {
  const int n_events = (argc > 1) ? std::atoi(argv[1]) : 20000;
  const double energy = (argc > 2) ? std::atof(argv[2]) : 100.0;
  const char* out = (argc > 3) ? argv[3] : "depth_dose.csv";
  // The production range cut, in mm. A parameter because it is the one knob that
  // switches delta-ray production off on *both* sides at once: raise it until the cut
  // energy exceeds the maximum transfer and no proton can make a transportable
  // electron. That is how a discrepancy gets attributed to the delta rays or ruled out.
  const double range_cut = (argc > 4) ? std::atof(argv[4]) : 0.7;
  // Slab thickness in mm; the count is chosen to keep the phantom 100 mm deep.
  const double slab = (argc > 5) ? std::atof(argv[5]) : 0.5;
  cfg::slab() = slab;
  cfg::bins() = static_cast<int>(100.0 / slab + 0.5);
  if (cfg::bins() > cfg::kMaxBins) { cfg::bins() = cfg::kMaxBins; }
  Gun::SetEnergy(energy);

  auto* rm = G4RunManagerFactory::CreateRunManager(G4RunManagerType::Serial);

  auto* phys = new G4VModularPhysicsList();
  phys->RegisterPhysics(new G4EmStandardPhysics(0));
  phys->SetDefaultCutValue(range_cut * mm);
  rm->SetUserInitialization(phys);
  rm->SetUserInitialization(new Phantom());
  rm->SetUserInitialization(new Actions());

#ifdef G4GPU_PORT
  // The port allocates n_scorers * batch doubles for per-event scores. With 200 scorers the
  // million-event default batch would be 1.6 GB, so the batch is sized to the run.
  rm->SetBatchSize((n_events < 5000) ? n_events : 5000);
#endif

  rm->Initialize();
  G4Random::setTheSeed(12345);
  for (int i = 0; i < cfg::bins(); ++i) { g_edep[i] = 0; }
  rm->BeamOn(n_events);

  double total = 0;
  for (int i = 0; i < cfg::bins(); ++i) { total += g_edep[i]; }

  FILE* f = std::fopen(out, "w");
  if (f == nullptr) {
    std::printf("cannot write %s\n", out);
    return 1;
  }
  std::fprintf(f, "# events=%d energy_MeV=%g bins=%d slab_mm=%g cut_mm=%g total_MeV=%.9g\n", n_events,
               energy, cfg::bins(), cfg::slab(), range_cut, total);
  std::fprintf(f, "bin,z_lo_mm,z_hi_mm,edep_MeV\n");
  for (int i = 0; i < cfg::bins(); ++i) {
    std::fprintf(f, "%d,%.4f,%.4f,%.9g\n", i, cfg::depth_lo(i), cfg::depth_hi(i), g_edep[i]);
  }
  std::fclose(f);

  std::printf("%d protons of %g MeV in water -> %s\n", n_events, energy, out);
  std::printf("  total deposited %.6g MeV of %.6g MeV in  (%.4f%%)\n", total, n_events * energy,
              100.0 * total / (n_events * energy));
  delete rm;
  return 0;
}
