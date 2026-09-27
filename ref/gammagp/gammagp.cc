// Which sub-process of QBBC's G4GammaGeneralProcess a photon's FIRST interaction is, counted in
// the running Geant4 11.1.1 - the measurement behind docs/RISK.md V207 and V208.
//
// WHY THIS PROGRAM EXISTS
//
// P19 read `G4GammaGeneralProcess::PostStepDoIt` and `BuildPhysicsTable` and found that above
// 100 MeV (zone 3) the photonNuclear branch tests `q + P14 <= 1` right after
// `q + P13 <= 1` failed, with table 13 = (sigN + sigM)/sum and table 14 = sigN/sum - identical
// when sigM = 0, which it is in QBBC (no G4GammaConversionToMuons). So the branch cannot be
// taken and the photo-nuclear share of the total goes to the final `else`, conversion. A reading
// of the source is an argument; this is the measurement. And the same general process sums no
// Rayleigh term above 2 m_e (zones 2 and 3), which the port's photon does (V208) - counted here
// on the same photons.
//
// WHAT IT DOES
//
// One box of one material, 10 km on a side, so that a primary never reaches the boundary. A
// photon of energy E starts at the centre along +z; the stepping action records which process
// DEFINED the primary's first post-step interaction - `G4GammaGeneralProcess::SelectedProcess`
// sets the post-step point's process to the SUB-process it chose, so that is `conv`, `compt`,
// `phot`, `Rayl` or `photonNuclear` - and kills the primary; the stacking action kills every
// secondary. Nothing else is transported, so a million photons cost seconds.
//
// Beside each count it prints what the count should be if each sub-process took its share of
// the summed cross section: `G4EmCalculator::ComputeCrossSectionPerVolume` for the four EM
// processes and `G4HadronicProcessStore::GetInelasticCrossSectionPerVolume` - photonNuclear's own
// data store, element by element - for sigN. The general process samples from its own
// interpolated tables, so the expected fractions are good to the tables' interpolation and the
// comparison is a counting one.
//
// Usage:  gammagp.exe <G4 material name> <events per energy> <E1 MeV> [<E2 MeV> ...]
//
//   ref\gammagp\run.bat G4_WATER 1000000 20 99.9 100 150

#include "G4Box.hh"
#include "G4EmCalculator.hh"
#include "G4Gamma.hh"
#include "G4HadronicProcessStore.hh"
#include "G4LogicalVolume.hh"
#include "G4NistManager.hh"
#include "G4PVPlacement.hh"
#include "G4ParticleGun.hh"
#include "G4RunManagerFactory.hh"
#include "G4Step.hh"
#include "G4SystemOfUnits.hh"
#include "G4Track.hh"
#include "G4UserStackingAction.hh"
#include "G4UserSteppingAction.hh"
#include "G4VProcess.hh"
#include "G4VUserActionInitialization.hh"
#include "G4VUserDetectorConstruction.hh"
#include "G4VUserPrimaryGeneratorAction.hh"
#include "QBBC.hh"
#include "Randomize.hh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>

namespace {

std::string g_material = "G4_WATER";
double g_energy = 20 * MeV;
std::map<std::string, long long> g_counts;

class Det : public G4VUserDetectorConstruction {
 public:
  G4VPhysicalVolume* Construct() override {
    G4Material* m = G4NistManager::Instance()->FindOrBuildMaterial(g_material);
    if (m == nullptr) {
      std::printf("FATAL: no material %s\n", g_material.c_str());
      std::exit(1);
    }
    auto* box = new G4Box("World", 5 * km, 5 * km, 5 * km);
    auto* lv = new G4LogicalVolume(box, m, "World");
    return new G4PVPlacement(nullptr, G4ThreeVector(), lv, "World", nullptr, false, 0, true);
  }
};

class Gun : public G4VUserPrimaryGeneratorAction {
 public:
  Gun() : gun_(1) {
    gun_.SetParticleDefinition(G4Gamma::Gamma());
    gun_.SetParticlePosition(G4ThreeVector());
    gun_.SetParticleMomentumDirection(G4ThreeVector(0, 0, 1));
  }
  void GeneratePrimaries(G4Event* ev) override {
    gun_.SetParticleEnergy(g_energy);
    gun_.GeneratePrimaryVertex(ev);
  }

 private:
  G4ParticleGun gun_;
};

class Stepping : public G4UserSteppingAction {
 public:
  void UserSteppingAction(const G4Step* step) override {
    G4Track* t = step->GetTrack();
    if (t->GetTrackID() != 1) { return; }
    const G4StepPoint* post = step->GetPostStepPoint();
    const G4VProcess* p = post->GetProcessDefinedStep();
    const std::string name = (p != nullptr) ? p->GetProcessName() : std::string("(none)");
    if (post->GetStepStatus() == fGeomBoundary || post->GetStepStatus() == fWorldBoundary) {
      ++g_counts["(left the box)"];
    } else {
      ++g_counts[name];
    }
    t->SetTrackStatus(fStopAndKill);
  }
};

class Stacking : public G4UserStackingAction {
 public:
  G4ClassificationOfNewTrack ClassifyNewTrack(const G4Track* t) override {
    return (t->GetParentID() > 0) ? fKill : fUrgent;
  }
};

class Actions : public G4VUserActionInitialization {
 public:
  void Build() const override {
    SetUserAction(new Gun());
    SetUserAction(new Stepping());
    SetUserAction(new Stacking());
  }
};

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::printf("usage: gammagp.exe <material> <events> <E1 MeV> [<E2 MeV> ...]\n");
    return 1;
  }
  g_material = argv[1];
  const long long n = std::atoll(argv[2]);

  auto* rm = G4RunManagerFactory::CreateRunManager(G4RunManagerType::Serial);
  rm->SetUserInitialization(new Det());
  rm->SetUserInitialization(new QBBC(0));
  rm->SetUserInitialization(new Actions());
  rm->Initialize();
  CLHEP::HepRandom::setTheSeed(20260926);

  const G4Material* mat = G4NistManager::Instance()->FindOrBuildMaterial(g_material);
  G4EmCalculator calc;
  const G4ParticleDefinition* gamma = G4Gamma::Gamma();
  std::printf("material,E_MeV,events,process,count,fraction,expected_fraction_if_shares\n");
  for (int k = 3; k < argc; ++k) {
    g_energy = std::atof(argv[k]) * MeV;
    g_counts.clear();
    rm->BeamOn(static_cast<G4int>(n));
    const double e = g_energy;
    const double s_compt = calc.ComputeCrossSectionPerVolume(e, gamma, "compt", mat);
    const double s_conv = calc.ComputeCrossSectionPerVolume(e, gamma, "conv", mat);
    const double s_phot = calc.ComputeCrossSectionPerVolume(e, gamma, "phot", mat);
    const double s_rayl = calc.ComputeCrossSectionPerVolume(e, gamma, "Rayl", mat);
    const double s_gn =
        G4HadronicProcessStore::Instance()->GetInelasticCrossSectionPerVolume(gamma, e, mat);
    // The shares each sub-process would have if all five took theirs of one summed total.
    const double sum = s_compt + s_conv + s_phot + s_rayl + s_gn;
    const std::map<std::string, double> share = {{"compt", s_compt / sum},
                                                 {"conv", s_conv / sum},
                                                 {"phot", s_phot / sum},
                                                 {"Rayl", s_rayl / sum},
                                                 {"photonNuclear", s_gn / sum}};
    for (const auto& kv : share) {
      if (g_counts.find(kv.first) == g_counts.end()) { g_counts[kv.first] = 0; }
    }
    for (const auto& kv : g_counts) {
      const auto it = share.find(kv.first);
      std::printf("%s,%.6g,%lld,%s,%lld,%.6e,%.6e\n", g_material.c_str(), e / MeV, n,
                  kv.first.c_str(), kv.second, double(kv.second) / double(n),
                  (it != share.end()) ? it->second : 0.0);
    }
    std::printf("# %s %.6g MeV: sigma/mm compt %.6e conv %.6e phot %.6e Rayl %.6e "
                "photonNuclear %.6e\n", g_material.c_str(), e / MeV, s_compt * mm,
                s_conv * mm, s_phot * mm, s_rayl * mm, s_gn * mm);
    std::fflush(stdout);
  }
  // NOT `delete rm`. Measured: the run manager's teardown dies with 0xC0000005 after every line
  // above is written - the shape docs/RISK.md's "a destructor that frees a class static" section
  // traced in the dumper, and `G4GammaGeneralProcess::theHandler` is such a static. The numbers
  // are complete and flushed by then; the process leaves without running static destructors.
  (void)rm;
  std::fflush(nullptr);
  std::_Exit(0);
}
