// P43: the photoelectric and Rayleigh models for every element, not only B1's ten - what
// tests/test_photoelectric.cu and tests/test_rayleigh.cu hold the port's tables to once the
// 12-shell cap is gone (docs/RISK.md V356).
//
//   pe_atom.csv   Z,energy_MeV,phot_mm2,rayl_mm2
//       Z = 1 .. 98, 441 energies from 1 keV to 10 GeV, 40 a decade (the
//       grid of gamma_xs.csv): G4LivermorePhotoElectricModel::ComputeCrossSectionPerAtom and
//       G4LivermoreRayleighModel::ComputeCrossSectionPerAtom, in Geant4's internal units (mm2).
//
//   pe_shell.csv  Z,energy_MeV,r0,produced,electron_MeV,deposit_MeV
//       the shell G4LivermorePhotoElectricModel::SampleSecondaries selects, deviate for deviate:
//       the first flat() of a scripted engine is r0 - the one deviate the shell walk draws, in
//       whichever of its three branches the energy falls - and the photoelectron's kinetic
//       energy is gammaEnergy minus the selected shell's binding energy (deposit_MeV, which is
//       what is deposited locally with fluorescence off, as in QBBC). produced = 0 when the shell
//       is bound tighter than the photon and the whole photon is deposited.
//
// The models are fresh instances, not the run's: their data are class statics (fParamHigh,
// fCrossSection, dataCS - read by InitialiseForElement for any Z on first use, from the
// directory the run's instance found), so a fresh instance computes exactly what the run's would
// for an element the run's materials never asked for. The photoelectric one is a subclass only
// to give it a particle change, which Initialise would otherwise fetch from a process. Neither is
// deleted: both destructors free those class statics when IsMaster(), which is true for any
// instance on the master thread, and the run's own instance still reads them.
//
// The material of SampleSecondaries is a one-element G4Material built here for each Z, with a
// couple that is not in the production-cuts table: the model reads only couple->GetMaterial(),
// and SelectRandomAtom draws no deviate for a material of one element - so r0 is the shell's.
#include "dump_registry.hh"

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "CLHEP/Random/RandomEngine.h"
#include "G4DynamicParticle.hh"
#include "G4Gamma.hh"
#include "G4LivermorePhotoElectricModel.hh"
#include "G4LivermoreRayleighModel.hh"
#include "G4Material.hh"
#include "G4MaterialCutsCouple.hh"
#include "G4NistManager.hh"
#include "G4ParticleChangeForGamma.hh"
#include "G4SystemOfUnits.hh"
#include "Randomize.hh"

namespace {

class PeProbe : public G4LivermorePhotoElectricModel {
 public:
  PeProbe() { fParticleChange = new G4ParticleChangeForGamma(); }
};

// Returns the scripted deviates in order, then the deviates of a real engine. NOT a constant
// tail: the photoelectron's direction (G4SauterGavrilaAngularDistribution::SampleDirection) is a
// rejection loop with no bound, and a constant deviate that it rejects once it rejects for ever -
// the first version of this dump hung there. The direction is not compared; only the shell is.
class ScriptEngine : public CLHEP::HepRandomEngine {
 public:
  void load(const std::vector<double>& v, CLHEP::HepRandomEngine* tail) {
    seq_ = v;
    n_ = 0;
    tail_ = tail;
  }
  double flat() override { return (n_ < seq_.size()) ? seq_[n_++] : tail_->flat(); }
  void flatArray(const int size, double* vect) override {
    for (int i = 0; i < size; ++i) { vect[i] = flat(); }
  }
  void setSeed(long, int) override {}
  void setSeeds(const long*, int) override {}
  void saveStatus(const char[]) const override {}
  void restoreStatus(const char[]) override {}
  void showStatus() const override {}
  std::string name() const override { return "ScriptEngine"; }

 private:
  std::vector<double> seq_;
  std::size_t n_ = 0;
  CLHEP::HepRandomEngine* tail_ = nullptr;
};

void dump_pe_shells(const DumpContext&) {
  const G4ParticleDefinition* gamma = G4Gamma::Gamma();
  auto* pe = new PeProbe();                       // never deleted: see the header
  auto* rayl = new G4LivermoreRayleighModel();    // never deleted: see the header

  {
    FILE* f = std::fopen("pe_atom.csv", "w");
    if (f == nullptr) { return; }
    std::fprintf(f, "Z,energy_MeV,phot_mm2,rayl_mm2\n");
    // 98 and not the models' maxZ of 100, MEASURED: the photoelectric model's ReadData calls
    // fShellCrossSection->InitialiseForComponent(Z, n2), and 11.1.1's G4ElementData holds
    // Z < maxNumElm = 99, so Z = 99 (Es) stops Geant4 with mat602 "Wrong data handling"
    // (docs/RISK.md V356). Geant4 11.1.1 runs the Livermore photoelectric effect to Z = 98.
    for (int z = 1; z <= 98; ++z) {
      for (int i = 0; i <= 440; ++i) {
        const double e = 1e-3 * std::pow(10.0, i / 40.0) * MeV;
        const double sp = pe->ComputeCrossSectionPerAtom(gamma, e, z);
        const double sr = rayl->ComputeCrossSectionPerAtom(gamma, e, z);
        std::fprintf(f, "%d,%.17g,%.17g,%.17g\n", z, e / MeV, sp / mm2, sr / mm2);
      }
    }
    std::fclose(f);
  }

  FILE* fs = std::fopen("pe_shell.csv", "w");
  if (fs == nullptr) { return; }
  std::fprintf(fs, "Z,energy_MeV,r0,produced,electron_MeV,deposit_MeV\n");
  const int zs[] = {1, 2, 6, 8, 13, 20, 26, 29, 36, 37, 47, 53, 56, 64, 74, 77, 79, 82, 83, 92};
  const double es[] = {0.001, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.04,
                       0.05,  0.06,  0.07,  0.08,  0.09,   0.1,  0.12,  0.15, 0.2,  0.3,
                       0.5,   1.0,   2.0,   5.0,   10.0};
  const double r0s[] = {1e-6, 0.01, 0.03, 0.1, 0.2, 0.3, 0.4,  0.5,   0.6,     0.7,
                        0.8,  0.85, 0.9,  0.95, 0.98, 0.99, 0.999, 0.9999, 0.999999};
  CLHEP::HepRandomEngine* saved = CLHEP::HepRandom::getTheEngine();
  ScriptEngine script;
  CLHEP::HepRandom::setTheEngine(&script);
  G4NistManager* nist = G4NistManager::Instance();
  for (const int z : zs) {
    auto* mat = new G4Material("p43_probe_" + std::to_string(z), 1.0 * g / cm3, 1);
    mat->AddElement(nist->FindOrBuildElement(z), 1);
    G4MaterialCutsCouple couple(mat, nullptr);
    for (const double e : es) {
      for (const double r0 : r0s) {
        script.load({r0}, saved);
        std::vector<G4DynamicParticle*> out;
        G4DynamicParticle dp(gamma, G4ThreeVector(0, 0, 1), e * MeV);
        pe->SampleSecondaries(&out, &couple, &dp, 0.0, DBL_MAX);
        const bool produced = !out.empty();
        const double te = produced ? out[0]->GetKineticEnergy() : 0.0;
        std::fprintf(fs, "%d,%.17g,%.17g,%d,%.17g,%.17g\n", z, e, r0, produced ? 1 : 0, te / MeV,
                     produced ? (e * MeV - te) / MeV : e);
        for (auto* d : out) { delete d; }
      }
    }
  }
  CLHEP::HepRandom::setTheEngine(saved);
  std::fclose(fs);
}

}  // namespace

G4GPU_REGISTER_DUMP("pe_shells", "pe_atom.csv pe_shell.csv", dump_pe_shells);
