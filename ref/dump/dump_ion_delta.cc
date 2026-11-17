// P26b: a real nucleus's spin and magnetic moment as G4IonTable::CreateIon gives them, and its
// delta rays as G4VEnergyLossProcess forms the rate and the ion's own model samples them.
//
//   ion_spin.csv        Z,A,found,two_j,spin,mass_MeV,mag_moment2
//       every nuclide 1 <= Z <= 110, Z <= A <= min(3Z + 12, 300) with a positive
//       G4NucleiProperties mass, but the five with definitions of their own: `GetIon(Z, A, 0)`'s
//       `GetPDGiSpin()` / `GetPDGSpin()`, and `magMoment2` computed by G4BetheBlochModel::
//       SetupParameters' own expression on its `GetPDGMagneticMoment()` and `GetPDGMass()`.
//       `found` is G4NuclideTable::GetIsotope(Z, A, 0) != nullptr - whether CreateIon had a row.
//
//   ion_delta.csv       material,Z,A,energy_MeV,cut_MeV,model,q2,xs_model_per_mm,
//                       xs_table_per_mm,mc_n,mc_produced,mc_sum_delta_MeV
//       per (material, nuclide, energy): the model GenericIon's `ionIoni` selects at the scaled
//       energy; the rate as PostStepGetPhysicalInteractionLength composes it from the model
//       (`CrossSectionPerVolume(mat, GenericIon, E*massRatio, cut, DBL_MAX)` times
//       `GetChargeSquareRatio(ion, mat, E)`) and as G4EmCalculator::GetCrossSectionPerVolume reads
//       GenericIon's lambda table; and, at the energies the B1 beams' fragments reach, mc_n calls
//       of `SampleSecondaries` on the ion's own definition with Geant4's engine - how many made an
//       electron and the energy they took. (G4EmCalculator::ComputeCrossSectionPerVolume is NOT
//       used: for a particle with a base particle 11.1.1 computes `e *= kinEnergy*massRatio`, the
//       kinetic energy squared, G4EmCalculator.cc:586.)
//
//   ion_delta_tape.csv  material,Z,A,energy_MeV,r0,delta_MeV,f_threshold,grej_threshold,
//                       delta_ux,delta_uy,delta_uz,primary_MeV,primary_ux,primary_uy,primary_uz
//       the SAME calls driven by a scripted engine, so the port can be held to them deviate for
//       deviate. For each r0: the transfer `SampleSecondaries` makes from (r0, r1 = 0); the
//       largest r1 its inner rejection still accepts (f/fmax, spin term included), found by
//       bisection on r1 with a fallback pair behind it; the largest third deviate its
//       form-factor rejection still accepts (grej, spin and moment included; 1 where no third
//       deviate is drawn); and the final state at (r0, 0, 0, 0.25). -1 everywhere where the
//       window is shut and nothing is drawn.
#include "dump_registry.hh"

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "CLHEP/Random/RandomEngine.h"
#include "G4DynamicParticle.hh"
#include "G4EmCalculator.hh"
#include "G4GenericIon.hh"
#include "G4IonTable.hh"
#include "G4Material.hh"
#include "G4MaterialCutsCouple.hh"
#include "G4NucleiProperties.hh"
#include "G4NuclideTable.hh"
#include "G4ParticleTable.hh"
#include "G4PhysicalConstants.hh"
#include "G4ProcessManager.hh"
#include "G4ProcessVector.hh"
#include "G4ProductionCutsTable.hh"
#include "G4SystemOfUnits.hh"
#include "G4VEmModel.hh"
#include "G4VEnergyLossProcess.hh"
#include "Randomize.hh"

namespace {

G4VEnergyLossProcess* ioni_of_generic_ion() {
  G4ProcessManager* pm = G4GenericIon::GenericIon()->GetProcessManager();
  if (pm == nullptr) { return nullptr; }
  G4ProcessVector* v = pm->GetProcessList();
  for (std::size_t i = 0; i < v->size(); ++i) {
    auto* e = dynamic_cast<G4VEnergyLossProcess*>((*v)[i]);
    if (e != nullptr && e->IsIonisationProcess()) { return e; }
  }
  return nullptr;
}

const G4MaterialCutsCouple* couple_of(const G4Material* m) {
  const G4ProductionCutsTable* t = G4ProductionCutsTable::GetProductionCutsTable();
  for (std::size_t i = 0; i < t->GetTableSize(); ++i) {
    const G4MaterialCutsCouple* c = t->GetMaterialCutsCouple(static_cast<G4int>(i));
    if (c->GetMaterial() == m) { return c; }
  }
  return nullptr;
}

// Returns the scripted deviates in order, then `tail_` for ever.
class ScriptEngine : public CLHEP::HepRandomEngine {
 public:
  void load(const std::vector<double>& v, double tail) {
    seq_ = v;
    n_ = 0;
    tail_ = tail;
  }
  double flat() override { return (n_ < seq_.size()) ? seq_[n_++] : (++n_, tail_); }
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
  double tail_ = 0.5;
};

struct Shot {
  bool produced = false;
  double delta = 0;
  G4ThreeVector ddir, pdir;
  double pkin = 0;
};

Shot shoot(G4VEmModel* model, const G4MaterialCutsCouple* couple, const G4ParticleDefinition* p,
           double e, double cut, ScriptEngine& eng, const std::vector<double>& seq) {
  eng.load(seq, 0.5);
  std::vector<G4DynamicParticle*> out;
  G4DynamicParticle dp(p, G4ThreeVector(0, 0, 1), e);
  model->SampleSecondaries(&out, couple, &dp, cut, DBL_MAX);
  Shot s;
  if (!out.empty()) {
    s.produced = true;
    s.delta = out[0]->GetKineticEnergy();
    s.ddir = out[0]->GetMomentumDirection();
    // The primary's proposed state is in the model's particle change, which is not ours to read;
    // it is the model's own two lines, recomputed here from what it handed back.
    s.pkin = e - s.delta;
    s.pdir = (dp.GetMomentum() - out[0]->GetMomentum()).unit();
    for (auto* d : out) { delete d; }
  }
  return s;
}

void dump_ion_delta(const DumpContext& ctx) {
  G4IonTable* ions = G4ParticleTable::GetParticleTable()->GetIonTable();
  G4NuclideTable* nt = G4NuclideTable::GetNuclideTable();

  // ---- the spin table, against G4IonTable itself.
  {
    FILE* f = std::fopen("ion_spin.csv", "w");
    if (f == nullptr) { return; }
    std::fprintf(f, "Z,A,found,two_j,spin,mass_MeV,mag_moment2\n");
    static const G4double aMag = 1. / (0.5 * eplus * CLHEP::hbar_Planck * CLHEP::c_squared);
    for (int z = 1; z <= 110; ++z) {
      for (int a = z; a <= std::min(3 * z + 12, 300); ++a) {
        if ((z == 1 && a <= 3) || (z == 2 && (a == 3 || a == 4))) { continue; }
        if (!(G4NucleiProperties::GetNuclearMass(a, z) > 0.0)) { continue; }
        const G4ParticleDefinition* p = ions->GetIon(z, a, 0.0);
        if (p == nullptr) { continue; }
        const bool found = nt->GetIsotope(z, a, 0.0) != nullptr;
        const G4double mass = p->GetPDGMass();
        const G4double magmom = p->GetPDGMagneticMoment() * mass * aMag;
        std::fprintf(f, "%d,%d,%d,%d,%.17g,%.17g,%.17g\n", z, a, found ? 1 : 0,
                     p->GetPDGiSpin(), p->GetPDGSpin(), mass / MeV, magmom * magmom - 1.0);
      }
    }
    std::fclose(f);
  }

  // ---- the delta rays.
  G4VEnergyLossProcess* proc = ioni_of_generic_ion();
  if (proc == nullptr) {
    std::printf("dump_ion_delta: GenericIon has no ionIoni\n");
    return;
  }
  const G4ParticleDefinition* gion = G4GenericIon::GenericIon();
  const G4double gion_mass = gion->GetPDGMass();
  const std::vector<G4double>* ecuts =
      G4ProductionCutsTable::GetProductionCutsTable()->GetEnergyCutsVector(idxG4ElectronCut);
  G4EmCalculator calc;

  FILE* f = std::fopen("ion_delta.csv", "w");
  FILE* g = std::fopen("ion_delta_tape.csv", "w");
  if (f == nullptr || g == nullptr) { return; }
  std::fprintf(f, "material,Z,A,energy_MeV,cut_MeV,model,q2,xs_model_per_mm,xs_table_per_mm,"
                  "mc_n,mc_produced,mc_sum_delta_MeV\n");
  std::fprintf(g, "material,Z,A,energy_MeV,r0,delta_MeV,f_threshold,grej_threshold,delta_ux,"
                  "delta_uy,delta_uz,primary_MeV,primary_ux,primary_uy,primary_uz\n");

  // Light fragments the B1 ion beams make above the window, their neighbours, and a few heavier
  // nuclei with and without spin. 2J: He6 0, Li6 2, Li7 3, Be7 3, Be9 3, B10 6, B11 3, C11 3,
  // C12 0, C13 1, N14 2, N15 1, O16 0, O17 5, Na23 3, Fe56 0.
  struct Nuc { int z, a; };
  const Nuc nucs[] = {{1, 4},  {2, 6},  {2, 8},  {3, 6},  {3, 7},  {3, 8},  {3, 9},  {4, 7},
                      {4, 9},  {4, 10}, {5, 8},  {5, 10}, {5, 11}, {5, 12}, {6, 10}, {6, 11},
                      {6, 12}, {6, 13}, {6, 14}, {7, 12}, {7, 13}, {7, 14}, {7, 15}, {8, 14},
                      {8, 15}, {8, 16}, {8, 17}, {8, 18}, {9, 19}, {10, 20}, {11, 23},
                      {13, 27}, {20, 40}, {26, 56}};
  const double per_nucleon[] = {0.5, 1.0, 1.9, 2.5, 10., 50., 150., 200., 300., 500., 1000., 2000.};
  const double r0s[] = {1e-6, 0.1, 0.45, 0.9, 0.999999};
  constexpr long kMc = 1000000;

  CLHEP::HepRandomEngine* saved = CLHEP::HepRandom::getTheEngine();
  ScriptEngine script;
  long seed = 26026;

  for (G4Material* m : ctx.materials) {
    const G4MaterialCutsCouple* couple = couple_of(m);
    if (couple == nullptr) { continue; }
    const std::size_t idx = couple->GetIndex();
    const G4double cut = (*ecuts)[idx];
    for (const Nuc& n : nucs) {
      const G4ParticleDefinition* p = ions->GetIon(n.z, n.a, 0.0);
      if (p == nullptr) { continue; }
      const G4double mass_ratio = gion_mass / p->GetPDGMass();
      for (double pn : per_nucleon) {
        const G4double e = pn * n.a * MeV;
        const G4double es = e * mass_ratio;
        std::size_t i2 = idx;
        G4VEmModel* model = proc->SelectModelForMaterial(es, i2);
        const G4double q2 = model->GetChargeSquareRatio(p, m, e);
        const G4double xs_model = model->CrossSectionPerVolume(m, gion, es, cut, DBL_MAX) * q2;
        const G4double xs_table = calc.GetCrossSectionPerVolume(e, p, "ionIoni", m);
        // The Monte Carlo, at the energies the window is open in water at.
        long produced = 0;
        double sum = 0;
        const long nmc = (pn >= 150.) ? kMc : 0;
        if (nmc > 0) {
          CLHEP::HepRandom::setTheEngine(saved);
          CLHEP::HepRandom::setTheSeed(++seed);
          G4DynamicParticle dp(p, G4ThreeVector(0, 0, 1), e);
          std::vector<G4DynamicParticle*> out;
          for (long k = 0; k < nmc; ++k) {
            out.clear();
            model->SampleSecondaries(&out, couple, &dp, cut, DBL_MAX);
            if (!out.empty()) {
              ++produced;
              sum += out[0]->GetKineticEnergy();
              for (auto* d : out) { delete d; }
            }
          }
        }
        std::fprintf(f, "%s,%d,%d,%.17g,%.17g,%s,%.17g,%.17g,%.17g,%ld,%ld,%.17g\n",
                     m->GetName().c_str(), n.z, n.a, e / MeV, cut / MeV,
                     model->GetName().c_str(), q2, xs_model * mm, xs_table * mm, nmc, produced,
                     sum / MeV);

        // The tape.
        CLHEP::HepRandom::setTheEngine(&script);
        for (double r0 : r0s) {
          const Shot s0 = shoot(model, couple, p, e, cut, script, {r0, 0.0, 0.0, 0.25});
          if (!s0.produced) {
            std::fprintf(g, "%s,%d,%d,%.17g,%.17g,-1,-1,-1,0,0,0,-1,0,0,0\n",
                         m->GetName().c_str(), n.z, n.a, e / MeV, r0);
            continue;
          }
          // f/fmax: the largest r1 whose first pair is accepted. Behind it, (0.5, 0) always
          // is, so a rejected first pair shows as the transfer of r0 = 0.5.
          double lo = 0.0, hi = 1.0;
          {
            const Shot s1 = shoot(model, couple, p, e, cut, script, {r0, 1.0, 0.5, 0.0, 0.0, 0.25});
            if (s1.delta == s0.delta) { lo = 1.0; }
          }
          if (lo < 1.0) {
            for (int it = 0; it < 64; ++it) {
              const double mid = 0.5 * (lo + hi);
              const Shot s1 =
                  shoot(model, couple, p, e, cut, script, {r0, mid, 0.5, 0.0, 0.0, 0.25});
              if (s1.delta == s0.delta) { lo = mid; } else { hi = mid; }
            }
          }
          const double fthr = lo;
          // grej: the largest third deviate that still produces.
          lo = 0.0;
          hi = 1.0;
          {
            const Shot s1 = shoot(model, couple, p, e, cut, script, {r0, 0.0, 1.0, 0.25});
            if (s1.produced) { lo = 1.0; }
          }
          if (lo < 1.0) {
            for (int it = 0; it < 64; ++it) {
              const double mid = 0.5 * (lo + hi);
              const Shot s1 = shoot(model, couple, p, e, cut, script, {r0, 0.0, mid, 0.25});
              if (s1.produced) { lo = mid; } else { hi = mid; }
            }
          }
          const double gthr = lo;
          std::fprintf(g, "%s,%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,"
                          "%.17g,%.17g\n",
                       m->GetName().c_str(), n.z, n.a, e / MeV, r0, s0.delta / MeV, fthr, gthr,
                       s0.ddir.x(), s0.ddir.y(), s0.ddir.z(), s0.pkin / MeV, s0.pdir.x(),
                       s0.pdir.y(), s0.pdir.z());
        }
        CLHEP::HepRandom::setTheEngine(saved);
      }
    }
  }
  CLHEP::HepRandom::setTheEngine(saved);
  std::fclose(f);
  std::fclose(g);
}

}  // namespace

G4GPU_REGISTER_DUMP("ion_delta", "ion_spin.csv ion_delta.csv ion_delta_tape.csv", dump_ion_delta);
