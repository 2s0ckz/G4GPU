// P46: the muon's three discrete EM final states as QBBC's own model objects make them, and the
// pair model's sampling tables (docs/RISK.md V346-V347).
//
//   mu_pair_table.csv   Z,iy,y,ix,x,value
//       G4MuPairProductionModel's G4ElementData, read off the model registered on mu-'s
//       `muPairProd` after initialisation: every node of the five zdat elements' 2D vectors
//       (MakeSamplingTables; G4EmParameters::RetrieveMuDataFromFile is false in option0).
//
//   mu_rad_xs.csv       material,particle,energy_MeV,gcut_MeV,ecut_MeV,pcut_MeV,
//                       muBB_xs,muBrem_xs,muPair_xs,muIoni_tab,muBrems_tab,muPairProd_tab
//       per volume, 1/mm, at the couples' own cuts: the three models' CrossSectionPerVolume
//       (the port's interaction length) and G4EmCalculator::GetCrossSectionPerVolume (the
//       lambda tables Geant4's integral approach reads), so the test can say how far apart they
//       are.
//
//   mu_tape.csv         process,particle,material,energy_MeV,cut_MeV,Z,n_u,u,n_sec,
//                       t0,e0,x0,y0,z0,t1,e1,x1,y1,z1,prim_e,prim_x,prim_y,prim_z
//       SampleSecondaries calls of the three models (muIoni's G4MuBetheBlochModel above 200 keV,
//       G4MuBremsstrahlungModel, G4MuPairProductionModel) with Geant4's engine RECORDED: every
//       deviate each call drew, ';'-separated, so the port can replay them and be held to the
//       final state deviate for deviate. Z is the element the model selected
//       (GetCurrentElement); in a material of more than one element the first deviate is
//       G4EmElementSelector's, which the port draws against partials at the energy instead of
//       the table (its convention), so the test replays the element it is given. The primary's
//       state is recomputed from what came back with the models' own lines - the particle
//       change is not ours to read: T - transfer, and (P dir - sum of secondary momenta).unit().
#include "dump_registry.hh"

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "CLHEP/Random/RandomEngine.h"
#include "G4DynamicParticle.hh"
#include "G4ElementData.hh"
#include "G4EmCalculator.hh"
#include "G4Material.hh"
#include "G4MaterialCutsCouple.hh"
#include "G4MuonMinus.hh"
#include "G4MuonPlus.hh"
#include "G4Physics2DVector.hh"
#include "G4ProcessManager.hh"
#include "G4ProcessVector.hh"
#include "G4ProductionCutsTable.hh"
#include "G4SystemOfUnits.hh"
#include "G4VEmModel.hh"
#include "G4VEnergyLossProcess.hh"
#include "Randomize.hh"

namespace {

G4VEnergyLossProcess* loss_process(const G4ParticleDefinition* p, const char* name) {
  G4ProcessManager* pm = p->GetProcessManager();
  if (pm == nullptr) { return nullptr; }
  G4ProcessVector* v = pm->GetProcessList();
  for (std::size_t i = 0; i < v->size(); ++i) {
    if ((*v)[i]->GetProcessName() == name) {
      return dynamic_cast<G4VEnergyLossProcess*>((*v)[i]);
    }
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

// Geant4's engine, every deviate it hands out written down.
class RecordEngine : public CLHEP::HepRandomEngine {
 public:
  explicit RecordEngine(CLHEP::HepRandomEngine* e) : e_(e) {}
  double flat() override {
    const double u = e_->flat();
    rec_.push_back(u);
    return u;
  }
  void flatArray(const int size, double* vect) override {
    for (int i = 0; i < size; ++i) { vect[i] = flat(); }
  }
  void setSeed(long s, int k) override { e_->setSeed(s, k); }
  void setSeeds(const long* s, int k) override { e_->setSeeds(s, k); }
  void saveStatus(const char[]) const override {}
  void restoreStatus(const char[]) override {}
  void showStatus() const override {}
  std::string name() const override { return "RecordEngine"; }
  std::vector<double> rec_;

 private:
  CLHEP::HepRandomEngine* e_;
};

int type_code(const G4ParticleDefinition* p) {
  if (p->GetParticleName() == "gamma") { return 0; }
  if (p->GetParticleName() == "e-") { return 1; }
  if (p->GetParticleName() == "e+") { return 2; }
  return 9;
}

void dump_mu_radiative(const DumpContext& ctx) {
  const G4ParticleDefinition* parts[2] = {G4MuonMinus::MuonMinus(), G4MuonPlus::MuonPlus()};
  const G4ProductionCutsTable* ct = G4ProductionCutsTable::GetProductionCutsTable();

  // ---- the pair model's tables, off mu-'s process.
  {
    G4VEnergyLossProcess* pp = loss_process(parts[0], "muPairProd");
    G4VEmModel* model = (pp != nullptr) ? pp->EmModel(0) : nullptr;
    G4ElementData* ed = (model != nullptr) ? model->GetElementData() : nullptr;
    FILE* f = std::fopen("mu_pair_table.csv", "w");
    if (f == nullptr || ed == nullptr) {
      std::printf("dump_mu_radiative: no muPairProd element data\n");
      if (f != nullptr) { std::fclose(f); }
      return;
    }
    std::fprintf(f, "Z,iy,y,ix,x,value\n");
    const int zdat[5] = {1, 4, 13, 29, 92};
    for (int z : zdat) {
      G4Physics2DVector* pv = ed->GetElement2DData(z);
      if (pv == nullptr) { continue; }
      for (std::size_t iy = 0; iy < pv->GetLengthY(); ++iy) {
        for (std::size_t ix = 0; ix < pv->GetLengthX(); ++ix) {
          std::fprintf(f, "%d,%zu,%.17g,%zu,%.17g,%.17g\n", z, iy, pv->GetY(iy), ix,
                       pv->GetX(ix), pv->GetValue(ix, iy));
        }
      }
    }
    std::fclose(f);
  }

  // ---- cross sections per volume at the couples' cuts: model and table.
  {
    FILE* f = std::fopen("mu_rad_xs.csv", "w");
    if (f == nullptr) { return; }
    std::fprintf(f, "material,particle,energy_MeV,gcut_MeV,ecut_MeV,pcut_MeV,muBB_xs,muBrem_xs,"
                    "muPair_xs,muIoni_tab,muBrems_tab,muPairProd_tab\n");
    G4EmCalculator calc;
    calc.SetVerbose(0);
    for (const G4ParticleDefinition* p : parts) {
      G4VEnergyLossProcess* pi = loss_process(p, "muIoni");
      G4VEnergyLossProcess* pb = loss_process(p, "muBrems");
      G4VEnergyLossProcess* pp = loss_process(p, "muPairProd");
      if (pi == nullptr || pb == nullptr || pp == nullptr) { continue; }
      for (G4Material* m : ctx.materials) {
        const G4MaterialCutsCouple* c = couple_of(m);
        if (c == nullptr) { continue; }
        const std::size_t idx = c->GetIndex();
        const double gc = (*ct->GetEnergyCutsVector(0))[idx];
        const double ec = (*ct->GetEnergyCutsVector(1))[idx];
        const double pc = (*ct->GetEnergyCutsVector(2))[idx];
        for (int i = 0; i <= 60; ++i) {  // 100 MeV to 100 TeV, ten a decade
          const double e = 100. * std::pow(10.0, i / 10.0) * MeV;
          std::size_t k = idx;
          G4VEmModel* mi = pi->SelectModelForMaterial(e, k);
          const double xi = mi->CrossSectionPerVolume(m, p, e, ec, DBL_MAX) * mm;
          const double xb = pb->EmModel(0)->CrossSectionPerVolume(m, p, e, gc, DBL_MAX) * mm;
          const double xp = pp->EmModel(0)->CrossSectionPerVolume(m, p, e, pc, DBL_MAX) * mm;
          std::fprintf(f, "%s,%s,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
                       m->GetName().c_str(), p->GetParticleName().c_str(), e / MeV, gc / MeV,
                       ec / MeV, pc / MeV, xi, xb, xp,
                       calc.GetCrossSectionPerVolume(e, p, "muIoni", m) * mm,
                       calc.GetCrossSectionPerVolume(e, p, "muBrems", m) * mm,
                       calc.GetCrossSectionPerVolume(e, p, "muPairProd", m) * mm);
        }
      }
    }
    std::fclose(f);
  }

  // ---- the tape.
  FILE* g = std::fopen("mu_tape.csv", "w");
  if (g == nullptr) { return; }
  std::fprintf(g, "process,particle,material,energy_MeV,cut_MeV,Z,n_u,u,n_sec,t0,e0,x0,y0,z0,"
                  "t1,e1,x1,y1,z1,prim_e,prim_x,prim_y,prim_z\n");
  CLHEP::HepRandomEngine* saved = CLHEP::HepRandom::getTheEngine();
  RecordEngine rec(saved);
  CLHEP::HepRandom::setTheSeed(46046);
  CLHEP::HepRandom::setTheEngine(&rec);
  struct Proc {
    const char* name;
    int cut_index;  // the secondary's: e- for muIoni, gamma for muBrems, e+ for muPairProd
    std::vector<double> energies;
  };
  const Proc procs[3] = {
      {"muIoni", 1, {0.25, 1., 10., 100., 300., 1000., 3000., 10000., 1e5, 1e6}},
      {"muBrems", 0, {150., 1000., 3000., 10000., 1e5, 1e6, 1e7}},
      {"muPairProd", 2, {900., 1000., 3000., 10000., 1e5, 1e6, 1e7}},
  };
  constexpr int kShots = 25;
  // A direction off every axis, so rotateUz's general branch is the one exercised.
  const G4ThreeVector dir0 = G4ThreeVector(0.3, -0.4, 0.866).unit();
  for (const Proc& pr : procs) {
    for (const G4ParticleDefinition* p : parts) {
      G4VEnergyLossProcess* proc = loss_process(p, pr.name);
      if (proc == nullptr) { continue; }
      for (G4Material* m : ctx.materials) {
        const G4MaterialCutsCouple* c = couple_of(m);
        if (c == nullptr) { continue; }
        const std::size_t idx = c->GetIndex();
        const double cut = (*ct->GetEnergyCutsVector(pr.cut_index))[idx];
        for (double e0 : pr.energies) {
          const double e = e0 * MeV;
          std::size_t k = idx;
          G4VEmModel* model = proc->SelectModelForMaterial(e, k);
          for (int s = 0; s < kShots; ++s) {
            std::vector<G4DynamicParticle*> out;
            G4DynamicParticle dp(p, dir0, e);
            rec.rec_.clear();
            model->SampleSecondaries(&out, c, &dp, cut, DBL_MAX);
            const G4Element* el = model->GetCurrentElement();
            const int z = (el != nullptr) ? static_cast<int>(el->GetZ() + 0.5) : 0;
            std::string u;
            for (std::size_t i = 0; i < rec.rec_.size(); ++i) {
              char b[40];
              std::snprintf(b, sizeof b, "%s%.17g", i ? ";" : "", rec.rec_[i]);
              u += b;
            }
            std::fprintf(g, "%s,%s,%s,%.17g,%.17g,%d,%zu,%s,%zu", pr.name,
                         p->GetParticleName().c_str(), m->GetName().c_str(), e / MeV, cut / MeV,
                         z, rec.rec_.size(), u.empty() ? "-" : u.c_str(), out.size());
            G4ThreeVector pfin = dp.GetMomentum();
            double transfer = 0;
            for (int j = 0; j < 2; ++j) {
              if (j < static_cast<int>(out.size())) {
                const G4DynamicParticle* d = out[j];
                const G4ThreeVector v = d->GetMomentumDirection();
                std::fprintf(g, ",%d,%.17g,%.17g,%.17g,%.17g", type_code(d->GetDefinition()),
                             d->GetKineticEnergy() / MeV, v.x(), v.y(), v.z());
                pfin -= d->GetMomentum();
                transfer += d->GetKineticEnergy();
                if (std::string(pr.name) == "muPairProd") {
                  transfer += d->GetDefinition()->GetPDGMass();  // a pair's rest masses
                }
              } else {
                std::fprintf(g, ",-1,0,0,0,0");
              }
            }
            const G4ThreeVector pd = out.empty() ? dir0 : pfin.unit();
            std::fprintf(g, ",%.17g,%.17g,%.17g,%.17g\n", (e - transfer) / MeV, pd.x(), pd.y(),
                         pd.z());
            for (auto* d : out) { delete d; }
          }
        }
      }
    }
  }
  CLHEP::HepRandom::setTheEngine(saved);
  std::fclose(g);
}

}  // namespace

G4GPU_REGISTER_DUMP("mu_radiative", "mu_pair_table.csv mu_rad_xs.csv mu_tape.csv",
                    dump_mu_radiative);
