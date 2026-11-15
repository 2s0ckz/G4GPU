// P26: `G4VEmModel::CorrectionsAlongStep` for the species `G4VEnergyLossProcess::AlongStepDoIt`
// calls it for (`if(isIon)`), asked of the running models rather than read.
//
// `ion_tables.csv` (dump_ions.cc) is deliberately blind to the correction - `G4EmCalculator::GetDEDX`
// runs it over a 1 nm step, where its own `eloss < preKinEnergy*0.05` guard makes it inert. This
// dump asks it the question a real step asks: for He3 and four real nuclides in each of B1's
// materials, at energies from 0.1 to 1000 MeV per nucleon and mean losses from 1% to 99% of the
// pre-step energy (5% itself included, and the value one ulp under it), the model the ion's
// `ionIoni` process selects at that energy (`SelectModelForMaterial` on the scaled energy, so
// G4BraggIonModel below 2 MeV per nucleon of the GenericIon scale and G4BetheBlochModel above) is
// handed `eloss` and returns what `AlongStepDoIt` would continue with. The alpha is asked too: its
// models return at once (`if(isAlpha) return;`), and `isIon` keeps the call from being made at
// all, so its rows must come back unchanged.
//
//   ion_corrections.csv   material,ion,Z,A,energy_MeV,eloss_in_MeV,eloss_out_MeV,model
#include "dump_registry.hh"

#include <cmath>
#include <cstdio>
#include <string>

#include "G4Alpha.hh"
#include "G4DynamicParticle.hh"
#include "G4GenericIon.hh"
#include "G4He3.hh"
#include "G4IonTable.hh"
#include "G4Material.hh"
#include "G4MaterialCutsCouple.hh"
#include "G4ParticleTable.hh"
#include "G4ProcessManager.hh"
#include "G4ProcessVector.hh"
#include "G4ProductionCutsTable.hh"
#include "G4SystemOfUnits.hh"
#include "G4VEmModel.hh"
#include "G4VEnergyLossProcess.hh"

namespace {

G4VEnergyLossProcess* ioni_of(const G4ParticleDefinition* p) {
  G4ProcessManager* pm = p->GetProcessManager();
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

void dump_ion_corrections(const DumpContext& ctx) {
  FILE* f = std::fopen("ion_corrections.csv", "w");
  if (f == nullptr) {
    std::printf("dump_ion_corrections: cannot write ion_corrections.csv\n");
    return;
  }
  std::fprintf(f, "material,ion,Z,A,energy_MeV,eloss_in_MeV,eloss_out_MeV,model\n");
  G4IonTable* ions = G4ParticleTable::GetParticleTable()->GetIonTable();
  struct Sp { const char* name; int z, a; };
  const Sp species[] = {{"alpha", 2, 4}, {"He3", 2, 3}, {"C12", 6, 12},
                        {"O16", 8, 16},  {"Fe56", 26, 56}};
  const double fracs[] = {0.01, 0.049, 0.05, 0.0, 0.06, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.99};
  const G4double gion_mass = G4GenericIon::GenericIon()->GetPDGMass();
  for (G4Material* m : ctx.materials) {
    const G4MaterialCutsCouple* couple = couple_of(m);
    if (couple == nullptr) { continue; }
    for (const Sp& s : species) {
      const G4ParticleDefinition* p =
          (s.a == 4)   ? static_cast<const G4ParticleDefinition*>(G4Alpha::Alpha())
          : (s.a == 3) ? static_cast<const G4ParticleDefinition*>(G4He3::He3())
                       : ions->GetIon(s.z, s.a, 0.0);
      if (p == nullptr) { continue; }
      // A real nuclide shares G4GenericIon's process manager; its `ionIoni` is GenericIon's.
      const bool real_ion = (s.a > 4);
      G4VEnergyLossProcess* proc = ioni_of(real_ion ? G4GenericIon::GenericIon() : p);
      if (proc == nullptr) { continue; }
      const G4double base_mass = real_ion ? gion_mass : p->GetPDGMass();
      for (int i = 0; i <= 16; ++i) {
        const G4double e = 0.1 * std::pow(10.0, i / 4.0) * s.a * MeV;
        std::size_t idx = couple->GetIndex();
        G4VEmModel* model =
            proc->SelectModelForMaterial(e * base_mass / p->GetPDGMass(), idx);
        for (double fr : fracs) {
          // 0.0 in the list stands for the largest double under 5%: the guard is `<`.
          G4double eloss_in = (fr == 0.0) ? std::nextafter(e * 0.05, 0.0) : e * fr;
          G4double eloss = eloss_in;
          G4DynamicParticle dp(p, G4ThreeVector(0, 0, 1), e);
          model->CorrectionsAlongStep(couple, &dp, 1.0 * mm, eloss);
          std::fprintf(f, "%s,%s,%d,%d,%.17g,%.17g,%.17g,%s\n", m->GetName().c_str(), s.name,
                       s.z, s.a, e / MeV, eloss_in / MeV, eloss / MeV,
                       model->GetName().c_str());
        }
      }
    }
  }
  std::fclose(f);
}

}  // namespace

G4GPU_REGISTER_DUMP("ion_corrections", "ion_corrections.csv", dump_ion_corrections);
