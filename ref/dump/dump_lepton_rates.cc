// P26: the e+- discrete rates above 100 MeV as Geant4 READS them against the models they are
// built from - README open question 5's "largest named candidate" (docs/RISK.md V78), asked of
// the running process rather than argued.
//
// This port draws an e+-'s discrete interaction lengths from the models at the pre-step energy
// (`em::` cross sections); `G4VEnergyLossProcess` and `G4VEmProcess` read their lambda tables
// (`GetLambda` through `G4EmCalculator::GetCrossSectionPerVolume`, which takes the couple's
// production cut and the process's own table) and `G4EmCalculator::ComputeCrossSectionPerVolume`
// evaluates the model with the same cut. Their ratio, per process, material and energy, is the
// size of the table-against-model difference V225 measured for the photon - the part of open
// question 5 that the discrete RATES can hold. The integral approach (`fXSType`) is not in either
// column; docs/RISK.md says what it would add.
//
//   lepton_rates.csv   particle,process,material,energy_MeV,table_per_mm,model_per_mm,cut_MeV
#include "dump_registry.hh"

#include <cmath>
#include <cstdio>

#include "G4Electron.hh"
#include "G4EmCalculator.hh"
#include "G4Material.hh"
#include "G4MaterialCutsCouple.hh"
#include "G4Positron.hh"
#include "G4ProductionCutsTable.hh"
#include "G4SystemOfUnits.hh"

namespace {

void dump_lepton_rates(const DumpContext& ctx) {
  FILE* f = std::fopen("lepton_rates.csv", "w");
  if (f == nullptr) {
    std::printf("dump_lepton_rates: cannot write lepton_rates.csv\n");
    return;
  }
  std::fprintf(f, "particle,process,material,energy_MeV,table_per_mm,model_per_mm,cut_MeV\n");
  const G4ProductionCutsTable* t = G4ProductionCutsTable::GetProductionCutsTable();
  G4EmCalculator calc;
  struct P { const G4ParticleDefinition* p; const char* proc; int cut_index; };
  // cut index: 0 gamma (bremsstrahlung), 1 e- (ionisation), -1 none (annihilation).
  const P procs[] = {{G4Electron::Electron(), "eIoni", 1}, {G4Electron::Electron(), "eBrem", 0},
                     {G4Positron::Positron(), "eIoni", 1}, {G4Positron::Positron(), "eBrem", 0},
                     {G4Positron::Positron(), "annihil", -1}};
  for (G4Material* m : ctx.materials) {
    const G4MaterialCutsCouple* couple = nullptr;
    for (std::size_t i = 0; i < t->GetTableSize(); ++i) {
      const G4MaterialCutsCouple* c = t->GetMaterialCutsCouple(static_cast<G4int>(i));
      if (c->GetMaterial() == m) { couple = c; }
    }
    if (couple == nullptr) { continue; }
    for (const P& p : procs) {
      const G4double cut =
          (p.cut_index < 0)
              ? 0.0
              : (*t->GetEnergyCutsVector(p.cut_index))[static_cast<std::size_t>(couple->GetIndex())];
      // 20 MeV to 10 GeV, 10 a decade: below and across the 100 MeV msc split and the 1 GeV
      // Seltzer-Berger/relativistic bremsstrahlung split.
      for (int i = 0; i <= 27; ++i) {
        const G4double e = 20.0 * std::pow(10.0, i / 10.0) * MeV;
        const G4double tab = calc.GetCrossSectionPerVolume(e, p.p, p.proc, m);
        const G4double mod = calc.ComputeCrossSectionPerVolume(e, p.p, p.proc, m, cut);
        std::fprintf(f, "%s,%s,%s,%.17g,%.17g,%.17g,%.17g\n", p.p->GetParticleName().c_str(),
                     p.proc, m->GetName().c_str(), e / MeV, tab * mm, mod * mm, cut / MeV);
      }
    }
  }
  std::fclose(f);
}

}  // namespace

G4GPU_REGISTER_DUMP("lepton_rates", "lepton_rates.csv", dump_lepton_rates);
