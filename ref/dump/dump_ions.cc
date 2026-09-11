// P8c: what Geant4 answers for a REAL NUCLIDE, as opposed to for G4GenericIon.
//
// `hadron_tables.csv` already carries `G4EmCalculator::GetDEDX` and `::GetRange` for
// G4GenericIon itself - the placeholder definition, 938.2723 MeV and charge 1, whose
// `G4ionIonisation` owns the tables. That is the row `HadronSpecies::kGenericIon` is built from
// and `tests/test_hadron_range.cu` compares it exactly. What it cannot carry is the thing a
// transported ion actually needs: the SCALING from that table to an oxygen recoil.
//
// `G4EmCalculator::UpdateParticle` is where that happens, and the condition is by process name:
//
//     isIon = (currentProcessName == "ionIoni" && p->GetParticleName() != "alpha");
//     if (isIon) { baseParticle = theGenericIon; }
//     ...
//     if (isIon && nullptr != currentProcess) {
//       chargeSquare = corr->EffectiveChargeSquareRatio(p, currentMaterial, kinEnergy);
//       currentProcess->SetDynamicMassCharge(massRatio, chargeSquare);
//     }
//
// so `massRatio = m(G4GenericIon)/m(ion)` and `chargeSquare` is the DYNAMIC effective charge
// squared, recomputed at every energy, and `G4VEnergyLossProcess` then reads
//
//     GetDEDX   = fFactor * DEDX_table(E*massRatio)            fFactor       = chargeSquare
//     GetRange  = reduceFactor * Range_table(E*massRatio)      reduceFactor  = 1/(fFactor*massRatio)
//
// Three columns, therefore, and the third is the one that cannot be derived from the other two
// without assuming the very formula under test: `q2_eff` is dumped from
// `G4EmCorrections::EffectiveChargeSquareRatio` directly, so the port's `ion_effective_charge`
// times its `chargeCorrection` is compared against Geant4's own number rather than against a
// ratio of two dE/dx values.
//
// ---------------------------------------------------------------------------------------------
// TWO THINGS ABOUT THE CALLS BELOW
//
// `GetDEDX`'s own `isIon` block runs `CorrectionsAlongStep` over a 1 nm step, and it is
// NUMERICALLY INERT: both `G4BraggIonModel::CorrectionsAlongStep` and
// `G4BetheBlochModel::CorrectionsAlongStep` open with
//     if(eloss >= preKinEnergy || eloss < preKinEnergy*0.05) { return; }
// and `res*1nm` is far under 5% of the energy. So this column is the table lookup and the
// effective charge, and nothing else - which is what makes it the right thing to compare a
// table against. The correction is NOT inert over a real step, and the port does not apply it;
// that is recorded in docs/PORTED.md rather than hidden here.
//
// `GetScaledRangeForScaledEnergy` caches on `(currentCoupleIndex, scaled energy)` and NOT on
// `reduceFactor`, so two different ions that happened to land on the same scaled energy in the
// same material would get the first one's range. The loops below are ordered (material, ion,
// energy) with one shared energy grid, so consecutive calls are the same ion at different
// energies - which cannot alias - and an ion change lands on a scaled energy that differs by
// the ratio of the two masses.
#include "dump_registry.hh"

#include <cmath>
#include <cstdio>
#include <vector>

#include "G4EmCalculator.hh"
#include "G4EmCorrections.hh"
#include "G4GenericIon.hh"
#include "G4IonTable.hh"
#include "G4LossTableManager.hh"
#include "G4Material.hh"
#include "G4NucleiProperties.hh"
#include "G4ParticleTable.hh"
#include "G4SystemOfUnits.hh"
#include "G4UnitsTable.hh"

namespace {

struct Nuclide {
  int z;
  int a;
  const char* name;
};

void dump_ions(const DumpContext& ctx) {
  // Every nuclide a natural target can recoil as, over the range the elastic process makes and
  // one decade above it, plus two the port must get right for a different reason: O18 is a
  // second isotope of the same element (the mass is the isotope's and not the element's mean
  // A), and Pb208 is the heaviest thing B1's world contains.
  const Nuclide nuclides[] = {
      {3, 7, "Li7"},    {6, 12, "C12"},   {7, 14, "N14"},   {8, 16, "O16"},
      {8, 18, "O18"},   {12, 24, "Mg24"}, {15, 31, "P31"},  {16, 32, "S32"},
      {20, 40, "Ca40"}, {26, 56, "Fe56"}, {82, 208, "Pb208"},
  };

  G4EmCalculator calc;
  G4EmCorrections* corr = G4LossTableManager::Instance()->EmCorrections();
  G4IonTable* ions = G4ParticleTable::GetParticleTable()->GetIonTable();
  const G4ParticleDefinition* gion = G4GenericIon::GenericIon();

  FILE* f = std::fopen("ion_tables.csv", "w");
  if (f == nullptr) {
    std::printf("dump_ions: cannot write ion_tables.csv\n");
    return;
  }
  std::fprintf(f, "material,ion,Z,A,mass_MeV,charge,energy_MeV,dedx_MeV_per_mm,range_mm,"
                  "q2_eff,mass_ratio\n");

  // The GenericIon definition's own mass, which is what massRatio is against - the literal
  // 0.9382723*GeV, NOT CLHEP's proton_mass_c2. The two differ by 3e-7 relative and the port
  // keeps them apart deliberately (core/particle.cuh), so it is dumped rather than assumed.
  std::fprintf(f, "#,G4GenericIon,0,0,%.17g,%.17g,0,0,0,0,1\n", gion->GetPDGMass() / MeV,
               gion->GetPDGCharge() / eplus);

  for (auto* m : ctx.materials) {
    for (const Nuclide& n : nuclides) {
      G4ParticleDefinition* ion = ions->GetIon(n.z, n.a, 0.0);
      if (ion == nullptr) {
        std::fprintf(f, "#,%s,%d,%d,MISSING,0,0,0,0,0,0\n", n.name, n.z, n.a);
        continue;
      }
      // 0.01 MeV to 1 GeV, 12 points per decade - the recoil band the elastic process
      // produces (70 keV threshold, a few MeV typical) with three decades of headroom either
      // side so that the Bragg/Bethe-Bloch hand-over at 2 MeV per nucleon is inside the set
      // for every nuclide in the list.
      for (int i = 0; i <= 60; ++i) {
        const double e = 0.01 * std::pow(10.0, i / 12.0);
        double dedx = 0, range = 0, q2 = 0;
        try { dedx = calc.GetDEDX(e * MeV, ion, m) / (MeV / mm); } catch (...) {}
        try { range = calc.GetRange(e * MeV, ion, m) / mm; } catch (...) {}
        try { q2 = corr->EffectiveChargeSquareRatio(ion, m, e * MeV); } catch (...) {}
        std::fprintf(f, "%s,%s,%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
                     m->GetName().c_str(), n.name, n.z, n.a, ion->GetPDGMass() / MeV,
                     ion->GetPDGCharge() / eplus, e, dedx, range, q2,
                     gion->GetPDGMass() / ion->GetPDGMass());
      }
    }
  }
  std::fclose(f);

  // And the definition itself, so that `em::ion_particle_def` can be compared field by field
  // against `G4IonTable::CreateIon`'s output rather than only through a stopping power. The
  // spin column is the one the port REFUSES: it is `GetiSpin()/2` from ENSDFSTATE.dat's 2J
  // column, which this port does not carry, and `ion_particle_def` therefore reports zero. The
  // dump is what says which nuclides that is wrong for.
  FILE* g = std::fopen("ion_definitions.csv", "w");
  if (g == nullptr) { return; }
  std::fprintf(g, "ion,Z,A,pdg,mass_MeV,nuclear_mass_MeV,charge,spin,magnetic_moment,stable,"
                  "lifetime_ns,shares_generic_ion_manager\n");
  for (const Nuclide& n : nuclides) {
    G4ParticleDefinition* ion = ions->GetIon(n.z, n.a, 0.0);
    if (ion == nullptr) { continue; }
    std::fprintf(g, "%s,%d,%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%d,%.17g,%d\n", n.name, n.z,
                 n.a, ion->GetPDGEncoding(), ion->GetPDGMass() / MeV,
                 G4NucleiProperties::GetNuclearMass(n.a, n.z) / MeV,
                 ion->GetPDGCharge() / eplus, ion->GetPDGSpin(),
                 ion->GetPDGMagneticMoment(), ion->GetPDGStable() ? 1 : 0,
                 ion->GetPDGLifeTime() / ns,
                 (ion->GetProcessManager() == gion->GetProcessManager()) ? 1 : 0);
  }
  std::fclose(g);
}

}  // namespace

G4GPU_REGISTER_DUMP("ions", "ion_tables.csv ion_definitions.csv", dump_ions);
