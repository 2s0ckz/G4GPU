// The three quantities a charged hadron's continuous energy loss is read out of, side by side.
//
// docs/RISK.md V44: Geant4's transported dose for every negative hadron is 4-10% below its
// positive partner's, while `G4EmCalculator::GetDEDX` - the process's own restricted dE/dx
// table, which is what `G4VEnergyLossProcess::AlongStepDoIt` multiplies by the step length -
// has the two charges 0.023% apart in water and 0.233% apart in bone. Those two statements
// cannot both describe the same code path, and `ref/chargeodd/` shows which path is taken:
//
//     eloss = length*GetDEDXForScaledEnergy(preStepScaledEnergy);        // theDEDXTable
//     if(eloss > preStepKinEnergy*linLossLimit) {                        // 0.01 by default
//       G4double x = (fRange - length)/reduceFactor;
//       eloss = preStepKinEnergy - ScaledKinEnergyForLoss(x)/massRatio;  // theRangeTable and
//     }                                                                 // theInverseRange
//
// A 200 MeV pion in G4_BONE_COMPACT_ICRU takes ~20 mm steps and loses ~5% of its energy on
// each, so the second branch fires on every step of every track and theDEDXTable is never
// read. Raise `linLossLimit` to 0.49 with the step lengths untouched and the pi+/pi- split in
// the slab deposit goes from +4.48% to -0.27%; the K+/K- split goes from +9.13% to -1.38%.
//
// So this dump puts the two branches next to each other on the same grid:
//
//   dedx_table            what the short branch would use            (theDEDXTable)
//   range_table           what the long branch reads                 (theRangeTableForLoss)
//   ekin_of_range         and its inverse                            (theInverseRangeTable)
//   eloss_long_<L>mm      E - R^-1(R(E) - L), the long branch verbatim, for L = 1, 5, 20 mm
//   eloss_short_<L>mm     L*dedx_table, the short branch, same L
//   dRdE_inv              1/(dR/dE) by central difference on the range table
//
// `dRdE_inv` against `dedx_table` is the whole finding in one column pair: the range table is
// supposed to be the integral of the dE/dx table, so their ratio is 1 by construction - for
// every positive hadron it is, and for every negative one it is not.
//
// Also dumped, because V44's candidate list asked for them: the model's own dE/dx restricted
// at the couple's cut and unrestricted, the delta-ray production rate from the lambda table
// against the model, and eth = 2 MeV * mass / m_proton - the energy at which G4hIonisation
// hands over from G4BraggModel (positive) or G4ICRU73QOModel (negative) to G4BetheBlochModel.
// If the table and the model disagree the mechanism is in the table build; if they agree it is
// not. G4EmCalculator::FindEmModel is private, so the boundary is recomputed rather than asked
// for.
#include "dump_registry.hh"

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <vector>

#include "G4AntiProton.hh"
#include "G4EmCalculator.hh"
#include "G4KaonMinus.hh"
#include "G4KaonPlus.hh"
#include "G4Material.hh"
#include "G4MuonMinus.hh"
#include "G4MuonPlus.hh"
#include "G4PionMinus.hh"
#include "G4PionPlus.hh"
#include "G4LossTableManager.hh"
#include "G4PhysicsTable.hh"
#include "G4PhysicsVector.hh"
#include "G4ProductionCutsTable.hh"
#include "G4VEnergyLossProcess.hh"
#include "G4Proton.hh"
#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"

namespace {

struct Species {
  const G4ParticleDefinition* def;
  const char* ioni;    ///< the energy-loss process name G4EmCalculator wants
};

/// The electron production threshold of the couple this material has in the dump geometry.
/// The restricted quantities are meaningless without it, and it is the number the transport
/// actually uses: (*theCuts)[currentCoupleIndex] in AlongStepDoIt.
double ElectronCut(const G4Material* m) {
  auto* table = G4ProductionCutsTable::GetProductionCutsTable();
  for (std::size_t i = 0; i < table->GetTableSize(); ++i) {
    if (table->GetMaterialCutsCouple(static_cast<G4int>(i))->GetMaterial() != m) { continue; }
    return (*table->GetEnergyCutsVector(1))[i];
  }
  return 0.99e-3 * MeV;
}

}  // namespace

static void dump_chargeodd(const DumpContext& ctx) {
  // The charge pairs V44 measured, plus the proton pair, which shares the pattern's premise
  // (its own tables, no base particle) and is the species every existing dose comparison in
  // docs/RESULT.md is built on.
  const Species sp[] = {
      {G4MuonPlus::MuonPlus(), "muIoni"},   {G4MuonMinus::MuonMinus(), "muIoni"},
      {G4PionPlus::PionPlus(), "hIoni"},    {G4PionMinus::PionMinus(), "hIoni"},
      {G4KaonPlus::KaonPlus(), "hIoni"},    {G4KaonMinus::KaonMinus(), "hIoni"},
      {G4Proton::Proton(), "hIoni"},        {G4AntiProton::AntiProton(), "hIoni"},
  };
  // 1, 5 and 20 mm. 20 mm is what a 200 MeV pion's step actually is in B1's bone; 1 mm is
  // short enough that the long branch would not have fired; 5 mm is between them, so the
  // three together say whether the error is proportional to the step or not.
  const double L[3] = {1.0, 5.0, 20.0};
  const double kHuge = 1e6 * MeV;

  FILE* f = std::fopen("chargeodd.csv", "w");
  std::fprintf(f, "material,particle,energy_MeV,cut_MeV,dedx_table,range_table_mm,"
                  "csda_range_mm,dRdE_inv,dedx_model_restricted,dedx_model_unrestricted,"
                  "dedx_total_model,delta_xs_table_per_mm,delta_xs_model_per_mm,eth_MeV,"
                  "eloss_long_1mm,eloss_short_1mm,eloss_long_5mm,eloss_short_5mm,"
                  "eloss_long_20mm,eloss_short_20mm\n");

  G4EmCalculator calc;
  calc.SetVerbose(0);
  for (auto* m : ctx.materials) {
    const double cut = ElectronCut(m);
    for (const Species& s : sp) {
      // Twelve points per decade from 1 MeV to 10 GeV. The split V44 measures is a low-
      // velocity effect that is gone by 1 GeV, so the grid has to cross that.
      for (int i = 0; i <= 48; ++i) {
        const double e = std::pow(10.0, i / 12.0);  // MeV
        double dedx = 0, rng = 0, csda = 0, dr = 0, dmr = 0, dmu = 0, dtot = 0;
        double xst = 0, xsm = 0;
        try { dedx = calc.GetDEDX(e * MeV, s.def, m) / (MeV / mm); } catch (...) {}
        try { rng = calc.GetRangeFromRestricteDEDX(e * MeV, s.def, m) / mm; } catch (...) {}
        try { csda = calc.GetCSDARange(e * MeV, s.def, m) / mm; } catch (...) {}
        try { dmr = calc.ComputeDEDX(e * MeV, s.def, s.ioni, m, cut) / (MeV / mm); }
        catch (...) {}
        try { dmu = calc.ComputeDEDX(e * MeV, s.def, s.ioni, m, kHuge) / (MeV / mm); }
        catch (...) {}
        try { dtot = calc.ComputeTotalDEDX(e * MeV, s.def, m, cut) / (MeV / mm); } catch (...) {}
        try { xst = calc.GetCrossSectionPerVolume(e * MeV, s.def, s.ioni, m) * mm; }
        catch (...) {}
        try { xsm = calc.ComputeCrossSectionPerVolume(e * MeV, s.def, s.ioni, m, cut) * mm; }
        catch (...) {}
        // 1/(dR/dE) by a symmetric difference over +-1% in energy. The range table is a log
        // vector with 7 bins per decade, so +-1% is well inside one bin and this measures the
        // vector's interpolated slope - which is what the long branch actually differentiates.
        try {
          const double rp = calc.GetRangeFromRestricteDEDX(1.01 * e * MeV, s.def, m) / mm;
          const double rm = calc.GetRangeFromRestricteDEDX(0.99 * e * MeV, s.def, m) / mm;
          if (rp > rm) { dr = 0.02 * e / (rp - rm); }
        } catch (...) {}
        std::fprintf(f, "%s,%s,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g",
                     m->GetName().c_str(), s.def->GetParticleName().c_str(), e, cut / MeV,
                     dedx, rng, csda, dr, dmr, dmu, dtot, xst, xsm,
                     2.0 * s.def->GetPDGMass() / CLHEP::proton_mass_c2);
        for (double l : L) {
          double lg = 0;  // NOLINT
          // G4VEnergyLossProcess::AlongStepDoIt's long branch, verbatim, with reduceFactor
          // and massRatio both 1 - which they are for every species here except mu+-, where
          // G4MuIonisation's base particle makes them p/mu and 1. GetKinEnergy divides the
          // range by the same reduceFactor, so the pair composes.
          try {
            if (rng > l) { lg = e - calc.GetKinEnergy((rng - l) * mm, s.def, m) / MeV; }
          } catch (...) {}
          std::fprintf(f, ",%.9g,%.9g", lg, l * dedx);
        }
        std::fprintf(f, "\n");
      }
    }
  }
  std::fclose(f);

  // ------------------------------------------------------------------ the vectors themselves
  //
  // `dRdE_inv` above says the negatives' range table is read with LINEAR interpolation and the
  // positives' with a spline: for pi+ the ratio dedx_table/(1/(dR/dE)) is 1 to 2e-4 at every
  // point, and for pi- the same column is piecewise CONSTANT over each 7-bins-per-decade cell.
  // A piecewise-constant derivative is what linear interpolation of R(E) gives. So the flag is
  // read off the vectors directly rather than inferred: G4PhysicsVector::GetSpline() is public,
  // and the range vector's own flag is the one G4PhysicsVector::Interpolation switches on.
  f = std::fopen("chargeodd_vectors.csv", "w");
  std::fprintf(f, "material,particle,dedx_nodes,dedx_spline,dedx_emin_MeV,dedx_v0,"
                  "range_nodes,range_spline,range_emin_MeV,range_v0_mm,"
                  "invrange_nodes,invrange_spline,ioni_nodes,ioni_spline\n");
  auto* ltm = G4LossTableManager::Instance();
  for (auto* m : ctx.materials) {
    // The couple index the transport would use for this material; the vectors are per couple.
    std::size_t idx = 0;
    auto* pct = G4ProductionCutsTable::GetProductionCutsTable();
    for (std::size_t i = 0; i < pct->GetTableSize(); ++i) {
      if (pct->GetMaterialCutsCouple(static_cast<G4int>(i))->GetMaterial() == m) {
        idx = i;
        break;
      }
    }
    for (const Species& s : sp) {
      const G4VEnergyLossProcess* p = ltm->GetEnergyLossProcess(s.def);
      if (nullptr == p) { continue; }
      auto get = [idx](const G4PhysicsTable* t) -> const G4PhysicsVector* {
        return (nullptr != t && idx < t->size()) ? (*t)[idx] : nullptr;
      };
      const G4PhysicsVector* dv = get(p->DEDXTable());
      const G4PhysicsVector* rv = get(p->RangeTableForLoss());
      const G4PhysicsVector* iv = get(p->InverseRangeTable());
      const G4PhysicsVector* nv = get(p->IonisationTable());
      std::fprintf(f, "%s,%s", m->GetName().c_str(), s.def->GetParticleName().c_str());
      std::fprintf(f, ",%d,%d,%.9g,%.9g", dv ? (int)dv->GetVectorLength() : -1,
                   dv ? (int)dv->GetSpline() : -1, dv ? dv->Energy(0) / MeV : 0.0,
                   dv ? (*dv)[0] / (MeV / mm) : 0.0);
      std::fprintf(f, ",%d,%d,%.9g,%.9g", rv ? (int)rv->GetVectorLength() : -1,
                   rv ? (int)rv->GetSpline() : -1, rv ? rv->Energy(0) / MeV : 0.0,
                   rv ? (*rv)[0] / mm : 0.0);
      std::fprintf(f, ",%d,%d", iv ? (int)iv->GetVectorLength() : -1,
                   iv ? (int)iv->GetSpline() : -1);
      std::fprintf(f, ",%d,%d\n", nv ? (int)nv->GetVectorLength() : -1,
                   nv ? (int)nv->GetSpline() : -1);
    }
  }
  std::fclose(f);
}

G4GPU_REGISTER_DUMP("chargeodd", "chargeodd.csv chargeodd_vectors.csv", dump_chargeodd);
