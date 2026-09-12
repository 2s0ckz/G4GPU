// P14c: the two e+- tables the transport reads above 100 MeV, straight off Geant4's objects.
//
// `electron_tables.csv` (in g4dump.cc) already carries `G4EmCalculator::GetDEDX` and
// `::GetRange` for e- and e+ from 1 keV to 100 TeV, which is the dE/dx, range and inverse-range
// comparison. What it does not carry - and what docs/RISK.md V64 turned out to be about - is
// the MULTIPLE SCATTERING table, so this file dumps that.
//
// WHY THE MSC TRANSPORT MEAN FREE PATH NEEDS AN ORACLE OF ITS OWN, AND WHY IT IS NOT
// `urban_msc.csv` OR `wentzel_msc_sample.csv`.
//
//   urban_msc.csv             `G4UrbanMscModel::ComputeCrossSectionPerAtom`, per (Z, energy).
//                             The Urban model's own cross section, below 100 MeV.
//   wentzel_msc_sample.csv    `G4WentzelOKandVIxSection::SampleSingleScattering` as
//                             `G4WentzelVIModel::SampleScattering` configures it - the ANGLE,
//                             at the material's electron production cut.
//   this file                 the TRANSPORT cross section per volume that
//                             `G4VMscModel::xSectionTable` is filled with, on that table's own
//                             43-node grid, and it is computed with a cut of ZERO.
//
// The cut is the point. `G4VMscModel::GetParticleChangeForMSC` builds `xSectionTable` from
// `builder->BuildTableForModel(xSectionTable, this, p, emin, emax, useSpline)` for any particle
// lighter than 1 GeV that is not GenericIon - so e- and e+ read a table where an ion evaluates
// the model (docs/PORTED.md 4.4). `BuildTableForModel` fills it with `model->Value(couple, part,
// E)`, which is `G4VEmModel::Value` = `pFactor * E*E * CrossSectionPerVolume(mat, p, E, 0.0,
// DBL_MAX)`; the `0.0` reaches `G4WentzelVIModel::ComputeCrossSectionPerAtom` as its
// `cutEnergy`, so `SetupTarget(Z, 0.0)` leaves `cosTetMaxElec` at 1 and the electron-scattering
// channel is CLOSED in the tabulated value. The same model's
// `ComputeTransportXSectionPerVolume` - which produces the `xtsec` the single-scattering
// sampler is driven by - reads `(*currentCuts)[currentMaterialIndex]` instead, the material's
// electron production cut, and the channel is open. One model, two cross sections, two cuts.
//
// Both are columns here, because a port that used one for the other would reproduce every
// other number in this file and get the step length wrong. `tests/test_electron_hi.cu` compares
// the zero-cut column against the table it builds and reports the production-cut column as the
// size of the mistake that was available to make.
//
// `SetupKinematic` then `SetupTarget` per element then `ComputeTransportCrossSectionPerAtom`,
// summed with the atom densities, is exactly what `G4VEmModel::CrossSectionPerVolume`'s base
// implementation does with `G4WentzelVIModel::ComputeCrossSectionPerAtom` - which is why this
// file drives `G4WentzelOKandVIxSection` directly rather than the model: the model needs a
// `G4MaterialCutsCouple` set on it and raises a `FatalException` without one.
// It also dumps `electron_hi_tables.csv`: `GetDEDX` and `GetRange` on the ENERGY-LOSS table's
// own 85 nodes, from 100 eV. `electron_tables.csv`'s grid starts at 1 keV and is 40 bins per
// decade, so six of every seven of its points fall BETWEEN the nodes of the table the transport
// reads, and none of them reaches the bottom decade at all. That matters twice: the range at
// 1 keV is the integral from 100 eV upwards, so a disagreement there is invisible from above,
// and comparing at a node removes the interpolation from both sides so that what is left is
// the value.
#include "dump_registry.hh"

#include <cmath>
#include <cstdio>
#include <vector>

#include "G4EmCalculator.hh"
#include "G4Electron.hh"
#include "G4IonisParamMat.hh"
#include "G4Material.hh"
#include "G4PhysicalConstants.hh"
#include "G4Positron.hh"
#include "G4ProductionCutsTable.hh"
#include "G4SystemOfUnits.hh"
#include "G4WentzelOKandVIxSection.hh"

namespace {

/// The electron production cut for one material, which is the cut an msc model is handed on
/// the `ComputeTransportXSectionPerVolume` path. Same function as dump_wentzel_msc.cc's.
double electron_cut(const G4Material* m) {
  auto* pct = G4ProductionCutsTable::GetProductionCutsTable();
  for (std::size_t i = 0; i < pct->GetTableSize(); ++i) {
    if (pct->GetMaterialCutsCouple(static_cast<G4int>(i))->GetMaterial() == m) {
      return (*pct->GetEnergyCutsVector(1))[i];
    }
  }
  return 0.0;
}

/// `G4LossTableBuilder::BuildTableForModel`'s grid for the e+- WentzelVI model:
///
///     emin = max(LowEnergyLimit(), LowEnergyActivationLimit()) -> max(., MinKinEnergy)
///     emax = min(HighEnergyLimit(), HighEnergyActivationLimit()) -> min(., MaxKinEnergy)
///     n    = NumberOfBinsPerDecade() * lrint(log10(emax/tmin))
///
/// with `G4EmStandardPhysics::ConstructProcess` having called
/// `msc2->SetLowEnergyLimit(MscEnergyLimit())`: 100 MeV to 100 TeV, 7 per decade, 43 nodes.
constexpr int kBins = 43;
constexpr double kEMin = 100.0;      // MeV
constexpr int kPerDecade = 7;

/// `G4EmParameters`' own grid for an energy-loss table: MinKinEnergy 100 eV to MaxKinEnergy
/// 100 TeV at NumberOfBinsPerDecade 7, so nBins = 84 and 85 nodes.
constexpr int kLossBins = 85;
constexpr double kLossEMin = 1e-4;   // MeV

/// One element's transport cross section per atom at the model's own nuclear cut-off angle,
/// with `cut` as the cut. `SetupKinematic` must be called before `SetupTarget`, and
/// `SetupTarget` returns the cut-off the transport integral is taken to.
double transport_xs_per_atom(G4WentzelOKandVIxSection* wokvi, const G4ParticleDefinition* p,
                             const G4Material* m, double ekin, G4int z, double cut) {
  wokvi->SetupParticle(p);
  const double cos_tet_max_nuc = wokvi->SetupKinematic(ekin, m);
  if (cos_tet_max_nuc >= 1.0) { return 0.0; }
  const double cost = wokvi->SetupTarget(z, cut);
  return wokvi->ComputeTransportCrossSectionPerAtom(cost);
}

/// `G4VEmModel::CrossSectionPerVolume`'s base implementation over the material's elements.
double transport_xs_per_volume(G4WentzelOKandVIxSection* wokvi, const G4ParticleDefinition* p,
                               const G4Material* m, double ekin, double cut) {
  const G4ElementVector* els = m->GetElementVector();
  const double* nat = m->GetVecNbOfAtomsPerVolume();
  double xs = 0.0;
  for (std::size_t i = 0; i < m->GetNumberOfElements(); ++i) {
    xs += nat[i] * transport_xs_per_atom(wokvi, p, m, ekin, (*els)[i]->GetZasInt(), cut);
  }
  return xs;
}

}  // namespace

static void dump_electron_hi(const DumpContext& ctx) {
  const G4ParticleDefinition* parts[2] = {G4Electron::Electron(), G4Positron::Positron()};

  // ONE OBJECT PER SPECIES, and that is not tidiness. `Initialise` builds the
  // `G4ScreeningMottCrossSection` only `if((p == theElectron || p == thePositron) &&
  // !fMottXSection)`, so a single object re-pointed at the positron with `SetupParticle` would
  // keep the ELECTRON's Mott cross section. `G4WentzelVIModel` has one object per model and one
  // model per particle, which is what this reproduces. (`fMottFactor`, the `1 + 2e-4 Z^2` that
  // does enter the transport cross section, comes from `SetupTarget` and is correct either
  // way - which is exactly why the distinction is easy to miss.)
  G4WentzelOKandVIxSection* wokvi[2];
  for (int i = 0; i < 2; ++i) {
    wokvi[i] = new G4WentzelOKandVIxSection(true);  // isCombined, as the model constructs it
    wokvi[i]->Initialise(parts[i], -1.0);           // cosThetaLim = -1: MscThetaLimit is pi
  }

  // `G4EmCalculator::GetCrossSectionPerVolume` with an msc process name LOOKS like the one
  // accessor that reads `G4VMscModel::xSectionTable`: its `procType == 2` branch does
  // `mscM->SetCurrentCouple(couple); res = 1/mscM->GetTransportMeanFreePath(p, kinEnergy)`.
  // IT IS NOT USABLE ABOVE THE FIRST ENERGY IT IS ASKED ABOUT, and the column is kept so that
  // nobody has to find that out twice.
  //
  // Measured, water, e-: 4.87131008e-05 /mm at 100 MeV and then exactly 0.487131008 /mm at
  // every one of the 42 energies above it - constant to nine figures across six decades, and
  // exactly 1e4 (= (100 MeV)^2) times the first row. Identical behaviour in all seven
  // materials and for both species. A transport cross section that does not change with energy
  // is not a transport cross section; something in `FindLambdaTable`/`FindEmModel`'s caching
  // does not survive the Urban -> WentzelVI model switch at 100 MeV.
  //
  // The FIRST row is meaningful, and it is worth having: at exactly 100 MeV the value is
  // URBAN's, which is an independent confirmation that `G4RegionModels::SelectIndex`'s
  // `e <= lowKineticEnergy[idx]` puts the boundary energy itself on the lower model - the same
  // `<=` that makes `step_lepton`'s WentzelVI branch a strict `>`. `tests/test_electron_hi.cu`
  // checks it against `em::UrbanTable`.
  //
  // So the column the port's `em::WentzelLeptonTable` is compared against is
  // `transport_xs_zerocut_per_mm`, which is `BuildTableForModel`'s own input:
  // `G4VEmModel::Value` = `E^2 * CrossSectionPerVolume(mat, p, E, 0.0, DBL_MAX)`, divided back
  // by E^2. The `_ecut_` column beside it is the same quantity at the production cut, which is
  // what `ComputeTransportXSectionPerVolume` uses and what the table does NOT.
  G4EmCalculator mcalc;
  mcalc.SetVerbose(0);

  FILE* f = std::fopen("electron_hi_msc.csv", "w");
  std::fprintf(f,
               "material,particle,energy_MeV,ecut_MeV,inv_a23,"
               "transport_xs_zerocut_per_mm,transport_xs_ecut_per_mm,"
               "lambda_zerocut_mm,cos_tet_max_nuc,g4emcalc_msc_xs_per_mm\n");
  for (auto* m : ctx.materials) {
    const double cut = electron_cut(m);
    const double inv_a23 = m->GetIonisation()->GetInvA23();
    for (int ip = 0; ip < 2; ++ip) {
      const G4ParticleDefinition* p = parts[ip];
      for (int b = 0; b < kBins; ++b) {
        const double e = kEMin * std::pow(10.0, double(b) / kPerDecade);
        const double xs0 = transport_xs_per_volume(wokvi[ip], p, m, e * MeV, 0.0) * mm;
        const double xsc = transport_xs_per_volume(wokvi[ip], p, m, e * MeV, cut) * mm;
        const double xst = mcalc.GetCrossSectionPerVolume(e * MeV, p, "msc", m) * mm;
        // SetupKinematic's material-level cut-off, re-read after the two sums above so the
        // value is the one belonging to this energy.
        wokvi[ip]->SetupParticle(p);
        const double ctmn = wokvi[ip]->SetupKinematic(e * MeV, m);
        // %.17g on the cross sections and not %.9g: the port reproduces them to the last bits,
        // so nine digits would make the CSV's own rounding the answer.
        std::fprintf(f, "%s,%s,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
                     m->GetName().c_str(), p->GetParticleName().c_str(), e, cut / MeV, inv_a23,
                     xs0, xsc, (xs0 > 0.0) ? 1.0 / xs0 : 0.0, ctmn, xst);
      }
    }
  }
  std::fclose(f);
  for (int i = 0; i < 2; ++i) { delete wokvi[i]; }

  // ---- the energy-loss table on its own nodes, from 100 eV.
  //
  // `GetDEDX` and `GetRange` read the tables `G4LossTableBuilder` built, so this is the vector
  // the transport interpolates, sampled at the points it was built on. `ComputeDEDX` is dumped
  // beside it - that one evaluates the MODELS - so the two columns separate "the port's model
  // is wrong here" from "the port's integral of the table is wrong here", which at the bottom
  // of the grid are different statements and the only place they differ.
  // The LAMBDA TABLES come out of the same accessor, and they are not the model.
  // `GetCrossSectionPerVolume` reads `(*currentLambda)[idx]->Value(e)` for an energy-loss
  // process and `emproc->GetCrossSection(...)` for a G4VEmProcess like `annihil`, where
  // `ComputeCrossSectionPerVolume` - which is what `electron_tables.csv`'s `delta_xs_per_mm`
  // and `brem_xs_per_mm` columns are - calls the model with `aCut = max(cut,
  // LowestElectronEnergy)`. Two differences hide in that gap, and both are real:
  //
  //   * the calculator's 1 keV cut clamp, which bites in AIR alone (whose gamma and electron
  //     cuts are both 0.99 keV) and is worth 18% of the delta-ray cross section at 2 keV;
  //   * `G4EmModelManager::FillLambdaVector`'s `1 + del/e` continuity factor across
  //     G4eBremsstrahlung's 1 GeV model boundary, which the model columns cannot show at all.
  //
  // So the table columns are the ones a port should be compared against, and the model columns
  // are kept beside them because the difference between the two is a measurement.
  G4EmCalculator calc;
  calc.SetVerbose(0);
  FILE* g = std::fopen("electron_hi_tables.csv", "w");
  std::fprintf(g, "material,particle,energy_MeV,ecut_MeV,gcut_MeV,dedx_table,range_mm,"
                  "dedx_ioni_model,dedx_brem_model,lambda_ioni_per_mm,lambda_brem_per_mm,"
                  "lambda_annihil_per_mm\n");
  for (auto* m : ctx.materials) {
    const double ecut = electron_cut(m);
    double gcut = 0.0;
    {
      auto* pct = G4ProductionCutsTable::GetProductionCutsTable();
      for (std::size_t i = 0; i < pct->GetTableSize(); ++i) {
        if (pct->GetMaterialCutsCouple(static_cast<G4int>(i))->GetMaterial() == m) {
          gcut = (*pct->GetEnergyCutsVector(0))[i];
          break;
        }
      }
    }
    for (const G4ParticleDefinition* p : parts) {
      for (int b = 0; b < kLossBins; ++b) {
        const double e = kLossEMin * std::pow(10.0, double(b) / kPerDecade);
        const double dt = calc.GetDEDX(e * MeV, p, m) / (MeV / mm);
        const double r = calc.GetRange(e * MeV, p, m) / mm;
        const double di = calc.ComputeDEDX(e * MeV, p, "eIoni", m, ecut) / (MeV / mm);
        const double db = calc.ComputeDEDX(e * MeV, p, "eBrem", m, gcut) / (MeV / mm);
        const double li = calc.GetCrossSectionPerVolume(e * MeV, p, "eIoni", m) * mm;
        const double lb = calc.GetCrossSectionPerVolume(e * MeV, p, "eBrem", m) * mm;
        const double la = (p == G4Positron::Positron())
                              ? calc.GetCrossSectionPerVolume(e * MeV, p, "annihil", m) * mm
                              : 0.0;
        std::fprintf(g, "%s,%s,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
                     m->GetName().c_str(), p->GetParticleName().c_str(), e, ecut / MeV,
                     gcut / MeV, dt, r, di, db, li, lb, la);
      }
    }
  }
  std::fclose(g);
}

G4GPU_REGISTER_DUMP("electron_hi", "electron_hi_msc.csv electron_hi_tables.csv",
                    dump_electron_hi);
