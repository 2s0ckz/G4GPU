// G4WentzelVIModel's single-scattering sampler, for src/physics/em/wentzel_msc.cuh.
//
// One file, `wentzel_msc_sample.csv`: `G4WentzelOKandVIxSection::SampleSingleScattering` set up
// the way `G4WentzelVIModel::SampleScattering` sets it up, N draws per cell under a per-cell
// seed, as cos(theta) moments and a log-spaced histogram of 1 - cos(theta).
//
// WHY THIS IS NOT `coulomb_sample.csv`. Both dump the same Geant4 function, and they hand it
// disjoint angular intervals, because the two processes exist to split the angular range
// between them:
//
//   G4CoulombScattering   cost1 = cosTetMaxNuc (SetupTarget's return), cost2 = -1.
//                         The LARGE angles, beyond the nuclear form-factor cut-off.
//   G4WentzelVIModel      cost1 = cosThetaMin  (a model state variable, below), cost2 = the
//                         same cosTetMaxNuc. The angles BETWEEN the multiple-scattering
//                         cut-off and that same limit, all of them near zero.
//
// So `coulomb_sample.csv` says nothing about this interval, and the port has two copies of the
// sampler for the same reason Geant4 has two callers. There are three further differences, and
// each of them changes a number the sampler reads:
//
//   * THE TARGET MASS. `SetupTarget` sets `factD = sqrt(mom2)/targetMass` through SetTargetMass
//     (G4WentzelOKandVIxSection.cc:206-208) with the ELEMENT's mean atomic mass -
//     `GetAtomicMassAmu(Z)*amu_c2`, or the proton mass at Z = 1. That is what the msc caller
//     samples with: `G4WentzelVIModel::SampleScattering` calls SetupTarget at G4WentzelVIModel
//     .cc:615 and SampleSingleScattering at :618 with nothing in between.
//     `G4eCoulombScatteringModel::SampleSecondaries` OVERRIDES it with the sampled isotope's
//     nuclear mass, which is why coulomb_sample.csv calls SetTargetMass by hand and this file
//     must not.
//   * THE CUT. `G4EmTableUtil::BuildMscProcess` calls `modelManager->Initialise(&part, nullptr,
//     verb)` (G4EmTableUtil.cc:552) - a NULL secondary particle - and G4EmModelManager::
//     Initialise's default is `std::size_t idx = 1` (G4EmModelManager.cc:463), the ELECTRON
//     production cut. G4CoulombScattering's is the proton cut (docs/RISK.md V48). The cut
//     reaches `cosTetMaxElec` and so the electron/nucleus split this sampler draws from, so the
//     two processes draw that split differently even where their intervals touch.
//   * THE SPECIES. See the alpha note below.
//
// WHAT cosThetaMin IS AND HOW THE GRID PICKS IT. It is G4WentzelVIModel's own state variable:
// `ComputeTrueStepLength` lowers it from 1 by `cosThetaMin -= ssFactor*tPathLength/lambdaeff`
// (G4WentzelVIModel.cc:450) once the geometric step is known, and the sampler is then given
// whatever came out. Its value is therefore a property of the STEP and not of the material or
// the energy, and no single number is "the" right one to dump. The grid parameterises it as
//
//     cosThetaMin = 1 - f*(1 - cosTetMaxNuc),   f in {0.05, 0.5}
//
// - f is the fraction of the whole nuclear angular range that the multiple-scattering sub-step
// covers and the single scatters therefore do NOT, which is exactly what ssFactor*t/lambdaeff
// means. f = 0.05 is a short step, so the single scatters carry almost the whole range; f = 0.5
// is a step that hands them half of it. Both are reachable: the model drops to pure single
// scattering when cosThetaMin falls below cosTetMaxNuc, i.e. at f = 1. The VALUE is a column, so
// the port is handed the same number rather than recomputing this expression.
//
// WHY ALPHA IS IN THE SPECIES LIST WHEN QBBC SCATTERS AN ALPHA BY URBAN.
// `G4EmBuilder::ConstructIonEmPhysics` gives alpha `new G4hMultipleScattering()` with no model
// set (G4EmBuilder.cc:133), and `G4hMultipleScattering::InitialiseProcess` then defaults to
// `new G4UrbanMscModel()` (G4hMultipleScattering.cc:76). So Geant4 never routes an alpha
// through the model dumped here. THIS PORT DOES: `uses_wentzel_msc` in core/particle.cuh sends
// alpha, He3, deuteron, triton and GenericIon to WentzelVI because the port's Urban stepping
// half is still the electron's, a substitution named in docs/PORTED.md. The alpha rows are the
// oracle for that substituted path - example B1's alpha beam is transported by this sampler -
// and `G4WentzelOKandVIxSection` is perfectly well defined for an alpha projectile: SetupParticle
// reads only the PDG mass, spin and charge. The rows say what Geant4's own engine would do,
// not what QBBC does.
//
// WHY e- STOPS AT 100 MeV. G4EmStandardPhysics gives e-/e+ UrbanMsc below
// G4EmParameters::MscEnergyLimit() and WentzelVI above it, and that limit is 100 MeV. The e-
// rows are also the control for the defect this file was written for: their rejection function
// is G4ScreeningMottCrossSection's Mott/Rutherford ratio, in which factD does not appear at all,
// so they must not move when it is added or removed.
//
// SetupTarget IS CALLED EXACTLY ONCE PER CELL, and the cell uses what that one call returned.
// `SetupTarget` recomputes only `if(Z != targetZ || tkin != etag)` and the proton-on-hydrogen
// clamp `if(targetZ == 1 && particle == theProton && cosTetMaxNuc2 < 0.0)` lives INSIDE that
// branch, so a second call at the same (Z, tkin) returns the unclamped cosTetMaxNuc. A fresh
// G4WentzelOKandVIxSection per cell makes the dumped interval the first call's, which is the one
// the port's `wentzel_setup` reproduces.
//
// SEEDS. Each cell sets CLHEP::HepRandom::setTheSeed from its own indices, so a cell's numbers
// do not depend on how many cells ran before it.
#include "dump_registry.hh"

#include <cmath>
#include <cstdio>
#include <vector>

#include "G4Alpha.hh"
#include "G4DataVector.hh"
#include "G4Electron.hh"
#include "G4IonisParamMat.hh"
#include "G4Material.hh"
#include "G4MuonMinus.hh"
#include "G4NistManager.hh"
#include "G4PhysicalConstants.hh"
#include "G4PionPlus.hh"
#include "G4ProductionCutsTable.hh"
#include "G4Proton.hh"
#include "G4SystemOfUnits.hh"
#include "G4ThreeVector.hh"
#include "G4WentzelOKandVIxSection.hh"
#include "Randomize.hh"

namespace {

struct Sp {
  const G4ParticleDefinition* def;
  const char* name;
  double emin;   ///< MeV, the lowest grid energy this species is dumped at
};

/// Five projectiles: the two extremes of the mass range the model is given (e- and alpha), the
/// species example B1 fires (proton, alpha) and two more of the light hadrons WentzelVI serves,
/// chosen for their different masses and so different factD at the same kinetic energy.
std::vector<Sp> species() {
  return {
      {G4Electron::Electron(), "e-", 100.0},   // Mott branch: no factD. The control.
      {G4MuonMinus::MuonMinus(), "mu-", 0.0},
      {G4PionPlus::PionPlus(), "pi+", 0.0},
      {G4Proton::Proton(), "proton", 0.0},
      {G4Alpha::Alpha(), "alpha", 0.0},        // the port's substitution; see the header
  };
}

/// The electron production cut for one material, which is the cut an msc model is handed.
double electron_cut(const G4Material* m) {
  auto* pct = G4ProductionCutsTable::GetProductionCutsTable();
  for (std::size_t i = 0; i < pct->GetTableSize(); ++i) {
    if (pct->GetMaterialCutsCouple(static_cast<G4int>(i))->GetMaterial() == m) {
      return (*pct->GetEnergyCutsVector(1))[i];
    }
  }
  return 0.0;
}

const int kZ[4] = {1, 8, 26, 82};

/// 210 MeV is B1's proton beam. The rest is a decade sweep, and the low end is where the term
/// this file exists to measure is largest. The biggest value the factor's argument takes over a
/// cell is max(z1)*factD = min(2, factorA2*<A^-2/3>/mom2) * sqrt(mom2)/targetMass, and with
/// factorA2*<A^-2/3> = 13935 MeV^2 in water that peaks at mom2 = 6968 MeV^2 - about 4 MeV for a
/// proton, 1 MeV for an alpha, 23 MeV for a pion - at sqrt(2*13935)/targetMass, which is 0.178
/// on hydrogen and 0.011 on oxygen. Both ends of the sweep are far from it, on purpose: a grid
/// that only sampled the maximum would say nothing about what the term is worth in a real run.
const double kE[7] = {0.3, 1.0, 3.0, 10.0, 30.0, 210.0, 1000.0};

/// The two step fractions, see the header.
const double kF[2] = {0.05, 0.5};

constexpr int kN = 400000;
constexpr int kBins = 24;

}  // namespace

static void dump_wentzel_msc(const DumpContext& ctx) {
  auto* nist = G4NistManager::Instance();
  const std::vector<Sp> sp = species();

  // Water only. The sampler sees the material through <A^-2/3>, which sets cosTetMaxNuc, and
  // through the production cut, which sets cosTetMaxElec; both are columns, and the port is
  // handed the column rather than its own material. A second material would repeat the same
  // comparison at two other values of two numbers that are already being varied by Z and energy.
  const G4Material* m = nullptr;
  for (auto* mm : ctx.materials) {
    if (mm->GetName() == "G4_WATER") { m = mm; }
  }
  if (nullptr == m) { return; }
  const double cut = electron_cut(m);
  const double inv_a23 = m->GetIonisation()->GetInvA23();

  FILE* f = std::fopen("wentzel_msc_sample.csv", "w");
  // cos_tet_max_nuc_mat is SetupKinematic's return, before SetupTarget's proton-on-hydrogen
  // clamp: the material-level max(cosThetaMax, 1 - factorA2*<A^-2/3>/mom2). It is a column
  // because it and cos_t_max differ for exactly one cell family, and a port that dropped the
  // clamp would still reproduce every other column.
  //
  // NOT a column: fMottFactor. G4WentzelOKandVIxSection exposes no accessor for it, so a column
  // would be this file computing `1 + 2e-4*Z*Z` and the port computing the same expression -
  // two transcriptions of one line compared with each other, which is not an oracle
  // (docs/RISK.md V37). What tests it is the accepted fraction: fMottFactor multiplies the
  // random number the rejection is compared against, and for an electron on lead it is 2.345,
  // so getting it wrong halves the acceptance and the e-/Z=82 cells say so.
  std::fprintf(f, "material,particle,Z,energy_MeV,f,cut_MeV,inv_a23,cos_theta_min,cos_t_max,"
                  "cos_tet_max_nuc_mat,cos_tet_max_elec,elec_ratio,target_mass_MeV,fact_d,"
                  "n,n_scattered,mean_cost,mean_cost2,mean_one_minus_cost,z1_min,z1_max");
  for (int b = 0; b < kBins; ++b) { std::fprintf(f, ",h%d", b); }
  std::fprintf(f, "\n");

  int si = 0;
  for (const Sp& s : sp) {
    ++si;
    for (int iz = 0; iz < 4; ++iz) {
      const int Z = kZ[iz];
      for (int ie = 0; ie < 7; ++ie) {
        const double e = kE[ie];
        if (e < s.emin) { continue; }
        for (int iff = 0; iff < 2; ++iff) {
          // A fresh engine per cell: see the SetupTarget note in the header, and so that a cell
          // cannot inherit tkin or targetZ from the cell before it.
          auto* wokvi = new G4WentzelOKandVIxSection(true);
          // cosThetaLim = -1: G4WentzelVIModel::Initialise leaves cosThetaMax at its in-class
          // -1 when MscThetaLimit is exactly pi, because its two branches test `tet <= 0` and
          // `tet < pi` and neither fires - and pi is the default. It then passes that to
          // wokvi->Initialise (G4WentzelVIModel.cc:118).
          wokvi->Initialise(s.def, -1.0);
          wokvi->SetupParticle(s.def);
          const double ct_nuc_mat = wokvi->SetupKinematic(e * MeV, m);
          const double ctmax = wokvi->SetupTarget(Z, cut);   // exactly once; see the header
          const double ctelec = wokvi->GetCosThetaElec();
          const double mom2 = wokvi->GetMomentumSquare();
          const double tmass = (1 == Z) ? CLHEP::proton_mass_c2
                                        : nist->GetAtomicMassAmu(Z) * CLHEP::amu_c2;
          const double factd = std::sqrt(mom2) / tmass;

          const double z1max = 1.0 - ctmax;
          const double ctmin = 1.0 - kF[iff] * z1max;
          const double z1min = 1.0 - ctmin;

          // The electron/nucleus split, exactly as ComputeTransportXSectionPerVolume builds it
          // (G4WentzelVIModel.cc:738-743) over this cell's interval.
          double ratio = 0.0;
          if (ctmax < ctmin) {
            const double xn = wokvi->ComputeNuclearCrossSection(ctmin, ctmax);
            const double xe = wokvi->ComputeElectronCrossSection(ctmin, ctmax);
            if (xn + xe > 0.0) { ratio = xe / (xn + xe); }
          }

          CLHEP::HepRandom::setTheSeed(910000UL + 10000UL * (unsigned long)si
                                       + 1000UL * (unsigned long)iz
                                       + 10UL * (unsigned long)ie + (unsigned long)iff);
          long h[kBins] = {0};
          double s1 = 0, s2 = 0, s3 = 0;
          long nsc = 0;
          const double lo = std::log10(z1min), hi = std::log10(z1max);
          for (int k = 0; k < kN; ++k) {
            const G4ThreeVector& v = wokvi->SampleSingleScattering(ctmin, ctmax, ratio);
            const double cost = v.z();
            s1 += cost;
            s2 += cost * cost;
            s3 += 1.0 - cost;
            // A rejected draw returns (0,0,1) unchanged - Geant4's "no scattering" answer - and
            // is counted in n but binned nowhere. The accepted fraction is what tests the
            // rejection function and the histogram is what tests its SHAPE, so mixing the two
            // would weaken both.
            if (cost < 1.0) {
              ++nsc;
              const double x = 1.0 - cost;
              int b = 0;
              if (hi > lo && x > 0.0) {
                b = int(kBins * (std::log10(x) - lo) / (hi - lo));
                if (b < 0) { b = 0; }
                if (b >= kBins) { b = kBins - 1; }
              }
              ++h[b];
            }
          }
          std::fprintf(f, "%s,%s,%d,%.9g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,"
                          "%.17g,%.17g,%d,%ld,%.9g,%.9g,%.9g,%.17g,%.17g",
                       m->GetName().c_str(), s.name, Z, e, kF[iff], cut / MeV, inv_a23, ctmin,
                       ctmax, ct_nuc_mat, ctelec, ratio, tmass / MeV, factd, kN, nsc, s1 / kN,
                       s2 / kN, s3 / kN, z1min, z1max);
          for (int b = 0; b < kBins; ++b) { std::fprintf(f, ",%ld", h[b]); }
          std::fprintf(f, "\n");
          delete wokvi;
        }
      }
    }
  }
  std::fclose(f);
}

G4GPU_REGISTER_DUMP("wentzel_msc", "wentzel_msc_sample.csv", dump_wentzel_msc);
