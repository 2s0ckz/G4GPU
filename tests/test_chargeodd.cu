// The continuous loss a step actually takes, for both charges, against Geant4's own tables.
//
// `tests/test_hadron_range.cu` already compares this port's dE/dx and range to
// `G4EmCalculator::GetDEDX` and `GetRangeFromRestricteDEDX`. It passed while the port's
// transported dose for every negative hadron disagreed with Geant4's by 4-9% (docs/RISK.md
// V44), because the quantity the transport reads is neither of those:
//
//   G4VEnergyLossProcess::AlongStepDoIt, 11.1.1 utils/src/G4VEnergyLossProcess.cc:825-836
//     eloss = length*GetDEDXForScaledEnergy(preStepScaledEnergy);
//     if(eloss > preStepKinEnergy*linLossLimit) {          // linLossLimit = 0.01
//       G4double x = (fRange - length)/reduceFactor;
//       eloss = preStepKinEnergy - ScaledKinEnergyForLoss(x)/massRatio;
//     }
//
// A 200 MeV hadron in B1's scoring volume takes ~20 mm steps and loses ~5% of its energy on
// each, so the SECOND line is what runs, on every step of every track, and it is a difference
// of two range lookups rather than a dE/dx. Its value depends on how the range table is
// interpolated, and for the negative of every charge pair Geant4 interpolates it LINEARLY -
// see `em::hadron_table_uses_spline` for the chain that decides that and docs/RISK.md V46.
//
// So the three checks here are, in the order they compose:
//
//   1. dE/dx at a grid point and between grid points - the interpolation, not just the values.
//   2. the range table, same.
//   3. `E - energy_from_range(range(E) - L)` for L = 1, 5 and 20 mm, which is the long branch
//      verbatim and the number a dose is made of.
//
// The oracle is ref/oracle/chargeodd.csv from ref/dump/dump_chargeodd.cc, twelve points per
// decade from 1 MeV to 10 GeV. Its `eloss_long_<L>mm` columns are
// `E - G4EmCalculator::GetKinEnergy(GetRangeFromRestricteDEDX(E) - L)` - Geant4's own two
// table lookups, composed by the oracle rather than by this port.
//
// WHAT IS NOT COMPARED, and why the tolerances are what they are. Geant4's restricted dE/dx
// table is the SUM over hIoni, hBrems and hPairProd; this port's `hadron_total_dedx` is
// ionisation only (docs/PORTED.md 1.3). That omission is worth 5e-5 of the total for a
// 1.6 GeV muon in water and 3.6e-4 at 10 GeV, so it is below every limit set here - but it is
// the reason the grid stops at 10 GeV rather than at 100 TeV.
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "core/particle.cuh"
#include "data/materials.cuh"
#include "physics/em/hadron_range.cuh"

using namespace g4gpu;
using real_t = double;

namespace {

struct Row {
  std::string mat, part;
  double e = 0, dedx = 0, range = 0;
  double eloss[3] = {0, 0, 0};   ///< the long branch at L = 1, 5, 20 mm
};

/// The three lengths dump_chargeodd.cc tabulates, in its column order.
constexpr double kLen[3] = {1.0, 5.0, 20.0};

std::vector<Row> load(const std::string& path) {
  std::vector<Row> out;
  FILE* f = std::fopen(path.c_str(), "r");
  if (f == nullptr) { return out; }
  char line[2048];
  if (std::fgets(line, sizeof line, f) == nullptr) {
    std::fclose(f);
    return out;
  }
  while (std::fgets(line, sizeof line, f) != nullptr) {
    char mat[128], part[32];
    double cut, dedx, rng, csda, drde, dmr, dmu, dtot, xst, xsm, eth;
    double e, l1, s1, l5, s5, l20, s20;
    const int n = std::sscanf(line,
                              "%127[^,],%31[^,],%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,"
                              "%lf,%lf,%lf,%lf,%lf,%lf,%lf",
                              mat, part, &e, &cut, &dedx, &rng, &csda, &drde, &dmr, &dmu,
                              &dtot, &xst, &xsm, &eth, &l1, &s1, &l5, &s5, &l20, &s20);
    if (n != 20) { continue; }
    Row r;
    r.mat = mat;
    r.part = part;
    r.e = e;
    r.dedx = dedx;
    r.range = rng;
    r.eloss[0] = l1;
    r.eloss[1] = l5;
    r.eloss[2] = l20;
    out.push_back(r);
  }
  std::fclose(f);
  return out;
}

int material_of(const std::string& n) {
  if (n == "G4_AIR") { return data::kAir; }
  if (n == "G4_WATER") { return data::kWater; }
  if (n == "G4_A-150_TISSUE") { return data::kA150Tissue; }
  if (n == "G4_BONE_COMPACT_ICRU") { return data::kBoneCompact; }
  return -1;  // the dump's custom materials, which this port's B1 material set has no row for
}

struct Species {
  const char* name;
  ParticleType type;
  em::HadronSpecies sp;
  bool spline;   ///< what hadron_table_uses_spline must say, asserted below
  /// Limits on the three quantities, per species, each the measured worst case with headroom
  /// and a 0.1% floor. Per species and not shared, for the reason tests/test_hadron_range.cu
  /// gives: a shared limit lets the species that is wrong hide behind the ones that are right,
  /// and here the spread between mu+ and pbar is a factor of fifty.
  ///
  /// Every worst point in the checked materials sits at exactly 10 MeV, the bottom of the band
  /// - the residual falls with energy - and the three wide rows (K-, proton, pbar) are the
  /// tail of the undiagnosed charge-odd air anomaly docs/PORTED.md 1.1 records, which is 4.7%
  /// in air for pbar at that same energy and 0.37% here. They bound it rather than explain it.
  double dedx, range, eloss;
};

// All four charge pairs G4EmBuilder::ConstructLightHadrons and the muon block build, in the
// order they are registered - which is the order that decides the spline flag, so writing them
// this way makes the pattern visible rather than incidental.
const Species kSp[] = {
    //                                                            spline  dedx   range  eloss
    {"mu+", ParticleType::kMuonPlus, em::HadronSpecies::kMuonPlus,
     true, 0.001, 0.001, 0.001},
    {"mu-", ParticleType::kMuonMinus, em::HadronSpecies::kMuonMinus,
     false, 0.001, 0.0015, 0.001},
    {"pi+", ParticleType::kPionPlus, em::HadronSpecies::kPionPlus,
     true, 0.001, 0.001, 0.001},
    {"pi-", ParticleType::kPionMinus, em::HadronSpecies::kPionMinus,
     false, 0.001, 0.0015, 0.001},
    {"kaon+", ParticleType::kKaonPlus, em::HadronSpecies::kKaonPlus,
     true, 0.001, 0.002, 0.001},
    {"kaon-", ParticleType::kKaonMinus, em::HadronSpecies::kKaonMinus,
     false, 0.003, 0.005, 0.003},
    {"proton", ParticleType::kProton, em::HadronSpecies::kProton,
     true, 0.002, 0.003, 0.002},
    {"anti_proton", ParticleType::kAntiProton, em::HadronSpecies::kAntiProton,
     false, 0.005, 0.008, 0.005},
};
constexpr int kN = int(sizeof kSp / sizeof kSp[0]);

int index_of(const std::string& n) {
  for (int i = 0; i < kN; ++i) {
    if (n == kSp[i].name) { return i; }
  }
  return -1;
}

struct Worst {
  double dev = 0;
  int n = 0;
  char where[256] = {0};
};

void note(Worst& w, double dev, const char* fmt, ...) {
  ++w.n;
  if (dev <= w.dev) { return; }
  w.dev = dev;
  va_list ap;
  va_start(ap, fmt);
  std::vsnprintf(w.where, sizeof w.where, fmt, ap);
  va_end(ap);
}

}  // namespace

int main() {
  data::Material<real_t> mats[data::kNumMaterials];
  data::build_b1_materials<real_t>(mats);
  static em::ShellTables<real_t> shell;
  em::build_shell_tables(shell);

  const char* env = std::getenv("G4GPU_ORACLE");
  const std::string dir = (env != nullptr) ? env : "ref/oracle";
  const auto rows = load(dir + "/chargeodd.csv");
  if (rows.empty()) {
    std::printf("cannot read %s/chargeodd.csv - run ref/oracle/run.bat tables first\n",
                dir.c_str());
    return 1;
  }

  static em::HadronRangeTable<real_t> table;
  real_t cuts[data::kNumMaterials];
  for (int m = 0; m < data::kNumMaterials; ++m) { cuts[m] = mats[m].cut_electron; }
  em::build_hadron_range_table<real_t>(mats, cuts, table, &shell);

  int fails = 0;

  // ---------------------------------------------------------------- 0. the flag itself
  //
  // The predicate before the numbers, because if it is wrong the numbers below say only that
  // one of the eight species is wrong and not which half of the pattern moved. The values are
  // ref/oracle/chargeodd_vectors.csv's dedx_spline / range_spline columns, transcribed into
  // kSp above: 1 for the first-registered member of each pair and 0 for the second.
  for (int i = 0; i < kN; ++i) {
    const bool got = em::hadron_table_uses_spline(kSp[i].sp);
    if (got != kSp[i].spline) {
      std::printf("  FAIL: hadron_table_uses_spline(%s) is %d, oracle says %d\n", kSp[i].name,
                  int(got), int(kSp[i].spline));
      ++fails;
    }
    // And the second derivatives must actually follow it. A predicate nothing reads would pass
    // the check above and change no number at all.
    double sum = 0;
    for (int b = 0; b < em::kHadronRangeBins; ++b) {
      sum += std::fabs(table.dedx_d2[int(kSp[i].sp)][data::kWater][b])
             + std::fabs(table.range_d2[int(kSp[i].sp)][data::kWater][b]);
    }
    if (kSp[i].spline && !(sum > 0.0)) {
      std::printf("  FAIL: %s should be splined but every dedx_d2/range_d2 is zero\n",
                  kSp[i].name);
      ++fails;
    }
    if (!kSp[i].spline && sum != 0.0) {
      std::printf("  FAIL: %s should not be splined but sum|d2| = %g\n", kSp[i].name, sum);
      ++fails;
    }
    // The INVERSE range table is splined for both charges - G4LossTableBuilder::
    // BuildInverseRangeTable builds a fresh vector rather than copying the range vector - and
    // that asymmetry is the whole mechanism, so it is checked and not assumed.
    double isum = 0;
    for (int b = 0; b < em::kHadronRangeBins; ++b) {
      isum += std::fabs(table.inv_d2[int(kSp[i].sp)][data::kWater][b]);
    }
    if (!(isum > 0.0)) {
      std::printf("  FAIL: %s inverse range table is not splined\n", kSp[i].name);
      ++fails;
    }
  }

  // ---------------------------------------------------------------- the three comparisons
  //
  // ---- what is in scope, and why each bound is where it is.
  //
  // E >= 10 MeV. Below it the low-energy models take over - G4hIonisation hands over at
  // eth = 2 MeV * mass / m_proton, so 0.2975 MeV for a pion, 1.055 for a kaon, 2.0 for a
  // proton, and G4MuIonisation at a flat 200 keV - and their own accuracy is
  // tests/test_hadron_range.cu's subject, including the charge-odd G4_AIR anomaly recorded
  // for pbar and K- in docs/PORTED.md 1.1. Bringing that here would mean this file failing
  // for a reason it is not about. Above 10 MeV every species is on G4BetheBlochModel or
  // G4MuBetheBlochModel and the only charge-odd term left is the Barkas piece inside
  // G4EmCorrections::HighOrderCorrections, which is already checked.
  //
  // eloss > 0.01 * E. Not a convenience: it is `if(eloss > preStepKinEnergy*linLossLimit)`,
  // AlongStepDoIt's own condition for reaching the long branch, with G4EmParameters'
  // linLossLimit default. A row that fails it is a step Geant4 would have taken the SHORT
  // branch on, so the long-branch value is not a number any transport reads - and in G4_AIR a
  // 20 mm step loses 1e-4 of the energy, so `E - R^-1(R(E) - L)` there is the difference of
  // two ranges of several hundred mm and is all cancellation. The first version of this test
  // had no such gate and reported 65839% for mu+ in air on a loss of 9e-9 MeV.
  // Both ends of the step, not just the pre-step energy. A 1 mm step at 10 MeV in
  // G4_A-150_TISSUE takes an antiproton from 10 MeV to 2.04 MeV - through the model boundary
  // and into the band docs/PORTED.md 1.1 carries a 25% recorded measurement for. Requiring
  // E - eloss >= kEMin keeps the whole step inside the band this file is about; without it the
  // antiproton failed at 1.43% on a step that nearly stopped it.
  const double kEMin = 10.0;
  constexpr double kLinLossLimit = 0.01;   // G4EmParameters::LinearLossLimit

  Worst dedx[kN], rng[kN], els[kN][3];
  Worst adedx[kN], arng[kN];   // G4_AIR: recorded, not a limit. See the print below.
  for (const Row& r : rows) {
    const int pi = index_of(r.part);
    const int mi = material_of(r.mat);
    if (pi < 0 || mi < 0 || r.e > 1e4 || r.e < kEMin) { continue; }
    // G4_AIR is measured and printed but not gated, and this is the one place that decides it.
    // docs/PORTED.md 1.1 and tests/test_hadron_range.cu record an undiagnosed charge-odd air
    // anomaly: at 10 MeV, above every model boundary, Geant4's own
    // G4EmCalculator::ComputeDEDX gives an antiproton 4% MORE ionisation than a proton in air
    // and the same to 0.01% in water, and this port does not reproduce it. So a negative
    // hadron's dE/dx in air is already 2-5% away from Geant4 before any question about
    // interpolation, and an air row cannot test an interpolation rule. Excluding it silently
    // would be hiding it; the numbers are printed beside the checked ones.
    const bool air = (mi == data::kAir);
    if (r.dedx > 0) {
      const real_t ours = table.dedx_at(kSp[pi].sp, mi, real_t(r.e));
      note(air ? adedx[pi] : dedx[pi], std::fabs(ours - r.dedx) / r.dedx,
           "%s at %.4g MeV (%.6g vs %.6g MeV/mm)", r.mat.c_str(), r.e, double(ours), r.dedx);
    }
    if (r.range > 0) {
      const real_t ours = table.lookup(kSp[pi].sp, mi, real_t(r.e));
      note(air ? arng[pi] : rng[pi], std::fabs(ours - r.range) / r.range,
           "%s at %.4g MeV (%.6g vs %.6g mm)", r.mat.c_str(), r.e, double(ours), r.range);
    }
    if (air) { continue; }
    for (int k = 0; k < 3; ++k) {
      if (!(r.eloss[k] > 0) || !(r.range > kLen[k])) { continue; }
      if (r.eloss[k] <= kLinLossLimit * r.e) { continue; }
      if (r.e - r.eloss[k] < kEMin) { continue; }
      // G4VEnergyLossProcess::AlongStepDoIt's long branch. reduceFactor and massRatio are 1
      // for all eight species here, so fRange is the range lookup and the subtraction is in
      // millimetres of real path.
      const real_t rr = table.lookup(kSp[pi].sp, mi, real_t(r.e));
      const real_t ours = real_t(r.e) - table.energy_from_range(kSp[pi].sp, mi,
                                                                rr - real_t(kLen[k]));
      note(els[pi][k], std::fabs(ours - r.eloss[k]) / r.eloss[k],
           "%s at %.4g MeV (%.6g vs %.6g MeV)", r.mat.c_str(), r.e, double(ours), r.eloss[k]);
    }
  }

  std::printf("== dE/dx table, interpolation included ==  (G4_AIR recorded, not a limit)\n");
  for (int i = 0; i < kN; ++i) {
    std::printf("  %-12s %8.4f%% / %.2f%%  air %7.3f%%  n=%-5d %s\n", kSp[i].name,
                100 * dedx[i].dev, 100 * kSp[i].dedx, 100 * adedx[i].dev, dedx[i].n,
                dedx[i].where);
    if (dedx[i].dev > kSp[i].dedx) { ++fails; }
    if (dedx[i].n == 0 || adedx[i].n == 0) {
      std::printf("  FAIL: no oracle rows for %s - is it in dump_chargeodd.cc?\n", kSp[i].name);
      ++fails;
    }
  }
  std::printf("\n== range table, interpolation included ==  (G4_AIR recorded, not a limit)\n");
  for (int i = 0; i < kN; ++i) {
    std::printf("  %-12s %8.4f%% / %.2f%%  air %7.3f%%  n=%-5d %s\n", kSp[i].name,
                100 * rng[i].dev, 100 * kSp[i].range, 100 * arng[i].dev, rng[i].n,
                rng[i].where);
    if (rng[i].dev > kSp[i].range) { ++fails; }
  }
  std::printf("\n== AlongStepDoIt long branch, E - R^-1(R(E)-L) ==\n");
  std::printf("  %-12s %10s %10s %10s %8s\n", "", "L=1mm", "L=5mm", "L=20mm", "limit");
  for (int i = 0; i < kN; ++i) {
    std::printf("  %-12s", kSp[i].name);
    for (int k = 0; k < 3; ++k) { std::printf(" %9.4f%%", 100 * els[i][k].dev); }
    std::printf(" %7.2f%%\n", 100 * kSp[i].eloss);
    for (int k = 0; k < 3; ++k) {
      if (els[i][k].dev > kSp[i].eloss) {
        std::printf("    FAIL: %s at L=%g mm, %s\n", kSp[i].name, kLen[k], els[i][k].where);
        ++fails;
      }
      if (els[i][k].n == 0) {
        std::printf("    FAIL: %s at L=%g mm compared nothing\n", kSp[i].name, kLen[k]);
        ++fails;
      }
    }
  }

  std::printf("\n%s (%d failures)\n", fails ? "FAILED" : "PASSED", fails);
  return fails ? 1 : 0;
}
