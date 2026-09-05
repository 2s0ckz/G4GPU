// Stopping power and range for every heavy charged particle, against Geant4's own tables.
//
// `dedx_total` and `range_mm` in ref/oracle/hadron_tables.csv are G4EmCalculator::GetDEDX and
// ::GetRange - the sum over every energy-loss process Geant4 attached to that particle, and the
// range table it built by integrating that sum.
//
// That matters the moment a stepper needs a range: a step is limited by a fraction of the
// remaining range, and the energy after a step of known length is the inverse lookup. A range
// table wrong by a few per cent puts a Bragg peak in the wrong place, and the depth of a Bragg
// peak is the number a hadron calculation exists to produce.
//
// ---------------------------------------------------------------------------------------
// WHAT CHANGED, AND WHY THE SHAPE OF THIS TEST CHANGED WITH IT
//
// It used to compare three particles - proton, alpha, He3 - because the port tabulated two
// species and mapped everything else onto one of them. He3's dE/dx was 91% wrong and this test
// said so, in a cell whose limit was set to 100% so that nothing gated on it.
//
// Geant4's structure is one table per *base particle*: G4hIonisation names eight by hand
// (proton, anti_proton, pi+, pi-, kaon+, kaon-, GenericIon, alpha), G4MuIonisation gives mu+
// and mu- no base at all, and everything else - He3, deuteron, triton, hyperons - looks one of
// those ten up at a scaled energy. The port now builds the same ten, so this compares all of
// them, and He3 through the scaling rather than through a table it should never have had.
//
// Two comparisons, in order, because the second depends on the first:
//
//   1. dE/dx: ours against Geant4's total, per particle per energy band. Where a *model* error
//      shows. This is also where the negative hadrons earn their place: pi-, K- and the
//      anti-proton take G4ICRU73QOModel below the boundary while the positives take
//      G4BraggModel, and a dispatch written on charge *magnitude* rather than sign passes
//      every proton and alpha check and gets all three of them wrong.
//   2. range: ours against Geant4's, per particle per band.
//
// Then the inverse, per particle, because a round trip that does not close is a stepper that
// loses or invents energy on every step.
//
// Not here: deuteron and triton. Both are in the oracle - G4EmBuilder gives them
// G4hIonisation with the proton as base particle - and neither is in this port's ParticleType
// yet. They are the obvious next two.
#include <cmath>
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
  double e = 0, dedx_ioni = 0, dedx_total = 0, range = 0, nuclear = 0;
};

/// hadron_tables.csv is:
///   material,particle,mass_MeV,charge,energy_MeV,dedx_ioni,dedx_total,range_mm,
///   delta_xs_per_mm,dedx_brem,dedx_pair,brem_xs_per_mm,pair_xs_per_mm,nuclear_dedx
std::vector<Row> load(const std::string& path) {
  std::vector<Row> out;
  FILE* f = std::fopen(path.c_str(), "r");
  if (f == nullptr) { return out; }
  char line[1024];
  if (std::fgets(line, sizeof line, f) == nullptr) {
    std::fclose(f);
    return out;
  }
  while (std::fgets(line, sizeof line, f) != nullptr) {
    char mat[128], part[32];
    double mass, q, e, di, dt, r, dx, db, dp, bx, px, nd;
    const int n = std::sscanf(line,
                              "%127[^,],%31[^,],%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf",
                              mat, part, &mass, &q, &e, &di, &dt, &r, &dx, &db, &dp, &bx, &px,
                              &nd);
    if (n != 14) { continue; }
    out.push_back({mat, part, e, di, dt, r, nd});
  }
  std::fclose(f);
  return out;
}

constexpr int kNBands = 5;

// ---------------------------------------------------------------- the particles compared
//
// One row per particle in the oracle this port can name, each with its own limits per band,
// because they are not the same physics: a proton on G4BraggModel below 2 MeV, a pi- on
// G4ICRU73QOModel below 297 keV, a muon on G4MuBetheBlochModel above 200 keV. A single
// threshold over all of them lets the one that is wrong hide behind the ones that are right,
// which is what the three-particle version of this test did.
struct Species {
  const char* name;
  ParticleType type;
  double dedx[kNBands];
  double range[kNBands];
};

// Bands: 1-10 keV, 10-100 keV, 0.1-2 MeV, 2-100 MeV, 0.1-10 GeV.
//
// ---------------------------------------------------------------------------------------
// TWO KINDS OF NUMBER LIVE IN THIS TABLE AND CONFUSING THEM WOULD DEFEAT THE POINT
//
// For **proton and alpha** - the two species this port actually transports - these are
// tolerances. Each is the measured worst case rounded up, and a regression past it is a bug.
//
// For everything else they are *recorded measurements* of a path that is not finished, in the
// same spirit as the He3 row of the version of this test that preceded it: wide enough to pass
// today, narrow enough to notice movement, and no evidence at all that the species is right.
// None of them is transportable - G4RunManager::CheckSpecies refuses them at the gun.
//
// WHAT THE TABLE GRID BOUGHT
//
// G4VEnergyLossProcess never evaluates a model during transport. It builds a dE/dx vector on a
// log grid - MinKinEnergy 100 eV to MaxKinEnergy 100 TeV, 7 bins per decade, 85 points -
// splines it, and integrates that for the range. This port used to evaluate the models exactly
// on 256 points from 1 keV with 16 trapezoid sub-steps: a finer description of the physics and
// a worse description of Geant4. It now builds Geant4's grid, with Geant4's spline and
// G4LossTableBuilder's integration. Measured, before and after:
//
//                       dE/dx                       range
//                    before    after            before    after
//     proton  1-10keV  0.407%   0.000%           0.044%    0.000%
//     proton  0.1-2MeV 0.848%   0.038%           0.167%    0.018%
//     alpha   0.1-2MeV 0.523%   0.000%           0.094%    0.000%
//     mu-     1-10keV  4.373%   2.013%          52.468%    0.836%
//     pi-     1-10keV  4.508%   2.139%          40.326%    1.359%
//     GenIon  1-10keV  0.889%   0.001%           3.433%    0.001%
//
// The zeros are not rounding: on Geant4's grid this port evaluates the same models at the same
// points, so away from a model boundary the two tables *are* the same table. See docs/RISK.md
// V7 - and note that the proton's old 1.3% worst point, which a comment here used to attribute
// to "the residual ICRU90 omission ... measured rather than assumed", was neither measured nor
// that.
//
// WHAT IS LEFT, AND IT IS ONE THING
//
// NEGATIVE HADRONS IN AIR. Not the grid - it is slightly *worse* on Geant4's grid - and not a
// model boundary: at 10 MeV, far above every boundary, Geant4's own
// G4EmCalculator::ComputeDEDX gives an anti-proton 4% MORE ionisation than a proton in air,
// and the same to 0.01% in water. The Barkas term is odd in the charge and would make it less,
// not more; ref/oracle/corrections.csv accounts for a sixth of the gap; and ICRU90 is not the
// answer either (G4EmParameters::fICRU90 defaults to false, and G4BetheBlochModel only ever
// loads it for proton, GenericIon and alpha). Something charge-odd in the air path is not a
// term this port knows about. Open, undiagnosed, and left visible in the anti_proton, kaon-
// and mu- rows rather than absorbed into a limit.
//
// Bands: 1-10 keV, 10-100 keV, 0.1-2 MeV, 2-100 MeV, 0.1-10 GeV.
const Species kSpecies[] = {
    // ---- TRANSPORTED. These two are tolerances.
    //
    // The 2-100 MeV band is the model boundary - 2 MeV for a proton, 7.95 MeV for an alpha -
    // where Geant4's 7-bin-per-decade spline rings across the discontinuity between
    // G4BraggModel and G4BetheBlochModel. It is the one band where two identical splines
    // through two identical sets of points still disagree, because the oracle's sampling
    // (20 points per decade) lands between the table's.
    {"proton", ParticleType::kProton,
     {0.002, 0.002, 0.002, 0.015, 0.002},
     {0.002, 0.002, 0.002, 0.008, 0.002}},
    {"alpha", ParticleType::kAlpha,
     {0.002, 0.002, 0.002, 0.020, 0.005},
     {0.002, 0.002, 0.002, 0.010, 0.005}},

    // ---- NOT TRANSPORTED. Measurements, not tolerances.
    //
    // The positive hadrons come out at the proton's level, which is the one thing this section
    // establishes: the Bragg / Bethe-Bloch path generalises across mass with nothing
    // species-specific in it.
    {"pi+", ParticleType::kPionPlus,
     {0.002, 0.002, 0.020, 0.005, 0.002},
     {0.002, 0.002, 0.010, 0.006, 0.002}},
    {"kaon+", ParticleType::kKaonPlus,
     {0.002, 0.002, 0.020, 0.010, 0.002},
     {0.002, 0.002, 0.010, 0.010, 0.002}},
    // Negative hadrons, on G4ICRU73QOModel below the boundary. The 2-100 MeV band is the air
    // anomaly above; the low bands are it too, diluted.
    {"anti_proton", ParticleType::kAntiProton,
     {0.080, 0.030, 0.020, 0.250, 0.030},
     {0.030, 0.020, 0.020, 0.150, 0.030}},
    {"pi-", ParticleType::kPionMinus,
     {0.030, 0.020, 0.030, 0.030, 0.020},
     {0.020, 0.020, 0.020, 0.020, 0.020}},
    {"kaon-", ParticleType::kKaonMinus,
     {0.080, 0.020, 0.200, 0.100, 0.030},
     {0.030, 0.010, 0.100, 0.090, 0.030}},
    // Muons: a flat 200 keV boundary rather than the mass-scaled 225 keV, and
    // G4MuBetheBlochModel above it. mu+ agrees exactly below 100 keV now; the 0.1-2 MeV band
    // straddles the boundary and mu- carries the air anomaly on top.
    {"mu+", ParticleType::kMuonPlus,
     {0.002, 0.002, 0.080, 0.010, 0.005},
     {0.002, 0.002, 0.040, 0.020, 0.005}},
    {"mu-", ParticleType::kMuonMinus,
     {0.030, 0.030, 0.120, 0.030, 0.010},
     {0.020, 0.020, 0.080, 0.040, 0.010}},
    // The base particle every non-alpha ion scales from. Charge 1, mass 938.2723 - and not
    // units::proton_mass_c2, which is 938.272013. Exact below 100 keV.
    {"GenericIon", ParticleType::kGenericIon,
     {0.002, 0.002, 0.005, 0.050, 0.002},
     {0.002, 0.002, 0.002, 0.030, 0.005}},
    // No table of its own: scaled from GenericIon by massRatio and by an effective charge
    // recomputed at every pre-step energy, which is what G4VEnergyLossProcess does under
    // `if(isIon)` and what G4EmTableUtil::CheckIon excludes the alpha from by name. The only
    // row here measuring the scaling rather than a table, and it now measures it as exact:
    //
    //     dE/dx 1-10 keV    80.9%  (own table)  ->  301%  (static ratio)  ->  0.001%
    //     range 1-10 keV       -                ->   75%                  ->  0.001%
    //
    // The middle column is worth keeping. Mapping He3 onto the alpha's table was wrong by a
    // factor of twelve; replacing that with GenericIon's table and a *static* charge ratio was
    // wrong by a factor of three and looked like progress in the high bands while getting
    // worse in the low ones. Only the third column is the mechanism Geant4 has.
    {"He3", ParticleType::kHe3,
     {0.002, 0.002, 0.002, 0.040, 0.005},
     {0.002, 0.002, 0.002, 0.025, 0.010}},
};

constexpr int kNPart = static_cast<int>(sizeof kSpecies / sizeof kSpecies[0]);

int part_index(const std::string& n) {
  for (int i = 0; i < kNPart; ++i) {
    if (n == kSpecies[i].name) { return i; }
  }
  return -1;
}

int material_of(const std::string& n) {
  if (n == "G4_AIR") { return data::kAir; }
  if (n == "G4_WATER") { return data::kWater; }
  if (n == "G4_A-150_TISSUE") { return data::kA150Tissue; }
  if (n == "G4_BONE_COMPACT_ICRU") { return data::kBoneCompact; }
  return -1;
}

struct Band {
  const char* label;
  double lo, hi;
};

struct Cell {
  int n = 0;
  double worst = 0;
  std::string where;
};

void note(Cell* cells, const Band* b, int nb, int pi, double e, double dev,
          const std::string& where) {
  if (pi < 0) { return; }
  for (int i = 0; i < nb; ++i) {
    if (e < b[i].lo || e >= b[i].hi) { continue; }
    Cell& c = cells[pi * nb + i];
    ++c.n;
    if (dev > c.worst) {
      c.worst = dev;
      c.where = where;
    }
    return;
  }
}

void report(const char* title, const Cell* cells, const Band* b, int nb) {
  std::printf("%s\n", title);
  std::printf("  %-12s", "particle");
  for (int i = 0; i < nb; ++i) { std::printf(" %12s", b[i].label); }
  std::printf("\n");
  for (int p = 0; p < kNPart; ++p) {
    std::printf("  %-12s", kSpecies[p].name);
    for (int i = 0; i < nb; ++i) {
      const Cell& c = cells[p * nb + i];
      if (c.n == 0) { std::printf(" %12s", "-"); }
      else { std::printf(" %11.3f%%", 100 * c.worst); }
    }
    std::printf("\n");
  }
  // The single worst point per particle, spelled out - a percentage says how wrong, not where,
  // and where is the first thing to look at.
  for (int p = 0; p < kNPart; ++p) {
    double w = 0;
    std::string where;
    for (int i = 0; i < nb; ++i) {
      if (cells[p * nb + i].worst > w) {
        w = cells[p * nb + i].worst;
        where = cells[p * nb + i].where;
      }
    }
    if (!where.empty()) {
      std::printf("    %-12s worst %7.3f%%  %s\n", kSpecies[p].name, 100 * w, where.c_str());
    }
  }
}

}  // namespace

int main() {
  data::Material<real_t> mats[data::kNumMaterials];
  data::build_b1_materials<real_t>(mats);
  static em::ShellTables<real_t> shell;
  em::build_shell_tables(shell);

  const char* env = std::getenv("G4GPU_ORACLE");
  const std::string dir = (env != nullptr) ? env : "ref/oracle";
  const auto rows = load(dir + "/hadron_tables.csv");
  if (rows.empty()) {
    std::printf("cannot read %s/hadron_tables.csv - run ref/oracle/run.bat first\n",
                dir.c_str());
    return 1;
  }

  int fails = 0;
  const Band bands[kNBands] = {
      {"1-10 keV", 1e-3, 1e-2}, {"10-100 keV", 1e-2, 1e-1}, {"0.1-2 MeV", 1e-1, 2.0},
      {"2-100 MeV", 2.0, 1e2},  {"0.1-10 GeV", 1e2, 1e4},
  };
  const int nb = kNBands;

  static em::HadronRangeTable<real_t> table;
  real_t cuts[data::kNumMaterials];
  for (int m = 0; m < data::kNumMaterials; ++m) { cuts[m] = mats[m].cut_electron; }
  em::build_hadron_range_table<real_t>(mats, cuts, table, &shell);

  // ---------------------------------------------------------------- 1. total dE/dx
  //
  // Through the table's own accessor rather than by calling the model directly, because for a
  // species with a base particle those are two different numbers and Geant4's is the scaled
  // one.
  static Cell dcell[kNPart * kNBands];
  int compared = 0;
  for (const Row& r : rows) {
    const int pi = part_index(r.part);
    if (pi < 0) { continue; }
    const int mi = material_of(r.mat);
    if (mi < 0 || r.dedx_total <= 0 || r.e > 1e4) { continue; }

    const real_t ours = table.dedx_for(mats[mi], kSpecies[pi].type, mi, real_t(r.e));
    const double dev = std::fabs(ours - r.dedx_total) / r.dedx_total;
    char buf[220];
    std::snprintf(buf, sizeof buf, "in %s at %.4g MeV (ours %.5g, G4 %.5g MeV/mm)",
                  r.mat.c_str(), r.e, static_cast<double>(ours), r.dedx_total);
    note(dcell, bands, nb, pi, r.e, dev, buf);
    ++compared;
  }
  report("== total dE/dx vs G4EmCalculator::GetDEDX ==", dcell, bands, nb);
  std::printf("  %d points compared\n\n", compared);

  // ---------------------------------------------------------------- 2. the range table
  static Cell rcell[kNPart * kNBands];
  int rcompared = 0;
  for (const Row& r : rows) {
    const int pi = part_index(r.part);
    if (pi < 0) { continue; }
    const int mi = material_of(r.mat);
    if (mi < 0 || r.range <= 0 || r.e > 1e4) { continue; }

    const real_t ours = table.range_for(mats[mi], kSpecies[pi].type, mi, real_t(r.e));
    const double dev = std::fabs(ours - r.range) / r.range;
    char buf[220];
    std::snprintf(buf, sizeof buf, "in %s at %.4g MeV (ours %.5g, G4 %.5g mm)", r.mat.c_str(),
                  r.e, static_cast<double>(ours), r.range);
    note(rcell, bands, nb, pi, r.e, dev, buf);
    ++rcompared;
  }
  report("== range vs G4EmCalculator::GetRange ==", rcell, bands, nb);
  std::printf("  %d points compared\n\n", rcompared);

  // ---------------------------------------------------------------- 3. the inverse
  //
  // Per particle rather than per species, so the scaled path is exercised too: for He3 this
  // round-trips through the mass and charge ratios in both directions, and a ratio applied the
  // wrong way round there cancels in nothing else.
  // It does not close exactly, and it cannot: Geant4's range table and its inverse are two
  // *independent* cubic splines through the same points (BuildRangeTable then
  // BuildInverseRangeTable, each calling FillSecondDerivatives). Between grid points they are
  // not exact inverses of each other, and reproducing Geant4 means reproducing that. What must
  // stay small is the residual for the species actually transported, on the grid the transport
  // uses; the limit below is per particle for that reason.
  {
    std::printf("== range -> energy round trip ==\n");
    for (int p = 0; p < kNPart; ++p) {
      const ParticleType t = kSpecies[p].type;
      double worst = 0;
      std::string where;
      for (int mi = 0; mi < data::kNumMaterials; ++mi) {
        for (double e = 1e-2; e < 5e3; e *= 1.3) {
          const real_t r = table.range_for(mats[mi], t, mi, real_t(e));
          const real_t back = table.energy_from_range_for(mats[mi], t, mi, r, real_t(e));
          const double dev = std::fabs(back - e) / e;
          if (dev > worst) {
            worst = dev;
            char buf[200];
            std::snprintf(buf, sizeof buf, "material %d at %.4g MeV -> %.6g", mi, e,
                          static_cast<double>(back));
            where = buf;
          }
        }
      }
      const bool transported =
          (t == ParticleType::kProton || t == ParticleType::kAlpha);
      const double limit = transported ? 2e-3 : 1e-2;
      std::printf("  %-12s worst %.4f%%  (%s)\n", kSpecies[p].name, 100 * worst,
                  where.c_str());
      if (worst > limit) {
        std::printf("  FAIL: %s range inverse is off by %.3f%%, limit %.1f%%\n",
                    kSpecies[p].name, 100 * worst, 100 * limit);
        ++fails;
      }
    }
    std::printf("\n");
  }

  // ---------------------------------------------------------------- verdicts
  for (int p = 0; p < kNPart; ++p) {
    for (int i = 0; i < nb; ++i) {
      const Cell& d = dcell[p * nb + i];
      if (d.n > 0 && d.worst > kSpecies[p].dedx[i]) {
        std::printf("  FAIL: %s dE/dx in %s is %.2f%%, limit %.1f%%  %s\n", kSpecies[p].name,
                    bands[i].label, 100 * d.worst, 100 * kSpecies[p].dedx[i], d.where.c_str());
        ++fails;
      }
      const Cell& r = rcell[p * nb + i];
      if (r.n > 0 && r.worst > kSpecies[p].range[i]) {
        std::printf("  FAIL: %s range in %s is %.2f%%, limit %.1f%%  %s\n", kSpecies[p].name,
                    bands[i].label, 100 * r.worst, 100 * kSpecies[p].range[i], r.where.c_str());
        ++fails;
      }
    }
  }

  // Every species must actually have been found in the oracle. One silently absent - renamed
  // in the dump, or never added to it - would pass every check above by comparing nothing, and
  // that is the failure mode this whole test was rewritten to remove.
  for (int p = 0; p < kNPart; ++p) {
    int n = 0;
    for (int i = 0; i < nb; ++i) { n += dcell[p * nb + i].n; }
    if (n == 0) {
      std::printf("  FAIL: no oracle rows for %s - is it in ref/dump/g4dump.cc?\n",
                  kSpecies[p].name);
      ++fails;
    }
  }
  if (compared == 0 || rcompared == 0) {
    std::printf("  FAIL: nothing compared\n");
    ++fails;
  }

  std::printf("\n%s (%d failures)\n", fails ? "FAILED" : "PASSED", fails);
  return fails ? 1 : 0;
}
