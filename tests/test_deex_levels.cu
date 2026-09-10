// The PhotonEvaporation5.7 reader against G4LevelReader's own parse of the same 3110 files.
//
// Two oracles, and the second is the wide one:
//
//   deex_levelmax.csv   every (Z, A) in G4NuclearLevelData's AMIN/AMAX window - 3188 rows -
//                       with the number of levels, the highest level energy, and the ground
//                       state's lifetime. Comparing the level COUNT for every nuclide in the
//                       dataset is what catches a field-width or a transition-consumption bug:
//                       mis-consume the ten internal-conversion coefficients of one transition
//                       and every following level index is wrong, the reader's `i1 != i` guard
//                       fires, and that nuclide's count collapses. A sample of forty nuclides
//                       would miss it in the other three thousand.
//   deex_levels.csv     every level and every transition of forty-six nuclides across the
//                       table - energies, lifetimes, 2J, parity, floating index, and per
//                       transition the final level, the multipolarity word, the gamma share
//                       and the normalised cumulative probability.
//
// Also compared here, and it is a finding rather than a check: G4NuclearLevelData's compiled
// LEVELMAX table against the level data actually installed. Geant4's comment says the table
// came from PhotonEvaporation **5.2** and the installed dataset is **5.7**. Both are
// load-bearing - the compiled table gates G4NuclearLevelData::GetLevelEnergy and decides which
// fragments enter the Fermi break-up pool, the read data drives the cascade - so this test
// measures the disagreement and asserts its size rather than assuming there is none.
//
// Anti-vacuity: the run in the commit message removed, one at a time, the ICC consumption, the
// 632-level cap's exactness, the cumulative-probability normalisation, and the
// `spin > 48 -> 0` rule.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

#include "data/level_data.cuh"
#include "host/g4data.cuh"
#include "physics/hadronic/deexcitation/corrections.cuh"

using namespace g4gpu;

namespace {

struct Cell {
  int n = 0;
  double worst = 0;
  std::string where;
};

void note(Cell& c, double dev, const char* what) {
  ++c.n;
  if (dev > c.worst) {
    c.worst = dev;
    c.where = what;
  }
}

int fails = 0;

void report(const Cell& c, double tol, const char* label) {
  const bool ok = (c.n > 0 && c.worst <= tol);
  std::printf("  %-38s %7d pts  worst %.3g  %s\n", label, c.n, c.worst, ok ? "OK" : "FAIL");
  if (!ok) {
    ++fails;
    std::printf("      worst at %s\n", c.where.c_str());
  }
}

double dev_of(double ours, double g4, double floor_abs) {
  if (std::fabs(g4) > floor_abs) { return std::fabs(ours - g4) / std::fabs(g4); }
  return (std::fabs(ours) > floor_abs) ? 1.0 : 0.0;
}

std::vector<std::string> read_lines(const std::string& path) {
  std::vector<std::string> out;
  FILE* f = std::fopen(path.c_str(), "r");
  if (f == nullptr) { return out; }
  char line[4096];
  while (std::fgets(line, sizeof line, f) != nullptr) { out.push_back(line); }
  std::fclose(f);
  return out;
}

}  // namespace

int main() {
  const char* env = std::getenv("G4GPU_ORACLE");
  const std::string dir = (env != nullptr) ? env : "ref/oracle";
  const std::string pe = host::g4photon_evaporation_dir();
  if (pe.empty()) {
    std::printf("PhotonEvaporation dataset not found - g4data.cuh could not resolve "
                "G4LEVELGAMMADATA\n");
    return 1;
  }
  std::printf("dataset: %s\n", pe.c_str());

  // Build the whole table, exactly as G4ExcitationHandler::SetParameters does through
  // UploadNuclearLevelData - except for every Z rather than the geometry's, because the
  // oracle asked every Z.
  data::LevelTableStorage st;
  data::read_all_level_data(
      st, pe, data::kLevelZMax,
      [](int Z, int A) { return deex::shell_correction(A, Z); },
      [](int Z, int A) { return deex::level_manager_level_density(Z, A); });
  const data::LevelTable t = st.view();
  std::printf("read %d managers, %d levels, %d transitions\n", t.n_managers, t.n_levels,
              t.n_transitions);

  // The reader's own notes: repaired data and refusals. Geant4 prints the same repairs, so a
  // note here is expected; a REFUSED one is not, and fails.
  int refused = 0;
  for (const std::string& s : st.notes) {
    std::printf("  note: %s\n", s.c_str());
    if (s.rfind("REFUSED", 0) == 0 || s.rfind("MISSING", 0) == 0) { ++refused; }
  }
  if (refused > 0) {
    std::printf("  %d refusals above\n", refused);
    ++fails;
  }

  // ---------------------------------------------------------------- every nuclide's shape
  {
    const auto lines = read_lines(dir + "/deex_levelmax.csv");
    if (lines.size() < 2) {
      std::printf("cannot read %s/deex_levelmax.csv - run ref/oracle/run.bat tables\n",
                  dir.c_str());
      return 1;
    }
    Cell compiled, readmax, lifetime0;
    int rows = 0, nlev_bad = 0, exists_bad = 0;
    // The two answers to "what is this nuclide's highest level". Not a check - a measurement.
    int disagree = 0, compiled_lower = 0, compiled_zero = 0;
    double worst_gap = 0;
    std::string worst_gap_at;
    for (std::size_t i = 1; i < lines.size(); ++i) {
      int Z = 0, A = 0, nlev = 0;
      double cm = 0, rm = 0, lt = 0;
      if (std::sscanf(lines[i].c_str(), "%d,%d,%lf,%lf,%d,%lf", &Z, &A, &cm, &rm, &nlev,
                      &lt) != 6) {
        continue;
      }
      ++rows;
      char buf[128];
      std::snprintf(buf, sizeof buf, "Z=%d A=%d", Z, A);
      // The compiled table is a straight transcription and must be exact.
      note(compiled, dev_of(deex::level_manager_level_density(Z, A) * 0.0 +
                                data::compiled_max_level_energy(Z, A),
                            cm, 1e-12), buf);
      const int m = data::find_manager(t, Z, A);
      const bool g4_has = (nlev > 0);
      if ((m >= 0) != g4_has) {
        ++exists_bad;
        if (exists_bad < 6) {
          std::printf("  manager existence differs at %s: ours %d, G4 %d levels\n", buf,
                      m >= 0 ? 1 : 0, nlev);
        }
        continue;
      }
      if (m < 0) { continue; }
      if (t.managers[m].n_levels != nlev) {
        ++nlev_bad;
        if (nlev_bad < 8) {
          std::printf("  level count differs at %s: ours %d, G4 %d\n", buf,
                      t.managers[m].n_levels, nlev);
        }
      }
      note(readmax, dev_of(data::read_max_level_energy(t, m), rm, 1e-12), buf);
      note(lifetime0, dev_of(data::level_lifetime(t, m, 0), lt, 1e-30), buf);

      if (std::fabs(cm - rm) > 1e-6) {
        ++disagree;
        if (cm < rm) { ++compiled_lower; }
        if (cm == 0.0) { ++compiled_zero; }
        if (std::fabs(cm - rm) > worst_gap) {
          worst_gap = std::fabs(cm - rm);
          char b2[160];
          std::snprintf(b2, sizeof b2, "Z=%d A=%d compiled %.6g MeV, read %.6g MeV", Z, A, cm,
                        rm);
          worst_gap_at = b2;
        }
      }
    }
    std::printf("levelmax: %d rows, manager-existence mismatches %d, level-count mismatches "
                "%d  %s\n", rows, exists_bad, nlev_bad,
                (exists_bad || nlev_bad) ? "FAIL" : "OK");
    if (exists_bad || nlev_bad) { ++fails; }
    report(compiled, 1e-12, "compiled LEVELMAX (PhotonEvap 5.2)");
    report(readmax, 1e-12, "read max level energy (5.7)");
    report(lifetime0, 1e-9, "ground-state lifetime");
    std::printf("  FINDING: compiled LEVELMAX disagrees with the installed data for %d of the "
                "%d nuclides\n           that have levels (%d of them with the compiled value "
                "LOWER, %d with it zero);\n           worst %s\n",
                disagree, readmax.n, compiled_lower, compiled_zero, worst_gap_at.c_str());
  }

  // ---------------------------------------------------------------- levels and transitions
  {
    const auto lines = read_lines(dir + "/deex_levels.csv");
    Cell energy, life, tr_prob, tr_cum, tr_ratio;
    int rows = 0, spin_bad = 0, parity_bad = 0, float_bad = 0, ntrans_bad = 0, final_bad = 0,
        type_bad = 0, missing = 0;
    for (std::size_t i = 1; i < lines.size(); ++i) {
      int Z = 0, A = 0, lev = 0, spin2 = 0, par = 0, fl = 0, nt = 0, j = 0, fi = 0, ty = 0;
      double e = 0, lt = 0, gp = 0, cp = 0, mr = 0;
      if (std::sscanf(lines[i].c_str(), "%d,%d,%d,%lf,%lf,%d,%d,%d,%d,%d,%d,%d,%lf,%lf,%lf",
                      &Z, &A, &lev, &e, &lt, &spin2, &par, &fl, &nt, &j, &fi, &ty, &gp, &cp,
                      &mr) != 15) {
        continue;
      }
      ++rows;
      const int m = data::find_manager(t, Z, A);
      if (lev < 0) {
        // The oracle says this nuclide has no manager.
        if (m >= 0) { ++missing; }
        continue;
      }
      if (m < 0 || lev >= t.managers[m].n_levels) {
        ++missing;
        continue;
      }
      char buf[160];
      std::snprintf(buf, sizeof buf, "Z=%d A=%d level %d", Z, A, lev);
      note(energy, dev_of(data::level_energy(t, m, lev), e, 1e-12), buf);
      note(life, dev_of(data::level_lifetime(t, m, lev), lt, 1e-30), buf);
      if (data::level_spin_two(t, m, lev) != spin2) { ++spin_bad; }
      if (data::level_parity(t, m, lev) != par) { ++parity_bad; }
      if (data::level_floating(t, m, lev) != fl) { ++float_bad; }
      if (data::level_ntrans(t, m, lev) != nt) {
        ++ntrans_bad;
        if (ntrans_bad < 6) {
          std::printf("  ntrans differs at %s: ours %d, G4 %d\n", buf,
                      data::level_ntrans(t, m, lev), nt);
        }
      }
      if (j < 0 || nt == 0) { continue; }
      if (j >= data::level_ntrans(t, m, lev)) { ++missing; continue; }
      const data::LevelTransition& tr = data::level_transition(t, m, lev, j);
      char b2[192];
      std::snprintf(b2, sizeof b2, "Z=%d A=%d level %d transition %d", Z, A, lev, j);
      if (data::transition_final_index(tr) != fi) { ++final_bad; }
      if (data::transition_type(tr) != ty) { ++type_bad; }
      // The transition probabilities are G4float in Geant4, so the tolerance is single
      // precision - and the dump printed them with %.9g, which is more digits than a float
      // carries. Anything above ~1e-7 relative is a parse error, not rounding.
      note(tr_prob, dev_of(tr.prob, gp, 1e-30), b2);
      note(tr_cum, dev_of(tr.cum_prob, cp, 1e-30), b2);
      note(tr_ratio, dev_of(tr.ratio, mr, 1e-30), b2);
    }
    std::printf("levels: %d rows; mismatches - spin2 %d, parity %d, floating %d, ntrans %d, "
                "final %d, type %d, out-of-range %d\n", rows, spin_bad, parity_bad, float_bad,
                ntrans_bad, final_bad, type_bad, missing);
    if (spin_bad || parity_bad || float_bad || ntrans_bad || final_bad || type_bad || missing) {
      ++fails;
    }
    report(energy, 1e-12, "level energy");
    report(life, 1e-9, "level lifetime");
    report(tr_prob, 1e-6, "transition gamma probability");
    report(tr_cum, 1e-6, "transition cumulative probability");
    report(tr_ratio, 1e-6, "transition multipolarity ratio");
  }

  std::printf("%s\n", fails == 0 ? "ALL OK" : "FAILURES ABOVE");
  return fails == 0 ? 0 : 1;
}
