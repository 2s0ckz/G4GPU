// P26: G4GammaGeneralProcess's tables - every one `BuildPhysicsTable` fills, and the
// sub-processes' own lambda tables they are built from - against the running Geant4's.
//
// `ref/gammagp/gammagp.cc` has the running process store its tables in the binary format, which
// keeps every double, and prints them node by node (`ref/oracle/gammagp_tables.csv`): the general
// tables as kind=node, `LogVectorValue` at 31 energies as kind=at, each sub-process's model
// (`G4EmCalculator`) as kind=model, and since P26 each sub-process's own stored lambda vectors
// as kind=subnode. This test builds all of it from this port's models with
// `host::build_gamma_general_host_tables` and holds it to those numbers:
//
//   1. each sub-process's lambda and prim vectors: the edges and every node energy to the bit,
//      every value to the models' agreement (the port's model against G4EmCalculator's at the
//      same energies is measured first, and the tables may differ by no more than that);
//   2. every general table 0-14, every node: energies to the bit, values as in 1 - with zone 3's
//      conversion above 30 GeV, where this port keeps the parameterisation and Geant4 integrates
//      (docs/PORTED.md 4.1), measured and reported separately;
//   3. `had::gamma_general_value` - the reader `step_gamma` calls - at the 31 energies, against
//      Geant4's own `LogVectorValue` on its own stored vector;
//   4. docs/RISK.md V225's table, completed: every zone total and selection fraction, the tables
//      against the models' sums at the photon's energy, over the 31 energies and four materials -
//      the size of what P26 changes in the photon's step;
//   5. `had::gamma_general_select` walks each zone as `PostStepDoIt` does: over a fine grid of q,
//      the fraction each process takes is its share of the table.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "data/materials.cuh"
#include "data/photoelectric_data.cuh"
#include "data/rayleigh_data.cuh"
#include "host/g4data.cuh"
#include "host/hadronic_upload.cuh"
#include "physics/em/gamma_processes.cuh"

using namespace g4gpu;
using real_t = double;

namespace {

int g_fails = 0;
void fail(const std::string& msg) {
  std::printf("  FAIL: %s\n", msg.c_str());
  ++g_fails;
}

std::string oracle_dir() {
  const char* e = std::getenv("G4GPU_ORACLE");
  return (e != nullptr && e[0] != '\0') ? std::string(e) : std::string("ref/oracle");
}

std::vector<std::string> split(const std::string& s) {
  std::vector<std::string> out;
  std::string cur;
  for (const char c : s) {
    if (c == ',') {
      out.push_back(cur);
      cur.clear();
    } else if (c != '\n' && c != '\r') {
      cur += c;
    }
  }
  out.push_back(cur);
  return out;
}

const char* kMatName[data::kNumMaterials] = {"G4_AIR", "G4_WATER", "G4_A-150_TISSUE",
                                            "G4_BONE_COMPACT_ICRU"};

struct Row {
  int mat = -1, table = -1, index = -1;
  std::string kind;
  double e = 0, v = 0;
};

std::vector<Row> load_rows(const std::string& path) {
  std::vector<Row> out;
  FILE* f = std::fopen(path.c_str(), "r");
  if (f == nullptr) { return out; }
  static char line[8192];
  while (std::fgets(line, sizeof line, f) != nullptr) {
    const std::vector<std::string> v = split(line);
    if (v.size() != 6) { continue; }
    Row r;
    for (int m = 0; m < data::kNumMaterials; ++m) {
      if (v[0] == kMatName[m]) { r.mat = m; }
    }
    if (r.mat < 0) { continue; }
    r.kind = v[2];
    if (r.kind != "node" && r.kind != "at" && r.kind != "model" && r.kind != "lambda"
        && r.kind != "subnode" && r.kind != "subhead") {
      continue;
    }
    r.table = std::atoi(v[1].c_str());
    r.index = std::atoi(v[3].c_str());
    r.e = std::strtod(v[4].c_str(), nullptr);
    r.v = std::strtod(v[5].c_str(), nullptr);
    out.push_back(r);
  }
  std::fclose(f);
  return out;
}

double rel(double a, double b) {
  if (a == b) { return 0.0; }
  const double d = std::fabs(b);
  return (d > 0) ? std::fabs(a - b) / d : std::fabs(a - b);
}

}  // namespace

int main() {
  const std::string dir = oracle_dir();
  static data::Material<real_t> mats[data::kNumMaterials];
  data::build_b1_materials<real_t>(mats);

  const std::string phot_dir = host::g4emlow_subdir("epics2017/phot", "pe-cs-1.dat");
  const std::string rayl_dir = host::g4emlow_subdir("epics2017/rayl", "re-cs-1.dat");
  int zlist[32];
  int nzl = 0;
  for (int m = 0; m < data::kNumMaterials; ++m) {
    for (int i = 0; i < mats[m].n_elements; ++i) {
      const int z = static_cast<int>(mats[m].z[i] + 0.5);
      bool seen = false;
      for (int k = 0; k < nzl; ++k) { seen |= (zlist[k] == z); }
      if (!seen) { zlist[nzl++] = z; }
    }
  }
  static data::PhotoElectricTable<real_t> pe{};
  static std::vector<real_t> pte, ptv;
  static data::RayleighTable<real_t> rt{};
  static std::vector<real_t> rte, rtv;
  const bool pe_ok = data::load_photoelectric<real_t>(phot_dir, zlist, nzl, pe, pte, ptv);
  pe.table_e = pte.data();
  pe.table_v = ptv.data();
  const bool ra_ok = data::load_rayleigh<real_t>(rayl_dir, zlist, nzl, rt, rte, rtv);
  rt.table_e = rte.data();
  rt.table_v = rtv.data();
  if (!pe_ok || !ra_ok) {
    std::printf("could not load the photoelectric / Rayleigh tables - nothing here can be tested\n");
    return 1;
  }
  static host::EmExtraHostTables<real_t> ht;
  host::build_emextra_host_tables<real_t>(ht, mats, data::kNumMaterials, true, &pe, &rt, true);
  if (!ht.gamma_ok || !ht.gg_ok) {
    std::printf("the gamma data set or the general tables are absent - nothing here can be tested\n");
    return 1;
  }
  const std::vector<Row> rows = load_rows(dir + "/gammagp_tables.csv");
  if (rows.empty()) {
    fail("cannot read " + dir + "/gammagp_tables.csv - run ref/oracle/run.bat tables");
    std::printf("\n%d failure(s)\n", g_fails);
    return 1;
  }

  // ============================================================================================
  // 0. The port's models against G4EmCalculator's at the 31 energies: the floor for 1 and 2.
  // ============================================================================================
  std::printf("== 0. this port's four models against G4EmCalculator's (kind=model) ==\n");
  double worst_model[4] = {0, 0, 0, 0};
  {
    int n = 0;
    for (const Row& r : rows) {
      if (r.kind != "model" || r.table < 100 || r.table > 103) { continue; }
      const int k = r.table - 100;
      const em::GammaXS<real_t> xs = em::gamma_macroscopic_xs<real_t>(
          mats[r.mat], r.e, &pe, &rt, true, true, false);
      // Conversion through its whole range: the parameterisation below 30 GeV, the integral above
      // (em::pair_xs_per_atom_rel), each element at the material's LPM energy.
      double conv = 0;
      for (int i = 0; i < mats[r.mat].n_elements; ++i) {
        conv += mats[r.mat].n_atoms[i] * em::pair_xs_per_atom_rel<real_t>(
                    r.e, mats[r.mat].z[i], mats[r.mat].radiation_length * em::lpm_constant<real_t>());
      }
      const double mine = (k == 0) ? xs.compton : (k == 1) ? conv
                          : (k == 2) ? xs.photoelectric : xs.rayleigh;
      const double d = rel(mine, r.v);
      if (d > worst_model[k]) { worst_model[k] = d; }
      ++n;
    }
    std::printf("  %d points; worst relative difference compt %.2e conv %.2e phot %.2e Rayl %.2e\n",
                n, worst_model[0], worst_model[1], worst_model[2], worst_model[3]);
    if (n != 4 * 44 * data::kNumMaterials) { fail("expected 704 model rows (44 energies)"); }
    for (int k = 0; k < 4; ++k) {
      if (worst_model[k] > 1e-13) { fail("a model differs from Geant4's by more than 1e-13"); }
    }
  }
  // The floor every table comparison below is held to: the models agree to this, and a table is
  // a sum and a ratio of them, so a few times it.
  const double kTol = 1e-12;

  // ============================================================================================
  // 1. The sub-processes' own lambda tables
  // ============================================================================================
  std::printf("== 1. each sub-process's lambda and prim vectors against its stored ones ==\n");
  {
    const char* names[4] = {"compt", "conv", "phot", "Rayl"};
    int nodes = 0, grid_bad = 0, val_bad = 0, heads = 0, head_bad = 0;
    double worst = 0;
    for (const Row& r : rows) {
      if (r.kind != "subnode" && r.kind != "subhead") { continue; }
      const int k = (r.table - 300) / 10, prim = (r.table - 300) % 10;
      if (k < 0 || k > 3 || prim < 0 || prim > 1) { continue; }
      const em::EmProcessLambda& sub = ht.gg_sub[static_cast<std::size_t>(r.mat) * 4 + k];
      const bool has = prim ? sub.has_prim : sub.has_lambda;
      const em::LogVecHost& vec = prim ? sub.prim : sub.lambda;
      if (!has) {
        ++head_bad;
        std::printf("  %s %s: Geant4 stores a %s vector this port did not build\n",
                    kMatName[r.mat], names[k], prim ? "prim" : "lambda");
        continue;
      }
      if (r.kind == "subhead") {
        ++heads;
        if (r.index == -1 && (vec.edge_min != r.e || vec.n() != static_cast<int>(r.v))) {
          ++head_bad;
          std::printf("  %s %s %s: edgeMin %.17g / %d nodes, Geant4 %.17g / %d\n",
                      kMatName[r.mat], names[k], prim ? "prim" : "lambda", vec.edge_min, vec.n(),
                      r.e, static_cast<int>(r.v));
        }
        if (r.index == -2 && vec.edge_max != r.e) { ++head_bad; }
        continue;
      }
      ++nodes;
      if (r.index >= vec.n()) {
        ++grid_bad;
        continue;
      }
      if (vec.e[static_cast<std::size_t>(r.index)] != r.e) { ++grid_bad; }
      const double d = rel(vec.v[static_cast<std::size_t>(r.index)], r.v);
      if (d > worst) { worst = d; }
      if (d > kTol) {
        if (val_bad < 8) {
          std::printf("  %s %s %s node %d (%.6g MeV): %.17g, Geant4 %.17g\n", kMatName[r.mat],
                      names[k], prim ? "prim" : "lambda", r.index, r.e,
                      vec.v[static_cast<std::size_t>(r.index)], r.v);
        }
        ++val_bad;
      }
    }
    // Vectors this port built that Geant4 did not store would be the other half of a mismatch.
    int built = 0;
    for (const em::EmProcessLambda& s : ht.gg_sub) { built += int(s.has_lambda) + int(s.has_prim); }
    std::printf("  %d stored vectors (%d headers), %d built; %d nodes, grid mismatches %d, "
                "values beyond %.0e: %d, worst %.2e\n", heads / 2, heads, built, nodes, grid_bad,
                kTol, val_bad, worst);
    if (nodes == 0) { fail("no subnode rows - regenerate the oracle"); }
    if (built != heads / 2) { fail("the port builds a different set of vectors from Geant4's"); }
    if (head_bad != 0) { fail("vector edges or lengths differ"); }
    if (grid_bad != 0) { fail("node energies differ"); }
    if (val_bad != 0) { fail("lambda-vector values differ beyond the models' agreement"); }
  }

  // ============================================================================================
  // 2. Every general table, node by node
  // ============================================================================================
  std::printf("== 2. tables 0-14 node by node against G4GammaGeneralProcess's ==\n");
  {
    int nodes = 0, grid_bad = 0, val_bad = 0, hi_conv = 0;
    double worst = 0, worst_hi = 0;
    int per_table[had::kGammaGeneralNumTables] = {};
    for (const Row& r : rows) {
      if (r.kind != "node" || r.table < 0 || r.table >= had::kGammaGeneralNumTables) { continue; }
      const int k = r.table;
      ++per_table[k];
      const int z = had::gamma_general_table_zone(k);
      const int n = static_cast<int>(ht.gg_e[z].size());
      if (ht.gg_t[k].empty() || r.index >= n) {
        ++grid_bad;
        continue;
      }
      ++nodes;
      if (ht.gg_e[z][static_cast<std::size_t>(r.index)] != r.e) { ++grid_bad; }
      const double mine = ht.gg_t[k][static_cast<std::size_t>(r.mat) * n + r.index];
      // Fractions near zero (table 13/14 off the giant resonance) are compared absolutely.
      const bool lam = (k == 0 || k == 2 || k == 6 || k == 10);
      const double d = lam ? rel(mine, r.v) : std::fabs(mine - r.v);
      if (z == 3 && r.e >= 3.0e4) {
        // Above 30 GeV conversion is the integrated model (em::pair_xs_per_atom_rel): reported apart.
        ++hi_conv;
        if (d > worst_hi) { worst_hi = d; }
      }
      if (d > worst) { worst = d; }
      if (d > kTol) {
        if (val_bad < 10) {
          std::printf("  %s table %d node %d (%.6g MeV): %.17g, Geant4 %.17g\n", kMatName[r.mat],
                      k, r.index, r.e, mine, r.v);
        }
        ++val_bad;
      }
    }
    std::printf("  %d nodes, %d below 30 GeV: grid mismatches %d, beyond %.0e: %d, worst %.2e\n",
                nodes, nodes - hi_conv, grid_bad, kTol, val_bad, worst);
    std::printf("  %d zone-3 nodes from 30 GeV up (conversion integrated, LPM above 100 GeV"
                "): worst %.3e\n", hi_conv, worst_hi);
    for (int k = 0; k < had::kGammaGeneralNumTables; ++k) {
      if (k == 5) {
        if (per_table[k] != 0) { fail("Geant4 stored a table 5"); }
        continue;
      }
      if (per_table[k] == 0) { fail("Geant4's table " + std::to_string(k) + " is missing"); }
    }
    if (grid_bad != 0) { fail("table node energies differ"); }
    if (val_bad != 0) { fail("table values differ beyond the models' agreement"); }
    // Table 14 is table 13 while sigM is 0 (V207): every node, every material.
    int t1314 = 0;
    for (std::size_t i = 0; i < ht.gg_t[13].size(); ++i) { t1314 += (ht.gg_t[13][i] != ht.gg_t[14][i]); }
    std::printf("  table 14 against table 13: %d nodes differ\n", t1314);
    if (t1314 != 0) { fail("table 14 is not table 13 with no muon-pair process"); }
  }

  // ============================================================================================
  // 3. The device reader at the 31 energies against Geant4's LogVectorValue
  // ============================================================================================
  std::printf("== 3. had::gamma_general_value against LogVectorValue (kind=at) ==\n");
  {
    const had::EmExtraTables<real_t> view = ht.view();
    const had::GammaGeneralTable9<real_t>& t = *view.p9;
    int n = 0, bad = 0;
    double worst = 0;
    for (const Row& r : rows) {
      if (r.kind != "at" || r.table < 0 || r.table >= had::kGammaGeneralNumTables) { continue; }
      const int k = r.table;
      const double mine = had::gamma_general_value<real_t>(t, k, r.mat, r.e, std::log(r.e));
      const bool lam = (k == 0 || k == 2 || k == 6 || k == 10);
      const double d = lam ? rel(mine, r.v) : std::fabs(mine - r.v);
      ++n;
      if (d > worst) { worst = d; }
      if (d > kTol) {
        if (bad < 10) {
          std::printf("  %s table %d at %.6g MeV: %.17g, Geant4 %.17g\n", kMatName[r.mat], k, r.e,
                      mine, r.v);
        }
        ++bad;
      }
    }
    std::printf("  %d values, beyond %.0e: %d, worst %.2e\n", n, kTol, bad, worst);
    if (n == 0) { fail("no kind=at rows"); }
    if (bad != 0) { fail("the reader differs from LogVectorValue"); }
  }

  // ============================================================================================
  // 4. V225's table completed: the tables against the models' sums at the photon's energy
  // ============================================================================================
  std::printf("== 4. the tables against the models at the photon's energy (what P26 changes) ==\n");
  {
    const had::EmExtraTables<real_t> view = ht.view();
    const had::GammaGeneralTable9<real_t>& t = *view.p9;
    const double es[] = {0.002, 0.01, 0.03, 0.05, 0.1, 0.149, 0.2, 0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2, 2.5, 3, 3.9, 4.5, 5.4, 6, 7, 8,
                           10, 11.2, 12.5, 14, 17, 20, 21, 22, 23, 25, 30, 40, 60, 80, 99.9, 100, 150,
                         300, 1000, 1e4, 1e5, 1e6, 1e7};
    std::printf("  total: (table - models)/models, %%; then the largest |table - model| of the "
                "zone's EM fractions\n");
    std::printf("  %8s %4s %10s %10s %10s %10s   %9s\n", "E(MeV)", "zone", kMatName[0], kMatName[1],
                "A-150", "bone", "fractions");
    double worst_total = 0, worst_frac = 0;
    for (const double e : es) {
      const int zone = had::gamma_general_zone<real_t>(e);
      double dt[4] = {}, df = 0;
      for (int m = 0; m < data::kNumMaterials; ++m) {
        const em::GammaXS<real_t> xs = em::gamma_macroscopic_xs<real_t>(mats[m], e, &pe, &rt, true,
                                                                       true, true);
        hadronic::xs::MaterialXs<real_t> mxs{};
        const double sig_n = (zone == 2 || zone == 3)
                                 ? had::emextra_xs_per_volume<real_t>(
                                       view, had::EmExtraProcess::kPhotonNuclear, m, mats[m], e, mxs)
                                 : 0.0;
        const had::GammaGeneralRead<real_t> rd = had::gamma_general_read<real_t>(t, zone, m, e);
        const double sum = xs.total + sig_n;
        const double total = (zone == 0) ? rd.lambda + xs.photoelectric : rd.lambda;
        dt[m] = 100.0 * (total - sum) / sum;
        double f[3] = {0, 0, 0}, g[3] = {0, 0, 0};
        if (zone == 0) {
          f[0] = rd.p0; g[0] = xs.rayleigh / (xs.compton + xs.rayleigh);
        } else if (zone == 1) {
          f[0] = rd.p0; g[0] = xs.photoelectric / sum;
          f[1] = rd.p1; g[1] = (xs.photoelectric + xs.compton) / sum;
        } else if (zone == 2) {
          f[0] = rd.p0; g[0] = xs.pair / sum;
          f[1] = rd.p1; g[1] = (xs.pair + xs.compton) / sum;
          f[2] = rd.p2; g[2] = (xs.pair + xs.compton + xs.photoelectric) / sum;
        } else if (zone == 3) {
          f[0] = rd.p0; g[0] = (xs.compton + xs.photoelectric + sig_n) / sum;
          f[1] = rd.p1; g[1] = (xs.photoelectric + sig_n) / sum;
          f[2] = rd.p2; g[2] = sig_n / sum;
        }
        for (int i = 0; i < 3; ++i) {
          if (std::fabs(f[i] - g[i]) > df) { df = std::fabs(f[i] - g[i]); }
        }
        if (std::fabs(dt[m]) > worst_total) { worst_total = std::fabs(dt[m]); }
      }
      if (df > worst_frac) { worst_frac = df; }
      std::printf("  %8.4g %4d %+9.3f%% %+9.3f%% %+9.3f%% %+9.3f%%   %9.2e\n", e, zone, dt[0],
                  dt[1], dt[2], dt[3], df);
    }
    std::printf("  worst |total| %.3f%%, worst |fraction| %.2e\n", worst_total, worst_frac);
    // Not a pass/fail on the size - it is the measurement - but the reading must not be vacuous:
    // V225 measured -0.377% at 1.2 MeV in water from Geant4's own tables, and these are those
    // tables, so the same number has to come out of this port's.
    if (worst_total < 0.3) { fail("the tables are indistinguishable from the models - V225's "
                                  "-0.38% at 1.2 MeV is not here"); }
  }

  // ============================================================================================
  // 5. The selection walk
  // ============================================================================================
  std::printf("== 5. had::gamma_general_select: each process's fraction of a q grid ==\n");
  {
    const had::EmExtraTables<real_t> view = ht.view();
    const had::GammaGeneralTable9<real_t>& t = *view.p9;
    const double es[6] = {0.05, 0.5, 1.5, 22.0, 150.0, 5000.0};
    const int nq = 2000000;
    int bad = 0;
    for (const double e : es) {
      for (int m = 0; m < data::kNumMaterials; ++m) {
        const int zone = had::gamma_general_zone<real_t>(e);
        const had::GammaGeneralRead<real_t> rd = had::gamma_general_read<real_t>(t, zone, m, e);
        const em::GammaXS<real_t> xs = em::gamma_macroscopic_xs<real_t>(mats[m], e, &pe, &rt);
        const double pe_l = xs.photoelectric;
        const double lam = (zone == 0) ? rd.lambda + pe_l : rd.lambda;
        long long c[6] = {};
        for (int i = 0; i < nq; ++i) {
          const double q = (i + 0.5) / nq;
          bool nuc = false, slice = false;
          const em::GammaProcess p =
              had::gamma_general_select<real_t>(rd, zone, lam, pe_l, q, true, nuc, slice);
          c[nuc ? 5 : static_cast<int>(p)] += 1;
          if (slice) { c[0] += 1; }  // the zone-3 slice: counted beside its conversion
        }
        double want[6] = {};
        if (zone == 0) {
          want[3] = pe_l / lam;
          want[4] = (lam - pe_l) * rd.p0 / lam;
          want[1] = 1 - want[3] - want[4];
        } else if (zone == 1) {
          want[3] = rd.p0; want[1] = rd.p1 - rd.p0; want[4] = 1 - rd.p1;
        } else if (zone == 2) {
          want[2] = rd.p0; want[1] = rd.p1 - rd.p0; want[3] = rd.p2 - rd.p1; want[5] = 1 - rd.p2;
        } else {
          want[2] = 1 - rd.p0; want[1] = rd.p0 - rd.p1; want[3] = rd.p1 - rd.p2;
          want[0] = rd.p2;  // the slice, which is ALSO in want[2]'s count below
          want[2] += rd.p2;
        }
        for (int k = 0; k < 6; ++k) {
          const double got = double(c[k]) / nq;
          if (std::fabs(got - want[k]) > 2.0 / nq) {
            if (bad < 10) {
              std::printf("  %s %.4g MeV zone %d process %d: %.8f of q, its share %.8f\n",
                          kMatName[m], e, zone, k, got, want[k]);
            }
            ++bad;
          }
        }
      }
    }
    std::printf("  %d cells off their share by more than 2 grid steps\n", bad);
    if (bad != 0) { fail("the selection walk does not give each process its table's share"); }
  }

  std::printf("\n%s: %d failure(s)\n", g_fails ? "FAILED" : "passed", g_fails);
  return g_fails ? 1 : 0;
}
