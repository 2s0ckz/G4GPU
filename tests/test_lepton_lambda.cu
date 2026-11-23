// P44 (docs/RISK.md V351): the e+- lambda tables of `em/lepton_lambda.cuh` against Geant4's own,
// node by node, and the cross-section types and peaks that decide the integral approach.
//
// THE ORACLE is Geant4 11.1.1 writing its tables: ref/b1mat with
//     /run/initialize, /run/particle/setStoredInAscii 1, /run/beamOn 1,
//     /run/particle/storePhysicsTable <dir>
// (B1 as it ships: couples G4_AIR, G4_WATER, G4_A-150_TISSUE, G4_BONE_COMPACT_ICRU, in that order,
// which is `data::build_b1_materials`' order - and those materials carry Geant4's own cuts to the
// digit, so this compares the TABLE construction and not the cut converter). The files are
// Lambda.eIoni.e-.asc, Lambda.eIoni.e+.asc, Lambda.eBrem.e-.asc, Lambda.eBrem.e+.asc, printed by
// G4PhysicsVector::Store at 12 significant digits.
//
//   tests\test_lepton_lambda.exe <dir with the four files>
//
// PASS: the same node count per couple, every node energy to 2e-9, every eIoni value to 2e-9 and
// every eBrem value below 100 GeV to 5e-5 (the reason for each is at the test below).
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "data/brems_data.cuh"
#include "data/materials.cuh"
#include "host/pe_upload.cuh"
#include "physics/em/lepton_lambda.cuh"

using namespace g4gpu;
using real_t = double;

struct G4Vec {
  std::vector<double> e, v;
};

static bool read_table(const std::string& path, std::vector<G4Vec>& out) {
  std::ifstream f(path);
  if (!f) { return false; }
  int nvec = 0;
  f >> nvec;
  for (int i = 0; i < nvec; ++i) {
    int type = 0;
    double emin = 0, emax = 0;
    int nbin = 0, n = 0;
    f >> type >> emin >> emax >> nbin >> n;
    G4Vec v;
    for (int j = 0; j < n; ++j) {
      double e = 0, x = 0;
      f >> e >> x;
      v.e.push_back(e);
      v.v.push_back(x);
    }
    out.push_back(v);
  }
  return static_cast<bool>(f);
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::printf("usage: test_lepton_lambda <dir of Geant4's Lambda.e*.asc>\n");
    return 2;
  }
  const std::string dir = argv[1];
  data::Material<real_t> mats[data::kNumMaterials];
  data::build_b1_materials<real_t>(mats);
  int nz = 0;
  const int* zs = host::b1_elements(nz);
  static data::SBTableSet<real_t> sb{};
  if (!data::load_sb_tables<real_t>(host::default_sb_dir(), zs, nz, sb)) {
    std::printf("FAIL: no Seltzer-Berger tables\n");
    return 1;
  }
  static em::LeptonLambdaTables<real_t> t;
  if (!em::build_lepton_lambda_tables<real_t>(mats, data::kNumMaterials, sb, t)) {
    std::printf("FAIL: build\n");
    return 1;
  }
  const char* names[4] = {"air", "water", "A-150", "bone"};
  int fails = 0;
  for (int proc = 0; proc < 2; ++proc) {
    for (int p = 0; p < 2; ++p) {
      const std::string file = dir + "/Lambda." + (proc == 0 ? "eIoni." : "eBrem.") +
                               (p == 0 ? "e-" : "e+") + ".asc";
      std::vector<G4Vec> g4;
      if (!read_table(file, g4)) {
        std::printf("FAIL: cannot read %s\n", file.c_str());
        return 1;
      }
      for (int m = 0; m < data::kNumMaterials && m < static_cast<int>(g4.size()); ++m) {
        const em::LepLamVec<real_t>& v = (proc == 0) ? t.ioni[p][m] : t.brem[p][m];
        double worst_e = 0, worst_v = 0, worst_v_e = 0, worst_low = 0;
        if (v.n != static_cast<int>(g4[m].e.size())) {
          std::printf("FAIL %s %s %s: %d nodes, Geant4 %zu\n", proc == 0 ? "eIoni" : "eBrem",
                      p == 0 ? "e-" : "e+", names[m], v.n, g4[m].e.size());
          ++fails;
          continue;
        }
        for (int j = 0; j < v.n; ++j) {
          const double de = std::fabs(v.e[j] / g4[m].e[j] - 1.0);
          if (de > worst_e) { worst_e = de; }
          const double ref = g4[m].v[j];
          const double dv = (ref != 0.0) ? std::fabs(v.v[j] / ref - 1.0) : std::fabs(v.v[j]);
          if (dv > worst_v) {
            worst_v = dv;
            worst_v_e = v.e[j];
          }
          if (v.e[j] < 1e5 && dv > worst_low) { worst_low = dv; }
        }
        // eIoni: every node to the oracle's digits (2e-9: build_b1_materials' cuts carry nine
        // significant digits, and the vector's first node is 2*cut). eBrem: the TABLE is
        // transcribed exactly - air, whose cut is the 990 eV floor, matches to 2e-12 - but the
        // brems MODEL it samples (data::brems_dedx_xs, not P44's) sits 1.7-2.4e-5 off Geant4's in
        // the three condensed materials below 100 GeV and up to 1.1e-3 at 10-100 TeV (the LPM
        // model); those are the model's, printed, and the pass line is the 100 GeV one.
        const bool ok = (worst_e < 2e-9)
                        && ((proc == 0) ? (worst_v < 2e-9) : (worst_low < 5e-5));
        if (!ok) { ++fails; }
        std::printf("%s %s %s %-6s %3d nodes: worst energy %.2e, worst value %.2e (at %.6g MeV), below 100 GeV %.2e\n",
                    ok ? "pass" : "FAIL", proc == 0 ? "eIoni" : "eBrem", p == 0 ? "e-" : "e+",
                    names[m], v.n, worst_e, worst_v, worst_v_e, worst_low);
      }
    }
  }
  for (int p = 0; p < 2; ++p) {
    std::printf("%s: eIoni type %d, eBrem type %d\n", p == 0 ? "e-" : "e+",
                static_cast<int>(t.ioni_type[p]), static_cast<int>(t.brem_type[p]));
    for (int m = 0; m < data::kNumMaterials; ++m) {
      const auto& pk = t.brem_peaks[p][m];
      std::printf("  %-6s eIoni peak %.6g MeV; eBrem peaks %.6g / deep %.6g / %.6g / deep %.6g / %.6g\n",
                  names[m], t.ioni_epeak[p][m], pk.e1peak, pk.e1deep, pk.e2peak, pk.e2deep,
                  pk.e3peak);
    }
  }
  // The majorant against the table: it must be at or above the table over [lambdaFactor*E, E]
  // wherever Geant4's scheme is exact, which is what thinning needs.
  // Not a pass criterion: where it falls short it is Geant4's own scheme falling short (a window
  // that straddles a deep, or a spline that overshoots its node peak), and the transcription is
  // of the scheme. Printed so the size is a number.
  int bad_major = 0;
  double worst_major = 0, worst_major_e = 0;
  for (int p = 0; p < 2; ++p) {
    for (int m = 0; m < data::kNumMaterials; ++m) {
      for (double e = 2e-3; e < 1e7; e *= 1.07) {
        const double maj_i = t.ioni_majorant(m, p == 1, e, std::log(e));
        const double maj_b = t.brem_majorant(m, p == 1, e, std::log(e));
        for (int k = 0; k <= 20; ++k) {
          const double ek = e * (1.0 - 0.2 * k / 20.0);
          const double ri = (maj_i > 0) ? t.ioni_post(m, p == 1, ek) / maj_i - 1.0 : 0.0;
          const double rb = (maj_b > 0) ? t.brem_post(m, p == 1, ek) / maj_b - 1.0 : 0.0;
          const double r = (ri > rb) ? ri : rb;
          if (r > 1e-12) {
            ++bad_major;
            if (r > worst_major) {
              worst_major = r;
              worst_major_e = e;
            }
          }
        }
      }
    }
  }
  std::printf("majorant below the table inside [0.8E, E]: %d points, worst %.3e (E = %.4g MeV)\n",
              bad_major, worst_major, worst_major_e);
  std::printf("%s: %d failures\n", fails == 0 ? "PASS" : "FAIL", fails);
  return fails == 0 ? 0 : 1;
}
