// Heavy charged particle ionisation, against Geant4's own G4BetheBlochModel.
//
// The oracle instantiates the model class and calls it directly rather than going through
// G4EmCalculator. That is not a stylistic choice: G4EmCalculator picks a model by energy,
// and for negative hadrons below 10 MeV it returns G4ICRU73QOModel instead, which made the
// antiproton look 18% wrong when the transcription was in fact correct to 0.5%. The same
// trap cost a false diagnosis in the relativistic bremsstrahlung work (docs/RISK.md O1).
//
// G4BetheBlochModel adds shell, Barkas, Bloch and Mott corrections to the Bethe-Bloch
// formula; all four are transcribed. The ICRU90 tabulated stopping powers (protons and
// alphas below 2 MeV in water, air and graphite) and the ion effective-charge Barkas branch
// are not. This test measures the residual by energy band so the cost is on the record.
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>
#include "core/particle.cuh"
#include "data/materials.cuh"
#include "physics/em/hadron_ionisation.cuh"

using namespace g4gpu;
using real_t = double;

struct Row { std::string mat, part; double e, cut, dedx, xs; };

static std::vector<Row> load(const std::string& path) {
  std::vector<Row> out;
  FILE* f = std::fopen(path.c_str(), "r");
  if (f == nullptr) { return out; }
  char line[512];
  if (std::fgets(line, sizeof line, f) == nullptr) { std::fclose(f); return out; }
  while (std::fgets(line, sizeof line, f) != nullptr) {
    char mat[128], part[32];
    double e, c, d, x;
    if (std::sscanf(line, "%127[^,],%31[^,],%lf,%lf,%lf,%lf", mat, part, &e, &c, &d, &x) == 6) {
      out.push_back({mat, part, e, c, d, x});
    }
  }
  std::fclose(f);
  return out;
}

static ParticleType type_of(const std::string& n) {
  if (n == "mu-") { return ParticleType::kMuonMinus; }
  if (n == "mu+") { return ParticleType::kMuonPlus; }
  if (n == "pi+") { return ParticleType::kPionPlus; }
  if (n == "pi-") { return ParticleType::kPionMinus; }
  if (n == "kaon+") { return ParticleType::kKaonPlus; }
  if (n == "proton") { return ParticleType::kProton; }
  if (n == "anti_proton") { return ParticleType::kAntiProton; }
  if (n == "alpha") { return ParticleType::kAlpha; }
  if (n == "He3") { return ParticleType::kHe3; }
  return ParticleType::kNumTypes;
}

int main() {
  data::Material<real_t> mats[data::kNumMaterials];
  data::build_b1_materials<real_t>(mats);
  const char* g4names[4] = {"G4_AIR", "G4_WATER", "G4_A-150_TISSUE", "G4_BONE_COMPACT_ICRU"};
  int fails = 0;
  static em::ShellTables<real_t> shell;
  em::build_shell_tables(shell);

  const char* env = std::getenv("G4GPU_ORACLE");
  const std::string dir = (env != nullptr) ? env : "ref/oracle";
  auto rows = load(dir + "/bethe_bloch.csv");
  if (rows.empty()) {
    std::printf("cannot read %s/bethe_bloch.csv - run ref/oracle/run.bat first\n", dir.c_str());
    return 1;
  }

  // The PDG masses and charges the port hardcodes, against Geant4's own table.
  printf("== particle table vs Geant4 PDG values ==\n");
  {
    auto pdg = load(dir + "/hadron_tables.csv");  // carries mass and charge columns
    FILE* f = std::fopen((dir + "/hadron_tables.csv").c_str(), "r");
    char line[1024];
    std::vector<std::string> seen;
    if (f != nullptr && std::fgets(line, sizeof line, f) != nullptr) {
      printf("  %-14s %16s %16s %8s %8s\n", "particle", "mass ours", "mass G4", "q ours",
             "q G4");
      while (std::fgets(line, sizeof line, f) != nullptr) {
        char mat[128], part[32];
        double mass, q;
        if (std::sscanf(line, "%127[^,],%31[^,],%lf,%lf", mat, part, &mass, &q) != 4) {
          continue;
        }
        bool done = false;
        for (const auto& s : seen) { if (s == part) { done = true; break; } }
        if (done) { continue; }
        seen.push_back(part);
        const ParticleType t = type_of(part);
        if (t == ParticleType::kNumTypes) { continue; }
        const auto pd = particle_def<real_t>(t);
        const bool bad =
            std::fabs(pd.mass - mass) > 1e-6 * mass || std::fabs(pd.charge - q) > 1e-9;
        printf("  %-14s %16.9g %16.9g %8.3g %8.3g%s\n", part, pd.mass, mass, pd.charge, q,
               bad ? "   <-- MISMATCH" : "");
        if (bad) { ++fails; }
      }
      std::fclose(f);
    }
  }

  struct Bucket { const char* label; double lo, hi; double worst; int n; std::string where; };
  Bucket b[] = {{"1 keV - 1 MeV", 0, 1, 0, 0, ""},
                {"1 - 10 MeV", 1, 10, 0, 0, ""},
                {"10 - 100 MeV", 10, 100, 0, 0, ""},
                {"0.1 - 1 GeV", 100, 1000, 0, 0, ""},
                {"1 GeV - 1 TeV", 1000, 1e6, 0, 0, ""},
                {"> 1 TeV", 1e6, 1e12, 0, 0, ""}};
  const int nb = sizeof b / sizeof b[0];

  int compared = 0;
  double worst_xs = 0;
  std::string where_xs;
  for (const Row& r : rows) {
    const ParticleType t = type_of(r.part);
    if (t == ParticleType::kNumTypes) { continue; }
    int mi = -1;
    for (int i = 0; i < data::kNumMaterials; ++i) {
      if (r.mat == g4names[i]) { mi = i; break; }
    }
    if (mi < 0 || r.dedx <= 0) { continue; }
    const auto pd = particle_def<real_t>(t);
    // Only where Geant4 itself uses Bethe-Bloch rather than a Bragg/ICRU73QO model.
    if (r.e < em::bragg_bethe_boundary(pd)) { continue; }
    const real_t ours = em::bethe_bloch_dedx(mats[mi], t, real_t(r.e), real_t(r.cut), &shell);
    if (!(ours > 0)) { continue; }
    ++compared;
    const double dev = std::fabs(ours / r.dedx - 1);
    for (int i = 0; i < nb; ++i) {
      if (r.e >= b[i].lo && r.e < b[i].hi) {
        ++b[i].n;
        if (dev > b[i].worst) { b[i].worst = dev; b[i].where = r.part + " in " + r.mat; }
        break;
      }
    }
    if (r.xs > 0) {
      const real_t ox = em::bethe_bloch_delta_xs(mats[mi], t, real_t(r.e), real_t(r.cut),
                                                 real_t(1e30));
      const double dx = std::fabs(ox / r.xs - 1);
      if (dx > worst_xs) {
        worst_xs = dx;
        char buf[160];
        std::snprintf(buf, sizeof buf, "%s in %s at %.4g MeV", r.part.c_str(), r.mat.c_str(),
                      r.e);
        where_xs = buf;
      }
    }
  }

  printf("\n== restricted dE/dx vs G4BetheBlochModel, by energy band ==\n");
  printf("  %-16s %8s %12s   %s\n", "band", "points", "worst dev", "where");
  for (int i = 0; i < nb; ++i) {
    if (b[i].n == 0) { continue; }
    printf("  %-16s %8d %11.3f%%   %s\n", b[i].label, b[i].n, 100 * b[i].worst,
           b[i].where.c_str());
  }
  printf("\n  %d points compared\n", compared);
  printf("  worst delta-ray cross section deviation: %.3f%%  (%s)\n", 100 * worst_xs,
         where_xs.c_str());

  double worst_all = 0;
  for (int i = 0; i < nb; ++i) { worst_all = std::max(worst_all, b[i].worst); }
  if (worst_all > 0.01) { printf("\n  FAIL: dE/dx exceeds 1%%\n"); ++fails; }
  if (worst_xs > 0.01) { printf("  FAIL: delta-ray cross section exceeds 1%%\n"); ++fails; }
  if (compared == 0) { printf("  FAIL: nothing compared\n"); ++fails; }

  printf("\n%s (%d failures)\n", fails ? "FAILED" : "PASSED", fails);
  return fails ? 1 : 0;
}
