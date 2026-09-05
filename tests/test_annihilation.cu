// Positron annihilation, against Geant4 11.1.1's own G4eeToTwoGammaModel.
//
// The oracle here calls the model class directly rather than going through G4EmCalculator:
// the calculator's FindEmModel returns null for the G4VEmProcess-derived processes and
// silently yields zero, which is worse than no oracle at all.
#include <cstdio>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include "core/rng.cuh"
#include "data/materials.cuh"
#include "physics/em/annihilation.cuh"

using namespace g4gpu;
using real_t = double;

struct Row { std::string mat; double e, xs; };

static std::vector<Row> load(const char* path) {
  std::vector<Row> out;
  FILE* f = std::fopen(path, "r");
  if (f == nullptr) { return out; }
  char line[512];
  if (std::fgets(line, sizeof line, f) == nullptr) { std::fclose(f); return out; }  // header
  while (std::fgets(line, sizeof line, f) != nullptr) {
    char name[128];
    double e, v;
    if (std::sscanf(line, "%127[^,],%lf,%lf", name, &e, &v) == 3) {
      out.push_back({name, e, v});
    }
  }
  std::fclose(f);
  return out;
}

int main() {
  data::Material<real_t> mats[data::kNumMaterials];
  data::build_b1_materials<real_t>(mats);
  const char* g4names[4] = {"G4_AIR", "G4_WATER", "G4_A-150_TISSUE", "G4_BONE_COMPACT_ICRU"};

  const char* env = std::getenv("G4GPU_ORACLE");
  const std::string dir = (env != nullptr) ? env : "ref/oracle";
  auto rows = load((dir + "/annihilation.csv").c_str());
  if (rows.empty()) {
    std::printf("cannot read %s/annihilation.csv - run ref/oracle/run.bat first\n", dir.c_str());
    return 1;
  }

  printf("== annihilation cross section per volume, 1/mm, vs G4eeToTwoGammaModel ==\n");
  printf("  %-22s %12s %14s %14s %8s\n", "material", "E (MeV)", "ours", "Geant4", "ratio");
  double worst = 0;
  std::string worst_where;
  int shown = 0, compared = 0;
  for (const Row& r : rows) {
    int mi = -1;
    for (int i = 0; i < data::kNumMaterials; ++i) {
      if (r.mat == g4names[i]) { mi = i; break; }
    }
    if (mi < 0 || r.xs <= 0) { continue; }
    const real_t ours = em::annihilation_xs(mats[mi], real_t(r.e));
    const double ratio = ours / r.xs;
    ++compared;
    if (std::fabs(ratio - 1) > worst) {
      worst = std::fabs(ratio - 1);
      char buf[160];
      std::snprintf(buf, sizeof buf, "%s at %.4g MeV", r.mat.c_str(), r.e);
      worst_where = buf;
    }
    if ((shown++ % 40) == 0) {
      printf("  %-22s %12.5g %14.7g %14.7g %8.4f\n", r.mat.c_str(), r.e, ours, r.xs, ratio);
    }
  }
  printf("\n  %d points compared; worst deviation %.4f%% (%s)\n", compared, 100 * worst,
         worst_where.c_str());

  // The in-flight sampler must conserve energy and produce forward-consistent momentum.
  printf("\n== in-flight sampling: energy and momentum conservation ==\n");
  printf("  %10s %14s %14s %14s\n", "E (MeV)", "max |dE|/E", "max |dp|/p", "samples");
  int fails = (worst > 0.001) ? 1 : 0;
  if (worst > 0.001) { printf("  FAIL: cross section off by more than 0.1%%\n"); }
  const real_t me = units::electron_mass_c2<real_t>();
  for (real_t e : {real_t(0.01), real_t(0.5), real_t(6.0), real_t(100.0)}) {
    Philox<real_t> rng(7u, unsigned(e * 10) + 1u);
    const Vec3<real_t> dir{real_t(0), real_t(0), real_t(1)};
    real_t worst_e = 0, worst_p = 0;
    const int N = 200000;
    for (int i = 0; i < N; ++i) {
      const auto a = em::sample_annihilation_in_flight(e, dir, rng);
      const real_t etot = e + 2 * me;
      worst_e = std::max(worst_e, std::fabs(a.energy1 + a.energy2 - etot) / etot);
      const real_t px = a.energy1 * a.dir1.x + a.energy2 * a.dir2.x;
      const real_t py = a.energy1 * a.dir1.y + a.energy2 * a.dir2.y;
      const real_t pz = a.energy1 * a.dir1.z + a.energy2 * a.dir2.z;
      const real_t p0 = std::sqrt(e * (e + 2 * me));
      worst_p = std::max(worst_p, std::sqrt(px * px + py * py + (pz - p0) * (pz - p0)) / p0);
    }
    printf("  %10.4g %14.3g %14.3g %14d\n", e, worst_e, worst_p, N);
    if (worst_e > 1e-12) { printf("    FAIL: energy not conserved\n"); ++fails; }
    // The second photon direction is fixed by momentum conservation, so the residual is
    // limited only by the energy the two photons are given, i.e. rounding.
    if (worst_p > 1e-9) { printf("    FAIL: momentum not conserved\n"); ++fails; }
  }

  printf("\n%s (%d failures)\n", fails ? "FAILED" : "PASSED", fails);
  return fails ? 1 : 0;
}
