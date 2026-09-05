// Relativistic bremsstrahlung with LPM suppression, against G4eBremsstrahlungRelModel.
//
// The oracle instantiates the Geant4 model class and calls it directly. Going through
// G4EmCalculator instead would be ambiguous: it selects a model by energy, so near the
// 1 GeV SeltzerBerger/relativistic boundary its answer does not say which model produced it,
// and a mismatch there cannot be distinguished from a transcription error.
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>
#include "core/rng.cuh"
#include "data/materials.cuh"
#include "physics/em/brems_rel.cuh"

using namespace g4gpu;
using real_t = double;

struct Row { std::string mat; double e, gcut, dedx, xs; };

static std::vector<Row> load(const std::string& path) {
  std::vector<Row> out;
  FILE* f = std::fopen(path.c_str(), "r");
  if (f == nullptr) { return out; }
  char line[512];
  if (std::fgets(line, sizeof line, f) == nullptr) { std::fclose(f); return out; }
  while (std::fgets(line, sizeof line, f) != nullptr) {
    char mat[128];
    double e, gc, d, x;
    int lpm;
    if (std::sscanf(line, "%127[^,],%lf,%lf,%lf,%lf,%d", mat, &e, &gc, &d, &x, &lpm) == 6) {
      out.push_back({mat, e, gc, d, x});
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
  auto rows = load(dir + "/brems_rel.csv");
  if (rows.empty()) {
    std::printf("cannot read %s/brems_rel.csv - run ref/oracle/run.bat first\n", dir.c_str());
    return 1;
  }

  auto per_volume_xs = [&](const data::Material<real_t>& m, real_t e, real_t gc) {
    real_t t = 0;
    for (int i = 0; i < m.n_elements; ++i) {
      t += m.n_atoms[i] * em::rel_brem_xs_per_atom(m, int(m.z[i] + real_t(0.5)), e, gc, e);
    }
    return t;
  };
  auto per_volume_dedx = [&](const data::Material<real_t>& m, real_t e, real_t gc) {
    real_t t = 0;
    for (int i = 0; i < m.n_elements; ++i) {
      t += m.n_atoms[i] * em::rel_brem_loss_per_atom(m, int(m.z[i] + real_t(0.5)), e, gc);
    }
    return t;
  };

  printf("== G4eBremsstrahlungRelModel, called directly ==\n");
  printf("  %-20s %11s %13s %13s %8s %13s %13s %8s\n", "material", "E (MeV)", "xs ours",
         "xs G4", "ratio", "dedx ours", "dedx G4", "ratio");
  double worst_xs = 0, worst_dedx = 0;
  std::string where_xs, where_dedx;
  int compared = 0, shown = 0;
  for (const Row& r : rows) {
    int mi = -1;
    for (int i = 0; i < data::kNumMaterials; ++i) {
      if (r.mat == g4names[i]) { mi = i; break; }
    }
    if (mi < 0 || r.xs <= 0 || r.dedx <= 0) { continue; }
    const real_t ox = per_volume_xs(mats[mi], real_t(r.e), real_t(r.gcut));
    const real_t od = per_volume_dedx(mats[mi], real_t(r.e), real_t(r.gcut));
    ++compared;
    const double rx = ox / r.xs, rd = od / r.dedx;
    char buf[160];
    if (std::fabs(rx - 1) > worst_xs) {
      worst_xs = std::fabs(rx - 1);
      std::snprintf(buf, sizeof buf, "%s at %.4g MeV", r.mat.c_str(), r.e);
      where_xs = buf;
    }
    if (std::fabs(rd - 1) > worst_dedx) {
      worst_dedx = std::fabs(rd - 1);
      std::snprintf(buf, sizeof buf, "%s at %.4g MeV", r.mat.c_str(), r.e);
      where_dedx = buf;
    }
    if (std::getenv("G4GPU_ALL") != nullptr || (shown++ % 60) == 0) {
      printf("  %-20s %11.4g %13.6g %13.6g %8.4f %13.6g %13.6g %8.4f\n", r.mat.c_str(), r.e,
             ox, r.xs, rx, od, r.dedx, rd);
    }
  }
  printf("\n  %d points compared, 1 MeV - 100 TeV\n", compared);
  printf("  worst cross section deviation: %.4f%%  (%s)\n", 100 * worst_xs, where_xs.c_str());
  printf("  worst dE/dx deviation:         %.4f%%  (%s)\n", 100 * worst_dedx,
         where_dedx.c_str());

  printf("\n== LPM suppression of the DCS in water at y = 0.01 ==\n");
  printf("  %14s %14s %14s %10s %s\n", "E (MeV)", "no LPM", "with LPM", "ratio", "state");
  {
    const auto& m = mats[data::kWater];
    const auto el = em::rel_brem_element<real_t>(8);
    for (real_t e : {real_t(1e3), real_t(1e5), real_t(1e7), real_t(1e9)}) {
      const real_t etot = e + units::electron_mass_c2<real_t>();
      const auto su = em::rel_brem_setup(m, etot);
      const real_t k = real_t(0.01) * etot;
      const real_t a = em::rel_brem_dxs(el, 8, k, etot);
      const real_t b = em::rel_brem_dxs_lpm(el, k, etot, su.lpm_energy, su.density_corr);
      printf("  %14.4g %14.6g %14.6g %10.4f %s\n", e, a, b, (a > 0) ? b / a : 0.0,
             su.lpm_active ? "LPM active" : "-");
    }
  }

  int fails = (worst_xs > 0.005 || worst_dedx > 0.005) ? 1 : 0;
  if (fails) { printf("\n  FAIL: exceeds 0.5%%\n"); }
  if (compared == 0) {
    printf("\n  FAIL: no oracle rows compared\n");
    ++fails;
  }
  printf("\n%s\n", fails ? "FAILED" : "PASSED");
  return fails;
}
