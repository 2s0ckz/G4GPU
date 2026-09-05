// Validates the general range-cut -> energy converter against Geant4's own thresholds.
// If this reproduces the B1 materials, it works for user-supplied materials too.
#include <cstdio>
#include <cmath>
#include "data/materials.cuh"
#include "data/production_cuts.cuh"

using namespace g4gpu;
using real_t = double;

int main() {
  data::Material<real_t> mats[data::kNumMaterials];
  data::build_b1_materials<real_t>(mats);
  const char* names[4] = {"air", "water", "A-150 tissue", "bone"};

  // Geant4 11.1.1's own values, ref/oracle/cuts.csv.
  const real_t g4_gamma[4] = {real_t(0.00099), real_t(0.00252520505), real_t(0.00228342803),
                              real_t(0.00393604038)};
  const real_t g4_elec[4] = {real_t(0.00099), real_t(0.277632595), real_t(0.301330518),
                             real_t(0.398359696)};
  const real_t g4_pos[4] = {real_t(0.00099), real_t(0.270822571), real_t(0.293732854),
                            real_t(0.386305646)};

  // B1 does not set a cut, so QBBC's default applies. Scan to identify it.
  const real_t candidates[3] = {real_t(0.7), real_t(1.0), real_t(0.1)};
  for (real_t rc : candidates) {
    printf("== range cut = %.3g mm ==\n", rc);
    printf("  %-14s %14s %14s %8s %14s %14s %8s\n", "material", "gamma ours", "gamma G4",
           "ratio", "e- ours", "e- G4", "ratio");
    real_t worst = 0;
    for (int m = 0; m < data::kNumMaterials; ++m) {
      const auto& mm = mats[m];
      const real_t cg =
          data::convert_cut_gamma<real_t>(rc, mm.n_elements, mm.z, mm.n_atoms);
      const real_t ce = data::convert_cut_electron<real_t>(rc, mm.n_elements, mm.z, mm.n_atoms,
                                                           mm.density);
      const real_t cp = data::convert_cut_electron<real_t>(rc, mm.n_elements, mm.z, mm.n_atoms,
                                                           mm.density, true);
      const real_t rp = cp / g4_pos[m];
      const real_t rg = cg / g4_gamma[m], re = ce / g4_elec[m];
      printf("  %-14s %14.6g %14.6g %8.4f %14.6g %14.6g %8.4f\n", names[m], cg, g4_gamma[m], rg,
             ce, g4_elec[m], re);
      printf("  %-14s %14s %14s %8s %14.6g %14.6g %8.4f  (e+)\n", "", "", "", "", cp,
             g4_pos[m], rp);
      worst = std::max(worst, std::max(std::fabs(rg - 1),
                                       std::max(std::fabs(re - 1), std::fabs(rp - 1))));
    }
    printf("  worst deviation: %.3f%%\n\n", 100 * worst);
  }
  return 0;
}
