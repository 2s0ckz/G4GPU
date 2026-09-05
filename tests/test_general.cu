// Checks the general material API: cuts and Sternheimer parameters computed from
// composition alone must reproduce Geant4's, with no hand-entered per-material values.
#include <cstdio>
#include <cmath>
#include "data/materials.cuh"

using namespace g4gpu;
using real_t = double;

int main() {
  // Build B1's materials through the GENERAL api - no tabulated cuts, no tabulated
  // Sternheimer coefficients.
  data::MaterialTable<real_t> t;
  { const int z[4] = {6, 7, 8, 18};
    const real_t w[4] = {0.000124, 0.755267, 0.231781, 0.012827};
    data::add_material<real_t>(t, 0.00120479, 4, z, w, 85.7); }
  { const real_t aH = data::atomic_mass<real_t>(1), aO = data::atomic_mass<real_t>(8);
    const real_t tot = 2 * aH + aO;
    const int z[2] = {1, 8};
    const real_t w[2] = {2 * aH / tot, aO / tot};
    data::add_material<real_t>(t, 1.0, 2, z, w, 78.0); }
  { const int z[6] = {1, 6, 7, 8, 9, 20};
    const real_t w[6] = {0.101327, 0.775501, 0.035057, 0.052316, 0.017422, 0.018378};
    data::add_material<real_t>(t, 1.127, 6, z, w, 65.1); }
  { const int z[8] = {1, 6, 7, 8, 12, 15, 16, 20};
    const real_t w[8] = {0.064, 0.278, 0.027, 0.410, 0.002, 0.070, 0.002, 0.147};
    data::add_material<real_t>(t, 1.85, 8, z, w, 91.9); }

  const char* nm[4] = {"air", "water", "tissue", "bone"};
  const real_t g4g[4] = {0.00099, 0.00252520505, 0.00228342803, 0.00393604038};
  const real_t g4e[4] = {0.00099, 0.277632595, 0.301330518, 0.398359696};
  // Geant4's tabulated Sternheimer C, for comparison with the analytic fallback.
  const real_t g4C[4] = {10.5961, 3.5017, 3.1100, 3.3390};

  printf("materials built through the general API: count = %d\n\n", t.count);
  printf("  %-8s %12s %12s %7s %12s %12s %7s %10s %10s %7s\n", "mat", "gcut ours", "gcut G4",
         "ratio", "ecut ours", "ecut G4", "ratio", "C ours", "C G4", "ratio");
  for (int i = 0; i < t.count; ++i) {
    printf("  %-8s %12.6g %12.6g %7.4f %12.6g %12.6g %7.4f %10.4f %10.4f %7.4f\n", nm[i],
           t[i].cut_gamma, g4g[i], t[i].cut_gamma / g4g[i], t[i].cut_electron, g4e[i],
           t[i].cut_electron / g4e[i], t[i].c_density, g4C[i], t[i].c_density / g4C[i]);
  }
  int zs[32];
  const int nz = t.elements(zs, 32);
  printf("\n  elements discovered from the materials (was hardcoded): %d ->", nz);
  for (int i = 0; i < nz; ++i) { printf(" %d", zs[i]); }
  printf("\n");
  return 0;
}
