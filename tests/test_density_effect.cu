// The Sternheimer density-effect correction, diffed against G4IonisParamMat directly.
//
// This exists because getting it from the published G4DensityEffectData table was wrong for
// water: Geant4 computes that material's coefficients analytically instead, and the two
// disagree in Cbar by 0.078. That is invisible at electron energies but shows up as a flat
// 0.39% error in every heavy-particle dE/dx at high energy. Comparing the coefficients is
// not enough - compare the function.
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>
#include "data/materials.cuh"

using namespace g4gpu;
using real_t = double;

int main() {
  data::Material<real_t> mats[data::kNumMaterials];
  data::build_b1_materials<real_t>(mats);
  const char* g4names[4] = {"G4_AIR", "G4_WATER", "G4_A-150_TISSUE", "G4_BONE_COMPACT_ICRU"};

  const char* env = std::getenv("G4GPU_ORACLE");
  const std::string dir = (env != nullptr) ? env : "ref/oracle";
  FILE* f = std::fopen((dir + "/density_correction.csv").c_str(), "r");
  if (f == nullptr) {
    std::printf("cannot read %s/density_correction.csv\n", dir.c_str());
    return 1;
  }
  char line[512];
  if (std::fgets(line, sizeof line, f) == nullptr) { std::fclose(f); return 1; }

  printf("== delta(x) vs G4IonisParamMat::DensityCorrection ==\n");
  printf("  %-22s %8s %18s %18s %12s\n", "material", "x", "ours", "Geant4", "abs diff");
  double worst[4] = {0, 0, 0, 0};
  int n = 0, shown = 0;
  while (std::fgets(line, sizeof line, f) != nullptr) {
    char mat[128];
    double x, d;
    if (std::sscanf(line, "%127[^,],%lf,%lf", mat, &x, &d) != 3) { continue; }
    int mi = -1;
    for (int i = 0; i < data::kNumMaterials; ++i) {
      if (std::string(mat) == g4names[i]) { mi = i; break; }
    }
    if (mi < 0) { continue; }
    const real_t ours = data::density_correction(mats[mi], real_t(x));
    const double diff = std::fabs(ours - d);
    if (diff > worst[mi]) { worst[mi] = diff; }
    ++n;
    if ((shown++ % 47) == 0) {
      printf("  %-22s %8.2f %18.10f %18.10f %12.3g\n", mat, x, ours, d, diff);
    }
  }
  std::fclose(f);

  printf("\n  %d points compared, x from -4 to 10\n", n);
  int fails = 0;
  for (int i = 0; i < data::kNumMaterials; ++i) {
    printf("  worst absolute deviation, %-22s %.3g\n", g4names[i], worst[i]);
    // delta enters the Bethe-Bloch bracket additively, and the bracket is order 20, so
    // 1e-6 here is 5e-8 relative in dE/dx.
    if (worst[i] > 1e-6) { ++fails; }
  }
  if (fails) { printf("\n  FAIL: exceeds 1e-6 absolute\n"); }
  if (n == 0) { printf("\n  FAIL: nothing compared\n"); ++fails; }
  printf("\n%s\n", fails ? "FAILED" : "PASSED");
  return fails;
}
