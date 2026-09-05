// Low-energy ionisation for negatively charged hadrons, against G4ICRU73QOModel.
//
// G4hIonisation and G4MuIonisation select this instead of G4BraggModel when the charge is
// negative, below 2 MeV per nucleon. The Barkas term is odd in the charge, which is what
// makes an antiproton stop differently from a proton at the same velocity - so this is not
// a cosmetic variant of the Bragg model but the reason negative hadrons need their own.
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <string>
#include "core/particle.cuh"
#include "data/materials.cuh"
#include "physics/em/icru73qo.cuh"

using namespace g4gpu;
using real_t = double;

static ParticleType type_of(const std::string& n) {
  if (n == "pi-") { return ParticleType::kPionMinus; }
  if (n == "anti_proton") { return ParticleType::kAntiProton; }
  if (n == "mu-") { return ParticleType::kMuonMinus; }
  return ParticleType::kNumTypes;
}

int main() {
  data::Material<real_t> mats[data::kNumMaterials];
  data::build_b1_materials<real_t>(mats);
  const char* g4names[4] = {"G4_AIR", "G4_WATER", "G4_A-150_TISSUE", "G4_BONE_COMPACT_ICRU"};

  const char* env = std::getenv("G4GPU_ORACLE");
  const std::string dir = (env != nullptr) ? env : "ref/oracle";
  FILE* f = std::fopen((dir + "/icru73qo.csv").c_str(), "r");
  if (f == nullptr) {
    std::printf("cannot read %s/icru73qo.csv\n", dir.c_str());
    return 1;
  }
  char line[512];
  if (std::fgets(line, sizeof line, f) == nullptr) { std::fclose(f); return 1; }

  printf("== G4ICRU73QOModel, negative hadrons ==\n");
  printf("  %-22s %-12s %11s %14s %14s %8s\n", "material", "part", "E (MeV)", "ours",
         "Geant4", "ratio");
  double worst = 0;
  std::string where;
  int n = 0, shown = 0;
  while (std::fgets(line, sizeof line, f) != nullptr) {
    char mat[128], part[32];
    double e, cut, d;
    if (std::sscanf(line, "%127[^,],%31[^,],%lf,%lf,%lf", mat, part, &e, &cut, &d) != 5) {
      continue;
    }
    const ParticleType t = type_of(part);
    if (t == ParticleType::kNumTypes || d <= 0) { continue; }
    int mi = -1;
    for (int i = 0; i < data::kNumMaterials; ++i) {
      if (std::string(mat) == g4names[i]) { mi = i; break; }
    }
    if (mi < 0) { continue; }
    const real_t ours = em::qo_dedx(mats[mi], t, real_t(e), real_t(cut));
    ++n;
    const double r = ours / d;
    if (std::fabs(r - 1) > worst) {
      worst = std::fabs(r - 1);
      char buf[180];
      std::snprintf(buf, sizeof buf, "%s in %s at %.4g MeV", part, mat, e);
      where = buf;
    }
    if ((shown++ % 60) == 0) {
      printf("  %-22s %-12s %11.5g %14.7g %14.7g %8.4f\n", mat, part, e, ours, d, r);
    }
  }
  std::fclose(f);

  printf("\n  %d points compared; worst deviation %.4f%%  (%s)\n", n, 100 * worst,
         where.c_str());
  int fails = 0;
  if (n == 0) { printf("  FAIL: nothing compared\n"); ++fails; }
  if (worst > 0.001) { printf("  FAIL: exceeds 0.1%%\n"); ++fails; }
  printf("\n%s (%d failures)\n", fails ? "FAILED" : "PASSED", fails);
  return fails ? 1 : 0;
}
