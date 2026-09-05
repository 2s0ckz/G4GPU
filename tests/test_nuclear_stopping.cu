// Nuclear stopping, against G4ICRU49NuclearStoppingModel called directly.
//
// G4EmStandardPhysics registers G4NuclearStopping for protons and every ion: the energy a
// slow heavy particle loses to elastic recoils of whole nuclei rather than to atomic
// electrons. It is active only below z1^2 MeV per nucleon, so it is a low-energy correction,
// but at the very end of an ion track it is comparable to the electronic stopping power.
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <string>
#include "core/particle.cuh"
#include "data/materials.cuh"
#include "physics/em/nuclear_stopping.cuh"

using namespace g4gpu;
using real_t = double;

static ParticleType type_of(const std::string& n) {
  if (n == "proton") { return ParticleType::kProton; }
  if (n == "alpha") { return ParticleType::kAlpha; }
  if (n == "He3") { return ParticleType::kHe3; }
  return ParticleType::kNumTypes;
}

int main() {
  data::Material<real_t> mats[data::kNumMaterials];
  data::build_b1_materials<real_t>(mats);
  const char* g4names[4] = {"G4_AIR", "G4_WATER", "G4_A-150_TISSUE", "G4_BONE_COMPACT_ICRU"};

  const char* env = std::getenv("G4GPU_ORACLE");
  const std::string dir = (env != nullptr) ? env : "ref/oracle";
  FILE* f = std::fopen((dir + "/nuclear_stopping.csv").c_str(), "r");
  if (f == nullptr) {
    std::printf("cannot read %s/nuclear_stopping.csv\n", dir.c_str());
    return 1;
  }
  char line[512];
  if (std::fgets(line, sizeof line, f) == nullptr) { std::fclose(f); return 1; }

  double worst = 0;
  std::string where;
  int n = 0, shown = 0;
  printf("== nuclear stopping vs G4ICRU49NuclearStoppingModel ==\n");
  printf("  %-22s %-8s %12s %14s %14s %8s\n", "material", "part", "E (MeV)", "ours", "Geant4",
         "ratio");
  while (std::fgets(line, sizeof line, f) != nullptr) {
    char mat[128], part[32];
    double e, d;
    if (std::sscanf(line, "%127[^,],%31[^,],%lf,%lf", mat, part, &e, &d) != 4) { continue; }
    const ParticleType t = type_of(part);
    if (t == ParticleType::kNumTypes || d <= 0) { continue; }
    int mi = -1;
    for (int i = 0; i < data::kNumMaterials; ++i) {
      if (std::string(mat) == g4names[i]) { mi = i; break; }
    }
    if (mi < 0) { continue; }
    const real_t ours = em::nuclear_stopping_dedx(mats[mi], t, real_t(e));
    ++n;
    const double r = ours / d;
    if (std::fabs(r - 1) > worst) {
      worst = std::fabs(r - 1);
      char buf[180];
      std::snprintf(buf, sizeof buf, "%s in %s at %.4g MeV", part, mat, e);
      where = buf;
    }
    if ((shown++ % 90) == 0) {
      printf("  %-22s %-8s %12.5g %14.7g %14.7g %8.4f\n", mat, part, e, ours, d, r);
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
