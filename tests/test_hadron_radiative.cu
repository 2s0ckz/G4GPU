// Hadron bremsstrahlung and direct pair production, against the Geant4 model classes.
//
// G4hBremsstrahlungModel derives from G4MuBremsstrahlungModel and overrides only the
// differential cross section - a different nuclear form-factor length, no scattering off
// atomic electrons, and a spin factor applied only when the spin is non-zero, which makes
// pions and kaons differ from protons. G4hPairProductionModel overrides nothing, so the
// mass-parameterised muon pair model covers hadrons as it stands; this checks that claim
// rather than assuming it.
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <string>
#include "core/particle.cuh"
#include "data/materials.cuh"
#include "physics/em/muon_radiative.cuh"

using namespace g4gpu;
using real_t = double;

static ParticleType type_of(const std::string& n) {
  if (n == "pi+") { return ParticleType::kPionPlus; }
  if (n == "kaon+") { return ParticleType::kKaonPlus; }
  if (n == "proton") { return ParticleType::kProton; }
  if (n == "anti_proton") { return ParticleType::kAntiProton; }
  return ParticleType::kNumTypes;
}

int main() {
  data::Material<real_t> mats[data::kNumMaterials];
  data::build_b1_materials<real_t>(mats);
  const char* g4names[4] = {"G4_AIR", "G4_WATER", "G4_A-150_TISSUE", "G4_BONE_COMPACT_ICRU"};

  const char* env = std::getenv("G4GPU_ORACLE");
  const std::string dir = (env != nullptr) ? env : "ref/oracle";
  FILE* f = std::fopen((dir + "/hadron_radiative.csv").c_str(), "r");
  if (f == nullptr) {
    std::printf("cannot read %s/hadron_radiative.csv\n", dir.c_str());
    return 1;
  }
  char line[512];
  if (std::fgets(line, sizeof line, f) == nullptr) { std::fclose(f); return 1; }

  struct Acc { double worst; std::string where; int n; };
  Acc br_d{0, "", 0}, br_x{0, "", 0}, pp_d{0, "", 0}, pp_x{0, "", 0};
  auto track = [](Acc& a, double ours, double g4, const char* p, const char* m, double e) {
    if (g4 <= 0) { return; }
    ++a.n;
    const double dev = std::fabs(ours / g4 - 1);
    if (dev > a.worst) {
      a.worst = dev;
      char buf[160];
      std::snprintf(buf, sizeof buf, "%s in %s at %.4g MeV", p, m, e);
      a.where = buf;
    }
  };

  while (std::fgets(line, sizeof line, f) != nullptr) {
    char mat[128], part[32];
    double e, gc, pcut, d1, x1, d2, x2;
    if (std::sscanf(line, "%127[^,],%31[^,],%lf,%lf,%lf,%lf,%lf,%lf,%lf", mat, part, &e, &gc,
                    &pcut, &d1, &x1, &d2, &x2) != 9) {
      continue;
    }
    const ParticleType t = type_of(part);
    if (t == ParticleType::kNumTypes) { continue; }
    int mi = -1;
    for (int i = 0; i < data::kNumMaterials; ++i) {
      if (std::string(mat) == g4names[i]) { mi = i; break; }
    }
    if (mi < 0) { continue; }
    track(br_d, em::h_brem_dedx(mats[mi], t, real_t(e), real_t(gc)), d1, part, mat, e);
    track(br_x, em::h_brem_xs(mats[mi], t, real_t(e), real_t(gc)), x1, part, mat, e);
    track(pp_d, em::mu_pair_dedx(mats[mi], t, real_t(e), real_t(10) * real_t(pcut)), d2, part,
          mat, e);
    track(pp_x, em::mu_pair_xs(mats[mi], t, real_t(e), real_t(pcut)), x2, part, mat, e);
  }
  std::fclose(f);

  printf("== hadron radiative models vs Geant4, called directly ==\n");
  printf("  %-34s %8s %12s   %s\n", "quantity", "points", "worst dev", "where");
  struct Row { const char* label; const Acc* a; };
  const Row rows[4] = {{"G4hBremsstrahlungModel dE/dx", &br_d},
                       {"G4hBremsstrahlungModel xs", &br_x},
                       {"G4hPairProductionModel dE/dx", &pp_d},
                       {"G4hPairProductionModel xs", &pp_x}};
  for (const Row& r : rows) {
    printf("  %-34s %8d %11.4f%%   %s\n", r.label, r.a->n, 100 * r.a->worst,
           r.a->where.c_str());
  }

  int fails = 0;
  for (const Row& r : rows) {
    if (r.a->n == 0) {
      printf("\n  FAIL: %s compared nothing\n", r.label);
      ++fails;
    } else if (r.a->worst > 0.01) {
      printf("\n  FAIL: %s exceeds 1%%\n", r.label);
      ++fails;
    }
  }
  printf("\n%s (%d failures)\n", fails ? "FAILED" : "PASSED", fails);
  return fails ? 1 : 0;
}
