// Muon ionisation, bremsstrahlung and direct pair production, against the Geant4 model
// classes called directly.
//
// G4MuIonisation uses G4MuBetheBlochModel above 200 keV rather than G4BetheBlochModel: it
// adds an O(alpha) radiative correction for delta rays above 100 keV. G4MuBremsstrahlung and
// G4MuPairProduction are registered for mu+/mu- whenever the physics list reaches high
// energy, which G4EmStandardPhysics does (MaxKinEnergy defaults to 100 TeV).
//
// Pair production is identically zero unless the production cut exceeds 4 m_e = 2.04 MeV,
// which no B1 material cut does, so the oracle probes it at cuts where the model is live.
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>
#include "core/particle.cuh"
#include "data/materials.cuh"
#include "physics/em/hadron_ionisation.cuh"
#include "physics/em/muon_radiative.cuh"

using namespace g4gpu;
using real_t = double;

int main() {
  data::Material<real_t> mats[data::kNumMaterials];
  data::build_b1_materials<real_t>(mats);
  const char* g4names[4] = {"G4_AIR", "G4_WATER", "G4_A-150_TISSUE", "G4_BONE_COMPACT_ICRU"};
  static em::ShellTables<real_t> st;
  em::build_shell_tables(st);

  const char* env = std::getenv("G4GPU_ORACLE");
  const std::string dir = (env != nullptr) ? env : "ref/oracle";
  FILE* f = std::fopen((dir + "/muon_models.csv").c_str(), "r");
  if (f == nullptr) {
    std::printf("cannot read %s/muon_models.csv\n", dir.c_str());
    return 1;
  }
  char line[512];
  if (std::fgets(line, sizeof line, f) == nullptr) { std::fclose(f); return 1; }

  struct Acc { double worst; std::string where; int n; };
  Acc bb_d{0, "", 0}, bb_x{0, "", 0}, br_d{0, "", 0}, br_x{0, "", 0}, pp_d{0, "", 0},
      pp_x{0, "", 0};
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
    double e, cut, gc, d1, x1, d2, x2, d3, x3;
    if (std::sscanf(line, "%127[^,],%31[^,],%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf", mat, part,
                    &e, &cut, &gc, &d1, &x1, &d2, &x2, &d3, &x3) != 11) {
      continue;
    }
    const ParticleType t =
        (std::string(part) == "mu-") ? ParticleType::kMuonMinus : ParticleType::kMuonPlus;
    int mi = -1;
    for (int i = 0; i < data::kNumMaterials; ++i) {
      if (std::string(mat) == g4names[i]) { mi = i; break; }
    }
    if (mi < 0) { continue; }

    // Only where G4MuIonisation actually selects G4MuBetheBlochModel.
    if (e >= em::kMuBetheBlochLow<real_t>()) {
      track(bb_d, em::mu_bethe_bloch_dedx(mats[mi], t, real_t(e), real_t(cut), &st), d1, part,
            mat, e);
      track(bb_x,
            em::mu_bethe_bloch_delta_xs(mats[mi], t, real_t(e), real_t(cut), real_t(1e30)), x1,
            part, mat, e);
    }
    track(br_d, em::mu_brem_dedx(mats[mi], t, real_t(e), real_t(gc)), d2, part, mat, e);
    track(br_x, em::mu_brem_xs(mats[mi], t, real_t(e), real_t(gc)), x2, part, mat, e);

    // Matching the cuts the oracle probes pair production at.
    const real_t pcut = em::kMinPairEnergy<real_t>();
    track(pp_d, em::mu_pair_dedx(mats[mi], t, real_t(e), real_t(10) * pcut), d3, part, mat, e);
    track(pp_x, em::mu_pair_xs(mats[mi], t, real_t(e), pcut), x3, part, mat, e);
  }
  std::fclose(f);

  printf("== muon models vs Geant4, called directly ==\n");
  printf("  %-34s %8s %12s   %s\n", "quantity", "points", "worst dev", "where");
  struct Row { const char* label; const Acc* a; double tol; };
  // Ionisation is a closed form and should be near exact; the radiative models are nested
  // numerical integrations, so a slightly looser bar.
  const Row rows[6] = {{"G4MuBetheBlochModel dE/dx", &bb_d, 0.001},
                       {"G4MuBetheBlochModel delta xs", &bb_x, 0.001},
                       {"G4MuBremsstrahlungModel dE/dx", &br_d, 0.01},
                       {"G4MuBremsstrahlungModel xs", &br_x, 0.01},
                       {"G4MuPairProductionModel dE/dx", &pp_d, 0.01},
                       {"G4MuPairProductionModel xs", &pp_x, 0.01}};
  for (const Row& r : rows) {
    printf("  %-34s %8d %11.4f%%   %s\n", r.label, r.a->n, 100 * r.a->worst,
           r.a->where.c_str());
  }

  int fails = 0;
  for (const Row& r : rows) {
    if (r.a->n == 0) {
      printf("\n  FAIL: %s compared nothing\n", r.label);
      ++fails;
    } else if (r.a->worst > r.tol) {
      printf("\n  FAIL: %s exceeds %.2f%%\n", r.label, 100 * r.tol);
      ++fails;
    }
  }

  printf("\n%s (%d failures)\n", fails ? "FAILED" : "PASSED", fails);
  return fails ? 1 : 0;
}
