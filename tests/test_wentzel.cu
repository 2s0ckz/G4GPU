// The Wentzel single-scattering cross section, against G4WentzelOKandVIxSection.
//
// This is the engine behind G4WentzelVIModel (multiple scattering above 100 MeV for e-/e+,
// and for every charged hadron) and G4eCoulombScatteringModel. Both are registered by
// G4EmStandardPhysics. The oracle drives the engine class directly, per element, so a
// mismatch localises to the cross section rather than to whichever model was selected.
//
// A note on conditioning, because it dominates how this test is written. The cut-off angles
// approach 1 as the energy rises: at 1 TeV, 1 - cos is of order 1e-15, and it is computed as
// a difference of momenta of order 1e12. That subtraction has no significant digits left in
// double precision - in Geant4 as much as here - so those points carry no information about
// the model and are counted and skipped rather than compared. Comparing cos itself instead
// would hide the problem rather than avoid it: an absolute agreement of 1e-9 on a quantity
// whose true value is 1e-15 is a 10^6 relative error being reported as a pass.
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <string>
#include "core/particle.cuh"
#include "data/materials.cuh"
#include "physics/em/wentzel_xs.cuh"

using namespace g4gpu;
using real_t = double;

/// Below this, 1 - cos is at the double-precision floor of the subtraction that produced it.
static constexpr double kResolvable = 1e-10;

static ParticleType type_of(const std::string& n) {
  if (n == "e-") { return ParticleType::kElectron; }
  if (n == "e+") { return ParticleType::kPositron; }
  if (n == "mu-") { return ParticleType::kMuonMinus; }
  if (n == "proton") { return ParticleType::kProton; }
  if (n == "pi+") { return ParticleType::kPionPlus; }
  return ParticleType::kNumTypes;
}

int main() {
  data::Material<real_t> mats[data::kNumMaterials];
  data::build_b1_materials<real_t>(mats);
  const char* g4names[4] = {"G4_AIR", "G4_WATER", "G4_A-150_TISSUE", "G4_BONE_COMPACT_ICRU"};
  const char* env = std::getenv("G4GPU_ORACLE");
  const std::string dir = (env != nullptr) ? env : "ref/oracle";
  int fails = 0;

  // The material constant every cut-off angle keys on.
  printf("== material <A^(-2/3)> vs G4IonisParamMat::GetInvA23 ==\n");
  {
    FILE* f = std::fopen((dir + "/ionisation_params.csv").c_str(), "r");
    if (f == nullptr) {
      std::printf("cannot read %s/ionisation_params.csv\n", dir.c_str());
      return 1;
    }
    char line[1024];
    std::fgets(line, sizeof line, f);
    printf("  %-22s %20s %20s %10s\n", "material", "ours", "Geant4", "ratio");
    while (std::fgets(line, sizeof line, f) != nullptr) {
      char mat[128];
      double v[13];
      if (std::sscanf(line, "%127[^,],%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf",
                      mat, &v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &v[6], &v[7], &v[8],
                      &v[9], &v[10], &v[11], &v[12]) != 14) {
        continue;
      }
      int mi = -1;
      for (int i = 0; i < data::kNumMaterials; ++i) {
        if (std::string(mat) == g4names[i]) { mi = i; break; }
      }
      if (mi < 0) { continue; }
      const double r = mats[mi].inv_a23 / v[11];
      printf("  %-22s %20.12g %20.12g %10.6f\n", mat, mats[mi].inv_a23, v[11], r);
      // Geant4 divides by G4Element::GetN()^(2/3), the abundance-weighted nucleon number.
      // For most elements that equals the standard atomic weight this port carries; a
      // couple differ in the last digits.
      if (std::fabs(r - 1) > 1e-5) { ++fails; }
    }
    std::fclose(f);
  }

  FILE* f = std::fopen((dir + "/wentzel.csv").c_str(), "r");
  if (f == nullptr) {
    std::printf("cannot read %s/wentzel.csv\n", dir.c_str());
    return 1;
  }
  char line[512];
  if (std::fgets(line, sizeof line, f) == nullptr) { std::fclose(f); return 1; }

  struct Acc { double worst; std::string where; int n; };
  Acc tx{0, "", 0}, nx{0, "", 0}, ex{0, "", 0}, ctn{0, "", 0}, cte{0, "", 0};
  int skipped = 0, total = 0;
  auto track = [](Acc& a, double ours, double g4, const char* p, const char* m, int z,
                  double e) {
    if (g4 <= 0) { return; }
    ++a.n;
    const double dev = std::fabs(ours / g4 - 1);
    if (dev > a.worst) {
      a.worst = dev;
      char buf[180];
      std::snprintf(buf, sizeof buf, "%s on Z=%d in %s at %.4g MeV", p, z, m, e);
      a.where = buf;
    }
  };

  while (std::fgets(line, sizeof line, f) != nullptr) {
    char mat[128], part[32];
    int z;
    double e, cut, coslim, t, n, el, one_cn, one_ce, sz;
    if (std::sscanf(line, "%127[^,],%31[^,],%d,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf", mat, part,
                    &z, &e, &cut, &coslim, &t, &n, &el, &one_cn, &one_ce, &sz) != 12) {
      continue;
    }
    const ParticleType pt = type_of(part);
    if (pt == ParticleType::kNumTypes) { continue; }
    int mi = -1;
    for (int i = 0; i < data::kNumMaterials; ++i) {
      if (std::string(mat) == g4names[i]) { mi = i; break; }
    }
    if (mi < 0) { continue; }
    ++total;
    // Both cut-off angles must still be resolvable for any of these numbers to mean
    // anything: the cross sections are built from the same differences.
    if (one_cn < kResolvable || one_ce < kResolvable) {
      ++skipped;
      continue;
    }
    const auto pd = particle_def<real_t>(pt);
    const auto s = em::wentzel_setup(pd, pt, real_t(e), mats[mi].inv_a23, z, real_t(cut),
                                     real_t(coslim));
    track(ctn, real_t(1) - s.cos_tet_max_nuc, one_cn, part, mat, z, e);
    track(cte, real_t(1) - s.cos_tet_max_elec, one_ce, part, mat, z, e);
    track(tx, em::wentzel_transport_xs_per_atom(s, z, s.cos_tet_max_nuc), t, part, mat, z, e);
    track(nx, em::wentzel_nuclear_xs(s, z, real_t(1), s.cos_tet_max_nuc), n, part, mat, z, e);
    track(ex, em::wentzel_electron_xs(s, real_t(1), s.cos_tet_max_nuc), el, part, mat, z, e);
  }
  std::fclose(f);

  printf("\n== Wentzel cross sections vs G4WentzelOKandVIxSection ==\n");
  printf("  %d of %d rows compared; %d skipped as unresolvable in double precision\n\n",
         total - skipped, total, skipped);
  printf("  %-32s %8s %12s   %s\n", "quantity", "points", "worst dev", "where");
  struct Row { const char* label; const Acc* a; };
  const Row rows[5] = {{"1 - cos(theta) max, nuclear", &ctn},
                       {"1 - cos(theta) max, electron", &cte},
                       {"transport cross section", &tx},
                       {"nuclear cross section", &nx},
                       {"electron cross section", &ex}};
  for (const Row& r : rows) {
    printf("  %-32s %8d %11.4f%%   %s\n", r.label, r.a->n, 100 * r.a->worst,
           r.a->where.c_str());
  }

  for (const Row& r : rows) {
    if (r.a->n == 0) {
      printf("\n  FAIL: %s compared nothing\n", r.label);
      ++fails;
    } else if (r.a->worst > 0.001) {
      printf("\n  FAIL: %s exceeds 0.1%%\n", r.label);
      ++fails;
    }
  }
  printf("\n%s (%d failures)\n", fails ? "FAILED" : "PASSED", fails);
  return fails ? 1 : 0;
}
