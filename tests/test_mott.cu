// The Mott/Rutherford ratio, against G4ScreeningMottCrossSection.
//
// G4WentzelOKandVIxSection creates one of these for electrons and positrons unconditionally -
// "Mott corrections always added" is the comment in Geant4 - and uses
// RatioMottRutherfordCosT as the rejection function when it samples a single scatter. Every
// other particle keeps an analytic Rutherford-plus-spin expression, and the analytic one is
// what this port had. So for e+- this is not a refinement of the single-scattering angular
// distribution; it is the distribution.
//
// Three things can be wrong and they fail differently, so they are separated:
//
//   1. the coefficient table, 2790 numbers out of G4MottData.hh;
//   2. beta, which is not the lab beta - G4ScreeningMottCrossSection works in the
//      projectile-nucleus relative system, so beta depends on the target nuclear mass;
//   3. the double polynomial itself.
//
// The oracle cannot dump beta (it is private with no accessor), so the split is done by
// angle: at fcost = 0 the ratio collapses to
//
//     R(0) = sum_k coef[Z][0][k] * (beta - 0.7181228)^k
//
// which involves beta and one row of coefficients and nothing else. A failure confined to
// fcost = 0 is beta or that row; a failure only at fcost > 0 is the angular polynomial.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "core/particle.cuh"
#include "core/rng.cuh"
#include "core/units.cuh"
#include "core/vec3.cuh"
#include "data/materials.cuh"
#include "data/mott.hh"
#include "physics/em/wentzel_msc.cuh"

using namespace g4gpu;
using real_t = double;

int main() {
  const char* env = std::getenv("G4GPU_ORACLE");
  const std::string dir = (env != nullptr) ? env : "ref/oracle";
  int fails = 0;

  // ---------------------------------------------------------- the target nuclear masses
  //
  // These are read from the same CSV the generator read, so agreeing proves only that the
  // generator copied them. What makes them a real check is the ratio below: beta depends on
  // the target mass, so a wrong mass moves R(0). This section is here to catch the other
  // failure - a table that lost or shifted a row - which the ratio would show as a wrong
  // answer for one element and would take much longer to read.
  {
    FILE* f = std::fopen((dir + "/mott_target.csv").c_str(), "r");
    if (f == nullptr) {
      std::printf("cannot read %s/mott_target.csv\n", dir.c_str());
      return 1;
    }
    char line[512];
    if (std::fgets(line, sizeof line, f) == nullptr) {
      std::fclose(f);
      return 1;
    }
    int n = 0, bad_mass = 0, bad_a = 0;
    while (std::fgets(line, sizeof line, f) != nullptr) {
      int z = 0, a = 0;
      double amu = 0, m = 0;
      if (std::sscanf(line, "%d,%d,%lf,%lf", &z, &a, &amu, &m) != 4) { continue; }
      ++n;
      if (data::mott_target_mass<real_t>(z) != m) { ++bad_mass; }
      if (data::mott_target_nucleons(z) != a) { ++bad_a; }
    }
    std::fclose(f);
    std::printf("== target nuclei ==\n  %d elements, %d wrong masses, %d wrong nucleon "
                "counts\n",
                n, bad_mass, bad_a);
    if (n != 92) {
      std::printf("  FAIL: %d elements, expected 92\n", n);
      ++fails;
    }
    if (bad_mass != 0 || bad_a != 0) {
      std::printf("  FAIL: the generated table does not match the oracle\n");
      ++fails;
    }
    // Spot-check the clamp Geant4 applies: std::min(92, Z).
    if (data::mott_target_mass<real_t>(95) != data::mott_target_mass<real_t>(92)) {
      std::printf("  FAIL: Z > 92 is not clamped to 92 as SetupKinematic clamps it\n");
      ++fails;
    }
  }

  // ---------------------------------------------------------- the ratio itself
  {
    FILE* f = std::fopen((dir + "/mott_ratio.csv").c_str(), "r");
    if (f == nullptr) {
      std::printf("cannot read %s/mott_ratio.csv\n", dir.c_str());
      return 1;
    }
    char line[512];
    if (std::fgets(line, sizeof line, f) == nullptr) {
      std::fclose(f);
      return 1;
    }

    struct Acc {
      double worst = 0;
      std::string where;
      int n = 0;
    };
    Acc head_on;   // fcost == 0: beta and coefficient row 0
    Acc angular;   // fcost > 0: the polynomial in the angle
    int z_seen[93] = {};
    int e_minus = 0, e_plus = 0;

    const real_t me = units::electron_mass_c2<real_t>();

    while (std::fgets(line, sizeof line, f) != nullptr) {
      char part[64] = {0};
      int z = 0;
      double ekin = 0, fcost = 0, g4 = 0;
      if (std::sscanf(line, "%63[^,],%d,%lf,%lf,%lf", part, &z, &ekin, &fcost, &g4) != 5) {
        continue;
      }
      if (z < 1 || z > 92) { continue; }
      z_seen[z] = 1;
      const bool minus = (std::strcmp(part, "e-") == 0);
      if (minus) { ++e_minus; } else { ++e_plus; }

      // Both charges have the electron mass, and the ratio depends on the mass and not the
      // sign - which is why the two species should give the same number and the oracle dumps
      // both: if they differed, one of them would be reading the wrong row.
      const real_t beta = data::mott_beta<real_t>(z, static_cast<real_t>(ekin), me);
      const real_t ours = data::mott_ratio<real_t>(z, beta, static_cast<real_t>(fcost));

      if (std::fabs(g4) <= 1e-300) { continue; }
      Acc& a = (fcost == 0.0) ? head_on : angular;
      ++a.n;
      const double dev = std::fabs(double(ours) / g4 - 1);
      if (dev > a.worst) {
        a.worst = dev;
        char what[220];
        std::snprintf(what, sizeof what, "%s Z=%d at %.4g MeV, fcost=%.4g", part, z, ekin,
                      fcost);
        a.where = what;
      }
    }
    std::fclose(f);

    int nz = 0;
    for (int z = 1; z <= 92; ++z) { nz += z_seen[z]; }

    std::printf("\n== Mott/Rutherford ratio vs G4ScreeningMottCrossSection ==\n");
    std::printf("  %-34s %8s %14s   %s\n", "quantity", "points", "worst dev", "where");
    std::printf("  %-34s %8d %13.3e   %s\n", "fcost = 0 (beta, coef row 0)", head_on.n,
                head_on.worst, head_on.where.c_str());
    std::printf("  %-34s %8d %13.3e   %s\n", "fcost > 0 (angular polynomial)", angular.n,
                angular.worst, angular.where.c_str());
    std::printf("  %d elements, %d e- points, %d e+ points\n", nz, e_minus, e_plus);

    if (head_on.n == 0 || angular.n == 0) {
      std::printf("\n  FAIL: one of the two halves was never exercised\n");
      ++fails;
    }
    if (nz < 6) {
      std::printf("\n  FAIL: only %d elements - beta depends on the target nuclear mass, so\n"
                  "        a single Z proves nothing about the mass table\n",
                  nz);
      ++fails;
    }
    if (e_minus == 0 || e_plus == 0) {
      std::printf("\n  FAIL: the oracle covers only one of e- and e+\n");
      ++fails;
    }
    // 1e-9: the same coefficients, the same polynomial, in double. beta goes through a couple
    // of square roots of quantities built from a dumped nuclear mass, which is where the last
    // few bits go.
    if (head_on.worst > 1e-9) {
      std::printf("\n  FAIL: the fcost = 0 ratio is %.3e from Geant4 (tolerance 1e-9).\n"
                  "        That is beta or coefficient row 0, not the angular polynomial.\n"
                  "        worst at %s\n",
                  head_on.worst, head_on.where.c_str());
      ++fails;
    }
    if (angular.worst > 1e-9) {
      std::printf("\n  FAIL: the angular ratio is %.3e from Geant4 (tolerance 1e-9).\n"
                  "        worst at %s\n",
                  angular.worst, angular.where.c_str());
      ++fails;
    }
  }

  // ---------------------------------------------------------- the sampler takes the branch
  //
  // The ratio being right is one claim; the sampler using it is another, and that one would
  // fail silently. `wv_sample_single` picks the rejection function on `s.use_mott`, which
  // `wentzel_setup` sets for e- and e+ and nobody else. If that flag never became true, or the
  // branch were ordered wrong, every number above would still be exact and the sampler would
  // still be using the analytic Rutherford-plus-spin expression it used before.
  //
  // So the same scatter is sampled twice off the same stream - once with the flag as
  // wentzel_setup set it, once with it forced off - and the two acceptance rates have to
  // differ. They should: the Mott ratio is not the Rutherford expression, and on gold at
  // 100 MeV it is not close to it.
  {
    std::printf("\n== the sampler takes the Mott branch ==\n");
    data::MaterialTable<real_t> t{};
    t.count = 0;
    const int zs[1] = {79};  // gold: the largest Z where the ratio departs most from Rutherford
    const real_t w[1] = {real_t(1)};
    const int mi = data::add_material<real_t>(t, real_t(19.3), 1, zs, w, real_t(790.0));
    if (mi < 0) {
      std::printf("  FAIL: could not build gold\n");
      ++fails;
    } else {
      const data::Material<real_t>& m = t.m[mi];
      const ParticleDef<real_t> pd = particle_def<real_t>(ParticleType::kElectron);
      std::printf("  %10s %14s %14s %10s\n", "E/MeV", "Mott accept", "analytic", "differ");
      double worst = 0;
      int flag_wrong = 0;
      // Low energies first: the Mott ratio departs from the Rutherford expression most where
      // beta is well below the 0.7181228 the fit is centred on, and the two converge as
      // beta -> 1. At 1 GeV they agree to 0.02%, which is why the sweep does not start there.
      const real_t energies[4] = {real_t(0.1), real_t(0.3), real_t(1.0), real_t(10.0)};
      for (int i = 0; i < 4; ++i) {
        const real_t e = energies[i];
        em::WentzelState<real_t> s =
            em::wentzel_setup(pd, ParticleType::kElectron, e, m.inv_a23, 79, real_t(1e-3),
                              real_t(-1));
        if (!s.use_mott) { ++flag_wrong; }
        em::WentzelState<real_t> s_off = s;
        s_off.use_mott = false;
        // The same stream for both, so the only difference is the rejection function.
        int acc_mott = 0, acc_plain = 0;
        constexpr int kN = 20000;
        for (int k = 0; k < kN; ++k) {
          Philox<real_t> r1(static_cast<uint32_t>(k), 1u, 7u);
          Philox<real_t> r2(static_cast<uint32_t>(k), 1u, 7u);
          const Vec3<real_t> a = em::wv_sample_single(s, 79, real_t(1), s.cos_tet_max_nuc,
                                                      real_t(0), r1);
          const Vec3<real_t> b = em::wv_sample_single(s_off, 79, real_t(1), s.cos_tet_max_nuc,
                                                      real_t(0), r2);
          // z == 1 exactly is the "rejected" return: the sampler leaves the direction alone.
          if (a.z != real_t(1)) { ++acc_mott; }
          if (b.z != real_t(1)) { ++acc_plain; }
        }
        const double pm = double(acc_mott) / kN, pp = double(acc_plain) / kN;
        const double d = (pp > 0) ? std::fabs(pm / pp - 1) : 0.0;
        worst = std::fmax(worst, d);
        std::printf("  %10.4g %14.4f %14.4f %9.2f%%\n", double(e), pm, pp, 100 * d);
      }
      if (flag_wrong != 0) {
        std::printf("  FAIL: wentzel_setup left use_mott false for an electron\n");
        ++fails;
      }
      // 1%: the two rejection functions are different functions. If the acceptance rates
      // agreed this closely at every energy, the sampler would not be reading the ratio.
      if (worst < 0.01) {
        std::printf("  FAIL: the Mott and analytic acceptance rates agree to %.3f%% - the\n"
                    "        sampler is not taking the Mott branch\n",
                    100 * worst);
        ++fails;
      }
    }
  }

  std::printf("\n%s (%d failures)\n", fails ? "FAILED" : "PASSED", fails);
  return fails ? 1 : 0;
}
