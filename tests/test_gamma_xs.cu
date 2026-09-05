// Validates the photon cross sections without Geant4:
//  - Geant4 parameterized Compton per atom, divided by Z, vs the EXACT Klein-Nishina
//    per-electron formula (binding is negligible at MeV energies, so they must agree)
//  - the same formula reduces to the Thomson limit as E -> 0
//  - pair production respects the 2*m_e threshold exactly
//  - resulting mass attenuation coefficients land on tabulated values
#include <cstdio>
#include <cmath>
#include "data/materials.cuh"
#include "physics/em/gamma_processes.cuh"

using namespace g4gpu;
using real_t = double;

static int fails = 0;
static void check(bool ok, const char* what) {
  printf(ok ? "  ok:   %s\n" : "  FAIL: %s\n", what);
  if (!ok) { ++fails; }
}

/// Exact Klein-Nishina total cross section per free electron, mm^2.
/// Standard closed form; reduces to Thomson (8/3)*pi*r_e^2 as eps -> 0.
static real_t kn_exact_per_electron(real_t energy) {
  const real_t re = units::classic_electron_radius<real_t>();
  const real_t eps = energy / units::electron_mass_c2<real_t>();
  const real_t t = 1.0 + 2.0 * eps;
  const real_t lt = std::log1p(2.0 * eps);  // log1p: log(1+2eps) cancels catastrophically as eps->0
  return 2.0 * units::pi<real_t>() * re * re
         * ((1.0 + eps) / (eps * eps) * (2.0 * (1.0 + eps) / t - lt / eps) + lt / (2.0 * eps)
            - (1.0 + 3.0 * eps) / (t * t));
}

int main() {
  const real_t barn = units::barn<real_t>();

  printf("== exact KN formula sanity: Thomson limit ==\n");
  const real_t thomson = 8.0 / 3.0 * units::pi<real_t>()
                         * units::classic_electron_radius<real_t>()
                         * units::classic_electron_radius<real_t>();
  const real_t kn_low = kn_exact_per_electron(1e-6);  // 1 eV
  printf("  KN(1 eV) = %.6f barn, Thomson = %.6f barn\n", kn_low / barn, thomson / barn);
  check(std::fabs(kn_low / thomson - 1.0) < 1e-4, "exact KN reduces to Thomson at low E");

  printf("== Geant4 Compton parameterization vs exact KN per electron ==\n");
  const real_t energies[5] = {1.0, 6.0, 10.0, 50.0, 100.0};
  const int zs[4] = {1, 8, 20, 82};
  for (int e = 0; e < 5; ++e) {
    const real_t exact = kn_exact_per_electron(energies[e]);
    printf("  E = %6.1f MeV : exact KN/e- = %.5f barn\n", energies[e], exact / barn);
    for (int i = 0; i < 4; ++i) {
      const real_t Z = real_t(zs[i]);
      const real_t per_e = em::compton_xs_per_atom<real_t>(energies[e], Z) / Z;
      const real_t ratio = per_e / exact;
      printf("      Z=%2d: param/Z = %.5f barn   ratio = %.4f\n", zs[i], per_e / barn, ratio);
      // Binding and Z-dependent corrections are small at these energies.
      if (energies[e] >= 1.0) {
        check(std::fabs(ratio - 1.0) < 0.10, "param Compton within 10% of exact KN per electron");
      }
    }
  }

  printf("== pair production threshold ==\n");
  const real_t me2 = 2.0 * units::electron_mass_c2<real_t>();
  check(em::pair_xs_per_atom<real_t>(me2 - 1e-6, 8.0) == 0.0, "pair XS is zero below 2*m_e");
  check(em::pair_xs_per_atom<real_t>(me2 + 0.1, 8.0) > 0.0, "pair XS is positive above 2*m_e");
  check(em::pair_xs_per_atom<real_t>(6.0, 8.0) > em::pair_xs_per_atom<real_t>(2.0, 8.0),
        "pair XS rises with energy");
  check(em::pair_xs_per_atom<real_t>(6.0, 20.0) > em::pair_xs_per_atom<real_t>(6.0, 8.0),
        "pair XS rises with Z");

  printf("== mass attenuation coefficients at 6 MeV ==\n");
  data::Material<real_t> mats[data::kNumMaterials];
  data::build_b1_materials<real_t>(mats);
  const char* names[4] = {"air", "water", "A-150 tissue", "bone compact"};
  for (int i = 0; i < data::kNumMaterials; ++i) {
    const auto xs = em::gamma_macroscopic_xs<real_t>(mats[i], 6.0);
    // mu [1/mm] -> mu/rho [cm^2/g]:  *10 mm/cm  / rho
    const real_t mu_rho = xs.total * 10.0 / mats[i].density;
    const real_t mfp_cm = (xs.total > 0) ? 1.0 / (xs.total * 10.0) : 0.0;
    printf("  %-13s mu/rho = %.5f cm2/g   mfp = %7.2f cm   (Compton %.0f%%, pair %.0f%%)\n",
           names[i], mu_rho, mfp_cm, 100.0 * xs.compton / xs.total, 100.0 * xs.pair / xs.total);
  }
  // NIST/XCOM for water at 6 MeV: mu/rho ~ 0.0277 cm2/g including coherent, which we omit.
  const auto w = em::gamma_macroscopic_xs<real_t>(mats[data::kWater], 6.0);
  const real_t mu_rho_water = w.total * 10.0 / mats[data::kWater].density;
  check(mu_rho_water > 0.020 && mu_rho_water < 0.032,
        "water mu/rho at 6 MeV is near the tabulated 0.0277 cm2/g");

  printf("== process selection ==\n");
  const auto b = em::gamma_macroscopic_xs<real_t>(mats[data::kBoneCompact], 6.0);
  int n_compton = 0, n_pair = 0;
  for (int i = 0; i < 100000; ++i) {
    const real_t u = (real_t(i) + 0.5) / 100000.0;
    if (em::select_gamma_process(b, u) == em::GammaProcess::kCompton) { ++n_compton; } else { ++n_pair; }
  }
  const real_t frac = real_t(n_compton) / 100000.0;
  printf("  bone: selected Compton %.4f of the time, XS fraction %.4f\n", frac, b.compton / b.total);
  check(std::fabs(frac - b.compton / b.total) < 1e-3, "selection frequency tracks XS ratio");

  printf("\n%s (%d failures)\n", fails ? "FAILED" : "ALL PASS", fails);
  return fails;
}
