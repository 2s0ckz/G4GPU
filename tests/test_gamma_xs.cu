// Validates the photon cross sections without Geant4:
//  - Geant4 parameterized Compton per atom, divided by Z, vs the EXACT Klein-Nishina
//    per-electron formula (binding is negligible at MeV energies, so they must agree)
//  - the same formula reduces to the Thomson limit as E -> 0
//  - pair production respects the 2*m_e threshold exactly
//  - resulting mass attenuation coefficients land on tabulated values
//  - P21: under G4GammaGeneralProcess the sum has NO Rayleigh term from 2 m_e up - its zones 2
//    and 3 are `sigComp + sigConv + sigPE (+ sigN)` - and below 2 m_e it has the same one as
//    without the general process; the edge is `minEEEnergy` itself, which is zone 2
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>
#include "data/materials.cuh"
#include "host/g4data.cuh"
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

  // ---- P21: Rayleigh and the general process's zones (docs/RISK.md V208 and the P21 entry).
  //
  // `G4GammaGeneralProcess::BuildPhysicsTable` sums `sigComp + sigR` in zone 0 and
  // `sigComp + sigR + sigPE` in zone 1, and NO `sigR` in zones 2 and 3; `TotalCrossSectionPerVolume`
  // puts a photon in zone 2 when `preStepKinEnergy < minEEEnergy` is false, so 2 m_e itself has
  // no Rayleigh term. Counted in the running Geant4 (ref/gammagp): 0 Rayleigh scatters in forty
  // million first interactions at 1.5 MeV in water and in bone, where the cross section's share
  // is 17,293 and 30,846 of them. Below the edge the general process changes nothing here, and
  // above it the other three terms are the same doubles - the condition is on Rayleigh alone.
  printf("== Rayleigh under G4GammaGeneralProcess ==\n");
  {
    const std::string phot_dir = host::g4emlow_subdir("epics2017/phot", "pe-cs-1.dat");
    const std::string rayl_dir = host::g4emlow_subdir("epics2017/rayl", "re-cs-1.dat");
    const int zl[] = {1, 6, 7, 8, 12, 15, 16, 18, 20};
    const int nzl = static_cast<int>(sizeof(zl) / sizeof(zl[0]));
    static data::PhotoElectricTable<real_t> pe{};
    static std::vector<real_t> pte, ptv;
    static data::RayleighTable<real_t> rt{};
    static std::vector<real_t> rte, rtv;
    const bool pe_ok = data::load_photoelectric<real_t>(phot_dir, zl, nzl, pe, pte, ptv);
    const bool ra_ok = data::load_rayleigh<real_t>(rayl_dir, zl, nzl, rt, rte, rtv);
    check(pe_ok && ra_ok, "the photoelectric and Rayleigh tables load");
    if (pe_ok && ra_ok) {
      pe.table_e = pte.data();
      pe.table_v = ptv.data();
      rt.table_e = rte.data();
      rt.table_v = rtv.data();
      const real_t ee = em::gamma_general_min_ee<real_t>();
      check(ee == 2.0 * units::electron_mass_c2<real_t>(), "minEEEnergy is 2 m_e");
      // The edge from both sides, the zone-1 and zone-2 interiors, and zone 3.
      const real_t es[] = {0.2, 0.8, ee * (1.0 - 1e-12), ee, 1.5, 6.0, 22.0, 99.9, 100.0, 500.0};
      int bad_on = 0, bad_off = 0, bad_rest = 0, bad_sel = 0;
      for (int m = 0; m < data::kNumMaterials; ++m) {
        for (const real_t e : es) {
          const auto off = em::gamma_macroscopic_xs<real_t>(mats[m], e, &pe, &rt, true, true, false);
          const auto on = em::gamma_macroscopic_xs<real_t>(mats[m], e, &pe, &rt, true, true, true);
          const bool zone01 = (e < ee);
          // Rayleigh: present in both below the edge, absent with the general process from it.
          if (!(off.rayleigh > 0)) { ++bad_off; }
          if (zone01 ? (on.rayleigh != off.rayleigh) : (on.rayleigh != 0.0)) { ++bad_on; }
          // The other three are the same doubles, and the total is their sum.
          if (on.compton != off.compton || on.pair != off.pair
              || on.photoelectric != off.photoelectric
              || on.total != on.compton + on.pair + on.photoelectric + on.rayleigh
              || (zone01 && on.total != off.total)) {
            ++bad_rest;
          }
          // A walk that falls past the photoelectric slice - a rounding residue at q -> 1 - takes
          // the last process IN the sum, never one the sum does not have.
          if (!zone01 && em::select_gamma_process(on, 1.0) == em::GammaProcess::kRayleigh) {
            ++bad_sel;
          }
          if (zone01 && em::select_gamma_process(off, 1.0) != em::GammaProcess::kRayleigh) {
            ++bad_sel;
          }
          if (m == data::kWater && (e == 1.5 || e == 6.0)) {
            printf("  water %4.1f MeV: Rayleigh %.4e /mm is %.3e of the sum without the general "
                   "process, and %.4e /mm with it\n", e, off.rayleigh, off.rayleigh / off.total,
                   on.rayleigh);
          }
        }
      }
      check(bad_off == 0, "Rayleigh is positive at every energy without the general process");
      check(bad_on == 0, "with it, Rayleigh is the same below 2 m_e and exactly 0 from 2 m_e up");
      check(bad_rest == 0, "Compton, conversion and photoelectric are untouched, and the total "
                           "is their sum");
      check(bad_sel == 0, "a walk past the last slice never selects a Rayleigh not in the sum");
    }
  }

  printf("\n%s (%d failures)\n", fails ? "FAILED" : "ALL PASS", fails);
  return fails;
}
