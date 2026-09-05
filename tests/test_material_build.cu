// Building a material the way a user builds one: from elements, with nothing else supplied.
//
// This is the path the model-building GUI takes, and it produced a **NaN dose**. A material
// with no tabulated mean excitation energy reached the device record with `mean_excitation = 0`,
// the density-effect parameters are built from `1 + 2 ln(I / plasma_energy)`, `log(0)` is
// negative infinity, and every dE/dx after that was NaN. Nothing caught it because every
// material in every other test and in the oracle had its excitation energy set explicitly.
//
// Geant4 does not pass zero through: `G4IonisParamMat::ComputeMeanParameters` derives one by
// Bragg additivity in the logarithm over the elements. `src/data/nist_excitation.hh` does the
// same, and `ref/oracle/material_ionis.csv` now carries a material - `CustomDerivedI`, three
// elements from hydrogen to lead, no `SetMeanExcitationEnergy` - built precisely so that the
// derivation has something behind it.
//
// Three sections: the derived value against Geant4's, the record built from it against
// Geant4's own derived quantities, and the thing that actually went wrong - that a dE/dx
// computed in such a material is a number.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "data/materials.cuh"
#include "data/nist_excitation.hh"
#include "physics/em/hadron_ionisation.cuh"
#include "physics/em/bragg.cuh"
#include "physics/em/electron_processes.cuh"

using namespace g4gpu;
using real_t = double;

namespace {

// CustomDerivedI, as ref/dump/g4dump.cc builds it.
constexpr int kN = 3;
const int kZs[kN] = {1, 6, 82};
const real_t kW[kN] = {real_t(0.1), real_t(0.6), real_t(0.3)};
constexpr real_t kDensity = real_t(2.5);

real_t a_of_z(int z) { return data::atomic_mass<real_t>(z); }

}  // namespace

int main() {
  const char* env = std::getenv("G4GPU_ORACLE");
  const std::string dir = (env != nullptr) ? env : "ref/oracle";
  int fails = 0;

  // The oracle's row for the material built with no excitation energy.
  double g4_zeff = 0, g4_fermi = 0, g4_inva23 = 0, g4_lf = 0, g4_iexc = 0, g4_ne = 0;
  bool found = false;
  {
    FILE* f = std::fopen((dir + "/material_ionis.csv").c_str(), "r");
    if (f == nullptr) {
      std::printf("cannot read %s/material_ionis.csv\n", dir.c_str());
      return 1;
    }
    char line[512];
    if (std::fgets(line, sizeof line, f) == nullptr) {
      std::fclose(f);
      return 1;
    }
    while (std::fgets(line, sizeof line, f) != nullptr) {
      char name[128] = {0};
      double z = 0, ef = 0, ia = 0, lf = 0, mex = 0, ne = 0;
      if (std::sscanf(line, "%127[^,],%lf,%lf,%lf,%lf,%lf,%lf", name, &z, &ef, &ia, &lf, &mex,
                      &ne)
          != 7) {
        continue;
      }
      if (std::strcmp(name, "CustomDerivedI") != 0) { continue; }
      g4_zeff = z;
      g4_fermi = ef;
      g4_inva23 = ia;
      g4_lf = lf;
      g4_iexc = mex * 1e6;  // MeV -> eV
      g4_ne = ne;
      found = true;
    }
    std::fclose(f);
  }
  if (!found) {
    std::printf("FAIL: the oracle has no CustomDerivedI row - the derivation is untested.\n"
                "      ref/dump/g4dump.cc builds it; re-run ref/dump/run.bat.\n");
    return 1;
  }

  // ------------------------------------------------- 1. the derived mean excitation energy
  std::printf("== derived mean excitation energy ==\n");
  const double ours_i =
      data::derive_mean_excitation_eV(kN, kZs, kW, [](int z) { return double(a_of_z(z)); });
  std::printf("  per-element I: H %g, C %g, Pb %g eV\n", data::nist_element_excitation_eV(1),
              data::nist_element_excitation_eV(6), data::nist_element_excitation_eV(82));
  std::printf("  ours %.10g eV, Geant4 %.10g eV, dev %.3e\n", ours_i, g4_iexc,
              (g4_iexc > 0) ? std::fabs(ours_i / g4_iexc - 1) : 1.0);
  if (!(ours_i > 0)) {
    std::printf("  FAIL: nothing was derived\n");
    ++fails;
  } else if (std::fabs(ours_i / g4_iexc - 1) > 1e-12) {
    // 1e-12: the same per-element values, the same weights, the same logarithm. The oracle
    // prints 17 significant digits, so there is nothing else in the way.
    std::printf("  FAIL: %.3e from Geant4, tolerance 1e-12\n",
                std::fabs(ours_i / g4_iexc - 1));
    ++fails;
  }
  // The per-element lookup is by Z over the generated NIST table, so a shifted or missing
  // elemental entry would show up here rather than as a slightly wrong average.
  if (data::nist_element_excitation_eV(1) != 19.2 ||
      data::nist_element_excitation_eV(6) != 78.0 ||
      data::nist_element_excitation_eV(82) != 823.0) {
    std::printf("  FAIL: the per-element table does not hold Geant4's values for H, C and Pb\n");
    ++fails;
  }
  if (data::nist_element_excitation_eV(0) != 0 || data::nist_element_excitation_eV(200) != 0) {
    std::printf("  FAIL: an out-of-range Z should give 0, not a value\n");
    ++fails;
  }

  // ------------------------------------------------- 2. the record built from it
  std::printf("\n== the device record, against Geant4's own derived quantities ==\n");
  data::MaterialTable<real_t> table{};
  table.count = 0;
  const int idx = data::add_material<real_t>(table, kDensity, kN, kZs, kW,
                                             static_cast<real_t>(ours_i));
  if (idx < 0) {
    std::printf("  FAIL: add_material refused the material\n");
    std::printf("\nFAILED (%d failures)\n", fails + 1);
    return 1;
  }
  const data::Material<real_t>& m = table.m[idx];

  struct Row {
    const char* what;
    double ours;
    double g4;
    double tol;
  };
  real_t lf_ours = 0, norm = 0;
  for (int i = 0; i < m.n_elements; ++i) {
    const int zi = static_cast<int>(m.z[i] + real_t(0.5));
    lf_ours += m.n_atoms[i] * static_cast<real_t>(data::ziegler_l_factor(zi));
    norm += m.n_atoms[i];
  }
  lf_ours = (norm > 0) ? lf_ours / norm : real_t(0);

  const Row rows[] = {
      {"Zeff", double(m.z_eff), g4_zeff, 1e-12},
      {"Fermi energy / MeV", double(m.fermi_energy), g4_fermi, 1e-12},
      {"<A^-2/3>", double(m.inv_a23), g4_inva23, 1e-12},
      {"L-factor", double(lf_ours), g4_lf, 1e-12},
      {"electron density /mm3", double(m.electron_density), g4_ne, 1e-12},
      {"mean excitation / eV", double(m.mean_excitation) * 1e6, g4_iexc, 1e-12},
  };
  std::printf("  %-24s %20s %20s %10s\n", "quantity", "ours", "Geant4", "dev");
  for (const Row& r : rows) {
    const double dev = (r.g4 != 0) ? std::fabs(r.ours / r.g4 - 1) : std::fabs(r.ours);
    std::printf("  %-24s %20.12g %20.12g %10.2e%s\n", r.what, r.ours, r.g4, dev,
                (dev > r.tol) ? "   <-- FAIL" : "");
    if (dev > r.tol) { ++fails; }
  }

  // ------------------------------------------------- 3. the dE/dx is a number
  //
  // This is the regression. Every quantity above can be right and the run can still produce
  // NaN if the density-effect parameters were built from log(0), so the thing the user
  // actually saw is checked directly: a dose is an integral of dE/dx, and a NaN anywhere in
  // it is a NaN dose.
  std::printf("\n== dE/dx is finite in a material built this way ==\n");
  std::printf("  %-14s %14s %14s %14s\n", "E/MeV", "e- ionis.", "proton Bragg", "p Bethe");
  int nonfinite = 0;
  for (int i = 0; i <= 8; ++i) {
    const real_t e = static_cast<real_t>(1e-3 * std::pow(10.0, i));
    const real_t cut = real_t(1e-3);
    const real_t el = em::collision_dedx(m, e, false, cut);
    const real_t bragg = em::bragg_dedx(m, ParticleType::kProton, e, cut);
    const real_t bb = em::bethe_bloch_dedx(m, ParticleType::kProton, e, cut);
    std::printf("  %-14.4g %14.6g %14.6g %14.6g\n", double(e), double(el), double(bragg),
                double(bb));
    if (!std::isfinite(double(el)) || !std::isfinite(double(bragg))
        || !std::isfinite(double(bb))) {
      ++nonfinite;
    }
  }
  // The density-effect parameters themselves, since they are where the infinity entered.
  const bool ster_finite = std::isfinite(double(m.c_density))
                           && std::isfinite(double(m.x0_density))
                           && std::isfinite(double(m.x1_density));
  std::printf("  Sternheimer: C %g, x0 %g, x1 %g\n", double(m.c_density),
              double(m.x0_density), double(m.x1_density));
  if (!ster_finite) {
    std::printf("  FAIL: the density-effect parameters are not finite - this is the log(0)\n");
    ++fails;
  }
  if (nonfinite != 0) {
    std::printf("  FAIL: %d energies gave a non-finite dE/dx. A run in this material would\n"
                "        score a NaN dose, which is exactly what it used to do.\n",
                nonfinite);
    ++fails;
  }

  std::printf("\n%s (%d failures)\n", fails ? "FAILED" : "PASSED", fails);
  return fails ? 1 : 0;
}
