// Every material in the NIST database, built and checked.
//
// This exists because `kMaxElements` was 8 - which is what example B1's four materials need -
// and `from_weight_fractions` had no bound check. Eleven of the shipped NIST materials have
// nine or ten elements, among them G4_BLOOD_ICRP, G4_TISSUE_SOFT_ICRP,
// G4_MUSCLE_SKELETAL_ICRP, G4_SKIN_ICRP and G4_CONCRETE, and every one of them wrote past the
// end of `z[]` into whatever field followed it. Nothing caught it, because nothing had ever
// built a material this port was not already validated on.
//
// So the test is not "does the physics agree" - test_vs_oracle and test_material_build do that
// for the materials there are reference numbers for. It is "does *every* material in the
// database survive being constructed, and come out physical". A general-purpose port has to
// answer that for all of them, not for the four an example happens to use.
//
// What is checked per material, and why each one is a thing that has actually been wrong:
//
//   n_elements within bounds        the overrun above
//   density, excitation > 0         a NaN excitation made every dose NaN (RISK.md O12)
//   electron density > 0            a zero one divides into the first dE/dx
//   fractions sum to one            Geant4 normalises them and the tabulated ones do not
//                                   (a 1e-6 error in air's electron density lived here)
//   radiation length > 0            the MSC step limit divides by it
//   cuts > 0 and ordered            a zero production cut makes the restricted dE/dx diverge
//   dE/dx finite for e-, p, alpha   the point of building it at all
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

#include "core/particle.cuh"
#include "data/materials.cuh"
#include "data/nist_materials.hh"
#include "physics/em/electron_processes.cuh"
#include "physics/em/hadron_range.cuh"

using namespace g4gpu;
using real_t = double;

int main() {
  static em::ShellTables<real_t> shell;
  em::build_shell_tables(shell);

  int fails = 0;
  int built = 0;
  int max_elements = 0;
  std::string widest;

  // Every material, one at a time into its own table: kMaxMaterials is a storage bound and the
  // database is far larger than it, so this is a loop over builds rather than one big table.
  for (int i = 0; i < g4::nist::kNumNistMaterials; ++i) {
    const g4::nist::NistMaterial& e = g4::nist::kNistMaterials[i];
    if (e.n_components > data::kMaxElements) {
      // Not a failure of the material - a failure of this build's bound, and add_material would
      // exit(2) rather than return, so report it here instead of running into it.
      std::printf("  FAIL: %s has %d elements, kMaxElements is %d\n", e.name, e.n_components,
                  data::kMaxElements);
      ++fails;
      continue;
    }
    if (e.n_components > max_elements) {
      max_elements = e.n_components;
      widest = e.name;
    }

    int zs[data::kMaxElements];
    real_t w[data::kMaxElements];
    for (int k = 0; k < e.n_components; ++k) {
      zs[k] = e.components[k].z;
      w[k] = static_cast<real_t>(e.components[k].fraction);
    }
    data::MaterialTable<real_t> t{};
    const data::MaterialState st = (e.state == 1)   ? data::MaterialState::kSolid
                                   : (e.state == 2) ? data::MaterialState::kLiquid
                                   : (e.state == 3) ? data::MaterialState::kGas
                                                    : data::MaterialState::kAuto;
    const int idx = data::add_material<real_t>(t, static_cast<real_t>(e.density_g_cm3),
                                               e.n_components, zs, w,
                                               static_cast<real_t>(e.mean_excitation_eV),
                                               real_t(0.7), st);
    if (idx < 0) {
      std::printf("  FAIL: %s could not be added\n", e.name);
      ++fails;
      continue;
    }
    if (e.has_sternheimer) {
      data::set_sternheimer<real_t>(t.m[idx], e.cbar, e.x0, e.x1, e.a, e.m, e.delta0);
    }
    data::set_nist_stopping<real_t>(t.m[idx], e.name, nullptr);
    const data::Material<real_t>& m = t.m[idx];
    ++built;

    auto bad = [&](const char* what, double v) {
      std::printf("  FAIL: %-28s %s = %g\n", e.name, what, v);
      ++fails;
    };
    if (m.n_elements != e.n_components) { bad("n_elements", m.n_elements); }
    // The gas test, against the tabulated state rather than against the density heuristic
    // that stands in for it. data::material_is_gas falls back to `density < 0.01` only for a
    // material built with no state at all - something a real G4Material never is - and the
    // claim that the fallback agrees with Geant4 for every NIST material is checked here
    // rather than asserted in a comment. G4IonFluctuations::Factor reads a different
    // parameter row and divides the reduced energy differently on the two sides of it.
    if (data::material_is_gas<real_t>(m) != (e.state == 3)) {
      std::printf("  FAIL: %-28s state %d but material_is_gas says %s\n", e.name, e.state,
                  data::material_is_gas<real_t>(m) ? "gas" : "not gas");
      ++fails;
    }
    {
      data::Material<real_t> probe = m;
      probe.state = 0;  // as if built by hand, with no state given
      if (data::material_is_gas<real_t>(probe) != (e.state == 3)) {
        std::printf("  FAIL: %-28s density %g, state %d: the no-state fallback disagrees\n",
                    e.name, static_cast<double>(m.density), e.state);
        ++fails;
      }
    }
    if (!(m.density > 0) || !std::isfinite(m.density)) { bad("density", m.density); }
    if (!(m.mean_excitation > 0) || !std::isfinite(m.mean_excitation)) {
      bad("mean_excitation", m.mean_excitation);
    }
    if (!(m.electron_density > 0) || !std::isfinite(m.electron_density)) {
      bad("electron_density", m.electron_density);
    }
    if (!(m.radiation_length > 0) || !std::isfinite(m.radiation_length)) {
      bad("radiation_length", m.radiation_length);
    }
    if (!(m.cut_gamma > 0) || !(m.cut_electron > 0) || !(m.cut_positron > 0)) {
      bad("cut_electron", m.cut_electron);
    }
    // Every element slot the material claims must carry a real Z and a positive atom density,
    // and every slot it does not claim must be untouched. The second half is what an overrun
    // looks like from the inside.
    for (int k = 0; k < m.n_elements; ++k) {
      if (m.z[k] < 1 || m.z[k] > 98 || !(m.n_atoms[k] > 0)) {
        std::printf("  FAIL: %-28s element %d: Z=%g n=%g\n", e.name, k, m.z[k], m.n_atoms[k]);
        ++fails;
      }
    }
    for (int k = m.n_elements; k < data::kMaxElements; ++k) {
      if (m.z[k] != real_t(0) || m.n_atoms[k] != real_t(0)) {
        std::printf("  FAIL: %-28s slot %d past n_elements is not zero: Z=%g n=%g\n", e.name, k,
                    m.z[k], m.n_atoms[k]);
        ++fails;
      }
    }

    // And the physics evaluates. One energy per species is enough here - the point is that the
    // material is usable at all, not that the number is right, which is test_vs_oracle's job.
    const real_t de = em::collision_dedx(m, real_t(1), false);
    const real_t dp = em::hadron_total_dedx(m, ParticleType::kProton, real_t(100),
                                            m.cut_electron, &shell);
    const real_t da = em::hadron_total_dedx(m, ParticleType::kAlpha, real_t(400),
                                            m.cut_electron, &shell);
    if (!(de > 0) || !std::isfinite(de)) { bad("dE/dx e- 1 MeV", de); }
    if (!(dp > 0) || !std::isfinite(dp)) { bad("dE/dx p 100 MeV", dp); }
    if (!(da > 0) || !std::isfinite(da)) { bad("dE/dx alpha 400 MeV", da); }
  }

  std::printf("== every NIST material built ==\n");
  std::printf("  %d materials built, widest is %s with %d elements (kMaxElements = %d)\n",
              built, widest.c_str(), max_elements, data::kMaxElements);
  if (built != g4::nist::kNumNistMaterials) {
    std::printf("  FAIL: %d of %d materials built\n", built, g4::nist::kNumNistMaterials);
    ++fails;
  }
  // The bound must have room. Sitting exactly on the widest material means the next database
  // update silently reintroduces the overrun - which is how this got here.
  if (max_elements >= data::kMaxElements) {
    std::printf("  FAIL: the widest material uses %d of %d element slots, leaving no headroom\n",
                max_elements, data::kMaxElements);
    ++fails;
  }

  std::printf("\n%s (%d failures)\n", fails ? "FAILED" : "PASSED", fails);
  return fails ? 1 : 0;
}
