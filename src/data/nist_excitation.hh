// Per-element mean excitation energies, and Geant4's Bragg-additivity average over them.
//
// A material built from components with no tabulated mean excitation energy has to derive one.
// Geant4 does that in `G4IonisParamMat::ComputeMeanParameters`:
//
//     ln(I) = sum_i ( n_i Z_i ln(I_i) ) / sum_i ( n_i Z_i )
//     I     = exp( ln(I) )
//
// with `I_i` the element's own value and the denominator the material's electron density. It
// is Bragg additivity in the logarithm, which is the standard construction.
//
// Without it the device record carried `mean_excitation = 0`, and the density-effect
// parameters are built from `1 + 2 ln(I / plasma_energy)`. `log(0)` is -infinity, every
// dE/dx downstream became NaN, and a run with a user-built material scored a **NaN dose**.
// That is the bug this file exists to fix; `tests/test_material_build.cu` is what keeps it
// fixed, against Geant4's own derived value for a material built the same way.
//
// The `I_i` values are the ones Geant4 carries for its 98 elemental NIST materials - `G4_H` is
// 19.2 eV, `G4_C` 78 eV, `G4_O` 95 eV - so they are read out of the generated table rather
// than tabulated a second time here. This header is hand-written and the table it reads is
// regenerated from the oracle, which is why the lookup lives here and not in there.
#pragma once
#include <cmath>

#include "data/nist_materials.hh"

namespace g4gpu::data {

/// The mean excitation energy in eV of element @p z, or 0 if the table has no entry.
///
/// Found by looking for the single-component NIST material whose one component is this Z -
/// which is what an elemental material is - rather than by name, because the name would mean
/// carrying a Z-to-symbol table as well and the two could disagree.
inline double nist_element_excitation_eV(int z) {
  for (int i = 0; i < g4::nist::kNumNistMaterials; ++i) {
    const g4::nist::NistMaterial& m = g4::nist::kNistMaterials[i];
    if (m.n_components != 1 || m.components == nullptr) { continue; }
    if (m.components[0].z == z) { return m.mean_excitation_eV; }
  }
  return 0.0;
}

/// Geant4's derived mean excitation energy in eV, for a material given as mass fractions.
///
/// @param n   number of components
/// @param zs  atomic numbers
/// @param w   mass fractions, which need not be normalised
/// @param a_of_z  molar mass in g/mole for an atomic number
///
/// The weights Geant4 uses are `n_i Z_i`, the electrons per volume contributed by each
/// element. Working from mass fractions, `n_i` is proportional to `w_i / A_i` and the density
/// cancels out of the ratio, so no density is needed here.
///
/// Returns 0 when nothing usable was given - no components, or no element with a tabulated
/// value - and the caller must treat that as a material it cannot build rather than as a
/// number.
template <typename AofZ>
inline double derive_mean_excitation_eV(int n, const int* zs, const double* w, AofZ a_of_z) {
  double num = 0, den = 0;
  for (int i = 0; i < n; ++i) {
    const int z = zs[i];
    if (z < 1 || w[i] <= 0) { continue; }
    const double a = a_of_z(z);
    if (a <= 0) { continue; }
    const double i_elm = nist_element_excitation_eV(z);
    if (i_elm <= 0) { continue; }
    // n_i Z_i, up to the common factor rho*N_A that cancels in the ratio.
    const double nz = w[i] / a * static_cast<double>(z);
    num += nz * std::log(i_elm);
    den += nz;
  }
  if (den <= 0) { return 0.0; }
  return std::exp(num / den);
}

}  // namespace g4gpu::data
