// Photon total cross sections and process selection.
//
// Compton   : G4KleinNishinaCompton::ComputeCrossSectionPerAtom, closed-form Z-parameterization
// Pair      : G4PairProductionRelModel::ComputeParametrizedXSectionPerAtom - the Urban fit to
//             Hubbell's tabulated pair cross sections. That is what G4GammaConversion's default
//             model uses below 30 GeV, and Geant4's own comment beside it records that it is
//             "same as in G4BetheHeitlerModel". Above 30 GeV Geant4 stops using the fit and
//             numerically integrates the differential cross section with LPM suppression;
//             this does not, and keeps extrapolating the fit. Recorded in docs/PORTED.md.
// Photoelec.: G4LivermorePhotoElectricModel, EPICS2017 data. See data/photoelectric_data.cuh.
// Rayleigh  : G4LivermoreRayleighModel, EPICS2017 data, with G4RayleighAngularGenerator for the
//             deflection. See data/rayleigh_data.cuh. It transfers no energy, so it moves a dose
//             only by changing where the photon goes next - which is not nothing at 6 MeV in a
//             centimetre-scale phantom, and is why it is here rather than dropped.
#pragma once
#include <cmath>
#include "core/units.cuh"
#include "data/materials.cuh"
#include "data/photoelectric_data.cuh"
#include "data/rayleigh_data.cuh"

namespace g4gpu::em {

/// Photon tracking floor. Photoelectric absorption is now modelled explicitly, so this is
/// no longer standing in for it - it is only a floor below which a photon has a sub-micron
/// attenuation length in these materials and is deposited on the spot. Geant4 tracks gammas
/// down to 990 eV.
template <typename real_t> __host__ __device__ constexpr real_t kPhotonAbsorbCut() {
  return real_t(0.99e-3);  // 990 eV, Geant4 lowest gamma cut
}

enum class GammaProcess : int {
  kNone = 0, kCompton = 1, kPair = 2, kPhotoelectric = 3, kRayleigh = 4
};

// ---------------------------------------------------------------- Compton

/// Klein-Nishina total cross section per atom, mm^2. Transcribed from Geant4 11.5.0.
template <typename real_t>
__host__ __device__ inline real_t compton_xs_per_atom(real_t gamma_energy, real_t Z) {
  constexpr real_t kLowLimit = real_t(10e-6);  // 10 eV, G4VEmModel default
  if (gamma_energy <= kLowLimit) { return real_t(0); }

  const real_t barn = units::barn<real_t>();
  const real_t a = real_t(20.0), b = real_t(230.0), c = real_t(440.0);
  const real_t d1 = real_t(2.7965e-1) * barn, d2 = real_t(-1.8300e-1) * barn,
               d3 = real_t(6.7527) * barn,    d4 = real_t(-1.9798e+1) * barn,
               e1 = real_t(1.9756e-5) * barn, e2 = real_t(-1.0205e-2) * barn,
               e3 = real_t(-7.3913e-2) * barn, e4 = real_t(2.7079e-2) * barn,
               f1 = real_t(-3.9178e-7) * barn, f2 = real_t(6.8241e-5) * barn,
               f3 = real_t(6.0480e-5) * barn,  f4 = real_t(3.0274e-4) * barn;

  const real_t Z2 = Z * Z;
  const real_t p1Z = Z * (d1 + e1 * Z + f1 * Z2);
  const real_t p2Z = Z * (d2 + e2 * Z + f2 * Z2);
  const real_t p3Z = Z * (d3 + e3 * Z + f3 * Z2);
  const real_t p4Z = Z * (d4 + e4 * Z + f4 * Z2);

  real_t T0 = real_t(15.0e-3);              // 15 keV
  if (Z < real_t(1.5)) { T0 = real_t(40.0e-3); }  // 40 keV for hydrogen

  const real_t me = units::electron_mass_c2<real_t>();
  real_t X = fmax(gamma_energy, T0) / me;
  real_t xs = p1Z * log(real_t(1) + real_t(2) * X) / X
              + (p2Z + p3Z * X + p4Z * X * X)
                    / (real_t(1) + a * X + b * X * X + c * X * X * X);

  if (gamma_energy < T0) {
    const real_t dT0 = real_t(1.0e-3);  // 1 keV
    X = (T0 + dT0) / me;
    const real_t sigma = p1Z * log(real_t(1) + real_t(2) * X) / X
                         + (p2Z + p3Z * X + p4Z * X * X)
                               / (real_t(1) + a * X + b * X * X + c * X * X * X);
    const real_t c1 = -T0 * (sigma - xs) / (xs * dT0);
    real_t c2 = real_t(0.150);
    if (Z > real_t(1.5)) { c2 = real_t(0.375) - real_t(0.0556) * log(Z); }
    const real_t y = log(gamma_energy / T0);
    xs *= exp(-y * (c1 + c2 * y));
  }
  return fmax(xs, real_t(0));
}

// ---------------------------------------------------------------- Pair production

/// Pair production total cross section per atom, mm^2.
///
/// Transcribed from G4PairProductionRelModel::ComputeParametrizedXSectionPerAtom (11.1.1),
/// which is the path G4GammaConversion's default model takes below its 30 GeV
/// fParametrizedXSectionThreshold. The same coefficients appear in G4BetheHeitlerModel.
///
/// Valid, as Geant4 says of it, from 1.5 MeV to 100 GeV; below 1.5 MeV both this and Geant4
/// evaluate the fit at 1.5 MeV and scale by (E - 2mc^2)^2. Above 30 GeV Geant4 switches to a
/// numerical integration of the DCS and this does not - see the file header.
template <typename real_t>
__host__ __device__ inline real_t pair_xs_per_atom(real_t gamma_energy, real_t Z) {
  const real_t me = units::electron_mass_c2<real_t>();
  if (Z < real_t(0.9) || gamma_energy <= real_t(2) * me) { return real_t(0); }

  const real_t ub = real_t(1e-6) * units::barn<real_t>();  // microbarn
  const real_t a0 = real_t(8.7842e+2) * ub, a1 = real_t(-1.9625e+3) * ub,
               a2 = real_t(1.2949e+3) * ub, a3 = real_t(-2.0028e+2) * ub,
               a4 = real_t(1.2575e+1) * ub, a5 = real_t(-2.8333e-1) * ub;
  const real_t b0 = real_t(-1.0342e+1) * ub, b1 = real_t(1.7692e+1) * ub,
               b2 = real_t(-8.2381) * ub,    b3 = real_t(1.3063) * ub,
               b4 = real_t(-9.0815e-2) * ub, b5 = real_t(2.3586e-3) * ub;
  const real_t c0 = real_t(-4.5263e+2) * ub, c1 = real_t(1.1161e+3) * ub,
               c2 = real_t(-8.6749e+2) * ub, c3 = real_t(2.1773e+2) * ub,
               c4 = real_t(-2.0467e+1) * ub, c5 = real_t(6.5372e-1) * ub;

  constexpr real_t kLimit = real_t(1.5);  // MeV, validity floor of the fit
  const real_t e_org = gamma_energy;
  if (gamma_energy < kLimit) { gamma_energy = kLimit; }

  const real_t x = log(gamma_energy / me);
  const real_t x2 = x * x, x3 = x2 * x, x4 = x3 * x, x5 = x4 * x;
  const real_t F1 = a0 + a1 * x + a2 * x2 + a3 * x3 + a4 * x4 + a5 * x5;
  const real_t F2 = b0 + b1 * x + b2 * x2 + b3 * x3 + b4 * x4 + b5 * x5;
  const real_t F3 = c0 + c1 * x + c2 * x2 + c3 * x3 + c4 * x4 + c5 * x5;

  real_t xs = (Z + real_t(1)) * (F1 * Z + F2 * Z * Z + F3);
  if (e_org < kLimit) {
    const real_t dum = (e_org - real_t(2) * me) / (kLimit - real_t(2) * me);
    xs *= dum * dum;
  }
  return fmax(xs, real_t(0));
}

// ---------------------------------------------------------------- macroscopic

/// Per-process macroscopic cross sections, 1/mm, summed over the material elements.
template <typename real_t>
struct GammaXS {
  real_t compton;
  real_t pair;
  real_t photoelectric;
  real_t rayleigh;
  real_t total;
};

/// @param pe Livermore photoelectric table. Pass nullptr to omit the process.
template <typename real_t>
__host__ __device__ inline GammaXS<real_t> gamma_macroscopic_xs(
    const data::Material<real_t>& m, real_t gamma_energy,
    const data::PhotoElectricTable<real_t>* pe = nullptr,
    const data::RayleighTable<real_t>* ray = nullptr, bool want_compton = true,
    bool want_pair = true) {
  GammaXS<real_t> xs{real_t(0), real_t(0), real_t(0), real_t(0), real_t(0)};
  for (int i = 0; i < m.n_elements; ++i) {
    if (want_compton) { xs.compton += m.n_atoms[i] * compton_xs_per_atom(gamma_energy, m.z[i]); }
    if (want_pair) { xs.pair += m.n_atoms[i] * pair_xs_per_atom(gamma_energy, m.z[i]); }
    if (pe != nullptr) {
      const int z = static_cast<int>(m.z[i] + real_t(0.5));
      xs.photoelectric += m.n_atoms[i] * data::photoelectric_xs_per_atom(*pe, z, gamma_energy);
    }
    if (ray != nullptr) {
      const int z = static_cast<int>(m.z[i] + real_t(0.5));
      xs.rayleigh += m.n_atoms[i] * data::rayleigh_xs_per_atom(*ray, z, gamma_energy);
    }
  }
  xs.total = xs.compton + xs.pair + xs.photoelectric + xs.rayleigh;
  return xs;
}

/// Selects the interacting process from the per-process cross sections.
template <typename real_t>
__host__ __device__ inline GammaProcess select_gamma_process(const GammaXS<real_t>& xs,
                                                             real_t rand01) {
  if (xs.total <= real_t(0)) { return GammaProcess::kNone; }
  real_t r = rand01 * xs.total;
  if (r < xs.compton) { return GammaProcess::kCompton; }
  r -= xs.compton;
  if (r < xs.pair) { return GammaProcess::kPair; }
  r -= xs.pair;
  if (r < xs.photoelectric) { return GammaProcess::kPhotoelectric; }
  return GammaProcess::kRayleigh;
}

}  // namespace g4gpu::em
