// Gamma conversion, transcribed from G4PairProductionRelModel (11.1.1).
//
// G4GammaConversion selects this model by default (G4GammaConversion::InitialiseProcess
// constructs a G4PairProductionRelModel when none is set), over its whole energy range.
//
// Three regimes, all transcribed:
//   Eg < 2 MeV     - eps sampled uniformly on [eps0, 0.5], as Geant4 itself does
//   Eg < 50 MeV    - screened Bethe-Heitler rejection, low-energy delta_max and FZ
//   Eg > 50 MeV    - the Coulomb correction enters FZ and delta_max
//   Eg > 100 GeV   - LPM suppression on top (fIsUseLPMCorrection defaults to true)
//
// The pair directions come from G4ModifiedTsai::SamplePairDirections, which shares one
// azimuth between the two leptons and mirrors the transverse components, so the pair is
// coplanar with the photon.
#pragma once
#include <cmath>
#include "core/particle.cuh"
#include "core/units.cuh"
#include "core/vec3.cuh"
#include "data/brems_data.cuh"   // tsai_cos_theta
#include "data/materials.cuh"    // coulomb_factor
#include "physics/em/brems_rel.cuh"  // lpm_gs_phis, lpm_constant

namespace g4gpu::em {

template <typename real_t>
struct PairResult {
  Vec3<real_t> electron_dir;
  Vec3<real_t> positron_dir;
  real_t electron_ekin;
  real_t positron_ekin;
  bool ok;  ///< false below the 2*m_e threshold
};

/// Per-element constants, from G4PairProductionRelModel::InitialiseElementData.
/// Note this model uses log(184) where the bremsstrahlung model uses log(184.15).
template <typename real_t>
struct PairElement {
  real_t log_z13, coulomb;
  real_t delta_factor, delta_max_low, delta_max_high;
  real_t lpm_var_s1_cond, lpm_il_var_s1_cond;
};

template <typename real_t>
__host__ __device__ inline PairElement<real_t> pair_element(int z) {
  PairElement<real_t> e;
  const real_t zd = real_t(z);
  e.log_z13 = log(zd) / real_t(3);
  const real_t z13 = exp(e.log_z13);
  e.coulomb = data::coulomb_factor<real_t>(zd);
  const real_t fz_low = real_t(8) * e.log_z13;
  const real_t fz_high = real_t(8) * (e.log_z13 + e.coulomb);
  e.delta_factor = real_t(136) / z13;
  e.delta_max_low = exp((real_t(42.038) - fz_low) / real_t(8.29)) - real_t(0.958);
  e.delta_max_high = exp((real_t(42.038) - fz_high) / real_t(8.29)) - real_t(0.958);
  e.lpm_var_s1_cond = sqrt(real_t(2)) * z13 * z13 / (real_t(184) * real_t(184));
  e.lpm_il_var_s1_cond = real_t(1) / log(e.lpm_var_s1_cond);
  return e;
}

/// Screening functions, from G4PairProductionRelModel::ScreenFunction1/2/12.
template <typename real_t>
__host__ __device__ inline void screen_function12(real_t delta, real_t& f1, real_t& f2) {
  if (delta > real_t(1.4)) {
    f1 = real_t(42.038) - real_t(8.29) * log(delta + real_t(0.958));
    f2 = f1;
  } else {
    f1 = real_t(42.184) - delta * (real_t(7.444) - real_t(1.623) * delta);
    f2 = real_t(41.326) - delta * (real_t(5.848) - real_t(0.902) * delta);
  }
}
template <typename real_t>
__host__ __device__ inline real_t screen_function1(real_t delta) {
  return (delta > real_t(1.4)) ? real_t(42.038) - real_t(8.29) * log(delta + real_t(0.958))
                               : real_t(42.184) - delta * (real_t(7.444) - real_t(1.623) * delta);
}
template <typename real_t>
__host__ __device__ inline real_t screen_function2(real_t delta) {
  return (delta > real_t(1.4)) ? real_t(42.038) - real_t(8.29) * log(delta + real_t(0.958))
                               : real_t(41.326) - delta * (real_t(5.848) - real_t(0.902) * delta);
}

/// phi1, phi2 for the LPM-corrected rejection.
/// Verbatim from G4PairProductionRelModel::ComputePhi12.
template <typename real_t>
__host__ __device__ inline void pair_phi12(real_t delta, real_t& phi1, real_t& phi2) {
  if (delta > real_t(1.4)) {
    phi1 = real_t(21.0190) - real_t(4.145) * log(delta + real_t(0.958));
    phi2 = phi1;
  } else {
    phi1 = real_t(20.806) - delta * (real_t(3.190) - real_t(0.5710) * delta);
    phi2 = real_t(20.234) - delta * (real_t(2.126) - real_t(0.0903) * delta);
  }
}

/// xi(s), G(s), phi(s) for pair production.
/// Verbatim from G4PairProductionRelModel::ComputeLPMfunctions. Note this differs from the
/// bremsstrahlung version: s' has no density correction and is built from eps rather than
/// the photon-to-primary energy ratio.
template <typename real_t>
__host__ __device__ inline void pair_lpm_functions(real_t& xi_s, real_t& gs, real_t& phis,
                                                   const PairElement<real_t>& el, real_t eps,
                                                   real_t egamma, real_t lpm_energy) {
  const real_t s_prime =
      sqrt(real_t(0.125) * lpm_energy / (eps * egamma * (real_t(1) - eps)));
  const real_t condition = el.lpm_var_s1_cond;
  xi_s = real_t(2);
  if (s_prime > real_t(1)) {
    xi_s = real_t(1);
  } else if (s_prime > condition) {
    const real_t d = el.lpm_il_var_s1_cond;
    const real_t h = log(s_prime) * d;
    xi_s = real_t(1) + h - real_t(0.08) * (real_t(1) - h) * h * (real_t(2) - h) * d;
  }
  const real_t s_hat = s_prime / sqrt(xi_s);
  lpm_gs_phis(gs, phis, s_hat);  // shared with the bremsstrahlung model
  if (xi_s * phis > real_t(1) || s_hat > real_t(0.57)) { xi_s = real_t(1) / phis; }
}

/// Photon energy above which the Coulomb correction is applied
/// (G4PairProductionRelModel::fCoulombCorrectionThreshold).
template <typename real_t> __host__ __device__ constexpr real_t kCoulombThreshold() {
  return real_t(50.0);  // MeV
}
/// Photon energy above which LPM suppression is applied (gEgLPMActivation).
template <typename real_t> __host__ __device__ constexpr real_t kLpmActivation() {
  return real_t(1e5);  // MeV = 100 GeV
}

/// P26: `G4PairProductionRelModel::ComputeLPMGsPhis` - THE PAIR MODEL'S OWN, which is not
/// `lpm_gs_phis` above: that one is `G4eBremsstrahlungRelModel`'s, whose coefficients are rounded
/// (0.415827 for 0.415827397755, -0.160723 for -0.16072300849123999, 0.011905 for 0.01190476,
/// 0.023065 for 0.0230655). The pair model's, verbatim.
template <typename real_t>
__host__ __device__ inline void pair_lpm_gs_phis_g4(real_t& gs, real_t& phis, real_t s) {
  constexpr real_t pi = real_t(3.14159265358979323846);
  if (s < real_t(0.01)) {
    phis = real_t(6.0) * s * (real_t(1.0) - pi * s);
    gs = real_t(12.0) * s - real_t(2.0) * phis;
    return;
  }
  const real_t s2 = s * s, s3 = s * s2, s4 = s2 * s2;
  if (s < real_t(0.415827397755)) {
    phis = real_t(1.0) - exp(real_t(-6.0) * s * (real_t(1.0) + s * (real_t(3.0) - pi))
                             + s3 / (real_t(0.623) + real_t(0.796) * s + real_t(0.658) * s2));
    const real_t psis =
        real_t(1.0) - exp(real_t(-4.0) * s
                          - real_t(8.0) * s2
                                / (real_t(1.0) + real_t(3.936) * s + real_t(4.97) * s2
                                   - real_t(0.05) * s3 + real_t(7.5) * s4));
    gs = real_t(3.0) * psis - real_t(2.0) * phis;
  } else if (s < real_t(1.55)) {
    phis = real_t(1.0) - exp(real_t(-6.0) * s * (real_t(1.0) + s * (real_t(3.0) - pi))
                             + s3 / (real_t(0.623) + real_t(0.796) * s + real_t(0.658) * s2));
    const real_t dum0 = real_t(-0.16072300849123999) + real_t(3.7550300067531581) * s
                        - real_t(1.7981383069010097) * s2 + real_t(0.67282686077812381) * s3
                        - real_t(0.1207722909879257) * s4;
    gs = tanh(dum0);
  } else {
    phis = real_t(1.0) - real_t(0.01190476) / s4;
    if (s < real_t(1.9156)) {
      const real_t dum0 = real_t(-0.16072300849123999) + real_t(3.7550300067531581) * s
                          - real_t(1.7981383069010097) * s2 + real_t(0.67282686077812381) * s3
                          - real_t(0.1207722909879257) * s4;
      gs = tanh(dum0);
    } else {
      gs = real_t(1.0) - real_t(0.0230655) / s4;
    }
  }
}

/// P26: `G4PairProductionRelModel::GetLPMFunctions` - NOT the functions themselves below s = 2
/// but `InitLPMFunctions`' table of them at s = i/fISDelta (fISDelta 100, fSLimit 2, 201
/// entries), interpolated linearly; above, the asymptotic forms. The two table entries are
/// computed here as `InitLPMFunctions` computes them, so they are the table's doubles.
template <typename real_t>
__host__ __device__ inline void pair_get_lpm_functions(real_t& lpm_gs, real_t& lpm_phis,
                                                       real_t sval) {
  constexpr real_t kISDelta = real_t(100.);
  constexpr real_t kSLimit = real_t(2.);
  if (sval < kSLimit) {
    real_t val = sval * kISDelta;
    const int ilow = static_cast<int>(val);
    val -= ilow;
    real_t g0, p0, g1, p1;
    pair_lpm_gs_phis_g4<real_t>(g0, p0, ilow / kISDelta);
    pair_lpm_gs_phis_g4<real_t>(g1, p1, (ilow + 1) / kISDelta);
    lpm_gs = (g1 - g0) * val + g0;
    lpm_phis = (p1 - p0) * val + p0;
  } else {
    real_t ss = sval * sval;
    ss *= ss;
    lpm_phis = real_t(1.0) - real_t(0.01190476) / ss;
    lpm_gs = real_t(1.0) - real_t(0.0230655) / ss;
  }
}

/// P26: `G4PairProductionRelModel::ComputeLPMfunctions` as the RATE uses it - the same as
/// `pair_lpm_functions` but with the pair model's own G(s) and phi(s) (`pair_get_lpm_functions`).
/// `sample_pair_production` still calls the bremsstrahlung model's; docs/PORTED.md 4.1 records it.
template <typename real_t>
__host__ __device__ inline void pair_lpm_functions_g4(real_t& xi_s, real_t& gs, real_t& phis,
                                                      const PairElement<real_t>& el, real_t eps,
                                                      real_t egamma, real_t lpm_energy) {
  const real_t s_prime = sqrt(real_t(0.125) * lpm_energy / (eps * egamma * (real_t(1.0) - eps)));
  const real_t condition = el.lpm_var_s1_cond;
  xi_s = real_t(2.0);
  if (s_prime > real_t(1.0)) {
    xi_s = real_t(1.0);
  } else if (s_prime > condition) {
    const real_t dum = el.lpm_il_var_s1_cond;
    const real_t h = log(s_prime) * dum;
    xi_s = real_t(1.0) + h - real_t(0.08) * (real_t(1.0) - h) * h * (real_t(2.0) - h) * dum;
  }
  const real_t s_hat = s_prime / sqrt(xi_s);
  pair_get_lpm_functions<real_t>(gs, phis, s_hat);
  if (xi_s * phis > real_t(1.) || s_hat > real_t(0.57)) { xi_s = real_t(1.) / phis; }
}

/// P26: `G4PairProductionRelModel::ComputeCrossSectionPerAtom` over its WHOLE range, mm^2 -
/// `pair_xs_per_atom`'s parameterisation below `fParametrizedXSectionThreshold` (30 GeV), and from
/// it up `ComputeXSectionPerAtom`: the differential cross section integrated over the energy
/// transfer with two sub-intervals of 8-point Gauss-Legendre, LPM-suppressed above
/// `gEgLPMActivation` (100 GeV), times `gXSecFactor*Z*(Z + eta)`. Non-complete screening, the
/// default (`fIsUseCompleteScreening` false).
///
/// Written for the general process's tables, whose zone-3 grid and conversion's own lambda vector
/// run to 100 TeV: built from the parameterisation they were off Geant4's by up to a factor of
/// three at the top (docs/PORTED.md 4.1), and - through conversion's spline - by 3.5e-3 at
/// 26.8 GeV, below the threshold. `step_gamma`'s per-element target choice still calls
/// `pair_xs_per_atom` (docs/PORTED.md 4.1 says so); this is the rate.
///
/// @param lpm_energy `fLPMEnergy` = `mat->GetRadlen()*gLPMconstant`, set by `SetupForMaterial`,
///        which `G4VEmModel::CrossSectionPerVolume` calls before the element loop.
template <typename real_t>
__host__ __device__ inline real_t pair_xs_per_atom_rel(real_t gamma_energy, real_t Z,
                                                       real_t lpm_energy) {
  const real_t me = units::electron_mass_c2<real_t>();
  if (gamma_energy <= real_t(2) * me) { return real_t(0); }
  constexpr real_t kParametrizedXSectionThreshold = real_t(30000.0);  // 30 GeV
  if (gamma_energy < kParametrizedXSectionThreshold) {
    return pair_xs_per_atom<real_t>(gamma_energy, Z);
  }
  // gMaxZet = 120; `iz = std::min(gMaxZet, G4lrint(Z))`.
  int iz = static_cast<int>(lrint(Z));
  if (iz > 120) { iz = 120; }
  // `InitialiseElementData` reads Z^(1/3) from `G4IonisParamElm::GetZ3()`, which is
  // `G4Pow::Z13(Z)` = `std::pow(Z, 1./3.)`; `pair_element` - the sampler's - takes
  // `exp(log(Z)/3)`, which can be an ulp away. The rate takes Geant4's.
  PairElement<real_t> el = pair_element<real_t>(iz);
  {
    const real_t z13 = pow(real_t(iz), real_t(1.0) / real_t(3.0));
    el.delta_factor = real_t(136.) / z13;
    el.lpm_var_s1_cond = sqrt(real_t(2.)) * z13 * z13 / (real_t(184.) * real_t(184.));
    el.lpm_il_var_s1_cond = real_t(1.) / log(el.lpm_var_s1_cond);
  }
  // `InitialiseElementData`: the elastic and inelastic radiation logarithms, Dirac-Fock below
  // Z = 5 and Thomas-Fermi-Moliere above, and `fEtaValue = Finel/(Fel - fc)`.
  constexpr real_t kFelLowZet[5] = {real_t(0.0), real_t(5.3104), real_t(4.7935), real_t(4.7402),
                                    real_t(4.7112)};
  constexpr real_t kFinelLowZet[5] = {real_t(0.0), real_t(5.9173), real_t(5.6125),
                                      real_t(5.5377), real_t(5.4728)};
  real_t fel, finel;
  if (iz < 5) {
    fel = kFelLowZet[iz];
    finel = kFinelLowZet[iz];
  } else {
    fel = log(real_t(184.)) - el.log_z13;
    finel = log(real_t(1194.)) - real_t(2.) * el.log_z13;
  }
  const real_t eta = finel / (fel - el.coulomb);
  // `ComputeXSectionPerAtom`.
  constexpr real_t kXGL[8] = {real_t(1.98550718e-02), real_t(1.01666761e-01),
                              real_t(2.37233795e-01), real_t(4.08282679e-01),
                              real_t(5.91717321e-01), real_t(7.62766205e-01),
                              real_t(8.98333239e-01), real_t(9.80144928e-01)};
  constexpr real_t kWGL[8] = {real_t(5.06142681e-02), real_t(1.11190517e-01),
                              real_t(1.56853323e-01), real_t(1.81341892e-01),
                              real_t(1.81341892e-01), real_t(1.56853323e-01),
                              real_t(1.11190517e-01), real_t(5.06142681e-02)};
  const bool is_lpm = gamma_energy > kLpmActivation<real_t>();
  const real_t eps0 = me / gamma_energy;
  const real_t dmax = el.delta_max_high;
  const real_t dmin = real_t(4.) * eps0 * el.delta_factor;
  const real_t eps1 = real_t(0.5) - real_t(0.5) * sqrt(real_t(1.) - dmin / dmax);
  const real_t eps_min = (eps0 > eps1) ? eps0 : eps1;
  const real_t eps_max = real_t(0.5);
  constexpr int kNumSub = 2;
  const real_t d_interv = (eps_max - eps_min) * gamma_energy / real_t(kNumSub);
  real_t min_eti = eps_min * gamma_energy;
  real_t xsec = real_t(0);
  for (int i = 0; i < kNumSub; ++i) {
    for (int ngl = 0; ngl < 8; ++ngl) {
      const real_t et = (min_eti + kXGL[ngl] * d_interv);
      // `ComputeDXSectionPerAtom` / `ComputeRelDXSectionPerAtom`, normal screening.
      const real_t eps = et / gamma_energy;
      const real_t epsm = real_t(1.) - eps;
      const real_t dum = eps * epsm;
      const real_t fc = el.coulomb;
      const real_t ln_z13 = el.log_z13;
      const real_t delta = el.delta_factor * eps0 / dum;
      real_t phi1, phi2;
      pair_phi12<real_t>(delta, phi1, phi2);
      real_t xs;
      if (is_lpm) {
        real_t xi_s, gs, phis;
        pair_lpm_functions_g4<real_t>(xi_s, gs, phis, el, eps, gamma_energy, lpm_energy);
        xs = (eps * eps + epsm * epsm) * (real_t(2.) * phis + gs)
                 * (real_t(0.25) * phi1 - ln_z13 - fc) / real_t(3.)
             + real_t(2.) * dum * gs * (real_t(0.25) * phi2 - ln_z13 - fc) / real_t(3.);
        xs = fmax(xi_s * xs, real_t(0.0)) / gamma_energy;
      } else {
        xs = (eps * eps + epsm * epsm) * (real_t(0.25) * phi1 - ln_z13 - fc)
             + real_t(2.) * dum * (real_t(0.25) * phi2 - ln_z13 - fc) / real_t(3.);
        xs = fmax(xs, real_t(0.0)) / gamma_energy;
      }
      xsec += kWGL[ngl] * xs;
    }
    min_eti += d_interv;
  }
  xsec = fmax(real_t(2.) * xsec * d_interv, real_t(0.));
  // `gXSecFactor = 4*fine_structure_const*classic_electr_radius*classic_electr_radius`.
  const real_t re = units::classic_electron_radius<real_t>();
  const real_t x_sec_factor = real_t(4.) * units::fine_structure_const<real_t>() * re * re;
  xsec *= x_sec_factor * Z * (Z + eta);
  return fmax(xsec, real_t(0.));
}

/// Converts a photon into an e-/e+ pair. The photon is always consumed on success.
/// Transcribed from G4PairProductionRelModel::SampleSecondaries.
///
/// @param z atomic number of the target atom
/// @param radiation_length material radiation length, mm (only used above 100 GeV)
template <typename real_t, typename Rng>
__host__ __device__ inline PairResult<real_t> sample_pair_production(
    real_t gamma_energy, const Vec3<real_t>& gamma_dir, int z, Rng& rng,
    real_t radiation_length = real_t(0)) {
  PairResult<real_t> out{gamma_dir, gamma_dir, real_t(0), real_t(0), false};
  const real_t me = units::electron_mass_c2<real_t>();
  const real_t eps0 = me / gamma_energy;
  if (eps0 > real_t(0.5)) { return out; }

  real_t eps;
  constexpr real_t kEgSmall = real_t(2.0);  // MeV
  if (gamma_energy < kEgSmall) {
    eps = eps0 + (real_t(0.5) - eps0) * rng.uniform();
  } else {
    const PairElement<real_t> el = pair_element<real_t>(z);
    const real_t delta_factor = el.delta_factor * eps0;
    const real_t delta_min = real_t(4) * delta_factor;
    real_t delta_max = el.delta_max_low;
    real_t FZ = real_t(8) * el.log_z13;
    if (gamma_energy > kCoulombThreshold<real_t>()) {
      FZ += real_t(8) * el.coulomb;
      delta_max = el.delta_max_high;
    }

    const real_t epsp =
        real_t(0.5) - real_t(0.5) * sqrt(fmax(real_t(0), real_t(1) - delta_min / delta_max));
    const real_t eps_min = fmax(eps0, epsp);
    const real_t eps_range = real_t(0.5) - eps_min;

    real_t F10, F20;
    screen_function12(delta_min, F10, F20);
    F10 -= FZ;
    F20 -= FZ;
    const real_t normF1 = fmax(F10 * eps_range * eps_range, real_t(0));
    const real_t normF2 = fmax(real_t(1.5) * F20, real_t(0));
    const real_t norm_cond = (normF1 + normF2 > real_t(0)) ? normF1 / (normF1 + normF2)
                                                           : real_t(0);
    const bool is_lpm =
        (gamma_energy > kLpmActivation<real_t>()) && (radiation_length > real_t(0));
    const real_t lpm_energy = radiation_length * lpm_constant<real_t>();

    eps = eps_min;
    for (int guard = 0; guard < 1000; ++guard) {
      const real_t r0 = rng.uniform(), r1 = rng.uniform(), r2 = rng.uniform();
      real_t greject;
      if (norm_cond > r0) {
        eps = real_t(0.5) - eps_range * cbrt(r1);  // G4Pow::A13 is the cube root
        const real_t delta = delta_factor / (eps * (real_t(1) - eps));
        if (is_lpm) {
          real_t xi_s, gs, phis, phi1, phi2;
          pair_phi12(delta, phi1, phi2);
          pair_lpm_functions(xi_s, gs, phis, el, eps, gamma_energy, lpm_energy);
          greject = (F10 != real_t(0))
                        ? xi_s * ((real_t(2) * phis + gs) * phi1 - gs * phi2 - phis * FZ) / F10
                        : real_t(0);
        } else {
          greject = (F10 != real_t(0)) ? (screen_function1(delta) - FZ) / F10 : real_t(0);
        }
      } else {
        eps = eps_min + eps_range * r1;
        const real_t delta = delta_factor / (eps * (real_t(1) - eps));
        if (is_lpm) {
          real_t xi_s, gs, phis, phi1, phi2;
          pair_phi12(delta, phi1, phi2);
          pair_lpm_functions(xi_s, gs, phis, el, eps, gamma_energy, lpm_energy);
          greject = (F20 != real_t(0))
                        ? xi_s
                              * ((phis + real_t(0.5) * gs) * phi1 + real_t(0.5) * gs * phi2
                                 - real_t(0.5) * (gs + phis) * FZ)
                              / F20
                        : real_t(0);
        } else {
          greject = (F20 != real_t(0)) ? (screen_function2(delta) - FZ) / F20 : real_t(0);
        }
      }
      if (greject >= r2) { break; }
    }
  }

  // Which member of the pair gets eps is decided by a coin flip, as Geant4 does.
  real_t e_total, p_total;
  if (rng.uniform() > real_t(0.5)) {
    e_total = (real_t(1) - eps) * gamma_energy;
    p_total = eps * gamma_energy;
  } else {
    p_total = (real_t(1) - eps) * gamma_energy;
    e_total = eps * gamma_energy;
  }
  out.electron_ekin = fmax(real_t(0), e_total - me);
  out.positron_ekin = fmax(real_t(0), p_total - me);

  // G4ModifiedTsai::SamplePairDirections: one azimuth shared by both leptons, with the
  // positron taking the mirrored transverse components so the pair stays coplanar.
  const real_t phi = units::twopi<real_t>() * rng.uniform();
  const real_t sinp = sin(phi), cosp = cos(phi);
  real_t ct = data::tsai_cos_theta(out.electron_ekin, rng);
  real_t st = sqrt(fmax(real_t(0), (real_t(1) - ct) * (real_t(1) + ct)));
  out.electron_dir =
      normalize(rotate_uz(Vec3<real_t>{st * cosp, st * sinp, ct}, gamma_dir));
  ct = data::tsai_cos_theta(out.positron_ekin, rng);
  st = sqrt(fmax(real_t(0), (real_t(1) - ct) * (real_t(1) + ct)));
  out.positron_dir =
      normalize(rotate_uz(Vec3<real_t>{-st * cosp, -st * sinp, ct}, gamma_dir));
  out.ok = true;
  return out;
}

}  // namespace g4gpu::em
