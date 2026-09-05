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
