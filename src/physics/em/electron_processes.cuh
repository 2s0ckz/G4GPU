// Electron and positron transport by condensed history.
//
// dE/dx     : Berger-Seltzer, transcribed from G4MollerBhabhaModel::ComputeDEDXPerVolume.
//             RESTRICTED to the material production cut - transfers above it produce an
//             explicit delta ray instead of depositing locally.
// Delta rays: G4MollerBhabhaModel cross section and sampler, transcribed verbatim.
// Range     : integrated on the host from the restricted stopping power, matching what
//             Geant4 builds its range table from.
// MSC       : Highland approximation, NOT Geant4 Urban MSC. See docs/RISK.md.
// Brems     : treated as an addition to the continuous loss via a radiative-yield
//             approximation; no explicit bremsstrahlung photons are generated.
// Annihil.  : positrons at rest emit two back-to-back 511 keV photons.
#pragma once
#include <cmath>
#include "core/particle.cuh"
#include "core/rng.cuh"
#include "core/units.cuh"
#include "core/vec3.cuh"
#include "data/brems_data.cuh"
#include "data/materials.cuh"

namespace g4gpu::em {

/// Electron rest mass, re-exported so the stepper need not include the units header.
template <typename real_t> __host__ __device__ constexpr real_t units_me() {
  return units::electron_mass_c2<real_t>();
}

/// Electrons below this deposit their remaining energy on the spot and stop.
template <typename real_t> __host__ __device__ constexpr real_t kElectronTrackingCut() {
  return real_t(1e-3);  // 1 keV
}

/// 2*pi*m_e*c^2*r_e^2 in MeV*mm^2, the Berger-Seltzer and Bethe-Bloch ionisation prefactor.
///
/// `units::twopi_mc2_rcl2` and not the product of the three factors. CLHEP derives this
/// quantity itself and pins it, and recomputing it from `twopi * m_e * r_e * r_e` gives a
/// different last bit.
///
/// This used to be defined *twice* - here as `constexpr` and again in hadron_ionisation.cuh as
/// `inline`, each computing the product its own way. Two definitions of the same symbol in the
/// same namespace is an ODR violation that only failed to compile because no translation unit
/// had ever included both headers; a test that needed an electron dE/dx and a hadron dE/dx in
/// one file found it immediately. Worse than the violation is what it invited: two prefactors
/// that could drift apart, with electron and hadron ionisation silently using different ones.
template <typename real_t> __host__ __device__ inline real_t twopi_mc2_rcl2() {
  return units::twopi_mc2_rcl2<real_t>();
}

/// Largest energy transferable to a delta ray, MeV. Transcribed from
/// G4MollerBhabhaModel::MaxSecondaryEnergy: half the kinetic energy for Moller (identical
/// particles), all of it for Bhabha.
template <typename real_t>
__host__ __device__ inline real_t max_secondary_energy(real_t kinetic, bool is_positron) {
  return is_positron ? kinetic : real_t(0.5) * kinetic;
}

/// Restricted collision stopping power, MeV/mm. @p is_positron selects the Bhabha branch.
///
/// "Restricted" means only energy transfers below @p cut are continuous; above it Geant4
/// emits an explicit delta ray instead, so that energy leaves the local deposit. Pass a
/// negative @p cut to use the material's own production threshold, or a huge value for the
/// unrestricted (total) stopping power.
template <typename real_t>
__host__ __device__ inline real_t collision_dedx(const data::Material<real_t>& m, real_t kinetic,
                                                 bool is_positron, real_t cut = real_t(-1)) {
  const real_t me = units::electron_mass_c2<real_t>();
  // Geant4 clamps below 0.25*sqrt(Zeff) keV and extrapolates underneath.
  const real_t th = real_t(0.25) * sqrt(m.z_eff) * real_t(1e-3);
  const real_t tkin = fmax(kinetic, th);

  const real_t tau = tkin / me;
  const real_t gam = tau + real_t(1);
  const real_t gamma2 = gam * gam;
  const real_t bg2 = tau * (tau + real_t(2));
  const real_t beta2 = bg2 / gamma2;

  const real_t eexc = m.mean_excitation / me;  // mean excitation energy, MeV -> units of m_e
  const real_t eexc2 = eexc * eexc;

  // G4MollerBhabhaModel: d = min(cut, MaxSecondaryEnergy) / m_e.
  const real_t tmax = max_secondary_energy(tkin, is_positron);
  const real_t use_cut = (cut < real_t(0)) ? m.cut_electron : cut;
  const real_t d = fmin(use_cut, tmax) / me;

  real_t dedx;
  if (!is_positron) {
    dedx = log(real_t(2) * (tau + real_t(2)) / eexc2) - real_t(1) - beta2 + log((tau - d) * d)
           + tau / (tau - d)
           + (real_t(0.5) * d * d + (real_t(2) * tau + real_t(1)) * log1p(-d / tau)) / gamma2;
  } else {
    const real_t d2 = d * d * real_t(0.5);
    const real_t d3 = d2 * d / real_t(1.5);
    const real_t d4 = d3 * d * real_t(0.75);
    const real_t y = real_t(1) / (real_t(1) + gam);
    dedx = log(real_t(2) * (tau + real_t(2)) / eexc2) + log(tau * d)
           - beta2
                 * (tau + real_t(2) * d
                    - y * (real_t(3) * d2 + y * (d - d3 + y * (d2 - tau * d3 + d4))))
                 / tau;
  }

  // Sternheimer density-effect correction, x = log10(beta*gamma) = log(bg2)/(2 ln 10).
  dedx -= data::density_correction(m, log(bg2) / data::twoln10<real_t>());

  dedx *= twopi_mc2_rcl2<real_t>() * m.electron_density / beta2;
  if (dedx < real_t(0)) { dedx = real_t(0); }

  // Geant4 low-energy extrapolation below the threshold.
  if (kinetic < th) {
    const real_t x = kinetic / th;
    dedx *= (x > real_t(0.25)) ? real_t(1) / sqrt(x) : real_t(1) / sqrt(real_t(0.25));
  }
  return dedx;
}

/// Radiative stopping power, MeV/mm, from an approximate radiation-yield scaling.
/// Crude next to G4SeltzerBergerModel; at a few MeV in these materials the radiative
/// fraction is only a few percent. See docs/RISK.md entry E2.
template <typename real_t>
__host__ __device__ inline real_t radiative_dedx(const data::Material<real_t>& m, real_t kinetic) {
  // (dE/dx)_rad / (dE/dx)_col ~ E * Z_eff / 800 MeV  (Evans, order-of-magnitude form)
  const real_t ratio = kinetic * m.z_eff / real_t(800);
  return collision_dedx(m, kinetic, false) * ratio;
}

/// Restricted total stopping power: restricted collision loss plus radiative loss. This is
/// what Geant4 integrates to build its range table, which is why an unrestricted integral
/// came out 4-8% short.
template <typename real_t>
__host__ __device__ inline real_t total_dedx(const data::Material<real_t>& m, real_t kinetic,
                                             bool is_positron) {
  return collision_dedx(m, kinetic, is_positron) + radiative_dedx(m, kinetic);
}

// ---------------------------------------------------------------- delta rays

/// Macroscopic cross section for producing a delta ray above @p cut, 1/mm.
/// Transcribed from G4MollerBhabhaModel::ComputeCrossSectionPerElectron, scaled by the
/// material electron density as G4VEmModel::CrossSectionPerVolume does.
template <typename real_t>
__host__ __device__ inline real_t delta_ray_xs(const data::Material<real_t>& m, real_t kinetic,
                                               bool is_positron, real_t cut = real_t(-1)) {
  const real_t me = units::electron_mass_c2<real_t>();
  const real_t cut_energy = (cut < real_t(0)) ? m.cut_electron : cut;
  const real_t tmax = max_secondary_energy(kinetic, is_positron);
  if (cut_energy >= tmax || kinetic <= real_t(0)) { return real_t(0); }

  const real_t xmin = cut_energy / kinetic;
  const real_t xmax = tmax / kinetic;
  const real_t tau = kinetic / me;
  const real_t gam = tau + real_t(1);
  const real_t gamma2 = gam * gam;
  const real_t beta2 = tau * (tau + real_t(2)) / gamma2;

  real_t cross;
  if (!is_positron) {  // Moller
    const real_t gg = (real_t(2) * gam - real_t(1)) / gamma2;
    cross = ((xmax - xmin) * (real_t(1) - gg + real_t(1) / (xmin * xmax)
                              + real_t(1) / ((real_t(1) - xmin) * (real_t(1) - xmax)))
             - gg * log(xmax * (real_t(1) - xmin) / (xmin * (real_t(1) - xmax))))
            / beta2;
  } else {  // Bhabha
    const real_t y = real_t(1) / (real_t(1) + gam);
    const real_t y2 = y * y;
    const real_t y12 = real_t(1) - real_t(2) * y;
    const real_t b1 = real_t(2) - y2;
    const real_t b2 = y12 * (real_t(3) + y2);
    const real_t y122 = y12 * y12;
    const real_t b4 = y122 * y12;
    const real_t b3 = b4 + y122;
    cross = (xmax - xmin) * (real_t(1) / (beta2 * xmin * xmax) + b2
                             - real_t(0.5) * b3 * (xmin + xmax)
                             + b4 * (xmin * xmin + xmin * xmax + xmax * xmax) / real_t(3))
            - b1 * log(xmax / xmin);
  }
  // G4: cross *= twopi_mc2_rcl2/kineticEnergy. Our constant is the same CLHEP quantity.
  cross *= twopi_mc2_rcl2<real_t>() / kinetic;
  return fmax(cross, real_t(0)) * m.electron_density;
}

template <typename real_t>
struct DeltaRay {
  Vec3<real_t> delta_dir;
  real_t delta_ekin;
  Vec3<real_t> primary_dir;  ///< recoil direction of the primary after the collision
  real_t primary_ekin;
  bool produced;
};

/// Samples one delta ray. Transcribed from G4MollerBhabhaModel::SampleSecondaries,
/// including the rejection majorants and the exact recoil kinematics (the primary
/// direction comes from momentum conservation, not from an angle formula).
template <typename real_t, typename Rng>
__host__ __device__ inline DeltaRay<real_t> sample_delta_ray(
    const data::Material<real_t>& m, real_t kinetic, const Vec3<real_t>& dir, bool is_positron,
    Rng& rng, real_t cut = real_t(-1)) {
  DeltaRay<real_t> out{dir, real_t(0), dir, kinetic, false};
  const real_t me = units::electron_mass_c2<real_t>();
  const real_t tmin = (cut < real_t(0)) ? m.cut_electron : cut;
  const real_t tmax = max_secondary_energy(kinetic, is_positron);
  if (tmin >= tmax) { return out; }

  const real_t energy = kinetic + me;          // total energy
  const real_t xmin = tmin / kinetic;
  const real_t xmax = tmax / kinetic;
  const real_t gam = energy / me;
  const real_t gamma2 = gam * gam;
  const real_t beta2 = real_t(1) - real_t(1) / gamma2;

  real_t x, z, grej;
  if (!is_positron) {  // Moller
    const real_t gg = (real_t(2) * gam - real_t(1)) / gamma2;
    real_t y = real_t(1) - xmax;
    grej = real_t(1) - gg * xmax
           + xmax * xmax * (real_t(1) - gg + (real_t(1) - gg * y) / (y * y));
    do {
      const real_t r0 = rng.uniform(), r1 = rng.uniform();
      x = xmin * xmax / (xmin * (real_t(1) - r0) + xmax * r0);
      y = real_t(1) - x;
      z = real_t(1) - gg * x + x * x * (real_t(1) - gg + (real_t(1) - gg * y) / (y * y));
      if (grej * r1 <= z) { break; }
    } while (true);
  } else {  // Bhabha
    const real_t yy = real_t(1) / (real_t(1) + gam);
    const real_t y2 = yy * yy;
    const real_t y12 = real_t(1) - real_t(2) * yy;
    const real_t b1 = real_t(2) - y2;
    const real_t b2 = y12 * (real_t(3) + y2);
    const real_t y122 = y12 * y12;
    const real_t b4 = y122 * y12;
    const real_t b3 = b4 + y122;
    real_t y = xmax * xmax;
    grej = real_t(1) + (y * y * b4 - xmin * xmin * xmin * b3 + y * b2 - xmin * b1) * beta2;
    do {
      const real_t r0 = rng.uniform(), r1 = rng.uniform();
      x = xmin * xmax / (xmin * (real_t(1) - r0) + xmax * r0);
      y = x * x;
      z = real_t(1) + (y * y * b4 - x * y * b3 + y * b2 - x * b1) * beta2;
      if (grej * r1 <= z) { break; }
    } while (true);
  }

  const real_t delta_kin = x * kinetic;
  const real_t delta_mom = sqrt(delta_kin * (delta_kin + real_t(2) * me));
  const real_t primary_mom = sqrt(kinetic * (kinetic + real_t(2) * me));
  real_t cost = delta_kin * (energy + me) / (delta_mom * primary_mom);
  if (cost > real_t(1)) { cost = real_t(1); }
  const real_t sint = sqrt((real_t(1) - cost) * (real_t(1) + cost));
  const real_t phi = units::twopi<real_t>() * rng.uniform();
  const Vec3<real_t> local{sint * cos(phi), sint * sin(phi), cost};
  out.delta_dir = normalize(rotate_uz(local, dir));
  out.delta_ekin = delta_kin;

  // Primary recoil from momentum conservation, exactly as Geant4 does it.
  const Vec3<real_t> p_before = primary_mom * dir;
  const Vec3<real_t> p_delta = delta_mom * out.delta_dir;
  out.primary_dir = normalize(p_before - p_delta);
  out.primary_ekin = kinetic - delta_kin;
  out.produced = true;
  return out;
}

// ---------------------------------------------------------------- CSDA range table

constexpr int kRangeBins = 128;

/// Flat log-spaced CSDA range table, one row per material. Built on the host, read on device.
template <typename real_t>
struct RangeTable {
  real_t e_min, e_max;                                  ///< MeV
  real_t range[data::kMaxMaterials][kRangeBins];        ///< mm
  int n_materials = data::kNumMaterials;
  real_t log_e_min, inv_dlog_e;

  /// Inverse lookup: kinetic energy whose CSDA range is @p r, by binary search on the
  /// (monotonic) range row. Used to find the residual energy after a step of known length.
  __host__ __device__ real_t energy_from_range(int material, real_t r) const {
    const real_t* row = range[material];
    if (r <= row[0]) { return e_min * r / row[0]; }
    if (r >= row[kRangeBins - 1]) { return e_max; }
    int lo = 0, hi = kRangeBins - 1;
    while (hi - lo > 1) {
      const int mid = (lo + hi) / 2;
      if (row[mid] <= r) { lo = mid; } else { hi = mid; }
    }
    const real_t frac = (r - row[lo]) / (row[hi] - row[lo]);
    const real_t dlog = real_t(1) / inv_dlog_e;
    return exp(log_e_min + dlog * (real_t(lo) + frac));
  }

  __host__ __device__ real_t lookup(int material, real_t kinetic) const {
    if (kinetic <= e_min) {
      // dE/dx is finite, so range goes linearly to zero below the first bin.
      return range[material][0] * kinetic / e_min;
    }
    if (kinetic >= e_max) { return range[material][kRangeBins - 1]; }
    const real_t f = (log(kinetic) - log_e_min) * inv_dlog_e;
    const int i = static_cast<int>(f);
    const real_t frac = f - real_t(i);
    return range[material][i] * (real_t(1) - frac) + range[material][i + 1] * frac;
  }
};

/// Total restricted stopping power used for the range integral: restricted collision loss
/// plus, when @p bt is supplied, the real restricted Seltzer-Berger radiative loss instead
/// of the crude yield scaling.
template <typename real_t>
__host__ inline real_t range_dedx(const data::Material<real_t>* mats, int m, real_t e,
                                  const data::BremsTable<real_t>* bt) {
  const real_t col = collision_dedx(mats[m], e, false);
  const real_t rad = (bt != nullptr) ? bt->dedx_at(m, false, e) : radiative_dedx(mats[m], e);
  return col + rad;
}

/// Integrates 1/(dE/dx) from e_min up, trapezoid rule on a fine sub-grid.
template <typename real_t>
__host__ inline void build_range_table(const data::Material<real_t>* mats, RangeTable<real_t>& t,
                                       const data::BremsTable<real_t>* bt = nullptr,
                                       int n_materials = data::kNumMaterials,
                                       real_t e_min = real_t(1e-3), real_t e_max = real_t(100)) {
  t.e_min = e_min;
  t.e_max = e_max;
  t.log_e_min = std::log(e_min);
  const real_t dlog = (std::log(e_max) - t.log_e_min) / real_t(kRangeBins - 1);
  t.inv_dlog_e = real_t(1) / dlog;

  t.n_materials = n_materials;
  for (int m = 0; m < n_materials; ++m) {
    // Seed: below e_min assume constant dE/dx, so R(e_min) = e_min / (dE/dx)(e_min).
    real_t r = e_min / range_dedx(mats, m, e_min, bt);
    t.range[m][0] = r;
    for (int i = 1; i < kRangeBins; ++i) {
      const real_t e0 = std::exp(t.log_e_min + dlog * real_t(i - 1));
      const real_t e1 = std::exp(t.log_e_min + dlog * real_t(i));
      // sub-integrate this bin for accuracy
      constexpr int kSub = 16;
      const real_t de = (e1 - e0) / real_t(kSub);
      for (int k = 0; k < kSub; ++k) {
        const real_t ea = e0 + de * real_t(k);
        const real_t eb = ea + de;
        const real_t inv_a = real_t(1) / range_dedx(mats, m, ea, bt);
        const real_t inv_b = real_t(1) / range_dedx(mats, m, eb, bt);
        r += real_t(0.5) * (inv_a + inv_b) * de;
      }
      t.range[m][i] = r;
    }
  }
}

// ---------------------------------------------------------------- MSC (Highland)

/// RMS multiple-scattering deflection over a step, radians. Highland formula.
/// Radiation length is approximated from Z_eff and A_eff; see docs/RISK.md entry E3.
template <typename real_t>
__host__ __device__ inline real_t msc_theta0(const data::Material<real_t>& m, real_t kinetic,
                                             real_t step_mm) {
  if (step_mm <= real_t(0)) { return real_t(0); }
  const real_t me = units::electron_mass_c2<real_t>();
  const real_t p = sqrt(kinetic * (kinetic + real_t(2) * me));  // MeV/c
  const real_t e_total = kinetic + me;
  const real_t beta = p / e_total;
  if (p <= real_t(0) || beta <= real_t(0)) { return real_t(0); }
  const real_t x_over_x0 = step_mm / m.radiation_length;
  if (x_over_x0 <= real_t(0)) { return real_t(0); }
  return real_t(13.6) / (beta * p) * sqrt(x_over_x0)
         * (real_t(1) + real_t(0.038) * log(x_over_x0));
}

/// Applies an MSC deflection to @p dir, sampling the polar angle from a Gaussian of
/// width theta0 and the azimuth uniformly.
template <typename real_t, typename Rng>
__host__ __device__ inline Vec3<real_t> msc_scatter(const Vec3<real_t>& dir, real_t theta0,
                                                    Rng& rng) {
  if (theta0 <= real_t(0)) { return dir; }
  // Box-Muller for one Gaussian deviate.
  const real_t u1 = rng.uniform(), u2 = rng.uniform();
  const real_t theta = theta0 * sqrt(real_t(-2) * log(u1)) * cos(units::twopi<real_t>() * u2);
  const real_t phi = units::twopi<real_t>() * rng.uniform();
  const real_t st = sin(theta), ct = cos(theta);
  const Vec3<real_t> local{st * cos(phi), st * sin(phi), ct};
  return normalize(rotate_uz(local, dir));
}

}  // namespace g4gpu::em
