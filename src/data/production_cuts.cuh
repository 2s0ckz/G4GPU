// Range cut -> energy threshold conversion, for arbitrary materials.
//
// Transcribed from G4VRangeToEnergyConverter (the inversion), G4RToEConvForElectron and
// G4RToEConvForGamma (the per-element loss and absorption approximations Geant4 uses for
// this purpose only - they are deliberately crude stand-ins, not the transport physics).
//
// This replaces per-material cut values copied out of a Geant4 dump, so a user-supplied
// material gets correct thresholds without anyone hand-entering them.
#pragma once
#include <cmath>
#include <vector>
#include "core/units.cuh"

namespace g4gpu::data {

/// Geant4 defaults: 1 keV to 10 GeV, 50 bins per decade -> 350 bins.
constexpr int kCutNbinPerDecade = 50;
template <typename real_t> __host__ constexpr real_t cut_emin() { return real_t(1e-3); }
template <typename real_t> __host__ constexpr real_t cut_emax() { return real_t(1e4); }

/// Lower and upper clamps Geant4 applies to any converted cut.
template <typename real_t> __host__ constexpr real_t cut_floor() { return real_t(0.99e-3); }
template <typename real_t> __host__ constexpr real_t cut_ceiling() { return real_t(1e4); }

/// Approximate electron stopping power used only for cut conversion.
/// Transcribed from G4RToEConvForElectron::ComputeValue.
template <typename real_t>
__host__ inline real_t cut_loss_electron(int Z, real_t kin) {
  constexpr real_t cbr1 = real_t(0.02), cbr2 = real_t(-5.7e-5), cbr3 = real_t(1.0),
                   cbr4 = real_t(0.072);
  const real_t me = units::electron_mass_c2<real_t>();
  const real_t Tlow = real_t(10e-3), Thigh = real_t(1e3);  // 10 keV, 1 GeV
  const real_t taul = Tlow / me;
  const real_t log05 = std::log(real_t(0.5));
  const real_t taul12 = std::sqrt(taul);
  constexpr real_t bremfactor = real_t(0.1);
  const real_t Zlog = std::log(real_t(Z));
  const real_t ionpot = real_t(1.6e-5) * std::exp(real_t(0.9) * Zlog) / me;
  const real_t ionpotlog = std::log(ionpot);
  const real_t tau = kin / me;

  real_t dEdx;
  if (tau < taul) {
    const real_t t1 = taul + real_t(1), t2 = taul + real_t(2), tsq = taul * taul;
    const real_t beta2 = taul * t2 / (t1 * t1);
    const real_t f = real_t(1) - beta2 + std::log(tsq / real_t(2))
                     + (real_t(0.5) + real_t(0.25) * tsq
                        + (real_t(1) + real_t(2) * taul) * log05) / (t1 * t1);
    dEdx = real_t(Z) * (std::log(real_t(2) * taul + real_t(4)) - real_t(2) * ionpotlog + f)
           / beta2;
    dEdx *= taul12 / std::sqrt(tau);
  } else {
    const real_t t1 = tau + real_t(1), t2 = tau + real_t(2), tsq = tau * tau;
    const real_t beta2 = tau * t2 / (t1 * t1);
    const real_t f = real_t(1) - beta2 + std::log(tsq / real_t(2))
                     + (real_t(0.5) + real_t(0.25) * tsq
                        + (real_t(1) + real_t(2) * tau) * log05) / (t1 * t1);
    dEdx = real_t(Z) * (std::log(real_t(2) * tau + real_t(4)) - real_t(2) * ionpotlog + f)
           / beta2;
    const real_t cbrem = (cbr1 + cbr2 * real_t(Z)) * (cbr3 + cbr4 * std::log(kin / Thigh));
    dEdx += real_t(Z) * real_t(Z + 1) * cbrem * bremfactor * tau / beta2;
  }
  // twopi_mc2_rcl2
  const real_t re = units::classic_electron_radius<real_t>();
  return dEdx * units::twopi<real_t>() * me * re * re;
}

/// Approximate positron stopping power used only for cut conversion.
/// Transcribed from G4RToEConvForPositron::ComputeValue. Geant4 uses a genuinely different
/// expression here from the electron one, not a rescaling, which is why the positron cut
/// comes out ~2% below the electron cut.
template <typename real_t>
__host__ inline real_t cut_loss_positron(int Z, real_t kin) {
  constexpr real_t cbr1 = real_t(0.02), cbr2 = real_t(-5.7e-5), cbr3 = real_t(1.0),
                   cbr4 = real_t(0.072);
  const real_t me = units::electron_mass_c2<real_t>();
  const real_t Tlow = real_t(10e-3), Thigh = real_t(1e3);
  const real_t taul = Tlow / me;
  const real_t logtaul = std::log(taul);
  const real_t taul12 = std::sqrt(taul);
  constexpr real_t bremfactor = real_t(0.1);
  const real_t Zlog = std::log(real_t(Z));
  const real_t ionpot = real_t(1.6e-5) * std::exp(real_t(0.9) * Zlog) / me;
  const real_t ionpotlog = std::log(ionpot);
  const real_t tau = kin / me;

  real_t dEdx;
  if (tau < taul) {
    const real_t t1 = taul + real_t(1), t2 = taul + real_t(2), tsq = taul * taul;
    const real_t beta2 = taul * t2 / (t1 * t1);
    const real_t f = real_t(2) * logtaul
                     - (real_t(6) * taul + real_t(1.5) * tsq
                        - taul * (real_t(1) - tsq / real_t(3)) / t2
                        - tsq * (real_t(0.5) - tsq / real_t(12)) / (t2 * t2)) / (t1 * t1);
    dEdx = (std::log(real_t(2) * taul + real_t(4)) - real_t(2) * ionpotlog + f) / beta2;
    dEdx *= real_t(Z) * taul12 / std::sqrt(tau);
  } else {
    const real_t t1 = tau + real_t(1), t2 = tau + real_t(2), tsq = tau * tau;
    const real_t beta2 = tau * t2 / (t1 * t1);
    const real_t f = real_t(2) * std::log(tau)
                     - (real_t(6) * tau + real_t(1.5) * tsq
                        - tau * (real_t(1) - tsq / real_t(3)) / t2
                        - tsq * (real_t(0.5) - tsq / real_t(12)) / (t2 * t2)) / (t1 * t1);
    dEdx = real_t(Z) * (std::log(real_t(2) * tau + real_t(4)) - real_t(2) * ionpotlog + f)
           / beta2;
    const real_t cbrem = (cbr1 + cbr2 * real_t(Z)) * (cbr3 + cbr4 * std::log(kin / Thigh));
    dEdx += cbrem * real_t(Z) * real_t(Z + 1) * bremfactor * tau / beta2;
  }
  const real_t re = units::classic_electron_radius<real_t>();
  return dEdx * units::twopi<real_t>() * me * re * re;
}

/// Approximate photon absorption cross section used only for cut conversion.
/// Transcribed from G4RToEConvForGamma::ComputeValue.
template <typename real_t>
__host__ inline real_t cut_xs_gamma(int Z, real_t energy) {
  const real_t t1keV = real_t(1e-3), t200keV = real_t(0.2), t100MeV = real_t(100.0);
  const real_t Zd = real_t(Z);
  const real_t Zsq = Zd * Zd;
  const real_t Zlog = std::log(Zd);
  const real_t Zlogsq = Zlog * Zlog;
  const real_t tmin = (real_t(0.552) + real_t(218.5) / Zd + real_t(557.17) / Zsq);
  const real_t tlow = real_t(0.2) * std::exp(real_t(-7.355) / std::sqrt(Zd));
  const real_t smin =
      (real_t(0.01239) + real_t(0.005585) * Zlog - real_t(0.000923) * Zlogsq)
      * std::exp(real_t(1.5) * Zlog);
  const real_t s200keV =
      (real_t(0.2651) - real_t(0.1501) * Zlog + real_t(0.02283) * Zlogsq) * Zsq;
  const real_t cminlog = std::log(tmin / t200keV);
  const real_t cmin = std::log(s200keV / smin) / (cminlog * cminlog);
  const real_t slowlog = std::log(t200keV / tlow);
  const real_t slow = s200keV * std::exp(real_t(0.042) * Zd * slowlog * slowlog);
  const real_t logtlow = std::log(tlow / t1keV);
  const real_t clow = std::log(real_t(300) * Zsq / slow) / logtlow;
  const real_t chigh =
      (real_t(7.55e-5) - real_t(0.0542e-5) * Zd) * Zsq * Zd / std::log(t100MeV / tmin);

  real_t xs;
  if (energy < tlow) {
    xs = (energy < t1keV) ? slow * std::exp(clow * logtlow)
                          : slow * std::exp(clow * std::log(tlow / energy));
  } else if (energy < t200keV) {
    const real_t x = std::log(t200keV / energy);
    xs = s200keV * std::exp(real_t(0.042) * Zd * x * x);
  } else if (energy < tmin) {
    const real_t x = std::log(tmin / energy);
    xs = smin * std::exp(cmin * x * x);
  } else {
    xs = smin + chigh * std::log(energy / tmin);
  }
  return xs * units::barn<real_t>();
}

template <typename real_t>
__host__ inline real_t linear_interp_cut(real_t e1, real_t e2, real_t r1, real_t r2, real_t r) {
  return (r2 > r1) ? e1 + (e2 - e1) * (r - r1) / (r2 - r1) : e1;
}

/// Shared log energy grid, built once, matching G4VRangeToEnergyConverter::FillEnergyVector.
template <typename real_t>
__host__ inline const std::vector<real_t>& cut_energy_grid() {
  static std::vector<real_t> g;
  if (g.empty()) {
    const real_t emin = cut_emin<real_t>(), emax = cut_emax<real_t>();
    const int nbin =
        kCutNbinPerDecade * static_cast<int>(std::lround(std::log10(emax / emin)));
    g.resize(nbin + 1);
    g[0] = emin;
    g[nbin] = emax;
    const real_t fact = std::log(emax / emin) / real_t(nbin);
    for (int i = 1; i < nbin; ++i) { g[i] = emin * std::exp(real_t(i) * fact); }
  }
  return g;
}

/// Electron or positron cut, mm range -> MeV. @p zs and @p n_atoms describe the material.
/// Transcribed from ConvertForElectron plus the low-energy tune in Convert(); the same
/// inversion serves both species, with the species-specific loss function selected by
/// @p is_positron.
template <typename real_t>
__host__ inline real_t convert_cut_electron(real_t range_cut_mm, int n_elements,
                                            const real_t* zs, const real_t* n_atoms,
                                            real_t density_g_cm3, bool is_positron = false) {
  const auto& grid = cut_energy_grid<real_t>();
  real_t dedx1 = 0, range = 0, range1 = 0, range2 = 0, e1 = 0, e2 = 0;
  for (size_t i = 0; i < grid.size(); ++i) {
    e2 = grid[i];
    real_t dedx2 = 0;
    for (int j = 0; j < n_elements; ++j) {
      const int z = static_cast<int>(zs[j] + real_t(0.5));
      dedx2 += n_atoms[j] * (is_positron ? cut_loss_positron<real_t>(z, e2)
                                         : cut_loss_electron<real_t>(z, e2));
    }
    range += (dedx1 + dedx2 > real_t(0)) ? real_t(2) * (e2 - e1) / (dedx1 + dedx2) : real_t(0);
    range2 = range;
    if (range2 < range_cut_mm) {
      e1 = e2;
      dedx1 = dedx2;
      range1 = range2;
    } else {
      break;
    }
  }
  real_t cut = linear_interp_cut(e1, e2, range1, range2, range_cut_mm);

  // Convert()'s low-energy tune. 0.025 mm * g/cm3 in our units, with density in g/cm3.
  constexpr real_t tune = real_t(0.025);
  constexpr real_t lowen = real_t(30e-3);  // 30 keV
  if (cut < lowen) {
    cut /= (real_t(1) + (real_t(1) - cut / lowen) * tune / (range_cut_mm * density_g_cm3));
  }
  return std::max(cut_floor<real_t>(), std::min(cut, cut_ceiling<real_t>()));
}

/// Gamma cut, mm range -> MeV. Transcribed from ConvertForGamma; note Geant4 defines the
/// gamma "range" as 5 absorption lengths.
template <typename real_t>
__host__ inline real_t convert_cut_gamma(real_t range_cut_mm, int n_elements, const real_t* zs,
                                         const real_t* n_atoms) {
  const auto& grid = cut_energy_grid<real_t>();
  real_t range1 = 0, range2 = 0, e1 = 0, e2 = 0;
  for (size_t i = 0; i < grid.size(); ++i) {
    e2 = grid[i];
    real_t sig = 0;
    for (int j = 0; j < n_elements; ++j) {
      sig += n_atoms[j] * cut_xs_gamma<real_t>(static_cast<int>(zs[j] + real_t(0.5)), e2);
    }
    range2 = (sig > real_t(0)) ? real_t(5) / sig : real_t(1e30);
    if (i == 0 || range2 < range_cut_mm) {
      e1 = e2;
      range1 = range2;
    } else {
      break;
    }
  }
  const real_t cut = linear_interp_cut(e1, e2, range1, range2, range_cut_mm);
  return std::max(cut_floor<real_t>(), std::min(cut, cut_ceiling<real_t>()));
}

}  // namespace g4gpu::data
