// Relativistic bremsstrahlung with LPM suppression, transcribed from
// G4eBremsstrahlungRelModel (11.1.1).
//
// G4eBremsstrahlung uses G4SeltzerBergerModel below 1 GeV with LPM off, and this model above
// 1 GeV with LPM on (G4EmParameters defaults flagLPM to true, and MaxKinEnergy to 100 TeV).
// B1 never reaches 1 GeV; this is here so the port covers the same energy range Geant4 does.
//
// The Landau-Pomeranchuk-Migdal effect suppresses emission of soft photons when the
// formation length of the radiation exceeds the mean free path between scatters, so the
// electron's successive interactions interfere destructively. It matters above roughly
// E_LPM = gLPMconstant * X0, which is tens of TeV in water and a few TeV in lead.
//
// Not transcribed: the triplet model branch (fIsScatOffElectron selecting scattering off an
// atomic electron rather than the nucleus, which hands off to G4eplusTo2GammaOKVIModel's
// sibling triplet model). The nuclear term dominates; see docs/RISK.md.
#pragma once
#include <cmath>
#include "core/units.cuh"
#include "core/vec3.cuh"
#include "data/materials.cuh"

namespace g4gpu::em {

/// Per-element constants, from G4eBremsstrahlungRelModel::InitialiseElementData.
template <typename real_t>
struct RelBremElement {
  real_t log_z, fz;
  real_t zfactor1, zfactor2;
  real_t var_s1, il_var_s1, il_var_s1_cond;
  real_t gamma_factor, epsilon_factor;
};

/// Coulomb correction fc, as G4Element::GetfCoulomb computes it (already transcribed in
/// data/materials.cuh as coulomb_factor).
template <typename real_t>
__host__ __device__ inline RelBremElement<real_t> rel_brem_element(int z) {
  RelBremElement<real_t> e;
  const real_t zet = real_t(z);
  const real_t fc = data::coulomb_factor<real_t>(zet);
  e.log_z = log(zet);
  e.fz = e.log_z / real_t(3) + fc;

  // Tabulated Fel/Finel for Z < 5, where the logarithmic forms are inaccurate.
  constexpr real_t fel_low[8] = {real_t(0.0),    real_t(5.3104), real_t(4.7935), real_t(4.7402),
                                 real_t(4.7112), real_t(4.6694), real_t(4.6134), real_t(4.5520)};
  constexpr real_t finel_low[8] = {real_t(0.0),    real_t(5.9173), real_t(5.6125),
                                   real_t(5.5377), real_t(5.4728), real_t(5.4174),
                                   real_t(5.3688), real_t(5.3236)};
  real_t Fel, Finel;
  if (z < 5) {
    Fel = fel_low[z];
    Finel = finel_low[z];
  } else {
    Fel = log(real_t(184.15)) - e.log_z / real_t(3);
    Finel = log(real_t(1194.0)) - real_t(2) * e.log_z / real_t(3);
  }
  const real_t z13 = exp(e.log_z / real_t(3));
  const real_t z23 = z13 * z13;
  e.zfactor1 = (Fel - fc) + Finel / zet;
  e.zfactor2 = (real_t(1) + real_t(1) / zet) / real_t(12);
  e.var_s1 = z23 / (real_t(184.15) * real_t(184.15));
  e.il_var_s1_cond = real_t(1) / log(sqrt(real_t(2)) * e.var_s1);
  e.il_var_s1 = real_t(1) / log(e.var_s1);
  const real_t me = units::electron_mass_c2<real_t>();
  e.gamma_factor = real_t(100) * me / z13;
  e.epsilon_factor = real_t(100) * me / z23;
  return e;
}

/// gMigdalConstant = 4 pi r_e * lambda_C^2.
template <typename real_t> __host__ __device__ inline real_t migdal_constant_rel() {
  const real_t re = units::classic_electron_radius<real_t>();
  constexpr real_t lambda_c = units::electron_compton_length<real_t>();  // mm, reduced Compton wavelength
  return real_t(4) * real_t(3.14159265358979323846) * re * lambda_c * lambda_c;
}

/// gLPMconstant = alpha * m_e^2 / (4 pi hbar c).
template <typename real_t> __host__ __device__ inline real_t lpm_constant() {
  constexpr real_t alpha = units::fine_structure_const<real_t>();
  const real_t me = units::electron_mass_c2<real_t>();
  constexpr real_t hbarc = units::hbarc<real_t>();  // MeV*mm, CLHEP's derived value
  return alpha * me * me / (real_t(4) * real_t(3.14159265358979323846) * hbarc);
}

/// gBremFactor = 16 alpha r_e^2 / 3.
template <typename real_t> __host__ __device__ inline real_t rel_brem_factor() {
  constexpr real_t alpha = units::fine_structure_const<real_t>();
  const real_t re = units::classic_electron_radius<real_t>();
  return real_t(16) * alpha * re * re / real_t(3);
}

/// Tsai screening functions phi1, phi1-phi2, psi1, psi1-psi2.
/// Verbatim from G4eBremsstrahlungRelModel::ComputeScreeningFunctions.
template <typename real_t>
__host__ __device__ inline void rel_brem_screening(real_t& phi1, real_t& phi1m2, real_t& psi1,
                                                   real_t& psi1m2, real_t gam, real_t eps) {
  const real_t gam2 = gam * gam;
  phi1 = real_t(16.863) - real_t(2) * log(real_t(1) + real_t(0.311877) * gam2)
         + real_t(2.4) * exp(real_t(-0.9) * gam) + real_t(1.6) * exp(real_t(-1.5) * gam);
  phi1m2 = real_t(2) / (real_t(3) + real_t(19.5) * gam + real_t(18) * gam2);
  const real_t eps2 = eps * eps;
  psi1 = real_t(24.34) - real_t(2) * log(real_t(1) + real_t(13.111641) * eps2)
         + real_t(2.8) * exp(real_t(-8) * eps) + real_t(1.2) * exp(real_t(-29.2) * eps);
  psi1m2 = real_t(2) / (real_t(3) + real_t(120) * eps + real_t(1200) * eps2);
}

/// LPM suppression functions G(s) and phi(s).
/// Verbatim from G4eBremsstrahlungRelModel::ComputeLPMGsPhis. Geant4 tabulates these at
/// initialisation and interpolates; evaluating them directly gives the same values without
/// a table, and at these energies the call count is negligible.
template <typename real_t>
__host__ __device__ inline void lpm_gs_phis(real_t& gs, real_t& phis, real_t s) {
  constexpr real_t pi = real_t(3.14159265358979323846);
  if (s < real_t(0.01)) {
    phis = real_t(6) * s * (real_t(1) - pi * s);
    gs = real_t(12) * s - real_t(2) * phis;
    return;
  }
  const real_t s2 = s * s, s3 = s * s2, s4 = s2 * s2;
  if (s < real_t(0.415827)) {
    phis = real_t(1)
           - exp(real_t(-6) * s * (real_t(1) + s * (real_t(3) - pi))
                 + s3 / (real_t(0.623) + real_t(0.796) * s + real_t(0.658) * s2));
    const real_t psis =
        real_t(1)
        - exp(real_t(-4) * s
              - real_t(8) * s2
                    / (real_t(1) + real_t(3.936) * s + real_t(4.97) * s2 - real_t(0.05) * s3
                       + real_t(7.5) * s4));
    gs = real_t(3) * psis - real_t(2) * phis;
  } else if (s < real_t(1.55)) {
    phis = real_t(1)
           - exp(real_t(-6) * s * (real_t(1) + s * (real_t(3) - pi))
                 + s3 / (real_t(0.623) + real_t(0.796) * s + real_t(0.658) * s2));
    const real_t d = real_t(-0.160723) + real_t(3.755030) * s - real_t(1.798138) * s2
                     + real_t(0.672827) * s3 - real_t(0.120772) * s4;
    gs = tanh(d);
  } else {
    phis = real_t(1) - real_t(0.011905) / s4;
    if (s < real_t(1.9156)) {
      const real_t d = real_t(-0.160723) + real_t(3.755030) * s - real_t(1.798138) * s2
                       + real_t(0.672827) * s3 - real_t(0.120772) * s4;
      gs = tanh(d);
    } else {
      gs = real_t(1) - real_t(0.023065) / s4;
    }
  }
}

/// xi(s), G(s), phi(s) for one photon energy.
/// Verbatim from G4eBremsstrahlungRelModel::ComputeLPMfunctions.
template <typename real_t>
__host__ __device__ inline void lpm_functions(real_t& xi_s, real_t& gs, real_t& phis,
                                              const RelBremElement<real_t>& el, real_t egamma,
                                              real_t total_energy, real_t lpm_energy,
                                              real_t density_corr) {
  const real_t sqrt2 = sqrt(real_t(2));
  const real_t red = egamma / total_energy;
  const real_t s_prime =
      sqrt(real_t(0.125) * red * lpm_energy / ((real_t(1) - red) * total_energy));
  const real_t condition = sqrt2 * el.var_s1;
  real_t xi_s_prime = real_t(2);
  if (s_prime > real_t(1)) {
    xi_s_prime = real_t(1);
  } else if (s_prime > condition) {
    const real_t h = log(s_prime) * el.il_var_s1_cond;
    xi_s_prime = real_t(1) + h
                 - real_t(0.08) * (real_t(1) - h) * h * (real_t(2) - h) * el.il_var_s1_cond;
  }
  const real_t s = s_prime / sqrt(xi_s_prime);
  const real_t s_hat = s * (real_t(1) + density_corr / (egamma * egamma));
  xi_s = real_t(2);
  if (s_hat > real_t(1)) {
    xi_s = real_t(1);
  } else if (s_hat > el.var_s1) {
    xi_s = real_t(1) + log(s_hat) * el.il_var_s1;
  }
  lpm_gs_phis(gs, phis, s_hat);
  if (xi_s * phis > real_t(1) || s_hat > real_t(0.57)) { xi_s = real_t(1) / phis; }
}

/// Differential cross section WITHOUT LPM (the "complete screening plus Tsai" form).
/// Verbatim from G4eBremsstrahlungRelModel::ComputeDXSectionPerAtom.
template <typename real_t>
__host__ __device__ inline real_t rel_brem_dxs(const RelBremElement<real_t>& el, int z,
                                               real_t egamma, real_t total_energy) {
  if (egamma < real_t(0)) { return real_t(0); }
  const real_t y = egamma / total_energy;
  const real_t onemy = real_t(1) - y;
  const real_t dum0 = onemy + real_t(0.75) * y * y;
  real_t dxsec;
  if (z < 5) {
    dxsec = dum0 * el.zfactor1 + onemy * el.zfactor2;
  } else {
    const real_t invZ = real_t(1) / real_t(z);
    const real_t dum1 = y / (total_energy - egamma);
    const real_t gamma = dum1 * el.gamma_factor;
    const real_t epsilon = dum1 * el.epsilon_factor;
    real_t phi1, phi1m2, psi1, psi1m2;
    rel_brem_screening(phi1, phi1m2, psi1, psi1m2, gamma, epsilon);
    dxsec = dum0
            * ((real_t(0.25) * phi1 - el.fz)
               + (real_t(0.25) * psi1 - real_t(2) * el.log_z / real_t(3)) * invZ);
    dxsec += real_t(0.125) * onemy * (phi1m2 + psi1m2 * invZ);
  }
  return fmax(dxsec, real_t(0));
}

/// Differential cross section WITH LPM suppression.
/// Verbatim from G4eBremsstrahlungRelModel::ComputeRelDXSectionPerAtom.
template <typename real_t>
__host__ __device__ inline real_t rel_brem_dxs_lpm(const RelBremElement<real_t>& el,
                                                   real_t egamma, real_t total_energy,
                                                   real_t lpm_energy, real_t density_corr) {
  if (egamma < real_t(0)) { return real_t(0); }
  const real_t y = egamma / total_energy;
  const real_t onemy = real_t(1) - y;
  const real_t dum0 = real_t(0.25) * y * y;
  real_t xi_s, gs, phis;
  lpm_functions(xi_s, gs, phis, el, egamma, total_energy, lpm_energy, density_corr);
  const real_t term1 = xi_s * (dum0 * gs + (onemy + real_t(2) * dum0) * phis);
  return fmax(term1 * el.zfactor1 + onemy * el.zfactor2, real_t(0));
}

/// Per-material quantities SetupForMaterial computes.
template <typename real_t>
struct RelBremSetup {
  real_t density_corr;   ///< fDensityFactor * E_total^2
  real_t lpm_energy;
  bool lpm_active;
};

template <typename real_t>
__host__ __device__ inline RelBremSetup<real_t> rel_brem_setup(const data::Material<real_t>& m,
                                                               real_t total_energy,
                                                               bool lpm_flag = true) {
  RelBremSetup<real_t> s;
  const real_t density_factor = migdal_constant_rel<real_t>() * m.electron_density;
  s.lpm_energy = lpm_constant<real_t>() * m.radiation_length;
  const real_t threshold =
      lpm_flag ? sqrt(density_factor) * s.lpm_energy : real_t(1e39);
  s.density_corr = density_factor * total_energy * total_energy;
  s.lpm_active = (total_energy > threshold);
  return s;
}

/// 8-point Gauss-Legendre abscissae and weights, as the model uses.
template <typename real_t> __host__ __device__ inline const real_t* rel_brem_xgl() {
  static const real_t v[8] = {real_t(1.98550718e-02), real_t(1.01666761e-01),
                              real_t(2.37233795e-01), real_t(4.08282679e-01),
                              real_t(5.91717321e-01), real_t(7.62766205e-01),
                              real_t(8.98333239e-01), real_t(9.80144928e-01)};
  return v;
}
template <typename real_t> __host__ __device__ inline const real_t* rel_brem_wgl() {
  static const real_t v[8] = {real_t(5.06142681e-02), real_t(1.11190517e-01),
                              real_t(1.56853323e-01), real_t(1.81341892e-01),
                              real_t(1.81341892e-01), real_t(1.56853323e-01),
                              real_t(1.11190517e-01), real_t(5.06142681e-02)};
  return v;
}

/// Integral of the DCS from tmin up to the primary kinetic energy.
/// Verbatim from G4eBremsstrahlungRelModel::ComputeXSectionPerAtom.
template <typename real_t>
__host__ __device__ inline real_t rel_brem_xs_integral(const RelBremElement<real_t>& el, int z,
                                                       real_t tmin, real_t kinetic,
                                                       real_t total_energy,
                                                       const RelBremSetup<real_t>& su) {
  const real_t alpha_min = log(tmin / total_energy);
  const real_t alpha_max = log(kinetic / total_energy);
  const int n_sub = static_cast<int>(real_t(0.45) * (alpha_max - alpha_min)) + 4;
  const real_t delta = (alpha_max - alpha_min) / real_t(n_sub);
  const real_t* xgl = rel_brem_xgl<real_t>();
  const real_t* wgl = rel_brem_wgl<real_t>();
  real_t xs = real_t(0);
  real_t alpha_i = alpha_min;
  for (int l = 0; l < n_sub; ++l) {
    for (int i = 0; i < 8; ++i) {
      const real_t k = exp(alpha_i + xgl[i] * delta) * total_energy;
      const real_t dcs = su.lpm_active
                             ? rel_brem_dxs_lpm(el, k, total_energy, su.lpm_energy,
                                                su.density_corr)
                             : rel_brem_dxs(el, z, k, total_energy);
      xs += wgl[i] * dcs / (real_t(1) + su.density_corr / (k * k));
    }
    alpha_i += delta;
  }
  return fmax(xs * delta, real_t(0));
}

/// Cross section per atom above the photon production cut, mm^2.
/// Verbatim from G4eBremsstrahlungRelModel::ComputeCrossSectionPerAtom.
template <typename real_t>
__host__ __device__ inline real_t rel_brem_xs_per_atom(const data::Material<real_t>& m, int z,
                                                       real_t kinetic, real_t cut,
                                                       real_t max_energy) {
  const real_t me = units::electron_mass_c2<real_t>();
  const real_t total_energy = kinetic + me;
  const real_t tmin = fmin(cut, kinetic);
  const real_t tmax = fmin(max_energy, kinetic);
  if (tmin >= tmax) { return real_t(0); }
  const RelBremElement<real_t> el = rel_brem_element<real_t>(z);
  const RelBremSetup<real_t> su = rel_brem_setup(m, total_energy);
  real_t xs = rel_brem_xs_integral(el, z, tmin, kinetic, total_energy, su);
  if (tmax < kinetic) {
    xs -= rel_brem_xs_integral(el, z, tmax, kinetic, total_energy, su);
  }
  xs *= real_t(z) * real_t(z) * rel_brem_factor<real_t>();
  return fmax(xs, real_t(0));
}

/// Restricted radiative energy loss per atom.
/// Verbatim from G4eBremsstrahlungRelModel::ComputeBremLoss.
template <typename real_t>
__host__ __device__ inline real_t rel_brem_loss_per_atom(const data::Material<real_t>& m, int z,
                                                         real_t kinetic, real_t tmax_in) {
  const real_t me = units::electron_mass_c2<real_t>();
  const real_t total_energy = kinetic + me;
  const real_t tmax = fmin(tmax_in, kinetic);
  if (tmax <= real_t(0)) { return real_t(0); }
  const RelBremElement<real_t> el = rel_brem_element<real_t>(z);
  const RelBremSetup<real_t> su = rel_brem_setup(m, total_energy);
  const real_t alpha_max = tmax / total_energy;
  const int n_sub = static_cast<int>(real_t(20) * alpha_max) + 3;
  const real_t delta = alpha_max / real_t(n_sub);
  const real_t* xgl = rel_brem_xgl<real_t>();
  const real_t* wgl = rel_brem_wgl<real_t>();
  real_t integ = real_t(0);
  real_t alpha_i = real_t(0);
  for (int l = 0; l < n_sub; ++l) {
    for (int i = 0; i < 8; ++i) {
      const real_t k = (alpha_i + xgl[i] * delta) * total_energy;
      const real_t dcs = su.lpm_active
                             ? rel_brem_dxs_lpm(el, k, total_energy, su.lpm_energy,
                                                su.density_corr)
                             : rel_brem_dxs(el, z, k, total_energy);
      integ += wgl[i] * dcs / (real_t(1) + su.density_corr / (k * k));
    }
    alpha_i += delta;
  }
  integ *= delta * total_energy;
  return fmax(integ, real_t(0)) * real_t(z) * real_t(z) * rel_brem_factor<real_t>();
}

// ------------------------------------------------- the 1 GeV boundary is not a step
//
// WHAT GEANT4 DOES AT EXACTLY 1 GeV, AND WHAT IT DOES JUST ABOVE IT. Two different answers,
// and the port had the first one right and the second one missing.
//
// **At exactly 1 GeV the model is Seltzer-Berger.** `G4eBremsstrahlung::
// InitialiseEnergyLossProcess` sets `EmModel(0)->SetHighEnergyLimit(min(SB high, 1 GeV))` and
// `EmModel(1)->SetLowEnergyLimit(energyLimit)`, and `G4RegionModels::SelectIndex` resolves an
// energy with `do {--idx;} while (idx > 0 && e <= lowKineticEnergy[idx])` - a `<=` against the
// second model's low edge. So `data::kSeltzerBergerLimit` is a strict `>` everywhere in this
// port, which is right, and it is the easy half.
//
// **Above 1 GeV the TABLE is not the model.** `G4EmModelManager::FillDEDXVector` and
// `::FillLambdaVector` both carry a continuity correction across a model boundary
// (G4EmModelManager.cc:596-606 and :700-711), and it is the same four lines in each:
//
//     if(k > 0 && k != k0) {
//       k0 = k;
//       G4double elow = regModels->LowEdgeEnergy(k);
//       G4double xs1  = mod1->CrossSection(couple, particle, elow, cut, tmax);   // model k-1
//       G4double xs2  = mod ->CrossSection(couple, particle, elow, cut, tmax);   // model k
//       del = (xs2 > 0.0) ? (xs1/xs2 - 1.0)*elow : 0.0;
//     }
//     G4double cross = (1.0 + del/e)*mod->CrossSection(couple, particle, e, cut, tmax);
//
// `del` is fixed once per material from the RATIO OF THE TWO MODELS AT THE BOUNDARY, and the
// factor `1 + del/e` is exactly 1 + (xs1/xs2 - 1) at `e = elow` - so the tabulated vector is
// continuous there - and falls off as 1/e above it. Geant4 is smoothing a discontinuity
// between two models it does not otherwise reconcile.
//
// It is worth 1.8% of a 1 GeV electron's bremsstrahlung rate in water and a fifth of a per cent
// at 10 GeV, which is the size of thing this package exists to stop being invisible: the raw
// models differ by 2.1% at 1 GeV in water (`ref/oracle/electron_tables.csv`, e- at 1000 MeV on
// Seltzer-Berger against 1059 MeV on the relativistic model), so without this the port draws a
// 1 GeV electron's bremsstrahlung interaction length from a cross section 1.8% too small.
//
// IT CANNOT MOVE ANYTHING BELOW 1 GeV. `del` is zero for the model below the boundary -
// Geant4's `k > 0` test - so every energy the Seltzer-Berger model covers is untouched, which
// includes every secondary B1's 6 MeV gamma gate ever makes.
//
// NOT applied to the msc transport mean free path, and that is not an omission: an msc model's
// table is built per MODEL by `G4LossTableBuilder::BuildTableForModel` over that model's own
// energy window, not by `FillLambdaVector` over a process's, so there is no boundary inside it
// to smooth. See `em/wentzel_msc.cuh`.

/// `G4EmModelManager`'s `del` for a two-model process, MeV - the numerator of the `1 + del/e`
/// continuity factor.
///
/// @param below the quantity from the model BELOW the boundary, evaluated AT the boundary
/// @param above the same from the model above it, at the same energy and the same cut
/// @param elow  the boundary energy, `regModels->LowEdgeEnergy(k)`
template <typename real_t>
__host__ __device__ inline real_t model_boundary_del(real_t below, real_t above, real_t elow) {
  return (above > real_t(0)) ? (below / above - real_t(1)) * elow : real_t(0);
}

/// The factor itself, for an energy above the boundary. Exactly 1 at or below it.
template <typename real_t>
__host__ __device__ inline real_t model_boundary_factor(real_t del, real_t e, real_t elow) {
  return (e > elow && e > real_t(0)) ? (real_t(1) + del / e) : real_t(1);
}

/// Samples the emitted photon energy. Verbatim from the rejection loop in
/// G4eBremsstrahlungRelModel::SampleSecondaries: uniform in log(k^2 + densityCorr),
/// rejected against the DCS with funcMax = zfactor1 + zfactor2.
template <typename real_t, typename Rng>
__host__ __device__ inline real_t sample_rel_brem_energy(const data::Material<real_t>& m, int z,
                                                         real_t kinetic, real_t cut,
                                                         real_t max_energy, Rng& rng) {
  const real_t me = units::electron_mass_c2<real_t>();
  const real_t total_energy = kinetic + me;
  const real_t tmin = fmin(cut, kinetic);
  const real_t tmax = fmin(max_energy, kinetic);
  if (tmin >= tmax) { return real_t(0); }
  const RelBremElement<real_t> el = rel_brem_element<real_t>(z);
  const RelBremSetup<real_t> su = rel_brem_setup(m, total_energy);
  const real_t func_max = el.zfactor1 + el.zfactor2;
  const real_t xmin = log(tmin * tmin + su.density_corr);
  const real_t xrange = log(tmax * tmax + su.density_corr) - xmin;

  real_t egamma = tmin;
  for (int guard = 0; guard < 1000; ++guard) {
    const real_t r0 = rng.uniform(), r1 = rng.uniform();
    egamma = sqrt(fmax(exp(xmin + r0 * xrange) - su.density_corr, real_t(0)));
    const real_t f = su.lpm_active
                         ? rel_brem_dxs_lpm(el, egamma, total_energy, su.lpm_energy,
                                            su.density_corr)
                         : rel_brem_dxs(el, z, egamma, total_energy);
    if (f >= func_max * r1) { break; }
  }
  return egamma;
}

}  // namespace g4gpu::em
