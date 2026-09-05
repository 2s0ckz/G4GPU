// Energy-loss fluctuations for an alpha or an ion: G4IonFluctuations (11.1.1).
//
// Why this is a separate model from G4UniversalFluctuation and not a special case of it:
//
//   G4EmStandUtil::ModelOfFluctuations(isIon) hands back G4IonFluctuations rather than
//   G4UniversalFluctuation, and G4hIonisation asks for it by name -
//
//       G4bool ion = (pname == "GenericIon" || pname == "alpha");
//       SetFluctModel(G4EmStandUtil::ModelOfFluctuations(ion));       G4hIonisation.cc:146
//
//   so an alpha in a stock physics list has never used the Universal model. G4ionIonisation
//   passes `true` unconditionally, and G4MuIonisation passes it for its low-energy model.
//   This port used the Universal model for alphas until it was measured against the source;
//   the mean is unchanged by any fluctuation model, so the error was invisible in a total dose
//   and lived entirely in the width of the straggling.
//
// What the model actually is, in three parts:
//
//   1. Above E > 10 MeV * charge * (mass/m_p) the ion is fast enough that its charge state no
//      longer fluctuates, and it hands the whole problem to G4UniversalFluctuation. For an
//      alpha that boundary is 79.45 MeV.
//   2. Below it, a *variance* is computed in closed form - Bohr's, times two empirical
//      corrections - and a distribution with that variance is sampled. There is no Glandz
//      branch and no explicit excitation/ionisation split.
//   3. The two corrections are the reason the model exists. A slow ion picks up and loses
//      electrons as it goes, so its charge is itself a fluctuating quantity, and that adds
//      variance on top of the Bohr term. Q. Yang, D.J. O'Connor and Z. Wang, NIM B61 (1991)
//      149-155 fit that addition; H. Geissel et al., NIM B195 (2002) 3 supply the
//      relativistic/Fermi-gas factor that multiplies the Bohr term itself.
//
// The effective charge. G4IonFluctuations carries `chargeSquare` (the bare PDG charge squared)
// and `effChargeSquare`, and `Factor` returns `s1*effChargeSquare/chargeSquare + s2`. The two
// are equal unless SetParticleAndCharge has been called, and that happens only from
// G4VEnergyLossProcess::AlongStepGPIL under `if(isIon)` - where `isIon` comes from
// G4EmTableUtil::CheckIon, which excludes deuteron, triton, alpha+ and alpha by name. So for
// an alpha the two are equal and the ratio is exactly one; for a generic ion they are not.
// Both are taken as arguments here rather than assumed, so this function is already right for
// ions when ions are transported.
#pragma once
#include <cmath>

#include "core/particle.cuh"
#include "data/g4pow.hh"
#include "data/materials.cuh"
#include "data/yang_fluctuation.hh"
#include "physics/em/fluctuation.cuh"

namespace g4gpu::em {

/// 50 keV / m_p, the squared beta at which an ion's velocity equals the Bohr velocity.
template <typename real_t> __host__ __device__ constexpr real_t kBohrBeta2() {
  return real_t(0.050) / units::proton_mass_c2<real_t>();
}

/// 10 MeV / m_p. The Vavilov threshold is this times charge times mass.
template <typename real_t> __host__ __device__ constexpr real_t kIonFlucParameter() {
  return real_t(10) / units::proton_mass_c2<real_t>();
}

/// G4IonFluctuations::minLoss, 0.001 eV. Note this is *not* G4UniversalFluctuation's 10 eV.
template <typename real_t> __host__ __device__ constexpr real_t kIonFlucMinLoss() {
  return real_t(1e-9);
}

/// beta^2 the way G4DynamicParticle::ComputeBeta computes it, including its shortcut to
/// exactly 1 above 1000 rest masses.
///
/// Written once and shared, because G4IonFluctuations uses the same beta2 in Dispersion and in
/// the large-fractional-loss widening, and `kinetic*(kinetic+2m)/E^2` - algebraically the same
/// thing - is not bit-identical to `(sqrt(T(T+2))/(T+1))^2`. Using both would put a 1e-16
/// wobble between two numbers Geant4 keeps equal.
template <typename real_t>
__host__ __device__ inline real_t ion_beta2(real_t kinetic, real_t mass) {
  if (mass <= real_t(0) || kinetic >= real_t(1000) * mass) { return real_t(1); }
  const real_t t = kinetic / mass;
  const real_t beta = sqrt(t * (t + real_t(2))) / (t + real_t(1));
  return beta * beta;
}

/// G4IonFluctuations::RelativisticFactor - H. Geissel et al., NIM B195 (2002) 3.
///
/// A Fermi-gas correction to the Bohr variance: the target electrons are not at rest, so the
/// maximum energy transfer in a close collision is smeared by their momentum distribution.
template <typename real_t>
__host__ __device__ inline real_t ion_relativistic_factor(const data::Material<real_t>& m,
                                                          real_t z_eff, real_t beta2) {
  const real_t eF = m.fermi_energy;
  const real_t I = m.mean_excitation;
  const real_t me = units::electron_mass_c2<real_t>();
  const real_t bF2 = real_t(2) * eF / me;
  real_t f = real_t(0.4) * (real_t(1) - beta2) / ((real_t(1) - real_t(0.5) * beta2) * z_eff);
  if (beta2 > bF2) {
    f *= log(real_t(2) * me * beta2 / I) * bF2 / beta2;
  } else {
    f *= log(real_t(4) * eF / I);
  }
  return real_t(1) + f;
}

/// G4IonFluctuations::Factor - the multiplier on the Bohr variance.
///
/// @param kinetic  MeV
/// @param beta2    must be the same beta^2 the dispersion was computed with
/// @param charge   bare charge in units of e, positive
/// @param charge_square      that charge squared
/// @param eff_charge_square  the effective charge squared; equal to charge_square for an alpha
template <typename real_t>
__host__ __device__ inline real_t ion_fluctuation_factor(const data::Material<real_t>& m,
                                                         real_t z_eff, real_t kinetic,
                                                         real_t mass, real_t beta2,
                                                         real_t charge, real_t charge_square,
                                                         real_t eff_charge_square) {
  // Reduced energy, MeV per amu.
  real_t energy = kinetic * units::amu_c2<real_t>() / mass;

  real_t s1 = ion_relativistic_factor<real_t>(m, z_eff, beta2);

  // Yang's tabulated charge-state straggling, used only while the ion is slow enough for its
  // charge state to be in play at all.
  if (beta2 < real_t(3) * kBohrBeta2<real_t>() * z_eff) {
    int iz = static_cast<int>(z_eff + real_t(0.5)) - data::kYangZMin;
    if (iz < 0) { iz = 0; }
    if (iz > data::kYangRows - 1) { iz = data::kYangRows - 1; }
    const double* a = data::yang_coefficients() + 4 * iz;
    const real_t ss = real_t(1)
                      + static_cast<real_t>(a[0])
                            * data::g4pow_pow_a<real_t>(energy, static_cast<real_t>(a[1]))
                      + static_cast<real_t>(a[2])
                            * data::g4pow_pow_a<real_t>(energy, static_cast<real_t>(a[3]));
    // Geant4's two guards: below slim the fit has left its validity range and is replaced by
    // its own floor, and above it the correction is never allowed to reduce the variance.
    constexpr real_t kSlim = real_t(0.001);
    if (ss < kSlim) {
      s1 = real_t(1) / kSlim;
    } else if (s1 * ss < real_t(1)) {
      s1 = real_t(1) / ss;
    }
  }

  // Yang's second term. The parameter row depends on what is moving through what:
  //   0 protons/hadrons in a gas      2 ions in an atomic (single-element) gas
  //   1 protons/hadrons in a solid    3 ions in a molecular gas
  //                                   4 ions in a solid
  constexpr real_t b[5][4] = {
      {real_t(0.1014), real_t(0.3700), real_t(0.9642), real_t(3.987)},
      {real_t(0.1955), real_t(0.6941), real_t(2.522), real_t(1.040)},
      {real_t(0.05058), real_t(0.08975), real_t(0.1419), real_t(10.80)},
      {real_t(0.05009), real_t(0.08660), real_t(0.2751), real_t(3.787)},
      {real_t(0.01273), real_t(0.03458), real_t(0.3951), real_t(3.812)},
  };
  const bool is_gas = data::material_is_gas<real_t>(m);
  int i = 0;
  real_t factor = real_t(1);
  if (charge < real_t(1.5)) {  // proton or other singly-charged hadron
    if (!is_gas) { i = 1; }
  } else {                     // ion
    factor = charge * data::g4pow_a13<real_t>(charge / z_eff);
    if (is_gas) {
      energy /= (charge * sqrt(charge));
      i = (m.n_elements == 1) ? 2 : 3;
    } else {
      energy /= (charge * sqrt(charge * z_eff));
      i = 4;
    }
  }

  real_t x = b[i][2];
  real_t y = energy * b[i][3];
  if (y <= real_t(0.2)) {
    x *= (y * (real_t(1) - real_t(0.5) * y));
  } else {
    x *= (real_t(1) - data::g4pow_exp_a<real_t>(-y));
  }
  y = energy - b[i][1];
  const real_t s2 = factor * x * b[i][0] / (y * y + x * x);

  return s1 * eff_charge_square / charge_square + s2;
}

/// G4IonFluctuations::Dispersion - the variance of the energy loss over one step, MeV^2.
///
/// Deterministic, which makes it the part of this model that can be compared against Geant4
/// exactly rather than statistically. tests/test_ion_fluctuation.cu does that.
///
/// @param tcut   MeV, the delta-ray production threshold
/// @param tmax   MeV, the maximum energy transferable to one electron
/// @param length mm
template <typename real_t>
__host__ __device__ inline real_t ion_dispersion(const data::Material<real_t>& m,
                                                 const ParticleDef<real_t>& pd, real_t kinetic,
                                                 real_t tcut, real_t tmax, real_t length,
                                                 real_t eff_charge_square) {
  const real_t mass = pd.mass;
  const real_t charge = fabs(pd.charge);
  const real_t charge_square = charge * charge;

  const real_t beta2 = ion_beta2<real_t>(kinetic, mass);

  real_t siga = (tmax / beta2 - real_t(0.5) * tcut) * units::twopi_mc2_rcl2<real_t>() * length
                * m.electron_density * eff_charge_square;

  const real_t fac = ion_fluctuation_factor<real_t>(m, m.z_eff, kinetic, mass, beta2, charge,
                                                    charge_square, eff_charge_square);

  // The cut correction. Only the close collisions above tcut are affected by the ion's charge
  // state, so the correction is diluted by the fraction of the variance they carry.
  const real_t me = units::electron_mass_c2<real_t>();
  const real_t fac_cut =
      real_t(1) + (fac - real_t(1)) * real_t(2) * me * beta2 / (tmax * (real_t(1) - beta2));
  if (fac_cut > real_t(0.01) && fac > real_t(0.01)) { siga *= fac_cut; }
  return siga;
}

/// The energy above which G4IonFluctuations stops being itself and defers to
/// G4UniversalFluctuation: `parameter * charge * particleMass` with parameter = 10 MeV / m_p.
///
/// 79.45 MeV for an alpha. Named rather than buried in the sampler because the test measures
/// it directly: a wrong factor here - the proton mass in place of the alpha's, say - moves the
/// handover from 79.45 MeV to 20 MeV while leaving every dispersion below it correct, so it
/// needs a check of its own rather than one that only samples either side of it.
template <typename real_t>
__host__ __device__ inline real_t ion_vavilov_threshold(const ParticleDef<real_t>& pd) {
  return kIonFlucParameter<real_t>() * fabs(pd.charge) * pd.mass;
}

/// G4IonFluctuations::SampleFluctuations.
///
/// @param eff_charge_square  effective charge squared. For an alpha this is the bare charge
///                           squared (4) - see the file comment on why nothing rescales it.
template <typename real_t, typename Rng>
__host__ __device__ inline real_t sample_ion_fluctuation(
    const data::Material<real_t>& m, const ParticleDef<real_t>& pd, real_t kinetic, real_t tcut,
    real_t tmax, real_t length, real_t mean_loss, real_t eff_charge_square, Rng& rng) {
  if (mean_loss <= kIonFlucMinLoss<real_t>()) { return mean_loss; }
  if (length <= real_t(0) || pd.mass <= real_t(0)) { return mean_loss; }

  // Fast enough that the ion's charge state is fixed: Vavilov, through the Universal model.
  // Note what is handed over is `charge*charge` and not the effective charge -
  // G4UniversalFluctuation's own chargeSquare comes from InitialiseMe and the PDG charge, and
  // SetParticleAndCharge, which would change it, is not called for an alpha.
  if (kinetic > ion_vavilov_threshold<real_t>(pd)) {
    const real_t q = fabs(pd.charge);
    return sample_fluctuation<real_t>(m, pd, kinetic, tcut, tmax, length, mean_loss, q * q, rng);
  }

  real_t siga = ion_dispersion<real_t>(m, pd, kinetic, tcut, tmax, length, eff_charge_square);

  // Losing a large fraction of the kinetic energy over one step means beta changes across the
  // step, so the variance computed at the pre-step energy understates it. Geant4 widens it by
  // a ratio built from the post-step beta, floored so a particle stopping inside the step does
  // not produce an unbounded factor.
  constexpr real_t kMinFraction = real_t(0.2);
  constexpr real_t kXmin = real_t(0.2);
  const real_t beta2 = ion_beta2<real_t>(kinetic, pd.mass);
  if (mean_loss > kMinFraction * kinetic) {
    const real_t gam = (kinetic - mean_loss) / pd.mass + real_t(1);
    real_t b2 = real_t(1) - real_t(1) / (gam * gam);
    if (b2 < kXmin * beta2) { b2 = kXmin * beta2; }
    const real_t x = b2 / beta2;
    const real_t x3 = real_t(1) / (x * x * x);
    siga *= real_t(0.25) * (real_t(1) + x)
            * (x3 + (real_t(1) / b2 - real_t(0.5)) / (real_t(1) / beta2 - real_t(0.5)));
  }
  siga = sqrt(siga);
  if (!(siga > real_t(0))) { return mean_loss; }

  const real_t sn = mean_loss / siga;
  const real_t two_mean = mean_loss + mean_loss;

  // Three regimes in the number of collisions per step, thick to thin.
  if (sn >= real_t(2)) {
    // Gaussian, truncated to (0, 2*meanLoss) by rejection - which is what keeps the sampled
    // mean equal to the mean handed in, since the truncation is symmetric about it.
    real_t loss;
    int guard = 0;
    do {
      loss = g4_gauss<real_t>(mean_loss, siga, rng);
      if (++guard > 1000) { return mean_loss; }
    } while (loss < real_t(0) || loss > two_mean);
    return loss;
  }
  if (sn > real_t(0.1)) {
    // Gamma with shape sn^2, scaled to have the right mean.
    const real_t neff = sn * sn;
    return mean_loss * g4_gamma<real_t>(neff, rng) / neff;
  }
  // So few collisions that nothing is known but the mean: uniform on (0, 2*meanLoss).
  return two_mean * rng.uniform();
}

}  // namespace g4gpu::em
