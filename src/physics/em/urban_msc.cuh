// Urban multiple scattering, transcribed from Geant4 11.1.1 G4UrbanMscModel.
//
// Covers the parts that set the scattering angle:
//   ComputeCrossSectionPerAtom  -> the transport cross section and mean free path
//   ComputeTheta0               -> the width, Highland form with Urban's Z corrections
//   SampleCosineTheta           -> the core-plus-tail angular distribution
//   SimpleScattering            -> the large-angle fallback
//
// NOT transcribed, and these are real gaps rather than judgement calls (docs/RISK.md):
//   ComputeTruePathLengthLimit  -> Geant4's step limitation; the port keeps its own
//                                  fraction-of-range limiter
//   ComputeGeomPathLength / ComputeTrueStepLength -> the true<->geometric path correction
//   SampleDisplacement          -> lateral displacement at the end of a step
//
// So the angular distribution should be right and the path-length detour factor is not
// modelled. Unlike every other process here there is no Geant4 table to diff this against,
// so it is transcription discipline only - no numerical confirmation.
#pragma once
#include <cmath>
#include "core/units.cuh"
#include "core/vec3.cuh"
#include "data/materials.cuh"
#include "data/urban_msc_tables.cuh"

namespace g4gpu::em {

/// Per-material Urban coefficients, derived from Zeff exactly as InitialiseModelCache does.
template <typename real_t>
struct UrbanCoeffs {
  real_t coeffth1, coeffth2;
  real_t coeffc1, coeffc2, coeffc3, coeffc4;
  real_t sqrtZ, Z23;
  real_t stepmina, stepminb;
  real_t doverra, doverrb;
  real_t posa, posb, posc, posd, pose;
};

template <typename real_t>
__host__ __device__ inline UrbanCoeffs<real_t> urban_coeffs(const data::Material<real_t>& m) {
  UrbanCoeffs<real_t> c;
  const real_t Zeff = m.z_eff;
  const real_t lnZ = log(Zeff);
  const real_t w = exp(lnZ / real_t(6));
  const real_t facz = real_t(0.990395) + w * (real_t(-0.168386) + w * real_t(0.093286));
  c.coeffth1 = facz * (real_t(1) - real_t(8.7780e-2) / Zeff);
  c.coeffth2 = facz * (real_t(4.0780e-2) + real_t(1.7315e-4) * Zeff);
  const real_t Z13 = w * w;
  c.coeffc1 = real_t(2.3785) - Z13 * (real_t(4.1981e-1) - Z13 * real_t(6.3100e-2));
  c.coeffc2 = real_t(4.7526e-1) + Z13 * (real_t(1.7694) - Z13 * real_t(3.3885e-1));
  c.coeffc3 = real_t(2.3683e-1) - Z13 * (real_t(1.8111) - Z13 * real_t(3.2774e-1));
  c.coeffc4 = real_t(1.7888e-2) + Z13 * (real_t(1.9659e-2) - Z13 * real_t(2.6664e-3));
  c.sqrtZ = sqrt(Zeff);
  c.Z23 = Z13 * Z13;
  c.stepmina = real_t(27.725) / (real_t(1) + real_t(0.203) * Zeff);
  c.stepminb = real_t(6.152) / (real_t(1) + real_t(0.111) * Zeff);
  c.doverra = real_t(9.6280e-1) - real_t(8.4848e-2) * c.sqrtZ + real_t(4.3769e-3) * Zeff;
  c.doverrb = real_t(1.15) - real_t(9.76e-4) * Zeff;
  c.posa = real_t(0.994) - real_t(4.08e-3) * Zeff;
  c.posb = real_t(7.16) + (real_t(52.6) + real_t(365) / Zeff) / Zeff;
  c.posc = real_t(1.000) - real_t(4.47e-3) * Zeff;
  c.posd = real_t(1.21e-3) * Zeff;
  c.pose = real_t(1) + Zeff * (real_t(1.84035e-4) * Zeff - real_t(1.86427e-2)) + real_t(0.41125);
  return c;
}

/// Transport cross section per atom, mm^2, for a particle of any mass and charge.
///
/// Transcribed from G4UrbanMscModel::ComputeCrossSectionPerAtom, complete rather than the
/// `mass == m_e` path it used to be. Urban is not the electron's model: G4hMultipleScattering
/// defaults to it, and G4EmBuilder::ConstructIonEmPhysics gives alpha, He3, deuteron, triton
/// and GenericIon a fresh G4hMultipleScattering with no model set - so every ion scatters by
/// Urban, and only the light hadrons and muons get WentzelVI (ConstructLightHadrons calls
/// SetEmModel on theirs).
///
/// The heavy-particle path is a change of variable, not a second formula. Urban maps the
/// particle onto the electron with the same velocity and then runs the electron expression:
///
///     TAU = T/mass;  c = mass*TAU*(TAU+2)/(m_e*(TAU+1));  w = c-2
///     tau = (w + sqrt(w*w + 4c))/2;   eKineticEnergy = m_e*tau
///
/// Two consequences that a single-mass transcription hides. The 10 MeV boundary between the
/// tabulated coefficients and the high-energy branch is in *that* energy, so for an alpha it
/// sits near 8 GeV of kinetic energy rather than 10 MeV. And the coefficient table is chosen
/// by the sign of the charge - `charge < 0` takes celectron - so an alpha reads the positron
/// table.
///
/// @param mass    MeV
/// @param charge  in units of the positron charge, signed
template <typename real_t>
__host__ __device__ inline real_t urban_xs_per_atom(real_t Z, real_t kinetic, real_t mass,
                                                    real_t charge) {
  constexpr real_t epsmin = real_t(1e-4), epsmax = real_t(1e10);
  const real_t me = units::electron_mass_c2<real_t>();

  const real_t Z23 = exp(log(Z) * real_t(2) / real_t(3));
  real_t eKin = kinetic;
  if (mass > me) {
    const real_t TAU = kinetic / mass;
    const real_t cc = mass * TAU * (TAU + real_t(2)) / (me * (TAU + real_t(1)));
    const real_t w = cc - real_t(2);
    const real_t tau = real_t(0.5) * (w + sqrt(w * w + real_t(4) * cc));
    eKin = me * tau;
  }
  const real_t eTot = eKin + me;
  const real_t beta2 = eKin * (eTot + me) / (eTot * eTot);
  const real_t bg2 = eKin * (eTot + me) / (me * me);

  // 2 * m_e^2 * a_Bohr^2 / (hbar c)^2.
  //
  // The Bohr radius is derived, not a literal. It was `0.5291772109e-7` here - the CODATA
  // value to ten digits, and wrong, because CLHEP does not use CODATA: it builds
  // Bohr_radius from electron_Compton_length/fine_structure_const, both of which are
  // themselves derived from its own hbarc and elm_coupling. The two differ by 2e-8, and
  // since epsfactor carries the radius squared and the small-eps branch squares epsfactor
  // again, that came out as 7.7e-8 on every particle at once - a uniform offset across eight
  // masses, which is what a shared constant looks like and a physics error does not.
  const real_t bohr = units::bohr_radius<real_t>();
  constexpr real_t hbarc = units::hbarc<real_t>();  // MeV*mm, CLHEP's derived value
  const real_t epsfactor = real_t(2) * me * me * bohr * bohr / (hbarc * hbarc);

  const real_t eps = epsfactor * bg2 / Z23;
  real_t sigma;
  if (eps < epsmin)      { sigma = real_t(2) * eps * eps; }
  else if (eps < epsmax) { sigma = log(real_t(1) + real_t(2) * eps)
                                   - real_t(2) * eps / (real_t(1) + real_t(2) * eps); }
  else                   { sigma = log(real_t(2) * eps) - real_t(1) + real_t(1) / eps; }
  sigma *= charge * charge * Z * Z / (beta2 * bg2);

  const real_t* Zdat = data::urban::Zdat<real_t>();
  const real_t* Tdat = data::urban::Tdat<real_t>();
  int iZ = 14;
  while (iZ >= 0 && Zdat[iZ] >= Z) { --iZ; }
  if (iZ < 0) { iZ = 0; }
  if (iZ > 13) { iZ = 13; }
  const real_t ZZ1 = Zdat[iZ], ZZ2 = Zdat[iZ + 1];
  const real_t ratZ = (Z - ZZ1) * (Z + ZZ1) / ((ZZ2 - ZZ1) * (ZZ2 + ZZ1));

  constexpr real_t Tlim = real_t(10.0);  // MeV
  const real_t re = units::classic_electron_radius<real_t>();
  const real_t sigmafactor = units::twopi<real_t>() * re * re;
  const real_t beta2lim = Tlim * (Tlim + real_t(2) * me) / ((Tlim + me) * (Tlim + me));
  const real_t bg2lim = Tlim * (Tlim + real_t(2) * me) / (me * me);

  if (eKin <= Tlim) {
    int iT = 21;
    while (iT >= 0 && Tdat[iT] >= eKin) { --iT; }
    if (iT < 0) { iT = 0; }
    if (iT > 20) { iT = 20; }
    real_t T = Tdat[iT], E = T + me;
    const real_t b2small = T * (E + me) / (E * E);
    T = Tdat[iT + 1];
    E = T + me;
    const real_t b2big = T * (E + me) / (E * E);
    const real_t ratb2 = (beta2 - b2small) / (b2big - b2small);

    // Geant4 selects on the *sign*: `charge < 0` reads celectron, everything else cpositron.
    // So a proton, an alpha and a positron share a table, which is Geant4's rule as written
    // and not an approximation of one.
    const real_t* tab = (charge < real_t(0)) ? data::urban::celectron<real_t>()
                                             : data::urban::cpositron<real_t>();
    const int NT = data::urban::kNT;
    real_t c1 = tab[iZ * NT + iT];
    real_t c2 = tab[(iZ + 1) * NT + iT];
    const real_t cc1 = c1 + ratZ * (c2 - c1);
    c1 = tab[iZ * NT + iT + 1];
    c2 = tab[(iZ + 1) * NT + iT + 1];
    const real_t cc2 = c1 + ratZ * (c2 - c1);
    sigma *= sigmafactor / (cc1 + ratb2 * (cc2 - cc1));
  } else {
    const real_t barn = units::barn<real_t>();
    const real_t* sig0 = data::urban::sig0<real_t>();
    const real_t* hec = data::urban::hecorr<real_t>();
    const real_t c1 = bg2lim * sig0[iZ] * barn * (real_t(1) + hec[iZ] * (beta2 - beta2lim)) / bg2;
    const real_t c2 =
        bg2lim * sig0[iZ + 1] * barn * (real_t(1) + hec[iZ + 1] * (beta2 - beta2lim)) / bg2;
    if (Z >= ZZ1 && Z <= ZZ2)      { sigma = c1 + ratZ * (c2 - c1); }
    else if (Z < ZZ1)              { sigma = Z * Z * c1 / (ZZ1 * ZZ1); }
    else                           { sigma = Z * Z * c2 / (ZZ2 * ZZ2); }
  }
  sigma *= (real_t(1) + real_t(0.30) / (real_t(1) + sqrt(real_t(1000) * eKin)));
  return sigma;
}

/// The e-/e+ form, for the call sites that only ever have a lepton.
template <typename real_t>
__host__ __device__ inline real_t urban_xs_per_atom(real_t Z, real_t kinetic,
                                                    bool is_positron) {
  return urban_xs_per_atom<real_t>(Z, kinetic, units::electron_mass_c2<real_t>(),
                                   is_positron ? real_t(1) : real_t(-1));
}

/// Transport mean free path, mm.
template <typename real_t>
__host__ __device__ inline real_t urban_lambda(const data::Material<real_t>& m, real_t kinetic,
                                               bool is_positron) {
  real_t inv = real_t(0);
  for (int i = 0; i < m.n_elements; ++i) {
    inv += m.n_atoms[i] * urban_xs_per_atom(m.z[i], kinetic, is_positron);
  }
  return (inv > real_t(0)) ? real_t(1) / inv : real_t(1e30);
}

/// Width of the angular distribution. Transcribed from G4UrbanMscModel::ComputeTheta0,
/// including the positron correction (G4EmParameters defaults MscPositronCorrection to on).
/// Note the correction scales y, so it enters through both the sqrt(y) and the log(y).
template <typename real_t>
__host__ __device__ inline real_t urban_theta0(const data::Material<real_t>& m,
                                               const UrbanCoeffs<real_t>& c, real_t true_step,
                                               real_t kinetic, real_t current_kinetic,
                                               bool is_positron, real_t mass,
                                               real_t charge) {
  real_t invbetacp = (kinetic + mass) / (kinetic * (kinetic + real_t(2) * mass));
  if (current_kinetic != kinetic) {
    invbetacp = sqrt(invbetacp * (current_kinetic + mass)
                     / (current_kinetic * (current_kinetic + real_t(2) * mass)));
  }
  real_t y = true_step / m.radiation_length;
  if (y <= real_t(0)) { return real_t(0); }

  if (is_positron) {
    constexpr real_t xl = real_t(0.6), xh = real_t(0.9), e = real_t(113.0);
    const real_t tau = sqrt(current_kinetic * kinetic) / mass;
    const real_t x = sqrt(tau * (tau + real_t(2)) / ((tau + real_t(1)) * (tau + real_t(1))));
    const real_t a = c.posa, b = c.posb, cc = c.posc, d = c.posd;
    real_t corr;
    if (x < xl) {
      corr = a * (real_t(1) - exp(-b * x));
    } else if (x > xh) {
      corr = cc + d * exp(e * (x - real_t(1)));
    } else {
      const real_t yl = a * (real_t(1) - exp(-b * xl));
      const real_t yh = cc + d * exp(e * (xh - real_t(1)));
      const real_t y0 = (yh - yl) / (xh - xl);
      const real_t y1 = yl - y0 * xl;
      corr = y0 * x + y1;
    }
    y *= corr * c.pose;
    if (y <= real_t(0)) { return real_t(0); }
  }

  constexpr real_t c_highland = real_t(13.6);  // MeV
  real_t theta0 = c_highland * fabs(charge) * sqrt(y) * invbetacp;
  theta0 *= (c.coeffth1 + c.coeffth2 * log(y));
  return theta0;
}

/// The e-/e+ form, for the call sites that only ever have a lepton.
template <typename real_t>
__host__ __device__ inline real_t urban_theta0(const data::Material<real_t>& m,
                                               const UrbanCoeffs<real_t>& c, real_t true_step,
                                               real_t kinetic, real_t current_kinetic,
                                               bool is_positron) {
  return urban_theta0<real_t>(m, c, true_step, kinetic, current_kinetic, is_positron,
                              units::electron_mass_c2<real_t>(), real_t(1));
}

/// Large-angle fallback. Transcribed from G4UrbanMscModel::SimpleScattering.
template <typename real_t, typename Rng>
__host__ __device__ inline real_t urban_simple_scattering(real_t xmeanth, real_t x2meanth,
                                                          Rng& rng) {
  const real_t a = (real_t(2) * xmeanth + real_t(9) * x2meanth - real_t(3))
                   / (real_t(2) * xmeanth - real_t(3) * x2meanth + real_t(1));
  const real_t prob = (a + real_t(2)) * xmeanth / a;
  const real_t r0 = rng.uniform(), r1 = rng.uniform();
  return (r1 < prob) ? real_t(-1) + real_t(2) * exp(log(r0) / (a + real_t(1)))
                     : real_t(-1) + real_t(2) * r0;
}

/// Intermediate quantities, for the consistency test in tests/test_msc.cu.
template <typename real_t>
struct UrbanDebug {
  real_t tau, theta0, xsi, xmeanth, xmean1, xmean2, prob, qprob;
  int branch;  ///< 0 main, 1 isotropic/taubig, 2 SimpleScattering, 3 no-scatter
};

/// cos(theta) for one step. Transcribed from G4UrbanMscModel::SampleCosineTheta.
///
/// @param current_kinetic energy at the start of the step; @p kinetic is the end energy
template <typename real_t, typename Rng>
__host__ __device__ inline real_t urban_sample_cos_theta(const data::Material<real_t>& m,
                                                         const UrbanCoeffs<real_t>& c,
                                                         real_t true_step, real_t kinetic,
                                                         real_t current_kinetic, real_t lambda0,
                                                         bool is_positron, Rng& rng,
                                                         real_t lambda_end = real_t(-1),
                                                         UrbanDebug<real_t>* dbg = nullptr) {
  constexpr real_t taubig = real_t(8.0);
  constexpr real_t tausmall = real_t(1e-16);
  constexpr real_t numlim = real_t(0.01);
  constexpr real_t onethird = real_t(1) / real_t(3);
  constexpr real_t onesixth = real_t(1) / real_t(6);
  constexpr real_t one12th = real_t(1) / real_t(12);

  real_t cth = real_t(1);
  if (lambda0 <= real_t(0)) { return cth; }
  real_t tau = true_step / lambda0;
  if (current_kinetic != kinetic && kinetic > real_t(0)) {
    // Supplied from the tabulated lambda when there is one; evaluated directly otherwise.
    const real_t lambda1 =
        (lambda_end > real_t(0)) ? lambda_end : urban_lambda(m, kinetic, is_positron);
    if (fabs(lambda1 - lambda0) > lambda0 * real_t(0.01) && lambda1 > real_t(0)) {
      tau = true_step * log(lambda0 / lambda1) / (lambda0 - lambda1);
    }
  }
  if (tau <= real_t(0)) { return cth; }
  const real_t lambdaeff = true_step / tau;

  if (tau >= taubig) { return real_t(-1) + real_t(2) * rng.uniform(); }
  if (tau < tausmall) { return cth; }

  real_t xmeanth, x2meanth;
  if (tau < numlim) {
    xmeanth = real_t(1) - tau * (real_t(1) - real_t(0.5) * tau);
    x2meanth = real_t(1) - tau * (real_t(5) - real_t(6.25) * tau) * onethird;
  } else {
    xmeanth = exp(-tau);
    x2meanth = (real_t(1) + real_t(2) * exp(real_t(-2.5) * tau)) * onethird;
  }

  // Every early exit below is one of the mean-preserving fallbacks; only the main branch
  // can miss xmeanth, and it does so by exactly the factor qprob when qprob > 1.
  if (dbg != nullptr) {
    dbg->tau = tau; dbg->xmeanth = xmeanth; dbg->qprob = real_t(1); dbg->branch = 2;
  }

  const real_t relloss = real_t(1) - kinetic / current_kinetic;
  constexpr real_t rellossmax = real_t(0.50);
  if (relloss > rellossmax) { return urban_simple_scattering(xmeanth, x2meanth, rng); }

  const real_t theta0 = urban_theta0(m, c, true_step, kinetic, current_kinetic, is_positron);
  constexpr real_t theta0max = real_t(3.14159265358979323846) * onesixth;
  const real_t theta2 = theta0 * theta0;
  if (theta2 < tausmall) { return cth; }
  if (theta0 > theta0max) { return urban_simple_scattering(xmeanth, x2meanth, rng); }

  real_t x = theta2 * (real_t(1) - theta2 * one12th);
  if (theta2 > numlim) {
    const real_t sth = real_t(2) * sin(real_t(0.5) * theta0);
    x = sth * sth;
  }

  const real_t ltau = log(tau);
  const real_t u = exp(ltau * onesixth);
  const real_t xx = log(lambdaeff / m.radiation_length);
  real_t xsi = c.coeffc1 + u * (c.coeffc2 + c.coeffc3 * u) + c.coeffc4 * xx;
  xsi = fmax(xsi, real_t(1.9));

  real_t cc = xsi;
  if (fabs(cc - real_t(3)) < real_t(0.001))      { cc = real_t(3.001); }
  else if (fabs(cc - real_t(2)) < real_t(0.001)) { cc = real_t(2.001); }
  const real_t c1 = cc - real_t(1);
  const real_t ea = exp(-xsi);
  const real_t eaa = real_t(1) - ea;
  const real_t xmean1 = real_t(1) - (real_t(1) - (real_t(1) + xsi) * ea) * x / eaa;
  const real_t x0 = real_t(1) - xsi * x;
  if (xmean1 <= real_t(0.999) * xmeanth) {
    return urban_simple_scattering(xmeanth, x2meanth, rng);
  }

  const real_t b = real_t(1) + (cc - xsi) * x;
  const real_t b1 = b + real_t(1);
  const real_t bx = cc * x;
  const real_t eb1 = exp(log(b1) * c1);
  const real_t ebx = exp(log(bx) * c1);
  const real_t d = ebx / eb1;
  const real_t xmean2 = (x0 + d - (bx - b1 * d) / (cc - real_t(2))) / (real_t(1) - d);
  const real_t f1x0 = ea / eaa;
  const real_t f2x0 = c1 / (cc * (real_t(1) - d));
  const real_t prob = f2x0 / (f1x0 + f2x0);
  const real_t qprob = xmeanth / (prob * xmean1 + (real_t(1) - prob) * xmean2);

  if (dbg != nullptr) {
    dbg->tau = tau; dbg->theta0 = theta0; dbg->xsi = xsi; dbg->xmeanth = xmeanth;
    dbg->xmean1 = xmean1; dbg->xmean2 = xmean2; dbg->prob = prob; dbg->qprob = qprob;
    dbg->branch = 0;
  }

  const real_t r0 = rng.uniform(), r1 = rng.uniform();
  if (r0 < qprob) {
    if (r1 < prob) {
      cth = real_t(1) + log(ea + rng.uniform() * eaa) * x;
    } else {
      real_t var = (real_t(1) - d) * rng.uniform();
      if (var < numlim * d) {
        var /= (d * c1);
        cth = real_t(-1) + var * (real_t(1) - real_t(0.5) * var * cc)
                               * (real_t(2) + (cc - xsi) * x);
      } else {
        cth = real_t(1) + x * (cc - xsi - cc * exp(-log(var + d) / c1));
      }
    }
  } else {
    cth = real_t(-1) + real_t(2) * r1;
  }
  if (cth < real_t(-1)) { cth = real_t(-1); }
  if (cth > real_t(1)) { cth = real_t(1); }
  return cth;
}

/// Per-material Urban table: transport mean free path on a log energy grid, plus the
/// coefficient cache. Geant4 also tabulates lambda rather than evaluating the parameterised
/// cross section every step (G4VEmModel builds a lambda table at initialisation), so this is
/// the faithful structure as well as the fast one - the alternative is ~30 transcendentals
/// per step, which on a consumer card running FP64 at 1/32 rate dominated everything else.
constexpr int kMscBins = 240;

template <typename real_t>
struct UrbanTable {
  real_t e_min, e_max, log_e_min, inv_dlog;
  UrbanCoeffs<real_t> coeffs[data::kMaxMaterials];
  real_t lambda[data::kMaxMaterials][2][kMscBins];  ///< [material][is_positron][bin]
  int n_materials;

  /// Transport mean free path, mm. Log-log interpolated, as Geant4 interpolates its own.
  __host__ __device__ real_t lambda_at(int mat, bool pos, real_t e) const {
    const int p = pos ? 1 : 0;
    if (e <= e_min) { return lambda[mat][p][0]; }
    if (e >= e_max) { return lambda[mat][p][kMscBins - 1]; }
    const real_t f = (log(e) - log_e_min) * inv_dlog;
    int i = static_cast<int>(f);
    if (i < 0) { i = 0; }
    if (i > kMscBins - 2) { i = kMscBins - 2; }
    const real_t frac = f - real_t(i);
    const real_t a = lambda[mat][p][i], b = lambda[mat][p][i + 1];
    // The mean free path spans several decades over the grid, so interpolate its log.
    return (a > real_t(0) && b > real_t(0)) ? exp(log(a) * (real_t(1) - frac) + log(b) * frac)
                                            : a * (real_t(1) - frac) + b * frac;
  }
};

template <typename real_t>
__host__ inline void build_urban_table(const data::Material<real_t>* mats, int n_materials,
                                       UrbanTable<real_t>& out, real_t e_min = real_t(1e-3),
                                       real_t e_max = real_t(100)) {
  out.e_min = e_min;
  out.e_max = e_max;
  out.log_e_min = log(e_min);
  out.inv_dlog = real_t(kMscBins - 1) / (log(e_max) - log(e_min));
  out.n_materials = n_materials;
  for (int m = 0; m < n_materials; ++m) {
    out.coeffs[m] = urban_coeffs(mats[m]);
    for (int p = 0; p < 2; ++p) {
      for (int i = 0; i < kMscBins; ++i) {
        const real_t e = exp(out.log_e_min + real_t(i) / out.inv_dlog);
        out.lambda[m][p][i] = urban_lambda(mats[m], e, p == 1);
      }
    }
  }
}

// Model constants, from the G4UrbanMscModel constructor and the G4VMscModel /
// G4EmParameters defaults. B1 leaves all of these at their defaults.
template <typename real_t> __host__ __device__ constexpr real_t kFacRange() { return real_t(0.04); }
template <typename real_t> __host__ __device__ constexpr real_t kFacSafety() { return real_t(0.6); }
template <typename real_t> __host__ __device__ constexpr real_t kLambdaLimit() { return real_t(1.0); }
template <typename real_t> __host__ __device__ constexpr real_t kTlimitMinFix() { return real_t(1e-8); }
template <typename real_t> __host__ __device__ constexpr real_t kTlimitMinFix2() { return real_t(1e-6); }
template <typename real_t> __host__ __device__ constexpr real_t kTauLim() { return real_t(1e-6); }
template <typename real_t> __host__ __device__ constexpr real_t kDtrl() { return real_t(0.05); }
template <typename real_t> __host__ __device__ constexpr real_t kTlow() { return real_t(5e-3); }

// G4VEnergyLossProcess continuous-step-limit parameters, from G4EmExtraParameters
// (dRoverRange) and G4VEnergyLossProcess (finalRange). QBBC leaves both at their defaults.
template <typename real_t> __host__ __device__ constexpr real_t kDRoverRange() { return real_t(0.2); }
template <typename real_t> __host__ __device__ constexpr real_t kFinalRange() { return real_t(1.0); }

/// State the true<->geometric conversion carries between the two halves of a step.
template <typename real_t>
struct MscStep {
  real_t lambda0;  ///< transport mfp at the pre-step energy
  real_t z;        ///< geometric length corresponding to the requested true length
  real_t par1, par2, par3;
  real_t range;
};

/// Transcribed from G4UrbanMscModel::ComputeStepmin.
template <typename real_t>
__host__ __device__ inline real_t urban_stepmin(const UrbanCoeffs<real_t>& c, real_t lambda0,
                                                real_t kinetic) {
  const real_t rat = kinetic;  // invmev = 1/MeV, and energies here are already in MeV
  return lambda0 * real_t(1e-3) / (real_t(2e-3) + rat * (c.stepmina + c.stepminb * rat));
}

/// Transcribed from G4UrbanMscModel::ComputeTlimitmin.
template <typename real_t>
__host__ __device__ inline real_t urban_tlimitmin(const UrbanCoeffs<real_t>& c, real_t stepmin,
                                                  real_t kinetic, bool is_positron) {
  real_t x = is_positron ? real_t(0.7) * c.sqrtZ * stepmin : real_t(0.87) * c.Z23 * stepmin;
  if (kinetic < kTlow<real_t>()) {
    x *= real_t(0.5) * (real_t(1) + kinetic / kTlow<real_t>());
  }
  return fmax(x, kTlimitMinFix<real_t>());
}

/// Gaussian deviate, for Randomizetlimit. Box-Muller; only one of the pair is used.
template <typename real_t, typename Rng>
__host__ __device__ inline real_t urban_gauss(real_t mean, real_t sigma, Rng& rng) {
  const real_t u1 = rng.uniform(), u2 = rng.uniform();
  return mean + sigma * sqrt(real_t(-2) * log(u1)) * cos(units::twopi<real_t>() * u2);
}

/// Step limitation, transcribed from the fUseSafety branch of ComputeTruePathLengthLimit.
/// G4EmParameters defaults mscStepLimit to fUseSafety, and G4EmStandardPhysics - the EM
/// constructor QBBC uses - does not override it, so that is the branch B1 takes.
///
/// Geant4 recomputes rangeinit, fr and tlimitmin only on the first step of a track and
/// after a geometry boundary, then holds them. fr and rangeinit are only ever used as a
/// product, so the caller carries that in @p tlimit_base and @p tlimitmin alongside it;
/// passing tlimit_base <= 0 requests a refresh of both.
template <typename real_t, typename Rng>
__host__ __device__ inline real_t urban_step_limit(const UrbanCoeffs<real_t>& c, real_t lambda0,
                                                   real_t kinetic, real_t range, real_t safety,
                                                   bool is_positron, Rng& rng,
                                                   real_t& tlimit_base, real_t& tlimitmin) {
  if (tlimit_base <= real_t(0)) {
    // mass < masslimite (0.6 MeV) holds for e-/e+, so the lambda0 floor and fr boost apply.
    const real_t rangeinit = fmax(range, lambda0);
    real_t fr = kFacRange<real_t>();
    if (lambda0 > kLambdaLimit<real_t>()) {
      fr *= (real_t(0.75) + real_t(0.25) * lambda0 / kLambdaLimit<real_t>());
    }
    tlimit_base = fr * rangeinit;
    const real_t stepmin = urban_stepmin(c, lambda0, kinetic);
    tlimitmin = urban_tlimitmin(c, stepmin, kinetic, is_positron);
  }

  real_t tlimit = (range > safety) ? fmax(tlimit_base, kFacSafety<real_t>() * safety) : range;
  tlimit = fmax(tlimit, tlimitmin);
  if (tlimit >= range) { return range; }
  // Randomizetlimit
  real_t res = tlimitmin;
  if (tlimit > tlimitmin) {
    res = fmax(urban_gauss(tlimit, real_t(0.1) * (tlimit - tlimitmin), rng), tlimitmin);
  }
  return fmin(range, res);
}

/// True -> geometric path length, transcribed from ComputeGeomPathLength. Fills par1..par3
/// so the inverse conversion can undo it.
///
/// @param lambda_at_rfin transport mfp at the energy left after the whole true step; pass a
///                       non-positive value to take the tPathLength == range branch
template <typename real_t>
__host__ __device__ inline real_t urban_geom_path(real_t t_path, real_t lambda0, real_t range,
                                                  real_t lambda_at_rfin, real_t kinetic,
                                                  MscStep<real_t>& st) {
  st.lambda0 = lambda0;
  st.par1 = real_t(-1);
  st.par2 = st.par3 = real_t(0);
  st.range = range;
  t_path = fmin(t_path, range);
  real_t z = t_path;
  if (t_path < kTlimitMinFix2<real_t>()) { st.z = z; return z; }

  const real_t tau = t_path / lambda0;
  if (tau <= real_t(1e-16)) {
    z = fmin(t_path, lambda0);
  } else if (t_path < range * kDtrl<real_t>()) {
    z = (tau < kTauLim<real_t>()) ? t_path * (real_t(1) - real_t(0.5) * tau)
                                  : lambda0 * (real_t(1) - exp(-tau));
  } else if (kinetic < units::electron_mass_c2<real_t>() || t_path >= range
             || lambda_at_rfin <= real_t(0)) {
    // Geant4: currentKinEnergy < mass || tPathLength == currentRange.
    st.par1 = real_t(1) / range;
    st.par2 = range / lambda0;
    st.par3 = real_t(1) + st.par2;
    z = (t_path < range)
            ? (real_t(1) - exp(st.par3 * log(real_t(1) - t_path / range))) / (st.par1 * st.par3)
            : real_t(1) / (st.par1 * st.par3);
  } else {
    st.par1 = (lambda0 - lambda_at_rfin) / (lambda0 * t_path);
    st.par2 = real_t(1) / (st.par1 * lambda0);
    st.par3 = real_t(1) + st.par2;
    z = (real_t(1) - exp(st.par3 * log(lambda_at_rfin / lambda0))) / (st.par1 * st.par3);
  }
  z = fmin(z, lambda0);
  st.z = z;
  return z;
}

/// Geometric -> true path length, transcribed from ComputeTrueStepLength. Call with the
/// geometric distance actually travelled, which is shorter than st.z when a boundary or a
/// discrete interaction cut the step short.
template <typename real_t>
__host__ __device__ inline real_t urban_true_path(real_t geom_step, real_t t_path,
                                                  const MscStep<real_t>& st) {
  if (geom_step == st.z) { return t_path; }
  if (geom_step < kTlimitMinFix2<real_t>()) { return geom_step; }
  real_t tlength = geom_step;
  if (geom_step > st.lambda0 * real_t(1e-16)) {
    if (st.par1 < real_t(0)) {
      const real_t r = real_t(1) - geom_step / st.lambda0;
      tlength = (r > real_t(0)) ? -st.lambda0 * log(r) : t_path;
    } else {
      const real_t par4 = st.par1 * st.par3;
      tlength = (par4 * geom_step < real_t(1))
                    ? (real_t(1) - exp(log(real_t(1) - par4 * geom_step) / st.par3)) / st.par1
                    : st.range;
    }
    if (tlength < geom_step)   { tlength = geom_step; }
    else if (tlength > t_path) { tlength = t_path; }
  }
  return tlength;
}

/// Smallest geometric length Geant4 treats as a real displacement (G4VMultipleScattering).
template <typename real_t> __host__ __device__ constexpr real_t kGeomMin() { return real_t(5e-8); }

template <typename real_t>
struct MscResult {
  Vec3<real_t> dir;
  Vec3<real_t> displacement;  ///< lateral shift, already rotated into the lab frame
};

/// One multiple-scattering interaction, transcribed from
/// G4UrbanMscModel::SampleScattering plus SampleDisplacement (dispAlg96, the default).
///
/// The lateral displacement shares its azimuth with the deflection - Geant4 passes the same
/// phi into SampleDisplacement - so the two are correlated and must be sampled together.
///
/// @param t_path true path length of the step
/// @param z_path geometric length of the step; the displacement scales with sqrt(t^2 - z^2)
template <typename real_t, typename Rng>
__host__ __device__ inline MscResult<real_t> urban_sample_scattering(
    const data::Material<real_t>& m, const UrbanCoeffs<real_t>& c, real_t lambda0,
    const Vec3<real_t>& old_dir, real_t t_path, real_t z_path, real_t kinetic,
    real_t current_kinetic, bool lat_displacement, bool is_positron, Rng& rng,
    real_t lambda_end = real_t(-1)) {
  MscResult<real_t> out{old_dir, Vec3<real_t>{real_t(0), real_t(0), real_t(0)}};
  constexpr real_t tausmall = real_t(1e-16);
  if (t_path <= kTlimitMinFix<real_t>() || t_path < tausmall * lambda0
      || kinetic <= real_t(1e-6)) {
    return out;
  }

  const real_t ct = urban_sample_cos_theta(m, c, t_path, kinetic, current_kinetic, lambda0,
                                           is_positron, rng, lambda_end);
  if (fabs(ct) >= real_t(1)) { return out; }
  const real_t st = sqrt((real_t(1) - ct) * (real_t(1) + ct));
  const real_t phi = units::twopi<real_t>() * rng.uniform();
  out.dir = normalize(rotate_uz(Vec3<real_t>{st * cos(phi), st * sin(phi), ct}, old_dir));

  const real_t tau = t_path / lambda0;
  if (lat_displacement && tau >= tausmall) {
    const real_t rmax2 = (t_path - z_path) * (t_path + z_path);
    if (rmax2 > real_t(0)) {
      const real_t r = real_t(0.73) * sqrt(rmax2);
      constexpr real_t cbeta = real_t(2.160);
      const real_t cbeta1 = real_t(1) - exp(-cbeta * real_t(3.14159265358979323846));
      const real_t psi = -log(real_t(1) - rng.uniform() * cbeta1) / cbeta;
      const real_t Phi = (rng.uniform() < real_t(0.5)) ? phi + psi : phi - psi;
      out.displacement =
          rotate_uz(Vec3<real_t>{r * cos(Phi), r * sin(Phi), real_t(0)}, old_dir);
    }
  }
  return out;
}

/// End-of-step energy the model scatters at, from the top of SampleScattering. For a step
/// short compared with the range Geant4 keeps the pre-step energy rather than looking the
/// residual range up again.
template <typename real_t>
__host__ __device__ inline real_t urban_scatter_energy(real_t current_kinetic, real_t t_path,
                                                       real_t range, real_t e_from_range,
                                                       real_t dedx) {
  if (t_path > range * kDtrl<real_t>()) { return e_from_range; }
  if (t_path > range * real_t(0.01)) { return current_kinetic - t_path * dedx; }
  return current_kinetic;
}

}  // namespace g4gpu::em
