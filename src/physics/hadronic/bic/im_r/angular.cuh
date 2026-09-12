// The three angular distributions the binary cascade's two-body channels sample from.
//
// Transcribed from G4AngularDistributionNP.cc, G4AngularDistributionPP.cc and
// G4AngularDistribution.cc (im_r_matrix, 11.1.1), with their data in `imr_tables.hh`.
//
//   * **NP and PP are tables**, one for np elastic and one for pp/nn elastic: 180 one-degree
//     bins of cumulative probability in the CM scattering angle, at 39 and 40 lab kinetic
//     energies, from R.A. Arndt's 1998 partial-wave analysis. They are sampled by inverting the
//     cumulative, which Geant4 does with a bisection whose comparand is itself interpolated in
//     energy on every step - so the interpolation is in the CUMULATIVE, bin by bin, and not in
//     the angle.
//   * **G4AngularDistribution is a formula**, a one-boson-exchange differential cross section
//     with pi, sigma and omega exchange and the pi-omega interference, normalised to its own
//     value at t = tMax and inverted by twelve steps of bisection plus one uniform inside the
//     last interval. `G4VScatteringCollision` constructs it with `symmetrize = true`, which is
//     the only way any BIC channel builds one, so the symmetric branch is the live one.
//
// ## Three things about the table sampling that decide the answer
//
// **The tables are `G4float`.** `sig`, `elab`, `pcm`, `dsigmax` and `sigtot` are all declared
// `const G4float`; `CosTheta` promotes each element to `G4double` as it reads it and does every
// subtraction and division in double. So the numbers being interpolated between are the
// float-rounded ones. `imr_tables.hh` stores them as `float` for that reason and this file
// promotes exactly where Geant4 promotes - an implicit `float -> double` at each read.
//
// **The angle is the bin INDEX interpolated, not the angle.** The last three lines are
// `kint = (sample - sigint1)/(sigint2 - sigint1) + ke1` and `theta = (0.5 + kint)*pi/180`, so
// the returned angle is a linear function of the sampled uniform between two integer bin
// centres. It can leave [0, 180): a `sample` below `sig[..][0]` makes `kint` negative and
// `theta` negative, and `cos` of a negative angle is the same as of its absolute value, so the
// forward hemisphere absorbs it silently. Reproduced as written.
//
// **`ek` is built from the FIRST track's mass twice.** `ek = ((S - m1^2 - m2^2)/(2 m1) - m1)`,
// which is the lab kinetic energy of particle 1 on a particle-2 target only when m1 == m2 - the
// nucleon-nucleon case these two tables are for. `G4VElasticCollision` passes
// `trk1.GetActualMass()` and `trk2.GetActualMass()`, which for a cascade nucleon are off the
// mass shell by the nuclear potential (docs/PORTED.md 2.1.10, the off-shell note), so m1 != m2
// even for pp and the formula is a small approximation in Geant4's own hands. Not corrected.
//
// ## REFUSED, by name
//
//   * `G4AngularDistributionNP::pcm`, `dsigmax` and `sigtot`, and the PP equivalents. They are
//     appended to the data file and read by nothing in 11.1.1 - `CosTheta` uses `sig` and `elab`
//     only. Not extracted, so that a reader does not think they feed something.
//   * `G4AngularDistributionPP::NENERGYC`, the 22-energy Coulomb-suppressed table the enum still
//     declares. There is no second `sig` array in the data file; the constant is left over from
//     the pre-2010 shape whose commented-out `enum { NENERGY=22, ... }` sits above the live one.
//     `tools/extract_bic_imr.pl` strips comments before reading the enum for exactly that reason.
//   * `G4AngularDistribution`'s **asymmetric branch** (`sym = false`). Nothing in the binary
//     cascade constructs one: `G4VScatteringCollision`'s constructor is the only `new
//     G4AngularDistribution(...)` any BIC channel reaches and it passes `true`. The branch is
//     two lines and is transcribed, because it is the `else` of the one that runs and leaving it
//     out would make the symmetric branch look like the whole function.
#ifndef G4GPU_BIC_IMR_ANGULAR_CUH
#define G4GPU_BIC_IMR_ANGULAR_CUH

#include <cmath>

#include "core/units.cuh"
#include "physics/hadronic/bic/im_r/imr_tables.hh"

namespace g4gpu::bic::imr {

namespace u = g4gpu::units;

/// What an angular sample could not do.
struct AngularRefusal {
  /// The bisection ran out of its budget - `2*NENERGY + 1` energy steps or `2*NANGLE + 1` angle
  /// steps. Geant4 raises a `FatalException` here; it cannot happen for a monotone table with
  /// the loop's halving, and `tools/extract_bic_imr.pl` asserts the monotonicity, so this is a
  /// guard on the guard rather than a live path.
  bool bisection_budget = false;
  /// `G4AngularDistribution::CosTheta`'s `cos(theta)` outside [-1, 1], which is also a
  /// `G4HadronicException` there.
  bool cos_theta_range = false;
};

/// `G4AngularDistributionNP::CosTheta` and `G4AngularDistributionPP::CosTheta`, which are the
/// same 60 lines over two different tables. Written once with the table passed in, because a
/// duplicated transcription is a duplicated place to get it wrong; the two Geant4 files are
/// character-for-character identical apart from the `#include` and the exception ids.
///
/// `s` is the Mandelstam s in MeV^2; `m1`, `m2` are the two actual masses in MeV.
template <typename Rng>
__host__ __device__ inline double angular_table_cos_theta(const float* sig, const float* elab,
                                                          int n_energy, double s, double m1,
                                                          double m2, Rng& rng,
                                                          AngularRefusal& ref) {
  constexpr int kNAngle = kAngularAngles;  // 180
  // `ek = ((S - sqr(m_1) - sqr(m_2))/(2*m_1) - m_1)/GeV`, the lab kinetic energy in GeV.
  const double ek = ((s - m1 * m1 - m2 * m2) / (2.0 * m1) - m1) / u::GeV<double>();

  int je1 = 0;
  int je2 = n_energy - 1;
  int iterations_left = 2 * n_energy + 1;
  do {
    const int mid_bin = (je1 + je2) / 2;
    if (ek < static_cast<double>(elab[mid_bin])) {
      je2 = mid_bin;
    } else {
      je1 = mid_bin;
    }
  } while ((je2 - je1) > 1 && --iterations_left > 0);
  if (iterations_left <= 0) { ref.bisection_budget = true; }

  const double delab = static_cast<double>(elab[je2]) - static_cast<double>(elab[je1]);
  const double sample = rng.uniform();
  int ke1 = 0;
  int ke2 = kNAngle - 1;
  // The cumulative at angle bin 0, linearly interpolated in energy between the two energy rows.
  // `rc` is the slope in energy and `b` the intercept, rebuilt from scratch on every angle step.
  double dsig = static_cast<double>(sig[je2 * kNAngle + 0]) -
                static_cast<double>(sig[je1 * kNAngle + 0]);
  double rc = dsig / delab;
  double b = static_cast<double>(sig[je1 * kNAngle + 0]) - rc * static_cast<double>(elab[je1]);
  double sigint1 = rc * ek + b;
  double sigint2 = 0.0;

  iterations_left = 2 * kNAngle + 1;
  do {
    const int mid_bin = (ke1 + ke2) / 2;
    dsig = static_cast<double>(sig[je2 * kNAngle + mid_bin]) -
           static_cast<double>(sig[je1 * kNAngle + mid_bin]);
    rc = dsig / delab;
    b = static_cast<double>(sig[je1 * kNAngle + mid_bin]) - rc * static_cast<double>(elab[je1]);
    const double sigint = rc * ek + b;
    if (sample < sigint) {
      ke2 = mid_bin;
      sigint2 = sigint;
    } else {
      ke1 = mid_bin;
      sigint1 = sigint;
    }
  } while ((ke2 - ke1) > 1 && --iterations_left > 0);
  if (iterations_left <= 0) { ref.bisection_budget = true; }

  // `sigint2` is still 0 if the bisection never took the upper branch, i.e. the sample is above
  // every interpolated cumulative it looked at. Then `dsig` is negative, `kint` runs past ke1
  // the wrong way, and the angle comes out large. That is Geant4's arithmetic; the alternative
  // would be a clamp that changes the distribution's tail.
  dsig = sigint2 - sigint1;
  rc = 1.0 / dsig;
  b = static_cast<double>(ke1) - rc * sigint1;
  const double kint = rc * sample + b;
  const double theta = (0.5 + kint) * u::pi<double>() / 180.0;
  return std::cos(theta);
}

/// G4AngularDistributionNP - np elastic.
template <typename Rng>
__host__ __device__ inline double angular_np_cos_theta(double s, double m1, double m2, Rng& rng,
                                                       AngularRefusal& ref) {
  return angular_table_cos_theta(angular_np_sig(), angular_np_elab(), kAngularNpEnergies, s, m1,
                                 m2, rng, ref);
}

/// G4AngularDistributionPP - pp and nn elastic.
template <typename Rng>
__host__ __device__ inline double angular_pp_cos_theta(double s, double m1, double m2, Rng& rng,
                                                       AngularRefusal& ref) {
  return angular_table_cos_theta(angular_pp_sig(), angular_pp_elab(), kAngularPpEnergies, s, m1,
                                 m2, rng, ref);
}

/// `G4VAngularDistribution::Phi` as both table classes override it: `twopi * G4UniformRand()`.
template <typename Rng>
__host__ __device__ inline double angular_phi(Rng& rng) {
  return u::twopi<double>() * rng.uniform();
}

// =============================================================================================
// G4AngularDistribution - the one-boson-exchange formula.
// =============================================================================================

/// The constants `G4AngularDistribution`'s constructor derives, every one of them, in the order
/// it derives them. They are a function of nine numbers - three masses, three cut-off masses and
/// three couplings - and the derivation is forty lines of products that a reader cannot check by
/// eye, so it is transcribed line for line rather than simplified. GeV units throughout: the
/// class works in GeV internally and `DifferentialCrossSection` converts its arguments on entry.
struct AngularObeConstants {
  double m42 = 0.0;
  double mPion2 = 0.0, cmPion2 = 0.0;
  double cPion_3 = 0.0, cPion_2 = 0.0, cPion_1 = 0.0, cPion_m = 0.0, cPion_L = 0.0, cPion_0 = 0.0;
  double mSigma2 = 0.0, cmSigma2 = 0.0, cmSigma4 = 0.0, cmSigma6 = 0.0;
  double dSigma1 = 0.0, dSigma2 = 0.0;
  double cSigma_3 = 0.0, cSigma_2 = 0.0, cSigma_1 = 0.0, cSigma_m = 0.0, cSigma_L = 0.0,
         cSigma_0 = 0.0;
  double mOmega2 = 0.0, cmOmega2 = 0.0, cmOmega4 = 0.0, cmOmega6 = 0.0;
  double dOmega1 = 0.0, dOmega2 = 0.0, sOmega1 = 0.0;
  double cOmega_3 = 0.0, cOmega_2 = 0.0, cOmega_1 = 0.0, cOmega_m = 0.0, cOmega_L = 0.0;
  double cMix_o1 = 0.0, cMix_s1 = 0.0, cMix_Omega = 0.0, cMix_sm = 0.0;
  double cMix_oLc = 0.0, cMix_oLs = 0.0, cMix_sLc = 0.0, cMix_sLs = 0.0;
};

/// G4AngularDistribution::G4AngularDistribution, the whole constructor body.
__host__ __device__ inline AngularObeConstants angular_obe_constants() {
  AngularObeConstants k;
  const double mSigma = 0.55, cmSigma = 1.20, gSigma = 9.4;
  const double mOmega = 0.783, cmOmega = 0.808, gOmega = 10.95;
  const double mPion = 0.138, cmPion = 0.51, gPion = 7.27;
  const double mNucleon = 0.938;
  k.m42 = 4.0 * mNucleon * mNucleon;

  k.mPion2 = mPion * mPion;
  k.cmPion2 = cmPion * cmPion;
  const double dPion1 = k.cmPion2 - k.mPion2;
  const double dPion2 = dPion1 * dPion1;
  const double cm6gp =
      1.5 * (k.cmPion2 * k.cmPion2 * k.cmPion2) * (gPion * gPion * gPion * gPion) * k.m42 *
      k.m42 / dPion2;
  k.cPion_3 = -(cm6gp / 3.0);
  k.cPion_2 = -(cm6gp * k.mPion2 / dPion1);
  k.cPion_1 = -(cm6gp * k.mPion2 * (2.0 * k.cmPion2 + k.mPion2) / dPion2);
  k.cPion_m = -(cm6gp * k.cmPion2 * k.mPion2 / dPion2);
  k.cPion_L = -(cm6gp * 2.0 * k.cmPion2 * k.mPion2 * (k.cmPion2 + k.mPion2) / dPion2 / dPion1);
  k.cPion_0 = -(k.cPion_3 + k.cPion_2 + k.cPion_1 + k.cPion_m);

  const double gSigmaSq = gSigma * gSigma;
  k.mSigma2 = mSigma * mSigma;
  k.cmSigma2 = cmSigma * cmSigma;
  k.cmSigma4 = k.cmSigma2 * k.cmSigma2;
  k.cmSigma6 = k.cmSigma2 * k.cmSigma4;
  k.dSigma1 = k.m42 - k.cmSigma2;
  k.dSigma2 = k.m42 - k.mSigma2;
  const double dSigma3 = k.cmSigma2 - k.mSigma2;
  const double dSigma1Sq = k.dSigma1 * k.dSigma1;
  const double dSigma2Sq = k.dSigma2 * k.dSigma2;
  const double dSigma3Sq = dSigma3 * dSigma3;
  const double cm2gs = 0.5 * k.cmSigma2 * gSigmaSq * gSigmaSq / dSigma3Sq;
  k.cSigma_3 = -(cm2gs * dSigma1Sq / 3.0);
  k.cSigma_2 = -(cm2gs * k.cmSigma2 * k.dSigma1 * k.dSigma2 / dSigma3);
  k.cSigma_1 = -(cm2gs * k.cmSigma4 * (2.0 * k.dSigma1 + k.dSigma2) * k.dSigma2 / dSigma3Sq);
  k.cSigma_m = -(cm2gs * k.cmSigma6 * dSigma2Sq / k.mSigma2 / dSigma3Sq);
  k.cSigma_L =
      -(cm2gs * k.cmSigma6 * k.dSigma2 * (k.dSigma1 + k.dSigma2) * 2.0 / (dSigma3 * dSigma3Sq));
  k.cSigma_0 = -(k.cSigma_3 + k.cSigma_2 + k.cSigma_1 + k.cSigma_m);

  const double gOmegaSq = gOmega * gOmega;
  k.mOmega2 = mOmega * mOmega;
  k.cmOmega2 = cmOmega * cmOmega;
  k.cmOmega4 = k.cmOmega2 * k.cmOmega2;
  k.cmOmega6 = k.cmOmega2 * k.cmOmega4;
  k.dOmega1 = k.m42 - k.cmOmega2;
  k.dOmega2 = k.m42 - k.mOmega2;
  const double dOmega3 = k.cmOmega2 - k.mOmega2;
  k.sOmega1 = k.cmOmega2 + k.mOmega2;
  const double dOmega3Sq = dOmega3 * dOmega3;
  const double cm2go = 0.5 * k.cmOmega2 * gOmegaSq * gOmegaSq / dOmega3Sq;
  k.cOmega_3 = cm2go / 3.0;
  k.cOmega_2 = -(cm2go * k.cmOmega2 / dOmega3);
  k.cOmega_1 = cm2go * k.cmOmega4 / dOmega3Sq;
  k.cOmega_m = cm2go * k.cmOmega6 / (dOmega3Sq * k.mOmega2);
  k.cOmega_L = -(cm2go * k.cmOmega6 * 4.0 / (dOmega3 * dOmega3Sq));

  const double fac1Tmp = (gSigma * gOmega * k.cmSigma2 * k.cmOmega2);
  const double fac1 = -(fac1Tmp * fac1Tmp * k.m42);
  const double dMix1 = k.cmOmega2 - k.cmSigma2;
  const double dMix2 = k.cmOmega2 - k.mSigma2;
  const double dMix3 = k.cmSigma2 - k.mOmega2;
  const double dMix1Sq = dMix1 * dMix1;
  const double dMix2Sq = dMix2 * dMix2;
  const double dMix3Sq = dMix3 * dMix3;
  k.cMix_o1 = fac1 / (k.cmOmega2 * dMix1Sq * dMix2 * dOmega3);
  k.cMix_s1 = fac1 / (k.cmSigma2 * dMix1Sq * dMix3 * dSigma3);
  k.cMix_Omega = fac1 / (dOmega3Sq * dMix3Sq * (k.mOmega2 - k.mSigma2));
  k.cMix_sm = fac1 / (dSigma3Sq * dMix2Sq * (k.mSigma2 - k.mOmega2));
  const double fac2 = (-fac1) / (dMix1 * dMix1Sq * dOmega3Sq * dMix2Sq);
  const double fac3 = (-fac1) / (dMix1 * dMix1Sq * dSigma3Sq * dMix3Sq);

  k.cMix_oLc = fac2 * (3.0 * k.cmOmega2 * k.cmOmega4 - k.cmOmega4 * k.cmSigma2 -
                       2.0 * k.cmOmega4 * k.mOmega2 - 2.0 * k.cmOmega4 * k.mSigma2 +
                       k.cmOmega2 * k.mOmega2 * k.mSigma2 + k.cmSigma2 * k.mOmega2 * k.mSigma2 -
                       4.0 * k.cmOmega4 * k.m42 + 2.0 * k.cmOmega2 * k.cmSigma2 * k.m42 +
                       3.0 * k.cmOmega2 * k.mOmega2 * k.m42 - k.cmSigma2 * k.mOmega2 * k.m42 +
                       3.0 * k.cmOmega2 * k.mSigma2 * k.m42 - k.cmSigma2 * k.mSigma2 * k.m42 -
                       2.0 * k.mOmega2 * k.mSigma2 * k.m42);
  k.cMix_oLs = fac2 * (8.0 * k.cmOmega4 - 4.0 * k.cmOmega2 * k.cmSigma2 -
                       6.0 * k.cmOmega2 * k.mOmega2 + 2.0 * k.cmSigma2 * k.mOmega2 -
                       6.0 * k.cmOmega2 * k.mSigma2 + 2.0 * k.cmSigma2 * k.mSigma2 +
                       4.0 * k.mOmega2 * k.mSigma2);
  k.cMix_sLc = fac3 * (k.cmOmega2 * k.cmSigma4 - 3.0 * k.cmSigma6 +
                       2.0 * k.cmSigma4 * k.mOmega2 + 2.0 * k.cmSigma4 * k.mSigma2 -
                       k.cmOmega2 * k.mOmega2 * k.mSigma2 - k.cmSigma2 * k.mOmega2 * k.mSigma2 -
                       2.0 * k.cmOmega2 * k.cmSigma2 * k.m42 + 4.0 * k.cmSigma4 * k.m42 +
                       k.cmOmega2 * k.mOmega2 * k.m42 - 3.0 * k.cmSigma2 * k.mOmega2 * k.m42 +
                       k.cmOmega2 * k.mSigma2 * k.m42 - 3.0 * k.cmSigma2 * k.mSigma2 * k.m42 +
                       2.0 * k.mOmega2 * k.mSigma2 * k.m42);
  k.cMix_sLs = fac3 * (4.0 * k.cmOmega2 * k.cmSigma2 - 8.0 * k.cmSigma4 -
                       2.0 * k.cmOmega2 * k.mOmega2 + 6.0 * k.cmSigma2 * k.mOmega2 -
                       2.0 * k.cmOmega2 * k.mSigma2 + 6.0 * k.cmSigma2 * k.mSigma2 -
                       4.0 * k.mOmega2 * k.mSigma2);
  return k;
}

/// G4AngularDistribution::Cross - the three exchange terms plus the six interference terms,
/// evaluated at one pair of propagator arguments.
__host__ __device__ inline double angular_obe_cross(
    const AngularObeConstants& k, double tpPion, double tpSigma, double tpOmega, double tmPion,
    double tmSigma, double tmOmega, double bMix_o1, double bMix_s1, double bMix_Omega,
    double bMix_sm, double bMix_oL, double bMix_sL, double bOmega_0, double bOmega_1,
    double bOmega_2, double bOmega_3, double bOmega_m, double bOmega_L) {
  double cross = 0.0;
  cross += ((k.cPion_3 * tpPion + k.cPion_2) * tpPion + k.cPion_1) * tpPion + k.cPion_m / tmPion +
           k.cPion_0 + k.cPion_L * std::log(tpPion * tmPion);
  cross += ((k.cSigma_3 * tpSigma + k.cSigma_2) * tpSigma + k.cSigma_1) * tpSigma +
           k.cSigma_m / tmSigma + k.cSigma_0 + k.cSigma_L * std::log(tpSigma * tmSigma);
  cross += ((bOmega_3 * tpOmega + bOmega_2) * tpOmega + bOmega_1) * tpOmega + bOmega_m / tmOmega +
           bOmega_0 + bOmega_L * std::log(tpOmega * tmOmega) + bMix_o1 * (tpOmega - 1.0) +
           bMix_s1 * (tpSigma - 1.0) + bMix_Omega * std::log(tmOmega) +
           bMix_sm * std::log(tmSigma) + bMix_oL * std::log(tpOmega) +
           bMix_sL * std::log(tpSigma);
  return cross;
}

/// G4AngularDistribution::DifferentialCrossSection - the NORMALISED cumulative, not a cross
/// section: it returns `Cross(t')/Cross(tMax)` in the asymmetric case and
/// `(Cross(t') - Cross(tMax - t'))/(2 Cross(tMax)) + 0.5` in the symmetric one, which is what
/// makes `CosTheta`'s bisection an inversion of a cumulative distribution.
///
/// `sIn` arrives in MeV^2 and `m_1`, `m_2` in MeV; the first three lines convert them and then
/// `S` is redefined as the s of two nucleons at the same relative momentum. Written in that
/// order, reusing the parameter names, because the reassignment of `sIn` before `S` is computed
/// from it is the step a reader has to see.
__host__ __device__ inline double angular_obe_dsigma(const AngularObeConstants& k, bool sym,
                                                     double s_in, double m_1, double m_2,
                                                     double cos_theta) {
  const double gev = u::GeV<double>();
  s_in = s_in / (gev * gev) + k.m42 / 2.0;
  m_1 = m_1 / gev;
  m_2 = m_2 / gev;
  const double S = s_in - (m_1 + m_2) * (m_1 + m_2) + k.m42;
  const double tMax = S - k.m42;
  const double tp = 0.5 * (cos_theta + 1.0) * tMax;
  const double twoS = 2.0 * S;

  const double brak1 = (twoS - k.m42) * (twoS - k.m42);
  const double bOmega_3 = k.cOmega_3 * (-2.0 * k.cmOmega4 - 2.0 * k.cmOmega2 * twoS - brak1);
  const double bOmega_2 =
      k.cOmega_2 * (2.0 * k.cmOmega2 * k.mOmega2 + k.sOmega1 * twoS + brak1);
  const double bOmega_1 =
      k.cOmega_1 * (-4.0 * k.cmOmega2 * k.mOmega2 - 2.0 * k.mOmega2 * k.mOmega2 -
                    2.0 * (k.cmOmega2 + 2 * k.mOmega2) * twoS - 3.0 * brak1);
  const double bOmega_m =
      k.cOmega_m * (-2.0 * k.mOmega2 * k.mOmega2 - 2.0 * k.mOmega2 * twoS - brak1);
  const double bOmega_L =
      k.cOmega_L * (k.sOmega1 * k.mOmega2 + (k.cmOmega2 + 3.0 * k.mOmega2) * S + brak1);
  const double bOmega_0 = -(bOmega_3 + bOmega_2 + bOmega_1 + bOmega_m);

  const double bMix_o1 = k.cMix_o1 * (k.dOmega1 - twoS);
  const double bMix_s1 = k.cMix_s1 * (k.dSigma1 - twoS);
  const double bMix_Omega = k.cMix_Omega * (k.dOmega2 - twoS);
  const double bMix_sm = k.cMix_sm * (k.dSigma2 - twoS);
  const double bMix_oL = k.cMix_oLc + k.cMix_oLs * S;
  const double bMix_sL = k.cMix_sLc + k.cMix_sLs * S;

  double t1_Pion = 1.0 / (1.0 + tMax / k.cmPion2);
  double t2_Pion = 1.0 + tMax / k.mPion2;
  double t1_Sigma = 1.0 / (1.0 + tMax / k.cmSigma2);
  double t2_Sigma = 1.0 + tMax / k.mSigma2;
  double t1_Omega = 1.0 / (1.0 + tMax / k.cmOmega2);
  double t2_Omega = 1.0 + tMax / k.mOmega2;

  double norm = angular_obe_cross(k, t1_Pion, t1_Sigma, t1_Omega, t2_Pion, t2_Sigma, t2_Omega,
                                  bMix_o1, bMix_s1, bMix_Omega, bMix_sm, bMix_oL, bMix_sL,
                                  bOmega_0, bOmega_1, bOmega_2, bOmega_3, bOmega_m, bOmega_L);

  t1_Pion = 1.0 / (1.0 + tp / k.cmPion2);
  t2_Pion = 1.0 + tp / k.mPion2;
  t1_Sigma = 1.0 / (1.0 + tp / k.cmSigma2);
  t2_Sigma = 1.0 + tp / k.mSigma2;
  t1_Omega = 1.0 / (1.0 + tp / k.cmOmega2);
  t2_Omega = 1.0 + tp / k.mOmega2;

  if (sym) {
    norm = 2.0 * norm;
    const double to = tMax - tp;
    const double t3_Pion = 1.0 / (1.0 + to / k.cmPion2);
    const double t4_Pion = 1.0 + to / k.mPion2;
    const double t3_Sigma = 1.0 / (1.0 + to / k.cmSigma2);
    const double t4_Sigma = 1.0 + to / k.mSigma2;
    const double t3_Omega = 1.0 / (1.0 + to / k.cmOmega2);
    const double t4_Omega = 1.0 + to / k.mOmega2;
    return (angular_obe_cross(k, t1_Pion, t1_Sigma, t1_Omega, t2_Pion, t2_Sigma, t2_Omega,
                              bMix_o1, bMix_s1, bMix_Omega, bMix_sm, bMix_oL, bMix_sL, bOmega_0,
                              bOmega_1, bOmega_2, bOmega_3, bOmega_m, bOmega_L) -
            angular_obe_cross(k, t3_Pion, t3_Sigma, t3_Omega, t4_Pion, t4_Sigma, t4_Omega,
                              bMix_o1, bMix_s1, bMix_Omega, bMix_sm, bMix_oL, bMix_sL, bOmega_0,
                              bOmega_1, bOmega_2, bOmega_3, bOmega_m, bOmega_L)) /
               norm +
           0.5;
  }
  return angular_obe_cross(k, t1_Pion, t1_Sigma, t1_Omega, t2_Pion, t2_Sigma, t2_Omega, bMix_o1,
                           bMix_s1, bMix_Omega, bMix_sm, bMix_oL, bMix_sL, bOmega_0, bOmega_1,
                           bOmega_2, bOmega_3, bOmega_m, bOmega_L) /
         norm;
}

/// G4AngularDistribution::CosTheta - twelve halvings of [-1, 1] against one uniform, then a
/// second uniform spread over the last interval.
///
/// Note that the bisection keeps the LOWER half when the cumulative is at or below the sample
/// (`if (dSigma <= random) cosTheta = cosTh`), so it converges to the largest grid point whose
/// cumulative is still below the sample, and the final `+ rand*dCosTheta` makes the result
/// uniform inside a 2/4096 = 4.9e-4 wide cell rather than interpolated. Two uniforms per call,
/// always, whatever the sample.
template <typename Rng>
__host__ __device__ inline double angular_obe_cos_theta(const AngularObeConstants& k, bool sym,
                                                        double s, double m1, double m2, Rng& rng,
                                                        AngularRefusal& ref) {
  const double random = rng.uniform();
  double d_cos_theta = 2.0;
  double cos_theta = -1.0;
  constexpr int kJMax = 12;
  for (int j = 1; j <= kJMax; ++j) {
    d_cos_theta *= 0.5;
    const double cos_th = cos_theta + d_cos_theta;
    if (angular_obe_dsigma(k, sym, s, m1, m2, cos_th) <= random) { cos_theta = cos_th; }
  }
  cos_theta += rng.uniform() * d_cos_theta;
  if (cos_theta > 1.0 || cos_theta < -1.0) { ref.cos_theta_range = true; }
  return cos_theta;
}

}  // namespace g4gpu::bic::imr

#endif
