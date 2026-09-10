// Competitive fission: channel 1 of the 68, open only for A >= 65 and Z > 16.
//
// Transcribed from G4CompetitiveFission, G4FissionBarrier, G4FissionProbability,
// G4FissionLevelDensityParameter and G4FissionParameters (11.1.1), with CLHEP::RandGauss's
// polar sampler under the two Gaussians the mass and charge split need.
//
// The chain is short and each link is a different kind of object:
//
//   G4FissionBarrier::FissionBarrier        fission_barrier()
//     Barashenkov's liquid-drop barrier over 1 + sqrt(U/2A). It is the ONLY consumer of
//     G4CameronShellPlusPairingCorrections in the default configuration, which makes it the
//     only way that table can be checked against the oracle at all - see below.
//   G4FissionProbability::EmissionProbability  fission_probability()
//     a closed form in two level-density parameters, the evaporation one for the compound and
//     a Z-scaled one for the saddle point.
//   G4FissionParameters::DefineParameters   FissionParameters::define()
//     the five-Gaussian mass distribution's centroids and widths, and `w`, the weight between
//     its symmetric and asymmetric components. Everything the sampler does is dispatched by
//     `w`, and `w` is set by Z: below 82 it is 1001 (symmetric only), above it is fitted.
//   G4CompetitiveFission::EmittedFragment   fission_emitted_fragment()
//     samples A1 by rejection against that distribution, Z1 from a Gaussian about
//     (A1/A)*Z + DeltaZ, and the kinetic energy from a third Gaussian, then shares the
//     leftover excitation between the two fragments in proportion to their mass numbers.
//
// **The Cameron shell-plus-pairing table becomes observable here.** The previous commit
// recorded that G4CameronShellPlusPairingCorrections cannot be oracle-checked directly: its
// tables are private statics behind a header-inline accessor that this Windows Geant4 does not
// export, so a dump that touches it does not link. G4FissionBarrier::FissionBarrier is public
// and reads that table through `SPtr->GetPairingCorrection(N, Z, res)`, so dumping the barrier
// over a (Z, A) grid checks the table indirectly and completely - every (N, Z) in the grid
// that falls in the table's window contributes its own value to a number the oracle can see.
// tests/test_deex_models.cu does that over 4,963 (Z, A), and the perturbation run recorded in
// the commit message confirms the table is load-bearing there: dropping the subtraction moves
// the barrier by up to a factor of 474.
//
// One thing REFUSED rather than reproduced: G4CompetitiveFission::EmittedFragment throws
// G4HadronicException when 100 trials fail to produce a non-negative fragment excitation. A
// device kernel cannot throw, so the port reports the refusal on the output and leaves the
// fragment untouched. Nothing in the validated grid reaches it.
#ifndef G4GPU_DEEX_FISSION_CUH
#define G4GPU_DEEX_FISSION_CUH

#include <cmath>

#include "core/units.cuh"
#include "data/g4pow.hh"
#include "data/level_data.cuh"
#include "physics/hadronic/deexcitation/corrections.cuh"
#include "physics/hadronic/deexcitation/deex_params.cuh"
#include "physics/hadronic/deexcitation/fragment.cuh"
#include "physics/hadronic/deexcitation/nuclear_masses.cuh"

namespace g4gpu::deex {

namespace u = g4gpu::units;

/// CLHEP::RandGauss::shoot, and its cache.
///
/// The polar (Marsaglia) method produces two standard normals per accepted pair and CLHEP
/// keeps the spare in a static, returning it on the next call - so RandGauss is stateful and
/// the state is global in Geant4. It is carried here on a small object instead, because a
/// device kernel cannot have a mutable global and because a hidden global would make two
/// fission events in flight interfere.
///
/// Note which of the pair is returned: `val = v1*fac` is CACHED and `v2*fac` is returned. The
/// order matters for a stream comparison and is reproduced, even though this port's engine is
/// Philox and Geant4's is HepJamesRandom, so no stream comparison is possible in either case -
/// the distribution is what is compared, and its tails come out of the same rejection.
struct GaussCache {
  bool has_spare = false;
  double spare = 0.0;
};

template <typename Rng>
__host__ __device__ inline double rand_gauss(GaussCache& g, Rng& rng) {
  if (g.has_spare) {
    g.has_spare = false;
    return g.spare;
  }
  double r, v1, v2;
  do {
    v1 = 2.0 * rng.uniform() - 1.0;
    v2 = 2.0 * rng.uniform() - 1.0;
    r = v1 * v1 + v2 * v2;
  } while (r > 1.0);
  const double fac = std::sqrt(-2.0 * std::log(r) / r);
  g.spare = v1 * fac;
  g.has_spare = true;
  return v2 * fac;
}

template <typename Rng>
__host__ __device__ inline double rand_gauss(GaussCache& g, Rng& rng, double mean,
                                             double std_dev) {
  return rand_gauss(g, rng) * std_dev + mean;
}

// ---------------------------------------------------------------------------------------------
// G4FissionBarrier
// ---------------------------------------------------------------------------------------------

/// G4FissionBarrier::BarashenkovFissionBarrier(A, Z).
///
/// The liquid-drop surface and Coulomb coefficients are Barashenkov's, the fissibility
/// parameter x carries the (N - Z)^2 asymmetry through k = 1.7826, and the two branches of
/// BF0 meet at x = 2/3. `d` is the odd-even indicator - 0, 1 or 2 - and D = 1.248 MeV its
/// coefficient; `res` is the Cameron shell-plus-pairing correction, SUBTRACTED.
__host__ __device__ inline double barashenkov_fission_barrier(int A, int Z) {
  const double aSurf = 17.9439 * u::MeV<double>();
  const double aCoul = 0.7053 * u::MeV<double>();
  const double k = 1.7826;
  const int N = A - Z;

  double x = (aCoul / (2.0 * aSurf)) * (static_cast<double>(Z) * Z) / static_cast<double>(A);
  x /= (1.0 - k * static_cast<double>(N - Z) * (N - Z) / static_cast<double>(A * A));

  double BF0 = aSurf * data::g4pow_z13<double>(A) * data::g4pow_z13<double>(A);
  if (x <= 2.0 / 3.0) {
    BF0 *= 0.38 * (0.75 - x);
  } else {
    BF0 *= 0.83 * (1.0 - x) * (1.0 - x) * (1.0 - x);
  }

  const int d = N - 2 * (N / 2) + Z - 2 * (Z / 2);
  double res = 0.0;
  deex::cameron_shell_plus_pairing(N, Z, res);
  const double D = 1.248 * u::MeV<double>();
  return BF0 + D * d - res;
}

/// G4FissionBarrier::FissionBarrier(A, Z, U). Below A = 65 it returns 100 GeV, which is how
/// the channel is closed for a light fragment without a separate flag - though
/// G4CompetitiveFission::GetEmissionProbability tests A >= 65 itself and never gets here.
__host__ __device__ inline double fission_barrier(int A, int Z, double U) {
  const double blimit = 100.0 * u::GeV<double>();
  if (A < 65) { return blimit; }
  return barashenkov_fission_barrier(A, Z) /
         (1.0 + std::sqrt(U / static_cast<double>(2 * A)));
}

// ---------------------------------------------------------------------------------------------
// G4FissionLevelDensityParameter and G4FissionProbability
// ---------------------------------------------------------------------------------------------

/// G4FissionLevelDensityParameter::LevelDensityParameter - the evaporation level density
/// scaled by 1.03 to 1.05 across the actinides, linearly interpolated between Z = 85 and 89.
__host__ __device__ inline double fission_level_density(int A, int Z, double U,
                                                        bool has_levels) {
  double ld = deex::level_density(Z, A, U, has_levels);
  if (Z >= 89) {
    ld *= 1.05;
  } else if (Z <= 85) {
    ld *= 1.03;
  } else {
    ld *= (1.03 + 0.005 * (Z - 85));
  }
  return ld;
}

/// G4FissionProbability::EmissionProbability - the integrated fission width.
///
/// Two excitations, and they are different: `Ucompound` is reduced by the ordinary pairing
/// correction and `Ufission` by the FISSION pairing correction, which is the same table
/// halved. Both must be non-negative or the channel is closed. The parenthesis around
/// `4*pi*afission` is JMQ's 14/02/09 fix; without it the width is 4*pi times too large and
/// then divided by afission again.
__host__ __device__ inline double fission_probability(const Fragment& frag, double max_ke,
                                                      const data::LevelTable& lt) {
  if (max_ke <= 0.0) { return 0.0; }
  const int A = frag.a;
  const int Z = frag.z;
  const double U = frag.excitation;

  const double u_compound = U - deex::level_data_pairing_correction(Z, A);
  const double u_fission = U - deex::fission_pairing_correction(A, Z);
  if (u_compound < 0.0 || u_fission < 0.0) { return 0.0; }

  const bool has_levels = (data::find_manager(lt, Z, A) >= 0);
  const double system_entropy =
      2.0 * std::sqrt(deex::level_density(Z, A, u_compound, has_levels) * u_compound);
  const double afission = fission_level_density(A, Z, u_fission, has_levels);

  const double Cf = 2.0 * std::sqrt(afission * max_ke);
  const double exp1 = (system_entropy <= 160.0) ? std::exp(-system_entropy) : 0.0;
  const double exp2 =
      (system_entropy - Cf <= 160.0) ? std::exp(-system_entropy + Cf) : 0.0;
  return (exp1 + (Cf - 1.0) * exp2) / (4.0 * u::pi<double>() * afission);
}

/// G4CompetitiveFission::GetEmissionProbability. The A >= 65 and Z > 16 window is the "saddle
/// point excitation energy" cut its comment names; below it there is no fission channel at all
/// rather than a small one.
struct FissionState {
  double barrier = 0.0;
  double max_kinetic_energy = 0.0;
  double probability = 0.0;
};

__host__ __device__ inline double fission_emission_probability(FissionState& s,
                                                               const Fragment& frag,
                                                               const data::LevelTable& lt) {
  s.probability = 0.0;
  const int Z = frag.z;
  const int A = frag.a;
  if (A >= 65 && Z > 16) {
    const double ex = frag.excitation - deex::fission_pairing_correction(A, Z);
    if (ex > 0.0) {
      s.barrier = fission_barrier(A, Z, ex);
      s.max_kinetic_energy = ex - s.barrier;
      s.probability = fission_probability(frag, s.max_kinetic_energy, lt);
    }
  }
  return s.probability;
}

// ---------------------------------------------------------------------------------------------
// G4FissionParameters
// ---------------------------------------------------------------------------------------------

/// G4FissionParameters::LocalExp - a Gaussian truncated hard at 8 sigma. It is the same
/// function as G4CompetitiveFission::LocalExp; both are here because both classes define
/// their own and a shared one would hide that.
__host__ __device__ inline double fission_local_exp(double x) {
  return (std::fabs(x) < 8.0) ? std::exp(-0.5 * x * x) : 0.0;
}

/// G4FissionParameters. A1 = 134 and A2 = 141 are fixed centroids in mass number - not
/// fractions of A - which is why the asymmetric mode disappears for a light actinide and why
/// `w` needs the exp(0.3*(227 - A)) boost below A = 227.
struct FissionParameters {
  double A1 = 134.0;
  double A2 = 141.0;
  double A3 = (134.0 + 141.0) * 0.5;
  double As = 0.0;
  double Sigma1 = 0.0;
  double Sigma2 = 0.0;
  double SigmaS = 0.0;
  double w = 0.0;

  /// G4FissionParameters::DefineParameters(A, Z, ExEnergy, FissionBarrier).
  ///
  /// `U` is capped at 200 MeV - "to avoid usage of units", says the comment, but the cap is
  /// physical too: SigmaS and `wa` are exponentials in U and both would run away.
  __host__ __device__ void define(int A, int Z, double ex_energy, double barrier) {
    const double MeV = u::MeV<double>();
    const double U = (ex_energy / MeV < 200.0) ? ex_energy / MeV : 200.0;

    As = A * 0.5;
    Sigma2 = (A <= 235) ? 5.6 : 5.6 + 0.096 * (A - 235);
    Sigma1 = 0.5 * Sigma2;
    // JMQ's 301009 retuning, after the CEM transition probabilities became the default.
    SigmaS = 0.8 * std::exp(0.00553 * U + 2.1386);

    double wa = 0.0;
    w = 0.0;
    if (Z >= 90) {
      wa = (U <= 16.25) ? std::exp(0.5385 * U - 9.9564) : std::exp(0.09197 * U - 2.7003);
    } else if (Z == 89) {
      wa = std::exp(0.09197 * U - 1.0808);
    } else if (Z >= 82) {
      const double X = (barrier / MeV - 7.5 > 0.0) ? barrier / MeV - 7.5 : 0.0;
      wa = std::exp(0.09197 * (U - X) - 1.0808);
    } else {
      // Z < 82: symmetric fission only, and 1001 is the sentinel every `w > 1000` test below
      // reads. It is a flag written as a number.
      w = 1001.0;
    }

    if (Z >= 82) {
      const double x1 = (A1 - As) / Sigma1;
      const double x2 = (A2 - As) / Sigma2;
      const double f_asym_asym = 2 * fission_local_exp(x2) + fission_local_exp(x1);
      const double x3 = (As - A3) / SigmaS;
      const double f_sym_a1a2 = fission_local_exp(x3);
      const double w1 = (1.03 * wa - f_asym_asym > 0.0001) ? 1.03 * wa - f_asym_asym : 0.0001;
      const double w2 = (1.0 - f_sym_a1a2 * wa > 0.0001) ? 1.0 - f_sym_a1a2 * wa : 0.0001;
      w = w1 / w2;
      if (A < 227) { w *= std::exp(0.3 * (227 - A)); }
    }
  }
};

// ---------------------------------------------------------------------------------------------
// G4CompetitiveFission - the sampler
// ---------------------------------------------------------------------------------------------

/// G4CompetitiveFission::MassDistribution(x, A) - two asymmetric Gaussians at A1 and A2 plus
/// their mirror images at A - A1 and A - A2 at half weight, and a symmetric one at A/2 with
/// weight w. `w > 1000` selects the symmetric term alone and `w < 0.001` the asymmetric ones.
__host__ __device__ inline double fission_mass_distribution(const FissionParameters& p, double x,
                                                             int A) {
  const double y0 = (x - p.As) / p.SigmaS;
  const double xsym = fission_local_exp(y0);

  const double y1 = (x - p.A1) / p.Sigma1;
  const double y2 = (x - p.A2) / p.Sigma2;
  const double z1 = (x - A + p.A1) / p.Sigma1;
  const double z2 = (x - A + p.A2) / p.Sigma2;
  const double xasym = fission_local_exp(y1) + fission_local_exp(y2) +
                       0.5 * (fission_local_exp(z1) + fission_local_exp(z2));

  if (p.w > 1000) { return xsym; }
  if (p.w < 0.001) { return xasym; }
  return p.w * xsym + xasym;
}

/// G4CompetitiveFission::FissionAtomicNumber - rejection sampling of the above between C1 and
/// C2, with the majorant taken as the largest of the distribution's value at five named points
/// (A/2, (As+A1)/2, A1, (A1+A2)/2, A2) rather than as a bound. It is not a bound: the
/// distribution can exceed it between those points, and the sampler is then biased. That is
/// Geant4's and it is what shapes the fission mass yield.
///
/// The `C1 < 30` clamp forces the light fragment to at least A = 30, which is why fission of
/// anything below about A = 65 would be nonsense even if the channel were open.
template <typename Rng>
__host__ __device__ inline int fission_atomic_number(const FissionParameters& p, int A,
                                                      Rng& rng) {
  const double C2A = p.A2 + 3.72 * p.Sigma2;
  const double C2S = p.As + 3.72 * p.SigmaS;
  double C2;
  if (p.w > 1000.0) { C2 = C2S; }
  else if (p.w < 0.001) { C2 = C2A; }
  else { C2 = (C2A > C2S) ? C2A : C2S; }

  double C1 = A - C2;
  if (C1 < 30.0) {
    C2 = A - 30.0;
    C1 = 30.0;
  }

  const double Am1 = (p.As + p.A1) * 0.5;
  const double Am2 = (p.A1 + p.A2) * 0.5;
  double mass_max = fission_mass_distribution(p, p.As, A);
  const double m2 = fission_mass_distribution(p, Am1, A);
  const double m3 = fission_mass_distribution(p, p.A1, A);
  const double m4 = fission_mass_distribution(p, Am2, A);
  const double m5 = fission_mass_distribution(p, p.A2, A);
  if (m2 > mass_max) { mass_max = m2; }
  if (m3 > mass_max) { mass_max = m3; }
  if (m4 > mass_max) { mass_max = m4; }
  if (m5 > mass_max) { mass_max = m5; }

  double xm;
  double pm;
  do {
    xm = C1 + rng.uniform() * (C2 - C1);
    pm = fission_mass_distribution(p, xm, A);
  } while (mass_max * rng.uniform() > pm);
  // G4lrint - round half to even, which is what std::lrint does with the default rounding
  // mode. std::floor(x + 0.5) rounds half away from zero and differs on exactly .5, which
  // xm hits with probability zero but not never.
  return static_cast<int>(std::nearbyint(xm));
}

/// G4CompetitiveFission::FissionCharge - a Gaussian of width 0.6 about (Af/A)*Z + DeltaZ,
/// rejected until it lands in [1, Z-1] and at or below Af. DeltaZ interpolates between +0.45
/// and -0.45 across the A = 134 shell, which is the charge polarisation of the fission valley.
template <typename Rng>
__host__ __device__ inline int fission_charge(GaussCache& g, int A, int Z, double Af,
                                              Rng& rng) {
  const double sigma = 0.6;
  double delta_z;
  if (Af >= 134.0) { delta_z = -0.45; }
  else if (Af <= (A - 134.0)) { delta_z = 0.45; }
  else { delta_z = -0.45 * (Af - A * 0.5) / (134.0 - A * 0.5); }

  const double zmean = (Af / A) * Z + delta_z;
  double the_z;
  do {
    the_z = rand_gauss(g, rng, zmean, sigma);
  } while (the_z < 1.0 || the_z > (Z - 1.0) || the_z > Af);
  return static_cast<int>(std::nearbyint(the_z));
}

/// G4CompetitiveFission::Ratio(A, A11, B1, A00) - the parabolic reduction of the average
/// kinetic energy away from A00, continued LINEARLY above A00 + 10 rather than quadratically.
/// The second branch is the tangent at A00 + 10, so the two meet with matching slope.
__host__ __device__ inline double fission_ratio(double A, double A11, double B1, double A00) {
  if (A11 >= A * 0.5 && A11 <= (A00 + 10.0)) {
    const double x = (A11 - A00) / A;
    return 1.0 - B1 * x * x;
  }
  const double x = 10.0 / A;
  return 1.0 - B1 * x * x - 2.0 * x * B1 * (A11 - A00 - 10.0) / A;
}

__host__ __device__ inline double fission_asymmetric_ratio(int A, double A11) {
  return fission_ratio(static_cast<double>(A), A11, 23.5, 134.0);
}
__host__ __device__ inline double fission_symmetric_ratio(int A, double A11) {
  const double a0 = static_cast<double>(A);
  return fission_ratio(a0, A11, 5.32, a0 * 0.5);
}

/// G4CompetitiveFission::FissionKineticEnergy.
///
/// `Eaverage = 0.1071 Z^2/A^(1/3) + 22.2` MeV is the Viola systematics. The mode - symmetric
/// or asymmetric - is chosen by a single random number against Psy, and the two modes differ
/// in their average energy (12.5 MeV apart, in opposite directions) and in their dispersion
/// (10 MeV asymmetric, 8 MeV symmetric).
///
/// The rejection window is +-3.72 sigma about Eaverage AND at or below Tmax, and after 100
/// failures the function returns Eaverage itself - not the last draw. That fall-back is a
/// spike in the spectrum at Eaverage and it is reached whenever Tmax is below
/// Eaverage - 3.72 sigma, i.e. for a fragment with too little energy to fission comfortably.
template <typename Rng>
__host__ __device__ inline double fission_kinetic_energy(const FissionParameters& p,
                                                          GaussCache& g, int A, int Z, int Af1,
                                                          int Af2, double Tmax, Rng& rng) {
  const int af_max = (Af1 > Af2) ? Af1 : Af2;

  double Pas = 0.0;
  if (p.w <= 1000) {
    const double x1 = (af_max - p.A1) / p.Sigma1;
    const double x2 = (af_max - p.A2) / p.Sigma2;
    Pas = 0.5 * fission_local_exp(x1) + fission_local_exp(x2);
  }
  double Ps = 0.0;
  if (p.w >= 0.001) {
    const double xs = (af_max - p.As) / p.SigmaS;
    Ps = p.w * fission_local_exp(xs);
  }
  const double Psy = (Pas + Ps > 0.0) ? Ps / (Pas + Ps) : 0.5;

  const double PPas = p.Sigma1 + 2.0 * p.Sigma2;
  const double PPsy = p.w * p.SigmaS;
  const double Xas = (PPas + PPsy > 0.0) ? PPas / (PPas + PPsy) : 0.5;
  const double Xsy = 1.0 - Xas;

  const double Eaverage =
      (0.1071 * (static_cast<double>(Z) * Z) / data::g4pow_z13<double>(A) + 22.2) *
      u::MeV<double>();

  double t_average;
  double ESigma = 10 * u::MeV<double>();
  if (rng.uniform() > Psy) {
    // Asymmetric. The +-0.7979 sigma points are where a Gaussian's mean absolute deviation
    // falls, and the scale factor normalises the ratio at those four points.
    const double A11 = p.A1 - 0.7979 * p.Sigma1;
    const double A12 = p.A1 + 0.7979 * p.Sigma1;
    const double A21 = p.A2 - 0.7979 * p.Sigma2;
    const double A22 = p.A2 + 0.7979 * p.Sigma2;
    const double scale =
        0.5 * p.Sigma1 * (fission_asymmetric_ratio(A, A11) + fission_asymmetric_ratio(A, A12)) +
        p.Sigma2 * (fission_asymmetric_ratio(A, A21) + fission_asymmetric_ratio(A, A22));
    t_average = (Eaverage + 12.5 * Xsy) * (PPas / scale) *
                fission_asymmetric_ratio(A, static_cast<double>(af_max));
  } else {
    const double As0 = p.As + 0.7979 * p.SigmaS;
    t_average = (Eaverage - 12.5 * u::MeV<double>() * Xas) *
                fission_symmetric_ratio(A, static_cast<double>(af_max)) /
                fission_symmetric_ratio(A, As0);
    ESigma = 8.0 * u::MeV<double>();
  }

  double ke;
  int i = 0;
  do {
    ke = rand_gauss(g, rng, t_average, ESigma);
    if (++i > 100) { return Eaverage; }
  } while (ke < Eaverage - 3.72 * ESigma || ke > Eaverage + 3.72 * ESigma || ke > Tmax);
  return ke;
}

/// What one fission produced.
struct FissionProducts {
  bool fissioned = false;
  Fragment fragment1;
  /// Set when 100 trials failed to give a non-negative fragment excitation. Geant4 throws
  /// G4HadronicException here; see the file header.
  bool refused_no_valid_split = false;
};

/// G4CompetitiveFission::EmittedFragment. `nucleus` becomes the heavy fragment.
///
/// The excitation the two fragments carry is `Tmax - KineticEnergy + pcorr`, and the `+ pcorr`
/// is JMQ's 04/03/09 fix: the fission pairing energy was subtracted from the available energy
/// and has to come back as excitation or the event does not conserve energy. It is then split
/// in proportion to A1 and A2 - not equally, and not by level density.
template <typename Rng>
__host__ __device__ inline FissionProducts fission_emitted_fragment(const FissionState& s,
                                                                     Fragment& nucleus,
                                                                     Rng& rng) {
  FissionProducts out;
  const int A = nucleus.a;
  const int Z = nucleus.z;
  const double U = nucleus.excitation;
  const double pcorr = deex::fission_pairing_correction(A, Z);
  if (U <= pcorr) { return out; }

  double M = nucleus.ground_state_mass;
  const LorentzVector nucleus_momentum = nucleus.momentum;

  FissionParameters p;
  p.define(A, Z, U - pcorr, s.barrier);

  GaussCache g;
  int A1 = 0, Z1 = 0, A2 = 0, Z2 = 0;
  double M1 = 0.0, M2 = 0.0;
  double frag_exc = 0.0;
  double frag_ke = 0.0;
  int trials = 0;
  do {
    A1 = fission_atomic_number(p, A, rng);
    Z1 = fission_charge(g, A, Z, static_cast<double>(A1), rng);
    M1 = deex::nuclear_mass(A1, Z1);

    A2 = A - A1;
    Z2 = Z - Z1;
    if (A2 < 1 || Z2 < 0 || Z2 > A2) {
      frag_exc = -1.0;
      continue;
    }
    M2 = deex::nuclear_mass(A2, Z2);
    const double Tmax = M + U - M1 - M2 - pcorr;
    if (Tmax < 0.0) {
      frag_exc = -1.0;
      continue;
    }
    frag_ke = fission_kinetic_energy(p, g, A, Z, A1, A2, Tmax, rng);
    frag_exc = Tmax - frag_ke + pcorr;
  } while (frag_exc < 0.0 && ++trials < 100);

  if (frag_exc <= 0.0) {
    out.refused_no_valid_split = true;
    return out;
  }

  M1 += frag_exc * A1 / static_cast<double>(A);
  M2 += frag_exc * A2 / static_cast<double>(A);
  M += U;

  const double etot1 = ((M - M2) * (M + M2) + M1 * M1) / (2 * M);
  const Vec3d dir = random_direction(rng);
  const double mom1 = std::sqrt((etot1 - M1) * (etot1 + M1));
  LorentzVector four1(dir * mom1, etot1);
  four1.boost(nucleus_momentum.boost_vector());

  out.fragment1 = make_fragment(A1, Z1, four1);
  LorentzVector rest = nucleus_momentum;
  rest -= four1;
  nucleus.set_za_and_momentum(rest, Z2, A2);
  out.fissioned = true;
  return out;
}

}  // namespace g4gpu::deex

#endif
