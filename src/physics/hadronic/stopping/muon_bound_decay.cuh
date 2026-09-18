// G4MuonMinusBoundDecay: whether a muon that has reached the K shell decays there or is captured
// by the nucleus, and - if it decays - the bound Michel spectrum it decays into.
//
// Transcribed from Geant4 11.1.1,
//   processes/hadronic/stopping/src/G4MuonMinusBoundDecay.cc
//     ApplyYourself, GetMuonCaptureRate, GetMuonDecayRate, GetMuonZeff
//
// ---------------------------------------------------------------------------------------------
// **This is the only place in the at-rest chain where the answer is a competition rather than a
// sequence.** Everything else - element selection, the atomic cascade, the nuclear model - always
// happens. Here a capture rate and a decay rate are computed for the (Z, A) the muon landed on,
// and ONE uniform decides which wins:
//
//     lambda = lambdac + lambdad
//     time  = t - log(u)/lambda        <- the muon's global time is advanced FIRST, always
//     if (u2 * lambda < lambdac)  -> isAlive,     the caller then runs the nuclear model
//     else                        -> stopAndKill, and this class emits e-, anti-nu_e, nu_mu
//
// The time advance happens on BOTH branches and before the branch is chosen, because the muon
// sits in the K shell for an exponentially distributed time whether it then decays or is
// captured; `G4HadronStoppingProcess` reads that time back as `capTime` and adds it to every
// nuclear secondary. So the two draws are not interchangeable and their order is part of the
// stream.
//
// **The capture rate is 93 measured (Z, A) rates and a formula for everything else.** The table
// is Suzuki, Measday and Roalsvig, Phys. Rev. C35 (1987) 2212, in units of 1/microsecond, and it
// is searched with an early exit: `if (capRates[j].Z > Z) break;` - so it relies on being sorted
// by Z, which it is, and a (Z, A) whose Z is tabulated but whose A is not falls through to the
// formula rather than to a neighbouring isotope. Carbon-12 and carbon-13 are both listed; carbon
// -14 is not, and gets Goulard-Primakoff.
//
// **The decay-rate table has one row and its lookup is commented out.** `decRates` contains the
// free muon at (Z=0, A=0) and the loop that would search it is commented out in the source with
// "we'll use the above code once we have the data" beside it. So for every real element the
// tabulated branch cannot fire and the formula always runs: a Z-dependent reduction of the free
// rate 0.45517005/microsecond, with two regimes at Z = 14. Reproduced as written, dead row and
// all, because deleting it would be deleting the statement that the data does not exist yet.
//
// **The bound Michel spectrum is sampled in the muon's rest frame and boosted by its own K-shell
// motion.** `KEnergy` is `projectile.GetBoundEnergy()`, which is the atomic cascade's telescoped
// total - the K-shell binding energy - and it is used twice: once to build the muon's
// four-momentum `(sqrt(K(K+2m)) * dir, K + m)` whose boost the electron gets, and once as
// `Eelect = EL.e() - m_e - 2*KEnergy`, which subtracts the binding energy A SECOND TIME on top of
// the boost. That asymmetry is Geant4's and is what makes the bound spectrum softer than the free
// one; the rejection loop re-samples whenever it drives the electron energy negative.
#ifndef G4GPU_STOPPING_MUON_BOUND_DECAY_CUH
#define G4GPU_STOPPING_MUON_BOUND_DECAY_CUH

#include <cmath>

#include "core/units.cuh"
#include "physics/hadronic/deexcitation/fragment.cuh"
#include "physics/hadronic/process.cuh"

namespace g4gpu::physics::hadronic::stopping {

/// `G4ThreeVector::unit()`, CLHEP's: the vector divided by its magnitude, and the vector ITSELF
/// when the magnitude is zero - not +z. docs/RISK.md V127 is the entry about the other convention;
/// this call site wants CLHEP's, because a decay electron with exactly zero momentum should give
/// a zero direction and not a fabricated one, and the caller can see the difference.
__host__ __device__ inline deex::Vec3d clhep_unit3(const deex::Vec3d& v) {
  const double m2 = g4gpu::mag2(v);
  if (m2 <= 0.0) { return v; }
  const double inv = 1.0 / std::sqrt(m2);
  return deex::Vec3d{v.x * inv, v.y * inv, v.z * inv};
}

/// CLHEP's microsecond in this port's nanoseconds, so that a rate in 1/microsecond becomes a rate
/// in 1/ns the way `cRate / microsecond` does in Geant4's unit system.
__host__ __device__ inline constexpr double bd_microsecond_ns() { return 1.0e3; }

/// CLHEP's `fine_structure_const`, which is DERIVED as `elm_coupling/hbarc` and is not the
/// CODATA number anyone would type. `core/units.cuh` already pins it, and this file asks for it
/// there rather than repeating it: the first version of this line carried 7.2973525693e-3 - CODATA
/// 2018, off in the eleventh digit - and the muon decay rate at Z = 77 came out 1.1e-10 wrong,
/// which the oracle caught. docs/RISK.md V8 is the entry about hand-typed constants; this is the
/// same mistake in a package written eight months later.
__host__ __device__ inline constexpr double bd_fine_structure() {
  return units::fine_structure_const<double>();
}

/// The 93 measured muon capture rates, Suzuki/Measday/Roalsvig Phys. Rev. C35 (1987) 2212, with
/// hydrogen from Phys. Rev. Lett. 99 (2007) 032002 and helium from Measday Phys. Rep. 354 (2001)
/// 243. Rates in 1/microsecond. Sorted by Z, which the early exit below depends on.
__host__ __device__ inline int bd_cap_rate_count() { return 93; }

__host__ __device__ inline void bd_cap_rate(int i, int& z, int& a, double& rate) {
  const short zz[93] = {1,  2,  2,  3,  3,  4,  5,  5,  6,  6,  7,  8,  8,  9,  10, 11, 12,
                        13, 14, 15, 16, 17, 17, 18, 19, 20, 21, 22, 23, 24, 24, 24, 24, 25,
                        26, 27, 28, 28, 28, 29, 30, 31, 32, 33, 34, 35, 35, 37, 38, 39, 40,
                        41, 42, 45, 46, 47, 48, 49, 50, 51, 52, 53, 55, 56, 57, 58, 59, 60,
                        62, 64, 65, 66, 67, 68, 72, 73, 74, 79, 80, 81, 82, 83, 90, 92, 92,
                        92, 92, 93, 94, 94, 0,  0,  0};
  const short aa[93] = {1,   3,   4,   6,   7,   9,   10,  11,  12,  13,  14,  16,  18,
                        19,  20,  23,  24,  27,  28,  31,  32,  35,  37,  40,  39,  40,
                        45,  48,  51,  50,  52,  53,  54,  55,  56,  59,  58,  60,  62,
                        63,  66,  69,  72,  75,  80,  79,  81,  85,  88,  89,  91,  93,
                        96,  103, 106, 107, 112, 115, 119, 121, 128, 127, 133, 138, 139,
                        140, 141, 144, 150, 157, 159, 163, 165, 167, 178, 181, 184, 197,
                        201, 205, 207, 209, 232, 238, 233, 235, 236, 237, 239, 242, 0, 0, 0};
  const double rr[93] = {
      0.000725, 0.002149, 0.000356, 0.004647, 0.002229, 0.006107, 0.02757,  0.02188,
      0.03807,  0.03474,  0.06885,  0.10242,  0.0880,   0.22905,  0.2288,   0.3773,
      0.4823,   0.6985,   0.8656,   1.1681,   1.3510,   1.800,    1.250,    1.2727,
      1.8492,   2.5359,   2.711,    2.5908,   3.073,    3.825,    3.465,    3.297,
      3.057,    3.900,    4.408,    4.945,    6.11,     5.56,     4.72,     5.691,
      5.806,    5.700,    5.561,    6.094,    5.687,    7.223,    7.547,    6.89,
      6.93,     7.89,     8.620,    10.38,    9.298,    10.010,   10.000,   10.869,
      10.624,   11.38,    10.60,    10.40,    9.174,    11.276,   10.98,    10.112,
      10.71,    11.501,   13.45,    12.35,    12.22,    12.00,    12.73,    12.29,
      12.95,    13.04,    13.03,    12.86,    12.76,    13.35,    12.74,    13.85,
      13.295,   13.238,   12.555,   12.592,   14.27,    13.470,   13.90,    13.58,
      13.90,    12.86,    0.0,      0.0,      0.0};
  z = zz[i];
  a = aa[i];
  rate = double(rr[i]);
}

/// G4MuonMinusBoundDecay::GetMuonZeff - effective charges from the same Suzuki paper, and where
/// it has none, Ford and Wills Nucl. Phys. 35 (1962) 295 or an interpolation. Index 0 is the 0.0
/// Geant4 writes and never reads: `Z = max(min(ZZ, 100), 1)` clamps into 1..100 first.
__host__ __device__ inline double muon_zeff(int zz) {
  const double zeff[101] = {
      0.0,     1.00,  1.98,  2.94,  3.89,  4.81,  5.72,  6.61,  7.49,  8.32,  9.14,
      9.95,   10.69, 11.48, 12.22, 12.90, 13.64, 14.24, 14.89, 15.53, 16.15, 16.77,
      17.38,  18.04, 18.49, 19.06, 19.59, 20.13, 20.66, 21.12, 21.61, 22.02, 22.43,
      22.84,  23.24, 23.65, 24.06, 24.47, 24.85, 25.23, 25.61, 25.99, 26.37, 26.69,
      27.00,  27.32, 27.63, 27.95, 28.20, 28.42, 28.64, 28.79, 29.03, 29.27, 29.51,
      29.75,  29.99, 30.22, 30.36, 30.53, 30.69, 30.85, 31.01, 31.18, 31.34, 31.48,
      31.62,  31.76, 31.90, 32.05, 32.19, 32.33, 32.47, 32.61, 32.76, 32.94, 33.11,
      33.29,  33.46, 33.64, 33.81, 34.21, 34.18, 34.00, 34.10, 34.21, 34.31, 34.42,
      34.52,  34.63, 34.73, 34.84, 34.94, 35.05, 35.16, 35.25, 35.36, 35.46, 35.57,
      35.67,  35.78};
  const int z = (zz > 100) ? 100 : ((zz < 1) ? 1 : zz);
  return double(zeff[z]);
}

/// G4MuonMinusBoundDecay::GetMuonCaptureRate, in 1/ns.
///
/// The table first, with Geant4's early exit on `capRates[j].Z > Z`; then Goulard and Primakoff,
/// Phys. Rev. C10 (1974) 2034, with the two "suggested by user" constants the source carries -
/// `t1 = 875e-9` (changed from -10) and `xmu = zeff^2 * 2.663e-5` (changed from ^-4).
__host__ __device__ inline double muon_capture_rate(int z, int a) {
  double lambda = -1.0;
  for (int j = 0; j < bd_cap_rate_count(); ++j) {
    int zj, aj;
    double rj;
    bd_cap_rate(j, zj, aj, rj);
    if (zj == z && aj == a) {
      lambda = rj / bd_microsecond_ns();
      break;
    }
    if (zj > z) { break; }
  }
  if (lambda < 0.0) {
    const double b0a = -0.03;
    const double b0b = -0.25;
    const double b0c = 3.24;
    const double t1 = 875.0e-9;
    const double r1 = muon_zeff(z);
    const double zeff2 = r1 * r1;
    const double xmu = zeff2 * 2.663e-5;
    const double a2ze = 0.5 * double(a) / double(z);
    const double r2 = 1.0 - xmu;
    // Written as ONE expression, as Geant4 writes it: the whole
    // `2*(A-Z) + std::abs(a2ze - 1.)` is inside the cast, so the integer and the double are
    // summed BEFORE the multiplication. Splitting it into two terms is algebraically identical
    // and not bit-identical, which is the only kind of identity a transcription can claim.
    lambda = t1 * zeff2 * zeff2 * (r2 * r2) * (1.0 - (1.0 - xmu) * 0.75704) *
             (a2ze * b0a + 1.0 - (a2ze - 1.0) * b0b -
              (double(2 * (a - z)) + std::fabs(a2ze - 1.0)) * b0c / double(a * 4));
  }
  return lambda;
}

/// G4MuonMinusBoundDecay::GetMuonDecayRate, in 1/ns.
///
/// The tabulated branch is unreachable for any element - see the header - so this is always the
/// formula: the free rate reduced by `(Z*alpha)^2 * (0.5 + 0.06*m_mu/M)` below Z = 14 and by
/// `(Z*alpha)^2 * (0.868699 - Z*alpha*0.708985)` at and above it, the second being a fit to the
/// Phys. Rev. C35 (1987) 2212 data.
__host__ __device__ inline double muon_decay_rate(int z, double mu_mass_MeV,
                                                  double nucl_mass_MeV) {
  if (z == 0) { return 0.45517005 / bd_microsecond_ns(); }
  const double free_rate = 0.45517005 / bd_microsecond_ns();
  double lambda = 1.0;
  const double x = double(z) * bd_fine_structure();
  if (z < 14) {
    lambda -= x * x * (0.5 + 0.06 * mu_mass_MeV / nucl_mass_MeV);
  } else {
    lambda -= x * x * (0.868699 - x * 0.708985);
  }
  return lambda * free_rate;
}

/// One bound-decay outcome.
struct BoundDecayResult {
  /// True when the muon decayed in orbit: the caller must NOT run the nuclear model.
  bool decayed = false;
  /// The muon's global time after the exponential wait, ns. Set on both branches.
  double time_ns = 0.0;
  int n = 0;
  int pdg[3] = {0, 0, 0};              ///< e-, anti_nu_e, nu_mu in that order
  double kin_energy[3] = {0, 0, 0};    ///< MeV
  Vec3<double> direction[3];
};

/// G4MuonMinusBoundDecay::ApplyYourself.
///
/// `bound_energy_MeV` is the atomic cascade's telescoped total, i.e. the K-shell energy.
/// `t0_ns` is the projectile's global time going in.
template <typename Rng>
__host__ __device__ inline void muon_bound_decay(int z, int a, double nucl_mass_MeV,
                                                 double bound_energy_MeV, double t0_ns,
                                                 Rng& rng, BoundDecayResult& out) {
  out = BoundDecayResult();
  const double mu = 105.6583715;
  const double me = 0.510998910;

  const double lambdac = muon_capture_rate(z, a);
  const double lambdad = muon_decay_rate(z, mu, nucl_mass_MeV);
  const double lambda = lambdac + lambdad;

  // The time advance happens on both branches and before the branch is chosen.
  out.time_ns = t0_ns - std::log(double(rng.uniform())) / lambda;

  if (double(rng.uniform()) * lambda < lambdac) {
    out.decayed = false;      // isAlive: the caller runs the nuclear model
    return;
  }
  out.decayed = true;         // stopAndKill

  const double xmax = 1.0 + me * me / (mu * mu);
  const double xmin = 2.0 * me / mu;
  const double k = bound_energy_MeV;

  const double pmu = std::sqrt(k * (k + 2.0 * mu));
  const double emu = k + mu;
  const deex::Vec3d mdir = deex::random_direction(rng);
  const deex::LorentzVector MU(deex::Vec3d{mdir.x * pmu, mdir.y * pmu, mdir.z * pmu}, emu);
  const deex::Vec3d bst = MU.boost_vector();

  double e_elect = 0.0;
  double ecm = 0.0;
  deex::LorentzVector EL;
  deex::LorentzVector NN;
  for (;;) {
    double x;
    do {
      x = xmin + (xmax - xmin) * double(rng.uniform());
    } while (double(rng.uniform()) > (3.0 - 2.0 * x) * x * x);
    double ee = x * mu * 0.5;
    double pe;
    if (ee > me) {
      pe = std::sqrt(ee * ee - me * me);
    } else {
      pe = 0.0;
      ee = me;
    }
    const deex::Vec3d d = deex::random_direction(rng);
    EL = deex::LorentzVector(deex::Vec3d{d.x * pe, d.y * pe, d.z * pe}, ee);
    EL.boost(bst);
    // The binding energy is subtracted a SECOND time here, on top of the boost. Geant4's.
    e_elect = EL.e - me - 2.0 * k;
    NN = MU - EL;
    ecm = NN.e * NN.e - g4gpu::mag2(NN.v);
    if (!(e_elect < 0.0 || ecm < 0.0)) { break; }
  }

  const deex::Vec3d eldir = clhep_unit3(EL.v);
  out.pdg[0] = 11;
  out.kin_energy[0] = e_elect;
  out.direction[0] = Vec3<double>{eldir.x, eldir.y, eldir.z};

  ecm = 0.5 * std::sqrt(ecm);
  const deex::Vec3d nbst = NN.boost_vector();
  const deex::Vec3d n1d = deex::random_direction(rng);
  deex::LorentzVector N1(deex::Vec3d{n1d.x * ecm, n1d.y * ecm, n1d.z * ecm}, ecm);
  N1.boost(nbst);
  out.pdg[1] = -12;                       // anti_nu_e
  out.kin_energy[1] = N1.e;               // massless: kinetic energy is the total energy
  const deex::Vec3d n1u = clhep_unit3(N1.v);
  out.direction[1] = Vec3<double>{n1u.x, n1u.y, n1u.z};

  NN = NN - N1;
  out.pdg[2] = 14;                        // nu_mu
  out.kin_energy[2] = NN.e;
  const deex::Vec3d n2u = clhep_unit3(NN.v);
  out.direction[2] = Vec3<double>{n2u.x, n2u.y, n2u.z};
  out.n = 3;
}

}  // namespace g4gpu::physics::hadronic::stopping

#endif  // G4GPU_STOPPING_MUON_BOUND_DECAY_CUH
