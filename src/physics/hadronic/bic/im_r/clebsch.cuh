// G4Clebsch - the isospin bookkeeping every resonance-production cross section is scaled by.
//
// Transcribed from G4Clebsch.{hh,cc} (hadronic/util) and G4Pow's `logfact` table (11.1.1):
// `TriangleCoeff`, `ClebschGordanCoeff`, `ClebschGordan`, `Weight` and `GenerateIso3`.
//
// `G4VXResonance::IsospinCorrection` divides the tabulated NN -> resonance cross section by the
// proton-proton weight and multiplies it by the weight for the actual pair, and `DetailedBalance`
// uses a third. So every one of the fifty resonance channels is a table lookup times a ratio of
// two of these numbers, and getting one of them wrong moves a channel's share of the total
// without moving the total.
//
// ## Everything here is a SQUARE, and that is the class's own convention
//
// `ClebschGordanCoeff` returns the Clebsch-Gordan coefficient and `ClebschGordan` returns its
// SQUARE - the name says coefficient and returns a probability. `Weight` sums `ClebschGordan`
// over the allowed total isospins, so a weight is a sum of squares and is never negative;
// `GenerateIso3` builds its cumulative from the same squares. Transcribed with the names Geant4
// uses, because a reader checking this against a table of Clebsch-Gordan coefficients will
// otherwise find every number squared and think the port is wrong.
//
// ## logfactorial is a TABLE, accumulated in one direction
//
// `G4Pow::logfactorial(Z)` is `logfact[Z]`, and `G4Pow`'s constructor fills it with
//
//     G4double logf = 0.0;
//     for (G4int i = 1; i < 512; ++i) { logf += G4Log(G4double(i)); logfact[i] = logf; }
//
// - a running sum in increasing `i`, not `lgamma` and not a fresh sum per entry. The port builds
// the same table the same way once, because `log(1) + log(2) + ... + log(n)` summed forwards is
// not the same double as any other arrangement of it, and `ClebschGordanCoeff` takes differences
// of six of these and exponentiates the result. `logfact[0]` is 0 from the vector's fill value,
// which is also `log(0!)`.
//
// ## REFUSED, by name
//
//   * **`GenerateIso3`'s final sampling loop**, and with it the whole function. Four things in it
//     cannot be transcribed into a kernel as written and none of them is a rounding:
//       - `prbout[20][20]` is a local array read at `[m1pos][m2pos]` but INITIALISED only at the
//         entries the `m1pr + m2pr == twoM3` test reaches; every other entry is whatever was on
//         the stack. The normalisation loop then divides all 400 of them by `sum`, reading the
//         uninitialised ones, and the sampling loop compares against them.
//       - the two sampling loops run `m1p < m1pos` and `m2p < m2pos`, both EXCLUSIVE, so the last
//         row and the last column of the table can never be selected however large their
//         probability - for a Delta (twoJOut = 3) that is one of the four charge states.
//       - `c34 = ClebschGordan(0,0,0,0,0)` is 1 by construction and is computed inside the
//         innermost loop.
//       - it falls off the end with a `JustWarning` and returns an EMPTY vector, which
//         `G4VXResonance::IsospinCorrection` then indexes at [0] and [1].
//     `ClebschRefusal::generate_iso3` is set at the point it would have been called. The rest of
//     this file - the three functions the cross sections actually use - is complete and exact.
//   * `Wigner3J` and `NormalizedClebschGordan`, which nothing in the binary cascade calls.
#ifndef G4GPU_BIC_IMR_CLEBSCH_CUH
#define G4GPU_BIC_IMR_CLEBSCH_CUH

#include <cmath>

namespace g4gpu::bic::imr {

/// What a Clebsch-Gordan call could not do.
struct ClebschRefusal {
  bool generate_iso3 = false;  ///< see the file header
  /// `kMin < 0`, `kMax < kMin` or `kMax >= 512` - all three are `JustWarning` plus `return 0` in
  /// Geant4, so the returned value is the same; the flag is what a kernel has instead of the
  /// warning.
  bool coefficient_range = false;
};

/// `G4Pow::logfact`, built exactly as `G4Pow`'s constructor builds it: a running sum of
/// `log(i)` in increasing `i`, with `logfact[0] = 0`.
///
/// 512 entries, because that is `maxZ` and because `ClebschGordanCoeff` refuses an index at or
/// above it (`G4POWLOGFACTMAX`). Built once into a function-local static; a device instantiation
/// gets it in constant/global memory like every other table in this package.
__host__ __device__ inline const double* log_factorial_table() {
  static double v[512];
  static bool built = false;
  if (!built) {
    double logf = 0.0;
    v[0] = 0.0;
    for (int i = 1; i < 512; ++i) {
      logf += std::log(static_cast<double>(i));
      v[i] = logf;
    }
    built = true;
  }
  return v;
}

/// `G4Pow::logfactorial(Z)`. Out of range is 0, which is what a `std::vector` would NOT do - but
/// every caller here has already range-checked, and a kernel cannot throw.
__host__ __device__ inline double log_factorial(int z) {
  return (z >= 0 && z < 512) ? log_factorial_table()[z] : 0.0;
}

/// `G4Clebsch::TriangleCoeff(2A, 2B, 2C)` -
/// `sqrt[(A+B-C)! (A-B+C)! (-A+B+C)! / (A+B+C+1)!]`, zero when the triad fails the triangle
/// inequality.
///
/// Note the parity check is on the FIRST combination only - Geant4's comment says "only have to
/// check that i is even the first time" - so a triad with `2A+2B-2C` even and `2A-2B+2C` odd
/// takes `logfactorial` of a truncated half-integer rather than returning zero. The three
/// callers always pass triads whose sum is even, so it never happens; written as Geant4 wrote it.
__host__ __device__ inline double clebsch_triangle_coeff(int two_a, int two_b, int two_c) {
  double val = 0.0;
  int i = two_a + two_b - two_c;
  if (i < 0 || (i % 2)) { return 0.0; }
  val += log_factorial(i / 2);

  i = two_a - two_b + two_c;
  if (i < 0) { return 0.0; }
  val += log_factorial(i / 2);

  i = -two_a + two_b + two_c;
  if (i < 0) { return 0.0; }
  val += log_factorial(i / 2);

  i = two_a + two_b + two_c + 2;
  if (i < 0) { return 0.0; }
  return std::exp(0.5 * (val - log_factorial(i / 2)));
}

/// `G4Clebsch::ClebschGordanCoeff` - the coefficient itself, by the Racah sum.
///
/// The sum runs over `k` from `kMin` to `kMax` and every term is an exponential of a difference
/// of seven log-factorials, so the cancellation is done in the exponent rather than between
/// large factorials. `factor` is halved once, outside the loop, exactly where Geant4 halves it.
__host__ __device__ inline double clebsch_gordan_coeff(int two_j1, int two_m1, int two_j2,
                                                       int two_m2, int two_j,
                                                       ClebschRefusal& ref) {
  if (two_j1 < 0 || two_j2 < 0 || two_j < 0 || ((two_j1 - two_m1) % 2) ||
      ((two_j2 - two_m2) % 2)) {
    return 0.0;
  }
  const int two_m = two_m1 + two_m2;
  if (two_m1 > two_j1 || two_m1 < -two_j1 || two_m2 > two_j2 || two_m2 < -two_j2 ||
      two_m > two_j || two_m < -two_j) {
    return 0.0;
  }
  const double triangle = clebsch_triangle_coeff(two_j1, two_j2, two_j);
  if (triangle == 0.0) { return 0.0; }

  double factor = log_factorial((two_j1 + two_m1) / 2) + log_factorial((two_j1 - two_m1) / 2);
  factor += log_factorial((two_j2 + two_m2) / 2) + log_factorial((two_j2 - two_m2) / 2);
  factor += log_factorial((two_j + two_m) / 2) + log_factorial((two_j - two_m) / 2);
  factor *= 0.5;

  int k_min = 0;
  const int sum1 = (two_j1 - two_m1) / 2;
  int k_max = sum1;
  const int sum2 = (two_j - two_j2 + two_m1) / 2;
  if (-sum2 > k_min) { k_min = -sum2; }
  const int sum3 = (two_j2 + two_m2) / 2;
  if (sum3 < k_max) { k_max = sum3; }
  const int sum4 = (two_j - two_j1 - two_m2) / 2;
  if (-sum4 > k_min) { k_min = -sum4; }
  const int sum5 = (two_j1 + two_j2 - two_j) / 2;
  if (sum5 < k_max) { k_max = sum5; }

  // All three are `JustWarning; return 0` in Geant4. `kMin < 0` cannot happen - it starts at 0
  // and only ever grows - and is transcribed because Geant4 tests for it.
  if (k_min < 0 || k_max < k_min || k_max >= 512) {
    ref.coefficient_range = true;
    return 0.0;
  }

  double k_sum = 0.0;
  for (int k = k_min; k <= k_max; ++k) {
    const double sign = (k % 2) ? -1.0 : 1.0;
    k_sum += sign * std::exp(factor - log_factorial(sum1 - k) - log_factorial(sum2 + k) -
                             log_factorial(sum3 - k) - log_factorial(sum4 + k) -
                             log_factorial(k) - log_factorial(sum5 - k));
  }
  return triangle * std::sqrt(static_cast<double>(two_j + 1)) * k_sum;
}

/// `G4Clebsch::ClebschGordan` - the SQUARE of the coefficient. See the file header.
__host__ __device__ inline double clebsch_gordan(int two_j1, int two_m1, int two_j2, int two_m2,
                                                 int two_j, ClebschRefusal& ref) {
  const double c = clebsch_gordan_coeff(two_j1, two_m1, two_j2, two_m2, two_j, ref);
  return c * c;
}

/// `G4Clebsch::Weight` - the sum of `ClebschGordan` over every total isospin the entrance and the
/// exit channel both allow.
///
/// The range is the intersection of `[|J1-J2|, J1+J2]` and `[|JOut1-JOut2|, JOut1+JOut2]`, each
/// raised to `|M|`. An empty intersection gives zero, which is the value
/// `G4VXResonance::IsospinCorrection` then throws a `G4HadronicException` on when it is the
/// proton-proton weight.
///
/// **The `|M|` floor cannot change the answer.** `ClebschGordan` already returns zero whenever
/// `|twoM| > twoJ`, so lowering the range below `|M|` only adds zero terms. MEASURED: removing
/// the floor from `twoJMinIn` changes none of the 1,600 weights the oracle compares. It is
/// transcribed because it is what decides how many terms the loop runs, and because a release
/// that made `ClebschGordan` return something non-zero out of range would need it.
__host__ __device__ inline double clebsch_weight(int two_j1, int two_m1, int two_j2, int two_m2,
                                                 int two_j_out1, int two_j_out2,
                                                 ClebschRefusal& ref) {
  double value = 0.0;
  const int two_m = two_m1 + two_m2;
  const int abs_diff_in = (two_j1 - two_j2 >= 0) ? (two_j1 - two_j2) : (two_j2 - two_j1);
  const int abs_m = (two_m >= 0) ? two_m : -two_m;
  const int two_j_min_in = (abs_diff_in > abs_m) ? abs_diff_in : abs_m;
  const int two_j_max_in = two_j1 + two_j2;

  const int abs_diff_out =
      (two_j_out1 - two_j_out2 >= 0) ? (two_j_out1 - two_j_out2) : (two_j_out2 - two_j_out1);
  const int two_j_min_out = (abs_diff_out > abs_m) ? abs_diff_out : abs_m;
  const int two_j_max_out = two_j_out1 + two_j_out2;

  const int two_j_min = (two_j_min_in > two_j_min_out) ? two_j_min_in : two_j_min_out;
  const int two_j_max = (two_j_max_in < two_j_max_out) ? two_j_max_in : two_j_max_out;

  for (int two_j = two_j_min; two_j <= two_j_max; two_j += 2) {
    value += clebsch_gordan(two_j1, two_m1, two_j2, two_m2, two_j, ref);
  }
  return value;
}

/// `G4Clebsch::NormalizedClebschGordan(2J, 2m, 2J1, 2J2, 2m1, 2m2)` - the probability that a
/// state (J, m) decomposes into exactly (J1, m1) and (J2, m2), normalised over every m1 the pair
/// allows. `G4XAnnihilationChannel` multiplies its cross section by it.
///
/// The argument order is not `ClebschGordan`'s: the TOTAL spin and projection come first, then
/// the two constituents' spins, then their projections. And the `twoJ1 == 0 || twoJ2 == 0` guard
/// returns ZERO, not one - the caller `G4XAnnihilationChannel::NormalizedClebsch` has already
/// returned 1 for that case one line earlier, so the guard here is unreachable from it.
__host__ __device__ inline double normalized_clebsch_gordan(int two_j, int two_m, int two_j1,
                                                            int two_j2, int two_m1, int two_m2,
                                                            ClebschRefusal& ref) {
  double cleb = 0.0;
  if (two_j1 == 0 || two_j2 == 0) { return cleb; }
  double sum = 0.0;
  for (int m1c = -two_j1; m1c <= two_j1; m1c += 2) {
    const int m2c = two_m - m1c;
    const double prob = clebsch_gordan(two_j1, m1c, two_j2, m2c, two_j, ref);
    sum += prob;
    if (m2c == two_m2 && m1c == two_m1) { cleb += prob; }
  }
  if (sum > 0.0) { cleb /= sum; }
  return cleb;
}

/// `G4Clebsch::GenerateIso3` - REFUSED. See the file header for the four reasons; the two
/// early returns that DO have a well-defined answer are kept, because they are the ones the
/// binary cascade's own pairs reach when one outgoing isospin is zero, and because a caller that
/// gets a refusal for those would be told a function is missing that answered.
struct Iso3Pair {
  double m1 = 0.0;
  double m2 = 0.0;
  bool valid = false;
};

__host__ __device__ inline Iso3Pair clebsch_generate_iso3(int two_j1, int two_m1, int two_j2,
                                                          int two_m2, int two_j_out1,
                                                          int two_j_out2, ClebschRefusal& ref) {
  Iso3Pair out;
  if (two_j1 == 0 && two_j2 == 0) {
    // Geant4 warns and returns (0, 0).
    out.m1 = 0.0;
    out.m2 = 0.0;
    out.valid = true;
    return out;
  }
  const int two_m3 = two_m1 + two_m2;
  if (two_j_out1 == 0) {
    out.m1 = 0.0;
    out.m2 = two_m3;
    out.valid = true;
    return out;
  }
  if (two_j_out2 == 0) {
    out.m1 = two_m3;
    out.m2 = 0.0;
    out.valid = true;
    return out;
  }
  ref.generate_iso3 = true;
  return out;
}

}  // namespace g4gpu::bic::imr

#endif
