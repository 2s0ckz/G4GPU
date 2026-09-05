// G4Pow's cube root, transcribed - because it is an approximation, not a cube root.
//
// Geant4 computes A^(1/3) through G4Pow::A13, which is a third-order Taylor expansion of the
// cube root about a tabulated nearby point. It is not std::cbrt and it does not agree with it:
// the neglected term is about -(10/3)x^4 with |x| <= 0.0417 on the domain used here, so
// `std::pow(a, 1.0/3.0)` differs from Geant4's answer by up to ~1e-5 relative.
//
// That matters in exactly one place so far. G4ionEffectiveCharge's heavy-ion branch computes
// the screening length from `g4calc->A23(1.0 - q)`, and the charge correction it produces is
// compared against Geant4's own at 1e-9. With an exact power the comparison fails at 1e-5 and
// the failure looks like a transcription error in Ziegler's formula, which it is not.
//
// So the approximation is reproduced rather than improved on. The two tables G4Pow holds are
// `pz13[i] = std::pow(i, 1/3)` and `lowa13[i] = std::pow(0.25*i, 1/3)`, both plain powers of
// exactly representable arguments, so they are computed here instead of tabulated - the values
// are identical and there is no table to get out of step with Geant4.
//
// Transcribed from G4Pow::A13, A13Low and A13High (11.1.1).
#ifndef G4GPU_DATA_G4POW_HH
#define G4GPU_DATA_G4POW_HH

#include <cmath>

namespace g4gpu::data {

/// G4Pow::A13Low - the branch for 1 <= a < 4, expanding about a quarter-integer.
template <typename real_t>
__host__ __device__ inline real_t g4pow_a13_low(real_t a, bool invert) {
  const int i = static_cast<int>(real_t(4) * (a + real_t(0.125)));
  const real_t y = real_t(0.25) * static_cast<real_t>(i);
  const real_t x = (a / y - real_t(1)) / real_t(3);
  // lowa13[i], which Geant4 fills as std::pow(0.25*i, 1/3).
  const real_t base = pow(y, real_t(1) / real_t(3));
  const real_t res = base * (real_t(1) + x - x * x * (real_t(1) - real_t(1.666667) * x));
  return invert ? real_t(1) / res : res;
}

/// G4Pow::A13High - the branch for a >= 4, expanding about the nearest integer.
template <typename real_t>
__host__ __device__ inline real_t g4pow_a13_high(real_t a, bool invert) {
  // maxA = -0.6 + maxZ with maxZ = 512, i.e. 511.4: above it Geant4 gives up on the table.
  constexpr real_t kMaxA = real_t(511.4);
  real_t res;
  if (a < kMaxA) {
    const int i = static_cast<int>(a + real_t(0.5));
    const real_t x = (a / static_cast<real_t>(i) - real_t(1)) / real_t(3);
    // pz13[i], which Geant4 fills as std::pow(i, 1/3).
    const real_t base = pow(static_cast<real_t>(i), real_t(1) / real_t(3));
    res = base * (real_t(1) + x - x * x * (real_t(1) - real_t(1.666667) * x));
  } else {
    res = exp(log(a) / real_t(3));
  }
  return invert ? real_t(1) / res : res;
}

/// G4Pow::A13. Zero and below give zero, as in Geant4.
template <typename real_t>
__host__ __device__ inline real_t g4pow_a13(real_t a) {
  if (a <= real_t(0)) { return real_t(0); }
  const bool invert = (a < real_t(1));
  const real_t x = invert ? real_t(1) / a : a;
  constexpr real_t kMaxLowA = real_t(4);
  return (x < kMaxLowA) ? g4pow_a13_low(x, invert) : g4pow_a13_high(x, invert);
}

/// G4Pow::A23, which is A13 squared - not a two-thirds power.
template <typename real_t>
__host__ __device__ inline real_t g4pow_a23(real_t a) {
  const real_t x = g4pow_a13(a);
  return x * x;
}

/// G4Pow::Z13, which is `pz13[Z]` - a plain cube root of an integer, unlike A13.
template <typename real_t>
__host__ __device__ inline real_t g4pow_z13(int z) {
  return (z <= 0) ? real_t(0) : pow(static_cast<real_t>(z), real_t(1) / real_t(3));
}

// ---------------------------------------------------------------- powA
//
// G4Pow::powA is not std::pow either, and for the same reason A13 is not std::cbrt: it is a
// Taylor expansion about a tabulated point, chosen for speed in 2010 and now part of the
// answer Geant4 gives. `powA(a, y) = expA(y * logX(a))`, with expA and logX each their own
// third-order expansion.
//
// The two tables G4Pow holds for these are `lz[i] = log(i)` and `fexp[i] = exp(0.5*i)`, both
// of exactly representable integer arguments, so they are computed here rather than tabulated
// - Geant4 fills them with G4Log/G4Exp, which agree with std::log/std::exp to within an ulp,
// and there is then no 682-entry table to drift out of step.
//
// This is used by G4IonFluctuations::Factor for the Yang straggling coefficients, where the
// difference from std::pow is about 1e-7 relative - far below the physics, far above a
// tolerance that could otherwise distinguish a transcription error from a rounding one.

/// G4Pow::logBase - log of an argument already folded into [1, 511.4].
template <typename real_t>
__host__ __device__ inline real_t g4pow_log_base(real_t a) {
  constexpr real_t kOneThird = real_t(1) / real_t(3);
  constexpr int kMax2 = 5;
  constexpr real_t kMaxA2 = real_t(1.25) + real_t(kMax2) * real_t(0.2);  // 2.25
  constexpr real_t kMaxA = real_t(-0.6) + real_t(512);                   // 511.4

  if (a <= kMaxA2) {
    int i = static_cast<int>(real_t(kMax2) * (a - real_t(1)) + real_t(0.5));
    if (i > kMax2) { i = kMax2; }
    // lz2[i] = log(1 + 0.2*i)
    const real_t lz2 = log(real_t(1) + real_t(i) * real_t(0.2));
    const real_t x = a / (real_t(i) / real_t(kMax2) + real_t(1)) - real_t(1);
    return lz2 + x * (real_t(1) - (real_t(0.5) - kOneThird * x) * x);
  }
  if (a <= kMaxA) {
    const int i = static_cast<int>(a + real_t(0.5));
    const real_t x = a / real_t(i) - real_t(1);
    return log(real_t(i)) + x * (real_t(1) - (real_t(0.5) - kOneThird * x) * x);
  }
  return log(a);
}

/// G4Pow::logX - log for any positive argument, folding x < 1 and x > 511.4 onto logBase.
template <typename real_t>
__host__ __device__ inline real_t g4pow_log_x(real_t x) {
  constexpr real_t kMaxA = real_t(-0.6) + real_t(512);
  const real_t a = (x >= real_t(1)) ? x : real_t(1) / x;
  // ener[i] = 500^i. Geant4 folds by ener[1] and by ener[2], then gives up and calls G4Log.
  // Written as that chain rather than as a loop over the array: the loop version was here
  // first and kept folding past ener[3], where Geant4 does not. Nothing reaches an argument
  // that large today, so the difference showed up in no test - which is why it is written out
  // in the shape of the original instead of the shape that generalises.
  constexpr real_t kEner1 = real_t(500);
  constexpr real_t kEner2 = kEner1 * real_t(500);  // 2.5e5
  constexpr real_t kEner3 = kEner2 * real_t(500);  // 1.25e8
  real_t res;
  if (a <= kMaxA) {
    res = g4pow_log_base(a);
  } else if (a <= kEner2) {
    res = log(kEner1) + g4pow_log_base(a / kEner1);
  } else if (a <= kEner3) {
    res = log(kEner2) + g4pow_log_base(a / kEner2);
  } else {
    res = log(a);
  }
  return (x < real_t(1)) ? -res : res;
}

/// G4Pow::expA.
template <typename real_t>
__host__ __device__ inline real_t g4pow_exp_a(real_t A) {
  constexpr real_t kOneThird = real_t(1) / real_t(3);
  constexpr real_t kMaxAexp = real_t(-0.76) + real_t(170) * real_t(0.5);  // 84.24
  const real_t a = (A >= real_t(0)) ? A : -A;
  real_t res;
  if (a <= kMaxAexp) {
    const int i = static_cast<int>(real_t(2) * a + real_t(0.5));
    const real_t x = a - real_t(i) * real_t(0.5);
    // fexp[i] = exp(0.5*i)
    res = exp(real_t(0.5) * real_t(i))
          * (real_t(1) + x * (real_t(1) + real_t(0.5) * (real_t(1) + kOneThird * x) * x));
  } else {
    res = exp(a);
  }
  return (A < real_t(0)) ? real_t(1) / res : res;
}

/// G4Pow::powA. Zero base gives zero, as in Geant4 - including for a zero or negative
/// exponent, where std::pow would give 1 or infinity.
template <typename real_t>
__host__ __device__ inline real_t g4pow_pow_a(real_t a, real_t y) {
  return (a == real_t(0)) ? real_t(0) : g4pow_exp_a<real_t>(y * g4pow_log_x<real_t>(a));
}

}  // namespace g4gpu::data

#endif  // G4GPU_DATA_G4POW_HH
