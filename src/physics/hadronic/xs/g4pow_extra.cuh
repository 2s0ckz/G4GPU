// Two more G4Pow entry points, for the hadronic cross sections that call them.
//
// data/g4pow.hh has A13, A23, Z13, logX, expA and powA - the ones the EM port needed. The
// hadronic cross sections reach for two it does not have, and both differ from the obvious
// standard-library expression in ways that matter at this tolerance:
//
//   powZ(Z, y) = expA(y * lz[Z])   with lz[Z] = G4Log(Z), the *exact* log of an integer.
//                powA(A, y) = expA(y * logX(A)) instead runs logX's third-order expansion,
//                so powZ and powA disagree at ~1e-7 for the same numeric argument. Geant4
//                calls powZ from G4NuclearRadii::Radius (A^0.27) and RadiusRMS (A^0.28) and
//                powA from G4UPiNuclearCrossSection (A^0.75). Swapping them is a 1e-7 error
//                in a number the oracle is compared to at 1e-12.
//
//   powN(x, n)  is repeated multiplication for |n| <= 8 and std::pow beyond. Repeated
//                multiplication is not std::pow: for x^7 the two differ in the last bits, and
//                G4HadronNucleonXsc::HadronNucleonXscNS takes a square root of powN(-x, 7),
//                which halves the exponent of the difference but keeps it well above 1e-12.
//
// They live here rather than in data/g4pow.hh because that file is shared and this package's
// rule is that new code goes in the directory it owns (docs/HADRONIC_PLAN.md section 6 rule
// 5). If a second package needs them, moving them up is a one-line change.
//
// Transcribed from G4Pow.hh (powZ, inline) and G4Pow.cc (powN), 11.1.1.
#pragma once
#include <cmath>

#include "data/g4pow.hh"

namespace g4gpu::hadronic::xs {

/// G4Pow::powZ - expA(y*log(Z)) with an exact log of the integer.
template <typename real_t>
__host__ __device__ inline real_t g4pow_pow_z(int z, real_t y) {
  return data::g4pow_exp_a<real_t>(y * log(static_cast<real_t>(z)));
}

/// G4Pow::powN - repeated multiplication up to |n| = 8, then std::pow. Zero base gives zero,
/// including for n = 0, where std::pow would give one.
template <typename real_t>
__host__ __device__ inline real_t g4pow_pow_n(real_t x, int n) {
  if (x == real_t(0)) { return real_t(0); }
  const int an = (n < 0) ? -n : n;
  if (an > 8) { return pow(x, static_cast<real_t>(n)); }
  real_t res = real_t(1);
  if (n >= 0) {
    for (int i = 0; i < n; ++i) { res *= x; }
  } else {
    const real_t y = real_t(1) / x;
    for (int i = 0; i < an; ++i) { res *= y; }
  }
  return res;
}

/// G4Pow::Z23 - Z13 squared, for an integer argument.
template <typename real_t>
__host__ __device__ inline real_t g4pow_z23(int z) {
  const real_t x = data::g4pow_z13<real_t>(z);
  return x * x;
}

}  // namespace g4gpu::hadronic::xs
