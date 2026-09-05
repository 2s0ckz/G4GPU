// Geant4's cubic spline over a tabulated function, so that a table read from Geant4 gives
// Geant4's number between its points as well as at them.
//
// Every table Geant4 loads into a G4PhysicsVector with `spline = true` is evaluated with this,
// and the tables where it matters most are the ones with a peak on a coarse grid: PSTAR's
// proton stopping power is 60 points from 1 keV to 10 MeV across the Bragg peak, where linear
// interpolation is visibly low in the curvature and visibly high on the flanks. Reproducing
// Geant4 at the grid points and not between them is not reproducing Geant4.
//
// Two pieces, and they belong on different sides of the machine:
//
//   fill_second_derivatives  a tridiagonal solve. Host only, once, at table-build time.
//   spline_value             three multiplies more than the linear form. Host and device.
//
// The endpoint condition is "not-a-knot", from G4PhysicsVector::ComputeSecDerivative1, which
// is what G4SplineType::Base selects and what every stopping-power table therefore uses. It is
// transcribed rather than reimplemented: a natural spline (second derivative zero at the ends)
// is the textbook default, is what one would write from memory, and differs from Geant4 by a
// few per cent in the first and last intervals.
#pragma once
#include <cmath>
#include <vector>

namespace g4gpu::data {

/// Second derivatives for a not-a-knot cubic spline through (@p x, @p y), @p n points.
///
/// Transcribed from `G4PhysicsVector::ComputeSecDerivative1`, including its use of the output
/// array as scratch during the decomposition - which is why the back-substitution reads
/// `d2[k + 1]` before writing `d2[k]`, and why the order of the last four statements matters.
///
/// Needs at least 5 points, as Geant4 does; with fewer it fills zeros, which makes
/// spline_value fall back to exactly the linear interpolation Geant4 falls back to.
inline void fill_second_derivatives(const double* x, const double* y, int n, double* d2) {
  for (int i = 0; i < n; ++i) { d2[i] = 0.0; }
  if (n < 5) { return; }

  const int last = n - 1;  // Geant4's `n`, which is the last index rather than the count
  std::vector<double> u(static_cast<std::size_t>(n), 0.0);

  u[1] = ((y[2] - y[1]) / (x[2] - x[1]) - (y[1] - y[0]) / (x[1] - x[0]));
  u[1] = 6.0 * u[1] * (x[2] - x[1]) / ((x[2] - x[0]) * (x[2] - x[0]));

  d2[1] = (2.0 * x[1] - x[0] - x[2]) / (2.0 * x[2] - x[0] - x[1]);

  for (int i = 2; i < last - 1; ++i) {
    const double sig = (x[i] - x[i - 1]) / (x[i + 1] - x[i - 1]);
    const double p = sig * d2[i - 1] + 2.0;
    d2[i] = (sig - 1.0) / p;
    u[i] = (y[i + 1] - y[i]) / (x[i + 1] - x[i]) - (y[i] - y[i - 1]) / (x[i] - x[i - 1]);
    u[i] = (6.0 * u[i] / (x[i + 1] - x[i - 1])) - sig * u[i - 1] / p;
  }

  double sig = (x[last - 1] - x[last - 2]) / (x[last] - x[last - 2]);
  const double p_end = sig * d2[last - 3] + 2.0;
  u[last - 1] = (y[last] - y[last - 1]) / (x[last] - x[last - 1])
                - (y[last - 1] - y[last - 2]) / (x[last - 1] - x[last - 2]);
  u[last - 1] = 6.0 * sig * u[last - 1] / (x[last] - x[last - 2])
                - (2.0 * sig - 1.0) * u[last - 2] / p_end;

  const double p = (1.0 + sig) + (2.0 * sig - 1.0) * d2[last - 2];
  d2[last - 1] = u[last - 1] / p;

  for (int k = last - 2; k > 1; --k) {
    d2[k] *= (d2[k + 1] - u[k] * (x[k + 1] - x[k - 1]) / (x[k + 1] - x[k]));
  }
  d2[last] = (d2[last - 1] - (1.0 - sig) * d2[last - 2]) / sig;
  sig = 1.0 - ((x[2] - x[1]) / (x[2] - x[0]));
  d2[1] *= (d2[2] - u[1] / (1.0 - sig));
  d2[0] = (d2[1] - sig * d2[2]) / (1.0 - sig);
}

/// The bin containing @p e: the largest i with x[i] <= e, clamped to [0, n-2].
///
/// A binary search rather than Geant4's log-scale index, which is a precomputed lookup table
/// on top of one. The tables this is used for are 60 to 78 points, so the search is six
/// comparisons; the answer is the same index either way, and that is what has to match.
template <typename real_t>
__host__ __device__ inline int spline_bin(const real_t* x, int n, real_t e) {
  if (n < 2) { return 0; }
  if (e <= x[0]) { return 0; }
  if (e >= x[n - 1]) { return n - 2; }
  int lo = 0, hi = n - 1;
  while (hi - lo > 1) {
    const int mid = (lo + hi) / 2;
    if (x[mid] <= e) { lo = mid; } else { hi = mid; }
  }
  return lo;
}

/// The spline through (@p x, @p y) with second derivatives @p d2, evaluated at @p e.
///
/// Verbatim from `G4PhysicsVector::Interpolation`. Outside the table it returns the end value,
/// because b saturates at 0 or 1 and the cubic term vanishes there - which is Geant4's
/// behaviour and is why it has no explicit clamp.
template <typename real_t>
__host__ __device__ inline real_t spline_value(const real_t* x, const real_t* y,
                                               const real_t* d2, int n, real_t e) {
  if (n <= 0) { return real_t(0); }
  if (n == 1) { return y[0]; }
  const int idx = spline_bin(x, n, e);
  const real_t x1 = x[idx];
  const real_t dl = x[idx + 1] - x1;
  const real_t y1 = y[idx];
  const real_t dy = y[idx + 1] - y1;
  if (dl <= real_t(0)) { return y1; }
  real_t b = (e - x1) / dl;
  if (b < real_t(0)) { b = real_t(0); }
  if (b > real_t(1)) { b = real_t(1); }
  real_t res = y1 + b * dy;
  if (d2 != nullptr) {
    const real_t c0 = (real_t(2) - b) * d2[idx];
    const real_t c1 = (real_t(1) + b) * d2[idx + 1];
    res += (b * (b - real_t(1))) * (c0 + c1) * (dl * dl * (real_t(1) / real_t(6)));
  }
  return res;
}

}  // namespace g4gpu::data
