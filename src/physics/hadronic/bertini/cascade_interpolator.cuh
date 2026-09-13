// G4CascadeInterpolator: the linear interpolator every table in the INUCL tree is read through.
//
// Transcribed from Geant4 11.1.1:
//   G4CascadeInterpolator::getBin / interpolate
//                                     (cascade/cascade/include/G4CascadeInterpolator.icc)
//
// Its own header calls it "more lightweight than G4PhysicsVector", and the difference matters:
// there is no spline, the bin search is linear from the low end, and the return value of
// `getBin` is a single double whose integer part is the bin index and whose fraction is the
// position inside it - so a value outside the table comes back as a NEGATIVE index or one above
// the last, and `interpolate` then extrapolates from the end bin. That is the default
// (`doExtrapolation = true`) and only one construction in the whole tree turns it off
// (`paraMaker`'s five-point Z scale, in inucl_particle.cuh).
//
// Two details that are easy to lose:
//
//   * The bin search is `for (i=1; i<last && x>xBins[i]; i++)`, which stops at `last` rather
//     than at `last+1`, so a value in the LAST bin lands on `xindex = last-1` with a fraction
//     approaching 1 - it is not a special case. The `i=1` start is the 20100520 fix in the
//     .icc's own history ("Loop in bin search should start at i=1, not i=0, since i-1 is the
//     key"); starting at 0 would index `xBins[-1]`.
//   * `interpolate` clamps the index to `[0, last-1]` and then tests `i == last`, which after
//     the clamp can never hold. The test is the 20100517 fix for "bin position exactly at the
//     upper edge" and the clamp introduced after it made it dead. Both are transcribed: the
//     clamp is what makes the top of the table work, and the dead test is left where it is so
//     that nobody re-derives it as missing.
//
// Geant4 caches the last (x, bin) pair in two `mutable` members. The cache is dropped here. It
// cannot change an answer - `getBin` is a pure function of x - and a mutable member on a table
// shared by every thread in a block is the one thing a device port must not have.
#ifndef G4GPU_BERTINI_CASCADE_INTERPOLATOR_CUH
#define G4GPU_BERTINI_CASCADE_INTERPOLATOR_CUH

namespace g4gpu::physics::hadronic::bert {

/// G4CascadeInterpolator::getBin, with extrapolation on (the default).
__host__ __device__ inline double interp_get_bin(double x, const double* xbins, int nbins) {
  const int last = nbins - 1;
  double xindex, xdiff, xbin;
  if (x < xbins[0]) {
    xindex = 0.0;
    xbin = xbins[1] - xbins[0];
    xdiff = x - xbins[0];                 // negative: extrapolating below the first edge
  } else if (x >= xbins[last]) {
    xindex = double(last);
    xbin = xbins[last] - xbins[last - 1];
    xdiff = x - xbins[last];
  } else {
    int i = 1;
    for (; i < last && x > xbins[i]; ++i) {}
    xindex = double(i - 1);
    xbin = xbins[i] - xbins[i - 1];
    xdiff = x - xbins[i - 1];
  }
  return xindex + xdiff / xbin;
}

/// G4CascadeInterpolator::interpolate(yb), applied to a bin position from `interp_get_bin`.
__host__ __device__ inline double interp_apply(double bin_pos, const double* yb, int nbins) {
  const int last = nbins - 1;
  const int i = (bin_pos < 0.0) ? 0 : (bin_pos > double(last)) ? last - 1 : int(bin_pos);
  const double frac = bin_pos - double(i);
  return (i == last) ? yb[last] : (yb[i] + frac * (yb[i + 1] - yb[i]));
}

__host__ __device__ inline double interp_value(double x, const double* xbins, const double* yb,
                                               int nbins) {
  return interp_apply(interp_get_bin(x, xbins, nbins), yb, nbins);
}

}  // namespace g4gpu::physics::hadronic::bert

#endif  // G4GPU_BERTINI_CASCADE_INTERPOLATOR_CUH
