// G4PhysicsVector evaluation, device-side - because Geant4 does not run the model, it runs a
// table built from the model (docs/RISK.md V5/V7).
//
// Every G4PARTICLEXS data set and the whole of G4NeutronGeneralProcess's combined table are
// G4PhysicsVectors. Reproducing the *numbers in the file* is not enough: what the transport
// reads is `LogVectorValue(e, loge)` or `Value(e)` on one of three vector shapes, each with a
// different bin lookup and each with the spline flag off in this configuration. Get the bin
// lookup wrong and every point between two nodes is wrong while every node is right - which
// is what a coarse-grid test would not see.
//
// Transcribed from G4PhysicsVector.icc and G4PhysicsVector.cc (11.1.1):
//   Interpolation, ComputeLogVectorBin, GetBin, Value(e), LogVectorValue(e, loge)
// and the Initialise() of G4PhysicsVector / G4PhysicsLogVector / G4PhysicsLinearVector.
//
// THE SPLINE IS OFF, IN EVERY VECTOR THIS PACKAGE TOUCHES, AND THAT IS NOT AN ASSUMPTION
//
//   * G4ParticleInelasticXS, G4NeutronInelasticXS, G4NeutronElasticXS and G4NeutronCaptureXS
//     all build their vectors as `new G4PhysicsLogVector()`, whose default argument is
//     `spline = false`, and none of them calls FillSecondDerivatives afterwards.
//   * G4GammaNuclearXS builds `new G4PhysicsLinearVector()` or `new G4PhysicsVector()`, both
//     defaulting to false, and likewise never fills derivatives.
//   * G4UPiNuclearCrossSection sets `spline = false` in its constructor and guards its
//     FillSecondDerivatives call with `if(spline)`.
//   * G4NeutronGeneralProcess passes `false` explicitly to the two G4PhysicsLogVectors it
//     copies its tables from.
//
// So `Interpolation` is a straight line between two nodes, in E and not in log E, and
// secDerivative is never allocated. FillSecondDerivatives is therefore NOT transcribed here;
// a vector that needs it would be a vector this package does not have, and the four
// second-derivative variants (Base / FixedEdges / the simplified one) differ from each other,
// so guessing which applies would be worse than not having it.
//
// A NOTE ON G4Log
//
// The bin of a log vector is found from `loge`, which the callers get from
// `G4DynamicParticle::GetLogKineticEnergy()`, i.e. `G4Log(ekin)`. On Windows - the platform
// the oracle is built on - G4Log.hh reduces to `#define G4Log std::log` and G4Exp.hh to
// `#define G4Exp std::exp`; the VDT Pade expansions are the `#else` branch. So `log` here is
// literally Geant4's function on this platform, not an approximation of it. On Linux it would
// not be, and the difference would show up as a one-bin error at points a few ulps from a
// node - where the interpolated value is the same either way, since the interpolant is
// continuous. Recorded because it is the kind of thing that is invisible until it is not.
#pragma once
#include <cmath>

namespace g4gpu::hadronic::xs {

/// G4PhysicsVectorType, in Geant4's own order and encoding (G4PhysicsVectorType.hh).
enum PhysVecType { kFreeVector = 0, kLogVector = 1, kLinearVector = 2 };

/// A read-only view of one G4PhysicsVector living in a flat table.
///
/// `inv_dbin` and `log_emin` are meaningful only for a log or linear vector, and are what
/// Initialise() computes:
///   log:    invdBin = (idxmax+1)/log(edgeMax/edgeMin),  logemin = log(edgeMin)
///   linear: invdBin = (idxmax+1)/(edgeMax-edgeMin)
/// with idxmax = n - 2.
template <typename real_t>
struct PhysVec {
  const real_t* e = nullptr;  ///< binVector
  const real_t* v = nullptr;  ///< dataVector
  int n = 0;                  ///< numberOfNodes
  real_t edge_min = 0;
  real_t edge_max = 0;
  real_t inv_dbin = 0;
  real_t log_emin = 0;
  int type = kFreeVector;

  __host__ __device__ bool empty() const { return n < 2 || e == nullptr || v == nullptr; }
  /// G4PhysicsVector::GetMaxEnergy / GetMinEnergy.
  __host__ __device__ real_t max_energy() const { return edge_max; }
  __host__ __device__ real_t min_energy() const { return edge_min; }
  /// G4PhysicsVector::Energy(i) and operator[](i).
  __host__ __device__ real_t energy(int i) const { return e[i]; }
  __host__ __device__ real_t value_at(int i) const { return v[i]; }
  __host__ __device__ int length() const { return n; }
};

/// Fills in edge_min/edge_max/inv_dbin/log_emin from the nodes, i.e. runs the Initialise()
/// of whichever vector type is set. Host-side; the device only reads.
template <typename real_t>
__host__ inline void phys_vec_initialise(PhysVec<real_t>& pv) {
  if (pv.n < 2) { return; }
  pv.edge_min = pv.e[0];
  pv.edge_max = pv.e[pv.n - 1];
  const real_t idxmax_plus1 = static_cast<real_t>(pv.n - 1);
  if (pv.type == kLogVector) {
    // G4PhysicsLogVector::Initialise
    pv.inv_dbin = idxmax_plus1 / log(pv.edge_max / pv.edge_min);
    pv.log_emin = log(pv.edge_min);
  } else if (pv.type == kLinearVector) {
    // G4PhysicsLinearVector::Initialise
    pv.inv_dbin = idxmax_plus1 / (pv.edge_max - pv.edge_min);
    pv.log_emin = real_t(0);
  } else {
    pv.inv_dbin = real_t(0);
    pv.log_emin = real_t(0);
  }
}

/// G4PhysicsVector::ComputeLogVectorBin. The truncation is toward zero, on a G4int, and the
/// clamp is to idxmax = n - 2.
template <typename real_t>
__host__ __device__ inline int phys_vec_log_bin(const PhysVec<real_t>& pv, real_t loge) {
  const int idxmax = pv.n - 2;
  const int b = static_cast<int>((loge - pv.log_emin) * pv.inv_dbin);
  return (b < idxmax) ? b : idxmax;
}

/// G4PhysicsVector::GetBin - the three vector types have three different lookups, and the
/// default one is `std::lower_bound(binVector) - begin - 1`.
template <typename real_t>
__host__ __device__ inline int phys_vec_bin(const PhysVec<real_t>& pv, real_t en) {
  const int idxmax = pv.n - 2;
  if (pv.type == kLogVector) { return phys_vec_log_bin(pv, log(en)); }
  if (pv.type == kLinearVector) {
    const int b = static_cast<int>((en - pv.edge_min) * pv.inv_dbin);
    return (b < idxmax) ? b : idxmax;
  }
  // lower_bound: first index with e[i] >= en, then minus one.
  int lo = 0;
  int hi = pv.n;  // one past the end, as lower_bound's range is [begin, end)
  while (lo < hi) {
    const int mid = lo + (hi - lo) / 2;
    if (pv.e[mid] < en) { lo = mid + 1; } else { hi = mid; }
  }
  return lo - 1;
}

/// G4PhysicsVector::Interpolation, with the spline term left out because useSpline is false
/// everywhere in this package - see the file header.
///
/// Written in Geant4's own order of operations. `y1 + b*dy` and `y1*(1-b) + y2*b` are not the
/// same double.
template <typename real_t>
__host__ __device__ inline real_t phys_vec_interpolate(const PhysVec<real_t>& pv, int idx,
                                                       real_t en) {
  const real_t x1 = pv.e[idx];
  const real_t dl = pv.e[idx + 1] - x1;
  const real_t y1 = pv.v[idx];
  const real_t dy = pv.v[idx + 1] - y1;
  const real_t b = (en - x1) / dl;
  return y1 + b * dy;
}

/// G4PhysicsVector::Value(G4double e) - the one-argument overload, no cached index.
template <typename real_t>
__host__ __device__ inline real_t phys_vec_value(const PhysVec<real_t>& pv, real_t en) {
  if (pv.empty()) { return real_t(0); }
  if (en > pv.edge_min && en < pv.edge_max) {
    return phys_vec_interpolate(pv, phys_vec_bin(pv, en), en);
  }
  return (en <= pv.edge_min) ? pv.v[0] : pv.v[pv.n - 1];
}

/// G4PhysicsVector::Value(G4double e, std::size_t& idx) - the two-argument overload, where
/// the caller supplies a starting index and gets it updated.
///
/// THE HINT CHANGES THE ANSWER, IN THE LAST BITS, AND EXACTLY AT A GRID POINT
///
/// It reads as pure memoisation, and it is not. When `e` lands exactly on node k and the hint
/// says bin k, the interpolation returns `v[k] + 0*dy`, i.e. v[k] exactly. When the hint
/// misses and GetBin runs, `std::lower_bound` returns k, so the bin is k-1 and the
/// interpolation returns `v[k-1] + 1*(v[k] - v[k-1])` - which is v[k] to within a rounding
/// and not bit-for-bit. G4UPiNuclearCrossSection::Interpolate hands in
/// `(size_t)(max(ekin - 20 MeV, 0)*0.06)`, which for a 100 MeV pion is bin 4 and 100 MeV is
/// node 4 of its energy grid, so this is not a hypothetical: it is a live difference of a few
/// ulps at every tabulated energy, and reproducing the hint is cheaper than explaining it.
///
/// @param idx  in and out, as Geant4's reference parameter.
template <typename real_t>
__host__ __device__ inline real_t phys_vec_value_cached(const PhysVec<real_t>& pv, real_t en,
                                                        int& idx) {
  if (pv.empty()) { return real_t(0); }
  if (idx + 1 < pv.n && en >= pv.e[idx] && en <= pv.e[idx + 1]) {
    return phys_vec_interpolate(pv, idx, en);
  }
  if (en > pv.edge_min && en < pv.edge_max) {
    idx = phys_vec_bin(pv, en);
    return phys_vec_interpolate(pv, idx, en);
  }
  if (en <= pv.edge_min) {
    idx = 0;
    return pv.v[0];
  }
  idx = pv.n - 2;
  return pv.v[pv.n - 1];
}

/// G4PhysicsVector::LogVectorValue(e, loge) - the fast path every G4PARTICLEXS reader and
/// G4NeutronGeneralProcess uses, where the caller has already computed log(e).
///
/// Note that it does NOT check the vector's type: it computes a log bin unconditionally, so
/// calling it on a free vector reads inv_dbin and log_emin that Initialise never set. Geant4
/// has the same hazard and avoids it the same way, by only calling it on log vectors.
template <typename real_t>
__host__ __device__ inline real_t phys_vec_log_value(const PhysVec<real_t>& pv, real_t en,
                                                     real_t loge) {
  if (pv.empty()) { return real_t(0); }
  if (en > pv.edge_min && en < pv.edge_max) {
    return phys_vec_interpolate(pv, phys_vec_log_bin(pv, loge), en);
  }
  return (en <= pv.edge_min) ? pv.v[0] : pv.v[pv.n - 1];
}

}  // namespace g4gpu::hadronic::xs
