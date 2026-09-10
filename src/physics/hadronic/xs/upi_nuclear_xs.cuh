// G4UPiNuclearCrossSection - Barashenkov's pion-nucleus elastic and inelastic cross sections,
// sixteen tabulated elements and an A^0.75 interpolation between them.
//
// Transcribed from source/processes/hadronic/cross_sections/src/G4UPiNuclearCrossSection.cc
// (11.1.1): Interpolate, AddDataSet, BuildPhysicsTable's index construction, and the
// GetElasticCrossSection / GetInelasticCrossSection inlines from the header. The data are in
// data/upi_nuclear.hh.
//
// This is the middle band - 20 MeV to 91 GeV - of QBBC's pi+ and pi- elastic and inelastic
// cross sections, under G4BGGPionElasticXS and G4BGGPionInelasticXS.
//
// HOW IT DIFFERS FROM THE NUCLEON VERSION, WHICH IS THE SAME AUTHOR'S SAME IDEA
//
// data/barashenkov.hh (nucleons) is interpolated in A with each neighbour scaled by A^(2/3);
// this one scales by A^0.75 and interpolates linearly in A with the weight computed from the
// *tabulated* masses. The nucleon version interpolates in energy on each element's own grid
// with its own linear search; this one goes through a G4PhysicsFreeVector, so the search is a
// binary one and the starting-index hint is part of the answer (see phys_vec_value_cached).
// Neither is a special case of the other, and the two files are the two halves of the same
// 1989 preprint.
//
// THREE EDGES
//
// * `Z = 1 is not applicable`. G4UPiNuclearCrossSection::IsElementApplicable is `1 < Z`, and
//   Interpolate would read theZ[idx-1] with idx = idxZ[1] = 0 - one before the front of a
//   16-element array. The BGG pion classes never get here for hydrogen because they answer
//   `1 == Z` from G4HadronNucleonXsc first. Refused by name rather than clamped, because a
//   clamp would invent a hydrogen cross section this model does not have.
// * `2 == iz` is tested alongside `idx < 0` because idxZ[2] is 0, which is not negative: the
//   index-building loop starts at Z = 3. Helium therefore takes the exact-element branch
//   through `table[abs(0)]`, and so would Z = 1 if it ever arrived, at the wrong element.
// * Below 20 MeV the energy is clamped to 20 MeV, which is also the tables' first node, so
//   the value is the first node's - flat, not extrapolated. Above 1 TeV it is the last node's.
#pragma once
#include <cmath>

#include "core/units.cuh"
#include "data/materials.cuh"
#include "data/upi_nuclear.hh"
#include "physics/hadronic/xs/g4pow_extra.cuh"
#include "physics/hadronic/xs/physics_vector.cuh"
#include "physics/hadronic/xs/projectile.cuh"
#include "physics/hadronic/xs/refusal.cuh"

namespace g4gpu::hadronic::xs {

/// `elow` in the constructor - the lower clamp, and the tables' own first node.
template <typename real_t> __host__ __device__ constexpr real_t upi_elow() {
  return real_t(20.0) * units::MeV<real_t>();
}
/// `aPower` in the constructor.
template <typename real_t> __host__ __device__ constexpr real_t upi_apower() {
  return real_t(0.75);
}

/// What BuildPhysicsTable computes on top of the tables: theA[16], APower[93] and idxZ[93].
template <typename real_t>
struct UPiNuclearTable {
  real_t theA[data::upi::kNZ] = {real_t(0)};  ///< GetAtomicMassAmu of each tabulated Z
  real_t apower[93] = {real_t(0)};            ///< powA(GetAtomicMassAmu(Z), 0.75)
  int idxz[93] = {0};                         ///< negative for a tabulated Z, see the header
};

/// G4UPiNuclearCrossSection::BuildPhysicsTable, the part that is not loading data.
template <typename real_t>
__host__ inline void upi_build_table(UPiNuclearTable<real_t>& t) {
  const int* zt = data::upi::z();
  for (int i = 0; i < data::upi::kNZ; ++i) {
    t.theA[i] = data::atomic_mass<real_t>(zt[i]);
  }
  for (int i = 1; i < 93; ++i) {
    // powA, not powZ: Geant4 writes g4pow->powA(nist->GetAtomicMassAmu(i), aPower), and the
    // argument is a non-integer mass. See g4pow_extra.cuh for why the distinction matters.
    t.apower[i] = data::g4pow_pow_a<real_t>(data::atomic_mass<real_t>(i), upi_apower<real_t>());
  }
  int idx = 1;
  for (int i = 3; i < 93; ++i) {
    if (zt[idx] == i) {
      t.idxz[i] = -idx;
      ++idx;
    } else {
      t.idxz[i] = idx;
    }
  }
  // idxz[0], idxz[1] and idxz[2] are never assigned - they stay at zero, and the `2 == iz`
  // test in Interpolate is what keeps helium from being treated as an interpolation.
}

/// Which of the four G4PhysicsTables to read.
enum class UPiChannel { kElastic, kInelastic };

/// One element's G4PhysicsFreeVector, as AddDataSet built it: energies `e[i]*GeV` and values
/// `in[i]*millibarn` (inelastic) or `max(0,(tot[i]-in[i]))*millibarn` (elastic).
///
/// Built here rather than stored, so that the two bracketing nodes are converted with exactly
/// the expression AddDataSet used before the interpolation sees them - see data/upi_nuclear.hh.
template <typename real_t>
__host__ __device__ inline real_t upi_node_value(int off, int i, bool is_piplus,
                                                 UPiChannel ch) {
  const double* tot = is_piplus ? data::upi::pip_total_mb() : data::upi::pim_total_mb();
  const double* in = is_piplus ? data::upi::pip_inelastic_mb() : data::upi::pim_inelastic_mb();
  const real_t inel = static_cast<real_t>(in[off + i]);
  if (ch == UPiChannel::kInelastic) { return inel * millibarn<real_t>(); }
  const real_t d = static_cast<real_t>(tot[off + i]) - inel;
  return ((d > real_t(0)) ? d : real_t(0)) * millibarn<real_t>();
}

/// G4PhysicsVector::Value(e, idx) over one element's converted nodes.
///
/// The nodes are generated on the fly, so this is phys_vec_value_cached written out rather
/// than called: building a PhysVec would need somewhere to put 39 doubles.
template <typename real_t>
__host__ __device__ inline real_t upi_vector_value(int iel, bool is_piplus, UPiChannel ch,
                                                   real_t en, int& idx) {
  const int off = data::upi::offset()[iel];
  const int n = data::upi::npoints()[iel];
  const double* eg = data::upi::energy_GeV();
  auto node_e = [&](int i) {
    return static_cast<real_t>(eg[off + i]) * units::GeV<real_t>();
  };
  auto node_v = [&](int i) { return upi_node_value<real_t>(off, i, is_piplus, ch); };

  const real_t edge_min = node_e(0);
  const real_t edge_max = node_e(n - 1);

  auto interp = [&](int i) {
    const real_t x1 = node_e(i);
    const real_t dl = node_e(i + 1) - x1;
    const real_t y1 = node_v(i);
    const real_t dy = node_v(i + 1) - y1;
    const real_t b = (en - x1) / dl;
    return y1 + b * dy;
  };

  if (idx + 1 < n && en >= node_e(idx) && en <= node_e(idx + 1)) { return interp(idx); }
  if (en > edge_min && en < edge_max) {
    // T_G4PhysicsFreeVector: lower_bound(binVector, e) - begin - 1.
    int lo = 0, hi = n;
    while (lo < hi) {
      const int mid = lo + (hi - lo) / 2;
      if (node_e(mid) < en) { lo = mid + 1; } else { hi = mid; }
    }
    idx = lo - 1;
    return interp(idx);
  }
  if (en <= edge_min) {
    idx = 0;
    return node_v(0);
  }
  idx = n - 2;
  return node_v(n - 1);
}

/// G4UPiNuclearCrossSection::Interpolate.
///
/// @param A  the target mass number the caller has decided on - the BGG pion classes pass
///           theA[Z] = G4lrint(GetAtomicMassAmu(Z)), an integer, not aeff[Z].
template <typename real_t>
__host__ __device__ inline XsValue<real_t> upi_interpolate(const UPiNuclearTable<real_t>& t,
                                                           UPiChannel ch, bool is_piplus,
                                                           int Z, int A, real_t e) {
  if (Z < 2) { return {real_t(0), XsRefusal::kUPiNuclearHydrogen}; }
  const real_t ekin = (e > upi_elow<real_t>()) ? e : upi_elow<real_t>();
  const int iz = (Z < 92) ? Z : 92;
  const int idx = t.idxz[iz];
  const real_t d = ekin - upi_elow<real_t>();
  int jdx = static_cast<int>(((d > real_t(0)) ? d : real_t(0)) * real_t(0.06));

  const int* zt = data::upi::z();
  if (idx < 0 || 2 == iz) {
    const int iel = (idx < 0) ? -idx : idx;
    return {upi_vector_value<real_t>(iel, is_piplus, ch, ekin, jdx), XsRefusal::kNone};
  }
  // Two neighbours, each scaled from its own A to the target's by A^0.75, then a linear
  // interpolation in A. jdx is passed by reference to both, as the one `size_t jdx` is in
  // Geant4, so the second lookup starts from wherever the first one left it.
  const int iz2 = zt[idx];
  const real_t x2 =
      upi_vector_value<real_t>(idx, is_piplus, ch, ekin, jdx) * t.apower[iz] / t.apower[iz2];
  const int iz1 = zt[idx - 1];
  const real_t x1 = upi_vector_value<real_t>(idx - 1, is_piplus, ch, ekin, jdx)
                    * t.apower[iz] / t.apower[iz1];
  const real_t w1 =
      (static_cast<real_t>(A) - t.theA[idx - 1]) / (t.theA[idx] - t.theA[idx - 1]);
  return {w1 * x2 + (real_t(1.0) - w1) * x1, XsRefusal::kNone};
}

/// G4UPiNuclearCrossSection::GetElasticCrossSection.
template <typename real_t>
__host__ __device__ inline XsValue<real_t> upi_elastic(const UPiNuclearTable<real_t>& t,
                                                       const Projectile<real_t>& p, int Z,
                                                       int A, real_t ekin) {
  return upi_interpolate<real_t>(t, UPiChannel::kElastic, p.pdg == pdg::kPiPlus, Z, A, ekin);
}

/// G4UPiNuclearCrossSection::GetInelasticCrossSection.
template <typename real_t>
__host__ __device__ inline XsValue<real_t> upi_inelastic(const UPiNuclearTable<real_t>& t,
                                                         const Projectile<real_t>& p, int Z,
                                                         int A, real_t ekin) {
  return upi_interpolate<real_t>(t, UPiChannel::kInelastic, p.pdg == pdg::kPiPlus, Z, A, ekin);
}

}  // namespace g4gpu::hadronic::xs
