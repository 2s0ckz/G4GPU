// Barashenkov nucleon-nucleus total, inelastic and elastic cross sections.
//
// Transcribed from G4ComponentBarNucleonNucleusXsc::ComputeCrossSections and the G4PiData
// interpolation it calls (11.1.1). This is the middle energy band of every nucleon cross
// section QBBC uses - G4BGGNucleonElasticXS and G4BGGNucleonInelasticXS both delegate to it
// between 14 MeV and 91 GeV, which for a proton is essentially the whole range anything is
// transported at.
//
// Two interpolations, in this order, and both are Geant4's:
//
//   1. In energy, linearly, inside one tabulated element's own grid. The grids are not shared
//      across all elements - there are six of them - so the kinks are at different energies
//      for carbon than for iron, and that is a property of the data rather than an artefact.
//   2. In A between the two tabulated elements that bracket the target Z, after scaling each
//      by A^(2/3). Seventeen elements are tabulated, from helium to uranium; everything else
//      is this interpolation, which is why the table covers Z = 2..92 and not seventeen values.
//
// The elastic cross section is total minus inelastic. It is not tabulated anywhere - not here,
// not in Geant4 - and the subtraction is the definition.
//
// What is NOT here, and is refused rather than approximated:
//   * Z = 1. G4BGGNucleonElasticXS sends hydrogen to G4HadronNucleonXsc::HadronNucleonXscNS,
//     a nucleon-nucleon parameterisation, and multiplies by 1.0115. Different function, not a
//     limiting case of this one.
//   * below 14 MeV, where G4BGGNucleonElasticXS switches to a Coulomb-barrier form, and above
//     91 GeV, where it switches to Glauber-Gribov.
// See nucleon_elastic_xs.cuh, which owns those branches and calls this one for the middle.
#pragma once
#include <cmath>

#include "data/barashenkov.hh"
#include "data/materials.cuh"  // atomic_mass, g4pow_a23

namespace g4gpu::hadronic {

template <typename real_t>
struct NucleonNucleusXs {
  real_t total = 0;      ///< mm^2
  real_t inelastic = 0;  ///< mm^2
  real_t elastic = 0;    ///< mm^2
};

/// G4PiData::TotalXSection / ::ReactionXSection: linear in energy, on one element's own grid.
///
/// Geant4 raises a FatalException past the top of the grid (1 TeV) and silently extrapolates
/// below the bottom, because `it==begin()` is nudged forward and the two lowest points are
/// then used as a line. Both edges are reproduced: the low one because it is what Geant4
/// computes, the high one by clamping rather than aborting - a caller that reaches 1 TeV has
/// already been routed to Glauber-Gribov by the dispatcher above, so arriving here is a bug in
/// the dispatcher and returning the endpoint keeps it a diagnosable one.
template <typename real_t>
__host__ __device__ inline real_t barashenkov_interp(const double* e, const double* x, int off,
                                                     int n, real_t kinetic) {
  const double* ee = e + off;
  const double* xx = x + off;
  int i = 0;
  while (i < n && kinetic > static_cast<real_t>(ee[i])) { ++i; }
  if (i >= n) { return static_cast<real_t>(xx[n - 1]); }
  if (i == 0) { i = 1; }
  const real_t e1 = static_cast<real_t>(ee[i - 1]);
  const real_t e2 = static_cast<real_t>(ee[i]);
  const real_t x1 = static_cast<real_t>(xx[i - 1]);
  const real_t x2 = static_cast<real_t>(xx[i]);
  const real_t d = e2 - e1;
  const real_t r = (d != real_t(0)) ? x1 + (kinetic - e1) * (x2 - x1) / d : x1;
  return fmax(real_t(0), r);
}

/// G4ComponentBarNucleonNucleusXsc::Interpolate - between two tabulated elements, in A, with
/// each scaled to the target by A^(2/3).
///
/// A^(2/3), despite Geant4 naming the array A75: the array is filled with `g4pow->A23(...)`
/// and the comment beside it says "interpolate by square ~ A^(2/3)". The name is a fossil.
template <typename real_t>
__host__ __device__ inline real_t barashenkov_z_interp(int z1, int z2, int z, real_t x1,
                                                       real_t x2) {
  const real_t a = data::atomic_mass<real_t>(z);
  const real_t a1 = data::atomic_mass<real_t>(z1);
  const real_t a2 = data::atomic_mass<real_t>(z2);
  const real_t s = data::g4pow_a23<real_t>(a);
  const real_t r1 = x1 * s / data::g4pow_a23<real_t>(a1);
  const real_t r2 = x2 * s / data::g4pow_a23<real_t>(a2);
  const real_t alp1 = a - a1;
  const real_t alp2 = a2 - a;
  const real_t den = alp1 + alp2;
  return (den != real_t(0)) ? (r1 * alp2 + r2 * alp1) / den : r1;
}

/// Total, inelastic and elastic for a nucleon on a nucleus of atomic number @p z.
///
/// @param z        target atomic number, 2..92. Above 92 it is clamped, as Geant4 clamps it;
///                 z = 1 is not this function's - see the file comment.
/// @param kinetic  MeV
/// @param is_proton  selects the proton or the neutron inelastic column. The total is shared.
template <typename real_t>
__host__ __device__ inline NucleonNucleusXs<real_t> barashenkov_xs(int z, real_t kinetic,
                                                                   bool is_proton) {
  NucleonNucleusXs<real_t> out;
  namespace b = data::barashenkov;
  const int* zt = b::z();
  const int* np = b::npoints();
  const int* off = b::offset();
  const double* et = b::energy();
  const double* tt = b::total();
  const double* it_n = b::inelastic_n();
  const double* it_p = b::inelastic_p();
  const double* inel = is_proton ? it_p : it_n;

  if (z < 2) { return out; }
  const int zz = (z > 92) ? 92 : z;

  int i = 0;
  for (; i < b::kNZ; ++i) {
    if (zz <= zt[i]) { break; }
  }
  if (i >= b::kNZ) { i = b::kNZ - 1; }

  if (zt[i] == zz) {
    out.total = barashenkov_interp<real_t>(et, tt, off[i], np[i], kinetic);
    out.inelastic = barashenkov_interp<real_t>(et, inel, off[i], np[i], kinetic);
  } else {
    if (i == 0) { i = 1; }
    const real_t t1 = barashenkov_interp<real_t>(et, tt, off[i - 1], np[i - 1], kinetic);
    const real_t t2 = barashenkov_interp<real_t>(et, tt, off[i], np[i], kinetic);
    const real_t n1 = barashenkov_interp<real_t>(et, inel, off[i - 1], np[i - 1], kinetic);
    const real_t n2 = barashenkov_interp<real_t>(et, inel, off[i], np[i], kinetic);
    out.total = barashenkov_z_interp<real_t>(zt[i - 1], zt[i], zz, t1, t2);
    out.inelastic = barashenkov_z_interp<real_t>(zt[i - 1], zt[i], zz, n1, n2);
  }
  out.elastic = fmax(out.total - out.inelastic, real_t(0));
  return out;
}

}  // namespace g4gpu::hadronic
