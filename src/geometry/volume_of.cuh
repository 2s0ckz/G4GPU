// The volume of a solid, in mm^3.
//
// Needed to turn an energy deposit into a dose, which is the one place a geometric quantity
// feeds directly into a reported physics number - example B1's answer is edep divided by the
// mass of Shape2, and a 1% error in that volume is a 1% error in the dose.
//
// Closed forms where they exist. Monte Carlo integration where they do not, which is every
// boolean solid and the swept solids with an inner radius: a subtraction has no analytic
// volume, and estimating one by subtracting the operands' volumes is wrong whenever they only
// partly overlap. The estimator reports its own statistical uncertainty so that a caller can
// see when it is not good enough, rather than being handed a number with no error bar.
#pragma once
#include <cmath>
#include "core/rng.cuh"
#include "geometry/solids.cuh"

namespace g4gpu::geom {

/// Closed-form volume, or a negative number when there is not one.
template <typename real_t>
__host__ inline real_t analytic_volume(const Solid<real_t>& s) {
  const real_t* p = s.p;
  const real_t pi = units::pi<real_t>();
  const real_t twopi = units::twopi<real_t>();
  switch (s.type) {
    case SolidType::kMesh:
      // Computed on the host when the mesh was closed, by the divergence theorem: exact for a
      // closed mesh, and cheaper and more accurate than sampling it. See geometry/bvh_build.hh.
      return s.p[6];
    case SolidType::kBox:
    case SolidType::kVoxelGrid:   // a box that happens to hold a material per cell
      return real_t(8) * p[0] * p[1] * p[2];
    case SolidType::kTrd:
      // The prismatoid formula: (h/3)(A1 + A2 + sqrt(A1 A2)) does not apply, because the
      // cross-section is a rectangle whose sides vary independently. Integrating
      // 4 x(z) y(z) over z gives this.
      return (real_t(2) * p[4] / real_t(3))
             * (real_t(4) * p[0] * p[2] + real_t(4) * p[1] * p[3]
                + real_t(2) * (p[0] * p[3] + p[1] * p[2]));
    case SolidType::kCons: {
      // Solid cone frustum, full phi: integral of pi r(z)^2.
      const real_t r1 = p[0], r2 = p[1], dz = p[2];
      return (pi * real_t(2) * dz / real_t(3)) * (r1 * r1 + r1 * r2 + r2 * r2);
    }
    case SolidType::kTubs: {
      const real_t frac = p[4] / twopi;
      return frac * pi * (p[1] * p[1] - p[0] * p[0]) * real_t(2) * p[2];
    }
    case SolidType::kConeSection: {
      const real_t frac = p[6] / twopi;
      const real_t dz = p[4];
      const real_t outer = (pi * real_t(2) * dz / real_t(3))
                           * (p[1] * p[1] + p[1] * p[3] + p[3] * p[3]);
      const real_t inner = (pi * real_t(2) * dz / real_t(3))
                           * (p[0] * p[0] + p[0] * p[2] + p[2] * p[2]);
      return frac * (outer - inner);
    }
    case SolidType::kOrb:
      return (real_t(4) / real_t(3)) * pi * p[0] * p[0] * p[0];
    case SolidType::kSphere: {
      // Integral over the phi wedge and theta band of r^2 sin(theta) dr dtheta dphi.
      const real_t r3 = (p[1] * p[1] * p[1] - p[0] * p[0] * p[0]) / real_t(3);
      const real_t dcos = cos(p[4]) - cos(p[4] + p[5]);
      return p[3] * r3 * dcos;
    }
    case SolidType::kTorus: {
      const real_t frac = p[4] / twopi;
      // 2 pi^2 R (rmax^2 - rmin^2), from Pappus' theorem.
      return frac * real_t(2) * pi * pi * p[2] * (p[1] * p[1] - p[0] * p[0]);
    }
    case SolidType::kEllipticalTube:
      return pi * p[0] * p[1] * real_t(2) * p[2];
    case SolidType::kEllipsoid: {
      // Integral of pi a b (1 - z^2/c^2) between the two z cuts.
      const real_t a = p[0], b = p[1], c = p[2];
      const real_t z1 = fmax(p[3], -c), z2 = fmin(p[4], c);
      if (z2 <= z1) { return real_t(0); }
      const auto f = [&](real_t z) { return z - z * z * z / (real_t(3) * c * c); };
      return pi * a * b * (f(z2) - f(z1));
    }
    case SolidType::kEllipticalCone: {
      // x^2/a^2 + y^2/b^2 = (zmax - z)^2, so the cross-section area is pi a b (zmax - z)^2.
      const real_t a = p[0], b = p[1], zm = p[2], zc = p[3];
      const auto f = [&](real_t z) {
        const real_t u = zm - z;
        return -u * u * u / real_t(3);
      };
      return pi * a * b * (f(zc) - f(-zc));
    }
    case SolidType::kParaboloid: {
      // r^2 = k1 z + k2; the cross-section area is pi r^2, linear in z, so the mean of the
      // end areas times the length.
      const real_t dz = p[0];
      return pi * (p[1] * p[1] + p[2] * p[2]) * dz;
    }
    case SolidType::kTet: {
      // The plane form does not carry the vertices, so no closed form is available here.
      return real_t(-1);
    }
    default:
      return real_t(-1);
  }
}

/// Monte Carlo volume, with the statistical uncertainty in @p rel_error.
///
/// @param half_extent  half-width of a cube, centred on the origin, that contains the solid
template <typename real_t>
__host__ inline real_t sampled_volume(const SolidStore<real_t>& st, const Solid<real_t>& s,
                                      real_t half_extent, real_t& rel_error,
                                      int n_samples = 4000000) {
  Philox<real_t> rng(0x5EEDu, 0u);
  long long hits = 0;
  for (int i = 0; i < n_samples; ++i) {
    const Vec3<real_t> q{half_extent * (real_t(2) * rng.uniform() - real_t(1)),
                         half_extent * (real_t(2) * rng.uniform() - real_t(1)),
                         half_extent * (real_t(2) * rng.uniform() - real_t(1))};
    if (inside(st, s, q)) { ++hits; }
  }
  const real_t box = real_t(8) * half_extent * half_extent * half_extent;
  const real_t f = static_cast<real_t>(hits) / static_cast<real_t>(n_samples);
  rel_error = (hits > 0) ? sqrt((real_t(1) - f) / static_cast<real_t>(hits)) : real_t(1);
  return f * box;
}

/// A bound on each *coordinate* of any point the solid contains: |x|, |y| and |z| are all at
/// most this. Used to size the axis-aligned sampling box below.
///
/// **Not a bounding radius.** A box of (30, 40, 50) returns 50 and reaches 71.4 mm from its
/// origin at the corner. Anything that wants a bounding sphere - a navigator that skips a
/// volume whose sphere excludes the point, say - must multiply by sqrt(3); `to_local` is a
/// rotation about `xform.trans` and so preserves length, which makes a local sphere of radius
/// R exactly a world sphere of radius R about the placement.
///
/// The contract is checked in `tests/test_solids.cu`, over all thirty solids the oracle
/// covers, which is worth having because two of the cases below are admitted guesses. It also
/// prints the measured |p|/extent ratio: the worst is 1.364 (a Trd), so sqrt(3) = 1.732 has
/// room, and the Trap and Tet bounds are 4x and 2.4x looser than they need to be.
template <typename real_t>
__host__ inline real_t solid_half_extent(const SolidStore<real_t>& st, const Solid<real_t>& s) {
  const real_t* p = s.p;
  switch (s.type) {
    case SolidType::kMesh:
      // The bounding box, offset from the solid's origin: p[0..2] are its half-extents and
      // p[3..5] its centre, so this is the farthest corner.
      return fmax(fabs(p[3]) + p[0], fmax(fabs(p[4]) + p[1], fabs(p[5]) + p[2]));
    case SolidType::kBox:
    case SolidType::kVoxelGrid:
      // A voxel grid IS a box as far as any bound is concerned: p[0..2] are its half extent
      // and every cell is inside it. Leaving it out - which is how it was for the whole life
      // of the type - dropped it through to the `default` below, which returns ZERO, and a
      // bound of zero is not conservative in either direction. What that silently broke:
      //
      //   * same-layer overlap detection. Its cheap first stage compares centre separation
      //     against the sum of the two bounding radii, so a phantom with radius 0 only
      //     registered when the other volume's centre was nearly on top of its own. Reported
      //     as the check "only firing if there is substantial overlap", which is what a test
      //     that can only see the middle of a volume looks like from outside.
      //   * the mass of anything scored on a grid, because solid_volume falls back to
      //     sampling a box of this size and a box of side zero has no volume. A dose is
      //     energy over mass.
      //   * the camera's idea of how large the scene is, so a phantom framed the view around
      //     everything except itself.
      return fmax(p[0], fmax(p[1], p[2]));
    case SolidType::kTrd:
      return fmax(fmax(p[0], p[1]), fmax(fmax(p[2], p[3]), p[4]));
    case SolidType::kCons:
      return fmax(fmax(p[0], p[1]), p[2]);
    case SolidType::kTet:
    case SolidType::kTrap:
    case SolidType::kPara: {
      // Bound from the plane offsets: no vertex can be farther than the largest offset
      // divided by the smallest normal component, and the offsets themselves are a safe
      // over-estimate for a unit-normal plane set.
      real_t e = real_t(0);
      if (s.type == SolidType::kPara) {
        return (fabs(p[0]) + fabs(p[1]) + fabs(p[2]))
               * (real_t(1) + fabs(p[3]) + fabs(p[4]) + fabs(p[5]));
      }
      if (st.aux != nullptr) {
        for (int i = 0; i < s.b; ++i) { e = fmax(e, fabs(st.aux[s.a + 4 * i + 3])); }
      }
      return e * real_t(4);  // a plane offset bounds the inradius, not the circumradius
    }
    case SolidType::kUnion:
    case SolidType::kSubtraction:
    case SolidType::kIntersection: {
      if (st.solids == nullptr) { return real_t(0); }
      const real_t ea = solid_half_extent(st, st.solids[s.a]);
      const real_t eb = solid_half_extent(st, st.solids[s.b]);
      real_t off = real_t(0);
      const Solid<real_t>& rhs = st.solids[s.b];
      if (rhs.xform >= 0 && st.xforms != nullptr) {
        const Vec3<real_t>& t = st.xforms[rhs.xform].trans;
        off = sqrt(dot(t, t));
      }
      return fmax(ea, eb + off);
    }
    case SolidType::kPolycone:
    case SolidType::kPolyhedra: {
      real_t e = real_t(0);
      if (st.aux != nullptr) {
        for (int i = 0; i <= s.b; ++i) {
          e = fmax(e, fabs(st.aux[s.a + 3 * i]));      // |z|
          e = fmax(e, fabs(st.aux[s.a + 3 * i + 2]));  // rmax
        }
      }
      return e * real_t(1.2);  // a polyhedra's corners sit outside its inscribed radius
    }
    default: {
      // Curved solids: the bounding radius already computed for the safety estimate.
      const real_t* q = s.p;
      switch (s.type) {
        case SolidType::kTubs:           return fmax(q[1], q[2]);
        case SolidType::kConeSection:    return fmax(fmax(q[1], q[3]), q[4]);
        case SolidType::kOrb:            return q[0];
        case SolidType::kSphere:         return q[1];
        case SolidType::kTorus:          return q[2] + q[1];
        case SolidType::kEllipticalTube: return fmax(fmax(q[0], q[1]), q[2]);
        case SolidType::kEllipsoid:      return fmax(fmax(q[0], q[1]), q[2]);
        case SolidType::kEllipticalCone:
          return fmax(fmax(q[0], q[1]) * (q[2] + q[3]), q[3]);
        case SolidType::kParaboloid:     return fmax(fmax(q[1], q[2]), q[0]);
        case SolidType::kHype:
          return fmax(q[1] + fabs(q[3]) * q[4], q[4]);
        default: return real_t(0);
      }
    }
  }
}

/// Volume of a solid, mm^3. Closed form where one exists, Monte Carlo otherwise.
template <typename real_t>
__host__ inline real_t solid_volume(const SolidStore<real_t>& st, const Solid<real_t>& s) {
  const real_t v = analytic_volume(s);
  if (v >= real_t(0)) { return v; }
  real_t err = 0;
  const real_t half = solid_half_extent(st, s) * real_t(1.02);
  if (half <= real_t(0)) { return real_t(0); }
  return sampled_volume(st, s, half, err);
}

}  // namespace g4gpu::geom
