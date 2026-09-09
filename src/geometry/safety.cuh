// Isotropic safety: a lower bound on the distance from a point to the nearest surface, in
// any direction. Geant4 calls this DistanceToOut(p) / DistanceToIn(p) - the versions without
// a direction - and requires only that they never overestimate. Urban MSC uses it to decide
// how long a step it can afford (the fUseSafety limitation), so a conservative underestimate
// costs performance but never correctness.
#pragma once
#include <cmath>
#include "geometry/navigator.cuh"
#include "geometry/solids.cuh"

namespace g4gpu::geom {

/// Distance from an interior point to the box surface. Exact.
template <typename real_t>
__host__ __device__ inline real_t box_safety_out(const real_t* p, const Vec3<real_t>& q) {
  const real_t sx = p[0] - fabs(q.x), sy = p[1] - fabs(q.y), sz = p[2] - fabs(q.z);
  return fmax(real_t(0), fmin(sx, fmin(sy, sz)));
}

/// Distance from an exterior point to the box, as G4Box::DistanceToIn(p) does it: the
/// largest single-axis overshoot, which underestimates the true distance at corners.
template <typename real_t>
__host__ __device__ inline real_t box_safety_in(const real_t* p, const Vec3<real_t>& q) {
  const real_t dx = fabs(q.x) - p[0], dy = fabs(q.y) - p[1], dz = fabs(q.z) - p[2];
  return fmax(real_t(0), fmax(dx, fmax(dy, dz)));
}

/// Cone with rmin = 0 and full phi: p[0] = rmax at -dz, p[1] = rmax at +dz, p[2] = dz.
/// The lateral distance is the radial gap projected onto the cone normal, which is what
/// G4Cons does with its secRMax factor.
template <typename real_t>
__host__ __device__ inline real_t cons_safety_out(const real_t* p, const Vec3<real_t>& q) {
  const real_t rmax1 = p[0], rmax2 = p[1], dz = p[2];
  const real_t rho = sqrt(q.x * q.x + q.y * q.y);
  const real_t tan_a = (rmax2 - rmax1) / (real_t(2) * dz);
  const real_t sec = sqrt(real_t(1) + tan_a * tan_a);
  const real_t rmax_here = rmax1 + (rmax2 - rmax1) * (q.z + dz) / (real_t(2) * dz);
  const real_t s_lat = (rmax_here - rho) / sec;
  const real_t s_z = dz - fabs(q.z);
  return fmax(real_t(0), fmin(s_lat, s_z));
}

template <typename real_t>
__host__ __device__ inline real_t cons_safety_in(const real_t* p, const Vec3<real_t>& q) {
  const real_t rmax1 = p[0], rmax2 = p[1], dz = p[2];
  const real_t rho = sqrt(q.x * q.x + q.y * q.y);
  const real_t tan_a = (rmax2 - rmax1) / (real_t(2) * dz);
  const real_t sec = sqrt(real_t(1) + tan_a * tan_a);
  const real_t rmax_here = rmax1 + (rmax2 - rmax1) * (q.z + dz) / (real_t(2) * dz);
  const real_t s_lat = (rho - rmax_here) / sec;
  const real_t s_z = fabs(q.z) - dz;
  return fmax(real_t(0), fmax(s_lat, s_z));
}

/// Trapezoid: p[0] = dx1, p[1] = dx2, p[2] = dy1, p[3] = dy2, p[4] = dz. Each sloped face
/// contributes its perpendicular distance, as G4Trd does.
template <typename real_t>
__host__ __device__ inline real_t trd_safety_out(const real_t* p, const Vec3<real_t>& q) {
  const real_t dx1 = p[0], dx2 = p[1], dy1 = p[2], dy2 = p[3], dz = p[4];
  const real_t tan_x = (dx2 - dx1) / (real_t(2) * dz);
  const real_t sec_x = sqrt(real_t(1) + tan_x * tan_x);
  const real_t tan_y = (dy2 - dy1) / (real_t(2) * dz);
  const real_t sec_y = sqrt(real_t(1) + tan_y * tan_y);
  const real_t dx_here = dx1 + (dx2 - dx1) * (q.z + dz) / (real_t(2) * dz);
  const real_t dy_here = dy1 + (dy2 - dy1) * (q.z + dz) / (real_t(2) * dz);
  const real_t s = fmin((dx_here - fabs(q.x)) / sec_x,
                        fmin((dy_here - fabs(q.y)) / sec_y, dz - fabs(q.z)));
  return fmax(real_t(0), s);
}

template <typename real_t>
__host__ __device__ inline real_t trd_safety_in(const real_t* p, const Vec3<real_t>& q) {
  const real_t dx1 = p[0], dx2 = p[1], dy1 = p[2], dy2 = p[3], dz = p[4];
  const real_t tan_x = (dx2 - dx1) / (real_t(2) * dz);
  const real_t sec_x = sqrt(real_t(1) + tan_x * tan_x);
  const real_t tan_y = (dy2 - dy1) / (real_t(2) * dz);
  const real_t sec_y = sqrt(real_t(1) + tan_y * tan_y);
  const real_t dx_here = dx1 + (dx2 - dx1) * (q.z + dz) / (real_t(2) * dz);
  const real_t dy_here = dy1 + (dy2 - dy1) * (q.z + dz) / (real_t(2) * dz);
  const real_t s = fmax((fabs(q.x) - dx_here) / sec_x,
                        fmax((fabs(q.y) - dy_here) / sec_y, fabs(q.z) - dz));
  return fmax(real_t(0), s);
}

/// Conservative safety for a solid with no closed form: half the distance to the bounding
/// sphere's near surface, which is always an underestimate.
///
/// Returning zero would be correct too - safety is only ever used to skip work - but zero
/// disables the MSC step-limiting optimisation entirely and roughly doubles the step count
/// near such a solid. Anything strictly positive and strictly conservative is a win, so the
/// bound is deliberately crude rather than tight.
template <typename real_t>
__host__ __device__ inline real_t bounding_radius(const Solid<real_t>& s) {
  const real_t* p = s.p;
  switch (s.type) {
    case SolidType::kMesh:
      return sqrt((fabs(p[3]) + p[0]) * (fabs(p[3]) + p[0])
                  + (fabs(p[4]) + p[1]) * (fabs(p[4]) + p[1])
                  + (fabs(p[5]) + p[2]) * (fabs(p[5]) + p[2]));
    case SolidType::kTubs:
      return sqrt(p[1] * p[1] + p[2] * p[2]);
    case SolidType::kConeSection:
      return sqrt(fmax(p[1], p[3]) * fmax(p[1], p[3]) + p[4] * p[4]);
    case SolidType::kOrb:            return p[0];
    case SolidType::kSphere:         return p[1];
    case SolidType::kTorus:          return p[2] + p[1];
    case SolidType::kEllipticalTube: return sqrt(fmax(p[0], p[1]) * fmax(p[0], p[1]) + p[2] * p[2]);
    case SolidType::kEllipsoid:      return fmax(p[0], fmax(p[1], p[2]));
    case SolidType::kEllipticalCone:
      return sqrt(fmax(p[0], p[1]) * (p[2] + p[3]) * fmax(p[0], p[1]) * (p[2] + p[3])
                  + p[3] * p[3]);
    case SolidType::kParaboloid:     return sqrt(fmax(p[1], p[2]) * fmax(p[1], p[2]) + p[0] * p[0]);
    case SolidType::kHype:
      return sqrt((p[1] + fabs(p[3]) * p[4]) * (p[1] + fabs(p[3]) * p[4]) + p[4] * p[4]);
    case SolidType::kPara:
      return fabs(p[0]) + fabs(p[1]) + fabs(p[2])
             + (fabs(p[3]) + fabs(p[4]) + fabs(p[5])) * fabs(p[2]);
    default:
      return real_t(0);  // unknown extent: no useful bound
  }
}

/// The same bound for a solid whose extent lives in the AUX POOL, which the p[]-only form
/// above cannot see.
///
/// kPolycone and kPolyhedra keep their (z, rmin, rmax) sections there, so they fall through
/// that switch to zero - and zero does not mean "no bound needed", it means "no bound". The
/// renderer starts each ray at the solid's bounding sphere so that the quadratic it solves has
/// coefficients of order the solid's size rather than of order the camera distance; a zero
/// bound turns that off silently.
///
/// It cost 61% of the rays at the limb of the builder's own Cone primitive at 1500 mm - the
/// default camera distance - and the shape of the loss is the same signature as the sphere's
/// cancelling discriminant that the origin shift was written for: 100% kept at 200 mm, 93% at
/// 600, 39% at 1500, 17% at 12000. Reported as the cone showing ray-tracing artifacts that
/// depend on the zoom, and it was the SAME BUG as the sphere, reached through a function that
/// had never heard of two of the solid types.
///
/// Separate from the p[]-only form rather than replacing it, because that one is on the
/// transport's safety path and returning a larger number there is a physics change. This is
/// the renderer asking a rendering question, and it has the store in hand.
template <typename real_t>
__host__ __device__ inline real_t bounding_radius(const SolidStore<real_t>& store,
                                                  const Solid<real_t>& s, int depth = 0) {
  if (s.type == SolidType::kPolycone || s.type == SolidType::kPolyhedra) {
    if (store.aux == nullptr || s.b < 0) { return real_t(0); }
    real_t r2 = real_t(0);
    // s.b is the number of SEGMENTS, so there are s.b + 1 planes. A bound that read only the
    // segments would miss the last plane, which on a bicone is one of the two widest.
    for (int i = 0; i <= s.b; ++i) {
      const real_t z = store.aux[s.a + 3 * i];
      const real_t rmax = store.aux[s.a + 3 * i + 2];
      const real_t q = z * z + rmax * rmax;
      if (q > r2) { r2 = q; }
    }
    return sqrt(r2);
  }
  // A BOOLEAN IS BOUNDED BY ITS CHILDREN, and it has to be: boolean_dist hands each child the
  // SAME ray origin it was given, so a union of two spheres solves the children's quadratics
  // from wherever the caller started - and with no bound of its own the origin shift never
  // happens. The bound is each child's own bound plus how far its frame is offset, which is an
  // over-estimate for a rotated child and sound in the direction that matters: a shift that
  // stops short of the real surface only leaves a little cancellation, while one that
  // overshoots would skip past a hit.
  if (s.type == SolidType::kUnion || s.type == SolidType::kSubtraction
      || s.type == SolidType::kIntersection) {
    if (store.solids == nullptr || depth >= 8) { return real_t(0); }
    real_t best = real_t(0);
    const int child[2] = {s.a, s.b};
    for (int k = 0; k < 2; ++k) {
      const Solid<real_t>& c = store.solids[child[k]];
      const real_t rc = bounding_radius(store, c, depth + 1);
      if (rc <= real_t(0)) { return real_t(0); }   // one unknown child is an unknown whole
      real_t off = real_t(0);
      if (c.xform >= 0 && store.xforms != nullptr) {
        const Vec3<real_t>& t = store.xforms[c.xform].trans;
        off = sqrt(t.x * t.x + t.y * t.y + t.z * t.z);
      }
      const real_t r = rc + off;
      if (r > best) { best = r; }
    }
    // A subtraction is bounded by its FIRST child alone, but taking the larger of the two is
    // still sound and is one less rule to get wrong.
    return best;
  }
  return bounding_radius(s);
}

/// Radius of the largest sphere centred on the solid's own origin that lies entirely inside
/// it. Zero when there is no such sphere, or when it is not worth working out.
///
/// This is what an isotropic safety for an interior point can be built from, and it is not the
/// same thing as the bounding radius. For a point at distance d from the origin with d < r_in,
/// the ball of radius (r_in - d) around it is inside the sphere of radius r_in and therefore
/// inside the solid, so (r_in - d) is a sound safety. Using the *bounding* radius there
/// instead - as this file did until the mesh work exposed it - overestimates badly for a flat
/// or elongated solid: a 1 mm radius, 100 mm long cylinder has a bounding radius of 100 mm and
/// would have claimed a 50 mm safety at its axis, where the wall is 1 mm away. Safety is only
/// ever used to skip work, so an overestimate is not a crash; it is Urban MSC taking a step
/// far longer than it should, which is a physics error and a silent one.
///
/// Zero is always sound. Anything here that is not obviously right should be zero.
template <typename real_t>
__host__ __device__ inline real_t inradius(const Solid<real_t>& s) {
  const real_t* p = s.p;
  switch (s.type) {
    case SolidType::kTubs:
      // p[0] rmin, p[1] rmax, p[2] dz, p[3..4] phi. An inner hole or a phi wedge puts the
      // origin on or outside the solid, so there is no origin-centred sphere inside it.
      if (p[0] > real_t(0)) { return real_t(0); }
      if (!phi_ok(real_t(0), real_t(0), p[3], p[4])) { return real_t(0); }
      return fmin(p[1], p[2]);
    case SolidType::kConeSection:
      // p[0..1] rmin1/rmax1, p[2..3] rmin2/rmax2, p[4] dz.
      if (p[0] > real_t(0) || p[2] > real_t(0)) { return real_t(0); }
      return fmin(fmin(p[1], p[3]), p[4]);
    case SolidType::kOrb:
      return p[0];
    case SolidType::kSphere:
      // p[0] rmin, p[1] rmax. A hollow shell or a cut in theta/phi excludes the origin.
      return (p[0] > real_t(0)) ? real_t(0) : p[1];
    case SolidType::kEllipticalTube:
      return fmin(fmin(p[0], p[1]), p[2]);
    case SolidType::kEllipsoid:
      // p[0..2] semi-axes, p[3..4] z cuts. The cuts can bite into the sphere at the centre.
      return fmin(fmin(p[0], fmin(p[1], p[2])), fmin(fabs(p[3]), fabs(p[4])));
    // Torus (the origin is in its hole), hyperbolic tube (an inner void), paraboloid and
    // elliptical cone (the origin is at or near the apex), and the flat-faced solids whose
    // planes are in the aux pool: no origin-centred sphere worth computing here.
    default:
      return real_t(0);
  }
}

template <typename real_t>
__host__ __device__ inline real_t safety_out(const SolidStore<real_t>& st,
                                             const Solid<real_t>& s, const Vec3<real_t>& q) {
  switch (s.type) {
    case SolidType::kBox:  return box_safety_out(s.p, q);
    case SolidType::kCons: return cons_safety_out(s.p, q);
    case SolidType::kTrd:  return trd_safety_out(s.p, q);
    // A mesh answers exactly, by finding the nearest triangle through its BVH. Same value
    // from either side: the distance to a surface does not care which side you are on.
    case SolidType::kMesh: return mesh_safety(st, s, q);
    default: {
      const real_t r = inradius(s);
      if (r <= real_t(0)) { return real_t(0); }
      const real_t d = sqrt(dot(q, q));
      return fmax(real_t(0), r - d);
    }
  }
}

template <typename real_t>
__host__ __device__ inline real_t safety_in(const SolidStore<real_t>& st,
                                            const Solid<real_t>& s, const Vec3<real_t>& q) {
  switch (s.type) {
    case SolidType::kBox:  return box_safety_in(s.p, q);
    case SolidType::kCons: return cons_safety_in(s.p, q);
    case SolidType::kTrd:  return trd_safety_in(s.p, q);
    case SolidType::kMesh: return mesh_safety(st, s, q);
    default: {
      // From outside, the bounding radius is the sound one: a point d from the origin with
      // d > r cannot be nearer than (d - r) to anything inside the bounding sphere.
      const real_t r = bounding_radius(s);
      if (r <= real_t(0)) { return real_t(0); }
      const real_t d = sqrt(dot(q, q));
      return fmax(real_t(0), d - r);
    }
  }
}

/// Safety within volume @p vol: the nearer of its own surface and the surface of any volume
/// that outranks it. Mirrors G4Navigator::ComputeSafety, which takes the same minimum over
/// daughters; here the set is "volumes that could take ownership", which is the layer-model
/// equivalent. Volumes it outranks are invisible from inside it and cannot shorten the step.
template <typename real_t>
__host__ __device__ inline real_t compute_safety(const Geometry<real_t>& g, int vol,
                                                 const Vec3<real_t>& p) {
  if (vol < 0) { return real_t(0); }
  const Volume<real_t>& v = g.volumes[vol];
  real_t s = safety_out(g.store, v.solid, to_local(v.xform, p));
  for (int i = 0; i < g.n_volumes; ++i) {
    if (i == vol || !outranks(g, i, vol)) { continue; }
    const Volume<real_t>& u = g.volumes[i];
    s = fmin(s, safety_in(g.store, u.solid, to_local(u.xform, p)));
  }
  return fmax(real_t(0), s);
}

}  // namespace g4gpu::geom
