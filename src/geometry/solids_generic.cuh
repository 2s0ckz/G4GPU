// The generic solid engine: everything except the three closed-form solids in solids.cuh.
//
// This file is included from the middle of solids.cuh, after the box/cone/trd routines and
// before the dispatch, because the dispatch has to see both. It is split out only for length.
//
// Two engines and one exception:
//
//   * a convex half-space engine, for kPara, kTrap and kTet - a solid is the intersection of
//     up to six planes, so `inside` is six dot products and the candidate surfaces are the
//     six plane crossings;
//   * an axis-aligned quadric engine, A x^2 + B y^2 + C z^2 + D z + E <= 0, optionally with an
//     inner quadric, a z range, a phi wedge and a theta cone. That one form covers kTubs,
//     kConeSection, kOrb, kSphere, kEllipticalTube, kEllipsoid, kEllipticalCone, kParaboloid
//     and kHype; only the coefficients differ;
//   * kTorus, which is quartic and gets its own root finder.
//
// kPolycone and kPolyhedra are loops over cone or prism sections. Booleans recurse.
//
// Distances are computed the same way for all of them: generate every ray parameter where the
// ray meets a bounding surface, then take the first one whose far side has the containment you
// are looking for. The far side is probed a fixed kProbe() past the crossing.
#pragma once

namespace g4gpu::geom {

/// How far past a candidate crossing to test containment. Must exceed the surface tolerance
/// by enough to land decisively on one side, and stay far below any real feature size.
/// Geometry here is in mm, so 1e-6 mm is 10 nm - smaller than anything anyone models.
template <typename real_t> __host__ __device__ constexpr real_t kProbe() {
  return (sizeof(real_t) == 4) ? real_t(1e-3) : real_t(1e-6);
}

/// Largest number of candidate crossings any single shape can produce in one query.
/// Sphere is the worst: 2 outer + 2 inner + 4 theta-cone + 2 phi = 10.
constexpr int kMaxCandidates = 12;

// ---------------------------------------------------------------- small helpers

template <typename real_t>
__host__ __device__ inline bool phi_is_full(real_t dphi) {
  return dphi >= units::twopi<real_t>() - kSurfTolerance<real_t>();
}

/// True if (x, y) lies within the wedge [sphi, sphi + dphi].
template <typename real_t>
__host__ __device__ inline bool phi_ok(real_t x, real_t y, real_t sphi, real_t dphi) {
  if (phi_is_full(dphi)) { return true; }
  const real_t twopi = units::twopi<real_t>();
  real_t rel = atan2(y, x) - sphi;
  rel = rel - twopi * floor(rel / twopi);  // wrap into [0, 2pi)
  return rel <= dphi + kSurfTolerance<real_t>();
}

/// Axis-aligned quadric: A x^2 + B y^2 + C z^2 + D z + E.
template <typename real_t>
struct Quadric {
  real_t A, B, C, D, E;
};

template <typename real_t>
__host__ __device__ inline real_t quadric_eval(const Quadric<real_t>& s, const Vec3<real_t>& q) {
  return s.A * q.x * q.x + s.B * q.y * q.y + s.C * q.z * q.z + s.D * q.z + s.E;
}

/// Ray parameters where the ray meets the quadric surface. Returns how many were written.
/// Uses the numerically stable form of the quadratic formula: computing both roots as
/// (-b +/- sqrt(disc)) / 2a loses all precision in the smaller root when b^2 >> 4ac, which for
/// a ray nearly tangent to a large cylinder is exactly the case that matters.
template <typename real_t>
__host__ __device__ inline int quadric_roots(const Quadric<real_t>& s, const Vec3<real_t>& p,
                                             const Vec3<real_t>& d, real_t* t) {
  const real_t a = s.A * d.x * d.x + s.B * d.y * d.y + s.C * d.z * d.z;
  const real_t b = real_t(2) * (s.A * p.x * d.x + s.B * p.y * d.y + s.C * p.z * d.z) + s.D * d.z;
  const real_t c = quadric_eval(s, p);
  const real_t tiny = kTolerance<real_t>();

  if (fabs(a) <= tiny) {
    if (fabs(b) <= tiny) { return 0; }
    t[0] = -c / b;
    return 1;
  }
  const real_t disc = b * b - real_t(4) * a * c;
  if (disc < real_t(0)) { return 0; }
  const real_t sq = sqrt(disc);
  const real_t qq = real_t(-0.5) * (b + ((b >= real_t(0)) ? sq : -sq));
  t[0] = qq / a;
  t[1] = (fabs(qq) > tiny) ? (c / qq) : t[0];
  return 2;
}

/// Ray parameter where the ray crosses the plane z = zp.
template <typename real_t>
__host__ __device__ inline int z_plane_root(real_t zp, const Vec3<real_t>& p,
                                            const Vec3<real_t>& d, real_t* t) {
  if (fabs(d.z) <= kTolerance<real_t>()) { return 0; }
  t[0] = (zp - p.z) / d.z;
  return 1;
}

/// Ray parameters where the ray crosses either bounding half-plane of a phi wedge.
template <typename real_t>
__host__ __device__ inline int phi_plane_roots(real_t sphi, real_t dphi, const Vec3<real_t>& p,
                                               const Vec3<real_t>& d, real_t* t) {
  if (phi_is_full(dphi)) { return 0; }
  int n = 0;
  for (int k = 0; k < 2; ++k) {
    const real_t a = sphi + (k == 0 ? real_t(0) : dphi);
    // Half-plane through the z axis with outward normal (sin a, -cos a) for the start edge.
    const real_t nx = -sin(a), ny = cos(a);
    const real_t den = nx * d.x + ny * d.y;
    if (fabs(den) <= kTolerance<real_t>()) { continue; }
    t[n++] = -(nx * p.x + ny * p.y) / den;
  }
  return n;
}

// ---------------------------------------------------------------- convex half-space engine

/// Planes live in the aux pool as (nx, ny, nz, d) quadruples; the solid is the intersection of
/// the half-spaces n . x <= d.
template <typename real_t>
__host__ __device__ inline bool planes_inside(const real_t* pl, int n, const Vec3<real_t>& q) {
  for (int i = 0; i < n; ++i) {
    const real_t* e = pl + 4 * i;
    if (e[0] * q.x + e[1] * q.y + e[2] * q.z - e[3] > kSurfTolerance<real_t>()) { return false; }
  }
  return true;
}

template <typename real_t>
__host__ __device__ inline int planes_candidates(const real_t* pl, int n, const Vec3<real_t>& p,
                                                 const Vec3<real_t>& d, real_t* t) {
  int m = 0;
  for (int i = 0; i < n && m < kMaxCandidates; ++i) {
    const real_t* e = pl + 4 * i;
    const real_t den = e[0] * d.x + e[1] * d.y + e[2] * d.z;
    if (fabs(den) <= kTolerance<real_t>()) { continue; }
    t[m++] = (e[3] - (e[0] * p.x + e[1] * p.y + e[2] * p.z)) / den;
  }
  return m;
}

template <typename real_t>
__host__ __device__ inline Vec3<real_t> planes_normal(const real_t* pl, int n,
                                                      const Vec3<real_t>& q) {
  int best = 0;
  real_t bestv = -kInfinity<real_t>();
  for (int i = 0; i < n; ++i) {
    const real_t* e = pl + 4 * i;
    const real_t v = e[0] * q.x + e[1] * q.y + e[2] * q.z - e[3];
    if (v > bestv) { bestv = v; best = i; }
  }
  const real_t* e = pl + 4 * best;
  return Vec3<real_t>{e[0], e[1], e[2]};
}

/// G4Para's six faces, derived from its half-lengths and the two shear angles.
///
/// The shears compose, and the order matters. A point is inside when
///     |z| <= dz,
///     |y - z ty| <= dy,
///     |x - z tx - (y - z ty) ta| <= dx,
/// where tx, ty are tan(theta) times cos/sin(phi) and ta is tan(alpha). The x condition is
/// stated relative to the *already y-shifted* centre, so its z coefficient is (tx - ta ty),
/// not tx - writing tx there tilts the x faces by the wrong amount whenever both shears are
/// present, which is exactly the case a single-shear test would not catch.
template <typename real_t>
__host__ __device__ inline int para_planes(const real_t* p, real_t* out) {
  const real_t dx = p[0], dy = p[1], dz = p[2];
  const real_t ta = p[3], tx = p[4], ty = p[5];
  // z faces
  out[0] = 0; out[1] = 0; out[2] = 1; out[3] = dz;
  out[4] = 0; out[5] = 0; out[6] = -1; out[7] = dz;
  // y faces: y - ty z = +/- dy
  const real_t ny = real_t(1) / sqrt(real_t(1) + ty * ty);
  out[8] = 0;  out[9] = ny;  out[10] = -ty * ny; out[11] = dy * ny;
  out[12] = 0; out[13] = -ny; out[14] = ty * ny; out[15] = dy * ny;
  // x faces: x - ta y - (tx - ta ty) z = +/- dx
  const real_t cz = tx - ta * ty;
  const real_t nx = real_t(1) / sqrt(real_t(1) + ta * ta + cz * cz);
  out[16] = nx;  out[17] = -ta * nx; out[18] = -cz * nx; out[19] = dx * nx;
  out[20] = -nx; out[21] = ta * nx;  out[22] = cz * nx;  out[23] = dx * nx;
  return 6;
}

// ---------------------------------------------------------------- quadric engine

/// The quadric description of one solid: an outer surface, an optional inner surface, a z
/// range, a phi wedge, and for the sphere a pair of theta cones.
template <typename real_t>
struct QuadricShape {
  Quadric<real_t> outer;
  Quadric<real_t> inner;
  bool has_inner;
  real_t zmin, zmax;
  real_t sphi, dphi;
  bool has_theta;
  Quadric<real_t> theta_lo, theta_hi;  ///< the two bounding cones, for candidate generation
  /// cos of the two polar angles. Testing cos theta rather than tan theta is what keeps the
  /// obtuse and the exactly-90-degree cases from needing their own branches: theta lies in
  /// [st, st+dt] exactly when z <= r cos(st) and z >= r cos(st+dt), for every angle.
  real_t cos_lo, cos_hi;
};

/// Cone with half-angle theta about +z: x^2 + y^2 - (z tan theta)^2 = 0. A point is inside the
/// cone of half-angle theta (measured from +z) when that form is <= 0 *and* z has the right
/// sign; theta > pi/2 flips the sign.
template <typename real_t>
__host__ __device__ inline Quadric<real_t> theta_cone(real_t theta) {
  const real_t tt = tan(theta);
  return Quadric<real_t>{real_t(1), real_t(1), -tt * tt, real_t(0), real_t(0)};
}

/// Gradient of a quadric form at a point: the outward normal of `Q(x) = 0` when Q is negative
/// inside, which is how every one of these is written.
template <typename real_t>
__host__ __device__ inline Vec3<real_t> quadric_grad(const Quadric<real_t>& s,
                                                     const Vec3<real_t>& q) {
  return Vec3<real_t>{real_t(2) * s.A * q.x, real_t(2) * s.B * q.y,
                      real_t(2) * s.C * q.z + s.D};
}

/// The outward normal of whichever bounding surface of a quadric shape @p q is on.
///
/// This exists because the fallback it replaces was a central difference of the *binary*
/// containment indicator:
///
///     f(v) = inside(v) ? -1 : +1
///     g    = { f(x+h) - f(x-h), ... }
///
/// Each component of that can only be -2, 0 or +2, so the normal it returns is one of 26
/// directions. On a sphere that is not a smooth normal: the shading came out as flat facets
/// whose boundaries are the coordinate planes and the diagonals, which is what "the sphere
/// renders with central cartesian planes" was. It is not a physics bug - normals are used for
/// shading and nothing else - but it made a sphere look like a cut-open shell.
///
/// Each candidate surface is scored by how nearly the point satisfies it with equality,
/// normalised by the gradient magnitude so that surfaces with very different scales - a
/// millimetre-radius sphere and a metre-long cylinder - compare fairly. The winner's analytic
/// gradient is the answer.
template <typename real_t>
__host__ __device__ inline Vec3<real_t> quadric_normal(const QuadricShape<real_t>& qs,
                                                       const Vec3<real_t>& q) {
  Vec3<real_t> best{0, 0, real_t(1)};
  real_t best_d = kInfinity<real_t>();
  auto consider = [&](real_t residual, const Vec3<real_t>& grad) {
    const real_t gl = sqrt(dot(grad, grad));
    if (gl <= kTolerance<real_t>()) { return; }
    // |Q| / |grad Q| is the first-order distance to the surface Q = 0.
    const real_t d = fabs(residual) / gl;
    if (d < best_d) {
      best_d = d;
      best = (real_t(1) / gl) * grad;
    }
  };

  consider(quadric_eval(qs.outer, q), quadric_grad(qs.outer, q));
  if (qs.has_inner) {
    // The inner surface faces the other way: the material is outside it.
    const Vec3<real_t> g = quadric_grad(qs.inner, q);
    consider(quadric_eval(qs.inner, q), real_t(-1) * g);
  }
  consider(q.z - qs.zmax, Vec3<real_t>{0, 0, real_t(1)});
  consider(q.z - qs.zmin, Vec3<real_t>{0, 0, real_t(-1)});
  if (qs.dphi < units::twopi<real_t>() - kSurfTolerance<real_t>()) {
    // The two half-planes bounding the wedge. Their normals point out of the wedge.
    const real_t s0 = qs.sphi, s1 = qs.sphi + qs.dphi;
    const Vec3<real_t> n0{sin(s0), -cos(s0), 0};
    const Vec3<real_t> n1{-sin(s1), cos(s1), 0};
    consider(dot(n0, q), n0);
    consider(dot(n1, q), n1);
  }
  if (qs.has_theta) {
    // A theta cone's outward direction depends on which side the solid is on, and that is
    // what cos_lo and cos_hi record: the solid satisfies z <= r cos_lo and z >= r cos_hi.
    const real_t r = sqrt(q.x * q.x + q.y * q.y + q.z * q.z);
    if (r > kTolerance<real_t>()) {
      // d/dx of (z - r cos) is (-cos x/r, -cos y/r, 1 - cos z/r).
      const Vec3<real_t> g_lo{-qs.cos_lo * q.x / r, -qs.cos_lo * q.y / r,
                              real_t(1) - qs.cos_lo * q.z / r};
      consider(q.z - r * qs.cos_lo, g_lo);
      const Vec3<real_t> g_hi{qs.cos_hi * q.x / r, qs.cos_hi * q.y / r,
                              qs.cos_hi * q.z / r - real_t(1)};
      consider(r * qs.cos_hi - q.z, g_hi);
    }
  }
  return best;
}

/// Whether this shape is described by the quadric engine - an outer surface, an optional inner
/// one, a z range, a phi wedge and for the sphere two theta cones.
///
/// Listed in one place because three switches used to list the same nine cases, and the
/// fourth - the normal - is the one that would silently take the fallback if a shape were
/// added to the other three and forgotten here.
__host__ __device__ inline bool is_quadric_shape(SolidType t) {
  switch (t) {
    case SolidType::kTubs:
    case SolidType::kConeSection:
    case SolidType::kOrb:
    case SolidType::kSphere:
    case SolidType::kEllipticalTube:
    case SolidType::kEllipsoid:
    case SolidType::kEllipticalCone:
    case SolidType::kParaboloid:
    case SolidType::kHype:
      return true;
    default:
      return false;
  }
}

/// Builds the quadric description for whichever of the curved solids this is.
template <typename real_t>
__host__ __device__ inline QuadricShape<real_t> quadric_of(const Solid<real_t>& s) {
  QuadricShape<real_t> q{};
  q.has_inner = false;
  q.has_theta = false;
  q.sphi = real_t(0);
  q.dphi = units::twopi<real_t>();
  const real_t* p = s.p;
  const real_t one = real_t(1), zero = real_t(0);

  switch (s.type) {
    case SolidType::kTubs: {
      q.outer = {one, one, zero, zero, -p[1] * p[1]};
      q.has_inner = (p[0] > zero);
      q.inner = {one, one, zero, zero, -p[0] * p[0]};
      q.zmin = -p[2];
      q.zmax = p[2];
      q.sphi = p[3];
      q.dphi = p[4];
      break;
    }
    case SolidType::kCons:
    case SolidType::kConeSection: {
      // kCons is the same cone with rmin = 0 and full phi; mapping it here rather than giving
      // it a second implementation keeps one source of truth for the lateral surface.
      const bool simple = (s.type == SolidType::kCons);
      const real_t rmin1 = simple ? zero : p[0];
      const real_t rmax1 = simple ? p[0] : p[1];
      const real_t rmin2 = simple ? zero : p[2];
      const real_t rmax2 = simple ? p[1] : p[3];
      const real_t hz = simple ? p[2] : p[4];
      const real_t s0 = simple ? zero : p[5];
      const real_t s1 = simple ? units::twopi<real_t>() : p[6];
      // r(z) = r1 + (r2 - r1) * (z + hz) / (2 hz) = m z + c
      const real_t mo = (rmax2 - rmax1) / (real_t(2) * hz);
      const real_t co = (rmax2 + rmax1) / real_t(2);
      q.outer = {one, one, -mo * mo, real_t(-2) * mo * co, -co * co};
      const real_t mi = (rmin2 - rmin1) / (real_t(2) * hz);
      const real_t ci = (rmin2 + rmin1) / real_t(2);
      q.has_inner = (rmin1 > zero || rmin2 > zero);
      q.inner = {one, one, -mi * mi, real_t(-2) * mi * ci, -ci * ci};
      q.zmin = -hz;
      q.zmax = hz;
      q.sphi = s0;
      q.dphi = s1;
      break;
    }
    case SolidType::kOrb: {
      q.outer = {one, one, one, zero, -p[0] * p[0]};
      q.zmin = -p[0];
      q.zmax = p[0];
      break;
    }
    case SolidType::kSphere: {
      q.outer = {one, one, one, zero, -p[1] * p[1]};
      q.has_inner = (p[0] > zero);
      q.inner = {one, one, one, zero, -p[0] * p[0]};
      q.zmin = -p[1];
      q.zmax = p[1];
      q.sphi = p[2];
      q.dphi = p[3];
      const real_t st = p[4], dt = p[5];
      q.has_theta = (st > kSurfTolerance<real_t>()
                     || st + dt < units::pi<real_t>() - kSurfTolerance<real_t>());
      if (q.has_theta) {
        q.theta_lo = theta_cone(st);
        q.theta_hi = theta_cone(st + dt);
        q.cos_lo = cos(st);
        q.cos_hi = cos(st + dt);
      }
      break;
    }
    case SolidType::kEllipticalTube: {
      q.outer = {one / (p[0] * p[0]), one / (p[1] * p[1]), zero, zero, -one};
      q.zmin = -p[2];
      q.zmax = p[2];
      break;
    }
    case SolidType::kEllipsoid: {
      q.outer = {one / (p[0] * p[0]), one / (p[1] * p[1]), one / (p[2] * p[2]), zero, -one};
      q.zmin = (p[3] < p[4]) ? p[3] : -p[2];
      q.zmax = (p[3] < p[4]) ? p[4] : p[2];
      if (q.zmin < -p[2]) { q.zmin = -p[2]; }
      if (q.zmax > p[2]) { q.zmax = p[2]; }
      break;
    }
    case SolidType::kEllipticalCone: {
      // x^2/a^2 + y^2/b^2 = (zMax - z)^2, cut at |z| <= zTopCut.
      const real_t a2 = p[0] * p[0], b2 = p[1] * p[1], zm = p[2];
      q.outer = {one / a2, one / b2, -one, real_t(2) * zm, -zm * zm};
      q.zmin = -p[3];
      q.zmax = p[3];
      break;
    }
    case SolidType::kParaboloid: {
      // r^2 = k1 z + k2 with r(-dz) = p[1], r(+dz) = p[2].
      const real_t dz = p[0], r1 = p[1], r2 = p[2];
      const real_t k1 = (r2 * r2 - r1 * r1) / (real_t(2) * dz);
      const real_t k2 = (r2 * r2 + r1 * r1) / real_t(2);
      q.outer = {one, one, zero, -k1, -k2};
      q.zmin = -dz;
      q.zmax = dz;
      break;
    }
    case SolidType::kHype: {
      // Outer: x^2 + y^2 - rmax^2 - (z tan_out)^2 <= 0. Inner mirrors it.
      const real_t ti = p[2], to = p[3];
      q.outer = {one, one, -to * to, zero, -p[1] * p[1]};
      q.has_inner = (p[0] > zero || ti > zero);
      q.inner = {one, one, -ti * ti, zero, -p[0] * p[0]};
      q.zmin = -p[4];
      q.zmax = p[4];
      break;
    }
    default: {
      q.outer = {zero, zero, zero, zero, one};  // never inside
      q.zmin = zero;
      q.zmax = zero;
      break;
    }
  }
  return q;
}

/// Is a point inside the theta band of a sphere?
///
/// Polar angle theta lies in [st, st+dt] exactly when cos theta lies in [cos(st+dt), cos(st)],
/// i.e. when r cos(st+dt) <= z <= r cos(st) with r the radius. Two comparisons, no tangents,
/// and no branch on whether either angle is acute, obtuse or exactly a right angle - the
/// version that tested the cone forms and the sign of z instead got the obtuse case backwards
/// and put the band on the wrong side of the equator.
template <typename real_t>
__host__ __device__ inline bool theta_ok(const QuadricShape<real_t>& q, const Vec3<real_t>& v) {
  if (!q.has_theta) { return true; }
  const real_t tol = kSurfTolerance<real_t>();
  const real_t r = sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
  if (r <= tol) { return true; }  // the origin is in every theta band
  return (v.z <= r * q.cos_lo + tol) && (v.z >= r * q.cos_hi - tol);
}

/// The quadric's residual AS A DISTANCE IN MILLIMETRES: |Q| / |grad Q| is the first-order
/// distance to the surface Q = 0.
///
/// The tolerance a containment test compares against is a length - kSurfTolerance is
/// documented as "half-width of the surface band, in mm" - so what it is compared to has to be
/// one too, and a quadric's raw value is not. The scale depends on how the shape was written:
///
///   orb        x^2 + y^2 + z^2 - r^2      gradient 2r      = 120 per mm at r = 60
///   ellipsoid  x^2/a^2 + ... - 1          gradient 2/c     = 0.04 per mm at c = 50
///
/// a factor of three thousand between two shapes in the same engine. So one fixed tolerance is
/// a band 8e-7 mm wide on the orb and 2.5e-3 mm wide on the ellipsoid, and the second is WIDER
/// THAN THE PROBE `is_crossing_to` steps by: both sides of a real crossing then report "on the
/// surface", the crossing is not seen, and the ellipsoid became invisible. In float, where the
/// tolerance is 1e-4; in double, where it is 1e-9, both bands sit far below the probe and
/// nothing changed - which is why this went unnoticed until the render pass became float.
///
/// quadric_normal directly above has always divided by the gradient, for the same reason
/// stated in its own comment. This is that division, in the test that decides what is solid.
template <typename real_t>
__host__ __device__ inline real_t quadric_residual(const Quadric<real_t>& s,
                                                   const Vec3<real_t>& v) {
  const real_t f = quadric_eval(s, v);
  const Vec3<real_t> g = quadric_grad(s, v);
  const real_t gl = sqrt(dot(g, g));
  // At the centre of a sphere the gradient vanishes and the point is not near any surface;
  // the raw value is signed correctly there, which is all this is asked for.
  return (gl > kTolerance<real_t>()) ? f / gl : f;
}

template <typename real_t>
__host__ __device__ inline bool quadric_inside(const QuadricShape<real_t>& q,
                                               const Vec3<real_t>& v) {
  const real_t tol = kSurfTolerance<real_t>();
  if (v.z < q.zmin - tol || v.z > q.zmax + tol) { return false; }
  if (quadric_residual(q.outer, v) > tol) { return false; }
  if (q.has_inner && quadric_residual(q.inner, v) < -tol) { return false; }
  if (!phi_ok(v.x, v.y, q.sphi, q.dphi)) { return false; }
  return theta_ok(q, v);
}

template <typename real_t>
__host__ __device__ inline int quadric_candidates(const QuadricShape<real_t>& q,
                                                  const Vec3<real_t>& p, const Vec3<real_t>& d,
                                                  real_t* t) {
  int n = 0;
  n += quadric_roots(q.outer, p, d, t + n);
  if (q.has_inner && n + 2 <= kMaxCandidates) { n += quadric_roots(q.inner, p, d, t + n); }
  if (n + 1 <= kMaxCandidates) { n += z_plane_root(q.zmin, p, d, t + n); }
  if (n + 1 <= kMaxCandidates) { n += z_plane_root(q.zmax, p, d, t + n); }
  if (n + 2 <= kMaxCandidates) { n += phi_plane_roots(q.sphi, q.dphi, p, d, t + n); }
  if (q.has_theta && n + 4 <= kMaxCandidates) {
    n += quadric_roots(q.theta_lo, p, d, t + n);
    n += quadric_roots(q.theta_hi, p, d, t + n);
  }
  return n;
}

// ---------------------------------------------------------------- torus

/// (sqrt(x^2 + y^2) - R)^2 + z^2 - r^2, the implicit torus.
template <typename real_t>
__host__ __device__ inline real_t torus_form(real_t rtor, real_t r, const Vec3<real_t>& v) {
  const real_t rp = sqrt(v.x * v.x + v.y * v.y) - rtor;
  return rp * rp + v.z * v.z - r * r;
}

/// Real roots of a quartic by Newton polish from a coarse scan. A closed-form quartic solve is
/// possible but notoriously ill-conditioned for near-tangent rays, which is the common case
/// here; a bracketed scan over the interval where the ray is within the torus bounding box is
/// slower but does not lose roots.
template <typename real_t>
__host__ __device__ inline int torus_roots(real_t rtor, real_t r, const Vec3<real_t>& p,
                                           const Vec3<real_t>& d, real_t* t) {
  // Bound the search: the torus lies within |z| <= r and rho in [rtor - r, rtor + r].
  const real_t reach = rtor + r;
  // Solve for the segment of the ray inside the bounding sphere of radius sqrt(reach^2 + r^2).
  const real_t rb = sqrt(reach * reach + r * r);
  const real_t b = real_t(2) * dot(p, d);
  const real_t c = dot(p, p) - rb * rb;
  const real_t disc = b * b - real_t(4) * c;
  if (disc < real_t(0)) { return 0; }
  const real_t sq = sqrt(disc);
  real_t t0 = real_t(-0.5) * (b + sq);
  real_t t1 = real_t(-0.5) * (b - sq);
  if (t1 <= real_t(0)) { return 0; }
  if (t0 < real_t(0)) { t0 = real_t(0); }

  const int kSteps = 96;
  const real_t h = (t1 - t0) / real_t(kSteps);
  int n = 0;
  real_t prev_t = t0;
  real_t prev_f = torus_form(rtor, r, p + t0 * d);
  for (int i = 1; i <= kSteps && n < 4; ++i) {
    const real_t tt = t0 + h * real_t(i);
    const real_t f = torus_form(rtor, r, p + tt * d);
    if ((prev_f <= real_t(0)) != (f <= real_t(0))) {
      // Bisect the bracket; 40 halvings takes h/2^40 well below any tolerance.
      real_t lo = prev_t, hi = tt, flo = prev_f;
      for (int k = 0; k < 40; ++k) {
        const real_t mid = real_t(0.5) * (lo + hi);
        const real_t fm = torus_form(rtor, r, p + mid * d);
        if ((flo <= real_t(0)) != (fm <= real_t(0))) { hi = mid; }
        else { lo = mid; flo = fm; }
      }
      t[n++] = real_t(0.5) * (lo + hi);
    }
    prev_t = tt;
    prev_f = f;
  }
  return n;
}

template <typename real_t>
__host__ __device__ inline bool torus_inside(const real_t* p, const Vec3<real_t>& v) {
  const real_t tol = kSurfTolerance<real_t>();
  if (torus_form(p[2], p[1], v) > tol) { return false; }
  if (p[0] > real_t(0) && torus_form(p[2], p[0], v) < -tol) { return false; }
  return phi_ok(v.x, v.y, p[3], p[4]);
}

// ---------------------------------------------------------------- polycone / polyhedra

/// A z-section: (z, rmin, rmax). Consecutive sections define a cone frustum.
/// Stored in the aux pool as triples.
template <typename real_t>
__host__ __device__ inline Solid<real_t> polycone_section(const Solid<real_t>& s,
                                                          const real_t* aux, int i) {
  const real_t* a = aux + s.a + 3 * i;
  const real_t* b = a + 3;
  Solid<real_t> c{};
  c.type = SolidType::kConeSection;
  const real_t dz = real_t(0.5) * (b[0] - a[0]);
  c.p[0] = a[1];
  c.p[1] = a[2];
  c.p[2] = b[1];
  c.p[3] = b[2];
  c.p[4] = dz;
  c.p[5] = s.p[0];
  c.p[6] = s.p[1];
  return c;
}

/// Centre z of section i, needed because each frustum is built about its own midpoint.
template <typename real_t>
__host__ __device__ inline real_t polycone_zmid(const Solid<real_t>& s, const real_t* aux,
                                                int i) {
  const real_t* a = aux + s.a + 3 * i;
  return real_t(0.5) * (a[0] + a[3]);
}

/// Largest number of sides a polyhedra may have, so the per-section plane buffer can be a
/// fixed-size local array.
constexpr int kMaxPolyhedraSides = 16;

/// The lateral faces of one polyhedra section, as half-space planes n . x <= d.
///
/// A section between z0 and z1 whose inscribed radius runs from r0 to r1 is a prism frustum,
/// and a prism frustum is a convex polyhedron - so its faces are exactly planes, with no
/// approximation anywhere:
///     x cos a_f + y sin a_f - m z <= c,   m = (r1 - r0) / (z1 - z0),  c = r0 - m z0.
/// Treating the section as a circumscribed *cone* and then clipping, which is what an earlier
/// version did, is only right when r0 == r1; as soon as the radius varies the cone's lateral
/// surface is not where the flat faces are, and both entry and exit distances come out wrong.
///
/// Returns the number of planes written (sides), and the two z planes are left to the caller.
template <typename real_t>
__host__ __device__ inline int polyhedra_planes(real_t z0, real_t z1, real_t r0, real_t r1,
                                                int sides, real_t sphi, real_t* out) {
  if (sides < 3 || sides > kMaxPolyhedraSides) { return 0; }
  const real_t dz = z1 - z0;
  const real_t m = (fabs(dz) > kTolerance<real_t>()) ? (r1 - r0) / dz : real_t(0);
  const real_t c = r0 - m * z0;
  const real_t step = units::twopi<real_t>() / real_t(sides);
  for (int f = 0; f < sides; ++f) {
    const real_t a = sphi + step * (real_t(f) + real_t(0.5));
    // Normalise so that plane distances are true distances; only the ratio matters for the
    // ray parameter, but a unit normal keeps the tolerance comparison meaningful.
    const real_t nx = cos(a), ny = sin(a), nz = -m;
    const real_t len = sqrt(nx * nx + ny * ny + nz * nz);
    out[4 * f + 0] = nx / len;
    out[4 * f + 1] = ny / len;
    out[4 * f + 2] = nz / len;
    out[4 * f + 3] = c / len;
  }
  return sides;
}

/// Containment in one polyhedra section: inside the outer frustum, outside the inner one.
template <typename real_t>
__host__ __device__ inline bool polyhedra_section_inside(const real_t* a, int sides, real_t sphi,
                                                         const Vec3<real_t>& q) {
  const real_t tol = kSurfTolerance<real_t>();
  if (q.z < a[0] - tol || q.z > a[3] + tol) { return false; }
  real_t pl[4 * kMaxPolyhedraSides];
  const int n = polyhedra_planes(a[0], a[3], a[2], a[5], sides, sphi, pl);
  if (n == 0 || !planes_inside(pl, n, q)) { return false; }
  if (a[1] > real_t(0) || a[4] > real_t(0)) {
    const int ni = polyhedra_planes(a[0], a[3], a[1], a[4], sides, sphi, pl);
    // Strictly inside the inner frustum means the point is in the hole.
    bool in_hole = true;
    for (int i = 0; i < ni; ++i) {
      const real_t* e = pl + 4 * i;
      if (e[0] * q.x + e[1] * q.y + e[2] * q.z - e[3] >= -tol) { in_hole = false; break; }
    }
    if (in_hole) { return false; }
  }
  return true;
}

/// The six faces of a box, as half-space planes.
template <typename real_t>
__host__ __device__ inline int box_planes(const real_t* p, real_t* out) {
  for (int i = 0; i < 3; ++i) {
    for (int s = 0; s < 2; ++s) {
      const int k = 8 * i + 4 * s;
      out[k] = out[k + 1] = out[k + 2] = real_t(0);
      out[k + i] = (s == 0) ? real_t(1) : real_t(-1);
      out[k + 3] = p[i];
    }
  }
  return 6;
}

/// The six faces of a G4Trd. Half-widths vary linearly in z, so each lateral face is the plane
/// x - b z = a with a the mid-z half-width and b its slope.
template <typename real_t>
__host__ __device__ inline int trd_planes(const real_t* p, real_t* out) {
  const real_t dx1 = p[0], dx2 = p[1], dy1 = p[2], dy2 = p[3], dz = p[4];
  out[0] = 0; out[1] = 0; out[2] = 1;  out[3] = dz;
  out[4] = 0; out[5] = 0; out[6] = -1; out[7] = dz;
  const real_t ax = real_t(0.5) * (dx1 + dx2), bx = (dx2 - dx1) / (real_t(2) * dz);
  const real_t lx = sqrt(real_t(1) + bx * bx);
  out[8]  = real_t(1) / lx;  out[9] = 0;  out[10] = -bx / lx; out[11] = ax / lx;
  out[12] = real_t(-1) / lx; out[13] = 0; out[14] = bx / lx;  out[15] = ax / lx;
  const real_t ay = real_t(0.5) * (dy1 + dy2), by = (dy2 - dy1) / (real_t(2) * dz);
  const real_t ly = sqrt(real_t(1) + by * by);
  out[16] = 0; out[17] = real_t(1) / ly;  out[18] = -by / ly; out[19] = ay / ly;
  out[20] = 0; out[21] = real_t(-1) / ly; out[22] = by / ly;  out[23] = ay / ly;
  return 6;
}

}  // namespace g4gpu::geom
