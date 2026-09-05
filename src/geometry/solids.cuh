// Flat, virtual-free solid representation. Dispatch is an explicit switch on SolidType.
//
// Two families live here, and the split is deliberate.
//
// kBox, kCons and kTrd carry hand-written closed-form DistanceToIn / DistanceToOut. They are
// the three solids example B1 uses, they are validated against Geant4 to the last digit of the
// dose, and they are left exactly as they were.
//
// Everything added since is built on a generic pair of primitives instead: an `inside`
// predicate and a `surface_candidates` routine that returns every ray parameter at which the
// ray meets any bounding surface of the solid. `dist_in` is then the first candidate whose far
// side is inside, and `dist_out` the first whose far side is outside. That costs a handful of
// extra `inside` evaluations per query and buys a very large reduction in the number of places
// a sign or a degenerate case can be got wrong - for eighteen solids, each with entry, exit,
// phi wedges and angular cuts, the closed forms are where the bugs would be.
//
// Most of the curved solids share one axis-aligned quadric,
//     A x^2 + B y^2 + C z^2 + D z + E <= 0,
// which with an optional inner quadric, a z range, a phi wedge and a theta cone covers tubes,
// cones, spheres, orbs, elliptical tubes and cones, ellipsoids, paraboloids and hyperboloids.
// The flat-faced solids share a convex half-space engine. Only the torus needs its own quartic.
#pragma once
#include <cmath>
#include "core/units.cuh"
#include "core/vec3.cuh"
#include "geometry/transform.cuh"

namespace g4gpu::geom {

// Type-aware: 9e99 overflows float to +inf, and 1e-9 sits below float epsilon (~1.2e-7).
// Distances are in mm and the B1 world is 180 mm, so 1e30 is unreachably far either way.
template <typename real_t> __host__ __device__ constexpr real_t kInfinity() { return real_t(1e30); }
template <typename real_t> __host__ __device__ constexpr real_t kTolerance() {
  return (sizeof(real_t) == 4) ? real_t(1e-5) : real_t(1e-9);
}

/// Half-width of the surface band, in mm. Geant4's kCarTolerance is 1e-9 mm; the same value
/// is used here for double and a float-appropriate one for float.
template <typename real_t> __host__ __device__ constexpr real_t kSurfTolerance() {
  return (sizeof(real_t) == 4) ? real_t(1e-4) : real_t(1e-9);
}

enum class SolidType : int {
  // Closed-form, validated against B1.
  kBox = 0,
  kCons = 1,  ///< rmin = 0, full phi - the B1 cone
  kTrd = 2,
  // Generic engine.
  kTubs = 3,
  kConeSection = 4,  ///< the general G4Cons: rmin1/rmax1/rmin2/rmax2, phi wedge
  kOrb = 5,
  kSphere = 6,
  kTorus = 7,
  kPara = 8,
  kTrap = 9,
  kEllipticalTube = 10,
  kEllipsoid = 11,
  kEllipticalCone = 12,
  kParaboloid = 13,
  kHype = 14,
  kTet = 15,
  kPolycone = 16,
  kPolyhedra = 17,
  /// A regular grid of cells, each with its own material. A box for containment and entry;
  /// what differs is inside it. See geometry/voxels.cuh.
  kVoxelGrid = 21,
  /// A triangle mesh with a BVH over it: a CAD import. See geometry/mesh.cuh.
  kMesh = 22,
  // Boolean nodes; `a` and `b` index the solid pool.
  kUnion = 18,
  kSubtraction = 19,
  kIntersection = 20,
};

/// Parameters, by type. Angles in radians, lengths in mm, all half-lengths where Geant4 uses
/// half-lengths.
///
///   kBox            p[0..2] = half x, y, z
///   kCons           p[0] = rmax at -dz, p[1] = rmax at +dz, p[2] = dz  (rmin=0, full phi)
///   kTrd            p[0]=dx1 p[1]=dx2 p[2]=dy1 p[3]=dy2 p[4]=dz
///   kTubs           p[0]=rmin p[1]=rmax p[2]=dz p[3]=sphi p[4]=dphi
///   kConeSection    p[0]=rmin1 p[1]=rmax1 p[2]=rmin2 p[3]=rmax2 p[4]=dz p[5]=sphi p[6]=dphi
///   kOrb            p[0]=r
///   kSphere         p[0]=rmin p[1]=rmax p[2]=sphi p[3]=dphi p[4]=stheta p[5]=dtheta
///   kTorus          p[0]=rmin p[1]=rmax p[2]=rtor p[3]=sphi p[4]=dphi
///   kPara           p[0]=dx p[1]=dy p[2]=dz p[3]=tan(alpha) p[4]=tan(theta)cos(phi)
///                   p[5]=tan(theta)sin(phi)
///   kTrap           a[..] = 6 planes in the aux pool (see make_trap)
///   kEllipticalTube p[0]=dx p[1]=dy p[2]=dz
///   kEllipsoid      p[0]=ax p[1]=by p[2]=cz p[3]=zbottom p[4]=ztop
///   kEllipticalCone p[0]=xSemiAxis p[1]=ySemiAxis p[2]=zMax p[3]=zTopCut
///   kParaboloid     p[0]=dz p[1]=r_at_-dz p[2]=r_at_+dz
///   kHype           p[0]=rmin p[1]=rmax p[2]=tan(stereoIn) p[3]=tan(stereoOut) p[4]=dz
///   kTet            a = plane offset in the aux pool (4 planes)
///   kPolycone       a = z-section offset, b = section count; p[0]=sphi p[1]=dphi
///   kPolyhedra      as kPolycone, plus p[2] = number of sides
///   booleans        a, b = child solid indices
///   kVoxelGrid      p[0..2] = half extent, p[3..5] = cell counts; a = cell-array offset
template <typename real_t>
struct Solid {
  SolidType type;
  real_t p[8];
  /// Placement of this solid inside its parent's frame. -1 means identity, which is the case
  /// for every top-level solid - a placed volume's transform lives on the Volume. Boolean
  /// children use it to offset and rotate the operand.
  int xform = -1;
  int a = 0;  ///< boolean left child, or aux-pool offset
  int b = 0;  ///< boolean right child, or aux-pool count
};

/// Backing storage shared by every solid in a scene: the solid pool that boolean nodes index
/// into, the transforms those children reference, and a pool of reals holding plane equations
/// (for kTrap and kTet) and z-sections (for kPolycone and kPolyhedra).
template <typename real_t>
struct SolidStore {
  const Solid<real_t>* solids = nullptr;
  const Transform<real_t>* xforms = nullptr;
  const real_t* aux = nullptr;
  /// Triangle mesh pools: nine reals per triangle, eight per BVH node. A mesh solid's `a` is
  /// its root node index; see geometry/mesh.cuh for the node layout.
  const real_t* tri = nullptr;
  const real_t* bvh = nullptr;
};

// ---------------------------------------------------------------- Box

template <typename real_t>
__host__ __device__ inline bool box_inside(const real_t* p, const Vec3<real_t>& q) {
  return fabs(q.x) <= p[0] && fabs(q.y) <= p[1] && fabs(q.z) <= p[2];
}

/// Slab exit distance. Assumes q is inside or on the surface.
template <typename real_t>
__host__ __device__ inline real_t box_dist_out(const real_t* p, const Vec3<real_t>& q,
                                               const Vec3<real_t>& d) {
  real_t t = kInfinity<real_t>();
  const real_t qq[3] = {q.x, q.y, q.z};
  const real_t dd[3] = {d.x, d.y, d.z};
  for (int i = 0; i < 3; ++i) {
    if (dd[i] > kTolerance<real_t>())       { t = fmin(t, (p[i] - qq[i]) / dd[i]); }
    else if (dd[i] < -kTolerance<real_t>()) { t = fmin(t, (-p[i] - qq[i]) / dd[i]); }
  }
  return (t < real_t(0)) ? real_t(0) : t;
}

/// Slab entry distance from outside; kInfinity on a miss.
template <typename real_t>
__host__ __device__ inline real_t box_dist_in(const real_t* p, const Vec3<real_t>& q,
                                              const Vec3<real_t>& d) {
  real_t tmin = real_t(0), tmax = kInfinity<real_t>();
  const real_t qq[3] = {q.x, q.y, q.z};
  const real_t dd[3] = {d.x, d.y, d.z};
  for (int i = 0; i < 3; ++i) {
    if (fabs(dd[i]) < kTolerance<real_t>()) {
      if (fabs(qq[i]) > p[i]) { return kInfinity<real_t>(); }  // parallel and outside the slab
      continue;
    }
    real_t t1 = (-p[i] - qq[i]) / dd[i];
    real_t t2 = (p[i] - qq[i]) / dd[i];
    if (t1 > t2) { const real_t s = t1; t1 = t2; t2 = s; }
    tmin = fmax(tmin, t1);
    tmax = fmin(tmax, t2);
    if (tmin > tmax) { return kInfinity<real_t>(); }
  }
  return tmin;
}

// ---------------------------------------------------------------- Cons (rmin=0, full phi)
// Lateral surface r(z) = a + b*z, with a = (r1+r2)/2 and b = (r2-r1)/(2*dz).

template <typename real_t>
__host__ __device__ inline void cons_ab(const real_t* p, real_t& a, real_t& b) {
  a = real_t(0.5) * (p[0] + p[1]);
  b = (p[1] - p[0]) / (real_t(2) * p[2]);
}

template <typename real_t>
__host__ __device__ inline bool cons_inside(const real_t* p, const Vec3<real_t>& q) {
  if (fabs(q.z) > p[2]) { return false; }
  real_t a, b; cons_ab(p, a, b);
  const real_t r = a + b * q.z;
  return (q.x * q.x + q.y * q.y) <= r * r;
}

/// Roots of the ray/cone quadratic. The z extent is tested by the caller.
template <typename real_t>
__host__ __device__ inline void cons_lateral_roots(const real_t* p, const Vec3<real_t>& q,
                                                   const Vec3<real_t>& d, real_t& t1, real_t& t2) {
  real_t a, b; cons_ab(p, a, b);
  const real_t A = d.x * d.x + d.y * d.y - b * b * d.z * d.z;
  const real_t B = real_t(2) * (q.x * d.x + q.y * d.y - b * b * q.z * d.z - a * b * d.z);
  const real_t C = q.x * q.x + q.y * q.y - (a + b * q.z) * (a + b * q.z);
  t1 = t2 = kInfinity<real_t>();
  if (fabs(A) < kTolerance<real_t>()) {
    if (fabs(B) > kTolerance<real_t>()) { t1 = -C / B; }
    return;
  }
  const real_t disc = B * B - real_t(4) * A * C;
  if (disc < real_t(0)) { return; }
  const real_t sq = sqrt(disc);
  t1 = (-B - sq) / (real_t(2) * A);
  t2 = (-B + sq) / (real_t(2) * A);
  if (t1 > t2) { const real_t s = t1; t1 = t2; t2 = s; }
}

template <typename real_t>
__host__ __device__ inline real_t cons_dist_out(const real_t* p, const Vec3<real_t>& q,
                                                const Vec3<real_t>& d) {
  real_t best = kInfinity<real_t>();
  // z caps
  if (d.z > kTolerance<real_t>())       { best = fmin(best, (p[2] - q.z) / d.z); }
  else if (d.z < -kTolerance<real_t>()) { best = fmin(best, (-p[2] - q.z) / d.z); }
  // lateral surface
  real_t t1, t2; cons_lateral_roots(p, q, d, t1, t2);
  const real_t cand[2] = {t1, t2};
  for (int i = 0; i < 2; ++i) {
    const real_t t = cand[i];
    if (t > kTolerance<real_t>() && t < best && fabs(q.z + t * d.z) <= p[2]) { best = t; }
  }
  return (best < real_t(0)) ? real_t(0) : best;
}

template <typename real_t>
__host__ __device__ inline real_t cons_dist_in(const real_t* p, const Vec3<real_t>& q,
                                               const Vec3<real_t>& d) {
  real_t a, b; cons_ab(p, a, b);
  real_t best = kInfinity<real_t>();
  // end caps: hit the disc at z = +/-dz within r(z)
  for (int s = -1; s <= 1; s += 2) {
    if (fabs(d.z) < kTolerance<real_t>()) { continue; }
    const real_t zc = real_t(s) * p[2];
    const real_t t = (zc - q.z) / d.z;
    if (t <= kTolerance<real_t>() || t >= best) { continue; }
    const real_t hx = q.x + t * d.x, hy = q.y + t * d.y;
    const real_t r = a + b * zc;
    if (hx * hx + hy * hy <= r * r) { best = t; }
  }
  // lateral surface
  real_t t1, t2; cons_lateral_roots(p, q, d, t1, t2);
  const real_t cand[2] = {t1, t2};
  for (int i = 0; i < 2; ++i) {
    const real_t t = cand[i];
    if (t <= kTolerance<real_t>() || t >= best) { continue; }
    const real_t hz = q.z + t * d.z;
    if (fabs(hz) <= p[2] && (a + b * hz) >= real_t(0)) { best = t; }
  }
  return best;
}

// ---------------------------------------------------------------- Trd
// Six planar faces, treated as an intersection of half-spaces and ray-clipped. This
// handles the general dx1 != dx2, dy1 != dy2 case, not only the constant-x form B1 uses.

/// Fills n[] and off[] with the six outward plane normals and offsets: dot(n,q) <= off inside.
template <typename real_t>
__host__ __device__ inline void trd_planes(const real_t* p, Vec3<real_t>* n, real_t* off) {
  const real_t dx1 = p[0], dx2 = p[1], dy1 = p[2], dy2 = p[3], dz = p[4];
  const real_t tz = real_t(2) * dz;
  // +/-x faces: 2*dz*x -/+ (dx2-dx1)*z = dz*(dx1+dx2)
  n[0] = Vec3<real_t>{tz, real_t(0), -(dx2 - dx1)};  off[0] = dz * (dx1 + dx2);
  n[1] = Vec3<real_t>{-tz, real_t(0), -(dx2 - dx1)}; off[1] = dz * (dx1 + dx2);
  // +/-y faces
  n[2] = Vec3<real_t>{real_t(0), tz, -(dy2 - dy1)};  off[2] = dz * (dy1 + dy2);
  n[3] = Vec3<real_t>{real_t(0), -tz, -(dy2 - dy1)}; off[3] = dz * (dy1 + dy2);
  // +/-z caps
  n[4] = Vec3<real_t>{real_t(0), real_t(0), real_t(1)};  off[4] = dz;
  n[5] = Vec3<real_t>{real_t(0), real_t(0), real_t(-1)}; off[5] = dz;
}

template <typename real_t>
__host__ __device__ inline bool trd_inside(const real_t* p, const Vec3<real_t>& q) {
  Vec3<real_t> n[6]; real_t off[6];
  trd_planes(p, n, off);
  for (int i = 0; i < 6; ++i) {
    if (dot(n[i], q) > off[i] + kTolerance<real_t>()) { return false; }
  }
  return true;
}

template <typename real_t>
__host__ __device__ inline real_t trd_dist_out(const real_t* p, const Vec3<real_t>& q,
                                               const Vec3<real_t>& d) {
  Vec3<real_t> n[6]; real_t off[6];
  trd_planes(p, n, off);
  real_t t = kInfinity<real_t>();
  for (int i = 0; i < 6; ++i) {
    const real_t nd = dot(n[i], d);
    if (nd > kTolerance<real_t>()) {  // heading toward this face
      t = fmin(t, (off[i] - dot(n[i], q)) / nd);
    }
  }
  return (t < real_t(0)) ? real_t(0) : t;
}

template <typename real_t>
__host__ __device__ inline real_t trd_dist_in(const real_t* p, const Vec3<real_t>& q,
                                              const Vec3<real_t>& d) {
  Vec3<real_t> n[6]; real_t off[6];
  trd_planes(p, n, off);
  real_t tmin = real_t(0), tmax = kInfinity<real_t>();
  for (int i = 0; i < 6; ++i) {
    const real_t nd = dot(n[i], d);
    const real_t dist = off[i] - dot(n[i], q);  // positive means inside this half-space
    if (fabs(nd) < kTolerance<real_t>()) {
      if (dist < real_t(0)) { return kInfinity<real_t>(); }  // parallel and outside
      continue;
    }
    const real_t t = dist / nd;
    if (nd > real_t(0)) { tmax = fmin(tmax, t); }  // exiting this half-space
    else                { tmin = fmax(tmin, t); }  // entering
    if (tmin > tmax) { return kInfinity<real_t>(); }
  }
  return tmin;
}

// ---------------------------------------------------------------- surface normals
// Needed only for rendering: transport never asks which way a surface faces.

template <typename real_t>
__host__ __device__ inline Vec3<real_t> box_normal(const real_t* p, const Vec3<real_t>& q) {
  // The face whose plane the point is closest to.
  const real_t gx = fabs(fabs(q.x) - p[0]);
  const real_t gy = fabs(fabs(q.y) - p[1]);
  const real_t gz = fabs(fabs(q.z) - p[2]);
  if (gx <= gy && gx <= gz) { return {(q.x > real_t(0)) ? real_t(1) : real_t(-1), 0, 0}; }
  if (gy <= gz)             { return {0, (q.y > real_t(0)) ? real_t(1) : real_t(-1), 0}; }
  return {0, 0, (q.z > real_t(0)) ? real_t(1) : real_t(-1)};
}

/// Lateral surface is sqrt(x^2+y^2) = a + b*z, so grad F = (x/r, y/r, -b).
template <typename real_t>
__host__ __device__ inline Vec3<real_t> cons_normal(const real_t* p, const Vec3<real_t>& q) {
  real_t a, b; cons_ab(p, a, b);
  if (fabs(fabs(q.z) - p[2]) < real_t(1e-3)) {
    return {0, 0, (q.z > real_t(0)) ? real_t(1) : real_t(-1)};
  }
  const real_t r = sqrt(q.x * q.x + q.y * q.y);
  if (r < real_t(1e-9)) { return {0, 0, real_t(1)}; }
  return normalize(Vec3<real_t>{q.x / r, q.y / r, -b});
}

/// The half-space plane the point is closest to satisfying with equality.
template <typename real_t>
__host__ __device__ inline Vec3<real_t> trd_normal(const real_t* p, const Vec3<real_t>& q) {
  Vec3<real_t> n[6]; real_t off[6];
  trd_planes(p, n, off);
  int best = 0;
  real_t best_gap = kInfinity<real_t>();
  for (int i = 0; i < 6; ++i) {
    const real_t len = mag(n[i]);
    if (len <= real_t(0)) { continue; }
    const real_t gap = fabs(off[i] - dot(n[i], q)) / len;  // distance to that plane
    if (gap < best_gap) { best_gap = gap; best = i; }
  }
  return normalize(n[best]);
}

}  // namespace g4gpu::geom

#include "geometry/solids_generic.cuh"
#include "geometry/mesh.cuh"

namespace g4gpu::geom {
// ---------------------------------------------------------------- dispatch
//
// Appended to solids.cuh. `inside`, `dist_in`, `dist_out`, `normal_at` and the two safeties
// take a SolidStore so that boolean nodes can reach their children; the shapes that ignore it
// take it and drop it, which keeps one call shape for everything.

template <typename real_t>
__host__ __device__ inline bool inside(const SolidStore<real_t>& st, const Solid<real_t>& s,
                                       const Vec3<real_t>& q_parent);

/// Containment, with @p q already expressed in this solid's own frame.
///
/// The split from `inside` below matters: the distance routines transform the ray into the
/// solid's frame once, then classify points along it. Calling the transforming `inside` from
/// there applies the placement a second time, which is invisible for an unplaced solid and
/// silently wrong for every boolean whose operand is offset or rotated.
///
/// Recursion depth here is the depth of the boolean tree - see kMaxBooleanDepth.
template <typename real_t>
__host__ __device__ inline bool inside_local(const SolidStore<real_t>& st,
                                             const Solid<real_t>& s, const Vec3<real_t>& q) {
  switch (s.type) {
    case SolidType::kBox:
    // A voxel grid is a box as far as containment and entry go; what differs is that the
    // material inside it varies, which is the navigator's business, not the solid's.
    case SolidType::kVoxelGrid:
      return box_inside(s.p, q);
    case SolidType::kCons: return cons_inside(s.p, q);
    case SolidType::kTrd:  return trd_inside(s.p, q);

    case SolidType::kTubs:
    case SolidType::kConeSection:
    case SolidType::kOrb:
    case SolidType::kSphere:
    case SolidType::kEllipticalTube:
    case SolidType::kEllipsoid:
    case SolidType::kEllipticalCone:
    case SolidType::kParaboloid:
    case SolidType::kHype:
      return quadric_inside(quadric_of(s), q);

    case SolidType::kTorus: return torus_inside(s.p, q);

    case SolidType::kMesh: return mesh_inside(st, s, q);


    case SolidType::kPara: {
      real_t pl[24];
      const int n = para_planes(s.p, pl);
      return planes_inside(pl, n, q);
    }
    case SolidType::kTrap:
    case SolidType::kTet:
      return (st.aux != nullptr) && planes_inside(st.aux + s.a, s.b, q);

    case SolidType::kPolycone:
    case SolidType::kPolyhedra: {
      if (st.aux == nullptr) { return false; }
      const int sides = (s.type == SolidType::kPolyhedra) ? static_cast<int>(s.p[2] + real_t(0.5))
                                                          : 0;
      if (!phi_ok(q.x, q.y, s.p[0], s.p[1])) { return false; }
      for (int i = 0; i < s.b; ++i) {
        const real_t* a = st.aux + s.a + 3 * i;
        if (sides > 0) {
          if (polyhedra_section_inside(a, sides, s.p[0], q)) { return true; }
        } else {
          const Solid<real_t> sec = polycone_section(s, st.aux, i);
          const real_t zmid = polycone_zmid(s, st.aux, i);
          const Vec3<real_t> v{q.x, q.y, q.z - zmid};
          if (quadric_inside(quadric_of(sec), v)) { return true; }
        }
      }
      return false;
    }

    case SolidType::kUnion:
      return inside(st, st.solids[s.a], q) || inside(st, st.solids[s.b], q);
    case SolidType::kSubtraction:
      return inside(st, st.solids[s.a], q) && !inside(st, st.solids[s.b], q);
    case SolidType::kIntersection:
      return inside(st, st.solids[s.a], q) && inside(st, st.solids[s.b], q);
  }
  return false;
}

/// Containment for a point in this solid's PARENT frame: applies the placement, then defers.
template <typename real_t>
__host__ __device__ inline bool inside(const SolidStore<real_t>& st, const Solid<real_t>& s,
                                       const Vec3<real_t>& q_parent) {
  const Vec3<real_t> q =
      (s.xform >= 0 && st.xforms != nullptr) ? to_local(st.xforms[s.xform], q_parent) : q_parent;
  return inside_local(st, s, q);
}

/// Every ray parameter at which the ray meets a bounding surface of @p s, written into @p t
/// (capacity kMaxCandidates). Booleans hand back their children's candidates, so the caller
/// has to make room for both; that is why boolean distance queries below batch by child.
template <typename real_t>
__host__ __device__ inline int surface_candidates(const SolidStore<real_t>& st,
                                                  const Solid<real_t>& s, const Vec3<real_t>& p,
                                                  const Vec3<real_t>& d, real_t* t) {
  switch (s.type) {
    case SolidType::kMesh:
      // Deliberately none. A mesh can cross a ray more times than this array holds, and a
      // truncated crossing list makes the boolean engine build a solid that is quietly the
      // wrong shape. G4TessellatedSolid refuses to be a boolean operand, and this returning
      // zero is what that refusal rests on - a mesh reaching here means one slipped through.
      return 0;

    case SolidType::kVoxelGrid:
    case SolidType::kBox: {
      real_t pl[24];
      const int n = box_planes(s.p, pl);
      return planes_candidates(pl, n, p, d, t);
    }
    case SolidType::kTrd: {
      real_t pl[24];
      const int n = trd_planes(s.p, pl);
      return planes_candidates(pl, n, p, d, t);
    }
    case SolidType::kCons:
    case SolidType::kTubs:
    case SolidType::kConeSection:
    case SolidType::kOrb:
    case SolidType::kSphere:
    case SolidType::kEllipticalTube:
    case SolidType::kEllipsoid:
    case SolidType::kEllipticalCone:
    case SolidType::kParaboloid:
    case SolidType::kHype:
      return quadric_candidates(quadric_of(s), p, d, t);

    case SolidType::kTorus: {
      int n = torus_roots(s.p[2], s.p[1], p, d, t);
      if (s.p[0] > real_t(0) && n + 4 <= kMaxCandidates) {
        n += torus_roots(s.p[2], s.p[0], p, d, t + n);
      }
      if (n + 2 <= kMaxCandidates) { n += phi_plane_roots(s.p[3], s.p[4], p, d, t + n); }
      return n;
    }

    case SolidType::kPara: {
      real_t pl[24];
      const int n = para_planes(s.p, pl);
      return planes_candidates(pl, n, p, d, t);
    }
    case SolidType::kTrap:
    case SolidType::kTet:
      return (st.aux != nullptr) ? planes_candidates(st.aux + s.a, s.b, p, d, t) : 0;

    default:
      return 0;  // polycone and booleans are handled by their own distance routines
  }
}

/// Deepest boolean nesting the distance routines will follow. Four levels is enough for the
/// GUI's compose-then-compose editing and keeps the device stack bounded.
constexpr int kMaxBooleanDepth = 4;

template <typename real_t>
__host__ __device__ inline real_t generic_dist(const SolidStore<real_t>& st,
                                               const Solid<real_t>& s, const Vec3<real_t>& p,
                                               const Vec3<real_t>& d, bool want_inside, int depth);

/// Is @p tc a genuine crossing into containment @p want_inside, rather than a graze?
///
/// Testing only the far side is not enough. A ray tangent to a surface has a double root
/// there, and at a tangency the point a probe-step beyond is still within tolerance of the
/// surface - so it reports as "inside" and the graze is mistaken for an entry. No probe
/// distance fixes that, because at a tangency the surface form barely changes along the ray.
/// Requiring containment to actually *differ* on the two sides distinguishes the two cases
/// exactly: a crossing flips it, a graze does not.
template <typename real_t>
__host__ __device__ inline bool is_crossing_to(const SolidStore<real_t>& st,
                                               const Solid<real_t>& s, const Vec3<real_t>& q,
                                               const Vec3<real_t>& d, real_t tc,
                                               bool want_inside) {
  const real_t probe = kProbe<real_t>();
  if (inside_local(st, s, q + (tc + probe) * d) != want_inside) { return false; }
  return inside_local(st, s, q + (tc - probe) * d) != want_inside;
}

/// Full distance dispatch, carrying the boolean recursion depth. The closed forms are used
/// only for an unplaced solid; see dist_out below for why.
template <typename real_t>
__host__ __device__ inline real_t dist_any(const SolidStore<real_t>& st, const Solid<real_t>& s,
                                           const Vec3<real_t>& q, const Vec3<real_t>& d,
                                           bool want_inside, int depth);

/// Walks the ray one crossing at a time until containment matches @p want_inside. Used for
/// booleans, whose surface set is the union of their children's - the walk asks the *combined*
/// predicate at each child crossing, so a face of A buried inside B is correctly not a surface.
///
/// Two details that are easy to get wrong. The point is already in the boolean node's frame,
/// so containment is asked with inside_local; going through the transforming `inside` would
/// apply the node's own placement a second time. And the accumulated probe offsets are tracked
/// separately from the crossing itself, so that a ray crossing many child faces does not drift
/// short by one probe per crossing.
template <typename real_t>
__host__ __device__ inline real_t boolean_dist(const SolidStore<real_t>& st,
                                               const Solid<real_t>& s, const Vec3<real_t>& p,
                                               const Vec3<real_t>& d, bool want_inside,
                                               int depth) {
  if (depth >= kMaxBooleanDepth) { return kInfinity<real_t>(); }
  const Solid<real_t>& L = st.solids[s.a];
  const Solid<real_t>& R = st.solids[s.b];
  const real_t probe = kProbe<real_t>();

  real_t travelled = real_t(0);
  Vec3<real_t> q = p;
  for (int iter = 0; iter < 128; ++iter) {
    real_t step = kInfinity<real_t>();
    for (int side = 0; side < 2; ++side) {
      const Solid<real_t>& c = (side == 0) ? L : R;
      const bool in_c = inside(st, c, q);
      const real_t dc = dist_any(st, c, q, d, !in_c, depth + 1);
      if (dc < step) { step = dc; }
    }
    if (step >= kInfinity<real_t>()) { return kInfinity<real_t>(); }
    const real_t crossing = travelled + step;
    travelled = crossing + probe;
    q = p + travelled * d;
    if (inside_local(st, s, q) == want_inside) { return crossing; }
  }
  return kInfinity<real_t>();
}

/// Distance along @p d from @p p to the first surface where containment becomes
/// @p want_inside. This is the single engine behind both dist_in and dist_out.
template <typename real_t>
__host__ __device__ inline real_t generic_dist(const SolidStore<real_t>& st,
                                               const Solid<real_t>& s, const Vec3<real_t>& p,
                                               const Vec3<real_t>& d, bool want_inside,
                                               int depth) {
  const Vec3<real_t> q =
      (s.xform >= 0 && st.xforms != nullptr) ? to_local(st.xforms[s.xform], p) : p;
  const Vec3<real_t> dd =
      (s.xform >= 0 && st.xforms != nullptr) ? dir_to_local(st.xforms[s.xform], d) : d;

  switch (s.type) {
    case SolidType::kUnion:
    case SolidType::kSubtraction:
    case SolidType::kIntersection:
      return boolean_dist(st, s, q, dd, want_inside, depth);

    case SolidType::kMesh:
      return mesh_dist(st, s, q, dd, want_inside);


    case SolidType::kPolycone:
    case SolidType::kPolyhedra: {
      if (st.aux == nullptr) { return kInfinity<real_t>(); }
      real_t best = kInfinity<real_t>();
      const int sides = (s.type == SolidType::kPolyhedra)
                            ? static_cast<int>(s.p[2] + real_t(0.5)) : 0;
      real_t cand[kMaxCandidates];
      for (int i = 0; i < s.b; ++i) {
        const real_t* a = st.aux + s.a + 3 * i;
        int n = 0;
        if (sides > 0) {
          // Exact frustum faces, plus the two z planes bounding the section.
          real_t pl[4 * kMaxPolyhedraSides];
          const int no = polyhedra_planes(a[0], a[3], a[2], a[5], sides, s.p[0], pl);
          n = planes_candidates(pl, no, q, dd, cand);
          for (int k = 0; k < n; ++k) {
            const real_t tc = cand[k];
            if (tc <= kSurfTolerance<real_t>() || tc >= best) { continue; }
            if (is_crossing_to(st, s, q, dd, tc, want_inside)) { best = tc; }
          }
          if (a[1] > real_t(0) || a[4] > real_t(0)) {
            const int ni = polyhedra_planes(a[0], a[3], a[1], a[4], sides, s.p[0], pl);
            n = planes_candidates(pl, ni, q, dd, cand);
            for (int k = 0; k < n; ++k) {
              const real_t tc = cand[k];
              if (tc <= kSurfTolerance<real_t>() || tc >= best) { continue; }
              if (is_crossing_to(st, s, q, dd, tc, want_inside)) { best = tc; }
            }
          }
          n = 0;
          n += z_plane_root(a[0], q, dd, cand + n);
          n += z_plane_root(a[3], q, dd, cand + n);
          n += phi_plane_roots(s.p[0], s.p[1], q, dd, cand + n);
        } else {
          const Solid<real_t> sec = polycone_section(s, st.aux, i);
          const real_t zmid = polycone_zmid(s, st.aux, i);
          const Vec3<real_t> v{q.x, q.y, q.z - zmid};
          n = quadric_candidates(quadric_of(sec), v, dd, cand);
        }
        for (int k = 0; k < n; ++k) {
          const real_t tc = cand[k];
          if (tc <= kSurfTolerance<real_t>() || tc >= best) { continue; }
          if (is_crossing_to(st, s, q, dd, tc, want_inside)) { best = tc; }
        }
      }
      return best;
    }

    default: {
      real_t cand[kMaxCandidates];
      const int n = surface_candidates(st, s, q, dd, cand);
      real_t best = kInfinity<real_t>();
      for (int k = 0; k < n; ++k) {
        const real_t tc = cand[k];
        if (tc <= kSurfTolerance<real_t>() || tc >= best) { continue; }
        if (is_crossing_to(st, s, q, dd, tc, want_inside)) { best = tc; }
      }
      return best;
    }
  }
}

template <typename real_t>
__host__ __device__ inline real_t dist_any(const SolidStore<real_t>& st, const Solid<real_t>& s,
                                           const Vec3<real_t>& q, const Vec3<real_t>& d,
                                           bool want_inside, int depth) {
  // The closed forms take the point in the solid's own frame and know nothing about a
  // placement, so they are only usable when there is none. A boolean operand carries its own
  // transform, and routing that through the closed form would silently ignore it.
  if (s.xform < 0) {
    switch (s.type) {
      case SolidType::kVoxelGrid:
      case SolidType::kBox:
        return want_inside ? box_dist_in(s.p, q, d) : box_dist_out(s.p, q, d);
      case SolidType::kCons:
        return want_inside ? cons_dist_in(s.p, q, d) : cons_dist_out(s.p, q, d);
      case SolidType::kTrd:
        return want_inside ? trd_dist_in(s.p, q, d) : trd_dist_out(s.p, q, d);
      default: break;
    }
  }
  return generic_dist(st, s, q, d, want_inside, depth);
}

template <typename real_t>
__host__ __device__ inline real_t dist_out(const SolidStore<real_t>& st, const Solid<real_t>& s,
                                           const Vec3<real_t>& q, const Vec3<real_t>& d) {
  return dist_any(st, s, q, d, false, 0);
}

template <typename real_t>
__host__ __device__ inline real_t dist_in(const SolidStore<real_t>& st, const Solid<real_t>& s,
                                          const Vec3<real_t>& q, const Vec3<real_t>& d) {
  return dist_any(st, s, q, d, true, 0);
}

/// Central difference of the containment indicator: the normal of last resort.
///
/// Coarse by construction. Each component is one of -2, 0 and +2, so the direction it returns
/// is one of 26 - which is why a sphere shaded with it looked like a faceted shell, and why
/// the quadric shapes and the booleans have analytic normals instead. What is left on this
/// path is the torus, a voxel grid, and the shapes whose surfaces are flat anyway.
template <typename real_t>
__host__ __device__ inline Vec3<real_t> numerical_normal(const SolidStore<real_t>& st,
                                                         const Solid<real_t>& s,
                                                         const Vec3<real_t>& q) {
  const real_t h = kProbe<real_t>() * real_t(10);
  auto f = [&](const Vec3<real_t>& v) -> real_t {
    return inside(st, s, v) ? real_t(-1) : real_t(1);
  };
  Vec3<real_t> g{f({q.x + h, q.y, q.z}) - f({q.x - h, q.y, q.z}),
                 f({q.x, q.y + h, q.z}) - f({q.x, q.y - h, q.z}),
                 f({q.x, q.y, q.z + h}) - f({q.x, q.y, q.z - h})};
  if (dot(g, g) < kTolerance<real_t>()) {
    // Flat region of the indicator: fall back to the radial direction, which is right for
    // every curved solid here and harmless for the rest.
    g = Vec3<real_t>{q.x, q.y, q.z};
    if (dot(g, g) < kTolerance<real_t>()) { return Vec3<real_t>{0, 0, real_t(1)}; }
  }
  return normalize(g);
}

/// Outward surface normal near @p q. For the generic shapes this is the gradient of whichever
/// bounding form the point is closest to satisfying with equality; it is used for shading and
/// for nothing that affects physics, so a numerical gradient is adequate where the analytic one
/// would be fiddly.
template <typename real_t>
__host__ __device__ inline Vec3<real_t> normal_at(const SolidStore<real_t>& st,
                                                  const Solid<real_t>& s,
                                                  const Vec3<real_t>& q_parent) {
  const Vec3<real_t> q =
      (s.xform >= 0 && st.xforms != nullptr) ? to_local(st.xforms[s.xform], q_parent) : q_parent;
  switch (s.type) {
    case SolidType::kBox:  return box_normal(s.p, q);
    case SolidType::kCons: return cons_normal(s.p, q);
    case SolidType::kTrd:  return trd_normal(s.p, q);
    case SolidType::kPara: {
      real_t pl[24];
      const int n = para_planes(s.p, pl);
      return planes_normal(pl, n, q);
    }
    case SolidType::kTrap:
    case SolidType::kTet:
      if (st.aux != nullptr) { return planes_normal(st.aux + s.a, s.b, q); }
      return Vec3<real_t>{0, 0, real_t(1)};
    case SolidType::kUnion:
    case SolidType::kSubtraction:
    case SolidType::kIntersection: {
      // A point on a boolean's surface lies on one operand's surface. Which one is found by
      // asking whether that operand's containment *changes* within a probe of the point: it
      // does only on that operand's own boundary. The answer is then that operand's normal,
      // flipped for the subtracted right-hand side, where the material is on the other side.
      //
      // Six containment queries rather than a distance-to-surface call, because safety.cuh
      // includes this file and cannot be included back. Six queries at a probe distance are
      // local and cheap, and the only case they cannot separate is a point exactly on the
      // edge where the two surfaces meet - where either normal is as good as the other.
      //
      // Without this a boolean fell to the numerical fallback below, whose gradient can only
      // point in 26 directions, so an Orb with a cylinder subtracted out of it rendered as a
      // faceted lump. Same cause as the sphere; see quadric_normal.
      if (st.solids == nullptr || s.a < 0 || s.b < 0) { return numerical_normal(st, s, q); }
      const real_t h = kProbe<real_t>() * real_t(20);
      auto changes_near = [&](const Solid<real_t>& x) {
        const bool c = inside(st, x, q);
        return inside(st, x, {q.x + h, q.y, q.z}) != c
               || inside(st, x, {q.x - h, q.y, q.z}) != c
               || inside(st, x, {q.x, q.y + h, q.z}) != c
               || inside(st, x, {q.x, q.y - h, q.z}) != c
               || inside(st, x, {q.x, q.y, q.z + h}) != c
               || inside(st, x, {q.x, q.y, q.z - h}) != c;
      };
      const Solid<real_t>& A = st.solids[s.a];
      const Solid<real_t>& B = st.solids[s.b];
      if (changes_near(A)) { return normal_at(st, A, q); }
      if (changes_near(B)) {
        const Vec3<real_t> nb = normal_at(st, B, q);
        return (s.type == SolidType::kSubtraction) ? real_t(-1) * nb : nb;
      }
      return numerical_normal(st, s, q);  // on neither surface: nothing better to say
    }
    default: {
      // The curved solids have an analytic normal: the gradient of whichever bounding form
      // the point is on. See quadric_normal, and the note there on why the numerical
      // alternative made a sphere look faceted.
      if (is_quadric_shape(s.type)) { return quadric_normal(quadric_of(s), q); }
      return numerical_normal(st, s, q);
    }
  }
}

// ------------------------------------------------------- store-free convenience overloads
//
// A solid with no boolean children and no aux data does not need the store. These keep the
// simple call readable and are what the closed-form solids and the tests use.

template <typename real_t>
__host__ __device__ inline bool inside(const Solid<real_t>& s, const Vec3<real_t>& q) {
  return inside(SolidStore<real_t>{}, s, q);
}
template <typename real_t>
__host__ __device__ inline real_t dist_out(const Solid<real_t>& s, const Vec3<real_t>& q,
                                           const Vec3<real_t>& d) {
  return dist_out(SolidStore<real_t>{}, s, q, d);
}
template <typename real_t>
__host__ __device__ inline real_t dist_in(const Solid<real_t>& s, const Vec3<real_t>& q,
                                          const Vec3<real_t>& d) {
  return dist_in(SolidStore<real_t>{}, s, q, d);
}
template <typename real_t>
__host__ __device__ inline Vec3<real_t> normal_at(const Solid<real_t>& s,
                                                  const Vec3<real_t>& q) {
  return normal_at(SolidStore<real_t>{}, s, q);
}

}  // namespace g4gpu::geom
