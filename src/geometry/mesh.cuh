// Triangle-mesh solids: a CAD import, transported rather than approximated by its bounding box.
//
// This file is included from the middle of solids.cuh, after Solid and SolidStore are defined
// and before the dispatch functions that call into it - the same arrangement as
// solids_generic.cuh, and for the same reason: the shape routines need the solid record, and
// the dispatch needs the shape routines.
//
// A mesh is a list of triangles plus a BVH over them, both in pools on the SolidStore. The
// solid record carries:
//
//   p[0..2]  bounding-box half-extents, in the solid's own frame
//   p[3..5]  bounding-box centre
//   p[6]     the enclosed volume, computed on the host by the divergence theorem
//   a        index of this mesh's BVH root node
//   b        triangle count, for diagnostics
//
// Containment is a parity test: fire a ray from the point, count triangle crossings, odd means
// inside. That is the part most likely to be wrong, and the reason is worth stating up front.
// A ray passing exactly through a shared edge or a vertex is counted twice or not at all, and
// on a CAD model - where vertices sit on round coordinates and faces are axis-aligned - a
// fixed axis direction hits that case constantly rather than rarely. So the direction is
// derived from the point (deterministic, but not aligned with anything), and a hit landing
// within tolerance of a triangle edge triggers a re-cast along a different derived direction.
//
// A mesh may not be an operand of a boolean. The boolean engine classifies intervals from the
// list of surface crossings its operands report, and a mesh can have more crossings along one
// ray than that list holds; truncating it would produce a solid that is subtly the wrong shape
// with nothing to indicate it. G4TessellatedSolid refuses at construction instead.

#pragma once

namespace g4gpu::geom {

/// Eight reals per BVH node: six of bounding box, then `first` and `count` packed as reals.
/// count == 0 marks an interior node, whose children are at `first` and `first + 1`.
/// Packing the indices as reals keeps it to one pool; the alternative was a parallel int
/// array and a second pointer to keep in step with it.
constexpr int kBvhStride = 8;
/// Deep enough for a median-split tree over a few million triangles (2^40 leaves), and the
/// traversal stack is this many ints on the device stack.
constexpr int kBvhMaxDepth = 40;
/// Triangles per leaf. Small enough that a leaf test is a handful of Moller-Trumbores, large
/// enough that the tree does not cost more than it saves.
constexpr int kBvhLeafSize = 4;

/// Moller-Trumbore. Returns true and fills @p t on a hit; @p edge_dist is how close the hit
/// came to a triangle edge in barycentric terms, which the parity test uses to decide whether
/// to trust the count.
template <typename real_t>
__host__ __device__ inline bool ray_triangle(const real_t* v, const Vec3<real_t>& o,
                                             const Vec3<real_t>& d, real_t& t,
                                             real_t& edge_dist) {
  const Vec3<real_t> v0{v[0], v[1], v[2]};
  const Vec3<real_t> v1{v[3], v[4], v[5]};
  const Vec3<real_t> v2{v[6], v[7], v[8]};
  const Vec3<real_t> e1 = v1 - v0;
  const Vec3<real_t> e2 = v2 - v0;
  const Vec3<real_t> pv = cross(d, e2);
  const real_t det = dot(e1, pv);
  // Two-sided: a mesh's winding cannot be relied on, and a parity count does not need it.
  if (fabs(det) < kTolerance<real_t>()) { return false; }
  const real_t inv = real_t(1) / det;
  const Vec3<real_t> tv = o - v0;
  const real_t u = dot(tv, pv) * inv;
  if (u < real_t(0) || u > real_t(1)) { return false; }
  const Vec3<real_t> qv = cross(tv, e1);
  const real_t vv = dot(d, qv) * inv;
  if (vv < real_t(0) || u + vv > real_t(1)) { return false; }
  t = dot(e2, qv) * inv;
  // Distance to the nearest edge, in barycentric coordinates: small means the hit is on a
  // shared edge, where a parity count is unreliable.
  const real_t w = real_t(1) - u - vv;
  edge_dist = fmin(u, fmin(vv, w));
  return true;
}

/// Slab test against a node's bounding box. True if the ray meets the box before @p t_max.
template <typename real_t>
__host__ __device__ inline bool bvh_box_hit(const real_t* node, const Vec3<real_t>& o,
                                            const Vec3<real_t>& inv_d, real_t t_max) {
  real_t t0 = real_t(0), t1 = t_max;
  const real_t oo[3] = {o.x, o.y, o.z};
  const real_t ii[3] = {inv_d.x, inv_d.y, inv_d.z};
  for (int k = 0; k < 3; ++k) {
    real_t a = (node[k] - oo[k]) * ii[k];
    real_t b = (node[3 + k] - oo[k]) * ii[k];
    if (a > b) { const real_t s = a; a = b; b = s; }
    t0 = fmax(t0, a);
    t1 = fmin(t1, b);
    if (t0 > t1) { return false; }
  }
  return true;
}

/// Reciprocal of a direction, with zero components pushed to a large finite value so that the
/// slab test degenerates to "always inside this slab" rather than producing a NaN.
template <typename real_t>
__host__ __device__ inline Vec3<real_t> bvh_inv_dir(const Vec3<real_t>& d) {
  const real_t big = real_t(1) / kTolerance<real_t>();
  return Vec3<real_t>{(fabs(d.x) > kTolerance<real_t>()) ? real_t(1) / d.x : big,
                      (fabs(d.y) > kTolerance<real_t>()) ? real_t(1) / d.y : big,
                      (fabs(d.z) > kTolerance<real_t>()) ? real_t(1) / d.z : big};
}

/// The mesh's bounding box, as a box solid in the same frame. Used for an early rejection
/// before any triangle is touched, and for the extent queries.
template <typename real_t>
__host__ __device__ inline bool mesh_bbox_inside(const real_t* p, const Vec3<real_t>& q) {
  return fabs(q.x - p[3]) <= p[0] && fabs(q.y - p[4]) <= p[1] && fabs(q.z - p[5]) <= p[2];
}

/// Nearest triangle hit with t in (t_min, t_max), or kInfinity. @p edge_dist_out reports how
/// close that hit came to a triangle edge.
template <typename real_t>
__host__ __device__ inline real_t mesh_nearest_hit(const SolidStore<real_t>& st,
                                                   const Solid<real_t>& s,
                                                   const Vec3<real_t>& o, const Vec3<real_t>& d,
                                                   real_t t_min, real_t t_max,
                                                   real_t& edge_dist_out) {
  edge_dist_out = real_t(1);
  if (st.tri == nullptr || st.bvh == nullptr) { return kInfinity<real_t>(); }
  const Vec3<real_t> inv_d = bvh_inv_dir(d);

  int stack[kBvhMaxDepth];
  int sp = 0;
  stack[sp++] = s.a;  // this mesh's root node index
  real_t best = t_max;
  real_t best_edge = real_t(1);

  while (sp > 0) {
    const int ni = stack[--sp];
    const real_t* node = st.bvh + static_cast<long long>(ni) * kBvhStride;
    if (!bvh_box_hit(node, o, inv_d, best)) { continue; }
    const int first = static_cast<int>(node[6]);
    const int count = static_cast<int>(node[7]);
    if (count == 0) {
      if (sp + 2 <= kBvhMaxDepth) {
        stack[sp++] = first;
        stack[sp++] = first + 1;
      }
      continue;
    }
    for (int k = 0; k < count; ++k) {
      const real_t* v = st.tri + static_cast<long long>(first + k) * 9;
      real_t t = 0, ed = 0;
      if (!ray_triangle(v, o, d, t, ed)) { continue; }
      if (t <= t_min || t >= best) { continue; }
      best = t;
      best_edge = ed;
    }
  }
  edge_dist_out = best_edge;
  return (best < t_max) ? best : kInfinity<real_t>();
}

/// How close to a triangle edge a hit may land before the parity count is untrustworthy: two
/// adjacent triangles may both claim it, or neither.
template <typename real_t>
__host__ __device__ inline real_t kMeshEdgeTol() {
  return real_t(1e-7);
}

/// Counts triangle crossings strictly ahead of @p o. `reliable` comes back false when any hit
/// landed on or near a triangle edge, where the count cannot be trusted.
template <typename real_t>
__host__ __device__ inline int mesh_crossings(const SolidStore<real_t>& st,
                                              const Solid<real_t>& s, const Vec3<real_t>& o,
                                              const Vec3<real_t>& d, bool& reliable) {
  reliable = true;
  if (st.tri == nullptr || st.bvh == nullptr) { return 0; }
  const Vec3<real_t> inv_d = bvh_inv_dir(d);

  int stack[kBvhMaxDepth];
  int sp = 0;
  stack[sp++] = s.a;
  int count_hits = 0;

  while (sp > 0) {
    const int ni = stack[--sp];
    const real_t* node = st.bvh + static_cast<long long>(ni) * kBvhStride;
    if (!bvh_box_hit(node, o, inv_d, kInfinity<real_t>())) { continue; }
    const int first = static_cast<int>(node[6]);
    const int n = static_cast<int>(node[7]);
    if (n == 0) {
      if (sp + 2 <= kBvhMaxDepth) {
        stack[sp++] = first;
        stack[sp++] = first + 1;
      }
      continue;
    }
    for (int k = 0; k < n; ++k) {
      const real_t* v = st.tri + static_cast<long long>(first + k) * 9;
      real_t t = 0, ed = 0;
      if (!ray_triangle(v, o, d, t, ed)) { continue; }
      if (t <= kSurfTolerance<real_t>()) { continue; }  // behind or on the start point
      if (ed < kMeshEdgeTol<real_t>()) { reliable = false; }
      ++count_hits;
    }
  }
  return count_hits;
}

/// A direction derived from the point: deterministic, reproducible, and aligned with nothing.
///
/// The `attempt` index gives a different direction on a re-cast. Hashing the coordinates
/// rather than using a fixed axis is the whole mitigation for the edge problem: an axis-aligned
/// ray on a CAD model passes through shared edges constantly, and each such hit is a
/// coin-flip in the parity.
template <typename real_t>
__host__ __device__ inline Vec3<real_t> parity_direction(const Vec3<real_t>& p, int attempt) {
  // A cheap integer hash of the coordinates plus the attempt, mapped to a unit vector.
  //
  // The coordinates are scaled and truncated rather than bit-reinterpreted: reinterpreting
  // needs memcpy or __float_as_uint, and neither is available in a function that has to
  // compile for both host and device. Scaling by a large non-round factor spreads nearby
  // points to different hashes, which is all this needs - it is choosing a direction, not
  // hashing for a table.
  unsigned int h = 0x9E3779B9u ^ static_cast<unsigned int>(attempt * 0x85EBCA6Bu);
  const real_t coords[3] = {p.x, p.y, p.z};
  for (int k = 0; k < 3; ++k) {
    const long long scaled = static_cast<long long>(coords[k] * real_t(7919.17));
    const unsigned int bits = static_cast<unsigned int>(scaled & 0xFFFFFFFFll);
    h ^= bits + 0x9E3779B9u + (h << 6) + (h >> 2);
  }
  const real_t u = static_cast<real_t>((h >> 8) & 0xFFFF) / real_t(65535);
  const real_t v = static_cast<real_t>((h >> 24) & 0xFF) / real_t(255);
  const real_t ct = real_t(2) * u - real_t(1);
  const real_t stz = sqrt(fmax(real_t(0), real_t(1) - ct * ct));
  const real_t ph = units::twopi<real_t>() * v;
  return Vec3<real_t>{stz * cos(ph), stz * sin(ph), ct};
}

/// Containment by parity, with a re-cast when a hit lands on an edge.
template <typename real_t>
__host__ __device__ inline bool mesh_inside(const SolidStore<real_t>& st,
                                            const Solid<real_t>& s, const Vec3<real_t>& q) {
  // Outside the bounding box is outside the mesh, and that is most of the queries a
  // navigator makes. Worth the three comparisons.
  if (!mesh_bbox_inside(s.p, q)) { return false; }
  for (int attempt = 0; attempt < 4; ++attempt) {
    bool reliable = true;
    const Vec3<real_t> d = parity_direction(q, attempt);
    const int n = mesh_crossings(st, s, q, d, reliable);
    if (reliable) { return (n & 1) != 0; }
  }
  // Four directions all hit an edge: the point is almost certainly *on* the surface, where
  // either answer is defensible. Outside is the safer choice - it makes the navigator step to
  // the surface and re-decide rather than trapping the track inside.
  return false;
}

/// Distance along @p d to the surface, from a point either outside (@p want_inside) or inside.
///
/// The nearest triangle hit is almost always the answer. It is not when the ray grazes the
/// surface - touches a triangle and carries on the same side - so each candidate is checked
/// the way the generic engine checks its own: probe a little before and a little after, and
/// require the containment to differ. A graze fails that and the search moves past it.
template <typename real_t>
__host__ __device__ inline real_t mesh_dist(const SolidStore<real_t>& st,
                                            const Solid<real_t>& s, const Vec3<real_t>& q,
                                            const Vec3<real_t>& d, bool want_inside) {
  const real_t probe = kProbe<real_t>();
  real_t t_min = kSurfTolerance<real_t>();
  // Bounded so that a pathological mesh - a stack of coincident faces - terminates rather
  // than spinning. Eight grazes in a row along one ray is not a mesh anyone can transport.
  for (int attempt = 0; attempt < 8; ++attempt) {
    real_t edge = 0;
    const real_t t =
        mesh_nearest_hit(st, s, q, d, t_min, kInfinity<real_t>(), edge);
    if (t >= kInfinity<real_t>()) { return kInfinity<real_t>(); }
    const bool before = mesh_inside(st, s, q + (t - probe) * d);
    const bool after = mesh_inside(st, s, q + (t + probe) * d);
    if (before != after && after == want_inside) { return t; }
    t_min = t + probe;
  }
  return kInfinity<real_t>();
}


/// Squared distance from @p p to the triangle at @p v (nine reals).
///
/// The closest point on a triangle is in its interior, on one of the three edges, or at one of
/// the three vertices, and which one is decided by the sign pattern of six dot products. This
/// is Ericson's formulation (Real-Time Collision Detection, 5.1.5), written out because the
/// short version - project onto the plane and clamp the barycentrics - is wrong for a point
/// whose projection falls outside the triangle near an obtuse corner.
template <typename real_t>
__host__ __device__ inline real_t point_triangle_dist2(const real_t* v, const Vec3<real_t>& p) {
  const Vec3<real_t> a{v[0], v[1], v[2]};
  const Vec3<real_t> b{v[3], v[4], v[5]};
  const Vec3<real_t> c{v[6], v[7], v[8]};
  const Vec3<real_t> ab = b - a;
  const Vec3<real_t> ac = c - a;
  const Vec3<real_t> ap = p - a;
  const real_t d1 = dot(ab, ap);
  const real_t d2 = dot(ac, ap);
  auto d2_of = [&](const Vec3<real_t>& q) { return dot(p - q, p - q); };
  if (d1 <= real_t(0) && d2 <= real_t(0)) { return d2_of(a); }

  const Vec3<real_t> bp = p - b;
  const real_t d3 = dot(ab, bp);
  const real_t d4 = dot(ac, bp);
  if (d3 >= real_t(0) && d4 <= d3) { return d2_of(b); }

  const real_t vc = d1 * d4 - d3 * d2;
  if (vc <= real_t(0) && d1 >= real_t(0) && d3 <= real_t(0)) {
    const real_t t = d1 / (d1 - d3);
    return d2_of(a + t * ab);
  }

  const Vec3<real_t> cp = p - c;
  const real_t d5 = dot(ab, cp);
  const real_t d6 = dot(ac, cp);
  if (d6 >= real_t(0) && d5 <= d6) { return d2_of(c); }

  const real_t vb = d5 * d2 - d1 * d6;
  if (vb <= real_t(0) && d2 >= real_t(0) && d6 <= real_t(0)) {
    const real_t t = d2 / (d2 - d6);
    return d2_of(a + t * ac);
  }

  const real_t va = d3 * d6 - d5 * d4;
  if (va <= real_t(0) && (d4 - d3) >= real_t(0) && (d5 - d6) >= real_t(0)) {
    const real_t t = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    return d2_of(b + t * (c - b));
  }

  const real_t denom = real_t(1) / (va + vb + vc);
  const real_t vv = vb * denom;
  const real_t ww = vc * denom;
  return d2_of(a + vv * ab + ww * ac);
}

/// Squared distance from a point to a node's bounding box; zero inside it.
template <typename real_t>
__host__ __device__ inline real_t bvh_box_dist2(const real_t* node, const Vec3<real_t>& p) {
  const real_t pp[3] = {p.x, p.y, p.z};
  real_t s = real_t(0);
  for (int k = 0; k < 3; ++k) {
    const real_t below = node[k] - pp[k];
    const real_t above = pp[k] - node[3 + k];
    const real_t gap = fmax(real_t(0), fmax(below, above));
    s += gap * gap;
  }
  return s;
}

/// Distance from @p q to the nearest triangle: the isotropic safety, exact.
///
/// This is what makes a mesh volume as cheap to transport as an analytic one. Without it the
/// safety has to be zero - a bounding box gives no lower bound on the distance to a surface
/// from a point inside it - and a zero safety switches off Urban MSC's step limitation, which
/// on B1's geometry costs about 40% more steps. The nearest-triangle query is a BVH walk with
/// the same shape as the ray query: descend, prune any node whose box is already farther than
/// the best triangle found.
template <typename real_t>
__host__ __device__ inline real_t mesh_safety(const SolidStore<real_t>& st,
                                              const Solid<real_t>& s, const Vec3<real_t>& q) {
  if (st.tri == nullptr || st.bvh == nullptr) { return real_t(0); }
  int stack[kBvhMaxDepth];
  int sp = 0;
  stack[sp++] = s.a;
  real_t best2 = kInfinity<real_t>();

  while (sp > 0) {
    const int ni = stack[--sp];
    const real_t* node = st.bvh + static_cast<long long>(ni) * kBvhStride;
    if (bvh_box_dist2(node, q) >= best2) { continue; }
    const int first = static_cast<int>(node[6]);
    const int n = static_cast<int>(node[7]);
    if (n == 0) {
      if (sp + 2 <= kBvhMaxDepth) {
        stack[sp++] = first;
        stack[sp++] = first + 1;
      }
      continue;
    }
    for (int k = 0; k < n; ++k) {
      const real_t* v = st.tri + static_cast<long long>(first + k) * 9;
      const real_t d2 = point_triangle_dist2(v, q);
      if (d2 < best2) { best2 = d2; }
    }
  }
  return (best2 < kInfinity<real_t>()) ? sqrt(best2) : real_t(0);
}

}  // namespace g4gpu::geom
