// Host-side BVH construction for triangle meshes.
//
// Host only: it sorts and allocates, and the device never builds a tree. The layout it writes
// is the one geometry/mesh.cuh traverses - eight reals per node, six of bounding box then
// `first` and `count`, with count == 0 marking an interior node whose children are at `first`
// and `first + 1`.
//
// Median split on the axis with the widest spread of triangle centroids. Not a surface-area
// heuristic: an SAH tree is perhaps 20% faster to traverse and considerably more code, and the
// thing this has to beat is testing every triangle. For a 100,000-triangle import that is the
// difference between about 17 box tests and 100,000 triangle tests per query.
//
// The two children of a node are allocated together, before either is filled, because the
// device layout stores one index for both. Filling one and then allocating the other would put
// the first child's whole subtree between them.
#pragma once
#include <algorithm>
#include <cmath>
#include <vector>
#include "geometry/solids.cuh"

namespace g4gpu::geom {

/// Builds a BVH over @p n triangles.
///
/// @param verts     input triangles, 9 reals each (three vertices), @p n of them
/// @param tri_pool  the scene's triangle pool; the triangles are appended to it in the order
///                  the tree wants them, so a leaf is a contiguous run
/// @param bvh_pool  the scene's node pool; nodes are appended
/// @param bbox_min  filled with the mesh's bounding-box minimum
/// @param bbox_max  filled with the mesh's bounding-box maximum
/// @return the index of the root node in @p bvh_pool
template <typename real_t>
int build_bvh(const real_t* verts, int n, std::vector<real_t>& tri_pool,
              std::vector<real_t>& bvh_pool, real_t bbox_min[3], real_t bbox_max[3]) {
  const long long tri_base = static_cast<long long>(tri_pool.size()) / 9;

  std::vector<int> order(static_cast<std::size_t>(std::max(n, 0)));
  for (int i = 0; i < n; ++i) { order[static_cast<std::size_t>(i)] = i; }

  std::vector<real_t> centroid(static_cast<std::size_t>(std::max(n, 0)) * 3, real_t(0));
  for (int i = 0; i < n; ++i) {
    for (int k = 0; k < 3; ++k) {
      centroid[static_cast<std::size_t>(i) * 3 + k] =
          (verts[i * 9 + k] + verts[i * 9 + 3 + k] + verts[i * 9 + 6 + k]) / real_t(3);
    }
  }

  for (int k = 0; k < 3; ++k) {
    bbox_min[k] = real_t(0);
    bbox_max[k] = real_t(0);
  }

  auto alloc = [&bvh_pool]() {
    const int idx = static_cast<int>(bvh_pool.size() / kBvhStride);
    bvh_pool.resize(bvh_pool.size() + kBvhStride, real_t(0));
    return idx;
  };

  // Recursion by explicit work list rather than a recursive lambda: the depth is bounded by
  // the tree's, and a work list makes that bound visible instead of putting it on the stack.
  struct Job {
    int node;
    int begin;
    int count;
  };
  std::vector<Job> work;

  if (n <= 0) {
    const int root = alloc();
    // An empty mesh: a node whose box is a point and which owns no triangles. Every query
    // against it misses, which is the right answer for a mesh with no facets.
    for (int k = 0; k < 6; ++k) { bvh_pool[static_cast<std::size_t>(root) * kBvhStride + k] = 0; }
    bvh_pool[static_cast<std::size_t>(root) * kBvhStride + 6] = real_t(tri_base);
    bvh_pool[static_cast<std::size_t>(root) * kBvhStride + 7] = real_t(0);
    return root;
  }

  const int root = alloc();
  work.push_back(Job{root, 0, n});

  while (!work.empty()) {
    const Job job = work.back();
    work.pop_back();

    real_t lo[3] = {verts[order[static_cast<std::size_t>(job.begin)] * 9 + 0],
                    verts[order[static_cast<std::size_t>(job.begin)] * 9 + 1],
                    verts[order[static_cast<std::size_t>(job.begin)] * 9 + 2]};
    real_t hi[3] = {lo[0], lo[1], lo[2]};
    real_t clo[3] = {centroid[static_cast<std::size_t>(order[static_cast<std::size_t>(job.begin)]) * 3 + 0],
                     centroid[static_cast<std::size_t>(order[static_cast<std::size_t>(job.begin)]) * 3 + 1],
                     centroid[static_cast<std::size_t>(order[static_cast<std::size_t>(job.begin)]) * 3 + 2]};
    real_t chi[3] = {clo[0], clo[1], clo[2]};
    for (int i = job.begin; i < job.begin + job.count; ++i) {
      const int t = order[static_cast<std::size_t>(i)];
      for (int v = 0; v < 3; ++v) {
        for (int k = 0; k < 3; ++k) {
          const real_t c = verts[t * 9 + v * 3 + k];
          lo[k] = std::min(lo[k], c);
          hi[k] = std::max(hi[k], c);
        }
      }
      for (int k = 0; k < 3; ++k) {
        const real_t c = centroid[static_cast<std::size_t>(t) * 3 + k];
        clo[k] = std::min(clo[k], c);
        chi[k] = std::max(chi[k], c);
      }
    }
    const std::size_t base = static_cast<std::size_t>(job.node) * kBvhStride;
    for (int k = 0; k < 3; ++k) {
      bvh_pool[base + k] = lo[k];
      bvh_pool[base + 3 + k] = hi[k];
    }
    if (job.node == root) {
      for (int k = 0; k < 3; ++k) {
        bbox_min[k] = lo[k];
        bbox_max[k] = hi[k];
      }
    }

    if (job.count <= kBvhLeafSize) {
      bvh_pool[base + 6] = static_cast<real_t>(tri_base + job.begin);
      bvh_pool[base + 7] = static_cast<real_t>(job.count);
      continue;
    }

    int axis = 0;
    real_t widest = chi[0] - clo[0];
    for (int k = 1; k < 3; ++k) {
      if (chi[k] - clo[k] > widest) {
        widest = chi[k] - clo[k];
        axis = k;
      }
    }
    const int mid = job.begin + job.count / 2;
    std::nth_element(order.begin() + job.begin, order.begin() + mid,
                     order.begin() + job.begin + job.count, [&](int a, int b) {
                       return centroid[static_cast<std::size_t>(a) * 3 + axis]
                              < centroid[static_cast<std::size_t>(b) * 3 + axis];
                     });

    const int l = alloc();
    const int r = alloc();
    // alloc() may have reallocated the pool, so index again rather than reusing a pointer.
    bvh_pool[static_cast<std::size_t>(job.node) * kBvhStride + 6] = static_cast<real_t>(l);
    bvh_pool[static_cast<std::size_t>(job.node) * kBvhStride + 7] = real_t(0);
    work.push_back(Job{l, job.begin, mid - job.begin});
    work.push_back(Job{r, mid, job.begin + job.count - mid});
  }

  // The triangles, in the order the leaves expect them.
  tri_pool.reserve(tri_pool.size() + static_cast<std::size_t>(n) * 9);
  for (int i = 0; i < n; ++i) {
    const int t = order[static_cast<std::size_t>(i)];
    for (int k = 0; k < 9; ++k) { tri_pool.push_back(verts[t * 9 + k]); }
  }
  return root;
}

/// Six times the SIGNED volume of a closed triangle mesh, by the divergence theorem: the signed
/// volume of the tetrahedron each triangle forms with the origin, summed. Exact and independent
/// of where the origin is, provided the mesh is closed.
///
/// The sign is the WINDING, and it is worth having on its own. A mesh wound outward gives a
/// positive volume and one wound inward a negative one, and the difference decides whether a
/// triangle's face normal points out of the solid or into it - which is how the renderer tells
/// a ray ENTERING a mesh from one LEAVING it without paying for a containment test. Both
/// windings occur in real files, so it cannot be assumed either way, and it used to be
/// discarded here with `|V|` and a note saying so.
template <typename real_t>
inline real_t mesh_signed_volume6(const real_t* verts, int n) {
  real_t v6 = real_t(0);
  for (int i = 0; i < n; ++i) {
    const real_t* t = verts + i * 9;
    const real_t x1 = t[0], y1 = t[1], z1 = t[2];
    const real_t x2 = t[3], y2 = t[4], z2 = t[5];
    const real_t x3 = t[6], y3 = t[7], z3 = t[8];
    v6 += x1 * (y2 * z3 - y3 * z2) - x2 * (y1 * z3 - y3 * z1) + x3 * (y1 * z2 - y2 * z1);
  }
  return v6;
}

/// The enclosed volume, which is a magnitude: |V|, because the winding may be either way.
template <typename real_t>
inline real_t mesh_volume(const real_t* verts, int n) {
  return std::fabs(mesh_signed_volume6(verts, n)) / real_t(6);
}

/// +1 if the mesh is wound so its face normals point OUT of the solid, -1 if they point in.
/// Zero only for a mesh that encloses nothing, where there is no answer and the caller should
/// not act on one.
template <typename real_t>
inline real_t mesh_winding(const real_t* verts, int n) {
  const real_t v6 = mesh_signed_volume6(verts, n);
  return (v6 > real_t(0)) ? real_t(1) : ((v6 < real_t(0)) ? real_t(-1) : real_t(0));
}

/// The total surface area, for reporting: a mesh whose area is far from its bounding box's is
/// usually a mesh with a unit problem.
template <typename real_t>
inline real_t mesh_area(const real_t* verts, int n) {
  real_t a = real_t(0);
  for (int i = 0; i < n; ++i) {
    const real_t* t = verts + i * 9;
    const real_t ex[3] = {t[3] - t[0], t[4] - t[1], t[5] - t[2]};
    const real_t ey[3] = {t[6] - t[0], t[7] - t[1], t[8] - t[2]};
    const real_t cx = ey[1] * ex[2] - ey[2] * ex[1];
    const real_t cy = ey[2] * ex[0] - ey[0] * ex[2];
    const real_t cz = ey[0] * ex[1] - ey[1] * ex[0];
    a += std::sqrt(cx * cx + cy * cy + cz * cz) / real_t(2);
  }
  return a;
}

}  // namespace g4gpu::geom
