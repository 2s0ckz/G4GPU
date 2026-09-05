// Voxel volumes: a regular grid of cells, each with its own material.
//
// Modelled on G4NestedPhantomParameterisation rather than on placing a volume per voxel. A
// 512^3 CT is one volume carrying 134 million cell indices, not 134 million volumes: the
// navigator's scan stays O(number of real volumes), and the material of a point is an array
// lookup rather than a search.
//
// The grid is entered and left like a box. Inside it, a step ends at the next *cell* boundary
// whenever the neighbouring cell has a different material - a homogeneous region is crossed in
// one step rather than one step per cell, which for a CT is the difference between a few steps
// and a few hundred. The traversal is a 3D DDA: the classic Amanatides-Woo formulation, one
// division per axis at entry and one comparison plus one addition per cell after that.
//
// Cell materials live in a separate device array indexed as x + nx*(y + ny*z), the same order
// a .raw file and a MATLAB array use, so an import is a memcpy rather than a transpose.
#pragma once
#include <cmath>
#include "core/vec3.cuh"
#include "geometry/solids.cuh"

namespace g4gpu::geom {

/// The per-cell data for every voxel volume in a scene, in one pool.
template <typename real_t>
struct VoxelStore {
  /// Material index per cell, concatenated over volumes; a solid's `a` field is its offset.
  const short* material = nullptr;
  int count = 0;
};

/// Grid geometry, unpacked from a solid's parameters.
///
/// kVoxelGrid packs: p[0..2] = half extent, p[3..5] = cell counts. The grid is axis-aligned in
/// the solid's own frame and centred on its origin, so a rotated or offset placement is
/// handled by the Volume transform exactly as for any other solid.
template <typename real_t>
struct VoxelGrid {
  real_t half[3];
  int n[3];
  real_t cell[3];      ///< cell size along each axis
  real_t inv_cell[3];
  int offset;          ///< where this volume's cells start in the store

  __host__ __device__ int index(int i, int j, int k) const {
    return offset + i + n[0] * (j + n[1] * k);
  }
};

template <typename real_t>
__host__ __device__ inline VoxelGrid<real_t> voxel_grid_of(const Solid<real_t>& s) {
  VoxelGrid<real_t> g{};
  for (int k = 0; k < 3; ++k) {
    g.half[k] = s.p[k];
    g.n[k] = static_cast<int>(s.p[3 + k] + real_t(0.5));
    if (g.n[k] < 1) { g.n[k] = 1; }
    g.cell[k] = real_t(2) * g.half[k] / static_cast<real_t>(g.n[k]);
    g.inv_cell[k] = (g.cell[k] > real_t(0)) ? real_t(1) / g.cell[k] : real_t(0);
  }
  g.offset = s.a;
  return g;
}

/// Cell containing @p q, clamped to the grid. Only meaningful for a point inside.
template <typename real_t>
__host__ __device__ inline void voxel_cell_of(const VoxelGrid<real_t>& g,
                                              const Vec3<real_t>& q, int* ijk) {
  const real_t p[3] = {q.x, q.y, q.z};
  for (int k = 0; k < 3; ++k) {
    int c = static_cast<int>((p[k] + g.half[k]) * g.inv_cell[k]);
    if (c < 0) { c = 0; }
    if (c >= g.n[k]) { c = g.n[k] - 1; }
    ijk[k] = c;
  }
}

/// Material of the cell containing @p q, or -1 outside the grid or with no store.
template <typename real_t>
__host__ __device__ inline int voxel_material_at(const VoxelStore<real_t>& vs,
                                                 const VoxelGrid<real_t>& g,
                                                 const Vec3<real_t>& q) {
  if (vs.material == nullptr) { return -1; }
  const real_t tol = kSurfTolerance<real_t>();
  if (fabs(q.x) > g.half[0] + tol || fabs(q.y) > g.half[1] + tol
      || fabs(q.z) > g.half[2] + tol) {
    return -1;
  }
  int ijk[3];
  voxel_cell_of(g, q, ijk);
  const int idx = g.index(ijk[0], ijk[1], ijk[2]);
  if (idx < 0 || idx >= vs.count) { return -1; }
  return vs.material[idx];
}

/// Distance to the next point where the *material* changes, walking cells along the ray.
///
/// Returns kInfinity if the material never changes before the ray leaves the grid; the caller
/// then uses the box exit distance, which it has anyway.
///
/// With @p every_cell set, the walk stops at the first cell boundary instead. That is what a
/// per-voxel scorer needs: a deposit has to belong to one cell, and a step that crossed forty
/// identical cells would have to be split between them after the fact - by length, which is
/// not how the energy was actually laid down. Stopping at every boundary makes the question
/// not arise, at the cost of one step per voxel, which is why it is only done for a volume
/// that is actually being scored per cell. Geant4 pays the same cost for the same reason:
/// a parameterised voxel geometry has a boundary at every cell and steps are limited by it.
///
/// @param q,d  in the grid's own frame
/// @param[out] out_material  material after the crossing, or -1 if the grid is left
template <typename real_t>
__host__ __device__ inline real_t voxel_step(const VoxelStore<real_t>& vs,
                                             const VoxelGrid<real_t>& g,
                                             const Vec3<real_t>& q, const Vec3<real_t>& d,
                                             int& out_material, bool every_cell = false) {
  out_material = -1;
  if (vs.material == nullptr) { return kInfinity<real_t>(); }

  int ijk[3];
  voxel_cell_of(g, q, ijk);
  const int here = g.index(ijk[0], ijk[1], ijk[2]);
  if (here < 0 || here >= vs.count) { return kInfinity<real_t>(); }
  short mat0 = vs.material[here];

  // Amanatides-Woo setup: for each axis, the parameter of the next cell boundary and the
  // parameter step between boundaries.
  const real_t p[3] = {q.x, q.y, q.z};
  const real_t dir[3] = {d.x, d.y, d.z};
  real_t t_next[3];
  real_t t_delta[3];
  int step[3];
  for (int k = 0; k < 3; ++k) {
    if (fabs(dir[k]) < kTolerance<real_t>()) {
      step[k] = 0;
      t_next[k] = kInfinity<real_t>();
      t_delta[k] = kInfinity<real_t>();
      continue;
    }
    step[k] = (dir[k] > real_t(0)) ? 1 : -1;
    // Boundary of the current cell in the direction of travel.
    const real_t edge = -g.half[k] + g.cell[k] * static_cast<real_t>(
                                          ijk[k] + ((step[k] > 0) ? 1 : 0));
    t_next[k] = (edge - p[k]) / dir[k];
    if (t_next[k] < real_t(0)) { t_next[k] = real_t(0); }
    t_delta[k] = g.cell[k] / fabs(dir[k]);
  }

  // A CT is mostly homogeneous, so this walks a long way through identical cells. The bound is
  // the diagonal cell count, which is what a ray crossing the whole grid touches; without it a
  // degenerate direction could loop.
  const int max_steps = 2 * (g.n[0] + g.n[1] + g.n[2]) + 8;
  for (int iter = 0; iter < max_steps; ++iter) {
    // Advance along whichever axis reaches its boundary first.
    int axis = 0;
    if (t_next[1] < t_next[axis]) { axis = 1; }
    if (t_next[2] < t_next[axis]) { axis = 2; }
    if (t_next[axis] >= kInfinity<real_t>()) { return kInfinity<real_t>(); }

    const real_t t = t_next[axis];
    ijk[axis] += step[axis];
    if (ijk[axis] < 0 || ijk[axis] >= g.n[axis]) {
      // Left the grid: the box exit governs, and the caller already knows that distance.
      return kInfinity<real_t>();
    }
    t_next[axis] += t_delta[axis];

    const int idx = g.index(ijk[0], ijk[1], ijk[2]);
    if (idx < 0 || idx >= vs.count) { return kInfinity<real_t>(); }
    const short mat = vs.material[idx];
    if (every_cell || mat != mat0) {
      // A boundary the track is already standing on is not a step. The setup clamps a
      // negative t_next to zero, which is right for "the boundary is behind you" - but with
      // every_cell the very next thing the walk does is return it, and a zero-length step
      // makes no progress: the track is re-proposed at the same point, burns through the step
      // budget and is killed. That zeroed every score in the scene, not only the voxel one,
      // because the tracks died before reaching anything.
      //
      // Only reachable with every_cell. Waiting for the material to change gave the walk
      // several cells to get past the boundary it started on, which is why the material-change
      // path never showed this.
      if (t > kTolerance<real_t>()) {
        out_material = mat;
        return t;
      }
      mat0 = mat;  // treat the zero-width cell as where we started, and keep walking
    }
  }
  return kInfinity<real_t>();
}

}  // namespace g4gpu::geom
