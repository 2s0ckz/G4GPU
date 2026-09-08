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

/// A volume's overlap priority, as one comparable number.
///
/// The layer rule is "highest layer wins, later definition breaks a tie", which is a compare
/// on a pair. Packing the pair makes it a compare on a scalar, and the point of that is not
/// brevity: with per-class layers a volume's priority is no longer constant over the volume,
/// so this gets computed at a POINT and carried around, and a scalar is what can be carried
/// in a register and compared in one instruction inside a cell march.
__host__ __device__ inline long long volume_rank(int layer, int index) {
  return (static_cast<long long>(layer) << 32) | static_cast<unsigned int>(index);
}

/// Sentinel for "this cell has no class, so the volume's own layer applies".
constexpr int kNoClassLayer = -2147483647;

/// The per-cell data for every voxel volume in a scene, in one pool.
template <typename real_t>
struct VoxelStore {
  /// Material index per cell, concatenated over volumes; a solid's `a` field is its offset.
  const short* material = nullptr;
  int count = 0;

  /// PER-CLASS LAYERS, which is how a phantom can win an overlap where it is bone and lose it
  /// where it is air.
  ///
  /// `cls` is the class index per cell - the same array the renderer colours by, at the same
  /// offsets - and `class_layer` is one layer per class, concatenated over volumes with a
  /// solid's p[6] naming where its run starts. Two arrays rather than a layer per cell because
  /// a cell is already paying two bytes for its class and a 512^3 phantom is 268 MB of them:
  /// the indirection is free and the alternative is not.
  ///
  /// Both null - the normal case - means every cell has the volume's own layer, and nothing
  /// below does any extra work. They are uploaded only when some class actually asks for a
  /// layer of its own, so a batch run with an ordinary phantom carries neither.
  const short* cls = nullptr;
  const int* class_layer = nullptr;
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
  int class_base;      ///< where this volume's per-class run starts (colours, layers)
  int class_count;     ///< how many classes long that run is; 0 for none

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
  // p[6] and p[7] are spare on a voxel grid, and carry the per-class run: where it starts and
  // how long it is. The run holds a colour per class for the renderer and, when the feature is
  // used, a layer per class for the navigator - the same indexing for both, because they are
  // the same list of classes.
  g.class_base = static_cast<int>(s.p[6]);
  g.class_count = static_cast<int>(s.p[7]);
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

/// The layer of cell @p cell, or kNoClassLayer when this scene has no per-class layers.
///
/// @p cell is a store index - what VoxelGrid::index returns - not a cell coordinate.
template <typename real_t>
__host__ __device__ inline int voxel_cell_layer(const VoxelStore<real_t>& vs,
                                                const VoxelGrid<real_t>& g, int cell) {
  if (vs.cls == nullptr || vs.class_layer == nullptr) { return kNoClassLayer; }
  if (cell < 0 || cell >= vs.count) { return kNoClassLayer; }
  const int c = static_cast<int>(vs.cls[cell]);
  if (c < 0 || c >= g.class_count) { return kNoClassLayer; }
  return vs.class_layer[g.class_base + c];
}

/// The layer that applies at @p q inside the grid: its cell's class layer, or @p fallback.
///
/// @p fallback is the volume's own layer, which is what a cell with no class gets - an
/// unclassified import, or a grid that is a detector rather than a segmentation.
template <typename real_t>
__host__ __device__ inline int voxel_layer_at(const VoxelStore<real_t>& vs,
                                              const VoxelGrid<real_t>& g,
                                              const Vec3<real_t>& q, int fallback) {
  if (vs.cls == nullptr || vs.class_layer == nullptr) { return fallback; }
  const real_t tol = kSurfTolerance<real_t>();
  if (fabs(q.x) > g.half[0] + tol || fabs(q.y) > g.half[1] + tol
      || fabs(q.z) > g.half[2] + tol) {
    return fallback;
  }
  int ijk[3];
  voxel_cell_of(g, q, ijk);
  const int lay = voxel_cell_layer(vs, g, g.index(ijk[0], ijk[1], ijk[2]));
  return (lay == kNoClassLayer) ? fallback : lay;
}

/// A cell-by-cell walk along a ray, as an iterator.
///
/// Amanatides-Woo: one division per axis at entry, then one comparison and one addition per
/// cell. Hands back every cell and THE AXIS IT WAS ENTERED BY. The axis is the face normal,
/// and a renderer without it composites a phantom into uniform fog: no face is lit differently
/// from any other, so there is no shape to see. Which is the whole point of drawing it.
///
/// voxel_step() below asks the transport's question - how far to the next material change -
/// and is written over this walk rather than repeating the traversal. For a while it was a
/// second hand-written copy of the DDA, which is the duplication this codebase has been bitten
/// by twice in a *constant* (docs/RISK.md V8, V9) with more surface area to get wrong. The
/// rewrite waited for a change whose evidence is the physics comparisons rather than a picture,
/// because voxel_step decides where tracks stop. See docs/RISK.md V15.
template <typename real_t>
struct VoxelWalk {
  int ijk[3];
  real_t t_next[3];
  real_t t_delta[3];
  int step[3];
  int axis;      ///< axis of the boundary just crossed; -1 before the first Next()
  real_t t;      ///< ray parameter of that crossing

  /// @param q,d in the grid's own frame. False if @p q is not inside the grid.
  __host__ __device__ bool Start(const VoxelGrid<real_t>& g, const Vec3<real_t>& q,
                                 const Vec3<real_t>& d) {
    voxel_cell_of(g, q, ijk);
    for (int k = 0; k < 3; ++k) {
      if (ijk[k] < 0 || ijk[k] >= g.n[k]) { return false; }
    }
    const real_t p[3] = {q.x, q.y, q.z};
    const real_t dir[3] = {d.x, d.y, d.z};
    for (int k = 0; k < 3; ++k) {
      if (fabs(dir[k]) < kTolerance<real_t>()) {
        step[k] = 0;
        t_next[k] = kInfinity<real_t>();
        t_delta[k] = kInfinity<real_t>();
        continue;
      }
      step[k] = (dir[k] > real_t(0)) ? 1 : -1;
      const real_t edge =
          -g.half[k] + g.cell[k] * static_cast<real_t>(ijk[k] + ((step[k] > 0) ? 1 : 0));
      t_next[k] = (edge - p[k]) / dir[k];
      if (t_next[k] < real_t(0)) { t_next[k] = real_t(0); }
      t_delta[k] = g.cell[k] / fabs(dir[k]);
    }
    axis = -1;
    t = real_t(0);
    return true;
  }

  /// Advances one cell. False when the ray leaves the grid.
  __host__ __device__ bool Next(const VoxelGrid<real_t>& g) {
    int k = 0;
    if (t_next[1] < t_next[k]) { k = 1; }
    if (t_next[2] < t_next[k]) { k = 2; }
    if (t_next[k] >= kInfinity<real_t>()) { return false; }
    t = t_next[k];
    ijk[k] += step[k];
    if (ijk[k] < 0 || ijk[k] >= g.n[k]) { return false; }
    t_next[k] += t_delta[k];
    axis = k;
    return true;
  }

  __host__ __device__ int Index(const VoxelGrid<real_t>& g) const {
    return g.index(ijk[0], ijk[1], ijk[2]);
  }
};

/// Distance to the next point where the material OR THE LAYER changes, walking cells along
/// the ray.
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

  VoxelWalk<real_t> walk;
  if (!walk.Start(g, q, d)) { return kInfinity<real_t>(); }
  const int here = walk.Index(g);
  if (here < 0 || here >= vs.count) { return kInfinity<real_t>(); }
  short mat0 = vs.material[here];
  // THE LAYER IS PART OF WHAT MUST NOT CHANGE INSIDE A STEP.
  //
  // A step ends where the material changes because the physics is different there. It has to
  // end where the *layer* changes for a different reason: the layer decides who owns the
  // point, so which volumes can end this step is decided by it, and step_to_boundary answers
  // that question once at the step's start. Two neighbouring classes sharing a material and
  // differing in layer - assign one material to every class, then raise bone's layer, which
  // is exactly what the GUI makes easy - would otherwise be crossed in one step, with a
  // higher-layer volume overlapping the far cell going unseen.
  //
  // kNoClassLayer when the scene has no per-class layers, and then this never changes and
  // costs one comparison per cell.
  int lay0 = voxel_cell_layer(vs, g, here);

  // A CT is mostly homogeneous, so this walks a long way through identical cells. The bound is
  // the diagonal cell count, which is what a ray crossing the whole grid touches; without it a
  // degenerate direction could loop.
  const int max_steps = 2 * (g.n[0] + g.n[1] + g.n[2]) + 8;
  for (int iter = 0; iter < max_steps; ++iter) {
    // Next() is false both for a direction with no boundary left to reach and for a step that
    // leaves the grid: either way the box exit governs, and the caller already knows that
    // distance.
    if (!walk.Next(g)) { return kInfinity<real_t>(); }

    const int idx = walk.Index(g);
    if (idx < 0 || idx >= vs.count) { return kInfinity<real_t>(); }
    const short mat = vs.material[idx];
    const int lay = voxel_cell_layer(vs, g, idx);
    if (every_cell || mat != mat0 || lay != lay0) {
      // A boundary the track is already standing on is not a step. Start() clamps a negative
      // t_next to zero, which is right for "the boundary is behind you" - but with every_cell
      // the very next thing the walk does is return it, and a zero-length step makes no
      // progress: the track is re-proposed at the same point, burns through the step budget
      // and is killed. That zeroed every score in the scene, not only the voxel one, because
      // the tracks died before reaching anything.
      //
      // Only reachable with every_cell. Waiting for the material to change gave the walk
      // several cells to get past the boundary it started on, which is why the material-change
      // path never showed this.
      //
      // The guard is voxel_step's and not the walk's: a zero-width cell is still a cell the
      // ray passes through, which is what the renderer wants, and this is only about what
      // counts as a step.
      if (walk.t > kTolerance<real_t>()) {
        out_material = mat;
        return walk.t;
      }
      mat0 = mat;  // treat the zero-width cell as where we started, and keep walking
      lay0 = lay;
    }
  }
  return kInfinity<real_t>();
}

/// Distance to the first cell of @p g whose class outranks @p rank, or kInfinity.
///
/// The question step_to_boundary asks of a grid whose classes are on different layers. For an
/// ordinary volume "where does it start outranking me" is "where does it start", and dist_in
/// answers it; for such a grid the answer is a cell boundary somewhere inside it, and only a
/// walk knows where.
///
/// @p q,d   in the grid's frame, with @p q INSIDE the grid - the caller finds the box entry
///          first, which it needs anyway
/// @p limit stop looking beyond this distance and report kInfinity. The caller passes the
///          best distance it already has, so the walk never looks further than the step was
///          going to go: a grid the ray only clips costs a few cells, not its whole diagonal.
template <typename real_t>
__host__ __device__ inline real_t voxel_first_outranking(const VoxelStore<real_t>& vs,
                                                         const VoxelGrid<real_t>& g,
                                                         const Vec3<real_t>& q,
                                                         const Vec3<real_t>& d, int vol_index,
                                                         int vol_layer, long long rank,
                                                         real_t limit) {
  VoxelWalk<real_t> walk;
  if (!walk.Start(g, q, d)) { return kInfinity<real_t>(); }
  auto rank_of = [&](int cell) {
    const int lay = voxel_cell_layer(vs, g, cell);
    return volume_rank((lay == kNoClassLayer) ? vol_layer : lay, vol_index);
  };
  // The cell the ray is standing in counts: a track inside the grid, in a cell whose class
  // loses, is exactly the case this exists for, and the winning region may begin at the very
  // next boundary or may already be here.
  if (rank_of(walk.Index(g)) > rank) { return real_t(0); }
  const int max_steps = 2 * (g.n[0] + g.n[1] + g.n[2]) + 8;
  for (int iter = 0; iter < max_steps; ++iter) {
    if (!walk.Next(g)) { return kInfinity<real_t>(); }
    if (walk.t > limit) { return kInfinity<real_t>(); }
    if (rank_of(walk.Index(g)) > rank) { return walk.t; }
  }
  return kInfinity<real_t>();
}

}  // namespace g4gpu::geom
