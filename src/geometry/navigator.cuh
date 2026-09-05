// Stateless navigator over a flat volume table. No G4NavigationHistory, no touchables:
// a track carries only its volume id, so a navigation state costs one int.
//
// Overlap is resolved by LAYER INDEX rather than by containment, which is the deliberate
// departure from Geant4 here. There are no mother and daughter volumes. Every volume names a
// layer; where two overlap, the higher layer owns the space. The world is layer 0, so any
// volume placed on a higher layer sits "inside" it without being declared its child, and two
// volumes may overlap freely as long as their layers differ. Ties on layer are broken by
// definition order, later wins.
//
// What this buys: a detector can be assembled by dropping solids into the world in any order,
// which is what the model-building GUI needs, and a solid can straddle what would have been a
// mother boundary. What it costs: locate() is a scan over all volumes rather than a descent,
// so this wants a spatial index once scenes get large.
//
// Everything outside the world volume is cut off at the world boundary - a track that leaves
// the world is killed, and a solid that pokes out of it is simply not there beyond the edge.
#pragma once
#include "geometry/solids.cuh"
#include "geometry/transform.cuh"
#include "geometry/voxels.cuh"

namespace g4gpu::geom {

constexpr int kOutsideWorld = -1;

/// One placed solid. `layer` is the overlap priority; the world is the only layer 0.
template <typename real_t>
struct Volume {
  Solid<real_t> solid;
  Transform<real_t> xform;
  int layer;
  int material;  ///< index into the material table
  /// Which scorer this volume feeds, or -1 for none. A list of scored volumes would have to
  /// be searched once per step; a field on the volume the track is already reading is free.
  int score_index = -1;
  /// Score each cell separately rather than the volume as a whole. Only meaningful for a
  /// kVoxelGrid, and it changes the *stepping*: the navigator then ends a step at every cell
  /// boundary, not only where the material changes, so a deposit belongs to exactly one cell.
  /// See geom::voxel_step and G4PSEnergyDeposit3D, which limits steps for the same reason.
  bool score_per_voxel = false;
};

template <typename real_t>
struct Geometry {
  const Volume<real_t>* volumes;
  int n_volumes;
  int world;  ///< index of the layer-0 volume; also the clipping boundary
  /// Shared pools for boolean children, their transforms, and plane / z-section data.
  SolidStore<real_t> store{};
  /// Per-cell materials for every voxel volume in the scene.
  VoxelStore<real_t> voxels{};
};

/// True if @p p is inside volume @p i, in that volume's own frame.
template <typename real_t>
__host__ __device__ inline bool inside_volume(const Geometry<real_t>& g, int i,
                                              const Vec3<real_t>& p) {
  const Volume<real_t>& v = g.volumes[i];
  return inside(g.store, v.solid, to_local(v.xform, p));
}

/// The volume owning @p p: the highest-layer volume containing it, later definition winning
/// ties. kOutsideWorld if the point is outside the world.
template <typename real_t>
__host__ __device__ inline int locate(const Geometry<real_t>& g, const Vec3<real_t>& p) {
  if (!inside_volume(g, g.world, p)) { return kOutsideWorld; }
  int best = g.world;
  int best_layer = g.volumes[g.world].layer;
  for (int i = 0; i < g.n_volumes; ++i) {
    if (i == g.world) { continue; }
    const int layer = g.volumes[i].layer;
    if (layer < best_layer) { continue; }  // cannot win, even on a tie-break
    if (!inside_volume(g, i, p)) { continue; }
    // layer > best_layer wins outright; layer == best_layer wins by being later.
    best = i;
    best_layer = layer;
  }
  return best;
}

/// True if entering volume @p i would take ownership away from volume @p cur.
template <typename real_t>
__host__ __device__ inline bool outranks(const Geometry<real_t>& g, int i, int cur) {
  const int li = g.volumes[i].layer;
  const int lc = g.volumes[cur].layer;
  return (li > lc) || (li == lc && i > cur);
}

/// Distance along @p dir from @p p to the next point where locate() would return something
/// other than @p vol, and the volume that takes over there.
///
/// Two things can end a step: leaving @p vol, or entering a volume that outranks it. A
/// lower-ranked volume overlapping @p vol is invisible from inside it, which is exactly the
/// property that makes the layer model work - no "subtract the daughter from the mother"
/// bookkeeping is needed anywhere.
///
/// @param[out] next_volume  volume entered, or kOutsideWorld when @p vol was left, in which
///                          case the caller resolves the new volume with resolve_after_step
template <typename real_t>
__host__ __device__ inline real_t step_to_boundary(const Geometry<real_t>& g, int vol,
                                                   const Vec3<real_t>& p, const Vec3<real_t>& dir,
                                                   int& next_volume) {
  const Volume<real_t>& v = g.volumes[vol];
  real_t best = dist_out(g.store, v.solid, to_local(v.xform, p), dir_to_local(v.xform, dir));
  next_volume = kOutsideWorld;

  for (int i = 0; i < g.n_volumes; ++i) {
    if (i == vol || !outranks(g, i, vol)) { continue; }
    const Volume<real_t>& u = g.volumes[i];
    const real_t t = dist_in(g.store, u.solid, to_local(u.xform, p), dir_to_local(u.xform, dir));
    if (t < best) {
      best = t;
      next_volume = i;
    }
  }

  // Inside a voxel grid, a step also ends where the *material* changes from one cell to the
  // next. Identical neighbouring cells are crossed in one step, so a homogeneous region of a
  // CT costs one step rather than one per voxel - the difference between a few steps and a
  // few hundred through soft tissue. The volume does not change, so next_volume stays as it
  // is and the caller simply re-reads the material at the new point.
  if (v.solid.type == SolidType::kVoxelGrid) {
    const VoxelGrid<real_t> grid = voxel_grid_of(v.solid);
    int changed_to = -1;
    const real_t t = voxel_step(g.voxels, grid, to_local(v.xform, p),
                                dir_to_local(v.xform, dir), changed_to, v.score_per_voxel);
    if (t < best) {
      best = t;
      next_volume = vol;  // still this volume, in a different cell
    }
  }
  return best;
}

/// The material at @p p inside volume @p vol.
///
/// For an ordinary volume this is the volume's own material index. For a voxel grid it is the
/// material of the cell containing the point, which is why the stepper asks this rather than
/// reading Volume::material - the whole point of a voxel volume is that one volume has many
/// materials.
template <typename real_t>
__host__ __device__ inline int material_at(const Geometry<real_t>& g, int vol,
                                           const Vec3<real_t>& p) {
  if (vol < 0 || vol >= g.n_volumes) { return -1; }
  const Volume<real_t>& v = g.volumes[vol];
  if (v.solid.type != SolidType::kVoxelGrid) { return v.material; }
  const VoxelGrid<real_t> grid = voxel_grid_of(v.solid);
  const int cell = voxel_material_at(g.voxels, grid, to_local(v.xform, p));
  // A cell with no material assigned falls back to the volume's own, so an incompletely
  // classified import still transports rather than dropping tracks into a void.
  return (cell >= 0) ? cell : v.material;
}

/// Push a point just across a boundary so the next locate() resolves unambiguously.
template <typename real_t> __host__ __device__ constexpr real_t kPushDistance() {
  // Must clear the boundary decisively at the type precision, yet stay negligible against
  // mm-scale volumes. Float has ~1.5e-5 mm of absolute precision at the 180 mm world edge,
  // so 1e-3 mm is ~66 epsilon - enough that a track cannot re-land on the surface it just
  // crossed (which showed up as non-terminating tracks bouncing between two volumes).
  return (sizeof(real_t) == 4) ? real_t(1e-3) : real_t(1e-7);  // mm
}

/// Resolves the volume after a boundary crossing. A volume that was *entered* is known
/// outright; a volume that was *left* drops the track into whatever the layer model says
/// owns the point beyond, which under overlapping layers need not be any kind of parent -
/// so that case is answered by a fresh locate() at the pushed-past point.
template <typename real_t>
__host__ __device__ inline int resolve_after_step(const Geometry<real_t>& g, int entered,
                                                  const Vec3<real_t>& pushed_pos) {
  if (entered != kOutsideWorld) { return entered; }
  return locate(g, pushed_pos);
}

}  // namespace g4gpu::geom
