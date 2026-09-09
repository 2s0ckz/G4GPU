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

  /// PER-CLASS LAYERS: whether this volume's layer varies from point to point, and over what
  /// range if it does. Only a voxel grid whose classes were given layers of their own can.
  ///
  /// The flag is what makes this safe to add. `layer` has no default and never has, so every
  /// site that builds a Volume sets it; two more ints with no default would have been
  /// whatever the stack held, and `layer_hi` reading as a small number silently PRUNES a
  /// volume that should have ended the step. Off by default means every existing caller keeps
  /// the behaviour it had, and the range is read only where the flag says it means something.
  ///
  /// What the range is for: `step_to_boundary` must decide, per volume, whether it can
  /// possibly end the step, and it cannot walk a phantom's cells to find out. `layer_hi`
  /// answers "could this outrank me anywhere", `layer_lo` answers "does it outrank me
  /// everywhere", and only a volume that is neither pays for a cell walk. G4Flatten fills
  /// them; `layer` itself is inside the range, because a cell with no class gets it.
  bool has_class_layers = false;
  int layer_lo = 0;
  int layer_hi = 0;
  /// Whether any of this volume's voxel classes is absent from the scene. See inside_volume,
  /// which is where it is honoured, and geom::kNullLayerTag for why absence is a fact about a
  /// class rather than a value in `layer_lo`.
  ///
  /// AT THE END OF THE STRUCT, like the three above it and for the same reason: test_navigation
  /// builds Volumes with positional brace-init, and a field inserted in the middle silently
  /// shifts every one of those initialisers by one.
  bool has_absent_classes = false;
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
///
/// A VOXEL CLASS SET TO NULL IS NOT PART OF THE VOLUME, and this is the one line that says so.
/// Not a layer that loses every overlap - not there: locate cannot return the volume for such
/// a point, so the space belongs to whatever else contains it, the transport steps through it
/// as that material, and nothing scores it. Rendering and the tally follow from the same fact
/// rather than from rules of their own.
///
/// Guarded by a flag, so a volume with no absent class - every volume until one is set - pays
/// one bool test on a hot path and nothing else.
template <typename real_t>
__host__ __device__ inline bool inside_volume(const Geometry<real_t>& g, int i,
                                              const Vec3<real_t>& p) {
  const Volume<real_t>& v = g.volumes[i];
  const Vec3<real_t> q = to_local(v.xform, p);
  if (!inside(g.store, v.solid, q)) { return false; }
  if (!v.has_absent_classes || v.solid.type != SolidType::kVoxelGrid) { return true; }
  return !voxel_absent_at(g.voxels, voxel_grid_of(v.solid), q);
}

/// True if volume @p i has a layer that varies from point to point - a voxel grid whose
/// classes were given layers of their own. Everything else has one layer and answers faster.
template <typename real_t>
__host__ __device__ inline bool layer_varies(const Geometry<real_t>& g, int i) {
  return g.volumes[i].has_class_layers;
}

/// The highest and lowest layer volume @p i can have anywhere. Its own layer unless its
/// classes carry layers of their own. See Volume::has_class_layers.
template <typename real_t>
__host__ __device__ inline int layer_hi_of(const Geometry<real_t>& g, int i) {
  const Volume<real_t>& v = g.volumes[i];
  return v.has_class_layers ? v.layer_hi : v.layer;
}
template <typename real_t>
__host__ __device__ inline int layer_lo_of(const Geometry<real_t>& g, int i) {
  const Volume<real_t>& v = g.volumes[i];
  return v.has_class_layers ? v.layer_lo : v.layer;
}

/// The layer volume @p i has AT @p p, in world coordinates.
///
/// For everything but a voxel grid with per-class layers this is the volume's own layer and
/// the point is not looked at.
template <typename real_t>
__host__ __device__ inline int volume_layer_at(const Geometry<real_t>& g, int i,
                                               const Vec3<real_t>& p) {
  const Volume<real_t>& v = g.volumes[i];
  if (!layer_varies(g, i) || v.solid.type != SolidType::kVoxelGrid) { return v.layer; }
  return voxel_layer_at(g.voxels, voxel_grid_of(v.solid), to_local(v.xform, p), v.layer);
}

/// Volume @p i's priority at @p p, as one comparable number. See geom::volume_rank.
template <typename real_t>
__host__ __device__ inline long long volume_rank_at(const Geometry<real_t>& g, int i,
                                                    const Vec3<real_t>& p) {
  return volume_rank(volume_layer_at(g, i, p), i);
}

/// The volume owning @p p: the highest-layer volume containing it, later definition winning
/// ties. kOutsideWorld if the point is outside the world.
template <typename real_t>
__host__ __device__ inline int locate(const Geometry<real_t>& g, const Vec3<real_t>& p) {
  if (!inside_volume(g, g.world, p)) { return kOutsideWorld; }
  int best = g.world;
  long long best_rank = volume_rank(g.volumes[g.world].layer, g.world);
  for (int i = 0; i < g.n_volumes; ++i) {
    if (i == g.world) { continue; }
    // The cheap rejection first, and it has to use the volume's HIGHEST possible layer: a
    // grid whose bone class outranks everything and whose air class outranks nothing cannot
    // be dismissed on one number. layer_hi == layer_lo for every ordinary volume, so this
    // costs the same comparison it always did.
    if (volume_rank(layer_hi_of(g, i), i) < best_rank) { continue; }
    if (!inside_volume(g, i, p)) { continue; }
    const long long rank = volume_rank_at(g, i, p);
    if (rank < best_rank) { continue; }
    best = i;
    best_rank = rank;
  }
  return best;
}

/// True if nothing that outranks @p own_rank contains @p p. The shared half of the two below.
///
/// @pre inside_volume(g, i, p)
template <typename real_t>
__host__ __device__ inline bool nothing_outranks_at(const Geometry<real_t>& g, int i,
                                                    const Vec3<real_t>& p, long long own) {
  // Outside the world is owned by nothing, which is locate's first answer as well. Without
  // this a volume poking out through the world face would draw the part that is not there.
  if (!inside_volume(g, g.world, p)) { return false; }
  for (int v = 0; v < g.n_volumes; ++v) {
    // could_outrank is the cheap rejection and it has to be conservative: it uses the
    // candidate's HIGHEST layer, so a grid whose bone class outranks everything and whose air
    // class outranks nothing cannot be dismissed on one number. For every ordinary volume
    // layer_hi == layer_lo and this is exact, which is why a mesh costs one comparison here.
    if (v == i || !could_outrank(g, v, own)) { continue; }
    if (!inside_volume(g, v, p)) { continue; }
    // AND THEN THE RANK AT THE POINT, which is the half that conservative rejection cannot
    // supply. Skipping it says a grid covers whatever its top class could cover: a box
    // coincident with a phantom lost its surface inside every air cell, because air is in the
    // same volume as bone. locate does this comparison too, after the same rejection.
    if (volume_rank_at(g, v, p) > own) { return false; }
  }
  return true;
}

/// True if volume @p i owns @p p, GIVEN THAT IT CONTAINS IT.
///
/// The same question as `locate(g, p) == i` restricted to a point already known to be inside
/// volume i - locate returns the highest-ranked volume containing the point, and this one
/// contains it, so locate returns something else exactly when something that outranks it
/// contains the point too.
///
/// It exists separately because locate also asks volume i whether it contains the point, and
/// for a mesh that is mesh_inside: a parity count, which has no best-so-far to reject a
/// subtree against and so visits every leaf along the ray. It is the expensive traversal that
/// the renderer's nearest-hit walk exists to avoid, and the renderer already knows the answer -
/// it just crossed the surface into the volume.
///
/// An OPAQUE volume hid the cost: the front-to-back walk stops at its first surface, so there
/// was one layer and one parity count. A TRANSLUCENT one does not stop, so a ray crossing
/// eight surfaces of a CAD import paid eight.
///
/// In a scene where nothing outranks the volume being asked about - one imported part in a
/// world, the common case - this does no containment tests at all.
///
/// @pre inside_volume(g, i, p)
template <typename real_t>
__host__ __device__ inline bool owns_contained_point(const Geometry<real_t>& g, int i,
                                                     const Vec3<real_t>& p) {
  // Per POINT, not per volume - on BOTH sides. A grid's class decides its rank, and the whole
  // point of per-class layers is that the answer differs cell to cell.
  return nothing_outranks_at(g, i, p, volume_rank_at(g, i, p));
}


/// True if entering volume @p i would take ownership away from volume @p cur.
///
/// Layers only, no point: for two ordinary volumes that is the whole answer, and for anything
/// with a per-point layer the caller has to ask at a point. The callers that must are
/// step_to_boundary and the renderer's cell march; this remains for the rest.
template <typename real_t>
__host__ __device__ inline bool outranks(const Geometry<real_t>& g, int i, int cur) {
  const int li = g.volumes[i].layer;
  const int lc = g.volumes[cur].layer;
  return (li > lc) || (li == lc && i > cur);
}

/// True if volume @p i outranks @p rank ANYWHERE, and hence might end a step.
template <typename real_t>
__host__ __device__ inline bool could_outrank(const Geometry<real_t>& g, int i,
                                              long long rank) {
  return volume_rank(layer_hi_of(g, i), i) > rank;
}

/// True if volume @p i outranks @p rank EVERYWHERE, so entering it is enough to end a step.
template <typename real_t>
__host__ __device__ inline bool always_outranks(const Geometry<real_t>& g, int i,
                                                long long rank) {
  return volume_rank(layer_lo_of(g, i), i) > rank;
}

/// Distance along @p dir from @p p to the next point where locate() would return something
/// other than @p vol, and the volume that takes over there.
///
/// Two things can end a step: leaving @p vol, or entering a volume that outranks it. A
/// lower-ranked volume overlapping @p vol is invisible from inside it, which is exactly the
/// property that makes the layer model work - no "subtract the daughter from the mother"
/// bookkeeping is needed anywhere.
///
/// WITH PER-CLASS LAYERS, "outranks it" IS A QUESTION ABOUT A POINT. @p vol's own rank is
/// taken at @p p, which is sound because a step inside a grid also ends wherever the class
/// layer changes - see voxel_step - so the rank cannot change part-way through one. And a
/// candidate grid may outrank @p vol in some of its cells and not others, in which case
/// entering its box is not the answer and a cell walk is; that walk is bounded by the best
/// distance already found, so a grid the ray merely clips costs a few cells.
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

  const long long cur_rank = volume_rank_at(g, vol, p);
  for (int i = 0; i < g.n_volumes; ++i) {
    if (i == vol || !could_outrank(g, i, cur_rank)) { continue; }
    const Volume<real_t>& u = g.volumes[i];
    const Vec3<real_t> ql = to_local(u.xform, p);
    const Vec3<real_t> dl = dir_to_local(u.xform, dir);
    if (layer_varies(g, i) && u.solid.type == SolidType::kVoxelGrid
        && !always_outranks(g, i, cur_rank)) {
      // Some of its classes win here and some do not, so the boundary that matters is a cell
      // boundary inside it rather than its own surface.
      const VoxelGrid<real_t> grid = voxel_grid_of(u.solid);
      real_t t0 = real_t(0);
      if (!inside(g.store, u.solid, ql)) {
        t0 = dist_in(g.store, u.solid, ql, dl);
        if (t0 >= best) { continue; }
      }
      const real_t tw = voxel_first_outranking(g.voxels, grid, ql + t0 * dl, dl, i, u.layer,
                                               cur_rank, best - t0);
      if (tw < kInfinity<real_t>() && t0 + tw < best) {
        best = t0 + tw;
        next_volume = i;
      }
      continue;
    }
    const real_t t = dist_in(g.store, u.solid, ql, dl);
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
