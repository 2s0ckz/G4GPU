// Turns the authored Geant4 object graph into the flat scene the GPU navigates.
//
// The mother/daughter tree is a *description*, not a runtime structure. Flattening walks it
// once and produces:
//   - one geom::Volume per placement, each with an absolute world transform and a layer index
//     one deeper than its mother's;
//   - one device material record per G4Material actually used;
//   - the shared solid, transform and aux pools that boolean solids index into.
//
// Repeated placement expands the way Geant4's touchables do: a logical volume placed twice,
// with daughters, yields two independent copies of the whole subtree. That is what makes
// "define once, place N times" behave as expected under a model that has no hierarchy at
// runtime.
//
// Anchors are resolved here too, before the tree walk, by folding an anchor's transform into
// the placements that name it.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <map>
#include <vector>
#include "data/materials.cuh"
#include "g4/G4PVPlacement.hh"
#include "g4/G4Solids.hh"   // G4VoxelGrid, for a grid's own cell materials
#include "g4/G4SDManager.hh"
#include "geometry/navigator.cuh"
#include "geometry/volume_of.cuh"

namespace g4gpu::g4 {

/// Everything a run needs, in device-ready form (still on the host).
struct FlatScene {
  std::vector<geom::Volume<G4double>> volumes;
  SolidPool pool;
  data::MaterialTable<G4double> materials;
  int world = 0;
  /// Volumes whose logical volume was marked sensitive; dose is summed over these.
  std::vector<int> scoring_volumes;
  /// The production cut as a *range*, mm - the number SetCutValue was given, not the
  /// per-material energy it converts to. G4WentzelVIModel's step limit reads it in these
  /// units, so it has to survive the flattening rather than be thrown away once the
  /// materials' thresholds are derived from it.
  double range_cut_mm = 0.7;
  /// Per-volume colour and visibility, parallel to `volumes`, for the renderer.
  struct Style {
    float r = 0.8f, g = 0.8f, b = 0.8f;
    /// 1 is opaque. Carried through the flattened scene because the renderer composites
    /// front to back and needs it per volume, and because G4VisAttributes has it too.
    float opacity = 1.0f;
    bool visible = true;
    bool wireframe = false;
  };
  std::vector<Style> styles;
  std::vector<G4String> names;
};

/// Composes two world->local transforms: first @p parent, then @p child expressed in the
/// parent's frame.
///
/// With local = R (p - t), applying the parent then the child gives
///     R = Rc Rp,   t = tp + Rp^T tc,
/// i.e. the child's offset has to be rotated into world coordinates by the parent's
/// orientation before it can be added. Adding tc directly is correct only when the parent is
/// unrotated, which is why a bug here hides until the first rotated mother.
inline geom::Transform<G4double> compose(const geom::Transform<G4double>& parent,
                                         const geom::Transform<G4double>& child) {
  geom::Transform<G4double> out{};
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      G4double sum = 0;
      for (int k = 0; k < 3; ++k) { sum += child.rot[3 * i + k] * parent.rot[3 * k + j]; }
      out.rot[3 * i + j] = sum;
    }
  }
  // Rp^T tc
  const Vec3<G4double>& tc = child.trans;
  const G4double* rp = parent.rot;
  const Vec3<G4double> rotated{rp[0] * tc.x + rp[3] * tc.y + rp[6] * tc.z,
                               rp[1] * tc.x + rp[4] * tc.y + rp[7] * tc.z,
                               rp[2] * tc.x + rp[5] * tc.y + rp[8] * tc.z};
  out.trans = parent.trans + rotated;
  out.identity = parent.identity && child.identity;
  return out;
}

/// The world->local transform a placement contributes, in its mother's frame.
inline geom::Transform<G4double> placement_transform(const G4PVPlacement* p) {
  geom::Transform<G4double> x{};
  const Vec3<G4double> t{p->GetTranslation().x(), p->GetTranslation().y(),
                         p->GetTranslation().z()};
  if (!p->HasRotation()) { return geom::make_translation<G4double>(t); }
  // Geant4 hands over the mother->daughter matrix already, which is exactly the world->local
  // map this representation stores. No transpose.
  const G4double* m = p->RotationData();
  for (int i = 0; i < 9; ++i) { x.rot[i] = m[i]; }
  x.trans = t;
  x.identity = false;
  return x;
}

/// Flattens every registered placement. @p range_cut_mm is the production cut used to derive
/// each material's secondary-production thresholds.
inline FlatScene flatten(G4double range_cut_mm = 0.7 /*mm*/) {
  FlatScene out;
  out.range_cut_mm = range_cut_mm;
  auto& placements = G4PVPlacement::Registry();
  G4SDManager::GetSDMpointer()->AssignIndices();

  // Materials first, so every volume can carry an index rather than a pointer.
  std::map<G4Material*, int> mat_index;
  auto build_material = [&](G4Material* mat) {
    if (mat == nullptr || mat_index.find(mat) != mat_index.end()) { return; }
    const int idx = mat->Build(out.materials, range_cut_mm);
    if (idx < 0) {
      std::printf("\nFATAL: material table full at \"%s\" (limit %d).\n",
                  mat->GetName().c_str(), data::kMaxMaterials);
      std::exit(2);
    }
    mat->device_index = idx;
    mat_index[mat] = idx;
  };
  for (G4PVPlacement* p : placements) {
    G4Material* mat = p->GetLogicalVolume()->GetMaterial();
    if (mat == nullptr) {
      std::printf("\nFATAL: logical volume \"%s\" has no material.\n",
                  p->GetLogicalVolume()->GetName().c_str());
      std::exit(2);
    }
    build_material(mat);
    // AND THE MATERIALS OF ITS CELLS, if it is a voxel grid that named them.
    //
    // A volume's own material is not the whole story for a voxel grid: the point of one is
    // that its cells have materials of their own, and for a segmented phantom those are
    // mostly materials no ordinary volume uses. Walking placements alone left them unbuilt,
    // with device_index -1, while the cells went to the device holding numbers that indexed a
    // table those materials were not in. See G4VoxelGrid::SetCellMaterials and RISK.md V20.
    if (const auto* grid = dynamic_cast<const G4VoxelGrid*>(p->GetLogicalVolume()->GetSolid())) {
      for (G4Material* cm : grid->CellMaterials()) { build_material(cm); }
    }
  }

  // The world is the placement with no mother and no explicit layer.
  G4PVPlacement* world = nullptr;
  for (G4PVPlacement* p : placements) {
    if (p->GetMotherLogical() == nullptr && p->ExplicitLayer() < 0) {
      if (world != nullptr) {
        std::printf("\nFATAL: two placements have no mother (\"%s\" and \"%s\").\n"
                    "  Exactly one is the world; give the other a mother, or an explicit\n"
                    "  layer with the G4PVPlacement(rot, tlate, logical, name, layer) form.\n",
                    world->GetName().c_str(), p->GetName().c_str());
        std::exit(2);
      }
      world = p;
    }
  }
  if (world == nullptr) {
    std::printf("\nFATAL: no world volume - every placement names a mother.\n");
    std::exit(2);
  }

  // Emit one volume, returning its index.
  auto emit = [&](G4PVPlacement* p, const geom::Transform<G4double>& x, int layer) {
    geom::Volume<G4double> v{};
    // A GRID HAS TO KNOW ITS OWN LAYER BEFORE IT IS BUILT, because Build writes one layer per
    // class and a class that named none gets the volume's. The solid does not otherwise know
    // where it was placed - this is the only point at which both facts are in hand.
    if (auto* grid = dynamic_cast<G4VoxelGrid*>(p->GetLogicalVolume()->GetSolid())) {
      grid->SetLayerHint(layer);
    }
    v.solid = out.pool.solids[p->GetLogicalVolume()->GetSolid()->Build(out.pool)];
    // Build appended a copy; the Volume carries it by value, so drop the pool entry's role as
    // the top-level solid but keep it - boolean children reference the pool by index and the
    // indices must stay stable.
    v.xform = x;
    v.layer = layer;
    // The range of layers this volume can have anywhere. For everything but a grid whose
    // classes named layers, that is one number and the flag stays off - see
    // geom::Volume::has_class_layers for why the flag matters more than the numbers.
    if (const auto* grid = dynamic_cast<const G4VoxelGrid*>(
            p->GetLogicalVolume()->GetSolid())) {
      const std::vector<int>& cl = grid->ClassLayers();
      if (!cl.empty()) {
        v.has_class_layers = true;
        v.layer_lo = layer;   // a cell with no class gets the volume's own layer
        v.layer_hi = layer;
        for (int L : cl) {
          if (L < v.layer_lo) { v.layer_lo = L; }
          if (L > v.layer_hi) { v.layer_hi = L; }
        }
      }
    }
    v.material = p->GetLogicalVolume()->GetMaterial()->device_index;
    const int idx = static_cast<int>(out.volumes.size());
    out.volumes.push_back(v);
    p->device_index = idx;

    FlatScene::Style st;
    if (const G4VisAttributes* va = p->GetLogicalVolume()->GetVisAttributes()) {
      st.r = static_cast<float>(va->GetColour().GetRed());
      st.g = static_cast<float>(va->GetColour().GetGreen());
      st.b = static_cast<float>(va->GetColour().GetBlue());
      // G4Colour carries alpha, and Geant4 means transparency by it. Now that the renderer
      // composites, it is used rather than dropped.
      st.opacity = static_cast<float>(va->GetColour().GetAlpha());
      st.visible = va->IsVisible();
      st.wireframe = va->IsForceWireframe();
    }
    out.styles.push_back(st);
    out.names.push_back(p->GetName());
    v.score_index = G4SDManager::GetSDMpointer()->SlotFor(p->GetLogicalVolume());
    out.volumes[idx].score_index = v.score_index;
    // Per-cell scoring, and only where it means something: a G4PSEnergyDeposit3D attached to
    // something that is not a voxel volume has no cells to key by, so it scores the volume as
    // a whole rather than silently doing nothing.
    v.score_per_voxel = v.score_index >= 0 && v.solid.type == geom::SolidType::kVoxelGrid
                        && G4SDManager::GetSDMpointer()->PerVoxelFor(p->GetLogicalVolume());
    out.volumes[idx].score_per_voxel = v.score_per_voxel;
    if (v.score_index >= 0) { out.scoring_volumes.push_back(idx); }
    return idx;
  };

  // Depth-first over the mother relation. Recursion depth is the depth of the detector
  // description, which is small; a cycle would be a construction error and is caught by the
  // depth guard rather than by hanging.
  std::function<void(G4LogicalVolume*, const geom::Transform<G4double>&, int, int)> descend =
      [&](G4LogicalVolume* mother, const geom::Transform<G4double>& parent_x, int layer,
          int depth) {
        if (depth > 32) {
          std::printf("\nFATAL: volume hierarchy deeper than 32, or a cycle in it.\n");
          std::exit(2);
        }
        for (G4PVPlacement* p : placements) {
          if (p->GetMotherLogical() != mother) { continue; }
          geom::Transform<G4double> x = placement_transform(p);
          if (G4PVPlacement* anchor = p->GetAnchor()) {
            x = compose(placement_transform(anchor), x);
          }
          const geom::Transform<G4double> world_x = compose(parent_x, x);
          emit(p, world_x, layer);
          descend(p->GetLogicalVolume(), world_x, layer + 1, depth + 1);
        }
      };

  const geom::Transform<G4double> world_x = placement_transform(world);
  out.world = emit(world, world_x, kWorldLayer);
  descend(world->GetLogicalVolume(), world_x, kWorldLayer + 1, 0);

  // Placements with an explicit layer sit directly in the world frame.
  for (G4PVPlacement* p : placements) {
    if (p->ExplicitLayer() < 0) { continue; }
    const geom::Transform<G4double> x = placement_transform(p);
    emit(p, x, p->ExplicitLayer());
    descend(p->GetLogicalVolume(), x, p->ExplicitLayer() + 1, 0);
  }

  return out;
}

}  // namespace g4gpu::g4
