// Detecting two volumes that overlap on the *same* layer.
//
// The layer model resolves an overlap by giving the space to the higher layer. Two volumes on
// the same layer have no such resolution: the tie-break is "the later one in the list wins",
// which is an implementation detail and not something a user chose. Whatever the transport
// then reports is arbitrary - the same detector with the volumes added in the other order
// gives a different answer, and nothing says so.
//
// "ON THE SAME LAYER" IS A QUESTION ABOUT A POINT, not about two volumes. Once a voxel class
// can carry a layer of its own, a phantom on layer 2 whose bone class is on 3 clashes with a
// box on 3 - in the bone cells, and nowhere else. This compared the two VOLUMES' layers and
// so saw nothing at all: 2 is not 3, and the pair was dismissed before any point was looked
// at. Reported exactly that way.
//
// So the pair test is now on the RANGE of layers each volume can have, which is one number
// for everything but such a grid, and every sample point is asked what layer each side has
// THERE. A phantom whose bone is on 3 and whose air is on 2, against a box on 3, is a
// candidate by range and a clash only in the bone.
//
// So it is refused. A run with an unresolved same-layer overlap does not produce a wrong
// number that needs interpreting; it produces a number whose meaning depends on the insertion
// order, which is worse.
//
// The test is in two stages because the exact one is expensive and the cheap one is
// conservative:
//
//   1. bounding boxes. `solid_half_extent` bounds each *coordinate* in the solid's own frame,
//      so the solid sits in a local cube of that half-width; rotating the cube and taking its
//      axis-aligned extent gives a world box that contains the solid. Boxes that miss cannot
//      overlap.
//   2. for the pairs that survive, sample the INTERSECTION of the two boxes and ask how many
//      samples are inside both solids AND on the same layer as each other there.
//
// Sampling where the overlap must be, rather than throughout one of the volumes, is what
// makes the test sensitive to a small overlap. Sampling the smaller volume's own bounding
// cube - which is what this did - dilutes by the ratio of the overlap to that whole cube, and
// the cube is sized by the solid's LARGEST half extent: a phantom 300 x 200 x 1700 mm sits in
// a cube of 3400 mm on a side and fills 2% of it, so 4000 samples put 80 points in the
// phantom and an overlap under about a percent of it went unseen. Reported as the check "only
// firing if there is substantial overlap".
//
// It is still sampling, and a sliver thin in all three axes can still hide - which is why
// this is a *warning path* and not something the physics depends on. The samples are a
// jittered lattice rather than independent points, because a lattice cannot leave half the
// box empty by chance, and a thin sliver is exactly the case where that matters.


/// The voxel class containing @p p in volume @p v, or -1 if it has none.
///
/// For the message. "The phantom overlaps the seat" is not something a user can act on when
/// the phantom has two hundred classes and one of them is the problem.
static int ClassAt(const geom::Geometry<G4double>& g, const g4::FlatScene& scene, int v,
                   const g4gpu::Vec3<G4double>& p) {
  (void)scene;
  const auto& vol = g.volumes[v];
  if (vol.solid.type != geom::SolidType::kVoxelGrid || g.voxels.cls == nullptr) { return -1; }
  const geom::VoxelGrid<G4double> grid = geom::voxel_grid_of(vol.solid);
  const g4gpu::Vec3<G4double> q = geom::to_local(vol.xform, p);
  int ijk[3];
  geom::voxel_cell_of(grid, q, ijk);
  for (int k = 0; k < 3; ++k) {
    if (ijk[k] < 0 || ijk[k] >= grid.n[k]) { return -1; }
  }
  const int cell = grid.index(ijk[0], ijk[1], ijk[2]);
  if (cell < 0 || cell >= g.voxels.count) { return -1; }
  const int c = static_cast<int>(g.voxels.cls[cell]);
  return (c >= 0 && c < grid.class_count) ? c : -1;
}

/// Finds same-layer overlaps in the flattened scene.
///
/// Works on the flattened scene rather than the model because that is what the transport
/// sees: the placements are resolved, the booleans are built, and `geom::inside` is the same
/// function the stepper calls. Checking the model would mean re-deriving all of that and
/// risking a different answer from the one that matters.
static std::vector<SameLayerOverlap> FindSameLayerOverlaps(const App& a) {
  std::vector<SameLayerOverlap> out;
  if (a.run_manager == nullptr) { return out; }
  const auto& scene = a.run_manager->GetScene();
  const auto store = scene.pool.store();
  const int n = static_cast<int>(scene.volumes.size());
  if (n < 2) { return out; }

  // The same Geometry the transport is given, so `inside_volume` here is the function the
  // stepper calls and cannot answer differently.
  geom::Geometry<G4double> g{};
  g.volumes = scene.volumes.data();
  g.n_volumes = n;
  g.world = scene.world;
  g.store = store;
  g.voxels = scene.pool.voxel_store();

  // The world-space bounding box of each volume: the solid's local cube of half-width
  // `solid_half_extent`, rotated into the world and bounded on each axis.
  //
  // `xform.rot` is world -> local, so local -> world is its transpose and the world extent
  // along axis k is the half-width times the sum of |.| down that column of the transpose -
  // which is exactly what makes this TIGHT for an unrotated solid, where it is the half
  // extent itself. The bounding sphere of the same cube would give sqrt(3) times as much on
  // every axis, and a box 3.5x too big on each side is 41x too much space to sample.
  std::vector<double> lo(static_cast<std::size_t>(n) * 3, 0.0);
  std::vector<double> hi(static_cast<std::size_t>(n) * 3, 0.0);
  for (int i = 0; i < n; ++i) {
    const auto& v = scene.volumes[i];
    const double h = geom::solid_half_extent(store, v.solid);
    const G4double* r = v.xform.rot;
    const double c[3] = {v.xform.trans.x, v.xform.trans.y, v.xform.trans.z};
    for (int k = 0; k < 3; ++k) {
      const double e = h * (std::fabs(r[k]) + std::fabs(r[k + 3]) + std::fabs(r[k + 6]));
      lo[static_cast<std::size_t>(i) * 3 + k] = c[k] - e;
      hi[static_cast<std::size_t>(i) * 3 + k] = c[k] + e;
    }
  }

  for (int i = 0; i < n; ++i) {
    // The world is layer 0 and contains everything by construction; an overlap with it is not
    // an overlap, it is containment.
    if (i == scene.world) { continue; }
    for (int j = i + 1; j < n; ++j) {
      if (j == scene.world) { continue; }
      // Could they be on the same layer ANYWHERE: do the two layer ranges meet. Identical to
      // comparing the two layers for every volume that has only one, which is all of them
      // until a voxel class is given one of its own.
      const int lo_i = geom::layer_lo_of(g, i), hi_i = geom::layer_hi_of(g, i);
      const int lo_j = geom::layer_lo_of(g, j), hi_j = geom::layer_hi_of(g, j);
      if (hi_i < lo_j || hi_j < lo_i) { continue; }

      // Where both boxes are. Empty on any axis and they cannot share a point.
      double blo[3], bhi[3];
      bool disjoint = false;
      for (int k = 0; k < 3 && !disjoint; ++k) {
        blo[k] = std::max(lo[static_cast<std::size_t>(i) * 3 + k],
                          lo[static_cast<std::size_t>(j) * 3 + k]);
        bhi[k] = std::min(hi[static_cast<std::size_t>(i) * 3 + k],
                          hi[static_cast<std::size_t>(j) * 3 + k]);
        disjoint = (bhi[k] <= blo[k]);
      }
      if (disjoint) { continue; }

      // A jittered 16^3 lattice over that box: 4096 samples, one per cell, placed at random
      // within its own cell. Same cost as 4096 independent points and it cannot happen to
      // leave a whole region unsampled.
      constexpr int kSide = 16;
      unsigned int state = 0x2545F491u + static_cast<unsigned int>(i * 7919 + j);
      auto u = [&state]() {
        state = state * 1664525u + 1013904223u;
        return static_cast<double>(state >> 8) / static_cast<double>(1u << 24);
      };
      int in_both = 0;
      int clash_layer = 0;
      int clash_cls_a = -1, clash_cls_b = -1;
      for (int ix = 0; ix < kSide; ++ix) {
        for (int iy = 0; iy < kSide; ++iy) {
          for (int iz = 0; iz < kSide; ++iz) {
            const double f[3] = {(ix + u()) / kSide, (iy + u()) / kSide, (iz + u()) / kSide};
            const g4gpu::Vec3<G4double> p{blo[0] + f[0] * (bhi[0] - blo[0]),
                                          blo[1] + f[1] * (bhi[1] - blo[1]),
                                          blo[2] + f[2] * (bhi[2] - blo[2])};
            if (!geom::inside_volume(g, i, p) || !geom::inside_volume(g, j, p)) { continue; }
            // Sharing a point is not the problem; sharing a point AT THE SAME LAYER is. A
            // box over a phantom's air on 2 while the box is on 3 is exactly what the layer
            // model is for, and refusing it would refuse the feature.
            const int li = geom::volume_layer_at(g, i, p);
            const int lj = geom::volume_layer_at(g, j, p);
            if (li != lj) { continue; }
            if (in_both == 0) {
              clash_layer = li;
              clash_cls_a = ClassAt(g, scene, i, p);
              clash_cls_b = ClassAt(g, scene, j, p);
            }
            ++in_both;
          }
        }
      }
      if (in_both == 0) { continue; }

      const double box_mm3 = (bhi[0] - blo[0]) * (bhi[1] - blo[1]) * (bhi[2] - blo[2]);
      constexpr int kSamples = kSide * kSide * kSide;
      SameLayerOverlap o;
      o.a = i;
      o.b = j;
      o.layer = clash_layer;
      o.class_a = clash_cls_a;
      o.class_b = clash_cls_b;
      o.mm3 = box_mm3 * in_both / kSamples;
      // As a share of the smaller solid - but only when BOTH have a closed form to divide by,
      // because otherwise there is no telling which of them is smaller and "% of the smaller
      // one" would be a label for a number that is not that. Most pairs qualify: a box, a
      // grid, a cylinder, a sphere and a closed mesh all have one, the mesh's computed exactly
      // when it was read. A boolean does not, and sampling one on every edit to put a
      // percentage in a warning is not a trade worth making - those pairs report mm3 alone.
      const double va = geom::analytic_volume(scene.volumes[i].solid);
      const double vb = geom::analytic_volume(scene.volumes[j].solid);
      const double vmin = (va > 0 && vb > 0) ? std::min(va, vb) : -1.0;
      o.fraction = (vmin > 0) ? o.mm3 / vmin : -1.0;
      out.push_back(o);
    }
  }
  return out;
}

/// The model solid a flattened volume came from, BY NAME, or -1.
///
/// Not by index. The index was assumed to match "for a model built here" - it is written down
/// that way in the pop-up below - and it does not: G4Flatten emits one volume per PLACEMENT,
/// and a boolean's two operands are model solids that are never placed. Put a subtraction in
/// the scene and every model solid after it sits at a different index from its volume.
///
/// What that was costing: the pop-up's "raise the second one's layer" button raised whatever
/// model solid happened to sit at the flattened index - a different solid, silently, and the
/// clash it was meant to fix stayed exactly where it was. Found because a class label came
/// out as "[class 1]" when the class has the label "odd": the lookup was reading a solid with
/// no classes at all.
///
/// Names are unique - InsertShape makes them so, and the flattener names each placement after
/// its solid - so a name is the back-reference the scene does not carry.
static int ModelIndexOf(const App& a, int vol) {
  if (a.run_manager == nullptr) { return -1; }
  const auto& scene = a.run_manager->GetScene();
  if (vol < 0 || vol >= static_cast<int>(scene.names.size())) { return -1; }
  const std::string& want = scene.names[static_cast<std::size_t>(vol)];
  for (std::size_t i = 0; i < a.model.solids.size(); ++i) {
    if (a.model.solids[i].name == want) { return static_cast<int>(i); }
  }
  return -1;
}

/// One side of a clash, named: the volume, and the class if it was one of a phantom's.
///
/// The class's LABEL where there is one, because that is what the user gave it - "bone" says
/// what to do about it and "class 37" does not.
static std::string OverlapSideName(const App& a, int vol, int cls) {
  const auto& scene = a.run_manager->GetScene();
  std::string name = (vol >= 0 && vol < static_cast<int>(scene.names.size()))
                         ? scene.names[static_cast<std::size_t>(vol)]
                         : std::string("?");
  if (cls < 0) { return name; }
  std::string label;
  const int mi = ModelIndexOf(a, vol);
  if (mi >= 0) {
    const Solid& s = a.model.solids[static_cast<std::size_t>(mi)];
    if (cls < static_cast<int>(s.voxel_classes.size())) {
      const VoxelClass& vc = s.voxel_classes[static_cast<std::size_t>(cls)];
      label = vc.label.empty() ? Fmt("%g", vc.value) : vc.label;
    }
  }
  return name + (label.empty() ? Fmt("[class %d]", cls) : ("[" + label + "]"));
}

/// Recomputes the same-layer overlap list. Called after a scene rebuild, not per frame: it
/// samples thousands of points per candidate pair, and the answer only changes when the
/// geometry does.
///
/// Logged only when the set *changes*. A rebuild happens on every edit - moving a solid a
/// millimetre is a rebuild - and re-reporting the same overlap each time is how a message
/// stops being read. Which is the same mistake the overlapped-mass note made.
static void RefreshOverlaps(App& a) {
  const std::vector<SameLayerOverlap> before = a.overlaps;
  a.overlaps = FindSameLayerOverlaps(a);

  bool same = (before.size() == a.overlaps.size());
  for (std::size_t i = 0; same && i < before.size(); ++i) {
    same = (before[i].a == a.overlaps[i].a) && (before[i].b == a.overlaps[i].b)
           && (before[i].layer == a.overlaps[i].layer);
  }
  if (same) { return; }

  if (a.overlaps.empty()) {
    if (!before.empty()) { Log("same-layer overlaps resolved"); }
    return;
  }
  const auto& scene = a.run_manager->GetScene();
  for (const SameLayerOverlap& o : a.overlaps) {
    const std::string na = OverlapSideName(a, o.a, o.class_a);
    const std::string nb = OverlapSideName(a, o.b, o.class_b);
    if (o.fraction >= 0) {
      Log(Fmt("WARNING: %s and %s overlap on layer %d (%.3g mm3, %.3g%% of the smaller one)",
              na.c_str(), nb.c_str(), o.layer, o.mm3, 100 * o.fraction));
    } else {
      Log(Fmt("WARNING: %s and %s overlap on layer %d (%.3g mm3)", na.c_str(), nb.c_str(),
              o.layer, o.mm3));
    }
  }
  Log("  Two volumes on the same layer have no resolution: which one owns the shared space");
  Log("  depends on the order they were added. Put one on a higher layer. A run is refused");
  Log("  until then - the dose would depend on the build order, which is not a number.");
}

/// The pop-up that refuses a run, listing what has to be fixed.
/// Asks before moving the world off layer 0. See SetSolidLayer for why.
static void DrawWorldLayerPopup(App& a) {
  ui::Context& c = a.uic;
  const int lh = a.font.glyph_h;
  const int row = lh + 6;
  const ui::Rect r = PopupFrame(a, 520, 9 * lh + 3 * row, "Move the world off layer 0?");
  int y = r.y;

  c.canvas.Text(r.x + 14, y, Fmt("The world would go from layer 0 to layer %d.",
                                 a.pending_world_layer),
                ui::theme::kWarn);
  y += lh + 8;
  c.canvas.Text(r.x + 14, y, "The world contains every other volume, and a higher layer takes",
                ui::theme::kTextDim);
  y += lh;
  c.canvas.Text(r.x + 14, y, "the space wherever two overlap. So a world above another",
                ui::theme::kTextDim);
  y += lh;
  c.canvas.Text(r.x + 14, y, "volume's layer outranks that volume EVERYWHERE, and the volume",
                ui::theme::kTextDim);
  y += lh;
  c.canvas.Text(r.x + 14, y, "stops being in the scene at all: no transport through it, no",
                ui::theme::kTextDim);
  y += lh;
  c.canvas.Text(r.x + 14, y, "dose in it, and nothing drawn where it is.", ui::theme::kTextDim);
  y += lh + 8;
  {
    int below = 0;
    for (std::size_t i = 0; i < a.model.solids.size(); ++i) {
      if (static_cast<int>(i) == a.model.world()) { continue; }
      const int L = a.model.solids[i].layer;
      if (L != kNullLayer && L < a.pending_world_layer) { ++below; }
    }
    // The count, because "some volumes would stop existing" is a warning and "eleven of your
    // twelve volumes would stop existing" is an answer. Zero is worth saying too: it is the
    // case where the change is harmless and the question is only a formality.
    c.canvas.Text(r.x + 14, y,
                  (below == 0)
                      ? std::string("Nothing in this scene is below that layer, so nothing "
                                    "would be lost.")
                      : Fmt("%d volume%s in this scene %s below that layer.", below,
                            below == 1 ? "" : "s", below == 1 ? "is" : "are"),
                  (below == 0) ? ui::theme::kTextDim : ui::theme::kAccent);
  }
  y += lh + 12;

  if (ui::Button(c, 880, {r.x + 14, y, 190, row + 2},
                 Fmt("Move it to layer %d", a.pending_world_layer))) {
    a.popup = Popup::kNone;
    SetSolidLayer(a, a.model.world(), a.pending_world_layer, true);
    Log(Fmt("the world is on layer %d", a.pending_world_layer));
  }
  if (ui::Button(c, 881, {r.x + r.w - 104, y, 90, row + 2}, "Leave it")) {
    a.popup = Popup::kNone;
  }
}

static void DrawOverlapPopup(App& a) {
  ui::Context& c = a.uic;
  const int lh = a.font.glyph_h;
  const int row = lh + 6;
  const int rows = static_cast<int>(a.overlaps.size());
  const ui::Rect r = PopupFrame(a, 560, (rows + 9) * lh + 3 * row,
                                "Overlapping volumes on the same layer");
  int y = r.y;

  c.canvas.Text(r.x + 14, y, "A run was refused. These pairs share a layer and overlap:",
                ui::theme::kWarn);
  y += lh + 6;
  if (a.run_manager != nullptr) {
    const auto& scene = a.run_manager->GetScene();
    for (const SameLayerOverlap& o : a.overlaps) {
      if (o.a < 0 || o.b < 0) { continue; }
      const std::string share = (o.fraction >= 0)
                                    ? Fmt("%.3g mm3, %.3g%% of the smaller", o.mm3,
                                          100 * o.fraction)
                                    : Fmt("%.3g mm3", o.mm3);
      c.canvas.TextFit(r.x + 22, y,
                       Fmt("%s and %s   layer %d   %s",
                           OverlapSideName(a, o.a, o.class_a).c_str(),
                           OverlapSideName(a, o.b, o.class_b).c_str(), o.layer,
                           share.c_str()),
                       ui::theme::kText, r.w - 36);
      y += lh;
    }
  }
  y += 8;
  c.canvas.Text(r.x + 14, y, "The layer model gives shared space to the higher layer. Two",
                ui::theme::kTextDim);
  y += lh;
  c.canvas.Text(r.x + 14, y, "volumes on the same layer have no such rule: the tie-break is",
                ui::theme::kTextDim);
  y += lh;
  c.canvas.Text(r.x + 14, y, "whichever was added later, so the dose would depend on the",
                ui::theme::kTextDim);
  y += lh;
  c.canvas.Text(r.x + 14, y, "order you built the detector in. That is not a number worth",
                ui::theme::kTextDim);
  y += lh;
  c.canvas.Text(r.x + 14, y, "reporting, which is why the run is refused rather than run.",
                ui::theme::kTextDim);
  y += lh + 8;
  c.canvas.Text(r.x + 14, y, "Fix it by raising one volume's layer, or by moving them apart.",
                ui::theme::kAccent);
  y += lh + 10;

  if (ui::Button(c, 870, {r.x + 14, y, 200, row + 2}, "Raise the second one's layer")) {
    if (!a.overlaps.empty() && a.run_manager != nullptr) {
      // By name, not by index. See ModelIndexOf: a boolean's operands are model solids that
      // are never placed, so the flattened index this list carries is not the model index and
      // this button was raising a different solid's layer.
      //
      // Through SetSolidLayer, so a voxel volume's classes follow it. Raising the volume and
      // leaving a class behind on the layer it was clashing on would have left the clash
      // exactly where it was, and the pop-up would have come straight back.
      const int b = ModelIndexOf(a, a.overlaps.front().b);
      if (b >= 0) {
        SetSolidLayer(a, b, a.model.solids[static_cast<std::size_t>(b)].layer + 1);
        Log(Fmt("%s is now on layer %d", a.model.solids[static_cast<std::size_t>(b)].name.c_str(),
                a.model.solids[static_cast<std::size_t>(b)].layer));
      }
    }
    a.popup = Popup::kNone;
  }
  if (ui::Button(c, 871, {r.x + r.w - 96, y, 82, row + 2}, "Close")) { a.popup = Popup::kNone; }
}
