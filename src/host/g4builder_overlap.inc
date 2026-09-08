// Detecting two volumes that overlap on the *same* layer.
//
// The layer model resolves an overlap by giving the space to the higher layer. Two volumes on
// the same layer have no such resolution: the tie-break is "the later one in the list wins",
// which is an implementation detail and not something a user chose. Whatever the transport
// then reports is arbitrary - the same detector with the volumes added in the other order
// gives a different answer, and nothing says so.
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
//      samples are inside both solids.
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
      if (scene.volumes[i].layer != scene.volumes[j].layer) { continue; }

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
      for (int ix = 0; ix < kSide; ++ix) {
        for (int iy = 0; iy < kSide; ++iy) {
          for (int iz = 0; iz < kSide; ++iz) {
            const double f[3] = {(ix + u()) / kSide, (iy + u()) / kSide, (iz + u()) / kSide};
            const g4gpu::Vec3<G4double> p{blo[0] + f[0] * (bhi[0] - blo[0]),
                                          blo[1] + f[1] * (bhi[1] - blo[1]),
                                          blo[2] + f[2] * (bhi[2] - blo[2])};
            if (geom::inside_volume(g, i, p) && geom::inside_volume(g, j, p)) { ++in_both; }
          }
        }
      }
      if (in_both == 0) { continue; }

      const double box_mm3 = (bhi[0] - blo[0]) * (bhi[1] - blo[1]) * (bhi[2] - blo[2]);
      constexpr int kSamples = kSide * kSide * kSide;
      SameLayerOverlap o;
      o.a = i;
      o.b = j;
      o.layer = scene.volumes[i].layer;
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
    if (o.fraction >= 0) {
      Log(Fmt("WARNING: %s and %s overlap on layer %d (%.3g mm3, %.3g%% of the smaller one)",
              scene.names[static_cast<std::size_t>(o.a)].c_str(),
              scene.names[static_cast<std::size_t>(o.b)].c_str(), o.layer, o.mm3,
              100 * o.fraction));
    } else {
      Log(Fmt("WARNING: %s and %s overlap on layer %d (%.3g mm3)",
              scene.names[static_cast<std::size_t>(o.a)].c_str(),
              scene.names[static_cast<std::size_t>(o.b)].c_str(), o.layer, o.mm3));
    }
  }
  Log("  Two volumes on the same layer have no resolution: which one owns the shared space");
  Log("  depends on the order they were added. Put one on a higher layer. A run is refused");
  Log("  until then - the dose would depend on the build order, which is not a number.");
}

/// The pop-up that refuses a run, listing what has to be fixed.
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
      c.canvas.Text(r.x + 22, y,
                    Fmt("%-16s and %-16s   layer %d   %s",
                        scene.names[static_cast<std::size_t>(o.a)].c_str(),
                        scene.names[static_cast<std::size_t>(o.b)].c_str(), o.layer,
                        share.c_str()),
                    ui::theme::kText);
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
      // The flattened scene's index is the placement's, and the model's solids are in the
      // same order for a model built here - which is why this is offered as a shortcut and
      // the panel is still where a layer is normally set.
      const int b = a.overlaps.front().b;
      if (b >= 0 && b < static_cast<int>(a.model.solids.size())) {
        ++a.model.solids[static_cast<std::size_t>(b)].layer;
        a.model.Touch();
        a.scene_dirty = true;
        Log(Fmt("%s is now on layer %d", a.model.solids[static_cast<std::size_t>(b)].name.c_str(),
                a.model.solids[static_cast<std::size_t>(b)].layer));
      }
    }
    a.popup = Popup::kNone;
  }
  if (ui::Button(c, 871, {r.x + r.w - 96, y, 82, row + 2}, "Close")) { a.popup = Popup::kNone; }
}
