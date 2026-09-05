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
//   1. bounding spheres. `solid_half_extent` bounds each *coordinate*, so the radius is
//      sqrt(3) times it - see the note at that function, and tests/test_solids.cu, which
//      checks that bound over all thirty solids. Spheres that miss cannot overlap.
//   2. for the pairs that survive, sample points in the smaller volume's box and ask whether
//      any of them is inside both. Sampling can miss a very thin sliver of overlap, which is
//      why stage 1 is not the answer on its own and why this is a *warning path* rather than
//      something the physics depends on.


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
  g.voxels.material =
      scene.pool.voxel_cells.empty() ? nullptr : scene.pool.voxel_cells.data();
  g.voxels.count = static_cast<int>(scene.pool.voxel_cells.size());

  // sqrt(3) times the coordinate bound is a sound bounding radius; to_local is a rotation
  // about the placement, so it preserves length and a local sphere is a world sphere.
  std::vector<double> radius(static_cast<std::size_t>(n), 0.0);
  for (int i = 0; i < n; ++i) {
    radius[static_cast<std::size_t>(i)] =
        geom::solid_half_extent(store, scene.volumes[i].solid) * 1.7320509;
  }

  for (int i = 0; i < n; ++i) {
    // The world is layer 0 and contains everything by construction; an overlap with it is not
    // an overlap, it is containment.
    if (i == scene.world) { continue; }
    for (int j = i + 1; j < n; ++j) {
      if (j == scene.world) { continue; }
      if (scene.volumes[i].layer != scene.volumes[j].layer) { continue; }

      const auto d = scene.volumes[j].xform.trans - scene.volumes[i].xform.trans;
      const double sep = std::sqrt(dot(d, d));
      if (sep >= radius[static_cast<std::size_t>(i)] + radius[static_cast<std::size_t>(j)]) {
        continue;
      }

      // Sample the smaller of the two, so a small volume inside a large one is found rather
      // than diluted to nothing.
      const int small = (radius[static_cast<std::size_t>(i)]
                         <= radius[static_cast<std::size_t>(j)]) ? i : j;
      const int other = (small == i) ? j : i;
      const auto& vs = scene.volumes[small];
      const double half = geom::solid_half_extent(store, vs.solid);
      constexpr int kSamples = 4000;
      unsigned int state = 0x2545F491u + static_cast<unsigned int>(i * 7919 + j);
      auto u = [&state]() {
        state = state * 1664525u + 1013904223u;
        return static_cast<double>(state >> 8) / static_cast<double>(1u << 24);
      };
      int in_small = 0, in_both = 0;
      const G4double* r = vs.xform.rot;
      for (int k = 0; k < kSamples; ++k) {
        const double lx = (2 * u() - 1) * half;
        const double ly = (2 * u() - 1) * half;
        const double lz = (2 * u() - 1) * half;
        // Local to world: the stored matrix is world -> local, so its transpose inverts it.
        const g4gpu::Vec3<G4double> p{
            r[0] * lx + r[3] * ly + r[6] * lz + vs.xform.trans.x,
            r[1] * lx + r[4] * ly + r[7] * lz + vs.xform.trans.y,
            r[2] * lx + r[5] * ly + r[8] * lz + vs.xform.trans.z};
        if (!geom::inside_volume(g, small, p)) { continue; }
        ++in_small;
        if (geom::inside_volume(g, other, p)) { ++in_both; }
      }
      if (in_both == 0) { continue; }
      SameLayerOverlap o;
      o.a = i;
      o.b = j;
      o.layer = scene.volumes[i].layer;
      o.fraction = (in_small > 0) ? static_cast<double>(in_both) / in_small : 0.0;
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
    Log(Fmt("WARNING: %s and %s overlap on layer %d (%.0f%% of the smaller one)",
            scene.names[static_cast<std::size_t>(o.a)].c_str(),
            scene.names[static_cast<std::size_t>(o.b)].c_str(), o.layer, 100 * o.fraction));
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
      c.canvas.Text(r.x + 22, y,
                    Fmt("%-16s and %-16s   layer %d   %.0f%% shared",
                        scene.names[static_cast<std::size_t>(o.a)].c_str(),
                        scene.names[static_cast<std::size_t>(o.b)].c_str(), o.layer,
                        100 * o.fraction),
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
