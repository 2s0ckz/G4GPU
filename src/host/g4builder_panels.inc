// The model builder's panels, menus and actions.
//
// Included from g4builder.cu rather than compiled separately: everything here operates on the
// one App instance and the immediate-mode UI context, so there is no interface worth
// extracting, and splitting it into a translation unit would mean passing a dozen references
// through every function for no benefit. It is a separate file only because g4builder.cu is
// long enough already.

// ---------------------------------------------------------------- actions

/// Adds a primitive at the origin, with dimensions scaled to the world so it is visible.
static void InsertPrimitive(App& a, Shape shape) {
  Solid s;
  s.shape = shape;
  s.name = ShapeName(shape);
  // A unique name: the flattener and the picker both match on it.
  int suffix = 1;
  auto taken = [&](const std::string& n) {
    for (const Solid& e : a.model.solids) {
      if (e.name == n) { return true; }
    }
    return false;
  };
  while (taken(s.name)) { s.name = std::string(ShapeName(shape)) + std::to_string(++suffix); }

  const double world_half = a.model.solids.empty() ? 500.0 : a.model.solids[0].p[0];
  const double d = world_half * 0.15;
  for (double& v : s.p) { v = 0; }
  switch (shape) {
    case Shape::kBox: s.p[0] = s.p[1] = s.p[2] = d; break;
    case Shape::kOrb: s.p[0] = d; break;
    case Shape::kTubs: s.p[0] = 0; s.p[1] = d; s.p[2] = d; s.p[4] = 360; break;
    case Shape::kCons:
      s.p[0] = 0; s.p[1] = d; s.p[2] = 0; s.p[3] = d * 1.6; s.p[4] = d; s.p[6] = 360;
      break;
    case Shape::kSphere:
      s.p[0] = 0; s.p[1] = d; s.p[3] = 360; s.p[5] = 180;
      break;
    case Shape::kTorus: s.p[0] = 0; s.p[1] = d * 0.3; s.p[2] = d; s.p[4] = 360; break;
    case Shape::kTrd:
      s.p[0] = d; s.p[1] = d * 0.6; s.p[2] = d; s.p[3] = d * 0.6; s.p[4] = d;
      break;
    case Shape::kTrap:
      s.p[0] = d; s.p[3] = d; s.p[4] = d; s.p[5] = d * 1.2;
      s.p[7] = d * 1.2; s.p[8] = d * 0.9; s.p[9] = d * 1.1;
      break;
    case Shape::kPara: s.p[0] = s.p[1] = s.p[2] = d; s.p[3] = 15; s.p[4] = 10; break;
    case Shape::kEllipticalTube: s.p[0] = d; s.p[1] = d * 0.6; s.p[2] = d; break;
    case Shape::kEllipsoid:
      s.p[0] = d; s.p[1] = d * 0.7; s.p[2] = d * 1.3; s.p[3] = -d * 1.3; s.p[4] = d * 1.3;
      break;
    case Shape::kEllipticalCone: s.p[0] = 0.5; s.p[1] = 0.8; s.p[2] = d * 2; s.p[3] = d; break;
    case Shape::kParaboloid: s.p[0] = d; s.p[1] = d * 0.3; s.p[2] = d; break;
    case Shape::kHype:
      s.p[0] = d * 0.3; s.p[1] = d * 0.7; s.p[2] = 10; s.p[3] = 25; s.p[4] = d;
      break;
    case Shape::kTet:
      s.p[0] = 0; s.p[1] = 0; s.p[2] = d;
      s.p[3] = d; s.p[4] = 0; s.p[5] = -d * 0.5;
      s.p[6] = -d * 0.6; s.p[7] = d * 0.8; s.p[8] = -d * 0.5;
      s.p[9] = -d * 0.6; s.p[10] = -d * 0.8; s.p[11] = -d * 0.5;
      break;
    case Shape::kPolycone:
    case Shape::kPolyhedra:
      s.p[0] = 0; s.p[1] = 360; s.p[2] = 6;
      s.sections = {-d, 0, d * 0.5, 0, 0, d, d, 0, d * 0.4};
      break;
    default: s.p[0] = s.p[1] = s.p[2] = d; break;
  }
  // The world's material by default, so a new solid is placeable immediately rather than
  // being silently dropped for having none.
  s.material = a.model.solids.empty() ? -1 : a.model.solids[0].material;
  s.layer = 1;
  // A distinguishable color per insertion, cycling a small palette.
  static const float pal[6][3] = {{0.85f, 0.55f, 0.35f}, {0.45f, 0.70f, 0.85f},
                                  {0.60f, 0.80f, 0.50f}, {0.85f, 0.75f, 0.45f},
                                  {0.75f, 0.55f, 0.80f}, {0.80f, 0.45f, 0.45f}};
  const int k = static_cast<int>(a.model.solids.size()) % 6;
  s.r = pal[k][0];
  s.g = pal[k][1];
  s.b = pal[k][2];

  a.sel_solid = a.model.AddSolid(s);
  a.pfield_for = -2;
  a.scene_dirty = true;
  Log("inserted " + s.name);
}

/// Combines the two most recently selected solids. The selection is the second operand and the
/// one before it the first, which matches the order a user reads "subtract this from that".
/// Whether a solid may be a boolean operand at all.
///
/// A mesh may not: the boolean engine works from a bounded list of surface crossings and a
/// mesh can have more than fits, so `G4BooleanSolid::BuildWith` refuses one. A voxel grid may
/// not either - it becomes a voxel volume, not a solid.
///
/// This is checked here, when the operand is chosen, rather than only at Build time. It used
/// not to be, and the consequence was worse than a bad selftest: `InsertBoolean` walks
/// backwards to the newest free solid, so importing a CAD file and then subtracting a cylinder
/// silently made the *mesh* the first operand, and the run died at Build with a FATAL. A
/// refusal a user can only discover by losing the run is not a refusal.
static bool CanBeBooleanOperand(const Solid& s) {
  return s.shape != Shape::kImportedMesh && s.shape != Shape::kVoxelGrid;
}

static void InsertBoolean(App& a, Shape op) {
  if (a.sel_solid < 1 || a.model.solids.size() < 3) {
    a.status = "select the second of two solids first";
    Log("boolean needs two solids; select the second one and try again");
    return;
  }
  const int b = a.sel_solid;
  if (!CanBeBooleanOperand(a.model.solids[b])) {
    a.status = "a mesh or voxel volume cannot be a boolean operand";
    Log(Fmt("\"%s\" is a %s: it can be placed, but not unioned, subtracted or intersected.",
            a.model.solids[b].name.c_str(), ShapeName(a.model.solids[b].shape)));
    Log("  Do the boolean in the CAD tool and import the result, or use primitives.");
    return;
  }
  int first = -1;
  for (int i = static_cast<int>(a.model.solids.size()) - 1; i >= 1; --i) {
    if (i == b || !CanBeBooleanOperand(a.model.solids[i])) { continue; }
    // Skip solids already consumed as operands: nesting them would place the same solid twice.
    bool used = false;
    for (const Solid& s : a.model.solids) {
      if (s.operand_a == i || s.operand_b == i) { used = true; }
    }
    if (!used) { first = i; break; }
  }
  if (first < 0) {
    a.status = "no free second solid for the boolean";
    Log("no free solid to be the boolean's first operand (a mesh or voxel grid cannot be one)");
    return;
  }

  Solid s;
  s.shape = op;
  s.name = std::string(ShapeName(op)) + std::to_string(a.model.solids.size());
  s.operand_a = first;
  s.operand_b = b;
  s.pos[0] = a.model.solids[first].pos[0];
  s.pos[1] = a.model.solids[first].pos[1];
  s.pos[2] = a.model.solids[first].pos[2];
  s.material = a.model.solids[first].material;
  s.layer = a.model.solids[first].layer;
  s.r = a.model.solids[first].r;
  s.g = a.model.solids[first].g;
  s.b = a.model.solids[first].b;
  a.sel_solid = a.model.AddSolid(s);
  a.scene_dirty = true;
  Log(Fmt("%s of %s and %s", ShapeName(op), a.model.solids[first].name.c_str(),
          a.model.solids[b].name.c_str()));
}

static void AddScorer(App& a, ScoreQuantity q) {
  Scorer sc;
  sc.quantity = q;
  sc.name = ScoreName(q);
  for (char& c : sc.name) {
    if (c == ' ') { c = '_'; }
  }
  sc.name += std::to_string(a.model.scorers.size() + 1);
  if (a.sel_solid >= 0) { sc.solids.push_back(a.sel_solid); }
  a.sel_scorer = a.model.AddScorer(sc);
  a.scene_dirty = true;
  Log(Fmt("added scorer \"%s\" (%s)%s", sc.name.c_str(), ScoreName(q),
          sc.solids.empty() ? " - select a solid and use Scoring again to attach it" : ""));
}

/// A Windows file dialog. Returns an empty string if the user cancelled.
static std::string FileDialog(App& a, const char* filter, bool save,
                             const char* default_ext) {
  char buf[MAX_PATH] = {0};
  OPENFILENAMEA ofn{};
  ofn.lStructSize = sizeof(ofn);
  ofn.hwndOwner = a.hwnd;
  ofn.lpstrFilter = filter;
  ofn.lpstrFile = buf;
  ofn.nMaxFile = MAX_PATH;
  ofn.lpstrDefExt = default_ext;
  ofn.Flags = OFN_PATHMUSTEXIST | (save ? OFN_OVERWRITEPROMPT : OFN_FILEMUSTEXIST);
  const BOOL ok = save ? GetSaveFileNameA(&ofn) : GetOpenFileNameA(&ofn);
  return ok ? std::string(buf) : std::string();
}

/// Writes the per-voxel map from the last run, one file per scored voxel volume.
///
/// A plain text header and then one number per cell, x fastest - the same order the voxel
/// store uses and the same order the import read the file in, so a numpy `loadtxt` and a
/// `reshape(nz, ny, nx)` gets the array back the way it went in.
///
/// Written on save rather than on run because a run happens constantly while a detector is
/// being adjusted, and a tool that writes a file every time you press Run is a tool that
/// fills a directory.
static void WriteVoxelDose(App& a, const std::string& dir) {
  if (a.last_voxel_scores.empty() || a.run_manager == nullptr) { return; }
  const auto& scene = a.run_manager->GetScene();
  for (std::size_t v = 0; v < scene.volumes.size(); ++v) {
    const auto& vol = scene.volumes[v];
    if (!vol.score_per_voxel) { continue; }
    const auto grid = geom::voxel_grid_of(vol.solid);
    const int n = grid.n[0] * grid.n[1] * grid.n[2];
    if (n <= 0 || grid.offset < 0
        || grid.offset + n > static_cast<int>(a.last_voxel_scores.size())) {
      continue;
    }
    const std::string path = dir + "\\" + scene.names[v] + ".dose";
    FILE* f = std::fopen(path.c_str(), "wb");
    if (f == nullptr) {
      Log("could not write " + path);
      continue;
    }
    std::fprintf(f, "# energy deposit per voxel, MeV, from %d events\n", a.last_events);
    std::fprintf(f, "# volume %s\n", scene.names[v].c_str());
    std::fprintf(f, "# nx ny nz  %d %d %d\n", grid.n[0], grid.n[1], grid.n[2]);
    std::fprintf(f, "# cell mm   %g %g %g\n", static_cast<double>(grid.cell[0]),
                 static_cast<double>(grid.cell[1]), static_cast<double>(grid.cell[2]));
    std::fprintf(f, "# order     x fastest, then y, then z\n");
    for (int i = 0; i < n; ++i) {
      std::fprintf(f, "%.10g\n", a.last_voxel_scores[static_cast<std::size_t>(grid.offset + i)]);
    }
    std::fclose(f);
    Log(Fmt("  wrote %s (%d cells)", path.c_str(), n));
  }
}

static void SaveProjectTo(App& a, const std::string& dir) {
  if (WriteProject(a.model, dir)) {
    a.model.dirty = false;
    a.status = "saved " + dir;
    Log("wrote a compilable project to " + dir);
    // Two backslashes in the source, not one: a single one before b is a *backspace*, which
    // printed as "selftest_projectuild.bat" and read as a missing path separator.
    Log("  build it with " + dir + "\\build.bat, then run the .exe or its run1.mac");
    WriteVoxelDose(a, dir);
  } else {
    a.status = "save failed";
    Log("could not write the project to " + dir);
  }
}

/// Runs the model as it stands, capturing trajectories for the view.
/// Throws away the trajectories and the last run's numbers.
///
/// One function because "clear the tracks" and "the model changed, so the last run no longer
/// describes it" are the same operation. Leaving a dose on screen next to a geometry it was
/// not measured in is the most confusing thing a tool like this can do: the number looks
/// current, and there is nothing to say it is stale.
static void ResetRunState(App& a) {
  if (a.traj.count != nullptr) {
    CUDA_CHECK(cudaMemset(a.traj.count, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(a.traj.dropped, 0, sizeof(int)));
  }
  a.n_segments = 0;
  a.last_events = 0;
  a.last_ms = 0;
  a.last_scores.clear();
  a.last_voxel_scores.clear();
}

/// Saves the project, asking for a path the first time.
static void SaveProjectFromGui(App& a) {
  if (a.model_path.empty()) {
    const std::string p = FileDialog(a, "g4gpu model\0*.model\0", true, "model");
    if (p.empty()) { return; }
    a.model_path = p;
  }
  const std::size_t slash = a.model_path.find_last_of("/\\");
  SaveProjectTo(a, (slash == std::string::npos) ? "." : a.model_path.substr(0, slash));
}

/// Runs the model, splitting the events between every enabled source by weight.
///
/// A run with several sources is several sub-runs, one per source, with
/// `round(n * w_i / sum(w))` events each and the scores summed. Three reasons for a fixed
/// split rather than picking a source per event:
///
///   * it is stratified sampling, so it has *less* variance than a random choice;
///   * it is reproducible - two runs of the same model give the same answer, which a random
///     choice would only do for a fixed seed *and* a fixed source order;
///   * it needs nothing from the device. The kernel takes one source; the host calls it once
///     per source. A weighted source array in the kernel would be a per-track branch on a
///     quantity that never varies within a sub-run.
///
/// Each sub-run gets its own RNG seed, derived from the run's and the source index, so the
/// two beams are independent samples rather than the same shower fired from two places.
static void RunFromGui(App& a, int n_events) {
  if (a.scene_dirty) { RebuildScene(a); }
  if (a.run_manager == nullptr || a.engine == nullptr || n_events <= 0) { return; }

  // Refused, not run. Two volumes overlapping on the same layer have no rule deciding which
  // owns the shared space - the tie-break is whichever was added later - so the answer would
  // depend on the order the detector was built in. See g4builder_overlap.inc.
  if (!a.overlaps.empty()) {
    a.popup = Popup::kOverlap;
    a.status = "run refused: volumes overlap on the same layer";
    Log("run refused: two volumes overlap on the same layer. See the message above.");
    return;
  }

  // Which sources are in, and their shares.
  std::vector<int> live;
  double wsum = 0;
  for (std::size_t i = 0; i < a.model.sources.size(); ++i) {
    const builder::Source& s = a.model.sources[i];
    if (s.enabled && s.weight > 0) {
      live.push_back(static_cast<int>(i));
      wsum += s.weight;
    }
  }
  if (live.empty()) {
    Log("no enabled source with a positive weight: nothing to run.");
    a.status = "add a source, or enable one";
    return;
  }

  CUDA_CHECK(cudaMemset(a.traj.count, 0, sizeof(int)));
  CUDA_CHECK(cudaMemset(a.traj.dropped, 0, sizeof(int)));
  // How many events are *drawn*, which is not how many are run. The two used to be tied - the
  // cap was min(n, 100) - and the selftest worked around a crowded picture by running only 25
  // events. That made the picture legible and the numbers meaningless: 25 gammas through a
  // 30 mm cube is a fraction of one expected interaction, so the dose was one rare event or
  // nothing, and it read 2.81 MeV or exactly 0 depending on where the RNG stream happened to
  // land. A change that shifted the stream - more steps per track, say - looked exactly like
  // a transport bug. Twenty-five events of trajectories is all a picture can show; the run
  // size is now free to be whatever gives a number worth reporting.
  a.traj.max_event = std::min(n_events, a.traj_max_events);

  // The run itself, through the run manager. Not a split written out here: this was a third
  // copy of that arithmetic, alongside the viewer's and G4RunManager's own, and the three
  // disagreed - the viewer's never called GeneratePrimaries at all. See RunEvents.
  //
  // The primary generator is the ModelPrimary the run manager already holds, with no source
  // selected, so it registers every enabled source with its weight and RunEvents splits by
  // them. The shares come back for the log.
  std::vector<double> total, total_sq;
  std::vector<int> share;
  const auto stats = a.run_manager->RunEvents(n_events, total, total_sq, a.traj, &share);
  const std::vector<double> voxel_total = a.run_manager->VoxelScores();
  const double total_ms = stats.milliseconds;
  int total_events = 0;
  for (int s : share) { total_events += s; }

  // Anything that means the run did not transport what it was asked to. Silent before, and
  // that silence is what made a stepping bug read as "the dose is zero" rather than as
  // "half the tracks were abandoned".
  if (stats.abandoned > 0 || stats.overflow > 0) {
    Log(Fmt("  WARNING: %lld track(s) abandoned, %d buffer overflow(s), %d iterations",
            stats.abandoned, stats.overflow, stats.max_iterations));
  }
  Log(Fmt("  %lld track steps, %d iterations, peaks g/e/p %d/%d/%d", stats.track_steps,
          stats.max_iterations, stats.peak_gamma, stats.peak_electron, stats.peak_positron));
  if (live.size() > 1) {
    for (std::size_t k = 0; k < live.size() && k < share.size(); ++k) {
      const builder::Source& ms = a.model.sources[live[k]];
      Log(Fmt("  %-12s %d events (%.0f%%)", ms.name.c_str(), share[k],
              100.0 * ms.weight / wsum));
    }
  }
  int seg = 0;
  CUDA_CHECK(cudaMemcpy(&seg, a.traj.count, sizeof(int), cudaMemcpyDeviceToHost));
  a.n_segments = std::min(seg, a.traj.capacity);
  a.last_ms = total_ms;
  a.last_events = total_events;
  a.last_scores = total;
  a.last_voxel_scores = voxel_total;

  Log(Fmt("/run/beamOn %d   (%.1f ms, %.3g events/s, %d source%s)", total_events, total_ms,
          total_events / (total_ms * 1e-3), static_cast<int>(live.size()),
          live.size() == 1 ? "" : "s"));
  const auto& scorers = G4SDManager::GetSDMpointer()->Scorers();
  for (std::size_t i = 0; i < total.size() && i < scorers.size(); ++i) {
    const double mass = a.run_manager->ScoredMass(static_cast<int>(i));
    std::string line = Fmt("  %-14s %.6g MeV", scorers[i]->GetName().c_str(), total[i]);
    if (mass > 0) {
      line += Fmt("   dose %.5g pGy", total[i] * MeV / joule / mass * gray / picogray);
    }
    Log(line);
  }
  if (total.empty()) { Log("  no scorers: add one in the SCORERS panel to get a number"); }

  // The per-voxel map, if one was asked for. Summarised rather than dumped: a map is
  // thousands of numbers and the log holds a few hundred lines, so what goes here is what
  // tells you whether the run did what you meant - how many cells took anything at all, and
  // where the most went. The map itself is written beside a saved project.
  if (!voxel_total.empty()) {
    double peak = 0;
    int peak_cell = -1;
    int hit_cells = 0;
    double sum = 0;
    for (std::size_t i = 0; i < voxel_total.size(); ++i) {
      sum += voxel_total[i];
      if (voxel_total[i] > 0) { ++hit_cells; }
      if (voxel_total[i] > peak) {
        peak = voxel_total[i];
        peak_cell = static_cast<int>(i);
      }
    }
    Log(Fmt("  per-voxel: %d of %d cells took energy, %.6g MeV in all", hit_cells,
            static_cast<int>(voxel_total.size()), sum));
    if (peak_cell >= 0) {
      // Which cell, in the grid's own indices, so it can be found in the phantom rather than
      // being a number in a flat array.
      const auto& scene = a.run_manager->GetScene();
      for (std::size_t v = 0; v < scene.volumes.size(); ++v) {
        const auto& vol = scene.volumes[v];
        if (!vol.score_per_voxel) { continue; }
        const auto grid = geom::voxel_grid_of(vol.solid);
        const int n = grid.n[0] * grid.n[1] * grid.n[2];
        const int rel = peak_cell - grid.offset;
        if (rel < 0 || rel >= n) { continue; }
        Log(Fmt("  peak cell (%d, %d, %d) of %s: %.6g MeV", rel % grid.n[0],
                (rel / grid.n[0]) % grid.n[1], rel / (grid.n[0] * grid.n[1]),
                scene.names[v].c_str(), peak));
        break;
      }
    }
    Log("  the full map is written as a .dose file when the project is saved");
  }
  a.status = Fmt("ran %d events in %.1f ms", total_events, total_ms);
}



/// What the import dialog was told about a voxel file.
///
/// A raw file has no header: nothing in it says how many cells there are, what type they are,
/// or how big one cell is. The reader can guess a cube from the file size and the type from
/// the value range, and does when this says nothing - but a 512x512x120 CT is not a cube, and
/// a wrong guess there produces a scrambled volume rather than an error, which is the worst
/// kind of wrong. So the dialog asks, and this carries the answers.
///
/// A .mat file carries its own shape and type. It does not carry a *resolution*: MATLAB has
/// no idea how big a voxel is either, so the spacing here applies to both kinds.
struct VoxelImportOptions {
  bool have_dims = false;             ///< false: guess a cube from the file size
  int nx = 0, ny = 0, nz = 0;
  bool have_type = false;             ///< false: try int16, then uint8, then float32
  VoxelType type = VoxelType::kInt16;
  /// What the numbers mean. Only kIndices imports; see VoxelMeaning in model.hh.
  VoxelMeaning meaning = VoxelMeaning::kIndices;
  double res_mm[3] = {1, 1, 1};       ///< cell size along each axis
};

/// Re-reads the raw values of every imported voxel volume, so they can be reclassified.
static void ReloadVoxelSources(App& a) {
  for (Solid& s : a.model.solids) {
    if (s.shape != Shape::kVoxelGrid || s.source_file.empty()) { continue; }
    if (!s.voxel_values.empty()) { continue; }
    VoxelData v;
    bool ok = false;
    if (detail::EndsWith(s.source_file, ".mat")) {
      ok = ReadMatVoxels(s.source_file, v);
    } else {
      // The dimensions come from the document, not from a fresh guess: the import already
      // established them - the user may have typed them in - and p[3..5] is where they were
      // written. Re-guessing a cube here would reopen a 512x512x120 CT as something else, or
      // fail outright, having been imported perfectly well the first time.
      const int nx = static_cast<int>(s.p[3] + 0.5);
      const int ny = static_cast<int>(s.p[4] + 0.5);
      const int nz = static_cast<int>(s.p[5] + 0.5);
      ok = ReadRawVoxels(s.source_file, VoxelType::kInt16, nx, ny, nz, v)
           || ReadRawVoxels(s.source_file, VoxelType::kUInt8, nx, ny, nz, v)
           || ReadRawVoxels(s.source_file, VoxelType::kFloat32, nx, ny, nz, v);
    }
    if (ok) {
      s.voxel_values = v.value;
      Log("reloaded voxel values for " + s.name + " from " + s.source_file);
    } else {
      Log(s.name + ": could not re-read " + s.source_file);
      Log("  the saved material indices still work; reclassification does not");
    }
  }
}

/// Re-reads the triangles of every imported mesh, so a reopened model can be transported.
///
/// The .model document stores the file a mesh came from, not its triangles - a coarse import
/// is a few hundred thousand numbers, which does not belong in a text document meant to be
/// read and edited. A saved *project* keeps its own binary sidecar and needs no file, but the
/// model document does. If the CAD file has moved, the mesh cannot be rebuilt, and saying so
/// here is much better than the alternative: G4TessellatedSolid refusing to be placed with no
/// facets, at /run/initialize, with nothing to connect it to a file that moved.
/// Recenters a mesh on the middle of its own extent, and reports where that was.
///
/// The anchor of an imported solid is its local origin: the point a placement puts somewhere,
/// and the point a rotation turns about. Straight from a CAD file that is wherever the
/// modeller's origin happened to be - often a corner, sometimes far outside the part - so
/// dragging the volume moved it from a handle that was not on it, and rotating it swung it
/// around a point in empty space.
///
/// Recentering the triangles and putting the placement at the old center leaves the mesh
/// exactly where the file said it was, while making the local origin the middle of the object
/// in each axis. Typing zeroes into the position boxes now centers it on the world origin,
/// which is what "position" reads as.
///
/// This also fixes a plain error: the placement used to be set to the bounding-box center
/// while the triangles kept their file coordinates, so an imported part appeared offset from
/// its file position by that center - doubled for a part already modelled about its own
/// middle, and invisible for one modelled about a corner, which is why it went unnoticed.
///
/// @param[in,out] mesh    triangle vertices, three floats per vertex
/// @param[out]    center  where the middle was, in the file's coordinates
/// @param[out]    half    half the extent in each axis
static void CenterMesh(std::vector<float>& mesh, double center[3], double half[3]) {
  for (int k = 0; k < 3; ++k) { center[k] = half[k] = 0; }
  if (mesh.size() < 3) { return; }
  double lo[3] = {1e30, 1e30, 1e30}, hi[3] = {-1e30, -1e30, -1e30};
  for (std::size_t i = 0; i + 2 < mesh.size(); i += 3) {
    for (int k = 0; k < 3; ++k) {
      lo[k] = std::min<double>(lo[k], mesh[i + k]);
      hi[k] = std::max<double>(hi[k], mesh[i + k]);
    }
  }
  for (int k = 0; k < 3; ++k) {
    center[k] = 0.5 * (hi[k] + lo[k]);
    half[k] = 0.5 * (hi[k] - lo[k]);
  }
  for (std::size_t i = 0; i + 2 < mesh.size(); i += 3) {
    for (int k = 0; k < 3; ++k) {
      mesh[i + k] = static_cast<float>(mesh[i + k] - center[k]);
    }
  }
}

static void ReloadMeshSources(App& a) {
  for (Solid& s : a.model.solids) {
    if (s.shape != Shape::kImportedMesh || !s.mesh.empty()) { continue; }
    if (s.source_file.empty()) {
      Log(s.name + ": an imported mesh with no source file; it cannot be rebuilt");
      continue;
    }
    std::string note;
    if (ReadMesh(s.source_file, s.mesh, note)) {
      // Recentered exactly as the import did, and the placement left alone: the document
      // already holds the placement the import computed, so re-adding the center here would
      // move the volume a second time every time the model was opened.
      double center[3], half[3];
      CenterMesh(s.mesh, center, half);
      Log("reloaded " + s.name + " from " + s.source_file);
      Log("  " + note);
    } else {
      Log(s.name + ": could not re-read " + s.source_file);
      Log("  " + note);
      Log("  re-import the file, or delete the volume; it has no triangles to transport");
    }
  }
}
// ---------------------------------------------------------------- import

/// Imports a mesh or a voxel volume, and reports what it found.
///
/// The kind of a voxel file - a segmented phantom or a CT in Hounsfield units - cannot be
/// determined from the data: a CT with few distinct values looks segmented, and a phantom
/// stored as float looks continuous. So it is asked, with the file's own value range shown to
/// make the answer obvious.
/// The enclosed volume of an imported mesh, for the import report. A number far from the
/// bounding box's, or a negative-looking one, is how a mesh that is not closed announces
/// itself - and an unclosed mesh cannot be transported, because containment is a parity count.
static double MeshVolumeOf(const std::vector<float>& mesh) {
  std::vector<double> d(mesh.begin(), mesh.end());
  return g4gpu::geom::mesh_volume(d.data(), static_cast<int>(d.size() / 9));
}

static bool ImportFile(App& a, const std::string& path,
                       const VoxelImportOptions* opt = nullptr) {
  if (detail::EndsWith(path, ".h5") || detail::EndsWith(path, ".hdf5")) {
    Log("HDF5 is not supported: this build does not link the HDF5 library.");
    Log("  Convert to .raw or an uncompressed .mat and import that.");
    a.status = "HDF5 not supported";
    return false;
  }

  Solid s;
  s.source_file = path;
  const std::size_t slash = path.find_last_of("/\\");
  s.name = (slash == std::string::npos) ? path : path.substr(slash + 1);
  for (char& c : s.name) {
    if (!std::isalnum(static_cast<unsigned char>(c))) { c = '_'; }
  }
  s.material = a.model.solids.empty() ? -1 : a.model.solids[0].material;
  s.layer = 2;

  if (detail::EndsWith(path, ".stl") || detail::EndsWith(path, ".obj")
      || detail::EndsWith(path, ".ply")) {
    std::string note;
    if (!ReadMesh(path, s.mesh, note)) {
      a.status = "mesh import failed";
      return false;
    }
    s.shape = Shape::kImportedMesh;
    // Anchored on the middle of its own extent, and placed where the file put it. See
    // CenterMesh. The half-extents come back too: they are the bounding box the volume falls
    // back to when its source file has moved.
    double center[3], half[3];
    CenterMesh(s.mesh, center, half);
    for (int k = 0; k < 3; ++k) {
      s.p[k] = half[k];
      s.pos[k] = center[k];
    }
    Log("imported " + path);
    Log("  " + note);
    Log(Fmt("  bounds %.1f x %.1f x %.1f mm about (%.1f, %.1f, %.1f)", 2 * s.p[0], 2 * s.p[1],
            2 * s.p[2], s.pos[0], s.pos[1], s.pos[2]));
    Log(Fmt("  volume %.4g mm3 (divergence theorem over %zu triangles)",
            MeshVolumeOf(s.mesh), s.mesh.size() / 9));
    a.sel_solid = a.model.AddSolid(s);
    a.pfield_for = -2;
    a.scene_dirty = true;
    a.status = "imported " + s.name;
    return true;
  }

  // Refused here as well as greyed out in the dialog. The dialog is one caller; a model
  // reopened, a file dropped in, or a future caller would otherwise reach the classifier with
  // a density map and get one "material" per distinct density - thousands of unassignable
  // classes, which looks like a successful import until you try to use it.
  if (opt != nullptr && opt->meaning == VoxelMeaning::kProperties) {
    Log("physical-property voxel data is not implemented.");
    Log("  Values that *are* a quantity - a density, a Hounsfield unit - need a calibration");
    Log("  curve and a material table generated from it. Import as material indices, or");
    Log("  segment the data first.");
    a.status = "physical properties not implemented";
    return false;
  }

  VoxelData v;
  bool ok = false;
  if (detail::EndsWith(path, ".mat")) {
    ok = ReadMatVoxels(path, v);
  } else if (detail::EndsWith(path, ".raw") || detail::EndsWith(path, ".bin")) {
    // Dimensions and type from the dialog when it asked, guessed when it did not. A raw file
    // has no header, so with nothing given the shape can only be a cube from the file size,
    // and the type only whichever of int16, uint8 and float32 makes the byte count work.
    // Both guesses are reported in v.note: a silently wrong one is a scrambled volume rather
    // than an error, which is the worse way to be wrong.
    const int nx = (opt != nullptr && opt->have_dims) ? opt->nx : 0;
    const int ny = (opt != nullptr && opt->have_dims) ? opt->ny : 0;
    const int nz = (opt != nullptr && opt->have_dims) ? opt->nz : 0;
    if (opt != nullptr && opt->have_type) {
      ok = ReadRawVoxels(path, opt->type, nx, ny, nz, v);
    } else {
      ok = ReadRawVoxels(path, VoxelType::kInt16, nx, ny, nz, v);
      if (!ok) { ok = ReadRawVoxels(path, VoxelType::kUInt8, nx, ny, nz, v); }
      if (!ok) { ok = ReadRawVoxels(path, VoxelType::kFloat32, nx, ny, nz, v); }
    }
  } else {
    Log("unrecognised extension: " + path);
    Log("  supported: .stl (mesh), .raw .bin .mat (voxels)");
    a.status = "unrecognised file type";
    return false;
  }
  if (!ok) {
    a.status = "voxel import failed";
    return false;
  }

  double lo = 1e30, hi = -1e30;
  for (float x : v.value) {
    lo = std::min<double>(lo, x);
    hi = std::max<double>(hi, x);
  }
  Log("imported " + path);
  Log("  " + v.note);
  Log(Fmt("  %d x %d x %d voxels, values %.0f to %.0f", v.nx, v.ny, v.nz, lo, hi));

  // The kind question. Negative values are HU by convention - nothing else is stored below
  // zero - which settles it without asking in the common case.
  const VoxelKind kind = (lo < -200) ? VoxelKind::kContinuous : VoxelKind::kDiscrete;
  Log(kind == VoxelKind::kContinuous
          ? "  values below -200 mean Hounsfield units: read as a CT"
          : "  no negative values: read as a segmented phantom, one class per value");
  Log("  (change it in the solid's panel if that is wrong)");

  // The cell size, and from it the extent. A voxel array has no idea how big a voxel is -
  // neither a raw dump nor a .mat carries a spacing - so it is asked for, and 1 mm is only
  // the default. Getting it wrong scales the whole phantom, and with it every dose.
  const double res[3] = {
      (opt != nullptr) ? opt->res_mm[0] : 1.0,
      (opt != nullptr) ? opt->res_mm[1] : 1.0,
      (opt != nullptr) ? opt->res_mm[2] : 1.0,
  };
  s.shape = Shape::kVoxelGrid;
  s.voxel_meaning = (opt != nullptr) ? opt->meaning : VoxelMeaning::kIndices;
  s.p[0] = v.nx * 0.5 * res[0];
  s.p[1] = v.ny * 0.5 * res[1];
  s.p[2] = v.nz * 0.5 * res[2];
  s.p[3] = v.nx;
  s.p[4] = v.ny;
  s.p[5] = v.nz;
  Log(Fmt("  %g x %g x %g mm voxels; the volume is %.1f x %.1f x %.1f mm, centered on its "
          "own middle",
          res[0], res[1], res[2], 2 * s.p[0], 2 * s.p[1], 2 * s.p[2]));
  s.voxel_values = v.value;
  std::string note;
  ClassifyVoxels(v, kind, s, note);
  Log("  " + note);
  Log("  assign a material to each class in the list to transport it");

  a.sel_solid = a.model.AddSolid(s);
  a.pfield_for = -2;
  a.scene_dirty = true;
  a.status = "imported " + s.name;
  return true;
}
// ---------------------------------------------------------------- menu bar

static void DrawMenuBar(App& a) {
  ui::Context& c = a.uic;
  a.menu.Begin(c, {0, 0, a.width, kMenuH});

  if (a.menu.Menu(c, "File")) {
    if (a.menu.Item(c, "New")) {
      // The dialog builds the model, rather than this replacing it and the dialog then
      // editing what it finds: a New that wiped the scene and *then* asked would have thrown
      // the old one away before the user had committed to a new one.
      SeedWorldDialog(a);
      a.popup = Popup::kWorld;
    }
    if (a.menu.Item(c, "Open")) {
      const std::string p =
          FileDialog(a, "g4gpu model\0*.model\0All files\0*.*\0", false, "model");
      if (!p.empty() && ReadModelFile(a.model, p)) {
        a.model_path = p;
        a.sel_solid = -1;
        a.scene_dirty = true;
        a.pfield_for = -2;
        // The .model file records where a voxel volume came from but not its millions of raw
        // values. Re-reading the source restores them, which is what makes reclassifying
        // (discrete to HU, or a different band set) possible after a reload. If the file has
        // moved, the volume still runs - the saved .cells hold the material indices - and only
        // reclassification is unavailable, which the log says.
        ReloadVoxelSources(a);
        ReloadMeshSources(a);
      }
    }
    a.menu.Separator(c);
    if (a.menu.Item(c, "Save project")) {
      if (a.model_path.empty()) {
        const std::string p =
            FileDialog(a, "g4gpu model\0*.model\0", true, "model");
        if (!p.empty()) {
          a.model_path = p;
          const std::size_t slash = p.find_last_of("/\\");
          SaveProjectTo(a, (slash == std::string::npos) ? "." : p.substr(0, slash));
        }
      } else {
        const std::size_t slash = a.model_path.find_last_of("/\\");
        SaveProjectTo(a, (slash == std::string::npos) ? "."
                                                      : a.model_path.substr(0, slash));
      }
    }
    if (a.menu.Item(c, "Save project as")) {
      const std::string p = FileDialog(a, "g4gpu model\0*.model\0", true, "model");
      if (!p.empty()) {
        a.model_path = p;
        const std::size_t slash = p.find_last_of("/\\");
        SaveProjectTo(a, (slash == std::string::npos) ? "." : p.substr(0, slash));
      }
    }
    a.menu.Separator(c);
    if (a.menu.Item(c, "Exit")) { a.running = false; }
    a.menu.EndMenu(c);
  }

  if (a.menu.Menu(c, "Insert")) {
    // A lateral submenu rather than a dialog. The four primitives that used to sit directly
    // on this menu are gone from it: they were a shortcut for the four most likely shapes,
    // and a submenu that lists all seventeen is both shorter and complete.
    if (a.menu.Submenu(c, "Primitive")) {
      struct P { const char* label; Shape shape; };
      static const P kPrims[] = {
          {"Box", Shape::kBox},
          {"Cylinder (Tubs)", Shape::kTubs},
          {"Cone (Cons)", Shape::kCons},
          {"Sphere", Shape::kSphere},
          {"Orb", Shape::kOrb},
          {"Torus", Shape::kTorus},
          {"Trapezoid (Trd)", Shape::kTrd},
          {"General trap", Shape::kTrap},
          {"Parallelepiped", Shape::kPara},
          {"Elliptical tube", Shape::kEllipticalTube},
          {"Ellipsoid", Shape::kEllipsoid},
          {"Elliptical cone", Shape::kEllipticalCone},
          {"Paraboloid", Shape::kParaboloid},
          {"Hyperboloid", Shape::kHype},
          {"Tetrahedron", Shape::kTet},
          {"Polycone", Shape::kPolycone},
          {"Polyhedra", Shape::kPolyhedra},
      };
      for (const P& p : kPrims) {
        if (a.menu.SubItem(c, p.label)) { InsertPrimitive(a, p.shape); }
      }
    }
    a.menu.EndSubmenu(c);
    a.menu.Separator(c);
    // Split, because they are different operations on different files with different
    // options, and one dialog that asked "is this a mesh or a stack of slices?" was a
    // question the file extension already answers.
    if (a.menu.Item(c, "Import CAD mesh")) { a.popup = Popup::kImportCad; }
    if (a.menu.Item(c, "Import voxel volume")) { a.popup = Popup::kImportVoxel; }
    if (a.menu.Item(c, "Flat detector")) { a.popup = Popup::kFlatDetector; }
    a.menu.EndMenu(c);
  }

  if (a.menu.Menu(c, "Boolean")) {
    if (a.menu.Item(c, "Union")) { InsertBoolean(a, Shape::kUnion); }
    if (a.menu.Item(c, "Subtract")) { InsertBoolean(a, Shape::kSubtraction); }
    if (a.menu.Item(c, "Intersect")) { InsertBoolean(a, Shape::kIntersection); }
    a.menu.EndMenu(c);
  }

  if (a.menu.Menu(c, "View")) {
    if (a.menu.Item(c, "Visualization attributes")) { a.popup = Popup::kVisAttributes; }
    a.menu.Separator(c);
    if (a.menu.Item(c, "Reset camera")) {
      a.azimuth = 0.7f;
      a.elevation = 0.35f;
      a.distance = a.home_distance;
      a.target = vis::Vec3f{0.f, 0.f, 0.f};
    }
    if (a.menu.Item(c, a.show_tracks ? "Hide tracks" : "Show tracks")) {
      a.show_tracks = !a.show_tracks;
    }
    a.menu.EndMenu(c);
  }
  a.menu.EndBar(c);

  // Title, right-aligned, with a dirty marker.
  const std::string title =
      a.model.name + (a.model.dirty ? " *" : "")
      + (a.model_path.empty() ? "   (unsaved)" : "   " + a.model_path);
  c.canvas.TextRight(a.width - 10, (kMenuH - a.font.glyph_h) / 2, title, ui::theme::kTextDim);
}

// ---------------------------------------------------------------- deleting, and the panels
//
// The panels are in their own files. They came to nearly seven hundred lines between them,
// which is most of this one, and they are the part that changes most.

// ---------------------------------------------------------------- deleting, and the panels
//
// The panels are in their own files. They came to well over a thousand lines between them,
// which is most of this one, and they are the part that changes most.

#include "host/g4builder_delete.inc"
#include "host/g4builder_left.inc"
#include "host/g4builder_solids.inc"
#include "host/g4builder_right.inc"
#include "host/g4builder_popups.inc"
#include "host/g4builder_pick.inc"
#include "host/g4builder_overlap.inc"

// ---------------------------------------------------------------- status bar and log

static void DrawStatusBar(App& a) {
  ui::Context& c = a.uic;
  const ui::Rect bar{0, a.height - kStatusH, a.width, kStatusH};
  ui::Panel(c, bar, ui::theme::kPanelAlt);
  const int ty = bar.y + (kStatusH - a.font.glyph_h) / 2;

  // Half the bar each. Both are ellipsized rather than merely clipped, because the two used
  // to be drawn without a budget: a long log line - a full path plus a sentence, which is
  // what "wrote a project to ..." is - ran off the right edge and collided with the status
  // on the way. Half is arbitrary but it is a *bound*, which is the part that was missing.
  const int mid = a.width / 2;
  c.canvas.TextFit(8, ty, a.status, ui::theme::kTextDim, mid - 16);
  if (!a.log.empty()) {
    // Right-aligned while it fits, because that is where the eye looks for it; left-aligned
    // from the half-way mark when it does not, since right-aligning a truncated string would
    // hide the beginning, which is the part that says what happened.
    const std::string& last = a.log.back();
    const int budget = a.width - mid - 10;
    if (c.canvas.TextWidth(last) <= budget) {
      c.canvas.TextRight(a.width - 10, ty, last, ui::theme::kText);
    } else {
      c.canvas.TextFit(mid, ty, last, ui::theme::kText, budget);
    }
  }
}

// ---------------------------------------------------------------- popups

/// A color picker: a hue/value grid plus a row of greys. Enough to pick a distinguishable
/// color for a volume, which is all a volume color is for.
static void DrawColorPopup(App& a) {
  ui::Context& c = a.uic;
  if (a.popup_target < 0 || a.popup_target >= static_cast<int>(a.model.solids.size())) {
    a.popup = Popup::kNone;
    return;
  }
  Solid& s = a.model.solids[a.popup_target];
  const int cell = 22;
  const int cols = 12, rows = 6;
  const ui::Rect box{a.width / 2 - (cols * cell) / 2 - 10, a.height / 2 - 140,
                     cols * cell + 20, rows * cell + 150};
  ui::Panel(c, box);
  c.canvas.Text(box.x + 10, box.y + 8, "color for " + s.name, ui::theme::kText);

  const int gx = box.x + 10, gy = box.y + 8 + a.font.glyph_h + 8;
  for (int r = 0; r < rows; ++r) {
    for (int q = 0; q < cols; ++q) {
      // A hue sweep across, value down. HSV converted inline: a full color-space helper is
      // more than a swatch grid needs.
      const double h = 360.0 * q / cols;
      const double v = 1.0 - 0.13 * r;
      const double sat = (r == rows - 1) ? 0.0 : 0.75;
      const double hh = h / 60.0;
      const int i = static_cast<int>(hh) % 6;
      const double f = hh - static_cast<int>(hh);
      const double p = v * (1 - sat);
      const double qq = v * (1 - sat * f);
      const double t = v * (1 - sat * (1 - f));
      double rr = v, gg = v, bb = v;
      switch (i) {
        case 0: rr = v; gg = t; bb = p; break;
        case 1: rr = qq; gg = v; bb = p; break;
        case 2: rr = p; gg = v; bb = t; break;
        case 3: rr = p; gg = qq; bb = v; break;
        case 4: rr = t; gg = p; bb = v; break;
        default: rr = v; gg = p; bb = qq; break;
      }
      const ui::Rect sw{gx + q * cell, gy + r * cell, cell - 2, cell - 2};
      c.canvas.FillRect(sw, ui::rgb(static_cast<int>(rr * 255), static_cast<int>(gg * 255),
                                    static_cast<int>(bb * 255)));
      if (sw.Contains(a.input.mouse_x, a.input.mouse_y)) {
        c.canvas.StrokeRect(sw, ui::theme::kText);
        if (a.input.left_pressed) {
          s.r = static_cast<float>(rr);
          s.g = static_cast<float>(gg);
          s.b = static_cast<float>(bb);
          a.model.Touch();
          a.scene_dirty = true;
          a.popup = Popup::kNone;
        }
      }
    }
  }

  int y = gy + rows * cell + 8;

  // The exact colour, as numbers. The swatch grid above is for picking one quickly; this is
  // for entering the one you were given, or matching another volume's.
  {
    float rgb[3] = {s.r, s.g, s.b};
    if (ColorFields(a, 906, {box.x + 10, y, box.w - 20, a.font.glyph_h + 4}, "colour", rgb)) {
      s.r = rgb[0];
      s.g = rgb[1];
      s.b = rgb[2];
      a.model.Touch();
      a.scene_dirty = true;
    }
  }
  y += a.font.glyph_h + 10;

  // Opacity, as a percentage. Numeric for the same reason the channels are: 45% is a value
  // somebody writes down, and a slider cannot be told it.
  c.canvas.Text(box.x + 10, y + 3, "opacity", ui::theme::kTextDim);
  {
    if (ui::NumberInput(c, 903, {box.x + 74, y, 60, a.font.glyph_h + 4}, a.opacity_field)) {
      double v = a.opacity_field.value;
      if (v < 5) { v = 5; }      // fully transparent is indistinguishable from deleted
      if (v > 100) { v = 100; }
      a.opacity_field.Init(v, "%.0f");
      s.opacity = static_cast<float>(v / 100.0);
      a.model.Touch();
      a.scene_dirty = true;
    }
    c.canvas.Text(box.x + 140, y + 3, "%  (5 to 100)", ui::theme::kTextDim);
  }
  y += a.font.glyph_h + 10;
  // The preview: the solid's color at its opacity, over a light and a dark square.
  {
    const int pw = 60, ph = a.font.glyph_h + 6;
    c.canvas.FillRect({box.x + 10, y, pw / 2, ph}, ui::rgb(70, 74, 82));
    c.canvas.FillRect({box.x + 10 + pw / 2, y, pw / 2, ph}, ui::rgb(20, 22, 25));
    const int al = static_cast<int>(s.opacity * 255);
    c.canvas.FillRect({box.x + 10, y, pw, ph},
                      ui::rgba(static_cast<int>(s.r * 255), static_cast<int>(s.g * 255),
                               static_cast<int>(s.b * 255), al));
    c.canvas.StrokeRect({box.x + 10, y, pw, ph}, ui::theme::kBorder);
    c.canvas.Text(box.x + 80, y + 3, "over dark and light", ui::theme::kTextDim);
  }
  y += a.font.glyph_h + 12;
  bool vis = s.visible;
  if (ui::Checkbox(c, 900, {box.x + 10, y, box.w - 20, a.font.glyph_h + 4}, "visible", vis)) {
    s.visible = vis;
    a.model.Touch();
    a.scene_dirty = true;
  }
  y += a.font.glyph_h + 8;
  bool wf = s.wireframe;
  if (ui::Checkbox(c, 901, {box.x + 10, y, box.w - 20, a.font.glyph_h + 4}, "wireframe", wf)) {
    s.wireframe = wf;
    a.model.Touch();
    a.scene_dirty = true;
  }
  y += a.font.glyph_h + 8;
  if (ui::Button(c, 902, {box.x + box.w - 80, y, 70, a.font.glyph_h + 6}, "Close")) {
    a.popup = Popup::kNone;
  }
}

/// The full primitive list, for the shapes not on the Insert menu directly.
static void DrawInsertPopup(App& a) {
  ui::Context& c = a.uic;
  const Shape all[] = {Shape::kBox, Shape::kTubs, Shape::kCons, Shape::kSphere, Shape::kOrb,
                       Shape::kTorus, Shape::kTrd, Shape::kTrap, Shape::kPara,
                       Shape::kEllipticalTube, Shape::kEllipsoid, Shape::kEllipticalCone,
                       Shape::kParaboloid, Shape::kHype, Shape::kTet, Shape::kPolycone,
                       Shape::kPolyhedra};
  const int n = sizeof(all) / sizeof(all[0]);
  const int row = a.font.glyph_h + 6;
  const ui::Rect box{a.width / 2 - 120, a.height / 2 - (n * row) / 2 - 20, 240,
                     n * row + 50};
  ui::Panel(c, box);
  c.canvas.Text(box.x + 10, box.y + 8, "insert a primitive", ui::theme::kText);
  int y = box.y + 8 + a.font.glyph_h + 6;
  for (int i = 0; i < n; ++i) {
    if (ui::ListRow(c, 1000 + i, {box.x + 6, y, box.w - 12, row}, ShapeName(all[i]), false)) {
      InsertPrimitive(a, all[i]);
      a.popup = Popup::kNone;
    }
    y += row;
  }
  if (ui::Button(c, 1100, {box.x + box.w - 80, y + 4, 70, row}, "Cancel")) {
    a.popup = Popup::kNone;
  }
}

/// Importing a CAD mesh. Separate from the voxel importer: different formats, different
/// options, and the file extension already answers "which of the two is this?".
static void DrawImportCadPopup(App& a) {
  ui::Context& c = a.uic;
  const int lh = a.font.glyph_h;
  const int row = lh + 6;
  const ui::Rect r = PopupFrame(a, 500, 9 * lh + 5 * row, "Import a CAD mesh");
  int y = r.y;
  c.canvas.Text(r.x + 14, y, ".stl   binary or ASCII, triangles", ui::theme::kTextDim);
  y += lh + 2;
  c.canvas.Text(r.x + 14, y, ".obj   Wavefront, triangles and quads, with face colors",
                ui::theme::kTextDim);
  y += lh + 2;
  c.canvas.Text(r.x + 14, y, ".ply   ASCII only", ui::theme::kTextDim);
  y += lh + 8;
  c.canvas.Text(r.x + 14, y, "The mesh is transported against its own triangles: a BVH, exact",
                ui::theme::kTextDim);
  y += lh + 2;
  c.canvas.Text(r.x + 14, y, "containment and exact safety. It agrees with the same shape as",
                ui::theme::kTextDim);
  y += lh + 2;
  c.canvas.Text(r.x + 14, y, "an analytic solid to 0.00 sigma.", ui::theme::kTextDim);
  y += lh + 8;
  c.canvas.Text(r.x + 14, y, "An .obj with a material library beside it gets its face colors;",
                ui::theme::kTextDim);
  y += lh + 2;
  c.canvas.Text(r.x + 14, y, "otherwise the whole mesh takes one color you can change.",
                ui::theme::kTextDim);
  y += lh + 12;

  if (ui::Button(c, 1200, {r.x + 14, y, 200, row + 2}, "Choose a mesh file")) {
    const std::string p = FileDialog(
        a, "CAD mesh\0*.stl;*.obj;*.ply\0STL\0*.stl\0Wavefront OBJ\0*.obj\0PLY\0*.ply\0"
           "All files\0*.*\0",
        false, nullptr);
    if (!p.empty()) {
      Log("import requested: " + p);
      if (!ImportFile(a, p)) { Log("  import failed; see the message above"); }
      a.popup = Popup::kNone;
    }
  }
  if (ui::Button(c, 1201, {r.x + r.w - 96, y, 82, row + 2}, "Cancel")) {
    a.popup = Popup::kNone;
  }
}

/// Importing a voxel volume - a CT stack, a segmentation, a dose grid.
/// The sample types a raw file can hold, in the order kVoxelTypeIds maps them.
static const char* const kVoxelTypeNames[7] = {
    "guess", "int16", "uint8", "uint16", "int32", "float32", "float64"};
static const VoxelType kVoxelTypeIds[7] = {
    VoxelType::kInt16,  // index 0 is "guess" and is never used as a type
    VoxelType::kInt16,   VoxelType::kUInt8,   VoxelType::kUInt16,
    VoxelType::kInt32,   VoxelType::kFloat32, VoxelType::kFloat64};

/// Seeds the voxel-import dialog. Called when it opens, not when it draws.
static void SeedVoxelDialog(App& a) {
  a.vox_n[0].Init(256, "%.0f");
  a.vox_n[1].Init(256, "%.0f");
  a.vox_n[2].Init(256, "%.0f");
  a.vox_res[0].Init(1, "%g");
  a.vox_res[1].Init(1, "%g");
  a.vox_res[2].Init(1, "%g");
  a.vox_type = 0;
  a.vox_res_unit = 1;  // mm
  a.vox_meaning = 0;
  a.vox_guess_dims = true;
  a.vox_dialog_seeded = true;
}

static void DrawImportVoxelPopup(App& a) {
  ui::Context& c = a.uic;
  const int lh = a.font.glyph_h;
  const int row = lh + 6;
  if (!a.vox_dialog_seeded) { SeedVoxelDialog(a); }

  const ui::Rect r = PopupFrame(a, 560, 11 * lh + 9 * row + 26, "Import a voxel volume");
  int y = r.y;

  y += c.canvas.TextWrapped(
      {r.x + 14, y, r.w - 28, 4 * lh},
      "A raw file has no header: nothing in it says how many cells there are or what type "
      "they hold. A .mat file carries both, but neither kind carries a voxel size.",
      ui::theme::kTextDim);
  y += 8;

  // ---- dimensions
  ui::Checkbox(c, 1220, {r.x + 14, y, r.w - 28, row}, "guess the shape from the file size",
               a.vox_guess_dims);
  y += row + 2;
  c.canvas.Text(r.x + 14, y + 3, "cells", ui::theme::kTextDim);
  for (int k = 0; k < 3; ++k) {
    ui::NumberInput(c, 1230 + k, {r.x + 76 + k * 84, y, 78, row}, a.vox_n[k], "",
                    !a.vox_guess_dims);
  }
  c.canvas.TextFit(r.x + 76 + 3 * 84 + 6, y + 3, "x  y  z", ui::theme::kTextDim,
                   r.w - (76 + 3 * 84 + 20));
  y += row + 4;

  c.canvas.Text(r.x + 14, y + 3, "type", ui::theme::kTextDim);
  ui::Select(c, 1240, {r.x + 76, y, 120, row}, a.vox_type, kVoxelTypeNames, 7);
  c.canvas.TextFit(r.x + 204, y + 3, "raw files only; .mat carries its own",
                   ui::theme::kTextDim, r.w - 218);
  y += row + 8;

  // ---- what the numbers mean
  //
  // The question the file cannot answer, and the one that decides everything downstream: a
  // grid of 1s and 2s is a segmentation with two materials to assign, and a grid of 1.02s and
  // 1.06s is a density map with no categories in it at all. Reading the second as the first
  // gives one "material" per distinct density - thousands of them - and a phantom nobody can
  // assign. See VoxelMeaning in builder/model.hh.
  static const char* const kMeaningNames[2] = {"material indices", "physical properties"};
  c.canvas.Text(r.x + 14, y + 3, "values", ui::theme::kTextDim);
  ui::Select(c, 1270, {r.x + 76, y, 190, row}, a.vox_meaning, kMeaningNames, 2);
  y += row + 2;
  y += c.canvas.TextWrapped(
      {r.x + 76, y, r.w - 90, 3 * lh},
      a.vox_meaning == 0
          ? "each distinct value names a material; you assign one per value after the import"
          : "each value is a quantity - a density, a Hounsfield unit. NOT IMPLEMENTED: this "
            "needs a calibration curve and a material table generated from it.",
      a.vox_meaning == 0 ? ui::theme::kTextDim : ui::theme::kError);
  y += 6;

  // ---- resolution
  c.canvas.Text(r.x + 14, y + 3, "voxel", ui::theme::kTextDim);
  for (int k = 0; k < 3; ++k) {
    ui::NumberInput(c, 1250 + k, {r.x + 76 + k * 84, y, 78, row}, a.vox_res[k]);
  }
  y += row + 4;
  c.canvas.Text(r.x + 14, y + 3, "units", ui::theme::kTextDim);
  ui::Select(c, 1260, {r.x + 76, y, 78, row}, a.vox_res_unit, kLengthUnits, 4);
  c.canvas.TextFit(r.x + 162, y + 3, "applies to .raw, .bin and .mat alike",
                   ui::theme::kTextDim, r.w - 176);
  y += row + 6;

  {
    const double u = LengthUnitMm(a.vox_res_unit);
    const double sx = (a.vox_guess_dims ? 0 : a.vox_n[0].value) * a.vox_res[0].value * u;
    const double sy = (a.vox_guess_dims ? 0 : a.vox_n[1].value) * a.vox_res[1].value * u;
    const double sz = (a.vox_guess_dims ? 0 : a.vox_n[2].value) * a.vox_res[2].value * u;
    // The extent, computed as you type, because "0.5 mm voxels" and "512 of them" is a
    // 25.6 cm phantom and neither number says so on its own.
    c.canvas.TextFit(r.x + 14, y,
                     a.vox_guess_dims
                         ? "extent: known once the shape is read from the file"
                         : Fmt("extent: %.1f x %.1f x %.1f mm", sx, sy, sz),
                     ui::theme::kAccent, r.w - 28);
  }
  y += lh + 10;

  c.canvas.TextFit(r.x + 14, y, ".h5 needs the HDF5 library, which this build does not link.",
                   ui::theme::kWarn, r.w - 28);
  y += lh + 10;

  // Disabled rather than hidden for physical properties: hiding it would make the dialog look
  // as though the choice above had no consequence, and the consequence is the whole point.
  const bool can_import = (a.vox_meaning == 0);
  if (ui::Button(c, 1210, {r.x + 14, y, 210, row + 2}, "Choose a voxel file", can_import)) {
    const std::string p = FileDialog(
        a, "Voxel data\0*.raw;*.bin;*.mat\0Raw\0*.raw;*.bin\0MATLAB\0*.mat\0All files\0*.*\0",
        false, nullptr);
    if (!p.empty()) {
      VoxelImportOptions opt;
      opt.have_dims = !a.vox_guess_dims;
      opt.nx = static_cast<int>(a.vox_n[0].value + 0.5);
      opt.ny = static_cast<int>(a.vox_n[1].value + 0.5);
      opt.nz = static_cast<int>(a.vox_n[2].value + 0.5);
      opt.have_type = (a.vox_type > 0);
      opt.type = kVoxelTypeIds[a.vox_type & 7];
      opt.meaning = (a.vox_meaning == 1) ? VoxelMeaning::kProperties
                                         : VoxelMeaning::kIndices;
      const double u = LengthUnitMm(a.vox_res_unit);
      for (int k = 0; k < 3; ++k) { opt.res_mm[k] = a.vox_res[k].value * u; }
      Log("import requested: " + p);
      if (!ImportFile(a, p, &opt)) { Log("  import failed; see the message above"); }
      a.vox_dialog_seeded = false;
      a.popup = Popup::kNone;
    }
  }
  if (!can_import) {
    c.canvas.TextFit(r.x + 232, y + 6, "physical properties are not implemented",
                     ui::theme::kError, r.w - 340);
  }
  if (ui::Button(c, 1211, {r.x + r.w - 96, y, 82, row + 2}, "Cancel")) {
    a.vox_dialog_seeded = false;
    a.popup = Popup::kNone;
  }
}

static void DrawFlatDetectorPopup(App& a) {
  ui::Context& c = a.uic;
  const int row = a.font.glyph_h + 6;
  const ui::Rect box{a.width / 2 - 200, a.height / 2 - 130, 400, 260};
  ui::Panel(c, box);
  int y = box.y + 10;
  c.canvas.Text(box.x + 12, y, "Flat detector", ui::theme::kText);
  y += a.font.glyph_h + 4;
  c.canvas.Text(box.x + 12, y, "a voxelised box, one cell per pixel", ui::theme::kTextDim);
  y += a.font.glyph_h + 10;

  static ui::NumberField w, h, d, rows_f, cols_f;
  static bool init = false;
  if (!init) {
    w.Init(200, "%.0f");
    h.Init(200, "%.0f");
    d.Init(10, "%.0f");
    rows_f.Init(64, "%.0f");
    cols_f.Init(64, "%.0f");
    init = true;
  }
  struct Row { const char* label; ui::NumberField* f; const char* unit; };
  const Row fields[] = {{"width", &w, "mm"}, {"height", &h, "mm"}, {"depth", &d, "mm"},
                        {"rows", &rows_f, ""}, {"columns", &cols_f, ""}};
  for (int i = 0; i < 5; ++i) {
    c.canvas.Text(box.x + 12, y + 3, fields[i].label, ui::theme::kTextDim);
    ui::NumberInput(c, 1300 + i, {box.x + 110, y, 90, row}, *fields[i].f);
    c.canvas.Text(box.x + 206, y + 3, fields[i].unit, ui::theme::kTextDim);
    y += row + 3;
  }
  y += 8;
  if (ui::Button(c, 1310, {box.x + 12, y, 140, row + 4}, "Create")) {
    Solid s;
    s.shape = Shape::kVoxelGrid;
    s.name = "Detector" + std::to_string(a.model.solids.size());
    s.p[0] = w.value * 0.5;
    s.p[1] = h.value * 0.5;
    s.p[2] = d.value * 0.5;
    s.p[3] = cols_f.value;   // nx
    s.p[4] = rows_f.value;   // ny
    s.p[5] = 1;              // nz: a flat detector is one cell deep
    s.det_rows = static_cast<int>(rows_f.value);
    s.det_cols = static_cast<int>(cols_f.value);
    s.material = a.model.solids.empty() ? -1 : a.model.solids[0].material;
    s.layer = 2;
    s.r = 0.4f;
    s.g = 0.8f;
    s.b = 0.9f;
    a.sel_solid = a.model.AddSolid(s);
    a.pfield_for = -2;
    a.scene_dirty = true;
    Log(Fmt("flat detector %s: %.0f x %.0f mm, %d x %d cells", s.name.c_str(), w.value,
            h.value, s.det_cols, s.det_rows));
    Log("  assign it a material; cells with none fall back to the volume material");
    a.popup = Popup::kNone;
  }
  if (ui::Button(c, 1311, {box.x + box.w - 90, y, 78, row + 4}, "Cancel")) {
    a.popup = Popup::kNone;
  }
}




/// Saves the current frame as a PNG. Used by the selftest to capture pop-ups.
///
/// The dialogs are the part of the GUI nothing else checks: the run exercises the model and
/// the render, and a dialog that laid out wrongly - text past its frame, a control off the
/// bottom, a list that did not fit - would look fine to every assertion in the pipeline and
/// wrong to the first person who opened it.
static void SaveFramePng(App& a, const char* path) {
  std::vector<unsigned char> rgb(static_cast<std::size_t>(a.width) * a.height * 3);
  for (std::size_t i = 0; i < static_cast<std::size_t>(a.width) * a.height; ++i) {
    const unsigned p = a.host_rgba[i];
    rgb[i * 3 + 0] = static_cast<unsigned char>(p & 255);
    rgb[i * 3 + 1] = static_cast<unsigned char>((p >> 8) & 255);
    rgb[i * 3 + 2] = static_cast<unsigned char>((p >> 16) & 255);
  }
  vis::write_png_rgb(path, rgb.data(), a.width, a.height);
  std::printf("selftest: saved %s\n", path);
}

/// Checks that flagging a scorer custom does not change what it reports.
///
/// The claim written into the generated file, and into the panel beside the checkbox, is that
/// an unedited custom scorer reports exactly what a stock one would - that the flag changes
/// the code and not the answer. It is worth checking rather than asserting, because it is not
/// free: a filtered scorer's total is accumulated event by event on the host, through
/// Accept(), while an unfiltered one's comes from the device array summed in a fixed order.
/// Two different additions of the same numbers.
///
/// So the test runs the same seed twice - once with the flag, once without - and requires the
/// totals to agree. They should agree *exactly*: both are sums of the same per-event values,
/// and the per-event values are doubles that are not being re-associated within an event. A
/// difference would mean the filtered path is picking up different events, not different
/// rounding.
static void SelftestCheckCustomScorer(App& a) {
  int idx = -1;
  for (std::size_t i = 0; i < a.model.scorers.size(); ++i) {
    if (a.model.scorers[i].custom) { idx = static_cast<int>(i); }
  }
  if (idx < 0) {
    std::printf("selftest: FAILED - no custom scorer to check\n");
    return;
  }
  const std::string name = a.model.scorers[static_cast<std::size_t>(idx)].name;

  auto total_of = [&](const std::string& want) {
    const auto& live = G4SDManager::GetSDMpointer()->Scorers();
    for (std::size_t i = 0; i < live.size() && i < a.last_scores.size(); ++i) {
      if (live[i]->GetName() == want) { return a.last_scores[i]; }
    }
    return 0.0;
  };

  // The same primaries both times.
  //
  // Two runs in one process no longer produce the same events on their own: primaries are
  // generated on the host now, and the RNG stream carries on from where the previous run left
  // it - which is what Geant4 does, and what makes the viewer's "accumulate across runs"
  // reduce variance rather than add the same run to itself. So this comparison has to seed
  // deliberately, exactly as a Geant4 user would with /random/setSeeds.
  //
  // Without it the two runs differ by a per cent or so at 2000 events, which is ordinary
  // statistics and reads exactly like a filter that is doing something.
  constexpr int kEvents = 2000;
  constexpr long kSeed = 20260904;

  a.model.scorers[static_cast<std::size_t>(idx)].custom = true;
  a.scene_dirty = true;
  G4Random::setTheSeed(kSeed);
  RunFromGui(a, kEvents);
  const double with = total_of(name);

  a.model.scorers[static_cast<std::size_t>(idx)].custom = false;
  a.scene_dirty = true;
  G4Random::setTheSeed(kSeed);
  RunFromGui(a, kEvents);
  const double without = total_of(name);

  // Left as it was found, so the saved project still exercises the generated subclass.
  a.model.scorers[static_cast<std::size_t>(idx)].custom = true;
  a.scene_dirty = true;

  if (without <= 0) {
    std::printf("selftest: FAILED - the scorer read zero, so the comparison is vacuous\n");
    return;
  }
  const double dev = std::fabs(with - without) / without;
  if (dev > 1e-12) {
    std::printf("selftest: FAILED - custom scorer reports %.10g MeV, stock reports %.10g MeV\n"
                "  (relative difference %.3g). Flagging a scorer custom must not change the\n"
                "  number it reports until the generated file is edited.\n",
                with, without, dev);
    return;
  }
  std::printf("selftest: a custom scorer reports what the stock one does (%.6g MeV, %.0e)\n",
              with, dev);
}

/// Runs the model at high statistics and logs the result in a form the pipeline can compare
/// against the generated project's.
///
/// This check exists because of a bug it would have caught immediately. The generated
/// PrimaryGeneratorAction sampled a source per event with G4UniformRand - which reads as
/// correct, and is what one would write for Geant4 - but GeneratePrimaries here is called
/// **once per run**, so the random pick chose one source for the entire run. A two-beam model
/// fired every event from whichever beam the first random number landed on, and the saved
/// project silently disagreed with the builder it came from.
///
/// Nothing caught it. The pipeline built the generated project and checked that its dose was
/// not zero, which a one-source run satisfies perfectly well. What was missing was the
/// comparison: the whole point of build_scene.hh is that Run and Save cannot disagree about
/// what a model means, and the *source* was the one part of the model with no check on that.
///
/// The event count is high enough for the comparison to bite. At the selftest's usual 4000 the
/// two beams' contributions overlap within a sigma or two of each other, which is how a run
/// that used one beam instead of two could pass - see RISK.md V1 for the last time a number
/// too small to be a measurement cost a day.
static void SelftestRunForComparison(App& a) {
  // G4GPU_SELFTEST_EVENTS raises the statistics. The comparison against the generated
  // project is between two INDEPENDENT Monte Carlos of the same model, so the two numbers
  // approach each other as 1/sqrt(N) and no faster - which means the only way to tell a
  // real disagreement from an unlucky sample is to run more of them and watch.
  // A million, not the 200000 this used to run.
  //
  // The comparison is between two INDEPENDENT Monte Carlos, so its noise floor is set by
  // the statistics and nothing else. Measured on this model: the builder and the project
  // differ by 6.4% at 200000 histories, 0.80% at a million and -0.08% at four million -
  // textbook 1/sqrt(N). The check they feed has a 6% threshold, and at 200000 the spread
  // between two independent sides is 4.4% on dose1 - so the threshold sat 1.36 sigma out,
  // and every pass it had ever given was luck rather than evidence.
  //
  // A million puts the noise near 1%, so the 6% threshold has real headroom while still
  // catching what it is for - a generated project that fires one of two beams, which is
  // wrong by tens of percent.
  int kEvents = 1000000;
  if (const char* env = std::getenv("G4GPU_SELFTEST_EVENTS")) {
    const int v = std::atoi(env);
    if (v > 0) { kEvents = v; }
  }
  RunFromGui(a, kEvents);
  const auto& scorers = G4SDManager::GetSDMpointer()->Scorers();
  for (std::size_t i = 0; i < a.last_scores.size() && i < scorers.size(); ++i) {
    std::printf("selftest: compare %s %.6f MeV over %d events\n",
                scorers[i]->GetName().c_str(), a.last_scores[i], a.last_events);
  }
  std::fflush(stdout);
}

/// Marches a ray through the scene on the host, printing every boundary it meets.
///
/// The same step_to_boundary the stepper calls, so what it prints is what the transport sees.
/// It exists because "the dose is zero" is the least informative failure a transport can have:
/// it is the same symptom for a beam that misses, a material that does not interact, a scorer
/// on the wrong volume, and a navigator that terminates tracks early. This tells them apart in
/// one line each.
///
/// The Geometry is assembled here from the flattened scene rather than taken from
/// `engine->geometry()`, whose pointers are *device* pointers - reading them on the host is an
/// access violation, which is exactly what the first version of this did.
static void ProbeRay(App& a, g4gpu::Vec3<G4double> from, g4gpu::Vec3<G4double> dir,
                     int max_steps = 40) {
  if (a.run_manager == nullptr) { return; }
  const auto& scene = a.run_manager->GetScene();
  geom::Geometry<G4double> g{};
  g.volumes = scene.volumes.data();
  g.n_volumes = static_cast<int>(scene.volumes.size());
  g.world = scene.world;
  g.store = scene.pool.store();
  g.voxels.material =
      scene.pool.voxel_cells.empty() ? nullptr : scene.pool.voxel_cells.data();
  g.voxels.count = static_cast<int>(scene.pool.voxel_cells.size());
  if (g.n_volumes <= 0) { return; }

  auto name_of = [&](int v) {
    return (v >= 0 && v < static_cast<int>(scene.names.size()))
               ? scene.names[static_cast<std::size_t>(v)].c_str()
               : "(none)";
  };

  int vol = geom::locate(g, from);
  std::printf("probe: from (%.1f, %.1f, %.1f) along (%.2f, %.2f, %.2f)\n", from.x, from.y,
              from.z, dir.x, dir.y, dir.z);
  double travelled = 0;
  for (int i = 0; i < max_steps && vol != geom::kOutsideWorld; ++i) {
    int next = geom::kOutsideWorld;
    const G4double d = geom::step_to_boundary(g, vol, from, dir, next);
    const int mat = geom::material_at(g, vol, from);
    std::printf("  %2d  in %-18s mat %2d  step %10.4f  -> %s\n", i, name_of(vol), mat, d,
                name_of(next));
    if (!(d < geom::kInfinity<G4double>() * 0.5)) {
      std::printf("      infinite step: the ray leaves the world here\n");
      break;
    }
    travelled += d;
    from = from + (d + geom::kPushDistance<G4double>()) * dir;
    vol = geom::resolve_after_step(g, next, from);
  }
  std::printf("  travelled %.3f mm\n", travelled);
  std::fflush(stdout);
}

/// Adds a second scorer over the voxel phantom, scoring every cell.
///
/// The pipeline's only check on per-voxel scoring. Its own scorer rather than turning the
/// existing one per-voxel, so that the volume total the dose check is measured against keeps
/// being produced the same way - and so the two can be compared: a per-voxel scorer's cells
/// must sum to what a whole-volume scorer over the same volume reports, and
/// SelftestCheckVoxelScoring below asserts exactly that.
static void AddSelftestVoxelScorer(App& a) {
  int phantom = -1;
  for (std::size_t i = 0; i < a.model.solids.size(); ++i) {
    if (a.model.solids[i].name == "Phantom") { phantom = static_cast<int>(i); }
  }
  if (phantom < 0) {
    Log("selftest: no Phantom to score per voxel");
    return;
  }
  Scorer sc;
  sc.name = "cells";
  sc.quantity = ScoreQuantity::kEnergyDeposit;
  sc.per_voxel = true;
  // Flagged custom as well, so the pipeline compiles the generated subclass. Two flags on
  // one scorer is also the combination most likely to be got wrong: a custom class deriving
  // from G4PSEnergyDeposit3D rather than from the plain scorer.
  sc.custom = true;
  sc.solids.push_back(phantom);
  a.model.scorers.push_back(sc);
  a.model.Touch();
  a.scene_dirty = true;
  Log("selftest: added a per-voxel scorer over the 8x8x8 phantom");
}

/// Checks that the per-voxel map sums to the volume total the same run reported.
///
/// The one thing that can be checked without a second implementation: the cells and the
/// volume total are accumulated by different code down different paths - a global atomic per
/// cell against a per-event slot - so if the cell attribution dropped a deposit, double
/// counted one, or put one in the wrong volume's range, the two would disagree.
///
/// It also catches the failure that the change was made to avoid: without every_cell stepping,
/// a step crossing several identical cells lands entirely in the first, which still *sums*
/// correctly. So the sum alone is not enough, and the spread is checked too - a map where one
/// cell holds everything is the signature of that bug.
static void SelftestCheckVoxelScoring(App& a) {
  if (a.last_voxel_scores.empty()) {
    std::printf("selftest: FAILED - a per-voxel scorer was set but no cells were scored\n");
    return;
  }
  // The device slot, found by name. Not by model index: the run's scores are indexed by the
  // slot G4SDManager assigned, and while that is now registration order, tying a check to
  // "the model's i-th scorer is the run's i-th number" is exactly the assumption that was
  // wrong before. Matching on the name is right whatever the ordering.
  int slot = -1;
  const auto& live = G4SDManager::GetSDMpointer()->Scorers();
  for (std::size_t i = 0; i < live.size(); ++i) {
    if (live[i]->GetName() == "cells") { slot = static_cast<int>(i); }
  }
  double cell_sum = 0;
  int hit = 0;
  for (double v : a.last_voxel_scores) {
    cell_sum += v;
    if (v > 0) { ++hit; }
  }
  const double whole = (slot >= 0 && slot < static_cast<int>(a.last_scores.size()))
                           ? a.last_scores[static_cast<std::size_t>(slot)]
                           : 0.0;
  const double dev = (whole > 0) ? std::fabs(cell_sum - whole) / whole : 1.0;
  if (dev > 1e-9) {
    std::printf("selftest: FAILED - per-voxel cells sum to %.10g MeV but the volume total is\n"
                "  %.10g MeV (relative difference %.3g)\n", cell_sum, whole, dev);
    return;
  }
  if (hit < 2) {
    std::printf("selftest: FAILED - %d cell(s) took energy; a step crossing the phantom should\n"
                "  deposit in several, which is what stopping at every cell boundary is for\n",
                hit);
    return;
  }
  std::printf("selftest: per-voxel scoring: %d cells hit, summing to the volume total to "
              "%.1e\n", hit, dev);
}

/// Adds the two shapes whose rendering needs the shared solid store: a boolean and a polycone.
///
/// It exists because the renderer used to call the store-free `dist_in` overload, which passes
/// an empty SolidStore - so a boolean operand lookup dereferenced null and a mesh found no
/// triangles. Nothing rendered these shapes in any automated run, so nothing noticed.
static void InsertSelftestStoreSolids(App& a) {
  // By name, not by index: the model's contents depend on which selftest frames ran before
  // this one, and picking index 2 for "the Tubs" is how the mesh ended up as an operand.
  int tubs = -1;
  for (std::size_t i = 1; i < a.model.solids.size(); ++i) {
    if (a.model.solids[i].shape == Shape::kTubs) { tubs = static_cast<int>(i); }
  }
  if (tubs < 0) {
    Log("selftest: no Tubs to subtract - the store-solid step did nothing");
    return;
  }
  const std::size_t before = a.model.solids.size();
  a.sel_solid = tubs;  // InsertBoolean takes the selection as the second operand
  InsertBoolean(a, Shape::kSubtraction);
  if (a.model.solids.size() == before) {
    // InsertBoolean refuses by logging and returning, so a silent no-op here would leave the
    // pipeline checking a model with no boolean in it and still passing.
    Log("selftest: FAILED to insert the subtraction");
    return;
  }

  // A polycone, whose (z, rmin, rmax) sections are the other thing the store carries.
  Solid p;
  p.shape = Shape::kPolycone;
  p.name = "Cone";
  p.p[0] = 0;    // phi start
  p.p[1] = 360;  // phi span
  p.sections = {-30, 0, 10, 0, 0, 25, 30, 0, 5};  // three (z, rmin, rmax) triples
  p.pos[0] = 90;
  p.pos[1] = 0;
  p.pos[2] = 0;
  p.layer = 2;
  p.material = a.model.solids.empty() ? -1 : a.model.solids[1].material;
  p.r = 0.85f;
  p.g = 0.45f;
  p.b = 0.85f;
  a.sel_solid = a.model.AddSolid(p);
  a.scene_dirty = true;
  Log("selftest: added a subtraction and a polycone - the two shapes whose rendering needs");
  Log("  the shared solid store, which the renderer used not to pass");
}

/// Adds water and assigns it to everything but the world.
///
/// Only the selftest uses this, and it exists because the default model is all air: a 6 MeV
/// gamma through 3 cm of air deposits nothing, so the generated project's dose came out as
/// exactly zero and the pipeline's "it ran" check was satisfied by it. A check a zero
/// satisfies is not a check.
static void SelftestUseWater(App& a) {
  Material w;
  w.name = "Water";
  w.nist_name = "G4_WATER";
  w.density = 1.0;
  w.state = MatterState::kLiquid;
  const int idx = a.model.AddMaterial(w);
  for (std::size_t i = 1; i < a.model.solids.size(); ++i) {
    a.model.solids[i].material = idx;
    for (VoxelClass& c : a.model.solids[i].voxel_classes) { c.material = idx; }
  }
  a.scene_dirty = true;
  Log("selftest: added G4_WATER and assigned it to every solid but the world");
}
// ---------------------------------------------------------------- selftest helpers

/// Reproduces the case a user reported: a half-transparent box with a sphere overlapping it.
///
/// Two things are being looked at.
///
///   * **the box has to be see-through.** The walk used to re-find the volume it had just
///     entered - `dist_in` from inside a solid is about zero - so it painted the same front
///     face until the accumulated alpha saturated, and a box at 50% rendered opaque with
///     nothing behind it ever reached.
///   * **the same layer has to be refused.** With the sphere on a higher layer the picture is
///     well defined; with both on the same layer it is not, and the run is refused rather than
///     producing a number that depends on which was added first.
static void InsertSelftestTransparency(App& a) {
  const int mat = (a.model.solids.size() > 1) ? a.model.solids[1].material : -1;

  Solid bx;
  bx.shape = Shape::kBox;
  bx.name = "GlassBox";
  bx.p[0] = bx.p[1] = bx.p[2] = 55;
  bx.pos[0] = 320;
  bx.layer = 2;
  bx.opacity = 0.45f;
  bx.r = 0.45f;
  bx.g = 0.70f;
  bx.b = 0.95f;
  bx.material = mat;
  a.model.AddSolid(bx);

  Solid sp;
  sp.shape = Shape::kSphere;
  sp.name = "InnerBall";
  sp.p[0] = 0;
  sp.p[1] = 42;
  sp.p[3] = 360;
  sp.p[5] = 180;
  sp.pos[0] = 348;
  sp.layer = 4;  // higher: it owns the shared space, and the box's surface is absent there
  sp.opacity = 0.9f;
  sp.r = 0.85f;
  sp.g = 0.80f;
  sp.b = 0.30f;
  sp.material = mat;
  a.model.AddSolid(sp);

  a.scene_dirty = true;
  Log("selftest: added a 45% box with a sphere inside it on a higher layer");
}
/// Proves the same-layer refusal fires, then undoes it.
///
/// Puts the transparent pair on one layer, asks for a run, and checks that no run happened -
/// then restores the layer so the rest of the selftest is unaffected. Without this the
/// refusal is a code path nothing has ever taken: the detector finding an overlap and the run
/// actually stopping are two different claims, and only the second one matters.
static void SelftestCheckOverlapRefusal(App& a) {
  int ball = -1;
  int box = -1;
  for (std::size_t i = 0; i < a.model.solids.size(); ++i) {
    if (a.model.solids[i].name == "InnerBall") { ball = static_cast<int>(i); }
    if (a.model.solids[i].name == "GlassBox") { box = static_cast<int>(i); }
  }
  if (ball < 0 || box < 0) {
    std::printf("FATAL: selftest could not find GlassBox and InnerBall to test the "
                "same-layer refusal.\n");
    std::exit(2);
  }
  const int saved = a.model.solids[static_cast<std::size_t>(ball)].layer;
  a.model.solids[static_cast<std::size_t>(ball)].layer =
      a.model.solids[static_cast<std::size_t>(box)].layer;
  a.model.Touch();
  a.scene_dirty = true;
  RebuildScene(a);

  if (a.overlaps.empty()) {
    std::printf("FATAL: GlassBox and InnerBall were put on the same layer and overlap, and\n"
                "  the detector found nothing. The refusal cannot fire.\n");
    std::exit(2);
  }
  const int events_before = a.last_events;
  RunFromGui(a, 25);
  if (a.last_events != events_before || a.popup != Popup::kOverlap) {
    std::printf("FATAL: a run with a same-layer overlap was NOT refused.\n"
                "  last_events %d -> %d, popup %d\n",
                events_before, a.last_events, static_cast<int>(a.popup));
    std::exit(2);
  }
  Log("selftest: a run with a same-layer overlap was refused, as it should be");

  a.popup = Popup::kNone;
  a.model.solids[static_cast<std::size_t>(ball)].layer = saved;
  a.model.Touch();
  a.scene_dirty = true;
  RebuildScene(a);
  if (!a.overlaps.empty()) {
    std::printf("FATAL: the layer was restored and the overlap did not clear.\n");
    std::exit(2);
  }
}

/// Adds what the rendering changes need in order to be looked at.
///
/// Three things, each of which was implemented and then not exercised by anything:
///
///   * a **Sphere**, whose normal is the analytic quadric gradient. It used to be a central
///     difference of a binary containment test, which can only point in 26 directions, so a
///     sphere rendered as flat facets bounded by the coordinate planes.
///   * a **transparent shell with an opaque core on a higher layer**, which is both halves of
///     the compositing change: the alpha, and the rule that a higher layer wins so the
///     covered part of the shell is not drawn at all.
///   * a **second source with a weight**, so the run splits its events between two beams.
///
/// All of it is placed off to one side, at x = -130, and on layers of its own. The first
/// version made the *existing* Orb transparent and put the core at the origin, which put a
/// high-layer box over the scored mesh - so the mesh owned nothing, scored zero, and the
/// pipeline's "the generated project scored a dose that is not zero" check caught it. A
/// selftest addition that changes the answer the selftest is checking is worse than no
/// addition.
static void InsertSelftestVisuals(App& a) {
  const int mat = (a.model.solids.size() > 1) ? a.model.solids[1].material : -1;

  // The shell: a sphere at a third opacity.
  Solid sp;
  sp.shape = Shape::kSphere;
  sp.name = "Shell";
  sp.p[0] = 0;     // rmin
  sp.p[1] = 60;    // rmax
  sp.p[3] = 360;   // delta phi
  sp.p[5] = 180;   // delta theta
  sp.pos[0] = -130;
  sp.layer = 2;
  sp.opacity = 0.35f;
  sp.r = 0.45f;
  sp.g = 0.75f;
  sp.b = 0.95f;
  sp.material = mat;
  a.model.AddSolid(sp);

  // The core: opaque, inside it, on a higher layer so it wins where they overlap.
  Solid in;
  in.shape = Shape::kBox;
  in.name = "Core";
  in.p[0] = in.p[1] = in.p[2] = 32;
  in.pos[0] = -130;
  in.layer = 5;
  in.r = 0.95f;
  in.g = 0.55f;
  in.b = 0.25f;
  in.material = mat;
  a.model.AddSolid(in);

  // Two beams, so the pipeline exercises the weighted multi-source split rather than the
  // one-source path that a single beam would take.
  //
  // The first one is added here rather than assumed: a new model has no source, on purpose -
  // a default beam is one the user did not choose - so the selftest has to make its own.
  if (a.model.sources.empty()) {
    builder::Source s1;
    s1.name = "Beam";
    s1.pos[2] = -400;
    a.model.AddSource(s1);
  }
  builder::Source s2 = a.model.sources[0];
  s2.name = "SideBeam";
  s2.weight = 0.25;
  s2.pos[0] = -200;
  s2.pos[1] = 0;
  s2.pos[2] = 0;
  s2.dir[0] = 1;
  s2.dir[1] = 0;
  s2.dir[2] = 0;
  a.model.AddSource(s2);
  a.scene_dirty = true;
  Log("selftest: added a sphere, a transparent shell over a higher layer, and a second beam");
}

/// Adds a small voxel volume with two materials in a checkerboard.
///
/// Only the selftest uses this, and it exists so that the pipeline exercises the voxel path
/// end to end: the classifier, the cell sidecar the generated project writes, and the loader
/// that reads it back. A voxel volume that a user has to create by hand in a dialog is a
/// volume no automated check ever sees.
static void InsertSelftestVoxels(App& a) {
  Solid s;
  s.shape = Shape::kVoxelGrid;
  s.name = "Phantom";
  s.p[0] = s.p[1] = s.p[2] = 20;  // 40 mm cube
  const int n = 8;
  s.p[3] = s.p[4] = s.p[5] = n;
  s.pos[0] = 0;
  s.pos[1] = 0;
  s.pos[2] = -60;
  s.material = a.model.solids.empty() ? -1 : a.model.solids[0].material;
  s.layer = 2;
  s.r = 0.55f;
  s.g = 0.75f;
  s.b = 0.55f;
  s.voxel_kind = VoxelKind::kDiscrete;
  s.voxel_values.assign(static_cast<std::size_t>(n) * n * n, 0.0f);
  for (int k = 0; k < n; ++k) {
    for (int j = 0; j < n; ++j) {
      for (int i = 0; i < n; ++i) {
        const std::size_t idx =
            static_cast<std::size_t>(i) + n * (j + static_cast<std::size_t>(n) * k);
        s.voxel_values[idx] = static_cast<float>((i + j + k) & 1);
      }
    }
  }
  // Two discrete classes, one per value, both on the first material: what the classifier
  // would produce for this data, set directly because there is no file to classify.
  s.voxel_classes.clear();
  for (int v = 0; v < 2; ++v) {
    VoxelClass c;
    c.value = v;
    c.value_max = v + 1;
    c.material = s.material;
    c.label = (v == 0) ? "even" : "odd";
    s.voxel_classes.push_back(c);
  }
  a.sel_solid = a.model.AddSolid(s);
  a.scene_dirty = true;
  Log("selftest: added an 8x8x8 voxel volume");
}

/// Writes a small ASCII STL and imports it, so the pipeline exercises the mesh reader, the
/// BVH build, mesh transport, the triangle sidecar and the loader that reads it back.
///
/// A real file rather than triangles pushed straight into the model, because the reader is
/// part of what needs checking: an import path that is only ever driven by hand is an import
/// path that breaks silently.
static void InsertSelftestMesh(App& a) {
  const char* path = "D:/g4gpu/out/selftest_cube.stl";
  {
    FILE* f = std::fopen(path, "w");
    if (f == nullptr) {
      Log("selftest: cannot write the test STL");
      return;
    }
    const double h = 15.0;
    const double v[8][3] = {{-h, -h, -h}, {h, -h, -h}, {h, h, -h}, {-h, h, -h},
                            {-h, -h, h},  {h, -h, h},  {h, h, h},  {-h, h, h}};
    const int t[12][3] = {{0, 1, 2}, {0, 2, 3}, {4, 6, 5}, {4, 7, 6}, {0, 5, 1}, {0, 4, 5},
                          {3, 2, 6}, {3, 6, 7}, {0, 3, 7}, {0, 7, 4}, {1, 5, 6}, {1, 6, 2}};
    std::fprintf(f, "solid selftest_cube\n");
    for (const auto& tt : t) {
      std::fprintf(f, "  facet normal 0 0 0\n    outer loop\n");
      for (int k = 0; k < 3; ++k) {
        std::fprintf(f, "      vertex %.9g %.9g %.9g\n", v[tt[k]][0], v[tt[k]][1],
                     v[tt[k]][2]);
      }
      std::fprintf(f, "    endloop\n  endfacet\n");
    }
    std::fprintf(f, "endsolid selftest_cube\n");
    std::fclose(f);
  }
  if (!ImportFile(a, path)) {
    Log("selftest: mesh import failed");
    return;
  }
  // Place it where it will be hit by the beam, and give it a color of its own.
  Solid& s = a.model.solids[static_cast<std::size_t>(a.sel_solid)];
  s.pos[0] = 0;
  s.pos[1] = 0;
  s.pos[2] = 40;
  s.r = 0.9f;
  s.g = 0.6f;
  s.b = 0.3f;
  a.scene_dirty = true;
}
static void DrawPopup(App& a) {
  switch (a.popup) {
    case Popup::kColor: DrawColorPopup(a); break;
    case Popup::kInsertPrimitive: DrawInsertPopup(a); break;
    case Popup::kImportCad: DrawImportCadPopup(a); break;
    case Popup::kImportVoxel: DrawImportVoxelPopup(a); break;
    case Popup::kFlatDetector: DrawFlatDetectorPopup(a); break;
    case Popup::kPhysics: DrawPhysicsPopup(a); break;
    case Popup::kVisAttributes: DrawVisPopup(a); break;
    case Popup::kPick: DrawPickPopup(a); break;
    case Popup::kOverlap: DrawOverlapPopup(a); break;
    case Popup::kWorld: DrawWorldPopup(a); break;
    default: break;
  }
  // A click in the view while no popup is up selects a solid. Done here, after every panel
  // has had its chance to claim the click, so clicking a list row does not also re-pick.
  if (a.popup == Popup::kNone && a.input.left_pressed && a.uic.hot == 0 && !a.drag.active
      && ViewRect(a).Contains(a.input.mouse_x, a.input.mouse_y)) {
    const int picked = PickSolid(a);
    if (picked != a.sel_solid) {
      a.sel_solid = picked;
      a.pfield_for = -2;
      a.status = (picked >= 0) ? ("selected " + a.model.solids[picked].name)
                               : std::string("nothing selected");
    }
  }
}
