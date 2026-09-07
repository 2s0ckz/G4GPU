// The right sidebar: sources, then scorers and the physics button.

/// Seeds the source form from the selection, or from a blank source.
static void SyncSourceDraft(App& a) {
  if (a.draft_for_source == a.sel_source) { return; }
  a.draft_for_source = a.sel_source;
  if (a.sel_source >= 0 && a.sel_source < static_cast<int>(a.model.sources.size())) {
    a.draft_source = a.model.sources[a.sel_source];
  } else {
    a.draft_source = builder::Source{};
    a.draft_source.name = "Source" + std::to_string(a.model.sources.size() + 1);
  }
  const builder::Source& s = a.draft_source;
  a.src_field[0].Init(s.energy_MeV);
  a.src_field[1].Init(s.pos[0]);
  a.src_field[2].Init(s.pos[1]);
  a.src_field[3].Init(s.pos[2]);
  a.src_field[4].Init(s.half_x);
  a.src_field[5].Init(s.half_y);
  a.src_field[6].Init(s.radius);
  a.src_field[7].Init(s.angular_spread_deg);
  a.src_activity.Init(s.activity);
  a.src_weight.Init(s.weight);
  a.src_name_buf.Reset();
  a.src_particle_buf.Reset();
  a.src_nuclide_buf.Reset();
}

/// The particle names the transport will accept, for the source form's autocomplete.
///
/// The list is short and it is short for a reason: this is what the engine carries. Anything
/// else is refused at /run/beamOn rather than transported as something it is not, and
/// completing from the real list is how a user finds that out before pressing Run instead of
/// after. See G4RunManager::BeamOn and docs/ROADMAP.md.
static const std::vector<const char*>& ParticleNames() {
  static const std::vector<const char*> v = {"gamma", "e-", "e+"};
  return v;
}

// ---------------------------------------------------------------- sources

static int DrawSourcesSection(App& a, const ui::Rect& panel, int y) {
  ui::Context& c = a.uic;
  const int lh = a.font.glyph_h;
  const int row = lh + 5;
  const int w = panel.w;
  SyncSourceDraft(a);

  y = ui::SectionHeader(c, panel, y, "SOURCES");
  static const char* kKindName[5] = {"beam", "point", "shell", "volume", "radio"};

  // The share each enabled source contributes, so a two-beam run is legible before it is run.
  double wsum = 0;
  for (const builder::Source& s : a.model.sources) {
    if (s.enabled && s.weight > 0) { wsum += s.weight; }
  }
  for (std::size_t i = 0; i < a.model.sources.size(); ++i) {
    const builder::Source& s = a.model.sources[i];
    const int key = 500000 + static_cast<int>(i);
    const bool editing = (a.rename_target == key);
    std::string& buf = a.rename_buf.For(key, s.name);
    const std::string share =
        (s.enabled && wsum > 0) ? Fmt("%3.0f%%", 100 * s.weight / wsum) : " off";
    const std::string label = Fmt("%s %-9s %-6s %s", share.c_str(), s.name.c_str(),
                                  kKindName[static_cast<int>(s.kind)], s.particle.c_str());
    const int hit =
        ui::ListRowRenamable(c, 690 + static_cast<int>(i), {panel.x + 8, y, w - 20, row},
                             label, a.sel_source == static_cast<int>(i), editing, buf);
    if (hit == 1) {
      if (a.sel_source == static_cast<int>(i)) {
        a.rename_target = key;
        a.rename_buf.Reset();
        a.rename_buf.For(key, s.name);
        c.FocusText(690 + static_cast<int>(i));
      } else {
        a.sel_source = static_cast<int>(i);
        a.draft_for_source = -2;
        a.rename_target = -1;
      }
    } else if (hit == 2) {
      if (!buf.empty()) {
        a.model.sources[i].name = buf;
        a.model.Touch();
        a.draft_for_source = -2;
      }
      a.rename_target = -1;
    }
    y += row;
  }
  if (a.model.sources.empty()) {
    c.canvas.Text(panel.x + 8, y, "none - fill in the form and press Add",
                  ui::theme::kTextDim);
    y += lh + 3;
  }
  y += 4;

  const int sbw = (w - 36) / 4;
  if (ui::Button(c, 680, {panel.x + 8, y, sbw, row + 2}, "Add")) {
    a.sel_source = a.model.AddSource(a.draft_source);
    a.draft_for_source = -2;
    Log("added source " + a.draft_source.name);
  }
  const bool have = a.sel_source >= 0
                    && a.sel_source < static_cast<int>(a.model.sources.size());
  if (ui::Button(c, 683, {panel.x + 12 + sbw, y, sbw, row + 2}, "Update", have)) {
    a.model.sources[a.sel_source] = a.draft_source;
    a.model.Touch();
    Log("source " + a.draft_source.name + " updated");
  }
  if (ui::Button(c, 681, {panel.x + 16 + 2 * sbw, y, sbw, row + 2}, "Delete", have)) {
    DeleteSelectedSource(a);
  }
  if (have) {
    builder::Source& s0 = a.model.sources[a.sel_source];
    if (ui::Button(c, 682, {panel.x + 20 + 3 * sbw, y, sbw, row + 2},
                   s0.enabled ? "Off" : "On")) {
      s0.enabled = !s0.enabled;
      a.model.Touch();
      a.draft_for_source = -2;
    }
  }
  y += row + 6;

  // ---- the form, editing the working copy
  builder::Source& src = a.draft_source;
  const char* kinds[5] = {"beam", "point", "shell", "volume", "radio"};
  const int kw = (w - 24) / 5;
  for (int k = 0; k < 5; ++k) {
    if (ui::Button(c, 700 + k, {panel.x + 8 + k * kw, y, kw - 2, row}, kinds[k])) {
      src.kind = static_cast<SourceKind>(k);
    }
    if (static_cast<int>(src.kind) == k) {
      c.canvas.StrokeRect({panel.x + 8 + k * kw, y, kw - 2, row}, ui::theme::kAccent);
    }
  }
  y += row + 4;

  const int vx = panel.x + 8 + 78;
  const int vw = 88;
  c.canvas.Text(panel.x + 8, y + 3, "name", ui::theme::kTextDim);
  {
    std::string& nm = a.src_name_buf.For(a.sel_source, src.name);
    ui::TextField(c, 709, {vx, y, w - 96, row}, nm, "Source", 24);
    src.name = nm;
  }
  y += row + 2;
  c.canvas.Text(panel.x + 8, y + 3, "particle", ui::theme::kTextDim);
  {
    std::string& pn = a.src_particle_buf.For(a.sel_source, src.particle);
    ui::TextFieldAuto(c, 710, {vx, y, w - 96, row}, pn, ParticleNames(), "gamma", 16);
    src.particle = pn;
  }
  y += row + 2;
  // Said here rather than at /run/beamOn, because finding out after pressing Run is finding
  // out too late. The engine carries gamma, e- and e+; anything else is refused.
  {
    bool known = false;
    for (const char* p : ParticleNames()) {
      if (src.particle == p) { known = true; }
    }
    if (!known && !src.particle.empty()) {
      c.canvas.Text(panel.x + 8, y, Fmt("\"%s\" is not transported yet", src.particle.c_str()),
                    ui::theme::kWarn);
      y += lh;
      c.canvas.Text(panel.x + 8, y, "gamma, e- and e+ only; a run would be refused",
                    ui::theme::kWarn);
      y += lh + 2;
    }
  }
  c.canvas.Text(panel.x + 8, y + 3, "energy", ui::theme::kTextDim);
  if (ui::NumberInput(c, 711, {vx, y, vw, row}, a.src_field[0])) {
    src.energy_MeV = a.src_field[0].value;
  }
  c.canvas.Text(vx + vw + 6, y + 3, "MeV", ui::theme::kTextDim);
  y += row + 2;
  c.canvas.Text(panel.x + 8, y + 3, "weight", ui::theme::kTextDim);
  if (ui::NumberInput(c, 712, {vx, y, vw, row}, a.src_weight)) {
    src.weight = a.src_weight.value;
  }
  c.canvas.Text(vx + vw + 6, y + 3, "share", ui::theme::kTextDim);
  y += row + 2;

  const char* pl[3] = {"pos x", "pos y", "pos z"};
  for (int k = 0; k < 3; ++k) {
    c.canvas.Text(panel.x + 8, y + 3, pl[k], ui::theme::kTextDim);
    if (ui::NumberInput(c, 720 + k, {vx, y, vw, row}, a.src_field[1 + k])) {
      src.pos[k] = a.src_field[1 + k].value;
    }
    c.canvas.Text(vx + vw + 6, y + 3, "mm", ui::theme::kTextDim);
    y += row + 1;
  }

  if (src.kind == SourceKind::kBeam || src.kind == SourceKind::kVolume) {
    c.canvas.Text(panel.x + 8, y + 3, "half x", ui::theme::kTextDim);
    if (ui::NumberInput(c, 730, {vx, y, vw, row}, a.src_field[4])) {
      src.half_x = a.src_field[4].value;
    }
    y += row + 1;
    c.canvas.Text(panel.x + 8, y + 3, "half y", ui::theme::kTextDim);
    if (ui::NumberInput(c, 731, {vx, y, vw, row}, a.src_field[5])) {
      src.half_y = a.src_field[5].value;
    }
    y += row + 2;
    if (src.kind == SourceKind::kBeam) {
      bool ell = (src.profile == BeamProfile::kElliptical);
      if (ui::Checkbox(c, 732, {panel.x + 8, y, w - 20, row}, "elliptical", ell)) {
        src.profile = ell ? BeamProfile::kElliptical : BeamProfile::kRectangular;
      }
      y += row + 2;
    }
  }
  if (src.kind == SourceKind::kPoint) {
    c.canvas.Text(panel.x + 8, y + 3, "spread", ui::theme::kTextDim);
    if (ui::NumberInput(c, 740, {vx, y, vw, row}, a.src_field[7])) {
      src.angular_spread_deg = a.src_field[7].value;
    }
    c.canvas.Text(vx + vw + 6, y + 3, "deg", ui::theme::kTextDim);
    y += row + 2;
  }
  if (src.kind == SourceKind::kIsotropicShell) {
    c.canvas.Text(panel.x + 8, y + 3, "radius", ui::theme::kTextDim);
    if (ui::NumberInput(c, 741, {vx, y, vw, row}, a.src_field[6])) {
      src.radius = a.src_field[6].value;
    }
    c.canvas.Text(vx + vw + 6, y + 3, "mm", ui::theme::kTextDim);
    y += row + 2;
  }
  if (src.kind == SourceKind::kRadioactive) {
    c.canvas.Text(panel.x + 8, y + 3, "nuclide", ui::theme::kTextDim);
    {
      std::string& nu = a.src_nuclide_buf.For(a.sel_source, src.nuclide);
      ui::TextField(c, 742, {vx, y, vw, row}, nu, "Cs137", 12);
      src.nuclide = nu;
    }
    y += row + 2;
    c.canvas.Text(panel.x + 8, y + 3, "activity", ui::theme::kTextDim);
    if (ui::NumberInput(c, 743, {vx, y, vw - 40, row}, a.src_activity)) {
      src.activity = a.src_activity.value;
    }
    if (ui::Button(c, 744, {vx + vw - 34, y, 40, row}, src.activity_in_curie ? "Ci" : "Bq")) {
      src.activity_in_curie = !src.activity_in_curie;
    }
    y += row + 4;
    c.canvas.Text(panel.x + 8, y, "decay sampling is not implemented:", ui::theme::kWarn);
    y += lh;
    c.canvas.Text(panel.x + 8, y, "emitted isotropically at the energy above",
                  ui::theme::kWarn);
    y += lh + 2;
  }
  if (ui::Button(c, 750, {panel.x + 8, y, w - 20, row}, "Import spectrum CSV")) {
    const std::string p = FileDialog(a, "CSV\0*.csv\0All files\0*.*\0", false, "csv");
    if (!p.empty()) {
      src.spectrum_file = p;
      Log("energy spectrum: " + p);
    }
  }
  y += row + 3;
  if (!src.spectrum_file.empty()) {
    c.canvas.Text(panel.x + 8, y, "spectrum overrides the energy", ui::theme::kAccent);
    y += lh + 3;
  }
  return y + 8;
}

// ---------------------------------------------------------------- the sidebar

static void DrawRightPanel(App& a) {
  ui::Context& c = a.uic;
  const int row = a.font.glyph_h + 5;
  const ui::Rect panel{a.width - a.right_w, kMenuH, a.right_w,
                       a.height - kMenuH - kStatusH};
  ui::Panel(c, panel);

  // A scroll box per section rather than one for the panel. See ui::DivideSections.
  const int want[2] = {a.solids_scroll.content_h, a.sources_scroll.content_h};
  int h[2] = {0, 0};
  ui::DivideSections(panel.h - 8, row * 5, want, h, 2);

  const ui::Rect r_solids{panel.x, panel.y + 4, panel.w, h[0]};
  int y = a.solids_scroll.Begin(c, r_solids, row * 3) + 8;
  y = DrawSolidsSection(a, panel, y);
  a.solids_scroll.End(c, r_solids, y);

  const ui::Rect r_sources{panel.x, r_solids.y + h[0], panel.w, h[1]};
  int y2 = a.sources_scroll.Begin(c, r_sources, row * 3) + 4;
  y2 = DrawSourcesSection(a, panel, y2);
  a.sources_scroll.End(c, r_sources, y2);
}

// ---------------------------------------------------------------- the bottom panel

/// Terminal output above the run control, spanning between the two sidebars.
///
/// The log used to be one right-aligned line in the status bar, which meant that anything the
/// builder said scrolled past unread. It is the same text that goes to stdout, so a run's
/// numbers, a refusal and its reason, and the note about an overlapped scoring volume all end
/// up somewhere they can be read and scrolled back through.
///
/// The wheel scrolls it and `log_scroll < 0` means "follow the tail", which is what a terminal
/// does: new output appears without moving the view away from where the user was reading, and
/// scrolling back to the bottom resumes following.
static void DrawBottomPanel(App& a) {
  ui::Context& c = a.uic;
  const int lh = a.font.glyph_h;
  const int row = lh + 5;
  const ui::Rect panel = BottomRect(a);
  ui::Panel(c, panel);

  const int run_h = row + 10;
  const ui::Rect log_r{panel.x + 6, panel.y + 6, panel.w - 12, panel.h - run_h - 14};
  const ui::Rect run_r{panel.x + 6, log_r.y + log_r.h + 5, panel.w - 12, run_h};

  // ---- terminal output
  ui::Panel(c, log_r, ui::theme::kField);
  {
    const int visible = (log_r.h - 6) / lh;
    const int n = static_cast<int>(a.log.size());
    if (c.Hovering(log_r) && a.input.wheel != 0) {
      const int at = (a.log_scroll < 0) ? n - visible : a.log_scroll;
      int next = at - a.input.wheel * 3;
      const int max_first = (n > visible) ? n - visible : 0;
      if (next >= max_first) {
        next = -1;  // back at the bottom: follow the tail again
      } else if (next < 0) {
        next = 0;
      }
      a.log_scroll = next;
      a.input.wheel = 0;  // consumed: the camera must not also zoom
    }
    int first = (a.log_scroll < 0) ? n - visible : a.log_scroll;
    if (first < 0) { first = 0; }
    const ui::Rect saved = c.canvas.clip;
    c.canvas.clip = log_r;
    int ly = log_r.y + 3;
    for (int i = first; i < n && ly + lh <= log_r.y + log_r.h; ++i) {
      // A line that starts with FATAL or WARNING is colored, because those are the lines a
      // user is looking for when they look here at all.
      const std::string& s = a.log[static_cast<std::size_t>(i)];
      ui::Color col = ui::theme::kText;
      if (s.rfind("FATAL", 0) == 0) {
        col = ui::theme::kError;
      } else if (s.rfind("WARNING", 0) == 0 || s.rfind("warning", 0) == 0) {
        col = ui::theme::kWarn;
      } else if (s.rfind("  ", 0) == 0) {
        col = ui::theme::kTextDim;
      }
      c.canvas.Text(log_r.x + 5, ly, s, col);
      ly += lh;
    }
    c.canvas.clip = saved;
    // A scroll position and a thumb, so "there is more above" is visible rather than guessed.
    if (n > visible) {
      const int thumb_h = (log_r.h * visible) / n;
      const int max_first = n - visible;
      const int thumb_y =
          log_r.y + ((log_r.h - thumb_h) * ((a.log_scroll < 0) ? max_first : a.log_scroll))
                        / (max_first > 0 ? max_first : 1);
      c.canvas.FillRect({log_r.x + log_r.w - 4, log_r.y, 3, log_r.h}, ui::theme::kBorder);
      c.canvas.FillRect({log_r.x + log_r.w - 4, thumb_y, 3, thumb_h > 10 ? thumb_h : 10},
                        ui::theme::kAccent);
    }
    if (a.log_scroll >= 0) {
      c.canvas.TextRight(log_r.x + log_r.w - 12, log_r.y + 3,
                         Fmt("line %d of %d - wheel down to follow", first + 1, n),
                         ui::theme::kAccent);
    }
  }

  // ---- run
  {
    int x = run_r.x + 4;
    c.canvas.Text(x, run_r.y + 5, "events", ui::theme::kTextDim);
    x += 58;
    ui::NumberInput(c, 500, {x, run_r.y + 2, 96, row}, a.events_field, "1000");
    x += 104;
    if (ui::Button(c, 501, {x, run_r.y, 82, row + 4}, "Run", a.overlaps.empty())) {
      RunFromGui(a, static_cast<int>(a.events_field.value));
    }
    x += 90;
    if (ui::Button(c, 502, {x, run_r.y, 104, row + 4}, "Save project")) {
      SaveProjectFromGui(a);
    }
    x += 112;
    if (ui::Button(c, 503, {x, run_r.y, 82, row + 4}, "Clear log")) {
      a.log.clear();
      a.log_scroll = -1;
    }
    x += 90;
    // "Reset", not "Clear tracks": it drops the trajectories *and* the last run's numbers,
    // and a button that named only half of what it did was the half people were surprised by.
    if (ui::Button(c, 505, {x, run_r.y, 82, row + 4}, "Reset",
                   a.n_segments > 0 || a.last_events > 0)) {
      ResetRunState(a);
    }
    x += 90;

    // How many events are *drawn*, which is not how many are run.
    //
    // A run of a million events cannot draw a million trajectories - the buffer would be tens
    // of gigabytes and the picture would be a solid block - so only the first few events' are
    // kept. That cutoff used to be a constant, so a run that showed less than you wanted gave
    // no way to ask for more, and one that showed a wall gave no way to ask for less.
    c.canvas.Text(x, run_r.y + 5, "draw", ui::theme::kTextDim);
    x += 40;
    if (ui::NumberInput(c, 506, {x, run_r.y + 2, 56, row}, a.traj_events_field)) {
      int v = static_cast<int>(a.traj_events_field.value + 0.5);
      if (v < 0) { v = 0; }
      a.traj_max_events = v;
      a.traj_events_field.Init(v, "%.0f");
    }
    x += 62;
    c.canvas.Text(x, run_r.y + 5, "tracks", ui::theme::kTextDim);
    x += 56;

    if (!a.overlaps.empty()) {
      // Said where the Run button is, greyed out beside it, rather than only when it is
      // pressed: a disabled button with no reason next to it is a worse experience than an
      // error message.
      if (ui::Button(c, 504, {x, run_r.y, 200, row + 4}, "Overlap on the same layer")) {
        a.popup = Popup::kOverlap;
      }
      c.canvas.StrokeRect({x, run_r.y, 200, row + 4}, ui::theme::kWarn);
    } else if (a.last_events > 0) {
      std::string line = Fmt("%d events, %.1f ms", a.last_events, a.last_ms);
      const auto& scorers = G4SDManager::GetSDMpointer()->Scorers();
      for (std::size_t i = 0; i < a.last_scores.size() && i < scorers.size(); ++i) {
        const double mass = a.run_manager->ScoredMass(static_cast<int>(i));
        line += (mass > 0)
                    ? Fmt("   %s %.5g pGy", scorers[i]->GetName().c_str(),
                          a.last_scores[i] * MeV / joule / mass * gray / picogray)
                    : Fmt("   %s %.6g MeV", scorers[i]->GetName().c_str(), a.last_scores[i]);
      }
      // Bounded by what is left of the row. With several scorers this line is long, and it
      // used to run off the end of the panel and into the right sidebar's text - two readouts
      // overlapping, both illegible. The log has the full version; this is the glance.
      const int left = run_r.x + run_r.w - 6 - x;
      c.canvas.TextFit(x, run_r.y + 5, line, ui::theme::kOk, left);
    }
  }
}
