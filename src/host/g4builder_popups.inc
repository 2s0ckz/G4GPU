// The pop-up windows: physics options, visualization attributes, and the primitive picker.

/// A centered modal frame. Returns the content rectangle; the caller draws inside it.
static ui::Rect PopupFrame(App& a, int w, int h, const std::string& title) {
  ui::Context& c = a.uic;
  const ui::Rect r{(a.width - w) / 2, (a.height - h) / 2, w, h};
  // A dim over the whole window, so it reads as modal and a stray click outside does not
  // silently land on a panel behind it.
  c.canvas.FillRect({0, 0, a.width, a.height}, ui::rgba(0, 0, 0, 110));
  ui::Panel(c, r, ui::theme::kPanelAlt);
  c.canvas.Text(r.x + 12, r.y + 10, title, ui::theme::kText);
  c.canvas.HLine(r.x + 8, r.x + r.w - 8, r.y + 10 + a.font.glyph_h + 6, ui::theme::kBorder);
  return {r.x, r.y + 10 + a.font.glyph_h + 14, r.w, r.h - (10 + a.font.glyph_h + 14)};
}

// ---------------------------------------------------------------- length units

/// The length units the dialogs offer, and what one of each is in mm.
///
/// The model is in mm throughout - Geant4's own internal unit - so a unit here is only a
/// factor applied on the way in. Offering the choice matters anyway: a CT resolution is
/// quoted in mm, a phantom in cm and a room in m, and making the user convert by hand is
/// where the factor-of-ten mistakes come from.
static const char* const kLengthUnits[4] = {"um", "mm", "cm", "m"};
static constexpr double kLengthUnitMm[4] = {0.001, 1.0, 10.0, 1000.0};

static double LengthUnitMm(int i) { return (i >= 0 && i < 4) ? kLengthUnitMm[i] : 1.0; }

// ---------------------------------------------------------------- the world

/// The shapes a world may take, and the labels for their dimensions.
///
/// Three, not thirty: the world is the volume everything sits inside and every track leaves
/// through, so what matters is that it is big enough and the right rough form - a box for a
/// room or a bench, a cylinder for a beam line, a sphere for a source in free space. It can
/// be changed to any other shape afterwards in the solid panel, like any solid.
struct WorldChoice {
  const char* label;
  Shape shape;
  const char* dim[3];   ///< nullptr where the shape has no second or third dimension
};
static const WorldChoice kWorldChoices[3] = {
    {"box", Shape::kBox, {"half x", "half y", "half z"}},
    {"cylinder", Shape::kTubs, {"radius", "half z", nullptr}},
    {"sphere", Shape::kOrb, {"radius", nullptr, nullptr}},
};

/// Seeds the New-world dialog. Called when it opens, not when it draws.
static void SeedWorldDialog(App& a) {
  a.world_shape = 0;
  a.world_unit = 2;  // cm: a metre-scale world in mm is four digits, three of them zeroes
  a.world_dim[0].Init(50, "%g");
  a.world_dim[1].Init(50, "%g");
  a.world_dim[2].Init(50, "%g");
  a.world_dialog_seeded = true;
}

/// Asks for the world's shape and size, then builds a model around it.
///
/// It opens at startup and on File > New, before anything else, because the world is the one
/// volume that cannot be added later or left out: every other solid is placed inside it, and
/// a track is followed only until it leaves it. The old behavior - a 1 m box, silently - meant
/// a user building a 3 m room got tracks killed at the wall of a world they never chose, with
/// nothing saying so.
///
/// There is no Cancel, because there is no scene without a world; the defaults are already
/// filled in for a user who does not care which one.
static void DrawWorldPopup(App& a) {
  ui::Context& c = a.uic;
  const int lh = a.font.glyph_h;
  const int row = lh + 6;
  if (!a.world_dialog_seeded) { SeedWorldDialog(a); }

  const ui::Rect r = PopupFrame(a, 470, 9 * row + 96, "New scene: the world volume");
  int y = r.y;
  y += c.canvas.TextWrapped({r.x + 14, y, r.w - 28, 4 * lh},
                            "Everything is placed inside the world, and a track is followed "
                            "only until it leaves it. Make it comfortably larger than the "
                            "detector.",
                            ui::theme::kTextDim);
  y += 8;

  static const char* const kShapeNames[3] = {"box", "cylinder", "sphere"};
  c.canvas.Text(r.x + 14, y + 3, "shape", ui::theme::kTextDim);
  ui::Select(c, 1400, {r.x + 90, y, 150, row}, a.world_shape, kShapeNames, 3);
  y += row + 6;

  const WorldChoice& wc =
      kWorldChoices[(a.world_shape >= 0 && a.world_shape < 3) ? a.world_shape : 0];
  for (int i = 0; i < 3 && wc.dim[i] != nullptr; ++i) {
    c.canvas.Text(r.x + 14, y + 3, wc.dim[i], ui::theme::kTextDim);
    ui::NumberInput(c, 1410 + i, {r.x + 90, y, 100, row}, a.world_dim[i]);
    y += row + 4;
  }
  c.canvas.Text(r.x + 14, y + 3, "units", ui::theme::kTextDim);
  ui::Select(c, 1420, {r.x + 90, y, 100, row}, a.world_unit, kLengthUnits, 4);
  y += row + 12;

  const double u = LengthUnitMm(a.world_unit);
  if (ui::Button(c, 1430, {r.x + 14, y, 150, row + 4}, "Create")) {
    // A fresh model, then its world replaced: Model::Default() also carries the Air the world
    // is filled with, and building that here would be the same code twice.
    a.model = Model::Default();
    a.model_path.clear();
    Solid& w = a.model.solids[0];
    w.shape = wc.shape;
    for (int i = 0; i < 12; ++i) { w.p[i] = 0; }
    if (wc.shape == Shape::kBox) {
      w.p[0] = a.world_dim[0].value * u;
      w.p[1] = a.world_dim[1].value * u;
      w.p[2] = a.world_dim[2].value * u;
    } else if (wc.shape == Shape::kTubs) {
      w.p[1] = a.world_dim[0].value * u;   // rmax; rmin stays 0
      w.p[2] = a.world_dim[1].value * u;   // half z
      w.p[4] = 360;                        // delta phi
    } else {
      w.p[0] = a.world_dim[0].value * u;   // orb radius
    }
    a.sel_solid = 0;
    a.sel_source = -1;
    a.sel_material = -1;
    a.sel_element = -1;
    a.sel_scorer = -1;
    a.pfield_for = -2;
    a.draft_for_element = -2;
    a.draft_for_material = -2;
    a.draft_for_source = -2;
    a.scene_dirty = true;
    a.world_dialog_seeded = false;
    a.popup = Popup::kNone;
    ResetRunState(a);
    Log(Fmt("new scene: %s world, %s %g %s", wc.label, wc.dim[0], a.world_dim[0].value,
            kLengthUnits[a.world_unit & 3]));
    a.status = "world created - insert a solid, then add a source";
  }
  c.canvas.TextFit(r.x + 178, y + 7, "no cancel: a scene needs a world", ui::theme::kTextDim,
                   r.w - 192);
}

// ---------------------------------------------------------------- physics

static void DrawPhysicsPopup(App& a) {
  ui::Context& c = a.uic;
  const int lh = a.font.glyph_h;
  const int row = lh + 6;
  const ui::Rect r = PopupFrame(a, 420, 15 * row + 60, "Physics options");
  int y = r.y;

  Physics& p = a.model.physics;
  struct Toggle {
    const char* label;
    bool* flag;
    const char* note;
  };
  const Toggle toggles[] = {
      {"photoelectric absorption", &p.photoelectric, "Livermore, from G4EMLOW"},
      {"Compton scattering", &p.compton, "Klein-Nishina with the Geant4 fit"},
      {"Rayleigh scattering", &p.rayleigh, "tabulated, all Z <= 100"},
      {"pair production", &p.pair_production, "Bethe-Heitler above 80 GeV"},
      {"bremsstrahlung", &p.bremsstrahlung, "Seltzer-Berger below 1 GeV"},
      {"annihilation", &p.annihilation, "e+ at rest and in flight"},
      {"multiple scattering", &p.multiple_scattering, "Urban, with the real step limiter"},
  };
  for (const Toggle& t : toggles) {
    if (ui::Checkbox(c, 800 + static_cast<int>(&t - toggles), {r.x + 14, y, 250, row},
                     t.label, *t.flag)) {
      a.model.Touch();
      a.scene_dirty = true;
    }
    c.canvas.Text(r.x + 270, y + 3, t.note, ui::theme::kTextDim);
    y += row;
  }
  y += 6;
  // Ionisation has no toggle and it is worth saying why, here, where someone would look for it.
  c.canvas.Text(r.x + 14, y, "Ionisation has no switch: it is the continuous energy loss",
                ui::theme::kTextDim);
  y += lh;
  c.canvas.Text(r.x + 14, y, "along a step, and without it a lepton has infinite range.",
                ui::theme::kTextDim);
  y += lh + 8;

  static ui::NumberField cut;
  static bool cut_init = false;
  if (!cut_init) {
    cut.Init(p.range_cut_mm);
    cut_init = true;
  }
  c.canvas.Text(r.x + 14, y + 3, "range cut", ui::theme::kTextDim);
  if (ui::NumberInput(c, 810, {r.x + 110, y, 90, row}, cut)) {
    p.range_cut_mm = cut.value;
    a.model.Touch();
    a.scene_dirty = true;
  }
  c.canvas.Text(r.x + 208, y + 3, "mm - the secondary production threshold",
                ui::theme::kTextDim);
  y += row + 10;

  if (ui::Button(c, 819, {r.x + r.w - 100, y, 84, row + 2}, "Close")) {
    a.popup = Popup::kNone;
  }
}

// ---------------------------------------------------------------- visualization

/// Three sliders for one color. Returns true when any of them moved.
/// An RGB row: a swatch and three 0-255 fields.
///
/// Numbers, not sliders. A slider cannot be typed into, cannot be read off exactly, and cannot
/// be matched to a value from somewhere else - and matching is most of what colour editing is:
/// making this volume the same blue as that one, entering the RGB a document specifies. The
/// swatch is what a slider was actually providing, and it is still here.
///
/// 0-255 rather than 0-1 because that is how colours are written down everywhere else.
///
/// The fields live in a small cache keyed by @p id, because a NumberField holds the text being
/// typed - "12" on the way to "128" - and rebuilding it from the float every frame would erase
/// each keystroke. Six rows of three is the whole population, so a flat array is the structure.
static bool ColorFields(App& a, int id, const ui::Rect& r, const char* label, float* rgb) {
  ui::Context& c = a.uic;
  const int lh = a.font.glyph_h;
  const int row = lh + 4;

  struct Cache {
    int id = -1;
    ui::NumberField f[3];
    float was[3] = {-1, -1, -1};
  };
  static Cache cache[8];
  Cache* slot = nullptr;
  for (Cache& q : cache) {
    if (q.id == id) { slot = &q; break; }
  }
  if (slot == nullptr) {
    for (Cache& q : cache) {
      if (q.id == -1) { slot = &q; break; }
    }
    if (slot == nullptr) { slot = &cache[0]; }
    slot->id = id;
    for (int k = 0; k < 3; ++k) { slot->was[k] = -1; }
  }
  // Re-seed only when the value changed behind our back - a different solid selected, or
  // Reset to defaults pressed - never merely because it is a new frame.
  for (int k = 0; k < 3; ++k) {
    if (slot->was[k] != rgb[k] && c.focus != id + k) {
      slot->f[k].Init(static_cast<double>(rgb[k]) * 255.0, "%.0f");
      slot->was[k] = rgb[k];
    }
  }

  c.canvas.Text(r.x, r.y + 3, label, ui::theme::kTextDim);
  const ui::Rect sw{r.x + 118, r.y + 2, 26, row - 4};
  c.canvas.FillRect(sw, ui::rgb(static_cast<int>(rgb[0] * 255), static_cast<int>(rgb[1] * 255),
                                static_cast<int>(rgb[2] * 255)));
  c.canvas.StrokeRect(sw, ui::theme::kBorder);

  bool changed = false;
  const char* nm[3] = {"R", "G", "B"};
  for (int k = 0; k < 3; ++k) {
    c.canvas.Text(r.x + 152 + k * 78, r.y + 3, nm[k], ui::theme::kTextDim);
    if (ui::NumberInput(c, id + k, {r.x + 168 + k * 78, r.y + 2, 52, row - 4}, slot->f[k])) {
      double v = slot->f[k].value;
      if (v < 0) { v = 0; }
      if (v > 255) { v = 255; }
      // Written back into the field as well, so a typed 300 becomes 255 in front of you
      // rather than being silently clamped somewhere you cannot see.
      slot->f[k].Init(v, "%.0f");
      rgb[k] = static_cast<float>(v / 255.0);
      slot->was[k] = rgb[k];
      changed = true;
    }
  }
  return changed;
}

static void DrawVisPopup(App& a) {
  ui::Context& c = a.uic;
  const int lh = a.font.glyph_h;
  const int row = lh + 6;
  const ui::Rect r = PopupFrame(a, 520, 13 * row + 52, "Visualization attributes");
  int y = r.y;

  VisAttributes& v = a.vis_attr;
  c.canvas.Text(r.x + 14, y, "These are how you look at the detector, not part of the model,",
                ui::theme::kTextDim);
  y += lh;
  c.canvas.Text(r.x + 14, y, "so they are not saved with it.", ui::theme::kTextDim);
  y += lh + 8;

  ColorFields(a, 830, {r.x + 14, y, r.w - 28, row}, "background", v.bg);
  y += row + 2;
  ColorFields(a, 834, {r.x + 14, y, r.w - 28, row}, "gamma", v.gamma);
  y += row + 2;
  ColorFields(a, 838, {r.x + 14, y, r.w - 28, row}, "electron", v.electron);
  y += row + 2;
  ColorFields(a, 842, {r.x + 14, y, r.w - 28, row}, "positron", v.positron);
  y += row + 2;
  ColorFields(a, 846, {r.x + 14, y, r.w - 28, row}, "other", v.other);
  y += row + 8;

  ui::Checkbox(c, 850, {r.x + 14, y, 160, row}, "solids", v.show_solids);
  ui::Checkbox(c, 851, {r.x + 180, y, 160, row}, "wireframe", v.show_wireframe);
  ui::Checkbox(c, 852, {r.x + 346, y, 160, row}, "axes", v.show_axes);
  y += row + 2;
  ui::Checkbox(c, 853, {r.x + 14, y, 160, row}, "tracks", a.show_tracks);
  ui::Checkbox(c, 854, {r.x + 180, y, 220, row}, "voxel gridlines", v.voxel_grid_lines);
  y += row + 2;
  y += c.canvas.TextWrapped(
      {r.x + 32, y, r.w - 46, 2 * lh},
      "gridlines fade where a cell is finer than the pixels; never dropped",
      ui::theme::kTextDim);
  y += 6;

  if (ui::Button(c, 858, {r.x + 14, y, 150, row + 2}, "Reset to defaults")) {
    a.vis_attr = VisAttributes{};
  }
  if (ui::Button(c, 859, {r.x + r.w - 100, y, 84, row + 2}, "Close")) {
    a.popup = Popup::kNone;
  }
}
