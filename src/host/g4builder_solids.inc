// The solids section, and the scorers section.
//
// Both live at the bottom of a sidebar, and both take the sidebar's width from the rectangle
// they are handed rather than from a constant, so either can be moved to the other side
// without touching the code that draws it.
//
// SOLIDS is one section, not two: a list with the selected item's properties directly under
// it, the way SOURCES is structured. Two separate panels meant looking in one place to pick a
// thing and another to change it.

/// THE LAYER CHOICES, one list shared by every row that offers them.
///
/// A menu rather than a number field: a field needs a per-row edit buffer that survives
/// frames, and there is one buffer for the whole UI because exactly one thing is being edited
/// at a time (see ui::EditBuffer) - so two hundred rows cannot each have one. A Select keeps no
/// state of its own at all; the open one lives in the Context, by id.
///
/// The numbers run from 0 to a little above the highest layer anything in the model uses, which
/// is the range a layer is ever chosen from: you put a volume just above or just below some
/// other volume. Built once per frame rather than per row - every solid has one of these menus
/// now, not only a voxel class, and rebuilding the strings two hundred times to get the same
/// list is work nobody asked for.
///
/// ENTRY 0 IS THE NULL LAYER, and it is first because it is not a bigger or smaller number
/// than the others - it is the answer "this is not in the scene", which does not belong at the
/// far end of a list of forty numbers.
static void RefreshLayerOptions(App& a) {
  int top = 1;
  for (const Solid& o : a.model.solids) {
    if (o.layer > top) { top = o.layer; }
    for (const VoxelClass& vc : o.voxel_classes) {
      if (vc.layer != kInheritLayer && vc.layer > top) { top = vc.layer; }
    }
  }
  top += 2;
  if (top > 40) { top = 40; }
  a.layer_opt_text.clear();
  // THE WORD, not a symbol. ∅ was drawn out of an ellipse and a stroke because the atlas is
  // indexed by byte and a three-byte UTF-8 character cannot reach it - and at eight pixels
  // across it was never going to be more than a smudge that had to be explained. "null" is
  // four characters, which is exactly the width the layer column already reserves for two
  // digits and a menu arrow, and it needs no explaining.
  a.layer_opt_text.push_back("null");
  for (int L = 0; L <= top; ++L) { a.layer_opt_text.push_back(Fmt("%d", L)); }
  a.layer_opt.clear();
  for (const std::string& t : a.layer_opt_text) { a.layer_opt.push_back(t.c_str()); }
}

/// Which entry of that list a layer is, and which layer an entry means. One pair of functions
/// because the list has an entry that is not a number and every caller would otherwise repeat
/// the same off-by-one - and get it wrong once.
static int LayerToOpt(const App& a, int layer) {
  if (layer == kNullLayer) { return 0; }
  const int n = static_cast<int>(a.layer_opt.size());
  return (layer >= 0 && layer + 1 < n) ? layer + 1 : 1;   // 1 is layer 0
}
static int OptToLayer(int opt) { return (opt <= 0) ? kNullLayer : opt - 1; }

static int DrawSolidsList(App& a, const ui::Rect& panel, int y) {
  ui::Context& c = a.uic;
  const int lh = a.font.glyph_h;
  const int row = lh + 5;
  const int w = panel.w;
  RefreshLayerOptions(a);
  // The layer column, reserved on every row: the label, then the menu. Wide enough for the
  // null glyph and for two digits, which is the whole range the list offers.
  const std::string lay_label = "Layer";
  const int lay_label_w = static_cast<int>(lay_label.size()) * a.font.glyph_w + 6;
  const int lay_w = 4 * a.font.glyph_w + 20;
  for (std::size_t i = 0; i < a.model.solids.size(); ++i) {
    Solid& s = a.model.solids[i];
    const std::string mat =
        (s.material >= 0 && s.material < static_cast<int>(a.model.materials.size()))
            ? a.model.materials[s.material].name
            : "(no material)";
    const ui::Color sw =
        ui::rgb(static_cast<int>(s.r * 255), static_cast<int>(s.g * 255),
                static_cast<int>(s.b * 255));
    const int key = 400000 + static_cast<int>(i);
    const bool editing = (a.rename_target == key);
    std::string& buf = a.rename_buf.For(key, s.name);

    // Whether this row can expand: a voxel volume whose values are material indices has a
    // class per distinct value, and each needs a material and a colour. Nothing else has
    // children, and a volume holding physical properties has no classes to show - there is
    // one material per *value* there, not one per category, which is why properties is not
    // implemented rather than shown empty.
    const bool has_children = s.shape == Shape::kVoxelGrid
                              && s.voxel_meaning == VoxelMeaning::kIndices
                              && !s.voxel_classes.empty();

    const ui::Rect rr{panel.x + 8, y, w - 20, row};
    // The disclosure triangle gets its own column at the left, so the swatches below it line
    // up with the swatches of solids that have no children.
    const int arrow_w = 14;
    if (has_children && !editing) {
      const ui::Rect ar{rr.x, rr.y, arrow_w, row};
      const bool over = c.Hovering(ar);
      c.canvas.Text(ar.x + 3, rr.y + (row - lh) / 2, s.ui_expanded ? "-" : "+",
                    over ? ui::theme::kText : ui::theme::kTextDim);
      if (over && a.input.left_pressed) {
        s.ui_expanded = !s.ui_expanded;
        // Not a model change: this is which rows are drawn, not what the detector is. Marking
        // the document dirty for it would put a star in the title bar for opening a list.
        continue;
      }
    }
    // Visibility gets a column of its own, left of the swatch, for the same reason the
    // disclosure triangle does: a hit region carved out of the row would fight the row's own
    // click, and the columns have to line up down a list where only some rows have children.
    //
    // The flag already existed and was reachable only by opening the colour dialog, which is
    // three clicks to answer "what is behind this shell" and three more to put it back.
    const int eye_w = lh + 4;
    if (!editing) {
      const ui::Rect er{rr.x + arrow_w, rr.y, eye_w, row};
      if (ui::EyeToggle(c, kIdSolidEye + static_cast<int>(i), er, s.visible)) {
        s.visible = !s.visible;
        a.model.Touch();
        a.scene_dirty = true;
      }
    }
    // The row stops short of the layer column, so the menu's clicks are the menu's: ListRow
    // claims everything inside its own rectangle. Same arrangement as a class row below.
    const ui::Rect sfull{rr.x + arrow_w + eye_w, rr.y, rr.w - arrow_w - eye_w, row};
    const ui::Rect row_r{sfull.x, sfull.y, sfull.w - lay_w - lay_label_w - 4, row};
    {
      // EVERY SOLID, not only a voxel class. The layer is the one concept a Geant4 user has
      // to learn here - it is what replaces the mother/daughter tree - and it was reachable
      // only by selecting a solid and finding a pair of +/- buttons in the form below. A
      // column in the list is where the question "what is on top of what" is actually asked,
      // because it is the only place the volumes are side by side.
      const ui::Rect slrow{sfull.x + sfull.w - lay_w, sfull.y, lay_w, row};
      c.canvas.Text(slrow.x - lay_label_w, rr.y + (row - lh) / 2, lay_label,
                    ui::theme::kTextDim);
      int ssel = LayerToOpt(a, s.layer);
      if (ui::Select(c, kIdSolidLayer + static_cast<int>(i), slrow, ssel,
                     a.layer_opt.data(), static_cast<int>(a.layer_opt.size()))) {
        SetSolidLayer(a, static_cast<int>(i), OptToLayer(ssel));
      }
    }

    // Name and material in COLUMNS, not one padded string. See ui::ListRow2: `%-12s` looks
    // like a column until a name is longer than twelve characters, and an imported phantom's
    // name is `adult_male_1mm_segmented_raw`.
    const int hit = ui::ListRowRenamable2(c, kIdSolidRow + static_cast<int>(i), row_r,
                                          s.name, mat, a.sel_solid == static_cast<int>(i),
                                          editing, buf, sw);
    if (hit == 1) {
      if (a.sel_solid == static_cast<int>(i)) {
        a.rename_target = key;
        a.rename_buf.Reset();
        a.rename_buf.For(key, s.name);
        c.FocusText(kIdSolidRow + static_cast<int>(i));
      } else {
        a.sel_solid = static_cast<int>(i);
        a.pfield_for = -2;
        a.rename_target = -1;
      }
    } else if (hit == 2) {
      if (!buf.empty()) {
        a.model.solids[i].name = buf;
        a.model.Touch();
        a.scene_dirty = true;
      }
      a.rename_target = -1;
    }
    if (!editing) {
      // The swatch is its own click target: it opens the color picker without selecting.
      // Every solid has one now, not only a voxel class - the colour is the fastest way to
      // tell two boxes apart in the view, and it used to be reachable only by selecting the
      // solid first and finding the Color button.
      const ui::Rect swr = ui::ListRowSwatchRect(c, row_r);
      if (c.Hovering(swr) && a.input.left_pressed) {
        a.popup = Popup::kColor;
        a.popup_target = static_cast<int>(i);
        a.popup_target2 = -1;   // this solid's own colour, not one of its classes
        a.opacity_field.Init(s.opacity * 100.0, "%.0f");   // shown as a percentage
      }
    }
    y += row;

    // A voxel volume expands into one row per class, each with its own material and color -
    // the only way to assign materials to a segmented phantom. Collapsed by default, because
    // a real segmentation has dozens and they would bury every other solid in the scene.
    if (!has_children || !s.ui_expanded) { continue; }

    // NO "INHERITS THE VOLUME'S LAYER" ENTRY, and there should not be one: a class inheriting
    // layer 2 and a class explicitly on layer 2 mean the same thing and cost the same nothing,
    // because build_scene compares the VALUES rather than who typed them. Two entries that
    // read differently for one state is a distinction the user has to learn and cannot use. So
    // a class with no layer of its own shows the volume's number, and picking that number is a
    // no-op.

    for (std::size_t k = 0; k < s.voxel_classes.size(); ++k) {
      const VoxelClass& vc = s.voxel_classes[k];
      const std::string vmat =
          (vc.material >= 0 && vc.material < static_cast<int>(a.model.materials.size()))
              ? a.model.materials[vc.material].name
              : "(assign)";
      const ui::Color vsw = ui::rgb(static_cast<int>(vc.r * 255),
                                    static_cast<int>(vc.g * 255),
                                    static_cast<int>(vc.b * 255));
      const std::string vlabel = (s.voxel_kind == VoxelKind::kDiscrete)
                                     ? Fmt("%g", vc.value)
                                     : Fmt("%.0f..%.0f", vc.value, vc.value_max);
      const int id =
          kIdClassRow + static_cast<int>(i) * kIdClassStride + static_cast<int>(k);
      // A class gets an eye too. It is the same question one level down - what is behind the
      // skin - and on a segmentation it is the only practical way to look inside one, since
      // there may be two hundred classes and the answer is usually "hide all but three".
      const ui::Rect cer{panel.x + 8 + arrow_w + 12, y, eye_w, row};
      if (ui::EyeToggle(c, kIdClassEye + static_cast<int>(i) * kIdClassStride
                               + static_cast<int>(k),
                        cer, vc.visible)) {
        s.voxel_classes[k].visible = !s.voxel_classes[k].visible;
        a.model.Touch();
        a.scene_dirty = true;
      }
      // The row stops short of the layer menu's column, so the menu's clicks are the menu's:
      // ListRow claims everything inside its own rectangle.
      const ui::Rect vfull{panel.x + 8 + arrow_w + 12 + eye_w, y,
                           w - 32 - arrow_w - eye_w, row};
      const ui::Rect vrow{vfull.x, vfull.y, vfull.w - lay_w - lay_label_w - 4, row};
      const ui::Rect lrow{vfull.x + vfull.w - lay_w, vfull.y, lay_w, row};
      c.canvas.Text(lrow.x - lay_label_w, y + (row - lh) / 2, lay_label, ui::theme::kTextDim);

      // ITS LAYER, next to its material, because that is where the question is asked. It was
      // in the colour pop-up behind the swatch first, which is two clicks away from the row
      // it belongs to and behind a control that says "colour".
      //
      // The EFFECTIVE layer: the class's own, or the volume's where it has none. Which is why
      // the menu is only numbers - see the note above the option list.
      const int eff = (vc.layer == kInheritLayer) ? s.layer : vc.layer;
      int lsel = LayerToOpt(a, eff);
      if (ui::Select(c, kIdClassLayer + static_cast<int>(i) * kIdClassStride
                            + static_cast<int>(k),
                     lrow, lsel, a.layer_opt.data(),
                     static_cast<int>(a.layer_opt.size()))) {
        s.voxel_classes[k].layer = OptToLayer(lsel);
        a.model.Touch();
        a.scene_dirty = true;
      }

      // The swatch takes the click, the rest of the row assigns a material. Tested before the
      // row is drawn but acted on after, so the row still draws its hover and its swatch on
      // the frame the swatch is clicked.
      const bool on_swatch = c.Hovering(ui::ListRowSwatchRect(c, vrow));
      if (ui::ListRow2(c, id, vrow, vlabel, vmat, false, vsw, 34)) {
        if (on_swatch) {
          a.popup = Popup::kColor;
          a.popup_target = static_cast<int>(i);
          a.popup_target2 = static_cast<int>(k);
          a.opacity_field.Init(vc.opacity * 100.0, "%.0f");
        } else {
          // A picker here too, so the row does not depend on a selection made elsewhere.
          a.popup = Popup::kPick;
          a.pick_kind = PickKind::kMaterialForVoxelClass;
          a.pick_target = static_cast<int>(i);
          a.pick_target2 = static_cast<int>(k);
        }
      }
      y += row;
    }
  }
  return y + 6;
}

// ---------------------------------------------------------------- the selected solid's form

static int DrawSolidForm(App& a, const ui::Rect& panel, int y) {
  ui::Context& c = a.uic;
  const int lh = a.font.glyph_h;
  const int row = lh + 5;
  const int w = panel.w;

  if (a.sel_solid < 0 || a.sel_solid >= static_cast<int>(a.model.solids.size())) {
    c.canvas.Text(panel.x + 8, y, "click a solid in the view or the list above",
                  ui::theme::kTextDim);
    return y + lh + 10;
  }

  Solid& s = a.model.solids[a.sel_solid];
  // kPendingFields while the insert form is up: it borrows these fields, so the value cached
  // here is the pending primitive's and not this solid's. Any sentinel that is not a valid
  // index re-seeds; naming it is what stops the next reader thinking -3 means "none".
  if (a.pfield_for != a.sel_solid) {
    // Seeded IN THE CHOSEN UNIT. Seeding the base value into a field labelled cm is the one
    // way this arrangement can lie: the number would be right for mm and wrong for what it
    // says beside it, and the first edit would then multiply it again.
    int nps = 0;
    const ShapeParam* sps = ShapeParams(s.shape, nps);
    for (int k = 0; k < 12; ++k) {
      const ui::UnitTable& u = ui::UnitsFor((k < nps) ? sps[k].unit : "");
      a.pfield[k].Init(ui::InUnit(s.p[k], a.p_unit[k], u));
    }
    for (int k = 0; k < 3; ++k) {
      a.pos_field[k].Init(ui::InUnit(s.pos[k], a.pos_unit[k], ui::LengthUnitTable()));
      a.rot_field[k].Init(ui::InUnit(s.rot[k], a.rot_unit[k], ui::AngleUnitTable()));
    }
    a.opacity_field.Init(s.opacity * 100.0, "%.0f");   // shown as a percentage
    a.pfield_for = a.sel_solid;
  }
  c.canvas.HLine(panel.x + 8, panel.x + w - 8, y, ui::theme::kBorder);
  y += 4;
  c.canvas.Text(panel.x + 8, y, Fmt("%s  (%s)", s.name.c_str(), ShapeName(s.shape)),
                ui::theme::kAccent);
  y += lh + 5;

  const int vx = panel.x + 8 + 92;          // where the value fields start
  const int vw = 76;
  // The unit menu takes the space the unit LABEL used to occupy, which is why the fields did
  // not have to move: the label was already telling you the unit, and now it lets you change
  // it. A menu narrower than this clips "mrad".
  const int uw = 56;
  auto unit_rect = [&](int yy) { return ui::Rect{vx + vw + 6, yy, uw, row}; };
  int np = 0;
  const ShapeParam* sp = ShapeParams(s.shape, np);
  for (int k = 0; k < np; ++k) {
    c.canvas.Text(panel.x + 8, y + 3, sp[k].label, ui::theme::kTextDim);
    const ui::UnitTable& u = ui::UnitsFor(sp[k].unit);
    if (u.n <= 0) {
      // A bare number - a segment count, a fraction. The label still says what it is.
      if (ui::NumberInput(c, 600 + k, {vx, y, vw, row}, a.pfield[k])) {
        s.p[k] = a.pfield[k].value;
        a.model.Touch();
        a.scene_dirty = true;
      }
      c.canvas.Text(vx + vw + 6, y + 3, sp[k].unit, ui::theme::kTextDim);
    } else if (ui::UnitField(c, 600 + k, {vx, y, vw, row}, unit_rect(y), a.pfield[k], s.p[k],
                             a.p_unit[k], u)) {
      a.model.Touch();
      a.scene_dirty = true;
    }
    y += row + 1;
  }
  y += 3;

  const char* axis[3] = {"x", "y", "z"};
  for (int k = 0; k < 3; ++k) {
    c.canvas.Text(panel.x + 8, y + 3, Fmt("pos %s", axis[k]), ui::theme::kTextDim);
    if (ui::UnitField(c, 620 + k, {vx, y, vw, row}, unit_rect(y), a.pos_field[k], s.pos[k],
                      a.pos_unit[k], ui::LengthUnitTable())) {
      a.model.Touch();
      a.scene_dirty = true;
    }
    y += row + 1;
  }
  for (int k = 0; k < 3; ++k) {
    c.canvas.Text(panel.x + 8, y + 3, Fmt("rot %s", axis[k]), ui::theme::kTextDim);
    if (ui::UnitField(c, 630 + k, {vx, y, vw, row}, unit_rect(y), a.rot_field[k], s.rot[k],
                      a.rot_unit[k], ui::AngleUnitTable())) {
      a.model.Touch();
      a.scene_dirty = true;
    }
    y += row + 1;
  }
  y += 3;

  // Layer, since it is the one concept a Geant4 user has to learn here.
  //
  // THE VOLUME'S LAYER IS ALSO "ALL ITS CLASSES", which is why this puts every class back to
  // inheriting. A segmentation's classes nearly all want one layer with a handful of
  // exceptions, and the way to say that is to set the volume and then correct the exceptions
  // in their own rows - not to find a second control that means "and all of them too".
  // Logged, because it discards choices that were made one row at a time.
  // THE SAME MENU THE LIST ROW HAS, not a pair of +/- buttons.
  //
  // Two controls for one value is one too many the moment the value stops being an integer
  // you can step: `s.layer - 1` on the null layer is two billion and one below nothing, and a
  // "-" button that stopped at 1 could never reach layer 0 either. The menu offers exactly
  // what a layer may be, which now includes not being in the scene at all.
  c.canvas.Text(panel.x + 8, y + 3, "layer", ui::theme::kTextDim);
  {
    int fsel = LayerToOpt(a, s.layer);
    if (ui::Select(c, 650, {panel.x + 62, y, 4 * a.font.glyph_w + 20, row}, fsel,
                   a.layer_opt.data(), static_cast<int>(a.layer_opt.size()))) {
      SetSolidLayer(a, a.sel_solid, OptToLayer(fsel));
    }
  }
  c.canvas.Text(panel.x + 62 + 4 * a.font.glyph_w + 30, y + 3,
                (s.layer == kNullLayer)
                    ? "not in the scene"
                    : (s.voxel_classes.empty() ? "higher wins"
                                               : "higher wins - and every class follows"),
                ui::theme::kTextDim);
  y += row + 4;


  const int bw3 = (w - 32) / 3;
  if (ui::Button(c, 640, {panel.x + 8, y, bw3, row + 2}, "Center")) {
    s.pos[0] = s.pos[1] = s.pos[2] = 0;
    for (int k = 0; k < 3; ++k) { a.pos_field[k].Init(0); }
    a.model.Touch();
    a.scene_dirty = true;
  }
  if (ui::Button(c, 641, {panel.x + 12 + bw3, y, bw3, row + 2}, "Color")) {
    a.popup = Popup::kColor;
    a.popup_target = a.sel_solid;
    a.popup_target2 = -1;
    a.opacity_field.Init(s.opacity * 100.0, "%.0f");   // shown as a percentage
  }
  if (ui::Button(c, 642, {panel.x + 16 + 2 * bw3, y, bw3, row + 2}, "Delete",
                 a.sel_solid > 0)) {
    DeleteSelectedSolid(a);
    return y + row + 10;
  }
  y += row + 5;

  // Both open a picker. A button that acted on "whatever is selected in that other list" did
  // nothing at all until the other selection had been made, and gave no hint which list it
  // meant.
  const int bw2 = (w - 28) / 2;
  if (ui::Button(c, 410, {panel.x + 8, y, bw2, row + 2}, "Assign material")) {
    a.popup = Popup::kPick;
    a.pick_kind = PickKind::kMaterialForSolid;
    a.pick_target = a.sel_solid;
  }
  if (ui::Button(c, 411, {panel.x + 12 + bw2, y, bw2, row + 2}, "Assign scorer")) {
    a.popup = Popup::kPick;
    a.pick_kind = PickKind::kScorerForSolid;
    a.pick_target = a.sel_solid;
  }
  y += row + 5;

  // The anchor. A volume anchored to another has its position and rotation read in that
  // volume's frame, so moving or turning the anchor carries it - a phantom on a couch, a
  // collimator stack, a gantry. It is a placement chain only: which volume owns a piece of
  // space is still the layer, above, and stays a separate question.
  if (ui::Button(c, 412, {panel.x + 8, y, bw2, row + 2}, "Anchor to", a.sel_solid > 0)) {
    a.popup = Popup::kPick;
    a.pick_kind = PickKind::kAnchorForSolid;
    a.pick_target = a.sel_solid;
  }
  const bool anchored = s.anchor >= 0 && s.anchor < static_cast<int>(a.model.solids.size());
  c.canvas.TextFit(panel.x + 14 + bw2, y + 4,
                   anchored ? a.model.solids[static_cast<std::size_t>(s.anchor)].name
                            : std::string("in world coordinates"),
                   anchored ? ui::theme::kAccent : ui::theme::kTextDim, bw2 - 4);
  y += row + 5;

  // The colour table lives in the import dialog, not here: it is one of the things you decide
  // about a file as you read it, alongside the cell size and the colormap, and a second copy
  // of the control in the solid's panel is a second place to look for it. See
  // DrawImportVoxelPopup.
  //
  // Nor is there a "material for all" button any more. Assign material, above, is that: for a
  // voxel volume it sets every class as well, because "the material of the phantom" is the
  // only thing the volume's own material could sensibly mean when every cell takes its
  // material from a class. Two controls, one of which was a special case of the other, is one
  // control and a thing to explain.
  return y + row + 10;
}

// ---------------------------------------------------------------- scorers

static int DrawScorersSection(App& a, const ui::Rect& panel, int y) {
  ui::Context& c = a.uic;
  const int lh = a.font.glyph_h;
  const int row = lh + 5;
  const int w = panel.w;

  for (std::size_t i = 0; i < a.model.scorers.size(); ++i) {
    const Scorer& sc = a.model.scorers[i];
    const int key = 300000 + static_cast<int>(i);
    const bool editing = (a.rename_target == key);
    std::string& buf = a.rename_buf.For(key, sc.name);
    const std::string label = Fmt("%-12s %-6s %d vol", sc.name.c_str(),
                                  ScoreName(sc.quantity),
                                  static_cast<int>(sc.solids.size()));
    const int hit =
        ui::ListRowRenamable(c, kIdScorerRow + static_cast<int>(i), {panel.x + 8, y, w - 20, row},
                             label, a.sel_scorer == static_cast<int>(i), editing, buf);
    if (hit == 1) {
      if (a.sel_scorer == static_cast<int>(i)) {
        a.rename_target = key;
        a.rename_buf.Reset();
        a.rename_buf.For(key, sc.name);
        c.FocusText(kIdScorerRow + static_cast<int>(i));
      } else {
        a.sel_scorer = static_cast<int>(i);
        a.rename_target = -1;
      }
    } else if (hit == 2) {
      if (!buf.empty()) {
        a.model.scorers[i].name = buf;
        a.model.Touch();
        a.scene_dirty = true;
      }
      a.rename_target = -1;
    }
    y += row;
  }
  if (a.model.scorers.empty()) {
    c.canvas.Text(panel.x + 8, y, "none yet - add one below", ui::theme::kTextDim);
    y += lh + 3;
  }
  y += 4;

  // Which quantity a new scorer measures. Five buttons rather than a dropdown, because the
  // dropdown was the menu that got removed and the five fit.
  static const ScoreQuantity kQ[5] = {ScoreQuantity::kDose, ScoreQuantity::kEnergyDeposit,
                                      ScoreQuantity::kFluence, ScoreQuantity::kTrackLength,
                                      ScoreQuantity::kStepCount};
  static const char* kQLabel[5] = {"dose", "edep", "flux", "len", "count"};
  const int qw = (w - 24) / 5;
  for (int k = 0; k < 5; ++k) {
    if (ui::Button(c, 910 + k, {panel.x + 8 + k * qw, y, qw - 2, row}, kQLabel[k])) {
      AddScorer(a, kQ[k]);
      a.sel_scorer = static_cast<int>(a.model.scorers.size()) - 1;
    }
  }
  y += row + 4;

  if (a.sel_scorer >= 0 && a.sel_scorer < static_cast<int>(a.model.scorers.size())) {
    Scorer& sc = a.model.scorers[a.sel_scorer];
    // Which volumes feed it, each with a way off the list. A scorer attached to nothing reads
    // zero, and the only way to find out used to be to run.
    for (std::size_t k = 0; k < sc.solids.size(); ++k) {
      const int vi = sc.solids[k];
      const std::string nm = (vi >= 0 && vi < static_cast<int>(a.model.solids.size()))
                                 ? a.model.solids[vi].name
                                 : "(deleted)";
      c.canvas.Text(panel.x + 14, y + 3, nm, ui::theme::kText);
      if (ui::Button(c, 930 + static_cast<int>(k), {panel.x + w - 40, y, 28, row}, "x")) {
        sc.solids.erase(sc.solids.begin() + static_cast<long>(k));
        a.model.Touch();
        a.scene_dirty = true;
        break;
      }
      y += row + 1;
    }
    if (sc.solids.empty()) {
      c.canvas.Text(panel.x + 14, y, "no volumes: this scorer will read zero",
                    ui::theme::kWarn);
      y += lh + 3;
    }

    // Per-voxel scoring, offered only when it would do something. It applies to a voxel
    // volume and to energy deposit or dose; a per-cell track length is a different quantity
    // and Geant4 has its own scorer for it.
    bool any_voxel = false;
    for (int vi : sc.solids) {
      if (vi >= 0 && vi < static_cast<int>(a.model.solids.size())
          && a.model.solids[vi].shape == Shape::kVoxelGrid) {
        any_voxel = true;
      }
    }
    const bool can_per_voxel = any_voxel && (sc.quantity == ScoreQuantity::kDose
                                             || sc.quantity == ScoreQuantity::kEnergyDeposit);
    if (can_per_voxel) {
      bool pv = sc.per_voxel;
      if (ui::Checkbox(c, 940, {panel.x + 14, y, w - 30, row}, "score each voxel", pv)) {
        sc.per_voxel = pv;
        a.model.Touch();
        a.scene_dirty = true;
      }
      y += row + 1;
      y += c.canvas.TextWrapped(
          {panel.x + 28, y, w - 44, 3 * lh},
          sc.per_voxel
              ? "G4PSEnergyDeposit3D. Steps end at every cell, so a run is slower; the map is "
                "written beside the project as a .dose file."
              : "one number for the whole volume",
          ui::theme::kTextDim);
      y += 3;
    } else if (sc.per_voxel) {
      // Left set from when it did apply - the volume was removed, or the quantity changed.
      // Said rather than silently ignored, because the scorer still reads "per voxel" in the
      // document and would produce a volume total without explanation.
      y += c.canvas.TextWrapped({panel.x + 14, y, w - 30, 3 * lh},
                                "per-voxel is set but no voxel volume feeds this scorer, so "
                                "it will score the volume as a whole",
                                ui::theme::kWarn);
      y += 3;
    }

    // Written out as its own editable class when the project is saved. A flag about the
    // *code*, not about the run: nothing here changes until the generated file is edited.
    {
      bool cu = sc.custom;
      if (ui::Checkbox(c, 941, {panel.x + 14, y, w - 30, row}, "custom scorer class", cu)) {
        sc.custom = cu;
        a.model.Touch();
      }
      y += row + 1;
      y += c.canvas.TextWrapped(
          {panel.x + 28, y, w - 44, 3 * lh},
          sc.custom ? Fmt("saved as %s.hh/.cc - a subclass with a per-event hook you edit",
                          detail::Ident(sc.name, "Scorer").c_str())
                    : "saved as a stock Geant4 primitive scorer",
          ui::theme::kTextDim);
      y += 4;
    }

    if (ui::Button(c, 920, {panel.x + 8, y, w - 20, row + 3}, "Delete scorer")) {
      DeleteSelectedScorer(a);
    }
    y += row + 6;
  }
  return y + 6;
}
