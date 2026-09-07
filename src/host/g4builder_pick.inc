// The picker pop-ups.
//
// "Assign material", "Assign scorer" and "Add element/isotope" used to act on whatever
// happened to be selected in another panel, which meant the button did nothing until you had
// gone and selected the right thing somewhere else - and gave no clue which thing that was.
// Each now opens a list and you pick from it. The button is the whole gesture.


/// A scrollable list of choices in a modal frame. Returns the index picked, or -1.
///
/// `labels` is built by the caller because each kind formats its rows differently - a
/// material shows its density, an element its Z and A - and a picker that showed only names
/// would make you close it and go look.
static int DrawPickList(App& a, const char* title, const std::vector<std::string>& labels,
                        const char* empty_note, ui::ScrollArea& scroll, int id_base) {
  ui::Context& c = a.uic;
  const int lh = a.font.glyph_h;
  const int row = lh + 6;
  const int rows_shown = 12;
  const ui::Rect r = PopupFrame(a, 420, rows_shown * row + 2 * lh + 24, title);
  int picked = -1;

  const ui::Rect list{r.x + 8, r.y, r.w - 16, rows_shown * row};
  int y = scroll.Begin(c, list, row * 3);
  for (std::size_t i = 0; i < labels.size(); ++i) {
    if (ui::ListRow(c, id_base + static_cast<int>(i), {list.x, y, list.w - 6, row}, labels[i],
                    false)) {
      picked = static_cast<int>(i);
    }
    y += row;
  }
  scroll.End(c, list, y);
  if (labels.empty()) {
    c.canvas.Text(list.x + 6, list.y + 4, empty_note, ui::theme::kWarn);
  }

  const int by = r.y + rows_shown * row + 8;
  if (ui::Button(c, id_base + 900, {r.x + r.w - 96, by, 82, row + 2}, "Cancel")) {
    a.popup = Popup::kNone;
  }
  if (picked >= 0) { a.popup = Popup::kNone; }
  return picked;
}

static void DrawPickPopup(App& a) {
  std::vector<std::string> labels;
  switch (a.pick_kind) {
    case PickKind::kMaterialForSolid: {
      for (const Material& m : a.model.materials) {
        labels.push_back(m.nist_name.empty()
                             ? Fmt("%-18s %.4g g/cm3", m.name.c_str(), m.density)
                             : Fmt("%-18s %s", m.name.c_str(), m.nist_name.c_str()));
      }
      const int k = DrawPickList(a, "Assign a material", labels,
                                 "no materials yet - add one on the left", a.pick_scroll, 2000);
      if (k >= 0 && a.pick_target >= 0
          && a.pick_target < static_cast<int>(a.model.solids.size())) {
        a.model.solids[a.pick_target].material = k;
        a.model.Touch();
        a.scene_dirty = true;
        Log(Fmt("%s is now %s", a.model.solids[a.pick_target].name.c_str(),
                a.model.materials[k].name.c_str()));
      }
      break;
    }
    case PickKind::kScorerForSolid: {
      for (const Scorer& sc : a.model.scorers) {
        labels.push_back(Fmt("%-16s %-6s %d volume(s)", sc.name.c_str(),
                             ScoreName(sc.quantity), static_cast<int>(sc.solids.size())));
      }
      const int k = DrawPickList(a, "Attach to a scorer", labels,
                                 "no scorers yet - add one in the SCORERS panel",
                                 a.pick_scroll, 2100);
      if (k >= 0) { AttachSolidToScorer(a, a.pick_target, k); }
      break;
    }
    case PickKind::kElementForMaterial: {
      for (const Element& e : a.model.elements) {
        labels.push_back(Fmt("%-6s Z=%-5g A=%g", e.symbol.c_str(), e.z, e.a));
      }
      const int k = DrawPickList(a, "Add an element or isotope", labels,
                                 "no elements yet - add one at the top left", a.pick_scroll,
                                 2200);
      if (k >= 0 && a.draft_material.components.size() < 8) {
        Component comp;
        comp.element = k;
        comp.amount =
            (a.draft_material.fraction_kind == Fraction::kAtomCount) ? 1.0 : 0.5;
        a.draft_material.components.push_back(comp);
        a.comp_field[a.draft_material.components.size() - 1].Init(comp.amount);
      }
      break;
    }
    case PickKind::kMaterialForVoxelClass: {
      for (const Material& m : a.model.materials) {
        labels.push_back(m.nist_name.empty()
                             ? Fmt("%-18s %.4g g/cm3", m.name.c_str(), m.density)
                             : Fmt("%-18s %s", m.name.c_str(), m.nist_name.c_str()));
      }
      // pick_target2 of -1 means every class, the same sentinel popup_target2 uses for "the
      // solid's own colour rather than one of its classes". The title says which, because a
      // picker that looks identical for one class and for two hundred is one you will use
      // wrongly once.
      const bool all = (a.pick_target2 < 0);
      const int k = DrawPickList(a, all ? "Material for EVERY voxel class"
                                        : "Material for this voxel class",
                                 labels, "no materials yet - add one on the left",
                                 a.pick_scroll, 2300);
      if (k >= 0 && a.pick_target >= 0
          && a.pick_target < static_cast<int>(a.model.solids.size())) {
        Solid& s = a.model.solids[a.pick_target];
        if (all) {
          for (VoxelClass& vc : s.voxel_classes) { vc.material = k; }
          a.model.Touch();
          a.scene_dirty = true;
          Log(Fmt("%s: all %d classes are now %s", s.name.c_str(),
                  static_cast<int>(s.voxel_classes.size()), a.model.materials[k].name.c_str()));
        } else if (a.pick_target2 < static_cast<int>(s.voxel_classes.size())) {
          s.voxel_classes[a.pick_target2].material = k;
          a.model.Touch();
          a.scene_dirty = true;
          Log(Fmt("%s value %g is now %s", s.name.c_str(),
                  s.voxel_classes[a.pick_target2].value, a.model.materials[k].name.c_str()));
        }
      }
      break;
    }
    case PickKind::kAnchorForSolid: {
      // Every solid except the target itself and anything anchored to it, plus a "none" row
      // at the top. The cycle test is what keeps the list honest: anchoring A to B when B is
      // already anchored to A leaves neither with a world position, so those rows are simply
      // not offered rather than offered and refused.
      std::vector<int> index;
      labels.push_back("(none - place in world coordinates)");
      index.push_back(-1);
      for (std::size_t i = 0; i < a.model.solids.size(); ++i) {
        const int k = static_cast<int>(i);
        if (k == a.pick_target) { continue; }
        if (AnchorWouldCycle(a.model, a.pick_target, k)) { continue; }
        const Solid& q = a.model.solids[i];
        labels.push_back(Fmt("%-18s %-10s at (%.0f, %.0f, %.0f)", q.name.c_str(),
                             ShapeName(q.shape), q.pos[0], q.pos[1], q.pos[2]));
        index.push_back(k);
      }
      const int k = DrawPickList(a, "Anchor this volume to", labels,
                                 "nothing to anchor to", a.pick_scroll, 2400);
      if (k >= 0 && a.pick_target >= 0
          && a.pick_target < static_cast<int>(a.model.solids.size())) {
        Solid& s = a.model.solids[a.pick_target];
        const int to = index[static_cast<std::size_t>(k)];

        // The position is rewritten so the volume does not move. It is stored relative to the
        // anchor from now on, and leaving the old world numbers in place would teleport it
        // the moment the anchor was chosen - which is a surprising thing for "anchor to" to
        // do, and would have to be undone by hand.
        double old_inv[9], old_pos[3];
        SolidWorldPlacement(a.model, a.pick_target, old_inv, old_pos);
        s.anchor = to;
        if (to >= 0) {
          double a_inv[9], a_pos[3];
          SolidWorldPlacement(a.model, to, a_inv, a_pos);
          const double d[3] = {old_pos[0] - a_pos[0], old_pos[1] - a_pos[1],
                               old_pos[2] - a_pos[2]};
          // a_inv is the world -> anchor map, so it takes a world offset into the anchor's
          // frame directly.
          for (int r = 0; r < 3; ++r) {
            s.pos[r] = a_inv[3 * r + 0] * d[0] + a_inv[3 * r + 1] * d[1]
                       + a_inv[3 * r + 2] * d[2];
          }
          Log(Fmt("%s is anchored to %s; its position is now read in that frame",
                  s.name.c_str(), a.model.solids[static_cast<std::size_t>(to)].name.c_str()));
        } else {
          for (int r = 0; r < 3; ++r) { s.pos[r] = old_pos[r]; }
          Log(Fmt("%s is no longer anchored; its position is in world coordinates",
                  s.name.c_str()));
        }
        a.model.Touch();
        a.pfield_for = -2;
        a.scene_dirty = true;
      }
      break;
    }
    case PickKind::kNone:
      a.popup = Popup::kNone;
      break;
  }
}
