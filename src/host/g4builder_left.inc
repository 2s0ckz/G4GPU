// The left sidebar: elements and isotopes, then materials.
//
// Both are *forms*: the fields edit a working copy, `Add` appends it and `Update` writes it
// back into the selection. Solids are not done this way - they are dragged in the view and
// live editing is the point - but an element or a material is typed, and typing into a live
// object means a half-configured one exists in the model while you are still deciding. See
// App::draft_element.
//
// The sidebar scrolls. A model with a dozen materials used to lose the rows that fell off the
// bottom, silently.

// Defined in g4builder_solids.inc, which is included after this file because the solids
// section is the longer of the two and reads better next to the scorers it sits beside in the
// other sidebar. A forward declaration rather than a reorder: the left sidebar needs solids
// and the right needs scorers, so one of the two orders has to be broken either way.
static int DrawSolidsSection(App& a, const ui::Rect& panel, int y);
static int DrawScorersSection(App& a, const ui::Rect& panel, int y);

/// The NIST material names, for the autocomplete on the material form.
///
/// Built once from the generated table rather than typed out: there are 300 of them and a
/// hand-written subset would be a list of the ones whoever wrote it happened to use.
static const std::vector<const char*>& NistNames() {
  static std::vector<const char*> v;
  if (v.empty()) {
    v.reserve(static_cast<std::size_t>(g4::nist::kNumNistMaterials));
    for (int i = 0; i < g4::nist::kNumNistMaterials; ++i) {
      v.push_back(g4::nist::kNistMaterials[i].name);
    }
  }
  return v;
}

/// Element symbols by Z, for the autocomplete on the element form.
static const std::vector<const char*>& ElementSymbols() {
  static std::vector<const char*> v;
  if (v.empty()) {
    // The elemental NIST materials are named G4_<symbol>, so the symbols come from the same
    // table as everything else rather than from a second list that could disagree with it.
    for (int i = 0; i < g4::nist::kNumNistMaterials; ++i) {
      const g4::nist::NistMaterial& m = g4::nist::kNistMaterials[i];
      if (m.n_components != 1 || m.components == nullptr) { continue; }
      const char* n = m.name;
      if (std::strncmp(n, "G4_", 3) != 0) { continue; }
      const char* sym = n + 3;
      const std::size_t len = std::strlen(sym);
      if (len < 1 || len > 2) { continue; }
      v.push_back(sym);
    }
  }
  return v;
}

/// Fills in Z and A from an element symbol, if the symbol is one Geant4 knows.
///
/// The point of the autocomplete: typing `Pb` should not then require looking up 82 and
/// 207.2. Returns false for a symbol that is not in the table, leaving the fields alone -
/// which is what lets an *isotope* be entered by symbol and then given its own A.
static bool ElementFromSymbol(const std::string& sym, double& z, double& a) {
  const std::string want = "G4_" + sym;
  for (int i = 0; i < g4::nist::kNumNistMaterials; ++i) {
    const g4::nist::NistMaterial& m = g4::nist::kNistMaterials[i];
    if (m.n_components != 1 || m.components == nullptr) { continue; }
    if (want != m.name) { continue; }
    z = m.components[0].z;
    a = static_cast<double>(data::atomic_mass<double>(m.components[0].z));
    return true;
  }
  return false;
}

/// Seeds the element form from the selection, or from a blank element when nothing is
/// selected.
static void SyncElementDraft(App& a) {
  if (a.draft_for_element == a.sel_element) { return; }
  a.draft_for_element = a.sel_element;
  if (a.sel_element >= 0 && a.sel_element < static_cast<int>(a.model.elements.size())) {
    a.draft_element = a.model.elements[a.sel_element];
  } else {
    a.draft_element = Element{};
  }
  a.elem_z.Init(a.draft_element.z);
  a.elem_a.Init(a.draft_element.a);
  a.elem_sym_buf.Reset();
}

/// The mean excitation energy the *build* arrived at for model material @p i, in eV, or 0.
///
/// Read from the built G4Material rather than recomputed, so the number in the box is the
/// number the transport uses. There is no second implementation to drift: derive it here and
/// it would be right until the day G4IonisParamMat changed.
static double DerivedExcitationEv(const App& a, int i) {
  if (a.detector == nullptr || i < 0) { return 0; }
  const G4Material* built = a.detector->BuiltMaterial(i);
  if (built == nullptr) { return 0; }
  return built->GetEffectiveMeanExcitationEnergy() / eV;
}

static void SyncMaterialDraft(App& a) {
  if (a.draft_for_material == a.sel_material) { return; }
  a.draft_for_material = a.sel_material;
  if (a.sel_material >= 0 && a.sel_material < static_cast<int>(a.model.materials.size())) {
    a.draft_material = a.model.materials[a.sel_material];
  } else {
    a.draft_material = Material{};
    a.draft_material.name = "Material" + std::to_string(a.model.materials.size() + 1);
  }
  a.mat_density.Init(a.draft_material.density);
  a.mat_excitation.Init(a.draft_material.mean_excitation_eV > 0
                            ? a.draft_material.mean_excitation_eV
                            : DerivedExcitationEv(a, a.sel_material));
  for (std::size_t k = 0; k < 8; ++k) {
    a.comp_field[k].Init(k < a.draft_material.components.size()
                             ? a.draft_material.components[k].amount
                             : 0.0);
  }
  a.mat_name_buf.Reset();
  a.mat_nist_buf.Reset();
}

// ---------------------------------------------------------------- elements and isotopes

static int DrawElementsSection(App& a, const ui::Rect& panel, int y) {
  ui::Context& c = a.uic;
  const int lh = a.font.glyph_h;
  const int row = lh + 5;
  SyncElementDraft(a);

  // Isotopes as well as elements: the same record with an A the table would not have given
  // it, which is how Geant4's G4Isotope differs from G4Element too.
  y = ui::SectionHeader(c, panel, y, "ELEMENTS / ISOTOPES");
  for (std::size_t i = 0; i < a.model.elements.size(); ++i) {
    const Element& e = a.model.elements[i];
    const int key = 100000 + static_cast<int>(i);
    const bool editing = (a.rename_target == key);
    std::string& buf = a.rename_buf.For(key, e.symbol);
    const int hit = ui::ListRowRenamable(
        c, 200 + static_cast<int>(i), {panel.x + 8, y, panel.w - 20, row},
        Fmt("%-4s Z=%-4g A=%g", e.symbol.c_str(), e.z, e.a),
        a.sel_element == static_cast<int>(i), editing, buf);
    if (hit == 1) {
      // A second click on an already-selected row starts a rename, which is the gesture a
      // file manager uses and the one a user reaches for without being told.
      if (a.sel_element == static_cast<int>(i)) {
        a.rename_target = key;
        a.rename_buf.Reset();
        a.rename_buf.For(key, e.symbol);
        c.FocusText(200 + static_cast<int>(i));
      } else {
        a.sel_element = static_cast<int>(i);
        a.rename_target = -1;
      }
    } else if (hit == 2) {
      if (!buf.empty()) {
        a.model.elements[i].symbol = buf;
        a.model.elements[i].name = buf;
        a.model.Touch();
        a.scene_dirty = true;
        a.draft_for_element = -2;  // the form has to re-read it
      }
      a.rename_target = -1;
    }
    y += row;
  }
  y += 4;

  // ---- the form
  Element& e = a.draft_element;
  c.canvas.Text(panel.x + 8, y + 3, "sym", ui::theme::kTextDim);
  {
    bool took = false;
    std::string& sym = a.elem_sym_buf.For(a.sel_element, e.symbol);
    ui::TextFieldAuto(c, 214, {panel.x + 46, y, 84, row}, sym, ElementSymbols(), "Pb", 4,
                      &took);
    e.symbol = sym;
    e.name = sym;
    if (took) {
      // Completing the symbol fills in Z and A, which is the whole reason to complete it. For
      // an isotope, type the symbol and then change A.
      double z = e.z, aa = e.a;
      if (ElementFromSymbol(sym, z, aa)) {
        e.z = z;
        e.a = aa;
        a.elem_z.Init(z);
        a.elem_a.Init(aa);
      }
    }
  }
  y += row + 3;
  c.canvas.Text(panel.x + 8, y + 3, "Z", ui::theme::kTextDim);
  if (ui::NumberInput(c, 212, {panel.x + 46, y, 60, row}, a.elem_z)) { e.z = a.elem_z.value; }
  c.canvas.Text(panel.x + 116, y + 3, "A", ui::theme::kTextDim);
  if (ui::NumberInput(c, 213, {panel.x + 138, y, 70, row}, a.elem_a)) { e.a = a.elem_a.value; }
  c.canvas.Text(panel.x + 214, y + 3, "g/mol", ui::theme::kTextDim);
  y += row + 4;

  const int bw = (panel.w - 32) / 3;
  if (ui::Button(c, 210, {panel.x + 8, y, bw, row + 3}, "Add")) {
    a.sel_element = a.model.AddElement(e);
    a.draft_for_element = -2;
    a.scene_dirty = true;
    Log(Fmt("added %s (Z=%g, A=%g)", e.symbol.c_str(), e.z, e.a));
  }
  if (ui::Button(c, 215, {panel.x + 12 + bw, y, bw, row + 3}, "Update",
                 a.sel_element >= 0)) {
    a.model.elements[a.sel_element] = e;
    a.model.Touch();
    a.scene_dirty = true;
    Log(Fmt("%s updated", e.symbol.c_str()));
  }
  if (ui::Button(c, 211, {panel.x + 16 + 2 * bw, y, bw, row + 3}, "Delete",
                 a.sel_element >= 0)) {
    DeleteSelectedElement(a);
  }
  return y + row + 10;
}

// ---------------------------------------------------------------- materials

static int DrawMaterialsSection(App& a, const ui::Rect& panel, int y) {
  ui::Context& c = a.uic;
  const int lh = a.font.glyph_h;
  const int row = lh + 5;
  SyncMaterialDraft(a);

  y = ui::SectionHeader(c, panel, y, "MATERIALS");
  for (std::size_t i = 0; i < a.model.materials.size(); ++i) {
    const Material& m = a.model.materials[i];
    const int key = 200000 + static_cast<int>(i);
    const bool editing = (a.rename_target == key);
    std::string& buf = a.rename_buf.For(key, m.name);
    const std::string label = m.nist_name.empty()
                                  ? Fmt("%-14s %.4g g/cm3", m.name.c_str(), m.density)
                                  : Fmt("%-14s %s", m.name.c_str(), m.nist_name.c_str());
    const int hit =
        ui::ListRowRenamable(c, 300 + static_cast<int>(i), {panel.x + 8, y, panel.w - 20, row},
                             label, a.sel_material == static_cast<int>(i), editing, buf);
    if (hit == 1) {
      if (a.sel_material == static_cast<int>(i)) {
        a.rename_target = key;
        a.rename_buf.Reset();
        a.rename_buf.For(key, m.name);
        c.FocusText(300 + static_cast<int>(i));
      } else {
        a.sel_material = static_cast<int>(i);
        a.rename_target = -1;
      }
    } else if (hit == 2) {
      if (!buf.empty()) {
        a.model.materials[i].name = buf;
        a.model.Touch();
        a.scene_dirty = true;
        a.draft_for_material = -2;
      }
      a.rename_target = -1;
    }
    y += row;
  }
  y += 4;

  // ---- the form
  Material& m = a.draft_material;
  c.canvas.Text(panel.x + 8, y + 3, "name", ui::theme::kTextDim);
  {
    std::string& nm = a.mat_name_buf.For(a.sel_material, m.name);
    ui::TextField(c, 332, {panel.x + 54, y, panel.w - 66, row}, nm, "MyPlastic", 32);
    m.name = nm;
  }
  y += row + 3;
  c.canvas.Text(panel.x + 8, y + 3, "NIST", ui::theme::kTextDim);
  {
    // Completion over the 300 NIST names. Typing `g4_wat` and pressing tab is the difference
    // between this field being useful and being a place to make a typo that silently gives
    // you a material Geant4 has never heard of.
    std::string& nn = a.mat_nist_buf.For(a.sel_material, m.nist_name);
    ui::TextFieldAuto(c, 330, {panel.x + 54, y, panel.w - 66, row}, nn, NistNames(),
                      "G4_WATER, or blank for a mixture", 40);
    m.nist_name = nn;
  }
  y += row + 3;
  if (!m.nist_name.empty()) {
    c.canvas.Text(panel.x + 8, y, "a NIST name supplies the density and composition",
                  ui::theme::kAccent);
    y += lh + 4;
  }

  c.canvas.Text(panel.x + 8, y + 3, "rho", ui::theme::kTextDim);
  if (ui::NumberInput(c, 312, {panel.x + 54, y, 80, row}, a.mat_density)) {
    m.density = a.mat_density.value;
  }
  c.canvas.Text(panel.x + 140, y + 3, "g/cm3", ui::theme::kTextDim);
  y += row + 3;
  // A derived I follows the composition, so the box has to follow the build rather than the
  // selection: adding an element changes the number without changing which material is
  // selected, and SyncMaterialDraft only runs on a change of selection. Not while the field
  // has focus - re-seeding a field someone is typing into is the bug ui::EditBuffer exists to
  // have fixed once already.
  if (m.mean_excitation_eV <= 0 && c.focus != 313) {
    const double derived = DerivedExcitationEv(a, a.sel_material);
    if (derived > 0 && std::fabs(derived - a.mat_excitation.value) > 1e-9 * (derived + 1)) {
      a.mat_excitation.Init(derived);
    }
  }
  c.canvas.Text(panel.x + 8, y + 3, "I", ui::theme::kTextDim);
  if (ui::NumberInput(c, 313, {panel.x + 54, y, 80, row}, a.mat_excitation)) {
    // Only an *edit* writes to the model. The box may be showing the value the build derived
    // - see SyncMaterialDraft - and copying that into the model on every frame would turn a
    // derived I into a fixed one, which would then no longer follow the composition when a
    // component changed. What is shown and what is stored are deliberately different here.
    m.mean_excitation_eV = a.mat_excitation.value;
  }
  c.canvas.TextFit(panel.x + 140, y + 3,
                   m.mean_excitation_eV > 0 ? "eV, 0 = derive" : "eV, derived",
                   m.mean_excitation_eV > 0 ? ui::theme::kTextDim : ui::theme::kAccent,
                   panel.w - 148);
  y += row + 3;

  // Static, because ui::Select holds the pointer until the list is painted at the end of the
  // frame, by which time a local array would be gone.
  static const char* const kStates[3] = {"solid", "liquid", "gas"};
  c.canvas.Text(panel.x + 8, y + 3, "state", ui::theme::kTextDim);
  int state_i = static_cast<int>(m.state);
  if (ui::Select(c, 320, {panel.x + 54, y, panel.w - 70, row}, state_i, kStates, 3)) {
    m.state = static_cast<MatterState>(state_i);
  }
  y += row + 4;

  if (m.nist_name.empty()) {
    // ---- composition
    //
    // The fractions are editable *here*, next to each component, which is the thing that was
    // missing: the old panel listed them as text with a delete button and no way to change
    // the number.
    const bool by_atoms = (m.fraction_kind == Fraction::kAtomCount);
    bool flag = by_atoms;
    if (ui::Checkbox(c, 331, {panel.x + 8, y, panel.w - 20, row},
                     by_atoms ? "by atom count" : "by mass fraction", flag)) {
      m.fraction_kind = flag ? Fraction::kAtomCount : Fraction::kMassFraction;
    }
    y += row + 2;

    double sum = 0;
    for (const Component& comp : m.components) { sum += comp.amount; }
    for (std::size_t k = 0; k < m.components.size() && k < 8; ++k) {
      Component& comp = m.components[k];
      const std::string sym =
          (comp.element >= 0 && comp.element < static_cast<int>(a.model.elements.size()))
              ? a.model.elements[comp.element].symbol
              : "?";
      c.canvas.Text(panel.x + 14, y + 3, Fmt("%-3s", sym.c_str()), ui::theme::kText);
      if (ui::NumberInput(c, 360 + static_cast<int>(k), {panel.x + 48, y, 66, row},
                          a.comp_field[k])) {
        comp.amount = a.comp_field[k].value;
      }
      // The share, so "by mass fraction" means something without arithmetic in the head.
      if (!by_atoms && sum > 0) {
        c.canvas.Text(panel.x + 120, y + 3, Fmt("%5.1f%%", 100 * comp.amount / sum),
                      ui::theme::kTextDim);
      }
      if (ui::Button(c, 340 + static_cast<int>(k), {panel.x + panel.w - 40, y, 28, row}, "x")) {
        m.components.erase(m.components.begin() + static_cast<long>(k));
        for (std::size_t j = 0; j < 8; ++j) {
          a.comp_field[j].Init(j < m.components.size() ? m.components[j].amount : 0.0);
        }
        break;
      }
      y += row + 1;
    }
    if (!by_atoms && !m.components.empty()) {
      const bool ok = std::fabs(sum - 1.0) < 1e-6;
      c.canvas.Text(panel.x + 14, y + 2,
                    ok ? Fmt("mass fractions sum to %.6g", sum)
                       : Fmt("sum %.6g - normalized on build, as Geant4 does", sum),
                    ok ? ui::theme::kOk : ui::theme::kWarn);
      y += lh + 3;
    }
    // A picker, not "the selected one": a button that acted on a selection made in another
    // list did nothing until you had gone and made it, and said nothing about which list.
    if (ui::Button(c, 350, {panel.x + 8, y, panel.w - 20, row},
                   "Add element / isotope", m.components.size() < 8)) {
      a.popup = Popup::kPick;
      a.pick_kind = PickKind::kElementForMaterial;
      a.pick_target = a.sel_material;
    }
    y += row + 3;
  }

  const int bw = (panel.w - 32) / 3;
  if (ui::Button(c, 310, {panel.x + 8, y, bw, row + 3}, "Add")) {
    a.sel_material = a.model.AddMaterial(m);
    a.draft_for_material = -2;
    a.scene_dirty = true;
    Log("added material " + m.name);
  }
  if (ui::Button(c, 315, {panel.x + 12 + bw, y, bw, row + 3}, "Update", a.sel_material >= 0)) {
    a.model.materials[a.sel_material] = m;
    a.model.Touch();
    a.scene_dirty = true;
    Log("material " + m.name + " updated");
  }
  if (ui::Button(c, 311, {panel.x + 16 + 2 * bw, y, bw, row + 3}, "Delete",
                 a.sel_material >= 0)) {
    DeleteSelectedMaterial(a);
  }
  return y + row + 10;
}

// ---------------------------------------------------------------- the sidebar

static void DrawLeftPanel(App& a) {
  ui::Context& c = a.uic;
  const int row = a.font.glyph_h + 5;
  const ui::Rect panel{0, kMenuH, a.left_w, a.height - kMenuH - kStatusH};
  ui::Panel(c, panel);

  // Physics is a button at the very foot, not a section: it is seven checkboxes and a cut
  // that a user sets once and then wants the space back from.
  const ui::Rect br{panel.x + 8, panel.y + panel.h - row - 8, panel.w - 16, row + 3};
  const ui::Rect inner{panel.x, panel.y, panel.w, panel.h - row - 14};

  int y = a.left_scroll.Begin(c, inner, row * 3) + 8;
  y = DrawElementsSection(a, panel, y);
  y = DrawMaterialsSection(a, panel, y);
  y = DrawScorersSection(a, panel, y);
  a.left_scroll.End(c, inner, y);

  if (ui::Button(c, 890, br, "Physics options")) { a.popup = Popup::kPhysics; }
}
