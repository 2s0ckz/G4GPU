// Deleting things, in one place.
//
// One function per kind, because each has to do more than erase a vector entry: an element is
// referenced by material components, a material by solids and voxel classes, a solid by
// scorers and by boolean operands. `Model::Remove*` already fixes up the indices; what these
// add is the selection, the edit buffers and the log line - and being callable from both a
// button and the Delete key, which is why they are functions and not the body of a button.

/// The Delete key deletes whatever is selected, in this order.
///
/// A single key that means "delete the selected thing" needs an order, because more than one
/// thing can be selected at once - a solid and a material, say. Solids first: they are what
/// the key is reached for, being the thing selected by clicking in the view.
enum class DeleteWhat { kNone, kSolid, kSource, kScorer, kMaterial, kElement };

static void DeleteSelectedSolid(App& a) {
  if (a.sel_solid <= 0 || a.sel_solid >= static_cast<int>(a.model.solids.size())) { return; }
  Log("deleted " + a.model.solids[a.sel_solid].name);
  a.model.RemoveSolid(a.sel_solid);
  a.sel_solid = -1;
  a.pfield_for = -2;
  a.rename_target = -1;
  a.scene_dirty = true;
}

static void DeleteSelectedElement(App& a) {
  if (a.sel_element < 0 || a.sel_element >= static_cast<int>(a.model.elements.size())) {
    return;
  }
  Log("deleted element " + a.model.elements[a.sel_element].symbol);
  a.model.RemoveElement(a.sel_element);
  a.sel_element = -1;
  a.draft_for_element = -2;
  a.draft_for_material = -2;  // its components may have shifted
  a.rename_target = -1;
  a.scene_dirty = true;
}

static void DeleteSelectedMaterial(App& a) {
  if (a.sel_material < 0 || a.sel_material >= static_cast<int>(a.model.materials.size())) {
    return;
  }
  Log("deleted material " + a.model.materials[a.sel_material].name);
  a.model.RemoveMaterial(a.sel_material);
  a.sel_material = -1;
  a.draft_for_material = -2;
  a.rename_target = -1;
  a.scene_dirty = true;
}

static void DeleteSelectedScorer(App& a) {
  if (a.sel_scorer < 0 || a.sel_scorer >= static_cast<int>(a.model.scorers.size())) { return; }
  Log("deleted scorer " + a.model.scorers[a.sel_scorer].name);
  a.model.scorers.erase(a.model.scorers.begin() + a.sel_scorer);
  a.model.Touch();
  a.sel_scorer = -1;
  a.rename_target = -1;
  a.scene_dirty = true;
}

/// Deletes the selected source, including the last one.
///
/// The Delete button used to be greyed out at one source, on the theory that a model needs a
/// source to run. It does not need one to *exist*: a model with no source is a perfectly good
/// intermediate state, and refusing the delete meant the only way to replace a source was to
/// add a second one first. What a run needs is checked when a run is asked for, which is where
/// it belongs.
static void DeleteSelectedSource(App& a) {
  if (a.sel_source < 0 || a.sel_source >= static_cast<int>(a.model.sources.size())) { return; }
  Log("deleted source " + a.model.sources[a.sel_source].name);
  a.model.sources.erase(a.model.sources.begin() + a.sel_source);
  a.model.Touch();
  if (a.sel_source >= static_cast<int>(a.model.sources.size())) {
    a.sel_source = static_cast<int>(a.model.sources.size()) - 1;
  }
  a.draft_for_source = -2;
  a.rename_target = -1;
}

/// Adds a solid to a scorer, if it is not already there.
static void AttachSolidToScorer(App& a, int solid, int scorer) {
  if (solid < 0 || solid >= static_cast<int>(a.model.solids.size())) { return; }
  if (scorer < 0 || scorer >= static_cast<int>(a.model.scorers.size())) { return; }
  Scorer& sc = a.model.scorers[scorer];
  for (int v : sc.solids) {
    if (v == solid) {
      a.status = a.model.solids[solid].name + " already feeds " + sc.name;
      return;
    }
  }
  sc.solids.push_back(solid);
  a.model.Touch();
  a.scene_dirty = true;
  Log(Fmt("%s now feeds scorer \"%s\"", a.model.solids[solid].name.c_str(), sc.name.c_str()));
}

/// What the Delete key would delete, or kNone.
///
/// Nothing is deleted while a text field has focus - Delete and Backspace belong to the field
/// then - and nothing while a rename is in progress, for the same reason.
static DeleteWhat DeleteTarget(const App& a) {
  if (a.uic.focus != 0 || a.rename_target >= 0) { return DeleteWhat::kNone; }
  if (a.sel_solid > 0) { return DeleteWhat::kSolid; }
  if (a.sel_source >= 0 && a.sel_source < static_cast<int>(a.model.sources.size())) {
    return DeleteWhat::kSource;
  }
  if (a.sel_scorer >= 0) { return DeleteWhat::kScorer; }
  if (a.sel_material >= 0) { return DeleteWhat::kMaterial; }
  if (a.sel_element >= 0) { return DeleteWhat::kElement; }
  return DeleteWhat::kNone;
}

static void DeleteSelected(App& a) {
  switch (DeleteTarget(a)) {
    case DeleteWhat::kSolid: DeleteSelectedSolid(a); break;
    case DeleteWhat::kSource: DeleteSelectedSource(a); break;
    case DeleteWhat::kScorer: DeleteSelectedScorer(a); break;
    case DeleteWhat::kMaterial: DeleteSelectedMaterial(a); break;
    case DeleteWhat::kElement: DeleteSelectedElement(a); break;
    case DeleteWhat::kNone: break;
  }
}
