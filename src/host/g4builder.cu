// The model-building GUI.
//
// Same renderer as the viewer, same immediate-mode UI, same run engine. What it adds is an
// editable document (src/builder/model.hh) and the panels to edit it: elements, materials and
// solids down the left, source and physics down the right, a menu bar across the top, and the
// 3D view in the middle where solids are picked and dragged.
//
// The dragging model follows Microsoft 3D Builder: click a solid to select it, and three axis
// handles appear at its center. Dragging a handle moves the solid along that axis only, which
// is the interaction that makes a mouse usable for placing things in three dimensions - free
// dragging in a perspective view is ambiguous and ends up somewhere the user did not intend.
//
// Picking is done on the CPU, by casting the cursor ray against each solid with the same
// dist_in the navigator uses. That means what you can click is exactly what the transport can
// hit; a separate picking representation would eventually disagree with the physics.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <commdlg.h>
#include <GL/gl.h>

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "builder/build_scene.hh"
#include "data/nist_excitation.hh"
#include "data/nist_materials.hh"
#include "builder/import.hh"
#include "builder/model.hh"
#include "builder/write_project.hh"
#include "g4/G4RunManager.hh"
#include "g4/G4UImanager.hh"
#include "host/transport_run.cuh"
#include "render/png.h"
#include "render/renderer.cuh"
#include "render/ui.h"

using namespace g4gpu;
using namespace g4gpu::builder;
using real_t = double;

#define CUDA_CHECK(call)                                                                 \
  do {                                                                                   \
    const cudaError_t err__ = (call);                                                    \
    if (err__ != cudaSuccess) {                                                          \
      printf("CUDA error %s at %s:%d\n", cudaGetErrorString(err__), __FILE__, __LINE__); \
      std::exit(1);                                                                      \
    }                                                                                    \
  } while (0)

// ---------------------------------------------------------------- layout

/// Height of the bottom panel, and the sidebar widths, as *defaults*.
///
/// The live values are `App::left_w`, `App::right_w` and `App::bottom_h`, because the splitters
/// between the panels can be dragged. Everything that used to read a constant now reads the
/// App, so a resize moves the 3D view and the bottom panel with it - they touch, and a layout
/// where one moved and the others did not would leave a gap or an overlap.
constexpr int kLeftWDefault = 300;
constexpr int kRightWDefault = 300;
constexpr int kBottomHDefault = 250;
constexpr int kMenuH = 26;
constexpr int kStatusH = 22;
/// How wide a splitter is to grab, and how small a panel may get.
constexpr int kSplitterW = 5;
constexpr int kPanelMin = 180;
constexpr int kBottomMin = 90;

/// One same-layer overlap: the two volume indices, and how much of the smaller one is shared.
/// Declared here rather than in g4builder_overlap.inc because App holds a list of them.
struct SameLayerOverlap {
  int a = -1;
  int b = -1;
  int layer = 0;
  double fraction = 0;  ///< of the smaller volume's sampled points, how many were in both
};

/// What a picker pop-up is choosing. Defined here rather than in g4builder_pick.inc because
/// App holds one.
enum class PickKind {
  kNone, kMaterialForSolid, kScorerForSolid, kElementForMaterial, kMaterialForVoxelClass,
  kAnchorForSolid
};

/// Which pop-up, if any, is covering the view.
enum class Popup {
  kNone, kColor, kInsertPrimitive, kImportCad, kImportVoxel, kFlatDetector, kAbout,
  kPick,
  kOverlap,
  kPhysics, kVisAttributes,
  kWorld
};

/// Visualization settings that are not part of the model.
///
/// Not saved with the document, and deliberately so: these are how the user wants to *look* at
/// the detector, not what the detector is. A saved model that reopened with someone else's
/// background color would be worse than one that reopened with the default.
struct VisAttributes {
  float bg[3] = {0.055f, 0.06f, 0.075f};
  /// Track colours by CHARGE, which is what Geant4's default trajectory model draws: negative
  /// red, neutral green, positive blue. Anyone who has read a Geant4 picture knows that
  /// convention, and a viewer that uses a different one silently misreports every track.
  ///
  /// These were wrong. The fields were named by species and the values had electron on light
  /// blue and positron on orange - so an electron drew as a positive particle's colour and a
  /// positron as nothing in the convention at all. Renamed as well as recoloured, because the
  /// species names are what made it plausible: nothing about "electron = light blue" looks
  /// wrong until you notice that electron means negative and negative means red.
  float neutral[3] = {0.235f, 0.863f, 0.353f};   ///< green
  float negative[3] = {1.0f, 0.235f, 0.235f};    ///< red
  float positive[3] = {0.314f, 0.510f, 1.0f};    ///< blue
  bool show_axes = true;
  bool show_wireframe = true;
  bool show_solids = true;
  /// Cell boundaries on voxel volumes. On by default - a voxel detector that draws as a
  /// featureless slab is indistinguishable from a plain box. A grid finer than the pixels
  /// fades rather than being dropped, so this toggle is the only way to be rid of it.
  bool voxel_grid_lines = true;
  float track_width = 1.0f;
};

struct Drag {
  bool active = false;
  int axis = 0;          ///< 0 = x, 1 = y, 2 = z
  double start_value = 0;
  int start_mouse = 0;
  double scale = 1.0;    ///< mm per pixel along the screen projection of the axis
};

struct App {
  // window / GL
  HWND hwnd = nullptr;
  HDC hdc = nullptr;
  HGLRC hglrc = nullptr;
  GLuint tex = 0;
  int width = 1680, height = 960;
  /// Live panel sizes. Dragged by the splitters between the panels; clamped so no panel can
  /// be squeezed out of existence and the 3D view always keeps some room.
  int left_w = kLeftWDefault;
  int right_w = kRightWDefault;
  int bottom_h = kBottomHDefault;
  /// Which splitter is being dragged: 0 none, 1 left, 2 right, 3 bottom.
  int dragging_split = 0;
  /// Which splitter the pointer is on or dragging: 1 left, 2 right, 3 bottom, 0 none.
  ///
  /// Read by WM_SETCURSOR, which is a different thread of control from the frame that
  /// computes it - hence a field rather than a local. One frame of lag on a cursor shape is
  /// not perceptible.
  int split_hot = 0;
  bool running = true;
  bool resized = true;

  // camera
  float azimuth = 0.7f, elevation = 0.35f, distance = 1500.0f;
  float home_distance = 1500.0f;
  vis::Vec3f target{0.f, 0.f, 0.f};
  bool orbiting = false, panning = false;
  int last_x = 0, last_y = 0;

  // document
  Model model = Model::Default();
  std::string model_path;
  int sel_solid = -1;
  int sel_material = -1;
  int sel_element = -1;
  int sel_source = 0;
  int sel_scorer = -1;
  Drag drag;
  Popup popup = Popup::kNone;
  int popup_target = -1;
  /// Which voxel class the colour popup is editing, or -1 for the solid's own colour. Every
  /// place that opens the popup sets BOTH, because a stale index here would silently retarget
  /// the dialog at whatever class was edited last.
  int popup_target2 = -1;

  // scene mirror: rebuilt whenever the model changes
  bool scene_dirty = true;
  G4RunManager* run_manager = nullptr;
  host::TransportEngine<real_t>* engine = nullptr;
  ModelDetector* detector = nullptr;

  // rendering
  vis::TrajectoryBuffer traj{};
  vis::VolumeStyle* d_styles = nullptr;
  unsigned long long* d_fb = nullptr;
  unsigned int* d_rgba = nullptr;
  /// The viewport size d_fb and d_rgba were actually allocated for. Compared against the
  /// current ViewRect every frame; see AllocViewportSurface for what went wrong without it.
  int fb_w = 0, fb_h = 0;
  std::vector<unsigned int> host_rgba;
  int n_segments = 0;
  geom::Volume<real_t>* d_vols = nullptr;
  int n_vols = 0;

  // run control
  ui::NumberField events_field;
  /// How many events' trajectories are kept for drawing. Not the run size: a million-event
  /// run cannot draw a million tracks, and the cutoff used to be a constant nobody could
  /// reach. Zero draws none, which is the fastest a run goes.
  int traj_max_events = 25;
  ui::NumberField traj_events_field;
  double last_ms = 0;
  int last_events = 0;
  std::vector<double> last_scores;
  /// Energy deposit per voxel cell from the last run, summed over every source, or empty.
  /// Indexed by the cell index in the scene's voxel pool - the same index the geometry uses.
  std::vector<double> last_voxel_scores;
  std::vector<std::string> log;
  /// Where the terminal panel is scrolled to, as the index of the first visible line, or -1
  /// for "follow the tail" - which is what a terminal does and what a user reading new output
  /// wants until they scroll up.
  int log_scroll = -1;
  bool show_tracks = true;

  // property fields for the selected solid, re-initialised when the selection changes
  ui::NumberField pfield[12];
  ui::NumberField pos_field[3];
  ui::NumberField rot_field[3];
  int pfield_for = -2;

  ui::NumberField src_field[8];
  ui::NumberField src_activity;
  std::string src_name;
  int src_name_for = -2;
  int src_field_for = -2;

  ui::NumberField elem_z, elem_a;
  ui::NumberField mat_density, mat_excitation;
  ui::NumberField comp_field[8];   ///< one per component of the material being edited
  ui::NumberField src_weight;
  ui::NumberField opacity_field;

  // Text buffers, one per field.
  //
  // These were a single shared `name_edit` with an integer tag saying which field owned it.
  // Four sites reset it, all of them every frame, so with an element and a material both
  // selected the two fought over it and every typed character was gone by the next frame -
  // which is why typing into the builder stopped working the moment anything was added. See
  // ui::EditBuffer.
  ui::EditBuffer elem_sym_buf;
  ui::EditBuffer mat_name_buf;
  ui::EditBuffer mat_nist_buf;
  ui::EditBuffer src_name_buf;
  ui::EditBuffer src_particle_buf;
  ui::EditBuffer src_nuclide_buf;
  ui::EditBuffer rename_buf;
  /// Which list row is being renamed in place, as kind*100000 + index, or -1 for none.
  int rename_target = -1;

  /// The working copies the element, material and source forms edit.
  ///
  /// Add appends one of these, Update writes it into the selection. That is what makes "enter
  /// the parameters, then press Add" possible: the alternative - Add creates a default and the
  /// fields edit it live - means a half-configured source exists in the model while it is
  /// being typed, and there is no way to say "no, not that one".
  ///
  /// Solids are deliberately *not* done this way. They are edited by dragging handles in the
  /// view, and a drag that only took effect on Update would be unusable.
  Element draft_element;
  Material draft_material;
  builder::Source draft_source;
  int draft_for_element = -2;   ///< which selection the working copy was taken from
  int draft_for_material = -2;
  int draft_for_source = -2;

  // Panel scrolling. A model with a dozen materials overflowed the left panel and the rows
  // that fell off the bottom were simply not drawn.
  /// One per SECTION, not one per panel. A single scroll area per sidebar put every section
  /// on one scrollbar, so a few hundred voxel classes in SOLIDS pushed SOURCES out of reach.
  ui::ScrollArea elements_scroll;
  ui::ScrollArea materials_scroll;
  ui::ScrollArea scorers_scroll;
  ui::ScrollArea solids_scroll;
  ui::ScrollArea sources_scroll;
  ui::ScrollArea bottom_solids_scroll;
  ui::ScrollArea bottom_scorers_scroll;
  ui::ScrollArea pick_scroll;

  /// What the picker pop-up is choosing, and for which solid or material.
  PickKind pick_kind = PickKind::kNone;
  int pick_target = -1;
  int pick_target2 = -1;   ///< a second index, for "which voxel class"

  /// Volumes that overlap another on the *same* layer, found after every scene rebuild.
  ///
  /// A run with one of these is refused: the layer model gives shared space to the higher
  /// layer, and two volumes on the same layer have no such rule - the tie-break is whichever
  /// was added later, so the dose would depend on the order the detector was built in.
  std::vector<SameLayerOverlap> overlaps;

  VisAttributes vis_attr;

  /// The New-world dialog's fields.
  ///
  /// On App rather than static locals inside the pop-up, because the dialog is re-opened for
  /// every new scene and has to be re-seeded each time - a function-static would keep
  /// whatever the last scene used, which is a different scene's numbers presented as this
  /// one's defaults.
  int world_shape = 0;             ///< index into the dialog's shape list, not a Shape
  int world_unit = 1;              ///< index into kLengthUnits; 1 is mm
  ui::NumberField world_dim[3];
  bool world_dialog_seeded = false;

  /// The voxel-import dialog's fields.
  ///
  /// A raw file carries no header: nothing in it says how many cells there are, what type
  /// they are, or how big one is. Guessing a cube from the file size is right often enough to
  /// offer, and wrong often enough that it cannot be the only way in.
  ui::NumberField vox_n[3];
  ui::NumberField vox_res[3];
  int vox_type = 1;                ///< index into kVoxelTypeNames
  int vox_res_unit = 1;            ///< index into kLengthUnits
  int vox_meaning = 0;             ///< 0 material indices, 1 physical properties
  bool vox_guess_dims = true;
  bool vox_dialog_seeded = false;

  ui::Font font;
  ui::Context uic;
  ui::Input input;
  ui::MenuBar menu;
  std::string status = "ready";
};

static App g_app;

static void Log(const std::string& s) {
  std::printf("%s\n", s.c_str());
  g_app.log.push_back(s);
  if (g_app.log.size() > 4000) {
    g_app.log.erase(g_app.log.begin(), g_app.log.begin() + 1000);
  }
  g_app.log_scroll = -1;
}

static std::string Fmt(const char* fmt, ...) {
  char buf[512];
  va_list ap;
  va_start(ap, fmt);
  std::vsnprintf(buf, sizeof buf, fmt, ap);
  va_end(ap);
  return buf;
}

static ui::Rect ViewRect(const App& a) {
  return {a.left_w, kMenuH, a.width - a.left_w - a.right_w,
          a.height - kMenuH - kStatusH - a.bottom_h};
}

/// The bottom panel, spanning between the two sidebars and touching both.
static ui::Rect BottomRect(const App& a) {
  return {a.left_w, a.height - kStatusH - a.bottom_h, a.width - a.left_w - a.right_w,
          a.bottom_h};
}

/// The three splitters, as grab rectangles. Each straddles the boundary it moves, so the
/// cursor does not have to be on the exact pixel.
static ui::Rect LeftSplitRect(const App& a) {
  return {a.left_w - kSplitterW / 2, kMenuH, kSplitterW, a.height - kMenuH - kStatusH};
}
static ui::Rect RightSplitRect(const App& a) {
  return {a.width - a.right_w - kSplitterW / 2, kMenuH, kSplitterW,
          a.height - kMenuH - kStatusH};
}
static ui::Rect BottomSplitRect(const App& a) {
  const ui::Rect b = BottomRect(a);
  return {b.x, b.y - kSplitterW / 2, b.w, kSplitterW};
}

/// Drags whichever splitter was grabbed, and keeps every panel above its minimum.
///
/// Clamped rather than free: a sidebar dragged to zero width cannot be dragged back, because
/// its splitter would be under the other panel's edge. The 3D view keeps 200 pixels for the
/// same reason - a view of nothing cannot be clicked in to get the space back.
static void UpdateSplitters(App& a) {
  ui::Context& c = a.uic;
  const ui::Rect l = LeftSplitRect(a);
  const ui::Rect r = RightSplitRect(a);
  const ui::Rect b = BottomSplitRect(a);

  if (a.dragging_split == 0 && a.input.left_pressed) {
    if (l.Contains(a.input.mouse_x, a.input.mouse_y)) {
      a.dragging_split = 1;
    } else if (r.Contains(a.input.mouse_x, a.input.mouse_y)) {
      a.dragging_split = 2;
    } else if (b.Contains(a.input.mouse_x, a.input.mouse_y)) {
      a.dragging_split = 3;
    }
  }
  if (!a.input.left_down) { a.dragging_split = 0; }

  if (a.dragging_split == 1) {
    a.left_w = a.input.mouse_x;
  } else if (a.dragging_split == 2) {
    a.right_w = a.width - a.input.mouse_x;
  } else if (a.dragging_split == 3) {
    a.bottom_h = a.height - kStatusH - a.input.mouse_y;
  }
  if (a.left_w < kPanelMin) { a.left_w = kPanelMin; }
  if (a.right_w < kPanelMin) { a.right_w = kPanelMin; }
  if (a.bottom_h < kBottomMin) { a.bottom_h = kBottomMin; }
  const int view_min = 200;
  if (a.left_w + a.right_w > a.width - view_min) {
    // Give back whichever is being dragged, so the other keeps what the user set.
    if (a.dragging_split == 2) {
      a.right_w = a.width - view_min - a.left_w;
    } else {
      a.left_w = a.width - view_min - a.right_w;
    }
    if (a.left_w < kPanelMin) { a.left_w = kPanelMin; }
    if (a.right_w < kPanelMin) { a.right_w = kPanelMin; }
  }
  const int vh = a.height - kMenuH - kStatusH - a.bottom_h;
  if (vh < 120) { a.bottom_h = a.height - kMenuH - kStatusH - 120; }
  if (a.bottom_h < kBottomMin) { a.bottom_h = kBottomMin; }

  // The splitters, as a plain border line, and the affordance is the CURSOR.
  //
  // This used to tint the edge blue on hover, which says "something happens here" without
  // saying what: every other blue thing in this GUI is a selection or a focused field. A
  // left-right arrow on a vertical edge is what every application on the machine uses and
  // needs no explaining. WM_SETCURSOR reads split_hot; see WndProc.
  //
  // Hot includes being DRAGGED and not only hovered, because the pointer leaves the splitter
  // as soon as the drag starts moving it and the cursor must not flick back to an arrow
  // half-way through.
  a.split_hot = 0;
  auto handle = [&](const ui::Rect& s, int which) {
    if (a.dragging_split == which || s.Contains(a.input.mouse_x, a.input.mouse_y)) {
      a.split_hot = which;
    }
    c.canvas.FillRect(s, ui::theme::kBorder);
  };
  handle(LeftSplitRect(a), 1);
  handle(RightSplitRect(a), 2);
  handle(BottomSplitRect(a), 3);
}

static vis::Camera CurrentCamera(const App& a) {
  const float el = std::max(-1.5f, std::min(1.5f, a.elevation));
  const ui::Rect v = ViewRect(a);
  const vis::Vec3f eye{a.target.x + a.distance * std::cos(el) * std::sin(a.azimuth),
                       a.target.y + a.distance * std::sin(el),
                       a.target.z + a.distance * std::cos(el) * std::cos(a.azimuth)};
  return vis::make_camera(eye, a.target, vis::Vec3f{0.f, 1.f, 0.f}, 45.0f,
                          std::max(1, v.w), std::max(1, v.h));
}

// ---------------------------------------------------------------- scene rebuild

/// Recomputes the same-layer overlap list. Defined in g4builder_overlap.inc, which is
/// included with the panels below; forward declared because a rebuild is the only thing that
/// can change the answer and so is the only place worth recomputing it.
static void RefreshOverlaps(App& a);
static void ResetRunState(App& a);
/// Seeds the New-world dialog. Forward declared because File > New opens it and the menu bar
/// is written above the pop-ups that it opens.
static void SeedWorldDialog(App& a);

/// Rebuilds the G4 graph and re-uploads it. Called whenever the model changes shape.
///
/// The whole graph is rebuilt rather than patched. Patching would mean tracking which device
/// buffer each edit invalidates, and getting that wrong produces a render that disagrees with
/// the transport - the single most confusing failure a tool like this can have. A rebuild of a
/// few hundred volumes is a few milliseconds.
static void RebuildScene(App& a) {
  // The registries are global and additive, so a rebuild has to start from empty or the
  // previous version's volumes are still in the scene.
  G4PVPlacement::Registry().clear();
  G4LogicalVolume::Registry().clear();
  G4Material::Registry().clear();
  G4Element::Registry().clear();
  // And the sensitive-detector map, which is global and additive in the same way - see
  // G4SDManager::Reset for what a stale entry does once an address is reused.
  G4SDManager::GetSDMpointer()->Reset();

  delete a.run_manager;
  a.run_manager = new G4RunManager;
  a.detector = new ModelDetector(&a.model);
  a.run_manager->SetUserInitialization(a.detector);
  a.run_manager->SetUserAction(new ModelPrimary(&a.model));
  a.run_manager->SetCutValue(a.model.physics.range_cut_mm * mm);
  {
    // The physics panel's toggles reach the engine here. Before this existed they were
    // stored, written to the saved project, and ignored by the run - a switch that looked
    // like it did something.
    g4gpu::ProcessFlags pf;
    pf.photoelectric = a.model.physics.photoelectric;
    pf.compton = a.model.physics.compton;
    pf.rayleigh = a.model.physics.rayleigh;
    pf.pair_production = a.model.physics.pair_production;
    pf.bremsstrahlung = a.model.physics.bremsstrahlung;
    pf.annihilation = a.model.physics.annihilation;
    pf.multiple_scattering = a.model.physics.multiple_scattering;
    a.run_manager->SetProcesses(pf);
  }
  a.run_manager->SetBatchSize(65536);
  a.run_manager->Initialize();
  a.engine = &a.run_manager->GetEngine();

  const auto& scene = a.run_manager->GetScene();
  a.n_vols = static_cast<int>(scene.volumes.size());

  if (a.d_styles != nullptr) { cudaFree(a.d_styles); }
  std::vector<vis::VolumeStyle> styles(scene.volumes.size());
  for (std::size_t i = 0; i < scene.volumes.size(); ++i) {
    const auto& st = scene.styles[i];
    styles[i].r = static_cast<unsigned char>(st.r * 255);
    styles[i].g = static_cast<unsigned char>(st.g * 255);
    styles[i].b = static_cast<unsigned char>(st.b * 255);
    styles[i].a = static_cast<unsigned char>(st.opacity * 255 + 0.5f);
    styles[i].solid = st.visible && !st.wireframe;
  }
  CUDA_CHECK(cudaMalloc(&a.d_styles, sizeof(vis::VolumeStyle) * std::max<size_t>(1, styles.size())));
  if (!styles.empty()) {
    CUDA_CHECK(cudaMemcpy(a.d_styles, styles.data(), sizeof(vis::VolumeStyle) * styles.size(),
                          cudaMemcpyHostToDevice));
  }

  // Frame the camera on the world the first time.
  static bool framed = false;
  if (!framed && !scene.volumes.empty()) {
    double reach = 0;
    for (const auto& v : scene.volumes) {
      reach = std::max(reach, geom::solid_half_extent(scene.pool.store(), v.solid));
    }
    a.home_distance = static_cast<float>(std::max(50.0, reach * 3.0));
    a.distance = a.home_distance;
    framed = true;
  }
  a.scene_dirty = false;
  // After the geometry, not per frame: this samples thousands of points per candidate pair
  // and the answer only changes when the geometry does.
  RefreshOverlaps(a);
  // The geometry just changed, so the trajectories on screen were computed in a different
  // detector and the dose beside them was measured in one. Both are cleared.
  ResetRunState(a);
}

// ---------------------------------------------------------------- picking

/// The world-space ray under the cursor.
static bool CursorRay(const App& a, Vec3<real_t>& origin, Vec3<real_t>& dir) {
  const ui::Rect v = ViewRect(a);
  if (!v.Contains(a.input.mouse_x, a.input.mouse_y)) { return false; }
  const vis::Camera cam = CurrentCamera(a);
  const vis::Vec3f d = cam.ray_dir(a.input.mouse_x - v.x, a.input.mouse_y - v.y);
  origin = Vec3<real_t>{cam.eye.x, cam.eye.y, cam.eye.z};
  dir = Vec3<real_t>{d.x, d.y, d.z};
  return true;
}

/// The model solid the cursor is over, or -1. The world is skipped: clicking through it to
/// what is inside is what a user expects, and it is the one volume that always contains the
/// ray's entry point anyway.
static int PickSolid(const App& a) {
  Vec3<real_t> o, d;
  if (!CursorRay(a, o, d)) { return -1; }
  if (a.run_manager == nullptr) { return -1; }
  const auto& scene = a.run_manager->GetScene();
  const auto store = scene.pool.store();

  double best_t = 1e29;
  int best = -1;
  for (std::size_t i = 0; i < scene.volumes.size(); ++i) {
    if (static_cast<int>(i) == scene.world) { continue; }
    const auto& v = scene.volumes[i];
    const double t = geom::dist_in(store, v.solid, geom::to_local(v.xform, o),
                                   geom::dir_to_local(v.xform, d));
    if (t < best_t) {
      best_t = t;
      best = static_cast<int>(i);
    }
  }
  if (best < 0 || best_t >= 1e29) { return -1; }
  // Device volume index back to a model solid index: the flattener emits placed solids in
  // model order, skipping boolean operands and unplaced ones, so the names are matched instead
  // of the indices - a positional mapping would silently shift the moment a solid is skipped.
  const auto& names = a.run_manager->GetScene().names;
  for (std::size_t k = 0; k < a.model.solids.size(); ++k) {
    if (a.model.solids[k].name == names[best]) { return static_cast<int>(k); }
  }
  return -1;
}

/// Screen position of a world point, in window coordinates.
static bool ProjectToScreen(const App& a, const Vec3<real_t>& p, int& sx, int& sy) {
  const ui::Rect v = ViewRect(a);
  const vis::Camera cam = CurrentCamera(a);
  const vis::Vec3f rel{static_cast<float>(p.x) - cam.eye.x,
                       static_cast<float>(p.y) - cam.eye.y,
                       static_cast<float>(p.z) - cam.eye.z};
  const float z = rel.x * cam.forward.x + rel.y * cam.forward.y + rel.z * cam.forward.z;
  if (z <= 1e-3f) { return false; }
  const float x = rel.x * cam.right.x + rel.y * cam.right.y + rel.z * cam.right.z;
  const float y = rel.x * cam.up.x + rel.y * cam.up.y + rel.z * cam.up.z;
  // Matches vis::make_camera's tan(fov/2) scaling.
  const float half = std::tan(45.0f * 0.5f * 3.14159265f / 180.0f);
  const float aspect = static_cast<float>(v.w) / std::max(1, v.h);
  sx = v.x + static_cast<int>((0.5f + 0.5f * (x / z) / (half * aspect)) * v.w);
  sy = v.y + static_cast<int>((0.5f - 0.5f * (y / z) / half) * v.h);
  return true;
}

// ---------------------------------------------------------------- gizmo

constexpr int kGizmoLen = 70;    ///< pixels

/// Draws the three axis handles for the selected solid and starts a drag if one is grabbed.
static void DrawGizmo(App& a) {
  if (a.sel_solid < 0 || a.sel_solid >= static_cast<int>(a.model.solids.size())) { return; }
  const Solid& s = a.model.solids[a.sel_solid];
  const Vec3<real_t> center{s.pos[0], s.pos[1], s.pos[2]};
  int cx = 0, cy = 0;
  if (!ProjectToScreen(a, center, cx, cy)) { return; }

  const ui::Color axis_col[3] = {ui::rgb(232, 96, 96), ui::rgb(120, 216, 120),
                                  ui::rgb(110, 160, 240)};
  const char* axis_name[3] = {"X", "Y", "Z"};

  // Handle direction on screen: project a point one unit along the axis and take the offset.
  // Doing it this way rather than assuming screen-aligned axes is what makes the handles point
  // where the axis actually goes after an orbit.
  for (int ax = 0; ax < 3; ++ax) {
    Vec3<real_t> along = center;
    const double step = std::max(1.0, a.distance * 0.05);
    if (ax == 0) { along.x += step; } else if (ax == 1) { along.y += step; } else { along.z += step; }
    int ax_x = 0, ax_y = 0;
    if (!ProjectToScreen(a, along, ax_x, ax_y)) { continue; }
    double dx = ax_x - cx, dy = ax_y - cy;
    const double len = std::sqrt(dx * dx + dy * dy);
    if (len < 1e-6) { continue; }  // the axis points at the camera; no usable handle
    const double px_per_mm = len / step;
    dx /= len;
    dy /= len;
    const int hx = cx + static_cast<int>(dx * kGizmoLen);
    const int hy = cy + static_cast<int>(dy * kGizmoLen);

    // The shaft.
    for (int i = 0; i < kGizmoLen; ++i) {
      a.uic.canvas.Put(cx + static_cast<int>(dx * i), cy + static_cast<int>(dy * i),
                       axis_col[ax]);
      a.uic.canvas.Put(cx + static_cast<int>(dx * i) + 1, cy + static_cast<int>(dy * i),
                       axis_col[ax]);
    }
    // The handle: an arrowhead pointing the way the axis goes.
    //
    // It was a square, which says "grab me" and says nothing about which direction dragging
    // moves the solid - the one thing the handle exists to communicate, and the one thing a
    // square cannot. The head is built from the axis's own screen direction (dx, dy) and the
    // perpendicular to it, so it turns with the view rather than pointing at a fixed corner.
    //
    // Drawn from a little behind the shaft's end to a little past it, so the arrow reads as
    // the end of the shaft rather than as a separate triangle floating near it.
    const double perp_x = -dy, perp_y = dx;
    const double head_half = 7.0;      // base half-width, pixels
    const double base_at = kGizmoLen - 6.0;
    const double tip_at = kGizmoLen + 9.0;
    const int tipx = cx + static_cast<int>(dx * tip_at);
    const int tipy = cy + static_cast<int>(dy * tip_at);
    const int bx = cx + static_cast<int>(dx * base_at);
    const int by = cy + static_cast<int>(dy * base_at);
    const ui::Rect grab{hx - 8, hy - 8, 17, 17};
    const bool hover = grab.Contains(a.input.mouse_x, a.input.mouse_y);
    const ui::Color head_col = hover ? ui::theme::kAccentHot : axis_col[ax];
    a.uic.canvas.FillTriangle(tipx, tipy,
                              bx + static_cast<int>(perp_x * head_half),
                              by + static_cast<int>(perp_y * head_half),
                              bx - static_cast<int>(perp_x * head_half),
                              by - static_cast<int>(perp_y * head_half), head_col);
    // The label sits beyond the tip rather than beside the old square, so it does not land on
    // top of the head for an axis pointing up and to the left.
    a.uic.canvas.Text(cx + static_cast<int>(dx * (tip_at + 6)) - 3,
                      cy + static_cast<int>(dy * (tip_at + 6)) - 7, axis_name[ax],
                      axis_col[ax]);

    if (hover && a.input.left_pressed && !a.drag.active) {
      a.drag.active = true;
      a.drag.axis = ax;
      a.drag.start_value = s.pos[ax];
      // Track the pointer along the axis's screen direction, so a diagonal axis follows the
      // cursor rather than only its x component.
      a.drag.start_mouse = static_cast<int>(a.input.mouse_x * dx + a.input.mouse_y * dy);
      a.drag.scale = (px_per_mm > 1e-9) ? 1.0 / px_per_mm : 1.0;
      a.uic.active = 9999;  // claim the click so the view does not also orbit
    }
    if (a.drag.active && a.drag.axis == ax && a.input.left_down) {
      const int now = static_cast<int>(a.input.mouse_x * dx + a.input.mouse_y * dy);
      const double moved = (now - a.drag.start_mouse) * a.drag.scale;
      a.model.solids[a.sel_solid].pos[ax] = a.drag.start_value + moved;
      a.model.Touch();
      a.scene_dirty = true;
      a.pfield_for = -2;  // the position fields must re-read
      a.status = Fmt("%s.%s = %.2f mm", s.name.c_str(), axis_name[ax],
                     a.model.solids[a.sel_solid].pos[ax]);
    }
  }
  if (a.drag.active && !a.input.left_down) { a.drag.active = false; }

  // A box round the selection, so it is obvious which solid the handles belong to.
  int bx = 0, by = 0;
  if (ProjectToScreen(a, center, bx, by)) {
    a.uic.canvas.StrokeRect({bx - 4, by - 4, 9, 9}, ui::theme::kAccent);
  }
}

#include "host/g4builder_panels.inc"

// ---------------------------------------------------------------- rendering

/// Reallocates the device framebuffer for the current viewport size.
///
/// The viewport is the window MINUS the panels, so it changes when a splitter is dragged and
/// not only when the window is resized - and that was the bug. This ran on `a.resized` alone,
/// so dragging a side panel narrower made the viewport wider than the buffer allocated for it
/// and render_geometry wrote past the end of a.d_fb: a device heap corruption a few frames
/// into the drag, reported as the builder crashing when a side panel is resized.
///
/// The bottom splitter has always had the same defect and it takes dragging DOWNWARD to
/// trigger, which is why only the side ones were seen to crash. Keyed on the size actually
/// allocated rather than on any flag, so nothing has to remember to set one.
static void AllocViewportSurface(App& a) {
  const ui::Rect v = ViewRect(a);
  const int w = std::max(1, v.w), h = std::max(1, v.h);
  if (a.d_fb != nullptr && w == a.fb_w && h == a.fb_h) { return; }
  if (a.d_fb != nullptr) { cudaFree(a.d_fb); }
  if (a.d_rgba != nullptr) { cudaFree(a.d_rgba); }
  CUDA_CHECK(cudaMalloc(&a.d_fb, sizeof(unsigned long long) * w * h));
  CUDA_CHECK(cudaMalloc(&a.d_rgba, sizeof(unsigned int) * w * h));
  a.fb_w = w;
  a.fb_h = h;
}

static void AllocSurface(App& a) {
  AllocViewportSurface(a);
  a.host_rgba.assign(static_cast<size_t>(a.width) * a.height, ui::theme::kPanel);

  if (a.tex == 0) { glGenTextures(1, &a.tex); }
  glBindTexture(GL_TEXTURE_2D, a.tex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, a.width, a.height, 0, GL_RGBA, GL_UNSIGNED_BYTE,
               nullptr);
  a.resized = false;
}

/// The palette the Visualization attributes window has set.
///
/// The background is one color there rather than a gradient, so it is used for both ends of
/// the gradient - a flat background when the user picks one, a gradient when they leave the
/// default alone.
static vis::Palette CurrentPalette(const App& a) {
  const VisAttributes& v = a.vis_attr;
  auto pack = [](const float* c) {
    const int r = static_cast<int>(c[0] * 255 + 0.5f);
    const int g = static_cast<int>(c[1] * 255 + 0.5f);
    const int b = static_cast<int>(c[2] * 255 + 0.5f);
    return (static_cast<unsigned int>(r) << 16) | (static_cast<unsigned int>(g) << 8)
           | static_cast<unsigned int>(b);
  };
  vis::Palette p;
  p.neutral = pack(v.neutral);
  p.negative = pack(v.negative);
  p.positive = pack(v.positive);
  const VisAttributes def{};
  if (v.bg[0] != def.bg[0] || v.bg[1] != def.bg[1] || v.bg[2] != def.bg[2]) {
    p.bg_top = pack(v.bg);
    p.bg_bottom = p.bg_top;
  }
  return p;
}

static void DrawFrame(App& a) {
  if (a.resized) { AllocSurface(a); }
  // Every frame, because a splitter drag resizes the viewport without resizing the window.
  // It returns immediately when the size has not moved, so this costs a comparison.
  AllocViewportSurface(a);
  if (a.scene_dirty) { RebuildScene(a); }

  const ui::Rect v = ViewRect(a);
  const int w = std::max(1, v.w), h = std::max(1, v.h);
  const vis::Camera cam = CurrentCamera(a);
  const dim3 block(16, 16);
  const dim3 grid((w + 15) / 16, (h + 15) / 16);
  const vis::Palette pal = CurrentPalette(a);

  if (a.vis_attr.show_solids) {
    vis::render_geometry<real_t><<<grid, block>>>(a.engine->geometry(), a.d_styles, cam,
                                                  a.d_fb, a.vis_attr.voxel_grid_lines);
  } else {
    CUDA_CHECK(cudaMemset(a.d_fb, 0xFF, sizeof(unsigned long long) * w * h));
  }
  if (a.show_tracks && a.n_segments > 0) {
    const int thick = static_cast<int>(a.vis_attr.track_width) - 1;
    vis::render_trajectories<<<(a.n_segments + 127) / 128, 128>>>(
        a.traj, a.n_segments, cam, a.d_fb, thick > 0 ? thick : 0, 1e-3f, pal);
  }
  vis::resolve_to_rgba<<<grid, block>>>(a.d_fb, a.d_rgba, w, h, pal);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaGetLastError());

  // Background for the panels, then the render blitted into the viewport rectangle.
  std::fill(a.host_rgba.begin(), a.host_rgba.end(), ui::theme::kPanel);
  for (int y = 0; y < h; ++y) {
    CUDA_CHECK(cudaMemcpy(&a.host_rgba[static_cast<size_t>(v.y + y) * a.width + v.x],
                          a.d_rgba + static_cast<size_t>(y) * w, sizeof(unsigned int) * w,
                          cudaMemcpyDeviceToHost));
  }

  a.uic.Begin(a.host_rgba.data(), a.width, a.height, &a.font, &a.input);
  DrawGizmo(a);
  // Before the panels, so a drag this frame moves them this frame rather than next: the
  // splitter would otherwise lag the cursor by one frame and feel detached from it.
  UpdateSplitters(a);
  DrawLeftPanel(a);
  DrawRightPanel(a);
  DrawBottomPanel(a);
  DrawStatusBar(a);
  // Once per layer, each immediately after that layer is drawn.
  //
  // A dropdown has to be painted after everything at its own layer - the panel that declared
  // it goes on drawing below it - and behind anything in front of that layer. One call after
  // the panels did the first and got the second wrong for pop-ups: a Select declared inside a
  // pop-up (the voxel import dialog has one) was painted here and then covered by the pop-up
  // itself, which is the bug this pair of calls fixes. See ui::DrawOpenSelect.
  ui::DrawOpenSelect(a.uic, ui::Context::kLayerPanel);
  DrawMenuBar(a);  // its own dropdown is drawn by DrawMenuBar and covers the panels
  {
    ui::LayerScope pop(a.uic, ui::Context::kLayerPopup);
    DrawPopup(a);
    ui::DrawOpenSelect(a.uic, ui::Context::kLayerPopup);
  }

  // The Delete key, after the panels: DeleteTarget looks at ui::Context::focus, which is only
  // meaningful once this frame's widgets have run. It deletes whatever is selected - solid,
  // source, scorer, material, element, in that order - and does nothing while a text field
  // has focus, where Delete belongs to the field.
  if (a.input.KeyPressed(VK_DELETE)) { DeleteSelected(a); }
  // Escape closes a pop-up and cancels a rename, which is what it does everywhere else.
  if (a.input.KeyPressed(VK_ESCAPE)) {
    if (a.popup != Popup::kNone) {
      a.popup = Popup::kNone;
    } else if (a.rename_target >= 0) {
      a.rename_target = -1;
      a.uic.focus = 0;
    }
  }
  a.uic.End();

  glViewport(0, 0, a.width, a.height);
  glBindTexture(GL_TEXTURE_2D, a.tex);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, a.width, a.height, GL_RGBA, GL_UNSIGNED_BYTE,
                  a.host_rgba.data());
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0, 1, 0, 1, -1, 1);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glEnable(GL_TEXTURE_2D);
  glBegin(GL_QUADS);
  glTexCoord2f(0, 1); glVertex2f(0, 0);
  glTexCoord2f(1, 1); glVertex2f(1, 0);
  glTexCoord2f(1, 0); glVertex2f(1, 1);
  glTexCoord2f(0, 0); glVertex2f(0, 1);
  glEnd();
  glDisable(GL_TEXTURE_2D);
  SwapBuffers(a.hdc);
  a.input.EndFrame();
}

// ---------------------------------------------------------------- input

static LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
  App& a = g_app;
  const ui::Rect view = ViewRect(a);
  const bool over_view = view.Contains(a.input.mouse_x, a.input.mouse_y);
  switch (msg) {
    case WM_CLOSE:
    case WM_DESTROY:
      a.running = false;
      return 0;
    // A resize cursor over a splitter.
    //
    // Handled here and not once at startup because the window class carries IDC_ARROW, and
    // Windows re-applies the class cursor on every mouse move unless WM_SETCURSOR says
    // otherwise. Returning TRUE is what stops that; falling through to DefWindowProc is what
    // restores the arrow everywhere else, so there is no state to put back.
    case WM_SETCURSOR:
      if (LOWORD(lp) == HTCLIENT) {
        if (a.split_hot == 1 || a.split_hot == 2) {
          SetCursor(LoadCursor(nullptr, IDC_SIZEWE));
          return TRUE;
        }
        if (a.split_hot == 3) {
          SetCursor(LoadCursor(nullptr, IDC_SIZENS));
          return TRUE;
        }
      }
      break;
    case WM_SIZE: {
      const int w = LOWORD(lp), h = HIWORD(lp);
      if (w > 0 && h > 0 && (w != a.width || h != a.height)) {
        a.width = w;
        a.height = h;
        a.resized = true;
      }
      return 0;
    }
    case WM_LBUTTONDBLCLK:
      // Raises left_pressed as well: with CS_DBLCLKS on the class, the second click comes
      // here *instead of* WM_LBUTTONDOWN, so every widget that reacts to a press - which is
      // all of them - would miss it otherwise.
      a.input.left_dbl = true;
      [[fallthrough]];
    case WM_LBUTTONDOWN:
      a.input.left_down = true;
      a.input.left_pressed = true;
      a.last_x = static_cast<short>(LOWORD(lp));
      a.last_y = static_cast<short>(HIWORD(lp));
      SetCapture(hwnd);
      return 0;
    case WM_LBUTTONUP:
      a.input.left_down = false;
      a.input.left_released = true;
      a.orbiting = false;
      ReleaseCapture();
      return 0;
    case WM_RBUTTONDOWN:
      a.input.right_down = true;
      a.input.right_pressed = true;
      if (over_view) {
        a.panning = true;
        a.last_x = static_cast<short>(LOWORD(lp));
        a.last_y = static_cast<short>(HIWORD(lp));
      }
      SetCapture(hwnd);
      return 0;
    case WM_RBUTTONUP:
      a.input.right_down = false;
      a.input.right_released = true;
      a.panning = false;
      ReleaseCapture();
      return 0;
    case WM_MOUSEMOVE: {
      const int x = static_cast<short>(LOWORD(lp)), y = static_cast<short>(HIWORD(lp));
      const int dx = x - a.last_x, dy = y - a.last_y;
      a.input.mouse_x = x;
      a.input.mouse_y = y;
      // Orbiting begins only after the cursor has moved with the button down inside the view
      // and no gizmo handle claimed the press, so a click-to-select does not also spin the
      // camera by a pixel.
      if (a.input.left_down && over_view && !a.drag.active && a.uic.active == 0
          && (std::abs(dx) + std::abs(dy)) > 2) {
        a.orbiting = true;
      }
      if (a.orbiting) {
        a.azimuth -= dx * 0.008f;
        a.elevation += dy * 0.008f;
      } else if (a.panning) {
        const vis::Camera cc = CurrentCamera(a);
        const float s = a.distance * 0.0015f;
        a.target = a.target - (dx * s) * cc.right + (dy * s) * cc.up;
      }
      a.last_x = x;
      a.last_y = y;
      return 0;
    }
    case WM_MOUSEWHEEL: {
      const int delta = GET_WHEEL_DELTA_WPARAM(wp);
      a.input.wheel += delta / 120;
      if (over_view) {
        a.distance *= std::pow(0.9f, delta / 120.0f);
        a.distance = std::max(a.home_distance * 0.02f,
                              std::min(a.home_distance * 20.0f, a.distance));
      }
      return 0;
    }
    case WM_CHAR:
      if (wp >= 8 && wp < 127) { a.input.typed.push_back(static_cast<char>(wp)); }
      return 0;
    case WM_KEYDOWN:
      a.input.keys.push_back(static_cast<int>(wp));
      a.input.ctrl = (GetKeyState(VK_CONTROL) & 0x8000) != 0;
      a.input.shift = (GetKeyState(VK_SHIFT) & 0x8000) != 0;
      return 0;
    default:
      break;
  }
  return DefWindowProc(hwnd, msg, wp, lp);
}

// ---------------------------------------------------------------- main

int main(int argc, char** argv) {
  App& a = g_app;
  int selftest_frames = 0;
  std::string open_path;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "-selftest") == 0) {
      selftest_frames = 44;
    } else if (std::strcmp(argv[i], "-w") == 0 && i + 1 < argc) {
      a.width = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "-h") == 0 && i + 1 < argc) {
      a.height = std::atoi(argv[++i]);
    } else if (argv[i][0] != '-') {
      open_path = argv[i];
    }
  }

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::printf("GPU: %s, CC %d.%d\n", prop.name, prop.major, prop.minor);

  if (!open_path.empty()) {
    if (ReadModelFile(a.model, open_path)) {
      a.model_path = open_path;
      // A .model document stores the file an import came from, not its data; both have to be
      // re-read before the scene can be built.
      ReloadVoxelSources(a);
      ReloadMeshSources(a);
    }
  } else if (selftest_frames == 0) {
    // No document on the command line, so this is a new scene and the world has to be asked
    // for. Skipped under -selftest, which has no one to answer and builds its own scene: the
    // dialog is modal, so leaving it up would make every selftest frame draw the dialog and
    // nothing else.
    SeedWorldDialog(a);
    a.popup = Popup::kWorld;
  }

  WNDCLASSA wc{};
  // CS_DBLCLKS is what makes Windows send WM_LBUTTONDBLCLK at all; without it the second
  // click of a double-click is an ordinary WM_LBUTTONDOWN and a text field cannot tell the
  // difference. See the WM_LBUTTONDBLCLK case: it must also raise left_pressed, because with
  // this style set the second click no longer arrives as a WM_LBUTTONDOWN.
  wc.style = CS_OWNDC | CS_DBLCLKS;
  wc.lpfnWndProc = WndProc;
  wc.hInstance = GetModuleHandle(nullptr);
  wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
  wc.lpszClassName = "g4gpu_builder";
  RegisterClassA(&wc);

  RECT r{0, 0, a.width, a.height};
  AdjustWindowRect(&r, WS_OVERLAPPEDWINDOW, FALSE);
  a.hwnd = CreateWindowA("g4gpu_builder", "g4gpu - model builder", WS_OVERLAPPEDWINDOW,
                         CW_USEDEFAULT, CW_USEDEFAULT, r.right - r.left, r.bottom - r.top,
                         nullptr, nullptr, wc.hInstance, nullptr);
  if (a.hwnd == nullptr) {
    std::printf("CreateWindow failed\n");
    return 1;
  }
  a.hdc = GetDC(a.hwnd);

  PIXELFORMATDESCRIPTOR pfd{};
  pfd.nSize = sizeof(pfd);
  pfd.nVersion = 1;
  pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
  pfd.iPixelType = PFD_TYPE_RGBA;
  pfd.cColorBits = 32;
  SetPixelFormat(a.hdc, ChoosePixelFormat(a.hdc, &pfd), &pfd);
  a.hglrc = wglCreateContext(a.hdc);
  if (a.hglrc == nullptr || !wglMakeCurrent(a.hdc, a.hglrc)) {
    std::printf("wglCreateContext failed\n");
    return 1;
  }
  ShowWindow(a.hwnd, SW_SHOW);

  if (!a.font.Build("Consolas", 14) && !a.font.Build("Courier New", 14)
      && !a.font.Build("Lucida Console", 14)) {
    std::printf("FATAL: could not build a font atlas\n");
    return 2;
  }

  {
    CUDA_CHECK(vis::allocate_trajectory(a.traj, 2 << 20, /*max_event=*/100));
  }

  a.events_field.Init(1000, "%.0f");
  a.traj_events_field.Init(a.traj_max_events, "%.0f");
  G4UImanager::GetUIpointer()->SetOutputSink(
      [](const G4String& s) { g_app.log.push_back(s); });
  Log("model builder ready - Insert to add a solid, click to select, drag a handle to move");

  int frame = 0;
  while (a.running) {
    MSG msg;
    while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
      if (msg.message == WM_QUIT) { a.running = false; }
      TranslateMessage(&msg);
      DispatchMessage(&msg);
    }
    if (!a.running) { break; }

    if (selftest_frames > 0) {
      // Exercise the whole path without a human: insert solids, import a mesh, score it,
      // run, and save a project.
      //
      // The voxel volume and the mesh are here because they are what the generated project's
      // sidecar writers and loaders need in order to be exercised at all, and the scorer is
      // here because without one the generated project reports no number - "it compiled and
      // ran" would be true of a project that scored nothing.
      if (frame == 2) { InsertPrimitive(a, Shape::kOrb); }
      if (frame == 4) { InsertPrimitive(a, Shape::kTubs); }
      if (frame == 6) { InsertSelftestVoxels(a); }
      if (frame == 7) { InsertSelftestMesh(a); }   // leaves the mesh selected
      if (frame == 8) { AddScorer(a, ScoreQuantity::kDose); }  // ... so the scorer lands on it
      if (frame == 9) { InsertSelftestStoreSolids(a); }
      if (frame == 10) { InsertSelftestVisuals(a); }
      if (frame == 11) { InsertSelftestTransparency(a); }
      if (frame == 12) { SelftestCheckOverlapRefusal(a); }
      if (frame == 13) { SelftestUseWater(a); }
      if (frame == 14) { a.sel_solid = 1; }
      if (frame == 14) {
        // The camera onto the transparent pair, and fewer events than a real run: 200 events
        // of 6 MeV gammas draw a wall of trajectories that hides every solid behind it, and
        // the picture is the only place the rendering gets looked at without a person here.
        a.target = vis::Vec3f{95.f, 0.f, 0.f};
        a.distance = 900.f;
        a.azimuth = 0.9f;
        a.elevation = 0.25f;
      }
      if (frame == 14) { AddSelftestVoxelScorer(a); }
      if (frame == 15) { RunFromGui(a, 4000); }
      if (frame == 16) { SelftestCheckVoxelScoring(a); }
      if (frame == 18) { SelftestCheckCustomScorer(a); }
      if (frame == 16) {
        ProbeRay(a, g4gpu::Vec3<G4double>{0, 0, -400}, g4gpu::Vec3<G4double>{0, 0, 1});
      }
      if (frame == 19) { SelftestRunForComparison(a); }
      if (frame == 20) { SaveProjectTo(a, "D:/g4gpu/out/selftest_project"); }

      // The dialogs, each held up for a frame and photographed.
      //
      // They are the part of the GUI nothing else touches: every other check reads a number,
      // and a dialog whose text runs past its frame, whose controls fall off the bottom, or
      // whose list does not fit looks correct to all of them. Opened one frame and captured
      // the next, because a pop-up is drawn at the end of the frame that sets it.
      if (frame == 22) {
        SeedWorldDialog(a);
        a.popup = Popup::kWorld;
      }
      if (frame == 23) { SaveFramePng(a, "D:/g4gpu/out/g4builder_dlg_world.png"); }
      if (frame == 24) {
        a.popup = Popup::kNone;
        SeedVoxelDialog(a);
        a.vox_guess_dims = false;   // so the cell fields are live rather than greyed
        a.popup = Popup::kImportVoxel;
      }
      if (frame == 25) { SaveFramePng(a, "D:/g4gpu/out/g4builder_dlg_voxel.png"); }
      // The same dialog with one of its dropdowns OPEN, which is a picture worth having on
      // its own account: a list declared inside a pop-up used to be painted at the panel
      // layer and then covered by the pop-up that owned it, so it simply did not appear. See
      // docs/RISK.md V14.
      //
      // Nothing here asserts it - this is a photograph, and the check is that somebody looks
      // at it. Opening by id alone works because ui::Select refreshes its own geometry every
      // frame it is drawn open.
      if (frame == 26) { a.uic.open_select = 1270; }  // "values" in DrawImportVoxelPopup
      if (frame == 27) { SaveFramePng(a, "D:/g4gpu/out/g4builder_dlg_voxel_open.png"); }
      if (frame == 28) {
        a.uic.open_select = 0;
        a.popup = Popup::kPick;
        a.pick_kind = PickKind::kAnchorForSolid;
        a.pick_target = a.sel_solid > 0 ? a.sel_solid : 1;
      }
      if (frame == 29) { SaveFramePng(a, "D:/g4gpu/out/g4builder_dlg_anchor.png"); }
      if (frame == 30) {
        a.popup = Popup::kPhysics;
      }
      if (frame == 31) { SaveFramePng(a, "D:/g4gpu/out/g4builder_dlg_physics.png"); }
      if (frame == 32) {
        a.popup = Popup::kVisAttributes;
      }
      if (frame == 33) { SaveFramePng(a, "D:/g4gpu/out/g4builder_dlg_vis.png"); }
      if (frame == 34) { a.popup = Popup::kNone; }

      // Splitter drags, which used to corrupt the device heap.
      //
      // The viewport is the window minus the panels, so moving a splitter resizes it. The
      // framebuffer was reallocated on window resize only, so widening the viewport by
      // dragging a side panel narrower made every kernel write past the end of it - the
      // builder crashed a few frames into the drag. See AllocViewportSurface.
      //
      // Each of these frames renders at a viewport size the previous frame did not have, in
      // both directions and on all three splitters, which is what a drag is. The check is
      // that the process is still here afterwards and CUDA has not reported anything: the
      // kernels are launched by DrawFrame and its CUDA_CHECK follows the synchronise.
      if (frame == 35) { a.left_w = kPanelMin; }
      if (frame == 36) { a.left_w = 460; a.right_w = kPanelMin; }
      if (frame == 37) { a.right_w = 420; a.bottom_h = kBottomMin; }
      if (frame == 38) {
        a.bottom_h = 200;
        std::printf("selftest: the viewport survived being resized by every splitter\n");
      }

      // The colour dialog aimed at ONE VOXEL CLASS rather than at a whole solid.
      //
      // Clicking a class's colour swatch used to open the material picker - the swatch had no
      // hit region of its own, so the click belonged to the row. The dialog it opens now is
      // the same dialog, retargeted by popup_target2, and this photographs it: the title has
      // to name the class's index, the fields have to show the class's colour, and the
      // wireframe checkbox has to be absent because a class has no surface of its own.
      if (frame == 39) {
        for (std::size_t i = 0; i < a.model.solids.size(); ++i) {
          if (!a.model.solids[i].voxel_classes.empty()) {
            a.popup = Popup::kColor;
            a.popup_target = static_cast<int>(i);
            a.popup_target2 = 0;
            a.opacity_field.Init(a.model.solids[i].voxel_classes[0].opacity * 100.0, "%.0f");
            break;
          }
        }
        if (a.popup != Popup::kColor) {
          std::printf("selftest: FAILED - no voxel volume had classes to colour\n");
        }
      }
      if (frame == 40) {
        SaveFramePng(a, "D:/g4gpu/out/g4builder_dlg_class_color.png");
        a.popup = Popup::kNone;
        a.popup_target2 = -1;
      }
      if (frame >= selftest_frames) {
        std::vector<unsigned char> rgb(static_cast<size_t>(a.width) * a.height * 3);
        for (size_t i = 0; i < static_cast<size_t>(a.width) * a.height; ++i) {
          const unsigned p = a.host_rgba[i];
          rgb[i * 3 + 0] = static_cast<unsigned char>(p & 255);
          rgb[i * 3 + 1] = static_cast<unsigned char>((p >> 8) & 255);
          rgb[i * 3 + 2] = static_cast<unsigned char>((p >> 16) & 255);
        }
        vis::write_png_rgb("D:\\g4gpu\\out\\g4builder_selftest.png", rgb.data(), a.width,
                           a.height);
        std::printf("selftest: %d frames, %d solids, saved out/g4builder_selftest.png\n",
                    frame, static_cast<int>(a.model.solids.size()));
        break;
      }
    }
    DrawFrame(a);
    ++frame;
  }

  wglMakeCurrent(nullptr, nullptr);
  wglDeleteContext(a.hglrc);
  ReleaseDC(a.hwnd, a.hdc);
  DestroyWindow(a.hwnd);
  return 0;
}
