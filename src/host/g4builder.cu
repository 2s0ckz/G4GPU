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
#include "render/edges.h"
#include "render/float_geometry.cuh"
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
  /// The layer they CLASH ON, which is not necessarily either volume's own: a phantom on
  /// layer 2 whose bone class is on 3 clashes with a box on 3, and the number to report is 3.
  int layer = 0;
  /// The voxel class each side was in where they clashed, or -1 for a volume with no classes.
  /// A phantom has two hundred of them and "the phantom overlaps the seat" is not enough to
  /// act on; which class is.
  int class_a = -1;
  int class_b = -1;
  double mm3 = 0;       ///< estimated volume of the shared region
  /// That, as a fraction of the smaller solid - or negative when the smaller solid has no
  /// closed-form volume to divide by. Estimating one would mean sampling a solid whose
  /// containment test is a BVH parity count, on every edit, which is not worth a percentage.
  double fraction = -1;
};

/// What a picker pop-up is choosing. Defined here rather than in g4builder_pick.inc because
/// App holds one.
/// Widget id bases for the lists that grow with the model.
///
/// EVERY WIDGET IS AN INTEGER, and two widgets sharing one share ctx.hot, ctx.active and
/// ctx.focus - so a rename focuses the wrong row, and a drag is picked up by a control the
/// cursor is nowhere near. A base plus an index is the natural scheme and it fails silently
/// the moment a list is long enough to reach the next base:
///
///   * `400 + i` for the solid rows ran into the 410 "Assign material" button at ELEVEN
///     solids, which is a model anyone builds in an afternoon;
///   * `690 + i` for the source rows ran into the 700 kind buttons at eleven sources;
///   * `200 + i` and `300 + i` did the same to the element and scorer controls at ten;
///   * and the voxel classes were laid out as `4000 + solid*100 + class` while the importer
///     caps classes at 4096, so any phantom past a hundred classes collided with itself.
///
/// Nothing about any of that announces itself. The widget still draws, still highlights, and
/// misbehaves only in the bookkeeping - which is why it survived this long.
///
/// So the growing lists get a block each, a million apart, and the fixed controls keep the
/// low numbers they have always had. A million is not tight: it is room for a million solids
/// against 4096 classes each, and the arithmetic is checked in tests/test_ui_layout.cu.
enum : int {
  kIdElementRow = 1000000,
  kIdMaterialRow = 2000000,
  kIdScorerRow = 3000000,
  kIdSolidRow = 4000000,
  kIdSolidEye = 5000000,
  /// + solid * kIdClassStride + class. The stride is the importer's class cap, so a solid's
  /// classes cannot reach into the next solid's block however many of them it has.
  kIdClassRow = 6000000,
  kIdClassEye = 7000000,
  kIdSourceRow = 8000000,
  kIdPickRow = 9000000,
  /// + solid * kIdClassStride + class, like the two above: the layer menu on each class row.
  kIdClassLayer = 10000000,
  /// The layer menu on each SOLID row, one per solid. Above kIdClassLayer's whole span rather
  /// than a million past it: that block is strided by class, so at 200 solids it already
  /// reaches 10,819,200 and a million would not have cleared it. tests/test_ui_layout.cu
  /// enumerates both.
  kIdSolidLayer = 11000000,
  /// One block per solid inside kIdClassRow and kIdClassEye. Equal to the importer's
  /// kMaxClasses; tests/test_ui_layout.cu checks the two have not drifted apart.
  kIdClassStride = 4096,
};

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
  kWorld,
  /// The size-and-position form the insert menu opens before it inserts. See SeedPrimitive.
  kPrimitive,
  /// Confirming a change to the WORLD's layer. See SetSolidLayer.
  kWorldLayer
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
  /// Retrace the pixels where a surface begins or ends, and average, so silhouettes are not
  /// staircases. On by default. What it costs is measured in docs/VIS.md - it is not free, and
  /// it depends on how much of the frame is silhouette.
  bool antialias = true;
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
  /// Did the press that is currently down land on a splitter's grab strip?
  ///
  /// A grab strip STRADDLES the boundary it moves, so its inner half lies inside the 3D view
  /// - deliberately, so the cursor need not find the exact pixel. That makes those few
  /// columns claimable by two things at once, and dragging the left splitter did both: the
  /// view rotated as the panel resized.
  ///
  /// Only the left one showed it, and the asymmetry says why it is arbitration and not a
  /// stray pixel. `left_w = mouse_x` puts the view's first column exactly under the cursor,
  /// so the pointer is inside the view for the whole drag; `right_w = width - mouse_x` puts
  /// its EXCLUSIVE right edge there, so the pointer is just outside for the whole drag. The
  /// bottom splitter is the same as the right one. One rule was being applied consistently
  /// and one of the three geometries happened to escape it.
  ///
  /// Decided at the press rather than from `dragging_split`, which is only set once the frame
  /// runs: a WM_MOUSEMOVE carrying more than two pixels can arrive first, and then the orbit
  /// has already begun.
  bool press_on_split = false;
  bool running = true;
  bool resized = true;
  /// WHERE A FRAME'S TIME GOES, accumulated while -benchmesh is running and printed with the
  /// frame rate.
  ///
  /// "The GUI is slow with a large CAD file, even opening a dropdown" is a claim about the
  /// frame, and a frame is four things: the CUDA passes, the read-back of the rendered
  /// viewport, the UI drawn on the CPU, and the GL upload and swap. A total tells you the
  /// interaction is slow; the split tells you which one to fix. Zero cost when the flag is
  /// off, which it is unless a benchmark asked.
  bool time_phases = false;
  double t_cuda = 0, t_readback = 0, t_ui = 0, t_present = 0;
  int t_frames = 0;

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
  /// THE WIREFRAME EDGES, which the builder had none of. Its styles already said
  /// `solid = visible && !wireframe` - right, since a wireframe volume must not be ray cast as
  /// a surface - but nothing drew the edges, so such a volume was invisible, and the default
  /// world arrives with wireframe on and had no outline at all.
  float *d_ex0 = nullptr, *d_ey0 = nullptr, *d_ez0 = nullptr;
  float *d_ex1 = nullptr, *d_ey1 = nullptr, *d_ez1 = nullptr;
  unsigned int* d_ergb = nullptr;
  int n_edges = 0;
  /// Per-cell class index, and the class colour table the renderer looks up in. Both
  /// render-only: the transport reads the per-cell MATERIAL, which is -1 until a class has one
  /// assigned, so a phantom nobody has assigned yet would be invisible if the picture came
  /// from the same array. Assigning by looking at the picture is the whole workflow.
  short* d_voxel_class = nullptr;
  unsigned int* d_class_rgba = nullptr;
  /// How many entries d_class_rgba holds. Kept so the selftest can read back the array the
  /// kernel reads, rather than assuming a length from the class list of one solid.
  int class_rgba_n = 0;
  /// The options for a voxel class's layer menu, rebuilt each frame and held here.
  ///
  /// In App rather than in the drawing function because ui::Select records the array it was
  /// given and DrawOpenSelect reads it back at the END of the frame - a local would be gone
  /// by then. One list serves every row, since the choices do not depend on the class.
  std::vector<std::string> layer_opt_text;
  std::vector<const char*> layer_opt;
  /// The layer the world would be moved to, held while Popup::kWorldLayer asks. See
  /// SetSolidLayer for why moving the world is worth a question.
  int pending_world_layer = 0;
  /// The primitive Popup::kPrimitive is editing, before it is in the model. Seeded by
  /// SeedPrimitive; added by the form's Insert button and dropped by its Cancel.
  Solid pending_solid;
  /// What pfield_for holds while the insert form owns the parameter fields. Any value that is
  /// not a valid solid index re-seeds the solid form; this one has a name so that reading it
  /// says which of the two is using them.
  static constexpr int kPendingFields = -3;
  /// The float copy of the scene the render pass walks. See render/float_geometry.cuh: the
  /// transport stays double because the dose depends on it, and the picture does not.
  vis::FloatGeometry render_geom;
  unsigned long long* d_fb = nullptr;
  unsigned int* d_rgba = nullptr;
  /// The pixels the anti-aliasing pass should sample again, COMPACTED into a list, and how
  /// many there are. See vis::mark_edges: a per-pixel flag left the refinement running one
  /// lane per warp and cost fifty times what the rays are worth.
  int* d_edge = nullptr;
  unsigned int* d_edge_count = nullptr;
  unsigned int edge_marked = 0;   ///< read back under -benchmesh, to explain the cost
  /// The viewport size d_fb and d_rgba were actually allocated for. Compared against the
  /// current ViewRect every frame; see AllocViewportSurface for what went wrong without it.
  int fb_w = 0, fb_h = 0;
  std::vector<unsigned int> host_rgba;

  /// THE RENDER DOES NOT HOLD THE UI UP.
  ///
  /// The panels are rasterised by the CPU into host_rgba, which is also where the render is
  /// copied to - so the UI was structurally downstream of the render and a slow render was a
  /// slow GUI. Reported as a hover highlight arriving a second late with a large translucent
  /// CAD file loaded. The render now runs on its own stream and the frame composites the newest
  /// COMPLETED image, which in a slow scene is one render behind the camera. See DrawFrame.
  cudaStream_t render_stream = nullptr;
  cudaEvent_t render_done = nullptr;
  bool render_inflight = false;
  /// Two pinned staging buffers: the read-back writes one while the compositor reads the other,
  /// so adopting a finished image is a swap of an index and not a six-megabyte copy. PINNED
  /// because a cudaMemcpy2DAsync out of pageable memory is not actually asynchronous - it would
  /// have put the whole read-back back into the UI frame and looked like no change at all.
  unsigned int* pin_rgba[2] = {nullptr, nullptr};
  int pin_write = 0;             ///< the buffer the in-flight render is writing
  int pin_read = -1;             ///< the newest completed image, -1 before the first one lands
  int issue_w = 0, issue_h = 0;  ///< the viewport size the in-flight render was launched for
  int shown_w = 0, shown_h = 0;  ///< and the size of the image in pin_read
  /// Wait for the render rather than polling for it.
  ///
  /// Set under -selftest and -benchmesh, and for opposite reasons that need the same thing: the
  /// selftest's checksums compare a recolour against the frame after it, so the picture has to
  /// be THIS frame's; and the benchmark is measuring the render, which asynchronously is just a
  /// launch. Every number -benchmesh has ever printed was taken this way, so they stay
  /// comparable.
  bool sync_render = false;
  int render_frames = 0;   ///< completed renders, for -benchmesh to divide UI frames by
  /// Compare the composited viewport against the device image, and count the pixels that
  /// differ. Off by default; the selftest turns it on for a frame. See CountBlitMismatch, and
  /// see the selftest for why a checksum taken through the same blit could not do this.
  bool check_blit = false;
  long long blit_mismatch = 0;
  int blit_checked = 0;    ///< frames the comparison actually ran on, so zero cannot pass
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

  /// energy, pos xyz, half x, half y, radius, spread, then dir xyz.
  ui::NumberField src_field[11];
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
  /// The selected item's form, scrolled separately from the list it was selected in - see
  /// ui::SplitListForm for why the two cannot share one.
  ui::ScrollArea solid_form_scroll;
  /// Which widget the SOLIDS half had claimed by the time it finished drawing, for the
  /// selftest. Zero when the cursor is not over anything in that half.
  ///
  /// Reading ctx.hot at the END of the frame does not answer this: hot is whatever tested
  /// last, and the SOURCES half draws after SOLIDS, so a source row overwrites a ghost from
  /// above it. That is what made the first version of the check pass with the bug present.
  int hot_after_solids = 0;
  ui::ScrollArea source_form_scroll;
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

  // Which unit each field of the two forms is being read in. View state, not the document:
  // the model is in mm, degrees and MeV whatever these say, so a saved project does not
  // depend on them and reopening it in different units changes nothing but the display.
  //
  // Held per ROW and across selections, not reset with the selection, because someone
  // working in cm is working in cm for the whole session and would otherwise have it undone
  // by every click in the list. Row k of the shape parameters means the same kind of
  // quantity for every shape - ShapeParams says which - so the choice still applies.
  int p_unit[12] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  int pos_unit[3] = {1, 1, 1};
  int rot_unit[3] = {0, 0, 0};
  /// Parallel to src_field. The three direction components have no unit - a direction is a
  /// ratio - so their entries are never read.
  int src_unit[11] = {2, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0};
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
  /// What the import dialog has been told so far. CHOOSING A FILE NO LONGER IMPORTS IT: the
  /// dialog collects the voxel file, a colour table if there is one, and every parameter,
  /// and the Import button does the work. Before this, picking the file ran the import
  /// immediately, which meant the cell size and the type had to be right BEFORE the file
  /// browser was opened - and getting them wrong meant deleting the solid and starting over.
  std::string vox_file;
  std::string vox_ctbl;
  /// Index into builder::ColormapNames(). Only used when vox_ctbl is empty; a table that
  /// names the colours outranks a map that guesses them.
  int vox_colormap = 0;

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

/// Did a press at (@p x, @p y) land on a splitter's grab strip?
///
/// One function so that the message handler and the selftest ask the same question. The
/// answer decides who owns a press in the few columns where a grab strip and the 3D view
/// overlap - see App::press_on_split, which is where the overlap is explained.
static bool PressOnSplitter(const App& a, int x, int y) {
  return LeftSplitRect(a).Contains(x, y) || RightSplitRect(a).Contains(x, y)
         || BottomSplitRect(a).Contains(x, y);
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

  // Not through an open dropdown's list: it is painted over the splitter and the click
  // belongs to whatever row is under the cursor. Same rule as ui::Context::Hovering, which
  // these three cannot use - a splitter is not a widget and has no clip of its own.
  const bool covered = a.uic.block.Contains(a.input.mouse_x, a.input.mouse_y);
  if (a.dragging_split == 0 && a.input.left_pressed && !covered) {
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
  // +Z IS UP, and elevation lifts out of the x-y plane rather than out of x-z.
  //
  // Not a preference: a detector is described in beam coordinates, where z is the beam axis
  // and the transverse plane is x-y. A viewer with y up shows a linac gantry lying on its
  // side and a phantom's axial slices edge-on, and every dimension typed into the panels
  // then has to be mentally rotated to match what is on screen. It also makes
  // /vis/viewer/set/viewpointThetaPhi mean what Geant4 means by it, since that command's
  // theta is measured from +z.
  //
  // Elevation stays clamped short of straight down the axis, which is also what keeps
  // make_camera's cross(forward, up) from degenerating.
  const vis::Vec3f eye{a.target.x + a.distance * std::cos(el) * std::sin(a.azimuth),
                       a.target.y + a.distance * std::cos(el) * std::cos(a.azimuth),
                       a.target.z + a.distance * std::sin(el)};
  return vis::make_camera(eye, a.target, vis::Vec3f{0.f, 0.f, 1.f}, 45.0f,
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
/// Takes the finished render as the image to composite, and hands the other buffer to the next.
static void AdoptRender(App& a) {
  a.pin_read = a.pin_write;
  a.pin_write ^= 1;
  a.shown_w = a.issue_w;
  a.shown_h = a.issue_h;
  a.render_inflight = false;
  ++a.render_frames;
}

/// Waits for the render in flight, if there is one.
///
/// EVERYTHING THAT FREES WHAT A RENDER IS USING CALLS THIS FIRST, and each of them does it
/// itself rather than trusting the frame to have noticed: the viewport buffers
/// (AllocViewportSurface), the staging pair (AllocSurface), and the device geometry - this
/// function, which the Run button reaches through RunFromGui without going near a frame at all.
/// A free while a kernel is still reading is a device heap corruption that surfaces several
/// frames later somewhere unrelated, which is the one class of bug worth being this blunt about.
///
/// All three happen on a resize or an edit, and neither of those is what "even opening a
/// dropdown menu is extremely slow" was about - so waiting in them costs nothing that was
/// ever reported.
static void DrainRender(App& a) {
  if (!a.render_inflight) { return; }
  CUDA_CHECK(cudaEventSynchronize(a.render_done));
  CUDA_CHECK(cudaGetLastError());
  AdoptRender(a);
}

static void RebuildScene(App& a) {
  // Before anything below is freed.
  DrainRender(a);
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
  vis::EdgeList edges;
  for (std::size_t i = 0; i < scene.volumes.size(); ++i) {
    const auto& st = scene.styles[i];
    styles[i].r = static_cast<unsigned char>(st.r * 255);
    styles[i].g = static_cast<unsigned char>(st.g * 255);
    styles[i].b = static_cast<unsigned char>(st.b * 255);
    styles[i].a = static_cast<unsigned char>(st.opacity * 255 + 0.5f);
    styles[i].solid = st.visible && !st.wireframe;
    // And its outline, when it is a wireframe volume and visible. Dimmer than its own colour,
    // so a wireframe box behind a solid one reads as an outline rather than competing with it.
    if (st.visible && st.wireframe) {
      edges.AddVolume(scene.volumes[i],
                      ui::rgb(static_cast<int>(st.r * 160), static_cast<int>(st.g * 160),
                              static_cast<int>(st.b * 160)));
    }
  }
  CUDA_CHECK(cudaMalloc(&a.d_styles, sizeof(vis::VolumeStyle) * std::max<size_t>(1, styles.size())));
  if (!styles.empty()) {
    CUDA_CHECK(cudaMemcpy(a.d_styles, styles.data(), sizeof(vis::VolumeStyle) * styles.size(),
                          cudaMemcpyHostToDevice));
  }

  // The wireframe edges, uploaded with the styles because the flag that decides them is there.
  {
    float** dst[6] = {&a.d_ex0, &a.d_ey0, &a.d_ez0, &a.d_ex1, &a.d_ey1, &a.d_ez1};
    for (float** p : dst) {
      if (*p != nullptr) {
        cudaFree(*p);
        *p = nullptr;
      }
    }
    if (a.d_ergb != nullptr) {
      cudaFree(a.d_ergb);
      a.d_ergb = nullptr;
    }
    a.n_edges = static_cast<int>(edges.Size());
    if (a.n_edges > 0) {
      const std::size_t nb = sizeof(float) * a.n_edges;
      const std::vector<float>* src[6] = {&edges.x0, &edges.y0, &edges.z0,
                                          &edges.x1, &edges.y1, &edges.z1};
      for (int k = 0; k < 6; ++k) {
        CUDA_CHECK(cudaMalloc(dst[k], nb));
        CUDA_CHECK(cudaMemcpy(*dst[k], src[k]->data(), nb, cudaMemcpyHostToDevice));
      }
      CUDA_CHECK(cudaMalloc(&a.d_ergb, sizeof(unsigned int) * a.n_edges));
      CUDA_CHECK(cudaMemcpy(a.d_ergb, edges.rgb.data(), sizeof(unsigned int) * a.n_edges,
                            cudaMemcpyHostToDevice));
    }
  }

  // The two render-only voxel arrays. Freed and reuploaded with the scene, because both are
  // indexed by the pool offsets the flattening just chose.
  if (a.d_voxel_class != nullptr) {
    cudaFree(a.d_voxel_class);
    a.d_voxel_class = nullptr;
  }
  if (!scene.pool.voxel_class_cells.empty()) {
    CUDA_CHECK(cudaMalloc(&a.d_voxel_class,
                          sizeof(short) * scene.pool.voxel_class_cells.size()));
    CUDA_CHECK(cudaMemcpy(a.d_voxel_class, scene.pool.voxel_class_cells.data(),
                          sizeof(short) * scene.pool.voxel_class_cells.size(),
                          cudaMemcpyHostToDevice));
  }
  if (a.d_class_rgba != nullptr) {
    cudaFree(a.d_class_rgba);
    a.d_class_rgba = nullptr;
    a.class_rgba_n = 0;
  }
  if (!scene.pool.voxel_class_rgba.empty()) {
    CUDA_CHECK(cudaMalloc(&a.d_class_rgba,
                          sizeof(unsigned int) * scene.pool.voxel_class_rgba.size()));
    a.class_rgba_n = static_cast<int>(scene.pool.voxel_class_rgba.size());
    CUDA_CHECK(cudaMemcpy(a.d_class_rgba, scene.pool.voxel_class_rgba.data(),
                          sizeof(unsigned int) * scene.pool.voxel_class_rgba.size(),
                          cudaMemcpyHostToDevice));
  }

  // THE FLOAT COPY THE RENDER PASS WALKS.
  //
  // Built here because this is where the geometry changes, and from the HOST pools rather than
  // from the device ones the engine just uploaded - the conversion is arithmetic and the host
  // is where the doubles are. The voxel arrays are handed over as they are: cells are shorts
  // and class layers are ints, so both precisions read the same bytes.
  {
    vis::HostGeometry hg{};
    hg.volumes = scene.volumes.data();
    hg.n_volumes = static_cast<int>(scene.volumes.size());
    hg.world = scene.world;
    hg.solids = scene.pool.solids.data();
    hg.n_solids = static_cast<int>(scene.pool.solids.size());
    hg.xforms = scene.pool.xforms.data();
    hg.n_xforms = static_cast<int>(scene.pool.xforms.size());
    hg.aux = scene.pool.aux.data();
    hg.n_aux = static_cast<int>(scene.pool.aux.size());
    hg.tri = scene.pool.tri.data();
    hg.n_tri = static_cast<int>(scene.pool.tri.size());
    hg.bvh = scene.pool.bvh.data();
    hg.n_bvh = static_cast<int>(scene.pool.bvh.size());
    const auto& gd = a.engine->geometry();
    geom::VoxelStore<float> vf{};
    vf.material = gd.voxels.material;
    vf.count = gd.voxels.count;
    vf.cls = gd.voxels.cls;
    vf.class_layer = gd.voxels.class_layer;
    vf.class_absent = gd.voxels.class_absent;
    a.render_geom.Build(hg, vf);
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
  // Past the early return, so this is a real reallocation and there is something to free. See
  // DrainRender: a free while a kernel is reading is a corruption that surfaces elsewhere.
  DrainRender(a);
  if (a.d_fb != nullptr) { cudaFree(a.d_fb); }
  if (a.d_rgba != nullptr) { cudaFree(a.d_rgba); }
  if (a.d_edge != nullptr) { cudaFree(a.d_edge); }
  CUDA_CHECK(cudaMalloc(&a.d_fb, sizeof(unsigned long long) * w * h));
  CUDA_CHECK(cudaMalloc(&a.d_rgba, sizeof(unsigned int) * w * h));
  CUDA_CHECK(cudaMalloc(&a.d_edge, sizeof(int) * static_cast<std::size_t>(w) * h));
  if (a.d_edge_count == nullptr) { CUDA_CHECK(cudaMalloc(&a.d_edge_count, sizeof(unsigned int))); }
  a.fb_w = w;
  a.fb_h = h;
}

static void AllocSurface(App& a) {
  AllocViewportSurface(a);
  a.host_rgba.assign(static_cast<size_t>(a.width) * a.height, ui::theme::kPanel);

  // THE WINDOW'S SIZE, NOT THE VIEWPORT'S, and that is the whole reason this is here rather
  // than beside the device buffers in AllocViewportSurface. cudaHostAlloc pins pages and
  // cudaFreeHost unpins them, which costs milliseconds for six megabytes - and the viewport
  // changes size on every frame of a splitter drag, so allocating with it would have put a
  // stall into exactly the interaction this work is about removing. The viewport is never
  // larger than the window, so one allocation per window resize covers every viewport.
  // AllocViewportSurface may have returned early - the window resized, the viewport did not -
  // in which case nothing has drained yet and a read-back is still landing in one of these.
  DrainRender(a);
  for (int i = 0; i < 2; ++i) {
    if (a.pin_rgba[i] != nullptr) { CUDA_CHECK(cudaFreeHost(a.pin_rgba[i])); }
    CUDA_CHECK(cudaHostAlloc(&a.pin_rgba[i],
                             sizeof(unsigned int) * static_cast<size_t>(a.width) * a.height,
                             cudaHostAllocDefault));
  }
  a.pin_write = 0;
  a.pin_read = -1;
  a.shown_w = 0;
  a.shown_h = 0;

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

/// Milliseconds from the performance counter, for the phase timers.
static double NowMs() {
  static LARGE_INTEGER freq{};
  if (freq.QuadPart == 0) { QueryPerformanceFrequency(&freq); }
  LARGE_INTEGER n{};
  QueryPerformanceCounter(&n);
  return 1000.0 * static_cast<double>(n.QuadPart) / static_cast<double>(freq.QuadPart);
}

/// Launches the render for the current camera and returns WITHOUT waiting for it.
///
/// Everything goes on a.render_stream, including the read-back, so the only thing the caller
/// pays is the launch. cudaEventRecord at the end is what a later frame polls to find out
/// whether the image is ready.
static void IssueRender(App& a, int w, int h) {
  if (a.render_stream == nullptr) {
    // NON-BLOCKING, which is not the default. A stream from cudaStreamCreate is a BLOCKING
    // stream: it implicitly synchronises with the legacy default stream, so any ordinary
    // cudaMemcpy or cudaMalloc anywhere else in the frame would wait for the render - which is
    // the stall this is removing, reintroduced by the allocator.
    CUDA_CHECK(cudaStreamCreateWithFlags(&a.render_stream, cudaStreamNonBlocking));
    // No timing on the event: it is polled once a frame and never subtracted from another.
    CUDA_CHECK(cudaEventCreateWithFlags(&a.render_done, cudaEventDisableTiming));
  }
  const vis::Camera cam = CurrentCamera(a);
  const dim3 block(16, 16);
  const dim3 grid((w + 15) / 16, (h + 15) / 16);
  const vis::Palette pal = CurrentPalette(a);

  if (a.vis_attr.show_solids) {
    vis::render_geometry<float><<<grid, block, 0, a.render_stream>>>(
        a.render_geom.geometry(), a.d_styles, cam, a.d_fb, a.vis_attr.voxel_grid_lines,
        a.d_voxel_class, a.d_class_rgba);
  } else {
    CUDA_CHECK(cudaMemsetAsync(a.d_fb, 0xFF, sizeof(unsigned long long) * w * h,
                               a.render_stream));
  }
  // ANTI-ALIASING, AT THE EDGES ONLY.
  //
  // One ray per pixel puts a hard step wherever a surface ends: the pixel is either the
  // surface or it is not, so a silhouette becomes a staircase. Four rays per pixel everywhere
  // would fix it and cost four times the render - on a scene where the render is already the
  // expensive part - to improve the few per cent of pixels that are on an edge. So the edges
  // are found first (one cheap pass over the framebuffer, no geometry) and only those pixels
  // are traced again.
  //
  // BEFORE the trajectories, and that ordering matters: the track pass atomicMins into the
  // same framebuffer, and refining afterwards would retrace the geometry over the top of a
  // track and delete it.
  if (a.vis_attr.antialias && a.vis_attr.show_solids) {
    CUDA_CHECK(cudaMemsetAsync(a.d_edge_count, 0, sizeof(unsigned int), a.render_stream));
    vis::mark_edges<<<grid, block, 0, a.render_stream>>>(a.d_fb, w, h, a.d_edge,
                                                        a.d_edge_count, 10);
    // Sized for the worst case, one thread per pixel. See refine_edges: the count is on the
    // device and reading it back to size this launch would put a synchronisation in the middle
    // of a frame, which is what taking the render off the UI thread was for.
    vis::refine_edges<float><<<(w * h + 255) / 256, 256, 0, a.render_stream>>>(
        a.render_geom.geometry(), a.d_styles, cam, a.d_fb, a.d_edge, a.d_edge_count,
        a.vis_attr.voxel_grid_lines, a.d_voxel_class, a.d_class_rgba);
  }
  // The wireframe outlines. AFTER the anti-aliasing, because refine_edges retraces the
  // geometry at the pixels it touches and would erase a line drawn under it; and these are
  // lines, one pixel wide by intent, so there is nothing in them to anti-alias.
  if (a.vis_attr.show_wireframe && a.n_edges > 0) {
    vis::render_edges<<<(a.n_edges + 63) / 64, 64, 0, a.render_stream>>>(
        a.d_ex0, a.d_ey0, a.d_ez0, a.d_ex1, a.d_ey1, a.d_ez1, a.d_ergb, a.n_edges, cam,
        a.d_fb, 0);
  }
  if (a.show_tracks && a.n_segments > 0) {
    const int thick = static_cast<int>(a.vis_attr.track_width) - 1;
    vis::render_trajectories<<<(a.n_segments + 127) / 128, 128, 0, a.render_stream>>>(
        a.traj, a.n_segments, cam, a.d_fb, thick > 0 ? thick : 0, 1e-3f, pal);
  }
  vis::resolve_to_rgba<<<grid, block, 0, a.render_stream>>>(a.d_fb, a.d_rgba, w, h, pal);
  // Into the staging buffer the compositor is NOT reading, tightly packed. The old copy went
  // straight into host_rgba with the window's pitch as the destination stride, which cannot be
  // done asynchronously: host_rgba is pageable and the panels are about to be drawn over it.
  CUDA_CHECK(cudaMemcpy2DAsync(a.pin_rgba[a.pin_write], sizeof(unsigned int) * w, a.d_rgba,
                               sizeof(unsigned int) * w, sizeof(unsigned int) * w, h,
                               cudaMemcpyDeviceToHost, a.render_stream));
  CUDA_CHECK(cudaEventRecord(a.render_done, a.render_stream));
  // Immediately, so a bad launch configuration is reported here and not blamed on the frame
  // that eventually waits for the event.
  CUDA_CHECK(cudaGetLastError());
  a.issue_w = w;
  a.issue_h = h;
  a.render_inflight = true;
}

static void DrawFrame(App& a) {
  const double t0 = a.time_phases ? NowMs() : 0.0;

  // No drain here on purpose. AllocSurface, AllocViewportSurface and RebuildScene each wait for
  // the render themselves, because each of them frees something it is using - see DrainRender.
  // A frame that has to work out which of them is about to run is a frame that will get it
  // wrong the first time a fourth one is added.
  if (a.resized) { AllocSurface(a); }
  // Every frame, because a splitter drag resizes the viewport without resizing the window.
  // It returns immediately when the size has not moved, so this costs a comparison.
  AllocViewportSurface(a);
  if (a.scene_dirty) { RebuildScene(a); }

  const ui::Rect v = ViewRect(a);
  const int w = std::max(1, v.w), h = std::max(1, v.h);

  // POLL, DO NOT WAIT.
  //
  // This is the whole of the decoupling. The frame asks whether the render it started earlier
  // has finished; if it has, that image becomes the one to composite and the next render goes
  // out; if it has not, the frame composites the PREVIOUS one and carries on. The UI runs at
  // the refresh rate whatever the render costs, and a slow scene shows as a viewport that
  // updates less often rather than as a window that stops responding.
  if (a.render_inflight) {
    const cudaError_t st = cudaEventQuery(a.render_done);
    if (st == cudaSuccess) {
      AdoptRender(a);
    } else if (st != cudaErrorNotReady) {
      CUDA_CHECK(st);
    }
  }
  if (!a.render_inflight) { IssueRender(a, w, h); }
  // And -selftest and -benchmesh do wait, so their picture is this frame's. See App::sync_render.
  if (a.sync_render) { DrainRender(a); }
  const double t1 = a.time_phases ? NowMs() : 0.0;

  // Background for the panels, then the render blitted into the viewport rectangle.
  //
  // ONE STRIDED COPY, NOT ONE PER ROW.
  //
  // This was a loop of cudaMemcpy, one per scanline. Each carries a fixed launch and
  // synchronisation cost of order ten microseconds whatever it moves, so 960 rows cost about
  // 16 ms - measured, and flat against triangle count, which is what gave it away. That is the
  // whole of the frame budget at 60 Hz spent on a copy of six megabytes that the hardware does
  // in under a millisecond, and it was charged to EVERY frame: opening a dropdown, typing in a
  // field, moving the mouse. Reported as the GUI being slow with a large CAD file loaded, which
  // it was - but it was slow with an empty scene too, and that is the part the report made
  // findable.
  //
  // cudaMemcpy2D takes the two pitches and does it in one call. The viewport is a window into
  // a wider host buffer, hence the destination pitch of the whole row.
  std::fill(a.host_rgba.begin(), a.host_rgba.end(), ui::theme::kPanel);
  if (a.pin_read >= 0) {
    // Only the overlap, because the viewport may have been resized since this image was
    // launched: the rest stays panel colour until the next render lands. A splitter drag shows
    // an image that is briefly the wrong size, which is the trade being made on purpose - the
    // alternative is waiting, and waiting is the thing being removed.
    const int cw = (w < a.shown_w) ? w : a.shown_w;
    const int ch = (h < a.shown_h) ? h : a.shown_h;
    for (int y = 0; y < ch; ++y) {
      std::memcpy(&a.host_rgba[static_cast<size_t>(v.y + y) * a.width + v.x],
                  &a.pin_rgba[a.pin_read][static_cast<size_t>(y) * a.shown_w],
                  sizeof(unsigned int) * static_cast<size_t>(cw));
    }
    // AGAINST THE DEVICE IMAGE, not against another frame's copy of it.
    //
    // Here and not later: the panels and the gizmo are drawn over host_rgba next, and the
    // gizmo is drawn INSIDE the viewport - so a comparison after them reports the gizmo as a
    // mismatch. And only in sync mode, where d_rgba still holds the image that was just
    // adopted; asynchronously the next render has already been launched over it.
    if (a.check_blit && a.sync_render && cw == w && ch == h) {
      std::vector<unsigned int> dev(static_cast<size_t>(w) * h);
      CUDA_CHECK(cudaMemcpy(dev.data(), a.d_rgba, sizeof(unsigned int) * dev.size(),
                            cudaMemcpyDeviceToHost));
      for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
          if (dev[static_cast<size_t>(y) * w + x]
              != a.host_rgba[static_cast<size_t>(v.y + y) * a.width + v.x + x]) {
            ++a.blit_mismatch;
          }
        }
      }
      ++a.blit_checked;
    }
  }

  const double t2 = a.time_phases ? NowMs() : 0.0;

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
  const double t3 = a.time_phases ? NowMs() : 0.0;

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
  if (a.time_phases) {
    const double t4 = NowMs();
    a.t_cuda += t1 - t0;
    a.t_readback += t2 - t1;
    a.t_ui += t3 - t2;
    a.t_present += t4 - t3;
    ++a.t_frames;
  }
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
      // A press on a splitter is a resize and nothing else. See App::press_on_split.
      a.press_on_split = PressOnSplitter(a, static_cast<short>(LOWORD(lp)),
                                         static_cast<short>(HIWORD(lp)));
      SetCapture(hwnd);
      return 0;
    case WM_LBUTTONUP:
      a.input.left_down = false;
      a.input.left_released = true;
      a.orbiting = false;
      a.press_on_split = false;
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
      // camera by a pixel. Nor when the press was on a splitter, and nor under an open
      // dropdown - both are things drawn over the view that own their own drags.
      if (a.input.left_down && over_view && !a.drag.active && a.uic.active == 0
          && !a.press_on_split && !a.uic.block.Contains(x, y)
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
  /// -benchmesh N: import an N-triangle sphere, render a fixed number of frames, and print
  /// milliseconds per frame. A performance claim about the viewer needs a number from the
  /// viewer, not from a micro-benchmark of one function.
  int bench_mesh = 0;
  double bench_opacity = 1.0;
  int bench_shells = 1;
  bool bench_aa = true;
  std::string open_path;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "-benchmesh") == 0) {
      bench_mesh = (i + 1 < argc) ? std::atoi(argv[i + 1]) : 200000;
      // An optional second number: the opacity to place it at. A translucent volume does not
      // let the front-to-back walk stop at its first surface, so it is a different
      // measurement and it is the one a report about a slow GUI turned out to be about.
      if (i + 2 < argc && argv[i + 2][0] != '-') { bench_opacity = std::atof(argv[i + 2]); }
      // And a third: how many concentric shells the one mesh holds, which is how many
      // surfaces a ray crosses. See InsertBenchMesh for why one sphere is not representative.
      if (i + 3 < argc && argv[i + 3][0] != '-') { bench_shells = std::atoi(argv[i + 3]); }
      // And a fourth, 0 or 1: whether to anti-alias. Here so that the two arms can be measured
      // without a rebuild between them, which is the only way to tell the cost of the extra
      // rays from the cost of the codegen change that lifting trace_pixel out caused.
      if (i + 4 < argc && argv[i + 4][0] != '-') {
        bench_aa = (std::atoi(argv[i + 4]) != 0);
      }
      continue;
    }
    if (std::strcmp(argv[i], "-selftest") == 0) {
      selftest_frames = 106;
    } else if (std::strcmp(argv[i], "-w") == 0 && i + 1 < argc) {
      a.width = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "-h") == 0 && i + 1 < argc) {
      a.height = std::atoi(argv[++i]);
    } else if (argv[i][0] != '-') {
      open_path = argv[i];
    }
  }

  // NEITHER BATCH MODE POLLS. The selftest's checksums compare a recolour against the frame
  // after it, so its picture has to be that frame's; and the benchmark is measuring the render,
  // which asynchronously is a launch and nothing else. -benchmesh turns it off again halfway,
  // to measure both - see below.
  a.sync_render = (selftest_frames > 0 || bench_mesh > 0);
  if (bench_mesh > 0) { a.vis_attr.antialias = bench_aa; }

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

    // -benchmesh: import a mesh of the requested size, aim at it, and time the frames.
    //
    // Its own block rather than a selftest frame, because it must not be entangled with the
    // selftest's model - and because the number it prints is the one to quote when claiming
    // the viewer got faster. A micro-benchmark of one function is evidence about that
    // function; this is evidence about the viewer.
    if (bench_mesh > 0) {
      static int bench_actual = 0;
      if (frame == 3) {
        // The generator lands on the next whole UV sphere, so what it built is what gets
        // reported - quoting the request would put a number in the log that is not the number
        // that was rendered.
        bench_actual = InsertBenchMesh(a, bench_mesh, 60.0, bench_opacity, bench_shells);
        a.scene_dirty = true;
      }
      if (frame == 6) {
        // Close enough that the mesh fills the view, so the rays actually reach triangles. A
        // benchmark of a mesh off screen measures the box test that rejects it.
        a.distance = 200.0f;
        a.azimuth = 0.7f;
        a.elevation = 0.35f;
      }
      if (frame >= 9) {
        // From here and not earlier: the frames before this built the scene, and a BVH over a
        // million triangles charged to "cuda" would have said the render was the problem.
        a.time_phases = true;
        // TWO PHASES IN ONE PROCESS, which is what makes the comparison paired: the same scene,
        // the same camera, the same binary, the same forty frames, one flag different. Phase 0
        // waits for the render inside the frame, which is what the builder used to do and what
        // every earlier number here was taken with. Phase 1 polls for it.
        static double t_sum[2] = {0, 0};
        static int t_n[2] = {0, 0};
        static double cuda_ms[2] = {0, 0};
        static int phase = 0;
        static LARGE_INTEGER freq{}, prev{};
        if (t_n[phase] == 0) {
          QueryPerformanceFrequency(&freq);
          QueryPerformanceCounter(&prev);
        } else {
          LARGE_INTEGER now{};
          QueryPerformanceCounter(&now);
          t_sum[phase] += 1000.0 * static_cast<double>(now.QuadPart - prev.QuadPart)
                          / static_cast<double>(freq.QuadPart);
          prev = now;
        }
        ++t_n[phase];
        // Bounded by frames as well as by samples, so a run that has nothing to time - an
        // import that failed, a mesh that never appeared - still leaves rather than sitting
        // in the message loop waiting for a frame count it will never reach. This is a batch
        // mode; it has no window anyone is watching.
        if (t_n[phase] > 40 || frame > 200 + 40 * phase) {
          if (a.t_frames > 0) { cuda_ms[phase] = a.t_cuda / a.t_frames; }
          if (a.d_edge_count != nullptr) {
            CUDA_CHECK(cudaMemcpy(&a.edge_marked, a.d_edge_count, sizeof(unsigned int),
                                  cudaMemcpyDeviceToHost));
          }
          if (phase == 0) {
            phase = 1;
            a.sync_render = false;
            a.t_cuda = a.t_readback = a.t_ui = a.t_present = 0;
            a.t_frames = 0;
            a.render_frames = 0;
          } else {
            // Both bounds can be reached with a single sample - an import that failed, a mesh
            // that never appeared - and a mean over zero intervals is not a number. Say so
            // rather than printing inf and letting the checks below read it as "fast".
            // A mean over zero intervals is not a number, and both bounds can be reached with
            // a single sample - an import that failed, a mesh that never appeared. Say so
            // rather than printing inf and letting the checks below read it as "fast".
            const int n0 = t_n[0] - 1, n1 = t_n[1] - 1;
            a.running = false;
            if (n0 < 1 || n1 < 1) {
              std::printf("benchmesh: FATAL: too few frames to time (%d, %d)\n", t_n[0],
                          t_n[1]);
              std::fflush(stdout);
            } else {
              const double sync_ms = t_sum[0] / n0;
              const double async_ms = t_sum[1] / n1;
              std::printf("benchmesh: %d triangles, opacity %.2f, %d shells at %dx%d\n",
                          bench_actual, bench_opacity, bench_shells, a.width, a.height);
              std::printf("  render inside the UI frame: %6.2f ms per frame (%5.1f fps), "
                          "cuda %.2f ms\n",
                          sync_ms, 1000.0 / sync_ms, cuda_ms[0]);
              {
                const double px_total = static_cast<double>(a.width) * a.height;
                std::printf("  anti-aliasing: %s, %u pixels marked as edges (%.1f%% of "
                            "the viewport)\n",
                            a.vis_attr.antialias ? "on" : "off", a.edge_marked,
                            (px_total > 0) ? 100.0 * a.edge_marked / px_total : 0.0);
              }
              std::printf("  render on its own stream:   %6.2f ms per frame (%5.1f fps), "
                          "cuda %.2f ms, %.2f renders per UI frame\n",
                          async_ms, 1000.0 / async_ms, cuda_ms[1],
                          static_cast<double>(a.render_frames) / n1);
              // THE STRUCTURAL INVARIANT, and it does not need a slow scene to bite. cuda here
              // is the time the UI frame spends between starting the render and moving on,
              // which asynchronously is a launch and a poll of an event: under a millisecond
              // on any scene at all. Put a cudaDeviceSynchronize back into DrawFrame and this
              // becomes the render time instead, whatever the picture looks like. Asserted
              // separately from the frame-time comparison below, because that one only says
              // anything when the render is slower than the refresh interval and most scenes
              // are not.
              std::printf("benchmesh: %s: the UI frame spent %.2f ms on a render costing "
                          "%.2f ms\n",
                          (cuda_ms[1] < 2.0) ? "render is off the UI frame"
                                             : "RENDER IS STILL ON THE UI FRAME",
                          cuda_ms[1], cuda_ms[0]);
              // And the payoff, checked by the run that claims it. A fast scene passes by
              // saying there was nothing to decouple, which is true and is not a pass smuggled
              // in: the invariant above is what holds everywhere.
              if (sync_ms > 18.0) {
                std::printf("benchmesh: %s: a render slower than the refresh (%.1f ms) left "
                            "the UI at %.1f ms\n",
                            (async_ms < sync_ms * 0.9) ? "decoupled" : "NOT DECOUPLED",
                            sync_ms, async_ms);
              } else {
                std::printf("benchmesh: decoupled: nothing to decouple, both frames are "
                            "vsync-limited at %.1f ms\n", sync_ms);
              }
              std::fflush(stdout);
            }
          }
        }
      }
    }

    static unsigned long long async_want = 0;
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
      // Before the run and before the project is saved, so both sides see it.
      if (frame == 13) { SelftestPhantomClassLayers(a); }
      if (frame == 13) { SelftestCheckClassOverlapRefusal(a); }
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
        // A chosen file and a chosen table, so the photograph shows those rows as a user
        // sees them rather than as three empty prompts - and shows the Import button
        // enabled, which is the state the dialog exists to reach. Nothing is read: these are
        // strings the dialog displays, and Import is not pressed.
        a.vox_file = "D:/phantom/adult_male_1mm.raw";
        a.vox_ctbl = "D:/phantom/icrp110_colours.txt";
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
      // A colour table, from a file, all the way to the bytes the render kernel reads.
      //
      // The parser has its own test - tests/test_voxel_import.cu - and this is the other half:
      // that what it wrote into the class list survives being flattened into the solid pool
      // and uploaded. Those are three separate arrays with three separate offsets, and a
      // table that parses perfectly and lands nowhere looks exactly like one that works.
      //
      // The file dialog is not driven: it is modal, and there is nothing here to answer it.
      // So this calls the reader the button calls, with a file written here.
      if (frame == 41) {
        int vs = -1;
        for (std::size_t i = 0; i < a.model.solids.size(); ++i) {
          if (!a.model.solids[i].voxel_classes.empty()) {
            vs = static_cast<int>(i);
            break;
          }
        }
        if (vs < 0) {
          std::printf("selftest: FAILED - no voxel volume to color from a table\n");
        } else {
          Solid& s = a.model.solids[static_cast<std::size_t>(vs)];
          // Row 0 as bytes, row 1 as fractions, so both halves of the scale rule travel.
          const std::string tp = "D:/g4gpu/out/selftest_colour_table.txt";
          std::FILE* tf = std::fopen(tp.c_str(), "wb");
          if (tf != nullptr) {
            std::fprintf(tf, "# selftest\n%g 255 0 0 255\n",
                         s.voxel_classes[0].value);
            if (s.voxel_classes.size() > 1) {
              std::fprintf(tf, "%g 0.0 1.0 0.0 0.5\n", s.voxel_classes[1].value);
            }
            std::fclose(tf);
          }
          std::string note;
          const ColourTableResult res = ReadVoxelColourTable(tp, s, note);
          std::printf("selftest: color table - %s\n", note.c_str());
          const bool red = res.ok && s.voxel_classes[0].r == 1.0f
                           && s.voxel_classes[0].g == 0.0f
                           && s.voxel_classes[0].opacity == 1.0f;
          if (!red) {
            std::printf("selftest: FAILED - the table did not reach the class list\n");
          }
          a.sel_solid = vs;
          a.model.Touch();
          a.scene_dirty = true;
        }
      }
      // The scene was rebuilt at the top of this frame, so the device array is the new one.
      if (frame == 42) {
        if (a.d_class_rgba == nullptr || a.class_rgba_n <= 0) {
          std::printf("selftest: FAILED - no class colours were uploaded\n");
        } else {
          std::vector<unsigned int> back(static_cast<std::size_t>(a.class_rgba_n));
          CUDA_CHECK(cudaMemcpy(back.data(), a.d_class_rgba,
                                sizeof(unsigned int) * back.size(), cudaMemcpyDeviceToHost));
          // 0xAARRGGBB, as build_scene packs it: opaque pure red.
          bool found = false;
          for (unsigned int v : back) {
            if (v == 0xFFFF0000u) { found = true; }
          }
          std::printf(found
                          ? "selftest: the colour table reached the render array\n"
                          : "selftest: FAILED - opaque red is not in the uploaded colours\n");
        }
        SaveFramePng(a, "D:/g4gpu/out/g4builder_colour_table.png");
        // And the control itself, which lives below the solid's numbers and so is off the
        // bottom of the section until it is scrolled to. A photograph rather than an
        // assertion: what is being checked is that a button is drawn where it was placed and
        // is legible next to its neighbours, and there is no way to assert legible.
        a.solids_scroll.offset = a.solids_scroll.content_h;
      }
      if (frame == 43) {
        SaveFramePng(a, "D:/g4gpu/out/g4builder_colour_table_button.png");
      }
      // The colormap list, open. It is the longest dropdown in any of these dialogs, so it
      // is the one most likely to run off the bottom of the pop-up that owns it - the defect
      // class of docs/RISK.md V14. The colour table is cleared first, because the menu is
      // drawn inert while a table overrides it and an inert menu does not open.
      if (frame == 44) {
        SeedVoxelDialog(a);
        a.vox_guess_dims = false;
        a.vox_file = "D:/phantom/adult_male_1mm.raw";
        a.vox_ctbl.clear();
        a.popup = Popup::kImportVoxel;
      }
      if (frame == 45) { a.uic.open_select = 1280; }
      if (frame == 46) {
        SaveFramePng(a, "D:/g4gpu/out/g4builder_dlg_voxel_cmap.png");
        a.uic.open_select = 0;
        a.popup = Popup::kNone;
        a.vox_file.clear();
        a.vox_dialog_seeded = false;
      }
      // A WIDGET SCROLLED OUT OF ITS SECTION MUST NOT CLAIM THE CURSOR.
      //
      // The solid form is taller than the half it lives in - a voxel volume's is a dozen
      // fields and six buttons - so its tail is clipped away below the section's bottom edge.
      // Those buttons kept their rectangles, and their rectangles lie over the SOURCES
      // section underneath, so hovering a source lit up a button that was not on screen and
      // clicking one opened the material picker belonging to it.
      //
      // Checked through ctx.hot rather than by clicking: hot is set by whichever widget the
      // cursor is over, which is the thing that was wrong, and reading it costs no side
      // effect. Begin clears it at the top of each frame, so frame 48 sees what frame 47's
      // draw decided. Solid-form widgets are ids 400 to 659; the sources section starts at
      // 680, so the two ranges say which section answered.
      if (frame == 47) {
        for (std::size_t i = 0; i < a.model.solids.size(); ++i) {
          if (!a.model.solids[i].voxel_classes.empty()) {
            a.sel_solid = static_cast<int>(i);
            a.pfield_for = -2;
            break;
          }
        }
        a.solid_form_scroll.offset = 0;   // so the tail of the form is below the fold
        // Into the SOURCES half, a little below where the solid form is cut off.
        a.input.mouse_x = a.width - a.right_w / 2;
        a.input.mouse_y = kMenuH + (a.height - kMenuH - kStatusH) / 2 + 40;
      }
      if (frame == 48) {
        // Sampled where the SOLIDS half finished, not at the end of the frame.
        const int hot = a.hot_after_solids;
        if (hot != 0) {
          std::printf("selftest: FAILED - widget %d in the SOLIDS half claimed the cursor at "
                      "(%d,%d), which is in SOURCES\n",
                      hot, a.input.mouse_x, a.input.mouse_y);
        } else {
          std::printf("selftest: nothing scrolled out of SOLIDS claims the cursor over "
                      "SOURCES\n");
        }
        a.input.mouse_x = 0;
        a.input.mouse_y = 0;
      }
      // WHAT A HIGHER LAYER COVERS CANNOT CHANGE THE PICTURE.
      //
      // The cell march used to walk a voxel volume's whole depth in one go, with no ownership
      // test along it - so cells sitting inside an opaque volume placed over the phantom were
      // composited anyway, and being nearer the eye they were blended ON TOP of that volume's
      // own surface. Reported as a phantom showing through a solid object in front of it.
      //
      // Checked as an invariant rather than by inspecting a pixel: recolour a class the slab
      // owns the space of, and NOTHING may change.
      //
      // With three positive controls, because "nothing changed" is what several other kinds
      // of broken look like too:
      //
      //   * recolouring the class in FRONT of the slab must change the picture, or the
      //     fixture is off screen and the invariant is about nothing;
      //   * recolouring the SLAB must change it, or the slab is not being painted - which
      //     would satisfy the invariant for the wrong reason, and would mean the search steps
      //     over the very surface it stopped for;
      //   * and once the slab is made translucent, recolouring the class BEYOND it must
      //     change the picture. That is the other half of the rule: the march has to resume
      //     past the covering volume rather than end there, or everything behind a
      //     transparent object placed over a phantom disappears.
      if (frame == 49) {
        InsertSelftestCoveredVoxels(a);
        // NEARLY straight down -z, so every ray that reaches the far side of the grid crosses
        // the slab: the eye is close to the grid's axis and the slab is wider than the grid,
        // so a ray is closer to the axis at the slab than at anything behind it.
        //
        // Elevation 1.5 and not 0, because +z is up: elevation lifts out of the x-y plane, so
        // looking down the z axis is the top of the range rather than the middle of it. 1.5 rad
        // is the clamp CurrentCamera applies, 4.1 degrees off the axis - which is deliberate,
        // since exactly on it is where cross(forward, up) has no answer. The margin is not
        // tight: the slab overhangs the grid by 20 mm on each side and 4.1 degrees over the
        // grid's 80 mm depth is a lateral 5.7 mm.
        //
        // This check reads the geometry rather than a flag, so it is the one place in the
        // selftest that the up direction could quietly invalidate - and it did, by pointing
        // the camera ACROSS the classes the slab is meant to hide instead of through them.
        a.target = vis::Vec3f{0.f, 250.f, 0.f};
        a.distance = 260.f;
        a.azimuth = 0.f;
        a.elevation = 1.5f;
      }
      static unsigned long long cover_base = 0;
      static unsigned long long cover_hidden = 0;
      if (frame == 50) {
        cover_base = ViewportChecksum(a);
        SaveFramePng(a, "D:/g4gpu/out/g4builder_covered_voxels.png");
        RecolourVoxelClass(a, "Covered", 2, 0.1f, 0.9f, 0.2f);   // inside the slab
      }
      static unsigned long long cover_beyond_opaque = 0;
      if (frame == 51) {
        cover_hidden = ViewportChecksum(a);
        RecolourVoxelClass(a, "Covered", 2, 0.9f, 0.2f, 0.9f);   // back as it was
        RecolourVoxelClass(a, "Covered", 3, 0.9f, 0.7f, 0.1f);   // BEYOND the opaque slab
      }
      static unsigned long long cover_seen = 0;
      if (frame == 52) {
        cover_beyond_opaque = ViewportChecksum(a);
        RecolourVoxelClass(a, "Covered", 3, 0.3f, 0.3f, 0.8f);   // back as it was
        RecolourVoxelClass(a, "Covered", 1, 0.2f, 0.4f, 0.9f);   // IN FRONT of the slab
      }
      if (frame == 53) {
        cover_seen = ViewportChecksum(a);
        RecolourVoxelClass(a, "Covered", 1, 0.55f, 0.75f, 0.55f);  // back as it was
        RecolourSolid(a, "Cover", 0.2f, 0.3f, 0.9f);               // and the SLAB itself
      }
      static unsigned long long cover_translucent = 0;
      if (frame == 54) {
        const unsigned long long cover_box = ViewportChecksum(a);
        if (cover_hidden != cover_base) {
          std::printf("selftest: FAILED - recolouring a voxel class whose space a "
                      "higher-layer volume owns changed the picture (%llu -> %llu), so "
                      "covered cells are still being drawn\n",
                      cover_base, cover_hidden);
        } else if (cover_beyond_opaque != cover_base) {
          std::printf("selftest: FAILED - recolouring a voxel class BEYOND an opaque "
                      "higher-layer volume changed the picture (%llu -> %llu)\n",
                      cover_base, cover_beyond_opaque);
        } else if (cover_seen == cover_base) {
          std::printf("selftest: FAILED - recolouring the voxel class in FRONT of the slab "
                      "changed nothing (%llu), so the checks above proved nothing\n",
                      cover_base);
        } else if (cover_box == cover_base) {
          std::printf("selftest: FAILED - recolouring the covering slab changed nothing "
                      "(%llu), so it is not being drawn and the invariant above is satisfied "
                      "by the wrong thing\n", cover_base);
        } else {
          std::printf("selftest: what a higher layer covers does not reach the picture - not "
                      "the cells inside it, not the cells behind it - and the cover and the "
                      "cells in front of it do\n");
        }
        RecolourSolid(a, "Cover", 0.85f, 0.85f, 0.35f);
        SetSolidOpacity(a, "Cover", 0.4f);   // now see through it
      }
      if (frame == 55) {
        cover_translucent = ViewportChecksum(a);
        RecolourVoxelClass(a, "Covered", 3, 0.9f, 0.7f, 0.1f);   // beyond a see-through slab
      }
      if (frame == 56) {
        const unsigned long long beyond_seen = ViewportChecksum(a);
        if (beyond_seen == cover_translucent) {
          std::printf("selftest: FAILED - with the slab translucent, recolouring the voxel "
                      "class BEYOND it changed nothing (%llu), so the cell march stops at a "
                      "covering volume instead of resuming past it\n", cover_translucent);
        } else {
          std::printf("selftest: with the covering volume made translucent, the cells beyond "
                      "it are drawn again\n");
        }
        RecolourVoxelClass(a, "Covered", 3, 0.3f, 0.3f, 0.8f);
        SetSolidOpacity(a, "Cover", 1.0f);
        // The splitter grab strips overlap the 3D view by design, so a press there is claimed
        // by two things at once unless something arbitrates. PressOnSplitter is that, and
        // this is the pixel it has to get right: the first column of the view.
        const ui::Rect vr = ViewRect(a);
        const int my = vr.y + vr.h / 2;
        const bool ambiguous = vr.Contains(vr.x, my);
        const bool claimed = PressOnSplitter(a, vr.x, my);
        const bool right_claimed = PressOnSplitter(a, vr.x + vr.w - 1, my);
        const bool middle_free = !PressOnSplitter(a, vr.x + vr.w / 2, my);
        if (!ambiguous || !claimed || !right_claimed || !middle_free) {
          std::printf("selftest: FAILED - splitter arbitration: view-owns-first-column %d, "
                      "splitter-claims-it %d, right edge %d, middle free %d\n",
                      ambiguous, claimed, right_claimed, middle_free);
        } else {
          std::printf("selftest: the view's edge columns are claimed by the splitters, not by "
                      "the camera\n");
        }
        // The camera back, so the overview picture at the end is the same one it has
        // always been.
        a.target = vis::Vec3f{95.f, 0.f, 0.f};
        a.distance = 900.f;
        a.azimuth = 0.9f;
        a.elevation = 0.25f;
      }
      // THE CLASS ROWS, AND WHAT THE VOLUME'S OWN FIELDS DO TO THEM.
      //
      // Three things at once, because they are one screen: the layer menu on each class row,
      // the two columns that keep a material visible beside a long name, and the two "for
      // all" rules that replaced a button.
      //
      // The rules are called rather than clicked - SetSolidMaterial and SetSolidLayer exist as
      // functions for exactly this reason, since a selftest cannot synthesise a click on a
      // pop-up's list row.
      if (frame == 57) {
        int ph = -1;
        for (std::size_t i = 0; i < a.model.solids.size(); ++i) {
          if (a.model.solids[i].name == "Phantom") { ph = static_cast<int>(i); }
        }
        if (ph < 0) {
          std::printf("selftest: FAILED - no Phantom to open\n");
        } else {
          Solid& p = a.model.solids[static_cast<std::size_t>(ph)];
          // A name far longer than the column, which is the reported case: an imported
          // phantom is called after its file.
          p.name = "adult_male_1mm_segmented_raw";
          p.ui_expanded = true;
          a.sel_solid = ph;
          a.pfield_for = -2;
          // And one class on a layer of its own, so the menu has something to show that is
          // not the volume's.
          if (p.voxel_classes.size() > 1) { p.voxel_classes[1].layer = p.layer + 3; }
          a.model.Touch();
          a.scene_dirty = true;
        }
      }
      if (frame == 58) {
        SaveFramePng(a, "D:/g4gpu/out/g4builder_class_rows.png");
        int ph = -1;
        for (std::size_t i = 0; i < a.model.solids.size(); ++i) {
          if (a.model.solids[i].name == "adult_male_1mm_segmented_raw") {
            ph = static_cast<int>(i);
          }
        }
        if (ph < 0) {
          std::printf("selftest: FAILED - the renamed Phantom is gone\n");
        } else {
          const Solid& p = a.model.solids[static_cast<std::size_t>(ph)];
          const int n_cls = static_cast<int>(p.voxel_classes.size());

          // The volume's MATERIAL is every class's. Deliberately a material the classes do
          // not already have, or the check would pass without the rule.
          // Any material the classes do not already have. Named by that property rather than
          // by looking for a bone: the first attempt asked for the last material in the list
          // and got the one every class already carried, which the guard below caught and
          // which is how this ended up specified rather than guessed.
          const int have = p.voxel_classes.empty() ? -1 : p.voxel_classes[0].material;
          int want = -1;
          for (std::size_t m = 0; m < a.model.materials.size(); ++m) {
            if (static_cast<int>(m) != have) {
              want = static_cast<int>(m);
              break;
            }
          }
          bool already = true;
          for (const VoxelClass& vc : p.voxel_classes) { already = already && vc.material == want; }
          SetSolidMaterial(a, ph, want);
          int wrong = 0;
          for (const VoxelClass& vc : a.model.solids[static_cast<std::size_t>(ph)]
                                          .voxel_classes) {
            if (vc.material != want) { ++wrong; }
          }
          if (already) {
            std::printf("selftest: FAILED - every class already had material %d, so setting "
                        "the volume's proves nothing\n", want);
          } else if (wrong != 0 || a.model.solids[static_cast<std::size_t>(ph)].material
                                       != want) {
            std::printf("selftest: FAILED - setting the volume's material left %d of %d "
                        "classes on another one\n", wrong, n_cls);
          } else {
            std::printf("selftest: the volume's material is every class's (%d classes)\n",
                        n_cls);
          }

          // And the volume's LAYER puts every class back on it.
          const int before_own = static_cast<int>(
              std::count_if(a.model.solids[static_cast<std::size_t>(ph)].voxel_classes.begin(),
                            a.model.solids[static_cast<std::size_t>(ph)].voxel_classes.end(),
                            [](const VoxelClass& vc) { return vc.layer != kInheritLayer; }));
          SetSolidLayer(a, ph, 6);
          int still_own = 0;
          for (const VoxelClass& vc : a.model.solids[static_cast<std::size_t>(ph)]
                                          .voxel_classes) {
            if (vc.layer != kInheritLayer) { ++still_own; }
          }
          if (before_own == 0) {
            std::printf("selftest: FAILED - no class had a layer of its own, so setting the "
                        "volume's proves nothing\n");
          } else if (still_own != 0
                     || a.model.solids[static_cast<std::size_t>(ph)].layer != 6) {
            std::printf("selftest: FAILED - setting the volume's layer left %d of %d classes "
                        "on their own\n", still_own, n_cls);
          } else {
            std::printf("selftest: the volume's layer carries every class (%d had their "
                        "own)\n", before_own);
          }
        }
      }
      // And the menu OPEN, which is the only way to see that its list lands where the row is
      // rather than off the panel or behind it. Opened by id, the way frame 45 opens the
      // colormap menu - there is no click to synthesise.
      if (frame == 59) {
        for (std::size_t i = 0; i < a.model.solids.size(); ++i) {
          if (a.model.solids[i].name == "adult_male_1mm_segmented_raw") {
            a.uic.open_select =
                kIdClassLayer + static_cast<int>(i) * kIdClassStride + 1;
          }
        }
        if (a.uic.open_select == 0) {
          std::printf("selftest: FAILED - no class layer menu to open\n");
        }
      }
      if (frame == 60) {
        if (a.uic.block.w <= 0 || a.uic.block.h <= 0) {
          std::printf("selftest: FAILED - the class layer menu opened but painted no list\n");
        } else {
          std::printf("selftest: a voxel class's layer menu opens on its own row (%d x %d at "
                      "%d,%d)\n", a.uic.block.w, a.uic.block.h, a.uic.block.x,
                      a.uic.block.y);
        }
        SaveFramePng(a, "D:/g4gpu/out/g4builder_class_layer_menu.png");
        a.uic.open_select = 0;
      }
      // COINCIDENT FACES: THE HIGHER LAYER IS DRAWN, THE LOWER ONE IS NOT.
      //
      // Two boxes in exactly the same place. Recolouring the one on top must change the
      // picture and recolouring the one underneath must not - and before the fix NEITHER did,
      // because the pixel drew nothing at all. So the first check is the one that fails on the
      // bug and the second is what says the layer rule is still being applied rather than the
      // pair simply being drawn in index order.
      if (frame == 61) {
        InsertSelftestCoincidentFaces(a);
        a.target = vis::Vec3f{0.f, -250.f, 0.f};
        a.distance = 200.f;
        a.azimuth = 0.6f;
        a.elevation = 0.3f;
      }
      static unsigned long long coinc_base = 0;
      static unsigned long long coinc_under = 0;
      if (frame == 62) {
        coinc_base = ViewportChecksum(a);
        RecolourSolid(a, "Underneath", 0.1f, 0.9f, 0.9f);
      }
      if (frame == 63) {
        coinc_under = ViewportChecksum(a);
        RecolourSolid(a, "Underneath", 0.9f, 0.2f, 0.2f);   // back as it was
        RecolourSolid(a, "OnTop", 0.9f, 0.9f, 0.1f);
      }
      if (frame == 64) {
        const unsigned long long coinc_top = ViewportChecksum(a);
        if (coinc_top == coinc_base) {
          std::printf("selftest: FAILED - recolouring the top one of two coincident faces "
                      "changed nothing (%llu), so neither is being drawn\n", coinc_base);
        } else if (coinc_under != coinc_base) {
          std::printf("selftest: FAILED - recolouring the box UNDERNEATH two coincident faces "
                      "changed the picture (%llu -> %llu), so the lower layer is winning\n",
                      coinc_base, coinc_under);
        } else {
          std::printf("selftest: where two faces coincide the higher layer is drawn and the "
                      "lower one is not\n");
        }
        RecolourSolid(a, "OnTop", 0.2f, 0.9f, 0.3f);
        a.target = vis::Vec3f{95.f, 0.f, 0.f};
        a.distance = 900.f;
        a.azimuth = 0.9f;
        a.elevation = 0.25f;
        // And from the next frame, compare the composited viewport against the device image.
        // See below for why this is a separate check and not the checksum one.
        a.check_blit = true;
        a.blit_mismatch = 0;
        a.blit_checked = 0;
      }
      // 1. THE BLIT PUTS THE WHOLE DEVICE IMAGE WHERE IT BELONGS.
      //
      // Against the device, because against another frame there is nothing to find: a
      // reference checksum goes through the SAME blit and matches it however wrong it is. That
      // is not a worry, it is what happened - the row stride was broken by forty columns
      // deliberately and check 2 below reported the picture unchanged, because both sides of
      // its comparison were short by the same forty. So this one reads d_rgba, in sync mode
      // where the device still holds the image that was just composited.
      if (frame == 66) {
        if (a.blit_checked == 0) {
          std::printf("selftest: FAILED - the composited viewport was never compared with the "
                      "device image, so nothing below means anything\n");
        } else if (a.blit_mismatch != 0) {
          std::printf("selftest: FAILED - the composited viewport differs from the device "
                      "image in %lld pixels\n", a.blit_mismatch);
        } else {
          std::printf("selftest: the composited viewport is the device image pixel for pixel "
                      "(%d frames)\n", a.blit_checked);
        }
        a.check_blit = false;

        // 2. AND THE POLLING PATH DELIVERS THAT SAME IMAGE.
        //
        // Everything above this point runs with sync_render on, because the checksums compare a
        // recolour against the frame after it - so nothing above exercises the polling path,
        // and a swapped staging index or an adopt that never fires would ship. Here the scene
        // and the camera stand still and the flag goes off: the same kernel on the same inputs
        // is bit-identical, so the checksum has to be too.
        async_want = ViewportChecksum(a);
        a.sync_render = false;
        // Zeroed here and required non-zero at 74: every frame so far completed a render
        // synchronously, so counting from the start would have made the "a render actually
        // landed" half of the check pass without one landing.
        a.render_frames = 0;
      }
      // 67 to 73 change nothing. Eight frames, not four: the first has no completed image yet
      // and composites frame 66's, and a render has to LAND inside the rest. The selftest scene
      // renders in tens of milliseconds - one completed in a four-frame window on this machine,
      // which is one bad day away from zero and a gate reporting a failure that is not one.
      if (frame == 74) {
        const unsigned long long got = ViewportChecksum(a);
        if (got != async_want) {
          std::printf("selftest: FAILED - the render off the UI frame drew a different picture "
                      "(%llu -> %llu)\n", async_want, got);
        } else if (a.render_frames == 0) {
          std::printf("selftest: FAILED - no render completed while polling for one, so the "
                      "picture only matched because it never changed\n");
        } else {
          std::printf("selftest: the render off the UI frame draws the same picture (%d "
                      "completed while polling)\n", a.render_frames);
        }
        // Back on, so the frame that writes the PNG below is this frame's render and not one
        // that happened to be in flight.
        a.sync_render = true;
      }

      // ---- THE NULL LAYER: a solid on it is not in the scene, and can come back.
      //
      // EVERY ONE OF THESE OPENS BY PROVING THE FIXTURE IS ON SCREEN, because every assertion
      // below is of the form "removing it changed the picture" and all of them are satisfied
      // by a volume that was never drawn. That is not a hypothetical: both fixtures were first
      // placed at y = 600, outside the 500 mm world, where nothing is drawn at all - and the
      // solid check passed. Recolouring is the operation that can only show up if the thing is
      // being painted, so it goes first and the rest depends on it.
      static unsigned long long nb_base = 0, nb_recol = 0, nb_with = 0;
      static int nb_vols = 0;
      if (frame == 75) {
        InsertSelftestNullLayer(a);
        a.target = vis::Vec3f{250.f, 0.f, 0.f};
        a.distance = 220.f;
        a.azimuth = 0.6f;
        a.elevation = 0.3f;
      }
      if (frame == 76) {
        nb_base = ViewportChecksum(a);
        RecolourSolid(a, "Nullable", 0.1f, 0.3f, 0.95f);
      }
      if (frame == 77) {
        nb_recol = ViewportChecksum(a);
        RecolourSolid(a, "Nullable", 0.95f, 0.75f, 0.15f);   // back as it was
      }
      if (frame == 78) {
        nb_with = ViewportChecksum(a);
        nb_vols = SceneVolumeCount(a);
        SetSolidLayer(a, ModelIndexByName(a, "Nullable"), kNullLayer);
      }
      if (frame == 79) {
        const unsigned long long got = ViewportChecksum(a);
        const int vols = SceneVolumeCount(a);
        // FOUR SEPARATE CLAIMS, and they fail separately. That the box is on screen at all.
        // That putting its colour back restores the picture, so the baseline is a baseline.
        // That one volume fewer reaches the flattened scene, which is what makes the transport
        // and the tally ignore it - there is nothing to step into and nothing to attach a
        // detector to. And that the picture changed. None of them implies another: a volume can
        // be absent from the scene and still leave its last render on screen, and it can be
        // invisible and still be transported through, which is what `visible` already does.
        if (nb_recol == nb_base) {
          std::printf("selftest: FAILED - the Nullable box is not on screen (recolouring it "
                      "changed nothing, %llu), so nothing below means anything\n", nb_base);
        } else if (nb_with != nb_base) {
          std::printf("selftest: FAILED - putting the box colour back did not restore the "
                      "picture (%llu -> %llu)\n", nb_base, nb_with);
        } else if (vols != nb_vols - 1) {
          std::printf("selftest: FAILED - a solid on the null layer is still in the scene "
                      "(%d volumes, was %d)\n", vols, nb_vols);
        } else if (got == nb_with) {
          std::printf("selftest: FAILED - removing a solid to the null layer changed nothing "
                      "in the picture (%llu)\n", nb_with);
        } else {
          std::printf("selftest: a solid on the null layer leaves the scene (%d volumes, was "
                      "%d) and leaves the picture\n", vols, nb_vols);
        }
        SetSolidLayer(a, ModelIndexByName(a, "Nullable"), 5);
      }
      if (frame == 80) {
        // AND COMES BACK IDENTICAL. Without this the check above is satisfied by a fixture
        // that merely broke: "the picture changed" is also what a crash that draws nothing
        // does.
        const unsigned long long back = ViewportChecksum(a);
        if (back != nb_with) {
          std::printf("selftest: FAILED - a solid put back from the null layer drew a "
                      "different picture (%llu -> %llu)\n", nb_with, back);
        } else {
          std::printf("selftest: and comes back to exactly the picture it left\n");
        }
      }

      // ---- THE WORLD IS ASKED ABOUT, AND CANNOT BE REMOVED AT ALL.
      if (frame == 81) {
        const int w0 = a.model.solids[0].layer;
        a.popup = Popup::kNone;
        SetSolidLayer(a, 0, 3);
        const bool asked = (a.popup == Popup::kWorldLayer)
                           && (a.model.solids[0].layer == w0);
        a.popup = Popup::kNone;
        SetSolidLayer(a, 0, kNullLayer);
        const bool refused = (a.model.solids[0].layer == w0) && (a.popup == Popup::kNone);
        // And the answer is allowed to be yes, which is what makes it a question rather than a
        // refusal - so the confirmed path is exercised too, and then put back.
        SetSolidLayer(a, 0, 3, true);
        const bool moved = (a.model.solids[0].layer == 3);
        SetSolidLayer(a, 0, 0, true);
        if (!asked) {
          std::printf("selftest: FAILED - moving the world off layer 0 was not confirmed\n");
        } else if (!refused) {
          std::printf("selftest: FAILED - the world was allowed onto the null layer\n");
        } else if (!moved) {
          std::printf("selftest: FAILED - confirming the world layer change did not apply it\n");
        } else {
          std::printf("selftest: the world layer is confirmed before it moves, and the null "
                      "layer is refused for it outright\n");
        }
      }

      // ---- A VOXEL CLASS ON THE NULL LAYER IS NOT DRAWN.
      //
      // The "Uncovered" grid, whose near half faces the camera with NOTHING over it. That is
      // the point of it: in the "Covered" fixture a null class is removed by the CLAMP - a cell
      // ranked below everything is covered by the first thing that outranks it and the march
      // ends before reaching it - so breaking the paint skip there left the check passing. It
      // was tried. With no cover the clamp never fires, and refusing to paint the cell is the
      // only thing left that can remove it.
      static unsigned long long vc_base = 0, vc_recol = 0, vc_with = 0, vc_without = 0;
      static unsigned long long vc_far = 0, vc_far_back = 0;
      if (frame == 82) {
        a.target = vis::Vec3f{0.f, 400.f, 0.f};
        a.distance = 160.f;
        a.azimuth = 0.f;
        a.elevation = 1.5f;   // down the z axis, so "near" is between the eye and "far"
      }
      if (frame == 83) {
        vc_base = ViewportChecksum(a);
        RecolourVoxelClass(a, "Uncovered", 0, 0.1f, 0.2f, 0.95f);
      }
      if (frame == 84) {
        vc_recol = ViewportChecksum(a);
        RecolourVoxelClass(a, "Uncovered", 0, 0.9f, 0.3f, 0.3f);   // back as it was
      }
      if (frame == 85) {
        vc_with = ViewportChecksum(a);
        SetVoxelClassLayer(a, "Uncovered", 0, kNullLayer);
      }
      if (frame == 86) {
        vc_without = ViewportChecksum(a);
        // THE FAR CLASS FIRST, because "the near class is gone" and "the whole grid is gone"
        // both change the picture and only one of them is the feature. Nulling one class
        // must leave the rest of the grid exactly where it was, and the way to ask is to
        // recolour a class that should have survived and require the picture to move.
        RecolourVoxelClass(a, "Uncovered", 1, 0.95f, 0.9f, 0.1f);
      }
      if (frame == 87) {
        vc_far = ViewportChecksum(a);
        RecolourVoxelClass(a, "Uncovered", 1, 0.2f, 0.8f, 0.9f);   // back as it was
      }
      if (frame == 88) {
        vc_far_back = ViewportChecksum(a);
        // Recolouring an ABSENT class must do nothing. This is the sharp half: "the picture
        // changed when I nulled it" is also satisfied by a class drawn in a different colour,
        // and only a recolour that changes NOTHING says the cells are gone.
        RecolourVoxelClass(a, "Uncovered", 0, 0.05f, 0.95f, 0.35f);
      }
      // ---- THE INSERT FORM: seeded from the world, and it inserts what it shows.
      //
      // The insert menu asks for a size and a position before it inserts, because those are
      // the two things anyone changes straight afterwards. Two claims worth checking and they
      // fail separately: the DEFAULT is set by the world rather than by a constant - the same
      // fixed fraction that gives a sensible box in a 500 mm world gives a speck in a 10 m one
      // - and what the form shows is what lands in the model.
      static double form_want = 0;
      static int form_solids = 0;
      if (frame == 90) {
        form_solids = static_cast<int>(a.model.solids.size());
        OpenPrimitiveForm(a, Shape::kBox);
        double world_half = a.model.solids[0].p[0];
        if (a.model.solids[0].p[1] < world_half) { world_half = a.model.solids[0].p[1]; }
        if (a.model.solids[0].p[2] < world_half) { world_half = a.model.solids[0].p[2]; }
        form_want = 0.5 * world_half;
      }
      if (frame == 91) {
        const Solid& p = a.pending_solid;
        double largest = 0;
        for (int k = 0; k < 3; ++k) {
          if (std::fabs(p.p[k]) > largest) { largest = std::fabs(p.p[k]); }
        }
        const bool centred = (p.pos[0] == 0 && p.pos[1] == 0 && p.pos[2] == 0);
        const bool sized = std::fabs(largest - form_want) < 1e-6;
        const bool open = (a.popup == Popup::kPrimitive);
        if (!open) {
          std::printf("selftest: FAILED - the insert menu did not open the size form\n");
        } else if (!sized) {
          std::printf("selftest: FAILED - the form seeded a largest dimension of %g, and half "
                      "the world is %g\n", largest, form_want);
        } else if (!centred) {
          std::printf("selftest: FAILED - the form did not centre the primitive on the world "
                      "(%g %g %g)\n", p.pos[0], p.pos[1], p.pos[2]);
        } else {
          std::printf("selftest: the insert form seeds half the world (%g mm) centred on it\n",
                      largest);
        }
        // Typed into, then inserted: what the form shows has to be what lands.
        a.pending_solid.p[1] = 37.5;
        a.pending_solid.pos[2] = -80.0;
        CommitPrimitiveForm(a);
      }
      if (frame == 92) {
        const int n_now = static_cast<int>(a.model.solids.size());
        if (n_now != form_solids + 1) {
          std::printf("selftest: FAILED - the insert form added %d solids, not one\n",
                      n_now - form_solids);
        } else {
          const Solid& s = a.model.solids[static_cast<std::size_t>(n_now - 1)];
          if (std::fabs(s.p[1] - 37.5) > 1e-9 || std::fabs(s.pos[2] + 80.0) > 1e-9) {
            std::printf("selftest: FAILED - the inserted solid does not carry what the form "
                        "held (half y %g, pos z %g)\n", s.p[1], s.pos[2]);
          } else if (a.popup != Popup::kNone) {
            std::printf("selftest: FAILED - the form stayed open after inserting\n");
          } else {
            std::printf("selftest: and it inserts exactly the size and position it was "
                        "showing\n");
          }
        }
      }

      if (frame == 89) {
        const unsigned long long after = ViewportChecksum(a);
        if (vc_recol == vc_base) {
          std::printf("selftest: FAILED - the uncovered grid near class is not on screen "
                      "(recolouring it changed nothing, %llu), so nothing below means "
                      "anything\n", vc_base);
        } else if (vc_with != vc_base) {
          std::printf("selftest: FAILED - putting the near class colour back did not restore "
                      "the picture (%llu -> %llu)\n", vc_base, vc_with);
        } else if (vc_without == vc_with) {
          std::printf("selftest: FAILED - a voxel class on the null layer is still drawn "
                      "(%llu)\n", vc_with);
        } else if (vc_far == vc_without) {
          std::printf("selftest: FAILED - nulling one class took the whole grid with it: "
                      "recolouring the far class changed nothing (%llu)\n", vc_without);
        } else if (vc_far_back != vc_without) {
          std::printf("selftest: FAILED - putting the far class colour back did not restore "
                      "the picture (%llu -> %llu)\n", vc_without, vc_far_back);
        } else if (after != vc_without) {
          std::printf("selftest: FAILED - recolouring a class on the null layer changed the "
                      "picture (%llu -> %llu), so its cells are still being painted\n",
                      vc_without, after);
        } else {
          std::printf("selftest: a voxel class on the null layer is not drawn even with "
                      "nothing over it, and recolouring it changes nothing\n");
        }
        SetVoxelClassLayer(a, "Uncovered", 0, kInheritLayer);
      }

      // ---- AND A VOLUME ON THE WORLD'S OWN LAYER CLASHES WITH THE WORLD.
      //
      // Uses the Nullable box from frame 75, which is inside the world and on layer 5.
      if (frame == 88) { SelftestCheckWorldOverlapRefusal(a); }

      // ---- AND THE LAYER MENU CAN SAY "null" AT ALL.
      //
      // The entry is a word now. It was a symbol drawn out of an ellipse and a stroke,
      // because the font atlas is indexed by byte and a three-byte UTF-8 character cannot
      // reach it - and at eight pixels across it never became more than a smudge that had
      // to be explained. What is worth checking either way is not how it looks but that the
      // menu maps it to the layer: an off-by-one between the option INDEX and the layer
      // NUMBER would put every solid one layer out, silently, and the list is the only
      // place that mapping exists.
      if (frame == 89) {
        RefreshLayerOptions(a);
        const int n_opt = static_cast<int>(a.layer_opt.size());
        bool round_trip = (n_opt > 3) && (std::strcmp(a.layer_opt[0], "null") == 0)
                          && (OptToLayer(0) == kNullLayer)
                          && (LayerToOpt(a, kNullLayer) == 0);
        for (int L = 0; L + 1 < n_opt && round_trip; ++L) {
          if (OptToLayer(LayerToOpt(a, L)) != L) { round_trip = false; }
          if (std::atoi(a.layer_opt[LayerToOpt(a, L)]) != L) { round_trip = false; }
        }
        if (!round_trip) {
          std::printf("selftest: FAILED - the layer menu does not map its entries to the "
                      "layers they name (%d entries, first \"%s\")\n", n_opt,
                      (n_opt > 0) ? a.layer_opt[0] : "");
        } else {
          std::printf("selftest: the layer menu says \"null\" and every entry maps to the "
                      "layer it names (%d entries)\n", n_opt);
        }
      }

      // ---- ANTI-ALIASING: partial coverage where a surface ends, and only there.
      //
      // Aimed at the Nullable box, which is opaque and on its own in that corner of the world.
      // Opaque matters: a translucent volume is partly covering every pixel it touches, which
      // would put a floor under the count that has nothing to do with edges.
      //
      // A box, not a sphere, on purpose: its silhouette is four straight lines, which is the
      // arrangement one ray per pixel turns into a staircase and the one anybody notices.
      // EVERYTHING ELSE HIDDEN, and that is not tidiness. The first version of this aimed at
      // the box and counted, and found 143,829 pixels already partly covered with the pass
      // switched off - the scene has translucent volumes in it and several were in frame, so
      // their coverage swamped the few hundred pixels of edge this is trying to measure. With
      // one opaque box in view the floor is zero and the signal is the whole count.
      static long long aa_off = 0;
      static std::vector<char> aa_hidden;
      if (frame == 93) {
        a.target = vis::Vec3f{250.f, 0.f, 0.f};
        a.distance = 220.f;
        a.azimuth = 0.6f;
        a.elevation = 0.3f;
        const int keep = ModelIndexByName(a, "Nullable");
        aa_hidden.assign(a.model.solids.size(), 0);
        for (std::size_t k = 0; k < a.model.solids.size(); ++k) {
          aa_hidden[k] = a.model.solids[k].visible ? 1 : 0;
          if (static_cast<int>(k) != keep && static_cast<int>(k) != a.model.world()) {
            a.model.solids[k].visible = false;
          }
        }
        a.model.Touch();
        a.scene_dirty = true;
        a.vis_attr.antialias = false;
      }
      if (frame == 94) {
        aa_off = CountPartialCoverage(a);
        a.vis_attr.antialias = true;
      }
      if (frame == 95) {
        const long long aa_on = CountPartialCoverage(a);
        // The box is 40 mm at 220 mm through a 45 degree field: its silhouette is a few
        // hundred pixels long, so a few hundred partly covered pixels is the right order and
        // "more than twice as many" is a bound that cannot be met by noise.
        if (aa_off > 40) {
          std::printf("selftest: FAILED - %lld pixels were already partly covered with "
                      "anti-aliasing off, so this scene cannot measure it\n", aa_off);
        } else if (aa_on < 100 || aa_on < 4 * (aa_off + 1)) {
          std::printf("selftest: FAILED - anti-aliasing produced %lld partly covered pixels "
                      "against %lld without it, which is not a smoothed edge\n", aa_on, aa_off);
        } else {
          std::printf("selftest: anti-aliasing softens the silhouette (%lld partly covered "
                      "pixels, against %lld with it off)\n", aa_on, aa_off);
        }
      }
      // AND IT CHANGES THE PICTURE AND NOTHING ELSE. A pass that retraced the whole frame
      // would also pass the count above; this says the interior is untouched, by turning it
      // off again and requiring the picture to come back.
      if (frame == 96) {
        const unsigned long long with = ViewportChecksum(a);
        a.vis_attr.antialias = false;
        aa_off = static_cast<long long>(with);
      }
      if (frame == 97) {
        const unsigned long long without = ViewportChecksum(a);
        if (without == static_cast<unsigned long long>(aa_off)) {
          std::printf("selftest: FAILED - turning anti-aliasing off changed nothing, so it was "
                      "never on\n");
        } else {
          std::printf("selftest: and turning it off changes the picture back\n");
        }
        a.vis_attr.antialias = true;
        for (std::size_t k = 0; k < a.model.solids.size() && k < aa_hidden.size(); ++k) {
          a.model.solids[k].visible = (aa_hidden[k] != 0);
        }
        a.model.Touch();
        a.scene_dirty = true;
      }

      // ---- A VOLUME IN A NULL CLASS'S SPACE IS DRAWN THERE.
      //
      // Reported: a volume on a non-null layer overlapping a voxel class set to null was not
      // rendered in the overlap region, while two ordinary volumes were fine.
      //
      // The cell march walks the grid's whole depth in one go and hands the ray back to the
      // outer search where something outranks a cell. An absent cell was SKIPPED before that
      // clamp was consulted, so nothing stopped the march inside a hole - it ran to the far
      // side of the grid and the search resumed beyond it, where a volume sitting in the hole
      // can never be found: a grid is not re-enterable from inside itself. And the scan that
      // collects candidate covers rejected any volume below the grid's lowest class before it
      // was considered at all, which an absent cell has no business being compared against.
      //
      // InHole is inside the grid's near half, on a LOWER layer than the grid - so while the
      // class is present the grid correctly hides it, and when the class is null it has to
      // appear.
      static unsigned long long hole_base = 0, hole_recol = 0;
      if (frame == 98) {
        a.target = vis::Vec3f{0.f, 400.f, 0.f};
        a.distance = 160.f;
        a.azimuth = 0.f;
        a.elevation = 1.5f;   // down the z axis, so the near half is in front
        SetVoxelClassLayer(a, "Uncovered", 0, kNullLayer);
      }
      if (frame == 99) {
        hole_base = ViewportChecksum(a);
        RecolourSolid(a, "InHole", 0.95f, 0.2f, 0.9f);
      }
      if (frame == 100) {
        hole_recol = ViewportChecksum(a);
        RecolourSolid(a, "InHole", 0.15f, 0.95f, 0.55f);   // back as it was
      }
      if (frame == 101) {
        const unsigned long long back = ViewportChecksum(a);
        // RECOLOURING IS THE QUESTION. "The picture changed when the class was nulled" is also
        // satisfied by the class vanishing and nothing taking its place; only a box that
        // responds to its own colour is a box that is being drawn.
        if (hole_recol == hole_base) {
          std::printf("selftest: FAILED - a volume inside a null voxel class is not drawn "
                      "there (recolouring it changed nothing, %llu)\n", hole_base);
        } else if (back != hole_base) {
          std::printf("selftest: FAILED - putting the box's colour back did not restore the "
                      "picture (%llu -> %llu)\n", hole_base, back);
        } else {
          std::printf("selftest: a volume overlapping a null voxel class is drawn in the "
                      "overlap\n");
        }
        // And with the class back, the grid owns that space again and hides it.
        SetVoxelClassLayer(a, "Uncovered", 0, kInheritLayer);
      }
      if (frame == 102) {
        const unsigned long long with_class = ViewportChecksum(a);
        RecolourSolid(a, "InHole", 0.95f, 0.2f, 0.9f);
        hole_base = with_class;
      }
      if (frame == 103) {
        const unsigned long long after = ViewportChecksum(a);
        // THE OTHER HALF OF THE RULE: with the class present the grid outranks the box
        // everywhere it covers, so recolouring the box changes nothing. Without this, "drawn
        // in the hole" would be satisfied by a renderer that ignores the layer rule entirely.
        if (after != hole_base) {
          std::printf("selftest: FAILED - the grid does not hide the box when its class is "
                      "present (%llu -> %llu)\n", hole_base, after);
        } else {
          std::printf("selftest: and hidden again when the class is put back\n");
        }
        RecolourSolid(a, "InHole", 0.15f, 0.95f, 0.55f);
      }

      // ---- AND THE WIREFRAME PASS RUNS AT ALL.
      //
      // The builder never launched one. Its styles said `solid = visible && !wireframe`, which
      // is right, so a volume set to wireframe was not ray cast - and nothing drew its edges
      // either, which made it invisible. The default world arrives with wireframe on, so the
      // world had no outline in the builder at all. Reported as wireframe rendering not
      // working.
      //
      // Turning the display of it off has to change the picture: if no edge is being drawn,
      // nothing changes.
      static unsigned long long wire_on = 0;
      if (frame == 104) { wire_on = ViewportChecksum(a); }
      if (frame == 105) { a.vis_attr.show_wireframe = false; }
      if (frame == 106) {
        const unsigned long long wire_off = ViewportChecksum(a);
        if (wire_off == wire_on) {
          std::printf("selftest: FAILED - turning the wireframe off changed nothing, so no "
                      "edge was being drawn (%llu)\n", wire_on);
        } else {
          std::printf("selftest: the wireframe pass draws edges (%d in this scene)\n",
                      a.n_edges);
        }
        a.vis_attr.show_wireframe = true;
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
