// The document the model-building GUI edits.
//
// This is deliberately a plain data description - elements, materials, solids, placements,
// sources, scorers, physics options - and not a live G4 object graph. Three reasons:
//
//   * it can be edited freely, including in ways that are momentarily invalid (a material
//     with no components yet, a solid with no material assigned), which a G4LogicalVolume
//     cannot represent;
//   * it can be saved to a file and reloaded without reconstructing anything;
//   * it can be *written out as C++*, which is the point of the Save command - the user gets a
//     project directory they can keep editing by hand and compile, not an opaque blob.
//
// Building the G4 graph from it happens in one place (builder/build_scene.hh), so "run from
// the GUI" and "save, compile, run" go through the same description and cannot drift.
#pragma once
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "g4/G4Types.hh"

namespace g4gpu::builder {

// ---------------------------------------------------------------- materials

struct Element {
  std::string name = "H";
  std::string symbol = "H";
  double z = 1;
  double a = 1.008;  ///< g/mole
};

/// How a component contributes to a compound.
enum class Fraction : int { kMassFraction = 0, kAtomCount = 1 };

struct Component {
  int element = -1;   ///< index into Model::elements
  double amount = 1;  ///< mass fraction in 0..1, or an atom count
};

enum class MatterState : int { kSolid = 0, kLiquid = 1, kGas = 2 };

struct Material {
  std::string name = "Material";
  /// A NIST name (G4_WATER, ...) makes this a lookup and ignores the components; empty means
  /// the components define it. Keeping both lets a model mix "just give me water" with a
  /// hand-built compound, which is what real detector descriptions do.
  std::string nist_name;
  double density = 1.0;  ///< g/cm3
  MatterState state = MatterState::kSolid;
  Fraction fraction_kind = Fraction::kMassFraction;
  std::vector<Component> components;
  /// Mean excitation energy, eV. Zero means "derive it", which is what Geant4 does when a
  /// material has no tabulated value.
  double mean_excitation_eV = 0;
};

// ---------------------------------------------------------------- solids

enum class Shape : int {
  kBox = 0, kTubs, kCons, kSphere, kOrb, kTorus, kTrd, kTrap, kPara,
  kEllipticalTube, kEllipsoid, kEllipticalCone, kParaboloid, kHype, kTet,
  kPolycone, kPolyhedra,
  kUnion, kSubtraction, kIntersection,
  kImportedMesh,   ///< a CAD import, held as triangles
  kVoxelGrid,      ///< an imported voxel volume or a flat detector
};

inline const char* ShapeName(Shape s) {
  switch (s) {
    case Shape::kBox: return "Box";
    case Shape::kTubs: return "Tubs";
    case Shape::kCons: return "Cons";
    case Shape::kSphere: return "Sphere";
    case Shape::kOrb: return "Orb";
    case Shape::kTorus: return "Torus";
    case Shape::kTrd: return "Trd";
    case Shape::kTrap: return "Trap";
    case Shape::kPara: return "Para";
    case Shape::kEllipticalTube: return "EllipticalTube";
    case Shape::kEllipsoid: return "Ellipsoid";
    case Shape::kEllipticalCone: return "EllipticalCone";
    case Shape::kParaboloid: return "Paraboloid";
    case Shape::kHype: return "Hype";
    case Shape::kTet: return "Tet";
    case Shape::kPolycone: return "Polycone";
    case Shape::kPolyhedra: return "Polyhedra";
    case Shape::kUnion: return "Union";
    case Shape::kSubtraction: return "Subtraction";
    case Shape::kIntersection: return "Intersection";
    case Shape::kImportedMesh: return "Mesh";
    case Shape::kVoxelGrid: return "VoxelGrid";
  }
  return "?";
}

/// The parameter names a shape takes, for the property panel. A null entry ends the list.
/// Keeping the labels next to the shape means the panel needs no per-shape code.
struct ShapeParam {
  const char* label;
  const char* unit;  ///< "mm", "deg", or "" for a dimensionless count
};

inline const ShapeParam* ShapeParams(Shape s, int& n) {
  static const ShapeParam kBox[] = {{"half x", "mm"}, {"half y", "mm"}, {"half z", "mm"}};
  static const ShapeParam kTubs[] = {{"rmin", "mm"}, {"rmax", "mm"}, {"half z", "mm"},
                                     {"start phi", "deg"}, {"delta phi", "deg"}};
  static const ShapeParam kCons[] = {{"rmin -z", "mm"}, {"rmax -z", "mm"}, {"rmin +z", "mm"},
                                     {"rmax +z", "mm"}, {"half z", "mm"},
                                     {"start phi", "deg"}, {"delta phi", "deg"}};
  static const ShapeParam kSphere[] = {{"rmin", "mm"}, {"rmax", "mm"}, {"start phi", "deg"},
                                       {"delta phi", "deg"}, {"start theta", "deg"},
                                       {"delta theta", "deg"}};
  static const ShapeParam kOrb[] = {{"radius", "mm"}};
  static const ShapeParam kTorus[] = {{"rmin", "mm"}, {"rmax", "mm"}, {"R torus", "mm"},
                                      {"start phi", "deg"}, {"delta phi", "deg"}};
  static const ShapeParam kTrd[] = {{"half x at -z", "mm"}, {"half x at +z", "mm"},
                                    {"half y at -z", "mm"}, {"half y at +z", "mm"},
                                    {"half z", "mm"}};
  static const ShapeParam kTrap[] = {{"half z", "mm"}, {"theta", "deg"}, {"phi", "deg"},
                                     {"half y1", "mm"}, {"half x1", "mm"}, {"half x2", "mm"},
                                     {"alpha1", "deg"}, {"half y2", "mm"}, {"half x3", "mm"},
                                     {"half x4", "mm"}, {"alpha2", "deg"}};
  static const ShapeParam kPara[] = {{"half x", "mm"}, {"half y", "mm"}, {"half z", "mm"},
                                     {"alpha", "deg"}, {"theta", "deg"}, {"phi", "deg"}};
  static const ShapeParam kEltube[] = {{"semi x", "mm"}, {"semi y", "mm"}, {"half z", "mm"}};
  static const ShapeParam kEllipsoid[] = {{"semi x", "mm"}, {"semi y", "mm"}, {"semi z", "mm"},
                                          {"z cut low", "mm"}, {"z cut high", "mm"}};
  static const ShapeParam kElcone[] = {{"x slope", ""}, {"y slope", ""}, {"z max", "mm"},
                                       {"z cut", "mm"}};
  static const ShapeParam kParaboloid[] = {{"half z", "mm"}, {"r at -z", "mm"},
                                           {"r at +z", "mm"}};
  static const ShapeParam kHype[] = {{"rmin", "mm"}, {"rmax", "mm"}, {"stereo in", "deg"},
                                     {"stereo out", "deg"}, {"half z", "mm"}};
  static const ShapeParam kTet[] = {{"p0 x", "mm"}, {"p0 y", "mm"}, {"p0 z", "mm"},
                                    {"p1 x", "mm"}, {"p1 y", "mm"}, {"p1 z", "mm"},
                                    {"p2 x", "mm"}, {"p2 y", "mm"}, {"p2 z", "mm"},
                                    {"p3 x", "mm"}, {"p3 y", "mm"}, {"p3 z", "mm"}};
  static const ShapeParam kPolycone[] = {{"start phi", "deg"}, {"delta phi", "deg"}};
  static const ShapeParam kPolyhedra[] = {{"start phi", "deg"}, {"delta phi", "deg"},
                                          {"sides", ""}};
  static const ShapeParam kVoxel[] = {{"half x", "mm"}, {"half y", "mm"}, {"half z", "mm"},
                                      {"nx", ""}, {"ny", ""}, {"nz", ""}};
  static const ShapeParam kNone[] = {{"", ""}};

  switch (s) {
    case Shape::kBox: n = 3; return kBox;
    case Shape::kTubs: n = 5; return kTubs;
    case Shape::kCons: n = 7; return kCons;
    case Shape::kSphere: n = 6; return kSphere;
    case Shape::kOrb: n = 1; return kOrb;
    case Shape::kTorus: n = 5; return kTorus;
    case Shape::kTrd: n = 5; return kTrd;
    case Shape::kTrap: n = 11; return kTrap;
    case Shape::kPara: n = 6; return kPara;
    case Shape::kEllipticalTube: n = 3; return kEltube;
    case Shape::kEllipsoid: n = 5; return kEllipsoid;
    case Shape::kEllipticalCone: n = 4; return kElcone;
    case Shape::kParaboloid: n = 3; return kParaboloid;
    case Shape::kHype: n = 5; return kHype;
    case Shape::kTet: n = 12; return kTet;
    case Shape::kPolycone: n = 2; return kPolycone;
    case Shape::kPolyhedra: n = 3; return kPolyhedra;
    case Shape::kVoxelGrid: n = 6; return kVoxel;
    default: n = 0; return kNone;
  }
}

/// What the numbers in a voxel file *are*.
///
/// A voxel array is a grid of numbers and the file does not say what they mean. Two readings
/// are common and they are not interchangeable:
///
///   * **indices** - the number identifies a material. A segmented phantom's 1 is "lung" and
///     its 2 is "bone", and 1.5 has no meaning at all. Every distinct value becomes a class
///     the user assigns a material and a colour to, which is what the sub-list in the SOLIDS
///     panel is for.
///   * **physical properties** - the number *is* a quantity: a density in g/cm3, a Hounsfield
///     unit, a stopping-power ratio. There is nothing to assign, because the material follows
///     from the number through a calibration curve, and two cells differing by one unit are
///     two slightly different materials rather than two categories.
///
/// Only indices are implemented. Properties needs a calibration curve, a way to edit it, and
/// a material table generated from it - and a decision about how finely to bin a continuous
/// range into materials, which is a modelling choice rather than an import detail. Asking the
/// question and refusing the unimplemented answer is better than guessing: a density field
/// read as indices produces one "material" per distinct density, which for real data is
/// thousands of classes and a phantom nobody can assign.
enum class VoxelMeaning : int {
  kIndices = 0,     ///< each distinct value names a material
  kProperties = 1,  ///< each value is a physical quantity; not implemented
};

inline const char* VoxelMeaningName(VoxelMeaning m) {
  return (m == VoxelMeaning::kProperties) ? "physical properties" : "material indices";
}

/// How the values in a voxel file are meant to be read.
enum class VoxelKind : int {
  kDiscrete = 0,    ///< each distinct value is a material - a segmented phantom
  kContinuous = 1,  ///< values are Hounsfield units, mapped to material and density
};

/// One distinct value in a discrete voxel volume, or one HU band in a continuous one.
/// A voxel class layer meaning "whatever layer the volume is on".
///
/// A sentinel rather than a copy of the volume's layer, and the reason is cost rather than
/// tidiness: the navigator only takes its per-point layer path when some class differs from
/// the volume, so a phantom whose classes all inherit has to be distinguishable from one
/// whose classes were each set to the same number. See build_scene, which drops the whole
/// per-class run when every class agrees with the volume.
constexpr int kInheritLayer = -2147483647;

/// A solid or a voxel class carrying THIS in its layer field is not in the scene at all.
///
/// Not transported through, not drawn, and not scored - one fact for all three rather than
/// three switches that can disagree. It shares the layer CONTROL because that is where the
/// question is asked ("what is here, and what wins"), and because "nothing, this is not here"
/// belongs beside "layer 3": someone deciding what a volume is for should not have to learn
/// that hiding it, excluding it from the run and excluding it from the tally live in three
/// different places.
///
/// IT IS NOT A LAYER, AND NOT A NUMBER. Nothing computes a rank from it. A solid carrying it
/// is not placed; a class carrying it keeps its cells' values and its colour, and its cells
/// are simply not part of the volume - geom::VoxelStore::class_absent carries that as its own
/// per-class fact, and geom::inside_volume is where it is honoured. The space belongs to
/// whatever else contains it, so no scorer sees it either: the tally follows ownership and
/// needs no rule of its own.
///
/// The first implementation did make it a number - a layer two billion below everything, on
/// the reasoning that such a cell would lose every overlap and the world would take its space
/// with no new mechanism at all. geom::kNullLayerTag records what that cost. Absence is not a
/// small number: a cell that is not there does not lose overlaps, it does not take part in
/// them.
///
/// geom::kNullLayerTag is the same value, asserted equal in build_scene.hh, and exists in the
/// geometry only so the flattener can refuse a placement that still carries it.
constexpr int kNullLayer = -2147483646;

struct VoxelClass {
  double value = 0;      ///< the voxel value, or the lower edge of the HU band
  double value_max = 0;  ///< upper edge, for a continuous band
  int material = -1;
  /// This class's overlap priority, or kInheritLayer to take the volume's.
  ///
  /// The same number a solid's layer is, compared the same way: higher wins where two volumes
  /// overlap, a tie goes to whichever was placed later. What it buys is a phantom that wins
  /// an overlap where it is bone and loses it where it is air - one volume, two answers,
  /// which the layer model could not say before.
  int layer = kInheritLayer;
  float r = 0.7f, g = 0.7f, b = 0.7f;
  /// 1 is opaque, 0 invisible. Carried per class because a segmented phantom is only
  /// readable when the outer tissue is transparent and the bone is not.
  float opacity = 1.0f;
  bool visible = true;
  std::string label;
};

struct Solid {
  std::string name = "Solid";
  Shape shape = Shape::kBox;
  double p[12] = {50, 50, 50, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  // Placement in the world.
  double pos[3] = {0, 0, 0};
  double rot[3] = {0, 0, 0};  ///< degrees about x, y, z, applied to the solid
  int layer = 1;
  int anchor = -1;  ///< index of another solid this one moves with, or -1

  int material = -1;
  float r = 0.8f, g = 0.8f, b = 0.8f;
  /// 1 is opaque, 0 invisible. Separate from `visible`, which is a hard on/off: a user wants
  /// to see the outer shell *and* what is inside it, which needs a value in between.
  float opacity = 1.0f;
  bool visible = true;
  bool wireframe = false;

  // Boolean operands, for kUnion / kSubtraction / kIntersection.
  int operand_a = -1, operand_b = -1;

  // Polycone / polyhedra z-sections, as (z, rmin, rmax) triples.
  std::vector<double> sections;

  // An import: the file it came from, so a reload can find it again.
  std::string source_file;

  // Voxel volume.
  VoxelKind voxel_kind = VoxelKind::kDiscrete;
  /// What the values mean. Only kIndices is implemented; see VoxelMeaning. The class list
  /// below, and the sub-list the SOLIDS panel draws from it, are meaningful only for indices.
  VoxelMeaning voxel_meaning = VoxelMeaning::kIndices;
  /// Whether the SOLIDS panel shows this volume's class sub-list. Collapsed by default: a
  /// segmented CT has dozens of classes and they would bury every other solid in the scene.
  /// Purely a view state, so it is not written to the document.
  bool ui_expanded = false;
  std::vector<VoxelClass> voxel_classes;
  /// Rows and columns for a flat detector; zero for a general voxel volume.
  int det_rows = 0, det_cols = 0;

  /// Raw voxel values as read from the file, nx*ny*nz. Kept rather than only the classes,
  /// because reclassifying (discrete to continuous, or a different band set) has to be
  /// possible without re-reading a file the user may have moved.
  std::vector<float> voxel_values;

  /// Mesh triangles for a CAD import, as flat xyz triples.
  std::vector<float> mesh;
};

// ---------------------------------------------------------------- sources

enum class SourceKind : int {
  kBeam = 0,        ///< a rectangular or elliptical beam
  kPoint = 1,       ///< a point with an angular spread
  kIsotropicShell = 2,
  kVolume = 3,      ///< uniform inside a box
  kRadioactive = 4, ///< a nuclide with an activity
};

enum class BeamProfile : int { kRectangular = 0, kElliptical = 1 };

struct Source {
  std::string name = "Source";
  SourceKind kind = SourceKind::kBeam;
  BeamProfile profile = BeamProfile::kRectangular;

  double pos[3] = {0, 0, -150};
  double dir[3] = {0, 0, 1};
  double half_x = 50, half_y = 50, half_z = 0;
  double radius = 100;
  double angular_spread_deg = 0;

  std::string particle = "gamma";
  double energy_MeV = 6.0;
  /// A CSV of energy and intensity; when set, it overrides `energy_MeV`.
  std::string spectrum_file;

  // Radioactive source.
  std::string nuclide = "Cs137";
  double activity = 1.0;
  bool activity_in_curie = false;

  bool enabled = true;
  /// Share of a run's events this source emits, relative to the other enabled sources.
  ///
  /// A run with several enabled sources is split by weight rather than sampled from one of
  /// them at random: `round(n * w_i / sum(w))` events come from source i. A fixed split is
  /// stratified sampling, so it has *less* variance than choosing per event, and it makes a
  /// two-beam run reproducible in a way a random choice would not.
  double weight = 1.0;
};

// ---------------------------------------------------------------- scoring

enum class ScoreQuantity : int {
  kEnergyDeposit = 0, kDose = 1, kFluence = 2, kTrackLength = 3, kStepCount = 4,
};

inline const char* ScoreName(ScoreQuantity q) {
  switch (q) {
    case ScoreQuantity::kEnergyDeposit: return "energy deposit";
    case ScoreQuantity::kDose: return "dose";
    case ScoreQuantity::kFluence: return "fluence";
    case ScoreQuantity::kTrackLength: return "track length";
    case ScoreQuantity::kStepCount: return "step count";
  }
  return "?";
}

struct Scorer {
  std::string name = "Scorer";
  ScoreQuantity quantity = ScoreQuantity::kDose;
  std::vector<int> solids;  ///< which solids feed it
  /// Score each voxel of a voxel volume separately, as G4PSEnergyDeposit3D does, instead of
  /// summing the whole volume into one number.
  ///
  /// Off by default, and worth a thought before turning on: it makes the stepper stop at every
  /// cell boundary rather than only where the material changes, which for a homogeneous region
  /// of a CT is the difference between one step and a few hundred. What it buys is a dose
  /// *distribution* - a depth-dose curve, a profile - rather than a single total, which for a
  /// voxelised phantom is usually the entire reason it was voxelised.
  bool per_voxel = false;

  /// Emit this scorer as a user-editable class in the saved project.
  ///
  /// A generated project registers stock primitives - G4PSDoseDeposit and friends - which do
  /// what they do and nothing else. Flagging a scorer custom writes it out as its own
  /// subclass, one file pair per scorer, with the hook where a filter or a weighting goes:
  /// score only above an energy, only a species, only part of a volume. The baseline
  /// behaviour is what the generated class does before you edit it, so a custom scorer that
  /// has not been edited reports exactly what the stock one would.
  ///
  /// It has no effect on a run in the builder: this is about the code that gets written.
  bool custom = false;
};

// ---------------------------------------------------------------- physics

struct Physics {
  bool photoelectric = true;
  bool compton = true;
  bool rayleigh = true;
  bool pair_production = true;
  /// Ionisation has no toggle: it is the continuous energy loss along a step, and without it
  /// a lepton has infinite range. See ProcessFlags in physics/scene.cuh.
  bool bremsstrahlung = true;
  bool annihilation = true;
  bool multiple_scattering = true;
  double range_cut_mm = 0.7;
};

// ---------------------------------------------------------------- the document

struct Model {
  std::string name = "MyDetector";
  /// The world. Index 0 of `solids` is always the world, on layer 0.
  std::vector<Element> elements;
  std::vector<Material> materials;
  std::vector<Solid> solids;
  std::vector<Source> sources;
  std::vector<Scorer> scorers;
  Physics physics;

  /// True once anything has changed since the last save, so the title can say so and Exit can
  /// warn. Tracked by the editing helpers below rather than by every call site.
  bool dirty = false;

  int world() const { return solids.empty() ? -1 : 0; }

  void Touch() { dirty = true; }

  int AddElement(const Element& e) {
    elements.push_back(e);
    Touch();
    return static_cast<int>(elements.size()) - 1;
  }
  int AddMaterial(const Material& m) {
    materials.push_back(m);
    Touch();
    return static_cast<int>(materials.size()) - 1;
  }
  int AddSolid(const Solid& s) {
    solids.push_back(s);
    Touch();
    return static_cast<int>(solids.size()) - 1;
  }
  int AddSource(const Source& s) {
    sources.push_back(s);
    Touch();
    return static_cast<int>(sources.size()) - 1;
  }
  int AddScorer(const Scorer& s) {
    scorers.push_back(s);
    Touch();
    return static_cast<int>(scorers.size()) - 1;
  }

  /// Removes a solid and repairs every index that pointed at or past it. Deleting without
  /// this leaves booleans referring to the wrong operand and scorers to the wrong volume -
  /// silently, and the scene still builds.
  void RemoveSolid(int idx);
  void RemoveMaterial(int idx);
  void RemoveElement(int idx);

  /// A default model: a 1 m air world, so the GUI opens on something.
  static Model Default();
};

namespace detail {
/// Shifts an index after a removal: -1 stays -1, the removed one becomes -1, later ones drop.
inline int Reindex(int v, int removed) {
  if (v < 0) { return v; }
  if (v == removed) { return -1; }
  return (v > removed) ? v - 1 : v;
}
}  // namespace detail

inline void Model::RemoveSolid(int idx) {
  if (idx < 0 || idx >= static_cast<int>(solids.size())) { return; }
  if (idx == 0) { return; }  // the world cannot be deleted
  solids.erase(solids.begin() + idx);
  for (Solid& s : solids) {
    s.operand_a = detail::Reindex(s.operand_a, idx);
    s.operand_b = detail::Reindex(s.operand_b, idx);
    s.anchor = detail::Reindex(s.anchor, idx);
  }
  for (Scorer& sc : scorers) {
    std::vector<int> kept;
    for (int v : sc.solids) {
      const int r = detail::Reindex(v, idx);
      if (r >= 0) { kept.push_back(r); }
    }
    sc.solids = kept;
  }
  Touch();
}

inline void Model::RemoveMaterial(int idx) {
  if (idx < 0 || idx >= static_cast<int>(materials.size())) { return; }
  materials.erase(materials.begin() + idx);
  for (Solid& s : solids) { s.material = detail::Reindex(s.material, idx); }
  for (Solid& s : solids) {
    for (VoxelClass& v : s.voxel_classes) { v.material = detail::Reindex(v.material, idx); }
  }
  Touch();
}

inline void Model::RemoveElement(int idx) {
  if (idx < 0 || idx >= static_cast<int>(elements.size())) { return; }
  elements.erase(elements.begin() + idx);
  for (Material& m : materials) {
    std::vector<Component> kept;
    for (Component& c : m.components) {
      c.element = detail::Reindex(c.element, idx);
      if (c.element >= 0) { kept.push_back(c); }
    }
    m.components = kept;
  }
  Touch();
}

// ---------------------------------------------------------------- anchored placement

namespace detail {

/// Row-major 3x3 product, `out = a * b`.
inline void Mat3Mul(const double a[9], const double b[9], double out[9]) {
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      out[3 * i + j] = a[3 * i + 0] * b[0 * 3 + j] + a[3 * i + 1] * b[1 * 3 + j]
                       + a[3 * i + 2] * b[2 * 3 + j];
    }
  }
}

/// `m = r * m`, the pre-multiply G4RotationMatrix::rotateX and friends do.
inline void Mat3PreMul(double m[9], const double r[9]) {
  double out[9];
  Mat3Mul(r, m, out);
  for (int k = 0; k < 9; ++k) { m[k] = out[k]; }
}

}  // namespace detail

/// The world placement of a solid, following its anchor chain.
///
/// A solid may be anchored to another: its position and rotation are then read in that
/// volume's frame rather than the world's, so moving or turning the anchor carries everything
/// anchored to it. That is what makes a gantry, a couch with a phantom on it, or a collimator
/// stack something you aim as one object instead of re-typing every position.
///
/// This is a *placement* chain, not a mother/daughter tree. Which volume owns a piece of
/// space is still settled by layer index - see geom::locate - and stays a separate question
/// from where a volume is, deliberately: a volume can be anchored to one it does not sit
/// inside, and a volume inside another need not be anchored to it.
///
/// It lives here, in the model, rather than in the scene builder, because the *project
/// writer* needs the same answer. Two implementations of this would be two chances to place a
/// saved project differently from the one that was run, which is the single failure this file
/// set out to make impossible.
///
/// @param[out] inv  the world -> solid rotation, row major - the convention G4PVPlacement
///                  takes, which is the inverse of the object's own rotation
/// @param[out] pos  the world position, in the model's units (mm)
inline void SolidWorldPlacement(const Model& m, int idx, double inv[9], double pos[3]) {
  static const double kIdentity[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  for (int k = 0; k < 9; ++k) { inv[k] = kIdentity[k]; }
  pos[0] = pos[1] = pos[2] = 0;
  if (idx < 0 || idx >= static_cast<int>(m.solids.size())) { return; }

  // Walk to the root of the chain first, so the composition can be built outermost-inward
  // with no recursion. A chain longer than the number of solids can only be a cycle; it stops
  // there. The panel refuses to create one, and this is the backstop for a hand-edited
  // document.
  int chain[64];
  int n = 0;
  int at = idx;
  const int limit = static_cast<int>(m.solids.size()) < 64
                        ? static_cast<int>(m.solids.size()) : 64;
  while (n < limit) {
    chain[n++] = at;
    const int next = m.solids[static_cast<std::size_t>(at)].anchor;
    if (next < 0 || next == at || next >= static_cast<int>(m.solids.size())) { break; }
    at = next;
  }

  // From the outermost anchor inward. At each step the running transform is the parent's, and
  // the child's own placement is applied in the parent's frame.
  for (int k = n - 1; k >= 0; --k) {
    const Solid& s = m.solids[static_cast<std::size_t>(chain[k])];

    double own[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    // Negated angles, and X then Y then Z: the model stores the rotation of the *object* and
    // this matrix is the world -> solid map, which is its inverse. Exactly what
    // G4RotationMatrix::rotateX/Y/Z in sequence produce, because that is what the placement
    // code does and the two have to agree to the bit.
    if (s.rot[0] != 0) {
      const double a = -s.rot[0] * 3.14159265358979323846 / 180.0;
      const double c = std::cos(a), sn = std::sin(a);
      const double r[9] = {1, 0, 0, 0, c, -sn, 0, sn, c};
      detail::Mat3PreMul(own, r);
    }
    if (s.rot[1] != 0) {
      const double a = -s.rot[1] * 3.14159265358979323846 / 180.0;
      const double c = std::cos(a), sn = std::sin(a);
      const double r[9] = {c, 0, sn, 0, 1, 0, -sn, 0, c};
      detail::Mat3PreMul(own, r);
    }
    if (s.rot[2] != 0) {
      const double a = -s.rot[2] * 3.14159265358979323846 / 180.0;
      const double c = std::cos(a), sn = std::sin(a);
      const double r[9] = {c, -sn, 0, sn, c, 0, 0, 0, 1};
      detail::Mat3PreMul(own, r);
    }

    // The offset is read in the parent's frame, so it turns with the parent. `inv` is the
    // world -> parent map; its transpose takes the parent's frame back to the world.
    for (int i = 0; i < 3; ++i) {
      pos[i] += inv[0 * 3 + i] * s.pos[0] + inv[1 * 3 + i] * s.pos[1]
                + inv[2 * 3 + i] * s.pos[2];
    }
    // World object rotation is parent * own, so the world -> solid map is own^-1 * parent^-1,
    // and both factors here are already those inverses.
    double composed[9];
    detail::Mat3Mul(own, inv, composed);
    for (int k2 = 0; k2 < 9; ++k2) { inv[k2] = composed[k2]; }
  }
}

/// Whether making @p solid an anchor of @p target would close a cycle.
///
/// A cycle has no world placement at all: every volume in it would be positioned relative to
/// another that is positioned relative to it. SolidWorldPlacement stops rather than looping,
/// but what it stops at is arbitrary, so the panel refuses the assignment instead.
inline bool AnchorWouldCycle(const Model& m, int target, int anchor) {
  if (target < 0 || anchor < 0) { return false; }
  if (target == anchor) { return true; }
  int at = anchor;
  for (std::size_t guard = 0; guard <= m.solids.size(); ++guard) {
    if (at < 0 || at >= static_cast<int>(m.solids.size())) { return false; }
    if (at == target) { return true; }
    at = m.solids[static_cast<std::size_t>(at)].anchor;
    if (at < 0) { return false; }
  }
  return true;
}

inline Model Model::Default() {
  Model m;
  Material air;
  air.name = "Air";
  air.nist_name = "G4_AIR";
  air.density = 0.00120479;
  air.state = MatterState::kGas;
  m.materials.push_back(air);

  Solid world;
  world.name = "World";
  world.shape = Shape::kBox;
  world.p[0] = world.p[1] = world.p[2] = 500;  // 1 m cube
  world.layer = 0;
  world.material = 0;
  world.r = 0.35f;
  world.g = 0.35f;
  world.b = 0.40f;
  world.wireframe = true;
  m.solids.push_back(world);

  // No source. A default one is a beam the user did not choose, at an energy and a position
  // they did not pick, and the first run either uses it by accident or has to be understood
  // and corrected before it can be replaced. An empty list says what the state is: nothing
  // has been aimed at anything yet. A run with no source is refused with that sentence.
  //
  // The world is different, and stays: geometry cannot exist without one, and the New dialog
  // asks for its shape and size rather than assuming.

  m.dirty = false;
  return m;
}

}  // namespace g4gpu::builder
