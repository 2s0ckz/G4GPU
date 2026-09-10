// Builds a live G4 object graph from a Model, so the GUI can run what it is showing.
//
// This is the *same* description the project writer emits as C++, and that is the point: Run
// and Save cannot disagree about what the model means, because both read this one file's idea
// of it. If a shape, a source or a scorer is added to the model, it has to be handled here and
// in write_project.cc, and a mismatch shows up immediately as "the GUI runs it but the saved
// project does not compile" rather than as a quiet numerical difference.
#pragma once
#include <cstdio>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "builder/model.hh"
#include "g4/G4Colour.hh"
#include "g4/G4NistManager.hh"
#include "g4/G4RunManager.hh"
#include "g4/G4SDManager.hh"
#include "g4/G4Solids.hh"
#include "g4/G4UserActions.hh"

namespace g4gpu::builder {

/// Detector construction over a Model. Holds the built objects so they outlive Construct().
class ModelDetector : public G4VUserDetectorConstruction {
 public:
  explicit ModelDetector(const Model* model) : model_(model) {}

  G4VPhysicalVolume* Construct() override;
  void ConstructSDandField() override;

  /// The built material for model index @p i, or nullptr.
  ///
  /// For reading back what the build *derived*. A material given no mean excitation energy
  /// gets one by Bragg additivity in G4IonisParamMat, and the only place that number exists
  /// is the built G4Material - the model still holds zero, which is what asks for it to be
  /// derived in the first place. The builder shows it in the I field, so that "0 = derive"
  /// does not also mean "and you will never see what it derived".
  const G4Material* BuiltMaterial(int i) const {
    return (i >= 0 && i < static_cast<int>(materials_.size())) ? materials_[i] : nullptr;
  }

 private:
  /// Builds one solid, recursing into boolean operands. Cached, so an operand shared between
  /// two booleans is built once.
  G4VSolid* BuildSolid(int idx);

  const Model* model_;
  std::vector<G4VSolid*> solids_;
  std::vector<G4LogicalVolume*> logicals_;
  std::vector<G4Material*> materials_;
  std::vector<G4Element*> elements_;
  std::vector<std::unique_ptr<G4VisAttributes>> vis_;
  G4VPhysicalVolume* world_ = nullptr;
};

/// Primary generator over a Model's first enabled source.
class ModelPrimary : public G4VUserPrimaryGeneratorAction {
 public:
  explicit ModelPrimary(const Model* model) : model_(model) {
    gun_ = new G4ParticleGun(1);
    if (G4RunManager::Instance() != nullptr) { G4RunManager::Instance()->SetGun(gun_); }
  }
  ~ModelPrimary() override { delete gun_; }

  void GeneratePrimaries(G4Event* event) override;
  G4ParticleGun* GetParticleGun() { return gun_; }

  /// Which source to configure the gun from, by index, or -1 for "the first enabled one".
  ///
  /// A run with several enabled sources is several sub-runs, one per source, and each needs
  /// the gun set up for *its* source. Without this the generator always took the first
  /// enabled one and a second beam did nothing but slow the run down.
  void SelectSource(int i) { selected_ = i; }

 /// Configures the gun for one source. Shared by the selected-source and the
  /// register-them-all paths so the two cannot describe the same source differently.
  void ConfigureFor(const Source& src);

 private:
  const Model* model_;
  G4ParticleGun* gun_ = nullptr;
  int selected_ = -1;
};

// ---------------------------------------------------------------- implementation

inline G4VSolid* ModelDetector::BuildSolid(int idx) {
  if (idx < 0 || idx >= static_cast<int>(model_->solids.size())) { return nullptr; }
  if (solids_[idx] != nullptr) { return solids_[idx]; }

  const Solid& s = model_->solids[idx];
  const double* p = s.p;
  const G4String name = s.name;
  G4VSolid* out = nullptr;

  switch (s.shape) {
    case Shape::kBox:
      out = new G4Box(name, p[0] * mm, p[1] * mm, p[2] * mm);
      break;
    case Shape::kTubs:
      out = new G4Tubs(name, p[0] * mm, p[1] * mm, p[2] * mm, p[3] * deg, p[4] * deg);
      break;
    case Shape::kCons:
      out = new G4Cons(name, p[0] * mm, p[1] * mm, p[2] * mm, p[3] * mm, p[4] * mm,
                       p[5] * deg, p[6] * deg);
      break;
    case Shape::kSphere:
      out = new G4Sphere(name, p[0] * mm, p[1] * mm, p[2] * deg, p[3] * deg, p[4] * deg,
                         p[5] * deg);
      break;
    case Shape::kOrb:
      out = new G4Orb(name, p[0] * mm);
      break;
    case Shape::kTorus:
      out = new G4Torus(name, p[0] * mm, p[1] * mm, p[2] * mm, p[3] * deg, p[4] * deg);
      break;
    case Shape::kTrd:
      out = new G4Trd(name, p[0] * mm, p[1] * mm, p[2] * mm, p[3] * mm, p[4] * mm);
      break;
    case Shape::kTrap:
      out = new G4Trap(name, p[0] * mm, p[1] * deg, p[2] * deg, p[3] * mm, p[4] * mm,
                       p[5] * mm, p[6] * deg, p[7] * mm, p[8] * mm, p[9] * mm, p[10] * deg);
      break;
    case Shape::kPara:
      out = new G4Para(name, p[0] * mm, p[1] * mm, p[2] * mm, p[3] * deg, p[4] * deg,
                       p[5] * deg);
      break;
    case Shape::kEllipticalTube:
      out = new G4EllipticalTube(name, p[0] * mm, p[1] * mm, p[2] * mm);
      break;
    case Shape::kEllipsoid:
      out = new G4Ellipsoid(name, p[0] * mm, p[1] * mm, p[2] * mm, p[3] * mm, p[4] * mm);
      break;
    case Shape::kEllipticalCone:
      out = new G4EllipticalCone(name, p[0], p[1], p[2] * mm, p[3] * mm);
      break;
    case Shape::kParaboloid:
      out = new G4Paraboloid(name, p[0] * mm, p[1] * mm, p[2] * mm);
      break;
    case Shape::kHype:
      out = new G4Hype(name, p[0] * mm, p[1] * mm, p[2] * deg, p[3] * deg, p[4] * mm);
      break;
    case Shape::kTet:
      out = new G4Tet(name, G4ThreeVector(p[0] * mm, p[1] * mm, p[2] * mm),
                      G4ThreeVector(p[3] * mm, p[4] * mm, p[5] * mm),
                      G4ThreeVector(p[6] * mm, p[7] * mm, p[8] * mm),
                      G4ThreeVector(p[9] * mm, p[10] * mm, p[11] * mm));
      break;
    case Shape::kPolycone:
    case Shape::kPolyhedra: {
      const std::size_t n = s.sections.size() / 3;
      if (n < 2) {
        std::printf("solid \"%s\": a %s needs at least two z-sections; skipped\n",
                    s.name.c_str(), ShapeName(s.shape));
        break;
      }
      std::vector<G4double> z(n), ri(n), ro(n);
      for (std::size_t k = 0; k < n; ++k) {
        z[k] = s.sections[3 * k + 0] * mm;
        ri[k] = s.sections[3 * k + 1] * mm;
        ro[k] = s.sections[3 * k + 2] * mm;
      }
      if (s.shape == Shape::kPolycone) {
        out = new G4Polycone(name, p[0] * deg, p[1] * deg, static_cast<G4int>(n), z.data(),
                             ri.data(), ro.data());
      } else {
        out = new G4Polyhedra(name, p[0] * deg, p[1] * deg,
                              static_cast<G4int>(p[2] + 0.5), static_cast<G4int>(n), z.data(),
                              ri.data(), ro.data());
      }
      break;
    }
    case Shape::kUnion:
    case Shape::kSubtraction:
    case Shape::kIntersection: {
      G4VSolid* a = BuildSolid(s.operand_a);
      G4VSolid* b = BuildSolid(s.operand_b);
      if (a == nullptr || b == nullptr) {
        std::printf("solid \"%s\": a %s needs both operands; skipped\n", s.name.c_str(),
                    ShapeName(s.shape));
        break;
      }
      // The operands carry absolute positions in the model, because the user placed them
      // individually before combining them. The boolean wants the second relative to the
      // first, so the difference is taken here - and the composite is then placed at the
      // first operand's position, which is what keeps the result where the user left it.
      const Solid& sa = model_->solids[s.operand_a];
      const Solid& sb = model_->solids[s.operand_b];
      const G4ThreeVector off((sb.pos[0] - sa.pos[0]) * mm, (sb.pos[1] - sa.pos[1]) * mm,
                              (sb.pos[2] - sa.pos[2]) * mm);
      if (s.shape == Shape::kUnion) {
        out = new G4UnionSolid(name, a, b, nullptr, off);
      } else if (s.shape == Shape::kSubtraction) {
        out = new G4SubtractionSolid(name, a, b, nullptr, off);
      } else {
        out = new G4IntersectionSolid(name, a, b, nullptr, off);
      }
      break;
    }
    case Shape::kImportedMesh: {
      // The triangles as imported, transported against a BVH over them. The model keeps them
      // in floats; the transport is double, so they widen here rather than at every query.
      //
      // A mesh with no triangles is a reopened model whose CAD file has moved - see
      // ReloadMeshSources. G4TessellatedSolid would exit(2) rather than be placed empty,
      // which is right in a batch run and wrong in a GUI the user is still working in, so
      // the bounding box stands in and the volume list says so.
      if (s.mesh.size() < 9) {
        std::printf("WARNING: mesh \"%s\" has no triangles (its source file moved?);\n"
                    "  placing its bounding box instead. Re-import to transport the mesh.\n",
                    s.name.c_str());
        out = new G4Box(name, s.p[0] * mm, s.p[1] * mm, s.p[2] * mm);
        break;
      }
      auto* mesh = new G4TessellatedSolid(name);
      std::vector<G4double> tri;
      tri.reserve(s.mesh.size());
      for (float v : s.mesh) { tri.push_back(static_cast<G4double>(v) * mm); }
      mesh->AddTriangles(tri);
      mesh->SetSolidClosed(true);
      out = mesh;
      break;
    }
    case Shape::kVoxelGrid: {
      const int nx = static_cast<int>(s.p[3] + 0.5);
      const int ny = static_cast<int>(s.p[4] + 0.5);
      const int nz = static_cast<int>(s.p[5] + 0.5);
      auto* grid = new G4VoxelGrid(name, s.p[0] * mm, s.p[1] * mm, s.p[2] * mm, nx, ny, nz);
      // Each cell takes the material of the class its value falls in. A discrete class
      // matches its value exactly; a continuous one is a half-open band, so that adjacent
      // bands cannot both claim a voxel and leave the choice to iteration order.
      const std::size_t n = static_cast<std::size_t>(nx) * ny * nz;
      const bool have_values = (s.voxel_values.size() == n);
      const bool classify = have_values && !s.voxel_classes.empty();
      if (classify) {
        grid->ClassCells().assign(n, static_cast<short>(-1));
        // The colour and opacity each class is drawn with. `visible` off is opacity zero
        // rather than a separate flag, because the renderer's only question per cell is how
        // much of the colour behind it survives - and index 0 arrives hidden from the
        // importer, which is what lets a phantom be seen at all.
        grid->ClassColours().clear();
        grid->ClassColours().reserve(s.voxel_classes.size());
        for (const VoxelClass& vc : s.voxel_classes) {
          // VISIBILITY ONLY, and not the null layer as well. A nulled class is removed by
          // ClassAbsent below, which is the statement that its cells are not in the scene at
          // all - no material, no step, no score, and no claim on the space. Zeroing its alpha
          // here too would be a second mechanism for one of those consequences, and this
          // project's own history says the copy nobody remembers is the one that goes wrong:
          // the two would agree until someone changed how a class is nulled.
          const unsigned int al =
              vc.visible ? static_cast<unsigned int>(vc.opacity * 255.0f + 0.5f) : 0u;
          grid->ClassColours().push_back(
              (al << 24) | (static_cast<unsigned int>(vc.r * 255.0f + 0.5f) << 16)
              | (static_cast<unsigned int>(vc.g * 255.0f + 0.5f) << 8)
              | static_cast<unsigned int>(vc.b * 255.0f + 0.5f));
        }

        // The model's tag and the geometry's are the same number, and neither file can see the
      // other's. They have to agree because G4Flatten refuses a placement carrying the tag -
      // see kNullLayerTag, which is not a layer and is never compared as one.
      static_assert(kNullLayer == geom::kNullLayerTag,
                    "builder::kNullLayer and geom::kNullLayerTag must be the same value");

      // PER-CLASS LAYERS, and only if they say something the volume's own layer does not.
        //
        // A run of layers that all equal the volume's layer means exactly what no run at all
        // means, and the difference between them is what the navigator pays: with a run
        // present it asks the class at every point where it used to read one number. So the
        // test is on the VALUES rather than on whether anyone touched the control - set a
        // class's layer to the volume's own and the cost goes away again.
        std::vector<int> layers(s.voxel_classes.size(), s.layer);
        std::vector<unsigned char> absent(s.voxel_classes.size(), 0u);
        bool differs = false;
        bool any_absent = false;
        for (std::size_t ci = 0; ci < s.voxel_classes.size(); ++ci) {
          const int L = s.voxel_classes[ci].layer;
          // AN ABSENT CLASS IS NOT GIVEN A LAYER. It keeps the volume's, which is what a cell
          // with no class of its own gets, and the flag beside it is what says the cells are
          // not in the scene. The tag never travels as a layer: a number two billion below
          // everything would be read as "loses every overlap", and losing an overlap is not
          // what being absent means - see geom::kNullLayerTag for what that cost.
          if (L == kNullLayer) {
            absent[ci] = 1u;
            any_absent = true;
            continue;
          }
          if (L != kInheritLayer && L != s.layer) {
            layers[ci] = L;
            differs = true;
          }
        }
        grid->ClassLayers().clear();
        if (differs) { grid->ClassLayers() = layers; }
        grid->ClassAbsent().clear();
        if (any_absent) { grid->ClassAbsent() = absent; }
      }

      // A DIRECT MAP from value to class for the discrete case, not a search per cell.
      //
      // This was a linear scan of the class list inside the loop over cells. At the 64 classes
      // the importer used to cap at, on a small phantom, nobody noticed. A 225x225x500
      // segmentation with 261 classes is 25.3 million cells times up to 261 comparisons -
      // six billion - and it runs on every scene rebuild, which is every edit. Raising the
      // class cap without this would have turned an import into a hang.
      //
      // Discrete classes are whole numbers over a modest range (the importer requires exactly
      // that, so a class list that got here has it), so the map is an array indexed by value.
      // The continuous case keeps the scan: its classes are bands, there are nine of them, and
      // nine comparisons a cell is not worth a second mechanism.
      std::vector<short> by_value;
      double vlo = 0;
      bool direct = false;
      if (classify && s.voxel_kind == VoxelKind::kDiscrete) {
        double vhi = 0;
        vlo = s.voxel_classes.front().value;
        vhi = vlo;
        for (const VoxelClass& c : s.voxel_classes) {
          if (c.value < vlo) { vlo = c.value; }
          if (c.value > vhi) { vhi = c.value; }
        }
        const double span = vhi - vlo + 1.0;
        if (span > 0 && span < 1.0e7) {
          by_value.assign(static_cast<std::size_t>(span), static_cast<short>(-1));
          for (std::size_t ci = 0; ci < s.voxel_classes.size(); ++ci) {
            const double off = s.voxel_classes[ci].value - vlo;
            if (off >= 0 && off < span) {
              by_value[static_cast<std::size_t>(off)] = static_cast<short>(ci);
            }
          }
          direct = true;
        }
      }

      for (std::size_t idx = 0; idx < n; ++idx) {
        int mat = s.material;
        int cls = -1;
        if (classify) {
          const float v = s.voxel_values[idx];
          if (direct) {
            const double off = static_cast<double>(v) - vlo;
            if (off >= 0 && off < static_cast<double>(by_value.size())) {
              cls = by_value[static_cast<std::size_t>(off)];
            }
          } else {
            for (std::size_t ci = 0; ci < s.voxel_classes.size(); ++ci) {
              const VoxelClass& c = s.voxel_classes[ci];
              const bool hit = (s.voxel_kind == VoxelKind::kDiscrete)
                                   ? (static_cast<double>(v) == c.value)
                                   : (v >= c.value && v < c.value_max);
              if (hit) {
                cls = static_cast<int>(ci);
                break;
              }
            }
          }
          if (cls >= 0) {
            mat = s.voxel_classes[static_cast<std::size_t>(cls)].material;
            grid->ClassCells()[idx] = static_cast<short>(cls);
          }
        }
        grid->Cells()[idx] = static_cast<short>(mat);
      }
      // The cells hold MODEL material indices, so the grid is told what those mean. Without
      // this the transport reads them as device indices - see G4VoxelGrid::SetCellMaterials
      // for what that did. Passing the whole list, in model order, is what makes the numbers
      // already written correct rather than needing a second pass over 25 million cells.
      grid->SetCellMaterials(materials_);
      if (!have_values && !s.voxel_values.empty()) {
        std::printf("solid \"%s\": %zu voxel values for %zu cells; using the solid material\n",
                    s.name.c_str(), s.voxel_values.size(), n);
      }
      out = grid;
      break;
    }
    default:
      std::printf("solid \"%s\": %s cannot be built yet; skipped\n", s.name.c_str(),
                  ShapeName(s.shape));
      break;
  }
  solids_[idx] = out;
  return out;
}

/// The world placement of a solid, in G4 types. See Model::SolidWorldPlacement, which does the
/// arithmetic - here so that write_project.cc gets the same answer from the same code.
static void WorldPlacement(const Model& m, int idx, G4RotationMatrix& inv, G4ThreeVector& pos) {
  double r[9], p[3];
  SolidWorldPlacement(m, idx, r, p);
  inv = G4RotationMatrix(r);
  pos = G4ThreeVector(p[0] * mm, p[1] * mm, p[2] * mm);
}


inline G4VPhysicalVolume* ModelDetector::Construct() {
  const Model& m = *model_;
  solids_.assign(m.solids.size(), nullptr);
  logicals_.assign(m.solids.size(), nullptr);

  // Elements, then materials.
  elements_.assign(m.elements.size(), nullptr);
  for (std::size_t i = 0; i < m.elements.size(); ++i) {
    const Element& e = m.elements[i];
    elements_[i] = new G4Element(e.name, e.symbol, e.z, e.a * g / mole);
  }
  materials_.assign(m.materials.size(), nullptr);
  auto* nist = G4NistManager::Instance();
  for (std::size_t i = 0; i < m.materials.size(); ++i) {
    const Material& mm = m.materials[i];
    if (!mm.nist_name.empty()) {
      materials_[i] = nist->FindOrBuildMaterial(mm.nist_name);
      continue;
    }
    const G4State st = (mm.state == MatterState::kGas)      ? kStateGas
                       : (mm.state == MatterState::kLiquid) ? kStateLiquid
                                                            : kStateSolid;
    auto* mat = new G4Material(mm.name, mm.density * g / cm3,
                               static_cast<G4int>(std::max<std::size_t>(1, mm.components.size())),
                               st);
    for (const Component& c : mm.components) {
      if (c.element < 0 || elements_[c.element] == nullptr) { continue; }
      if (mm.fraction_kind == Fraction::kAtomCount) {
        mat->AddElement(elements_[c.element], static_cast<G4int>(c.amount + 0.5));
      } else {
        mat->AddElement(elements_[c.element], c.amount);
      }
    }
    if (mm.mean_excitation_eV > 0) {
      mat->SetMeanExcitationEnergy(mm.mean_excitation_eV * eV);
    }
    materials_[i] = mat;
  }

  // Which solids are consumed by a boolean and so are not placed on their own.
  std::vector<char> operand(m.solids.size(), 0);
  for (const Solid& s : m.solids) {
    if (s.operand_a >= 0) { operand[s.operand_a] = 1; }
    if (s.operand_b >= 0) { operand[s.operand_b] = 1; }
  }

  const int world_idx = m.world();
  for (std::size_t i = 0; i < m.solids.size(); ++i) {
    const int idx = static_cast<int>(i);
    if (operand[i] != 0) { continue; }
    // NOT PLACED, which is the whole of what the null layer means for a solid.
    //
    // Not a flag consulted by the transport, the renderer and the scorer in three places, each
    // of which could be missed: there is no volume, so there is nothing to consult. The
    // sensitive-detector pass below keys off logicals_[i], which stays null here, so the
    // tally goes with it.
    //
    // Never the world. A scene with no world has nothing to transport in at all, so the UI
    // refuses it rather than leaving this to notice - see the layer control in
    // g4builder_solids.inc.
    if (m.solids[i].layer == kNullLayer && idx != world_idx) { continue; }
    G4VSolid* solid = BuildSolid(idx);
    if (solid == nullptr) { continue; }

    const Solid& s = m.solids[i];
    G4Material* mat = (s.material >= 0 && s.material < static_cast<int>(materials_.size()))
                          ? materials_[s.material]
                          : nullptr;
    if (mat == nullptr) {
      // A volume with no material cannot be transported through. Falling back to the world's
      // material would hide the mistake; refusing to place it makes the gap visible in the
      // volume list and in the render.
      std::printf("solid \"%s\" has no material assigned; not placed\n", s.name.c_str());
      continue;
    }
    auto* logic = new G4LogicalVolume(solid, mat, s.name);
    logicals_[i] = logic;

    auto vis = std::make_unique<G4VisAttributes>(s.visible,
                                                 G4Colour(s.r, s.g, s.b, s.opacity));
    if (s.wireframe) { vis->SetForceWireframe(true); }
    logic->SetVisAttributes(vis.get());
    vis_.push_back(std::move(vis));

    // The rotation and position after the anchor chain has been followed. WorldPlacement
    // builds the same world -> solid map an unanchored placement would - the model stores the
    // rotation of the *object* and Geant4's matrix is its inverse, hence the negated angles
    // there - and composes it with whatever the volume is anchored to.
    G4RotationMatrix world_rot;
    G4ThreeVector pos;
    WorldPlacement(*model_, idx, world_rot, pos);
    G4RotationMatrix* rot = nullptr;
    const bool rotated = s.rot[0] != 0 || s.rot[1] != 0 || s.rot[2] != 0 || s.anchor >= 0;
    if (rotated) { rot = new G4RotationMatrix(world_rot); }

    if (idx == world_idx) {
      world_ = new G4PVPlacement(rot, pos, logic, s.name, nullptr, false, 0, true);
    } else {
      new G4PVPlacement(rot, pos, logic, s.name, s.layer);
    }
  }

  if (world_ == nullptr) {
    std::printf("\nFATAL: the model has no usable world volume (solid 0).\n");
    std::exit(2);
  }
  return world_;
}

inline void ModelDetector::ConstructSDandField() {
  const Model& m = *model_;
  for (const Scorer& sc : m.scorers) {
    const char* prim_name = sc.name.c_str();
    auto* det = new G4MultiFunctionalDetector(sc.name + "SD");
    // Per-voxel scoring replaces the primitive with the 3D one, which is what makes the
    // stepper stop at every cell boundary and the deposit land in one cell. It is only
    // offered for energy deposit and dose: a per-cell track length or step count is a
    // different quantity from the volume's, and Geant4 has separate scorers for those.
    if (sc.per_voxel && (sc.quantity == ScoreQuantity::kEnergyDeposit
                         || sc.quantity == ScoreQuantity::kDose)) {
      det->RegisterPrimitive(new G4PSEnergyDeposit3D(prim_name));
    } else {
      switch (sc.quantity) {
        case ScoreQuantity::kDose:
          det->RegisterPrimitive(new G4PSDoseDeposit(prim_name));
          break;
        case ScoreQuantity::kFluence:
          det->RegisterPrimitive(new G4PSCellFlux(prim_name));
          break;
        case ScoreQuantity::kTrackLength:
          det->RegisterPrimitive(new G4PSTrackLength(prim_name));
          break;
        case ScoreQuantity::kStepCount:
          det->RegisterPrimitive(new G4PSNofStep(prim_name));
          break;
        case ScoreQuantity::kEnergyDeposit:
        default:
          det->RegisterPrimitive(new G4PSEnergyDeposit(prim_name));
          break;
      }
    }
    bool attached = false;
    for (int v : sc.solids) {
      if (v >= 0 && v < static_cast<int>(logicals_.size()) && logicals_[v] != nullptr) {
        SetSensitiveDetector(logicals_[v], det);
        attached = true;
      }
    }
    if (!attached) {
      std::printf("scorer \"%s\" is not attached to any placed volume; it will read zero\n",
                  sc.name.c_str());
    }
  }
}

/// Configures the gun for one source of the model.
inline void ModelPrimary::ConfigureFor(const Source& src) {
  gun_->SetParticleDefinition(
      G4ParticleTable::GetParticleTable()->FindParticle(src.particle));
  if (src.spectrum_file.empty()) {
    gun_->SetParticleEnergy(src.energy_MeV * MeV);
  } else {
    gun_->SetEnergySpectrumFromCSV(src.spectrum_file);
  }
  gun_->SetParticlePosition(
      G4ThreeVector(src.pos[0] * mm, src.pos[1] * mm, src.pos[2] * mm));
  gun_->SetParticleMomentumDirection(G4ThreeVector(src.dir[0], src.dir[1], src.dir[2]));

  switch (src.kind) {
    case SourceKind::kBeam:
      if (src.profile == BeamProfile::kElliptical) {
        gun_->SetBeamCrossSectionElliptical(src.half_x * mm, src.half_y * mm);
      } else {
        gun_->SetBeamCrossSectionRectangular(src.half_x * mm, src.half_y * mm);
      }
      break;
    case SourceKind::kPoint:
      if (src.angular_spread_deg > 0) {
        gun_->SetAngularSpread(src.angular_spread_deg * deg);
      }
      break;
    case SourceKind::kIsotropicShell:
      gun_->SetIsotropicShell(src.radius * mm);
      break;
    case SourceKind::kVolume:
      gun_->SetSourceVolumeBox(src.half_x * mm, src.half_y * mm, src.half_z * mm);
      break;
    case SourceKind::kRadioactive:
      // Decay sampling is not implemented; an isotropic source of the configured energy is
      // the closest honest stand-in, and the GUI says so next to the field.
      gun_->SetIsotropic();
      break;
  }
}

/// Builds this event's primary. Called once per event, as Geant4 calls it.
///
/// With several enabled sources, one is chosen per event by weight. That is now the natural
/// thing to write - a uniform draw against a cumulative weight - because the method really is
/// called per event. It was not always: this used to be called once per run with the device
/// sampling a Source record, and a version of exactly this loop picked one source for the
/// whole run. The architecture is what was wrong, not the loop.
///
/// A run of any size therefore gives each source its share, and per event a source gets all
/// or nothing, which is what a source is.
inline void ModelPrimary::GeneratePrimaries(G4Event* event) {
  if (selected_ >= 0 && selected_ < static_cast<int>(model_->sources.size())) {
    ConfigureFor(model_->sources[selected_]);
    gun_->GeneratePrimaryVertex(event);
    return;
  }

  G4double wsum = 0;
  for (const Source& s : model_->sources) {
    if (s.enabled && s.weight > 0) { wsum += s.weight; }
  }
  if (wsum <= 0) { return; }

  const G4double u = G4UniformRand() * wsum;
  G4double run = 0;
  for (const Source& s : model_->sources) {
    if (!s.enabled || s.weight <= 0) { continue; }
    run += s.weight;
    if (u < run) {
      ConfigureFor(s);
      gun_->GeneratePrimaryVertex(event);
      return;
    }
  }
  // Only reachable if u lands exactly on wsum, which G4UniformRand does not return; the last
  // enabled source is the right answer if it ever does.
  for (std::size_t i = model_->sources.size(); i-- > 0;) {
    const Source& s = model_->sources[i];
    if (s.enabled && s.weight > 0) {
      ConfigureFor(s);
      gun_->GeneratePrimaryVertex(event);
      return;
    }
  }
}

}  // namespace g4gpu::builder
