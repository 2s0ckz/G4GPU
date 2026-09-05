// G4LogicalVolume and G4VisAttributes.
//
// A logical volume is a solid plus a material plus how it should look. It is defined once and
// may be placed any number of times, exactly as in Geant4 - the placements live in
// G4PVPlacement.hh.
#pragma once
#include <vector>
#include "g4/G4Material.hh"
#include "g4/G4VSolid.hh"
#include "geometry/volume_of.cuh"

/// RGBA in 0..1, with Geant4's named constructors.
class G4Colour {
 public:
  G4Colour(G4double r = 1, G4double g = 1, G4double b = 1, G4double a = 1)
      : r_(r), g_(g), b_(b), a_(a) {}
  G4double GetRed() const { return r_; }
  G4double GetGreen() const { return g_; }
  G4double GetBlue() const { return b_; }
  G4double GetAlpha() const { return a_; }

  static G4Colour White() { return {1, 1, 1}; }
  static G4Colour Gray() { return {0.5, 0.5, 0.5}; }
  static G4Colour Grey() { return {0.5, 0.5, 0.5}; }
  static G4Colour Black() { return {0, 0, 0}; }
  static G4Colour Brown() { return {0.7, 0.4, 0.1}; }
  static G4Colour Red() { return {1, 0, 0}; }
  static G4Colour Green() { return {0, 1, 0}; }
  static G4Colour Blue() { return {0, 0, 1}; }
  static G4Colour Cyan() { return {0, 1, 1}; }
  static G4Colour Magenta() { return {1, 0, 1}; }
  static G4Colour Yellow() { return {1, 1, 0}; }

 private:
  G4double r_, g_, b_, a_;
};

class G4VisAttributes {
 public:
  G4VisAttributes() = default;
  explicit G4VisAttributes(G4bool visible) : visible_(visible) {}
  explicit G4VisAttributes(const G4Colour& c) : colour_(c) {}
  G4VisAttributes(G4bool visible, const G4Colour& c) : visible_(visible), colour_(c) {}

  void SetVisibility(G4bool v) { visible_ = v; }
  void SetColour(const G4Colour& c) { colour_ = c; }
  void SetColor(const G4Colour& c) { colour_ = c; }
  void SetForceWireframe(G4bool v) { wireframe_ = v; }
  void SetForceSolid(G4bool v) { wireframe_ = !v; }

  G4bool IsVisible() const { return visible_; }
  const G4Colour& GetColour() const { return colour_; }
  G4bool IsForceWireframe() const { return wireframe_; }

  /// Geant4's sentinel for "not set"; a null G4VisAttributes* means the same thing.
  static const G4VisAttributes* GetInvisible() {
    static const G4VisAttributes inv(false);
    return &inv;
  }

 private:
  G4bool visible_ = true;
  G4bool wireframe_ = false;
  G4Colour colour_{0.8, 0.8, 0.8, 1.0};
};

class G4LogicalVolume {
 public:
  G4LogicalVolume(G4VSolid* solid, G4Material* material, const G4String& name)
      : solid_(solid), material_(material), name_(name) {
    Registry().push_back(this);
  }

  G4VSolid* GetSolid() const { return solid_; }
  G4Material* GetMaterial() const { return material_; }
  const G4String& GetName() const { return name_; }

  void SetVisAttributes(const G4VisAttributes* v) { vis_ = v; }
  void SetVisAttributes(const G4VisAttributes& v) { vis_ = &v; }
  const G4VisAttributes* GetVisAttributes() const { return vis_; }

  /// Marks this volume as a scorer target. The dose reported at end of run is summed over
  /// every volume so marked, which is how example B1's fScoringVolume works.
  void SetSensitive(G4bool s) { sensitive_ = s; }
  G4bool IsSensitive() const { return sensitive_; }

  /// The volume of the solid, in internal units.
  ///
  /// Closed form where one exists, and a Monte Carlo estimate where it does not - which is
  /// what G4VSolid::GetCubicVolume() does in Geant4 too. See geometry/volume_of.cuh.
  G4double GetCubicVolume() const {
    g4gpu::g4::SolidPool pool;
    const G4int idx = solid_->Build(pool);
    const auto store = pool.store();
    return g4gpu::geom::solid_volume(store, pool.solids[idx]) * mm3;
  }

  /// The mass of the solid times its material density, in internal units.
  ///
  /// Geant4 subtracts the mass of the daughter volumes, because a daughter displaces its
  /// mother's material. This does not: in the layer model any number of volumes may overlap,
  /// and which one owns a point is decided per point by the layer, so there is no daughter
  /// list to subtract. For a volume with nothing inside it - a scoring volume, normally -
  /// the two agree exactly. For one that contains others, use G4RunManager::ScoredMass,
  /// which has the flattened scene and integrates over the region this volume actually wins.
  ///
  /// The arguments are Geant4's and are accepted for source compatibility; there is no
  /// cached value to force a recomputation of, and nothing to propagate to.
  G4double GetMass(G4bool forced = false, G4bool propagate = true,
                   G4Material* parMaterial = nullptr) const {
    (void)forced;
    (void)propagate;
    const G4Material* mat = (parMaterial != nullptr) ? parMaterial : material_;
    if (mat == nullptr) { return 0; }
    return GetCubicVolume() * mat->GetDensity();
  }

  static std::vector<G4LogicalVolume*>& Registry() {
    static std::vector<G4LogicalVolume*> r;
    return r;
  }

 private:
  G4VSolid* solid_;
  G4Material* material_;
  G4String name_;
  const G4VisAttributes* vis_ = nullptr;
  G4bool sensitive_ = false;
};
