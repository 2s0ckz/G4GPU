// G4PVPlacement: one placement of a logical volume.
//
// Geant4's constructor signature is kept exactly, mother logical volume and all. The mother is
// not decoration and it is not a layer under another name: it *derives* the layer, as
// mother.layer + 1, and the placement's transform composes with the mother's. An existing
// DetectorConstruction therefore compiles and produces the scene it always did, expressed in
// the layer model - see docs/ROADMAP.md and src/geometry/navigator.cuh.
//
// One extra constructor takes an explicit layer instead of a mother, for what the hierarchy
// cannot express: volumes that overlap on purpose, or a solid straddling what would have been
// a mother boundary. That is the only thing new here.
//
// The rotation argument follows Geant4's convention, which is the inverse of the intuitive
// one: the matrix is the mother-to-daughter map, so `rot->rotateY(35*deg)` tilts the *frame*
// by +35 degrees and therefore the object by -35. This is preserved deliberately; "fixing" it
// would make existing detector code place things differently here than in Geant4.
#pragma once
#include <vector>
#include "g4/G4LogicalVolume.hh"
#include "g4/G4ThreeVector.hh"

/// Layer assigned when a placement names no mother: the world.
inline constexpr G4int kWorldLayer = 0;

class G4PVPlacement {
 public:
  /// Geant4's principal constructor.
  G4PVPlacement(G4RotationMatrix* pRot, const G4ThreeVector& tlate, G4LogicalVolume* pLogical,
                const G4String& pName, G4LogicalVolume* pMotherLogical, G4bool pMany,
                G4int pCopyNo, G4bool pSurfChk = false)
      : logical_(pLogical), mother_(pMotherLogical), name_(pName), translation_(tlate),
        copy_no_(pCopyNo), has_rot_(pRot != nullptr) {
    (void)pMany;
    (void)pSurfChk;
    if (pRot != nullptr) {
      const G4double* m = pRot->data();
      for (int i = 0; i < 9; ++i) { rot_[i] = m[i]; }
    }
    Registry().push_back(this);
  }

  /// The layer-model constructor: no mother, an explicit layer instead.
  G4PVPlacement(G4RotationMatrix* pRot, const G4ThreeVector& tlate, G4LogicalVolume* pLogical,
                const G4String& pName, G4int layer)
      : logical_(pLogical), mother_(nullptr), name_(pName), translation_(tlate),
        explicit_layer_(layer), has_rot_(pRot != nullptr) {
    if (pRot != nullptr) {
      const G4double* m = pRot->data();
      for (int i = 0; i < 9; ++i) { rot_[i] = m[i]; }
    }
    Registry().push_back(this);
  }

  G4LogicalVolume* GetLogicalVolume() const { return logical_; }
  G4LogicalVolume* GetMotherLogical() const { return mother_; }
  const G4String& GetName() const { return name_; }
  const G4ThreeVector& GetTranslation() const { return translation_; }
  G4int GetCopyNo() const { return copy_no_; }
  G4bool HasRotation() const { return has_rot_; }
  const G4double* RotationData() const { return rot_; }
  G4int ExplicitLayer() const { return explicit_layer_; }

  /// Anchors this placement to another, so that moving or rotating the anchor carries this
  /// one with it. Purely an authoring relationship: the scene is flattened to absolute world
  /// transforms before upload, so the navigator never sees an anchor.
  void AnchorTo(G4PVPlacement* anchor) { anchor_ = anchor; }
  G4PVPlacement* GetAnchor() const { return anchor_; }

  static std::vector<G4PVPlacement*>& Registry() {
    static std::vector<G4PVPlacement*> r;
    return r;
  }

  /// Index in the flattened device volume table, filled in by the flattener. -1 until then.
  G4int device_index = -1;

 private:
  G4LogicalVolume* logical_;
  G4LogicalVolume* mother_;
  G4String name_;
  G4ThreeVector translation_;
  G4int copy_no_ = 0;
  G4int explicit_layer_ = -1;  ///< -1 means "derive from the mother"
  G4double rot_[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  G4bool has_rot_ = false;
  G4PVPlacement* anchor_ = nullptr;
};

// Geant4 distinguishes the abstract G4VPhysicalVolume from its concrete G4PVPlacement. Here
// there is only placement, so the abstract name is an alias - which keeps Construct()'s
// return type, and a touchable's GetVolume(), reading the way they do in every Geant4 example.
// G4UserActions.hh declares the same alias; a typedef may be repeated for the same type.
using G4VPhysicalVolume = G4PVPlacement;
