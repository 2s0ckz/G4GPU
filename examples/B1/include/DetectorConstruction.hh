/// \file DetectorConstruction.hh
/// \brief Definition of the B1::DetectorConstruction class

#ifndef B1DetectorConstruction_h
#define B1DetectorConstruction_h 1

#include "G4VUserDetectorConstruction.hh"
#include "globals.hh"

// forward-declared as an alias target; see G4UserActions.hh
class G4LogicalVolume;

namespace B1 {

/// Detector construction class to define materials and geometry.
/// The calorimeter is a box made of a given number of layers. A layer consists
/// of an absorber plate and of a detection gap. The layer is replicated.
class DetectorConstruction : public G4VUserDetectorConstruction {
 public:
  DetectorConstruction() = default;
  ~DetectorConstruction() override = default;

  G4VPhysicalVolume* Construct() override;
  void ConstructSDandField() override;

  G4LogicalVolume* GetScoringVolume() const { return fScoringVolume; }

  /// The ONE command this example adds to Geant4's B1 (P27, docs/RISK.md V336):
  ///
  ///   /B1/det/material <NIST name>     the scoring trapezoid's (Shape2's) material
  ///
  /// It must come before /run/initialize, because Construct() reads it - the same rule as the
  /// G4UImessenger of ref/b1mat, the Geant4 build the acceptance sweep's Geant4 columns run.
  /// Unset, Shape2 is G4_BONE_COMPACT_ICRU exactly as B1 ships it, so every shipped macro and
  /// every gate is unchanged: none of them names the command.
  ///
  /// This port's G4UImanager has no G4UImessenger to register with, so exampleB1.cc offers each
  /// macro line here first. The return is -1 for a line that is not this class's, 0 for one
  /// applied and 1 for one refused (no name given, or the command after /run/initialize, when
  /// Geant4's messenger would be out of its G4State_PreInit window too).
  G4int ApplyCommand(const G4String& line);
  const G4String& GetShape2Material() const { return fShape2Material; }

 protected:
  G4LogicalVolume* fScoringVolume = nullptr;
  G4String fShape2Material = "G4_BONE_COMPACT_ICRU";
  G4bool fConstructed = false;
};

}  // namespace B1

#endif
