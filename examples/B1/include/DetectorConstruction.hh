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

 protected:
  G4LogicalVolume* fScoringVolume = nullptr;
};

}  // namespace B1

#endif
