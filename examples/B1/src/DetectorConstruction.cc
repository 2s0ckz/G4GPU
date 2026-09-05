/// \file DetectorConstruction.cc
/// \brief Implementation of the B1::DetectorConstruction class
///
/// This is example B1's geometry, written the way it is written in Geant4: NIST materials, a
/// world, an envelope, a cone and a trapezoid, and a scoring volume. The only difference from
/// the upstream file is that the sensitive detector is attached with a G4MultiFunctionalDetector
/// carrying a G4PSEnergyDeposit, rather than through a stepping action - which is closer to
/// how a modern Geant4 example scores anyway.

#include "DetectorConstruction.hh"

#include "G4Box.hh"
#include "G4Cons.hh"
#include "G4LogicalVolume.hh"
#include "G4NistManager.hh"
#include "G4PVPlacement.hh"
#include "G4SDManager.hh"
#include "G4SystemOfUnits.hh"
#include "G4Trd.hh"

namespace B1 {

G4VPhysicalVolume* DetectorConstruction::Construct() {
  // Get nist material manager
  G4NistManager* nist = G4NistManager::Instance();

  // Envelope parameters
  G4double env_sizeXY = 20 * cm, env_sizeZ = 30 * cm;
  G4Material* env_mat = nist->FindOrBuildMaterial("G4_WATER");

  // World
  G4double world_sizeXY = 1.2 * env_sizeXY;
  G4double world_sizeZ = 1.2 * env_sizeZ;
  G4Material* world_mat = nist->FindOrBuildMaterial("G4_AIR");

  auto* solidWorld =
      new G4Box("World", 0.5 * world_sizeXY, 0.5 * world_sizeXY, 0.5 * world_sizeZ);
  auto* logicWorld = new G4LogicalVolume(solidWorld, world_mat, "World");
  auto* physWorld = new G4PVPlacement(nullptr,           // no rotation
                                      G4ThreeVector(),   // at (0,0,0)
                                      logicWorld,        // its logical volume
                                      "World",           // its name
                                      nullptr,           // its mother volume
                                      false,             // no boolean operation
                                      0,                 // copy number
                                      true);             // checking overlaps

  // Envelope
  auto* solidEnv =
      new G4Box("Envelope", 0.5 * env_sizeXY, 0.5 * env_sizeXY, 0.5 * env_sizeZ);
  auto* logicEnv = new G4LogicalVolume(solidEnv, env_mat, "Envelope");
  new G4PVPlacement(nullptr, G4ThreeVector(), logicEnv, "Envelope", logicWorld, false, 0, true);

  // Shape 1: a cone of A-150 tissue
  G4Material* shape1_mat = nist->FindOrBuildMaterial("G4_A-150_TISSUE");
  G4ThreeVector pos1 = G4ThreeVector(0, 2 * cm, -7 * cm);

  G4double shape1_rmina = 0. * cm, shape1_rmaxa = 2. * cm;
  G4double shape1_rminb = 0. * cm, shape1_rmaxb = 4. * cm;
  G4double shape1_hz = 3. * cm;
  G4double shape1_phimin = 0. * deg, shape1_phimax = 360. * deg;
  auto* solidShape1 = new G4Cons("Shape1", shape1_rmina, shape1_rmaxa, shape1_rminb,
                                 shape1_rmaxb, shape1_hz, shape1_phimin, shape1_phimax);
  auto* logicShape1 = new G4LogicalVolume(solidShape1, shape1_mat, "Shape1");
  new G4PVPlacement(nullptr, pos1, logicShape1, "Shape1", logicEnv, false, 0, true);

  // Shape 2: a trapezoid of compact bone
  G4Material* shape2_mat = nist->FindOrBuildMaterial("G4_BONE_COMPACT_ICRU");
  G4ThreeVector pos2 = G4ThreeVector(0, -1 * cm, 7 * cm);

  G4double shape2_dxa = 12 * cm, shape2_dxb = 12 * cm;
  G4double shape2_dya = 10 * cm, shape2_dyb = 16 * cm;
  G4double shape2_dz = 6 * cm;
  auto* solidShape2 = new G4Trd("Shape2", 0.5 * shape2_dxa, 0.5 * shape2_dxb, 0.5 * shape2_dya,
                                0.5 * shape2_dyb, 0.5 * shape2_dz);
  auto* logicShape2 = new G4LogicalVolume(solidShape2, shape2_mat, "Shape2");
  new G4PVPlacement(nullptr, pos2, logicShape2, "Shape2", logicEnv, false, 0, true);

  // Set Shape2 as the scoring volume
  fScoringVolume = logicShape2;

  return physWorld;
}

/// Attaches the sensitive detector.
///
/// Geant4's B1 has no ConstructSDandField: its dose comes entirely from the stepping action,
/// which can decide per step whether the step was in the scoring volume. This one does,
/// because the steps happen on the device and the decision has to be made before the run
/// rather than during it: a volume with a sensitive detector gets a scorer slot, and the
/// stepping kernels accumulate into it. The stepping action still runs, and still reads the
/// energy deposit off the step it is given - see src/SteppingAction.cc - but what makes the
/// number exist is this method.
///
/// So this is the one addition to B1's detector description, and it is the one place where
/// "the same example" needed a line Geant4 does not have.
void DetectorConstruction::ConstructSDandField() {
  auto* det = new G4MultiFunctionalDetector("Shape2SD");
  det->RegisterPrimitive(new G4PSEnergyDeposit("edep"));
  SetSensitiveDetector(fScoringVolume, det);
}

}  // namespace B1
