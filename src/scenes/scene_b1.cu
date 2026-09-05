// Example B1 as a viewer scene: the same geometry, gun and scoring as examples/B1, with
// visualisation attributes added so the shapes are distinguishable on screen.
//
// This duplicates examples/B1's detector description rather than including it, deliberately.
// examples/B1 is the reference for what a user's own project looks like and is validated
// against Geant4 to 0.09 sigma; keeping the viewer's copy separate means the viewer can gain
// colours, extra scorers and camera hints without any of that appearing in the example a user
// is meant to read and copy.
#include "scenes/scene_registry.hh"

namespace g4gpu::scenes {
namespace {

class B1Detector : public G4VUserDetectorConstruction {
 public:
  G4VPhysicalVolume* Construct() override {
    auto* nist = G4NistManager::Instance();
    const G4double env_xy = 20 * cm, env_z = 30 * cm;

    auto* solidWorld = new G4Box("World", 0.6 * env_xy, 0.6 * env_xy, 0.6 * env_z);
    auto* logicWorld =
        new G4LogicalVolume(solidWorld, nist->FindOrBuildMaterial("G4_AIR"), "World");
    static G4VisAttributes world_vis(false, G4Colour(0.35, 0.35, 0.40));
    world_vis.SetForceWireframe(true);
    logicWorld->SetVisAttributes(&world_vis);
    auto* physWorld = new G4PVPlacement(nullptr, G4ThreeVector(), logicWorld, "World", nullptr,
                                        false, 0, true);

    auto* solidEnv = new G4Box("Envelope", 0.5 * env_xy, 0.5 * env_xy, 0.5 * env_z);
    auto* logicEnv =
        new G4LogicalVolume(solidEnv, nist->FindOrBuildMaterial("G4_WATER"), "Envelope");
    static G4VisAttributes env_vis(G4Colour(0.30, 0.45, 0.65));
    env_vis.SetForceWireframe(true);
    logicEnv->SetVisAttributes(&env_vis);
    new G4PVPlacement(nullptr, G4ThreeVector(), logicEnv, "Envelope", logicWorld, false, 0,
                      true);

    auto* solidShape1 = new G4Cons("Shape1", 0, 2 * cm, 0, 4 * cm, 3 * cm, 0, 360 * deg);
    auto* logicShape1 = new G4LogicalVolume(
        solidShape1, nist->FindOrBuildMaterial("G4_A-150_TISSUE"), "Shape1");
    static const G4VisAttributes shape1_vis(G4Colour(0.85, 0.70, 0.45));
    logicShape1->SetVisAttributes(&shape1_vis);
    new G4PVPlacement(nullptr, G4ThreeVector(0, 2 * cm, -7 * cm), logicShape1, "Shape1",
                      logicEnv, false, 0, true);

    auto* solidShape2 =
        new G4Trd("Shape2", 6 * cm, 6 * cm, 5 * cm, 8 * cm, 3 * cm);
    auto* logicShape2 = new G4LogicalVolume(
        solidShape2, nist->FindOrBuildMaterial("G4_BONE_COMPACT_ICRU"), "Shape2");
    static const G4VisAttributes shape2_vis(G4Colour(0.90, 0.88, 0.80));
    logicShape2->SetVisAttributes(&shape2_vis);
    new G4PVPlacement(nullptr, G4ThreeVector(0, -1 * cm, 7 * cm), logicShape2, "Shape2",
                      logicEnv, false, 0, true);

    scoring_ = logicShape2;
    return physWorld;
  }

  void ConstructSDandField() override {
    auto* det = new G4MultiFunctionalDetector("Shape2SD");
    det->RegisterPrimitive(new G4PSEnergyDeposit("edep"));
    SetSensitiveDetector(scoring_, det);
  }

 private:
  G4LogicalVolume* scoring_ = nullptr;
};

class B1Primary : public G4VUserPrimaryGeneratorAction {
 public:
  B1Primary() {
    gun_ = new G4ParticleGun(1);
    gun_->SetParticleDefinition(G4ParticleTable::GetParticleTable()->FindParticle("gamma"));
    gun_->SetParticleMomentumDirection(G4ThreeVector(0, 0, 1));
    gun_->SetParticleEnergy(6 * MeV);
    if (G4RunManager::Instance() != nullptr) { G4RunManager::Instance()->SetGun(gun_); }
  }
  ~B1Primary() override { delete gun_; }

  void GeneratePrimaries(G4Event* event) override {
    // Once per event, as Geant4 calls it. The cross-section on the gun is sampled by
    // GeneratePrimaryVertex below, on the host, so the distribution is what it always was.
    gun_->SetParticlePosition(G4ThreeVector(0, 0, -15 * cm));
    gun_->SetBeamCrossSectionRectangular(8 * cm, 8 * cm);
    gun_->GeneratePrimaryVertex(event);
  }

 private:
  G4ParticleGun* gun_ = nullptr;
};

// [[maybe_unused]] because the work happens in the constructor, not through the name. Without
// it nvcc warns, and a compiler that acted on that warning would drop the scene entirely.
[[maybe_unused]] const AutoRegister kB1("B1", [](G4RunManager* rm) {
  rm->SetUserInitialization(new B1Detector);
  rm->SetUserAction(new B1Primary);
  rm->SetCutValue(0.7 * mm);
});

}  // namespace
}  // namespace g4gpu::scenes
