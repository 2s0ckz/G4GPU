// Example B1 with its scoring volume replaced by a triangle mesh of the same shape.
//
// This exists to check mesh transport end to end, and it is built to be the strongest check
// available rather than a demonstration. B1's Shape2 is a G4Trd - a trapezoid with six planar
// faces - so a twelve-triangle mesh of it is not an approximation of the solid, it *is* the
// solid, to the last bit of the vertex coordinates. Everything else in the scene, the source
// and the random seed are identical.
//
// So the dose from this scene and the dose from "B1" must agree. Not approximately, not
// within some tolerance chosen to make it pass: the same geometry with the same seed makes the
// same physics decisions. They are not bit-identical, because the distance to a triangle and
// the distance to a plane differ in the last few digits of a double and a step that ends
// 1e-14 mm further along resamples the next interaction from the same stream at a marginally
// different point - but the difference has to be far below the statistical error of the run.
//
// What this catches that tests/test_mesh.cu cannot: the upload path. test_mesh.cu runs the
// mesh routines on the host against host pools. A triangle pool that was allocated but not
// copied, or a BVH root index that survived the host but not the struct that carries it to
// the device, would pass every check in that file and produce a scene here that particles
// pass straight through.
#include "scenes/scene_registry.hh"

namespace g4gpu::scenes {
namespace {

/// The twelve triangles of a G4Trd: half-lengths dx1,dx2 in x at -dz,+dz, dy1,dy2 in y.
///
/// Vertex order matters only for the winding, which the ray-triangle test ignores; what
/// matters is that the eight corners are exactly the corners of the G4Trd, so they are
/// written out from the same five numbers the G4Trd constructor takes.
std::vector<G4double> TrdMesh(G4double dx1, G4double dx2, G4double dy1, G4double dy2,
                              G4double dz) {
  const G4double v[8][3] = {
      {-dx1, -dy1, -dz}, {dx1, -dy1, -dz}, {dx1, dy1, -dz}, {-dx1, dy1, -dz},
      {-dx2, -dy2, dz},  {dx2, -dy2, dz},  {dx2, dy2, dz},  {-dx2, dy2, dz}};
  const int f[12][3] = {{0, 1, 2}, {0, 2, 3},   // -z
                        {4, 6, 5}, {4, 7, 6},   // +z
                        {0, 5, 1}, {0, 4, 5},   // -y
                        {3, 2, 6}, {3, 6, 7},   // +y
                        {0, 3, 7}, {0, 7, 4},   // -x
                        {1, 5, 6}, {1, 6, 2}};  // +x
  std::vector<G4double> tri;
  tri.reserve(12 * 9);
  for (const auto& t : f) {
    for (int k = 0; k < 3; ++k) {
      for (int c = 0; c < 3; ++c) { tri.push_back(v[t[k]][c]); }
    }
  }
  return tri;
}

class B1MeshDetector : public G4VUserDetectorConstruction {
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
    auto* logicShape1 =
        new G4LogicalVolume(solidShape1, nist->FindOrBuildMaterial("G4_A-150_TISSUE"),
                            "Shape1");
    static const G4VisAttributes shape1_vis(G4Colour(0.85, 0.70, 0.45));
    logicShape1->SetVisAttributes(&shape1_vis);
    new G4PVPlacement(nullptr, G4ThreeVector(0, 2 * cm, -7 * cm), logicShape1, "Shape1",
                      logicEnv, false, 0, true);

    // The one difference from the B1 scene: a mesh instead of a G4Trd, from the same numbers.
    auto* solidShape2 = new G4TessellatedSolid("Shape2");
    solidShape2->AddTriangles(TrdMesh(6 * cm, 6 * cm, 5 * cm, 8 * cm, 3 * cm));
    solidShape2->SetSolidClosed(true);
    auto* logicShape2 =
        new G4LogicalVolume(solidShape2, nist->FindOrBuildMaterial("G4_BONE_COMPACT_ICRU"),
                            "Shape2");
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

class B1MeshPrimary : public G4VUserPrimaryGeneratorAction {
 public:
  B1MeshPrimary() {
    gun_ = new G4ParticleGun(1);
    gun_->SetParticleDefinition(G4ParticleTable::GetParticleTable()->FindParticle("gamma"));
    gun_->SetParticleMomentumDirection(G4ThreeVector(0, 0, 1));
    gun_->SetParticleEnergy(6 * MeV);
    if (G4RunManager::Instance() != nullptr) { G4RunManager::Instance()->SetGun(gun_); }
  }
  ~B1MeshPrimary() override { delete gun_; }

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

// [[maybe_unused]] because the work happens in the constructor, not through the name.
[[maybe_unused]] const AutoRegister kB1Mesh("B1mesh", [](G4RunManager* rm) {
  rm->SetUserInitialization(new B1MeshDetector);
  rm->SetUserAction(new B1MeshPrimary);
  rm->SetCutValue(0.7 * mm);
});

}  // namespace
}  // namespace g4gpu::scenes
