// SetParticlePosition means the primary starts there, and SetParticleMomentumDirection means it
// travels that way. Both of them exactly.
//
// This exists because of a bug that made one example disagree with itself. examples/B1's
// vis.mac applied `/gun/beamRectangular 8 8 cm`, so that pressing Run in the viewer fired what
// the example fires. Its generator then drew its own beam spot in +-8 cm and called
// SetParticlePosition. The cross-section was still set, so the position was sampled a *second*
// time: +-16 cm against a 20 cm envelope, a third of the beam missing the detector, and 287 pGy
// where the same example in batch - no macro, so no cross-section - gave 429.
//
// Neither number looked wrong on its own. The batch run agreed with Geant4 to 0.015 sigma and
// the interactive run reported a plausible dose. What was wrong was that they were the same
// example.
//
// In Geant4 the question does not arise: G4ParticleGun has no spatial distribution at all, so
// SetParticlePosition cannot be the centre of anything. The distributions here are an
// extension, and an extension that changes what a Geant4 method means is a trap for exactly the
// code most likely to be written - code copied from a Geant4 example.
#include <cstdio>

#include "g4/G4ParticleGun.hh"
#include "g4/G4SystemOfUnits.hh"
#include "g4/G4UserActions.hh"
#include "g4/Randomize.hh"

namespace {

int failures = 0;

void expect(bool ok, const char* what) {
  if (!ok) {
    std::printf("FAIL: %s\n", what);
    ++failures;
  }
}

/// The vertex one call to the gun produces.
G4ThreeVector VertexOf(G4ParticleGun& gun) {
  G4Event event(0);
  gun.GeneratePrimaryVertex(&event);
  if (event.GetNumberOfPrimaryVertex() != 1) { return G4ThreeVector(1e30, 1e30, 1e30); }
  return event.GetPrimaryVertex(0).GetPosition();
}

G4ThreeVector DirectionOf(G4ParticleGun& gun) {
  G4Event event(0);
  gun.GeneratePrimaryVertex(&event);
  if (event.GetNumberOfPrimaryVertex() != 1) { return G4ThreeVector(0, 0, 0); }
  return event.GetPrimaryVertex(0).GetPrimary(0).GetMomentumDirection();
}

}  // namespace

int main() {
  G4Random::setTheSeed(12345);
  auto* table = G4ParticleTable::GetParticleTable();

  // ---- a cross-section, then a position: the position wins, exactly.
  {
    G4ParticleGun gun(1);
    gun.SetParticleDefinition(table->FindParticle("gamma"));
    gun.SetParticleEnergy(6 * MeV);
    gun.SetParticleMomentumDirection(G4ThreeVector(0, 0, 1));
    gun.SetBeamCrossSectionRectangular(8 * cm, 8 * cm);
    gun.SetParticlePosition(G4ThreeVector(1 * cm, -2 * cm, -15 * cm));

    // Every vertex at the named point. Sampled a few hundred times, because the failure being
    // guarded against is a *distribution* - one draw could land near the centre by luck.
    bool all_exact = true;
    for (int i = 0; i < 500; ++i) {
      const G4ThreeVector v = VertexOf(gun);
      if (std::abs(v.x() - 1 * cm) > 1e-9 * cm || std::abs(v.y() + 2 * cm) > 1e-9 * cm
          || std::abs(v.z() + 15 * cm) > 1e-9 * cm) {
        all_exact = false;
        std::printf("  vertex %d at (%.4f, %.4f, %.4f) cm, expected (1, -2, -15)\n", i,
                    v.x() / cm, v.y() / cm, v.z() / cm);
        break;
      }
    }
    expect(all_exact, "SetParticlePosition after a beam cross-section gives a point source");
  }

  // ---- the documented order still gives a beam: position, then cross-section.
  {
    G4ParticleGun gun(1);
    gun.SetParticleDefinition(table->FindParticle("gamma"));
    gun.SetParticleEnergy(6 * MeV);
    gun.SetParticleMomentumDirection(G4ThreeVector(0, 0, 1));
    gun.SetParticlePosition(G4ThreeVector(0, 0, -15 * cm));
    gun.SetBeamCrossSectionRectangular(8 * cm, 8 * cm);

    G4double lo_x = 1e30, hi_x = -1e30;
    for (int i = 0; i < 5000; ++i) {
      const G4ThreeVector v = VertexOf(gun);
      if (v.x() < lo_x) { lo_x = v.x(); }
      if (v.x() > hi_x) { hi_x = v.x(); }
      if (std::abs(v.z() + 15 * cm) > 1e-9 * cm) {
        expect(false, "a beam cross-section moved the primary along the beam axis");
        break;
      }
    }
    // Inside the half-width, and filling most of it - a distribution that collapsed to a point
    // would pass a bound check alone.
    expect(hi_x <= 8 * cm && lo_x >= -8 * cm, "beam spot stays inside its half-width");
    expect(hi_x > 7 * cm && lo_x < -7 * cm, "beam spot fills its half-width");
  }

  // ---- an angular spread, then a direction: the direction wins, exactly.
  {
    G4ParticleGun gun(1);
    gun.SetParticleDefinition(table->FindParticle("gamma"));
    gun.SetParticleEnergy(6 * MeV);
    gun.SetParticlePosition(G4ThreeVector(0, 0, 0));
    gun.SetAngularSpread(30 * deg);
    gun.SetParticleMomentumDirection(G4ThreeVector(0, 0, 1));

    bool all_exact = true;
    for (int i = 0; i < 500; ++i) {
      const G4ThreeVector d = DirectionOf(gun);
      if (std::abs(d.z() - 1.0) > 1e-9) {
        all_exact = false;
        std::printf("  direction %d is (%.6f, %.6f, %.6f), expected (0, 0, 1)\n", i, d.x(),
                    d.y(), d.z());
        break;
      }
    }
    expect(all_exact, "SetParticleMomentumDirection after an angular spread is unidirectional");
  }

  if (failures == 0) { std::printf("test_gun_position: OK\n"); }
  return (failures == 0) ? 0 : 1;
}
