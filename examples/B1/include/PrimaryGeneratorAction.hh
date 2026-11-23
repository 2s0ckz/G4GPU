/// \file PrimaryGeneratorAction.hh
/// \brief Definition of the B1::PrimaryGeneratorAction class

#ifndef B1PrimaryGeneratorAction_h
#define B1PrimaryGeneratorAction_h 1

#include "G4ParticleGun.hh"
#include "G4VUserPrimaryGeneratorAction.hh"
#include "globals.hh"

class G4Event;

namespace B1 {

/// The primary generator action class with a particle gun.
///
/// The default kinematic is a 6 MeV gamma, randomly distributed in front of the phantom
/// across 80% of the transverse (X,Y) phantom size.
class PrimaryGeneratorAction : public G4VUserPrimaryGeneratorAction {
 public:
  PrimaryGeneratorAction();
  ~PrimaryGeneratorAction() override;

  void GeneratePrimaries(G4Event* anEvent) override;

  // method to access particle gun
  const G4ParticleGun* GetParticleGun() const { return fParticleGun; }

  /// The second command this example adds to Geant4's B1 (P27's tier 4, docs/RISK.md V336):
  ///
  ///   /B1/gun/z <value> <unit>     the plane the gun fires from, in world coordinates
  ///
  /// B1 fires from the envelope's -z face (z = -15 cm), 19 cm of water and tissue from the scoring
  /// trapezoid, which no primary below ~200 MeV crosses. With the command the x-y randomisation
  /// (80% of the envelope's width, two G4UniformRand draws in B1's order) and the +z direction
  /// are B1's and only z0 is the one given. Unset, z0 is B1's -0.5 x the envelope's length, by the
  /// same line as before. The same command on the Geant4 side is ref/b1mat's. Returns -1 for a
  /// line that is not this command's, 0 applied, 1 refused (no value, or an unknown unit).
  static G4int ApplyCommand(const G4String& line);

 private:
  G4ParticleGun* fParticleGun = nullptr;
  static G4bool fHasGunZ;
  static G4double fGunZ;
};

}  // namespace B1

#endif
