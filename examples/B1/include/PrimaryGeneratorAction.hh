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

 private:
  G4ParticleGun* fParticleGun = nullptr;
};

}  // namespace B1

#endif
