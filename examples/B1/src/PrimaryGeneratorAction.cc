/// \file PrimaryGeneratorAction.cc
/// \brief Implementation of the B1::PrimaryGeneratorAction class

#include "PrimaryGeneratorAction.hh"

#include "G4Box.hh"
#include "G4LogicalVolume.hh"
#include "G4ParticleTable.hh"
#include "G4RunManager.hh"
#include "G4SystemOfUnits.hh"
#include "Randomize.hh"

namespace B1 {

PrimaryGeneratorAction::PrimaryGeneratorAction() {
  fParticleGun = new G4ParticleGun(1);

  auto* particleTable = G4ParticleTable::GetParticleTable();
  fParticleGun->SetParticleDefinition(particleTable->FindParticle("gamma"));
  fParticleGun->SetParticleMomentumDirection(G4ThreeVector(0., 0., 1.));
  fParticleGun->SetParticleEnergy(6. * MeV);

  // The run manager needs the gun so that /gun/... macro commands reach it.
  if (G4RunManager::Instance() != nullptr) { G4RunManager::Instance()->SetGun(fParticleGun); }
}

PrimaryGeneratorAction::~PrimaryGeneratorAction() { delete fParticleGun; }

void PrimaryGeneratorAction::GeneratePrimaries(G4Event* anEvent) {
  // Character for character what Geant4's B1 does, and for the first time it can be: this
  // method is called once per event, so the two random draws below are per event, as upstream
  // intends. It used to be called once per run with the device sampling a beam cross-section
  // on the gun, and the comment here used to explain that substitution. There is nothing left
  // to substitute.
  G4double envSizeXY = 0;
  G4double envSizeZ = 0;
  for (G4LogicalVolume* lv : G4LogicalVolume::Registry()) {
    if (lv->GetName() != "Envelope") { continue; }
    if (auto* box = dynamic_cast<G4Box*>(lv->GetSolid())) {
      envSizeXY = box->GetXHalfLength() * 2.;
      envSizeZ = box->GetZHalfLength() * 2.;
    }
  }
  if (envSizeXY == 0. || envSizeZ == 0.) {
    envSizeXY = 20 * cm;
    envSizeZ = 30 * cm;
  }

  const G4double size = 0.8;
  const G4double x0 = size * envSizeXY * (G4UniformRand() - 0.5);
  const G4double y0 = size * envSizeXY * (G4UniformRand() - 0.5);
  const G4double z0 = -0.5 * envSizeZ;

  fParticleGun->SetParticlePosition(G4ThreeVector(x0, y0, z0));
  fParticleGun->GeneratePrimaryVertex(anEvent);
}

}  // namespace B1
