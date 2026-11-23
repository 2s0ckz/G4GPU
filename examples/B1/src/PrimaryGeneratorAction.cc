/// \file PrimaryGeneratorAction.cc
/// \brief Implementation of the B1::PrimaryGeneratorAction class

#include "PrimaryGeneratorAction.hh"

#include "G4Box.hh"
#include "G4LogicalVolume.hh"
#include "G4ParticleTable.hh"
#include "G4RunManager.hh"
#include "G4SystemOfUnits.hh"
#include "Randomize.hh"
#include "G4ios.hh"

#include <cctype>
#include <cstdlib>
#include <string>

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
  const G4double z0 = fHasGunZ ? fGunZ : -0.5 * envSizeZ;  // /B1/gun/z (P27 tier 4) or B1's

  fParticleGun->SetParticlePosition(G4ThreeVector(x0, y0, z0));
  fParticleGun->GeneratePrimaryVertex(anEvent);
}

G4bool PrimaryGeneratorAction::fHasGunZ = false;
G4double PrimaryGeneratorAction::fGunZ = 0.;

/// /B1/gun/z <value> <unit> (P27, V336): the gun's z plane. The units are G4UIcmdWithADoubleAndUnit's
/// lengths that a B1 geometry needs; Geant4's messenger on the other side takes the same.
G4int PrimaryGeneratorAction::ApplyCommand(const G4String& line) {
  const std::string kCmd = "/B1/gun/z";
  if (line.compare(0, kCmd.size(), kCmd) != 0
      || (line.size() > kCmd.size()
          && std::isspace(static_cast<unsigned char>(line[kCmd.size()])) == 0)) {
    return -1;
  }
  std::string tok[2];
  int n = 0;
  for (std::size_t i = kCmd.size(); i < line.size() && n < 2;) {
    while (i < line.size() && std::isspace(static_cast<unsigned char>(line[i])) != 0) { ++i; }
    const std::size_t s = i;
    while (i < line.size() && std::isspace(static_cast<unsigned char>(line[i])) == 0) { ++i; }
    if (i > s) { tok[n++] = line.substr(s, i - s); }
  }
  if (n == 0) {
    G4cout << "/B1/gun/z needs a value and a unit, e.g. /B1/gun/z 39.999 mm" << G4endl;
    return 1;
  }
  const std::string unit = (n > 1) ? tok[1] : std::string("mm");  // the command's default unit
  G4double f = 0.;
  if (unit == "um") { f = um; }
  else if (unit == "mm") { f = mm; }
  else if (unit == "cm") { f = cm; }
  else if (unit == "m") { f = m; }
  else {
    G4cout << "/B1/gun/z: unit \"" << unit << "\" refused - um, mm, cm or m" << G4endl;
    return 1;
  }
  fGunZ = std::atof(tok[0].c_str()) * f;
  fHasGunZ = true;
  G4cout << "/B1/gun/z: the gun fires from z = " << fGunZ / mm << " mm" << G4endl;
  return 0;
}

}  // namespace B1
