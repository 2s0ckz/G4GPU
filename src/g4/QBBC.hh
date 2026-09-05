// QBBC: the reference physics list example B1 uses.
//
// In Geant4, QBBC is G4EmStandardPhysics plus a hadronic set - quark-gluon string above
// 12 GeV, binary cascade below, Bertini for kaons and hyperons - plus decay, ion and
// stopping physics. Example B1's dose figure comes from the EM component: 6 MeV gammas in
// water, tissue and bone never reach a hadronic process.
//
// Here the EM component is implemented and validated against Geant4 11.1.1 to 0.09 sigma
// (docs/RESULT.md). The hadronic component is not, and this constructor says so once, at
// construction, because B1's own run1.mac and run2.mac fire 210 MeV protons and a run that
// quietly transported them as something else - or as nothing - would produce a plausible
// number for a physics that was never run. G4RunManager::BeamOn refuses a primary species
// the transport does not carry, for the same reason.
#pragma once
#include <cstdio>
#include "g4/G4VModularPhysicsList.hh"

class QBBC : public G4VModularPhysicsList {
 public:
  QBBC() {
    RegisterPhysics(new G4EmStandardPhysics(1));
    std::printf(
        "QBBC: EM standard physics is active (photoelectric, Compton, Rayleigh, pair,\n"
        "      bremsstrahlung, annihilation, multiple scattering, continuous ionisation).\n"
        "      The hadronic, decay and ion components of QBBC are NOT implemented: gamma,\n"
        "      e- and e+ are transported, and any other primary is refused at /run/beamOn\n"
        "      rather than transported as something it is not. See docs/PHYSICS_PLAN.md.\n");
  }
};
