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
//
// The message below used to say "gamma, e- and e+ are transported", which was true when the
// species set was three and stayed on screen after it was five. It is built from the species
// list itself now, so it cannot say something different from what the transport does.
#pragma once
#include <cstdio>
#include "core/track_buffer.cuh"
#include "g4/G4VModularPhysicsList.hh"

class QBBC : public G4VModularPhysicsList {
 public:
  QBBC() {
    RegisterPhysics(new G4EmStandardPhysics(1));
    std::printf(
        "QBBC: EM standard physics is active (photoelectric, Compton, Rayleigh, pair,\n"
        "      bremsstrahlung, annihilation, multiple scattering, continuous ionisation).\n"
        "      Transported:");
    for (int sp = 0; sp < g4gpu::kNumTrackSpecies; ++sp) {
      std::printf("%s %s", (sp % 8 == 0 && sp > 0) ? "\n                  " : "",
                  g4gpu::particle_name(g4gpu::species_of_index(sp)));
    }
    std::printf(
        "\n"
        "      The HADRONIC and DECAY components of QBBC are still not implemented. The\n"
        "      charged hadrons above are transported by their electromagnetic physics alone:\n"
        "      no elastic scattering, no inelastic reaction, no decay, no capture at rest.\n"
        "      A neutron has no cross section at all and streams to the world boundary or\n"
        "      dies on the 10 us tracking cut, which is what a Geant4 neutron does with\n"
        "      NeutronGeneralProc inactivated. Neutrinos are counted as carrying their energy\n"
        "      out of the event. Any other primary is refused at /run/beamOn rather than\n"
        "      transported as something it is not. See docs/HADRONIC_PLAN.md.\n");
  }
};
