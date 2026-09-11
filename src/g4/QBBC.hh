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
        "      DECAY is active (G4Decay: in flight for pi+- pi0 K+- mu+- and the neutron, at\n"
        "      rest for pi+ K+ mu+ - and, in stage 1, for the negatives too, because stage 1\n"
        "      is compared against a Geant4 run with the three at-rest captures inactivated).\n"
        "      The HADRONIC component is still not implemented: no elastic scattering, no\n"
        "      inelastic reaction, no capture at rest. A neutron has no cross section at all\n"
        "      and streams to the world boundary or dies on the 10 us tracking cut, which is\n"
        "      what a Geant4 neutron does with NeutronGeneralProc inactivated. Neutrinos are\n"
        "      counted as carrying their energy out of the event. Any other primary is refused\n"
        "      at /run/beamOn rather than transported as something it is not. What a hadronic\n"
        "      process the transport reaches and cannot apply costs is counted BY NAME and\n"
        "      printed at the end of the run, with the energy it took with it. See\n"
        "      docs/HADRONIC_PLAN.md and docs/PORTED.md 2.1.5.\n");
  }
};
