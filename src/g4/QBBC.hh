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
        "      hadELASTIC is active for every charged hadron Geant4 gives it to, with\n"
        "      CoulombScat beside it, and the recoil NUCLEUS it makes is transported - as\n"
        "      itself for the five light nuclei and as a GenericIon carrying its own (Z, A)\n"
        "      for everything heavier.\n"
        "      What is still MISSING is the inelastic reactions (P9-P11), the capture of a\n"
        "      stopped negative hadron (P12), the gamma-/electro-/muon-nuclear processes\n"
        "      (P13), the ion's own ionElastic, and the neutron's general process - so a\n"
        "      neutron has no cross section at all and streams to the world boundary or dies\n"
        "      on the 10 us tracking cut, which is what a Geant4 neutron does with\n"
        "      NeutronGeneralProc inactivated. Neutrinos are counted as carrying their energy\n"
        "      out of the event. Any other primary is refused at /run/beamOn rather than\n"
        "      transported as something it is not. What a hadronic process the transport\n"
        "      reaches and cannot apply costs is counted BY NAME and printed at the end of the\n"
        "      run, with the energy it took with it. See docs/HADRONIC_PLAN.md and\n"
        "      docs/PORTED.md 2.1.5 to 2.1.7.\n");
  }
};
