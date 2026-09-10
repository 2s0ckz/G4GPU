// G4HadronicParameters as QBBC sees them: the energies at which one model hands over to the
// next, the cross-section factors, and the neutron tracking cut.
//
// The first dump written through dump_registry.hh, and it is here for two reasons. It proves
// the registration path end to end - a file nobody named in g4dump.cc or CMakeLists.txt gets
// compiled, linked, run, and reported. And the numbers are the ones every Phase-2 wiring
// decision turns on: BIC to 1.5 GeV, Bertini from 1 GeV, FTF from 3 GeV, the handover complete
// at 6 GeV. docs/HADRONIC_PLAN.md quotes them; this is where a port checks that the quoted
// value is the installed value.
#include "dump_registry.hh"

#include <cstdio>

#include "G4HadronicParameters.hh"
#include "G4SystemOfUnits.hh"

static void dump_hadronic_params(const DumpContext&) {
  const G4HadronicParameters* p = G4HadronicParameters::Instance();
  FILE* f = std::fopen("hadronic_params.csv", "w");
  std::fprintf(f, "name,value,unit\n");
  std::fprintf(f, "MaxEnergy,%.17g,MeV\n", p->GetMaxEnergy() / MeV);
  std::fprintf(f, "MinEnergyTransitionFTF_Cascade,%.17g,MeV\n",
               p->GetMinEnergyTransitionFTF_Cascade() / MeV);
  std::fprintf(f, "MaxEnergyTransitionFTF_Cascade,%.17g,MeV\n",
               p->GetMaxEnergyTransitionFTF_Cascade() / MeV);
  std::fprintf(f, "MinEnergyTransitionQGS_FTF,%.17g,MeV\n",
               p->GetMinEnergyTransitionQGS_FTF() / MeV);
  std::fprintf(f, "MaxEnergyTransitionQGS_FTF,%.17g,MeV\n",
               p->GetMaxEnergyTransitionQGS_FTF() / MeV);
  std::fprintf(f, "EnergyThresholdForHeavyHadrons,%.17g,MeV\n",
               p->EnergyThresholdForHeavyHadrons() / MeV);
  std::fprintf(f, "ApplyFactorXS,%d,\n", p->ApplyFactorXS() ? 1 : 0);
  std::fprintf(f, "XSFactorNucleonInelastic,%.17g,\n", p->XSFactorNucleonInelastic());
  std::fprintf(f, "XSFactorNucleonElastic,%.17g,\n", p->XSFactorNucleonElastic());
  std::fprintf(f, "XSFactorPionInelastic,%.17g,\n", p->XSFactorPionInelastic());
  std::fprintf(f, "XSFactorPionElastic,%.17g,\n", p->XSFactorPionElastic());
  std::fprintf(f, "XSFactorHadronInelastic,%.17g,\n", p->XSFactorHadronInelastic());
  std::fprintf(f, "XSFactorHadronElastic,%.17g,\n", p->XSFactorHadronElastic());
  std::fprintf(f, "EnableBCParticles,%d,\n", p->EnableBCParticles() ? 1 : 0);
  std::fprintf(f, "EnableHyperNuclei,%d,\n", p->EnableHyperNuclei() ? 1 : 0);
  std::fprintf(f, "EnableNeutronGeneralProcess,%d,\n",
               p->EnableNeutronGeneralProcess() ? 1 : 0);
  std::fclose(f);
}

G4GPU_REGISTER_DUMP("hadronic_params", "hadronic_params.csv", dump_hadronic_params);
