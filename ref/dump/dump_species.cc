// The PDG table Geant4 11.1.1 actually builds, for every species this port carries or refuses.
//
// `hadron_tables.csv` already carries a mass and a charge column for the eleven charged
// species G4EmCalculator can be asked about, and tests/test_hadron.cu checks core/particle.cuh
// against them. It cannot carry the four this package adds, for two different reasons:
//
//   * neutron and pi0 have no ionisation process at all, so there is no `GetDEDX` call to hang
//     them off - a neutral particle is absent from an EM table dump by construction.
//   * spin, lepton number and the magnetic moment are not EM-table quantities. The port's
//     ParticleDef carries all three (spin gates the Bethe-Bloch delta-ray form factor, lepton
//     number gates `tlimit`, the moment gates the rejection inside the delta sampler), and
//     until now they were transcribed from the constructor arguments with nothing to check
//     them against.
//
// So this dumps G4ParticleDefinition itself. Every column is a getter on the definition rather
// than a number computed here, and the two derived columns say which expression derived them.
//
// It also dumps the processes the real QBBC attaches to the neutron. That is not a table but a
// FACT about the physics list, and it is the fact P1's step_neutral and P8's wiring both turn
// on: with `EnableNeutronGeneralProcess = 1` the elastic, inelastic and capture processes are
// sub-processes of one G4NeutronGeneralProcess, so a neutron has ONE discrete process and
// `/process/inactivate hadElastic` does not name anything it has. See
// docs/RISK.md and the header of step_neutral.
#include "dump_registry.hh"

#include <cfloat>
#include <cmath>
#include <cstdio>

#include "G4AntiNeutrinoE.hh"
#include "G4AntiNeutrinoMu.hh"
#include "G4AntiNeutrinoTau.hh"
#include "G4AntiProton.hh"
#include "G4Alpha.hh"
#include "G4Deuteron.hh"
#include "G4Electron.hh"
#include "G4Gamma.hh"
#include "G4GenericIon.hh"
#include "G4He3.hh"
#include "G4KaonMinus.hh"
#include "G4KaonPlus.hh"
#include "G4KaonZeroLong.hh"
#include "G4KaonZeroShort.hh"
#include "G4Lambda.hh"
#include "G4MuonMinus.hh"
#include "G4MuonPlus.hh"
#include "G4NistManager.hh"
#include "G4Neutron.hh"
#include "G4NeutrinoE.hh"
#include "G4NeutrinoMu.hh"
#include "G4NeutrinoTau.hh"
#include "G4PhysicalConstants.hh"
#include "G4PionMinus.hh"
#include "G4PionPlus.hh"
#include "G4PionZero.hh"
#include "G4Positron.hh"
#include "G4ProcessManager.hh"
#include "G4ProcessVector.hh"
#include "G4Proton.hh"
#include "G4SigmaMinus.hh"
#include "G4SigmaPlus.hh"
#include "G4SystemOfUnits.hh"
#include "G4Triton.hh"
#include "G4VEmFluctuationModel.hh"
#include "G4VEmModel.hh"
#include "G4VEnergyLossProcess.hh"
#include "G4VMultipleScattering.hh"
#include "G4VProcess.hh"
#include "G4XiMinus.hh"

static void dump_species(const DumpContext&) {
  const G4ParticleDefinition* parts[] = {
      // Transported today.
      G4Gamma::Gamma(), G4Electron::Electron(), G4Positron::Positron(),
      G4Proton::Proton(), G4Alpha::Alpha(),
      // Validated physics, transport added by this package.
      G4MuonMinus::MuonMinus(), G4MuonPlus::MuonPlus(),
      G4PionPlus::PionPlus(), G4PionMinus::PionMinus(),
      G4KaonPlus::KaonPlus(), G4KaonMinus::KaonMinus(),
      G4AntiProton::AntiProton(), G4Deuteron::Deuteron(), G4Triton::Triton(),
      G4He3::He3(), G4GenericIon::GenericIon(),
      // New neutral species.
      G4Neutron::Neutron(), G4PionZero::PionZero(),
      // Counted, never stepped.
      G4NeutrinoE::NeutrinoE(), G4AntiNeutrinoE::AntiNeutrinoE(),
      G4NeutrinoMu::NeutrinoMu(), G4AntiNeutrinoMu::AntiNeutrinoMu(),
      G4NeutrinoTau::NeutrinoTau(), G4AntiNeutrinoTau::AntiNeutrinoTau(),
      // Refused by name. Their rows are here so that the refusal can quote a real mass and a
      // real PDG code rather than a placeholder, and so that whoever implements them later
      // starts from a checked table.
      G4KaonZeroLong::KaonZeroLong(), G4KaonZeroShort::KaonZeroShort(),
      G4Lambda::Lambda(), G4SigmaPlus::SigmaPlus(), G4SigmaMinus::SigmaMinus(),
      G4XiMinus::XiMinus(),
  };

  FILE* f = std::fopen("species_tables.csv", "w");
  std::fprintf(f, "particle,pdg,mass_MeV,charge,spin,lepton_number,baryon_number,"
                  "mag_moment2,tlimit_MeV,lifetime_ns,stable,type,subtype\n");
  // G4BetheBlochModel::SetupParameters, verbatim:
  //     static const G4double aMag = 1./(0.5*eplus*CLHEP::hbar_Planck*CLHEP::c_squared);
  //     G4double magmom = particle->GetPDGMagneticMoment()*mass*aMag;
  //     magMoment2 = magmom*magmom - 1.0;
  const double aMag = 1.0 / (0.5 * eplus * CLHEP::hbar_Planck * CLHEP::c_squared);
  for (const G4ParticleDefinition* p : parts) {
    const double mass = p->GetPDGMass();
    const double magmom = p->GetPDGMagneticMoment() * mass * aMag;
    const double spin = p->GetPDGSpin();
    const double q = p->GetPDGCharge() / eplus;
    // G4BetheBlochModel::SetupParameters, the `GetLeptonNumber() == 0` branch. Dumped rather
    // than transcribed twice, because the branch depends on spin, mass AND charge and the
    // port re-derives it in em/hadron_ionisation.cuh: hadron_tlimit.
    double tlimit = DBL_MAX;
    if (p->GetLeptonNumber() == 0) {
      double x = 0.8426 * CLHEP::GeV;
      if (spin == 0.0 && mass < CLHEP::GeV) {
        x = 0.736 * CLHEP::GeV;
      } else if (mass > CLHEP::GeV) {
        const int iz = G4lrint(std::abs(q));
        if (iz > 1) { x /= G4NistManager::Instance()->GetA27(iz); }
      }
      tlimit = 2.0 / (2.0 * CLHEP::electron_mass_c2 / (x * x));
    }
    std::fprintf(f, "%s,%d,%.17g,%.17g,%.17g,%d,%d,%.17g,%.17g,%.17g,%d,%s,%s\n",
                 p->GetParticleName().c_str(), p->GetPDGEncoding(), mass / MeV, q, spin,
                 p->GetLeptonNumber(), p->GetBaryonNumber(), magmom * magmom - 1.0,
                 tlimit / MeV, p->GetPDGLifeTime() / ns, p->GetPDGStable() ? 1 : 0,
                 p->GetParticleType().c_str(), p->GetParticleSubType().c_str());
  }
  std::fclose(f);

  // ---------------- which processes and models QBBC attaches to each species
  //
  // The port answers four questions per species with a hand-written table each, and every one
  // of them splits the charged hadrons somewhere no property of the particle would:
  //
  //     uses_ion_ionisation      G4ionIonisation or G4hIonisation?     -> low-E model,
  //                                                                       fluctuation model
  //     uses_nuclear_stopping    is G4NuclearStopping registered?      -> a loss along the step
  //     uses_wentzel_msc         WentzelVI or Urban?                   -> the step limit
  //     hadron_base_particle     whose dE/dx table is scaled?          -> the whole range table
  //
  // Those were read out of G4EmStandardPhysics.cc and G4EmBuilder.cc by eye, which is how
  // docs/PORTED.md 4.2 came to record the alpha's fluctuation model being wrong for as long as
  // it was: the code that decides is four functions deep and the answer is not a rule. This
  // dump asks the constructed physics list instead. Every column is a public accessor on the
  // process the list actually registered, so a table in the port that disagrees is a table
  // that is wrong rather than a reading somebody has to re-do.
  //
  // `models` is semicolon-separated because the CSV is comma-separated and a model list is
  // plural: G4MuIonisation carries Bragg (or ICRU73QO) below 200 keV and MuBetheBloch above.
  {
    FILE* g = std::fopen("species_processes.csv", "w");
    std::fprintf(g, "particle,process,subtype,models,base_particle,fluct_model\n");
    for (const G4ParticleDefinition* p : parts) {
      const G4ProcessManager* pm = p->GetProcessManager();
      if (pm == nullptr) {
        // A species QBBC constructs but registers no process for. Recorded as a row rather
        // than omitted, so that "no processes" is distinguishable from "not in the dump".
        std::fprintf(g, "%s,,,,,\n", p->GetParticleName().c_str());
        continue;
      }
      G4ProcessVector* pv = pm->GetProcessList();
      if (pv->size() == 0) { std::fprintf(g, "%s,,,,,\n", p->GetParticleName().c_str()); }
      for (std::size_t i = 0; i < pv->size(); ++i) {
        G4VProcess* pr = (*pv)[i];
        G4String models;
        G4String base;
        G4String fluct;
        // The two families that carry models. Both accessors walk G4EmModelManager's own list,
        // which is the list the process will select from at run time - not the list somebody
        // passed to SetEmModel, which for an ion is empty and is exactly the trap: an empty
        // model list means the process's DEFAULT, and G4hMultipleScattering's default is Urban.
        if (auto* msc = dynamic_cast<G4VMultipleScattering*>(pr)) {
          for (G4int k = 0; k < msc->NumberOfModels(); ++k) {
            const G4VEmModel* m = msc->GetModelByIndex(k);
            if (m != nullptr) { models += (models.empty() ? "" : ";") + m->GetName(); }
          }
        } else if (auto* el = dynamic_cast<G4VEnergyLossProcess*>(pr)) {
          for (std::size_t k = 0; k < el->NumberOfModels(); ++k) {
            const G4VEmModel* m = el->GetModelByIndex(k);
            if (m != nullptr) { models += (models.empty() ? "" : ";") + m->GetName(); }
          }
          if (el->BaseParticle() != nullptr) { base = el->BaseParticle()->GetParticleName(); }
          if (el->FluctModel() != nullptr) { fluct = el->FluctModel()->GetName(); }
        }
        std::fprintf(g, "%s,%s,%d,%s,%s,%s\n", p->GetParticleName().c_str(),
                     pr->GetProcessName().c_str(), pr->GetProcessSubType(), models.c_str(),
                     base.c_str(), fluct.c_str());
      }
    }
    std::fclose(g);
  }

  // ---------------- what QBBC attaches to the neutron
  //
  // One row per process on the neutron's process manager, with the sub-type, so that a reader
  // can see for themselves that there is a `NeutronGeneralProc` of sub-type fNeutronGeneral
  // (116, from G4HadronicProcessType.hh - not 161, which is what reading the enum too fast
  // gives) and NOT a separate hadElastic / neutronInelastic / nCapture / nKiller. The port's
  // step_neutral has one discrete-interaction slot for exactly that reason.
  {
    FILE* g = std::fopen("neutron_processes.csv", "w");
    std::fprintf(g, "process,type,subtype\n");
    const G4ProcessManager* pm = G4Neutron::Neutron()->GetProcessManager();
    if (pm != nullptr) {
      G4ProcessVector* pv = pm->GetProcessList();
      for (std::size_t i = 0; i < pv->size(); ++i) {
        const G4VProcess* pr = (*pv)[i];
        std::fprintf(g, "%s,%d,%d\n", pr->GetProcessName().c_str(),
                     static_cast<int>(pr->GetProcessType()), pr->GetProcessSubType());
      }
    }
    std::fclose(g);
  }
}

G4GPU_REGISTER_DUMP("species",
                    "species_tables.csv species_processes.csv neutron_processes.csv",
                    dump_species);
