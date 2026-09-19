// G4EmExtraPhysics as QBBC configures it: the oracle for package P13.
//
// The first file here is the CONFIGURATION, and it is first because every other number in this
// package depends on it. docs/HADRONIC_PLAN.md section 2 says the gamma gets
// "`G4LowEGammaNuclearModel` (<200 MeV, via PreCompound), Bertini above" and calls the
// high-energy generator FTFP. Two of those three statements are wrong, and only asking the
// running physics list could have said so - which is what `emextra_config.csv` and
// `emextra_windows.csv` do:
//
//   * the low-energy limit is 200 MeV and the Bertini instance starts at 199 MeV, not 200 -
//     `cascade->SetMinEnergy(fGNLowEnergyLimit - CLHEP::MeV)` - so there is a 1 MeV OVERLAP
//     between G4LowEGammaNuclearModel and Bertini and a photon in it is assigned at random;
//   * the photon's high-energy generator is `G4QGSModel<G4GammaParticipants>` with
//     `G4QGSMFragmentation`, NOT FTF. QGS is a different string model from the one P11 ported.
//     The electro- and muon-nuclear models build their OWN G4TheoFSGenerator and those two ARE
//     FTF (`G4FTFModel` + `G4LundStringFragmentation`), so the same photon at the same energy
//     goes to a different string model depending on which lepton made it.
//
// How the configuration is read. `G4HadronicProcessStore::FindProcess(particle, subType)` is
// public and its `p_map` is filled by every G4HadronicProcess that has been prepared for a
// particle, INCLUDING `photonNuclear`, which lives inside `G4GammaGeneralProcess` and is
// therefore absent from the gamma's process manager. So both are dumped: the process manager's
// list (which says what the transport actually sees) and the store's (which says what the
// hadronic configuration is). A test that only had the first would conclude the gamma has no
// photonuclear process at all.
//
// Files:
//   emextra_config.csv     one row per (particle, process): the process's name, subtype, and
//                          which of the two lists it came from.
//   emextra_windows.csv    one row per (process, model): the model's name and its
//                          [GetMinEnergy, GetMaxEnergy] window, in MeV, at 17 digits - P11's
//                          ftf_windows.csv pattern.
//   emextra_xs.csv         one row per (process, data set): the data set's name, so that
//                          "which cross section does QBBC give photonNuclear" is a measured
//                          fact and not a reading of `fUseGammaNuclearXS`.
//   emextra_flags.csv      the G4EmExtraPhysics switches that are decidable from a run:
//                          which processes exist tells you which flag was true, and the two
//                          that are not decidable that way (LEND, the neutrinos) are decided
//                          by the absence of their processes on the same particles.
//   emextra_params.csv     the G4HadronicParameters numbers the photon's windows are built
//                          from, so that a window can be checked against its source and not
//                          only against itself.
//
// A note on what is NOT dumped, because it cannot be: `G4EmExtraPhysics`' eleven bool members
// are private with no getters and the constructor takes only a verbosity. The flags are
// therefore inferred from the processes that exist, which is the stronger statement anyway -
// it is what the run does, not what a field says.
#include "dump_registry.hh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

#include "G4CascadeInterface.hh"
#include "G4CrossSectionDataSetRegistry.hh"
#include "G4CrossSectionDataStore.hh"
#include "G4DynamicParticle.hh"
#include "G4ElectroNuclearCrossSection.hh"
#include "G4ElectroVDNuclearModel.hh"
#include "G4HadFinalState.hh"
#include "G4HadProjectile.hh"
#include "G4HadSecondary.hh"
#include "G4LowEGammaNuclearModel.hh"
#include "G4MuonVDNuclearModel.hh"
#include "G4Nucleus.hh"
#include "G4Element.hh"
#include "G4ElementTable.hh"
#include "G4GammaNuclearXS.hh"
#include "G4Isotope.hh"
#include "G4KokoulinMuonNuclearXS.hh"
#include "G4Material.hh"
#include "G4NistManager.hh"
#include "G4PhotoNuclearCrossSection.hh"
#include "G4ThreeVector.hh"
#include "Randomize.hh"

#include "CLHEP/Random/RandomEngine.h"
#include "G4Electron.hh"
#include "G4Gamma.hh"
#include "G4HadronicInteraction.hh"
#include "G4HadronicParameters.hh"
#include "G4HadronicProcess.hh"
#include "G4HadronicProcessStore.hh"
#include "G4HadronicProcessType.hh"
#include "G4MuonMinus.hh"
#include "G4MuonPlus.hh"
#include "G4ParticleDefinition.hh"
#include "G4TheoFSGenerator.hh"
#include "G4VHighEnergyGenerator.hh"
#include "G4VIntraNuclearTransportModel.hh"
#include "G4Positron.hh"
#include "G4ProcessManager.hh"
#include "G4ProcessVector.hh"
#include "G4SystemOfUnits.hh"
#include "G4VCrossSectionDataSet.hh"
#include "G4VProcess.hh"

namespace {

// ---------------------------------------------------------------------------------------------
// The prescribed eight-value engine, as dump_elastic.cc and dump_precompound.cc define it.
//
// The same eight values and the same reasoning: spread over (0,1), away from both endpoints,
// shared with the port, so that every sampler becomes a deterministic function of (inputs,
// phase). The draw COUNT is dumped beside the answer because a transcription that gets the
// right number from the wrong number of deviates is wrong in a way only the count reveals -
// and G4ElectroNuclearCrossSection::GetEquivalentPhotonQ2 takes between one and three,
// depending on a rejection nothing else in its output shows.
// ---------------------------------------------------------------------------------------------

const double kSeq[8] = {0.05, 0.37, 0.63, 0.91, 0.12, 0.78, 0.29, 0.55};

class CycleEngine : public CLHEP::HepRandomEngine {
 public:
  void reset(int phase) { phase_ = phase; n_ = 0; }
  int draws() const { return n_; }
  double flat() override {
    const double v = kSeq[(n_ + phase_) % 8];
    ++n_;
    return v;
  }
  void flatArray(const int size, double* vect) override {
    for (int i = 0; i < size; ++i) { vect[i] = flat(); }
  }
  void setSeed(long, int) override {}
  void setSeeds(const long*, int) override {}
  void saveStatus(const char[]) const override {}
  void restoreStatus(const char[]) override {}
  void showStatus() const override {}
  std::string name() const override { return "CycleEngine"; }

 private:
  int phase_ = 0;
  int n_ = 0;
};

struct Species {
  const char* name;
  G4ParticleDefinition* (*get)();
};

G4ParticleDefinition* get_gamma() { return G4Gamma::Gamma(); }
G4ParticleDefinition* get_electron() { return G4Electron::Electron(); }
G4ParticleDefinition* get_positron() { return G4Positron::Positron(); }
G4ParticleDefinition* get_muminus() { return G4MuonMinus::MuonMinus(); }
G4ParticleDefinition* get_muplus() { return G4MuonPlus::MuonPlus(); }

const Species kSpecies[] = {
    {"gamma", get_gamma},     {"e-", get_electron},    {"e+", get_positron},
    {"mu-", get_muminus},     {"mu+", get_muplus},
};
const int kNSpecies = 5;

/// Every G4HadronicProcessType this package could see, by name, so the CSV carries the enum's
/// meaning and not just its integer.
const char* subtype_name(G4int t) {
  switch (t) {
    case fHadronElastic: return "fHadronElastic";
    case fNeutronGeneral: return "fNeutronGeneral";
    case fHadronInelastic: return "fHadronInelastic";
    case fCapture: return "fCapture";
    case fFission: return "fFission";
    case fHadronAtRest: return "fHadronAtRest";
    case fLeptonAtRest: return "fLeptonAtRest";
    case fChargeExchange: return "fChargeExchange";
    case fRadioactiveDecay: return "fRadioactiveDecay";
    case fEMDissociation: return "fEMDissociation";
    default: return "other";
  }
}

void dump_emextra(const DumpContext&) {
  auto* store = G4HadronicProcessStore::Instance();

  // -------------------------------------------------------------------------------------------
  // emextra_config.csv - what processes each of the five particles has, from both lists.
  // -------------------------------------------------------------------------------------------
  std::map<std::string, G4HadronicProcess*> found;  // "particle|process" -> process
  {
    FILE* f = std::fopen("emextra_config.csv", "w");
    std::fprintf(f, "particle,source,process,process_type,process_subtype,subtype_name\n");
    for (int i = 0; i < kNSpecies; ++i) {
      G4ParticleDefinition* p = kSpecies[i].get();
      // (a) the process manager: what the transport sees at the top level.
      auto* pm = p->GetProcessManager();
      if (pm != nullptr) {
        G4ProcessVector* pv = pm->GetProcessList();
        for (G4int j = 0; j < G4int(pv->size()); ++j) {
          G4VProcess* proc = (*pv)[j];
          std::fprintf(f, "%s,manager,%s,%d,%d,%s\n", kSpecies[i].name,
                       proc->GetProcessName().c_str(), G4int(proc->GetProcessType()),
                       proc->GetProcessSubType(), subtype_name(proc->GetProcessSubType()));
        }
      }
      // (b) the hadronic process store, which knows the sub-processes of G4GammaGeneralProcess.
      // FindProcess takes a subtype, so every subtype this package could produce is asked for.
      const G4int subtypes[] = {fHadronElastic, fHadronInelastic, fCapture, fChargeExchange};
      for (G4int st : subtypes) {
        G4HadronicProcess* hp = store->FindProcess(p, G4HadronicProcessType(st));
        if (hp == nullptr) { continue; }
        std::fprintf(f, "%s,store,%s,%d,%d,%s\n", kSpecies[i].name,
                     hp->GetProcessName().c_str(), G4int(hp->GetProcessType()),
                     hp->GetProcessSubType(), subtype_name(hp->GetProcessSubType()));
        found[std::string(kSpecies[i].name) + "|" + hp->GetProcessName()] = hp;
      }
    }
    std::fclose(f);
  }

  // -------------------------------------------------------------------------------------------
  // emextra_windows.csv - the G4EnergyRangeManager windows, per process per model.
  //
  // GetHadronicInteractionList() returns the models in REGISTRATION order, which is the order
  // G4EnergyRangeManager stores them and therefore the order the overlap logic walks. It is
  // dumped as `index` so the port cannot reorder them silently.
  // -------------------------------------------------------------------------------------------
  //
  // `high_energy_generator` is the column that matters most in this file. A
  // `G4TheoFSGenerator` is a wrapper: it is named "TheoFSGenerator" whatever string model is
  // inside it, and `GetHighEnergyGenerator()` is the only public way to ask which. The photon's
  // is `QGSModel`, not `FTF` - see this file's header.
  {
    FILE* f = std::fopen("emextra_windows.csv", "w");
    std::fprintf(f, "particle,process,index,model,min_MeV,max_MeV,high_energy_generator,"
                    "transport\n");
    for (auto& kv : found) {
      const std::string& key = kv.first;
      const std::string part = key.substr(0, key.find('|'));
      G4HadronicProcess* hp = kv.second;
      std::vector<G4HadronicInteraction*>& models = hp->GetHadronicInteractionList();
      for (std::size_t m = 0; m < models.size(); ++m) {
        const char* heg = "(none)";
        const char* trans = "(none)";
        auto* theo = dynamic_cast<G4TheoFSGenerator*>(models[m]);
        if (theo != nullptr) {
          if (theo->GetHighEnergyGenerator() != nullptr) {
            heg = theo->GetHighEnergyGenerator()->GetModelName().c_str();
          }
          if (theo->GetTransport() != nullptr) {
            trans = theo->GetTransport()->GetModelName().c_str();
          }
        }
        std::fprintf(f, "%s,%s,%d,%s,%.17g,%.17g,%s,%s\n", part.c_str(),
                     hp->GetProcessName().c_str(), G4int(m),
                     models[m]->GetModelName().c_str(), models[m]->GetMinEnergy() / MeV,
                     models[m]->GetMaxEnergy() / MeV, heg, trans);
      }
    }
    std::fclose(f);
  }

  // -------------------------------------------------------------------------------------------
  // emextra_xs.csv - which cross-section data sets each process carries, in store order.
  //
  // G4CrossSectionDataStore::ComputeCrossSection walks the list BACKWARDS (the last data set
  // added is asked first), so the index matters and is dumped.
  // -------------------------------------------------------------------------------------------
  {
    FILE* f = std::fopen("emextra_xs.csv", "w");
    std::fprintf(f, "particle,process,index,dataset,min_MeV,max_MeV\n");
    for (auto& kv : found) {
      const std::string& key = kv.first;
      const std::string part = key.substr(0, key.find('|'));
      G4HadronicProcess* hp = kv.second;
      const auto& sets = hp->GetCrossSectionDataStore()->GetDataSetList();
      for (std::size_t s = 0; s < sets.size(); ++s) {
        std::fprintf(f, "%s,%s,%d,%s,%.17g,%.17g\n", part.c_str(),
                     hp->GetProcessName().c_str(), G4int(s), sets[s]->GetName().c_str(),
                     sets[s]->GetMinKinEnergy() / MeV, sets[s]->GetMaxKinEnergy() / MeV);
      }
    }
    std::fclose(f);
  }

  // -------------------------------------------------------------------------------------------
  // emextra_params.csv - the G4HadronicParameters the photon's windows are built from, plus the
  // one environment variable the Bertini photon arm is gated by.
  //
  // `G4CASCADE_CHECK_PHOTONUCLEAR` is read with a raw std::getenv inside G4InuclCollider::collide
  // and gates `photonuclearOkay`. It is the only switch in the cascade tree that is not a
  // G4CascadeParameters member, so it cannot be read from that class and is recorded here.
  // -------------------------------------------------------------------------------------------
  {
    FILE* f = std::fopen("emextra_params.csv", "w");
    std::fprintf(f, "name,value\n");
    auto* par = G4HadronicParameters::Instance();
    std::fprintf(f, "MinEnergyTransitionFTF_Cascade_MeV,%.17g\n",
                 par->GetMinEnergyTransitionFTF_Cascade() / MeV);
    std::fprintf(f, "MaxEnergyTransitionFTF_Cascade_MeV,%.17g\n",
                 par->GetMaxEnergyTransitionFTF_Cascade() / MeV);
    std::fprintf(f, "MaxEnergy_MeV,%.17g\n", par->GetMaxEnergy() / MeV);
    // QGS is not FTF, and the photon's high-energy generator is QGS - so its transition
    // energies are dumped too even though no window here is built from them.
    std::fprintf(f, "MinEnergyTransitionQGS_FTF_MeV,%.17g\n",
                 par->GetMinEnergyTransitionQGS_FTF() / MeV);
    std::fprintf(f, "MaxEnergyTransitionQGS_FTF_MeV,%.17g\n",
                 par->GetMaxEnergyTransitionQGS_FTF() / MeV);
    std::fprintf(f, "G4CASCADE_CHECK_PHOTONUCLEAR_set,%d\n",
                 std::getenv("G4CASCADE_CHECK_PHOTONUCLEAR") != nullptr ? 1 : 0);
    std::fprintf(f, "G4LENDDATA_set,%d\n", std::getenv("G4LENDDATA") != nullptr ? 1 : 0);
    // WHICH GAMMA CROSS SECTION G4ElectroVDNuclearModel FOUND. Its constructor asks the
    // registry for "PhotoNuclearXS" first and only falls back to "GammaNuclearXS" when that is
    // absent. `G4GammaNuclearXS`'s own constructor asks for "PhotoNuclearXS" too and, finding
    // none, does `new G4PhotoNuclearCrossSection()` - whose base constructor REGISTERS it under
    // that name. `ConstructGammaElectroNuclear` builds the G4GammaNuclearXS before it builds the
    // model, so by the time the model looks the CHIPS object is there and the model takes it.
    // The electron and positron acceptance test therefore runs on the pure CHIPS photo-nuclear
    // cross section and NOT on the IAEA-data-based G4GammaNuclearXS the photon process uses.
    //
    // This is dumped BEFORE any dump in this file constructs a G4PhotoNuclearCrossSection of
    // its own, which would put one in the registry and make the answer trivially yes.
    auto* reg = G4CrossSectionDataSetRegistry::Instance();
    std::fprintf(f, "registry_has_PhotoNuclearXS,%d\n",
                 reg->GetCrossSectionDataSet("PhotoNuclearXS") != nullptr ? 1 : 0);
    std::fprintf(f, "registry_has_GammaNuclearXS,%d\n",
                 reg->GetCrossSectionDataSet("GammaNuclearXS") != nullptr ? 1 : 0);
    std::fprintf(f, "registry_has_ElectroNuclearXS,%d\n",
                 reg->GetCrossSectionDataSet("ElectroNuclearXS") != nullptr ? 1 : 0);
    std::fprintf(f, "registry_has_KokoulinMuonNuclearXS,%d\n",
                 reg->GetCrossSectionDataSet("KokoulinMuonNuclearXS") != nullptr ? 1 : 0);
    std::fclose(f);
  }
}

// =============================================================================================
// The three cross sections
// =============================================================================================

/// Twelve points per decade from 1 MeV to 10 TeV, plus every boundary the two CHIPS classes
/// switch branches at, exactly. The boundaries matter more than the grid does: THmin = 2,
/// Emin = 106 (where the GDR table hands over to the ln(E) table), Emax = 50000 (where the
/// ln(E) table hands over to the Regge form), the electro class's EMi = 2.0612 and
/// EMa = 50000, and G4GammaNuclearXS's rTransitionBound = 150. A grid that steps over a
/// boundary compares two functions that agree there and never compares the choice between
/// them.
std::vector<double> energy_grid() {
  std::vector<double> e;
  for (int i = 0; i <= 12 * 7; ++i) { e.push_back(std::pow(10.0, i / 12.0)); }
  const double edges[] = {2.0,      2.0612,   2.06121,  1.9999,   105.999,  106.0,
                          106.001,  129.9,    130.0,    130.1,    144.6821, 149.999,
                          150.0,    150.001,  199.0,    200.0,    49999.9,  50000.0,
                          50000.1,  10000.0,  100000.0};
  for (double x : edges) { e.push_back(x); }
  std::sort(e.begin(), e.end());
  return e;
}

void dump_emextra_xs(const DumpContext&) {
  const std::vector<double> grid = energy_grid();
  G4ParticleDefinition* gamma = G4Gamma::Gamma();
  G4ParticleDefinition* electron = G4Electron::Electron();

  // -------------------------------------------------------------------------------------------
  // emextra_photonuc.csv - G4PhotoNuclearCrossSection, element and the three light isotopes.
  //
  // The class is constructed here rather than taken from the registry: QBBC builds a
  // G4GammaNuclearXS which OWNS one (`ggXsection`), but that member is private and the registry
  // holds "GammaNuclearXS" under its own name. A second instance shares nothing mutable that
  // matters - `GDR`/`HEN` are per-object caches of a pure function - so the numbers are the
  // same and nothing in the run is perturbed.
  // -------------------------------------------------------------------------------------------
  {
    auto* chips = new G4PhotoNuclearCrossSection();
    FILE* f = std::fopen("emextra_photonuc.csv", "w");
    std::fprintf(f, "Z,A,kind,energy_MeV,xs_mb\n");
    G4DynamicParticle dp(gamma, G4ThreeVector(0, 0, 1), 1.0 * MeV);
    for (G4int Z = 1; Z <= 98; ++Z) {
      for (double e : grid) {
        dp.SetKineticEnergy(e * MeV);
        const G4double xs = chips->GetElementCrossSection(&dp, Z, nullptr);
        std::fprintf(f, "%d,0,element,%.17g,%.17g\n", Z, e, xs / millibarn);
      }
    }
    // The three isotope arms: deuteron, triton, He3. Every other (Z, A) falls through to the
    // element cross section with A discarded, and two of those are dumped to prove it.
    const G4int isos[][2] = {{1, 2}, {1, 3}, {2, 3}, {6, 12}, {82, 208}};
    for (const auto& za : isos) {
      for (double e : grid) {
        dp.SetKineticEnergy(e * MeV);
        const G4double xs =
            chips->GetIsoCrossSection(&dp, za[0], za[1], nullptr, nullptr, nullptr);
        std::fprintf(f, "%d,%d,isotope,%.17g,%.17g\n", za[0], za[1], e, xs / millibarn);
      }
    }
    std::fclose(f);
  }

  // -------------------------------------------------------------------------------------------
  // emextra_electronuc.csv - G4ElectroNuclearCrossSection, element cross section.
  // -------------------------------------------------------------------------------------------
  {
    auto* eln = new G4ElectroNuclearCrossSection();
    FILE* f = std::fopen("emextra_electronuc.csv", "w");
    std::fprintf(f, "Z,energy_MeV,xs_mb\n");
    G4DynamicParticle dp(electron, G4ThreeVector(0, 0, 1), 1.0 * MeV);
    for (G4int Z = 1; Z <= 98; ++Z) {
      for (double e : grid) {
        dp.SetKineticEnergy(e * MeV);
        const G4double xs = eln->GetElementCrossSection(&dp, Z, nullptr);
        std::fprintf(f, "%d,%.17g,%.17g\n", Z, e, xs / millibarn);
      }
    }
    std::fclose(f);
    delete eln;
  }

  // -------------------------------------------------------------------------------------------
  // emextra_eqphoton.csv - the three equivalent-photon functions, under the eight-value cycle.
  //
  // `GetEquivalentPhotonEnergy()` takes no arguments and reads five members
  // `GetElementCrossSection` left behind, so each row calls the cross section first - which is
  // what G4ElectroVDNuclearModel::ApplyYourself does, and the comment there says so. The draw
  // count is dumped beside the answer: the cumulative-walk branch takes ONE deviate and the
  // Newton branch takes one as well, but `GetEquivalentPhotonQ2` takes between one and three,
  // and a transcription that took the wrong number would still return a plausible energy.
  //
  // A FRESH INSTANCE PER ROW. The class caches `lastZ`, and when the requested Z equals it the
  // cross section is not recomputed and `lastSig` is reused - correct in a run, wrong in a
  // grid that walks energies within one Z and expects each row to be independent. Constructing
  // one object per row costs nothing here and removes the whole question.
  // -------------------------------------------------------------------------------------------
  {
    FILE* f = std::fopen("emextra_eqphoton.csv", "w");
    std::fprintf(f, "Z,lepton_MeV,phase,xs_mb,draws_nu,nu_MeV,draws_q2,Q2_MeV2,virtual_factor,"
                    "draws_total\n");
    const G4int zs[] = {1, 6, 8, 13, 26, 82};
    const double es[] = {10.0, 50.0, 200.0, 1000.0, 10000.0, 100000.0};
    CLHEP::HepRandomEngine* saved = G4Random::getTheEngine();
    CycleEngine cyc;
    G4Random::setTheEngine(&cyc);
    for (G4int Z : zs) {
      for (double lepE : es) {
        for (int phase = 0; phase < 8; ++phase) {
          auto* eln = new G4ElectroNuclearCrossSection();
          G4DynamicParticle dp(electron, G4ThreeVector(0, 0, 1), lepE * MeV);
          cyc.reset(phase);
          const G4double xs = eln->GetElementCrossSection(&dp, Z, nullptr);
          const int d0 = cyc.draws();
          const G4double nu = eln->GetEquivalentPhotonEnergy();
          const int d1 = cyc.draws();
          const G4double q2 = eln->GetEquivalentPhotonQ2(nu);
          const int d2 = cyc.draws();
          const G4double vf = eln->GetVirtualFactor(nu, q2);
          std::fprintf(f, "%d,%.17g,%d,%.17g,%d,%.17g,%d,%.17g,%.17g,%d\n", Z, lepE, phase,
                       xs / millibarn, d1 - d0, nu / MeV, d2 - d1, q2 / (MeV * MeV), vf,
                       cyc.draws());
          delete eln;
        }
      }
    }
    G4Random::setTheEngine(saved);
    std::fclose(f);
  }

  // -------------------------------------------------------------------------------------------
  // emextra_kokoulin.csv - G4KokoulinMuonNuclearXS.
  //
  //   dd    ComputeDDMicroscopicCrossSection(T, 0, A, eps) - the double-differential form, and
  //         the ONLY public entry to the parameterisation. It is also every term of the
  //         sampling table G4MuonVDNuclearModel builds, so the port's table is checked term by
  //         term against this and not against a re-derivation.
  //   elem  GetElementCrossSection at the table's own 61 log nodes AND between them. At a node
  //         the value IS ComputeMicroscopicCrossSection (which is private and unreachable), so
  //         the node rows check the eight-point Gauss-Legendre integration exactly; the rows
  //         between nodes check the G4PhysicsLogVector interpolation on top of it.
  //
  // `theCrossSection[Z]` is filled only for elements in the geometry - `BuildCrossSectionTable`
  // walks `G4Element::GetElementTable()` - and is a null pointer for every other Z, which
  // `GetElementCrossSection` dereferences without checking. So the element rows are exactly the
  // elements this dump program's materials contain, and the CSV says which; the port is general
  // over Z and the test records the coverage rather than assuming it.
  // -------------------------------------------------------------------------------------------
  {
    auto* mu = static_cast<G4KokoulinMuonNuclearXS*>(
        G4CrossSectionDataSetRegistry::Instance()->GetCrossSectionDataSet(
            G4KokoulinMuonNuclearXS::Default_Name()));
    FILE* f = std::fopen("emextra_kokoulin.csv", "w");
    std::fprintf(f, "kind,Z,A_amu,T_MeV,eps_MeV,value\n");
    auto* nist = G4NistManager::Instance();

    // The double-differential form on a grid that includes CutFixed = 200 MeV exactly (where
    // it returns zero) and `TotalEnergy - 0.5*m_p` (the upper limit, likewise).
    //
    // TWO ROWS PER POINT, WITH TWO DIFFERENT `A`, BECAUSE GEANT4'S TWO CALLERS DISAGREE.
    //
    //   kind = dd        `A` in plain amu, which is what
    //                    G4KokoulinMuonNuclearXS::BuildCrossSectionTable passes:
    //                    `A = nistManager->GetAtomicMassAmu(Z)`.
    //   kind = dd_gmole  `A * (g/mole)`, which is what G4MuonVDNuclearModel::MakeSamplingTable
    //                    passes: `AtomicWeight = adat[iz]*(g/mole)`. In Geant4's internal units
    //                    `g/mole` is 6.2415e21, not 1, so the muon model's sampling table is
    //                    built from a nucleus with A = 6.3e21 rather than 1.01.
    //
    // It changes nothing, and only measurement can say so: `A` enters the double-differential
    // cross section in exactly one place, `aeff = 0.22*A + 0.78*A^0.89`, which is independent
    // of the energy loss and therefore factors out of `MakeSamplingTable`'s integral and
    // cancels when the table is normalised by its own total. Both rows are dumped so that the
    // port can be compared against the argument each caller actually passes, and so that the
    // cancellation is a number in a test rather than an argument in a comment.
    const double as[] = {1.01, 9.01, 26.98, 63.55, 238.03};  // G4MuonVDNuclearModel::adat
    const double ts[] = {1.e3, 1.e4, 1.e5, 1.e6, 1.e9, 1.e11};
    for (double A : as) {
      for (double T : ts) {
        const double epmax = T + G4MuonMinus::MuonMinus()->GetPDGMass() / MeV - 0.5 * 938.272013;
        const double eps[] = {199.9, 200.0, 200.1, 250.0, 0.5 * epmax, 0.9 * epmax,
                              epmax - 1.0, epmax, epmax + 1.0};
        for (double ep : eps) {
          if (ep <= 0.0) { continue; }
          const G4double v1 = mu->ComputeDDMicroscopicCrossSection(T * MeV, 0.0, A, ep * MeV);
          std::fprintf(f, "dd,0,%.17g,%.17g,%.17g,%.17g\n", A, T, ep, v1 / millibarn);
          const G4double v2 =
              mu->ComputeDDMicroscopicCrossSection(T * MeV, 0.0, A * (g / mole), ep * MeV);
          std::fprintf(f, "dd_gmole,0,%.17g,%.17g,%.17g,%.17g\n", A, T, ep, v2 / millibarn);
        }
      }
    }
    // `g/mole` itself, so the port's own constant is checked rather than assumed.
    std::fprintf(f, "gmole,0,0,0,0,%.17g\n", g / mole);
    std::fflush(f);

    // The element cross section, at the 61 log nodes of G4PhysicsLogVector(1 GeV, 1 PeV, 60)
    // and at the geometric midpoints between them.
    // `theCrossSection[Z]` is a NULL POINTER for every Z the table was never built for, and
    // `GetElementCrossSection` dereferences it without a test. `BuildCrossSectionTable` is
    // public and idempotent (`if(Z < MAXZMUN && !theCrossSection[Z])`), so it is called here
    // rather than assumed to have run: whether the physics list built it depends on
    // `BuildPhysicsTable`'s `isMaster` latch, which returns early if ANY Z is already filled.
    mu->BuildCrossSectionTable();
    const G4ElementTable* et = G4Element::GetElementTable();
    G4DynamicParticle dmu(G4MuonMinus::MuonMinus(), G4ThreeVector(0, 0, 1), 1.0 * GeV);
    std::vector<G4int> zs;
    for (auto* el : *et) {
      const G4int Z = el->GetZasInt();
      if (Z > 0 && Z < 93) { zs.push_back(Z); }
    }
    std::sort(zs.begin(), zs.end());
    zs.erase(std::unique(zs.begin(), zs.end()), zs.end());
    for (G4int Z : zs) {
      const double A = nist->GetAtomicMassAmu(Z);
      for (int i = 0; i <= 120; ++i) {
        // 61 nodes: E_i = 1 GeV * (1e6)^(i/60). The half-steps land between them.
        const double e = 1.0e3 * std::pow(1.0e6, i / 120.0);
        dmu.SetKineticEnergy(e * MeV);
        const G4double v = mu->GetElementCrossSection(&dmu, Z, nullptr);
        std::fprintf(f, "elem,%d,%.17g,%.17g,0,%.17g\n", Z, A, e, v / millibarn);
      }
      std::fflush(f);
    }
    std::fclose(f);
  }

  // -------------------------------------------------------------------------------------------
  // emextra_gammanuc.csv - G4GammaNuclearXS itself, across the transition CHIPS closes.
  //
  // docs/PORTED.md 2.1.1 records this class as `P` with 1,128 element points and 760 isotope
  // points refused: everything above the IAEA files' top energy, everything for hydrogen, and
  // the straight line between the two, whose right-hand anchor `xs150[Z]` is CHIPS at 150 MeV.
  // `xs150` is a private static, so it is dumped as what it is - the CHIPS ELEMENT cross
  // section at exactly 150 MeV - and the port builds it the same way.
  // -------------------------------------------------------------------------------------------
  //
  // THE ELEMENTS ARE THE ONES THE RUN ALREADY HAS, and that is not fastidiousness. Asking
  // `G4GammaNuclearXS` for a Z the geometry does not contain takes `InitialiseOnFly`, which
  // reads a data file, allocates into a shared static `G4ElementData` and fills `xs150[Z]` -
  // i.e. it MUTATES the cross section QBBC's own photonNuclear process is using, from inside a
  // dump that is supposed to observe it. The element list is therefore taken from
  // `G4Element::GetElementTable()`, exactly as the Kokoulin block above does.
  //
  // The file is opened and flushed BEFORE any Geant4 call in this block, and flushed again at
  // every stage. A dump that dies takes its buffered stdout with it - which is how the first
  // version of this block was invisible - so the progress markers are rows in the CSV itself
  // (`stage,...`), which survive because they are flushed. The test skips them by `kind`.
  {
    FILE* f = std::fopen("emextra_gammanuc.csv", "w");
    if (f == nullptr) { return; }
    std::fprintf(f, "kind,Z,A,energy_MeV,xs_mb\n");
    std::fprintf(f, "stage,0,0,0,0\n");
    std::fflush(f);

    auto* gxs = static_cast<G4GammaNuclearXS*>(
        G4CrossSectionDataSetRegistry::Instance()->GetCrossSectionDataSet(
            G4GammaNuclearXS::Default_Name()));
    std::fprintf(f, "stage,1,0,0,%d\n", gxs != nullptr ? 1 : 0);
    std::fflush(f);
    if (gxs == nullptr) {
      std::fclose(f);
      return;
    }
    auto* chips = new G4PhotoNuclearCrossSection();
    std::fprintf(f, "stage,2,0,0,0\n");
    std::fflush(f);
    // xs150[Z] is CHIPS at 150 MeV - a private static of G4GammaNuclearXS, dumped here as the
    // number it is built from rather than as itself. Z <= 94 because that is the highest
    // `gamma/inel<Z>` file and therefore the highest Z the class can be initialised for.
    G4DynamicParticle dp(gamma, G4ThreeVector(0, 0, 1), 150.0 * MeV);
    for (G4int Z = 1; Z <= 94; ++Z) {
      std::fprintf(f, "xs150,%d,0,150,%.17g\n", Z,
                   chips->GetElementCrossSection(&dp, Z, nullptr) / millibarn);
    }
    std::fprintf(f, "stage,3,0,0,0\n");
    std::fflush(f);

    const G4ElementTable* et2 = G4Element::GetElementTable();
    std::vector<G4int> zs;
    for (auto* el : *et2) {
      const G4int Z = el->GetZasInt();
      if (Z > 0 && Z < 95) { zs.push_back(Z); }
    }
    std::sort(zs.begin(), zs.end());
    zs.erase(std::unique(zs.begin(), zs.end()), zs.end());
    for (G4int Z : zs) {
      std::fprintf(f, "stage,4,%d,0,0\n", Z);
      std::fflush(f);
      for (double e : grid) {
        std::fprintf(f, "element,%d,0,%.17g,%.17g\n", Z, e,
                     gxs->ElementCrossSection(e * MeV, Z) / millibarn);
      }
      std::fflush(f);
    }
    std::fprintf(f, "stage,5,0,0,0\n");
    std::fflush(f);
    // The isotope path, over every isotope the run's elements actually carry, so that no
    // (Z, A) is asked for that `Initialise` has not already built a component vector for.
    for (auto* el : *et2) {
      const G4int Z = el->GetZasInt();
      if (Z < 1 || Z > 94) { continue; }
      for (std::size_t i = 0; i < el->GetNumberOfIsotopes(); ++i) {
        const G4int A = el->GetIsotope(G4int(i))->GetN();
        std::fprintf(f, "stage,6,%d,%d,0\n", Z, A);
        std::fflush(f);
        for (double e : grid) {
          std::fprintf(f, "isotope,%d,%d,%.17g,%.17g\n", Z, A, e,
                       gxs->IsoCrossSection(e * MeV, Z, A) / millibarn);
        }
      }
      std::fflush(f);
    }
    std::fprintf(f, "stage,7,0,0,0\n");
    std::fclose(f);
  }
}

// =============================================================================================
// The statistical oracle: the four models' final states, as distributions
// =============================================================================================
//
// `emextra_apply.csv` and `emextra_apply_species.csv`, the same shape
// `ref/dump/dump_bertini.cc` writes for `bertini_apply.csv` and for the same reason: the models
// cannot be compared value for value. `G4LowEGammaNuclearModel` alone could be - it draws
// nothing of its own - but everything under it is P6's and P3's rejection samplers, and
// `G4CascadeInterface` has three nested retry loops in which a prescribed engine exhausts
// rather than samples (docs/RISK.md V132). So each case is N events under a fixed seed and what
// is dumped is moments and per-species yields.
//
// EACH MODEL IS DRIVEN DIRECTLY AND NOT THROUGH THE PROCESS. The model CHOICE - which of
// GammaNPreco, Bertini and the QGS generator a photon of a given energy gets - is a
// deterministic function plus one uniform, and `tests/test_emextra_config.cu` already compares
// it against the closed form over 200,000 draws per point. Mixing it into this campaign would
// put two thirds of the 5 GeV events into a model this port refuses and measure nothing. So the
// choice is validated exactly, each model is validated statistically, and the two are separate.
//
// `G4GPU_EMEXTRA_EVENTS` sets N; the default is 2,000, which is what `bertini_apply.csv` uses
// and what keeps a full oracle run inside a few minutes. The count is a COLUMN, so the port's
// test pools the two standard errors correctly rather than assuming they match.

struct SpeciesBucket {
  long long count = 0;      ///< how many of this species over the whole case
  long long events = 0;     ///< how many events contained at least one
  double sum_n = 0.0;       ///< sum over events of the per-event multiplicity
  double sum_n2 = 0.0;
  double sum_e = 0.0;       ///< sum over particles of the kinetic energy
  double sum_e2 = 0.0;
  double sum_cos = 0.0;     ///< sum over particles of cos(theta) about the beam
  /// THE HARD COMPONENT, SPLIT OFF, and it is not tidiness.
  ///
  /// A gamma-p collision inside the nucleus has an ELASTIC channel - `{gam, pro}` is the first
  /// two-body final state of `G4CascadeT1GamNChannel`, with a cross section of 0.1 to 2.7
  /// microbarn against a total of a hundred times that - so a few events in a thousand let the
  /// projectile photon out with nearly its full energy. The de-excitation gammas that make up
  /// the rest of the bucket are one or two MeV. A MEAN over both is not a statistic: seven
  /// events in two thousand carrying 2.9 GeV each move the gamma bucket's mean by a factor of
  /// four and its rms by a factor of forty, and two samples of two thousand cannot agree on it
  /// however right they both are. So the bucket is split at a tenth of the projectile energy
  /// and the two halves are compared as what they are: a spectrum, and a RATE.
  long long count_hard = 0;
  double sum_e_hard = 0.0;
  double sum_e2_hard = 0.0;
  double max_e = 0.0;
};

/// The species this package can produce, bucketed the way `bertini_apply_species.csv` buckets
/// them: by PDG code for a particle and by mass number for a nucleus, with everything above
/// A = 4 in one "heavier" bucket.
const char* species_bucket_name(int i) {
  static const char* kNames[13] = {"n",   "p",   "d",    "t",     "He3", "alpha", "heavier",
                                   "pi+", "pi-", "pi0",  "gamma", "e-",  "other"};
  return (i >= 0 && i < 13) ? kNames[i] : "other";
}

int species_bucket_of(const G4ParticleDefinition* d) {
  if (d == nullptr) { return 12; }
  const G4int pdg = d->GetPDGEncoding();
  switch (pdg) {
    case 2112: return 0;
    case 2212: return 1;
    case 211: return 7;
    case -211: return 8;
    case 111: return 9;
    case 22: return 10;
    case 11: return 11;
    default: break;
  }
  const G4int A = d->GetBaryonNumber();
  const G4int Z = G4lrint(d->GetPDGCharge() / CLHEP::eplus);
  if (A == 2 && Z == 1) { return 2; }
  if (A == 3 && Z == 1) { return 3; }
  if (A == 3 && Z == 2) { return 4; }
  if (A == 4 && Z == 2) { return 5; }
  if (A > 4) { return 6; }
  return 12;
}

/// One case's accumulators.
struct CaseStats {
  long long events = 0;
  long long empty = 0;          ///< no secondaries at all
  long long no_photon = 0;      ///< the lepton models' three gates and the muon's CutFixed
  double sum_n = 0.0, sum_n2 = 0.0;          ///< secondary multiplicity
  double sum_ekin = 0.0, sum_ekin2 = 0.0;    ///< summed secondary kinetic energy per event
  double sum_lep = 0.0, sum_lep2 = 0.0;      ///< the scattered lepton's kinetic energy
  double sum_coslep = 0.0;
  double sum_pz = 0.0;                       ///< summed secondary z-momentum per event
  SpeciesBucket sp[13];
};

void accumulate(CaseStats& st, const G4HadFinalState* hfs, G4bool lepton_survives,
                G4double lepton_kin, G4double hard_threshold) {
  ++st.events;
  const G4int n = (hfs != nullptr) ? G4int(hfs->GetNumberOfSecondaries()) : 0;
  if (n == 0) { ++st.empty; }
  st.sum_n += n;
  st.sum_n2 += double(n) * double(n);
  if (lepton_survives) {
    st.sum_lep += lepton_kin;
    st.sum_lep2 += lepton_kin * lepton_kin;
  }
  double ekin_sum = 0.0;
  double pz_sum = 0.0;
  G4int per_event[13] = {0};
  for (G4int i = 0; i < n; ++i) {
    const G4DynamicParticle* dp = hfs->GetSecondary(i)->GetParticle();
    const G4int b = species_bucket_of(dp->GetDefinition());
    ++per_event[b];
    const G4double e = dp->GetKineticEnergy() / MeV;
    const G4ThreeVector p = dp->GetMomentum();
    const G4double pm = p.mag();
    const G4double c = (pm > 0.0) ? (p.z() / pm) : 1.0;
    SpeciesBucket& s = st.sp[b];
    ++s.count;
    s.sum_e += e;
    s.sum_e2 += e * e;
    s.sum_cos += c;
    if (e > s.max_e) { s.max_e = e; }
    if (e >= hard_threshold) {
      ++s.count_hard;
      s.sum_e_hard += e;
      s.sum_e2_hard += e * e;
    }
    ekin_sum += e;
    pz_sum += p.z() / MeV;
  }
  for (G4int b = 0; b < 13; ++b) {
    st.sp[b].sum_n += per_event[b];
    st.sp[b].sum_n2 += double(per_event[b]) * double(per_event[b]);
    if (per_event[b] > 0) { ++st.sp[b].events; }
  }
  st.sum_ekin += ekin_sum;
  st.sum_ekin2 += ekin_sum * ekin_sum;
  st.sum_pz += pz_sum;
}

/// Delete the secondaries' `G4DynamicParticle`s and empty the list.
///
/// NOT optional bookkeeping. `G4HadSecondary::~G4HadSecondary()` is EMPTY - it does not delete
/// `theP` - and `G4HadFinalState::ClearSecondaries()` is `theSecs.clear()`, which destroys
/// G4HadSecondary values whose destructors free nothing. In a real run
/// `G4HadronicProcess::FillResult` takes ownership into the G4ParticleChange; a program that
/// calls `ApplyYourself` directly and does not, leaks one G4DynamicParticle per secondary.
/// This campaign is 126 cases x 2,000 events x up to eighty secondaries, which is millions of
/// them, and that is what killed the first version of it - see docs/RISK.md V174.
void release(G4HadFinalState* hfs) {
  if (hfs == nullptr) { return; }
  const G4int n = G4int(hfs->GetNumberOfSecondaries());
  for (G4int i = 0; i < n; ++i) { delete hfs->GetSecondary(i)->GetParticle(); }
  hfs->Clear();
}

/// A marker row, flushed, written BEFORE a case runs. A dump that dies takes its buffered
/// stdout with it - three runs of this file died invisibly before the gamma-nuclear block
/// learned the lesson - so the progress is a row in the CSV, which survives because it is
/// flushed. The test skips any row whose `model` is "start".
void mark_case(FILE* f, const char* model, const char* particle, double ke, G4int Z, G4int A) {
  std::fprintf(f, "start,%s|%s,%.17g,%d,%d,0,0,0,0,0,0,0,0,0,0,0\n", model, particle, ke, Z, A);
  std::fflush(f);
}

void write_case(FILE* f, FILE* fs, const char* model, const char* particle, double ke, G4int Z,
                G4int A, const CaseStats& st) {
  const double n = double(st.events);
  if (n <= 0.0) { return; }
  std::fprintf(f, "%s,%s,%.17g,%d,%d,%lld,%lld,%lld,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,"
                  "%.17g,%.17g\n",
               model, particle, ke, Z, A, st.events, st.empty, st.no_photon,
               st.sum_n / n, std::sqrt(std::max(0.0, st.sum_n2 / n - (st.sum_n / n) * (st.sum_n / n))),
               st.sum_ekin / n,
               std::sqrt(std::max(0.0, st.sum_ekin2 / n - (st.sum_ekin / n) * (st.sum_ekin / n))),
               st.sum_lep / n,
               std::sqrt(std::max(0.0, st.sum_lep2 / n - (st.sum_lep / n) * (st.sum_lep / n))),
               st.sum_coslep / n, st.sum_pz / n);
  for (G4int b = 0; b < 13; ++b) {
    const SpeciesBucket& s = st.sp[b];
    const double c = double(s.count);
    // The SOFT moments - everything below the hard threshold - are what a spectrum comparison
    // can use; `count_hard`, `ekin_hard_mean` and `max_ekin` are the rare component, compared
    // as a rate. See SpeciesBucket's comment for why the two cannot share a mean.
    const double csoft = double(s.count - s.count_hard);
    const double soft_sum = s.sum_e - s.sum_e_hard;
    const double soft_sum2 = s.sum_e2 - s.sum_e2_hard;
    const double soft_mean = (csoft > 0.0) ? soft_sum / csoft : 0.0;
    // The SOFT rms as well as the soft mean, because the test compares the two means with
    // Welch's standard error and needs both samples' own spreads: for one row of this campaign
    // the oracle's spread is fourteen times the port's, twelve samples against a hundred and
    // forty, and using either side's alone gives 2.2 sigma or 8.0.
    const double soft_rms =
        (csoft > 0.0) ? std::sqrt(std::max(0.0, soft_sum2 / csoft - soft_mean * soft_mean)) : 0.0;
    std::fprintf(fs,
                 "%s,%s,%.17g,%d,%d,%s,%lld,%lld,%.17g,%.17g,%.17g,%.17g,%.17g,%lld,%.17g,"
                 "%.17g,%.17g,%.17g\n",
                 model, particle, ke, Z, A, species_bucket_name(b), s.count, s.events,
                 s.sum_n / n,
                 std::sqrt(std::max(0.0, s.sum_n2 / n - (s.sum_n / n) * (s.sum_n / n))),
                 c > 0.0 ? s.sum_e / c : 0.0,
                 c > 0.0 ? std::sqrt(std::max(0.0, s.sum_e2 / c - (s.sum_e / c) * (s.sum_e / c)))
                         : 0.0,
                 c > 0.0 ? s.sum_cos / c : 0.0,
                 s.count_hard,
                 s.count_hard > 0 ? s.sum_e_hard / double(s.count_hard) : 0.0,
                 soft_mean, s.max_e, soft_rms);
  }
}

void dump_emextra_apply(const DumpContext&) {
  // THE DEFAULT IS ONE EVENT PER CASE, AND IT IS A CHOICE ABOUT THE LEAD'S INTEGRATION CHAIN
  // TAKING TWELVE MINUTES RATHER THAN TWENTY. It used to be a way round a crash; that crash is
  // fixed and the history below is kept because the wrong conclusions in it were reasonable.
  //
  // THE CRASH, AND WHY IT WAS NOT THIS FILE'S. Until 2026-09-19 this dump died with no message
  // inside its sixth case - a 10 MeV photon on carbon through `G4LowEGammaNuclearModel` - and
  // took every dump after it down. The fault was four STACK objects in
  // `ref/dump/dump_deexcitation.cc`: `~G4FermiBreakUpVI()` does
  // `if(IsMasterThread()) { delete thePool; thePool = nullptr; }` and `thePool` is a CLASS
  // STATIC shared with QBBC's own de-excitation, which is what this dump uses, because
  // `G4LowEGammaNuclearModel` takes the shared "PRECO" model out of the registry rather than
  // building its own. `~G4ExcitationHandler()` and `~G4PreCompoundModel()` reach the same
  // destructor. This dump was simply the first one that could see the damage. docs/RISK.md
  // V174 has the bisect, the cdb stack and the fix.
  //
  // What was measured on the way, and each row was true when it was taken:
  //
  //   this dump alone, 2,000 events                    126 of 126 cases, exit 0
  //   this dump alone, 20,000 events                   126 of 126 cases, exit 0, 4.9 minutes
  //   this dump + the nine that link before it, 2,000  126 of 126 cases, exit 0
  //   this dump + dump_ftf.cc, 1 event                 exit 0
  //   the EIGHTEEN OTHERS WITHOUT THIS ONE             exit 0, every dump ran
  //   this dump + all eighteen others, 1 event         0xC0000005 after this dump's files
  //   this dump + all eighteen others, 2,000           dies in case 6, every time
  //
  // so this dump was NECESSARY for the crash and the campaign size was not: one event per case
  // died too. It was not memory exhaustion - the process is at 380 MB and the machine has
  // gigabytes - and it was not this file's `release()`, which was replaced with a leak and
  // rebuilt and changed nothing. It was two dumps away, in a destructor.
  //
  // The campaign that the lead's `ref/oracle/run.bat` runs is ONE event
  // per case - enough to prove every model is reachable and every column is written, and not
  // enough to compare a distribution - and the statistical oracle is regenerated deliberately
  // with `G4GPU_EMEXTRA_EVENTS=20000`. `tests/test_emextra_models.cu` reads the count out of the
  // `events` column and pools the two standard errors, so it is correct either way and says
  // which it had.
  //
  // TWENTY thousand and not two, since docs/RISK.md V177. At 2,000 events a rare species'
  // SOFT spectrum can rest on two samples - 23 of 991 rows did - and a two-sample standard
  // deviation has one degree of freedom, so a row read 7.54 sigma on an oracle spread of 0.77
  // MeV that the same case at 20,000 events measured as 4.75. Ten times the events costs four
  // and a half minutes for all 126 cases standalone, which is nothing, and leaves only 10 rows
  // of 1,034 on two samples; the comparison itself was corrected as well, because no event
  // count removes the last of them.
  long long n_events = 1;
  if (const char* e = std::getenv("G4GPU_EMEXTRA_EVENTS")) {
    const long long v = std::atoll(e);
    if (v > 0) { n_events = v; }
  }

  struct Target { G4int z, a; };
  const Target targets[] = {{1, 1}, {6, 12}, {8, 16}, {13, 27}, {26, 56}, {82, 208}};

  FILE* f = std::fopen("emextra_apply.csv", "w");
  FILE* fs = std::fopen("emextra_apply_species.csv", "w");
  if (f == nullptr || fs == nullptr) { return; }
  std::fprintf(f, "model,particle,ke_MeV,Z,A,events,empty,no_photon,mult_mean,mult_rms,"
                  "ekin_mean,ekin_rms,lep_mean,lep_rms,coslep_mean,pz_mean\n");
  std::fprintf(fs, "model,particle,ke_MeV,Z,A,species,count,events,yield_mean,yield_rms,"
                   "ekin_mean,ekin_rms,cos_mean,count_hard,ekin_hard_mean,ekin_soft_mean,"
                   "max_ekin,ekin_soft_rms\n");

  // The models. Each is constructible after the run manager has initialised: the low-energy
  // gamma model finds QBBC's "PRECO" in G4HadronicInteractionRegistry, and both lepton models
  // find their cross sections in G4CrossSectionDataSetRegistry - which is exactly the order
  // dependence `registry_has_PhotoNuclearXS` records.
  auto* lemod = new G4LowEGammaNuclearModel();
  auto* cascade = new G4CascadeInterface();
  auto* evd = new G4ElectroVDNuclearModel();
  auto* mvd = new G4MuonVDNuclearModel();

  const double preco_energies[] = {10.0, 30.0, 100.0, 150.0, 199.0};
  const double bert_energies[] = {200.0, 300.0, 1000.0, 3000.0, 5000.0};
  const double lep_energies[] = {50.0, 200.0, 1000.0, 10000.0};
  const double mu_energies[] = {200.0, 1000.0, 10000.0};

  // A fixed seed per case, derived from the case itself so that adding a case does not move
  // any other case's stream.
  auto seed_for = [](const char* m, double ke, G4int Z, G4int A) {
    long s = 20130000;
    for (const char* p = m; *p != '\0'; ++p) { s = s * 31 + G4int(*p); }
    s += G4long(ke * 10.0) * 1000 + Z * 17 + A;
    return (s % 900000000L) + 1L;
  };

  for (const Target& t : targets) {
    for (double ke : preco_energies) {
      CaseStats st;
      mark_case(f, "preco", "gamma", ke, t.z, t.a);
      G4Random::setTheSeed(seed_for("preco", ke, t.z, t.a));
      for (long long k = 0; k < n_events; ++k) {
        G4DynamicParticle dp(G4Gamma::Gamma(), G4ThreeVector(0, 0, 1), ke * MeV);
        G4HadProjectile proj(dp);
        G4Nucleus nuc(t.a, t.z);
        G4HadFinalState* hfs = lemod->ApplyYourself(proj, nuc);
        accumulate(st, hfs, false, 0.0, 0.1 * ke);
        release(hfs);
      }
      write_case(f, fs, "preco", "gamma", ke, t.z, t.a, st);
    }
    std::fflush(f);
    std::fflush(fs);
  }

  for (const Target& t : targets) {
    for (double ke : bert_energies) {
      CaseStats st;
      mark_case(f, "bert", "gamma", ke, t.z, t.a);
      G4Random::setTheSeed(seed_for("bert", ke, t.z, t.a));
      for (long long k = 0; k < n_events; ++k) {
        G4DynamicParticle dp(G4Gamma::Gamma(), G4ThreeVector(0, 0, 1), ke * MeV);
        G4HadProjectile proj(dp);
        G4Nucleus nuc(t.a, t.z);
        G4HadFinalState* hfs = nullptr;
        try {
          hfs = cascade->ApplyYourself(proj, nuc);
        } catch (...) {
          // `throwNonConservationFailure()` is a G4HadronicException that ends the job in a
          // real run. Counted as an empty event here, the same way dump_bertini.cc counts it.
          hfs = nullptr;
        }
        accumulate(st, hfs, false, 0.0, 0.1 * ke);
        release(hfs);
      }
      write_case(f, fs, "bert", "gamma", ke, t.z, t.a, st);
    }
    std::fflush(f);
    std::fflush(fs);
  }

  struct Lep { const char* name; G4ParticleDefinition* (*get)(); };
  const Lep leps[] = {{"e-", get_electron}, {"e+", get_positron}};
  for (const Lep& l : leps) {
    for (const Target& t : targets) {
      for (double ke : lep_energies) {
        CaseStats st;
        mark_case(f, "evd", l.name, ke, t.z, t.a);
        G4Random::setTheSeed(seed_for(l.name, ke, t.z, t.a));
        for (long long k = 0; k < n_events; ++k) {
          G4DynamicParticle dp(l.get(), G4ThreeVector(0, 0, 1), ke * MeV);
          G4HadProjectile proj(dp);
          G4Nucleus nuc(t.a, t.z);
          G4HadFinalState* hfs = nullptr;
          try {
            hfs = evd->ApplyYourself(proj, nuc);
          } catch (...) { hfs = nullptr; }
          // The model writes the SCATTERED LEPTON into the particle change rather than as a
          // secondary, so its energy and angle come off GetEnergyChange/GetMomentumChange.
          G4double lep = ke;
          G4double cosl = 1.0;
          G4bool none = false;
          if (hfs != nullptr) {
            lep = hfs->GetEnergyChange() / MeV;
            cosl = hfs->GetMomentumChange().z();
            // "No photon produced" is indistinguishable from outside except by its signature:
            // the lepton keeps its energy exactly and there are no secondaries.
            none = (hfs->GetNumberOfSecondaries() == 0 && std::fabs(lep - ke) < 1.0e-9);
          }
          if (none) { ++st.no_photon; }
          st.sum_coslep += cosl;
          accumulate(st, hfs, true, lep, 0.1 * ke);
          release(hfs);
        }
        write_case(f, fs, "evd", l.name, ke, t.z, t.a, st);
      }
      std::fflush(f);
      std::fflush(fs);
    }
  }

  for (const Target& t : targets) {
    for (double ke : mu_energies) {
      CaseStats st;
      mark_case(f, "mvd", "mu-", ke, t.z, t.a);
      G4Random::setTheSeed(seed_for("mu-", ke, t.z, t.a));
      for (long long k = 0; k < n_events; ++k) {
        G4DynamicParticle dp(G4MuonMinus::MuonMinus(), G4ThreeVector(0, 0, 1), ke * MeV);
        G4HadProjectile proj(dp);
        G4Nucleus nuc(t.a, t.z);
        G4HadFinalState* hfs = nullptr;
        try {
          hfs = mvd->ApplyYourself(proj, nuc);
        } catch (...) { hfs = nullptr; }
        G4double lep = ke;
        G4double cosl = 1.0;
        G4bool none = false;
        if (hfs != nullptr) {
          lep = hfs->GetEnergyChange() / MeV;
          cosl = hfs->GetMomentumChange().z();
          none = (hfs->GetNumberOfSecondaries() == 0 && std::fabs(lep - ke) < 1.0e-9);
        }
        if (none) { ++st.no_photon; }
        st.sum_coslep += cosl;
        accumulate(st, hfs, true, lep, 0.1 * ke);
        release(hfs);
      }
      write_case(f, fs, "mvd", "mu-", ke, t.z, t.a, st);
    }
    std::fflush(f);
    std::fflush(fs);
  }

  std::fclose(f);
  std::fclose(fs);
}

}  // namespace

G4GPU_REGISTER_DUMP("emextra",
                    "emextra_config.csv emextra_windows.csv emextra_xs.csv emextra_params.csv",
                    dump_emextra);
G4GPU_REGISTER_DUMP("emextra_xs",
                    "emextra_photonuc.csv emextra_electronuc.csv emextra_eqphoton.csv "
                    "emextra_kokoulin.csv emextra_gammanuc.csv",
                    dump_emextra_xs);
G4GPU_REGISTER_DUMP("emextra_apply",
                    "emextra_apply.csv emextra_apply_species.csv",
                    dump_emextra_apply);
