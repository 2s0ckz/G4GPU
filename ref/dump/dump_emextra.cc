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

#include "G4CrossSectionDataSetRegistry.hh"
#include "G4CrossSectionDataStore.hh"
#include "G4DynamicParticle.hh"
#include "G4ElectroNuclearCrossSection.hh"
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
    const double as[] = {1.01, 9.01, 26.98, 63.55, 238.03};  // G4MuonVDNuclearModel::adat
    const double ts[] = {1.e3, 1.e4, 1.e5, 1.e6, 1.e9, 1.e11};
    for (double A : as) {
      for (double T : ts) {
        const double epmax = T + G4MuonMinus::MuonMinus()->GetPDGMass() / MeV - 0.5 * 938.272013;
        const double eps[] = {199.9, 200.0, 200.1, 250.0, 0.5 * epmax, 0.9 * epmax,
                              epmax - 1.0, epmax, epmax + 1.0};
        for (double ep : eps) {
          if (ep <= 0.0) { continue; }
          const G4double v =
              mu->ComputeDDMicroscopicCrossSection(T * MeV, 0.0, A * (g / mole), ep * MeV);
          std::fprintf(f, "dd,0,%.17g,%.17g,%.17g,%.17g\n", A, T, ep, v / millibarn);
        }
      }
    }

    // The element cross section, at the 61 log nodes of G4PhysicsLogVector(1 GeV, 1 PeV, 60)
    // and at the geometric midpoints between them.
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

}  // namespace

G4GPU_REGISTER_DUMP("emextra",
                    "emextra_config.csv emextra_windows.csv emextra_xs.csv emextra_params.csv",
                    dump_emextra);
G4GPU_REGISTER_DUMP("emextra_xs",
                    "emextra_photonuc.csv emextra_electronuc.csv emextra_eqphoton.csv "
                    "emextra_kokoulin.csv emextra_gammanuc.csv",
                    dump_emextra_xs);
