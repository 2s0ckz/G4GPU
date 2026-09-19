// G4EmExtraPhysics as QBBC configures it: the four processes, their cross sections, their
// models and the energy windows between them.
//
// Read from the RUNNING physics list and not from `G4EmExtraPhysics.hh`'s eleven private bools.
// `ref/oracle/emextra_config.csv`, `emextra_windows.csv` and `emextra_xs.csv` are what
// `ref/dump/dump_emextra.cc` finds by walking the five particles' process managers and
// `G4HadronicProcessStore`, and `tests/test_emextra_config.cu` asserts every row of this file
// against them in BOTH directions - a process this file claims and the run does not have is as
// much a failure as the reverse.
//
// ------------------------------------------------------------------------------------------
// WHAT THE RUN SAYS, AND WHERE docs/HADRONIC_PLAN.md's TABLE IS WRONG
//
// The plan's row reads "gamma | G4GammaNuclearXS | G4LowEGammaNuclearModel (<200 MeV, via
// PreCompound), Bertini above" and lists the constructor under `electromagnetic/`. Four
// corrections, all of them measured:
//
//   1. **The class lives in `physics_lists/constructors/gamma_lepto_nuclear/`**, not
//      `electromagnetic/`.
//   2. **The photon's high-energy generator is QGS, not FTF.** `ConstructGammaElectroNuclear`
//      builds `G4QGSModel<G4GammaParticipants>` with `G4QGSMFragmentation`, wraps it in a
//      `G4TheoFSGenerator` and registers that from 3 GeV to 100 TeV. P11 ported FTF. The
//      `emextra_windows.csv` column `high_energy_generator` says `QGSModel` for the photon and
//      it is the only place in the whole configuration where the distinction is visible - the
//      wrapper is called "TheoFSGenerator" either way. Refused by name; see `kQgsGammaString`.
//   3. **The low-energy limit and Bertini's floor differ by 1 MeV**, deliberately:
//      `cascade->SetMinEnergy(fGNLowEnergyLimit - CLHEP::MeV)`. So `GammaNPreco` runs 0 - 200
//      and `BertiniCascade` 199 - 6000, and a photon between 199 and 200 MeV is assigned to one
//      of them at random with P(Bertini) = (E - 199)/(200 - 199). That 1 MeV window is a real
//      part of QBBC and `test_emextra_config.cu` counts draws in it.
//   4. **`photonNuclear` is NOT on the gamma's process manager.** `G4EmStandardPhysics::
//      ConstructProcess` runs first and sets `SetGeneralProcessActive(true)`, so
//      `G4LossTableManager::GetGammaGeneralProcess()` is non-null by the time
//      `G4EmExtraPhysics` runs and the process goes in through `AddHadProcess` instead of
//      `RegisterProcess`. The gamma's manager holds `Transportation` and `GammaGeneralProc`
//      and nothing else. The electron's and positron's general processes are NOT created -
//      nothing in 11.1.1 calls `SetElectronGeneralProcess` - so `electronNuclear` and
//      `positronNuclear` ARE ordinary processes on their managers. That asymmetry is why
//      `dump_emextra.cc` reads both lists.
//
// The fourth one has a consequence the plan's table cannot show: **LEND is unreachable in
// QBBC twice over**. `gLENDActivated` is false, and even if it were true
// `ConstructLENDGammaNuclear` is only called in the `else` branch taken when the gamma general
// process does NOT exist. `G4LENDDATA` is also unset on this machine, which is the third.
//
// ------------------------------------------------------------------------------------------
// WHAT IS OFF, WITH THE FLAG VALUES
//
// `G4EmExtraPhysics`' defaults, from the header, and each confirmed by the ABSENCE of its
// process in `emextra_config.csv` (QBBC constructs the class with `new G4EmExtraPhysics(ver)`
// and never touches `G4EmMessenger`, so the defaults are what run):
//
//   gnActivated        true    -> photonNuclear exists
//   eActivated         true    -> electronNuclear, positronNuclear exist
//   munActivated       true    -> muonNuclear exists
//   fUseGammaNuclearXS true    -> the data set is GammaNuclearXS and not PhotoNuclearXS
//   gLENDActivated     false   -> no G4LENDorBERTModel, no G4LENDCombinedCrossSection
//   synActivated       false   -> no G4SynchrotronRadiation on e-, e+ (nor on mu, p, pi, ion,
//   synActivatedForAll false      which is the second flag)
//   gmumuActivated     false   -> no G4GammaConversionToMuons
//   mmumuActivated     false   -> no G4MuonToMuonPairProduction
//   pmumuActivated     false   -> no G4AnnihiToMuPair, and no "AnnihiToTauPair" either
//   phadActivated      false   -> no G4eeToHadrons
//   fNuActivated       false   -> none of the 18 neutrino classes; the six neutrino species
//                                 are still CONSTRUCTED by ConstructParticle and have no
//                                 process at all
#ifndef G4GPU_HADRONIC_EMEXTRA_CONFIG_CUH
#define G4GPU_HADRONIC_EMEXTRA_CONFIG_CUH

#include "core/units.cuh"
#include "physics/hadronic/process.cuh"

namespace g4gpu::physics::hadronic::emextra {

/// The four processes `G4EmExtraPhysics::ConstructProcess` creates in QBBC. The names are the
/// process names Geant4 gives them, which is what `emextra_config.csv` carries.
enum class Process : int {
  kNone = 0,
  kPhotonNuclear,    ///< "photonNuclear",   gamma
  kElectronNuclear,  ///< "electronNuclear", e-
  kPositronNuclear,  ///< "positronNuclear", e+
  kMuonNuclear       ///< "muonNuclear",     mu- and mu+
};

__host__ __device__ inline const char* process_name(Process p) {
  switch (p) {
    case Process::kPhotonNuclear: return "photonNuclear";
    case Process::kElectronNuclear: return "electronNuclear";
    case Process::kPositronNuclear: return "positronNuclear";
    case Process::kMuonNuclear: return "muonNuclear";
    case Process::kNone: return "(none)";
  }
  return "(unknown)";
}

/// The cross-section data set each process carries. One each - `emextra_xs.csv` has exactly one
/// row per process, so `G4CrossSectionDataStore`'s backwards walk over its list has nothing to
/// walk past.
enum class DataSet : int {
  kNone = 0,
  kGammaNuclearXS,          ///< "GammaNuclearXS"       - P2's reader + CHIPS above 130 MeV
  kElectroNuclearXS,        ///< "ElectroNuclearXS"     - xs/chips_electronuclear.cuh
  kKokoulinMuonNuclearXS    ///< "KokoulinMuonNuclearXS"- xs/kokoulin_muon_xs.cuh
};

__host__ __device__ inline const char* dataset_name(DataSet d) {
  switch (d) {
    case DataSet::kGammaNuclearXS: return "GammaNuclearXS";
    case DataSet::kElectroNuclearXS: return "ElectroNuclearXS";
    case DataSet::kKokoulinMuonNuclearXS: return "KokoulinMuonNuclearXS";
    case DataSet::kNone: return "(none)";
  }
  return "(unknown)";
}

/// The final-state models, by the `GetModelName()` string Geant4 answers with.
enum class Model : int {
  kNone = 0,
  kGammaNPreco,      ///< "GammaNPreco"             - G4LowEGammaNuclearModel
  kBertiniCascade,   ///< "BertiniCascade"          - G4CascadeInterface (P10)
  kTheoFSGenerator,  ///< "TheoFSGenerator" wrapping QGSModel - NOT ported, see kQgsGammaString
  kElectroVD,        ///< "G4ElectroVDNuclearModel"
  kMuonVD            ///< "G4MuonVDNuclearModel"
};

__host__ __device__ inline const char* model_name(Model m) {
  switch (m) {
    case Model::kGammaNPreco: return "GammaNPreco";
    case Model::kBertiniCascade: return "BertiniCascade";
    case Model::kTheoFSGenerator: return "TheoFSGenerator";
    case Model::kElectroVD: return "G4ElectroVDNuclearModel";
    case Model::kMuonVD: return "G4MuonVDNuclearModel";
    case Model::kNone: return "(none)";
  }
  return "(unknown)";
}

/// One `G4EnergyRangeManager` row: a model and the window it is asked for, MeV.
struct ModelWindow {
  Model model = Model::kNone;
  double emin_MeV = 0.0;
  double emax_MeV = 0.0;
};

/// The whole configuration for one process.
struct ProcessConfig {
  Process process = Process::kNone;
  DataSet dataset = DataSet::kNone;
  int n_models = 0;
  ModelWindow models[3];
  bool ok = false;  ///< false: this particle has no G4EmExtraPhysics process at all
};

/// `G4HadronicParameters::GetMaxEnergy()` - 100 TeV. Every ceiling in QBBC's hadronic chain is
/// this and not the 50 GeV a reading of the model classes would suggest (docs/PORTED.md
/// 2.1.11b makes the same point for FTFP).
__host__ __device__ inline constexpr double max_energy_MeV() { return 1.0e8; }
/// `G4HadronicInteraction`'s own ceiling for the two VD models: `SetMaxEnergy(1*PeV)`, which is
/// 1e9 MeV and is ten times `max_energy_MeV()`. A model is allowed a window wider than the
/// parameters' ceiling because nothing clips it; the process simply never sees such a track.
__host__ __device__ inline constexpr double vd_max_energy_MeV() { return 1.0e9; }
/// `fGNLowEnergyLimit`, the G4EmExtraPhysics constructor's `200*CLHEP::MeV`.
__host__ __device__ inline constexpr double gn_low_energy_limit_MeV() { return 200.0; }
/// `GetMinEnergyTransitionFTF_Cascade()` and `GetMaxEnergyTransitionFTF_Cascade()` - 3 and
/// 6 GeV. Named for FTF in G4HadronicParameters and used here to bound a QGS model, which is
/// where the plan's "FTFP above" came from.
__host__ __device__ inline constexpr double transition_min_MeV() { return 3000.0; }
__host__ __device__ inline constexpr double transition_max_MeV() { return 6000.0; }

/// QBBC's configuration, by PDG code. Returns `ok = false` for every particle
/// `G4EmExtraPhysics` gives no process to.
///
/// The photon's three rows are in REGISTRATION order, which is the order
/// `G4EnergyRangeManager` stores them in and therefore the order
/// `choose_hadronic_interaction`'s "last two matches" walk sees. `emextra_windows.csv` carries
/// the index for that reason and the test compares it.
__host__ __device__ inline ProcessConfig qbbc_config(int pdg) {
  ProcessConfig c;
  switch (pdg) {
    case 22:  // gamma
      c.process = Process::kPhotonNuclear;
      c.dataset = DataSet::kGammaNuclearXS;
      c.n_models = 3;
      // lemod->SetMaxEnergy(fGNLowEnergyLimit); its min is G4HadronicInteraction's 0.
      c.models[0] = {Model::kGammaNPreco, 0.0, gn_low_energy_limit_MeV()};
      // cascade->SetMinEnergy(fGNLowEnergyLimit - 1 MeV); SetMaxEnergy(transition max).
      c.models[1] = {Model::kBertiniCascade, gn_low_energy_limit_MeV() - 1.0,
                     transition_max_MeV()};
      // theModel->SetMinEnergy(transition min); SetMaxEnergy(param->GetMaxEnergy()).
      c.models[2] = {Model::kTheoFSGenerator, transition_min_MeV(), max_energy_MeV()};
      c.ok = true;
      return c;
    case 11:  // e-
      c.process = Process::kElectronNuclear;
      c.dataset = DataSet::kElectroNuclearXS;
      c.n_models = 1;
      c.models[0] = {Model::kElectroVD, 0.0, vd_max_energy_MeV()};
      c.ok = true;
      return c;
    case -11:  // e+
      // The SAME G4ElectroVDNuclearModel object is registered on both processes - one `new`,
      // two `RegisterMe` calls - so an e+ and an e- share the model's cached lepton energy and
      // photon (Q2, nu). Harmless in a serial run because ApplyYourself writes all three before
      // it reads any, and worth writing down because a port with per-track state must not share
      // them.
      c.process = Process::kPositronNuclear;
      c.dataset = DataSet::kElectroNuclearXS;
      c.n_models = 1;
      c.models[0] = {Model::kElectroVD, 0.0, vd_max_energy_MeV()};
      c.ok = true;
      return c;
    case 13:   // mu-
    case -13:  // mu+
      // One G4MuonNuclearProcess object registered on BOTH muons, unlike the electron's pair of
      // processes: `ph->RegisterProcess(muNucProcess, muonplus)` and the same pointer for
      // muonminus. So mu+ and mu- share a process AND a model.
      c.process = Process::kMuonNuclear;
      c.dataset = DataSet::kKokoulinMuonNuclearXS;
      c.n_models = 1;
      c.models[0] = {Model::kMuonVD, 0.0, vd_max_energy_MeV()};
      c.ok = true;
      return c;
    default:
      return c;
  }
}

/// The `ModelRange` array P5's `choose_hadronic_interaction` wants, built from `qbbc_config`.
/// Returns the number of rows written.
template <typename real_t>
__host__ __device__ inline int model_ranges(const ProcessConfig& c, ModelRange<real_t>* out) {
  for (int i = 0; i < c.n_models; ++i) {
    out[i].min_energy = static_cast<real_t>(c.models[i].emin_MeV) * units::MeV<real_t>();
    out[i].max_energy = static_cast<real_t>(c.models[i].emax_MeV) * units::MeV<real_t>();
    out[i].applicable = true;
  }
  return c.n_models;
}

// ---------------------------------------------------------------------------------------------
// What this package does not do, by name
// ---------------------------------------------------------------------------------------------

/// Every sub-case of `G4EmExtraPhysics` this port refuses, and the Geant4 symbol it would have
/// needed. Each is a symbol that exists in 11.1.1, so a refusal message can be grepped for.
enum class EmExtraRefusal : int {
  kNone = 0,
  /// `G4QGSModel<G4GammaParticipants>` + `G4QGSMFragmentation` + `G4ExcitedStringDecay` - the
  /// photon's high-energy generator above 3 GeV. P11 ported FTF, which is a DIFFERENT string
  /// model; QGS has its own participants, its own fragmentation and its own parameters. Not an
  /// approximation away from FTF: substituting one for the other would be a different physics
  /// list. See `emextra_windows.csv`'s `high_energy_generator` column.
  kQgsGammaString,
  /// `G4LightTargetCollider` - a gamma on A < 3. WRITTEN since this package's light-target
  /// file; the enumerator stays because a build without it, or a projectile other than a
  /// gamma reaching that branch, still has to report by name.
  kLightTargetCollider,
  /// `G4SynchrotronRadiation`. `synActivated` is false in QBBC and no process exists.
  kSynchrotronRadiation,
  /// `G4GammaConversionToMuons`, `G4AnnihiToMuPair` (and its "AnnihiToTauPair" second
  /// instance), `G4MuonToMuonPairProduction`, `G4eeToHadrons`. All four flags false.
  kMuPairAndHadronChannels,
  /// The eighteen neutrino classes - `G4NeutrinoElectronProcess` and the three
  /// `G4*NeutrinoNucleusProcess` with their twelve models and three total cross sections.
  /// `fNuActivated` is false.
  kNeutrinoProcesses,
  /// `G4LENDorBERTModel` + `G4LENDCombinedCrossSection`. Off three times over - see this
  /// file's header.
  kLend,
  /// A hyper-nuclear target (L != 0), which no model in this package handles.
  kHyperNucleus,
  /// A capacity: a secondary buffer, a product list, a workspace array.
  kCapacity,
  /// The model chain reported a refusal of its own - Bertini's, PreCompound's or the
  /// equivalent-photon sampler's. The caller reads the sub-status for which.
  kSubModel,
  /// `G4EnergyRangeManager` could not choose: no model in range, fully nested ranges, or more
  /// than two competing. P5's `ModelChoice` says which.
  kNoModelInRange
};

__host__ __device__ inline const char* refusal_name(EmExtraRefusal r) {
  switch (r) {
    case EmExtraRefusal::kNone: return "(none)";
    case EmExtraRefusal::kQgsGammaString:
      return "G4QGSModel<G4GammaParticipants> + G4QGSMFragmentation";
    case EmExtraRefusal::kLightTargetCollider: return "G4LightTargetCollider";
    case EmExtraRefusal::kSynchrotronRadiation: return "G4SynchrotronRadiation";
    case EmExtraRefusal::kMuPairAndHadronChannels:
      return "G4GammaConversionToMuons / G4AnnihiToMuPair / G4MuonToMuonPairProduction / "
             "G4eeToHadrons";
    case EmExtraRefusal::kNeutrinoProcesses:
      return "G4NeutrinoElectronProcess and the three G4*NeutrinoNucleusProcess";
    case EmExtraRefusal::kLend: return "G4LENDorBERTModel + G4LENDCombinedCrossSection";
    case EmExtraRefusal::kHyperNucleus: return "a target with L != 0";
    case EmExtraRefusal::kCapacity: return "a buffer capacity in this package";
    case EmExtraRefusal::kSubModel: return "a refusal from Bertini, PreCompound or the sampler";
    case EmExtraRefusal::kNoModelInRange: return "G4EnergyRangeManager::GetHadronicInteraction";
  }
  return "(unknown)";
}

}  // namespace g4gpu::physics::hadronic::emextra

#endif
