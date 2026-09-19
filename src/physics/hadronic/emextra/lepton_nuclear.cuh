// `electronNuclear`, `positronNuclear` and `muonNuclear` - the three lepto-nuclear processes
// QBBC puts on e-, e+, mu- and mu+, their model choice, and the entry points the lepton
// steppers will call.
//
// These are the siblings of `photon_nuclear` in photon_nuclear.cuh and they exist for the same
// reason: a MODEL is `ApplyYourself` and a PROCESS is `G4EnergyRangeManager` plus a model, and
// P15 calls the process. `electro_vd_apply` and `muon_vd_apply` in lepton_vd.cuh are the models
// - `G4ElectroVDNuclearModel::ApplyYourself` and `G4MuonVDNuclearModel::ApplyYourself`
// transcribed - and a caller that reached them directly would be skipping the energy range,
// which is the thing that decides whether there is a model at all.
//
// ONE MODEL EACH, SO THE WINDOW IS INERT - AND THAT IS GEANT4'S BEHAVIOUR, NOT A SHORTCUT HERE
//
// `emextra::qbbc_config(11)`, `(-11)` and `(13)` each have exactly one model over 0 - 1e9 MeV.
// That 1 PeV is not an inherited default: `G4HadronicInteraction`'s constructor takes
// `G4HadronicParameters::GetMaxEnergy()`, which is 100 TeV and is what the gamma's QGS arm
// gets, and BOTH VD models then override it in their own constructors - `SetMinEnergy(0.0);
// SetMaxEnergy(1*PeV);`. So a window was written on purpose, twice. config.cuh's
// `vd_max_energy_MeV()` is that number, dumped from the running physics list and asserted
// rather than remembered. And with one model registered,
// `G4EnergyRangeManager::GetHadronicInteraction` does not look at it at all:
//
//     // VI shortcut: if only one interaction is registered skip all checks
//     if(1 == theHadronicInteractionCounter) { return theHadronicInteraction[0]; }
//
// so a 10 PeV muon gets `G4MuonVDNuclearModel` in Geant4 exactly as it does here, 1 PeV window
// or not. There is no cliff and no "no model found" for these three processes, and P5's
// `choose_hadronic_interaction` reproduces the shortcut for the same reason (its comment says
// so for QBBC's elastic, which is the other place it bites). What the 1 PeV number IS good for
// is the cross section: `G4KokoulinMuonNuclearXS`'s table stops at 1 PeV and above it the
// muon's mean free path is the 1 PeV one, which is a separate inertness recorded in
// docs/RISK.md. `tests/test_emextra_models.cu` drives 1 PeV and 10 PeV through this entry point
// and asserts that BOTH run the model, because "the window is enforced" was the natural thing
// to assume and it is false.
//
// So the wrapper is thin on purpose. What it adds over calling the model directly is the
// particle check, the hyper-nuclear refusal, the model identity, and a single result type -
// which is what makes "the port lost a secondary" distinguishable from "there was no model".
// The `kNoModelInRange` arms below cannot fire for a `qbbc_config` of these three particles,
// for the reason just given; they are kept because this entry point takes a pdg from a caller
// and a wrong one must be named rather than run, and the test drives that arm with a proton.
//
// THREE PROCESSES, TWO PROCESS OBJECTS, ONE MODEL OBJECT
//
// `G4EmExtraPhysics::ConstructGammaElectroNuclear` builds ONE `G4ElectroVDNuclearModel` and
// registers the same pointer on both `electronNuclear` and `positronNuclear`, and ONE
// `G4MuonNuclearProcess` registered on both muons. config.cuh records that and why it matters
// for a port: the Geant4 model caches the lepton's energy and the photon's (Q2, nu) in members,
// so two tracks in flight at once would share them. Nothing here caches: every value lives in
// the `LeptonVdResult` the call returns, which is the per-thread version of the same state.
// That is why `electron_nuclear` takes the pdg and does not care which of the two processes
// the caller thinks it is in - the physics is identical and the sign only reaches the final
// state through the scattered lepton it puts back.
#ifndef G4GPU_HADRONIC_EMEXTRA_LEPTON_NUCLEAR_CUH
#define G4GPU_HADRONIC_EMEXTRA_LEPTON_NUCLEAR_CUH

#include "physics/hadronic/emextra/config.cuh"
#include "physics/hadronic/emextra/lepton_vd.cuh"
#include "physics/hadronic/process.cuh"

namespace g4gpu::physics::hadronic::emextra {

/// What one `electron_nuclear` or `muon_nuclear` call did.
///
/// `vd` is the model's own result - the equivalent photon, the scattered lepton, the gamma
/// chain's refusal if it had one - and is left at its default when the range manager found no
/// model, which is the case `refusal == kNoModelInRange` names.
struct LeptonNuclearResult {
  Process process = Process::kNone;
  Model model = Model::kNone;
  ModelChoice choice = ModelChoice::kOk;
  int n_secondaries = 0;
  EmExtraRefusal refusal = EmExtraRefusal::kNone;
  LeptonVdResult vd;
};

/// Shared by the two entry points: check the target, choose the model, and report the window.
///
/// Returns true when a model was chosen and the caller should run it. `cfg` is the process's
/// own configuration, so the window that is enforced is the one `emextra_windows.csv` carries.
template <typename real_t, typename Rng>
__host__ __device__ inline bool lepton_choose_model(const ProcessConfig& cfg,
                                                    const HadProjectile<real_t>& projectile,
                                                    const HadNucleus& target, Rng& rng,
                                                    LeptonNuclearResult& r) {
  if (target.l != 0) {
    // A hyper-nuclear target. `G4Nucleus` carries no lambda count in 11.1.1 and neither model
    // has a branch for one, so this is refused here rather than silently treated as A - L.
    r.refusal = EmExtraRefusal::kHyperNucleus;
    return false;
  }
  r.process = cfg.process;
  ModelRange<real_t> ranges[3];
  const int n = model_ranges<real_t>(cfg, ranges);
  const ModelSelection sel =
      choose_hadronic_interaction<real_t>(ranges, n, projectile.kin_energy, 0, rng);
  r.choice = sel.status;
  if (sel.status != ModelChoice::kOk) {
    r.refusal = EmExtraRefusal::kNoModelInRange;
    return false;
  }
  r.model = cfg.models[sel.index].model;
  return true;
}

/// `electronNuclear` and `positronNuclear`: choose a model by energy and run it.
///
/// P5's shape - projectile, target, final state, caller-owned workspace, rng. The pdg selects
/// which of the two processes this is (11 -> electronNuclear, -11 -> positronNuclear) and both
/// run the same `G4ElectroVDNuclearModel`. A pdg that is neither is a wiring error, not a
/// physics case, and is reported as `kNoModelInRange` the way `photon_nuclear` reports a
/// non-photon.
template <typename real_t, int kCap, typename Rng>
__host__ __device__ inline LeptonNuclearResult electron_nuclear(
    const HadProjectile<real_t>& projectile, const HadNucleus& target,
    HadFinalState<real_t, kCap>& fs, const GammaWorkspace& ws, const data::LevelTable& lt,
    const deex::FermiPool& pool, Rng& rng) {
  LeptonNuclearResult r;
  fs.clear();
  if (projectile.pdg != 11 && projectile.pdg != -11) {
    r.refusal = EmExtraRefusal::kNoModelInRange;
    return r;
  }
  const ProcessConfig cfg = qbbc_config(projectile.pdg);
  if (!lepton_choose_model<real_t>(cfg, projectile, target, rng, r)) { return r; }
  if (r.model != Model::kElectroVD) {
    r.refusal = EmExtraRefusal::kNoModelInRange;
    return r;
  }
  r.vd = electro_vd_apply(projectile, target, fs, ws, lt, pool, rng);
  r.refusal = r.vd.refusal;
  r.n_secondaries = fs.n_secondaries;
  if (fs.secondary_overflow > 0) { r.refusal = EmExtraRefusal::kCapacity; }
  return r;
}

/// `muonNuclear`: choose a model by energy and run it.
///
/// `tab` is the 5 x 73 x 801 sampling table `mu_vd_make_sampling_table` builds ONCE on the host
/// - 2.3 MB, shared read-only by every thread, never rebuilt per call - and it is a parameter
/// rather than a member for the reason lepton_vd.cuh gives: `G4MuonVDNuclearModel` builds it in
/// its constructor and a device model has no constructor to build it in.
template <typename real_t, int kCap, typename Rng>
__host__ __device__ inline LeptonNuclearResult muon_nuclear(
    const HadProjectile<real_t>& projectile, const HadNucleus& target,
    HadFinalState<real_t, kCap>& fs, const MuVdTable& tab, const GammaWorkspace& ws,
    const data::LevelTable& lt, const deex::FermiPool& pool, Rng& rng) {
  LeptonNuclearResult r;
  fs.clear();
  if (projectile.pdg != 13 && projectile.pdg != -13) {
    r.refusal = EmExtraRefusal::kNoModelInRange;
    return r;
  }
  const ProcessConfig cfg = qbbc_config(projectile.pdg);
  if (!lepton_choose_model<real_t>(cfg, projectile, target, rng, r)) { return r; }
  if (r.model != Model::kMuonVD) {
    r.refusal = EmExtraRefusal::kNoModelInRange;
    return r;
  }
  r.vd = muon_vd_apply(projectile, target, fs, tab, ws, lt, pool, rng);
  r.refusal = r.vd.refusal;
  r.n_secondaries = fs.n_secondaries;
  if (fs.secondary_overflow > 0) { r.refusal = EmExtraRefusal::kCapacity; }
  return r;
}

}  // namespace g4gpu::physics::hadronic::emextra

#endif
