// `photonNuclear` - the process QBBC puts on the gamma, its model choice, and the entry point
// the gamma stepper will call.
//
// The configuration is `emextra::qbbc_config(22)` and is measured, not assumed; see config.cuh.
// Three models in registration order:
//
//   0  GammaNPreco       0 - 200 MeV      G4LowEGammaNuclearModel -> P6 -> P3
//   1  BertiniCascade   199 - 6000 MeV    G4CascadeInterface (P10)
//   2  TheoFSGenerator 3000 - 100000 MeV  G4QGSModel<G4GammaParticipants> - REFUSED
//
// `G4EnergyRangeManager::GetHadronicInteraction` picks between two overlapping models with a
// probability linear across the overlap, which P5's `choose_hadronic_interaction` reproduces.
// There are two overlaps here and they are very different sizes:
//
//   199 - 200 MeV    1 MeV wide. `cascade->SetMinEnergy(fGNLowEnergyLimit - CLHEP::MeV)` -
//                    a deliberate one-MeV seam so that neither model has a hard edge.
//   3000 - 6000 MeV  3 GeV wide, and the upper model is not ported, so a photon in it is
//                    refused with probability (E - 3000)/3000: 0 at 3 GeV, 2/3 at 5 GeV, 1 at
//                    6 GeV. `PhotonNuclearResult::chose_unported` carries that, and this
//                    package's statistical campaign reports the rate per case rather than
//                    quietly running Bertini for the QGS share.
//
// THE GAMMA'S BERTINI IS A FOURTH INSTANCE, AND IT DOES NOT USE PRECOMPOUND
//
// docs/PORTED.md 2.1.12 says "QBBC contains Bertini three times, in two configurations" and
// lists p/n, pi+/pi- (both `usePreCompoundDeexcitation`) and the kaon/hyperon set (the
// cascade's own evaporators). There is a FOURTH: `G4EmExtraPhysics::ConstructGammaElectroNuclear`
// does `G4CascadeInterface* cascade = new G4CascadeInterface;` and never calls
// `usePreCompoundDeexcitation()`, so the constructor's `G4CascadeParameters::usePreCompound()`
// test - false - leaves it on `useCascadeDeexcitation()`. A photo-nuclear cascade in QBBC
// therefore de-excites through `bertini/deexcite.cuh` and NOT through P6, which is the opposite
// of what happens to the same residual made by a proton. `qbbc_gamma_deexcite_choice()` below
// is that fact as a function, and the test pins it.
//
// A photon on A < 3 never reaches the cascade at all: `G4CascadeInterface::ApplyYourself` tests
// `aTrack.GetDefinition() == G4Gamma::Gamma() && theNucleus.GetA_asInt() < 3` BEFORE
// `createBullet` and sends it to `G4LightTargetCollider`. That branch is in light_target.cuh
// and is taken here, because P10's `bert::apply_yourself` refuses it by name
// (`InterfaceRefusal::kLightTargetCollider`) and this package owns the arm.
#ifndef G4GPU_HADRONIC_EMEXTRA_PHOTON_NUCLEAR_CUH
#define G4GPU_HADRONIC_EMEXTRA_PHOTON_NUCLEAR_CUH

#include "physics/hadronic/bertini/cascade_interface.cuh"
#include "physics/hadronic/emextra/config.cuh"
#include "physics/hadronic/emextra/gamma_chain.cuh"
#include "physics/hadronic/emextra/light_target.cuh"
#include "physics/hadronic/emextra/low_e_gamma.cuh"
#include "physics/hadronic/process.cuh"

namespace g4gpu::physics::hadronic::emextra {

/// What one `photon_nuclear` call did.
struct PhotonNuclearResult {
  Model model = Model::kNone;        ///< which model the range manager chose
  ModelChoice choice = ModelChoice::kOk;
  bool chose_unported = false;       ///< the choice was TheoFSGenerator/QGS
  int n_secondaries = 0;
  EmExtraRefusal refusal = EmExtraRefusal::kNone;
  /// Filled when `model` is kGammaNPreco.
  LowEGammaResult low_e;
  /// Filled when `model` is kBertiniCascade and the target has A >= 3.
  bert::ApplyResult bertini;
  /// Filled when `model` is kBertiniCascade and the target has A < 3.
  LightTargetResult light;
};

/// `photonNuclear`'s final state: choose a model by energy and run it.
///
/// This is P5's shape - projectile, target, final state, caller-owned workspace, rng - and it
/// is what P15 will call from the gamma stepper once the photo-nuclear cross section has
/// selected an element and an isotope. It does NOT sample the target: `HadNucleus` arrives
/// already chosen by `sample_z_and_a`, as it does for every other inelastic model.
template <typename real_t, int kCap, typename Rng>
__host__ __device__ inline PhotonNuclearResult photon_nuclear(
    const HadProjectile<real_t>& projectile, const HadNucleus& target,
    HadFinalState<real_t, kCap>& fs, const GammaWorkspace& ws, const data::LevelTable& lt,
    const deex::FermiPool& pool, Rng& rng) {
  PhotonNuclearResult r;
  fs.clear();

  if (projectile.pdg != 22) {
    // No model here has an IsApplicable of its own that would catch this, and every caller
    // reaches it through a process registered on the gamma alone. A non-photon is a wiring
    // error, not a physics case.
    r.refusal = EmExtraRefusal::kNoModelInRange;
    return r;
  }
  if (target.l != 0) {
    r.refusal = EmExtraRefusal::kHyperNucleus;
    return r;
  }

  const ProcessConfig cfg = qbbc_config(22);
  ModelRange<real_t> ranges[3];
  const int n = model_ranges<real_t>(cfg, ranges);
  const ModelSelection sel =
      choose_hadronic_interaction<real_t>(ranges, n, projectile.kin_energy, 0, rng);
  r.choice = sel.status;
  if (sel.status != ModelChoice::kOk) {
    r.refusal = EmExtraRefusal::kNoModelInRange;
    return r;
  }
  r.model = cfg.models[sel.index].model;

  switch (r.model) {
    case Model::kGammaNPreco:
      r.low_e = low_e_gamma_apply(projectile, target, fs, lt, pool, ws.preco, rng);
      r.refusal = r.low_e.refusal;
      break;

    case Model::kBertiniCascade: {
      if (target.a < 3) {
        // G4CascadeInterface::ApplyYourself's own branch, before createBullet.
        r.light = light_target_collide(projectile, target, fs, bert::default_cascade_params(),
                                       ws.bert_ws, *ws.epo, *ws.global_out, rng);
        r.refusal = r.light.refusal;
        break;
      }
      if (!ws.complete()) {
        r.refusal = EmExtraRefusal::kCapacity;
        break;
      }
      r.bertini = bert::apply_yourself(
          projectile, target, fs, qbbc_gamma_deexcite_choice(), bert::default_cascade_params(),
          bert::default_interface_limits(), *ws.model, *ws.global_out, *ws.out, *ws.dex_out,
          *ws.tmp, *ws.epo, *ws.bert_ws, lt, pool, ws.preco, 0, rng);
      if (r.bertini.refusal != bert::InterfaceRefusal::kNone) {
        r.refusal = (r.bertini.refusal == bert::InterfaceRefusal::kLightTargetCollider)
                        ? EmExtraRefusal::kLightTargetCollider
                        : EmExtraRefusal::kSubModel;
      }
      break;
    }

    case Model::kTheoFSGenerator:
      // G4QGSModel<G4GammaParticipants> with G4QGSMFragmentation. Refused by name at the point
      // it would have been needed; see config.cuh's kQgsGammaString.
      r.chose_unported = true;
      r.refusal = EmExtraRefusal::kQgsGammaString;
      break;

    default:
      r.refusal = EmExtraRefusal::kNoModelInRange;
      break;
  }

  r.n_secondaries = fs.n_secondaries;
  if (fs.secondary_overflow > 0) { r.refusal = EmExtraRefusal::kCapacity; }
  return r;
}

}  // namespace g4gpu::physics::hadronic::emextra

#endif
