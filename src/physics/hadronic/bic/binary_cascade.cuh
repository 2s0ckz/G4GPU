// G4BinaryCascade - the entry point, and the one branch of it this package has.
//
// Transcribed from G4BinaryCascade.{hh,cc} (models/binary_cascade) in 11.1.1.
//
// QBBC registers `G4BinaryCascade` for protons, neutrons, pi+ and pi- from 0 to 1.5 GeV
// (`G4HadronInelasticQBBC::ConstructProcess`, four `RegisterMe(theBIC)` calls, `emaxBic =
// 1.5*CLHEP::GeV`, no `SetMinEnergy`) and for nothing else - kaons and hyperons go to
// `G4HadronicBuilder::BuildKaonsFTFP_BERT` and `BuildHyperonsFTFP_BERT`. `bic_params.cuh`
// carries the numbers; `tools/extract_bic_constants.pl` pins them in the physics list's source.
//
// ## The first statement of ApplyYourself decides everything this file can do
//
//     if (initial4Momentum.e()-initial4Momentum.m() < theBCminP &&
//         (definition==G4Neutron::NeutronDefinition() || definition==G4Proton::ProtonDefinition()))
//         return theDeExcitation->ApplyYourself(aTrack, aNucleus);
//
// A NUCLEON below `theBCminP` = 45 MeV never enters the cascade. The whole reaction is
// `G4PreCompoundModel::ApplyYourself` - an (A+1, Z+Zp) fragment with two particle excitons of
// which exactly one is charged and one hole, then `DeExcite` - which is P6's
// `preco::apply_yourself_initial_fragment` plus `preco::deexcite`, both on main and both
// validated. That branch is COMPLETE here.
//
// A PION below 45 MeV does enter the cascade, because the species test is an `&&` and not an
// `||`. That asymmetry is the whole content of the constant, and it is why this file cannot
// answer for a pion at any energy.
//
// ## REFUSED, by name
//
//   * **the cascade proper**, for every projectile at or above 45 MeV and for every pion at any
//     energy: `G4BinaryCascade::Propagate` with `G4CollisionManager`, `G4Scatterer` and the
//     whole `im_r_matrix` collision tree - the cross sections, the angular distributions and the
//     resonance widths. None of it is transcribed in this package. `BicRefusal::cascade` is set
//     at the point it would have been needed and the final state is left empty, because
//     returning the compound-nucleus answer for an energy where Geant4 runs a cascade would be
//     an approximation wearing a result's clothes.
//   * **every projectile that is not a nucleon or a charged pion.** Geant4 throws
//     `G4HadronicException` for one unless the environment variable
//     `I_Am_G4BinaryCascade_Developer` is set; a kernel cannot throw, so it is
//     `BicRefusal::species`.
//   * **the secondaries' creator model id.** `G4PreCompoundModel::ApplyYourself` writes
//     `aNew.SetCreatorModelID(prod->GetCreatorModelID())` - the id of whichever model emitted
//     THAT product, not a blanket one - and P3's `deex::DeexProduct` does not carry a per-product
//     creator id. So `creator_model_id` is -1 here rather than the 24000 of `model_PRECO` or the
//     23100 of `model_G4BinaryCascade`, and `BicReport::creator_ids_unavailable` says so. It is
//     a report and not a refusal: nothing in the port reads the field yet, and inventing a
//     plausible id is exactly what would stop anyone noticing.
//   * **the secondaries' time.** Geant4 sets `timePrimary + max(prod->GetFormationTime(), 0)`.
//     P5's `HadProjectile` carries no global time and P3's product carries no formation time, so
//     both terms are zero here. For a primary at global time zero - which is every call the
//     framework makes today - that is the same number.
#ifndef G4GPU_BIC_BINARY_CASCADE_CUH
#define G4GPU_BIC_BINARY_CASCADE_CUH

#include <cmath>

#include "core/units.cuh"
#include "core/vec3.cuh"
#include "physics/hadronic/bic/bic_params.cuh"
#include "physics/hadronic/bic/light_ion_reaction.cuh"
#include "physics/hadronic/precompound/precompound_model.cuh"
#include "physics/hadronic/process.cuh"

namespace g4gpu::bic {

/// What `apply_yourself` could not do. `any()` is true only for the three that make the final
/// state meaningless; the two reports below it are true on every successful call.
struct BicRefusal {
  /// At or above `theBCminP`, or a pion at any energy: the cascade proper, which this package
  /// does not have. Named at the point it would have been needed.
  bool cascade = false;
  /// Not a proton, a neutron, a pi+ or a pi-: Geant4's `G4HadronicException`.
  bool species = false;
  /// The caller's secondary buffer.
  bool capacity = false;
  /// P6 refused the projectile - it accepts only nucleons, which this file has already checked,
  /// so it can only mean P6's own set changed.
  bool preco_projectile = false;
  int refused_pdg = 0;
  double refused_kin = 0.0;

  __host__ __device__ bool any() const {
    return cascade || species || capacity || preco_projectile;
  }
};

/// The two things every successful call cannot supply, reported rather than invented.
struct BicReport {
  bool creator_ids_unavailable = false;  ///< see the file header
  bool times_unavailable = false;        ///< likewise
};

/// The `HadFinalState` width this model instantiates. A 44 MeV proton on Pb208 makes a compound
/// of A = 209 at 50 MeV of excitation, and P3's evaporation cascade on that emits about seven
/// products; 128 is generous and the overflow is reported, never silent.
inline constexpr int kBicMaxSecondaries = 128;
using BicFinalState = physics::hadronic::HadFinalState<double, kBicMaxSecondaries>;

/// `G4BinaryCascade::ApplyYourself`, with everything past its first statement refused.
///
/// `projectile` and `target` are P5's shapes. Geant4's test is on
/// `initial4Momentum.e() - initial4Momentum.m()`, and for a `G4HadProjectile` that is the
/// kinetic energy exactly: `InitialiseLocal` stores `(0, 0, sqrt(T(T+2m)), T+m)`, whose
/// invariant mass is `m` to the last bit, so `e - m` is `T`. Written as `kin_energy` with this
/// note rather than as a subtraction that reconstructs the same number less accurately.
template <typename Rng>
__host__ __device__ inline preco::PrecoStatus apply_yourself(
    const physics::hadronic::HadProjectile<double>& projectile,
    const physics::hadronic::HadNucleus& target, const data::LevelTable& lt,
    const deex::FermiPool& pool, const preco::PrecoWorkspace& ws, Rng& rng,
    BicFinalState& result, BicRefusal& ref, BicReport& rep) {
  preco::PrecoStatus status;
  result.clear();

  const int pdg = projectile.pdg;
  const bool is_nucleon = (pdg == 2212 || pdg == 2112);
  const bool is_charged_pion = (pdg == 211 || pdg == -211);
  if (!is_nucleon && !is_charged_pion) {
    ref.species = true;
    ref.refused_pdg = pdg;
    ref.refused_kin = projectile.kin_energy;
    return status;
  }

  // `initial4Momentum.e()-initial4Momentum.m() < theBCminP && (neutron || proton)`.
  if (!(projectile.kin_energy < bc_min_p() && is_nucleon)) {
    ref.cascade = true;
    ref.refused_pdg = pdg;
    ref.refused_kin = projectile.kin_energy;
    return status;
  }

  // `return theDeExcitation->ApplyYourself(aTrack, aNucleus)` - P6's, in two calls.
  preco::Excitons ex;
  bool preco_refused = false;
  const deex::Fragment frag = preco::apply_yourself_initial_fragment(
      pdg, projectile.kin_energy, Vec3<double>{0.0, 0.0, 1.0}, target.z, target.a, ex,
      preco_refused);
  if (preco_refused) {
    ref.preco_projectile = true;
    ref.refused_pdg = pdg;
    return status;
  }

  status = preco::deexcite(frag, ex, lt, pool, ws, rng);

  result.status = physics::hadronic::HadFinalStateStatus::kStopAndKill;
  rep.creator_ids_unavailable = true;
  rep.times_unavailable = true;
  for (int i = 0; i < status.n_products; ++i) {
    const deex::DeexProduct& p = ws.products[i];
    physics::hadronic::HadSecondary<double> s;
    s.pdg = p.pdg;
    s.z = p.z;
    s.a = p.a;
    s.time = 0.0;
    s.weight = 1.0;
    s.creator_model_id = -1;   // see the file header
    // The same two-branch mass rule `deex::deex_kinetic_energy` uses and
    // `light_ion_reaction.cuh` writes out: for a nucleus the table mass plus whatever excitation
    // P3 left on it, for A = 0 the electron's mass or nothing.
    const double pdg_mass =
        (p.a > 0) ? (deex::nuclear_mass(p.a, p.z) + p.excitation)
                  : ((p.pdg == deex::kPdgElectron) ? u::electron_mass_c2<double>() : 0.0);
    capture::set_four_momentum(s, p.momentum, pdg_mass);
    if (!result.add_secondary(s)) {
      ref.capacity = true;
      break;
    }
  }
  return status;
}

}  // namespace g4gpu::bic

#endif
