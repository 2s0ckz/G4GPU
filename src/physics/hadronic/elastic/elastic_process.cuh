// G4HadronElasticProcess::PostStepDoIt - and it is NOT the generic G4HadronicProcess one.
//
// Transcribed from Geant4 11.1.1,
//   processes/hadronic/processes/src/G4HadronElasticProcess.cc
//
// The differences from `G4HadronicProcess::PostStepDoIt` (../process.cuh) are all deliberate and
// all visible in the source. Listed here because "it uses FillResult" is the natural assumption
// and it is wrong:
//
//  1. **It does not call FillResult** for ordinary elastic scattering. It writes the particle
//     change itself, and it decides the primary's fate from the FINAL ENERGY rather than from
//     `G4HadFinalState::GetStatusChange()`: `efinal > 0` -> alive with a rotated direction,
//     `efinal == 0` -> fStopButAlive if the particle has at-rest processes, else fStopAndKill.
//     A model that set `stopAndKill` and left a positive energy would be ignored here and obeyed
//     by FillResult. (FillResult IS used, once, in the diffraction branch - which QBBC never
//     enables, because nothing calls SetDiffraction.)
//
//  2. **It does not call CheckResult**, and the source says why:
//         // Check the result for catastrophic energy non-conservation
//         // cannot be applied because is not guranteed that recoil
//         // nucleus is created
//         // result = CheckResult(theProj, targetNucleus, result);
//     A suppressed recoil is a genuine energy imbalance of up to the recoil threshold, so the
//     check would fire on correct physics. With `epReportLevel` defaulting to 0 as well
//     (../process.cuh), **QBBC's elastic scattering runs with no energy-momentum check at all**.
//     That is the answer to "what does the default ep-check level do here": nothing.
//
//  3. **The recoil threshold is set per step, from the PROTON production cut.**
//         G4double tcut = (*(G4ProductionCutsTable::GetProductionCutsTable()
//                            ->GetEnergyCutsVector(3)))[idx];
//         hadi->SetRecoilEnergyThreshold(tcut);
//     Index 3 of the energy-cuts vector is the proton. `G4RToEConvForProton::Convert` overrides
//     the base class and is a one-liner with no material dependence and no clamping:
//         return (rangeCut/mm) * (100 keV);
//     so with QBBC's default 0.7 mm range cut the recoil threshold is **70 keV in every
//     material**. `proton_recoil_cut_energy` below is that function.
//
//  4. **Only secondary 0 is looked at.** The recoil is `result->GetSecondary(0)`, and if its
//     kinetic energy is at or below tcut its energy is added to the local deposit and the
//     particle is deleted. Any further secondary is neither emitted nor deposited - it is
//     dropped by `result->Clear()`. No elastic model produces more than one, so this is a real
//     but currently unreachable difference; `dropped_secondaries` reports it rather than hiding
//     it.
//
//  5. **Both energy deposits are set to the same value**: `ProposeLocalEnergyDeposit(edep)` and
//     `ProposeNonIonizingEnergyDeposit(edep)`. So a sub-threshold nuclear recoil is counted as
//     entirely NON-IONIZING - which is what makes elastic scattering contribute to NIEL and not
//     to dose in a scorer that separates them.
//
//  6. `theNumberOfInteractionLengthLeft = -1.0` is set FIRST, with the comment "For elastic
//     scattering, _any_ result is considered an interaction" - so even the integral-cross-section
//     rejection, or a zero-energy track, still costs a fresh interaction length.
//
//  7. A track with zero kinetic energy or a status other than fAlive returns immediately, before
//     the cross section is recomputed and before any random number is drawn.
#pragma once

#include <cmath>

#include "core/units.cuh"
#include "core/vec3.cuh"
#include "physics/hadronic/elastic/hadron_elastic.cuh"
#include "physics/hadronic/process.cuh"

namespace g4gpu::physics::hadronic::elastic {

/// G4RToEConvForProton::Convert - the proton production threshold from a range cut.
///
/// `(rangeCut/mm) * 100 keV`, with the comment "Simple formula - range = Ekin/(100*keV)*(1*mm)".
/// It overrides G4VRangeToEnergyConverter::Convert entirely, so none of the base class's energy
/// grid, iteration or [1 keV, 10 GeV] clamping applies: a 0 mm cut gives 0 and a 1 m cut gives
/// 100 MeV. There is no material argument. QBBC's default range cut is 0.7 mm -> 70 keV.
template <typename real_t>
__host__ __device__ real_t proton_recoil_cut_energy(real_t range_cut_mm) {
  return (range_cut_mm / units::mm<real_t>()) * (real_t(100) * units::keV<real_t>());
}

/// What one call of the elastic process produced.
template <typename real_t>
struct ElasticStepResult {
  bool interacted = false;             ///< false = the track is returned unchanged
  TrackStatusChange status = TrackStatusChange::kAlive;
  real_t energy = real_t(0);           ///< the primary's new kinetic energy
  Vec3<real_t> momentum_direction{real_t(0), real_t(0), real_t(1)};
  real_t local_energy_deposit = real_t(0);
  real_t non_ionizing_energy_deposit = real_t(0);
  real_t weight = real_t(1);

  int n_secondaries = 0;               ///< 0 or 1
  HadSecondary<real_t> recoil;
  int dropped_secondaries = 0;         ///< see note 4 in the header

  /// Diagnostics, so a test can assert on the path taken rather than only on the numbers.
  bool rejected_by_integral_xs = false;
  bool recoil_below_threshold = false;
  real_t recoil_threshold = real_t(0);
};

/// G4HadronElasticProcess::PostStepDoIt, without the diffraction branch.
///
/// The diffraction branch is omitted rather than stubbed: `fDiffraction` is null unless
/// `SetDiffraction(hi, xsr)` was called, and no constructor in QBBC's chain calls it - only
/// G4ChargeExchangePhysics, which QBBC does not register. If a caller ever needs it, the branch
/// is: sample the ratio from a G4VCrossSectionRatio, and on success run the diffraction model,
/// `CheckResult` it, and `FillResult` it, with the ep report if the level is non-zero. That path
/// is the only one in this process that uses the generic machinery.
///
/// `apply` is the model, called as
///   `apply(projectile, target, recoil_threshold, rng, &final_state)`,
/// which for every elastic model is `hadron_elastic_apply_yourself` with a different sampler.
///
/// `xs_now` and `xs_at_step_start` implement the integral-cross-section rejection; pass
/// `HadXsType::kNoIntegral` to skip it, which is what a neutron does (`hadronic_xs_type` returns
/// kNoIntegral for a neutral particle) and therefore what the elastic sub-process of
/// G4NeutronGeneralProcess does.
template <typename real_t, int kCap, typename ApplyFn, typename Rng>
__host__ __device__ ElasticStepResult<real_t> elastic_post_step_do_it(
    const HadProjectile<real_t>& projectile, const HadNucleus& target,
    const Vec3<real_t>& track_direction, real_t track_weight, bool track_is_alive,
    bool has_at_rest_processes, HadXsType xs_type, real_t xs_now, real_t xs_at_step_start,
    real_t recoil_threshold, const ApplyFn& apply, Rng& rng,
    HadFinalState<real_t, kCap>* scratch) {
  ElasticStepResult<real_t> out;
  out.weight = track_weight;
  out.energy = projectile.kin_energy;
  out.momentum_direction = track_direction;
  out.recoil_threshold = recoil_threshold;

  // theNumberOfInteractionLengthLeft = -1 happens here, before every early return: for elastic
  // scattering any outcome, including "no interaction", consumes the interaction length.
  if (projectile.kin_energy == real_t(0) || !track_is_alive) { return out; }

  if (integral_xs_rejects<real_t>(xs_type, xs_now, xs_at_step_start, rng)) {
    out.rejected_by_integral_xs = true;
    return out;
  }

  apply(projectile, target, recoil_threshold, rng, scratch);
  out.interacted = true;

  real_t edep = (scratch->local_energy_deposit > real_t(0)) ? scratch->local_energy_deposit
                                                            : real_t(0);
  const real_t efinal = (scratch->energy_change > real_t(0)) ? scratch->energy_change
                                                             : real_t(0);
  out.energy = efinal;
  if (efinal > real_t(0)) {
    out.status = TrackStatusChange::kAlive;
    out.momentum_direction = rotate_uz(scratch->momentum_change, track_direction);
  } else {
    out.status = has_at_rest_processes ? TrackStatusChange::kStopButAlive
                                      : TrackStatusChange::kStopAndKill;
  }

  if (scratch->n_secondaries > 0) {
    HadSecondary<real_t> p = scratch->secondaries[0];
    if (p.kin_energy > recoil_threshold) {
      p.direction = rotate_uz(p.direction, track_direction);
      // "in elastic scattering time and weight are not changed": the recoil track takes the
      // parent's global time and weight, and NOT `max(secTime,0)+globalTime` as FillResult does.
      p.weight = track_weight;
      out.recoil = p;
      out.n_secondaries = 1;
    } else {
      out.recoil_below_threshold = true;
      edep += p.kin_energy;
    }
    out.dropped_secondaries = scratch->n_secondaries - 1;
  }
  out.local_energy_deposit = edep;
  out.non_ionizing_energy_deposit = edep;
  return out;
}

}  // namespace g4gpu::physics::hadronic::elastic
