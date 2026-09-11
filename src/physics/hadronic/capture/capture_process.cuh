// `nCapture` - G4NeutronCaptureProcess, which is the generic G4HadronicProcess and nothing else.
//
// Transcribed from Geant4 11.1.1,
//   processes/hadronic/processes/src/G4NeutronCaptureProcess.cc
//     the whole class: a constructor that is `G4HadronicProcess(name, fCapture)` plus
//     `AddDataSet(new G4NeutronCaptureXS())`, an IsApplicable that is `== G4Neutron::Neutron()`,
//     and a ProcessDescription. There is no PostStepDoIt, no PostStepGetPhysicalInteractionLength
//     and no BuildPhysicsTable of its own.
//   physics_lists/util/src/G4HadProcesses.cc
//     BuildNeutronInelasticAndCapture - `nCap->RegisterMe(new G4NeutronRadCapture())`, and then
//     either `nGen->SetCaptureProcess(nCap)` or `ph->RegisterProcess(nCap, neutron)`.
//   processes/hadronic/management/src/G4HadronicProcess.cc
//     PostStepDoIt, UpdateCrossSectionAndMFP, DefineXSandMFP - ../process.cuh has FillResult and
//     CheckResult; this file is the sequence they sit in.
//
// SO WHY A FILE AT ALL
//
// Because the process is a COMPOSITION and the composition is where a port goes wrong. Six
// things happen in an order, and three of them are decided elsewhere:
//
//   1. `fXSType` is `fHadNoIntegral`, because `hadronic_xs_type` refuses the integral approach
//      for a neutral particle (`charge != 0.0` fails). So there is no integral rejection at the
//      top of PostStepDoIt and NO random number is drawn there - which is also what makes this
//      process usable as a sub-process of G4NeutronGeneralProcess, whose own table has already
//      decided that an interaction happens.
//   2. The cross section is EVALUATED, not tabulated. `DefineXSandMFP` asks the data store,
//      which caches on (material, particle, energy) and otherwise recomputes - so the partial
//      per-element sums `SampleZandA` reads are always the ones for this step's energy. See
//      docs/PORTED.md 2.1.1: G4HadronicProcess holds no G4PhysicsVector of its own.
//   3. `SampleZandA` needs those partial sums. As a stand-alone process it has them from step 2;
//      as a sub-process it has them because `G4NeutronGeneralProcess::PostStepDoIt` calls
//      `fCurrentXSS->ComputeCrossSection` before delegating - AND ONLY WHEN THE MATERIAL HAS
//      MORE THAN ONE ELEMENT, because with one element `SampleZandA`'s element loop is skipped
//      and draws no random number. Both paths land in the same function here, with the caller
//      saying which.
//   4. One model, `G4NeutronRadCapture`, registered over [0, 100 TeV]. With exactly one
//      registered model `G4EnergyRangeManager::GetHadronicInteraction` returns it without
//      looking at any energy range, so the window is inert - and `choose_hadronic_interaction`
//      in ../process.cuh takes the same shortcut.
//   5. `CheckResult` IS called - unconditionally, by the generic PostStepDoIt, unlike
//      `G4HadronElasticProcess::PostStepDoIt` which skips it. Its levels are (2%, 1 GeV) and
//      BOTH must be exceeded, so for a capture releasing 2 to 9 MeV it cannot fire; it is wired
//      anyway, with the resample loop and its 100-attempt limit, because the arithmetic is the
//      definition of what this process conserves and a test can then assert on it.
//   6. `FillResult`. The primary is dead (`SetStatusChange(stopAndKill)` is the second line of
//      ApplyYourself), so the branch order in FillResult that matters for a stopped hadron -
//      docs/RISK.md V37 - is not reached: stopAndKill wins first.
//
// WHAT `epReportLevel` DOES: nothing. It is 0 by default and QBBC never sets it, so
// CheckEnergyMomentumConservation never runs. ../process.cuh has the arithmetic; this process
// does not call it, and that is the transcription rather than an omission.
#pragma once

#include <cmath>

#include "core/units.cuh"
#include "physics/hadronic/capture/neutron_rad_capture.cuh"
#include "physics/hadronic/process.cuh"
#include "physics/hadronic/xs/particlexs.cuh"
#include "physics/hadronic/xs/sample_za.cuh"

namespace g4gpu::physics::hadronic::capture {

/// P2's cross sections live in `g4gpu::hadronic::xs`, which is NOT reachable as `hadronic::xs`
/// from inside `g4gpu::physics::hadronic::capture`: unqualified lookup of `hadronic` finds the
/// enclosing `g4gpu::physics::hadronic` first and stops there. The alias says which one, once.
namespace xs = g4gpu::hadronic::xs;

/// G4NeutronCaptureProcess::IsApplicable - `&aParticleType == G4Neutron::Neutron()`, and the
/// process is `final`, so nothing widens it. A PDG code and not a ParticleType because
/// core/particle.cuh belongs to another package and this predicate is about Geant4's answer.
__host__ __device__ inline bool capture_is_applicable(int pdg) { return pdg == 2112; }

/// `G4HadronicProcess(processName, fCapture)`: G4HadronicProcessType's `fCapture` is 12.
/// Written down because the enum is easy to miscount - docs/RISK.md V43 records `fNeutronGeneral`
/// being read as 161 when it is 116 - and because the sub-type is what a process dump prints.
inline constexpr int kCaptureProcessSubType = 12;

/// `SetMinEnergy(0.0)` / `SetMaxEnergy(G4HadronicParameters::GetMaxEnergy())` on
/// G4NeutronRadCapture, i.e. the model's own window. Inert with one registered model (see the
/// header), carried so that a second capture model - there is none in QBBC - would have
/// something to overlap with.
template <typename real_t> __host__ __device__ constexpr real_t capture_model_min_energy() {
  return real_t(0);
}
template <typename real_t> __host__ __device__ constexpr real_t capture_model_max_energy() {
  return real_t(1e8) * units::MeV<real_t>();  // 100 TeV
}

/// One nCapture PostStepDoIt, plus the diagnostics a test wants to assert on.
template <typename real_t, int kSecCap>
struct CaptureStepResult {
  HadronicStepResult<real_t, kSecCap> change;
  CaptureInfo<real_t> info;
  /// The (Z, A) SampleZandA drew.
  int target_z = 0;
  int target_a = 0;
  /// CheckResult's verdict on the accepted attempt, and how many attempts it took.
  CheckResultVerdict verdict = CheckResultVerdict::kAccept;
  int attempts = 0;
  real_t delta_e = real_t(0);
  /// CheckResult rejected 100 attempts in a row, where Geant4 raises FatalException "had006".
  bool refused_resample_limit = false;
};

/// `G4HadronicProcess::PostStepDoIt` for nCapture, with the target already chosen.
///
/// Split from the sampling deliberately: the target draw needs the material and the whole
/// cross-section data set, and the final state needs neither. A test that wants to check the
/// mass balance on Fe56 says Fe56.
///
/// @param has_at_rest_processes the neutron's answer to
///        `GetAtRestProcessVector()->size() > 0`. In QBBC 11.1.1 the neutron's at-rest vector
///        is EMPTY - `ref/oracle/neutron_processes.csv` gives it Transportation, Decay and
///        NeutronGeneralProc, and G4Decay is a rest-discrete process whose at-rest vector entry
///        exists, so this is the caller's measurement and not a constant here. It is only read
///        when the model leaves a positive-energy primary, which capture never does.
template <typename real_t, int kCap, int kSecCap, typename Rng>
__host__ __device__ inline CaptureStepResult<real_t, kSecCap> capture_final_state(
    const HadProjectile<real_t>& projectile, const HadNucleus& target,
    const Vec3<real_t>& track_direction, real_t track_global_time, real_t track_weight,
    bool has_at_rest_processes, const data::LevelTable& levels, Rng& rng,
    HadFinalState<real_t, kCap>* scratch,
    deex::PhotonEvaporationState* persistent = nullptr) {
  CaptureStepResult<real_t, kSecCap> out;
  out.target_z = target.z;
  out.target_a = target.a;

  // The resample loop of the generic PostStepDoIt: `do { ApplyYourself; CheckResult } while
  // (!result)`, with a FatalException after 100 attempts. Reproduced with the count reported
  // instead of thrown, because a kernel cannot throw.
  const int kMaxAttempts = 100;
  real_t pdg_mass[kCap];
  for (int attempt = 0; attempt < kMaxAttempts; ++attempt) {
    out.info = neutron_rad_capture_apply<real_t, kCap>(projectile, target, levels, rng, scratch,
                                                       persistent);
    ++out.attempts;
    // The PDG mass of each secondary, which is what CheckResult and FillResult compare the
    // DYNAMICAL mass in HadSecondary::mass against. The two are not the same number on the
    // A <= 4 branch: there Geant4 builds both secondaries with the four-momentum constructor,
    // whose `|PDGmass^2 - mass2| > EnergyMRA2` test is satisfied by one ulp of a nucleus's
    // mass squared, so the dynamical mass is `sqrt(t^2 - |p|^2)` and not the definition's.
    // They differ by an ulp, far below FillResult's 1 keV shell tolerance, so nothing is
    // corrected - which is a claim tests/test_capture.cu measures rather than a comment.
    for (int i = 0; i < scratch->n_secondaries; ++i) {
      pdg_mass[i] = static_cast<real_t>(
          capture_secondary_pdg_mass<real_t>(scratch->secondaries[i],
                                             static_cast<double>(out.info.residual_excitation)));
    }
    real_t de = real_t(0);
    out.verdict = check_result<real_t, kCap>(
        projectile, static_cast<real_t>(deex::nuclear_mass(target.a, target.z)), *scratch,
        FatalEnergyCheckLevels<real_t>{}, pdg_mass, &de);
    out.delta_e = de;
    if (out.verdict == CheckResultVerdict::kAccept) { break; }
    if (attempt + 1 == kMaxAttempts) { out.refused_resample_limit = true; }
  }

  out.change = fill_result<real_t, kCap, kSecCap>(*scratch, track_direction, track_global_time,
                                                  track_weight, has_at_rest_processes, pdg_mass);
  return out;
}

/// The whole stand-alone process: the cross section, the target draw and the final state.
///
/// This is the stage-1 configuration - `EnableNeutronGeneralProcess` off, so `nCapture` is a
/// process of its own on the neutron's process manager with its own interaction length. As a
/// sub-process of G4NeutronGeneralProcess the interaction length is the general table's and the
/// caller passes the partial sums it already computed; see `capture_sample_target`.
///
/// @param mxs must have been filled by `store_compute_cross_section` at THIS energy and in THIS
///        material. Passing a stale one selects an element by another energy's cross sections
///        and nothing complains - which is why it is an argument and not a member.
/// @param q_elm,q_iso the two uniforms SampleZandA draws, in its order: the element first (and
///        only when the material has more than one element), the isotope second (and only when
///        the element has more than one isotope). A caller that draws them unconditionally
///        desynchronises the stream from Geant4's; `xs::store_sample_za_fn` takes them
///        as values for exactly that reason and this function passes them through.
template <typename real_t>
__host__ __device__ inline xs::TargetZA capture_sample_target(
    const xs::PxsDataSet<real_t>& capture_xs, real_t ekin, real_t loge,
    const data::Material<real_t>& material,
    const xs::ElementIsotopes<real_t>* isotopes,
    const xs::MaterialXs<real_t>& mxs, real_t q_elm, real_t q_iso) {
  return xs::store_sample_za<real_t>(capture_xs, ekin, loge, material, isotopes, mxs,
                                               q_elm, q_iso);
}

/// The macroscopic capture cross section of a material, 1/mm, and the per-element partial sums
/// SampleZandA will need. `G4CrossSectionDataStore::ComputeCrossSection` with
/// G4NeutronCaptureXS as the one registered data set.
template <typename real_t>
__host__ __device__ inline xs::XsValue<real_t> capture_material_xs(
    const xs::PxsDataSet<real_t>& capture_xs, real_t ekin, real_t loge,
    const data::Material<real_t>& material,
    const xs::ElementIsotopes<real_t>* isotopes,
    xs::MaterialXs<real_t>& mxs) {
  return xs::store_compute_cross_section<real_t>(capture_xs, ekin, loge, material,
                                                           isotopes, mxs);
}

}  // namespace g4gpu::physics::hadronic::capture
