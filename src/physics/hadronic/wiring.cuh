// P8's wiring layer: the decisions a stepper has to make that are not any one model's.
//
// Everything here is about the boundary between a Geant4 process and this transport, and every
// item is a decision that would otherwise be spread over four kernels:
//
//   * WHICH STAGE the run is comparing itself against (`HadronicStage`), and the one thing it
//     changes: whether a stopped negative hadron decays.
//   * WHAT A DISCRETE INTERACTION LENGTH IS in a transport that re-draws it every step rather
//     than carrying `theNumberOfInteractionLengthLeft` on the track (`decay_in_flight_length`,
//     and the note below about why that is the same distribution and not an approximation).
//   * WHAT IS ABSENT, by name and counted (`HadronicRefusal`). Inelastic final states, the
//     stopping processes, anti-nucleus elastic. A refusal here is a hole in the answer with a
//     number attached, not a silent zero.
//
// It owns no physics. The models are P4's (`decay/`), P5's (`process.cuh`, `elastic/`), P7's
// (`capture/`) and P2's cross sections; this file is where a stepper asks them the right
// question in the right order.
//
// WHAT IS ACTUALLY ASKED TODAY IS P4's DECAY, and nothing else. The elastic and capture flags
// below exist and are wired through to the kernels, and the models behind them are transcribed
// and tested, but neither is reached: `step_neutral` is handed a null cross-section table and
// `step_hadron` computes no hadronic interaction length. The reason in each case is a piece of
// DATA that is not on the device rather than a model that is not written, and it is written out
// where the flag is declared so that reading this struct cannot leave the impression that the
// processes are running.
#pragma once

#include <cmath>

#include "core/particle.cuh"
#include "core/units.cuh"
#include "physics/decay/decay.cuh"

namespace g4gpu::had {

// =============================================================================================
// The stage
// =============================================================================================

/// Which Geant4 configuration this run is the like-for-like partner of.
///
/// A RUN-TIME SWITCH AND NOT A COMPILE-TIME ONE, deliberately: the same binary has to produce
/// both columns of the comparison table, so that a difference between them is a difference of
/// configuration and not of build. docs/RISK.md V45 is what a second binary costs.
///
/// The plan's staged method (docs/HADRONIC_PLAN.md section 4) is that Geant4 runs with exactly
/// the processes the port has, inactivated one at a time as the port gains them. Stage 1 is the
/// first such configuration and it is NOT the final physics: it inactivates the three at-rest
/// captures as well as the inelastic processes, because P12 is not written. In THAT
/// configuration Geant4's stopped pi-, K- and mu- decay - `G4Decay` is the only at-rest process
/// they have left - so the port's stage-1 mode decays them too.
///
/// In the final configuration they do not. `G4HadronStoppingProcess::AtRestGetPhysicalInteraction
/// Length` returns 0.0 (G4HadronStoppingProcess.cc:115), which is a pre-emption and not a
/// competitor: no sample of `-log(rand)*tau` can beat zero. So a stopped negative hadron is
/// captured, always, and the port refuses it by name - see `HadronicRefusal::kStoppedPiMinus`
/// and its neighbours. docs/RISK.md V38 is the entry about getting that wrong in the other
/// direction, and P4's `decay_at_rest_competitor` is the hook that made it impossible to forget
/// here.
enum class HadronicStage : int {
  /// Geant4 with `*Inelastic`, `hBertiniCaptureAtRest`, `hFritiofCaptureAtRest` and
  /// `muMinusCaptureAtRest` inactivated and the neutron general process off. A stopped negative
  /// hadron decays on both sides.
  kStage1 = 0,
  /// QBBC as it ships. A stopped negative hadron is captured, which this port does not have, so
  /// it is refused by name and its rest mass is lost from the answer.
  kFinal = 1,
};

__host__ __device__ inline const char* hadronic_stage_name(HadronicStage s) {
  return (s == HadronicStage::kStage1)
             ? "stage1 (QBBC with *Inelastic and the three at-rest captures inactivated, "
               "NeutronGeneralProc off)"
             : "final (QBBC as it ships; stopping is refused by name)";
}

// =============================================================================================
// What is absent
// =============================================================================================

/// A hadronic process this transport reaches and cannot apply. One counter each, per run.
///
/// The list is the plan's Phase-3 packages seen from the wiring side, so every entry names the
/// package that closes it. A refusal is hit at the point the final state would have been
/// applied, which is after the cross section has already decided an interaction happens - so
/// the count is a number of interactions the answer is missing, not a number of chances.
enum class HadronicRefusal : int {
  /// The neutron general process selected `inelastic`. P9/P10/P11.
  ///
  /// AT ZERO TODAY, AND FOR A BLUNTER REASON THAN THE STAGE. `step_neutral` is handed a null
  /// cross-section table - `TransportEngine::Upload` refuses a non-null one until the final
  /// states land with it - so no sub-process of any kind is selected and this counter cannot
  /// move. When the table is uploaded it will be selectable above the capture threshold, and
  /// this is the counter that says how often; the stage-1 configuration is the one in which
  /// the inelastic partial is left out of the sum, matching Geant4 with `neutronInelastic`
  /// inactivated and the general process off.
  kNeutronInelastic = 0,
  /// A charged hadron's inelastic process. Also P9/P10/P11, and also absent from the cross
  /// section rather than present and refused - a charged hadron in this transport has NO
  /// hadronic process at all, so this counter is at zero for the same reason as the one above
  /// and not because nothing happened.
  kChargedHadronInelastic,
  /// A stopped pi- or K-: `G4HadronicAbsorptionBertini`, P12. Reached only in the final stage.
  /// The particle's REST MASS is lost from the answer, which for a pi- is 139.6 MeV.
  kStoppedNegativeHadron,
  /// A stopped mu-: `G4MuonMinusCapture` and the bound-decay sampler inside it
  /// (`G4MuonMinusBoundDecay`), P12. Reached only in the final stage.
  kStoppedMuonMinus,
  /// A stopped antiproton: `G4HadronicAbsorptionFritiof`, P12. Reached only in the final stage,
  /// and worth 2 x 938 MeV of annihilation.
  kStoppedAntiProton,
  /// An antinucleus's elastic scattering: `G4ComponentAntiNuclNuclearXS` (refused by name in
  /// P2's `xs/refusal.cuh`) and `G4AntiNuclElastic` (not started in P5). The antiproton
  /// therefore has NO hadronic process at all in this transport.
  kAntiNucleusElastic,
  /// A decay whose channel P4 refused - `DecayStatus` other than kOK or kStable. The parent is
  /// killed with its energy deposited, which is what Geant4's DECAY101 path does, and the
  /// products are missing.
  kDecayChannel,
  /// A capture cascade that produced more secondaries than the final state can hold.
  kCaptureOverflow,
  kNumHadronicRefusals,
};

__host__ __device__ inline const char* hadronic_refusal_name(HadronicRefusal r) {
  switch (r) {
    case HadronicRefusal::kNeutronInelastic:
      return "neutronInelastic final state (G4BinaryCascade / Bertini / FTFP; P9-P11)";
    case HadronicRefusal::kChargedHadronInelastic:
      return "charged-hadron inelastic final state (P9-P11)";
    case HadronicRefusal::kStoppedNegativeHadron:
      return "stopped pi-/K- capture (G4HadronicAbsorptionBertini; P12) - its rest mass is "
             "lost from the answer";
    case HadronicRefusal::kStoppedMuonMinus:
      return "stopped mu- capture and bound decay (G4MuonMinusCapture / "
             "G4MuonMinusBoundDecay; P12)";
    case HadronicRefusal::kStoppedAntiProton:
      return "stopped antiproton annihilation (G4HadronicAbsorptionFritiof; P12)";
    case HadronicRefusal::kAntiNucleusElastic:
      return "antinucleus elastic (G4ComponentAntiNuclNuclearXS + G4AntiNuclElastic)";
    case HadronicRefusal::kDecayChannel:
      return "a decay channel P4 refused (see DecayStatus)";
    case HadronicRefusal::kCaptureOverflow:
      return "a capture cascade longer than the final state can hold";
    case HadronicRefusal::kNumHadronicRefusals: break;
  }
  return "unknown";
}

/// Where a kernel books a refusal, and the energy it cost.
///
/// TWO NUMBERS PER REFUSAL, and the second is the point. P1's `refused_by_type` counts
/// PARTICLES this port cannot transport, which answers "how many" and not "how much" - and for
/// a hadronic hole the second question is the one a dose comparison needs. A stopped pi- refused
/// here is 139.6 MeV of rest mass that Geant4 puts into a nucleus and this transport puts
/// nowhere; an elastic recoil refused for being a carbon ion is a few hundred keV. Counting
/// only the events would leave the size of the gap to be inferred from a dose that disagrees,
/// which is the failure mode this project keeps writing up.
///
/// Null pointers disable the books; the run then still reports what it does keep.
struct HadronicRefusalBooks {
  int* count = nullptr;      ///< kNumHadronicRefusals ints
  double* energy = nullptr;  ///< kNumHadronicRefusals doubles, MeV
};

/// Books one refusal. `energy` is what the answer is missing because of it: for a stopping
/// process the total energy the particle would have released, for an absent final state the
/// projectile's kinetic energy.
template <typename real_t>
__device__ inline void book_refusal(const HadronicRefusalBooks& books, HadronicRefusal r,
                                    real_t energy) {
  const int i = static_cast<int>(r);
  if (books.count != nullptr) { atomicAdd(&books.count[i], 1); }
  if (books.energy != nullptr) { atomicAdd(&books.energy[i], static_cast<double>(energy)); }
}

// =============================================================================================
// What a stepping kernel is handed
// =============================================================================================

/// Everything the hadronic half of a step needs that is not in the Scene.
///
/// A KERNEL ARGUMENT AND NOT A FIELD ON `Scene`, for the reason P1 gave when it passed the
/// neutron's cross-section table the same way: it keeps the one call site that can turn hadronic
/// physics on visible in `transport_run_impl.cuh` rather than buried in a struct every stepper
/// already has. `physics/scene.cuh` also belongs to no Phase-2 package.
///
/// The three flags are the `/process/inactivate` equivalents for what P8 adds, and they exist
/// for the same reason `ProcessFlags` does: a study switch, not a physics choice. All on is the
/// configuration the like-for-like table is measured in.
template <typename real_t>
struct HadronicWiring {
  HadronicStage stage = HadronicStage::kStage1;
  /// `G4DecayPhysics`. Off means every unstable species is transported to a stop and stays
  /// stopped, which is what this port did before P8. THE ONLY ONE OF THE THREE THAT CHANGES AN
  /// ANSWER TODAY.
  bool decay = true;
  /// `hadElastic`: the elastic sub-process of the neutron general process, and - when a charged
  /// hadron gets one - `hadElastic` on its own process manager.
  ///
  /// READ IN ONE PLACE, `step_neutral`'s cross-section gate, and inert there too because the
  /// table it would gate is null. It is here rather than added later because the flag is the
  /// `/process/inactivate` equivalent and its meaning does not depend on whether the process
  /// exists yet; what it must not do is imply that it does. `step_hadron` does not read it, and
  /// that is the honest state: see the `hadElastic` bullet in that function's header for what
  /// the kernel is missing (the isotope abundances `SampleZandA` needs) and what it costs.
  bool hadron_elastic = true;
  /// `nCapture` - the capture sub-process of the neutron general process. Same state as
  /// `hadron_elastic`: read by the gate, inert because the table is null. P7's model is
  /// written and tested (`tests/test_capture.cu`); what is not written is the upload of P3's
  /// level table, which a cascade on the device would read.
  bool neutron_capture = true;
  HadronicRefusalBooks books{};
};

// =============================================================================================
// The at-rest question
// =============================================================================================

/// Does a stopped particle of this species decay, in this stage?
///
/// pi+, K+, mu+ and pi0 always: they are Coulomb-repelled from every nucleus, have no at-rest
/// process but `G4Decay`, and QBBC gives them none either (`ref/oracle/decay_atrest.csv`).
///
/// pi-, K-, mu- only in stage 1, where Geant4's own at-rest capture is inactivated. In the final
/// stage the answer is no and the caller must book
/// `kStoppedNegativeHadron`/`kStoppedMuonMinus` instead - which is what
/// `stopped_negative_refusal` is for, so that "does it decay" and "what do I do when it does
/// not" cannot drift apart.
///
/// The antiproton decays in NEITHER stage: it is stable, so `G4Decay::IsApplicable` is false for
/// it, and its at-rest process is `G4HadronicAbsorptionFritiof`. A stopped antiproton is always
/// a refusal.
/// The TRITON is why this asks about `stable` as well as about `IsApplicable`. Geant4 gives it
/// `G4Decay` - `IsApplicable` reads the lifetime and 17.774 years is non-negative - and
/// `ref/oracle/decay_atrest.csv` shows the process on its at-rest vector offering
/// 6.4e32 ns. `DecayIt` then returns an untouched particle change, because `GetPDGStable()` is
/// true. So the at-rest branch would be entered and would do nothing; entering it here would
/// draw a uniform for a decay that cannot happen, which moves the triton's random stream for
/// no physics. Asked in the order P4's file asks it: applicable first, stable second.
__host__ __device__ inline bool decay_at_rest_allowed(ParticleType t, HadronicStage stage) {
  const int pdg = pdg_code(t);
  if (!decay::decay_is_applicable(pdg) || decay::particle_is_stable(pdg)) { return false; }
  const decay::AtRestCompetitor c = decay::decay_at_rest_competitor(pdg);
  if (c == decay::AtRestCompetitor::kNone) { return true; }
  return stage == HadronicStage::kStage1;
}

/// The refusal a stopped particle earns when `decay_at_rest_allowed` says no because a stopping
/// process would have run. `kNumHadronicRefusals` when there is nothing to refuse - a stable
/// species with no at-rest process at all, which is every species but the four below.
__host__ __device__ inline HadronicRefusal stopped_refusal(ParticleType t) {
  switch (t) {
    case ParticleType::kPionMinus:
    case ParticleType::kKaonMinus:
      return HadronicRefusal::kStoppedNegativeHadron;
    case ParticleType::kMuonMinus:
      return HadronicRefusal::kStoppedMuonMinus;
    case ParticleType::kAntiProton:
      return HadronicRefusal::kStoppedAntiProton;
    default:
      return HadronicRefusal::kNumHadronicRefusals;
  }
}

/// The energy a refused stopping process would have released: the particle's rest mass plus
/// whatever kinetic energy the transport had not yet spent.
///
/// For a pi- that is 139.6 MeV, most of which Geant4's Bertini absorption gives to the nucleus
/// as nucleons and pions; for an antiproton, 1876 MeV of annihilation. It is the whole of what
/// the answer is missing, and it is a large number - which is why the final stage is not a
/// configuration anyone should quote a dose from until P12 lands, and why the counter says so.
template <typename real_t>
__host__ __device__ inline real_t stopped_refusal_energy(ParticleType t, real_t residual_ekin) {
  const ParticleDef<real_t> pd = particle_def<real_t>(t);
  const real_t m = (pd.mass > real_t(0)) ? pd.mass : real_t(0);
  // The antiproton annihilates against a nucleon, so the released energy is two rest masses.
  const real_t rest = (t == ParticleType::kAntiProton)
                          ? m + units::proton_mass_c2<real_t>()
                          : m;
  return rest + residual_ekin;
}

// =============================================================================================
// The in-flight decay length
// =============================================================================================

/// The distance to a decay in flight, mm, or +inf for a species that does not decay.
///
/// WHY THIS RE-DRAWS EVERY STEP AND GEANT4 DOES NOT, AND WHY THAT IS THE SAME DISTRIBUTION
///
/// `G4Decay::PostStepGetPhysicalInteractionLength` keeps `theNumberOfInteractionLengthLeft` on
/// the process: drawn once per track by `StartTracking`, decremented by
/// `previousStepSize/currentInteractionLength` on every subsequent step, and offered as
/// `n * lambda`. P4 ports both halves (`reset_number_of_interaction_lengths`,
/// `subtract_interaction_lengths`) precisely so a wiring package can carry it.
///
/// This transport does not carry it, for either of the two discrete processes P8 adds, and the
/// reason is that it does not need to. Over one step lambda is a constant - Geant4 evaluates it
/// at the PRE-step energy too, and `currentInteractionLength` is that value - so the probability
/// of firing within a step of length L is `1 - exp(-L/lambda)` whether the remaining count is
/// carried or re-drawn, because the exponential is memoryless. The two are the same
/// distribution, not an approximation of one another, and the position within the step has the
/// same conditional distribution as well. What differs is the random STREAM, which already
/// differs from Geant4's by construction.
///
/// It is also what this port already does for every EM discrete process - `d_delta`, `d_brem`
/// and `d_annih` in `stepper.cuh` are all `-log(rand)/sigma`, re-drawn each step - and B1's
/// 0.09 sigma agreement with Geant4 is measured against that. Carrying a counter would need a
/// field on `TrackState`, which docs/RISK.md V22 names as the one thing the register budget
/// cannot afford ("a fourth kernel instantiation per species is fine, a wider TrackState is
/// not"). Two counters would be 16 bytes on a 236-byte slot.
///
/// The one thing that WOULD break the equivalence is a lambda that varies inside a step, and
/// none does here: every interaction length in this stepper is computed once, from the pre-step
/// energy, before the step is taken.
///
/// THE TWO SENTINELS ARE NOT LENGTHS. `in_flight_mean_free_path` returns DBL_MAX for a stable
/// particle and DBL_MIN for one whose `c*tau` underflows or whose `Ekin/m` does - the second
/// meaning "it decays here, now". Both are mapped to something a smallest-wins competition can
/// use: infinity for the first and zero for the second.
template <typename real_t, typename Rng>
__host__ __device__ inline real_t decay_in_flight_length(ParticleType t, real_t mass,
                                                          real_t kinetic_energy, Rng& rng) {
  const int pdg = pdg_code(t);
  if (!decay::decay_is_applicable(pdg) || decay::particle_is_stable(pdg)) {
    return decay::decay_infinite<real_t>();
  }
  const real_t mfp = decay::in_flight_mean_free_path<real_t>(pdg, mass, kinetic_energy);
  if (!(mfp < decay::decay_infinite<real_t>())) { return decay::decay_infinite<real_t>(); }
  if (mfp <= decay::decay_zero<real_t>()) { return real_t(0); }
  return -log(rng.uniform()) * mfp;
}

/// True when this species draws a decay length at all - and therefore when
/// `decay_in_flight_length` consumes a uniform.
///
/// A PREDICATE AND NOT A COMMENT, because the random stream of every species that does NOT
/// decay has to be unchanged by P8. A proton's B1 dose is this project's headline number and it
/// is checked to a fraction of a sigma; adding one unconditional draw to `step_hadron` would
/// move it, and the move would look like physics.
__host__ __device__ inline bool decays_in_flight(ParticleType t) {
  const int pdg = pdg_code(t);
  return decay::decay_is_applicable(pdg) && !decay::particle_is_stable(pdg);
}

// =============================================================================================
// Emitting what a decay made
// =============================================================================================

/// Pushes a decay's products into the track pool, and reports what could not be pushed.
///
/// `emitter` is `BufferEmitter`, whose `push` already answers the three dispositions - a
/// species with a kernel becomes a track, a neutrino is booked as carrying its energy out, and
/// anything else is counted by name. So this function does the mapping and nothing else: PDG
/// code to `ParticleType`, and a code with no row at all to a refusal rather than to a species
/// that happens to be nearby.
///
/// @return the number of products that became tracks or bookings; the rest are refused.
template <typename real_t, typename Emitter>
__device__ inline int emit_decay_products(const decay::DecayProducts<real_t>& products,
                                          Emitter& emitter,
                                          const HadronicRefusalBooks& books) {
  int emitted = 0;
  for (int i = 0; i < products.n; ++i) {
    const decay::DecayProduct<real_t>& q = products.p[i];
    const ParticleType t = particle_type_of_pdg(q.pdg);
    if (t == ParticleType::kNumTypes) {
      // A PDG code core/particle.cuh has no row for. Not reachable from any table P4
      // transcribed - every daughter of pi+-, pi0, mu+-, K+- and the neutron is in the enum -
      // and counted rather than dropped so that adding a channel without adding a species is
      // loud.
      book_refusal<real_t>(books, HadronicRefusal::kDecayChannel, q.ekin);
      continue;
    }
    emitter.push(t, Vec3<real_t>{q.dir[0], q.dir[1], q.dir[2]}, q.ekin, 0);
    ++emitted;
  }
  return emitted;
}

}  // namespace g4gpu::had
