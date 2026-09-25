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
// WHAT IS ACTUALLY ASKED TODAY IS ALL THREE OF THEM. That paragraph used to read "P4's decay,
// and nothing else", with the elastic and capture flags wired through to the kernels and neither
// reached - `step_neutral` handed a null cross-section table and `step_hadron` computing no
// hadronic interaction length - because in each case a piece of DATA was not on the device.
// P8b closed the first (the isotope abundances) and P8d the second (P2's five combined tables
// and the two per-process ones, `host/neutron_upload.cuh`), so every flag in `HadronicWiring`
// now changes an answer and the notes below say for which species.
#pragma once

#include <cmath>

#include "core/particle.cuh"
#include "core/units.cuh"
#include "data/level_data.cuh"
#include "physics/decay/decay.cuh"
#include "physics/hadronic/elastic_wiring.cuh"
#include "physics/hadronic/interaction_queue.cuh"
#include "physics/hadronic/neutron_wiring.cuh"

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
/// AND FOR THE NEUTRON THE STAGE IS NOT A SUBSET, IT IS A DIFFERENT PROCESS LIST. P8d found
/// this where P8b and P8c had left it as "the inelastic partial is left out of the sum".
/// `EnableNeutronGeneralProcess` is set in `G4HadronInelasticQBBC`'s CONSTRUCTOR (not, as V53
/// and docs/PORTED.md 2.1.2 both say, in its `ConstructProcess`), and there is no UI command for
/// it anywhere in 11.1.1 - `G4HadronicParametersMessenger` builds exactly three commands, and
/// `enableNeutronGeneralProcess` is not one of them. So the only stage-1 reference that exists
/// is a Geant4 whose flag is turned off in C++ before `G4State_PreInit` ends, after which
/// `neutronInelastic` IS a process on the neutron's manager and `/process/inactivate` reaches
/// it. In that configuration the neutron has separate `hadElastic` and `nCapture` processes,
/// each with its own cross-section data store and its own interaction length, and a real
/// `G4NeutronKiller` carrying the 10 us cut - not one table with a term removed.
enum class HadronicStage : int {
  /// Geant4 with `*Inelastic`, `hBertiniCaptureAtRest`, `hFritiofCaptureAtRest` and
  /// `muMinusCaptureAtRest` inactivated. A stopped negative hadron decays on both sides. The
  /// NEUTRON's half of it is `EnableNeutronGeneralProcess(false)` plus
  /// `/process/inactivate neutronInelastic`, so its elastic and capture run as two competing
  /// processes on their own tables - which is what `step_neutral` does in this stage.
  kStage1 = 0,
  /// QBBC as it ships. A stopped negative hadron is captured, which this port does not have, so
  /// it is refused by name and its rest mass is lost from the answer; and the neutron's one
  /// interaction length comes from `G4NeutronGeneralProcess`'s combined table, whose inelastic
  /// sub-process is refused by name (`HadronicRefusal::kNeutronInelastic`) until P9-P11.
  kFinal = 1,
};

__host__ __device__ inline const char* hadronic_stage_name(HadronicStage s) {
  return (s == HadronicStage::kStage1)
             ? "stage1 (QBBC with *Inelastic and the three at-rest captures inactivated; the "
               "neutron with EnableNeutronGeneralProcess false and neutronInelastic off, i.e. "
               "hadElastic and nCapture as separate processes)"
             : "final (QBBC as it ships; stopping and neutronInelastic are refused by name)";
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
  /// A neutron's inelastic interaction that produced no final state. P15 wired the process;
  /// this counts what is still missing from it, whatever the cause.
  ///
  /// **THE TWO GROUPS IN THIS ENUM MUST NOT BE ADDED TOGETHER.** Since P15 the ledger answers
  /// two different questions and answers both about the same event:
  ///
  ///   * `kNeutronInelastic` and `kChargedHadronInelastic` say **how much is missing**, one
  ///     booking per lost interaction with the projectile's kinetic energy on it. Summing these
  ///     two gives the size of the inelastic hole in the answer.
  ///   * `kLightIonCascade`, `kBinaryHydrogenTarget`, `kFtfpRefused`, `kBertiniRefused`,
  ///     `kBinaryRefused`, `kNoInelasticModel`, `kInelasticSecondaryOverflow`,
  ///     `kInelasticQueueFull`, `kInteractionNoSlot` and `kInelasticReentryExhausted` say
  ///     **why**, and every one of them is booked on an event that is ALSO in the first group.
  ///
  /// Two groups rather than one deep enum because the questions have different consumers: a
  /// dose comparison wants the first and a package triage wants the second, and a single
  /// counter that tried to be both would have to be read with a key. The run's report prints
  /// them under separate headings for exactly that reason.
  ///
  /// REACHED IN BOTH STAGES SINCE P15, and it used to be reached only in `kFinal`.
  /// `G4NeutronGeneralProcess::BuildPhysicsTable` sums elastic + inelastic + capture
  /// unconditionally, so in the final configuration the interaction length includes the
  /// inelastic term, `select()` can name it, and this counter says how often with what energy
  /// on it. The neutron is then killed with its kinetic energy deposited locally, which is the
  /// conservative disposal and is NOT what Geant4 does with it.
  ///
  /// In `kStage1` it stays at zero, and not by omitting a term from a sum: the stage-1
  /// reference is a Geant4 whose `EnableNeutronGeneralProcess` is false and whose
  /// `neutronInelastic` is then inactivated by name, so the neutron carries separate
  /// `hadElastic` and `nCapture` processes on their OWN data stores and there is no inelastic
  /// process to select. docs/RISK.md V53's addendum is why "left out of the total" was the
  /// wrong description of it.
  kNeutronInelastic = 0,
  /// A charged hadron's inelastic interaction that produced no final state - the "how much"
  /// counter of the other species, and the other half of the note above.
  ///
  /// REACHED SINCE P15. This used to read "absent from the cross section rather than present
  /// and refused - a charged hadron in this transport has NO hadronic process at all, so this
  /// counter is at zero". It has one now: `had::inelastic_xs_per_volume` is a fourth discrete
  /// competitor in `step_hadron` for p, pi+-, K+-, d, t, He3, alpha and GenericIon, and what is
  /// left is the interactions the models refuse. The antiproton is still structurally zero and
  /// is `kAntiNucleusInelastic` below, which says why.
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
  ///
  /// STRUCTURALLY ZERO, and it stays that way now that every other charged hadron has an
  /// elastic process. `had::elastic_channel(kAntiProton)` is `kAntiNucleusRefused` and
  /// `elastic_xs_per_volume` returns zero for it, so an antiproton draws no hadronic
  /// interaction length at all - the gap is in the CROSS SECTION and not in a final state that
  /// could not be applied, and this ledger counts interactions the answer is missing rather
  /// than steps that had no chance of one. Booking it per step would count chances. The size of
  /// the gap is not measurable from this port; it is the difference between Geant4's own two
  /// pbar columns, which `ref/b1hadron/` does not run because P1 transports no pbar primary in
  /// the stage-1 set.
  kAntiNucleusElastic,
  /// A decay whose channel P4 refused - `DecayStatus` other than kOK or kStable. The parent is
  /// killed with its energy deposited, which is what Geant4's DECAY101 path does, and the
  /// products are missing.
  kDecayChannel,
  /// A capture cascade that produced more secondaries than the final state can hold.
  ///
  /// A TRIPWIRE ON A MEASURED NUMBER, since P8d reached it. The capacity is 32
  /// (`had::kNeutronCaptureSecondaryCap`), which is what `tests/test_capture_device.cu` ran 448
  /// captures at and what P7's statistical oracle ran 20,000 per (target, energy) at. If this
  /// moves, a cascade got longer than anything either of those saw and the gammas past the
  /// thirty-second are missing from the answer.
  kCaptureOverflow,
  /// A capture secondary whose PDG code `core/particle.cuh` has no row for.
  ///
  /// Unreachable from a radiative capture, whose every secondary is a gamma, an
  /// internal-conversion electron or a nucleus - and counted rather than dropped so that a
  /// cascade which starts emitting something else is loud. The same shape as
  /// `kDecayChannel`'s unmapped-daughter arm and for the same reason.
  kCaptureSecondarySpecies,
  /// A capture the model could not complete: `CaptureRefusal::kUnphysicalTarget` (a target with
  /// Z > A, A < 1 or Z < 1, for which `G4NucleiProperties::GetNuclearMass` is zero) or
  /// `CheckResult` rejecting a hundred attempts in a row, where Geant4 raises its `had006`
  /// FatalException. Both are tripwires: `SampleZandA` draws from `data/isotope_abundance.hh`,
  /// whose isotopes are all physical, and the check's levels are (2%, 1 GeV) with BOTH required,
  /// so a capture releasing 2 to 9 MeV cannot fail it.
  ///
  /// `CaptureRefusal::kIsomerIonMass` is deliberately NOT booked here. It fires on 101 of the
  /// 600 points of P7's deterministic oracle and it is not a missing interaction: it says the
  /// residual's excitation went through `G4IonTable::GetIon`'s snapping, whose MASS this port
  /// reproduces to the last bit for all 568 residuals and whose run-dependent isomer DIGIT it
  /// does not carry. Booking it would fill this ledger with a fact about a PDG code.
  kCaptureRefused,
  /// An elastic final state with a secondary beyond the first.
  ///
  /// A TRIPWIRE, NOT AN EXPECTED COUNT. `G4HadronElasticProcess::PostStepDoIt` looks at
  /// `GetSecondary(0)` and drops the rest with `result->Clear()` (note 4 of
  /// `elastic/elastic_process.cuh`), and no elastic model in QBBC emits more than one - which
  /// is why `step_hadron` gives the final state a capacity of ONE, 56 bytes of kernel stack
  /// against the package default's 450. If this counter ever moves, a model started emitting
  /// something this transport is throwing away, and it says so instead of the stack quietly
  /// being too small.
  kElasticDropped,
  /// A `kGenericIon` track with no nuclide, or with one outside AME2012.
  ///
  /// `data::nuclear_mass` answers from the particle table for the six light nuclei and from
  /// G4NucleiPropertiesTableAME12 for everything else, and refuses `G4NucleiProperties`'
  /// remaining two levels - the Moller-Nix theoretical table and the Cameron/Weizsaecker
  /// formula - by returning zero (data/nuclei_mass_ame12.hh). Nothing a QBBC elastic recoil can
  /// be reaches them: the target is one of Geant4's elements and its isotopes are the naturally
  /// occurring ones, all of which AME2012 carries. So this counter is the tripwire on that
  /// claim, and it is also where a primary ion would land if one ever got past
  /// `G4RunManager::CheckSpecies`. The track is killed with its energy deposited locally, which
  /// is the conservative disposal for a particle whose mass is not known.
  kIonWithoutNuclide,
  /// A real nucleus whose delta-ray transfer window is open.
  ///
  /// ONE PER STEP, NOT ONE PER INTERACTION, which is the opposite of every other entry here and
  /// is why it says so. What is missing is the whole delta-ray channel of an ion, on every step
  /// of it, rather than one final state that could not be applied: `em::hadron_delta_xs` and
  /// `em::sample_hadron_delta` take a SPECIES and would compute G4GenericIon's placeholder
  /// (938.2723 MeV, charge 1) for an oxygen recoil, and the projectile form factor needs the
  /// mass NUMBER as well (`G4NistManager::GetA27`), so the channel is refused rather than
  /// sampled with the wrong particle.
  ///
  /// AT ZERO FOR EVERYTHING THIS PORT CAN PRODUCE, and the threshold is arithmetic rather than
  /// a hope: an ion's window is `tmax > cut` with `tmax = 2 m_e b2g2/(1 + 2 gamma m_e/M +
  /// (m_e/M)^2)`, so water's 350 keV electron cut needs `beta^2 gamma^2 > 342`, i.e. an ion
  /// above about 17 GeV per nucleon. `step_hadron` tests it with the ion's OWN definition,
  /// which is exact.
  kIonDeltaRay,

  // -------------------------------------------------------------------------------------------
  // P15's second group: WHY an inelastic interaction produced no final state. Every booking
  // here is on an event that is ALSO booked under `kNeutronInelastic` or
  // `kChargedHadronInelastic`, so the two groups are not summed - see the note on
  // `kNeutronInelastic`.
  // -------------------------------------------------------------------------------------------

  /// `G4BinaryLightIonReaction`'s CASCADE arm refused - `G4BinaryCascade::Propagate`, inside
  /// `Interact`, could not finish; `bic::BlirRefusal::cascade_ref` says which of its refusals.
  ///
  /// **UNTIL P9e THIS WAS THE WHOLE CASCADE ARM, AND THE LARGEST NAMED HOLE P15 HAD.** Every ion
  /// at or above 50 MeV per nucleon - the `(mom.t()-mom.mag())/pA < 50*MeV` test at
  /// G4BinaryLightIonReaction.cc:119 - was booked here, because `Interact` did not exist: 10,266
  /// of 11,223 queued interactions of an 840 MeV alpha beam (91.5%), and 914 of the 1,021
  /// refusals over `tests/test_inelastic_transport.cu`'s grid. It is kept as the NAME of what is
  /// left rather than retired, so that a ledger read across the two builds shows the rate
  /// falling instead of a line disappearing: the same grid books 49 here with `Interact` wired.
  /// The five ion processes came off the inactivation list of every like-for-like column with it
  /// (docs/RISK.md V192, V198).
  kLightIonCascade,
  /// `G4BinaryCascade::Propagate1H1` - a nucleon or charged pion on a HYDROGEN target, which P9
  /// refused by name. Small but not zero in water: hydrogen is 2 of every 3 atoms and about
  /// 11% of the electrons, though its inelastic cross section is the smallest of the two.
  kBinaryHydrogenTarget,
  /// `ftf::entry::apply` came back `kRefused`, with a reason of its own that `entry::Report`
  /// carries. The FTFP-side refusals are P11's and are listed in docs/PORTED.md 2.1.11b.
  kFtfpRefused,
  /// `bert::apply_yourself` came back with an `InterfaceRefusal`, or with `no_interaction`.
  /// P10's, and listed in docs/PORTED.md 2.1.12 - `kFate`, K0S/K0L, hyper-nuclei.
  kBertiniRefused,
  /// `bic::apply_yourself` or `bic::blir_apply_yourself` refused for a reason that is not the
  /// hydrogen target or the cascade arm: an inapplicable species, a PreCompound projectile
  /// refusal, or one of the two capacity guards. A TRIPWIRE - this wiring sends the Binary
  /// cascade only nucleons and charged pions and the light-ion reaction only ions, so a
  /// species refusal here means the model table and `inelastic_models` have drifted apart.
  kBinaryRefused,
  /// `G4EnergyRangeManager::GetHadronicInteraction` found no model covering the energy, or more
  /// than two competing, or two fully nested. Geant4 prints its model table and returns
  /// nullptr, and `G4HadronicProcess::PostStepDoIt` then raises the `had005` FatalException; a
  /// kernel cannot throw, so it is carried out by name.
  ///
  /// A TRIPWIRE AT ZERO for every species this port transports, and the arithmetic is in
  /// `inelastic_models`: the three-model lists have no gap (BIC to 1.5 GeV, Bertini from 1) and
  /// no triple overlap (BIC ends at 1.5 and FTFP starts at 3), and the two-model lists cover
  /// [0, 100 TeV] between them. If it ever moves, a window was transcribed wrong.
  kNoInelasticModel,
  /// A final state with more secondaries than `kInteractionSecondaryCap` (256). A TRIPWIRE on a
  /// measured number: P10's campaign of 190,000 Bertini events and P11d's 36,000 ion events at
  /// a 512-track list both fit inside it, and P12b's 1,000,000 at-rest captures ran at exactly
  /// this capacity. If it moves, a model started emitting more than anything either campaign
  /// saw and the secondaries past the 256th are missing from the answer.
  kInelasticSecondaryOverflow,
  /// An inelastic secondary whose PDG code `core/particle.cuh` has no row for.
  ///
  /// NOT A TRIPWIRE and not expected to be zero: an inelastic model at a few GeV emits
  /// hyperons, K0S/K0L and anti-nuclei, all of which this port refuses at emission already
  /// (README's species table). This counter separates "the model produced something this
  /// transport cannot step" from "the model produced nothing", which the two `*Inelastic`
  /// counters above cannot distinguish.
  kInelasticSecondarySpecies,
  /// The `do { ApplyYourself } while(!CheckResult)` loop ran its 100 attempts out. Geant4
  /// raises the `had006` FatalException there; a kernel cannot throw.
  ///
  /// Note what is NOT in this port's `check_result`: the short-lived escape clause, which lets
  /// a resonance's dynamic mass sit within three PDG widths of its definition mass
  /// (G4HadronicProcess.cc:657). P5 refused it by name for want of a width table. All three
  /// models decay their strong resonances before returning, so no short-lived secondary should
  /// reach the test - and this counter is what says whether that is true.
  kInelasticReentryExhausted,
  /// The interaction queue was full when a stepper tried to enqueue. A TRIPWIRE, and the file
  /// header of `interaction_queue.cuh` has the proof that it cannot fire at the default
  /// capacity: one track queues at most one interaction per launch and a launch steps at most
  /// `pool` tracks. The track is killed with its kinetic energy deposited locally.
  kInelasticQueueFull,
  /// A thread of the interaction kernel found no workspace slot. A TRIPWIRE for the same
  /// reason: the engine launches `min(n_queued, n_slots)` threads per chunk and drains the
  /// queue in chunks, so a thread index past the last slot cannot occur. It is kept because the
  /// alternative to refusing is two threads sharing one 1.48 MB workspace, which is a data race
  /// whose symptom is a wrong shower (docs/RISK.md V145).
  kInteractionNoSlot,
  /// An antiproton's inelastic process. STRUCTURALLY ZERO, and for the same reason as
  /// `kAntiNucleusElastic`: `G4HadronicBuilder::BuildAntiLightIonsFTFP` gives it
  /// `G4CrossSectionInelastic(G4ComponentAntiNuclNuclearXS)`, the component P2 refuses by name,
  /// so an antiproton draws no inelastic interaction length at all. The gap is in the CROSS
  /// SECTION and not in a final state that could not be applied - the MODEL runs, as P12b's
  /// at-rest campaign shows - and booking it per step would count chances.
  kAntiNucleusInelastic,
  /// A stopped negative hadron's at-rest capture that produced no final state:
  /// `stopping::at_rest` came back with a `StoppingRefusal`. The chain's own reasons are
  /// P12's and are listed in docs/PORTED.md 2.1.13 - the anti-nucleus hand-over, the P6
  /// interface refusals and the handful of Bertini ones.
  ///
  /// It REPLACES `kStoppedNegativeHadron`, `kStoppedMuonMinus` and `kStoppedAntiProton` as the
  /// thing that is booked when the capture runs and fails; those three stay for the case P15
  /// did not change, which is a stage whose at-rest processes are inactivated on both sides.
  kAtRestRefused,

  kNumHadronicRefusals,
};

__host__ __device__ inline const char* hadronic_refusal_name(HadronicRefusal r) {
  switch (r) {
    case HadronicRefusal::kNeutronInelastic:
      return "a neutron's inelastic interaction with no final state [SIZE - do not add to the "
             "WHY group]";
    case HadronicRefusal::kChargedHadronInelastic:
      return "a charged hadron's inelastic interaction with no final state [SIZE - do not add "
             "to the WHY group]";
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
      return "a capture cascade longer than the final state can hold (capacity 32)";
    case HadronicRefusal::kCaptureSecondarySpecies:
      return "a capture secondary whose PDG code core/particle.cuh has no row for";
    case HadronicRefusal::kCaptureRefused:
      return "a capture G4NeutronRadCapture could not complete - an unphysical target, or "
             "CheckResult rejecting 100 attempts (G4Exception had006)";
    case HadronicRefusal::kElasticDropped:
      return "an elastic final state with more than one secondary - G4HadronElasticProcess "
             "keeps only GetSecondary(0)";
    case HadronicRefusal::kIonWithoutNuclide:
      return "a GenericIon track with no (Z, A), or one outside AME2012 - killed with its "
             "energy deposited";
    case HadronicRefusal::kIonDeltaRay:
      return "an ion's delta-ray channel (G4ionIonisation above ~17 GeV/u) - counted PER STEP, "
             "not per interaction";
    case HadronicRefusal::kLightIonCascade:
      return "WHY: G4BinaryLightIonReaction's cascade arm - G4BinaryCascade::Propagate "
             "refused inside Interact";
    case HadronicRefusal::kBinaryHydrogenTarget:
      return "WHY: G4BinaryCascade::Propagate1H1 - a nucleon or pion on hydrogen (P9)";
    case HadronicRefusal::kFtfpRefused:
      return "WHY: ftf::entry::apply refused by name (P11; see its Report)";
    case HadronicRefusal::kBertiniRefused:
      return "WHY: bert::apply_yourself refused by name (P10)";
    case HadronicRefusal::kBinaryRefused:
      return "WHY: the Binary cascade or the light-ion reaction refused for a reason that is "
             "neither hydrogen nor the cascade arm - a tripwire on the model table";
    case HadronicRefusal::kNoInelasticModel:
      return "WHY: G4EnergyRangeManager found no model in range (G4Exception had005) - a "
             "tripwire on the transcribed windows";
    case HadronicRefusal::kInelasticSecondaryOverflow:
      return "WHY: an inelastic final state with more than 256 secondaries";
    case HadronicRefusal::kInelasticSecondarySpecies:
      return "an inelastic secondary whose PDG code core/particle.cuh has no row for - "
             "hyperons, K0S/K0L, anti-nuclei; NOT expected to be zero";
    case HadronicRefusal::kInelasticReentryExhausted:
      return "WHY: ApplyYourself/CheckResult ran 100 attempts out (G4Exception had006)";
    case HadronicRefusal::kInelasticQueueFull:
      return "WHY: the interaction queue was full - a tripwire; the track is killed with its "
             "energy deposited locally";
    case HadronicRefusal::kInteractionNoSlot:
      return "WHY: no workspace slot for this interaction thread - a tripwire under the "
             "chunked drain";
    case HadronicRefusal::kAntiNucleusInelastic:
      return "antiproton inelastic (G4ComponentAntiNuclNuclearXS refused by P2) - "
             "structurally zero";
    case HadronicRefusal::kAtRestRefused:
      return "a stopped negative hadron's at-rest capture with no final state "
             "(stopping::at_rest; P12's own refusals)";
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
/// `__host__ __device__` since P8b, so that `step_hadron` can be run on the host and compared
/// against the device bit for bit (`tests/test_step_hadron.cu`). A host caller is
/// single-threaded, so the host arm is a plain increment and not a serialised atomic.
template <typename real_t>
__host__ __device__ inline void book_refusal(const HadronicRefusalBooks& books,
                                             HadronicRefusal r, real_t energy) {
  const int i = static_cast<int>(r);
#ifdef __CUDA_ARCH__
  if (books.count != nullptr) { atomicAdd(&books.count[i], 1); }
  if (books.energy != nullptr) { atomicAdd(&books.energy[i], static_cast<double>(energy)); }
#else
  if (books.count != nullptr) { books.count[i] += 1; }
  if (books.energy != nullptr) { books.energy[i] += static_cast<double>(energy); }
#endif
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
  /// READ IN TWO PLACES NOW: `step_neutral`'s cross-section gate, and `step_hadron`'s elastic
  /// interaction length. P8b closed the gap P8 named here - the isotope abundances
  /// `SampleZandA` draws a target from are in `data/isotope_abundance.hh` - so a charged hadron
  /// in this transport has a real hadronic process. Which (cross section, model) pair per
  /// species is `had::elastic_channel`, transcribed from
  /// `G4HadronElasticPhysics::ConstructProcess` in `elastic_wiring.cuh`.
  bool hadron_elastic = true;
  /// The device tables `hadElastic` reads: the two BGG per-Z tables and
  /// G4ElasticHadrNucleusHE's per-(pion, Z) G4ElasticData. Null in a run with no charged
  /// hadron, which costs nothing and behaves exactly as a species with no elastic process
  /// does. `host/hadronic_upload.cuh` fills them.
  ElasticTables<real_t> elastic{};
  /// `nCapture` - the capture sub-process of the neutron general process, or in `kStage1` a
  /// process of its own on the neutron's manager.
  ///
  /// REACHED SINCE P8d. P7's model is written and tested (`tests/test_capture.cu`), P8b put the
  /// level scheme it walks on the device (`tests/test_capture_device.cu`, `level_data` below),
  /// and P8d supplies the cross section (`neutron.capture`) and the branch in `step_neutral`.
  /// Off means a neutron draws no capture interaction length at all, which is
  /// `/process/inactivate nCapture` in the stage-1 configuration and has no Geant4 equivalent
  /// in the final one (V53: the sub-processes cannot be inactivated one at a time).
  bool neutron_capture = true;
  /// The neutron's two per-process cross sections, `G4NeutronElasticXS` and
  /// `G4NeutronCaptureXS`, on the device. Null in a run whose `G4PARTICLEXSDATA` could not be
  /// resolved, which is the same "no process" state a species with no elastic channel is in.
  /// `host/neutron_upload.cuh` fills them; `physics/hadronic/neutron_wiring.cuh` says which
  /// stage reads them for what.
  NeutronSubTables<real_t> neutron{};
  /// P3's PhotonEvaporation5.7 level scheme, or a null view.
  ///
  /// The second of the two tables P8 named as blocking the neutron: 174,411 levels and 268,190
  /// transitions, 9.52 MB on the device, which `G4PhotonEvaporation::BreakUpChain` walks inside
  /// a capture. `host/level_upload.cuh` fills it. It was off by default while nothing read it;
  /// `TransportEngine::SetNuclearLevelData` defaults to ON since P8d, because the consumer
  /// exists now and Geant4's own answer is unconditional
  /// (`G4ExcitationHandler::SetParameters` calls `UploadNuclearLevelData(Zmax+1)` at
  /// initialisation whether a neutron arrives or not). A null view is what a capture cascade
  /// with no levels to walk sees, and `neutron_capture_apply` then reports the model's own
  /// refusal rather than inventing a gamma.
  data::LevelTable level_data{};
  /// `*Inelastic` for every species QBBC gives one: `protonInelastic`, `pi+-Inelastic`,
  /// `kaon+-Inelastic`, `dInelastic`, `tInelastic`, `He3Inelastic`, `alphaInelastic`,
  /// `ionInelastic`, and the inelastic sub-process of `G4NeutronGeneralProcess`.
  ///
  /// Off means those species draw no inelastic interaction length at all, which is the state
  /// every one of them was in before P15 and is what the `/process/inactivate <x>Inelastic`
  /// column of `tools/b1_sweep.ps1` compares against. It does NOT switch off the at-rest
  /// captures - `hadron_at_rest` is that - because Geant4 has separate UI names for them and
  /// the like-for-like columns need the same separation.
  bool hadron_inelastic = true;
  /// The five ion UI names alone - `dInelastic`, `tInelastic`, `He3Inelastic`, `alphaInelastic`
  /// and `ionInelastic`, which `had::is_ion_inelastic_species` selects. AND-ed with
  /// `hadron_inelastic`, so off here leaves p, n, pi and K untouched.
  ///
  /// IT EXISTS BECAUSE THE LIKE-FOR-LIKE COLUMN CANNOT BE BUILT WITHOUT IT. P9e's
  /// `G4BinaryLightIonReaction::Interact` is not wired, so 91.5% of an 840 MeV alpha's
  /// interactions are refused - measured, 10,266 of 11,223 queued in 20,000 events - and the
  /// refusal disposes of the ion by killing it with its kinetic energy deposited AT THE POINT
  /// OF THE REFUSAL: 6.03e6 MeV, 36% of that beam's energy, dumped in the water upstream of
  /// B1's scoring trapezoid. A Geant4 run with `alphaInelastic` inactivated carries every alpha
  /// to full range instead, and the two columns then differ by -39.27% (-201.8 sigma) for a
  /// reason that is the refusal's disposal and not the physics. Off on both sides WAS the only
  /// honest comparison until P9e landed. docs/RISK.md V192.
  ///
  /// SINCE P9e IT IS A STUDY KNOB AND NOTHING SETS IT. `Interact` is wired, the five come off
  /// the Geant4 side's inactivation list, and the default - on - is the like-for-like
  /// configuration. What it is still good for is the question it was built to answer: what the
  /// ion inelastic process is worth to a beam, on the port side, one binary run both ways.
  bool ion_inelastic = true;
  /// `G4HadronStoppingProcess` for a stopped mu-, pi-, K-, Sigma-, Xi-, Omega-, pbar or nbar:
  /// `hBertiniCaptureAtRest`, `hFritiofCaptureAtRest` and `muMinusCaptureAtRest`, which are
  /// three UI names on the Geant4 side and one switch here because they are one process class.
  ///
  /// Off is `HadronicStage::kStage1`'s configuration, where all three are inactivated and a
  /// stopped negative hadron decays on both sides instead.
  bool hadron_at_rest = true;
  /// The five `G4ParticleInelasticXS` data sets and `G4BGGPionInelasticXS`, on the device.
  /// Null in a run whose `G4PARTICLEXSDATA` could not be resolved, which behaves exactly as a
  /// species with no inelastic process does. `host/hadronic_upload.cuh` fills them; the
  /// neutron's own `G4NeutronInelasticXS` travels in `neutron.inelastic` instead, because all
  /// three of its data sets have to come from the one load that built the combined table.
  InelasticTables<real_t> inelastic{};
  /// Where a stepper records an interaction it is not going to run itself. See
  /// `interaction_queue.cuh`: the models are too large for a stepping kernel (docs/RISK.md
  /// V188), so the stepper does the whole step except the model call and the interaction
  /// kernel does the rest.
  InteractionQueue<real_t> queue{};
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
__host__ __device__ inline int emit_decay_products(const decay::DecayProducts<real_t>& products,
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
