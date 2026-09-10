// G4Decay, the one process QBBC attaches to every unstable particle.
//
// Transcribed from G4Decay.cc and G4VRestDiscreteProcess.cc (11.1.1) plus
// G4DecayTable::SelectADecayChannel and G4DecayProducts::Boost. `G4DecayPhysics` builds ONE
// G4Decay and registers it against every particle for which `IsApplicable` is true, so there
// is nothing per-species in the process itself: everything species-dependent is the decay
// table, which is data (decay_tables.hh).
//
// WHAT A REST-DISCRETE PROCESS IS. G4Decay derives from G4VRestDiscreteProcess, so it offers
// the stepper two lengths and the stepper picks whichever comes first:
//
//   in flight   `GetMeanFreePath` * -log(rand), a LENGTH in mm. The mean free path is
//               beta*gamma*c*tau, which the code writes as p/m * c*tau.
//   at rest     `GetMeanLifeTime` * -log(rand), a TIME in ns. The mean life is just tau -
//               there is no Lorentz factor, because there is no motion.
//
// THE AT-REST BRANCH IS PRE-EMPTED, NOT CONTESTED - AND FOR THE MUON TOO.
//
//   The plan says a stopped pi- is captured rather than decayed. Measuring it
//   (ref/oracle/decay_atrest.csv, which asks every at-rest process QBBC registered for the
//   length it would offer a stopped track) says something stronger, and it changes what the
//   wiring package has to do.
//
//   Three species carry a second at-rest process, and in all three it is FIRST in the at-rest
//   vector and its interaction length is exactly 0.0:
//
//     pi-     hBertiniCaptureAtRest    0.0        G4Decay  -log(rand) * 26.033 ns
//     kaon-   hBertiniCaptureAtRest    0.0        G4Decay  -log(rand) * 12.380 ns
//     mu-     muMinusCaptureAtRest     0.0        G4Decay  -log(rand) * 2196.98 ns
//
//   All three capture processes derive from `G4HadronStoppingProcess`, whose
//   `AtRestGetPhysicalInteractionLength` is `return 0.0;` with the track not even read
//   (G4HadronStoppingProcess.cc:115). A zero is not a competitor in a smallest-wins race, it
//   is a pre-emption: no sample of `-log(rand) * tau` can beat it. So G4Decay's at-rest
//   branch NEVER fires for pi-, kaon- or mu-, and the port must say that rather than describe
//   a race. This file said mu- was "the real competition ... both processes have a finite
//   at-rest length and the stepper's ordinary smallest-wins rule decides", and that was
//   wrong in both halves.
//
//   Where the muon's decay actually goes is inside the capture process.
//   `G4MuonMinusCapture` hands the stopped muon to `G4MuonMinusBoundDecay`, which computes
//   the nuclear capture rate lambda_c(Z, A) from Suzuki et al. Phys.Rev. C35 (1987) 2212 and
//   the BOUND decay rate lambda_d(Z, A), samples the time from lambda_c + lambda_d, and takes
//   capture with probability lambda_c/lambda. On the decay branch it samples its OWN Michel
//   spectrum - `x` uniform on (2 m_e/m_mu, 1 + (m_e/m_mu)^2) against (3-2x)x^2, in the
//   muonic-atom K shell, with the bound energy subtracted from the electron and the two
//   neutrinos built from the recoiling four-vector. That is a DIFFERENT sampler from
//   G4MuonDecayChannel and it belongs to P12, not here.
//
//   A stopped pi+, K+ or mu+ is Coulomb-repelled from every nucleus, has no at-rest process
//   but G4Decay, and does decay at rest. So the at-rest branch of this file is reachable for
//   the POSITIVE species only, and that asymmetry is a factor of two in the muon yield of a
//   pion shower - not a detail.
//
//   This package provides `at_rest_mean_life`, `at_rest_interaction_length` and
//   `decay_at_rest_competitor` so the wiring package (P8) can see all of that, and it does
//   NOT implement capture - that is P12, behind Bertini (P10). Until P12 exists, a stopped
//   pi-/K-/mu- that decays here is wrong in a way the plan already knows about, and
//   `decay_at_rest_competitor` names the species so a wiring that forgets cannot forget
//   silently.
//
// THE PRE-ASSIGNED DECAY PATH IS NOT USED, AND THAT IS VERIFIED RATHER THAN ASSUMED.
//
//   G4Decay has a whole second code path for a track whose G4DynamicParticle carries
//   pre-assigned decay products or a pre-assigned decay proper time: the products are copied
//   instead of sampled, and the length is built from the remaining proper time instead of
//   from `-log(rand)`. That length has two arms of its own - `c_light * fRemainderLifeTime *
//   GetTotalMomentum()/mass` when the lifetime is positive, and the short-lived arm at
//   G4Decay.cc:463 - and the remainder is clamped to 0.0 in PostStepGPIL (:459) but to
//   DBL_MIN in AtRestGPIL (:488), which are different answers to the same question. It all
//   exists for event generators that decayed the particle themselves (HepMC, an external
//   decayer). G4DynamicParticle's defaults are `thePreAssignedDecayProducts = nullptr` and
//   `thePreAssignedDecayTime = -1.0` (G4DynamicParticle.hh:241, :264), nothing in QBBC's
//   chain calls either setter, and the primary generator this port models is a particle gun.
//   ref/dump/dump_decay.cc dumps both for a freshly built G4DynamicParticle of every species
//   so the claim is a number in a CSV: 0 of 507.
//
//   So the path is REFUSED, not implemented: neither `in_flight_interaction_length` nor
//   `at_rest_interaction_length` takes a pre-assigned time, and `sample_decay` has no
//   pre-assigned-products branch. A caller that ever needs one will find this comment rather
//   than a silently wrong length. The external-decayer branch (`pExtDecayer`, nullptr in
//   QBBC) is refused on the same evidence and by the same absence.
//
// WHAT THIS FILE DOES NOT COVER, so that the gap is a sentence and not a surprise. G4Decay
// ends DecayIt by proposing `fStopAndKill` for the parent and calling
// `ClearNumberOfInteractionLengthLeft` (G4Decay.cc:376, :381). Both are track bookkeeping
// rather than kinematics, and both belong to whatever owns the track: this API returns
// daughters, an energy deposit and a time advance, and the caller kills the parent. The
// interaction-length count itself is process state and IS here -
// `reset_number_of_interaction_lengths` and `subtract_interaction_lengths` below.
#pragma once
#include <cmath>

#include "core/units.cuh"
#include "physics/decay/decay_channels.cuh"
#include "physics/decay/decay_products.cuh"
#include "physics/decay/decay_tables.hh"

namespace g4gpu::decay {

/// The `HighestValue` G4Decay's constructor sets (G4Decay.cc:65): the normalised kinetic
/// energy above which the mean free path is computed as (Ekin/m + 1)*c*tau rather than from
/// the momentum.
///
/// The threshold is on Ekin/m, which is gamma MINUS ONE, so the handover is at gamma = 21,
/// beta = 0.998866, where gamma*c*tau exceeds beta*gamma*c*tau by 1.13e-3. (This comment
/// said "a gamma of 20, beta = 0.99875, 1.3e-3", which are the gamma = 20 numbers - off by
/// one in exactly the way the variable name invites.) It is a real discontinuity in the mean
/// free path, not a rounding one, and it is Geant4's.
__host__ __device__ inline constexpr double decay_highest_value() { return 20.0; }

/// DBL_MAX and DBL_MIN as G4Decay uses them: DBL_MAX for "never decays" and DBL_MIN for "the
/// particle stops here". They are sentinels rather than physical lengths and the caller has
/// to treat them as such - a step limit of DBL_MIN is not a step, it is a decay now.
///
/// WITH real_t = float THE SENTINELS DEGRADE, and the caller has to know which way. DBL_MIN
/// is below the smallest positive float and DBL_MAX above the largest, so a float
/// instantiation returns 0 and +inf for them, and both `< DBL_MIN` tests inside
/// `in_flight_mean_free_path` become `< 0` and never fire. A stopped particle then falls
/// through to the momentum form and gets p/m*ctau with p = 0, which is 0 - the same answer
/// the sentinel branch would have given to within a denormal, and the same meaning. So a
/// caller must test `<= decay_zero_length()` rather than `== ` it, and `>=
/// decay_infinite_length()` rather than `==`. Nothing is approximated; the double
/// instantiation, which is what the oracle is compared at, returns Geant4's literal values.
__host__ __device__ inline constexpr double decay_infinite_length() { return 1.7976931348623157e308; }
__host__ __device__ inline constexpr double decay_zero_length() { return 2.2250738585072014e-308; }

/// G4Decay::IsApplicable, G4Decay.cc:88. Note it reads the LIFETIME, not the stable flag:
/// a particle with a non-negative lifetime and a positive mass gets the process, so the
/// triton (flagged stable, lifetime 17.774 years) gets one and then never decays.
///
/// A PDG code with no row has lifetime -1 here, so it is declined - which is the right answer
/// for a code this package refuses: no process rather than a process that cannot fire.
__host__ __device__ inline bool decay_is_applicable(int pdg) {
  const double life = particle_lifetime(pdg);
  if (life < 0.0) { return false; }
  return particle_mass(pdg) > 0.0;
}

/// G4Decay::GetMeanLifeTime, G4Decay.cc:100 - the AT REST mean life, ns.
///
/// A stable particle gets 1e24 s, which G4Decay's own comment calls "1000000 times the life
/// time of the universe". It is a finite number and not DBL_MAX, deliberately: the at-rest
/// interaction length is `-log(rand) * this`, and DBL_MAX there would overflow.
template <typename real_t>
__host__ __device__ inline real_t at_rest_mean_life(int pdg) {
  if (particle_is_stable(pdg)) { return real_t(1e24) * units::s<real_t>(); }
  return static_cast<real_t>(particle_lifetime(pdg));
}

/// G4Decay::GetMeanFreePath, G4Decay.cc:129 - the IN FLIGHT mean free path, mm.
///
/// Five branches, tested in this order, and the middle three are easy to get wrong:
///
///   stable                       DBL_MAX
///   c*tau below DBL_MIN          DBL_MIN - "very short life time", decay immediately.
///                                pi0's c*tau is 2.55e-5 mm, nowhere near it; this guards a
///                                short-lived resonance, not anything transported.
///   Ekin/m > 20                  (Ekin/m + 1) * c*tau. That is gamma*c*tau, i.e. it drops
///                                the beta factor because beta is within 1.3e-3 of one.
///   Ekin/m < DBL_MIN             DBL_MIN - "too slow particle", i.e. it stops here. This is
///                                the branch that hands a stopped pi- to the at-rest
///                                competition.
///   otherwise                    p/m * c*tau = beta*gamma*c*tau, the exact form.
///
/// `mass` is the DYNAMIC mass (`aParticle->GetMass()`), and `lifetime` comes from the
/// definition, so a particle whose mass has drifted off the PDG value gets a mean free path
/// built from the drifted mass and the PDG lifetime. Both are taken as arguments here for
/// that reason.
template <typename real_t>
__host__ __device__ inline real_t in_flight_mean_free_path(int pdg, real_t mass,
                                                           real_t kinetic_energy) {
  const real_t life = static_cast<real_t>(particle_lifetime(pdg));
  const real_t ctau = units::c_light<real_t>() * life;
  if (particle_is_stable(pdg)) { return static_cast<real_t>(decay_infinite_length()); }
  if (ctau < static_cast<real_t>(decay_zero_length())) {
    return static_cast<real_t>(decay_zero_length());
  }
  const real_t r_kinetic = kinetic_energy / mass;
  if (r_kinetic > static_cast<real_t>(decay_highest_value())) {
    return (r_kinetic + real_t(1)) * ctau;
  }
  if (r_kinetic < static_cast<real_t>(decay_zero_length())) {
    return static_cast<real_t>(decay_zero_length());
  }
  // G4DynamicParticle::GetTotalMomentum: sqrt((Ekin + 2m)*Ekin), not sqrt(Ekin^2+2m*Ekin).
  const real_t p = sqrt((kinetic_energy + real_t(2) * mass) * kinetic_energy);
  return p / mass * ctau;
}

/// G4VProcess::ResetNumberOfInteractionLengthLeft, G4VProcess.cc:80: `-log(G4UniformRand())`.
/// Drawn ONCE PER TRACK, by `G4Decay::StartTracking` (G4Decay.cc:393), and not again.
template <typename real_t, typename rng_t>
__host__ __device__ inline real_t reset_number_of_interaction_lengths(rng_t& rng) {
  return -log(static_cast<real_t>(rng.uniform()));
}

/// `G4VProcess::SubtractNumberOfInteractionLengthLeft` (G4VProcess.hh:530) followed by
/// G4Decay's own second floor (G4Decay.cc:429), which is what carries the count from one step
/// to the next. Returns the new number of interaction lengths left.
///
/// THIS IS PROCESS CODE, NOT STEPPER CODE, and this file said the opposite. G4Decay
/// OVERRIDES `PostStepGetPhysicalInteractionLength` (G4Decay.cc:410) rather than inheriting
/// G4VRestDiscreteProcess's, and its override does the subtraction itself:
///
///   theNumberOfInteractionLengthLeft -= previousStepSize/currentInteractionLength
///   if (theNumberOfInteractionLengthLeft < perMillion) it = perMillion    (in the base)
///   if (theNumberOfInteractionLengthLeft < 0.)         it = perMillion    (again, in G4Decay)
///
/// The double floor is Geant4's; the second test can only fire if the first was skipped
/// because `currentInteractionLength <= 0`. `perMillion` is CLHEP's 1e-6.
///
/// It also matters that NEITHER `G4Decay::PostStepGetPhysicalInteractionLength` nor
/// `G4Decay::AtRestGetPhysicalInteractionLength` ever calls
/// ResetNumberOfInteractionLengthLeft - the base class would, on a negative previous step or
/// an exhausted count, and G4Decay does not. So the number used AT REST is whatever was left
/// over from stepping the particle in flight, not a fresh draw made when it stopped. A wiring
/// that redraws at the stop is a different process.
__host__ __device__ inline constexpr double per_million() { return 1.0e-6; }

template <typename real_t>
__host__ __device__ inline real_t subtract_interaction_lengths(real_t n_lengths_left,
                                                               real_t previous_step,
                                                               real_t current_length) {
  if (current_length > real_t(0)) {
    n_lengths_left -= previous_step / current_length;
    if (n_lengths_left < static_cast<real_t>(per_million())) {
      n_lengths_left = static_cast<real_t>(per_million());
    }
  }
  if (n_lengths_left < real_t(0)) { n_lengths_left = static_cast<real_t>(per_million()); }
  return n_lengths_left;
}

/// The in-flight length G4Decay offers the stepper: `n * lambda`, with lambda the mean free
/// path above. `G4Decay::PostStepGetPhysicalInteractionLength` (G4Decay.cc:446) and
/// `G4VRestDiscreteProcess::PostStepGetPhysicalInteractionLength` (G4VRestDiscreteProcess.cc:99)
/// spell it identically; G4Decay's own is the one that runs.
template <typename real_t>
__host__ __device__ inline real_t in_flight_interaction_length(int pdg, real_t mass,
                                                               real_t kinetic_energy,
                                                               real_t n_lengths_left) {
  const real_t mfp = in_flight_mean_free_path<real_t>(pdg, mass, kinetic_energy);
  // `if (currentInteractionLength < DBL_MAX) value = n * lambda; else value = DBL_MAX`.
  // Without the test a stable particle's DBL_MAX would be multiplied and overflow to infinity.
  if (mfp < static_cast<real_t>(decay_infinite_length())) { return n_lengths_left * mfp; }
  return static_cast<real_t>(decay_infinite_length());
}

/// G4Decay::AtRestGetPhysicalInteractionLength, G4Decay.cc:477 (the pTime < 0 branch), ns.
///
/// No DBL_MAX guard here, and that is deliberate: G4Decay's at-rest form is the bare product
/// `theNumberOfInteractionLengthLeft * GetMeanLifeTime(...)` (G4Decay.cc:490). The base
/// class's at-rest version DOES have the guard (G4VRestDiscreteProcess.cc:145) and copying it
/// here would have been the wrong parent. It needs none: GetMeanLifeTime answers a finite
/// 1e24 s for a stable particle rather than DBL_MAX, which is exactly why that magic number
/// is finite.
///
/// What a stopped pi-, kaon- or mu- does with this number is nothing: its capture process
/// offers zero and pre-empts it. See the file header.
template <typename real_t>
__host__ __device__ inline real_t at_rest_interaction_length(int pdg, real_t n_lengths_left) {
  return n_lengths_left * at_rest_mean_life<real_t>(pdg);
}

/// Whether a stopped particle of this species has an at-rest process competing with decay in
/// QBBC, and which. Not a physics function - a hook, so that a wiring package that puts decay
/// into the at-rest queue without its competitor has to pass this and say so.
///
/// `G4StoppingPhysics::ConstructProcess` in 11.1.1 registers G4HadronicAbsorptionBertini on
/// pi-, kaon-, Sigma-, Xi-, Omega- and G4HadronicAbsorptionFritiof on the antibaryons, plus
/// G4MuonMinusCapture on mu-. Only three of those are species this port transports.
enum class AtRestCompetitor : int {
  kNone = 0,
  /// G4HadronicAbsorptionBertini, an at-rest process with a zero interaction length: capture
  /// ALWAYS wins against decay. P12.
  kHadronicAbsorptionBertini,
  /// G4MuonMinusCapture, ALSO an at-rest length of exactly zero, so also a pre-emption and
  /// not a race. The capture-versus-decay split happens inside it, in G4MuonMinusBoundDecay,
  /// against its own bound Michel spectrum. P12.
  kMuonMinusCapture,
};

__host__ __device__ inline AtRestCompetitor decay_at_rest_competitor(int pdg) {
  switch (pdg) {
    case kPdgPiMinus:
    case kPdgKaonMinus:
      return AtRestCompetitor::kHadronicAbsorptionBertini;
    case kPdgMuMinus:
      return AtRestCompetitor::kMuonMinusCapture;
    default:
      return AtRestCompetitor::kNone;
  }
}

__host__ __device__ inline const char* decay_at_rest_competitor_name(AtRestCompetitor c) {
  switch (c) {
    case AtRestCompetitor::kNone: return "none: decay is the only at-rest process";
    case AtRestCompetitor::kHadronicAbsorptionBertini:
      return "G4HadronicAbsorptionBertini (P12): nuclear capture of a stopped negative hadron, "
             "at-rest length zero - capture always wins, this decay must not fire";
    case AtRestCompetitor::kMuonMinusCapture:
      return "G4MuonMinusCapture (P12): at-rest length zero, so capture always wins and this "
             "decay must not fire either - the capture-versus-bound-decay split is inside "
             "G4MuonMinusBoundDecay, with its own K-shell Michel spectrum";
  }
  return "unknown";
}

/// G4DecayTable::SelectADecayChannel, G4DecayTable.cc:82. Returns an index into the parent's
/// slice of kChannels, or -1.
///
/// Three things it does that a straight cumulative-BR draw would not:
///
///   It sums only the channels that pass `IsOKWithParentMass`, and normalises the draw by
///   that sum. So the BRs need not add to one - K+/K-'s add to 0.99991 - and a channel closed
///   by a low parent mass redistributes its share over the open ones.
///
///   In the selection loop it accumulates `sum += BR` for EVERY channel but only RETURNS one
///   that passes the mass test. So a closed channel still advances the cumulative sum: its
///   share of the interval selects nothing and the loop falls through to the outer retry. The
///   outer loop runs up to 10000 times for that reason, and this is the only way it is ever
///   entered more than once.
///
///   `parentMass < 0` means "use the PDG mass".
template <typename real_t, typename rng_t>
__host__ __device__ inline int select_a_decay_channel(const DecayTableRow& table,
                                                      real_t parent_mass, rng_t& rng) {
  if (table.count < 1) { return -1; }
  double pm = static_cast<double>(parent_mass);
  if (pm < 0.0) { pm = particle_mass(table.parent_pdg); }

  double sum_br = 0.0;
  for (int i = 0; i < table.count; ++i) {
    if (!channel_ok_with_parent_mass(channel_rows()[table.first + i], pm)) { continue; }
    sum_br += channel_rows()[table.first + i].br;
  }
  if (sum_br <= 0.0) { return -1; }

  const int kMaxLoop = 10000;
  for (int loop = 0; loop < kMaxLoop; ++loop) {
    double sum = 0.0;
    const double br = sum_br * static_cast<double>(rng.uniform());
    for (int i = 0; i < table.count; ++i) {
      const ChannelRow& ch = channel_rows()[table.first + i];
      sum += ch.br;
      if (!channel_ok_with_parent_mass(ch, pm)) { continue; }
      if (br < sum) { return i; }
    }
  }
  return -1;
}

/// The device-callable entry point: a parent in, its daughters out.
///
/// `pdg`             the parent's PDG code. A code with no transcribed table is refused by
///                   code (status kNoTable) and never approximated by a similar species.
/// `mass`            the parent's DYNAMIC mass, MeV. G4Decay passes `aParticle->GetMass()`
///                   into SelectADecayChannel and into DecayIt; pass the PDG mass if the
///                   transport does not track a dynamic one.
/// `kinetic_energy`  MeV. Used only for the boost.
/// `direction`       the parent's momentum direction, a unit vector. Used only for the boost.
/// `at_rest`         true for the AtRest branch (G4Track status fStopButAlive). It changes
///                   two things and neither is the kinematics: the parent's kinetic energy is
///                   DEPOSITED LOCALLY rather than carried by the daughters, and the products
///                   are NOT boosted. `local_energy_deposit` returns what to deposit.
/// `out`             the caller's buffer. See decay_products.cuh for the contract.
///
/// The four-momentum is not summed and checked here. G4Decay does not check it either - it
/// builds the products in the rest frame, boosts them, and trusts the channel - and
/// tests/test_decay.cu is where the sum is required to close, per channel, over a large
/// sample, because that is a property of the transcription rather than of a decay.
template <typename real_t, typename rng_t>
__host__ __device__ inline void sample_decay(int pdg, real_t mass, real_t kinetic_energy,
                                             const real_t direction[3], bool at_rest,
                                             rng_t& rng, DecayProducts<real_t>& out) {
  out.clear();

  // REFUSED BEFORE STABLE, and the order is the whole point. `particle_is_stable` answers
  // `true` for a PDG code it has never heard of, so that an unknown code can never reach a
  // sampler; but if the stable test came first, K0L would come back "stable" - which is not
  // a refusal, it is a wrong physics answer that a caller would act on. A code with no row in
  // the particle table is refused by code, with a message, whatever else is true of it.
  if (particle_index(pdg) < 0) {
    out.fail(DecayStatus::kNoTable);
    return;
  }

  // G4Decay::DecayIt, first test: a stable particle returns an untouched particle change.
  // Nothing happens, nothing is killed, and it is not an error.
  if (particle_is_stable(pdg)) {
    out.fail(DecayStatus::kStable);
    return;
  }
  const DecayTableRow table = decay_table_for(pdg);
  if (table.count < 1) {
    // G4Decay::DecayIt raises DECAY101 (JustWarning) and kills the parent with no
    // secondaries when there is no decay table. Here it means the species is refused.
    out.fail(DecayStatus::kNoTable);
    return;
  }

  const int ich = select_a_decay_channel<real_t>(table, mass, rng);
  if (ich < 0) {
    // G4Decay::DecayIt, "DECAY003", FatalException.
    out.fail(DecayStatus::kNoChannel);
    return;
  }
  const ChannelRow& ch = channel_rows()[table.first + ich];
  channel_decay_it<real_t>(ch, pdg, mass, rng, out);
  if (out.status != DecayStatus::kOK) { return; }
  out.channel = ich;

  // G4Decay::DecayIt: ParentEnergy = total energy, floored at the mass with a JustWarning
  // ("Total Energy is less than its mass - increased the energy", DECAY102, G4Decay.cc:318).
  real_t parent_energy = kinetic_energy + mass;
  if (parent_energy < mass) { parent_energy = mass; }

  // The AtRest branch does not boost. The PostStep branch does - and it boosts by the
  // PARENT'S OWN energy and direction, using the mass the products were built with, which is
  // the channel's parent mass and not necessarily `mass` (the muon, KL3, Dalitz and neutron
  // channels all build their products with the PDG mass whatever `mass` was).
  if (!at_rest) { boost_products(out, parent_energy, direction); }
}

/// G4Decay::DecayIt's `energyDeposit`: the parent's kinetic energy in the AtRest case and
/// zero in flight. In the AtRest case the parent is at rest by definition, so this is the
/// residual kinetic energy the stepper had not yet spent - normally zero - and depositing it
/// is a bookkeeping guarantee rather than a physics effect.
template <typename real_t>
__host__ __device__ inline real_t decay_local_energy_deposit(real_t kinetic_energy,
                                                             bool at_rest) {
  return at_rest ? kinetic_energy : real_t(0);
}

/// G4Decay::DecayIt's time bookkeeping. In the AtRest case both the global and the local time
/// advance by `fRemainderLifeTime` - the at-rest interaction length that fired, in ns - and in
/// flight neither does here (the stepper has already advanced them along the step).
///
/// The proper time is NOT touched by G4Decay: `G4Track::GetProperTime` is advanced by the
/// stepping manager, and G4Decay only reads it, and only on the pre-assigned path this
/// package refuses. So there is no proper-time output.
template <typename real_t>
__host__ __device__ inline real_t decay_time_advance(real_t remainder_life_time, bool at_rest) {
  return at_rest ? remainder_life_time : real_t(0);
}

}  // namespace g4gpu::decay
