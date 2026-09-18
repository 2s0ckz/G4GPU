// G4HadronStoppingProcess and the two absorption processes built on it, plus the at-rest entry
// point the stepper will call.
//
// Transcribed from Geant4 11.1.1:
//   processes/hadronic/stopping/src/G4HadronStoppingProcess.cc
//     AtRestGetPhysicalInteractionLength, AtRestDoIt, IsApplicable
//   processes/hadronic/stopping/src/G4HadronicAbsorptionBertini.cc   (ctor, IsApplicable)
//   processes/hadronic/stopping/src/G4HadronicAbsorptionFritiof.cc   (ctor, IsApplicable)
//   processes/hadronic/stopping/src/G4MuonMinusCapture.cc            (ctor, IsApplicable)
//   physics_lists/constructors/stopping/src/G4StoppingPhysics.cc     (ConstructProcess)
//
// ---------------------------------------------------------------------------------------------
// **The at-rest interaction length is ZERO and the condition is NotForced.** That one line is why
// P4's `decay_at_rest_competitor` says what it says: a stopped negative hadron or muon is captured
// before it can decay, always, because a zero at-rest length wins every competition against
// G4Decay's finite one. It is not a race with a small probability of decay - the decay never
// happens, and the capture-versus-decay question is asked LATER and INSIDE this process, by
// G4MuonMinusBoundDecay, for the muon only.
//
// **WHICH MODEL EACH SPECIES GETS, read from G4StoppingPhysics::ConstructProcess rather than
// assumed.** The gate is `GetPDGCharge() <= 0.0 && GetPDGMass() > 130 MeV && !IsShortLived()`,
// and then:
//
//   mu-                                          G4MuonMinusCapture
//   pi-, K-, Sigma-, Xi-, Omega-                 G4HadronicAbsorptionBertini
//   anti-p, anti-n, anti-Lambda, anti-Sigma0,    G4HadronicAbsorptionFritiof
//     anti-Sigma+, anti-Xi0, anti-nuclei (B<-1)
//
// Three of the Fritiof species are NEUTRAL - anti-neutron, anti-Lambda, anti-Sigma0, anti-Xi0 -
// and they pass the gate because it is `<= 0.0` and not `< 0.0`. The BASE class's own
// `IsApplicable` is `< 0.0`, strictly, and it is only consulted on `G4HadronicAbsorptionBertini`'s
// generic path; Fritiof lists its species explicitly and never asks. So a neutral antiparticle is
// stopped by a process whose base class says it is not applicable to it, and that is not a
// contradiction Geant4 resolves anywhere - the explicit list simply wins.
//
// **AND THE MUON'S BERTINI IS NOT THE OTHERS' BERTINI.** `G4HadronicAbsorptionBertini`'s
// constructor does
//
//     theCascade = new G4CascadeInterface;
//     theCascade->SetMinEnergy(0.);
//     theCascade->usePreCompoundDeexcitation();
//
// and `G4MuonMinusCapture`'s does `hiptr = new G4CascadeInterface();` and nothing else. So muon
// capture runs Bertini with the CASCADE's own evaporators and the other five run it with P6's
// PreCompound - a FOURTH G4CascadeInterface instance in a QBBC run, in a THIRD configuration,
// after the nucleon, pion and kaon/hyperon instances docs/RISK.md V118 describes. The file's own
// change log says why it exists: "20121002 K. Genser -- Replaced G4MuMinusCapturePrecompound with
// G4CascadeInterface (Bertini)". `G4MuMinusCapturePrecompound` is still in the package, is not
// wired to anything QBBC builds, and is therefore NOT transcribed - named here instead, which is
// what "refuse by name" means for a class that no caller reaches.
//
// ---------------------------------------------------------------------------------------------
// **The order inside AtRestDoIt, and what each step is allowed to change.**
//
//   1. `fElementSelector->SelectZandA` picks the atom AND the isotope, and writes them into the
//      G4Nucleus the models will be handed.
//   2. `fEmCascade->ApplyYourself` runs the atomic cascade. Its "local energy deposit" is the
//      binding energy released, is carried by its own secondaries, and is NOT deposited - see
//      em_capture_cascade.cuh.
//   3. `thePro.SetBoundEnergy(ebound)`, then `fBoundDecay->ApplyYourself` IF there is one, which
//      there is only for the muon. If it returns `stopAndKill` the muon decayed in orbit and
//      `nuclearCapture` goes false; the nuclear model is then skipped entirely.
//   4. `capTime = thePro.GetGlobalTime()` - the exponential wait the bound decay wrote - and the
//      projectile's time is reset to zero before the nuclear model runs. Every nuclear secondary
//      then gets `capTime + its own time`, and every secondary of all three steps gets `time0`,
//      the track's global time, added at the end.
//   5. The nuclear model, in a do/while that re-runs it while `CheckResult` returns null, up to
//      100 times - and then a FatalException. A kernel cannot throw, so `kReentryExhausted` is
//      carried out by name and counted.
//
// **Only the NUCLEAR model's local deposit is deposited.** `edep` is initialised to 0.0, set from
// `resultNuc->GetLocalEnergyDeposit()` inside the capture branch, and proposed as the step's
// deposit. When the muon decays in orbit nothing is deposited at all, and the atomic cascade's
// `ebound` is never deposited on either path.
#ifndef G4GPU_STOPPING_STOPPING_PROCESS_CUH
#define G4GPU_STOPPING_STOPPING_PROCESS_CUH

#include <cmath>

#include "physics/hadronic/bertini/cascade_interface.cuh"
#include "physics/hadronic/stopping/element_selector.cuh"
#include "physics/hadronic/stopping/em_capture_cascade.cuh"
#include "physics/hadronic/stopping/muon_bound_decay.cuh"

namespace g4gpu::physics::hadronic::stopping {

/// Which nuclear model QBBC's G4StoppingPhysics gives a species at rest.
enum class NuclearArm : int {
  kNone = 0,     ///< the species gets no stopping process at all
  kBertini,      ///< G4HadronicAbsorptionBertini: Bertini + P6's PreCompound
  kFritiof,      ///< G4HadronicAbsorptionFritiof: FTFP + P6's PreCompound
  kMuonCapture   ///< G4MuonMinusCapture: bound decay, then Bertini with its OWN evaporators
};

/// What the at-rest chain could not do.
enum class StoppingRefusal : int {
  kNone = 0,
  kNotApplicable,        ///< QBBC gives this species no at-rest process
  kSelector,             ///< G4ElementSelector refused - see SelectorRefusal
  kBertiniRefused,       ///< bert::apply_yourself refused
  kFtfRefused,           ///< ftf::apply_yourself refused
  /// FTF at rest reached `DecayStrongResonances`, which is P11d's and is not ported. Counted
  /// rather than approximated; the campaign reports how often.
  kFtfResonanceDecay,
  kSecondaryOverflow,    ///< the caller's HadFinalState filled up
  /// The 100-attempt `do { ... } while(!resultNuc)` loop in AtRestDoIt ran out. Geant4 raises a
  /// FatalException here; a kernel cannot throw, so this is carried out by name.
  kReentryExhausted
};

/// G4StoppingPhysics::ConstructProcess, as a function of the PDG code.
///
/// The mass and charge gate is written out because it is the reason three NEUTRAL antiparticles
/// are in the list and the reason a pi0 and a neutron are not: `charge <= 0`, `mass > 130 MeV`,
/// not short-lived. A neutron is neutral and heavy and not short-lived and still gets nothing,
/// because it matches none of the two explicit lists and falls to the `else` that prints a
/// warning - "not able to deal with nuclear stopping of" - and adds no process.
__host__ __device__ inline NuclearArm stopping_arm(int pdg) {
  if (pdg == 13) { return NuclearArm::kMuonCapture; }         // mu-
  switch (pdg) {
    case -211:    // pi-
    case -321:    // K-
    case 3112:    // Sigma-
    case 3312:    // Xi-
    case 3334:    // Omega-
      return NuclearArm::kBertini;
    case -2212:   // anti-proton
    case -2112:   // anti-neutron          (neutral, and in the list)
    case -3122:   // anti-Lambda           (neutral)
    case -3212:   // anti-Sigma0           (neutral)
    case -3222:   // anti-Sigma+           (charge -1)
    case -3322:   // anti-Xi0              (neutral)
      return NuclearArm::kFritiof;
    default:
      break;
  }
  // Anti-nuclei: `particle->GetBaryonNumber() < -1`. A PDG ion code is 10LZZZAAAI and an
  // anti-ion is its negative, so an anti-deuteron is -1000010020 and its baryon number is -2.
  if (pdg <= -1000000000) { return NuclearArm::kFritiof; }
  return NuclearArm::kNone;
}

/// The de-excitation choice each arm's Bertini instance was built with. See the header: the
/// muon's is a bare G4CascadeInterface and the other five call usePreCompoundDeexcitation().
__host__ __device__ inline bert::DeexciteChoice stopping_deexcite_choice(NuclearArm arm) {
  return (arm == NuclearArm::kMuonCapture) ? bert::DeexciteChoice::kCascade
                                           : bert::DeexciteChoice::kPreCompound;
}

/// `G4HadronStoppingProcess::AtRestGetPhysicalInteractionLength` - zero, NotForced.
///
/// Its own function so that P4's `decay_at_rest_competitor` can be checked against it rather
/// than against a comment: a zero at-rest length pre-empts every competing at-rest process.
__host__ __device__ inline constexpr double at_rest_interaction_length() { return 0.0; }

/// What one at-rest capture did, beyond the final state.
struct AtRestResult {
  StoppingRefusal refusal = StoppingRefusal::kNone;
  SelectorRefusal selector_refusal = SelectorRefusal::kNone;
  NuclearArm arm = NuclearArm::kNone;
  int z = 0;                       ///< the atom the capture happened on
  int a = 0;
  int element_index = -1;
  int n_em_cascade = 0;            ///< how many secondaries step 2 made
  double e_bound_MeV = 0.0;        ///< the atomic cascade's telescoped total
  bool bound_decay_ran = false;    ///< true only for the muon
  bool decayed_in_orbit = false;   ///< the muon decayed; the nuclear model did NOT run
  double capture_time_ns = 0.0;    ///< the exponential wait before nuclear capture
  int reentry_count = 0;           ///< the do/while's counter
  double local_deposit_MeV = 0.0;  ///< what the step deposits: the NUCLEAR model's only
  bert::ApplyResult bertini;       ///< when the arm ran Bertini
};

/// Everything the Bertini arm needs, bundled so the entry point's signature stays readable. Every
/// member is the caller's storage; nothing here is owned.
struct BertiniArmState {
  bert::NucleiModel* model = nullptr;
  bert::CollisionOutput* global_out = nullptr;
  bert::CollisionOutput* out = nullptr;
  bert::CollisionOutput* dex_out = nullptr;
  bert::CollisionOutput* tmp = nullptr;
  bert::ColliderOutput* epo = nullptr;
  bert::BertiniWorkspace* ws = nullptr;
};

/// `G4HadronStoppingProcess::AtRestDoIt`, and the entry point the stepper's at-rest branch will
/// call. P5's shapes: a projectile, a material, a final state.
///
/// **Where the stepper will call it.** The at-rest branch belongs in `physics/stepper.cuh`
/// beside the decay one, gated exactly as P8b gated hadElastic - a stopped track whose species
/// `stopping_arm(pdg)` does not answer `kNone` goes here INSTEAD of to `G4Decay`, because
/// `at_rest_interaction_length()` is zero and a zero at-rest length pre-empts. The wiring itself
/// is P15's; this note is here so that the call site is not guessed at.
///
/// `ftf_invoke` is a callable `(const HadProjectile&, const HadNucleus&, HadFinalState&, Rng&)`
/// returning a `StoppingRefusal`, so that a caller without P11's FTF workspace does not have to
/// instantiate its templates. The anti-baryon arm is the only one that uses it.
///
/// `nuclear_mass` is `G4NucleiProperties::GetNuclearMass(A, Z)` in MeV, taken as a callable so
/// this header does not choose between P3's mass table and P6's.
template <typename real_t, int kCap, typename Rng, typename FtfInvoke, typename NuclearMassFn>
__host__ __device__ inline AtRestResult at_rest(
    const HadProjectile<real_t>& projectile, const MaterialComposition<real_t>& material,
    HadFinalState<real_t, kCap>& fs, HadFinalState<real_t, kCap>* nuclear_fs,
    const bert::CascadeParams& par,
    const bert::InterfaceLimits& lim, const BertiniArmState& bert_state,
    const data::LevelTable& lt, const deex::FermiPool& pool, const preco::PrecoWorkspace& pws,
    const NuclearMassFn& nuclear_mass, const FtfInvoke& ftf_invoke, int emc_model_id,
    int nc_model_id, int dio_model_id, Rng& rng) {
  AtRestResult r;
  fs.clear();
  fs.status = HadFinalStateStatus::kStopAndKill;
  fs.energy_change = real_t(0);

  r.arm = stopping_arm(projectile.pdg);
  if (r.arm == NuclearArm::kNone) {
    r.refusal = StoppingRefusal::kNotApplicable;
    return r;
  }

  // --- 1. Which atom, and which isotope of it. ------------------------------------------------
  const CaptureTarget tgt = select_z_and_a(material, rng);
  if (tgt.refusal != SelectorRefusal::kNone) {
    r.refusal = StoppingRefusal::kSelector;
    r.selector_refusal = tgt.refusal;
    return r;
  }
  r.z = tgt.z;
  r.a = tgt.a;
  r.element_index = tgt.element_index;
  const double m_nucleus = double(nuclear_mass(r.a, r.z));

  // --- 2. The atomic cascade. Its deposit is a BOUND ENERGY, not a deposit. --------------------
  EmCascadeResult emc;
  em_capture_cascade(r.z, m_nucleus, rng, emc);
  r.n_em_cascade = emc.n;
  r.e_bound_MeV = emc.e_bound;

  // --- 3. Bound decay, for the muon only. ------------------------------------------------------
  double cap_time = 0.0;
  BoundDecayResult bd;
  bool nuclear_capture = true;
  if (r.arm == NuclearArm::kMuonCapture) {
    r.bound_decay_ran = true;
    muon_bound_decay(r.z, r.a, m_nucleus, r.e_bound_MeV, 0.0, rng, bd);
    cap_time = bd.time_ns;
    r.capture_time_ns = cap_time;
    if (bd.decayed) { nuclear_capture = false; }
  }
  r.decayed_in_orbit = !nuclear_capture;

  // The EM cascade's secondaries first, then the bound decay's, then the nuclear model's - the
  // order `result->AddSecondaries` builds and the order the creator-model labels follow.
  for (int i = 0; i < emc.n; ++i) {
    HadSecondary<real_t> s;
    s.pdg = emc.p[i].pdg;
    s.z = (emc.p[i].pdg == 11) ? -1 : 0;
    s.a = 0;
    s.mass = real_t((emc.p[i].pdg == 11) ? em_cascade_electron_mass() : 0.0);
    s.kin_energy = real_t(emc.p[i].kin_energy);
    s.direction = Vec3<real_t>{real_t(emc.p[i].direction.x), real_t(emc.p[i].direction.y),
                               real_t(emc.p[i].direction.z)};
    s.creator_model_id = emc_model_id;
    if (!fs.add_secondary(s)) { r.refusal = StoppingRefusal::kSecondaryOverflow; return r; }
  }
  if (r.bound_decay_ran && bd.decayed) {
    for (int i = 0; i < bd.n; ++i) {
      HadSecondary<real_t> s;
      s.pdg = bd.pdg[i];
      s.z = (bd.pdg[i] == 11) ? -1 : 0;
      s.a = 0;
      s.mass = real_t((bd.pdg[i] == 11) ? em_cascade_electron_mass() : 0.0);
      s.kin_energy = real_t(bd.kin_energy[i]);
      s.direction = Vec3<real_t>{real_t(bd.direction[i].x), real_t(bd.direction[i].y),
                                 real_t(bd.direction[i].z)};
      s.creator_model_id = dio_model_id;
      if (!fs.add_secondary(s)) { r.refusal = StoppingRefusal::kSecondaryOverflow; return r; }
    }
  }

  if (!nuclear_capture) {
    // The muon decayed in orbit. Nothing else runs and nothing is deposited.
    return r;
  }

  // --- 4 and 5. The nuclear model, at rest. ----------------------------------------------------
  // `thePro` with its kinetic energy zero and its bound energy set - P5's `HadProjectile` already
  // carries `bound_energy`, which is exactly `G4HadProjectile::SetBoundEnergy`, and the muon's
  // bound decay above read it from the same place.
  HadProjectile<real_t> stopped = projectile;
  stopped.kin_energy = real_t(0);      // at rest, by definition of this process
  stopped.bound_energy = real_t(r.e_bound_MeV);
  HadNucleus nuc;
  nuc.a = r.a;
  nuc.z = r.z;

  const int n_before = fs.n_secondaries;
  if (r.arm == NuclearArm::kFritiof) {
    r.refusal = ftf_invoke(stopped, nuc, fs, rng);
    if (r.refusal != StoppingRefusal::kNone) { return r; }
  } else {
    // Bertini, with the de-excitation the instance was BUILT with - see the header.
    // `nucfs` is the CALLER's second buffer and not a local, and the reason is measured: a
    // `HadFinalState<double, 256>` is 256 HadSecondary structs, and as a local here it put
    // **34,000 bytes** on the device probe's stack frame against 11,872 for the whole Bertini
    // entry point it wraps. `bert::apply_yourself` begins with `fs.clear()`, so the nuclear
    // model cannot be pointed at the same buffer that already holds the atomic cascade's
    // electrons and gammas - hence two, and hence this parameter rather than a temporary.
    HadFinalState<real_t, kCap>& nucfs = *nuclear_fs;
    r.bertini = bert::apply_yourself(
        stopped, nuc, nucfs, stopping_deexcite_choice(r.arm), par, lim, *bert_state.model,
        *bert_state.global_out, *bert_state.out, *bert_state.dex_out, *bert_state.tmp,
        *bert_state.epo, *bert_state.ws, lt, pool, pws, nc_model_id, rng);
    if (r.bertini.refusal != bert::InterfaceRefusal::kNone) {
      r.refusal = StoppingRefusal::kBertiniRefused;
      return r;
    }
    r.local_deposit_MeV = double(nucfs.local_energy_deposit);
    for (int i = 0; i < nucfs.n_secondaries; ++i) {
      if (!fs.add_secondary(nucfs.secondaries[i])) {
        r.refusal = StoppingRefusal::kSecondaryOverflow;
        return r;
      }
    }
  }
  r.reentry_count = 1;   // one ApplyYourself; CheckResult never returns null in this port

  // Every NUCLEAR secondary carries the capture delay. The EM cascade's and the bound decay's do
  // not - they happened at the top of the step.
  for (int i = n_before; i < fs.n_secondaries; ++i) {
    fs.secondaries[i].time = real_t(double(fs.secondaries[i].time) + cap_time);
  }
  return r;
}

}  // namespace g4gpu::physics::hadronic::stopping

#endif  // G4GPU_STOPPING_STOPPING_PROCESS_CUH
