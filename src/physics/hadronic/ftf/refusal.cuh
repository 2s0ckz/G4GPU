// How the FTFP package refuses a sub-case, and why every refusal is a named value.
//
// docs/HADRONIC_PLAN.md section 6 rule 4: a sub-case that is not done is refused with a
// message that names it, at the point it would have been needed. The same argument
// xs/refusal.cuh makes applies here and harder: the natural sentinel for "I could not
// fragment this string" is an empty hadron list, and an empty hadron list is ALSO what
// Geant4 returns when the fragmentation genuinely fails and the caller is expected to retry
// (G4VPartonStringModel::Scatter's `if (strings->empty()) { Success = false; continue; }`).
// So a refusal that returns nothing is indistinguishable from a legitimate retry request, in
// the direction that silently deletes an inelastic interaction.
//
// Every entry names a symbol that exists in 11.1.1, so the message can be grepped for in the
// tree tools/g4src.sh points at.
#pragma once
#include <cstdio>
#include <cstdlib>

namespace g4gpu::hadronic::ftf {

enum class FtfRefusal : int {
  kNone = 0,

  // ---- not in QBBC, refused by name rather than transcribed ----

  /// G4QGSModel / G4QGSParticipants and everything under parton_string/qgsm/. QBBC registers
  /// no QGSP builder - G4HadronInelasticQBBC gives FTFP the whole range above 3 GeV - so the
  /// quark-gluon-string model is not a gap in this port, it is out of scope. It is named here
  /// because G4GeneratorPrecompoundInterface::Propagate has a QGS *arm* which a wounded
  /// nucleus with off-shell nucleons selects (docs/RISK.md V50), and this package's hand-over
  /// has to say which arm it meant.
  kQGSModel,

  /// G4FTFTunings tune 1..9 (`baryon-tune2022-v0`, `pion-tune2022-v0`,
  /// `combined-tune2022-v0` and six dummies). Selectable only by the two UI commands
  /// `/process/had/models/ftf/selectTuneBy{Index,Name}`, and
  /// G4FTFTunings::fApplicabilityOfTunes is `{1,0,0,0,0,0,0,0,0,0}`, so
  /// GetIndexTune's scan over i = 1..9 finds nothing switched on and returns 0 for every
  /// projectile at every energy. ref/oracle/ftf_lund_params.csv dumps all ten states, so
  /// this is measured and not assumed. Tune 0 - the default set, which IS the tuned set,
  /// since every default comes from G4HadronicDeveloperParameters::SetDefault - is ported in
  /// full.
  kFtfTuneNonDefault,

  /// G4CRCoalescence. `G4HadronicParameters::EnableCRCoalescence()` is false by default, so
  /// G4TheoFSGenerator::ApplyYourself never constructs it. Dumped in
  /// ref/oracle/ftf_lund_params.csv.
  kCRCoalescence,

  /// G4QuasiElasticChannel. G4TheoFSGenerator's `theQuasielastic` is null unless
  /// SetQuasiElasticChannel is called, and no QBBC builder calls it for the FTFP model
  /// (G4HadronicBuilder and G4FTFBuilder construct G4TheoFSGenerator and set only the
  /// transport, the high-energy generator and the fragmentation). So the quasi-elastic
  /// branch of ApplyYourself is unreachable in QBBC and is refused rather than transcribed.
  kQuasiElasticChannel,

  // ---- species this package cannot be asked about ----

  /// A hyper-nucleus or anti-hyper-nucleus projectile - `IsHypernucleus()` /
  /// `IsAntiHypernucleus()`. P3 refuses the same path on the de-excitation side
  /// (docs/PORTED.md 2.1.3) and P1 does not transport one, so there is nothing downstream
  /// that could receive the lambda. G4TheoFSGenerator has a dummy elastic branch for one
  /// below 100 MeV and G4VPartonStringModel::Scatter has a whole residual-legality table for
  /// the projectile's lambdas; neither is ported.
  kHyperNucleus,

  /// A charm or bottom HADRON as the projectile. `EnableBCParticles` is 1 in 11.1.1, so QBBC
  /// does register FTFP_BERT for b- and c-hadrons, and G4TheoFSGenerator returns the primary
  /// unchanged below 100 MeV for one. They are reachable only as the projectile of a beam
  /// this port has no species for (P1's refused set), so the projectile case is refused.
  /// NOTE that charm and bottom hadrons as *products* are NOT refused: ProbCCbar = 2e-4 and
  /// ProbBBbar = 5e-5 in 11.1.1, so SampleQuarkFlavor draws a c or a b from the vacuum at
  /// that rate and G4HadronBuilder's substitution tables are reachable. They are ported.
  kHeavyFlavourProjectile,

  /// The Glauber-Gribov hadron-nucleon cross section under G4FTFParameters refused the
  /// projectile. In 11.1.1 that means one of G4HadronNucleonXsc's two unported branches -
  /// HyperonNucleonXscNS for an s/c/b hyperon or SCBMesonNucleonXscNS for an s/c/b meson,
  /// both `P` in docs/PORTED.md 2.1.1 - so a Lambda or a D meson gets no FTF parameters at
  /// all. The xs-level reason is carried alongside in FtfParameters::xs_refused, because
  /// "which of the two" is what a future agent needs.
  kHadronNucleonXscRefused,

  /// A projectile G4FTFParameters::InitForInteraction has no arm for and whose cross section
  /// came out zero, which its last `if (Xtotal == 0.0)` block silently replaces with a
  /// PROTON's. The port reports instead: substituting a proton for an unknown projectile is
  /// exactly the kind of plausible answer docs/HADRONIC_PLAN.md section 6 rule 4 forbids.
  kUndefinedProjectileNucleonAssumed,

  // ---- parts of the model not finished ----

  /// G4FTFModel::GetStrings and everything it drives - G4FTFParticipants' impact-parameter
  /// sampling, ReggeonCascade, PutOnMassShell, ExciteParticipants,
  /// G4DiffractiveExcitation::ExciteParticipants, G4ElasticHNScattering::ElasticScattering,
  /// BuildStrings, AdjustNucleons and GetResiduals. See the file header of ftf_model.cuh for
  /// what is and is not there.
  kFtfModelGetStrings,

  /// G4FTFAnnihilation - the five annihilation channels for an anti-baryon projectile.
  kFtfAnnihilation,

  /// G4Fancy3DNucleus / G4NuclearFermiDensity / G4FermiMomentum / G4Nucleon, the nucleus
  /// model this package shares with the binary cascade. P9 owns it (bic/nucleus/); this
  /// package is written against the thin contract in nucleus_interface.cuh and refuses when
  /// no nucleus implementation is bound.
  kNucleusModelNotBound,

  /// G4DiffractiveSplitableHadron's kinky-string arm and G4FTFParameters' Pt2Kink. Pt2kink is
  /// set to 0 in the G4FTFParameters constructor with the comment "To switch off kinky
  /// strings (bad results obtained with 6.0*GeV*GeV)", so no run reaches it; the zero is
  /// carried and the arm refused.
  kKinkyStrings,

  /// A DiQuark - AntiDiQuark pair handed to G4HadronBuilder::Build, which dispatches it to
  /// Barion() and throws "Illegal quark content as input" there. Reachable in Geant4 only by
  /// calling Build directly with two diquarks, which no caller does: Splitup dispatches on
  /// DecayIsQuark, and both DiQuarkSplitup arms build from a quark pair. A kernel cannot
  /// throw, so it is reported.
  kHadronBuilderIllegalContent,

  /// G4VLongitudinalStringDecay::SetMinimalStringMass's two "Illegal quark content as input"
  /// exceptions - two partons of the same sub-type with the same sign, or of different
  /// sub-types with opposite signs.
  kIllegalPartonPair,

  /// A hadron PDG code that the Meson / Baryon tables or G4HadronBuilder produced and
  /// data/ftf_hadrons.hh does not carry. In Geant4 FindParticle returns a null pointer and
  /// every caller has a branch for it; here the code is reported so that a missing table row
  /// cannot look like a legitimate null.
  kUnknownHadronCode,

  // ---- capacities, every one of them ----

  /// More hadrons from one string than the port's per-string list holds.
  kStringHadronCapacity,
  /// More strings from one interaction than the port's string list holds.
  kStringCapacity,
  /// More than 350 final states in SplitLast's FS_LeftHadron / FS_RightHadron / FS_Weight.
  /// Geant4 clamps NumberOf_FS to 349 with a JustWarning and keeps going; the port reports
  /// and clamps identically, so the numbers still match and the event is flagged.
  kFinalStateCapacity,
  /// More wounded nucleons than the hand-over's WoundedNucleus holds.
  kWoundedNucleonCapacity,

  // ---- loop guards, which in Geant4 are warnings and here are reports ----

  /// G4VPartonStringModel::Scatter's 1000 attempts exhausted. Geant4 raises a JustWarning and
  /// returns the primary unchanged as a single kinetic track; the port does the same and says
  /// so, because "the primary, unchanged" is also a legitimate elastic-looking final state.
  kScatterAttemptsExhausted,
  /// G4LundStringFragmentation::Loop_toFragmentString's StringLoopInterrupt (1000) attempts,
  /// or its inner 1000-iteration fragmentation loop.
  kFragmentLoopExhausted,
  /// G4ExcitedStringDecay::FragmentStrings' 100 attempts, or EnergyAndMomentumCorrector's 500
  /// scale iterations without reaching |Scale - 1| <= 1e-5.
  kEnergyCorrectorFailed,
};

__host__ __device__ inline const char* ftf_refusal_name(FtfRefusal r) {
  switch (r) {
    case FtfRefusal::kNone: return "(none)";
    case FtfRefusal::kQGSModel: return "G4QGSModel / G4QGSParticipants";
    case FtfRefusal::kFtfTuneNonDefault: return "G4FTFTunings tune index != 0";
    case FtfRefusal::kCRCoalescence: return "G4CRCoalescence";
    case FtfRefusal::kQuasiElasticChannel: return "G4QuasiElasticChannel";
    case FtfRefusal::kHyperNucleus: return "a hyper-nucleus or anti-hyper-nucleus projectile";
    case FtfRefusal::kHeavyFlavourProjectile: return "a charm or bottom hadron projectile";
    case FtfRefusal::kHadronNucleonXscRefused:
      return "G4HadronNucleonXsc refused the projectile (see FtfParameters::xs_refused)";
    case FtfRefusal::kUndefinedProjectileNucleonAssumed:
      return "G4FTFParameters::InitForInteraction's `Xtotal == 0` nucleon substitution";
    case FtfRefusal::kFtfModelGetStrings: return "G4FTFModel::GetStrings";
    case FtfRefusal::kFtfAnnihilation: return "G4FTFAnnihilation";
    case FtfRefusal::kNucleusModelNotBound:
      return "G4Fancy3DNucleus (P9's bic/nucleus/) is not bound";
    case FtfRefusal::kKinkyStrings: return "G4FTFParameters::Pt2Kink / a kinky string";
    case FtfRefusal::kHadronBuilderIllegalContent:
      return "G4HadronBuilder::Barion: Illegal quark content as input";
    case FtfRefusal::kIllegalPartonPair:
      return "G4VLongitudinalStringDecay::SetMinimalStringMass: Illegal quark content";
    case FtfRefusal::kUnknownHadronCode: return "a PDG code absent from data/ftf_hadrons.hh";
    case FtfRefusal::kStringHadronCapacity: return "hadrons-per-string capacity";
    case FtfRefusal::kStringCapacity: return "strings-per-interaction capacity";
    case FtfRefusal::kFinalStateCapacity: return "SplitLast's 350 final states";
    case FtfRefusal::kWoundedNucleonCapacity: return "wounded-nucleon capacity";
    case FtfRefusal::kScatterAttemptsExhausted:
      return "G4VPartonStringModel::Scatter's 1000 attempts";
    case FtfRefusal::kFragmentLoopExhausted:
      return "G4LundStringFragmentation::Loop_toFragmentString's loop limits";
    case FtfRefusal::kEnergyCorrectorFailed:
      return "G4ExcitedStringDecay::EnergyAndMomentumCorrector";
  }
  return "(unknown)";
}

/// The host-side end of a refusal, for a table builder or a test. Device callers read the
/// field; nothing in this package returns a plausible number in place of one.
inline void ftf_fatal_if_refused(FtfRefusal r, const char* where) {
  if (r != FtfRefusal::kNone) {
    std::fprintf(stderr, "FATAL: %s: FTFP refuses %s\n", where, ftf_refusal_name(r));
    std::exit(1);
  }
}

}  // namespace g4gpu::hadronic::ftf
