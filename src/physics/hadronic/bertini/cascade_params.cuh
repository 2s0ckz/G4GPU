// G4CascadeParameters: the twenty numbers that configure the whole Bertini cascade.
//
// Transcribed from Geant4 11.1.1:
//   G4CascadeParameters::Initialize   (processes/hadronic/models/cascade/cascade/src/
//                                      G4CascadeParameters.cc)
//   G4HadronInelasticQBBC::ConstructProcess  (physics_lists/constructors/hadron_inelastic/src/)
//   G4HadronicBuilder::BuildFTFP_BERT        (physics_lists/builders/src/G4HadronicBuilder.cc)
//
// Every value below is checked against `ref/oracle/bertini_params.csv`, which reads them out of
// the install. That is not ceremony: three of them are not what the source reads like.
//
// **doCoalescence is TRUE by default.** Its initializer is
//     DO_COALESCENCE = (0==G4CASCADE_DO_COALESCENCE || G4CASCADE_DO_COALESCENCE[0]!='0');
// and the envvar pointer is null when the variable is unset, so the first disjunct is true. The
// line four above it,
//     USE_PRECOMPOUND = (0!=G4CASCADE_USE_PRECOMPOUND && G4CASCADE_USE_PRECOMPOUND[0]!='0');
// is the same shape with `&&` and defaults FALSE. The two flags are read from the same kind of
// unset envvar and come out opposite ways, and only the dump settles which is which.
//
// **Four of the nuclear-structure parameters do not come from this function at all.** When the
// envvar is unset, `RADIUS_SCALE`, `RADIUS_TRAILING`, `FERMI_SCALE` and `XSEC_SCALE` are
// overwritten by `HDP.DeveloperGet("BERT_...")` from G4HadronicDeveloperParameters, whose
// defaults a file-scope `BERTParameters` object in the same translation unit registers. Nothing
// else in the 11.1.1 tree writes those four HDP entries, so the values agree with the
// initializer - but the ARITHMETIC differs, because FERMI_SCALE is then multiplied by
// RADIUS_SCALE a second time: `1.932/OLD_RADIUS_UNITS * RADIUS_SCALE`. The dump says that comes
// out as 1.9319999999999999 and not 1.932, and radiusSmall's `(8.0/OLD_RADIUS_UNITS)*RADIUS_SCALE`
// comes out as exactly 8. Both are written here as the dumped doubles.
//
// **QBBC contains Bertini in two configurations and only one of them uses PreCompound.**
// `G4CascadeParameters::usePreCompound()` is false, so a bare G4CascadeInterface de-excites with
// its own G4NonEquilibriumEvaporator/G4EquilibriumEvaporator/G4Fissioner/G4BigBanger. But
// `G4HadronInelasticQBBC::ConstructProcess` calls `theBERT->usePreCompoundDeexcitation()`
// explicitly on both of the instances it builds, so for **p, n, pi+ and pi-** the tail is P6's
// `preco::` module through G4GeneratorPrecompoundInterface, not the cascade evaporators. For
// **K+, K-, K0L, K0S and the hyperons**, which come from `G4HadronicBuilder::BuildFTFP_BERT`
// with `bert = true`, that call is NOT made, so those use the cascade's own de-excitation. The
// same class, two tails, chosen by which physics-list builder constructed it. docs/RISK.md V118.
//
// The energy limits follow the same split, and the third column is what the builder omits:
//
//   p, n          Emin 1 GeV   Emax 6 GeV    PreCompound   (SetMinEnergy + SetMaxEnergy called)
//   pi+, pi-      Emin 1 GeV   Emax 12 GeV   PreCompound
//   K, hyperons   Emin 0       Emax 6 GeV    cascade       (only SetMaxEnergy called, so
//                                                           G4HadronicInteraction's
//                                                           theMinEnergy(0.0) stands)
#ifndef G4GPU_BERTINI_CASCADE_PARAMS_CUH
#define G4GPU_BERTINI_CASCADE_PARAMS_CUH

namespace g4gpu::physics::hadronic::bert {

/// Which de-excitation tail an instance uses. Not a preference: it is set by the physics-list
/// builder that constructed the interface, and the two builders QBBC uses disagree.
enum class DeexciteChoice { kCascade = 0, kPreCompound = 1 };

/// G4CascadeParameters, as the install answers. A struct rather than compile-time constants
/// because every one of them is settable from an environment variable at run time, and a port
/// that folded the defaults into the code could not reproduce a run that set one.
struct CascadeParams {
  // Top-level configuration flags
  int verbose = 0;
  bool check_conservation = false;   ///< CHECK_ECONS: gates G4CascadeCheckBalance inside the
                                     ///< nuclei model and the cascader, not the interface's own
  bool use_precompound = false;      ///< USE_PRECOMPOUND - see the header comment: QBBC
                                     ///< overrides it per instance rather than setting it
  bool do_coalescence = true;        ///< DO_COALESCENCE - defaults TRUE, unlike its neighbour
  bool show_history = false;
  bool use_3body_mom = false;        ///< USE_3BODYMOM: selects the 3-body momentum generators
  bool use_phase_space = false;      ///< USE_PHASESPACE: G4CascadeFinalStateAlgorithm's N-body
                                     ///< phase-space generator instead of the parametrisations
  double pin_absorption = 0.0;       ///< PIN_ABSORPTION, GeV: direct pi-N absorption above 0

  // Nuclear structure parameters. The four HDP-sourced ones carry the dumped double.
  bool use_two_param = false;        ///< TWOPARAM_RADIUS: R = 1.16 A^1/3 - 1.3456 A^-1/3
  double radius_scale = 2.8196666666666665;    ///< OLD_RADIUS_UNITS = 3.3836/1.2
  double radius_small = 8.0;                   ///< (8.0/OLD_RADIUS_UNITS)*radius_scale
  double radius_alpha = 0.69999999999999996;
  double radius_trailing = 0.0;                ///< 0 => the trailing effect is OFF
  double fermi_scale = 1.9319999999999999;     ///< (1.932/OLD_RADIUS_UNITS)*radius_scale
  double xsec_scale = 1.0;
  double gamma_qd_scale = 1.0;

  // Final-state clustering cuts, GeV/c
  double dp_max_doublet = 0.089999999999999997;
  double dp_max_triplet = 0.108;
  double dp_max_alpha = 0.115;
};

/// The defaults, as `ref/oracle/bertini_params.csv` reads them from the install.
__host__ __device__ inline CascadeParams default_cascade_params() { return CascadeParams(); }

/// G4CascadeInterface's own constants (its constructor, G4CascadeInterface.cc:145-157).
struct InterfaceLimits {
  /// SetEnergyMomentumCheckLevels(5*perCent, 10*MeV) - the levels G4HadronicProcess::CheckResult
  /// reads through GetFatalEnergyCheckLevels. Note that Bertini OVERRIDES
  /// G4HadronicInteraction's (2%, 1 GeV) default with (5%, 10 MeV), which is a far tighter
  /// absolute limit: 10 MeV of non-conservation on a 1 GeV projectile IS enough to trip it,
  /// where the base class's 1 GeV never could.
  double ep_relative = 0.050000000000000003;
  double ep_absolute_MeV = 10.0;
  /// balance->setLimits(5*perCent, 10*MeV/GeV) - G4CascadeCheckBalance's own limits, in
  /// Bertini's internal GeV.
  double balance_relative = 0.050000000000000003;
  double balance_absolute_GeV = 0.01;
  /// G4CascadeInterface::maximumTries, the retry loop's bound.
  int maximum_tries = 20;
};

__host__ __device__ inline InterfaceLimits default_interface_limits() {
  return InterfaceLimits();
}

/// The energy window and de-excitation tail QBBC gives one projectile species. MeV.
struct QbbcBertiniRange {
  double emin_MeV;
  double emax_MeV;
  DeexciteChoice deexcite;
};

/// G4HadronInelasticQBBC's two instances and G4HadronicBuilder::BuildFTFP_BERT's one, by PDG
/// code. `ok` is false for a species QBBC does not give Bertini at all.
struct QbbcBertiniLookup {
  QbbcBertiniRange range{0.0, 0.0, DeexciteChoice::kCascade};
  bool ok = false;
};

__host__ __device__ inline QbbcBertiniLookup qbbc_bertini_range(int pdg) {
  QbbcBertiniLookup r;
  switch (pdg) {
    case 2212:    // proton
    case 2112:    // neutron
      r.range = {1000.0, 6000.0, DeexciteChoice::kPreCompound};
      r.ok = true;
      return r;
    case 211:     // pi+
    case -211:    // pi-
      r.range = {1000.0, 12000.0, DeexciteChoice::kPreCompound};
      r.ok = true;
      return r;
    // G4HadParticles::sKaons is {321, -321, 310, 130} - K+, K-, K0S, K0L. The strong states
    // K0/anti-K0 (311, -311) are NOT in it: a physics list never registers a process on them,
    // and G4HadronicProcess::PostStepDoIt substitutes K0S/K0L for one that arrives.
    case 321:     // K+
    case -321:    // K-
    case 310:     // K0S
    case 130:     // K0L
    // G4HadParticles::sHyperons is {3122, 3222, 3112, 3322, 3312, 3334}. Sigma0 (3212) is NOT
    // in it - the comment beside the anti-hyperon list says why, "it decays very quickly" -
    // even though Bertini HAS a sigma0-nucleon channel table and the cascade emits sigma0s
    // internally. So a sigma0 that leaves the cascade is a transported particle with no
    // inelastic process, not a projectile Bertini is ever handed.
    case 3122:    // lambda
    case 3222:    // sigma+
    case 3112:    // sigma-
    case 3322:    // xi0
    case 3312:    // xi-
    case 3334:    // omega-
      r.range = {0.0, 6000.0, DeexciteChoice::kCascade};
      r.ok = true;
      return r;
    // Named, and named because the brief asks for them by name rather than by silence. An
    // ANTI-NUCLEON is a species Bertini declares itself applicable to in no sense at all: INUCL
    // has type codes for it (51 and 53) but neither 51 nor 53 is among the 34 initial states in
    // src/data/bertini_channels.hh, so `IsApplicable` is already false one level up. QBBC gives
    // an antiproton to FTFP and to CHIPS's annihilation, never to this model. The case is here
    // so that a reader looking for "what about antiprotons" finds an answer instead of a
    // default arm, and tests/test_bertini_apply.cu pins both halves.
    case -2212:   // anti-proton
    case -2112:   // anti-neutron
    case -1000010020:  // anti-deuteron, anti-triton, anti-He3, anti-alpha: G4HadronicBuilder
    case -1000010030:  // builds an FTFP/BERT chain for the light anti-nuclei and Bertini is
    case -1000020030:  // not in it
    case -1000020040:
      return r;   // ok = false, by name
    // A PHOTON is the other one worth naming: G4CascadeInterface IS applicable to it and the
    // gamma-nucleon channel tables exist, but QBBC routes photons to G4EmExtraPhysics'
    // gamma-nuclear (P13), not here. So "Bertini could do this" and "QBBC asks Bertini to do
    // this" differ for exactly one species, and this is it.
    case 22:
      return r;   // ok = false, by name
    default:
      return r;   // ok = false: QBBC registers no Bertini for this species
  }
}

}  // namespace g4gpu::physics::hadronic::bert

#endif  // G4GPU_BERTINI_CASCADE_PARAMS_CUH
