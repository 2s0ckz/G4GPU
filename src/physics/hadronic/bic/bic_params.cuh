// The binary cascade's constants: the energy windows QBBC gives it, the cuts that decide which
// secondaries are captured, and the energy-momentum check levels.
//
// Transcribed from G4BinaryCascade's constructor and `Propagate`, G4BinaryLightIonReaction's
// constructor and `ApplyYourself`, and the two physics-list constructors that configure them -
// G4HadronInelasticQBBC::ConstructProcess (physics_lists/constructors/hadron_inelastic) and
// G4IonPhysics::ConstructProcess (constructors/ions) - in 11.1.1.
//
// ## The energy windows, read from the constructors
//
// `G4BinaryCascade`'s own constructor sets 0 to **10.1 GeV**. QBBC then calls
// `theBIC->SetMaxEnergy(emaxBic)` with `emaxBic = 1.5*CLHEP::GeV`, a local in
// `ConstructProcess`, and never touches the minimum. So BIC runs from **0 to 1.5 GeV** in QBBC,
// and it is registered for **p, n, pi+ and pi- and for nothing else** - not for kaons and not
// for hyperons, which go to `G4HadronicBuilder::BuildKaonsFTFP_BERT` and
// `BuildHyperonsFTFP_BERT`. `G4BinaryCascade::ApplyYourself` enforces the same four species
// itself, with a `G4HadronicException` for anything else unless the environment variable
// `I_Am_G4BinaryCascade_Developer` is set.
//
// `G4BinaryLightIonReaction`'s constructor leaves G4HadronicInteraction's default 0 to
// **100 TeV**. `G4IonPhysics::ConstructProcess` then sets 0 to
// `GetMaxEnergyTransitionFTF_Cascade()` = **6 GeV**, for d, t, He3, alpha and GenericIon.
//
// **Those 6 GeV are PER NUCLEON and the 1.5 GeV are not.**
// `G4EnergyRangeManager::GetHadronicInteraction` divides the kinetic energy by
// `|GetBaryonNumber()|` when that is greater than 1 (G4EnergyRangeManager.cc:50-52), so the
// window a deuteron is selected in is 0 to 6 GeV/nucleon while a proton's is 0 to 1.5 GeV
// total. One `SetMaxEnergy` call, two meanings, decided by the projectile.
//
// ## The energy-momentum check: on for one model, off for the other
//
// `G4BinaryCascade`'s constructor calls `SetEnergyMomentumCheckLevels(1.0*perCent, 1.0*MeV)`.
// `G4BinaryLightIonReaction`'s does not, so it keeps `G4HadronicInteraction`'s default, which
// is `(DBL_MAX, DBL_MAX)` - the check is off. Both are dumped in `bic_limits.csv` and asserted
// on the oracle side, because the difference is not a rounding: the same secondary list would
// be reported as non-conserving by one model and accepted by the other.
//
// ## The cuts, and why they are checked by an extractor
//
// `theBCminP` (45 MeV), `theCutOnP` (90 MeV, lowered by nucleus mass) and `theCutOnPAbsorb`
// (0) are private members of G4BinaryCascade with no getters, no setters and no observable
// consequence a dump can reach: `theCutOnP` decides which nucleons `Capture()` moves into
// `theCapturedList`, and that list is not exposed. So there is no oracle these three can be
// compared against, and a test that compared the port's copy against a literal in the test
// would be comparing a copy with itself - docs/RISK.md V52's exact failure.
//
// They are checked instead by `tools/extract_bic_constants.pl`, which reads them out of the
// Geant4 SOURCE and asserts the set is exactly this one. That is a check that can fail: a
// release which changes 90 MeV to 80 MeV fails the extractor rather than being transcribed as
// though nothing had happened. docs/RISK.md V41 is where the form comes from.
#ifndef G4GPU_BIC_PARAMS_CUH
#define G4GPU_BIC_PARAMS_CUH

#include <cfloat>

#include "core/units.cuh"

namespace g4gpu::bic {

namespace u = g4gpu::units;

// ---------------------------------------------------------------------------------------------
// The energy windows
// ---------------------------------------------------------------------------------------------

/// G4BinaryCascade's constructor: `SetMinEnergy(0.0*GeV)`, `SetMaxEnergy(10.1*GeV)`.
__host__ __device__ inline constexpr double bic_ctor_min_energy() { return 0.0; }
__host__ __device__ inline constexpr double bic_ctor_max_energy() {
  return 10.1 * u::GeV<double>();
}

/// What QBBC leaves: `theBIC->SetMaxEnergy(1.5*CLHEP::GeV)`, the minimum untouched.
__host__ __device__ inline constexpr double bic_qbbc_min_energy() { return 0.0; }
__host__ __device__ inline constexpr double bic_qbbc_max_energy() {
  return 1.5 * u::GeV<double>();
}

/// G4BinaryLightIonReaction's constructor leaves G4HadronicInteraction's defaults, which are
/// `theMinEnergy = 0` and `theMaxEnergy = 100*TeV` = 1e8 MeV.
__host__ __device__ inline constexpr double blir_ctor_min_energy() { return 0.0; }
__host__ __device__ inline constexpr double blir_ctor_max_energy() {
  return 1.0e8 * u::MeV<double>();
}

/// What G4IonPhysics leaves: 0 to `GetMaxEnergyTransitionFTF_Cascade()` = 6 GeV, and those are
/// per NUCLEON for any projectile with |baryon number| > 1 - see the file header.
__host__ __device__ inline constexpr double blir_qbbc_min_energy() { return 0.0; }
__host__ __device__ inline constexpr double blir_qbbc_max_energy() {
  return 6.0 * u::GeV<double>();
}

/// `SetEnergyMomentumCheckLevels(1.0*perCent, 1.0*MeV)` in G4BinaryCascade's constructor.
__host__ __device__ inline constexpr double bic_ep_check_relative() { return 0.01; }
__host__ __device__ inline constexpr double bic_ep_check_absolute() {
  return 1.0 * u::MeV<double>();
}

/// G4BinaryLightIonReaction never sets them, so they stay at G4HadronicInteraction's
/// `epCheckLevels(DBL_MAX, DBL_MAX)` and the check is off. Not a transcription convenience: the
/// value is in the oracle and this test asserts the oracle still says so.
__host__ __device__ inline double blir_ep_check_relative() { return DBL_MAX; }
__host__ __device__ inline double blir_ep_check_absolute() { return DBL_MAX; }

/// `GetFatalEnergyCheckLevels()`, which G4HadronicInteraction returns as
/// `(2*perCent, 2*GeV)` - and which the oracle reports as (0.02, 1000 MeV), so the absolute
/// level is 1 GeV and not 2. Taken from the oracle, which is the authority.
__host__ __device__ inline constexpr double fatal_check_relative() { return 0.02; }
__host__ __device__ inline constexpr double fatal_check_absolute() {
  return 1000.0 * u::MeV<double>();
}

// ---------------------------------------------------------------------------------------------
// The cuts
// ---------------------------------------------------------------------------------------------

/// `theBCminP = 45*MeV`. `ApplyYourself`'s first statement is
///
///     if (initial4Momentum.e()-initial4Momentum.m() < theBCminP &&
///         (definition == neutron || definition == proton))
///       return theDeExcitation->ApplyYourself(aTrack, aNucleus);
///
/// so a nucleon below 45 MeV never enters the cascade at all - it goes straight to
/// `G4PreCompoundModel::ApplyYourself`, which is P6's `preco::apply_yourself_initial_fragment`
/// plus `preco::deexcite`. A PION below 45 MeV does enter the cascade, because the species test
/// is an `&&`. That asymmetry is the whole of the constant's content.
__host__ __device__ inline constexpr double bc_min_p() { return 45.0 * u::MeV<double>(); }

/// `theCutOnPAbsorb = 0*MeV`, with the comment "No Absorption of slow Mesons, other than above
/// G4MesonAbsorption". It is the threshold `G4Absorber`'s `WillBeAbsorbed` compares a meson's
/// momentum against, and at zero nothing is ever absorbed by that path.
__host__ __device__ inline constexpr double cut_on_p_absorb() { return 0.0; }

/// `theCutOnP`, as `Propagate` sets it from the nucleus mass:
///
///     theCutOnP = 90*MeV;
///     if (the3DNucleus->GetMass() >  30) theCutOnP = 70*MeV;
///     if (the3DNucleus->GetMass() >  60) theCutOnP = 50*MeV;
///     if (the3DNucleus->GetMass() > 120) theCutOnP = 45*MeV;
///
/// **The thresholds are compared against a MASS IN MeV, not a mass number.** `GetMass()` is
/// `Z m_p + (A-Z) m_n - BE` in MeV, which is above 120 for every nucleus including a single
/// neutron (939.6 MeV). So the first three branches are dead for every target that exists and
/// `theCutOnP` is **always 45 MeV**. The 30/60/120 that look like mass numbers - and would give
/// 90, 70, 50 and 45 MeV for A <= 30, A <= 60, A <= 120 and heavier - are being compared
/// against a number a thousand times larger. docs/RISK.md V72.
///
/// Reproduced as written, with the mass-number reading beside it, because "port Geant4" and
/// "fix Geant4" are different jobs. `cut_on_p_by_mass_number` is the other reading and is
/// called by nothing.
__host__ __device__ inline double cut_on_p(double nucleus_mass_mev) {
  double v = 90.0 * u::MeV<double>();
  if (nucleus_mass_mev > 30.0) { v = 70.0 * u::MeV<double>(); }
  if (nucleus_mass_mev > 60.0) { v = 50.0 * u::MeV<double>(); }
  if (nucleus_mass_mev > 120.0) { v = 45.0 * u::MeV<double>(); }
  return v;
}

/// What the four branches would give if the thresholds were mass numbers. Unused; see above.
__host__ __device__ inline double cut_on_p_by_mass_number(int a) {
  double v = 90.0 * u::MeV<double>();
  if (a > 30) { v = 70.0 * u::MeV<double>(); }
  if (a > 60) { v = 50.0 * u::MeV<double>(); }
  if (a > 120) { v = 45.0 * u::MeV<double>(); }
  return v;
}

/// `G4BinaryLightIonReaction::ApplyYourself`'s fusion threshold: `(mom.t()-mom.mag())/pA <
/// 50*MeV` sends the whole reaction to `FuseNucleiAndPrompound` - no cascade at all - so 50 MeV
/// per nucleon is where the ion arm changes model. The numerator is the projectile's kinetic
/// energy in the lab and `pA` its baryon number AFTER the projectile/target swap, so for a
/// heavy projectile on a light target it is the TARGET's mass number that divides.
__host__ __device__ inline constexpr double blir_fusion_threshold_per_nucleon() {
  return 50.0 * u::MeV<double>();
}

/// The two model ids `G4PhysicsModelCatalog::GetModelID` hands out, which every secondary
/// carries as its `SetCreatorModelID`. They are positions in a name list built at run time, not
/// compiled constants, so they are read out of the install rather than transcribed:
/// `ref/oracle/bic_limits.csv` carries both and `tests/test_bic_nucleus.cu` compares them. A
/// release that adds a model above these in the catalogue shifts them, and the test says so.
__host__ __device__ inline constexpr int bic_model_id() { return 23100; }
__host__ __device__ inline constexpr int blir_model_id() { return 23110; }

/// `G4GeneratorPrecompoundInterface`'s three constants are P6's and live in
/// `precompound/generator_interface.cuh`; they are named here only so that a reader looking for
/// "the cascade's constants" finds the cross-reference rather than a second copy.
__host__ __device__ inline constexpr int bic_constant_count() { return 7; }

}  // namespace g4gpu::bic

#endif
