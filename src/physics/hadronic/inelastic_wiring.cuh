// P15's inelastic wiring: which (cross section, model list) each species' `*Inelastic` process
// has, and the energy windows the model choice is made over.
//
// The sibling of `elastic_wiring.cuh`, and the same division of labour: this file owns no
// physics. P2 has the cross sections (`xs/`), P5 has the framework (`process.cuh`, and in
// particular `choose_hadronic_interaction`, which IS `G4EnergyRangeManager::
// GetHadronicInteraction`), P9/P10/P11 have the final-state models. What P15 owns is the answer
// to "for THIS species at THIS energy, which of them" - and that answer is a table in three
// Geant4 constructors, transcribed below rather than inferred.
//
// ---------------------------------------------------------------------------------------------
// THE TABLE, and the three constructors it comes from (11.1.1)
//
// `G4HadronInelasticQBBC::ConstructProcess` (hadron_inelastic/src), lines 102-197:
//
//     eminFtf       = param->GetMinEnergyTransitionFTF_Cascade()   3 GeV
//     eminBert      = 1.0*CLHEP::GeV                               1 GeV
//     emaxBic       = 1.5*CLHEP::GeV                               1.5 GeV
//     emaxBert      = param->GetMaxEnergyTransitionFTF_Cascade()   6 GeV
//     emaxBertPions = 12.*CLHEP::GeV                               12 GeV
//     emax          = param->GetMaxEnergy()                        100 TeV
//
// and then, IN THIS REGISTRATION ORDER, because `G4EnergyRangeManager` keeps the LAST TWO
// matching models and the index it returns is an index into the registration list:
//
//     p, n        RegisterMe(theFTFP)  [3 GeV, 100 TeV]
//                 RegisterMe(theBERT)  [1 GeV, 6 GeV]      usePreCompoundDeexcitation()
//                 RegisterMe(theBIC)   [0, 1.5 GeV]        SetMaxEnergy only; min stays 0
//     pi+, pi-    RegisterMe(theFTFP)  [3 GeV, 100 TeV]
//                 RegisterMe(theBERT1) [1 GeV, 12 GeV]     usePreCompoundDeexcitation()
//                 RegisterMe(theBIC)   [0, 1.5 GeV]        the SAME theBIC object as p and n
//
// `G4HadronicBuilder::BuildFTFP_BERT(partList, bert, xsName)` (builders/src), lines 70-105 -
// which is what `BuildKaonsFTFP_BERT`, `BuildHyperonsFTFP_BERT` and `BuildAntiLightIonsFTFP`
// all call, differing only in `partList`, `bert` and `xsName`:
//
//     theModel->SetMaxEnergy(param->GetMaxEnergy());               always
//     if (bert) {
//       theCascade->SetMaxEnergy(param->GetMaxEnergyTransitionFTF_Cascade());
//       theModel->SetMinEnergy(param->GetMinEnergyTransitionFTF_Cascade());
//     }
//     RegisterMe(theModel); if (theCascade) RegisterMe(theCascade);
//
//   **THE `if (bert)` IS THE WHOLE OF THE ANTI-NUCLEON ROW.** With `bert` false nothing ever
//   calls `SetMinEnergy` on the FTFP instance, so its minimum stays at
//   `G4HadronicInteraction`'s default of ZERO and FTFP covers an antiproton at every energy,
//   including the 1 MeV one P12b's at-rest campaign measured. That is not a special case
//   anybody wrote; it is a line that is not there.
//
//     K+, K-              bert=true,  "Glauber-Gribov"  FTFP [3 GeV, 100 TeV] + BERT [0, 6 GeV]
//     hyperons            bert=true,  "Glauber-Gribov"  the same two
//     anti-hyperons       bert=false, "AntiAGlauber"    FTFP [0, 100 TeV] alone
//     pbar, nbar, anti-   bert=false, "AntiAGlauber"    FTFP [0, 100 TeV] alone
//       light-ions
//
//   Note the cascade's minimum is ZERO for the kaons and hyperons and 1 GeV for the nucleons
//   and pions: `BuildFTFP_BERT` never calls `SetMinEnergy` on `theCascade`, while
//   `G4HadronInelasticQBBC` does. A port that gave every Bertini instance the same window
//   would have a kaon below 1 GeV with no model at all.
//
// `G4IonPhysics::ConstructProcess` (ions/src), lines 103-137 - `G4IonPhysicsXS` overrides only
// `AddProcess`'s cross-section choice, not the models:
//
//     theIonBC = new G4BinaryLightIonReaction(thePreCompound);
//     theIonBC->SetMinEnergy(0.0);
//     theIonBC->SetMaxEnergy(GetMaxEnergyTransitionFTF_Cascade());     [0, 6 GeV]
//     theFTFP->SetMinEnergy(GetMinEnergyTransitionFTF_Cascade());      [3 GeV, 100 TeV]
//     RegisterMe(theIonBC); RegisterMe(theFTFP);                        <- BC FIRST here
//
//     d, t, He3, alpha    G4ParticleInelasticXS(part)
//     GenericIon          G4CrossSectionInelastic(G4ComponentGGNuclNuclXsc)
//
//   **AND FOR AN ION THE ENERGY COMPARED AGAINST THOSE WINDOWS IS PER NUCLEON.**
//   `G4EnergyRangeManager::GetHadronicInteraction` divides by `|baryon number|` whenever it is
//   greater than one, so a 6 GeV alpha is a 1.5 GeV/n projectile and gets the light-ion
//   reaction, not FTFP. P5's `choose_hadronic_interaction` carries that division; this file's
//   job is to hand it the right `baryon_number`, which for `kGenericIon` is the track's own A
//   and not G4GenericIon's placeholder.
//
// ---------------------------------------------------------------------------------------------
// WHAT THIS PORT CAN RUN, AND WHAT IT REFUSES - by arm, with the package that closes it
//
//   FTFP        `ftf::entry::apply`, PORTED 2.1.11/2.1.11b. Runs for nucleons, pions, kaons,
//               anti-nucleons and ions. Its own refusals travel out in `entry::Report`.
//   BERT        `bert::apply_yourself`, PORTED 2.1.12. Runs for nucleons, pions, kaons and
//               hyperons.
//   BIC         `bic::apply_yourself`, PORTED 2.1.10. Runs for nucleons and charged pions on
//               any target with A > 1; a HYDROGEN target is `Propagate1H1` and is refused by
//               name inside the model (`BicRefusal::hydrogen`).
//   BLIR        `bic::blir_apply_yourself`, PORTED 2.1.10. Runs the FUSION arm only - the
//               `(mom.t()-mom.mag())/pA < 50*MeV` branch of
//               `G4BinaryLightIonReaction::ApplyYourself` (line 119). At or above **50 MeV per
//               nucleon** Geant4 calls `Interact` and this port has no cascade there;
//               `BlirRefusal::cascade` names it and P9e is writing it.
//
//   THE ANTI-NUCLEON ROW HAS NO CROSS SECTION AT ALL, and that is a P2 refusal rather than a
//   P15 one. `BuildAntiLightIonsFTFP` passes `xsName = "AntiAGlauber"`, which
//   `G4HadProcesses::InelasticXS` turns into
//   `G4CrossSectionInelastic(new G4ComponentAntiNuclNuclearXS())` - the same component
//   `elastic_wiring.cuh` records as refused for the antiproton's `hadElastic`. So an
//   antiproton in this transport draws no inelastic interaction length either, and the gap is
//   structurally zero for the reason `HadronicRefusal::kAntiNucleusElastic`'s comment gives at
//   length: a cross section that is absent is not an interaction that could not be applied,
//   and booking it per step would count chances. The FINAL STATE for a stopped antiproton is a
//   different matter and P15 does wire it - see `stopping::at_rest`'s Fritiof arm.
//
// ---------------------------------------------------------------------------------------------
// WHAT IS UPLOADED
//
// Three of the five cross sections carry tables that `BuildPhysicsTable` fills on the host:
// `G4ParticleInelasticXS` for the proton and for each of d/t/He3/alpha (G4PARTICLEXS4.0's
// `proton/inel<Z>`, `deuteron/`, ... per-element and per-isotope files), `G4NeutronInelasticXS`
// for the neutron, and `G4BGGPionInelasticXS` for the two pions. The other two - the two
// Glauber-Gribov components for the kaons and for GenericIon - are closed form and compiled in.
// `host/inelastic_upload.cuh` builds and uploads the first three; `InelasticTables` is the view.
//
// Five `G4ParticleInelasticXS` data sets is five copies of a per-element/per-isotope table, and
// they are NOT interchangeable: `G4ParticleInelasticXS`'s constructor picks its high-energy
// component by particle name - Glauber-Gribov hadron-nucleus for a proton and Glauber-Gribov
// NUCL-NUCL for d, t, He3 and alpha (`pxs_high_energy` carries that branch) - and the data
// directory is per species. A port that loaded `proton/` once and scaled it would be wrong in
// both halves.
#pragma once

#include "core/particle.cuh"
#include "data/isotope_abundance.hh"
#include "data/materials.cuh"
#include "physics/hadronic/bertini/cascade_params.cuh"
#include "physics/hadronic/bic/bic_params.cuh"
#include "physics/hadronic/process.cuh"
#include "physics/hadronic/xs/bgg_pion_xs.cuh"
#include "physics/hadronic/xs/gg_hadron_nucleus_xsc.cuh"
#include "physics/hadronic/xs/gg_nucl_nucl_xsc.cuh"
#include "physics/hadronic/xs/particlexs.cuh"
#include "physics/hadronic/xs/sample_za.cuh"

namespace g4gpu::had {

namespace hp = g4gpu::physics::hadronic;
namespace hxs = g4gpu::hadronic::xs;

// =============================================================================================
// The parameters the three constructors read
// =============================================================================================

/// `G4HadronicParameters::GetMinEnergyTransitionFTF_Cascade()`, MeV. `ref/oracle/
/// hadronic_params.csv` is the authority and reads 3000; the default in
/// `G4HadronicParameters.cc` is `3*CLHEP::GeV` and QBBC does not move it.
template <typename real_t>
__host__ __device__ inline constexpr real_t ftf_transition_min() { return real_t(3000); }

/// `G4HadronicParameters::GetMaxEnergyTransitionFTF_Cascade()`, MeV: 6 GeV.
template <typename real_t>
__host__ __device__ inline constexpr real_t ftf_transition_max() { return real_t(6000); }

/// `G4HadronicParameters::GetMaxEnergy()`, MeV: 100 TeV. The ceiling of every FTFP instance and
/// of Bertini's own constructor - not 50 GeV, which is what reading `G4CascadeInterface`'s
/// header too fast gives (docs/RISK.md V136).
template <typename real_t>
__host__ __device__ inline constexpr real_t hadronic_max_energy() { return real_t(1e8); }

/// `G4HadronInelasticQBBC`'s `eminBert`, MeV. A literal `1.0*CLHEP::GeV` in that file and NOT a
/// `G4HadronicParameters` accessor, which is why it is spelled here and not read off one.
template <typename real_t>
__host__ __device__ inline constexpr real_t qbbc_bert_min() { return real_t(1000); }

/// `G4HadronInelasticQBBC`'s `emaxBertPions`, MeV: a literal `12.*CLHEP::GeV`.
template <typename real_t>
__host__ __device__ inline constexpr real_t qbbc_bert_max_pions() { return real_t(12000); }

// =============================================================================================
// Which cross section
// =============================================================================================

/// Which `*Inelastic` data set a species' process carries.
enum class InelasticChannel : int {
  /// No inelastic process on this species' manager at all: the leptons, and pi0 (which QBBC
  /// gives `G4Decay` and nothing else - it is in none of the three constructors' lists).
  kNone = 0,
  kParticleInelastic,  ///< p, d, t, He3, alpha - G4ParticleInelasticXS, one data set per species
  kNeutronInelastic,   ///< n - G4NeutronInelasticXS, inside G4NeutronGeneralProcess
  kBggPion,            ///< pi+, pi- - G4BGGPionInelasticXS
  kGgHadronNucleus,    ///< K+, K-, hyperons - G4CrossSectionInelastic(G4ComponentGGHadronNucleusXsc)
  kGgNuclNucl,         ///< GenericIon - G4CrossSectionInelastic(G4ComponentGGNuclNuclXsc)
  /// pbar, nbar, anti-hyperons, anti-light-ions - `G4CrossSectionInelastic(
  /// G4ComponentAntiNuclNuclearXS)`, which P2 refuses by name in `xs/refusal.cuh`. The same
  /// component and the same refusal as `ElasticChannel::kAntiNucleusRefused`.
  kAntiNucleusRefused,
};

__host__ __device__ inline InelasticChannel inelastic_channel(ParticleType t) {
  switch (t) {
    case ParticleType::kProton:
    case ParticleType::kDeuteron:
    case ParticleType::kTriton:
    case ParticleType::kHe3:
    case ParticleType::kAlpha:      return InelasticChannel::kParticleInelastic;
    case ParticleType::kNeutron:    return InelasticChannel::kNeutronInelastic;
    case ParticleType::kPionPlus:
    case ParticleType::kPionMinus:  return InelasticChannel::kBggPion;
    case ParticleType::kKaonPlus:
    case ParticleType::kKaonMinus:  return InelasticChannel::kGgHadronNucleus;
    case ParticleType::kGenericIon: return InelasticChannel::kGgNuclNucl;
    case ParticleType::kAntiProton: return InelasticChannel::kAntiNucleusRefused;
    default:                        return InelasticChannel::kNone;
  }
}

__host__ __device__ inline const char* inelastic_channel_name(InelasticChannel c) {
  switch (c) {
    case InelasticChannel::kNone:              return "no inelastic process";
    case InelasticChannel::kParticleInelastic: return "G4ParticleInelasticXS";
    case InelasticChannel::kNeutronInelastic:  return "G4NeutronInelasticXS";
    case InelasticChannel::kBggPion:           return "G4BGGPionInelasticXS";
    case InelasticChannel::kGgHadronNucleus:
      return "G4CrossSectionInelastic(G4ComponentGGHadronNucleusXsc)";
    case InelasticChannel::kGgNuclNucl:
      return "G4CrossSectionInelastic(G4ComponentGGNuclNuclXsc)";
    case InelasticChannel::kAntiNucleusRefused:
      return "G4CrossSectionInelastic(G4ComponentAntiNuclNuclearXS) - refused by name (P2)";
  }
  return "unknown";
}

/// Does this channel have a cross section this port can evaluate? Two answers collapse to "no"
/// and they are different statements - see `elastic_channel_has_xs`, which says the same thing
/// about the same two kinds of gap.
__host__ __device__ inline bool inelastic_channel_has_xs(InelasticChannel c) {
  return c != InelasticChannel::kNone && c != InelasticChannel::kAntiNucleusRefused;
}

// =============================================================================================
// Which model
// =============================================================================================

/// One registered `G4HadronicInteraction`, as this port has it.
enum class InelasticModel : int {
  kNone = 0,
  kFtfp,     ///< G4TheoFSGenerator("FTFP") - `ftf::entry::apply`
  kBertini,  ///< G4CascadeInterface - `bert::apply_yourself`
  kBinary,   ///< G4BinaryCascade - `bic::apply_yourself`
  kLightIon, ///< G4BinaryLightIonReaction - `bic::blir_apply_yourself`
};

__host__ __device__ inline const char* inelastic_model_name(InelasticModel m) {
  switch (m) {
    case InelasticModel::kNone:     return "(none)";
    case InelasticModel::kFtfp:     return "FTFP";
    case InelasticModel::kBertini:  return "Bertini";
    case InelasticModel::kBinary:   return "BinaryCascade";
    case InelasticModel::kLightIon: return "BinaryLightIonReaction";
  }
  return "unknown";
}

/// The most models any one species has registered. Three (p, n, pi+-); everything else has two
/// or one. A compile-time bound because `choose_hadronic_interaction` takes an array.
inline constexpr int kMaxInelasticModels = 3;

/// The registered model list of one species, in `RegisterMe` order.
///
/// THE ORDER IS PART OF THE ANSWER and not a presentation choice.
/// `G4EnergyRangeManager::GetHadronicInteraction` walks the list forwards and remembers only the
/// LAST TWO models whose window contains the energy; with three registered and two matching, the
/// pair it keeps and the index it returns both depend on the order. Reproducing the order is
/// what makes `choose_hadronic_interaction`'s answer the same index Geant4 would have used.
template <typename real_t>
struct InelasticModelList {
  int n = 0;
  hp::ModelRange<real_t> range[kMaxInelasticModels];
  InelasticModel model[kMaxInelasticModels] = {InelasticModel::kNone, InelasticModel::kNone,
                                               InelasticModel::kNone};
};

/// The model list QBBC gives @p t, in registration order and with the windows spelled above.
///
/// Energies are MeV and are compared against the projectile's kinetic energy PER NUCLEON for
/// |baryon number| > 1 - the division is inside `choose_hadronic_interaction`, not here, so
/// these numbers are the ones the three constructors write.
template <typename real_t>
__host__ __device__ inline InelasticModelList<real_t> inelastic_models(ParticleType t) {
  InelasticModelList<real_t> L;
  auto add = [&L](InelasticModel m, real_t lo, real_t hi) {
    L.model[L.n] = m;
    L.range[L.n].min_energy = lo;
    L.range[L.n].max_energy = hi;
    L.range[L.n].applicable = true;
    ++L.n;
  };
  switch (t) {
    // G4HadronInelasticQBBC: FTFP, BERT, BIC - in that order.
    case ParticleType::kProton:
    case ParticleType::kNeutron:
      add(InelasticModel::kFtfp, ftf_transition_min<real_t>(), hadronic_max_energy<real_t>());
      add(InelasticModel::kBertini, qbbc_bert_min<real_t>(), ftf_transition_max<real_t>());
      add(InelasticModel::kBinary, real_t(bic::bic_qbbc_min_energy()),
          real_t(bic::bic_qbbc_max_energy()));
      return L;
    case ParticleType::kPionPlus:
    case ParticleType::kPionMinus:
      add(InelasticModel::kFtfp, ftf_transition_min<real_t>(), hadronic_max_energy<real_t>());
      add(InelasticModel::kBertini, qbbc_bert_min<real_t>(), qbbc_bert_max_pions<real_t>());
      add(InelasticModel::kBinary, real_t(bic::bic_qbbc_min_energy()),
          real_t(bic::bic_qbbc_max_energy()));
      return L;
    // G4HadronicBuilder::BuildFTFP_BERT with bert = true: FTFP then the cascade, and the
    // cascade's MINIMUM is zero because nothing calls SetMinEnergy on it there.
    case ParticleType::kKaonPlus:
    case ParticleType::kKaonMinus:
      add(InelasticModel::kFtfp, ftf_transition_min<real_t>(), hadronic_max_energy<real_t>());
      add(InelasticModel::kBertini, real_t(0), ftf_transition_max<real_t>());
      return L;
    // BuildAntiLightIonsFTFP, bert = false: one model, and its minimum is G4HadronicInteraction's
    // own zero. `choose_hadronic_interaction`'s `n_models == 1` shortcut then never looks at the
    // window at all, which is Geant4's behaviour and is why this row cannot be got wrong by
    // getting the window wrong - only by registering a second model that is not there.
    case ParticleType::kAntiProton:
      add(InelasticModel::kFtfp, real_t(0), hadronic_max_energy<real_t>());
      return L;
    // G4IonPhysics: the light-ion reaction FIRST, then FTFP.
    case ParticleType::kDeuteron:
    case ParticleType::kTriton:
    case ParticleType::kHe3:
    case ParticleType::kAlpha:
    case ParticleType::kGenericIon:
      add(InelasticModel::kLightIon, real_t(bic::blir_qbbc_min_energy()),
          real_t(bic::blir_qbbc_max_energy()));
      add(InelasticModel::kFtfp, ftf_transition_min<real_t>(), hadronic_max_energy<real_t>());
      return L;
    default:
      return L;  // n == 0: no inelastic process
  }
}

/// `G4EnergyRangeManager::GetHadronicInteraction` for this species at this energy.
///
/// @param baryon_number  the projectile's, so the per-nucleon division happens. For
///        `kGenericIon` this is the TRACK's mass number and not G4GenericIon's placeholder;
///        passing 1 would give a 4 GeV carbon ion FTFP where Geant4 gives it the light-ion
///        reaction at 0.33 GeV/n.
/// @return `kNone` when the species has no process, or when no model covers the energy - in
///         which case `status` says which, and the caller must refuse by name rather than pick
///         the nearest. Geant4 prints its model table and returns nullptr there, and
///         `G4HadronicProcess::PostStepDoIt` then raises a FatalException.
template <typename real_t, typename Rng>
__host__ __device__ inline InelasticModel choose_inelastic_model(ParticleType t,
                                                                 real_t kin_energy,
                                                                 int baryon_number, Rng& rng,
                                                                 hp::ModelChoice& status) {
  const InelasticModelList<real_t> L = inelastic_models<real_t>(t);
  if (L.n == 0) {
    status = hp::ModelChoice::kNoModelRegistered;
    return InelasticModel::kNone;
  }
  const hp::ModelSelection sel =
      hp::choose_hadronic_interaction<real_t>(L.range, L.n, kin_energy, baryon_number, rng);
  status = sel.status;
  if (sel.status != hp::ModelChoice::kOk || sel.index < 0) { return InelasticModel::kNone; }
  return L.model[sel.index];
}

/// True when this species' model choice CONSUMES A UNIFORM at this energy, which is the one
/// thing about the choice a caller cannot afford to get wrong.
///
/// A PREDICATE AND NOT A COMMENT, for the reason `decays_in_flight` gives: every existing dose
/// in this project is measured against a random stream, and a draw made where Geant4 makes none
/// moves every number downstream of it. `choose_hadronic_interaction` draws if and only if two
/// models match and their ranges are not nested; with one registered model it does not even
/// look at the window. This re-answers that question without running the choice, so a test can
/// compare the two and a caller can budget.
template <typename real_t>
__host__ __device__ inline bool inelastic_choice_draws(ParticleType t, real_t kin_energy,
                                                       int baryon_number) {
  const InelasticModelList<real_t> L = inelastic_models<real_t>(t);
  if (L.n <= 1) { return false; }
  real_t e = kin_energy;
  const int ab = (baryon_number < 0) ? -baryon_number : baryon_number;
  if (ab > 1) { e /= real_t(ab); }
  int cou = 0;
  real_t emi1 = 0, ema1 = 0, emi2 = 0, ema2 = 0;
  for (int i = 0; i < L.n; ++i) {
    if (!L.range[i].applicable) { continue; }
    const real_t lo = L.range[i].min_energy, hi = L.range[i].max_energy;
    if (lo <= e && hi >= e) {
      ++cou;
      emi2 = emi1; ema2 = ema1;
      emi1 = lo;   ema1 = hi;
    }
  }
  if (cou != 2) { return false; }
  // The nested test, which returns before the draw.
  return !((emi2 <= emi1 && ema2 >= ema1) || (emi2 >= emi1 && ema2 <= ema1));
}

// =============================================================================================
// The cross section
// =============================================================================================

/// The device tables the inelastic cross sections read. Null pointers mean "not uploaded", and
/// `inelastic_xs_per_volume` then returns zero for the channels that need one - which is the
/// same "no process" state a species with `kNone` is in, and is what a run whose
/// `G4PARTICLEXSDATA` could not be resolved sees.
template <typename real_t>
struct InelasticTables {
  /// `G4ParticleInelasticXS`, one data set per species: proton, deuteron, triton, He3, alpha.
  /// Indexed by `particle_inelastic_slot`.
  const hxs::PxsDataSet<real_t>* particle[5] = {nullptr, nullptr, nullptr, nullptr, nullptr};
  /// `G4NeutronInelasticXS` - the same data set `G4NeutronGeneralProcess`'s combined table was
  /// built from, uploaded so the sub-process can draw a target from it. `host/neutron_upload.
  /// cuh` builds it; before P15 it was built and thrown away, because the combined table was
  /// its only consumer.
  const hxs::PxsDataSet<real_t>* neutron = nullptr;
  /// `G4BGGPionInelasticXS`, is_elastic == false.
  const hxs::BggPionTable<real_t>* bgg_pion = nullptr;
};

/// Which slot of `InelasticTables::particle` a species uses. -1 for a species with no
/// `G4ParticleInelasticXS`.
__host__ __device__ inline int particle_inelastic_slot(ParticleType t) {
  switch (t) {
    case ParticleType::kProton:   return 0;
    case ParticleType::kDeuteron: return 1;
    case ParticleType::kTriton:   return 2;
    case ParticleType::kHe3:      return 3;
    case ParticleType::kAlpha:    return 4;
    default:                      return -1;
  }
}

/// The species as the cross sections ask about it, built through `xs/projectile.cuh`'s own
/// factories - see `elastic_projectile`, which says why that matters.
///
/// `kGenericIon` is the one that is not a constant: the Glauber-Gribov nucl-nucl component needs
/// the projectile's own (Z, A), so the caller passes the track's nuclide.
template <typename real_t>
__host__ __device__ inline hxs::Projectile<real_t> inelastic_projectile(ParticleType t, int z,
                                                                        int a) {
  switch (t) {
    case ParticleType::kProton:     return hxs::proton<real_t>();
    case ParticleType::kNeutron:    return hxs::neutron<real_t>();
    case ParticleType::kAntiProton: return hxs::anti_proton<real_t>();
    case ParticleType::kPionPlus:   return hxs::pi_plus<real_t>();
    case ParticleType::kPionMinus:  return hxs::pi_minus<real_t>();
    case ParticleType::kKaonPlus:   return hxs::kaon_plus<real_t>();
    case ParticleType::kKaonMinus:  return hxs::kaon_minus<real_t>();
    case ParticleType::kDeuteron:   return hxs::deuteron<real_t>();
    case ParticleType::kTriton:     return hxs::triton<real_t>();
    case ParticleType::kHe3:        return hxs::he3<real_t>();
    case ParticleType::kAlpha:      return hxs::alpha<real_t>();
    case ParticleType::kGenericIon: return hxs::generic_ion<real_t>(z, a);
    default:                        return hxs::Projectile<real_t>{};
  }
}

/// The three-function contract of `xs/sample_za.cuh` for one species at one energy.
///
/// `abundance_only` is NOT constant here, and that is the difference from the elastic side.
/// `G4ParticleInelasticXS` and `G4NeutronInelasticXS` DO override `SelectIsotope` - they carry
/// per-isotope files below 20 MeV (`pxs_iso_limit`) and their `SelectIsotope` draws against the
/// isotope cross sections there - while the two Glauber-Gribov components and
/// `G4BGGPionInelasticXS` do not. So the target draw's uniform COUNT differs by channel, and
/// `store_sample_za_rng` is told which through this predicate rather than through a constant.
template <typename real_t>
struct InelasticXsFn {
  const InelasticTables<real_t>* tables = nullptr;
  hxs::Projectile<real_t> proj{};
  InelasticChannel channel = InelasticChannel::kNone;
  int slot = -1;     ///< `particle_inelastic_slot`, for kParticleInelastic
  real_t ekin = 0;
  real_t loge = 0;   ///< log(ekin), as G4DynamicParticle::GetLogKineticEnergy supplies it

  __host__ __device__ const hxs::PxsDataSet<real_t>* pxs() const {
    if (channel == InelasticChannel::kNeutronInelastic) { return tables->neutron; }
    if (channel == InelasticChannel::kParticleInelastic && slot >= 0) {
      return tables->particle[slot];
    }
    return nullptr;
  }

  __host__ __device__ hxs::XsValue<real_t> element(int Z) const {
    switch (channel) {
      case InelasticChannel::kParticleInelastic:
      case InelasticChannel::kNeutronInelastic: {
        const hxs::PxsDataSet<real_t>* ds = pxs();
        if (ds == nullptr) { return {real_t(0), hxs::XsRefusal::kNone}; }
        return hxs::pxs_element_xs<real_t>(*ds, ekin, loge, Z);
      }
      case InelasticChannel::kBggPion:
        if (tables->bgg_pion == nullptr) { return {real_t(0), hxs::XsRefusal::kNone}; }
        return hxs::bgg_pion_element_xs<real_t>(*tables->bgg_pion, proj, ekin, Z);
      case InelasticChannel::kGgHadronNucleus:
        // G4CrossSectionInelastic::GetElementCrossSection passes `nist->GetAtomicMassAmu(Z)`,
        // the NIST mean atomic mass - not G4IsotopeList's aeff[Z]. The same distinction the
        // elastic side records, and the same fifth digit.
        return hxs::ggh_inelastic_element<real_t>(proj, ekin, Z, data::atomic_mass<real_t>(Z));
      case InelasticChannel::kGgNuclNucl:
        return hxs::ggnn_inelastic_element<real_t>(proj, ekin, Z, data::atomic_mass<real_t>(Z));
      case InelasticChannel::kNone:
      case InelasticChannel::kAntiNucleusRefused:
        break;
    }
    return {real_t(0), hxs::XsRefusal::kNone};
  }

  __host__ __device__ hxs::XsValue<real_t> isotope(int Z, int A) const {
    const hxs::PxsDataSet<real_t>* ds = pxs();
    if (ds == nullptr) { return element(Z); }
    return hxs::pxs_iso_xs<real_t>(*ds, ekin, loge, Z, A);
  }

  /// `G4VCrossSectionDataSet::SelectIsotope`'s two branches, as the data set decides them.
  ///
  /// The two G4PARTICLEXS classes answer from their isotope vectors below `elimit` = 20 MeV and
  /// fall back to the abundance draw above it, which is exactly the shape
  /// `G4ParticleInelasticXS::SelectIsotope` has. Everything else has no isotope data at all.
  __host__ __device__ bool abundance_only(int /*Z*/) const {
    if (channel != InelasticChannel::kParticleInelastic
        && channel != InelasticChannel::kNeutronInelastic) {
      return true;
    }
    return !(ekin < hxs::pxs_iso_limit<real_t>(
                        (channel == InelasticChannel::kNeutronInelastic)
                            ? hxs::PxsKind::kNeutronInelastic
                            : hxs::PxsKind::kParticleInelastic));
  }
};

template <typename real_t>
__host__ __device__ inline InelasticXsFn<real_t> inelastic_xs_fn(const InelasticTables<real_t>& t,
                                                                  ParticleType type, real_t ekin,
                                                                  int z, int a) {
  InelasticXsFn<real_t> fn;
  fn.tables = &t;
  fn.proj = inelastic_projectile<real_t>(type, z, a);
  fn.channel = inelastic_channel(type);
  fn.slot = particle_inelastic_slot(type);
  fn.ekin = ekin;
  fn.loge = (ekin > real_t(0)) ? log(ekin) : real_t(0);
  return fn;
}

/// `*Inelastic`'s macroscopic cross section, 1/mm, and the cumulative partial sums the target
/// draw needs. Zero when the species has no such process, and zero when it has one this port
/// cannot evaluate (the antiproton).
///
/// `__noinline__` for exactly the reason `elastic_xs_per_volume` gives, and with more to drag
/// in: inlined it pulls G4ParticleInelasticXS's table walk, G4BGGPionInelasticXS,
/// G4UPiNuclearCrossSection, Barashenkov, G4HadronNucleonXsc and BOTH Glauber-Gribov components
/// into `run_step_hadron`'s body once per species, on top of the elastic ones already there.
/// docs/RISK.md V55 is the entry about what that costs.
template <typename real_t>
__host__ __device__ __noinline__ real_t inelastic_xs_per_volume(
    const InelasticTables<real_t>& t, const data::Material<real_t>& mat, ParticleType type,
    real_t ekin, int z, int a, hxs::MaterialXs<real_t>& mxs) {
  const InelasticChannel c = inelastic_channel(type);
  if (!inelastic_channel_has_xs(c)) {
    mxs.total = real_t(0);
    mxs.n_elements = 0;
    return real_t(0);
  }
  const InelasticXsFn<real_t> fn = inelastic_xs_fn<real_t>(t, type, ekin, z, a);
  const hxs::XsValue<real_t> v = hxs::store_compute_cross_section_fn<real_t>(
      fn, mat, hxs::nist_isotopes_of<real_t>(mat), mxs);
  return (v.value > real_t(0)) ? v.value : real_t(0);
}

/// `G4CrossSectionDataStore::SampleZandA`, with Geant4's draw order and Geant4's draw COUNT.
/// See `elastic_sample_target`; the only difference is that `abundance_only` is a real question
/// here, because the two G4PARTICLEXS data sets have isotope data below 20 MeV.
template <typename real_t, typename Rng>
__host__ __device__ inline hxs::TargetZA inelastic_sample_target(
    const InelasticTables<real_t>& t, const data::Material<real_t>& mat, ParticleType type,
    real_t ekin, int z, int a, const hxs::MaterialXs<real_t>& mxs, Rng& rng) {
  const InelasticXsFn<real_t> fn = inelastic_xs_fn<real_t>(t, type, ekin, z, a);
  return hxs::store_sample_za_rng<real_t>(fn, mat, hxs::nist_isotopes_of<real_t>(mat), mxs,
                                          rng);
}

/// `G4HadronicProcess`'s integral-approach flag for an inelastic process.
///
/// The SAME six arguments `elastic_apply` passes, written through the same projectile, because
/// `G4HadronicProcess::GetXSType` does not know which process it is on: it reads the PARTICLE's
/// charge, mass and atomic number, so `hadElastic` and `protonInelastic` get the same
/// `fHadTwoPeaks` for a proton. Mirrored rather than re-derived - the elastic side's call is the
/// validated one (`tests/test_step_hadron.cu` rejects 27 of 132 direct calls with it).
template <typename real_t>
__host__ __device__ inline hp::HadXsType inelastic_xs_type(const hxs::Projectile<real_t>& pj) {
  return hp::hadronic_xs_type<real_t>(
      pj.pdg, pj.charge, pj.mass, (pj.baryon_number > 1) ? pj.baryon_number : 0, false, true);
}

}  // namespace g4gpu::had
