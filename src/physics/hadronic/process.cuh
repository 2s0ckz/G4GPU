// The hadronic process framework: what happens around a final-state model.
//
// Transcribed from Geant4 11.1.1:
//   G4HadronicProcess::PostStepDoIt / FillResult / CheckResult /
//     CheckEnergyMomentumConservation   (processes/hadronic/management/src/G4HadronicProcess.cc)
//   G4EnergyRangeManager::GetHadronicInteraction
//                                       (processes/hadronic/management/src/G4EnergyRangeManager.cc)
//   G4CrossSectionDataStore::ComputeCrossSection / SampleZandA
//                                       (processes/hadronic/cross_sections/src/G4CrossSectionDataStore.cc)
//   G4HadProjectile::Initialise         (processes/hadronic/util/src/G4HadProjectile.cc)
//   G4HadFinalState                     (processes/hadronic/util/src/G4HadFinalState.cc)
//   G4HadronicInteraction               (processes/hadronic/management/src/G4HadronicInteraction.cc)
//
// This file is the part every hadronic process shares. `elastic/elastic_process.cuh` has the
// elastic process's own PostStepDoIt, which is NOT this one - read the comment there.
//
// ---------------------------------------------------------------------------------------------
// The cross sections come in through a functor, not through this file.
//
// Package P2 owns the data classes (G4BGGNucleonElasticXS, G4NeutronElasticXS, ...). What this
// framework needs of them is three numbers - a per-element cross section, a per-isotope one, and
// whether the data set answers per element at all - so it takes them as a small interface
// (`XsFunctor` below) and never names a data class. `barashenkov_xs.cuh` satisfies it today.
//
// ---------------------------------------------------------------------------------------------
// The energy-momentum check, as 11.1.1 configures it by default.
//
// There are two separate checks in G4HadronicProcess and they have different defaults.
//
//   CheckResult, which RE-SAMPLES a bad interaction, is called unconditionally from the generic
//   PostStepDoIt. Its thresholds come from G4HadronicInteraction::GetFatalEnergyCheckLevels(),
//   whose default (G4HadronicInteraction.cc:210) is (relative 2%, absolute 1 GeV), and BOTH must
//   be exceeded - `|dE| > 1 GeV && |dE| > 0.02*Ekin` - before the result is thrown away. So at
//   a few hundred MeV nothing can trip it: 1 GeV of non-conservation is more than the whole
//   projectile. It also rejects a secondary whose dynamic mass is off its PDG mass by more than
//   `0.1*m + 1 MeV`.
//
//   CheckEnergyMomentumConservation, which only REPORTS, is gated on `epReportLevel != 0`.
//   `epReportLevel` is a G4HadronicProcess data member initialised to 0 (G4HadronicProcess.hh:227)
//   and is changed only by the environment variable G4Hadronic_epReportLevel, the
//   /process/had/verbose UI commands of G4HadronicEPTestMessenger, or an explicit
//   SetEpReportLevel. QBBC sets none of them, so **the default is 0: the report never runs**,
//   and with it `epCheckLevels` stays (DBL_MAX, DBL_MAX), which is the value that would make
//   every check pass anyway. There is no G4HadronicParameters::GetEpReportLevel in 11.1.1 - the
//   level lives on the process, not in the parameters singleton.
//
// Both are transcribed below because a port that only implemented the default would be a port of
// one configuration rather than of the class; `report_energy_momentum` returns its verdict rather
// than printing, so a test can assert on it.
//
// ---------------------------------------------------------------------------------------------
// What is deliberately not here.
//
//   - The kaon0/anti_kaon0 -> kaon0S/kaon0L 50/50 substitution in PostStepDoIt. It needs the K0
//     species, which is P1's refused set; `fill_result` reports it through `kaon0_seen` instead of
//     doing it silently, and no elastic model can produce a K0 anyway.
//   - `nICelectrons`: the count of PDG-11 secondaries, which the ep check uses to move electron
//     masses from the target to the final state. Counted, because an inelastic model with
//     internal conversion will need it.
//   - Cross-section biasing (`aScaleFactor`, XBiasSurvivalProbability). ApplyFactorXS is 0 in
//     11.1.1 (ref/oracle/hadronic_params.csv) and no QBBC constructor calls
//     BiasCrossSectionByFactor, so the factor is 1 and `scale_factor` below carries it
//     explicitly rather than being dropped.
#pragma once

#include <cmath>
#include <cstdint>

#include "core/units.cuh"
#include "core/vec3.cuh"

namespace g4gpu::physics::hadronic {

// =============================================================================================
// G4HadProjectile
// =============================================================================================

/// G4HadProjectile::Initialise / InitialiseLocal.
///
/// Geant4 keeps the four-momentum ROTATED INTO +z (`theMom.set(0,0,p,E)`) and remembers the
/// inverse rotation in `toLabFrame`. Every elastic model therefore works in a frame where the
/// projectile flies along +z, and the process rotates the answer back with `rotateUz(indir)`.
/// The LorentzRotation itself is not carried here: `SetTrafoToLab` exists for models that build
/// their own lab-frame secondaries, and no elastic model does - G4HadronElastic returns a
/// direction relative to +z and the process rotates it. If an inelastic model needs the rotation
/// it is `rotate_uz` about the track's own direction, which the process already has.
///
/// `theBoundEnergy` is 0 for a track (only the neutron-HP models set it) and `theTime` is 0 -
/// "time of interaction starts from zero, not global time of a track", the comment in
/// G4HadProjectile.cc. Both are kept so that a secondary's time is `max(secTime,0) + globalTime`
/// as FillResult has it.
template <typename real_t>
struct HadProjectile {
  int pdg = 0;
  int baryon_number = 0;      ///< for G4EnergyRangeManager's per-nucleon energy
  real_t charge = real_t(0);  ///< in units of eplus
  real_t mass = real_t(0);    ///< PDG mass, MeV
  real_t kin_energy = real_t(0);
  real_t bound_energy = real_t(0);

  /// |p| in the lab, MeV. Geant4 recomputes this as sqrt(T(T+2m)) rather than reading the
  /// track's momentum, so a track whose momentum drifted from its energy does not matter.
  __host__ __device__ real_t momentum() const {
    return sqrt(kin_energy * (kin_energy + real_t(2) * mass));
  }
  __host__ __device__ real_t total_energy() const { return mass + kin_energy; }
};

/// G4Nucleus, reduced to what the elastic models ask of it.
///
/// G4Nucleus carries (A, Z, L), an effective (aEff, zEff), a temperature, an excitation energy,
/// an accumulated momentum, and a Fermi momentum. Of those the elastic models read **only
/// GetA_asInt and GetZ_asInt**: G4HadronElastic, G4ChipsElasticModel, G4ElasticHadrNucleusHE and
/// G4NuclNuclDiffuseElastic never call GetThermalNucleus, GetFermiMomentum, Cinema or
/// EvaporationEffects. So there is no Fermi motion in QBBC's elastic scattering - the target is
/// at rest, and `G4HadronElastic::ApplyYourself` builds `G4LorentzVector lv(0,0,plab,e1+mass2)`
/// with the target contributing only its mass. The fields Geant4 has and this does not are
/// listed here rather than omitted, because an inelastic model WILL ask for them:
///   theL (hypernuclei), aEff/zEff (ChooseParameters for a compound material),
///   theTemp/GetThermalNucleus (target motion), fermiMomentum (GetFermiMomentum),
///   excitationEnergy + momentum (AddExcitationEnergy/AddMomentum, filled by a cascade),
///   pnBlackTrackEnergy and the dta/annihilation variants (the old Gheisha cascade).
/// L is kept at zero and asserted, not assumed.
struct HadNucleus {
  int z = 0;
  int a = 0;
  int l = 0;  ///< number of lambdas; hypernuclei are refused, see `select_isotope`
};

// =============================================================================================
// G4HadFinalState
// =============================================================================================

/// G4HadFinalStateStatus (G4HadFinalState.hh:42).
enum class HadFinalStateStatus { kIsAlive = 0, kStopAndKill = 1, kSuspend = 2 };

/// One secondary, as (PDG code or (Z,A)) plus a four-momentum.
///
/// Species enums belong to P1, so nothing here names a species: `pdg` is the PDG code when the
/// particle has one, and for a nucleus `z`/`a` carry (Z,A) with `pdg` set to the PDG nuclear code
/// 10LZZZAAA0 so that a consumer can use either. `mass` is the PDG mass the emitting model used,
/// because FillResult compares it against the definition's mass and puts the secondary back on
/// the mass shell if they differ by more than 1 keV.
template <typename real_t>
struct HadSecondary {
  int pdg = 0;
  int z = 0;
  int a = 0;
  real_t mass = real_t(0);
  real_t kin_energy = real_t(0);
  Vec3<real_t> direction{real_t(0), real_t(0), real_t(1)};
  real_t time = real_t(0);    ///< G4HadSecondary::GetTime, relative to the interaction
  real_t weight = real_t(1);  ///< G4HadSecondary::GetWeight
  int creator_model_id = -1;

  __host__ __device__ real_t total_energy() const { return mass + kin_energy; }
  __host__ __device__ real_t momentum() const {
    return sqrt(kin_energy * (kin_energy + real_t(2) * mass));
  }
};

/// PDG nuclear code, as G4IonTable::GetNucleusEncoding builds it: 10LZZZAAAI with I=0 for the
/// ground state and L the number of lambdas.
__host__ __device__ inline int pdg_nuclear_code(int z, int a, int l = 0) {
  return 1000000000 + l * 10000000 + z * 10000 + a * 10;
}

/// G4HadFinalState. Fixed capacity, because this runs on a device.
///
/// Overflow is never silent: `secondary_overflow` counts what did not fit, and every caller in
/// this package reports it. An elastic model emits at most one secondary, so `kMaxSecondaries`
/// is small here; an inelastic model at a few GeV emits tens and will instantiate it wider.
template <typename real_t, int kMaxSecondaries = 8>
struct HadFinalState {
  static constexpr int kCapacity = kMaxSecondaries;

  HadFinalStateStatus status = HadFinalStateStatus::kIsAlive;

  /// G4HadFinalState's constructor sets theEnergy = -1, and `SetEnergyChange` throws on a
  /// negative value. -1 therefore means "the model did not set it", which is distinct from 0
  /// ("the primary stopped"), and the process's `max(GetEnergyChange(), 0.0)` maps both to 0.
  real_t energy_change = real_t(-1);
  Vec3<real_t> momentum_change{real_t(0), real_t(0), real_t(1)};
  real_t local_energy_deposit = real_t(0);
  real_t weight = real_t(1);

  int n_secondaries = 0;
  int secondary_overflow = 0;
  HadSecondary<real_t> secondaries[kMaxSecondaries];

  __host__ __device__ void clear() {
    status = HadFinalStateStatus::kIsAlive;
    energy_change = real_t(-1);
    momentum_change = Vec3<real_t>{real_t(0), real_t(0), real_t(1)};
    local_energy_deposit = real_t(0);
    weight = real_t(1);
    n_secondaries = 0;
    secondary_overflow = 0;
  }

  /// G4HadFinalState::AddSecondary. Returns false on overflow, which the caller must report.
  __host__ __device__ bool add_secondary(const HadSecondary<real_t>& s) {
    if (n_secondaries >= kMaxSecondaries) { ++secondary_overflow; return false; }
    secondaries[n_secondaries++] = s;
    return true;
  }
};

// =============================================================================================
// G4CrossSectionDataStore::ComputeCrossSection and SampleZandA
// =============================================================================================

/// The material composition SampleZandA walks: elements, their atom densities, and their
/// isotopes with relative abundances.
///
/// This mirrors G4Material's own layout (GetElement(i), GetVecNbOfAtomsPerVolume(),
/// G4Element::GetIsotope(j) / GetRelativeAbundanceVector()) rather than inventing one, so that a
/// material built anywhere in this port can be handed over without a conversion.
template <typename real_t>
struct MaterialComposition {
  int n_elements = 0;
  const int* element_z = nullptr;
  /// Atoms per unit volume, per element. mm^-3 in this port's units.
  const real_t* n_atoms_per_volume = nullptr;
  /// Per element: how many isotopes, where its isotopes start in the flat arrays below, and
  /// whether the element carries Geant4's natural-abundance flag.
  const int* n_isotopes = nullptr;
  const int* isotope_offset = nullptr;
  const bool* natural_abundance = nullptr;
  /// Flat isotope arrays, indexed by isotope_offset[i] + j.
  const int* isotope_a = nullptr;
  const real_t* isotope_abundance = nullptr;
};

/// What this framework asks of P2's cross-section data.
///
/// `element_xs(z, kin, pdg)` is G4VCrossSectionDataSet::GetElementCrossSection.
/// `iso_xs(z, a, kin, pdg)` is GetIsoCrossSection.
/// `is_element_applicable(z, kin, pdg)` is IsElementApplicable - it decides which of the two
/// branches of GetCrossSection and SampleZandA is taken, and it is a real branch, not a
/// convenience: an element-wise data set never computes an isotope cross section at all and
/// selects the isotope through the data set's own SelectIsotope instead.
/// `select_isotope(...)` is G4VCrossSectionDataSet::SelectIsotope, whose default implementation
/// is abundance-weighted; a data set that overrides it (G4NeutronElasticXS does) returns
/// something else, which is why it is part of the interface and not done here.
///
/// A functor type has to provide these four; there is no default, so a data set that forgets one
/// fails to compile rather than falling back to a plausible number.
struct XsFunctorContract {};

/// G4CrossSectionDataStore::ComputeCrossSection.
///
/// Returns the macroscopic cross section (1/length) and fills `xsecelm` with the RUNNING SUM of
/// the per-element macroscopic cross sections - not the individual values. SampleZandA then
/// compares one uniform against that cumulative array, so the two functions are a pair and the
/// running sum is part of the contract, not an implementation choice.
///
/// `max(xs, 0.0)` per element is Geant4's: a data set that returns a negative number contributes
/// nothing rather than subtracting.
template <typename real_t, typename XsFunctor>
__host__ __device__ real_t compute_cross_section(const MaterialComposition<real_t>& mat,
                                                 const XsFunctor& xs, int pdg, real_t kin_energy,
                                                 real_t* xsecelm) {
  real_t total = real_t(0);
  for (int i = 0; i < mat.n_elements; ++i) {
    const int z = mat.element_z[i];
    real_t per_atom;
    // G4CrossSectionDataStore::GetCrossSection: the LAST data set is asked first, and an element
    // that carries the natural-abundance flag takes the element-wise branch. Otherwise the
    // abundance-weighted sum over isotopes is formed - Geant4 sums abundVector[j]*isoXS(j) even
    // when the data set has no isotope data, because GetIsoCrossSection then falls back to the
    // element cross section for each isotope, which sums the abundances to 1 and returns it.
    if (mat.natural_abundance[i] && xs.is_element_applicable(z, kin_energy, pdg)) {
      per_atom = xs.element_xs(z, kin_energy, pdg);
    } else {
      per_atom = real_t(0);
      const int off = mat.isotope_offset[i];
      for (int j = 0; j < mat.n_isotopes[i]; ++j) {
        per_atom += mat.isotope_abundance[off + j] *
                    xs.iso_xs(z, mat.isotope_a[off + j], kin_energy, pdg);
      }
    }
    const real_t contrib = mat.n_atoms_per_volume[i] * per_atom;
    total += (contrib > real_t(0)) ? contrib : real_t(0);
    xsecelm[i] = total;
  }
  return total;
}

/// The (element, isotope) SampleZandA chose.
struct ZandA {
  int element_index = 0;
  int z = 0;
  int a = 0;
};

/// G4CrossSectionDataStore::SampleZandA.
///
/// Two things about it are easy to get wrong and are both load-bearing.
///
/// First, the element loop uses the CUMULATIVE array left behind by a previous
/// `compute_cross_section` call for THIS particle, material and energy - it does not recompute
/// anything. The caller is responsible for that call having happened; `G4NeutronGeneralProcess`
/// does it explicitly (`fCurrentXSS->ComputeCrossSection(...)`) before delegating to a
/// sub-process, and only when the material has more than one element, because with one element
/// the loop is skipped entirely. Passing a stale `xsecelm` here selects an element by the wrong
/// energy's cross sections and nothing complains, so `total_xs` is taken as an argument rather
/// than read back out of the array.
///
/// Second, a single-element material returns element 0 WITHOUT drawing a random number. That is
/// not an optimisation: it changes the random sequence, and a port that always draws would
/// diverge from Geant4 on the very next sample in water.
template <typename real_t, typename XsFunctor, typename Rng>
__host__ __device__ ZandA sample_z_and_a(const MaterialComposition<real_t>& mat,
                                         const XsFunctor& xs, int pdg, real_t kin_energy,
                                         real_t total_xs, const real_t* xsecelm, Rng& rng,
                                         real_t* xseciso) {
  ZandA out;
  out.element_index = 0;
  if (mat.n_elements > 1) {
    const real_t cross = total_xs * rng.uniform();
    for (int i = 0; i < mat.n_elements; ++i) {
      if (cross <= xsecelm[i]) { out.element_index = i; break; }
    }
  }
  const int i = out.element_index;
  const int z = mat.element_z[i];
  out.z = z;

  const int n_iso = mat.n_isotopes[i];
  const int off = mat.isotope_offset[i];
  out.a = mat.isotope_a[off];  // Geant4: iso = anElement->GetIsotope(0) before either branch

  if (n_iso <= 1) { return out; }

  if (xs.is_element_applicable(z, kin_energy, pdg)) {
    // Element-wise cross section: the isotope cross section is NOT computed, and the data set's
    // own SelectIsotope decides. Note that Geant4 does not check the natural-abundance flag
    // here, unlike GetCrossSection - so an element with user-set abundances and an element-wise
    // data set has its cross section summed over isotopes but its isotope chosen by the data
    // set. Both branches are Geant4's; neither is a simplification of the other.
    out.a = xs.select_isotope(z, i, kin_energy, pdg, rng);
    return out;
  }

  // Isotope-wise: abundance times isotope cross section, cumulative, one uniform.
  real_t cross = real_t(0);
  for (int j = 0; j < n_iso; ++j) {
    real_t xsec = real_t(0);
    if (mat.isotope_abundance[off + j] > real_t(0)) {
      xsec = mat.isotope_abundance[off + j] *
             xs.iso_xs(z, mat.isotope_a[off + j], kin_energy, pdg);
    }
    cross += xsec;
    xseciso[j] = cross;
  }
  cross *= rng.uniform();
  for (int j = 0; j < n_iso; ++j) {
    if (cross <= xseciso[j]) { out.a = mat.isotope_a[off + j]; break; }
  }
  return out;
}

// =============================================================================================
// G4EnergyRangeManager::GetHadronicInteraction
// =============================================================================================

/// One registered model's applicability window, as G4HadronicInteraction reports it.
///
/// `min_energy`/`max_energy` are GetMinEnergy(material, element)/GetMaxEnergy(material, element),
/// which fall back to the model's own theMinEnergy/theMaxEnergy unless a per-material or
/// per-element override was registered. `applicable` is IsApplicable(projectile, nucleus).
template <typename real_t>
struct ModelRange {
  real_t min_energy = real_t(0);
  real_t max_energy = real_t(0);
  bool applicable = true;
};

/// Why the model choice failed, when it did. Never a silent default.
enum class ModelChoice {
  kOk = 0,
  kNoModelRegistered,      ///< 0 == theHadronicInteractionCounter
  kNoModelInRange,         ///< cou == 0: Geant4 prints the table and returns nullptr; the
                           ///  calling process then raises a FatalException
  kFullyOverlapping,       ///< cou == 2 and one range contains the other
  kMoreThanTwoCompeting    ///< cou > 2
};

struct ModelSelection {
  int index = -1;
  ModelChoice status = ModelChoice::kNoModelRegistered;
};

/// G4EnergyRangeManager::GetHadronicInteraction.
///
/// The overlap rule is the part that matters, and it is a physics-list decision rather than an
/// implementation detail: where two models both cover the energy, Geant4 picks one at random
/// with a probability that runs linearly across the overlap.
///
/// Reading the source's arithmetic out: the loop keeps only the LAST TWO matching models, in
/// `(emi1, ema1)` = the most recently found and `(emi2, ema2)` = the one before. With L the model
/// of lower `emin` and U the other, both branches test
///
///     (L.emax - E) < rand * (L.emax - U.emin)      ->  choose U
///
/// so P(U) = 1 - (L.emax - E)/(L.emax - U.emin) = (E - U.emin)/(L.emax - U.emin): zero at the
/// bottom of the overlap, one at the top. For QBBC's proton and neutron that gives
/// P(Bertini) = (E - 1 GeV)/0.5 GeV across the BIC/BERT overlap 1 - 1.5 GeV, and
/// P(FTFP) = (E - 3 GeV)/3 GeV across the BERT/FTFP overlap 3 - 6 GeV.
///
/// Two details that a rewrite loses. Only the last two matches are remembered, so with three
/// overlapping models Geant4 does not choose among three - it reports and returns nullptr
/// (cou > 2 falls to `default:`). And for an ion the energy compared against the ranges is
/// **per nucleon**, `Ekin/|A|`, for any |baryon number| > 1 - so a 6 GeV alpha is a 1.5 GeV/n
/// projectile as far as model selection is concerned.
///
/// One uniform is drawn if and only if two models compete and their ranges are not nested. A
/// port that draws unconditionally desynchronises the stream.
template <typename real_t, typename Rng>
__host__ __device__ ModelSelection choose_hadronic_interaction(
    const ModelRange<real_t>* models, int n_models, real_t kin_energy, int baryon_number,
    Rng& rng) {
  ModelSelection out;
  if (n_models == 1) {
    // Geant4's shortcut: with one registered model no range is even looked at, so a model whose
    // window excludes the energy is still used. G4HadronElasticPhysics registers exactly one
    // model per particle, so this is the branch QBBC's elastic takes for p, n, pi+-, the light
    // ions and GenericIon - the energy windows in the table below never come into it.
    out.index = 0;
    out.status = ModelChoice::kOk;
    return out;
  }
  if (n_models <= 0) {
    out.status = ModelChoice::kNoModelRegistered;
    return out;
  }

  real_t e = kin_energy;
  const int ab = (baryon_number < 0) ? -baryon_number : baryon_number;
  if (ab > 1) { e /= real_t(ab); }

  int cou = 0, memory = 0, memor2 = 0;
  real_t emi1 = real_t(0), ema1 = real_t(0), emi2 = real_t(0), ema2 = real_t(0);
  for (int i = 0; i < n_models; ++i) {
    if (!models[i].applicable) { continue; }
    const real_t low = models[i].min_energy, high = models[i].max_energy;
    if (low <= e && high >= e) {
      ++cou;
      emi2 = emi1; ema2 = ema1;
      emi1 = low;  ema1 = high;
      memor2 = memory; memory = i;
    }
  }

  if (cou == 0) { out.status = ModelChoice::kNoModelInRange; return out; }
  if (cou == 1) { out.index = memory; out.status = ModelChoice::kOk; return out; }
  if (cou > 2)  { out.status = ModelChoice::kMoreThanTwoCompeting; return out; }

  if ((emi2 <= emi1 && ema2 >= ema1) || (emi2 >= emi1 && ema2 <= ema1)) {
    out.status = ModelChoice::kFullyOverlapping;
    return out;
  }
  const real_t rand = rng.uniform();
  int mem;
  if (emi1 < emi2) {
    mem = ((ema1 - e) < rand * (ema1 - emi2)) ? memor2 : memory;
  } else {
    mem = ((ema2 - e) < rand * (ema2 - emi1)) ? memory : memor2;
  }
  out.index = mem;
  out.status = ModelChoice::kOk;
  return out;
}

/// The probability that `choose_hadronic_interaction` picks the model with the HIGHER minimum
/// energy, for two models overlapping at `kin_energy`. Written out so a test can compare the
/// counted frequency against a formula rather than against another copy of the same code.
template <typename real_t>
__host__ __device__ real_t overlap_probability_upper(real_t lower_emax, real_t upper_emin,
                                                     real_t kin_energy) {
  if (kin_energy <= upper_emin) { return real_t(0); }
  if (kin_energy >= lower_emax) { return real_t(1); }
  return (kin_energy - upper_emin) / (lower_emax - upper_emin);
}

// =============================================================================================
// G4HadronicProcess::FillResult
// =============================================================================================

/// What a step reports back to the transport: the generic G4ParticleChange a hadronic process
/// proposes. `fill_result` and the elastic process both produce one of these.
enum class TrackStatusChange { kAlive = 0, kStopButAlive = 1, kStopAndKill = 2 };

template <typename real_t, int kMaxSecondaries = 8>
struct HadronicStepResult {
  TrackStatusChange status = TrackStatusChange::kAlive;
  real_t energy = real_t(0);                            ///< the primary's new kinetic energy
  Vec3<real_t> momentum_direction{real_t(0), real_t(0), real_t(1)};  ///< lab frame
  real_t local_energy_deposit = real_t(0);
  real_t non_ionizing_energy_deposit = real_t(0);
  real_t weight = real_t(1);

  int n_secondaries = 0;
  int secondary_overflow = 0;
  int n_ic_electrons = 0;   ///< PDG 11 secondaries; the ep check needs the count
  int kaon0_seen = 0;       ///< see the header comment: refused, not silently substituted
  int off_shell_fixed = 0;  ///< secondaries put back on the mass shell
  HadSecondary<real_t> secondaries[kMaxSecondaries];
};

/// G4HadronicProcess::FillResult.
///
/// The order of the three primary branches is Geant4's and is not interchangeable: `stopAndKill`
/// from the model wins even if the model also left a positive energy, and only then is a zero
/// final energy read as a stop. The middle branch is why a stopped hadron with at-rest processes
/// becomes fStopButAlive rather than being killed - a stopped pi- is captured, not deleted, and
/// that is the hook P4's decay and P12's absorption compete on.
///
/// `has_at_rest_processes` is the caller's answer to
/// `aT.GetParticleDefinition()->GetProcessManager()->GetAtRestProcessVector()->size() > 0`.
///
/// Secondaries are rotated from the model's +z frame into the lab with `rotateUz(dir)`, put back
/// on the mass shell if the model's dynamic mass is more than 1 keV from the PDG mass (with the
/// new kinetic energy floored at 0.001 eV), given `max(secTime, 0) + globalTime`, and weighted
/// by `fWeight * secondaryWeight`.
template <typename real_t, int kCap, int kSecCap>
__host__ __device__ HadronicStepResult<real_t, kSecCap> fill_result(
    const HadFinalState<real_t, kCap>& r, const Vec3<real_t>& track_direction,
    real_t track_global_time, real_t track_weight, bool has_at_rest_processes,
    const real_t* pdg_mass_of_secondary) {
  HadronicStepResult<real_t, kSecCap> out;
  out.weight = track_weight;
  out.local_energy_deposit = r.local_energy_deposit;

  const real_t efinal = (r.energy_change > real_t(0)) ? r.energy_change : real_t(0);

  if (r.status == HadFinalStateStatus::kStopAndKill) {
    out.status = TrackStatusChange::kStopAndKill;
    out.energy = real_t(0);
  } else if (efinal == real_t(0)) {
    out.energy = real_t(0);
    out.status = has_at_rest_processes ? TrackStatusChange::kStopButAlive
                                       : TrackStatusChange::kStopAndKill;
  } else {
    out.status = TrackStatusChange::kAlive;
    out.momentum_direction = rotate_uz(r.momentum_change, track_direction);
    out.energy = efinal;
  }

  constexpr real_t kDeltaMassLim = real_t(1e-3);   // 1 keV
  constexpr real_t kDeltaEkin = real_t(1e-9);      // 0.001 eV

  for (int i = 0; i < r.n_secondaries; ++i) {
    HadSecondary<real_t> s = r.secondaries[i];
    s.direction = rotate_uz(s.direction, track_direction);

    const real_t pdg_mass = pdg_mass_of_secondary ? pdg_mass_of_secondary[i] : s.mass;
    const real_t dm = s.mass - pdg_mass;
    if ((dm > kDeltaMassLim) || (-dm > kDeltaMassLim)) {
      const real_t e = s.kin_energy + dm;
      s.kin_energy = (e > kDeltaEkin) ? e : kDeltaEkin;
      s.mass = pdg_mass;
      ++out.off_shell_fixed;
    }
    if (s.pdg == 11) { ++out.n_ic_electrons; }
    if (s.pdg == 311 || s.pdg == -311) { ++out.kaon0_seen; }

    s.time = ((s.time > real_t(0)) ? s.time : real_t(0)) + track_global_time;
    s.weight = track_weight * s.weight;

    if (out.n_secondaries < kSecCap) {
      out.secondaries[out.n_secondaries++] = s;
    } else {
      ++out.secondary_overflow;
    }
  }
  out.secondary_overflow += r.secondary_overflow;
  return out;
}

// =============================================================================================
// G4HadronicProcess::CheckResult
// =============================================================================================

/// G4HadronicInteraction::GetFatalEnergyCheckLevels - (2%, 1 GeV) by default.
template <typename real_t>
struct FatalEnergyCheckLevels {
  real_t relative = real_t(0.02);
  real_t absolute = real_t(1000);  // 1 GeV in MeV
};

enum class CheckResultVerdict {
  kAccept = 0,
  kResampleOffShellSecondary,  ///< a secondary's dynamic mass is off its PDG mass
  kResampleEnergyBalance       ///< |dE| over BOTH the relative and the absolute level
};

/// G4HadronicProcess::CheckResult.
///
/// `delta_e = M_target + E_projectile_total - E_final`, and the result is thrown away only when
/// `|dE| > absolute && |dE| > relative*Ekin` - an AND, so the 1 GeV floor makes this inert for
/// anything QBBC does below a GeV. `nuclear_mass_target` must be
/// G4NucleiProperties::GetNuclearMass(A, Z); pass it in rather than computing it, because that
/// table is package P3's.
///
/// Two subtleties from the source. When the model did NOT return stopAndKill, the primary is
/// counted in the final state as `localEnergyDeposit + PDGMass + energyChange` - note the PDG
/// mass, not the total energy, so `finalE` is a total energy only because the mass is added
/// back. And if there are no secondaries at all in that case, the target mass is dropped from
/// the initial side too ("since there are no secondaries, there is no recoil nucleus"), which is
/// exactly the elastic case with a suppressed recoil: without it every elastic scatter would
/// look like it lost a whole nucleus.
///
/// This is transcribed for completeness. `G4HadronElasticProcess::PostStepDoIt` does not call
/// it - see the comment in elastic/elastic_process.cuh.
template <typename real_t, int kCap>
__host__ __device__ CheckResultVerdict check_result(
    const HadProjectile<real_t>& projectile, real_t nuclear_mass_target,
    const HadFinalState<real_t, kCap>& r, const FatalEnergyCheckLevels<real_t>& levels,
    const real_t* pdg_mass_of_secondary, real_t* delta_e_out) {
  real_t nuclear_mass = nuclear_mass_target;
  real_t final_e = real_t(0);

  if (r.status != HadFinalStateStatus::kStopAndKill) {
    final_e = r.local_energy_deposit + projectile.mass + r.energy_change;
    if (r.n_secondaries == 0) { nuclear_mass = real_t(0); }
  }
  for (int i = 0; i < r.n_secondaries; ++i) {
    final_e += r.secondaries[i].total_energy();
    const real_t mass_pdg = pdg_mass_of_secondary ? pdg_mass_of_secondary[i]
                                                  : r.secondaries[i].mass;
    const real_t mass_dyn = r.secondaries[i].mass;
    const real_t d = (mass_pdg > mass_dyn) ? (mass_pdg - mass_dyn) : (mass_dyn - mass_pdg);
    // The short-lived escape clause (3 * PDG width) needs a width table, which no elastic
    // secondary has: a recoil nucleus, a proton, a deuteron, a triton, He3 or an alpha are all
    // stable. An inelastic model that emits resonances must add it.
    if (d > real_t(0.1) * mass_pdg + real_t(1)) {
      if (delta_e_out) { *delta_e_out = real_t(0); }
      return CheckResultVerdict::kResampleOffShellSecondary;
    }
  }
  const real_t delta_e = nuclear_mass + projectile.total_energy() - final_e;
  if (delta_e_out) { *delta_e_out = delta_e; }
  const real_t ad = (delta_e > real_t(0)) ? delta_e : -delta_e;
  if (ad > levels.absolute && ad > levels.relative * projectile.kin_energy) {
    return CheckResultVerdict::kResampleEnergyBalance;
  }
  return CheckResultVerdict::kAccept;
}

// =============================================================================================
// G4HadronicProcess::CheckEnergyMomentumConservation
// =============================================================================================

/// The verdict CheckEnergyMomentumConservation computes before it decides what to print.
template <typename real_t>
struct EpReport {
  real_t absolute = real_t(0);      ///< (initial - final).e()
  real_t relative = real_t(0);      ///< absolute / Ekin, or 0 when the check is not relative
  real_t absolute_mom = real_t(0);  ///< |(initial - final).vect()|
  real_t relative_mom = real_t(0);
  int delta_a = 0;                  ///< initial_A - final_A
  int delta_z = 0;
  bool relative_pass = true;
  bool absolute_pass = true;
  bool charge_pass = true;
  bool conservation_pass = true;
};

/// G4HadronicProcess::CheckEnergyMomentumConservation.
///
/// Gated on `epReportLevel != 0`, whose default is 0, so in QBBC this never runs - and with the
/// default `epCheckLevels` of (DBL_MAX, DBL_MAX) every comparison here would pass anyway. It is
/// transcribed because the arithmetic is the definition of "what Geant4 conserves", and a test
/// can assert on it directly instead of on a printout.
///
/// The shape of it: the target contributes `M(A,Z) + nICelectrons*m_e` at rest and takes
/// `initial_Z = target_Z + track_Z - nICelectrons`. If the primary survives (status is not
/// StopAndKill) the final state is seeded with the WHOLE initial four-momentum, and only when
/// there is at least one secondary is that replaced by the primary's own final four-momentum -
/// which is how "interaction didn't complete" and "suppressed recoil (e.g. neutron elastic)"
/// both come out conserving exactly. Every secondary then adds its four-momentum, baryon number
/// and charge.
///
/// `checkRelative` is `Ekin > checkLevels.second` - the ABSOLUTE level, in energy, used as the
/// threshold for whether a relative comparison is meaningful at all. That reads like a bug and
/// is Geant4's: with the default absolute level of DBL_MAX no track is ever above it, so
/// `relative` is always reported as 0 and `relResult` as "N/A".
///
/// The verdict is `(relative_pass || absolute_pass) && charge_pass` - an OR, so one of the two
/// energy checks passing is enough. And `charge_pass` is forced true when the absolute level is
/// DBL_MAX, i.e. by default a baryon-number or charge imbalance does not fail the check either.
template <typename real_t, int kSecCap>
__host__ __device__ EpReport<real_t> report_energy_momentum(
    const HadProjectile<real_t>& projectile, const Vec3<real_t>& track_direction,
    real_t nuclear_mass_target, int target_z, int target_a,
    const HadronicStepResult<real_t, kSecCap>& res, const int* secondary_baryon_number,
    const real_t* secondary_charge, real_t check_relative_level, real_t check_absolute_level,
    bool absolute_level_is_infinite) {
  EpReport<real_t> out;

  const real_t me = units::electron_mass_c2<real_t>();
  const int n_ic = res.n_ic_electrons;

  const real_t target_e = nuclear_mass_target + real_t(n_ic) * me;
  const real_t plab = projectile.momentum();
  Vec3<real_t> p_init = plab * track_direction;
  real_t e_init = projectile.total_energy() + target_e;

  const int track_a = projectile.baryon_number;
  const int track_z = static_cast<int>(projectile.charge > real_t(0)
                                           ? projectile.charge + real_t(0.5)
                                           : projectile.charge - real_t(0.5));
  const int initial_a = target_a + track_a;
  const int initial_z = target_z + track_z - n_ic;

  Vec3<real_t> p_final{real_t(0), real_t(0), real_t(0)};
  real_t e_final = real_t(0);
  int final_a = 0, final_z = 0;

  if (res.status != TrackStatusChange::kStopAndKill) {
    p_final = p_init;
    e_final = e_init;
    final_a = initial_a;
    final_z = initial_z;
    if (res.n_secondaries > 0) {
      const real_t ekin = res.energy;
      const real_t mass = projectile.mass;
      const real_t ptot = sqrt(ekin * (ekin + real_t(2) * mass));
      p_final = ptot * res.momentum_direction;
      e_final = mass + ekin;
      final_a = track_a;
      final_z = track_z;
    }
  }
  for (int i = 0; i < res.n_secondaries; ++i) {
    const real_t p = res.secondaries[i].momentum();
    p_final = p_final + p * res.secondaries[i].direction;
    e_final += res.secondaries[i].total_energy();
    final_a += secondary_baryon_number ? secondary_baryon_number[i] : 0;
    if (secondary_charge) {
      const real_t q = secondary_charge[i];
      final_z += static_cast<int>(q > real_t(0) ? q + real_t(0.5) : q - real_t(0.5));
    }
  }

  const bool check_relative = (projectile.kin_energy > check_absolute_level);

  out.absolute = e_init - e_final;
  const Vec3<real_t> dp = p_init - p_final;
  out.absolute_mom = mag(dp);
  out.relative = check_relative ? out.absolute / projectile.kin_energy : real_t(0);
  out.relative_mom = check_relative ? out.absolute_mom / plab : real_t(0);

  const real_t ar = (out.relative > real_t(0)) ? out.relative : -out.relative;
  const real_t arm = (out.relative_mom > real_t(0)) ? out.relative_mom : -out.relative_mom;
  out.relative_pass = !(ar > check_relative_level || arm > check_relative_level);

  const real_t aa = (out.absolute > real_t(0)) ? out.absolute : -out.absolute;
  const real_t aam = (out.absolute_mom > real_t(0)) ? out.absolute_mom : -out.absolute_mom;
  out.absolute_pass = !(aa > check_absolute_level || aam > check_absolute_level);

  out.delta_a = initial_a - final_a;
  out.delta_z = initial_z - final_z;
  out.charge_pass = true;
  if (out.delta_a != 0 || out.delta_z != 0) {
    out.charge_pass = absolute_level_is_infinite;
  }
  out.conservation_pass = (out.relative_pass || out.absolute_pass) && out.charge_pass;
  return out;
}

/// The default ep report level in 11.1.1: 0, meaning `report_energy_momentum` is never called.
/// Named rather than written as a bare 0 at the call site, so that the default is a fact in the
/// code and not a habit.
inline constexpr int kDefaultEpReportLevel = 0;

// =============================================================================================
// The integral cross-section rejection
// =============================================================================================

/// G4HadXSType: the shape G4HadronicProcess::BuildPhysicsTable decided the cross section has.
enum class HadXsType { kNoIntegral = 0, kIncreasing, kDecreasing, kOnePeak, kTwoPeaks };

/// `G4HadronicProcess::BuildPhysicsTable`'s choice of cross-section shape.
///
/// The integral approach is on by default for both elastic and inelastic
/// (G4HadronicParameters.hh:172-173, both true), but it applies only to a CHARGED, non-leptonic
/// particle that is either an ion or lighter than 1 GeV. For a neutron `charge != 0.0` is false,
/// so a neutron's process keeps fHadNoIntegral and never does the rejection below - which is
/// what makes the neutron's elastic usable as a sub-process of G4NeutronGeneralProcess, whose
/// own table has already decided that an interaction happens.
///
/// The specific shapes: |pdg| == 211 (pions) and pdg == 2212 (proton) are fHadTwoPeaks,
/// pdg == 321 (K+) is fHadOnePeak, and everything else charged is increasing or decreasing by
/// the sign of the charge.
template <typename real_t>
__host__ __device__ inline HadXsType hadronic_xs_type(int pdg, real_t charge, real_t pdg_mass,
                                                      int atomic_number, bool is_lepton,
                                                      bool use_integral_xs) {
  const bool ok = (atomic_number != 0 || pdg_mass < units::GeV<real_t>());
  if (charge == real_t(0) || !use_integral_xs || is_lepton || !ok) {
    return HadXsType::kNoIntegral;
  }
  const int apdg = (pdg < 0) ? -pdg : pdg;
  if (apdg == 211 || pdg == 2212) { return HadXsType::kTwoPeaks; }
  if (pdg == 321) { return HadXsType::kOnePeak; }
  return (charge > real_t(0)) ? HadXsType::kIncreasing : HadXsType::kDecreasing;
}

/// The integral-approach rejection at the top of PostStepDoIt.
///
/// The mean free path was sampled with the cross section at the START of the step, which for a
/// charged hadron that lost energy is not the cross section where the step ended. Geant4 keeps
/// the larger value and rejects the interaction with probability `1 - xs/xs_last`. Returns true
/// when the interaction is REJECTED, in which case PostStepDoIt returns the track unchanged - and
/// note that it has already set `theNumberOfInteractionLengthLeft = -1`, so the next step
/// re-samples a fresh interaction length rather than continuing the old one.
///
/// One uniform is drawn, and only when the type is not kNoIntegral.
template <typename real_t, typename Rng>
__host__ __device__ bool integral_xs_rejects(HadXsType type, real_t xs_now, real_t xs_at_step_start,
                                             Rng& rng) {
  if (type == HadXsType::kNoIntegral) { return false; }
  return xs_now < xs_at_step_start * rng.uniform();
}

}  // namespace g4gpu::physics::hadronic
