// P8b's elastic wiring: which (cross section, model) pair each charged hadron's `hadElastic`
// uses, and one function that runs it.
//
// This file owns no physics. P2 has the cross sections (`xs/`), P5 has the framework
// (`process.cuh`) and the final states (`elastic/`), P8b has the answer to "for THIS species,
// which of them". That answer is a table in `G4HadronElasticPhysics::ConstructProcess` and it
// is transcribed below rather than inferred, because three of the five rows are easy to guess
// wrong.
//
// ---------------------------------------------------------------------------------------------
// THE TABLE, from G4HadronElasticPhysics::ConstructProcess (11.1.1). G4HadronElasticPhysicsXS
// ::ConstructProcess is one line - `G4HadronElasticPhysics::ConstructProcess();` - so the "XS"
// variant QBBC registers adds nothing of its own.
//
//   species          cross section                                    final-state model
//   p                G4BGGNucleonElasticXS(proton)                     G4ChipsElasticModel
//   n                G4NeutronElasticXS, inside G4NeutronGeneralProcess G4ChipsElasticModel
//   pi+, pi-         G4BGGPionElasticXS(pi+/pi-)                        G4ElasticHadrNucleusHE
//   K+, K-           G4CrossSectionElastic(G4ComponentGGHadronNucleusXsc) G4HadronElastic
//   d, t, He3, alpha G4CrossSectionElastic(G4ComponentGGNuclNuclXsc)    G4HadronElastic
//   pbar             G4CrossSectionElastic(G4ComponentAntiNuclNuclearXsc) G4HadronElastic < 100.1
//                                                                       MeV, G4AntiNuclElastic
//                                                                       above 100 MeV
//   mu+, mu-         NONE - no lepton gets an elastic process at all
//
// Three things in it that a reader would otherwise have to take on trust:
//
//  * THE KAONS DO NOT GET THE PION'S MODEL. `G4HadronicBuilder::BuildElastic` - which is what
//    the "kaons" line of the constructor calls - builds `G4HadProcesses::ElasticXS(
//    "Glauber-Gribov")`, i.e. `G4CrossSectionElastic(new G4ComponentGGHadronNucleusXsc())`,
//    and registers a plain `G4HadronElastic`. So a kaon's angular distribution is Gheisha's
//    `SampleInvariantT`, not `G4ElasticHadrNucleusHE`'s, even though that model's table has
//    rows for K+ and K-.
//  * THE LIGHT IONS GET Gheisha TOO, with the NUCL-NUCL component - `xsNN` in the constructor
//    is `ElasticXS("Glauber-Gribov Nucl-nucl")` - while `G4NuclNuclDiffuseElastic` is
//    `G4IonElasticPhysics`' model for GenericIon, a different constructor and a different
//    process (named `"ionElastic"`, not `"hadElastic"`). P8c transports the ion and did NOT
//    wire that channel; see `ElasticChannel::kIonDiffuseNotWired`, which is the row.
//  * pbar HAS A PORTED MODEL AND NO PORTED CROSS SECTION. `lhep2` (a `G4HadronElastic` capped
//    at 100.1 MeV) covers it below 100 MeV, but the data set is
//    `G4ComponentAntiNuclNuclearXS`, which P2 refuses by name, and above 100 MeV the model is
//    `G4AntiNuclElastic`, which P5 did not start. So the antiproton has NO elastic scattering
//    here and it is booked as `HadronicRefusal::kAntiNucleusElastic`, which is the counter
//    wiring.cuh already reserved for it.
//
// And one thing about the cross sections that decides the target draw: none of these four data
// sets overrides `G4VCrossSectionDataSet::SelectIsotope`. The BGG classes' IsElementApplicable
// returns true and `G4CrossSectionElastic`'s is `Z in [1, 256] && E in [0, Emax]`, so
// `SampleZandA` takes its ELEMENT-WISE branch for all of them and the isotope comes from the
// base class's abundance-weighted draw. `elastic_abundance_only` is therefore true, always, and
// the isotope cross section is never computed - which is why this file needs no per-isotope
// cross section at all even though the recoil is (Z, A)-resolved.
//
// ---------------------------------------------------------------------------------------------
// WHAT IS UPLOADED, AND WHAT IS CLOSED FORM
//
// Two of the four cross sections carry per-Z tables that `BuildPhysicsTable` fills on the host
// (`BggNucleonTable`, `BggPionTable` - about 3 KB and 30 KB), and one of the three models does
// (`G4ElasticHadrNucleusHE`'s `G4ElasticData`, 24 energies by 102 cumulative points per
// (hadron, Z), about 20 KB each). The rest is closed form: the two Glauber-Gribov components are
// compiled-in per-Z arrays and arithmetic, `G4ChipsElasticModel`'s parameters are compiled in,
// and `G4HadronElastic::SampleInvariantT` is a formula. So `ElasticTables` is five pointers and
// `host/hadronic_upload.cuh` fills them once per run.
//
// The HE tables are built only for pi+ and pi- and only for the Z values the SCENE contains,
// because that is who reads them: `he_sample_invariant_t` uses the table only for Z > 1 (a
// hadron-proton scatter is closed form), and no other species in QBBC uses the model.
#pragma once

#include "core/particle.cuh"
#include "data/isotope_abundance.hh"
#include "data/materials.cuh"
#include "physics/hadronic/elastic/chips_elastic.cuh"
#include "physics/hadronic/elastic/elastic_hadr_nucleus_he.cuh"
#include "physics/hadronic/elastic/elastic_process.cuh"
#include "physics/hadronic/elastic/hadron_elastic.cuh"
#include "physics/hadronic/process.cuh"
#include "physics/hadronic/xs/bgg_nucleon_xs.cuh"
#include "physics/hadronic/xs/bgg_pion_xs.cuh"
#include "physics/hadronic/xs/gg_hadron_nucleus_xsc.cuh"
#include "physics/hadronic/xs/gg_nucl_nucl_xsc.cuh"
#include "physics/hadronic/xs/sample_za.cuh"

namespace g4gpu::had {

namespace el = g4gpu::physics::hadronic::elastic;
namespace hp = g4gpu::physics::hadronic;
namespace hxs = g4gpu::hadronic::xs;

/// Which row of the table at the top of this file a species takes.
enum class ElasticChannel : int {
  /// No `hadElastic` on this species' process manager at all: the leptons, and the neutral
  /// hadrons, whose elastic is a sub-process of `G4NeutronGeneralProcess` instead.
  kNone = 0,
  kChipsNucleon,      ///< p  - G4BGGNucleonElasticXS + G4ChipsElasticModel
  kHePion,            ///< pi+- - G4BGGPionElasticXS + G4ElasticHadrNucleusHE
  kGheishaKaon,       ///< K+- - GG hadron-nucleus + G4HadronElastic
  kGheishaLightIon,   ///< d, t, He3, alpha - GG nucl-nucl + G4HadronElastic
  /// pbar - the cross section (G4ComponentAntiNuclNuclearXS) is refused by P2 and the
  /// high-energy model (G4AntiNuclElastic) is not written. Booked, not approximated.
  kAntiNucleusRefused,
  /// GenericIon - `G4IonElasticPhysics::ConstructProcess` gives it a process named
  /// `"ionElastic"`, `G4CrossSectionElastic(G4ComponentGGNuclNuclXsc)` with `SetMinKinEnergy(0)`
  /// and `G4NuclNuclDiffuseElastic` with `SetMinEnergy(0)`, registered straight onto
  /// `G4GenericIon::GenericIon()->GetProcessManager()` with `AddDiscreteProcess` - and therefore
  /// active for every real nuclide, which shares that manager.
  ///
  /// BOTH HALVES ARE PORTED AND NEITHER IS WIRED. `xs::ggnn_elastic_element` is the cross
  /// section (`kGheishaLightIon` already reads it) and
  /// `elastic/nucl_nucl_diffuse_elastic.cuh::sample_invariant_t` is the model. What is missing
  /// is this row: the channel needs the projectile to be `xs::generic_ion(Z, A)` rather than a
  /// species constant, which means threading the nuclide through `elastic_xs_fn`,
  /// `elastic_sample_target` and `elastic_apply`.
  ///
  /// Structurally zero rather than booked, for the reason `HadronicRefusal::
  /// kAntiNucleusElastic` gives at length: the gap is in the CROSS SECTION, so an ion draws no
  /// hadronic interaction length at all and there is no interaction that could not be applied.
  /// Booking it per step would count chances rather than interactions. The size is bounded by
  /// what an ion this port produces could do with it: an elastic recoil of a 200 MeV proton in
  /// water is an oxygen ion of a few hundred keV whose RANGE is about a micrometre, against a
  /// nucleus-nucleus elastic mean free path of metres, so the probability that one scatters
  /// before it stops is of order 1e-9.
  kIonDiffuseNotWired,
};

__host__ __device__ inline ElasticChannel elastic_channel(ParticleType t) {
  switch (t) {
    case ParticleType::kProton:     return ElasticChannel::kChipsNucleon;
    case ParticleType::kPionPlus:
    case ParticleType::kPionMinus:  return ElasticChannel::kHePion;
    case ParticleType::kKaonPlus:
    case ParticleType::kKaonMinus:  return ElasticChannel::kGheishaKaon;
    case ParticleType::kDeuteron:
    case ParticleType::kTriton:
    case ParticleType::kHe3:
    case ParticleType::kAlpha:      return ElasticChannel::kGheishaLightIon;
    case ParticleType::kAntiProton: return ElasticChannel::kAntiNucleusRefused;
    case ParticleType::kGenericIon: return ElasticChannel::kIonDiffuseNotWired;
    default:                        return ElasticChannel::kNone;
  }
}

__host__ __device__ inline const char* elastic_channel_name(ElasticChannel c) {
  switch (c) {
    case ElasticChannel::kNone: return "no hadElastic";
    case ElasticChannel::kChipsNucleon:
      return "G4BGGNucleonElasticXS + G4ChipsElasticModel";
    case ElasticChannel::kHePion:
      return "G4BGGPionElasticXS + G4ElasticHadrNucleusHE";
    case ElasticChannel::kGheishaKaon:
      return "G4CrossSectionElastic(G4ComponentGGHadronNucleusXsc) + G4HadronElastic";
    case ElasticChannel::kGheishaLightIon:
      return "G4CrossSectionElastic(G4ComponentGGNuclNuclXsc) + G4HadronElastic";
    case ElasticChannel::kAntiNucleusRefused:
      return "G4ComponentAntiNuclNuclearXS + G4AntiNuclElastic - refused by name";
    case ElasticChannel::kIonDiffuseNotWired:
      return "ionElastic: G4ComponentGGNuclNuclXsc + G4NuclNuclDiffuseElastic - both ported, "
             "this channel not wired";
  }
  return "unknown";
}

/// Does this channel have a cross section this port can evaluate?
///
/// Three answers collapse to "no" and they are three different statements, which is why the
/// enum keeps them apart and only this predicate merges them: `kNone` is a species Geant4 gives
/// no elastic process to at all, `kAntiNucleusRefused` is a process whose data set P2 refuses,
/// and `kIonDiffuseNotWired` is a process whose data set and model are both ported and whose
/// channel nobody has written. All three mean the same thing to `elastic_xs_per_volume` - zero,
/// and therefore an infinite interaction length and no uniform drawn - and `elastic_apply` must
/// refuse all three for the reason its own guard gives (a zero `MaterialXs` makes
/// `store_sample_za_rng` hand element 0 to the sampler on `cross <= cumulative[0]`, both sides
/// zero, and the sampler is happy to scatter a muon off hydrogen).
__host__ __device__ inline bool elastic_channel_has_xs(ElasticChannel c) {
  return c != ElasticChannel::kNone && c != ElasticChannel::kAntiNucleusRefused
         && c != ElasticChannel::kIonDiffuseNotWired;
}

/// `G4ElasticHadrNucleusHE`'s table rows this port builds: pi+ at index 0 and pi- at 1, which
/// are also their indices in `he_hadron_code()` - and that array's index is what
/// `fElasticData`'s first subscript is, so the two orders must be the same one.
constexpr int kHeNumHadrons = 2;
__host__ __device__ inline int he_hadron_slot(ParticleType t) {
  if (t == ParticleType::kPionPlus) { return 0; }
  if (t == ParticleType::kPionMinus) { return 1; }
  return -1;
}

/// The device tables `hadElastic` reads. Null pointers mean the tables were not uploaded, and
/// `elastic_xs_per_volume` then returns zero for the channels that need them - which is the same
/// "no process" state a species with `kNone` is in, and is why `Upload` can leave them null for
/// a run with no hadrons in it.
template <typename real_t>
struct ElasticTables {
  const hxs::BggNucleonTable<real_t>* bgg_nucleon = nullptr;  ///< is_elastic == true
  const hxs::BggPionTable<real_t>* bgg_pion = nullptr;        ///< is_elastic == true
  /// One `G4ElasticData` per (hadron slot, Z slot), at `kHeNumHadrons * z_slot + hadron_slot`.
  const el::HeElasticData<real_t>* he = nullptr;
  /// Z -> z_slot, or -1 for a Z the scene does not contain. Sized `he_max_z + 1`.
  const short* he_slot_of_z = nullptr;
  int he_max_z = 0;
  const el::HeEnergyGrid<real_t>* he_grid = nullptr;
  const el::HeBoundary<real_t>* he_bnd = nullptr;
};

/// The species as the cross sections ask about it. `xs::Projectile` and not `ParticleType`, for
/// the reasons `xs/projectile.cuh` gives - and built through that file's own factories so the
/// masses are the `x.yz * GeV` literals Geant4 writes and not a re-spelling of them.
template <typename real_t>
__host__ __device__ inline hxs::Projectile<real_t> elastic_projectile(ParticleType t) {
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
    default:                        return hxs::Projectile<real_t>{};
  }
}

/// The three-function contract of `xs/sample_za.cuh`, for one species at one energy.
///
/// `abundance_only` is TRUE for every channel, and that is not a simplification - see the note
/// at the top of this file. `isotope` is therefore unreachable through `store_*`, and it returns
/// the element cross section rather than a zero so that a future caller that does reach it gets
/// Geant4's own fall-back (`G4VCrossSectionDataSet::GetIsoCrossSection` answers from the element
/// cross section when the data set has no isotope data) instead of silence.
template <typename real_t>
struct ElasticXsFn {
  const ElasticTables<real_t>* tables = nullptr;
  hxs::Projectile<real_t> proj{};
  ElasticChannel channel = ElasticChannel::kNone;
  real_t ekin = 0;

  __host__ __device__ hxs::XsValue<real_t> element(int Z) const {
    switch (channel) {
      case ElasticChannel::kChipsNucleon:
        if (tables->bgg_nucleon == nullptr) { return {real_t(0), hxs::XsRefusal::kNone}; }
        return hxs::bgg_nucleon_element_xs<real_t>(*tables->bgg_nucleon, proj, ekin, Z);
      case ElasticChannel::kHePion:
        if (tables->bgg_pion == nullptr) { return {real_t(0), hxs::XsRefusal::kNone}; }
        return hxs::bgg_pion_element_xs<real_t>(*tables->bgg_pion, proj, ekin, Z);
      case ElasticChannel::kGheishaKaon:
        // G4CrossSectionElastic::GetElementCrossSection passes
        // `nist->GetAtomicMassAmu(Z)` - the NIST mean atomic mass, NOT G4IsotopeList's
        // aeff[Z], which is a different number in the fifth digit.
        return hxs::ggh_elastic_element<real_t>(proj, ekin, Z, data::atomic_mass<real_t>(Z));
      case ElasticChannel::kGheishaLightIon:
        return hxs::ggnn_elastic_element<real_t>(proj, ekin, Z, data::atomic_mass<real_t>(Z));
      case ElasticChannel::kNone:
      case ElasticChannel::kAntiNucleusRefused:
      case ElasticChannel::kIonDiffuseNotWired:
        break;
    }
    return {real_t(0), hxs::XsRefusal::kNone};
  }
  __host__ __device__ hxs::XsValue<real_t> isotope(int Z, int /*A*/) const {
    return element(Z);
  }
  __host__ __device__ bool abundance_only(int /*Z*/) const { return true; }
};

template <typename real_t>
__host__ __device__ inline ElasticXsFn<real_t> elastic_xs_fn(const ElasticTables<real_t>& t,
                                                              ParticleType type, real_t ekin) {
  return {&t, elastic_projectile<real_t>(type), elastic_channel(type), ekin};
}

/// `hadElastic`'s macroscopic cross section, 1/mm, and the cumulative partial sums the target
/// draw needs. Zero when the species has no such process, and zero when it has one this port
/// cannot evaluate (the antiproton).
///
/// `G4CrossSectionDataStore::ComputeCrossSection` and nothing else: one call fills `mxs` and
/// returns the total, and `elastic_sample_target` below must be called at the same energy and
/// material or it selects an element by the wrong energy's cross sections. That pairing is the
/// whole reason `MaterialXs` is a struct.
///
/// `__noinline__` for the reason `elastic_apply` gives at length, and this one is the bigger
/// half of it: inlined, it drags G4BGGNucleonElasticXS, G4BGGPionElasticXS,
/// G4UPiNuclearCrossSection, Barashenkov, G4HadronNucleonXsc and both Glauber-Gribov components
/// into `run_step_hadron`'s body once per species. With both this and `elastic_apply` left
/// inline the proton kernel measured 2912 B of stack and 1052/1292 B of spill against a
/// baseline of 3696 B and 68/36, and the whole translation unit killed ptxas.
template <typename real_t>
__host__ __device__ __noinline__ real_t elastic_xs_per_volume(
    const ElasticTables<real_t>& t, const data::Material<real_t>& mat, ParticleType type,
    real_t ekin, hxs::MaterialXs<real_t>& mxs) {
  const ElasticChannel c = elastic_channel(type);
  if (!elastic_channel_has_xs(c)) {
    mxs.total = real_t(0);
    mxs.n_elements = 0;
    return real_t(0);
  }
  const ElasticXsFn<real_t> fn = elastic_xs_fn<real_t>(t, type, ekin);
  const hxs::XsValue<real_t> v = hxs::store_compute_cross_section_fn<real_t>(
      fn, mat, hxs::nist_isotopes_of<real_t>(mat), mxs);
  return (v.value > real_t(0)) ? v.value : real_t(0);
}

/// `G4CrossSectionDataStore::SampleZandA`, with Geant4's draw order and Geant4's draw COUNT.
///
/// A single-element material draws no element uniform and a single-isotope element draws no
/// isotope uniform - both are `if(1 < n)` tests in the source, and both change the random
/// stream rather than only the cost. `xs::store_sample_za_rng` is where that lives; this
/// function is the elastic caller's shorthand for it.
template <typename real_t, typename Rng>
__host__ __device__ inline hxs::TargetZA elastic_sample_target(
    const ElasticTables<real_t>& t, const data::Material<real_t>& mat, ParticleType type,
    real_t ekin, const hxs::MaterialXs<real_t>& mxs, Rng& rng) {
  const ElasticXsFn<real_t> fn = elastic_xs_fn<real_t>(t, type, ekin);
  return hxs::store_sample_za_rng<real_t>(fn, mat, hxs::nist_isotopes_of<real_t>(mat), mxs,
                                          rng);
}

/// What one `hadElastic` PostStepDoIt did to a track.
template <typename real_t>
struct ElasticStepOutcome {
  bool interacted = false;
  bool rejected_by_integral_xs = false;
  Vec3<real_t> dir{real_t(0), real_t(0), real_t(1)};
  real_t energy = 0;             ///< the primary's new kinetic energy
  bool primary_survives = true;
  real_t edep = 0;               ///< local AND non-ionizing - see elastic_process.cuh note 5
  bool emit_recoil = false;
  int recoil_z = 0, recoil_a = 0;
  real_t recoil_ekin = 0;
  Vec3<real_t> recoil_dir{real_t(0), real_t(0), real_t(1)};
  int dropped_secondaries = 0;
  /// Diagnostics a test can assert the PATH on rather than only the numbers.
  bool chips_fell_back = false;
  bool he_fell_back = false;
  bool he_unknown_hadron = false;
};

/// One whole `G4HadronElasticProcess::PostStepDoIt` for a charged hadron.
///
/// THE CAPACITY IS ONE AND THAT IS PROVABLE, not optimistic.
/// `hadron_elastic_apply_yourself` calls `add_secondary` exactly once, in one branch, and no
/// elastic model in QBBC emits more - so `HadFinalState<real_t, 1>` is 56 bytes of kernel stack
/// where the package's default of 8 would be 450. `secondary_overflow` still counts a model
/// that tried, and `dropped_secondaries` still reports Geant4's own
/// "only secondary 0 is looked at" (note 4 of elastic_process.cuh).
///
/// `__noinline__`, AND THAT IS NOT A HINT - IT IS WHAT MAKES THE ENGINE COMPILE.
///
/// Inlined, this function puts G4ChipsElasticModel's 52-parameter tables, G4ElasticHadrNucleusHE's
/// sampler, Gheisha's SampleInvariantT, the BGG cross sections and the two Glauber-Gribov
/// components into the body of `run_step_hadron` - once per species, fourteen times in
/// `transport_run.cu`. Measured on this branch:
///
///   one kernel alone (the proton)     2912 B stack, 1052/1292 B spill, 255 registers
///   the whole translation unit        ptxas died with 0xC0000005 (ACCESS_VIOLATION) after
///                                     printing "Internal error"
///
/// So the inline version does not build at all, and the version that does builds a hot path
/// carrying fifteen times the spill traffic it had before - for a branch that fires 16 times in
/// 900 steps (`tests/test_step_hadron.cu`; the elastic mean free path in water is 466 mm for a
/// 200 MeV pion and 2013 mm for a 200 MeV proton against steps of tens of mm). A real call is
/// therefore both the only thing that compiles and the right answer for the common case.
///
/// @param xs_at_step_start the macroscopic cross section the interaction length was drawn with,
///        for the integral-approach rejection. For a charged hadron that is a real rejection -
///        `hadronic_xs_type` gives the pions and the proton `fHadTwoPeaks` and K+ `fHadOnePeak`
///        - and it consumes one uniform before anything else.
template <typename real_t, typename Rng>
__host__ __device__ __noinline__ ElasticStepOutcome<real_t> elastic_apply(
    const ElasticTables<real_t>& t, const data::Material<real_t>& mat, ParticleType type,
    real_t ekin, const Vec3<real_t>& in_dir, real_t range_cut_mm, real_t xs_at_step_start,
    const hxs::MaterialXs<real_t>& mxs, Rng& rng) {
  ElasticStepOutcome<real_t> out;
  out.dir = in_dir;
  out.energy = ekin;

  const ElasticChannel c = elastic_channel(type);
  // A SPECIES WITH NO ELASTIC PROCESS MUST NOT GET ONE HERE, and the guard is not redundant.
  //
  // `step_hadron` never reaches this for a muon or an antiproton, because their cross section
  // is zero and the interaction length is infinite - but `elastic_apply` is also called
  // directly, by `tests/test_step_hadron.cu` on a grid of every species, and without this it
  // sampled a Chips scatter for a muon off hydrogen: the zero cross section leaves `mxs` all
  // zero, `store_sample_za_rng` then takes element 0 on `cross <= cumulative[0]` with both
  // sides zero, and the sampler is perfectly happy to be handed (Z=1, A=1) and a muon. The test
  // reported 132 interactions in 132 cases including the eleven muon rows, which is how this
  // was found; a guard on the caller's side alone would have left a function that answers a
  // question it should refuse.
  if (!elastic_channel_has_xs(c)) { return out; }
  const ParticleDef<real_t> pd = particle_def<real_t>(type);
  const hxs::Projectile<real_t> pj = elastic_projectile<real_t>(type);

  hp::HadProjectile<real_t> projectile;
  projectile.pdg = pj.pdg;
  projectile.baryon_number = pj.baryon_number;
  projectile.charge = pj.charge;
  projectile.mass = pj.mass;
  projectile.kin_energy = ekin;

  // The integral-approach rejection, and it comes FIRST: G4HadronElasticProcess::PostStepDoIt
  // recomputes the cross section at the post-step energy and throws the interaction away with
  // probability 1 - xs/xs_at_step_start, having already reset its interaction length. One
  // uniform, and only for a type that is not kNoIntegral.
  //
  // AND THE RECOMPUTE IS WHAT THE TARGET DRAW THEN READS, which is a second thing and not the
  // same one. `ComputeCrossSection` leaves the per-element partial sums behind and
  // `SampleZandA` reads them without recomputing anything (see xs/sample_za.cuh) - so the
  // element is drawn from the POST-step energy's partial sums for a charged hadron, whose
  // fXSType is not fHadNoIntegral, and from the PRE-step ones for a neutral particle, whose
  // PostStepDoIt skips the recompute entirely. Both branches are here: `now` when the
  // recompute happened, the caller's `mxs` when it did not. Using the caller's array in the
  // charged case would select an element by the cross sections at an energy the track no
  // longer has, and nothing would complain.
  const hp::HadXsType xt = hp::hadronic_xs_type<real_t>(
      pj.pdg, pj.charge, pj.mass, (pj.baryon_number > 1) ? pj.baryon_number : 0, false, true);
  hxs::MaterialXs<real_t> now = mxs;
  real_t xs_now = xs_at_step_start;
  if (xt != hp::HadXsType::kNoIntegral) {
    xs_now = elastic_xs_per_volume<real_t>(t, mat, type, ekin, now);
    if (hp::integral_xs_rejects<real_t>(xt, xs_now, xs_at_step_start, rng)) {
      out.rejected_by_integral_xs = true;
      return out;
    }
  }

  const hxs::TargetZA tgt = elastic_sample_target<real_t>(t, mat, type, ekin, now, rng);
  if (tgt.a <= 0 || tgt.z <= 0) { return out; }
  hp::HadNucleus target;
  target.z = tgt.z;
  target.a = tgt.a;

  const real_t recoil_threshold = el::proton_recoil_cut_energy<real_t>(range_cut_mm);
  // PLAIN LAMBDAS, not `__host__ __device__` ones: nvcc gives a lambda defined inside a
  // `__host__ __device__` function the enclosing function's execution space, and the explicit
  // annotation would need `--extended-lambda`, which neither build_all.bat nor
  // build_one_test.bat passes.
  auto nuclear_mass = [](int z, int a) { return data::nuclear_mass<real_t>(a, z); };
  auto recoil_species = [](int z, int a, int* pdg, real_t* mass) {
    *pdg = hp::pdg_nuclear_code(z, a);
    *mass = data::nuclear_mass<real_t>(a, z);
  };

  hp::HadFinalState<real_t, 1> fs;
  bool chips_fb = false, he_fb = false, he_unknown = false;

  // The sampler is the only thing that differs between the channels, and each one's fall-back
  // is its own: G4ChipsElasticModel falls back to Gheisha INSIDE SampleInvariantT (so the
  // uniforms are consumed there), while G4ElasticHadrNucleusHE returns and leaves the fall-back
  // to its caller - which is what Geant4's own `G4ElasticHadrNucleusHE::SampleInvariantT` does
  // on its first line. Both are reproduced where they belong.
  const int he_slot = he_hadron_slot(type);
  const real_t h_mass = pd.mass;
  auto apply = [&](const hp::HadProjectile<real_t>& p, const hp::HadNucleus& n, real_t thr,
                   Rng& r, hp::HadFinalState<real_t, 1>* o) {
    auto sampler = [&](int pdg, real_t plab, int z, int a, real_t p_local_tmax,
                       Rng& rr) -> real_t {
      if (c == ElasticChannel::kChipsNucleon) {
        bool unsupported = false;
        return el::chips_sample_invariant_t<real_t>(pdg, plab, z, a, p_local_tmax,
                                                    nuclear_mass, rr, &chips_fb, &unsupported);
      }
      if (c == ElasticChannel::kHePion) {
        const el::HeElasticData<real_t>* tab = nullptr;
        if (t.he != nullptr && t.he_slot_of_z != nullptr && z >= 0 && z <= t.he_max_z
            && he_slot >= 0) {
          const int zs = t.he_slot_of_z[z];
          if (zs >= 0) { tab = &t.he[kHeNumHadrons * zs + he_slot]; }
        }
        // Z == 1 needs no table at all - a hadron-proton scatter is closed form - so a missing
        // table for hydrogen is not a missing answer. For Z > 1 it is, and so is a missing
        // grid; the fall-back is then Gheisha rather than a zero, which is what the model
        // itself does below 400 MeV, and `he_fell_back` reports how often.
        if (t.he_grid == nullptr || t.he_bnd == nullptr || (tab == nullptr && z != 1)) {
          he_fb = true;
          return el::hadron_elastic_sample_invariant_t<real_t>(pdg, plab, z, a, p_local_tmax,
                                                               rr);
        }
        int hidx = -1;
        const real_t tt = el::he_sample_invariant_t<real_t>(
            pdg, plab, z, a, h_mass, p_local_tmax, tab, *t.he_grid, *t.he_bnd, rr, &he_fb,
            &he_unknown, &hidx);
        if (he_fb) {
          return el::hadron_elastic_sample_invariant_t<real_t>(pdg, plab, z, a, p_local_tmax,
                                                               rr);
        }
        return tt;
      }
      return el::hadron_elastic_sample_invariant_t<real_t>(pdg, plab, z, a, p_local_tmax, rr);
    };
    (void)el::hadron_elastic_apply_yourself<real_t, 1>(p, n, sampler, nuclear_mass,
                                                        recoil_species, thr, 0, r, o);
  };

  const el::ElasticStepResult<real_t> r = el::elastic_post_step_do_it<real_t, 1>(
      projectile, target, in_dir, real_t(1), true, /*has_at_rest_processes=*/false,
      hp::HadXsType::kNoIntegral, xs_now, xs_at_step_start, recoil_threshold, apply, rng, &fs);

  out.interacted = r.interacted;
  out.dir = r.momentum_direction;
  out.energy = r.energy;
  out.primary_survives = (r.status == hp::TrackStatusChange::kAlive);
  out.edep = r.local_energy_deposit;
  out.dropped_secondaries = r.dropped_secondaries + fs.secondary_overflow;
  out.chips_fell_back = chips_fb;
  out.he_fell_back = he_fb;
  out.he_unknown_hadron = he_unknown;
  if (r.n_secondaries > 0) {
    out.emit_recoil = true;
    out.recoil_z = r.recoil.z;
    out.recoil_a = r.recoil.a;
    out.recoil_ekin = r.recoil.kin_energy;
    out.recoil_dir = r.recoil.direction;
  }
  return out;
}

}  // namespace g4gpu::had
