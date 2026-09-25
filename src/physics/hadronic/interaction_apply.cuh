// P15: the interaction kernel's body - one queued interaction, from the model choice to the
// secondaries, and the per-thread pool it runs in.
//
// This is the ONLY file in the port that includes all four inelastic entry points, and that is
// the point of it: `physics/stepper.cuh` includes `interaction_queue.cuh`, which includes none
// of them, so no stepping kernel carries a byte of model code. docs/RISK.md V188 has the
// measurement that made that a rule rather than a preference.
//
// ---------------------------------------------------------------------------------------------
// WHAT IS TRANSCRIBED HERE
//
// `G4HadronicProcess::PostStepDoIt` (management/src/G4HadronicProcess.cc:324-482), in its own
// order, because every step of it either draws a uniform or decides which array a later step
// reads:
//
//   1. `theNumberOfInteractionLengthLeft = -1`      - not carried; see `decay_in_flight_length`
//   2. if `fXSType != fHadNoIntegral`: recompute the cross section at the END of the step and
//      reject with `xs < theLastCrossSection*G4UniformRand()`. ONE uniform, and it comes FIRST.
//   3. `theCrossSectionDataStore->SampleZandA(...)` - the target, from the partial sums the
//      RECOMPUTE left behind, not from the ones the interaction length was drawn with.
//   4. `thePro.Initialise(aTrack)`
//   5. `ChooseHadronicInteraction(...)` = `G4EnergyRangeManager::GetHadronicInteraction`, which
//      draws one uniform if and only if two models overlap here.
//   6. `do { result = ApplyYourself(...); result = CheckResult(...); } while(!result)`, bounded
//      at 100 re-entries, past which Geant4 raises `had006`.
//   7. K0 / anti-K0 among the secondaries are mixed to K0S / K0L with ONE uniform each.
//   8. `FillResult(result, aTrack)`.
//
// and `G4HadronStoppingProcess::AtRestDoIt` through `stopping::at_rest`, which P12 ported whole
// and which this file only has to call and drain.
//
// ---------------------------------------------------------------------------------------------
// THE POOL, AND WHY IT IS ONE STRUCT AND NOT TWENTY ARRAYS
//
// `InteractionSlot` holds every per-thread buffer the four arms need BY VALUE, so the pool is
// one `cudaMalloc` of `n_slots * sizeof(InteractionSlot)` and a thread's slot is
// `&slots[tid]` - the shape `ftf::entry::Handle` already has, and for the same reasons:
//
//   * ONE constructed host image, uploaded into every slot, and NOT a `cudaMemset`.
//     docs/RISK.md V104 is a member that only a constructor sets and that decides physics when
//     it is zero, and `bert::NucleiModel` and `bic::CascadeBuffers` are the same shape of risk.
//   * The capacities are the ones the models' own campaigns were validated at -
//     `tests/test_bic_apply.cu`'s 256 nucleons / 512 cascade tracks / 2048 collisions and
//     `tests/test_stopping.cu`'s 4096 / 1024 / 512 de-excitation buffers - so a slot is inside
//     the envelope those 1.96 million cascades and 1,000,000 captures were run in. Shrinking
//     them would be choosing a new envelope with no campaign behind it.
//
// FTFP keeps its OWN pool, because it has its own contract: `ftf::entry::build` allocates,
// initialises and uploads `entry::Workspace` slots and hands back a handle, and re-implementing
// that here would be exactly the duplication `ftf/ftf_entry.cuh` exists to prevent (docs/RISK.md
// V145). The two pools are allocated with the same `n_slots` and indexed by the same `tid`.
//
// `entry::Workspace` and not `entry::HadronWorkspace`: the ion arm is real here - a GenericIon
// above 3 GeV per nucleon goes to FTFP - and `HadronWorkspace` refuses an ion by capacity
// (`kMaxProjA = 1`). 81,008 bytes a slot more, stated rather than discovered.
#pragma once

#include "physics/hadronic/bertini/cascade_interface.cuh"
#include "physics/hadronic/bic/binary_cascade.cuh"
#include "physics/hadronic/bic/light_ion_reaction.cuh"
#include "physics/hadronic/ftf/ftf_entry.cuh"
#include "physics/hadronic/interaction_queue.cuh"
#include "physics/hadronic/stopping/stopping_process.cuh"
#include "physics/hadronic/wiring.cuh"

namespace g4gpu::had {

namespace bert = g4gpu::physics::hadronic::bert;
namespace stop = g4gpu::physics::hadronic::stopping;
namespace ftfe = g4gpu::hadronic::ftf::entry;

/// `kMaxSecondaries` of the final states in a slot. 256 is what `tests/test_stopping.cu` ran
/// 1,000,000 at-rest captures at and comfortably above the 128 `bic::kBlirMaxSecondaries` a
/// fusion event can fill; an overflow is counted by name (`kInelasticSecondaryOverflow`), never
/// silent.
inline constexpr int kInteractionSecondaryCap = 256;

/// The Binary cascade's list capacities in a slot - `tests/test_bic_apply.cu`'s ION envelope,
/// which contains the nucleon one. See `InteractionSlot`'s cascade block.
inline constexpr int kCascadePoolCap = 1024;
inline constexpr int kCascadeCollisionCap = 8192;
inline constexpr int kCascadeProductCap = 512;
inline constexpr int kCascadePrecoCap = 256;

/// Everything ONE thread inside the interaction kernel needs, by value.
template <typename real_t>
struct InteractionSlot {
  // ---- Bertini: `bert::apply_yourself`'s seven caller-owned buffers.
  bert::NucleiModel bert_model;
  bert::BertiniWorkspace bert_ws;
  bert::CollisionOutput co_global;
  bert::CollisionOutput co_out;
  bert::CollisionOutput co_dex;
  bert::CollisionOutput co_tmp;
  bert::ColliderOutput epo;

  // ---- Binary cascade: what `bic::BicStorage` and `bic::BlirStorage` point at.
  //
  // ONE SET OF CASCADE ARRAYS FOR BOTH ENTRY POINTS, sized to the LARGER envelope. The nucleon
  // arm (`bic::apply_yourself`) and the ion arm (`G4BinaryLightIonReaction::Interact`, P9e) run
  // in different kernels and never in the same slot at once, so they share the target nucleus,
  // the fields, the track pool, the collision list and the product lists; the capacities are
  // `tests/test_bic_apply.cu`'s ION envelope - 1,024 tracks, 8,192 collisions, 512 products and
  // 256 PreCompound products - because P9e's 800,000-event campaign measured that "every case in
  // the campaign that takes the cascade arm reads all of it", and the nucleon arm's own
  // envelope (512 / 2,048 / 256 / 64) fits inside it. The ion arm's own arrays follow. The cost,
  // measured with `sizeof`: 411,136 bytes a slot, 1,069,920 -> 1,481,056.
  bic::Nucleon nucleons[256];
  deex::Vec3d nucleus_mom[256];
  double fermi_p[256];
  bic::NucleusSortEntry sort_sums[256];
  double flat_block[bic::kFlatBlock];
  double proton_field[bic::kMaxFieldTable];
  double neutron_field[bic::kMaxFieldTable];
  bic::CascadeTrack cascade_pool[kCascadePoolCap];
  bic::imr::CollisionInitialState collisions[kCascadeCollisionCap];
  bic::CascadeBuffers cascade_buffers;
  bic::CascadeProduct products[kCascadeProductCap];
  bic::CascadeProduct preco_products[kCascadePrecoCap];
  // ---- the ion arm's own: the PROJECTILE nucleus (the target is `nucleons` above, and both
  // nucleus builds share `nucleus_mom`/`fermi_p`/`sort_sums`/`flat_block` on purpose - see
  // `bic::BlirStorage`), `SortResult`'s two lists, and `Interact`'s one-track-per-nucleon
  // secondary list, which P9e measured as a 38 kB stack local that killed its test when it was one.
  bic::Nucleon projectile_nucleons[256];
  bic::BlirProduct spectators[512];
  bic::BlirProduct cascaders[512];
  bic::CascadeTrack initial[256];

  // ---- PreCompound and P3's de-excitation, the tail of all four arms.
  deex::Fragment evap_list[4096];
  deex::Fragment evap_results[1024];
  deex::Fragment evap_step[512];
  deex::DeexProduct deex_products[1024];
  deex::DeexProduct preco_out[1024];

  // ---- the final states, which are far too big for a kernel stack: one
  // HadFinalState<real_t, 256> alone is about 18 kB against `run_step_hadron`'s whole 3,936 B
  // frame.
  physics::hadronic::HadFinalState<real_t, kInteractionSecondaryCap> fs;
  physics::hadronic::HadFinalState<real_t, kInteractionSecondaryCap> nuclear_fs;
  physics::hadronic::HadronicStepResult<real_t, kInteractionSecondaryCap> filled;
  /// The DEFINITION mass of each secondary, which `fill_result` and `check_result` compare the
  /// model's dynamic mass against. Computed per secondary by `definition_mass_of`.
  real_t pdg_mass[kInteractionSecondaryCap];
  /// The two models with final-state types of their own.
  bic::BicFinalState bic_fs;
  bic::BlirFinalState blir_fs;
};

/// What a launch is handed by value.
template <typename real_t>
struct InteractionPool {
  InteractionSlot<real_t>* slots = nullptr;
  int n_slots = 0;
  /// `bic::imr::build_concrete_channels`' 306 channels - read-only and identical for every
  /// thread, so ONE device copy for the whole run (9,792 B), exactly as `LundTables` is one.
  const bic::imr::ConcreteChannel* bic_channels = nullptr;
  int n_bic_channels = 0;
  /// FTFP's own pool, from `ftf::entry::build`. Same `n_slots`, same `tid`.
  ftfe::Handle<ftfe::Workspace> ftf{};
  /// P3's Fermi break-up table, which every arm's de-excitation tail walks.
  ///
  /// HERE AND NOT IN `had::HadronicWiring`, and the reason is the include graph rather than
  /// taste: `deex::FermiPool` lives in `deexcitation/fermi_breakup.cuh`, and putting it on the
  /// struct every stepping kernel takes by value would put P3's de-excitation headers inside
  /// `physics/stepper.cuh`. `level_data` can sit there because `data::LevelTable` is a view
  /// struct in `data/`, with no model behind it. docs/RISK.md V188.
  deex::FermiPool fermi{};

  __host__ __device__ InteractionSlot<real_t>* slot(int i) const {
    return (slots != nullptr && i >= 0 && i < n_slots) ? &slots[i] : nullptr;
  }
  __host__ __device__ bool ok() const { return slots != nullptr && n_slots > 0; }
};

/// `bic::BicStorage` pointed at one slot's arrays. A value, built per call: `BicStorage` is
/// thirteen pointers and two ints, and building it in the kernel keeps the pool a plain array
/// of PODs with no device-side pointer fix-up.
template <typename real_t>
__host__ __device__ inline bic::BicStorage bic_storage_of(InteractionSlot<real_t>& s,
                                                          const InteractionPool<real_t>& pool,
                                                          const preco::PrecoWorkspace& pws) {
  bic::BicStorage st;
  st.nucleons = s.nucleons;
  st.scratch.momentum = s.nucleus_mom;
  st.scratch.fermi_p = s.fermi_p;
  st.scratch.test_sums = s.sort_sums;
  st.scratch.flat_block = s.flat_block;
  st.scratch.capacity = 256;
  st.proton_field = s.proton_field;
  st.neutron_field = s.neutron_field;
  st.field_capacity = bic::kMaxFieldTable;
  st.cascade.pool = s.cascade_pool;
  st.cascade.pool_capacity = kCascadePoolCap;
  st.cascade.collisions = s.collisions;
  st.cascade.collision_capacity = kCascadeCollisionCap;
  st.cascade.channels = pool.bic_channels;
  st.cascade.n_channels = pool.n_bic_channels;
  st.cascade.buffers = &s.cascade_buffers;
  st.cascade.products = s.products;
  st.cascade.product_capacity = kCascadeProductCap;
  st.cascade.preco_products = s.preco_products;
  st.cascade.preco_capacity = kCascadePrecoCap;
  st.preco = &pws;
  return st;
}

/// `bic::BlirStorage` pointed at one slot's arrays - P9e's `G4BinaryLightIonReaction` entry.
///
/// The same target nucleus, scratch, fields and cascade lists `bic_storage_of` hands the nucleon
/// arm, plus the ion arm's own four. `scratch` is ONE object serving both nucleus builds, which
/// is P9e's contract and not an economy: `projectile3dNucleus->Init` and
/// `target3dNucleus->Init` run back to back on one random stream (`bic::BlirStorage`'s header).
template <typename real_t>
__host__ __device__ inline bic::BlirStorage blir_storage_of(InteractionSlot<real_t>& s,
                                                            const InteractionPool<real_t>& pool) {
  bic::BlirStorage st;
  st.projectile_nucleons = s.projectile_nucleons;
  st.target_nucleons = s.nucleons;
  st.scratch.momentum = s.nucleus_mom;
  st.scratch.fermi_p = s.fermi_p;
  st.scratch.test_sums = s.sort_sums;
  st.scratch.flat_block = s.flat_block;
  st.scratch.capacity = 256;
  st.proton_field = s.proton_field;
  st.neutron_field = s.neutron_field;
  st.field_capacity = bic::kMaxFieldTable;
  st.cascade.pool = s.cascade_pool;
  st.cascade.pool_capacity = kCascadePoolCap;
  st.cascade.collisions = s.collisions;
  st.cascade.collision_capacity = kCascadeCollisionCap;
  st.cascade.channels = pool.bic_channels;
  st.cascade.n_channels = pool.n_bic_channels;
  st.cascade.buffers = &s.cascade_buffers;
  st.cascade.products = s.products;
  st.cascade.product_capacity = kCascadeProductCap;
  st.cascade.preco_products = s.preco_products;
  st.cascade.preco_capacity = kCascadePrecoCap;
  st.spectators = s.spectators;
  st.spectator_capacity = 512;
  st.cascaders = s.cascaders;
  st.cascader_capacity = 512;
  st.initial = s.initial;
  st.initial_capacity = 256;
  return st;
}

/// `preco::PrecoWorkspace` pointed at one slot's arrays.
template <typename real_t>
__host__ __device__ inline preco::PrecoWorkspace preco_workspace_of(InteractionSlot<real_t>& s) {
  preco::PrecoWorkspace w;
  w.deex.evap_list = s.evap_list;
  w.deex.evap_capacity = 4096;
  w.deex.results = s.evap_results;
  w.deex.results_capacity = 1024;
  w.deex.step = s.evap_step;
  w.deex.step_capacity = 512;
  w.deex.products = s.deex_products;
  w.deex.products_capacity = 1024;
  w.products = s.preco_out;
  w.products_capacity = 1024;
  return w;
}

/// Scratch a `MaterialComposition` points at. Three small arrays, on the caller's stack.
template <typename real_t>
struct CompositionScratch {
  int n_isotopes[data::kMaxElements];
  int isotope_offset[data::kMaxElements];
  int element_z[data::kMaxElements];
  bool natural[data::kMaxElements];
};

/// `data::Material` as P5's `MaterialComposition`, which is what `G4ElementSelector` asks for.
///
/// The isotope arrays are the compiled-in NIST table's own and are not copied: `isotope_a` and
/// `isotope_abundance` point straight at `data/isotope_abundance.hh`'s flat arrays, and the
/// per-element `isotope_offset` is the table's own offset for that element's Z. That is the
/// same pair of pointers `nist_element_isotopes` hands the cross-section side, so the element
/// selector and `SampleZandA` draw from ONE table rather than from two copies of it.
///
/// `natural_abundance` is true for every element, because `G4NistElementBuilder::BuildElement`
/// ends with `theElement->SetNaturalAbundanceFlag(true)` - the same line
/// `nist_element_isotopes`' own header records.
///
/// An element whose Z has no NIST row gets `n_isotopes = 0`, which `select_z_and_a` reports as
/// `SelectorRefusal` rather than drawing an A of zero. A nucleus with no mass number is the
/// failure this project keeps writing up.
template <typename real_t>
__host__ __device__ inline physics::hadronic::MaterialComposition<real_t>
material_composition_of(const data::Material<real_t>& m, CompositionScratch<real_t>& sc) {
  physics::hadronic::MaterialComposition<real_t> c;
  const int n = (m.n_elements < data::kMaxElements) ? m.n_elements : data::kMaxElements;
  for (int i = 0; i < n; ++i) {
    const int z = static_cast<int>(m.z[i] + real_t(0.5));
    sc.element_z[i] = z;
    sc.n_isotopes[i] = data::nist_element_n_isotopes(z);
    sc.isotope_offset[i] = (sc.n_isotopes[i] > 0) ? data::nist_element_iso_offset()[z] : 0;
    sc.natural[i] = true;
  }
  c.n_elements = n;
  c.element_z = sc.element_z;
  c.n_atoms_per_volume = m.n_atoms;
  c.n_isotopes = sc.n_isotopes;
  c.isotope_offset = sc.isotope_offset;
  c.natural_abundance = sc.natural;
  c.isotope_a = data::nist_element_iso_a();
  c.isotope_abundance = data::NistIsotopeAbundance<real_t>::values();
  return c;
}

/// `G4NucleiProperties::GetNuclearMass(A, Z)` as a functor.
///
/// A STRUCT AND NOT A LAMBDA. `stopping::at_rest` takes the mass function as a callable, and the
/// only caller that can supply one here is a `__global__` kernel compiled by
/// `build_engine_unit.bat`, which passes no `--extended-lambda`. A `__device__`-annotated lambda
/// would not compile there and an unannotated one inside the kernel would - but then the
/// interaction kernel and the host tests would be handing `at_rest` two different types for the
/// same function, which is how a host/device comparison stops comparing anything.
struct NuclearMassMeV {
  template <typename T = double>
  __host__ __device__ double operator()(int a, int z) const {
    return deex::nuclear_mass(a, z);
  }
};

/// The DEFINITION mass of a secondary, which is what `FillResult`'s 1 keV shell test and
/// `CheckResult`'s off-shell test compare the model's dynamic mass against.
///
/// Two branches and they are the two kinds of secondary a hadronic model emits. For a NUCLEUS
/// (`a > 0`) it is `G4NucleiProperties::GetNuclearMass(A, Z)` with no excitation on it - the
/// model's own `s.mass` carries `M + E*`, which is exactly the difference the shell test is
/// looking at. For everything else it is `G4ParticleDefinition::GetPDGMass()`, from
/// `core/particle.cuh`'s table by PDG code; a code with no row there answers with the model's
/// own mass, which makes the test pass rather than inventing a number, and the secondary is
/// refused at emission a few lines later anyway.
template <typename real_t>
__host__ __device__ inline real_t definition_mass_of(
    const physics::hadronic::HadSecondary<real_t>& s) {
  if (s.a > 0) { return static_cast<real_t>(deex::nuclear_mass(s.a, s.z)); }
  const ParticleType t = particle_type_of_pdg(s.pdg);
  if (t == ParticleType::kNumTypes) { return s.mass; }
  return particle_def<real_t>(t).mass;
}

/// Moves a model's own final state into the slot's common one.
///
/// The Binary cascade and the light-ion reaction have final-state types of their OWN -
/// `HadFinalState<double, 128>` both, through `bic::kBicMaxSecondaries` and
/// `bic::kBlirMaxSecondaries` - while Bertini and FTFP fill the caller's. Rather than
/// instantiate those two models at 256 (which would change a capacity two validated campaigns
/// were run at), their answers are copied across here, and a source that produced more than the
/// destination holds is counted rather than truncated.
template <typename real_t, int kSrc, int kDst>
__host__ __device__ inline void copy_final_state(
    const physics::hadronic::HadFinalState<double, kSrc>& src,
    physics::hadronic::HadFinalState<real_t, kDst>& dst) {
  dst.clear();
  dst.status = src.status;
  dst.energy_change = static_cast<real_t>(src.energy_change);
  dst.momentum_change = Vec3<real_t>{static_cast<real_t>(src.momentum_change.x),
                                     static_cast<real_t>(src.momentum_change.y),
                                     static_cast<real_t>(src.momentum_change.z)};
  dst.local_energy_deposit = static_cast<real_t>(src.local_energy_deposit);
  dst.secondary_overflow = src.secondary_overflow;
  for (int i = 0; i < src.n_secondaries; ++i) {
    const physics::hadronic::HadSecondary<double>& a = src.secondaries[i];
    physics::hadronic::HadSecondary<real_t> b;
    b.pdg = a.pdg;
    b.z = a.z;
    b.a = a.a;
    b.mass = static_cast<real_t>(a.mass);
    b.kin_energy = static_cast<real_t>(a.kin_energy);
    b.direction = Vec3<real_t>{static_cast<real_t>(a.direction.x),
                               static_cast<real_t>(a.direction.y),
                               static_cast<real_t>(a.direction.z)};
    b.time = static_cast<real_t>(a.time);
    b.weight = static_cast<real_t>(a.weight);
    b.creator_model_id = a.creator_model_id;
    if (!dst.add_secondary(b)) { ++dst.secondary_overflow; }
  }
}

/// What one queued interaction did. Every field is a number the run reports.
struct InteractionOutcome {
  bool ran = false;             ///< a final state was produced and applied
  bool rejected_by_integral_xs = false;  ///< step 2 above: no interaction, the primary lives on
  bool primary_survives = true; ///< FillResult said `fAlive` (or the model did nothing)
  InelasticModel model = InelasticModel::kNone;
  HadronicRefusal refusal = HadronicRefusal::kNumHadronicRefusals;  ///< kNum... = none
  int n_secondaries = 0;
  int n_emitted = 0;
  int attempts = 0;             ///< the `do { } while(!result)` re-entry count
  int target_z = 0, target_a = 0;
};

// =============================================================================================
// The four arms
// =============================================================================================

// ---------------------------------------------------------------------------------------------
// ONE `__noinline__` FUNCTION PER MODEL, AND THAT IS A COMPILER REQUIREMENT RATHER THAN A STYLE
//
// These four were one `run_one_model` with a four-way switch, which is the shape the code wants
// to have. Compiling `transport_run_interaction.cu` that way drove ptxas past **15.5 GB of
// working set in 120 seconds** and it was still climbing when it was stopped - the same curve
// docs/RISK.md V188 measured for the inline-into-the-stepper arrangement, which reached 33.5 GB
// and never finished.
//
// One function with four 255-register, ten-to-thirteen-kilobyte call trees inside it is one
// problem for ptxas to solve; four functions behind call boundaries are four smaller ones. The
// STACK FRAME is unchanged either way - a kernel's frame is the maximum over its call tree
// however the tree is split - so what this buys is the compiler finishing, which is what V55,
// V63 and V65 are each about in their own way.
//
// If any of these four ever loses its `__noinline__`, the unit stops compiling. That is the
// anti-vacuity check for this comment and it was run.
// ---------------------------------------------------------------------------------------------

/// `G4TheoFSGenerator("FTFP")::ApplyYourself`, through P11d's entry contract.
template <typename real_t, typename Rng>
__host__ __device__ __noinline__ bool run_arm_ftfp(
    const physics::hadronic::HadProjectile<real_t>& proj,
    const physics::hadronic::HadNucleus& tgt, InteractionSlot<real_t>& s,
    const InteractionPool<real_t>& pool, int slot_index, Rng& rng, InteractionOutcome& out) {
  ftfe::Report rep;
  const ftfe::Status st =
      ftfe::apply<real_t, kInteractionSecondaryCap>(pool.ftf, slot_index, proj, tgt, s.fs, rep,
                                                    rng);
  // `kPrimaryUnchanged` IS a final state - `G4VPartonStringModel::Scatter`'s 1,000-attempt
  // fallback returns the primary, and Geant4 calls that the answer with a JustWarning.
  //
  // THIS COMMENT WAS HERE BEFORE THE CODE AGREED WITH IT. It went on "treating it as nothing
  // happened would be right and treating it as a hole would not" above a line that returned
  // false for it with `kFtfpRefused` - a hole, and the refusal's disposal kills the track and
  // deposits its energy where it stands. The entry contract says a final state came back (it
  // is tested before the generic refusal for exactly that reason), so it is applied like any
  // other; `tests/test_inelastic_transport.cu` section 8 asserts what it holds.
  if (st == ftfe::Status::kRan || st == ftfe::Status::kPrimaryUnchanged) { return true; }
  out.refusal = (st == ftfe::Status::kNoWorkspaceSlot) ? HadronicRefusal::kInteractionNoSlot
                                                       : HadronicRefusal::kFtfpRefused;
  return false;
}

/// `G4CascadeInterface::ApplyYourself`, with `usePreCompoundDeexcitation()`.
///
/// That choice is every instance QBBC's inelastic chain builds: `theBERT` and `theBERT1` in
/// `G4HadronInelasticQBBC::ConstructProcess` and the kaon/hyperon one in
/// `G4HadronicBuilder::BuildFTFP_BERT` all call it. The only `kCascade` instance in the whole
/// physics list is the muon's at-rest one, which is `stopping::at_rest`'s and not this
/// function's - `stopping_deexcite_choice` picks it there.
template <typename real_t, typename Rng>
__host__ __device__ __noinline__ bool run_arm_bertini(
    const physics::hadronic::HadProjectile<real_t>& proj,
    const physics::hadronic::HadNucleus& tgt, InteractionSlot<real_t>& s,
    const preco::PrecoWorkspace& pws, const data::LevelTable& lt, const deex::FermiPool& fpool,
    Rng& rng, InteractionOutcome& out) {
  const bert::CascadeParams par = bert::default_cascade_params();
  const bert::InterfaceLimits lim = bert::default_interface_limits();
  const bert::ApplyResult r = bert::apply_yourself<real_t, kInteractionSecondaryCap>(
      proj, tgt, s.fs, bert::DeexciteChoice::kPreCompound, par, lim, s.bert_model, s.co_global,
      s.co_out, s.co_dex, s.co_tmp, s.epo, s.bert_ws, lt, fpool, pws,
      /*secondary_model_id=*/0, rng);
  // `NoInteraction` IS AN ANSWER, AND IT IS GEANT4'S. `G4CascadeInterface::ApplyYourself`
  // gives up after `maximumTries` = 20 attempts that produced no collision and calls
  // `NoInteraction(aTrack)`, which leaves the track ALIVE with its energy and direction
  // unchanged; `bert::apply_yourself` builds exactly that into `fs` (`kIsAlive`,
  // `energy_change` = the kinetic energy) and says so with `no_interaction`. This arm used to
  // test `!r.no_interaction` as well, which turned Geant4's "nothing happened" into a refusal
  // - and the refusal's disposal KILLS the track and deposits its energy on the spot. Asserted in
  // `tests/test_inelastic_transport.cu` section 8.
  //
  // What IS a refusal: `refusal != kNone` - which carries the cascader's own, `kFate`
  // included, as `kCascader` - and `would_throw`, `throwNonConservationFailure`, which in
  // Geant4 ends the job and therefore has no final state this port could claim is Geant4's.
  if (r.refusal == bert::InterfaceRefusal::kNone && !r.would_throw) { return true; }
  out.refusal = HadronicRefusal::kBertiniRefused;
  return false;
}

/// `G4BinaryCascade::ApplyYourself`.
template <typename real_t, typename Rng>
__host__ __device__ __noinline__ bool run_arm_binary(
    const physics::hadronic::HadProjectile<real_t>& proj,
    const physics::hadronic::HadNucleus& tgt, InteractionSlot<real_t>& s,
    const InteractionPool<real_t>& pool, const preco::PrecoWorkspace& pws,
    const data::LevelTable& lt, const deex::FermiPool& fpool, Rng& rng,
    InteractionOutcome& out) {
  bic::BicRefusal ref;
  bic::BicReport rep;
  bic::apply_yourself(proj, tgt, lt, fpool, pws, bic_storage_of<real_t>(s, pool, pws), rng,
                      s.bic_fs, ref, rep);
  // `ref.any()`, AND NOT A LIST OF FLAGS. This tested `species`, `hydrogen`,
  // `preco_projectile` and `capacity` from P15's first commit, and `BicRefusal` has two more
  // that its own `any()` counts as making the final state meaningless: `nucleus` - 
  // `G4Fancy3DNucleus::Init` could not place the nucleons - and `cascade`, which is whatever
  // `Propagate` refused, `FillVoidNucleusProducts` included. Both return early with `bic_fs`
  // holding nothing a caller may apply, and this arm copied it into `s.fs` and reported a final
  // state. Found by reading P9e's `BicRefusal` beside this one; asserted in
  // `tests/test_inelastic_transport.cu` section 7, which fails with the flag list restored.
  if (ref.any()) {
    // `hydrogen` is `Propagate1H1`, which P9 refused by name and which is still not written.
    // Every other flag is `kBinaryRefused`: `species`, `preco_projectile` and `capacity`
    // are tripwires on a projectile this wiring should never send here, and `nucleus` and
    // `cascade` are the model's own refusals.
    out.refusal = ref.hydrogen ? HadronicRefusal::kBinaryHydrogenTarget
                               : HadronicRefusal::kBinaryRefused;
    return false;
  }
  copy_final_state<real_t>(s.bic_fs, s.fs);
  return true;
}

/// `G4BinaryLightIonReaction::ApplyYourself` - BOTH arms, since P9e.
///
/// Below 50 MeV per nucleon the fusion arm, which P9 ported; at or above it `Interact`, which
/// P9e ported - a `G4Fancy3DNucleus` for the projectile as well as the target, one
/// `G4KineticTrack` per projectile nucleon handed to the same `G4BinaryCascade::Propagate` the
/// nucleon arm runs, and `SortResult`'s spectators de-excited by the handler. Until P9e this
/// arm refused every ion above 50 MeV/n by name (`kLightIonCascade`), 91.5% of an 840 MeV
/// alpha's interactions, which is why the five ion processes were inactivated on BOTH sides of
/// every like-for-like column (docs/RISK.md V192).
///
/// THE MAPPING, and it cannot be `ref.any()` the way the nucleon arm's is: P9e's `any()` also
/// counts three outcomes that are Geant4's own answer, the primary returned alive and unchanged,
/// which this port applies exactly as Geant4 does:
///
///   `no_fusion`               "abort!! happens for too low energy for nuclei to fuse"
///   `no_final_state`          150 impact parameters and nothing - "no final state for:"
///   `momentum_not_conserved`  from "invalid final state for:", which prints and returns
///
/// and ONE that is not: `momentum_not_conserved` with `correction_gave_up`, the correction
/// loop that ends in `throw G4HadronicException(... "G4BinaryCasacde::ApplyCollision()")`
/// (G4BinaryLightIonReaction.cc:220). `G4HadronicProcess::PostStepDoIt` catches that and raises
/// `had006` as a FatalException (G4HadronicProcess.cc:425) - the job ends - so there is no
/// Geant4 final state to claim, and it is booked. The genuine refusals are `cascade`
/// (`Propagate` could not finish), `nucleus` (a `G4Fancy3DNucleus::Init` failed), `capacity`
/// and `anti_or_hyper`.
template <typename real_t, typename Rng>
__host__ __device__ __noinline__ bool run_arm_light_ion(
    const physics::hadronic::HadProjectile<real_t>& proj,
    const physics::hadronic::HadNucleus& tgt, InteractionSlot<real_t>& s,
    const InteractionPool<real_t>& pool, const preco::PrecoWorkspace& pws,
    const data::LevelTable& lt, const deex::FermiPool& fpool, Rng& rng, InteractionOutcome& out) {
  bic::BlirRefusal ref;
  bic::BlirReport rep;
  bic::BlirStorage store = blir_storage_of<real_t>(s, pool);
  bic::blir_apply_yourself(proj, tgt, lt, fpool, pws, store, rng, s.blir_fs, ref, rep);
  if (ref.cascade) {
    // `G4BinaryCascade::Propagate` refused inside `Interact` - `ref.cascade_ref` says which of
    // its refusals. The name is the one this hole had while `Interact` was missing, because a
    // ledger read across the two builds should show the rate falling rather than a name
    // disappearing.
    out.refusal = HadronicRefusal::kLightIonCascade;
    return false;
  }
  const bool would_throw = ref.momentum_not_conserved && rep.correction_gave_up;
  if (ref.anti_or_hyper || ref.capacity || ref.nucleus || would_throw) {
    out.refusal = HadronicRefusal::kBinaryRefused;
    return false;
  }
  copy_final_state<real_t>(s.blir_fs, s.fs);
  return true;
}

/// Runs ONE model once into `slot.fs`, chosen at COMPILE TIME.
///
/// `kModel` is a template parameter and not an argument, and that is the whole of the split:
/// `if constexpr` means a translation unit that instantiates this for `kBinary` contains no
/// FTFP, no Bertini and no light-ion code at all. A runtime switch over the same four arms is
/// what drove ptxas past 22 GB - see `InteractionBucket`'s own comment and docs/RISK.md V189.
///
/// @return false when the model refused; `out.refusal` then names which.
template <typename real_t, InelasticModel kModel, typename Rng>
__host__ __device__ inline bool run_one_model(
    const physics::hadronic::HadProjectile<real_t>& proj,
    const physics::hadronic::HadNucleus& tgt, InteractionSlot<real_t>& s,
    const InteractionPool<real_t>& pool, int slot_index, const data::LevelTable& lt,
    const deex::FermiPool& fpool, Rng& rng, InteractionOutcome& out) {
  if constexpr (kModel == InelasticModel::kFtfp) {
    return run_arm_ftfp<real_t>(proj, tgt, s, pool, slot_index, rng, out);
  } else if constexpr (kModel == InelasticModel::kBertini) {
    const preco::PrecoWorkspace pws = preco_workspace_of<real_t>(s);
    return run_arm_bertini<real_t>(proj, tgt, s, pws, lt, fpool, rng, out);
  } else if constexpr (kModel == InelasticModel::kBinary) {
    const preco::PrecoWorkspace pws = preco_workspace_of<real_t>(s);
    return run_arm_binary<real_t>(proj, tgt, s, pool, pws, lt, fpool, rng, out);
  } else if constexpr (kModel == InelasticModel::kLightIon) {
    const preco::PrecoWorkspace pws = preco_workspace_of<real_t>(s);
    return run_arm_light_ion<real_t>(proj, tgt, s, pool, pws, lt, fpool, rng, out);
  } else {
    out.refusal = HadronicRefusal::kNoInelasticModel;
    return false;
  }
}

// =============================================================================================
// G4HadronicProcess::PostStepDoIt
// =============================================================================================

/// The in-flight arm: everything between "the interaction length won" and "the final state is
/// in `slot.filled`".
///
/// Step by step against G4HadronicProcess.cc:324-482: the integral rejection first, then
/// `SampleZandA`, then the model. The MODEL CHOICE is not here - `choose_inelastic_model` runs
/// in the stepper, at enqueue, because it decides which kernel runs the entry. The stepper's own
/// note says what that reordering costs and why it is nothing.
///
/// The partial sums are filled by this function and NOT by the caller: `PostStepDoIt` recomputes
/// the cross section at the END of the step for a charged projectile and `SampleZandA` then
/// draws the element from THOSE partial sums. Handing it the stepper's pre-step array would
/// select an element by an energy the track no longer has, and nothing would complain - the same
/// trap `elastic_apply`'s own note records.
template <typename real_t, InelasticModel kModel, typename Rng>
__host__ __device__ __noinline__ InteractionOutcome run_inelastic(
    const physics::hadronic::HadProjectile<real_t>& proj, ParticleType species,
    const data::Material<real_t>& mat, const InelasticTables<real_t>& xs_tables,
    real_t xs_at_step_start, InteractionSlot<real_t>& s, const InteractionPool<real_t>& pool,
    int slot_index, const data::LevelTable& lt, const deex::FermiPool& fpool, int proj_z,
    int proj_a, Rng& rng) {
  InteractionOutcome out;

  // ---- 2. the integral-approach rejection, and it draws BEFORE anything else.
  hxs::MaterialXs<real_t> mxs{};
  const hxs::Projectile<real_t> pj = inelastic_projectile<real_t>(species, proj_z, proj_a);
  const hp::HadXsType xt = inelastic_xs_type<real_t>(pj);
  real_t xs_now = xs_at_step_start;
  if (xt != hp::HadXsType::kNoIntegral) {
    xs_now = inelastic_xs_per_volume<real_t>(xs_tables, mat, species, proj.kin_energy, proj_z,
                                             proj_a, mxs);
    if (hp::integral_xs_rejects<real_t>(xt, xs_now, xs_at_step_start, rng)) {
      out.rejected_by_integral_xs = true;
      return out;
    }
  } else {
    // A NEUTRAL projectile takes no rejection and no recompute, which is `fHadNoIntegral`'s
    // whole meaning - and it still needs the partial sums, because `SampleZandA` reads them.
    // `PostStepDoIt` gets them from the data store's own cache; here the array is rebuilt at
    // the same energy, which is the same number.
    xs_now = inelastic_xs_per_volume<real_t>(xs_tables, mat, species, proj.kin_energy, proj_z,
                                             proj_a, mxs);
  }
  if (!(xs_now > real_t(0)) || mxs.n_elements <= 0) {
    // No cross section here: the species has no process, or P2 refuses its data set. The
    // caller should never have drawn an interaction length, so this is a tripwire.
    out.refusal = HadronicRefusal::kNoInelasticModel;
    return out;
  }

  // ---- 3. the target.
  const hxs::TargetZA tgt = inelastic_sample_target<real_t>(xs_tables, mat, species,
                                                            proj.kin_energy, proj_z, proj_a,
                                                            mxs, rng);
  out.target_z = tgt.z;
  out.target_a = tgt.a;
  physics::hadronic::HadNucleus nucleus{tgt.z, tgt.a, 0};

  // ---- 5. the model was chosen at enqueue and is this kernel's `kModel`.
  out.model = kModel;

  // ---- 6. `do { ApplyYourself } while(!CheckResult)`, bounded at 100.
  //
  // The bound is Geant4's own `reentryCount > 100` and the disposal past it is not: Geant4
  // raises the `had006` FatalException and a kernel cannot throw, so it is carried out by name.
  constexpr int kMaxReentry = 100;
  const physics::hadronic::FatalEnergyCheckLevels<real_t> levels{};
  const real_t target_mass = static_cast<real_t>(deex::nuclear_mass(tgt.a, tgt.z));
  bool accepted = false;
  for (out.attempts = 0; out.attempts < kMaxReentry; ++out.attempts) {
    if (!run_one_model<real_t, kModel>(proj, nucleus, s, pool, slot_index, lt, fpool, rng,
                                       out)) {
      return out;  // the model refused by name; `out.refusal` says which
    }
    for (int i = 0; i < s.fs.n_secondaries; ++i) {
      s.pdg_mass[i] = definition_mass_of<real_t>(s.fs.secondaries[i]);
    }
    real_t delta_e = real_t(0);
    const physics::hadronic::CheckResultVerdict v =
        physics::hadronic::check_result<real_t, kInteractionSecondaryCap>(
            proj, target_mass, s.fs, levels, s.pdg_mass, &delta_e);
    if (v == physics::hadronic::CheckResultVerdict::kAccept) {
      accepted = true;
      break;
    }
  }
  if (!accepted) {
    out.refusal = HadronicRefusal::kInelasticReentryExhausted;
    return out;
  }

  // ---- 7. K0 and anti-K0 are mixed to K0S or K0L with ONE uniform each, drawn here and
  // therefore part of the stream. `core/particle.cuh` has no row for 311/-311 either way, so
  // the secondary is refused at emission whichever it becomes - but the DRAW happens, because
  // Geant4 makes it and every later draw of this interaction depends on the position.
  for (int i = 0; i < s.fs.n_secondaries; ++i) {
    const int p = s.fs.secondaries[i].pdg;
    if (p == 311 || p == -311) {
      s.fs.secondaries[i].pdg = (rng.uniform() > real_t(0.5)) ? 310 : 130;
    }
  }

  // ---- 8. FillResult. The direction it rotates the secondaries into is the TRACK's, which the
  // caller supplies; the model built them about +z.
  out.n_secondaries = s.fs.n_secondaries;
  out.ran = true;
  return out;
}

// =============================================================================================
// G4HadronStoppingProcess::AtRestDoIt
// =============================================================================================

/// `stopping::fritiof_at_rest` with the workspace type widened, and that is the only difference.
///
/// P12's own helper is hard-typed on `entry::Handle<entry::HadronWorkspace>`, which is right for
/// its campaign - an at-rest projectile is one anti-hadron, so `kMaxProjA = 1` saves 81,008
/// bytes a slot. P15 has ONE pool serving both arms, and the in-flight arm needs
/// `entry::Workspace` because a GenericIon above 3 GeV per nucleon goes to FTFP. `Workspace` is
/// strictly the more capable of the two, so the at-rest arm runs in it unchanged.
///
/// A DEBT, stated: the day `stopping/stopping_process.cuh` templates `fritiof_at_rest` on the
/// workspace type, this function goes and the call site takes P12's. It is written out here
/// rather than by editing that file because P15 owns no model, and the four statuses it maps
/// are `entry::Status`'s own - so a fifth status added to the contract breaks this switch at
/// compile time rather than silently landing in the default arm.
template <typename real_t, int kCap, typename Rng>
__host__ __device__ inline stop::StoppingRefusal fritiof_at_rest_wide(
    const ftfe::Handle<ftfe::Workspace>& h, int slot_index,
    const physics::hadronic::HadProjectile<real_t>& proj,
    const physics::hadronic::HadNucleus& target,
    physics::hadronic::HadFinalState<real_t, kCap>& out, ftfe::Report& rep, Rng& rng) {
  const ftfe::Status st = ftfe::apply(h, slot_index, proj, target, out, rep, rng);
  switch (st) {
    case ftfe::Status::kRan:             return stop::StoppingRefusal::kNone;
    case ftfe::Status::kNoWorkspaceSlot: return stop::StoppingRefusal::kFtfNoSlot;
    case ftfe::Status::kPrimaryUnchanged:return stop::StoppingRefusal::kFtfPrimaryUnchanged;
    case ftfe::Status::kRefused:         break;
  }
  if (rep.refused == g4gpu::hadronic::ftf::FtfRefusal::kDecayStrongResonances) {
    return stop::StoppingRefusal::kFtfResonanceDecay;
  }
  return stop::StoppingRefusal::kFtfRefused;
}

/// The at-rest arm: a stopped mu-, pi-, K-, Sigma-, Xi-, Omega-, pbar or nbar.
///
/// `stopping::at_rest` is P12's and is called whole. What this function supplies is the six
/// pieces of caller storage it names - the Bertini arm's seven buffers, two final states, the
/// nuclear mass function and the FTFP invoke - all out of one slot.
///
/// **THE FINAL STATE IS `fs`, NOT `nuclear_fs`.** `at_rest` builds the EM cascade's gammas and
/// the bound decay's products into `fs`, runs the nuclear model into `nuclear_fs`, and merges;
/// that split is docs/RISK.md V164's finding on the other side - `entry::apply` REPLACES the
/// final state it is handed, so the atomic cascade's gammas have to be kept somewhere it does
/// not reach.
template <typename real_t, typename Rng, typename NuclearMassFn>
__host__ __device__ __noinline__ InteractionOutcome run_at_rest(
    const physics::hadronic::HadProjectile<real_t>& proj,
    const physics::hadronic::MaterialComposition<real_t>& mat, InteractionSlot<real_t>& s,
    const InteractionPool<real_t>& pool, int slot_index, const data::LevelTable& lt,
    const deex::FermiPool& fpool, const NuclearMassFn& nuclear_mass, Rng& rng,
    stop::AtRestResult& r) {
  InteractionOutcome out;
  const preco::PrecoWorkspace pws = preco_workspace_of<real_t>(s);

  stop::BertiniArmState bs;
  bs.model = &s.bert_model;
  bs.ws = &s.bert_ws;
  bs.global_out = &s.co_global;
  bs.out = &s.co_out;
  bs.dex_out = &s.co_dex;
  bs.tmp = &s.co_tmp;
  bs.epo = &s.epo;

  const bert::CascadeParams par = bert::default_cascade_params();
  const bert::InterfaceLimits lim = bert::default_interface_limits();

  auto ftf_invoke = [&](const physics::hadronic::HadProjectile<real_t>& p,
                        const physics::hadronic::HadNucleus& t,
                        physics::hadronic::HadFinalState<real_t, kInteractionSecondaryCap>& o,
                        Rng& g) -> stop::StoppingRefusal {
    ftfe::Report rep;
    return fritiof_at_rest_wide<real_t, kInteractionSecondaryCap>(pool.ftf, slot_index, p, t, o,
                                                                  rep, g);
  };

  r = stop::at_rest<real_t, kInteractionSecondaryCap>(
      proj, mat, s.fs, &s.nuclear_fs, par, lim, bs, lt, fpool, pws, nuclear_mass, ftf_invoke,
      /*emc_model_id=*/0, /*nc_model_id=*/0, /*dio_model_id=*/0, rng);

  out.target_z = r.z;
  out.target_a = r.a;
  out.n_secondaries = s.fs.n_secondaries;
  if (r.refusal != stop::StoppingRefusal::kNone) {
    out.refusal = (r.refusal == stop::StoppingRefusal::kFtfNoSlot)
                      ? HadronicRefusal::kInteractionNoSlot
                      : HadronicRefusal::kAtRestRefused;
    // A refusal here still leaves the EM cascade's gammas in `fs`, and they are REAL: P12b
    // asserted 1,080,164 of them survive the Fritiof arm. So the caller emits them anyway and
    // the refusal says only that the NUCLEAR half is missing - which is what
    // `AtRestResult::local_deposit_MeV` being the nuclear model's alone already says.
    out.ran = (s.fs.n_secondaries > 0);
    return out;
  }
  out.ran = true;
  return out;
}

// =============================================================================================
// Applying what came back
// =============================================================================================

/// Pushes a filled result's secondaries into the emitter and puts the primary where FillResult
/// says it goes.
///
/// The mapping is `emit_decay_products`' with one addition: an inelastic model emits NUCLEI, and
/// a nucleus is the one secondary whose species does not say what it is. `a > 0` goes through
/// `push_nucleus`, which is what carries (Z, A) onto the track; everything else goes by PDG
/// code, and a code `core/particle.cuh` has no row for is booked under
/// `kInelasticSecondarySpecies` rather than dropped.
///
/// @return the number of secondaries that became tracks or bookings.
template <typename real_t, typename Emitter>
__host__ __device__ inline int emit_interaction_result(
    const physics::hadronic::HadronicStepResult<real_t, kInteractionSecondaryCap>& r,
    Emitter& em, const HadronicRefusalBooks& books) {
  int emitted = 0;
  for (int i = 0; i < r.n_secondaries; ++i) {
    const physics::hadronic::HadSecondary<real_t>& q = r.secondaries[i];
    if (q.a > 0) {
      // A nucleus, including a bare proton or neutron a cascade reports as (Z=1,A=1)/(0,1).
      // `push_nucleus` answers with one of the five light species or with kGenericIon carrying
      // its own nuclide, which is exactly what `step_hadron` needs to step it.
      em.push_nucleus(q.z, q.a, q.direction, q.kin_energy, 0);
      ++emitted;
      continue;
    }
    const ParticleType t = particle_type_of_pdg(q.pdg);
    if (t == ParticleType::kNumTypes) {
      book_refusal<real_t>(books, HadronicRefusal::kInelasticSecondarySpecies, q.kin_energy);
      continue;
    }
    em.push(t, q.direction, q.kin_energy, 0);
    ++emitted;
  }
  if (r.secondary_overflow > 0) {
    book_refusal<real_t>(books, HadronicRefusal::kInelasticSecondaryOverflow, real_t(0));
  }
  return emitted;
}

}  // namespace g4gpu::had
