// P36: the Binary cascade as data - `G4BinaryCascade::ApplyYourself` and `Propagate` cut at
// their time steps into STAGES, so that every resident cascade advances one stage per launch of
// a small phase kernel instead of running start to finish on one thread. docs/RISK.md V308 is
// the design and V309 its measurements.
//
// WHAT THIS FILE IS. A per-cascade record (`FlatCascade`) that holds everything `propagate` and
// `apply_yourself` keep in locals between two time steps, and one function per stage that runs
// exactly the lines of those two functions that lie between two cuts - calling the same
// functions they call (`nucleus_init`, `make_rk_propagation`, `get_sphere_point`,
// `build_target_list`, `find_collisions`, `absorb`, `capture_decision`, `RkPropagation::
// transport_one`, `correct_barions_on_boundary`, `apply_collision`, `fill_void_nucleus_products`,
// `get_excitation_energy`, `correct_final_p_and_e`, `find_fragments`, `bic_deexcite_fragment`,
// `decay_kinetic_tracks`, `products_add_*`) with the same arguments in the same order. Nothing
// under bic/ is edited: the one-thread transcription (`propagate`, `do_time_step`,
// `step_particles_out`, `apply_yourself`) is the thread path, byte for byte as it was, and it is
// what a straggler, a refusal and every exact oracle test still run.
//
// THE CUTS, and why the draws come out in the thread path's order. A cascade draws only in
// stages that run on its own record with its own `Philox` (the nucleus, the impact point, the
// collisions' final states, the decays, the de-excitation); the two stages that are split
// across items - the field transport of one time step, one item per (cascade, secondary) -
// draw nothing (docs/RISK.md V300). So a flat cascade consumes its generator in the order the
// thread cascade does, and the two differ only where the same arithmetic compiled into a
// different function rounds differently (V301: a product fused into an add in one context and
// not in the other). That is what the relaxed rule of V302 admits; the campaign
// (tests/test_bic_flat.cu) holds the two statistically.
//
// THE ONE SUM THE SPLIT TOUCHES. `G4RKPropagation::Transport` adds each track's momentum
// transfer to `theMomentumTranfer` in list order from zero. Here each item hands back its
// track's transfer from zero (`transfer[i]`) and `stage_post` sums them in pool order from
// zero: one `+=` a track, +0 added to a sum that never becomes -0 (P35's proof, V301), so the
// sum is the thread path's to the bit.
#ifndef G4GPU_BIC_FLAT_CASCADE_FLAT_CUH
#define G4GPU_BIC_FLAT_CASCADE_FLAT_CUH

#include <cfloat>
#include <cmath>

#include "core/rng.cuh"
#include "physics/hadronic/bic/binary_cascade.cuh"
#include "physics/hadronic/npool/nucleus_factory.cuh"

namespace g4gpu::bic::flat {

/// Where a cascade is. Every launch of a phase kernel acts on the records in its stages only.
enum Stage : int {
  kStFree = 0,       ///< no interaction in this lane
  kStModel,          ///< `apply_yourself` from its top: species, thresholds, the outer loop
  kStNucleus,        ///< the outer loop's turn: a new nucleus, then the inner loop's impacts
  kStStep,           ///< `Propagate`'s collision loop: one iteration a round
  kStFinishA,        ///< after the collision loop: the void test, Absorb, Capture
  kStStepOut,        ///< `StepParticlesOut`'s loop: one iteration a round
  kStFinishB,        ///< after it: left-overs, the excitation, `FindFragments`
  kStTail,           ///< needs a tail slot: the de-excitation, the products, the result
  kStDone,           ///< finished here; the result is in place
  kStThread          ///< handed to the one-thread path, which reruns it from its key
};

/// What the tail slot is to do for a record in `kStTail`.
enum TailWork : int {
  kTailDeexcite = 0,   ///< `DeExcite`, the final decays and the products, then the outer test
  kTailVoid = 1,       ///< `FillVoidNucleusProducts`, then the outer test
  kTailResult = 2,     ///< the outer loop has ended: pack the final state
  /// The two arms `apply_yourself` answers before its loops, whole, on the tail slot: A == 1
  /// (`apply_yourself_h1`, the thread path's own function) and a nucleon under `theBCminP`
  /// (PreCompound alone). V309: 17% of the proton beam's Binary calls.
  kTailHydrogen = 3,
  kTailPrecompound = 4
};

/// Why a cascade went to the thread path. Every one is a function of the cascade alone, so the
/// set of cascades the thread path answers is the same however the drain was batched.
enum HandOff : int {
  kHandNone = 0,
  kHandStraggler,      ///< more than K time steps of its own
  kHandCapacity,       ///< a flat list was too short (the thread path's are the ion envelope)
  kHandRefused,        ///< a refusal of any kind - the thread path books its own
  kHandHydrogen,       ///< A == 1: `Propagate1H1`, which is not cut into stages
  kHandPrecompound,    ///< a nucleon under `theBCminP`: PreCompound alone, no cascade
  kHandReentry,        ///< `CheckResult` refused 100 times
  /// P41: a tail-pool lane whose compact de-excitation workspace bound found no slot-sized one
  /// free within its bound (`TailFallback::spin_bound`) - the one tail case still rerun from
  /// its key, reported by name (docs/RISK.md V326).
  kHandTailScratch,
  kNumHandOff
};

/// `propagate`'s and `step_particles_out`'s two loops share one step shape: PRE decides the
/// time step, TRANSPORT moves the secondaries, POST does the rest of `DoTimeStep` and the mode's
/// own tail. `kModeLast` is `StepParticlesOut`'s closing `DoTimeStep(DBL_MAX)`.
enum StepMode : int { kModeMain = 0, kModeOut = 1, kModeLast = 2 };

/// Which model the record runs. The stages from `kStStep` to the de-excitation are
/// `G4BinaryCascade::Propagate` for both; the set-up, the outer loop and the packing differ.
enum FlatModel : int { kModelNucleon = 0, kModelIon = 1 };

/// One cascade's state between two stages. Capacities are template parameters so the Binary
/// unit and the light-ion unit size their own; a list that would overflow hands the cascade to
/// the thread path, whose lists are the ion envelope (`kSlot*`), so a capacity here is a
/// performance choice and never a physics one.
template <int kNuc, int kPool, int kColl>
struct FlatCascade {
  // ---- identity and the clock the straggler rule reads
  int stage = kStFree;
  int entry = -1;             ///< the queue entry (caller's numbering)
  int steps = 0;              ///< time steps of its own, over every try
  int hand_off = kHandNone;
  int tail_work = kTailDeexcite;
  Philox<double> rng{0u, 0u, 0u};

  // ---- `ApplyYourself`
  int pdg = 0;
  double kin = 0.0;
  int tz = 0, ta = 0;
  int model = 0;             ///< `FlatModel`
  int try_count = 0;          ///< `G4BinaryLightIonReaction::Interact`'s `tryCount`
  int interaction_counter = 0;
  int loop_max = 0;           ///< the inner loop's `collisionLoopMaxCount`
  int inner_tries = 0;
  bool have = false;
  int outcome = kPropagateOk;
  int n_products = 0;
  int n_preco = 0;

  // ---- `Propagate`
  BicCascadeState st;
  Nucleus3D nucleus;
  RkPropagation prop;
  imr::CollisionList colls;
  int loop_guard = 0;
  bool have_products = false;
  double excitation = 0.0;
  CascadeFragment frag;

  // ---- the current time step
  int mode = kModeMain;
  int next = -1;
  double step = 0.0;
  bool stepping = false;      ///< PRE chose a step and TRANSPORT owes this record its items
  int out_steps = 0;          ///< `StepParticlesOut`'s `counter`
  int out_resets = 0;         ///< and its `countreset`

  // ---- P39: the nucleus from the factory (docs/RISK.md V320, V321)
  unsigned nkey_lo = 0;       ///< the cascade's key - its generator's (rng_key, step) - which
  unsigned nkey_hi = 0;       ///< every nucleus it asks the factory for is keyed on
  int n_nuclei = 0;           ///< nuclei (pairs, for the light ion) asked for: the next tag
  int nuc_ready = 0;          ///< the factory has filled `nucleons` for this set-up
  NucleusReport nuc_rep;      ///< and what it reported, as `nucleus_init` would have
  int solo = 0;               ///< past K: waits for `flat_run_solo` (FlatEnv::solo)
  /// P42 (docs/RISK.md V331): the record's next stage could outgrow it, so it waits - stopped
  /// at that stage boundary, its generator where the boundary left it - to be CONTINUED on a
  /// record with the thread path's capacities (FlatEnv::grow).
  int grow = 0;

  // ---- the arrays; every pointer above points into these, inside the record
  Nucleon nucleons[kNuc];
  double proton_field[kMaxFieldTable];
  double neutron_field[kMaxFieldTable];
  CascadeTrack pool[kPool];
  imr::CollisionInitialState coll_items[kColl];
  /// The state each secondary had when PRE chose the step - `DoTimeStep`'s `kt_outside` and
  /// `kt_inside` lists, kept as a per-track tag and rebuilt in pool order by POST.
  signed char prestate[kPool];
  /// Each secondary's momentum transfer from TRANSPORT, from zero.
  deex::Vec3d transfer[kPool];
};

/// The cascade's constants: the masses `apply_yourself` puts in `CascadeSpecies`.
__host__ __device__ inline CascadeSpecies flat_species() {
  CascadeSpecies sp;
  sp.proton_mass = u::proton_mass_c2<double>();
  sp.neutron_mass = u::neutron_mass_c2<double>();
  sp.pi_plus_mass = pdg_mass_pion_charged();
  sp.pi_zero_mass = pdg_mass_pion_zero();
  return sp;
}

/// What every stage function is handed that is not the record: read-only tables and the lane's
/// own scratch. `buffers` is ONE prebuilt `CascadeBuffers` for every cascade (all ten pairs built
/// at the pool's build by the functions the thread path calls lazily), so `ensure_*_buffer` only
/// ever finds; a pair it would have to build fills no free entry and refuses by capacity, which
/// hands the cascade to the thread path rather than writing a table other lanes read.
struct FlatEnv {
  const imr::ConcreteChannel* channels = nullptr;
  int n_channels = 0;
  CascadeBuffers* buffers = nullptr;
  int k_steps = 1 << 30;      ///< the straggler bound K
  /// tests/test_bic_flat.cu's anti-vacuity switches: 1 skips `Capture`, 2 drops
  /// `G4MesonAbsorption` from `FindCollisions`. Zero in the transport.
  int perturb = 0;
  /// P39 (docs/RISK.md V321): a straggler is finished on its own lane (`flat_run_solo`) instead
  /// of being handed to the thread path - set whenever the nuclei come from the factory, whose
  /// nucleus the thread path's rerun from the key could not reproduce.
  int solo = 0;
  /// P42 (docs/RISK.md V331): a record whose next step could outgrow its pool or its collision
  /// list stops before it (`flat_room_for_step`) and is continued on a larger record, instead of
  /// refusing for capacity and being rerun from its key. Zero runs P36's capacity hand-back.
  int grow = 0;
};

/// The lists the per-cascade stages fill and empty within one call - `DoTimeStep`'s five,
/// `Capture`'s, `StepParticlesOut`'s two, `Propagate`'s initial list and `FindAndFillCluster`'s
/// three - ON THE CALLING THREAD'S FRAME, where the thread path keeps them in the slot's
/// `BicScratch` (51,200 B). A cascade's lists hold pool indices, so `P` entries bound every
/// list, and the cluster lists hold target nucleons, at most `N`.
///
/// P42 (docs/RISK.md V331): the lists' length is `min(P, kCascadeMaxScratch)` - the thread path's
/// own 256 - and no longer the pool's size. Until P42 a `static_assert(P <= kCascadeMaxScratch)`
/// tied the two, so no flat record could hold more than 256 tracks however much memory it had;
/// the thread path's slot holds 1,024 tracks with these same 256-entry lists. Every write into
/// them is bounded as the thread path bounds it (cascade_step.cuh, cascade_propagate.cuh,
/// cascade_finish.cuh): a silent cut at 256 where it cuts, a capacity refusal where it refuses
/// (`flat_prologue`'s initial list, `flat_capture`). For a pool of at most 256 nothing changes:
/// a list never holds more entries than the pool has tracks, so no bound can bind.
template <int N, int P>
struct FlatScratch {
  static constexpr int kLen = (P < kCascadeMaxScratch) ? P : kCascadeMaxScratch;
  int initial[kLen];
  int was_outside[kLen], was_inside[kLen], gone_in[kLen], gone_out[kLen], captured[kLen];
  int capture[kLen];
  int stepout_secondaries[kLen], stepout_captured[kLen];
  int cluster_charges[N];
  deex::Vec3d cluster_positions[N];
  int cluster_ids[N];
};

/// `G4MesonAbsorption::GetCollisions` for one track - `schedule_absorption` (cascade_find.cuh)
/// line for line, its `FindAndFillCluster` lists on the frame. `time_to_absorption` and
/// `find_and_fill_cluster` are the thread path's own.
template <int N, int P, typename Prop>
__host__ __device__ inline void flat_schedule_absorption(BicCascadeState& st,
                                                         imr::CollisionList& colls, int index,
                                                         const Prop& propagator,
                                                         FlatScratch<N, P>& scratch,
                                                         imr::AbsorptionRefusal& aref) {
  const CascadeTrack& pro = st.lists.pool[index];
  int n_cand = 0;
  for (int i = 0; i < st.lists.n_pool; ++i) {
    if (st.lists.pool[i].list == kListTarget) { ++n_cand; }
  }
  if (n_cand <= 1) { return; }
  for (int i = 0; i < st.lists.n_pool; ++i) {
    const CascadeTrack& tgt = st.lists.pool[i];
    if (tgt.list != kListTarget) { continue; }
    const double time = imr::time_to_absorption(pro.pdg, tgt.pdg, pro.charge, tgt.charge,
                                                pro.position, tgt.position, pro.momentum,
                                                pro.momentum, tgt.momentum, pro.actual_mass(),
                                                tgt.actual_mass(), aref);
    if (time == DBL_MAX) { continue; }
    int* charges = scratch.cluster_charges;
    deex::Vec3d* positions = scratch.cluster_positions;
    int* ids = scratch.cluster_ids;
    int n = 0;
    int first_slot = -1;
    for (int j = 0; j < st.lists.n_pool && n < N; ++j) {
      if (st.lists.pool[j].list != kListTarget) { continue; }
      charges[n] = st.lists.pool[j].charge;
      positions[n] = st.lists.pool[j].position;
      ids[n] = j;
      if (j == i) { first_slot = n; }
      ++n;
    }
    if (first_slot < 0) { continue; }
    const int partner = imr::find_and_fill_cluster(pro.charge, tgt.charge, tgt.position, charges,
                                                   positions, n, first_slot);
    if (partner < 0) { continue; }
    imr::ScatterRefusal cref;
    colls.add(time + st.current_time, index, i, kGenAbsorption, cref, ids[partner]);
  }
  (void)propagator;
}

/// `G4BinaryCascade::FindCollisions` - `find_collisions` with the absorption's lists above.
/// The decay (the one draw) and the scattering are the thread path's functions.
template <int N, int P, typename Prop, typename Rng>
__host__ __device__ inline void flat_find_collisions(BicCascadeState& st,
                                                     imr::CollisionList& colls,
                                                     const int* indices, int n,
                                                     const FlatEnv& env, const Prop& propagator,
                                                     Rng& rng, FlatScratch<N, P>& scratch,
                                                     CascadeRefusal& ref) {
  const CascadeSpecies sp = flat_species();
  for (int k = 0; k < n; ++k) {
    KineticDecayRefusal dref;
    imr::AbsorptionRefusal aref;
    imr::ScatterRefusal sref;
    schedule_decay(st, colls, indices[k], rng, dref);
    if (env.perturb != 2) {
      flat_schedule_absorption(st, colls, indices[k], propagator, scratch, aref);
    }
    schedule_scattering(st, colls, indices[k], *env.buffers, sp, propagator, sref, ref);
    if (dref.unknown_species) {
      ref.unknown_species = true;
      ref.refused_pdg = dref.refused_pdg;
    }
  }
}

template <int N, int P, int C>
__host__ __device__ inline void flat_hand_off(FlatCascade<N, P, C>& c, int why) {
  c.stage = kStThread;
  c.hand_off = why;
}

/// The record's pointers into itself, set once when a lane takes an entry.
template <int N, int P, int C>
__host__ __device__ inline void flat_bind(FlatCascade<N, P, C>& c) {
  c.nucleus.nucleons = c.nucleons;
  c.nucleus.capacity = N;
  c.st.lists.pool = c.pool;
  c.st.lists.capacity = P;
  c.st.nucleons = c.nucleons;
  c.colls.items = c.coll_items;
  c.colls.capacity = C;
}

/// A lane takes a model call: `apply_yourself`'s arguments, and the generator it draws from.
template <int N, int P, int C>
__host__ __device__ inline void flat_start(FlatCascade<N, P, C>& c, int entry, int pdg,
                                           double kin, int tz, int ta,
                                           const Philox<double>& rng) {
  c.stage = kStModel;
  c.entry = entry;
  c.steps = 0;
  c.hand_off = kHandNone;
  c.model = kModelNucleon;
  c.try_count = 0;
  c.pdg = pdg;
  c.kin = kin;
  c.tz = tz;
  c.ta = ta;
  c.rng = rng;
  c.st = BicCascadeState{};
  c.nucleus = Nucleus3D{};
  c.nkey_lo = 0u;
  c.nkey_hi = 0u;
  c.n_nuclei = 0;
  c.nuc_ready = 0;
  c.solo = 0;
  c.grow = 0;
  flat_bind(c);
}

/// A cascade past the straggler bound K. P36 hands it to the thread path, which reruns the
/// entry from its key - exactly the flat cascade's answer while the flat cascade drew its nucleus
/// from that key's generator. With the nuclei from the factory the rerun is a DIFFERENT cascade,
/// so the hand-back would replace every long cascade by an ordinary one: P39's first campaign
/// (docs/RISK.md V321) measured it - 192 of 640,000 calls, `StepParticlesOut`'s 100 x 100 steps
/// around a slow hyperon, and the pi+ 800 MeV points' K+ and Lambda multiplicities 8.7-9.9
/// sigma under Geant4's. So with `env.solo` the cascade is flagged instead, and finished on its
/// own lane by `flat_run_solo` - the same stage functions in the same order, its own generator.
template <int N, int P, int C>
__host__ __device__ inline void flat_straggle(FlatCascade<N, P, C>& c, const FlatEnv& env) {
  if (env.solo) {
    c.solo = 1;
  } else {
    flat_hand_off(c, kHandStraggler);
  }
}

// =============================================================================================
// P39: the nucleus from the factory (npool/nucleus_factory.cuh, docs/RISK.md V320, V321).
//
// THE KEY. Every nucleus a cascade asks for is keyed on the cascade - the (rng_key, step) its
// own generator is made from - and on its index among the cascade's nuclei (the tag), never on
// the lane, the round or the request's place in a list. So a rerun gets the same nucleus, the
// next outer-loop turn, `CheckResult` re-entry or light-ion try a fresh one, and two cascades
// never share one. The factory's draws are no longer the cascade generator's: the impact points
// and everything after draw from `rng` as before, minus the nucleus's ~1,522 uniforms - a
// different stream of the same law, which is what the relaxed rule of V302 admits and the
// campaigns hold statistically (P38 proved the per-nucleon phases are `Init`'s law, V319).
//
// THE FALL-BACK, by name: a cascade past `kFactoryMaxTag` nuclei, a nuclide outside A 1..64,
// or the factory switched off (`G4GPU_FLAT_FACTORY=0`) samples on its lane with
// `nucleus_init` from `rng`, as P36 did - each a function of the cascade alone. A factory
// REFUSAL (placement or ReduceSum's logic) is not a fall-back: it is the same law's refusal,
// and it hands the cascade back exactly as `nucleus_init`'s would.
// =============================================================================================

/// The cascade's key, from the caller that made its generator.
template <int N, int P, int C>
__host__ __device__ inline void flat_set_key(FlatCascade<N, P, C>& c, unsigned key_lo,
                                             unsigned key_hi) {
  c.nkey_lo = key_lo;
  c.nkey_hi = key_hi & 0xffffffu;
}

/// A request for `(a, z)` into `nucleons` / `rep`, tag `tag`; false if the factory cannot take
/// it (the caller samples on the lane).
__device__ inline bool flat_push_nucleus(const npool::FactoryLists& lists, Nucleon* nucleons,
                                         NucleusReport* rep, int a, int z, unsigned key_lo,
                                         unsigned key_hi, int tag) {
  if (tag < 0 || static_cast<unsigned>(tag) > npool::kFactoryMaxTag) { return false; }
  npool::FactoryRequest r;
  r.nucleons = nucleons;
  r.report = rep;
  r.a = a;
  r.z = z;
  r.key_lo = key_lo;
  r.key_hi = key_hi;
  r.tag = static_cast<unsigned>(tag);
  return npool::factory_push(lists, r);
}

/// The Binary set-up's request: the target of this outer-loop turn, tag = the turn's index.
template <int N, int P, int C>
__device__ inline void flat_request_target(FlatCascade<N, P, C>& c,
                                           const npool::FactoryLists& lists) {
  const int tag = c.n_nuclei++;
  c.nuc_ready = flat_push_nucleus(lists, c.nucleons, &c.nuc_rep, c.ta, c.tz, c.nkey_lo,
                                  c.nkey_hi, tag)
                    ? 1
                    : 0;
}

/// What `nucleus_init` leaves in `nuc` besides the nucleons - `Init`'s constants, the header the
/// factory itself built with - and its report, for a nucleus the factory made in `nucleons`.
__host__ __device__ inline NucleusReport flat_take_factory_nucleus(Nucleus3D& nuc,
                                                                   Nucleon* nucleons,
                                                                   int capacity, int a, int z,
                                                                   const NucleusReport& rep) {
  npool::factory_nucleus_header(nuc, nucleons, a, z);
  nuc.capacity = capacity;  // `nucleus_init` leaves the caller's capacity, not A
  return rep;
}

/// `G4BinaryCascade::ApplyYourself`'s outer-loop test, after a `Propagate` that collided:
/// `if(++interactionCounter>99) break;` then `while(products && products->size() == 0)`.
template <int N, int P, int C>
__host__ __device__ inline void flat_outer_test(FlatCascade<N, P, C>& c) {
  if (c.model == kModelIon) {
    // `Interact`'s loop: `if (result && result->size()==0) result=0;` and
    // `while (!result && tryCount < 150)` - a NULL and an empty vector both try again.
    c.have = (c.outcome != kPropagateNoCollision && c.n_products > 0);
    if (c.have || c.try_count >= 150) {
      c.stage = kStTail;
      c.tail_work = kTailResult;
    } else {
      c.stage = kStNucleus;
    }
    return;
  }
  if (++c.interaction_counter > 99 || !c.have || c.n_products > 0) {
    c.stage = kStTail;
    c.tail_work = kTailResult;
  } else {
    c.stage = kStNucleus;
  }
}

// =============================================================================================
// kStModel: `apply_yourself` above the loops.
// =============================================================================================
template <int N, int P, int C>
__host__ __device__ inline void stage_model(FlatCascade<N, P, C>& c) {
  const bool is_nucleon = (c.pdg == 2212 || c.pdg == 2112);
  const bool is_charged_pion = (c.pdg == 211 || c.pdg == -211);
  if (!is_nucleon && !is_charged_pion) { return flat_hand_off(c, kHandRefused); }
  if (c.kin < bc_min_p() && is_nucleon) {
    c.stage = kStTail;
    c.tail_work = kTailPrecompound;
    return;
  }
  if (c.ta <= 1) {
    c.stage = kStTail;
    c.tail_work = kTailHydrogen;
    return;
  }
  if (c.ta > N) { return flat_hand_off(c, kHandCapacity); }
  c.interaction_counter = 0;
  c.inner_tries = 0;
  c.have = false;
  c.outcome = kPropagateOk;
  c.n_products = 0;
  c.stage = kStNucleus;
}

// =============================================================================================
// kStNucleus: the outer loop's turn and the whole inner loop - a nucleus, its fields, and up to
// 200 impact points until one collides. `Propagate`'s prologue is here, up to the first
// `colls.size() == 0` test; a miss is cheap (no time step), so the inner loop stays in one call.
// =============================================================================================
template <int N, int P, int C>
__host__ __device__ inline bool flat_prologue(FlatCascade<N, P, C>& c, CascadeTrack* secs, int n_secs,
                                              const FlatEnv& env, FlatScratch<N, P>& scr) {
  const CascadeSpecies sp = flat_species();
  BicCascadeState& st = c.st;
  CascadeRefusal ref;
  c.colls.n = 0;
  st.lists.pool = c.pool;
  st.nucleons = c.nucleus.nucleons;
  st.lists.capacity = P;
  st.lists.n_pool = 0;
  st.n_final_pushed = 0;
  st.n_captured_pushed = 0;
  st.outer_radius = c.nucleus.outer_radius();
  st.current_time = 0.0;
  st.projectile_4mom = imr::LorentzVector(deex::Vec3d{0.0, 0.0, 0.0}, 0.0);
  st.momentum_transfer = deex::Vec3d{0.0, 0.0, 0.0};
  NucleusReport nrep;
  st.cut_on_p = cut_on_p(c.nucleus.mass(nrep));
  if (nrep.hyper_nucleus) {
    flat_hand_off(c, kHandRefused);
    return false;
  }
  build_target_list(c.nucleus, st, sp.proton_mass, sp.neutron_mass, ref);
  if (ref.any()) {
    flat_hand_off(c, ref.capacity ? kHandCapacity : kHandRefused);
    return false;
  }
  for (int i = 0; i < n_secs; ++i) {
    if (secs[i].state == kUndefined) {
      flat_hand_off(c, kHandRefused);
      return false;
    }
  }
  if (!build_late_particle_collisions(secs, n_secs, st, ref)) {
    flat_hand_off(c, kHandCapacity);
    return false;
  }
  int* initial = scr.initial;
  int n_initial = 0;
  for (int i = 0; i < st.lists.n_pool; ++i) {
    if (st.lists.pool[i].list != kListSecondary) { continue; }
    // P42: the thread path's bound (cascade_propagate.cuh), unreachable below 257 tracks.
    if (n_initial >= kCascadeMaxScratch) {
      flat_hand_off(c, kHandCapacity);
      return false;
    }
    initial[n_initial++] = i;
  }
  flat_find_collisions(st, c.colls, initial, n_initial, env, c.prop, c.rng, scr, ref);
  if (ref.any()) {
    flat_hand_off(c, ref.capacity ? kHandCapacity : kHandRefused);
    return false;
  }
  return true;
}

template <int N, int P, int C>
__host__ __device__ inline void stage_nucleus(FlatCascade<N, P, C>& c, const FlatEnv& env,
                                              Nucleus3DScratch& nsc, FlatScratch<N, P>& scr) {
  // An outer-loop turn is a nucleus and up to 200 impact points: it counts on the straggler
  // clock like a time step, because V307's slowest cascades are the ones that turn it most.
  if (++c.steps > env.k_steps) { return flat_straggle(c, env); }
  // P39: the nucleus the factory built for this turn, or (the fall-back) one sampled here.
  NucleusReport nrep;
  if (c.nuc_ready) {
    c.nuc_ready = 0;
    nrep = flat_take_factory_nucleus(c.nucleus, c.nucleons, N, c.ta, c.tz, c.nuc_rep);
  } else {
    nrep = nucleus_init(c.nucleus, nsc, c.ta, c.tz, c.rng);
  }
  if (nrep.fatal()) { return flat_hand_off(c, nrep.capacity ? kHandCapacity : kHandRefused); }
  c.prop = make_rk_propagation(c.nucleus, nrep, c.proton_field, c.neutron_field,
                               kMaxFieldTable);
  const double mass = (c.pdg == 2212)   ? u::proton_mass_c2<double>()
                    : (c.pdg == 2112) ? u::neutron_mass_c2<double>()
                                      : pdg_mass_pion_charged();
  const double e = c.kin + mass;
  const double p_mag = std::sqrt(c.kin * (c.kin + 2.0 * mass));
  const imr::LorentzVector initial4(Vec3<double>{0.0, 0.0, p_mag}, e);
  int collision_loop_max_count = 200;
  bool have = false;
  do {
    const double radius = c.nucleus.outer_radius() + 3.0 * deex::fermi();
    const Vec3<double> pos = get_sphere_point(1.1 * radius, initial4.v, c.rng);
    CascadeTrack kt;
    kt.pdg = c.pdg;
    kt.pdg_mass = mass;
    kt.charge = (c.pdg == 2212 || c.pdg == 211) ? 1 : ((c.pdg == -211) ? -1 : 0);
    kt.baryon = (c.pdg == 2212 || c.pdg == 2112) ? 1 : 0;
    kt.momentum = initial4;
    kt.position = pos;
    kt.formation_time = 0.0;
    kt.state = kOutside;
    kt.creator_model_id = bic_model_id();
    if (!flat_prologue(c, &kt, 1, env, scr)) { return; }
    ++c.inner_tries;
    // `Entries() == 0` is `Propagate`'s NULL: the impact parameter missed.
    have = (c.colls.size() != 0);
  } while (!have && --collision_loop_max_count > 0);
  c.have = have;
  if (!have) {
    c.outcome = kPropagateNoCollision;
    // `if(++interactionCounter>99) break;` and then the NULL ends the outer loop as well.
    ++c.interaction_counter;
    c.stage = kStTail;
    c.tail_work = kTailResult;
    return;
  }
  c.loop_guard = 1000000;
  c.have_products = false;
  c.mode = kModeMain;
  c.stepping = false;
  c.stage = kStStep;
}

// =============================================================================================
// The step: PRE (per cascade), TRANSPORT (per secondary), POST (per cascade).
// =============================================================================================

/// `Capture` as `capture_secondaries` does it, with the slot list it fills being the lane's.
template <int N, int P, int C>
__host__ __device__ inline bool flat_capture(FlatCascade<N, P, C>& c, const FlatEnv& env,
                                             FlatScratch<N, P>& scr) {
  if (env.perturb == 1) { return false; }
  // `capture_secondaries` (cascade_propagate.cuh) line for line, its list on the frame.
  BicCascadeState& st = c.st;
  const CaptureDecision d = capture_decision(st.lists.pool, st.lists.n_pool, st.cut_on_p, c.prop);
  if (!d.capture) { return false; }
  int* captured = scr.capture;
  int n_cap = 0;
  for (int i = 0; i < st.lists.n_pool; ++i) {
    CascadeTrack& t = st.lists.pool[i];
    if (t.list != kListSecondary || t.state != kInside) { continue; }
    if (t.pdg != imr::kPdgProton && t.pdg != imr::kPdgNeutron) { continue; }
    // P42: the thread path's bound on `captured` (cascade_propagate.cuh: a capacity refusal of
    // the whole call). Unreachable for a pool of at most 256 tracks.
    if (n_cap >= kCascadeMaxScratch) {
      flat_hand_off(c, kHandCapacity);
      return false;
    }
    push_captured(st, i);
    mark_hit(st, i);
    captured[n_cap++] = i;
  }
  c.colls.remove_tracks(captured, n_cap);
  return true;
}

/// `DoTimeStep` up to the transport: who is outside and who inside, by pool index.
template <int N, int P, int C>
__host__ __device__ inline void flat_choose_step(FlatCascade<N, P, C>& c, double step) {
  c.step = step;
  for (int i = 0; i < c.st.lists.n_pool; ++i) {
    const CascadeTrack& t = c.pool[i];
    c.prestate[i] = (t.list == kListSecondary) ? static_cast<signed char>(t.state) : -1;
  }
  c.stepping = true;
}

/// P42 (docs/RISK.md V331): can the iteration PRE is about to start finish in this record? An
/// upper bound on what one iteration adds, from the record alone:
///
///   * tracks: at most 4 - one `ApplyCollision` an iteration, in both loops, and its products
///     are `CollisionProducts::track[4]` (cascade_find.cuh); `DoTimeStep`, `Absorb`, `Capture`
///     and the finish stages add none;
///   * collisions: each new track's `FindCollisions` is one decay, at most one absorption and at
///     most one scattering per target nucleon (`schedule_*`, cascade_find.cuh), so 4 x (1 + 2 T)
///     for T targets; and a `StepParticlesOut` iteration that may reset (`counter` at 100) runs
///     `FindCollisions` over every secondary as well (at most 256: its list's bound).
///
/// Both bounds are loose by design - a record that fails them is not refused, only continued
/// elsewhere - and both are exact upper bounds, so a record that passes them cannot refuse for
/// capacity in the iteration, and the record's numbers never depend on where it ran.
template <int N, int P, int C>
__host__ __device__ inline bool flat_room_for_step(const FlatCascade<N, P, C>& c) {
  const BicCascadeState& st = c.st;
  if (c.mode == kModeLast) { return true; }  // `DoTimeStep(DBL_MAX)` then the finish: no adds
  if (st.lists.n_pool + 4 > P) { return false; }
  int n_target = 0, n_sec = 0;
  for (int i = 0; i < st.lists.n_pool; ++i) {
    n_target += (st.lists.pool[i].list == kListTarget) ? 1 : 0;
    n_sec += (st.lists.pool[i].list == kListSecondary) ? 1 : 0;
  }
  const int per_track = 1 + 2 * n_target;
  int colls = 4 * per_track;
  if (c.mode == kModeOut && c.out_steps >= 100) {
    const int reset = (n_sec + 4 < kCascadeMaxScratch) ? n_sec + 4 : kCascadeMaxScratch;
    colls += reset * per_track;
  }
  return c.colls.n + colls <= C;
}

/// PRE: the top of an iteration of `Propagate`'s loop or of `StepParticlesOut`'s, up to the
/// time step. Leaves `stepping` set when TRANSPORT owes the record a step.
template <int N, int P, int C>
__host__ __device__ inline void stage_pre(FlatCascade<N, P, C>& c, const FlatEnv& env,
                                          FlatScratch<N, P>& scr) {
  c.stepping = false;
  // P42: an iteration that could outgrow the record is not started here (FlatEnv::grow).
  if (env.grow && !flat_room_for_step(c)) {
    c.grow = 1;
    return;
  }
  BicCascadeState& st = c.st;
  if (c.mode == kModeMain) {
    // `while (Entries() && currentZ && --collisionLoopMaxCount > 0)`
    if (!(c.colls.size() > 0 && st.current_z && --c.loop_guard > 0)) {
      c.stage = kStFinishA;
      return;
    }
    AbsorbRefusal abs_ref;
    if (absorb(st.lists.pool, st.lists.n_pool, cut_on_p_absorb(), abs_ref)) {
      c.have_products = true;
    }
    if (flat_capture(c, env, scr)) { c.have_products = true; }
    if (c.stage == kStThread) { return; }
    if (c.colls.size() > 0) {
      c.next = c.colls.next_collision();
      if (c.next >= 0) {
        flat_choose_step(c, c.colls.items[c.next].collision_time - st.current_time);
      }
    }
    return;
  }
  if (c.mode == kModeLast) {
    flat_choose_step(c, DBL_MAX);
    return;
  }
  // kModeOut: `while (theSecondaryList.size() > 0)`, and the step `StepParticlesOut` takes.
  if (st.lists.count(kListSecondary) <= 0) {
    c.mode = kModeLast;
    flat_choose_step(c, DBL_MAX);
    return;
  }
  double min_time_step = 1.0e-12 * u::ns<double>();
  for (int i = 0; i < st.lists.n_pool; ++i) {
    const CascadeTrack& t = st.lists.pool[i];
    if (t.list != kListSecondary) { continue; }
    if (t.state == kInside) {
      const KineticTrack kt = as_kinetic_track(t);
      double t_dummy = 0.0;
      double t_step = 0.0;
      const bool intersect = c.prop.sphere_intersection_times(kt, t_dummy, t_step);
      if (intersect && t_step < min_time_step && t_step > 0.0) { min_time_step = t_step; }
    } else if (t.state != kOutside) {
      // `StepParticlesOut`'s throw for a secondary that is neither: the thread path's report.
    }
  }
  min_time_step *= 1.2;
  double time_to_collision = DBL_MAX;
  c.next = -1;
  if (c.colls.size() > 0) {
    c.next = c.colls.next_collision();
    if (c.next >= 0) { time_to_collision = c.colls.items[c.next].collision_time - st.current_time; }
  }
  if (time_to_collision > min_time_step) {
    c.next = -2;  ///< the plain step: no collision follows it
    flat_choose_step(c, min_time_step);
  } else {
    flat_choose_step(c, time_to_collision);
  }
}

/// TRANSPORT, one item: `G4RKPropagation::Transport` for one secondary, from the propagator the
/// cascade's field maps define, with the transfer handed back from zero.
template <int N, int P, int C>
__host__ __device__ inline void item_transport(FlatCascade<N, P, C>& c, int i) {
  RkPropagation p = c.prop;
  p.momentum_transfer = deex::Vec3d{0.0, 0.0, 0.0};
  CascadeTrack& t = c.pool[i];
  KineticTrack kt = as_kinetic_track(t);
  RkAdvanceReport rep;
  p.transport_one(kt, c.step, rep);
  from_kinetic_track(kt, t);
  c.transfer[i] = p.momentum_transfer;
}

/// The rest of `DoTimeStep` after the transport, as cascade_step.cuh has it, the two lists
/// rebuilt in pool order from `prestate`.
template <int N, int P, int C>
__host__ __device__ inline TimeStepReport flat_after_transport(FlatCascade<N, P, C>& c,
                                                               FlatScratch<N, P>& scratch) {
  BicCascadeState& st = c.st;
  imr::CollisionList& collisions = c.colls;
  const CascadeSpecies sp = flat_species();
  CascadeRefusal ref;
  TimeStepReport out;
  int* was_outside = scratch.was_outside;
  int* was_inside = scratch.was_inside;
  int n_was_out = 0;
  int n_was_in = 0;
  deex::Vec3d sum{0.0, 0.0, 0.0};
  for (int i = 0; i < st.lists.n_pool; ++i) {
    const int s = c.prestate[i];
    if (s < 0) { continue; }
    if (s == kOutside && n_was_out < kCascadeMaxScratch) { was_outside[n_was_out++] = i; }
    if (s == kInside && n_was_in < kCascadeMaxScratch) { was_inside[n_was_in++] = i; }
    sum = sum + c.transfer[i];
  }
  c.prop.momentum_transfer = sum;
  st.momentum_transfer = st.momentum_transfer + c.prop.momentum_transfer;

  int* gone_in = scratch.gone_in;
  int* gone_out = scratch.gone_out;
  int n_in = 0;
  int n_out = 0;
  for (int k = 0; k < n_was_out; ++k) {
    if (st.lists.pool[was_outside[k]].state == kInside && n_in < kCascadeMaxScratch) {
      gone_in[n_in++] = was_outside[k];
    }
  }
  for (int k = 0; k < n_was_in; ++k) {
    if (st.lists.pool[was_inside[k]].state == kGoneOut && n_out < kCascadeMaxScratch) {
      gone_out[n_out++] = was_inside[k];
    }
  }
  const BoundaryFailures fail = correct_barions_on_boundary(
      st, gone_in, n_in, gone_out, n_out, c.prop, sp.proton_mass, sp.neutron_mass, ref);
  if (ref.any()) { flat_hand_off(c, kHandRefused); }
  out.boundary_overflow = fail.overflow;
  if (fail.n > 0) {
    n_in = 0;
    n_out = 0;
    for (int k = 0; k < n_was_out; ++k) {
      if (st.lists.pool[was_outside[k]].state == kInside && n_in < kCascadeMaxScratch) {
        gone_in[n_in++] = was_outside[k];
      }
    }
    for (int k = 0; k < n_was_in; ++k) {
      if (st.lists.pool[was_inside[k]].state == kGoneOut && n_out < kCascadeMaxScratch) {
        gone_out[n_out++] = was_inside[k];
      }
    }
  }
  out.n_gone_in = n_in;
  for (int k = 0; k < n_was_out; ++k) {
    const int s = st.lists.pool[was_outside[k]].state;
    if ((s == kMissNucleus || s == kGoneOut) && n_out < kCascadeMaxScratch) {
      gone_out[n_out++] = was_outside[k];
    }
  }
  for (int k = 0; k < n_out; ++k) { push_final(st, gone_out[k]); }
  out.n_gone_out = n_out;
  int* captured = scratch.captured;
  int n_cap = 0;
  for (int i = 0; i < st.lists.n_pool; ++i) {
    const CascadeTrack& t = st.lists.pool[i];
    if (t.list == kListSecondary && t.state == kCaptured && n_cap < kCascadeMaxScratch) {
      captured[n_cap++] = i;
    }
  }
  if (collisions.size() > 0) {
    const int next = collisions.next_collision();
    if (next >= 0) {
      const int primary = collisions.items[next].primary;
      for (int k = 0; k < n_out; ++k) {
        if (gone_out[k] == primary) { out.success = false; }
      }
      for (int k = 0; k < n_cap; ++k) {
        if (captured[k] == primary) { out.success = false; }
      }
    }
  }
  collisions.remove_tracks(gone_out, n_out);
  if (n_cap > 0) {
    for (int k = 0; k < n_cap; ++k) {
      push_captured(st, captured[k]);
      mark_hit(st, captured[k]);
    }
    collisions.remove_tracks(captured, n_cap);
  }
  out.n_captured = n_cap;
  st.current_time += c.step;
  return out;
}

/// `ApplyCollision` with `Propagate`'s closure around it: the collision, and `FindCollisions`
/// for what it made.
template <int N, int P, int C>
__host__ __device__ inline bool flat_apply(FlatCascade<N, P, C>& c, int ci, const FlatEnv& env,
                                           FlatScratch<N, P>& scr) {
  const CascadeSpecies sp = flat_species();
  CascadeRefusal ref;
  auto find = [&](const int* idx, int n) {
    flat_find_collisions(c.st, c.colls, idx, n, env, c.prop, c.rng, scr, ref);
  };
  const bool ok = apply_collision(c.st, c.colls, ci, env.channels, env.n_channels,
                                  *env.buffers, sp, c.prop, c.nucleus.density,
                                  coulomb_barrier_mev(c.ta, c.tz), c.rng, find, ref);
  if (ref.any()) { flat_hand_off(c, ref.capacity ? kHandCapacity : kHandRefused); }
  return ok;
}

/// POST: the rest of the iteration, by mode.
template <int N, int P, int C>
__host__ __device__ inline void stage_post(FlatCascade<N, P, C>& c, const FlatEnv& env,
                                           FlatScratch<N, P>& scr) {
  if (!c.stepping) { return; }
  c.stepping = false;
  ++c.steps;
  const TimeStepReport tsr = flat_after_transport(c, scr);
  if (c.stage == kStThread) { return; }
  BicCascadeState& st = c.st;
  if (c.mode == kModeMain) {
    int next = c.next;
    if (!tsr.success) {
      if (c.colls.next_collision() != next) { next = -1; }
    }
    if (next >= 0) {
      if (flat_apply(c, next, env, scr)) {
        c.have_products = true;
      } else if (c.stage != kStThread) {
        c.colls.remove(next);
      }
    }
  } else if (c.mode == kModeLast) {
    c.stage = kStFinishB;
    return;
  } else {
    // `StepParticlesOut`'s body after its `DoTimeStep`.
    if (c.next == -2) {
      ++c.out_steps;
    } else {
      int next = c.next;
      if (!tsr.success) {
        if (c.colls.next_collision() != next) { next = -1; }
      }
      if (next >= 0) {
        if (!flat_apply(c, next, env, scr) && c.stage != kStThread) { c.colls.remove(next); }
      }
    }
    if (c.stage == kStThread) { return; }
    if (c.out_resets > 100) {
      for (int i = 0; i < st.lists.n_pool; ++i) {
        if (st.lists.pool[i].list == kListSecondary) { push_final(st, i); }
      }
      c.mode = kModeLast;
    } else {
      AbsorbRefusal abs_ref;
      int* secondaries = scr.stepout_secondaries;
      int n_sec = 0;
      for (int i = 0; i < st.lists.n_pool && n_sec < kCascadeMaxScratch; ++i) {
        if (st.lists.pool[i].list == kListSecondary) { secondaries[n_sec++] = i; }
      }
      absorb(st.lists.pool, st.lists.n_pool, cut_on_p_absorb(), abs_ref);
      const CaptureDecision cap = capture_decision(st.lists.pool, st.lists.n_pool, st.cut_on_p,
                                                   c.prop);
      if (cap.capture) {
        int* captured = scr.stepout_captured;
        int n_cap = 0;
        for (int i = 0; i < st.lists.n_pool && n_cap < kCascadeMaxScratch; ++i) {
          CascadeTrack& t = st.lists.pool[i];
          if (t.list != kListSecondary || t.state != kInside) { continue; }
          if (t.pdg != imr::kPdgProton && t.pdg != imr::kPdgNeutron) { continue; }
          push_captured(st, i);
          mark_hit(st, i);
          captured[n_cap++] = i;
        }
        c.colls.remove_tracks(captured, n_cap);
      }
      if (c.out_steps > 100 && c.colls.size() == 0) {
        n_sec = 0;
        for (int i = 0; i < st.lists.n_pool && n_sec < kCascadeMaxScratch; ++i) {
          if (st.lists.pool[i].list == kListSecondary) { secondaries[n_sec++] = i; }
        }
        const CascadeSpecies sp = flat_species();
        CascadeRefusal ref;
        flat_find_collisions(st, c.colls, secondaries, n_sec, env, c.prop, c.rng, scr, ref);
        if (ref.any()) { return flat_hand_off(c, ref.capacity ? kHandCapacity : kHandRefused); }
        c.out_steps = 0;
        ++c.out_resets;
      }
    }
  }
  if (c.steps > env.k_steps) { flat_straggle(c, env); }
}

// =============================================================================================
// kStFinishA and kStFinishB: `Propagate` between and after the loops.
// =============================================================================================
template <int N, int P, int C>
__host__ __device__ inline void stage_finish_a(FlatCascade<N, P, C>& c, const FlatEnv& env,
                                               FlatScratch<N, P>& scr) {
  BicCascadeState& st = c.st;
  int n_target = 0;
  int n_protons = 0;
  for (int i = 0; i < st.lists.n_pool; ++i) {
    if (st.lists.pool[i].list != kListTarget) { continue; }
    ++n_target;
    if (st.lists.pool[i].pdg == imr::kPdgProton) { ++n_protons; }
  }
  if (n_target == 0 || n_protons == 0) {
    c.stage = kStTail;
    c.tail_work = kTailVoid;
    return;
  }
  AbsorbRefusal abs_ref;
  if (absorb(st.lists.pool, st.lists.n_pool, cut_on_p_absorb(), abs_ref)) {
    c.have_products = true;
  }
  if (flat_capture(c, env, scr)) { c.have_products = true; }
  if (c.stage == kStThread) { return; }
  if (!c.have_products) {
    c.outcome = kPropagateNoProducts;
    c.n_products = 0;
    return flat_outer_test(c);
  }
  c.mode = kModeOut;
  c.out_steps = 0;
  c.out_resets = 0;
  c.stepping = false;
  c.stage = kStStepOut;
}

template <int N, int P, int C>
__host__ __device__ inline void stage_finish_b(FlatCascade<N, P, C>& c) {
  BicCascadeState& st = c.st;
  const CascadeSpecies sp = flat_species();
  for (int i = 0; i < st.lists.n_pool; ++i) {
    if (st.lists.pool[i].list == kListSecondary) { push_final(st, i); }
  }
  while (c.colls.size() > 0) {
    const int next = c.colls.next_collision();
    if (next < 0) { break; }
    c.colls.remove(next);
  }
  double excitation = get_excitation_energy(st, sp.neutron_mass);
  if (excitation < 0.0) {
    int ntry = 0;
    do {
      correct_final_p_and_e(st, sp.neutron_mass);
      excitation = get_excitation_energy(st, sp.neutron_mass);
    } while (++ntry < 5 && excitation < 0.0);
  }
  c.excitation = excitation;
  if (excitation < 0.0) {
    c.outcome = kPropagateNegativeExcitation;
    c.n_products = 0;
    return flat_outer_test(c);
  }
  c.frag = find_fragments(st);
  c.stage = kStTail;
  c.tail_work = kTailDeexcite;
}

// =============================================================================================
// P39: a straggler finished on its own lane (docs/RISK.md V321). Every stage the drain would
// have given it, in the drain's order - the set-up (`setup`, the model's own: its nucleus from
// the factory if one waits in the record, else `nucleus_init` from the cascade's generator), the
// step's PRE, its move items one after another, POST, the finish - until the cascade needs a
// tail slot or has left. Nothing here reads another record, so a solo cascade is the one the
// drain would have computed, bit for bit; only its time is a single thread's. `env.k_steps` is
// the caller's to lift (the bound has done its work: it found the straggler).
// =============================================================================================
template <int N, int P, int C, typename Setup>
__host__ __device__ inline void flat_run_solo(FlatCascade<N, P, C>& c, const FlatEnv& env,
                                              FlatScratch<N, P>& scr, Setup&& setup) {
  // Geant4's own bounds end every loop below (`StepParticlesOut`'s 100 resets of 100 steps, the
  // collision loop's 1,000,000, the outer loop's 100 turns); this guard only keeps a defect
  // from hanging the card, and hands the cascade back by name if it is ever reached.
  for (int guard = 0; guard < (1 << 22); ++guard) {
    // P42: a straggler that would outgrow its record stops at that boundary and is continued
    // on a larger one (`FlatEnv::grow`), unflagged as a straggler: the larger record's runner
    // finishes it on one thread as this loop would have.
    if (c.grow) {
      c.solo = 0;
      return;
    }
    switch (c.stage) {
      case kStModel:
      case kStNucleus:
        setup();
        break;
      case kStStep:
      case kStStepOut:
        stage_pre(c, env, scr);
        if (c.stage != kStStep && c.stage != kStStepOut) { break; }
        if (c.stepping) {
          for (int i = 0; i < c.st.lists.n_pool; ++i) {
            if (c.prestate[i] >= 0) { item_transport(c, i); }
          }
        }
        stage_post(c, env, scr);
        break;
      case kStFinishA:
        stage_finish_a(c, env, scr);
        break;
      case kStFinishB:
        stage_finish_b(c);
        break;
      default:
        c.solo = 0;
        return;
    }
  }
  c.solo = 0;
  flat_hand_off(c, kHandStraggler);
}

// =============================================================================================
// kStTail: what `Propagate` does after `FindFragments`, and the packing `ApplyYourself` ends in.
// Returns true when the model call has its final state in `result`.
//
// P41 (docs/RISK.md V326): the buffers are the caller's, so that one body serves the two places
// a tail runs - a thread-path slot (`ws` the slot's `CascadeWorkspace`, `decay_list` its
// `BicScratch::decay_list` at `kCascadeMaxScratch`), and an entry of the wide tail pool
// (deexcitation/flat/tail_pool.cuh: compact product lists and decay list, no `BicScratch`).
// `ws.scratch` is read only by `DecayVoidNucleus` (an all-neutron remnant); a tail without one
// hands the cascade back, and the drain never sends that case to a pool entry (`tail_needs_slot`).
// The lines are the thread path's in the thread path's order: what changed is where the decay
// list lives. Its capacity, and the product lists', are the slot's own, so no refusal arises
// that the slot tail would not make. The ONE compact buffer is the de-excitation workspace
// `pws`, and when it binds the call is CONTINUED, never rerun from the key: the generator is put
// back where it was before the call - nothing else the call wrote is read again - and the same
// call runs on a slot-sized workspace borrowed from `fb`. P39 measured why (the lead, 2026-11-20):
// a rerun from the key is a different cascade, and the subset rerun is selected by the failure,
// which biases the sample. `fb` null is the slot tail: the thread path's behaviour, unchanged.
// =============================================================================================
/// P41: did PreCompound or the handler run out of a buffer - the only refusals a compact
/// de-excitation workspace can add to the slot's (the rest are functions of the fragment).
__host__ __device__ inline bool tail_capacity_refused(const preco::PrecoStatus& s) {
  return s.ref.capacity || s.deex.refused_capacity;
}

/// P41: the slot-sized de-excitation workspaces (536,576 B each: the slot's 4,096 / 1,024 / 512 /
/// 1,024 / 1,024) a pool lane borrows when its compact one binds. A flag a workspace; a lane that
/// finds none free retries at most `spin_bound` times and is then handed back by name
/// (`kHandTailScratch`) - a bound, so that no lane can wait forever (P39's 2 h 44 min hang).
struct TailFallback {
  const preco::PrecoWorkspace* big = nullptr;
  int* busy = nullptr;
  int n = 0;
  int spin_bound = 1 << 20;
  __host__ __device__ int acquire() const {
#ifdef __CUDA_ARCH__
    atomicAdd(&busy[n], 1);   // busy[n] counts the continuations, for the run's report
#else
    ++busy[n];
#endif
    for (int s = 0; s < spin_bound; ++s) {
      for (int k = 0; k < n; ++k) {
#ifdef __CUDA_ARCH__
        if (atomicCAS(&busy[k], 0, 1) == 0) { __threadfence(); return k; }
#else
        if (busy[k] == 0) { busy[k] = 1; return k; }
#endif
      }
    }
    return -1;
  }
  __host__ __device__ void release(int k) const {
#ifdef __CUDA_ARCH__
    __threadfence();
    atomicExch(&busy[k], 0);
#else
    busy[k] = 0;
#endif
  }
};

template <int N, int P, int C>
__host__ __device__ inline bool stage_tail(FlatCascade<N, P, C>& c, const CascadeWorkspace& ws,
                                           DecayTrack* decay_list, int decay_capacity,
                                           const data::LevelTable& lt,
                                           const deex::FermiPool& fpool,
                                           const preco::PrecoWorkspace& pws,
                                           const TailFallback* fb = nullptr, int grow = 0) {
  BicCascadeState& st = c.st;
  CascadeRefusal ref;
  // P42 (docs/RISK.md V331): the generator where the tail found it, so that a tail whose final
  // decays would outgrow the record's pool can be put back - nothing else it wrote is in the
  // record yet - and CONTINUED, the same call on a record with the thread path's capacities.
  const Philox<double> rng_tail = c.rng;
  if (c.tail_work == kTailVoid) {
    VoidReport vrep;
    const int n = fill_void_nucleus_products(st, c.colls, ws.products, ws.product_capacity,
                                             bic_model_id(), c.rng, ref, vrep);
    if (n < 0 || ref.any()) {
      flat_hand_off(c, ref.capacity ? kHandCapacity : kHandRefused);
      return false;
    }
    c.outcome = kPropagateVoidNucleus;
    c.n_products = n;
    flat_outer_test(c);
    if (c.stage != kStTail) { return false; }
  } else if (c.tail_work == kTailDeexcite) {
    preco::PrecoStatus status;
    int n_preco = 0;
    const CascadeFragment& frag = c.frag;
    if (frag.a > 1) {
      bool overflow = false;
      const Philox<double> rng_before = c.rng;
      n_preco = bic_deexcite_fragment(frag, ws.preco_products, ws.preco_capacity, lt, fpool,
                                      pws, c.rng, status, overflow);
      // P41: the compact workspace bound - the same call again, from the same generator
      // position, on a slot-sized workspace (the continuation; see above).
      if (fb != nullptr && tail_capacity_refused(status)) {
        const int k = fb->acquire();
        if (k < 0) {
          flat_hand_off(c, kHandTailScratch);
          return false;
        }
        c.rng = rng_before;
        overflow = false;
        status = preco::PrecoStatus{};
        n_preco = bic_deexcite_fragment(frag, ws.preco_products, ws.preco_capacity, lt, fpool,
                                        fb->big[k], c.rng, status, overflow);
        fb->release(k);
      }
      if (overflow) {
        flat_hand_off(c, kHandCapacity);
        return false;
      }
    } else if (frag.a == 1) {
      int found = -1;
      int n_left = 0;
      for (int i = 0; i < st.lists.n_pool; ++i) {
        const int l = st.lists.pool[i].list;
        if (l != kListTarget && l != kListCaptured) { continue; }
        ++n_left;
        found = i;
      }
      if (n_left != 1 || ws.preco_capacity < 1) {
        flat_hand_off(c, kHandRefused);
        return false;
      }
      const CascadeTrack& t = st.lists.pool[found];
      ws.preco_products[0] = CascadeProduct{};
      ws.preco_products[0].pdg = t.pdg;
      ws.preco_products[0].pdg_mass = t.pdg_mass;
      ws.preco_products[0].nucleus_z = t.charge;
      ws.preco_products[0].nucleus_a = t.baryon;
      ws.preco_products[0].momentum = imr::LorentzVector(deex::Vec3d{0.0, 0.0, 0.0}, t.pdg_mass);
      ws.preco_products[0].creator_model_id = bic_model_id();
      ws.preco_products[0].parent_resonance_pdg = t.parent_resonance_pdg;
      ws.preco_products[0].parent_resonance_id = t.parent_resonance_id;
      n_preco = 1;
    } else {
      if (ws.scratch == nullptr) {
        flat_hand_off(c, kHandCapacity);
        return false;
      }
      n_preco = decay_void_nucleus(st, ws.preco_products, ws.preco_capacity, bic_model_id(),
                                   *ws.scratch, c.rng, ref);
      if (ref.any()) {
        flat_hand_off(c, ref.capacity ? kHandCapacity : kHandRefused);
        return false;
      }
    }
    // `G4DecayKineticTracks decay(&theFinalState)`, as `propagate` ends.
    DecayTrack* dl = decay_list;
    int n = 0;
    for (int seq = 0; seq < st.n_final_pushed; ++seq) {
      int i = -1;
      for (int k = 0; k < st.lists.n_pool; ++k) {
        if (st.lists.pool[k].list == kListFinal && st.lists.pool[k].final_seq == seq) {
          i = k;
          break;
        }
      }
      if (i < 0) { continue; }
      const CascadeTrack& t = st.lists.pool[i];
      if (n >= decay_capacity) {
        flat_hand_off(c, kHandCapacity);
        return false;
      }
      dl[n] = DecayTrack{};
      dl[n].pdg = t.pdg;
      dl[n].momentum = t.momentum;
      dl[n].position = t.position;
      dl[n].formation_time = t.formation_time;
      dl[n].creator_model_id = t.creator_model_id;
      dl[n].parent_resonance_pdg = t.parent_resonance_pdg;
      dl[n].parent_resonance_id = t.parent_resonance_id;
      dl[n].caller_tag = i;
      ++n;
    }
    KineticDecayRefusal kref;
    int m = n;
    decay_kinetic_tracks(dl, m, decay_capacity, c.rng, kref);
    if (kref.any()) {
      flat_hand_off(c, kHandRefused);
      return false;
    }
    // P42: the `m` adds below fail exactly when `n_pool + m` passes the pool (`CascadeLists::
    // add` refuses only when full), so this is the refusal they would make, made before any of
    // them. Up to here the tail has written its own buffers and the generator only (the
    // void-nucleus arm, which writes the record, is excluded): the generator goes back.
    if (grow && frag.a >= 1 && st.lists.n_pool + m > st.lists.capacity) {
      c.rng = rng_tail;
      c.grow = 1;
      return false;
    }
    for (int i = 0; i < st.lists.n_pool; ++i) {
      if (st.lists.pool[i].list == kListFinal) { st.lists.pool[i].list = kListNone; }
    }
    st.n_final_pushed = 0;
    for (int k = 0; k < m; ++k) {
      CascadeTrack t;
      if (dl[k].caller_tag >= 0) {
        t = st.lists.pool[dl[k].caller_tag];
      } else {
        const int si = imr::decay_species_index(dl[k].pdg);
        if (si < 0) {
          flat_hand_off(c, kHandRefused);
          return false;
        }
        t.pdg = dl[k].pdg;
        t.pdg_mass = imr::decay_species_mass()[si];
        t.charge = imr::decay_species_charge()[si];
        t.baryon = imr::decay_species_baryon()[si];
        t.state = kGoneOut;
        t.nucleon_index = -1;
        t.hit = false;
      }
      t.momentum = dl[k].momentum;
      t.position = dl[k].position;
      t.formation_time = dl[k].formation_time;
      t.creator_model_id = dl[k].creator_model_id;
      t.parent_resonance_pdg = dl[k].parent_resonance_pdg;
      t.parent_resonance_id = dl[k].parent_resonance_id;
      const int added = st.lists.add(t, kListNone);
      if (added < 0) {
        flat_hand_off(c, kHandCapacity);
        return false;
      }
      push_final(st, added);
    }
    int n_products = products_add_final_state(st, ws.products, 0, ws.product_capacity,
                                              bic_model_id(), ref);
    n_products = products_add_precompound(st, ws.preco_products, n_preco, ws.products,
                                          n_products, ws.product_capacity, ref);
    if (ref.any()) {
      flat_hand_off(c, ref.capacity ? kHandCapacity : kHandRefused);
      return false;
    }
    c.outcome = kPropagateOk;
    c.n_products = n_products;
    flat_outer_test(c);
    if (c.stage != kStTail) { return false; }
  }
  // kTailResult: the caller packs the final state (`flat_pack_nucleon` or the light-ion arm's).
  return true;
}

/// The two arms `apply_yourself` answers before its loops (`kTailHydrogen`, `kTailPrecompound`),
/// on the tail slot, into `result`: the A == 1 arm is `apply_yourself_h1` itself; the PreCompound
/// arm is `apply_yourself`'s six lines around `preco::deexcite`, the packing included. Returns
/// false when the record was handed back (a refusal the thread path books).
template <int N, int P, int C>
__host__ __device__ inline bool flat_tail_precompound(FlatCascade<N, P, C>& c,
                                                      const data::LevelTable& lt,
                                                      const deex::FermiPool& fpool,
                                                      const preco::PrecoWorkspace& pws,
                                                      BicFinalState& result,
                                                      const TailFallback* fb = nullptr);

template <int N, int P, int C>
__host__ __device__ inline bool flat_tail_direct(FlatCascade<N, P, C>& c, BicStorage& store,
                                                 const data::LevelTable& lt,
                                                 const deex::FermiPool& fpool,
                                                 const preco::PrecoWorkspace& pws,
                                                 BicFinalState& result) {
  result.clear();
  physics::hadronic::HadProjectile<double> projectile;
  projectile.pdg = c.pdg;
  projectile.kin_energy = c.kin;
  const physics::hadronic::HadNucleus target{c.tz, c.ta, 0};
  if (c.tail_work == kTailHydrogen) {
    BicRefusal ref;
    BicReport rep;
    apply_yourself_h1(projectile, target, store, c.rng, result, ref, rep);
    if (ref.any()) {
      flat_hand_off(c, kHandRefused);
      return false;
    }
    return true;
  }
  return flat_tail_precompound(c, lt, fpool, pws, result);
}

/// The PreCompound arm alone (`kTailPrecompound`): `apply_yourself`'s six lines around
/// `preco::deexcite`, the packing included, into `result`. P41: its own function because it needs
/// no `BicStorage` - only the de-excitation workspace - so the wide tail pool runs it on an entry
/// (docs/RISK.md V326); `flat_tail_direct` calls it for the slot's tail. `result` is cleared by
/// the caller.
template <int N, int P, int C>
__host__ __device__ inline bool flat_tail_precompound(FlatCascade<N, P, C>& c,
                                                      const data::LevelTable& lt,
                                                      const deex::FermiPool& fpool,
                                                      const preco::PrecoWorkspace& pws,
                                                      BicFinalState& result,
                                                      const TailFallback* fb) {
  preco::Excitons ex;
  bool preco_refused = false;
  const deex::Fragment frag = preco::apply_yourself_initial_fragment(
      c.pdg, c.kin, Vec3<double>{0.0, 0.0, 1.0}, c.tz, c.ta, ex, preco_refused);
  if (preco_refused) {
    flat_hand_off(c, kHandRefused);
    return false;
  }
  const Philox<double> rng_before = c.rng;
  preco::PrecoStatus status = preco::deexcite(frag, ex, lt, fpool, pws, c.rng);
  const preco::PrecoWorkspace* used = &pws;
  int borrowed = -1;
  if (fb != nullptr && tail_capacity_refused(status)) {
    // P41: continued from the same generator position on a slot-sized workspace.
    borrowed = fb->acquire();
    if (borrowed < 0) {
      flat_hand_off(c, kHandTailScratch);
      return false;
    }
    c.rng = rng_before;
    used = &fb->big[borrowed];
    status = preco::deexcite(frag, ex, lt, fpool, *used, c.rng);
  }
  result.status = physics::hadronic::HadFinalStateStatus::kStopAndKill;
  for (int i = 0; i < status.n_products; ++i) {
    const deex::DeexProduct& p = used->products[i];
    physics::hadronic::HadSecondary<double> s;
    s.pdg = p.pdg;
    s.z = p.z;
    s.a = p.a;
    s.time = 0.0;
    s.weight = 1.0;
    s.creator_model_id = -1;
    const double pdg_mass =
        (p.a > 0) ? (deex::nuclear_mass(p.a, p.z) + p.excitation)
                  : ((p.pdg == deex::kPdgElectron) ? u::electron_mass_c2<double>() : 0.0);
    capture::set_four_momentum(s, p.momentum, pdg_mass);
    if (!result.add_secondary(s)) {
      if (borrowed >= 0) { fb->release(borrowed); }
      flat_hand_off(c, kHandCapacity);
      return false;
    }
  }
  if (borrowed >= 0) { fb->release(borrowed); }
  return true;
}

/// `apply_yourself`'s packing, from the products `stage_tail` left in `ws.products`.
template <int N, int P, int C>
__host__ __device__ inline bool flat_pack_nucleon(FlatCascade<N, P, C>& c,
                                                  const CascadeWorkspace& ws,
                                                  BicFinalState& result) {

  result.clear();
  if (!c.have || c.n_products == 0) {
    result.status = physics::hadronic::HadFinalStateStatus::kIsAlive;
    result.energy_change = c.kin;
    result.momentum_change = Vec3<double>{0.0, 0.0, 1.0};
    return true;
  }
  result.status = physics::hadronic::HadFinalStateStatus::kStopAndKill;
  for (int i = 0; i < c.n_products; ++i) {
    const CascadeProduct& p = ws.products[i];
    physics::hadronic::HadSecondary<double> s;
    s.pdg = p.pdg;
    s.z = p.nucleus_z;
    s.a = p.nucleus_a;
    s.time = 0.0;
    s.weight = 1.0;
    s.creator_model_id = p.creator_model_id;
    capture::set_four_momentum(s, p.momentum, p.pdg_mass);
    if (!result.add_secondary(s)) {
      flat_hand_off(c, kHandCapacity);
      return false;
    }
  }
  return true;
}
}  // namespace g4gpu::bic::flat

#endif
