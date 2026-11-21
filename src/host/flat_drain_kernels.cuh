// P36: the flat drain's kernels and loop, for either model (docs/RISK.md V308-V310). Included
// by exactly two units - transport_run_int_binary_flat.cu and transport_run_int_lightion_flat.cu
// - each with its own `Policy` (the record type, the model's set-up and its packing). Everything
// between the set-up and the product list is the same `Propagate` for both and is shared here.
//
// ONE ROUND of the drain, every kernel over the pool's lanes (lane = cascade):
//
//     claim   free lanes take queue entries: `run_inelastic`'s integral rejection and
//             `SampleZandA`, from the entry's own generator
//     setup   the model above its loops, then an outer-loop turn (the nucleus or nuclei, the
//             fields, the impact points, `Propagate`'s prologue)
//     pre     the top of a collision-loop or step-out iteration: the time step, and one move
//             item per secondary the step moves
//     move    one item per (cascade, secondary): `G4RKPropagation::Transport` of that track
//     post    the rest of `DoTimeStep`, the collision and `FindCollisions` for what it made
//     finish  `Propagate` between and after its two loops
//     tail    on a tail slot (the thread path's own slots, idle while this drain runs): the
//             de-excitation, the final decays, the products, the model's packing,
//             `CheckResult`, the K0 mixing and the emission
//     reap    hand-backs to the thread path's list, finished lanes freed, the host's counts
//
// A record's stage changes only inside its own lane's code, and nothing a lane computes reads
// another lane: which lane, which round or which tail slot an entry got moves time, not numbers.
#pragma once

#include "host/flat_drain.cuh"
#include "host/transport_run_int_impl.cuh"
#include "physics/hadronic/bic/flat/ion_flat.cuh"
#include "physics/hadronic/npool/nucleus_factory_launch.cuh"

namespace g4gpu::host::flatk {

namespace bf = g4gpu::bic::flat;
namespace fhp = g4gpu::physics::hadronic;
namespace fhxs = g4gpu::hadronic::xs;

/// The drain's device counters.
enum FlatCounter : int {
  kFcNext = 0,      ///< the next entry to claim
  kFcItems,         ///< move items this round
  kFcTailUsed,      ///< tail slots taken this round
  kFcActive,        ///< lanes with an entry, after reap
  kFcStrag,         ///< entries handed back
  kFcFinished,
  kFcTailWait,
  kFcSetupN,        ///< lanes in `kStModel` / `kStNucleus`, listed in `setup_list`
  kFcStepN,         ///< lanes in a step or finish stage
  kFcTailN,         ///< lanes in `kStTail`, listed in `tail_list`
  // P39's set-up counters (summed over the drain; printed with `G4GPU_FLAT_PROFILE=1`):
  kFcSetupNuc,      ///< set-ups that reached the nucleus (an outer-loop turn, a light-ion call)
  kFcSetupRetry,    ///< ... of a record that had had a nucleus set-up before
  kFcNuclei,        ///< nuclei built, on the lane or by the factory
  kFcFactoryLane,   ///< factory requests that fell back to the lane (V321: none expected)
  kFcSoloN,         ///< stragglers waiting for `flat_solo`, listed in `solo_list`
  kFcSolo,          ///< stragglers finished on their own lane, over the drain
  kFcHand0,         ///< + `bf::HandOff`
  kFcNum = kFcHand0 + 8
};

/// Per-entry transport state the model record does not carry.
struct FlatEntry {
  int qi = -1;            ///< the queue index, `idx[base + e]`
  int attempts = 0;       ///< `run_inelastic`'s re-entries
  int rejected = 0;       ///< the integral rejection said no: no model call
  int target_z = 0, target_a = 0;
};

template <class Pol>
struct FlatArgs {
  typename Pol::Rec* rec;
  FlatEntry* ent;
  bic::CascadeBuffers* buffers;
  int* ctr;
  int* items;
  int* strag;
  int* setup_list;
  int* tail_list;
  int* solo_list;         ///< P39: stragglers waiting for `flat_solo`
  int solo;               ///< P39: stragglers run solo (the factory is on), not handed back
  int lanes;
  int item_capacity;
  int k_steps;
  int n;
  int base;
  const int* idx;
  const had::PendingInteraction<double>* q;
};

template <class Pol>
__device__ __forceinline__ bf::FlatEnv flat_env(const FlatArgs<Pol>& a,
                                                const had::InteractionPool<double>& pool) {
  bf::FlatEnv e;
  e.channels = pool.bic_channels;
  e.n_channels = pool.n_bic_channels;
  e.buffers = a.buffers;
  e.k_steps = a.k_steps;
  e.solo = a.solo;
  return e;
}

/// The projectile `run_interaction_entry` builds from a queue entry, for the model and for
/// `CheckResult`.
__device__ __forceinline__ fhp::HadProjectile<double> flat_projectile(
    const had::PendingInteraction<double>& q) {
  const TrackState<double>& p = q.track;
  const int pz = (q.species == ParticleType::kGenericIon) ? ion_z_of(p.ion_za) : 0;
  const int pa = (q.species == ParticleType::kGenericIon) ? ion_a_of(p.ion_za) : 0;
  fhp::HadProjectile<double> proj;
  proj.pdg = (q.species == ParticleType::kGenericIon) ? fhp::pdg_nuclear_code(pz, pa)
                                                      : pdg_code(q.species);
  proj.charge = (q.species == ParticleType::kGenericIon) ? static_cast<double>(pz)
                                                         : particle_def<double>(q.species).charge;
  proj.mass = (q.species == ParticleType::kGenericIon)
                  ? em::ion_particle_def<double>(pz, pa).mass
                  : particle_def<double>(q.species).mass;
  proj.kin_energy = p.ekin;
  proj.baryon_number = had::baryon_number_of(q.species, ion_a_of(p.ion_za));
  return proj;
}

// ---- claim: a free lane takes the next entry and runs `run_inelastic`'s head on it.
template <class Pol>
__global__ void flat_claim(FlatArgs<Pol> a, Scene<double> scene,
                           had::HadronicWiring<double> had) {
  const int lane = blockIdx.x * blockDim.x + threadIdx.x;
  if (lane >= a.lanes) { return; }
  typename Pol::Rec& r = a.rec[lane];
  bf::FlatCascade<Pol::kN, Pol::kP, Pol::kC>& c = Pol::core(r);
  if (c.stage != bf::kStFree) { return; }
  const int e = atomicAdd(&a.ctr[kFcNext], 1);
  if (e >= a.n) { return; }
  FlatEntry& en = a.ent[lane];
  en.qi = a.idx[a.base + e];
  en.attempts = 0;
  en.rejected = 0;
  const had::PendingInteraction<double>& q = a.q[en.qi];
  const TrackState<double>& p = q.track;
  // The generator `run_interaction_entry` makes, from the same key.
  Philox<double> rng(p.rng_key, p.step, had::kInteractionRngPurpose);
  const fhp::HadProjectile<double> proj = flat_projectile(q);
  const int pz = (q.species == ParticleType::kGenericIon) ? ion_z_of(p.ion_za) : 0;
  const int pa = (q.species == ParticleType::kGenericIon) ? ion_a_of(p.ion_za) : 0;
  // `run_inelastic`'s steps 2 and 3, line for line (interaction_apply.cuh).
  had::InelasticTables<double> xs = had.inelastic;
  xs.neutron = had.neutron.inelastic;
  const data::Material<double>& mat = scene.materials[q.material];
  fhxs::MaterialXs<double> mxs{};
  const fhxs::Projectile<double> pj = had::inelastic_projectile<double>(q.species, pz, pa);
  const fhp::HadXsType xt = had::inelastic_xs_type<double>(pj);
  double xs_now = q.xs_at_step_start;
  if (xt != fhp::HadXsType::kNoIntegral) {
    xs_now = had::inelastic_xs_per_volume<double>(xs, mat, q.species, proj.kin_energy, pz, pa,
                                                  mxs);
    if (fhp::integral_xs_rejects<double>(xt, xs_now, q.xs_at_step_start, rng)) {
      en.rejected = 1;
    }
  } else {
    xs_now = had::inelastic_xs_per_volume<double>(xs, mat, q.species, proj.kin_energy, pz, pa,
                                                  mxs);
  }
  Pol::start(r, e, proj, rng);
  // P39: the key every nucleus of this cascade is built from - its generator's own.
  bf::flat_set_key(c, p.rng_key, static_cast<unsigned>(p.step));
  if (en.rejected) {
    c.stage = bf::kStTail;
    return;
  }
  if (!(xs_now > 0.0) || mxs.n_elements <= 0) {
    return bf::flat_hand_off(c, bf::kHandRefused);  // `kNoInelasticModel`: the thread path's
  }
  const fhxs::TargetZA tgt = had::inelastic_sample_target<double>(xs, mat, q.species,
                                                                  proj.kin_energy, pz, pa, mxs,
                                                                  c.rng);
  en.target_z = tgt.z;
  en.target_a = tgt.a;
  c.tz = tgt.z;
  c.ta = tgt.a;
}

// ---- setup
template <class Pol>
__global__ void flat_setup(FlatArgs<Pol> a, const had::InteractionPool<double>* pool_ptr,
                           int n_list) {
  const int k_ = blockIdx.x * blockDim.x + threadIdx.x;
  if (k_ >= n_list) { return; }
  const int lane = a.setup_list[k_];
  typename Pol::Rec& r = a.rec[lane];
  bf::FlatCascade<Pol::kN, Pol::kP, Pol::kC>& c = Pol::core(r);
  if (c.stage == bf::kStModel) { Pol::model(r); }
  if (c.stage != bf::kStNucleus) { return; }
  // P39's counts: a set-up that reaches the nucleus, whether the record has had one before (a
  // new outer-loop turn, a `CheckResult` re-entry, a light-ion try after a miss), and the nuclei
  // it builds - one a turn, two a light-ion try (`c.steps` counts both, V307).
  const int steps_before = c.steps;
  atomicAdd(&a.ctr[kFcSetupNuc], 1);
  if (steps_before > 0) { atomicAdd(&a.ctr[kFcSetupRetry], 1); }
  // The nucleus's set-up scratch, on this thread's frame: `kN` nucleons' worth - the fall-back's
  // (`G4GPU_FLAT_FACTORY=0`, or a request the factory could not take).
  deex::Vec3d mom[Pol::kN];
  double fermi_p[Pol::kN];
  bic::NucleusSortEntry sums[Pol::kN];
  double flat_block[bic::kFlatBlock];
  bic::Nucleus3DScratch nsc;
  nsc.momentum = mom;
  nsc.fermi_p = fermi_p;
  nsc.test_sums = sums;
  nsc.flat_block = flat_block;
  nsc.capacity = Pol::kN;
  bf::FlatScratch<Pol::kN, Pol::kP> scr;
  Pol::nucleus(r, flat_env(a, *pool_ptr), nsc, scr);
  atomicAdd(&a.ctr[kFcNuclei], (c.steps - steps_before) * Pol::kNucleiPerTry);
}

// ---- P39: the set-up's requests to the nucleus factory (docs/RISK.md V321). Over the same
// `setup_list` as `flat_setup`, before it: the model above its loops (`apply_yourself`'s top,
// `blir_apply_yourself` above `Interact` - which decide the nuclide), then one request a target
// (two a light-ion try: projectile and target), keyed on the cascade and its nucleus count.
// The factory builds them into the records; `flat_setup` then reads them instead of sampling.
template <class Pol>
__global__ void flat_request(FlatArgs<Pol> a, int n_list, npool::FactoryLists lists) {
  const int k_ = blockIdx.x * blockDim.x + threadIdx.x;
  if (k_ >= n_list) { return; }
  const int lane = a.setup_list[k_];
  typename Pol::Rec& r = a.rec[lane];
  bf::FlatCascade<Pol::kN, Pol::kP, Pol::kC>& c = Pol::core(r);
  if (c.stage == bf::kStModel) { Pol::model(r); }
  if (c.stage != bf::kStNucleus) { return; }
  Pol::request(r, lists);
  if (!c.nuc_ready) { atomicAdd(&a.ctr[kFcFactoryLane], 1); }
}

// ---- P39: a straggler finished on its own lane (cascade_flat.cuh's `flat_run_solo`), the
// bound lifted - one thread a straggler, launched when the drain has nothing else.
template <class Pol>
__global__ void flat_solo(FlatArgs<Pol> a, const had::InteractionPool<double>* pool_ptr,
                          int n_list) {
  const int k_ = blockIdx.x * blockDim.x + threadIdx.x;
  if (k_ >= n_list) { return; }
  const int lane = a.solo_list[k_];
  typename Pol::Rec& r = a.rec[lane];
  bf::FlatCascade<Pol::kN, Pol::kP, Pol::kC>& c = Pol::core(r);
  bf::FlatEnv env = flat_env(a, *pool_ptr);
  env.k_steps = 1 << 30;
  deex::Vec3d mom[Pol::kN];
  double fermi_p[Pol::kN];
  bic::NucleusSortEntry sums[Pol::kN];
  double flat_block[bic::kFlatBlock];
  bic::Nucleus3DScratch nsc;
  nsc.momentum = mom;
  nsc.fermi_p = fermi_p;
  nsc.test_sums = sums;
  nsc.flat_block = flat_block;
  nsc.capacity = Pol::kN;
  bf::FlatScratch<Pol::kN, Pol::kP> scr;
  atomicAdd(&a.ctr[kFcSolo], 1);
  bf::flat_run_solo(c, env, scr, [&] {
    if (c.stage == bf::kStModel) { Pol::model(r); }
    if (c.stage == bf::kStNucleus) { Pol::nucleus(r, env, nsc, scr); }
  });
}

// ---- pre
template <class Pol>
__global__ void flat_pre(FlatArgs<Pol> a, const had::InteractionPool<double>* pool_ptr) {
  const int lane = blockIdx.x * blockDim.x + threadIdx.x;
  if (lane >= a.lanes) { return; }
  bf::FlatCascade<Pol::kN, Pol::kP, Pol::kC>& c = Pol::core(a.rec[lane]);
  if (c.stage != bf::kStStep && c.stage != bf::kStStepOut) { return; }
  if (c.solo) { return; }  // P39: a straggler waits for `flat_solo`
  bf::FlatScratch<Pol::kN, Pol::kP> scr;
  bf::stage_pre(c, flat_env(a, *pool_ptr), scr);
  if (!c.stepping) { return; }
  int n_sec = 0;
  for (int i = 0; i < c.st.lists.n_pool; ++i) { n_sec += (c.prestate[i] >= 0) ? 1 : 0; }
  if (n_sec == 0) { return; }
  // The items are every secondary of every lane at most (`item_capacity`), so a round never
  // runs short: a hand-off for want of an item would depend on the other lanes.
  int k = atomicAdd(&a.ctr[kFcItems], n_sec);
  for (int i = 0; i < c.st.lists.n_pool; ++i) {
    if (c.prestate[i] >= 0) { a.items[k++] = (lane << 9) | i; }
  }
}

// ---- move: one item per (cascade, secondary)
template <class Pol>
__global__ void flat_move(FlatArgs<Pol> a) {
  const int n = a.ctr[kFcItems];
  const int stride = gridDim.x * blockDim.x;
  for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < n && k < a.item_capacity;
       k += stride) {
    const int it = a.items[k];
    bf::item_transport(Pol::core(a.rec[it >> 9]), it & 511);
  }
}

// ---- post
template <class Pol>
__global__ void flat_post(FlatArgs<Pol> a, const had::InteractionPool<double>* pool_ptr) {
  const int lane = blockIdx.x * blockDim.x + threadIdx.x;
  if (lane >= a.lanes) { return; }
  bf::FlatCascade<Pol::kN, Pol::kP, Pol::kC>& c = Pol::core(a.rec[lane]);
  if (c.stage != bf::kStStep && c.stage != bf::kStStepOut) { return; }
  if (c.solo) { return; }  // P39: a straggler waits for `flat_solo`
  bf::FlatScratch<Pol::kN, Pol::kP> scr;
  bf::stage_post(c, flat_env(a, *pool_ptr), scr);
}

// ---- finish
template <class Pol>
__global__ void flat_finish(FlatArgs<Pol> a, const had::InteractionPool<double>* pool_ptr) {
  const int lane = blockIdx.x * blockDim.x + threadIdx.x;
  if (lane >= a.lanes) { return; }
  bf::FlatCascade<Pol::kN, Pol::kP, Pol::kC>& c = Pol::core(a.rec[lane]);
  if (c.solo) { return; }  // P39: a straggler waits for `flat_solo`
  if (c.stage == bf::kStFinishA) {
    bf::FlatScratch<Pol::kN, Pol::kP> scr;
    bf::stage_finish_a(c, flat_env(a, *pool_ptr), scr);
  } else if (c.stage == bf::kStFinishB) {
    bf::stage_finish_b(c);
  }
}

// ---- tail: on a tail slot, the end of the model call and `run_interaction_entry`'s tail.
//
// From the model's packing on, this is interaction_apply.cuh's and transport_run_int_impl.cuh's
// code with the slot the tail claimed: `copy_final_state`, `run_inelastic`'s `CheckResult`
// re-entry loop (a rejection sends the record back to `kStModel` with its generator where it
// is, which is the loop's next `ApplyYourself`), the K0 mixing, `fill_result_into` and the
// emission. Written out here rather than shared, so that transport_run_int_impl.cuh - and with
// it the seven interaction units - stays as it is (docs/RISK.md V308 names the duplication).
template <class Pol>
__global__ void flat_tail(FlatArgs<Pol> a, Scene<double> scene, TrackBuffer<double> out,
                          int batch, double* score, double* voxel_score,
                          had::HadronicWiring<double> had,
                          const had::InteractionPool<double>* pool_ptr, int n_tail_slots,
                          vis::TrajectoryBuffer traj, int* status_warn, SecondaryArena sec,
                          EmitterBooks books, StepTap<double> hook, int n_list) {
  using real_t = double;
  const int k_ = blockIdx.x * blockDim.x + threadIdx.x;
  if (k_ >= n_list || k_ >= n_tail_slots) { return; }
  const int lane = a.tail_list[k_];
  typename Pol::Rec& r = a.rec[lane];
  bf::FlatCascade<Pol::kN, Pol::kP, Pol::kC>& c = Pol::core(r);
  if (c.stage != bf::kStTail) { return; }
  const int slot_index = k_;
  const had::InteractionPool<real_t>& pool = *pool_ptr;
  had::InteractionSlot<real_t>* slot = pool.slot(slot_index);
  FlatEntry& en = a.ent[lane];
  const had::PendingInteraction<real_t>& q = a.q[en.qi];

  had::InteractionOutcome outc;
  if (en.rejected) {
    outc.rejected_by_integral_xs = true;
  } else {
    const preco::PrecoWorkspace pws = had::preco_workspace_of<real_t>(*slot);
    bic::BicStorage store = had::bic_storage_of<real_t>(*slot, pool, pws);
    if (c.tail_work == bf::kTailHydrogen || c.tail_work == bf::kTailPrecompound) {
      if (!bf::flat_tail_direct(c, store, had.level_data, pool.fermi, pws, slot->bic_fs)) {
        return;
      }
      had::copy_final_state<real_t>(slot->bic_fs, slot->fs);
    } else if (!bf::stage_tail(c, store, had.level_data, pool.fermi, pws) ||
               !Pol::pack(r, *slot, pool, had.level_data, pws)) {
      return;  // back to the flat stages, or handed to the thread path
    }
    // `run_inelastic`'s step 6, from `slot->fs` (the model's `copy_final_state` is `pack`'s).
    for (int i = 0; i < slot->fs.n_secondaries; ++i) {
      slot->pdg_mass[i] = had::definition_mass_of<real_t>(slot->fs.secondaries[i]);
    }
    const fhp::HadProjectile<real_t> proj = flat_projectile(q);
    const fhp::FatalEnergyCheckLevels<real_t> levels{};
    const real_t target_mass = static_cast<real_t>(deex::nuclear_mass(en.target_a, en.target_z));
    real_t delta_e = real_t(0);
    const fhp::CheckResultVerdict v = fhp::check_result<real_t, had::kInteractionSecondaryCap>(
        proj, target_mass, slot->fs, levels, slot->pdg_mass, &delta_e);
    if (v != fhp::CheckResultVerdict::kAccept) {
      if (++en.attempts >= 100) { return bf::flat_hand_off(c, bf::kHandReentry); }
      c.stage = bf::kStModel;
      return;
    }
    for (int i = 0; i < slot->fs.n_secondaries; ++i) {
      const int pc = slot->fs.secondaries[i].pdg;
      if (pc == 311 || pc == -311) {
        slot->fs.secondaries[i].pdg = (c.rng.uniform() > real_t(0.5)) ? 310 : 130;
      }
    }
    outc.model = Pol::kModel;
    outc.target_z = en.target_z;
    outc.target_a = en.target_a;
    outc.attempts = en.attempts;
    outc.n_secondaries = slot->fs.n_secondaries;
    outc.ran = true;
  }

  // ---- `run_interaction_entry` around the model call (transport_run_int_impl.cuh).
  TrackState<real_t> p = q.track;
  int vcell = -1;
  if (q.score_slot >= 0 && q.volume_pre >= 0 && voxel_score != nullptr) {
    const auto& vv = scene.geometry.volumes[q.volume_pre];
    if (vv.score_per_voxel) {
      const auto grid = geom::voxel_grid_of(vv.solid);
      const auto qp = geom::to_local(vv.xform, q.pos_pre);
      int ijk[3];
      geom::voxel_cell_of(grid, qp, ijk);
      vcell = grid.index(ijk[0], ijk[1], ijk[2]);
    }
  }
  StepReport<real_t> srep{};
  srep.true_length = q.true_length;
  srep.safety = q.safety;
  srep.non_ionizing = q.non_ionizing;
  srep.material = q.material;
  srep.status = q.status;
  srep.process = ProcessId::fHadronInelastic;
  const real_t track_mass =
      (q.species == ParticleType::kGenericIon)
          ? em::ion_particle_def<real_t>(ion_z_of(p.ion_za), ion_a_of(p.ion_za)).mass
          : particle_def<real_t>(q.species).mass;
  BufferEmitter<real_t> em{out, p.pos, p.volume, p.event, p.rng_key, p.step, q.child_count,
                           q.track.global_time, p.weight,
                           TrackState<real_t>::pre_step_velocity(q.ekin_pre, track_mass),
                           sec, q.sec_last, &srep, books};
  real_t edep = q.edep;
  bool alive = false;
  if (outc.rejected_by_integral_xs) {
    alive = (p.ekin > em::kHadronTrackingCut<real_t>() && p.volume != geom::kOutsideWorld);
  } else if (outc.ran) {
    had::fill_result_into<real_t, had::kInteractionSecondaryCap, had::kInteractionSecondaryCap>(
        slot->fs, p.dir, p.global_time, p.weight,
        /*has_at_rest_processes=*/had::has_at_rest_arm(q.species), slot->pdg_mass,
        slot->filled);
    edep += slot->filled.local_energy_deposit;
    srep.non_ionizing += slot->filled.non_ionizing_energy_deposit;
    em.pos = p.pos;
    em.volume = p.volume;
    em.event = p.event;
    had::emit_interaction_result<real_t>(slot->filled, em, had.books);
    if (pool.ftfp_tape_max > 0) {
      had::norow_tape_rows<real_t>(pdg_code(q.species), outc.ftf_ran, slot->filled);
    }
    if (slot->filled.status == fhp::TrackStatusChange::kAlive) {
      p.ekin = slot->filled.energy;
      p.dir = slot->filled.momentum_direction;
      alive = (p.ekin > em::kHadronTrackingCut<real_t>() && p.volume != geom::kOutsideWorld);
    } else {
      p.ekin = real_t(0);
    }
  }
  ++p.step;
  p.advance(srep.true_length, q.ekin_pre, track_mass);
  p.flags &= ~kFirstStepInVolume;
  p.ngp_left = real_t(-1);
  p.ngp_prev_step = srep.true_length;
  if (edep != real_t(0) && q.score_slot >= 0) {
    atomicAdd(&score[static_cast<size_t>(q.score_slot) * batch + p.event],
              static_cast<double>(edep));
    if (vcell >= 0 && voxel_score != nullptr) {
      atomicAdd(&voxel_score[vcell], static_cast<double>(edep));
    }
  }
  (void)traj;
  DeviceStep<real_t> ds{};
  ds.species = q.species;
  p.species = q.species;
  ds.ekin_pre = q.ekin_pre;
  ds.ekin_post = alive ? p.ekin : real_t(0);
  ds.edep = edep;
  ds.length = srep.true_length;
  ds.pos_pre = q.pos_pre;
  ds.pos_post = p.pos;
  ds.dir_pre = q.dir_pre;
  ds.dir_post = p.dir;
  ds.volume_pre = q.volume_pre;
  ds.volume_post = p.volume;
  ds.score_slot = q.score_slot;
  ds.event = p.event;
  ds.alive = alive;
  ds.track_ptr = &p;
  ds.first_in_volume = q.first_in_volume;
  ds.material = srep.material;
  ds.safety = srep.safety;
  ds.non_ionizing = srep.non_ionizing;
  ds.n_secondaries = static_cast<int>(em.child_count);
  ds.sec_arena = sec;
  ds.sec_pool = &out;
  ds.sec_last = em.last_secondary;
  ds.status = srep.status;
  ds.process = srep.process;
  hook(ds);
  const bool requeue = resolve_track_status<real_t>(alive, p.status);
  if (status_warn != nullptr && is_unsupported_track_status(p.status)) {
    atomicAdd(status_warn, 1);
  }
  if (requeue) { out.append(p); }
  // The tail runs beside the other stages on its own stream (`drain`): every write of this
  // lane's record and emission is made visible before the stage that frees the lane.
  __threadfence();
  c.stage = bf::kStDone;
}

// ---- reap: hand-offs back to the queue, finished lanes freed, the host's counts.
template <class Pol>
__global__ void flat_reap(FlatArgs<Pol> a) {
  const int lane = blockIdx.x * blockDim.x + threadIdx.x;
  if (lane >= a.lanes) { return; }
  bf::FlatCascade<Pol::kN, Pol::kP, Pol::kC>& c = Pol::core(a.rec[lane]);
  if (c.stage == bf::kStThread) {
    a.strag[atomicAdd(&a.ctr[kFcStrag], 1)] = a.ent[lane].qi;
    atomicAdd(&a.ctr[kFcHand0 + (c.hand_off & 7)], 1);
    c.stage = bf::kStFree;
  } else if (c.stage == bf::kStDone) {
    atomicAdd(&a.ctr[kFcFinished], 1);
    c.stage = bf::kStFree;
  }
  if (c.stage != bf::kStFree) { atomicAdd(&a.ctr[kFcActive], 1); }
  if (c.solo && c.stage != bf::kStFree && c.stage != bf::kStTail) {
    // P39: a straggler, frozen where it is until the drain has nothing else (`flat_solo`).
    a.solo_list[atomicAdd(&a.ctr[kFcSoloN], 1)] = lane;
  } else if (c.stage == bf::kStModel || c.stage == bf::kStNucleus) {
    a.setup_list[atomicAdd(&a.ctr[kFcSetupN], 1)] = lane;
  } else if (c.stage == bf::kStTail) {
    a.tail_list[atomicAdd(&a.ctr[kFcTailN], 1)] = lane;
  } else if (c.stage != bf::kStFree) {
    atomicAdd(&a.ctr[kFcStepN], 1);
  }
}

// ---- the prebuilt cross-section buffers: every pair `ensure_*_buffer` can be asked for, by
// the functions the thread path calls lazily. A template on the policy, so that the two units
// that include this header each have their own and the archive links.
template <class Pol>
__global__ void flat_build_buffers(bic::CascadeBuffers* buf,
                                   const had::InteractionPool<double>* pool_ptr) {
  if (blockIdx.x != 0 || threadIdx.x != 0) { return; }
  *buf = bic::CascadeBuffers{};
  const bic::CascadeSpecies sp = bf::flat_species();
  bic::CascadeRefusal ref;
  const int pions[3] = {211, -211, 111};
  const int nucl[2] = {2212, 2112};
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 2; ++j) {
      bic::ensure_meson_buffer(*buf, pions[i], nucl[j], sp.mass_of(pions[i]),
                               sp.mass_of(nucl[j]), bic::CascadeSpecies::iso3_of(pions[i]),
                               bic::CascadeSpecies::iso3_of(nucl[j]), sp.pi_plus_mass,
                               sp.proton_mass, ref);
    }
  }
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      bic::ensure_nn_buffer(*buf, pool_ptr->bic_channels, pool_ptr->n_bic_channels, nucl[i],
                            nucl[j], sp.mass_of(nucl[i]), sp.mass_of(nucl[j]), ref);
    }
  }
}

template <class Pol>
__global__ void flat_init_records(typename Pol::Rec* rec, int lanes) {
  const int lane = blockIdx.x * blockDim.x + threadIdx.x;
  if (lane >= lanes) { return; }
  Pol::core(rec[lane]).stage = bf::kStFree;
}

inline void pool_free(FlatPool& p) {
  cudaFree(p.records);
  cudaFree(p.buffers);
  cudaFree(p.counters);
  cudaFree(p.items);
  cudaFree(p.strag);
  cudaFree(p.factory_req);
  cudaFree(p.factory_count);
  p = FlatPool{};
}
template <class Pol>
std::size_t lane_bytes() {
  return sizeof(typename Pol::Rec) + sizeof(FlatEntry) +
         2 * Pol::kNucleiPerTry * sizeof(npool::FactoryRequest);
}

template <class Pol>
FlatPool pool_build(int lanes, int k_steps, int strag_capacity) {
  FlatPool p;
  if (lanes <= 0) { return p; }
  p.lanes = lanes;
  p.k_steps = k_steps;
  p.item_capacity = lanes * Pol::kP;
  p.record_bytes = sizeof(typename Pol::Rec);
  // P39: two request lists, each every lane's worst case (V321: a push never finds one full).
  p.factory_cap = lanes * Pol::kNucleiPerTry;
  const std::size_t req_bytes =
      2 * static_cast<std::size_t>(p.factory_cap) * sizeof(npool::FactoryRequest);
  const std::size_t rec =
      (sizeof(typename Pol::Rec) + sizeof(FlatEntry)) * static_cast<std::size_t>(lanes);
  if (cudaMalloc(&p.records, rec) != cudaSuccess ||
      cudaMalloc(&p.buffers, sizeof(bic::CascadeBuffers)) != cudaSuccess ||
      cudaMalloc(&p.counters, sizeof(int) * kFcNum) != cudaSuccess ||
      cudaMalloc(&p.items, sizeof(int) * (static_cast<std::size_t>(p.item_capacity) +
                                          3 * static_cast<std::size_t>(lanes))) != cudaSuccess ||
      cudaMalloc(&p.strag, sizeof(int) * static_cast<std::size_t>(strag_capacity)) !=
          cudaSuccess ||
      cudaMalloc(&p.factory_req, req_bytes) != cudaSuccess ||
      cudaMalloc(&p.factory_count, 2 * sizeof(int)) != cudaSuccess) {
    cudaGetLastError();
    pool_free(p);
    return FlatPool{};
  }
  p.bytes = rec + sizeof(bic::CascadeBuffers) + sizeof(int) * kFcNum +
            sizeof(int) * (static_cast<std::size_t>(p.item_capacity) + 3 * static_cast<std::size_t>(lanes)) +
            sizeof(int) * static_cast<std::size_t>(strag_capacity) + req_bytes +
            2 * sizeof(int);
  flat_init_records<Pol><<<(lanes + 127) / 128, 128>>>(static_cast<typename Pol::Rec*>(p.records),
                                                       lanes);
  cudaDeviceSynchronize();
  return p;
}


/// `G4GPU_FLAT_PROFILE=1`: each kernel of a round timed with events (which serialises the round
/// a little more than the read-back already does), summed over the run and printed at exit.
struct KernelClock {
  static constexpr int kN = 9;
  const char* name[kN] = {"claim", "reap+read", "factory", "setup", "pre",
                          "move",  "post",      "finish",  "tail"};
  double ms[kN] = {};
  long long rounds = 0;
  bool on = false;
  const char* model = "";
  // P39 (docs/RISK.md V320): what the set-up stage did, summed over the run.
  long long setup_launches = 0;  ///< set-up launches
  long long setup_lanes = 0;     ///< lanes listed for them (`setup_list`)
  long long setup_nuc = 0;       ///< set-ups that reached the nucleus
  long long setup_retry = 0;     ///< ... of a record that had had one before
  long long nuclei = 0;          ///< nuclei built
  long long factory_lane = 0;    ///< factory requests that fell back to the lane
  long long solo_launches = 0;   ///< `flat_solo` launches
  long long solo = 0;            ///< stragglers finished on their own lane
  bool factory = false;
  ~KernelClock() {
    if (!on || rounds == 0) { return; }
    double total = 0.0;
    for (double m : ms) { total += m; }
    std::printf("flat %s kernels over %lld rounds:", model, rounds);
    for (int k = 0; k < kN; ++k) { std::printf(" %s %.1f ms (%.0f%%)", name[k], ms[k], 100.0 * ms[k] / total); }
    std::printf("; %.1f ms\n", total);
    std::printf("flat %s set-up: %lld launches in %lld rounds, %lld lanes listed (%.1f a launch), "
                "%lld reached the nucleus (%.1f%% retries), %lld nuclei (%.2f a set-up) %s; "
                "%lld fell back to the lane; %lld stragglers finished solo in %lld launches "
                "(docs/RISK.md V320, V321)\n",
                model, setup_launches, rounds, setup_lanes,
                setup_launches ? double(setup_lanes) / setup_launches : 0.0, setup_nuc,
                setup_nuc ? 100.0 * setup_retry / setup_nuc : 0.0, nuclei,
                setup_nuc ? double(nuclei) / setup_nuc : 0.0,
                factory ? "from the factory" : "on the lanes", factory_lane, solo, solo_launches);
  }
};

template <class Pol>
int drain(
FlatPool& fp, const Scene<double>& scene, const had::PendingInteraction<double>* items,
          const int* idx, int n, int base, TrackBuffer<double> out, int batch, double* score,
          double* voxel_score, const had::HadronicWiring<double>& had,
          const had::InteractionPool<double>* pool_ptr, int n_tail_slots,
          vis::TrajectoryBuffer traj, int* status_warn, SecondaryArena sec, EmitterBooks books,
          StepTap<double> hook, FlatStats& stats) {
  if (n <= 0 || !fp.ok()) { return 0; }
  if (!fp.buffers_built) {
    flat_build_buffers<Pol><<<1, 1>>>(static_cast<bic::CascadeBuffers*>(fp.buffers), pool_ptr);
    fp.buffers_built = true;
  }
  FlatArgs<Pol> a;
  a.rec = static_cast<typename Pol::Rec*>(fp.records);
  a.ent = reinterpret_cast<FlatEntry*>(a.rec + fp.lanes);
  a.buffers = static_cast<bic::CascadeBuffers*>(fp.buffers);
  a.ctr = fp.counters;
  a.items = fp.items;
  a.strag = fp.strag;
  a.item_capacity = fp.item_capacity;
  a.k_steps = fp.k_steps;
  a.n = n;
  a.base = base;
  a.idx = idx;
  a.q = items;
  cudaMemsetAsync(fp.counters, 0, sizeof(int) * kFcNum);
  // Only as many lanes as there are entries: a drain of 300 runs 300 lanes.
  a.lanes = (n < fp.lanes) ? n : fp.lanes;
  // THE SPREAD. A launch of n busy lanes is cut into blocks of b so that it makes about 1,024
  // warps - the card holds 640 at these kernels' registers - and no more lanes share a warp
  // than that needs: a drain of 2,000 cascades in full 32-lane warps is 63 warps on 80 SMs
  // (V309 measured it), in 2-lane blocks 1,000 warps. `G4GPU_FLAT_SPREAD=n` sets the target.
  static const int spread_warps = [] {
    const char* v = std::getenv("G4GPU_FLAT_SPREAD");
    return v ? std::max(1, std::atoi(v)) : 1024;
  }();
  auto block_for = [](int busy) {
    int b = 1;
    while (b < 32 && b * spread_warps < busy) { b *= 2; }
    return b;
  };
  const int kTh = block_for(a.lanes);
  const int blocks = (a.lanes + kTh - 1) / kTh;
  int h[kFcNum];
  static KernelClock clk;
  static cudaEvent_t ev[KernelClock::kN + 1];
  // P39: the set-up's nuclei from the factory (docs/RISK.md V321). `G4GPU_FLAT_FACTORY=0`
  // samples them on the set-up's lanes as P36 did - integ/p36's flat path, to the digit.
  static const bool factory_on = [] {
    const char* v = std::getenv("G4GPU_FLAT_FACTORY");
    return !(v != nullptr && v[0] == '0');
  }();
  clk.factory = factory_on;
  a.solo = factory_on ? 1 : 0;
  if (!clk.on) {
    const char* v = std::getenv("G4GPU_FLAT_PROFILE");
    if (v != nullptr && v[0] == '1') {
      clk.on = true;
      clk.model = (Pol::kModel == had::InelasticModel::kBinary) ? "Binary" : "light-ion";
      for (cudaEvent_t& e : ev) { cudaEventCreate(&e); }
    }
  }
  auto tick = [&](int k) { if (clk.on) { cudaEventRecord(ev[k]); } };
  a.setup_list = fp.items + fp.item_capacity;
  a.tail_list = a.setup_list + fp.lanes;
  a.solo_list = a.tail_list + fp.lanes;
  // THE GATES. A set-up (a nucleus, up to 200 impact points) and a tail (the de-excitation) cost
  // a cascade milliseconds, a time step a fraction of one, and every launch waits for its slowest
  // lane - so the two long stages run only when enough lanes are waiting for them, or when
  // nothing is stepping. V309 measured the alternative (both every round): 85% of the drain.
  static const int setup_min_env = [] {
    const char* v = std::getenv("G4GPU_FLAT_SETUP_MIN");
    return v ? std::atoi(v) : -1;
  }();
  static const int tail_min_env = [] {
    const char* v = std::getenv("G4GPU_FLAT_TAIL_MIN");
    return v ? std::atoi(v) : -1;
  }();
  static const int setup_block = [] {
    const char* v = std::getenv("G4GPU_FLAT_SETUP_BLOCK");
    return v ? std::max(1, std::atoi(v)) : 0;
  }();
  static const int tail_block = [] {
    const char* v = std::getenv("G4GPU_FLAT_TAIL_BLOCK");
    return v ? std::max(1, std::atoi(v)) : 1;
  }();
  // P39 (docs/RISK.md V321): with the nuclei from the factory a set-up launch no longer waits
  // for a lane's `G4Fancy3DNucleus::Init` (1,522 draws of rejection sampling), so it runs every
  // round a lane waits for one. Measured on the factory build: the proton beam 5.55 -> 5.06 s and
  // its Binary drains 1,536 -> 597 rounds, Fe56 5k 25.9 -> 21.7 s (a light-ion miss now costs a
  // set-up round, V322); a gate of 8 lanes sat between the two (22.5 s).
  const int setup_min = (setup_min_env >= 0) ? setup_min_env
                        : factory_on          ? 1
                                              : std::max(32, a.lanes / 16);
  const int tail_min = (tail_min_env >= 0) ? tail_min_env : std::max(1, n_tail_slots / 2);
  // THE TAIL ON ITS OWN STREAM (P37, V303-V306 on integ/p37: this card runs kernels on separate
  // streams concurrently when they are launched together). A tail launch takes the lanes listed
  // in `tail_list` and nothing else touches a lane in `kStTail`: the set-up takes `setup_list`'s,
  // the step kernels the stepping ones, the claim free ones. So the de-excitation of one batch
  // runs while the others step. The round's claim and reap - the two kernels that read every
  // lane's stage and change a tail lane's - wait on the device for the last tail to finish
  // (`tail_busy`), so the overlap is the set-up and the step kernels', never the bookkeeping's.
  // `G4GPU_FLAT_TAIL_STREAM=0` puts it back in the round, for the comparison.
  static const bool tail_stream_on = [] {
    const char* v = std::getenv("G4GPU_FLAT_TAIL_STREAM");
    return !(v != nullptr && v[0] == '0');
  }();
  static cudaStream_t ts = nullptr;
  static cudaEvent_t tdone = nullptr;
  if (tail_stream_on && ts == nullptr) {
    cudaStreamCreateWithFlags(&ts, cudaStreamNonBlocking);
    cudaEventCreateWithFlags(&tdone, cudaEventDisableTiming);
  }
  bool tail_busy = false;
  for (;;) {
    if (tail_busy) {
      cudaStreamWaitEvent(0, tdone, 0);
      tail_busy = false;
    }
    tick(0);
    flat_claim<Pol><<<blocks, kTh>>>(a, scene, had);
    tick(1);
    cudaMemsetAsync(fp.counters + kFcActive, 0, sizeof(int));
    cudaMemsetAsync(fp.counters + kFcSetupN, 0, sizeof(int) * 3);
    cudaMemsetAsync(fp.counters + kFcSoloN, 0, sizeof(int));
    flat_reap<Pol><<<blocks, kTh>>>(a);
    cudaMemcpy(h, fp.counters, sizeof(h), cudaMemcpyDeviceToHost);
    tick(2);
    if (h[kFcActive] == 0 && h[kFcNext] >= n) { break; }
    const int n_setup = h[kFcSetupN], n_step = h[kFcStepN], n_tail = h[kFcTailN];
    const bool run_setup = n_setup > 0 && (n_setup >= setup_min || n_step == 0);
    const int sb = (setup_block > 0) ? setup_block : block_for(std::max(1, n_setup));
    if (run_setup && factory_on) {
      // P39: the model's top and the requests over `setup_list`, then the factory over the
      // requests (a block a nucleus, a lane a nucleon), in this stream before the set-up that
      // reads them. The lists are reset each launch; their grids are sized from `n_setup`, an
      // upper bound, and the blocks past the device's count return at once.
      cudaMemsetAsync(fp.factory_count, 0, 2 * sizeof(int));
      npool::FactoryLists lists;
      lists.req32 = static_cast<npool::FactoryRequest*>(fp.factory_req);
      lists.req64 = lists.req32 + fp.factory_cap;
      lists.count = fp.factory_count;
      lists.cap = fp.factory_cap;
      flat_request<Pol><<<(n_setup + sb - 1) / sb, sb>>>(a, n_setup, lists);
      const int most = std::min(n_setup * Pol::kNucleiPerTry, fp.factory_cap);
      npool::launch_nucleus_factory_list(lists.req32, lists.count, most, lists.req64,
                                         lists.count + 1, most, 0);
      stats.launches += 3;
    }
    tick(3);
    if (run_setup) {
      flat_setup<Pol><<<(n_setup + sb - 1) / sb, sb>>>(a, pool_ptr, n_setup);
      ++stats.launches;
      ++clk.setup_launches;
      clk.setup_lanes += n_setup;
    }
    tick(4);
    if (n_step > 0 || n_setup > 0) {
      cudaMemsetAsync(fp.counters + kFcItems, 0, sizeof(int));
      flat_pre<Pol><<<blocks, kTh>>>(a, pool_ptr);
      tick(5);
      {
        // About three items a stepping lane (V307: 2.6 for the Binary cascade, 16 for ions) -
        // counting the lanes the set-up above has just started, and never fewer than 1,024
        // threads: the loop strides, and a grid sized to nothing runs every item on one thread.
        const int est = std::max(1024, (n_step + n_setup) * Pol::kItemsPerLane);
        const int mb = block_for(est);
        flat_move<Pol><<<(est + mb - 1) / mb, mb>>>(a);
      }
      tick(6);
      flat_post<Pol><<<blocks, kTh>>>(a, pool_ptr);
      tick(7);
      flat_finish<Pol><<<blocks, kTh>>>(a, pool_ptr);
      stats.launches += 4;
    } else {
      tick(5);
      tick(6);
      tick(7);
    }
    tick(8);
    const bool idle = (n_step == 0 && n_setup == 0);
    if (n_tail > 0 && (n_tail >= tail_min || idle)) {
      const int nt = std::min(n_tail, n_tail_slots);
      cudaStream_t s = tail_stream_on ? ts : cudaStream_t(0);
      flat_tail<Pol><<<(nt + tail_block - 1) / tail_block, tail_block, 0, s>>>(
          a, scene, out, batch, score, voxel_score, had, pool_ptr, n_tail_slots, traj,
          status_warn, sec, books, hook, nt);
      if (tail_stream_on) {
        cudaEventRecord(tdone, ts);
        tail_busy = true;
      }
      ++stats.launches;
    } else if (n_tail > 0) {
      stats.tail_waits += n_tail;
    }
    // P39: the stragglers, when the drain has nothing else to give the card - every entry
    // claimed, nothing to set up, step or de-excite - all at once, a thread each, as the thread
    // path ran P36's hand-backs at the drain's end. They come back as ordinary records (a tail,
    // or the drain's next set-up) with their flag cleared.
    const int n_solo = h[kFcSoloN];
    if (n_solo > 0 && idle && n_tail == 0 && h[kFcNext] >= n) {
      const int b = block_for(n_solo);
      flat_solo<Pol><<<(n_solo + b - 1) / b, b>>>(a, pool_ptr, n_solo);
      ++stats.launches;
      clk.solo_launches += 1;
    }
    tick(9);
    if (clk.on) {
      cudaEventSynchronize(ev[9]);
      for (int k = 0; k < KernelClock::kN; ++k) {
        float m = 0.f;
        cudaEventElapsedTime(&m, ev[k], ev[k + 1]);
        clk.ms[k] += m;
      }
      ++clk.rounds;
    }
    stats.launches += 2;
    ++stats.rounds;
  }
  if (tail_busy) { cudaEventSynchronize(tdone); }
  stats.entries += n;
  stats.finished += h[kFcFinished];
  // The set-up counters are summed by the kernels over the drain; the last read-back (the round
  // that found nothing left) carries their totals.
  clk.setup_nuc += h[kFcSetupNuc];
  clk.setup_retry += h[kFcSetupRetry];
  clk.nuclei += h[kFcNuclei];
  clk.factory_lane += h[kFcFactoryLane];
  clk.solo += h[kFcSolo];
  for (int k = 0; k < 8; ++k) { stats.handed[k] += h[kFcHand0 + k]; }
  return h[kFcStrag];
}

}  // namespace g4gpu::host::flatk
