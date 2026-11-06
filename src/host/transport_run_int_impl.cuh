// The transport engine's two model-carrying kernel families: P15's interaction kernel and P19's
// EM-extra drain. INCLUDED BY THE SEVEN `transport_run_int_*.cu` UNITS, BY THE GENERATED
// `hook_int_*.cu` UNITS OF A HOOK PROJECT, AND BY NOTHING ELSE.
//
// WHY A SECOND IMPLEMENTATION HEADER. docs/RISK.md V65 and V189 split the KERNELS one per
// translation unit, so that ptxas sees one stepping kernel, or one hadronic model, at a time.
// The HEADERS were never split: both kernels below lived in host/transport_run_impl.cuh beside
// the stepping kernels, and their bodies need `physics/hadronic/interaction_apply.cuh`, the one
// header that includes every model - so every unit that included the implementation header read
// every model, and nvcc's dependency files (V221) showed a touched cascade header making all
// twenty-three units stale, 78 minutes, for a change that reaches one of them. The bodies are
// here now; host/transport_run_impl.cuh declares the two kernels, which is all the launches in
// `BeamOn`, `RaiseStackForInteractions` and the `extern template` block need. docs/RISK.md
// V233 has the before and after.
//
// THE RULE FOR INCLUDING IT. A unit that instantiates `run_interaction` or `run_emextra_drain`
// for a hook includes this header and writes one `template G4GPU_INTERACTION(...);` or
// `template G4GPU_EMX_DRAIN(...);` against the macros transport_run_impl.cuh defines;
// tools/gen_hook_units.ps1 writes exactly that for a hook project's heavy units. Everything
// else - the engine's host unit, the stepping units, a project's own .cu - includes
// host/transport_run_impl.cuh and never this. The two kernels' explicit instantiation
// DECLARATIONS stay in that header, so a launch there compiles against the declaration and
// links against the unit that instantiated the body; what keeps a launching unit from
// instantiating a body it cannot see is that it cannot see it.
#pragma once

#include "host/transport_run_impl.cuh"
#include "physics/hadronic/interaction_apply.cuh"

namespace g4gpu::host {

// ---------------------------------------------------------------- P15: the interaction kernel
//
/// One queued inelastic interaction or at-rest capture, per thread.
///
/// THE FIFTH KERNEL, and the one that carries the models. Every other kernel in this file steps
/// a track; this one finishes a step another kernel started. The split is docs/RISK.md V188: the
/// four entry points are 255 registers and ten to thirteen kilobytes of frame apiece and need
/// 1.6 MB of workspace per thread in flight, so a stepping kernel that called them would need
/// 92 GB for a 65,536-track batch - and would not compile, which was measured before it was
/// argued.
///
/// @tparam kBucket  WHICH MODEL this kernel carries, and therefore which translation unit it is
///                  in. One per bucket, because ptxas cannot compile four models into one
///                  module - `had::InteractionBucket`'s own comment and docs/RISK.md V189 have
///                  the measurement. The host bins the queue and launches only the buckets that
///                  have entries, so a proton run never launches the FTFP kernel at all.
/// @param idx     the queue indices of THIS bucket's entries, `n` of them from `base`. Built by
///                `count_interactions` and `scatter_interactions`, which are the same counting
///                sort the species dispatch already uses and for the same reason: a warp whose
///                threads take different physics paths serialises through all of them.
/// @param out     the same output pool the stepping kernels wrote into. The primary goes back
///                into it if `FillResult` says it survives, and the secondaries go into it
///                through the same `BufferEmitter` every other kernel uses.
///
/// THE STEP HOOK IS CALLED HERE, ONCE, for the step that queued this entry. `run_step_hadron`
/// deliberately did not call it - see `step_hadron`'s `queued` parameter - because a stepping
/// action reads `GetNumberOfSecondariesInCurrentStep()` and that count does not exist until the
/// model has run.
template <typename real_t, had::InteractionBucket kBucket, typename StepHook>
__global__ void run_interaction(Scene<real_t> scene, const had::PendingInteraction<real_t>* items,
                                const int* idx, int n, int base, TrackBuffer<real_t> out,
                                int batch, double* score, double* voxel_score,
                                had::HadronicWiring<real_t> had,
                                const had::InteractionPool<real_t>* pool_ptr,
                                vis::TrajectoryBuffer traj, int* status_warn, SecondaryArena sec,
                                EmitterBooks books, StepHook hook) {
  // The view, read through the one device copy the builder wrote: a pointer and not a value, so
  // that no launching translation unit needs its definition (docs/RISK.md V233).
  const had::InteractionPool<real_t>& pool = *pool_ptr;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) { return; }
  const had::PendingInteraction<real_t> q = items[idx[base + i]];
  TrackState<real_t> p = q.track;

  // The voxel cell the step STARTED in, recomputed rather than carried: it is three integer
  // divisions off `(volume_pre, pos_pre)` and the queue entry is already 408 bytes. The same
  // block `run_step_gamma` has, and the same reason - a per-voxel scorer wants the cell the
  // step began in, not the one it ended in.
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

  // A fourth RNG purpose, keyed by the same `(rng_key, step)` the stepper used. See
  // `had::kInteractionRngPurpose`: Philox cannot resume a partly consumed stream, so the
  // interaction opens its own rather than pretending to continue one.
  Philox<real_t> rng(p.rng_key, p.step, had::kInteractionRngPurpose);

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
  // `child_count` CONTINUES the step's own count rather than restarting at zero, so a delta ray
  // the stepper already emitted and a cascade proton emitted here get different
  // `child_rng_key`s. Restarting would give two secondaries of one step the same stream.
  BufferEmitter<real_t> em{out, p.pos, p.volume, p.event, p.rng_key, p.step, q.child_count,
                           q.track.global_time, p.weight,
                           TrackState<real_t>::pre_step_velocity(q.ekin_pre, track_mass),
                           sec, q.sec_last, &srep, books};

  real_t edep = q.edep;
  had::InteractionOutcome outc;
  had::InteractionSlot<real_t>* slot = pool.slot(i);
  if (slot == nullptr) {
    // Cannot happen under the chunked drain - the launch is `min(n_queued, n_slots)` wide - and
    // it is kept because the alternative to refusing is two threads in one 1.6 MB workspace.
    had::book_refusal<real_t>(had.books, had::HadronicRefusal::kInteractionNoSlot, p.ekin);
    outc.refusal = had::HadronicRefusal::kInteractionNoSlot;
  } else if constexpr (kBucket == had::InteractionBucket::kAtRest) {
    had::CompositionScratch<real_t> sc{};
    const physics::hadronic::MaterialComposition<real_t> mc =
        had::material_composition_of<real_t>(scene.materials[q.material], sc);
    physics::hadronic::HadProjectile<real_t> proj;
    proj.pdg = pdg_code(q.species);
    // -1 for the antiproton, 0 for mu-, pi- and K-. Read only by the Fritiof arm, which tests
    // `< -1` for an anti-nucleus; `stopping_arm` does the rest off the PDG code.
    proj.baryon_number = had::baryon_number_of(q.species, ion_a_of(p.ion_za));
    proj.charge = particle_def<real_t>(q.species).charge;
    proj.mass = track_mass;
    proj.kin_energy = real_t(0);  // at rest, by definition
    physics::hadronic::stopping::AtRestResult ar;
    outc = had::run_at_rest<real_t>(proj, mc, *slot, pool, i, had.level_data, pool.fermi,
                                    had::NuclearMassMeV(), rng, ar);
    edep += static_cast<real_t>(ar.local_deposit_MeV);
  } else {
    physics::hadronic::HadProjectile<real_t> proj;
    proj.pdg = pdg_code(q.species);
    proj.charge = particle_def<real_t>(q.species).charge;
    proj.mass = track_mass;
    proj.kin_energy = p.ekin;
    // THE BARYON NUMBER IS THE TRACK'S, and for a `kGenericIon` that is not the definition's.
    // `choose_hadronic_interaction` divides the energy by it, so a carbon ion whose baryon
    // number arrived as G4GenericIon's placeholder 1 would be offered FTFP at 4 GeV where
    // Geant4 offers the light-ion reaction at 0.33 GeV per nucleon.
    proj.baryon_number = had::baryon_number_of(q.species, ion_a_of(p.ion_za));
    const int pz = (q.species == ParticleType::kGenericIon) ? ion_z_of(p.ion_za) : 0;
    const int pa = (q.species == ParticleType::kGenericIon) ? ion_a_of(p.ion_za) : 0;
    // The neutron reads its own data set, which is `NeutronSubTables::inelastic` and not one of
    // the five `G4ParticleInelasticXS` ones - see `had::InelasticTables`'s own note.
    had::InelasticTables<real_t> xs = had.inelastic;
    xs.neutron = had.neutron.inelastic;
    // `kModelOfBucket` makes the arm a compile-time constant, so this unit contains one model.
    constexpr had::InelasticModel kModelOfBucket =
        (kBucket == had::InteractionBucket::kFtfp)     ? had::InelasticModel::kFtfp
        : (kBucket == had::InteractionBucket::kBertini) ? had::InelasticModel::kBertini
        : (kBucket == had::InteractionBucket::kBinary)  ? had::InelasticModel::kBinary
                                                        : had::InelasticModel::kLightIon;
    outc = had::run_inelastic<real_t, kModelOfBucket>(
        proj, q.species, scene.materials[q.material], xs, q.xs_at_step_start, *slot, pool, i,
        had.level_data, pool.fermi, pz, pa, rng);
  }

  bool alive = false;
  if (outc.rejected_by_integral_xs) {
    // `G4HadronicProcess::PostStepDoIt` returns the track unchanged: no interaction happened.
    // The step still ended here and still reports `fHadronInelastic` as what defined it, which
    // is what Geant4's own step reports after a rejection.
    alive = (p.ekin > em::kHadronTrackingCut<real_t>() && p.volume != geom::kOutsideWorld);
  } else if (outc.ran && slot != nullptr) {
    // FillResult, then the primary and the secondaries.
    //
    // FILLED IN PLACE, in the slot, by `had::fill_result_into` - the drains' way since P19 - and
    // not by `fill_result` returned by value into a placement new. The placement new stood here
    // from P15 to P24 on the argument that C++17's copy elision puts the prvalue straight into the
    // slot, and the elision does happen; what it cannot elide is the callee's own `out`, a named
    // local that `fill_result` returns, and nvcc keeps that in the kernel's frame. The frames say
    // so: at 256 entries every interaction kernel carried about 20 kB above its model
    // (`run_interaction<kFtfp>` 23,408 bytes against FTFP's own few), and the day P24 widened
    // the final state to 856 entries the four frames that had compiled grew by 42 to 48 kB each -
    // ftfp 23,408 -> 71,264, binary 90,592 -> 138,496 - which is 600 secondaries of 80 bytes,
    // and would have made the device stack raise ask for 143,360 bytes a thread, more than the
    // 3070 has for its resident threads (V190's arithmetic). `fill_result_into` writes the same
    // fields into `slot->filled` and nothing into the frame; `tests/test_emextra_wiring.cu`
    // holds it equal to `fill_result` field for field. docs/RISK.md V237's addendum has the
    // frames before and after.
    //
    // THE DIRECTION IT ROTATES INTO IS `p.dir` FOR AN IN-FLIGHT INTERACTION AND +z FOR AN
    // AT-REST ONE, and the second is not a shortcut.
    //
    // `G4HadronicProcess::FillResult` rotates every secondary by `aT.GetMomentumDirection()`,
    // because an inelastic model builds its final state about +z relative to the projectile.
    // The projectile's direction at the interaction point is `p.dir` - what the step's own
    // multiple scattering left it with - and not `q.dir_pre`, which the queue carries for the
    // step hook.
    //
    // `G4HadronStoppingProcess::AtRestDoIt` does NO such rotation: it adds the EM cascade's
    // gammas, the bound decay's products and the nuclear model's secondaries to the particle
    // change directly, because a stopped particle has no direction to rotate about.
    // `stopping::at_rest` builds them in the lab frame for the same reason. Passing `p.dir`
    // here would turn every at-rest capture through the arbitrary direction the track happened
    // to stop with - harmless for an isotropic distribution and wrong for the code, so +z it
    // is, which makes `rotate_uz` the identity.
    const Vec3<real_t> fill_dir = (q.kind == had::InteractionKind::kAtRest)
                                      ? Vec3<real_t>{real_t(0), real_t(0), real_t(1)}
                                      : p.dir;
    had::fill_result_into<real_t, had::kInteractionSecondaryCap, had::kInteractionSecondaryCap>(
        slot->fs, fill_dir, p.global_time, p.weight,
        /*has_at_rest_processes=*/had::has_at_rest_arm(q.species), slot->pdg_mass,
        slot->filled);
    edep += slot->filled.local_energy_deposit;
    srep.non_ionizing += slot->filled.non_ionizing_energy_deposit;
    em.pos = p.pos;
    em.volume = p.volume;
    em.event = p.event;
    had::emit_interaction_result<real_t>(slot->filled, em, had.books);
    if (slot->filled.status == physics::hadronic::TrackStatusChange::kAlive) {
      p.ekin = slot->filled.energy;
      p.dir = slot->filled.momentum_direction;
      alive = (p.ekin > em::kHadronTrackingCut<real_t>() && p.volume != geom::kOutsideWorld);
    } else {
      p.ekin = real_t(0);
    }
  }
  // ---- the two ledger groups, and they are booked on DIFFERENT conditions.
  //
  // THE `WHY` IS BOOKED WHENEVER THERE IS ONE, `ran` or not, and that is not a detail. An
  // at-rest capture that refuses its NUCLEAR half still emitted the atomic cascade's gammas -
  // P12b asserted 1,080,164 of them survive the Fritiof arm - so `run_at_rest` reports
  // `ran = true` and a real refusal together. Booking the reason only when nothing came back
  // would have made every one of those silent, which is the shape of hole this project keeps
  // writing up.
  if (outc.refusal != had::HadronicRefusal::kNumHadronicRefusals) {
    had::book_refusal<real_t>(had.books, outc.refusal, q.track.ekin);
    // P20: and, for a cascade refusal, WHICH of `bic::CascadeRefusal`'s names it carried - a
    // third booking on the same event, read beside the WHY row and never added to it.
    had::book_cascade_refusal_names<real_t>(had.books, outc.cascade_ref, q.track.ekin);
  }
  if (!outc.ran && !outc.rejected_by_integral_xs) {
    // The `HOW MUCH`: one booking per lost interaction. Which counter depends on WHICH PROCESS
    // was lost, not on the species alone - a stopped pi- whose capture produced nothing is
    // missing its whole rest mass, which is what `stopped_refusal_energy` computes and what the
    // three `kStopped*` counters have always meant.
    if (q.kind == had::InteractionKind::kAtRest) {
      const had::HadronicRefusal r = had::stopped_refusal(q.species);
      if (r != had::HadronicRefusal::kNumHadronicRefusals) {
        had::book_refusal<real_t>(
            had.books, r, had::stopped_refusal_energy<real_t>(q.species, q.track.ekin));
      }
    } else {
      had::book_refusal<real_t>(
          had.books,
          (q.species == ParticleType::kNeutron) ? had::HadronicRefusal::kNeutronInelastic
                                                : had::HadronicRefusal::kChargedHadronInelastic,
          q.track.ekin);
    }
    // The conservative disposal, as `kNeutronInelastic` has used since P8d and NOT what Geant4
    // does: the energy goes to the volume rather than into secondaries that leave it.
    //
    // AND ONLY FOR AN IN-FLIGHT INTERACTION. A stopped track's residual kinetic energy was
    // already deposited by `step_hadron`'s dying branch - that is the port's stop-at-the-
    // tracking-cut convention and it is also `G4Decay::DecayIt`'s own `energyDeposit` on the
    // at-rest branch - and `q.edep` carries it. Adding it again here would score the last few
    // tens of keV of every refused capture twice.
    if (q.kind != had::InteractionKind::kAtRest && q.score_slot >= 0) {
      edep += p.ekin;
      // AND WHERE IT WENT, booked a second time: how much of the scored dose this disposal put
      // in the scorer. `kRefusedEnergyScored` is what separates "a hole the dose cannot see"
      // from "a hole that IS the discrepancy" (docs/RISK.md V201).
      had::book_refusal<real_t>(had.books, had::HadronicRefusal::kRefusedEnergyScored, p.ekin);
    }
    p.ekin = real_t(0);
    srep.status = StepStatus::fStopAndKill;
  }

  ++p.step;
  p.advance(srep.true_length, q.ekin_pre, track_mass);
  // Whether the NEXT step of this track starts on a boundary. `run_step_hadron` does this and
  // the interaction kernel has to as well, because it is the kernel that writes the track out:
  // a queued step ended on the interaction and not on geometry, so the flag is CLEARED, and a
  // surviving primary that kept a stale `kFirstStepInVolume` would tell the next step's
  // multiple scattering to refresh its `fMinimal` limit at a boundary it never crossed.
  p.flags &= ~kFirstStepInVolume;
  // P22: A NEUTRON THAT SURVIVES THIS DRAWS ITS NEXT LENGTH AFRESH. The entry is a hadronic
  // PostStepDoIt, and `G4NeutronGeneralProcess::PostStepDoIt` clears its count before it
  // delegates to the inelastic sub-process ("In all cases clear number of interaction lengths").
  // `step_neutral` cleared it before queuing; it is written again here as constants so that the
  // two fields copied out of `q.track` are dead in this kernel and cost it no registers - the
  // primary's species is a runtime value here, so `TrackBuffer::append` cannot fold its test.
  p.ngp_left = real_t(-1);
  p.ngp_prev_step = srep.true_length;
  if (edep != real_t(0) && q.score_slot >= 0) {
    atomicAdd(&score[static_cast<size_t>(q.score_slot) * batch + p.event],
              static_cast<double>(edep));
    if (vcell >= 0 && voxel_score != nullptr) {
      atomicAdd(&voxel_score[vcell], static_cast<double>(edep));
    }
  }
  // NO `traj.add` HERE. The step's trajectory segment was recorded by the STEPPER, before the
  // enqueue: `step_hadron` calls it after the boundary handling and `step_neutral` after the
  // sub-process is named, and the inelastic branch of both is downstream of that call. Adding
  // it again here would draw every inelastic step twice in the viewer - not a dose, but a
  // picture that says a track was somewhere twice. `traj` is still passed so that a future
  // secondary drawn from this kernel has somewhere to go.
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
}

/// Every queued photo- or lepto-nuclear interaction of ONE bucket, drained by a fixed grid that
/// reads the queue's cursor on the device.
///
/// WHY THIS IS NOT `run_interaction` BEHIND P15's HOST DRAIN. That drain reads the queue's
/// cursor back to the host - a full device synchronisation - and only runs on an iteration with
/// a hadron alive, because running it unconditionally cost B1's 2,000,000-event photon gate 66%
/// of its event loop (`BeamOn`'s own measurement: 1,172 ms to 1,947 ms). A photon run can queue
/// a photo-nuclear interaction on any iteration it has a photon above 2 m_e, so it would pay
/// that on every one. This kernel is launched on every iteration with a photon (or an e+-, or a
/// muon) alive, as `n_slots` threads that each read `*cursor` and walk the queue in strides;
/// with nothing queued every thread returns at once and the launch costs its launch - no copy
/// back, no host wait. `tests/test_emextra_transport.cu` and the gate's own clock are the
/// measurement.
///
/// THREAD i TAKES SLOT i FOR EVERY ENTRY IT RUNS, one after another, which is P15's chunked drain
/// done inside one launch: the slot count is still a capacity whose shortage costs time and not
/// interactions. P15's drain runs BEFORE this one on the same stream, so the two never hold a
/// slot at once.
///
/// @param cursor   `HadronicWiring::emx_queue.cursor`, read here and never written: the engine
///                 rewinds it at the top of each iteration.
/// @param capacity the queue's; a cursor past it counts refused pushes, which the stepper has
///                 already booked as `kInelasticQueueFull`.
template <typename real_t, had::InteractionBucket kBucket, typename StepHook>
__global__ void run_emextra_drain(Scene<real_t> scene, const had::PendingInteraction<real_t>* items,
                                  const int* cursor, int capacity, TrackBuffer<real_t> out,
                                  int batch, double* score, double* voxel_score,
                                  had::HadronicWiring<real_t> had,
                                  const had::InteractionPool<real_t>* pool_ptr,
                                  vis::TrajectoryBuffer traj, int* status_warn,
                                  SecondaryArena sec, EmitterBooks books, StepHook hook) {
  // The view, read through the one device copy the builder wrote: a pointer and not a value, so
  // that no launching translation unit needs its definition (docs/RISK.md V233).
  const had::InteractionPool<real_t>& pool = *pool_ptr;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (items == nullptr || cursor == nullptr) { return; }
  const int n_queued = (*cursor < capacity) ? *cursor : capacity;
  if (tid == 0 && had.emx_stats != nullptr) {
    atomicMax(&had.emx_stats[kEmxQueueDepth], static_cast<unsigned long long>(*cursor));
  }
  const int stride = gridDim.x * blockDim.x;
  (void)traj;  // the stepper recorded the step's segment before it enqueued; see run_interaction
  for (int k = tid; k < n_queued; k += stride) {
    const had::PendingInteraction<real_t>& q = items[k];
    if (q.bucket != kBucket) { continue; }
    TrackState<real_t> p = q.track;
    if (had.emx_stats != nullptr) {
      atomicAdd(&had.emx_stats[(kBucket == had::InteractionBucket::kPhotoNuclear)
                                   ? had::kEmxPhotoDrained
                                   : had::kEmxLeptoDrained],
                1ull);
    }

    // The voxel cell the step STARTED in - `run_interaction`'s block, for the same reason.
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

    // `had::kInteractionRngPurpose`, keyed by the step as `run_interaction` keys it. The two
    // drains never see the same (key, step): a track queues at most one interaction per step.
    Philox<real_t> rng(p.rng_key, p.step, had::kInteractionRngPurpose);

    const had::EmExtraProcess process = had::emextra_process_of(q.species);
    StepReport<real_t> srep{};
    srep.true_length = q.true_length;
    srep.safety = q.safety;
    srep.non_ionizing = q.non_ionizing;
    srep.material = q.material;
    srep.status = q.status;
    srep.process = had::emextra_process_id(process);

    const real_t track_mass = particle_def<real_t>(q.species).mass;
    BufferEmitter<real_t> em{out, p.pos, p.volume, p.event, p.rng_key, p.step, q.child_count,
                             q.track.global_time, p.weight,
                             TrackState<real_t>::pre_step_velocity(q.ekin_pre, track_mass),
                             sec, q.sec_last, &srep, books};

    real_t edep = q.edep;
    bool alive = false;
    had::InteractionSlot<real_t>* slot = pool.slot(tid);
    had::InteractionOutcome outc;
    if (slot == nullptr) {
      // The grid is `n_slots` wide, so this cannot happen; kept for the reason P15 keeps its
      // own: the alternative to refusing is two threads in one workspace.
      outc.refusal = had::HadronicRefusal::kInteractionNoSlot;
    } else {
      physics::hadronic::HadProjectile<real_t> proj;
      proj.pdg = pdg_code(q.species);
      proj.charge = particle_def<real_t>(q.species).charge;
      proj.mass = track_mass;
      proj.kin_energy = p.ekin;
      proj.baryon_number = 0;
      // `had.gamma_general`: an empty photon store is Geant4's table-9 edge with the general
      // process on and a tripwire with it off (P21; `run_emextra`'s step 3).
      outc = had::run_emextra<real_t, kBucket>(proj, q.species, scene.materials[q.material],
                                               q.material, had.emextra, q.ekin_pre, *slot,
                                               had.level_data, pool.fermi, rng, nullptr,
                                               had.gamma_general);
    }
    if (outc.ran && slot != nullptr) {
      had::fill_result_into<real_t, had::kInteractionSecondaryCap, had::kInteractionSecondaryCap>(
          slot->fs, p.dir, p.global_time, p.weight, had::emextra_has_at_rest(q.species),
          slot->pdg_mass, slot->filled);
      edep += slot->filled.local_energy_deposit;
      srep.non_ionizing += slot->filled.non_ionizing_energy_deposit;
      em.pos = p.pos;
      em.volume = p.volume;
      em.event = p.event;
      had::emit_interaction_result<real_t>(slot->filled, em, had.books);
      // The primary goes back into the pool whenever `FillResult` did not kill it, at any
      // energy: a lepton below its tracking cut is the STEPPER's to finish - the electron
      // deposits the rest and the positron annihilates at rest - and a stopped mu- is captured
      // from `step_hadron`'s dying branch. Deciding that here, with a hadron's cut, would have
      // lost a positron's two photons.
      if (slot->filled.status == physics::hadronic::TrackStatusChange::kAlive) {
        p.ekin = slot->filled.energy;
        p.dir = slot->filled.momentum_direction;
        alive = (p.volume != geom::kOutsideWorld);
      } else if (slot->filled.status == physics::hadronic::TrackStatusChange::kStopButAlive) {
        p.ekin = real_t(0);
        alive = (p.volume != geom::kOutsideWorld);
      } else {
        p.ekin = real_t(0);
      }
    }
    // The two ledger groups, on the conditions `run_interaction` books them on.
    if (outc.refusal != had::HadronicRefusal::kNumHadronicRefusals) {
      had::book_refusal<real_t>(had.books, outc.refusal, q.track.ekin);
    }
    if (!outc.ran) {
      had::book_refusal<real_t>(had.books,
                                (process == had::EmExtraProcess::kPhotonNuclear)
                                    ? had::HadronicRefusal::kPhotoNuclear
                                    : had::HadronicRefusal::kLeptoNuclear,
                                q.track.ekin);
      // The conservative disposal every refused interaction in this engine gets, and NOT what
      // Geant4 does: the energy stays where the interaction was. A positron keeps its rest mass
      // out of the answer this way, which the WHERE row below says nothing about - it is booked
      // under the SIZE row's energy, which is the kinetic energy only.
      if (q.score_slot >= 0) {
        edep += p.ekin;
        had::book_refusal<real_t>(had.books, had::HadronicRefusal::kRefusedEnergyScored, p.ekin);
      }
      p.ekin = real_t(0);
      srep.status = StepStatus::fStopAndKill;
    }

    ++p.step;
    p.advance(srep.true_length, q.ekin_pre, track_mass);
    p.flags &= ~kFirstStepInVolume;
    // P22: no neutron reaches this kernel - its primaries are photons and leptons - so these
    // are a new track's values, written as constants for `run_interaction`'s reason: the
    // primary's species is a runtime value here and the copies out of `q.track` would
    // otherwise be live to the append.
    p.ngp_left = real_t(-1);
    p.ngp_prev_step = real_t(0);
    if (edep != real_t(0) && q.score_slot >= 0) {
      atomicAdd(&score[static_cast<size_t>(q.score_slot) * batch + p.event],
                static_cast<double>(edep));
      if (vcell >= 0 && voxel_score != nullptr) {
        atomicAdd(&voxel_score[vcell], static_cast<double>(edep));
      }
    }

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
  }
}

}  // namespace g4gpu::host
