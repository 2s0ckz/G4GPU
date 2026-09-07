// The transport engine's kernels and method bodies.
//
// WHY THIS IS A HEADER, AND WHAT THAT IS FOR
//
// It was a .cu until the step hook made it worth splitting. The kernels are __global__
// templates, and the reason they were kept out of a header is real: a __global__ template
// instantiated in two translation units of the same program emits a device stub into each,
// and nvcc rejects the duplicates with "explicit specialization ... is not a specialization
// of a function template", pointing at a generated stub file rather than at anything you
// wrote. That is still true and is still the trap.
//
// What makes it survivable now is that the collision is per *specialization*, not per
// template. TransportEngine<double, StepTap<double>> is instantiated exactly once, in
// transport_run.cu, and transport_run.cuh carries the matching `extern template` so that no
// other translation unit instantiates it implicitly. A project that wants a different hook
// instantiates a DIFFERENT specialization, whose kernels mangle to different symbols, and the
// two coexist.
//
// So the rule for including this file is narrow and worth stating plainly:
//
//   * A project that uses the stock hook must NOT include it. Include host/transport_run.cuh
//     and link out/transport_run.obj - which is what every example here does. Its own files
//     still go through nvcc, because the headers they include carry __host__ __device__
//     functions, but they compile with -c rather than -dc and instantiate no kernels, so they
//     compile in seconds.
//
//   * A project that wants its own hook includes it in exactly ONE .cu of its own, and writes
//     one explicit instantiation for its own hook type. It then compiles its own kernels and
//     never rebuilds g4gpu. That is the arrangement Geant4 has - the library is built once and
//     a project compiles against it - and it is what this split is for.
//
//   * Including it in two of your own translation units, or instantiating a hook type twice,
//     brings the duplicate-stub error back. One .cu, one instantiation.
//
// tests/test_custom_hook.cu is that second case, built and run by build_all.bat, so the
// arrangement is exercised rather than merely described.
#pragma once
#include <cstdio>
#include <cstdlib>

#include "host/transport_run.cuh"

namespace g4gpu::host {

// ---------------------------------------------------------------- kernels

/// Seeds the track buffers from primaries the host generated.
///
/// Replaces the kernel that sampled a Source record on the device. The reason is not
/// performance - that one was faster - but that it made the set of possible primaries closed:
/// a primary could only be whatever the gun's distributions could express. Generation happens
/// on the host now, once per event, in the user's own GeneratePrimaries. See
/// g4gpu::Primary and G4VUserPrimaryGeneratorAction.
///
/// The atomic is new and is the cost of the generality. The old kernel knew every primary was
/// the same species, so a track's slot was its event index and no counter was needed. Now an
/// event may start a gamma and the next an electron - a generator is free to do that - so each
/// species' buffer is appended to. One atomicAdd per event against a whole shower's transport
/// is not a cost worth designing around.
template <typename real_t>
__global__ void seed_from_primaries(TrackBuffer<real_t> pool,
                                    geom::Geometry<real_t> geometry,
                                    const Primary<real_t>* prim, int n, int event_base,
                                    unsigned int seed, const int* only = nullptr) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n) { return; }
  // `only` names the events to seed, for a repair pass that transports just the events
  // which lost a track. Without it every event in the batch is seeded, which is the normal
  // case and the one where `only` is null and this costs a predicated load.
  const int i = (only != nullptr) ? only[t] : t;
  const Primary<real_t> p = prim[i];

  // No routing. A primary of any species goes into the one pool and says what it is; whether a
  // kernel exists to step it is checked once, here, rather than discovered by an if-chain that
  // silently fell through to gammas for anything it did not recognise.
  TrackBuffer<real_t>* buf = &pool;
  const int slot = atomicAdd(buf->count, 1);
  if (slot >= buf->capacity) {
    // Counted, never silent - the same discipline TrackBuffer::append keeps, and the run
    // reports it as an overflow rather than quietly transporting fewer primaries than asked.
    atomicAdd(buf->overflow, 1);
    return;
  }

  // The per-track RNG key still comes from the event index and the run's seed, so the shower
  // a primary produces is reproducible for a given seed even though the primary itself was
  // generated on the host from Geant4's engine.
  // The GLOBAL event index, not the batch-local one. With the batch-local index every batch
  // replayed the same set of random streams - event 0 of the second batch drew exactly what
  // event 0 of the first had - so a 2M-event run in two batches was two correlated halves
  // rather than 2M independent showers. It went unseen because the batch was a fixed
  // 1048576 and nothing ever varied it; automatic sizing changed the batch, B1 moved from
  // 0.019 to 0.450 sigma against Geant4, and this was underneath.
  const unsigned int key = seed ^ static_cast<unsigned int>(event_base + i);
  buf->x[slot] = p.pos.x;   buf->y[slot] = p.pos.y;   buf->z[slot] = p.pos.z;
  buf->dx[slot] = p.dir.x;  buf->dy[slot] = p.dir.y;  buf->dz[slot] = p.dir.z;
  buf->ekin[slot] = p.ekin;
  buf->volume[slot] = geom::locate(geometry, p.pos);
  buf->event[slot] = i;
  buf->species[slot] = static_cast<int>(p.particle);
  buf->rng_key[slot] = key;
  buf->step[slot] = 0u;
  buf->msc_tlimit[slot] = real_t(0);
  buf->msc_tlimitmin[slot] = real_t(0);

  // The G4Track block for a primary. A primary has no parent and no creating process, which
  // is what Geant4 reports too - GetParentID() 0 and GetCreatorProcess() null - and its clock
  // starts at the vertex time the generator asked for rather than at zero, so a stepping
  // action reading GetGlobalTime() sees the gun's clock.
  seed_track_slot(*buf, slot, p.pos, p.dir, p.ekin, buf->volume[slot], p.t0);
}


// ---------------------------------------------------------------- the step hook
//
// Each kernel below is templated on a StepHook and calls it once, after the step, with the
// real pre- and post-step state. See core/step_hook.cuh for what a hook may do and - more
// importantly - what it must not store. NoStepHook is the do-nothing default and compiles
// away to nothing at all.
//
// The three kernels build their DeviceStep identically, so the rule for the fields that are
// not a plain copy is written once, here:
//
//   ekin_post   A track that dies inside the geometry has had its energy disposed of into
//               local deposits and secondaries, so its post-step kinetic energy is zero -
//               which is what Geant4 reports at the post-step point of a killed track. The
//               steppers do not bother zeroing p.ekin on those paths (an annihilating
//               positron and a Compton-absorbed photon both return with the pre-step value
//               still sitting in p.ekin), so reading p.ekin directly would be wrong. A track
//               that leaves the world keeps its energy: it was not absorbed, it left.
//
//   length      The TRUE path length reported by the stepper, never |pos_post - pos_pre|.
//               MSC deflects within a step, so the displacement is shorter than the distance
//               the particle actually ran, and a LET taken from the chord would be biased
//               high. This is what G4Step::GetStepLength returns.
//
//   score_slot  The scoring index of the volume the step STARTED in - the same slot the
//               deposit is added to a few lines below, so a hook that sums edep over a slot
//               gets exactly the number the scorer gets. tests/test_step_hook.cu asserts
//               that equality rather than trusting this comment.
/// Rebuilds the per-species index lists from the pool.
///
/// This is what buys the single pool. Storage is agnostic - a track sits wherever there was
/// room and says what it is - but dispatch cannot be: a warp whose threads take different
/// physics paths serialises through all of them, so each specialised kernel has to launch over
/// a contiguous run of one species. The lists provide that run without the tracks themselves
/// being separated, which is the distinction the five buffers used to conflate.
///
/// One atomic per track into its species' cursor. Order within a list is not deterministic and
/// does not need to be: a track's random stream is keyed on its own key and step, not on where
/// it sits or what it sits beside.
///
/// A species with no kernel gets index -1 from species_index and is counted rather than
/// scattered. Nothing routes it somewhere plausible; the engine reports it.
template <typename real_t>
__global__ void build_species_lists(TrackBuffer<real_t> pool, int n, int* const* lists,
                                    int* counts, int* unknown) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) { return; }
  const int sp = species_index(static_cast<ParticleType>(pool.species[i]));
  if (sp < 0) {
    atomicAdd(unknown, 1);
    return;
  }
  lists[sp][atomicAdd(&counts[sp], 1)] = i;
}

template <typename real_t, typename StepHook>
__global__ void run_step_gamma(Scene<real_t> scene, TrackBuffer<real_t> in, const int* idx,
                               TrackBuffer<real_t> out, int n, int batch, double* score,
                               double* voxel_score,
                               int n_step, vis::TrajectoryBuffer traj, int* status_warn,
                               SecondaryArena sec, StepHook hook) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) { return; }
  // Beyond the throttle this track is not stepped; it is written on unchanged and waits for
  // the next iteration.
  //
  // This is what replaces dropping. A kernel used to discover mid-flight that the output was
  // full, and the only thing it could do then was throw the track away - which loses the
  // energy that track would have deposited and makes the dose too low. The engine now works
  // out beforehand how many tracks the output can take, including the secondaries they will
  // make, and steps only that many. Nothing is discarded, nothing is transported twice, and a
  // tight iteration costs an extra pass rather than an answer.
  //
  // Passing through is not a step: p.step, the clocks and the path length are all left alone,
  // so a track that waits is indistinguishable from one that was never offered.
  // Through the index list: `i` walks this species' run, `idx[i]` is where that track
  // actually lives in the pool. The list is what keeps a warp's threads on one physics
  // path now that the pool holds every species together.
  TrackState<real_t> p;
  in.load(idx[i], p);
  if (i >= n_step) {
    out.append(p);
    return;
  }
  const int slot = (p.volume >= 0) ? scene.geometry.volumes[p.volume].score_index : -1;

  // Which cell the step *starts* in, if this volume is scored per voxel. Taken before the
  // step, because the step moves p.pos - and it is the pre-step point that G4PSEnergyDeposit3D
  // keys its hits map by. The navigator ends a step at every cell boundary for such a volume,
  // so the whole deposit belongs to this one cell rather than being spread over several.
  int vcell = -1;
  if (slot >= 0 && p.volume >= 0 && voxel_score != nullptr) {
    const auto& vv = scene.geometry.volumes[p.volume];
    if (vv.score_per_voxel) {
      const auto grid = geom::voxel_grid_of(vv.solid);
      const auto q = geom::to_local(vv.xform, p.pos);
      int ijk[3];
      geom::voxel_cell_of(grid, q, ijk);
      vcell = grid.index(ijk[0], ijk[1], ijk[2]);
    }
  }

  StepReport<real_t> srep;
  Philox<real_t> rng(p.rng_key, p.step, 0u);
  BufferEmitter<real_t> em{out, p.pos, p.volume, p.event, p.rng_key, p.step,
                           0u, p.global_time, p.weight, sec, -1, &srep};

  const real_t ekin_pre = p.ekin;
  const Vec3<real_t> pos_pre = p.pos;
  const Vec3<real_t> dir_pre = p.dir;
  const int volume_pre = p.volume;
  const bool first_in_vol = (p.flags & kFirstStepInVolume) != 0u;

  real_t edep = 0;
  constexpr ParticleType kSpecies = ParticleType::kGamma;
  const bool alive = step_gamma(scene, p, rng, em, edep, srep, traj);
  ++p.step;
  // The clocks, from the PRE-step energy: see TrackState::advance, transcribed from
  // G4Transportation::AlongStepDoIt. Done before the hook so a stepping action reads the
  // time at the end of its own step, as it would in Geant4.
  p.advance(srep.true_length, ekin_pre, particle_def<real_t>(kSpecies).mass);
  // Whether the NEXT step of this track starts on a boundary. What THIS step should report
  // was captured into first_in_vol before the step ran.
  p.flags = (srep.status == StepStatus::fGeomBoundary) ? (p.flags | kFirstStepInVolume)
                                                       : (p.flags & ~kFirstStepInVolume);
  if (edep != real_t(0) && slot >= 0) {
    atomicAdd(&score[static_cast<size_t>(slot) * batch + p.event], static_cast<double>(edep));
    if (vcell >= 0) { atomicAdd(&voxel_score[vcell], static_cast<double>(edep)); }
  }
  const bool escaped = (p.volume == geom::kOutsideWorld);
  // Everything a G4Step carries that the kernel already holds. Assigned by name rather than
  // built positionally: the struct has twenty fields and a silent reordering would be a
  // physics bug that still compiles. Fields the hook does not read are eliminated when it
  // inlines, so a hook that only wants edep pays for nothing else here.
  DeviceStep<real_t> ds{};
  ds.species = kSpecies;
  p.species = kSpecies;
  ds.ekin_pre = ekin_pre;
  ds.ekin_post = (alive || escaped) ? p.ekin : real_t(0);
  ds.edep = edep;
  ds.length = srep.true_length;
  ds.pos_pre = pos_pre;
  ds.pos_post = p.pos;
  ds.dir_pre = dir_pre;
  ds.dir_post = p.dir;
  ds.volume_pre = volume_pre;
  ds.volume_post = p.volume;
  ds.score_slot = slot;
  ds.event = p.event;
  ds.alive = alive;
  ds.track_ptr = &p;
  ds.first_in_volume = first_in_vol;
  ds.material = srep.material;
  ds.safety = srep.safety;
  ds.non_ionizing = srep.non_ionizing;
  // child_count is the emitter's own per-step counter - it exists to give each secondary a
  // deterministic RNG key - so this costs nothing.
  ds.n_secondaries = static_cast<int>(em.child_count);
  ds.sec_arena = sec;
  ds.sec_pool = &out;
  ds.sec_last = em.last_secondary;
  // The stepper reports fGeomBoundary for any boundary; only here is it known whether what
  // lay on the far side was the outside of the world.
  ds.status = (escaped && srep.status == StepStatus::fGeomBoundary) ? StepStatus::fWorldBoundary
                                                                   : srep.status;
  ds.process = srep.process;
  hook(ds);
  // A stepping action may have killed the track. Its request is resolved here rather than
  // inside the hook so that every kernel does it identically, and anything this transport
  // cannot honour is counted rather than quietly reinterpreted.
  const bool requeue = resolve_track_status<real_t>(alive, p.status);
  if (status_warn != nullptr && is_unsupported_track_status(p.status)) {
    atomicAdd(status_warn, 1);
  }
  if (requeue) { out.append(p); }
}

template <typename real_t, bool kIsPositron, typename StepHook>
__global__ void run_step_lepton(Scene<real_t> scene, TrackBuffer<real_t> in, const int* idx,
                                TrackBuffer<real_t> out, int n, int batch, double* score,
                                double* voxel_score,
                                int n_step, vis::TrajectoryBuffer traj, int* status_warn,
                                SecondaryArena sec, StepHook hook) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) { return; }
  // Through the index list: `i` walks this species' run, `idx[i]` is where that track
  // actually lives in the pool. The list is what keeps a warp's threads on one physics
  // path now that the pool holds every species together.
  TrackState<real_t> p;
  in.load(idx[i], p);
  if (i >= n_step) {
    // See run_step_gamma: not stepped, written on unchanged.
    out.append(p);
    return;
  }
  const int slot = (p.volume >= 0) ? scene.geometry.volumes[p.volume].score_index : -1;

  // See the note in run_step_gamma: the cell the step *starts* in, which is what
  // G4PSEnergyDeposit3D keys its hits map by.
  int vcell = -1;
  if (slot >= 0 && p.volume >= 0 && voxel_score != nullptr) {
    const auto& vv = scene.geometry.volumes[p.volume];
    if (vv.score_per_voxel) {
      const auto grid = geom::voxel_grid_of(vv.solid);
      const auto q = geom::to_local(vv.xform, p.pos);
      int ijk[3];
      geom::voxel_cell_of(grid, q, ijk);
      vcell = grid.index(ijk[0], ijk[1], ijk[2]);
    }
  }

  StepReport<real_t> srep;
  Philox<real_t> rng(p.rng_key, p.step, 0x5A5Au);
  BufferEmitter<real_t> em{out, p.pos, p.volume, p.event, p.rng_key, p.step,
                           0u, p.global_time, p.weight, sec, -1, &srep};

  const real_t ekin_pre = p.ekin;
  const Vec3<real_t> pos_pre = p.pos;
  const Vec3<real_t> dir_pre = p.dir;
  const int volume_pre = p.volume;
  const bool first_in_vol = (p.flags & kFirstStepInVolume) != 0u;

  real_t edep = 0;
  constexpr ParticleType kSpecies =
      kIsPositron ? ParticleType::kPositron : ParticleType::kElectron;
  const bool alive = step_lepton(scene, p, kIsPositron, rng, em, edep, srep, traj);
  ++p.step;
  // The clocks, from the PRE-step energy: see TrackState::advance, transcribed from
  // G4Transportation::AlongStepDoIt. Done before the hook so a stepping action reads the
  // time at the end of its own step, as it would in Geant4.
  p.advance(srep.true_length, ekin_pre, particle_def<real_t>(kSpecies).mass);
  // Whether the NEXT step of this track starts on a boundary. What THIS step should report
  // was captured into first_in_vol before the step ran.
  p.flags = (srep.status == StepStatus::fGeomBoundary) ? (p.flags | kFirstStepInVolume)
                                                       : (p.flags & ~kFirstStepInVolume);
  if (edep != real_t(0) && slot >= 0) {
    atomicAdd(&score[static_cast<size_t>(slot) * batch + p.event], static_cast<double>(edep));
    if (vcell >= 0) { atomicAdd(&voxel_score[vcell], static_cast<double>(edep)); }
  }
  const bool escaped = (p.volume == geom::kOutsideWorld);
  // Everything a G4Step carries that the kernel already holds. Assigned by name rather than
  // built positionally: the struct has twenty fields and a silent reordering would be a
  // physics bug that still compiles. Fields the hook does not read are eliminated when it
  // inlines, so a hook that only wants edep pays for nothing else here.
  DeviceStep<real_t> ds{};
  ds.species = kSpecies;
  p.species = kSpecies;
  ds.ekin_pre = ekin_pre;
  ds.ekin_post = (alive || escaped) ? p.ekin : real_t(0);
  ds.edep = edep;
  ds.length = srep.true_length;
  ds.pos_pre = pos_pre;
  ds.pos_post = p.pos;
  ds.dir_pre = dir_pre;
  ds.dir_post = p.dir;
  ds.volume_pre = volume_pre;
  ds.volume_post = p.volume;
  ds.score_slot = slot;
  ds.event = p.event;
  ds.alive = alive;
  ds.track_ptr = &p;
  ds.first_in_volume = first_in_vol;
  ds.material = srep.material;
  ds.safety = srep.safety;
  ds.non_ionizing = srep.non_ionizing;
  // child_count is the emitter's own per-step counter - it exists to give each secondary a
  // deterministic RNG key - so this costs nothing.
  ds.n_secondaries = static_cast<int>(em.child_count);
  ds.sec_arena = sec;
  ds.sec_pool = &out;
  ds.sec_last = em.last_secondary;
  // The stepper reports fGeomBoundary for any boundary; only here is it known whether what
  // lay on the far side was the outside of the world.
  ds.status = (escaped && srep.status == StepStatus::fGeomBoundary) ? StepStatus::fWorldBoundary
                                                                   : srep.status;
  ds.process = srep.process;
  hook(ds);
  // A stepping action may have killed the track. Its request is resolved here rather than
  // inside the hook so that every kernel does it identically, and anything this transport
  // cannot honour is counted rather than quietly reinterpreted.
  const bool requeue = resolve_track_status<real_t>(alive, p.status);
  if (status_warn != nullptr && is_unsupported_track_status(p.status)) {
    atomicAdd(status_warn, 1);
  }
  if (requeue) { out.append(p); }
}

/// One step of one proton or alpha.
///
/// Templated on the species rather than reading it from the track, for the same reason the
/// lepton kernel is templated on kIsPositron: a buffer holds exactly one species, so the type
/// is a compile-time constant and no thread in a warp diverges on it.
template <typename real_t, ParticleType kType, typename StepHook>
__global__ void run_step_hadron(Scene<real_t> scene, TrackBuffer<real_t> in, const int* idx,
                                TrackBuffer<real_t> out, int n, int batch, double* score,
                                double* voxel_score, int n_step,
                                vis::TrajectoryBuffer traj, int* status_warn,
                                SecondaryArena sec, StepHook hook) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) { return; }
  // Through the index list: `i` walks this species' run, `idx[i]` is where that track
  // actually lives in the pool. The list is what keeps a warp's threads on one physics
  // path now that the pool holds every species together.
  TrackState<real_t> p;
  in.load(idx[i], p);
  if (i >= n_step) {
    // See run_step_gamma: not stepped, written on unchanged.
    out.append(p);
    return;
  }
  const int slot = (p.volume >= 0) ? scene.geometry.volumes[p.volume].score_index : -1;

  // See the note in run_step_gamma: the cell the step *starts* in.
  int vcell = -1;
  if (slot >= 0 && p.volume >= 0 && voxel_score != nullptr) {
    const auto& vv = scene.geometry.volumes[p.volume];
    if (vv.score_per_voxel) {
      const auto grid = geom::voxel_grid_of(vv.solid);
      const auto q = geom::to_local(vv.xform, p.pos);
      int ijk[3];
      geom::voxel_cell_of(grid, q, ijk);
      vcell = grid.index(ijk[0], ijk[1], ijk[2]);
    }
  }

  // A third purpose value, so a hadron's stream is independent of the gamma and lepton streams
  // a track of the same key would have drawn.
  StepReport<real_t> srep;
  Philox<real_t> rng(p.rng_key, p.step, 0xB19Du);
  BufferEmitter<real_t> em{out, p.pos, p.volume, p.event, p.rng_key, p.step,
                           0u, p.global_time, p.weight, sec, -1, &srep};

  const real_t ekin_pre = p.ekin;
  const Vec3<real_t> pos_pre = p.pos;
  const Vec3<real_t> dir_pre = p.dir;
  const int volume_pre = p.volume;
  const bool first_in_vol = (p.flags & kFirstStepInVolume) != 0u;

  real_t edep = 0;
  constexpr ParticleType kSpecies = kType;
  const bool alive = step_hadron(scene, p, kType, rng, em, edep, srep, traj);
  ++p.step;
  // The clocks, from the PRE-step energy: see TrackState::advance, transcribed from
  // G4Transportation::AlongStepDoIt. Done before the hook so a stepping action reads the
  // time at the end of its own step, as it would in Geant4.
  p.advance(srep.true_length, ekin_pre, particle_def<real_t>(kSpecies).mass);
  // Whether the NEXT step of this track starts on a boundary. What THIS step should report
  // was captured into first_in_vol before the step ran.
  p.flags = (srep.status == StepStatus::fGeomBoundary) ? (p.flags | kFirstStepInVolume)
                                                       : (p.flags & ~kFirstStepInVolume);
  if (edep != real_t(0) && slot >= 0) {
    atomicAdd(&score[static_cast<size_t>(slot) * batch + p.event], static_cast<double>(edep));
    if (vcell >= 0) { atomicAdd(&voxel_score[vcell], static_cast<double>(edep)); }
  }
  const bool escaped = (p.volume == geom::kOutsideWorld);
  // Everything a G4Step carries that the kernel already holds. Assigned by name rather than
  // built positionally: the struct has twenty fields and a silent reordering would be a
  // physics bug that still compiles. Fields the hook does not read are eliminated when it
  // inlines, so a hook that only wants edep pays for nothing else here.
  DeviceStep<real_t> ds{};
  ds.species = kSpecies;
  p.species = kSpecies;
  ds.ekin_pre = ekin_pre;
  ds.ekin_post = (alive || escaped) ? p.ekin : real_t(0);
  ds.edep = edep;
  ds.length = srep.true_length;
  ds.pos_pre = pos_pre;
  ds.pos_post = p.pos;
  ds.dir_pre = dir_pre;
  ds.dir_post = p.dir;
  ds.volume_pre = volume_pre;
  ds.volume_post = p.volume;
  ds.score_slot = slot;
  ds.event = p.event;
  ds.alive = alive;
  ds.track_ptr = &p;
  ds.first_in_volume = first_in_vol;
  ds.material = srep.material;
  ds.safety = srep.safety;
  ds.non_ionizing = srep.non_ionizing;
  // child_count is the emitter's own per-step counter - it exists to give each secondary a
  // deterministic RNG key - so this costs nothing.
  ds.n_secondaries = static_cast<int>(em.child_count);
  ds.sec_arena = sec;
  ds.sec_pool = &out;
  ds.sec_last = em.last_secondary;
  // The stepper reports fGeomBoundary for any boundary; only here is it known whether what
  // lay on the far side was the outside of the world.
  ds.status = (escaped && srep.status == StepStatus::fGeomBoundary) ? StepStatus::fWorldBoundary
                                                                   : srep.status;
  ds.process = srep.process;
  hook(ds);
  // A stepping action may have killed the track. Its request is resolved here rather than
  // inside the hook so that every kernel does it identically, and anything this transport
  // cannot honour is counted rather than quietly reinterpreted.
  const bool requeue = resolve_track_status<real_t>(alive, p.status);
  if (status_warn != nullptr && is_unsupported_track_status(p.status)) {
    atomicAdd(status_warn, 1);
  }
  if (requeue) { out.append(p); }
}


// ---------------------------------------------------------------- method bodies

template <typename real_t, typename StepHook>
void TransportEngine<real_t, StepHook>::Upload(const g4::FlatScene& scene, int batch_size,
                                     int threads) {
    batch_ = batch_size;
    threads_ = threads;
    n_scorers_ = std::max<int>(1, static_cast<int>(G4SDManager::GetSDMpointer()->Scorers().size()));

    // The solid engine is mutually recursive, so the kernels need a real call stack; the 1 KB
    // default is not enough for one frame of the distance routine. Overflow surfaces as an
    // illegal memory access from an unrelated API call, with nothing pointing at the cause.
    G4GPU_CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 16384));

    n_volumes_ = static_cast<int>(scene.volumes.size());
    h_mats_.assign(scene.materials.m, scene.materials.m + scene.materials.count);
    n_materials_ = scene.materials.count;

    // Solid pool, transforms and aux first: the volumes' boolean children index into them.
    G4GPU_CUDA_CHECK(cudaMalloc(&d_pool_solids_,
                                sizeof(geom::Solid<real_t>) * std::max<size_t>(1, scene.pool.solids.size())));
    G4GPU_CUDA_CHECK(cudaMalloc(&d_pool_xforms_,
                                sizeof(geom::Transform<real_t>) * std::max<size_t>(1, scene.pool.xforms.size())));
    G4GPU_CUDA_CHECK(cudaMalloc(&d_pool_aux_,
                                sizeof(real_t) * std::max<size_t>(1, scene.pool.aux.size())));
    if (!scene.pool.solids.empty()) {
      G4GPU_CUDA_CHECK(cudaMemcpy(d_pool_solids_, scene.pool.solids.data(),
                                  sizeof(geom::Solid<real_t>) * scene.pool.solids.size(),
                                  cudaMemcpyHostToDevice));
    }
    if (!scene.pool.xforms.empty()) {
      G4GPU_CUDA_CHECK(cudaMemcpy(d_pool_xforms_, scene.pool.xforms.data(),
                                  sizeof(geom::Transform<real_t>) * scene.pool.xforms.size(),
                                  cudaMemcpyHostToDevice));
    }
    if (!scene.pool.aux.empty()) {
      G4GPU_CUDA_CHECK(cudaMemcpy(d_pool_aux_, scene.pool.aux.data(),
                                  sizeof(real_t) * scene.pool.aux.size(),
                                  cudaMemcpyHostToDevice));
    }

    G4GPU_CUDA_CHECK(cudaMalloc(&d_vols_, sizeof(geom::Volume<real_t>) * n_volumes_));
    G4GPU_CUDA_CHECK(cudaMemcpy(d_vols_, scene.volumes.data(),
                                sizeof(geom::Volume<real_t>) * n_volumes_,
                                cudaMemcpyHostToDevice));

    G4GPU_CUDA_CHECK(cudaMalloc(&d_mats_, sizeof(data::Material<real_t>) * n_materials_));
    G4GPU_CUDA_CHECK(cudaMemcpy(d_mats_, h_mats_.data(),
                                sizeof(data::Material<real_t>) * n_materials_,
                                cudaMemcpyHostToDevice));

    em::build_range_table<real_t>(h_mats_.data(), h_rt_, nullptr, n_materials_);
    G4GPU_CUDA_CHECK(cudaMalloc(&d_rt_, sizeof(h_rt_)));

    geom::Geometry<real_t> g{};
    g.volumes = d_vols_;
    g.n_volumes = n_volumes_;
    g.world = scene.world;
    // Voxel cells: one pool for every voxel volume in the scene, indexed by each solid's
    // own offset. A 512^3 CT is 268 MB of shorts, so it is uploaded once here and never
    // touched again.
    G4GPU_CUDA_CHECK(cudaMalloc(&d_voxels_,
                                sizeof(short) * std::max<size_t>(1, scene.pool.voxel_cells.size())));
    if (!scene.pool.voxel_cells.empty()) {
      G4GPU_CUDA_CHECK(cudaMemcpy(d_voxels_, scene.pool.voxel_cells.data(),
                                  sizeof(short) * scene.pool.voxel_cells.size(),
                                  cudaMemcpyHostToDevice));
    }

    // Mesh triangles and BVH nodes: one pool each for every tessellated solid in the scene,
    // with each solid's `a` naming its root node. A coarse CAD import is tens of thousands of
    // triangles at 72 bytes each, so this is uploaded once and never touched again.
    G4GPU_CUDA_CHECK(cudaMalloc(&d_tri_,
                                sizeof(real_t) * std::max<size_t>(1, scene.pool.tri.size())));
    G4GPU_CUDA_CHECK(cudaMalloc(&d_bvh_,
                                sizeof(real_t) * std::max<size_t>(1, scene.pool.bvh.size())));
    if (!scene.pool.tri.empty()) {
      G4GPU_CUDA_CHECK(cudaMemcpy(d_tri_, scene.pool.tri.data(),
                                  sizeof(real_t) * scene.pool.tri.size(),
                                  cudaMemcpyHostToDevice));
    }
    if (!scene.pool.bvh.empty()) {
      G4GPU_CUDA_CHECK(cudaMemcpy(d_bvh_, scene.pool.bvh.data(),
                                  sizeof(real_t) * scene.pool.bvh.size(),
                                  cudaMemcpyHostToDevice));
    }

    g.store.solids = d_pool_solids_;
    g.store.xforms = d_pool_xforms_;
    g.store.aux = d_pool_aux_;
    g.store.tri = d_tri_;
    g.store.bvh = d_bvh_;
    g.voxels.material = d_voxels_;
    g.voxels.count = static_cast<int>(scene.pool.voxel_cells.size());
    geom_ = g;

    // Physics tables. The element list comes from the materials actually in the scene, so a
    // detector made of anything is covered without a hand-maintained list.
    std::vector<int> zs = DistinctZ(scene);
    auto* d_pe = upload_photoelectric_for<real_t>(default_phot_dir(), zs);
    auto brem = upload_brems_for<real_t>(default_sb_dir(), h_mats_.data(), n_materials_, h_rt_,
                                         zs);
    G4GPU_CUDA_CHECK(cudaMemcpy(d_rt_, &h_rt_, sizeof(h_rt_), cudaMemcpyHostToDevice));
    auto* d_ray = upload_rayleigh<real_t>(default_rayl_dir(), zs.data(),
                                          static_cast<int>(zs.size()));
    auto* d_msc = upload_msc<real_t>(h_mats_.data(), n_materials_);

    // The hadron range table. Built here rather than on demand because a run that will carry
    // a proton needs it before the first primary is seeded, and the engine cannot know what
    // species the generator will produce until it has produced one. Two megabytes in double
    // precision, which is not worth a conditional - the electron table beside it is larger.
    {
      auto* h = new em::HadronRangeTable<real_t>();
      std::vector<real_t> cuts(n_materials_);
      for (int i = 0; i < n_materials_; ++i) { cuts[i] = h_mats_[i].cut_electron; }
      static em::ShellTables<real_t> shell;
      em::build_shell_tables(shell);
      em::build_hadron_range_table<real_t>(h_mats_.data(), cuts.data(), *h, &shell,
                                           n_materials_);
      G4GPU_CUDA_CHECK(cudaMalloc(&d_hrt_, sizeof(*h)));
      G4GPU_CUDA_CHECK(cudaMemcpy(d_hrt_, h, sizeof(*h), cudaMemcpyHostToDevice));
      delete h;
    }

    scene_ = Scene<real_t>{geom_,      d_mats_,  d_rt_,  d_pe,
                           brem.table, brem.sb,  d_ray,  d_msc,
                           d_hrt_,     static_cast<real_t>(scene.range_cut_mm),
                           -1,         processes_};

    // ---- how big a batch, and who decided.
    //
    // A batch needs pool_ = batch * live_per_event slots on each side of the ping-pong, and
    // track_arena_half_bytes() in core/track_buffer.cuh turns that into bytes by dry-running
    // the same allocator the carve then walks. One function, so the number that picks the
    // batch and the number that is consumed cannot disagree.
    {
      size_t free_b = 0, total_b = 0;
      const cudaError_t mem_err = cudaMemGetInfo(&free_b, &total_b);

      if (batch_ <= 0) {
        if (mem_err != cudaSuccess || free_b == 0) {
          // No way to ask: fall back to something that fits any card this runs on at all.
          batch_ = 262144;
          std::printf("batch: could not read device memory, defaulting to %d events\n", batch_);
        } else {
          // See SetMemoryFraction for why this is a share of free memory rather than all of it.
          const size_t budget = static_cast<size_t>(static_cast<double>(free_b) * mem_fraction_);
          long long lo = 1024, hi = 4194304;
          while (lo < hi) {
            const long long mid = lo + (hi - lo + 1) / 2;
            if (track_arena_half_bytes<real_t>(static_cast<long long>(mid * live_per_event_)) * 2 <= budget) {
              lo = mid;
            } else {
              hi = mid - 1;
            }
          }
          batch_ = static_cast<int>(lo);
          std::printf("batch: %d events (auto) - %.2f GB of track buffers, %.0f%% of %.2f GB "
                      "free\n",
                      batch_,
                      double(track_arena_half_bytes<real_t>(static_cast<long long>(batch_ * live_per_event_)) * 2) / 1073741824.0,
                      mem_fraction_ * 100.0, double(free_b) / 1073741824.0);
        }
      } else {
        const size_t want = track_arena_half_bytes<real_t>(static_cast<long long>(batch_ * live_per_event_)) * 2;
        std::printf("batch: %d events (set by the caller) - %.2f GB of track buffers\n", batch_,
                    double(want) / 1073741824.0);
        // REFUSED, not warned about and then attempted.
        //
        // This was a warning until it was tried. The reasoning behind the warning was that an
        // over-large request would fail in cudaMalloc, so announcing it first would give that
        // failure a cause. That is not what happens on Windows: WDDM lets CUDA oversubscribe
        // into host memory, so the allocation SUCCEEDS and the run then crawls, paging tracks
        // over PCIe. A batch 40% too large did not fail - it pinned the GPU until it was
        // killed.
        //
        // So the check has to be the thing that stops it. A caller who names a batch keeps the
        // right to name one; what they do not get is a run that looks like it started. The
        // largest batch that does fit is printed, because the useful response to this message
        // is to pick a number and guessing is a poor use of anyone's time.
        const size_t safe = static_cast<size_t>(static_cast<double>(free_b) * mem_fraction_);
        if (mem_err == cudaSuccess && want > safe) {
          long long lo = 1024, hi = batch_;
          while (lo < hi) {
            const long long mid = lo + (hi - lo + 1) / 2;
            if (track_arena_half_bytes<real_t>(static_cast<long long>(mid * live_per_event_)) * 2 <= safe) {
              lo = mid;
            } else {
              hi = mid - 1;
            }
          }
          std::printf("\nFATAL: a batch of %d needs %.2f GB of track buffers. This run may use\n"
                      "       %.0f%% of the %.2f GB free on this device, which is %.2f GB.\n"
                      "       Refused rather than attempted: CUDA on Windows oversubscribes\n"
                      "       into host memory instead of failing, so the run would appear to\n"
                      "       start and then crawl.\n"
                      "       The largest batch that fits is %lld. Pass 0 to size it\n"
                      "       automatically, or raise SetMemoryFraction if you know what else\n"
                      "       is on the card.\n",
                      batch_, double(want) / 1073741824.0, mem_fraction_ * 100.0,
                      double(free_b) / 1073741824.0, double(safe) / 1073741824.0, lo);
          std::exit(2);
        }
      }
    }

    // One pool per side of the ping-pong, and no division of it at all: a track carries its
    // own species, so storage does not need to know what is in it. Which kernel steps a track
    // is settled later and separately, by the index lists - dispatch, not storage.
    //
    // What that replaced: five capacities fixed before the run - gamma 2 per event, electron 4,
    // positron 1/2, proton 1, alpha 1, doubled for the ping-pong, 17 slots an event in total.
    // They had to be guessed, and measurement showed the guesses wrong in shape as well as
    // size: B1 peaks at 1.00 live gammas an event against 2 reserved and 0.90 electrons against
    // 4, because an electron's range is short and it dies within a step or two while a
    // Compton-scattered gamma keeps flying. Memory set aside for electrons could not be used by
    // gammas however the shower actually went.
    //
    // A pool has no shape to get wrong, so there is nothing here to seed it with.
    {
      // G4GPU_LIVE_PER_EVENT overrides the pool for any program, flag or no flag. It exists to
      // answer one question - is this answer converged in the pool size - which cannot be
      // asked of a program that has no such option, and which is exactly the question a
      // result that moved when the pool changed demands.
      if (const char* env = std::getenv("G4GPU_LIVE_PER_EVENT")) {
        const double v = std::atof(env);
        if (v >= 1.0) { live_per_event_ = v; }
      }
      pool_ = static_cast<long long>(batch_ * live_per_event_);
      // A capacity is an int the whole way down. The automatic sizer cannot ask for more than
      // this - it bisects to at most 4194304 events - but a caller who sets both a batch and a
      // live-track count can, and the arithmetic that follows would truncate rather than fail.
      if (pool_ > 2147483647LL) {
        std::printf("\nFATAL: a pool of %lld live track slots (%d events x %.1f) does not fit\n"
                    "       an int, and a track buffer's capacity is an int the whole way\n"
                    "       down. Lower -batch or -live.\n",
                    pool_, batch_, live_per_event_);
        std::exit(2);
      }
      arena_half_ = track_arena_half_bytes<real_t>(pool_);
      G4GPU_CUDA_CHECK(cudaMalloc(&d_track_arena_, arena_half_ * 2));
      G4GPU_CUDA_CHECK(cudaMalloc(&d_unknown_, sizeof(int)));
      G4GPU_CUDA_CHECK(cudaMemset(d_unknown_, 0, sizeof(int)));
      for (int i = 0; i < 2; ++i) {
        TrackSlab slab;
        slab.base = d_track_arena_ + arena_half_ * i;
        slab.capacity = arena_half_;
        // One buffer, the whole pool. No division, so nothing to guess.
        tracks_[i].alloc(static_cast<int>(pool_), &slab);
        if (slab.overflowed) {
          std::printf("\nFATAL: half the track arena measured %zu bytes but the pool carved\n"
                      "       from it did not fit. track_arena_half_bytes and\n"
                      "       allocate_track_buffer have diverged.\n", arena_half_);
          std::exit(2);
        }
        // The dispatch lists. Sized at the whole pool each because any one species may, in
        // principle, be all of it - a gamma beam's first iteration very nearly is. Four bytes
        // a slot times five is 8.6% on top of a 232-byte track, which is what the separation
        // between storage and dispatch costs.
        for (int sp = 0; sp < kNumTrackSpecies; ++sp) {
          G4GPU_CUDA_CHECK(cudaMalloc(&idx_[i][sp], sizeof(int) * pool_));
        }
        G4GPU_CUDA_CHECK(cudaMalloc(&d_idx_[i], sizeof(int*) * kNumTrackSpecies));
        G4GPU_CUDA_CHECK(cudaMemcpy(d_idx_[i], idx_[i], sizeof(int*) * kNumTrackSpecies,
                                    cudaMemcpyHostToDevice));
        G4GPU_CUDA_CHECK(cudaMalloc(&idx_n_[i], sizeof(int) * kNumTrackSpecies));
      }
      std::printf("track pool: %lld live slots per side (%.1f per event), %.2f GB total\n",
                  pool_, live_per_event_, double(arena_half_ * 2) / 1073741824.0);

      // The drop list, and the pointers into it that every buffer shares. Owned here rather
      // than carved from the slab, because a re-carve would lay track data over it.
      //
      // Sized at a sixteenth of the batch: an overflow that touches more events than that is
      // not an outlier to repair but a pool that is systematically too small, and the batch
      // retry is the right answer to that. The list running past its end is the signal.
      dropped_capacity_ = batch_ / 16 + 1024;
      G4GPU_CUDA_CHECK(cudaMalloc(&d_dropped_, sizeof(int) * dropped_capacity_));
      G4GPU_CUDA_CHECK(cudaMalloc(&d_dropped_count_, sizeof(int)));
      G4GPU_CUDA_CHECK(cudaMemset(d_dropped_count_, 0, sizeof(int)));
      for (int i = 0; i < 2; ++i) {
        tracks_[i].view.dropped_event = d_dropped_;
        tracks_[i].view.dropped_count = d_dropped_count_;
        tracks_[i].view.dropped_capacity = dropped_capacity_;
      }
    }
    G4GPU_CUDA_CHECK(cudaMalloc(&d_score_, sizeof(double) * n_scorers_ * batch_));
    G4GPU_CUDA_CHECK(cudaMalloc(&d_status_warn_, sizeof(int)));
    G4GPU_CUDA_CHECK(cudaMemset(d_status_warn_, 0, sizeof(int)));

    // The secondary arena. One entry per secondary created in a single kernel launch, so it is
    // sized against how many tracks can be in flight rather than against the length of a run.
    // Half the batch is generous for the EM physics here - B1 makes about 0.3 secondaries per
    // step - and the overflow counter says so if a future process makes it wrong.
    sec_.capacity = (sec_capacity_ > 0) ? sec_capacity_ : (batch_ / 2 + 1024);
    G4GPU_CUDA_CHECK(cudaMalloc(&sec_.entry, sizeof(unsigned int) * sec_.capacity));
    G4GPU_CUDA_CHECK(cudaMalloc(&sec_.prev, sizeof(int) * sec_.capacity));
    G4GPU_CUDA_CHECK(cudaMalloc(&sec_.cursor, sizeof(int)));
    G4GPU_CUDA_CHECK(cudaMalloc(&sec_.overflow, sizeof(int)));
    G4GPU_CUDA_CHECK(cudaMemset(sec_.cursor, 0, sizeof(int)));
    G4GPU_CUDA_CHECK(cudaMemset(sec_.overflow, 0, sizeof(int)));

    // Per-cell scoring, allocated only when a volume actually asks for it. One double per
    // cell in the whole pool - the cell index is already global across voxel volumes, so no
    // per-volume offsets are needed here beyond the ones the store already carries.
    //
    // Summed over the run, not per event: the score array above is n_scorers by batch, and a
    // per-event copy of that shape for a 512-cubed grid would be a hundred million doubles a
    // batch to allocate and to copy back. A dose *distribution* is a run-level quantity; the
    // per-event statistics that the volume totals carry are what an uncertainty is computed
    // from, and those are unaffected.
    n_voxel_cells_ = 0;
    for (int i = 0; i < static_cast<int>(scene.volumes.size()); ++i) {
      if (scene.volumes[i].score_per_voxel) {
        n_voxel_cells_ = static_cast<int>(scene.pool.voxel_cells.size());
        break;
      }
    }
    if (n_voxel_cells_ > 0) {
      G4GPU_CUDA_CHECK(cudaMalloc(&d_voxel_score_, sizeof(double) * n_voxel_cells_));
      G4GPU_CUDA_CHECK(cudaMemset(d_voxel_score_, 0, sizeof(double) * n_voxel_cells_));
    }
  }


template <typename real_t, typename StepHook>
RunStats TransportEngine<real_t, StepHook>::BeamOn(int n_events, const Primary<real_t>* primaries,
                                         unsigned int seed,
                                         std::vector<double>& score_sum,
                                         std::vector<double>& score_sum_sq,
                                         vis::TrajectoryBuffer traj, EventSink* sink) {
    score_sum.assign(n_scorers_, 0.0);
    score_sum_sq.assign(n_scorers_, 0.0);
    // Zeroed per run, not per batch: this accumulates the whole run's deposit per cell.
    if (d_voxel_score_ != nullptr) {
      G4GPU_CUDA_CHECK(cudaMemset(d_voxel_score_, 0, sizeof(double) * n_voxel_cells_));
    }
    RunStats st;
    if (d_status_warn_ != nullptr) {
      G4GPU_CUDA_CHECK(cudaMemset(d_status_warn_, 0, sizeof(int)));
    }
    if (sec_.overflow != nullptr) {
      G4GPU_CUDA_CHECK(cudaMemset(sec_.overflow, 0, sizeof(int)));
    }
    if (primaries == nullptr || n_events <= 0) { return st; }

    // The three species the track buffers carry. A primary of any other species would land in
    // the gamma buffer by default and be transported as a gamma - a run that produced a
    // plausible dose for physics that never happened. Refuse it instead.
    //
    // Checked over every primary now rather than once for the source, because a generator is
    // free to fire a different species per event and the guard has to cover what it actually
    // produced. It is one comparison per event against a whole shower.
    for (int i = 0; i < n_events; ++i) {
      const ParticleType p = primaries[i].particle;
      if (p == ParticleType::kGamma || p == ParticleType::kElectron
          || p == ParticleType::kPositron) {
        continue;
      }
      if (p == ParticleType::kProton || p == ParticleType::kAlpha) {
        // Stepped by run_step_hadron, which needs the range table. A scene uploaded without
        // one - b1_gpu_sched builds a gamma-only Scene deliberately - would step the primary
        // once, find a null table, and drop it. That is a silent zero, so it is refused here.
        if (scene_.hadron_range == nullptr) {
          std::printf(
              "\nFATAL: a proton or alpha primary was given to a Scene with no hadron range\n"
              "  table (event %d). step_hadron cannot advance a track without one, and\n"
              "  dropping it would look like a run that simply deposited nothing.\n",
              i);
          std::exit(2);
        }
        continue;
      }
      std::printf(
          "\nFATAL: primary species %d is not transported (event %d).\n"
          "  This transport carries gamma, e-, e+, proton and alpha. Other hadrons and ions\n"
          "  are refused: see G4RunManager::CheckSpecies for what is missing for each, and\n"
          "  tests/test_hadron_range.cu for the measured size of the He3 gap. Refusing\n"
          "  rather than transporting it as something it is not.\n",
          static_cast<int>(p), i);
      std::exit(2);
    }

    // The primaries, uploaded a batch at a time. Sized once for the batch rather than for the
    // run: a ten-million-event run would otherwise want half a gigabyte resident for
    // something the device reads once.
    if (d_primaries_ == nullptr) {
      G4GPU_CUDA_CHECK(cudaMalloc(&d_primaries_, sizeof(Primary<real_t>) * batch_));
    }

    std::vector<double> ev_scores(static_cast<size_t>(n_scorers_), 0.0);

    std::vector<double> h_score(static_cast<size_t>(n_scorers_) * batch_);
    cudaEvent_t t0, t1;
    G4GPU_CUDA_CHECK(cudaEventCreate(&t0));
    G4GPU_CUDA_CHECK(cudaEventCreate(&t1));
    G4GPU_CUDA_CHECK(cudaEventRecord(t0));

    // Batches, with a retry if one runs out of buffer space.
    //
    // A track that cannot be appended is a track that stops being transported, and its energy
    // never reaches a scorer - so a run that drops one reports a dose that is too low while
    // looking exactly like a result. Ending the run on that is safe but wasteful: the work is
    // thrown away and the user is left to guess a setting.
    //
    // Retrying is better and costs nothing but the attempt. The arena is a fixed size, so
    // HALVING THE BATCH DOUBLES THE SLOTS PER EVENT - the same memory, fewer events sharing
    // it - which is exactly the shortage that caused the overflow. And the retry is not an
    // approximation of the run that failed: every track's random stream is keyed on its GLOBAL
    // event index, so events transported in a smaller batch draw precisely what they would
    // have drawn in a larger one. The answer is the one a correctly sized run would have given.
    //
    // Nothing is accumulated until an attempt completes without dropping anything, so a failed
    // attempt contributes nothing to the totals.
    int base = 0;
    int try_batch = batch_;
    while (base < n_events) {
      const int n_batch = std::min(try_batch, n_events - base);
      G4GPU_CUDA_CHECK(cudaMemset(d_score_, 0, sizeof(double) * n_scorers_ * batch_));
      // This attempt's drops, not a previous attempt's.
      for (int i = 0; i < 2; ++i) { tracks_[i].reset_overflow(); }
      const long long steps_before = st.track_steps;
      const long long abandoned_before = st.abandoned;

      int cur = 0, nxt = 1;
      tracks_[cur].reset();
      G4GPU_CUDA_CHECK(cudaMemcpy(d_primaries_, primaries + base,
                                  sizeof(Primary<real_t>) * n_batch, cudaMemcpyHostToDevice));
      // No partition to do: the primaries all go into the one pool.
      //
      // This is where five buffers hurt most. Every primary needs a slot at the same instant,
      // and in a gamma beam they are all one species, so a fixed split gave that species its
      // fraction and dropped the rest - and the throttle could not help, because nothing had
      // been stepped yet. With one pool the only question is whether the batch fits, which is
      // one comparison rather than a division nobody could get right in advance.
      if (n_batch > tracks_[cur].view.capacity) {
        std::printf("\nFATAL: %d primaries do not fit a pool of %d slots. Raise\n"
                    "       SetLiveTracksPerEvent (now %.1f per event) or lower the batch.\n",
                    n_batch, tracks_[cur].view.capacity, live_per_event_);
        std::exit(3);
      }
      seed_from_primaries<real_t><<<(n_batch + threads_ - 1) / threads_, threads_>>>(
          tracks_[cur].view, geom_, d_primaries_, n_batch, base, seed);
      G4GPU_CUDA_CHECK(cudaGetLastError());

      // Bounded so a non-terminating track shows up as an explicit abandonment rather than a
      // hang. B1 drains in about 110 iterations.
      constexpr int kMaxIterations = 1000;
      int leftover = 0;
      for (int iteration = 0; iteration < kMaxIterations; ++iteration) {
        // How many tracks are alive, of any species. One number, because there is one pool.
        const int n_live = tracks_[cur].count();

        // The dispatch lists for this iteration: which slots hold gammas, which electrons, and
        // so on. Rebuilt every time because the pool is written in append order, which is
        // whatever the atomics gave out, and a kernel needs its own species contiguous.
        int nsp[kNumTrackSpecies] = {0, 0, 0, 0, 0};
        if (n_live > 0) {
          G4GPU_CUDA_CHECK(cudaMemset(idx_n_[cur], 0, sizeof(int) * kNumTrackSpecies));
          build_species_lists<real_t><<<(n_live + threads_ - 1) / threads_, threads_>>>(
              tracks_[cur].view, n_live, d_idx_[cur], idx_n_[cur], d_unknown_);
          G4GPU_CUDA_CHECK(cudaGetLastError());
          G4GPU_CUDA_CHECK(cudaMemcpy(nsp, idx_n_[cur], sizeof(int) * kNumTrackSpecies,
                                      cudaMemcpyDeviceToHost));
        }
        const int ng = nsp[kSpeciesGamma];
        const int ne = nsp[kSpeciesElectron];
        const int np = nsp[kSpeciesPositron];
        const int nh = nsp[kSpeciesProton];
        const int na = nsp[kSpeciesAlpha];
        if (n_live == 0) {
          st.max_iterations = std::max(st.max_iterations, iteration);

          // Every track is done - unless some were dropped, in which case the events that lost
          // them are transported again, here, and nothing else is.
          //
          // This is the whole reason the drop list exists. Discarding the batch would redo a
          // million healthy events to repair one shower; redoing the events that actually lost
          // something costs their showers and nothing more. The answer is the same either way,
          // because a track's random stream is keyed on its global event index and not on when
          // or alongside what it was transported.
          int n_dropped = 0;
          if (d_dropped_count_ != nullptr) {
            G4GPU_CUDA_CHECK(cudaMemcpy(&n_dropped, d_dropped_count_, sizeof(int),
                                        cudaMemcpyDeviceToHost));
          }
          if (n_dropped == 0) { break; }
          if (n_dropped > dropped_capacity_) {
            // More events were hit than the list can name, so it is a partial record and there
            // is no way to know which events to repair. That is not an outlier - it is a pool
            // that is too small for this problem - and the batch retry outside handles it.
            break;
          }

          std::vector<int> ids(static_cast<size_t>(n_dropped));
          G4GPU_CUDA_CHECK(cudaMemcpy(ids.data(), d_dropped_, sizeof(int) * n_dropped,
                                      cudaMemcpyDeviceToHost));
          std::sort(ids.begin(), ids.end());
          ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
          // An id outside the batch would mean the list itself is corrupt; drop the pass rather
          // than seed from it.
          ids.erase(std::remove_if(ids.begin(), ids.end(),
                                   [&](int e) { return e < 0 || e >= n_batch; }),
                    ids.end());
          if (ids.empty()) { break; }

          // What those events deposited on the attempt that lost a track is not a partial
          // answer to be added to - it is wrong, and it is replaced. Zeroing first is what
          // makes the repair a redo rather than a double count.
          for (int e : ids) {
            for (int sc = 0; sc < n_scorers_; ++sc) {
              G4GPU_CUDA_CHECK(cudaMemset(d_score_ + static_cast<size_t>(sc) * batch_ + e, 0,
                                          sizeof(double)));
            }
          }
          st.events_repaired += static_cast<long long>(ids.size());
          std::printf("repair: %zu of %d events lost a track and are being transported\n"
                      "        again; the rest of the batch is untouched\n",
                      ids.size(), n_batch);

          const int n_repair = static_cast<int>(ids.size());
          G4GPU_CUDA_CHECK(cudaMemcpy(d_dropped_, ids.data(), sizeof(int) * n_repair,
                                      cudaMemcpyHostToDevice));
          G4GPU_CUDA_CHECK(cudaMemset(d_dropped_count_, 0, sizeof(int)));

          tracks_[cur].reset();
          seed_from_primaries<real_t><<<(n_repair + threads_ - 1) / threads_, threads_>>>(
              tracks_[cur].view, geom_, d_primaries_, n_repair, base, seed, d_dropped_);
          G4GPU_CUDA_CHECK(cudaGetLastError());
          continue;  // drain again, this time carrying only the repaired events
        }
        if (iteration == kMaxIterations - 1) { leftover = ng + ne + np + nh + na; }
        
        st.peak_gamma = std::max(st.peak_gamma, ng);
        st.peak_electron = std::max(st.peak_electron, ne);
        st.peak_positron = std::max(st.peak_positron, np);
        st.peak_proton = std::max(st.peak_proton, nh);
        st.peak_alpha = std::max(st.peak_alpha, na);

        // How many tracks may be stepped this iteration.
        //
        // Nothing is re-divided any more. There is one output pool, so there is one budget:
        // every live track needs a slot in it whether it is stepped or merely passed through,
        // and what is left over after that is what pays for secondaries.
        //
        // The fan-out bound is per STEP rather than per species, which is the simplification
        // the single pool buys. One step runs one discrete process, and the largest number of
        // tracks any single step in stepper.cuh produces is three - a delta ray, then the two
        // annihilation photons of a positron that stops. Four is that with a margin, and it no
        // longer has to be reasoned separately for each species and each destination.
        //
        // A species that cannot be stepped in full this iteration is not truncated: the
        // remainder passes through untouched and is stepped next time.
        constexpr int kMaxSecondariesPerStep = 4;
        const int cap_out = tracks_[nxt].view.capacity;
        int budget = (cap_out > n_live) ? (cap_out - n_live) / kMaxSecondariesPerStep : 0;
        int step_n[kNumTrackSpecies] = {0, 0, 0, 0, 0};
        for (int sp = 0; sp < kNumTrackSpecies; ++sp) {
          step_n[sp] = std::min(nsp[sp], budget);
          budget -= step_n[sp];
        }
        const int step_g = step_n[kSpeciesGamma];
        const int step_e = step_n[kSpeciesElectron];
        const int step_p = step_n[kSpeciesPositron];
        const int step_h = step_n[kSpeciesProton];
        const int step_a = step_n[kSpeciesAlpha];
        if (step_g + step_e + step_p + step_h + step_a == 0) {
          std::printf("\nFATAL: no track can be stepped without overrunning the pool.\n"
                      "       %d tracks are live and the pool holds %d slots a side. Raise\n"
                      "       SetLiveTracksPerEvent (now %.1f per event) or lower the batch.\n",
                      n_live, cap_out, live_per_event_);
          std::exit(3);
        }
        st.throttled += n_live - (step_g + step_e + step_p + step_h + step_a);
        st.track_steps += step_g + step_e + step_p + step_h + step_a;

        tracks_[nxt].reset();
        // The secondary arena holds one iteration worth of chains and no more: the chains
        // a stepping action walks are the ones made by the launch it is running in, and
        // nothing reads them afterwards. Resetting here is what keeps the arena sized
        // against tracks in flight rather than against the length of the run.
        if (sec_.cursor != nullptr) {
          G4GPU_CUDA_CHECK(cudaMemsetAsync(sec_.cursor, 0, sizeof(int)));
        }
        // One launch per species, each over its own index list, each into the one output pool.
        //
        // Five specialised kernels remain because a warp whose threads take different physics
        // paths serialises through all of them - that is the reason the species list survives
        // at all. What has gone is the idea that a specialised KERNEL needs a separate BUFFER.
        if (ng > 0) {
          run_step_gamma<real_t><<<(ng + threads_ - 1) / threads_, threads_>>>(
              scene_, tracks_[cur].view, idx_[cur][kSpeciesGamma], tracks_[nxt].view, ng,
              batch_, d_score_, d_voxel_score_, step_g, traj, d_status_warn_, sec_, hook_);
        }
        if (ne > 0) {
          run_step_lepton<real_t, false><<<(ne + threads_ - 1) / threads_, threads_>>>(
              scene_, tracks_[cur].view, idx_[cur][kSpeciesElectron], tracks_[nxt].view, ne,
              batch_, d_score_, d_voxel_score_, step_e, traj, d_status_warn_, sec_, hook_);
        }
        if (np > 0) {
          run_step_lepton<real_t, true><<<(np + threads_ - 1) / threads_, threads_>>>(
              scene_, tracks_[cur].view, idx_[cur][kSpeciesPositron], tracks_[nxt].view, np,
              batch_, d_score_, d_voxel_score_, step_p, traj, d_status_warn_, sec_, hook_);
        }
        if (nh > 0) {
          run_step_hadron<real_t, ParticleType::kProton>
              <<<(nh + threads_ - 1) / threads_, threads_>>>(
                  scene_, tracks_[cur].view, idx_[cur][kSpeciesProton], tracks_[nxt].view, nh,
                  batch_, d_score_, d_voxel_score_, step_h, traj, d_status_warn_, sec_, hook_);
        }
        if (na > 0) {
          run_step_hadron<real_t, ParticleType::kAlpha>
              <<<(na + threads_ - 1) / threads_, threads_>>>(
                  scene_, tracks_[cur].view, idx_[cur][kSpeciesAlpha], tracks_[nxt].view, na,
                  batch_, d_score_, d_voxel_score_, step_a, traj, d_status_warn_, sec_, hook_);
        }
        G4GPU_CUDA_CHECK(cudaGetLastError());
        cur ^= 1;
        nxt ^= 1;
      }
      st.abandoned += leftover;

      // Did this attempt lose anybody? Checked before a single number is accumulated.
      const long long dropped =
          tracks_[0].overflow() + tracks_[1].overflow();
      if (dropped > 0) {
        // Below this a batch is too small to be worth splitting further, and something other
        // than batch size is wrong - a single event whose shower cannot fit the whole arena.
        constexpr int kMinBatch = 256;
        if (try_batch > kMinBatch && !allow_dropped_) {
          try_batch /= 2;
          ++st.batch_retries;
          st.track_steps = steps_before;
          st.abandoned = abandoned_before;
          std::printf("batch %d..%d ran out of track slots; retrying %d events at a time\n",
                      base, base + n_batch, try_batch);
          continue;  // same events, smaller batch, nothing accumulated
        }
        if (!allow_dropped_) {
          std::printf("\n*** %lld TRACKS WERE DROPPED ***\n\n"
                      "    A single batch of %d events could not fit the track arena even\n"
                      "    after halving it down from %d. One event's shower needs more live\n"
                      "    slots than the whole pool holds, so a smaller batch cannot help.\n"
                      "    Raise SetLiveTracksPerEvent (currently %.1f per event).\n\n",
                      dropped, n_batch, batch_, live_per_event_);
          std::exit(3);
        }
        st.overflow += dropped;
      }

      G4GPU_CUDA_CHECK(cudaMemcpy(h_score.data(), d_score_,
                                  sizeof(double) * n_scorers_ * batch_,
                                  cudaMemcpyDeviceToHost));
      // Event-major, so that a sink sees one event's scores together. Each accumulator still
      // sums over events in ascending order, so the totals are bit-identical to the
      // scorer-major loop this replaced - which matters, because those totals are what the
      // 0.09-sigma agreement with Geant4 is measured on.
      for (int e = 0; e < n_batch; ++e) {
        for (int sc = 0; sc < n_scorers_; ++sc) {
          const double v = h_score[static_cast<size_t>(sc) * batch_ + e];
          score_sum[sc] += v;
          score_sum_sq[sc] += v * v;
          ev_scores[sc] = v;
        }
        if (sink != nullptr) { sink->Event(base + e, ev_scores.data(), n_scorers_); }
      }
      base += n_batch;
    }

    // The per-cell deposit, brought back once at the end of the run.
    h_voxel_score_.clear();
    if (d_voxel_score_ != nullptr && n_voxel_cells_ > 0) {
      h_voxel_score_.resize(static_cast<std::size_t>(n_voxel_cells_));
      G4GPU_CUDA_CHECK(cudaMemcpy(h_voxel_score_.data(), d_voxel_score_,
                                  sizeof(double) * n_voxel_cells_, cudaMemcpyDeviceToHost));
    }

    G4GPU_CUDA_CHECK(cudaEventRecord(t1));
    G4GPU_CUDA_CHECK(cudaEventSynchronize(t1));
    float ms = 0;
    G4GPU_CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    st.milliseconds = ms;
    // st.overflow is accumulated per attempt above, and is only ever non-zero when a caller
    // asked for best-effort transport with AllowDroppedTracks. Reading the counters here
    // instead would report the last attempt's, which is zero by construction.
    if (d_status_warn_ != nullptr) {
      int warn = 0;
      G4GPU_CUDA_CHECK(cudaMemcpy(&warn, d_status_warn_, sizeof(int), cudaMemcpyDeviceToHost));
      st.unsupported_track_status = warn;
    }
    if (st.throttled > 0) {
      std::printf("throttled: %lld track-steps deferred to a later iteration (pool %.1f per "
                  "event)\n", st.throttled, live_per_event_);
    }
    if (d_unknown_ != nullptr) {
      int unk = 0;
      G4GPU_CUDA_CHECK(cudaMemcpy(&unk, d_unknown_, sizeof(int), cudaMemcpyDeviceToHost));
      if (unk > 0) {
        std::printf("\n*** %d TRACKS HAVE NO STEPPING KERNEL ***\n"
                    "    Their species is not in species_index, so they were never\n"
                    "    dispatched and their energy was never deposited.\n\n", unk);
      }
    }
    if (sec_.overflow != nullptr) {
      int over = 0;
      G4GPU_CUDA_CHECK(cudaMemcpy(&over, sec_.overflow, sizeof(int), cudaMemcpyDeviceToHost));
      st.secondary_overflow = over;
    }
    // A dropped track is a track whose energy never reached a scorer, so the dose this run
    // reports is too low by whatever it would have deposited - and the run otherwise finishes
    // normally and prints a number that looks exactly like a result.
    //
    // So this ends the run. "Report it and carry on" is not defensible in a code whose whole
    // claim is agreeing with Geant4 to fractions of a sigma: a quietly-low dose is worse than
    // no dose, because someone will use it. Until today the only thing that looked at this at
    // all was example B1's RunAction - g4dose, the viewer, the builder and every generated
    // project ignored it, and would have reported a short answer as a result.
    //
    // AllowDroppedTracks() exists for the case where somebody genuinely wants best-effort
    // transport and has decided the shortfall is acceptable. It has to be asked for.
    if (st.overflow > 0) {
      std::printf("\n*** %lld TRACKS WERE DROPPED ***\n\n"
                  "    A species buffer had no room, so those particles stopped being\n"
                  "    transported. Their energy was never deposited, so the dose from this\n"
                  "    run is TOO LOW - it is not a measurement.\n\n"
                  "    Raise SetLiveTracksPerEvent (currently %.1f per event) or lower the\n"
                  "    batch. Call AllowDroppedTracks(true) if a short answer is genuinely\n"
                  "    what you want.\n\n",
                  st.overflow, live_per_event_);
      if (!allow_dropped_) { std::exit(3); }
    }
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    return st;
  }



template <typename real_t, typename StepHook>
void TransportEngine<real_t, StepHook>::Free() {
    for (int i = 0; i < 2; ++i) {
      tracks_[i].free_all();
      for (int sp = 0; sp < kNumTrackSpecies; ++sp) {
        cudaFree(idx_[i][sp]);
        idx_[i][sp] = nullptr;
      }
      cudaFree(d_idx_[i]);
      cudaFree(idx_n_[i]);
      d_idx_[i] = nullptr;
      idx_n_[i] = nullptr;
    }
    cudaFree(d_unknown_);
    d_unknown_ = nullptr;
    cudaFree(d_vols_);
    cudaFree(d_mats_);
    cudaFree(d_rt_);
    cudaFree(d_score_);
    cudaFree(d_status_warn_);
    d_status_warn_ = nullptr;
    cudaFree(d_track_arena_);
    d_track_arena_ = nullptr;
    cudaFree(d_dropped_);
    cudaFree(d_dropped_count_);
    d_dropped_ = nullptr;
    d_dropped_count_ = nullptr;
    cudaFree(sec_.entry);
    cudaFree(sec_.prev);
    cudaFree(sec_.cursor);
    cudaFree(sec_.overflow);
    sec_ = SecondaryArena{};
    cudaFree(d_primaries_);
    d_primaries_ = nullptr;
    cudaFree(d_voxel_score_);
    d_voxel_score_ = nullptr;
    n_voxel_cells_ = 0;
    cudaFree(d_pool_solids_);
    cudaFree(d_pool_xforms_);
    cudaFree(d_pool_aux_);
    cudaFree(d_voxels_);
    cudaFree(d_tri_);
    cudaFree(d_bvh_);
  }

}  // namespace g4gpu::host
