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
/// Where in the seed's random stream a batch starts: the run's own offset plus the batch's.
///
/// Wraps at 2^32, which is 4.3e9 primaries into a process's life and is the same wrap the key
/// arithmetic has always had. Folded into one function because a repair pass has to use the
/// same base as the attempt it repairs - re-transporting an event with a different key would
/// give it a different shower, and the dose would depend on whether a batch had to be
/// repaired.
inline unsigned int key_base_for(long long stream_pos, int batch_base) {
  return static_cast<unsigned int>(static_cast<unsigned long long>(stream_pos)
                                   + static_cast<unsigned long long>(batch_base));
}

template <typename real_t>
__global__ void seed_from_primaries(TrackBuffer<real_t> pool,
                                    geom::Geometry<real_t> geometry,
                                    const Primary<real_t>* prim, int n, unsigned int key_base,
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

  // The per-track RNG key comes from the run's seed and the primary's position in the random
  // stream, so the shower a primary produces is reproducible for a given seed even though the
  // primary itself was generated on the host from Geant4's engine.
  //
  // key_base is a position in the WHOLE stream, not in this batch and not in this run. Two
  // things fold into it and both were bugs before they were folded in:
  //
  //   the batch offset   With the batch-local index every batch replayed the same set of
  //                      streams - event 0 of the second batch drew exactly what event 0 of
  //                      the first had - so a 2M-event run in two batches was two correlated
  //                      halves rather than 2M independent showers. Unseen because the batch
  //                      was a fixed 1048576 and nothing varied it; automatic sizing changed
  //                      the batch and B1 moved from 0.019 to 0.450 sigma against Geant4.
  //
  //   the run offset     Without it, the second BeamOn of a process drew the same streams as
  //                      the first. B1 still differed between the two, because its gun draws
  //                      two host random numbers per event and the host engine carries on -
  //                      so the primaries differed while every shower repeated. A generator
  //                      that draws nothing repeated exactly. See G4RunManager::stream_pos_.
  //
  // Note `event[slot] = i` below: the SCORING index stays batch-local, because it indexes the
  // per-event score array. Only the key is global. Conflating the two is what made the first
  // of those two bugs invisible.
  const unsigned int key = seed ^ (key_base + static_cast<unsigned int>(i));
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
///
/// TWO PASSES, NOT ONE, and one array instead of one per species. The previous version wrote
/// straight into `lists[sp]`, an array per species each sized at the whole pool - so naming a
/// sixteenth species cost `4 * pool` bytes a side whether any run ever produced one. A counting
/// sort pays four bytes a slot for any number of species: histogram here, prefix sum on the
/// host (it already reads the counts back to size the launches), scatter below.
///
/// The scatter's offsets travel as a by-value argument rather than a third device buffer, which
/// is why SpeciesOffsets exists: sixteen ints is 64 bytes against a 4 KB kernel parameter
/// space, and a buffer would need its own allocation, upload and lifetime.
template <typename real_t>
__global__ void count_species(TrackBuffer<real_t> pool, int n, int* counts, int* unknown) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) { return; }
  const int sp = species_index(static_cast<ParticleType>(pool.species[i]));
  if (sp < 0) {
    // A tripwire, not an accounting line. BufferEmitter::push refuses a species with no kernel
    // before it can be appended, and BeamOn refuses one as a primary, so a track reaching here
    // got past both - which is a hole in the guards rather than a particle to be counted.
    atomicAdd(unknown, 1);
    return;
  }
  atomicAdd(&counts[sp], 1);
}

/// Start of each species' contiguous range in the one index list.
struct SpeciesOffsets {
  int base[kNumTrackSpecies];
};

/// Writes each track's pool slot into its own species' range.
///
/// `cursors` is the same allocation the histogram used, zeroed again - it is a per-species
/// bump counter now rather than a count. Reusing it is not a saving worth making on its own;
/// what it avoids is a second `kNumTrackSpecies`-sized buffer whose only distinguishing
/// feature would be which of the two passes wrote it.
template <typename real_t>
__global__ void scatter_species(TrackBuffer<real_t> pool, int n, int* list,
                                SpeciesOffsets off, int* cursors) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) { return; }
  const int sp = species_index(static_cast<ParticleType>(pool.species[i]));
  if (sp < 0) { return; }  // already counted by count_species
  list[off.base[sp] + atomicAdd(&cursors[sp], 1)] = i;
}

template <typename real_t, typename StepHook>
__global__ void run_step_gamma(Scene<real_t> scene, TrackBuffer<real_t> in, const int* idx,
                               TrackBuffer<real_t> out, int n, int batch, double* score,
                               double* voxel_score,
                               int n_step, vis::TrajectoryBuffer traj, int* status_warn,
                               SecondaryArena sec, EmitterBooks books, StepHook hook) {
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
  // kSpecies is here rather than below the emitter because the emitter needs the mass: a
  // secondary is born at the POST-step point's time, which is `parent_time + length/velocity`
  // on the PRE-step velocity, and push() computes it from parent_velocity. See
  // BufferEmitter::parent_velocity for the Geant4 sources that say so.
  constexpr ParticleType kSpecies = ParticleType::kGamma;
  BufferEmitter<real_t> em{out, p.pos, p.volume, p.event, p.rng_key, p.step,
                           0u, p.global_time, p.weight,
                           TrackState<real_t>::pre_step_velocity(
                               p.ekin, particle_def<real_t>(kSpecies).mass),
                           sec, -1, &srep, books};

  const real_t ekin_pre = p.ekin;
  const Vec3<real_t> pos_pre = p.pos;
  const Vec3<real_t> dir_pre = p.dir;
  const int volume_pre = p.volume;
  const bool first_in_vol = (p.flags & kFirstStepInVolume) != 0u;

  real_t edep = 0;
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
                                SecondaryArena sec, EmitterBooks books, StepHook hook) {
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
  // See run_step_gamma: the emitter needs the species' mass, for the POST-step birth time.
  constexpr ParticleType kSpecies =
      kIsPositron ? ParticleType::kPositron : ParticleType::kElectron;
  BufferEmitter<real_t> em{out, p.pos, p.volume, p.event, p.rng_key, p.step,
                           0u, p.global_time, p.weight,
                           TrackState<real_t>::pre_step_velocity(
                               p.ekin, particle_def<real_t>(kSpecies).mass),
                           sec, -1, &srep, books};

  const real_t ekin_pre = p.ekin;
  const Vec3<real_t> pos_pre = p.pos;
  const Vec3<real_t> dir_pre = p.dir;
  const int volume_pre = p.volume;
  const bool first_in_vol = (p.flags & kFirstStepInVolume) != 0u;

  real_t edep = 0;
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
                                had::HadronicWiring<real_t> had,
                                vis::TrajectoryBuffer traj, int* status_warn,
                                SecondaryArena sec, EmitterBooks books, StepHook hook) {
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
  // See run_step_gamma: the emitter needs the species' mass, for the POST-step birth time.
  constexpr ParticleType kSpecies = kType;
  BufferEmitter<real_t> em{out, p.pos, p.volume, p.event, p.rng_key, p.step,
                           0u, p.global_time, p.weight,
                           TrackState<real_t>::pre_step_velocity(
                               p.ekin, particle_def<real_t>(kSpecies).mass),
                           sec, -1, &srep, books};

  const real_t ekin_pre = p.ekin;
  const Vec3<real_t> pos_pre = p.pos;
  const Vec3<real_t> dir_pre = p.dir;
  const int volume_pre = p.volume;
  const bool first_in_vol = (p.flags & kFirstStepInVolume) != 0u;

  real_t edep = 0;
  const bool alive = step_hadron(scene, p, kType, had, rng, em, edep, srep, traj);
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

/// One step of one neutral hadron - a neutron or a pi0.
///
/// The same frame as the three kernels above, and it is worth saying why it is a fourth kernel
/// rather than a branch inside run_step_hadron: step_neutral shares no code with step_hadron at
/// all. No range table, no step function, no fluctuation, no MSC, no delta ray. A warp holding
/// both would run the union of the two and pay for the half its threads skipped, which is the
/// same argument that made one kernel per species the rule in the first place.
///
/// @param neutron_xs the combined cross-section table, or null. Passed as a parameter rather
///        than carried on the Scene deliberately: it makes the one call site that can turn the
///        neutron's physics on visible in this file, and it keeps the field out of Scene until
///        P8 - which owns the wiring - decides it wants it there.
/// @param killed_energy,killed_n where the neutron time cut's discarded energy is booked. It
///        is not a deposit and not an escape, so it cannot ride on `edep` or on the emitter's
///        ledger; it is a third kind of loss and it gets its own two words. See step_neutral.
template <typename real_t, ParticleType kType, typename StepHook>
__global__ void run_step_neutral(Scene<real_t> scene, TrackBuffer<real_t> in, const int* idx,
                                 TrackBuffer<real_t> out, int n, int batch, double* score,
                                 double* voxel_score, int n_step,
                                 const had::NeutronGeneralXs<real_t>* neutron_xs,
                                 had::HadronicWiring<real_t> had,
                                 double* killed_energy, int* killed_n,
                                 vis::TrajectoryBuffer traj, int* status_warn,
                                 SecondaryArena sec, EmitterBooks books, StepHook hook) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) { return; }
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

  // A fourth purpose value, so a neutral hadron's stream is independent of the gamma, lepton
  // and charged-hadron streams a track of the same key would have drawn.
  StepReport<real_t> srep;
  Philox<real_t> rng(p.rng_key, p.step, 0x4E7Au);
  // See run_step_gamma: the emitter needs the species' mass, for the POST-step birth time.
  constexpr ParticleType kSpecies = kType;
  BufferEmitter<real_t> em{out, p.pos, p.volume, p.event, p.rng_key, p.step,
                           0u, p.global_time, p.weight,
                           TrackState<real_t>::pre_step_velocity(
                               p.ekin, particle_def<real_t>(kSpecies).mass),
                           sec, -1, &srep, books};

  const real_t ekin_pre = p.ekin;
  const Vec3<real_t> pos_pre = p.pos;
  const Vec3<real_t> dir_pre = p.dir;
  const int volume_pre = p.volume;
  const bool first_in_vol = (p.flags & kFirstStepInVolume) != 0u;

  real_t edep = 0;
  const bool alive = step_neutral(scene, p, kType, neutron_xs, had, rng, em, edep, srep, traj);
  ++p.step;
  // The clocks, from the PRE-step energy: see TrackState::advance. This is the one that decides
  // whether the neutron time cut ever fires, so it is load-bearing here in a way it is not for
  // a charged track - a neutron thermalising in a shield takes microseconds of flight to cross
  // millimetres, and 10 us is reached by the clock and not by the geometry.
  p.advance(srep.true_length, ekin_pre, particle_def<real_t>(kSpecies).mass);
  p.flags = (srep.status == StepStatus::fGeomBoundary) ? (p.flags | kFirstStepInVolume)
                                                       : (p.flags & ~kFirstStepInVolume);
  // The time cut discards the kinetic energy rather than depositing it, so it is booked here
  // where the pre-step energy is still in hand. Recorded before the hook, so a stepping action
  // that reads the step sees a consistent set of numbers.
  if (srep.process == ProcessId::fNeutronKiller) {
    if (killed_energy != nullptr) { atomicAdd(killed_energy, static_cast<double>(ekin_pre)); }
    if (killed_n != nullptr) { atomicAdd(killed_n, 1); }
  }
  if (edep != real_t(0) && slot >= 0) {
    atomicAdd(&score[static_cast<size_t>(slot) * batch + p.event], static_cast<double>(edep));
    if (vcell >= 0) { atomicAdd(&voxel_score[vcell], static_cast<double>(edep)); }
  }
  const bool escaped = (p.volume == geom::kOutsideWorld);
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
  ds.n_secondaries = static_cast<int>(em.child_count);
  ds.sec_arena = sec;
  ds.sec_pool = &out;
  ds.sec_last = em.last_secondary;
  ds.status = (escaped && srep.status == StepStatus::fGeomBoundary) ? StepStatus::fWorldBoundary
                                                                   : srep.status;
  ds.process = srep.process;
  hook(ds);
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

    // PER-CLASS LAYERS, and only when some class asked for one.
    //
    // The class array is another short per cell - 268 MB for a 512^3 phantom, the same as the
    // materials - so a batch run whose classes all sit on one layer must not be made to carry
    // it. `any_class_layers` is the scene saying whether the navigator will ever look.
    // Either reason: a class can be absent without any class naming a layer, and the
    // navigator still has to read the per-cell class array to know which cells are gone.
    if ((scene.pool.any_class_layers || scene.pool.any_class_absent)
        && scene.pool.voxel_class_cells.size() == scene.pool.voxel_cells.size()
        && !scene.pool.voxel_class_layer.empty()) {
      G4GPU_CUDA_CHECK(cudaMalloc(&d_voxel_class_,
                                  sizeof(short) * scene.pool.voxel_class_cells.size()));
      G4GPU_CUDA_CHECK(cudaMemcpy(d_voxel_class_, scene.pool.voxel_class_cells.data(),
                                  sizeof(short) * scene.pool.voxel_class_cells.size(),
                                  cudaMemcpyHostToDevice));
      G4GPU_CUDA_CHECK(cudaMalloc(&d_class_layer_,
                                  sizeof(int) * scene.pool.voxel_class_layer.size()));
      G4GPU_CUDA_CHECK(cudaMemcpy(d_class_layer_, scene.pool.voxel_class_layer.data(),
                                  sizeof(int) * scene.pool.voxel_class_layer.size(),
                                  cudaMemcpyHostToDevice));
      g.voxels.cls = d_voxel_class_;
      g.voxels.class_layer = d_class_layer_;
      // One byte per CLASS, not per cell, so this is bounded by the class cap however large
      // the phantom is - and it is uploaded only when some class is actually absent.
      if (scene.pool.any_class_absent
          && scene.pool.voxel_class_absent.size() == scene.pool.voxel_class_layer.size()) {
        G4GPU_CUDA_CHECK(cudaMalloc(&d_class_absent_,
                                    scene.pool.voxel_class_absent.size()));
        G4GPU_CUDA_CHECK(cudaMemcpy(d_class_absent_, scene.pool.voxel_class_absent.data(),
                                    scene.pool.voxel_class_absent.size(),
                                    cudaMemcpyHostToDevice));
        g.voxels.class_absent = d_class_absent_;
      }
    }
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

    // hadElastic's device tables, beside the hadron range table and for the same reason: a run
    // that will carry a charged hadron needs them before the first primary is seeded. The
    // element list is the scene's, so the per-(pion, Z) G4ElasticData is built for the elements
    // the detector is actually made of rather than for all 92. See host/hadronic_upload.cuh.
    elastic_tables_ = upload_elastic_tables<real_t>(zs);

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
        // The dispatch list: ONE array of pool_ ints, bucketed by species each iteration.
        // Sized at the whole pool because the buckets together are the whole pool - which is
        // the point, and is what one array per species could not say: each of those had to be
        // sized at the pool on its own, because any single species may in principle be all of
        // it, so N species cost N times the pool to describe pool-many tracks. Four bytes a
        // slot here, 1.7% on top of a 236-byte track, independent of how many species exist.
        G4GPU_CUDA_CHECK(cudaMalloc(&idx_[i], sizeof(int) * pool_));
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

    // ---- the three ledgers for energy and particles that do not become tracks.
    //
    // Allocated unconditionally. `carried_away` is batch_ doubles, the same shape as one row
    // of the score array, so it is not the allocation worth making conditional; the other two
    // are a few dozen words. What would be worth avoiding is a NULL that means "the accounting
    // is off" being indistinguishable from a zero that means "nothing happened", which is why
    // these exist at all - see RunStats.
    G4GPU_CUDA_CHECK(cudaMalloc(&d_carried_away_, sizeof(double) * batch_));
    G4GPU_CUDA_CHECK(cudaMalloc(&d_carried_n_,
                                sizeof(int) * static_cast<int>(ParticleType::kNumTypes)));
    G4GPU_CUDA_CHECK(cudaMalloc(&d_refused_,
                                sizeof(int) * static_cast<int>(ParticleType::kNumTypes)));
    G4GPU_CUDA_CHECK(cudaMalloc(&d_refused_e_,
                                sizeof(double) * static_cast<int>(ParticleType::kNumTypes)));
    {
      const int nr = static_cast<int>(had::HadronicRefusal::kNumHadronicRefusals);
      G4GPU_CUDA_CHECK(cudaMalloc(&d_had_refused_n_, sizeof(int) * nr));
      G4GPU_CUDA_CHECK(cudaMalloc(&d_had_refused_e_, sizeof(double) * nr));
    }
    G4GPU_CUDA_CHECK(cudaMalloc(&d_killed_energy_, sizeof(double)));
    G4GPU_CUDA_CHECK(cudaMalloc(&d_killed_n_, sizeof(int)));

    // ---- a neutron cross section without a final state is not an allowed state.
    //
    // Refused here rather than discovered on the device. If a table is ever uploaded without
    // the samplers to go with it, every neutron would draw a finite interaction length, arrive
    // at a branch with nothing to do, and be killed with its energy dumped locally - a dose
    // that looks like hadronic transport and is a deposit at the first interaction point. The
    // device cannot report that usefully at kernel rates, so the combination is refused where
    // it is set up. Today d_neutron_xs_ is always null and this never fires, which is exactly
    // the state the message describes; it fires the first time half of P8 lands.
    if (d_neutron_xs_ != nullptr) {
      std::printf(
          "\nFATAL: a neutron cross-section table is present but no final states are wired.\n"
          "  step_neutral would sample an interaction length, reach the sub-process branch\n"
          "  with nothing to apply, and kill the neutron with its energy deposited at that\n"
          "  point - which is a plausible-looking dose for physics that did not run.\n"
          "  See physics/hadronic/neutron_general_xs.cuh for the contract: the table and the\n"
          "  final states land together.\n");
      std::exit(2);
    }

    // The secondary arena. One entry per secondary created in a single kernel launch, so it is
    // sized against how many tracks can be in flight rather than against the length of a run.
    //
    // THE DEFAULT IS NOW THE POOL, not half the batch. Half the batch was a measurement of B1 -
    // "about 0.3 secondaries per step" - and a measurement of one physics list is a poor bound
    // for the next one: a hadronic inelastic reaction at a few GeV emits tens of secondaries in
    // one step, and 0.3 per step is not a fact about transport, it is a fact about 6 MeV gammas
    // in water.
    //
    // The pool is a bound rather than an estimate, and it comes from arithmetic that is already
    // in this file. The throttle in BeamOn steps only as many tracks as the output can hold
    // together with their reserved secondaries, so the secondaries one launch can create is at
    // most `cap_out - n_live`, which is at most the pool. An arena of pool_ entries therefore
    // cannot overflow while the output does not - which is the property worth having, because
    // the output's overflow ends the run and the arena's only degrades
    // GetSecondaryInCurrentStep(). Before this the arena was the smaller of the two and could
    // fail quietly first.
    //
    // It costs 8 bytes an entry against a track slot's 236, so pool_ entries is 3.4% on top of
    // the track arena - measured by track_bytes_per_slot, not guessed.
    sec_.capacity = (sec_capacity_ > 0) ? sec_capacity_ : static_cast<int>(pool_);
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
                                         vis::TrajectoryBuffer traj, EventSink* sink,
                                         long long stream_pos) {
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
    // Per run, not per batch: these are counts and totals over the whole run, unlike the
    // per-event carried-away array which is batch-local and zeroed with the scores.
    const size_t kTypeBytes = sizeof(int) * static_cast<size_t>(ParticleType::kNumTypes);
    if (d_carried_n_ != nullptr) {
      G4GPU_CUDA_CHECK(cudaMemset(d_carried_n_, 0, kTypeBytes));
    }
    if (d_refused_ != nullptr) { G4GPU_CUDA_CHECK(cudaMemset(d_refused_, 0, kTypeBytes)); }
    if (d_refused_e_ != nullptr) {
      G4GPU_CUDA_CHECK(cudaMemset(d_refused_e_, 0,
                                  sizeof(double) * static_cast<size_t>(ParticleType::kNumTypes)));
    }
    {
      const size_t nr = static_cast<size_t>(had::HadronicRefusal::kNumHadronicRefusals);
      if (d_had_refused_n_ != nullptr) {
        G4GPU_CUDA_CHECK(cudaMemset(d_had_refused_n_, 0, sizeof(int) * nr));
      }
      if (d_had_refused_e_ != nullptr) {
        G4GPU_CUDA_CHECK(cudaMemset(d_had_refused_e_, 0, sizeof(double) * nr));
      }
    }
    if (d_killed_energy_ != nullptr) {
      G4GPU_CUDA_CHECK(cudaMemset(d_killed_energy_, 0, sizeof(double)));
    }
    if (d_killed_n_ != nullptr) { G4GPU_CUDA_CHECK(cudaMemset(d_killed_n_, 0, sizeof(int))); }
    if (primaries == nullptr || n_events <= 0) { return st; }

    // Every primary is checked against what a kernel can step. A primary with no kernel used
    // to land in the gamma buffer and be transported as a gamma - a run that produced a
    // plausible dose for physics that never happened - and the guard that replaced that was a
    // hand-written list of five species. It is species_disposition now, so the list cannot
    // drift from the set of kernels the way prose does.
    //
    // Checked over every primary rather than once for the source, because a generator is free
    // to fire a different species per event and the guard has to cover what it actually
    // produced. It is one switch per event against a whole shower.
    for (int i = 0; i < n_events; ++i) {
      const ParticleType p = primaries[i].particle;
      const SpeciesDisposition disp = species_disposition(p);
      if (disp == SpeciesDisposition::kStepped) {
        // Every charged hadron is stepped from the hadron dE/dx and range tables. A scene
        // uploaded without them (b1_gpu_sched builds a gamma-only Scene deliberately) would
        // step the primary once, find a null table, and drop it. That is a silent zero, so it
        // is refused here.
        //
        // `is_heavy_charged` and not "everything but the three EM species". A neutron and a
        // pi0 are stepped by step_neutral, which has no continuous loss and therefore never
        // reads the range table - so refusing them for want of it would refuse the one run
        // that is worth making today: a neutron primary with no cross section, streaming to
        // the world boundary. The predicate names the species that read the table rather than
        // the species that do not.
        if (scene_.hadron_range == nullptr && is_heavy_charged(p)) {
          std::printf(
              "\nFATAL: a \"%s\" primary was given to a Scene with no hadron range table\n"
              "  (event %d). step_hadron cannot advance a track without one, and dropping it\n"
              "  would look like a run that simply deposited nothing.\n",
              particle_name(p), i);
          std::exit(2);
        }
        continue;
      }
      if (disp == SpeciesDisposition::kCounted) {
        // A neutrino primary is not an error and not transportable either: its whole energy
        // leaves the event at the vertex. Refused rather than silently booked as escaped,
        // because a run of neutrino primaries would report a dose of exactly zero and that is
        // a number somebody would take at face value. QBBC would transport it - across the
        // world in one step, depositing nothing - so the answer is the same and saying so is
        // better than producing it.
        std::printf(
            "\nFATAL: \"%s\" is not a primary this transport will fire (event %d).\n"
            "  A neutrino has no process in QBBC, so every event would deposit exactly zero\n"
            "  and the run would look like a result. Its energy is booked as carried away\n"
            "  when a decay makes one; as a primary there would be nothing else in the event.\n",
            particle_name(p), i);
        std::exit(2);
      }
      std::printf(
          "\nFATAL: \"%s\" (species %d) is not transported (event %d).\n"
          "  See G4RunManager::CheckSpecies for the current species set and what each\n"
          "  missing one needs. Refusing rather than transporting it as something it is not.\n",
          particle_name(p), static_cast<int>(p), i);
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
      // Zeroed with the scores and for the same reason: it is indexed by the BATCH-local event
      // id, so a batch that reused a slot would add this batch's neutrinos to the last one's.
      if (d_carried_away_ != nullptr) {
        G4GPU_CUDA_CHECK(cudaMemset(d_carried_away_, 0, sizeof(double) * batch_));
      }
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
          tracks_[cur].view, geom_, d_primaries_, n_batch, key_base_for(stream_pos, base),
          seed);
      G4GPU_CUDA_CHECK(cudaGetLastError());

      // Bounded so a non-terminating track shows up as an explicit abandonment rather than a
      // hang. B1 drains in about 110 iterations.
      constexpr int kMaxIterations = 1000;
      int leftover = 0;
      for (int iteration = 0; iteration < kMaxIterations; ++iteration) {
        // How many tracks are alive, of any species. One number, because there is one pool.
        const int n_live = tracks_[cur].count();

        // The dispatch layout for this iteration: which slots hold gammas, which electrons,
        // and so on. Rebuilt every time because the pool is written in append order, which is
        // whatever the atomics gave out, and a kernel needs its own species contiguous.
        //
        // Two passes and a prefix sum - a counting sort into the one index array. The counts
        // have to come back to the host anyway to size the launches, so the sum costs nothing
        // extra beyond the second launch.
        int nsp[kNumTrackSpecies] = {};
        SpeciesOffsets off{};
        if (n_live > 0) {
          const int blocks = (n_live + threads_ - 1) / threads_;
          G4GPU_CUDA_CHECK(cudaMemset(idx_n_[cur], 0, sizeof(int) * kNumTrackSpecies));
          count_species<real_t><<<blocks, threads_>>>(tracks_[cur].view, n_live, idx_n_[cur],
                                                      d_unknown_);
          G4GPU_CUDA_CHECK(cudaGetLastError());
          G4GPU_CUDA_CHECK(cudaMemcpy(nsp, idx_n_[cur], sizeof(int) * kNumTrackSpecies,
                                      cudaMemcpyDeviceToHost));
          int at = 0;
          for (int sp = 0; sp < kNumTrackSpecies; ++sp) {
            off.base[sp] = at;
            at += nsp[sp];
          }
          // Zeroed again and reused as the scatter cursors; see scatter_species.
          G4GPU_CUDA_CHECK(cudaMemset(idx_n_[cur], 0, sizeof(int) * kNumTrackSpecies));
          scatter_species<real_t><<<blocks, threads_>>>(tracks_[cur].view, n_live, idx_[cur],
                                                        off, idx_n_[cur]);
          G4GPU_CUDA_CHECK(cudaGetLastError());
        }
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
            // The carried-away ledger is per event exactly as the score is, so a repaired
            // event's neutrinos have to be forgotten alongside its deposit. Leaving it would
            // double-count every neutrino of every repaired event - an error that shows up
            // only as an energy balance that does not close, which is the last place anyone
            // looks.
            if (d_carried_away_ != nullptr) {
              G4GPU_CUDA_CHECK(cudaMemset(d_carried_away_ + e, 0, sizeof(double)));
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
              tracks_[cur].view, geom_, d_primaries_, n_repair,
              key_base_for(stream_pos, base), seed, d_dropped_);
          G4GPU_CUDA_CHECK(cudaGetLastError());
          continue;  // drain again, this time carrying only the repaired events
        }
        if (iteration == kMaxIterations - 1) { leftover = n_live; }

        for (int sp = 0; sp < kNumTrackSpecies; ++sp) {
          st.peak_live[sp] = std::max(st.peak_live[sp], nsp[sp]);
        }

        // How many tracks of each species may be stepped this iteration.
        //
        // Nothing is re-divided any more. There is one output pool, so there is one budget:
        // every live track needs a slot in it whether it is stepped or merely passed through,
        // and what is left over after that is what pays for secondaries.
        //
        // The fan-out reservation is PER SPECIES - max_secondaries_per_step in
        // core/track_buffer.cuh - and the budget below is therefore counted in SLOTS rather
        // than in tracks. It was one constant of four for every species, which was correct
        // while every species could make at most three secondaries in a step. A hadronic
        // inelastic reaction makes tens, and raising a shared constant to cover it would
        // divide the gamma's budget by the same factor: on B1's own numbers, ~2 spare slots an
        // event at a reservation of 4 is 0.5 tracks an event, and at 64 it is 0.03 - sixty
        // times the iterations to drain a gamma run that will never make a neutron. Every
        // reservation is still 4 today, so this loop chooses exactly what the old one did.
        //
        // A species that cannot be stepped in full this iteration is not truncated: the
        // remainder passes through untouched and is stepped next time.
        const int cap_out = tracks_[nxt].view.capacity;
        int slots = (cap_out > n_live) ? (cap_out - n_live) : 0;
        int step_n[kNumTrackSpecies] = {};
        int total_step = 0;
        for (int sp = 0; sp < kNumTrackSpecies; ++sp) {
          const int per = max_secondaries_per_step(sp);
          const int can = (per > 0) ? (slots / per) : nsp[sp];
          step_n[sp] = std::min(nsp[sp], can);
          slots -= step_n[sp] * per;
          total_step += step_n[sp];
        }
        if (total_step == 0) {
          std::printf("\nFATAL: no track can be stepped without overrunning the pool.\n"
                      "       %d tracks are live and the pool holds %d slots a side. Raise\n"
                      "       SetLiveTracksPerEvent (now %.1f per event) or lower the batch.\n",
                      n_live, cap_out, live_per_event_);
          std::exit(3);
        }
        st.throttled += n_live - total_step;
        st.track_steps += total_step;

        tracks_[nxt].reset();
        // The secondary arena holds one iteration worth of chains and no more: the chains
        // a stepping action walks are the ones made by the launch it is running in, and
        // nothing reads them afterwards. Resetting here is what keeps the arena sized
        // against tracks in flight rather than against the length of the run.
        if (sec_.cursor != nullptr) {
          G4GPU_CUDA_CHECK(cudaMemsetAsync(sec_.cursor, 0, sizeof(int)));
        }
        // One launch per species, each over its own range of the one index list, each into the
        // one output pool.
        //
        // A specialised kernel per species remains because a warp whose threads take different
        // physics paths serialises through all of them - that is the reason the species list
        // survives at all. What has gone is the idea that a specialised KERNEL needs a separate
        // BUFFER, and now also the idea that it needs a separate index ARRAY.
        //
        // The switch is written out rather than driven from a table of function pointers
        // because a __global__ template's address is not something a host table can hold
        // portably, and because each line names the specialisation it launches - which is what
        // makes a missing species a compile error in the switch rather than a track that is
        // counted and never stepped.
        const EmitterBooks books{d_carried_away_, d_carried_n_, d_refused_, d_refused_e_};
        // P8's wiring, built here and passed by value into every hadronic launch. One struct
        // rather than five arguments, and built per launch rather than at Upload so that the
        // stage can change between two BeamOn calls in one process - which is what
        // ref/b1hadron/stage1_compare.ps1 needs of it.
        had::HadronicWiring<real_t> had_wiring{};
        had_wiring.stage = had_stage_;
        had_wiring.decay = had_decay_;
        had_wiring.hadron_elastic = had_elastic_;
        had_wiring.neutron_capture = had_capture_;
        // Five device pointers, copied by value into the launch like the rest of the struct.
        // Null in a run whose scene can see no hadron - Upload only builds them when it builds
        // the hadron range table - and `elastic_xs_per_volume` then returns zero, which is the
        // same "no process" state a lepton is in.
        had_wiring.elastic = elastic_tables_.view;
        had_wiring.books.count = d_had_refused_n_;
        had_wiring.books.energy = d_had_refused_e_;
        for (int sp = 0; sp < kNumTrackSpecies; ++sp) {
          const int n_sp = nsp[sp];
          if (n_sp <= 0) { continue; }
          const int* list = idx_[cur] + off.base[sp];
          const int blocks = (n_sp + threads_ - 1) / threads_;
#define G4GPU_LAUNCH_HADRON(TYPE)                                                            \
  run_step_hadron<real_t, TYPE><<<blocks, threads_>>>(                                       \
      scene_, tracks_[cur].view, list, tracks_[nxt].view, n_sp, batch_, d_score_,             \
      d_voxel_score_, step_n[sp], had_wiring, traj, d_status_warn_, sec_, books, hook_)
#define G4GPU_LAUNCH_NEUTRAL(TYPE)                                                           \
  run_step_neutral<real_t, TYPE><<<blocks, threads_>>>(                                      \
      scene_, tracks_[cur].view, list, tracks_[nxt].view, n_sp, batch_, d_score_,             \
      d_voxel_score_, step_n[sp], d_neutron_xs_, had_wiring, d_killed_energy_, d_killed_n_,   \
      traj, d_status_warn_, sec_, books, hook_)
          switch (sp) {
            case kSpeciesGamma:
              run_step_gamma<real_t><<<blocks, threads_>>>(
                  scene_, tracks_[cur].view, list, tracks_[nxt].view, n_sp, batch_, d_score_,
                  d_voxel_score_, step_n[sp], traj, d_status_warn_, sec_, books, hook_);
              break;
            case kSpeciesElectron:
              run_step_lepton<real_t, false><<<blocks, threads_>>>(
                  scene_, tracks_[cur].view, list, tracks_[nxt].view, n_sp, batch_, d_score_,
                  d_voxel_score_, step_n[sp], traj, d_status_warn_, sec_, books, hook_);
              break;
            case kSpeciesPositron:
              run_step_lepton<real_t, true><<<blocks, threads_>>>(
                  scene_, tracks_[cur].view, list, tracks_[nxt].view, n_sp, batch_, d_score_,
                  d_voxel_score_, step_n[sp], traj, d_status_warn_, sec_, books, hook_);
              break;
            case kSpeciesProton:     G4GPU_LAUNCH_HADRON(ParticleType::kProton); break;
            case kSpeciesAlpha:      G4GPU_LAUNCH_HADRON(ParticleType::kAlpha); break;
            case kSpeciesMuonMinus:  G4GPU_LAUNCH_HADRON(ParticleType::kMuonMinus); break;
            case kSpeciesMuonPlus:   G4GPU_LAUNCH_HADRON(ParticleType::kMuonPlus); break;
            case kSpeciesPionPlus:   G4GPU_LAUNCH_HADRON(ParticleType::kPionPlus); break;
            case kSpeciesPionMinus:  G4GPU_LAUNCH_HADRON(ParticleType::kPionMinus); break;
            case kSpeciesKaonPlus:   G4GPU_LAUNCH_HADRON(ParticleType::kKaonPlus); break;
            case kSpeciesKaonMinus:  G4GPU_LAUNCH_HADRON(ParticleType::kKaonMinus); break;
            case kSpeciesAntiProton: G4GPU_LAUNCH_HADRON(ParticleType::kAntiProton); break;
            case kSpeciesDeuteron:   G4GPU_LAUNCH_HADRON(ParticleType::kDeuteron); break;
            case kSpeciesTriton:     G4GPU_LAUNCH_HADRON(ParticleType::kTriton); break;
            case kSpeciesNeutron:    G4GPU_LAUNCH_NEUTRAL(ParticleType::kNeutron); break;
            case kSpeciesPiZero:     G4GPU_LAUNCH_NEUTRAL(ParticleType::kPiZero); break;
            default:
              // Not reachable, and not silent if it becomes so: a TrackSpeciesIndex added
              // without a line above would land here, and a species counted by the histogram
              // and never launched is a track that stops being transported.
              std::printf("\nFATAL: species index %d has no stepping kernel, but %d tracks\n"
                          "       were sorted into its range. Add a case to the switch in\n"
                          "       TransportEngine::BeamOn.\n", sp, n_sp);
              std::exit(3);
          }
#undef G4GPU_LAUNCH_HADRON
#undef G4GPU_LAUNCH_NEUTRAL
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
      // The carried-away ledger, brought back with the scores and appended to the run-length
      // vector. Accumulated on the host per event rather than reduced on the device because the
      // per-event number is the one that means anything - see RunStats::carried_away - and the
      // array is already being copied for the batch.
      if (d_carried_away_ != nullptr) {
        if (st.carried_away.empty()) { st.carried_away.assign(n_events, 0.0); }
        std::vector<double> h_nu(static_cast<size_t>(n_batch));
        G4GPU_CUDA_CHECK(cudaMemcpy(h_nu.data(), d_carried_away_, sizeof(double) * n_batch,
                                    cudaMemcpyDeviceToHost));
        for (int e = 0; e < n_batch; ++e) {
          st.carried_away[base + e] = h_nu[e];
          st.carried_away_total += h_nu[e];
        }
      }
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
                    "    dispatched and their energy was never deposited. BufferEmitter::push\n"
                    "    refuses such a species before appending it and BeamOn refuses it as a\n"
                    "    primary, so this is a hole in one of those two guards rather than a\n"
                    "    particle to be accounted for.\n\n", unk);
      }
    }

    // ---- the three ledgers.
    //
    // Read back and reported here, together, because the three of them are one statement about
    // the run: of the energy that did not reach a scorer, this much left as neutrinos, this
    // much was deleted by the neutron time cut, and these particles were not transported at
    // all. Reported only when non-zero, so a gamma run's output is unchanged.
    {
      const int kNT = static_cast<int>(ParticleType::kNumTypes);
      std::vector<int> n_carried(kNT, 0), n_refused(kNT, 0);
      if (d_carried_n_ != nullptr) {
        G4GPU_CUDA_CHECK(cudaMemcpy(n_carried.data(), d_carried_n_, sizeof(int) * kNT,
                                    cudaMemcpyDeviceToHost));
      }
      if (d_refused_ != nullptr) {
        G4GPU_CUDA_CHECK(cudaMemcpy(n_refused.data(), d_refused_, sizeof(int) * kNT,
                                    cudaMemcpyDeviceToHost));
      }
      std::vector<double> e_refused(kNT, 0.0);
      if (d_refused_e_ != nullptr) {
        G4GPU_CUDA_CHECK(cudaMemcpy(e_refused.data(), d_refused_e_, sizeof(double) * kNT,
                                    cudaMemcpyDeviceToHost));
      }
      for (int t = 0; t < kNT; ++t) {
        st.carried_by_species[t] = n_carried[t];
        st.carried_away_n += n_carried[t];
        st.refused_by_species[t] = n_refused[t];
        st.refused_total += n_refused[t];
        st.refused_energy_by_species[t] = e_refused[t];
        st.refused_energy_total += e_refused[t];
      }
      if (d_killed_n_ != nullptr) {
        int kn = 0;
        G4GPU_CUDA_CHECK(cudaMemcpy(&kn, d_killed_n_, sizeof(int), cudaMemcpyDeviceToHost));
        st.neutron_killed_n = kn;
      }
      if (d_killed_energy_ != nullptr) {
        G4GPU_CUDA_CHECK(cudaMemcpy(&st.neutron_killed_energy, d_killed_energy_, sizeof(double),
                                    cudaMemcpyDeviceToHost));
      }

      if (st.carried_away_n > 0) {
        std::printf("carried away: %lld particles, %.6g MeV total (not deposited, not"
                    " transported)\n", st.carried_away_n, st.carried_away_total);
        for (int t = 0; t < kNT; ++t) {
          if (n_carried[t] > 0) {
            std::printf("              %-12s %d\n",
                        particle_name(static_cast<ParticleType>(t)), n_carried[t]);
          }
        }
      }
      if (st.neutron_killed_n > 0) {
        // Not a warning. Geant4 does exactly this and also does not conserve energy across it;
        // the number is printed so that a run can be held to an energy balance rather than
        // leaving the shortfall to be found later and blamed on the transport.
        std::printf("neutron time cut: %lld neutrons killed past 10 us, %.6g MeV discarded\n"
                    "                  (G4NeutronGeneralProcess deposits nothing here - see"
                    " step_neutral)\n",
                    st.neutron_killed_n, st.neutron_killed_energy);
      }
      if (st.refused_total > 0) {
        // This one IS a warning, and it names what is missing. A refused secondary is energy
        // the shower had and this port did not carry, so the dose is low by whatever those
        // particles would have deposited - the same failure mode as a dropped track, arrived at
        // for a different reason.
        std::printf("\n*** %lld SECONDARIES OF SPECIES THIS PORT CANNOT TRANSPORT, %.6g MeV"
                    " ***\n\n"
                    "    They were counted at the point a process created them and no track\n"
                    "    was made, so the dose from this run is TOO LOW by whatever they\n"
                    "    would have deposited. The energy column is what the answer is\n"
                    "    missing; the count alone does not say.\n\n",
                    st.refused_total, st.refused_energy_total);
        for (int t = 0; t < kNT; ++t) {
          if (n_refused[t] > 0) {
            std::printf("      %-12s %10lld   %12.6g MeV\n",
                        particle_name(static_cast<ParticleType>(t)),
                        st.refused_by_species[t], st.refused_energy_by_species[t]);
          }
        }
        std::printf("\n");
      }
    }

    // ---- P8's per-process refusals.
    //
    // Separate from the species ledger above and NOT foldable into it: that one is "this
    // transport has no kernel for that particle", this one is "this transport reached that
    // PROCESS and has no final state for it". A stopped pi- is not a refused species - pi- has
    // a kernel and was transported all the way to rest - it is a refused capture, and the
    // energy it costs is a rest mass rather than a kinetic one.
    {
      const int kNR = static_cast<int>(had::HadronicRefusal::kNumHadronicRefusals);
      std::vector<int> hn(kNR, 0);
      std::vector<double> he(kNR, 0.0);
      if (d_had_refused_n_ != nullptr) {
        G4GPU_CUDA_CHECK(cudaMemcpy(hn.data(), d_had_refused_n_, sizeof(int) * kNR,
                                    cudaMemcpyDeviceToHost));
      }
      if (d_had_refused_e_ != nullptr) {
        G4GPU_CUDA_CHECK(cudaMemcpy(he.data(), d_had_refused_e_, sizeof(double) * kNR,
                                    cudaMemcpyDeviceToHost));
      }
      for (int r = 0; r < kNR; ++r) {
        st.had_refused_count[r] = hn[r];
        st.had_refused_energy[r] = he[r];
        st.had_refused_total += hn[r];
      }
      if (st.had_refused_total > 0) {
        std::printf("\n*** %lld HADRONIC INTERACTIONS WITH NO FINAL STATE IN THIS PORT ***\n\n"
                    "    Stage: %s\n"
                    "    The cross section decided each of these happened; the model that\n"
                    "    would have said what came out is not written. The dose is low by the\n"
                    "    energy column, which is what the missing model would have moved.\n\n",
                    st.had_refused_total, had::hadronic_stage_name(had_stage_));
        for (int r = 0; r < kNR; ++r) {
          if (hn[r] > 0) {
            std::printf("      %10lld   %12.6g MeV   %s\n", st.had_refused_count[r],
                        st.had_refused_energy[r],
                        had::hadronic_refusal_name(static_cast<had::HadronicRefusal>(r)));
          }
        }
        std::printf("\n");
      }
    }
    if (sec_.overflow != nullptr) {
      int over = 0;
      G4GPU_CUDA_CHECK(cudaMemcpy(&over, sec_.overflow, sizeof(int), cudaMemcpyDeviceToHost));
      st.secondary_overflow = over;
      // PRINTED, WHICH IT WAS NOT.
      //
      // The count has been recorded on RunStats since the arena was written, and the field's
      // own comment says "Zero in every run of this project's pipeline; if it is not zero,
      // raise the arena" - which is advice to whoever reads the field, and until now the only
      // thing that read it was tests/test_custom_hook.cu. g4dose, example B1, the viewer, the
      // builder and every generated project ignored it, so an arena that overflowed said
      // nothing at all.
      //
      // What overflow costs is narrow and worth stating exactly, because it is NOT a wrong
      // dose: the arena holds one (buffer, slot) pair per secondary so that a stepping action
      // can walk this step's children, and a pair that does not fit only breaks that walk.
      // Every track is still in the pool and every deposit is still scored. So this is a
      // warning about a hook's input, not about the physics - which is precisely why it needs
      // printing rather than exiting, and precisely why it could sit unread: nothing downstream
      // gets worse in a way anyone would notice.
      //
      // The default capacity is now the whole pool rather than half the batch, so overflow
      // needs the output buffer to have overflowed first - and that ends the run. It is
      // reachable only through SetSecondaryArenaCapacity.
      if (over > 0) {
        std::printf("\n*** %d SECONDARY CHAIN ENTRIES DID NOT FIT THE ARENA ***\n\n"
                    "    The arena holds %d entries. Every track was still appended to the\n"
                    "    pool and every deposit was scored, so the DOSE IS UNAFFECTED - what\n"
                    "    is incomplete is DeviceStep::sec_last, the chain a stepping action\n"
                    "    walks to see this step's secondaries. A hook that reads it saw fewer\n"
                    "    children than the step made. Raise it with\n"
                    "    SetSecondaryArenaCapacity(n) - the default is the whole track pool.\n\n",
                    over, sec_.capacity);
      }
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
      std::printf("\n*** %d TRACKS WERE DROPPED ***\n\n"
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
      cudaFree(idx_[i]);
      cudaFree(idx_n_[i]);
      idx_[i] = nullptr;
      idx_n_[i] = nullptr;
    }
    cudaFree(d_unknown_);
    d_unknown_ = nullptr;
    cudaFree(d_carried_away_);
    cudaFree(d_carried_n_);
    cudaFree(d_refused_);
    cudaFree(d_refused_e_);
    cudaFree(d_had_refused_n_);
    cudaFree(d_had_refused_e_);
    cudaFree(d_killed_energy_);
    cudaFree(d_killed_n_);
    d_carried_away_ = nullptr;
    d_carried_n_ = nullptr;
    d_refused_ = nullptr;
    d_refused_e_ = nullptr;
    d_had_refused_n_ = nullptr;
    d_had_refused_e_ = nullptr;
    d_killed_energy_ = nullptr;
    d_killed_n_ = nullptr;
    cudaFree(d_neutron_xs_);
    d_neutron_xs_ = nullptr;
    free_elastic_tables<real_t>(elastic_tables_);
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
    cudaFree(d_voxel_class_);
    cudaFree(d_class_layer_);
    cudaFree(d_class_absent_);
    cudaFree(d_tri_);
    cudaFree(d_bvh_);
  }

}  // namespace g4gpu::host
