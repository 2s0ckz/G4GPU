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
// SINCE P8e THE STOCK HOOK'S KERNELS ARE NOT IN transport_run.cu EITHER. They are one
// translation unit per kernel family - gamma, lepton, neutral, nucleon, muon, meson, ion -
// each holding nothing but the explicit instantiations for its own family, compiled in
// parallel and archived into out/transport_run.lib. What puts them there is the block of
// explicit instantiation DECLARATIONS below the kernels in this file; read it before adding a
// kernel or a species, because a launch without a matching declaration silently goes back to
// being compiled into the engine's own unit. docs/RISK.md V65.
//
// So the rule for including this file is narrow and worth stating plainly:
//
//   * A project that uses the stock hook must NOT include it. Include host/transport_run.cuh
//     and link out/transport_run.lib - which is what every example here does. Its own files
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
#include <new>  // placement new; see `run_interaction`'s note on fill_result's 18 kB return

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

// ---------------------------------------------------------------- P15: binning the queue
//
// `count_species` and `scatter_species` again, over the interaction queue instead of the track
// pool, and for the same two reasons: a kernel can only be launched over a contiguous range,
// and a warp whose threads take different physics paths serialises through all of them. The
// second reason is stronger here than it is for the steppers - "different physics paths" is
// Bertini against FTFP rather than two multiple-scattering models - and so is the first, because
// the kernels are in different translation units and a thread cannot choose between them at all.
//
// Both carry no physics and are templated on `real_t` alone, so they stay in the engine's own
// object like the three utility kernels above.
template <typename real_t>
__global__ void count_interactions(const had::PendingInteraction<real_t>* items, int n,
                                   int* counts) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) { return; }
  atomicAdd(&counts[static_cast<int>(items[i].bucket)], 1);
}

/// Start of each bucket's contiguous range in the one index list.
struct InteractionOffsets {
  int base[static_cast<int>(had::InteractionBucket::kNumInteractionBuckets)];
};

template <typename real_t>
__global__ void scatter_interactions(const had::PendingInteraction<real_t>* items, int n,
                                     int* list, InteractionOffsets off, int* cursors) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) { return; }
  const int b = static_cast<int>(items[i].bucket);
  list[off.base[b] + atomicAdd(&cursors[b], 1)] = i;
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
  // ...AND FOR `kGenericIon` THE SPECIES' MASS IS NOT THE TRACK'S. `particle_def(kGenericIon)`
  // is G4GenericIon's 938.2723 MeV placeholder and the track is a real nuclide; both the
  // pre-step velocity (which times the step, sets every secondary's birth clock and advances
  // the three clocks below) and the proper time need the ion's own mass. The test is on a
  // template parameter, so the branch does not exist in the other fifteen instantiations.
  const real_t track_mass =
      (kType == ParticleType::kGenericIon)
          ? em::ion_particle_def<real_t>(ion_z_of(p.ion_za), ion_a_of(p.ion_za)).mass
          : particle_def<real_t>(kSpecies).mass;
  BufferEmitter<real_t> em{out, p.pos, p.volume, p.event, p.rng_key, p.step,
                           0u, p.global_time, p.weight,
                           TrackState<real_t>::pre_step_velocity(p.ekin, track_mass),
                           sec, -1, &srep, books};

  const real_t ekin_pre = p.ekin;
  const Vec3<real_t> pos_pre = p.pos;
  const Vec3<real_t> dir_pre = p.dir;
  const int volume_pre = p.volume;
  const bool first_in_vol = (p.flags & kFirstStepInVolume) != 0u;

  real_t edep = 0;
  // `queued` is P15's one new exit from a step: the track ended in an interaction this kernel
  // does not run. It goes to `had::InteractionQueue` instead of to `out`, and `run_interaction`
  // finishes the step - including the deposit, the clocks and the ONE hook call this step gets.
  // Returning here rather than appending is what keeps a queued track from being stepped twice.
  bool queued = false;
  const bool alive = step_hadron(scene, p, kType, had, rng, em, edep, srep, traj, &queued);
  if (queued) { return; }
  ++p.step;
  // The clocks, from the PRE-step energy: see TrackState::advance, transcribed from
  // G4Transportation::AlongStepDoIt. Done before the hook so a stepping action reads the
  // time at the end of its own step, as it would in Geant4.
  p.advance(srep.true_length, ekin_pre, track_mass);
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
  // See `run_step_hadron`'s note: a neutron whose general process named `inelastic` leaves this
  // kernel through the queue rather than through `out`.
  bool queued = false;
  const bool alive =
      step_neutral(scene, p, kType, neutron_xs, had, rng, em, edep, srep, traj, &queued);
  if (queued) { return; }
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
                                had::InteractionPool<real_t> pool,
                                vis::TrajectoryBuffer traj, int* status_warn, SecondaryArena sec,
                                EmitterBooks books, StepHook hook) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) { return; }
  const had::PendingInteraction<real_t> q = items[idx[base + i]];
  TrackState<real_t> p = q.track;

  // The voxel cell the step STARTED in, recomputed rather than carried: it is three integer
  // divisions off `(volume_pre, pos_pre)` and the queue entry is already 384 bytes. The same
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
    // PLACEMENT NEW AND NOT AN ASSIGNMENT. `fill_result` returns a
    // `HadronicStepResult<real_t, 256>` BY VALUE, which is about 18 kB; assigning it to
    // `slot->filled` would materialise that temporary on the kernel's own stack, where 18 kB a
    // thread over the resident set is hundreds of megabytes of local memory for a struct that
    // has a home in the pool already. C++17's guaranteed copy elision initialises the prvalue
    // directly into the storage a placement new names, so nothing is copied and nothing is on
    // the stack.
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
    ::new (&slot->filled)
        physics::hadronic::HadronicStepResult<real_t, had::kInteractionSecondaryCap>(
            physics::hadronic::fill_result<real_t, had::kInteractionSecondaryCap,
                                           had::kInteractionSecondaryCap>(
                slot->fs, fill_dir, p.global_time, p.weight,
                /*has_at_rest_processes=*/had::has_at_rest_arm(q.species), slot->pdg_mass));
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
    if (q.kind != had::InteractionKind::kAtRest && q.score_slot >= 0) { edep += p.ekin; }
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


// ---------------------------------------------------------------- where the stock kernels live
//
// ONE TRANSLATION UNIT PER KERNEL FAMILY, AND THIS BLOCK IS WHAT PUTS THEM THERE.
//
// Every kernel above is a __global__ template, and until P8e all twenty specialisations the
// stock hook needs were instantiated implicitly, by the launches in BeamOn, into whichever
// translation unit instantiated the engine class - src/host/transport_run.cu, one file, about
// 24 minutes of nvcc. With the ion's Urban msc dispatched in step_hadron that file stopped
// compiling at all: `ptxas died with status 0xC0000005`, deterministically, and no arrangement
// of __noinline__ got it back (docs/RISK.md V55, V63 - nine builds, one of which compiled).
//
// An explicit instantiation DECLARATION suppresses the implicit instantiation. So the engine's
// translation unit now emits no device code for these kernels at all: it emits a call to a
// host-side launch stub, and the stub and the kernel behind it come from one of the sixteen
// src/host/transport_run_*.cu beside it, compiled in parallel by build_engine.bat and archived
// into out/transport_run.lib.
//
// ONE UNIT PER KERNEL, AND THE GRANULARITY IS A MEASUREMENT AND NOT A PREFERENCE. The first
// arrangement was one unit per kernel FAMILY, which put the four charged mesons - pi+, pi-, K+,
// K- - in a unit of their own, and ptxas died on it with the same 0xC0000005 that V63 is about,
// in 100 seconds, deterministically, alone on the machine with nothing else compiling. The same
// four kernels compile perfectly well inside the eighteen-kernel unit this split replaced. So a
// translation unit does not get safer by being made smaller: what V63 called a cliff is a cliff
// in both directions, and the only shape this project has ever measured ptxas to compile for
// every arrangement of the physics is ONE `run_step_hadron` on its own. The thirteen are
// therefore thirteen units. `run_step_lepton`'s and `run_step_neutral`'s pairs share a unit
// each, measured to compile in 269 and 260 seconds; those two templates are switched by a
// compile-time constant and neither pair is the build's long pole.
//
// Measured, on a ten-line pair rather than assumed - and inverted, because the whole split
// rests on it: with the declaration, `cuobjdump -res-usage` on the launching object reports no
// device function and the program still runs; with the line deleted, the launching object
// carries the kernel again. That is the check, and build_engine.bat runs the same cuobjdump
// over out/transport_run.obj so that a launch added without a declaration here is caught by
// the build rather than by somebody wondering why the engine takes 24 minutes again.
//
// IT HAS TO BE CAUGHT THAT WAY, because the link will not do it. The comment at the top of
// this file says a __global__ template instantiated in two translation units is rejected with
// "explicit specialization ... is not a specialization of a function template". That is the
// error this arrangement was built to avoid, and on CUDA 11.6 it does not happen: explicit
// instantiation definitions of the same specialisation in two objects link CLEANLY, the stubs
// being COMDAT-folded (measured on the same pair). So the duplicate is silent, and what it
// costs is compile time rather than a link error. Exactly one definition per specialisation,
// enforced by the object check, is the rule; the error message is not a safety net.
//
// A project with its own hook is unaffected. These declarations name StepTap<double> and
// nothing else, so a different hook's specialisations still instantiate implicitly in the one
// translation unit that instantiates the engine for it - see the note at the top of this file,
// and tests/test_custom_hook.cu, which is that arrangement built and run by build_all.bat.
//
// The three utility kernels - seed_from_primaries, count_species, scatter_species - are NOT
// declared here. They are templated on real_t alone, hold no physics, and compile in seconds;
// they stay in the engine's own object, which is where their only launches are.
// The four signatures, written once. Each family's .cu says `template G4GPU_STEP_ION(...);`
// against the same macro this block says `extern template G4GPU_STEP_ION(...);` against, so a
// parameter added to a kernel cannot leave a declaration and a definition disagreeing - which
// would not be a compile error, only a specialisation that stopped matching and quietly went
// back to being instantiated wherever it was launched.
#define G4GPU_STEP_GAMMA(HOOK)                                                               \
  __global__ void run_step_gamma<double, HOOK>(                                              \
      Scene<double>, TrackBuffer<double>, const int*, TrackBuffer<double>, int, int, double*, \
      double*, int, vis::TrajectoryBuffer, int*, SecondaryArena, EmitterBooks, HOOK)
#define G4GPU_STEP_LEPTON(POSITRON, HOOK)                                                    \
  __global__ void run_step_lepton<double, POSITRON, HOOK>(                                   \
      Scene<double>, TrackBuffer<double>, const int*, TrackBuffer<double>, int, int, double*, \
      double*, int, vis::TrajectoryBuffer, int*, SecondaryArena, EmitterBooks, HOOK)
#define G4GPU_STEP_HADRON(TYPE, HOOK)                                                        \
  __global__ void run_step_hadron<double, TYPE, HOOK>(                                       \
      Scene<double>, TrackBuffer<double>, const int*, TrackBuffer<double>, int, int, double*, \
      double*, int, had::HadronicWiring<double>, vis::TrajectoryBuffer, int*, SecondaryArena, \
      EmitterBooks, HOOK)
#define G4GPU_STEP_NEUTRAL(TYPE, HOOK)                                                       \
  __global__ void run_step_neutral<double, TYPE, HOOK>(                                      \
      Scene<double>, TrackBuffer<double>, const int*, TrackBuffer<double>, int, int, double*, \
      double*, int, const had::NeutronGeneralXs<double>*, had::HadronicWiring<double>,       \
      double*, int*, vis::TrajectoryBuffer, int*, SecondaryArena, EmitterBooks, HOOK)
// P15's interaction kernels: FIVE of them, one per model, one translation unit each
// (`host/transport_run_int_*.cu`). Those five are the only objects in the build that carry any
// hadronic model at all, and the split is not a preference - ptxas cannot compile four models
// into one module (docs/RISK.md V189; the measurement is in `had::InteractionBucket`).
#define G4GPU_INTERACTION(BUCKET, HOOK)                                                      \
  __global__ void run_interaction<double, BUCKET, HOOK>(                                     \
      Scene<double>, const had::PendingInteraction<double>*, const int*, int, int,           \
      TrackBuffer<double>, int, double*, double*, had::HadronicWiring<double>,               \
      had::InteractionPool<double>, vis::TrajectoryBuffer, int*, SecondaryArena, EmitterBooks,\
      HOOK)

extern template G4GPU_STEP_GAMMA(StepTap<double>);
extern template G4GPU_STEP_LEPTON(false, StepTap<double>);
extern template G4GPU_STEP_LEPTON(true, StepTap<double>);
extern template G4GPU_STEP_HADRON(ParticleType::kProton, StepTap<double>);
extern template G4GPU_STEP_HADRON(ParticleType::kAntiProton, StepTap<double>);
extern template G4GPU_STEP_HADRON(ParticleType::kMuonMinus, StepTap<double>);
extern template G4GPU_STEP_HADRON(ParticleType::kMuonPlus, StepTap<double>);
extern template G4GPU_STEP_HADRON(ParticleType::kPionPlus, StepTap<double>);
extern template G4GPU_STEP_HADRON(ParticleType::kPionMinus, StepTap<double>);
extern template G4GPU_STEP_HADRON(ParticleType::kKaonPlus, StepTap<double>);
extern template G4GPU_STEP_HADRON(ParticleType::kKaonMinus, StepTap<double>);
extern template G4GPU_STEP_HADRON(ParticleType::kAlpha, StepTap<double>);
extern template G4GPU_STEP_HADRON(ParticleType::kDeuteron, StepTap<double>);
extern template G4GPU_STEP_HADRON(ParticleType::kTriton, StepTap<double>);
extern template G4GPU_STEP_HADRON(ParticleType::kHe3, StepTap<double>);
extern template G4GPU_STEP_HADRON(ParticleType::kGenericIon, StepTap<double>);
extern template G4GPU_STEP_NEUTRAL(ParticleType::kNeutron, StepTap<double>);
extern template G4GPU_STEP_NEUTRAL(ParticleType::kPiZero, StepTap<double>);
extern template G4GPU_INTERACTION(had::InteractionBucket::kFtfp, StepTap<double>);
extern template G4GPU_INTERACTION(had::InteractionBucket::kBertini, StepTap<double>);
extern template G4GPU_INTERACTION(had::InteractionBucket::kBinary, StepTap<double>);
extern template G4GPU_INTERACTION(had::InteractionBucket::kLightIon, StepTap<double>);
extern template G4GPU_INTERACTION(had::InteractionBucket::kAtRest, StepTap<double>);


// ---------------------------------------------------------------- method bodies

/// Raises the per-thread device stack to what the interaction kernels need, once, the first
/// time an iteration has a hadron in it.
///
/// WHY IT IS DEFERRED AT ALL is in `Upload`'s own comment and in docs/RISK.md V190: 86,016
/// bytes a thread reserves five of an 8 GB card's gigabytes and costs B1's 6 MeV gamma gate
/// 28% of its throughput, on a run that can never reach a hadronic model.
///
/// WHY IT IS FATAL WHEN IT FAILS. The reservation is made after the pools are allocated, so it
/// can fail for want of memory where the same call at `Upload` would have succeeded. There is
/// no safe fall-back: a kernel whose frame is larger than the limit does not warn, it takes an
/// illegal memory access, and the run after it reports a plausible number. Lowering the slot
/// count or the batch is what a caller does about it, and the message says so.
template <typename real_t, typename StepHook>
void TransportEngine<real_t, StepHook>::RaiseStackForInteractions() {
  if (stack_raised_) { return; }
  stack_raised_ = true;
  // THE NUMBER IS READ OFF THE FIVE KERNELS, and until P15's second pass it was written down -
  // 86,016, "81,584 rounded up to a 4 kB boundary with one page of margin". A constant like that
  // is right for exactly one build: P9e's `G4BinaryLightIonReaction::Interact` puts a whole
  // Binary cascade behind the light-ion arm, and its own probe measured 63,088 bytes of frame
  // for that entry point alone (V183 on its branch), against the 31,952 `run_interaction<
  // kLightIon>` had when the constant was chosen. So the reservation is the largest
  // `localSizeBytes` of the five, rounded up the same way.
  //
  // WHY `localSizeBytes` IS THE RIGHT QUANTITY, measured rather than assumed (docs/RISK.md
  // V196). For a call graph ptxas can size, it overlays every non-inlined callee's frame INTO
  // THE ENTRY'S: `-Xptxas -v` on `transport_run_int_binary` reports `propagate`,
  // `do_time_step`, `apply_collision`, `bic_deexcite_fragment` and P6's `deexcite` at 0 bytes
  // each and the kernel at 81,584, and `cudaFuncGetAttributes` returns the entry's number. So
  // the entry's frame IS the tree's requirement, and V190's "maximum over the call tree" was
  // right.
  //
  // AND WHAT A SHORTFALL WOULD COST, which is less than this function's first version claimed.
  // For a kernel ptxas can size, this driver (610.62, CUDA 11.6) RAISES THE LIMIT ITSELF at the
  // launch: a probe needing 40,000 B, launched under a 1,024 B limit, ran to the right answer and
  // left the limit at 40,000. So a reservation that came up short would not fault - it would
  // spend the memory at an interaction launch nobody chose, after the pools were sized against
  // the free memory printed below. That is still a defect, because every 4 kB is 270 MB of an
  // 8 GB card, and the report checks for it: `stack_reserved_` against the limit after the run.
  struct KernelFrame {
    const char* name;
    cudaFuncAttributes attr;
  };
  KernelFrame k[5] = {{"run_interaction<kFtfp>", {}},     {"run_interaction<kBertini>", {}},
                      {"run_interaction<kBinary>", {}},   {"run_interaction<kLightIon>", {}},
                      {"run_interaction<kAtRest>", {}}};
  G4GPU_CUDA_CHECK(cudaFuncGetAttributes(
      &k[0].attr, run_interaction<real_t, had::InteractionBucket::kFtfp, StepHook>));
  G4GPU_CUDA_CHECK(cudaFuncGetAttributes(
      &k[1].attr, run_interaction<real_t, had::InteractionBucket::kBertini, StepHook>));
  G4GPU_CUDA_CHECK(cudaFuncGetAttributes(
      &k[2].attr, run_interaction<real_t, had::InteractionBucket::kBinary, StepHook>));
  G4GPU_CUDA_CHECK(cudaFuncGetAttributes(
      &k[3].attr, run_interaction<real_t, had::InteractionBucket::kLightIon, StepHook>));
  G4GPU_CUDA_CHECK(cudaFuncGetAttributes(
      &k[4].attr, run_interaction<real_t, had::InteractionBucket::kAtRest, StepHook>));
  std::size_t need = 0;
  for (const KernelFrame& kf : k) {
    if (kf.attr.localSizeBytes > need) {
      need = kf.attr.localSizeBytes;
      stack_kernel_ = kf.name;
    }
  }
  // V190's rule: up to a 4 kB boundary, and one page of margin on top. Never below the
  // stepping kernels' floor, because the stepping kernels run under the same limit and theirs
  // is the one the driver cannot enforce for them.
  std::size_t want = ((need + 4095u) / 4096u) * 4096u + 4096u;
  want = std::max(want, stepping_stack_bytes());
  std::size_t have = 0;
  cudaDeviceGetLimit(&have, cudaLimitStackSize);
  stack_reserved_ = std::max(want, have);
  if (have >= want) { return; }
  const cudaError_t e = cudaDeviceSetLimit(cudaLimitStackSize, want);
  size_t f = 0, t = 0;
  cudaMemGetInfo(&f, &t);
  if (e != cudaSuccess) {
    std::printf("\nFATAL: the first hadron of this run needs %zu bytes of device stack a\n"
                "       thread - %s's frame is %zu - and the driver refused: %s.\n"
                "       %.2f GB of %.2f GB is free. Lower the batch, lower\n"
                "       SetInteractionSlots (now %d, %.1f MB), or run without hadrons.\n"
                "       docs/RISK.md V190 and V196.\n",
                want, stack_kernel_, need, cudaGetErrorString(e), double(f) / 1073741824.0,
                double(t) / 1073741824.0, n_interaction_slots_,
                double(interaction_bytes_) / 1048576.0);
    std::exit(2);
  }
  std::printf("device stack raised to %zu B a thread for the interaction kernels - %s's frame is "
              "%zu B - %.2f GB of %.2f GB left (docs/RISK.md V190, V196)\n",
              want, stack_kernel_, need, double(f) / 1073741824.0, double(t) / 1073741824.0);
}

template <typename real_t, typename StepHook>
void TransportEngine<real_t, StepHook>::Upload(const g4::FlatScene& scene, int batch_size,
                                     int threads) {
    batch_ = batch_size;
    threads_ = threads;
    n_scorers_ = std::max<int>(1, static_cast<int>(G4SDManager::GetSDMpointer()->Scorers().size()));

    // The solid engine is mutually recursive, so the kernels need a real call stack; the 1 KB
    // default is not enough for one frame of the distance routine. Overflow surfaces as an
    // illegal memory access from an unrelated API call, with nothing pointing at the cause.
    //
    // **86,016 SINCE P15, AND EVERY DIGIT OF IT IS A MEASUREMENT.** 16,384 was the stepping
    // kernels' ceiling and they are still far under it (`run_step_hadron<kProton>` is 3,936 B).
    // The interaction kernels are not. `-Xptxas -v` on the engine's own five units:
    //
    //     run_interaction<kFtfp>       23,008 B      run_interaction<kAtRest>     35,360 B
    //     run_interaction<kLightIon>   31,968 B      run_interaction<kBinary>   **81,600 B**
    //     run_interaction<kBertini>    33,680 B
    //
    // A kernel's frame is the maximum over its call tree, and the Binary cascade's tree -
    // `bic::apply_yourself` through `Propagate`, `ApplyCollision` and `DeExcite` into P6's
    // PreCompound and P3's evaporation - is 2.4 times the next largest. A limit below the frame
    // is not a warning: it is an illegal memory access from an unrelated API call, which is
    // exactly the failure this line was written for in the first place.
    //
    // **AND IT IS EXPENSIVE, DEVICE-WIDE, AND NOT BOUNDED BY THE LAUNCH.** CUDA reserves the
    // stack for the maximum threads the DEVICE can hold resident, not for the threads a kernel
    // is launched with, so `SetInteractionSlots` does not bound it. Measured on this RTX 3070
    // (46 SMs, 1536 threads an SM, 8.00 GB) with `cudaMemGetInfo` after the limit is set:
    //
    //      16,384 B/thread   5.938 GB free      86,016 B/thread   0.952 GB free
    //      81,920 B/thread   1.221 GB free      98,304 B/thread   0.099 GB free
    //
    // So the Binary cascade's frame costs this port **five of the card's eight gigabytes**, and
    // 86,016 is 81,600 rounded up to a 4 kB boundary with one page of margin - not a round
    // number chosen for comfort, because every 4 kB past it is another 270 MB. What is left is
    // what `SetInteractionSlots` and the batch have to fit in, which is why the pool's default
    // dropped to 128 slots and why `Upload` prints the free memory after both.
    // `G4GPU_STACK_BYTES` overrides it, for the same reason `G4GPU_LIVE_PER_EVENT` overrides
    // the pool: the only way to answer "what did this cost" is to run the same binary both
    // ways, and a program that has no such option cannot be asked. It is a RESOURCE knob and
    // not a physics one - but setting it below a kernel's frame is an illegal memory access in
    // that kernel, not a warning, so anything under 86,016 is only for a run that cannot make a
    // hadron.
    // **RAISED LAZILY, ON THE FIRST ITERATION THAT HAS A HADRON IN IT.** `Upload` sets the
    // stepping kernels' 16,384 and `RaiseStackForInteractions` sets the interaction kernels'
    // 86,016 the first time one could run, because the difference is 28% of a photon run's
    // throughput and a photon run can never reach a hadronic model.
    //
    // MEASURED, on B1's own 2,000,000-event 6 MeV gamma gate, same binary, one environment
    // variable apart:
    //
    //     16,384 B/thread   1,160 ms of event loop   1.72e6 events/s   5.94 GB free
    //     86,016 B/thread   1,616 ms                 1.24e6 events/s   1.17 GB free
    //
    // and the dose is 425.86 pGy either way, to every digit it prints. So this is not physics
    // and it is not memory pressure - halving the track pool moved it by 1.4% - it is what a
    // device-wide stack reservation does to occupancy and local-memory addressing.
    //
    // `G4GPU_STACK_BYTES` overrides both, for the same reason `G4GPU_LIVE_PER_EVENT` overrides
    // the pool: the only way to answer "what did this cost" is to run the same binary both ways.
    //
    // **TWO THINGS ABOVE ARE CORRECTED BY V196, AND THE TABLE IS HISTORY.** The frames are the
    // ones this comment was written against; `RaiseStackForInteractions` now READS the number
    // off the five kernels, so it cannot go stale when P9e's `Interact` grows the light-ion
    // arm. And "a limit below the frame is an illegal memory access" is true of THIS limit - the
    // stepping kernels', whose stack ptxas cannot size because the solid engine recurses - and
    // not of the interaction kernels', which ptxas sizes exactly and which this driver simply
    // raises the limit for at launch if nothing else has. Both measured, with two probes of
    // thirty lines each, in docs/RISK.md V196.
    G4GPU_CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, stepping_stack_bytes()));

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

    // The allocation only, here: the e+- dE/dx table is the sum over G4eIonisation AND
    // G4eBremsstrahlung, so it cannot be built until the Seltzer-Berger data is loaded, which
    // happens below in `upload_brems_for`. This used to build it with a crude radiative-yield
    // scaling and then throw that away - a table nothing could read, built twice.
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
    // The other half of the lepton's multiple scattering. Urban's table above stops at
    // `G4EmParameters::MscEnergyLimit()` because that is where Geant4's Urban model stops;
    // this one starts there and runs to 100 TeV. Unconditional: a run that makes any lepton
    // can make one above 100 MeV, and `step_lepton` reads it without a null check for the
    // reason Scene's field says.
    d_wv_ = upload_wv_lepton<real_t>(h_mats_.data(), n_materials_);

    // hadElastic's device tables, beside the hadron range table and for the same reason: a run
    // that will carry a charged hadron needs them before the first primary is seeded. The
    // element list is the scene's, so the per-(pion, Z) G4ElasticData is built for the elements
    // the detector is actually made of rather than for all 92. See host/hadronic_upload.cuh.
    elastic_tables_ = upload_elastic_tables<real_t>(zs);

    // The neutron's cross sections. Both configurations, because the stage is a run-time switch
    // and the same binary has to produce both columns of the comparison table: the two
    // per-process G4PARTICLEXS data sets for `kStage1` and G4NeutronGeneralProcess's five
    // combined tables for `kFinal`. The combined ones are per SCENE, not per element, because
    // `BuildPhysicsTable` sums MACROSCOPIC cross sections and so needs the atom densities.
    neutron_tables_ = upload_neutron_tables<real_t>(h_mats_.data(), n_materials_);
    d_neutron_xs_ = neutron_tables_.d_general;

    // P15: the five G4ParticleInelasticXS data sets and G4BGGPionInelasticXS. Beside the
    // elastic tables and for the same reason - a run that will carry a charged hadron needs
    // them before the first primary is seeded, and the engine cannot know what the generator
    // will produce until it has produced one.
    inelastic_tables_ = upload_inelastic_tables<real_t>();

    // P3's nuclear level data, which the capture sub-process walks. Unconditional by default
    // since P8d - `SetNuclearLevelData` has the reason, and it is Geant4's own answer
    // (`G4ExcitationHandler::SetParameters` at initialisation, whether a neutron arrives or
    // not).
    if (load_level_data_) {
      // `Zmax + 1`, which is the convention G4ExcitationHandler::SetParameters applies -
      // `UploadNuclearLevelData(Zmax+1)` - and `read_all_level_data`'s strict `Z < mZ` is why
      // the element with the largest Z in the geometry is loaded only because of the +1.
      int zmax = 20;
      for (int z : zs) {
        if (z > zmax) { zmax = z; }
      }
      level_tables_ = upload_level_data(level_storage_, zmax + 1);
    }

    // The hadron range table. Built here rather than on demand because a run that will carry
    // a proton needs it before the first primary is seeded, and the engine cannot know what
    // species the generator will produce until it has produced one. A megabyte in double
    // precision - ten species against the electron table's two - which is not worth a
    // conditional beside the geometry and the photon data.
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
                           d_wv_,      d_hrt_,
                           static_cast<real_t>(scene.range_cut_mm),
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
    // Refused here rather than discovered on the device. If a table is uploaded without the
    // samplers to go with it, every neutron draws a finite interaction length, arrives at a
    // sub-process branch with nothing to apply, and is killed with its energy dumped locally -
    // a dose that looks like hadronic transport and is a deposit at the first interaction
    // point. The device cannot report that usefully at kernel rates, so the combination is
    // refused where it is set up.
    //
    // P8d LANDED THE FINAL STATES AND THE REFUSAL STAYED, with what it checks turned around.
    // It used to fire on any non-null table, because there were no final states at all; what it
    // now asserts is that the three things `step_neutral` dereferences arrived together. Two of
    // them are cross sections the elastic and capture sub-processes draw a TARGET from
    // (`SampleZandA` on their own data stores, which the general process's table cannot
    // answer), and the third is the level scheme the capture cascade walks.
    //
    // A TABLE WITH NO LEVELS IS THE SUBTLER FAILURE AND IT IS NOT THE OBVIOUS ONE. This comment
    // said the capture "would emit no gamma, so the neutron's binding energy would silently
    // vanish". Measured - `tests/test_neutron_general.cu` section 4, 2000 thermal captures in
    // lead with the table and with a null view - it is the opposite: the cascade takes the
    // CONTINUUM arm of `generate_gamma` instead of walking a discrete scheme and emits
    // **26,485 secondaries carrying 14,609.9 MeV against 9,021 carrying 13,694.7 MeV**, three
    // times the multiplicity and 6.7% more energy. So the failure mode is a plausible capture
    // with a wrong spectrum rather than a missing one, which is exactly the kind of thing that
    // is found in a dose comparison months later. docs/RISK.md V60.
    if (d_neutron_xs_ != nullptr
        && (neutron_tables_.sub.elastic == nullptr || neutron_tables_.sub.capture == nullptr)) {
      std::printf(
          "\nFATAL: the neutron's combined cross-section table is present but a sub-process\n"
          "  data set is not (elastic %s, capture %s). `SampleZandA` draws the target from the\n"
          "  sub-process's OWN data store - G4NeutronGeneralProcess::PostStepDoIt calls\n"
          "  `fCurrentXSS->ComputeCrossSection` before delegating - so the combined table\n"
          "  cannot stand in for it. See physics/hadronic/neutron_wiring.cuh.\n",
          (neutron_tables_.sub.elastic != nullptr) ? "present" : "MISSING",
          (neutron_tables_.sub.capture != nullptr) ? "present" : "MISSING");
      std::exit(2);
    }
    if (d_neutron_xs_ != nullptr && level_tables_.view.n_managers == 0) {
      std::printf(
          "\nFATAL: the neutron's cross sections are on the device and the nuclear level\n"
          "  scheme is not. A capture would still run - and emit the WRONG cascade rather than\n"
          "  none: measured in lead at thermal energy, 2000 captures give 26,485 secondaries\n"
          "  carrying 14,609.9 MeV with a null table against 9,021 carrying 13,694.7 MeV with\n"
          "  the real one, because G4PhotonEvaporation falls back to its continuum arm.\n"
          "  Either let SetNuclearLevelData stay on (the default since P8d) or resolve\n"
          "  G4LEVELGAMMADATA; see host/level_upload.cuh.\n");
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

    // ---- P15: the interaction queue and the workspace pool it is drained into.
    //
    // THE QUEUE IS A BOUND AND THE POOL IS A CHOICE, and the difference is the whole of
    // docs/RISK.md V188. A launch steps at most `pool_` tracks and one track queues at most one
    // interaction, so a queue of `pool_` entries cannot overflow - the same argument the arena
    // above makes for itself. The POOL is how many of those run at once; a shortage costs
    // launches and not interactions, because the drain loop in BeamOn chunks.
    //
    // THE DEFAULT SLOT COUNT IS 128 AND THE BINDING CONSTRAINT IS NOT THROUGHPUT, IT IS THE
    // DEVICE STACK. At 2,020,152 bytes a slot, 128 slots is 246.6 MB (195.9 MB before P9e's
    // ion arm) - and what is left of an
    // 8 GB card after 86,016 bytes a thread of reserved stack is 0.95 GB, measured (see the
    // `cudaDeviceSetLimit` above and docs/RISK.md V190). 256 slots would take 392 MB of that
    // 0.95 GB and leave the track pool, the queue, the 9.52 MB level scheme and the scene to
    // share what remains. The run prints how deep the queue actually got, so
    // `SetInteractionSlots` can be raised against a measurement rather than a hope - and a
    // shortage costs launches, not interactions.
    n_interaction_slots_ = (interaction_slots_request_ > 0) ? interaction_slots_request_ : 128;
    // THE QUEUE'S BOUND IS THE THROTTLE'S, NOT THE POOL'S, and that is a factor of forty-eight.
    //
    // A track queues at most one interaction per launch, so "one entry per pool slot" is a
    // bound - but a loose one, because a launch cannot STEP every live track. `BeamOn`'s
    // throttle lets it step at most `(capacity - live) / max_secondaries_per_step(species)`
    // tracks of a species, and that reservation is 48 for every species that can enqueue. So
    // the number of interactions one launch can produce is at most `pool_ / 48`, and the
    // slack below is for the case where a future species enqueues at a smaller reservation.
    //
    // It is worth having as memory rather than as elegance: `pool_` entries was 368 MB on B1's
    // own batch, out of the 1.17 GB the device stack leaves free (docs/RISK.md V190). This is
    // 8 MB. `kInelasticQueueFull` is still the refusal if the bound is ever wrong, and the
    // stepper's disposal for it is the conservative one.
    const long long q_bound = pool_ / max_secondaries_per_step(kSpeciesProton) + 1024;
    queue_capacity_ = (queue_capacity_request_ > 0)
                          ? queue_capacity_request_
                          : static_cast<int>((q_bound < pool_) ? q_bound : pool_);
    {
      const std::size_t q_bytes =
          sizeof(had::PendingInteraction<real_t>) * static_cast<std::size_t>(queue_capacity_);
      G4GPU_CUDA_CHECK(cudaMalloc(&d_queue_, q_bytes));
      G4GPU_CUDA_CHECK(cudaMalloc(&d_queue_cursor_, sizeof(int)));
      G4GPU_CUDA_CHECK(cudaMemset(d_queue_cursor_, 0, sizeof(int)));
      G4GPU_CUDA_CHECK(cudaMalloc(&d_qidx_, sizeof(int) * queue_capacity_));
      G4GPU_CUDA_CHECK(cudaMalloc(
          &d_qcount_,
          sizeof(int) * static_cast<int>(had::InteractionBucket::kNumInteractionBuckets)));
      interaction_bytes_ = q_bytes + sizeof(int) + sizeof(int) * queue_capacity_;

      // The slots. ONE CONSTRUCTED HOST IMAGE, uploaded into every slot, and not a memset:
      // docs/RISK.md V104 is a member that only a constructor sets and that decides physics
      // when it is zero, and `bert::NucleiModel` and `bic::CascadeBuffers` are the same shape
      // of risk. `ftf::entry::build` does exactly this for its own pool and says so.
      const std::size_t s_bytes = sizeof(had::InteractionSlot<real_t>)
                                  * static_cast<std::size_t>(n_interaction_slots_);
      G4GPU_CUDA_CHECK(cudaMalloc(&d_slots_, s_bytes));
      {
        auto* image = new had::InteractionSlot<real_t>();
        for (int i = 0; i < n_interaction_slots_; ++i) {
          G4GPU_CUDA_CHECK(cudaMemcpy(static_cast<had::InteractionSlot<real_t>*>(d_slots_) + i,
                                      image, sizeof(*image), cudaMemcpyHostToDevice));
        }
        delete image;
      }
      interaction_bytes_ += s_bytes;

      // The Binary cascade's 306 concrete channels: read-only, identical for every thread, one
      // copy for the whole run - the same relationship `LundTables` has to FTFP's pool.
      {
        auto* ch = new bic::imr::ConcreteChannel[bic::imr::kConcreteChannelCount];
        const int n_ch = bic::imr::build_concrete_channels(ch,
                                                           bic::imr::kConcreteChannelCount);
        const std::size_t c_bytes = sizeof(bic::imr::ConcreteChannel)
                                    * static_cast<std::size_t>(n_ch);
        G4GPU_CUDA_CHECK(cudaMalloc(&d_bic_channels_, c_bytes));
        G4GPU_CUDA_CHECK(cudaMemcpy(d_bic_channels_, ch, c_bytes, cudaMemcpyHostToDevice));
        n_bic_channels_ = n_ch;
        interaction_bytes_ += c_bytes;
        delete[] ch;
      }

      // FTFP's pool, through its own contract. Same slot count, same thread index - and
      // `entry::Workspace` rather than `entry::HadronWorkspace` because a GenericIon above
      // 3 GeV per nucleon goes to FTFP and the hadron-only type refuses an ion by capacity.
      ftf_pool_ = g4gpu::hadronic::ftf::entry::build<g4gpu::hadronic::ftf::entry::Workspace>(
          n_interaction_slots_, /*enable_bc_particles=*/true, /*verbose=*/false);
      if (!ftf_pool_.view.ok()) {
        std::printf("\nFATAL: FTFP's workspace pool (%d slots) could not be built. The "
                    "inelastic\n       process cannot run without it; see "
                    "ftf/ftf_entry.cuh.\n", n_interaction_slots_);
        std::exit(1);
      }
      interaction_bytes_ += ftf_pool_.bytes;

      // P3's Fermi break-up table, which every arm's de-excitation tail walks. Built on the
      // host from the level scheme that is already there, and uploaded once.
      fermi_pool_ = upload_fermi_pool(level_storage_.view(), /*verbose=*/false);
      interaction_bytes_ += fermi_pool_.bytes;

      std::printf("interaction pool: %d slots of %zu B + FTFP's %zu B = %.1f MB; queue %d "
                  "entries of %zu B = %.1f MB\n",
                  n_interaction_slots_, sizeof(had::InteractionSlot<real_t>),
                  g4gpu::hadronic::ftf::entry::kWorkspaceBytes,
                  double(s_bytes + ftf_pool_.bytes) / 1048576.0, queue_capacity_,
                  sizeof(had::PendingInteraction<real_t>), double(q_bytes) / 1048576.0);
    }

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
        // ---- can anything be queued this iteration at all?
        //
        // A TRACK THAT IS NOT A HADRON CANNOT ENQUEUE, so an iteration with no hadronic species
        // live needs no cursor rewind, no read-back and no drain - and the read-back is the
        // expensive half, because `cudaMemcpy` D2H is a full device synchronisation and this
        // loop only had one of those (the species histogram) before P15.
        //
        // MEASURED, AND IT IS WHY THIS TEST IS HERE: B1's 2,000,000-event 6 MeV gamma gate went
        // from 1,172 ms of event loop to 1,947 ms with the drain running unconditionally - 1.7
        // times slower on a run that cannot produce one interaction. It is not memory: halving
        // the pool with `G4GPU_LIVE_PER_EVENT=2` moved it by 1.4%, which rules out the 368 MB
        // the queue was taking. It is the second synchronisation per iteration, 88 of them in
        // that run.
        //
        // `nsp[]` is the species histogram the counting sort already read back, so this costs
        // nothing: it is a sum over the hadronic rows of a `kNumTrackSpecies` array on the host.
        bool any_hadron = false;
        for (int sp = 0; sp < kNumTrackSpecies && !any_hadron; ++sp) {
          if (nsp[sp] > 0 && max_secondaries_per_step(sp) > 4) { any_hadron = true; }
        }
        if (any_hadron) { RaiseStackForInteractions(); }
        if (any_hadron && d_queue_cursor_ != nullptr) {
          // The interaction queue holds ONE iteration's interactions and is drained at the end
          // of it, so its cursor is rewound here beside the arena's and for the same reason:
          // the bound that says it cannot overflow is "one launch's tracks", not "one run's".
          G4GPU_CUDA_CHECK(cudaMemsetAsync(d_queue_cursor_, 0, sizeof(int)));
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
        // `G4GPU_HADRONIC_STAGE=final` (or `stage1`) overrides the engine's default, for the
        // same reason `G4GPU_LIVE_PER_EVENT` and `G4GPU_ION_INELASTIC` exist: `examples/B1` is
        // Geant4's own B1 and has no UI command that reaches a setter.
        //
        // AND UNTIL P15 THERE WAS NOTHING IN THIS REPOSITORY THAT SELECTED `kFinal` AT ALL.
        // The default is `kStage1`, which for a neutron means `hadElastic` and `nCapture` as two
        // separate processes and NO inelastic - see `step_neutral`'s `has_stage1_elastic` /
        // `has_stage1_capture`, which have no inelastic sibling. So P15's neutron inelastic
        // sub-process, the port's largest named hole from P8d, was wired and then reachable only
        // from `tests/test_inelastic_transport.cu`, which sets the stage itself. A deliverable
        // no production run can reach is not a deliverable. docs/RISK.md V194.
        //
        // The DEFAULT is deliberately left at `kStage1` here rather than flipped: every gate in
        // this project was measured in it, and which stage ships is a decision about all of them
        // and not about P15. What P15 owes is that the final stage can be selected and has been
        // measured, which is what this variable and the sweep's use of it are.
        had::HadronicStage stage = had_stage_;
        if (const char* env = std::getenv("G4GPU_HADRONIC_STAGE")) {
          if (env[0] == 'f') { stage = had::HadronicStage::kFinal; }
          else if (env[0] == 's') { stage = had::HadronicStage::kStage1; }
        }
        had_wiring.stage = stage;
        had_wiring.decay = had_decay_;
        had_wiring.hadron_elastic = had_elastic_;
        had_wiring.neutron_capture = had_capture_;
        // Five device pointers, copied by value into the launch like the rest of the struct.
        // Null in a run whose scene can see no hadron - Upload only builds them when it builds
        // the hadron range table - and `elastic_xs_per_volume` then returns zero, which is the
        // same "no process" state a lepton is in.
        had_wiring.elastic = elastic_tables_.view;
        // The neutron's two per-process cross sections. Read in `kStage1` as two whole processes
        // with their own interaction lengths, and in `kFinal` as the data stores the chosen
        // sub-process draws its target from. `host/neutron_upload.cuh`.
        had_wiring.neutron = neutron_tables_.sub;
        // P3's level scheme. Reached since P8d - the capture sub-process walks it - and
        // `Upload`'s refusal above is what keeps it from arriving separately from the cross
        // sections.
        had_wiring.level_data = level_tables_.view;
        // P15's two processes and the queue they go through. The tables are null in a run
        // whose G4PARTICLEXSDATA could not be resolved, which is the same "no process" state a
        // species with no inelastic channel is in.
        had_wiring.hadron_inelastic = had_inelastic_;
        // `G4GPU_ION_INELASTIC=0` switches the five ion names off without rebuilding B1, the
        // way `G4GPU_LIVE_PER_EVENT` sets the live-track pool. Read here rather than cached at
        // construction, so a sweep that sets it per beam gets it per beam.
        bool ion_inel = had_ion_inelastic_;
        if (const char* env = std::getenv("G4GPU_ION_INELASTIC")) {
          ion_inel = !(env[0] == '0' && env[1] == '\0');
        }
        had_wiring.ion_inelastic = ion_inel;
        had_wiring.hadron_at_rest = had_at_rest_;
        had_wiring.inelastic = inelastic_tables_.view;
        had_wiring.queue.items = d_queue_;
        had_wiring.queue.cursor = d_queue_cursor_;
        had_wiring.queue.capacity = queue_capacity_;
        had_wiring.books.count = d_had_refused_n_;
        had_wiring.books.energy = d_had_refused_e_;

        // The pool the interaction kernel runs in, built once per launch out of what Upload
        // allocated - the same relationship `had_wiring` has to the tables, and for the same
        // reason: a by-value struct the launch takes rather than a field on Scene.
        had::InteractionPool<real_t> pool_view;
        pool_view.slots = static_cast<had::InteractionSlot<real_t>*>(d_slots_);
        pool_view.n_slots = n_interaction_slots_;
        pool_view.bic_channels =
            static_cast<const bic::imr::ConcreteChannel*>(d_bic_channels_);
        pool_view.n_bic_channels = n_bic_channels_;
        pool_view.ftf = ftf_pool_.view;
        pool_view.fermi = fermi_pool_.view;
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
            case kSpeciesHe3:        G4GPU_LAUNCH_HADRON(ParticleType::kHe3); break;
            // The sixteenth instantiation, and the only one whose tracks are not all the same
            // particle: each carries its own nuclide in `TrackState::ion_za`.
            case kSpeciesGenericIon: G4GPU_LAUNCH_HADRON(ParticleType::kGenericIon); break;
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

        // ---- P15: drain the interaction queue, in chunks of `n_interaction_slots_`.
        //
        // THE CHUNKING IS THE WHOLE DESIGN, and it is what makes the slot count a capacity
        // whose shortage costs TIME rather than one whose shortage costs INTERACTIONS
        // (docs/RISK.md V188). Thread `i` of a chunk takes slot `i`, so `pool.slot(i)` is never
        // null and `kInteractionNoSlot` is a tripwire rather than a rate.
        //
        // AFTER every stepping launch and BEFORE the buffers swap, because the secondaries this
        // makes belong in `tracks_[nxt]` with the ones the steppers made: a cascade proton must
        // be stepped in the NEXT iteration, not this one. The read-back of the cursor is a
        // synchronising copy, which is the one place per iteration this loop waits for the
        // device - the counting sort above already reads `idx_n_` back for the same reason.
        if (any_hadron && d_queue_cursor_ != nullptr && d_slots_ != nullptr) {
          int n_queued = 0;
          G4GPU_CUDA_CHECK(cudaMemcpy(&n_queued, d_queue_cursor_, sizeof(int),
                                      cudaMemcpyDeviceToHost));
          if (n_queued > queue_capacity_) { n_queued = queue_capacity_; }
          if (n_queued > 0) {
            interactions_queued_ += n_queued;
            if (n_queued > max_queued_per_launch_) { max_queued_per_launch_ = n_queued; }

            // Bin by model, exactly as the species dispatch bins the track pool: histogram,
            // host-side prefix sum, scatter. `d_qcount_` is the histogram and then the bump
            // cursors, one allocation used twice - the same arrangement `idx_n_` has.
            constexpr int kNB = static_cast<int>(had::InteractionBucket::kNumInteractionBuckets);
            const int qblocks = (n_queued + threads_ - 1) / threads_;
            G4GPU_CUDA_CHECK(cudaMemset(d_qcount_, 0, sizeof(int) * kNB));
            count_interactions<real_t><<<qblocks, threads_>>>(d_queue_, n_queued, d_qcount_);
            int nb[kNB] = {};
            G4GPU_CUDA_CHECK(cudaMemcpy(nb, d_qcount_, sizeof(int) * kNB,
                                        cudaMemcpyDeviceToHost));
            InteractionOffsets qoff{};
            int acc = 0;
            for (int b = 0; b < kNB; ++b) { qoff.base[b] = acc; acc += nb[b]; }
            G4GPU_CUDA_CHECK(cudaMemset(d_qcount_, 0, sizeof(int) * kNB));
            scatter_interactions<real_t><<<qblocks, threads_>>>(d_queue_, n_queued, d_qidx_,
                                                                qoff, d_qcount_);

            // A block of 32 and not `threads_`: an interaction kernel is 255 registers with a
            // large frame, so a 256-thread block would not be resident anyway, and a small one
            // lets a chunk of 40 interactions use two SMs instead of one.
            constexpr int kIth = 32;
#define G4GPU_DRAIN(BUCKET)                                                                  \
  do {                                                                                       \
    const int b_ = static_cast<int>(BUCKET);                                                 \
    for (int base = 0; base < nb[b_]; base += n_interaction_slots_) {                        \
      const int n_this = (nb[b_] - base < n_interaction_slots_) ? (nb[b_] - base)            \
                                                                : n_interaction_slots_;      \
      run_interaction<real_t, BUCKET><<<(n_this + kIth - 1) / kIth, kIth>>>(                 \
          scene_, d_queue_, d_qidx_, n_this, qoff.base[b_] + base, tracks_[nxt].view, batch_, \
          d_score_, d_voxel_score_, had_wiring, pool_view, traj, d_status_warn_, sec_, books, \
          hook_);                                                                            \
      ++interaction_chunks_;                                                                 \
    }                                                                                        \
  } while (0)
            G4GPU_DRAIN(had::InteractionBucket::kFtfp);
            G4GPU_DRAIN(had::InteractionBucket::kBertini);
            G4GPU_DRAIN(had::InteractionBucket::kBinary);
            G4GPU_DRAIN(had::InteractionBucket::kLightIon);
            G4GPU_DRAIN(had::InteractionBucket::kAtRest);
#undef G4GPU_DRAIN
            // `kNone` is not drained and is not silent: the stepper booked
            // `kNoInelasticModel` for every entry in it before it enqueued, and an entry with
            // no model has no kernel to run. It is counted here so the report can say how many
            // interactions never reached a launch.
            interactions_no_model_ +=
                nb[static_cast<int>(had::InteractionBucket::kNone)];
            G4GPU_CUDA_CHECK(cudaGetLastError());
          }
        }

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
                    "    energy column, which is what the missing model would have moved.\n"
                    "\n"
                    "    SINCE P15 THE LEDGER HAS TWO GROUPS AND THEY MUST NOT BE ADDED.\n"
                    "    The first says HOW MUCH is missing - one entry per lost interaction,\n"
                    "    with the projectile's kinetic energy on it. The second says WHY, and\n"
                    "    every one of its entries is a second booking on an event that is\n"
                    "    already in the first. Sum the first group, read the second.\n\n",
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
    // ---- P15: the interaction queue and the pool, whether or not anything was refused.
    //
    // PRINTED EVEN WHEN NOTHING WENT WRONG, because the whole argument for the slot count is
    // that it is a measured capacity rather than an implicit one: a run that does not say how
    // many interactions it produced and how many launches it took to run them has not made the
    // capacity visible, it has only made it invisible in a different place.
    st.interactions = interactions_queued_;
    st.interaction_chunks = interaction_chunks_;
    st.max_queued_per_launch = max_queued_per_launch_;
    st.interaction_slots = n_interaction_slots_;
    st.interaction_bytes = interaction_bytes_;
    if (interactions_queued_ > 0) {
      std::printf("interactions: %lld queued in %lld chunk launches over %d slots (%.1f MB); "
                  "the busiest iteration queued %d (%.1f%% of the %d-entry queue)\n",
                  interactions_queued_, interaction_chunks_, n_interaction_slots_,
                  double(interaction_bytes_) / 1048576.0, max_queued_per_launch_,
                  100.0 * double(max_queued_per_launch_)
                      / double(queue_capacity_ ? queue_capacity_ : 1),
                  queue_capacity_);
      if (interactions_no_model_ > 0) {
        std::printf("             %lld of them had no model in range and never reached a "
                    "kernel (kNoInelasticModel above)\n", interactions_no_model_);
      }
    }
    // THE TRIPWIRE ON THE STACK RESERVATION. If the driver found the limit short at an
    // interaction launch it raised it itself (V196) - no fault, no error code, and 270 MB of an
    // 8 GB card per 4 kB spent after the pools were sized against the free memory printed at
    // the raise. The only trace it leaves is the limit, so the limit is read here and compared.
    // A run with no hadron never raised anything and has nothing to compare.
    if (stack_raised_) {
      std::size_t lim = 0;
      G4GPU_CUDA_CHECK(cudaDeviceGetLimit(&lim, cudaLimitStackSize));
      if (lim > stack_reserved_) {
        std::printf("STACK: the driver raised the device stack from the %zu B reserved off %s to "
                    "%zu B at an interaction launch - the reservation read off the kernels was "
                    "short. docs/RISK.md V196.\n", stack_reserved_, stack_kernel_, lim);
      } else {
        std::printf("stack: %zu B a thread reserved off %s, and no launch needed more\n",
                    stack_reserved_, stack_kernel_);
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
    // NOT a cudaFree of d_neutron_xs_: it points INTO neutron_tables_, which owns every
    // allocation behind it, so freeing both would be a double free. It used to be its own
    // allocation - and it used to be permanently null, which is why nobody noticed.
    d_neutron_xs_ = nullptr;
    free_neutron_tables<real_t>(neutron_tables_);
    free_elastic_tables<real_t>(elastic_tables_);
    free_inelastic_tables<real_t>(inelastic_tables_);
    free_level_data(level_tables_);
    // P15's pool and queue. `ftf::entry::free` is FTFP's own, for the allocations its contract
    // made; the rest is this file's.
    g4gpu::hadronic::ftf::entry::free<g4gpu::hadronic::ftf::entry::Workspace>(ftf_pool_);
    free_fermi_pool(fermi_pool_);
    cudaFree(d_slots_);
    cudaFree(d_bic_channels_);
    cudaFree(d_queue_);
    cudaFree(d_queue_cursor_);
    cudaFree(d_qidx_);
    cudaFree(d_qcount_);
    d_slots_ = nullptr;
    d_bic_channels_ = nullptr;
    d_queue_ = nullptr;
    d_queue_cursor_ = nullptr;
    d_qidx_ = nullptr;
    d_qcount_ = nullptr;
    cudaFree(d_vols_);
    cudaFree(d_mats_);
    cudaFree(d_rt_);
    cudaFree(d_wv_);
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
