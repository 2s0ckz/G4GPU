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

#include "host/device_select.cuh"
#include "host/paging_guard.cuh"
#include "host/transport_run.cuh"

namespace g4gpu::host {

// ---------------------------------------------------------------- kernels
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
/// One step of one photon.
///
/// @param had P19: the hadronic wiring, for `photonNuclear` - the photon's first hadronic
///        process, and why this kernel takes the struct `run_step_hadron` has always taken. A
///        photon whose step ends in a photo-nuclear interaction leaves through
///        `had.emx_queue` rather than through `out`; see `run_step_hadron`'s `queued`.
///
///        `__grid_constant__`, AND THAT IS MEASURED: `step_gamma` takes the struct's address, and
///        an ordinary by-value kernel parameter whose address is taken is copied into every
///        thread's local memory first - 416 bytes of frame on this kernel, which is the one B1's
///        6 MeV gate spends its time in. With the qualifier the address is the parameter
///        bank's own and nothing is copied (docs/RISK.md V210 has the gate before and after).
template <typename real_t, typename StepHook>
__global__ void run_step_gamma(Scene<real_t> scene, TrackBuffer<real_t> in, const int* idx,
                               TrackBuffer<real_t> out, int n, int batch, double* score,
                               double* voxel_score, int n_step,
                               const __grid_constant__ had::HadronicWiring<real_t> had,
                               vis::TrajectoryBuffer traj, int* status_warn,
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
  // `queued`: P19's exit, the same as `run_step_hadron`'s - a photo-nuclear interaction's
  // photon is the queue entry, and `run_emextra_drain` finishes its step, hook call included.
  bool queued = false;
  const bool alive = step_gamma(scene, p, rng, em, edep, srep, traj, &had, &queued);
  if (queued) { return; }
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

/// One step of one electron or positron. `had` is P19's, for `electronNuclear` and
/// `positronNuclear`, and `__grid_constant__` for the reason `run_step_gamma` gives.
template <typename real_t, bool kIsPositron, typename StepHook>
__global__ void run_step_lepton(Scene<real_t> scene, TrackBuffer<real_t> in, const int* idx,
                                TrackBuffer<real_t> out, int n, int batch, double* score,
                                double* voxel_score, int n_step,
                                const __grid_constant__ had::HadronicWiring<real_t> had,
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
  // `queued`: see `run_step_gamma`. A lepto-nuclear interaction's lepton comes back alive from
  // `run_emextra_drain`, which writes it into the pool itself.
  bool queued = false;
  const bool alive =
      step_lepton(scene, p, kIsPositron, rng, em, edep, srep, traj, &had, &queued);
  if (queued) { return; }
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

/// P23a: one step of one STRANGE track - a GROUP kernel, the first in this engine.
///
/// Every kernel above is one species, fixed at compile time, because `run_step_hadron`'s species
/// decides which msc model, which EM processes and which ion arms exist in the body (V55, V65).
/// The eight charged strange baryons (`is_strange_charged_track`) agree on every one of those
/// answers - Urban, G4hIonisation with UrbanFluc, no CoulombScat, no muonNuclear, no ion arm -
/// and differ only in data, so ONE kernel steps all eight: `step_hadron` is handed the track's
/// own species as `type` and a representative as its structural `kernel_type`. The eight
/// neutral ones (`is_strange_neutral_track`) share `step_neutral_strange`, which has no
/// structural branch at all. docs/RISK.md V240 has the measurement: what one unit per species
/// would cost against one per group, and the frame and spill the group kernels compile to.
///
/// What differs from `run_step_hadron` is only where the species comes from - the TRACK, here,
/// so `p.species` is read and never overwritten - and the stream's purpose value, which is
/// `run_step_hadron`'s for the charged group and `run_step_neutral`'s for the neutral one: a
/// track has one species, so no two tracks of one key can draw the same stream either way.
template <typename real_t, bool kCharged, typename StepHook>
__global__ void run_step_strange(Scene<real_t> scene, TrackBuffer<real_t> in, const int* idx,
                                 TrackBuffer<real_t> out, int n, int batch, double* score,
                                 double* voxel_score, int n_step,
                                 had::HadronicWiring<real_t> had,
                                 vis::TrajectoryBuffer traj, int* status_warn,
                                 SecondaryArena sec, EmitterBooks books, StepHook hook) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) { return; }
  TrackState<real_t> p;
  in.load(idx[i], p);
  if (i >= n_step) {
    out.append(p);
    return;
  }
  const int slot = (p.volume >= 0) ? scene.geometry.volumes[p.volume].score_index : -1;
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
  Philox<real_t> rng(p.rng_key, p.step, kCharged ? 0xB19Du : 0x4E7Au);
  const ParticleType species = p.species;
  const real_t track_mass = particle_def<real_t>(species).mass;
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
  bool queued = false;
  bool alive = false;
  if constexpr (kCharged) {
    // kSigmaPlus is the representative; `tests/test_species.cu` holds the other seven to its
    // structural answers.
    alive = step_hadron(scene, p, species, had, rng, em, edep, srep, traj, &queued,
                        ParticleType::kSigmaPlus);
  } else {
    alive = step_neutral_strange(scene, p, species, had, rng, em, edep, srep, traj, &queued);
  }
  if (queued) { return; }
  ++p.step;
  p.advance(srep.true_length, ekin_pre, track_mass);
  p.flags = (srep.status == StepStatus::fGeomBoundary) ? (p.flags | kFirstStepInVolume)
                                                       : (p.flags & ~kFirstStepInVolume);
  if (edep != real_t(0) && slot >= 0) {
    atomicAdd(&score[static_cast<size_t>(slot) * batch + p.event], static_cast<double>(edep));
    if (vcell >= 0) { atomicAdd(&voxel_score[vcell], static_cast<double>(edep)); }
  }
  const bool escaped = (p.volume == geom::kOutsideWorld);
  DeviceStep<real_t> ds{};
  ds.species = species;
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
// DECLARED HERE, DEFINED IN host/transport_run_int_impl.cuh - and the split is docs/RISK.md
// V221's finding. This kernel and the drain below are the only ones that carry a hadronic model,
// and while their bodies lived in this header every unit that included it - the gamma stepping
// unit included - read every model header, so a change to one cascade file rebuilt all
// twenty-three (78 minutes). What this header needs of them is the signature: the launches in
// `BeamOn`, `RaiseStackForInteractions`'s `cudaFuncGetAttributes` and the `extern template`
// block below. The seven `transport_run_int_*.cu` units include the other header and
// instantiate the bodies, so a model-header change rebuilds the units that reach it and nothing
// else (V233). The pool is a device POINTER for the same reason: a by-value parameter would
// put `had::InteractionPool`'s definition, and the models behind it, back in every launching
// translation unit.
template <typename real_t, had::InteractionBucket kBucket, typename StepHook>
__global__ void run_interaction(Scene<real_t> scene, const had::PendingInteraction<real_t>* items,
                                const int* idx, int n, int base, TrackBuffer<real_t> out,
                                int batch, double* score, double* voxel_score,
                                had::HadronicWiring<real_t> had,
                                const had::InteractionPool<real_t>* pool,
                                vis::TrajectoryBuffer traj, int* status_warn, SecondaryArena sec,
                                EmitterBooks books, StepHook hook);


// ---------------------------------------------------------------- P19: the EM-extra drain
//
/// One counter past `had::EmxStat`'s, owned by the engine rather than the steppers: the deepest
/// the queue's cursor got in any iteration of the run - pushes the capacity refused included -
/// recorded by the drain's thread 0, so that the capacity `Upload` chose is printed against a
/// measurement rather than an argument.
inline constexpr int kEmxQueueDepth = had::kNumEmxStats;
inline constexpr int kEmxCounters = had::kNumEmxStats + 1;

/// P19's drain of the photo- and lepto-nuclear queue, one bucket per instantiation. Declared
/// here and defined in host/transport_run_int_impl.cuh, with `run_interaction` and for its
/// reason; the header comment above the definition says what the kernel does and why it is not
/// `run_interaction` behind P15's host drain.
template <typename real_t, had::InteractionBucket kBucket, typename StepHook>
__global__ void run_emextra_drain(Scene<real_t> scene, const had::PendingInteraction<real_t>* items,
                                  const int* cursor, int capacity, TrackBuffer<real_t> out,
                                  int batch, double* score, double* voxel_score,
                                  had::HadronicWiring<real_t> had,
                                  const had::InteractionPool<real_t>* pool,
                                  vis::TrajectoryBuffer traj, int* status_warn,
                                  SecondaryArena sec, EmitterBooks books, StepHook hook);



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
// nothing else, so a different hook's specialisations would instantiate implicitly in the one
// translation unit that instantiates the engine for it - which is why tools/gen_hook_units.ps1
// READS this block, writes the same declarations for the project's hook type, and compiles each
// kernel in a unit of its own (build_hook_engine.bat). Every line here becomes one of those, the
// interaction kernels and the drains included since P21 (docs/RISK.md V210), so a kernel added
// here is split for a hook project the day it is added.
//
// The three utility kernels - seed_from_primaries, count_species, scatter_species - are NOT
// declared here. They are templated on real_t alone, hold no physics, and compile in seconds;
// they stay in the engine's own object, which is where their only launches are. Since P32 they
// are DEFINED in host/transport_run_engine.cuh beside BeamOn, with the two interaction-queue
// sorts, so that a change to them is a change to the loop and rebuilds the loop's unit alone.
// The four signatures, written once. Each family's .cu says `template G4GPU_STEP_ION(...);`
// against the same macro this block says `extern template G4GPU_STEP_ION(...);` against, so a
// parameter added to a kernel cannot leave a declaration and a definition disagreeing - which
// would not be a compile error, only a specialisation that stopped matching and quietly went
// back to being instantiated wherever it was launched.
// The gamma and lepton kernels' wiring is `const __grid_constant__` (see `run_step_gamma`), and
// nvcc requires the annotation on every instantiation directive as well as the definition.
#define G4GPU_STEP_GAMMA(HOOK)                                                               \
  __global__ void run_step_gamma<double, HOOK>(                                              \
      Scene<double>, TrackBuffer<double>, const int*, TrackBuffer<double>, int, int, double*, \
      double*, int, const __grid_constant__ had::HadronicWiring<double>,                     \
      vis::TrajectoryBuffer, int*, SecondaryArena, EmitterBooks, HOOK)
#define G4GPU_STEP_LEPTON(POSITRON, HOOK)                                                    \
  __global__ void run_step_lepton<double, POSITRON, HOOK>(                                   \
      Scene<double>, TrackBuffer<double>, const int*, TrackBuffer<double>, int, int, double*, \
      double*, int, const __grid_constant__ had::HadronicWiring<double>,                     \
      vis::TrajectoryBuffer, int*, SecondaryArena, EmitterBooks, HOOK)
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
// P23a's two group kernels, `run_step_strange<true>` (the eight charged strange baryons) and
// `<false>` (the eight neutral strange species), one translation unit each.
#define G4GPU_STEP_STRANGE(CHARGED, HOOK)                                                    \
  __global__ void run_step_strange<double, CHARGED, HOOK>(                                   \
      Scene<double>, TrackBuffer<double>, const int*, TrackBuffer<double>, int, int, double*, \
      double*, int, had::HadronicWiring<double>, vis::TrajectoryBuffer, int*, SecondaryArena, \
      EmitterBooks, HOOK)
// P15's interaction kernels: FIVE of them, one per model, one translation unit each
// (`host/transport_run_int_*.cu`). Those five are the only objects in the build that carry any
// hadronic model at all, and the split is not a preference - ptxas cannot compile four models
// into one module (docs/RISK.md V189; the measurement is in `had::InteractionBucket`).
#define G4GPU_INTERACTION(BUCKET, HOOK)                                                      \
  __global__ void run_interaction<double, BUCKET, HOOK>(                                     \
      Scene<double>, const had::PendingInteraction<double>*, const int*, int, int,           \
      TrackBuffer<double>, int, double*, double*, had::HadronicWiring<double>,               \
      const had::InteractionPool<double>*, vis::TrajectoryBuffer, int*, SecondaryArena,     \
      EmitterBooks, HOOK)
// P19's two drains (`host/transport_run_int_photonuclear.cu`, `..._leptonuclear.cu`): the only
// objects carrying P13's models, one model family each for V189's reason.
#define G4GPU_EMX_DRAIN(BUCKET, HOOK)                                                        \
  __global__ void run_emextra_drain<double, BUCKET, HOOK>(                                   \
      Scene<double>, const had::PendingInteraction<double>*, const int*, int,                \
      TrackBuffer<double>, int, double*, double*, had::HadronicWiring<double>,               \
      const had::InteractionPool<double>*, vis::TrajectoryBuffer, int*, SecondaryArena,     \
      EmitterBooks, HOOK)

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
extern template G4GPU_EMX_DRAIN(had::InteractionBucket::kPhotoNuclear, StepTap<double>);
extern template G4GPU_EMX_DRAIN(had::InteractionBucket::kLeptoNuclear, StepTap<double>);
extern template G4GPU_STEP_STRANGE(true, StepTap<double>);
extern template G4GPU_STEP_STRANGE(false, StepTap<double>);

}  // namespace g4gpu::host

// ---------------------------------------------------------------- the method bodies, elsewhere
//
// SINCE P32 THE ENGINE'S HOST CODE IS host/transport_run_engine.cuh, included here - so every
// project that includes this header and instantiates `TransportEngine` for its own hook gets the
// whole class, exactly as before - and SKIPPED by the engine's own kernel units, which define
// G4GPU_ENGINE_KERNELS_ONLY before including this file. They instantiate kernels and nothing
// else, so they never needed `Upload`, the stack raise or `BeamOn`; what they did need was not to
// be rebuilt when one of those changed. nvcc's -MD dependency file lists only the headers a unit
// actually read, so with the include skipped `tools/freshness.ps1` keeps the 26 kernel units'
// objects and a change to the loop rebuilds `transport_run.cu` alone - minutes, where the same
// change rebuilt all 27 units (85 minutes at one target) when the loop lived in this file.
// docs/RISK.md V283 has the measurement. The performance phase is a sequence of loop changes,
// each checked to the digit against the eleven dose gates, which is why it came first.
#ifndef G4GPU_ENGINE_KERNELS_ONLY
#include "host/transport_run_engine.cuh"
#endif
