// Structure-of-arrays track buffer for the track-parallel scheduler.
//
// Tracks from many events are in flight at once; the event id is a *tag* on each track,
// never a scheduling unit. Buffers ping-pong: a kernel reads `in` and appends surviving
// tracks plus new secondaries to `out`, which compacts implicitly - dead tracks are simply
// never written, so no prefix scan is needed.
//
// RNG streams need no storage. A track's stream is keyed on
// (event id, slot in the current buffer, iteration), which is unique because slots are
// unique within a buffer and the iteration disambiguates reuse across ping-pongs. That
// gives every track a fresh independent stream each step for the cost of one int.
#pragma once
#include <cuda_runtime.h>

#include "core/particle.cuh"
#include "core/step_report.cuh"
#include "core/units.cuh"
#include "core/vec3.cuh"

namespace g4gpu {

/// Whether the G4Track block is carried on every track.
///
/// ON by default. It is not free and the number is known: the block - parent, three clocks,
/// track length, vertex, creator process, weight, polarization, user data - is 136 of the 236
/// bytes a slot costs, and a track is loaded and stored on every step of every iteration.
/// Measured interleaved against a build without it, five runs each way with no overlap
/// between the two sets, it costs 7.6% of throughput on B1 - about 2.30 to 2.13 million
/// events a second - and is paid by every run whether a hook reads any of it or not.
///
/// It is on anyway, because the point of this port is to answer what Geant4 answers. A
/// GetParentID() that does not work is a bigger defect than an 8% slower run.
///
/// A project that wants the speed back and does not need the accessors builds with
/// -DG4GPU_FULL_TRACK_STATE=0; every accessor behind it then becomes a compile error naming
/// this flag, rather than silently returning a stale value. TrackState::status and ::flags
/// are outside the switch and always present - killing a track from a stepping action and
/// knowing whether a step is the first in its volume are read by the kernel itself.
///
/// What the block is, and what it weighs: parent, three clocks, track length, vertex, creator
/// process, weight, polarization, user data - 136 bytes, taking a slot from 100 to 236. It was
/// 96 to 232 on the day the throughput was measured; the four bytes since are the species
/// field the pooled scheduler added. A track is loaded and stored on every step of every
/// iteration: 8.0% of throughput on B1, interleaved against a build without it, with no
/// overlap between the two sets of runs.
///
/// The part of that number worth keeping in view is who pays it: every run, including every
/// run whose stepping action never looks at any of it. That was the argument for defaulting it
/// off, and it lost to the one above - a port exists to give Geant4's answers - but it is the
/// argument to weigh again for anything added to TrackState later. Per-step state that only
/// some hooks read does not have to be carried on the track: see the note further down about
/// keying it by track id instead.
///
/// What is NOT behind this switch, and stays on always: TrackState::status and ::flags, 8
/// bytes between them. Killing a track from a stepping action and knowing whether a step is
/// the first in its volume are worth having in every build, and neither is an accessor onto
/// stored history - they are read by the kernel itself.
#ifndef G4GPU_FULL_TRACK_STATE
#define G4GPU_FULL_TRACK_STATE 1
#endif
constexpr bool kFullTrackState = (G4GPU_FULL_TRACK_STATE != 0);

/// The species that have a stepping kernel, as a dense index for the dispatch lists.
///
/// This is the ONE place a species list still exists, and it is now about dispatch rather
/// than storage: a track can only be stepped by a kernel that was compiled for it, so the
/// set of kernels is the set of species, and no arrangement of buffers changes that. What
/// used to be here as well - five buffers, five capacities to guess, five branches in every
/// router, a silent drop for anything not on the list - has gone. Tracks live in one pool
/// and carry what they are.
///
/// Adding a species is now: an entry here, and a kernel instantiated for it. Nothing else.
enum TrackSpeciesIndex : int {
  kSpeciesGamma = 0,
  kSpeciesElectron = 1,
  kSpeciesPositron = 2,
  kSpeciesProton = 3,
  kSpeciesAlpha = 4,
  kNumTrackSpecies = 5,
};

/// -1 for a particle no kernel steps. The caller decides what that means; nothing here
/// quietly routes it somewhere plausible.
__host__ __device__ inline int species_index(ParticleType t) {
  switch (t) {
    case ParticleType::kGamma:    return kSpeciesGamma;
    case ParticleType::kElectron: return kSpeciesElectron;
    case ParticleType::kPositron: return kSpeciesPositron;
    case ParticleType::kProton:   return kSpeciesProton;
    case ParticleType::kAlpha:    return kSpeciesAlpha;
    default:                      return -1;
  }
}

/// Bits in TrackState::flags.
enum TrackFlag : unsigned int {
  /// The previous step of this track ended on a boundary, so this step starts on one.
  /// G4Step::IsFirstStepInVolume reads it.
  kFirstStepInVolume = 1u << 0,
  /// G4Track::IsBelowThreshold. Set by nothing in this transport yet; a stepping action may
  /// set it, and it is then carried across steps like any other track state.
  kBelowThreshold = 1u << 1,
  /// G4Track::IsGoodForTracking. Same.
  kGoodForTracking = 1u << 2,
};

/// One track in flight.
///
/// The first block is what transport needs. The second is G4Track state - provenance, timing,
/// and the vertex the track was created at - which no physics here reads, but which
/// G4VUserDeviceSteppingAction exposes through DeviceStep::GetTrack(). It is carried because
/// the alternative is accessors that return zero, and a zero meaning "not implemented" is
/// indistinguishable from a zero meaning zero.
///
/// It is not free: this struct is loaded and stored for every track on every step, so the
/// second block costs bandwidth in every run whether or not a hook reads it. What that costs
/// is measured rather than asserted - see docs/RESULT.md.
template <typename real_t>
struct TrackState {
  /// What this track is.
  ///
  /// The pool holds every species together, so a track has to say which it is. It used to be
  /// inferable from WHICH buffer a track sat in - five buffers, one per species - and that is
  /// what tied the storage layout to the physics list: adding a neutron meant a sixth buffer,
  /// a sixth capacity to guess, and a sixth branch in every router. Four bytes here buys all
  /// of that back.
  ///
  /// Dispatch still runs one specialised kernel per species, because a warp whose threads take
  /// different physics paths serialises through all of them. That needs each species to be
  /// CONTIGUOUS when a kernel launches, which is what the index lists provide - not separate
  /// storage, which is what it used to be conflated with.
  ParticleType species;
  Vec3<real_t> pos;
  Vec3<real_t> dir;
  real_t ekin;
  int volume;
  int event;
  /// Deterministic per-track RNG key. A primary uses its global event id; a secondary gets
  /// child_rng_key(parent). Never derived from a buffer slot, so the random stream does not
  /// depend on atomic ordering.
  unsigned int rng_key;
  /// Steps taken so far. Combined with rng_key it gives a fresh stream every step.
  unsigned int step;
  /// Frozen Urban MSC step base, fr * rangeinit. Geant4 recomputes this only on the first
  /// step of a track and after a boundary crossing, then holds it; refreshing it every step
  /// shortens the steps and, through the ln(y) term in theta0, under-scatters the track.
  /// Zero means stale - recompute it this step.
  real_t msc_tlimit;
  /// Frozen alongside it: Geant4 recomputes tlimitmin at the same points.
  real_t msc_tlimitmin;

  // ---------------------------------------------------------------- G4Track state

  /// G4Track::GetParentID. The parent's rng_key, or 0 for a primary - a key rather than
  /// Geant4's small dense integer, for the same reason GetTrackID() is.
  unsigned int parent_key;
  /// G4Track::GetGlobalTime, ns. Time since the event began. A secondary inherits its
  /// parent's value at the moment it was created.
  real_t global_time;
  /// G4Track::GetLocalTime, ns. Time since THIS track began, so a secondary starts at zero.
  real_t local_time;
  /// G4Track::GetProperTime, ns. Time in the particle's own rest frame: dt/gamma.
  real_t proper_time;
  /// G4Track::GetTrackLength, mm. The true path summed over every step so far.
  real_t track_length;
  /// G4Track::GetVertexPosition - where this track was created.
  Vec3<real_t> vertex_pos;
  /// G4Track::GetVertexMomentumDirection.
  Vec3<real_t> vertex_dir;
  /// G4Track::GetVertexKineticEnergy.
  real_t vertex_ekin;
  /// G4Track::GetLogicalVolumeAtVertex, as a volume index.
  int vertex_volume;
  /// G4Track::GetCreatorProcess - which process made this track. fNotDefined for a primary,
  /// which is what Geant4 reports too, as a null pointer.
  ProcessId creator_process;
  /// G4Track::GetWeight. There is no variance reduction here so nothing changes it, but a
  /// stepping action may, and it is then carried and inherited like any other track state.
  real_t weight;
  /// G4Track::GetPolarization. No process in this transport produces or consumes it - there
  /// is no polarised Compton and there are no optical photons - so it is user-owned state: a
  /// stepping action can set it and read it back on later steps.
  Vec3<real_t> polarization;
  /// G4Track::GetTrackStatus. A stepping action sets it; the kernel reads it back after the
  /// hook returns, to decide whether to requeue the track.
  TrackStatus status;
  /// TrackFlag bits.
  unsigned int flags;
  /// The device answer to G4VUserTrackInformation. That is a pointer to a host object and
  /// cannot exist here, but the capability - attach your own data to a track and get it back
  /// on the next step - is meaningful, so what is offered is a POD slot. Nothing in the
  /// engine reads or writes it.
  unsigned int user_data;

  /// Fills the G4Track block for a track that starts now.
  ///
  /// Used for primaries (parent 0, fNotDefined creator, whatever the gun's t0 is) and for
  /// secondaries (the parent's key, clock and creating process). One function for both is the
  /// point: a secondary that forgot to inherit its parent's clock is a bug no test of the
  /// energy would ever see.
  __host__ __device__ void begin(const Vec3<real_t>& where, const Vec3<real_t>& direction,
                                 real_t kinetic, int vol, unsigned int parent,
                                 ProcessId creator, real_t t_global, real_t w) {
    status = TrackStatus::fAlive;
    flags = 0u;
    if (!kFullTrackState) { return; }
    parent_key = parent;
    global_time = t_global;
    local_time = real_t(0);
    proper_time = real_t(0);
    track_length = real_t(0);
    vertex_pos = where;
    vertex_dir = direction;
    vertex_ekin = kinetic;
    vertex_volume = vol;
    creator_process = creator;
    weight = w;
    polarization = Vec3<real_t>{0, 0, 0};
    user_data = 0u;
  }

  /// Advances the three clocks and the path length over one step.
  ///
  /// Transcribed from G4Transportation::AlongStepDoIt, which is where Geant4 advances a
  /// track's clocks, rather than derived from what looks reasonable:
  ///
  ///     initialVelocity = stepData.GetPreStepPoint()->GetVelocity();
  ///     if (initialVelocity > 0.0) deltaTime = stepLength / initialVelocity;
  ///     fCandidateEndGlobalTime = startTime + deltaTime;
  ///     ProposeLocalTime(track.GetLocalTime() + deltaTime);
  ///     deltaProperTime = deltaTime * (restMass / track.GetTotalEnergy());
  ///
  /// Three things in that would be easy to get wrong by guessing, and all three matter: the
  /// velocity is the PRE-step one and not a mean over the step; `stepLength` is the true path
  /// length, so multiple scattering is already accounted for; and the proper time is dt*m/E
  /// against the pre-step energy, which for a massless particle is exactly zero rather than a
  /// division by zero.
  ///
  /// The velocity itself is G4Track::GetVelocity - `c_light * fpDynamicParticle->GetBeta()`.
  ///
  /// @param length    mm, the true path length of the step
  /// @param ekin_pre  MeV, the kinetic energy the track had ENTERING the step
  /// @param mass      MeV, the particle's rest mass
  __host__ __device__ void advance(real_t length, real_t ekin_pre, real_t mass) {
    if (!kFullTrackState) { return; }
    track_length += length;
    const real_t velocity = units::c_light<real_t>() * dynamic_particle_beta(ekin_pre, mass);
    if (velocity <= real_t(0)) { return; }
    const real_t dt = length / velocity;
    global_time += dt;
    local_time += dt;
    const real_t e_total = ekin_pre + mass;
    if (e_total > real_t(0)) { proper_time += dt * (mass / e_total); }
  }
};

/// Structure-of-arrays storage for one species' tracks.
template <typename real_t>
struct TrackBuffer {
  int* species;
  real_t* x;
  real_t* y;
  real_t* z;
  real_t* dx;
  real_t* dy;
  real_t* dz;
  real_t* ekin;
  int* volume;
  int* event;
  unsigned int* rng_key;
  unsigned int* step;
  real_t* msc_tlimit;
  real_t* msc_tlimitmin;
  // G4Track state; see TrackState.
  unsigned int* parent_key;
  real_t* global_time;
  real_t* local_time;
  real_t* proper_time;
  real_t* track_length;
  real_t* vx;
  real_t* vy;
  real_t* vz;
  real_t* vdx;
  real_t* vdy;
  real_t* vdz;
  real_t* vertex_ekin;
  int* vertex_volume;
  int* creator_process;
  real_t* weight;
  real_t* polx;
  real_t* poly;
  real_t* polz;
  int* status;
  unsigned int* flags;
  unsigned int* user_data;

  int capacity;
  int* count;      ///< device-resident append cursor
  int* overflow;   ///< tracks dropped because the buffer filled; never silent

  /// Which EVENTS lost a track, so that only those need transporting again.
  ///
  /// A dropped track carries the id of the event it belongs to, and that is the whole
  /// difference between redoing a million events and redoing three. Without it the only honest
  /// response to an overflow is to discard the batch, because there is no way to tell which
  /// events were short-changed - and discarding a batch to repair one shower is a poor trade.
  ///
  /// Shared by every buffer and owned by the engine, not carved from the slab: a re-carve would
  /// walk over it, which is a mistake this file has already made once with the counters.
  /// Null disables recording. If more events are lost than the list can hold, the count runs
  /// past its capacity and the engine falls back to redoing the batch - the list being
  /// incomplete is itself the signal.
  int* dropped_event = nullptr;
  int* dropped_count = nullptr;
  int dropped_capacity = 0;

  __host__ __device__ void load(int i, TrackState<real_t>& t) const {
    t.species = static_cast<ParticleType>(species[i]);
    t.pos = Vec3<real_t>{x[i], y[i], z[i]};
    t.dir = Vec3<real_t>{dx[i], dy[i], dz[i]};
    t.ekin = ekin[i];
    t.volume = volume[i];
    t.event = event[i];
    t.rng_key = rng_key[i];
    t.step = step[i];
    t.msc_tlimit = msc_tlimit[i];
    t.msc_tlimitmin = msc_tlimitmin[i];
    t.status = static_cast<TrackStatus>(status[i]);
    t.flags = flags[i];
    if (!kFullTrackState) { return; }
    t.parent_key = parent_key[i];
    t.global_time = global_time[i];
    t.local_time = local_time[i];
    t.proper_time = proper_time[i];
    t.track_length = track_length[i];
    t.vertex_pos = Vec3<real_t>{vx[i], vy[i], vz[i]};
    t.vertex_dir = Vec3<real_t>{vdx[i], vdy[i], vdz[i]};
    t.vertex_ekin = vertex_ekin[i];
    t.vertex_volume = vertex_volume[i];
    t.creator_process = static_cast<ProcessId>(creator_process[i]);
    t.weight = weight[i];
    t.polarization = Vec3<real_t>{polx[i], poly[i], polz[i]};
    t.user_data = user_data[i];
  }

  __device__ int append(const TrackState<real_t>& t) {
    const int slot = atomicAdd(count, 1);
    if (slot >= capacity) {
      atomicAdd(overflow, 1);
      if (dropped_event != nullptr) {
        const int at = atomicAdd(dropped_count, 1);
        // Past the end is not written, but the count still advances - the engine compares it
        // against the capacity to know the list is a partial record rather than a complete one.
        if (at < dropped_capacity) { dropped_event[at] = t.event; }
      }
      return -1;
    }
    species[slot] = static_cast<int>(t.species);
    x[slot] = t.pos.x;   y[slot] = t.pos.y;   z[slot] = t.pos.z;
    dx[slot] = t.dir.x;  dy[slot] = t.dir.y;  dz[slot] = t.dir.z;
    ekin[slot] = t.ekin;
    volume[slot] = t.volume;
    event[slot] = t.event;
    rng_key[slot] = t.rng_key;
    step[slot] = t.step;
    msc_tlimit[slot] = t.msc_tlimit;
    msc_tlimitmin[slot] = t.msc_tlimitmin;
    status[slot] = static_cast<int>(t.status);
    flags[slot] = t.flags;
    if (!kFullTrackState) { return slot; }
    parent_key[slot] = t.parent_key;
    global_time[slot] = t.global_time;
    local_time[slot] = t.local_time;
    proper_time[slot] = t.proper_time;
    track_length[slot] = t.track_length;
    vx[slot] = t.vertex_pos.x;   vy[slot] = t.vertex_pos.y;   vz[slot] = t.vertex_pos.z;
    vdx[slot] = t.vertex_dir.x;  vdy[slot] = t.vertex_dir.y;  vdz[slot] = t.vertex_dir.z;
    vertex_ekin[slot] = t.vertex_ekin;
    vertex_volume[slot] = t.vertex_volume;
    creator_process[slot] = static_cast<int>(t.creator_process);
    weight[slot] = t.weight;
    polx[slot] = t.polarization.x;
    poly[slot] = t.polarization.y;
    polz[slot] = t.polarization.z;
    user_data[slot] = t.user_data;
    return slot;
  }
};

/// Allocates every array in a TrackBuffer, and frees them.
///
/// These live next to the struct because the alternative was two copies of the list, and the
/// second copy went stale the moment the G4Track block was added: b1_gpu_sched.cu had its own
/// allocator, the fifteen new arrays stayed null in it, and load() dereferenced a null pointer
/// on the first track of the first step. Nothing could have caught that but running it - which
/// is exactly what a duplicated list buys you.
///
/// A field added to TrackBuffer now needs one edit, here, or it fails to compile rather than
/// failing at run time in whichever caller was forgotten.
/// Carves aligned sub-ranges out of one device allocation.
///
/// The species buffers used to be five independent cudaMalloc sets with capacities fixed in
/// advance, which is why those capacities had to be guessed: memory handed to gammas could not
/// be used by electrons however the shower actually turned out. Separate ALLOCATIONS were never
/// what the kernels needed - what they need is for each species to occupy a contiguous RANGE,
/// so that one thread per track runs identical physics with no type divergence in a warp. Those
/// two things had simply been tied together.
///
/// A slab separates them. One allocation, ranges carved from it, warp coherence unchanged - and
/// the split becomes a decision that can be revisited between iterations rather than a constant
/// chosen before the run.
///
/// The alignment is CUDA's own: cudaMalloc returns 256-byte-aligned memory, and every sub-range
/// here is aligned the same way, so a carved buffer's loads coalesce exactly as a separately
/// allocated one's did. That matters - this refactor is required to leave the dose bit-identical
/// and the throughput unmoved, and an alignment change would quietly break the second half.
struct TrackSlab {
  char* base = nullptr;
  size_t used = 0;
  size_t capacity = 0;
  bool overflowed = false;

  static constexpr size_t kAlign = 256;
  static __host__ __device__ size_t align_up(size_t n) {
    return (n + kAlign - 1) / kAlign * kAlign;
  }

  void* take(size_t n) {
    const size_t at = used;
    used += align_up(n);
    if (used > capacity) {
      overflowed = true;
      return nullptr;
    }
    return base + at;
  }
};

/// Allocates a TrackBuffer.
///
/// Three modes, one field list. `dry` non-null measures what would be allocated - including the
/// alignment padding, so a measurement and a carve agree to the byte. `slab` non-null carves
/// from a shared allocation. Neither means one cudaMalloc per array, which is what this did
/// before the slab existed and what b1_gpu_sched.cu still does.
template <typename real_t>
inline cudaError_t allocate_track_buffer(TrackBuffer<real_t>& v, int capacity,
                                         size_t* dry = nullptr, TrackSlab* slab = nullptr) {
  v.capacity = capacity;
  const size_t nr = sizeof(real_t) * capacity;
  const size_t ni = sizeof(int) * capacity;
  const size_t nu = sizeof(unsigned int) * capacity;
  cudaError_t e = cudaSuccess;
  auto get = [&](void** p, size_t n) {
    if (dry != nullptr) {
      *dry += TrackSlab::align_up(n);
      return;
    }
    if (slab != nullptr) {
      *p = slab->take(n);
      if (*p == nullptr) { e = cudaErrorMemoryAllocation; }
      return;
    }
    if (e == cudaSuccess) { e = cudaMalloc(p, n); }
  };
  get(reinterpret_cast<void**>(&v.species), ni);
  get(reinterpret_cast<void**>(&v.x), nr);
  get(reinterpret_cast<void**>(&v.y), nr);
  get(reinterpret_cast<void**>(&v.z), nr);
  get(reinterpret_cast<void**>(&v.dx), nr);
  get(reinterpret_cast<void**>(&v.dy), nr);
  get(reinterpret_cast<void**>(&v.dz), nr);
  get(reinterpret_cast<void**>(&v.ekin), nr);
  get(reinterpret_cast<void**>(&v.volume), ni);
  get(reinterpret_cast<void**>(&v.event), ni);
  get(reinterpret_cast<void**>(&v.rng_key), nu);
  get(reinterpret_cast<void**>(&v.step), nu);
  get(reinterpret_cast<void**>(&v.msc_tlimit), nr);
  get(reinterpret_cast<void**>(&v.msc_tlimitmin), nr);
  get(reinterpret_cast<void**>(&v.status), ni);
  get(reinterpret_cast<void**>(&v.flags), nu);
  if (!kFullTrackState) {
    get(reinterpret_cast<void**>(&v.count), sizeof(int));
    get(reinterpret_cast<void**>(&v.overflow), sizeof(int));
    return e;
  }
  get(reinterpret_cast<void**>(&v.parent_key), nu);
  get(reinterpret_cast<void**>(&v.global_time), nr);
  get(reinterpret_cast<void**>(&v.local_time), nr);
  get(reinterpret_cast<void**>(&v.proper_time), nr);
  get(reinterpret_cast<void**>(&v.track_length), nr);
  get(reinterpret_cast<void**>(&v.vx), nr);
  get(reinterpret_cast<void**>(&v.vy), nr);
  get(reinterpret_cast<void**>(&v.vz), nr);
  get(reinterpret_cast<void**>(&v.vdx), nr);
  get(reinterpret_cast<void**>(&v.vdy), nr);
  get(reinterpret_cast<void**>(&v.vdz), nr);
  get(reinterpret_cast<void**>(&v.vertex_ekin), nr);
  get(reinterpret_cast<void**>(&v.vertex_volume), ni);
  get(reinterpret_cast<void**>(&v.creator_process), ni);
  get(reinterpret_cast<void**>(&v.weight), nr);
  get(reinterpret_cast<void**>(&v.polx), nr);
  get(reinterpret_cast<void**>(&v.poly), nr);
  get(reinterpret_cast<void**>(&v.polz), nr);
  get(reinterpret_cast<void**>(&v.user_data), nu);
  get(reinterpret_cast<void**>(&v.count), sizeof(int));
  get(reinterpret_cast<void**>(&v.overflow), sizeof(int));
  return e;
}

/// Device bytes one track buffer of `capacity` slots occupies, measured by walking the
/// allocator itself.
template <typename real_t>
inline size_t track_buffer_bytes(int capacity) {
  TrackBuffer<real_t> probe{};
  size_t bytes = 0;
  allocate_track_buffer<real_t>(probe, capacity, &bytes);
  return bytes;
}

/// Device bytes ONE more track slot costs.
///
/// The difference between two capacities rather than the size at capacity 1, because a
/// buffer also allocates its count and overflow counters - eight bytes that do not scale
/// with capacity. Reading them as part of a slot over-counted every slot by 8 of 236 bytes,
/// which made the automatic batch about 3.5% smaller than it needed to be. Conservative, so
/// nothing broke; wrong, so it is subtracted here rather than left as a fudge factor.
template <typename real_t>
inline size_t track_bytes_per_slot() {
  // Measured at a LARGE capacity, and this matters. Taking the difference between one slot
  // and two returns very nearly zero: every array is aligned up to 256 bytes, so an eight
  // byte array and a sixteen byte array occupy the same 256 and the difference vanishes.
  // That is not a rounding error, it is the whole quantity - and it under-sized the track
  // arena until a re-carve ran out of room mid-run.
  //
  // At 65536 slots every array is an exact multiple of the alignment, so align_up is the
  // identity and the difference is exactly per_slot * 65536.
  constexpr int kN = 65536;
  return (track_buffer_bytes<real_t>(2 * kN) - track_buffer_bytes<real_t>(kN)) / kN;
}

/// Device bytes one half of the arena needs to hold `pool` live-track slots.
///
/// EXACT, not estimated: it dry-runs the same allocator the carve then walks, so the number
/// that sizes the arena and the number that is consumed out of it are produced by the same
/// code and agree to the byte. It used to be per-slot times the pool plus a slack term sized
/// for five buffers, which is what the arena held before one pool replaced them; with one
/// buffer per half there is nothing left to be slack about.
///
/// That also makes the alignment structural rather than lucky. Every array is taken with
/// align_up, so a sum of them is already a multiple of kAlign - which matters because the
/// engine allocates both halves as one block and puts the second at `base + this`, making it
/// the alignment of every array in the second half. The estimate it replaces was
/// `236 * pool + slack`, and 236 is not a multiple of 8, so an ODD pool put the whole second
/// half at 4 mod 8 and every double in it was misaligned: a device-side "misaligned address"
/// from whichever kernel touched it first, reported at the next cudaMemcpy with no hint of
/// where it came from. An odd pool is not exotic - `batch * live_per_event` gives one whenever
/// the product is not whole, so `-live 2.5` reached it and the default `-live 4` never could.
/// The align_up below is now belt and braces; tests/test_track_arena.cu asserts both that it
/// holds and that this equals what a carve consumes. See docs/RISK.md V11.
///
/// A pool larger than an int can hold reports an impossible size rather than a truncated one.
/// A capacity is an int everywhere - in TrackBuffer, in allocate_track_buffer, in the carve
/// this measures - so the honest answer for a pool that does not fit one is "this does not
/// fit", and the batch sizer bisecting on this number then walks down until it does.
///
/// Saturating rather than truncating is the whole point. The estimate this replaced computed
/// `per_slot * pool` in size_t, so an over-large pool came out enormous and memory refused it;
/// an exact measurement of a TRUNCATED capacity would instead agree with an equally truncated
/// carve, and the run would proceed quietly with a pool that is not the one it was asked for.
/// Being exact is only an improvement where it does not buy silence.
template <typename real_t>
inline size_t track_arena_half_bytes(long long pool) {
  if (pool < 0 || pool > static_cast<long long>(2147483647)) {
    return static_cast<size_t>(-1) / 4;  // /4 so that doubling it cannot wrap
  }
  return TrackSlab::align_up(track_buffer_bytes<real_t>(static_cast<int>(pool)));
}

template <typename real_t>
inline void free_track_buffer(TrackBuffer<real_t>& v) {
  cudaFree(v.species);
  cudaFree(v.x); cudaFree(v.y); cudaFree(v.z);
  cudaFree(v.dx); cudaFree(v.dy); cudaFree(v.dz);
  cudaFree(v.ekin); cudaFree(v.volume); cudaFree(v.event);
  cudaFree(v.rng_key); cudaFree(v.step);
  cudaFree(v.msc_tlimit); cudaFree(v.msc_tlimitmin);
  cudaFree(v.parent_key); cudaFree(v.global_time); cudaFree(v.local_time);
  cudaFree(v.proper_time); cudaFree(v.track_length);
  cudaFree(v.vx); cudaFree(v.vy); cudaFree(v.vz);
  cudaFree(v.vdx); cudaFree(v.vdy); cudaFree(v.vdz);
  cudaFree(v.vertex_ekin); cudaFree(v.vertex_volume); cudaFree(v.creator_process);
  cudaFree(v.weight); cudaFree(v.polx); cudaFree(v.poly); cudaFree(v.polz);
  cudaFree(v.status); cudaFree(v.flags); cudaFree(v.user_data);
  cudaFree(v.count); cudaFree(v.overflow);
  v = TrackBuffer<real_t>{};
}

/// Fills the G4Track block of one slot for a track that is starting there.
///
/// The seeding kernels write TrackBuffer arrays directly rather than through a TrackState, so
/// this is the shared version of what TrackState::begin does - for the same reason the
/// allocator above is shared.
template <typename real_t>
__host__ __device__ inline void seed_track_slot(TrackBuffer<real_t>& v, int slot,
                                                const Vec3<real_t>& pos,
                                                const Vec3<real_t>& dir, real_t ekin, int vol,
                                                real_t t0) {
  v.status[slot] = static_cast<int>(TrackStatus::fAlive);
  // A primary begins its life inside a volume, so its first step is the first in that volume.
  v.flags[slot] = kFirstStepInVolume;
  if (!kFullTrackState) { return; }
  v.parent_key[slot] = 0u;
  v.global_time[slot] = t0;
  v.local_time[slot] = real_t(0);
  v.proper_time[slot] = real_t(0);
  v.track_length[slot] = real_t(0);
  v.vx[slot] = pos.x;   v.vy[slot] = pos.y;   v.vz[slot] = pos.z;
  v.vdx[slot] = dir.x;  v.vdy[slot] = dir.y;  v.vdz[slot] = dir.z;
  v.vertex_ekin[slot] = ekin;
  v.vertex_volume[slot] = vol;
  v.creator_process[slot] = static_cast<int>(ProcessId::fNotDefined);
  v.weight[slot] = real_t(1);
  v.polx[slot] = real_t(0);
  v.poly[slot] = real_t(0);
  v.polz[slot] = real_t(0);
  v.user_data[slot] = 0u;
}

/// Where the secondaries created during a step are recorded, so that
/// G4Step::GetSecondaryInCurrentStep() can hand back the tracks themselves.
///
/// THERE IS NO PER-STEP LIMIT, which is the whole design constraint. The obvious
/// implementation - a fixed array on the emitter - puts a cap on how many secondaries one step
/// may make, and any number chosen for that cap is a claim about physics that has not happened
/// yet. Today nothing here makes more than three; a hadronic inelastic interaction makes
/// dozens, and the cap would be discovered by someone losing data.
///
/// So each secondary instead takes one slot from a shared arena and points at the previous
/// slot from the same step. A step's secondaries are a backward-linked chain, and a step can
/// have as many as it likes. What is bounded is the arena as a whole - the total across every
/// track in flight in one kernel launch - which is a resource, sized once and reported when
/// exhausted, rather than a statement about what a process is allowed to do.
///
/// A slot holds four bytes of payload, not a copy of the track: the species buffer the
/// secondary landed in, and its index there. Reading a secondary therefore reads the real
/// track out of the real buffer - it IS the G4Track, the way Geant4's `const G4Track*` is -
/// and costs nothing at all for a hook that never asks.
struct SecondaryArena {
  /// Encoded (buffer id, slot) for each recorded secondary.
  unsigned int* entry = nullptr;
  /// Index of the previous secondary of the SAME step, or -1. This is the chain.
  int* prev = nullptr;
  /// Bump cursor, reset once per kernel launch.
  int* cursor = nullptr;
  /// Secondaries that did not fit. Never silent.
  int* overflow = nullptr;
  int capacity = 0;

  __host__ __device__ static unsigned int encode(int buffer_id, int slot) {
    return (static_cast<unsigned int>(buffer_id) << 30) | static_cast<unsigned int>(slot);
  }
  __host__ __device__ static int buffer_of(unsigned int e) { return static_cast<int>(e >> 30); }
  __host__ __device__ static int slot_of(unsigned int e) {
    return static_cast<int>(e & 0x3FFFFFFFu);
  }
};

/// Adapter giving the physics models the push() signature they already expect.
///
/// ONE POOL, NOT ONE BUFFER PER SPECIES. This used to route each secondary into its own
/// species' buffer with a switch, and the switch had a `default: return -1` at the bottom -
/// a silent drop for any particle the transport had no buffer for. The comment above it said
/// such a case "should fail to compile rather than be routed somewhere plausible", which is
/// what was wanted but not what the code did: it was a runtime discard waiting for the first
/// process that made a neutron.
///
/// There is nothing to route now. A track carries its species, the pool takes anything, and a
/// process that starts producing a new particle needs no change here at all. Which kernel
/// eventually steps it is decided later, by the index lists, and that is a dispatch question
/// rather than a storage one.
template <typename real_t>
struct BufferEmitter {
  /// Where every secondary goes, whatever it is.
  TrackBuffer<real_t> out;
  Vec3<real_t> pos;
  int volume;
  int event;
  /// Parent identity, so each secondary can be given a deterministic RNG key.
  unsigned int parent_key;
  unsigned int parent_step;
  /// Secondaries pushed so far in this step. Counted per emitter, which lives for exactly
  /// one step of one track, so it is a deterministic child index - not a shared atomic.
  unsigned int child_count;

  /// The parent's clock and weight, so a secondary starts where its parent is rather than at
  /// zero. Set once in the kernel, alongside the buffer.
  real_t parent_time;
  real_t parent_weight;
  /// Where to record each secondary, and the last slot this step took. A null arena disables
  /// recording; `last_secondary` is then never anything but -1.
  SecondaryArena arena;
  int last_secondary = -1;
  /// The step in progress, so push() can record which process created the secondary. Points at
  /// the caller's StepReport, which outlives every push within the step.
  const StepReport<real_t>* report;

  __device__ int push(ParticleType type, const Vec3<real_t>& dir, real_t ekin,
                      int /*event_id*/) {
    TrackState<real_t> t{};
    t.species = type;
    t.pos = pos;
    t.dir = dir;
    t.ekin = ekin;
    t.volume = volume;
    t.event = event;
    t.msc_tlimit = real_t(0);
    t.msc_tlimitmin = real_t(0);
    t.rng_key = child_rng_key(parent_key, parent_step, child_count++);
    t.step = 0u;
    t.begin(pos, dir, ekin, volume, parent_key,
            (report != nullptr) ? report->process : ProcessId::fNotDefined, parent_time,
            parent_weight);
    const int slot = out.append(t);
    if (slot < 0) { return slot; }
    if (arena.entry != nullptr) {
      const int at = atomicAdd(arena.cursor, 1);
      if (at < arena.capacity) {
        arena.entry[at] = SecondaryArena::encode(0, slot);
        arena.prev[at] = last_secondary;
        last_secondary = at;
      } else {
        atomicAdd(arena.overflow, 1);
      }
    }
    return slot;
  }
};

}  // namespace g4gpu

