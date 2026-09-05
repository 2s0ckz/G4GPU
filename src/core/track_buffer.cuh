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
#include "core/particle.cuh"
#include "core/vec3.cuh"

namespace g4gpu {

template <typename real_t>
struct TrackState {
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
};

/// 88 bytes per track in double precision: 9 reals, two ints, two uints.
template <typename real_t>
struct TrackBuffer {
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
  int capacity;
  int* count;      ///< device-resident append cursor
  int* overflow;   ///< tracks dropped because the buffer filled; never silent

  __host__ __device__ void load(int i, TrackState<real_t>& t) const {
    t.pos = Vec3<real_t>{x[i], y[i], z[i]};
    t.dir = Vec3<real_t>{dx[i], dy[i], dz[i]};
    t.ekin = ekin[i];
    t.volume = volume[i];
    t.event = event[i];
    t.rng_key = rng_key[i];
    t.step = step[i];
    t.msc_tlimit = msc_tlimit[i];
    t.msc_tlimitmin = msc_tlimitmin[i];
  }

  __device__ int append(const TrackState<real_t>& t) {
    const int slot = atomicAdd(count, 1);
    if (slot >= capacity) {
      atomicAdd(overflow, 1);
      return -1;
    }
    x[slot] = t.pos.x;   y[slot] = t.pos.y;   z[slot] = t.pos.z;
    dx[slot] = t.dir.x;  dy[slot] = t.dir.y;  dz[slot] = t.dir.z;
    ekin[slot] = t.ekin;
    volume[slot] = t.volume;
    event[slot] = t.event;
    rng_key[slot] = t.rng_key;
    step[slot] = t.step;
    msc_tlimit[slot] = t.msc_tlimit;
    msc_tlimitmin[slot] = t.msc_tlimitmin;
    return slot;
  }
};

/// Adapter giving the physics models the push() signature they already expect, while
/// routing each species into its own buffer.
///
/// One buffer per species, rather than a shared buffer with a type field, for two reasons:
/// every thread in a kernel then runs identical physics (no type divergence within a warp),
/// and the type need not be stored per track at all.
template <typename real_t>
struct BufferEmitter {
  TrackBuffer<real_t> gamma_out;
  TrackBuffer<real_t> electron_out;
  TrackBuffer<real_t> positron_out;
  Vec3<real_t> pos;
  int volume;
  int event;
  /// Parent identity, so each secondary can be given a deterministic RNG key.
  unsigned int parent_key;
  unsigned int parent_step;
  /// Secondaries pushed so far in this step. Counted per emitter, which lives for exactly
  /// one step of one track, so it is a deterministic child index - not a shared atomic.
  unsigned int child_count;

  __device__ int push(ParticleType type, const Vec3<real_t>& dir, real_t ekin,
                      int /*event_id*/) {
    TrackState<real_t> t{pos, dir, ekin, volume, event, 0u, 0u, real_t(0), real_t(0)};
    t.rng_key = child_rng_key(parent_key, parent_step, child_count++);
    t.step = 0u;
    switch (type) {
      case ParticleType::kGamma:    return gamma_out.append(t);
      case ParticleType::kElectron: return electron_out.append(t);
      case ParticleType::kPositron: return positron_out.append(t);
      // No hadron case, and that is a statement about the physics rather than an omission:
      // nothing in this transport *creates* a proton or an alpha. They arrive only as
      // primaries, through seed_from_primaries, and the delta rays they knock out are
      // electrons. A hadron reaching here would mean a process was added that makes one -
      // a nuclear interaction, say - and it should fail to compile rather than be routed
      // somewhere plausible.
      default:                      return -1;
    }
  }
};

}  // namespace g4gpu
