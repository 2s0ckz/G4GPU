// Flat SoA track store shared by all events in flight. Secondaries are appended by
// atomic bump; there is no device-side allocation. Overflow is recorded, never silent.
#pragma once
#include <cstdint>
#include "core/particle.cuh"
#include "core/vec3.cuh"

namespace g4gpu {

template <typename real_t>
struct TrackSoA {
  real_t* x;
  real_t* y;
  real_t* z;
  real_t* dx;
  real_t* dy;
  real_t* dz;
  real_t* ekin;
  int* particle;
  int* volume;   ///< volume id the track currently sits in
  int* event;    ///< event tag; events are a label, never a scheduling unit
  int* birth;    ///< serial id within the event, seeds the RNG stream
  uint8_t* alive;

  int capacity;
  int* n_used;    ///< high-water mark, bumped atomically by push()
  int* n_overflow;

  __host__ __device__ int push(ParticleType type, const Vec3<real_t>& dir, real_t e_kin,
                               const Vec3<real_t>& pos, int volume_id, int event_id, int birth_id)
  {
#ifdef __CUDA_ARCH__
    const int slot = atomicAdd(n_used, 1);
#else
    const int slot = (*n_used)++;
#endif
    if (slot >= capacity) {
#ifdef __CUDA_ARCH__
      atomicAdd(n_overflow, 1);
#else
      ++(*n_overflow);
#endif
      return -1;
    }
    x[slot] = pos.x;   y[slot] = pos.y;   z[slot] = pos.z;
    dx[slot] = dir.x;  dy[slot] = dir.y;  dz[slot] = dir.z;
    ekin[slot] = e_kin;
    particle[slot] = static_cast<int>(type);
    volume[slot] = volume_id;
    event[slot] = event_id;
    birth[slot] = birth_id;
    alive[slot] = 1;
    return slot;
  }
};

/// The subset of push() arguments a physics model knows about. Position, volume and
/// event are filled in by the stepping kernel, which owns that context.
template <typename real_t>
struct SecondaryEmitter {
  TrackSoA<real_t>* store;
  Vec3<real_t> pos;
  int volume_id;
  int event_id;
  int birth_counter;

  __host__ __device__ int push(ParticleType type, const Vec3<real_t>& dir, real_t e_kin,
                               int /*event_id_unused*/)
  {
    return store->push(type, dir, e_kin, pos, volume_id, event_id, ++birth_counter);
  }
};

}  // namespace g4gpu
