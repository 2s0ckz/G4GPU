// Trajectory capture for visualization.
//
// Each transport step appends one line segment. Storage is float regardless of the
// transport scalar type - rendering never needs more - and capture is limited to the first
// `max_event` events so a 10M-event physics run does not try to record 400M segments.
#pragma once
#include "core/particle.cuh"
#include "core/vec3.cuh"

namespace g4gpu::vis {

/// Geant4 default trajectory colouring, by charge: negative red, neutral green,
/// positive blue.
enum TrackKind : unsigned char { kKindGamma = 0, kKindElectron = 1, kKindPositron = 2 };

struct TrajectoryBuffer {
  float* x0;
  float* y0;
  float* z0;
  float* x1;
  float* y1;
  float* z1;
  unsigned char* kind;
  int capacity;
  int* count;
  int* dropped;
  int max_event;  ///< capture only events with id < this; 0 disables capture entirely

  template <typename real_t>
  __device__ void add(const Vec3<real_t>& a, const Vec3<real_t>& b, ParticleType type,
                      int event) {
    if (max_event <= 0 || event >= max_event) { return; }
    const int slot = atomicAdd(count, 1);
    if (slot >= capacity) {
      atomicAdd(dropped, 1);
      return;
    }
    x0[slot] = static_cast<float>(a.x);
    y0[slot] = static_cast<float>(a.y);
    z0[slot] = static_cast<float>(a.z);
    x1[slot] = static_cast<float>(b.x);
    y1[slot] = static_cast<float>(b.y);
    z1[slot] = static_cast<float>(b.z);
    switch (type) {
      case ParticleType::kElectron: kind[slot] = kKindElectron; break;
      case ParticleType::kPositron: kind[slot] = kKindPositron; break;
      default:                      kind[slot] = kKindGamma;    break;
    }
  }
};

/// A disabled buffer: capture calls compile away to a branch that never fires.
__host__ __device__ inline TrajectoryBuffer no_capture() {
  TrajectoryBuffer t{};
  t.max_event = 0;
  return t;
}

}  // namespace g4gpu::vis
