// Trajectory capture for visualization.
//
// Each transport step appends one line segment. Storage is float regardless of the
// transport scalar type - rendering never needs more - and capture is limited to the first
// `max_event` events so a 10M-event physics run does not try to record 400M segments.
#pragma once
#include <cuda_runtime.h>

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
  /// Which track drew the segment - the track's RNG key, the same value G4Track::GetTrackID()
  /// returns. Segments land in whatever order threads reach the atomic, so without this a
  /// segment cannot be attributed to anything and the store is a bag of unrelated lines.
  ///
  /// Not a hypothetical loss. Charged trajectories drew as disconnected dashes for the whole
  /// life of the viewer - the record was written before multiple scattering displaced the
  /// track, so consecutive segments did not share an endpoint - and nothing could check it,
  /// because checking it means asking whether ONE TRACK's segments join up. See docs/RISK.md
  /// V13 and tests/test_trajectory.cu.
  unsigned int* track;
  int capacity;
  int* count;
  int* dropped;
  int max_event;  ///< capture only events with id < this; 0 disables capture entirely

  template <typename real_t>
  __device__ void add(const Vec3<real_t>& a, const Vec3<real_t>& b, ParticleType type,
                      int event, unsigned int track_id) {
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
    track[slot] = track_id;
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

/// Allocates every array a capture buffer needs and zeroes its counters.
///
/// ONE field list. The viewer and the model builder each carried their own copy of this block -
/// six cudaMallocs and two counters apiece - so adding `track` to the struct meant finding both,
/// with a buffer whose new array is null in one of them and a kernel writing through it as the
/// failure mode. Same argument as allocate_track_buffer's, same reason, a twentieth of the size.
///
/// @param capacity  segments; beyond it `add` counts drops rather than writing past the end
/// @param max_event capture only events with id below this; 0 disables capture entirely
inline cudaError_t allocate_trajectory(TrajectoryBuffer& t, int capacity, int max_event) {
  t = TrajectoryBuffer{};
  t.capacity = capacity;
  t.max_event = max_event;
  const size_t nf = sizeof(float) * static_cast<size_t>(capacity);
  cudaError_t e = cudaSuccess;
  auto get = [&](void** p, size_t n) {
    if (e == cudaSuccess) { e = cudaMalloc(p, n); }
  };
  get(reinterpret_cast<void**>(&t.x0), nf);
  get(reinterpret_cast<void**>(&t.y0), nf);
  get(reinterpret_cast<void**>(&t.z0), nf);
  get(reinterpret_cast<void**>(&t.x1), nf);
  get(reinterpret_cast<void**>(&t.y1), nf);
  get(reinterpret_cast<void**>(&t.z1), nf);
  get(reinterpret_cast<void**>(&t.kind), static_cast<size_t>(capacity));
  get(reinterpret_cast<void**>(&t.track),
      sizeof(unsigned int) * static_cast<size_t>(capacity));
  get(reinterpret_cast<void**>(&t.count), sizeof(int));
  get(reinterpret_cast<void**>(&t.dropped), sizeof(int));
  if (e != cudaSuccess) { return e; }
  e = cudaMemset(t.count, 0, sizeof(int));
  if (e != cudaSuccess) { return e; }
  return cudaMemset(t.dropped, 0, sizeof(int));
}

inline void free_trajectory(TrajectoryBuffer& t) {
  cudaFree(t.x0); cudaFree(t.y0); cudaFree(t.z0);
  cudaFree(t.x1); cudaFree(t.y1); cudaFree(t.z1);
  cudaFree(t.kind); cudaFree(t.track);
  cudaFree(t.count); cudaFree(t.dropped);
  t = TrajectoryBuffer{};
}

}  // namespace g4gpu::vis
