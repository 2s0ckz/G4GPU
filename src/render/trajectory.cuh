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

/// Geant4's default trajectory colouring, which is G4TrajectoryDrawByCharge: negative red,
/// neutral green, positive blue.
///
/// BY CHARGE, and the names say so now. They used to be kKindGamma/kKindElectron/kKindPositron
/// and the mapping was a switch on the species with `default: kKindGamma` at the bottom - so
/// every proton and every alpha, both positive, were recorded as the neutral class and drew
/// green. A colour rule stated by species cannot help but be wrong about the species nobody
/// listed. Stated by charge, there is nothing to list. See docs/RISK.md V14.
enum TrackKind : unsigned char { kKindNeutral = 0, kKindNegative = 1, kKindPositive = 2 };

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

  /// `__host__ __device__`, and the host arm is not decoration.
  ///
  /// The steppers call this, and P8b made `step_hadron` host-callable so that a test can run
  /// the SAME function on both sides and compare the two bit for bit - which is how every
  /// other piece of physics in this port is checked, and which `step_hadron` had no way to be
  /// (`tests/test_step_hadron.cu`). The only thing in this function that was device-only was
  /// the two atomics, and a host caller is single-threaded by construction, so the host arm is
  /// a plain increment rather than a serialisation of a parallel algorithm.
  template <typename real_t>
  __host__ __device__ void add(const Vec3<real_t>& a, const Vec3<real_t>& b, ParticleType type,
                               int event, unsigned int track_id) {
    if (max_event <= 0 || event >= max_event) { return; }
#ifdef __CUDA_ARCH__
    const int slot = atomicAdd(count, 1);
#else
    const int slot = (*count)++;
#endif
    if (slot >= capacity) {
#ifdef __CUDA_ARCH__
      atomicAdd(dropped, 1);
#else
      ++(*dropped);
#endif
      return;
    }
    x0[slot] = static_cast<float>(a.x);
    y0[slot] = static_cast<float>(a.y);
    z0[slot] = static_cast<float>(a.z);
    x1[slot] = static_cast<float>(b.x);
    y1[slot] = static_cast<float>(b.y);
    z1[slot] = static_cast<float>(b.z);
    track[slot] = track_id;
    // The charge comes from particle_def, the same table the physics reads, so the colour of
    // a track and the charge it is transported with cannot disagree - and a species added to
    // that table is coloured correctly the day it is added, with nothing to update here.
    const real_t q = particle_def<real_t>(type).charge;
    kind[slot] = (q < real_t(0)) ? kKindNegative
                                 : ((q > real_t(0)) ? kKindPositive : kKindNeutral);
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
