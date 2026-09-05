// Track-parallel GPU driver for Geant4 example B1.
//
// Many events in flight at once. Tracks live in structure-of-arrays buffers, one per
// species, and are advanced one step per kernel launch. Buffers ping-pong: a step kernel
// reads `in` and appends survivors plus new secondaries to `out`, which compacts
// implicitly. Events are a tag on each track, never a scheduling unit.
//
// Compare src/host/b1_gpu.cu, the thread-per-event version: 108 registers, ~19% occupancy,
// a private 128-deep secondary stack per thread.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include "physics/stepper.cuh"
#include "host/pe_upload.cuh"

using namespace g4gpu;
#ifdef G4GPU_FP32
// FP32 on a consumer card: the RTX 3070 runs FP64 at 1/32 the FP32 rate. The scalar type
// is a template parameter throughout precisely so this is a compile switch, not a rewrite.
// The per-event dose accumulator stays double regardless.
using real_t = float;
#else
using real_t = double;
#endif

enum VolId : int { kWorld = 0, kEnvelope = 1, kShape1 = 2, kShape2 = 3, kNumVols = 4 };

#define CUDA_CHECK(call)                                                                 \
  do {                                                                                   \
    const cudaError_t err__ = (call);                                                    \
    if (err__ != cudaSuccess) {                                                          \
      printf("CUDA error %s at %s:%d\n", cudaGetErrorString(err__), __FILE__, __LINE__); \
      std::exit(1);                                                                      \
    }                                                                                    \
  } while (0)

/// B1's geometry in the layer model. What Geant4 expresses as world > envelope > {shape1,
/// shape2} becomes four volumes on three layers: the envelope outranks the world, and the two
/// shapes outrank the envelope. No containment is declared and none is required.
void build_b1_volumes(geom::Volume<real_t>* v) {
  using geom::SolidType;
  using geom::make_translation;
  v[kWorld] = {{SolidType::kBox, {120.0, 120.0, 180.0}},
               make_translation<real_t>({0, 0, 0}), 0, data::kAir};
  v[kEnvelope] = {{SolidType::kBox, {100.0, 100.0, 150.0}},
                  make_translation<real_t>({0, 0, 0}), 1, data::kWater};
  v[kShape1] = {{SolidType::kCons, {20.0, 40.0, 30.0}},
                make_translation<real_t>({0.0, 20.0, -70.0}), 2, data::kA150Tissue};
  v[kShape2] = {{SolidType::kTrd, {60.0, 60.0, 50.0, 80.0, 30.0}},
                make_translation<real_t>({0.0, -10.0, 70.0}), 2, data::kBoneCompact,
                /*score_index=*/0};  // B1 scores dose in Shape2
}

real_t trd_volume(real_t dx1, real_t dx2, real_t dy1, real_t dy2, real_t dz) {
  return (2.0 * dz / 3.0)
         * (4.0 * dx1 * dy1 + 4.0 * dx2 * dy2 + 2.0 * (dx1 * dy2 + dx2 * dy1));
}

// ---------------------------------------------------------------- buffer management

struct DeviceBuffer {
  TrackBuffer<real_t> view{};

  void alloc(int capacity) {
    view.capacity = capacity;
    const size_t nr = sizeof(real_t) * capacity;
    CUDA_CHECK(cudaMalloc(&view.x, nr));
    CUDA_CHECK(cudaMalloc(&view.y, nr));
    CUDA_CHECK(cudaMalloc(&view.z, nr));
    CUDA_CHECK(cudaMalloc(&view.dx, nr));
    CUDA_CHECK(cudaMalloc(&view.dy, nr));
    CUDA_CHECK(cudaMalloc(&view.dz, nr));
    CUDA_CHECK(cudaMalloc(&view.ekin, nr));
    CUDA_CHECK(cudaMalloc(&view.volume, sizeof(int) * capacity));
    CUDA_CHECK(cudaMalloc(&view.event, sizeof(int) * capacity));
    CUDA_CHECK(cudaMalloc(&view.rng_key, sizeof(unsigned int) * capacity));
    CUDA_CHECK(cudaMalloc(&view.msc_tlimit, sizeof(real_t) * capacity));
    CUDA_CHECK(cudaMalloc(&view.msc_tlimitmin, sizeof(real_t) * capacity));
    CUDA_CHECK(cudaMalloc(&view.step, sizeof(unsigned int) * capacity));
    CUDA_CHECK(cudaMalloc(&view.count, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&view.overflow, sizeof(int)));
    reset();
    CUDA_CHECK(cudaMemset(view.overflow, 0, sizeof(int)));
  }
  void reset() { CUDA_CHECK(cudaMemset(view.count, 0, sizeof(int))); }
  int count() const {
    int c = 0;
    CUDA_CHECK(cudaMemcpy(&c, view.count, sizeof(int), cudaMemcpyDeviceToHost));
    return c;
  }
  int overflow() const {
    int c = 0;
    CUDA_CHECK(cudaMemcpy(&c, view.overflow, sizeof(int), cudaMemcpyDeviceToHost));
    return c;
  }
  void free_all() {
    cudaFree(view.x); cudaFree(view.y); cudaFree(view.z);
    cudaFree(view.dx); cudaFree(view.dy); cudaFree(view.dz);
    cudaFree(view.ekin); cudaFree(view.volume); cudaFree(view.event);
    cudaFree(view.rng_key); cudaFree(view.step);
    cudaFree(view.count); cudaFree(view.overflow);
  }
};

// ---------------------------------------------------------------- kernels

/// Seeds one primary photon per event directly into the gamma buffer. Slot == local event
/// index, so no atomics are needed here.
__global__ void seed_primaries(TrackBuffer<real_t> gamma, geom::Geometry<real_t> geometry,
                               int n, int event_base, real_t spread, real_t z0, real_t e0) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) { return; }
  Philox<real_t> gen(0xF00Du, static_cast<unsigned int>(event_base + i));
  const real_t x0 = spread * (gen.uniform() - real_t(0.5));
  const real_t y0 = spread * (gen.uniform() - real_t(0.5));
  const Vec3<real_t> pos{x0, y0, z0};
  gamma.x[i] = pos.x;   gamma.y[i] = pos.y;   gamma.z[i] = pos.z;
  gamma.dx[i] = 0;      gamma.dy[i] = 0;      gamma.dz[i] = 1;
  gamma.ekin[i] = e0;
  gamma.volume[i] = geom::locate(geometry, pos);
  gamma.event[i] = i;  // local index within the batch, used to index the dose array
  gamma.rng_key[i] = static_cast<unsigned int>(event_base + i);  // globally unique
  gamma.step[i] = 0u;
  gamma.msc_tlimit[i] = real_t(0);  // stale: first step recomputes it
  gamma.msc_tlimitmin[i] = real_t(0);
  if (i == 0) { *gamma.count = n; }
}

__global__ void step_gamma_kernel(Scene<real_t> scene, TrackBuffer<real_t> in,
                                  TrackBuffer<real_t> gamma_out,
                                  TrackBuffer<real_t> electron_out,
                                  TrackBuffer<real_t> positron_out, int n, int iteration,
                                  double* edep_per_event) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) { return; }
  TrackState<real_t> p;
  in.load(i, p);

  // Deterministic: rng_key is carried by the track, not derived from its buffer slot.
  Philox<real_t> rng(p.rng_key, p.step, 0u);
  BufferEmitter<real_t> em{gamma_out, electron_out, positron_out, p.pos, p.volume,
                           p.event,     p.rng_key,     p.step,        0u};

  real_t edep = 0;
  const bool alive = step_gamma(scene, p, rng, em, edep);
  ++p.step;
  if (edep != real_t(0)) { atomicAdd(&edep_per_event[p.event], static_cast<double>(edep)); }
  if (alive) { gamma_out.append(p); }
}

template <bool kIsPositron>
__global__ void step_lepton_kernel(Scene<real_t> scene, TrackBuffer<real_t> in,
                                   TrackBuffer<real_t> gamma_out,
                                   TrackBuffer<real_t> electron_out,
                                   TrackBuffer<real_t> positron_out, int n, int iteration,
                                   double* edep_per_event) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) { return; }
  TrackState<real_t> p;
  in.load(i, p);

  Philox<real_t> rng(p.rng_key, p.step, 0x5A5Au);  // no p.event: it is batch-local
  BufferEmitter<real_t> em{gamma_out, electron_out, positron_out, p.pos, p.volume,
                           p.event,     p.rng_key,     p.step,        0u};

  real_t edep = 0;
  const bool alive = step_lepton(scene, p, kIsPositron, rng, em, edep);
  ++p.step;
  if (edep != real_t(0)) { atomicAdd(&edep_per_event[p.event], static_cast<double>(edep)); }
  if (alive) {
    if (kIsPositron) { positron_out.append(p); } else { electron_out.append(p); }
  }
}

// ---------------------------------------------------------------- main

int main(int argc, char** argv) {
  const int n_events = (argc > 1) ? std::atoi(argv[1]) : 10000;
  // Larger batches keep the drain tail busy: 2.23M events/s at batch 256k vs 2.52M at 1M.
  // Clamped to n_events so a small run does not allocate for a large batch.
  const int requested_batch = (argc > 2) ? std::atoi(argv[2]) : 1048576;
  const int batch_size = std::min(requested_batch, std::max(n_events, 1));
  const int threads = (argc > 3) ? std::atoi(argv[3]) : 128;

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  // The solid engine is mutually recursive (a boolean node asks its children for distances),
  // so the stepping kernels need a real call stack. The default is 1 KB per thread, which one
  // frame of the distance routine can exhaust on its own; exceeding it shows up as an illegal
  // memory access from whichever API call happens to synchronise next, with no hint of the
  // cause. 16 KB covers kMaxBooleanDepth nesting with room to spare.
  CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 16384));
  printf("GPU: %s, CC %d.%d, %d SMs\n", prop.name, prop.major, prop.minor,
         prop.multiProcessorCount);
  printf("track-parallel scheduler: batch %d events, %d threads/block\n\n", batch_size, threads);

  geom::Volume<real_t> h_vols[kNumVols];
  build_b1_volumes(h_vols);
  data::Material<real_t> h_mats[data::kNumMaterials];
  data::build_b1_materials<real_t>(h_mats);
  em::RangeTable<real_t> h_rt;
  em::build_range_table<real_t>(h_mats, h_rt);

  geom::Volume<real_t>* d_vols = nullptr;
  data::Material<real_t>* d_mats = nullptr;
  em::RangeTable<real_t>* d_rt = nullptr;
  CUDA_CHECK(cudaMalloc(&d_vols, sizeof(h_vols)));
  CUDA_CHECK(cudaMalloc(&d_mats, sizeof(h_mats)));
  CUDA_CHECK(cudaMalloc(&d_rt, sizeof(h_rt)));
  CUDA_CHECK(cudaMemcpy(d_vols, h_vols, sizeof(h_vols), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_mats, h_mats, sizeof(h_mats), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_rt, &h_rt, sizeof(h_rt), cudaMemcpyHostToDevice));

  geom::Geometry<real_t> d_geom{d_vols, kNumVols, kWorld};
  auto* d_pe = host::upload_photoelectric<real_t>(host::default_phot_dir());
  auto brem = host::upload_brems<real_t>(host::default_sb_dir(), h_mats, h_rt);
  // upload_brems re-integrates the range table, so push the new one to the device.
  CUDA_CHECK(cudaMemcpy(d_rt, &h_rt, sizeof(h_rt), cudaMemcpyHostToDevice));
  int n_z_ = 0;
  const int* zs_ = host::b1_elements(n_z_);
  auto* d_ray = host::upload_rayleigh<real_t>(host::default_rayl_dir(), zs_, n_z_);
  auto* d_msc = host::upload_msc<real_t>(h_mats, data::kNumMaterials);
  // No hadron range table and no hadron primaries: this driver runs the B1 gamma beam only,
  // so step_hadron is unreachable and a null table is the honest value rather than a
  // two-megabyte one nothing reads. The range cut is B1's, for the MSC step limit.
  Scene<real_t> scene{d_geom,     d_mats,   d_rt,  d_pe,
                     brem.table, brem.sb,  d_ray, d_msc,
                     nullptr,    real_t(0.7),
                     kShape2};

  // Capacities are multiples of the batch; peak occupancy is reported so they can be tuned.
  DeviceBuffer gamma[2], electron[2], positron[2];
  for (int i = 0; i < 2; ++i) {
    gamma[i].alloc(batch_size * 2);
    electron[i].alloc(batch_size * 4);
    positron[i].alloc(batch_size / 2 + 1024);
  }

  double* d_edep = nullptr;
  CUDA_CHECK(cudaMalloc(&d_edep, sizeof(double) * batch_size));

  const real_t env_xy = 200.0, env_z = 300.0;
  const real_t spread = 0.8 * env_xy;
  const real_t z0 = -0.5 * env_z;
  const real_t e0 = 6.0;

  double sum = 0, sum_sq = 0;
  long long total_steps = 0;
  int max_iterations = 0, peak_gamma = 0, peak_electron = 0, peak_positron = 0;
  long long abandoned = 0;
  std::vector<double> h_edep(batch_size);

  cudaEvent_t t0, t1;
  CUDA_CHECK(cudaEventCreate(&t0));
  CUDA_CHECK(cudaEventCreate(&t1));
  CUDA_CHECK(cudaEventRecord(t0));

  for (int base = 0; base < n_events; base += batch_size) {
    const int n_batch = std::min(batch_size, n_events - base);
    CUDA_CHECK(cudaMemset(d_edep, 0, sizeof(double) * n_batch));

    int cur = 0, nxt = 1;
    gamma[cur].reset(); electron[cur].reset(); positron[cur].reset();
    seed_primaries<<<(n_batch + threads - 1) / threads, threads>>>(
        gamma[cur].view, d_geom, n_batch, base, spread, z0, e0);
    CUDA_CHECK(cudaGetLastError());

    // Bounded so a non-terminating track shows up as an explicit abandonment rather than
    // an 86x slowdown. B1 drains in ~110 iterations.
    constexpr int kMaxIterations = 1000;
    int leftover = 0;
    for (int iteration = 0; iteration < kMaxIterations; ++iteration) {
      const int ng = gamma[cur].count();
      const int ne = electron[cur].count();
      const int np = positron[cur].count();
      if (ng + ne + np == 0) { max_iterations = std::max(max_iterations, iteration); break; }
      if (iteration == kMaxIterations - 1) { leftover = ng + ne + np; }
      total_steps += ng + ne + np;
      peak_gamma = std::max(peak_gamma, ng);
      peak_electron = std::max(peak_electron, ne);
      peak_positron = std::max(peak_positron, np);

      gamma[nxt].reset(); electron[nxt].reset(); positron[nxt].reset();

      if (ng > 0) {
        step_gamma_kernel<<<(ng + threads - 1) / threads, threads>>>(
            scene, gamma[cur].view, gamma[nxt].view, electron[nxt].view, positron[nxt].view,
            ng, iteration, d_edep);
      }
      if (ne > 0) {
        step_lepton_kernel<false><<<(ne + threads - 1) / threads, threads>>>(
            scene, electron[cur].view, gamma[nxt].view, electron[nxt].view, positron[nxt].view,
            ne, iteration, d_edep);
      }
      if (np > 0) {
        step_lepton_kernel<true><<<(np + threads - 1) / threads, threads>>>(
            scene, positron[cur].view, gamma[nxt].view, electron[nxt].view, positron[nxt].view,
            np, iteration, d_edep);
      }
      CUDA_CHECK(cudaGetLastError());
      cur ^= 1;
      nxt ^= 1;
    }

    if (leftover > 0) {
      printf("WARNING: %d tracks abandoned after %d iterations, batch at %d\n", leftover,
             kMaxIterations, base);
      abandoned += leftover;
    }
    CUDA_CHECK(cudaMemcpy(h_edep.data(), d_edep, sizeof(double) * n_batch,
                          cudaMemcpyDeviceToHost));
    for (int i = 0; i < n_batch; ++i) {
      sum += h_edep[i];
      sum_sq += h_edep[i] * h_edep[i];
    }
  }

  CUDA_CHECK(cudaEventRecord(t1));
  CUDA_CHECK(cudaDeviceSynchronize());
  float ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));

  const real_t vol_mm3 = trd_volume(60.0, 60.0, 50.0, 80.0, 30.0);
  const real_t mass_kg = vol_mm3 / 1000.0 * h_mats[data::kBoneCompact].density / 1000.0;
  real_t rms = sum_sq - sum * sum / real_t(n_events);
  rms = (rms > 0) ? std::sqrt(rms) : real_t(0);
  const real_t mev_to_joule = 1.602176634e-13;
  const real_t dose_gy = sum * mev_to_joule / mass_kg;
  const real_t rms_gy = rms * mev_to_joule / mass_kg;

  printf("--------------------End of Global Run-----------------------\n");
  printf(" The run is %d gamma of %.3g MeV\n\n", n_events, e0);
  printf("  --> cumulated edep per run in scoring volume = %.6g GeV = %.6g joule\n",
         sum / 1000.0, sum * mev_to_joule);
  printf("  --> mass of scoring volume = %.5g kg\n\n", mass_kg);
  printf(" Absorbed dose per run in scoring volume = edep/mass = %.6g picoGy; rms = %.6g picoGy\n",
         dose_gy * 1e12, rms_gy * 1e12);
  printf("------------------------------------------------------------\n\n");

  printf("time %.3f ms for %d events = %.3g events/s   (%.3g track-steps/s)\n", ms, n_events,
         n_events / (ms * 1e-3), total_steps / (ms * 1e-3));
  printf("total track-steps %lld = %.1f per event; max iterations per batch %d\n", total_steps,
         double(total_steps) / n_events, max_iterations);
  printf("peak live: gamma %d/%d, electron %d/%d, positron %d/%d\n", peak_gamma,
         gamma[0].view.capacity, peak_electron, electron[0].view.capacity, peak_positron,
         positron[0].view.capacity);
  // Full precision, so reproducibility can be checked bit-for-bit rather than to 6 digits.
  printf("raw edep sum = %.17g MeV\n", sum);
  const real_t per10k = sum / 1000.0 * (10000.0 / n_events);
  // Reference: Geant4 11.1.1 - the version this port was transcribed from - running
  // exampleB1 with 2,000,000 events, scaled to 10k. The version-matched run, not the
  // 11.5.0 exampleB1.out shipped with the source: those two differ by 1.6%, which is
  // larger than anything being measured here.
  constexpr double kRefDosePer10k = 427.385;  // pGy, +/- 0.870 (2M events)
  const double dose_per10k = dose_gy * 1e12 * (10000.0 / n_events);
  printf("scaled to 10k: edep = %.5g GeV, dose = %.3f pGy\n", per10k, dose_per10k);
  printf("Geant4 11.1.1 (2M events, scaled to 10k): %.3f pGy +/- 0.870   ratio = %.4f\n",
         kRefDosePer10k, dose_per10k / kRefDosePer10k);
  // B1 reports rms as the uncertainty on the accumulated total, so a 10,000-event run
  // carries +/-2.9% - far too loose to resolve a 1% difference in the physics.
  if (rms_gy > 0 && dose_gy > 0) {
    const double rel = 100.0 * rms_gy / dose_gy;
    printf("  this run: +/-%.2f%% statistical uncertainty%s\n", rel,
           (rel > 0.5) ? "  <-- run 2000000 events before comparing" : "");
  }
  int ovf = 0;
  for (int i = 0; i < 2; ++i) {
    ovf += gamma[i].overflow() + electron[i].overflow() + positron[i].overflow();
  }
  if (ovf > 0) { printf("WARNING: track buffer overflowed %d times\n", ovf); }
  if (abandoned > 0) {
    printf("WARNING: %lld non-terminating tracks abandoned\n", abandoned);
  }

  for (int i = 0; i < 2; ++i) { gamma[i].free_all(); electron[i].free_all(); positron[i].free_all(); }
  cudaFree(d_edep); cudaFree(d_vols); cudaFree(d_mats); cudaFree(d_rt);
  return 0;
}
