// The flat Binary cascade (P36, bic/flat/cascade_flat.cuh) against the one-thread transcription
// and against Geant4, statistically (docs/RISK.md V302's rule, V309's campaign).
//
// WHAT IT RUNS. For every point - a projectile, an energy and a target - N = 20,000 model calls
// `G4BinaryCascade::ApplyYourself` three ways, each from the same keyed generator:
//
//   * FLAT: the stages of cascade_flat.cuh on the device, lane = cascade, exactly as the
//     transport's drain runs them (the tail on the interaction pool's own slots); a call the flat
//     path hands back is answered by the thread path, as in the transport;
//   * THREAD: `bic::apply_yourself`, the one-thread transcription, on the host - the code the
//     exact oracle tests (test_bic_apply, test_bic_imr) hold line for line;
//   * GEANT4: ref/oracle/bic_apply*.csv, 5,000 events a case, where the case is in the oracle.
//
// WHAT IT COMPARES, each row a z-score: per species (p, n, pi+, pi-, pi0, gamma, d, t, He3,
// alpha, A > 4 fragments) the mean multiplicity; per hadron species the kinetic-energy spectrum
// in 12 logarithmic bins and the polar-angle spectrum in 10 bins of cos(theta); the final-state
// size in 12 bins; the energy balance's mean (sum of total energies out against in, with the
// target's mass) and the baryon balance's mean. Flat against thread: z = (a - b)/sqrt(va + vb)
// with each side's sample variance. Flat against Geant4: the mean multiplicity and the mean
// kinetic energy of every (case, species) row the oracle has, with the per-event variance it
// reports. A row with fewer than 25 entries on both sides is not scored (its z is not normal).
// NO ROW PAST 5 SIGMA, and the worst row is printed by name.
//
// ANTI-VACUITY: the same flat campaign with `FlatEnv::perturb = 1`, which skips `Capture` (the
// nucleons below the cut stay in the cascade), must fail the thread comparison past 5 sigma -
// otherwise the comparison could not see a change of that size and the pass above says nothing.
//
// DETERMINISM: the flat campaign is run twice and every final state must agree to the bit.
//
// P39 (docs/RISK.md V321, V322): the flat path's nuclei come from the nucleus factory
// (npool/nucleus_factory.cuh), requested before the set-up and keyed on the call's key and its
// nucleus count, as the transport's drain does it; a straggler is finished on its own lane
// (`k_solo`) rather than handed back. `G4GPU_FLAT_FACTORY=0` runs P36's flat path (the nucleus
// on the lane from the call's generator, the stragglers handed back).
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "core/rng.cuh"
#include "host/g4data.cuh"
#include "host/interaction_pool_build.cuh"
#include "host/level_upload.cuh"
#include "physics/hadronic/bic/flat/ion_flat.cuh"
#include "physics/hadronic/interaction_apply.cuh"

using real_t = double;
using namespace g4gpu;
namespace bf = g4gpu::bic::flat;

namespace g4gpu::had {
template InteractionPoolOwner<double> build_interaction_pool<double>(int, const data::LevelTable&,
                                                                     bool);
template void free_interaction_pool<double>(InteractionPoolOwner<double>&);
}  // namespace g4gpu::had

namespace {

// tests/test_blir_flat.cu defines G4GPU_TEST_BLIR_FLAT and includes this file: the same campaign
// for `G4BinaryLightIonReaction` (bic/flat/ion_flat.cuh) against `blir_apply_yourself` and
// ref/oracle/bic_blirapply.csv.
#ifdef G4GPU_TEST_BLIR_FLAT
constexpr bool kIon = true;
constexpr int kNuc = 64, kPool = 256, kColl = 512;
using Rec = bf::FlatIon<kNuc, kPool, kColl>;
constexpr int kMaxOut = bic::kBlirMaxSecondaries;
#else
constexpr bool kIon = false;
constexpr int kNuc = 64, kPool = 128, kColl = 256;
using Rec = bf::FlatCascade<kNuc, kPool, kColl>;
constexpr int kMaxOut = bic::kBicMaxSecondaries;
#endif
using Core = bf::FlatCascade<kNuc, kPool, kColl>;
__host__ __device__ inline Core& core(Core& r) { return r; }
__host__ __device__ inline Core& core(bf::FlatIon<kNuc, kPool, kColl>& r) { return r.c; }

struct Job {
  int pdg = 0;
  int pz = 0, pa = 0;     ///< an ion's Z and A
  double mass = 0.0, charge = 0.0;
  double ekin = 0.0;
  int tz = 0, ta = 0;
  uint32_t key = 0;
};

/// One model call's final state, as the comparison reads it.
struct Out {
  int status = -1;       ///< HadFinalStateStatus, -1 for a refusal
  int n = 0;
  int pdg[kMaxOut];
  int a[kMaxOut];
  int z[kMaxOut];
  float ekin[kMaxOut];
  float cost[kMaxOut];
  double e_sum = 0.0;    ///< total energy out (kinetic + mass)
  int a_sum = 0;
  int handed = 0;        ///< the flat path handed it back (answered by the thread path)
  uint64_t hash = 0;
};

__host__ __device__ inline void mix(uint64_t& h, uint64_t v) {
  h ^= v + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2);
  h *= 0x100000001B3ull;
}
__host__ __device__ inline uint64_t bits(double x) {
  uint64_t b;
  memcpy(&b, &x, sizeof b);
  return b;
}

template <typename FS>
__host__ __device__ void record(const FS& fs, Out& o) {
  o.status = static_cast<int>(fs.status);
  o.n = 0;
  o.e_sum = 0.0;
  o.a_sum = 0;
  uint64_t h = 0xCBF29CE484222325ull;
  mix(h, static_cast<uint64_t>(fs.status));
  mix(h, bits(fs.energy_change));
  for (int i = 0; i < fs.n_secondaries && i < kMaxOut; ++i) {
    const auto& s = fs.secondaries[i];
    o.pdg[o.n] = s.pdg;
    o.a[o.n] = s.a;
    o.z[o.n] = s.z;
    o.ekin[o.n] = static_cast<float>(s.kin_energy);
    o.cost[o.n] = static_cast<float>(s.direction.z);
    o.e_sum += s.kin_energy + s.mass;
    o.a_sum += (s.a > 0) ? s.a : ((s.pdg == 2212 || s.pdg == 2112) ? 1 : 0);
    mix(h, static_cast<uint64_t>(static_cast<int64_t>(s.pdg)));
    mix(h, bits(s.kin_energy));
    mix(h, bits(s.direction.x));
    mix(h, bits(s.direction.z));
    ++o.n;
  }
  o.hash = h;
}

// ---------------------------------------------------------------------------------------------
// The flat path, model level: the unit's kernels without the transport around them.
// ---------------------------------------------------------------------------------------------
struct Args {
  Rec* rec;
  int* job_of;            ///< lane -> job
  bic::CascadeBuffers* buffers;
  int* ctr;               ///< 0 next job, 1 items, 2 tail used, 3 active
  int* items;
  int item_capacity;
  int lanes;
  int n;
  int perturb;
  int solo;               ///< P39: stragglers finished on their lane (the factory is on)
  const Job* jobs;
  Out* out;
  const had::InteractionPool<real_t>* pool;
  data::LevelTable lt;
};

__device__ bf::FlatEnv env_of(const Args& a) {
  bf::FlatEnv e;
  e.channels = a.pool->bic_channels;
  e.n_channels = a.pool->n_bic_channels;
  e.buffers = a.buffers;
  // A bound far past V307's longest cascade (149 steps), so no real call is a straggler, and a
  // perturbed one that never ends (a secondary that never moves never leaves) is handed back.
  e.k_steps = 4096;
  e.perturb = a.perturb;
  e.solo = a.solo;
  return e;
}

__global__ void k_claim(Args a) {
  const int lane = blockIdx.x * blockDim.x + threadIdx.x;
  if (lane >= a.lanes || core(a.rec[lane]).stage != bf::kStFree) { return; }
  const int e = atomicAdd(&a.ctr[0], 1);
  if (e >= a.n) { return; }
  a.job_of[lane] = e;
  const Job& j = a.jobs[e];
  const Philox<real_t> rng(j.key, 7u, had::kInteractionRngPurpose);
#ifdef G4GPU_TEST_BLIR_FLAT
  bf::ion_start(a.rec[lane], e, j.pdg, j.ekin, j.mass, j.charge, j.pa, j.tz, j.ta, rng);
#else
  bf::flat_start(a.rec[lane], e, j.pdg, j.ekin, j.tz, j.ta, rng);
#endif
  // P39: the nuclei are keyed on the call's generator key, as the transport keys them.
  bf::flat_set_key(core(a.rec[lane]), j.key, 7u);
}
// P39 (docs/RISK.md V321, V322): the transport's set-up - the model's top and the requests to
// the nucleus factory, then the factory, then `k_setup` reading what it built.
__global__ void k_request(Args a, npool::FactoryLists lists) {
  const int lane = blockIdx.x * blockDim.x + threadIdx.x;
  if (lane >= a.lanes) { return; }
  Core& c = core(a.rec[lane]);
  if (c.stage == bf::kStModel) {
#ifdef G4GPU_TEST_BLIR_FLAT
    bf::stage_ion_model(a.rec[lane], 0);
#else
    bf::stage_model(c);
#endif
  }
  if (c.stage != bf::kStNucleus) { return; }
#ifdef G4GPU_TEST_BLIR_FLAT
  bf::ion_request(a.rec[lane], lists);
#else
  bf::flat_request_target(c, lists);
#endif
  if (!c.nuc_ready) { atomicAdd(&a.ctr[4], 1); }
}
__global__ void k_setup(Args a) {
  const int lane = blockIdx.x * blockDim.x + threadIdx.x;
  if (lane >= a.lanes) { return; }
  Core& c = core(a.rec[lane]);
  if (c.stage == bf::kStModel) {
#ifdef G4GPU_TEST_BLIR_FLAT
    bf::stage_ion_model(a.rec[lane], 0);
#else
    bf::stage_model(c);
#endif
  }
  if (c.stage != bf::kStNucleus) { return; }
  deex::Vec3d mom[kNuc];
  double fermi_p[kNuc];
  bic::NucleusSortEntry sums[kNuc];
  double flat_block[bic::kFlatBlock];
  bic::Nucleus3DScratch nsc;
  nsc.momentum = mom;
  nsc.fermi_p = fermi_p;
  nsc.test_sums = sums;
  nsc.flat_block = flat_block;
  nsc.capacity = kNuc;
  bf::FlatScratch<kNuc, kPool> scr;
#ifdef G4GPU_TEST_BLIR_FLAT
  bf::stage_ion_nucleus(a.rec[lane], env_of(a), nsc, scr);
#else
  bf::stage_nucleus(c, env_of(a), nsc, scr);
#endif
}
__global__ void k_pre(Args a) {
  const int lane = blockIdx.x * blockDim.x + threadIdx.x;
  if (lane >= a.lanes) { return; }
  Core& c = core(a.rec[lane]);
  if (c.stage != bf::kStStep && c.stage != bf::kStStepOut) { return; }
  if (c.solo) { return; }
  bf::FlatScratch<kNuc, kPool> scr;
  bf::stage_pre(c, env_of(a), scr);
  if (!c.stepping) { return; }
  int n_sec = 0;
  int last = -1;
  for (int i = 0; i < c.st.lists.n_pool; ++i) {
    if (c.prestate[i] >= 0) {
      ++n_sec;
      last = i;
    }
  }
  // Perturbation 3, the ion campaign's anti-vacuity: an off-by-one in this enumeration - the
  // step's last secondary gets no move item (it stands still for the step, its transfer zero).
  if (a.perturb == 3 && last >= 0) {
    c.transfer[last] = deex::Vec3d{0.0, 0.0, 0.0};
    --n_sec;
  }
  const int at = atomicAdd(&a.ctr[1], n_sec);
  int k = at;
  for (int i = 0; i < c.st.lists.n_pool; ++i) {
    if (c.prestate[i] >= 0 && !(a.perturb == 3 && i == last)) { a.items[k++] = (lane << 9) | i; }
  }
}
__global__ void k_move(Args a) {
  const int n = a.ctr[1];
  for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < n; k += gridDim.x * blockDim.x) {
    const int it = a.items[k];
    bf::item_transport(core(a.rec[it >> 9]), it & 511);
  }
}
__global__ void k_post(Args a) {
  const int lane = blockIdx.x * blockDim.x + threadIdx.x;
  if (lane >= a.lanes) { return; }
  Core& c = core(a.rec[lane]);
  if (c.stage != bf::kStStep && c.stage != bf::kStStepOut) { return; }
  if (c.solo) { return; }
  bf::FlatScratch<kNuc, kPool> scr;
  bf::stage_post(c, env_of(a), scr);
}
__global__ void k_finish(Args a) {
  const int lane = blockIdx.x * blockDim.x + threadIdx.x;
  if (lane >= a.lanes) { return; }
  Core& c = core(a.rec[lane]);
  if (c.solo) { return; }
  bf::FlatScratch<kNuc, kPool> scr;
  if (c.stage == bf::kStFinishA) { bf::stage_finish_a(c, env_of(a), scr); }
  else if (c.stage == bf::kStFinishB) { bf::stage_finish_b(c); }
}
// P39 (docs/RISK.md V321): a straggler finished on its own lane, as the transport's `flat_solo`.
__global__ void k_solo(Args a) {
  const int lane = blockIdx.x * blockDim.x + threadIdx.x;
  if (lane >= a.lanes) { return; }
  Core& c = core(a.rec[lane]);
  if (!c.solo) { return; }
  if (c.stage == bf::kStFree || c.stage == bf::kStTail) {
    c.solo = 0;
    return;
  }
  atomicAdd(&a.ctr[5], 1);
  bf::FlatEnv env = env_of(a);
  env.k_steps = 1 << 30;
  deex::Vec3d mom[kNuc];
  double fermi_p[kNuc];
  bic::NucleusSortEntry sums[kNuc];
  double flat_block[bic::kFlatBlock];
  bic::Nucleus3DScratch nsc;
  nsc.momentum = mom;
  nsc.fermi_p = fermi_p;
  nsc.test_sums = sums;
  nsc.flat_block = flat_block;
  nsc.capacity = kNuc;
  bf::FlatScratch<kNuc, kPool> scr;
  bf::flat_run_solo(c, env, scr, [&] {
#ifdef G4GPU_TEST_BLIR_FLAT
    if (c.stage == bf::kStModel) { bf::stage_ion_model(a.rec[lane], 0); }
    if (c.stage == bf::kStNucleus) { bf::stage_ion_nucleus(a.rec[lane], env, nsc, scr); }
#else
    if (c.stage == bf::kStModel) { bf::stage_model(c); }
    if (c.stage == bf::kStNucleus) { bf::stage_nucleus(c, env, nsc, scr); }
#endif
  });
}
__global__ void k_tail(Args a, int n_slots) {
  const int lane = blockIdx.x * blockDim.x + threadIdx.x;
  if (lane >= a.lanes) { return; }
  Core& c = core(a.rec[lane]);
  if (c.stage != bf::kStTail) { return; }
  const int si = atomicAdd(&a.ctr[2], 1);
  if (si >= n_slots) { return; }
  had::InteractionSlot<real_t>& s = *a.pool->slot(si);
  const preco::PrecoWorkspace pws = had::preco_workspace_of<real_t>(s);
  const bic::BicStorage store = had::bic_storage_of<real_t>(s, *a.pool, pws);
  if (!bf::stage_tail(c, store, a.lt, a.pool->fermi, pws)) { return; }
#ifdef G4GPU_TEST_BLIR_FLAT
  {
    bic::BlirStorage bs = had::blir_storage_of<real_t>(s, *a.pool);
    if (bf::ion_pack(a.rec[lane], bs, a.lt, a.pool->fermi, pws, s.blir_fs) !=
        bf::kIonFinalState) {
      return;
    }
    record(s.blir_fs, a.out[a.job_of[lane]]);
  }
#else
  if (!bf::flat_pack_nucleon(c, store.cascade, s.bic_fs)) { return; }
  record(s.bic_fs, a.out[a.job_of[lane]]);
#endif
  c.stage = bf::kStDone;
}
__global__ void k_reap(Args a) {
  const int lane = blockIdx.x * blockDim.x + threadIdx.x;
  if (lane >= a.lanes) { return; }
  Core& c = core(a.rec[lane]);
  if (c.stage == bf::kStThread) {
    Out& o = a.out[a.job_of[lane]];
    o.handed = 1 + c.hand_off;  // 1 + `bf::HandOff`, so the report can say why
    o.status = -2;
    c.stage = bf::kStFree;
  } else if (c.stage == bf::kStDone) {
    c.stage = bf::kStFree;
  }
  if (c.stage != bf::kStFree) { atomicAdd(&a.ctr[3], 1); }
}
__global__ void k_init(Rec* rec, int lanes) {
  const int lane = blockIdx.x * blockDim.x + threadIdx.x;
  if (lane < lanes) { core(rec[lane]).stage = bf::kStFree; }
}
__global__ void k_buffers(bic::CascadeBuffers* buf, const had::InteractionPool<real_t>* pool) {
  *buf = bic::CascadeBuffers{};
  const bic::CascadeSpecies sp = bf::flat_species();
  bic::CascadeRefusal ref;
  const int pions[3] = {211, -211, 111};
  const int nucl[2] = {2212, 2112};
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 2; ++j) {
      bic::ensure_meson_buffer(*buf, pions[i], nucl[j], sp.mass_of(pions[i]),
                               sp.mass_of(nucl[j]), bic::CascadeSpecies::iso3_of(pions[i]),
                               bic::CascadeSpecies::iso3_of(nucl[j]), sp.pi_plus_mass,
                               sp.proton_mass, ref);
    }
  }
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      bic::ensure_nn_buffer(*buf, pool->bic_channels, pool->n_bic_channels, nucl[i], nucl[j],
                            sp.mass_of(nucl[i]), sp.mass_of(nucl[j]), ref);
    }
  }
}

bool factory_enabled() {
  const char* v = std::getenv("G4GPU_FLAT_FACTORY");
  return !(v != nullptr && v[0] == '0');
}

/// The flat campaign: returns the records, `ms` the device time, `rounds` the lockstep length.
std::vector<Out> run_flat(const std::vector<Job>& jobs, const had::InteractionPoolOwner<real_t>& po,
                          const data::LevelTable& dev_lt, int lanes, int perturb, float& ms,
                          int& rounds) {
  const int n = static_cast<int>(jobs.size());
  Args a{};
  a.lanes = lanes;
  a.n = n;
  a.perturb = perturb;
  a.item_capacity = lanes * kPool;
  a.pool = po.d_pool;
  a.lt = dev_lt;
  cudaMalloc(&a.rec, sizeof(Rec) * lanes);
  cudaMalloc(&a.job_of, sizeof(int) * lanes);
  cudaMalloc(&a.buffers, sizeof(bic::CascadeBuffers));
  cudaMalloc(&a.ctr, sizeof(int) * 6);
  cudaMalloc(&a.items, sizeof(int) * a.item_capacity);
  Job* dj = nullptr;
  cudaMalloc(&dj, sizeof(Job) * n);
  cudaMemcpy(dj, jobs.data(), sizeof(Job) * n, cudaMemcpyHostToDevice);
  a.jobs = dj;
  cudaMalloc(&a.out, sizeof(Out) * n);
  cudaMemset(a.out, 0, sizeof(Out) * n);
  cudaMemset(a.ctr, 0, sizeof(int) * 6);
  // P39: the factory's two request lists, every lane's worst case each (one nucleus a Binary
  // set-up, two a light-ion try). `G4GPU_FLAT_FACTORY=0` samples on the lanes as P36 did.
  const bool factory_on = factory_enabled();
  a.solo = factory_on ? 1 : 0;
  constexpr int kNucleiPerTry = kIon ? 2 : 1;
  npool::FactoryLists lists;
  lists.cap = lanes * kNucleiPerTry;
  cudaMalloc(&lists.req32, sizeof(npool::FactoryRequest) * 2 * lists.cap);
  lists.req64 = lists.req32 + lists.cap;
  cudaMalloc(&lists.count, sizeof(int) * 2);
  const int th = 64, bl = (lanes + th - 1) / th;
  k_init<<<bl, th>>>(a.rec, lanes);
  k_buffers<<<1, 1>>>(a.buffers, po.d_pool);
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  cudaEventRecord(e0);
  rounds = 0;
  for (;;) {
    k_claim<<<bl, th>>>(a);
    if (factory_on) {
      cudaMemsetAsync(lists.count, 0, sizeof(int) * 2);
      k_request<<<bl, th>>>(a, lists);
      npool::nucleus_factory_list_kernel<32><<<lists.cap, 32>>>(lists.req32, lists.count,
                                                               lists.cap);
      npool::nucleus_factory_list_kernel<64><<<lists.cap, 64>>>(lists.req64, lists.count + 1,
                                                               lists.cap);
    }
    k_setup<<<bl, th>>>(a);
    cudaMemsetAsync(a.ctr + 1, 0, sizeof(int));
    k_pre<<<bl, th>>>(a);
    k_move<<<(lanes * 4 + 127) / 128, 128>>>(a);
    k_post<<<bl, th>>>(a);
    k_finish<<<bl, th>>>(a);
    if (factory_on) { k_solo<<<bl, th>>>(a); }
    cudaMemsetAsync(a.ctr + 2, 0, sizeof(int));
    k_tail<<<bl, th>>>(a, po.n_slots);
    cudaMemsetAsync(a.ctr + 3, 0, sizeof(int));
    k_reap<<<bl, th>>>(a);
    int h[5];
    if (cudaMemcpy(h, a.ctr, sizeof h, cudaMemcpyDeviceToHost) != cudaSuccess) {
      std::printf("  FATAL: %s\n", cudaGetErrorString(cudaGetLastError()));
      std::exit(1);
    }
    ++rounds;
    if (h[3] == 0 && h[0] >= n) { break; }
  }
  cudaEventRecord(e1);
  cudaEventSynchronize(e1);
  cudaEventElapsedTime(&ms, e0, e1);
  std::vector<Out> out(n);
  cudaMemcpy(out.data(), a.out, sizeof(Out) * n, cudaMemcpyDeviceToHost);
  cudaFree(a.rec); cudaFree(a.job_of); cudaFree(a.buffers);
  int fell[2] = {0, 0};
  cudaMemcpy(fell, a.ctr + 4, sizeof(fell), cudaMemcpyDeviceToHost);
  if (factory_on) {
    std::printf("  (factory on: %d requests fell back to the lane; %d stragglers finished solo)\n",
                fell[0], fell[1]);
  }
  cudaFree(a.ctr); cudaFree(a.items); cudaFree(dj); cudaFree(a.out);
  cudaFree(lists.req32); cudaFree(lists.count);
  return out;
}

// ---------------------------------------------------------------------------------------------
// The thread path, on the host: `bic::apply_yourself` as test_bic_apply calls it.
// ---------------------------------------------------------------------------------------------
struct HostStore {
  std::vector<bic::Nucleon> nucleons;
  std::vector<deex::Vec3d> mom;
  std::vector<double> fermi, flat, pfield, nfield;
  std::vector<bic::NucleusSortEntry> sums;
  std::vector<bic::CascadeTrack> pool;
  std::vector<bic::imr::CollisionInitialState> colls;
  std::vector<bic::imr::ConcreteChannel> chans;
  std::vector<bic::CascadeProduct> products, preco;
  std::vector<deex::Fragment> evap, results, step;
  std::vector<deex::DeexProduct> deex_products, pproducts;
  bic::CascadeBuffers buffers;
  bic::BicScratch scratch;
  bic::BicStorage store;
  std::vector<bic::Nucleon> pnucleons;
  std::vector<bic::BlirProduct> spec, casc;
  std::vector<bic::CascadeTrack> initial;
  bic::BlirStorage ion;
  preco::PrecoWorkspace ws;
  HostStore()
      : nucleons(bic::kSlotNucleons), mom(bic::kSlotNucleons), fermi(bic::kSlotNucleons),
        flat(bic::kFlatBlock), pfield(bic::kMaxFieldTable), nfield(bic::kMaxFieldTable),
        sums(bic::kSlotNucleons), pool(bic::kSlotPoolCap), colls(bic::kSlotCollisionCap),
        chans(bic::imr::kConcreteChannelCount), products(bic::kSlotProductCap),
        preco(bic::kSlotPrecoCap), evap(4096), results(1024), step(512), deex_products(1024),
        pproducts(1024), pnucleons(bic::kSlotNucleons), spec(bic::kSlotSpectatorCap),
        casc(bic::kSlotCascaderCap), initial(bic::kSlotInitialCap) {
    store.nucleons = nucleons.data();
    store.scratch.momentum = mom.data();
    store.scratch.fermi_p = fermi.data();
    store.scratch.test_sums = sums.data();
    store.scratch.flat_block = flat.data();
    store.scratch.capacity = bic::kSlotNucleons;
    store.proton_field = pfield.data();
    store.neutron_field = nfield.data();
    store.field_capacity = bic::kMaxFieldTable;
    store.cascade.pool = pool.data();
    store.cascade.pool_capacity = bic::kSlotPoolCap;
    store.cascade.collisions = colls.data();
    store.cascade.collision_capacity = bic::kSlotCollisionCap;
    store.cascade.channels = chans.data();
    store.cascade.n_channels =
        bic::imr::build_concrete_channels(chans.data(), bic::imr::kConcreteChannelCount);
    store.cascade.buffers = &buffers;
    store.cascade.scratch = &scratch;
    store.cascade.products = products.data();
    store.cascade.product_capacity = bic::kSlotProductCap;
    store.cascade.preco_products = preco.data();
    store.cascade.preco_capacity = bic::kSlotPrecoCap;
    ws.deex.evap_list = evap.data();
    ws.deex.evap_capacity = 4096;
    ws.deex.results = results.data();
    ws.deex.results_capacity = 1024;
    ws.deex.step = step.data();
    ws.deex.step_capacity = 512;
    ws.deex.products = deex_products.data();
    ws.deex.products_capacity = 1024;
    ws.products = pproducts.data();
    ws.products_capacity = 1024;
    ion.projectile_nucleons = pnucleons.data();
    ion.target_nucleons = nucleons.data();
    ion.scratch = store.scratch;
    ion.proton_field = pfield.data();
    ion.neutron_field = nfield.data();
    ion.field_capacity = bic::kMaxFieldTable;
    ion.cascade = store.cascade;
    ion.spectators = spec.data();
    ion.spectator_capacity = bic::kSlotSpectatorCap;
    ion.cascaders = casc.data();
    ion.cascader_capacity = bic::kSlotCascaderCap;
    ion.initial = initial.data();
    ion.initial_capacity = bic::kSlotInitialCap;
  }
};

double proj_mass(int pdg, double ion_mass = 0.0) {
  if (kIon) { return ion_mass; }
  return (pdg == 2212)   ? units::proton_mass_c2<double>()
       : (pdg == 2112) ? units::neutron_mass_c2<double>()
                         : bic::pdg_mass_pion_charged();
}

void run_thread_one(const Job& j, HostStore& hs, const data::LevelTable& lt,
                    const deex::FermiPool& fp, Out& o) {
  physics::hadronic::HadProjectile<double> proj;
  proj.pdg = j.pdg;
  proj.mass = proj_mass(j.pdg);
  proj.charge = (j.pdg == 2212 || j.pdg == 211) ? 1.0 : ((j.pdg == -211) ? -1.0 : 0.0);
  proj.kin_energy = j.ekin;
  proj.baryon_number = (j.pdg == 2212 || j.pdg == 2112) ? 1 : 0;
  physics::hadronic::HadNucleus tgt{j.tz, j.ta, 0};
  Philox<double> rng(j.key, 7u, had::kInteractionRngPurpose);
  if constexpr (kIon) {
    proj.mass = j.mass;
    proj.charge = j.charge;
    proj.baryon_number = j.pa;
    bic::BlirFinalState bfs;
    bic::BlirRefusal bref;
    bic::BlirReport brep;
    bic::blir_apply_yourself(proj, tgt, lt, fp, hs.ws, hs.ion, rng, bfs, bref, brep);
    // `run_arm_light_ion`'s mapping: these are refusals; the rest are Geant4's answers.
    if (bref.cascade || bref.anti_or_hyper || bref.capacity || bref.nucleus ||
        (bref.momentum_not_conserved && brep.correction_gave_up)) {
      o = Out{};
      o.status = -1;
      return;
    }
    record(bfs, o);
    return;
  }
  bic::BicFinalState fs;
  bic::BicRefusal ref;
  bic::BicReport rep;
  bic::apply_yourself(proj, tgt, lt, fp, hs.ws, hs.store, rng, fs, ref, rep);
  if (ref.any()) {
    o = Out{};
    o.status = -1;
    return;
  }
  record(fs, o);
}

void run_thread(const std::vector<Job>& jobs, const std::vector<int>& which,
                const data::LevelTable& lt, const deex::FermiPool& fp, std::vector<Out>& out,
                int n_threads) {
  std::vector<std::thread> th;
  for (int t = 0; t < n_threads; ++t) {
    th.emplace_back([&, t] {
      HostStore* hs = new HostStore();
      for (std::size_t k = t; k < which.size(); k += n_threads) {
        const int i = which[k];
        run_thread_one(jobs[i], *hs, lt, fp, out[i]);
      }
      delete hs;
    });
  }
  for (auto& x : th) { x.join(); }
}

// ---------------------------------------------------------------------------------------------
// The statistics.
// ---------------------------------------------------------------------------------------------
int g_fails = 0;
void check(bool ok, const char* what) {
  if (!ok) {
    std::printf("  FAIL: %s\n", what);
    ++g_fails;
  }
}

enum Sp { kP, kN, kPip, kPim, kPi0, kGam, kD, kT, kHe3, kAlpha, kFrag, kOther, kNSp };
const char* sp_name[kNSp] = {"p", "n", "pi+", "pi-", "pi0", "gamma", "d", "t", "He3", "alpha",
                             "A>4", "other"};
int species_of(int pdg, int a) {
  switch (pdg) {
    case 2212: return kP;
    case 2112: return kN;
    case 211: return kPip;
    case -211: return kPim;
    case 111: return kPi0;
    case 22: return kGam;
    case 1000010020: return kD;
    case 1000010030: return kT;
    case 1000020030: return kHe3;
    case 1000020040: return kAlpha;
    default: return (a > 4) ? kFrag : kOther;
  }
}

/// A set of rows for one sample: every quantity is accumulated as (sum, sum of squares) per
/// event, so each row's mean and its variance come from the same N.
struct Stats {
  static constexpr int kKe = 12, kCos = 10, kSize = 12;
  double n = 0;
  double mult[kNSp] = {}, mult2[kNSp] = {};
  double ke[5][kKe] = {}, ke2[5][kKe] = {};       ///< p, n, pi+, pi-, pi0: per-event counts
  double cs[5][kCos] = {}, cs2[5][kCos] = {};
  double size[kSize] = {}, size2[kSize] = {};
  static constexpr int kBal = 9;
  double ebal[kBal] = {}, abal[5] = {};
  void add(const Out& o, double e_in, int a_in) {
    if (o.status < 0) { return; }
    n += 1;
    double m[kNSp] = {};
    double k[5][kKe] = {}, c[5][kCos] = {};
    for (int i = 0; i < o.n; ++i) {
      const int s = species_of(o.pdg[i], o.a[i]);
      m[s] += 1;
      if (s <= kPi0) {
        const double e = (o.ekin[i] > 0.1f) ? o.ekin[i] : 0.1;
        int b = static_cast<int>(std::floor(std::log10(e / 0.1) * 3.0));  // 0.1 MeV .. 1e4
        b = (b < 0) ? 0 : ((b >= kKe) ? kKe - 1 : b);
        k[s][b] += 1;
        int cb = static_cast<int>((o.cost[i] + 1.0) * 0.5 * kCos);
        cb = (cb < 0) ? 0 : ((cb >= kCos) ? kCos - 1 : cb);
        c[s][cb] += 1;
      }
    }
    for (int s = 0; s < kNSp; ++s) { mult[s] += m[s]; mult2[s] += m[s] * m[s]; }
    for (int s = 0; s < 5; ++s) {
      for (int b = 0; b < kKe; ++b) { ke[s][b] += k[s][b]; ke2[s][b] += k[s][b] * k[s][b]; }
      for (int b = 0; b < kCos; ++b) { cs[s][b] += c[s][b]; cs2[s][b] += c[s][b] * c[s][b]; }
    }
    int sb = o.n / 2;
    sb = (sb >= kSize) ? kSize - 1 : sb;
    size[sb] += 1;
    size2[sb] += 1;
    const double de = o.e_sum - e_in;
    {
      const double edge[kBal - 1] = {-100, -10, -1, -0.01, 0.01, 1, 10, 100};
      int b = 0;
      while (b < kBal - 1 && de >= edge[b]) { ++b; }
      ebal[b] += 1;
    }
    const double da = o.a_sum - a_in;
    {
      int b = static_cast<int>(da) + 2;
      b = (b < 0) ? 0 : ((b > 4) ? 4 : b);
      abal[b] += 1;
    }
  }
};

struct Worst {
  double z = 0;
  std::string row;
  int rows = 0;
  std::vector<std::pair<double, std::string>> all;
  void print(const char* what) const {
    std::vector<std::pair<double, std::string>> s = all;
    std::sort(s.begin(), s.end(), [](const auto& x, const auto& y) { return std::fabs(x.first) > std::fabs(y.first); });
    for (std::size_t i = 0; i < s.size() && i < 5; ++i) {
      std::printf("    %s worst %zu: %+.2f sigma  %s\n", what, i + 1, s[i].first, s[i].second.c_str());
    }
  }
  void see(double z_, const std::string& r) {
    ++rows;
    all.emplace_back(z_, r);
    if (std::fabs(z_) > std::fabs(z)) {
      z = z_;
      row = r;
    }
  }
};

/// z of two event means, from per-event sums: (a - b) / sqrt(var_a/Na + var_b/Nb).
double zmean(double sa, double s2a, double na, double sb, double s2b, double nb) {
  if (na < 2 || nb < 2) { return 0; }
  const double ma = sa / na, mb = sb / nb;
  const double va = (s2a / na - ma * ma) / na, vb = (s2b / nb - mb * mb) / nb;
  if (va + vb <= 0) { return (ma == mb) ? 0.0 : 1e9; }
  return (ma - mb) / std::sqrt(va + vb);
}

void compare(const std::string& point, const Stats& a, const Stats& b, Worst& w) {
  auto row = [&](const std::string& name, double sa, double s2a, double sb, double s2b) {
    if (sa < 25 && sb < 25) { return; }
    w.see(zmean(sa, s2a, a.n, sb, s2b, b.n), point + " " + name);
  };
  for (int s = 0; s < kNSp; ++s) {
    row(std::string("mult ") + sp_name[s], a.mult[s], a.mult2[s], b.mult[s], b.mult2[s]);
  }
  for (int s = 0; s < 5; ++s) {
    for (int k = 0; k < Stats::kKe; ++k) {
      char nm[64];
      std::snprintf(nm, sizeof nm, "ke %s bin %d", sp_name[s], k);
      row(nm, a.ke[s][k], a.ke2[s][k], b.ke[s][k], b.ke2[s][k]);
    }
    for (int k = 0; k < Stats::kCos; ++k) {
      char nm[64];
      std::snprintf(nm, sizeof nm, "cos %s bin %d", sp_name[s], k);
      row(nm, a.cs[s][k], a.cs2[s][k], b.cs[s][k], b.cs2[s][k]);
    }
  }
  for (int k = 0; k < Stats::kSize; ++k) {
    char nm[64];
    std::snprintf(nm, sizeof nm, "size %d-%d", 2 * k, 2 * k + 1);
    row(nm, a.size[k], a.size2[k], b.size[k], b.size2[k]);
  }
  for (int k = 0; k < Stats::kBal; ++k) {
    char nm[64];
    std::snprintf(nm, sizeof nm, "energy balance bin %d", k);
    row(nm, a.ebal[k], a.ebal[k], b.ebal[k], b.ebal[k]);
  }
  for (int k = 0; k < 5; ++k) {
    char nm[64];
    std::snprintf(nm, sizeof nm, "baryon balance %+d", k - 2);
    row(nm, a.abal[k], a.abal[k], b.abal[k], b.abal[k]);
  }
}

struct Point {
  std::string name;   ///< the oracle's case name
  int pdg;
  double ekin;
  int tz, ta;
  int pz = 0, pa = 0;   ///< an ion's
  double mass = 0.0;    ///< an ion's
};

std::vector<Point> points() {
  std::vector<Point> p;
  if (kIon) {
    // ref/oracle/bic_blirapply.csv's cases (20,000 events each in Geant4), the light ions and the
    // heavy one on the B1 materials' nuclei and on Al27: d, alpha, C12 and Fe56.
    struct I { const char* name; int pz, pa; double e_per_a; int tz, ta; };
    const I ions[] = {{"ic_d1000_O16", 1, 2, 1000, 8, 16},   {"ic_a200_C12", 2, 4, 200, 6, 12},
                      {"ic_a1000_O16", 2, 4, 1000, 8, 16},   {"ic_C12200_O16", 6, 12, 200, 8, 16},
                      {"ic_C121000_C12", 6, 12, 1000, 6, 12}, {"ic_Fe56_1000_Al27", 26, 56, 1000, 13, 27}};
    for (const I& i : ions) {
      Point q{i.name, physics::hadronic::pdg_nuclear_code(i.pz, i.pa), i.e_per_a * i.pa, i.tz, i.ta};
      q.pz = i.pz;
      q.pa = i.pa;
      q.mass = deex::nuclear_mass(i.pa, i.pz);
      p.push_back(q);
    }
    return p;
  }
  struct T { const char* n; int z, a; };
  const T targets[] = {{"C12", 6, 12}, {"O16", 8, 16}, {"Al27", 13, 27}, {"Fe56", 26, 56}};
  for (const T& t : targets) {
    const bool all = (t.a <= 16);
    for (int e : {100, 400, 800, 1400}) {
      if (all || e == 400) { p.push_back({std::string("camp_p") + std::to_string(e) + "_" + t.n, 2212, double(e), t.z, t.a}); }
      if (all || e == 800) { p.push_back({std::string("camp_n") + std::to_string(e) + "_" + t.n, 2112, double(e), t.z, t.a}); }
    }
    for (int e : {200, 800}) {
      if (all || e == 200) { p.push_back({std::string("camp_pip") + std::to_string(e) + "_" + t.n, 211, double(e), t.z, t.a}); }
      if (all || e == 800) { p.push_back({std::string("camp_pim") + std::to_string(e) + "_" + t.n, -211, double(e), t.z, t.a}); }
    }
  }
  return p;
}

std::vector<std::vector<std::string>> read_csv(const std::string& path) {
  std::vector<std::vector<std::string>> rows;
  std::ifstream f(path);
  std::string line;
  bool header = true;
  while (std::getline(f, line)) {
    if (header) { header = false; continue; }
    std::vector<std::string> r;
    std::stringstream ss(line);
    std::string x;
    while (std::getline(ss, x, ',')) { r.push_back(x); }
    rows.push_back(r);
  }
  return rows;
}

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  std::printf(kIon ? "== the flat light-ion reaction against the thread path and Geant4 (P36) ==\n"
                   : "== the flat Binary cascade against the thread path and Geant4 (P36) ==\n");
  int per_point = 20000;
  if (argc > 1 && std::atoi(argv[1]) > 0) { per_point = std::atoi(argv[1]); }
  const std::string pe = host::g4photon_evaporation_dir();
  if (pe.empty()) {
    std::printf("  skipped: no PhotonEvaporation dataset\nOK\n");
    return 0;
  }
  // The host's tables, for the thread path.
  static data::LevelTableStorage lts;
  data::read_all_level_data(
      lts, pe, data::kLevelZMax, [](int z, int a) { return deex::shell_correction(a, z); },
      [](int z, int a) { return deex::level_manager_level_density(z, a); });
  const data::LevelTable host_lt = lts.view();
  static deex::FermiPoolStorage fps;
  deex::build_fermi_pool(fps, host_lt);
  const deex::FermiPool host_fp = fps.view();
  // The device's, for the flat path: the transport's own pool builder.
  static data::LevelTableStorage dstorage;
  host::LevelTableOwner lown = host::upload_level_data(dstorage, data::kLevelZMax, false);
  had::ftf_slot_request() = 1;
  constexpr int kSlots = 512;
  had::InteractionPoolOwner<real_t> pown =
      had::build_interaction_pool<real_t>(kSlots, dstorage.view(), false);
  cudaDeviceSetLimit(cudaLimitStackSize, 28672);

  const std::vector<Point> pts = points();
  std::vector<Job> jobs;
  for (std::size_t p = 0; p < pts.size(); ++p) {
    for (int k = 0; k < per_point; ++k) {
      Job j;
      j.pdg = pts[p].pdg;
      j.pz = pts[p].pz;
      j.pa = pts[p].pa;
      j.mass = pts[p].mass;
      j.charge = pts[p].pz;
      j.ekin = pts[p].ekin;
      j.tz = pts[p].tz;
      j.ta = pts[p].ta;
      j.key = static_cast<uint32_t>(p + 1) * 1000003u + static_cast<uint32_t>(k);
      jobs.push_back(j);
    }
  }
  const int n = static_cast<int>(jobs.size());
  std::printf("  %zu points x %d calls = %d\n", pts.size(), per_point, n);

  int lanes = 8192;
  if (const char* v = std::getenv("G4GPU_FLAT_TEST_LANES")) { lanes = std::atoi(v); }
  float ms = 0.f;
  int rounds = 0;
  std::vector<Out> flat = run_flat(jobs, pown, lown.view, lanes, 0, ms, rounds);
  std::printf("  flat: %d calls on %d lanes in %.1f ms, %d rounds\n", n, lanes, ms, rounds);
  float ms2 = 0.f;
  int rounds2 = 0;
  const std::vector<Out> flat2 = run_flat(jobs, pown, lown.view, lanes, 0, ms2, rounds2);
  int differ = 0;
  for (int i = 0; i < n; ++i) {
    if (flat[i].hash != flat2[i].hash || flat[i].status != flat2[i].status) { ++differ; }
  }
  std::printf("  flat twice: %d of %d final states differ\n", differ, n);
  check(differ == 0, "the flat path reproduces itself to the bit");
  // A second run at another width: which lane an entry gets moves nothing.
  float ms3 = 0.f;
  int rounds3 = 0;
  const std::vector<Out> flat3 = run_flat(jobs, pown, lown.view, lanes / 2 + 7, 0, ms3, rounds3);
  differ = 0;
  for (int i = 0; i < n; ++i) {
    if (flat[i].hash != flat3[i].hash || flat[i].status != flat3[i].status) { ++differ; }
  }
  std::printf("  flat at %d lanes: %d of %d final states differ from %d lanes\n",
              lanes / 2 + 7, differ, n, lanes);
  check(differ == 0, "the flat path's answer does not depend on its width");

  // The thread path, on the host. TWICE: once on the flat campaign's own keys - the hand-backs'
  // answers, and the pairing that says how close the two paths are cascade by cascade - and once
  // on keys of its own, the INDEPENDENT sample every z-score below is taken against. (V309: on
  // the same keys the two samples share their draws, the rows are correlated, and the z-scores
  // come out far under 1 - a comparison that could not have seen anything.)
  const int host_threads = 8;
  std::vector<int> all(n), handed;
  int why[bf::kNumHandOff] = {};
  for (int i = 0; i < n; ++i) {
    all[i] = i;
    if (flat[i].handed) {
      handed.push_back(i);
      const int r = flat[i].handed - 1;
      if (r >= 0 && r < bf::kNumHandOff) { ++why[r]; }
    }
  }
  std::vector<Out> paired(n);
  run_thread(jobs, all, host_lt, host_fp, paired, host_threads);
  for (int i : handed) { flat[i] = paired[i]; }
  std::printf("  %zu of %d calls handed back by the flat path (answered by the thread path)\n",
              handed.size(), n);
  std::printf("    by reason: straggler %d, capacity %d, refused %d, hydrogen %d, "
              "PreCompound or fusion %d, re-entry %d\n",
              why[bf::kHandStraggler], why[bf::kHandCapacity], why[bf::kHandRefused],
              why[bf::kHandHydrogen], why[bf::kHandPrecompound], why[bf::kHandReentry]);
  int same = 0;
  for (int i = 0; i < n; ++i) { same += (flat[i].hash == paired[i].hash) ? 1 : 0; }
  std::printf("  %d of %d flat final states are the host thread path's on the same key, to the "
              "bit\n", same, n);
  std::vector<Job> jobs_ind = jobs;
  for (Job& j : jobs_ind) { j.key ^= 0x5BD1E995u; }
  std::vector<Out> thread(n);
  run_thread(jobs_ind, all, host_lt, host_fp, thread, host_threads);

  auto against = [&](const std::vector<Out>& x, const std::vector<Out>& y, Worst& w) {
    for (std::size_t p = 0; p < pts.size(); ++p) {
      Stats a, b;
      const double m = proj_mass(pts[p].pdg, pts[p].mass);
      const double e_in = pts[p].ekin + m + deex::nuclear_mass(pts[p].ta, pts[p].tz);
      const int a_in = pts[p].ta + (kIon ? pts[p].pa : ((pts[p].pdg == 2212 || pts[p].pdg == 2112) ? 1 : 0));
      for (int k = 0; k < per_point; ++k) {
        a.add(x[p * per_point + k], e_in, a_in);
        b.add(y[p * per_point + k], e_in, a_in);
      }
      compare(pts[p].name, a, b, w);
    }
  };
  Worst wt;
  against(flat, thread, wt);
  std::printf("  flat against thread (independent keys): %d rows, the worst %+.2f sigma (%s)\n",
              wt.rows, wt.z, wt.row.c_str());
  wt.print("thread");
  check(std::fabs(wt.z) < 5.0, "no flat-against-thread row past 5 sigma");

  // Against Geant4: the oracle's mean multiplicity and mean kinetic energy per (case, species),
  // a nucleus keyed by (Z, A) and its isomer rows summed (at most one residual of a given (Z, A)
  // in an event, so the per-event second moments add).
  struct ORow {
    double n = 0, count = 0, ke = 0, ke2 = 0, m2n = 0;
  };
  std::map<std::pair<int, long long>, ORow> orows;  // (point, key)
  {
    const char* od = std::getenv("G4GPU_ORACLE");
    const std::string dir = od ? od : "ref/oracle";
    std::map<std::string, int> pidx;
    for (std::size_t p = 0; p < pts.size(); ++p) { pidx[pts[p].name] = static_cast<int>(p); }
    for (const auto& r : read_csv(dir + (kIon ? "/bic_blirapply.csv" : "/bic_apply.csv"))) {
      if (r.size() < 12) { continue; }
      auto it = pidx.find(r[0]);
      if (it == pidx.end()) { continue; }
      const long long pdg = std::atoll(r[7].c_str());
      // A nucleus as 2e9 + 1000 Z + A, clear of every particle code (pi- is -211).
      const long long key = (pdg > 1000000000LL)
                                ? 2000000000LL + ((pdg / 10000) % 1000) * 1000 + (pdg / 10) % 1000
                                : pdg;
      ORow& o = orows[{it->second, key}];
      o.n = std::atof(r[6].c_str());
      const double c = std::atof(r[8].c_str());
      o.count += c;
      o.ke += c * std::atof(r[9].c_str());
      o.ke2 += c * std::atof(r[10].c_str());
      o.m2n += o.n * std::atof(r[11].c_str());
    }
  }
  auto oracle = [&](const std::vector<Out>& res, Worst& w) {
    for (const auto& kv : orows) {
      const int p = kv.first.first;
      const long long key = kv.first.second;
      const ORow& g = kv.second;
      double cnt = 0, cnt2 = 0, ke = 0, ke2 = 0, nn = 0;
      for (int k = 0; k < per_point; ++k) {
        const Out& o = res[p * per_point + k];
        // The oracle tallies its killed events only (test_bic_apply.cu's `n_kill`): an event
        // that returned the primary alive has no products and is not in its denominator.
        if (o.status != static_cast<int>(physics::hadronic::HadFinalStateStatus::kStopAndKill)) {
          continue;
        }
        nn += 1;
        double c = 0;
        for (int i = 0; i < o.n; ++i) {
          const bool match = (key >= 2000000000LL)
                                 ? (o.a[i] > 0 && (o.pdg[i] == 0 || o.pdg[i] > 1000000000) &&
                                    2000000000LL + o.z[i] * 1000 + o.a[i] == key)
                                 : (o.pdg[i] == key);
          if (!match) { continue; }
          c += 1;
          ke += o.ekin[i];
          ke2 += double(o.ekin[i]) * o.ekin[i];
        }
        cnt += c;
        cnt2 += c * c;
      }
      if (cnt < 25 && g.count < 25) { continue; }
      char nm[96];
      std::snprintf(nm, sizeof nm, "%s %s %lld", pts[p].name.c_str(), "mult",
                    (key >= 2000000000LL) ? key - 2000000000LL : key);
      const double zmul = zmean(cnt, cnt2, nn, g.count, g.m2n, g.n);
      if (std::fabs(zmul) > 5.0 && std::getenv("G4GPU_FLAT_TEST_VERBOSE") != nullptr) {
        std::printf("    %s: port %.0f in %.0f killed events (%.4f), Geant4 %.0f in %.0f (%.4f)\n",
                    nm, cnt, nn, cnt / (nn > 0 ? nn : 1), g.count, g.n,
                    g.count / (g.n > 0 ? g.n : 1));
      }
      w.see(zmul, nm);
      if (cnt >= 25 && g.count >= 25) {
        const double ma = ke / cnt, va = (ke2 / cnt - ma * ma) / cnt;
        const double mb = g.ke / g.count, vb = (g.ke2 / g.count - mb * mb) / g.count;
        std::snprintf(nm, sizeof nm, "%s %s %lld", pts[p].name.c_str(), "ke",
                      (key >= 2000000000LL) ? key - 2000000000LL : key);
        w.see((ma - mb) / std::sqrt(va + vb), nm);
      }
    }
  };
  Worst wo, wot;
  oracle(flat, wo);
  oracle(thread, wot);
  std::printf(kIon ? "  flat against Geant4 (bic_blirapply.csv): %d rows, the worst %+.2f sigma (%s)\n" : "  flat against Geant4 (bic_apply.csv): %d rows, the worst %+.2f sigma (%s)\n",
              wo.rows, wo.z, wo.row.c_str());
  wo.print("Geant4");
  std::printf("  (the thread path against Geant4, the same rows: the worst %+.2f sigma (%s))\n",
              wot.z, wot.row.c_str());
  check(wo.rows > 50, "the oracle comparison has rows");
  check(std::fabs(wo.z) < 5.0, "no flat-against-Geant4 row past 5 sigma");

  // Anti-vacuity: a flat path whose `FindCollisions` drops `G4MesonAbsorption` - one lost call in the
  // re-composed glue, the kind of defect the copies in cascade_flat.cuh could carry - must be
  // caught. (V309: `Capture` skipped is NOT caught at 2,000 calls a point - 3.4 sigma at most.)
  {
    float msp = 0.f;
    int rp = 0;
    // The ion arm moves few pions, and `Capture` skipped is not caught on it either (V310: 3.85
    // and 3.16 sigma), so its perturbation is 3, the move items' off-by-one (`k_pre`); the
    // nucleon arm's is the absorption dropped (2).
    std::vector<Out> pert = run_flat(jobs, pown, lown.view, lanes, kIon ? 3 : 2, msp, rp);
    for (int i = 0; i < n; ++i) {
      if (pert[i].handed) { pert[i] = paired[i]; }
    }
    Worst wp;
    for (std::size_t p = 0; p < pts.size(); ++p) {
      Stats a, b;
      const double m = proj_mass(pts[p].pdg, pts[p].mass);
      const double e_in = pts[p].ekin + m + deex::nuclear_mass(pts[p].ta, pts[p].tz);
      const int a_in = pts[p].ta + (kIon ? pts[p].pa : ((pts[p].pdg == 2212 || pts[p].pdg == 2112) ? 1 : 0));
      for (int k = 0; k < per_point; ++k) {
        a.add(pert[p * per_point + k], e_in, a_in);
        b.add(thread[p * per_point + k], e_in, a_in);
      }
      compare(pts[p].name, a, b, wp);
    }
    std::printf(kIon ? "  ANTI-VACUITY (the step's last secondary not moved): the worst row %+.2f sigma (%s)\n" : "  ANTI-VACUITY (G4MesonAbsorption dropped from FindCollisions): the worst row %+.2f sigma (%s)\n", wp.z,
                wp.row.c_str());
    wp.print("perturbed");
    check(std::fabs(wp.z) > 5.0, "the campaign catches the perturbed flat path");
  }

  had::free_interaction_pool(pown);
  host::free_level_data(lown);
  std::printf("%s\n", g_fails == 0 ? "OK" : "FAILURES");
  return g_fails == 0 ? 0 : 1;
}
