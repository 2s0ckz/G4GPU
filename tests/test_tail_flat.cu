// P41: the de-excitation tail on the wide tail pool's compact workspace against the thread path's
// `deexcite`, residual class by residual class (docs/RISK.md V326's rule (3) comparison).
//
// WHAT IT RUNS. For each residual class the proton and Fe56 beams produce (V325's tape: the
// Binary cascade's residuals, the light-ion reaction's cascade residuals and its projectile
// spectators), N = 20,000 residuals two ways:
//
//   * POOL: on the device, one lane per residual on an entry of the Binary tail pool
//     (deexcitation/flat/tail_pool.cuh: the entry's de-excitation workspace and PreCompound
//     product list at the pool's capacities) - exactly the call `bic::flat::stage_tail` makes on
//     an entry (`bic_deexcite_fragment` for a cascade residual, `deex::deexcite` for a spectator,
//     as `blir_deexcite_spectator` makes it), with the pool's capacity rule (a buffer that binds
//     hands the residual back - counted here, and required to be none);
//   * THREAD: the same functions on the host with the slot's capacities (4,096 / 1,024 / 512 /
//     1,024 / 1,024 and 256 PreCompound products), the thread path's code and buffers.
//
// A residual is at rest, its (Z, A) the class's, its excitation uniform over the class's bin and
// its exciton numbers the class's tape means; the two sides draw from INDEPENDENT keys (V309: on
// shared keys the samples share their draws and the z-scores mean nothing).
//
// WHAT IT COMPARES, each row a z-score of two event means: per species (n, p, d, t, He3, alpha,
// gamma, e-, A > 4) the multiplicity; the kinetic-energy spectrum in 12 logarithmic bins for n,
// p, alpha, gamma and the A > 4 fragments; the number of products in 12 bins; the summed
// kinetic energy. NO ROW PAST 5 SIGMA.
//
// DETERMINISM: the pool campaign twice and at a second width (lanes striding the residuals),
// every residual's products the same to the bit. ANTI-VACUITY: the pool path with the last
// product of every de-excitation dropped in its hand-back (an off-by-one in the copy out of the
// entry) must be caught past 5 sigma.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <string>
#include <thread>
#include <vector>

#include "core/rng.cuh"
#include "host/g4data.cuh"
#include "host/interaction_pool_build.cuh"
#include "host/level_upload.cuh"
#include "physics/hadronic/deexcitation/flat/tail_pool.cuh"
#include "physics/hadronic/interaction_apply.cuh"

using namespace g4gpu;
namespace dfl = g4gpu::deex::flat;

namespace g4gpu::had {
template InteractionPoolOwner<double> build_interaction_pool<double>(int, const data::LevelTable&,
                                                                     bool);
template void free_interaction_pool<double>(InteractionPoolOwner<double>&);
}  // namespace g4gpu::had

namespace {

using Entry = dfl::TailEntry<dfl::BinaryTailCaps, bic::BicFinalState>;
constexpr int kMaxP = 96;

/// One residual class: what V325's tape found the beams make.
struct Class {
  const char* name;
  int kind;          ///< 0 a cascade residual (PreCompound + handler), 1 a spectator (handler)
  int z, a;
  double ex_lo, ex_hi;   ///< MeV
  int particles, holes, charged;
};

struct Res {
  int kind = 0;
  int z = 0, a = 0, particles = 0, holes = 0, charged = 0;
  double ex = 0.0;
  uint32_t key = 0;
};

struct ROut {
  int n = 0;
  int handed = 0;
  int continued = 0;   ///< continued on a slot-sized workspace
  // V325's phase counters, from the two status reports: PreCompound's transitions and
  // emissions, the handler's BreakFragment passes, the draws and the products.
  int transitions = 0, emissions = 0, evap_steps = 0, draws = 0;
  long long clk = 0;
  int pdg[kMaxP];
  int a[kMaxP];
  int z[kMaxP];
  float ke[kMaxP];
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

/// The residual as the cascade hands it over: `G4Fragment(A, Z, p)` at rest with E*.
__host__ __device__ inline bic::CascadeFragment fragment_of(const Res& r) {
  bic::CascadeFragment f;
  f.a = r.a;
  f.z = r.z;
  f.holes = r.holes;
  f.particles = r.particles;
  f.charged = r.charged;
  f.momentum = bic::imr::LorentzVector(deex::Vec3d{0.0, 0.0, 0.0},
                                       deex::nuclear_mass(r.a, r.z) + r.ex);
  return f;
}

/// One residual's de-excitation, the products into `o`. `out`/`cap` is the PreCompound product
/// list, `ws` the workspace - the entry's on the device, the slot-sized ones on the host. With
/// `fb` (the pool), a compact workspace that binds is CONTINUED as the transport's tail does it:
/// the generator put back, the same call on a slot-sized workspace (`o.continued`); `o.handed`
/// is a residual past the continuation's bound, or a product list that overflowed.
__host__ __device__ inline void deexcite_one(const Res& r, const data::LevelTable& lt,
                                             const deex::FermiPool& fp,
                                             const preco::PrecoWorkspace& ws,
                                             bic::CascadeProduct* out, int cap,
                                             const bic::flat::TailFallback* fb, int perturb,
                                             ROut& o) {
  Philox<double> rng(r.key, 11u, had::kInteractionRngPurpose);
  const Philox<double> rng_before = rng;
  o.n = 0;
  o.handed = 0;
  o.continued = 0;
  uint64_t h = 0xCBF29CE484222325ull;
  auto put = [&](int pdg, int a, int z, const bic::imr::LorentzVector& p, double mass) {
    if (o.n >= kMaxP) { return; }
    o.pdg[o.n] = pdg;
    o.a[o.n] = a;
    o.z[o.n] = z;
    const double ke = p.e - mass;
    o.ke[o.n] = static_cast<float>(ke > 0.0 ? ke : 0.0);
    mix(h, static_cast<uint64_t>(static_cast<int64_t>(pdg)));
    mix(h, bits(p.e));
    mix(h, bits(p.v.x));
    mix(h, bits(p.v.z));
    ++o.n;
  };
  if (r.kind == 0) {
    preco::PrecoStatus st;
    bool overflow = false;
    int n = bic::bic_deexcite_fragment(fragment_of(r), out, cap, lt, fp, ws, rng, st, overflow);
    if (fb != nullptr && bic::flat::tail_capacity_refused(st)) {
      const int k = fb->acquire();
      if (k < 0) {
        o.handed = 1;
      } else {
        o.continued = 1;
        rng = rng_before;
        overflow = false;
        st = preco::PrecoStatus{};
        n = bic::bic_deexcite_fragment(fragment_of(r), out, cap, lt, fp, fb->big[k], rng, st,
                                       overflow);
        fb->release(k);
      }
    }
    o.transitions = st.n_transitions;
    o.emissions = st.n_emissions;
    o.evap_steps = st.deex.n_evaporation_steps;
    if (overflow) { o.handed = 1; }
    if (perturb == 1 && n > 0) { --n; }
    for (int i = 0; i < n; ++i) {
      put(out[i].pdg, out[i].nucleus_a, out[i].nucleus_z, out[i].momentum, out[i].pdg_mass);
    }
  } else {
    // `blir_deexcite_spectator`'s call: the spectator at rest with its statistical excitation.
    deex::Fragment f;
    const double e = deex::nuclear_mass(r.a, r.z) + r.ex;
    f.set_za_and_momentum(bic::imr::LorentzVector(deex::Vec3d{0.0, 0.0, 0.0}, e), r.z, r.a);
    deex::DeexStatus st = deex::deexcite(f, lt, fp, ws.deex, rng);
    const deex::DeexWorkspace* used = &ws.deex;
    int k = -1;
    if (fb != nullptr && st.refused_capacity) {
      k = fb->acquire();
      if (k < 0) {
        o.handed = 1;
      } else {
        o.continued = 1;
        rng = rng_before;
        used = &fb->big[k].deex;
        st = deex::deexcite(f, lt, fp, *used, rng);
      }
    }
    o.evap_steps = st.n_evaporation_steps;
    int n = st.n_products;
    if (perturb == 1 && n > 0) { --n; }
    for (int i = 0; i < n; ++i) {
      const deex::DeexProduct& d = used->products[i];
      const double m = (d.a > 0) ? (deex::nuclear_mass(d.a, d.z) + d.excitation)
                                 : ((d.pdg == deex::kPdgElectron)
                                        ? units::electron_mass_c2<double>() : 0.0);
      put(d.pdg, d.a, d.z, d.momentum, m);
    }
    if (k >= 0) { fb->release(k); }
  }
  o.hash = h;
  {
    const uint32_t* w = reinterpret_cast<const uint32_t*>(&rng);
    const uint32_t* w0 = reinterpret_cast<const uint32_t*>(&rng_before);
    o.draws = static_cast<int>((4u * w[0] + w[10]) - (4u * w0[0] + w0[10]));
  }
}

__global__ void k_pool(const Res* res, int n, Entry* entries, ROut* out,
                       const had::InteractionPool<double>* pool, data::LevelTable lt,
                       bic::flat::TailFallback fb, int perturb, int tiny) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  Entry& e = entries[tid];
  preco::PrecoWorkspace ws = dfl::tail_deex_workspace(e);
  if (tiny) {
    // The continuation's own check: a workspace so small that nearly every residual binds it
    // and is continued on a slot-sized one - which must give every residual's products to the bit.
    ws.deex.evap_capacity = 1;
    ws.deex.results_capacity = 2;
    ws.deex.step_capacity = 2;
    ws.deex.products_capacity = 2;
    ws.products_capacity = 2;
  }
  for (int i = tid; i < n; i += gridDim.x * blockDim.x) {
    const long long t0 = clock64();
    deexcite_one(res[i], lt, pool->fermi, ws, e.preco_products, dfl::BinaryTailCaps::kPreco, &fb,
                 perturb, out[i]);
    out[i].clk = clock64() - t0;
  }
}

std::vector<ROut> run_pool(const std::vector<Res>& res, const had::InteractionPoolOwner<double>& po,
                           const data::LevelTable& lt, int width, int perturb, float& ms,
                           int tiny = 0) {
  const int n = static_cast<int>(res.size());
  Res* dr = nullptr;
  ROut* dout = nullptr;
  Entry* de = nullptr;
  cudaMalloc(&dr, sizeof(Res) * n);
  cudaMalloc(&dout, sizeof(ROut) * n);
  cudaMalloc(&de, sizeof(Entry) * width);
  cudaMemcpy(dr, res.data(), sizeof(Res) * n, cudaMemcpyHostToDevice);
  const dfl::BigWorkspaces big = dfl::build_big_workspaces(64);
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  cudaEventRecord(e0);
  const int th = 32;
  k_pool<<<width / th, th>>>(dr, n, de, dout, po.d_pool, lt, dfl::fallback_of(big), perturb,
                              tiny);
  cudaEventRecord(e1);
  cudaEventSynchronize(e1);
  cudaEventElapsedTime(&ms, e0, e1);
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::printf("  FATAL: %s\n", cudaGetErrorString(err));
    std::exit(1);
  }
  std::vector<ROut> out(n);
  cudaMemcpy(out.data(), dout, sizeof(ROut) * n, cudaMemcpyDeviceToHost);
  cudaFree(dr);
  cudaFree(dout);
  cudaFree(de);
  cudaFree(big.arrays);
  cudaFree(big.views);
  cudaFree(big.busy);
  return out;
}

/// The thread path's buffers: the slot's capacities.
struct HostWs {
  std::vector<deex::Fragment> evap{4096}, results{1024}, step{512};
  std::vector<deex::DeexProduct> deex_products{1024}, pproducts{1024};
  std::vector<bic::CascadeProduct> preco{bic::kSlotPrecoCap};
  preco::PrecoWorkspace ws;
  HostWs() {
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
  }
};

std::vector<ROut> run_thread(const std::vector<Res>& res, const data::LevelTable& lt,
                             const deex::FermiPool& fp, int n_threads) {
  std::vector<ROut> out(res.size());
  std::vector<std::thread> th;
  for (int t = 0; t < n_threads; ++t) {
    th.emplace_back([&, t] {
      HostWs* hw = new HostWs();
      for (std::size_t i = t; i < res.size(); i += n_threads) {
        deexcite_one(res[i], lt, fp, hw->ws, hw->preco.data(), bic::kSlotPrecoCap, nullptr, 0,
                     out[i]);
      }
      delete hw;
    });
  }
  for (auto& x : th) { x.join(); }
  return out;
}

// ---- the statistics
int g_fails = 0;
void check(bool ok, const char* what) {
  if (!ok) {
    std::printf("  FAIL: %s\n", what);
    ++g_fails;
  }
}

enum Sp { kN, kP, kD, kT, kHe3, kAlpha, kGam, kEl, kFrag, kNSp };
const char* sp_name[kNSp] = {"n", "p", "d", "t", "He3", "alpha", "gamma", "e-", "A>4"};
int species_of(int pdg, int a, int z) {
  if (a == 0) { return (pdg == 22) ? kGam : kEl; }
  if (a == 1) { return (z == 0) ? kN : kP; }
  if (a == 2) { return kD; }
  if (a == 3) { return (z == 1) ? kT : kHe3; }
  if (a == 4 && z == 2) { return kAlpha; }
  return kFrag;
}
constexpr int kKeSp = 5;   ///< n, p, alpha, gamma, A > 4
int ke_slot(int s) {
  switch (s) {
    case kN: return 0;
    case kP: return 1;
    case kAlpha: return 2;
    case kGam: return 3;
    case kFrag: return 4;
    default: return -1;
  }
}
const char* ke_name[kKeSp] = {"n", "p", "alpha", "gamma", "A>4"};

struct Stats {
  static constexpr int kKe = 12, kSize = 12;
  double n = 0;
  double mult[kNSp] = {}, mult2[kNSp] = {};
  double ke[kKeSp][kKe] = {}, ke2[kKeSp][kKe] = {};
  double size[kSize] = {};
  double tke = 0, tke2 = 0;
  void add(const ROut& o) {
    if (o.handed) { return; }
    n += 1;
    double m[kNSp] = {};
    double k[kKeSp][kKe] = {};
    double t = 0;
    for (int i = 0; i < o.n; ++i) {
      const int s = species_of(o.pdg[i], o.a[i], o.z[i]);
      m[s] += 1;
      t += o.ke[i];
      const int ks = ke_slot(s);
      if (ks >= 0) {
        const double e = (o.ke[i] > 0.001f) ? o.ke[i] : 0.001;
        int b = static_cast<int>(std::floor(std::log10(e / 0.001) * 2.0));   // 1 keV .. 1 GeV
        b = (b < 0) ? 0 : ((b >= kKe) ? kKe - 1 : b);
        k[ks][b] += 1;
      }
    }
    for (int s = 0; s < kNSp; ++s) { mult[s] += m[s]; mult2[s] += m[s] * m[s]; }
    for (int s = 0; s < kKeSp; ++s) {
      for (int b = 0; b < kKe; ++b) { ke[s][b] += k[s][b]; ke2[s][b] += k[s][b] * k[s][b]; }
    }
    int sb = o.n;
    sb = (sb >= kSize) ? kSize - 1 : sb;
    size[sb] += 1;
    tke += t;
    tke2 += t * t;
  }
};

struct Worst {
  double z = 0;
  std::string row;
  int rows = 0;
  std::vector<std::pair<double, std::string>> all;
  void see(double z_, const std::string& r) {
    ++rows;
    all.emplace_back(z_, r);
    if (std::fabs(z_) > std::fabs(z)) {
      z = z_;
      row = r;
    }
  }
  void print(const char* what) const {
    std::vector<std::pair<double, std::string>> s = all;
    std::sort(s.begin(), s.end(),
              [](const auto& x, const auto& y) { return std::fabs(x.first) > std::fabs(y.first); });
    for (std::size_t i = 0; i < s.size() && i < 5; ++i) {
      std::printf("    %s worst %zu: %+.2f sigma  %s\n", what, i + 1, s[i].first,
                  s[i].second.c_str());
    }
  }
};

double zmean(double sa, double s2a, double na, double sb, double s2b, double nb) {
  if (na < 2 || nb < 2) { return 0; }
  const double ma = sa / na, mb = sb / nb;
  const double va = (s2a / na - ma * ma) / na, vb = (s2b / nb - mb * mb) / nb;
  if (va + vb <= 0) { return (ma == mb) ? 0.0 : 1e9; }
  return (ma - mb) / std::sqrt(va + vb);
}

void compare(const std::string& cls, const Stats& a, const Stats& b, Worst& w) {
  auto row = [&](const std::string& name, double sa, double s2a, double sb, double s2b) {
    if (sa < 25 && sb < 25) { return; }
    w.see(zmean(sa, s2a, a.n, sb, s2b, b.n), cls + " " + name);
  };
  for (int s = 0; s < kNSp; ++s) {
    row(std::string("mult ") + sp_name[s], a.mult[s], a.mult2[s], b.mult[s], b.mult2[s]);
  }
  for (int s = 0; s < kKeSp; ++s) {
    for (int k = 0; k < Stats::kKe; ++k) {
      char nm[64];
      std::snprintf(nm, sizeof nm, "ke %s bin %d", ke_name[s], k);
      row(nm, a.ke[s][k], a.ke2[s][k], b.ke[s][k], b.ke2[s][k]);
    }
  }
  for (int k = 0; k < Stats::kSize; ++k) {
    char nm[64];
    std::snprintf(nm, sizeof nm, "products %d", k);
    row(nm, a.size[k], a.size[k], b.size[k], b.size[k]);
  }
  row("summed kinetic energy", a.tke, a.tke2, b.tke, b.tke2);
}

/// V325's classes: the (Z, A, E*) bins that hold most of the proton beam's Binary residuals and
/// the Fe56 beam's light-ion residuals and spectators, with the bin's tape-mean exciton numbers.
std::vector<Class> classes() {
  return {
#include "test_tail_flat_classes.inc"
  };
}

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  std::printf("== the de-excitation tail on the pool's compact workspace against the thread "
              "path (P41) ==\n");
  int per_class = 20000;
  if (argc > 1 && std::atoi(argv[1]) > 0) { per_class = std::atoi(argv[1]); }
  const std::string pe = host::g4photon_evaporation_dir();
  if (pe.empty()) {
    std::printf("  skipped: no PhotonEvaporation dataset\nOK\n");
    return 0;
  }
  static data::LevelTableStorage lts;
  data::read_all_level_data(
      lts, pe, data::kLevelZMax, [](int z, int a) { return deex::shell_correction(a, z); },
      [](int z, int a) { return deex::level_manager_level_density(z, a); });
  const data::LevelTable host_lt = lts.view();
  static deex::FermiPoolStorage fps;
  deex::build_fermi_pool(fps, host_lt);
  const deex::FermiPool host_fp = fps.view();
  static data::LevelTableStorage dstorage;
  host::LevelTableOwner lown = host::upload_level_data(dstorage, data::kLevelZMax, false);
  had::ftf_slot_request() = 1;
  had::InteractionPoolOwner<double> pown =
      had::build_interaction_pool<double>(1, dstorage.view(), false);
  cudaDeviceSetLimit(cudaLimitStackSize, 28672);
  std::printf("  a pool entry: %zu B (the slot's de-excitation part: 536,576 B)\n",
              sizeof(Entry));

  const std::vector<Class> cl = classes();
  std::vector<Res> res, res_ind;
  for (std::size_t c = 0; c < cl.size(); ++c) {
    Philox<double> pick(static_cast<uint32_t>(c + 1), 99u, 5u);
    for (int k = 0; k < per_class; ++k) {
      Res r;
      r.kind = cl[c].kind;
      r.z = cl[c].z;
      r.a = cl[c].a;
      r.particles = cl[c].particles;
      r.holes = cl[c].holes;
      r.charged = cl[c].charged;
      r.ex = cl[c].ex_lo + (cl[c].ex_hi - cl[c].ex_lo) * pick.uniform();
      r.key = static_cast<uint32_t>(c + 1) * 1000003u + static_cast<uint32_t>(k);
      res.push_back(r);
      r.key ^= 0x5BD1E995u;
      res_ind.push_back(r);
    }
  }
  const int n = static_cast<int>(res.size());
  std::printf("  %zu classes x %d residuals = %d\n", cl.size(), per_class, n);

  int width = 4096;
  if (const char* v = std::getenv("G4GPU_TAIL_TEST_WIDTH")) { width = std::max(32, std::atoi(v)); }
  float ms = 0.f, ms2 = 0.f, ms3 = 0.f;
  const std::vector<ROut> pool = run_pool(res, pown, lown.view, width, 0, ms);
  std::printf("  pool: %d residuals on %d entries in %.1f ms\n", n, width, ms);
  const std::vector<ROut> pool2 = run_pool(res, pown, lown.view, width, 0, ms2);
  const int w3 = 1024 + 32;
  const std::vector<ROut> pool3 = run_pool(res, pown, lown.view, w3, 0, ms3);
  int differ2 = 0, differ3 = 0, handed = 0;
  for (int i = 0; i < n; ++i) {
    differ2 += (pool[i].hash != pool2[i].hash || pool[i].n != pool2[i].n) ? 1 : 0;
    differ3 += (pool[i].hash != pool3[i].hash || pool[i].n != pool3[i].n) ? 1 : 0;
    handed += pool[i].handed;
  }
  // V325's per-class table: the phases, draws, products and the lane's time (1,455 MHz).
  for (std::size_t c = 0; c < cl.size(); ++c) {
    double s[6] = {}; int mx[6] = {};
    for (int k = 0; k < per_class; ++k) {
      const ROut& o = pool[c * per_class + k];
      const int v[6] = {o.transitions, o.emissions, o.evap_steps, o.draws, o.n,
                        static_cast<int>(o.clk / 1455)};
      for (int j = 0; j < 6; ++j) { s[j] += v[j]; mx[j] = std::max(mx[j], v[j]); }
    }
    std::printf("  %-26s transitions %.2f (%d) emissions %.2f (%d) evaporation steps %.2f (%d) "
                "draws %.1f (%d) products %.2f (%d) lane %.3f ms (%.3f)\n",
                cl[c].name, s[0] / per_class, mx[0], s[1] / per_class, mx[1], s[2] / per_class,
                mx[2], s[3] / per_class, mx[3], s[4] / per_class, mx[4],
                s[5] / per_class / 1000.0, mx[5] / 1000.0);
  }
  std::printf("  pool twice: %d of %d residuals differ; at %d entries against %d: %d differ\n",
              differ2, n, w3, width, differ3);
  check(differ2 == 0, "the pool reproduces itself to the bit");
  check(differ3 == 0, "the pool's answer does not depend on its width");
  int continued = 0;
  for (int i = 0; i < n; ++i) { continued += pool[i].continued; }
  std::printf("  handed back: %d of %d; continued on a slot-sized workspace: %d\n", handed, n,
              continued);
  check(handed == 0, "no residual is handed back");
  {
    // THE CONTINUATION's own check: on a 1-2 entry workspace nearly every residual binds it and
    // is continued - which must give exactly the uncontinued residual's products.
    float mst = 0.f;
    const std::vector<ROut> tiny = run_pool(res, pown, lown.view, width, 0, mst, 1);
    int cont = 0, diff = 0, past = 0;
    for (int i = 0; i < n; ++i) {
      cont += tiny[i].continued;
      past += tiny[i].handed;
      diff += (tiny[i].hash != pool[i].hash || tiny[i].n != pool[i].n) ? 1 : 0;
    }
    std::printf("  THE CONTINUATION (a 1-2 entry workspace): %d of %d residuals continued, %d "
                "past the bound, %d differ from the pool's to the bit\n", cont, n, past, diff);
    check(cont > n / 2, "the tiny workspace makes most residuals continue");
    check(diff == past, "a continued residual is the uncontinued one to the bit");
  }

  const int host_threads = 8;
  const std::vector<ROut> paired = run_thread(res, host_lt, host_fp, host_threads);
  int same = 0;
  for (int i = 0; i < n; ++i) { same += (pool[i].hash == paired[i].hash) ? 1 : 0; }
  std::printf("  %d of %d pool residuals are the host thread path's on the same key, to the bit\n",
              same, n);
  const std::vector<ROut> thread = run_thread(res_ind, host_lt, host_fp, host_threads);

  auto against = [&](const std::vector<ROut>& x, const std::vector<ROut>& y, Worst& w) {
    for (std::size_t c = 0; c < cl.size(); ++c) {
      Stats a, b;
      for (int k = 0; k < per_class; ++k) {
        a.add(x[c * per_class + k]);
        b.add(y[c * per_class + k]);
      }
      compare(cl[c].name, a, b, w);
    }
  };
  Worst wt;
  against(pool, thread, wt);
  std::printf("  pool against thread (independent keys): %d rows, the worst %+.2f sigma (%s)\n",
              wt.rows, wt.z, wt.row.c_str());
  wt.print("thread");
  check(std::fabs(wt.z) < 5.0, "no pool-against-thread row past 5 sigma");

  float msp = 0.f;
  const std::vector<ROut> pert = run_pool(res, pown, lown.view, width, 1, msp);
  Worst wp;
  against(pert, thread, wp);
  std::printf("  ANTI-VACUITY, the last product dropped in the hand-back: %d rows, the worst "
              "%+.2f sigma (%s)\n", wp.rows, wp.z, wp.row.c_str());
  check(std::fabs(wp.z) > 5.0, "the dropped product is caught past 5 sigma");

  had::free_interaction_pool(pown);
  if (g_fails == 0) {
    std::printf("OK\n");
    return 0;
  }
  std::printf("FAILED (%d)\n", g_fails);
  return 1;
}
