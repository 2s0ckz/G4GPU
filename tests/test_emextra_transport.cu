// P19: the photon's, electron's, positron's and muon's nuclear interactions in the STEPPERS, on
// the device - the half `tests/test_emextra_wiring.cu` cannot reach, because `step_gamma` and
// `step_lepton` are device-only.
//
// Six sections:
//
//  1. BIT-IDENTITY WHERE THE PROCESS IS UNREACHABLE. One step of 20,000 photons through
//     `step_gamma` with the wiring and without it (the pre-P19 call), every output field
//     compared with tolerance ZERO: below 2 m_e in all four of B1's materials, and in water up
//     to oxygen's 11.499 MeV threshold - the 6 MeV gamma gate's photons among them. The same
//     for electrons and positrons in water below oxygen's electro-nuclear 7.296 MeV. And the
//     converse, so that the comparison is shown able to fail: at 6 MeV in air, where argon is
//     open from 0.5 MeV, and for a 50 MeV electron in water, the steps differ.
//  2. THE PHOTON'S SLICE, AT ITS RATE, IN EACH ZONE. First (a): the four processes' cross
//     sections evaluated on the device equal the host's - the evaluation the host test holds
//     against Geant4 - to 1e-12, at 1,760 points. Then a million first interactions per cell in
//     a medium nothing leaves: in zone 2 the fraction queued as photo-nuclear against
//     sigN/(total + sigN), binomially; at 100, 150 and 500 MeV with the general process ON,
//     NONE queued, the zone-3 hand-offs counted at that fraction instead and the conversions at
//     (pair + sigN)/(total + sigN) (docs/RISK.md V207); with it OFF, queued at
//     sigN/(total + sigN) again; below 2 m_e, nothing.
//  3. THE LEPTON'S LENGTH, AT ITS RATE. Electrons and positrons of 1 GeV and 200 MeV followed in
//     bone until they stop or their step goes to the queue, the queued interactions counted
//     against the Poisson mean sum(L_i * sigma(E_pre,i)) over the realised path - P15's
//     section-3 method, with its V52 lesson kept: the cross section itself is tested against
//     Geant4's element oracles in the host test, and this asks only whether the LENGTH is drawn
//     from it. Every entry those tracks queued is checked field by field.
//  4. THE MUON draws on every step (the clamped Kokoulin table is never zero, docs/RISK.md V178)
//     and enqueues - the enqueue checked with a test-scaled table, because at a 13 km mean free
//     path a realistic muon never reaches it.
//  5. THE QUEUE ENTRY carries what `run_emextra_drain` reads: species, bucket, kind, the
//     pre-step energy (where a lepton's target is drawn), the pre-step point and direction, the
//     interaction point, the material, the score slot and the cross section. A sample of the
//     photon and lepton entries is then run through `had::run_emextra` on the HOST, keyed as the
//     drain keys it - the models' own device path is the engine build's and every B1 beam's,
//     because a model in a test kernel costs ptxas 6-20 GB (docs/RISK.md V189).
//  6. A FULL QUEUE is refused by name - `kInelasticQueueFull` and the `kPhotoNuclear` SIZE row -
//     with the conservative disposal, and nothing is lost silently.
#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "core/rng.cuh"
#include "core/track_buffer.cuh"
#include "data/level_data.cuh"
#include "data/materials.cuh"
#include "data/photoelectric_data.cuh"
#include "data/rayleigh_data.cuh"
#include "host/g4data.cuh"
#include "host/hadronic_upload.cuh"
#include "host/pe_upload.cuh"
#include "physics/em/gamma_processes.cuh"
#include "physics/em/hadron_range.cuh"
#include "physics/hadronic/interaction_apply.cuh"
#include "physics/scene.cuh"
#include "physics/stepper.cuh"
#include "render/trajectory.cuh"

using real_t = double;
using namespace g4gpu;
namespace hp = g4gpu::physics::hadronic;
namespace hxs = g4gpu::hadronic::xs;

namespace {

int g_fails = 0;
void fail(const char* fmt, ...) {
  std::printf("  FAIL: ");
  va_list ap;
  va_start(ap, fmt);
  std::vprintf(fmt, ap);
  va_end(ap);
  std::printf("\n");
  ++g_fails;
}

constexpr int kMaxSec = 6;

/// Everything one step leaves behind, compared field by field in section 1.
struct Outcome {
  double x = 0, y = 0, z = 0, dx = 0, dy = 0, dz = 0, ekin = 0;
  int volume = 0;
  double length = 0, edep = 0;
  int process = 0, status = 0, alive = 0, queued = 0, n_sec = 0;
  int sec_type[kMaxSec] = {};
  double sec_ekin[kMaxSec] = {};
  double sec_dir[kMaxSec][3] = {};
};

/// Records what a step emits without transporting it. `last_secondary` because `step_hadron`
/// reads it and `books` because `step_lepton` does; the rest is `BufferEmitter`'s interface.
struct RecEmitter {
  Outcome* out = nullptr;
  Vec3<real_t> pos{};
  int volume = 0;
  int event = 0;
  unsigned int child_count = 0u;
  int last_secondary = -1;
  EmitterBooks books{};

  __host__ __device__ int push(ParticleType t, const Vec3<real_t>& dir, real_t ekin, int,
                               unsigned short = 0) {
    ++child_count;
    if (out != nullptr && out->n_sec < kMaxSec) {
      const int i = out->n_sec++;
      out->sec_type[i] = static_cast<int>(t);
      out->sec_ekin[i] = static_cast<double>(ekin);
      out->sec_dir[i][0] = dir.x;
      out->sec_dir[i][1] = dir.y;
      out->sec_dir[i][2] = dir.z;
    }
    return 0;
  }
  __host__ __device__ int push_nucleus(int z, int a, const Vec3<real_t>& dir, real_t ekin,
                                       int event_id) {
    return push(particle_type_of_nucleus(z, a), dir, ekin, event_id, ion_za_of(z, a));
  }
};

__device__ TrackState<real_t> fresh_track(ParticleType t, real_t e0, unsigned int key) {
  TrackState<real_t> p{};
  p.species = t;
  p.pos = Vec3<real_t>{0, 0, 0};
  p.dir = Vec3<real_t>{0, 0, 1};
  p.ekin = e0;
  p.volume = 0;
  p.event = 0;
  p.rng_key = key;
  p.step = 0u;
  p.begin(p.pos, p.dir, p.ekin, 0, p.rng_key, ProcessId::fNotDefined, real_t(0), real_t(1));
  return p;
}

__device__ void record(const TrackState<real_t>& p, const StepReport<real_t>& rep, real_t edep,
                       bool alive, bool queued, Outcome& o) {
  o.x = p.pos.x;
  o.y = p.pos.y;
  o.z = p.pos.z;
  o.dx = p.dir.x;
  o.dy = p.dir.y;
  o.dz = p.dir.z;
  o.ekin = p.ekin;
  o.volume = p.volume;
  o.length = rep.true_length;
  o.edep = edep;
  o.process = static_cast<int>(rep.process);
  o.status = static_cast<int>(rep.status);
  o.alive = alive ? 1 : 0;
  o.queued = queued ? 1 : 0;
}

/// One step of one photon, with (`use_had`) or without the wiring - the second is the call every
/// caller made before P19, and the one `b1_gpu_sched` still makes. The purpose is
/// `run_step_gamma`'s.
__global__ void k_gamma(Scene<real_t> s, had::HadronicWiring<real_t> had, bool use_had,
                        real_t e0, int n, unsigned int key0, Outcome* out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) { return; }
  TrackState<real_t> p = fresh_track(ParticleType::kGamma, e0, key0 + static_cast<unsigned>(i));
  Outcome o{};
  RecEmitter em{&o, p.pos, p.volume, 0, 0u, -1, {}};
  Philox<real_t> rng(p.rng_key, p.step, 0u);
  real_t edep = 0;
  StepReport<real_t> rep{};
  bool queued = false;
  const bool alive = step_gamma(s, p, rng, em, edep, rep, vis::no_capture(),
                                use_had ? &had : nullptr, &queued);
  record(p, rep, edep, alive, queued, o);
  out[i] = o;
}

/// One step of one e- or e+, the same way, with `run_step_lepton`'s purpose.
template <bool kPositron>
__global__ void k_lepton(Scene<real_t> s, had::HadronicWiring<real_t> had, bool use_had,
                         real_t e0, int n, unsigned int key0, Outcome* out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) { return; }
  TrackState<real_t> p = fresh_track(kPositron ? ParticleType::kPositron : ParticleType::kElectron,
                                     e0, key0 + static_cast<unsigned>(i));
  Outcome o{};
  RecEmitter em{&o, p.pos, p.volume, 0, 0u, -1, {}};
  Philox<real_t> rng(p.rng_key, p.step, 0x5A5Au);
  real_t edep = 0;
  StepReport<real_t> rep{};
  bool queued = false;
  const bool alive = step_lepton(s, p, kPositron, rng, em, edep, rep, vis::no_capture(),
                                 use_had ? &had : nullptr, &queued);
  record(p, rep, edep, alive, queued, o);
  out[i] = o;
}

/// A lepton followed until its step goes to the queue, it stops, or it has taken 20,000 steps:
/// the realised path's Poisson mean, sum(L_i * sigma(E_pre,i)), from the cross section the
/// stepper drew its length with.
struct Follow {
  double mu = 0;   ///< sum of L_i * sigma_i along the steps actually taken
  int queued = 0;  ///< 1 when the track's last step went to the queue
  int steps = 0;
};
template <bool kPositron>
__global__ void k_follow_lepton(Scene<real_t> s, had::HadronicWiring<real_t> had, int mat,
                                real_t e0, int n, unsigned int key0, Follow* out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) { return; }
  TrackState<real_t> p = fresh_track(kPositron ? ParticleType::kPositron : ParticleType::kElectron,
                                     e0, key0 + static_cast<unsigned>(i));
  Follow f{};
  const had::EmExtraProcess proc =
      kPositron ? had::EmExtraProcess::kPositronNuclear : had::EmExtraProcess::kElectronNuclear;
  for (int k = 0; k < 20000; ++k) {
    hxs::MaterialXs<real_t> mx{};
    const real_t sig =
        had::emextra_xs_per_volume<real_t>(had.emextra, proc, mat, s.materials[mat], p.ekin, mx);
    RecEmitter em{nullptr, p.pos, p.volume, 0, 0u, -1, {}};
    Philox<real_t> rng(p.rng_key, p.step, 0x5A5Au);
    real_t edep = 0;
    StepReport<real_t> rep{};
    bool queued = false;
    const bool alive = step_lepton(s, p, kPositron, rng, em, edep, rep, vis::no_capture(), &had,
                                   &queued);
    ++f.steps;
    f.mu += static_cast<double>(rep.true_length) * static_cast<double>(sig);
    if (queued) {
      f.queued = 1;
      break;
    }
    ++p.step;
    if (!alive || p.volume == geom::kOutsideWorld) { break; }
  }
  out[i] = f;
}

/// One per-volume cross section of one process, evaluated on the device.
struct XsProbe {
  int process = 0;  ///< had::EmExtraProcess
  int mat = 0;
  double e = 0;
  double xs = -1;
};
__global__ void k_xs(had::EmExtraTables<real_t> t, const data::Material<real_t>* mats,
                     XsProbe* probes, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) { return; }
  XsProbe& pr = probes[i];
  const auto proc = static_cast<had::EmExtraProcess>(pr.process);
  hxs::MaterialXs<real_t> mx{};
  pr.xs = (proc == had::EmExtraProcess::kPhotonNuclear)
              ? had::photon_nuclear_xs<real_t>(t, had::GammaGeneralProcess::kOn, pr.mat,
                                               mats[pr.mat], pr.e)
              : had::emextra_xs_per_volume<real_t>(t, proc, pr.mat, mats[pr.mat], pr.e, mx);
}

/// One step of one mu- through `step_hadron`, with `run_step_hadron`'s purpose.
__global__ void k_muon(Scene<real_t> s, had::HadronicWiring<real_t> had, real_t e0, int n,
                       unsigned int key0, Outcome* out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) { return; }
  TrackState<real_t> p =
      fresh_track(ParticleType::kMuonMinus, e0, key0 + static_cast<unsigned>(i));
  Outcome o{};
  RecEmitter em{&o, p.pos, p.volume, 0, 0u, -1, {}};
  Philox<real_t> rng(p.rng_key, p.step, 0xB19Du);
  real_t edep = 0;
  StepReport<real_t> rep{};
  bool queued = false;
  const bool alive = step_hadron(s, p, ParticleType::kMuonMinus, had, rng, em, edep, rep,
                                 vis::no_capture(), &queued);
  record(p, rep, edep, alive, queued, o);
  out[i] = o;
}

bool same(const Outcome& a, const Outcome& b) {
  if (a.x != b.x || a.y != b.y || a.z != b.z || a.dx != b.dx || a.dy != b.dy || a.dz != b.dz
      || a.ekin != b.ekin || a.volume != b.volume || a.length != b.length || a.edep != b.edep
      || a.process != b.process || a.status != b.status || a.alive != b.alive
      || a.queued != b.queued || a.n_sec != b.n_sec) {
    return false;
  }
  for (int k = 0; k < a.n_sec && k < kMaxSec; ++k) {
    if (a.sec_type[k] != b.sec_type[k] || a.sec_ekin[k] != b.sec_ekin[k]
        || a.sec_dir[k][0] != b.sec_dir[k][0] || a.sec_dir[k][1] != b.sec_dir[k][1]
        || a.sec_dir[k][2] != b.sec_dir[k][2]) {
      return false;
    }
  }
  return true;
}

const char* mat_name(int m) {
  switch (m) {
    case data::kAir: return "air";
    case data::kWater: return "water";
    case data::kA150Tissue: return "A-150";
    case data::kBoneCompact: return "bone";
    default: return "?";
  }
}

bool close_rel(double a, double b, double rel) {
  return std::fabs(a - b) <= rel * std::fmax(std::fabs(a), std::fabs(b));
}

}  // namespace

int main() {
  std::printf("== P19: photo- and lepto-nuclear interactions in the steppers ==\n");
  int dev = 0;
  if (cudaGetDevice(&dev) != cudaSuccess) {
    std::printf("  no CUDA device\n");
    return 1;
  }
  // The stepping kernels' limit, which `TransportEngine::Upload` sets: their frames cannot be
  // sized by ptxas (docs/RISK.md V196), and the device default of 1,024 bytes faults.
  cudaDeviceSetLimit(cudaLimitStackSize, 16384);

  // ---------------------------------------------------------------- the tables
  static data::Material<real_t> mats[data::kNumMaterials];
  data::build_b1_materials<real_t>(mats);
  const std::vector<int> zs = {1, 6, 7, 8, 9, 12, 15, 16, 18, 20};
  static em::RangeTable<real_t> h_rt;
  const host::BremsUpload<real_t> brem = host::upload_brems_for<real_t>(
      host::default_sb_dir(), mats, data::kNumMaterials, h_rt, zs);
  em::RangeTable<real_t>* d_rt = nullptr;
  cudaMalloc(&d_rt, sizeof(h_rt));
  cudaMemcpy(d_rt, &h_rt, sizeof(h_rt), cudaMemcpyHostToDevice);
  auto* d_pe = host::upload_photoelectric_for<real_t>(host::default_phot_dir(), zs);
  auto* d_ray = host::upload_rayleigh<real_t>(host::default_rayl_dir(), zs.data(),
                                              static_cast<int>(zs.size()));
  auto* d_msc = host::upload_msc<real_t>(mats, data::kNumMaterials);
  auto* d_wv = host::upload_wv_lepton<real_t>(mats, data::kNumMaterials);
  auto* h_hrt = new em::HadronRangeTable<real_t>();
  {
    std::vector<real_t> cuts(data::kNumMaterials);
    for (int i = 0; i < data::kNumMaterials; ++i) { cuts[i] = mats[i].cut_electron; }
    static em::ShellTables<real_t> shell;
    em::build_shell_tables(shell);
    em::build_hadron_range_table<real_t>(mats, cuts.data(), *h_hrt, &shell,
                                         data::kNumMaterials);
  }
  em::HadronRangeTable<real_t>* d_hrt = nullptr;
  cudaMalloc(&d_hrt, sizeof(*h_hrt));
  cudaMemcpy(d_hrt, h_hrt, sizeof(*h_hrt), cudaMemcpyHostToDevice);
  data::Material<real_t>* d_mats = nullptr;
  cudaMalloc(&d_mats, sizeof(mats));
  cudaMemcpy(d_mats, mats, sizeof(mats), cudaMemcpyHostToDevice);

  host::EmExtraTableOwner<real_t> emx =
      host::upload_emextra_tables<real_t>(mats, data::kNumMaterials);
  static host::EmExtraHostTables<real_t> hemx;
  host::build_emextra_host_tables<real_t>(hemx, mats, data::kNumMaterials, false);
  if (!emx.gamma_ok || !hemx.gamma_ok) {
    std::printf("  FAIL: G4PARTICLEXS gamma data could not be resolved\n");
    return 1;
  }
  const had::EmExtraTables<real_t> htab = hemx.view();

  // One box per material, 10 km of half-width: nothing leaves on a first step - a 17 MeV photon
  // in air has a 420 m mean free path - and every lepton followed in section 3 stops inside it.
  const real_t kHalf = 1e7;
  geom::Volume<real_t>* d_vols[data::kNumMaterials] = {};
  for (int m = 0; m < data::kNumMaterials; ++m) {
    const geom::Volume<real_t> v[1] = {{{geom::SolidType::kBox, {kHalf, kHalf, kHalf}},
                                        geom::make_translation<real_t>({0, 0, 0}), 0, m,
                                        /*score_index=*/0}};
    cudaMalloc(&d_vols[m], sizeof(v));
    cudaMemcpy(d_vols[m], v, sizeof(v), cudaMemcpyHostToDevice);
  }
  auto scene_for = [&](int m) {
    Scene<real_t> s{};
    s.geometry = geom::Geometry<real_t>{d_vols[m], 1, 0};
    s.materials = d_mats;
    s.range_table = d_rt;
    s.photoelectric = d_pe;
    s.brems = brem.table;
    s.sb = brem.sb;
    s.rayleigh = d_ray;
    s.msc = d_msc;
    s.wv_lepton = d_wv;
    s.hadron_range = d_hrt;
    s.range_cut = real_t(0.7);
    s.scoring_volume = 0;
    return s;
  };

  // The wiring: the P19 fields as QBBC has them, a queue as deep as any section needs, the
  // refusal ledger and the three counters.
  constexpr int kQueueCap = 1 << 18;
  had::PendingInteraction<real_t>* d_q = nullptr;
  int* d_qc = nullptr;
  cudaMalloc(&d_q, sizeof(had::PendingInteraction<real_t>) * kQueueCap);
  cudaMalloc(&d_qc, sizeof(int));
  constexpr int kNR = static_cast<int>(had::HadronicRefusal::kNumHadronicRefusals);
  int* d_rn = nullptr;
  double* d_re = nullptr;
  unsigned long long* d_stats = nullptr;
  cudaMalloc(&d_rn, sizeof(int) * kNR);
  cudaMalloc(&d_re, sizeof(double) * kNR);
  cudaMalloc(&d_stats, sizeof(unsigned long long) * had::kNumEmxStats);
  had::HadronicWiring<real_t> had{};
  had.stage = had::HadronicStage::kFinal;
  had.emextra = emx.view;
  had.emx_queue.items = d_q;
  had.emx_queue.cursor = d_qc;
  had.emx_queue.capacity = kQueueCap;
  had.books.count = d_rn;
  had.books.energy = d_re;
  had.emx_stats = d_stats;
  auto reset = [&]() {
    cudaMemset(d_qc, 0, sizeof(int));
    cudaMemset(d_rn, 0, sizeof(int) * kNR);
    cudaMemset(d_re, 0, sizeof(double) * kNR);
    cudaMemset(d_stats, 0, sizeof(unsigned long long) * had::kNumEmxStats);
  };
  auto cursor = [&]() {
    int c = 0;
    cudaMemcpy(&c, d_qc, sizeof(int), cudaMemcpyDeviceToHost);
    return c;
  };
  auto entries = [&](int n) {
    std::vector<had::PendingInteraction<real_t>> v(static_cast<std::size_t>(std::max(n, 0)));
    if (n > 0) { cudaMemcpy(v.data(), d_q, sizeof(v[0]) * n, cudaMemcpyDeviceToHost); }
    return v;
  };

  constexpr int kMaxOut = 200000;
  Outcome* d_out = nullptr;
  cudaMalloc(&d_out, sizeof(Outcome) * kMaxOut);
  std::vector<Outcome> a(kMaxOut), b(kMaxOut);
  auto get = [&](std::vector<Outcome>& v, int n) {
    const cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      fail("kernel: %s", cudaGetErrorString(err));
      return false;
    }
    cudaMemcpy(v.data(), d_out, sizeof(Outcome) * n, cudaMemcpyDeviceToHost);
    return true;
  };
  const int kT = 64;
  auto blocks = [&](int n) { return (n + kT - 1) / kT; };

  // ============================================================================================
  // 1. Bit-identity where the process is unreachable, and its converse
  // ============================================================================================
  std::printf("-- 1. one step with the wiring against one without, tolerance zero --\n");
  {
    constexpr int kN = 20000;
    struct Cell { int kind; int m; real_t e; bool must_match; const char* why; };
    // kind: 0 gamma, 1 e-, 2 e+
    const Cell cells[] = {
        // Air at 0.8 MeV is the sharp one: argon's cross section is POSITIVE from 0.5 MeV, so
        // only the general process's zones - no photo-nuclear term below 2 m_e - keep it off.
        {0, data::kAir, 0.8, true, "below 2 m_e, above argon's 0.5: zones 0-1 sum no sigN"},
        {0, data::kWater, 1.0, true, "below 2 m_e"},
        {0, data::kA150Tissue, 0.9, true, "below 2 m_e"},
        {0, data::kBoneCompact, 1.02, true, "below 2 m_e"},
        {0, data::kWater, 1.5, true, "zone 2, below oxygen's 11.499 MeV"},
        {0, data::kWater, 6.0, true, "the gamma gate's energy, below 11.499"},
        {0, data::kWater, 11.49, true, "a hair under 11.499"},
        {1, data::kWater, 5.0, true, "below oxygen's electro-nuclear 7.296 MeV"},
        {2, data::kWater, 7.2, true, "below 7.296"},
        {0, data::kAir, 6.0, false, "argon is open from 0.5 MeV, so the steps MUST differ"},
        {1, data::kWater, 50.0, false, "open, so the extra uniform MUST move the steps"},
    };
    for (const Cell& c : cells) {
      const Scene<real_t> s = scene_for(c.m);
      const unsigned int key0 = 0xC0FFEEu + 977u * static_cast<unsigned>(c.m);
      for (int pass = 0; pass < 2; ++pass) {
        reset();
        const bool use = (pass == 0);
        if (c.kind == 0) {
          k_gamma<<<blocks(kN), kT>>>(s, had, use, c.e, kN, key0, d_out);
        } else if (c.kind == 1) {
          k_lepton<false><<<blocks(kN), kT>>>(s, had, use, c.e, kN, key0, d_out);
        } else {
          k_lepton<true><<<blocks(kN), kT>>>(s, had, use, c.e, kN, key0, d_out);
        }
        if (!get(pass == 0 ? a : b, kN)) { return 1; }
      }
      int differ = 0, queued = 0;
      for (int i = 0; i < kN; ++i) {
        if (!same(a[i], b[i])) { ++differ; }
        queued += a[i].queued;
      }
      std::printf("   %-2s %-6s %6.2f MeV  %5d of %d steps differ, %3d queued  (%s)\n",
                  c.kind == 0 ? "g" : (c.kind == 1 ? "e-" : "e+"), mat_name(c.m), c.e, differ,
                  kN, queued, c.why);
      if (c.must_match && differ != 0) {
        fail("%s %.2f MeV: %d steps differ where the process is unreachable", mat_name(c.m), c.e,
             differ);
      }
      if (!c.must_match && differ < kN / 2) {
        fail("%s %.2f MeV: only %d steps differ, so the identity cells above could have passed "
             "blind", mat_name(c.m), c.e, differ);
      }
      // The air cell is sharp only if argon really is open there: with the general process off
      // - no zones - the same photon has a photo-nuclear term.
      if (c.kind == 0 && c.m == data::kAir && c.must_match) {
        const real_t off = had::photon_nuclear_xs<real_t>(htab, had::GammaGeneralProcess::kOff,
                                                          c.m, mats[c.m], c.e);
        std::printf("      air %.2f MeV with the general process off: sigN = %.4g /mm\n", c.e,
                    off);
        if (!(off > 0)) { fail("air at %.2f MeV is not open, so its cell tests no zone", c.e); }
      }
    }
  }

  // ============================================================================================
  // 2. The photon's slice at its rate, zone by zone, in both configurations
  // ============================================================================================
  std::printf("-- 2. the photo-nuclear slice of the selection uniform --\n");
  {
    // The four EM cross sections exactly as `step_gamma` computes them, on the host.
    static data::PhotoElectricTable<real_t> pe{};
    static std::vector<real_t> pte, ptv;
    static data::RayleighTable<real_t> rt{};
    static std::vector<real_t> rte, rtv;
    const bool pe_ok = data::load_photoelectric<real_t>(host::default_phot_dir(), zs.data(),
                                                        static_cast<int>(zs.size()), pe, pte, ptv);
    const bool ra_ok = data::load_rayleigh<real_t>(host::default_rayl_dir(), zs.data(),
                                                   static_cast<int>(zs.size()), rt, rte, rtv);
    if (!pe_ok || !ra_ok) {
      fail("could not load the photoelectric / Rayleigh tables on the host");
      return 1;
    }
    pe.table_e = pte.data();
    pe.table_v = ptv.data();
    rt.table_e = rte.data();
    rt.table_v = rtv.data();

    // (a) THE DEVICE'S CROSS SECTIONS ARE THE HOST'S. The host evaluation is the one
    // `tests/test_emextra_wiring.cu` compares with Geant4's element oracles; the stepper draws
    // from the device evaluation of the same functions on uploaded copies of the same tables.
    // Four processes, four materials, 97 log-spaced energies from 0.5 MeV to 100 GeV and the
    // edges the functions branch on: 2 m_e, oxygen's photo-nuclear threshold, the zone-3 edge,
    // the G4PARTICLEXS gamma data's 130 MeV top and its 150 MeV hand-over to CHIPS.
    {
      std::vector<XsProbe> probes;
      const had::EmExtraProcess procs[4] = {
          had::EmExtraProcess::kPhotonNuclear, had::EmExtraProcess::kElectronNuclear,
          had::EmExtraProcess::kPositronNuclear, had::EmExtraProcess::kMuonNuclear};
      std::vector<double> es;
      for (int k = 0; k <= 96; ++k) { es.push_back(0.5 * std::pow(10.0, k / 16.0)); }
      for (const double e : {1.0219978, 1.022, 11.499, 11.5, 99.999999, 100.0, 130.0, 149.9999,
                             150.0, 150.0001, 199.0, 200.0, 3000.0}) {
        es.push_back(e);
      }
      for (const auto p : procs) {
        for (int m = 0; m < data::kNumMaterials; ++m) {
          for (const double e : es) { probes.push_back({static_cast<int>(p), m, e, -1.0}); }
        }
      }
      const int np = static_cast<int>(probes.size());
      XsProbe* d_probes = nullptr;
      cudaMalloc(&d_probes, sizeof(XsProbe) * np);
      cudaMemcpy(d_probes, probes.data(), sizeof(XsProbe) * np, cudaMemcpyHostToDevice);
      k_xs<<<blocks(np), kT>>>(had.emextra, d_mats, d_probes, np);
      if (cudaDeviceSynchronize() != cudaSuccess) {
        fail("k_xs failed");
        return 1;
      }
      cudaMemcpy(probes.data(), d_probes, sizeof(XsProbe) * np, cudaMemcpyDeviceToHost);
      cudaFree(d_probes);
      int mismatch = 0, positive = 0;
      double worst = 0;
      for (const XsProbe& pr : probes) {
        const auto proc = static_cast<had::EmExtraProcess>(pr.process);
        hxs::MaterialXs<real_t> mx{};
        const double h =
            (proc == had::EmExtraProcess::kPhotonNuclear)
                ? had::photon_nuclear_xs<real_t>(htab, had::GammaGeneralProcess::kOn, pr.mat,
                                                 mats[pr.mat], pr.e)
                : had::emextra_xs_per_volume<real_t>(htab, proc, pr.mat, mats[pr.mat], pr.e, mx);
        if (h > 0) { ++positive; }
        const bool both_zero = (h == 0 && pr.xs == 0);
        const double rel = both_zero ? 0.0
                                     : std::fabs(h - pr.xs) / std::fmax(std::fabs(h),
                                                                        std::fabs(pr.xs));
        if (rel > worst) { worst = rel; }
        if (!(rel <= 1e-12)) {
          if (mismatch < 5) {
            std::printf("     %s in %s at %.7g MeV: host %.17g device %.17g\n",
                        had::emextra_process_name(proc), mat_name(pr.mat), pr.e, h, pr.xs);
          }
          ++mismatch;
        }
      }
      std::printf("   (a) %d device evaluations against the host's, %d positive: %d differ by "
                  "more than 1e-12, worst %.3g\n", np, positive, mismatch, worst);
      if (mismatch != 0) { fail("the device's cross sections are not the host's"); }
      if (positive < np / 2) { fail("too few positive cross sections for (a) to test anything"); }
    }

    // `edge` marks a cell a hair above its material's threshold, where the photo-nuclear share
    // is far too small for its rate to be counted and the per-photon replay is the whole test:
    // water at 11.55 MeV, half a per cent above oxygen's 11.499, catches a stepper that skips
    // the term a little too eagerly - its lengths then lack a sigN the host replay has.
    struct Cell { int m; real_t e; bool general_on; bool edge; };
    const Cell cells[] = {{data::kWater, 20.0, true, false},        {data::kBoneCompact, 22.0, true, false},
                          {data::kAir, 17.0, true, false},          {data::kA150Tissue, 60.0, true, false},
                          {data::kWater, 100.0, true, false},       {data::kWater, 150.0, true, false},
                          {data::kBoneCompact, 500.0, true, false}, {data::kWater, 100.0, false, false},
                          {data::kWater, 150.0, false, false},      {data::kBoneCompact, 500.0, false, false},
                          {data::kAir, 0.8, true, false},           {data::kWater, 11.55, true, true}};
    // A million photons a cell, in five launches of 200,000 with their own keys.
    constexpr int kN2 = 200000;
    constexpr int kBatches2 = 5;
    for (const Cell& c : cells) {
      const Scene<real_t> s = scene_for(c.m);
      had::HadronicWiring<real_t> h2 = had;
      h2.gamma_general =
          c.general_on ? had::GammaGeneralProcess::kOn : had::GammaGeneralProcess::kOff;
      reset();
      const auto xs = em::gamma_macroscopic_xs<real_t>(mats[c.m], c.e, &pe, &rt);
      const real_t sn =
          had::photon_nuclear_xs<real_t>(htab, h2.gamma_general, c.m, mats[c.m], c.e);
      const real_t xs_total = (sn > real_t(0)) ? xs.total + sn : xs.total;
      const bool zone3_on = c.general_on && c.e >= 100.0;
      // THE SELECTION REPLAYED, PHOTON BY PHOTON. Each photon's two uniforms are drawn again on
      // the host from its own key, and its step must be the one `G4GammaGeneralProcess` makes
      // of them: the length `-log(u1)` over the SUMMED total (sigN in it), and the photo-nuclear
      // branch exactly when `u2 * total` lands in the top slice - queued in zone 2 and with the
      // general process off, a conversion in zone 3. A binomial count cannot tell the top slice
      // from the bottom one, or the second uniform from a third; this can, with no statistics.
      int q = 0, conv = 0, boundary = 0;
      long long in_slice = 0, wrong_branch = 0, wrong_length = 0;
      for (int batch = 0; batch < kBatches2; ++batch) {
        const unsigned int key0 = 0xABCD000u + 0x100000u * static_cast<unsigned>(c.m)
                                  + static_cast<unsigned>(batch * kN2);
        k_gamma<<<blocks(kN2), kT>>>(s, h2, true, c.e, kN2, key0, d_out);
        if (!get(a, kN2)) { return 1; }
        for (int i = 0; i < kN2; ++i) {
          q += a[i].queued;
          const bool is_conv = a[i].process == static_cast<int>(ProcessId::fGammaConversion);
          if (is_conv) { ++conv; }
          if (a[i].status == static_cast<int>(StepStatus::fGeomBoundary)) { ++boundary; }
          Philox<real_t> rng(key0 + static_cast<unsigned>(i), 0u, 0u);
          const real_t u1 = rng.uniform();
          const real_t u2 = rng.uniform();
          const bool top = (sn > real_t(0)) && (u2 * xs_total >= xs.total);
          if (top) { ++in_slice; }
          const bool branch_ok = zone3_on ? (!a[i].queued && (!top || is_conv))
                                          : ((a[i].queued != 0) == top);
          if (!branch_ok) { ++wrong_branch; }
          if (!close_rel(a[i].length, double(-log(u1) / xs_total), 1e-13)) { ++wrong_length; }
        }
      }
      unsigned long long st[had::kNumEmxStats] = {};
      cudaMemcpy(st, d_stats, sizeof(st), cudaMemcpyDeviceToHost);
      const double tot = double(xs_total);
      const double pn = double(sn) / tot;
      std::printf("   %-6s %7.2f MeV general %-3s replayed: %lld in the top slice, %lld steps on "
                  "the wrong branch, %lld with a length not drawn from the summed total\n",
                  mat_name(c.m), c.e, c.general_on ? "on" : "off", in_slice, wrong_branch,
                  wrong_length);
      if (wrong_branch != 0 || wrong_length != 0) {
        fail("%s %.2f MeV: the device's selection is not the top slice of the second uniform "
             "over the summed total", mat_name(c.m), c.e);
      }
      if (zone3_on && static_cast<long long>(st[had::kEmxZone3Conversion]) != in_slice) {
        fail("%s %.1f MeV: %llu zone-3 hand-offs counted for %lld photons in the slice",
             mat_name(c.m), c.e, st[had::kEmxZone3Conversion], in_slice);
      }
      // Conditioned on an interaction inside the box, which is every photon here: the selection
      // uniform is independent of the length uniform, so the condition does not bias it.
      const double n = double(kBatches2 * kN2 - boundary);
      auto zscore = [&](double k, double p) {
        const double se = std::sqrt(std::fmax(n * p * (1 - p), 1.0));
        return (k - n * p) / se;
      };
      const double z_q = zscore(q, zone3_on ? 0.0 : pn);
      const double z_h = zscore(double(st[had::kEmxZone3Conversion]), zone3_on ? pn : 0.0);
      const double p_conv = (double(xs.pair) + (zone3_on ? double(sn) : 0.0)) / tot;
      const double z_c = zscore(conv, p_conv);
      std::printf("   %-6s %7.2f MeV general %-3s sigN/tot %.4e | queued %5d (z %+5.2f) | "
                  "zone-3 hand-offs %5llu (z %+5.2f) | conversions %6d (z %+5.2f)\n",
                  mat_name(c.m), c.e, c.general_on ? "on" : "off", pn, q, z_q,
                  st[had::kEmxZone3Conversion], z_h, conv, z_c);
      if (boundary > kBatches2 * kN2 / 100) {
        fail("%d first steps reached the boundary; the box is meant to hold them", boundary);
      }
      if (std::fabs(z_q) > 5 || std::fabs(z_h) > 5 || std::fabs(z_c) > 5) {
        fail("the photo-nuclear slice is not at its rate at %.2f MeV in %s (general %s)", c.e,
             mat_name(c.m), c.general_on ? "on" : "off");
      }
      if (zone3_on && q != 0) { fail("zone 3 queued %d photo-nuclear interactions", q); }
      if (c.e < 1.022 && (q != 0 || sn != 0)) {
        fail("a photon below 2 m_e has a photo-nuclear term");
      }
      if (c.e > 1.022 && !c.edge && !(pn > 1e-4)) {
        fail("%s %.1f MeV: sigN/tot %.3g is too small for this cell to test anything",
             mat_name(c.m), c.e, pn);
      }
      if (c.edge && !(sn > 0)) {
        fail("%s %.2f MeV: the edge cell is not above the threshold, so it tests no skip",
             mat_name(c.m), c.e);
      }
    }
  }

  // ============================================================================================
  // 3. The lepton's length, at its rate
  // ============================================================================================
  std::printf("-- 3. electro-nuclear interactions along the realised path, in bone --\n");
  std::vector<had::PendingInteraction<real_t>> lepton_q;
  {
    constexpr int kNF = 100000;
    Follow* d_f = nullptr;
    cudaMalloc(&d_f, sizeof(Follow) * kNF);
    std::vector<Follow> f(kNF);
    const int m = data::kBoneCompact;
    const Scene<real_t> s = scene_for(m);
    for (int pos = 0; pos < 2; ++pos) {
      for (const real_t e0 : {real_t(1000), real_t(200)}) {
        reset();
        if (pos) {
          k_follow_lepton<true><<<blocks(kNF), kT>>>(s, had, m, e0, kNF, 0x3000u, d_f);
        } else {
          k_follow_lepton<false><<<blocks(kNF), kT>>>(s, had, m, e0, kNF, 0x3000u, d_f);
        }
        const cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
          fail("kernel: %s", cudaGetErrorString(err));
          return 1;
        }
        cudaMemcpy(f.data(), d_f, sizeof(Follow) * kNF, cudaMemcpyDeviceToHost);
        double mu = 0;
        long long k = 0, steps = 0;
        int capped = 0;
        for (int i = 0; i < kNF; ++i) {
          mu += f[i].mu;
          k += f[i].queued;
          steps += f[i].steps;
          if (f[i].steps >= 20000) { ++capped; }
        }
        const double z = (double(k) - mu) / std::sqrt(std::fmax(mu, 1.0));
        std::printf("   %s %5.0f MeV: %lld queued against a Poisson mean of %.1f over %lld steps"
                    ", z %+.2f\n", pos ? "e+" : "e-", e0, k, mu, steps, z);
        if (std::fabs(z) > 5) { fail("the lepto-nuclear rate is off its cross section"); }
        if (mu < 20) { fail("too few expected interactions (%.1f) to test the rate", mu); }
        if (capped != 0) { fail("%d tracks hit the 20,000-step guard", capped); }
        const int nq = cursor();
        if (nq != k) { fail("the queue holds %d entries for %lld queued steps", nq, k); }
        // Every entry, field by field: what `run_emextra_drain` reads off it.
        const ParticleType want = pos ? ParticleType::kPositron : ParticleType::kElectron;
        const had::EmExtraProcess proc = had::emextra_process_of(want);
        int bad = 0, lost = 0;
        for (const auto& e : entries(std::min(nq, kQueueCap))) {
          hxs::MaterialXs<real_t> mx{};
          const double want_xs =
              had::emextra_xs_per_volume<real_t>(htab, proc, m, mats[m], e.ekin_pre, mx);
          // The track is the POST-step lepton (see the muon's note in section 4).
          if (e.track.ekin < e.ekin_pre) { ++lost; }
          const double dir_norm = std::sqrt(e.dir_pre.x * e.dir_pre.x + e.dir_pre.y * e.dir_pre.y
                                            + e.dir_pre.z * e.dir_pre.z);
          const bool ok = e.species == want && e.track.species == want
                          && e.kind == had::InteractionKind::kInelastic
                          && e.bucket == had::InteractionBucket::kLeptoNuclear
                          && e.model == had::InelasticModel::kNone && e.material == m
                          && e.score_slot == 0 && e.volume_pre == 0
                          && e.status == StepStatus::fPostStepDoItProc && e.true_length > 0
                          && e.ekin_pre <= e0 && e.track.ekin > 0 && e.track.ekin <= e.ekin_pre
                          && e.xs_at_step_start > 0
                          && close_rel(double(e.xs_at_step_start), want_xs, 1e-12)
                          && std::fabs(dir_norm - 1.0) < 1e-9;
          if (!ok) { ++bad; }
          lepton_q.push_back(e);
        }
        if (bad != 0) { fail("%d lepto-nuclear entries do not carry what the drain reads", bad); }
        if (lost < nq * 9 / 10) { fail("the lepton entries are not the post-step state"); }
      }
    }
    cudaFree(d_f);
  }

  // ============================================================================================
  // 4. The muon: a draw every step, and the enqueue
  // ============================================================================================
  std::printf("-- 4. muonNuclear in step_hadron --\n");
  {
    constexpr int kN = 20000;
    const Scene<real_t> s = scene_for(data::kWater);
    // With muonNuclear off the muon's step is the pre-P19 step; with it on every step draws one
    // uniform more, so the two MUST differ - the clamp makes the cross section positive at
    // 300 MeV, a third of the table's lowest node.
    had::HadronicWiring<real_t> off = had;
    off.muon_nuclear = false;
    reset();
    k_muon<<<blocks(kN), kT>>>(s, off, 300, kN, 0x5000u, d_out);
    if (!get(a, kN)) { return 1; }
    reset();
    k_muon<<<blocks(kN), kT>>>(s, had, 300, kN, 0x5000u, d_out);
    if (!get(b, kN)) { return 1; }
    int differ = 0, q_real = 0;
    for (int i = 0; i < kN; ++i) {
      differ += same(a[i], b[i]) ? 0 : 1;
      q_real += b[i].queued;
    }
    std::printf("   300 MeV mu- in water, muonNuclear off against on: %d of %d steps differ, "
                "%d queued\n", differ, kN, q_real);
    if (differ < kN / 2) {
      fail("muonNuclear on moved only %d muon steps - it drew nothing", differ);
    }
    // The enqueue, with the Kokoulin table scaled by 1e8 so that a step reaches it - a 0.13 mm
    // mean free path in place of 13 km. A MECHANISM check and nothing else: the table is the
    // test's own, uploaded beside the real one.
    auto* scaled = new hxs::kokoulin::KokoulinTable<real_t>(*hemx.kokoulin);
    for (int z = 0; z < hxs::kokoulin::kMaxZ; ++z) {
      for (int k = 0; k < hxs::kokoulin::kNodes; ++k) { scaled->value[z][k] *= 1e8; }
    }
    hxs::kokoulin::KokoulinTable<real_t>* d_scaled = nullptr;
    cudaMalloc(&d_scaled, sizeof(*scaled));
    cudaMemcpy(d_scaled, scaled, sizeof(*scaled), cudaMemcpyHostToDevice);
    had::HadronicWiring<real_t> boosted = had;
    boosted.emextra.kokoulin = d_scaled;
    reset();
    k_muon<<<blocks(kN), kT>>>(s, boosted, 2000, kN, 0x6000u, d_out);
    if (!get(a, kN)) { return 1; }
    int q = 0, named = 0;
    for (int i = 0; i < kN; ++i) {
      q += a[i].queued;
      if (a[i].queued && a[i].process == static_cast<int>(ProcessId::fMuonNuclear)) { ++named; }
    }
    int bad = 0, lost = 0;
    const int nq = cursor();
    for (const auto& e : entries(std::min(nq, kQueueCap))) {
      if (e.species != ParticleType::kMuonMinus
          || e.bucket != had::InteractionBucket::kLeptoNuclear
          || e.kind != had::InteractionKind::kInelastic || e.ekin_pre != real_t(2000)
          || !(e.xs_at_step_start > 0) || e.material != data::kWater
          || !(e.track.ekin <= e.ekin_pre) || e.status != StepStatus::fPostStepDoItProc) {
        ++bad;
      }
      // The entry is the POST-step muon, its continuous loss applied. Not every one has lost
      // energy - at a 0.13 mm mean free path the Urban fluctuation model gives a step with no
      // collision about one time in 180 (111 of 19,967 on the first run) - but a queue that
      // held the PRE-step state would show none that had.
      if (e.track.ekin < e.ekin_pre) { ++lost; }
    }
    std::printf("   2 GeV mu- in water with the table x1e8: %d of %d steps queued (%d named "
                "fMuonNuclear), %d of %d entries malformed, %d past their continuous loss\n", q,
                kN, named, bad, nq, lost);
    if (q < kN / 2) { fail("a muon step reached the lepto-nuclear queue only %d times", q); }
    if (named != q || nq != q) {
      fail("queued muon steps are not all fMuonNuclear queue entries");
    }
    if (bad != 0) { fail("muon queue entries do not carry what the drain reads"); }
    if (lost < nq * 9 / 10) { fail("the muon entries are not the post-step state"); }
    cudaFree(d_scaled);
    delete scaled;
  }

  // ============================================================================================
  // 5. The photon's queue entry, and the host run of a sample of photon and lepton entries
  // ============================================================================================
  std::printf("-- 5. queue entries through run_emextra, keyed as the drain keys them --\n");
  {
    constexpr int kN5 = 200000;
    const int m = data::kBoneCompact;
    const real_t e0 = 20.0;
    const Scene<real_t> s = scene_for(m);
    reset();
    k_gamma<<<blocks(kN5), kT>>>(s, had, true, e0, kN5, 0x7000u, d_out);
    if (!get(a, kN5)) { return 1; }
    const int nq = cursor();
    const std::vector<had::PendingInteraction<real_t>> photon_q =
        entries(std::min(nq, kQueueCap));
    const double want_xs =
        had::photon_nuclear_xs<real_t>(htab, had::GammaGeneralProcess::kOn, m, mats[m], e0);
    int bad = 0;
    for (const auto& e : photon_q) {
      // A photon has no along-step physics: the entry is the pre-step photon moved to the
      // interaction point, (0,0,0) + L * (0,0,1).
      const bool ok = e.species == ParticleType::kGamma && e.track.species == ParticleType::kGamma
                      && e.kind == had::InteractionKind::kInelastic
                      && e.bucket == had::InteractionBucket::kPhotoNuclear
                      && e.model == had::InelasticModel::kNone && e.ekin_pre == e0
                      && e.track.ekin == e0 && e.pos_pre.x == 0 && e.pos_pre.y == 0
                      && e.pos_pre.z == 0 && e.dir_pre.z == 1 && e.track.pos.z == e.true_length
                      && e.true_length > 0 && e.material == m && e.score_slot == 0
                      && e.volume_pre == 0 && e.edep == 0 && e.child_count == 0
                      && e.status == StepStatus::fPostStepDoItProc
                      && close_rel(double(e.xs_at_step_start), want_xs, 1e-12);
      if (!ok) { ++bad; }
    }
    std::printf("   g 20 MeV bone: %d queued, %d malformed\n", nq, bad);
    if (nq == 0) { fail("no photo-nuclear entry to check"); }
    if (bad != 0) { fail("%d photo-nuclear entries do not carry what the drain reads", bad); }

    const std::string pe_dir = host::g4photon_evaporation_dir();
    if (pe_dir.empty()) {
      fail("PhotonEvaporation not found");
    } else {
      static data::LevelTableStorage lts;
      data::read_all_level_data(
          lts, pe_dir, data::kLevelZMax, [](int Z, int A) { return deex::shell_correction(A, Z); },
          [](int Z, int A) { return deex::level_manager_level_density(Z, A); });
      const data::LevelTable lt = lts.view();
      static deex::FermiPoolStorage fps;
      deex::build_fermi_pool(fps, lt);
      const deex::FermiPool fpool = fps.view();
      auto* slot = new had::InteractionSlot<real_t>();
      struct Tally { int tried = 0, ran = 0, tripwire = 0, refused = 0, unbalanced = 0; };
      auto run_one = [&](const had::PendingInteraction<real_t>& e, Tally& t) {
        ++t.tried;
        hp::HadProjectile<real_t> proj;
        proj.pdg = pdg_code(e.species);
        proj.charge = particle_def<real_t>(e.species).charge;
        proj.mass = particle_def<real_t>(e.species).mass;
        proj.kin_energy = e.track.ekin;
        proj.baryon_number = 0;
        // `run_emextra_drain`'s key: the queued track's own, and the interaction purpose.
        Philox<real_t> rng(e.track.rng_key, e.track.step, had::kInteractionRngPurpose);
        had::EmExtraDiag dg;
        had::InteractionOutcome o;
        if (e.bucket == had::InteractionBucket::kPhotoNuclear) {
          o = had::run_emextra<real_t, had::InteractionBucket::kPhotoNuclear>(
              proj, e.species, mats[e.material], e.material, htab, e.ekin_pre, *slot, lt, fpool,
              rng, &dg);
        } else {
          o = had::run_emextra<real_t, had::InteractionBucket::kLeptoNuclear>(
              proj, e.species, mats[e.material], e.material, htab, e.ekin_pre, *slot, lt, fpool,
              rng, &dg);
        }
        if (!o.ran) {
          // The tripwire is the store finding no cross section where the stepper drew one: the
          // refusal `run_emextra` books before it calls any model.
          if (o.refusal == had::HadronicRefusal::kEmExtraRefused && dg.emextra_refusal == 0) {
            ++t.tripwire;
          } else {
            ++t.refused;
          }
          return;
        }
        ++t.ran;
        had::fill_result_into<real_t, had::kInteractionSecondaryCap,
                              had::kInteractionSecondaryCap>(
            slot->fs, e.track.dir, real_t(0), real_t(1), had::emextra_has_at_rest(e.species),
            slot->pdg_mass, slot->filled);
        const auto& fr = slot->filled;
        int bsum = 0, qsum = 0;
        for (int i = 0; i < fr.n_secondaries; ++i) {
          const auto& sj = fr.secondaries[i];
          if (sj.a > 0) {
            bsum += sj.a;
            qsum += sj.z;
          } else if (sj.pdg != 11) {
            const ParticleType st = particle_type_of_pdg(sj.pdg);
            if (st != ParticleType::kNumTypes) {
              bsum += had::baryon_number_of(st, 0);
              qsum += static_cast<int>(std::lrint(particle_def<real_t>(st).charge));
            }
          }
        }
        const bool survives = fr.status != hp::TrackStatusChange::kStopAndKill;
        if (survives) { qsum += static_cast<int>(std::lrint(double(proj.charge))); }
        const bool untouched = survives && fr.n_secondaries == 0;
        const int want_q = static_cast<int>(std::lrint(double(proj.charge))) + o.target_z;
        if (!untouched && (bsum != o.target_a || qsum != want_q)) { ++t.unbalanced; }
      };
      Tally tp, tl;
      for (std::size_t k = 0; k < photon_q.size() && k < 60; ++k) { run_one(photon_q[k], tp); }
      for (std::size_t k = 0; k < lepton_q.size() && k < 60; ++k) { run_one(lepton_q[k], tl); }
      std::printf("   photon entries: %d run, %d ran, %d refused by name, %d tripwires, %d "
                  "unbalanced\n", tp.tried, tp.ran, tp.refused, tp.tripwire, tp.unbalanced);
      std::printf("   lepton entries: %d run, %d ran, %d refused by name, %d tripwires, %d "
                  "unbalanced\n", tl.tried, tl.ran, tl.refused, tl.tripwire, tl.unbalanced);
      if (tp.tried == 0 || tl.tried == 0) { fail("no entries to run"); }
      if (tp.tripwire + tl.tripwire != 0) {
        fail("an entry's pre-step energy and material give the drain no cross section");
      }
      if (tp.ran != tp.tried || tl.ran != tl.tried) {
        fail("an entry below the FTF and QGS arms did not run");
      }
      if (tp.unbalanced + tl.unbalanced != 0) { fail("baryon number or charge not conserved"); }
      delete slot;
    }
  }

  // ============================================================================================
  // 6. A full queue: refused by name, disposed of conservatively
  // ============================================================================================
  std::printf("-- 6. a queue of four entries --\n");
  {
    constexpr int kN6 = 200000;
    const real_t e0 = 22.0;
    const Scene<real_t> s = scene_for(data::kWater);
    had::HadronicWiring<real_t> small = had;
    small.emx_queue.capacity = 4;
    reset();
    k_gamma<<<blocks(kN6), kT>>>(s, small, true, e0, kN6, 0x8000u, d_out);
    if (!get(a, kN6)) { return 1; }
    std::vector<int> rn(kNR);
    std::vector<double> re(kNR);
    cudaMemcpy(rn.data(), d_rn, sizeof(int) * kNR, cudaMemcpyDeviceToHost);
    cudaMemcpy(re.data(), d_re, sizeof(double) * kNR, cudaMemcpyDeviceToHost);
    int q = 0, disposed = 0, pn_steps = 0;
    for (int i = 0; i < kN6; ++i) {
      q += a[i].queued;
      if (a[i].process == static_cast<int>(ProcessId::fPhotoNuclear)) {
        ++pn_steps;
        if (!a[i].queued && a[i].status == static_cast<int>(StepStatus::fStopAndKill)
            && a[i].edep == e0 && a[i].ekin == 0.0 && a[i].alive == 0) {
          ++disposed;
        }
      }
    }
    const int full = rn[static_cast<int>(had::HadronicRefusal::kInelasticQueueFull)];
    const int size = rn[static_cast<int>(had::HadronicRefusal::kPhotoNuclear)];
    std::printf("   22 MeV photons in water: %d photo-nuclear steps, %d queued, %d refused as "
                "kInelasticQueueFull, %d in the kPhotoNuclear SIZE row, %d killed with their "
                "energy deposited\n", pn_steps, q, full, size, disposed);
    if (q != 4) { fail("a four-entry queue took %d entries", q); }
    if (full == 0 || full != size || full != disposed || full + q != pn_steps) {
      fail("the full queue's refusals are not booked in both groups and disposed of");
    }
    if (std::fabs(re[static_cast<int>(had::HadronicRefusal::kPhotoNuclear)] - e0 * size) > 1e-6
        || std::fabs(re[static_cast<int>(had::HadronicRefusal::kInelasticQueueFull)]
                     - e0 * full) > 1e-6) {
      fail("the refusal rows' energy is not the refused photons' energy");
    }
  }

  if (g_fails == 0) {
    std::printf("test_emextra_transport: ALL OK\n");
    return 0;
  }
  std::printf("test_emextra_transport: %d FAILURE(S)\n", g_fails);
  return 1;
}
