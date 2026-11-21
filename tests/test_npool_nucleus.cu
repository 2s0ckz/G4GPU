// P38 deliverable 2 (docs/RISK.md V319): the nucleus factory against the thread path's
// `bic::nucleus_init`, nuclide by nuclide, on the card.
//
// The factory (`npool/nucleus_factory.cuh`) samples G4Fancy3DNucleus with a lane per nucleon and
// its own keyed draws, so it is held to the DISTRIBUTION of `nucleus_init` and not to its stream
// (the user's rule inside an interaction, V302). For every natural isotope of the elements of
// exampleB1's four materials - G4_WATER (H, O), G4_A-150_TISSUE (H, C, N, O, F, Ca),
// G4_BONE_COMPACT_ICRU (H, C, N, O, Mg, P, S, Ca) and G4_AIR (C, N, O, Ar), 27 nuclides - it
// samples N nuclei each way (20,000 by default) and compares, with the NUCLEUS as the unit of
// every variance (the nucleons of one nucleus are correlated by the hard core and by ReduceSum,
// so a per-nucleon binomial sigma would be the wrong one):
//
//   radial      every nucleon's |r|, and the protons' alone (the barrier test is theirs)
//   momentum    every nucleon's |p|
//   nn          every nucleon's nearest-neighbour distance - the hard core's own signature
//   outer       the outer radius (`GetOuterRadius`), which sets the impact parameters
//   slot kind   P(slot k is a proton) for each k - ChooseNucleons, after ReduceSum's swaps
//   balance     |sum p| per nucleus in decades, the ReduceSum failure and zeroed-proton rates
//   moments     per-nucleus mean |r|, mean |p| and mean nn
//
// Each bin is a two-sample z of the per-nucleus means; NO bin may pass 5 sigma. The
// anti-vacuity run builds O16, C12 and Ca40 with the factory's hard core HALVED
// (`FactoryJob::distance_scale` = 0.5) and must be caught past 5 sigma on each. The two paths'
// throughputs (nuclei/s, kernel time by events) are printed for V319.
//
// GPU test: CUDA_VISIBLE_DEVICES=1 (the Titan V) and out\gpu.lock are the caller's.
//   test_npool_nucleus [N]
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "core/rng.cuh"
#include "physics/hadronic/npool/nucleus_factory.cuh"

using namespace g4gpu;

/// The thread path, one thread a nucleus: `bic::nucleus_init` exactly as the cascade calls it,
/// on a scratch of its own.
__global__ void thread_nucleus_kernel(int a, int z, int n, unsigned seed, bic::Nucleon* nucleons,
                                      bic::NucleusReport* reports, deex::Vec3d* momentum,
                                      double* fermi_p, bic::NucleusSortEntry* sums,
                                      double* flat) {
  const int k = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (k >= n) { return; }
  bic::Nucleus3D nuc;
  nuc.nucleons = nucleons + static_cast<size_t>(k) * a;
  nuc.capacity = a;
  bic::Nucleus3DScratch sc;
  sc.momentum = momentum + static_cast<size_t>(k) * a;
  sc.fermi_p = fermi_p + static_cast<size_t>(k) * a;
  sc.test_sums = sums + static_cast<size_t>(k) * a;
  sc.flat_block = flat + static_cast<size_t>(k) * bic::kFlatBlock;
  sc.capacity = a;
  Philox<double> rng(seed, static_cast<unsigned>(k), 0x7e57u);
  reports[k] = bic::nucleus_init(nuc, sc, a, z, rng);
}

namespace {

int fails = 0;

#define CK(x)                                                                         \
  do {                                                                                \
    cudaError_t e_ = (x);                                                             \
    if (e_ != cudaSuccess) {                                                          \
      std::printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
      std::exit(2);                                                                   \
    }                                                                                 \
  } while (0)

struct Sample {
  std::vector<bic::Nucleon> nuc;
  std::vector<bic::NucleusReport> rep;
  std::vector<int> rounds;
  float ms = 0.0f;
};

/// Per-nucleus statistics: for every bin the sum and sum of squares of the per-nucleus count.
struct Hist {
  double lo = 0.0, hi = 1.0;
  int nb = 40;
  std::vector<double> s, s2;
  std::vector<double> cur;
  Hist() = default;
  Hist(double l, double h, int n) : lo(l), hi(h), nb(n), s(n + 2), s2(n + 2), cur(n + 2) {}
  void fill(double x) {
    int b = (x < lo) ? 0 : (x >= hi) ? nb + 1 : 1 + static_cast<int>((x - lo) / (hi - lo) * nb);
    if (b > nb + 1) { b = nb + 1; }
    cur[b] += 1.0;
  }
  void end_nucleus() {
    for (int b = 0; b < nb + 2; ++b) {
      s[b] += cur[b];
      s2[b] += cur[b] * cur[b];
      cur[b] = 0.0;
    }
  }
};

struct Stats {
  Hist radial, radial_p, mom, nn, outer, balance;
  std::vector<double> slot_p;  // count of protons per slot
  Hist m_r, m_p, m_nn;         // per-nucleus moments as one-bin "histograms" of the value
  double sum_r = 0, sum_r2 = 0, sum_p = 0, sum_p2 = 0, sum_nn = 0, sum_nn2 = 0;
  long long n = 0, reduce_failed = 0, zeroed = 0, refused = 0;
};

Stats analyse(const Sample& s, int a, double rmax, double pmax) {
  Stats st;
  st.radial = Hist(0.0, rmax, 40);
  st.radial_p = Hist(0.0, rmax, 40);
  st.mom = Hist(0.0, pmax, 40);
  st.nn = Hist(0.0, 4.0, 40);
  st.outer = Hist(0.0, rmax + 1.0, 40);
  st.balance = Hist(-16.0, 4.0, 20);
  st.slot_p.assign(a, 0.0);
  const size_t nn_count = s.rep.size();
  std::vector<double> dmin(a);
  for (size_t k = 0; k < nn_count; ++k) {
    if (s.rep[k].fatal()) {
      ++st.refused;
      continue;
    }
    ++st.n;
    if (s.rep[k].reduce_sum_failed) { ++st.reduce_failed; }
    st.zeroed += s.rep[k].proton_momentum_zeroed;
    const bic::Nucleon* nu = &s.nuc[k * a];
    double outer2 = 0.0;
    deex::Vec3d psum{0.0, 0.0, 0.0};
    double mr = 0.0, mp = 0.0, mnn = 0.0;
    for (int i = 0; i < a; ++i) {
      const double r = g4gpu::mag(nu[i].position);
      const double p = g4gpu::mag(nu[i].momentum.v);
      st.radial.fill(r);
      if (nu[i].type == bic::kProton) {
        st.radial_p.fill(r);
        st.slot_p[i] += 1.0;
      }
      st.mom.fill(p);
      outer2 = std::max(outer2, g4gpu::mag2(nu[i].position));
      psum = psum + nu[i].momentum.v;
      mr += r;
      mp += p;
      double d = 1e30;
      for (int j = 0; j < a; ++j) {
        if (j != i) { d = std::min(d, g4gpu::mag(nu[i].position - nu[j].position)); }
      }
      if (a > 1) {
        st.nn.fill(d);
        mnn += d;
      }
    }
    // the outer radius without its hard-core term, which is a constant per nuclide
    st.outer.fill(std::sqrt(outer2));
    const double bal = g4gpu::mag(psum);
    st.balance.fill(bal > 0.0 ? std::log10(bal) : -15.99);
    st.radial.end_nucleus();
    st.radial_p.end_nucleus();
    st.mom.end_nucleus();
    st.nn.end_nucleus();
    st.outer.end_nucleus();
    st.balance.end_nucleus();
    mr /= a;
    mp /= a;
    mnn = (a > 1) ? mnn / (a - 1 > 0 ? a : 1) : 0.0;
    st.sum_r += mr; st.sum_r2 += mr * mr;
    st.sum_p += mp; st.sum_p2 += mp * mp;
    st.sum_nn += mnn; st.sum_nn2 += mnn * mnn;
  }
  return st;
}

/// Two-sample z of per-nucleus means, from sums and sums of squares.
double zmean(double s1, double q1, double n1, double s2, double q2, double n2) {
  if (n1 < 2 || n2 < 2) { return 0.0; }
  const double m1 = s1 / n1, m2 = s2 / n2;
  const double v1 = std::max(0.0, q1 / n1 - m1 * m1) * n1 / (n1 - 1);
  const double v2 = std::max(0.0, q2 / n2 - m2 * m2) * n2 / (n2 - 1);
  const double se = std::sqrt(v1 / n1 + v2 / n2);
  if (se <= 0.0) { return (m1 == m2) ? 0.0 : 1e9; }
  return (m1 - m2) / se;
}

struct Worst {
  double z = 0.0;
  std::string where;
  int tests = 0;
  void take(double zz, const std::string& w) {
    ++tests;
    if (std::fabs(zz) > std::fabs(z)) {
      z = zz;
      where = w;
    }
  }
};

void compare_hist(const Hist& h1, double n1, const Hist& h2, double n2, const char* name,
                  Worst& w) {
  for (int b = 0; b < h1.nb + 2; ++b) {
    // A bin both samples leave (all but) empty says nothing; its z would be noise on zero.
    if (h1.s[b] + h2.s[b] < 20.0) { continue; }
    const double z = zmean(h1.s[b], h1.s2[b], n1, h2.s[b], h2.s2[b], n2);
    char buf[96];
    std::snprintf(buf, sizeof(buf), "%s bin %d", name, b);
    w.take(z, buf);
  }
}

Worst compare(const Stats& f, const Stats& t, int a) {
  Worst w;
  const double n1 = static_cast<double>(f.n), n2 = static_cast<double>(t.n);
  compare_hist(f.radial, n1, t.radial, n2, "radial", w);
  compare_hist(f.radial_p, n1, t.radial_p, n2, "radial(protons)", w);
  compare_hist(f.mom, n1, t.mom, n2, "momentum", w);
  if (a > 1) { compare_hist(f.nn, n1, t.nn, n2, "nearest-neighbour", w); }
  compare_hist(f.outer, n1, t.outer, n2, "outer radius", w);
  compare_hist(f.balance, n1, t.balance, n2, "log10|sum p|", w);
  for (int i = 0; i < a; ++i) {
    // a Bernoulli per nucleus: sum = sum of squares
    const double z = zmean(f.slot_p[i], f.slot_p[i], n1, t.slot_p[i], t.slot_p[i], n2);
    char buf[64];
    std::snprintf(buf, sizeof(buf), "P(slot %d is a proton)", i);
    w.take(z, buf);
  }
  w.take(zmean(f.sum_r, f.sum_r2, n1, t.sum_r, t.sum_r2, n2), "mean |r| a nucleus");
  w.take(zmean(f.sum_p, f.sum_p2, n1, t.sum_p, t.sum_p2, n2), "mean |p| a nucleus");
  if (a > 1) { w.take(zmean(f.sum_nn, f.sum_nn2, n1, t.sum_nn, t.sum_nn2, n2), "mean nn a nucleus"); }
  {
    // rates per nucleus, Poisson-ish
    const double rf = double(f.reduce_failed), rt = double(t.reduce_failed);
    w.take(zmean(rf, rf, n1, rt, rt, n2), "ReduceSum failure rate");
    const double zf = double(f.zeroed), zt = double(t.zeroed);
    if (zf + zt > 0) { w.take((zf / n1 - zt / n2) / std::sqrt(zf / (n1 * n1) + zt / (n2 * n2) + 1e-300), "zeroed protons"); }
  }
  return w;
}

Sample run_factory(int a, int z, int n, double dscale, int grid_override = 0) {
  Sample s;
  bic::Nucleon* d_nuc = nullptr;
  bic::NucleusReport* d_rep = nullptr;
  deex::Vec3d* d_mom = nullptr;
  double* d_fp = nullptr;
  bic::NucleusSortEntry* d_sums = nullptr;
  int* d_rounds = nullptr;
  const size_t na = static_cast<size_t>(n) * a;
  CK(cudaMalloc(&d_nuc, na * sizeof(bic::Nucleon)));
  CK(cudaMalloc(&d_rep, n * sizeof(bic::NucleusReport)));
  CK(cudaMalloc(&d_mom, na * sizeof(deex::Vec3d)));
  CK(cudaMalloc(&d_fp, na * sizeof(double)));
  CK(cudaMalloc(&d_sums, na * sizeof(bic::NucleusSortEntry)));
  CK(cudaMalloc(&d_rounds, n * sizeof(int)));
  npool::FactoryJob job;
  job.a = a;
  job.z = z;
  job.n = n;
  job.key_lo = 0x1000u;
  job.key_hi = static_cast<unsigned>(a * 1000 + z);
  job.distance_scale = dscale;
  npool::FactoryOutput out;
  out.nucleons = d_nuc;
  out.reports = d_rep;
  out.momentum = d_mom;
  out.fermi_p = d_fp;
  out.test_sums = d_sums;
  out.rounds = d_rounds;
  const int w = npool::factory_width(a);
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  const int grid = (grid_override > 0) ? grid_override : std::min(n, 65535);
  CK(cudaEventRecord(e0));
  switch (w) {
    case 32: npool::nucleus_factory_kernel<32><<<grid, 32>>>(job, out); break;
    case 64: npool::nucleus_factory_kernel<64><<<grid, 64>>>(job, out); break;
    case 128: npool::nucleus_factory_kernel<128><<<grid, 128>>>(job, out); break;
    default: npool::nucleus_factory_kernel<256><<<grid, 256>>>(job, out); break;
  }
  CK(cudaGetLastError());
  CK(cudaEventRecord(e1));
  CK(cudaEventSynchronize(e1));
  CK(cudaEventElapsedTime(&s.ms, e0, e1));
  s.nuc.resize(na);
  s.rep.resize(n);
  s.rounds.resize(n);
  CK(cudaMemcpy(s.nuc.data(), d_nuc, na * sizeof(bic::Nucleon), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(s.rep.data(), d_rep, n * sizeof(bic::NucleusReport), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(s.rounds.data(), d_rounds, n * sizeof(int), cudaMemcpyDeviceToHost));
  cudaFree(d_nuc); cudaFree(d_rep); cudaFree(d_mom); cudaFree(d_fp); cudaFree(d_sums);
  cudaFree(d_rounds);
  cudaEventDestroy(e0); cudaEventDestroy(e1);
  return s;
}

Sample run_thread(int a, int z, int n) {
  Sample s;
  bic::Nucleon* d_nuc = nullptr;
  bic::NucleusReport* d_rep = nullptr;
  deex::Vec3d* d_mom = nullptr;
  double* d_fp = nullptr;
  bic::NucleusSortEntry* d_sums = nullptr;
  double* d_flat = nullptr;
  const size_t na = static_cast<size_t>(n) * a;
  CK(cudaMalloc(&d_nuc, na * sizeof(bic::Nucleon)));
  CK(cudaMalloc(&d_rep, n * sizeof(bic::NucleusReport)));
  CK(cudaMalloc(&d_mom, na * sizeof(deex::Vec3d)));
  CK(cudaMalloc(&d_fp, na * sizeof(double)));
  CK(cudaMalloc(&d_sums, na * sizeof(bic::NucleusSortEntry)));
  CK(cudaMalloc(&d_flat, static_cast<size_t>(n) * bic::kFlatBlock * sizeof(double)));
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  CK(cudaEventRecord(e0));
  thread_nucleus_kernel<<<(n + 63) / 64, 64>>>(a, z, n, static_cast<unsigned>(a * 1000 + z),
                                               d_nuc, d_rep, d_mom, d_fp, d_sums, d_flat);
  CK(cudaGetLastError());
  CK(cudaEventRecord(e1));
  CK(cudaEventSynchronize(e1));
  CK(cudaEventElapsedTime(&s.ms, e0, e1));
  s.nuc.resize(na);
  s.rep.resize(n);
  CK(cudaMemcpy(s.nuc.data(), d_nuc, na * sizeof(bic::Nucleon), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(s.rep.data(), d_rep, n * sizeof(bic::NucleusReport), cudaMemcpyDeviceToHost));
  cudaFree(d_nuc); cudaFree(d_rep); cudaFree(d_mom); cudaFree(d_fp); cudaFree(d_sums);
  cudaFree(d_flat);
  cudaEventDestroy(e0); cudaEventDestroy(e1);
  return s;
}

}  // namespace

int main(int argc, char** argv) {
  const int n = (argc > 1) ? std::atoi(argv[1]) : 20000;
  struct Nuclide { int z, a; const char* name; };
  // Every natural isotope of H, C, N, O, F, Mg, P, S, Ca and Ar - the elements of
  // src/data/materials.cuh's G4_WATER, G4_A-150_TISSUE, G4_BONE_COMPACT_ICRU and G4_AIR.
  const Nuclide nuclides[] = {
      {1, 1, "H1"},    {1, 2, "H2"},    {6, 12, "C12"},  {6, 13, "C13"},  {7, 14, "N14"},
      {7, 15, "N15"},  {8, 16, "O16"},  {8, 17, "O17"},  {8, 18, "O18"},  {9, 19, "F19"},
      {12, 24, "Mg24"}, {12, 25, "Mg25"}, {12, 26, "Mg26"}, {15, 31, "P31"},  {16, 32, "S32"},
      {16, 33, "S33"}, {16, 34, "S34"}, {16, 36, "S36"}, {18, 36, "Ar36"}, {18, 38, "Ar38"},
      {18, 40, "Ar40"}, {20, 40, "Ca40"}, {20, 42, "Ca42"}, {20, 43, "Ca43"}, {20, 44, "Ca44"},
      {20, 46, "Ca46"}, {20, 48, "Ca48"}};

  int dev = 0;
  cudaDeviceProp prop;
  CK(cudaGetDevice(&dev));
  CK(cudaGetDeviceProperties(&prop, dev));
  std::printf("device %d: %s; %d nuclei per nuclide and path\n", dev, prop.name, n);
  // warm-up launches so that neither path's first timing carries the module load
  (void)run_factory(16, 8, 64, 1.0);
  (void)run_thread(16, 8, 64);

  std::printf("%-5s %8s %8s %10s %10s %7s %7s  %-34s %8s\n", "", "factory", "thread", "fac nuc/s",
              "thr nuc/s", "rounds", "tests", "worst row", "z");
  Worst all;
  double fac_ms = 0.0, thr_ms = 0.0;
  for (const Nuclide& nd : nuclides) {
    const Sample f = run_factory(nd.a, nd.z, n, 1.0);
    const Sample t = run_thread(nd.a, nd.z, n);
    fac_ms += f.ms;
    thr_ms += t.ms;
    const double rmax = (nd.a < 17) ? 6.0 : 8.0;  // fermi; the trial ball is ~4.5-7 fm
    const double pmax = 300.0;                    // MeV/c; the Fermi momentum is < 280
    const Stats sf = analyse(f, nd.a, rmax, pmax);
    const Stats stt = analyse(t, nd.a, rmax, pmax);
    const Worst w = compare(sf, stt, nd.a);
    double mean_rounds = 0.0;
    for (int r : f.rounds) { mean_rounds += r; }
    mean_rounds /= std::max<size_t>(1, f.rounds.size());
    std::printf("%-5s %7.2fms %7.2fms %10.3g %10.3g %7.2f %7d  %-34s %+8.2f  (refused %lld/%lld)\n",
                nd.name, f.ms, t.ms, n / (f.ms * 1e-3), n / (t.ms * 1e-3), mean_rounds, w.tests,
                w.where.c_str(), w.z, sf.refused, stt.refused);
    if (std::fabs(w.z) > 5.0 || sf.refused > 0) { ++fails; }
    if (std::fabs(w.z) > std::fabs(all.z)) {
      all.z = w.z;
      all.where = std::string(nd.name) + " " + w.where;
    }
    all.tests += w.tests;
  }
  std::printf("\nall %d nuclides: %d tests, worst %s at %+.2f sigma; factory %.1f ms, thread path "
              "%.1f ms (%.2fx)\n",
              static_cast<int>(sizeof(nuclides) / sizeof(nuclides[0])), all.tests,
              all.where.c_str(), all.z, fac_ms, thr_ms, thr_ms / fac_ms);

  // ---- determinism: the same nuclei to the bit from a grid of 37 blocks as from one a nucleus,
  // because every draw is keyed on (nucleus, lane, phase) and never on the block that ran it
  {
    const Sample g1 = run_factory(16, 8, n, 1.0);
    const Sample g2 = run_factory(16, 8, n, 1.0, 37);
    const Sample c1 = run_factory(12, 6, n, 1.0);
    const Sample c2 = run_factory(12, 6, n, 1.0, 37);
    // Field by field (the struct's padding is not a number anyone reads).
    auto same_nuclei = [](const Sample& x, const Sample& y) {
      if (x.nuc.size() != y.nuc.size()) { return false; }
      for (size_t i = 0; i < x.nuc.size(); ++i) {
        const bic::Nucleon& p = x.nuc[i];
        const bic::Nucleon& q = y.nuc[i];
        const double a[8] = {p.position.x, p.position.y, p.position.z, p.momentum.v.x,
                             p.momentum.v.y, p.momentum.v.z, p.momentum.e, p.binding_energy};
        const double b[8] = {q.position.x, q.position.y, q.position.z, q.momentum.v.x,
                             q.momentum.v.y, q.momentum.v.z, q.momentum.e, q.binding_energy};
        if (std::memcmp(a, b, sizeof(a)) != 0 || p.type != q.type) { return false; }
      }
      return true;
    };
    const bool same = same_nuclei(g1, g2) && same_nuclei(c1, c2);
    std::printf("\ndeterminism: O16 and C12, %d nuclei each, grid %d against grid 37: %s\n", n,
                std::min(n, 65535), same ? "identical to the bit" : "DIFFERENT");
    if (!same) { ++fails; }
  }

  // ---- anti-vacuity: the hard core halved must be caught
  std::printf("\nanti-vacuity: the factory's hard-core distance halved\n");
  const Nuclide av[] = {{8, 16, "O16"}, {6, 12, "C12"}, {20, 40, "Ca40"}};
  for (const Nuclide& nd : av) {
    const Sample f = run_factory(nd.a, nd.z, n, 0.5);
    const Sample t = run_thread(nd.a, nd.z, n);
    const double rmax = (nd.a < 17) ? 6.0 : 8.0;
    const Worst w = compare(analyse(f, nd.a, rmax, 300.0), analyse(t, nd.a, rmax, 300.0), nd.a);
    const bool caught = std::fabs(w.z) > 5.0;
    std::printf("  %-5s worst %-34s %+9.2f sigma: %s\n", nd.name, w.where.c_str(), w.z,
                caught ? "CAUGHT" : "NOT CAUGHT");
    if (!caught) { ++fails; }
  }

  std::printf("\n%s (%d failure%s)\n", fails ? "FAIL" : "PASS", fails, fails == 1 ? "" : "s");
  return fails ? 1 : 0;
}
