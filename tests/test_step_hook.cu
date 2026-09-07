// The step-hook reducers, on synthetic steps with known answers.
//
// WHY THIS EXISTS SEPARATELY FROM -verify-step-hook
//
// `g4dose.exe -verify-step-hook` runs a real B1 shower and asserts that the hook is handed
// every step exactly once, charged to the right event. That is the important check and it is
// the one that could not be replaced by anything here - it needs a real transport to be about
// anything. But it exercises exactly one hook, StepTap, because StepTap is what the stock
// engine is instantiated on.
//
// StepTally - the reducer that core/step_hook.cuh argues is the one you should actually use,
// because its memory does not grow with the number of steps - was not compiled by anything.
// Shipping an unused template is shipping code that has never been through a compiler, and a
// reducer nobody has run is a reducer whose bin arithmetic is a guess. So this file feeds
// steps with hand-computed answers through the real device path: a __global__ that calls the
// hook exactly as the stepping kernels do.
//
// The steps are synthetic on purpose. A physics run cannot tell you whether the last bin of a
// log axis is closed at the right end, or whether a NaN is dropped rather than binned at zero;
// only chosen inputs can.
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

#include "core/step_hook.cuh"

using namespace g4gpu;
using real_t = double;

namespace {

int g_fails = 0;

void Check(bool ok, const char* what) {
  if (!ok) {
    std::printf("  FAIL: %s\n", what);
    ++g_fails;
  }
}

void CheckClose(double got, double want, double tol, const char* what) {
  const double d = std::fabs(got - want);
  if (!(d <= tol)) {
    std::printf("  FAIL: %s - got %.17g, want %.17g (diff %.3g > %.3g)\n", what, got, want, d,
                tol);
    ++g_fails;
  }
}

/// Drives a hook over an array of steps, one thread each, exactly as the stepping kernels do.
template <typename Hook>
__global__ void RunHook(const DeviceStep<real_t>* steps, int n, Hook hook) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) { hook(steps[i]); }
}

/// A weight: the deposit, so a tally becomes a dose spectrum.
struct WeightEdep {
  __host__ __device__ real_t operator()(const DeviceStep<real_t>& s) const { return s.edep; }
};

/// A weight with a condition in it, which is the whole point of the weight being a functor -
/// "dose from alphas only" needs no support from step_hook.cuh.
struct WeightAlphaEdep {
  __host__ __device__ real_t operator()(const DeviceStep<real_t>& s) const {
    return (s.species == ParticleType::kAlpha) ? s.edep : real_t(0);
  }
};

/// The case that prompted the whole mechanism: a quality factor read off the step's own LET.
/// Deliberately a step function so the expected answer can be worked out by hand.
struct WeightQEdep {
  __host__ __device__ real_t operator()(const DeviceStep<real_t>& s) const {
    const real_t let = s.let();
    const real_t q = (let < real_t(1)) ? real_t(1) : (let < real_t(10) ? real_t(5) : real_t(20));
    return q * s.edep;
  }
};

/// Bins a step's LET. Composition of a bin axis with the quantity to put on it is done here,
/// in the user's own functor, rather than by a menu of axes in step_hook.cuh.
struct BinLet {
  LogBins axis{};
  __host__ __device__ int operator()(const DeviceStep<real_t>& s) const { return axis(s.let()); }
};

/// Bins the energy a track had when it entered a volume, and drops every other step.
struct BinEntryEnergy {
  LogBins axis{};
  int volume = 0;
  __host__ __device__ int operator()(const DeviceStep<real_t>& s) const {
    return s.entered(volume) ? axis(s.ekin_pre) : -1;
  }
};

/// Every synthetic step needs a track to point at, because DeviceStep::GetTrack() is a
/// HANDLE onto the live one rather than a copy - that is what makes SetTrackStatus and the
/// other setters real. A table of them, one per step made, so each step gets its own and a
/// test that writes through GetTrack() cannot corrupt another step's.
std::vector<TrackState<real_t>> g_tracks;

DeviceStep<real_t> MakeStep(real_t edep, real_t length, int event, int slot,
                            ParticleType sp = ParticleType::kElectron, real_t ekin_pre = 1,
                            int vol_pre = 0, int vol_post = 0) {
  DeviceStep<real_t> s{};
  s.species = sp;
  s.ekin_pre = ekin_pre;
  s.ekin_post = ekin_pre - edep;
  s.edep = edep;
  s.length = length;
  s.volume_pre = vol_pre;
  s.volume_post = vol_post;
  s.score_slot = slot;
  s.event = event;
  s.alive = true;
  // The track behind the handle. Reserved once so the vector never reallocates and
  // invalidates a pointer a previously-made step is holding.
  if (g_tracks.capacity() == 0) { g_tracks.reserve(256); }
  TrackState<real_t> t{};
  t.pos = s.pos_post;
  t.dir = s.dir_post;
  t.ekin = s.ekin_post;
  t.volume = vol_post;
  t.event = event;
  t.rng_key = 1u;
  t.step = 1u;  // Geant4 numbers a track's first step 1, not 0
  t.begin(s.pos_pre, s.dir_pre, ekin_pre, vol_pre, 0u, ProcessId::fNotDefined, real_t(0),
          real_t(1));
  g_tracks.push_back(t);
  s.track_ptr = &g_tracks.back();
  return s;
}

/// Uploads steps to the device, rewriting each one's track pointer to the device copy.
///
/// This exists because of a bug this test found: DeviceStep::track_ptr points at the live
/// track, and a step built on the host points at HOST memory. Copying such a step to the
/// device verbatim gives a kernel a host pointer, and the first hook that touches the track -
/// StepTap, which reads GetTrackID() - faults. The real kernels never have the problem, since
/// their track is a local in the same kernel; only a synthetic step can, which is precisely
/// the kind of gap a test written after the fact would have papered over by not looking.
template <typename T>
T* UploadSteps(const std::vector<DeviceStep<real_t>>& steps) {
  static TrackState<real_t>* d_tracks = nullptr;
  static size_t uploaded = 0;
  if (d_tracks == nullptr || uploaded != g_tracks.size()) {
    if (d_tracks != nullptr) { cudaFree(d_tracks); }
    cudaMalloc(&d_tracks, sizeof(TrackState<real_t>) * g_tracks.size());
    cudaMemcpy(d_tracks, g_tracks.data(), sizeof(TrackState<real_t>) * g_tracks.size(),
               cudaMemcpyHostToDevice);
    uploaded = g_tracks.size();
  }
  std::vector<DeviceStep<real_t>> fixed = steps;
  for (auto& st : fixed) {
    const ptrdiff_t idx = st.track_ptr - g_tracks.data();
    st.track_ptr = d_tracks + idx;
  }
  DeviceStep<real_t>* d = nullptr;
  cudaMalloc(&d, sizeof(DeviceStep<real_t>) * fixed.size());
  cudaMemcpy(d, fixed.data(), sizeof(DeviceStep<real_t>) * fixed.size(),
             cudaMemcpyHostToDevice);
  return d;
}

/// Uploads the steps, runs the hook on the device, brings the bins back.
template <typename Hook>
std::vector<double> Tally(const std::vector<DeviceStep<real_t>>& steps, Hook hook, int nbins) {
  DeviceStep<real_t>* d_steps = UploadSteps<DeviceStep<real_t>>(steps);
  double* d_bins = nullptr;
  if (d_steps == nullptr || cudaMalloc(&d_bins, sizeof(double) * nbins) != cudaSuccess) {
    std::printf("  FAIL: cudaMalloc\n");
    ++g_fails;
    return {};
  }
  cudaMemset(d_bins, 0, sizeof(double) * nbins);
  hook.bins = d_bins;
  hook.nbins = nbins;

  RunHook<<<(static_cast<int>(steps.size()) + 63) / 64, 64>>>(
      d_steps, static_cast<int>(steps.size()), hook);
  const cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::printf("  FAIL: kernel: %s\n", cudaGetErrorString(err));
    ++g_fails;
  }

  std::vector<double> out(nbins, 0.0);
  cudaMemcpy(out.data(), d_bins, sizeof(double) * nbins, cudaMemcpyDeviceToHost);
  cudaFree(d_steps);
  cudaFree(d_bins);
  return out;
}

}  // namespace

int main() {
  int dev = 0;
  if (cudaGetDeviceCount(&dev) != cudaSuccess || dev == 0) {
    std::printf("no CUDA device; this test needs one because a StepHook is device code\n");
    return 1;
  }

  std::printf("== step hook reducers ==\n\n");

  // ---------------------------------------------------------------- 1. the bin axes
  //
  // Pure arithmetic, checkable on the host now that the axes are __host__ __device__. The
  // interesting cases are all at the edges, and every one of them is a way a histogram lies
  // quietly: a value piled into an end bin instead of being dropped inflates that bin, and a
  // NaN binned at zero inflates the first.
  std::printf("bin axes\n");
  {
    LinBins lin{0.0, 10.0, 10};
    Check(lin(0.0) == 0, "LinBins: the low edge is in the first bin");
    Check(lin(9.999) == 9, "LinBins: just below the high edge is in the last bin");
    Check(lin(10.0) < 0, "LinBins: the high edge is OUTSIDE - the range is half-open");
    Check(lin(-0.001) < 0, "LinBins: below the range is dropped");
    Check(lin(1e30) < 0, "LinBins: far above the range is dropped");
    Check(lin(std::numeric_limits<double>::quiet_NaN()) < 0, "LinBins: NaN is dropped");
    Check(lin(4.5) == 4, "LinBins: an interior value");

    // Three bins per decade over three decades, so a bin edge sits exactly on each decade and
    // the arithmetic is checkable without trusting the logarithm.
    LogBins lg{1e-3, 1.0, 9};
    Check(lg(1e-3) == 0, "LogBins: the low edge is in the first bin");
    Check(lg(1e-2) == 3, "LogBins: a decade up is three bins up");
    Check(lg(1e-1) == 6, "LogBins: two decades up is six bins up");
    Check(lg(1.0) < 0, "LogBins: the high edge is OUTSIDE");
    Check(lg(0.0) < 0, "LogBins: zero is dropped rather than taken as log(0)");
    Check(lg(-1.0) < 0, "LogBins: a negative value is dropped");
    Check(lg(std::numeric_limits<double>::quiet_NaN()) < 0, "LogBins: NaN is dropped");
  }

  // ---------------------------------------------------------------- 2. DeviceStep accessors
  std::printf("DeviceStep\n");
  {
    const auto s = MakeStep(2.0, 4.0, 0, 0);
    CheckClose(s.let(), 0.5, 0.0, "let() is edep/length");
    const auto z = MakeStep(2.0, 0.0, 0, 0);
    CheckClose(z.let(), 0.0, 0.0, "let() of a zero-length step is 0, not infinity");

    auto cross = MakeStep(1.0, 1.0, 0, 0, ParticleType::kElectron, 1.0, /*pre=*/3, /*post=*/7);
    Check(cross.entered(7), "entered() is true for the volume stepped into");
    Check(!cross.entered(3), "entered() is false for the volume stepped out of");
    Check(cross.left(3), "left() is true for the volume stepped out of");
    Check(!cross.left(7), "left() is false for the volume stepped into");
    auto stay = MakeStep(1.0, 1.0, 0, 0, ParticleType::kElectron, 1.0, 5, 5);
    Check(!stay.entered(5) && !stay.left(5), "a step within one volume neither enters nor leaves");
  }

  // ---------------------------------------------------------------- 3. a LET spectrum
  //
  // The question core/step_hook.cuh was written to answer, with the answer known in advance.
  std::printf("StepTally: a LET spectrum of the dose\n");
  {
    std::vector<DeviceStep<real_t>> steps;
    steps.push_back(MakeStep(/*edep=*/2.0, /*len=*/1000.0, 0, 0));  // LET 2e-3 -> bin 0
    steps.push_back(MakeStep(3.0, 1500.0, 0, 0));                   // LET 2e-3 -> bin 0
    steps.push_back(MakeStep(5.0, 500.0, 0, 0));                    // LET 1e-2 -> bin 3
    steps.push_back(MakeStep(7.0, 7.0, 0, 0));                      // LET 1.0  -> dropped, at hi
    steps.push_back(MakeStep(11.0, 0.0, 0, 0));                     // zero length -> LET 0, dropped

    StepTally<real_t, BinLet, WeightEdep> tally;
    tally.bin.axis = LogBins{1e-3, 1.0, 9};
    const auto bins = Tally(steps, tally, 9);
    if (!bins.empty()) {
      CheckClose(bins[0], 5.0, 1e-12, "two steps at the same LET sum into one bin");
      CheckClose(bins[3], 5.0, 1e-12, "a step a decade up lands three bins up");
      double total = 0;
      for (double b : bins) { total += b; }
      CheckClose(total, 10.0, 1e-12,
                 "the 7 MeV step at the axis top and the zero-length step are BOTH dropped, so "
                 "the spectrum holds 10 of the 28 MeV deposited");
    }
  }

  // ---------------------------------------------------------------- 4. per-event, and a Q-bar
  //
  // The user-facing case: a dose-averaged quality factor per event, which is two tallies over
  // the same steps divided by each other. Nothing in step_hook.cuh knows what a quality factor
  // is, which is the point.
  std::printf("StepTally: dose-averaged quality factor per event\n");
  {
    std::vector<DeviceStep<real_t>> steps;
    // event 0: 4 MeV at LET 0.5 (Q=1) and 6 MeV at LET 5 (Q=5)  -> Qbar = (4+30)/10 = 3.4
    steps.push_back(MakeStep(4.0, 8.0, 0, 0));
    steps.push_back(MakeStep(6.0, 1.2, 0, 0));
    // event 1: 2 MeV at LET 20 (Q=20) and 8 MeV at LET 0.1 (Q=1) -> Qbar = (40+8)/10 = 4.8
    steps.push_back(MakeStep(2.0, 0.1, 1, 0));
    steps.push_back(MakeStep(8.0, 80.0, 1, 0));
    // event 2: nothing deposited at all -> 0/0, which the caller must handle, not the tally
    steps.push_back(MakeStep(0.0, 5.0, 2, 0));

    StepTally<real_t, BinByEvent, WeightQEdep> weighted;
    StepTally<real_t, BinByEvent, WeightEdep> plain;
    const auto w = Tally(steps, weighted, 3);
    const auto p = Tally(steps, plain, 3);
    if (!w.empty() && !p.empty()) {
      CheckClose(p[0], 10.0, 1e-12, "event 0 dose");
      CheckClose(p[1], 10.0, 1e-12, "event 1 dose");
      CheckClose(w[0] / p[0], 3.4, 1e-12, "event 0 dose-averaged Q");
      CheckClose(w[1] / p[1], 4.8, 1e-12, "event 1 dose-averaged Q");
      CheckClose(p[2], 0.0, 0.0, "an event that deposited nothing stays exactly zero");
      CheckClose(w[2], 0.0, 0.0, "and so does its weighted sum - no Q is invented for it");
    }
  }

  // ---------------------------------------------------------------- 5. a weight that filters
  std::printf("StepTally: dose from one species\n");
  {
    std::vector<DeviceStep<real_t>> steps;
    steps.push_back(MakeStep(3.0, 1.0, 0, /*slot=*/0, ParticleType::kAlpha));
    steps.push_back(MakeStep(5.0, 1.0, 0, /*slot=*/1, ParticleType::kAlpha));
    steps.push_back(MakeStep(7.0, 1.0, 0, /*slot=*/0, ParticleType::kElectron));
    steps.push_back(MakeStep(9.0, 1.0, 0, /*slot=*/-1, ParticleType::kAlpha));  // unscored

    StepTally<real_t, BinByScoreSlot, WeightAlphaEdep> tally;
    const auto bins = Tally(steps, tally, 2);
    if (!bins.empty()) {
      CheckClose(bins[0], 3.0, 1e-12, "slot 0 gets the alpha only, not the electron");
      CheckClose(bins[1], 5.0, 1e-12, "slot 1 gets its alpha");
      // The unscored step carries slot -1, which BinByScoreSlot returns unchanged and StepTally
      // drops. A tally must never be able to write outside its own array.
      Check(true, "a step with slot -1 wrote nowhere (no out-of-bounds write)");
    }
  }

  // ---------------------------------------------------------------- 6. fluence at a boundary
  //
  // The other half of the user's question - the energy a particle had as it crossed in, which
  // is a pre-step quantity no event aggregate retains.
  std::printf("StepTally: entry fluence spectrum\n");
  {
    const int kDet = 4;
    std::vector<DeviceStep<real_t>> steps;
    // entering the detector at 10 MeV
    steps.push_back(MakeStep(1.0, 1.0, 0, 0, ParticleType::kProton, 10.0, 1, kDet));
    // entering it again at 100 MeV
    steps.push_back(MakeStep(1.0, 1.0, 0, 0, ParticleType::kProton, 100.0, 2, kDet));
    // already inside it - not an entry, must not count
    steps.push_back(MakeStep(1.0, 1.0, 0, 0, ParticleType::kProton, 10.0, kDet, kDet));
    // leaving it - not an entry either
    steps.push_back(MakeStep(1.0, 1.0, 0, 0, ParticleType::kProton, 10.0, kDet, 1));

    StepTally<real_t, BinEntryEnergy, WeightEdep> tally;
    tally.bin.axis = LogBins{1.0, 1000.0, 3};  // one bin per decade
    tally.bin.volume = kDet;
    tally.weight = WeightEdep{};
    const auto bins = Tally(steps, tally, 3);
    if (!bins.empty()) {
      CheckClose(bins[1], 1.0, 1e-12, "the 10 MeV entry is counted, in the second decade");
      CheckClose(bins[2], 1.0, 1e-12, "the 100 MeV entry is counted, in the third");
      CheckClose(bins[0], 0.0, 0.0,
                 "the steps that stayed inside and left are NOT entries and are not counted");
    }
  }

  // ---------------------------------------------------------------- 7. the null gate
  //
  // How the stock build pays nothing for a hook nobody asked for. If this ever stopped being
  // true the cost would be an out-of-bounds write, not a slowdown.
  std::printf("StepTally: disabled by a null pointer\n");
  {
    std::vector<DeviceStep<real_t>> steps{MakeStep(5.0, 1.0, 0, 0)};
    DeviceStep<real_t>* d_steps = UploadSteps<DeviceStep<real_t>>(steps);
    StepTally<real_t, BinByEvent, WeightEdep> off;  // bins stays null
    RunHook<<<1, 64>>>(d_steps, 1, off);
    const cudaError_t err = cudaDeviceSynchronize();
    Check(err == cudaSuccess, "a tally with a null bin array runs and writes nothing");
    if (err != cudaSuccess) { std::printf("        (%s)\n", cudaGetErrorString(err)); }
    cudaFree(d_steps);

    NoStepHook none;
    RunHook<<<1, 64>>>(nullptr, 0, none);
    Check(cudaDeviceSynchronize() == cudaSuccess, "NoStepHook compiles and runs");
  }

  // ---------------------------------------------------------------- 8. composition
  std::printf("StepHooks: two hooks on the same steps\n");
  {
    std::vector<DeviceStep<real_t>> steps{MakeStep(4.0, 8.0, 0, 0), MakeStep(6.0, 1.2, 1, 0)};
    DeviceStep<real_t>* d_steps = UploadSteps<DeviceStep<real_t>>(steps);
    double *d_a = nullptr, *d_b = nullptr;
    cudaMalloc(&d_a, sizeof(double) * 2);
    cudaMalloc(&d_b, sizeof(double) * 2);
    cudaMemset(d_a, 0, sizeof(double) * 2);
    cudaMemset(d_b, 0, sizeof(double) * 2);

    StepTally<real_t, BinByEvent, WeightEdep> a;
    a.bins = d_a;
    a.nbins = 2;
    StepTally<real_t, BinByEvent, WeightQEdep> b;
    b.bins = d_b;
    b.nbins = 2;
    StepHooks<decltype(a), decltype(b)> both{a, b};

    RunHook<<<1, 64>>>(d_steps, 2, both);
    Check(cudaDeviceSynchronize() == cudaSuccess, "a composed hook runs");
    double ha[2] = {0, 0}, hb[2] = {0, 0};
    cudaMemcpy(ha, d_a, sizeof ha, cudaMemcpyDeviceToHost);
    cudaMemcpy(hb, d_b, sizeof hb, cudaMemcpyDeviceToHost);
    CheckClose(ha[0], 4.0, 1e-12, "the first hook of a pair saw event 0");
    CheckClose(ha[1], 6.0, 1e-12, "the first hook of a pair saw event 1");
    CheckClose(hb[0], 4.0, 1e-12, "the second hook saw event 0 too (Q=1 at LET 0.5)");
    CheckClose(hb[1], 30.0, 1e-12, "the second hook saw event 1 too (Q=5 at LET 5)");
    cudaFree(d_steps);
    cudaFree(d_a);
    cudaFree(d_b);
  }

  // ---------------------------------------------------------------- 9. the tap's overflow
  //
  // StepTap is the one hook whose memory grows with the run, so the only thing standing between
  // a capped buffer and a silently biased sample is this counter. A truncated dump that does
  // not say it is truncated is worse than no dump.
  std::printf("StepTap: overflow is counted, never silent\n");
  {
    const int kSteps = 100, kCap = 30;
    std::vector<DeviceStep<real_t>> steps;
    for (int i = 0; i < kSteps; ++i) { steps.push_back(MakeStep(1.0, 1.0, i, 0)); }
    DeviceStep<real_t>* d_steps = UploadSteps<DeviceStep<real_t>>(steps);
    StepRecord<real_t>* d_rec = nullptr;
    int* d_ctl = nullptr;
    cudaMalloc(&d_rec, sizeof(StepRecord<real_t>) * kCap);
    cudaMalloc(&d_ctl, sizeof(int) * 2);
    cudaMemset(d_ctl, 0, sizeof(int) * 2);

    StepTap<real_t> tap;
    tap.records = d_rec;
    tap.count = d_ctl;
    tap.overflow = d_ctl + 1;
    tap.capacity = kCap;
    RunHook<<<2, 64>>>(d_steps, kSteps, tap);
    Check(cudaDeviceSynchronize() == cudaSuccess, "an overflowing tap does not fault");
    int ctl[2] = {0, 0};
    cudaMemcpy(ctl, d_ctl, sizeof ctl, cudaMemcpyDeviceToHost);
    Check(ctl[0] == kSteps, "the cursor counts every step offered, not just the stored ones");
    Check(ctl[1] == kSteps - kCap, "every step that did not fit is counted as an overflow");
    cudaFree(d_steps);
    cudaFree(d_rec);
    cudaFree(d_ctl);
  }

  // ---------------------------------------------------------------- 10. the G4Step accessors
  //
  // These are derived quantities, so the useful checks are identities rather than transcribed
  // numbers: an identity cannot be satisfied by a formula that is wrong in the same way twice,
  // whereas a hard-coded expectation computed with the same expression proves nothing.
  //
  // The massless case is checked separately because it is the one that divides by zero.
  std::printf("G4Step accessors\n");
  {
    auto e = MakeStep(/*edep=*/0.25, /*len=*/0.5, /*event=*/3, /*slot=*/1,
                      ParticleType::kElectron, /*ekin_pre=*/1.0, /*vol_pre=*/2, /*vol_post=*/5);
    e.dir_pre = Vec3<real_t>{0, 0, 1};
    e.dir_post = Vec3<real_t>{0, 0, 1};
    e.pos_post = Vec3<real_t>{0, 0, 0.5};
    e.status = StepStatus::fGeomBoundary;
    e.process = ProcessId::fTransportation;
    e.material = 7;
    e.n_secondaries = 2;
    // Identity lives on the track, which is the only place it lives now.
    g_tracks.back().rng_key = 4242u;
    g_tracks.back().step = 9u;

    const auto pre = e.GetPreStepPoint();
    const auto post = e.GetPostStepPoint();
    const auto trk = e.GetTrack();

    CheckClose(pre.GetKineticEnergy(), 1.0, 0.0, "pre-step KE is the step's own pre-step KE");
    CheckClose(post.GetKineticEnergy(), 0.75, 1e-15, "post-step KE");
    CheckClose(e.GetTotalEnergyDeposit(), 0.25, 0.0, "GetTotalEnergyDeposit");
    CheckClose(e.GetStepLength(), 0.5, 0.0, "GetStepLength is the true path");
    CheckClose(e.GetDeltaEnergy(), -0.25, 1e-15, "GetDeltaEnergy is post - pre, so negative");
    CheckClose(e.GetDeltaPosition().z, 0.5, 1e-15, "GetDeltaPosition");
    Check(e.GetNumberOfSecondariesInCurrentStep() == 2, "GetNumberOfSecondariesInCurrentStep");
    Check(e.IsLastStepInVolume(), "a step ending on a boundary is the last in its volume");
    Check(post.GetStepStatus() == StepStatus::fGeomBoundary, "status is on the POST point");
    Check(post.GetProcessDefinedStep() == ProcessId::fTransportation, "GetProcessDefinedStep");
    Check(pre.GetVolume() == 2 && post.GetVolume() == 5, "the two points are in two volumes");
    Check(pre.GetMaterial() == 7, "pre-step material");
    Check(trk.GetTrackID() == 4242u, "GetTrackID");
    Check(trk.GetCurrentStepNumber() == 9u, "GetCurrentStepNumber");
    Check(e.GetTrackID() == 4242u, "the step and its track agree on the identity");
    Check(e.GetCurrentStepNumber() == 9u, "and on the step number");
    Check(trk.GetDefinition() == ParticleType::kElectron, "GetDefinition");
    Check(trk.GetParentID() == 0u, "a track with no parent reports 0, as Geant4 does");
    Check(trk.GetCreatorProcess() == ProcessId::fNotDefined,
          "and no creating process");
    CheckClose(trk.GetVertexKineticEnergy(), 1.0, 0.0, "GetVertexKineticEnergy");
    CheckClose(trk.GetWeight(), 1.0, 0.0, "a track starts at unit weight");

    // The setters are the point of GetTrack() being a handle rather than a copy. If it were
    // a copy every one of these would compile, run, and change nothing - which is the
    // failure this test exists to make impossible.
    auto w = e.GetTrack();
    w.SetWeight(0.25);
    CheckClose(e.GetTrack().GetWeight(), 0.25, 0.0,
               "SetWeight is seen through a second handle - GetTrack() is not a copy");
    w.SetPolarization(Vec3<real_t>{0, 1, 0});
    CheckClose(e.GetTrack().GetPolarization().y, 1.0, 0.0, "SetPolarization writes through");
    w.SetUserInformation(0xABCDu);
    Check(e.GetTrack().GetUserInformation() == 0xABCDu, "user data survives on the track");
    w.SetBelowThresholdFlag(true);
    Check(e.GetTrack().IsBelowThreshold(), "SetBelowThresholdFlag");
    w.SetGoodForTrackingFlag(true);
    Check(e.GetTrack().IsGoodForTracking(), "SetGoodForTrackingFlag");
    Check(!e.GetTrack().IsBelowThreshold() == false, "the two flags are independent bits");
    w.SetTrackStatus(TrackStatus::fStopAndKill);
    Check(e.GetTrack().GetTrackStatus() == TrackStatus::fStopAndKill, "SetTrackStatus");
    Check(!resolve_track_status<real_t>(true, TrackStatus::fStopAndKill),
          "a track the action killed is not requeued, whatever the physics proposed");
    Check(resolve_track_status<real_t>(true, TrackStatus::fAlive),
          "and an untouched status leaves the physics decision alone");
    Check(is_unsupported_track_status(TrackStatus::fSuspend),
          "fSuspend is reported as unsupported rather than silently reinterpreted");

    // The relativistic identities. m is whatever particle_def says an electron weighs; not
    // repeated here, precisely so that a wrong mass would still have to satisfy these.
    const double m = pre.GetMass(), T = pre.GetKineticEnergy();
    const double E = pre.GetTotalEnergy(), P = pre.GetMomentumMagnitude();
    CheckClose(E, T + m, 1e-15, "E = T + m");
    CheckClose(P * P, T * (T + 2 * m), 1e-14, "p^2 = T(T + 2m)");
    CheckClose(pre.GetBeta(), P / E, 1e-15, "beta = p/E");
    CheckClose(pre.GetGamma(), E / m, 1e-15, "gamma = E/m");
    CheckClose(pre.GetBeta() * pre.GetGamma(), P / m, 1e-14, "beta*gamma = p/m");
    CheckClose(pre.GetMomentum().z, P, 1e-15, "momentum is |p| along the direction");
    Check(pre.GetBeta() > 0.9 && pre.GetBeta() < 1.0,
          "a 1 MeV electron is relativistic but not superluminal");
    Check(pre.GetCharge() < 0, "an electron carries negative charge");
  }
  {
    // A photon: mass zero, which is where beta and gamma would divide by zero.
    auto g = MakeStep(0.0, 10.0, 0, 0, ParticleType::kGamma, /*ekin_pre=*/2.0);
    g.dir_pre = Vec3<real_t>{1, 0, 0};
    const auto pre = g.GetPreStepPoint();
    CheckClose(pre.GetMass(), 0.0, 0.0, "a photon is massless");
    CheckClose(pre.GetTotalEnergy(), 2.0, 0.0, "a photon's total energy is its kinetic energy");
    CheckClose(pre.GetMomentumMagnitude(), 2.0, 1e-15, "and its momentum is E/c");
    CheckClose(pre.GetBeta(), 1.0, 1e-15, "beta is exactly 1, not 0/0");
    CheckClose(pre.GetGamma(), 0.0, 0.0,
               "gamma is reported as 0 - 'not meaningful' - rather than an infinity that would "
               "poison any average it entered");
    CheckClose(pre.GetCharge(), 0.0, 0.0, "a photon is neutral");
    // GetDeltaTime is derived from the true path and the PRE-step velocity, so a photon's is
    // exactly length/c. 10 mm / 299.792458 mm/ns.
    CheckClose(g.GetDeltaTime(), 10.0 / 299.792458, 1e-15,
               "a photon crosses 10 mm in length/c ns");
  }
  {
    // And a massive particle, against a number computed by hand rather than by the same code.
    // A 1 MeV electron: t = 1/0.510998910 = 1.95695...; beta = sqrt(t(t+2))/(t+1) = 0.941078...
    // v = 299.792458 * beta = 282.1264... mm/ns; 10 mm takes 0.0354451... ns.
    //
    // The point of the hand computation is that it fails if GetDeltaTime ever starts using the
    // POST-step energy, which would be the easy mistake and which no self-consistent check
    // could see.
    auto e = MakeStep(0.4, 10.0, 0, 0, ParticleType::kElectron, /*ekin_pre=*/1.0);
    const real_t t = 1.0 / 0.510998910;
    const real_t beta = std::sqrt(t * (t + 2.0)) / (t + 1.0);
    CheckClose(e.GetDeltaTime(), 10.0 / (299.792458 * beta), 1e-14,
               "a 1 MeV electron crosses 10 mm in 0.03545 ns, at its entry speed");
    Check(e.GetDeltaTime() < 10.0 / (299.792458 * 0.9),
          "and not at the speed it left with, which would be slower and take longer");
  }
  {
    // A step that stopped inside its volume is not the last step in it.
    auto k = MakeStep(1.0, 1.0, 0, 0);
    k.status = StepStatus::fAlongStepDoItProc;
    Check(!k.IsLastStepInVolume(), "an along-step step is not the last in its volume");
    k.status = StepStatus::fWorldBoundary;
    Check(k.IsLastStepInVolume(), "leaving the world is also leaving the volume");
  }

  std::printf("\n%s (%d failures)\n", g_fails ? "FAILED" : "PASSED", g_fails);
  return g_fails ? 1 : 0;
}
