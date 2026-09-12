// A project's own stepping action, compiled in the project, with g4gpu not rebuilt.
//
// WHAT THIS IS TESTING
//
// Not physics. The arrangement. The claim is that a project can define its own step-level
// stepping action - an ordinary class, in the project's own file, in the shape a Geant4 user
// would write - and get it into the transport WITHOUT rebuilding the engine, exactly as a
// Geant4 project compiles against a Geant4 that was built once.
//
// That claim rests on something that could easily be false, and the comment at the top of
// transport_run_impl.cuh is the record of why: a __global__ template instantiated in two
// translation units emits a device stub into each, and nvcc rejects the duplicates. The whole
// arrangement works only because the collision is per *specialization*. The engine object
// holds TransportEngine<double, StepTap<double>>; this file holds
// TransportEngine<double, QualityFactorScoring>; the kernels mangle differently and coexist.
//
// So this file is built the way a real project would be built - its own translation unit, the
// impl header included once, one explicit instantiation of the ENGINE, its kernels compiled
// one per unit into out/hook_qfs.lib beside the stock engine's out/transport_run.lib - and if
// the arrangement is wrong it fails at compile or link time rather than silently. Since P8e
// that last clause has a gate behind it as well as a hope: build_all.bat runs cuobjdump over
// this file's object and fails the build if a stepping kernel is in it, because a launch whose
// declaration went missing would otherwise cost minutes of nvcc and say nothing.
//
// WHAT IT ALSO DEMONSTRATES
//
// The scoring case the whole step hook was built for, which no per-event aggregate can
// produce: a dose-averaged quality factor. Q depends on the LET of each individual step, and
// an aggregate has neither a step nor a length. Note what the class holds - two device
// pointers and an integer - and what it does not: it stores nothing per step, so its memory is
// the same for twenty thousand events as for twenty billion.
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <vector>

// The action itself is include/QualityFactorScoring.hh, and it is a header rather than a class
// in this file because the kernels are no longer compiled here - see the engine block below.
#include "QualityFactorScoring.hh"

// ---------------------------------------------------------------- the project's engine
//
// Name the hook, pull in the engine, and instantiate it for the hook - and DECLARE the kernels
// rather than defining them here, which is the one line that changed in P8e and the reason
// this file compiles at all.
//
// It used to define them. `template class TransportEngine<double, QualityFactorScoring>` makes
// the eighteen `<<<>>>` launches inside BeamOn instantiate eighteen stepping kernels for this
// hook type, into this translation unit, and build_all.bat's comment called the three and a
// half minutes that cost "the honest cost of the arrangement". It was honest and it was never
// necessary, and with the Urban ion branch live (docs/RISK.md V66) it stopped being possible:
// ptxas died with 0xC0000005 on run_step_hadron<double, ParticleType(13), QualityFactorScoring>
// - the same crash, from the same cause, that V65 split the ENGINE'S translation unit to cure.
// The hook is a template parameter like any other, so a project's kernels split exactly as the
// engine's do: build_hook_engine.bat generates one unit per kernel for this class and archives
// them into out\hook_qfs.lib, and hook_kernels.cuh below is what stops them being compiled
// twice. The declarations must come after transport_run_impl.cuh, whose macros they are
// written against; the header says so and fails the build if they do not.
#define G4STEP_HOOK QualityFactorScoring
#include "g4/G4RunManager.hh"
#include "g4/G4SDManager.hh"
#include "g4/G4SystemOfUnits.hh"
#include "host/transport_run_impl.cuh"
#include "scenes/scene_registry.hh"

#include "hook_kernels.cuh"  // generated beside out\hook_qfs.lib by build_hook_engine.bat

namespace g4gpu::host {
template class TransportEngine<double, QualityFactorScoring>;
}  // namespace g4gpu::host

using namespace g4gpu;

int main() {
  int ndev = 0;
  if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
    std::printf("no CUDA device\n");
    return 1;
  }

  const int kEvents = 20000;
  std::printf("== a project's own stepping action, engine not rebuilt ==\n\n");

  auto* rm = new G4RunManager;
  if (!scenes::Install("B1", rm)) {
    std::printf("FAIL: could not install scene B1\n");
    return 1;
  }
  rm->SetRandomSeed(0xF00Du);
  // A batch just big enough for this run. The default is 1048576, which sizes the track
  // buffers at 17.8 million slots - 4.1 GB once the G4Track block is on them, which does not
  // fit an 8 GB card with a desktop running. A test has no reason to ask for it.
  rm->SetBatchSize(kEvents + 1024);
  rm->Initialize();

  // One batch, so that the event index the action sees identifies an event uniquely.
  if (kEvents > rm->GetEngine().batch()) {
    std::printf("FAIL: %d events exceeds the batch size %d\n", kEvents,
                rm->GetEngine().batch());
    return 1;
  }

  // The host allocates; the action only ever holds pointers, exactly as a G4VPrimitiveScorer
  // is handed its hits collection rather than allocating one on the device.
  G4double *d_w = nullptr, *d_p = nullptr, *d_sec = nullptr;
  if (cudaMalloc(&d_w, sizeof(G4double) * kEvents) != cudaSuccess
      || cudaMalloc(&d_p, sizeof(G4double) * kEvents) != cudaSuccess
      || cudaMalloc(&d_sec, sizeof(G4double) * 3) != cudaSuccess) {
    std::printf("FAIL: cudaMalloc\n");
    return 1;
  }
  cudaMemset(d_w, 0, sizeof(G4double) * kEvents);
  cudaMemset(d_p, 0, sizeof(G4double) * kEvents);
  cudaMemset(d_sec, 0, sizeof(G4double) * 3);

  rm->SetStepHook(QualityFactorScoring(d_w, d_p, d_sec, kEvents, /*slot=*/0));
  rm->BeamOn(kEvents);

  std::vector<G4double> w(kEvents), p(kEvents);
  cudaMemcpy(w.data(), d_w, sizeof(G4double) * kEvents, cudaMemcpyDeviceToHost);
  cudaMemcpy(p.data(), d_p, sizeof(G4double) * kEvents, cudaMemcpyDeviceToHost);
  G4double sec[3] = {0, 0, 0};
  cudaMemcpy(sec, d_sec, sizeof sec, cudaMemcpyDeviceToHost);
  cudaFree(d_w);
  cudaFree(d_p);
  cudaFree(d_sec);

  int fails = 0;

  // ---- 1. the action ran at all, and saw the same energy the scorer did.
  //
  // The scorer counted this independently, with atomicAdd on the device as the steps happened.
  // The action's plain sum is over the same steps by a different route, so agreeing is
  // evidence that a project-defined hook is wired in exactly as the built-in one is.
  const auto& run = *rm->GetCurrentRun();
  double got = 0;
  for (double e : p) { got += e; }
  const double want = run.score_sum.empty() ? 0.0 : run.score_sum[0];
  const double rel = (want != 0) ? std::fabs(got - want) / std::fabs(want) : 1.0;
  std::printf("  energy   action %.15g   scorer %.15g   rel %.2e\n", got, want, rel);
  if (!(rel <= 1e-12)) {
    std::printf("  FAIL: a project-defined action did not see every step (rel %.2e > 1e-12)\n",
                rel);
    ++fails;
  }
  if (got <= 0) {
    std::printf("  FAIL: the action deposited nothing - it is not being called\n");
    ++fails;
  }

  // ---- 2. the quality factor is in range and actually varies.
  //
  // Q was defined as 1, 5 or 20, so every event's dose-averaged Q must lie in [1, 20]. A run
  // where every event came out at exactly 1.0 would mean s.let() was returning zero - which is
  // what a step length of zero, or a chord mistaken for a path length, would look like.
  double qmin = 1e300, qmax = -1e300, qsum = 0;
  int scored = 0;
  for (int i = 0; i < kEvents; ++i) {
    if (p[i] <= 0) { continue; }
    const double q = w[i] / p[i];
    if (q < qmin) { qmin = q; }
    if (q > qmax) { qmax = q; }
    qsum += q;
    ++scored;
    if (!(q >= 1.0 - 1e-12 && q <= 20.0 + 1e-12)) {
      if (fails < 5) { std::printf("  FAIL: event %d has Q = %g, outside [1, 20]\n", i, q); }
      ++fails;
    }
  }
  if (scored == 0) {
    std::printf("  FAIL: no event scored anything\n");
    ++fails;
  } else {
    std::printf("  Qbar     mean %.4f over %d scored events, range [%.4f, %.4f]\n",
                qsum / scored, scored, qmin, qmax);
    if (qmax - qmin < 1e-9) {
      std::printf("  FAIL: every event has the same Q - the LET is not varying, which means\n"
                  "        step lengths are not reaching the action\n");
      ++fails;
    }
  }

  // ---- 3. the secondary chain is exactly what the step says it is.
  //
  // GetNumberOfSecondariesInCurrentStep() counts what the step handed to the emitter;
  // GetSecondaryInCurrentStep() walks a chain built one link at a time as they were pushed.
  // The two are computed by different means, so their agreeing is evidence rather than
  // tautology - and it is the check that no per-step cap is quietly truncating the chain,
  // because there is no cap to truncate it with.
  std::printf("  secondaries walked %.0f, reported %.0f\n", sec[0], sec[1]);
  if (sec[1] <= 0) {
    std::printf("  FAIL: no secondaries at all - a 6 MeV gamma shower makes plenty\n");
    ++fails;
  }
  if (sec[0] != sec[1]) {
    std::printf("  FAIL: walked %.0f secondaries but the steps reported %.0f. The chain is\n"
                "        losing or inventing links.\n", sec[0], sec[1]);
    ++fails;
  }
  if (sec[2] != 0) {
    std::printf("  FAIL: secondaries exceeded the energy their parents could liberate by %g MeV, so\n"
                "        the chain is reading the wrong track out of the buffer.\n", sec[2]);
    ++fails;
  }
  if (rm->GetLastRunStats().secondary_overflow != 0) {
    std::printf("  FAIL: %lld secondaries did not fit the arena, so some lists were short.\n",
                rm->GetLastRunStats().secondary_overflow);
    ++fails;
  }

  std::printf("\n%s (%d failures)\n", fails ? "FAILED" : "PASSED", fails);
  return fails ? 1 : 0;
}
