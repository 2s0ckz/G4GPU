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
// impl header included once, one explicit instantiation, linked against the same
// out/transport_run.obj every other program here links - and if the arrangement is wrong it
// fails at compile or link time rather than silently.
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

#include "g4/G4VUserDeviceSteppingAction.hh"

// ---------------------------------------------------------------- the project's own action
//
// In a real project this is include/QualityFactorScoring.hh. It is an ordinary class: base
// class, member data, a named method. The two differences from Geant4 are that the base takes
// the derived type (CRTP - so the call inlines instead of going through a vtable, which at
// 3e7 steps a second is worth the odd-looking declaration) and that the method is __device__.
class QualityFactorScoring : public G4VUserDeviceSteppingAction<QualityFactorScoring> {
 public:
  QualityFactorScoring() = default;
  QualityFactorScoring(G4double* weighted, G4double* plain, G4double* sec_walked,
                       int n_events, int slot)
      : weighted_(weighted), plain_(plain), sec_walked_(sec_walked), n_events_(n_events),
        slot_(slot) {}

  /// Called once per real step of every track.
  ///
  /// Written in Geant4's spellings throughout - GetTotalEnergyDeposit, GetStepLength,
  /// GetPreStepPoint()->GetKineticEnergy() - because the point of the exercise is that a
  /// stepping action ported from a Geant4 project reads the way it did there. The two
  /// differences are visible in the signature, not the body: the method is __device__, and the
  /// step arrives by reference rather than as a G4Step*.
  __device__ void UserSteppingAction(const G4DeviceStep& step) const {
    if (weighted_ == nullptr) { return; }
    if (step.GetScoreSlot() != slot_) { return; }
    if (step.GetTotalEnergyDeposit() == 0) { return; }
    if (step.GetEventID() < 0 || step.GetEventID() >= n_events_) { return; }

    // The quantity that needs a real step: energy deposited per unit of the path actually
    // travelled. A deliberately blocky Q(LET) so the expected answer can be checked by hand.
    const G4double edep = step.GetTotalEnergyDeposit();
    const G4double len = step.GetStepLength();
    const G4double let = (len > 0) ? edep / len : 0.0;
    const G4double q = (let < 1.0) ? 1.0 : (let < 10.0 ? 5.0 : 20.0);

    // Nothing below is used by the arithmetic; it is here so that the test fails to compile if
    // any of these stops being reachable the way a Geant4 stepping action reaches it.
    const G4double ke_in = step.GetPreStepPoint()->GetKineticEnergy();
    const G4double ke_out = step.GetPostStepPoint()->GetKineticEnergy();
    const G4StepStatus st = step.GetPostStepPoint()->GetStepStatus();
    const G4ProcessId pr = step.GetPostStepPoint()->GetProcessDefinedStep();
    const G4double beta = step.GetPreStepPoint()->GetBeta();
    const int mat = step.GetPreStepPoint()->GetMaterial();
    const unsigned int tid = step.GetTrack()->GetTrackID();
    (void)ke_in; (void)ke_out; (void)st; (void)pr; (void)beta; (void)mat; (void)tid;

    // The secondaries this step made, walked as real tracks. Two things are asserted on the
    // host afterwards from what this accumulates: that the chain is exactly as long as
    // GetNumberOfSecondariesInCurrentStep() says, and that every secondary carries less
    // energy than the step that made it had - which is the cheapest statement that would
    // fail if the chain were walking into the wrong buffer or the wrong slot.
    int walked = 0;
    G4double worst_excess = 0;
    for (auto it = step.GetSecondaryInCurrentStep().begin(); it.valid(); it.advance()) {
      const auto sec = it.get();
      ++walked;
      // The most energy a secondary can carry is the parent's TOTAL energy plus one
      // target electron's rest mass. Kinetic energy alone is the wrong bound and this test
      // found that out: annihilation turns rest mass into two 511 keV photons, so a
      // positron that has all but stopped produces secondaries far above its own kinetic
      // energy. The bound below is exactly tight for that case - T + m_e + m_e - and holds
      // for every other process here, where a secondary cannot exceed the parent's T.
      const G4double ceiling = ke_in + step.GetPreStepPoint()->GetMass()
                               + G4double(0.510998910);
      const G4double excess = sec.GetKineticEnergy() - ceiling;
      if (excess > worst_excess) { worst_excess = excess; }
    }
    atomicAdd(&sec_walked_[0], static_cast<G4double>(walked));
    atomicAdd(&sec_walked_[1],
              static_cast<G4double>(step.GetNumberOfSecondariesInCurrentStep()));
    if (worst_excess > 0) { atomicAdd(&sec_walked_[2], worst_excess); }

    // Two running sums per event. Their ratio is the dose-averaged Q. Bounded by the number of
    // events, not by the number of steps - which is the whole discipline of core/step_hook.cuh.
    atomicAdd(&weighted_[step.GetEventID()], q * edep);
    atomicAdd(&plain_[step.GetEventID()], edep);
  }

 private:
  G4double* weighted_ = nullptr;
  G4double* plain_ = nullptr;
  /// [0] secondaries walked, [1] secondaries reported, [2] total energy excess.
  G4double* sec_walked_ = nullptr;
  int n_events_ = 0;
  int slot_ = 0;
};

// ---------------------------------------------------------------- the project's engine
//
// Name the hook, then pull in the kernels and instantiate them for it. In a real project these
// four lines are the whole of the ceremony, and they sit in one .cu.
#define G4STEP_HOOK QualityFactorScoring
#include "g4/G4RunManager.hh"
#include "g4/G4SDManager.hh"
#include "g4/G4SystemOfUnits.hh"
#include "host/transport_run_impl.cuh"
#include "scenes/scene_registry.hh"

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
