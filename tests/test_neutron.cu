// step_neutral, on the device, with hand-computed answers.
//
// WHY A GPU TEST AND NOT A HOST ONE
//
// step_neutral is `__device__`, like every other stepper, so a host translation unit cannot
// call it. `tests/test_step_hook.cu` is the pattern: a `__global__` that calls the thing under
// test exactly as the stepping kernel does, over an array of chosen inputs, with the answers
// worked out by hand rather than by a second implementation.
//
// WHY NOT A DOSE RUN INSTEAD
//
// A dose run says one number about a whole shower. What has to be checked here is three
// separate claims about one step, and two of them are about a track a shower does not produce:
//
//   1. A neutron with no cross section streams. It crosses the world in straight lines, its
//      clock advances by L/v with v from the PRE-step energy, it deposits nothing, and it
//      leaves with its kinetic energy intact. ref/b1hadron/neutron_nogeneral.mac is the same
//      claim measured against Geant4 - which gives exactly 0 pGy in B1's scoring volume - but a
//      dose of zero cannot distinguish "streamed out" from "was never stepped".
//   2. A neutron past 10 us is killed by G4NeutronGeneralProcess's time limit, and the kill
//      DEPOSITS NOTHING. That is the claim worth a test of its own: every other death in
//      stepper.cuh hands its energy to the volume it stood in, and the natural assumption -
//      that a killer deposits what it kills - would put energy into a phantom Geant4 puts
//      nowhere. See the transcription in step_neutral's header.
//   3. The cut is the NEUTRON's and not the pi0's. `G4NeutronKiller::IsApplicable` is
//      `GetParticleName() == "neutron"` and G4NeutronGeneralProcess exists only for the
//      neutron, so a pi0 of the same age keeps going.
//
// A 100 MeV neutron covers B1's 300 mm in 2.5 ns, so no shower this project runs can reach
// 10 us. Case 2 is unreachable except by choosing the input, which is exactly what a test is
// for.
//
// AND THE OTHER TWO DISPOSITIONS, FOR THE SAME REASON
//
// The second half of this file is about `BufferEmitter::push` rather than about the neutron: a
// neutrino is created, its energy booked as carried out of the event and no track made; a
// species this port cannot transport is counted by name and no track made. Both are unreachable
// today - nothing here decays and no cascade runs - so both are dead code until P4 or P8 lands,
// and dead code is code that has never been executed. `tests/test_species.cu` asserts that
// `species_disposition` puts each species in the right group, which is a claim about a switch on
// the host; what is checked here is what the DEVICE actually does with each answer, which is a
// different claim and the one the run report depends on.
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <vector>

#include "core/track_buffer.cuh"
#include "physics/stepper.cuh"

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
  const double d = (want != 0.0) ? std::fabs(got - want) / std::fabs(want)
                                 : std::fabs(got - want);
  if (!(d <= tol)) {
    std::printf("  FAIL: %s - got %.17g, want %.17g (rel %.3g > %.3g)\n", what, got, want, d,
                tol);
    ++g_fails;
  } else {
    std::printf("    %-44s %.17g  (rel %.2e)\n", what, got, d);
  }
}

/// What one step of step_neutral did, copied back.
struct Outcome {
  real_t edep;
  real_t true_length;
  real_t ekin_after;
  real_t time_after;
  real_t pos_z;
  int volume_after;
  int status;
  int process;
  int alive;
  int n_secondaries;
};

/// A do-nothing RNG. step_neutral draws exactly one uniform, for the interaction length, and
/// only when the cross section is positive - which it never is with a null table. A fixed
/// value rather than Philox so that a change in the number of draws shows up as a changed
/// answer here rather than being absorbed by a different stream position.
struct FixedRng {
  real_t v;
  __device__ real_t uniform() { return v; }
};

/// An emitter that records rather than stores. step_neutral emits nothing today; asserting the
/// count is what would notice if it started to.
struct CountingEmitter {
  int* pushes;
  unsigned int child_count = 0u;
  int last_secondary = -1;
  __device__ int push(ParticleType, const Vec3<real_t>&, real_t, int) {
    atomicAdd(pushes, 1);
    ++child_count;
    return -1;
  }
};

__global__ void RunNeutral(Scene<real_t> scene, const TrackState<real_t>* in, int n,
                           ParticleType type, const had::NeutronGeneralXs<real_t>* xs,
                           Outcome* out, int* pushes) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) { return; }
  TrackState<real_t> p = in[i];
  const real_t ekin_pre = p.ekin;
  StepReport<real_t> rep;
  FixedRng rng{real_t(0.5)};
  CountingEmitter em{pushes};
  real_t edep = 0;
  const bool alive = step_neutral(scene, p, type, xs, rng, em, edep, rep);
  // The clocks, exactly as run_step_neutral advances them: from the PRE-step energy.
  p.advance(rep.true_length, ekin_pre, particle_def<real_t>(type).mass);
  Outcome o{};
  o.edep = edep;
  o.true_length = rep.true_length;
  o.ekin_after = p.ekin;
  o.time_after = p.global_time;
  o.pos_z = p.pos.z;
  o.volume_after = p.volume;
  o.status = static_cast<int>(rep.status);
  o.process = static_cast<int>(rep.process);
  o.alive = alive ? 1 : 0;
  o.n_secondaries = static_cast<int>(em.child_count);
  out[i] = o;
}

/// Pushes one of each disposition through a real BufferEmitter and reports what came back.
///
/// One thread, because what is being checked is the decision and the ledgers, not a race. The
/// event ids are chosen so that the carried-away array is indexed rather than summed: both
/// neutrinos belong to event 2, so a bug that booked against event 0 or against a running total
/// would show as a zero in the slot that should hold 18 MeV.
__global__ void PushDispositions(TrackBuffer<real_t> pool, SecondaryArena arena,
                                 EmitterBooks books, int* slots, unsigned int* children) {
  StepReport<real_t> rep{};
  rep.process = ProcessId::fDecay;
  BufferEmitter<real_t> em{pool, Vec3<real_t>{1, 2, 3}, 0, 2, 999u, 0u, 0u,
                           real_t(0), real_t(1), arena, -1, &rep, books};
  const Vec3<real_t> d{0, 0, 1};
  // A proton first, as the control: a species that IS stepped must still get a slot.
  slots[0] = em.push(ParticleType::kProton, d, real_t(5), 2);
  slots[1] = em.push(ParticleType::kNeutrinoMu, d, real_t(7), 2);
  slots[2] = em.push(ParticleType::kAntiNeutrinoE, d, real_t(11), 2);
  slots[3] = em.push(ParticleType::kLambda, d, real_t(13), 2);
  *children = em.child_count;
}

}  // namespace

int main() {
  // THE DEVICE STACK, WHICH IS NOT OPTIONAL AND IS NOT THE DEFAULT.
  //
  // `TransportEngine::Upload` does this and the reason is not stated there, so it is stated
  // here: ptxas reports run_step_neutral at a 2176-byte stack frame - the geometry helpers
  // recurse through boolean solids, so ptxas cannot determine the size statically and the
  // frame is a real stack - and CUDA's default per-thread limit is 1 KB. A kernel that calls
  // step_neutral therefore overruns the stack, and what the runtime reports is
  // `cudaErrorIllegalAddress`, which reads exactly like an out-of-bounds pointer.
  //
  // It cost half an hour here, because it did not fire on the first launch: the first track
  // streamed correctly and the SECOND launch faulted, which pointed at the second track's
  // input rather than at the stack. A per-thread stack overrun is a plausible-looking pointer
  // bug, and the tell is that the kernel is one of the transport steppers - all of which need
  // this line, and none of which say so.
  cudaError_t lim = cudaDeviceSetLimit(cudaLimitStackSize, 16384);
  if (lim != cudaSuccess) {
    std::printf("FATAL: cudaDeviceSetLimit: %s\n", cudaGetErrorString(lim));
    return 2;
  }

  // ---------------------------------------------------------------- the geometry
  //
  // One water box, 400 mm on a side, as the world. Deliberately the simplest geometry that has
  // a boundary: what is being tested is the step, and a nested geometry would put
  // step_to_boundary's answer between the input and the assertion.
  //
  // score_index 0, so that if step_neutral ever did deposit, the deposit would be reported
  // rather than dropped by the `score_index >= 0` guard. A scored volume is what makes the
  // "deposits nothing" assertions mean something.
  const real_t kHalf = 200;
  geom::Volume<real_t> vols[1] = {
      {{geom::SolidType::kBox, {kHalf, kHalf, kHalf}},
       geom::make_translation<real_t>({0, 0, 0}), 0, data::kWater, /*score_index=*/0},
  };

  geom::Volume<real_t>* d_vols = nullptr;
  cudaMalloc(&d_vols, sizeof(vols));
  cudaMemcpy(d_vols, vols, sizeof(vols), cudaMemcpyHostToDevice);

  data::Material<real_t> mats[data::kNumMaterials];
  data::build_b1_materials<real_t>(mats);
  data::Material<real_t>* d_mats = nullptr;
  cudaMalloc(&d_mats, sizeof(mats));
  cudaMemcpy(d_mats, mats, sizeof(mats), cudaMemcpyHostToDevice);

  Scene<real_t> scene{};
  scene.geometry = geom::Geometry<real_t>{d_vols, 1, 0};
  scene.materials = d_mats;
  scene.hadron_range = nullptr;  // the state every run is in until P8; step_neutral never reads it
  scene.range_cut = real_t(0.7);
  scene.scoring_volume = 0;

  // ---------------------------------------------------------------- the tracks
  //
  // Four, each chosen for one claim. All start at the centre heading +z, so the distance to the
  // boundary is exactly kHalf and the expected step is kHalf + kPushDistance.
  const real_t kEkin = 100;  // MeV
  auto make = [&](ParticleType t, real_t ekin, real_t t0) {
    TrackState<real_t> p{};
    p.species = t;
    p.pos = Vec3<real_t>{0, 0, 0};
    p.dir = Vec3<real_t>{0, 0, 1};
    p.ekin = ekin;
    p.volume = 0;
    p.event = 0;
    p.global_time = t0;
    p.weight = 1;
    p.rng_key = 12345u;
    p.step = 0u;
    return p;
  };
  std::vector<TrackState<real_t>> tracks = {
      make(ParticleType::kNeutron, kEkin, 0),                            // 0: streams
      make(ParticleType::kNeutron, kEkin, had::kNeutronTimeLimit<real_t>()),  // 1: exactly at 10 us
      make(ParticleType::kNeutron, kEkin, real_t(9999.9)),               // 2: a hair under 10 us
      make(ParticleType::kPiZero, kEkin, real_t(2e6)),                   // 3: a very old pi0
  };
  TrackState<real_t>* d_in = nullptr;
  cudaMalloc(&d_in, sizeof(TrackState<real_t>) * tracks.size());
  cudaMemcpy(d_in, tracks.data(), sizeof(TrackState<real_t>) * tracks.size(),
             cudaMemcpyHostToDevice);
  Outcome* d_out = nullptr;
  cudaMalloc(&d_out, sizeof(Outcome) * tracks.size());
  int* d_pushes = nullptr;
  cudaMalloc(&d_pushes, sizeof(int));
  cudaMemset(d_pushes, 0, sizeof(int));

  std::vector<Outcome> got(tracks.size());
  // Three of the four are neutrons and one is a pi0, and the SPECIES IS A TEMPLATE PARAMETER of
  // the real kernel - so they are launched as two kernels, as the engine launches them, rather
  // than by passing the species per track. That is the arrangement being tested: a pi0 that
  // reached the neutron's kernel would get the neutron's time cut.
  RunNeutral<<<1, 3>>>(scene, d_in, 3, ParticleType::kNeutron, nullptr, d_out, d_pushes);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::printf("FATAL (neutron launch): %s\n", cudaGetErrorString(err));
    return 2;
  }
  RunNeutral<<<1, 1>>>(scene, d_in + 3, 1, ParticleType::kPiZero, nullptr, d_out + 3, d_pushes);
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::printf("FATAL (pi0 launch): %s\n", cudaGetErrorString(err));
    return 2;
  }
  cudaMemcpy(got.data(), d_out, sizeof(Outcome) * tracks.size(), cudaMemcpyDeviceToHost);
  int pushes = -1;
  cudaMemcpy(&pushes, d_pushes, sizeof(int), cudaMemcpyDeviceToHost);

  // ---------------------------------------------------------------- 1. streaming
  //
  // The expected step is the distance to the boundary plus the navigator's push, and the
  // expected time is that length over the neutron's speed at 100 MeV. beta from
  // dynamic_particle_beta, so this checks TrackState::advance's use of the pre-step energy and
  // the mass row, not a hand-typed velocity: t = L / (beta c), with c in mm/ns.
  const real_t mass = particle_def<real_t>(ParticleType::kNeutron).mass;
  const real_t beta = dynamic_particle_beta<real_t>(kEkin, mass);
  const real_t len = kHalf + geom::kPushDistance<real_t>();
  const real_t want_time = len / (beta * units::c_light<real_t>());
  std::printf("== a neutron with no cross section streams to the world boundary ==\n");
  std::printf("    beta %.9g at 100 MeV, so %g mm takes %.9g ns\n", beta, len, want_time);
  CheckClose(got[0].true_length, len, 1e-14, "step length = boundary + push");
  CheckClose(got[0].time_after, want_time, 1e-14, "global time = L / (beta c)");
  CheckClose(got[0].ekin_after, kEkin, 0.0, "kinetic energy unchanged, exactly");
  CheckClose(got[0].edep, 0.0, 0.0, "deposits nothing, exactly");
  Check(got[0].volume_after == geom::kOutsideWorld, "left the world");
  Check(got[0].alive == 0, "not requeued");
  Check(got[0].status == static_cast<int>(StepStatus::fGeomBoundary), "status fGeomBoundary");
  Check(got[0].process == static_cast<int>(ProcessId::fTransportation),
        "process fTransportation");
  Check(got[0].n_secondaries == 0, "emitted no secondary");

  // ---------------------------------------------------------------- 2. the time cut
  //
  // `>=`, not `>`: G4NeutronGeneralProcess::PostStepGetPhysicalInteractionLength is
  //     if(track.GetGlobalTime() >= fTimeLimit) { fLambda = 0.0; return 0.0; }
  // so a neutron exactly AT 10 us is killed and one a hair under is not. Both are here,
  // because a `>` would pass the first assertion below on the second track and only the pair
  // pins the boundary.
  std::printf("== a neutron at or past 10 us is killed, and deposits nothing ==\n");
  Check(got[1].status == static_cast<int>(StepStatus::fStopAndKill),
        "at exactly 10 us: fStopAndKill");
  Check(got[1].process == static_cast<int>(ProcessId::fNeutronKiller),
        "at exactly 10 us: process fNeutronKiller");
  CheckClose(got[1].edep, 0.0, 0.0, "the kill deposits nothing, exactly");
  CheckClose(got[1].true_length, 0.0, 0.0, "and takes no step");
  Check(got[1].alive == 0, "at exactly 10 us: not requeued");
  CheckClose(got[1].pos_z, 0.0, 0.0, "and does not move");
  // The energy is still on the track: run_step_neutral books it from the pre-step energy, and
  // there would be nothing to book if the stepper had zeroed it.
  CheckClose(got[1].ekin_after, kEkin, 0.0, "its energy is intact and therefore bookable");

  std::printf("== a neutron a hair under 10 us is not ==\n");
  Check(got[2].status == static_cast<int>(StepStatus::fGeomBoundary),
        "at 9999.9 ns: still streaming");
  Check(got[2].process == static_cast<int>(ProcessId::fTransportation),
        "at 9999.9 ns: process fTransportation");
  CheckClose(got[2].true_length, len, 1e-14, "at 9999.9 ns: a full step");

  // ---------------------------------------------------------------- 3. the pi0 has no time cut
  std::printf("== a pi0 has no time cut, at any age ==\n");
  Check(got[3].status == static_cast<int>(StepStatus::fGeomBoundary),
        "a 2 ms pi0 still streams");
  Check(got[3].process == static_cast<int>(ProcessId::fTransportation),
        "a 2 ms pi0: process fTransportation");
  CheckClose(got[3].edep, 0.0, 0.0, "a 2 ms pi0 deposits nothing");

  // ---------------------------------------------------------------- 4. nothing was emitted
  std::printf("== the emitter was never called ==\n");
  Check(pushes == 0, "no secondary pushed by any of the four steps");

  // ---------------------------------------------------------------- 5. the other two dispositions
  //
  // A real BufferEmitter this time, with a real pool and real ledgers, pushing one of each
  // kind. See the note at the top of the file for why this cannot be reached by a run yet.
  std::printf("\n== what push() does with a species it will not step ==\n");
  {
    TrackBuffer<real_t> pool{};
    if (allocate_track_buffer<real_t>(pool, 64) != cudaSuccess) {
      std::printf("  FAIL: could not allocate a 64-slot pool\n");
      ++g_fails;
    } else {
      SecondaryArena arena{};
      arena.capacity = 64;
      cudaMalloc(&arena.entry, sizeof(unsigned int) * 64);
      cudaMalloc(&arena.prev, sizeof(int) * 64);
      cudaMalloc(&arena.cursor, sizeof(int));
      cudaMalloc(&arena.overflow, sizeof(int));
      cudaMemset(arena.cursor, 0, sizeof(int));
      cudaMemset(arena.overflow, 0, sizeof(int));

      const int kNT = static_cast<int>(ParticleType::kNumTypes);
      EmitterBooks books{};
      cudaMalloc(&books.carried_away, sizeof(double) * 4);
      cudaMalloc(&books.carried_by_type, sizeof(int) * kNT);
      cudaMalloc(&books.refused_by_type, sizeof(int) * kNT);
      cudaMemset(books.carried_away, 0, sizeof(double) * 4);
      cudaMemset(books.carried_by_type, 0, sizeof(int) * kNT);
      cudaMemset(books.refused_by_type, 0, sizeof(int) * kNT);

      int* d_slots = nullptr;
      unsigned int* d_children = nullptr;
      cudaMalloc(&d_slots, sizeof(int) * 4);
      cudaMalloc(&d_children, sizeof(unsigned int));
      PushDispositions<<<1, 1>>>(pool, arena, books, d_slots, d_children);
      err = cudaDeviceSynchronize();
      if (err != cudaSuccess) {
        std::printf("  FAIL: %s\n", cudaGetErrorString(err));
        ++g_fails;
      } else {
        int slots[4] = {0, 0, 0, 0};
        unsigned int children = 0;
        int live = 0;
        std::vector<double> away(4, 0.0);
        std::vector<int> by_type(kNT, 0), refused(kNT, 0);
        cudaMemcpy(slots, d_slots, sizeof(slots), cudaMemcpyDeviceToHost);
        cudaMemcpy(&children, d_children, sizeof(children), cudaMemcpyDeviceToHost);
        cudaMemcpy(&live, pool.count, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(away.data(), books.carried_away, sizeof(double) * 4,
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(by_type.data(), books.carried_by_type, sizeof(int) * kNT,
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(refused.data(), books.refused_by_type, sizeof(int) * kNT,
                   cudaMemcpyDeviceToHost);

        // The proton is the control: without it, "no track was made" could equally mean the
        // emitter is broken for everything.
        Check(slots[0] >= 0, "a proton got a pool slot");
        Check(slots[1] < 0, "a nu_mu got no pool slot");
        Check(slots[2] < 0, "an anti_nu_e got no pool slot");
        Check(slots[3] < 0, "a lambda got no pool slot");
        Check(live == 1, "exactly one track is in the pool");

        // A neutrino's energy is booked against the event it belongs to and nowhere else.
        CheckClose(away[2], 7.0 + 11.0, 0.0, "event 2 carried away 18 MeV, exactly");
        CheckClose(away[0], 0.0, 0.0, "event 0 carried away nothing");
        CheckClose(away[1], 0.0, 0.0, "event 1 carried away nothing");
        // Per FLAVOUR, which is the whole reason the six neutrinos are six species.
        Check(by_type[static_cast<int>(ParticleType::kNeutrinoMu)] == 1, "one nu_mu counted");
        Check(by_type[static_cast<int>(ParticleType::kAntiNeutrinoE)] == 1,
              "one anti_nu_e counted");
        Check(by_type[static_cast<int>(ParticleType::kNeutrinoE)] == 0,
              "no nu_e counted - the flavours are not pooled");
        // A refused species is counted under its own name and is NOT counted as carried away:
        // its energy did not leave the event, it was never carried at all.
        Check(refused[static_cast<int>(ParticleType::kLambda)] == 1, "one lambda refused");
        Check(by_type[static_cast<int>(ParticleType::kLambda)] == 0,
              "the lambda is not also booked as carried away");
        Check(refused[static_cast<int>(ParticleType::kNeutrinoMu)] == 0,
              "the nu_mu is not also booked as refused");
        // G4Step::GetNumberOfSecondariesInCurrentStep counts a neutrino, so push consumes a
        // child index for it - which is what keeps a sibling's RNG key the same whether or not
        // this transport materialises the neutrino. A refused species does not exist here at
        // all, so it consumes nothing. 1 proton + 2 neutrinos = 3.
        Check(children == 3u, "child_count is 3: the proton and both neutrinos, not the lambda");
        std::printf("    slots %d/%d/%d/%d, %d live, child_count %u, %g MeV carried from"
                    " event 2\n", slots[0], slots[1], slots[2], slots[3], live, children,
                    away[2]);
      }
      cudaFree(d_slots);
      cudaFree(d_children);
      cudaFree(books.carried_away);
      cudaFree(books.carried_by_type);
      cudaFree(books.refused_by_type);
      cudaFree(arena.entry);
      cudaFree(arena.prev);
      cudaFree(arena.cursor);
      cudaFree(arena.overflow);
    }
  }

  cudaFree(d_vols);
  cudaFree(d_mats);
  cudaFree(d_in);
  cudaFree(d_out);
  cudaFree(d_pushes);

  if (g_fails == 0) {
    std::printf("\nPASSED (0 failures)\n");
    return 0;
  }
  std::printf("\nFAILED (%d failures)\n", g_fails);
  return 1;
}
