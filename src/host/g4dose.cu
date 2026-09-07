// g4dose: run a registered scene headless and print - or check - what it scored.
//
// The viewer needs a window and an example needs its own main(), so neither can answer
// "what does this scene score, with these processes, with this seed" from a script. This can,
// and two checks in build_all.bat are built on it.
//
//   -compare A B    the same geometry expressed two ways must score the same. B1's scoring
//                   volume is a G4Trd; B1mesh's is a twelve-triangle mesh of the identical
//                   trapezoid. Same seed, same source, so the answers must agree inside the
//                   statistical error - and if the triangle pool never reached the device,
//                   they will not.
//
//   -verify-processes  switching a process off must change the answer. Until this existed,
//                   "everything on reproduces Geant4" was the only evidence the switches did
//                   anything at all, and RISK.md A3 is the story of eight switches that did
//                   nothing while looking fine. Note what it asserts: a *measurable* change,
//                   in the dose or in the step count, not a change of a particular size.
//                   Rayleigh scattering deposits no energy and multiple scattering conserves
//                   it, so demanding a big dose shift from those two would be demanding the
//                   wrong physics; both change the number of steps enormously.
//
//   -verify-step-hook  the StepHook must see every step exactly once, with each step charged
//                   to the right event. Both halves are checked against numbers the scorer
//                   produced independently, by a completely different route - the scorer sums
//                   on the device with atomicAdd as the steps happen, the hook is summed on
//                   the host afterwards from the steps it was handed:
//
//                     sum over steps of edep                     == score_sum[0]
//                     sum over events of (event's step sum)^2    == score_sum_sq[0]
//
//                   The first fails if a step is missed or counted twice or charged to the
//                   wrong volume. The second fails if the steps are all there but attributed
//                   to the wrong events, which the first cannot see - it is the same total
//                   either way. This matters because "the hook sees a real step" is the whole
//                   claim of core/step_hook.cuh, and a hook that quietly missed a class of
//                   steps would still look completely reasonable in its output.
//
// Usage
//   g4dose.exe                                 the B1 scene, 200000 events
//   g4dose.exe -scene B1mesh -n 1000000
//   g4dose.exe -seed 7                         an independent sample of the same scene
//   g4dose.exe -off compton -off rayleigh      run with those processes disabled
//   g4dose.exe -compare B1 B1mesh -n 500000
//   g4dose.exe -verify-processes -n 200000
//   g4dose.exe -verify-step-hook -n 20000    the per-step hook sees every step, exactly
//   g4dose.exe -batch 500000                   choose the batch instead of sizing it from memory
//   g4dose.exe -mem-frac 0.8                   let the track buffers take 80% of free memory
//   g4dose.exe -list                           the registered scenes and process names
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "g4/G4RunManager.hh"
#include "g4/G4SDManager.hh"
#include "g4/G4SystemOfUnits.hh"
#include "scenes/scene_registry.hh"

using namespace g4gpu;

namespace {

struct Switch {
  const char* name;
  bool ProcessFlags::*field;
};

/// Every switch the stepper reads, by the name this driver and the GUI use for it.
const Switch kSwitches[] = {
    {"photoelectric", &ProcessFlags::photoelectric},
    {"compton", &ProcessFlags::compton},
    {"rayleigh", &ProcessFlags::rayleigh},
    {"pair", &ProcessFlags::pair_production},
    {"bremsstrahlung", &ProcessFlags::bremsstrahlung},
    {"annihilation", &ProcessFlags::annihilation},
    {"msc", &ProcessFlags::multiple_scattering},
};

void PrintNames() {
  std::printf("scenes:    %s\n", scenes::Names().c_str());
  std::printf("processes:");
  for (const Switch& s : kSwitches) { std::printf(" %s", s.name); }
  std::printf("\n");
}

struct Result {
  double dose10k = 0;   ///< picogray, scaled to 10,000 events
  double sigma10k = 0;  ///< the run's own statistical uncertainty on that
  double edep = 0;      ///< MeV
  double mass = 0;      ///< kg
  long long steps = 0;
  double ms = 0;
  bool ok = false;
};

/// Builds the scene, runs it and reads scorer 0. A run manager is constructed per call: the
/// geometry, the material table and the scorer registry are all global state that a second
/// scene would otherwise add itself to rather than replace.
///
/// Which is exactly why -compare and -verify-processes each re-exec this program rather than
/// looping in one process. See RunChild.
Result RunOnce(const std::string& scene_name, int n_events, unsigned int seed,
               const ProcessFlags& flags, int batch = 0, double mem_frac = 0,
               double live = 0) {
  Result r;
  auto* rm = new G4RunManager;
  if (!scenes::Install(scene_name, rm)) {
    std::printf("\nFATAL: no scene named \"%s\".\n", scene_name.c_str());
    PrintNames();
    return r;
  }
  // After Install, because a scene's installer may set its own cut but never its processes.
  rm->SetProcesses(flags);
  rm->SetRandomSeed(seed);
  if (mem_frac > 0) { rm->SetMemoryFraction(mem_frac); }
  if (live > 0) { rm->GetEngine().SetLiveTracksPerEvent(live); }
  if (batch > 0) { rm->SetBatchSize(batch); }
  rm->Initialize();
  rm->BeamOn(n_events);

  const auto& run = *rm->GetCurrentRun();
  const auto& scorers = G4SDManager::GetSDMpointer()->Scorers();
  if (scorers.empty() || run.score_sum.empty()) {
    std::printf("\nFATAL: scene \"%s\" has no scorer; there is nothing to report.\n",
                scene_name.c_str());
    return r;
  }
  const double edep = run.score_sum[0];
  const double edep2 = run.score_sum_sq[0];
  double rms = edep2 - edep * edep / n_events;
  rms = (rms > 0) ? std::sqrt(rms) : 0.0;
  const double mass = rm->ScoredMass(0);
  const double scale = 10000.0 / n_events;
  r.edep = edep;
  r.mass = mass;
  r.dose10k = (mass > 0) ? edep * MeV / joule / mass * gray / picogray * scale : 0.0;
  r.sigma10k = (mass > 0) ? rms * MeV / joule / mass * gray / picogray * scale : 0.0;
  // A run that dropped tracks reported a dose that is too low, so it is not a result.
  if (rm->GetLastRunStats().overflow > 0) {
    std::printf("\nFATAL: %d tracks were dropped; this run is not a measurement.\n",
                rm->GetLastRunStats().overflow);
    return r;  // r.ok stays false
  }
  r.steps = rm->GetLastRunStats().track_steps;
  r.ms = rm->GetLastRunStats().milliseconds;
  r.ok = true;
  return r;
}

std::string g_exe;

/// Runs this program again, in a child process, and parses the one line it prints for a
/// machine reader.
///
/// A child process rather than a second RunOnce in this one, because a scene registers its
/// materials, its logical volumes and its scorers in global registries - G4NistManager's
/// table, G4LogicalVolume::Registry(), G4SDManager - and nothing here unregisters them. Two
/// scenes in one process would give the second one the first one's volumes as well as its
/// own. Fixing that properly means making those registries per-run-manager, which is a real
/// change to the API's shape; until then, a process boundary is the honest isolation.
bool RunChild(const std::string& args, Result& out) {
  const std::string cmd = "\"" + g_exe + "\" " + args + " -machine";
  FILE* p = _popen(cmd.c_str(), "r");
  if (p == nullptr) {
    std::printf("\nFATAL: could not run \"%s\"\n", cmd.c_str());
    return false;
  }
  char line[1024];
  bool found = false;
  while (std::fgets(line, sizeof line, p) != nullptr) {
    double dose = 0, sigma = 0, edep = 0, mass = 0, ms = 0;
    long long steps = 0;
    if (std::sscanf(line, "MACHINE dose10k %lf sigma %lf edep %lf mass %lf steps %lld ms %lf",
                    &dose, &sigma, &edep, &mass, &steps, &ms)
        == 6) {
      out.dose10k = dose;
      out.sigma10k = sigma;
      out.edep = edep;
      out.mass = mass;
      out.steps = steps;
      out.ms = ms;
      out.ok = true;
      found = true;
    } else if (std::strstr(line, "FATAL") != nullptr) {
      std::printf("  child: %s", line);
    }
  }
  const int rc = _pclose(p);
  if (rc != 0 || !found) {
    std::printf("\nFATAL: child run failed (exit %d): %s\n", rc, cmd.c_str());
    return false;
  }
  return true;
}

/// -verify-step-hook. See the note at the top of this file for what is being asserted.
///
/// Runs in this process rather than a child: it needs to reach into the engine and read a
/// device buffer back, which no line of printed output could carry.
bool VerifyStepHook(const std::string& scene_name, int n_events, unsigned int seed) {
  using Rec = StepRecord<G4double>;

  auto* rm = new G4RunManager;
  if (!scenes::Install(scene_name, rm)) {
    std::printf("\nFATAL: no scene named \"%s\".\n", scene_name.c_str());
    return false;
  }
  rm->SetRandomSeed(seed);
  rm->Initialize();

  // One batch, so that DeviceStep::event - an index within the batch - identifies an event
  // uniquely. Across batches those indices repeat and the per-event half of the check would
  // be comparing sums of unrelated events.
  const int batch = rm->GetEngine().batch();
  if (n_events > batch) {
    std::printf("\nFATAL: -verify-step-hook needs n <= the batch size (%d).\n", batch);
    return false;
  }

  // Sized from the measured step count: B1 runs about 13 track-steps per event across all
  // volumes, and the tap keeps only those in scorer 0, so this is roughly a 4x margin. The
  // run asserts overflow == 0 rather than trusting the estimate.
  const int capacity = n_events * 52 + 4096;
  Rec* d_rec = nullptr;
  int* d_ctl = nullptr;
  if (cudaMalloc(&d_rec, sizeof(Rec) * static_cast<size_t>(capacity)) != cudaSuccess
      || cudaMalloc(&d_ctl, sizeof(int) * 2) != cudaSuccess) {
    std::printf("\nFATAL: could not allocate the step tap (%d records, %.0f MB).\n", capacity,
                sizeof(Rec) * double(capacity) / 1048576.0);
    return false;
  }
  cudaMemset(d_ctl, 0, sizeof(int) * 2);

  StepTap<G4double> tap;
  tap.records = d_rec;
  tap.count = d_ctl;
  tap.overflow = d_ctl + 1;
  tap.capacity = capacity;
  tap.score_slot = 0;     // only the scoring volume, which is what score_sum[0] counts
  tap.only_deposits = false;  // boundary crossings with no deposit are steps too
  rm->SetStepHook(tap);  // the passthrough users of the Geant4-shaped API get

  rm->BeamOn(n_events);

  int ctl[2] = {0, 0};
  cudaMemcpy(ctl, d_ctl, sizeof(ctl), cudaMemcpyDeviceToHost);
  const int n_rec = ctl[0], n_over = ctl[1];
  std::vector<Rec> rec(n_rec > capacity ? capacity : n_rec);
  if (!rec.empty()) {
    cudaMemcpy(rec.data(), d_rec, sizeof(Rec) * rec.size(), cudaMemcpyDeviceToHost);
  }
  cudaFree(d_rec);
  cudaFree(d_ctl);

  const auto& run = *rm->GetCurrentRun();
  if (run.score_sum.empty()) {
    std::printf("\nFATAL: scene \"%s\" has no scorer.\n", scene_name.c_str());
    return false;
  }
  const double want_sum = run.score_sum[0];
  const double want_sq = run.score_sum_sq[0];

  std::printf("== the step hook sees every step ==\n");
  std::printf("  scene %s, %d events, seed 0x%X\n", scene_name.c_str(), n_events, seed);
  std::printf("  %d steps tapped in scorer 0, %d dropped\n", n_rec, n_over);

  int fails = 0;
  if (n_over != 0) {
    std::printf("  FAIL: the tap overflowed by %d - raise the capacity; the sums below are\n"
                "        a truncated sample and cannot be compared to anything\n", n_over);
    return false;
  }
  if (n_rec == 0) {
    std::printf("  FAIL: no steps were tapped at all. The hook is not being called.\n");
    return false;
  }

  // Per-event sums first; the two totals are then built from them.
  std::vector<double> per_event(n_events, 0.0);
  double worst_len = 0, worst_gain = 0;
  int bad_event = 0, bad_species = 0;
  for (const Rec& r : rec) {
    if (r.event < 0 || r.event >= n_events) {
      ++bad_event;
      continue;
    }
    per_event[r.event] += r.edep;
    if (r.length < 0) { worst_len = std::fmin(worst_len, r.length); }
    // Kinetic energy can only fall across a step. A track that leaves the world keeps its
    // energy and one that dies inside is reported at zero, so this holds for every path.
    const double gain = r.ekin_post - r.ekin_pre;
    if (gain > worst_gain) { worst_gain = gain; }
    if (r.species < 0 || r.species > 32) { ++bad_species; }
  }

  double got_sum = 0, got_sq = 0;
  for (double e : per_event) {
    got_sum += e;
    got_sq += e * e;
  }

  // The two sides add the same numbers in different orders - the scorer with atomicAdd on the
  // device as the steps happen, this loop on the host afterwards - and floating-point addition
  // is not associative, so they agree to rounding rather than bit for bit. At 1e-12 relative
  // this still catches a single missed step: B1's smallest per-step deposit is far above
  // 1e-12 of the run total.
  const double kRel = 1e-12;
  const double d_sum = std::fabs(got_sum - want_sum) / (want_sum != 0 ? std::fabs(want_sum) : 1);
  const double d_sq = std::fabs(got_sq - want_sq) / (want_sq != 0 ? std::fabs(want_sq) : 1);

  std::printf("  sum edep      hook %.15g  scorer %.15g   rel %.2e\n", got_sum, want_sum, d_sum);
  std::printf("  sum edep^2    hook %.15g  scorer %.15g   rel %.2e\n", got_sq, want_sq, d_sq);

  if (d_sum > kRel) {
    std::printf("  FAIL: the hook and the scorer do not see the same energy (rel %.2e > %.0e).\n"
                "        A step is being missed, counted twice, or charged to the wrong volume.\n",
                d_sum, kRel);
    ++fails;
  }
  // Only worth diagnosing when the totals DID agree. If they did not, the per-event sums are
  // bound to differ too, and "charged to the wrong event" would be the wrong diagnosis for
  // what is really a missing step.
  if (d_sq > kRel && d_sum <= kRel) {
    std::printf("  FAIL: the totals agree but the per-event sums do not (rel %.2e > %.0e).\n"
                "        Every step is present and charged to the wrong event.\n", d_sq, kRel);
    ++fails;
  }
  if (bad_event != 0) {
    std::printf("  FAIL: %d steps carried an event index outside [0, %d).\n", bad_event, n_events);
    ++fails;
  }
  if (bad_species != 0) {
    std::printf("  FAIL: %d steps carried an unrecognised species.\n", bad_species);
    ++fails;
  }
  if (worst_len < 0) {
    std::printf("  FAIL: a step reported a negative path length (%g mm).\n", worst_len);
    ++fails;
  }
  if (worst_gain > 0) {
    std::printf("  FAIL: a step gained %g MeV of kinetic energy.\n", worst_gain);
    ++fails;
  }

  // ---- 3. every real step says what ended it.
  //
  // The point of this one is coverage rather than correctness. StepReport is filled in on many
  // separate paths through three steppers, and the failure mode of adding a field like that is
  // not a wrong value - it is a path nobody annotated, which then reports the default forever.
  // A default is exactly what an un-annotated path leaves behind, so requiring that no step
  // reports one turns "did I cover every branch" into something the machine answers.
  //
  // The histogram is printed rather than asserted against expected fractions: B1's mix is a
  // property of B1, and pinning it here would make this test fail for the wrong reason the
  // first time the geometry or the beam changed.
  {
    const char* kProc[] = {"none",    "transport", "compton", "photoelectric", "conversion",
                           "rayleigh", "ionisation", "brems",  "annihilation",  "msc",
                           "nuclearstopping", "belowcut"};
    const char* kStat[] = {"undefined", "geomboundary", "worldboundary",
                           "poststep",  "alongstep",    "stopandkill"};
    const int kNProc = static_cast<int>(sizeof kProc / sizeof kProc[0]);
    const int kNStat = static_cast<int>(sizeof kStat / sizeof kStat[0]);
    std::vector<long long> proc_n(kNProc, 0), stat_n(kNStat, 0);
    int undefined_proc = 0, undefined_stat = 0, out_of_range = 0, no_material = 0;
    for (const Rec& r : rec) {
      if (r.process < 0 || r.process >= kNProc || r.status < 0 || r.status >= kNStat) {
        ++out_of_range;
        continue;
      }
      ++proc_n[r.process];
      ++stat_n[r.status];
      if (r.process == 0) { ++undefined_proc; }
      if (r.status == 0) { ++undefined_stat; }
      if (r.material < 0) { ++no_material; }
    }
    std::printf("  processes: ");
    for (int i = 0; i < kNProc; ++i) {
      if (proc_n[i] > 0) { std::printf("%s %lld  ", kProc[i], proc_n[i]); }
    }
    std::printf("\n  status:    ");
    for (int i = 0; i < kNStat; ++i) {
      if (stat_n[i] > 0) { std::printf("%s %lld  ", kStat[i], stat_n[i]); }
    }
    std::printf("\n");
    if (out_of_range != 0) {
      std::printf("  FAIL: %d steps carried a status or process outside the enum.\n",
                  out_of_range);
      ++fails;
    }
    if (undefined_proc != 0) {
      std::printf("  FAIL: %d steps reported no defining process. Some path through the\n"
                  "        steppers sets a deposit but never says what ended the step.\n",
                  undefined_proc);
      ++fails;
    }
    if (undefined_stat != 0) {
      std::printf("  FAIL: %d steps reported no status, same cause as above.\n", undefined_stat);
      ++fails;
    }
    if (no_material != 0) {
      std::printf("  FAIL: %d steps reported no material, though they were inside a scored\n"
                  "        volume and so were certainly inside some material.\n", no_material);
      ++fails;
    }
  }

  // Not an assertion - a demonstration that the thing the hook exists for is now computable.
  // LET is the per-step quantity that no event aggregate can reconstruct, because the
  // aggregate has neither a step length nor a step.
  double dose_w_let = 0, dose_tot = 0, max_let = 0;
  long long moved = 0;
  for (const Rec& r : rec) {
    if (r.length <= 0) { continue; }
    ++moved;
    const double let = r.edep / r.length;
    dose_w_let += let * r.edep;
    dose_tot += r.edep;
    if (let > max_let) { max_let = let; }
  }
  if (dose_tot > 0) {
    std::printf("  (dose-averaged LET %.4f MeV/mm, peak %.4f, over %lld steps that moved)\n",
                dose_w_let / dose_tot, max_let, moved);
  }

  std::printf("\n%s (%d failures)\n", fails ? "FAILED" : "PASSED", fails);
  return fails == 0;
}

}  // namespace

int main(int argc, char** argv) {
  g_exe = argv[0];
  std::string scene_name = "B1";
  std::string compare_a, compare_b;
  int n_events = 200000;
  unsigned int seed = 0xF00Du;
  ProcessFlags flags{};
  std::vector<std::string> disabled;
  bool machine = false;
  bool verify_processes = false;
  bool verify_step_hook = false;
  int batch = 0;       // 0 = let the engine size it from device memory
  double mem_frac = 0; // 0 = leave the engine's default
  double live = 0;     // 0 = leave the engine's default live-tracks-per-event

  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "-scene") == 0 && i + 1 < argc) {
      scene_name = argv[++i];
    } else if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
      n_events = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "-seed") == 0 && i + 1 < argc) {
      seed = static_cast<unsigned int>(std::strtoul(argv[++i], nullptr, 0));
    } else if (std::strcmp(argv[i], "-machine") == 0) {
      machine = true;
    } else if (std::strcmp(argv[i], "-compare") == 0 && i + 2 < argc) {
      compare_a = argv[++i];
      compare_b = argv[++i];
    } else if (std::strcmp(argv[i], "-batch") == 0 && i + 1 < argc) {
      batch = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "-live") == 0 && i + 1 < argc) {
      live = std::atof(argv[++i]);
    } else if (std::strcmp(argv[i], "-mem-frac") == 0 && i + 1 < argc) {
      mem_frac = std::atof(argv[++i]);
    } else if (std::strcmp(argv[i], "-verify-step-hook") == 0) {
      verify_step_hook = true;
    } else if (std::strcmp(argv[i], "-verify-processes") == 0) {
      verify_processes = true;
    } else if (std::strcmp(argv[i], "-list") == 0) {
      PrintNames();
      return 0;
    } else if (std::strcmp(argv[i], "-off") == 0 && i + 1 < argc) {
      const std::string want = argv[++i];
      bool found = false;
      for (const Switch& s : kSwitches) {
        if (want == s.name) {
          flags.*(s.field) = false;
          disabled.push_back(want);
          found = true;
        }
      }
      if (!found) {
        std::printf("\nFATAL: no process called \"%s\".\n", want.c_str());
        PrintNames();
        return 2;
      }
    } else {
      std::printf("\nFATAL: unrecognised argument \"%s\".\n", argv[i]);
      return 2;
    }
  }

  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
    std::printf("\nFATAL: no CUDA device\n");
    return 2;
  }

  // ------------------------------------------------------------------ -compare
  if (!compare_a.empty()) {
    char args[512];
    std::snprintf(args, sizeof args, "-scene %s -n %d -seed %u", compare_a.c_str(), n_events,
                  seed);
    Result ra;
    if (!RunChild(args, ra)) { return 2; }
    std::snprintf(args, sizeof args, "-scene %s -n %d -seed %u", compare_b.c_str(), n_events,
                  seed);
    Result rb;
    if (!RunChild(args, rb)) { return 2; }

    const double combined = std::sqrt(ra.sigma10k * ra.sigma10k + rb.sigma10k * rb.sigma10k);
    const double diff = rb.dose10k - ra.dose10k;
    const double nsig = (combined > 0) ? std::fabs(diff) / combined : 0.0;
    std::printf("compare %s vs %s, %d events, seed 0x%X\n", compare_a.c_str(),
                compare_b.c_str(), n_events, seed);
    std::printf("  %-8s dose10k %.4f +/- %.4f pGy, mass %.6f kg, %lld steps, %.1f ms\n",
                compare_a.c_str(), ra.dose10k, ra.sigma10k, ra.mass, ra.steps, ra.ms);
    std::printf("  %-8s dose10k %.4f +/- %.4f pGy, mass %.6f kg, %lld steps, %.1f ms\n",
                compare_b.c_str(), rb.dose10k, rb.sigma10k, rb.mass, rb.steps, rb.ms);
    std::printf("  difference %+.4f pGy = %.2f sigma; steps %+.1f%%\n", diff, nsig,
                (ra.steps > 0) ? 100.0 * (rb.steps - ra.steps) / ra.steps : 0.0);

    // Three sigma on the dose, and the masses must agree outright: a mesh whose volume is
    // wrong gives the right dose per unit mass and the wrong dose.
    if (std::fabs(ra.mass - rb.mass) > 1e-9 * std::fabs(ra.mass)) {
      std::printf("\nFATAL: the two scenes disagree about the scoring volume's mass.\n");
      return 1;
    }
    if (nsig > 3.0) {
      std::printf("\nFATAL: the two scenes disagree by %.2f sigma.\n", nsig);
      return 1;
    }
    std::printf("  OK\n");
    return 0;
  }

  // --------------------------------------------------------- -verify-step-hook
  if (verify_step_hook) {
    return VerifyStepHook(scene_name, n_events, seed) ? 0 : 1;
  }

  // --------------------------------------------------------- -verify-processes
  if (verify_processes) {
    char args[512];
    std::snprintf(args, sizeof args, "-scene %s -n %d -seed %u", scene_name.c_str(), n_events,
                  seed);
    Result base;
    if (!RunChild(args, base)) { return 2; }
    std::printf("verify-processes: scene %s, %d events, seed 0x%X\n", scene_name.c_str(),
                n_events, seed);
    std::printf("  all on:            dose10k %.4f +/- %.4f pGy, %lld steps\n", base.dose10k,
                base.sigma10k, base.steps);

    int dead = 0;
    for (const Switch& s : kSwitches) {
      std::snprintf(args, sizeof args, "-scene %s -n %d -seed %u -off %s", scene_name.c_str(),
                    n_events, seed, s.name);
      Result r;
      if (!RunChild(args, r)) { return 2; }
      const double combined =
          std::sqrt(base.sigma10k * base.sigma10k + r.sigma10k * r.sigma10k);
      const double nsig =
          (combined > 0) ? std::fabs(r.dose10k - base.dose10k) / combined : 0.0;
      const long long dsteps = r.steps - base.steps;

      // The step count is the sensitive test, and it is exact. With the seed fixed, the
      // transport is deterministic: the same scene with the same switches takes the same
      // number of steps, to the last one. So *any* change in it proves the stepper read the
      // switch, and a switch it does not read changes it by exactly zero.
      //
      // That matters because the dose is not a sensitive test for every process. Rayleigh
      // scattering deposits no energy, so switching it off moves B1's dose by 0.02 pGy
      // against a 2.7 pGy statistical error - 0.0 sigma - while removing about ten thousand
      // steps. A threshold on the dose alone called that switch dead. It is not; the
      // threshold was.
      const bool alive = (dsteps != 0) || (nsig > 3.0);
      std::printf(
          "  %-14s off: dose10k %.4f pGy (%+.4f, %.1f sigma), steps %+lld (%+.2f%%)  %s\n",
          s.name, r.dose10k, r.dose10k - base.dose10k, nsig, dsteps,
          (base.steps > 0) ? 100.0 * static_cast<double>(dsteps) / base.steps : 0.0,
          alive ? "" : "<-- NO EFFECT");
      if (!alive) { ++dead; }
    }
    if (dead > 0) {
      std::printf(
          "\nFATAL: %d process switch(es) changed neither the dose nor the step count, by\n"
          "  exactly zero. Either the stepper does not read the switch - which is worse than\n"
          "  a missing switch, because the run looks configured and is not - or the process\n"
          "  never occurs in this scene, in which case check it in one where it does.\n",
          dead);
      return 1;
    }
    std::printf("  OK: every switch changes the result\n");
    return 0;
  }

  // ------------------------------------------------------------------ one run
  const Result r = RunOnce(scene_name, n_events, seed, flags, batch, mem_frac, live);
  if (!r.ok) { return 2; }

  if (machine) {
    // One line, fixed shape, for RunChild to parse.
    std::printf("MACHINE dose10k %.10g sigma %.10g edep %.10g mass %.10g steps %lld ms %.10g\n",
                r.dose10k, r.sigma10k, r.edep, r.mass, r.steps, r.ms);
    return 0;
  }

  std::printf("scene %s, %d events, seed 0x%X", scene_name.c_str(), n_events, seed);
  if (disabled.empty()) {
    std::printf(", all processes on\n");
  } else {
    std::printf(", off:");
    for (const std::string& d : disabled) { std::printf(" %s", d.c_str()); }
    std::printf("\n");
  }
  std::printf("scorer 0: edep %.6g MeV, mass %.6f kg\n", r.edep, r.mass);
  std::printf("dose10k %.4f pGy  sigma %.4f pGy\n", r.dose10k, r.sigma10k);
  std::printf("time %.1f ms, %.4g events/s, %lld track-steps\n", r.ms,
              n_events / (r.ms * 1e-3), r.steps);
  return 0;
}
