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
// Usage
//   g4dose.exe                                 the B1 scene, 200000 events
//   g4dose.exe -scene B1mesh -n 1000000
//   g4dose.exe -seed 7                         an independent sample of the same scene
//   g4dose.exe -off compton -off rayleigh      run with those processes disabled
//   g4dose.exe -compare B1 B1mesh -n 500000
//   g4dose.exe -verify-processes -n 200000
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
               const ProcessFlags& flags) {
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
  const Result r = RunOnce(scene_name, n_events, seed, flags);
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
