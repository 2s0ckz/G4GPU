# Result

> The sections up to "Scheduler optimization" were rewritten after the geometry rewrite and
> the Geant4-shaped API landed. The optimization history below them is kept because the
> reasoning is still the reasoning, but its throughput figures predate the layer navigator and
> the general solid engine, which together cost about 8%. Current numbers are in the table at
> the end of the performance section.

## Example B1 against Geant4 11.1.1

Built entirely through the Geant4-shaped API in `src/g4` - `G4NistManager`, `G4Box`, `G4Cons`,
`G4Trd`, `G4LogicalVolume`, `G4PVPlacement`, `G4MultiFunctionalDetector`, and the full action
chain - from a project laid out the way a Geant4 project is:

```
examples/B1/
  exampleB1.cc
  include/  ActionInitialization.hh  DetectorConstruction.hh  EventAction.hh
            PrimaryGeneratorAction.hh  RunAction.hh  SteppingAction.hh
  src/      ActionInitialization.cc   DetectorConstruction.cc  EventAction.cc
            PrimaryGeneratorAction.cc RunAction.cc             SteppingAction.cc
  exampleB1.in  run1.mac  run2.mac  init_vis.mac  vis.mac  init.mac
  CMakeLists.txt  README  build.bat
```

```
> examples\B1\exampleB1.exe -n 2000000

--------------------End of Global Run-----------------------
 The run consists of 2000000 gamma of 6 MeV
 Cumulated dose per run, in scoring volume : 85.4998 nanoGy rms = 174.009 picoGy
------------------------------------------------------------

mass of scoring volume = 1.7316 kg, edep = 924.065 GeV
scaled to 10k events: 427.499 pGy  +/- 0.870044 (this run)
Geant4 11.1.1        : 427.385 pGy  +/- 0.87    ratio = 1.00027
difference           : 0.114111 pGy = 0.0927435 sigma

time 757.916 ms for 2000000 events = 2.63881e+06 events/s   (3.45367e+07 track-steps/s)
```

**0.09 sigma.** Both numbers come from 2,000,000 events and both carry their uncertainty. An
earlier version of this document compared a 10,000-event run (+/-2.9%) against a single
10,000-event Geant4 run and concluded "1.5% low, 0.52 sigma, statistically
indistinguishable". That conclusion was arithmetically fine and methodologically useless: at
that precision nothing below a 3% systematic is resolvable, and hours were then spent
bisecting a multiple-scattering model that was correct. The reference is now 2,000,000 events
from `ref/run/runb1.bat`, and both sides print their own uncertainty. See docs/RISK.md S1.

**Update, 2026-09-11.** The same 2,000,000-event run on main gives 425.847 +/- 0.870 pGy against
the same reference, 1.25 sigma, and had since before the hadronic port began - build_all.bat
prints it as its gate on every run. The run quoted at the top of this section is older than that
shift.

**Still 425.847 after P8e, and that is a measurement rather than an absence of one.** P8e split
the transport translation unit and turned on the two switches it had been holding shut - the
ion's `G4UrbanMscModel` and the electron's `extremesmallstep` branch. The gate was re-taken on
five seeds after each: 425.847 +/- 0.86768 pGy, **1.25195 sigma**, against 1.25138 before, with
426.195 / 426.917 / 427.489 / 427.288 on the other four. docs/RISK.md V65 and V66.

**425.860 since P14d, and this is the number that ships.** Two figures above it are now history
and are kept because the history is the point. P14c's rebuilt e+- tables moved the gate to
425.945 without this document being told, which is how it and the README came to disagree with
what `build_all.bat` printed; P14d then put the continuous loss into
`G4VEnergyLossProcess::AlongStepDoIt`'s shape - `length*dE/dx` first, the range inversion only
above `linLossLimit` - and moved it to **425.860 +/- 0.867731 pGy, 1.24145 sigma**. The other
four seeds are **426.191 / 426.908 / 427.485 / 427.287**, mean of five **426.7462** against
426.8172 before, so the change is **-0.0710 +/- 0.0045 pGy**: resolved, since every seed moves
the same way and the spread of the five differences is 0.010 pGy, and worth 0.0166% of the dose
and 0.082 of one run's standard error. The direction is second-order arithmetic - a restricted
electron stopping power rises as the track slows, so the inversion it replaces returned slightly
more than `L*dE/dx(E_pre)` - and the gate sits at a quarter of its 3-sigma limit either way.
docs/RISK.md V96 has the five-seed table, the depth-dose curves and the vacuum defect this
closed; V95 has the msc switch that shipped with it and did not move this gate at all.

## Geometry against Geant4's own G4VSolid

All 18 primitives and the three boolean operations, over 120,000 pseudo-random
(point, direction) pairs, comparing `Inside`, `DistanceToIn(p,v)` and `DistanceToOut(p,v)`:

| | |
|---|---|
| solids compared | 30 (18 primitives, wedge and hollow variants, 4 booleans) |
| rays | 120,000 |
| agreeing to ~1e-13 mm | 29 of 30 |
| disagreements | 2 rays, both `G4Hype`, both cases where **Geant4** contradicts its own `Inside()` |

The `G4Hype` disagreement is settled by asking Geant4 twice. `Inside()` defines the solid;
scanning it along the two disputed rays puts the entry where the port says it is, not where
`DistanceToIn` says. The oracle now dumps that independent determination for every ray, and
the test reports such rows in their own column rather than as failures. See docs/RISK.md G1.

## Example B1 with a proton and an alpha beam

The same example, the same geometry, the same scoring volume, three beams. Geant4's own B1
already ships a 210 MeV proton run - commented out at the bottom of `run1.mac` - and that is the
beam used here; `examples/B1/alpha.mac` matches it at 210 MeV per nucleon, so the alpha's range
in water (288 mm) is within a millimetre of the proton's (287 mm) and both Bragg peaks land in
the same place inside the bone trapezoid. A beam that stopped short of the scoring volume would
report zero from both codes and prove nothing.

`tools\compare_b1_beams.ps1` runs all of it. 200,000 events each, and **Geant4 forced to serial**
with `G4FORCE_RUN_MANAGER_TYPE=Serial` - the install is a multithreaded build, and left alone it
would put twenty cores against one GPU.

Three columns per beam, because B1's physics list is QBBC and this port has no hadronic physics:

| beam | port | Geant4 QBBC | Geant4 EM-only | port vs EM-only |
|---|---|---|---|---|
| gamma 6 MeV | 8599.0 pGy | 8546.1 pGy | - | 1.0062, 0.7 sigma |
| proton 210 MeV | 1,234,420 pGy | 998,595 pGy | 1,232,560 pGy | 1.0015, 0.6 sigma |
| alpha 840 MeV | 4,955,820 pGy | 3,286,330 pGy | 4,940,020 pGy | 1.0032, 1.4 sigma |

The EM-only column is the like-for-like comparison: the same QBBC run with `hadElastic`,
`protonInelastic`, `alphaInelastic`, `ionElastic` and `ionInelastic` switched off by UI command,
which leaves the tables and the cuts exactly as QBBC built them. The port agrees with it to
**0.15% for the proton and 0.32% for the alpha**, both inside one sigma of the Monte Carlo.

The gap between the two Geant4 columns is what the missing hadronic physics is worth here, and
it is not small: **19% of the proton dose and 33% of the alpha dose**. Inelastic reactions
remove primaries from the beam before they reach the trapezoid. Nothing about the EM agreement
above changes that; a proton calculation that needs an absolute dose needs those processes, and
this port does not have them.

The gamma row is a control rather than a result - its answer was already known - and it is in
the table for a reason. See docs/RISK.md V6.

### Runtime

Both codes measure their own event loop, so the middle column is like-for-like: the port's host
clock over primary generation plus every batch, and Geant4's own `G4Timer`, which runs from
`InitializeEventLoop` to `TerminateEventLoop` and so excludes physics-table building on both
sides. Wall clock is the whole process.

| beam | port loop | G4 QBBC loop | G4 EM-only loop | speedup vs EM-only |
|---|---|---|---|---|
| gamma | 0.159 s | 3.10 s | - | 19.5x (vs QBBC) |
| proton | 0.629 s | 27.0 s | 8.72 s | **13.9x** |
| alpha | 0.901 s | 92.6 s | 18.0 s | **20.0x** |

Quote the EM-only column. Against QBBC the port looks 43x and 103x faster, and a good part of
that is simply that it is not doing the hadronic work at all - a speedup measured against
physics you did not implement is not a speedup.

Wall clock tells a different story at these event counts: 26x and 70x against QBBC, but only
6.7x for gammas, because the port spends about half a second reading G4EMLOW and the
Seltzer-Berger tables before it starts and that is most of a short gamma run. The port's own
CUDA-event timing around the stepping kernels is within 5% of its event loop for every beam, so
almost none of the loop is host-side overhead.

One GPU against one CPU thread. A threaded Geant4 on this machine has twenty cores.

## Protons against Geant4 11.1.1: a Bragg peak in water

The charged-particle counterpart of the B1 comparison, and methodologically a step better than
it. `ref/proton/proton_depth.cc` is **one source file compiled twice** - once against the real
Geant4 11.1.1 install (`ref/proton/build.bat`), once against this port's Geant4-shaped headers
(`build_proton.bat`). The phantom, the beam, the binning, the scorers and the physics list are
one description rather than two that have to be kept in step, so the only thing that differs
between the two curves is the transport.

100 MeV protons into a water phantom binned in 0.5 mm slabs, each its own logical volume with
its own `G4MultiFunctionalDetector`. Physics list `G4EmStandardPhysics` and nothing else - not
QBBC - because this port has no hadronic physics and comparing against a list that does would
measure that absence rather than the stepper.

| | Geant4 11.1.1 | this port | |
|---|---|---|---|
| energy deposited | 100.0000% | 100.0000% | of the beam energy |
| plateau, 0-60 mm | - | +0.052% | per proton |
| R80 (distal 80%) | 77.798 mm | 77.783 mm | -0.014 mm, -0.018% |
| distal 80-20 width | 1.126 mm | 1.130 mm | +0.004 mm, +0.4% |

**That table is a different experiment from the gate `build_all.bat` runs today**, and the
paragraph above it says which: `G4EmStandardPhysics` and nothing else, taken when the premise
"this port has no hadronic physics" was true. It is QBBC on both sides since P8c, with only what
the port lacks inactivated on Geant4's, and the reference CSV was regenerated for it - which
moved Geant4's own R80 by 0.068 mm. The current numbers are in docs/PORTED.md 2.1.7 and in the
README's table: G4 77.730, port 77.742, +0.012 mm; plateau +0.158%; width 1.152 against 1.166.
P14d re-took them before and after putting the lepton's continuous loss into
`G4VEnergyLossProcess::AlongStepDoIt`'s shape and they are identical in every printed column,
which is the check that the change reached no hadron (docs/RISK.md V96).

The width is the interesting column. It is range straggling and nothing else, so it is a test of
`G4UniversalFluctuation` on its own: before that model was transcribed the port's protons all
stopped within one bin of each other and this number was three times too small, with R80 and the
plateau already right.

R80 was -0.232 mm for a long time, and `docs/RISK.md` V5 accumulated a list of things it was
not - the range table, the stopping power, the material construction, the fluctuation's mean
and variance, the delta rays, step-size bias, the MSC path conversion. Every item on that list
was right. What it could not contain was that the discrepancy was not physics at all:

> `G4VEnergyLossProcess` never evaluates a model during transport. It builds a dE/dx vector on
> a log grid - 100 eV to 100 TeV, 7 bins per decade, 85 points - splines it, and integrates
> *that* for a range. This port evaluated the models exactly on a finer grid: a better
> description of the physics and a worse description of Geant4.

Building Geant4's grid, with Geant4's spline and `G4LossTableBuilder::BuildRangeTable`'s own
100 midpoint sub-steps, moved all three numbers at once:

| | before | after |
|---|--:|--:|
| R80 | -0.232 mm | **-0.014 mm** |
| plateau | +0.324% | **+0.052%** |
| 80-20 width | +0.025 mm | **+0.004 mm** |

`docs/RISK.md` V7 has the finding and what it cost to miss; the amendment to V5 has what it
says about lists of eliminated causes.

`tools/compare_depth.ps1` scores these four separately rather than aggregating them, because a
stopping power 1% high and a range table 1% long cancel in the plateau and add in R80. The
Geant4 half is run once and checked in as `ref/oracle/proton_depth.csv` at 100,000 events, like
every other oracle file; the pipeline runs the port's half at 6,000 and compares per proton.

## What the pipeline checks

`build_all.bat` is the only entry point, and it runs all of this:

| stage | what it proves |
|---|---|
| build | every driver, test, the viewer, the GUI and example B1 compile |
| 32 tests | physics against `ref/oracle/*.csv`; geometry against `G4VSolid`; voxel and mesh traversal against their own invariants; CLHEP's constants against CLHEP; a user-built material's derived quantities against Geant4's |
| 2M-event dose | the reference driver still reproduces 427.6 pGy |
| example B1 | the Geant4-API path agrees with Geant4 to 1.25 sigma (2026-09-11; 0.09 sigma when this table was written) |
| B1's vis macros | `init_vis.mac` and `tsg_offscreen.mac` run clean and write seven pictures |
| mesh transport | the same geometry as an analytic solid and as a triangle mesh, to 0.00 sigma |
| every physics switch | turning any one of seven processes off changes the step count or the dose |
| viewer selftest | Win32, WGL, the CUDA render path and the font atlas all work |
| builder selftest | inserting solids - including a boolean, a polycone, a voxel grid, an imported mesh and a transparent shell over a higher layer - running them, and writing a project |
| same-layer overlap | a run with two volumes overlapping on one layer is **refused**, and the selftest proves the refusal fires |
| generated project | that project **compiles and runs**, from a directory that is not this one, and scores a dose that is not zero |

The last row exists because "Save gives you a compilable project" is worth nothing unless
something checks it, and generated code that does not compile is exactly the breakage no
other stage would notice.

## Performance, current

Two numbers, and only one of them answers "how many events per second, in real time".

| example B1, 6 MeV gammas | events/s | 2M events | what it measures |
|---|--:|--:|---|
| **event loop** | **1.91e6** | 1049 ms | host wall clock: primary generation + every batch |
| GPU kernels alone | 2.37e6 | 843 ms | CUDA events, from the first kernel |
| reference driver (`b1_gpu_sched.exe`) | 2.5e6 | 790 ms | no host per-event loop; seeds on the device |

**The event-loop number is the one to quote and the one to compare against Geant4**, whose
`G4Timer` brackets `InitializeEventLoop` to `TerminateEventLoop` - primary generation included,
physics-table building excluded. Same scope on both sides. The GPU number is measured with CUDA
events from the first kernel and so cannot see primary generation at all; it is a smaller number
measuring a smaller thing.

This document previously quoted 2.69e6 and 2.71e6 without saying which timer produced them.
They were the GPU-only figures, and stale. A performance number whose scope is not stated is
not a performance number.

**Where the rest goes, and where half of it went.**

The gap between the first two rows was a flat 198 ns per event - linear from 250k to 2M, so a
per-event constant rather than a fixed overhead. The obvious suspect was B1's generator, which
scans `G4LogicalVolume::Registry()` with a `dynamic_cast` and a string compare *every event* to
find the envelope, exactly as upstream B1 does. Caching it changed nothing measurable: the
registry holds four volumes.

The actual cost was allocation. A `G4PrimaryVertex` owns a `std::vector<G4PrimaryParticle>`, so
one primary per event cost four heap operations:

    G4PrimaryVertex vertex(...);        // malloc, building it on the stack
    vertex.SetPrimary(particle);
    event->AddPrimaryVertex(vertex);    // malloc, copying it in
    ...
    event.Reset(i);                     // two frees

`G4Event::NewPrimaryVertex` now returns a reused slot whose particle vector has been cleared
but not freed; `AddPrimaryVertex` assigns into a slot instead of pushing a new one; `Reset`
drops the count without destroying the storage. The gun builds its vertex in place.

| events | wall | gpu | gap | per event |
|--:|--:|--:|--:|--:|
| 250,000 | 180.8 ms | 143.3 ms | 37.5 ms | 150 ns |
| 1,000,000 | 522.6 ms | 419.7 ms | 102.9 ms | 103 ns |
| 2,000,000 | 1049.3 ms | 842.7 ms | 206.6 ms | 103 ns |

1.6e6 -> 1.91e6 events/s, dose bit-identical at 0.0230088 pGy and 0.0187 sigma, and nothing
removed from the API - `AddPrimaryVertex` still works for a generator that builds its own
vertex.

The remaining 103 ns is the virtual call, two `G4UniformRand()` draws, `sample_primary` and the
readback into the primary array. Closing it means sampling on the device, which is what this
port did *before* `GeneratePrimaries` was called per event - and which closed the set of
expressible generators, a Gaussian beam spot included. See the comment on
`G4VUserPrimaryGeneratorAction::GeneratePrimaries`. It could return as an opt-in fast path for
a plain `G4ParticleGun`, worth about 1.24x on light events; it is not worth reinstating as the
only path.

Bit-reproducible across batch sizes and thread counts: the RNG is counter-based and keyed on
`(rng_key, step)` carried by the track, never on its buffer slot.

The layer navigator and the general solid engine cost about 8% against the pre-rewrite 2.91e6
events/s. Two causes: `locate()` is an O(n) scan rather than a tree descent, and the solid
engine is mutually recursive, which blocks inlining and needs a real device call stack. The
scan is what will matter as scenes grow; a BVH is the fix.

## Scheduler optimization

`b1_gpu_sched.exe` replaces thread-per-event with **track-parallel SoA**: many events in
flight, tracks in structure-of-arrays buffers, one step per kernel launch, ping-pong buffers
that compact implicitly (survivors and new secondaries are appended to `out`, dead tracks
are simply never written, so no prefix scan is needed).

Two design choices carried most of the win:

- **One buffer per species** (gamma / electron / positron) rather than one buffer with a
  type field. Every thread in a kernel then runs identical physics - no type divergence
  within a warp - and the type need not be stored per track at all.
- **RNG needs no per-track storage.** A stream is keyed on (event, slot, iteration), which
  is unique because slots are unique within a buffer and the iteration disambiguates reuse
  across ping-pongs. Each track gets a fresh independent stream every step for the cost of
  one int, and the result stays reproducible.

| Step | events/s | vs 1 CPU core | notes |
|---|---|---|---|
| CPU, 1 core | 48,300 | 1x | |
| GPU thread-per-event | 229,000 | 4.7x | 108 registers, 8 KB private stack per thread |
| GPU track-parallel | 1,230,000 | 25x | 90 registers, no stack |
| + absolute range guard | 2,070,000 | 43x | steps/event 71.4 -> 38.9 |
| + 1M-event batch | **2,510,000** | **52x** | 98M track-steps/s |

**11x over the thread-per-event GPU version, 52x over one CPU core.** Against all 20 CPU
cores at perfect scaling (~966k events/s) it is **~2.6x** - the honest node-versus-node
number, and now favourable rather than a loss.

### What the levers actually did

**Absolute range guard (the biggest single win, 1.7x).** The FP64 build was spending ~32 of
its 71 steps per event grinding an electron's last few keV from 10 um of residual range down
to 0.01 um. Terminating at a physically motivated 10 um - two orders of magnitude below
Geant4's own 1 mm default production cut - removed that for no change in dose.

**Batch size, not block size.** 2.23M/s at batch 256k versus 2.52M/s at 1M, while block size
barely mattered (64/128/256 within 1% at the large batch). Bigger batches keep the drain tail
busy; a batch takes ~84 iterations to empty and the last iterations are nearly idle otherwise.

**FP32 was a dead end, informatively.** Registers dropped 90 -> 56, but throughput was
*identical* to FP64 (2.09M vs 2.07M). The kernel is bound by transcendentals and memory
traffic, not FP64 arithmetic, so the 1/32 FP64 rate of a consumer card never bites on this
workload. There is therefore no reason to accept FP32's precision loss here - though the
`real_t` templating that made this a one-line experiment is still worth having, since the
answer would differ for a geometry-heavy or table-heavy kernel.

### A robustness bug FP32 exposed

The first FP32 run was **86x slower** than FP64, not faster: 14.5k events/s, with
`steps/event` doubled and the drain loop hitting its iteration cap. Tracks were not running
slowly, they were failing to terminate - in single precision `energy_from_range` stops making
progress once the residual range gets small, so low-energy electrons never fell below the
tracking cut. Two guards fixed it, and both are correct in double precision too:

1. terminate when residual range falls below 10 um;
2. force the energy to zero if a step fails to strictly reduce it.

A wider boundary push for float (1e-3 mm, ~66 epsilon at the 180 mm world edge) removed a
second failure mode: tracks bouncing between two volumes because the push did not clear the
surface they had just crossed. Abandoned tracks went 1152 -> 17 per 2M events. The drain loop
is now capped at 1000 iterations and reports abandonment explicitly, so a future stall shows
up as a warning rather than as an 86x slowdown.

## Superseded: the "1.6% low" era

This document previously ended with a table concluding "1.6% low, 0.57 sigma, statistically
indistinguishable" from a comparison against a single 10,000-event Geant4 run. That 1.6% was
real and it was physics, not statistics - it was closed by fixing the water density-effect
parameters (Geant4 computes Cbar = 3.5801 analytically for water where the tabulated value is
3.5017), by completing the Urban MSC model, and by the rest of the work in
docs/EM_COMPLETION.md. The residual is now 0.09 sigma against a 2,000,000-event reference.

The lesson is recorded as docs/RISK.md S1, and it is the reason every comparison in this
project now prints both uncertainties: a discrepancy that is "inside the reference's
statistics" is not thereby absent, it is merely unmeasured.

## The step hook

`src/core/step_hook.cuh`. A device functor called once per real step of every track - the
general step-level customisation point, as against the per-event aggregate that
`G4UserSteppingAction` and `G4VPrimitiveScorer::Accept` are handed here.

### What it costs

Measured with the stock null-gated `StepTap` compiled into all three stepping kernels, against
the same build with the hook removed entirely.

| | baseline | with the hook |
|---|---|---|
| B1 dose10k, seed 1 / 2 / 3 (pGy) | 429.9790 / 426.1801 / 427.8784 | **identical to the last digit** |
| track-steps, seed 1 / 2 / 3 | 1299624 / 1296532 / 1298099 | **exactly equal** |
| 2M events, median of 6 interleaved runs | 852.0 ms | 853.2 ms (**+0.14%**) |
| 2M events, spread within a build | ±2.4% | ±2.4% |

+0.14% on the median sits an order of magnitude inside the run-to-run spread, so the honest
statement is that the cost is not measurable, not that it is 0.14%. The two builds were run
interleaved rather than in blocks - run in blocks, the *baseline* measured 2.3% slower, which
is the size of the drift this machine produces over a few minutes and a reminder of what a
non-interleaved A/B is worth.

The dose being bit-identical is the stronger result and the one that was actually in doubt: it
says the hook changed no random draw and no branch. The step-length out-parameter added to
`step_gamma`/`step_lepton`/`step_hadron` is written on every path and read by nothing but the
hook.

ptxas survived, which was not a given - it has died twice on `run_step_hadron` in this
project's life (RISK.md S12), and this change adds a template parameter to it. That is also why
the stock hook is gated at runtime by a null pointer rather than by a second instantiation: a
predicated load is a known cost, and doubling the instantiation count of that kernel is not.

### That it sees every step

`g4dose.exe -verify-step-hook`, run by `build_all.bat`. The scorer and the hook count the same
energy by different routes - the scorer with `atomicAdd` on the device as the steps happen, the
hook by a host sum over the records afterwards - so agreement is evidence rather than tautology:

```
20000 events, B1, seed 0xF00D, 43487 steps tapped in scorer 0, 0 dropped
  sum edep    hook 9088.66987121152   scorer 9088.66987121152    rel 0.00e+00
  sum edep^2  hook 38748.4215030902   scorer 38748.4215030902    rel 0.00e+00
```

Two sums, because they fail differently. The first fails if a step is missed, double-counted or
charged to the wrong volume. The second - the sum over events of the square of each event's own
total - fails if every step is present but attributed to the wrong event, which the first cannot
see, being the same total either way.

Both came out bit-identical rather than merely within the 1e-12 gate. That is luck about
summation order, not a guarantee, and the gate stays where it is.

**Falsified before being believed.** A check that has never failed is not known to be capable of
failing. Making `run_step_gamma` skip the steps that tracks die on - 0.038% of the energy, a
plausible-looking bug that leaves the output entirely reasonable - fails it at 3.8e-4 and
5.9e-4, nine orders above the gate:

```
  sum edep    hook 9085.19521732155   scorer 9088.66987121152    rel 3.82e-04
  sum edep^2  hook 38725.4361767218   scorer 38748.4215030902    rel 5.93e-04
```

### The true path length

`DeviceStep::length` is the stepper's own `step_len` - the true path - and not
`|pos_post - pos_pre|`. The two differ by however much MSC deflected the track inside the step,
always in the same direction, so a LET computed from the displacement is biased high. This is
what `G4Step::GetStepLength` returns and what the energy loss over that step was computed
against. It is reported by the stepper rather than reconstructed by the caller because the
caller cannot reconstruct it.

For the record, on B1 the hook gives a dose-averaged LET of 0.3786 MeV/mm with a peak of
6.1995 MeV/mm - a quantity no event aggregate can produce, having neither a step nor a length.

### Memory, which is the part that decides whether this scales

The mechanism stores nothing: `DeviceStep` is a view built in registers from state the kernel
already holds. What a step-level analysis costs is decided entirely by what the hook does, and
there are two kinds of hook.

`StepTally` reduces into a fixed device array - `atomicAdd` with the bin and the weight given as
device functors. Its footprint is chosen at setup: a 200-bin LET spectrum is 1.6 kB whether it
sees a thousand steps or a trillion, and a per-volume tally of a 1e5-volume geometry is 800 kB.
Nothing about that is new here - it is what `score` and `voxel_score` already are, and
`voxel_score` already runs at ~1e6 cells.

`StepTap` materialises steps, and is therefore the one thing in this file whose cost grows with
the amount of transport: 2.6e7 track-steps in a 2M-event B1 run at 104 bytes a record - the
size of `StepRecord<double>`, measured rather than counted by hand - is 2.7 GB, and B1 is four
volumes with one pencil beam. It is capped, it counts what it drops rather than
truncating silently, and it is documented as being for debugging and small runs. It is the stock
instantiation only because it is the one hook that answers an arbitrary question without a
rebuild.

The general point is the one worth keeping: a design whose memory scales with the number of
steps fails exactly on the runs that are worth doing, because step count is the quantity a Monte
Carlo exists to increase.

## One track pool, and what it cost

The five per-species buffers are one species-agnostic pool. A track carries its species; the
pool takes anything; five index arrays scatter the pool's slots into per-species runs so each
kernel still launches over its own species contiguously. The separation is between **storage**
and **dispatch**: warp coherence needs a species contiguous *at launch*, which is what the index
lists give, and never needed it separately *allocated*, which is what five buffers were.

Why it matters is not elegance. A capacity per species is a guess per species, and the guess was
2/4/0.5/1/1 slots per event for gamma/electron/positron/proton/alpha - which is a photon beam's
shape, written into the scheduler. A proton run overflows the electron buffer with the proton
buffer standing empty. One pool has no shape.

### The answer did not move

B1's dose is **bit-identical** before and after, which is a stronger result than it sounds: the
prediction was that it would need a tolerance, because the pool changes which slot a secondary
lands in and the RNG key is derived from the parent rather than the slot. Being wrong in that
direction is the evidence that the key really is slot-independent.

### The throughput

Example B1, 2M events, interleaved, five runs each way against the same program built at the
commit before this work, real time (the event-loop clock, `GeneratePrimaries` included):

```
before  1039.9  1040.5  1049.9  1052.3  1052.3 ms    median 1049.9   1.91e6 events/s
after   1164.7  1165.2  1172.4  1173.2  1174.4 ms    median 1172.4   1.71e6 events/s
```

**10.4%**, with no overlap between the two sets. 8.0% of it is the `G4Track` state, measured
separately and on its own (docs/G4STEP.md); the remaining ~2.5% is this - four bytes of species
on every track, and five index arrays sized at the whole pool each, which is 20 bytes a slot
against a 236-byte track.

The reference driver `b1_gpu_sched.exe` is unchanged at 2.40e6 events/s. It keeps its own three
buffers deliberately: it exists to be an independent implementation of the same physics, and a
cross-check that shares the thing being checked is not a cross-check.

### Nothing is dropped, and that is checked rather than argued

A pool sized to the memory available is a pool that can run out, and what happens then decides
whether the number at the end means anything. When the output buffer cannot hold what a track
would produce, the track is carried forward untouched and stepped in a later iteration -
`if (i >= n_step) { out.append(p); return; }` - and `RunStats::throttled` counts how often.

The claim is that this costs iterations and not energy, so the check is to vary the pool and
require the dose not to move at all:

```
dose10k by slots per event - 8.0: 429.8198, 4.0: 429.8198, 2.5: 429.8198
```

Identical, not close. Deferring a track changes the order work is done in and nothing else,
because every random stream is keyed by the track rather than by its slot. `build_all.bat` runs
all three through `tools/compare_pool.ps1`; the builder's model gives the same answer at 4, 8
and 16 slots per event, deferring 49152 track-steps at the smallest and none at the largest.

2.5 earns its place twice over. It is the only pool in this pipeline that is an ODD number, and
an odd pool used to fault - see docs/RISK.md V11.

## Two runs in one process, and the same two every time

Geant4's semantics, which a user relies on without stating: the program run twice gives the same
answer, and two runs inside one program give different answers within noise. Both are checked by
`tools/check_run_sequence.ps1`, on example B1 through `two_runs.mac`:

```
two runs in one process: 832.457 +/- 17.168 then 880.011 +/- 17.724 pGy
they differ by 47.554 pGy = 1.93 sigma, which must be noise and not physics
a second process replays both runs exactly
two runs of N summed 1,712.468 pGy against 1,712.470 for one run of 2N (rel 1.17E-006)
```

Four assertions, not one, because three of them can hold while the mechanism is wrong:

1. **The runs differ.** Without this a second run is not a second sample.
2. **They differ only within statistics.** 1.93 sigma against the rms each run reports. A run
   that differs by tens of sigma is a geometry that moved or a scorer that did not reset, not a
   new sample - and the threshold is in sigma rather than per cent precisely because the first
   version of this check used 5% against a spread of 2.9% and failed on its first real run.
3. **A fresh process replays both.** Exactly, not within noise.
4. **Two runs of N are one run of 2N.** This is the sharp one. An offset that jumped too far
   would still differ and still replay - it would just skip part of the stream, making each run
   a different sample from the one it would have been as part of a longer run. 1.2e-6 is the
   printed precision of the number being compared, not the agreement.

The mechanism is two streams that both carry on. CLHEP's engine on the host draws the primaries
and is a static constructed once per process. The device draws the showers from a counter-based
stream keyed on `(seed, position)`, where the position is `G4RunManager::stream_pos_` - zero at
construction, advanced by one per primary.

The device half was missing until this was checked, and the host half hid it: run two fired
different primaries into identical showers, and a generator drawing no random numbers repeated
its run bit for bit. docs/RISK.md V12 has the measurement that proved it and why B1 could not.

**What no seed can do** is return a process to the state it started in. The host engine begins on
CLHEP's default seed and the device stream on `G4RunManager`'s, which are unrelated numbers from
unrelated code; `/random/setSeeds n` puts both on `n`, which is a reproducible state and not that
one. Unifying them would make the program's start expressible and would move every number on this
page by a fraction of a sigma, since it changes which primaries B1 fires. Not done, and listed
here rather than left to be discovered.

## What is still on the table

See docs/ROADMAP.md, which lists what is unfinished in the order it matters. In brief: mesh
transport, radioactive decay sampling, six ion and low-energy-hadron physics items, a spatial
index for `locate()`, and a quantitative check that disabling a process changes the answer by
the right amount.

## Defects found by testing, not inspection

| # | Defect | How it surfaced |
|---|---|---|
| 1 | `namespace a::b` needs C++17 | compile error |
| 2 | local `em` variable shadowed the `em` namespace | compile error |
| 3 | `log(1+x)` catastrophic cancellation | reference formula returned -0.834 barn where +0.665 was required |
| 4 | **missing Sternheimer density correction** | dE/dx error drifting monotonically with energy (1.009 at 1 MeV to 1.127 at 10 MeV) - found from the *shape* of the error across a multi-energy scan |
| 5 | **secondary stack overflow at depth 16** | the overflow counter, 0.7% of events; silently lost secondaries and biased dose low. Depth is set by Compton multiplicity, not generation count |

Items 4 and 5 both moved the physics result. Neither would have been caught by reading the
code, and neither announced itself as an error - #4 looked like plausible numbers and #5
would have been a quiet 0.7% deficit.

## Validation summary (no Geant4 build required)

| Check | Result |
|---|---|
| Trd scoring mass | 1.7317 kg vs B1 1.7316 |
| Compton backscatter edge | 0.2451 vs analytic 0.2451 MeV |
| Compton XS vs exact Klein-Nishina | within 1.5%, Z=1..82, E=1..100 MeV |
| Water mu/rho at 6 MeV | 0.02752 vs NIST 0.0277 cm2/g |
| Collision dE/dx vs NIST ESTAR | within 1.1%, 0.1..10 MeV |
| CSDA range vs ESTAR | within 1.6% |
| Navigator | 45,691 crossings plus rotated and layered cases, 0 disagreements |
| Solids vs G4VSolid | 120,000 rays, 30 solids, agreement ~1e-13 mm |
| Voxel traversal | slab boundaries exact; a 64^3 uniform grid crossed in 1 step |
| Solid volumes | Trd 936.07 vs 936.00, Cons 175.90 vs 175.93 cm3 |
| Triangle mesh vs the same shape as a solid | 0.00 sigma, +0.0% steps |
| PSTAR / ASTAR stopping power | 0.0000% over 1685 and 495 points |
| ICRU 90 stopping power | 2e-16 over 1206 points, three materials, both particles |
| G4ionEffectiveCharge | exact for helium, 1e-14 for nine ions to uranium |
| Mott/Rutherford ratio | **exact** over 17136 points, eight elements, both charges |
| CLHEP's constants | exact for all thirteen, both of the port's sets |
| **B1 dose** | **0.09 sigma**, against a 2M-event reference |

## Approximations still in place

Every approximation this section used to list has been replaced: the Livermore photoelectric
and Rayleigh cross sections are read from G4EMLOW, pair production uses the full
Bethe-Heitler model with the Coulomb correction and LPM suppression, multiple scattering is
Urban rather than Highland, and bremsstrahlung is generated from the Seltzer-Berger tables
rather than treated as escaping. That is what closed the 1.6%.

What remains is listed in docs/ROADMAP.md. None of it affects example B1; it is ions,
low-energy hadrons, mesh transport and radioactive decay.

Energy-loss fluctuations are applied to leptons and hadrons alike - `G4UniversalFluctuation` on
every step of both - so this is no longer on the list. It is worth recording what closing it
took, because the shape of the answer was not what it looked like from this side.

`step_lepton` computes one combined continuous loss and splits it by a collision fraction
afterwards, where Geant4 appears to have a separate fluctuation model per loss process, and
matching that looked like it needed the loss split before the fluctuation rather than after.
It does not. `G4LossTableManager::BuildTables` sums every loss process for a particle into one
restricted DEDX table, hands that table to the single process it flags `SetIonisation(true)` -
`G4eIonisation` for a lepton - and every other loss process returns from `AlongStepDoIt` on its
first line, `if(!isIonisation)`. So Geant4 also applies one fluctuation to one combined number,
and the port's structure was already the right one.

The one real subtlety is ordering. Multiple scattering must see the *unfluctuated* mean:
`G4VMultipleScattering::AlongStepDoIt` runs before the ionisation process's and takes its
post-step energy from the range table, so it never sees the sampled loss. `step_lepton` keeps
`e_after_mean` for that reason.

For an electron in B1 the change is worth nothing measurable, and that is the expected result
rather than a disappointment - the model is mean-preserving and B1 reports an integral dose over
two million events. The agreement moved from 0.0154 to 0.0188 sigma, which is the run-to-run
scatter of a shifted RNG stream and not a physics change. For a proton it is not optional at
all: without it the Bragg peak's distal 80-20 width is 0.58 mm against Geant4's 1.13 mm.


## Build

```
build_all.bat            build everything, run the tests, the dose checks and both selftests
build_all.bat build      compile only
build_all.bat test       compile, then the tests only
```

Requires CUDA 11.6 with **MSVC 14.29 (VS 2019)** - CUDA 11.6 rejects MSVC 14.44 (VS 2022).
`setupenv.bat` finds it, and every build script goes through that one file because
`vcvars64.bat` cannot be run twice in the same environment: the second call fails, and the
failure surfaces later as an `nvcc fatal` naming a path nvcc built by walking out of its own
install directory, with nothing pointing at the nested script that caused it.
