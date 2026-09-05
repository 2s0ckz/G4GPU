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
| example B1 | the Geant4-API path agrees with Geant4 to 0.09 sigma |
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

| | events/s | 2M events |
|---|---|---|
| reference driver (`b1_gpu_sched.exe`) | 2.69e6 | 744 ms |
| example B1, via the general engine | 2.71e6 | 739 ms |

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
