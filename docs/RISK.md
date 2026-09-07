# Risk register

Every judgment call, approximation, and unverified assumption, so debugging is targeted
rather than a search. Ordered by expected impact on the B1 dose.

## Physics approximations

| # | Where | What | Expected impact on B1 dose |
|---|---|---|---|
| P1 | `physics/em/gamma_processes.cuh` | **Photoelectric effect is not implemented as a cross section.** Photons below `kPhotonAbsorbCut` (20 keV) are absorbed and deposited locally. Geant4 uses the Sandia parameterization, which needs a data table. Above 20 keV, PE absorption is simply missing, so photons Compton-scatter longer than they should and may escape the scoring volume. | Largest single approximation. Small at 6 MeV (PE is <1% of total there) but grows for degraded secondaries. Test by varying the cut. |
| P2 | `physics/em/gamma_processes.cuh` | **Rayleigh (coherent) scattering omitted.** Transfers no energy, only perturbs direction. Geant4/NIST water mu/rho at 6 MeV including coherent is 0.0277; we compute 0.02752 without it. | Sub-percent at 6 MeV. |
| P3 | pair production sampling (not yet written) | Planned first pass samples the e-/e+ energy split as uniform in available kinetic energy rather than from the Bethe-Heitler differential distribution. | Small: both particles range out within ~2 cm in bone and Shape2 is 6 cm thick, so the split barely moves the deposit. |
| P4 | Compton | Atomic binding neglected, matching `G4KleinNishinaCompton`. Geant4 itself uses this model in several physics lists. | None relative to Geant4 at these energies. |

## Numerical

| # | Where | What |
|---|---|---|
| N1 | everywhere `log(1+x)` appears | `log(1+x)` for small x loses precision to catastrophic cancellation. Found the hard way: the exact-KN reference in `test_gamma_xs.cu` returned -0.834 barn instead of the Thomson 0.665 barn at 1 eV. Fixed with `log1p`. **The Klein-Nishina sampler itself is safe** - its `-log(eps0)` has eps0 far from 1 at MeV energies - but this pattern recurs throughout Geant4 EM code and must be checked at every transcription site. |
| N2 | `real_t` = float builds | FP32 has now been exercised and works (dose agrees with FP64 to 0.1%), but it is not the default and is less validated. It exposed a real termination bug: `energy_from_range` stops making progress at small residual range in single precision, so leptons never fell below the tracking cut and the drain loop stalled - an **86x slowdown**, not a wrong answer. Both guards added for it (absolute 10 um range floor, and forcing energy to zero when a step fails to reduce it) are correct in double too. FP32 also needed a wider boundary push (1e-3 mm) to stop tracks bouncing between volumes. 17 tracks per 2M events still fail to terminate in FP32; the drain loop caps at 1000 iterations and reports abandonment. |

## Geometry / navigation

| # | Where | What |
|---|---|---|
| G1 | `geometry/navigator.cuh` | **Translations only, no rotations.** True for B1; not checked at runtime. A rotated placement would silently give wrong answers. |
| G2 | `geometry/navigator.cuh` | Daughters assumed non-overlapping, and `locate()` returns the first containing daughter it finds. B1 satisfies this. |
| G3 | `geometry/navigator.cuh` | `kPushDistance` is a fixed nudge across boundaries, now type-aware: 1e-7 mm in double, 1e-3 mm in float (float has only ~1.5e-5 mm of absolute precision at the 180 mm world edge, and too small a push left tracks bouncing between volumes). Adequate at B1 centimetre scales; a general navigator needs a surface-relative tolerance. |
| G4 | `geometry/solids.cuh` | `G4Cons` supports rmin=0 and full phi only. B1 uses exactly that. An inner radius or phi cut would be silently ignored. |

## Untested

| # | What |
|---|---|
| U1 | ~~No GPU kernel has run.~~ **Closed.** Three GPU drivers run: thread-per-event, track-parallel, and the interactive viewer. The `__host__ __device__` discipline held - the thread-per-event build produced results bit-identical to the CPU driver on the first attempt. |
| U2 | ~~Overflow path never triggered.~~ **Closed, and it caught a real bug.** The secondary stack at depth 16 overflowed on 0.7% of B1 events, silently losing secondaries and biasing dose low. Stack depth turns out to be set by Compton multiplicity, not generation count: a photon degrading from 6 MeV to the 20 keV cut stacks tens of electrons while it keeps going. |
| U3 | The viewer is exercised only by `-selftest` (120 automated frames). Real mouse/keyboard interaction, window resizing, and multi-monitor DPI have not been tried. |
| U4 | Only one geometry, one primary particle, one energy. Nothing here is tested against a second scene. |

## Reproducibility in the track-parallel scheduler (RESOLVED)

**Was a violation of convention 7 in docs/PLAN.md ("reproducible independent of scheduling
order"); now fixed and verified bit-identical. History kept below.**

The thread-per-event build (`b1_gpu.cu`) is bit-reproducible: its RNG stream is keyed on
(event id, track serial within the event), both deterministic, and it produced results
bit-identical to the CPU driver.

The track-parallel build keys the stream on (event id, **slot in the current buffer**,
iteration). The slot comes from an `atomicAdd` on the output buffer cursor, and atomic
ordering is not deterministic across runs. So identical inputs give slightly different
answers run to run:

| run | dose ratio at 2M events |
|---|---|
| a | 0.9821 |
| b | 0.9817 |
| c | 0.9801 |

The spread (~0.2%) matches the statistical error at 2M events, so it is physically
harmless - these are independent samples of the same distribution, not a bias. But it means
a result cannot be reproduced exactly, which is normally a hard requirement for a physics
tool and makes regression testing awkward (a change of 0.2% cannot be attributed).

### RESOLVED

Each track now carries a deterministic `rng_key` and `step`, and the stream is keyed on
those alone. Verified bit-identical to all 17 significant digits:

```
same config, 3 runs      923377.20819378435   923377.20819378435   923377.20819378435
batch 1048576 / 262144 / 65536      all      923377.20819378435
block 64 / 128 / 256                all      923377.20819378435
```

This is a stronger guarantee than originally scoped: the result is invariant not just run to
run but under **any** scheduling configuration, because nothing in the RNG path depends on
atomic ordering, batch size, or launch geometry. It depends only on the physics and the
global event ids.

Two subtleties found while implementing it:

1. `p.event` is the **batch-local** index (it indexes the per-event dose array). Including it
   in the RNG key made results depend on batch size - reproducible per configuration but not
   across them. `rng_key` is already globally unique, so `p.event` was removed from the key
   entirely. This is the difference between "reproducible" and "reproducible however you
   choose to batch", and only the second is actually useful.
2. The child index must come from a counter local to the emitter, which lives for exactly one
   step of one track. Anything shared would reintroduce an atomic and with it the
   nondeterminism.

Cost: 8 bytes per track (64 -> 72, +12.5% track memory) and **2.4% throughput**
(2.51M -> 2.45M events/s at 8M events). Physics unchanged: ratio 0.9814 at 8M events
against 0.9836 before, a 1.6 sigma difference between two independent samples, both well
inside the reference's own +/-2.9%.

### Fix as originally planned

Give each track a deterministic id instead of using its buffer slot. Add two fields to
`TrackState` / `TrackBuffer`:

- `rng_key` (uint32): for a primary, the event id; for a secondary, a cheap mix of
  (parent rng_key, parent step count, child index).
- `step` (uint32): incremented every step, so each step draws a fresh stream.

Then key on (event, rng_key, step), all deterministic. Costs 8 bytes per track - about 12%
more track memory - and one mix per secondary. No effect on the physics.

---

## S1. Reading signal into a statistically noisy comparison

**What happened.** Geant4's exampleB1 reports `rms` as the uncertainty on the accumulated
total dose, not the per-event spread. A 10,000-event run carries **+/-2.9%**. The reference
figure used throughout this project, 427.385 +/- 0.870 pGy, is a 2,000,000-event run scaled
down to 10k.

While transcribing Urban MSC I compared 10,000-event GPU runs against that reference and
treated a ~1% shift as a physics regression. It was noise. Several hours went into
bisecting the MSC pieces - angular width, the true/geometric detour, lateral displacement,
step-size convergence - each experiment returning a different value in the 420-424 range,
which I read as meaningful differences between configurations. They were independent draws
from a distribution with a 12 pGy standard deviation.

At 2,000,000 events the port sits at 428.372 +/- 0.872 against Geant4's 427.385 +/- 0.870:
+0.23%, or 0.80 sigma.

**Why it was hard to see.** Every individual comparison looked reasonable, and the values
were *stable across repeated runs of the same binary* (the RNG is deterministic), which made
them feel like measurements rather than samples. Changing the physics changes the random
stream, so each variant is effectively an independent sample - reproducibility within a
configuration says nothing about the uncertainty between configurations.

**The tell that was there all along.** `docs/RESULT.md` already recorded "B1 reports rms
12.4101 pGy on 434.324 pGy, so the published number carries +/-2.9% statistical
uncertainty." The information needed to avoid this was in the project's own notes.

**Fix.** Both drivers now print the run's own statistical uncertainty and warn when it is
too loose to compare against the reference. Any dose conclusion needs 2,000,000 events.

**Generalisation.** Before treating a difference as signal, state the uncertainty on *both*
sides. A precise reference does not make an imprecise measurement precise.

## M1. Urban MSC: what is transcribed and what is not

Transcribed from `G4UrbanMscModel` (11.1.1): the transport cross section with both 15x22
correction tables, the `fUseSafety` step limiter, the true<->geometric path conversion, the
angular distribution, `SimpleScattering`, and the correlated lateral displacement.

Not transcribed, and these are real gaps rather than judgement calls:

- **The positron correction in `ComputeTheta0`** (`posa`..`pose`). ~7% of leptons in B1.
- **`fUseDistanceToBoundary` / `fUseSafetyPlus`** step limiters - not what B1 selects, and
  they need per-track skin state the port does not carry.
- **`tlimitmin` frozen with `rangeinit`.** Recomputed each step here; it is a ~1e-5 mm floor.

Unlike every other process in the port there is no Geant4 table to diff against, so
`tests/test_msc.cu` instead asserts the construction's own internal invariant
`<cos> == xmeanth / max(1, qprob)`. That is a strong check - it constrains every intermediate
quantity at once - but it validates the transcription, not the physics.

## O1. Trusting an oracle that silently picks a different model

`G4EmCalculator` is the supported way to interrogate Geant4's EM tables, and it is what this
project used throughout. It selects a model **by energy**, and when the lookup misses it
returns zero rather than failing. Both behaviours produced false diagnoses here:

- **Bremsstrahlung above 1 GeV.** Compared through `G4EmCalculator`, the transcription
  looked 5% wrong in dE/dx near the 1 GeV Seltzer-Berger / relativistic boundary, converging
  to agreement over three decades. Hours went into looking for an energy-dependent error.
  There was none: the calculator was blending the two models near the boundary. Against
  `G4eBremsstrahlungRelModel` instantiated directly, the agreement is 0.0003%.

- **Antiprotons below 10 MeV.** The calculator returns `G4ICRU73QOModel` there, not
  `G4BetheBlochModel`, so the antiproton looked 18% wrong. Against the model class directly
  it was 0.5%, and the remaining 0.5% was the water density effect.

- **Annihilation and Coulomb scattering.** `FindEmModel` returns null for the
  `G4VEmProcess`-derived models, and `ComputeCrossSectionPerVolume` then returns 0. A dump
  full of zeros is easy to mistake for "the process is off in this physics list".

**Rule adopted.** When validating a specific model, instantiate that model class in the
oracle and call it. Use `G4EmCalculator` only for quantities that are genuinely process-level
(range, total dE/dx). `ref/dump/g4dump.cc` now does this for `G4eeToTwoGammaModel`,
`G4eBremsstrahlungRelModel`, `G4BetheBlochModel`, `G4EmCorrections` and `G4IonisParamMat`.

**Second rule.** A test that compares nothing must fail, not pass. The first version of
`tests/test_brems_rel.cu` reported PASSED on zero rows because the oracle did not extend
above 1 GeV.

## O2. Reading one line of a condition instead of the whole initialisation

`G4BetheBlochModel::SetupParameters` contains `isIon = (!isAlpha && q > 1.1)`, which states
plainly that alpha is not treated as an ion. That line only executes for a particle other
than the one the model was initialised with. `Initialise` sets `isIon = true` for any charge
above one unit, so for alpha it is true, and alpha takes `IonBarkasCorrection`.

The transcription was 1.6% off for alpha and correct for everything else. What localised it
was dumping the correction terms individually: shell, Barkas, Bloch, Mott and the effective
charge each matched Geant4 to better than 0.02%, which ruled out all of them and pointed at
how they were being combined.

**Generalisation.** A flag's value comes from every place that writes it, not from the most
readable one. When a term-by-term diff says every part is right but the total is wrong, the
error is in the assembly.

## O3. Comparing a quantity that has no significant digits left

The Wentzel scattering cut-off angles approach 1 as the energy rises. At 1 TeV, `1 - cos` is
of order 1e-15, and it is computed as a difference of momenta of order 1e12 - a subtraction
with no significant digits left in double precision, in Geant4 exactly as much as in this
port.

The first version of `tests/test_wentzel.cu` compared `cos` itself, absolutely, and reported
agreement to 2e-9. That looked like a pass. It was a 10^6 relative error on the quantity
that actually enters the cross section, and it hid a real 33% disagreement in `1 - cos`,
which in turn was hiding nothing at all - both numbers were noise.

The test now compares `1 - cos` relatively, and skips any point where the oracle's own
`1 - cos` falls below 1e-10, counting and reporting how many were skipped (3834 of 16100).
Over the 12266 that remain, every quantity agrees to 0.0003%.

**Generalisation.** Choose the comparison variable before choosing the tolerance. An
absolute tolerance on a quantity that tends to a constant will pass regardless of the model,
and a relative tolerance on a quantity at the precision floor will fail regardless of it.
Where a test cannot say anything, it should say so and count it, not quietly pass.

## O4. An oracle that returns a different answer every call

`G4ICRU49NuclearStoppingModel::ComputeDEDXPerVolume` multiplies its result by a Gaussian
deviate: `G4VEmModel::lossFlucFlag` defaults to true, and the model applies straggling inside
the dE/dx accessor rather than leaving it to the caller.

Diffed against those single samples, the transcription looked wrong by a few percent
everywhere and by 60% at one point - scatter that crossed 1 non-monotonically, which is the
signature of an interpolation-index error, so that is what was hunted first: the table
extraction was re-verified against the source, the bracketing loop re-read, the reduced
energy recomputed by hand. All of it was correct.

With `SetFluctuationFlag(false)` the agreement is 0.0017% over 1392 points.

**Generalisation.** Before treating a mismatch as a transcription error, check that the
reference is deterministic. Calling it twice with the same arguments is a one-line test and
would have settled this immediately.

## O5. Commented-out rows inside a data table

`G4BraggIonModel`'s 92 x 5 Ziegler table contains two commented-out alternative rows (Be and
Au from ICRU). A naive extraction picks their numbers up and shifts every element after them
by four positions - silently, because the total still looks like a plausible count.

This is the second time this exact failure has appeared in this project; the first was a
Rayleigh parameter table where inline comments shifted an awk index. Every extractor now
strips `//` comments before parsing and asserts the element count against the declared array
dimensions. That assertion is what caught it: 474 extracted where 460 were expected.

A related case in the same family: `G4ICRU73QOModel::L0` is declared `[67][2]` but its
initialiser has only 66 rows, so C++ zero-fills the last. That is *not* an extraction error
to be corrected - `GetL0` reaches the zero row for any normalised energy above 1000 and
extrapolates through it, so the port reproduces the zero row deliberately. Asserting "the
table must have as many rows as its declared size" would have been the wrong fix.

## S2. A build that reported success while broken, and physics that switched itself off

Two failures that compounded, both of the same shape: a step that could not have succeeded
reported success, and nothing downstream disagreed.

**The build.** `build_all.bat` ended in `|| exit /b 1` per command, but it was being invoked
through a pipeline that discarded the exit code. A background build task reported "exited
with code 0" while `stepper.cuh` had an unresolved `geom::compute_safety` - the include had
been lost when `physics/transport.cuh` was split into `physics/scene.cuh`. The stale output
file of an *earlier* run was then read as if it were this one's, and reported as "24 tests
pass, dose unchanged". None of that had happened.

**The data.** `G4LEDATA` on this machine points at a Geant4 10.7.3 install whose G4EMLOW 7.13
does have an `epics2017` directory - holding only pair-production data, no photoelectric and
no Rayleigh. The path resolver accepted it, both loaders failed, and both processes were
disabled with a `WARNING:` line. The run then completed and printed a dose of 425.759 pGy
against a reference of 427.385 - low by 0.43%, which reads as ordinary disagreement rather
than as two missing processes.

Three fixes, in increasing order of importance:

1. `g4_dataset_dir` checks for a sentinel *file inside the specific subdirectory* it needs
   (`epics2017/phot/pe-cs-1.dat`), not for the subdirectory itself, and searches every
   candidate root instead of committing to the first one that exists.
2. Missing data is `std::exit(2)` with the searched path, not a warning.
3. `build_all.bat` is one entry point that builds, runs all 24 tests, runs the 2M-event dose
   check, greps the output for `FATAL`, and propagates failure. Building is no longer treated
   as evidence that anything works.

**Generalisation.** The recurring pattern in this project - S1, O3, "a test that compared
nothing reported PASSED", and now this - is a success report that no evidence supports.
A green result is only worth what its failure mode is worth: if a check cannot fail, it is
not a check. Both fixes here are about making the failing path reachable and loud.

## G1. Geant4's own G4VSolid answers are not all correct

The 30-solid ray comparison found `G4Hype::DistanceToIn(p, v)` disagreeing with the port on 2
of 4000 rays. Both were rays *starting inside the inner hyperboloid* - in the hole - and
crossing outward into the material. Geant4 returned `kInfinity` for one and, for the other, a
crossing at z = -58 mm on a solid whose half-length is 45 mm.

Neither is a port defect. The adjudication is Geant4 against itself: `Inside()` is the
definition of the solid, and scanning `Inside()` along those two rays puts the entry at
t = 3.25 and t = 51.25, matching the port's 3.0954 and 51.1489 (the scan grid is 0.5 mm), not
Geant4's 1e30 and 21.23.

The oracle now dumps `solid_scan.csv`: for every ray, the first `Inside() == kInside` parameter,
found by a 0.5 mm scan and bisection. The test uses it only to adjudicate a disagreement, and
reports such rows in their own `g4_wrong` column rather than as failures.

**Generalisation.** An oracle is not automatically right, and "the reference says X" is not the
end of an investigation. Where a reference exposes two things that must agree - here a
predicate and an algorithm that implements motion through it - checking them against each other
turns "who is wrong" from an argument into a measurement. The cost was one extra dump; it
replaced what would otherwise have been either an unexplained 2-in-4000 failure carried
indefinitely, or a tolerance quietly widened until it passed.

## G2. A mutually recursive device function needs a device stack

The generic solid engine recurses: a boolean node asks its children for distances, and those
children may be boolean nodes. nvcc therefore cannot inline it and emits real calls with a real
stack. The default per-thread stack is 1 KB, and one frame of the distance routine can exhaust
that on its own.

The symptom is `an illegal memory access was encountered`, reported from whichever CUDA API
call happens to synchronise next - in this case a `cudaMemcpy` of a track count, 100 lines and
one kernel launch away from the actual fault. Nothing in the message mentions the stack.

`cudaDeviceSetLimit(cudaLimitStackSize, 16384)` in the drivers, with the reason recorded at the
call site. Worth remembering as a category: on the GPU, "illegal memory access" from a
synchronising call means "look at every kernel since the last sync", and stack overflow is one
of the causes that leaves no other trace.

## A1. A per-event CPU callback cannot exist at GPU rates

**Superseded in part - read the amendment at the end of this entry. The event actions ARE
called now. The conclusion below was too broad, and being wrong about it cost example B1 three
of its classes for a session.**

Geant4's user hooks are per event: `GeneratePrimaries(G4Event*)`, `BeginOfEventAction`,
`EndOfEventAction`. At 2.7 million events per second an empty virtual call, made three times
per event, is already a substantial fraction of the budget, and B1's actual
`GeneratePrimaries` does two `G4UniformRand` draws and a geometry lookup.

So `GeneratePrimaries` is called **once per run** here, and what it configures is a *source
description* that the seeding kernel samples per event. Both facts are stated in the class
comment at the point where a user would otherwise discover them by having their randomisation
silently apply to a whole run.

What makes this acceptable rather than a hole: every distribution Geant4's own
`G4GeneralParticleSource` offers is available as a setter on the gun - rectangular and
elliptical beam cross-sections, angular spread, isotropic, an isotropic shell, a distributed
volume source, an imported energy spectrum. B1's two random draws become one call to
`SetBeamCrossSectionRectangular`, which is both shorter and clearer than the original.

**The trap to avoid here** was to keep the per-event signature and quietly call it once,
leaving a user's `G4UniformRand` to freeze at one value for two million events. The dose would
have come out close enough to look right - a fixed beam position within the same rectangle is
still a plausible physics setup - and the beam would have been a pencil rather than a field.

### Amendment: the event actions cost 2.6%, and they are called

The paragraph above originally continued "the event actions are not called at all", and
`examples/B1` shipped without `EventAction`, `SteppingAction` or `ActionInitialization` on the
strength of it. That was wrong, and the reasoning was wrong in a specific way worth recording:
I priced the callback without looking at what it would have to compute.

The energy deposit was *already* per event on the device. `d_score_` is one row per scorer and
one column per event in the batch, and the host already copied the whole thing back and walked
it to accumulate the total and the sum of squares. Calling `BeginOfEventAction`, one aggregated
step, and `EndOfEventAction` inside that walk costs three host virtual calls per event and no
device work whatsoever. Measured on B1 at 2 million events: 2.71e6 events/s before, 2.64e6
after - **2.6%**, for the complete Geant4 action chain.

The estimate that said otherwise was of the wrong thing. Three virtual calls per event is not
"a substantial fraction of the budget" when the budget is 370 ns per event and a virtual call
is 2 ns; what would have been substantial is a per-*step* callback, at 13 steps per event, each
one needing a `G4Step` assembled from device memory. That distinction - per event versus per
step - is the whole of it, and I collapsed the two.

What genuinely cannot exist is the per-step hook, and the honest thing was not to refuse it but
to say precisely what is handed to it: one aggregate per event and per scored volume, carrying
that event's totals. B1's stepping action adds up what it is given, and the sum over an event of
the per-step deposits *is* the event's deposit, so the number is exact - the 0.09-sigma
agreement is now computed through that chain and is unchanged to the last digit. An action that
did something non-additive would not be so lucky, and `G4Step::IsAggregated()` exists so that
one can say so. The whole story is at the top of `src/g4/G4Step.hh`.

There is a second dividend, which is why this is filed under a mistake rather than a
refinement. Going through `EventAction` and `RunAction::AddEdep` means the run's rms is
computed the way Geant4's B1 computes it: the second moment of the per-event energy deposit
distribution. It comes out at 0.870 picogray per 10,000 events, which is Geant4's own figure
for the same run. That is an independent check on the physics, and a stronger one than the
mean: two transports can agree on an average while disagreeing about the distribution behind
it. A whole class of check was sitting behind an assumption I had not measured.

**Generalisation.** "Too expensive" is a measurement, not an opinion, and the cost of a
callback is the cost of the work it does - not the cost of the call. Before removing a feature
on performance grounds, write the cheap version and time it; the version that seemed impossible
was three lines inside a loop that already existed.

## A2. Batch size, not the abstraction, was the 1.7x

The generalised engine first measured 1280 ms against the reference driver's 744 ms for two
million events, which looked like the cost of going from four hard-coded volumes to a scanned
list, from one scorer to an indexed array, and from a hard-coded beam to a sampled source.

It was none of those. The engine defaulted to a batch of 100,000 events and the reference
driver to 1,048,576; ten times as many kernel launches, each with a tenth of the parallelism.
Matching the batch brought the engine to 731 ms - marginally *faster* than the driver it
generalises.

**Generalisation.** Before attributing a slowdown to a design change, check the knobs. The
plausible story (abstraction costs performance) was wrong, and it was wrong in the direction
that would have prompted undoing good structure to recover speed that was never lost.

## A3. A switch that looked like it did something

The model builder's physics panel had eight toggles - photoelectric, Compton, Rayleigh, pair
production, ionisation, bremsstrahlung, annihilation, multiple scattering. They were stored in
the model, serialised to the .model file, and written into the generated project's source. The
engine never read any of them. Turning Compton off and pressing Run produced exactly the same
dose, to the last digit.

Nothing about the UI said so. A toggle that visibly changes state and changes nothing is a
worse failure than an absent one: the user's next act is to trust a number they think they
configured.

Two fixes, and the second is the more interesting:

1. `ProcessFlags` on the Scene, checked in the stepper. All on is the default and is the
   configuration the 0.09-sigma agreement with Geant4 is measured in.
2. The ionisation toggle was **removed**, not wired up. Ionisation is the continuous energy
   loss along a step; without it a lepton has infinite range and never stops, so there is no
   setting of that switch that produces a number worth having. Offering it and honouring it
   would have been worse than offering it and ignoring it - the run would have looked like it
   worked.

**Generalisation.** When a control cannot be honoured meaningfully, delete the control. The
temptation is to wire it up because it is already drawn; the question to ask is what the user
would do with the result, and if the answer is "misinterpret it", the control is the bug.

## S3. Editing sources while a build runs

Two full pipeline runs were wasted because I edited build scripts and sources while the
pipeline was compiling them. The first failed with `'nsport_run.cu' is not recognized as an
internal or external command` - a batch file read halfway through being rewritten. The second
linked a viewer built from one revision of a header against a GUI built from another.

Neither failure looked like what it was. The first pointed at a filename that does not exist;
the second at a duplicate declaration in a file I had already fixed.

The build takes several minutes, and the temptation during it is to keep working. The rule is
simply not to: a build is a measurement of a particular state of the tree, and changing the
tree underneath it makes the measurement meaningless in a way that is not obvious from the
error. Do documentation, planning, or nothing.

Related and worth having done anyway: `transport_run.cu` was being compiled three times per
pipeline - once each for the viewer, the GUI and example B1 - at several minutes apiece,
because it owns the `__global__` template kernels and cannot live in a header.
`build_engine.bat` compiles it once to an object and the three link it, which roughly halves
the pipeline.

## G3. A safety that was allowed to be too large

`safety_out` is the isotropic safety: a lower bound on the distance from an interior point to
the solid's surface, in any direction. Urban MSC uses it to decide how long a step it can
afford. Geant4 requires only that it never *over*estimate - an underestimate costs performance
and nothing else.

The default branch, for any solid without a closed form, was:

```cpp
const real_t r = bounding_radius(s);
return fmax(real_t(0), real_t(0.5) * (r - d));      // d = |q|
```

That is sound for a sphere, where the bounding radius *is* the distance to the surface from the
centre. It is unsound for anything flat or elongated, and badly so. Take `G4Tubs` with
rmax = 1 mm and dz = 100 mm - a wire. Its bounding radius is 100.005 mm, so at a point on the
axis the expression claims a safety of 50 mm, where the wall is 1 mm away. MSC would then take
a 50 mm step inside a 1 mm tube.

Nothing in the test suite could see it. B1's three solids are a box, a cone and a trapezoid,
all of which have exact closed forms and never reach the default. The solids that do reach it -
tubs, sphere, ellipsoid, paraboloid, hype, para - are checked in `tests/test_solids.cu` against
Geant4's `G4VSolid`, but that test compares `Inside`, `DistanceToIn` and `DistanceToOut`, which
are the *directional* queries. It does not compare the isotropic safety, because Geant4's own
`DistanceToOut(p)` is a different underestimate and comparing two underestimates tells you
nothing.

What found it was the mesh work, from the other direction: a mesh had no safety at all, and
making the mesh scene agree with the analytic scene meant understanding what the analytic
scene's safety actually was. Reading it to compare it was the first time anyone had read it.

The fix separates the two directions, which are not symmetric:

- from **outside**, the bounding (circum)radius is the sound one: a point at distance d > r
  from the origin cannot be closer than d - r to anything inside the bounding sphere.
- from **inside**, the *in*radius is: if the largest origin-centred sphere inside the solid has
  radius r_in and the point is at d < r_in, then the ball of radius (r_in - d) around the point
  is inside that sphere and therefore inside the solid.

`inradius()` returns 0 - always sound, no safety information - for every solid where the answer
is not obvious: a torus (whose origin is in its hole), a hyperbolic tube (an inner void), a
paraboloid and an elliptical cone (origin at the apex), a phi wedge or a hollow shell (origin
not in the solid at all), and everything plane-based.

**Generalisation.** A quantity whose contract is one-sided - "never overestimate" - needs a
test that can only fail in that direction. Comparing it against another implementation's
underestimate cannot do that. The check that works is against an exact value: for the mesh, the
nearest triangle; for a box, the closed form. `tests/test_mesh.cu` now compares the mesh's
safety against the trapezoid's closed form and agrees to 5e-15 mm, which is a real check
because one of the two is exact.

## A4. Zero safety is sound, and cost 40% of the run

The first working version of mesh transport had no isotropic safety: a bounding box gives no
lower bound on the distance from an interior point to the surface, so the honest answer was
zero. Zero is sound. It is also what switches Urban MSC's step limitation off entirely.

`g4dose -compare B1 B1mesh` is the check that saw it. B1's scoring volume is a `G4Trd`;
B1mesh's is a twelve-triangle mesh of the identical trapezoid, with the same seed and the same
source, so the two must agree. They did not:

| | dose per 10k events | track-steps |
|---|---|---|
| B1, a G4Trd | 427.36 +/- 1.74 pGy | 6,535,373 |
| B1mesh, its mesh | 430.48 +/- 1.75 pGy | 9,296,080 |

1.26 sigma apart - which on its own is nothing, well inside the noise, and would have been
dismissed. The 42% step count is what made it a defect rather than a fluctuation: the same
geometry cannot cost 2.8 million more steps.

The fix is `mesh_safety()`: the exact distance to the nearest triangle, by a BVH walk with the
same shape as the ray query - descend, prune any node whose bounding box is already farther
than the best triangle found. It is not an approximation and not a bound; it is the answer, and
it agrees with the trapezoid's closed form to 5e-15 mm.

**Generalisation.** Two symptoms, one from the physics and one from the machinery, and only the
second was decisive. A dose that differs by a sigma is a coin-flip; a step count that differs
by 42% for the same geometry is a fact. Report both, and when the noisy one disagrees, look at
the other before deciding it was noise.

## S4. The guard that makes a build fast also serves you a stale one

`build_engine.bat` starts with `if defined G4GPU_ENGINE_OBJ exit /b 0`. That is what stops the
four-minute compile of `transport_run.cu` from happening once per target within a pipeline run,
and it is right.

While iterating on the mesh code I set that variable by hand to skip the rebuild - and then
built and ran `g4dose.exe` against an engine object compiled before the mesh upload existed.
The triangle pool was never copied to the device, so `store.tri` was null, so `mesh_inside`
returned false for every point, so the mesh volume did not exist. The run completed and printed
a dose.

I noticed before drawing a conclusion, because the object's timestamp was older than the source
I had just edited. That is not a check, it is luck.

**Generalisation.** A cache keyed on "someone said it was done" is a cache with no invalidation.
The variable should be a *path plus a stamp*, or the script should compare timestamps itself.
Until it does: never set `G4GPU_ENGINE_OBJ` by hand, and if a result surprises you, look at
`out\transport_run.obj`'s modification time before looking anywhere else.

## S5. FATAL printed eleven times, exit code zero

Running `exampleB1 vis.mac` - the visualisation macro on its own, without the
`/run/initialize` that `init_vis.mac` does first - printed

```
FATAL: /vis/open before /run/initialize - there is no geometry to draw.
```

eleven times, once per `/vis/` command in the file, and exited 0.

Three separate faults in one line of output:

1. **The word.** Nothing was fatal; the command failed and the program carried on. "FATAL" is
   the string `build_all.bat` greps for to fail the pipeline, so misusing it makes the word
   mean less exactly where it needs to mean the most.
2. **Eleven times.** A viewer that failed to open will fail identically for every later command
   in the macro. The reason is now latched and reported once.
3. **Exit code 0.** `G4UImanager::ApplyCommand` returned the error count and
   `exampleB1.cc` discarded it, because Geant4's `exampleB1.cc` discards it. This one now
   returns it: a macro whose commands failed has not done what the file said, and a program
   that reports success anyway is precisely how a run gets published with the physics it was
   supposed to have switched on still off (see S2).

**Generalisation.** Fidelity to an upstream example is worth a lot, and it is not worth
propagating a discarded error code. Where the two conflict, deviate and say so in a comment -
which is what the comment at that line now does.

## O6. A generated table of the right shape and the wrong data

The PSTAR and ASTAR stopping powers are hard-coded in Geant4 - they are the published NIST
tabulations, 74 materials by 60 or 78 energies - so `tools/extract_stopping.sh` reads them out
of `G4PSTARStopping.cc` and `G4ASTARStopping.cc` and emits a header. Reading the same files the
library compiles is what makes "every number is the number Geant4 has" a fact rather than a
hope. That part worked.

The material *names* are in a third file, and they are annotated:

```cpp
static const G4String nameNIST[74] = {"G4_A-150_TISSUE",
                                      ...
                                      "G4_B-100_BONE",  // 0 - 9
                                      "G4_Be",
```

The extractor flattens an initialiser list onto one line and splits it on commas. With the
annotation left in, the flattened text reads `..."G4_B-100_BONE",//0-9"G4_Be",...`, whose next
comma-separated field is `//0-9"G4_Be"`. That emits as a C++ comment. The name is swallowed,
every later name shifts up by one, and C++ zero-fills the tail.

Every structural check passed. The array was still declared `[74]`. It still had 74 rows in
the file. It still compiled. The generator's own count assertions - which exist, and which
caught two real extraction faults in the float arrays - all passed, because the *count* was
right. What was wrong was the correspondence: `nist_stopping_index("G4_WATER")` returned 64,
row 64 held G4_TEFLON's table, and water got teflon's stopping power. Off by 54% at 3 keV,
where the Bragg peak is.

It was found by `tools/check_pstar.cc`, which compares every extracted point against
`G4PSTARStopping::GetElectronicDEDX`'s own answer in `ref/oracle/bragg.csv`. That check took
ten minutes to write and is the only thing in the chain that could have found it.

Two fixes:

- the extractor strips `//` comments before flattening, with the reason written at the
  function;
- `tools/regen_stopping.sh` runs extract, **recompile**, generate and check in that order. The
  recompile step is there because of a second trap that cost as long as the first: the
  generator compiles the raw table header *in*, so regenerating the raw header and re-running
  the existing generator binary silently produces the old tables from the new data. Same
  family as S4.

**Generalisation.** For generated data, a check on the shape is not a check on the contents,
and the shape is what generators get right. The check has to compare values against the
source of truth, at points chosen by something other than the generator. And the thing to
compare is not the table - it is the *answer*: reading it through Geant4's own spline caught
the interpolation and the sub-keV extrapolation at the same time, neither of which a
table-versus-table diff would have looked at.

## O7. Transcribing from a Geant4 that is not the oracle's Geant4

There are three Geant4 trees on this machine:

| path | what it is |
|---|---|
| `D:/g4gpu/reference-geant4` | a shallow clone, **11.5.0** |
| `D:/Documents/Geant4/Windows/geant4-v10.7.3-install` | an older install, no source |
| `D:/Documents/Geant4/Windows/geant4-v11.1.1` + `-install` | the source **and** install matching the oracle |

`ref/oracle/*.csv` is produced by `ref/dump/g4dump.cc`, which links the **11.1.1 install**. So
11.1.1 is the only version whose answers this port is measured against.

For most of a day I transcribed physics out of `reference-geant4` - the 11.5.0 clone. PSTAR,
ASTAR, the material name list, the 78-point grid, the not-a-knot spline, the Fermi velocity
and lFactor tables: all read from the wrong version, then checked against the right one.

The mistake was not caught by a measurement, because every one of those things happens to be
identical between 11.5.0 and 11.1.1:

```
extracted PSTAR/ASTAR/names/grid      0 differing lines
vFermi[92] and lFactor[92]            identical
ComputeSecDerivative1, Interpolation  same algorithm, same expressions
```

So the checks passed, at 1e-6, and told me nothing about the version. It surfaced only when I
tried to *call* something: `G4ionEffectiveCharge::ComputeCharge` exists in 11.5.0 and does not
exist in 11.1.1, and the oracle dumper would not compile.

`docs/PLAN.md` had recorded "Geant4 11.5.0 reference clone" from the day it was cloned. The
information was written down and I still used it wrong, which is the part worth noticing: a
fact in a document is not a guard.

**What 11.1.1 and 11.5.0 actually differ on**, of what has been read so far - only
`G4ionEffectiveCharge`, and only in shape:

- 11.5.0 exposes `ComputeCharge(p, m, E, corr)`, returning the charge and writing the
  correction. 11.1.1 has no such member: `chargeCorrection` is private with no accessor.
- 11.1.1 gates the branches on `Zi <= 1` then `Zi <= 2`; 11.5.0 on `effCharge <= 1.5` then
  `Zi == 2`. For an integer ion charge these are the same test.
- The two formulae themselves - the six-coefficient helium fit and Ziegler's heavy-ion form -
  are character-for-character the same.

**Fix.** `tools/g4src.sh` resolves the source tree and *verifies* it, by reading
`G4VERSION_NUMBER` out of both the source tree and the install the oracle links, and refusing
if they differ:

```
$ sh tools/g4src.sh D:/g4gpu/reference-geant4
g4src.sh: VERSION MISMATCH - refusing to transcribe from the wrong Geant4.
          source  D:/g4gpu/reference-geant4     G4VERSION_NUMBER 1150
          oracle  ...geant4-v11.1.1-install     G4VERSION_NUMBER 1111
```

`tools/regen_stopping.sh` takes its tree from it rather than naming one. The authority is the
install, not a constant in a script, because the install is what produces the numbers.

**Generalisation.** When the reference implementation is *on disk*, "which copy" is part of
the reference. A comparison against an oracle validates the transcription and says nothing
about its provenance - two versions that agree hide the error completely, and the day they
stop agreeing the failure looks like a physics bug. Derive the source from the oracle and
check it mechanically; do not write the path in a comment and trust yourself to read it.

## S6. A line-splice with an empty bound duplicates the file

Every edit today that could not go through the Write tool went through this shape:

```sh
A=$(grep -n '<start>' f | cut -d: -f1)
B=$(grep -n '<end>'   f | cut -d: -f1)
{ sed -n "1,$((A-1))p" f; cat new; sed -n "$((B+1)),\$p" f; } > tmp && cp tmp f
```

When a grep finds nothing - a pattern with a backslash in it, most often, since the pattern
has to survive both the shell and grep - the variable is empty, `$((B+1))` is `1`, and the
final clause appends **the entire original file** after the replacement. No error, no warning,
exit zero. `build_all.bat` came out as its first 110 lines, the new block, and then a complete
second copy of itself, 258 lines where 146 were expected.

It is a quiet failure in the worst place: the file still starts correctly, so the head of a
diff looks right, and `cmd` reads `@echo off` in the middle of a script without complaint.

**Fix.** `tools/splice.sh` does the same job and refuses an empty or non-numeric bound, a last
line before the first, or a bound past the end of the file, and reports the resulting line
count so a duplication is visible in the output:

```
splice: first line '' is not a number
splice: build_all.bat lines 111..112 -> 18 lines, now 146 lines
```

**Generalisation.** A shell arithmetic expansion turns a missing value into `0` and carries
on. Any splice, slice or range computed from `grep -n` needs the bound checked before it is
used - and the cheapest check that would have caught this one is printing the line count
afterwards, because the failure mode is always "much longer than it should be".

## O8. A physical constant that is more accurate than the reference

`G4ionEffectiveCharge`'s helium branch came out 1.5e-8 from Geant4's answer, and every
plausible cause was a formula: the six-coefficient fit, the two series expansions, the
`Zi <= 1` gate. It was none of them. The port held

```
kMassFactor = 931.49410242 / (938.272013 * 1e-3)     // amu_c2 / (proton_mass_c2 * keV)
```

and CLHEP, in 11.1.1, defines `amu_c2 = 931.494028 * MeV`. The port's value is the current
CODATA one - **more accurate, and wrong**, because the job is to reproduce Geant4's numbers and
Geant4 is using the older figure. The relative difference is 8e-8, which is precisely the size
of the discrepancy once the logarithm has damped it.

Looking for the rest of the family found four more:

| where | the port had | 11.1.1's CLHEP | off by |
|---|---|---|---|
| `nuclear_stopping.cuh` | `amu = 931.49410242` | `931.494028` | 8e-8 |
| `units.cuh` | `r_e = 2.8179403262e-12 mm` | `2.817940545232519e-12` | 7.8e-8 |
| `brems_rel.cuh`, `brems_data.cuh` | `lambda_c = 3.8615926796e-10 mm` | `3.8615929818578764e-10` | 7.8e-8 |
| fourteen files | `alpha = 1/137.035999139` | `0.007297352565305215` | 1.5e-10 |

Two things made this worse than a single stale literal.

**The port disagreed with itself.** `bragg.cuh` had `931.494028` and `nuclear_stopping.cuh`,
two files away, had `931.49410242`. Fourteen files each spelled out the fine structure constant.
Every copy is an independent chance to be a different vintage.

**CLHEP derives some of these rather than tabulating them.** `classic_electr_radius` is
`elm_coupling/electron_mass_c2`, built from the unit system's own e, mu0 and c;
`fine_structure_const` is `elm_coupling/hbarc`. There is no literal in CLHEP to copy, so the
only way to match it is to read the value Geant4 computes. That is now what the port holds, and
where the 8e-8 in every ionisation prefactor came from - `r_e` appears squared in all of them.

**Fix.** `ref/dump/g4dump.cc` writes `constants.csv`: thirteen CLHEP constants at `%.17g`.
`tests/test_constants.cu` compares both of the port's sets - `core/units.cuh` for the device
physics and `g4/G4PhysicalConstants.hh` for the Geant4-shaped API - against it, at one bit,
and fails if `constants.csv` has fewer rows than the test checks. Every duplicate literal in
`src/` now calls `units::`.

```
  constant                                 Geant4    units.cuh    G4 header
  electron_mass_c2            0.51099890999999997     0.00e+00     0.00e+00
  amu_c2                       931.49402799999996     0.00e+00     0.00e+00
  classic_electr_radius     2.817940545232519e-12     0.00e+00     0.00e+00
  fine_structure_const       0.007297352565305215     0.00e+00     0.00e+00
```

After it, `G4ionEffectiveCharge`'s helium branch agrees with Geant4 **exactly** - 0.000e+00
over 606 points - and the heavy-ion branch to 1e-14.

**Generalisation.** In a port, a constant is data, and data gets checked against the oracle
like everything else. Accuracy is not the criterion; agreement is. And "it's just a constant"
is what makes this expensive: it is the one class of error that survives every formula review,
because the formula is right.

## O9. Mass fractions that do not sum to one

Found while chasing the constants: the port's electron density was 1e-6 below Geant4's, in
every material. Not a constant this time - `G4_AIR`'s four NIST mass fractions are

```
0.000124 + 0.755267 + 0.231781 + 0.012827 = 0.999999
```

and `G4Material::FillProperties` normalises them:

```cpp
G4double coeff = (wtSum > 0.0) ? 1./wtSum : 1.0;
...
fMassFractionVector[i] *= coeff;
```

The port did not. So its air held 1e-6 fewer electrons per mm^3 than Geant4's air, and since
every macroscopic cross section is a number density times a per-atom cross section, everything
in air was 1e-6 low.

The part worth recording is not the bug. It is that **`test_vs_oracle` had been printing this
ratio since the day it was written**:

```
  G4_AIR         electron density /mm3    3.62182e+17  3.62182e+17   1.0000
```

Four significant figures of a quantity that was wrong in the seventh. The check existed, the
number was on the screen, and the format hid it - `%8.4f` of a 1e-6 error is `1.0000`. Nothing
asserted on it, so nothing had to notice.

Normalising took three lines. `tests/test_ion_charge.cu` now asserts the electron density at
1e-12 alongside Zeff, the Fermi energy, <A^-2/3> and the L-factor, and it lands at 2.2e-16.

**Generalisation.** A printed ratio is not a check, and a printed ratio at four decimal places
cannot even be read as one. If a quantity is worth dumping from the oracle it is worth
asserting on, at a tolerance someone chose on purpose - and the tolerance is what turns the
number into information. This is S1 and O6 in a third costume: the machinery was all there and
no one had said what "agrees" meant.

## O10. A model feature that exists, is documented, is switchable, and does nothing

`docs/ROADMAP.md` had "WentzelVI second moment" on the list of six remaining physics items,
and `src/physics/em/wentzel_msc.cuh` said

```
// Not transcribed: the second-moment correction (useSecondMoment is false by default), ...
```

which reads as a gap. It is not one. Following it through 11.1.1:

```cpp
// G4WentzelVIModel::SampleScattering
if(useSecondMoment) {
  G4double z1 = invlambda*invlambda;
  G4double z2 = SecondMoment(particle, currentCouple, effKinEnergy);
  prob2 = (z2 - z1)/(1.5*z1 - z2);
}
...
if(prob2 > 0.0 && rndmEngine->flat() < prob2) { isFirst = false; }
```

```cpp
// G4WentzelOKandVIxSection::ComputeSecondTransportMoment
G4double
G4WentzelOKandVIxSection::ComputeSecondTransportMoment(G4double /*CosThetaMax*/)
{
  return 0.0;
}
```

The leaf returns zero. It is not virtual and has one definition. So `z2 == 0`,
`prob2 == -z1/(1.5 z1) == -2/3`, and after `prob2 /= (1 + prob2)` it is `-2`. The guard
`prob2 > 0.0` is never true, and the gamma-distributed branch it selects is unreachable.

**Turning `useSecondMoment` on in Geant4 11.1.1 changes nothing.** There is no correction to
transcribe: the machinery is complete - a flag, a setter, a getter, a per-material physics
table built at initialisation, an interpolating accessor - around a function that returns
zero. The port's comment was wrong in a way that overstated the gap, and the item is closed by
reading the code rather than by writing any.

This is the third time this shape has appeared:

- the photoelectric "spline approximation", which turned out to apply only to the `livermore`
  data directory and not to `epics2017`, so it was never on the path this port takes;
- `G4BraggModel`'s molecular branch, unreachable for the material that was supposed to select
  it because the name match wins first;
- and this.

**Generalisation.** "Geant4 has X and we do not" is a claim about a call path, not about a
symbol. Before writing a transcription, follow the value to where it is consumed and check
that something consumes it - the flag, the table and the accessor can all be real while the
number is zero. Each of these three cost hours of transcription that would have produced code
no test could distinguish from its absence, and one of them was avoided only because the
measurement refused to move.

## S7. Generated code that does not compile, in the shape nothing had ever generated

The model builder's "Save" writes a compilable Geant4-shaped project, and `build_all.bat`
compiles and runs it, because "you get a working project" is a claim worth checking. What the
pipeline had never done was save a project containing a **polycone**, and the emitter's default
case is:

```cpp
default:
  return "nullptr /* " + std::string(ShapeName(s.shape)) + " is not yet emitted */";
```

which produces

```cpp
auto* solidCone = nullptr /* Polycone is not yet emitted */;
```

`error: cannot deduce "auto" type`. Two of the twenty-two shapes the builder can create -
polycone and polyhedra - could not be saved at all, and the failure mode was a compile error
in a file the user is told is a working project.

It surfaced because the builder selftest gained a polycone for an unrelated reason: the
renderer had been calling the store-free `dist_in` overload, and a boolean and a polycone are
the two shapes whose rendering needs the solid store, so the selftest started inserting both.
The compile then failed on the next pipeline run.

Both are emitted now, through a generated `MakePolycone`/`MakePolyhedra` helper - `EmitSolid`
returns one expression and `G4Polycone` wants three parallel arrays, so the helper splits a
flat list of `(z, rInner, rOuter)` triples. And the default case no longer emits anything:
`CanEmitSolid` is checked in `WriteProject` before a file is opened, so an unemittable shape is
**refused by name**, with its section count, instead of being handed to the compiler.

**Generalisation.** A placeholder that compiles is a bug you find later; a placeholder that
does not compile is a bug you find at the worst moment, in someone else's build. Neither
belongs in generated output - the generator should refuse. And a check that exercises one path
through a generator ("does the saved project compile?") is only a check on the shapes that
path happens to contain: this one covered five shapes out of twenty-two for weeks and read as
though it covered the feature.

## O11. The reference's *precision* is part of the reference

The ICRU 90 tables came out 6e-8 from Geant4's answers - all six of them, at every energy, in
every material. A uniform relative offset like that is not an arithmetic difference; it is a
type. Geant4 stores these stopping powers as `G4float` and widens them on the way into the
physics vector:

```cpp
static const G4float e0_proton[57] = { 119.70f, 146.70f, ... };
...
data->PutValues(i, e[i]*CLHEP::MeV, ((G4double)dedx[i])*fac);
```

So the number Geant4 splines is `(double)(float)119.70` - which is `119.69999694824219` - and
the extractor had emitted `119.70` as a double. Float has about 1.2e-7 of relative precision,
and half of that is 6e-8.

The generator now rounds each stopping power through `float` before emitting it **and before
computing the second derivatives**, because Geant4 splines the widened floats. All six tables
went from 6e-8 to 2.2e-16.

The energy grids are `G4double` in Geant4 and are left alone - which is the point: the rule is
not "round everything", it is "match the type the reference used, field by field".

**Generalisation.** This is O8 again in a different costume. There, the port held a *more
accurate* physical constant than Geant4 and was wrong for it; here it held a *more precise*
table value and was wrong for it. In a port, accuracy is not the objective and neither is
precision - agreement is. A `float` literal in the reference is a fact about the reference,
and a uniform offset at the 1e-7 level is the fingerprint to look for before suspecting the
formula.

### Amendment to S6: the tool only helps where it is used

`tools/splice.sh` exists because a line splice with an empty bound appends the whole file. It
then happened again, in a different direction, on `tests/test_icru90.cu`:

```sh
A=$(grep -n '<a pattern that did not match>' f | cut -d: -f1)
{ sed -n "1,$((A-1))p" f; cat new; sed -n "$A,\$p" f; } > tmp
cp tmp f
```

The first `sed` errored, the second produced nothing usable, the redirection succeeded anyway,
and `cp` replaced a 200-line test with the 80-line fragment. Exit status of the block: zero.

The reason the tool was bypassed is that this was an *insert*, not a replace, and `splice.sh`
takes a range. It handles inserts fine - `splice A A` with the original line repeated at the
top of the replacement - and using it that way would have failed loudly on the empty bound.

So the rule is not "there is a tool", it is: **never compute a line number and use it in the
same breath.** Either the bound goes through `splice.sh`, or it is checked before use. A
pattern that stops matching is the normal case, not the unlucky one - it happens every time a
file is edited twice.

## G4. A helper whose doc comment reads as a stronger promise than it makes

`geometry/volume_of.cuh` had:

```cpp
/// A bound on how far the solid reaches from its own origin. Used to size the sampling box.
__host__ inline real_t solid_half_extent(...)
```

"How far the solid reaches from its own origin" is a radius. What the function returns is
`max(dx, dy, dz)` for a box - a bound on each *coordinate*. A box of (30, 40, 50) returns 50
and reaches 71.4 mm at its corner, 43% further.

For the one caller it had, sizing an axis-aligned Monte-Carlo sampling box, the per-coordinate
reading is the right one and the function is correct. The problem is the next caller. A
bounding sphere per volume is the obvious way to make `locate()` and `step_to_boundary()`
better than O(number of volumes), and this function is the obvious thing to build it from -
and doing so would have made the navigator skip volumes it should have entered, in a way that
shows up as a wrong dose and not as a crash.

The failure would have been particularly hard to see: it needs a point in the corner region of
a box-like volume that overlaps another, so most geometries and most tracks would be fine.

**Fix.** The comment now states the per-coordinate contract, says in bold that it is not a
radius, gives the sqrt(3) conversion and the reason it is exactly sqrt(3) (`to_local` is a
rotation about the placement, so it preserves length). And `tests/test_solids.cu` checks the
contract over all thirty solids the oracle covers - 200,000 samples each - because two of the
cases are admitted guesses:

```
  solid              inside     extent  worst |x|  worst |p|  |p|/ext
  box                  2973         50      49.99      66.96    1.339
  trd                  2063         50         50      68.22    1.364
  trap                   24        200      49.95      54.02    0.270
  tet                    60         80      33.84       38.3    0.479
```

The contract holds everywhere, the worst radius ratio is 1.364 against sqrt(3) = 1.732, and
the Trap and Tet bounds turn out to be 4x and 2.4x looser than they need to be - which is the
sort of thing worth knowing *before* building an acceleration structure on them, since a loose
bound is a slow one.

**Generalisation.** A comment is an interface. When it describes a *stronger* property than the
code provides, it is not documentation but a trap laid for the next caller - and unlike wrong
code, nothing compiles it or runs it. The check here is worth more than the correction: the
comment can drift again, and 200,000 samples per solid cannot.

## U1. One buffer, four fields, all of them reset every frame

Typing into the model builder stopped working as soon as anything was added to the model. Not
intermittently - every character was gone by the next frame.

An immediate-mode text field needs somewhere to keep the partially typed string. There was one
`App::name_edit` and an integer tag saying which field owned it, reset like this:

```cpp
if (a.name_edit_for != 10000 + a.sel_element) {
  a.name_edit = e.symbol;
  a.name_edit_for = 10000 + a.sel_element;
}
```

at four different sites - the element symbol, the material's NIST name, the source's particle,
the source's nuclide. Every one of them ran every frame. With an element *and* a material
selected, the element block claimed the buffer, the material block saw a tag that did not
match and claimed it back, and the next frame reversed it. Two fields fighting over one string,
sixty times a second.

Before anything was added nothing was selected, so only one site ran and typing worked. That is
why the bug looked like "typing breaks after you add an element" rather than "two of these
cannot coexist".

**Fix.** `ui::EditBuffer` - a key and a string - one per field site. The key identifies what is
being edited, so a buffer only ever holds one field's text and the aliasing is impossible
rather than avoided.

**Generalisation.** Shared mutable scratch space with a tag saying who owns it is a lock
without a lock: it works exactly as long as one thing wants it at a time, and the failure is
not an error but a value that silently belongs to someone else. In an immediate-mode UI, where
every widget re-runs every frame, "one at a time" is not a property the code can have.

## U2. A panel that ran a Monte Carlo, sixty times a second

`G4RunManager::ScoredMass` returns the mass of a scored volume. When a higher layer overlaps
that volume the mass has no closed form - the region it owns has to be integrated - so it
samples 200,000 points and prints a note saying it did.

The GUI called it once per scorer to label its readout. From inside the draw function. Which
runs every frame.

So a detector with an overlapped scoring volume ran a 200,000-sample integration per scorer per
frame and printed the note each time, which is what "the terminal starts repeatedly printing
out an error message" was. The note was not an error and it was not repeating because anything
was wrong; it was repeating because it was being *recomputed*.

**Fix.** A cache on the run manager, cleared in `Initialize()` - the only thing that can change
the answer. And the note reworded: an overlap resolved by the layer rule is how the model
works, and the dose already uses the corrected mass.

**Generalisation.** A function that is expensive *and* talks is a function that must not be
called from a draw. The print is the tell: output that repeats at the frame rate is not a
message, it is a measurement of how often something is being redone.

## G5. A normal that could only point in 26 directions

A sphere rendered as flat facets whose boundaries were the coordinate planes and the diagonals -
reported as "the sphere renders with central cartesian planes".

The normal for the curved solids was a central difference of the *containment indicator*:

```cpp
auto f = [&](v) { return inside(st, s, v) ? -1 : +1; };
g = { f(x+h) - f(x-h), f(y+h) - f(y-h), f(z+h) - f(z-h) };
```

Each component of that is -2, 0 or +2. The gradient is therefore one of 26 directions, and
shading a smooth surface with it quantizes the normal into 26 patches. On a sphere those
patches are bounded by the planes where a component changes sign, which are exactly the
coordinate planes.

It affects nothing physical - normals are used for shading and for nothing else - which is why
it survived: every geometry test compares containment and distances, and 120,000 rays of
`G4VSolid` agreement say nothing about how a surface looks.

**Fix.** The quadric shapes get the analytic gradient of whichever bounding form the point lies
on, chosen by `|Q| / |grad Q|` so surfaces of different scales compare fairly. Booleans pick the
operand whose containment changes within a probe of the point, negated for a subtracted
right-hand side. What is left on the numerical path is the torus, a voxel grid, and shapes that
are flat anyway - and it is now a named function whose doc comment says why it is the last
resort.

**Generalization.** A quantity that only feeds the picture has no test, so its errors are
reported by users in the language of what they saw. "Central cartesian planes" is a precise bug
report about a normal, and it took reading the fallback to hear it that way.

## R1. A ray-marcher that re-found the surface it had just crossed

Front-to-back compositing was added so a transparent shell would show what is inside it. A box
at 45% opacity rendered fully opaque, with nothing behind it visible at all.

The walk searched for the nearest entry point, drew it, advanced just past it, and searched
again. But `dist_in` asked from *inside* a solid returns approximately zero - the point is
already in it - so the search re-found the volume it had just entered, painted the same front
face again, and did that until the accumulated alpha reached 1. Eight passes over one surface
at 0.45 each is opaque.

The failure is worse than "transparency does not work": it looks exactly like an opaque object,
so the natural conclusion is that the opacity never reached the renderer. Three of the four
places it passes through were checked before the walk itself was suspected.

**Fix.** Skip any volume the ray is already inside. A volume you are inside has no entry
surface ahead of you, and the next thing in front is by definition something else.

**Generalization.** `dist_in` from inside and `dist_out` from outside are both answerable and
both meaningless, and a marcher that does not track which side it is on will ask one of them.
The invariant has to be maintained explicitly, because the function will not complain.

## G6. Two volumes on one layer is not a geometry, it is a coin toss

The layer model replaces the mother/daughter tree: where two volumes overlap, the higher layer
owns the space. Two volumes on the *same* layer have no such rule. The tie-break in `locate`
is "the later index wins", which is an implementation detail - the same detector with the two
solids added in the other order gives a different dose, and nothing said so.

Geant4 checks for overlapping placements at construction and complains. This did not, because
overlap is *normal* here: it is the mechanism. What is not normal is an overlap the layer rule
cannot resolve.

**Fix.** After every scene rebuild, same-layer pairs are tested - bounding spheres to filter,
then sampling inside the smaller one to confirm - and a run is **refused** while any exist,
with the pairs listed, the fraction shared, and a shortcut to raise one layer. Refused rather
than warned: the number a run would produce is not wrong in a way that can be corrected for, it
is a number whose meaning depends on the build order.

The refusal is checked by the pipeline rather than asserted in a comment. The builder selftest
puts two volumes on one layer, asks for a run, and the build fails if a run happened:

```
WARNING: GlassBox and InnerBall overlap on layer 2 (91% of the smaller one)
run refused: two volumes overlap on the same layer.
selftest: a run with a same-layer overlap was refused, as it should be
same-layer overlaps resolved
```

**Generalization.** When a model's whole point is to permit something other systems forbid -
here, overlap - the check that gets dropped is the one for the case the new rule *still* cannot
decide. Permitting overlap is not the same as deciding every overlap, and the difference is
where the arbitrary answer lives.

## O12. A material with no mean excitation energy scored NaN

A material built in the GUI - elements, fractions, a density, nothing else - produced a dose of
NaN.

`Material::mean_excitation` reached the device record as 0, because that is what
`G4Material::GetMeanExcitationEnergy` returns for a material nobody set it on. The
density-effect parameters are built from

```cpp
m.c_density = 1 + 2 * log(m.mean_excitation / plasma);
```

`log(0)` is negative infinity, `c_density` is -inf, `density_correction` is NaN, every dE/dx
after it is NaN, and the integral of NaN is NaN.

Geant4 does not pass zero through. `G4IonisParamMat::ComputeMeanParameters` derives one when
the material has no tabulated value, by Bragg additivity in the logarithm:

```
ln(I) = sum_i ( n_i Z_i ln(I_i) ) / sum_i ( n_i Z_i )
```

Nothing caught it because **every material in every test and in the oracle had its excitation
energy set explicitly** - the four B1 materials from the NIST tables, and the two custom ones
built by the dumper with a value passed in. The one configuration that had none was the one a
user builds.

**Fix.** `src/data/nist_excitation.hh` derives it, from the same per-element values Geant4
uses, read out of the generated NIST table rather than tabulated a second time. It reproduces
Geant4's answer to 4e-16 against `CustomDerivedI` - a three-element material, hydrogen to lead,
added to the oracle specifically so the derivation has something behind it - and
`tests/test_material_build.cu` checks the derived value, every derived quantity built from it,
and that a dE/dx in such a material is finite. A material with no usable components at all is
now refused by name instead of producing a record with zero electron density.

Two more things fell out of the same hunt:

- **the ionisation prefactor was defined twice.** `em::twopi_mc2_rcl2` existed in both
  `electron_processes.cuh` (as `constexpr`) and `hadron_ionisation.cuh` (as `inline`), each
  computing the product its own way. Two definitions of one symbol in one namespace is an ODR
  violation that only failed to compile because no translation unit had ever included both
  headers - a test needing an electron dE/dx and a hadron dE/dx in one file found it in
  seconds. The real risk was not the violation but what it invited: two prefactors that could
  drift, with electron and hadron ionisation quietly using different ones.
- **the oracle dumper's geometry was sized for five materials.** It places one box per material
  at `(i - 2.0) * 35 cm` under a comment saying "spaced to fit however many materials are in
  the list". The seventh landed 150 cm out and Geant4 aborted with "Daughter physical volume
  pv6 is entirely outside mother logical volume World". The spacing is computed from the count
  now.

**Generalization.** The configuration nothing tests is the one the *tool* produces rather than
the one the test fixtures produce. Every material here was built by code that knew what a
material needs; the GUI builds one from what a user typed, which is a different distribution
of inputs and the only one with a hole in it.

## S8. The same splice, a third time, and what finally fixed it

`tools/splice.sh` was written because a line splice with an empty bound appends the whole file
(S6). It then happened again as an *insert* rather than a replace, truncating a test to the
fragment being inserted (the amendment to S6). It then happened a third time and destroyed
**two build scripts**:

```sh
A=$(grep -n '^nvcc ... -o g4builder.exe' build_gui.bat | cut -d: -f1)
{ sed -n "1,$((A-1))p" build_gui.bat; cat new; sed -n "$A,\$p" build_gui.bat; } > tmp
cp tmp build_gui.bat
```

The pattern contained a backslash, grep found nothing, `A` was empty. `1,-1p` printed nothing,
`,$p` was a syntax error, and `cp` installed a file containing only the inserted text.
`build_gui.bat` and `build_view.bat` went from working scripts to fourteen lines of guard
apiece. There is no git here, so they were reconstructed from `build_dose.bat`'s pattern and the
link inputs visible in the build logs, then verified by deleting both executables and building
from scratch.

Twice the tool existed and was bypassed, because the tool took *line numbers* and the thing
being done was an insert addressed by a pattern - so the pattern had to be resolved outside it,
which is precisely where the resolution goes wrong.

**Fix.** The tool takes the pattern:

```
sh tools/splice.sh <file> before|after|replace <pattern> <newfile>
```

A pattern that matches nothing, or matches more than once, is an error printed by the tool. The
line-number form is still there for when a number is genuinely what you have, and still refuses
an empty bound.

**Generalization.** Three occurrences of one mistake, the first two "fixed" by adding a tool
that the mistake routed around. A guard only helps at the boundary the error actually crosses -
and here the error was never in the splice, it was in resolving a line number in a shell where
failure is a value rather than a stop. Moving the resolution inside the checked thing is the
fix; adding a checked thing beside it was not.

### Amendment to S4: the link guard has to be at the link

`LNK1104: cannot open file 'g4builder.exe'` ended a twenty-minute pipeline run twice. The first
time added a check at the top of `build_all.bat` for a running executable. The second time that
check *passed* and the failure happened anyway: a GUI process can outlive the wait that started
it, so it reappeared after the check and before the link.

The check is now also immediately before each link, in `build_gui.bat` and `build_view.bat`,
where the thing it protects actually happens. A guard at the start of a twenty-minute pipeline
is a guard against the state at minute zero.

### V1: a test whose statistics were a coin flip, and the day it cost

**Symptom.** Turning on per-voxel scoring made *every* score in the builder's selftest read
exactly `0` - including a scorer on a volume the voxel phantom has nothing to do with. Turning
it off restored `2.81066 MeV`, to the digit. It looked exactly like a transport bug introduced
by the one change that touched the navigator.

**What was checked, in order.** Tracks abandoned: none. Buffer overflows: none. A host-side ray
probe through the scene along the beam axis (`ProbeRay`, kept - see
`src/host/g4builder_panels.inc`) printed the whole path: `World 320 -> Phantom 8 x 5.0000 ->
World 65 -> selftest_cube_stl 30.0000 -> World 445`, materials right, eight cells crossed one
at a time exactly as the change intended. Device-side counters showed 32 gamma steps starting
inside the per-voxel volume and *zero* deposits anywhere in the run. Geometry correct,
navigation correct, nothing deposited.

**Cause.** The selftest ran **25 events**. Twenty of them, spread over a 100 mm beam, through a
30 mm water cube: the expected number of interactions in the scored volume is a fraction of
one. The dose was therefore one rare event or nothing at all - and *which* is decided by where
the RNG stream lands, because `Philox` is seeded from `p.step`. Per-voxel scoring ends a step at
every cell boundary, so a track that crosses the phantom arrives downstream with a different
`p.step` and a different stream. `2.81066` and `0` were two draws from the same distribution.

There was no bug. The change was correct from the first build.

**Fix.** The trajectory cap and the run size were tied together - `traj.max_event = min(n, 100)`
- so the selftest had shrunk the *run* to 25 to keep the *picture* legible. They are now
separate: the picture draws at most 25 events' trajectories and the run is 4000. The selftest's
per-voxel check then passes at 5.6e-16, comparing the cells' sum against the volume total the
same run reported.

Two checks were added to `build_all.bat`: `dose 0 pGy` anywhere in the builder selftest's output
is fatal, and the per-voxel check must be seen to have run. The first is the one that would have
caught this - the pipeline had that check for the *generated project*, which runs 10000 events,
and not for the builder's own run, which ran 25.

**Generalization.** A number small enough to be zero by chance is not a measurement, and a test
built on one reports the RNG's state rather than the code's. Worse, it fails *asymmetrically*:
it passes quietly for as long as nothing perturbs the stream, and then indicts whatever change
happened to perturb it. Two other things were fixed while chasing it - a stale
`G4SDManager::attached_` map surviving rebuilds (real, latent, see `G4SDManager::Reset`) and a
zero-length step in `voxel_step` (real, only reachable with `every_cell`) - and neither was the
cause. Being led to real bugs by a false alarm is not the same as the alarm being right.

### S9: a quoted heredoc that ate backslashes, twice

**Symptom, first time.** The builder logged `build it with D:/g4gpu/out/selftest_projectuild.bat`.
The source said `dir + "\build.bat"` - one backslash, which in C++ is a **backspace** followed by
`uild.bat`. It printed as a missing path separator and read as a typo.

**Symptom, second time.** Five PNG paths written as `"D:\g4gpu\out\..."` produced files called
`g4gpuoutg4builder_dlg_world.png` in the working directory: `\g` is not an escape, `\o` is not
an escape, and MSVC silently drops the backslash before an unrecognised letter. The pictures
were written, somewhere nobody would look.

**Cause.** Both blocks were written through

```sh
cat > file <<'XEOF'
    ... "D:\\g4gpu\\out\\x.png" ...
XEOF
```

A *quoted* heredoc delimiter is supposed to make the body literal, and every `\\` should have
survived. In this environment it does not: the pairs arrive collapsed to single backslashes.
The first occurrence was written off as a typing mistake, which is why it happened again.

**Rule.** Do not write a C++ string literal containing a backslash through a shell heredoc. Two
ways out, in order of preference:

1. **Use forward slashes.** Every path in this project is handed to the CRT or to Win32, both of
   which accept `/` on Windows. `"D:/g4gpu/out/x.png"` has nothing to escape and cannot be
   mangled by anything downstream. Most of the codebase already does this.
2. **Use the file-writing tool, not the shell**, when a backslash is genuinely required - a
   `"\\"` separator, a `"/\\"` character set, an emitted `\n` in generated code.

**Generalization.** Twice is a pattern, and the first time was misdiagnosed as carelessness
rather than as a property of the tool. A mangled *path* fails loudly and gets fixed; a mangled
path that is still a valid relative path writes the file somewhere else and reports success -
which is the same shape as S6, where a broken splice produced a file rather than an error. The
fix is not to be more careful with the escaping, it is to have nothing to escape.

### S10: a splice that removed two lines nobody was looking at

**Symptom.** `exampleB1.exe` exited 0, having printed its initialisation and then nothing. No
crash, no FATAL, no dose. Every check in the pipeline passed except the one added an hour
earlier, which said the example had not reported its agreement with Geant4.

**Cause.** Consolidating three copies of the run loop into `G4RunManager::RunEvents` meant
replacing a range of `BeamOn` by line number. The range was one line longer than intended at the
top and took `run_.n_events = n_events;` and the `BeginOfRunAction` call with it. `n_events` on
the run record is what `G4Run::GetNumberOfEvent()` returns, B1's `EndOfRunAction` opens with
`if (n == 0) { return; }`, and so a run that transported two million events perfectly well
reported nothing.

**Why nothing else caught it.** The dose was computed, the scorers were filled, the transport
was correct - the only casualty was the *reporting*, and every other check reads reported
output. `findstr /C:"FATAL"` found no FATAL because there was none. The exit code was 0 because
nothing failed. A check for "the program printed something" would not have helped either: it
printed nine lines of initialisation.

**Generalization.** This is S6 again in a third costume: a pattern-addressed edit resolved to
line numbers, off by a little, silently. The tool has taken pattern arguments since S6 precisely
so that the resolution happens inside the checked thing - and this edit used the line-number
form anyway, because a *range* was wanted and the tool's pattern mode addresses single lines.
**Fixed.** `splice.sh` now takes a range by pattern:

```
sh tools/splice.sh <file> between <first> <replacement file> <last>
```

Both ends must match exactly once and the second must not precede the first, so the two ways
this went wrong - a pattern that matched nothing, and a range resolved slightly too wide - are
both errors printed by the tool rather than a file quietly rewritten.

The other half of the lesson is the one worth keeping: what saved it was a check added for an
unrelated reason, minutes earlier, that asserted a *number the program is supposed to produce*
rather than the absence of an error. Checks for "did it fail" cannot see work that quietly did
not happen.

### V2: what moving the primaries to the host did to reproducibility

**Not a bug, but a change of contract that broke a check and would have surprised a user.**

While primaries were sampled on the device, every one of them was keyed on the run's seed and
the event index. Two `/run/beamOn 1000` calls in one process therefore fired the *same thousand
events*, and a run was reproducible from `G4RunManager::SetRandomSeed` alone.

Generating them on the host, from Geant4's own engine, means the stream carries on where the
last run left it. Two successive runs are now independent samples - which is what Geant4 does,
and what a user expects.

It is also, on its own, a bug fix. The viewer has an "accumulate across runs" checkbox. With
identical runs, accumulating added the same result to itself: the total grew and the *variance
did not shrink*, which is the opposite of what accumulating is for. Nobody had noticed, because
a number that grows looks like it is working.

**What it broke.** The selftest compares a custom scorer against a stock one by running the same
model twice in one process. Those two runs stopped being the same run, and the check started
failing by about a per cent - ordinary statistics at 2000 events, and indistinguishable at a
glance from a filter that was quietly doing something. It now calls `G4Random::setTheSeed`
before each, exactly as a Geant4 user would with `/random/setSeeds`.

**What it made false.** The comment at the top of `g4/Randomize.hh` said, at length and with
emphasis, that reseeding the host engine "changes nothing about a run's result" and that this
was "a deliberate property, not an oversight". That was true when it was written and is now
precisely backwards: the host engine is where the primaries come from. A confident comment about
an invariant is a liability the moment the invariant moves, and this one would have cost someone
an afternoon in exactly the way it warned against.

**Generalization.** A property held by construction - "runs are reproducible because the device
keys on the seed" - can stop being true because the construction changed somewhere else
entirely. Nothing in the primary-generation rewrite mentions reproducibility. The check that
caught it was testing something else.

### S11: three checks written against an implementation, three pipeline failures on fixes

**Symptom.** Three consecutive full-pipeline runs failed, none of them on a bug. Each failed
because a check was looking for a symbol that a *fix* had just removed:

* `kSourceCdf` - the per-event sampler emitted into a generated project. It was wrong (it ran
  once per run, so it chose one source for the whole run). Correcting it deleted the symbol and
  the check failed on the correction.
* `AddSource` - the declarative multi-source API that replaced it. Making primary generation
  per-event made that API unnecessary, so it was deleted, and the check failed on that too.
* And in between, `run_.n_events` disappearing from BeamOn - not a stale check, but the same
  shape of problem: something that had always been there and that nothing named.

**Cause.** Each check was written by looking at what the code said at that moment and grepping
for a distinctive piece of it. That tests *this implementation*, not the behaviour the
implementation exists to produce, so it goes stale precisely when the implementation improves -
and the signal it gives is indistinguishable from a real regression.

**Fix.** The symbol check now looks for `GeneratePrimaryVertex`, which is the architecture
rather than a detail of the emitter, and its comment says why. The thing that actually
establishes correctness - that a two-source model fires both beams - is the numerical comparison
between the builder and the project it generated, which is stated in doses and cannot go stale
because it names nothing in the code.

**Generalization.** A cheap check for a symbol is worth having as an early signal, but it has to
name something whose *disappearance would itself be the bug*. `GeneratePrimaryVertex` qualifies:
a generated project without it produces no primaries. `kSourceCdf` never did - it was one way of
writing a correct thing, and there were others. The test for whether a check is written at the
right level: if somebody improved this code, would the check still be true? If the answer
depends on how they improved it, the check is aimed at the wrong thing.

### V3: an extension that changed what a Geant4 method means

**Symptom.** `examples/B1` reported 429 pGy in batch and 287 pGy when run interactively and told
to fire the same 10,000 events. Both numbers were self-consistent and neither looked wrong: the
batch run agreed with Geant4 to 0.015 sigma, and the interactive one reported a plausible dose
with a plausible rms. What was wrong was that they were the same example.

**Cause.** This gun carries optional spatial and angular distributions -
`SetBeamCrossSectionRectangular`, `SetAngularSpread` - which Geant4's `G4ParticleGun` does not
have. `SetParticlePosition` set the *centre* of whichever was configured rather than the
position. Geant4 has no such distinction, because there is nothing to be the centre of.

B1's vis.mac applied `/gun/beamRectangular 8 8 cm` so that pressing Run in the viewer fired what
the example fires - correct when the generator used the gun's cross-section. The generator was
then rewritten to Geant4's idiom, drawing its own spot in +-8 cm and calling
`SetParticlePosition`. The cross-section was still set. The spot was sampled twice, over +-16 cm
against a 20 cm envelope; a third of the beam missed the detector.

**Fix.** `SetParticlePosition` sets the shape to a point and `SetParticleMomentumDirection` sets
the angular distribution to unidirectional. To get a beam spot: position first, cross-section
second - which is the order every caller already used. `tests/test_gun_position.cu` samples 500
vertices and requires every one at the named point, and 5000 more to check that the documented
order still produces a beam that fills its half-width. Verified by reverting the fix: the first
vertex lands at (-0.59, 0.28) cm instead of (1, -2).

**Generalization.** The dangerous extensions are the ones that add a meaning to a method that
already has one. A new method with a new name cannot mislead; `SetParticlePosition` that means
"the centre of a distribution, if one happens to be set" is a trap laid specifically for code
copied out of a Geant4 example, which is the code most likely to exist. If an extension makes an
inherited method's behaviour depend on state the original API has no concept of, the inherited
method has to win.

The other half: **batch and interactive were never compared.** The pipeline runs everything in
batch. Every interactive path - the viewer's Run button, a macro that touches the gun - was
exercised only by a person, and two of the three bugs in this session lived there.

### V4: a lambda taken from the wrong return value, and a proton beam that never scattered

**Symptom.** None. `step_hadron` ran, the depth-dose curve looked right, the pipeline was green.
The proton beam simply did not undergo multiple scattering at all.

**Cause.** `wv_transport_xs` has two outputs and they are different quantities. Its *return* is
the transport cross section above the cut-off angle it is handed; its out-parameter `xtsec` is
the total single-scattering rate, and it also fills the per-element table the sampler picks an
atom from. The stepper called it with `cos_theta = 1` - which is what you pass to get `xtsec`
and the element table - and took the return value as the transport cross section. Above
`cos_theta = 1` there is no solid angle, so the return is identically zero.

`lambda_eff` was then 1e30. `wv_geom_path` saw a mean free path longer than any step, dropped
into single-scattering mode, and set the single-scattering rate to zero; `wv_sample_scattering`
saw a zero rate in single-scattering mode and returned the incoming direction. Every branch
behaved exactly as designed, and the composition of them was a proton that travelled in a
straight line.

**Fix.** `lambda_eff` comes from `wentzel_lambda`, which integrates over the whole angular range.
`wv_transport_xs` is still called, still with `cos_theta = 1`, and its return is now explicitly
discarded - it is called for `xtsec` and the element table, which is what the existing
`tests/test_wentzel_msc.cu` does too. That test was the model for the call and it does not make
this mistake; the stepper was written from the function signature instead.

**Generalization.** A function whose return value is meaningful for some arguments and
identically zero for others is a trap, and the trap springs silently when zero is a legal value
downstream. The three functions that consumed it - `wv_geom_path`, `wv_true_path`,
`wv_sample_scattering` - each had a correct, documented branch for "no scattering here", so
nothing could report the absurdity. Where a physics process can legitimately be absent, its
absence and its misconfiguration look identical, and only a test that asserts the process
*happened* can tell them apart.

### V5: the proton Bragg peak is 0.3% proximal of Geant4's, and what that is not

**SOLVED. See the amendment at the end of this entry, and V7 for the finding.** The list
below of things it is not was correct and complete; what it never contained was the thing it
was, because that thing was not a piece of physics but the grid the physics was tabulated on.

**Status: open, quantified, and bounded.** Recorded here rather than fixed because the search
narrowed it a long way without closing it, and the next reader should not repeat the search.

**Symptom.** `tools/compare_depth.ps1`, comparing the port and real Geant4 11.1.1 running the
*same source file* over the same water phantom:

    total       exact on both sides
    plateau     port 0.33% high per proton
    R80         port 77.57 mm, Geant4 77.80 mm - 0.23 mm, 0.30% short
    80-20 width port 1.15 mm, Geant4 1.13 mm - 2% wide

The plateau excess and the range deficit are one fact, not two: a curve shifted 0.23 mm proximal
reads high everywhere on the rising plateau, and integrating that over the entrance half gives
0.3%.

**What it is not.** Each of these was measured, not argued:

  - *The range table.* The port's proton range in water at 100 MeV is 77.604 mm against Geant4's
    77.562 - 0.05% **long**, the wrong sign and a fifth of the size.
  - *The stopping power.* 0.7256316 against 0.725605 MeV/mm, 0.004% high.
  - *The material.* Water built through `G4NistManager` and water built by `build_b1_materials`
    give dE/dx equal to eight digits.
  - *The fluctuation sampler.* `tests/test_fluctuation.cu` puts its mean within 0.014% of the
    mean handed in for exactly the cell this curve's plateau is made of (water, 100 MeV,
    0.5 mm), and its variance within 0.5% of `G4UniversalFluctuation::Dispersion`.
  - *Delta rays.* At 100 MeV in water the production cut (0.278 MeV) already exceeds the maximum
    transfer (0.229 MeV), so no delta ray can be produced at all. Raising the cut to 6.4 MeV
    changed the port's output not by one bit, which is how this was established.
  - *A step-size bias in the loss chain.* Walking a proton to rest with steps capped at 0.5 mm,
    2 mm and unlimited gives path lengths within 0.14% of each other and of the table.
  - *The multiple-scattering path conversion.* The measured detour is 2e-6 per 0.5 mm step at
    100 MeV, five orders of magnitude short of what would be needed.

**What is known about it.** The discrepancy grows with the phantom's slab thickness - the
curve centroids differ by 0.063, 0.098 and 0.266 mm at 0.25, 0.5 and 2 mm slabs - so it is tied
to the step length that the geometry imposes, and it extrapolates towards zero as the steps get
short. That is the thread to pull.

**Why it is being left.** 0.23 mm on a 77.6 mm range is inside the tolerance any proton
calculation is specified to, the sign and size are stable, and the four metrics are in the
pipeline (`tools/compare_depth.ps1`) with limits that would catch a regression long before it
mattered. What is *not* acceptable is not knowing, which is why the list above is written down.

### V6: twenty threads, one twentieth of the dose

**Symptom.** A new comparison of example B1 across three beams reported the port 20x high on
every one - including the gamma beam, which is the most validated number in this project and
had agreed to 0.015 sigma an hour earlier. The port's own numbers were right; the *reference*
was a twentieth of what it should have been.

**Cause.** The Geant4 install is a multithreaded build, and `G4RunManagerFactory` hands B1 a
`G4TaskRunManager` that uses every core. Each of the twenty workers runs its slice of the
events, calls `EndOfRunAction`, and prints a complete, well-formed

    --------------------End of Local Run------------------------
     The run consists of 100 gamma of 6 MeV
     Cumulated dose per run, in scoring volume : ... rms = ...

before the master prints the merged **Global** Run block. The parser took the first matching
line. Twenty workers, 2000 events, 100 events each: a factor of 20.

**Why it was caught.** Only because the comparison carried a **control** - B1's own 6 MeV gamma
beam, whose answer is already known - alongside the two new beams it existed to measure. A
proton and an alpha both reading 15x and 32x high are two new numbers with no history; there is
no way to tell a broken reference from broken physics. The same 20x on the gamma said
immediately that the fault was on the reading side, because the gamma physics had not changed.

**Fix.** Two, at different levels. The parser now searches from the last `End of Global Run`
marker rather than from the top. And `tools/compare_b1_beams.ps1` forces
`G4FORCE_RUN_MANAGER_TYPE=Serial`, which is the right thing regardless: a wall-clock number from
a threaded Geant4 is a whole machine against one GPU, which is not a comparison of two
transports. `ref/run/runb1.bat` now says both of these at the top, since it is the shared entry
point that will be reused.

**And then the fix silently did not apply.** The first attempt set the variable inside the
command line that launches Geant4 - `cmd /c "set VAR=Serial&& runb1.bat ..."` - which PowerShell
hands to cmd as one already-quoted string that cmd then splits in the middle of the variable
name (`'RCE_RUN_MANAGER_TYPE' is not recognized`). A short run before that had already printed
plausible doses, because by then the *parser* fix alone was enough to make them right. So the
serial setting appeared to work while doing nothing at all, and only the timing column - which
nobody had reason to distrust yet - would have carried the error.

The variable is now set with `$env:G4FORCE_RUN_MANAGER_TYPE`, which every child inherits, and
the script **asserts** it took: any line matching "starts on worker thread" in Geant4's output
fails the run. A configuration that is meant to be off should be checked to be off, not assumed
from the absence of an error message.

**Generalization.** A new measurement should carry an old one. Every number in a fresh
comparison is unfalsifiable on its own - the whole reason for making it is that nobody knows
what it should be - so the cheapest possible insurance is to run one case whose answer is
already on the record, through the identical path, and look at it first. It costs one extra row
in the table and it is the row that tells you whether to believe the others.

The other half is about output that is *correct*: nothing here was malformed or mislabelled.
Geant4 printed exactly what it should have, twice, for two different scopes, and the parser
picked the wrong scope. Text output whose meaning depends on which block it appears in is a
format that any regex will eventually read wrongly.

### Amendment to S3: the third time, and what actually reduces it

It happened again, on `build_all.bat` itself. A pipeline was running; a comment block was added
to the top of the file it was executing; cmd resumed at a byte offset that no longer meant
anything and reported `'...' is not recognized as an internal or external command`. Identical
to the first occurrence in S3, six months of intent to be careful later.

The reminder has not worked three times, so the mitigation is not a better reminder. It is
`tools/quick.ps1`, added for a different reason - iterating without a twenty-minute rebuild -
which removes most of the *opportunity*. A subset run takes fifteen seconds, so the window in
which the tree is being compiled and might also be edited is fifteen seconds wide instead of
twenty minutes, and there is no reason to be editing anything during it.

The rule that follows, and it is about sequencing rather than care: while a full pipeline is
running, only create files nothing includes yet. Not "edit carefully" - edit nothing that is on
its compile path, including the pipeline script.

### S12: the same compiler crash, the opposite fix, and a bisect that lied

`ptxas` (CUDA 11.6, `-arch=sm_86`) died with `0xC0000005 ACCESS_VIOLATION` compiling
`run_step_hadron<double, kAlpha>` after `G4IonFluctuations` was wired into the stepper. No
diagnostic, no line, no symbol - just a Windows access violation from the assembler.

This had happened once before, in S-something-earlier: adding lepton fluctuations killed ptxas
on `b1_gpu_sched.cu`, and the fix was `__noinline__` on `sample_fluctuation`. So the first
thing tried here was more of the same - `__noinline__` on `ion_dispersion`,
`ion_fluctuation_factor`, `g4pow_pow_a`, and a restructure that split the sampler in two so one
`__noinline__` function would stop calling another. **None of it worked, and the last of them
made the code worse for a reason that turned out to be fictional.**

Then a bisect: stub `ion_fluctuation_factor` - compiles. Stub the second half of it - dies.
Disable the Yang table lookup - compiles. Replace `g4pow_pow_a` with `std::pow`, table still
present - dies. Replace the table lookup with four literals - compiles. That reads as an
unambiguous verdict: *the 384-element table is the trigger*. A generated header, a macro so the
values are written once, a `__constant__` device copy and a host copy, and a long comment
explaining why this one table is shaped differently from every other table in `src/data`.

It still died.

The actual cause was `__noinline__` - the fix from last time. With every function in the ion
path left ordinary and inlinable, and the table back in the plain `static const` form every
other table uses, it compiles. `sample_fluctuation` still carries `__noinline__` from the
earlier fix and that is still fine; what ptxas could not survive was *this* kernel with *those*
functions out of line.

**Two things went wrong, and the second is the one worth keeping.**

The first is ordinary: a known fix for a known symptom was applied to a symptom that only
looked the same. Same assembler, same crash code, same phase of the build - different cause.

The second is that the bisect was *sound* and its conclusion was *wrong*. Every step of it was
run correctly and reproducibly. But every step was run with `__noinline__` still in place, so
what was actually being measured throughout was "does this kernel survive with these functions
out of line **and** this much work in them" - and every stub reduced the work. The table was
the largest single contributor to the work, so removing it passed. It was a genuine cause of
crossing the threshold and not the reason the threshold was there.

Had the elaborate `__constant__` version been kept, the repository would now carry a generated
header in a shape nothing else uses, a macro indirection, and a forty-line comment confidently
attributing a compiler crash to the wrong thing - permanently, because nobody re-tests a
workaround that appears to be working.

**Generalization.** A bisect on a compiler crash finds *a* thing that pushes it over a
threshold, not the thing that put the threshold there. Two rules follow. First: before
believing a bisect, remove the thing you already changed - the earlier workaround, the flag you
added, the restructure - and re-run it; if the crash goes away, you were bisecting the wrong
axis. Second: when a fix is a workaround for a compiler bug rather than a fix for a defect,
delete it and confirm the failure comes back before writing the comment that explains it. The
comment is the expensive part. A workaround whose stated reason is wrong is worse than no
comment at all, because it is an instruction to future work to preserve something pointless.

The corollary for `__noinline__` specifically: it is not a general remedy for ptxas crashes in
this project. It fixed one and caused another. Treat it as a thing to try *and to try
removing*, and record which kernels currently need it - today that is `sample_fluctuation` and
nothing else.

### V7: the reference is a table, not a model - and what that means for V5

Comparing a stopping power against `G4EmCalculator::ComputeDEDX` is comparing against Geant4's
*model*. Transporting a particle is not. `G4VEnergyLossProcess` never evaluates a model during
a step. At initialisation it builds a dE/dx vector on a log grid and from then on it
interpolates:

```
G4EmParameters:  MinKinEnergy 100 eV,  MaxKinEnergy 100 TeV,  nbinsPerDecade 7
                 -> 85 points over twelve decades
G4LossTableBuilder: splineFlag = true  -> cubic spline between them
```

and the range table is that vector integrated (`BuildRangeTable`), on the *same* grid:

```
range(0) = 2 * E(0) / dedx(0)
range(j) = range(j-1) + sum of 100 midpoint sub-steps of de / dedx_spline(e)
```

This port evaluates the models exactly, on 256 log points from 1 keV to 10 GeV, and integrates
with 16 trapezoid sub-steps. The seed's factor of two was transcribed years ago - it is the one
piece of that algorithm already here, and finding it cost a day (see the comment in
`build_hadron_range_table`). The grid it sits on was not.

**Measured**, by building this port's own models on Geant4's grid with Geant4's spline and
Geant4's integration and comparing both against the oracle:

| | this port's grid | Geant4's grid |
|---|--:|--:|
| mu- range, water | 52.5% | **2.1%** |
| mu- range, air | 51.6% | **6.0%** |
| proton dE/dx, water | 0.415% | 0.160% |
| proton range, water | 0.131% | 0.080% |
| proton dE/dx, air | 1.337% | 1.049% |

Two things follow.

**The proton's residual 1.3% was misattributed.** Its worst point sits just above 2 MeV, where
G4BraggModel hands over to G4BetheBlochModel; Geant4's 7-bin-per-decade grid straddles that
discontinuity and its spline rings across it. `tests/test_hadron_range.cu` said, for a long
time, that this was "the residual ICRU90 omission ... measured rather than assumed". It was
neither measured nor the ICRU90 omission. It was a plausible sentence attached to a number
nobody had taken apart, and it survived precisely because it sounded like it had been.

**The 1 keV floor is a species-dependent error.** For a proton, the range below 1 keV is a
negligible part of the range anywhere it is used. For a muon it is not, and the range comes out
52% wrong at the bottom of the table. Nothing about the choice of 1 keV was wrong for the
particle it was chosen for, and it was never re-examined when the species set grew - which is
the general failure, not the number.

#### What this says about V5

V5 records the proton Bragg peak sitting 0.232 mm (0.3%) proximal of Geant4's and lists what it
is not. Add one line to that list, and it changes the question:

```
Geant4's own range table, 100 MeV proton in water   77.5621 mm
this port's transported R80                         77.566  mm
Geant4's transported R80                            77.798  mm
```

The port's *transport* lands on Geant4's *table*, to four digits. Geant4's transport overshoots
its own table by 0.24 mm. So the peak is not proximal because this port's range is short - the
range agrees with the reference's own tabulated answer better than the reference's own
transport does. Whatever V5 is, it is something that makes a Geant4 proton travel 0.3% further
than its range table says it should: the along-step algorithm, the step limiter, or the
fluctuation's effect on the mean range. It is not the stopping power and it is not the
integral.

**Generalization.** When a reference has both a model and a table built from it, check which
one the reference itself uses at the moment being compared. A model comparison and a transport
comparison can both be right and disagree, and the difference is not error in either. More
generally: a discrepancy with an explanation attached is not the same as a discrepancy with an
explanation *tested*, and the wording that makes them indistinguishable in a comment is "the
cost of X, measured rather than assumed". Write down the measurement or write down that there
isn't one.

### Amendment to V5: solved, by V7

Rebuilding the hadron dE/dx and range tables on Geant4's own grid - 100 eV to 100 TeV, 7 bins
per decade, cubic spline, and `G4LossTableBuilder::BuildRangeTable`'s 100 midpoint sub-steps
against that spline instead of 16 trapezoids against the model - moved the depth-dose
comparison by more than an order of magnitude:

| 100 MeV protons in water, 0.5 mm slabs | before | after |
|---|--:|--:|
| R80 against Geant4 | -0.232 mm | **-0.014 mm** |
| plateau dose | +0.324% | **+0.052%** |
| distal 80-20 width | +0.025 mm | **+0.004 mm** |

Everything V5 ruled out stays ruled out. The range table was not wrong in the sense V5 was
looking for - every stopping power in it agreed with Geant4's *model* to a fraction of a per
cent, and that is what was checked, repeatedly, for months. It was wrong in a sense nobody had
thought to check: it was a different table from the one Geant4 transports on, built from the
same physics on a finer grid. Better physics, and 0.3% of a Bragg peak away from the answer.

**The general lesson, and it is not "check the grid".** V5 accumulated a long list of things
the discrepancy was not. Every entry on that list was earned - the range table, the stopping
power, the material construction, the fluctuation's mean and variance, the delta rays, the step
bias, the MSC path conversion - and the list was correct. What it could not contain was a
category error: every item on it was a piece of *physics*, and the cause was a piece of
*numerics* in the reference. A list of eliminated causes is only as good as the space it is
drawn from, and the space is invisible from inside it.

What finally exposed it was not looking harder at the proton. It was adding muons, whose range
came out 52% wrong at the bottom of the table because a 1 keV floor that is invisible for a
proton is most of a muon's range. The proton's 0.3% and the muon's 52% were the same defect at
two masses, and only the second was large enough to be undeniable. **Widening the species set
was a better debugging tool for the proton than any amount of further work on the proton.**

### V8: a constant that had been wrong for electrons since the model was written

`G4UrbanMscModel::ComputeCrossSectionPerAtom` needs the Bohr radius. The port had:

```cpp
constexpr real_t bohr = real_t(0.5291772109e-7);   // mm
```

which is the CODATA value to ten digits, and wrong, because CLHEP does not use CODATA. It
builds `Bohr_radius` from `electron_Compton_length / fine_structure_const`, both derived in
turn from its own `hbarc` and `elm_coupling`. The two differ by 2e-8 relative. `epsfactor`
carries the radius squared and the small-eps branch squares `epsfactor` again, so the cross
section came out 7.7e-8 high.

`src/core/units.cuh` already had `bohr_radius()` derived exactly the way CLHEP derives it,
with a comment explaining why. This one call site had a literal instead.

**What is interesting is not the constant. It is that nothing could have found it.**

`tests/test_msc.cu` says so in its own opening line: *"There is no Geant4 table to diff this
against, so these are internal consistency checks on the transcription rather than an external
validation."* And it is a good test - it pins tau, theta0, the xsi polynomial, xmean1, xmean2,
prob, qprob and both branch samplers at once, by checking that the sampled `<cos(theta)>`
equals the value the construction is built to produce. Every one of those relationships holds
exactly as well with a Bohr radius that is 2e-8 out, because the error is *common to both sides
of the identity*. A self-consistency test cannot see a scale error in its own input.

It was found because the species work needed `ComputeCrossSectionPerAtom` for alphas, which
meant dumping it from Geant4 for the first time - and the electron came along in the same dump
because the eight-particle table was cheaper to write than a one-particle one. The failure was
7.65e-8 on all eight species simultaneously, which is what a shared constant looks like and
what a physics error does not.

**Generalization.** An internal-consistency test and an oracle test are not two strengths of
the same thing; they fail on disjoint sets of bugs. A self-consistency check finds errors in
*relationships* and is blind to errors in *inputs*; an oracle finds errors in inputs and says
nothing about whether the sampler is coherent. Where a component has only the first kind,
write down that it has only the first kind - and treat "no external reference exists" as a
statement about effort, not about possibility. The reference here was a forty-line block in a
dumper that already existed.

The narrower rule: a derived physical constant should be derived once, in `units.cuh`, and
never spelled as a literal at a call site - even a literal that is more accurate, because
matching Geant4 is the requirement and Geant4's constants are its own.

### V9: Rayleigh scattering deflected nothing, and the story I told about finding it was false

`G4RayleighAngularGenerator` needs `fFactor = 0.5*(cm/(h_Planck*c_light))^2`. The port had:

```cpp
// h*c = 1.23984193e-18 MeV*mm; cm = 10 mm.
return real_t(0.5) * (real_t(10.0) / real_t(1.23984193e-18))
     * (real_t(10.0) / real_t(1.23984193e-18));
```

`h*c` is `2*pi*hbarc` = 2*pi * 197.327e-12 = **1.23984e-9** MeV*mm. The literal was nine orders
of magnitude too small, so `fFactor` was 1e18 too large, so in

```cpp
cost = 1.0 - x/(b*xx);          // xx = fFactor * ekin^2
```

the quotient underflowed to zero and `cost` came out exactly 1.0 on every scatter. Every
Rayleigh interaction in every run this port has ever done consumed a step and turned the photon
by nothing at all.

It is derived now - `units::twopi<real_t>() * units::hbarc<real_t>()` - which is what V8 was
written about, one entry earlier in this same file, about a different hand-typed constant in a
different model. Two for two. The rule is not "be careful with constants", it is that a derived
physical constant has exactly one home and a literal at a call site is a defect on sight.

#### Why it survived

Coherent scattering transfers no energy. It only turns a photon, so a completely dead
deflection changes a dose by a fraction of a per cent and nothing else. `test_rayleigh`
validated the *cross section* against G4EMLOW's epics2017/rayl to better than 1e-6 across 3,400
points and never touched the angular distribution, because there was no test of the angular
distribution - `rayleigh_angular_tables.cuh` was the one baked table in the port with neither
an extractor nor a test, which `tools/refresh_tables.sh` had said in as many words.

The bug was found by going and writing that missing test. The first run reported 4.4e9 standard
errors.

Worth noting what was *not* wrong: `tools/extract_rayleigh_angular.sh`, written at the same
time, reproduced all 909 hand-transcribed fit parameters exactly. The table was perfect. The
constant feeding the sampler that reads it was not.

#### The part that matters more

On seeing the fix, I wrote that the pipeline had been reporting this all along, and quoted:

```
rayleigh off: dose10k 429.8201 pGy (+0.0017, 0.0 sigma), steps -9104 (-0.35%)
```

with the line "a process whose removal changes the answer by zero sigma is a process that is
not doing anything", and added that I had read past it every run of the session.

**That was wrong.** With the bug fixed, the same toggle reads:

```
rayleigh off: dose10k 429.8198 pGy (-0.0000, 0.0 sigma), steps -9181 (-0.35%)
```

Zero sigma either way. The toggle is not sensitive to whether Rayleigh deflects, because at
6 MeV in this geometry Rayleigh is 0.35% of steps and the scoring volume is large enough that
turning photons inside it changes nothing measurable. The line was never evidence of the bug.
B1's dose moved by 0.0003 pGy across the fix - 427.408 pGy, 0.0187 sigma to 0.0190 sigma.

So the diagnosis was right and the *story* about the diagnosis was invented: a satisfying
narrative - the evidence was there all along, I just wasn't reading - attached to a number that
does not support it, and asserted without running the one check that would have refuted it. The
check was free. The pipeline had already printed the post-fix line by the time I wrote the
claim.

This is S12 again, in the same session, one entry after S12 was written: a conclusion that
felt explanatory, was consistent with what I had in front of me, and was never tested against
the case that would have falsified it. S12's rule was "before believing a bisect, remove the
thing you already changed and re-run it". The general form is broader and this is the third
instance of it in this file: **an explanation that accounts for the evidence is not thereby
supported by it.** Ask what the evidence would look like if the explanation were false. Here it
would look identical, and one grep would have shown that.

#### What this says about the process-toggle test

It is a good test and it did not fail here - it was never designed to catch this. It measures
whether a process contributes *dose*, which is exactly the right question for Compton, pair
and photoelectric, and structurally the wrong one for a process that transfers no energy. The
same blind spot covers anything else whose only observable is a direction.

Rayleigh matters where a photon's direction after scattering matters - scatter fractions,
imaging geometries, anything below about 100 keV where its share of the cross section is not
0.35%. B1 at 6 MeV is not that problem, which is why B1 could not see this and why B1 passing
is not evidence that it is fixed. `tests/test_rayleigh_angular.cu` is.

### V10: a threshold that sat on its own noise floor, and a story about it that was wrong twice

`compare_project.ps1` requires the builder's selftest and the project the builder generated
from the same model to agree within 6%. This session both numbers moved and it failed. Nothing
was wrong with either program.

#### The check could not tell agreement from disagreement

It ran 200000 events a side. At that size the two numbers are two independent Monte Carlo
samples and the spread between them is the whole story:

| | dose1 | cells |
|---|--:|--:|
| relative uncertainty at 1M events, from the rms the project itself prints | 1.39% | 0.98% |
| one side at 200000, scaled by sqrt(5) | 3.12% | 2.18% |
| two independent sides, x sqrt(2) | **4.4%** | **3.1%** |

Against a 6% tolerance that is 1.36 sigma on `dose1`: roughly a one-in-six chance of failing on
any given pair, every time the pipeline ran. Every pass it had ever given was luck rather than
evidence, and the failure that finally exposed that was not a defect in anything it was
watching. Measured end to end, the two sides differ by 6.4% at 200000 histories, 0.80% at a
million and -0.08% at four million - 1/sqrt(N), which is what two independent Monte Carlos
agreeing looks like.

Both sides run a million now. That puts the noise near 1% and leaves the tolerance three sigma
away, while still catching what the check exists for: a generated project that fires one of a
two-beam model's sources, which is wrong by tens of per cent.

The rule, and it is not about this check: **a threshold is not a number you pick, it is a number
you compare against the spread of the thing being thresholded.** Nobody had ever measured that
spread. It took two minutes - the program prints its own rms. V1 is the same lesson from the
other side, a comparison whose statistics were too weak to mean anything, and it cost a day.

#### And the explanation I gave for the failure was invented

I wrote, and put in a summary of this session's work, that the builder had been *silently
dropping tracks* - that its fixed per-species buffers overflowed, that `RunStats.overflow` went
unchecked, and that the dose had been under-reported by 8.6% as a result. It was a good story:
the per-species buffers really were replaced this session by one species-agnostic pool with a
throttle, precisely because partitioned capacity is the wrong shape, and a number that moved
8.6% the moment that landed fits perfectly.

Built at HEAD and run on the same model, the old builder reports:

```
  1458480 track steps, 114 iterations, peaks g/e/p 65536/8735/558
selftest: compare dose1 3483.286491 MeV over 200000 events
selftest: compare cells 8202.527743 MeV over 200000 events
```

No overflow, no abandonment - `g4builder_panels.inc` prints a WARNING for either and printed
neither. The 65536 peak that looks like a saturated buffer is `SetBatchSize(65536)`: every
primary of the batch alive at iteration 0, against a gamma capacity of twice that. And the two
scorers moved in **opposite directions** - `dose1` +8.6%, `cells` -2.6% - which is the signature
of noise and not of loss. Loss moves everything one way.

What actually changed the numbers is the RNG key fix from earlier in the same session (the key
was batch-local, `seed ^ i`, so every batch replayed the same streams; it is the global track
index now). That re-randomised every stream, which makes HEAD and now two independent samples of
the same model. 8.6% on `dose1` is 2.0 sigma of the spread tabulated above and 2.6% on `cells`
is 0.85. Ordinary.

The "8.6%" was worse than a coincidence: there is an 8.6% in this session's notes, and it is the
memory overhead of the five species index arrays against a 232-byte track. I attached a number
from one part of the work to a conclusion in another because they matched to one decimal place.

This is V9 again, one entry later: **an explanation that accounts for the evidence is not
thereby supported by it.** V9's version was a satisfying narrative attached to a number that did
not support it. This one is a satisfying narrative attached to a number from somewhere else
entirely. Both times the refuting check was cheap - here, build the old binary and run it, which
is twenty minutes of nvcc and no thought at all. Both times I wrote the conclusion first.

The throttle does need to be shown lossless, and separately is: the same model reports
`dose1 3783.696869 MeV` identically at 4, 8 and 16 live slots per event, deferring 49152
track-steps at the smallest and none at the largest. That is what convergence in the pool size
looks like, and it is evidence about the throttle. It was never evidence about HEAD.

### V11: an odd number of track slots, and a fault reported four hundred lines from its cause

`-live 2.5`:

```
track pool: 7396897 live slots per side (2.5 per event), 3.25 GB total
CUDA error misaligned address at D:/g4gpu/src\host/transport_run.cuh:96
```

Line 96 is a `cudaMemcpy` inside `DeviceTracks::count()`. It has nothing to do with the fault -
it is merely the first synchronising call after the kernel that faulted, which is where an
asynchronous device error surfaces. The fault is in `track_arena_half_bytes`.

The engine allocates both halves of the ping-pong as one block and puts the second at
`base + track_arena_half_bytes(pool)`. That number is therefore the alignment of every array in
the second half, and it was not rounded up to the slab's alignment. A track slot is 236 bytes,
so:

| pool | `236 * pool + slack` | mod 8 |
|---|--:|--:|
| 2 500 000 | 590 053 760 | 0 |
| 2 499 997 | 590 053 052 | 4 |

Every `double` in the second half of the arena was then at a 4-byte offset. `TrackSlab::take`
aligns each array to 256 bytes and was doing its job perfectly; it was the base handed to it
that was wrong.

#### Why nothing had ever seen it

`pool = batch * live_per_event`. The default is 4.0, and 236 times any even number is a multiple
of 8, so **every configuration this pipeline has ever run landed on the lucky case**. Only a
fractional slots-per-event makes the product odd, and only `-live` can ask for one. The
alignment was a property of the inputs that had been tried, not of the code.

#### How it was found, which is the part worth keeping

Not by a test. I was writing a commit message and it contained the line "identical dose at 8.0,
4.0 and 2.5 slots per event". That had been measured earlier in the same session and was true
when it was measured. I ran it again before quoting it, because V10 - two entries earlier, the
same afternoon - is about asserting numbers I had not checked. The 2.5 case did not produce a
dose at all.

**A measurement is about a version of the code, and the code moved.** The claim was not
fabricated, which makes it the more dangerous kind: it was true, it stayed in my notes, and
nothing about restating it felt like an assertion.

#### What now checks it

`tests/test_track_arena.cu`, host-only, sweeps odd pools, even pools, powers of two and the two
sizes the automatic batch sizer actually produced on this machine, and asserts three things:
that the half size is a multiple of `TrackSlab::kAlign`; that carving both halves the way the
engine does yields nothing misaligned for a `double`; and that the measured size is within its
documented slack of what the carve used, which is the invariant that "measure and carve agree"
rests on. Reverting the fix makes it fail at pools 1, 3, 7, 63, 65, and on down the list.

`build_all.bat` also runs `g4dose -n 200000` at 8.0, 4.0 and 2.5 slots per event and requires
one dose, through `tools/compare_pool.ps1`. That is a different claim from the arithmetic - it
is the throttle's claim, that deferring a track changes the order of work and nothing else - and
the odd pool comes along with it for free.

#### Amendment to V11: the fix, and the second fix the first one needed

`track_arena_half_bytes` is exact now rather than estimated. `track_buffer_bytes` already
dry-runs the same allocator the carve walks, so the arena size is that number and there is no
per-slot arithmetic and no slack term to be misaligned. Every array is taken with `align_up`, so
a sum of them is a multiple of `kAlign` by construction: the fault V11 is about cannot recur by
arithmetic, only by someone reintroducing an estimate, and `tests/test_track_arena.cu` asserts
`half == used` exactly so that would fail immediately.

Which would have been a clean improvement, except that being exact very nearly bought silence.
A capacity is an `int` everywhere - `TrackBuffer`, `allocate_track_buffer`, the carve. The
estimate computed `per_slot * pool` in `size_t`, so a pool too large for an `int` came out
enormous and the memory check refused the run. An **exact** measurement of a *truncated*
capacity agrees perfectly with an equally truncated carve, and the run proceeds with a pool that
is not the one it was asked for and nothing anywhere disagreeing.

So the function saturates: a pool over `INT_MAX` reports an impossible size, which makes the
batch sizer's bisection walk down instead of up, and `Upload` refuses an explicitly-set batch
that lands there rather than truncating it. Neither is reachable today - the sizer bisects to at
most 4194304 events, and 4194304 x 4 slots is a fifth of `INT_MAX` - which is exactly why it is
worth writing down: the guard exists for the configuration nobody has tried yet, and its absence
would have shown up as a wrong answer rather than as a failure.

The general form, and it applies beyond sizing: **a more accurate number is only an improvement
where it does not remove a disagreement that was doing work.** The estimate's inaccuracy was
load-bearing at the top of its range. Replacing it meant replacing that too.

### V12: two runs in one process that were not two samples

Geant4's behaviour, and what a user relies on without thinking about it: run `B1.exe` twice and
you get the same number, because that is what makes a result quotable. Issue two `/run/beamOn`
in one session and you get two different numbers within noise, because that is what makes the
second run worth doing.

This port had the first and only half of the second. Every track's shower was drawn from a
counter-based stream keyed on `(seed, index within the run)`, and the seed never changed between
runs - so run two used **exactly the same shower streams as run one**. What differed was the
primaries: CLHEP's engine is a static constructed once per process, so B1's two random draws per
event carried on and the second run fired a different beam spot into an identical shower.

Two things follow, and the second is worse than the first:

- A generator that draws no host random numbers **repeated its run bit for bit.** Proved by
  neutralising B1's two `G4UniformRand()` calls: 1.38967 nanoGy, twice, identical.
- Even with a random beam spot the two runs were not independent samples. Run two's event *i*
  reused run one's event *i* shower. Averaging them reduces less variance than averaging two
  real samples, and nothing says so.

`G4RunManager::stream_pos_` fixes it: a position in the seed's stream, zero at construction,
advanced by one per primary, folded into the key alongside the batch offset. So the streams carry
on across runs exactly as CLHEP's do on the host, and a fresh process starts at zero and replays.

#### Why B1 could not see it

B1's gun draws two random numbers per event. That was enough to make its second run differ,
which is what anyone looking would have checked, and it made the port's behaviour look correct
while resting entirely on the generator. **A test that passes because of the test program's
incidental properties is not testing the port.** The same blind spot covers any port behaviour
that a randomised primary can mask.

`tools/check_run_sequence.ps1` now asserts four things rather than one: that two runs in a
process differ, that they differ only within statistics, that a second process replays both
exactly, and that two runs of N sum to one run of 2N. The last is the sharp one - an offset that
jumped too far would still differ and still replay, and would silently skip part of the stream.
It comes out at 1.2e-6 relative, which is the printed precision of the number being compared.

#### And the threshold on the second of those, which I got wrong first

I wrote "differs by more than 5%" for the within-statistics check. B1 at 20000 events has a 2.1%
standard error, so two runs are 2.9% apart on average: the check would have failed about one run
in twenty for no reason, and the first time I ran it against the real B1 it did - 5.71%, which is
1.93 sigma and entirely ordinary.

That is V10, two entries earlier, in a check written *because of* V10, by the person who wrote
V10, on the same day. The number was available: the run prints its own rms on the same line as
the dose the check was already parsing. It reads the rms now and the threshold is four sigma.

The lesson V10 states is "a threshold is not a number you pick, it is a number you compare
against the spread of the thing being thresholded". Knowing it did not help. What would have
helped is a habit: **when writing a threshold, parse the uncertainty that is already in front of
you, or go and measure it - never type a round number.** A round number in a comparison is a
defect on sight, in the same way a hand-typed physical constant is (V8, V9).

### V13: a trajectory that ended where the track did not

Reported from the viewer: charged tracks draw as rows of disconnected dashes while gammas draw
as continuous lines. `step_lepton` did this:

```cpp
p.pos = p.pos + geom_step * p.dir;      // advance along the direction
traj.add(pos_before, p.pos, ...);       // record the segment
... msc sampling ...
p.pos = p.pos + d;                      // the MSC lateral displacement
```

The segment was recorded, and then multiple scattering moved `p.pos` sideways. The next step
began from the displaced position, so segment N ended at **A** and segment N+1 started at
**A + d** - a gap of exactly the lateral displacement, on every charged step.

**The transport was never wrong.** The displacement is applied to the track state and the next
step proceeds from there; the dose, the step count and the agreement with Geant4 are all
untouched by the fix. What was wrong was the *record*: it drew a line to a place the track did
not end. Gammas were unaffected because they have no MSC, and hadrons because
`kHadronLateralDisplacement` is false for heavy particles - which is why the one visibly broken
thing was the one particle species that scatters.

Recording after the displacement fixes it, and is also what Geant4 draws: a step is a straight
line between its two step points, and the post-step point is post-`AlongStepDoIt`.

#### Why this is not merely cosmetic

The viewer exists to make transport bugs visible, and it has done it repeatedly - R1 was a
ray-marcher re-finding a surface, G5 a normal that could only point 26 ways, G6 two volumes on
one layer. Every one of those was found by looking at a picture and noticing something
discontinuous.

A picture with a built-in discontinuity is one nobody can read a real discontinuity out of. The
defect was not in what the transport computed; it was in the instrument used to check the
transport, which is the worse place for it to be.

#### What it says about where tests were pointed

Every check in the pipeline reads a number: a dose, a step count, a sigma, a ratio. Not one of
them reads the geometry of a trajectory, so a systematic error in the recorded path was invisible
to all of them and stayed that way until somebody looked at the screen. The suite is well
defended against wrong numbers and undefended against wrong pictures.

`kind` and the two endpoints were all a segment carried, so segments could not even be grouped
into tracks - which is why this went unchecked rather than merely unchecked-for. A track id now
rides with each segment so the chaining can be asserted: for one track, every segment's start
must be another segment's end, exactly, since consecutive segments share a `float` converted from
the same `double`.

### V14: three GUI defects, all of them a rule stated in the wrong terms

Reported from using the builder. Different symptoms, one shape.

#### A dropdown inside a pop-up rendered behind the pop-up

`DrawOpenSelect` painted the open list once a frame, after the panels, because a Select cannot
draw its own list - everything the panel draws afterwards would cover it. The frame said so:

```
// After every panel, so an open dropdown list is not painted over by whatever the panel
// drew below it - and before the menu bar and the pop-ups, which are further forward still.
```

Both halves are right for a dropdown declared in a panel. For one declared *inside* a pop-up -
the voxel import dialog has one - "before the pop-ups" is exactly wrong: the list was painted at
the panel layer and the pop-up that owned it then painted over it.

The rule is not "after the panels", it is **after everything at the widget's own layer**. A
Select now records `Context::layer` when it opens and `DrawOpenSelect(ctx, layer)` draws only its
own; the frame calls it once per layer. `LayerScope` sets and restores the layer, because the
failure from forgetting to put it back is a dropdown that renders behind something occasionally.

#### More than 64 material indices "looked continuous"

`ClassifyVoxels` stopped at 64 distinct values and refused:

```
more than 64 distinct values: this looks continuous, not segmented.
  Re-import as continuous (HU) if it is a CT.
```

A segmented phantom can have as many organs as it likes; 200 is an ordinary ICRP model. The
count carries no information about what the values mean, and the user had *already answered*
that question - `kDiscrete` is set because they chose "material indices". The code overrode an
explicit answer with an inference, and its suggested remedy would have banded indices as if they
were densities.

What does distinguish an index volume is that indices are **integers**. That is the test now;
the count is only a resource limit, at 4096, and says so. The scan became a direct-address
bitmap over the integer range, because the linear search this had cost 64 comparisons a voxel at
the old cap and would have cost 4096 at the new one - over 1e8 voxels that is not a slower
import, it is one that never finishes.

Nothing had ever tested this function, though it is pure host arithmetic over a vector.
`tests/test_voxel_import.cu` now covers it, and fails at 200 indices with the cap put back.

#### Track colours were by species, so the species nobody listed were wrong

Geant4's default trajectory model is `G4TrajectoryDrawByCharge`: negative red, neutral green,
positive blue. This port had an enum named `kKindGamma`/`kKindElectron`/`kKindPositron`, and:

```cpp
case ParticleType::kElectron: kind[slot] = kKindElectron; break;
case ParticleType::kPositron: kind[slot] = kKindPositron; break;
default:                      kind[slot] = kKindGamma;    break;
```

Every proton and every alpha - both positive - fell to `default` and drew as the neutral class.
And the builder's own defaults had electron on light blue and positron on orange, so in the GUI
a negative particle drew in the positive colour and a positive one in no convention at all.

Both are the same mistake: **a rule about charge, restated in terms of species.** Restated in
terms of charge there is nothing to enumerate and nothing to forget - the class comes from
`particle_def(t).charge`, the same table the physics reads, so a species added to that table is
coloured correctly the day it is added. The enum, the palette and the GUI labels all say charge
now, because the species names are what made "electron = light blue" look unremarkable.

#### The shape they share

Each of the three was a rule expressed in the wrong vocabulary: draw order as "after the panels"
rather than "after my layer"; segmentation as "few values" rather than "whole numbers"; colour as
"which particle" rather than "what charge". In each case the wrong vocabulary was *right for the
cases in front of the author* and silently wrong for the first case outside them - a pop-up with
a dropdown, a phantom with 200 organs, a beam of protons.

None of the three could have been caught by any check here, because all three are about what
appears on a screen and the suite reads numbers. Two of them are testable anyway and now are;
the draw order is not, and is guarded by a comment and a scope guard instead.

#### Amendment to V14: the dropdown is now photographed, and a second bug in the same widget

The layering fix could not be asserted by anything in the suite - it is a question about what is
in front of what on a screen - so the builder's selftest now holds the voxel dialog up with one
of its dropdowns **open** and photographs it, as `out/g4builder_dlg_voxel_open.png`.
`build_all.bat` requires the file to exist, alongside the five dialog captures that were already
there for the same reason: a dialog whose text runs past its frame looks correct to every check
that reads a number.

It is a photograph and not an assertion, and the difference matters. It fails only if the
capture stops happening; a list drawn behind the pop-up again would produce a picture that looks
like the closed one, and catching that needs somebody to look. That is still worth having -
before this there was no artifact in which the bug was even visible.

Opening a list by id alone turned up the second defect. `Select` recorded its rectangle **only
on the frame the list opened**, so a panel that scrolls, a splitter that moves or a window that
resizes while a list is open left the list pinned to where the button used to be. Nobody had
reported it, because it needs a dropdown open across a layout change; it is the same class of
error as the layering - state captured once when it needed to be refreshed - and the same edit
fixes both, since the geometry is now re-recorded every frame the list is drawn open.

### V15: an imported phantom was a grey box, and the reason was one style per volume

Reported: importing a 225x225x500 segmentation with 261 material indices shows only the outer
box; making that box transparent makes everything disappear rather than revealing the anatomy.

Both halves are the same fact. A voxel volume is **one volume** in the flattened geometry, so it
got **one `VolumeStyle`** - one colour, one opacity - and `render_geometry` shaded its bounding
box and drew gridlines on the surface. The 261 classes an import produces were stored, editable
in the GUI, and connected to nothing that draws. Turning the box transparent removed the only
thing being drawn, which is exactly what "everything just disappears" is.

The fix is a cell march: `VoxelWalk` in geometry/voxels.cuh steps the grid cell by cell, and the
renderer composites each cell's class colour front to back with the same accumulation it already
used for whole volumes. A class at zero opacity contributes nothing, which is what lets index 0 -
air, background, outside the patient - get out of the way.

#### Why the picture cannot come from the transport's data

The cells already on the device hold a **material** index, and that was the tempting thing to
colour by. It does not work, and the reason is the workflow: a freshly imported phantom has no
materials assigned - the log says "assign a material to each class in the list to transport it" -
so every cell is -1 and every cell is identical. **You assign materials by looking at the
picture**, so a picture that needs the assignment first is no use.

So there is a second per-cell array, holding the class, render-only and parallel to the material
one. It is 50 MB for a 25-million-cell phantom, which is 0.6% of this card against being unable
to see what you are assigning.

#### Shading by the face, which is what makes it anatomy

Compositing colours alone gives fog with an outline: every cell contributes the same shade, so no
surface reads. `VoxelWalk` reports the axis of the boundary each cell was entered through, and
that axis is the face normal - so the boundary between two classes lights like a surface. That is
the difference between a colour field and something you can recognise an organ in.

#### The six billion comparisons underneath it

`build_scene.hh` mapped each cell's value to its class by **scanning the class list**, inside the
loop over cells. At the 64 classes the importer used to cap at, on a small phantom, nobody
noticed. 25.3 million cells times up to 261 classes is 6.6e9 comparisons, and it runs on every
scene rebuild - which is every edit. Raising the class cap (V14) without this would have turned
an import into an apparent hang: the two changes had to land together, and the fact that they
did is luck rather than design, because the cap was raised first.

It is a direct-address map now, which the importer's own requirement makes possible: indices are
whole numbers over a modest range, so the map is an array indexed by value. The continuous case
keeps the scan - nine bands, nine comparisons.

#### The second DDA, and what closed it

For one change there were **two DDAs** over a voxel grid: `VoxelWalk`, and `voxel_step` which
answers the transport's question of how far to the next material change. That was deliberate,
and it was recorded here rather than left to be discovered - a duplicated traversal is the
defect this codebase has been bitten by twice in a duplicated *constant* (V8, V9), with more
surface area to get wrong. The note also named its own re-entry condition: `voxel_step` decides
where tracks stop, so the rewrite belonged in a change whose evidence is the physics comparisons
rather than a screenshot.

**It is now one traversal.** `voxel_step` calls `VoxelWalk::Start` and `Next` and keeps only
what is its own question: the material comparison, the `every_cell` mode, the step-budget bound,
and the guard against returning a zero-length step for a boundary the track is already standing
on. Seventy-four lines of setup and advance arithmetic went away; the DDA now exists once.

The guard stays in `voxel_step` rather than moving into the walk. A zero-width cell is still a
cell the ray passes through, which is what the renderer wants from the walk, and the guard is
about what counts as a *step*. That is the zero-length step V1 records as real and reachable
only with `every_cell` - not the cause of V1's own symptom, which was the run size.

#### Why a green pipeline is not the whole of the evidence

`build_all.bat` says the physics still agrees, and it is the thing that decides. Green, on the
run that landed this: 47 of 47 tests; B1 against Geant4 at 1.25 sigma of 3; the mesh comparison
at 0.01 sigma with the step count moved by **-0.0%**, which is the sharpest single number here
because an altered traversal shows up as a different number of steps before it shows up as a
different dose; per-voxel scoring at **174 cells hit, summing to the volume total to 6.9e-16**;
the builder and the generated project at **0.00%** on both scorers, `cells` included, which is
what would move if a cell's material assignment had changed; proton R80 at -0.015 mm.
`tests/test_voxels.exe` still puts the slab boundaries at exactly -10, +20 and +50 mm and still
conserves the chord to 1.7e-13 mm.

All necessary - and all of it would also pass a rewrite that differed only in a direction this
scene never probes. B1 and the builder's 8x8x8 phantom produce a particular set of grid
geometries, entry points and directions; "the dose is unchanged" is evidence about those, and
a traversal is a function over all of them.

So the refactor was also checked *as a refactor*, against a verbatim copy of the function it
replaced: **561,720** (grid, q, d, `every_cell`) cases over six grids - checkerboard, slabs,
homogeneous, runs, a single cell - with isotropic, axis-aligned and degenerate directions, and
points exactly on cell boundaries as well as a hair either side. The returned distance was
compared bit-for-bit and the reported material by value. **Zero differences.** 203,646 of the
`every_cell` calls returned a finite step and 868 cases reached the zero-length-step guard's own
case, so the two things most easily got wrong were exercised rather than hoped for.

**And the equivalence check was itself checked.** "Zero differences" is worth nothing from a
harness that cannot produce one. Weakening the guard from `kTolerance` to `0` in the new code
makes the same harness report 424 mismatches; that negative control is the reason the zero is
quotable. A comparison against a reference implementation only becomes evidence once you have
watched it fail. The harness was then deleted rather than kept: what it asserts is "this refactor
changed nothing", which stops being a meaningful claim the moment the old copy is gone.

**Generalization.** A refactor's own claim is *equivalence*, and the pipeline does not test
equivalence - it tests the answer on one scene. Both are needed, and they fail in different
directions: the pipeline catches a rewrite that is wrong where it matters, the A/B catches a
rewrite that is wrong where this scene happens not to look. Deleting the reference copy is part
of the job, because a second implementation kept "for comparison" is just the duplication again.

### V16: one class, two sizes, in one link

`tests/test_custom_hook.cu` defines `G4STEP_HOOK` and is compiled together with
`src/scenes/scene_b1.cu`, which includes the same run manager header without defining it. The
engine stores its hook by value, so:

```
with    G4STEP_HOOK: sizeof(G4RunManager) = 48120
without G4STEP_HOOK: sizeof(G4RunManager) = 48136
```

One class, two layouts, across one link. Measured with a three-translation-unit program rather
than reasoned about, because "the sizes probably differ" is not a finding.

#### How it would have failed, which is not where you would look

Every member of `G4RunManager` is defined in the class body, so all of them are inline: the
linker keeps ONE copy of each and discards the rest. `new G4RunManager` sizes its allocation in
whichever translation unit writes it - the test's, at 48120 bytes.

If the surviving copies had come from `scene_b1.cu`'s 48136-byte layout, every member declared
after `engine_` - `last_stats_`, `run_`, `run_voxel_scores_`, `primaries_`, `slot_pv_`,
`slot_mat_` - would be addressed sixteen bytes too high, off the end of a heap block. MSVC keeps
the first COMDAT it encounters and the link order happened to agree, so it worked. Nothing about
the code arranged that.

Note what does NOT protect it: the members `scene_b1.cu` actually touches all precede `engine_`,
so their offsets agree in both layouts. That is why the first diagnosis, written before the
measurement, said it "works by luck of member ordering". Member ordering is not what it turns
on - the discarded-COMDAT choice is - and a fix aimed at the ordering would have left it exactly
as fragile.

#### The fix is not the one the defect suggests

The obvious remedy is to stop linking those two files, which is what the task raising this
proposed. It fixes the test and leaves the defect: `scenes/scene_registry.hh` is shipped API, so
any user project that combines its own step hook with a registered scene hits the same thing,
and that is an ordinary thing to want to do rather than a contrivance.

So the engine is held behind a pointer instead. A pointer is the same size whatever it points
at, so the layout no longer depends on the including translation unit's choice of hook:

```
after: with hook 13952, without hook 13952
```

- the same measurement that found it, showing it gone. `G4RunManager` also stops being a 48 kB
object, because it had a 34 kB engine embedded in it.

What remains true is the rule that was already documented: a translation unit which CONSTRUCTS a
run manager must define the same hook as the one that instantiated the engine. One .cu, one
instantiation. That is a rule about behaviour, not about layout, and it is enforced by the link
failing rather than by luck.

#### And what found it

Nothing was looking for it. It turned up while adding a SECOND test with its own hook
(`test_voxel_scoring.cu`) and asking where to put its build rule - the question "can this be
compiled alongside a scene like the other hook test is" has no answer that is not this. A
defect that only surfaces when somebody does the same thing twice is one a single instance
cannot show you.

### V17: a parser that cannot fail, and an expected count written from impression

Two things about reading a colour table, neither of which reached a wrong answer, both recorded
because the first was one keystroke from shipping and the second is a habit this register
already has an entry for.

#### `atof` has no way to say no

The first draft of `ReadVoxelColourTable` read each field with `std::atof`. That function returns
zero for text it cannot parse and reports nothing. A colour table with a header row -

```
index, r, g, b, a
1, 255, 0, 0, 255
```

- has five fields on its first line, all of them words, and `atof` reads that line as index 0,
red 0, green 0, blue 0, alpha 0. So the header would have been applied: class 0 painted opaque
black, counted among the rows that worked, and reported in the log as a row that applied. The
one class where this is hardest to see is exactly class 0, which the importer has already made
invisible.

The fix is `std::strtod` with the end pointer checked against the end of the token, plus an
`isfinite` test, because `nan` and `inf` do parse whole and a NaN colour is undefined behaviour
by the time `build_scene` casts it to a byte. A line of four or more fields that are not all
numbers is now counted as malformed and reported, rather than being read as zeros.

The general point is not about `atof`. It is that a reader whose failure mode is a plausible
value cannot be tested by feeding it good input, and every test written while thinking about
the format is good input. The test that catches this had to be written by asking what the file
would look like if a person made it in a spreadsheet - which is where a header comes from.

Reverting the validation, the way the fix was checked:

```
FAIL: comments, blanks, headers and short lines are not rows
FAIL: one row named a class
FAIL: the header and the nan are counted, not silently read as zero
FAIL: and the nan row changed nothing
```

#### The count written from impression

The same test asserts how many values were read on each scale, because the scale rule is per
value - `0.5` is a half and `128` is a half - and the counts are what make a table read the
wrong way visible in the log. For the mixed-form file the assertion was first written as
`on_unit == 13`. The file has ten fractional fields: four on each of the first two rows, and
two on the third, where `255` and `0` are the bytes. Thirteen is not a number that appears
anywhere in it.

It was corrected by counting the fields before the test was first run, so it never produced a
result. That is luck about timing, not method: V10 in this register is a threshold written the
same way, from impression rather than from the thing it measures, and that one did produce a
false failure at 1.93 sigma. An expected value in a test is a prediction, and a prediction
worth asserting is one worth deriving.

### V18: an assertion stronger than the request, and the regression hiding behind it

Two findings from one test, in the order they came out, because the second was only reachable
once the first was corrected.

#### 300 classes cannot have 300 distinct greys

The request for importable colormaps said they "should not repeat". The test written for it
asserted that a 300-class phantom comes out with 300 distinct colours, for every map. It
failed on gray, viridis and plasma.

Gray has 256 colours in it. A one-dimensional map sampled 300 times MUST give some
neighbouring pair the same eight-bit value; there is no implementation that does otherwise,
and the perceptual maps fail it for the same reason said approvingly - they are built to vary
smoothly, and smooth means neighbours are close.

So the assertion was impossible, not the code wrong. What "does not repeat" can mean, and what
the request was actually about, is that the map is not CYCLED once it runs out - that class 1
and class 257 cannot come out identical. Stated exactly: the map never returns to a colour it
has left. Collapsing runs of equal neighbours and then looking for any repeat says precisely
that, and every map passes it.

The general shape of the mistake: an assertion that sounds like the requirement, is stronger
than the requirement, and is unsatisfiable. It fails on correct code, and the tempting repair
is to weaken it until the code passes - which is how a test ends up asserting nothing. The
repair here was to work out what property was being asked for.

#### And underneath it, a real regression

The corrected test then failed on one map: the hue sweep, which is the default and the one
this file has always used.

Colouring by POSITION in the class list, which is what the code did before, put the last of N
classes at 360*(N-1)/N degrees - one step short of the circle. Colouring by the VALUE, which
is what "resized based on the range of the voxel index values" asks for, puts the highest
class at exactly 360 degrees. 360 is the same red as 0. So the lowest and highest index in
every segmentation came out identical.

The old code was not right about this on purpose; it was right by accident, and the accident
did not survive a change to the parameterisation. The sweep is 330 degrees now, which leaves
the ends visibly apart whatever N is.

Worth noting what would NOT have found it: the phantom looks correct. Two classes out of two
hundred sharing a colour, at opposite ends of the index range and so rarely adjacent in space,
is not something anyone sees. It took an assertion about the map itself.

#### The three-row list, for completeness

Not a test finding - a photograph found this one. The first version of ui::SplitListForm gave
the selected item's form whatever it asked for and left the list a floor of three rows. A
voxel volume's form is a dozen fields and six buttons, so an eleven-solid list came out three
rows tall.

The floor is half the section now. It is the same lesson the section heights taught and this
register already records: a floor measured in rows guessed at the point of writing does not
scale with the panel, and a floor that is a SHARE does.

The same photograph showed the other half of it: the form was anchored to the bottom of its
section, so a two-source list had a band of nothing between it and its form. The form begins
under the last row now, and both properties are asserted in tests/test_ui_layout.cu rather
than left to the next screenshot.

### V19: two ways a widget can be somewhere it is not

Reported as one bug - "when the solids list becomes scrollable, clicking in the sources panel
opens a material picker" - with the observation that it might be systemic. It was, and it was
two unrelated systemic faults sitting on top of each other, both of which had been there for
some time and neither of which announces itself.

#### A widget you cannot see is still clickable

`ui::Context::Hovering` is what every widget in this UI asks to find out whether the cursor is
on it. It answered from the widget's rectangle alone. The canvas clip - the thing that stops a
scrolled-away row from being PAINTED - had no say in whether it could be CLICKED.

So the tail of the solid form, which is taller than the half of the panel it lives in, was
clipped away below the section's bottom edge and kept its rectangles. Those rectangles lie
over the SOURCES section underneath. Hovering a source lit a button that was not on screen;
clicking one opened the material picker belonging to a solid scrolled out of view.

The fix is one line - the clip is part of the hit test - and it is one line rather than a
change in every widget because there is no widget for which "clickable but invisible" is
correct. Four call sites were hit-testing with a bare `rect.Contains(mouse)` and bypassing
`Hovering` altogether; those now go through it.

What makes this class of bug survive: the two halves of "is this widget here" were written in
different places and neither is obviously incomplete on its own. `Hovering` reads like a
complete answer to its own question. The clip reads like a drawing concern. Nothing in either
says the other exists.

#### And the widget ids collide

Chasing the first turned up a second. Widgets are identified by an integer, and the lists that
grow with the model were numbered `base + index`, with the bases ten or a hundred apart:

```
solid rows   400 + i      "Assign material" button   410      -> collides at 11 solids
source rows  690 + i      kind buttons               700      -> collides at 11 sources
element rows 200 + i      element controls           210..215 -> collides at 10 elements
material     300 + i      material controls          310..320 -> collides at 10 materials
voxel classes  4000 + solid*100 + class,  importer cap 4096   -> collides WITH ITSELF at 101
```

The last one is the worst and is a direct consequence of a change made earlier in this
register: raising the class cap from 64 to 4096 (V14) made the stride of 100 wrong, and the
261-class segmentations the program was changed to accept are all past it. Widening a limit
somewhere else invalidated an assumption nothing recorded.

A shared id does NOT draw wrong. Both widgets still paint, still highlight, still fire on
their own rectangle - `hot` is recomputed per widget from its own geometry. What is shared is
`ctx.hot`, `ctx.active` and `ctx.focus`, so a rename types into the wrong row and a drag is
picked up by a control the cursor is nowhere near. That is a description nobody connects to
numbering.

The blocks are a million apart now, with the class stride equal to the importer's class cap.
`tests/test_ui_layout.cu` enumerates the ids of a 200-solid, 4096-class-each model plus five
thousand of everything else and looks for a repeat - and asserts that the OLD stride does
collide, so the test cannot pass by being vacuous.

#### Why it surfaced now

The ghost the selftest actually catches is widget 40630 - the unit dropdown for `rot x`. Those
dropdowns were added one commit earlier. Each one made the solid form a little taller, and a
taller form pushes more of its tail below the fold, so a defect that had always been there
became something you hit by clicking a source.

Which is the ordinary way a latent fault is found: not by anyone looking for it, but by an
unrelated change making its precondition common. Worth noting because the instinct on being
handed "this broke after your change" is to look at the change.

#### What was checked, and how

The one-line fix is falsified by reverting it: three assertions in tests/test_ui_layout.cu
fail, at the section boundary and past it.

The selftest checks the whole path: the cursor goes into the SOURCES half with the solid form
scrolled so its tail is clipped, and the check reads back which widget claimed it.

THE FIRST VERSION OF THAT CHECK PASSED WITH THE BUG PRESENT. It read `ctx.hot` at the end of
the frame, and `hot` is whatever tested LAST - the SOURCES half draws after SOLIDS, so the
source row under the cursor overwrote the ghost that had claimed it a moment earlier. The
check was measuring the right quantity at the wrong time, and a check that reports success on
broken code is worse than none, because it is now evidence.

It was caught by building the broken version deliberately and confirming the check failed on
it - which is the only thing that distinguishes a passing test from a test that cannot fail.
The fix is to sample `hot` where it means what the check needs: `App::hot_after_solids`,
recorded the moment the SOLIDS half finishes drawing. Zero there means nothing in that half
claimed the cursor. Broken, it reads 40630.

Reading `hot` rather than clicking is still deliberate: it is exactly the state that was
wrong, and reading it has no side effect, where a simulated click on whatever lies under those
coordinates might add or delete a source.
