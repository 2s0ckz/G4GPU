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

### V20: a voxel cell's material was a number from the wrong table

Reported as two symptoms, which is what made it worth chasing rather than dismissing:

> gammas appear to pass straight through the phantom, even at low energy. electrons and
> positrons on the other hand cannot seem to penetrate any non-air voxel, no matter how high
> energy they are

Opposite failures usually mean two faults. These were one, and the two directions are what
identified it: no physics error makes a gamma too transparent AND an electron too absorbing.
A wrong number read out of an array does exactly that.

#### The two halves

The builder fills a voxel grid's cells during `Construct()`, with indices into
`Model::materials`, because that is the only numbering that exists then. DEVICE material
indices are handed out later, in `G4Flatten`. The transport read the cells as device indices.

**Model order and device order are not the same order.** Device indices are assigned by
walking placements and taking each logical volume's material, so they cover only the materials
of placed volumes, in placement order. One material in the user's list that is not on any
volume slides every index after it.

**And a material used only by voxel classes was never built at all.** Nothing walked voxel
classes looking for materials, so such a material kept `device_index = -1` and never entered
the table the kernels read. For a segmented phantom this is the ordinary case: the volume
carries one material and its two hundred classes carry others.

The result was an index out of range of the material table. The kernels read past the end of
it, so a gamma got an enormous mean free path and crossed the phantom depositing nothing,
while an electron got an enormous stopping power and dumped its whole energy at the entry
face. Measured, on 200 mm of soft tissue built as a box and as an identical grid:

```
  beam              box MeV/evt   grid MeV/evt    ratio
  gamma 6 MeV          1.818155       0.000000   0.0000
  gamma 0.1 MeV        0.045223       0.000000   0.0000
  e- 100 MeV          45.654891      99.995802   2.1903
  e- 10 MeV            9.745349       9.997658   1.0259
  e+ 20 MeV           18.392694      19.997902   1.0873
```

x0.0000 and x2.1903 are the two sentences of the report, in numbers.

#### Why nothing caught it

Every test that touched a voxel grid used **Air and Water only**, and with those two the model
order and the device order coincide: the world is air, the phantom is water, so model 0 is
device 0 and model 1 is device 1. The builder's own selftest phantom is exactly that, and it
reports 174 cells hit summing to the volume total to 5.6e-16 - correct, and blind to this.

`tests/test_voxel_scoring.cu` could not see it either, for a different reason: it builds its
grid by hand with `SetCell(k, water->device_index)`, so it writes device indices directly and
never exercises the builder's numbering at all.

So the fault needed a third material to appear, and every test had two. The new test uses five,
with the spares deliberately BETWEEN the ones that get placed, and gives the cells a material
nothing else uses - because that is what a phantom looks like and it is the case the old code
got wrong.

#### The fix

A grid now says what its cell numbers mean. `G4VoxelGrid::SetCellMaterials` takes the table the
numbers index; `Build()` translates through it to device indices; `G4Flatten` builds every
material in it, alongside the materials of placed volumes. A grid built by hand leaves the
table empty and its cells pass through untouched, so the direct `device_index` route still
works.

A material in that table with no device index is a FATAL rather than a fallback to the volume's
own. Falling back is what the ordinary `material_at` path does for a cell that was never
assigned, and it is right there; but a class the user DID assign, silently transported as
something else, is precisely the failure this mechanism exists to end, and the version of that
failure that was live for weeks was silent.

#### Three wrong theories, in order

Worth recording because each was cheap to test and expensive to assume.

**Unassigned materials.** A cell whose class has no material gets -1, and `material_at` falls
back to the volume's own. That would have explained the symptoms. The user's own note killed
it - they had assigned water to every class and air to class 0.

**The step budget.** Every voxel boundary ends a step, and the transport advances every live
track one step per iteration of a loop bounded at 1000. A 400-cell traverse costs 400 of them,
so a fine grid starving its own tracks was plausible. Falsified by measurement: the same water
as a box and as grids of 1, 8, 32, 64, 128, 200 and 400 cells gave 9.6766 MeV/event in every
case, 74 iterations, nothing abandoned. `every_cell` is only switched on for a volume with a
per-voxel scorer, which that geometry did not have.

**"The wrong material."** Once the numbering was understood I described the effect as reading
the wrong material. It is worse than that: the index is out of RANGE, so the read is past the
end of the table. That distinction matters for what to expect - a wrong material gives wrong
but bounded physics, and an out-of-range read gives whatever is in memory.

#### Amendment: the same fault a second time, in the emitter

Fixing the builder and giving the selftest phantom a third material turned the pipeline red on a
check nobody had aimed at this:

```
FATAL: 'cells' disagrees. builder 41570.230751 MeV, generated project 1515.64 MeV (96.35%).
```

The `.cells` sidecar a saved project ships holds MODEL material indices - `write_project.cc`
walks the classes and stores `c.material` - and the generated `LoadVoxelCells` reads it straight
into `grid->Cells()`. So a project you save and run had the identical fault: its phantom
transported as whatever material the number happened to name. 1515 of 41570 MeV is 3.6%, which
is about what a phantom of air deposits.

Two instances of one mistake, in two files, because the same file format is read in two places
and only one of them was thought about. The emitter now writes what the builder writes:

```cpp
LoadVoxelCells(solidPhantom, "Phantom.cells");
solidPhantom->SetCellMaterials({matAir, matWater, matSoftTissue});
```

and the two sides agree to 0.00% on both scorers.

What is worth taking from it: the check that caught this is the one whose own comment says it
exists so that "run it here" and "save it and run that" cannot disagree about what a model
means. It had been passing for months on a model of air and water, where nothing could
disagree. A comparison is only as good as the model it compares - and the cheapest way to
strengthen one is usually to make the model less symmetric, not to tighten the threshold.

#### Open, and measured while here: that comparison's threshold is far too loose

`compare_project.ps1` allows 6%, calibrated in V10 on the belief that the builder and the
generated project are two independent Monte Carlos - 6.4% apart at 200000 histories, 0.80% at a
million. They now agree to 0.00% at a million, which is not what two independent samples do. The
seed unification means both sides start from the same default, so this is a determinism check
wearing a statistical threshold.

That matters because 6% is three orders of magnitude looser than the check can now afford, and
this bug was only caught because it produced 96%. A material mix-up worth 3% would still pass.
Not tightened here: it needs its own measurement of what the two sides actually do across
models, and doing it inside a fix for something else is how a threshold gets miscalibrated in
the first place - which is V10.

### V21: the picture was paying the transport's price, and two optimisation claims that were not measured

Reported as: the viewer is overwhelmed by CAD imports past about 100,000 triangles, while
MeshLab shows the same file without trouble.

Those are not the same job - MeshLab rasterises and this ray-traces - but that explains a
constant factor, not a wall. A BVH ray-traces millions of triangles interactively.

#### What it actually was

Not the triangle count, and not the BVH. Measured on the host, `mesh_nearest_hit` at a million
triangles costs 1.6x what it costs at two thousand, with tree depth going 11 to 19. The tree is
a perfectly ordinary median split and it works.

The renderer calls the TRANSPORT's geometry functions - deliberately, so that the picture shows
what the transport sees, which is a good principle that turned out to have a price nobody had
priced. Per pixel, per composited layer, a mesh volume was paying:

```
  inside_volume  -> mesh_inside          up to  4 full parity counts
  dist_in        -> mesh_dist            1 nearest hit + up to 16 parity counts
  normal_at      -> numerical_normal     up to 24 parity counts
```

A parity count is the expensive kind. A nearest-hit walk rejects any subtree whose box begins
beyond the closest hit so far; a parity count has no best-so-far to reject against, so it
visits every leaf along the ray. So the renderer was doing about forty-four uncullable
traversals per pixel where one cullable one would do.

`normal_at` is the worst and the silliest of the three: `numerical_normal` finite-differences
the containment test six times to APPROXIMATE a normal that the BVH walk already knew exactly.
The triangle it hit has a face normal. Returning it from the same walk is cheaper and more
accurate than the gradient it replaces.

A picture needs none of the rest either. The nearest hit ahead is the surface to draw, a graze
is harmless to shade, and the inside test was there to stop the search re-finding the surface
it had just crossed - which a strictly-ahead hit cannot do anyway.

Measured in the viewer at 1680x960, before and after:

```
   triangles        before                after            gain
      40,000    166.2 ms   6.0 fps    45.1 ms  22.2 fps    3.7x
     200,704    223.4 ms   4.5 fps    53.5 ms  18.7 fps    4.2x
     802,816    285.8 ms   3.5 fps    61.2 ms  16.3 fps    4.7x
```

#### And the two claims that were wrong

**"double is costing 5.5 to 19x."** That one holds and is measured on the card: at one ray per
pixel for 1440x880, `mesh_nearest_hit` takes 4.70 ms in float and 89.57 ms in double at 20k
triangles, 18.76 against 103.75 at a million. Note what the double column does as triangles
grow - 89.6, 88.9, 103.8, essentially FLAT - while float goes 4.7, 11.3, 18.8. In double the
card is so short of FP64 units that the tree barely matters; the pass is arithmetic-bound. In
float the cost tracks the tree, which is the healthy regime. The render pass is still double;
that is the next change and it needs a render-only float copy of the geometry.

**"front-to-back child ordering is usually worth a further 1.5 to 2x."** Said without measuring
it. Implemented, it first appeared 1-3% SLOWER, which I reported as a regression - also wrong,
because run-to-run variation ACROSS BUILDS of identical source is about 6% on this card, and a
1-3% difference sits inside it. This is V10 again: a difference compared against nothing.

Settled by building both variants and alternating them within one session, where the spread is
about 1%:

```
  pair   unordered   ordered    diff
     1       53.45     54.53   +1.08
     2       53.59     57.54   +3.95
     3       55.84     55.17   -0.67
     4       54.16     54.43   +0.27
     5       53.33     56.41   +3.08
```

Mean difference +1.54 ms, paired sd 1.93, so t = 1.79 on 4 degrees of freedom - not
significant. The honest conclusion is neither of the two I gave: the change makes no
measurable difference. It is dropped for that reason, which is different from being slower.

Consistent with the arithmetic-bound finding above: when the limit is FP64 throughput,
reordering traversal cannot help much either way. Worth retrying after the render pass is
float, because the regime it failed in will have changed.

#### The lesson worth keeping

Two of the three things in this entry were standard optimisations quoted from memory rather
than measured, and both were wrong for this code: one because the bottleneck was somewhere
else entirely, one because the effect was below the noise. The one that paid was found by
reading what the renderer actually called, per pixel, and counting.

### V22: five reports from one afternoon at the keyboard, and what each of them actually was

Five things reported in one message. They are together because the pattern is: none of the five
was where it looked, and three were one line each in a place nobody would have gone looking.

#### 1. A phantom visible through an opaque object in front of it

The cell march inside a voxel volume had no ownership test along it. The outer walk asks
`locate` at every surface it draws - a higher layer takes the space wherever it overlaps, so a
lower volume's surface is not there and is not drawn - and the march that walks a grid's cells
skipped that question entirely and ran the grid's whole depth in one go. Cells inside an opaque
box placed over the phantom were therefore composited, and being nearer the eye they arrived
BEFORE that box's own surface, so front-to-back blending put them on top of it.

Two halves to the fix, and the second is the one that would have been missed: the march stops
where an outranking volume begins, AND resumes past it, or everything behind a translucent
object over a phantom vanishes instead of being drawn wrongly.

Then a third thing, found by the test failing: the outermost hidden cells were still drawn,
because `walk.t` counts from a point nudged 1e-4 mm past the grid's entry while the ownership
distance counts from the entry itself. Out by a nudge, in the direction of drawing a cell the
cover owns - and it only shows when a box face is FLUSH with a cell boundary, which is the
normal case rather than the corner case, because cells are on a lattice and a box placed over
them lands on it. One cell at a quarter opacity in front of an opaque surface is a quarter of
the pixel.

Worth writing down about the fixture: the first version put the hidden cells one cell clear of
the cover, the invariant passed, and the flush arithmetic would have shipped. It failed only
because the first fixture happened to be flush, and the honest sequence was: fixture fails,
hypothesis (the nudge), test the hypothesis by giving the fixture a margin, watch it pass, fix
the arithmetic, put the fixture back to flush because flush is the harder case.

#### 2. An import arriving on layer 2

No reason. `ImportFile` said `s.layer = 2` and `InsertShape` said 1. The consequence is not
cosmetic: an imported mesh or phantom outranked everything already placed, won every overlap
without anyone choosing that, and was exempt from the same-layer refusal that would otherwise
have pointed the overlap out.

Changing it moved the selftest scene, because the selftest's mesh cube sits inside the orb and
the cylinder that earlier frames place and its RANK decides who owns that space - which the
scorer on it then measures. The trace was three new overlap warnings and nothing else; the dose
happened to be unchanged. The fixture now pins its own layer, because a default two files away
is not where a compared number should be decided from.

#### 3. Overlap checking that only fired on a substantial overlap

`kVoxelGrid` was missing from `solid_half_extent`'s switch, so it returned zero. See VOXELS.md
for the three things that broke; the reported one is that the check's first stage compares
centre separation against the sum of the bounding radii, and a radius of zero only clears when
the other volume's centre is nearly on top of the grid's own.

The one worth dwelling on is the one nobody reported. `ScoredMass` sizes its integration box
from that bound too, so the mass of a scored phantom was zero - and because both the builder
and every generated project print the dose only `if (mass > 0)`, the symptom was not an
infinity but a MISSING LINE. A phantom you scored reported its megaelectronvolts and no dose,
and nothing said why. Which is the failure mode this register keeps rediscovering: the wrong
answer that declines to print itself.

The check was also insensitive for a second, independent reason, so the one-line fix would not
have been enough: it sampled the smaller volume's own bounding CUBE, sized by that solid's
LARGEST half extent. A phantom 300 x 200 x 1700 mm sits in a cube of 3400 mm on a side and
fills 2% of it, so 4000 samples put 80 points in the phantom and an overlap under about a
percent of it went unseen. It now samples the INTERSECTION of the two bounding boxes - which is
where an overlap has to be - on a jittered lattice, and the boxes are the rotated bounding cube
bounded per axis rather than the bounding sphere, which is tight for an unrotated solid where
the sphere is sqrt(3) too big on every axis. Measured on the selftest scene: a 30 mm cube inside
a 75 mm sphere comes back as 2.7e+04 mm3 and 100% of the smaller, which is exact.

#### 4. Picking a row of an open dropdown opening the menu underneath it

An overlay painted last has to be hit-tested first, and in one pass over the frame it cannot be
both. The list is drawn after the panel that declared it - everything drawn after a widget
paints over it - so by the time its rows are tested, every widget it covers has had its turn
and one of them has taken the click.

`Context::block` is the region the list occupied, carried over from the frame it was painted on,
and `Hovering` refuses the cursor to anything at or behind its layer inside it. A frame stale,
which is where the list actually is on screen. The same rule covers every widget rather than
dropdowns only, which is why the test checks a plain Button under the list too.

**And then the menu bar, which has the same fault and a worse symptom.** Not reported - found by
asking what else in this UI paints an overlay after the thing it covers. `MenuBar::Item` records
its rows and `EndMenu` paints them, and the bar draws after the panels. Two ways it is worse
than the dropdown case: `Item()` returns true on hover-and-press whoever else took the click, so
a menu entry over a panel button fired BOTH of them; and `EndBar` only closes the menu when
`ctx.hot == 0`, which the control underneath had just set to itself, so the menu did not close
either. The bar borrows the same single blocked region, saving and restoring what it found, or a
bar drawn between `DrawOpenSelect` and the next frame would wipe an open dropdown's region on
its way past.

The test for that half was vacuous on its first attempt and said so only when the fix was
removed: `Button` fires on RELEASE while it holds `active`, so a press-only frame proves nothing
about a button, and "the button underneath did not fire" passed with the bug in place. Three
assertions in this entry needed the bug put back before they could be trusted; two of them were
wrong until that was done.

#### 5. Resizing the left panel rotating the view

A splitter's grab strip straddles the boundary it moves, deliberately, so its inner half lies
inside the 3D view - and those columns were claimed by two things at once. What makes this worth
an entry is the asymmetry, which the report noticed and which says it is arbitration and not a
stray pixel: `left_w = mouse_x` puts the view's FIRST column under the cursor, so the pointer is
inside the view for the whole drag, while `right_w = width - mouse_x` puts the view's EXCLUSIVE
right edge there and the pointer is just outside for the whole drag. One rule, applied
consistently, and one of the three geometries happened to escape it.

#### And a measurement, because the fix changed the kernel

The additions took `render_geometry` from 250 registers and no spilling to 255 registers and 222
bytes of spill stores, with the stack frame 1760 -> 2144 bytes. A single benchmark run then read
7-9% slower than the figures from the previous session, which is exactly the shape of the mistake
V10 and V21 are about, so it was measured properly: two binaries differing only in
`renderer.cuh`, alternated within one session.

```
  pair        with        without      diff
     1       58.04         58.15      -0.11
     2       57.99         57.73      +0.26
     3       62.01         61.44      +0.57
     4       61.36         61.59      -0.23
     5       61.26         61.42      -0.16
```

Mean difference +0.07 ms on 58 ms, paired sd 0.34, t = 0.43 on 4 degrees of freedom. Not
significant. Note what BOTH columns do down the table - 58 to 61 ms - which is 6% of drift
inside one session, and is the whole of the 7-9% the single run showed.

Two things learned that are worth keeping. `__noinline__` on the new ownership helpers, tried to
keep their frames out of the hot path, made it WORSE - 2352 bytes and 306 spill bytes - because
an ABI call forces the caller to save its live registers around it. And the register report is
cheap evidence where a benchmark is expensive: `nvcc -Xptxas -v -cubin` on a one-line translation
unit that instantiates the kernel takes twenty seconds and says exactly what changed, where the
benchmark needs a paired design and ten minutes to say anything at all.

Also corrected here: the triangle counts in V21's table are the numbers `-benchmesh` was ASKED
for, not the numbers it built. The generator lands on the next whole UV sphere, which is 2x the
request - 80,656 rather than 40,000, and so on. The speedups are unaffected, since both columns
measured the same geometry; the labels were wrong.


### V23: a worry implemented before it was tested, and the test that deleted it

Two changes: per-class layers for voxel volumes, and a float render pass. The entry is about the
second, and about one habit.

#### The sixty lines that should not have been written

The render pass needed a float copy of the scene. The worry worth having was the BVH: a box
rounded INWARD by one ulp no longer contains its own triangles, the traversal rejects any
subtree whose box the ray misses, and the result is a hole in the mesh rather than a shading
error. Correct worry.

So the conversion was written to RECOMPUTE every node's box from the float triangles - a
post-order walk, leaves from their triangles and interior nodes from their children, about
sixty lines with a recursion and an empty-node special case - and the file's header comment
explained at length why an epsilon would have been a guess and this was not.

Then the test. It asked the direct question: does a plain cast break containment? In 4095
nodes, over a sphere offset 400 mm from the origin so that float rounding had something to
bite on: NO. Not once.

The reason is one line of `build_bvh`: `lo[k] = std::min(lo[k], c)` over vertex coordinates. So
every box bound IS one of the numbers that will later be compared against it, and rounding to
float is monotonic - `float(bound) <= float(vertex)` wherever `bound <= vertex`. Containment
survives the cast by construction. The sixty lines were guarding against arithmetic that does
not happen.

They are gone. What replaced them is the test asserting the property the cast depends on, so
that a future change to `build_bvh` which COMPUTES a bound rather than picking one fails here
instead of appearing as holes in somebody's CAD import.

The habit worth naming: the worry was real, the reasoning about the fix was plausible, and the
comment justifying it was written before anything was measured. One test on the actual data
answered it in a minute. THE ORDER MATTERS - test the worry, then write the fix it justifies.

#### And a limit of ray_triangle, found by a fixture that was too tidy

The first ray sweep aimed every ray at the sphere's exact centre from a lattice of directions.
Result: 14 of 4000 rays that hit in double missed in float, with a worst distance difference of
120 mm - the sphere's diameter, so float had found the FAR surface.

That reads like a broken conversion. It is not. Look at the other number in the same run: 21 of
those 4000 rays missed in DOUBLE as well. Those rays land on the shared edges of a UV sphere,
where Moller-Trumbore's barycentric test can reject both adjacent triangles and leave a hole -
`ray_triangle`'s `u < 0 || u > 1` on a hit that is exactly on the edge. Both precisions have
the hole; float lands on it more often because it rounds more.

Jitter the aim by a third of a millimetre, off the lattice: 4000 of 4000 hit in both, nothing
missing, nothing extra, worst difference 0.0007 mm.

So the sweep is jittered, and the alignment case is recorded rather than asserted - asserting
it would be asserting a known limit of the intersector. Two things worth keeping from it: a
fixture aligned with the geometry it tests measures the alignment, not the code; and the second
number in the output - how many missed in double - is what turned a diagnosis into a fact.

#### What the float pass is worth

Paired, alternating two builds differing only in `renderer.cuh`'s instantiation, within one
session at 1680x960:

```
   triangles      double     float    gain
      40,000    47.6 ms   25.5 ms    1.86x
     401,956    61.6 ms   28.5 ms    2.16x
   1,607,824    70.6 ms   31.9 ms    2.21x
```

Spread inside each triple under 1%. Paired because variation across builds of identical source
on this card is about 6%, which is the lesson of V21 and V10 applied rather than relearned.

The picture's cost: the viewer's selftest counts the pixels solid geometry covers, and it went
from 57768 to 57766.

#### One more thing the per-class layer work turned up

The equivalence test for per-class layers first reported exactly 2.0000x the energy for the
scene with a scorer covering both a phantom and the box overlapping it. Not physics:
`SetSensitiveDetector` calls `AddNewDetector` once per volume, so a detector attached to two
volumes appears twice in `Detectors()`, and the test's helper summed its scorer twice.
`AssignIndices` already de-duplicates and `Scorers()` is the list it built - which the comment
in G4SDManager says in as many words. The helper was copied from `test_voxel_materials.cu`,
where it was latent because that test scores one volume; both are fixed.


### V24: a rule extended in one place and left behind in another, twice

Per-class layers (V23) made a volume's priority a function of the point. Two things that ask
about layers were not told.

#### The one that was reported

"Overlap checking is not working for objects on one layer and voxel classes that are set to
that layer." Exactly right, and the code says why in one line:

```
  if (scene.volumes[i].layer != scene.volumes[j].layer) { continue; }
```

A phantom on layer 2, a box over it on layer 3, one of the phantom's classes raised to 3. In
those cells two volumes share the space at the same layer, which is what the refusal exists to
prevent - the tie-break is list order, so the dose depends on the order the detector was built
in. The pair was dismissed before any point was looked at, because 2 is not 3.

The fix is the same shape as V23's: the pair test is on the RANGE of layers each volume can
have, and every sample point is asked what layer each SIDE has there. Both halves matter. Range
alone would refuse every phantom with a raised class, because the range meets the box's layer
whether or not any cell does - which is why the test asserts that the clash CLEARS when the
class moves off that layer, and not only that it appears when the class is on it.

The lesson is not "remember the overlap check". It is that a feature which changes what a word
means - here, what "on the same layer" means - has to be followed everywhere the word is used,
and grep for the word is the way to do that. `layer` appears in the navigator, the renderer, the
overlap check, the scored mass and the flattener; V23 did the first two and the last.

#### The one that was not reported, and was worse

The warning was made to name the offending class, because "the phantom overlaps the seat" is
not actionable when the phantom has two hundred classes. It printed `Phantom[class 1]` where the
class has the label `odd`.

A label of "[class 1]" is the fallback for "this solid has no such class". So the lookup was
reading the wrong solid - and the reason is a comment that had been in the file for a long time
saying it could not:

```
  // The flattened scene's index is the placement's, and the model's solids are in the
  // same order for a model built here
```

They are not. `G4Flatten` emits one volume per PLACEMENT, and a boolean's two operands are model
solids that are never placed. One subtraction in the scene shifts every model index after it.

What that was costing, in code that had nothing to do with this change: the pop-up's "raise the
second one's layer" button read the flattened index as a model index and raised a DIFFERENT
solid's layer - silently, leaving the clash it was offered to fix exactly where it was, and
moving something the user had not asked about. Both sites now look the solid up by name, which
is the back-reference the flattened scene does not carry.

Two things worth keeping. An assumption written in a comment is not a checked assumption, and
this one had been read and reproduced rather than tested. And it surfaced because a message was
made to say something specific: "[class 1]" was wrong in a way "[the phantom]" could never have
been. A message that names the thing it is about is a test that runs every time anyone looks.


### V25: five reports, and the one that was flat against the thing it was blamed on

Five things reported after the float render pass went in. Two were regressions it caused, two
were older bugs it exposed, and one was not about the thing it looked like.

#### The one worth the entry: a cost that did not move

"The GUI slows down dramatically when I load a large CAD file, even for tasks that have nothing
to do with rendering, like adding a new element." Then, unprompted: "even something like opening
a dropdown menu, not even making any changes."

That second sentence is the whole diagnosis. Adding an element rebuilds the scene, so a slow
edit could be a slow rebuild; opening a dropdown changes nothing at all, so whatever is slow is
PER FRAME. So the frame was split into its four parts and measured:

```
   triangles      cuda   readback     ui   present
       4,096   5.82 ms   16.45 ms   2.80    1.13 ms
     401,956  14.03 ms   16.38 ms   2.78    1.22 ms
   1,607,824  31.32 ms   16.34 ms   2.81    1.16 ms
```

16.4 milliseconds of read-back, and IT DOES NOT MOVE. A cost that is identical at four thousand
triangles and at one million six hundred thousand is not the geometry, whatever the report was
about. It was a loop of `cudaMemcpy`, one per scanline: 960 calls, each carrying about ten
microseconds of launch and synchronisation cost whatever it moves. Six megabytes the hardware
copies in under a millisecond, charged 16 ms because it was asked 960 times.

`cudaMemcpy2D` takes both pitches and does it once. 16.4 ms became 0.97 ms, and all three sizes
went to 60.0 fps - vsync-limited rather than work-limited.

Three lessons, in order of how much they were worth:

* **the flat column is the answer.** Not the growing one. The instinct was to look at `cuda`,
  which does grow with triangles and which had just been changed; the bug was in the column
  that ignored the variable entirely.
* **the second sentence of a report is often the diagnosis.** "Adding an element is slow" is
  consistent with a dozen causes. "Opening a dropdown is slow" eliminates all but one class.
* and the earlier float-vs-double benchmark was measured on frames that BOTH carried this
  16.4 ms. The paired comparison was still valid as a frame-time comparison, and the ratio it
  reported - 1.86 to 2.21x - understated the render pass itself. Subtracting the read-back from
  both arms puts the render-only figure near 3.5x. Quoted here as an inference, because it is
  one: the two were never measured with the read-back removed.

#### The two the float pass caused

Both were arithmetic assumptions that double had been carrying silently, and both appeared as
missing pixels rather than as wrong ones - which is why they read as "artifacts".

`b^2 - 4ac` for a 60 mm sphere seen from 1500 mm is 9.00e6 - 8.99e6 = 1.44e4. Three digits
cancel; float has seven, so four are left. The root is about 0.004 mm out, `is_crossing_to`
probes 0.001 mm either side of it, both probes land the same side of the surface, and the
crossing is not seen. The hit rate is 100% at 200 mm, 47% at 1500 and 10% at 4000 - and the
DEFAULT camera distance in the builder is 1500, which is why it read as "the sphere renders with
speckle". Starting the ray at the solid's bounding sphere makes b and c both O(radius).

The second was in the same file and older than the float pass: `quadric_inside` compared the raw
quadric value against `kSurfTolerance`, which its own documentation calls a half-width in
millimetres. An orb is written with gradient 120 per mm and an ellipsoid with 0.04, so one
tolerance is a band 8e-7 mm wide on one and 2.5e-3 mm on the other - and the second is wider
than the probe, so both sides of every crossing reported "on the surface" and the ellipsoid had
no surfaces at all. `quadric_normal`, twenty lines above it, had always divided by the gradient
and says in its own comment why. The containment test never did.

What identified both without a debugger: the SHAPE of the failure against camera distance.
Falling from 100% to 10% is a cancellation; 0.1% everywhere is a unit error. The test asserts
the sweep rather than a single number, for that reason.

#### And the two older ones

Two coincident faces on different layers drew NEITHER. The search took whichever volume it found
first at the nearest distance, the ownership test threw it away because the other owned the space
behind it, and the next iteration started just inside both - where both are "already inside" and
neither is entered again. The fix is a tie rule in the search; the test recolours the top box and
requires the picture to change, which is what fails on the bug, and recolours the bottom one and
requires it not to, which is what says the layer rule still applies.

And a nearly transparent volume looked black. The geometry pass accumulates PREMULTIPLIED colour,
so 3% opacity leaves 3% of the colour and nothing else; written out that is dark, when what it
means is "3% of this and 97% of what is behind". The coverage now travels in the framebuffer's
spare alpha byte. Worth noting: three resolve kernels had the same arithmetic and only the
on-screen one was fixed at first, so the two used for offscreen export would have written a
different picture from the one on screen. They share one function now.


### V26: a cheaper equivalent that was not equivalent, and the assertion that said so first run

"The speed thing is still a problem for large CAD files. Particularly when opacity is set to
anything less than 100." The word that mattered was `opacity`.

The render walks surfaces front to back and stops accumulating at 99.5% coverage, so an OPAQUE
volume ends the walk at its first surface: one surface, one ownership test. A TRANSLUCENT one
does not end it, so the walk runs up to `kMaxLayers` = 8 surfaces and pays the ownership test
eight times. And the ownership test was `locate(hit + nudge * dir) == best_vol`, which asks every
volume - including this one - whether it contains the point. For a mesh that is `mesh_inside`: a
parity count, which unlike a nearest-hit walk has no best-so-far to reject a subtree against and
so visits every leaf along the ray. It is the one traversal the whole BVH fast path exists to
avoid, and it was being paid per composited layer.

The renderer already knows the answer to the part that costs: its walk just crossed the surface
INTO the volume, so containment is established. What it actually needs is "does anything that
outranks this volume contain this point", with the volume's own test left out - `locate` restricted
to a point already known to be inside.

#### The half of the comparison that got left behind

`locate` does two things per candidate, and the replacement copied one of them:

    if (volume_rank(layer_hi_of(g, i), i) < best_rank) { continue; }   // cheap rejection
    if (!inside_volume(g, i, p)) { continue; }
    const long long rank = volume_rank_at(g, i, p);                    // and then AT THE POINT
    if (rank < best_rank) { continue; }

`layer_hi_of` is the candidate's HIGHEST possible layer, and it has to be: a phantom whose bone
class outranks everything and whose air class outranks nothing cannot be dismissed on one number.
That makes it right for rejection and wrong for acceptance, which is why `locate` follows it with
`volume_rank_at`. The replacement kept the rejection and dropped the comparison, so it read "a
grid covers whatever its top class could cover". A box coincident with a phantom lost its surface
inside every AIR cell, because air is in the same volume as bone.

This is V24's pattern for the third time: a rule that is per-point applied on one side of a
comparison and left behind on the other.

#### What found it, and what would not have

The equivalence was asserted rather than argued: over a jittered lattice, for EVERY volume
containing each point - which is the helper's precondition - owning the point and being what
`locate` returns must be the same thing. 528 of 6295 (point, containing volume) pairs disagreed,
on the first run, before any measurement was taken.

Nothing else in the suite would have caught it. `-benchmesh` builds one mesh in a world and has no
voxel grid. The coincident-face selftest is two boxes, and also has no grid. Both would have
passed, and the picture would have been wrong only for the one scene the feature was added for:
an object on the same layer as one class of a phantom.

The lesson is about how to choose the scene. When a function is replaced by a cheaper one CLAIMED
to be equivalent, assert the equivalence - and pick the scene by asking what could make the two
differ. Here the difference is a volume whose rank varies point to point, so the scene has to
contain one. A scene without one agrees for the wrong reason and says nothing.

#### What it was worth

Paired, two reps per arm in one session, `-benchmesh 1600000 <opacity> <shells>` - 3.2 million
triangles at 1680x960, the only difference between the arms being that one line:

```
                              locate()          owns_contained_point()      ratio
  opacity 1.00, 5 shells    12.92 12.09 ms          5.74  5.65 ms           2.20x
  opacity 0.40, 5 shells    81.92 82.18 ms         20.68 19.08 ms           4.13x
  opacity 0.40, 1 shell     27.00 28.95 ms         11.94 11.37 ms           2.40x
```

86.4 ms per frame to 24.2 ms on the case that was reported: 11.6 fps to 41.3. Within-arm spread
is 0.3% to 8%, so the ratios are well clear of it - which is the only reason two reps were run,
per V10 and V21.

The OPAQUE row was not predicted. An opaque volume ends the walk at its first surface, so the
saving was expected to be one parity count out of one - and it is, and one out of one is still
2.2x, because a `mesh_inside` over 3.2 million triangles is most of what the pixel costs.

What is NOT fixed: each composited surface restarts the whole search from a new origin, so a
five-shell mesh does eight separate BVH nearest-hit descents per pixel. A multi-hit traversal
collecting the first N hits in one descent would replace eight with one, and it is a larger win
than this. It is also a restructuring of the search rather than a fix to it, which is why it is
recorded here rather than attempted alongside.

### V27: the render was not slow, the frame was serial

"But the rendering shouldn't affect the GUI. It should be self-contained, no?"

It should, and it was not, and the reason was one line and one shared buffer. The panels are
rasterised by the CPU into `host_rgba`, which is also where the rendered viewport is copied to,
and between the two sat `cudaDeviceSynchronize()`. So the UI was not merely waiting on the render
out of laziness - it was structurally DOWNSTREAM of it. Every millisecond the render cost was a
millisecond before a hover highlight could be repainted, and at 82 ms of render that is a
highlight arriving a tenth of a second late on a scene one order of magnitude smaller than the
one that was reported.

Two rounds of making the render faster had gone before this question, and both were worth doing -
the read-back was 16.4 ms of nothing, and the ownership test was a mesh parity count per
composited surface. Neither addressed the actual complaint. A render made twice as fast still
holds the UI up for half as long, and the next scene is twice as big.

#### What decoupling costs, and what it does not

The render goes on its own stream and the frame POLLS a `cudaEvent_t` instead of waiting for it:
if the image is ready it is adopted and the next render goes out, and if it is not the frame
composites the previous one and carries on. The UI runs at the refresh rate whatever the render
costs. The visible cost is that in a slow scene the viewport is one render behind the camera -
which is what every application of this kind does, and is not what anybody was complaining about.

Two details that are the whole difference between this working and not:

**`cudaStreamCreate` returns a BLOCKING stream.** It implicitly synchronises with the legacy
default stream, so any ordinary `cudaMemcpy` or `cudaMalloc` anywhere else in the frame would
wait for the render - the stall removed from one line and reinstated by the allocator, invisible
in the code. `cudaStreamCreateWithFlags(..., cudaStreamNonBlocking)`.

**Pinned staging, sized to the WINDOW.** `cudaMemcpy2DAsync` out of pageable memory is not
asynchronous, so the read-back had to move to pinned memory or nothing would have changed at all.
But `cudaHostAlloc` pins pages and `cudaFreeHost` unpins them, which costs milliseconds for six
megabytes - and the VIEWPORT changes size on every frame of a splitter drag. Allocating with the
viewport would have put a fresh stall into exactly the interaction being fixed. The viewport is
never larger than the window, so one allocation per window resize covers every viewport, and a
render smaller than the buffer is handled by carrying its own dimensions alongside it.

#### The invariant is not the speed-up

`-benchmesh` now times the same scene twice in one process, render inside the frame and then
render on its own stream, which makes the comparison paired by construction. But the check in
build_all.bat is not the frame-time ratio: below the refresh interval both arms are vsync-limited
and the ratio says nothing, so a gate on it would pass every fast scene for the wrong reason.

The gate is that the UI frame spends UNDER A MILLISECOND on the render - a launch and a poll of
an event - which is true on any scene and false the moment a `cudaDeviceSynchronize` reappears in
`DrawFrame`. It needs no slow geometry to bite, so it costs the pipeline a small mesh rather than
a large one.

The selftest checks the BLIT against the device, and that is not the check it started as.

The first version compared a checksum taken with the render inside the frame against one taken
with it polled, and required them to be equal. That check is worth having - it is what says the
staging swap and the adopt deliver the image at all - but as the ONLY check it was vacuous about
the blit, and not as a worry: the row stride was broken by forty columns deliberately and the
comparison reported the picture UNCHANGED, because the reference had gone through the same blit
and was short by the same forty. A reference that shares the code path under test cannot fail.

So the blit is now compared against `d_rgba` itself, in sync mode where the device still holds
the image just composited - placement, stride and completeness, pixel for pixel. Immediately
after the blit and before the panels, because the gizmo is drawn INSIDE the viewport and a
comparison after it would report the gizmo as a mismatch. Both breaks are caught now: the short
stride by the guard saying the comparison never ran, and a forty-column displacement by 201,600
mismatched pixels. The checksum comparison called both of them fine.

The polling check stays alongside it, over eight frames rather than four - the selftest scene
renders in tens of milliseconds and only ONE render completed in a four-frame window on this
machine, which is one bad day from zero and a gate reporting a failure that is not one. It
carries a count of renders that actually completed, without which a matching picture would only
prove it never changed.

### V28: absence spelled as a very small number, and the two bugs that came free with it

A volume or a voxel class can now be set to the null layer, ∅, meaning it is not in the scene:
not transported through, not drawn, not scored. The feature is small. Getting it wrong was not.

#### The first implementation made absence a number

`kNullLayer = -2147483646`, a layer two billion below everything. The reasoning was that
`volume_rank` packs the layer into the high half of a signed 64-bit word, so such a cell ranks
below every ordinary cell AND below the world - which means `locate` hands its space to whatever
else contains it, with no new mechanism at all. One constant, no new arrays, no new branches on
the navigator's hot path. It looked like the cheap way to do it.

Two bugs came with it, and neither was written down anywhere as a rule:

**The renderer deleted the phantom.** The cell march asks, per cell, whether anything outranks
it and clamps the walk at that distance - and setting `covered` there means "a higher layer takes
the space from here on, hand the ray back to the outer search", which BREAKS the march. A cell
ranked below everything is outranked by the first candidate cover in the list, which for the
world is at zero distance. So nulling a phantom's front class ended the walk at the first cell
and took every cell behind it: the phantom disappeared, far side included. No line of code said
"stop here"; the rank said it.

**The cover scan admitted every volume in the scene.** That scan runs once per ray and collects
the volumes that could outrank the grid's LOWEST-ranked cell. With the null layer in that
minimum the answer is "all of them", the world included at zero distance - nearest, and so never
evicted from a list of four, where it can push out a cover that matters. The workaround was a
second minimum computed over the non-null classes, `layer_lo_live`, carried on every Volume.

That field is the tell. A feature that needs a parallel copy of an existing quantity, differing
only in which values it pretends not to see, is a feature encoded in the wrong place.

#### Absence is not a small number

The second implementation says so directly: `VoxelStore::class_absent`, one flag per class, and
one line in `inside_volume`:

    if (!v.has_absent_classes || v.solid.type != SolidType::kVoxelGrid) { return true; }
    return !voxel_absent_at(g.voxels, voxel_grid_of(v.solid), q);

A cell of an absent class is not contained by the volume. Everything else follows from that
rather than from rules of its own: `locate` cannot return the volume there, so the transport
steps through as whatever does own the space, and no scorer attached to the grid ever sees it -
the tally follows ownership and needed no code at all. A solid on the null layer is simply not
placed, which is the same idea one level up.

`layer_lo_live` was deleted. The renderer's special case at the grid's entry surface was deleted.
The overlap checker's rule about two null classes not clashing was deleted - an absent cell is
not inside either volume, so the sample loop never reaches the comparison. The correct model is
SHORTER than the workaround it replaced, which is usually how this goes.

One number remains, renamed `geom::kNullLayerTag`, and it is a tag rather than a layer: the value
the model uses in its own layer FIELD, present in the geometry only so the flattener can refuse a
placement that still carries it. Nothing computes a rank from it.

#### What found the bug, and what did not

Three checks were written for the voxel class, and the first two both passed on the broken
version:

  1. nulling a class changes the picture - passed, because deleting the whole grid changes it too
  2. recolouring the nulled class changes nothing - passed, because it was not there to recolour
  3. **recolouring a class that should have SURVIVED still changes the picture** - failed

Only the third distinguishes "this class is gone" from "the grid is gone", and it was written
last, after the first two were already green. Worth stating as a rule: a check that something is
absent must be paired with a check that its neighbour is still present, or "absent" and
"everything is absent" are the same answer.

Two more things the fixtures taught, both of which passed for the wrong reason first:

**A visibility precondition is not optional.** Every assertion of the form "removing it changed
the picture" is satisfied by a volume that was never on screen. Both new fixtures were first
placed at y = 600, outside a 500 mm world, where nothing is drawn at all - and the solid check
passed. The checks now open by recolouring the fixture and requiring the picture to move, which
is the one operation that can only show up if the thing is being painted.

**The fixture has to be able to tell the mechanisms apart.** A null class over a COVERED grid is
removed by the clamp whether the paint rule works or not, so breaking the paint rule deliberately
left that check green. The check moved to a grid with nothing over it, where the clamp never
fires. The same deliberate break now fails.

#### And the up direction, which invalidated a fixture two hundred lines away

The viewer's up axis moved from +y to +z, so that a detector described in beam coordinates - z
the beam axis, x-y transverse - is drawn the way it is typed. Elevation now lifts out of the x-y
plane, which also makes `/vis/viewer/set/viewpointThetaPhi` mean what Geant4 means by it, since
that command's theta is measured from +z.

It broke the covered-voxel check, which is the one selftest that reads GEOMETRY rather than a
flag: its camera was aimed "straight down -z" as azimuth 0, elevation 0, and under the new
convention that points down -y - across the classes the slab is meant to hide instead of through
them. Aimed down z it wants elevation 1.5, the clamp, 4.1 degrees off the axis; the slab
overhangs the grid by 20 mm on each side and 4.1 degrees over the grid's 80 mm depth is a lateral
5.7 mm, so the margin is not tight. Recorded because "change the up vector" reads like a
one-line change and had a consequence two hundred lines away in a file that does not mention
cameras.

### V29: the same bug in two more places, found by looking at the picture instead of the test

Three reports. The interesting one is the first, because the fix for it had already been written
and the test that verified it had been written too - and both had a hole in the same shape.

#### "The cone has the same artifact we solved for the sphere"

It did, and it was the same bug, reached through a function that had never heard of two of the
solid types.

V25 fixed a cancelling discriminant by starting each ray at the solid's bounding sphere, so that
the quadratic the generic engine solves has coefficients of order the solid's size rather than of
order the camera distance. The shift is gated on `geom::bounding_radius` returning something
positive - and that function is a switch over solid types with `default: return 0`. kPolycone and
kPolyhedra keep their (z, rmin, rmax) sections in the AUX POOL, which the p[]-only form cannot
see, so they fell to the default. Zero does not mean "no bound needed"; it means the shift is
silently off.

The builder's "Cone" primitive is a POLYCONE. Measured at its limb:

```
      camera     rays kept
      200 mm     100%
      600 mm      93%
     1500 mm      39%     <- the default camera distance
     4000 mm      22%
    12000 mm      17%
```

That is V25's signature exactly, in a solid V25 never looked at.

Booleans have the same hole and it is worse. `boolean_dist` hands each child the ray origin it
was given, so a union of two orbs solves the orbs' quadratics from wherever the caller started -
and a boolean node has no p[] extent either. At 1500 mm a union of two orbs kept 28% of its rays,
and at 12 m it kept 0.26%. The selftest scene has a subtraction in it, and it was speckled the
whole time.

Both are fixed by a store-aware overload: kPolycone and kPolyhedra read their planes out of the
aux pool, and a boolean takes the larger of its children's bounds plus their frame offsets. The
p[]-only form is left alone deliberately - it is on the transport's safety path, where returning
a larger number is a physics change, and this is a rendering question asked by code that has the
store in hand.

#### What the old test could not see, and why

The sweep that verified V25 covered orb, sphere, ellipsoid and tubs, and a note beside it said
box, trd and cons use closed forms and "were never affected - which the measurement said before
anything was changed". The measurement said nothing of the kind: those three were not in it. A
comment claiming a measurement that was never taken is worse than no comment, because the next
reader stops there.

Two things were wrong with the sweep and both had to be fixed before it could see anything:

**It swept a hand-picked list.** It now sweeps the enum - seventeen solids from p[], plus the two
aux-pool ones and the three booleans in blocks of their own. The list is the type list, not a
hunch about which types are quadratic.

**Its rays were all near the axis.** The first attempt at a polycone kept 100% of 2000 rays at
every distance out to 12 m, which said the shape was fine. It was not: the aim disc was 20 mm on
a 60 mm solid, and a cancelling discriminant shows at the LIMB, where the two roots are nearly
equal. Aimed at 98% of the silhouette the same shape lost 61% of its rays. The sweep now aims at
each shape's own material - an annulus per case, since a ray down the axis of a hollow tube never
enters it and double misses too - and out to the limb.

And the profile mattered. A monotonic taper kept everything; the builder's own Cone is a BICONE,
widest in the middle, with an outward crease where the slope reverses. The fixture is the
builder's own numbers now, for that reason.

#### The assertion for a boolean is FLATNESS, not a score

A boolean's surface has creases - where two orb surfaces meet, the roots really are nearly equal -
and a ray grazing one is a coin flip at any precision, worth about 2% on a deliberately grazing
aim. What the origin shift claims is not perfection, it is that the hit rate STOPS DEPENDING ON
WHERE THE CAMERA IS. So that is what is asserted, per operation: the ratio at 12 m must match the
ratio up close. Comparing the worst row against the best was tried first and compared two
different shapes rather than two distances.

#### The null symbol did not look right, and the test said it did

∅ is drawn rather than rasterised, because the atlas is indexed by byte and a three-byte UTF-8
character cannot reach it. The first version rasterised U+2205 into an extra atlas slot with the
wide GDI call, and the check for it read the ATLAS: ink present, and ink across the middle where
a notdef box has none. Both passed, and the glyph still looked wrong - twice.

First because a face missing the code point does not draw notdef. GDI FONT-LINKS to some other
installed face at that face's metrics, so the glyph arrives the wrong size on the wrong baseline,
which neither check could see. Canvas::EmptySet draws it now, out of an ellipse and a stroke, the
way EyeToggle draws its eye.

Then twice more, and each time the test was the problem:

  * clipping the stroke to the ring left a glyph indistinguishable from the 0 directly beneath it
    in the same menu. A circle with an invisible stroke is a zero. The stroke has to OVERSHOOT.
  * a fixed 0.9 px stroke half-width is a hairline at 25 px and thicker than the radius at 8,
    where the ring filled itself in and the menu showed a blob. Width scales with the glyph now.

The blob passed three successive tests. A probe straight up from the centre landed on the
feathered stroke, because at eight pixels across the interior is nine pixels and the stroke
crosses most of them - so the shape check was moved to 25 px, where the probe works and the
8 px glyph that was actually broken was never looked at. Then "encloses an empty pixel" passed
too: an over-thick ring leaves a scatter of single empty pixels, and nine is greater than zero.

What works is the LARGEST enclosed empty region, as a fraction of the glyph's own box, asserted
at both sizes: 14% for the drawn glyph at 8 px, 3% for the blob. A hole you can see is a hole of
some size.

#### A volume on layer 0 did not clash with the world

The overlap check skipped the world outright, on the reasoning that the world contains everything
by construction, so an overlap with it is containment rather than a clash. That is true for a
volume on layer 1 or above - and that case never needed the exemption, because the layer-range
test prunes it: the world spans layer 0 alone and cannot meet layer 1 anywhere.

What the exemption hid is a volume placed ON layer 0. That is a genuine same-layer overlap with
the world, resolved only by "whichever was added later wins" - which is the tie-break this whole
check exists to refuse, because it makes the answer depend on the order the detector was built
in. The world is one more volume to the general rule now, and the rule was already right.

#### The lesson that ties the three together

All three had a test that passed. The float sweep passed because it swept the wrong list with the
wrong rays; the glyph checks passed because they read the atlas and then probed the wrong pixel;
the overlap check passed because the case was excluded by name before the rule could see it. In
each case the code was believed because a green check was pointed at something adjacent to it.

What found all three was looking at the artefact - the render, the menu, the scene - which is
what the reports were. A test is evidence about the thing it measures, and every one of these was
measuring something else.

### V30: a cost in the wrong place twice, and a measurement that found it both times

Anti-aliasing the render. The feature is four rays instead of one at the pixels on a silhouette,
and the whole of the work was finding out where the time was actually going.

#### First guess: the edge test

`mark_edges` compared each pixel's COLOUR against its neighbours. Measured cost: +47% on an
opaque mesh and +78% on a translucent one - against a comment in the same commit claiming "a few
per cent, only edge pixels are retraced".

The reason is obvious once written down and was not obvious before: a 3.2 million triangle sphere
is shaded per triangle, so the colour steps between neighbouring pixels almost everywhere, and
almost every pixel got marked. Those steps are also not aliasing - a facet boundary is real
detail, and smoothing it blurs the model. What aliases is the silhouette, which shows as a jump
in COVERAGE (against the background) or in DEPTH (against other geometry), and a smoothly curved
interior has neither.

So the test became coverage and depth. And the cost did not move: 8.44 ms against 8.45.

#### Which is when the pass was made to report its own number

`mark_edges` now counts what it marks, and `-benchmesh` prints it. **0.2% of the frame.** The
refinement was tracing almost nothing, and the cost was not in the rays at all - so the colour
theory had been wrong about the mechanism as well as the fix, and two rebuilds had gone into
improving something that was not the problem.

With the count in hand, and a runtime toggle so both arms could be measured from one binary:

```
                  before AA    aa off     aa on
   opaque          5.70 ms     6.12 ms    8.50 ms
   translucent    19.88 ms    20.01 ms   35.47 ms
```

Two separate costs, and they had been added together and blamed on one. Lifting `trace_pixel`
out of the kernel so both passes could call it cost 5.70 -> 6.12 (codegen: the function is now
inlined into two kernels). The refinement itself cost 6.12 -> 8.50, which is 2.38 ms for 2950
pixels: 200 ns a ray, where an ordinary ray in the same frame is 3.8 ns.

#### Second guess, and this one was right

Fifty times the cost per ray is not arithmetic, it is occupancy. Edge pixels are a thin scattered
curve, so nearly every warp contains one or two of them - and a pass that reads a per-pixel flag
runs four full traces on that one lane while thirty-one idle.

`mark_edges` appends the marked pixels to a LIST instead, and the refinement runs one thread per
entry, so the rays are contiguous and the warps are full: 2.38 ms -> 1.87 ms opaque, 15.46 ->
10.31 ms translucent.

Still 158 ns a ray, and that residue is not divergence - it is that these are the most expensive
rays in the frame. A ray grazing a silhouette descends deep into the BVH without a quick hit,
while the "3.8 ns average" it was compared against is dominated by background pixels that cost
nothing. The two numbers were never comparable, which is worth stating because the 50x that
motivated the compaction was partly an artefact of dividing by the wrong denominator - the
compaction was still worth 25%, but the argument for it was better than the evidence given.

#### What the measurement discipline actually was, in order

  1. a comment asserting "a few per cent" with no number behind it - wrong by an order of
     magnitude
  2. a fix to the edge criterion, correct on its own terms, worth nothing on the clock
  3. a counter in the pass, which said the criterion was never the problem
  4. a runtime toggle, which separated the codegen cost from the ray cost
  5. compaction, worth 25%

Steps 3 and 4 are cheap and should have come first. Both are permanent now: the count is in
`-benchmesh`'s report and the toggle is in the Visualization window, so the next person to be
surprised by this cost has the two numbers that explain it without building anything.

#### The test is coverage, and the fixture had to be emptied first

One ray per pixel gives a pixel coverage of 0 or 255 and nothing between; averaging four gives
the values between. So the count of partly covered pixels IS the anti-aliasing, and it is read
from the framebuffer rather than from the picture - the resolve pass has already blended coverage
and colour into one number, and cannot tell "half covered by something bright" from "covered by
something dim".

The first attempt counted 143,829 partly covered pixels with the pass switched OFF. The selftest
scene has translucent volumes in it and several were in frame, and a translucent volume is partly
covering every pixel it touches. With everything but one opaque box hidden the floor is exactly
zero and the signal is the whole count: **1119 against 0**.

A box rather than a sphere, deliberately: its silhouette is four straight lines, which is the
arrangement one ray per pixel turns into a staircase and the one anybody notices.

### V31: the hole was the shape of the answer

Two reports. One was a pass the builder never launched, and one was a flaw in the surface search
that had been there all along and needed a very particular arrangement to show.

#### Wireframe did nothing in the builder

The builder's styles said `solid = visible && !wireframe`, which is right - a wireframe volume
must not be ray cast as a surface. Nothing then drew its edges, because the builder never
launched `render_edges` at all: `EdgeList`, which builds the twelve lines of a box, lived inside
`vis_manager.cu` where only the viewer could reach it. So a volume set to wireframe was invisible
in the builder, and the DEFAULT WORLD arrives with wireframe on and had no outline there ever.

`EdgeList` moved to `src/render/edges.h` and both apps use it. One implementation rather than
two, for the reason that keeps coming up here: a second copy is a second thing to fix, and the
one nobody remembers is the one that is wrong.

#### A volume overlapping a null voxel class drew as a hole

Reported as not being rendered in the overlap region, and the shape of the failure is what named
it: what appeared was a HOLE, exactly the silhouette of the volume that should have been there.
Not a missing volume - a volume-shaped absence, which means the march was handing the ray back at
the right place and the thing it handed to was never drawn.

`box_dist_in` starts its `tmin` at zero and clamps, so from INSIDE a box it returns exactly 0,
not infinity. A voxel grid is a box. The surface search therefore re-finds a grid the walk is
already standing in, at distance nothing, on every iteration - and a comment in that very loop
claimed the opposite ("a volume the ray is already inside has no entry surface ahead of it"),
which is why it went unexamined.

The tie rule normally hides it. The volume the march handed back to sits a nudge ahead, ties with
the zero, and wins on rank because it is a higher layer - which is what a cover is. Put something
on a LOWER layer inside the grid and the grid wins the tie, is re-entered, marches, clamps at
that volume again, and the pixel goes round until `kMaxLayers` with nothing accumulated.

A lower-layer volume inside a grid is exactly what a null class makes possible: while the class is
present the grid owns that space and the volume is correctly hidden, so nobody had ever put one
there. The fix is one line - a candidate at zero distance is a volume you are already inside, not
a surface ahead - and it is a fix to the search, not to anything about absence.

Two other things had to be true and were separately wrong:

**The cover scan rejected it before considering it.** That scan collects the volumes that could
outrank the grid's lowest-ranked cell, and a volume below the grid's lowest class layer fails
that test. An absent cell has no business being compared against - nothing is there, so anything
takes the space - so where the grid has absent classes every volume is a candidate.

**And the world had to be excluded from that scan.** For the same `box_dist_in` reason: the world
is a box the ray is always inside, so it enters the candidate list at distance zero, outranks an
absent cell (which ranks below everything), and the clamp ends the march at the very first cell.
Nulling one class deleted the whole phantom. That was caught by the check from V28 asking whether
the FAR class still responds to a recolour - written for a different bug and it earned its place
twice.

#### What actually found it

Three rounds of reading the code and reasoning about it produced three wrong diagnoses. What
settled it in one step was a screenshot: the cyan far class visible THROUGH the nulled near class
- so absence was working - with a black square in the middle the size of the box. Absence was
never the problem, and every minute spent reading the absence machinery was spent in the wrong
file.

Before that, one experiment worth more than the reasoning: put the box on a HIGHER layer than the
grid and it drew. That single bit - works above, fails below - is the whole signature of a rank
tie-break, and it was available for the cost of one rebuild at any point.

### V32: a justification is not a decision, and a second rendering nobody asked for

Four reports about the wireframe and the null layer, and three of the four were mine to begin with.

#### The comment that stopped the gap being looked at

`EdgeList` drew boxes and voxel grids and nothing else, and above it I had written that a bounding
cube round a cone "is not a hint about its shape, it is a lie about it - and the curved solids are
ray cast as surfaces anyway, so they need no outline". Every clause of that is true. None of it is
an argument for the code below it: it argues against drawing a *cube*, and what the code did was
draw *nothing*, so a sphere set to wireframe disappeared.

The test could not see it either. It asked whether turning the wireframe display off changed the
picture - and the default world is a box, so it did.

This is the second time in this project that a comment has protected the thing it sat above.
V31 was a comment in the surface search asserting that a volume the ray is already inside has no
entry surface ahead of it, which is the opposite of what `box_dist_in` returns; it went unexamined
for as long as it did *because* it was explained. A gap with a rationale over it reads as a
decision, and nobody re-derives a decision. The tell in both cases is the same and it is
recognisable: the comment argues for a *weaker* claim than the code makes.

What the work turned out to be is worth recording against the "it needs a real modeller" instinct.
A cylinder, a cone, a sphere, an ellipsoid, a paraboloid, a hyperboloid and a polycone are all
surfaces of revolution about z, so each is a table of `(z, rx, ry)` levels and one function draws
all of them. `kPara`, `kTrap` and `kTet` keep no dimensions at all - only half-spaces - and their
corners come back exactly by intersecting plane triples and keeping the points that satisfy every
plane. Forty lines, and it agrees with the engine's own containment test by construction because it
reads the engine's own planes. The para came out with corners at `x = dx + tan(alpha) y0 + tx z`,
which is not the naive `x = dx + tan(alpha) y + tx z` - `tan(alpha)` multiplies the *unsheared* y -
and that fell out of using `para_planes` rather than re-deriving it.

The one convention I did re-derive I got wrong: a polyhedra's `rmax` is the APOTHEM, so the
corners are `r / cos(pi/sides)` out, and drawn at `r` a hexagonal prism's outline sits 15% inside
the solid it outlines. Reading `polyhedra_planes` had not settled it - the offset is `r` and the
normals are at `sphi + step*(f + 0.5)`, and the half-step is the whole answer. What settled it was
a dozen lines that print `geom::inside` across azimuth and radius: inside to r = 23 at 0 and 60
degrees, inside only to r = 20 at 15, 30 and 45. The same lesson as the screenshot in V31 and the
higher-layer experiment before it - a probe that asks the code what it does beats another pass of
reading it, and it is usually available for less effort than the reading.

The verification of the whole set has the same shape. Every drawn vertex is required to be ON the
engine's own surface, bracketed rather than tested at the point: pulled 0.1% toward the axis it
must be inside, pushed 2% out it must not be. Bracketed because `EdgeList` stores floats, and an
exact corner round-tripped through float lands about 5e-7 outside a double-precision surface
tolerance - which says nothing about the geometry, and did read as 30 failures until I looked at
what the number was. The bracket also happens to identify an INNER surface, where it inverts:
the hollow tubs reports exactly 64, which is its two inner rings of 32.

#### One framebuffer, so one rule

`draw_line` ends in an `atomicMin` against the geometry's packed word. Among lines that is exactly
right: they are opaque, so the nearest one wins and nothing else about them matters. Against
*geometry* the same operation is a pure depth test, and it threw away every line behind a
translucent surface however little of the pixel that surface actually covered - 3% coverage still
won the min. Reported as not seeing wireframe behind transparent objects.

No tuning fixes that, because the operation cannot express the question. A depth test answers
"which is nearer"; what a translucent surface poses is "how much of the pixel did the nearer one
take". Two buffers and one composite in the resolve pass, where both depths and the coverage are
in hand.

And the reason it is a second BUFFER rather than a read-modify-write in the line pass: compositing
is not commutative, so two lines reaching one pixel in either order would give two different
pixels, and half the checks in this project are "did the picture change". A non-deterministic frame
would have made them unreliable in a way that shows up as flakiness weeks later, not as a failure.
`atomicMin` per buffer keeps every pass order-independent.

#### The null layer got a rendering of its own, and only one was wanted

The rest of it is a cost I created. A null voxel class needs the renderer told *something*, since
the grid is still in the scene and only some cells are gone - and what I built was a rule: an
absent cell ranks below every volume, so any cover clamps the march there and a volume sitting
inside the hole is drawn in it. That took a widened cover scan, an exclusion for the world, a
special case in the march, and three commits of debugging, one of which produced a volume-shaped
black hole (V31).

The requirement, when the user finally put it in one sentence, was: *turning off an object's
visibility renders it exactly how I want something in a null layer to be rendered.* There was
already a rendering for "not drawn". It works, it has worked for months, and routing the new state
into it is a one-line change in `build_scene.hh` - the class's colour arrives with zero alpha - plus
the *deletion* of everything above.

The lesson is a shape, not a detail. When a new state is meant to look like an existing state, the
implementation is to route it into that state, not to teach the renderer a second one; a second
behaviour that is reachable only through the new feature is a second thing to debug, and it is the
one nobody else has ever exercised. I did not ask which rendering was wanted, because inventing one
did not feel like a decision - it felt like implementing the feature.

Two smaller notes from the removal:

- **Withhold the input rather than branch on it.** `FloatGeometry::Build` does not copy
  `class_absent` or `has_absent_classes` into the render geometry at all. There is no branch to get
  wrong that way: `voxel_cell_absent` returns false with no array, and `inside_volume` is gated on
  the flag. Absence still reaches the transport, which shares the same arrays and is where a class
  not being in the scene has to mean something - no material, no step, no score.
- **The check has to compare the two pictures, not restate the rule.** "A null class is drawn as a
  hidden class is" is asserted by hiding the class and checksumming, then putting it back and
  nulling it instead, and requiring the two checksums to be *equal* - with "hiding it changed
  anything at all" as the precondition. A checker that instead re-encoded the expected pixels would
  agree with itself.

One thing the removal costs, stated so it is not rediscovered as a bug: a volume inside a nulled
class's cells on a lower layer than the grid is not drawn there. The grid still owns that space as
far as the picture is concerned - exactly as it does for a hidden class, which is the whole point.

### V33: hidden is not deleted, and zero distance from a boundary for the third time

Five reports. Two of them were mine from the commit immediately before, one had been mistaken for
a camera-distance problem and was not one, and the pair that looked like a single bug were two.

#### The check that could not see the value

The wireframe came out with red and blue exchanged, because `ui::rgb` packs `0xAABBGGRR` - which
is what an OpenGL RGBA upload wants on a little-endian host - and the framebuffer word is
`0xAARRGGBB`. One spelling, `vis::pack_rgb`, fixes it.

The interesting part is that this shipped with a test written for it. The test asked whether
recolouring the volume **changed the picture**, and every colour changes the picture. The user had
already reported "the wireframe colour does not match" once; I removed a brightness scale, wrote a
check for it, and shipped the swap underneath.

Third time in this project that an "it changed" check has passed over a wrong value: the empty-set
glyph was a blob that changed when recoloured, the anti-aliasing count moved for a scene full of
translucent volumes, and now this. **A check that something changed cannot check what it changed
to.** Where the right answer is a number - a colour, a radius, a count - the check has to name the
number.

#### A failure invisible to every check that compares two precisions

"The sphere still has speckling around the edges when I zoom in close." It is not about zoom, and
it is not about the origin shift that V30 added for the far field.

Float loses rays in a band at the limb whose width is a **fixed fraction of the radius**, at every
camera distance: 0.2 mm on a 40 mm sphere, measured at eleven distances from 1.02 R to 300 R with
no trend. Since the band scales with R, it occupies a fixed fraction of the drawn disc, so zooming
in makes it a wider ribbon of pixels - which is why it reads as a zoom problem.

The mechanism is a step along a ray not being a step away from a surface. At impact parameter q the
half-chord is `h = sqrt(R^2 - q^2)`, and stepping `d` back along the ray from the entry point
leaves the surface by only `d * h / R`. Where h is small that is less than `kSurfTolerance`, the
"before" probe reports **inside**, and `is_crossing_to` concludes there was no crossing. These are
not grazes: at the outer edge of the band the half-chord is 4.1 mm, so rays whose CHORD through a
40 mm sphere is eight millimetres long were being dropped.

    band  =  (kSurfTolerance / kProbe)^2  *  R / 2

What matters is the RATIO, not either constant. In double it is 1e-3 and the band is twenty
nanometres; in float it was 1e-1 and the band was 0.005 R. Raising `kProbe<float>` from 1e-3 to
1e-2 takes it to 2e-5 R - a tenth of a pixel with the sphere filling the window - and touches only
float, so the transport cannot move.

**None of the existing float-render checks could see it, and they all passed.** Every one of them
compares float against double, and a ray that BOTH precisions lose looks like agreement. What was
needed was an oracle, and an orb has one in closed form: a ray hits a sphere iff its perpendicular
distance from the centre is under the radius. That is the only test in this project that checks the
renderer against truth rather than against another implementation of itself, and it took twelve
lines.

#### Hidden is not deleted

Two reports that looked like one. A phantom on layer 1 under a translucent volume on layer 2 showed
the volume and nothing behind it; and a volume overlapping cells whose class was nulled was not
drawn. The second was a consequence I had documented in the previous commit as a deliberate cost.
It should not have been one.

The user named the invariant: *treat voxel volumes as independent voxels rather than a collection
unified under a mother volume.* Which is right, and it is what the ownership rule already half
said - a cell's rank comes from its class's layer - with the other half withheld. I had stopped
telling the renderer which classes are absent, on the reading that a nulled class should be drawn
the way a hidden one is.

It is. **A nulled cell and a hidden cell paint identically, and they say opposite things about the
SPACE.** A hidden cell is still the grid's and still outranks whatever is inside it; a deleted cell
is nobody's, so anything there owns it - including a volume on a *lower* layer, which is what makes
this unlike every other overlap in the scene. I collapsed the two because their appearance is the
same, and appearance was the only thing I checked.

The general form is worth keeping: when two states look alike, that is not evidence that they mean
alike. The question to ask is what each one claims about something other than its own pixels.

#### And `box_dist_in` returns zero from a boundary, for the third time

The translucent-cover bug is not about null at all. When a cell march is interrupted by a cover,
the surface search draws that cover and hands the grid back at the cover's **exit** - so the
resumed march begins standing exactly on the cover's far face. `box_dist_in` starts its `tmin` at
zero and clamps, so it reports the cover as beginning right there, the clamp fires immediately, the
march paints nothing, and the next iteration finds no candidate at all. The pixel keeps the cover
and nothing else. Opaque covers were unaffected, because the march is never resumed through one.

That single fact has now produced three separate bugs: the volume-shaped hole and the world
entering the cover scan (both V31), and this. It is not a subtlety, it is a **hazard class** in
this codebase, and it has the same shape every time - a distance of zero read as "starts here"
when it means "you are standing on it". Anywhere a distance is asked from a point that was derived
from a surface, zero means behind, not ahead. Both places that ask now say so in one line.

#### What made the difference: the renderer became testable

Every renderer bug in this project so far was found by taking a screenshot and looking at it. That
worked, and V31 records what it cost: three wrong diagnoses in a row, each a ten-minute GUI rebuild
apart, all of them in machinery that was working.

`vis::trace_pixel` is `__host__ __device__` now - two words, plus a `memcpy` for the bit cast that
`__float_as_uint` did - and `tests/test_render_layers.cu` builds a scene by hand, traces one ray,
and reads the colour as a number. Both of these reports reproduced there in a millisecond, and the
translucent-cover mechanism fell out of tracing the numbers by hand against the code, which is not
something a screenshot can support. It should have happened three bugs earlier; the reason it did
not is that the kernel being a kernel felt like a fact rather than a choice.

### V34: one decision too many, at two points in the same pipeline

Three reports, and the useful part is not any of the three fixes.

#### The same mistake twice, and a grep that could not find the second one

"Index 0 of a segmentation arrives non-visible" was reported, fixed, and reported again. The fix
was real: `ClassifyVoxels` gives every value the same opacity and leaves `visible` alone, and
`tests/test_voxel_import.cu` had already pinned it. The second report was a DIFFERENT piece of
code doing the same thing one step later - the colour-table reader:

    target->opacity = aa;
    target->visible = (aa > 0.0f);   // and this

An ICRP-style table gives air alpha 0, so importing a phantom with a table set index 0's opacity
to zero *and* flipped the checkbox. Same symptom, different file, and the test file asserted the
second one on purpose - "alpha 0 also clears visible" - with a comment explaining why it was
right.

Two things went wrong in the search, and both are repeatable:

- **I grepped for the symptom, not the principle.** `visible = false` never appears; the code
  says `visible = (aa > 0.0f)`. A literal search for the wrong state cannot find the general
  expression that produces it.
- **I looked for code that treats INDEX 0 specially, and there was none.** The reader hides any
  class a table calls transparent; index 0 is merely the class that tables call transparent. The
  bug was not "a rule about index 0", so a search framed that way was guaranteed to come up
  empty - and it did, three times, which I reported as "I cannot reproduce it".

The rule worth keeping: when a paternalistic default is removed, look for every place that makes
a decision OF THAT KIND, not every place that makes that decision about that input.

And the substance, which is the same both times: a table row gives an opacity. Zero opacity
already draws nothing, so honouring the number costs nothing; also clearing a control the user
owns means they have to discover it before raising the opacity appears to do anything.

#### A precise question was worth more than three of my searches

Having failed to find it, I wrote up a theory: the phantom probably came from a project saved
before the first fix, since `visible` and `opacity` are both serialised. The user asked one
question - *do you mean that the visibility toggle shows false when opacity is zero?* - and the
theory died on the spot. It cannot: the old rule set OPACITY, and the eye is drawn from `visible`.
The two are independent fields and I had written "keeps `visible = false` (or opacity 0)",
merging them, which is what made the theory sound plausible to me.

A conflated pair of fields in the explanation is a conflated pair in the reasoning. The tell was
in my own sentence and I could not see it until someone asked which of the two I meant.

#### When a predicate's meaning changes, every caller asked a different question

`inside_volume` became per-CELL in V33 - a cell whose class is not in the scene is not part of the
volume - which is right, and is what makes a nulled cell hand its space over. The renderer's cell
march has a branch gated on it that means something else: *have I already marched this grid?*

Those are not the same question, and after V33 one of them was being answered by the other. A ray
that stopped inside a translucent box sitting in nulled air was told it was not inside the grid,
took the not-inside path, asked `dist_in` from inside the grid's own box, got zero, and had it
rejected as a volume it was already in. The grid could not be found again at all, so the box
composited over the BACKGROUND instead of over the tissue - which against a dark background reads
as the box having gone opaque. Both features were right; their intersection was not.

The fix is to ask the question that was meant: for a grid drawn cell by cell, "standing inside it"
is a question about its BOX, and the per-cell decisions belong to the march. Ownership stays per
cell.

What is general here: changing what a predicate MEANS is not a local edit, even when every call
site still compiles and reads correctly. Each caller has to be re-read for which of the two
questions it was asking - and a caller that wanted the old meaning is now silently wrong, because
the name still describes what it does.

### V35: a fix applied to the case it was reported for

Two reports, and the first of them I had already fixed - for one solid type.

#### "Only when it is on a higher layer" is the whole diagnosis

A phantom under a translucent vest was invisible; the same phantom on a HIGHER layer than the
vest was fine. Two translucent boxes, same story: the far one could not be seen through the near
one.

That single bit - works above, fails below - names the mechanism, because a volume on a higher
layer is never COVERED, so it never has to resume. It is the third time an A/B on the layer has
identified a renderer bug in this project (V31 was "put the box on a higher layer and it drew"),
and it is worth naming as a signature: *works above, fails below* means the resume path or the
rank tie-break, and nothing else.

The mechanism: the surface search skips any volume the ray is already inside. That test earns its
place - without it the search re-finds the volume just entered, paints its front face again, and a
half-transparent box renders fully opaque. What it also did was drop a volume whose space a higher
layer had taken for part of the ray, because "already inside" is true there and there is no entry
surface ahead. So the cover could be seen through and nothing was behind it.

Any volume now resumes at the far side of whatever outranks it, with one extra containment test at
that point - a volume wholly inside its cover stops before the cover does, and offering it at the
cover's far side would paint a surface where the volume is not.

#### The restriction I did not question was my own, from one commit earlier

The branch that does this said, in as many words:

    // A GRID DRAWN CELL BY CELL IS THE EXCEPTION, because its interior is not one surface.
    if (!per_cell_grid(v)) { continue; }

I wrote that. Last commit I fixed the same bug for a voxel grid, reached the line, and treated the
`per_cell_grid` restriction as the shape of the problem rather than as the shape of the case I had
in front of me. The comment even explains why a grid is special - and being special is not the
same as being the only thing that needs to resume.

This is V32's lesson arriving from inside: a justification stops the gap being re-examined, and it
does so whether the justification is old or written five minutes ago. The tell is the same both
times - the comment argues for a weaker claim than the code makes. "A grid's interior is not one
surface" is true; "therefore only a grid resumes" does not follow, and the code said the second.

What would have caught it: when a fix lands for the case it was reported for, ask what else reaches
that line. There were two callers of the concept and I had a test for one.


#### And a mesh could not tell an exit from an entry

Fixing the resume for ordinary volumes did not fix the case it was reported for, because the cover
in that case is a MESH and a mesh does not go through that branch at all. It takes a fast path:
one BVH walk for the nearest hit and no containment test, deliberately - a containment test on a
mesh is a parity count, and paying one per composited layer is what made a 100k-triangle import
unusable. The note above it said the inside test could go because "with a strictly-ahead hit the
search cannot re-find the surface it just crossed".

True, and silent about the case that matters. From INSIDE a closed mesh the nearest hit ahead is
its far wall, and with nothing to classify it, it was offered as though the ray were entering
there. So the mesh composited twice - coverage 131 where one 77-alpha surface would give 77, which
is how it was confirmed - and that phantom entry sat at exactly the distance at which the volume
the mesh COVERS resumes owning the ray. Being the higher layer it won that tie, the covered volume
never resumed, and the next iteration found it standing in space it owned and skipped it for good.

Third weaker-claim comment in three commits, and the same shape each time: the sentence is correct
and it argues for less than the code assumes.

#### A comment that explained why information was being thrown away

The fix is the winning triangle's own normal, which is already being fetched to shade with - facing
along the ray means the ray is leaving. That needs the mesh's WINDING, and the first attempt simply
assumed outward.

Two things then went right for one reason: a twelve-line probe printing `inside`, `dist_in`,
`dist_out` and the hit normal at seven points along one ray. It showed `dist_out` on a mesh working
perfectly, which killed the hypothesis I had actually written the probe to confirm; and it showed
`n.d = +1` from OUTSIDE the mesh, which meant the test fixture I had just written was wound inward.
The fix was wrong and the fixture was wrong, and a measurement aimed at neither found both.

Then the real find. `mesh_volume` computes the signed volume by the divergence theorem and returns

    return std::fabs(v6) / real_t(6);   // |V| because the winding may be either way

That comment names precisely the fact that makes the sign worth keeping. Both windings occur in
real files - so the sign is not noise to be normalised away, it is the answer to a question the
renderer needs and cannot otherwise afford. It is now `mesh_signed_volume6`, with `mesh_volume`
taking the magnitude and `mesh_winding` taking the sign, and the sign rides in the mesh solid's
spare `p[7]`.

Worth generalising, because it is a different failure from the weaker-claim comment: **a comment
explaining why information is discarded is worth re-reading whenever something needs that
information.** This one was not wrong, it was answering a narrower question than the code later
asked. Zero in `p[7]` means nobody filled it in, and the renderer falls back to treating a hit as
an entry rather than guessing.

#### A vertex on the surface says nothing about the line between two of them

The wireframe drew a sphere's latitudes as 48-segment circles and its longitudes as one straight
chord per ring gap - nine rings meaning an eight-sided meridian. Reported as a mixture of round
and straight lines, which is exactly what it was.

The existing check could not see it. `on_surface` requires every drawn VERTEX to be on the solid,
and a chord straight across a curve has both of its ends on the surface too. Vertex-on-surface is
blind to the entire question.

What measures it is SEGMENT LENGTH relative to the solid's size: on a curve of radius R sampled
every angle t a segment is about R*t, so bounding the length bounds the angle each segment turns
through, and a chord across a whole meridian is half a circumference. Measured: 0.39 of the radius
before, 0.131 after, on both the sphere and the torus.

The torus was worse and in a way that reads the same. Each family of curve was drawn with the OTHER
family's station count - cross-sections got 8 segments and long-way rings 16 - so both families
were polygons on a solid that is nothing but curvature. How many curves and how many segments in
one curve are different numbers, and using one for the other looks plausible right up to drawing
it.

And fixing the sampling exposed a direction bug in my own bracket. `on_surface` pushed x and y out
by 2% and shrank z by 0.1%; near a POLE the outward direction is almost entirely z, so that moves
the point inward, and it read as 48 vertices of a sphere not being bounded by their own surface.
It appeared only once the profile was sampled finely enough to have vertices near a pole - the
check had been right for nine samples and wrong for thirty-three, without changing. A bracket
scaled about the origin, all three coordinates together, is the radial direction for every solid
these tests use.

### V36: a gate that could not pass

V29 and V32 are about checks that cannot fail. This is the mirror image, and it was found by a
speed-up: the benchmesh gate that asserts the render is decoupled from the UI frame demanded
`async < 0.9 * sync` once the synchronous render exceeded 18 ms. The UI frame cannot go below the
refresh interval - 16.6 ms at 60 Hz - so for any render between 18.0 and 18.5 ms the ratio sits
at 0.90-0.92 and the gate fails **while the frame is exactly as decoupled as it is at 22.9 ms**,
where it passed. V33's exit cull made the mesh render about a quarter faster and moved the
benchmark into that band; a run failed on a change to log-file names.

The ratio was also too weak at the other end. Against a 40 ms render it would have accepted a
35 ms UI frame - twice vsync - as decoupled. What the gate means is one sentence: *the UI frame is
at the refresh interval whatever the render costs*. A coupled frame is at least the render, and the
branch condition has already put the render above 18 ms, so a UI frame under 17.5 ms is decoupled
at every render cost the test reaches. That is the assertion now, and it names the ceiling in its
output.

Two lessons. A criterion has to be checked at the edges of the region it is applied in, not at the
one operating point it was written at - 22.9 ms passed, and nobody asked what 18.1 would do. And a
speed-up is a change to the inputs of every timing gate downstream of it; the commit that made the
render faster should have been the one that re-read this gate, and was not.

Left open when the GUI work was parked (2026-09-10). At build_all's benchmark size the render now
sits on the entry threshold - 17.8 ms one run, 18.3 the next - so the payoff check is reached on
some runs and not others, and the structural invariant (the UI frame spends under 2 ms on the
render) is what holds every run. The fixed 17.5 ms ceiling is also load-sensitive: a UI frame at
vsync is 16.6 ms only on a quiet machine. The complete fix is to time a baseline phase with
nothing to render in the same process, require the asynchronous frame to sit nearer that baseline
than the synchronous frame, enter the check only when the two are separated by a few milliseconds,
and size the benchmark so it is entered every run - failing when it is not, so that the next
speed-up cannot make the check vacuous silently. By hand, renders of 20.0, 22.4, 23.8 and 29.6 ms
left the UI at 16.3-16.5 ms against the ceiling: the criterion holds where it applies.

---

### V37: the oracle was wrong, and it was wrong in the direction that looks like a port bug

Four elastic models, eighteen failing assertions, and seventeen of them were the oracle's fault.

#### What the file said and what the file meant

`ref/dump/dump_elastic.cc` runs each elastic model under a prescribed eight-value uniform cycle
so that `ApplyYourself` becomes a deterministic function of its inputs, and records what came
out. It recorded `-t` like this:

    o.t = 2.0 * mass2 * o.erec;   // -t = 2*M*T_rec, exact for elastic scattering at rest

That identity is exact. The evaluation is not. Geant4 gets the recoil energy by subtracting
four-momenta - `lv -= nlv1; erec = max(lv.e() - mass2, 0.0)` - and for a 6 GeV alpha on Pb207 at
a small angle that is two cancellations of ~1.97e5 MeV energies to produce 3.2e-10 MeV. One
significant digit. The `t` column was therefore carrying up to **8% error** on a value the test
compared at 1e-13, and 7e-8 error on the well-behaved rows.

The failure report read exactly like a port bug, and better than a port bug would have: a
consistent 7e-8 on three models, which is the size of the `G4Pow`-versus-`std::pow` difference
this project has been bitten by before (HADRONIC_PLAN section 8), at 1e-7 the obvious first
suspect. A long detour went into G4Pow, `G4Exp`, `G4Log`, `pz13[]` and `expA` before the pattern
that settles it: **the recoil ENERGY agreed to 2.3e-10 while the momentum transfer derived from
that same recoil energy disagreed by 7e-8.** A derived quantity cannot be less accurate than what
it was derived from unless the derivation is the problem.

`SampleInvariantT` is public and virtual on `G4HadronicInteraction`, and `pLocalTmax` - the only
state it needs - is left set by the `ApplyYourself` immediately before. So the dump now calls it
directly with the engine wound back to the same phase and records the model's own `-t`. With that
one column fixed, and two ulp-level fixes in the port (below), all four models agree with Geant4
**bitwise** on `-t`, cos(theta_cm), the final energy, the recoil energy and both direction
vectors, and the tolerance went from 1e-13/1e-9 to 1e-15. The reconstructed value is still in the
file, as `t_from_erec_MeV2`, and the test prints how far it is from the real one - so the reason
the comparison is not made against the recoil is a number in the output rather than a claim in a
comment.

#### An oracle is code, and this project had not been treating it as code

Every discipline this port applies to `src/` had skipped `ref/dump/`. The identity in that line is
right, the comment above it is right, and nothing in the pipeline asked whether the arithmetic
could deliver it. A dump that reconstructs a quantity instead of reading it is doing physics, and
physics in the oracle is not checked by anything - by construction, since it IS the check.

The rule that follows: **a dump reads values out, it does not compute them.** Where a value is
only reachable by reconstruction, the reconstruction's precision has to be stated and the
tolerance derived from it, not from what the port can achieve. And where the real value is one
public virtual call away, make the call.

#### The one real bug, which the oracle's noise was hiding

`G4ChipsNeutronElasticXS` asks two different questions about the same target:

    GetQ2max      line 2101:  if(tgZ==0 && tgN==1)     <- a free NEUTRON
    GetPTables    line 1611:  if(tgZ==1 && tgN==0)     <- a free PROTON
    GetTabValues  line 2021:  if(tgZ==1 && tgN==0)
    GetExchangeT  line 1878:  if(tgZ==1 && tgN==0)        "===> n+p=n+p"

So for a neutron on free hydrogen, Geant4 takes the two-channel np parameter row and the
two-channel np t-sampling - the one that carries the u-channel charge-exchange term - but computes
(-t)max from the nuclear Mandelstam expression with `mt = m_proton`. `G4ChipsProtonElasticXS` is
consistent; the neutron class is not. The port had one flag for both and sent n+p through the
four-channel nuclear sampler: a factor of 20 in `-t` at 3 GeV/c and a 38-sigma histogram, on
**hydrogen, for neutrons** - the single most important elastic channel in a water phantom, since
np elastic is how a fast neutron deposits dose. Two flags, one per function, reproduce both.

That failure was in the same report as the seventeen artefacts and indistinguishable from them at
a glance. Noise in a test does not merely waste time; it *conceals*, and it conceals in proportion
to how much of it there is.

#### And two ulps, one of which matters and one of which does not

`HepLorentzVector::boostVector()` is `pp * (1./ee)`, not `pp/ee` (LorentzVector.cc:189). Writing
the division instead puts the recoil direction 5.8e-13 out and the alpha-on-Pb recoil energy
2.3e-10 out - visible, because the recoil is a cancellation and a cancellation amplifies an ulp by
the ratio of the operands to the result. `Hep3Vector::unit()` is `p *= (1.0/std::sqrt(mag2()))`,
also not a division; that one is reproduced as well and makes **no measurable difference** on any
of the 1408 points. It is in the code because it is what CLHEP does, and its comment says exactly
that rather than claiming it fixed something - a check that cannot fail must not be described as
one that can.

#### Two assertions that could not fail, found by trying to make them fail

The anti-vacuity pass (working rule 7) does not only confirm that a check works; twice here it
found that a check could not.

`G4EnergyRangeManager::GetHadronicInteraction` branches on `emi1 < emi2` and has two arms.
Registering the lower-threshold model first - which is what a physics list does and what the dump
did for both of its overlap pairs - always makes the upper model the most recent match, so only
the second arm ever runs. Inverting the first arm's ternary in the port changed no test result.
The comment over the dump's registration claimed both arms were covered *because* there were two
pairs, which was a plausible-sounding reason attached to a fact that was not true. Each pair is
now registered both ways round: 28 points instead of 14, and the mutation fails 6 assertions.

`G4HadronicProcess::FillResult` tests `stopAndKill` FIRST and only then reads a zero final energy
as a stop. Swapping those two branches is behaviour-preserving *except* for stopAndKill together
with zero energy on a particle that has at-rest processes - where Geant4 kills the track and the
swapped order hands it to the at-rest chain a model explicitly asked to end. The test had the
positive-energy case, where both orders agree. A stopped pi- is the particle that distinguishes
them, so this was a live gap on the exact hook P4's decay and P12's absorption compete on.

Both are the same shape as V32 and V35: a gap with a rationale over it reads as a decision. The
new part is that **inverting a branch and finding that nothing failed is a positive result**, and
the cheapest way there is to mutate one line and rebuild.

#### The CHIPS `lastTH` latch, recorded rather than reproduced

`G4Chips{Proton,Neutron}ElasticXS::GetChipsCrossSection` keeps a per-isotope threshold:

    if(lastCS<=0. && pEn>lastTH) lastTH=pEn;

Once the parameterisation has gone non-positive at some momentum, every LOWER momentum for that
isotope returns zero for the rest of the run - so the answer depends on the order in which
momenta were asked for, over the whole job. For pA the expression is a sum of positive terms and
cannot trip it; for pp it can, because `(par1 + par2*dl1^2 + par4/p)/(1 + 0.425*lp)/(...)` changes
sign at `lp = -1/0.425`, i.e. p = 95 MeV/c. A device port has no per-isotope run history and
cannot have one, so `chips_cross_section` computes the value and reports `non_positive`, leaving
the latch to a caller that has somewhere to keep it. Below ~100 MeV/c on hydrogen the port and
Geant4 can therefore disagree about whether the cross section is zero, depending on Geant4's
history. It is 100 MeV/c protons on hydrogen; it is recorded here because "depends on the order
of previous calls" is not a property anyone expects a cross section to have, and the next person
to compare a low-energy proton on water needs to know it is there.

---

### V38: ask the code what it does; a comment saying it is unreachable is not a measurement

The decay package (`docs/HADRONIC_PLAN.md` P4). Three of its claims were wrong, all three had a
rationale written above them, and all three were settled by running something rather than reading
something.

#### An at-rest length of zero is not a competitor

The plan says a stopped pi- is captured rather than decayed - `G4HadronicAbsorptionBertini` is an
at-rest process, its length is zero, capture wins - and it says a stopped mu- "competes between
capture and decay". `src/physics/decay/decay.cuh` expanded that into a paragraph: a real
competition, both processes with a finite at-rest length, the stepper's ordinary smallest-wins
rule deciding, most stopped mu- decaying in a light material and being captured in a heavy one.

Every clause of that is a plausible description of muon physics. None of it describes the code.
`G4MuonMinusCapture` derives from `G4HadronStoppingProcess`, and
`G4HadronStoppingProcess::AtRestGetPhysicalInteractionLength` is `return 0.0;` - it does not read
the track, the material or anything else (line 115). So it pre-empts `G4Decay` exactly as Bertini
does, for the muon as much as for the pion, and G4Decay's at-rest branch never runs for any
negative species. The capture-versus-decay split for a stopped muon is *inside* the capture
process: `G4MuonMinusBoundDecay` computes lambda_c(Z, A) from Suzuki et al. and a bound decay
rate, samples the time from their sum, and on the decay branch runs its own K-shell Michel
sampler with the bound energy subtracted. Different sampler, different process, different package.

What settled it was `ref/oracle/decay_atrest.csv`: walk every particle's
`GetAtRestProcessVector()` and ask each process for the length it would offer a stopped track.
Twelve lines of dump. And the first version of those twelve lines reported *minus the lifetime*
for every unstable species, because `G4Decay::AtRestGetPhysicalInteractionLength` is
`theNumberOfInteractionLengthLeft * GetMeanLifeTime` and that member is -1.0 from `G4VProcess`'s
constructor until `StartTracking` draws it. A number that is exactly minus something recognisable
is a state error, not a physics error.

This is V32's pattern for the third time (V31 and V35 are the others): the comment argues for a
*weaker* claim than the code makes. "Bertini's length is zero so capture wins" is true and is not
an argument about the muon; the paragraph that generalised it into a race was written from what
muon capture is, not from what the class does.

#### A parameter with no check on it, behind a comment explaining why it needed none

`G4KL3DecayChannel` carries two form factor parameters, pLambda and pXi0, chosen by (parent,
lepton) in its constructor. `tests/test_decay.cu` said they are private with no accessor, so the
only thing that can see them is the shape of the sampled pion spectrum, and that this is why the
40-bin Poisson comparison exists - "substituting K0L's pLambda = 0.0300 for K+'s 0.0286 has to
fail it".

It does not. Both halves were false and the anti-vacuity run is what said so. `-perturb
kl3-lambda` and `-perturb kl3-xi0` substitute K0L's values and **passed every assertion in the
file**. They pass for a reason that is obvious once measured: F = 1 + lambda*q2/m_pi^2 moves by
1.6% at the edge of the Dalitz region and less inside it, so the pion spectrum moves by parts in
a thousand - a third of a sigma per bin at 400,000 samples - and Ke3's pXi0 multiplies
m_e^2 = 0.261 MeV^2 against a coefficient of order m_K^3, which no sample size will ever resolve.
Reaching 5 sigma on the lambda effect needs about 1e8 samples per channel.

And they were reachable all along: `GetDalitzParameterLambda()` and `GetDalitzParameterXi()` are
public inline accessors (`G4KL3DecayChannel.hh:52`), and `DalitzDensity` is *protected*, so a
three-line derived probe re-exports it - the same trick the dump was already using on `G4Decay`'s
protected lengths, in the same file. The density is now dumped on a 606-point grid and agrees
bit-for-bit, which pins both parameters and the whole Chounet expression to machine precision.

The general lesson is the one the plan already states and this is the sharpest instance of it so
far: **a statistical check is not a substitute for a deterministic one, and "the only thing that
can see it is the distribution" is a claim to verify, not to assert.** If a parameter can be read
out of Geant4, read it. The perturbation harness is what turns that from an opinion into a
measurement, and it only works if the perturbation is the *real* alternative value - 0.0286
against 0.0300, not against 0.1.

#### An assertion asking whether a string literal is non-empty

Also found by the same pass: `require(why != nullptr && why[0] != '\0')` on a refusal message,
where every arm of the function's switch returns a string literal. It cannot fail. It is replaced
by a comparison of each named refusal against the catch-all message, which fails if a case is
deleted from the switch.

#### And a channel selector that starves channels that are open

Not a comment problem - a transcription that would have been wrong and had nothing to catch it.
`G4DecayTable::SelectADecayChannel` draws `br = sumBR * rand` against the sum of the channels that
pass `IsOKWithParentMass`, then walks the table accumulating `sum += GetBR()` over **every**
channel and only then tests the mass. Two consequences that a clean rewrite does not have: a
closed channel's slice of the cumulative axis is absorbed by the next open channel after it, and
any channel whose cumulative bound lies past sumBR is never selected at all. Measured for kaon+ at
0.84 of its PDG mass, where pi+pi+pi- is shut: Ke3 fires 10.68% of the time instead of 5.37%, and
Kmu3 fires **0 times out of 200,000** with a branching ratio of 0.0335 and IsOKWithParentMass
true.

At the PDG mass none of that is visible - every channel is open and the frequencies are just
BR/sumBR - so a test that only sampled at the nominal mass would have passed either
implementation. The reduced masses exist in `decay_select.csv` for exactly that reason, and
`-perturb select-normalise` (the clean rewrite) misses by 61 sigma. **A branch that only differs
off the nominal input needs a test off the nominal input**, and for a decay table that means a
parent mass that shuts something.

#### One transcription defect worth its own line

`0.493677 * gev()` is not `493.677`. The kaon rows carried the decimal literal and the table
comparison reported a worst relative deviation of 1.15e-16 against a 1e-15 tolerance - passing,
and the only nonzero number in a block of twenty. Writing every mass and width as the Geant4
source writes it, product and all, takes every deterministic comparison in the file to exactly
zero. The same class of error as V8, one ulp instead of 1e-8, and the tell was the same: one
number in a table that should have been zero and was not.

---

### V39: a constant written the short way, three times, and the subtraction that noticed

Three findings from the P2 hadronic cross sections (`phys/xs`), and they are one finding: **a
physical constant that Geant4 computes must be computed the same way, not pasted as the decimal
it comes to.** Each of the three was invisible to every test that multiplied by it and visible
to the first one that subtracted.

#### `0.493677*GeV` is not `493.677`

They are adjacent doubles - 493.67699999999996 and 493.67700000000002. `G4KaonPlus.cc` writes
the first. `projectile.cuh` wrote the second, and every Coulomb factor for a kaon disagreed with
Geant4 by **1.4e-12**, a thousand times the comparison tolerance.

The amplifier is `G4NuclearRadii::CoulombFactor`, which computes

```
totTcm = sqrt(pM*pM + tM*tM + 2*pElab*tM) - pM - tM;
return (totTcm > bC) ? 1. - bC/totTcm : 0.0;
```

At 1 MeV that subtracts 1432 MeV from 1432 MeV to get 0.65 - a cancellation of about 2200 - and
then `1 - bC/totTcm` near its threshold divides by 0.11, another factor of 8. One ulp in, 1.4e-12
out. Every hadron mass in the port is now spelled `x.yz * GeV` as its `G4*.cc` does. Only the
kaon actually differed; the pion, the lambda, the neutral kaons and the four light ions come to
the same double either way. They are written that way so that the next one cannot.

#### `barn` was the literal 1e-22 and CLHEP derives it

CLHEP: `meter = 1000.*millimeter; meter2 = meter*meter; barn = 1.e-28*meter2`. That is
9.9999999999999993e-23. The literal `1e-22` is 1.0000000000000000385e-22 - one double higher,
1.148e-16 relative.

`ref/oracle/constants.csv` has carried CLHEP's own value from the beginning and
`tests/test_constants.cu` compares against it **at 1e-15**, which is exactly loose enough not to
notice. Four call sites used the constant and none of them could see it: they all multiply a
tabulated cross section by it.

The fifth was `G4ComponentGGHadronNucleusXsc`, whose elastic cross section is
`fTotalXsc - fInelasticXsc`. For a pi- on Li7 at 121 keV that difference is 1/371 of either
term, so 1.148e-16 in the millibarn scaling both came out as **2.2e-12** in the elastic - and
read exactly like a wrong per-Z bar-correction table, which is what it was blamed on for an
hour. Found by inverting Geant4's own published total for the `sigma` it must have had and
substituting the CLHEP-derived millibarn: bit-exact.

`src/core/units.cuh` is shared, so the fix was verified rather than argued: all 28 host tests
that exist were rebuilt and rerun, and all pass.

**The generalisation.** A tolerance is only a tolerance for the quantity it is applied to. A
1e-15 gate on a constant is a 1e-12 gate on anything that subtracts two numbers built from it,
and the tolerance that catches a constant should be the tightest one downstream of it, not a
comfortable one. Where a comparison is a difference of two nearly equal numbers - GG elastic,
`hpInXsc = total - elastic`, `difratio - log(1+difratio)` - the honest thing is to say so and
compare the inputs too; `tests/test_hadronic_xs.cu` compares all five GG columns for that
reason, and its worst residual (3.5e-13) is a cancellation with bit-exact inputs.

#### And a Geant4 asymmetry that is not arithmetic at all

`G4BGGNucleonElasticXS::CoulombFactor(kinEnergy, Z)` is a member function that branches on
`isProton` - a per-instance flag - and takes no particle argument. `BuildPhysicsTable` divides
**both** `theCoulombFacP[]` and `theCoulombFacN[]` by it. Those arrays are `static`, and the
build returns early once `theA[0]` is set, so exactly one instance ever fills them: the first
one initialised, with its own `isProton`.

So a neutron instance whose tables a proton instance built returns
`barashenkov_n(14 MeV) / CoulombFactor_proton(14 MeV, Z)` below 14 MeV - a factor 2.13 at
Z = 92, and a factor of ten for the inelastic class's neutron at 1.8 keV. The port divided each
column by its own projectile's factor, which is what the code looks like it means, and was 53%
and 90% wrong there. `bgg_build_nucleon_table` now takes `built_for_proton`.

**Worth generalising:** where a Geant4 class has static per-Z tables and a per-instance flag,
the tables belong to whichever instance was constructed first, and *which* instance that is
belongs in the port's interface rather than in a comment. QBBC constructs the proton one; the
default says so and the argument makes it changeable.

### V40: two answers to "the highest level of this nuclide", 952 of them different

`G4NuclearLevelData` carries a compiled table of maximum level energies. The comment above it
says where it came from - "obtained from PhotonEvaporation5.2" - and the dataset the same class
reads at run time is **PhotonEvaporation5.7**. Two answers to one question, and they disagree
for 952 of the 3,108 nuclides that have level data at all: 798 where the compiled value is
LOWER than the data's, 166 where it is exactly zero and the data has levels, worst gap 36 MeV
(He6: compiled 0, read 36).

Neither is dead. `GetLevelEnergy` snaps an excitation onto a tabulated level only when it is
below the COMPILED maximum, so for those 798 nuclides the snapping is skipped inside a band
where levels demonstrably exist. And `G4FermiFragmentsPoolVI::Initialise` skips a fragment when
`MaxLevelEnergy(Z,A) == 0.0f && LifeTime(0) == 0.0f`, reading the compiled table - so **which
fragments the Fermi break-up pool contains is decided by the 5.2 numbers** while the cascade
that runs on them reads the 5.7 ones. Delete the skip and the pool holds 1,119 fragments
instead of 991: 128 fragments enter, 336 applicability flags change, 734 channel counts change.

#### What is general

A compiled constant and a data file that answer the same question are a *pair*, and the pair is
a version pin whether or not anyone wrote one down. The tell here was cheap and I nearly did not
spend it: dump both answers for every nuclide and subtract. 952 rows came back. A sampled
comparison would have shown agreement, because the nuclides a human picks are the well-measured
ones and those are the ones the two versions agree about.

The port reproduces each where Geant4 uses it, and `tests/test_deex_levels.cu` measures the
disagreement rather than assuming it away - 3,108 max-level-energy rows and 3,188 level counts,
compared for EVERY nuclide and not for a sample. The same test found that mis-consuming one
transition's ten internal-conversion coefficients drops the dataset from 174,411 levels to
8,254, and a forty-nuclide sample would have called the remaining three thousand fine.

### V41: two of the sixty GEM channels are built out of two different nuclides

A GEM evaporation channel is two objects - a `G4GEMChannel` carrying an (A, Z) and a
`G4GEMProbability` carrying another - and they are meant to be the same nuclide. For 58 of the
60 they are. For two they are not:

    G4Be12GEMChannel   (12, 4)      G4Be12GEMProbability   (9, 4)
    G4O17GEMChannel    (17, 8)      G4O17GEMProbability    (17, 9)

Both are in the default channel set, and each pair splits its channel down the middle. The
CHANNEL's nuclide decides what is emitted, its mass, its Coulomb-barrier object and the residual
the parent becomes. The PROBABILITY's decides the residual the width is integrated for, the
nuclear mass in the spin factor, the alpha and beta parameters and Furihata's Rb. So the Be12
channel emits a beryllium-12 at a beryllium-9 channel's rate, and the O17 channel emits an
oxygen-17 at the rate of a channel whose residual has one proton fewer. Unifying each pair to
the channel's nuclide moves the GEM emission probability by 79%.

Be12's probability object is a copy of Be9's header with the mass number edited, so it also
carries no excited states where the real Be12 file has them - one transcription, two defects.

#### What is general

Two constructors, two hand-written constants, one invariant between them and no code that
checks it. There are sixty of these classes and they were certainly generated by editing the
previous one; that is exactly the population in which a single-token error survives review,
because every file looks like its neighbour.

`tools/extract_gem_tables.pl` asserts that these two and only these two disagree. That is the
form worth copying: not "skip the known-bad rows" but "assert the known-bad set is exactly this
set", so a Geant4 release that fixes them fails the extractor loudly instead of being
transcribed as though nothing had changed.

### V42: the pointer was written before the object existed, and it cost 9% to 36% of a spectrum

`G4GEMProbability::GetCoulombBarrier(fragment)` returns exactly zero for every one of the sixty
GEM channels, and the reason is an initialisation order:

    class G4Li6GEMChannel : public G4GEMChannel {
     public:
      explicit G4Li6GEMChannel() : G4GEMChannel(6, 3, "Li6", &theEvaporationProbability) {}
     private:
      G4Li6GEMProbability theEvaporationProbability;   // declared AFTER the initialiser
    };

The base constructor runs first and its body does
`theEvaporationProbabilityPtr->SetCoulomBarrier(theCoulombBarrierPtr)`, writing into storage
whose object has not been constructed. The member is constructed next, and
`G4GEMProbability`'s own constructor initialises that pointer to `nullptr` - discarding the
write. Every later `GetCoulombBarrier` call takes the `if (ptr)` branch as false. All sixty
default channels are written this way, and so are the six GEM light-ejectile channels.

#### Why no probability test could have found it

In `CalcProbability` the barrier appears only as `(Beta + V)` with `Beta = -GetCoulombBarrier`,
so the pair is zero whether the pointer works or not; its other appearance, as the energy
argument of the level-density parameter, is inert because the default level-density flag makes
that parameter energy-independent. That is why all 4,320 GEM emission probabilities in
`tests/test_deex_models.cu` come out exact with either behaviour.

`G4GEMChannel::SampleKineticEnergy` uses **Beta alone**, as the linear prefactor
`ConstantFactor*(KineticEnergy + Beta)`. There the difference between `KineticEnergy` and
`KineticEnergy - CoulombBarrier` moves the mean kinetic energy of an emitted GEM nucleus by 9%
to 36%: with the barrier subtracted the spectrum starts at zero at the barrier, without it there
is finite weight at the barrier itself. Ca40 at E* = 80 MeV emits a Li6 with a mean of
10.19 MeV in Geant4 and 13.15 MeV if the barrier is subtracted.

#### What is general

A quantity that cancels in one consumer and not in another is invisible to any test that only
exercises the first, and "the probabilities are exact to the last bit" reads like proof that the
inputs are right. It is not: it is proof that the inputs are right *up to the null space of that
function*. The defect surfaced only when a SAMPLER was compared - `deex_channel_spectrum.csv`
fixes the channel and dumps 20,000 emitted kinetic energies per channel - and the shape of the
evidence named the cause on its own, because a 9% shift in the mean with a bit-exact width can
only come from a term the width is blind to.

Reproduced rather than corrected, because it is what the installed Geant4 computes; the
expression the working pointer would have supplied is written out in the port beside the zero,
so a release that fixes the order needs no rediscovery.

### V43: a process the reference does not register, and two predicates that were lists

Species and transport plumbing (P1 of `docs/HADRONIC_PLAN.md`): eleven charged hadrons, two
neutral ones and six neutrinos joined the fourteen species this transport had. Nothing here is a
rendering bug or a sampler bug. All three findings are the same shape - **a question about the
physics list that was answered by reading Geant4's source instead of by asking the object Geant4
built** - and the fix was the same in each case: dump the constructed `G4ProcessManager` and
compare.

#### The port has been running G4NuclearStopping. QBBC does not register it.

`step_hadron` applied `G4ICRU49NuclearStoppingModel` along every step of every charged hadron,
unconditionally, since the day the hadron stepper was written. QBBC attaches that process to no
particle at all.

Reading `G4EmStandardPhysics.cc` and `G4EmBuilder.cc` gives the opposite impression, and reading
them carefully gives it more strongly, because the source really does contain a four-way split:
one `G4NuclearStopping* pnuc` handed to GenericIon, then to the proton by
`G4EmBuilder::ConstructCharged` and to the alpha and He3 by `ConstructIonEmPhysics`, passing over
the deuteron and triton two lines above them and never reaching mu+-, pi+-, K+- or pbar. That
split is real. It is also downstream of three lines four functions further up:

```
G4double nielEnergyLimit = param->MaxNIELEnergy();
G4NuclearStopping* pnuc = nullptr;
if(nielEnergyLimit > 0.0) { pnuc = new G4NuclearStopping(); ... }
```

and `G4EmParameters::Initialise` sets `maxNIELEnergy = 0.0`. So `pnuc` is null, every
`if(nullptr != pnuc)` fails, and the elaborate split distributes nothing. The condition is not
near the code it controls, it is not in the class the process belongs to, and it is spelled as a
`G4double` limit rather than as a `G4bool` - three reasons why reading downward from
`ConstructProcess` never reaches it. `ref/oracle/species_processes.csv` - one row per process on
each species own process manager in the QBBC this project's own oracle constructs - has no
`nuclearStopping` row for any particle, and that took one dump and no reading at all.

**What it was worth.** Exactly measurable without a GPU run, because the quantity is an integral
over the port's own tables: walk a track from E0 to 1 eV on a fine energy grid, take the step
length from the range table, and accumulate what `step_hadron` removed. Grid-independent to six
figures at 100,000 and 200,000 steps. In `G4_WATER`:

| beam | energy through the NIEL channel | share of the primary |
|---|--:|--:|
| 210 MeV proton | 2.03 keV | 9.7e-6 |
| 840 MeV alpha | 26.0 keV | 3.1e-5 |
| 200 MeV deuteron | 5.22 keV | 2.6e-5 |
| 200 MeV mu- | 0.21 keV | 1.1e-6 |

Air is the worst of the four B1 materials and is 1.6x water. So the answer is that it changed no
dose anyone has quoted: the model returns zero above z1^2 MeV per nucleon, so it touches only the
last microns of a track, and the energy it removed was **deposited locally in the same volume the
dying track would have deposited it in anyway**. What it changed is the step the deposit happened
on and the non-ionising share of it. RESULT.md's 0.15% proton agreement is 1.5e-3 and this is
1e-5, two orders of magnitude below it.

That is the uncomfortable part, and it is why this is an entry rather than a commit message. The
error was harmless *for the two species that were transported and the one quantity that was
compared*, and there was no signal anywhere in this project that could have found it: the dose
agreed, the range table agreed (nuclear stopping is deliberately not in it - see
`hadron_total_dedx`), and no test asked which processes the physics list holds. It was found by
dumping the process manager for an unrelated reason, while adding species. A wrong process that
costs 1e-5 today is still a wrong process, and the next species or the next scored quantity is
where it stops being free - `G4NIELCalculator` exists precisely because somebody scores the
non-ionising share.

`uses_nuclear_stopping` in `core/particle.cuh` now returns false for everything, and is kept as a
predicate rather than deleted because `/process/em/setMaxNIEL <E>` turns the process on and the
source's four-species split is then the right answer. One place to change.

#### Two predicates that were lists of the species that existed when they were written

`is_heavy_charged` was `t >= kMuonMinus && t < kNumTypes`. True of every species in the enum on
the day it was written, and false the moment a neutral one was appended - a neutron and a
neutrino both answered yes. Nothing called it, which is the only reason this is a note. An
enum-range predicate is a claim about the ORDER of an enum, and `ParticleType`'s order is a
storage format: it is written into every device track and into the trajectory records the viewer
reads back, so it is append-only, so a range over it is a claim that cannot be maintained.

`em::hadron_tlimit` tested `type == kMuonMinus || type == kMuonPlus` where Geant4's
`G4BetheBlochModel::SetupParameters` tests `GetLeptonNumber() == 0`. The two agreed while the
muons were the only leptons anything called it for. They stopped agreeing when the electron, the
positron and six neutrinos became reachable - eight species being handed a hadron's nuclear form
factor. Latent, because a lepton goes through Moller-Bhabha and never reaches the Bethe-Bloch
chain; found because `ref/oracle/species_tables.csv` has a `tlimit` column and eight rows
disagreed. The fix is one word: the property, `pd.is_lepton`, which is Geant4's own condition.

Both are the same failure. A predicate that enumerates its members is a *snapshot* of a rule, and
it is correct exactly until the set changes - which for this file is every time a package lands.
Five per-species questions are now answered against `ref/oracle/species_processes.csv`
(`uses_ion_ionisation`, `uses_ion_fluctuations`, `uses_wentzel_msc`, `uses_nuclear_stopping`,
`hadron_base_particle`), and `hadron_base_particle` is transcribed as Geant4's *rule* - spin,
then the sign of the charge - rather than as its list, which is why it comes out right for
sigma+, sigma- and xi-, three species this port refuses to transport and therefore checks
further than it uses.

#### Recorded and not fixed: A^0.27 where the port computes Z^0.27

`hadron_tlimit` divides by `pow(Z, 0.27)` where `G4BetheBlochModel::SetupParameters` divides by
`G4NistManager::GetA27(iz)` - a table of A^0.27 for the natural-abundance atomic weight. For
Z = 2 that is 2^0.27 = 1.206 against 4.0026^0.27 = 1.454, so the alpha's and He3's `tlimit` comes
out **45% high** (measured 4.54e-01). They are the only species here with |charge| > 1 and
therefore the only ones that reach the line.

Not fixed, and the reason is worth stating rather than the fix being quietly skipped: `tlimit`
binds only where the kinematic `tmax` exceeds it, which for an alpha is above about 3 TeV, and
the correction needs 101 transcribed A(Z) values in `data/` - a file P1 does not own. So
`tests/test_species.cu` prints it as a `gap` with its measured value and asserts on the
twenty-two species that do agree, rather than widening one tolerance until all twenty-four pass.
That is the distinction V29 is about: a recorded measurement of an unfinished path is evidence; a
tolerance wide enough to cover it is not.

#### The discrete radiative processes, which nobody had noticed were missing either

Found while checking what else `species_processes.csv` says about the newly transported species,
and it turned out to be about the old ones too. **Every** charged hadron in QBBC carries
`hBrems` and `hPairProd`, and mu+- carry `muBrems` and `muPairProd`. `step_hadron` samples no
discrete radiative interaction for any species, and the range table is ionisation only. So the
proton has been transported without two of its registered processes for as long as it has been
transported, exactly as it was with nuclear stopping - except that this one is a process that
should be there rather than one that should not.

The continuous half is nothing: the restricted sub-cut radiative share of dE/dx for mu- in water
is 5e-5 of the total at 1.6 GeV, 3.6e-4 at 10 GeV and 2.7e-3 at 100 TeV. The **discrete** half is
the whole loss channel at high energy, and the mean free paths say where it starts to matter:

| mu- in water | brem mfp | pair mfp | CSDA range | P(one interaction) |
|---|--:|--:|--:|--:|
| 1 GeV | 649 m | 1.81 km | ~6 m | ~1% |
| 10 GeV | 462 m | 86 m | ~58 m | ~0.8 |

For pi+- the brem mean free path is 1.8e6 mm at 200 MeV and pair production does not begin until
`max(850 MeV, 8m)` = 1.12 GeV; for the proton, 7.5 GeV. So below a GeV this is 1e-4 of a track
and above ten GeV a muon is being transported without its dominant loss.

Not closed here, and named rather than left: what is missing is
`G4MuBremsstrahlungModel::SampleSecondaries` and `G4MuPairProductionModel::SampleSecondaries`
(the second needs its sampling tables), which is physics and not plumbing. `docs/PORTED.md` 1.3
carries it as `P` with these numbers. The judgement recorded here is that no energy refusal was
added: a cap at the model's own `lowestKinEnergy` of 100 MeV would refuse the 210 MeV proton this
project's headline result is measured on, for a process worth 1.6e-4 of that track, and a cap
chosen per species from a mean free path is a physics-list decision rather than P1's.

#### What the neutron's own process list settled

Worth keeping because two packages have to build against it. `ref/oracle/neutron_processes.csv`
says the neutron in QBBC 11.1.1 carries `Transportation`, `Decay` and `NeutronGeneralProc`
(subtype 116, `fNeutronGeneral` - not 161, which is what reading the enum too fast gives) and
**nothing else** - no `hadElastic`, no `neutronInelastic`, no `nCapture`, no `nKiller`. So
`/process/inactivate hadElastic` names nothing a neutron has, the three cross sections compete
through one summed interaction length, and the 10 us cut is two lines inside
`PostStepGetPhysicalInteractionLength` rather than a process of its own.

And that cut **deposits nothing**: `PostStepDoIt` calls `theTotalResult->Initialize(track)`,
whose `InitializeLocalEnergyDeposit()` zeroes both deposits, then proposes `fStopAndKill`. The
neutron's kinetic energy is discarded. Every other way a track dies in `stepper.cuh` hands its
energy to the volume it stood in, so the natural assumption is the opposite one, and acting on it
would put energy into a phantom that Geant4 puts nowhere. `G4NeutronKiller.cc` describes itself
as "The process to kill particles to save CPU"; it is a budget, not physics, and it does not
conserve energy. `RunStats::neutron_killed_energy` books it so a run can be held to an energy
balance instead of the shortfall being found later and blamed on the transport.

The grid is the other thing two packages need. `G4NeutronGeneralProcess::PreparePhysicsTable`
builds two `G4PhysicsLogVector`s - 400 bins from 1 keV to 20 MeV, 70 more to 100 TeV - and passes
`false` for the spline flag, so the interpolation is **linear** where the dE/dx and range tables
in this port are splined. Two tables in one transport with different interpolation rules is
visible only by going and reading which flag was passed, and V5 is what mismatching a grid costs.
The partials also swap order at 20 MeV: `PostStepDoIt` tests elastic first below it and inelastic
first above it, so the high zone's single stored partial is the inelastic one. Getting that
backwards exchanges two cross sections that differ by a factor of a few and still looks like a
plausible neutron.

#### The muon was transported with a different function from the one its test checks

The like-for-like run that P1 exists to make - example B1 with a mu- beam, against Geant4 QBBC
with everything a muon has and this port does not switched off - came out **3.2% high at 5
sigma**. And the same run with mu+ came out 1.4% LOW. The port's two answers were not merely
close: they were **bit-identical**, 23.8377 nGy for both, where Geant4 gives 23.11 nGy for mu-
and 24.1721 for mu+ - a 4.6% asymmetry at 8 sigma between two particles of the same mass and
opposite charge.

One number where the reference has two is the shape of a **missing charge-odd term**, and there
is exactly one in the chain: `high_order_bracket` is `2*(Barkas + Bloch) + Mott` and the Barkas
correction goes as z^3. `G4MuBetheBlochModel::ComputeDEDXPerVolume`'s last line is

    //High order corrections
    dedx += corr->HighOrderCorrections(p,material,kineticEnergy,cutEnergy);

and eighteen lines above it, `dedx -= 2.0*corr->ShellCorrection(p,material,kineticEnergy);`.
Both are in the port's `mu_bethe_bloch_dedx`, both guarded by `if (shell != nullptr)`, and
`shell` is a **defaulted parameter**. `hadron_ioni_dedx` dispatches five models and passes the
tables to four of them:

    case HadronIoniModel::kMuBetheBloch: return mu_bethe_bloch_dedx(m, type, kinetic, cut);
    case HadronIoniModel::kBetheBloch:   return bethe_bloch_dedx(m, type, kinetic, cut, shell);

So every dE/dx the muon was transported with, and the whole muon range table, was the
Bethe-Bloch bracket plus Kokoulin's radiative term and nothing else. One argument.

**Why no test caught it.** `tests/test_muon.cu` compares `mu_bethe_bloch_dedx` against Geant4's
model over 1392 points and reports `0.0000%`. It passes the tables, because it calls the model
directly - which is the right thing for a test of a model, and is why it was checking a function
the transport did not use. `tests/test_hadron_range.cu` builds its table through
`hadron_total_dedx`, so it DID see it, as a mu- dE/dx error of 10.2% and a mu+ of 6.4% - and
those numbers were sitting in its table as *recorded measurements of a species with no kernel*,
which is exactly the licence a recorded measurement gives you and exactly what it costs. mu-
was four times worse than pi- in the same column and nothing asked why.

Measured, after passing the argument: mu+ dE/dx worst 6.397% -> **0.861%** and range 3.213% ->
**0.447%**; mu- 10.176% -> **2.209%** and 6.015% -> **1.500%**. What remains for mu- is the same
~2% every negative hadron carries in `G4_AIR` (pi- is 2.367% in the same column), so the muon
has stopped being an outlier - which is the shape the table should have had from the start.

Three things worth keeping:

- **A defaulted parameter let one call site mean something different from its four neighbours.**
  Nothing in the signature says the transport may not use the default; nothing in the call site
  says it did. The default is still there, because a test that wants the bare model has a
  legitimate reason to pass nothing - but the diagnosis is that a dispatcher which forwards
  four of five arguments is not obviously wrong at a glance, and no compiler warning exists for
  it.
- **A model test and a transport test are different tests, and 0.0000% on the first says
  nothing about the second.** This project has both disciplines and they met at a function
  boundary neither of them crossed.
- **"Two particles, one answer" is a diagnostic.** The port's mu+ and mu- doses agreeing to the
  last bit was more informative than either of them being 3% out, and it named the term before
  any source was read: charge-even where the reference is charge-odd.

The B1 numbers after the fix are in the commit that made it.


### V44: every negative hadron, and a reference that disagrees with its own table

V43 left one thing open with numbers on it: this port's mu+ dose agreed with Geant4 and its mu-
did not, and the suspect named there was multiple scattering, "EM territory and P14's
neighbourhood rather than P1's". Extending the like-for-like to the other two charge pairs the
plan asked for - pi+- and K+- - turned that from one species' anomaly into a pattern, and four
diagnostics narrowed it. It is still not P1's to fix. It is now a located question instead of a
named one, and the diagnostics are what is worth keeping.

#### The pattern: the positives agree, the negatives do not, and the size orders by 1/beta

Example B1, 200 MeV, 500,000 events per run on each side, port against Geant4 QBBC with
everything QBBC gives each species and this port does not have inactivated. The macros are one
per species per side under `ref/b1hadron/`, each carrying its own numbers.

| species | port (nGy) | Geant4 EM-only (nGy) | diff | sigma |
|---|---|---|---|---|
| mu+   | 598.660 +/- 0.634 | 600.681 +/- 0.634 | -0.34% | 2.25 |
| mu-   | 597.326 +/- 0.633 | 575.568 +/- 0.611 | **+3.78%** | 24.7 |
| pi+   | 632.349 +/- 0.668 | 633.446 +/- 0.668 | -0.17% | 1.16 |
| pi-   | 630.922 +/- 0.666 | 600.831 +/- 0.636 | **+5.01%** | 32.7 |
| kaon+ | 1242.97 +/- 1.306 | 1240.74 +/- 1.302 | +0.18% | 1.21 |
| kaon- | 1239.62 +/- 1.303 | 1131.87 +/- 1.195 | **+9.52%** | 61.0 |

Three positives inside 0.35%; three negatives high by 3.8%, 5.0% and 9.5%. Read down the
charge pairs instead of across:

|  | mu | pi | kaon |
|---|---|---|---|
| beta at 200 MeV | 0.938 | 0.911 | 0.703 |
| Geant4's own (+ minus -) | 4.36% | 5.43% | 9.62% |
| this port's own (+ minus -) | 0.223% | 0.226% | 0.270% |
| what the dE/dx table implies | 0.02% (water) / 0.23% (bone) | same | kaon- 0.29% **above** kaon+ |

The port is charge-blind to a quarter of a percent, which is what Geant4's own tables say it
should be. Geant4's transport is not, by a factor of twenty to forty, and for the kaon the sign
of the table's split is opposite to the sign of the dose's.

#### None of these particles stops in B1, so it is not a Bragg peak

The premise that would explain a large dose split from a small range split - a peak sitting on
the edge of the scoring volume, where d(dose)/d(range) is enormous - is false here, and three
macro headers said otherwise. `ref/oracle/hadron_tables.csv`, G4_WATER at 199.5262 MeV: mu-
855.0 mm, pi+ 782.2 mm, kaon+ 411.1 mm, against B1's 300 mm envelope. All three cross.
`muon_emonly.mac` had said "about 46 cm ... so it stops well inside B1's envelope",
`pion_emonly.mac` "about 25 cm ... it stops inside B1's envelope", `kaon_emonly.mac` "about
14 cm ... it is the stopped kaon that decays". All three are corrected, and the commit that
wrote the first of them had the right figure in its own body. **A number in a comment is a
measurement and decays like one.**

#### What is excluded, each by a measurement rather than an argument

- **Multiple scattering.** `ref/b1hadron/pion_nomsc.mac`. With `msc` inactivated as well, so
  the pion carries Transportation and hIoni and nothing else, the pion split goes from 5.43% to
  5.35% and the muon's from 4.36% to 3.99%. That was V43's named suspect. A first pass at 2,000
  events had said the opposite - the split appearing to fall from 3.58% to 1.19% - and that
  reading was one sigma of a +/-1.7% measurement, which is why the count is written into the
  macro.
- **The inactivations themselves.** `/particle/select pi- ; /particle/process/dump` after the
  seven commands: Transportation, msc and hIoni Active; hBrems, hPairProd, CoulombScat, Decay,
  hadElastic, hBertiniCaptureAtRest and pi-Inelastic InActive. The comparison is not resting on
  a command name being right. The split also survives with **nothing** inactivated (pi+ 657.396
  against pi- 575.366).
- **At-rest capture and decay.** Both inactivated on the Geant4 side, and neither can fire
  anyway because nothing stops. P4's measurement holds and is why the negative macros carry the
  extra line: `G4HadronStoppingProcess::AtRestGetPhysicalInteractionLength` returns 0.0, so
  capture pre-empts `G4Decay` for a stopped pi-, K- or mu-.
- **The tables.** `dedx_total` in `hadron_tables.csv` is `G4EmCalculator::GetDEDX`, which reads
  the process's own built table - the one transport uses. pi+ 0.18734421 and pi- 0.18730131
  MeV/mm in water, 0.023% apart; 0.32733467 and 0.32657222 in G4_BONE_COMPACT_ICRU, the material
  the dose is scored in, 0.233% apart. Range 782.16484 against 779.33985 mm, 0.36% apart. A
  single-track dump with a fixed seed and msc off gave **identical step lengths** for the two
  charges across all eight ionisation steps, which is the step limitation and therefore the
  range table agreeing to better than it can show.
- **The model split at low energy.** `G4hIonisation::InitialiseEnergyLossProcess` sets
  `eth = 2 MeV * mass / m_proton` and hands the low-energy model everything below it: 0.2975 MeV
  for a pion, 1.05 MeV for a kaon. `G4ICRU73QOModel`'s constructor sets a 10 MeV high limit, so
  `emax1` is `eth` and not `emax` - the negative's QO model covers a third of an MeV, not the
  whole range. Above it both charges are `G4BetheBlochModel`, whose only charge-odd term is the
  z^3 Barkas piece inside `G4EmCorrections::HighOrderCorrections`, and that is the 0.23% the
  table shows. `G4BetheBlochModel::CorrectionsAlongStep` returns immediately for anything that
  is not an ion.

#### What is left, with its two numbers

- **It is in both halves of the loss.** `ref/b1hadron/pion_nodelta.mac`. With `/run/setCut 5 cm`
  no delta ray is produced at all and the scored dose is purely the primary's continuous loss:
  the split is about 2.0%, against 5.43% at the default 0.7 mm cut. So roughly two fifths sits in
  the mean continuous loss and three fifths arrives through the delta-ray channel - and
  production above the cut goes as z^2, so the delta rays cannot introduce an asymmetry of their
  own.
- **It switches off with velocity.** `ref/b1hadron/pion_1gev.mac`. 200 MeV: 5.43% at 32.7 sigma.
  500 MeV: 2.56% at 16.6 sigma. 1000 MeV: below 0.25% and under 1.5 sigma in both of two
  independent runs. Not a small residue - agreement.

So: a charge-odd, low-velocity, species-independent suppression of about 2% in Geant4's own
continuous energy loss for negative hadrons, roughly doubled by the delta-ray channel, absent
from the DEDX and range tables the same install dumps, and absent from this port. **The port
reproduces the reference's tables and does not reproduce the reference's transport, and the two
reference quantities do not agree with each other.** Which of the two is right is not something
this package can settle from outside the EM chain, and it is the largest open item P1 leaves.

**CLOSED by V46.** The last sentence above was right about where to look and wrong about which
table: the transport reads neither the DEDX table nor the range table's values but the range
table's *interpolation*, which Geant4 makes LINEAR for the second-registered particle of every
charge pair and cubic for the first. Every exclusion in this entry holds; each of them tested a
value, and the defect is in a rule.

**What this cost, and the rule it earns.** Half a day of it went into hypotheses that a 2,000-
event run had appeared to support and a 500,000-event run then killed - msc twice. B1's printed
rms is the standard error and scales as 1/sqrt(N) exactly; a 2,000-event B1 run is +/-1.7%, so
it can neither confirm nor exclude a 3% effect. **Choose the event count from the size of the
effect being tested, before running, and write it into the macro.** The same rule already sits
in `tools/compare_b1_beams.ps1` for timing - "the event count is part of the measurement, not a
knob for how long you want to wait" - and it is just as true of a dose.


### V45: build scripts that compile a worktree against main

Every driver build script in this repository hardcodes the absolute path of the primary
checkout:

```
build_dose.bat:4        call "D:/g4gpu/build_engine.bat"
build_gui.bat:12        call "D:/g4gpu/build_engine.bat"
build_proton.bat:12     call "D:/g4gpu/build_engine.bat"
build_view.bat:10       call "D:/g4gpu/build_engine.bat"
examples/B1/build.bat:25 call "D:/g4gpu/build_engine.bat"
examples/B1/build.bat:29 set OBJ=D:\g4gpu\out\B1
examples/B1/build.bat:31 set INC=-I "%~dp0include" -I D:\g4gpu\src -I D:\g4gpu\src\g4
```

`build_engine.bat` and `build_vis.bat` are themselves correct - they use `%~dp0` throughout and
export `G4GPU_ENGINE_OBJ` from it. The defect is in the callers: `call`ing them by absolute path
makes `%~dp0` the primary checkout, so from a git worktree

* the transport engine is compiled from **main's** `src/host/transport_run.cu` and its headers,
  into **main's** `out/transport_run.obj`;
* `examples/B1/build.bat` additionally compiles the worktree's B1 example against **main's**
  `src` include path and writes its objects into main's `out/B1`;
* and the resulting `exampleB1.exe` or `g4dose.exe` links that engine, runs, and prints a dose.

The dose is main's physics wearing the branch's name. This is exactly S4's failure - "the run
prints a plausible number for physics that was never compiled" - reached by a different route,
and the branch under test is the one it silently discards. It also races the lead's build for
the same object file.

Not fixed here: these are not P1's files and five branches editing them is five conflicts. The
fix is mechanical - `call "%~dp0build_engine.bat"`, `set OBJ=%~dp0..\..\out\B1`,
`-I "%~dp0..\..\src"` - and it belongs in one commit on main, not in a package branch. Until it
lands, **anything built from a worktree has to be built with the worktree's own paths and the
size of the binary checked**: the P1 worktree's engine object is 14.5 MB against main's 7.4 MB,
because P1 instantiates sixteen kernels and main five, and that difference is the only reason
the mistake would be visible at all.

`build_one_test.bat` has a smaller relative of the same problem: it passes no `-arch`, so nvcc
defaults below compute 6.0, and `tests/test_neutron.cu` fails to compile because
`atomicAdd(double*, double)` does not exist there. What it reports is an overload-resolution
error inside `src/core/track_buffer.cuh`, naming neither the architecture nor the flag.
`build_all.bat`'s `TESTS_GPU` line already passes `-arch=sm_86`.


### V46: the negative of every charge pair is the second particle registered, and loses its spline

V44 left a located question with numbers on it: Geant4's transported B1 dose is 4.36% (mu),
5.43% (pi) and 9.62% (K) higher for the positive of each charge pair, while
`G4EmCalculator::GetDEDX` - the process's own restricted dE/dx table - has each pair 0.023%
apart in water and 0.233% apart in the scored bone. Multiple scattering, the at-rest and
hadronic processes, decay, a Bragg peak and the dE/dx and range tables' VALUES were all
excluded there by measurement, and every one of those exclusions holds. What is guilty is the
INTERPOLATION RULE on one of those tables, and the charge enters only through which member of
the pair `G4EmBuilder` registers second.

#### The table V44 checked is not the table the transport reads

`G4VEnergyLossProcess::AlongStepDoIt` (utils/src/G4VEnergyLossProcess.cc:825-836) computes the
continuous loss twice:

```
  eloss = length*GetDEDXForScaledEnergy(preStepScaledEnergy);      // theDEDXTable
  if(eloss > preStepKinEnergy*linLossLimit) {                      // linLossLimit = 0.01
    G4double x = (fRange - length)/reduceFactor;
    eloss = preStepKinEnergy - ScaledKinEnergyForLoss(x)/massRatio;// theRangeTableForLoss and
  }                                                               // theInverseRangeTable
```

A 200 MeV pion in `G4_BONE_COMPACT_ICRU` takes **3.0 steps** across a 60 mm slab, losing about
5% of its energy on each, so the second branch fires on every step of every track and
`theDEDXTable` is never read. `ref/chargeodd/` measures it: 20,000 events a side, same seed,
same geometry, the seven inactivations of `pion_minus_emonly.mac`.

| slab deposit (MeV/event) | pi+ | pi- | split | K+ | K- | split |
|---|---|---|---|---|---|---|
| default (linLossLimit 0.01) | 22.8848 | 21.9030 | **+4.48%** | 45.7077 | 41.8842 | **+9.13%** |
| linLossLimit 0.49 | 22.6465 | 22.7072 | -0.27% | 43.4504 | 44.0598 | -1.38% |
| dRoverRange 0.002 | 22.8998 | 22.9380 | -0.17% | | | |

The middle row keeps the step lengths and disables only the branch; the last shortens the steps
so it would not have fired. Either kills the split. And it is the primary's own loss:
`ke_in - ke_out` carries all of it (22.8804 against 21.8953) while its path length does not
(60.1273 against 60.1278), and the 200 mm of water upstream splits the arrival energy the same
way (157.42 against 159.02 MeV) and stops splitting it when the branch is off (157.67 against
157.62).

#### Why the range table is charge-odd, in five source lines

`ref/oracle/chargeodd_vectors.csv` reads the spline flag off the vectors themselves. Every
material, 85 nodes and a 100 eV lower edge on all of them:

| particle | dedx_spline | range_spline | invrange_spline |
|---|---|---|---|
| mu+, pi+, K+, p | 1 | 1 | 1 |
| mu-, pi-, K-, pbar | **0** | **0** | 1 |

1. `G4EmBuilder::ConstructLightHadrons` (G4EmBuilder.cc:175-204) creates **one**
   `G4hBremsstrahlung` and **one** `G4hPairProduction` and registers each to BOTH members of
   the pair - constructed after part1's `G4hIonisation` and before part2's. G4EmBuilder.cc:
   248-269 does the same for mu± with `G4MuBremsstrahlung` / `G4MuPairProduction`.
2. `G4MuBremsstrahlung.cc:82` and `G4MuPairProduction.cc:88` call `SetSpline(false)` in their
   constructors, and rightly: a radiative restricted dE/dx is identically zero below threshold
   and a cubic spline through that rings. `G4hBremsstrahlung` and `G4hPairProduction` derive
   from them (G4hBremsstrahlung.cc:51, G4hPairProduction.cc:51) and inherit it.
3. `G4LossTableManager::BuildTables` (G4LossTableManager.cc:800-838) walks `loss_vector` in
   **construction order** and picks the shared radiative processes up through its own "possible
   case of process sharing between particle/anti-particle" pointer scan. part1 gets
   `t_list = [hIoni, hBrems, hPairProd]`; part2 gets `[hBrems, hPairProd, hIoni]`.
4. `G4LossTableBuilder::BuildDEDXTable` (G4LossTableBuilder.cc:161-166) builds the summed
   vector as `new G4PhysicsLogVector(*pv0)` with `pv0 = (*(list[0]))[i]` - a **copy** of
   `t_list[0]`'s vector, carrying its `useSpline`. part2's sum inherits hBrems's false.
5. `BuildRangeTable` (:224-226) copies that vector again for the range table, and its
   `if(splineFlag) v->FillSecondDerivatives()` is a no-op because
   `G4PhysicsVector::FillSecondDerivatives` opens with `if(!useSpline) return;`.
   `BuildInverseRangeTable` (:275) builds a **fresh** `G4PhysicsFreeVector(npoints,
   splineFlag)` instead, so the inverse stays splined for both charges.

alpha, He3, deuteron, triton and GenericIon are splined because they get no radiative process
at all (`G4EmBuilder::ConstructIonEmPhysics`, G4EmBuilder.cc:119-145, registers only msc and an
ionisation process), so `n_dedx` is 1, `BuildTables`' `if (1 < n_dedx)` is false and no summed
vector is ever made.

#### The consequence, in one column pair

`ref/oracle/chargeodd.csv` dumps `dedx_table` beside `1/(dR/dE)` by central difference on the
range table. The range table is the integral of the dE/dx table, so their ratio is 1 by
construction. `G4_BONE_COMPACT_ICRU`, 100 MeV to 1 GeV:

* **pi+** ratio 0.99969 .. 1.00027, smooth.
* **pi-** ratio 0.97788 .. 1.02953, **piecewise constant over each grid cell** - which is what
  linear interpolation of R(E) gives, on a 7-bins-per-decade grid whose cells span a factor
  10^(1/7) = 1.389 in energy.

Over the 20 mm step the transport actually takes, `E - R^-1(R(E) - 20mm)` against `20mm*dedx`:
pi+ within 0.3%, pi- 7.6% low, kaon- 21% low, anti_proton 28% low - always too little loss,
because R(E) is convex and a chord over-estimates it.

That also explains every shape V44 recorded and could not attribute. It is **gone by 1 GeV**
because R(E) is nearly straight on the minimum-ionising plateau (the ratio is 1.0002 there). It
**orders by 1/beta** because the curvature of R(E) is largest at low velocity. It is **absent
from the tables' values** because dE/dx is smooth and slowly varying, so its own chord error is
a tenth of a per cent - it is R(E), spanning five orders of magnitude across the table, whose
DERIVATIVE the chord destroys.

#### The physics judgement, and what the port does

Geant4 is inconsistent with itself here: its restricted dE/dx table and its range table are
meant to be an integral pair and for a negative hadron they are not, so the same install gives
two different stopping powers depending on which of its own tables is asked. The port
reproduces the transport, because the port's contract is 11.1.1 as it runs and the number a
user compares is a dose. `em::hadron_table_uses_spline` is the one place that decides, named
for the registration order rather than for the charge, with the chain above in its comment; the
second derivatives of the dE/dx and range rows are zeroed for those four species, which is
exact rather than approximate because a cubic spline with zero second derivatives everywhere
IS linear interpolation. `tests/test_chargeodd.cu` checks the composition `E - R^-1(R(E) - L)`
against Geant4's own two lookups: 0.02-0.36% for all eight species after, 10-83% for the four
negatives before.

#### The like-for-like that closes V44

Example B1 at 200 MeV, 500,000 events a side, docs/RESULT.md's method; the port built from the
`phys/em_extra` worktree and the Geant4 EM-only column unchanged from what `ref/b1hadron/
*_emonly.mac` recorded for V44. The macro headers carry the same table.

| dose, nGy | port BEFORE | port AFTER | Geant4 EM-only | after |
|---|---|---|---|---|
| mu- | 597.326 +/- 0.633 | **575.145 +/- 0.611** | 575.568 +/- 0.611 | -0.073%, **0.49 sigma** |
| mu+ | 598.660 +/- 0.634 | 598.660 +/- 0.634 | 600.681 +/- 0.634 | -0.34%, 2.25 sigma |
| pi- | 630.922 +/- 0.666 | **600.871 +/- 0.636** | 600.831 +/- 0.636 | +0.007%, **0.04 sigma** |
| pi+ | 632.349 +/- 0.668 | 632.349 +/- 0.668 | 633.446 +/- 0.668 | -0.17%, 1.16 sigma |
| kaon- | 1239.62 +/- 1.303 | **1132.65 +/- 1.199** | 1131.87 +/- 1.195 | +0.069%, **0.46 sigma** |
| kaon+ | 1242.97 +/- 1.306 | 1242.97 +/- 1.306 | 1240.74 +/- 1.302 | +0.18%, 1.21 sigma |

So 24.7, 32.7 and 61.0 sigma became 0.49, 0.04 and 0.46, and the port's own charge splits are
4.09%, 5.24% and 9.74% against Geant4's 4.36%, 5.43% and 9.62% - against 0.223%, 0.226% and
0.270% before.

**The three positives are bit-identical**, to every digit B1 prints, and that is the
load-bearing half of the table rather than a courtesy: `hadron_table_uses_spline` names four
species out of ten, and a fix that had rescaled everything, or named the wrong four, would have
moved them. It also means those three rows are not a re-measurement - they are the same
computation - so quoting them as "still agrees" would claim more than was run.

The rms tracked the mean, which is a second and independent sign: pi- 0.666 -> 0.636 nGy
against the reference's 0.636, mu- 0.633 -> 0.611 against 0.611, kaon- 1.303 -> 1.199 against
1.195. Nothing in the fix touches fluctuations, so a per-event spread that lands on the
reference's says the distribution moved and not only its first moment.

**What this cost in event count.** The three `*_port.mac` files asked for 20,000, 20,000 and
2,000 events while their headers already quoted 500,000-event numbers, which the macros as
written could not produce - B1's printed dose is CUMULATIVE, so 20,000 events give 23.8 nGy
where 500,000 give 597. The effect being measured after the fix is 0.2% rather than 5%, and
V44's own rule applies to its closure: choose the event count from the size of the effect
before running, and write it into the macro.

**What this is worth reporting upstream.** Nothing in the Geant4 code is a typo. Each of the
five steps is defensible on its own; the defect is that step 2's correct local decision about a
radiative table leaks into step 4's choice of template vector, and step 3's ordering decides
which particle it leaks to. A one-line fix would be for `BuildDEDXTable` to take the spline
flag from `splineFlag` rather than from `list[0]`'s vector, or for `BuildTables` to put the
ionisation process first in `t_list`.

**The rule it earns.** V44's four exclusions were all correct and none of them found this,
because every one of them tested a VALUE. A table is a value, a grid and an interpolation rule,
and the third is invisible to any comparison that evaluates both sides at the same point. The
diagnostic that found it evaluates the derivative instead, and it is two columns.


### V47: a comment asserting a term is zero, above the code that drops it

`em/wentzel_msc.cuh`'s `wv_sample_single` is `G4WentzelOKandVIxSection::SampleSingleScattering`
without one factor. Geant4's analytic rejection function is

```
  grej = (1. - z1*factB + factB1*targetZ*sqrt(z1*factB)*(2. - z1))*fm*fm/(1.0 + z1*factD);
```

and the port's has no `/(1.0 + z1*factD)`, under this comment:

> factD is sqrt(mom2)/value, set only for particles with a magnetic-moment correction; it is
> zero for the particles here, so the 1/(1 + z1*factD) factor in the analytic branch is 1.

It is not. `factD = sqrt(mom2)/targetMass` is set by `SetTargetMass`
(G4WentzelOKandVIxSection.hh), which `SetupTarget` calls for **every** target
(G4WentzelOKandVIxSection.cc:205), and `G4WentzelVIModel::SampleScattering` calls `SetupTarget`
at G4WentzelVIModel.cc:615 immediately before `SampleSingleScattering` at :618. For a 200 MeV
proton on oxygen, `sqrt(mom2)` is 644.5 MeV and `targetMass` is 14903.9 MeV, so factD is 0.0433
and `1/(1 + z1*factD)` runs from 1 at zero angle to 0.920 at 180 degrees.

Measured, with `tests/test_coulomb_scattering.cu`'s sampler block - 20,000 draws per cell, the
same (Z, A) and target mass on both sides. Dropping the factor from
`em::coulomb_sample_single`, which is what `wv_sample_single` does, fails eight of 240 cells:

| species | Z | E | P(scatter) G4 / ours | sigma | chi2/bin |
|---|---|---|---|---|---|
| pi+ | 1 | 200 MeV | 0.6085 / 0.6746 | 13.6 | 70.3 |
| pi- | 1 | 200 MeV | 0.6101 / 0.6683 | 11.9 | 51.8 |
| mu± | 1 | 200 MeV | 0.522 / 0.573 | 10.1 | 37 |
| kaon+ | 1 | 200 MeV | 0.5988 / 0.6475 | 9.9 | 33.1 |
| proton | 1 | 200 MeV | 0.6084 / 0.6489 | 8.3 | 20.4 |
| pbar | 1 | 200 MeV | 0.5988 / 0.6303 | 6.4 | 13.2 |

All at Z = 1 and the lowest energy sampled, which is where `sqrt(mom2)/targetMass` is largest -
the scaling the term has, so the failure pattern is itself evidence about which term is
missing.

`src/physics/em/coulomb_scattering.cuh`'s `coulomb_sample_single` has the factor and takes the
target mass as an argument, because `G4eCoulombScatteringModel::SampleSecondaries` **overrides**
SetupTarget's mean atomic mass with the sampled isotope's nuclear mass after the cross sections
and before the sampler, so the cross section and the sampler see different targets by design.
`em/wentzel_msc.cuh` is not fixed here: it belongs to no Phase-2 package and one line in it is
one merge conflict for whoever does own it. What the WentzelVI multiple-scattering angle
distribution is worth today is not measured - `tests/test_wentzel_msc.cu` passes, so whatever
it checks is not sensitive to this.

**The rule it earns is V32's and V35's, for the third time.** A comment that argues for a
weaker claim than the code makes is protecting a gap, and a comment that explains why a term
can be discarded is the one to re-read when the term turns out to matter. This one names the
right variable, gives the right formula for it, and then asserts the wrong value.


### V48: the cut G4CoulombScattering hands its model is the proton's, and it gates a closed door

Two facts, and the second is why the first is written down rather than fixed in a hurry.

#### It is the proton cut, in four source lines

`G4eCoulombScatteringModel` uses its `cutEnergy` argument in exactly one place:
`G4WentzelOKandVIxSection::SetupTarget(iz, cut)` passes it to
`ComputeMaxElectronScattering(cut)`, where it bounds the energy transfer to an atomic ELECTRON
and so sets `cosTetMaxElec`. It is a delta-ray production threshold by construction. What
arrives in it is the PROTON production cut:

1. `G4CoulombScattering`'s constructor calls `SetSecondaryParticle(G4Proton::Proton())`
   (G4CoulombScattering.cc:68), because the recoil it can emit is an ion.
2. `G4EmModelManager::Initialise` turns the secondary into a cuts INDEX - gamma 0, e- 1, e+ 2,
   `else { idx = 3; }` (G4EmModelManager.cc:463-468) - and takes
   `theCuts = theCoupleTable->GetEnergyCutsVector(idx)` (:471).
3. Every later use of a cut reads that vector: the lambda table through `FillLambdaVector`'s
   `G4double cut = (*theCuts)[i]` (:634), and the final state through
   `G4VEmProcess::PostStepDoIt`'s `SampleSecondaries(..., (*theCuts)[currentCoupleIndex])`
   (G4VEmProcess.cc:527).
4. `G4eCoulombScatteringModel::Initialise` stores the same vector as `pCuts`
   (G4eCoulombScatteringModel.cc:117), which is what the recoil threshold reads. So the two
   cuts in this model are one number arriving by two routes.

The electron production cut therefore never reaches this model in QBBC, not for any species,
the e- projectile included. In G4_WATER the two differ by a factor of four: 0.07 MeV against
0.2776 MeV. The proton cut needs no table to reproduce - `G4RToEConvForProton::Convert` is
`(rangeCut/mm) * 100 keV`, linear and with no material argument, which is why the oracle's
`pcut_MeV` column is 0.07 in all seven materials while `ecut_MeV` spans 0.00099 to 0.61.

#### And it changes nothing, because the channel it gates is shut

`ComputeElectronCrossSection` (G4WentzelOKandVIxSection.hh:222-230) is

```
  G4double cost1 = std::max(cosTMin, cosTetMaxElec);
  G4double cost2 = std::max(cosTMax, cosTetMaxElec);
  return (cost1 <= cost2) ? 0.0 : kinFactor*fMottFactor*(cost1 - cost2)/...
```

and this process integrates from `cosTMin = cosTetMaxNuc` out to `cosTMax = -1`, so the channel
is open only when `cosTetMaxElec < cosTetMaxNuc`. For a heavy projectile those two are
`1 - cut*m_e/mom2` and `1 - 0.5*q2Max*<A^-2/3>/mom2`, so the condition is

    cut * m_e  >  0.5 * q2Max * <A^-2/3>

with `q2Max` = 19469 MeV^2 and `<A^-2/3>` = 0.1686 in water: 0.0358 MeV^2 against 1641, a factor
of 46,000 - and `mom2` cancels, so no energy changes the answer. Measured rather than argued:
over the 20,182 active rows of `ref/oracle/coulomb_xs.csv`, at BOTH cuts, `xs_electron` is zero
in every one, the smallest `cosTetMaxElec - cosTetMaxNuc` is +3.2e-7, and `elec_ratio` is zero
in all 240 sampler cells. It would take a ~3.2 GeV production cut in water to open it.

#### What that costs, and what it buys

* The CALL-SITE CONTRACT is still the proton cut, and `em::coulomb_secondary_cut(range_cut_mm)`
  is the one line that produces it. `coulomb_xs_per_volume` and `coulomb_select_element` take
  the cut as an argument for this reason; they read `Material::cut_electron` in the first
  version of this package and nothing measured it, because nothing could - the two give the
  same cross section. A wiring error here is invisible in option0 and becomes visible the
  moment anyone sets `MscThetaLimit` to something other than pi, which moves `cosTMin` off
  `cosTetMaxNuc` and can open the channel.
* `em::wentzel_electron_xs` and `coulomb_sample_single`'s electron branch are transcribed and
  are NOT exercised through this process by any cell in the oracle. Their correctness rests on
  `G4WentzelVIModel`, the other caller, which has a different `cosThetaMin`.
  `tests/test_coulomb_scattering.cu` prints the channel's state and the margin on every run
  rather than leaving this to be rediscovered.

**The rule it earns.** A quantity can be plumbed wrongly and measured correctly at the same
time, if what it feeds is multiplied by zero. The test that would have caught the wrong cut is
not a comparison of the cross section - that one passes either way - but a comparison of
`cosTetMaxElec`, the intermediate. Compare the intermediate whose value the argument actually
reaches, not only the answer the argument is supposed to change.

### V49: a flag that switches off the stage it selects

`G4DeexPrecoParameters::fUseGNASH` chooses `G4GNASHTransitions` over `G4PreCompoundTransitions`
(`G4PreCompoundModel.cc:122`). Setting it does not change which transition rates
pre-equilibrium uses. It stops pre-equilibrium happening at all.

`G4GNASHTransitions::CalculateProbability` computes one number and returns it. It never assigns
`TransitionProb1`, `TransitionProb2` or `TransitionProb3` - the three protected members
`G4VPreCompoundTransitions`' constructor initialises to 0.0 - and no other code writes them.
`G4PreCompoundModel::DeExcite` reads all three back on the next three lines
(`G4PreCompoundModel.cc:263-265`) and its first equilibrium test is

    if(!go_ahead || P1 <= P2+P3 || Z < minZ || A < minA || U <= fLowLimitExc*A || ...) {
      PerformEquilibriumEmission(aFragment, Result);
      return Result;
    }

which for (0, 0, 0) is `0 <= 0`, true. So every fragment leaves the loop on its first iteration,
no ejectile is ever emitted pre-equilibrium, and every product comes from
`G4ExcitationHandler`. The comment three lines above the read is Geant4's own warning about
exactly this failure mode - "WARNING: CalculateProbability MUST be called prior to Get!! (0
values would be returned otherwise)" - and `G4GNASHTransitions` satisfies the letter of it while
returning zeros anyway.

Measured, not argued. `ref/oracle/preco_transitions.csv` calls the three getters after a GNASH
`CalculateProbability` on all 5,120 grid points and every one reads exactly 0, where
`G4PreCompoundTransitions` on the same grid gives P1 > 0 on every row. `tests/test_precompound.cu`
asserts the zeros on the ORACLE side, so a Geant4 release that fixes the class fails the test
loudly instead of quietly bringing a dead branch to life.

Reproduced rather than corrected, because it is what the installed Geant4 computes. The port
sets `PrecoRefusal::gnash` so a caller that selects GNASH is told, rather than receiving pure
equilibrium emission that looks like a working answer.

#### The same shape twice more in the same package

`useSICB` is `G4VPreCompoundFragment`'s own dead flag: initialised to `true`, plumbed down by
`G4PreCompoundEmission::UseSICB` and `G4PreCompoundFragmentVector::UseSICB`, described as
defaulting to false by five separate comments in the package - and read nowhere in 11.1.1. The
superimposed Coulomb barrier it used to gate is now the unconditional
`elim = theCoulombBarrier*0.5` line in `G4VPreCompoundFragment::Initialize`, so the lower limit
of every charged channel's emission integral is HALF the barrier whichever value the flag holds.
`DeltaR` in `G4GeneratorPrecompoundInterface` is the third: set to 0.0 in the constructor, with
the space cut in `MakeCoalescence` that would read it commented out.

None of the three is a port bug and none of them can be tested, because none of them changes an
answer. What they cost is the reader's time. What makes them findable is asking who READS a
member, not who writes it - the same question docs/RISK.md V42's initialisation-order defect
needed.

---

### V50: the hand-over every cascade uses has an arm that cannot be called

`G4GeneratorPrecompoundInterface::Propagate` is where every cascade and string model in QBBC
turns its residual into the fragment `G4PreCompoundModel::DeExcite` receives. It has two arms,
chosen by a loop the source labels "Check that we use QGS model"
(`G4GeneratorPrecompoundInterface.cc:248-256`): `QGSM` is set if ANY hit nucleon satisfies
`Get4Momentum().mag() < GetDefinition()->GetPDGMass()`. The QGS arm then builds the exciton
four-momentum from

    GetPrimaryProjectile()->Get4Momentum() + G4LorentzVector(0.,0.,0.,InitialTargetMass)

at line 301. `GetPrimaryProjectile()` is `G4VIntraNuclearTransportModel`'s accessor for
`thePrimaryProjectile`, which that class's constructor sets to `nullptr` and only
`SetPrimaryProjectile` ever writes. There is no null check anywhere on the path.

**The predicate is not a rare condition; it is the default.**
`G4Fancy3DNucleus::ChooseFermiMomenta` gives every nucleon
`energy = GetPDGMass() - BindingEnergy()/A` on top of a sampled Fermi three-momentum
(`G4Fancy3DNucleus.cc:511-517`), so `mag() = sqrt(E^2 - p^2)` is below the PDG mass for ALL of
them. Any nucleus straight out of `Init()` therefore selects the QGS arm on its first hit
nucleon. `G4FTFModel` does not crash because it re-sets each hit nucleon to the on-shell
`sqrt(mt^2 + pz^2)` form before handing over; a model that does not, and any oracle or test that
wounds a nucleus by hand, does.

Found by writing `ref/dump/dump_precompound.cc`: the first `Propagate` call killed `g4dump.exe`
with no output at all. The dump now sets a primary projectile for every case, and three of its
four cases put the hit nucleons back on shell so that the FTF arm is exercised as well.

#### Why the branch verdict is a dumped column and not an inference

Putting a nucleon back on shell through `e = sqrt(p^2 + m^2)` does not guarantee
`Get4Momentum().mag() >= m` afterwards - the round trip through two square roots can land a
half-ulp low, and one of the four dumped cases does exactly that. So the dump carries `on_shell`
(what it intended) and `qgsm` (what Propagate's own loop concluded) as separate columns, and the
port's predicate is compared against the second. An `on_shell`-implies-`!qgsm` test would have
been wrong on a quarter of the grid.

For P9, P10 and P11: `preco::propagate_residual` takes the primary's four-momentum as a
parameter and sets `GeneratorRefusal::missing_primary` when the QGS arm is reached without one,
so the port reports a refusal where Geant4 dereferences null.

---

### V51: a neutron projectile is given a charged particle exciton

`G4PreCompoundModel::ApplyYourself` builds the initial state for a nucleon on a nucleus:

    G4int Zp = 0;
    G4int Ap = 1;
    if(primary == proton) { Zp = 1; }
    ...
    G4Fragment anInitialState(A + Ap, Z + Zp, p);
    anInitialState.SetNumberOfExcitedParticle(2, 1);
    anInitialState.SetNumberOfHoles(1,0);

`Zp` distinguishes the two projectiles, so the compound nucleus has the right charge. The
SECOND argument of `SetNumberOfExcitedParticle` is the number of CHARGED particle excitons, and
it is `1` for both of them.

What reads that number is `GetRj(nParticles, nCharged)`, the combinatorial factor every
channel's emission probability is multiplied by. At the initial `(P, Pc) = (2, 1)` the six come
out as

    neutron    (P-Pc)/P                                        = 1/2
    proton     Pc/P                                            = 1/2
    deuteron   2 Pc (P-Pc) / (P(P-1))                          = 1
    triton     0, guarded by  (P - Pc) >= 2
    He3        0, guarded by  Pc >= 2
    alpha      0, guarded by  Pc >= 2 and (P - Pc) >= 2

and **not one of the six depends on which nucleon came in**, because `Pc` does not. So the first
emission of any nucleon-induced reaction can only be a neutron, a proton or a deuteron, at
relative weights 1/2, 1/2 and 1, whether the projectile was a proton or a neutron. That is
exact, not statistical: `Pc` is a constant in the expression.

#### What it is worth, measured

`ref/oracle/preco_apply.csv` runs `ApplyYourself` 20,000 times per case on four matched
(neutron, proton) pairs. The pre-equilibrium ejectile counts, and the proton-to-neutron ratio
of them:

| target, 100 MeV | projectile | n | p | d | p/n |
|---|---|---|---|---|---|
| Al27 | p | 8,649 | 10,573 | 7,308 | 1.222 |
| Al27 | n | 10,688 | 8,693 | 8,035 | 0.813 |
| Fe56 | p | 12,761 | 13,727 | 5,231 | 1.076 |
| Fe56 | n | 14,346 | 12,002 | 5,413 | 0.837 |
| Au197 | p | 21,527 | 11,920 | 1,942 | 0.554 |
| Au197 | n | 22,456 | 11,838 | 1,956 | 0.527 |

The two projectiles do NOT give the same yields - the p/n ratio is 1.50x apart on Al27, 1.29x on
Fe56 and 1.05x on Au197 - and that difference is *entirely* the compound nucleus's extra proton,
because the exciton bookkeeping contributes none. Its shrinking with mass number is the
signature: one unit of Z matters less to a heavier compound's binding energies and Coulomb
barrier, whereas an exciton-charge effect would not care about A at all.

So what the unconditional `1` costs is measured by removing it. With `SetNumberOfExcitedParticle(2, 0)`
for a neutron projectile - the "physical" count if the projectile itself is the only exciton
whose charge is known - `GetRj` for the proton channel becomes 0/2 and n + Fe56 emits 26,252
pre-equilibrium neutrons instead of 14,346, an 83% increase, and no pre-equilibrium protons at
all. `tests/test_precompound.cu` fails that at 87 sigma, so the number is not a formality.

If the two particle excitons are instead taken to be the projectile and one struck nucleon drawn
at random from the target, the expected number of charged ones is `Zp + Z/A`: 1.40 for a proton
on Au197 and 0.40 for a neutron. Geant4's fixed 1 is therefore low for a proton and high for a
neutron, and the port does not invent a better number - `apply_yourself_initial_fragment`
reproduces the unconditional 1 and says in its header that it is unconditional, because
"improve the physics" and "port Geant4" are different jobs and only one of them can be validated
against this oracle.

Why it matters here rather than in general: neutron-induced reactions are most of the secondary
production in a shielding calculation, and the pre-equilibrium protons are the part of that
spectrum with enough energy to leave the shield. The port reproduces Geant4, so a comparison
against Geant4 will not show this; a comparison against data might.

---

### V52: an assertion that never reaches the code it is named after

Three of the twenty perturbations in P6's first anti-vacuity campaign were not caught, and all
three failed for the same reason: the test computed the expected value AND the actual value
itself, and never called the function under test.

    // tests/test_precompound.cu, as first written
    const int neq = static_cast<int>(std::lrint(std::sqrt(ldfact * e * ld)));
    cmp_int(b_neq, neq, iv(f, 7), w + " n_eq");

`b_neq` is a bucket named `EquilibriumExcitonNumber` with 640 points and a worst relative error
of exactly zero, and it stays at exactly zero when the equilibrium exciton number inside
`precompound_model.cuh` is changed from `G4lrint` to truncation. So did the bucket named
`EntryGate`, with the entry gate's `&&` changed to `||`, and again with the loop's `U <= low*A`
changed to `U < low*A`. Three green buckets, 1,920 points, and no coverage of the three lines
they are named after.

The fix is structural and it is not "call the function from the test as well": it is that the
port must have a function to call. The three decisions were expressions inline in
`preco::deexcite`, so there was nothing a test could address. They are now
`preco_equilibrium_exciton_number`, `preco_entry_gate` and `preco_loop_gate`, `deexcite` calls
them, and the same three perturbations fail.

#### What is general

The shape to look for is a test line that mentions neither the module's namespace nor any of its
functions. Every genuine comparison in this package reads
`preco::something(...)` against a CSV column; these three read arithmetic against a CSV column,
and the arithmetic was a copy of the source line the CSV was dumped from. A copy agrees with
itself.

It is worth saying why the statistical half did not catch them either, because the natural
assumption is that a 300,000-event campaign covers what an exact grid misses. It cannot, here:

  * the entry gate's `(Z < minZ && A < minA)` and the loop's `(Z < minZ || A < minA)` disagree
    only about fragments with `Z < 3, A >= 5` or `Z >= 3, A < 5`. Such a fragment passes the
    entry gate, fails the loop's OR on the same iteration, and reaches the same handler having
    consumed exactly ONE extra uniform deviate. The product distribution is identical; only the
    random stream moves. No campaign, at any statistics, can see it.
  * `U <= fLowLimitExc*A` against `U < fLowLimitExc*A` differ on a set of measure zero, and the
    excitation energy reaching that comparison has been through a `sqrt` round trip in
    `G4Fragment`'s constructor, so no decimal input lands on it.

Both are now pinned by construction rather than by sampling: `preco_equilibrium.csv` dumps both
gates' verdicts for nuclides on each side of the (Z, A) disagreement (He6 and Li4 among them)
and at U set to `fPrecoLowEnergy*A` and `fPrecoHighEnergy*A` exactly, plus the adjacent
representable double either side. Both sides read the same 17-digit double out of the file, so
`<` and `<=` return different verdicts and the comparison sees the difference.

Four further perturbations across the two campaigns were not caught and are NOT holes, which is
the distinction this entry exists to draw. An uncaught perturbation is a question, not a verdict;
what settles it is whether the two forms can differ on any reachable input.

  * `G4PreCompoundFragmentVector::ChooseFragment`'s `x <= probabilities[i]` against `x <` is
    measure-zero on a continuous deviate. The port has Geant4's form and there is no input at
    which the two differ.
  * `G4PreCompoundTransitions::PerformTransition`'s final `if (Npart < Ncharged)` clamp reads the
    PRE-transition locals. Round one changed it to read the post-transition counts and nothing
    moved, which could have meant either "dead" or "untested". Round two made the branch write a
    sentinel value, so that taking it at all would be visible, and nothing moved again. That
    settles it: the branch is never taken. **The lesson is that "dead" needs its own
    perturbation, one that fires if the line runs at all, and not merely a different value.**
  * `deex::level_density`'s `has_levels` dispatch input inverted. `fLD` is 1 in the install, so
    the function returns `A * fLevelDensity` and never reads the flag - the branch is
    unreachable under 11.1.1's defaults, which is P3's finding, re-measured here. The sensitive
    input perturbation for the same 640-point bucket is `A + 1` instead of `A`, and that one
    fails.

The third of those is the one to copy: when an input perturbation passes, the next question is
whether the input is read at all, and the answer is usually two lines up in the function.

### V53: a neutron that cannot be validated by a dose, and two transcriptions of its table

> **CLOSED BY P8d, AND THE TITLE IS THE PART THAT WAS WRONG.** The neutron's stage-1 row is
> `54.4664 +/- 0.5408` for the port against `54.4573 +/- 0.5361` for Geant4 - **+0.02%, 0.01
> sigma** - where it had been `0.0000 +/- 0.0000` against `0.0000 +/- 0.0000`, and the
> configuration that produces it took one line of C++ rather than a UI command. What this entry
> got wrong is one function name - the flag is set in `G4HadronInelasticQBBC`'s CONSTRUCTOR, not
> in its `ConstructProcess`, and there is a `G4State_PreInit` window between them in which
> `SetEnableNeutronGeneralProcess(false)` takes. Everything below about the process manager, the
> two transcriptions of the grid and the 7.62e-14 is unchanged and still the reason the socket is
> a view of P2's tables. **docs/RISK.md V60** is the finding and `ref/b1neutron/` the reference.
> The paragraph beginning "What follows is that the port's neutron transport cannot be validated
> by a B1 dose comparison until P9-P11 land" is the sentence that is now false.

**The neutron has no stage-1 configuration, and no UI command can make one.**

Every other species in `ref/b1hadron/` can be run against a Geant4 whose process list matches
the port's, one process at a time, because `/process/inactivate <name>` names a process on that
species' process manager. The neutron's manager holds three entries and that is all
(`ref/oracle/neutron_processes.csv`, dumped from the constructed QBBC):

```
Transportation        type 1  subtype 91
Decay                 type 6  subtype 201
NeutronGeneralProc    type 4  subtype 116
```

Elastic, inelastic and capture are **sub-processes inside** `G4NeutronGeneralProcess`, reachable
only through its own summed cross-section table, and `EnableNeutronGeneralProcess` is set
unconditionally by `G4HadronInelasticQBBC::ConstructProcess` with no messenger anywhere in
11.1.1. So `/process/inactivate NeutronGeneralProc` takes elastic, inelastic AND capture
together, or nothing. There is no configuration in which elastic and capture are active and
inelastic is not.

What follows is that the port's neutron transport **cannot be validated by a B1 dose comparison
until P9-P11 land**. `ref/b1hadron/stage1_neutron.mac` inactivates the general process, both
sides deposit exactly zero in the scoring volume, and the stage-1 table's neutron row is
`0.0000 +/- 0.0000` against `0.0000 +/- 0.0000`. That zero is a prediction and worth having -
it is what a Geant4 neutron does with its one hadronic process switched off, and the port
reproduces the streaming and the 10 us time cut that produce it - but it is not evidence about a
cross section or a final state. Until P9-P11 the neutron is validated by its table (bit-exact,
`tests/test_particlexs.cu`), its sub-process selection (`tests/test_wiring.cu` section 5) and
its final states (`tests/test_elastic_models.cu`, `tests/test_capture.cu`), and by nothing that
looks like a dose.

**And the table had two transcriptions.** P1 wrote the socket `step_neutral` reads -
`src/physics/hadronic/neutron_general_xs.cuh` - including its own `G4PhysicsLogVector` lookup;
P2 wrote the builder - `src/physics/hadronic/xs/neutron_general_xs.cuh` - whose output is
compared with the oracle bit for bit. Both compute the grid's node energies, and they disagree:

```
P1   x[j] = e_min * pow(r, j),  r = exp((log(e_max) - log(e_min)) / n_bins)
P2   x[j] = e_min * exp(j / invdBin),  invdBin = (n_nodes-1) / log(e_max/e_min),
     x[0] and x[n-1] assigned exactly            <- G4PhysicsLogVector::Initialise
```

`log(a) - log(b)` is not `log(a/b)` and `pow(r, j)` is not `exp(j*ln r)`. On the real table -
the three G4PARTICLEXS4.0 neutron data sets summed onto both grids for water and lead, every
node and every bin midpoint, 1884 points - the socket's own formula differed from P2's by up to
**7.62e-14** relative in the low zone and **2.02e-16** in the high one. The factor of 400
between the zones is amplification, not noise: a log vector's interpolation divides by a bin
width, the low grid's 400 bins over 4.301 decades are 23 times narrower in log(e) than the high
grid's 70 over 6.699, and an ulp of node energy is that much larger a fraction of `x2 - x1`.
7.6e-14 out from inputs correct to an ulp is V37's mechanism a third time.

Fixed by P8: the socket is now a view of P2's `PhysVec` tables and evaluates them with P2's
`phys_vec_log_value`, there is one grid and one lookup, and `tests/test_wiring.cu` section 5
asserts the two agree bitwise over those 1884 points and over 5664 (energy, q) sub-process
choices. The size is far below anything a dose resolves. That is not why it is written up: the
transport was reading the transcription no oracle had seen while the validated one sat in the
next directory, which is how V5/V7 and V40 each started.

`NeutronGeneralXs::log_vector_value` is kept, because `tests/test_species.cu` checks the lookup
MECHANICS through it - that the bin comes from log(e) and the interpolation is then linear in e -
and that claim is still true and still worth a test. It is marked as not being what the transport
reads.

---

### V54: the NIST element table has 107 elements and 104 of them can exist

`G4NistElementBuilder` carries 107 elements - `maxNumElements` is 108 and `AddElement` is called
for Z = 1..107. `G4NistManager::FindOrBuildElement(105)` aborts the process.

The path is three classes deep and every step of it is reasonable on its own:

1. For every element above uranium the table's abundance array holds a fabricated 100 on one
   isotope. Dubnium's is `DbW[11] = {0,0,0,0,0,0,0,100,0,0,0}` over A = 255..265, so Db-262 comes
   out of `AddElement`'s normalisation with `relAbundance = 1.0`.
2. `BuildElement` keeps every isotope whose abundance is `> 0.0`, so it does NOT skip these
   elements for having no natural isotopes - it builds a one-isotope `G4Element` for each.
3. `G4Element::AddIsotope`, once the declared count is filled, calls
   `G4AtomicShells::GetNumberOfShells(iz)`. `G4AtomicShells`' tables are declared
   `fNumberOfShells[105]` and `fIndexOfShells[105]`, so Z = 105 is out of range and the class
   raises `mat060` as a **FatalException**.

Found by writing `ref/dump/dump_isotopes.cc` to loop over the whole table: it wrote every line up
to Z = 104 and then killed `g4dump.exe` with

    *** G4Exception : mat060
          issued by : G4AtomicShells::GetNumberOfShells()
    Atomic number out of range Z= 105

So 104 is the highest Z at which a `G4Element`, and therefore a `G4Material`, can exist in
11.1.1 - whatever the NIST database holds. `data/isotope_abundance.hh` carries all 311 abundances
because they are the data `G4NistElementBuilder` holds, and it carries `kNistBuildableMaxZ = 104`
beside them because that is the highest Z anything can ask about. `tests/test_isotopes.cu` states
the 104 and the 3 as arithmetic rather than as literals, so a Geant4 that raises either limit
fails with the reason visible instead of with a count that is off by three.

Two things worth keeping from it. The first is that `is_natural_isotope(105, 262)` is TRUE in
`data/natural_isotopes.hh`, and always was: that header's own comment explains the count as "the
primordial radioisotopes that a list of STABLE nuclides leaves out", which is true of K40 and
U238 and is not the whole story - 15 of the 311 are trans-uranic nuclides with a fabricated
abundance, and the de-excitation module's `GetIsotopeAbundance(Z, A) > 0.0` treats them as
natural because Geant4 does. Reproduced, not corrected. The second is the shape: the failure is
not in the class that has the limit, it is in the class three levels up that never asks whether
the element it is building can have electron shells.


---

### V55: the compiler died before the physics could be wrong

Wiring `hadElastic` into `step_hadron` killed ptxas.

    ptxas warning : Stack size for entry function 'run_step_neutral<...>' cannot be
                    statically determined
    Internal error
    nvcc error   : 'ptxas' died with status 0xC0000005 (ACCESS_VIOLATION)

Nothing about the message says what to do, and two things about it are misleading. The warning
immediately above names `run_step_neutral`, which is not the kernel that grew - it is a
pre-existing warning that every stepper emits because the boolean-solid distance routine
recurses. And "Internal error" arrives with no file, line or symbol, so the only thing
distinguishing it from a hung machine is the exit code.

**What it was.** `run_step_hadron` is instantiated fourteen times in `transport_run.cu`, once per
charged species, and each instantiation inlines the whole of whatever the elastic branch calls.
That branch is G4ChipsElasticModel's 52-parameter tables, G4ElasticHadrNucleusHE's sampler,
Gheisha's `SampleInvariantT`, G4BGGNucleonElasticXS, G4BGGPionElasticXS,
G4UPiNuclearCrossSection, Barashenkov, G4HadronNucleonXsc and both Glauber-Gribov components.
One kernel's worth of that compiles; fourteen do not.

**How it was isolated, and this is the part worth keeping.** The failing compile is
`transport_run.cu`, which takes eight minutes - so bisecting `__noinline__` placements against it
would have been an afternoon. A ten-line translation unit that explicitly instantiates ONE
kernel,

    template __global__ void
    run_step_hadron<double, ParticleType::kProton, StepTap<double>>(...);

compiles in under two minutes and reports the same per-kernel register and stack numbers. It
also answered the first question immediately: the single kernel compiled fine, which said the
failure was the TRANSLATION UNIT's size and not any one kernel's, and therefore that the fix was
to stop inlining rather than to simplify the physics.

**The fix is also the right answer for the hot path**, which is why it is not a workaround.
`had::elastic_apply` and `had::elastic_xs_per_volume` are `__noinline__`. The elastic branch
fires 16 times in 900 steps (`tests/test_step_hadron.cu`), because the elastic mean free path in
water is 466 mm for a 200 MeV pion and 2013 mm for a 200 MeV proton against steps of tens of mm -
so what was being inlined into every step of every charged hadron is a branch that almost never
runs. Measured with the one-kernel reproducer, on the proton kernel:

    baseline (no elastic)                    2464 B stack,  652/1728 B spill, 255 registers
    elastic inlined                          2912 B stack, 1052/1292 B spill, 255 registers
    elastic_apply __noinline__               3376 B stack,  916/1592 B spill, 255 registers
    both __noinline__                        3440 B stack,  916/1512 B spill, 255 registers

Those numbers are the REPRODUCER's, not the engine's: ptxas allocates differently when it sees
one entry point instead of fourteen, and the same baseline measured through the full
`transport_run.cu` is 3696 B and 68/36 B of spill. Comparing a reproducer number against a
full-build number is how a measurement like this goes wrong, so both series are stated.

What the engine actually ends at, `transport_run.cu` with `-Xptxas -v`, against the same file
on main:

    kernel             registers   stack frame        spill st/ld      cmem[0]
    run_step_hadron    255 -> 255  3696 -> 4512 B     68/36 -> 96/52   1472 -> 1584
    run_step_neutral   255 -> 255  3072 -> 3728 B     80/36 -> 92/52   1496 -> 1608
    run_step_lepton    255 -> 255  2992 -> 3056 B     56/20 -> 76/28   1448 -> 1448
    run_step_gamma     255 -> 255  3024 -> 2400 B     44/20 -> 404/868 1448 -> 1448

No change in register count, +816 bytes on the hadron kernel's frame against the 16384-byte
limit `Upload` sets, and the last 48 of those are the `data::LevelTable` added to
`HadronicWiring` - four pointers and three counts, passed by value, which is also the cmem[0]
growth. Two things worth noting in that table rather than skipping past. The GAMMA kernel's
frame went DOWN 624 bytes and its spill UP by 360/848 while it gained no physics at all, which
is ptxas reallocating and not a cost anyone chose - the only thing that reached it is the
emitter's new `parent_velocity`. And two stack allocations were removed on the way, for the same
reason this entry exists: `store_sample_za_fn`'s `real_t temp[64]` (512 B, replaced by a second
pass over the same terms in the same order, so the numbers did not move) and an
`xs::ElementIsotopes[16]` that a `NistIsotopeView` answers instead (another 512 B).

**P8c added two more instantiations and the frame is not what to watch.** He3 and GenericIon
take `run_step_hadron` from eleven entry points to thirteen, and the cost of that is small and
measured - `transport_run.cu` with `-Xptxas -v`, before and after on `phys/wiring3`:

    kernel             inst.     registers   stack frame      spill st/ld     cmem[0]
    run_step_hadron    11 -> 13  255 -> 255  4512 -> 4576 B   96/52 -> 100/52 1584 -> 1600
    run_step_neutral   2         255 -> 255  3728 -> 3744 B   92/52 -> 96/52  1608 -> 1624
    run_step_lepton    2         255 -> 255  3056 -> 3040 B   76/28 -> 80/28  1448 -> 1464
    run_step_gamma     1         255 -> 255  2400 -> 2416 B   404/868->368/676 1448 -> 1464

+64 bytes against the 16384-byte limit, no register change, ptxas survived. The +16 bytes of
cmem[0] on every kernel is the `ion_za` pointer on two `TrackBuffer`s passed by value, which is
also the check on the claim that the field itself is free: `sizeof(TrackState<double>)` is 248
before and after and only the argument list grew.

**COMPILE TIME IS THE THING THIS ENTRY SHOULD HAVE PREDICTED AND DID NOT.** The frame grew 1.4%
and the translation unit took about ninety minutes against the eight minutes recorded above -
with three Geant4 cmake builds running beside it, so it is not a clean measurement, but it is
the wrong direction by an order of magnitude and the numbers above give no warning of it. ptxas
was alive throughout at 4 GB of working set. What that means for the next package: the
reproducer this entry recommends - a ten-line translation unit instantiating ONE kernel - is now
the only affordable way to iterate on `transport_run.cu`, and P8c's own iteration used the host
test (`tests/test_ion_transport.cu`, one minute) for every loop and paid the ninety minutes
once, at the end, for the measurement. The neutron general process is the next thing to be
wired into a stepper and it drags the whole de-excitation chain into this file through
`capture/capture_process.cuh`; whoever does it should put `__noinline__` on the capture cascade
before measuring anything, for the reason this entry gives about the elastic branch.


---

### V56: both kaons, 0.6% high, and the two things that measurement does not separate

The stage-1 like-for-like with `hadElastic` and `CoulombScat` active on both sides
(`ref/b1hadron/stage1_README.md`) puts seven of nine species inside two sigma and the two kaons
outside it:

    kaon_plus    1,215.0200 +/- 1.4305   G4  1,208.1700 +/- 1.3920    +0.57%   3.4 sigma
    kaon_minus   1,115.7700 +/- 1.3779   G4  1,108.7600 +/- 1.3248    +0.63%   3.7 sigma

**BOTH CHARGES BY THE SAME AMOUNT, which is what rules the obvious cause out.** V44/V46 - the
range-table interpolation rule that split every charge pair - is one-sided by construction, and
in this table it is closed: mu- is at 0.5 sigma, pi- at 1.8. A defect that moves K+ and K- by
+0.57% and +0.63% is charge-blind.

**What the third column establishes, and it is not small.** Running the Geant4 side again with
`hadElastic` inactivated gives 1,201.91 for K+ and 1,097.39 for K-, so Geant4's own elastic
scattering RAISES a 400 MeV kaon's B1 dose by 0.52% and 1.04% - the opposite sign to the pion's
-3.92%, because elastic scattering moves a pion's dose out of the 12 cm scoring volume and a
kaon's into it. The port is 3.4 and 3.7 sigma from the column with elastic and 6.7 and 9.9 from
the column without it. So the port's kaon elastic acts in the right direction and with roughly
the right size; what is left is about a fifth of the effect.

**The two candidates, neither excluded.**

1. *The channel.* The kaons are the only species whose `had::elastic_channel` is
   `G4CrossSectionElastic(G4ComponentGGHadronNucleusXsc)` plus a plain Gheisha `G4HadronElastic`
   - `G4HadronicBuilder::BuildElastic`'s pair, reached through the "kaons" line of
   `G4HadronElasticPhysics::ConstructProcess`. The proton's CHIPS and the pion's
   `G4ElasticHadrNucleusHE` are different code, and both of those species are inside two sigma.
   Against it: `tests/test_hadronic_xs.cu` compares the Glauber-Gribov component exactly and
   `tests/test_elastic_models.cu` compares Gheisha's `SampleInvariantT` against the oracle, so
   if this is the cause it is in the composition rather than in either piece.
2. *The dE/dx table in air.* README open question 1 records `test_hadron_range` finding K- and
   pbar off Geant4's `GetDEDX` by 11% and 15% in AIR between 1 and 3 MeV and nowhere else,
   undiagnosed since it was measured. B1's world is air and its envelope is 12 cm of water in
   30 cm of it. Against it: that finding is for the NEGATIVE of each pair only, and this is both
   charges.

Neither candidate explains a charge-blind 0.6% on its own, which is why this is written up
rather than attributed. What would separate them is the measurement P14's `ref/chargeodd/`
already has the shape for: a Geant4-linked program that asks `GetDEDX` for K+ and K- in B1's
four materials over the whole range, and a stage-1 run with the kaon's elastic channel forced to
the proton's CHIPS model, which is wrong physics and a clean bisection.

The size is worth keeping in proportion. 0.6% is an eighth of what `hadElastic` was worth for
the pion and a fortieth of what the missing inelastic final states are worth for a proton
(docs/RESULT.md: 19%). It is recorded because it is the largest unexplained residual in the
stage-1 table and because three of the nine species were at 24 to 58 sigma one commit ago.


---

### V57: the two validated lookups sat next to the two the transport read

`em::HadronRangeTable` has two sets of entry points and `step_hadron` was calling the wrong one.

    dedx_at(species, mat, E)           the table in the TABLE's own terms
    lookup(species, mat, E)
    energy_from_range(species, mat, R)

    dedx_for(mat, type, imat, E)       G4VEnergyLossProcess's base-particle scaling on top:
    range_for(mat, type, imat, E)          scaledE = E * massRatio
    energy_from_range_for(...)             dE/dx   = chargeSqRatio * table(scaledE)
                                           range   = table(scaledE)/(chargeSqRatio*massRatio)

The second set is transcribed from `G4VEnergyLossProcess::GetDEDX`, `::GetRange` and
`::GetKineticEnergy` - `GetDEDXForScaledEnergy` multiplies the table by
`fFactor = chargeSqRatio`, `GetScaledRangeForScaledEnergy` multiplies by
`reduceFactor = 1/(chargeSqRatio*massRatio)`, and both look up at `kinEnergy*massRatio` - and
`tests/test_hadron_range.cu` compares it against `G4EmCalculator::GetDEDX` and `::GetRange` for
every species this port names. So the scaling was measured against Geant4 and the transport did
not use it.

**For the ten species that own a table both ratios are exactly 1.0**, which is why this survived
so long: `G4VEnergyLossProcess::PreparePhysicsTable` sets them from the base particle and leaves
them at their 1.0 initialisers when there is none, so for p, pbar, pi+-, K+-, mu+-, alpha and
GenericIon the two sets are the same double and every number this port has ever produced for
those ten is unchanged. The proton's and the alpha's B1 doses and the proton depth-dose curve do
not move.

**The deuteron and the triton have kernels and do not own tables.** `hadron_base_particle`
returns the proton for both - correctly, and its own header explains the arithmetic at length:

    massRatio = m_p/m_d = 0.500248192      massRatio = m_p/m_t = 0.334032895

so the transport was reading a PROTON's range and restricted dE/dx **at the deuteron's own
kinetic energy**. Measured in water by stepping 200 tracks to a stop and summing the true path
length (`tests/test_ion_transport.cu` section 2):

    species    E       transport    range_for    error       what the two were
    deuteron   50 MeV  22.388 mm    12.809 mm    +74.79%     both 22.385 mm - the proton's
    triton     50 MeV  22.388 mm     9.265 mm   +141.64%     row, at 50 MeV, for both

Two different particles stopping at the same depth to five figures is the signature: 22.387947
and 22.388005 mm, differing only where the fluctuation sampler sees the particle's own mass.

He3 was not transported when this was found and would have been worse: its base particle is
GenericIon, `massRatio` is 0.334096 and `chargeSqRatio` is the dynamic effective charge squared
(4.0007 at 50 MeV in water), so `range_for/lookup` is **0.1032** - a factor of 9.7.

`core/particle.cuh`'s `hadron_base_particle` predicted this exactly - "a deuteron's range would
have come out as a proton's of the same kinetic energy, which is a factor of about two" - and
the fix it describes was made in the TABLE and not at the call site. That is the shape worth
keeping: the package that found the defect fixed the function it owned, the comment recorded the
consequence, and the one caller that mattered was in another package's file. V53 is the same
mechanism (the transport read P1's transcription of the neutron grid while P2's bit-exact one sat
in the next directory) and so is V5/V7.

**Anti-vacuity.** Section 1 of `tests/test_ion_transport.cu` asserts the ratios themselves -
exactly 1.0 for the ten, and more than 20% away from 1.0 for the three that scale - because
without it eight of section 2's rows would pass with the unscaled lookup in place. Run with the
fix removed, section 2 failed as quoted above; run with only the RANGE unscaled and the dE/dx and
the inverse lookup scaled, all 200 deuterons and all 200 tritons left the 400 mm water cube with
their full energy (`balance -1.00e+00`), because a long range against a scaled stopping power
makes the step function propose steps that lose almost nothing.

---

### V58: the reference was taken without the process it was being compared with

`build_all.bat`'s proton depth-dose gate failed the moment P8b wired `hadElastic`, and one of
its two failures was not in the port at all:

    plateau (0-59.8 mm)  port/G4 per proton = 1.01322  (1.322%)
    FAIL: plateau dose off by 1.322%, limit 1.000%

`ref/oracle/proton_depth.csv` was generated by `ref/proton/proton_depth.cc` running
`G4EmStandardPhysics` **and nothing else**, and the file said why at length: "A proton in QBBC
undergoes inelastic nuclear reactions... This port has no hadronic physics at all, so comparing
against QBBC would measure that absence rather than the stepper." That was true when it was
written and had been false since P8: the port has `G4Decay`, `hadElastic` and `CoulombScat` on
the proton now, and 1.32% is what elastic scattering is worth to a 100 MeV proton's plateau in
water. A comment that justified a gap outlived the gap, which is V32's shape.

Fixed the way docs/HADRONIC_PLAN.md section 4 says: QBBC on both sides, with only what the port
lacks inactivated on the Geant4 side, and `/particle/process/dump` printed by the run so the
configuration is recorded by what ran. Two things came out of writing that list.

**The UI refuses two of the names, and both are V53's mechanism.** `/process/inactivate
neutronInelastic` and `/process/inactivate photonNuclear` both answer `illegal process (or
type) name`. Each is a sub-process inside a general process and is on no process manager:

  * `G4HadProcesses::BuildNeutronInelasticAndCapture` takes the `useNeutronGeneral` branch and
    calls `nGen->SetInelasticProcess(nInel)` instead of `ph->RegisterProcess(nInel, neutron)`.
    V53 is the entry about that; what is new is that the same thing is true of the gamma.
  * `G4EmStandardPhysics::ConstructProcess` calls `param->SetGeneralProcessActive(true)`, so
    `G4EmExtraPhysics::ConstructGammaElectroNuclear`'s `gproc != nullptr` branch runs
    `gproc->AddHadProcess(gnuc)` rather than registering `photonNuclear` on the gamma.
    `electronNuclear` and `positronNuclear` ARE accepted, because no electron or positron
    general process is built in option0 - so three sibling processes written in one function
    split into two reachable by name and one not.

The neutron's comes off with `NeutronGeneralProc`, which is in the list. The gamma's cannot come
off at all without `GammaGeneralProc`, which would take Compton, the photoelectric effect,
Rayleigh and conversion with it - processes this port HAS - so it is left ACTIVE and the
arithmetic that makes it inert is recorded instead: nothing in this configuration makes a photon
above about 0.5 MeV (`protonInelastic` and `hBrems` are off, so the only photons are the
bremsstrahlung of a delta ray) and `G4GammaNuclearXS` is a giant-resonance cross section
starting near 10 MeV. **A gap that cannot be switched off has to be bounded instead**, and that
is the general lesson: the plan's "inactivate exactly what is missing" is not always available.

**And the phantom was too narrow for its own conservation check.** `tools/compare_depth.ps1`'s
first metric is energy in against energy deposited, limit 1e-6, justified as "it should be exact
on both sides - the phantom is deeper than the range". Deeper is not wider. An elastic scatter
off oxygen leaves a 100 MeV proton nearly all of its energy at any angle - the target is heavy,
so the lab-frame energy loss is small at every angle - and one scattered near 90 degrees runs
its whole 77 mm range sideways, out of a 50 mm half-width box and into the vacuum world.
Measured on the new reference: **112 MeV of 10,000,000, i.e. 1.1e-5, eleven times the limit, on
the GEANT4 side.** So the new reference would have failed the gate's conservation check by
itself.

The phantom is 150 mm wide now rather than the limit being 1e-4. That is the same rule the
package was given for the plateau - "if it still differs, that is a finding to run down, not a
limit to widen" - applied to a check that had stopped meaning what it said. 150 mm exceeds a
100 MeV proton's range, so a proton scattered at any angle stops inside; the deposited total is
100.0000% of the beam energy on both sides again. What it changes about the curve is the same
1.1e-5, in the direction of a standard integral depth dose: the energy that used to leave is
binned at the depth it was scattered from.

---

### V59: V45's last three files, and they were the ones that write the oracle

docs/RISK.md V45 is "build scripts that compile a worktree against main": five scripts called
`D:/g4gpu/build_engine.bat` and friends by absolute path, so a package developed in a git
worktree built MAIN's engine from MAIN's headers and linked it to its own driver. The commit
that fixed it made `build_dose.bat`, `build_gui.bat`, `build_proton.bat`, `build_view.bat` and
`examples/B1/build.bat` relative to `%~dp0`.

Three files were not in it, and they are the three that produce the REFERENCE rather than the
port:

    ref/proton/build.bat     cmake -S D:/g4gpu/ref/proton -B D:/g4gpu/ref/protonbuild
    ref/proton/run.bat       D:\g4gpu\ref\protonbuild\Release\g4proton.exe
    ref/oracle/run.bat       call D:\g4gpu\ref\proton\build.bat
                             call D:\g4gpu\ref\proton\run.bat 100000 100
                                  D:\g4gpu\ref\oracle\proton_depth.csv 0.7 0.5

So `ref/oracle/run.bat` run from a worktree configured cmake on **main's** `ref/proton`, built
**main's** `proton_depth.cc`, and wrote **main's** `ref/oracle/proton_depth.csv`. P8c's whole
second deliverable is a change to that source file's physics list; regenerating the reference
from this worktree would have regenerated it from the version that had not been changed, written
the result into the other checkout, and left this one comparing against a file it did not
produce. The disagreement would have looked exactly like a port defect, which is V45's own
sentence.

Worth noting about the shape rather than the paths. V45's five files all built the thing under
test, so the symptom was an object file of the wrong size and the agents noticed. These three
build the thing it is MEASURED AGAINST, and there is no size to notice: the failure mode is a
number that is right for a different experiment. `ref/dump/build.bat` beside them is already
`%~dp0`-relative and says why in its header - "several packages are developed in worktrees at
once and each has its own `ref/dump/dump_<package>.cc` to build" - so the reasoning existed one
directory away from the files that needed it.

---

### V60: the flag is set in a constructor, and a missing level scheme does not emit nothing

Two findings from wiring `G4NeutronGeneralProcess` into the neutral stepper (P8d), and they are
opposite shapes: the first is a configuration everyone believed did not exist, and the second is
a failure mode that looks like physics instead of like an absence.

#### `EnableNeutronGeneralProcess` is settable, and V53's own mechanism was one function out

V53 is titled "a neutron that cannot be validated by a dose" and its central sentence is that
`EnableNeutronGeneralProcess` "is set unconditionally by `G4HadronInelasticQBBC::
ConstructProcess` with no messenger anywhere in 11.1.1". docs/PORTED.md 2.1.2 and
`ref/b1hadron/stage1_README.md` repeat it. **The second half is true and the first half names the
wrong function.**

```
G4HadronInelasticQBBC::G4HadronInelasticQBBC(G4int ver)          // the CONSTRUCTOR
  : G4VHadronPhysics("hInelasticQBBC")
{
  SetPhysicsType(bHadronInelastic);
  auto param = G4HadronicParameters::Instance();
  param->SetEnableBCParticles(true);
  param->SetEnableNeutronGeneralProcess(true);                   // <- here, not ConstructProcess
  param->SetVerboseLevel(ver);
}
```

`ConstructProcess` only READS it (`G4HadProcesses::BuildNeutronElastic` and
`BuildNeutronInelasticAndCapture` each open with `G4bool useNeutronGeneral =
param->EnableNeutronGeneralProcess()`), and the two run at different times: the constructor when
`new QBBC` registers its physics constructors, `ConstructProcess` at `/run/initialize`. In
between, the state is `G4State_PreInit`, and the setter's guard is

    G4bool G4HadronicParameters::IsLocked() const {
      return ( ! G4Threading::IsMasterThread() ||
               G4StateManager::GetStateManager()->GetCurrentState() != G4State_PreInit );
    }

so one line of C++ between those two points turns the general process off. The "no messenger"
half is confirmed rather than assumed: `G4HadronicParametersMessenger` builds exactly three
commands - `/process/had/verbose`, `/process/had/maxEnergy`, `/process/had/enableCRCoalescence` -
so there is no `/process/had/enableNeutronGeneralProcess` and no macro can do it.

**What that is worth is the neutron's whole stage-1 row, and the row came out at 0.01 sigma:**
`54.4664 +/- 0.5408 nGy` for the port against `54.4573 +/- 0.5361` for Geant4, 500,000 neutrons
of 100 MeV each side, with the diagnostic column - the same Geant4 run with `hadElastic`
inactivated too - at exactly `0 picoGy`. So the whole of a neutron's B1 dose is elastic
scattering, and the row is a test of `G4NeutronElasticXS`, `G4ChipsElasticModel` and P8c's recoil
transport with no dE/dx, no multiple scattering and no capture in it.

With the flag off, the process manager holds six entries where it held three, and the run prints
them (`ref/b1neutron/run.bat` on `ref/b1hadron/stage1_neutron.mac`):

```
[0] Transportation      Active
[1] Decay               Active
[2] hadElastic          Active     G4NeutronElasticXS:   0 eV ---> 100 TeV
[3] neutronInelastic    InActive   G4NeutronInelasticXS: 0 eV ---> 100 TeV
[4] nCapture            Active     G4NeutronCaptureXS:   0 eV ---> 100 TeV
[5] nKiller             General    TimeCut(ns)= 10000  KinEnergyCut(MeV)= 0
```

`/process/inactivate neutronInelastic` reaches entry 3 - the command V53 and docs/PORTED.md 2.1.7
both record as answering `illegal process (or type) name`, which it does whenever the flag is on.
And `G4NeutronTrackingCut::ConstructProcess` no longer returns early, so the 10 us cut arrives as
a real `G4NeutronKiller` carrying the same two numbers the general process carries internally.

So "the neutron cannot be validated by a B1 dose comparison until P9-P11 land" is now false, and
the sentence it rested on was a reading of the source rather than a question put to the object -
which is docs/RISK.md V43's lesson, arriving a fourth time in the file V43 is in. The stage-1
neutron row is a number now.

**And it is a different competition, not the general table minus a term.** P8c's V53 addendum said
"left out of the total" was the wrong description and did not say what the right one is. It is
this: in that configuration the neutron has two independent discrete processes, each evaluating
its OWN `G4CrossSectionDataStore` at the track's energy and drawing its own interaction length,
where the general process has one interaction length off a 401-node interpolation of the summed
cross section at ITS node energies. `step_neutral` runs both, switched by `had::HadronicStage`,
and `tests/test_neutron_general.cu` predicts the two by two different formulas.

#### A capture with no level scheme emits three times as much, not nothing

`TransportEngine::Upload` refuses a neutron cross-section table that arrives without P3's
PhotonEvaporation5.7 level data, and the reason written over the refusal was that
"`G4PhotonEvaporation::BreakUpChain` with nothing to walk emits no gamma - so the neutron's
binding energy would silently vanish". That was a guess and it is wrong by a factor of three in
the other direction. Measured, 2000 thermal captures in lead, the same tracks and the same seeds
with the table and with a null view (`tests/test_neutron_general.cu` section 4):

```
with the level scheme    2000 captures,  9,021 secondaries, 13,694.7198 MeV emitted
with a null table        2000 captures, 26,485 secondaries, 14,609.9069 MeV emitted
```

Nearly three times the multiplicity and 6.7% more energy, because the cascade takes the CONTINUUM
arm of `generate_gamma` instead of walking a discrete level scheme. A missing dataset that
produced zero would announce itself in any energy balance; one that produces a plausible capture
with a wrong spectrum is what gets found in a dose comparison six weeks later. The refusal stays
and its message carries the two numbers.

**And the obvious material was the wrong one to measure it in.** The first version of that
comparison ran in water at thermal energy and the two columns were IDENTICAL to the last digit -
2000 captures, exactly 2 secondaries each, 4448.7461 MeV either way. A thermal neutron in water
captures on HYDROGEN, and `G4NeutronRadCapture::ApplyYourself`'s `A <= 1` branch is a closed-form
two-body decay (n + p -> d + gamma) that never opens the level scheme at all. So the one capture a
water phantom mostly makes is the one that does not read the 9.52 MB this port uploads for it -
which is worth knowing for the opposite reason as well: a B1 neutron dose is not a test of the
level data.

---

### V61: the model is allowed to be a model, and the condition is a species name

`docs/PORTED.md` 4.3 is this project's most reliable rule. Geant4 does not run the model; it
runs a table built from the model, so match the grid and the spline or the numbers will be
right and the transport wrong. V5, V7 and V53 are three separate half-days spent learning it.

P14b is where it inverts, and the inversion is one line of
`G4VMscModel::GetParticleChangeForMSC` (G4VMscModel.cc:94):

```
    if(p->GetParticleName() != "GenericIon" &&
       (p->GetPDGMass() < CLHEP::GeV || ForceBuildTableFlag()) ) {
      ... xSectionTable = builder->BuildTableForModel(...) ...
    }
```

`G4VMscModel::GetTransportMeanFreePath` reads that table when it exists,

```
    x = pFactor*(*xSectionTable)[basedCoupleIndex]->Value(ekin)/(ekin*ekin);
```

and evaluates the model when it does not:

```
    x = pFactor*CrossSectionPerVolume(pBaseMaterial, part, ekin, 0.0, DBL_MAX);
```

`SetForceBuildTable` is called **nowhere** in 11.1.1. `grep -rn "SetForceBuildTable" source/`
returns its declaration in `G4VEmModel.hh`, its definition eight hundred lines below, and the
`flagForceBuildTable = false` initialiser. No caller. So the flag is always false and the test
is "not named GenericIon, and under a GeV".

Every species `G4UrbanMscModel` serves fails it. GenericIon by name; alpha at 3727.4 MeV, He3
and triton at 2808, deuteron at 1875.6 by mass. e-, e+, the muons and the singly charged
hadrons pass it - and the last two use WentzelVI, so the only species in QBBC that both reads
an Urban lambda table and exists is the electron and the positron.

**What it would have cost to get this backwards.** The port's `UrbanTable` is a 240-bin log
grid from 1 keV to 100 MeV with log-log interpolation. An 840 MeV alpha is off the top of it
entirely, so the natural "add the ion to the table that is already there" would have clamped
every B1 alpha step to the 100 MeV value - and the alpha's transport mean free path in water is
4.72e6 mm at 840 MeV against 3.08e5 mm at 200 MeV, a factor of 15. That is not an interpolation
error, it is the wrong number by an order of magnitude, and it would have arrived as a dose
difference of a per cent with nothing pointing at the table. `urban_heavy_lambda` evaluates
instead, and `tests/test_ion_msc.cu` section 2 is exact against `1/CrossSectionPerVolume` to
6.9e-16 over 1,200 points.

**The other half of the entry, which is the one that will come up again.** The three parameters
that differ between a lepton's msc and a hadron's - the step limit type, `facrange` and the
lateral-displacement flag - are chosen by the PARTICLE and not by the model.
`G4EmTableUtil::PrepareMscProcess` (G4EmTableUtil.cc:531-539):

```
    if(part.GetPDGMass() > CLHEP::MeV) {
      stepLimit = param->MscMuHadStepLimitType();          // fMinimal
      facrange = param->MscMuHadRangeFactor();             // 0.2
      latDisplacement = param->MuHadLateralDisplacement(); // false
    } else {
      stepLimit = param->MscStepLimitType();               // fUseSafety
      facrange = param->MscRangeFactor();                  // 0.04
      latDisplacement = param->LateralDisplacement();      // true
    }
```

and `G4VMscModel::InitialiseParameters` makes the same split on `abs(PDGEncoding) == 11`.
`src/physics/stepper.cuh`'s header said, for as long as it had a hadron kernel, that "`facrange`
is 0.2 rather than 0.04" as a property of **WentzelVI** - and every number it produced was
right, because every species that reached that code was a hadron. The comment had the mechanism
backwards and would have been read as authority the first time anyone gave a lepton a WentzelVI
step or an ion an Urban one. Which is what P14b did.

**What the substitution was worth.** The five species Geant4 scatters by Urban were stepped
with WentzelVI from the day each gained a kernel. Measured three ways, in increasing order of
what they touch:

- **the step count**, `tests/test_ion_transport.cu` section 2, a track stepped to a stop in
  water: alpha 200 MeV 16.5 steps -> 16.0, deuteron 50 MeV 14.5 -> 14.0, He3 20 MeV 3.5 -> 3.0,
  O16 20 MeV 1.6 -> 1.0, with the total path length unmoved to 4e-7 in every row. The proton is
  bit-identical, as it must be.
- **the step limit**, `ref/oracle/ion_msc_limit.csv`: 436 of 450 cells are not limited by msc
  at all, because `facrange*max(range, lambda0)` exceeds the whole remaining range whenever the
  transport mean free path is more than five times it - and for an ion it is 0.64 to 40,000
  times it. The seven cells where it bites are Ca40 at 0.05 MeV/u in all five materials and O16
  in the lead-bearing one, and they include both arms of
  `(currentRange > lambda0) ? facrange*currentRange : facrange*lambda0`. So the model Geant4
  gives an ion mostly declines to shorten its step, and the model this port was substituting
  did shorten it.
- **the dose**, and the honest summary is that it is small and was always going to be. Multiple
  scattering moves a track sideways; B1's scoring volume is 12 cm wide and the quantity is the
  energy deposited in it. Example B1's stage-1 alpha at 500,000 events a side: Geant4 stage 1
  12,336.9 +/- 13.0077 nGy against 12,313.6 +/- 12.9789 for the port at 2a6b379, **-0.19% and
  1.27 sigma** - and the after-number is not measured, for the reason V63 is about.

The order of those three is the lesson. The step count moved by 3%, the step limit's behaviour
changed qualitatively, and the dose did not move measurably - so a package with only the dose to
go on could not have told whether it had done anything at all. What told it was the oracle, and
the oracle had to be built for a model with no callable surface (see
`ref/dump/dump_ion_msc.cc`'s header for how: drive the real process off the particle's own
process manager, and use the fact that `ComputeTrueStepLength(g)` with `g == zPathLength`
returns `tPathLength` and mutates nothing).


---

### V62: a branch missing from the electron path, found by generalising it and left alone

`G4UrbanMscModel::SampleCosineTheta` has a sub-case this port has never had:

```
    G4bool extremesmallstep = false;
    G4double tsmall = std::min(tlimitmin,lambdalimit);
    G4double theta0;
    if(trueStepLength > tsmall) {
      theta0 = ComputeTheta0(trueStepLength,kinEnergy);
    } else {
      theta0 = std::sqrt(trueStepLength/tsmall)*ComputeTheta0(tsmall,kinEnergy);
      extremesmallstep = true;
    }
    ...
    G4double u = !extremesmallstep ? G4Exp(ltau*onesixth)
      : G4Exp(G4Log(tsmall/lambda0)*onesixth);
```

Two halves, sixteen lines apart, and the second is the one that gets dropped: `theta0` is right
where the flag is set and `u` is right where it is read.

It is reachable for an electron. `lambdalimit` is 1 mm and `ComputeTlimitmin` gives
`0.87*Z23*stepmin`, so for a 1 MeV electron in water - `stepmin = lambda0*1e-3/(2e-3 + T*
(stepmina + stepminb*T))` is 6.9e-5 mm - `tsmall` comes out at 2.3e-4 mm. Steps that short
happen at the end of an electron's range, which is where most of the dose is.

**Measured and not fixed, and the reason is the package boundary.** P14b's whole claim is that
the ion path changed and the lepton path did not, proved by an FNV-1a over every double the
e-/e+ path produces - lambda, theta0, 64 sampler draws, both path conversions, the step limit
and its two carried state variables, and the full sample_scattering including the displacement -
over 3,840 (material, charge, energy, step) cells. That hash is `7b153737a45f0b9e` before and
after. Switch `t_small` on for the lepton overload and it becomes `d366e264fe441b2f`, which is
at once the anti-vacuity check on the probe and the measurement of the gap: it is real, and it
moves B1's 6 MeV gamma dose, which is the gate P14b is required not to move.

So the ion path has the branch - `t_small` there is the 1e-7 mm that `fMinimal` never
recomputes, which makes it live only for a step between 0.1 and 1 angstrom and therefore
unreachable in transport - and the lepton path does not, named in `urban_msc.cuh`'s lepton
overload, in `docs/PORTED.md` 1.1 and in README's open question 2. Whoever owns the lepton path
next should turn it on and re-take the gamma gate deliberately.

**The anti-vacuity story here is worth more than the finding.** The first check written for
this branch - that the sampled cos(theta) is continuous at `t == tsmall` and discontinuous below
it - passes with the `u` half deleted, because `theta0`'s `sqrt(t/tsmall)` alone already
separates the branched result from the unbranched one. What sees `u` is that `lambdaeff` is
`trueStepLength/tau` with `tau = trueStepLength/lambda0`, hence lambda0 identically: so with the
branch on, every term of `xsi` is independent of the step length and two sub-tsmall steps must
give the same `xsi`. Making that comparison non-vacuous took two further conditions. `xsi` is
clamped at 1.9 wherever it lands below it, which hides the question entirely in 110 of 380
cells. And `UrbanDebug::branch` cannot be used to ask "did the main branch run", because the
struct is value-initialised to 0 and `branch` is set to 2 only once the fallbacks are live - so
a 2 nm step against air's 1e10 mm mean free path takes the `tau < tausmall` exit and leaves
`branch` 0 with `xsi` 0, indistinguishable from a main-branch call by that field. Both are
conditions in `tests/test_ion_msc.cu` section 7 now, and both counts are printed, so a material
that lifts `xsi` off the floor makes the check stronger instead of quietly weaker.


---

### V63: nine builds to find the one that compiles, and it was not the one that ran

P14b generalised `G4UrbanMscModel`'s stepping half off the electron and wired the five species
QBBC scatters by Urban - alpha, He3, deuteron, triton, GenericIon - to it in `step_hadron`.
Every test passes. `src/host/transport_run.cu` does not compile.

```
ptxas warning : Stack size for entry function
                'run_step_hadron<double, ParticleType 13, StepTap<double>>' cannot be
                statically determined
Internal error
nvcc error   : 'ptxas' died with status 0xC0000005 (ACCESS_VIOLATION)
```

V55 is the same file and the same message one size smaller, and its remedy - `__noinline__` on
the branch that almost never runs - is the first thing tried here. **It does not work, and
applying more of it makes things worse.** Nine builds, each a full `examples\B1\build.bat`
against the same worktree, with the same source at 2a6b379 as the control:

| arrangement | result |
|---|---|
| 2a6b379, no Urban dispatch at all | **built, 24 min** |
| Urban inlined at the call site | died, 3 min |
| `urban_step_limit_heavy` and the sampler `__noinline__` | died, 3 min |
| + the range lookups and the conversion moved into two helpers | **built, 28 min** |
| the Zeff coefficients bound by reference rather than copied | died, 3 min |
| + two more `__noinline__` wrappers, coefficients resolved inside | died, 4 min |
| those wrappers removed, `alignas(16)` on `UrbanCoeffs` kept | died, 15 min |
| coefficients computed instead of read, two helpers | died, 4 min |
| three helpers, kernel body holds only the calls | died, 18 min |
| + WentzelVI's 264-byte state made dead in the Urban kernels | died, 20 min |

Four things that table settles, and they are why it is here rather than a sentence.

**It is deterministic.** The same source compiled twice dies at the same four minutes. The
varying times are how far ptxas gets before it falls over, not noise.

**It names the culprit.** Species 13 is `kGenericIon`; 12, 16 and 17 - He3, deuteron, triton -
appear in the warnings of the later runs. The Urban kernels and only those. Nothing on the
lepton, gamma, neutral or singly-charged-hadron side is implicated, and the proton kernel is
byte-identical to the control in every column of `-Xptxas -v`.

**Per-kernel size is not the variable.** V55's one-kernel reproducer - a ten-line translation
unit instantiating exactly one `run_step_hadron` - compiles every one of these arrangements in
about seventy seconds, and it says the Urban kernels got *smaller*:

    kernel        registers   stack frame      spill st/ld          compile
    proton        255 -> 255  3904 -> 3904 B   200/444 -> 200/444   60 -> 65 s
    alpha         255 -> 255  3904 -> 3888 B   252/344 -> 160/224   65 -> 71 s
    GenericIon    255 -> 255  3984 -> 3888 B   360/432 -> 244/340   67 -> 78 s

The register budget this package was warned about had room it did not need: Urban's stepping
half has no per-element table where WentzelVI's `WentzelElementXs` is two arrays of sixteen
doubles the sampler picks a target atom out of. So the thing that dies is the TRANSLATION UNIT,
exactly as V55 said - and unlike V55 there is no amount of `__noinline__` that fixes it.

**It is a cliff and not a slope.** Rows 4 and 5 differ by one binding of one local. Rows 9 and
10 remove code and get further without getting there. An arrangement either falls on the right
side or it does not, for no reason visible in the source, and the only lever left with real
headroom is to stop asking one translation unit to hold twenty kernels.

#### The one that compiled then faulted

Row 4 built, and the 500,000-event stage-1 alpha run through it ended with

    CUDA error misaligned address at src/host/transport_run.cuh:89

on the first batch, while the 2,000,000-event gamma run through the same binary was fine. Line
89 is the `cudaMemcpy` in `TrackBuffer::count()`, which is simply the first synchronising call
after the launch, so it says nothing about where.

The obvious suspect was wrong, and eliminating it is worth recording because it is a real
hazard that happens not to be this one. `em::UrbanCoeffs` is `G4UrbanMscModel::mscData`'s field
list - seventeen doubles, 136 bytes, not a multiple of 16 - so in `UrbanTable::coeffs[]` the
odd-indexed entries start 8 bytes off a 16-byte boundary, and B1's water envelope is material
1. Row 4 copied one of those out by value where `step_lepton` has always bound a reference, and
nvcc reads a struct copy of that size with `ld.global.v2.f64`. That is a complete and plausible
account of the symptom, including why only the alpha run saw it. **It is also not what
happened**: a twelve-line device probe that copies `coeffs[mat]` by value for every material
out of a real uploaded table returns `cudaSuccess` on all four, odd indices included. The
alignment reasoning was right and the conclusion was wrong, which is V37's shape again.

Where the fault actually is remains unknown. `compute-sanitizer --tool memcheck` on this card
answers "Device not supported. Please refer to the Supported Devices section" and reports only
the host-side API error, so the faulting kernel and instruction are not available, and the
arrangement that produced it cannot be rebuilt to try again - rows 5 through 10 are every
attempt to get back to a compiling shape and none of them compiled.

#### What is on the branch, and what it is worth

`kUrbanIonMscWired` in `src/physics/stepper.cuh` is `false`. The dispatch, the model and the
oracle are all there; the flag makes the Urban branch compile-time dead so the engine builds,
and every ion is still scattered by WentzelVI - the substitution `docs/PORTED.md` has recorded
since the alpha was first transported. It is one line, and the work to flip it is to split
`transport_run.cu` so the hadron kernels are their own translation unit. That file,
`build_engine.bat` and `build_all.bat` belong to P1.

The physics behind the flag is not a sketch. `tests/test_ion_msc.cu` against three new oracle
files - `ion_msc_step.csv`, `ion_msc_limit.csv`, `ion_msc_sample.csv`, 750 + 450 + 300 rows over
(alpha, He3, C12, O16, Ca40) x five materials x six energies per nucleon:

- the transport mean free path to **6.9e-16** over 1,200 points;
- the step limit and both directions of the true<->geometric path conversion to **exactly 0**
  over 750 rows;
- the randomised limit within **1.1 sigma** of its mean and 1.5 of its standard deviation on the
  seven of 150 cells it is reached in;
- the angle within **3.4 sigma** on `<1 - cos>` at **chi2/bin 2.18** over 300 cells of 400,000
  draws, with the lateral displacement exactly zero on all 300 rows on both sides;
- and it runs on the DEVICE: `tests/test_step_hadron.cu` agrees host against device to 1.06e-13
  over 900 steps with the flag on, and `tests/test_ion_transport.cu` steps an alpha, a
  deuteron, a triton, He3 and an oxygen recoil to a stop through it.

And what the substitution is worth, measured with the flag on, which is the number this entry
exists to leave behind:

- **Step counts**, `tests/test_ion_transport.cu` section 2, a track stepped to a stop in water:
  alpha 200 MeV 16.5 steps -> 16.0, deuteron 50 MeV 14.5 -> 14.0, He3 20 MeV 3.5 -> 3.0, O16
  20 MeV 1.6 -> 1.0, total path length unmoved to 4e-7 in every row, proton bit-identical.
- **The step limit**: `facrange*max(range, lambda0)` exceeds the whole remaining range in 436 of
  450 oracle cells, because Urban's transport mean free path for an ion runs from 0.64 to 40,000
  times its range. So Geant4's model mostly declines to shorten an ion's step where the
  substituted one shortened it.
- **The dose**: example B1's stage-1 alpha, 840 MeV, 500,000 events a side. Geant4 stage 1 gives
  12,336.9 +/- 13.0077 nGy and the port at 2a6b379 gives 12,313.6 +/- 12.9789 - **-0.19%, 1.27
  sigma**. The after-number is not measured, because measuring it needs the engine the first
  half of this entry is about. B1's scoring volume is 12 cm wide and multiple scattering moves a
  track sideways, so the expectation was always that this would be the least sensitive of the
  three, and the step counts are the reason to believe the model changed at all.

B1's 6 MeV gamma gate through the row-4 binary - the only engine ever built with this code -
reads **425.847 pGy against 427.385, 1.2514 sigma**, which is main's recorded number to every
digit it prints. The lepton path did not move, which is the other half of what P14b had to show.

### V64: every electron above 100 MeV was a 100 MeV electron

Found by the twelve-beam B1 sweep of 2026-09-11 (docs/B1_SWEEP.md), the first time this port was
run with an electron above 100 MeV and compared to anything. At 1 GeV the port deposits 46% of
Geant4's dose in B1's trapezoid; at 150, 300, 600 and 1000 MeV it deposits 110.6, 110.8, 110.7 and
110.6 nGy per 100,000 electrons - the 100 MeV row's 110.4 - while Geant4 rises from 110.7 to 240.3.

The mechanism is a table ceiling. `src/physics/em/electron_processes.cuh` builds the e+- dE/dx,
range and inverse-range table from `e_min = 1 keV` to `e_max = 100 MeV` in 128 bins; `range()`
returns the last bin for any kinetic energy at or above `e_max`, and `energy_from_range()` returns
`e_max` for any range past the table. So a 1 GeV electron's first step reads a 100 MeV range,
takes its step, inverts the remaining range and comes out at or below 100 MeV. The other 900 MeV
is not deposited, not carried away by a secondary and not counted: it is gone. Every dose
downstream of that step is a 100 MeV electron's. Geant4 builds these tables from 100 eV to 100 TeV
at 7 bins per decade (`G4EmParameters`: `minKinEnergy`, `maxKinEnergy`, `nbinsPerDecade`), and the
port's hadron tables already use that grid through the same `G4LossTableBuilder` transcription.

Three things let it sit there. The B1 gate is a 6 MeV photon beam, whose electrons never reach
10 MeV. Every electron oracle compares a model function at a point, which is right for any energy
the caller passes - the ceiling is in the table the transport reads, not in the physics. And the
missing energy went nowhere that any counter watched: the refusal ledger counts species, the
overflow counter counts tracks, and neither counts an energy that a clamp discarded. The lesson
for the checks is the one V29 drew for assertions: a transport needs an energy balance, primary
energy in against deposited plus escaped plus refused, and this port has one only in the
depth-dose gate, for protons.

The fix is package P14c: the e+- tables on Geant4's grid with the clamp replaced by a loud refusal,
`G4eBremsstrahlungRelModel` dispatched above 1 GeV (it is ported and tested, and used nowhere in
transport), `G4WentzelVIModel` for e+- above `MscEnergyLimit()` where Geant4 switches from Urban,
and an energy balance in the device stepping test. Until it lands the port is not to be used for
electrons or positrons above 100 MeV.

### V65: the translation unit was not the variable, and a smaller one dies where a bigger one lives

`src/host/transport_run.cu` instantiated the engine class, and the eighteen `<<<>>>` launches
inside `BeamOn` implicitly instantiated eighteen stepping kernels into that one file. It took
**25 minutes 02 seconds** of nvcc, three packages had waited on it, and with P14b's Urban branch
live it did not compile at all (V63). This is the split, what it measured, and the two things in
V55 and V63 that it corrects.

#### The mechanism, and the check that it is working

An explicit instantiation DECLARATION suppresses implicit instantiation, and nvcc honours it for
a `__global__` template. Measured on a ten-line pair rather than assumed, because everything
below rests on it:

| | launching object carries the kernel | links | runs |
|---|---|---|---|
| `extern template __global__ void k<int,3>(int*);` in the header | **no** - `cuobjdump -res-usage` reports `GLOBAL:0` and no `Function` line | yes | yes, right answer |
| the same pair with that line deleted | **yes** - `_Z1kIiLi3EEvPT_`, REG:8 | yes | yes |

So the declaration is load-bearing and the inversion says so. The kernels stay in
`transport_run_impl.cuh`; the block under them declares all eighteen stock specialisations
`extern template`, and one `transport_run_<species>.cu` per kernel provides the definition.
`transport_run.cu` keeps the engine's host code and the three utility kernels - the launching
object's own device code went from 21 functions to 3.

**The link will not catch a mistake here, which is why there is a check on the artefact.** The
comment at the top of `transport_run_impl.cuh` says a `__global__` template instantiated in two
translation units is rejected with "explicit specialization ... is not a specialization of a
function template". On CUDA 11.6 it is not: explicit instantiation definitions of the same
specialisation in two objects link **cleanly**, the stubs being COMDAT-folded (row 3 of the same
experiment). A launch added without a declaration therefore costs twenty-five minutes of nvcc
and says nothing at all. `build_engine.bat` runs `cuobjdump -res-usage out\transport_run.obj |
findstr run_step_` and fails the build on a hit.

#### The thing this entry exists to correct: smaller is not safer

The first arrangement was one unit per kernel FAMILY - gamma, lepton, neutral, nucleon (p, pbar),
muon (mu-, mu+), meson (pi+, pi-, K+, K-), ion (the five Urban species). Seven of the eight
compiled. **The meson unit died with `ptxas died with status 0xC0000005 (ACCESS_VIOLATION)`** -
the same message as V55 and V63 - and it died with `kUrbanIonMscWired` still FALSE, on code that
compiles perfectly well as four of the eighteen kernels in the single unit this split replaced.
It died twice: at 215 s inside the eight-way parallel build, and at **99.7 s alone on an idle
machine with 31 GB free**, which is what rules out memory pressure and confirms V63's
"deterministic".

Then the decisive measurement: **one `run_step_hadron` in a unit of its own compiles in 90
seconds.** pi+ 89.5 s, K+ 93.4 s, both from the family that had just failed.

So V63's conclusion - "the thing that dies is the TRANSLATION UNIT ... the only lever left with
real headroom is to stop asking one translation unit to hold twenty kernels" - is half right. The
translation unit's SHAPE is the variable; its SIZE is not, and not even monotonically. Eighteen
kernels live, four die, one lives. V63 called it "a cliff and not a slope"; the correction is that
the cliff faces both ways, and the only shape this project has ever measured ptxas to compile for
every arrangement of the physics - V55's reproducer, V63's nine builds, and these - is one
`run_step_hadron` on its own. The thirteen charged-hadron kernels are therefore thirteen units.
`run_step_lepton`'s two instantiations and `run_step_neutral`'s two share a unit each, measured at
269 s and 260 s; both are one template switched by a compile-time constant and neither is the
build's long pole.

#### The compile time

Same machine, `nvcc -std=c++17 -O2 -arch=sm_86 -Xptxas -v -c`, with three other worktrees
building beside it as they always are:

| | wall |
|---|---|
| one translation unit, eighteen kernels (main at c5c3922) | **25 min 02 s** |
| eight family units, all started at once (meson died) | 8 min 54 s |
| seventeen units, six at a time | **6 min 31 s** |

Per unit, measured serially and from the sentinel timestamps of the parallel runs: engine 68 s,
one hadron kernel 90 s, gamma 146 s, neutral 260 s, lepton 269 s. The FAMILY units, for the
record that the per-kernel cost rises as the unit shrinks: nucleon (2 kernels) 337 s, muon (2)
331 s, ion (5) ~560 s - against (1501 - 68)/18 = 80 s a kernel in the single unit. The split
costs more CPU in total and less wall time, which is the trade it was made for. It also costs
object size: each unit carries its own copy of every shared `__device__` function, so the
seventeen objects total **93,738,021 bytes** against one object of 21,829,799, and the archive
over them is 98,010,048.

#### The registers, which are NOT byte-identical, and the dose, which is

`-Xptxas -v`, before in the single unit and after in the split units. Registers are 255 in every
stepping kernel on both sides, which is the number the 16384-byte stack limit `Upload` sets is
measured against (V22). The stack frames and the spills move in both directions, because ptxas
allocates per MODULE and the modules changed:

| kernel | stack frame | spill st/ld | registers |
|---|---|---|---|
| `run_step_gamma` | 2416 -> **3040** B | 368/676 -> **52/20** | 255 |
| `run_step_lepton` (both) | 3040 -> **3024** B | 80/28 -> **84/32** | 255 |
| `run_step_hadron` pi+, pi-, K+, K-, mu-, mu+ | 4592 B, unchanged | 100/52 -> **76/52** | 255 |
| `run_step_hadron` proton | 4592 -> **3920** B | 100/52 -> **200/444** | 255 |
| `run_step_hadron` antiproton | 4592 -> **3664** B | 100/52 -> **200/444** | 255 |
| `run_step_hadron` alpha | 4592 -> **3920** B | 100/52 -> **252/344** | 255 |
| `run_step_hadron` deuteron, triton | 4592 -> **3920** B | 100/52 -> **184/396** | 255 |
| `run_step_hadron` He3 | 4592 -> **3984** B | 100/52 -> **192/396** | 255 |
| `run_step_hadron` GenericIon | 4592 -> **4000** B | 184/140 -> **360/432** | 255 |
| `run_step_neutral<kNeutron>` | 7264 -> **7472** B | 248/540 -> **296/516** | 255 |
| `run_step_neutral<kPiZero>` | 3152 -> **3008** B | 524/848 -> **528/920** | 255 |
| `seed_from_primaries` | 704 -> **624** B | 96/88 -> **0/0** | 255 -> **234** |

`cmem[0]` is unchanged in every row - 1464 for the gamma and lepton kernels, 1616 for the
hadron, 1640 for the neutral - which is the check that only the allocation moved and not the
argument lists.

Three things in that table are worth reading twice. The gamma kernel's frame GREW 624 bytes
while its spill traffic FELL from 368/676 to 52/20 without one character of its code changing -
the exact reverse of the movement V55 recorded for the same kernel when the unit grew, and the
same phenomenon: with eighteen entry points in a module ptxas was rationing, and with one it is
not. `seed_from_primaries` is the clearest case, 21 registers and 88 bytes of spill loads
cheaper for being alone. And the six singly charged mesons and muons held 4592 bytes exactly
while every nucleon and ion kernel dropped 600-900 bytes and took several hundred bytes of extra
spill instead - so the reallocation is not uniform even across instantiations of one template.

**These are V55's and V63's reproducer numbers, and that is the cross-check on the whole
arrangement.** V63's one-kernel control column reads proton 3904 B / 200/444, alpha 3904 /
252/344, GenericIon 3984 / 360/432. The shipped units read 3920 / 200/444, 3920 / 252/344 and
4000 / 360/432 - +16 bytes, which is main having moved since. The engine is now built in exactly
the shape V55 recommended for measuring it.

None of this reaches the arithmetic, and the check that it does not is the dose. B1's
2,000,000-event gamma gate through the split engine reads **425.847 pGy +/- 0.867682 against
Geant4's 427.385 +/- 0.87, 1.25138 sigma** - main's recorded number to every digit it prints,
taken twice. Four more seeds through the same binary: 426.195 (0.968 sigma), 426.917 (0.381),
427.489 (0.085), 427.288 (0.079) - mean of the five 426.747 +/- 0.315 pGy of seed scatter,
which is the reference the next two sections are measured against.

#### The two claims this arrangement rests on, each inverted once

**"A species with no instantiation anywhere is an unresolved symbol rather than a track that is
never stepped."** Archived sixteen of the seventeen objects, leaving the proton's out, and
relinked example B1 against that:

```
bad.lib(transport_run.obj) : error LNK2019: unresolved external symbol
  "void __cdecl g4gpu::host::run_step_hadron<double,9,struct g4gpu::StepTap<double> >(...)"
  referenced in function
  "...g4gpu::host::TransportEngine<double,struct g4gpu::StepTap<double> >::BeamOn(...)"
fatal error LNK1120: 1 unresolved externals
```

Species 9 is `kProton`, and the function it is missing from is named. So the glob in
`build_engine.bat` cannot silently lose a species.

**"The engine's own object carries no stepping kernel."** Removed `extern template
G4GPU_STEP_GAMMA(StepTap<double>);` from `transport_run_impl.cuh`, recompiled
`transport_run.cu` alone - 77.2 s, which is itself the measure of what the eighteen kernels were
costing that file - and ran the gate's own command:

```
> cuobjdump -res-usage out\transport_run.obj | findstr /c:"run_step_"
 Function _ZN5g4gpu4host14run_step_gammaIdNS_7StepTapIdEEEE...
```

With the declaration in place the same command matches nothing. Both halves measured on the real
engine rather than on the ten-line pair, and the declaration put back afterwards.

#### Appended: the same translation unit, arriving through a different door

This entry split the ENGINE'S translation unit and stopped there, because the engine was the
file that would not compile. It was not the only file of that shape, and the first `build_all`
over V65 and V66 found the other one at the drivers stage.

`tests/test_custom_hook.cu` is a PROJECT rather than a test of a function: it defines its own
device stepping action and instantiates `TransportEngine<double, QualityFactorScoring>` in its
own `.cu`, which is how a user gets a hook into the transport without rebuilding g4gpu. The
eighteen `<<<>>>` launches inside `BeamOn` therefore instantiated eighteen stepping kernels
into THAT file - and the engine's split bought it nothing, because a kernel templated on the
hook type is a different specialisation and `out/transport_run.lib` holds none of them. With
V66's two switches on it died exactly as the engine had:

```
ptxas warning : Stack size for entry function
                'run_step_hadron<double, ParticleType(13), QualityFactorScoring>' ...
Internal error
nvcc error   : 'ptxas' died with status 0xC0000005 (ACCESS_VIOLATION)
```

Species 13 is `kGenericIon` - V63's named culprit again, in a file V63 never looked at.
Everything ahead of it built: the seventeen-unit engine, all 76 tests, both GPU suites.
`build_all.bat`'s own comment had called the three and a half minutes that unit cost "the
honest cost of the arrangement", which it was; what it was not is necessary.

**The fix generalises because the hook is a template parameter like any other.**
`build_hook_engine.bat <header> <type> <tag>` writes one translation unit per stepping kernel
for a project's hook type, compiles them six at a time through the same `build_engine_unit.bat`
the engine uses, and archives them into `out\hook_<tag>.lib`. The project includes the generated
`hook_kernels.cuh`, whose `extern template` declarations keep those kernels out of its own
object, and links the archive. Two projects in this repository have their own hook and both get
it: `test_custom_hook` (QualityFactorScoring) and `test_voxel_scoring` (CellTap).

**The kernel list is READ OUT of `transport_run_impl.cuh`'s `extern template` block, not copied
into the generator.** A copy would be right the day it was written and silently wrong the first
time a species was added: the new kernel would have no unit and no declaration and would go back
to being instantiated in the project's own object, which is the failure this exists to cure. The
generator substitutes the hook type for `StepTap<double>` and refuses to run if a line it parsed
does not name it.

**What it costs and what it buys.** `QualityFactorScoring`'s eighteen kernels take **346.7 s**
as eighteen units six at a time - against 366.4 s for the engine's own seventeen - and all
eighteen compiled first time, `run_step_hadron<double, kGenericIon, QualityFactorScoring>`
among them. The project's own translation unit now compiles in **18.4 s**, where the one-unit
arrangement was 208 s when it still worked. `tests/test_custom_hook.exe` links and passes: the
action's energy sum against the scorer's is **rel 0.00e+00**, dose-averaged Q 1.5134 over 3536
scored events in [1, 5], 3421 secondaries walked against 3421 reported.

`tests/test_voxel_scoring.cu` is the second project of that shape and it had been waiting its
turn to die - build_all reached `test_custom_hook` first and stopped there. CellTap's eighteen
units take **375.0 s**, its own unit 21.3 s, and it passes: 147,794 steps recorded over 2,000
events with **0 dropped**, 4,205 depositing steps ending exactly at a cell boundary and **0**
crossing one, worst overshoot 1.0e-07 mm against a 1.25 mm cell, 64 of 64 cells scoring, and the
device total against the host's to **exactly 0**. `CellRec` came out of an anonymous namespace to
reach the header, which is a fix rather than a move: in a header it would have been a different
type in every unit, and `CellTap` - whose member is a `CellRec*` - would have had one definition
per unit with a different member type in each, linking cleanly because nothing about `CellRec`
reaches the kernels' mangled names.

**A hook kernel costs almost exactly what the stock one costs**, which is the measurement that
says the arrangement is carrying no hidden price. `-Xptxas -v`, the same species, one kernel per
unit either way:

| kernel | `StepTap<double>` | `QualityFactorScoring` |
|---|---|---|
| `run_step_gamma` | 3040 B frame, 52/20 spill, 255 reg, cmem[0] 1464 | 3024 B, 36/16, 255, **1456** |
| `run_step_hadron` kAlpha | 3760 B, 148/208, 255, 1616 | 3760 B, 144/200, 255, **1608** |
| `run_step_hadron` kGenericIon | 3760 B, 228/296, 255, 1616 | 3760 B, 224/296, 255, **1608** |

The eight bytes of `cmem[0]` are the hook itself: it travels as a by-value kernel argument, and
`StepTap<double>` is three pointers, two ints and a bool padded to 40 bytes where
`QualityFactorScoring` is three pointers and two ints at 32. Everything else is within the
allocation noise V65 already recorded for this compiler. There is no "before" column for a hook
kernel in the one-unit arrangement and there cannot be one: ptxas does not reach a register
report, it dies.

**Inverted twice.**

First, the one line is the whole mechanism, so it was taken out again. `test_custom_hook.cu`
with `#include "hook_kernels.cuh"` commented out - everything else identical, the archive still
on the link line - is the arrangement that failed, and it fails the same way:

```
ptxas warning : Stack size for entry function
                'run_step_neutral<double, ParticleType(15), QualityFactorScoring>' ...
ptxas warning : Stack size for entry function
                'run_step_neutral<double, ParticleType(14), QualityFactorScoring>' ...
ptxas warning : Stack size for entry function
                'run_step_hadron<double, ParticleType(13), QualityFactorScoring>' ...
Internal error
nvcc error   : 'ptxas' died with status 0xC0000005 (ACCESS_VIOLATION)
```

**504.6 s to die**, on the same three kernels build_all named, which is also the measure of what
that unit was costing when it worked. The declarations restored, the same file compiles in 18.4
s. So the split is load-bearing rather than decorative, and the passing build is evidence about
the code.

Second, the generated header's ordering requirement is a `#error` rather than a convention, and
it fires: a ten-line `.cu` that includes `hook_kernels.cuh` without `transport_run_impl.cuh`
ahead of it stops at `fatal error C1189: #error: "include host/transport_run_impl.cuh before
hook_kernels.cuh"`. Without the guard that mistake is a pile of "identifier not found" errors
pointing at the generated file rather than at the include order that caused them.

**The generated project needs none of this, and that was checked rather than assumed.**
`src/builder/write_project.cc` writes a `SteppingAction` deriving from `G4UserSteppingAction`
that takes a `const G4Step*` - a HOST action - and the project links `%G4GPU_ENGINE_OBJ%`, the
stock `StepTap<double>` archive. `G4STEP_HOOK`, `G4VUserDeviceSteppingAction` and `SetStepHook`
appear zero times in the generator, so a generated project instantiates no kernel of its own and
compiles in seconds. Its template is left alone; if it ever gains a device hook, this mechanism
is what it needs and the two call sites in `build_all.bat` are the pattern.

It still builds and runs with the engine in seventeen units and two hook archives beside it:
**56.9 s** for the eight generated sources and the link, and `MyDetector.exe -n 10000` reports
`dose1 220.721 MeV / 1309.8 pGy`, `cells 194.774 MeV / 975 pGy`, and the host `SteppingAction`'s
own sum at **220.721 MeV** - the same number `dose1` reports, by the other route, which is what
`tools/compare_project.ps1` exists to compare.

**And a caveat that this package does not own but had to work around.** The pipeline's
generated-project stage cannot be run from a worktree as written. `src/host/g4builder.cu` line
1874 saves the selftest's project to the literal `"D:/g4gpu/out/selftest_project"`, and the
dialog PNGs beside it to the same hard-coded root; `src/builder/write_project.cc` writes
`call "D:/g4gpu/build_engine.bat"` into every project's `build.bat` and `set(G4GPU_DIR
"D:/g4gpu")` into its CMakeLists. So a `g4builder.exe -selftest` built in a worktree writes into
MAIN'S `out\`, and the `build.bat` it generates would rebuild MAIN'S engine - which is V45, and
what every worktree agent is told not to do. The numbers above were therefore taken on a COPY:
the generated project copied into the worktree with those absolute references rewritten, built
against this branch's engine. What is under test - the generated sources and the generated build
recipe - is untouched by that substitution. Recorded because it also means the generated-project
stage of a `build_all` run from a worktree is testing main's tree rather than the branch.


### V66: two switches that had been shut by a compiler and by a package boundary

V65 split the transport translation unit. This is what the split freed, and both halves are the
same shape: a piece of Geant4 that had been transcribed, checked against the oracle and then
left switched off for a reason that was not the physics.

#### `kUrbanIonMscWired`: the flag V63 could not turn on

The dispatch, the model and three oracle files were all in place when V63 was written, and the
flag was false for one reason: `transport_run.cu` would not compile with it true. With V65's
split it does. The five species Geant4 scatters by Urban are five translation units now, and
all five compiled first time, in the same ninety seconds each as the eight WentzelVI ones -
which is also the answer to V63's own open question, because V63 could not distinguish "the
Urban branch is too big" from "the translation unit is too big" and the one-kernel units settle
it: the branch was never the problem.

**Nothing in the physics changed to make that happen**, and this entry claims no new physics.
`tests/test_ion_msc.cu` with the flag true reproduces every number V63 recorded, against
`ref/oracle/ion_msc_{step,limit,sample}.csv`: the transport mean free path to **6.93e-16** over
1,200 points, the Zeff coefficients to **exactly 0**, the step limit and both path conversions
to **exactly 0** over 750 rows, the randomised limit within **1.10 sigma** of its mean and 1.49
of its standard deviation on the 7 of 450 cells it is reached in (436 rows the limit does not
touch at all), and the angle to **3.43 sigma** on `<1 - cos>` at **chi2/bin 2.18** over 300
cells of 400,000 draws.

**And it runs on the device.** `tests/test_step_hadron.cu` agrees host against device to
**1.060e-13** over 900 steps of six species, and `tests/test_ion_transport.cu` steps all five to
a stop: alpha 200 MeV in **16.0** steps, deuteron 50 MeV in **14.0**, He3 20 MeV in **3.0**, O16
20 MeV in **1.0** - V63's after-column, now taken through the shipped engine rather than through
a test.

**The Urban branch is CHEAPER than the WentzelVI it replaces**, which is V63's finding confirmed
in the shipped units. `-Xptxas -v`, the same per-species unit with the flag off and on:

| kernel | stack frame | spill st/ld | registers | cmem[0] |
|---|---|---|---|---|
| alpha | 3920 -> **3760** B | 252/344 -> **148/208** | 255 | 1616 |
| He3 | 3984 -> **3760** B | 192/396 -> **144/192** | 255 | 1616 |
| GenericIon | 4000 -> **3760** B | 360/432 -> **228/296** | 255 | 1616 |
| deuteron, triton | 3920 -> **4112** B | 184/396 -> **132/180** | 255 | 1616 |
| proton | 3920 B, unchanged | 200/444, unchanged | 255 | 1616 |
| gamma, lepton, neutral | unchanged | unchanged | 255 | unchanged |

Urban's stepping half has no per-element table, where WentzelVI's `WentzelElementXs` is two
arrays of sixteen doubles the sampler picks a target atom out of. So the register budget V63 was
warned about had room it did not need, and the thing that killed ptxas was never the size of
this branch. The last two rows are the other half of the claim: only the five species Geant4
gives an Urban model moved at all, and the proton's kernel is identical in every column, which
it has to be because `uses_wentzel_msc(kType)` is a compile-time constant in each instantiation.
The whole engine took **376.4 s** to build with the flag true against 391.1 s with it false -
the branch is cheaper to compile as well as to run.

**The dose, which is the number V63 could not take.** Example B1's stage-1 alpha, 840 MeV,
500,000 events, against the Geant4 side V60 and V63 record:

| | dose in the scoring volume | vs Geant4 |
|---|---|---|
| Geant4 11.1.1, stage 1 | 12,336.9 +/- 13.0077 nGy | - |
| the port with WentzelVI substituted (2a6b379) | 12,313.6 +/- 12.9789 nGy | -0.19%, 1.27 sigma |
| **the port with Urban, this branch** | **12,316.4 +/- 12.9790 nGy** | **-0.17%, 1.12 sigma** |

So the substitution was worth **+2.8 nGy, +0.023%**, and it moved the row TOWARD Geant4 by 0.15
sigma. That difference is not resolvable and was never going to be: the two port runs are
independent samples whose difference carries 13*sqrt(2) = 18.4 nGy of noise, B1's scoring volume
is 12 cm wide, and multiple scattering moves a track sideways. **The order of the three
measurements is the lesson, and it is V61's:** the step count moved 3%, the step limit's
behaviour changed qualitatively in the 14 of 450 oracle cells where it bites at all, and the
dose did not move measurably - so a package with only the dose to go on could not have told
whether it had changed anything. What told it was the oracle.

**And B1's gamma gate could not move, which is worth measuring rather than assuming.** A 6 MeV
photon beam makes no ion: Compton, the photoelectric effect, Rayleigh and pair production make
electrons and positrons, and nothing in that chain reaches `run_step_hadron` at all. Measured:
**425.847 pGy +/- 0.867682, 1.25138 sigma**, and 426.195 / 426.917 / 427.489 / 427.288 on the
four extra seeds - bit-identical to the flag-off engine in all five, to every digit printed.

#### The electron's `extremesmallstep`: V62's gap, decided by measurement

V62 is the entry; the short form is that `G4UrbanMscModel::SampleCosineTheta` evaluates `theta0`
at `tsmall = min(tlimitmin, lambdalimit)` and scales it by `sqrt(t/tsmall)` for a step below
that, and sixteen lines further down takes the tail parameter `u` from `log(tsmall/lambda0)`
rather than from `log(tau)`. Both halves have been in `em::urban_sample_cos_theta` since P14b
generalised it. What the lepton path lacked was the THRESHOLD: it passed zero.

**The threshold is the one the track carries, and that is the whole correctness of the branch.**
`p.msc_tlimitmin` is written by `urban_step_limit` on the first step and after each boundary and
held for every step in between, which is exactly how Geant4 holds its member. A tlimitmin
recomputed at the sampling site would be a different number wherever the branch actually fires -
many steps after the last refresh, at the end of a range - so it would be right precisely where
the branch never runs. `em::urban_t_small` does the `min`; `step_lepton` passes it.

**And the `min` is load-bearing, which was not obvious.** Measured over B1's four materials:

```
material   E(MeV)    lambda0(mm)    stepmin(mm)  tlimitmin(mm)     tsmall(mm)
water           1        6.29035    0.000299137    0.000580732    0.000580732
water           6         113.72    0.000435918    0.000846272    0.000846272
bone            1        2.74506    0.000146284    0.000342135    0.000342135
air             1        5195.56        0.35552         1.1599              1
air             6          93979       0.494985        1.61491              1
```

In water and bone `tsmall` IS `tlimitmin`, three to eight ten-thousandths of a millimetre. In AIR at
1 and 6 MeV `tlimitmin` runs past `lambdalimit` and the 1 mm cap is what takes effect - so a
version that dropped the `min` as "almost always tlimitmin" would have been wrong in the one
material B1's world volume is made of.

**The branch is reachable and it does something, and neither is assumed.** The same step sampled
from the same seed with the threshold off and on, 1 MeV electron in water, `tsmall` = 5.80732e-4
mm:

| step | cos off | cos on | |
|---|---|---|---|
| 0.01 tsmall | 0.999999084 | 0.999999674 | differs |
| 0.5 tsmall | 0.999985984 | 0.999983694 | differs |
| 0.999 tsmall | 0.999967428 | 0.999967421 | differs |
| **1.000 tsmall** | 0.999967389 | 0.999967389 | **identical** |
| 1.001 tsmall | 0.999967349 | 0.999967349 | identical |
| 100 tsmall | 0.992640282 | 0.992640282 | identical |

Continuous at `t == tsmall`, live below it, inert above it - which is the shape the transcription
has to have, and is what V62's own anti-vacuity note says is easy to get half right.

**The decision is the gate's, and the gate barely notices.** Example B1's 2,000,000-event gamma
gate, five seeds, the same source tree rebuilt with one `constexpr` different and every other
object in the archive the same file. The dose column is `427.385 +` the difference each run
prints, because the "scaled to 10k events" line prints six significant figures and the movement
is in the seventh:

| seed | off (pGy) | on (pGy) | change | track-steps, off -> on |
|---|---|---|---|---|
| default | 425.84739 | 425.84670 | -0.00069 | 26,064,061 -> 26,064,064 |
| 1 | 426.19466 | 426.19534 | +0.00068 | 26,026,886 -> 26,026,732 |
| 2 | 426.91679 | 426.91701 | +0.00022 | 26,038,801 -> 26,038,750 |
| 3 | 427.48937 | 427.48706 | **-0.00231** | 26,038,916 -> 26,038,655 |
| 4 | 427.28796 | 427.28824 | +0.00028 | 26,050,353 -> 26,050,418 |
| **mean of five** | **426.74723** | **426.74687** | **-0.00036 +/- 0.00053** | |

**It stays on.** The largest single-seed movement is 0.0023 pGy - 0.0026 of the 0.87 pGy that
run's own rms gives it, and 0.00054% of the dose - and the mean of five is -0.00036 +/- 0.00053
pGy, which is consistent with zero at 0.7 standard errors. The port's agreement with Geant4
reads 1.25138 sigma before and **1.25195** after on the gate's
own seed, and 0.67 sigma either way on the five-seed mean (426.747 +/- 0.388 against 427.385 +/-
0.87). The rule of this project is that Geant4's code is the answer unless the port moves away
from Geant4 by more than statistics, and this does not move the port at all; there is nothing
here to run down.

**V62's sentence needs one qualification, and it is a qualification and not a correction.**
"Switching it on moves every lepton number in the port including B1's 6 MeV gamma dose" is true,
and it is what P14b was right to protect: the gate is NOT bit-identical across this flag, and
P14b's claim was that its commit changed the ion path and did not touch a lepton, which any
movement at all would have broken. What V62 had no reason to measure is the SIZE of it. The
branch fires on roughly one lepton step in 10^5 of B1's shower - the track-step count moves by
+3, -154, -51, -261 and +65 out of 26 million - and where it fires it is worth a few
milliradians of a deflection that was a few milliradians anyway (the probe table above: 1.4
mrad against 0.8 at a hundredth of `tsmall`). The threshold table says where those steps are:
`tsmall` is 3 to 8e-4 mm in the water, A-150 and bone that hold all of B1's dose, and the 1 mm
cap in the AIR of the world volume - so the material the branch is most often reached in is the
one with nothing to deposit. A flag that had to be decided by this dose could not have been
decided; what decides it is that it is Geant4's code and the dose says it costs nothing.

**The stage-1 alpha row does not move either, and that is not a tautology**: an 840 MeV alpha
makes delta rays, and every one of them is stepped by `run_step_lepton`. 500,000 events with the
branch on read **12,316.4 +/- 12.9790 nGy, -1.12 sigma**, this entry's own alpha row to every
printed digit, with **33,432,276** track-steps against 33,432,228 - 48 lepton steps of 33 million
took the branch and the dose did not notice.

**And the kernel costs nothing for it, measured rather than expected.** `-Xptxas -v` on the
shipped lepton unit: `run_step_lepton<double,true,...>` and `<double,false,...>` both 3024 bytes
of stack frame, 84/32 bytes of spill, 255 registers and 1464 bytes of cmem[0] - V65's flag-off
column to the byte, for a branch that adds a `min`, a compare and two transcendentals on the path
it takes. The kernel was already at the 255-register cap and ptxas found the room in the frame it
had.

#### The inversion: a null result is worth nothing unless the gate could have seen it

"The dose does not move" and "the measurement is not looking" are the same observation from the
outside, so the threshold was made wrong on purpose and the gate taken again. `step_lepton`
passing `urban_t_small(p.msc_tlimitmin * 1000)` - the branch unchanged, the threshold a thousand
times too large, so it fires on steps Geant4 samples normally - rebuilt as the ONE unit that
holds `run_step_lepton`, with the other sixteen objects in the archive byte-for-byte the ones the
table above was taken with, relinked, same seed:

| | dose | vs Geant4 | track-steps |
|---|---|---|---|
| threshold off (what shipped until now) | 425.84739 pGy | 1.25138 sigma | 26,064,061 |
| **threshold as Geant4 computes it** | **425.84670** | **1.25195** | **26,064,064** |
| threshold x1000 | **425.828** | **1.26693** | **26,071,487** |

These runs are deterministic to the last digit, so every digit of a difference is signal rather
than sample: x1000 moves the dose **27 times** as far as the branch itself does (-0.0184 pGy
against -0.00069) and changes the sixth significant figure of the line the gate prints, where the
real branch changes the seventh, and it moves **7,423** track-steps where the real branch moves
3. The gate can see this code. It reports a small number because the number is small.

The perturbation was then reverted, the same one unit rebuilt and example B1 relinked a third
time, and the gate came back to **425.847 pGy +/- 0.86768, -1.5383 pGy, 1.25195 sigma,
26,064,064 track-steps** - the "on" row above in every field, which is also the check that the
three runs differ by the threshold and by nothing else in the machine.

#### Both switches on, which is the state that ships

Every number this entry claims for the ion was re-taken with BOTH flags true, because the first
half of it was measured with the lepton branch still off and a claim taken in a configuration
that is not the shipped one is not a claim about the port. `test_ion_msc` 6.93e-16 / exactly 0 /
1.10 and 1.49 sigma / 3.43 sigma at chi2/bin 2.18; `test_step_hadron` 1.060e-13 over 900 steps
host against device, with `elastic_apply` at 5.799e-11 and `coulomb_fire` at 2.160e-14;
`test_ion_transport` alpha 16.0 steps, deuteron 14.0, He3 3.0, O16 1.0. Unchanged, as they have
to be - `kLeptonExtremeSmallStep` is read at exactly one place in the tree and it is inside
`step_lepton` - but "has to be" is the sentence this project does not accept on its own.



### V67: a square formed by two multiplications is not the square G4Pow forms

`G4NuclearShellModelDensity`'s constructor is `theRsquare = r0sq*G4Pow::GetInstance()->Z23(theA)`,
and `Z23(Z)` is `{ G4double x = Z13(Z); return x*x; }` - the square is formed FIRST and the product
with `r0sq` second. Transcribed as `r0sq * z13 * z13` it groups left to right, and the two differ by
one ulp of `theRsquare`.

One ulp, amplified: `GetRelativeDensity` is `G4Exp(-r^2/theRsquare)` and the exponent reaches -439
at 30 fm on He4, so a relative error of 1.1e-16 in `theRsquare` arrives as 4.8e-14 in the exponent
and 1.1e-13 in the density. Measured by making exactly that mistake and running
`tests/test_bic_nucleus.cu`: 1.137e-13 relative on He4 at 30 fm, against a bucket tolerance of
1e-15, on all three of `DensityRelative`, `DensityAbsolute` and `DensityDeriv`.

The lesson is the one docs/HADRONIC_PLAN.md section 8 gives for tables and this gives for
expressions: Geant4 does not evaluate the formula, it evaluates the code, and where an exponential
is downstream the association order is part of the answer. The port writes the square first.

### V68: two things about C12 that are not corners, because C12 is the target

`G4Fancy3DNucleus::ChoosePositions` has a branch for A = 12 alone - three alpha clusters on the
corners of an equilateral triangle (Bozek et al., Phys. Rev. C90, 064902) - and for a space-shielding
calculation carbon is the main line, not an edge case. Two things in it are recorded rather than
corrected, because porting Geant4 and improving it are different jobs.

**The cluster spread is a variance used as a standard deviation.** The code is

    const G4double Disp=0.552;        // 0.91^2*2/3 fermi^2
    R1=G4ThreeVector(G4RandGauss::shoot(0.,Disp), ... )*fermi + Corner1;

and the comment's `fermi^2` says the number is a dispersion - 0.91^2*2/3 = 0.5521, so it is. CLHEP's
second argument is a STANDARD DEVIATION (RandGauss.icc:25, `shoot()*stdDev + mean`). The sampled
displacement therefore has sigma = 0.552 fm where the cited parametrisation gives sqrt(0.552) =
0.743 fm, and the clusters come out 26% tighter than the paper's.

**The random stream depends on a process-wide latch.** CLHEP's `RandGauss::shoot` generates two
deviates per polar-method trial and caches the second in a `CLHEP_THREAD_LOCAL` static. Whether the
FIRST Gaussian of a C12 `Init` consumes two uniforms or none therefore depends on how many Gaussians
the process drew earlier, anywhere, in any model. The port keeps the latch in `Nucleus3DScratch`, so
a caller that reuses one scratch reproduces a sequence and a caller that does not gets a fresh one;
which of the two matches a given Geant4 run depends on what else that run did. It is not a bug in
either program, and it is why the C12 configuration is compared through the replay set rather than
by seeding.

### V69: a member assigned in an initialiser list and discarded two lines later

`G4KineticTrack`'s `(G4Nucleon*, position, 4momentum)` constructor - the one the binary cascade
builds every target nucleon with - initialises `theFermi3Momentum(nucleon->GetMomentum())` and then
its body is

    theFermi3Momentum.setE(0);
    Set4Momentum(a4Momentum);

where `Set4Momentum` ends with `theFermi3Momentum = G4LorentzVector(0)`. No other code in the class
writes the member. So `theFermi3Momentum` is zero for every kinetic track in every event, and
`Get4Momentum()` and `GetTrackingMomentum()` - which `SetTrackingMomentum` ties together through
`theTotal4Momentum = the4Momentum + theFermi3Momentum` - return the same four-vector up to the
`sqrt(m^2+p^2)` round trip.

The nucleon's Fermi motion is loaded into the track and thrown away. Both members are kept in the
port with the round trip, because BIC reads one in `Capture` and the other in
`G4RKPropagation::Transport` and the ulp between them is real, and because a release that removes
the `Set4Momentum` call would bring the Fermi momentum to life inside the propagator without
touching a line of the propagator. `tools/extract_bic_constants.pl` pins all three lines.

### V70: two sign errors in the nuclear fields, one of them with the wrong units

**The nucleon field table's tail returns a momentum where a field belongs.** `G4ProtonField` and
`G4NeutronField` precompute the local Fermi momentum at r = 0, 0.3, 0.6, ... out to
`2*GetOuterRadius()`, then push `fermiMom(2R)`, then `0`, then `0`. `GetField`'s out-of-range branch
is `if ((index+2) > size) return theFermiMomBuffer.back()` - it returns that trailing zero AS A
FIELD, not as a momentum to be turned into one. So beyond r = 2R + 0.6 fm a proton's field is exactly
0, where just inside it is `+theBarrier` (p_F has already fallen to zero), and the potential steps
DOWN by the Coulomb barrier - 5.1 MeV for lead - at the edge of the table. The two
`G4ThreeVector aPosition(0,0,...)` locals that are constructed and never read in those last two
blocks are what shows the zeros were meant to be evaluated.

**The pion fields build a nucleus mass by ADDING the binding energy.** All three of
`G4PionPlusField`, `G4PionMinusField` and `G4PionZeroField`, and `G4KM_OpticalEqRhs::SetFactor` as
well, compute

    nucleusMass = Z*proton_mass_c2 + (A-Z)*neutron_mass_c2 + bindingEnergy;

where a nuclear mass subtracts it - which is how `G4Fancy3DNucleus::GetMass()` writes the same
expression, twenty lines of Geant4 away. It enters only through `reducedMass = m_pi M/(m_pi + M)`,
so the error is diluted by m_pi/M: 2e-4 relative on carbon, 1.2e-5 on lead.

Both are reproduced as written and pinned by `tools/extract_bic_constants.pl`, which asserts the
`+` in all four files, so a release that fixes any of them fails the extractor rather than changing
four answers quietly.

### V71: an integrator whose position tolerance is a time compared to a length

`G4RKPropagation` steps the cascade in TIME and drives it with Geant4's magnetic-field integrator,
which steps in curve length. It gets away with it because `dydx[0..2] = c p/E` is a velocity, so the
driver's "curve length" parameter advances by a time while the position advances by a distance.
Every place the driver compares a length to its step therefore compares a length in mm to a time in
ns, and two of those comparisons are live:

  * `G4MagInt_Driver::OneGoodStep`'s position tolerance is `eps_pos = eps_rel_max * max(h,
    fMinimumStep)`, with `h` a time. At eps = 0.01 and a cascade step of 0.01 ns that is 1e-4 -
    read as mm, one hundred million fermi. The position error can never fail a trial step, and the
    adaptive step size is set entirely by the momentum error `|dp|^2/|p|^2/eps^2`.
  * `AccurateAdvance`'s `endPointDist >= hdid*(1.+perMillion)` compares a chord in mm against a step
    in ns with c = 299.79 mm/ns between them, so it is true on essentially every step and
    `fNoBadSteps` counts every step. The warning it guards is inside `#ifdef G4DEBUG_FIELD`, so
    nothing is printed and no value changes; the statistic is meaningless.

Neither changes an answer, and the port reproduces both - the first because it decides the step
sizes and therefore the trajectory, the second by omitting a counter nothing reads. Recorded because
a reader who assumes the position error is controlled will not understand the step sizes
`tests/test_bic_nucleus.cu` reproduces.

### V72: theCutOnP's three mass-number thresholds are compared against a mass in MeV

`G4BinaryCascade::Propagate` sets the momentum cut that decides which nucleons `Capture()` moves
into `theCapturedList`:

    theCutOnP = 90*MeV;
    if (the3DNucleus->GetMass() >  30) theCutOnP = 70*MeV;
    if (the3DNucleus->GetMass() >  60) theCutOnP = 50*MeV;
    if (the3DNucleus->GetMass() > 120) theCutOnP = 45*MeV;

The 30, 60 and 120 read as mass numbers and would give 90, 70, 50 and 45 MeV for A <= 30, A <= 60,
A <= 120 and heavier. `GetMass()` is `Z m_p + (A-Z) m_n - BE` in MeV, which is 939.6 for a single
neutron and 193,687 for Pb208. Every nucleus that exists is above 120, so the first three
assignments are dead and `theCutOnP` is always 45 MeV.

Reproduced as written in `bic_params.cuh`'s `cut_on_p(nucleus_mass_mev)`, with the mass-number
reading beside it as `cut_on_p_by_mass_number` and called by nothing. Both the thresholds and the
accessor are pinned by `tools/extract_bic_constants.pl`, because a release that corrects the
accessor would change the capture rate on every light target at once, and the extractor is the only
thing that can see it: `theCutOnP` is private, and the captured list is not exposed.

### V73: the port's own A13 is 1.5e-15 away from G4Pow's, and two hadronic packages inherit it

`src/data/g4pow.hh`'s `g4pow_a13_high` - the above-table branch - is written `exp(log(a)/3.0)` where
`G4Pow::A13` writes `G4Exp(G4Log(a)*onethird)` with `onethird = 1.0/3.0`. A division by three and a
multiplication by one third are not the same double. Measured: A13(4e32) is 73680629972.807739 here
and 73680629972.807632 in Geant4, 1.5e-15 relative.

Two of this package's quantities are built on it and cannot be compared at 1e-15 because of it:

  * `G4FermiMomentum::GetFermiMomentum` is `constofpmax * A13(density*A)`, and every density above
    about 1.9e2 in these units takes the above-table branch. `tests/test_bic_nucleus.cu`'s
    `FermiMomentum` bucket measures 3.8e-15 and is set at 1e-14.
  * the nucleon nuclear field is `-p_F^2/(2m) + barrier`, which squares that error and then nearly
    cancels the two terms. Measured at 2.4e-14 on Al27 at 5.4 fm; the bucket is set at 1e-13.

Not fixed here. `g4pow.hh` is shared with the EM port, whose tests sit near their own tolerances,
and changing the last bit of A13 under them from a hadronic branch is how an integration goes wrong.
It belongs in a package that owns `src/data/` and can run the whole suite after the change.

### V74: G4Fancy3DNucleus discards ReduceSum's verdict, and one nucleus in twenty thousand needs it

`ChooseFermiMomenta`'s retry loop is

    for (G4int ntry=0; ntry<1 ; ntry ++ )
    {
        ... sample a momentum for every nucleon ...
        if ( ReduceSum() ) break;
    }

- one iteration, so the `break` is not loop control and `ReduceSum`'s return value is read by
nothing. A configuration whose momenta cannot be balanced is used anyway, with a non-zero total
three-momentum and no message. The port returns it as `NucleusReport::reduce_sum_failed` instead,
because the only other way to find out is to add the momenta up.

How often it matters is measurable and small when the method is transcribed correctly: zero of
20,000 nuclei for each of C12, O16, Al27, Fe56 and Pb208. What the flag is worth is what it catches
when something else is wrong - see the commit that introduced it, where a one-index error in
`ReduceSum`'s Fermi-momentum budget produced 1,402 unbalanced nuclei out of 100,000 with up to
876 MeV of net momentum, and one O16 that never left the method at all.

### V75: a rotation to the lab frame that has already been applied

`G4BinaryLightIonReaction::ApplyYourself` ends by building

    G4LorentzRotation toZ;
    toZ.rotateZ(-1*mom.phi());
    toZ.rotateY(-1*mom.theta());
    G4LorentzRotation toLab(toZ.inverse());

and applying `toLab` to every secondary. Those are the same five lines as
`G4HadProjectile::InitialiseLocal` (G4HadProjectile.cc:72-76) - and that method has already run
them: it stores `theMom.set(0, 0, sqrt(T(T+2m)), T+m)`, a four-momentum along +z, and keeps the
inverse in `toLabFrame` for `G4HadronicProcess::FillResult` to apply at the end. So
`aTrack.Get4Momentum()` has `phi() == 0` and `theta() == 0`, both rotations are the identity, and so
is `toLab`. The swapped case is the same: `toBreit * G4LorentzVector(m1, (0,0,0))` boosts an at-rest
nucleus along the original projectile's velocity, which is +z.

The port does not carry the block. It is recorded because the only way to know it is a no-op is to
read `G4HadProjectile`, and because a caller that hands the model a projectile NOT along +z - which
nothing in 11.1.1 does - would need it. `G4BinaryCascade::ApplyYourself` has the same block, with
the same reasoning.

### V76: G4PhotonEvaporation creates 511 keV out of nothing, once per conversion electron

**This entry belongs to P3, not to P9, and is filed here because P9's oracle is what found it.**

`G4PhotonEvaporation::GenerateGamma` computes the emitting system's invariant mass as
`ecm = lv.mag()` and then, for an internal-conversion transition,

    if (!isGamma) { ecm += (electron_mass_c2 - bond_energy); }

with `bond_energy` a local initialised to 0 and never assigned - it is the atomic binding energy
that was meant to pay for the electron's rest mass. The emission then splits `ecm` between the
electron and the residual, so the whole four-momentum is rescaled by `(1 + m_e/M)` and 511 keV
appears that was not there before.

Measured on `preco::deexcite` alone, one compound nucleus in, its product list out, 5,000 events:

  compound          conversion electrons/event   product list heavy by   n_e * m_e
  d + Pb208                              0.26          +0.13286 MeV      0.13286 MeV
  C12 + Pb208                            0.4198        +0.214539 MeV     0.214517 MeV
  d + Al27                               0             -1.1e-13 MeV      0
  alpha + C12                            0             -4.5e-14 MeV      0

It is Geant4 11.1.1's arithmetic and this port reproduces it; `tests/test_bic_apply.cu` pays for it on
both sides and then asserts the balance exactly. What it costs a user is a dose: a calculation that
sums secondary kinetic energies plus rest masses against the primary's will find more energy out
than in, by 511 keV per conversion electron, and for a heavy compound that is a quarter of an MeV
per event on top of totals of a few hundred MeV. Small, real, and in a direction that cannot be
blamed on sampling.

The check that catches it costs nothing and is not in this port anywhere else: after a
de-excitation, `sum(product four-momenta) - n_conversion_electrons * m_e` must equal the fragment
that went in. `tests/test_bic_apply.cu` asserts it at 2e-10 MeV per event on a 205 GeV total.

### V77: the ceiling came off, and the bottom of the table was wrong too

V64's fix, and what building it found. The e+- dE/dx, range and inverse-range table is now
`G4LossTableBuilder`'s on `G4EmParameters`' own grid - `MinKinEnergy` 100 eV to `MaxKinEnergy`
100 TeV, `NumberOfBinsPerDecade` 7, so 85 nodes - cubic-splined with
`G4PhysicsVector::ComputeSecDerivative1`'s conditions and integrated with the seed `2*E0/dedx0`
and 100 midpoint sub-steps per bin, which is the same transcription `em/hadron_range.cuh` has
used since docs/PORTED.md 4.3. The e+- table was the last place in the port that 4.3's rule had
not been applied. Above the top node there is no Geant4 answer to reproduce, so `above_table`
refuses the track by name into the refusal ledger, energy and all, rather than clamping it.

**Two defects were sitting under the old 1 keV floor, and neither could be seen from above it.**

*The low-energy extrapolation was a constant.* `G4MollerBhabhaModel::ComputeDEDXPerVolume` ends

    if (kineticEnergy < th) {
      x = kineticEnergy/th;
      if(x > 0.25) { dedx /= sqrt(x); }
      else         { dedx *= 1.4*sqrt(x)/(0.1 + x); }
    }

with `th = 0.25*sqrt(Zeff)` keV. The port's second branch read `dedx *= 1/sqrt(0.25)`, which is
the first branch frozen at the breakpoint. The two agree there exactly - 1.4*0.5/0.35 = 2.0 -
which is why the substitution looks harmless, and they diverge below it: the correct form falls
off as sqrt(x)/0.1 towards zero while a constant 2 keeps the full stopping power. `x <= 0.25`
means `E <= 0.0625*sqrt(Zeff)` keV, 168 eV in water and 306 eV in lead, so no electron is ever
TRACKED there - `G4EmParameters::LowestElectronEnergy` is 1 keV - and nothing in the port or in
`ref/oracle/electron_tables.csv` reached it. What is there is the first two or three nodes of
the range table, and the range at 1 keV is the integral from 100 eV upwards. Restored, it is
worth 9.6% of the dE/dx and 10.6% of the range in the bottom decade, and it is still 3.8e-8 of
the range at 10-100 MeV - four decades above where it was introduced, with nothing at that
energy to point at.

*Every positron read the electron's range.* The table was built once with `is_positron = false`
and had no species dimension. The restricted collision stopping power is Moller for one and
Bhabha for the other: in `CustomSiGe` the two ranges differ by 32.4% at 100 eV, 2.3% at
1-10 MeV and 4.3e-4 at 1-10 GeV. The ceiling was hiding it because every test that could have
seen it compared a MODEL - `collision_dedx(mat, E, is_positron)` takes the flag and was right
all along - and nothing compared the TABLE, which is the same sentence V64 ends with.

**The gamma gate never saw any of it, and could not.** B1's gate is a 6 MeV photon beam at
2,000,000 events; its secondaries are Compton electrons and pair electrons of a few MeV, which
is the middle of the table where the old grid was dense and correct. The 1 keV floor is below
the tracking cut, the 100 MeV ceiling is sixteen times the beam energy, and the positron rows
only matter where the pair channel is open at all. A gate is a measurement at one point; it
cannot be a measurement of a function.

**What a test of the TABLE costs and what it finds.** `tests/test_electron_hi.cu` is 6,174
points of `G4EmCalculator::GetDEDX`/`GetRange` in seven materials from 1 keV to 100 TeV, banded
by decade and by species, plus 1,190 nodes against a 17-digit dump from 100 eV
(`ref/dump/dump_electron_hi.cc`) - which is where both defects above appeared. Above 100 eV the
worst residual is 6.8e-5, which is the oracle's own `%.9g`; from 0.1 MeV up it is 3e-9; at the
nodes above 1 MeV it is exactly zero, because on the same grid the port samples the same models
at the same points.

---

### V78: the lambda tables are half of a pair, and this port has neither half

Refused by name rather than approximated, and recorded because the size of it is measured and
not small.

`G4VEnergyLossProcess` does not evaluate a discrete cross section during transport and does not
simply read one out of a table either. `PostStepGetPhysicalInteractionLength` uses the INTEGRAL
APPROACH: it caches `preStepLambda` from `ComputeLambdaForScaledEnergy` at up to `1/lambdaFactor`
= 1/0.8 of the current energy (the peak structure is per process - `G4eIonisation` is
`fEmOnePeak`, `G4eBremsstrahlung` is `fEmTwoPeaks`), draws the interaction length from that, and
then `PostStepDoIt` REJECTS the interaction with probability `1 - lambda(E_post)/preStepLambda`.
The lambda vector itself is built on `G4EmParameters`' grid from `MinPrimaryEnergy` - `2*cut`
for e-, `cut` for e+ - with `startFromNull` forcing the first node to zero.

This port draws from the model's cross section at the pre-step energy with no rejection. That is
neither of Geant4's two objects, and tabulating the vector without the rejection would not be
closer: the table alone moves the rate the wrong way near threshold, where a 7-per-decade table
of a function that rises from zero across one coarse bin disagrees with the model by tens of per
cent. `tests/test_electron_hi.cu` prints that difference every run - 47.1% for the e- delta-ray
cross section at 1 MeV in the lead-bearing material, 66.7% for e+ at 1 keV in air - as a
measured refusal rather than a failure, because it is Geant4's answer and not an error in
either. Closing it means transcribing `ComputeLambdaForScaledEnergy`, the cached
`preStepLambda`, the two peak shapes and `PostStepDoIt`'s rejection together, and it belongs to
whoever next needs the discrete rates to be Geant4's rather than the models'.

---

### V79: one msc model, three cuts, and the port uses a different one on each path

`G4WentzelVIModel` asks `G4WentzelOKandVIxSection` for a cross section in three places and hands
it a different cut each time. This is not a Geant4 defect and it is very easy to port as one.

    G4VEmModel::Value, through G4LossTableBuilder::BuildTableForModel          cut = 0
      -> xSectionTable, which is what GetTransportMeanFreePath reads
    G4WentzelVIModel::ComputeTransportXSectionPerVolume                        cut = electron
      -> xtsec, the single-scattering rate the sampler draws its intervals from    production cut
    G4WentzelVIModel::ComputeCrossSectionPerAtom, called by the process         cut = the
                                                                                   process's

The first is the point: `G4VEmModel::Value` is `pFactor * E*E * CrossSectionPerVolume(mat, p, E,
0.0, DBL_MAX)` and that `0.0` reaches `ComputeCrossSectionPerAtom` as its `cutEnergy`, so
`SetupTarget(Z, 0.0)` leaves `cosTetMaxElec` at 1 and the electron-scattering channel is CLOSED
in the tabulated transport cross section - while it is open in `xtsec`. Using one for the other
is worth up to 8.75% of the transport cross section at the top of the range (measured,
A-150 tissue, e- at 51.8 TeV) and lengthens every step limit.

The e+- path added by P14c reads the table for `lambda_eff` and the production cut for `xtsec`,
which is Geant4's arrangement, and `tests/test_electron_hi.cu` compares both columns against
`ref/oracle/electron_hi_msc.csv`. (That path is written, tested and NOT dispatched: V83.) **The HADRON path in `step_hadron` still calls
`wentzel_lambda` directly with the production cut, which is neither**, and it is left exactly as
it is: the proton and alpha B1 rows agree with Geant4 to 0.25% today, so changing the
quantity that sets a proton's step length is a package with its own before-and-after, not a
side effect of the electron one. A proton is 1836 times heavier than an electron and its
`cosTetMaxElec` sits far closer to 1, so the term this leaves out is much smaller there than
the 8.75% above - but "much smaller" is an argument, not a measurement, and the measurement is
what is owed.

---

### V80: the two lepton msc tables in this port are now built to different rules

`em::UrbanTable` is 240 log-spaced points from 1 keV to 100 MeV, interpolated LINEARLY.
Geant4's is `G4LossTableBuilder::BuildTableForModel`'s - the model's own energy window at
`NumberOfBinsPerDecade` 7, cubic-splined. The MODEL underneath agrees to 6.7e-16 over 41,952
points (`tests/test_urban_general.cu`), so anywhere between two nodes the two tables differ by
their interpolation alone and by nothing physical.

Two things make this worth an entry rather than a footnote. The first is that the e+- WentzelVI
table P14c added IS on `BuildTableForModel`'s grid, with an oracle at its own nodes
(`ref/oracle/electron_hi_msc.csv`, 602 points at 5.1e-16) - so the port now holds one lepton msc
table built to Geant4's rule and one built to its own, and a reader comparing an msc number has
to know which. The second is how easily this hides: at exactly 100 MeV, where
`tests/test_electron_hi.cu` compares the Urban table against `G4EmCalculator` to confirm which
model owns the boundary energy, the two agree to better than 5e-7 in all seven rows - because
100 MeV is a node on BOTH grids (`1e-4 * 10^(84/7)` is exactly 100, and it is the port's
`e_max`). An agreement measured only at a shared node says nothing about the twelve points
between them.

The ceiling itself is not a V64: `em::kMscEnergyLimit()` is 100 MeV and `step_lepton` takes the
WentzelVI branch strictly above it, so the Urban table's last node is exactly where Geant4's
Urban model stops and `lambda_at`'s clamp above it is unreachable from the transport. Whoever
needs the sub-per-cent agreement of an Urban step should put that table on
`BuildTableForModel`'s grid; until then an Urban msc number in this port carries an
interpolation error that is not in the model.

---

### V81: ptxas died again, in the same place, for the same reason, on a different branch

    Internal error
    nvcc error   : 'ptxas' died with status 0xC0000005 (ACCESS_VIOLATION)

V55's failure and V63's, now on the lepton. Adding `G4WentzelVIModel` to `step_lepton` for e+-
above 100 MeV made `src/host/transport_run.cu` uncompilable: twenty `__global__` instantiations
in one translation unit, each inlining the whole of whatever its stepper reaches, and this put
four more copies of `wentzel_setup`, `wv_transport_xs`, `wv_step_limit`, `wv_geom_path`,
`wv_true_path` and `wv_sample_scattering` into it - one per lepton kernel.

V55's diagnostic worked unchanged and is worth repeating for that reason: the one-kernel
reproducer, ten lines that instantiate `run_step_lepton<double, true/false, StepTap<double>>`
and nothing else, compiles in two minutes against a TU that takes the better part of an hour,
and it compiled FINE - 255 registers, 3,088 byte frame, 96/56 spill. A kernel that compiles
alone and not in company is a translation-unit problem, which says the answer is to stop
inlining rather than to simplify the physics.

The fix is four `__device__ __noinline__` wrappers in `em/wentzel_msc.cuh` -
`wv_lepton_limit`, `wv_lepton_geom`, `wv_lepton_true`, `wv_lepton_scatter` - called only by
`step_lepton`. It is also the right answer for the hot path, which is why it is not a
workaround: `G4EmParameters::MscEnergyLimit()` is 100 MeV, so the branch is dead for every
electron B1's 6 MeV gate makes and for every delta ray any hadron in this port makes, and what
was being inlined into every step of every lepton is a branch that almost never runs. That is
V55's argument for `had::elastic_apply` in the same shape.

**The wrappers are lepton-only and the inline functions are untouched**, deliberately:
`step_hadron` calls those directly, `__noinline__` moves a floating-point contraction boundary,
and P14b's byte-identical proton and alpha kernels are a claim this package must not spend.

**It did not work.** The same message, from the same file, after a 34-minute build. V63's
nine-build table said it would not and says why no further arrangement was tried: the thing
that dies is the translation unit, it is a cliff and not a slope, and the lever with headroom
is P8e's split. The wrappers are KEPT because they are right for the hot path independently of
this, and the branch is switched off at compile time instead: `em::kWentzelLeptonMscWired`,
docs/RISK.md V83.

---

### V82: the continuous loss was split in two and one half went nowhere

Found by `tests/test_lepton_transport.cu`, the device energy balance V64's last paragraph asked
for, on its first run.

`step_lepton` deposited `loss * col/(col + rad)` - the collision share of the continuous loss -
where `col` is `G4MollerBhabhaModel`'s restricted collision stopping power and `rad` is the
restricted radiative one. The other share was neither deposited nor handed to a secondary. It
simply left the arithmetic.

`G4VEnergyLossProcess::AlongStepDoIt` has no such split. It ends

    eloss = std::max(eloss, 0.0);
    fParticleChange.SetProposedKineticEnergy(finalT);
    fParticleChange.ProposeLocalEnergyDeposit(eloss);

(G4VEnergyLossProcess.cc:922-925) and the only things subtracted from `eloss` above those lines
are atomic de-excitation, which option0 has off, and `subcutProducer`, which is null unless a
region asks for one. The reason there is no split is that the RESTRICTED radiative term is by
construction the part of the bremsstrahlung spectrum below the gamma production cut: the photon
that would have carried it is not produced, so its energy is local. Everything above the cut
leaves as an explicit photon from the discrete branch and was never in `eloss` at all.

What it was worth, per track, as the balance reports it: 5.3e-5 of a 1 MeV electron in water,
3.2e-5 at 50 MeV, 2.1e-6 at 1 GeV and 2.1e-7 at 10 GeV. It grows towards low energy because the
gamma cut is a larger fraction of a smaller electron's bremsstrahlung spectrum. With the split
removed every track closes to floating point: worst residual 1.8e-15 of the primary energy over
2,048 tracks at four energies and both species.

**Two things about this are worth more than the number.** The first is that it is the shape of
V64 again at a thousandth of the size - an energy that no counter held - and the same test
catches both: with the old table in place the balance fails by 90% at 1 GeV. The second is what
the balance had to be told before it could close at all: a positron's input side is
`E0 + 2 m_e c^2`, because `G4eplusAnnihilation`'s two 511 keV photons are rest mass and not
anything the track carried. Written down, that is a statement about the physics; left out, it
reads as a 102% energy gain at 1 MeV.

---

### V83: the second model is written, tested and switched off, and the switch is a compiler

`em::kWentzelLeptonMscWired` is `false`. `G4EmStandardPhysics::ConstructProcess` gives e+- a
`G4UrbanMscModel` below `MscEnergyLimit()` = 100 MeV and a `G4WentzelVIModel` above it; this
port dispatches Urban at every energy, and it is V63's situation on the other stepper.

What IS done, and none of it is waiting on physics: `G4WentzelVIModel`'s electron path is
transcribed; `G4VMscModel::xSectionTable` for it is built on `BuildTableForModel`'s own grid -
43 nodes from 100 MeV to 100 TeV holding `E^2 * CrossSectionPerVolume(..., 0.0, DBL_MAX)` - and
compared against `ref/oracle/electron_hi_msc.csv` at its nodes to **5.1e-16** over 602 points,
with the interpolation between them at 0.20%; the dispatch is written in `step_lepton` behind
`if constexpr`; and it runs on the device in `tests/test_lepton_transport.cu`, where a 1 GeV
electron takes 9,585 of its 15,933 steps above the boundary and the energy balance closes to
1.8e-15 with it on. The boundary itself is checked against Geant4's own answer at 100 MeV
(`tests/test_electron_hi.cu`: `G4RegionModels::SelectIndex` tests `e <= lowKineticEnergy[idx]`,
so 100 MeV belongs to Urban and the branch is a strict `>`).

What is not done is `src/host/transport_run.cu` compiling with it on:

    Internal error
    nvcc error   : 'ptxas' died with status 0xC0000005 (ACCESS_VIOLATION)

V55's remedy - `__noinline__` on the branch that almost never runs - is applied and kept
(`wv_lepton_limit`, `wv_lepton_geom`, `wv_lepton_true`, `wv_lepton_scatter`, V81) because it is
right for the hot path either way; it did not move the wall. V63's nine-build table is why no
further arrangement was tried: the failure is the translation unit and not the kernel, it is a
cliff rather than a slope, and "the only lever left with real headroom is to stop asking one
translation unit to hold twenty kernels" - which is P8e's package. `if constexpr` and not a
runtime `false`, because a runtime false leaves every arm in the TU.

**WHAT THE SUBSTITUTION COSTS, STATED RATHER THAN ARGUED.** Above 100 MeV a lepton is stepped
by Urban, and `em::UrbanTable` ends at 100 MeV - so its transport mean free path above that is
the 100 MeV one, clamped. That is a clamp of V64's family and it is named here because it would
otherwise be found the same way V64 was. It differs from V64 in what it touches: the msc STEP
LENGTH and the deflection of an electron above 100 MeV, not its energy loss. The dE/dx, range
and inverse-range tables this package rebuilt are read correctly at every energy with the
switch in either position, which is why the 1 GeV B1 row moves by what it moves by
(docs/B1_SWEEP.md) with the switch off. Extending the Urban table instead is not a smaller
change than it looks: its 240 nodes are one log grid from 1 keV, so moving its ceiling moves
every node below it, and B1's 6 MeV gamma gate reads those nodes.

---

### V84: an electron fired through a vacuum is annihilated on its first step

Found by giving `ref/proton/proton_depth.cc` a particle name and firing an electron into it. The
harness's world is `G4_Galactic` and the gun sits 1 mm outside the phantom, so the first step of
every primary is a 1 mm step in a vacuum. A proton crosses it and deposits 100.0000% of the beam
energy in the water behind it. A 20 MeV electron deposits **0.0000%**, and a 20 MeV positron
deposits 1.03% - which is its two annihilation photons and nothing else, emitted where the
positron died.

**The mechanism, and it is one line.** `step_lepton`:

    real_t e_after = s.range_table->energy_from_range(mat, is_positron, range - step_len);
    // Second guard: the step must strictly reduce the energy...
    if (e_after >= p.ekin) { e_after = real_t(0); }

`G4_Galactic` is hydrogen at 1e-25 g/cm3, so a 20 MeV electron's range in it is 6.61e26 mm
(measured, from the port's own table, which inverts correctly there - `energy_from_range` of
that range returns 20.0027 MeV and of half of it 9.62 MeV). `range - step_len` for a 1 mm step
is `6.61e26 - 1`, which in double IS `6.61e26`: the subtraction is below the last bit. So
`e_after == p.ekin`, the guard fires, and the electron's entire kinetic energy is taken as the
loss of a 1 mm vacuum step. It is deposited in the world volume, which nothing scores, and the
track is dead before it reaches the phantom.

Three things were ruled out with measurements rather than reasoning. The range table is not the
problem: it is well-conditioned in the vacuum and inverts to 4 significant figures at every
energy tried. The Urban msc machinery is not the problem: with `lambda0 = 3.15e28 mm` and
`range = 6.61e26 mm` it returns `z_step = 1.32e26`, a geometric step cut to the 1 mm boundary,
and a true path of exactly 1 mm - no NaN anywhere. And it is not new: the guard predates P14c,
and the OLD table's vacuum range was the same order of magnitude, so the same subtraction was
the same no-op.

**Why nothing had seen it.** Example B1's world is `G4_AIR`, where a 20 MeV electron's range is
about 1.3e5 mm and `range - step_len` is perfectly representable; the proton and alpha depth-dose
gates cross the same vacuum but `step_hadron` reads a different table with a different guard; and
no other harness in this port fires a lepton into a vacuum. The defect needs a density ratio of
about 1e16 between the range and the step, which a vacuum gives and no real material does.

**What the right structure is, and it is not a bigger epsilon.**
`G4VEnergyLossProcess::AlongStepDoIt` computes the loss LINEARLY first and only inverts the range
when the linear answer is large (G4VEnergyLossProcess.cc:825-834):

    eloss = length*GetDEDXForScaledEnergy(preStepScaledEnergy, ...);
    ...
    if(eloss > linLossLimit*preStepKinEnergy) {
      ...
      eloss = preStepKinEnergy - ScaledKinEnergyForLoss(x)/massRatio;
    }

with `linLossLimit` = 0.01. This port always inverts and guards. In a vacuum the linear form
gives `1e-26 MeV/mm * 1 mm`, which is the right answer; the inversion cannot, because the
information is not in the difference of two doubles that equal each other.

**Refused by P14c rather than fixed, and the reason is the blast radius.** Putting
`linLossLimit` in changes the energy loss of EVERY electron step whose loss is under 1% of its
energy, which is most of them, in every run - the 6 MeV gamma gate included. That is a package
with its own before-and-after, not the last commit of this one. What P14c owed here was the
electron depth-dose comparison (`ref/oracle/run.bat` now produces
`ref/oracle/electron_depth.csv`, 100,000 events of 1 GeV in 4 m of water, peak at 680 mm =
1.88 X0); the Geant4 half is produced and checked, and the PORT half is not quoted, because
what it would measure is this.

### V85: two lines of the meson table write their weights one index high, and the eta loses it

`G4VLongitudinalStringDecay::SetMinMasses` builds the tables that decide what a string's LAST
splitting can decay into. The d-dbar row of the meson table is written as

    Meson[0][0][2] = 221; MesonWeight[0][0][3] = pspin*(1-mix0-mix1);   // Eta
    Meson[0][0][3] = 331; MesonWeight[0][0][4] = pspin*mix1;            // Eta'
    Meson[0][0][4] = 223; MesonWeight[0][0][4] = (1-pspin)*mix1;        // omega

- both weights land one slot past their code, and the omega then overwrites slot 4. The u-ubar
row eleven lines below (`Meson[1][1][*]`) is the same five lines with consistent indices. So
`MesonWeight[0][0][2]`, the eta's, is 0; `[3]` holds 0.125, which is the eta's number sitting on
the eta'; `[4]` holds the omega's 0.25; and the row sums to 0.875 where u-ubar's sums to 1.0.

The consequence is exact rather than statistical. `Quark_AntiQuark_lastSplitting` enumerates
every (left, right) meson pair the row allows and `SampleState` picks one with probability
proportional to its weight, so a d-quark string that produces a d-dbar pair from the vacuum
enumerates the eta and samples it with weight zero: that channel yields no eta at all, and yields
the eta' at the eta's rate. It is reachable from any ordinary proton or pion beam - it is the
last splitting of a light q-qbar string, which is most strings.

Two more index errors in the same function. `Meson[3][3][0] *= ProbEta_c/pspin_meson[2]` and its
three neighbours multiply the PDG CODE by a probability ratio instead of the weight:
441*(0.1/0.3) truncates to the integer 147 and 443*(0.9/0.7) to 569, and the Baryon table's final
null sweep is not applied to Meson, so 147 survives in the table as a particle code with no
particle behind it. Those four are unreachable while `Prob_QQbar[3] = Prob_QQbar[4] = 0`, which
is every QBBC run, because the c-cbar and b-bbar production probabilities of the LAST splitting
are zero.

The port transcribes all of it with Geant4's indices (`src/physics/hadronic/ftf/lund_tables.cuh`),
because the alternative is a port that produces etas Geant4 does not. `tests/test_ftf_params.cu`
asserts the two row sums by name - 0.875 for d-dbar, 1.0 for u-ubar - and asserts
`MesonWeight[0][0][2] == 0`, so a future "fix" to the indices fails the test rather than silently
changing every small-string decay. `ref/oracle/ftf_lund_tables.csv` carries all 1,510 entries.
If Geant4 ever fixes this, those three assertions are the ones to delete, and the oracle will say
so first.

### V86: FTF's elastic cross section on a neutron is a proton's inelastic subtracted

`G4FTFParameters::G4FTFParameters` builds the hadron-nucleon cross sections that drive the whole
impact-parameter sampling of an FTF interaction. For the "interaction on a neutron" pair it asks
`G4ComponentGGHadronNucleusXsc` for (Z, A) = (0, 1). That component's A == 1 branch computes the
proton and neutron cross sections separately and then combines them: `fTotalXsc` is the
Z-weighted sum, which with Z = 0 is correctly the hadron-NEUTRON total, but `fInelasticXsc` is
assigned `hpInXsc`, the hadron-PROTON inelastic, computed before the Z = 0 weighting that removed
it from the total. So

    Xelastic(on a neutron) = sigma_tot(h n) - sigma_inel(h p)

mixing two different targets. It propagates into `FTFXelastic`, the elastic slope, `Gamma0` and
the average Pt^2 of elastic scattering, i.e. into every nucleus with neutrons in it - which is
every nucleus this port is for. The nucleus-projectile arm is worse: its "PN" pair is
`GetTotalIsotopeCrossSection(Neutron, ..., 0, 1)`, a NEUTRON projectile on a neutron target, so
the mixed term of a nucleus-nucleus average is built from n+n rather than from p+n, with its
elastic part then n+n total minus n+p inelastic.

This is Geant4 11.1.1's arithmetic and the port reproduces it exactly - 7,020 cross-section
points and 5,616 geometry points at 4e-16 in `tests/test_ftf_params.cu`. It is recorded here
because it is the kind of difference a reader of the port will find and take for a transcription
error, and because if Geant4 changes it the port's numbers move with no other warning.

### V87: the string tension is 1e30, and nothing in QBBC reads it

`G4VLongitudinalStringDecay`'s base constructor assigns `Kappa = 1.0*GeV/fermi` directly.
`G4LundStringFragmentation`'s constructor then calls `SetStringTensionParameter(1.*GeV/fermi)`,
and that setter multiplies its argument by `GeV/fermi` again. The result is
1e30 MeV^2/mm^2 rather than 1e15 MeV/mm; `ref/oracle/ftf_lund_params.csv` says `Kappa,1e+30`.

It has no symptom because its only reader is `CalculateHadronTimePosition`, which only
`G4QGSMFragmentation` calls, and QGS is not in QBBC. The port carries 1e30 and
`tests/test_ftf_params.cu` asserts it, so that a port that "corrects" the units fails rather than
diverges from Geant4 by a factor of 1e15 in a quantity nobody looks at. If QGS is ever ported,
this is the first number to check, and the second is whether Geant4 has fixed it by then.

### V88: 20,000 events is the oracle's limit, and four port seeds agreed on its fluctuation

The Lund fragmentation's statistical half (`ref/oracle/ftf_fragstat*.csv`) runs 20,000 strings per
case on each side and compares species counts and multiplicities. At that size the port's mean
multiplicity came out ABOVE Geant4's in 12 of the 15 fragmenting cases, combining across cases to
+3.3 sigma, and it stayed above when the port's Philox seed was changed three more times: +3.3,
+1.1, +2.7, +1.8. Four independent port samples, all high. That reads as a real excess of about
0.2% in hadrons per string, and it is not one.

The reason the four agree is that they share one oracle sample. Each z is
`(m_port - m_oracle)/sigma`, and a single low fluctuation of the 20,000-event ORACLE biases every
comparison the same way no matter how many times the port is re-sampled. Re-running this dump's
fragstat block at N = 200,000 against the port at the same 200,000 moved the combination to
**-1.8**, with 8 of 15 cases negative, every case within 0.16%, and the largest single case at
1.8 sigma. The 20,000-event oracle was the limit, not the port.

Two lessons, both general. Re-seeding the PORT is not an independent measurement when the
reference is a fixed sample - it measures the port's own variance and says nothing about the
reference's. And a statistical oracle needs its own size recorded and raised, not just the port's:
`n_events` is now a column of `ftf_fragstat.csv` and `tests/test_ftf_lund.cu` runs the port at
whatever the oracle ran at, so that enlarging the campaign is a one-line change in the dump
instead of two constants that can go out of step. The committed size stays 20,000 because 200,000
takes this dump from about one minute to seven, and the 5-sigma per-species tolerance absorbs
the difference.

### V89: the test asserted a number that belonged to another package's dump

`ref/oracle/ftf_hadrons.csv` is the whole initialised `G4ParticleTable`, which includes every
nucleus `G4IonTable` has been asked to create SO FAR IN THAT PROCESS. `ref/dump/g4dump.cc` runs
the registered dumps in one process, in link order, and the ftf dump runs after the decay,
de-excitation and elastic dumps, each of which creates ions of its own. The same g4dump.exe wrote
31 nucleus rows on one run and 1,783 on another, with the 485 hadrons identical to the last bit
both times.

`tools/ftf_hadrons.pl` and `tests/test_ftf_params.cu` both asserted that exactly 31 rows were
dropped, and the second run of the oracle turned that into a FAIL of a passing port: the header
`src/data/ftf_hadrons.hh` regenerates byte-identically from either file. Both assertions are
gone, replaced by a printed count; what is still asserted is the part that is a property of
Geant4 - the 485 non-ion particles compared in both directions, the 12 quarks, the 50 diquarks,
and every mass, width, charge and subtype. An assertion on a number that another package's dump
can change is a test that fails for the wrong package, and this one would have failed for P3.

### V90: the string fragmentation's energy balance is broken on purpose, and put back by a loop

`G4ExcitedStringDecay::FragmentStrings` does something to the Lund fragmentation's output that a
reader of `G4LundStringFragmentation` alone would not expect: it REPLACES THE MASS of every
short-lived product. A rho+ leaves the fragmentation at the PDG pole mass, and FragmentStrings
redraws it from a Breit-Wigner between `G4SampleResonance::GetMinimumMass(def) + 10 MeV` and
`pole + 5*width`, keeps the three-momentum and recomputes the energy. The resonances are most of
what a string makes - rho, omega, K*, Delta - so after this step the hadrons of a string no
longer carry the string's four-momentum.

`EnergyAndMomentumCorrector` is what puts it back, and it is not a safety net: it runs on
essentially every event. The trigger is `|(E_hadrons - E_string)/E_hadrons| > perMillion` for any
one string, and the loop then scales every hadron's three-momentum by a common factor in the
c.m.s. of the strings, recomputing energies from the masses the hadrons now have, until the
energies sum to the collision mass within 1e-5 or 500 iterations have passed.

Measured, 20,000 events on each of eight string-vector cases: 99% of Geant4's events end with a
relative energy error in the 1e-6 decade - just inside the corrector's own limit - and a few
hundred at 1e-12, which are the events with no short-lived product to redraw. That histogram is
now an oracle file (`ftf_stringstat_balance.csv`) and the test compares it bin by bin, because it
is the only thing that distinguishes a port that ran the correction from one that did not: with
the `perMillion` trigger raised to 1e-2 the port's events move from the -6 bin to the -3 bin,
9,070 of them in one case, which is 95 sigma. Every exact per-hadron comparison in the same test
would still have passed at the FIRST string, and failed later for reasons that look like noise.

Three consequences worth keeping in view. A caller cannot assume the hadrons of one string
balance that string - only the TOTAL balances, and only to 1e-5. `G4TheoFSGenerator` and P6's
`Propagate` receive that 1e-5, so an energy-balance assertion anywhere downstream has to be
looser than the corrector's own limit. And the corrector boosts by `TotalCollisionMom`, not by
the hadrons' own `SumMom` - the alternative is written and commented out in the source one line
above - which matters because the two differ by exactly the imbalance being corrected: putting
the commented-out line back changes a corrected momentum by 76 relative units on the oracle's
`far-off` case while leaving the whole-event answer within 3e-12, so it is a difference only the
corrector's own oracle can see.

### V91: the np cross-section grid is stretched by 1% and the pp grid beside it is not

`G4XNNElasticLowE`'s constructor builds two `G4PhysicsLogVector`s from the same `_eMax` and two
DIFFERENT `_eMin`s, because it reassigns the member between them:

    _eMin = _eMinTable * GeV;                                   // 1.8964808 GeV
    _eMax = G4Exp(G4Log(_eMinTable) + tableSize*_eStepLog)*GeV;  // 1.8964808*e^1.01 GeV
    G4PhysicsVector* pp = new G4PhysicsLogVector(_eMin,_eMax,tableSize);
    _eMin = G4Exp(G4Log(_eMinTable)-_eStepLog)*GeV;             // one log step lower
    G4PhysicsVector* np = new G4PhysicsLogVector(_eMin,_eMax,tableSize);

A `G4PhysicsLogVector(Emin, Emax, 101)` spreads 102 nodes evenly in log across
`log(Emax/Emin)`. For pp that span is 1.01 and the step comes out at exactly 1.01/101 = 0.01,
which is `_eStepLog` - the grid the 101 tabulated values were measured on. For np the span is
1.02 and the step is 1.02/101 = 0.0100990. The np table's 101 values are therefore laid down 1%
too far apart: value 100, which the data places at `_eMinTable*e^0.99`, is read at
`_eMinTable*e^0.9999`.

`G4XnpElasticLowE` and `G4XnpTotalLowE` do the same thing - their first `_eMin` assignment is
dead, overwritten one line later by the shifted one - so every np cross section the binary
cascade evaluates comes off the stretched grid and every pp one off the correct grid. The two
tables are the same physics measured the same way; only the placement differs.

How much it is worth, measured: putting the np vector on the pp grid moves the np elastic cross
section by up to **4.93 relative** (`tests/test_bic_imr.cu`, pn at sqrt(s) = 1897.6 MeV, where
the table is falling from 1500 mb to 248 mb over one log step and a 1% shift in energy is a
factor of five in sigma). Near the minimum, at 2.5 GeV, the same shift is worth 0.3%.

Reproduced as written in `src/physics/hadronic/bic/im_r/xsec_nn.cuh`, with the two `Emin`
helpers named `lowe_emin_pp` and `lowe_emin_np` so that the asymmetry is visible in the call
rather than hidden in a constant, and `tools/extract_bic_imr.pl` pins `_eMinTable` and
`_eStepLog` in all three source files.

### V92: four cross-section tables have a 102nd node holding zero, and one energy grid is short by one

`G4PhysicsLogVector(Emin, Emax, Nbin)` sets `numberOfNodes = Nbin + 1` and zero-fills
`dataVector`. All four im_r_matrix low-energy tables pass `tableSize = 101` as `Nbin` and then
call `PutValue` 101 times, for indices 0 to 100 - so `dataVector[101]` stays **zero** and the
last interpolation interval runs from the last tabulated cross section down to nothing over the
top 1% of the vector's range. At `sqrtS == edgeMax` exactly, `Value()` returns 0.

`G4XNNTotalLowE` has the same shape in a different container: `ss[29]` is declared with 29 slots
and initialised with 28 energies, so `ss[28]` is zero, and the constructor's
`for (i=0; i<29; i++)` pushes a 29th (energy, sigma) pair at **sqrt(s) = 0** carrying the last
cross section. `G4LowEXsection::CrossSection` takes logs of both members of the pair it
interpolates between, so reaching that pair would evaluate `log(0)`; and the loop that finds the
pair leaves its iterator on `end()-1` for an argument past the last energy, after which `*(it+1)`
reads one past the end of the vector.

None of this is reachable in QBBC. `G4XNNTotalLowE::IsValid` is `e > 0 && e < 3*GeV` and
`G4XNNElasticLowE`'s is `InLimits(e, 0, 3*GeV)`, while `edgeMax` is 5.21 GeV and the 29th pair
needs 3002.71 MeV; the `G4CrossSectionPatch` above them hands everything past 3 GeV to
`G4XPDGTotal` or to the transition blend. So the zeros sit one validity check away from being
read, and what keeps them unread is a property of the COMPOSITION and not of the classes.

The port reproduces the zero node (`LogVec101::node_value` returns 0 past index 100, scaled by
millibarn like every other node) and the zero 29th energy, and refuses the out-of-bounds read
rather than performing it: `lowe_xsection` stops at the last complete pair and sets a flag, because
a kernel that reads past an array does not throw, it returns a number. `tests/test_bic_imr.cu`
asserts `ss[28] == 0` and `ss[27] == 3002.71`, so a release that fills the slot fails the
extractor's count check first and this second.

### V93: a 200 mb cap truncates the proton-proton cross section below 15 MeV, and two gates beside it can never do anything

`G4Scatterer::GetTimeToInteraction` decides whether two tracks collide by comparing the squared
impact parameter against `sigma/pi`. Before it gets there it applies four cheap rejections:

    static const G4double maxCrossSection = 500*millibarn;
    if (0.7*pi*distance_fast > maxCrossSection) return time;          // (a) LAB transverse
    ...
    if (pi*distance > maxCrossSection) return time;                   // (b) CM impact parameter
    static const G4double maxChargedCrossSection = 200*millibarn;
    if (both charged && pi*distance > maxChargedCrossSection) return time;          // (c)
    if (either is a neutron && sqrtS > 1.91*GeV && pi*distance > maxChargedCrossSection) ... // (d)

Three of the four read as speed optimisations - reject obviously-distant pairs before paying for a
boost and a table lookup. **(c) is not one.** One millibarn is 0.1 fm^2, so 200 mb is a disc of
radius 2.523 fm, and the tabulated pp total cross section is **above** 200 mb whenever
`sqrt(s) < 1884 MeV` - which is a kinetic energy below about 15 MeV, where `G4XNNTotalLowE` reads
250 mb at 1882.7 MeV, 600 at 1879.6 and 2000 at 1877.05. Between 2.523 fm and `sqrt(sigma/pi)`
the final test would have accepted the collision and (c) refuses it.

Measured on the 8,448 configurations `tests/test_bic_imr.cu` compares: of the 1,056 pp rows with
the target in front of the projectile, **128 have their collision suppressed by (c) alone**, at
2, 5 and 12 MeV of relative kinetic energy where the total is 1499, 597 and 249 mb. Removing the
gate changes the verdict of every one of them; the first is `T = 2 MeV, b = 2.5232 fm,
sigma = 1499.3 mb`.

Whether that matters to a dose is a separate question - a 2 MeV pp collision inside a nucleus
transfers little - but it is not a rounding and it is not an optimisation: it is a hard ceiling on
the pp interaction radius that no cross section can exceed, and nothing in the class says so.

**(a) and (d), by contrast, provably cannot change an answer, and one of them says so itself.**

  * (d) fires only above `sqrt(s) = 1.91 GeV`, where the NN total is under 50 mb, so `pi*distance`
    above 200 mb is four times what the final test could accept anyway. Geant4's own comment is
    the proof: "neutrons special - pn is largest cross-section, but above 1.91 GeV is less than
    200 mb". Removing it changed none of the 8,448 verdicts.
  * (a) is (b) evaluated in the lab frame with a 0.7 safety margin, so it can only pass pairs that
    (b) will reject. Dropping the margin - making it strictly stronger - changed none of the 8,448
    verdicts either. The margin exists in case the CM impact parameter exceeds the lab transverse
    distance; on this grid it never does by enough to matter.

All four are reproduced in `src/physics/hadronic/bic/im_r/scatterer.cuh`, and the port reports
WHICH gate stopped a pair (`TimeGate`) rather than collapsing all of them into DBL_MAX, because
"no collision" is five different physical statements and a cascade that ends early is diagnosed by
which one.

### V94: five of the six resonance tables halve their cross section and the sixth does not

The six `G4X*Table` classes that carry the NN -> resonance production cross sections each build a
`G4PhysicsFreeVector` in a method called `CrossSectionTable()`, and each has exactly one line that
scales the tabulated number:

    G4XNDeltaTable          0.5*sigmaND1232[i] * millibarn
    G4XDeltaDeltaTable      0.5*sigmaDD1232[i] * millibarn
    G4XNDeltastarTable      *(sigmaPointer + i) * 0.5* millibarn
    G4XDeltaDeltastarTable  *(sigmaPointer + i) * 0.5* millibarn
    G4XDeltaNstarTable      *(sigmaPointer + i) * 0.5* millibarn
    G4XNNstarTable          *(sigmaPointer + i) * millibarn          <-- no 0.5

So every NN -> N N* cross section is twice its five siblings' convention. The fifteen N*
resonances - N(1440) through N(2250) - are the ones affected, and `G4XDeltaNstarTable` carries the
same fifteen columns WITH the half, so the same resonance is halved in one channel and not in the
other.

The half is undocumented in all six. It reads as the isospin-averaging factor that
`G4VXResonance::IsospinCorrection` divides back out through `pWeight` - the proton-proton
Clebsch-Gordan weight - in which case the N N* channel is the one that is not divided by it and
is a factor of two high relative to N Delta*, Delta Delta* and Delta N*. It could equally be that
the N N* tables were generated already halved and the five others were not; nothing in the source
says which.

Either way it is not a rounding. `G4CollisionComposite::FinalState` picks a channel by throwing
one uniform against the SUM of the partial cross sections, so a factor of two on one of eight
partials changes how often that channel is selected at every energy where it is open.

Reproduced per table in `src/physics/hadronic/bic/im_r/resonance_tables.cuh`
(`resonance_table_scale`), and `tools/extract_bic_imr.pl` reads the line out of each of the six
classes and asserts which five have the factor - so a release that regularises it fails the
extractor rather than moving a channel's weight quietly. Found by the oracle: the port applied the
half uniformly and the first `nnstar` comparison came back at exactly 0.5 relative.

Two smaller things in the same six files, recorded here because they are the same kind of fact:

  * `G4XNNstarTable::sigmaNN1535` and `::sigmaNN2190` are declared `[121]` and initialised with
    113 values, so their top eight entries are zero where their neighbours are still 0.005 mb.
    Unreachable below 39 GeV; pinned by the extractor as an exact set.
  * `G4XNDeltastarTable.hh` says `// 40 is missing... @@@@@@@` beside `sigmaND1930`, and there is
    indeed no `sigmaND1940` column - `delta(1940)` exists as a particle and has no N Delta*
    production cross section. The extractor asserts the absence.

---

### V95: the second msc model was worth a fifth of the row, and the reference moved further than that

V83's switch, flipped. `em::kWentzelLeptonMscWired` is `true`: `G4EmStandardPhysics::
ConstructProcess` gives e+- a `G4UrbanMscModel` below `MscEnergyLimit()` = 100 MeV and a
`G4WentzelVIModel` above it, and `step_lepton` now dispatches both. Nothing in the physics moved
to make that possible - P14c had the model, the table and the dispatch written and oracled
(V83) - and what held it off was the translation unit, exactly as V63/V65/V66 held
`kUrbanIonMscWired` off for the ion. With P8e's split the whole engine compiles in **361.8 s**,
seventeen units, first time, and `src/host/transport_run_lepton.cu` alone rebuilds in 149 s.

#### The kernel, and it is 448 bytes

`-Xptxas -v` on the shipped lepton unit, both instantiations, every column identical in each:

| `run_step_lepton<double, _, StepTap<double>>` | stack frame | spill st/ld | registers | cmem[0] |
|---|---|---|---|---|
| `kWentzelLeptonMscWired` false | 3040 B | 96/56 | 255 | 1472 |
| **true** | **3488 B** | **100/56** | 255 | 1472 |

The four `__device__ __noinline__` wrappers V81 added are in the log as functions of their own -
`wv_lepton_limit`, `wv_lepton_geom`, `wv_lepton_true`, `wv_lepton_scatter`, none of them holding
a frame - which is why a second complete msc model costs a frame and not an explosion. The
register count is at the 255 cap on both sides, as it has been since P8e.

#### The gate cannot move, and does not, to the track-step

B1's 6 MeV gamma gate at 2,000,000 events: **425.945 pGy +/- 0.867349, 1.17179 sigma,
25,993,577 track-steps**, with the flag in EITHER position - every printed digit and every step
the same. The branch is a strict `>` against `kMscEnergyLimit()` (`G4RegionModels::SelectIndex`
tests `e <= lowKineticEnergy[idx]`, so 100 MeV itself belongs to Urban: `tests/
test_electron_hi.cu` checks that against Geant4's own answer), and no secondary of a 6 MeV
photon reaches 100 MeV, so no uniform this branch draws can reach the gate. The same holds for
the sweep's **e- 100 MeV row**, whose primaries start at exactly 100 MeV: 3.31744E-007 Gy before
and after, identical, -0.09% and -0.4 sigma against Geant4 either way.

`tests/test_lepton_transport.cu` passes on the device both ways. The balance closes to
**1.023e-15** of the primary energy with the branch on (9.095e-16 with it off) and block 4 reads
**9,585 of 15,933** steps above 100 MeV against **9,508 of 13,728** - which are V83's two
recorded columns, reproduced. The 1 MeV and 50 MeV rows of the balance are identical in every
column, which is the gate's claim again at the level of one track.

#### The 1 GeV row: what the substitution was worth, and what it did not close

The sweep at its own event counts says -1.67% (3.7 sigma) before and **-1.28% (2.9 sigma)**
after. That comparison is not good enough to carry the finding, and the reason is the reference.
Both sides re-taken at **1,000,000 events**, port and Geant4, same configuration, same script's
macros:

| 1 GeV e- into B1, dose per 10,000 events | | vs Geant4 (1M) |
|---|--:|--:|
| port, `kWentzelLeptonMscWired` **false** | 23,734.2 +/- 24.06 pGy | -0.532%, **-3.74 sigma** |
| port, `kWentzelLeptonMscWired` **true** | **23,782.0 +/- 23.97 pGy** | -0.332%, **-2.33 sigma** |
| *the same, after V96's continuous loss - what ships* | *23,767.8 +/- 23.95 pGy* | *-0.391%, -2.75 sigma* |
| Geant4 11.1.1 EM-only, 1,000,000 events | 23,861.2 +/- 24.04 pGy | - |
| *Geant4 11.1.1 EM-only, 100,000 events (the sweep's)* | *24,027.3 +/- 76.2 pGy* | |

The third row is there so that this entry's number is not read as the shipping one: P14d's second
change (V96) moved this row a further -14.2 pGy, -0.060%, which two 1,000,000-event samples
cannot resolve - their difference carries 34 pGy of noise, so 0.42 sigma - and the two changes
together ship at -0.391%. The msc number this entry measures is the +47.8 pGy between the first
two rows, taken with everything else held fixed.

So **the second msc model is worth +47.8 pGy, +0.201% of the row**, and it moves the row from
-0.53% to -0.33%. It is a fifth of the deficit. V83 called it "the leading candidate for the
residual - not a proven attribution"; it is now measured, and it was not the whole of it.

**And the reference moved further than the fix did.** Geant4's own two samples of the same
configuration differ by -0.69% - 24,027.3 pGy at 100,000 events against 23,861.2 at 1,000,000 -
which is **2.3 sigma of the 100,000-event run's own quoted rms**, computed on the pair that
shares its first 100,000 events. A 1 GeV electron deposits in a 6 cm trapezoid 19 cm inside the
envelope through a heavy-tailed distribution: rare showers that happen to develop in the right
place carry a large share of the dose, and B1's rms estimator converges slowly on such a
distribution. **So part of the -1.67% the sweep reported was the reference, not the port**, and
the 1 GeV electron row should not be read at 100,000 events by either side. The estimator itself
is not broken - Geant4's own error falls from 76.2 pGy at 100,000 events to 24.04 at 1,000,000,
a ratio of 3.17 against sqrt(10) = 3.16, and the port's from 75.9 to 23.97 - so what this is is
the tail, arriving late.

#### V62 is excluded, by the measurement and not by the argument

V83 named two other terms in this row: the discrete rates (V78) and what V62's `extremesmallstep`
branch, on since P8e, does at high energy. The second is now settled. The lepton unit rebuilt
alone with `kLeptonExtremeSmallStep` forced false and the other sixteen objects in the archive
byte-for-byte the ones the row above was taken with:

| 1,000,000 events, 1 GeV e- | dose per 10k | track-steps |
|---|--:|--:|
| WentzelVI on, `kLeptonExtremeSmallStep` **on** (ships) | 23,782.0 pGy | 205,579,159 |
| WentzelVI on, `kLeptonExtremeSmallStep` **off** | 23,781.9 pGy | 205,578,950 |
| WentzelVI **off**, `kLeptonExtremeSmallStep` on | 23,734.2 pGy | 203,516,181 |

One part in 240,000 of the dose and 209 steps of 205.6 million, against 47.8 pGy and 2.06 million
steps for the msc model on the same row and the same statistics - a factor of **478**. V66
measured the same branch worth -0.00069 pGy on the gamma gate and this is the same conclusion at
a hundred times the beam energy: `tsmall` is three to eight ten-thousandths of a millimetre in
water, which is the end of a range and not the top of a shower. The flag was then restored, the
one unit rebuilt a third time and the row re-run: **23,782.0 pGy and 205,579,159 track-steps**,
the first row to the digit, which is the check that the three runs differ by the two flags and by
nothing else in the machine.

**What is left is -0.33%, 2.3 sigma, and it is not attributed.** V78's discrete rates are the
largest named candidate: this port draws brems and delta-ray interaction lengths from the models
at the pre-step energy, where `G4VEnergyLossProcess::PostStepGetPhysicalInteractionLength` caches
`preStepLambda` from a lambda table at up to `1/lambdaFactor` of the current energy and
`PostStepDoIt` then rejects with probability `1 - lambda(E_post)/preStepLambda`. That
approximation is a function of the STEP LENGTH, so it is not independent of the msc model this
entry just changed - the primary's mean step in this beam went from 34.9 mm to 12.9 mm
(`tests/test_lepton_transport.cu` block 2: 1872.8 mm over 53.6 steps to 800.9 mm over 62.2), and
whatever V78 is worth here was re-weighted by that. A second candidate is that the residual is
not on the lepton path at all: the sweep's photon rows carry -0.27% at 100 MeV and -0.52% at
6 MeV on their own, and most of a 1 GeV electron's dose in that trapezoid arrives as a shower
photon. Both are directions to look, not attributions, and the way to settle either is the way
this entry settled V62.

---

### V96: the loss was an inversion where Geant4 has a line, and it annihilated every lepton in a vacuum

V84's defect, closed. `step_lepton`'s continuous loss is
`G4VEnergyLossProcess::AlongStepDoIt`'s shape now (utils/src/G4VEnergyLossProcess.cc:811-835):

```
  if (length >= fRange || preStepKinEnergy <= lowestKinEnergy) {          // :812  "stopping"
    eloss = preStepKinEnergy; ... SetProposedKineticEnergy(0.0); return;
  }
  eloss = length*GetDEDXForScaledEnergy(preStepScaledEnergy, ...);        // :825  "Short step"
  if(eloss > preStepKinEnergy*linLossLimit) {                             // :830  "Long step"
    G4double x = (fRange - length)/reduceFactor;
    eloss = preStepKinEnergy - ScaledKinEnergyForLoss(x)/massRatio;
  }
```

then, after the fluctuation, `finalT = preStepKinEnergy - eloss; if (finalT <= lowestKinEnergy)
{ eloss += finalT; finalT = 0.0; }` at :913-917. What stood there was the long branch alone,
with a guard beneath it that took the WHOLE kinetic energy whenever the inversion came back
unchanged. For an e+- `massRatio` and `chargeSqRatio` are 1 and so is `reduceFactor` (:198), so
the long branch is exactly the expression the port already had; what was missing was the linear
one in front of it and `linLossLimit` = 0.01 deciding between them. `em::kLinearLossLimit` is a
lepton constant and not a global, because `G4ionIonisation` sets 0.02 - which is why
`step_hadron` computes its own.

#### The hadron branch did not have this defect and the proton proves it

`step_hadron` has had the linear-first structure since it was written, with the per-species
`kLinLossLimit` P8c added, and it has no "strictly reduce the energy" guard at all. The comment
above it says why in the first person: "That is exactly what the first run of this stepper did:
2000 protons, 100 MeV each, zero deposited." So the fix that went into the hadron stepper on
day one never reached the lepton one, and no harness fired a lepton through a vacuum until P14c
gave `ref/proton/proton_depth.cc` a particle name. Measured through the same `G4_Galactic` world
that kills the electron, before and after this change: **100 MeV proton 100.0000% both times**,
and the 6,000-event depth-dose gate identical in every printed column (plateau 1.00158, R80
+0.012 mm, 80-20 width +0.014 mm).

What `step_hadron` does NOT have is AlongStepDoIt's :914 balance - it lets a track fall below
`kHadronTrackingCut` and catches it on the NEXT step's entry guard rather than zeroing it here.
That is a real difference from Geant4 and it is left alone by name: it moves the proton gate, so
it is a package with its own before-and-after and not a side effect of this one.

#### What the vacuum did, and the two faces of one defect

`G4_Galactic` is hydrogen at 1e-25 g/cm3. A 20 MeV electron's range in it is **9.42e26 mm**, so
`range - step_len` for any step a geometry can produce IS `range` in double - the subtraction is
below the last bit - and `energy_from_range` then returns the pre-step energy to within the
inverse-range spline's own round-trip error. Which side of that error a run lands on decides
which face of the defect it gets, and both are wrong:

* **round-trip just OVER the pre-step energy**: the guard fires, the whole kinetic energy is
  taken as the loss of a step through a vacuum, and the track dies in a volume nothing scores.
  This is what the depth-dose harness's `G4_Galactic` does.
* **round-trip just UNDER**: the track survives, poorer by the round-trip error.
  `tests/test_lepton_transport.cu`'s hand-built vacuum does this, and it is worth **3.58e-10** of
  a 20 MeV track where the right answer is **4.14e-24** - fourteen orders of magnitude, arriving
  as a deposit because a step in a vacuum has nothing else in it.

The linear form gives `1e-26 MeV/mm * length` in both cases, which is right; no epsilon on the
inversion could be, because the information is not in the difference of two doubles that are
equal. **The second face is why that test's tolerance is 1e-15 and not 1e-9**: 1e-9 was written
first, it passes the benign face, and it would have passed the code the block exists to fail.

Measured through the harness, before and after, one engine and one source change:

| 20,000 events through the `G4_Galactic` world into a 4 m water phantom | before | after |
|---|--:|--:|
| 20 MeV e- | **0.0000%** | **99.2390%** |
| 20 MeV e+ | 1.9463% | 102.0987% |
| 1 GeV e- (100,000 events) | **0.0000%** | **97.8498%** |
| 100 MeV proton | 100.0000% | 100.0000% |

The positron's 102% is not an error: `G4eplusAnnihilation`'s two 511 keV photons are rest mass
and not beam energy, 1.022 MeV on a 20 MeV track being 5.11%, and its "before" 1.9463% was those
photons alone, emitted where the positron was killed.

#### The gate is the judge, and it moved by 0.0166%

B1's 6 MeV gamma gate at 2,000,000 events, five seeds, the same tree with one source change:

| seed | before (pGy) | after (pGy) | change | track-steps, before -> after |
|---|--:|--:|--:|---|
| default | 425.945 | **425.860** | -0.085 | 25,993,577 -> 26,061,108 |
| 1 | 426.266 | 426.191 | -0.075 | 25,960,428 -> 26,028,632 |
| 2 | 426.980 | 426.908 | -0.072 | 25,969,166 -> 26,037,076 |
| 3 | 427.548 | 427.485 | -0.063 | 25,974,297 -> 26,039,317 |
| 4 | 427.347 | 427.287 | -0.060 | 25,985,464 -> 26,050,907 |
| **mean of five** | **426.8172** | **426.7462** | **-0.0710 +/- 0.0045** | |

Against Geant4's 427.385 +/- 0.87 the gate's own seed reads **1.24145 sigma** after and 1.17179
before. So this IS resolved - every seed moves the same way and the spread of the five changes is
0.010 pGy, seven times smaller than the shift - and it is 0.0166% of the dose, 0.082 of one
run's standard error, and it leaves the gate at a quarter of its 3-sigma limit. It is not a
tolerance that was widened: the number in README and docs/RESULT.md is replaced by this one.

**The direction is the arithmetic and not a surprise.** The two forms differ at second order:
the inversion is `L*dedx(E) + (L^2/2)*d(dedx)/dx + ...`, and a restricted electron stopping power
RISES as the track slows, so it returns slightly more than `L*dedx(E_pre)`. Dropping to the
linear form on every step whose loss is under 1% of the energy therefore takes slightly less per
step, which is why the track-step count rises 0.26% and the dose falls 0.017%. Geant4 makes
exactly this approximation, and `linLossLimit` is where it stops making it.

**The sweep's electron rows move the same way and by more, because they are electron beams.**
`tools/b1_sweep.ps1 -Only "e-_1000,e-_100"` on the shipping engine against the same Geant4
columns: the 100 MeV row 3.3174E-007 -> **3.3132E-007** Gy, -0.09% -> **-0.22%** (0.4 -> 0.8
sigma), and the 1 GeV row 2.3720E-007 -> **2.3713E-007**, -1.28% -> -1.31% at 2.9 sigma either
way. At 1,000,000 events a side the two rows read **-0.205% (-1.46 sigma)** at 100 MeV and
**-0.391% (-2.75 sigma)** at 1 GeV, the latter a further -0.060% on V95's msc row - a movement
two independent 1,000,000-event samples cannot resolve (0.42 sigma) and which is therefore
quoted from the gate, where five seeds do resolve it. docs/B1_SWEEP.md carries all three engine
states side by side.

**What does NOT move is the msc arm.** `G4UrbanMscModel::SampleScattering` (G4UrbanMscModel.cc:
786-792) takes its own post-step energy from `GetEnergy(particle, currentRange-tPathLength,
couple)` - the inverse range table, read fresh - because `G4VMultipleScattering::AlongStepDoIt`
runs BEFORE the ionisation process's and has no `eloss` to read. `e_after_mean` is now that
lookup explicitly rather than a by-product of the loss, so every uniform `step_lepton` draws is
where it was.

#### The electron depth-dose, which is the curve P14c prepared and could not run

100,000 e- of 1 GeV into 4 m of water, 20 mm slabs, 400 mm half-width, 0.7 mm cut - one source
file, two builds, as the proton curve is:

| | Geant4 11.1.1 | port | diff |
|---|--:|--:|--:|
| contained in the phantom | 97.6448% | 97.8498% | +0.205 pp, **+0.21%** |
| entrance half, 0-2,390 mm | - | 0.99989 of G4 | **-0.011%** |
| R80, the distal 80% of the peak | 1,079.258 mm | 1,081.695 mm | **+2.437 mm, +0.23%** |
| 80%-to-20% distal falloff | 1,152.746 mm | 1,172.994 mm | +20.248 mm, +1.76% |
| shower maximum, parabolic | 683.6 mm = 1.90 X0 | 652.9 mm = 1.81 X0 | -30.7 mm |

**The shower maximum is the weakest number in that table and it should not be read as the
strongest.** The curve is flat to 0.4% over +/- 80 mm around it - the Geant4 bins from 610 to
770 mm run 1.2838e6, 1.2912e6, 1.2949e6, 1.2990e6, 1.2994e6, 1.2973e6, 1.2906e6, 1.2844e6,
1.2773e6 MeV - so a parabola through three 20 mm bins at 100,000 events is fitting noise, and bin
for bin through that whole region the two curves agree to between 0.2% and 0.6%. The numbers that
are well determined are the integral (+0.21%), the entrance half (-0.011%) and R80 (+0.23%), and
they say the same thing: the port puts the same energy in the same place to about two parts in a
thousand, and keeps a little more of it, which is consistent with its 1.76%-longer distal tail.

#### One thing the harness needed, and it had never been reached

`ref/proton/proton_depth.cc` left the engine's live-track pool at its default 4.0 per event,
which is right for a proton and not for anything that showers. With the leptons transporting,
both the 1 GeV and the 20 MeV electron runs stopped at once with

    FATAL: no track can be stepped without overrunning the pool.
           19997 tracks are live and the pool holds 20000 slots a side.

A showering beam now gets `max(32, E/10)` per event and a hadron keeps 4.0, so no run that
existed before is changed. Two things are worth keeping: the engine REFUSES rather than
truncating a shower, which is how both failures announced themselves in one line; and the reason
nobody had hit it is the defect this entry closes - every lepton had died in the vacuum before it
could make a second track.

---

### V97: a positron that leaves the world is annihilated on the way out

Found by V96's new vacuum block in `tests/test_lepton_transport.cu`, and pinned there rather
than fixed.

`step_lepton`'s dying block is reached by two different tracks - one that fell below its cut and
one whose `p.volume` became `geom::kOutsideWorld` - and it ends

    if (is_positron) { ... push(kGamma, a.dir1, ...); push(kGamma, a.dir2, ...); }

unconditionally. So a positron that simply LEAVES emits two 511 keV photons it never stopped to
make, from rest mass that is still on the track. Measured: a 20 MeV positron crossing 4 m of
`G4_Galactic` escapes with **100.000000%** of its energy and hands **5.109989%** of it to
secondaries at the same time, which is 1.022/20 exactly.

Geant4 does not do this. `G4eplusAnnihilation`'s at-rest branch is an AtRest process, and a track
that reaches the world boundary is killed by transportation with `fWorldBoundary`; nothing
invokes an AtRest DoIt on it.

**It costs no dose anywhere and it is still worth an entry.** The photons are pushed at a
position outside the world, so the gamma kernel kills them on their first step and nothing scores
there - which is exactly why it has survived every gate this project has. What it costs is work,
and what it breaks is a books claim: 1.022 MeV per escaping positron appears in the port that was
not in the beam. It is not fixed here because it is not this package's change and it would move
the same gate V96's continuous loss is being judged on; the number is asserted in the test
instead, so that fixing it has to come through that line.

### V98: a parton's constructor samples, and only the draw count could have said so

`G4Parton::G4Parton( G4int PDGcode )` (util/src/G4Parton.cc:38) draws a colour -
`(G4int)(3.*G4UniformRand())+1`, once for a quark, once for a diquark and twice for a gluon -
and then a spin projection, `(G4int)((iSpin+1)*G4UniformRand())`, whenever `GetPDGiSpin()` is
non-zero. Nothing in FTF ever READS `theColour`, `theSpinZ` or `theIsoSpinZ`: the fragmentation
uses the PDG code and nothing else. So the constructor is, from the port's point of view, three
uniforms and no information.

`G4DiffractiveSplitableHadron::SplitUp` builds two of them on every string, so a proton costs
FIVE deviates - two in `ChooseStringEnds`, two for the u quark, one for the spin-0 diquark - and
the first transcription spent two. Every hadron of every event after the first string would have
been drawn from a stream shifted by three.

The answer was right and the count was wrong, which is the only shape this error has:
`ref/oracle/ftf_splitup.csv` compares both parton codes AND the draw count, and the codes agreed
at every one of 296 (code, phase) points while the count read `got 2 want 5`. The same argument
2.1.11 makes for the K0 coin toss, one layer up. Fixing it needed a new oracle column,
`GetPDGiSpin`, because a spin-0 diquark costs one deviate and a spin-1 diquark two, and no rule
about the PDG code is the port's to invent.

### V99: Geant4 divides two integers to decide whether a collision may be inelastic

`G4FTFModel::ExciteParticipants` (G4FTFModel.cc:917) reads

    if ( G4UniformRand() < ( 1.0 - target->GetSoftCollisionCount()     / MaxNumOfInelCollisions ) *
                           ( 1.0 - projectile->GetSoftCollisionCount() / MaxNumOfInelCollisions ) )

and `GetSoftCollisionCount()` returns `G4int` while `MaxNumOfInelCollisions` is a `G4int`. The
division is INTEGER. The factor is therefore exactly 1 until a hadron has had `Nmax` soft
collisions and exactly 0 at `Nmax` - a hard cap, not a linear suppression. (At `2*Nmax` it is
-1, so two saturated hadrons give a product of +1 and the collision is accepted again, which is
why the count is compared and never clamped.)

Written with a floating-point division - which is what the expression looks like it means, and
what a reader who has not checked the return type will write - the model rejects inelastic
collisions gradually and turns them into elastic ones. MEASURED against
`ref/oracle/ftf_modelstat_*.csv` for a 10 GeV proton on lead: **12% fewer pions**, 0.33 fewer
quark-exchange tracks per event, 7.5% too many excited strings, and an excited-string mass
spectrum 27% short in the 1.26-1.58 GeV bin.

What makes it worth an entry is what did NOT move. The wounded-nucleon count agreed to 0.2%, the
string-count distribution agreed, the impact parameter agreed, and the total multiplicity was
within the gate. A statistical test built only out of "how many nucleons were hit" and "how many
hadrons came out" would have passed. It was found by adding three counters that split those two
apart - the impact parameter alone, `A - NumberOfTargetSpectatorNucleons` (the Glauber count,
separately from the reggeon cascade) and `GetNumberOfNNcollisions` (how many of those were
actually excited) - all three of which are PUBLIC on `G4FTFModel` and cost one line each to
dump. A histogram that measures an outcome cannot say which step produced it; one that measures
a step can.

### V100: P6 refuses an edge case that is on FTFP's main path

`docs/PORTED.md` 2.1.4 refuses `G4DecayKineticTracks` - the first line of both
`G4GeneratorPrecompoundInterface::Propagate` entry points, which decays every short-lived track
in the list through `G4KineticTrack::Decay()`. The refusal is written as one of four edge cases,
alongside anti-nuclei and hypernuclei, and `CascadeTrack::is_short_lived` carries the fact so
that a list containing one is refused rather than de-excited as though the resonance were a
stable secondary. That is the right behaviour and the wrong expectation of how often it happens.

The FTF fragmentation produces resonances as its NORMAL output. `G4ExcitedStringDecay::
FragmentStrings` redraws every short-lived product's mass from a Breit-Wigner and hands the rho,
omega, K*, eta and Delta on undecayed - 2.1.11's own note says the resonances "are most of what
a string makes". Measured: 4 of the first 5 `ftf::apply_yourself` calls for a 10 GeV proton on
carbon were refused by `preco::propagate_residual` on the short-lived test, and the one that was
not had only two secondaries.

So the FTFP chain is complete up to the hand-over and stops there, for a reason that belongs to
neither package alone: P11b produces exactly what Geant4 produces, and P6 ported the interface
without the decay stage in front of it. `G4KineticTrack::Decay` needs the resonance-width
machinery (`G4SampleResonance`, `G4Integrator`, `IntegrateCMMomentum`) that
`bic/kinetic_track.cuh` also refuses by name - it belongs with `G4BCDecay` - so whoever writes it
closes three refusals at once. Until then the FTFP arm makes strings, fragments them and reports
at the boundary; it does not silently de-excite a rho.

### V101: the FTF interaction workspace is 332 kB, and that is per track

`sizeof(ftf::FtfWorkspace<250, 64, 512, 320, 256, 96>)` is 332,368 bytes. It holds two
`bic::Nucleon` arrays (250 + 64 nucleons at 72 bytes), one shared `Nucleus3DScratch`, a
565-slot splitable-hadron pool at 168 bytes each, 512 interactions at 32, 320 `ExcitedString`s
at 184, and P11's 53,816-byte string-decay workspace. Every byte is behind a pointer and none of
it is on the stack - the device probe is an 864-byte frame with nothing spilled - but it is one
per TRACK IN FLIGHT, not one per string or per event.

For scale: P11's string-decay workspace alone is 53,816 bytes and P9's nucleus 18 kB, so this is
five times the largest thing the hadronic port had before it. A kernel with a few thousand
resident tracks wants most of a gigabyte for workspaces alone.

Three of the capacities are template parameters and all three are REFUSED rather than truncated
when exceeded, so the trade is available to whoever wires this in: `kMaxProjA` is 250 only
because nothing should be refused by default and 64 covers every ion a galactic-cosmic-ray
problem contains; `kMaxStrings` at 320 is above what a proton on lead produces (the
20,000-event maximum is 31) but below the `1 + A_target + A_proj` that cannot overflow;
`kMaxInteractions` at 512 the same. Shrinking them is a decision about which events get reported
instead of simulated, which is why it is recorded here rather than made here.

### V102: two columns with the same name, and the reader took the second

`ref/oracle/ftf_nucleus.csv` was dumped with the header
`nucleus,a,z,index,type,x,y,z,px,py,pz,e,binding` - the nucleus's CHARGE and the nucleon's z
COORDINATE both called `z`. A reader that builds a name-to-index map keeps the last, so the port
replayed Al27 with `Z = 3`, being the integer part of the first nucleon's z in fermi.

What makes it worth recording is how nearly it hid. Z enters the replayed configuration only
through `G4FTFParameters::InitForInteraction`, which mixes the proton and neutron cross sections
by Z/A, and from there only through `RadiusOfHNinteractions2` in the step function
`RadiusOfHNinteractions2 > b^2`. So the impact parameter agreed exactly, the draw count agreed
exactly, and the identity and interaction time of the participant the two sides DID agree on
agreed exactly - 1,260 of 1,460 comparisons were still zero. One nucleon in one of the 200
(nucleus, projectile, phase) groups changed sides.

The general lesson is for the dump and not for the port: a CSV column name is an identifier in a
namespace the writer does not control, and `x,y,z` for a position sitting next to `a,z` for a
nuclide is a collision waiting for a reader. They are `posx,posy,posz` now.

### V103: the two log-distribution probabilities are distinct parameters with the same value

`G4DiffractiveExcitation::ExciteParticipants_doNonDiffraction` samples the projectile's
light-cone minus-component with `GetProbLogDistrPrD()` and the target's plus-component with
`GetProbLogDistr()`. They are separate members of `G4FTFParameters`, set in separate statements,
and in 11.1.1 every branch of `InitForInteraction` sets both to 0.55 - the baryon collection,
the pion collection and the two literal fallbacks.

Swapping them in the port changed nothing: 10,749 comparisons, all still exact. That is a FAILED
perturbation, and it is recorded rather than quietly replaced because the reason it failed is a
property of Geant4 and not of the test. No oracle this package can build will ever separate the
two, and a future tune that gives them different values would make a silently-swapped port wrong
with no existing test able to see it. The perturbation that DOES bite that arm is the sign of
`Qplus = -(TPlusNew - Ptarget.plus())`, which fails four exact buckets and two statistics.

### V104: an uninitialised bool decided whether a nucleus diffracts, and it read as true

`G4FTFParameters` sets `EnableDiffDissociationForBGreater10` in its CONSTRUCTOR, from
`G4HadronicParameters::Instance()->EnableDiffDissociationForBGreater10()` - false in every stock
build. `Reset()`, which `InitForInteraction` calls first, does not touch it, and neither do the
gluon-splitting probabilities, the kink switch, or row 4 of the `[5][7]` `ProcParams` array. The
port had all of those in `ftf_parameters_construct`, and `ftf_parameters_construct` HAD NO
CALLER: every `FtfParameters<double> p;` on a stack, and every one inside a workspace, went into
`ftf_init_for_interaction` with those members indeterminate.

The one that decides physics is the bool. `InitForInteraction` reads

    if ((AbsProjectileBaryonNumber > 10 || NumberOfTargetNucleons > 10)
        && !EnableDiffDissociationForBGreater10) { SetParams(2, 0,...,-100); SetParams(3, ...); }

so for any target heavier than A = 10 - carbon included - Geant4 switches BOTH projectile and
target diffraction dissociation off, and `GetProcProb(2, y)` and `GetProcProb(3, y)` return zero.
Reading the stack byte as true kept them on: 0.207 each for p + C at 4.7 GeV/c instead of 0.

Two things about how it surfaced are worth keeping. The first is that it is undefined behaviour,
so it moved: the excite rows passed for days and then began failing after unrelated code was
added above them, and the first hypothesis - that the oracle table had changed - was wrong. Two
runs of the same `g4dump.exe` write all 38 `ftf_*.csv` byte-identically, which is what finally
ruled the oracle out.

The second is where it shows. `ProbOfDiffraction == 0` is also the third arm of

    if (SqrtS < M0projectile + TargetDiffStateMinMass || SqrtS < ProjectileDiffStateMinMass +
        M0target || ProbOfDiffraction == 0.0) ProbExc = 0.0;

at the end of `ExciteParticipants_doChargeExchange`, so with diffraction off, `ProbExc` is zero,
`G4UniformRand() > 0` is always true, and EVERY charge exchange on a nuclear target ends in an
elastic scattering of the two new hadrons rather than an excitation. Half of the p + C phases
changed branch. The `excite verdict` and `excite species` buckets stayed exact through all of it
- the two hadrons come out with the same PDG codes either way - and only the status, the
four-momenta and the draw count could see it. Perturbing the default back to true reproduces the
original failure exactly (`p_C_4GeV ph 0 tstatus got 0 want 2`, `draws got 14 want 11`,
`p_C_50GeV ph 0 tpz` at 5.140e+01) and takes eleven statistical rows with it, the worst at
44.2 sigma.

The fix is that every member the Geant4 constructor sets and `Reset` does not now carries its
constructor value as a default member initialiser, so a default-constructed `FtfParameters`
equals a default-constructed `G4FTFParameters`. `ftf_parameters_construct` stays for a caller
that wants to pass a non-stock `EnableDiffDissociationForBGreater10`.

### V105: the oracle measured the model, the test measured the model plus a rejection loop

`G4VPartonStringModel::Scatter` does not just call `G4FTFModel::Init` and `GetStrings`. It wraps
them in a loop that throws the whole attempt away and re-samples if the nuclear residual is
unphysical - more than three protons with no neutrons, or no protons with more than one neutron,
on either nucleus - and re-samples again if the string list is empty, if `FragmentStrings`
returns nothing, or if `SumMass > InvMass`.

That loop is a CONDITIONING, not a filter on failures. The residual test rejects exactly the
attempts that hit the most nucleons, so the events that survive it sit at larger impact
parameter and carry fewer participants than the events the model produced. Measured on the port:
for C12 on carbon at 8 GeV/nucleon the residual table rejects **7.2%** of attempts (3.4% on the
target residual, 3.6% on the projectile's), and it moves `<participants>` from 4.275 to 4.139
and `<b>` from 3.135 fm to 3.196 fm.

`ref/dump/dump_ftf.cc`'s `dump_modelstat` has TWO passes and its own comment says why: the
string-level counters - `nstrings`, `log10mass`, `excited`, `b_halffm`, `nncoll`, `participants`
- come from `Init` + `GetStrings` with no `Scatter` around them, and only the hadron-level ones -
the species, the multiplicity, the holes - come from `Scatter`. `tests/test_ftf_model.cu` read
BOTH out of one `ftf_scatter` call. For a proton beam the residual table fires on 1.6% of
attempts and the mismatch is invisible; for an ion beam it fires on 7.2% and it read as

    model participants   C12_C_8 part 6   got 1204 want 1471    5.34 sigma
    model excited        C12_C_8 excited 0                      6.32 sigma
    200k participants    C12_C_8 part 1   got 46182 want 42812  12.81 sigma

- and docs/RISK.md V88's rule, applied honestly, made it WORSE rather than better, because the
difference was real: it grew by sqrt(10) at ten times the statistics exactly as a real difference
does. V88 is right about what growth means; it cannot tell a difference in the physics from a
difference in what the two sides measured.

Three separate things were exonerated before the cause was found, and each of them cost an
oracle run. The C12 nucleus: 200,000 sampled configurations of its outer radius, transverse RMS
and radial RMS, all under 3 sigma (`C12big` in `ftf_nucstat.csv`). The FTF parameters for an ion
projectile: 336 comparisons at 0. The nucleus-nucleus `GetList` on replayed nuclei: 483
comparisons at 0. And the one number nothing had measured - the projectile nucleus's outer
radius AFTER `G4FTFModel::Init` boosts and Lorentz-contracts it, which sets the impact-parameter
range - agrees to 0.16% (`ftf_aaradius.csv`, added for this).

What found it was measuring the port WITHOUT the loop: the port's unconditioned
`<participants>` = 4.275 against the oracle's 4.297, `<nncoll>` 6.415 against 6.433, `<nstrings>`
12.212 against 12.283 and `<b>` 3.135 fm against 3.124. Four numbers that had been 4% apart
were half a percent apart, which is not what a model bug looks like.

The fix is that the test now runs the two passes the dump runs, under two different Philox keys
because the dump's two passes are two different sets of events. With it, every statistical row
is under 5 sigma - worst 4.13 - and the C12-on-carbon rows are 2.95 and 2.61. The lesson is the
one the dump's own comment was already making and that the test did not honour: when an oracle
is generated by two different calls, the port has to make the same two calls, and a case where
the two agree anyway is not evidence that they are the same call. `ftf_prescatter.csv` is here so
that the pre-rejection distributions and the rejection's own inputs - the proton and neutron hit
counts of BOTH nuclei - are compared directly rather than inferred.

### V106: G4Clebsch::GenerateIso3 reads an uninitialised array and cannot select its last row

`G4Clebsch::GenerateIso3` chooses the two outgoing isospin projections of a resonance pair. It is
called from one place - `G4VXResonance::IsospinCorrection`, when either incoming track is
short-lived - and four things in it are transcribable only by writing them down:

**A 400-entry array, initialised where a condition holds and divided through everywhere.**

    G4double prbout[size][size];          // size = 20, a plain local, no initialiser
    ...
        if(m1pr + m2pr == twoM3) { ... prbout[m1pos][m2pos] = cleb; sum += cleb; }
        else                     { prbout[m1pos][m2pos] = 0.; }

The `else` covers every `(m1pos, m2pos)` the two loops VISIT, and the loops visit only
`m1pos < (2*twoJOut1/2 + 1)` by `m2pos < (2*twoJOut2/2 + 1)` - at most 4 x 4 for two Deltas. The
other 384 entries of the 400 are never written. The normalisation that follows is

    for (G4int i=0; i<size; i++) for (G4int j=0; j<size; j++) prbout[i][j] /= sum;

over all 400, reading and writing the uninitialised ones; and the sampling loop then walks
`prbout[m1p][m2p]` and subtracts each from `rand`. Whether it reaches an uninitialised entry
depends on where the sampling loop stops, which is the next item.

**The sampling loops are exclusive on both indices, so the last row and the last column can never
be chosen.**

    for (m1p=0; m1p<m1pos; m1p++) for (m2p=0; m2p<m2pos; m2p++)

`m1pos` and `m2pos` are left holding the index of the LAST entry written, not one past it. For a
Delta (`twoJOut = 3`) the four charge states are m = -3, -1, +1, +3 and `m1pos` ends at 3, so
`m1p` runs 0, 1, 2 and the `+3` state is unreachable. Every probability in that row and column is
subtracted from `rand` on the way past but can never be returned.

**And falling off the end returns an empty vector.** The function ends with a `JustWarning`
("Should never get here") and `return temp` with nothing pushed - and the caller does

    std::vector<G4double> iso = clebsch.GenerateIso3(...);
    G4int isoA = G4lrint(iso[0]);
    G4int isoB = G4lrint(iso[1]);

which indexes a vector of size 0.

The fourth thing is harmless and is the reason to doubt the rest: `c34 = ClebschGordan(0,0,0,0,0)`
is identically 1 and is recomputed inside the innermost loop.

**What the port does.** `clebsch_generate_iso3` keeps the three early returns that have a
well-defined answer - both isospins zero, and either outgoing isospin zero, which is what the
binary cascade's own pairs reach when one product is a nucleon - and REFUSES the sampling branch
by name (`ClebschRefusal::generate_iso3`). Transcribing it would mean transcribing a read of
uninitialised stack, and a kernel has no stack to read that a host would even agree with.

The three functions the cross sections actually multiply by - `TriangleCoeff`,
`ClebschGordanCoeff` and `Weight` - are complete and bitwise over 39,291 points.

Two smaller facts measured in the same pass, both of them dead code rather than bugs:

  * `ClebschGordanCoeff`'s three `JustWarning; return 0` exits - `kMin < 0`, `kMax < kMin` and
    `kMax >= 512` - are unreachable. `kMin` starts at 0 and only grows; once the two M checks and
    the triangle inequality have passed, the Racah bounds give `kMax >= kMin`. Geant4 agrees from
    its side: the 18,225-point sweep raises zero G4Exceptions. `tests/test_bic_imr.cu` asserts the
    count is zero, and removing the M guard makes it 900 - so the assertion is an interlock on the
    guard above it, not a tautology.
  * `Weight` raises both ends of its isospin range to `|M|`, and it cannot matter: `ClebschGordan`
    already returns zero for `|twoM| > twoJ`, so the floor only removes zero terms. Removing it
    changes none of the 1,600 weights.

### V107: the binary cascade's pion-nucleon ELASTIC cross section is exactly zero everywhere it runs

QBBC registers `G4BinaryCascade` for pi+ and pi- from 0 to 1.5 GeV, and
`G4BinaryCascade::ApplyYourself`'s species test is an `&&`, so a pion enters the cascade at every
energy (docs/PORTED.md 2.1.10). Inside it, `G4Scatterer`'s second channel is
`G4CollisionMesonBaryon`, a composite of `G4CollisionMesonBaryonToResonance` and
`G4CollisionMesonBaryonElastic`. The elastic one contributes **nothing at all** over that window,
and the reason is three classes deep.

`G4CollisionMesonBaryonElastic`'s cross section is `G4XMesonBaryonElastic`, which does not use its
own tracks' identities: it builds a dummy pi+ and a dummy proton carrying the REAL four-momenta,
evaluates `G4XPDGElastic` on that pair, and scales by a ratio of two AQM cross sections. So the
answer is the PDG pi+p elastic fit read at a `pLab` built from the real kinematics and the pi+ and
proton PDG masses - and that fit's first parameter is

    const G4double G4XPDGElastic::pPiPlusPDGFit[7] = { 2., 200., 0., 11.4, -0.4, 0.079, 0. };

with `if (pLab < pMinFit) return 0.0;` and `pMinFit = 2 GeV`. A pion on a nucleon at rest reaches
`pLab = 2 GeV` only at a pion kinetic energy of **1870 MeV**, which is past the 1.5 GeV where QBBC
stops handing pions to BIC at all.

Measured on the 900 in-charge configurations `tests/test_bic_imr.cu` compares: every one of the
600 pion-NUCLEON rows is exactly zero, from 20 MeV to 1500 MeV of pion kinetic energy, with the
target at rest and with the target carrying its own Fermi-scale momentum. The only non-zero rows -
113 of them - are the pairs where the baryon is a RESONANCE: a pi+ on a Delta(1232) turns on at
sqrt(s) = 2166 MeV and a pi- on an N(1440) earlier still, because the heavier baryon raises
sqrt(s) at the same kinetic energy and the dummy `pLab`, which is built with the PROTON mass, goes
past 2 GeV.

So every pion-nucleon collision the binary cascade performs goes through
`G4CollisionMesonBaryonToResonance` - Delta and N* production - and the elastic half exists only
for a pion that finds a resonance still alive. That is worth knowing before anyone reads a
pion-induced spectrum out of this model, and it is also why the composite's total cannot be
approximated by its elastic partial: at 300 MeV the Delta dominates and the elastic partial is 0.

Two smaller facts from the same three classes, both reproduced:

  * **`G4XAqmTotal`'s strangeness ratio is an integer division.** `G4int sTrk1`, `G4int qTrk1`,
    `G4double sRatio1 = sTrk1 / qTrk1;` - so a Lambda (one strange quark, two light) gets 0 where
    the physical ratio is 0.5 and the `(1 - 0.4*sRatio)` suppression is skipped. Only a particle
    with at least as many strange quarks as light ones gets a non-zero ratio. Nothing this cascade
    reaches is strange, so it cannot bite here.
  * **`G4XAqmElastic` raises an area to the power 1.5.** `0.39 * powA(sigmaTot, 1.5)` with
    `sigmaTot` about 2.7e-24 mm^2 gives 1.7e-36, and the `if (sigma > sigmaTot) throw` that
    follows can therefore never fire - asserted, over all 2,100 points. The number is meaningless
    alone and is only ever used as a ratio, where the units and the 0.39 cancel. MEASURED: that
    ratio is exactly 1 for every species pair this channel accepts, because every meson here is a
    pion and every baryon a nucleon or a non-strange resonance, so forcing the factor to 1 changes
    none of the 900 cross sections. It is still computed, because a release that gave BIC a kaon
    would make it stop being 1.

**And the kaon and hyperon channels are not reachable because they are commented out.**
`G4CollisionMesonBaryonToResonance`'s constructor carries 11 Lambda and 7 Sigma
`G4ConcreteMesonBaryonToResonance` channels inside a `/* ... */` block in 11.1.1. The brief's
question - whether QBBC's species set reaches the kaon and hyperon channels - has the answer that
nothing does, because the code that would is not compiled.

### V108: the detailed-balance machinery is dead code in the binary cascade, and so is GenerateIso3

`G4XResonance::CrossSection` is three statements:

    sigma  = table->GetValue(sqrtS,dummy);
    sigma *= IsospinCorrection(trk1,trk2,isoOut1,isoOut2,iSpinOut1,iSpinOut2);
    if (trk1.GetDefinition()->IsShortLived() || trk2.GetDefinition()->IsShortLived())
       sigma *= DetailedBalance(trk1,trk2, isoOut1,isoOut2, iSpinOut1,iSpinOut2, mOut1,mOut2);

and the third never runs. `G4XResonance` is constructed in exactly ONE place -
`G4ConcreteNNTwoBodyResonance`'s constructor - and that class's `IsInCharge` is

    if (trk1.GetDefinition()==thePrimary1 && trk2.GetDefinition()==thePrimary2) return true;
    if (trk1.GetDefinition()==thePrimary2 && trk2.GetDefinition()==thePrimary1) return true;

with `thePrimary1` and `thePrimary2` always a proton or a neutron, because that is what all six
`G4CollisionNNTo*` constructors pass. So both entrance tracks are stable, always, and:

  * **`G4VXResonance::DetailedBalance` is never called**, and with it
    `G4DetailedBalancePhaseSpaceIntegral`, whose only caller it is. That class's twenty-five
    tables of 120 - 3,000 numbers - are dead code in the binary cascade.
  * **`G4VXResonance::IsospinCorrection`'s short-lived branch is never taken**, and with it
    `G4Clebsch::GenerateIso3` - the function docs/RISK.md V106 refuses for reading uninitialised
    stack. Nothing in the cascade reaches it.
  * **`G4VXResonance::DegeneracyFactor` is never called**, since only those two branches call it.

What is left of the isospin correction is `weight / pWeight`, a ratio of two Clebsch-Gordan
weights, and that IS live: measured, dropping it changes every one of 4,950 cross sections, and
computing `pWeight` for a pn entrance instead of pp changes them too.

**Could a resonance ever be an entrance track?** Only through the meson-baryon ELASTIC channel,
whose `IsInCharge` is by parton count and accepts a pion on a Delta (docs/RISK.md V107) - and that
channel goes through `G4XMesonBaryonElastic`, not `G4XResonance`. The three other ways in are all
closed: `G4GeneralNNCollision::IsInCharge` demands two nucleons;
`G4ConcreteMesonBaryonToResonance::IsInCharge` compares `G4ParticleTypeConverter` generic types,
and a Delta's is `D1232` where a proton's is `NUCLEON`; and `G4CollisionNStarNToNN`, which exists
precisely to put a resonance in the entrance channel, is registered by nothing - like
`G4CollisionPN`, it is included only by itself.

The port keeps all of it. `dbi_phase_space_integral` and its 25 columns are transcribed and
checked bitwise against Geant4, because they are one `AddComponent` away from being live and
because a reader who found them missing would have to re-derive why. What is NOT kept is
`GenerateIso3`'s sampling branch, which is refused for its own reasons in V106 - and this entry is
the second, independent reason nothing needs it.

### V109: the collision tree's channel totals come from a 32-point cache, and the nesting changes them

`G4CollisionNN`'s eight components decide which channel a nucleon-nucleon collision takes:
`G4CollisionComposite::FinalState` evaluates all eight partial cross sections, throws one uniform
against their sum, and hands the collision to the first component the running sum passes. Two of
the eight - the elastic ones - answer directly. The other six do not.

**None of the six resonance composites has a cross-section source**, so
`G4CollisionComposite::CrossSection` takes its other branch and answers from a BUFFER: the sum of
its components evaluated once on the fixed 32-point kinetic-energy grid
`G4CollisionComposite::theT`, cached against the pair of particle definitions, and linearly
interpolated forever after. So a resonance-production cross section in the binary cascade is never
evaluated at the collision's own energy. It is read off a 32-point piecewise-linear cache built
from 0.01 GeV to 100 GeV.

**And two of the six have a middle layer, which is not a detail.** `G4CollisionNNToNDeltastar` and
`G4CollisionNNToDeltaDeltastar` each hold nine `G4CollisionNNToNDelta1600`-style children, one per
Delta* multiplet, and each child is itself a bufferless composite - so those two totals are a
buffer of a sum of buffers. That matters because `G4CrossSectionBuffer::CrossSection` ends with

    if(y1<0.01*CLHEP::millibarn) result = 0;

a floor on the LEFT NODE, applied once per buffer. A Delta* multiplet whose own buffered node is
below 0.01 mb contributes exactly zero to its parent, where the same channels summed in one
buffer would have contributed their sum. MEASURED: flattening the middle layer - summing the
children's raw nodes instead of their buffered values - changes the N Delta* and Delta Delta*
partials by up to 5.4e+07 relative and moves the selected channel at 2.9 GeV. Removing the floor
alone changes them by 6.7e+08.

Three more things in `G4CrossSectionBuffer::CrossSection` that are not interpolation, all
reproduced and all measured:

  * **Above the whole grid the cross section is ZERO**, not the last node. The search breaks on
    the first node past `sqrts`; with none, `x1, y1, x2, y2` stay at their declaration values
    `(1, 0, 2, 0)` and the result is `0 + (sqrts-1)*0/1`. At the last node itself - where `>` is
    false for every entry - the same thing happens, so a collision at exactly the top grid point
    has no resonance cross section at all. Clamping to the last node instead changes the node
    values by 2.0e+09 relative.
  * **Below the grid it extrapolates BACKWARDS** along the line through nodes 0 and 1.
  * **The kinetic energy goes on the LIGHTER particle**, in either track order - Geant4's own
    comment says why ("A.R. 28-Sep-2012 Fix reproducibility problem"). For an np pair that is the
    proton. Putting it on the first track instead moves the grid by 6.8e-4 and the partials by
    4.8e-2.

The port builds the same three grids (pp, nn, np) and the same nesting, into a caller-owned
`NNChannelBuffers` - 6.9 kB, which is cascade state and does not belong in a frame.

**One composite is not built by the template its five siblings use.**
`G4CollisionNNToDeltaDelta`'s constructor lists an explicit `GROUP6` of Delta(1232) x Delta(1232)
pairs - a different set, and a different order, from `MakeNNToDeltaDelta`'s `GROUP10`. Building it
from the template gives 310 concrete channels instead of 306 and moves the selected channel. The
port lists the six out, and `tests/test_bic_imr.cu` asserts the total is exactly 306 and that
every one of them balances charge - the check `G4CollisionComposite::Resolve` does at construction
time and prints to `G4cerr`.
