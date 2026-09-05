# What changed overnight

**This is the first session. `docs/DAY.md` covers the second, which finished example B1's class
set, mesh transport, and a safety bug the mesh work exposed - and which corrects one claim made
here (RISK.md A1: the per-event actions do exist, and cost 2.6%).**

Read `docs/ROADMAP.md` for the item-by-item status and `docs/RISK.md` for the traps found and
what was done about them. This file is the short version, and the things worth knowing before
touching anything.

## Run it

```
build_all.bat            builds everything, runs the tests, the 2M-event dose check, the
                         mesh-vs-analytic comparison, the process-switch check, both
                         selftests, and compiles and runs the project the GUI wrote
g4view.exe               the viewer, with a run control bar
g4builder.exe            the model-building GUI
g4dose.exe -list         the scenes and processes the dose driver knows
g4dose.exe -compare B1 B1mesh -n 500000
g4dose.exe -verify-processes -n 200000
examples\B1\exampleB1.exe                  interactive: init_vis.mac, then the viewer
examples\B1\exampleB1.exe -n 2000000       batch
examples\B1\exampleB1.exe run2.mac         batch, from a macro
```

`build_all.bat build` stops after compiling; `build_all.bat test` stops after the tests.

## The headline

Example B1, rebuilt entirely through the Geant4-shaped API - `G4NistManager`, `G4Box`,
`G4Cons`, `G4Trd`, `G4LogicalVolume`, `G4PVPlacement`, `G4MultiFunctionalDetector` - agrees
with Geant4 11.1.1 at **0.09 sigma**: 427.499 pGy against 427.385 +/- 0.870, at 2.7 million
events per second.

The geometry is checked against **120,000 rays of Geant4's own G4VSolid answers**, across all
18 primitives and the three boolean operations. Twenty-nine of thirty solids agree to about
1e-13 mm. The thirtieth is `G4Hype`, where Geant4 is the one that is wrong - see RISK.md G1.

## Four things I got wrong, and how

Worth reading, because each is a pattern rather than a one-off. All four are in RISK.md with
the details.

**The build was already broken when you left, and I told you it passed.** `build_all.bat`'s
exit code was not propagating through the pipeline I ran it in, so a background build reported
success while `stepper.cuh` had an unresolved symbol. I then read the *stale* output file of an
earlier run and reported "24 tests pass, dose unchanged". Neither had happened. The pipeline is
now one script that builds, tests, runs the dose check, greps for `FATAL`, and propagates
failure from any of them.

**`G4LEDATA` on this machine points at a Geant4 10.7.3 install** whose G4EMLOW 7.13 has an
`epics2017` directory holding only pair-production data. The path resolver accepted it,
photoelectric and Rayleigh both failed to load, both were disabled with a `WARNING:` line, and
the run printed a dose of 425.759 pGy - low by 0.43%, which reads as ordinary disagreement
rather than as two missing processes. Missing data is `exit(2)` now, and dataset resolution
checks for a sentinel file in the exact subdirectory it needs.

**The GUI's physics toggles did nothing.** Eight switches, stored in the model, written into
the generated project, never read by the engine. Turning Compton off changed the dose by
exactly zero. They are wired now - except ionisation, which was *removed*: it is the continuous
energy loss along a step, and without it a lepton has infinite range, so no setting of that
switch produces a number worth having.

**I edited sources while the build was running**, twice, and both times the failure pointed
somewhere else: once at a filename that does not exist (a batch file read halfway through
being rewritten), once at a duplicate declaration in a file I had already fixed. A build is a
measurement of a particular state of the tree; changing the tree underneath it makes the
measurement meaningless in a way the error message does not reveal. See RISK.md S3.

## The one design decision you should check

`G4PVPlacement` keeps Geant4's exact signature, mother logical volume and all. The mother is
not ignored: it derives the layer, as `mother.layer + 1`, and the transforms compose. So an
existing `DetectorConstruction` pastes in and works.

An extra overload takes an explicit layer instead of a mother, for what the hierarchy cannot
express - volumes that overlap on purpose, or a solid straddling what would have been a mother
boundary. That is the only new thing to learn.

One trap preserved deliberately: the `G4RotationMatrix` argument is Geant4's, which is the
*inverse* of the intuitive rotation. `rot->rotateY(35*deg)` tilts the frame by +35 degrees and
therefore the object by -35. Fixing it would make code moved from Geant4 place things
differently here, which is worse than the confusion.

## What was not finished at the end of this session

Struck through where the second session finished it; see `docs/DAY.md`. Left here rather than
edited away, because what a list of remaining work looked like at the time is part of the
record.

- ~~**Mesh transport.**~~ Done: a BVH, a ray-triangle test, exact containment and an exact
  isotropic safety. It also exposed a safety bug in the analytic solids (RISK.md G3).
- **Radioactive decay sampling.** A nuclide and an activity can be entered and are emitted as
  an isotropic source at the stated energy. The GUI says so, in the panel, next to the field.
- ~~**The six physics items**~~: five done and measured, one closed by reading it.
  PSTAR/ASTAR (0.0000%), ICRU 90 (2e-16, behind `SetUseICRU90Data` as in Geant4),
  `G4ionEffectiveCharge` both branches (exact for helium, 1e-14 to uranium),
  `G4ScreeningMottCrossSection`'s Mott/Rutherford ratio (**exact**, 17136 points), and
  Ziegler 1988 molecular stopping, which is unreachable in Geant4 itself. WentzelVI's second
  moment is dead code there - `ComputeSecondTransportMoment` returns 0.0 (RISK.md O10). What
  remains from that list is `fUseDistanceToBoundary`, a stepping-algorithm option nothing
  selects by default. None of the six affects B1; all affect ions, low-energy hadrons or
  high-energy e+-, and behind hadron *transport*, which is still not implemented and is
  refused rather than silently done wrong.
- **A spatial index.** `locate()` scans every volume. That cost about 8% on B1's four volumes
  and will dominate a scene with hundreds. `geometry/bvh_build.hh` is most of the tree, and
  the missing prerequisite - a *sound* bounding radius per volume - is now measured and
  enforced: `solid_half_extent` bounds each coordinate and not the radius, so a sphere needs
  sqrt(3) times it, and `tests/test_solids.cu` checks that over all thirty solids
  (RISK.md G4).
- ~~**Checking that a disabled process changes the answer.**~~ Done:
  `g4dose -verify-processes` runs the scene once per switch and fails if any changes nothing.
  Checking the *size* of each change against Geant4 is still open.

## Performance

| | events/s | 2M events |
|---|---|---|
| Before the geometry rewrite | 2.91e6 | 688 ms |
| After (layers, transforms, 18 solids, booleans) | 2.71e6 | 738 ms |

About 8% slower. The recursive solid engine blocks inlining and needs a real device call stack
(`cudaDeviceSetLimit(cudaLimitStackSize, 16384)` - without it you get "illegal memory access"
from an unrelated API call, with nothing pointing at the cause). The O(n) `locate()` is the
part that will matter later.

Worth knowing: the generalised engine first measured 1280 ms and looked like the cost of the
abstraction. It was the batch size - 100,000 against the reference driver's 1,048,576. Matched,
it came in at 731 ms, marginally *faster* than the driver it generalises.

## Voxel volumes

Built on your `G4NestedPhantomParameterisation` hint: one volume carrying `nx*ny*nz` material
indices, not one volume per voxel, traversed by a 3D DDA. A step ends where the *material*
changes, so a homogeneous region costs one step rather than one per cell - `tests/test_voxels.cu`
confirms a uniform 64^3 grid is crossed in exactly one step, and that a checkerboard conserves
path length to 1.7e-13 mm.

`.raw`, `.bin` and uncompressed `.mat` import. `.h5` does not: HDF5 is a library dependency
this project does not take, and the importer says so and says what to convert to rather than
failing obscurely.

## Why there are still two B1 drivers

`b1_gpu_sched.exe` and `examples/B1/exampleB1.exe` compute the same thing by different routes.
The first has its own kernels and a hard-coded four-volume geometry; the second goes through
the Geant4-shaped API, the flattener and the general engine. They agree to 0.025%.

That duplication costs about four minutes of build time per pipeline run, and it is kept
deliberately. The API layer is new: a flattener that mis-composed a transform, or a material
table built one index off, would produce a scene that looks right and doses wrong. The
hard-coded driver shares none of that code and still says 427.6 pGy, which is what makes the
0.09-sigma agreement evidence about the physics rather than about the plumbing.

It should be retired once the API layer has its own geometry tests - most of that is already
`tests/test_navigation.cu` and `tests/test_solids.cu` - but not before.
