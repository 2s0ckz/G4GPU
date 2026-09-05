# The second session

`docs/OVERNIGHT.md` covers the first. This is what changed after it, and the two things in it
worth arguing with.

## Example B1 is the whole example now

You said the project layout was the bare minimum of what you asked for, and pointed at
`geant4-v11.1.1/examples/basic/B1`. It was: three classes where Geant4 has six, and no
`CMakeLists.txt`, `README`, `exampleB1.in`, `run2.mac` or `init_vis.mac`.

`examples/B1` now has all of it, with Geant4's class names and Geant4's file layout:

    exampleB1.cc              G4RunManagerFactory, QBBC, ActionInitialization, G4VisExecutive
    include/ + src/
      ActionInitialization    Build() and BuildForMaster()
      DetectorConstruction    the geometry, and the sensitive detector
      PrimaryGeneratorAction  the gun
      RunAction               G4Accumulable, G4BestUnit, G4cout, the dose and its rms
      EventAction             per-event accumulation
      SteppingAction          filters to the scoring volume, adds the deposit
    exampleB1.in  run1.mac  run2.mac  init_vis.mac  vis.mac  init.mac
    CMakeLists.txt            cmake, with CUDA in place of find_package(Geant4)
    README                    what it is, and the three ways it differs from Geant4's
    build.bat                 the direct nvcc build the pipeline uses

The API those files need was mostly missing and is now there: `G4VUserActionInitialization`,
`G4UserEventAction`, `G4UserSteppingAction`, `G4Step` / `G4StepPoint` / `G4Track` / the
touchable, `G4Accumulable` and `G4AccumulableManager`, `G4UnitsTable` with `G4UnitDefinition`
and `G4BestUnit`, `G4cout` / `G4cerr` / `G4endl`, `G4RunManagerFactory`, `G4VModularPhysicsList`
and `QBBC`, `G4VisExecutive` and `G4UIExecutive`, `G4LogicalVolumeStore`, `Randomize.hh`,
`G4SteppingVerbose`, and `G4LogicalVolume::GetMass()`.

**The dose is unchanged: 427.499 pGy, 0.09 sigma from Geant4** - computed now through
`SteppingAction` to `EventAction` to `RunAction::AddEdep`, with `G4Accumulable`, rather than
read off the scorer. Bit for bit the same number by a completely different route, which is the
best evidence available that the chain is right.

### I was wrong about why the event actions could not exist

`docs/RISK.md` said a per-event host callback could not exist at 2.7 million events per second,
and that is why those three classes were missing. It cost 2.6%.

The reason the estimate was wrong is worth knowing, because it is not "I mis-measured". The
energy deposit was *already* per event on the device - the score buffer is one row per scorer
and one column per event in the batch - and the host already copied it back and walked it.
Calling the actions inside that walk is three virtual calls per event and no device work. What
genuinely cannot exist is a per-*step* callback, at 13 steps an event, each needing a `G4Step`
assembled from device memory. I collapsed "per event" and "per step" into one impossibility.

There is a dividend. Going through `EventAction` means the run's rms is now computed the way
Geant4's B1 computes it, from the second moment of the per-event deposit distribution. It comes
out at **0.870 picogray per 10,000 events, which is Geant4's own figure**. That is an
independent check on the physics and a stronger one than the mean: two transports can agree on
an average and disagree about the distribution behind it. See RISK.md A1.

### What a step is here, and the one thing to be careful about

`UserSteppingAction` is called once per event and per volume with a sensitive detector, with
that event's totals - not once per step. B1's stepping action adds up what it is given, and the
sum over an event of the per-step deposits is the event's deposit, so it is exact. An action
that took a maximum, or tested a threshold per step, or cared where inside the volume a step
happened, would not be. `G4Step::IsAggregated()` returns true so such an action can detect it.
The full statement is at the top of `src/g4/G4Step.hh`, in the file where someone writing a
stepping action will meet it.

## The .obj files

Those were object files - the intermediates from compiling `exampleB1.cc` and friends. In a
project that imports CAD meshes, a directory of files called `*.obj` is a genuinely bad place
to put them, so they are gone from there: object files now go to `out/B1`, generated projects
put theirs in their own `build/`, and the stray `.lib` and `.exp` files the linker leaves have
been redirected to `out/` as well. The repository root has nine build scripts and four
executables in it and nothing else.

## The viewer is shared, so `/vis/open` works from an example

`init_vis.mac`'s `/control/execute vis.mac`, and `vis.mac`'s `/vis/open OGL 1280x800`, used to
reach nothing unless you were running `g4view.exe`. The window, the GL context, the UI and the
render dispatch are now `src/render/vis_manager.cu`, compiled once into `out/vis_manager.obj` -
which is what Geant4 ships as `libG4vis`, for the same reason. `g4view.exe` is a thin `main`
over it, `G4VisExecutive::Initialize()` registers it as the `/vis/` handler, and
`G4UIExecutive::SessionStart()` runs its loop. So `exampleB1` with no arguments opens the
viewer on B1's geometry, with the run control bar, exactly as Geant4's does.

With no window open, `SessionStart()` falls back to an `Idle>` prompt on stdin, which is what a
Geant4 terminal session does.

## Mesh transport

An STL import used to be read, measured and placed as its bounding box. It is now transported
against its triangles. The pieces are listed in `docs/ROADMAP.md`; what matters here:

- STL, Wavefront OBJ and ASCII PLY are read - the set `christopherpoole/cadmesh` reads without
  ASSIMP. Thank you for the pointer: `src/g4/CADMesh.hh` presents that library's own interface,
  so `CADMesh::TessellatedMesh::FromSTL(path)->GetSolid()` compiles here unchanged.
- `G4TessellatedSolid` with `G4TriangularFacet` and `G4QuadrangularFacet`, as Geant4 has them.
  `SetSolidClosed(true)` builds the BVH and computes the enclosed volume by the divergence
  theorem.
- Containment is a parity count along a ray whose direction is *derived from the point*, not
  fixed. On a CAD model, where vertices sit on round coordinates and faces are axis-aligned, a
  fixed axis passes through shared edges constantly, and each such hit is a coin-flip in the
  parity. A hit landing within tolerance of an edge triggers a re-cast.
- Two things are refused rather than approximated: an open mesh cannot be transported (parity
  is meaningless on one), and a mesh cannot be a boolean operand (the boolean engine works
  from a bounded crossing list that a mesh can exceed). Both say so, and say what to do
  instead.

### The check that mattered, and what it found

`g4dose.exe -compare B1 B1mesh` runs example B1's geometry twice: once with the scoring volume
as a `G4Trd`, once as a twelve-triangle mesh of the identical trapezoid, same seed, same
source. A mesh of a trapezoid is not an approximation of it - the eight corners are the same
eight numbers - so the two must agree.

They did not. The doses were 1.26 sigma apart, which is nothing and would have been dismissed
as noise. The **track-step count was 42% higher**, which cannot be noise: the same geometry
cannot cost 2.8 million more steps.

The cause was the isotropic safety. A bounding box gives no lower bound on the distance from an
interior point to the surface, so a mesh's safety was zero - which is *sound*, and which
switches Urban MSC's step limitation off entirely. `mesh_safety()` now answers exactly, by
finding the nearest triangle through the BVH, and agrees with the trapezoid's closed form to
5e-15 mm.

### And a pre-existing bug it exposed

Reading the analytic safety in order to compare against it was the first time anyone had read
it. The default branch, used by every solid without a closed form, was
`0.5 * (bounding_radius - |q|)`. That is sound for a sphere and unsound for anything flat or
elongated: a `G4Tubs` with rmax 1 mm and dz 100 mm would have claimed a 50 mm safety on its
axis, where the wall is 1 mm away, and MSC would have taken a 50 mm step inside a 1 mm tube.

Nothing could have caught it. B1's three solids all have exact closed forms and never reach the
default; `tests/test_solids.cu` compares the *directional* queries against Geant4, not the
isotropic safety, and comparing two underestimates tells you nothing anyway. It is fixed with
an inradius-based bound, which is provably sound, and 0 - always sound - wherever the answer is
not obvious. See RISK.md G3.

## Two more checks in the pipeline

Both are new executables' worth of "this could fail", which is the only kind of check worth
having:

    g4dose.exe -compare B1 B1mesh -n 500000     the same geometry two ways must score the same
    g4dose.exe -verify-processes -n 200000      every physics switch must change the answer

The second closes RISK.md A3 - the eight GUI toggles that were stored, serialised, written into
generated projects, and never read. "Everything on reproduces Geant4" was the only evidence
they did anything. Now the scene is run once per switch and the run fails if any of them
changes neither the dose (beyond statistics) nor the step count (beyond a per cent). Note what
it does *not* assert: a change of a particular size. Rayleigh scattering deposits no energy and
multiple scattering conserves it, so demanding a large dose shift from those two would be
demanding the wrong physics - both change the step count enormously.


## Three of the six physics items

Item 1 on your list was the six remaining physics items. Three are done, and they were the
three that mattered most because they are the ones a *material* selects rather than a rare
particle.

`G4BraggModel::DEDX` - low-energy heavy-particle ionisation, the model that produces the
Bragg peak - makes a three-way choice per material:

1. one of the 74 NIST materials -> the tabulated **PSTAR** proton stopping power;
2. one of eleven compounds recognised by chemical formula -> the **Ziegler 1988 molecular**
   parameterisation;
3. otherwise -> the per-element Ziegler fit.

Only the third was implemented. The other two were reported as a "known gap" by
`tests/test_bragg.cu`, which measured it at **29.3%** for protons and **19.3%** for alphas -
in water, at the Bragg peak. All four of B1's materials are NIST materials, so that gap
applied to every one of them.

All three branches are now there, and the alpha equivalent (**ASTAR**) too:

| branch | before | now |
|---|---|---|
| PSTAR, 74 NIST materials | 29.3% gap | **0.0001%** |
| ASTAR, alphas | 19.3% gap | **0.0000%** (asserted below 1e-6) |
| ASTAR, other ions | - | 0.0009% (asserted below 5e-5) |
| Ziegler 1988 molecular | not implemented | deleted: unreachable in Geant4 too, see below |
| per-element Ziegler | 0.0001% | 0.0001% (unchanged) |

### How the tables got here, and why that is the interesting part

The data is hard-coded in Geant4 - it is the published NIST tabulation, not something computed
- so on your ruling it may be hard-coded here. `tools/extract_stopping.sh` reads it out of
`G4PSTARStopping.cc` and `G4ASTARStopping.cc`, the same files the library compiles, so "every
number is the number Geant4 has" is a fact rather than a hope. `tools/regen_stopping.sh` runs
the whole chain and checks it.

Two things beyond copying the numbers were needed, and each was worth more than the copying:

**Geant4's spline.** The tables go into a `G4PhysicsFreeVector` with spline interpolation, and
the grids are 60 and 78 points across a Bragg peak. Linear interpolation between them is low
in the curvature and high on the flanks. `src/data/g4spline.hh` is Geant4's own
`ComputeSecDerivative1` - the *not-a-knot* end condition, not the natural spline anyone would
write from memory, which differs by a few per cent in the first and last intervals - and its
`Interpolation`. The second derivatives are computed once by the generator and emitted as
data, because a tridiagonal solve has no place on a GPU.

That machinery was written expecting to point it at the photoelectric tables next, which
`src/data/photoelectric_data.cuh` said were "interpolated linearly where Geant4 uses a cubic
spline". They are not. `G4LivermorePhotoElectricModel::ReadData` enables the spline only when
the Livermore data directory is `livermore`, and the default is `epics2017`; the below-K-shell
vector has a comment saying "no spline" outright. So the linear interpolation there is exactly
Geant4's, and the file's own measurement had been saying so all along - `docs/PHYSICS_PLAN.md`
records that branch agreeing to **0.0004%**, which is not what a cubic spline replaced by a
straight line looks like on a coarse grid. The comments are corrected. A stated approximation
that does not exist is as much a misstatement as an unstated one, and this one nearly cost an
afternoon of removing something that was already right.

### What this changes about the answers, and what it does not

Nothing about example B1. B1 is 6 MeV gammas in water, tissue and bone; the secondaries are
electrons and positrons, which are not heavy particles and never reach this model. The dose is
**427.499 pGy, 0.09 sigma**, unchanged to the last digit - which is the right outcome and is
checked rather than assumed.

What it changes is any proton or alpha dose in a NIST material, which is most of proton
therapy and most of radiation-hardness work. Those are still not *transported* - see below -
but the stopping power they will be transported with is now Geant4's.


### And a branch of Geant4 that cannot be reached

The third item was "Ziegler 1988 molecular stopping" - `G4BraggModel`'s own ICRU 49
parameterisation for eleven compounds, chosen by chemical formula when a material's *name* is
not one of the 74. It was transcribed, and then deleted, because it cannot be reached.

`G4PSTARStopping::Initialise` matches a material by name and, failing that, by chemical formula
against **twelve** formulae - and G4BraggModel's eleven are a strict subset of them. So any
material whose formula G4BraggModel would recognise, PSTAR has already resolved to a table, and
the table wins.

That is not a reading of the code; it is measured. `ref/dump/g4dump.cc` now builds a material
called `CustomMolecularWater` with the formula `H_2O`, and Geant4 gives it G4_WATER's
*tabulated* stopping power on all 337 of its rows. What is implemented is therefore the
reachable behaviour - name, then formula, to a table - and `tests/test_bragg.cu` checks it on
that material to 0.0001%.

Forty lines of correct transcription that nothing can call would have been worse than a
documented absence: the next person to read it has to work out whether it is a missing wire or
dead code. The same reasoning retired the GUI's ionisation toggle (RISK.md A3).

### The three that are left, and why they are behind hadron transport

- **ICRU 90** (3 materials: water, air, graphite). Geant4 prefers it over PSTAR when
  `G4EmParameters::SetUseICRU90Data(true)`, which is **off by default** - so it is not what
  the default answer uses, and implementing it without the switch would silently change the
  default. It needs the switch as much as the data.
- **`G4ScreeningMottCrossSection`** and **`G4ionEffectiveCharge` for Zi > 2**: single-scattering
  and effective-charge corrections for ions heavier than helium.
- **WentzelVI's second moment and `fUseDistanceToBoundary`**: a multiple-scattering refinement
  that matters for hadrons near a boundary.

All three affect particles this transport does not step through the geometry. Hadron and ion
transport is the item they are behind: the cross sections and stopping powers are here and
tested (`tests/test_hadron.cu`, `test_bragg.cu`, `test_nuclear_stopping.cu`, and now PSTAR and
ASTAR), but nothing carries a proton from volume to volume. `/run/beamOn` refuses a species it
cannot transport - which it did *not* do until today; the seeding kernel picked a track buffer
by species with the gamma buffer as its default, so `/gun/particle proton` produced a plausible
dose for a run of 6 MeV gammas.


## Smaller things

- **`/gun/particle proton` was silently transported as a gamma.** The seeding kernel picked a
  track buffer by species with the gamma buffer as its default, so any species the transport
  does not carry became a 6 MeV gamma and produced a plausible dose. It is refused at
  `/run/beamOn` now, with a message that says what is and is not transported. This is why
  example B1's proton runs - which Geant4's `run1.mac`, `run2.mac` and `exampleB1.in` all
  contain - are present but commented out, with a note, rather than quietly deleted.
- **A run seed.** `G4RunManager::SetRandomSeed` sets the Philox key the whole run derives from,
  so two runs can be independent samples of the same scene. `g4dose -seed` uses it. The host
  engine in `Randomize.hh` does not affect the transport and says so when you set it.
- **The generated project builds in 40 seconds instead of four minutes**, by linking
  `out/transport_run.obj` rather than compiling `transport_run.cu` itself. Its voxel-cell
  sidecar is now found relative to the executable rather than the working directory, so it runs
  from anywhere; and it gets a `G4VisExecutive`, so a generated project can be looked at.
- **The builder selftest now inserts a voxel volume and imports a mesh**, so the pipeline
  compiles and runs a generated project containing both, exercising the sidecar writers and
  both loaders. It was previously three primitives, none of which needed a sidecar.
- **`exampleB1 vis.mac` printed FATAL eleven times and exited 0.** Fixed three ways; see
  RISK.md S5. The exit code was the interesting one: `ApplyCommand` returns the error count and
  Geant4's `exampleB1.cc` discards it, so mine did too.

## Item 4: the ion effective charge, and the version mistake behind it

`G4ionEffectiveCharge` has two branches. The helium one was transcribed; the heavy-ion one was
not, because it needs the material's Fermi energy and `Material` did not carry one. Both are in
now, and they agree with Geant4 **exactly** for helium - 0.000e+00 over 606 points - and to
1e-14 for nine ions from lithium to uranium over 1 keV to 100 GeV in six materials.

Getting there turned up four things worth more than the branch itself.

**The reference tree was the wrong Geant4.** `D:/g4gpu/reference-geant4` is **11.5.0**. The
oracle links **11.1.1**. Every transcription I did yesterday came from the wrong version, and
no measurement could tell me, because PSTAR, ASTAR, the material names, the 78-point grid, the
not-a-knot spline and the two Fermi tables are all byte-identical between the two. It surfaced
only when I tried to *call* something: `G4ionEffectiveCharge::ComputeCharge` exists in 11.5.0
and does not exist in 11.1.1, and the dumper would not compile.

Everything has been re-verified against 11.1.1's own source, and `tools/g4src.sh` now resolves
the tree by reading `G4VERSION_NUMBER` out of both the source and the install the oracle links,
and refuses a mismatch:

```
g4src.sh: VERSION MISMATCH - refusing to transcribe from the wrong Geant4.
          source  D:/g4gpu/reference-geant4     G4VERSION_NUMBER 1150
          oracle  ...geant4-v11.1.1-install     G4VERSION_NUMBER 1111
```

RISK.md O7. The uncomfortable part is that `docs/PLAN.md` had said "Geant4 11.5.0 reference
clone" since the day it was cloned. A fact in a document is not a guard.

**A constant that was more accurate than Geant4's.** The helium branch came out 1.5e-8 off, and
every candidate explanation was a formula. It was `amu_c2`: the port had the current CODATA
931.49410242 and CLHEP 11.1.1 has 931.494028. Sweeping the family found four more - the
classical electron radius and the Compton wavelength were each 7.8e-8 out, because CLHEP
*derives* them from its own e, mu0 and c rather than tabulating them, so there is no literal to
copy. `r_e` appears squared in every ionisation prefactor.

There is now a `constants.csv` in the oracle and a `tests/test_constants.cu` that compares both
of the port's constant sets against it at one bit. Fourteen files had spelled out the fine
structure constant; they all call `units::` now. RISK.md O8.

**Mass fractions that do not sum to one.** Found in passing: `G4_AIR`'s four NIST fractions sum
to 0.999999, `G4Material::FillProperties` normalises them, and the port did not - so its air
held 1e-6 fewer electrons per mm^3 than Geant4's, and every macroscopic cross section in air
was 1e-6 low. `test_vs_oracle` had been printing that ratio as `1.0000` for weeks. RISK.md O9.

**G4Pow is not a cube root.** `G4Pow::A13` is a third-order Taylor expansion about a tabulated
point, and `std::pow(a, 1.0/3.0)` is up to 1e-5 away from what Geant4 computes. The heavy-ion
screening length goes through `A23(1-q)`, so an exact power fails a 1e-9 comparison by four
orders of magnitude and the failure looks like Ziegler's formula being wrong. `src/data/g4pow.hh`
reproduces the approximation. `Material::inv_a23`, which feeds every WentzelVI cut-off angle,
now goes through it too.

The new checks:

```
== G4IonisParamMat, per material ==
  quantity               points      worst dev
  Zeff                        6     4.441e-16
  Fermi energy                6     5.551e-16
  <A^-2/3>                    6     6.661e-16
  L-factor                    6     2.220e-16
  electron density            6     2.220e-16     (was 1.000e-06)

== G4ionEffectiveCharge ==
  helium branch, charge                   606     0.000e+00
  helium branch, (q*corr)^2               606     0.000e+00
  heavy-ion branch, charge               4848     9.770e-15
  heavy-ion branch, (q*corr)^2           4848     3.331e-15
```

The correction has no consumer in the port yet - it is used only by
`G4EmCorrections::BuildCorrectionVector`, which needs ion transport - and the code says so
where it is defined. It is measured now anyway, because the alternative is writing it later
from a formula nothing has ever checked.

## Two more things the pipeline can now fail on

- **`InsertBoolean` would pick a mesh as a boolean operand.** Found because my own selftest did
  it: the builder walks backwards to the newest free solid, so importing a CAD file and then
  subtracting a cylinder silently made the *mesh* the first operand, and the run died at
  `Build` with a FATAL. The refusal was right and it arrived far too late. `CanBeBooleanOperand`
  now rejects a mesh or a voxel grid when the operand is *chosen*, with a message, and the
  selftest picks its operands by name instead of by index.
- **The builder selftest could log a failure and pass.** `findstr "selftest:"` was satisfied by
  the failure line. The pipeline now requires the specific success line and treats
  `selftest: FAILED` as fatal.

## And a tool, because the same edit broke the same file twice

Every edit that could not go through a whole-file write went through a line splice built from
`grep -n`. When the grep finds nothing the bound is empty, `$((B+1))` is 1, and the tail clause
appends **the entire original file** - silently, exit zero. `build_all.bat` came out as 258
lines where 146 were expected, twice. `tools/splice.sh` does the same job, refuses an empty or
out-of-range bound, and prints the resulting line count so a duplication is visible. RISK.md S6.

## Items 5 and 6: ICRU 90, and the Mott ratio

Both done, and both measured. That closes five of the six physics items; the sixth turned out
not to exist (above), and what remains from that list is a stepping-algorithm option nothing
selects by default.

**ICRU 90.** `G4EmParameters::SetUseICRU90Data(true)` now exists and does what Geant4's does:
set before initialisation, it replaces PSTAR/ASTAR with the ICRU Report 90 tables for G4_AIR,
G4_WATER and G4_GRAPHITE. `G4BraggModel` resolves its ICRU 90 index *before* the PSTAR index
and returns as soon as it has one, so this is a replacement, not a correction, and the port
takes the branches in the same order.

The six tables agree with `G4ICRU90StoppingData` to **2e-16** over 1206 points from 1 eV to
20 GeV, which covers the sqrt extrapolation below the first grid point and the clamp above the
last. The switch moves water's stopping power by up to 1.5%, about 1% at the Bragg peak - which
is the reason ICRU 90 exists, and which `tests/test_icru90.cu` requires, because a switch that
changed nothing would pass a test that only checked the tables.

One real trap. Geant4 stores these stopping powers as `G4float` and widens them:

```cpp
static const G4float e0_proton[57] = { 119.70f, ... };
data->PutValues(i, e[i]*CLHEP::MeV, ((G4double)dedx[i])*fac);
```

so the number it splines is `(double)(float)119.70`, not `(double)119.70`. All six tables were
6e-8 out - uniformly, at every energy, in every material, which is the fingerprint of a type
and not of arithmetic. RISK.md O11, and the same lesson as the constants: the reference's
precision is part of the reference.

**The Mott ratio.** `G4WentzelOKandVIxSection` builds a `G4ScreeningMottCrossSection` for
electrons and positrons unconditionally - the comment in Geant4 is "Mott corrections always
added" - and uses its Mott/Rutherford ratio as the rejection function when it samples a single
scatter. Everything else keeps an analytic Rutherford-plus-spin expression, which is what this
port had for all species. So for e+- this was not a refinement of the single-scattering angular
distribution; it *was* the distribution.

Agreement is **exact**: 0.000e+00 over 17136 points, eight elements from hydrogen to uranium,
10 keV to 1 GeV, both charges. Three things had to be right simultaneously:

- the 2790-coefficient table from `G4MottData.hh`, extracted with the count asserted;
- beta - which is not the lab beta. `SetupKinematic` works in the projectile-nucleus relative
  system, with the relativistic reduced mass of Martynenko and Faustov, so beta depends on the
  *target nuclear mass*. Geant4 takes that from `G4NucleiProperties`, an AME12 table with the
  Cameron mass-excess formula behind it; what this needs out of all that is 92 numbers, so the
  92 numbers are dumped from the oracle;
- the double polynomial, order 5 in `beta - 0.7181228` and order 4 in `sqrt(1 - cos theta)`.

The oracle cannot dump beta - it is private with no accessor - so the split is done by angle
instead. At `fcost = 0` the ratio collapses to one coefficient row and beta, so a failure there
is the beta chain and a failure only at `fcost > 0` is the angular polynomial. That cost nothing
to arrange and would have saved an afternoon had either been wrong.

`tests/test_mott.cu` also checks that the sampler *takes* the branch, by sampling the same
scatter twice off the same random stream with the flag on and off:

```
== the sampler takes the Mott branch ==
       E/MeV    Mott accept       analytic     differ
         0.1         0.4450         0.4663      4.58%
         0.3         0.4456         0.4611      3.36%
           1         0.4476         0.4548      1.58%
          10         0.4456         0.4468      0.28%
```

Converging as beta rises toward 1, which is what a fit centred on beta = 0.718 should do.

## A polycone the generator could not write

The pipeline's "the saved project compiles and runs" check found a real one. The builder can
create twenty-two shapes; the emitter could write twenty of them, and for the other two -
polycone and polyhedra - it emitted

```cpp
auto* solidCone = nullptr /* Polycone is not yet emitted */;
```

which is a compile error in a file the user is told is a working project. It had been that way
for as long as the feature existed, because no automated run had ever saved a model containing
one. The selftest grew a polycone yesterday for an unrelated reason - a boolean and a polycone
are the two shapes whose *rendering* needs the shared solid store - and the next pipeline run
failed on it.

Both are emitted now, through a generated `MakePolycone`/`MakePolyhedra` helper, and the
default case no longer emits anything: `CanEmitSolid` is checked before a file is opened, so an
unemittable shape is refused **by name** instead of being handed to the compiler. RISK.md S7.

`InsertBoolean` got the same treatment: it walked backwards to the newest free solid, so
importing a CAD file and then subtracting a cylinder made the *mesh* the first operand and the
run died at Build with a FATAL. A mesh or a voxel grid is now rejected when the operand is
chosen, with a message.

## `/vis/verbose` before `/vis/open` opened the viewer at the wrong size

Writing `tsg_offscreen.mac` - Geant4's B1 ships one, so this one should too - turned up a
defect in the viewer's command handling. `ApplyVisCommand` opens the window for any command it
does not recognise, so that a macro missing `/vis/open` still works. But `/vis/verbose
confirmations` on the line *before* `/vis/open` therefore opened the viewer at the default
size, and `/vis/open TSG_OFFSCREEN 1200x1200` then found it already open and returned true.
The pictures came out at 1440x880 with no complaint. Geant4's own `vis.mac` starts with
`/vis/open`, which is why nothing had noticed.

The commands that need no window are now answered before anything opens. And when the window
manager gives less than was asked for - a client area cannot exceed the screen, so 1200x1200
comes back as 1200x1061 - the viewer says so, because a picture silently rendered at the wrong
size is the kind of thing someone measures against.

The offscreen commands themselves are implemented for the one format this renderer can
produce, and refuse the rest with the reason: the gl2ps vector formats would need a vector back
end that a GPU rasteriser does not have. `build_all.bat` runs the macro and requires all seven
pictures to exist.

## The prerequisite for a spatial index, and what it turned up

`locate()` scans every volume and `step_to_boundary()` scans every volume that outranks the
current one. A bounding sphere per volume is the cheap way to make both better than O(n), and
`to_local` makes it exact rather than approximate: it is a rotation about `xform.trans`, so it
preserves length, and a local sphere of radius R is exactly a world sphere of radius R about
the placement. One field, two early-`continue`s.

The radius would come from `solid_half_extent`, whose comment read:

```
/// A bound on how far the solid reaches from its own origin.
```

It is not. It returns `max(dx, dy, dz)` for a box - a bound on each *coordinate*. A box of
(30, 40, 50) returns 50 and reaches 71.4 mm at the corner. Correct for its one caller, which
sizes an axis-aligned sampling box; a 43% underestimate for the next one, which would have made
the navigator skip volumes it should have entered and shown up as a wrong dose rather than a
crash.

So the sphere is not built yet. What is built is the check that would let it be:
`tests/test_solids.cu` now samples 200,000 points per solid, over all thirty the oracle covers,
and requires every contained point to lie inside the cube of half-side `extent`. It holds
everywhere. It also prints the measured radius ratio, which is what a future bounding sphere
needs:

```
  solid              inside     extent  worst |x|  worst |p|  |p|/ext
  box                  2973         50      49.99      66.96    1.339
  trd                  2063         50         50      68.22    1.364
  trap                   24        200      49.95      54.02    0.270
  tet                    60         80      33.84       38.3    0.479
```

Worst 1.364 against sqrt(3) = 1.732, so the conversion has room - and the Trap and Tet bounds
are 4x and 2.4x looser than they need to be, which matters because a loose bound is a slow one.
Two of those cases are admitted guesses in the source ("a plane offset bounds the inradius, not
the circumradius"), which is precisely why 200,000 samples each was the right thing to spend an
hour on rather than reading the arithmetic and believing it. RISK.md G4.

That is where this stops: the measurement is in, the contract is enforced, and building the
tree on top of it is a clean piece of work rather than a gamble on a comment.

# The builder, after your review

Twenty-one items, then five more, then two bug reports. What follows is what changed, grouped by
what it was rather than by the order it arrived in.

## The four real bugs

**Typing stopped working.** One shared string, four fields, all reset every frame - two of them
fought over it and every character was gone by the next frame. RISK.md U1.

**A custom material scored NaN.** No mean excitation energy meant `log(0)` in the
density-effect parameters and NaN in every dE/dx after it. Geant4 derives one; this now does
too, matching Geant4 to 4e-16 against a material added to the oracle for the purpose. The same
hunt found the ionisation prefactor defined twice in one namespace, and an oracle dumper whose
geometry was sized for five materials. RISK.md O12.

**The terminal filled with a repeating message.** A panel was calling `ScoredMass` per scorer
per frame, and that runs a 200,000-sample Monte Carlo - and prints a note - when a scored volume
is overlapped. Cached, and reworded: it is not an error. RISK.md U2.

**The sphere rendered with cartesian planes.** The normal was a central difference of a binary
containment test, so it could point in 26 directions and quantized a smooth surface into 26
patches bounded by the coordinate planes. Quadric shapes and booleans now have analytic
normals. RISK.md G5.

## The two you found after that

**A 45% box rendered opaque.** The compositing walk re-found the volume it had just entered -
`dist_in` from inside returns about zero - and repainted its front face until the alpha
saturated. One test fixed it: skip volumes the ray is already inside. RISK.md R1.

**Same-layer overlaps are now refused, not warned.** Two volumes on one layer have no rule
deciding which owns the shared space; the tie-break is insertion order, so the dose would depend
on the order the detector was built in. A run is refused, the pairs are listed with the fraction
shared, and the selftest proves the refusal fires - it puts two volumes on one layer, asks for a
run, and fails the build if a run happened. RISK.md G6.

## The layout

Left sidebar: ELEMENTS / ISOTOPES, MATERIALS, SOLIDS. Right sidebar: SOURCES, SCORERS, and
Physics options as a button at the foot. Between them at the bottom: a scrollable terminal with
a thumb and a "line n of m" indicator, and the run row. Both sidebars scroll. The Scoring menu
is gone; Insert has a lateral submenu with all seventeen primitives and separate CAD and voxel
imports; View opens the visualization attributes.

## The editing model

Elements, materials and sources are **forms**: the fields edit a working copy, `Add` appends it,
`Update` writes it back. Nothing enters the model until you say so, which is what makes "enter
the parameters, then press Add" possible. Solids are deliberately still live - they are dragged
in the view, and a drag that only took effect on Update would be unusable.

Click a selected row again to rename it. NIST names complete over all 300 with Tab, and
completing an element symbol fills in Z and A. Mass and atom fractions are editable next to each
component with a running share and a note when they do not sum to one - Geant4 normalizes them,
and so does this. "Add element / isotope", "Assign material" and "Assign scorer" open pickers
rather than acting on a selection made in another list; a button that does nothing until you
have selected something elsewhere, and does not say which list, is not a button.

Delete works from the keyboard, on whatever is selected, and never while a field has focus. The
last source is deletable - what a *run* needs is checked when a run is asked for.

## Visualization

Opacity per solid, in the color popup, with a preview over light and dark; it reaches the
renderer as `G4Colour`'s alpha, which Geant4 has always carried and this used to drop. Higher
layers now win in the picture as well as in the physics - the covered part of a lower volume is
not drawn, which is what makes a transparent shell over a target readable. Background and
per-species track colors are settable and are *not* saved with the model: they are how you want
to look at the detector, not what the detector is.

## Runs

Several enabled sources split a run by weight - `round(n * w_i / sum(w))` events each, summed -
with a separate RNG stream per source so two identical beams from different places are
independent samples rather than the same shower twice. A fixed split rather than a random choice
because it is stratified sampling, so it has *less* variance, and because it is reproducible.
The source list shows each share as a percentage before you run.

## The eighteen-item list

All eighteen are in. The ones with something to say about them:

**The world is asked for, not assumed** (8). Opening the builder, or File > New, now puts up a
dialog for the world's shape - box, cylinder or sphere - its size and its units, before anything
else. There is no Cancel, because there is no scene without a world. The old default was a
silent 1 m box, which meant a user building a 3 m room got tracks killed at a wall they never
chose, with nothing saying so.

**No default beam** (14). A new model has no source. A default one is a beam at an energy and a
position nobody picked, and the first run either uses it by accident or has to be understood
before it can be replaced. An empty list says what the state is. A run with no source is refused
with that sentence.

**Imported meshes are anchored on their own centre** (6). This turned out to be a plain bug as
well as a request: the placement was set to the bounding-box centre while the triangles kept
their file coordinates, so an imported part appeared offset from its file position by that
centre - doubled for a part modelled about its own middle, invisible for one modelled about a
corner, which is why nobody noticed. `CenterMesh` now moves the triangles and puts the placement
where they were, so the part is where the file says and the local origin is the middle of it.
Voxel volumes were already centred on their own origin.

**The voxel import asks** (10). A raw file has no header: nothing in it says how many cells
there are, what type they hold, or how big one is. It still guesses a cube from the file size
when told nothing, but a 512x512x120 CT is not a cube, and a wrong guess produces a scrambled
volume rather than an error. The dialog asks for the shape, the sample type and the voxel size
with units, and shows the resulting extent as you type - "0.5 mm voxels" and "512 of them" is a
25.6 cm phantom and neither number says so alone. The reload path now takes the dimensions from
the document rather than re-guessing, which was the same bug in the other direction.

**Anchors** (12). Any volume can be anchored to another: its position and rotation are then read
in that volume's frame, so moving or turning the anchor carries everything anchored to it - a
phantom on a couch, a collimator stack, a gantry. Choosing an anchor rewrites the offset so the
volume does not move, which is the difference between "anchor to" and "teleport onto".

It is a *placement* chain only. Which volume owns a piece of space is still the layer index, and
stays a separate question: a volume can be anchored to one it does not sit inside. The
arithmetic lives in `builder/model.hh` because the project writer needs the same answer - two
implementations would be two chances for a saved project to place a volume somewhere other than
where it was run, which is the one thing `build_scene.hh` exists to prevent. The generated C++
emits the chain as the `rotateX/Y/Z` calls a person would have written, and the composed
position as a number.

**Text fields have a caret and a selection** (17). Click to place it, drag to select, shift with
the arrows and Home/End, double-click for all of it, Ctrl+A/C/X/V through the Windows clipboard.
Typing or Backspace with a selection replaces it. The text scrolls horizontally to keep the
caret in view, so a long value is editable at its far end - which is why the focused field does
not ellipsize: an ellipsis would make the character index stop matching the pixel position and
every click after it would land somewhere else.

**Voxel gridlines** (9). A voxel volume is one box to the geometry, so nothing about its surface
said how it was divided and a 64x64x1 detector drew as the same grey slab as a 512-cubed CT. The
cell boundaries are drawn by darkening the shade near one, in the two axes tangent to the face
that was hit. The legibility rule matters as much as the drawing: the line width is set in
pixels and converted to millimetres at the hit distance, and once a cell is smaller than a few
pixels the lines are not drawn at all - without that, a real CT zoomed out is a black rectangle.

## Per-voxel scoring, and the day it cost

Item 11: score each cell separately, as `G4PSEnergyDeposit3D` does. It works - the cells sum to
the volume total the same run reports, to 5.6e-16 - and `docs/VOXELS.md` sets out how this
voxel scheme relates to Geant4's, since the answer is "not nested parameterisation" and the
divergences are worth knowing before comparing a number.

The part worth recording is that turning it on made every score in the selftest read exactly
zero, including a scorer on a volume the phantom has nothing to do with, and that this was not a
bug. The selftest ran 25 events. Twenty of them through a 30 mm water cube is a fraction of one
expected interaction, so the dose was one rare event or nothing, and *which* was decided by
where the RNG stream landed - and per-voxel scoring shifts that stream, because it adds steps
and Philox is seeded from the step number. `2.81066 MeV` and `0` were two draws from the same
distribution.

Getting there went through a host-side ray probe (kept: `ProbeRay`, which prints the whole path
a ray takes with materials and distances), device-side counters, and two real bugs that were not
the cause - a stale `G4SDManager` attachment map surviving rebuilds, and a zero-length step in
`voxel_step` only reachable with the new stepping. Being led to real bugs by a false alarm is
not the same as the alarm being right. See RISK.md V1.

The fix is that the run size and the trajectory cap are no longer the same number. The selftest
had shrunk the *run* to 25 to keep the *picture* legible; the picture now draws at most 25
events' trajectories and the run is 4000. `build_all.bat` gained two checks: `dose 0 pGy`
anywhere in the builder selftest is fatal, and the per-voxel check must be seen to have run. The
pipeline already had the first of those for the generated project, which runs 10000 events, and
not for the builder's own run, which ran 25.

## The ten-item list

### The generated project was not a Geant4 project

You said it looked minimal, and it was: three files in `src/` and `include/` against a Geant4
example's six, no `CMakeLists.txt`, no `README`. That is not a cosmetic difference. There was
no `ActionInitialization` - the class Geant4's own documentation tells you to edit first - and
no `EventAction` or `SteppingAction`, so there was nowhere to put per-event or per-step code.
Somebody opening a generated project to *do* something with it found no hook and had to write
the file themselves.

All six classes are emitted now, close to `examples/B1`'s, plus a `CMakeLists.txt` and a
`README` that says where to edit. `main` registers one `ActionInitialization` instead of wiring
actions by hand. Two things had to be added to the emitter to make the classes real rather than
decorative: `DetectorConstruction::GetScoringVolume()`, which SteppingAction filters on, and
`RunAction::AddEdep`, which EventAction feeds.

The result is a cross-check that did not exist before. A generated project now prints:

    dose1            177.605 MeV  rms 25.21   dose 1053.9 pGy
    stepping         177.605 MeV  rms 25.21   (SteppingAction)

The first is the device-side scorer. The second is the `SteppingAction -> EventAction ->
RunAction` chain adding up the same run independently. They agree to every printed digit.

### One gun, several sources

The emitted `PrimaryGeneratorAction` used to fire the first enabled source and carry a comment
apologising for it. The builder splits a run between sources by weight, so a two-beam model ran
one way in the builder and another way in the project saved from it - the exact disagreement
`build_scene.hh` exists to prevent, in the one file that did not go through it.

It now emits a cumulative weight table and picks per event:

    const G4double kSourceCdf[2] = {0.8, 1};

Literals, not a computation, so the weights are visible in the generated file and can be edited
there. Both forms - single source and several - go through one `WriteOneSource`, so the two
cannot describe the same source differently.

### Custom scorers, and a hook that is actually called

A scorer can be flagged custom, and is then written out as its own subclass, one file pair per
scorer, with a hook to edit.

The hook took some deciding. You described it per deposit, and it cannot be: steps happen
inside a device kernel with tens of thousands of tracks in flight, and there is no host callback
per step - the same constraint that makes the `G4Step` handed to a stepping action an aggregate.
A hook that claimed to filter per deposit and was never called would be worse than no hook. So
`Accept` is **per event**, and the generated header and the base class both say so bluntly: a
threshold written there is a trigger on the event, not on the step, and an override meaning the
second will quietly compute the first.

Wiring it needed care in one place. A stock scorer's total comes from the device's per-event
array summed in a fixed order, and *that* is the number example B1's 0.0929-sigma agreement is
measured on. Routing it through a host loop, even one that changes nothing, would re-associate
the additions and move the last digits of a validated number. So `IsFiltered()` gates a separate
path taken only by scorers that declare a hook.

The claim that comes with the flag - that it changes the code and not the answer, until the
generated file is edited - is checked rather than asserted. The selftest runs the same seed both
ways:

    selftest: a custom scorer reports what the stock one does (72.1836 MeV, 0e+00)

Exactly zero difference.

### The smaller ones

**Voxel values: indices or properties** (1). A voxel file is a grid of numbers and does not say
what they are. A segmentation's 1 and 2 are categories to assign materials to; a density map's
1.02 and 1.06 are quantities, and reading the second as the first gives one "material" per
distinct density - thousands of unassignable classes, which looks like a successful import until
you try to use it. The dialog asks; properties is refused, in the dialog *and* in `ImportFile`,
so a second caller cannot route a density map into the classifier. The class sub-list in the
SOLIDS panel is gated on the answer.

**Collapsible sub-lists** (2), and a colour swatch on every solid row rather than only on voxel
classes (3). The swatch is its own click target - it opens the colour picker without selecting -
which is the fastest way to tell two boxes apart in a view.

**The drawn-history cutoff is a field** (4). It had been a constant since the run size and the
trajectory cap were separated, so a run that drew less than you wanted gave no way to ask for
more.

**Gridlines fade rather than drop** (8). The old rule dropped them below a few pixels per cell,
which meant a CT seen from across the room drew as a featureless slab - the one thing the
gridlines exist to show. Drawing them at full strength there paints every pixel dark instead.
So the strength scales with how much of a cell one line would cover: a line where the cells are
visible, a wash where they are not. Nothing is dropped; the toggle still turns them off.

**"Reset" rather than "Clear tracks"** (9) - it drops the trajectories *and* the last run's
numbers, and naming half of what it did was the half people were surprised by.

**RGB as numbers** (10), 0-255, and opacity as a percentage. A slider cannot be typed into, read
off exactly, or matched to a value from somewhere else - and matching is most of what colour
editing is. The swatch, which is what the slider was really providing, is still there.

## Primary generation, rewritten

You said the generated `PrimaryGeneratorAction.cc` gave no developer freedom - that a Gaussian
beam spot could not simply be coded into it - and that `GeneratePrimaryVertex` should not be an
option but the only architecture. Both are right, and the second is the part that mattered.

### What it was

`GeneratePrimaries` was called **once per run**. It configured a `G4ParticleGun`, and the gun's
configuration - a `Source` record - was handed to the device, where a seeding kernel sampled it
per event. That is fast, and it made the set of things a primary could be *closed*: whatever
distributions the gun happened to offer, and nothing else. A rectangular beam, an ellipse, an
isotropic shell, a cone, a box, a spectrum from a CSV. Anything not on that list could not be
written at all.

The clearest evidence of how bad that was: `examples/B1` could not be Geant4's B1. Upstream
draws two random numbers per event to place the beam spot; this project's copy had to replace
them with a beam cross-section on the gun, and carried a comment explaining the substitution.
The example that exists to demonstrate fidelity had been edited to fit the engine.

It also produced a bug that reached you twice. A generated project with two weighted sources
emitted a `G4UniformRand()` switch inside `GeneratePrimaries` - the obvious thing to write, and
wrong, because being called once per run meant it chose one source for the *whole* run. Two
million events fired from whichever beam the first random number landed on.

### What it is

`GeneratePrimaries(G4Event*)` is called once per event, on the host, as Geant4 calls it. It
builds `G4PrimaryVertex` and `G4PrimaryParticle` objects; `G4RunManager::RunEvents` collects
them into a flat array of `g4gpu::Primary`, uploads it a batch at a time, and
`seed_from_primaries` reads it. The device no longer samples anything about a primary.

`G4ParticleGun::GeneratePrimaryVertex(event)` samples whatever the gun is configured for - via
`sample_primary`, the same `__host__ __device__` function the kernel used to call, so the
distributions are unchanged down to the order the random numbers are drawn in. Which means the
conveniences still work *and* anything set before the call wins:

    fParticleGun->SetParticlePosition(G4ThreeVector(G4RandGauss::shoot(0, 3*mm),
                                                    G4RandGauss::shoot(0, 3*mm), -20*cm));
    fParticleGun->GeneratePrimaryVertex(anEvent);

`examples/B1` is back to Geant4's own idiom, two `G4UniformRand()` draws per event, with nothing
left to explain. And multi-source in a generated project became a plain per-event weighted draw
- which is what was written the first time, and was wrong only because the architecture was.

### What it cost, measured

**3.7%.** 785.9 ms against 757.9 ms for 2,000,000 events of B1. The estimate before doing it was
10-15%; the host call and the upload are cheaper than that against 26 million track-steps.

The one price worth naming is in the seeding kernel. The old one knew every primary was the same
species, so a track's slot was its event index and no counter was needed. A generator may now
fire a gamma on one event and an electron on the next, so each species' buffer is appended to
with an atomic. One `atomicAdd` per event against a whole shower is not a cost worth designing
around.

### And the check that should have existed

B1's agreement with Geant4 is the headline claim of this project, and until now the pipeline
**printed** it and moved on. `0.0929 sigma` went into a log for a person to read; a drift to five
sigma would have passed every check and been reported as ALL OK.

`tools/check_sigma.ps1` now gates it at 3 sigma - on the *agreement*, not on the exact value,
because a changed random stream legitimately moves the number and demanding a fixed one would
make every such change look like a physics error. Moving generation to the host moved B1 from
0.0929 to 0.0154 sigma, which is a different draw from the same distribution and not an
improvement in anything.

The gate earned its place immediately. A splice in this change dropped `run_.n_events = n_events`
from BeamOn; B1's `EndOfRunAction` returns silently when the event count is zero, so the example
exited 0 having printed nothing at all. Every other check passed. The gate is what said so.

## Not done

**`.obj` face colors** - withdrawn.

**Protons, neutrons, ions.** The models are there and measured: Bragg, Bethe-Bloch, ICRU73QO,
nuclear stopping and WentzelVI all agree with Geant4 to 1e-6 or better, and the ion effective
charge and the Mott ratio joined them at 1e-14 and *exactly*. What is missing is a stepper for a
heavy charged particle - its own range table, delta-ray production, the true-to-geometric step
conversion - and a transport-level comparison against a Geant4 proton run to prove it. Neutrons
need hadronic interactions, which is a much larger thing. The source field warns before Run
rather than after.

**Nested parameterisation**, and the rest of what `docs/VOXELS.md` sets out: per-cell densities
rather than per-cell material indices, and a step boundary at every cell by default rather than
only where the material changes. The last of those is a deliberate choice, not an omission - it
is what makes a CT affordable - but it means a comparison against Geant4 that counted *steps*
rather than energy would disagree.

**A spatial index for `locate()`.** Still a linear scan over volumes. The prerequisite work is
done and the bound is measured; the scene sizes reached so far do not need it.

**A mesh as a boolean operand.** `G4BooleanSolid::BuildWith` refuses one in Geant4 for the same
reason it is refused here, so this matches rather than lags - but it is a thing the builder
cannot do and says so rather than doing it wrongly.

## Where it stands

`build_all.bat`: **ALL OK**. 32 tests, example B1 at 0.0929 sigma against a 2M-event Geant4
reference, mesh transport at 0.00 sigma, all seven physics switches alive, seven offscreen
pictures, the same-layer refusal proved, per-voxel scoring agreeing with the volume total to
5.6e-16, a custom scorer agreeing with a stock one exactly, five dialogs photographed, and a
generated project - now the full Geant4 file set - that compiles and runs from a directory that
is not this one.
