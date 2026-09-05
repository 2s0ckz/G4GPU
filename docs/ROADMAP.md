# Roadmap for the overnight run

Ordered so each stage compiles and passes before the next starts, and so the later
user-facing work has foundations to sit on.

## 1. Geant4 data files, and the last six physics items

Scope, after the clarification: values that Geant4 itself compiles into its libraries as C++
arrays may be transcribed the same way here - Ziegler coefficients, the Urban MSC grid, ICRU
73 shells, PSTAR/ASTAR curves. What must be read at runtime is the real data *files* under
`share/Geant4/data`. Each transcribed table names its Geant4 source file and symbol at the
top of its header, so the two categories stay distinguishable.

- [x] `src/host/g4data.cuh` resolves datasets the way Geant4 does - the dataset's own
      environment variable first (`G4LEDATA`, ...), then a scan of candidate roots, each
      candidate validated by a sentinel file in the exact subdirectory needed
- [x] missing data is fatal, not a warning (see RISK.md S2)
- [x] Rayleigh angular distribution generalised from 10 hand-picked elements to all Z ≤ 100
- [x] PSTAR (74 materials), through Geant4's own not-a-knot spline - was a reported 29.3% gap,
      now 0.0001% (tools/regen_stopping.sh, tests/test_bragg.cu)
- [x] ASTAR (74 materials) - was 19.3%, now below 1e-6 for alphas
- [x] Ziegler 1988 molecular stopping - transcribed, then **deleted as unreachable**.
      G4BraggModel's eleven-compound formula list is a strict subset of the twelve formulae
      G4PSTARStopping resolves, so a table always wins and Geant4 never reaches its own
      molecular branch. What is implemented instead is the reachable behaviour: PSTAR resolves
      a material by name and, failing that, by chemical formula. Measured against Geant4 with
      a material built for the purpose - see the note at the end of this file.
- [x] ICRU90 (3 materials). `G4EmParameters::SetUseICRU90Data(true)` now exists and, when set
      before initialisation, replaces PSTAR/ASTAR with ICRU 90 for G4_AIR, G4_WATER and
      G4_GRAPHITE - which is what G4BraggModel does, resolving iICRU90 before iPSTAR. Off by
      default, as in Geant4. The six tables agree with `G4ICRU90StoppingData` to 2e-16 over
      1206 points from 1 eV to 20 GeV, and the switch moves water's stopping power by up to
      1.5% (about 1% at the Bragg peak), which is the reason the option exists.
      `tests/test_icru90.cu` checks both halves: the tables, and that the model reaches them.
- [x] `G4ionEffectiveCharge`, **both** branches. The helium fit agrees with Geant4 exactly -
      0.000e+00 over 606 points - and Ziegler's heavy-ion form to 1e-14, for nine ions from
      lithium to uranium over 1 keV to 100 GeV in six materials
      (ref/oracle/ion_charge.csv, tests/test_ion_charge.cu). It needed the material's Fermi
      energy, which `Material` now carries, and `G4Pow`'s approximate cube root, which
      `src/data/g4pow.hh` now reproduces - an exact power is 1e-5 away from what Geant4
      computes and the comparison is at 1e-9.
- [x] WentzelVI second moment - **closed by reading it, not by writing it.**
      `G4WentzelOKandVIxSection::ComputeSecondTransportMoment` returns `0.0` in 11.1.1; it is
      not virtual and has one definition. So `z2 == 0`, `prob2` comes out negative, and the
      `prob2 > 0.0` guard makes the branch it selects unreachable. Turning `useSecondMoment`
      on changes nothing. There is a flag, a setter, a getter, a per-material table built at
      initialisation and an interpolating accessor, around a function that returns zero.
      RISK.md O10.
- [x] `G4ScreeningMottCrossSection`'s Mott/Rutherford ratio, which is the single-scattering
      rejection function `G4WentzelOKandVIxSection` uses for e+- - it builds one
      unconditionally for those two species, "Mott corrections always added", and keeps the
      analytic Rutherford-plus-spin expression for everything else. So for electrons and
      positrons this was not a refinement of the angular distribution; it was the
      distribution.
      Agreement is **exact** - 0.000e+00 over 17136 points, eight elements from hydrogen to
      uranium, 10 keV to 1 GeV, both charges. Three pieces had to be right at once: the
      2790-coefficient table from `G4MottData.hh`; the relative-system beta, which is not the
      lab beta and needs the target nuclear mass (92 values, dumped from the oracle because
      Geant4 gets them from an AME12 table); and the double polynomial. The oracle cannot dump
      beta - private, no accessor - so the fcost = 0 column separates a beta error from a
      polynomial error. `tests/test_mott.cu` also checks that the sampler *takes* the branch:
      the acceptance rate on gold differs from the analytic one by 4.6% at 100 keV.
      What is still absent is the *cross section* half of that class, which no standard
      physics list reaches through WentzelVI.
- [ ] `fUseDistanceToBoundary` step limiter. Live, but only when selected: it is
      `min(tlimit, ComputeGeomLimit(...)/facgeom)` at a geometry boundary, and
      `ComputeGeomLimit` is the distance to the next boundary along the track. WentzelVI's
      default is `fUseSafety`, which is what B1 gets, so this needs a stepping-algorithm
      switch before the branch has anything to select it.

## 2. Geometry

- [x] layer-index model: no mother/daughter, higher layer wins on overlap, layer 0 is the
      world, everything outside the world is cut
- [x] all solid primitives - 18 of them, validated against 120000 rays of Geant4 G4VSolid answers
- [x] boolean union / subtraction / intersection
- [x] triangle meshes: STL / OBJ / PLY import, a BVH, exact containment and safety, and
      cadmesh's own interface (see the note at the end of this file)
- [x] rotations as well as translations

## 3. Geant4-shaped API and project layout

- [x] `G4double`, `G4int`, `G4ThreeVector`, `G4RotationMatrix`, `G4String`, units, constants
- [x] `G4VUserDetectorConstruction`, `G4VUserPrimaryGeneratorAction`, `G4UserRunAction`,
      `G4RunManager`, `G4UImanager`, `G4NistManager` (309 NIST materials), `G4SDManager`
      with `G4MultiFunctionalDetector` and the primitive scorers
- [x] the full action chain: `G4VUserActionInitialization`, `G4UserEventAction`,
      `G4UserSteppingAction`, `G4Accumulable`, `G4UnitsTable` / `G4BestUnit`, `G4cout`,
      `G4RunManagerFactory`, `QBBC`, `G4VisExecutive`, `G4UIExecutive`, `Randomize.hh`
- [x] example B1 rebuilt as `examples/B1/{exampleB1.cc, include/, src/, *.mac}` - 0.09 sigma
      from Geant4, 2.71M events/s

## 4. Sources

- [x] beam (rectangular / elliptical cross-section), point + angular spread, isotropic shell,
      distributed volume source
- [x] particle type, fixed energy or imported spectrum CSV
- [ ] radioactive source by nuclide and activity - captured and emitted isotropically at the
      stated energy; real decay sampling is not done

## 5. Visualisation

- [x] control bar: event count, Run, Reset (src/host/g4view.cu)
- [x] text output panel, colour-coded by severity
- [x] macro files, `/run/beamOn` and friends (src/g4/G4UImanager.hh)

## 6. Model-building GUI

- [x] same renderer (src/host/g4builder.cu)
- [x] menus: File, Insert, Boolean, Scoring, View
- [x] element / material / solid sidebars, with colour swatches, a picker, and a voxel-class
      sublist under each voxel volume
- [x] source sidebar with a list, add/delete/enable, all five source kinds, spectrum import,
      and nuclide/activity fields; physics sidebar with the process toggles and the range cut
- [x] gizmo editing after the 3D Builder pattern - click to select, drag an axis handle
- [x] save to a compilable project directory - checked by build_all.bat, which compiles and
      runs what the builder wrote
- [x] run in place

## Cleanup, throughout

- [x] drop `b1_vis` and the legacy `transport.cuh` thread-per-event path
- [x] one build entry point: `build_all.bat` builds, tests, and runs the dose check, and
      propagates failure from any of the three

## Note: voxel volumes

Follow `G4NestedPhantomParameterisation` rather than placing a volume per voxel: one regular
grid, a material index per cell, and the material resolved by cell index at lookup time.
Under the layer model that becomes a single `kVoxelGrid` solid - a box carrying nx*ny*nz cell
material indices - traversed by a 3D DDA. A 512^3 CT is then one volume in the scan that
locate() does, not 134 million, and the per-voxel material lookup is an array index.

The GUI's "flat detector" (a voxelised box given width/height/depth and a row/column count)
is the same solid with a coarser grid and a scorer attached.

## Note: solid / logical / physical, and anchoring

Keep Geant4's three-level split - it fits the layer model better than it fits Geant4's own
mother/daughter tree, and it is what veteran users expect:

- **solid** (`G4Box`, `G4Tubs`, ...) - shape and dimensions only.
- **logical volume** - a solid plus material, vis attributes, sensitivity, cuts. Defined once,
  placed any number of times.
- **physical volume** (`G4PVPlacement`) - one placement of a logical volume.

`G4PVPlacement` keeps Geant4's exact signature, mother logical volume and all:

```cpp
G4Sphere* solidInterior = new G4Sphere("Interior", 0, 57.5*cm - 1e-6*m,
                                       0.*deg, 360.*deg, 0.*deg, 180.*deg);
G4LogicalVolume* logicInterior = new G4LogicalVolume(solidInterior, Air, "Interior");
new G4PVPlacement(0, G4ThreeVector(0,0,0), logicInterior, "Interior", logicWorld, false, 0, 0);
```

The mother argument is not ignored and it is not a layer in disguise: it *derives* the layer,
as `mother.layer + 1`, and the placement's transform composes with the mother's. So an existing
DetectorConstruction pastes in and works, and the scene that results is the one the layer model
would have been handed. The world is layer 0 because it is the one placement with a null mother.

An extra overload takes an explicit layer in place of a mother, for what the hierarchy cannot
express: two volumes that overlap on purpose, or a solid straddling what would have been a
mother boundary. That is the only new thing to learn, and only for those who want it.

Repeated placement then works exactly as in Geant4: one `G4LogicalVolume` for a detector
element, N placements at different transforms, each with its own layer.

**Anchoring** is an authoring-time relationship, not a runtime one. A placement may name
another placement as its anchor; moving or rotating the anchor carries its dependents with it,
composing about the anchor's origin. The scene is flattened to absolute world transforms
before upload, so the navigator never sees an anchor and the stepping loop stays a flat scan.
That keeps grouped drag-and-rotate in the GUI without reintroducing a transform hierarchy into
the hot path. Anchors form a DAG that must be acyclic; the flattener checks it.

## Where this stands, and what is left

Done and checked by `build_all.bat`: the layer geometry with all 18 primitives, the three
boolean operations and triangle meshes (120,000 rays against Geant4's own G4VSolid answers,
plus `tests/test_mesh.cu`); voxel volumes with a DDA traversal (`tests/test_voxels.cu`); the
Geant4-shaped API with example B1 at 0.09 sigma from Geant4, through its full action chain;
generalised sources; the viewer with its control bar, shared with any example whose macro says
`/vis/open`; and the model builder, whose saved project - now including a voxel volume and an
imported mesh - is compiled and run as part of the pipeline.

Not done, in the order they matter:

- **Radioactive sources.** A nuclide and an activity can be entered, and are emitted as an
  isotropic source at the stated energy. Real decay sampling needs the RadioactiveDecay and
  PhotonEvaporation data, which `g4data.cuh` can already locate.
- **Hadron and ion transport.** The cross sections and stopping powers are there and tested
  (`tests/test_hadron.cu`, `test_bragg.cu`, `test_nuclear_stopping.cu`); nothing steps a proton
  through the geometry. `/run/beamOn` refuses a species the transport does not carry rather
  than treating it as a gamma - which it silently did until this was noticed. This is why
  example B1's proton runs are commented out in its macros.
- **The one physics item still open** of the six in section 1: `fUseDistanceToBoundary`, which
  is a stepping-algorithm option nothing selects by default. Of the other five, four are done
  and measured and one - WentzelVI's second moment - turned out to be dead code in Geant4
  itself (RISK.md O10). None of the six affects example B1; all affect ions, low-energy
  hadrons or high-energy e+-, and are therefore behind the item above.
- **A spatial index.** `locate()` scans every volume, which cost about 8% on B1's four volumes
  and will dominate a scene with hundreds. A BVH over volume bounding boxes is the fix, and
  `geometry/bvh_build.hh` is now most of it - the mesh work needed the same tree.
- **A mesh as a boolean operand.** Refused at construction, with a message. The boolean engine
  classifies intervals from a bounded list of surface crossings and a mesh can exceed it;
  truncating would give a solid of quietly the wrong shape. Doing it properly means an
  interval-based boolean walk rather than a candidate list.
- **Per-process agreement with Geant4.** `g4dose -verify-processes` now proves every switch
  changes the answer, which is what RISK.md A3 was missing. What it does not do is check the
  *size* of each change against Geant4 with the same process disabled. That needs seven more
  reference runs.
## Note: mesh transport, as built

A CAD import is transported against its triangles, not approximated by its bounding box.

- `src/geometry/mesh_io.hh` reads STL (binary and ASCII), Wavefront OBJ and ASCII PLY - the
  same set `christopherpoole/cadmesh` reads without its optional ASSIMP dependency. One
  implementation, used by the GUI's importer, by `G4TessellatedSolid` and by `CADMesh.hh`.
- `src/g4/CADMesh.hh` presents cadmesh's own interface, so an existing detector description
  written against that library compiles here:
  `CADMesh::TessellatedMesh::FromSTL(path)`, `SetScale`, `SetOffset`, `GetSolid()`.
  Its `TetrahedralMesh` is not implemented, and the reason is not a missing dependency: its
  purpose is to speed up navigation through Geant4's `G4TessellatedSolid`, which the BVH does
  directly, so the tetrahedralisation has nothing to buy.
- `src/geometry/bvh_build.hh` builds the tree on the host - median split on the axis with the
  widest spread of triangle centroids, two children allocated together because the device
  layout stores one index for both. Not a surface-area heuristic: an SAH tree is perhaps 20%
  faster to traverse and considerably more code, and the thing this has to beat is testing
  every triangle.
- `src/geometry/mesh.cuh` is the device side: Möller-Trumbore, the BVH ray walk, containment by
  parity with a point-derived ray direction and a re-cast when a hit lands on a shared edge,
  and `mesh_safety` - the exact distance to the nearest triangle, which is what lets a mesh
  volume cost what an analytic one costs. See RISK.md A4 for what happened without it.
- `G4TessellatedSolid` takes `G4TriangularFacet` and `G4QuadrangularFacet` as Geant4's does;
  `SetSolidClosed(true)` is what builds the BVH, computes the bounding box, and computes the
  enclosed volume by the divergence theorem. A mesh placed without it is refused, because a
  solid with no surface is one every track passes straight through.

Two limits, both refused rather than approximated:

- **The mesh must be closed.** Containment is a parity count, and an open surface makes that
  meaningless - a point can be inside along one ray and outside along another. Nothing here can
  detect an open mesh reliably (a watertightness check needs edge adjacency, which an STL's
  duplicated vertices destroy), so it is the caller's to guarantee. The import prints the
  enclosed volume, and a volume near zero or near the whole bounding box is the symptom.
- **A mesh may not be a boolean operand.** `G4UnionSolid` and its siblings refuse one at build
  time with a message. The boolean engine classifies intervals from a bounded list of surface
  crossings; a mesh can exceed it, and a truncated list gives a solid of quietly the wrong
  shape.

What checks it, and why each check exists:

| check | what only it can catch |
|---|---|
| `tests/test_mesh.cu` box section | a wrong ray-triangle test, parity rule or tree - a mesh of a box *is* the box, so the comparison is exact |
| `tests/test_mesh.cu` BVH vs brute force | the tree specifically, on 2256 triangles: bit-identical distances |
| `tests/test_mesh.cu` volume convergence | the divergence-theorem volume, by refining the tessellation three times and requiring the deficit to quarter |
| `tests/test_mesh.cu` navigation section | `locate`, `step_to_boundary` and the safety through a three-volume scene, against the analytic trapezoid |
| `g4dose -compare B1 B1mesh` | the upload path - every host-side check passes with a triangle pool that was allocated and never copied |
| builder selftest | the reader, the sidecar the generated project writes, and the loader that reads it back |

## Note: the NIST stopping tables, and a dead branch in Geant4

`G4BraggModel::DEDX` looks like a three-way choice per material:

1. one of the 74 NIST materials -> the tabulated PSTAR proton stopping power;
2. one of eleven compounds recognised by *chemical formula* -> its own ICRU 49
   parameterisation;
3. otherwise -> the per-element Ziegler fit.

Branch 2 is unreachable. `G4PSTARStopping::Initialise` matches a material by name and, failing
that, by chemical formula against **twelve** formulae - and G4BraggModel's eleven are a strict
subset of them. So any material whose formula G4BraggModel would recognise, PSTAR has already
resolved to a table, and `iPSTAR >= 0` returns first.

That was established rather than argued: `ref/dump/g4dump.cc` builds a material named
`CustomMolecularWater` with the formula `H_2O`, and Geant4 gives it G4_WATER's *tabulated*
stopping power - `pstar_index` is set and `bragg_dedx` equals `pstar_dedx` on every one of its
337 rows in `ref/oracle/bragg.csv`.

So what is implemented is the reachable behaviour, and the eleven-compound parameterisation was
transcribed, measured to be unreachable, and deleted. Carrying forty lines of correct
transcription that nothing can call is worse than a documented absence: the next person to read
it has to work out for themselves whether it is a missing wire or dead code.

The rest of it:

- `tools/extract_stopping.sh` reads the tables out of `G4PSTARStopping.cc`,
  `G4ASTARStopping.cc` and `G4NISTStoppingData.hh` - the same files the library compiles.
- `tools/gen_stopping.cc` computes the not-a-knot second derivatives of Geant4's spline and
  emits `src/data/nist_stopping_names.hh` (names, formulae, the energy grid, the lookups) and
  `src/data/nist_stopping.hh` (the tables and their derivatives).
- `src/data/g4spline.hh` is `G4PhysicsVector::ComputeSecDerivative1` and `Interpolation`.
  Reproducing Geant4 *between* grid points needs it: the grids are 60 and 78 points across a
  Bragg peak.
- `tools/regen_stopping.sh` runs extract, **recompile**, generate and check in that order, and
  `tools/check_pstar.cc` compares every point against Geant4's own answers. Both of those
  steps exist because of a specific failure; see RISK.md O6 and S4.

A note on where the spline is *not* needed, because the obvious next target turned out to be a
false one: `src/data/photoelectric_data.cuh` used to say its tabulated branch was "interpolated
linearly where Geant4 uses a cubic spline". Geant4 enables that spline only for the `livermore`
data directory, and the default is `epics2017`, so the linear interpolation there is Geant4's
own. The 0.0004% agreement in docs/PHYSICS_PLAN.md had been saying so all along.
