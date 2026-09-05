# Voxel volumes: what this is, and how it differs from Geant4

Short answer to the obvious question: **this is not nested parameterisation.** There is no
`G4PVParameterised`, no `G4VNestedParameterisation`, no `G4PhantomParameterisation` anywhere in
the tree. A voxel volume here is one solid with a raster inside it, and the navigator walks
that raster directly.

It is closest in spirit to Geant4's **regular navigation** - `G4RegularNavigation` over a
`G4PhantomParameterisation` - which is the fast path Geant4 itself uses for CT phantoms, and
which also skips the daughter tree and steps the raster. The differences from *nested*
parameterisation, and from regular navigation, are set out below because they are the kind of
thing that decides whether a number here is comparable with a number from Geant4.

## What a voxel volume is

A `Solid` of type `kVoxelGrid`:

```
p[0..2]  half extent of the whole box, mm
p[3..5]  cell counts nx, ny, nz
a        offset into the scene's shared cell-material pool
```

The cells themselves are one flat `short` array for the entire scene - `FlatScene::pool
.voxel_cells` - holding a material index per cell, `x` fastest. A volume's cells are the slice
starting at `a`. So the cell index is *global across voxel volumes*, and that global index is
what stands in for Geant4's copy number.

Everything else about the volume is ordinary: it is placed with a transform like any solid, it
takes part in the layer rule like any solid, and `dist_in` / `dist_out` / `inside` treat it as
a box. What differs is only what is *inside* it.

## How a step inside one works

`geom::voxel_step` (`src/geometry/voxels.cuh`) is an Amanatides-Woo DDA in the solid's own
frame. It returns the distance to the next point where the **material** changes, or - when the
volume is being scored per voxel - to the next **cell boundary**.

Stopping only at material changes is the important optimisation and the main divergence from
Geant4: a homogeneous region of a CT is crossed in one step rather than one step per voxel.
Through soft tissue that is the difference between a few steps and a few hundred. It is exact
for transport, because nothing about the physics changes while the material does not - the
cross-sections, the range table and the density-effect parameters are all per material - but it
means **there is no step boundary at every cell**, which Geant4's regular navigation does have.

Two consequences follow, and both are visible:

* A per-voxel scorer needs a boundary at every cell, or a deposit that spans several cells has
  to be divided between them after the fact, by length, which is not how the energy was laid
  down. So attaching a `G4PSEnergyDeposit3D` switches that volume to stopping at every cell -
  `Volume::score_per_voxel`, set by the flattener - and the run gets slower by roughly the
  factor you would expect. Geant4 pays that cost always; here it is paid when it is asked for.
* Anything else that depends on step *boundaries* rather than on the material - a step-limiting
  process, a user stepping action counting steps - would see a different number of steps here
  than in Geant4 for the same phantom. Nothing in this project currently does, but a comparison
  against Geant4 that counted steps would disagree, and would be right to.

## What is not implemented

* **Nested parameterisation.** `G4VNestedParameterisation` lets a material depend on the copy
  number of an *ancestor* replica - the usual construction for a phantom sliced z, then y, then
  x, with the material read from the innermost index. There is no mother/daughter hierarchy in
  this project at all (overlap is resolved by layer index, see `geom::locate`), so there is
  nothing for a nested parameterisation to nest in. The flat global cell index carries the same
  information.
* **Per-cell densities.** A cell carries a material index, not a density. A CT import is
  classified into bands and each band becomes a material - `ClassifyVoxels` - which is what
  `G4PhantomParameterisation` does with `SetMaterialIndices`, and not what a continuously
  varying density would need.
* **A cell-level bounding structure.** `voxel_step` walks cells one at a time. For a 512-cubed
  grid crossed corner to corner that is up to ~1500 iterations in the worst case, bounded by
  `max_steps`. Geant4's regular navigation does the same thing.

## Scoring

`G4PSEnergyDeposit3D` (`src/g4/G4SDManager.hh`) is the per-cell scorer. Its `cells` vector is
filled after a run with energy deposit per cell in MeV, indexed by the global cell index. The
kernels accumulate into it with an atomic per deposit, keyed by the cell the step **started**
in - which is what Geant4's 3D scorers key by, the pre-step point's touchable.

The scorer's `total` and `total_sq` still hold the volume's sum, so the same scorer reports both
the distribution and the single number it would have reported without it. The builder's selftest
checks the two against each other: the cells must sum to the volume total, and they do, to
5.6e-16.

The map is written beside a saved project as `<volume>.dose` - a short text header and then one
number per cell, `x` fastest, so `numpy.loadtxt(...).reshape(nz, ny, nx)` gets the array back in
the order the phantom was imported in.
