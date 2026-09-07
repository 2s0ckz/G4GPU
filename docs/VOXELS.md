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

## Seeing one

A segmented phantom is drawn cell by cell, not as the box that bounds it. `render_geometry`
walks the cells along each ray with `geom::VoxelWalk` and composites each class's colour front
to back, shading by the face the cell was entered through - without the face normal every cell
shades identically and the result is fog with an outline.

Two consequences of compositing front to back:

* **Index 0 is imported invisible.** In every segmentation convention 0 means "nothing here",
  and it is also the commonest value, so an opaque 0 is a solid block with the anatomy inside
  it.
* **Every other class is imported at a tenth.** Opaque cells show only the first surface a ray
  meets, which for a segmentation is the skin. At 0.1 about thirty cells of tissue accumulate
  to 96%, so the interior reads through and the outline still reads as a surface.

Both are defaults, in `ClassifyVoxels`; the class list in the SOLIDS panel edits them per
class. The eye beside each row toggles a class off entirely, which is the practical way to
look inside a segmentation with two hundred of them - hide all but the three you want.
**Material for all** assigns one material to every class at once, which is how a phantom whose
classes are nearly all soft tissue gets set up before the exceptions are corrected.

### Importing one

**Insert > Voxel volume** collects everything about the file and reads nothing until **Import**
is pressed. That ordering matters: a raw file carries no shape, no type and no voxel size, so
those are answers the dialog has to ask for - and while choosing the file WAS the import, they
had to be right before the file browser opened, and getting one wrong meant deleting the solid
and starting again.

The dialog holds the voxel file, an optional colour table, the cell shape and type, what the
values mean, the voxel size, and which colormap to use.

### Colormaps

Without a colour table, the classes are coloured from a map, stretched over the range of the
values: a class at value `v` sits at `(v - lowest) / (highest - lowest)` along it. So the map
is fitted to the indices actually present rather than sampled out of a fixed 256, and it is
never cycled - no class repeats another's colour, because no two classes share a value.

By VALUE and not by position in the list, which is what "the range of the index values" means
and which has a use: two phantoms labelled by the same convention come out with the same
colours whether or not both contain every structure. The cost is that a sparse numbering -
three classes at 0, 1, 2 and two more at 700, 701 - gives each cluster nearly one colour. A
colour table is the answer for a phantom numbered like that.

`distinct hues` is the default and is what the importer did before there was a choice: the hue
circle swept once, which puts adjacent classes far apart and is the best of them for telling
one organ from its neighbour. The seven others - viridis, plasma, inferno, magma, turbo, jet,
gray - represent a *continuum*, and they are there because a phantom's indices are often
ordered (outwards from the skin, or by tissue density) and reading it as a continuum is then
the clearer picture. They are held as a handful of stops with linear interpolation
(`ColormapStops`), not as 256-entry tables: nine stops reproduce the shape of these maps to
within a couple of levels, and a 256-entry table has no way to check any one of its numbers.

The sweep is 330 degrees, not 360, because the circle closes and 360 is the same red as 0.
See docs/RISK.md V18 for how that was found.

### The colour table

A real segmentation has dozens or hundreds of classes and arrives with a table naming them.
**Colour table...** in the import dialog reads one, and `ReadVoxelColourTable`
(`src/builder/import.hh`) is the parser. It is applied *after* classifying, over the top of
whatever the colormap chose, so a table covering some of the classes leaves the rest
distinguishable rather than grey. One row per class:

```
# comment; ; and // also start one
1, 255, 0, 0, 255
2  0.0  1.0  0.0  0.5
```

* Fields are separated by commas, whitespace, or both. `index r g b` is enough; a fifth field
  is alpha, and a row without one leaves the class's opacity alone.
* **The first column is the value in the segmentation file**, not the row number - that is what
  a phantom's own table is keyed by, so a class list sorted or filtered differently still takes
  the same colours. For a volume classified into *bands* (a CT, `VoxelKind::kContinuous`) there
  is no such external number, so there the first column is the band's position, 0 first.
* **The scale is decided per value by how the number is written.** A field holding `.`, `e` or
  `E` is read on 0-1; anything else on 0-255. So `0.5` and `128` are both a half, and the two
  forms may be mixed in one file. The edge is real: `1` is 1/255 and `1.0` is full. The log
  line reports how many values were read each way, so a table read the wrong way is visible
  there rather than only in the picture.
* A row whose index names no class, and a line of four-plus fields that are not all numbers -
  a header, say - are counted and reported rather than silently dropped. A table from a
  different phantom looks exactly like one that partly matches, and `atof` would have read
  `index,r,g,b,a` as class 0 painted black.

The class colours are baked into the solid pool when the scene is flattened, so they reach the
picture on the next rebuild, like a colour picked by hand.

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
