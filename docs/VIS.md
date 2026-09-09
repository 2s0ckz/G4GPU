# Visualization

Two applications over one GPU renderer and one immediate-mode UI:

| | |
|---|---|
| `g4view.exe` | the viewer: orbit, pan, zoom, a run control bar, an output log, `/vis/` macros |
| `g4builder.exe` | the model builder: the same renderer, plus editing panels and a gizmo |

`b1_view.exe` and `b1_vis.exe`, which this document used to describe, are gone. `g4view.exe`
replaces the first and takes its scene from a registered `G4VUserDetectorConstruction` rather
than from a hard-coded B1, so it shows whatever the detector describes.

```
g4view.exe                       example B1, the default scene
g4view.exe -scene B1 -n 5000     run 5000 events on startup
g4view.exe examples\B1\vis.mac   drive it from a Geant4-style macro
g4view.exe -shot 65 55           one frame from a fixed viewpoint, saved as a PNG
g4view.exe -selftest             drive the camera and one run for 120 frames, save, exit
g4builder.exe                    an empty 1 m air world, ready to build in
g4builder.exe my.model           reopen a saved model
```


## The viewer is a library now, not a program

Everything that draws lives in `src/render/vis_manager.cu`, compiled once by `build_vis.bat`
into `out/vis_manager.obj` and linked into whatever wants a window. That is what Geant4 ships
as `libG4vis`, for the same reason: the viewer pulls in Win32, WGL, GDI and the CUDA renderer,
and every program that included it as a header would pay to compile all of that again.

Three things use it:

- `g4view.exe` is a thin `main` that picks a scene from the registry by name and calls
  `viewer::Open` then `viewer::Loop`.
- `G4VisExecutive::Initialize()` registers `viewer::ApplyVisCommand` as the `/vis/` handler,
  so **an example's own vis.mac now works**. `/vis/open OGL 1280x800` opens the same window,
  with the same run control bar, on whatever geometry that example's `DetectorConstruction`
  built.
- `G4UIExecutive::SessionStart()` runs that window's event loop. With no window open it falls
  back to an `Idle>` prompt on stdin, which is what a Geant4 terminal session does - and which
  makes an example scriptable by piping commands into it.

So this works, and is what `examples/B1/exampleB1` with no arguments does:

```
/control/execute init_vis.mac     ->  /run/initialize, then /control/execute vis.mac
                                  ->  /vis/open OGL 1280x800     opens the window
                                  ->  /vis/drawVolume            draws B1's four volumes
                                  ->  /vis/viewer/set/viewpointThetaPhi 70 20
                                  ->  /vis/scene/endOfEventAction accumulate
ui->SessionStart()                ->  the event loop; SPACE runs the event count in the box
```

`/vis/open` before `/run/initialize` is an error rather than a crash, and says so once rather
than once per command in the macro - see RISK.md S5 for the three separate faults that one line
of output used to have.
## The UI

Drawn on the CPU into the same RGBA buffer the CUDA renderer fills (`src/render/ui.h`), then
uploaded as one texture. That is what keeps it dependency-free - no Qt, no ImGui, no GLFW -
and it costs one pass over the panel pixels, about a quarter of a megapixel at 1440x880.

Text comes from GDI: a fixed-pitch face is rendered once into a coverage atlas at startup,
which avoids hand-typing glyph bitmaps and gives properly hinted text. gdi32 is already
linked for the window.

Immediate mode, because the widget state that matters is a handful of integers - which
control the mouse is over, which has focus, what is in a text field - and expressing a panel
as straight-line code keeps its layout and its behaviour in one place. There is no widget tree
to keep in sync with the model.

### The builder's panels

`g4builder` puts five lists in two sidebars, and how their heights are decided is a policy
worth writing down because it has been wrong twice.

One scroll area per sidebar fails on the case that matters: a segmented phantom puts a few
hundred rows in SOLIDS, and reaching SOURCES then means scrolling past an organ list. So each
section scrolls in its own box. The heights were then shared out in proportion to what each
section asked for (`ui::DivideSections`, since removed), which put SOURCES wherever the solid
list happened to end - so the panel's furniture moved every time a solid was added.

They are **fixed** now: two halves on the right, three thirds on the left above the physics
button. That leaves space unused on a small model, which is the cost of a layout that is the
same on every model, and both halves scroll so nothing is unreachable.

Within a section, `ui::FrozenTitle` keeps the heading out of the scroll area - a heading that
scrolls away stops being one exactly when the list is long enough to need it - and
`ui::SplitListForm` divides what is left between the list and the selected item's form. The
list takes what it needs and the form takes the rest, capped so the form is always owed what
it asked for or half the section, whichever is less. See docs/RISK.md V18 for the two ways
that split was got wrong first.

### Scrollbars, and units

A scroll area's bar is nine pixels wide, in the panel's own right margin. It does not reserve
width from the content, because a bar that did would change the content's layout, which
changes the content height, which decides whether there is a bar at all - two layouts on
alternate frames.

The drag is handled in `ScrollArea::Begin`, before the content is drawn, rather than in `End`
beside the painting: a click is claimed by the first widget that tests for it, and by `End`
every row in the list has already had its turn. Both ends work from `BarRects`, so the thumb
is painted where the drag picks it up.

Every numeric field with a unit has a menu in place of its unit label (`ui::UnitField`). The
model is always in mm, degrees and MeV - Geant4's own units - and the menu is a factor applied
on the way in and out, so **changing the unit re-displays the same quantity**: 50 mm becomes
5 cm and back to 50 mm. The other reading, keeping the number and changing what it means,
would move the geometry every time the menu was touched and would do it silently. Which
family a field offers is keyed off the unit string `builder::ShapeParams` already carries, so
every shape's fields get the right menu without a second table to keep in step.

### Two rules for anyone adding a widget

**Hit-test with `Context::Hovering`, never with `rect.Contains(mouse)`.** `Hovering` requires
three things: the cursor inside the widget, inside the canvas clip, and outside `Context::block`
- the region an open dropdown's list is painted over. The clip is what keeps a control scrolled
out of its section from going on claiming clicks over whatever is drawn below it; the block is
what keeps a widget the list covers from taking a click meant for one of its rows. The bare
`Contains` is the version without either. See docs/RISK.md V19 and V22.

`block` is a frame old, deliberately. The list is painted after the panel that declared it, so
its rows are hit-tested last in the frame and everything underneath has already had its turn -
in one pass there is no other order available. Carrying the region over from the frame it was
painted on puts it where the list actually is on screen, which is where the cursor is aiming.
It carries a layer with it, so a pop-up drawn in FRONT of an open panel dropdown still takes
its own clicks.

The menu bar is the other overlay with this shape - `Item` records rows and `EndMenu` paints
them, after the panels - and it uses the same one region, saving and restoring whatever it found
so that a bar drawn while a dropdown is open does not wipe the dropdown's.

Anything that hit-tests and is not a widget has to arbitrate for itself. There are two: the
splitters, whose grab strips straddle the boundary they move and so overlap the 3D view by
design, and the camera. A press on a strip is a resize and nothing else - `PressOnSplitter`
decides, at the press, and the answer holds for the whole drag. This is not hypothetical
tidiness: it was only ever wrong for the LEFT splitter, because `left_w = mouse_x` keeps the
cursor on the view's first column for the whole drag while the right and bottom splitters put
the view's exclusive edge there, and the camera claimed those columns too. Resizing the left
panel rotated the view.

**Take an id from the table in `g4builder.cu`, not from the next free number.** Anything
indexed by a model list - one id per solid, per class, per source - needs a block of its own;
the fixed controls use the low numbers. Two widgets sharing an id still draw and still fire
correctly on their own rectangles, so the symptom is not "the wrong button" but a rename
typing into the wrong row, and nothing about that points at numbering.

### Two faces in the same place

The layer model gives every shared point to the higher layer, so of two coincident faces the
higher one is entirely visible and the lower one entirely invisible. The renderer used to draw
**neither**: the surface search took whichever volume it found first at the nearest distance,
the ownership test threw it away because the other one owned the space behind it, and the next
iteration started just inside both - where both are "already inside" and neither is ever
entered again. A hole exactly where two faces meet.

The search now prefers the higher-ranked volume among surfaces within one nudge of each other,
which is the distance at which the walk already treats two surfaces as one. It uses the
volume's highest possible rank, because the hit point is not known yet, and `locate` still has
the last word - so a voxel grid whose class loses at that point behaves exactly as before.

### A nearly transparent surface is mostly the background

The geometry pass accumulates PREMULTIPLIED colour: each surface adds
`alpha * (1 - alpha_so_far) * colour`. A volume at 3% opacity therefore leaves 3% of its colour
in the framebuffer and nothing else, and writing that out is a nearly black pixel - reported as
"a very low opacity looks like the background is black". What it means is 3% of the colour and
97% of whatever is behind, and behind it is the viewer's gradient.

The coverage now rides in the framebuffer word's spare alpha byte (`vis::pack_rgba`) and
`vis::resolve_pixel` finishes the compositing. One function for all three resolve kernels,
because the three had already drifted - only the on-screen one was fixed first, and the two
used for offscreen export would have written a different picture from the one on screen.

### List rows are columns

`ui::ListRow2` gives a row's two fields a fixed share of the width each, and ellipsizes both
inside their own share. One padded string - `Fmt("%-12s %s", name, material)` - looks like
columns until a name is longer than the padding, and then the second field slides right and off
the end of the row. That is not cosmetic: the material is the field a solids list is scanned
FOR, and a phantom imported from a file called `adult_male_1mm_segmented.raw` pushed it out of
view entirely. Below about nine characters of room for the right column the LEFT one gives way,
because a truncated name is still recognisable and a truncated material is not.

Checked by the pixels in `tests/test_ui_layout.cu` - the row is drawn with a solid-coverage
font and the ink is measured - rather than by re-deriving the arithmetic, which would pass
whatever the row did.

### The source form

A source's direction is three fields, shown for every kind except the isotropic shell, which
overrides the angular shape and so would have a control that visibly does nothing. The gun
normalises, so 0,0,2 and 0,0,1 are the same beam and the form shows the unit vector it will
become. A zero direction falls back to +z in `G4ParticleGun::SetParticleMomentumDirection`
rather than in the form, because a direction also arrives from a loaded project file.

## Where a frame's time goes

`-benchmesh N` reports it, because "the GUI is slow" is not a number and the four things a
frame does have very different fixes:

```
   triangles      cuda   readback     ui   present     frame
       4,096   4.91 ms    0.95 ms   2.73    8.08 ms   16.7 ms  (60.0 fps)
     401,956   8.72 ms    1.03 ms   2.70    4.22 ms   16.7 ms  (60.0 fps)
   1,607,824  11.60 ms    0.97 ms   2.74    1.32 ms   16.6 ms  (60.1 fps)
```

All three are at 60 fps because they are now VSYNC-limited rather than work-limited: `present`
is SwapBuffers waiting for the refresh, and it shrinks as the render grows. The work is
8.6 / 12.5 / 15.3 ms, all inside the 16.7 ms budget.

**It was not.** The read-back of the rendered viewport into the host buffer was a loop of
`cudaMemcpy`, one per scanline. Each carries a fixed launch and synchronisation cost of order
ten microseconds whatever it moves, so 960 rows cost **16.4 ms - a flat cost, every frame,
whatever was in the scene**. That is the entire frame budget spent on a six-megabyte copy the
hardware does in under a millisecond, and it was charged to opening a dropdown as much as to
orbiting the camera. `cudaMemcpy2D` takes the two pitches and does it in one call: 16.4 ms
became 0.97 ms.

Reported as the GUI slowing down with a large CAD file loaded, which it did - but it was slow
with an empty scene too, and that is the half of the report that made it findable. The flatness
against triangle count is what named the culprit: a cost that does not move when the geometry
grows a hundredfold is not the geometry.

## The control bar

An event count, Run, Reset, and an "accumulate across runs" toggle; below it the last run's
event count, energy deposit, scorer mass, dose and rate; below that the display toggles and
the volume list, with each volume's colour, layer and whether it is scored. The bottom of the
sidebar is the program output, colour-coded - red for `FATAL`, amber for `WARNING`, green for
a dose line, blue for an echoed macro command - because a long log is otherwise unsearchable
by eye.

SPACE runs the event count in the box, which is the shortcut worth knowing.

## Macros

`/vis/` commands reach the viewer through `G4UImanager`, which forwards anything under that
prefix to whatever viewer has registered itself. `examples/B1/vis.mac` is written in the shape
Geant4's own `vis.mac` has, and the handled subset covers `/vis/open`, `/vis/drawVolume`,
`/vis/viewer/set/viewpointThetaPhi`, `/vis/viewer/zoom`, `/vis/viewer/set/style`,
`/vis/scene/endOfEventAction`, `/vis/geometry/set/visibility` and the trajectory model
commands. An unrecognised `/vis/` command is reported rather than ignored - a macro that
silently does nothing is worse than one that stops.

## Picking and the gizmo

The builder picks solids by casting the cursor ray against each one with the same `dist_in`
the navigator uses, on the CPU. What you can click is therefore exactly what the transport can
hit; a separate picking representation would eventually disagree with the physics.

Selecting a solid raises three axis handles at its centre, after the Microsoft 3D Builder
pattern: dragging one moves the solid along that axis only. The handle direction is computed
by projecting a point one unit along the axis and taking the screen offset, so the handles
point where the axis actually goes after an orbit; and the drag tracks the cursor along that
screen direction rather than along screen x, so a diagonal axis follows the mouse.

## How the renderer works

**The geometry pass is a ray cast through `geom::dist_in` and `geom::normal_at` - the same
functions transport uses for distance-to-boundary.** Visualization therefore adds no new
geometry code, and the picture cannot disagree with the physics about where a surface is.
Only `normal_at` was new; transport never needs to know which way a surface faces.

**Depth resolution uses a packed 64-bit framebuffer.** The high 32 bits hold the IEEE bit
pattern of the depth - monotonic for positive floats - and the low 32 bits hold packed RGB.
A single `atomicMin` then performs a correct nearest-wins depth test, so the wireframe and
trajectory passes composite against the ray-cast surfaces with no separate z-buffer and no
sorting.

Three passes:

1. `render_geometry` - one thread per pixel, ray casts the solid volumes (Shape1 cone,
   Shape2 trapezoid) with two-sided diffuse shading.
2. `render_edges` and `render_trajectories` - one thread per line, rasterized in screen
   space with the atomicMin depth test. Wireframe boxes for World and Envelope, as Geant4
   draws a containing world by default.
3. `resolve_to_rgb` - unpacks to top-down RGB over a gradient background.

**Trajectory capture** hooks `step_gamma` and `step_lepton`, appending one segment per step
via `atomicAdd`. Storage is float regardless of the transport scalar type, and the physics
builds pass `vis::no_capture()`, so capture compiles down to a branch that never fires and
the physics path is unaffected (verified: 2.46M events/s and unchanged dose after the hook
was added).

**Colouring follows Geant4** - by charge: negative red, neutral green, positive blue.

### The render pass is float; the transport is not

The geometry pass calls the transport's own functions, and on a GeForce card that meant paying
FP64 rates for a picture: one FP64 unit per 64 FP32 units on sm_86. So the render pass is given
a float copy of the scene (`render/float_geometry.cuh`) and the transport keeps its doubles.

The split is not caution. Float mesh geometry puts about 1e-5 mm of error into where a track
stops, and the dose is the number this project is judged on; a pixel is a third of a millimetre
at any useful zoom.

Measured by alternating two builds differing only in this, within one session, at 1680x960:

```
   triangles      double     float    gain
      40,000    47.6 ms   25.5 ms    1.86x
     401,956    61.6 ms   28.5 ms    2.16x
   1,607,824    70.6 ms   31.9 ms    2.21x
```

which is 21 fps to 39, 16 to 35, and 14 to 33. The spread within each triple was under 1%, and
the paired design is there because run-to-run variation across builds on this card is about 6% -
see docs/RISK.md V21 for the time that ate a wrong conclusion.

**The conversion is a cast, and the BVH is the reason to check that.** A box rounded INWARD by
one ulp no longer contains its own triangles, and the traversal rejects any subtree whose box
the ray misses - a hole in the mesh, not a shading error. It cannot happen here because
`build_bvh` sets each bound to the min or max of vertex COORDINATES, so every bound is one of
the numbers it will be compared against, and rounding to float is monotonic. That property is
asserted in `tests/test_float_render.cu` rather than assumed: the first version of the
conversion spent sixty lines recomputing every box from the float triangles, and the test
written to justify them showed a plain cast breaks containment in none of 4095 nodes.

The voxel arrays are shared rather than copied - cells are shorts, class layers are ints - so a
512^3 phantom does not get a second 268 MB.

**What it costs in the picture**: the viewer's selftest counts the pixels solid geometry
covers, and it moved from 57768 to 57766. Two pixels of silhouette in fifty-seven thousand.

**And what it cost before it was measured properly.** Float exposed two arithmetic assumptions
that double had been carrying, and both showed up as missing pixels:

* the generic engine solves a quadratic whose discriminant is `b^2 - 4ac`. For a 60 mm sphere
  seen from 1500 mm that is 9.00e6 - 8.99e6 = 1.44e4 - two numbers agreeing to three digits,
  which in float leaves four. The root lands about 0.004 mm out, `is_crossing_to` probes
  0.001 mm either side of it, both probes fall on the same side, and the crossing is not seen.
  Measured: an orb keeps 100% of its rays at 200 mm, **47% at 1500** and 10% at 4000. The
  renderer therefore starts each ray at the solid's bounding sphere (`geom::bounding_radius`),
  where b and c are both O(radius) and nothing cancels. Closed-form solids - box, trd, cons -
  never went through that path and were never affected, which is what the shape of the
  measurement said before any code was changed.
* `quadric_inside` compared the raw quadric value against `kSurfTolerance`, which is a LENGTH.
  An orb is written `x^2+y^2+z^2-r^2`, gradient 120 per mm at r = 60; an ellipsoid is written
  `x^2/a^2+...-1`, gradient 0.04 per mm. One tolerance is therefore a band 8e-7 mm wide on one
  and 2.5e-3 mm on the other - wider than the probe - so on an ellipsoid **both** sides of a
  crossing reported "on the surface" and it had no surfaces at all, at any distance.
  `quadric_residual` divides by the gradient, which is exactly what `quadric_normal` beside it
  had always done, and for the stated reason.

Both are asserted in `tests/test_float_render.cu` as a sweep over camera distance, because the
SHAPE of the failure is what identifies it: 100% at 200 mm falling to 10% at 4000 is a
cancelling discriminant, and 0.1% everywhere is a tolerance in the wrong units.

### A voxel volume is not one surface, and the layer rule still applies to it

A grid whose cells are coloured individually is marched cell by cell inside the outer walk,
because the ray's entry into the grid's bounding box is the outside of a box with the anatomy
inside it. That march has to answer the same ownership question the outer walk answers for an
ordinary surface: **a higher layer takes the space wherever it overlaps, so the cells under it
are not there as far as the transport is concerned and are not drawn.**

Two halves, and both are needed:

* the march STOPS where the nearest volume that outranks the grid begins. Without it the march
  ran the grid's whole depth and composited cells sitting inside an opaque volume placed over
  the phantom - and front to back those cells arrive before that volume's own surface, so they
  were blended on top of it. A phantom showing through a solid object in front of it.
* and it RESUMES past that volume, at the far side of whatever covers the grid there. Without
  that, everything behind a translucent object over a phantom disappears instead.

The distances are compared from the same origin and ties go to the covering volume, because
cells are on a lattice and a volume placed over them lands on it: a box face flush with a cell
boundary is the normal case, not the corner case, and getting it wrong paints one cell's worth
of colour over an opaque surface.

Style is not consulted. A volume that is hidden or wireframe still owns its space, which is
what `locate` says and therefore what the transport sees, so hiding a box over a phantom
leaves the hole it occupies - the same as it already does over an ordinary solid.

Both halves are asserted end to end by the builder's selftest, against a fixture whose classes
are chosen by which side of a covering slab a cell is on: recolouring a class the slab owns
must change nothing, and recolouring the class beyond it must change nothing while the slab is
opaque and something once it is not. With three more controls, because "nothing changed" is
also what a fixture that is off screen, or a cover that is not being painted, looks like.

## X-ray mode

Opaque surfaces hide the tracks inside them, which is where the interesting physics is -
Shape2 *is* the scoring volume. `render_trajectories` takes a `depth_scale`; at `1e-3` every
track depth is compressed below any surface depth, so tracks show through solids while still
sorting correctly among themselves. Geant4's default viewer is see-through for the same
reason. Toggle with X.

## PNG without a dependency

`render/png.h` writes PNG using deflate **stored** (uncompressed) blocks, which are legal
zlib and trivially encoded - about 60 lines including CRC32 and Adler32, no zlib link. Files
are larger than a compressed PNG (3 MB at 1280x800) but directly viewable. BMP output is
still available via `write_bmp` and `resolve_to_bgr`.

## Correctness check

Judging a render by eye is unreliable: the first image looked as though tracks were escaping
the world. They were not. The check that settled it was numeric - the bounding box of every
captured segment, against the world half-extents:

```
segment bounds: x [-120.0, 120.0]  y [-120.0, 120.0]  z [-180.0, 180.0] mm
world half-extents: 120, 120, 180 mm -> 0 endpoints outside
```

Exactly the world boundary, zero escapes.

`g4view.exe` does not currently re-run that check - it was in the offline driver that has
since been removed, and re-adding it to the viewer is a few lines. Worth doing: it is the only
thing standing between "the picture looks wrong" and knowing whether it is.


## Not done

- No CUDA/GL buffer interop; there is a host round trip per frame. It also happens to be what
  makes CPU-side UI compositing free, so removing it would mean moving the UI to the device.
- No per-track inspection. Clicking a *trajectory* to read its particle type and energy would
  need an ID buffer alongside the colour buffer - cheap, since the packed framebuffer already
  carries spare bits. Clicking a *volume* works.
- No true alpha blending; only the x-ray trick.
- A mesh has no wireframe. An imported CAD part is ray cast against its triangles and looks
  right in surface mode, but the wireframe pass only knows how to draw a box's twelve edges.
  Drawing every triangle edge of a 100,000-triangle import would be useless anyway; the useful
  thing is a silhouette, which needs the edges where the surface normal changes sign.
- No dose overlay. Deposits are scored but not visualised; colour-mapping deposition onto the
  scoring volume would be a natural next step, and for a voxel volume it is the obvious way to
  look at a dose distribution at all.
- Solids are ray cast one at a time, so the render is O(volumes) per pixel. Fine for tens of
  volumes; a scene with hundreds wants the same BVH the navigator needs.
- Linux/macOS. The window layer is Win32-specific and the font atlas is GDI; the renderer
  itself is portable CUDA.
