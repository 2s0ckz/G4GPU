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


### And a translucent volume is a different measurement entirely

The four numbers above are all OPAQUE. The front-to-back walk stops accumulating at 99.5%
coverage, so an opaque volume ends the walk at its first surface: one surface per pixel whatever
is behind it. Set the opacity below 1 and the walk does not stop, and the cost is no longer
about the triangle count at all - it is about **how many surfaces a ray crosses**:

```
  3,200,000 tri   opacity 1.00, 5 shells    cuda 12.92 ms    17.5 ms/frame   57 fps
  3,200,000 tri   opacity 0.40, 5 shells    cuda 81.92 ms    86.3 ms/frame   12 fps
```

Same geometry, same camera, one number different - and 12 fps is the report: "particularly when
opacity is set to anything less than 100, the GUI becomes super slow; hovering over a button
that should change colour, it only changes about a full second later."

Note what the benchmark had to change to see this. A sphere is crossed TWICE by any ray, so the
walk runs out of surfaces after two layers whatever the opacity is, and a benchmark built on one
understates a translucent import fourfold. `-benchmesh N opacity shells` writes concentric shells
into the one mesh, because a real CAD part is not a sphere: an assembly, a housing, a panel with
ribs behind it all present many surfaces along one ray. `kMaxLayers` = 8 in `render_geometry` is
the cap on how many are composited - which is a limit on the PICTURE as well as the cost, since
a translucent assembly with more than eight walls loses the ones past the eighth.


### The render is not in the UI frame

Making the render faster twice over did not answer the actual complaint, and could not have. The
panels are rasterised by the CPU into `host_rgba`, which is also where the rendered viewport is
copied to, and between the two sat `cudaDeviceSynchronize()` - so the UI was not waiting on the
render by choice, it was structurally DOWNSTREAM of it. Every millisecond the render cost was a
millisecond before a hover highlight could repaint. A render made twice as fast still holds the
UI up for half as long, and the next scene is twice as big.

The render now runs on its own stream and the frame POLLS a `cudaEvent_t`:

```
  if (render_inflight) {
    st = cudaEventQuery(render_done);       // NOT cudaEventSynchronize
    if (st == cudaSuccess) { AdoptRender(a); }
  }
  if (!render_inflight) { IssueRender(a, w, h); }
```

Ready means adopt that image and launch the next; not ready means composite the PREVIOUS one and
carry on. The UI runs at the refresh rate whatever the render costs. The visible cost is that in
a slow scene the viewport is one render behind the camera, which is what every application of
this kind does and is not what was reported.

Three things that are the whole difference between this working and looking like it does nothing:

**`cudaStreamCreate` returns a BLOCKING stream.** It implicitly synchronises with the legacy
default stream, so any ordinary `cudaMemcpy` or `cudaMalloc` elsewhere in the frame waits for the
render - the stall deleted from one line and reinstated by the allocator, invisible in the code.
`cudaStreamCreateWithFlags(..., cudaStreamNonBlocking)`.

**Pinned staging, sized to the WINDOW.** `cudaMemcpy2DAsync` out of pageable memory is not
asynchronous, so the read-back had to move to pinned memory or nothing would have changed. But
`cudaHostAlloc` pins pages and `cudaFreeHost` unpins them - milliseconds for six megabytes - and
the VIEWPORT resizes on every frame of a splitter drag. Sizing the staging pair with the viewport
would have put a fresh stall into exactly the interaction being fixed. The viewport is never
larger than the window, so one allocation per window resize covers every viewport; a render
smaller than the buffer carries its own dimensions alongside it and only the overlap is blitted.

**Everything that frees what the render is using drains first, itself.** `AllocViewportSurface`,
`AllocSurface` and `RebuildScene` each call `DrainRender` rather than trusting the frame to have
noticed - and `RebuildScene` matters most, because the Run button reaches it through
`RunFromGui` without going near a frame at all. A free while a kernel is still reading is a
device heap corruption that surfaces several frames later somewhere unrelated.

`-selftest` and `-benchmesh` wait rather than poll, for opposite reasons needing the same thing:
the selftest's checksums compare a recolour against the frame after it, so its picture has to be
that frame's; and the benchmark is measuring the render, which asynchronously is a launch.

#### What is asserted, and why it is not the speed-up

`-benchmesh` times the same scene twice in one process - render inside the frame, then render on
its own stream - which makes the comparison paired by construction. But the gate in
`build_all.bat` is not the frame-time ratio: below the refresh interval both arms are
vsync-limited and the ratio says nothing, so a gate on it would pass every fast scene for the
wrong reason.

The gate is that the UI frame spends UNDER A MILLISECOND on the render - a launch and a poll -
which holds on any scene and fails the moment a `cudaDeviceSynchronize` reappears in `DrawFrame`.
It needs no slow geometry, so it costs the pipeline a small mesh rather than a large one.

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


## null: a volume or a class that is not in the scene

Every layer menu - on each solid row, on each voxel class row, and in the selected solid's form -
opens with `null` rather than a number. A solid or a class set to it is **not in the scene**: not
transported through, not drawn, and not scored.

One fact, not three switches. A solid on it is not placed at all, so there is nothing to step
into and nothing to attach a detector to. A voxel class on it keeps its cells' values and its
colour, and its cells are not part of the volume - `geom::inside_volume` returns false there, so
`locate` cannot return the grid, the transport steps through as whatever does own the space, and
the tally follows ownership without a rule of its own.

**It is not a layer, and not a number.** No rank is computed from it. That is the second design;
the first made absence a layer two billion below everything, on the reasoning that such a cell
loses every overlap and the world takes its space for free. What it actually bought was a
renderer that read "something outranks this cell" as "stop the march here" - deleting a phantom
from its front class backwards - and a cover scan that admitted every volume in the scene.
docs/RISK.md V28 has both. A cell that is not there does not lose an overlap; it does not take
part in one.

`visible` is still a different thing and still separate: a hidden volume owns its space, is
transported through, and is scored. `null` is the answer to "this should not be here at all".

The world is refused it outright - there is nothing to transport in without a world - and moving
the world off layer 0 asks first, since the world contains everything and a world above another
volume's layer outranks that volume everywhere, emptying the detector in response to a one-click
menu change.

**The glyph.** The font atlas is indexed by byte and the strings drawn through it are
`std::string`, so a three-byte UTF-8 character cannot reach it - each byte lands outside the
printable run and draws nothing, which is why `ui::EyeToggle` draws its icon from primitives
instead of typing one. A layer menu cannot do that: its options go through `ui::Select` as text.
So one atlas slot is rasterised from U+2205 with the wide GDI call, and given a byte nothing else
uses (`ui::Font::kEmptySetByte`, 0x01 - not printable, cannot appear in a name or a number, and
one glyph wide, which the fixed-pitch layout arithmetic depends on).

## Imports arrive assignable

Two things an imported phantom used to leave for the user, both now defaults:

**Index 0 is not special.** It arrived at zero opacity, because 0 means "nothing here" in most
segmentation conventions and is also the commonest value, so an opaque import is a solid block.
Usually true, and not this code's call: the convention belongs to the file, index 0 is sometimes
a real structure, and a class that arrives invisible without being asked for has to be discovered
before it can be wondered about. Anything unwanted in the scene now has a control that means
exactly that, and it applies to any class rather than to whichever one is numbered zero. Every
class arrives at a tenth.

**Every class starts on the container's material.** A class with none makes the whole volume
unbuildable - `build_scene` refuses to place a volume whose material is missing - so an import
used to arrive in a state where the scene could not be built until every one of what may be two
hundred rows had been visited. The volume's own material is what a cell matching no class already
falls back to, so this agrees with the rest of the grid rather than inventing a value: wrong for
most classes on a segmentation, and wrong in a column the user can see rather than absent.

## +Z is up

Elevation lifts out of the x-y plane rather than out of x-z, in the builder and in the viewer
both. A detector is described in beam coordinates - z the beam axis, x-y transverse - and a
viewer with y up shows a linac gantry on its side and a phantom's axial slices edge-on, so every
dimension typed into the panels has to be mentally rotated to match the screen. It also makes
`/vis/viewer/set/viewpointThetaPhi` mean what Geant4 means by it, since that command's theta is
measured from +z.

Elevation stays clamped short of the axis, which is what keeps `make_camera`'s
`cross(forward, up)` from degenerating - so looking exactly down z is 1.5 rad, not 1.5708.


### The origin shift needs a bound, and two solid families had none

`bounding_radius` is a switch over solid types with `default: return 0`, and zero turns the origin
shift off rather than saying it is unnecessary. kPolycone and kPolyhedra keep their sections in
the aux pool, which the p[]-only form cannot read, so both fell to the default - and a boolean
node has no p[] extent either, while `boolean_dist` hands each child the origin it was given.

Measured at the limb, rays kept in float against double:

```
   camera      polycone      union of two orbs
   200 mm        100%              99.7%
   600 mm         93%                94%
  1500 mm         39%                28%      <- the default camera distance
  4000 mm         22%                 7%
 12000 mm         17%              0.26%
```

The builder's own "Cone" primitive is a polycone and the selftest scene contains a subtraction, so
both were speckled the whole time - the same cancelling discriminant as the sphere, reached
through a function that had never heard of two of the types. A store-aware overload reads the
polycone's planes out of the aux pool and gives a boolean the larger of its children's bounds plus
their frame offsets; all of it goes to 100%, flat with distance.

The p[]-only form is deliberately untouched: it is on the transport's safety path, where a larger
number is a physics change. This is a rendering question asked by code that has the store.

**What the sweep had to become to see it.** It swept four hand-picked shapes with rays 20 mm from
the axis of a 60 mm solid - and a cancelling discriminant shows at the LIMB, where the two roots
are nearly equal. The same polycone kept 100% near the axis and 39% at 98% of the silhouette. It
sweeps the type list now, aims at each shape's own material, and goes out to the limb. See
docs/RISK.md V29.

### The entry is the word "null"

It was ∅, drawn out of an ellipse and a stroke, because the atlas is indexed by byte and a
three-byte UTF-8 character cannot reach it. Three attempts and as many passing tests later it
was still a smudge at eight pixels across - docs/RISK.md V29 has the whole of that - and a
symbol that has to be explained is worse than the word it stands for. `Canvas::EmptySet` and
the byte it was reached by are gone.

What is worth checking is not how the entry looks but that the menu maps it to the layer: an
off-by-one between the option INDEX and the layer NUMBER would put every solid one layer out,
silently, and the option list is the only place that mapping exists. The builder selftest
walks it.

### A volume on layer 0 clashes with the world

The overlap check no longer exempts the world. It did, on the reasoning that the world contains
everything so an overlap with it is containment - true for layer 1 and above, and that case never
needed an exemption, because the layer-range test prunes it: the world spans layer 0 alone and
cannot meet layer 1 anywhere.

A volume placed ON layer 0 is a real same-layer overlap with the world, resolved only by
"whichever was added later wins" - the tie-break the check exists to refuse, since it makes the
dose depend on the order the detector was built in. The world is one more volume to the general
rule now.


## Anti-aliasing, at the edges only

One ray per pixel puts a hard step wherever a surface ends: the pixel is the surface or it is
not, so its coverage is 0 or 255 and never between, and a silhouette becomes a staircase. Four
rays per pixel everywhere would fix that and cost four times the render, on a scene where the
render is already the expensive part, to improve the small fraction of pixels that are on an
edge.

So the edges are found first and only those pixels are traced again:

```
  render_geometry   one ray per pixel, as before
  mark_edges        one cheap pass over the framebuffer, no geometry: appends the pixels
                    where a surface begins or ends to a list
  refine_edges      four rays at each listed pixel, on a rotated grid, averaged
  render_trajectories, resolve_to_rgba   as before
```

`trace_pixel` is the old kernel body, lifted out so that both passes can call it; the kernel is
now four lines. **Before the trajectories**, deliberately: the track pass `atomicMin`s into the
same framebuffer, and refining afterwards would retrace the geometry over a track and delete it.

Averaging is sound because what the framebuffer holds is premultiplied colour and coverage:
summing those over the samples and dividing is the coverage-weighted average of the surfaces
seen. The depth is the nearest of the samples, since depth is what sorts tracks and edges against
the first surface.

**Coverage and depth decide what is an edge, not colour.** Colour was tried first and marks
almost the whole frame on an imported mesh: a million-triangle sphere is shaded per triangle, so
the colour steps between neighbouring pixels everywhere. The cost went to +47% opaque and +78%
translucent. Those steps are also not aliasing - a facet boundary is real detail in the picture,
and smoothing it blurs the model. What aliases is the silhouette, and that shows as a jump in
coverage (against the background) or in depth (against other geometry), neither of which a
smoothly curved interior has. The depth test is relative, because a 1 mm step matters at 50 mm
and is nothing at 5 m.

**The marked pixels are compacted into a list, and that is most of the performance.** Edge pixels
are a thin scattered curve - 0.2% of the frame, spread so that nearly every warp holds one or two
of them. A pass that read a per-pixel flag therefore ran four full traces on one lane while
thirty-one idled: 2.38 ms for 2950 pixels, 200 ns a ray against 3.8 ns for an ordinary one.
Compacted, the same rays are contiguous and every warp is full - 1.87 ms for the same pixels.

What it costs, paired in one binary (`-benchmesh N opacity shells aa`):

```
   scene                          aa off     aa on    marked        frame
   4,096 tri  opaque             3.39 ms    4.50 ms   3028 (0.2%)   60.0 fps both
   100,000 tri  opaque           5.55 ms    6.69 ms   2925 (0.2%)   60.0 fps both
   1,600,000 tri  opaque         6.69 ms    9.97 ms   2960 (0.2%)   60.0 fps both
   3,200,000 tri  opacity 0.40  19.92 ms   30.23 ms   7377 (0.5%)   42 -> 29 fps
```

20% to 50% of the render, and the frame rate is unchanged in every case where the render fits
inside the refresh interval - which, with the render on its own stream, is every case for the UI.
The toggle is in the Visualization window for anyone who wants the milliseconds back.

`-benchmesh` reports the marked count for the same reason it reports the phase split: when the
cost of this pass is a surprise, that number is the whole explanation.

## The insert menu asks for a size

Choosing a primitive from Insert opens a form with its parameters and its position, rather than
inserting one and leaving you to find the fields. Those are the two things anybody changes
straight afterwards.

**The default size comes from the world.** Each shape has proportions written in units of one,
and they are scaled so that the largest LENGTH among them is half the world's half-extent -
centred at the origin. A fixed fraction cannot do that job: the 0.15 that gave a sensible box in
a 500 mm world gives a 1.5 mm speck in a 10 m one. Which parameters are lengths comes from
`ShapeParams`' unit column, so a shape added later scales without anyone updating a table of
maxima; the polycone's z-sections are scaled too, since its size is not in `p[]` at all.

The smallest of the world's three half-extents is the reference, so the result fits in every
direction. For the default cubic world that is exactly "half the world extent in that direction".

The form borrows the selected solid's parameter fields rather than carrying fifteen of its own -
`App::pfield_for` is the cache key those already had, and `kPendingFields` both claims them and
guarantees the solid form re-seeds when the form closes.

## The null layer says "null"

It was ∅, drawn out of an ellipse and a stroke because the font atlas is indexed by byte and a
three-byte UTF-8 character cannot reach it. At eight pixels across it was never going to be more
than a smudge that had to be explained, and getting it that far took three attempts and as many
tests that passed on a broken glyph (docs/RISK.md V29). The word is four characters, which is
exactly the width the layer column already reserves, and it needs no explaining.

What is worth checking is not how it looks but that the menu maps it to the layer: an off-by-one
between the option INDEX and the layer NUMBER would put every solid one layer out, silently, and
the option list is the only place that mapping exists.


### A candidate at zero distance is a volume you are already inside

`box_dist_in` starts its `tmin` at zero and clamps, so from inside a box it returns **exactly 0**,
not infinity - and a voxel grid is a box. The surface search would otherwise re-find a grid the
walk is standing in, at distance nothing, on every iteration.

The tie rule hid that for a long time: the volume the march handed the ray back to sits a nudge
ahead, ties with the zero, and wins on rank because it is a higher layer, which is what a cover
is. A volume on a LOWER layer inside a grid loses that tie, so the grid is re-entered, marches,
clamps at that volume again, and the pixel goes round until `kMaxLayers` with nothing accumulated
- drawing as a HOLE the shape of the volume that should have been there.

A lower-layer volume inside a grid is what a null class makes possible: while the class is present
the grid owns that space and the volume is correctly hidden, so nobody had put one there before.
The search now requires a strictly positive distance. Nothing legitimate sits at zero - the walk
nudges past every surface it crosses - and a boolean or a mesh whose next crossing genuinely lies
ahead reports that distance and is unaffected.

Two things go with it, where a grid has absent classes:

- **every volume is a candidate cover.** The scan normally rejects anything below the grid's
  lowest class layer, and an absent cell has no business being compared against: nothing is
  there, so anything takes the space.
- **except the world.** Same `box_dist_in` reason - the world is a box the ray is always inside,
  so it would enter the list at distance zero, outrank an absent cell, and end the march at the
  first cell. It is also right on its own terms: the world contains everything by construction,
  so it can never take space away in front of a cell.


### A voxel is its own volume, as far as owning space goes

The grid is one solid for containment and entry - that is what makes a 25-million-cell phantom
affordable to ray cast - but **which volume owns a given point is a question about the cell there**:
its class's layer, and whether that class is in the scene at all.

Both halves matter and they are different statements:

- a **hidden** class (the eye toggle, or zero opacity) is still there. Nothing of it is painted,
  and it goes on outranking whatever is inside it, so a volume buried in hidden cells stays
  buried.
- a **nulled** class is gone. Its cells own nothing, so whatever occupies that space owns it -
  *including a volume on a lower layer than the grid*, which is what makes this unlike every other
  overlap in the scene. An absent cell's rank is the lowest a signed 64-bit number holds, so every
  candidate cover outranks it and the march's clamp hands the space over at whichever starts
  nearest.

It is the clamp rather than `covered` that does the handing over, and that distinction is load
bearing: `covered` ends the march, so ending it at the first absent cell would take every cell
behind it as well and a phantom with its near class nulled would vanish, far side included.
Something in the hole stops the march there; an empty hole does not stop it at all.

Where a grid has any absent class, the cover scan admits **every** volume rather than pruning on
the grid's lowest class layer - a volume below that layer would otherwise be rejected before it
was ever considered, and an absent cell has no business being compared against anything.

### A volume resumes at the far side of whatever outranks it

The surface search skips any volume the ray is already inside, and that test earns its place:
without it the search re-finds the volume just entered, paints its front face again, and a
half-transparent box renders fully opaque with nothing behind it ever reached.

What it also did was drop a volume whose space a higher layer takes for part of the ray. "Already
inside" is true there and there is no entry surface ahead - so a **translucent** volume on a higher
layer could be seen through, and the volume underneath it was not drawn beyond it. A phantom under
a vest appeared only with the PHANTOM on the higher layer, and *works above, fails below* is the
signature: a higher volume is never covered, so it never has to resume.

So `locate` is asked who owns the current point. If it is this volume, it has already been drawn or
marched from here. If it is something else, that something outranks it and this volume resumes at
the far side of it - any volume, not just a grid drawn cell by cell, which is what this branch was
restricted to for one commit. One extra containment test goes with it: a volume wholly inside its
cover stops before the cover does, and offering it at the cover's far side would paint a surface
where the volume is not.

The world is excluded, for the reason the cover scan excludes it: it contains everything, so it
never takes space away, and resuming at its far side would put the volume beyond the scene.

**A mesh cover needs its winding to be told apart from itself.** A mesh takes a fast path through
the search - one BVH walk, no containment test, because a containment test on a mesh is a parity
count and one per composited layer is what made a 100k-triangle import unusable. With nothing to
classify the hit, from inside a closed mesh the nearest hit ahead is its far wall and it was
offered as an ENTRY: the mesh composited twice, and that phantom entry sat exactly where the
volume it covers resumes, winning the tie on layer. A phantom under a vest was therefore visible
only with the phantom on the higher layer.

The winning triangle's normal decides it - facing along the ray means leaving - but only once the
winding is known, and a mesh may be wound either way. `geom::mesh_winding` takes the sign of the
same divergence-theorem sum the volume comes from, and it rides in the mesh solid's `p[7]`. Zero
there means nobody filled it in, and the hit is taken as an entry, which is the older behaviour.

What is still missing: a MESH that is itself covered does not resume, because reaching the branch
above would cost the containment test the fast path exists to avoid. No report has needed it - a
cover is the higher layer by definition - and it is written down here rather than left to be
rediscovered.

**"Standing inside it" is a question about the box, though.** `inside_volume` is per cell, and the
march has a branch gated on it that means something else entirely: *have I already marched this
grid?* Those are not the same question, and answering the second with the first broke a case that
neither feature broke alone - a ray that stopped inside a translucent volume sitting in nulled air
was told it was not inside the grid, took the not-inside path, asked `dist_in` from inside the
grid's own box, got zero, and had it rejected as a volume it was already in. The grid could not be
found again at all, so the translucent volume composited over the background rather than over the
tissue behind it. So the resume test asks the solid's own containment, and ownership stays per
cell.

### A cover at zero distance is one the ray has come out of

When a cell march is interrupted by a covering volume, the surface search draws that cover and
hands the grid back at the cover's **exit** - so the resumed march begins standing exactly on the
cover's far face. `box_dist_in` starts its `tmin` at zero and clamps, so it reports that cover as
beginning right there; the clamp fires immediately, the march paints nothing, and the next
iteration finds no candidate at all. The pixel keeps the cover and nothing else.

What that looked like: a phantom under a **translucent** volume on a higher layer showed the volume
and nothing behind it, however transparent it was. Opaque covers were fine, because there is
nothing to see through them and the march is never resumed.

Nothing legitimate sits at zero there. `owns_contained_point` has just confirmed the grid owns the
point a nudge ahead, so the ray is not inside anything that outranks the cell - a cover reporting
zero is one whose surface is behind the ray. This is the third bug from `box_dist_in` returning
exactly zero from a boundary; see docs/RISK.md V33 for the other two.

### The limb, and why the probe is a hundred tolerances

`is_crossing_to` tests containment a fixed `kProbe` either side of a candidate crossing, and a step
along a ray is not a step away from a surface. At impact parameter q on a sphere of radius R the
half-chord is `h = sqrt(R^2 - q^2)`, and stepping `d` back along the ray leaves the surface by only
`d * h / R`. Where h is small that falls inside `kSurfTolerance`, the "before" probe reports inside,
and the crossing is discarded. Not grazes: at the outer edge of the band the half-chord is 4.1 mm,
so rays whose chord through a 40 mm sphere is eight millimetres long were being lost.

    band width  =  (kSurfTolerance / kProbe)^2  *  R / 2

Proportional to R, so it is a fixed fraction of the drawn disc and zooming in makes it a wider
ribbon of pixels - reported as speckling at the edges that gets worse close up. What matters is the
RATIO: 1e-3 in double, where the band is twenty nanometres, and 1e-1 in float, where it was
0.005 R. `kProbe<float>` is 1e-2 now rather than 1e-3, which takes it to 2e-5 R. The cost is that a
float render steps over a feature thinner than 20 microns; the transport is double and untouched.

### The renderer is testable without a GPU or a window

`vis::trace_pixel` and `vis::pack_pixel` are `__host__ __device__`, so a ray can be traced against
a hand-built scene on the host and the resulting pixel inspected as a number.

This is worth its own note because of what it replaced. Every renderer bug in this project was
found by taking a screenshot and looking at it - which works, and cost three consecutive wrong
diagnoses on one occasion, each a ten-minute GUI rebuild apart. `tests/test_render_layers.cu`
builds a phantom and a box, traces the axis, and asserts the colour: the overlap rules are four
rays and four expected colours. Both of the layer bugs above reproduced there in a millisecond.

### The wireframe pass, which the builder never launched

`EdgeList` - the twelve lines of a box - lived inside `vis_manager.cu`, so only the viewer could
reach it. The builder's styles said `solid = visible && !wireframe`, which is right, and nothing
drew the edges: a volume set to wireframe was **invisible** there, and the default world arrives
with wireframe on, so it never had an outline in the builder at all.

It is `src/render/edges.h` now and both apps use it.

**Every solid, in its own colour, in a framebuffer of its own.** Three things were wrong with the
first version and each was reported separately:

- **It drew boxes and voxel grids only**, behind a comment of mine arguing that a bounding cube
  round a cone would be "a lie about its shape". That is an argument against drawing a *cube*, not
  an argument for drawing *nothing*, and a sphere set to wireframe simply disappeared. Almost all
  of it turned out to be one function: a cylinder, a cone, a sphere, an ellipsoid, a paraboloid, a
  hyperboloid and a polycone are all surfaces of revolution about z, so each is a table of
  `(z, rx, ry)` levels handed to `AddProfile` - rings at the levels, meridians between them - and
  they differ only in the table. `kTrd` has its eight corners in `p[]`. `kPara`, `kTrap` and
  `kTet` keep no dimensions at all, only half-spaces `n.q <= d`, so their corners are recovered by
  intersecting plane triples and keeping the points that satisfy every plane; two vertices lying
  on the same two planes are the ends of an edge. Exact for any convex polyhedron, and it comes
  out agreeing with the engine's own plane convention because it uses the same planes.

  A polycone's shape is in the aux pool rather than in `p[]`, so `AddVolume` takes the pool -
  the same silent zero that `bounding_radius` carries a note about. A mesh and a boolean still
  get nothing: a CAD import is tens of thousands of triangles, and a boolean's shape is not its
  operands'.

  **Every line that follows a curve is drawn curved.** A meridian used to be one straight chord
  between consecutive rings - nine rings meaning an eight-sided longitude beside 48-segment
  latitudes - so the profile is sampled at 33 points for the meridians to follow and a ring is
  drawn every fourth sample. The torus had it in both families at once, each drawn with the
  OTHER family's station count: cross-sections of 8 segments and long-way rings of 16. How many
  curves and how many segments in one curve are different numbers. Longest segment on a sphere
  or a torus: 0.131 of the radius, against 0.39 before.

  One convention had to be derived rather than read, and it was wrong first: **a polyhedra's
  `rmax` is the apothem, not the corner radius.** `polyhedra_planes` puts face *f* at distance
  `r` from the axis along a normal at `sphi + step*(f + 0.5)`, so the faces are centred *between*
  the drawn vertices and a corner is `r / cos(pi/sides)` out - 15% further at six sides, which
  taken as a corner radius puts the outline inside the solid it is outlining. A probe of
  `geom::inside` across azimuth settled it in one run where reading the plane builder had not:
  inside to r = 23 at 0 and 60 degrees, inside only to r = 20 at 15, 30 and 45.

- **It dimmed the colour to 160/255.** A defensible look, and not what a colour picker is for.

- **It shared the geometry's framebuffer, so it inherited a depth test where it needed
  compositing.** `draw_line` ends in an `atomicMin` against the geometry's packed word. Among
  lines that is exactly right - they are opaque, so the nearest wins - but against *geometry* it
  is a pure depth test, and a line further away than the nearest surface was discarded outright
  even where that surface claimed 3% of the pixel and the other 97% was background. Reported as
  not seeing wireframe behind transparent objects, and a depth test cannot express it: the
  question is not which is nearer, it is how much of the pixel the nearer one took.

  The lines go into their own buffer, where `atomicMin` is still the right rule, and
  `vis::resolve_pixel` composites the two once per pixel with both depths in hand. Doing the
  read-modify-write inside the line pass instead would have made the picture depend on the order
  two lines happened to reach a pixel, which a frame-to-frame checksum comparison cannot live
  with. The line is opaque, so there are only two cases and no blending weight to pick: in front
  of the nearest surface it covers the pixel, and behind it it fills exactly the coverage the
  geometry left - taking the background's place, since nothing is further than the background.

  It also frees the ordering. The pass used to have to run after the anti-aliasing, because
  `refine_edges` retraces the geometry over the pixels it touches and would erase a line drawn
  under it. Separate buffers, so nothing constrains it.

  What it does not reproduce is a line *between* two translucent surfaces, which comes out behind
  both. That needs the line carried through the depth peeling in `trace_pixel`, and what it buys
  is a shade on a line that is already visible.

### The null layer is drawn the way a hidden object is drawn

A volume on the null layer is not placed, so there is nothing to draw and nothing to decide. A
null voxel *class* is different: the grid is still in the scene and only some of its cells are
gone, so the renderer has to be told something about them - and what it was told was a rule of its
own. An absent cell ranked below every volume, so any cover clamped the march there and a volume
sitting inside the hole was drawn in it; the cover scan admitted volumes it would otherwise reject,
and the world had to be excluded from that scan to stop the clamp ending the march at the first
cell.

That is a second rendering of the same scene, reachable only through the null layer, and it is not
the one that was wanted. The requirement in its own words: *turning off an object's visibility
renders it exactly how I want something in a null layer to be rendered.*

So a null class now arrives with **zero alpha**, through the same field a hidden class uses, and
the renderer has no null-layer rule at all. The march's one rule is the one it always had - a cell
whose class is not drawn is skipped and the walk goes on - and `FloatGeometry::Build` withholds
`class_absent` from the render geometry entirely, along with the per-volume `has_absent_classes`
flag. Withheld rather than tested for, because there is no branch to get wrong that way:
`voxel_cell_absent` returns false with no array, and `inside_volume` is gated on the flag.

Absence still reaches the **transport**, which shares the same arrays and is where a class not
being in the scene has to mean something: no material, no step, no score. What the removal costs is
the one thing the special case bought - a volume inside a nulled class's cells on a lower layer
than the grid is not drawn there, because the grid still owns that space as far as the picture is
concerned, exactly as it does for a hidden class.

What checks it is a comparison of the two pictures rather than a restatement of the rule: hide the
class and checksum the viewport, then put it back and null it instead, and require the two
checksums to be equal - with "hiding it changed the picture at all" as the precondition, since
without that everything below it passes on a class that was never on screen.

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
- A mesh has no wireframe, and neither does a boolean. Every other solid does - see
  `EdgeList::AddVolume` - but drawing every triangle edge of a 100,000-triangle import would put
  more line segments in the frame than the ray cast costs, and the useful thing is a silhouette,
  which needs the edges where the surface normal changes sign. A boolean would need its own
  surface: a subtraction drawn as both of its operands is drawn wrong.
- No dose overlay. Deposits are scored but not visualised; colour-mapping deposition onto the
  scoring volume would be a natural next step, and for a voxel volume it is the obvious way to
  look at a dose distribution at all.
- Solids are ray cast one at a time, so the render is O(volumes) per pixel. Fine for tens of
  volumes; a scene with hundreds wants the same BVH the navigator needs.
- Linux/macOS. The window layer is Win32-specific and the font atlas is GDI; the renderer
  itself is portable CUDA.
