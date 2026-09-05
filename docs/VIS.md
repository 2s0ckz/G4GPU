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
