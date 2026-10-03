@echo off
rem The other entry point: everything that is NOT physics, split off build_all.bat on 2026-10-02
rem (docs/RISK.md V220). It builds the viewer and the model-builder GUI; builds and runs the
rem render, UI and builder-scene tests, the trajectory store's, and the two custom-hook PROJECTS;
rem runs example B1's vis macros, the viewer and builder selftests and the render-off-the-UI-frame
rem benchmark; and builds, runs and compares the project the builder generates. Exits non-zero
rem on the first thing that fails.
rem
rem   build_extras.bat          build, run tests, run every check
rem   build_extras.bat build    build only
rem   build_extras.bat test     build, then tests only
rem
rem It builds on what build_all.bat builds - the engine archive, g4dose.exe, example B1 - through
rem the same incremental build scripts, so it runs on its own after a build_all.bat run or after
rem it in one chain. The two hook projects are the expensive half: each compiles its own project
rem object in one ptxas of about 45 minutes until the drains and the interaction kernels are
rem split per hook as the stepping kernels are (docs/RISK.md V210).
rem
rem When this must be green: any change under src/render, src/builder, src/scenes,
rem src/host/g4view.cu, src/host/g4builder.cu, tools/gen_hook_units.ps1, build_hook_engine.bat or
rem build_vis.bat, and any change to the kernels, the drains or the interaction units a hook
rem project instantiates - build_all.bat proves the stock engine, this proves a user's.
setlocal EnableDelayedExpansion
call "%~dp0setupenv.bat" || exit /b 1
cd /d "%~dp0"

rem A running executable cannot be relinked, and the error the linker gives is
rem "LNK1104: cannot open file 'g4builder.exe'" with no hint that a window is open. That cost
rem a pipeline run, so the check is here rather than in the reader's memory. The viewer and the
rem builder are the two that get left open; the rest are listed because the same applies.
for %%E in (g4builder.exe g4view.exe exampleB1.exe MyDetector.exe) do (
  tasklist /fi "imagename eq %%E" 2>nul | findstr /i /c:"%%E" >nul
  if not errorlevel 1 (
    echo FATAL: %%E is running, and a running executable cannot be relinked.
    echo        Close its window - or: taskkill /f /im %%E
    exit /b 1
  )
)

rem Every log this run writes carries a per-run id, so two builds - two worktrees integrating,
rem or a build beside a stray selftest - cannot read each other's output. They could: the
rem logs were fixed names under %TEMP%, and the findstr gates below would have passed on the
rem other run's file. RANDOM twice, because one is 15 bits.
set RUNID=%RANDOM%%RANDOM%
set MODE=%1
if "%MODE%"=="" set MODE=all

set SRC=%~dp0src
set NV=nvcc -std=c++17 -O2 -I "%SRC%"
set NVG=nvcc -std=c++17 -O2 %G4GPU_ARCH% -I "%SRC%"
rem The host-only tests of the viewer, the UI and the builder's importer. They compile in seconds,
rem like the physics tests, and read nothing of the engine.
set TESTS=test_ui_layout test_float_render test_wireframe test_render_layers test_voxel_import

rem test_custom_hook is a *project*, not a test of a function: it defines its own stepping
rem action, instantiates the engine for it, and links nothing of g4gpu's stock engine. It is
rem built the way a user project with a custom hook is built, so if that arrangement ever stops
rem working this fails at compile or link time.
rem
rem IT USED TO COMPILE ITS OWN KERNELS IN ONE TRANSLATION UNIT, and this comment used to call
rem the ~3.5 minutes that took "the honest cost of the arrangement". It was honest, it was
rem measured, and it was never necessary: the hook is a template parameter, so a project's
rem kernels split one-per-unit exactly as the engine's did in docs/RISK.md V65. What forced the
rem issue is that with V66's two switches on, that unit stopped compiling at all - ptxas died
rem with 0xC0000005 on run_step_hadron<double, ParticleType(13), QualityFactorScoring>, which is
rem V65's crash arriving through a different door, in a file the engine's split never touched.
rem build_hook_engine.bat below gives it the same structure the engine has.
set TESTS_PROJECT=test_custom_hook

rem Tests that run a real scene through the STOCK engine: they link out\transport_run.lib
rem rather than instantiating their own kernels, so they build in seconds where
rem TESTS_PROJECT takes minutes. test_trajectory needs a real shower because what it checks -
rem that a track's recorded segments join up - is a property of a path, and a synthetic step
rem has no path. test_voxel_materials needs one because what it checks - that a voxel cell is
rem transported as the material its class was assigned - was wrong for every phantom whose
rem classes used a material no ordinary volume did, and the symptom was a gamma crossing 200 mm
rem of tissue depositing nothing. See docs/RISK.md V20.
set TESTS_SCENE=test_trajectory test_voxel_materials test_voxel_layers

rem test_voxel_layers is here for the same reason: what it checks - that a voxel class on a
rem layer of its own transports as the volume that WINS its cells - is a property of a run,
rem and the equivalence it checks it against is a second run of a second scene.
rem
rem A project with its own step hook AND NOTHING ELSE LINKED. It builds its own detector, so
rem it needs no scene, and it must not be given one: a second translation unit that includes
rem g4/G4RunManager.hh without the same G4STEP_HOOK sees G4RunManager holding a differently
rem sized engine, which is one class with two layouts across the link - undefined, and the
rem kind of undefined that works until a member moves.
set TESTS_HOOK=test_voxel_scoring

echo --- drivers ---
rem Both call build_engine.bat, which is incremental: after a build_all.bat run this only links.
call "%~dp0build_view.bat" || exit /b 1
call "%~dp0build_gui.bat" || exit /b 1

echo --- example B1 (Geant4-shaped API) ---
call "%~dp0examples\B1\build.bat" || exit /b 1

echo --- tests ---
rem The host-only tests through tools\build_tests.ps1, as build_all.bat compiles its own (docs/RISK.md
rem V221): only the stale ones, six at a time. The projects and the scene tests below keep their own
rem recipes - each links something a plain test does not.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\build_tests.ps1" -Root "%~dp0." ^
  -Plain "%TESTS%" -Mode build -Jobs 6 || exit /b 1
rem THE TWO CUSTOM-HOOK PROJECTS ARE BUILT WITHOUT A `for` LOOP, and that is not an oversight.
rem Each one needs its own hook header, its own type name, its own archive and its own -I, so a
rem loop over a list of names could never have served two of them; the list existed because the
rem list had one entry. They are also the only two blocks here that cannot carry `rem` lines
rem inside them, since an unescaped `)` in a comment closes a parenthesised block early.
rem
rem Compiled to an OBJECT first, so the check build_engine.bat runs over the engine's object can
rem be run over this one. It has to be a check on the artefact: two objects holding the same
rem specialisation link cleanly on CUDA 11.6 - docs/RISK.md V65 - so a launch that lost its
rem declaration would simply be compiled here again, cost minutes of nvcc, and say nothing at
rem all. That is exactly how this file came to hold eighteen kernels.
rem
rem The hook class lives in tests\include so that the generated units and the project can both
rem see it, which is where a Geant4 project's action class lives anyway.
call "%~dp0build_hook_engine.bat" "%~dp0tests\include\QualityFactorScoring.hh" ^
  QualityFactorScoring qfs || exit /b 1
%NVG% -I "%SRC%\g4" -I "%~dp0tests\include" -I "%G4GPU_HOOK_INC%" ^
  -c -o out\test_custom_hook.obj tests\test_custom_hook.cu || exit /b 1
call :hook_object_gate test_custom_hook || exit /b 1
%NVG% -I "%SRC%\g4" -o tests\test_custom_hook.exe out\test_custom_hook.obj ^
  src\scenes\scene_b1.cu "%G4GPU_HOOK_LIB%" ^
  -Xlinker /IMPLIB:out/test_custom_hook.lib || exit /b 1
for %%T in (%TESTS_SCENE%) do (
  %NVG% -I "%SRC%\g4" -o tests\%%T.exe tests\%%T.cu src\scenes\scene_b1.cu ^
    "%~dp0out\transport_run.lib" -Xlinker /IMPLIB:out/%%T.lib || exit /b 1
)
rem The second custom-hook project, and it gets the same treatment for the same reason: its
rem eighteen kernels are CellTap's rather than QualityFactorScoring's, so they are a second
rem archive. It links no scene, for the reason its own comment above gives.
call "%~dp0build_hook_engine.bat" "%~dp0tests\include\CellTap.hh" CellTap cell || exit /b 1
%NVG% -I "%SRC%\g4" -I "%~dp0tests\include" -I "%G4GPU_HOOK_INC%" ^
  -c -o out\test_voxel_scoring.obj tests\test_voxel_scoring.cu || exit /b 1
call :hook_object_gate test_voxel_scoring || exit /b 1
%NVG% -I "%SRC%\g4" -o tests\test_voxel_scoring.exe out\test_voxel_scoring.obj ^
  "%G4GPU_HOOK_LIB%" -Xlinker /IMPLIB:out/test_voxel_scoring.lib || exit /b 1
rem From here on they are just tests - run and counted with the rest; the host-only list is kept
rem apart because the run schedules the two kinds differently, and every project and scene test
rem runs on the GPU.
set TESTS_HOST=%TESTS%
set TESTS_RUN_GPU=%TESTS_PROJECT% %TESTS_SCENE% %TESTS_HOOK%
set TESTS=%TESTS% %TESTS_PROJECT% %TESTS_SCENE% %TESTS_HOOK%
echo BUILD OK
if "%MODE%"=="build" exit /b 0

echo.
echo --- running tests ---
if "%G4GPU_ORACLE%"=="" set G4GPU_ORACLE=%~dp0ref\oracle
rem Six host-only tests at a time, the kernel-launching ones one at a time on the one GPU, and the
rem tail of any failing test's output printed - the loop this replaced ran them one by one into nul.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\build_tests.ps1" -Root "%~dp0." ^
  -Plain "%TESTS_HOST%" -Gpu "%TESTS_RUN_GPU%" -Mode run -Jobs 6 || exit /b 1
if "%MODE%"=="test" exit /b 0

echo.
echo --- example B1: the vis macros run, and produce pictures ---
rem init_vis.mac opens the viewer from the example rather than from g4view, which is what
rem G4VisExecutive's handler exists for; tsg_offscreen.mac then writes PNGs through the
rem /vis/ogl/export and /vis/tsg/offscreen commands. Both are checked because a macro that
rem printed FATAL eleven times and exited 0 is how a broken viewer went unnoticed before
rem (docs/RISK.md S5), and because "the offscreen commands work" is otherwise a claim with
rem nothing behind it.
pushd "%~dp0examples\B1"
del /q B1_*.png g4gpu_offscreen_*.png 2>nul
"%~dp0examples\B1\exampleB1.exe" tsg_offscreen.mac > "%TEMP%\g4gpu_%RUNID%_tsg.txt" 2>&1 || (
  echo FATAL: tsg_offscreen.mac failed.
  type "%TEMP%\g4gpu_%RUNID%_tsg.txt"
  popd
  exit /b 1
)
findstr /C:"ERROR" /C:"FATAL" /C:"unhandled" "%TEMP%\g4gpu_%RUNID%_tsg.txt" >nul
if not errorlevel 1 (
  echo FATAL: tsg_offscreen.mac reported an error.
  findstr /C:"ERROR" /C:"FATAL" /C:"unhandled" "%TEMP%\g4gpu_%RUNID%_tsg.txt"
  popd
  exit /b 1
)
rem Seven pictures, and each has to exist: an export command that logged "saved" without
rem writing anything would otherwise pass.
for %%P in (g4gpu_offscreen_1.png g4gpu_offscreen_2.png B1_1.png B1_2.png B1_zb.png B1_side.png B1_front.png) do (
  if not exist "%%P" (
    echo FATAL: tsg_offscreen.mac did not write %%P
    popd
    exit /b 1
  )
)
echo   seven offscreen pictures written
popd

echo --- viewer selftest ---
rem An interactive window cannot be tested by clicking, so the viewer drives its own camera
rem and one run for a fixed number of frames and saves a PNG. This catches a broken Win32,
rem WGL, CUDA-render or font path, all of which are invisible to every other check here.
"%~dp0g4view.exe" -selftest > "%TEMP%\g4gpu_%RUNID%_view.txt" 2>&1 || exit /b 1
rem A logged failure is fatal, same as for the builder. Without this the viewer's selftest was
rem satisfied by writing a PNG, whatever was in it.
findstr /C:"selftest: FAILED" "%TEMP%\g4gpu_%RUNID%_view.txt" >nul
if not errorlevel 1 (
  echo FATAL: the viewer selftest logged a failure.
  findstr /C:"selftest: FAILED" "%TEMP%\g4gpu_%RUNID%_view.txt"
  exit /b 1
)
rem Solid geometry has to actually render.
rem
rem Everything else about the viewer is satisfied by a picture made entirely of lines - the PNG
rem is written, the offscreen files exist, the trajectory count is non-zero. So when
rem VolumeStyle grew an alpha channel and the viewer's style setup was not updated to fill it
rem in, every volume sat at alpha zero, render_geometry skipped all of them, and the viewer
rem drew no solid geometry at all while every check kept passing. The selftest now turns the
rem line passes off and counts what the solid pass covered.
findstr /C:"selftest: solid geometry rendered" "%TEMP%\g4gpu_%RUNID%_view.txt" >nul
if errorlevel 1 (
  echo FATAL: the viewer selftest did not confirm that solid geometry renders.
  exit /b 1
)
findstr /C:"selftest:" "%TEMP%\g4gpu_%RUNID%_view.txt" || exit /b 1

echo.
echo --- model builder selftest ---
rem The builder's selftest inserts solids, runs them, and writes a project. That project is
rem then compiled and run here, because "Save gives you a compilable project" is a claim only
rem worth something if something checks it - and generated code that does not compile is
rem exactly the breakage no other test in this pipeline would notice.
if exist "%~dp0out\selftest_project" rd /s /q "%~dp0out\selftest_project"
"%~dp0g4builder.exe" -selftest > "%TEMP%\g4gpu_%RUNID%_builder.txt" 2>&1 || exit /b 1
findstr /C:"selftest:" "%TEMP%\g4gpu_%RUNID%_builder.txt" || exit /b 1
rem "selftest:" alone is satisfied by a selftest that logged a failure and carried on, and the
rem subtraction step did exactly that: it picked the imported mesh as the boolean's operand,
rem which the boolean engine rightly refuses. So the two shapes that need the shared solid
rem store are named explicitly here, and a logged FAILED is fatal.
findstr /C:"added a subtraction and a polycone" "%TEMP%\g4gpu_%RUNID%_builder.txt" >nul || (
  echo FATAL: the builder selftest did not insert the boolean and the polycone,
  echo        so nothing exercised the renderer's solid store.
  type "%TEMP%\g4gpu_%RUNID%_builder.txt"
  exit /b 1
)
findstr /C:"a run with a same-layer overlap was refused" "%TEMP%\g4gpu_%RUNID%_builder.txt" >nul || (
  echo FATAL: the builder selftest did not prove the same-layer overlap refusal.
  echo        Two volumes overlapping on one layer have no rule deciding which owns the
  echo        shared space, so a run must be refused rather than report a dose that depends
  echo        on the order the detector was built in.
  type "%TEMP%\g4gpu_%RUNID%_builder.txt"
  exit /b 1
)
rem And the same rule where the layer belongs to a CLASS rather than to the volume: a phantom
rem on layer 2 whose bone class is on 3 clashes with a box on 3, in those cells and nowhere
rem else. The check compared the two VOLUMES and saw nothing at all.
findstr /C:"a voxel class on another volume" "%TEMP%\g4gpu_%RUNID%_builder.txt" >nul || (
  echo FATAL: the builder selftest did not prove the PER-CLASS same-layer refusal.
  echo        A voxel class raised to another volume's layer shares that space with it at
  echo        the same layer, so which one owns it depends on list order and a run must be
  echo        refused rather than report a dose that depends on the build order.
  type "%TEMP%\g4gpu_%RUNID%_builder.txt"
  exit /b 1
)
findstr /C:"it clears when the class moves off that layer" "%TEMP%\g4gpu_%RUNID%_builder.txt" >nul || (
  echo FATAL: the per-class overlap check did not CLEAR when the class moved off that layer,
  echo        so it is flagging pairs on their layer RANGE rather than on where the two are
  echo        actually on one layer - which would refuse every phantom with a raised class.
  type "%TEMP%\g4gpu_%RUNID%_builder.txt"
  exit /b 1
)
rem Two faces in exactly the same place, on different layers: the higher one has to be drawn.
rem Neither was - the surface search took whichever it found first, the ownership test threw it
rem away, and the next iteration was already inside both. A hole where two faces meet.
findstr /C:"where two faces coincide" "%TEMP%\g4gpu_%RUNID%_builder.txt" >nul || (
  echo FATAL: the builder selftest did not prove that coincident faces draw the higher layer.
  type "%TEMP%\g4gpu_%RUNID%_builder.txt"
  exit /b 1
)
findstr /C:"selftest: FAILED" "%TEMP%\g4gpu_%RUNID%_builder.txt" >nul
if not errorlevel 1 (
  echo FATAL: the builder selftest logged a failure.
  findstr /C:"selftest: FAILED" "%TEMP%\g4gpu_%RUNID%_builder.txt"
  exit /b 1
)
rem The builder's own run has to report a dose that is not zero, for the same reason the
rem generated project's does - and for a sharper one. The selftest used to run 25 events,
rem which through a 30 mm cube is a fraction of one expected interaction: the dose was one
rem rare event or nothing at all, and it read 2.81 MeV or exactly 0 depending on where the
rem RNG stream landed. Anything that shifted that stream - more steps per track - looked
rem exactly like a transport bug, and cost most of a day. The run is now large enough for the
rem number to be stable, and this is the check that says so.
findstr /C:"dose 0 pGy" "%TEMP%\g4gpu_%RUNID%_builder.txt" >nul
if not errorlevel 1 (
  echo FATAL: the builder selftest scored a dose of exactly zero.
  findstr /C:"dose" "%TEMP%\g4gpu_%RUNID%_builder.txt"
  exit /b 1
)
rem Per-voxel scoring, which is checked inside the selftest by summing the cells and comparing
rem with the volume total the same run reported. This only asserts that the check ran at all:
rem a selftest that stopped exercising it would otherwise pass silently.
findstr /C:"selftest: per-voxel scoring:" "%TEMP%\g4gpu_%RUNID%_builder.txt" >nul
if errorlevel 1 (
  echo FATAL: the builder selftest did not exercise per-voxel scoring.
  exit /b 1
)
rem The dialogs. Every other check on the builder reads a number, and a dialog whose text ran
rem past its frame, whose buttons fell off the bottom, or whose label overflowed its own box
rem would look correct to all of them - which is how a button caption came to straddle the edge
rem of the dialog it was in. The selftest holds each one up for a frame and photographs it;
rem this asserts the pictures exist, and a person can look at them when something reads oddly.
rem voxel_open is the voxel dialog with one of its dropdowns open. It is here because a
rem dropdown declared inside a pop-up was painted at the panel layer and then covered by the
rem pop-up that owned it, so it did not appear at all - the one GUI defect in docs/RISK.md V14
rem that no check reading numbers could ever have seen.
for %%D in (world voxel voxel_open voxel_cmap anchor physics vis class_color) do (
  if not exist "%~dp0out\g4builder_dlg_%%D.png" (
    echo FATAL: the builder selftest did not capture the %%D dialog.
    exit /b 1
  )
)
echo the eight dialogs were captured
rem The custom-scorer flag must not change the number a scorer reports until the generated
rem file is edited. It is not free: a filtered scorer's total is accumulated event by event on
rem the host through Accept(), while a stock one's comes from the device array. The selftest
rem runs the same seed both ways and requires them to agree exactly.
findstr /C:"selftest: a custom scorer reports what the stock one does" "%TEMP%\g4gpu_%RUNID%_builder.txt" >nul
if errorlevel 1 (
  echo FATAL: the builder selftest did not check the custom-scorer equivalence.
  exit /b 1
)
rem THE NULL LAYER: a solid on it is not in the scene, and comes back when it is put back.
rem
rem Two claims that fail separately - one volume fewer in the flattened scene, which is what
rem makes the transport and the tally ignore it, and a changed picture. Each of them is also
rem satisfied by a fixture that was never on screen, so the selftest proves the box visible
rem first by recolouring it; both fixtures were once placed outside the world, where nothing
rem is drawn, and passed.
findstr /C:"a solid on the null layer leaves the scene" "%TEMP%\g4gpu_%RUNID%_builder.txt" >nul
if errorlevel 1 (
  echo FATAL: the builder selftest did not prove a null-layer solid leaves the scene.
  exit /b 1
)
findstr /C:"and comes back to exactly the picture it left" "%TEMP%\g4gpu_%RUNID%_builder.txt" >nul
if errorlevel 1 (
  echo FATAL: the builder selftest did not prove a null-layer solid can be put back.
  exit /b 1
)
rem And a voxel class on it, checked on a grid with NOTHING over it - on a covered one the
rem clamp removes such a cell whether the code means to or not, and breaking the rule
rem deliberately left that check passing.
findstr /C:"a voxel class on the null layer is not drawn" "%TEMP%\g4gpu_%RUNID%_builder.txt" >nul
if errorlevel 1 (
  echo FATAL: the builder selftest did not prove a null-layer voxel class is not drawn.
  exit /b 1
)
rem The world is asked about before it moves off layer 0, and refused the null layer outright.
findstr /C:"the world layer is confirmed before it moves" "%TEMP%\g4gpu_%RUNID%_builder.txt" >nul
if errorlevel 1 (
  echo FATAL: the builder selftest did not prove the world's layer is confirmed.
  exit /b 1
)
rem A NULLED CELL IS NOBODY'S SPACE, AND A HIDDEN ONE IS STILL THE GRID'S.
rem
rem A voxel is its own volume as far as owning space goes, so a class that is not in the scene
rem has no layer to compare and whatever is in those cells owns them - including a volume on a
rem LOWER layer than the grid, which is what makes this unlike every other overlap. Hiding a
rem class is a different statement: the cells are still there and still outrank what is inside
rem them. A previous version of this check asserted the two gave the same picture; they do not,
rem wherever something is inside the cells.
rem
rem Both halves are checked, because each alone is satisfied by a broken renderer: drawn in the
rem hole by one that has stopped honouring layers, hidden when the class is back by one that
rem never draws it.
findstr /C:"gives its space to the volume inside it" "%TEMP%\g4gpu_%RUNID%_builder.txt" >nul
if errorlevel 1 (
  echo FATAL: the builder selftest did not prove a nulled class hands over its space.
  exit /b 1
)
rem AND THE WIREFRAME PASS RUNS. The builder never launched one: its styles said
rem `solid = visible && !wireframe`, which is right, and nothing drew the edges - so a volume
rem set to wireframe was invisible, and the default world arrives with wireframe on and had no
rem outline at all.
findstr /C:"the wireframe pass draws edges" "%TEMP%\g4gpu_%RUNID%_builder.txt" >nul
if errorlevel 1 (
  echo FATAL: the builder selftest did not prove the wireframe pass draws anything.
  exit /b 1
)
rem AND A ROUND SOLID HAS ONE, IN ITS OWN COLOUR, VISIBLE THROUGH GLASS.
rem
rem The line above passes on a box and the world is a box, which is why it passed while a
rem sphere set to wireframe was invisible: the pass drew boxes and voxel grids and nothing
rem else. One fixture answers all four of the reports - an orb contributes 720 segments where
rem a box contributes twelve, recolouring it moves the picture, and it sits wholly behind a
rem translucent pane so every edge that reaches the screen came through it. The opaque half of
rem that pair is what stops an x-ray line pass passing as compositing.
rem
rem And the COLOUR, channel by channel, against the model's own floats - not "recolouring it
rem changed the picture", which passes on every colour there is and did pass over an outline
rem drawn with its red and blue exchanged. The check counts pixels that are exactly the orb's
rem colour and requires the reversed colour to appear zero times.
findstr /C:"pixels of them exactly its own colour" "%TEMP%\g4gpu_%RUNID%_builder.txt" >nul
if errorlevel 1 (
  echo FATAL: the builder selftest did not prove a round solid has a wireframe.
  exit /b 1
)
rem ANTI-ALIASING puts partial coverage where a surface ends, and nowhere else.
rem
rem One ray per pixel makes a silhouette a staircase: the pixel is the surface or it is not, so
rem its coverage is 0 or 255 and never between. Four rays and an average is what produces the
rem values between, so counting them counts the smoothing - measured on ONE OPAQUE BOX with the
rem rest of the scene hidden, because a translucent volume is partly covering every pixel it
rem touches and would swamp the few hundred that are edge.
findstr /C:"anti-aliasing softens the silhouette" "%TEMP%\g4gpu_%RUNID%_builder.txt" >nul
if errorlevel 1 (
  echo FATAL: the builder selftest did not prove the render is anti-aliased.
  exit /b 1
)
findstr /C:"turning it off changes the picture back" "%TEMP%\g4gpu_%RUNID%_builder.txt" >nul
if errorlevel 1 (
  echo FATAL: the builder selftest did not prove the anti-aliasing switch works.
  exit /b 1
)
rem THE INSERT FORM: sized from the world, and it inserts what it shows. Two claims, and the
rem first is the one a constant cannot meet - the same fraction that gives a sensible box in a
rem 500 mm world gives a speck in a 10 m one.
findstr /C:"the insert form seeds half the world" "%TEMP%\g4gpu_%RUNID%_builder.txt" >nul
if errorlevel 1 (
  echo FATAL: the builder selftest did not prove the insert form is sized from the world.
  exit /b 1
)
findstr /C:"it inserts exactly the size and position it was showing" "%TEMP%\g4gpu_%RUNID%_builder.txt" >nul
if errorlevel 1 (
  echo FATAL: the builder selftest did not prove the insert form inserts what it shows.
  exit /b 1
)
rem And that the layer menu can say "null" at all, and maps every entry to the layer it names.
rem An off-by-one between the option INDEX and the layer NUMBER would put every solid one layer
rem out, silently, and the option list is the only place that mapping exists.
findstr /C:"every entry maps to the layer it names" "%TEMP%\g4gpu_%RUNID%_builder.txt" >nul
if errorlevel 1 (
  echo FATAL: the builder selftest did not prove the layer menu maps its entries.
  exit /b 1
)
rem A VOLUME ON THE WORLD'S OWN LAYER clashes with the world, and the run is refused.
rem
rem The world used to be exempt from the overlap check, on the reasoning that it contains
rem everything so an overlap with it is containment. True for layer 1 and above - and that case
rem needs no exemption, since the layer-range test prunes it. What the exemption hid was a
rem volume placed ON layer 0, which is a real same-layer overlap resolved only by "whichever was
rem added later wins" - the tie-break this whole check exists to refuse.
findstr /C:"a volume on the world's own layer clashes with the world" "%TEMP%\g4gpu_%RUNID%_builder.txt" >nul
if errorlevel 1 (
  echo FATAL: the builder selftest did not prove a layer-0 volume clashes with the world.
  exit /b 1
)
findstr /C:"it clears when the volume moves off that layer" "%TEMP%\g4gpu_%RUNID%_builder.txt" >nul
if errorlevel 1 (
  echo FATAL: the builder selftest did not prove that clash clears again.
  exit /b 1
)
rem The composited viewport must BE the device image - the blit's placement, stride and
rem completeness, compared against d_rgba rather than against another frame that went through
rem the same blit. A checksum comparison cannot see this: break the row stride and both sides
rem of it are wrong identically. Verified by doing exactly that.
findstr /C:"the composited viewport is the device image" "%TEMP%\g4gpu_%RUNID%_builder.txt" >nul
if errorlevel 1 (
  echo FATAL: the builder selftest did not compare the composited viewport with the device.
  exit /b 1
)
rem And the selftest's own check that the polling path draws the same picture the waiting one does.
rem Everything else in the selftest runs with the render inside the frame, so without this the
rem staging swap, the row stride and the adopt would ship unexercised.
findstr /C:"the render off the UI frame draws the same picture" "%TEMP%\g4gpu_%RUNID%_builder.txt" >nul
if errorlevel 1 (
  echo FATAL: the builder selftest did not prove the async render draws the same picture.
  exit /b 1
)
echo --- the render does not hold the UI up ---
rem A GUI whose frame waits for its own render is a GUI that stops responding when the scene
rem gets heavy, and that is what "hovering over a button that should change colour, it only
rem changes about a full second later" was. -benchmesh times the same scene twice in one
rem process - render inside the frame, then render on its own stream - and reports whether the
rem UI frame still pays for it.
rem
rem A SMALL mesh on purpose. The invariant checked here is that the UI frame spends nothing on
rem the render, and that does not need a slow render to fail: a cudaDeviceSynchronize put back
rem into DrawFrame breaks it on any scene at all. The big-mesh numbers, where the payoff is
rem visible, are in docs\VIS.md - they cost forty seconds that nothing here is waiting on.
"%~dp0g4builder.exe" -benchmesh 200000 0.40 3 > "%TEMP%\g4gpu_%RUNID%_bench.txt" 2>&1 || exit /b 1
findstr /C:"benchmesh: render is off the UI frame" "%TEMP%\g4gpu_%RUNID%_bench.txt" >nul
if errorlevel 1 (
  echo FATAL: the render is back inside the UI frame.
  findstr /C:"benchmesh:" "%TEMP%\g4gpu_%RUNID%_bench.txt"
  exit /b 1
)
rem And when the render IS slower than the refresh interval, the UI frame has to be faster than
rem it. Below that both are vsync-limited and -benchmesh says so instead, which is why this is
rem a check for the failure string rather than for a success one.
findstr /C:"benchmesh: NOT DECOUPLED" "%TEMP%\g4gpu_%RUNID%_bench.txt" >nul
if not errorlevel 1 (
  echo FATAL: a render slower than the refresh interval still slowed the UI frame.
  findstr /C:"benchmesh:" "%TEMP%\g4gpu_%RUNID%_bench.txt"
  exit /b 1
)
findstr /C:"benchmesh:" "%TEMP%\g4gpu_%RUNID%_bench.txt" || exit /b 1
rem The generated project has to be a Geant4 project, not three files that happen to compile.
rem It shipped DetectorConstruction, PrimaryGeneratorAction and RunAction and nothing else -
rem no ActionInitialization, no EventAction, no SteppingAction - so there was nowhere to put
rem per-event or per-step code, which is the first thing somebody opening it wants to do.
for %%F in (ActionInitialization EventAction SteppingAction RunAction DetectorConstruction ^
            PrimaryGeneratorAction) do (
  if not exist "%~dp0out\selftest_project\src\%%F.cc" (
    echo FATAL: the generated project has no src\%%F.cc
    exit /b 1
  )
  if not exist "%~dp0out\selftest_project\include\%%F.hh" (
    echo FATAL: the generated project has no include\%%F.hh
    exit /b 1
  )
)
for %%F in (CMakeLists.txt README build.bat run1.mac) do (
  if not exist "%~dp0out\selftest_project\%%F" (
    echo FATAL: the generated project has no %%F
    exit /b 1
  )
)
rem One file pair per scorer flagged custom. The selftest flags "cells", so this pair has to
rem exist and has to be in the build - it is compiled by build.bat below, which is the check
rem that the unit list and the emitted files agree.
if not exist "%~dp0out\selftest_project\src\ScorerCells.cc" (
  echo FATAL: the generated project has no custom scorer class.
  exit /b 1
)
rem The generated project must build its primaries the way the architecture requires.
rem
rem GeneratePrimaryVertex is the only route a primary reaches the transport, so a generated
rem generator that does not call it produces nothing - the run manager refuses with a FATAL
rem rather than transporting an empty run. Checking for it is cheap and it is stable: it is the
rem architecture, not an implementation detail of the emitter.
rem
rem That distinction has now cost two pipeline runs. This check first looked for `kSourceCdf`,
rem a symbol from a per-event sampler that turned out to be wrong; fixing the bug deleted the
rem symbol and the check failed on the fix. It was then rewritten to look for `AddSource`, from
rem the declarative multi-source API that replaced it - and that API was deleted an hour later
rem when per-event generation made it unnecessary, so the check failed on that fix too. A check
rem written against whatever the code currently says goes stale exactly when the code improves.
rem
rem What actually establishes that both of a two-source model's beams fire is the numerical
rem comparison against the builder, further down. This is the cheap early signal, nothing more.
findstr /C:"GeneratePrimaryVertex" "%~dp0out\selftest_project\src\PrimaryGeneratorAction.cc" >nul
if errorlevel 1 (
  echo FATAL: the generated primary generator never calls GeneratePrimaryVertex,
  echo        so it produces no primaries at all.
  exit /b 1
)
echo the generated project has the full Geant4 file set

call "%~dp0out\selftest_project\build.bat" || exit /b 1
"%~dp0out\selftest_project\MyDetector.exe" -n 10000 > "%TEMP%\g4gpu_%RUNID%_gen.txt" 2>&1 || exit /b 1
findstr /C:"events" "%TEMP%\g4gpu_%RUNID%_gen.txt" >nul || exit /b 1
rem The generated project has a dose scorer on the imported mesh, so it has to report a dose
rem that is not zero. Requiring only the word "dose" was satisfied by "dose 0 pGy", which is
rem what a mesh made of air deposits - and what the check reported as a pass.
findstr /C:"dose" "%TEMP%\g4gpu_%RUNID%_gen.txt" || exit /b 1
findstr /C:"dose 0 pGy" "%TEMP%\g4gpu_%RUNID%_gen.txt" >nul
if not errorlevel 1 (
  echo FATAL: the generated project scored a dose of exactly zero.
  type "%TEMP%\g4gpu_%RUNID%_gen.txt"
  exit /b 1
)
rem The generated project must agree with the builder about the same model.
rem
rem This is the check that was missing. build_scene.hh exists so that "run it here" and "save
rem it and run that" cannot disagree about what a model means - but the *source* went through a
rem separate emitter, and nothing compared the two. So a generated PrimaryGeneratorAction that
rem fired one of the model's two beams for the whole run passed everything: it built, it ran,
rem and it reported a dose that was not zero.
rem
rem A million events each, same model. The builder logs "selftest: compare <scorer> <MeV>";
rem the project prints the same scorers in its end-of-run block. A one-beam-instead-of-two run
rem misses by tens of percent.
rem
rem A million, and not the 200000 this ran for most of its life, because these are two
rem INDEPENDENT Monte Carlos and their difference is noise until the statistics say otherwise.
rem Measured on this model: 6.4% apart at 200000 histories, 0.80% at a million, -0.08% at four
rem million - 1/sqrt(N), which is what agreement looks like. The threshold in
rem compare_project.ps1 is 6%, and at 200000 the spread between two independent sides is 4.4%
rem on dose1, so it sat 1.36 sigma out: a coin toss, and every pass it had ever given was luck
rem rather than evidence. See docs/RISK.md V10.
"%~dp0out\selftest_project\MyDetector.exe" -n 1000000 > "%TEMP%\g4gpu_%RUNID%_cmp.txt" 2>&1 || exit /b 1
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\compare_project.ps1" ^
  -Builder "%TEMP%\g4gpu_%RUNID%_builder.txt" -Project "%TEMP%\g4gpu_%RUNID%_cmp.txt" || exit /b 1

echo generated project compiled and ran

echo.
echo ALL OK (extras)
exit /b 0

rem ---------------------------------------------------------------- the custom-hook gate
rem
rem %1 is a test name; out\%1.obj must carry no stepping kernel.
rem
rem The whole per-hook split rests on the `extern template` declarations in the generated
rem hook_kernels.cuh suppressing implicit instantiation in the project that launches the
rem kernels, and the LINK will not catch a mistake: two objects holding the same specialisation
rem link cleanly on CUDA 11.6, the stubs being COMDAT-folded, so a launch added without a
rem declaration costs minutes of nvcc and says nothing at all. build_engine.bat runs this same
rem cuobjdump over the engine's own object; docs/RISK.md V65 has both halves and the inversion.
rem
rem A subroutine rather than the same six lines twice, and a subroutine rather than a `for`
rem body, because a `rem` containing a closing parenthesis inside a parenthesised block ends
rem the block there.
:hook_object_gate
for /f "usebackq delims=" %%N in (`cuobjdump -res-usage "%~dp0out\%~1.obj" ^| findstr /c:"run_step_"`) do (
  echo FATAL: out\%~1.obj carries a stepping kernel:
  echo        %%N
  echo        A launch was added without a matching declaration in the generated
  echo        hook_kernels.cuh, so that kernel is compiled into the project's own
  echo        translation unit again. See docs/RISK.md V65.
  exit /b 1
)
exit /b 0
