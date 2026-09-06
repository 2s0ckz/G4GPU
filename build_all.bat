@echo off
rem The one entry point: builds every driver and test, runs the tests, and checks the dose
rem against Geant4. Exits non-zero on the first thing that fails.
rem
rem   build_all.bat          build, run tests, run the 2M-event dose check
rem   build_all.bat build    build only
rem   build_all.bat test     build, then tests only
rem
rem For iterating, not for deciding: tools\quick.ps1 builds and runs a named subset in seconds
rem rather than the twenty minutes this takes, most of which is rebuilding the transport
rem engine, the viewer, the GUI and example B1.
rem
rem   tools\quick.ps1 hadron          every test matching *hadron*
rem   tools\quick.ps1 -Check proton   the proton depth-dose comparison alone
rem   tools\quick.ps1 -List           what there is
rem
rem Nothing there replaces this file. The sigma gates, the batch-vs-macro, mesh, per-voxel and
rem generated-project comparisons and both selftests live only here, and no test links the
rem drivers - so a change under src/host, src/render, src/builder or src/g4 is not tested by
rem anything quick.ps1 can run.
rem
rem It runs the whole pipeline because building alone proved not to be evidence of anything:
rem a driver that compiled and printed a dose had two physics processes silently switched off.
setlocal EnableDelayedExpansion
call "D:/g4gpu/setupenv.bat" || exit /b 1
cd /d "%~dp0"

rem A running executable cannot be relinked, and the error the linker gives is
rem "LNK1104: cannot open file 'g4builder.exe'" with no hint that a window is open. That cost
rem a pipeline run, so the check is here rather than in the reader's memory. The viewer and the
rem builder are the two that get left open; the rest are listed because the same applies.
for %%E in (g4builder.exe g4view.exe g4dose.exe b1_gpu_sched.exe exampleB1.exe MyDetector.exe) do (
  tasklist /fi "imagename eq %%E" 2>nul | findstr /i /c:"%%E" >nul
  if not errorlevel 1 (
    echo FATAL: %%E is running, and a running executable cannot be relinked.
    echo        Close its window - or: taskkill /f /im %%E
    exit /b 1
  )
)

set MODE=%1
if "%MODE%"=="" set MODE=all

set SRC=%~dp0src
set NV=nvcc -std=c++17 -O2 -I "%SRC%"
set NVG=nvcc -std=c++17 -O2 -arch=sm_86 -I "%SRC%"
set TESTS=test_core test_geometry test_navigation test_solids test_voxels test_mesh test_gamma_xs test_electron test_photoelectric test_brems test_rayleigh test_rayleigh_angular test_cuts test_general test_msc test_annihilation test_brems_rel test_hadron test_hadron_range test_hadron_delta test_fluctuation test_density_effect test_corrections test_muon test_hadron_radiative test_wentzel test_bragg test_ion_charge test_constants test_icru90 test_mott test_material_build test_all_materials test_nuclear_stopping test_wentzel_msc test_icru73qo test_nucleon_xs test_ion_fluctuation test_urban_general test_gun_position test_vs_oracle

echo --- drivers ---
rem /IMPLIB keeps the import library and its .exp out of the project root. They are a
rem link-time artefact of an exe that exports symbols, not something anyone runs.
%NVG% -o b1_gpu_sched.exe src\host\b1_gpu_sched.cu -Xlinker /IMPLIB:out/b1_gpu_sched.lib || exit /b 1
call "%~dp0build_dose.bat" || exit /b 1
call "%~dp0build_view.bat" || exit /b 1
call "%~dp0build_gui.bat" || exit /b 1

echo --- example B1 (Geant4-shaped API) ---
call "%~dp0examples\B1\build.bat" || exit /b 1

echo --- tests ---
for %%T in (%TESTS%) do (
  %NV% -o tests\%%T.exe tests\%%T.cu || exit /b 1
)
echo BUILD OK
if "%MODE%"=="build" exit /b 0

echo.
echo --- running tests ---
if "%G4GPU_ORACLE%"=="" set G4GPU_ORACLE=%~dp0ref\oracle
set PASS=0
set FAIL=0
set FAILED=
for %%T in (%TESTS%) do (
  tests\%%T.exe >nul 2>&1
  if errorlevel 1 (
    set /a FAIL+=1
    set FAILED=!FAILED! %%T
  ) else (
    set /a PASS+=1
  )
)
echo !PASS! passed, !FAIL! failed
if not "!FAILED!"=="" (
  echo FAILED:!FAILED!
  exit /b 1
)
if "%MODE%"=="test" exit /b 0

echo.
echo --- dose vs Geant4, 2M events ---
"%~dp0b1_gpu_sched.exe" 2000000 > "%TEMP%\g4gpu_dose.txt" 2>&1 || exit /b 1
findstr /C:"FATAL" "%TEMP%\g4gpu_dose.txt" >nul 2>&1
if not errorlevel 1 (
  type "%TEMP%\g4gpu_dose.txt"
  exit /b 1
)
findstr /C:"photoelectric:" /C:"rayleigh:" /C:"bremsstrahlung:" /C:"scaled to 10k" /C:"Geant4 11.1.1" /C:"uncertainty" "%TEMP%\g4gpu_dose.txt"
echo.
echo --- example B1, 2M events ---
"%~dp0examples\B1\exampleB1.exe" -n 2000000 > "%TEMP%\g4gpu_b1.txt" 2>&1 || exit /b 1
findstr /C:"FATAL" "%TEMP%\g4gpu_b1.txt" >nul 2>&1
if not errorlevel 1 (
  type "%TEMP%\g4gpu_b1.txt"
  exit /b 1
)
findstr /C:"scaled to 10k" /C:"Geant4 11.1.1" /C:"difference" "%TEMP%\g4gpu_b1.txt"
rem And *checked*, not merely printed. The agreement with Geant4 is what this project claims;
rem printing it into a log for a person to read is not a test of it, and a drift to five sigma
rem would have passed here and been reported as ALL OK.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\check_sigma.ps1" ^
  -Log "%TEMP%\g4gpu_b1.txt" || exit /b 1

echo.
rem The same example, in batch and driven by a macro, must give the same answer.
rem
rem Nothing compared them until now, and they disagreed by a third. vis.mac set
rem `/gun/beamRectangular 8 8 cm` so that pressing Run in the viewer fired what the example
rem fires; B1's generator was later rewritten to draw its own spot and call
rem SetParticlePosition, which set the *centre* of that cross-section rather than the position.
rem The spot was sampled twice, a third of the beam missed the detector, and the interactive run
rem reported 287 pGy against batch's 429. Both numbers looked reasonable in isolation.
rem
rem The pipeline runs everything in batch, so every interactive path - the viewer's Run button,
rem any macro that touches the gun - was exercised only by a person. See docs/RISK.md V3.
pushd "%~dp0examples\B1"
"%~dp0examples\B1\exampleB1.exe" gunstate.mac > "%TEMP%\g4gpu_b1mac.txt" 2>&1 || (popd & exit /b 1)
popd
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\compare_runs.ps1" ^
  -BatchLog "%TEMP%\g4gpu_b1.txt" -MacroLog "%TEMP%\g4gpu_b1mac.txt" || exit /b 1

echo --- example B1: the vis macros run, and produce pictures ---
rem init_vis.mac opens the viewer from the example rather than from g4view, which is what
rem G4VisExecutive's handler exists for; tsg_offscreen.mac then writes PNGs through the
rem /vis/ogl/export and /vis/tsg/offscreen commands. Both are checked because a macro that
rem printed FATAL eleven times and exited 0 is how a broken viewer went unnoticed before
rem (docs/RISK.md S5), and because "the offscreen commands work" is otherwise a claim with
rem nothing behind it.
pushd "%~dp0examples\B1"
del /q B1_*.png g4gpu_offscreen_*.png 2>nul
"%~dp0examples\B1\exampleB1.exe" tsg_offscreen.mac > "%TEMP%\g4gpu_tsg.txt" 2>&1 || (
  echo FATAL: tsg_offscreen.mac failed.
  type "%TEMP%\g4gpu_tsg.txt"
  popd
  exit /b 1
)
findstr /C:"ERROR" /C:"FATAL" /C:"unhandled" "%TEMP%\g4gpu_tsg.txt" >nul
if not errorlevel 1 (
  echo FATAL: tsg_offscreen.mac reported an error.
  findstr /C:"ERROR" /C:"FATAL" /C:"unhandled" "%TEMP%\g4gpu_tsg.txt"
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

echo --- mesh transport: the same geometry as a solid and as a mesh ---
rem B1's scoring volume is a G4Trd; B1mesh's is a twelve-triangle mesh of the identical
rem trapezoid. Same seed, same source, so the two must score the same. This is the check that
rem the triangles and the BVH actually reached the device: tests\test_mesh.exe exercises the
rem mesh routines on the host, and would pass with an upload path that copied nothing.
"%~dp0g4dose.exe" -compare B1 B1mesh -n 500000 || exit /b 1

echo.
echo --- every physics switch changes the answer ---
rem Eight switches once sat in the GUI, were written into generated projects, and were never
rem read by the stepper: turning Compton off changed the dose by exactly zero. See RISK.md A3.
rem This runs the scene once per switch and fails if any of them changes nothing.
"%~dp0g4dose.exe" -verify-processes -n 200000 || exit /b 1
echo.
echo --- viewer selftest ---
rem An interactive window cannot be tested by clicking, so the viewer drives its own camera
rem and one run for a fixed number of frames and saves a PNG. This catches a broken Win32,
rem WGL, CUDA-render or font path, all of which are invisible to every other check here.
"%~dp0g4view.exe" -selftest > "%TEMP%\g4gpu_view.txt" 2>&1 || exit /b 1
rem A logged failure is fatal, same as for the builder. Without this the viewer's selftest was
rem satisfied by writing a PNG, whatever was in it.
findstr /C:"selftest: FAILED" "%TEMP%\g4gpu_view.txt" >nul
if not errorlevel 1 (
  echo FATAL: the viewer selftest logged a failure.
  findstr /C:"selftest: FAILED" "%TEMP%\g4gpu_view.txt"
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
findstr /C:"selftest: solid geometry rendered" "%TEMP%\g4gpu_view.txt" >nul
if errorlevel 1 (
  echo FATAL: the viewer selftest did not confirm that solid geometry renders.
  exit /b 1
)
findstr /C:"selftest:" "%TEMP%\g4gpu_view.txt" || exit /b 1

echo.
echo --- model builder selftest ---
rem The builder's selftest inserts solids, runs them, and writes a project. That project is
rem then compiled and run here, because "Save gives you a compilable project" is a claim only
rem worth something if something checks it - and generated code that does not compile is
rem exactly the breakage no other test in this pipeline would notice.
if exist "%~dp0out\selftest_project" rd /s /q "%~dp0out\selftest_project"
"%~dp0g4builder.exe" -selftest > "%TEMP%\g4gpu_builder.txt" 2>&1 || exit /b 1
findstr /C:"selftest:" "%TEMP%\g4gpu_builder.txt" || exit /b 1
rem "selftest:" alone is satisfied by a selftest that logged a failure and carried on, and the
rem subtraction step did exactly that: it picked the imported mesh as the boolean's operand,
rem which the boolean engine rightly refuses. So the two shapes that need the shared solid
rem store are named explicitly here, and a logged FAILED is fatal.
findstr /C:"added a subtraction and a polycone" "%TEMP%\g4gpu_builder.txt" >nul || (
  echo FATAL: the builder selftest did not insert the boolean and the polycone,
  echo        so nothing exercised the renderer's solid store.
  type "%TEMP%\g4gpu_builder.txt"
  exit /b 1
)
findstr /C:"a run with a same-layer overlap was refused" "%TEMP%\g4gpu_builder.txt" >nul || (
  echo FATAL: the builder selftest did not prove the same-layer overlap refusal.
  echo        Two volumes overlapping on one layer have no rule deciding which owns the
  echo        shared space, so a run must be refused rather than report a dose that depends
  echo        on the order the detector was built in.
  type "%TEMP%\g4gpu_builder.txt"
  exit /b 1
)
findstr /C:"selftest: FAILED" "%TEMP%\g4gpu_builder.txt" >nul
if not errorlevel 1 (
  echo FATAL: the builder selftest logged a failure.
  findstr /C:"selftest: FAILED" "%TEMP%\g4gpu_builder.txt"
  exit /b 1
)
rem The builder's own run has to report a dose that is not zero, for the same reason the
rem generated project's does - and for a sharper one. The selftest used to run 25 events,
rem which through a 30 mm cube is a fraction of one expected interaction: the dose was one
rem rare event or nothing at all, and it read 2.81 MeV or exactly 0 depending on where the
rem RNG stream landed. Anything that shifted that stream - more steps per track - looked
rem exactly like a transport bug, and cost most of a day. The run is now large enough for the
rem number to be stable, and this is the check that says so.
findstr /C:"dose 0 pGy" "%TEMP%\g4gpu_builder.txt" >nul
if not errorlevel 1 (
  echo FATAL: the builder selftest scored a dose of exactly zero.
  findstr /C:"dose" "%TEMP%\g4gpu_builder.txt"
  exit /b 1
)
rem Per-voxel scoring, which is checked inside the selftest by summing the cells and comparing
rem with the volume total the same run reported. This only asserts that the check ran at all:
rem a selftest that stopped exercising it would otherwise pass silently.
findstr /C:"selftest: per-voxel scoring:" "%TEMP%\g4gpu_builder.txt" >nul
if errorlevel 1 (
  echo FATAL: the builder selftest did not exercise per-voxel scoring.
  exit /b 1
)
rem The dialogs. Every other check on the builder reads a number, and a dialog whose text ran
rem past its frame, whose buttons fell off the bottom, or whose label overflowed its own box
rem would look correct to all of them - which is how a button caption came to straddle the edge
rem of the dialog it was in. The selftest holds each one up for a frame and photographs it;
rem this asserts the pictures exist, and a person can look at them when something reads oddly.
for %%D in (world voxel anchor physics vis) do (
  if not exist "%~dp0out\g4builder_dlg_%%D.png" (
    echo FATAL: the builder selftest did not capture the %%D dialog.
    exit /b 1
  )
)
echo the five dialogs were captured
rem The custom-scorer flag must not change the number a scorer reports until the generated
rem file is edited. It is not free: a filtered scorer's total is accumulated event by event on
rem the host through Accept(), while a stock one's comes from the device array. The selftest
rem runs the same seed both ways and requires them to agree exactly.
findstr /C:"selftest: a custom scorer reports what the stock one does" "%TEMP%\g4gpu_builder.txt" >nul
if errorlevel 1 (
  echo FATAL: the builder selftest did not check the custom-scorer equivalence.
  exit /b 1
)
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
"%~dp0out\selftest_project\MyDetector.exe" -n 10000 > "%TEMP%\g4gpu_gen.txt" 2>&1 || exit /b 1
findstr /C:"events" "%TEMP%\g4gpu_gen.txt" >nul || exit /b 1
rem The generated project has a dose scorer on the imported mesh, so it has to report a dose
rem that is not zero. Requiring only the word "dose" was satisfied by "dose 0 pGy", which is
rem what a mesh made of air deposits - and what the check reported as a pass.
findstr /C:"dose" "%TEMP%\g4gpu_gen.txt" || exit /b 1
findstr /C:"dose 0 pGy" "%TEMP%\g4gpu_gen.txt" >nul
if not errorlevel 1 (
  echo FATAL: the generated project scored a dose of exactly zero.
  type "%TEMP%\g4gpu_gen.txt"
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
rem 200000 events each, same model. The builder logs "selftest: compare <scorer> <MeV>"; the
rem project prints the same scorers in its end-of-run block. A one-beam-instead-of-two run
rem misses by tens of percent, far outside the few per cent two independent 200k-event Monte
rem Carlos can differ by.
"%~dp0out\selftest_project\MyDetector.exe" -n 200000 > "%TEMP%\g4gpu_cmp.txt" 2>&1 || exit /b 1
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\compare_project.ps1" ^
  -Builder "%TEMP%\g4gpu_builder.txt" -Project "%TEMP%\g4gpu_cmp.txt" || exit /b 1

echo generated project compiled and ran

echo --- proton depth-dose against Geant4 ---
rem The transport-level check for the charged-particle physics, and the only one in this
rem pipeline that compares a *curve* rather than a number.
rem
rem ref/proton/proton_depth.cc is compiled twice - once against the real Geant4 11.1.1 by
rem ref/proton/build.bat, once against this port's headers by build_proton.bat - so the
rem phantom, the beam, the binning and the physics list are one description, not two that
rem have to be kept in step. The Geant4 half is run once and checked in as
rem ref/oracle/proton_depth.csv, exactly like every other oracle file; this runs the port's
rem half short and compares per proton.
rem
rem 6000 events, which is a few seconds and puts the statistical error on R80 near 0.02 mm -
rem an order of magnitude under the limit, so a failure here means physics and not luck.
call "%~dp0build_proton.bat" || exit /b 1
"%~dp0proton_depth.exe" 6000 100 "%~dp0out\port_depth.csv" 0.7 0.5 || exit /b 1
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\compare_depth.ps1" ^
  -Reference "%~dp0ref\oracle\proton_depth.csv" -Port "%~dp0out\port_depth.csv" || exit /b 1


echo.
echo ALL OK
exit /b 0
