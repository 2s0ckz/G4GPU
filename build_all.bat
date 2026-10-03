@echo off
rem The entry point for the PHYSICS: builds the transport engine, the dose drivers and example
rem B1; builds and runs every physics, geometry, material and transport test; and checks the dose
rem against Geant4. Exits non-zero on the first thing that fails. Nothing lands on main without
rem it green.
rem
rem   build_all.bat          build, run tests, run the 2M-event dose check
rem   build_all.bat build    build only
rem   build_all.bat test     build, then tests only
rem
rem What is NOT here since 2026-10-02 is in build_extras.bat: the viewer, the model-builder GUI
rem and the project it generates, the builder's voxel scenes, the trajectory store, and the two
rem custom-hook PROJECTS. None of it is physics, and the two hook projects alone cost two hours
rem of ptxas (docs/RISK.md V210, V220). A change under src/render, src/builder, src/scenes,
rem src/host/g4view.cu, src/host/g4builder.cu, tools/gen_hook_units.ps1 or build_hook_engine.bat
rem needs build_extras.bat green as well.
rem
rem For iterating, not for deciding: tools\quick.ps1 builds and runs a named subset in seconds
rem rather than the hours this takes, most of which is rebuilding the transport engine - one
rem kernel per translation unit, docs/RISK.md V65, V189 - and compiling a hundred tests.
rem
rem   tools\quick.ps1 hadron          every test matching *hadron*
rem   tools\quick.ps1 -Check proton   the proton depth-dose comparison alone
rem   tools\quick.ps1 -List           what there is
rem
rem Nothing there replaces this file. The sigma gates and the batch-vs-macro, mesh, physics-switch
rem and pool comparisons live only here, and no test links the drivers - so a change under
rem src/host or src/g4 is not tested by anything quick.ps1 can run.
rem
rem It runs the whole pipeline because building alone proved not to be evidence of anything:
rem a driver that compiled and printed a dose had two physics processes silently switched off.
setlocal EnableDelayedExpansion
call "%~dp0setupenv.bat" || exit /b 1
cd /d "%~dp0"

rem A running executable cannot be relinked, and the error the linker gives is
rem "LNK1104: cannot open file 'g4dose.exe'" with no hint that a window is open. That cost a
rem pipeline run, so the check is here rather than in the reader's memory. These are the four
rem this file links; the viewer and the builder, the two that get left open, are checked by
rem build_extras.bat, which is where they are built now.
for %%E in (g4dose.exe b1_gpu_sched.exe exampleB1.exe proton_depth.exe) do (
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
set TESTS=test_core test_geometry test_navigation test_solids test_voxels test_mesh test_gamma_xs test_electron test_photoelectric test_brems test_rayleigh test_rayleigh_angular test_cuts test_general test_msc test_annihilation test_brems_rel test_hadron test_hadron_range test_hadron_delta test_fluctuation test_density_effect test_corrections test_muon test_hadron_radiative test_wentzel test_bragg test_ion_charge test_constants test_icru90 test_mott test_material_build test_all_materials test_nuclear_stopping test_wentzel_msc test_icru73qo test_nucleon_xs test_ion_fluctuation test_urban_general test_gun_position test_vs_oracle test_track_arena test_hadronic_process test_elastic_models test_decay test_hadronic_xs test_particlexs test_deex_nuclear test_deex_levels test_deex_probs test_deex_models test_deex_breakup test_species test_chargeodd test_coulomb_scattering test_precompound test_capture test_wiring test_isotopes test_ion_msc test_bic_nucleus test_bic_apply test_electron_hi test_ftf_params test_ftf_lund test_ftf_strings test_bic_imr test_ftf_model test_bertini_data test_bertini_collide test_bertini_cascade test_bertini_deex test_bertini_apply test_stopping test_ftf_entry test_emextra_config test_emextra_xs test_emextra_models test_rand_gauss_q test_inelastic_models test_bic_1h1 test_emextra_wiring test_bic_void

rem Tests that launch real kernels rather than calling __host__ __device__ code on the
rem host. They need the arch flag: StepTally reduces with atomicAdd on a double, which
rem does not exist before sm_60, and %NV% has no -arch so it defaults below that. This is
rem how that was found - the test would not compile until it was built like the engine.
set TESTS_GPU=test_step_hook test_neutron test_step_hadron test_capture_device test_ion_transport test_neutron_general test_lepton_transport test_inelastic_transport test_emextra_transport

echo --- drivers ---
rem /IMPLIB keeps the import library and its .exp out of the project root. They are a
rem link-time artefact of an exe that exports symbols, not something anyone runs.
%NVG% -o b1_gpu_sched.exe src\host\b1_gpu_sched.cu -Xlinker /IMPLIB:out/b1_gpu_sched.lib || exit /b 1
call "%~dp0build_dose.bat" || exit /b 1

echo --- example B1 (Geant4-shaped API) ---
call "%~dp0examples\B1\build.bat" || exit /b 1

echo --- tests ---
rem Incremental and parallel since 2026-10-02 (docs/RISK.md V221): tools\build_tests.ps1 compiles
rem only the tests whose own include set changed - nvcc's -MD dependency file per test, under
rem out\deps, read by tools\freshness.ps1 - six host-only tests at a time and the kernel-launching
rem ones one at a time, and prints each compile's output as it finishes.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\build_tests.ps1" -Root "%~dp0." ^
  -Plain "%TESTS%" -Gpu "%TESTS_GPU%" -Arch "%G4GPU_ARCH%" -Mode build -Jobs 6 || exit /b 1
rem From here on they are just tests - run and counted with the rest; the host-only list is kept
rem apart because the run schedules the two kinds differently.
set TESTS_HOST=%TESTS%
set TESTS_RUN_GPU=%TESTS_GPU%
set TESTS=%TESTS% %TESTS_GPU%
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
echo --- dose vs Geant4, 2M events ---
"%~dp0b1_gpu_sched.exe" 2000000 > "%TEMP%\g4gpu_%RUNID%_dose.txt" 2>&1 || exit /b 1
findstr /C:"FATAL" "%TEMP%\g4gpu_%RUNID%_dose.txt" >nul 2>&1
if not errorlevel 1 (
  type "%TEMP%\g4gpu_%RUNID%_dose.txt"
  exit /b 1
)
findstr /C:"photoelectric:" /C:"rayleigh:" /C:"bremsstrahlung:" /C:"scaled to 10k" /C:"Geant4 11.1.1" /C:"uncertainty" "%TEMP%\g4gpu_%RUNID%_dose.txt"
echo.
echo --- example B1, 2M events ---
"%~dp0examples\B1\exampleB1.exe" -n 2000000 > "%TEMP%\g4gpu_%RUNID%_b1.txt" 2>&1 || exit /b 1
findstr /C:"FATAL" "%TEMP%\g4gpu_%RUNID%_b1.txt" >nul 2>&1
if not errorlevel 1 (
  type "%TEMP%\g4gpu_%RUNID%_b1.txt"
  exit /b 1
)
findstr /C:"scaled to 10k" /C:"Geant4 11.1.1" /C:"difference" "%TEMP%\g4gpu_%RUNID%_b1.txt"
rem And *checked*, not merely printed. The agreement with Geant4 is what this project claims;
rem printing it into a log for a person to read is not a test of it, and a drift to five sigma
rem would have passed here and been reported as ALL OK.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\check_sigma.ps1" ^
  -Log "%TEMP%\g4gpu_%RUNID%_b1.txt" || exit /b 1

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
"%~dp0examples\B1\exampleB1.exe" gunstate.mac > "%TEMP%\g4gpu_%RUNID%_b1mac.txt" 2>&1 || (popd & exit /b 1)
popd
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\compare_runs.ps1" ^
  -BatchLog "%TEMP%\g4gpu_%RUNID%_b1.txt" -MacroLog "%TEMP%\g4gpu_%RUNID%_b1mac.txt" || exit /b 1

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
echo --- the step hook sees every step ---
rem The general step-level customisation point (core/step_hook.cuh). Its whole claim is that
rem a hook is handed every real step exactly once, charged to the right event - so that is
rem what this asserts, against the scorer, which counts the same energy by a different route.
rem A hook that quietly missed a class of steps would still produce plausible-looking output,
rem which is why this is checked rather than assumed.
"%~dp0g4dose.exe" -verify-step-hook -n 20000 || exit /b 1

echo.
echo --- the pool size does not change the answer, odd or even ---
rem The throttle defers a track it has no room for rather than dropping it, so the dose must
rem not depend on how many live slots a run was given. Three pools, and 2.5 is there for a
rem second reason: it makes the pool ODD, which every other configuration in this pipeline
rem could not. A track slot is 236 bytes, so an odd pool put the second half of the ping-pong
rem arena at 4 mod 8 and every double in it was misaligned - a device fault that the default
rem -live 4 could never reach. tests\test_track_arena.exe checks the arithmetic; this checks
rem that a real run gives the same number three ways.
for %%L in (8.0 4.0 2.5) do (
  "%~dp0g4dose.exe" -n 200000 -live %%L > "%TEMP%\g4gpu_%RUNID%_pool_%%L.txt" 2>&1 || (
    echo FATAL: g4dose failed at -live %%L
    type "%TEMP%\g4gpu_%RUNID%_pool_%%L.txt"
    exit /b 1
  )
)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\compare_pool.ps1" ^
  -Prefix "%TEMP%\g4gpu_%RUNID%_pool_" -Labels "8.0,4.0,2.5" || exit /b 1

echo.
echo --- a second run in one process is a second sample, and a new process replays ---
rem Geant4's behaviour, and two claims rather than one: B1.exe run twice gives the same
rem answer, while two /run/beamOn in one session give different answers within noise. A port
rem with only the first cannot offer a second sample; a port with only the second cannot be
rem checked against a reference. Three runs of the same program, because the sharp form of the
rem claim is that two runs of 20000 sum to one run of 40000 - an offset that jumped too far
rem would still differ and still replay, and would silently skip part of the stream.
pushd "%~dp0examples\B1"
"%~dp0examples\B1\exampleB1.exe" two_runs.mac > "%TEMP%\g4gpu_%RUNID%_seq_a.txt" 2>&1 || (
  echo FATAL: two_runs.mac failed.
  type "%TEMP%\g4gpu_%RUNID%_seq_a.txt"
  popd
  exit /b 1
)
"%~dp0examples\B1\exampleB1.exe" two_runs.mac > "%TEMP%\g4gpu_%RUNID%_seq_b.txt" 2>&1 || (
  echo FATAL: two_runs.mac failed on its second process.
  popd
  exit /b 1
)
"%~dp0examples\B1\exampleB1.exe" two_runs_one.mac > "%TEMP%\g4gpu_%RUNID%_seq_one.txt" 2>&1 || (
  echo FATAL: two_runs_one.mac failed.
  popd
  exit /b 1
)
popd
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\check_run_sequence.ps1" ^
  -TwoRunLog "%TEMP%\g4gpu_%RUNID%_seq_a.txt" -RepeatLog "%TEMP%\g4gpu_%RUNID%_seq_b.txt" ^
  -SingleLog "%TEMP%\g4gpu_%RUNID%_seq_one.txt" || exit /b 1
echo.
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
rem WHAT THE REFERENCE IS, and it changed in P8c. It was `G4EmStandardPhysics` and nothing
rem else, which is a Geant4 with no hadronic process at all - and the port has had `G4Decay`
rem since P8 and `hadElastic` and `CoulombScat` since P8b, so this gate was comparing two
rem different experiments and failed by the 1.32% that elastic scattering is worth to a
rem 100 MeV proton's plateau in water. It is QBBC on both sides now with ONLY what the port
rem lacks inactivated on the Geant4 side - every `*Inelastic`, `hBrems`/`hPairProd`,
rem `ionElastic`, `NeutronGeneralProc`, the electro-/muon-nuclear processes and the three
rem at-rest captures - and the reference run prints `/particle/process/dump` for the proton
rem and for GenericIon so the configuration is recorded by what ran rather than by what was
rem intended. docs/RISK.md V58, docs/PORTED.md 2.1.7.
rem
rem AND IT CHANGED AGAIN IN P15, in the other direction: `protonInelastic`, `NeutronGeneralProc`
rem and the THREE AT-REST CAPTURES came OFF that list, because the port does all of them now.
rem What is still inactivated is `dInelastic`, `tInelastic`, `He3Inelastic`, `alphaInelastic` and
rem `ionInelastic` - `G4BinaryLightIonReaction::Interact`, every ion at or above 50 MeV per
rem nucleon, is P9e's and is refused by name - plus `hBrems`/`hPairProd`, `ionElastic` and the
rem lepto-nuclear processes. Both references were regenerated.
rem
rem AND THE PORT SIDE HOLDS THE SAME TWO SWITCHES, which is the half that was missing and is
rem what docs/RISK.md V192 and V194 are. `proton_depth.cc` calls `SetIonInelastic(false)` so the
rem five ion names are off on BOTH sides rather than on Geant4's alone - the port refuses nine
rem in ten of an ion's interactions and disposes of each by killing the ion where the refusal
rem happened, which moves the deposit upstream of its range - and `SetHadronicStage(kFinal)` so
rem the neutron actually HAS an inelastic sub-process and a stopped negative hadron is actually
rem captured. The engine default is `kStage1`, in which neither is true, and nothing in this
rem repository selected anything else before P15.
rem
rem THE FOUR NUMBERS MOVED AND THE CONSERVATION CHECK CHANGED SHAPE. With inelastic processes
rem active on both sides a 100 MeV proton makes NEUTRONS, and a neutron leaves a 150 mm
rem phantom, so neither side contains the beam energy any more: Geant4 97.5749%, the port
rem 97.3203%. `compare_depth.ps1` compares the two CONTAINED fractions against each other now
rem rather than each against one - that difference is a statement about the escaping-neutron
rem budget and is one of the things an inelastic wiring can get wrong - and the old
rem one-part-in-a-million total check survives only as a floor. Measured, 100,000 reference
rem events against 6,000 port events:
rem
rem                   before P15            P15 first pass           P15 second pass
rem                   (no inelastic)        (ions off both sides,     (P9e's Interact: ions ON both
rem                                         port in stage 1)          sides; port in the final stage)
rem     contained     100% both sides       G4 97.5749%               G4 97.6119%
rem                                         port 97.3203%, -0.261%    port 97.4244%, -0.192%
rem     plateau       +0.158%               +0.048%                   +0.096%
rem     R80           77.730 / 77.742       77.714 / 77.656           77.720 / 77.657, -0.063 mm
rem     80-20 width   1.152 / 1.166         1.166 / 1.208             1.169 / 1.205, +0.036 mm
rem
rem All four are inside their limits in every column. The second pass is the like-for-like one:
rem the five ion inelastic processes are active on BOTH sides (the first pass had them off on
rem both, because the port refused Interact), and the port runs the final stage, whose neutron
rem reacts inelastically and whose stopped negatives are captured - the first pass had the port
rem in stage 1 while Geant4 ran both (docs/RISK.md V192, V194, V198). docs/PORTED.md 2.1.15.
rem
rem The phantom is also 150 mm wide rather than 50, because the conservation check's premise
rem ("the phantom is deeper than the range") is about depth and an elastic scatter sends a
rem proton sideways with nearly all of its energy: 1.1e-5 of the beam left a 50 mm box, on
rem the GEANT4 side, against a 1e-6 limit. Widened rather than excused - V58 has the number.
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
