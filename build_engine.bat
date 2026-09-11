@echo off
rem Compiles the transport engine to out\transport_run.lib, when it needs it.
rem
rem SEVENTEEN TRANSLATION UNITS, SIX AT A TIME, ARCHIVED INTO ONE LIBRARY.
rem
rem It was one unit until P8e - src\host\transport_run.cu instantiated the engine class, and
rem the launches inside BeamOn implicitly instantiated all eighteen stepping kernels for every
rem species. That took 25 minutes of nvcc, and with the ion's Urban multiple scattering
rem dispatched in step_hadron it stopped compiling at all: `ptxas died with status 0xC0000005`,
rem deterministically, in nine different arrangements of the same code (docs/RISK.md V55, V63).
rem There is now one unit per stepping kernel - held there by the explicit instantiation
rem declarations under the kernels in src\host\transport_run_impl.cuh - and transport_run.cu
rem holds the engine's host code and the three utility kernels that carry no physics.
rem docs/RISK.md V65 has the timings and the reason the granularity is per KERNEL and not per
rem kernel family: four charged-meson kernels in a unit of their own kill ptxas on their own,
rem alone on the machine, where the same four inside the eighteen-kernel unit compiled fine.
rem
rem The units are found by globbing rather than listed here, so adding or splitting one needs
rem no change to this file. What keeps the glob honest is the LINK: the engine's launches
rem reference a stub per specialisation, so a species with no instantiation anywhere is an
rem unresolved symbol rather than a track that is never stepped.
rem
rem THE OUTPUT IS A .lib AND THAT IS DELIBERATE. Every consumer links "%G4GPU_ENGINE_OBJ%" as
rem one quoted path - build_dose, build_view, build_gui, build_proton, examples\B1\build.bat
rem and every project the model builder generates - so making the variable a LIST of eight
rem objects would have meant editing all of them, including the generated build script in
rem src\builder\write_project.cc. One archive keeps that interface exactly as it was.
rem
rem The skip is decided by comparing timestamps - tools/freshness.ps1 - and not by an
rem environment variable. An "if defined, skip" guard is a cache with no invalidation: it is
rem right within one pipeline run and wrong the moment anyone sets the variable by hand, at
rem which point the build links an engine compiled before the change under test and the run
rem prints a plausible number for physics that was never compiled. See docs/RISK.md S4.
rem
rem freshness.ps1 is asked once PER UNIT, which costs seventeen PowerShell starts - about six
rem seconds - on every call. It takes one -Unit and its header scan covers .cuh/.hh/.h/.inc and
rem not .cu, so asking it once about transport_run.cu would miss an edit to any of the other
rem sixteen units entirely. That is S4 again, for six seconds saved.
call "%~dp0setupenv.bat" || exit /b 1
pushd "%~dp0"
if not exist out mkdir out
set G4GPU_ENGINE_OBJ=%~dp0out\transport_run.lib

set STALE=
for %%U in ("%~dp0src\host\transport_run*.cu") do (
  for /f "usebackq delims=" %%R in (`powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\freshness.ps1" -Obj "%~dp0out\transport_run.lib" -Unit "%%~fU" -SrcDir "%~dp0src"`) do if /i "%%R"=="stale" set STALE=1
)
if not defined STALE (
  popd
  echo transport_run.lib is up to date
  exit /b 0
)

rem SIX AT A TIME, and the cap is a measurement. A single-kernel unit's ptxas peaks at 2.1 to
rem 2.8 GB of working set, so six is about 18 GB of the 64 this machine has - and it leaves
rem cores and memory for whatever else is building, which on this project is usually three or
rem four other worktrees. Unbounded is not free: the first version of this script started all
rem of them and the machine held eight ptxas processes at once, one of which wanted 8.7 GB.
set CAP=6
echo compiling the transport engine, %CAP% units at a time
for %%U in ("%~dp0src\host\transport_run*.cu") do (
  if exist "%~dp0out\%%~nU.rc" del /q "%~dp0out\%%~nU.rc"
)
call :launch_units

rem Waiting on the sentinel files, because cmd cannot wait on a process it started with
rem `start`. ping rather than `timeout /t`: timeout refuses to run when stdin is redirected
rem ("ERROR: Input redirection is not supported"), and every pipeline that calls this script
rem redirects, which would turn this into a hot loop.
rem
rem AND IT IS BOUNDED, because the unbounded version is what a mis-started unit turns into: a
rem script that prints "compiling" and then waits for a sentinel that will never be written,
rem for ever, with no output. 1200 turns of about two seconds is 40 minutes, against a whole
rem build that takes about five; a build that has not finished by then has something wrong with
rem it that waiting longer will not fix.
set WAITED=0
:wait
set PENDING=
for %%U in ("%~dp0src\host\transport_run*.cu") do (
  if not exist "%~dp0out\%%~nU.rc" set PENDING=1
)
if defined PENDING (
  set /a WAITED+=1
  if %WAITED% GEQ 1200 (
    echo FATAL: the engine units did not all finish within 40 minutes. Missing:
    for %%U in ("%~dp0src\host\transport_run*.cu") do (
      if not exist "%~dp0out\%%~nU.rc" echo        %%~nU
    )
    popd
    exit /b 1
  )
  ping -n 3 127.0.0.1 >nul
  goto wait
)

set BAD=
for %%U in ("%~dp0src\host\transport_run*.cu") do (
  for /f "usebackq delims=" %%R in ("%~dp0out\%%~nU.rc") do if not "%%R"=="0" set BAD=1
)
if defined BAD (
  echo FATAL: a transport engine unit did not compile. Its output follows.
  for %%U in ("%~dp0src\host\transport_run*.cu") do (
    for /f "usebackq delims=" %%R in ("%~dp0out\%%~nU.rc") do if not "%%R"=="0" (
      echo ---- %%~nU ----
      type "%~dp0out\%%~nU.log"
    )
  )
  popd
  exit /b 1
)

rem THE ENGINE UNIT MUST CARRY NO STEPPING KERNEL, AND THIS IS THE CHECK.
rem
rem The whole split rests on the explicit instantiation declarations in transport_run_impl.cuh
rem suppressing implicit instantiation in the unit that launches the kernels. Measured, and
rem inverted: with the declarations, cuobjdump reports no device function of that name in this
rem object and the program still runs; with them deleted it carries the kernels again. What it
rem does NOT do is fail - two objects holding the same specialisation link cleanly on CUDA
rem 11.6, the stubs being COMDAT-folded - so a launch added without a declaration would cost
rem 24 minutes of nvcc and say nothing at all. Hence a check on the artefact, not on the link.
for /f "usebackq delims=" %%N in (`cuobjdump -res-usage "%~dp0out\transport_run.obj" ^| findstr /c:"run_step_"`) do (
  echo FATAL: out\transport_run.obj carries a stepping kernel:
  echo        %%N
  echo        A launch was added without a matching `extern template` in
  echo        src\host\transport_run_impl.cuh, so that kernel is compiled into the engine's
  echo        own translation unit again. See docs/RISK.md V65.
  popd
  exit /b 1
)

rem A response file, because lib.exe does not expand wildcards and the unit list is the glob's
rem to know rather than this script's. Rewritten from scratch each time: an appended-to list
rem would carry a unit that had been deleted into every later archive.
> "%~dp0out\transport_run_lib.rsp" echo /NOLOGO
>> "%~dp0out\transport_run_lib.rsp" echo /OUT:"%~dp0out\transport_run.lib"
for %%U in ("%~dp0src\host\transport_run*.cu") do (
  >> "%~dp0out\transport_run_lib.rsp" echo "%~dp0out\%%~nU.obj"
)
lib @"%~dp0out\transport_run_lib.rsp"
if errorlevel 1 (
  popd
  exit /b 1
)
popd
echo built transport_run.lib
exit /b 0

rem ---------------------------------------------------------------- the throttled launcher
rem
rem A subroutine, and it is a subroutine for one reason: it needs delayed expansion for its
rem counter, and `setlocal EnableDelayedExpansion` at the top of the script would discard
rem G4GPU_ENGINE_OBJ on the way out - which is the variable every caller links. A `call`d label
rem gets its own setlocal and the main scope keeps its variables.
rem
rem `cmd /c call "..."` and not `start /b "the.bat" "arg" "arg"`, and that is a rule of cmd's
rem own parser rather than a preference. When the string after `/c` BEGINS with a quote, cmd
rem strips the first and the last quote of the whole line - so `cmd /c "a.bat" "b" "c"` becomes
rem `a.bat" "b" "c`, one nonsense path, and the child reports "The filename, directory name, or
rem volume label syntax is incorrect" from inside a process this script never sees. Starting
rem the line with `call` means it does not begin with a quote and nothing is stripped. Every
rem unit failed that way at once and the wait loop above then waited for ever, which is how
rem this was found; `start`'s own quoted-argument handling has the same fault without `cmd /c`.
:launch_units
setlocal EnableDelayedExpansion
set LAUNCHED=0
for %%U in ("%~dp0src\host\transport_run*.cu") do (
  call :await_slot
  start "" /b cmd /c call "%~dp0build_engine_unit.bat" "%%~fU" "%~dp0out"
  set /a LAUNCHED+=1
)
endlocal
exit /b 0

rem Blocks until fewer than CAP units are in flight. A unit is in flight from the moment it is
rem started until its .rc appears, so counting the .rc files is counting the ones that are done
rem - and the launcher deleted every stale .rc before starting anything, which is what makes
rem that count mean this build rather than the last one.
:await_slot
set DONE=0
for %%R in ("%~dp0out\transport_run*.rc") do set /a DONE+=1
set /a INFLIGHT=!LAUNCHED!-!DONE!
if !INFLIGHT! LSS %CAP% exit /b 0
ping -n 3 127.0.0.1 >nul
goto await_slot
