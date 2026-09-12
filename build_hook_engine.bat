@echo off
rem Compiles the transport kernels for a PROJECT'S OWN step hook, one kernel per translation
rem unit, into out\hook_<tag>.lib - the same arrangement build_engine.bat gives the stock
rem engine, for the same reason.
rem
rem   build_hook_engine.bat <hook header> <hook type> <tag>
rem
rem     <hook header>  the project's header that defines the hook class
rem     <hook type>    the class name, exactly as the project writes it
rem     <tag>          a short name; out\hook_<tag>\ holds the units, out\hook_<tag>.lib the
rem                    archive, and out\hook_<tag>\hook_kernels.cuh the declarations the
rem                    project must include
rem
rem On success it sets, for the caller:
rem     G4GPU_HOOK_LIB  the archive to link
rem     G4GPU_HOOK_INC  the directory to put on the project's -I line
rem
rem WHY A SECOND SCRIPT AND NOT build_engine.bat. The stock engine has one hook type and a
rem fixed list of units on disk; a project has a hook type nobody here knows the name of, so its
rem units have to be generated. Everything downstream of that - one kernel per unit, six at a
rem time, a .rc per unit, one archive - is build_engine.bat's arrangement and this script reuses
rem its per-unit compiler, build_engine_unit.bat, rather than restating the nvcc line.
rem
rem WHAT IT IS FOR. docs/RISK.md V65 split the ENGINE'S translation unit and stopped there,
rem because the engine was the file that would not compile. It was not the only one.
rem tests\test_custom_hook.cu instantiates TransportEngine<double, QualityFactorScoring> and the
rem eighteen launches inside BeamOn instantiate eighteen kernels into THAT file - a different
rem specialisation of the same templates, so the split bought it nothing, and with both of V66's
rem switches on it died exactly as the engine had:
rem
rem     ptxas warning : Stack size for entry function
rem                     'run_step_hadron<double, ParticleType(13), QualityFactorScoring>' ...
rem     Internal error
rem     nvcc error   : 'ptxas' died with status 0xC0000005 (ACCESS_VIOLATION)
rem
rem The build_all.bat comment used to call that unit "the honest cost of the arrangement". It
rem was honestly measured and it was never necessary: the hook is a template parameter like any
rem other, so a project's kernels split exactly as the engine's do.
call "%~dp0setupenv.bat" || exit /b 1
if "%~3"=="" (
  echo usage: build_hook_engine.bat ^<hook header^> ^<hook type^> ^<tag^>
  exit /b 1
)
pushd "%~dp0"
if not exist out mkdir out

set HOOK_HDR=%~f1
set HOOK_TYPE=%~2
set HOOK_TAG=%~3
set HOOK_DIR=%~dp0out\hook_%HOOK_TAG%
set G4GPU_HOOK_LIB=%~dp0out\hook_%HOOK_TAG%.lib
set G4GPU_HOOK_INC=%HOOK_DIR%

if not exist "%HOOK_HDR%" (
  echo FATAL: no hook header at %HOOK_HDR%
  popd
  exit /b 1
)

rem The freshness question is asked about the ARCHIVE against the hook header and every header
rem under src\, which is what tools\freshness.ps1 compares - one call rather than the stock
rem engine's seventeen, because these units are generated from that one header and are
rem identical to each other in everything but the kernel they name. An "if defined, skip" guard
rem here would be docs/RISK.md S4 again: right within one pipeline run, and wrong the moment
rem anyone sets the variable by hand, at which point the project links kernels compiled before
rem the change under test.
for /f "usebackq delims=" %%R in (`powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\freshness.ps1" -Obj "%G4GPU_HOOK_LIB%" -Unit "%HOOK_HDR%" -SrcDir "%~dp0src"`) do set FRESH=%%R
if /i "%FRESH%"=="fresh" (
  popd
  echo hook_%HOOK_TAG%.lib is up to date
  exit /b 0
)

rem Regenerated from scratch every time, and the directory is emptied first. A unit left behind
rem from a previous hook - or from a species that has since been removed - would be compiled by
rem the glob below and archived into this lib, where it would either duplicate a specialisation
rem or drag a deleted kernel back into the link.
if exist "%HOOK_DIR%" rd /s /q "%HOOK_DIR%"
mkdir "%HOOK_DIR%"
rem Not `for /f` around the generator, and that is not style. Its failure messages quote the
rem line it could not parse, which contains `StepTap<double>` - and a captured `<` fed back
rem through `echo` is a redirection, so the diagnostic would take the error path out with it.
rem Redirect to a file, test the exit code, print the file.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\gen_hook_units.ps1" ^
  -ImplHeader "%~dp0src\host\transport_run_impl.cuh" -HookHeader "%HOOK_HDR%" ^
  -HookType "%HOOK_TYPE%" -OutDir "%HOOK_DIR%" > "%HOOK_DIR%\gen.log" 2>&1
if errorlevel 1 (
  echo FATAL: the hook unit generator failed. Its output follows.
  type "%HOOK_DIR%\gen.log"
  popd
  exit /b 1
)
for /f "usebackq delims=" %%N in ("%HOOK_DIR%\gen.log") do set NUNITS=%%N
echo compiling %NUNITS% kernels for %HOOK_TYPE%, 6 at a time

set CAP=6
call :launch_units

rem The same bounded wait build_engine.bat uses, and bounded for the same reason: a unit that
rem never starts leaves a sentinel that never appears, and an unbounded loop then prints
rem nothing for ever. ping rather than `timeout /t`, which refuses to run under a redirected
rem stdin - and every pipeline that calls this redirects.
set WAITED=0
:wait
set PENDING=
for %%U in ("%HOOK_DIR%\*.cu") do (
  if not exist "%HOOK_DIR%\%%~nU.rc" set PENDING=1
)
if defined PENDING (
  set /a WAITED+=1
  if %WAITED% GEQ 1200 (
    echo FATAL: the hook units did not all finish within 40 minutes. Missing:
    for %%U in ("%HOOK_DIR%\*.cu") do (
      if not exist "%HOOK_DIR%\%%~nU.rc" echo        %%~nU
    )
    popd
    exit /b 1
  )
  ping -n 3 127.0.0.1 >nul
  goto wait
)

set BAD=
for %%U in ("%HOOK_DIR%\*.cu") do (
  for /f "usebackq delims=" %%R in ("%HOOK_DIR%\%%~nU.rc") do if not "%%R"=="0" set BAD=1
)
if defined BAD (
  echo FATAL: a hook kernel unit did not compile. Its output follows.
  for %%U in ("%HOOK_DIR%\*.cu") do (
    for /f "usebackq delims=" %%R in ("%HOOK_DIR%\%%~nU.rc") do if not "%%R"=="0" (
      echo ---- %%~nU ----
      type "%HOOK_DIR%\%%~nU.log"
    )
  )
  popd
  exit /b 1
)

rem A response file, because lib.exe does not expand wildcards and the unit list belongs to the
rem glob rather than to this script. Written from scratch, so a unit deleted between builds
rem cannot be carried into a later archive.
> "%HOOK_DIR%\lib.rsp" echo /NOLOGO
>> "%HOOK_DIR%\lib.rsp" echo /OUT:"%G4GPU_HOOK_LIB%"
for %%U in ("%HOOK_DIR%\*.cu") do (
  >> "%HOOK_DIR%\lib.rsp" echo "%HOOK_DIR%\%%~nU.obj"
)
lib @"%HOOK_DIR%\lib.rsp"
if errorlevel 1 (
  popd
  exit /b 1
)
popd
echo built hook_%HOOK_TAG%.lib
exit /b 0

rem ---------------------------------------------------------------- the throttled launcher
rem
rem A subroutine for the reason build_engine.bat's is: it needs delayed expansion for its
rem counter, and `setlocal EnableDelayedExpansion` at the top of the script would discard
rem G4GPU_HOOK_LIB and G4GPU_HOOK_INC on the way out, which are the two variables the caller
rem came for. `cmd /c call "..."` and not `start /b "the.bat" ...` because when the string after
rem /c begins with a quote cmd strips the first and last quote of the whole line; starting it
rem with `call` means it does not begin with one.
:launch_units
setlocal EnableDelayedExpansion
set LAUNCHED=0
for %%U in ("%HOOK_DIR%\*.cu") do (
  call :await_slot
  start "" /b cmd /c call "%~dp0build_engine_unit.bat" "%%~fU" "%HOOK_DIR%"
  set /a LAUNCHED+=1
)
endlocal
exit /b 0

:await_slot
set DONE=0
for %%R in ("%HOOK_DIR%\*.rc") do set /a DONE+=1
set /a INFLIGHT=!LAUNCHED!-!DONE!
if !INFLIGHT! LSS %CAP% exit /b 0
ping -n 3 127.0.0.1 >nul
goto await_slot
