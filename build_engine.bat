@echo off
rem Compiles the transport engine to an object file, when it needs it.
rem
rem transport_run.cu owns the __global__ template kernels and takes several minutes to
rem compile. It is linked into the dose driver, the viewer, the GUI and every example, so
rem compiling it per target quadrupled the pipeline's build time for no reason.
rem
rem The skip is decided by comparing timestamps - tools/freshness.ps1 - and not by an
rem environment variable. An "if defined, skip" guard is a cache with no invalidation: it is
rem right within one pipeline run and wrong the moment anyone sets the variable by hand, at
rem which point the build links an engine compiled before the change under test and the run
rem prints a plausible number for physics that was never compiled. See docs/RISK.md S4.
call "%~dp0setupenv.bat" || exit /b 1
pushd "%~dp0"
if not exist out mkdir out
set G4GPU_ENGINE_OBJ=%~dp0out\transport_run.obj
set FRESH=stale
for /f "usebackq delims=" %%R in (`powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\freshness.ps1" -Obj "%~dp0out\transport_run.obj" -Unit "%~dp0src\host\transport_run.cu" -SrcDir "%~dp0src"`) do set FRESH=%%R
if /i "%FRESH%"=="fresh" (
  popd
  echo transport_run.obj is up to date
  exit /b 0
)
nvcc -std=c++17 -O2 -arch=sm_86 -I "%~dp0src" -I "%~dp0src\g4" -c ^
  -o out\transport_run.obj src\host\transport_run.cu
if errorlevel 1 (
  popd
  exit /b 1
)
popd
echo built transport_run.obj
exit /b 0
