@echo off
rem Builds the port's half of the proton depth-dose comparison.
rem
rem The source is ref/proton/proton_depth.cc, and the *same* file is built against the real
rem Geant4 by ref/proton/build.bat. That is deliberate: one description of the geometry, the
rem beam and the binning, compiled twice, so the two sides cannot be set up differently.
rem
rem -x cu because the port's Geant4-shaped headers reach down into device code, exactly as
rem examples/B1/build.bat does for the same reason. -DG4GPU_PORT is the only thing the source
rem branches on, and it selects the batch size - a knob the real Geant4 does not have.
call "%~dp0setupenv.bat" || exit /b 1
call "%~dp0build_engine.bat" || exit /b 1
pushd "%~dp0"
if not exist out mkdir out
nvcc -std=c++17 -O2 -arch=sm_86 -I "%~dp0src" -I "%~dp0src\g4" -DG4GPU_PORT ^
  -x cu -c -o out\proton_depth.obj ref\proton\proton_depth.cc
if errorlevel 1 (
  popd
  exit /b 1
)
nvcc -std=c++17 -O2 -arch=sm_86 -o proton_depth.exe ^
  out\proton_depth.obj "%G4GPU_ENGINE_OBJ%" -Xlinker /IMPLIB:out/proton_depth.lib
if errorlevel 1 (
  popd
  exit /b 1
)
popd
echo built proton_depth.exe
exit /b 0
