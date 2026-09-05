@echo off
rem Builds g4dose: a scene run headless, with a seed and a set of process switches.
call "D:/g4gpu/setupenv.bat" || exit /b 1
call "D:/g4gpu/build_engine.bat" || exit /b 1
pushd "%~dp0"
nvcc -std=c++17 -O2 -arch=sm_86 -I "%~dp0src" -I "%~dp0src\g4" -o g4dose.exe ^
  src\host\g4dose.cu src\scenes\scene_b1.cu src\scenes\scene_b1mesh.cu ^
  "%G4GPU_ENGINE_OBJ%" -Xlinker /IMPLIB:out/g4dose.lib
if errorlevel 1 (
  popd
  exit /b 1
)
popd
echo built g4dose.exe
