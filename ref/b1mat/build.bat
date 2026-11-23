@echo off
rem Builds ref/b1mat: example B1 with /B1/det/material (P27, docs/RISK.md V336). Relative to
rem this file, like ref/b1ngp/build.bat, so a worktree builds its own; the build tree is
rem ref/b1matbuild beside this directory and is gitignored.
call "%~dp0..\..\setupenv.bat" || exit /b 1
if not exist "%~dp0..\b1matbuild" mkdir "%~dp0..\b1matbuild"
pushd "%~dp0..\b1matbuild"
if not exist CMakeCache.txt (
  cmake -G "Visual Studio 16 2019" -A x64 -DCMAKE_BUILD_TYPE=Release ^
    -DGeant4_DIR=D:/Documents/Geant4/Windows/geant4-v11.1.1-install/lib/cmake/Geant4 ^
    "%~dp0."
  if errorlevel 1 (
    popd
    exit /b 1
  )
)
cmake --build . --config Release
set RC=%errorlevel%
popd
exit /b %RC%
