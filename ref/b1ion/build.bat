@echo off
rem Builds ref/b1ion: example B1 on QBBC with ionElastic counted (docs/RISK.md V267). Relative to
rem this file, like ref/b1ngp/build.bat, so a worktree builds its own; the build tree is
rem ref/b1ionbuild beside this directory and is gitignored.
call "%~dp0..\..\setupenv.bat" || exit /b 1
if not exist "%~dp0..\b1ionbuild" mkdir "%~dp0..\b1ionbuild"
pushd "%~dp0..\b1ionbuild"
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
