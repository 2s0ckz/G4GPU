@echo off
rem Builds ref/b1ngp: example B1 on QBBC with the neutron general process's step charging as a
rem switch (docs/RISK.md V219). Relative to this file, like ref/b1neutron/build.bat, so a worktree
rem builds its own; the build tree is ref/b1ngpbuild beside this directory and is gitignored.
call "%~dp0..\..\setupenv.bat" || exit /b 1
if not exist "%~dp0..\b1ngpbuild" mkdir "%~dp0..\b1ngpbuild"
pushd "%~dp0..\b1ngpbuild"
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
