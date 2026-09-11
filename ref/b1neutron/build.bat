@echo off
rem Builds the stage-1 neutron reference: Geant4's example B1 with the general process off.
rem
rem RELATIVE TO THIS FILE, which is docs/RISK.md V45 and V59's rule: a package developed in a
rem git worktree must build ITS OWN reference from ITS OWN source, not D:\g4gpu's. The build
rem tree is ref/b1neutronbuild beside this directory and is gitignored like the other three.
rem
rem Visual Studio 16 2019 to match ref/dumpbuild and ref/protonbuild - the Geant4 install was
rem built with it, and mixing runtimes across a library boundary is not worth discovering at
rem link time.
call "%~dp0..\..\setupenv.bat" || exit /b 1
if not exist "%~dp0..\b1neutronbuild" mkdir "%~dp0..\b1neutronbuild"
pushd "%~dp0..\b1neutronbuild"
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
