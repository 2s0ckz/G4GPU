@echo off
rem Builds the real-Geant4 half of the proton depth-dose comparison.
rem
rem Same source file as build_proton.bat at the repository root, without -DG4GPU_PORT. That is
rem the point of the exercise: the geometry, the beam, the binning and the physics list are one
rem description compiled twice, so the two sides cannot be set up differently.
rem
rem RELATIVE TO THIS FILE, AND THAT IS V45's LAST INSTANCE. Every path here was `D:/g4gpu/...`,
rem so this script run from a git worktree configured cmake on MAIN's ref/proton, built MAIN's
rem proton_depth.cc, and wrote MAIN's ref/protonbuild - and then ref/oracle/run.bat ran that
rem binary and overwrote MAIN's ref/oracle/proton_depth.csv with it. A package changing the
rem reference's physics list would have regenerated the reference from a source file it had not
rem changed, and the disagreement would have been blamed on the port. docs/RISK.md V45 fixed
rem build_dose.bat, build_gui.bat, build_proton.bat, build_view.bat and examples/B1/build.bat
rem the same way; these three files (this one, run.bat beside it, and the last two lines of
rem ref/oracle/run.bat) were not in that commit.
rem
rem Configures on first use. Visual Studio 16 2019 to match ref/dumpbuild - the Geant4 install
rem was built with it, and mixing runtimes across a library boundary is not worth discovering
rem at link time.
call "%~dp0..\..\setupenv.bat" || exit /b 1
if not exist "%~dp0..\protonbuild" mkdir "%~dp0..\protonbuild"
pushd "%~dp0..\protonbuild"
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
