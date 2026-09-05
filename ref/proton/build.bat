@echo off
rem Builds the real-Geant4 half of the proton depth-dose comparison.
rem
rem Same source file as build_proton.bat at the repository root, without -DG4GPU_PORT. That is
rem the point of the exercise: the geometry, the beam, the binning and the physics list are one
rem description compiled twice, so the two sides cannot be set up differently.
rem
rem Configures on first use. Visual Studio 16 2019 to match ref/dumpbuild - the Geant4 install
rem was built with it, and mixing runtimes across a library boundary is not worth discovering
rem at link time.
call "D:/g4gpu/setupenv.bat" || exit /b 1
if not exist "D:/g4gpu/ref/protonbuild" mkdir "D:/g4gpu/ref/protonbuild"
pushd "D:/g4gpu/ref/protonbuild"
if not exist CMakeCache.txt (
  cmake -G "Visual Studio 16 2019" -A x64 -DCMAKE_BUILD_TYPE=Release ^
    -DGeant4_DIR=D:/Documents/Geant4/Windows/geant4-v11.1.1-install/lib/cmake/Geant4 ^
    "D:/g4gpu/ref/proton"
  if errorlevel 1 (
    popd
    exit /b 1
  )
)
cmake --build . --config Release
set RC=%errorlevel%
popd
exit /b %RC%
