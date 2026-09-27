@echo off
rem Builds ref\gammagp - the gamma general-process counter (docs/RISK.md V207, V208).
rem
rem Relative to this file (docs/RISK.md V45, V59: a worktree builds its own reference). The
rem build tree is out\gammagpbuild, under the gitignored out\ rather than beside this directory,
rem so that adding the program needed no .gitignore line. Visual Studio 16 2019 for the reason
rem ref\b1neutron\build.bat gives: the Geant4 install was built with it.
call "%~dp0..\..\setupenv.bat" || exit /b 1
if not exist "%~dp0..\..\out\gammagpbuild" mkdir "%~dp0..\..\out\gammagpbuild"
pushd "%~dp0..\..\out\gammagpbuild"
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
