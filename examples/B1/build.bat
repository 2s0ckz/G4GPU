@echo off
rem Builds example B1. The layout is a Geant4 project's - exampleB1.cc, src/, include/ - and
rem the sources are .cc as they would be there, compiled by nvcc with -x cu because the
rem headers they include reach down into device code.
rem
rem CMakeLists.txt builds the same thing with cmake, which is what a Geant4 user will reach
rem for; this script is what build_all.bat uses because it is faster and has no dependency
rem beyond the CUDA toolkit.
rem
rem Object files go to out\B1 at the repository root rather than into a directory here. A
rem project that imports CAD meshes should not have a pile of files called *.obj sitting next
rem to its source, meaning something completely different.
rem
rem Two compiler invocations, not one. -x cu applies to *every* input that follows it and nvcc
rem has no -x none to turn it back off, so a single command that also listed the prebuilt
rem engine object fed that object to the CUDA frontend and reported
rem "transport_run.obj(4): fatal error C1004: unexpected end-of-file found" - a parse error,
rem on a binary, at line 4. Compiling to objects and linking them separately avoids the
rem question entirely.
rem
rem -c, not -dc: these files launch no kernels (the launches live in transport_run.cu), so they
rem need no relocatable device code, and mixing -dc objects with the engine object - which is
rem compiled without it - would fail at device link.
call "%~dp0../../setupenv.bat" || exit /b 1
call "%~dp0../../build_engine.bat" || exit /b 1
call "%~dp0../../build_vis.bat" || exit /b 1
rem pushd, not cd: build_all.bat calls this, and a permanent directory change would leave the
rem caller compiling tests\*.cu relative to this directory.
pushd "%~dp0"
set OBJ=%~dp0../../out/B1
if not exist "%OBJ%" mkdir "%OBJ%"
set INC=-I "%~dp0include" -I "%~dp0../../src" -I "%~dp0../../src/g4"
set SRCS=exampleB1 ActionInitialization DetectorConstruction EventAction PrimaryGeneratorAction RunAction SteppingAction

for %%F in (%SRCS%) do (
  if exist "%%F.cc" (
    nvcc -std=c++17 -O2 -arch=sm_86 %INC% -x cu -c -o "%OBJ%\%%F.obj" "%%F.cc" || goto :fail
  ) else (
    nvcc -std=c++17 -O2 -arch=sm_86 %INC% -x cu -c -o "%OBJ%\%%F.obj" "src\%%F.cc" || goto :fail
  )
)

nvcc -std=c++17 -O2 -arch=sm_86 -o exampleB1.exe ^
  "%OBJ%\exampleB1.obj" "%OBJ%\ActionInitialization.obj" "%OBJ%\DetectorConstruction.obj" ^
  "%OBJ%\EventAction.obj" "%OBJ%\PrimaryGeneratorAction.obj" "%OBJ%\RunAction.obj" ^
  "%OBJ%\SteppingAction.obj" "%G4GPU_ENGINE_OBJ%" "%G4GPU_VIS_OBJ%" ^
  -lopengl32 -luser32 -lgdi32 -Xlinker /IMPLIB:%OBJ%/exampleB1.lib || goto :fail
popd
echo built exampleB1.exe
exit /b 0

:fail
popd
exit /b 1
