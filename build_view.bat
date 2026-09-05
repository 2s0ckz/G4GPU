@echo off
rem Builds g4view: the interactive viewer.
rem
rem Links two prebuilt objects: the transport engine and the viewer itself. vis_manager.cu is
rem compiled once by build_vis.bat and linked into whatever needs it - g4view, example B1, a
rem generated project - for the same reason Geant4 ships libG4vis rather than a header: it
rem pulls in Win32, WGL, GDI and the CUDA renderer, and every program that included it would
rem pay to compile all of that again.
call "D:/g4gpu/setupenv.bat" || exit /b 1
call "D:/g4gpu/build_engine.bat" || exit /b 1
call "D:/g4gpu/build_vis.bat" || exit /b 1

rem A running executable cannot be relinked, and the linker says only
rem "LNK1104: cannot open file 'g4view.exe'" with no hint that a window is open. The same
rem check is at the top of build_all.bat, and it is here as well because a GUI process can
rem outlive the wait that started it.
tasklist /fi "imagename eq g4view.exe" 2>nul | findstr /i /c:"g4view.exe" >nul
if not errorlevel 1 (
  echo FATAL: g4view.exe is running, and a running executable cannot be relinked.
  echo        Close its window - or: taskkill /f /im g4view.exe
  exit /b 1
)

pushd "%~dp0"
nvcc -std=c++17 -O2 -arch=sm_86 -I "%~dp0src" -I "%~dp0src\g4" -o g4view.exe ^
  src\host\g4view.cu src\scenes\scene_b1.cu src\scenes\scene_b1mesh.cu ^
  "%G4GPU_ENGINE_OBJ%" "%G4GPU_VIS_OBJ%" -lopengl32 -luser32 -lgdi32 ^
  -Xlinker /IMPLIB:out/g4view.lib
if errorlevel 1 (
  popd
  exit /b 1
)
popd
echo built g4view.exe
