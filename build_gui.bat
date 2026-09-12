@echo off
rem Builds g4builder: the model-building GUI.
rem
rem Links the prebuilt engine archive rather than compiling the engine's seventeen translation
rem units itself - minutes of nvcc for files that have not changed. build_engine.bat decides by
rem timestamp whether they need rebuilding (tools/freshness.ps1).
rem
rem The viewer's translation unit is *not* linked here: the builder has its own render path
rem and its own panels, and pulling in vis_manager.obj would give it a second window
rem implementation it never opens.
call "%~dp0setupenv.bat" || exit /b 1
call "%~dp0build_engine.bat" || exit /b 1

rem A running executable cannot be relinked, and the linker says only
rem "LNK1104: cannot open file 'g4builder.exe'" with no hint that a window is open. The same
rem check is at the top of build_all.bat, and it is here as well because a GUI process can
rem outlive the wait that started it: the pipeline's check passed, the process reappeared, and
rem twenty minutes of build ended on that line.
tasklist /fi "imagename eq g4builder.exe" 2>nul | findstr /i /c:"g4builder.exe" >nul
if not errorlevel 1 (
  echo FATAL: g4builder.exe is running, and a running executable cannot be relinked.
  echo        Close its window - or: taskkill /f /im g4builder.exe
  exit /b 1
)

pushd "%~dp0"
nvcc -std=c++17 -O2 -arch=sm_86 -I "%~dp0src" -I "%~dp0src\g4" -o g4builder.exe ^
  src\host\g4builder.cu src\builder\write_project.cc ^
  "%G4GPU_ENGINE_OBJ%" -lopengl32 -luser32 -lgdi32 -lcomdlg32 ^
  -Xlinker /IMPLIB:out/g4builder.lib
if errorlevel 1 (
  popd
  exit /b 1
)
popd
echo built g4builder.exe
