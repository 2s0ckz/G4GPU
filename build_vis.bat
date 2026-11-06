@echo off
rem Compiles the visualisation manager to an object file, when it needs it.
rem
rem render/vis_manager.cu is the window, the GL context, the immediate-mode UI and the CUDA
rem render dispatch. It is what Geant4 ships as libG4vis: the viewer, linked into whatever
rem program wants one. g4view.exe is a thin main over it, and any example whose macro says
rem /vis/open reaches the same code.
rem
rem The skip is a timestamp comparison, not an environment variable - see build_engine.bat
rem and docs/RISK.md S4 for why that distinction is not pedantry.
call "%~dp0setupenv.bat" || exit /b 1
pushd "%~dp0"
if not exist out mkdir out
set G4GPU_VIS_OBJ=%~dp0out\vis_manager.obj
set FRESH=stale
rem Against the recipe too, as build_hook_engine.bat since docs/RISK.md V239: the object carries
rem device code for whatever G4GPU_ARCH said when it was compiled.
for /f "usebackq delims=" %%R in (`powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\freshness.ps1" -Obj "%~dp0out\vis_manager.obj" -Unit "%~dp0src\render\vis_manager.cu" -SrcDir "%~dp0src" -Recipe "%~dp0out\vis_manager.obj.recipe" -RecipeNow "%G4GPU_ARCH%"`) do set FRESH=%%R
if /i "%FRESH%"=="fresh" (
  popd
  echo vis_manager.obj is up to date
  exit /b 0
)
nvcc -std=c++17 -O2 %G4GPU_ARCH% -I "%~dp0src" -I "%~dp0src\g4" -c ^
  -o out\vis_manager.obj src\render\vis_manager.cu
if errorlevel 1 (
  popd
  exit /b 1
)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\freshness.ps1" -Recipe "%~dp0out\vis_manager.obj.recipe" -RecipeNow "%G4GPU_ARCH%" -Stamp
popd
echo built vis_manager.obj
exit /b 0
