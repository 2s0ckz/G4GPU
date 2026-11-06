@echo off
rem Sets up the MSVC environment nvcc needs, the CUDA toolkit, and the GPU architecture the
rem build compiles for - at most once per environment.
rem
rem vcvars64.bat cannot be run twice in the same environment: the second call fails, and the
rem failure surfaces later as "nvcc fatal : Could not set up the environment for Microsoft
rem Visual Studio" - reported by nvcc, from a path it built by walking up out of its own
rem install directory, with nothing pointing at the nested build script that actually caused
rem it. Every build script here goes through this file so that nesting is safe.
if defined VSINSTALLDIR exit /b 0

rem ---------------------------------------------------------------- the CUDA toolkit
rem
rem CUDA 12.9.2 (nvcc and ptxas 12.9.86), as a PORTABLE toolkit under D:\cuda129 on the gating
rem machine: NVIDIA's own component archives (cuda_nvcc, cuda_cudart, cuda_cccl, cuda_cuobjdump),
rem SHA-256 checked against the redistrib_12.9.2.json index, unzipped and merged, no installer and
rem nothing system-wide changed. It goes FIRST on the PATH so nvcc, ptxas and cuobjdump are
rem 12.9's. The reason is docs/RISK.md V203: CUDA 11.6's ptxas dies with an access violation on
rem the big stepping kernels at -O3 and -O2, and at -O1 it died or compiled on the same input
rem (V202), so the engine was shipping a GenericIon kernel compiled at -O0; 12.9's ptxas
rem compiles it at -O3 in two minutes. ANY OTHER MACHINE has no D:\cuda129 and uses the nvcc on
rem its PATH, which must be a CUDA 12.x toolkit: 11.x cannot compile these kernels (V203) and
rem 13.x dropped Volta, the floor this tree is built to. The build says which nvcc it found.
if exist "D:\cuda129\bin\nvcc.exe" (
  set "PATH=D:\cuda129\bin;%PATH%"
  set "CUDA_PATH=D:\cuda129"
) else (
  echo setupenv: D:\cuda129 not found - using the nvcc on the PATH ^(docs/RISK.md V203: it must be CUDA 12.x^)
)
where nvcc >nul 2>&1
if errorlevel 1 (
  echo FATAL: no nvcc on the PATH. Install a CUDA 12.x toolkit ^(12.9 is what this tree is gated
  echo   with^) and put its bin folder on the PATH, or set CUDA_PATH and add %%CUDA_PATH%%\bin.
  exit /b 1
)
rem NVCC_APPEND_FLAGS reaches every nvcc call in this tree and in the projects the model builder
rem writes, which is why these flags live here and not on fourteen command lines - and they are
rem set whatever toolkit was found above, because every one of them is about the code, not the
rem machine. The engine is one kernel per translation unit, declared `extern template` in
rem transport_run_impl.cuh (V65); nvcc 12 warns (#20279-D) that its future default,
rem -static-global-template-stub=true, will break exactly that, so the build says false.
rem -Wno-deprecated-gpu-targets: nvcc 12.9 warns on every compile that offline compilation for
rem architectures before sm_75 "will be removed in a future release". That is the compute_70
rem floor below, it is deliberate (V239), and CUDA 12.x is the toolkit this tree stays on, so the
rem warning says nothing a build log needs forty times. Anything the caller had in the variable
rem already is kept after them.
if defined NVCC_APPEND_FLAGS (set "NVCC_APPEND_FLAGS=-static-global-template-stub=false -Wno-deprecated-gpu-targets %NVCC_APPEND_FLAGS%") else (set "NVCC_APPEND_FLAGS=-static-global-template-stub=false -Wno-deprecated-gpu-targets")

rem ---------------------------------------------------------------- the GPU architecture
rem
rem Every nvcc line that builds device code says %G4GPU_ARCH% (build_all.bat, build_engine_unit.bat,
rem the drivers, examples/B1 and the projects the model builder writes), so the cards are chosen
rem here once - and since docs/RISK.md V239 they are DETECTED rather than assumed: nvidia-smi is
rem asked for the compute capability of every card present, and each distinct one from 7.0 up
rem becomes a -gencode pair with its own SASS, so a clone on another machine compiles for exactly
rem the card it has (an H100 gets sm_90 and nothing else) and the gating machine gets its RTX 3070
rem (8.6, CUDA device 0, where every gate runs) and its Titan V (7.0, device 1). Every unit's
rem ptxas runs once per card. The compute_70 PTX is embedded as well - beside the sm_70 SASS when
rem a Volta card is present, which costs no ptxas time, or on its own - so the same binary runs
rem on ANY NVIDIA card from Volta up under a CUDA 12 driver: a card no pair names JITs the PTX at
rem its first launch and the driver caches the result (V239 measured what that costs). Volta is
rem the floor because the gamma and lepton stepping kernels take `__grid_constant__` parameters
rem (compute 7.0), not because of the toolkit: a card below it is refused by name here, and a
rem machine whose nvidia-smi answers nothing gets the PTX pair alone and a line saying so.
rem A caller that wants something else sets G4GPU_ARCH before calling any build script - for
rem instance -arch=sm_86 for one card and no PTX - and tools\freshness.ps1 rebuilds everything
rem the recipe no longer matches.
if defined G4GPU_ARCH goto :arch_done
for /f "usebackq tokens=1 delims=," %%C in (`nvidia-smi --query-gpu=compute_cap --format=csv^,noheader 2^>nul`) do call :add_cap %%C
if not defined G4GPU_ARCH (
  echo setupenv: nvidia-smi reported no GPU of compute capability 7.0 or above - building the
  echo   compute_70 PTX and sm_70 SASS alone, which any Volta-or-newer card runs ^(JIT at first launch^)
  set "G4GPU_ARCH=-gencode arch=compute_70,code=[sm_70,compute_70]"
  goto :arch_done
)
echo %G4GPU_ARCH% | findstr /c:"compute_70]" >nul
if errorlevel 1 set "G4GPU_ARCH=%G4GPU_ARCH% -gencode arch=compute_70,code=compute_70"
:arch_done
echo setupenv: G4GPU_ARCH=%G4GPU_ARCH%

rem ---------------------------------------------------------------- MSVC
rem
rem Found through vswhere, which every Visual Studio installer puts in the same place, so that a
rem clone on another machine builds with whatever edition of 2019 or 2022 it has. 2019 (MSVC
rem 14.29) is preferred when both are present, because it is what this tree is gated with; CUDA
rem 12.9 accepts 14.29 through 14.4x, so 2022 is a different compiler from the gated one and not
rem a wrong one. The old fixed path is the last resort.
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
set "VCVARS="
if exist "%VSWHERE%" for /f "usebackq delims=" %%P in (`"%VSWHERE%" -products * -version [16.0^,17.0^) -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do call :take_vs "%%P"
if exist "%VSWHERE%" for /f "usebackq delims=" %%P in (`"%VSWHERE%" -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -sort -property installationPath`) do call :take_vs "%%P"
if not defined VCVARS call :take_vs "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community"
if not defined VCVARS (
  echo FATAL: no Visual Studio with the x64 C++ build tools was found ^(vswhere^).
  echo   Install Visual Studio 2019 or 2022 with "Desktop development with C++"; CUDA 12.9 accepts
  echo   MSVC 14.29 ^(2019^) through 14.4x ^(2022^), and this tree is built and gated with 14.29.
  exit /b 1
)
call "%VCVARS%" >nul 2>&1
if not defined VSINSTALLDIR (
  echo FATAL: "%VCVARS%" did not set up the x64 build tools.
  exit /b 1
)

rem ---------------------------------------------------------------- the Geant4 datasets
rem
rem Not a build input - a RUN input, which host/g4data.cuh resolves at Upload through the Geant4
rem dataset variables, G4GPU_DATA_DIR, GEANT4_DATA_DIR or the gating machine's install. Said once
rem here, so that a fresh clone learns it before its first run and not from the run.
if defined G4GPU_DATA_DIR goto :data_done
if defined GEANT4_DATA_DIR goto :data_done
if defined G4LEDATA goto :data_done
if exist "D:\Documents\Geant4\Windows\geant4-v11.1.1-install\share\Geant4\data" goto :data_done
echo setupenv: note - no Geant4 datasets are configured. A run needs G4EMLOW, PhotonEvaporation,
echo   G4PARTICLEXS, G4ENSDFSTATE and RadioactiveDecay ^(Geant4 11.1.1's versions^) in one folder
echo   named by G4GPU_DATA_DIR; see README.md, "A fresh clone". The build does not need them.
:data_done
exit /b 0

rem ---------------------------------------------------------------- helpers
rem One nvidia-smi line, "8.6": a -gencode pair for its SASS, once per distinct capability,
rem refused by name below 7.0.
:add_cap
set "CAP=%~1"
set "CAP=%CAP:.=%"
if "%CAP%"=="" exit /b 0
if %CAP% LSS 70 (
  echo setupenv: a GPU of compute capability %~1 is present and is NOT built for - this tree needs
  echo   7.0 or above ^(the stepping kernels take __grid_constant__ parameters^)
  exit /b 0
)
if defined G4GPU_ARCH echo %G4GPU_ARCH% | findstr /c:"sm_%CAP%" >nul && exit /b 0
if "%CAP%"=="70" (set "PAIR=-gencode arch=compute_70,code=[sm_70,compute_70]") else (set "PAIR=-gencode arch=compute_%CAP%,code=sm_%CAP%")
if defined G4GPU_ARCH (set "G4GPU_ARCH=%G4GPU_ARCH% %PAIR%") else (set "G4GPU_ARCH=%PAIR%")
set "PAIR="
exit /b 0

rem One vswhere line: the first install whose vcvars64.bat exists wins.
:take_vs
if defined VCVARS exit /b 0
if exist "%~1\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS=%~1\VC\Auxiliary\Build\vcvars64.bat"
exit /b 0
