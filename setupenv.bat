@echo off
rem Sets up the MSVC environment nvcc needs, at most once.
rem
rem vcvars64.bat cannot be run twice in the same environment: the second call fails, and the
rem failure surfaces later as "nvcc fatal : Could not set up the environment for Microsoft
rem Visual Studio" - reported by nvcc, from a path it built by walking up out of its own
rem install directory, with nothing pointing at the nested build script that actually caused
rem it. Every build script here goes through this file so that nesting is safe.
if defined VSINSTALLDIR exit /b 0
rem ---------------------------------------------------------------- the GPU architecture
rem
rem Every nvcc line that builds device code says %G4GPU_ARCH% (build_all.bat, build_engine_unit.bat,
rem the drivers, examples/B1 and the projects the model builder writes), so the cards are chosen
rem here once. THE DEFAULT IS BOTH CARDS OF THE GATING MACHINE since docs/RISK.md V234: the RTX
rem 3070 (compute capability 8.6, CUDA device 0, where every gate runs) and the Titan V (7.0,
rem device 1, 12 GB of HBM2 and FP64 at half rate), as two -gencode pairs, so every unit is
rem compiled for both at twice the ptxas time and one binary runs on either card - the Titan V
rem under CUDA_VISIBLE_DEVICES=1 for the whole tree, or G4GPU_DEVICE=titan for the engine and
rem the drivers. AND THE compute_70 PTX IS EMBEDDED BESIDE THE sm_70 SASS, which costs no ptxas
rem time (the PTX is generated on the way to the SASS) and makes the same binary run on ANY
rem NVIDIA card from Volta up under this driver: a card neither pair names - Turing, Ada,
rem Hopper, Blackwell - JITs the PTX at its first launch and the driver caches the result.
rem Volta is the floor because the kernels take `__grid_constant__` parameters (compute 7.0),
rem not because of the toolkit. A tree for one card sets this to its -arch before calling any
rem build script, for instance -arch=sm_86, and tools\freshness.ps1 rebuilds everything the
rem recipe no longer matches.
if not defined G4GPU_ARCH set "G4GPU_ARCH=-gencode arch=compute_70,code=[sm_70,compute_70] -gencode arch=compute_86,code=sm_86"
rem ---------------------------------------------------------------- the CUDA toolkit
rem
rem CUDA 12.9.2 (nvcc and ptxas 12.9.86), as a PORTABLE toolkit under D:\cuda129: NVIDIA's own
rem component archives (cuda_nvcc, cuda_cudart, cuda_cccl, cuda_cuobjdump), SHA-256 checked
rem against the redistrib_12.9.2.json index, unzipped and merged, no installer and nothing
rem system-wide changed. It goes FIRST on the PATH so nvcc, ptxas and cuobjdump are 12.9's.
rem The reason is docs/RISK.md V203: CUDA 11.6's ptxas dies with an access violation on the
rem big stepping kernels at -O3 and -O2, and at -O1 it died or compiled on the same input
rem (V202), so the engine was shipping a GenericIon kernel compiled at -O0; 12.9's ptxas
rem compiles it at -O3 in two minutes. Without the folder the build falls back to the nvcc on
rem the PATH, and says which one it found.
rem
rem NVCC_APPEND_FLAGS reaches every nvcc call in this tree and in the projects the model
rem builder writes, which is why the flag lives here and not on fourteen command lines. The
rem engine is one kernel per translation unit, declared `extern template` in
rem transport_run_impl.cuh (V65); nvcc 12 warns (#20279-D) that its future default,
rem -static-global-template-stub=true, will break exactly that, so the build says false.
if exist "D:\cuda129\bin\nvcc.exe" (
  set "PATH=D:\cuda129\bin;%PATH%"
  set "CUDA_PATH=D:\cuda129"
  rem -Wno-deprecated-gpu-targets: nvcc 12.9 warns on every compile that offline compilation
  rem for architectures before sm_75 "will be removed in a future release". It is the Titan V's
  rem sm_70, it is deliberate (V234), and CUDA 12.x is the toolkit this tree stays on (V203:
  rem 13.x dropped Volta), so the warning says nothing a build log needs forty times.
  set "NVCC_APPEND_FLAGS=-static-global-template-stub=false -Wno-deprecated-gpu-targets"
) else (
  echo setupenv: D:\cuda129 not found - using the nvcc on the PATH ^(see docs/RISK.md V203^)
)
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if not defined VSINSTALLDIR (
  echo FATAL: could not find Visual Studio 2019 x64 build tools.
  echo   CUDA 12.9 accepts MSVC 14.29 ^(VS 2019^); this tree is built and gated with it.
  exit /b 1
)
exit /b 0
