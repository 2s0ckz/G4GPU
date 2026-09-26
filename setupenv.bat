@echo off
rem Sets up the MSVC environment nvcc needs, at most once.
rem
rem vcvars64.bat cannot be run twice in the same environment: the second call fails, and the
rem failure surfaces later as "nvcc fatal : Could not set up the environment for Microsoft
rem Visual Studio" - reported by nvcc, from a path it built by walking up out of its own
rem install directory, with nothing pointing at the nested build script that actually caused
rem it. Every build script here goes through this file so that nesting is safe.
if defined VSINSTALLDIR exit /b 0
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
  set "NVCC_APPEND_FLAGS=-static-global-template-stub=false"
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
