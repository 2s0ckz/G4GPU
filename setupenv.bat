@echo off
rem Sets up the MSVC environment nvcc needs, at most once.
rem
rem vcvars64.bat cannot be run twice in the same environment: the second call fails, and the
rem failure surfaces later as "nvcc fatal : Could not set up the environment for Microsoft
rem Visual Studio" - reported by nvcc, from a path it built by walking up out of its own
rem install directory, with nothing pointing at the nested build script that actually caused
rem it. Every build script here goes through this file so that nesting is safe.
if defined VSINSTALLDIR exit /b 0
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if not defined VSINSTALLDIR (
  echo FATAL: could not find Visual Studio 2019 x64 build tools.
  echo   CUDA 11.6 requires MSVC 14.29 ^(VS 2019^); it rejects 14.4x ^(VS 2022^).
  exit /b 1
)
exit /b 0
