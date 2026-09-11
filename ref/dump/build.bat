@echo off
rem Rebuilds the oracle dumper, which links against the real Geant4.
rem
rem Every path is relative to this file, so it works from any git worktree - several packages
rem are developed in worktrees at once (docs/HADRONIC_PLAN.md section 6) and each has its own
rem ref/dump/dump_<package>.cc to build. A worktree has no ref/dumpbuild, because that directory
rem is gitignored, so the build tree is configured here on first use with the same generator
rem and Geant4 the original was.
call "%~dp0..\..\setupenv.bat" || exit /b 1
set "SRC=%~dp0."
set "BLD=%~dp0..\dumpbuild"
if not exist "%BLD%\CMakeCache.txt" (
  echo configuring %BLD%
  rem SRC must not end in a backslash: %~dp0 does, and inside -S "%SRC%" that backslash escapes
  rem the closing quote for cmake's argument parser, which then swallows `-B ... -G "Visual Studio 16`
  rem into the source path and reports a source directory of `2019 -A x64 -DGeant4_DIR=...`. The
  rem trailing dot on SRC above is the fix (P7 found it; the caret theory before it was wrong). It
  rem only ever bit in a fresh worktree - the main checkout has a CMakeCache.txt and skips this block.
  cmake -S "%SRC%" -B "%BLD%" -G "Visual Studio 16 2019" -A x64 -DGeant4_DIR="D:/Documents/Geant4/Windows/geant4-v11.1.1-install/lib/cmake/Geant4" || exit /b 1
)
cmake --build "%BLD%" --config Release
exit /b %errorlevel%
