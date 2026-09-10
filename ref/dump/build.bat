@echo off
rem Rebuilds the oracle dumper, which links against the real Geant4.
rem
rem Every path is relative to this file, so it works from any git worktree - several packages
rem are developed in worktrees at once (docs/HADRONIC_PLAN.md section 6) and each has its own
rem ref/dump/dump_<package>.cc to build. A worktree has no ref/dumpbuild, because that directory
rem is gitignored, so the build tree is configured here on first use with the same generator
rem and Geant4 the original was.
call "%~dp0..\..\setupenv.bat" || exit /b 1
set "SRC=%~dp0"
set "BLD=%~dp0..\dumpbuild"
if not exist "%BLD%\CMakeCache.txt" (
  echo configuring %BLD%
  rem One line on purpose: a caret continuation inside this block fails under a CRLF checkout -
  rem which is what a fresh worktree gets with core.autocrlf - and cmd then passes the tail of
  rem the command as the source directory. The main checkout has LF endings and never showed it.
  cmake -S "%SRC%" -B "%BLD%" -G "Visual Studio 16 2019" -A x64 -DGeant4_DIR="D:/Documents/Geant4/Windows/geant4-v11.1.1-install/lib/cmake/Geant4" || exit /b 1
)
cmake --build "%BLD%" --config Release
exit /b %errorlevel%
