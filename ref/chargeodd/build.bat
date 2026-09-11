@echo off
rem Builds the V44 charge-split diagnostic against the real Geant4 11.1.1.
rem
rem EVERY path here is relative to this file (%~dp0). docs/RISK.md V45: every other driver
rem build script in this repository calls D:/g4gpu/build_engine.bat by absolute path, so a
rem build launched from a git worktree compiles main's sources and prints main's physics under
rem the branch's name. This program has no port code in it, but the build TREE would still
rem have been shared with the main checkout and with the other Phase-2 agents, and two
rem cmake --build runs into one tree at once is the same class of mistake.
call "%~dp0..\..\setupenv.bat" || exit /b 1
rem The trailing dot matters. %~dp0 ends in a backslash, so "%~dp0" hands cmake a `\"` that
rem MSVCRT's argv parser reads as an ESCAPED QUOTE: the -S argument then swallows -B and -G and
rem cmake reports `The source directory ".../2019 -A x64 ..." does not exist`. Appending "." is
rem the same directory without the trailing separator. ref/dump/build.bat has this defect too.
set "SRC=%~dp0."
set "BLD=%~dp0..\chargeoddbuild"
if not exist "%BLD%" mkdir "%BLD%"
if not exist "%BLD%\CMakeCache.txt" (
  rem One line, no caret continuation: under a CRLF checkout cmd reads the continuation as the
  rem end of the command and passes the tail as the source directory. See ref/dump/build.bat.
  cmake -S "%SRC%" -B "%BLD%" -G "Visual Studio 16 2019" -A x64 -DGeant4_DIR="D:/Documents/Geant4/Windows/geant4-v11.1.1-install/lib/cmake/Geant4" || exit /b 1
)
cmake --build "%BLD%" --config Release
exit /b %errorlevel%
