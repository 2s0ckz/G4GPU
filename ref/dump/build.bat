@echo off
rem Rebuilds the oracle dumper, which links against the real Geant4.
call "D:/g4gpu/setupenv.bat" || exit /b 1
pushd "D:/g4gpu/ref/dumpbuild"
cmake --build . --config Release
set RC=%errorlevel%
popd
exit /b %RC%
