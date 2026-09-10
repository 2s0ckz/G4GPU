@echo off
rem Builds one test by name: build_one_test.bat test_solids
call "%~dp0setupenv.bat" || exit /b 1
pushd "%~dp0"
nvcc -std=c++17 -O2 -I "%~dp0src" -I "%~dp0src\g4" -o tests\%1.exe tests\%1.cu
set RC=%errorlevel%
popd
exit /b %RC%
