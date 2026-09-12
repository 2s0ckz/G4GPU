@echo off
rem Builds one test by name: build_one_test.bat test_solids
call "%~dp0setupenv.bat" || exit /b 1

rem The two custom-hook PROJECTS are refused here rather than attempted, because the line below
rem cannot build them and the way it fails is unhelpful: each needs its hook class on the -I
rem line, its generated declarations from out\hook_<tag>, its own kernel archive to link, and
rem -arch=sm_86, none of which this script has. Without the declarations it would instead
rem instantiate all eighteen stepping kernels into the test's own translation unit - minutes of
rem nvcc ending in `ptxas died with status 0xC0000005`. docs/RISK.md V65.
for %%P in (test_custom_hook test_voxel_scoring) do (
  if /i "%1"=="%%P" (
    echo %%P is a project with its own step hook, not a plain test, and it is built by
    echo build_all.bat: build_hook_engine.bat compiles its kernels into out\hook_^<tag^>.lib
    echo and the project links that. See the TESTS_PROJECT and TESTS_HOOK blocks there.
    exit /b 1
  )
)
pushd "%~dp0"
nvcc -std=c++17 -O2 -I "%~dp0src" -I "%~dp0src\g4" -o tests\%1.exe tests\%1.cu
set RC=%errorlevel%
popd
exit /b %RC%
