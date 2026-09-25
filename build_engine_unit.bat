@echo off
rem Compiles ONE translation unit of the transport engine, and records its exit code.
rem
rem build_engine.bat starts eight of these at once - one per kernel family - and then waits for
rem the .rc files. This is a separate script rather than a `start /b cmd /c "..."` written into
rem that loop because the inner command needs its own redirections and && || operators, and
rem inside a quoted cmd /c inside a for body every one of them has to be escaped with a caret.
rem That is unreadable, and a build script nobody can read is how docs/RISK.md S4 happened.
rem
rem   %1  the .cu to compile, full path
rem   %2  the output directory
rem
rem Writes %2\<unit>.obj, %2\<unit>.log and %2\<unit>.rc. The .rc is written LAST and is the
rem only thing build_engine.bat waits on, so a unit that is still compiling cannot be mistaken
rem for one that finished: the object appears before the exit code does.
rem
rem -Xptxas -v is on, and the log is where the engine's register report lives. It costs nothing
rem - ptxas is already running - and this project's register budget is a measured quantity
rem rather than a remembered one (docs/RISK.md V22): out\transport_run_*.log holds the
rem registers, the stack frame and the spill counts for every kernel of the build that is
rem actually linked, per unit, without anyone having to rebuild 24 minutes of nvcc to ask.
call "%~dp0setupenv.bat" || exit /b 1
set UNIT=%~n1
set OUTDIR=%~2
if exist "%OUTDIR%\%UNIT%.rc" del /q "%OUTDIR%\%UNIT%.rc"
nvcc -std=c++17 -O2 -arch=sm_86 -I "%~dp0src" -I "%~dp0src\g4" -Xptxas -v -c ^
  -o "%OUTDIR%\%UNIT%.obj" "%~1" > "%OUTDIR%\%UNIT%.log" 2>&1

rem ---------------------------------------------------------------- the -O1 retry, and why
rem
rem `ptxas died with status 0xC0000005 (ACCESS_VIOLATION)` is docs/RISK.md V55, V63 and V65's
rem failure mode, and P15 met it again: with the inelastic process wired into `step_hadron`,
rem `transport_run_proton.cu` and `transport_run_antiproton.cu` - and ONLY those two of the
rem seventeen stepping units - stopped compiling at `-Xptxas -O2`. Four `__noinline__`
rem boundaries were tried first and each moved the crash without removing it:
rem `inelastic_xs_per_volume`, `choose_inelastic_model`, `enqueue_interaction` and
rem `inelastic_length`, which between them leave a MaterialXs, a ModelRange array and a
rem 392-byte queue entry entirely out of the kernel's own frame. V63 tried nine arrangements
rem and got one to compile; this is the same wall.
rem
rem `-Xptxas -O1` compiles the same source in the same time - 78 s against 69 - and the frame
rem it reports is 4,096 bytes against the 3,936 `-O2` gave before the process was wired. So the
rem retry is here, it is PER UNIT so nothing that compiles at -O2 is degraded, and it PRINTS,
rem because a build that silently drops an optimisation level is a build whose numbers nobody
rem can explain later. The `.o1` file beside the log records which units needed it, so the list
rem is an artefact rather than a memory.
if errorlevel 1 (
  findstr /c:"ACCESS_VIOLATION" "%OUTDIR%\%UNIT%.log" >nul
  if not errorlevel 1 (
    echo   %UNIT%: ptxas died at -O2 with an access violation; retrying at -Xptxas -O1 ^(docs/RISK.md V63^)
    nvcc -std=c++17 -O2 -arch=sm_86 -Xptxas -O1 -I "%~dp0src" -I "%~dp0src\g4" -Xptxas -v -c ^
      -o "%OUTDIR%\%UNIT%.obj" "%~1" > "%OUTDIR%\%UNIT%.log" 2>&1
    if not errorlevel 1 (
      > "%OUTDIR%\%UNIT%.o1" echo ptxas -O1
      > "%OUTDIR%\%UNIT%.rc" echo 0
      exit /b 0
    )
  )
  > "%OUTDIR%\%UNIT%.rc" echo 1
  exit /b 1
)
rem The redirection comes FIRST on both lines. `echo 0 > file` writes "0 " - echo takes
rem everything up to the redirect, trailing space included - and build_engine.bat compares the
rem line against "0", so that one space would have reported every successful unit as a
rem failure. Found by reading it; it is the sort of thing that only fails once it matters.
if exist "%OUTDIR%\%UNIT%.o1" del /q "%OUTDIR%\%UNIT%.o1"
> "%OUTDIR%\%UNIT%.rc" echo 0
exit /b 0
