# Builds the pipeline's test lists incrementally and in parallel, and runs them.
#
#   powershell -File tools\build_tests.ps1 -Root <repo> -Plain "<names>" -Gpu "<names>" -Arch "<flags>" -Mode build
#   powershell -File tools\build_tests.ps1 -Root <repo> -Plain "<names>" -Gpu "<names>" -Mode run
#
# WHY THIS EXISTS. Until 2026-10-02 build_all.bat recompiled every test on every run and ran them
# one after another: 43 minutes of nvcc and 35 of running for 92 tests, on a gate whose actual
# validation - the two 2M-event dose gates and the depth-dose comparison - takes 25 (docs/RISK.md
# V221). A test is a standalone translation unit, so whether it needs compiling is decided by
# exactly the files nvcc read the last time it compiled it, and nvcc writes that list itself with
# -MD -MF: out\deps\<test>.d. tools\freshness.ps1 reads it; a test is rebuilt when its exe is
# missing, its dependency file is missing, any file in it is missing or newer than the exe, or the
# recipe (nvcc and the flags) changed. There is no heuristic in it, and every doubt recompiles.
#
# PARALLEL, WITH ONE RULE PER KIND. The -Plain tests are host-only translation units that call
# __host__ __device__ code on the host: one core and about a gigabyte each, so -Jobs of them
# compile at once and -Jobs of them run at once. The -Gpu tests launch real kernels: each compile
# is a ptxas of 5 to 12 GB (test_emextra_transport instantiates the steppers), so they compile
# ONE at a time - beside the plain ones, which cost little - and they run one at a time, because
# there is one GPU and a test that measures anything about it must have it to itself. No test in
# either list writes a file, which is what makes running them side by side safe; the oracle CSVs
# are read-only and G4GPU_ORACLE is inherited.
#
# WHAT IT PRINTS. Each compile's own output when it finishes (nvcc's warnings are evidence the
# gate log is read for, so they are kept, in completion order), one line per test that was up to
# date, and for the run the two lines build_all.bat has always printed - "N passed, M failed" and
# "FAILED: ..." - plus the tail of every failing test's output, which the old loop threw away.
# Exit 1 on the first compile that fails (after the ones in flight finish) and when any test fails.
param(
  [Parameter(Mandatory = $true)][string]$Root,
  [string]$Plain = "",
  [string]$Gpu = "",
  [string]$Arch = "",
  [ValidateSet("build", "run", "all")][string]$Mode = "all",
  [int]$Jobs = 6
)
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path -LiteralPath $Root).Path
Set-Location $Root
$deps = Join-Path $Root "out\deps"
if (-not (Test-Path -LiteralPath $deps)) { New-Item -ItemType Directory -Force $deps | Out-Null }
if (-not $env:G4GPU_ORACLE) { $env:G4GPU_ORACLE = Join-Path $Root "ref\oracle" }
$fresh = Join-Path $PSScriptRoot "freshness.ps1"

function Split-Names([string]$s) { return @($s -split '\s+' | Where-Object { $_ }) }
$plainNames = Split-Names $Plain
$gpuNames = Split-Names $Gpu
if ($Jobs -lt 1) { $Jobs = 1 }

# One entry per test: how it is compiled and how it is run.
function New-Item2([string]$name, [bool]$isGpu) {
  $exe = Join-Path $Root "tests\$name.exe"
  $src = Join-Path $Root "tests\$name.cu"
  $d = Join-Path $deps "$name.d"
  $flags = "-std=c++17 -O2"
  if ($isGpu) { $flags = "$flags $Arch" }
  $cmd = "$flags -I `"$Root\src`" -MD -MF `"$d`" -o `"$exe`" `"$src`""
  if ($isGpu) { $cmd = "$cmd -Xlinker /IMPLIB:out/$name.lib" }
  return [pscustomobject]@{ Name = $name; Gpu = $isGpu; Exe = $exe; Src = $src; Deps = $d
    Recipe = (Join-Path $deps "$name.recipe"); RecipeNow = $cmd; Args = $cmd
    Out = (Join-Path $deps "$name.out.log"); Err = (Join-Path $deps "$name.err.log")
    RunOut = (Join-Path $deps "$name.run.log"); RunErr = (Join-Path $deps "$name.runerr.log")
    Proc = $null; Started = $null }
}
$items = @()
foreach ($n in $plainNames) { $items += New-Item2 $n $false }
foreach ($n in $gpuNames) { $items += New-Item2 $n $true }
if ($items.Count -eq 0) { Write-Host "build_tests.ps1: no tests named"; exit 1 }

function Show-File([string]$path, [int]$tail = 0) {
  if (-not (Test-Path -LiteralPath $path)) { return }
  $lines = @(Get-Content -LiteralPath $path)
  if ($tail -gt 0 -and $lines.Count -gt $tail) { $lines = $lines[($lines.Count - $tail)..($lines.Count - 1)] }
  foreach ($l in $lines) { if ($l -ne "") { Write-Host $l } }
}

# A scheduler: launches from $queue while slots are free - a GPU item only when no other GPU item
# is in flight - and calls $onDone for each finished process. Returns the number of failures;
# stops launching after the first failure when $failFast.
function Invoke-Scheduled($queue, [scriptblock]$launch, [scriptblock]$onDone, [bool]$failFast) {
  $pending = [System.Collections.ArrayList]@($queue)
  $running = [System.Collections.ArrayList]@()
  $failures = 0
  while ($pending.Count -gt 0 -or $running.Count -gt 0) {
    $gpuBusy = @($running | Where-Object { $_.Gpu }).Count -gt 0
    if (-not ($failFast -and $failures -gt 0)) {
      $i = 0
      while ($i -lt $pending.Count -and $running.Count -lt $Jobs) {
        $it = $pending[$i]
        if ($it.Gpu -and $gpuBusy) { $i++; continue }
        $pending.RemoveAt($i)
        $it.Started = Get-Date
        $it.Proc = & $launch $it
        # Reading Handle once makes the Process object cache it, without which ExitCode reads
        # back as null after the process has gone - a documented PowerShell quirk of -PassThru.
        $null = $it.Proc.Handle
        [void]$running.Add($it)
        if ($it.Gpu) { $gpuBusy = $true }
      }
    } elseif ($running.Count -eq 0) { break }
    Start-Sleep -Milliseconds 300
    $done = @($running | Where-Object { $_.Proc.HasExited })
    foreach ($it in $done) {
      $running.Remove($it)
      $it.Proc.WaitForExit()
      if (-not (& $onDone $it $it.Proc.ExitCode)) { $failures++ }
    }
  }
  return $failures
}

$swAll = [System.Diagnostics.Stopwatch]::StartNew()
if ($Mode -eq "build" -or $Mode -eq "all") {
  $toBuild = @()
  foreach ($it in $items) {
    $why = (& $fresh -Target $it.Exe -Deps $it.Deps -Recipe $it.Recipe -RecipeNow $it.RecipeNow | Select-Object -Last 1)
    if ($why -eq "fresh") { Write-Host ("  {0}: up to date" -f $it.Name) } else { $it | Add-Member -NotePropertyName Why -NotePropertyValue $why -Force; $toBuild += $it }
  }
  Write-Host ("compiling {0} of {1} tests ({2} up to date), {3} host-only at a time, kernel tests one at a time" -f $toBuild.Count, $items.Count, ($items.Count - $toBuild.Count), $Jobs)
  $launch = {
    param($it)
    foreach ($f in @($it.Out, $it.Err)) { if (Test-Path -LiteralPath $f) { Remove-Item -LiteralPath $f -Force } }
    if (Test-Path -LiteralPath $it.Exe) { Remove-Item -LiteralPath $it.Exe -Force }
    return Start-Process -FilePath "nvcc" -ArgumentList $it.Args -WorkingDirectory $Root -NoNewWindow -PassThru `
      -RedirectStandardOutput $it.Out -RedirectStandardError $it.Err
  }
  $onDone = {
    param($it, $code)
    $secs = [int]((Get-Date) - $it.Started).TotalSeconds
    Write-Host ("--- {0} ({1}; {2} s, exit {3})" -f $it.Name, $it.Why, $secs, $code)
    Show-File $it.Out
    Show-File $it.Err
    if ($code -eq 0 -and (Test-Path -LiteralPath $it.Exe)) {
      & $fresh -Recipe $it.Recipe -RecipeNow $it.RecipeNow -Stamp | Out-Null
      return $true
    }
    Write-Host ("FATAL: {0} did not compile." -f $it.Name)
    return $false
  }
  $failed = Invoke-Scheduled $toBuild $launch $onDone $true
  Write-Host ("tests built in {0} s" -f [int]$swAll.Elapsed.TotalSeconds)
  if ($failed -gt 0) { exit 1 }
}

if ($Mode -eq "run" -or $Mode -eq "all") {
  $swRun = [System.Diagnostics.Stopwatch]::StartNew()
  foreach ($it in $items) {
    if (-not (Test-Path -LiteralPath $it.Exe)) { Write-Host ("FATAL: tests\{0}.exe does not exist - was it built?" -f $it.Name); exit 1 }
  }
  $passed = @(); $failedNames = @()
  $launch = {
    param($it)
    foreach ($f in @($it.RunOut, $it.RunErr)) { if (Test-Path -LiteralPath $f) { Remove-Item -LiteralPath $f -Force } }
    return Start-Process -FilePath $it.Exe -WorkingDirectory $Root -NoNewWindow -PassThru `
      -RedirectStandardOutput $it.RunOut -RedirectStandardError $it.RunErr
  }
  $onDone = {
    param($it, $code)
    if ($code -eq 0) { $script:passed += $it.Name; return $true }
    $script:failedNames += $it.Name
    Write-Host ("--- {0} FAILED (exit {1}); the last lines of its output:" -f $it.Name, $code)
    Show-File $it.RunOut 40
    Show-File $it.RunErr 20
    return $false
  }
  [void](Invoke-Scheduled $items $launch $onDone $false)
  Write-Host ("{0} passed, {1} failed" -f $passed.Count, $failedNames.Count)
  Write-Host ("tests ran in {0} s" -f [int]$swRun.Elapsed.TotalSeconds)
  if ($failedNames.Count -gt 0) { Write-Host ("FAILED: " + ($failedNames -join " ")); exit 1 }
}
exit 0
