# Builds and runs a named subset of the pipeline, for iterating on one change.
#
#   tools\quick.ps1 test_all_materials test_hadron_range     two tests
#   tools\quick.ps1 hadron                                   every test matching *hadron*
#   tools\quick.ps1 -Check proton                            the proton depth-dose check
#   tools\quick.ps1 -Check b1                                example B1 and the sigma gate
#   tools\quick.ps1 -List                                    what is available
#
# `build_all.bat` is the only thing that decides whether the port is correct, and nothing here
# replaces it - the last word before saying a change is good is still a full run. What this is
# for is the fifty runs before that one, where a full pipeline spends nineteen of its twenty
# minutes rebuilding the transport engine, the viewer, the GUI and example B1 to re-run a test
# that reads four CSV files and does arithmetic.
#
# The split it exploits: a test in tests/ is a standalone translation unit with no device
# kernels and no link against the engine, so it compiles in seconds with plain -O2 and no
# -arch. The drivers are the expensive half. A change to a physics header that only tests read
# needs none of them.
#
# When a full run is still required:
#   * anything under src/host, src/render, src/builder or src/g4 - those are what the drivers
#     are made of, and no test links them
#   * any change that could alter a dose, because the sigma gates and the batch-vs-macro,
#     mesh, per-voxel and generated-project comparisons live only in build_all.bat
#   * before saying a piece of work is finished
# Parameter sets, because ValueFromRemainingArguments does not take part in positional binding:
# with a plain `[string]$Check` declared after it, position 0 belongs to $Check and
# `quick.ps1 material hadron_range` binds "material" to -Check. Putting each mode in its own set
# with -Check and -List mandatory there makes them named-only and leaves position 0 to $Names.
[CmdletBinding(DefaultParameterSetName = 'Tests')]
param(
  [Parameter(ParameterSetName = 'Tests', Position = 0, ValueFromRemainingArguments = $true)]
  [string[]]$Names,
  [Parameter(ParameterSetName = 'Check', Mandatory = $true)][string]$Check,
  [Parameter(ParameterSetName = 'List', Mandatory = $true)][switch]$List,
  [Parameter(ParameterSetName = 'Tests')][switch]$NoBuild
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

# The pipeline's own list, read from build_all.bat rather than duplicated. A second copy of it
# here would drift, and the failure mode is a test that quietly stops being run.
$line = Select-String -Path "$root\build_all.bat" -Pattern '^set TESTS=' | Select-Object -First 1
if (-not $line) { Write-Host "could not find 'set TESTS=' in build_all.bat"; exit 1 }
$all = ($line.Line -replace '^set TESTS=', '').Trim() -split '\s+'

if ($List) {
  Write-Host ("{0} tests in build_all.bat:" -f $all.Count)
  $all | ForEach-Object { Write-Host "  $_" }
  Write-Host ""
  Write-Host "checks: proton (depth-dose vs Geant4), b1 (example B1 + sigma gate), beams (B1 proton/alpha)"
  exit 0
}

function Invoke-Env([string]$cmd) {
  & cmd /c "call `"$root\setupenv.bat`" && $cmd" 2>&1 | ForEach-Object { "$_" }
}

# ---------------------------------------------------------------- checks
if ($Check) {
  switch -Regex ($Check) {
    '^proton' {
      Write-Host "--- proton depth-dose against Geant4 ---"
      $o = Invoke-Env "`"$root\build_proton.bat`""
      if ($LASTEXITCODE -ne 0) { $o | Select-Object -Last 15; exit 1 }
      & "$root\proton_depth.exe" 6000 100 "$root\out\port_depth.csv" 0.7 0.5 | Out-Null
      if ($LASTEXITCODE -ne 0) { Write-Host "proton_depth.exe failed"; exit 1 }
      & powershell -NoProfile -ExecutionPolicy Bypass -File "$root\tools\compare_depth.ps1" `
          -Reference "$root\ref\oracle\proton_depth.csv" -Port "$root\out\port_depth.csv"
      exit $LASTEXITCODE
    }
    '^b1' {
      Write-Host "--- example B1 ---"
      $o = Invoke-Env "`"$root\examples\B1\build.bat`""
      if ($LASTEXITCODE -ne 0) { $o | Select-Object -Last 15; exit 1 }
      & "$root\examples\B1\exampleB1.exe" -n 200000 | Select-String -Pattern "dose|sigma|events/s"
      exit $LASTEXITCODE
    }
    '^beams' {
      & powershell -NoProfile -ExecutionPolicy Bypass -File "$root\tools\compare_b1_beams.ps1" -Events 20000
      exit $LASTEXITCODE
    }
    default { Write-Host "unknown check '$Check' - try proton, b1 or beams"; exit 1 }
  }
}

if (-not $Names -or $Names.Count -eq 0) {
  Write-Host "nothing named. tools\quick.ps1 -List for what there is."
  exit 1
}

# A name is either a test or a substring of several. Substring so that `quick.ps1 hadron` picks
# up every hadron test without having to remember which they are.
$sel = New-Object System.Collections.Generic.List[string]
foreach ($n in $Names) {
  $hit = $all | Where-Object { $_ -eq $n }
  if (-not $hit) { $hit = $all | Where-Object { $_ -like "*$n*" } }
  if (-not $hit) { Write-Host "no test matches '$n'"; exit 1 }
  foreach ($h in $hit) { if (-not $sel.Contains($h)) { $sel.Add($h) } }
}

Write-Host ("--- {0} of {1} tests ---" -f $sel.Count, $all.Count)
if (-not $env:G4GPU_ORACLE) { $env:G4GPU_ORACLE = "$root\ref\oracle" }

$pass = 0; $fail = 0; $failed = @()
foreach ($t in $sel) {
  if (-not $NoBuild) {
    $o = Invoke-Env "nvcc -std=c++17 -O2 -I `"$root\src`" -o `"$root\tests\$t.exe`" `"$root\tests\$t.cu`""
    if ($LASTEXITCODE -ne 0) {
      Write-Host "$t : BUILD FAILED"
      $o | Select-String -Pattern "error" | Select-Object -First 6 | ForEach-Object { "    $_" }
      $fail++; $failed += $t
      continue
    }
  }
  $out = & "$root\tests\$t.exe" 2>&1 | ForEach-Object { "$_" }
  if ($LASTEXITCODE -eq 0) {
    Write-Host ("{0,-26} ok" -f $t)
    $pass++
  } else {
    Write-Host ("{0,-26} FAILED" -f $t)
    $out | Select-String -Pattern "FAIL" | Select-Object -First 8 | ForEach-Object { "    $_" }
    $fail++; $failed += $t
  }
}

Write-Host ""
Write-Host "$pass passed, $fail failed"
if ($fail -gt 0) { Write-Host ("  " + ($failed -join " ")); exit 1 }
Write-Host "(this is a subset - build_all.bat is still what decides)"
exit 0
