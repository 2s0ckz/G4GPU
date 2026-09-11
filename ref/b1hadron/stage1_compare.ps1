# The stage-1 like-for-like: this port against Geant4 11.1.1, example B1, per species.
#
#   ref\b1hadron\stage1_compare.ps1 [-Events 500000] [-Species "pion_plus,kaon_minus"]
#
# QUOTE THE SPECIES LIST. Unquoted, PowerShell reads `a,b` as an array, binds it to this
# `[string]` parameter by stringifying it space-separated, and `-split ","` then yields one
# element - so the script goes looking for `stage1_pion_plus kaon_minus_port.mac` and fails on
# a path, not on the argument.
#
# THREE RUNS PER SPECIES, and the third is what makes the table readable.
#
#   port                stage1_<s>_port.mac      through examples\B1\exampleB1.exe
#   Geant4 stage 1      stage1_<s>.mac           Decay AND hadElastic active
#   Geant4 no elastic   stage1_<s>_noelastic.mac Decay active, hadElastic inactivated
#
# The port has decay and does not yet have hadElastic, so the LIKE-FOR-LIKE column is the third
# one and the second is where it is going. Their difference is what hadElastic is worth for that
# species in that geometry - measured, not asserted - and it is the number the next package is
# judged against. Printing only one of them would have left that as a sentence.
#
# See stage1_README.md for what stage 1 inactivates and why the neutron has no third column.
#
# WHY THE GEANT4 SIDE COMES FROM THE MAIN CHECKOUT AND THAT IS NOT V45
#
# `ref\B1build` is gitignored, so a worktree has none, and building the real Geant4's example B1
# takes minutes. It is also the one binary in this comparison with NO dependence on this port's
# source: it links Geant4 and B1's own example sources and nothing from src/. So using
# D:\g4gpu\ref\B1build is safe in a way that using D:\g4gpu\out\transport_run.obj would not be -
# docs/RISK.md V45 is about the PORT's engine being taken from main, which discards the change
# under test. The port's side here is the worktree's own exampleB1.exe and the script refuses to
# run without it.
#
# **The event count is part of the measurement, not a knob for how long you want to wait.** B1's
# printed rms is the standard error and scales as 1/sqrt(N): at 500,000 events a 200 MeV pion row
# is +/-0.11%, and at 2,000 it is +/-1.7% - which can neither confirm nor exclude a 3% effect.
# docs/RISK.md V44 is the half day that cost. 500,000 is the default for that reason.
param(
  [int]$Events = 500000,
  [string]$Species = "",
  [switch]$SkipGeant4
)

$ErrorActionPreference = "Stop"

# Serial, forced, for the reason tools\compare_b1_beams.ps1 gives at length: the install is a
# multithreaded build and G4RunManagerFactory would otherwise hand B1 a G4TaskRunManager, which
# makes every worker print its own "End of Local Run" block holding its slice of the events.
# Reading the first dose line in that output reads one thread's share.
$env:G4FORCE_RUN_MANAGER_TYPE = "Serial"

$here = $PSScriptRoot
$root = Split-Path -Parent (Split-Path -Parent $here)
$tmp = Join-Path $env:TEMP ("g4gpu_stage1_" + $PID)
New-Item -ItemType Directory -Force -Path $tmp | Out-Null

$portExe = Join-Path $root "examples\B1\exampleB1.exe"
$refExe = "D:\g4gpu\ref\B1build\Release\exampleB1.exe"

if (-not (Test-Path -LiteralPath $portExe)) {
  Write-Output "FATAL: no $portExe"
  Write-Output "       Build it with examples\B1\build.bat IN THIS WORKTREE. A stage-1 table"
  Write-Output "       taken from another checkout's binary is main's physics under this"
  Write-Output "       branch's name - docs/RISK.md V45."
  exit 1
}

$all = @("proton", "alpha", "muon_plus", "muon_minus", "pion_plus", "pion_minus",
         "kaon_plus", "kaon_minus", "neutron")
$want = if ($Species -ne "") { $Species -split "," } else { $all }

# B1 prints "Cumulated dose per run, in scoring volume : <v> <unit> rms = <v> <unit>", and
# G4BestUnit picks the unit by magnitude - a proton run is in nanoGy where a gamma run is in
# picoGy - so the unit is parsed and not assumed. Getting this wrong is compare_runs.ps1's
# lesson repeated.
$doseUnit = @{ "picoGy" = 1e-12; "nanoGy" = 1e-9; "microGy" = 1e-6; "milliGy" = 1e-3;
               "Gy" = 1.0; "kGy" = 1e3 }

function Get-Dose([string[]]$out) {
  # A run that produced NOTHING is a null array here, and indexing one throws from inside this
  # function rather than reporting at the call site that a run failed. Guarded, because that is
  # exactly how a missing Geant4 environment presented itself: "Cannot index into a null array"
  # at line 79 of this file, with no mention of the exe that exited 53.
  if ($null -eq $out -or $out.Count -eq 0) { return $null }
  # From the GLOBAL run block and nothing else. See the note about threading above.
  $start = 0
  for ($i = 0; $i -lt $out.Count; $i++) {
    if ($out[$i] -match 'End of Global Run') { $start = $i }
  }
  foreach ($l in $out[$start..($out.Count - 1)]) {
    if ($l -match 'Cumulated dose per run, in scoring volume :\s*([0-9.eE+-]+)\s+(\S+)\s+rms\s*=\s*([0-9.eE+-]+)\s+(\S+)') {
      $d = [double]$Matches[1]; $du = $Matches[2]
      $r = [double]$Matches[3]; $ru = $Matches[4]
      if (-not $doseUnit.ContainsKey($du)) { throw "unrecognised dose unit '$du'" }
      if (-not $doseUnit.ContainsKey($ru)) { throw "unrecognised rms unit '$ru'" }
      return @{ Dose = $d * $doseUnit[$du]; Rms = $r * $doseUnit[$ru] }
    }
  }
  return $null
}

# A macro with the event count substituted, so -Events is honoured without editing the checked-in
# file. The rest of the macro - every /process/inactivate line and the process dump - is copied
# verbatim, because that is the part the number has to be recorded against.
function New-Macro([string]$src, [int]$n) {
  $lines = Get-Content -LiteralPath $src | ForEach-Object {
    if ($_ -match '^\s*/run/beamOn\s') { "/run/beamOn $n" }
    elseif ($_ -match '^\s*/run/printProgress\s') { "/run/printProgress " + [Math]::Max(1, [int]($n / 5)) }
    else { $_ }
  }
  $dst = Join-Path $tmp ((Split-Path -Leaf $src) -replace '\.mac$', "_$n.mac")
  Set-Content -LiteralPath $dst -Value $lines -Encoding ASCII
  return $dst
}

function Invoke-Run([string]$exe, [string]$mac, [string]$cwd) {
  Push-Location $cwd
  try {
    $out = & $exe $mac 2>&1 | ForEach-Object { "$_" }
  } finally {
    Pop-Location
  }
  return $out
}

# THE GEANT4 SIDE GOES THROUGH runb1.bat AND NOT THROUGH THE EXE, and the difference is not
# cosmetic: the install needs eleven G4*DATA environment variables and its DLLs on PATH, and
# `ref\run\runb1.bat` is where this repository keeps them. Calling the exe directly - which this
# script did until the numbers were first taken - exits with 53 before printing a line, and the
# only symptom upstream is a null output array and "Cannot index into a null array" from inside
# the dose parser. A second copy of the variable list here would be a second thing to keep in
# step with the install; one launcher is the point.
function Invoke-G4([string]$mac) {
  return & cmd /c "`"$root\ref\run\runb1.bat`" `"$mac`"" 2>&1 | ForEach-Object { "$_" }
}

$rows = @()
$dumps = @{}

foreach ($s in $want) {
  Write-Output "=== $s ==="
  $portMac = New-Macro (Join-Path $here "stage1_${s}_port.mac") $Events
  $portOut = Invoke-Run $portExe $portMac $root
  $port = Get-Dose $portOut
  if ($null -eq $port) {
    Write-Output "  port run produced no dose line; last lines:"
    $portOut | Select-Object -Last 10 | ForEach-Object { "    $_" }
    continue
  }

  $g4 = $null
  $g4ne = $null
  if (-not $SkipGeant4) {
    if (-not (Test-Path -LiteralPath $refExe)) {
      Write-Output "  no $refExe - Geant4 columns skipped"
    } else {
      $g4Mac = New-Macro (Join-Path $here "stage1_${s}.mac") $Events
      $g4Out = Invoke-G4 $g4Mac
      $g4 = Get-Dose $g4Out
      if ($null -eq $g4) {
        Write-Output "  Geant4 stage-1 run produced no dose line; last lines:"
        $g4Out | Select-Object -Last 10 | ForEach-Object { "    $_" }
      }
      # The process dump, kept so the stage a number was measured at is recorded by what RAN.
      $dumps[$s] = ($g4Out | Select-String -Pattern 'Active|InActive' | ForEach-Object { $_.Line })
      $neSrc = Join-Path $here "stage1_${s}_noelastic.mac"
      if (Test-Path -LiteralPath $neSrc) {
        $neMac = New-Macro $neSrc $Events
        $neOut = Invoke-G4 $neMac
        $g4ne = Get-Dose $neOut
        if ($null -eq $g4ne) {
          Write-Output "  Geant4 no-elastic run produced no dose line; last lines:"
          $neOut | Select-Object -Last 10 | ForEach-Object { "    $_" }
        }
      }
    }
  }
  $rows += @{ Name = $s; Port = $port; G4 = $g4; G4NoEl = $g4ne }
}

function Fmt($v) { if ($null -eq $v) { "-" } else { "{0:N4}" -f ($v * 1e9) } }

Write-Output ""
Write-Output "STAGE 1, example B1, $Events events per run per side, doses in nGy"
Write-Output ""
Write-Output ("{0,-11} {1,20} {2,20} {3,9} {4,7} {5,20} {6,9} {7,7}" -f `
  "species", "port", "G4 no elastic", "diff", "sigma", "G4 stage 1", "diff", "sigma")
foreach ($r in $rows) {
  $p = $r.Port
  $line = "{0,-11} {1,11} +/- {2,-6}" -f $r.Name, (Fmt $p.Dose), (Fmt $p.Rms)
  foreach ($col in @($r.G4NoEl, $r.G4)) {
    if ($null -eq $col) {
      $line += " {0,20} {1,9} {2,7}" -f "-", "-", "-"
    } else {
      $d = $col.Dose
      $rel = if ($d -ne 0) { ($p.Dose - $d) / $d * 100 } else { 0 }
      $sd = [Math]::Sqrt($p.Rms * $p.Rms + $col.Rms * $col.Rms)
      $sig = if ($sd -gt 0) { [Math]::Abs($p.Dose - $d) / $sd } else { 0 }
      $line += " {0,11} +/- {1,-6} {2,8}% {3,7}" -f (Fmt $d), (Fmt $col.Rms),
               ("{0:N2}" -f $rel), ("{0:N1}" -f $sig)
    }
  }
  Write-Output $line
}

Write-Output ""
Write-Output "THE GEANT4 PROCESS DUMP OF THE STAGE, per species - what ran, not what was asked"
foreach ($s in $dumps.Keys) {
  Write-Output ""
  Write-Output "  $s"
  $dumps[$s] | ForEach-Object { "    $_" }
}
