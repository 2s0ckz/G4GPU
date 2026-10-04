# The rows docs/RISK.md V219 and P22 measure - B1 with a neutron, deuteron, alpha or He3 beam -
# on either side, one row at a time, with the settings each row was measured with written here
# rather than remembered.
#
#   powershell -File ref\b1ngp\rows.ps1 -Side port [-Rows "neutron_100,neutron_1000"]
#              [-PortExe <exe>] [-NeutronOrder g4111|reordered] [-Out <prefix>]
#   powershell -File ref\b1ngp\rows.ps1 -Side g4 -Patch 0|1 [-Rows ...] [-Out <prefix>]
#
# THE PORT SIDE is examples\B1\exampleB1.exe (or -PortExe) at the B1 sweep's like-for-like
# settings: `G4GPU_HADRONIC_STAGE=final` (G4NeutronGeneralProcess as QBBC builds it - the engine's
# default stage is kStage1, docs/RISK.md V194), `G4GPU_LIVE_PER_EVENT=32` (tools\b1_sweep.ps1's
# hadron pool; the pool does not move the dose, build_all.bat's pool check asserts it), and
# `G4GPU_ION_INELASTIC` cleared. `-NeutronOrder` sets `G4GPU_NEUTRON_GP_ORDER`, the P22 switch
# between 11.1.1's charging of a boundary step (the default) and the reordered one; empty leaves
# the engine's default, which is what a binary from before P22 needs.
#
# THE GEANT4 SIDE is ref\b1ngp\run.bat: Geant4's own B1 on QBBC's constructor list, with
# `G4NeutronGeneralProcess` as 11.1.1 ships it (-Patch 0, which reads ref/B1build's exampleB1 to
# every digit) or with each step charged at the mean free path it was taken with (-Patch 1).
# Build it first with ref\b1ngp\build.bat.
#
# THE MACROS ARE THE ROWS' OWN. neutron_100 is P15's (`ref/b1hadron/p15_neutron.mac`, which
# inactivates `ionElastic` alone), so its Geant4 column is the 224.945 / 133.458 nGy pair V219
# measured; every other row inactivates the B1 sweep's `$emOnly` list for a hadron beam -
# `hBrems`, `hPairProd`, `ionElastic`, the processes the port still lacks - as
# ref/b1ngp/neutron_1000.mac and alpha_4000.mac do. A different list is a different random
# stream on the Geant4 side, which is why the list is per row and not per run.
param(
  [string]$Rows = "",
  [ValidateSet("port", "g4")] [string]$Side = "port",
  [ValidateSet("0", "1")] [string]$Patch = "0",
  [string]$PortExe = "",
  [string]$NeutronOrder = "",
  [string]$Out = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$tmp = Join-Path $env:TEMP ("g4gpu_b1ngp_rows_" + $PID)
New-Item -ItemType Directory -Force -Path $tmp | Out-Null
if ($PortExe -eq "") { $PortExe = Join-Path $root "examples\B1\exampleB1.exe" }
$runG4 = Join-Path $PSScriptRoot "run.bat"
if ($Out -eq "") { $Out = Join-Path $root ("out\b1ngp_rows_" + $Side) }
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Out) | Out-Null

$hadronList = @("hBrems", "hPairProd", "ionElastic")
$table = @(
  @{ Name = "neutron_100";   Particle = "neutron";  Energy = 100;  Events = 500000; G4Off = @("ionElastic") },
  @{ Name = "neutron_1000";  Particle = "neutron";  Energy = 1000; Events = 100000; G4Off = $hadronList },
  @{ Name = "deuteron_2000"; Particle = "deuteron"; Energy = 2000; Events = 100000; G4Off = $hadronList },
  @{ Name = "alpha_4000";    Particle = "alpha";    Energy = 4000; Events = 100000; G4Off = $hadronList },
  @{ Name = "He3_3000";      Particle = "He3";      Energy = 3000; Events = 100000; G4Off = $hadronList },
  @{ Name = "alpha_840";     Particle = "alpha";    Energy = 840;  Events = 300000; G4Off = $hadronList }
)
if ($Rows -ne "") {
  $want = $Rows -split ","
  foreach ($w in $want) {
    if (-not ($table | Where-Object { $_.Name -eq $w })) { Write-Output "FATAL: no row '$w'"; exit 1 }
  }
  $table = @($table | Where-Object { $want -contains $_.Name })
}

function New-Macro([hashtable]$r) {
  $lines = @("/run/initialize", "/control/verbose 0", "/run/verbose 1", "")
  if ($Side -eq "g4") {
    foreach ($p in $r.G4Off) { $lines += "/process/inactivate $p" }
    # The configuration recorded by what ran: the neutron's manager, after the inactivations.
    # Three entries - Transportation, Decay, NeutronGeneralProc - is the general process ON.
    $lines += @("", "/particle/select neutron", "/particle/process/dump", "")
  }
  $lines += "/gun/particle $($r.Particle)"
  $lines += "/gun/energy $($r.Energy) MeV"
  $lines += "/run/printProgress $([math]::Max(1000, [int]($r.Events / 10)))"
  $lines += "/run/beamOn $($r.Events)"
  $path = Join-Path $tmp "$($r.Name)_$Side.mac"
  Set-Content -Path $path -Value $lines -Encoding ASCII
  return $path
}

$doseUnit = @{ "picoGy" = 1e-12; "nanoGy" = 1e-9; "microGy" = 1e-6; "milliGy" = 1e-3; "Gy" = 1.0 }
function Get-Dose([string[]]$text) {
  foreach ($l in $text) {
    if ($l -match 'Cumulated dose per run, in scoring volume :\s*([0-9.eE+-]+)\s+(\S+)\s+rms\s*=\s*([0-9.eE+-]+)\s+(\S+)') {
      return @{ Dose = [double]$Matches[1] * $doseUnit[$Matches[2]]; Rms = [double]$Matches[3] * $doseUnit[$Matches[4]] }
    }
  }
  return $null
}

$md = @("| row | events | side | dose (nGy) | rms (nGy) | loop | wall |", "|---|--:|---|--:|--:|--:|--:|")
foreach ($r in $table) {
  $mac = New-Macro $r
  $label = if ($Side -eq "port") { "port" + $(if ($NeutronOrder -ne "") { " ($NeutronOrder)" } else { "" }) } else { "b1ngp PATCH=$Patch" }
  Write-Output ("=== {0}: {1:N0} events, {2}" -f $r.Name, $r.Events, $label)
  $sw = [Diagnostics.Stopwatch]::StartNew()
  if ($Side -eq "port") {
    $env:G4GPU_HADRONIC_STAGE = "final"
    $env:G4GPU_LIVE_PER_EVENT = "32"
    Remove-Item Env:\G4GPU_ION_INELASTIC -ErrorAction SilentlyContinue
    if ($NeutronOrder -ne "") { $env:G4GPU_NEUTRON_GP_ORDER = $NeutronOrder }
    else { Remove-Item Env:\G4GPU_NEUTRON_GP_ORDER -ErrorAction SilentlyContinue }
    $lines = & $PortExe $mac 2>&1 | ForEach-Object { "$_" }
    Remove-Item Env:\G4GPU_HADRONIC_STAGE, Env:\G4GPU_LIVE_PER_EVENT -ErrorAction SilentlyContinue
    Remove-Item Env:\G4GPU_NEUTRON_GP_ORDER -ErrorAction SilentlyContinue
  } else {
    $lines = & cmd /c "`"$runG4`" `"$mac`" $Patch" 2>&1 | ForEach-Object { "$_" }
  }
  $sw.Stop()
  $log = "$Out.$($r.Name).log"
  $lines | Set-Content -Path $log -Encoding UTF8
  $d = Get-Dose $lines
  if ($null -eq $d) { Write-Output "  no dose line; last lines:"; $lines | Select-Object -Last 15; exit 1 }
  $loop = ""
  foreach ($l in $lines) {
    if ($l -match '^event loop\s+([0-9.eE+-]+)\s+ms for') { $loop = ("{0:N0} ms" -f [double]$Matches[1]) }
    if ($l -match 'User=([0-9.eE+-]+)s\s+Real=([0-9.eE+-]+)s') { $loop = ("{0:N0} ms" -f (1000 * [double]$Matches[2])) }
  }
  # What a reader needs beside the number: the neutron-order line the port prints, the ledger,
  # the interaction counts, and on the Geant4 side the instrument's own counters.
  foreach ($l in $lines) {
    if ($l -match 'neutron general process|HADRONIC INTERACTIONS WITH NO FINAL STATE|^interactions:|^b1ngp:|^\s+\d+\s+[0-9.eE+-]+\s+MeV\s+\S') {
      Write-Output ("  " + $l.Trim())
    }
  }
  Write-Output ("  dose {0:F3} +/- {1:F3} nGy   loop {2}   wall {3:N0} ms   log {4}" -f ($d.Dose * 1e9), ($d.Rms * 1e9), $loop, $sw.Elapsed.TotalMilliseconds, $log)
  $md += ("| {0} | {1:N0} | {2} | {3:F3} | {4:F3} | {5} | {6:N0} ms |" -f $r.Name, $r.Events, $label, ($d.Dose * 1e9), ($d.Rms * 1e9), $loop, $sw.Elapsed.TotalMilliseconds)
}
$md | Set-Content -Path "$Out.md" -Encoding UTF8
Write-Output ""
$md | ForEach-Object { Write-Output $_ }
Write-Output "written: $Out.md and one .log per row"
