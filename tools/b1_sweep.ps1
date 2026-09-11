# A twelve-beam sweep of example B1: this port against Geant4 11.1.1, four species at three
# energies each, dose in the Shape2 bone trapezoid and the time each side took.
#
#   tools\b1_sweep.ps1 [-Only "gamma_6,proton_210"] [-Scale 1.0] [-SkipQBBC] [-Out out\b1_sweep]
#
# Three runs per beam, as tools\compare_b1_beams.ps1 does, and for the same reason:
#
#   port          this port's examples\B1\exampleB1.exe
#   G4 EM-only    the real Geant4 with ONLY what the port still lacks after Phase 2 inactivated:
#                 the inelastic processes, the hadron radiative processes hBrems/hPairProd (left
#                 at P by P1), the photo-/electro-nuclear ones, and ionElastic/ionInelastic for
#                 the recoil nuclei (transported since P8c, their own hadronic processes not
#                 wired). Elastic, capture, decay and single Coulomb scattering stay active on
#                 both sides. This is the LIKE-FOR-LIKE
#                 column; agreement here is the claim
#   G4 QBBC       the real Geant4 as QBBC ships; its distance from EM-only is what the missing
#                 physics is worth for that beam, measured rather than argued
#
# WHAT IS INACTIVATED, per species, is read off ref\oracle\species_processes.csv - the dump of
# the constructed QBBC's own process managers - and not off the source, because
# /process/inactivate silently ignores a name the species does not carry (docs\RISK.md V43).
# The photon needs one more step: QBBC folds phot/compt/conv/Rayl/photonNuclear into one
# G4GammaGeneralProcess, inside which photonNuclear cannot be inactivated, so the EM-only photon
# run switches the general process off BEFORE /run/initialize and then inactivates
# photonNuclear. That changes the interpolation grid the reference reads its cross sections
# from, not the physics; the QBBC column keeps the general process, so the two Geant4 photon
# columns bracket that too.
#
# ENERGIES. B1's gun sits on the -z face of a 30 cm water envelope and the scoring volume is
# the 6 cm bone trapezoid centred at z = +7 cm, so a charged primary has to cross about 19 cm of
# water and tissue before it can deposit anything there. That is why the lowest proton energy
# is 210 MeV (it stops inside the trapezoid - RESULT.md's headline - so its row is a Bragg-peak
# dose, the most range-sensitive kind), the lowest alpha 840 MeV (the same range), and the
# lowest electron 20 MeV, whose own range is 10 cm and whose row is therefore the dose its
# bremsstrahlung photons carry forward. The photon rows have no such constraint.
#
# EVENTS. B1's printed rms is the standard error and scales as 1/sqrt(N); the counts below are
# chosen so that every row is at or under 0.2%, and -Scale multiplies all of them (0.1 for a
# smoke test). The count is part of the measurement (docs\RISK.md V44 is the half day that
# taught it), so the table prints it.
#
# TIMING. Each side measures its own event loop: the port's host clock over primary generation
# plus every batch ("event loop ... ms"), Geant4's G4Timer from InitializeEventLoop to
# TerminateEventLoop (the "User= Real=" line, printed only at /run/verbose 1). Wall clock is
# the whole process, start-up and table building included. Geant4 is forced serial; the port
# runs on the one GPU. Both are quoted, and neither is a speedup until the port carries the same
# physics - see RESULT.md for why the EM-only column is the one to quote.
param(
  [string]$Only = "",
  [double]$Scale = 1.0,
  [switch]$SkipQBBC,
  [string]$Out = ""
)

$ErrorActionPreference = "Stop"
$env:G4FORCE_RUN_MANAGER_TYPE = "Serial"
$root = Split-Path -Parent $PSScriptRoot
$tmp  = Join-Path $env:TEMP ("g4gpu_b1sweep_" + $PID)
New-Item -ItemType Directory -Force -Path $tmp | Out-Null
if ($Out -eq "") { $Out = Join-Path $root "out\b1_sweep" }
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Out) | Out-Null

$portExe = Join-Path $root "examples\B1\exampleB1.exe"
$runb1   = Join-Path $root "ref\run\runb1.bat"
if (-not (Test-Path -LiteralPath $portExe)) { Write-Output "FATAL: no $portExe"; exit 1 }
if (-not (Test-Path -LiteralPath $runb1))   { Write-Output "FATAL: no $runb1"; exit 1 }

# Per species: what the port does not transport, by the process names QBBC's managers carry.
$emOnly = @{
  "gamma"  = @{ PreInit = @("/process/em/UseGeneralProcess false"); Inactivate = @("photonNuclear", "ionElastic", "ionInelastic") }
  "e-"     = @{ PreInit = @(); Inactivate = @("electronNuclear", "positronNuclear", "ionElastic", "ionInelastic") }
  "proton" = @{ PreInit = @(); Inactivate = @("hBrems", "hPairProd", "protonInelastic", "ionElastic", "ionInelastic") }
  "alpha"  = @{ PreInit = @(); Inactivate = @("alphaInelastic", "ionElastic", "ionInelastic") }
}

$beams = @(
  @{ Name = "gamma_1";     Particle = "gamma";  Energy = 1;    Unit = "MeV"; Events = 2000000 },
  @{ Name = "gamma_6";     Particle = "gamma";  Energy = 6;    Unit = "MeV"; Events = 2000000 },
  @{ Name = "gamma_100";   Particle = "gamma";  Energy = 100;  Unit = "MeV"; Events = 1000000 },
  @{ Name = "e-_20";       Particle = "e-";     Energy = 20;   Unit = "MeV"; Events = 500000 },
  @{ Name = "e-_100";      Particle = "e-";     Energy = 100;  Unit = "MeV"; Events = 300000 },
  @{ Name = "e-_1000";     Particle = "e-";     Energy = 1000; Unit = "MeV"; Events = 100000 },
  @{ Name = "proton_210";  Particle = "proton"; Energy = 210;  Unit = "MeV"; Events = 500000 },
  @{ Name = "proton_400";  Particle = "proton"; Energy = 400;  Unit = "MeV"; Events = 300000 },
  @{ Name = "proton_1000"; Particle = "proton"; Energy = 1000; Unit = "MeV"; Events = 200000 },
  @{ Name = "alpha_840";   Particle = "alpha";  Energy = 840;  Unit = "MeV"; Events = 300000 },
  @{ Name = "alpha_1600";  Particle = "alpha";  Energy = 1600; Unit = "MeV"; Events = 200000 },
  @{ Name = "alpha_4000";  Particle = "alpha";  Energy = 4000; Unit = "MeV"; Events = 100000 }
)
if ($Only -ne "") { $want = $Only -split ","; $beams = @($beams | Where-Object { $want -contains $_.Name }) }

function New-Macro([hashtable]$b, [string]$side, [int]$n) {
  $lines = @()
  if ($side -eq "em") { foreach ($c in $emOnly[$b.Particle].PreInit) { $lines += $c } }
  $lines += @("/run/initialize", "/control/verbose 0", "/run/verbose 1", "")
  if ($side -eq "em") { foreach ($p in $emOnly[$b.Particle].Inactivate) { $lines += "/process/inactivate $p" }; $lines += "" }
  if ($side -ne "port") {
    # The stage recorded by what ran: Active/InActive for every process on the beam's manager.
    $lines += "/particle/select $($b.Particle)"; $lines += "/particle/process/dump"; $lines += ""
  }
  $lines += "/gun/particle $($b.Particle)"
  $lines += "/gun/energy $($b.Energy) $($b.Unit)"
  $lines += "/run/printProgress $([math]::Max(1000, [int]($n / 10)))"
  $lines += "/run/beamOn $n"
  $path = Join-Path $tmp "$($b.Name)_$side.mac"
  Set-Content -Path $path -Value $lines -Encoding ASCII
  return $path
}

$doseUnit = @{ "picoGy" = 1e-12; "nanoGy" = 1e-9; "microGy" = 1e-6; "milliGy" = 1e-3; "Gy" = 1.0; "kGy" = 1e3; "femtoGy" = 1e-15 }
function Get-Dose([string[]]$out) {
  foreach ($l in $out) {
    if ($l -match 'Cumulated dose per run, in scoring volume :\s*([0-9.eE+-]+)\s+(\S+)\s+rms\s*=\s*([0-9.eE+-]+)\s+(\S+)') {
      $d = [double]$Matches[1]; $du = $Matches[2]; $r = [double]$Matches[3]; $ru = $Matches[4]
      if (-not $doseUnit.ContainsKey($du)) { throw "unrecognised dose unit '$du'" }
      if (-not $doseUnit.ContainsKey($ru)) { throw "unrecognised rms unit '$ru'" }
      return @{ Dose = $d * $doseUnit[$du]; Rms = $r * $doseUnit[$ru] }
    }
  }
  return $null
}
function Get-PortLoopMs([string[]]$out) { foreach ($l in $out) { if ($l -match '^event loop\s+([0-9.eE+-]+)\s+ms for') { return [double]$Matches[1] } }; return [double]::NaN }
function Get-PortGpuMs([string[]]$out)  { foreach ($l in $out) { if ($l -match '^time\s+([0-9.eE+-]+)\s+ms for') { return [double]$Matches[1] } }; return [double]::NaN }
function Get-G4LoopMs([string[]]$out)   { $ms = [double]::NaN; foreach ($l in $out) { if ($l -match 'User=([0-9.eE+-]+)s\s+Real=([0-9.eE+-]+)s') { $ms = 1000 * [double]$Matches[2] } }; return $ms }
function Get-ProcessDump([string[]]$out) {
  $keep = @(); $on = $false
  foreach ($l in $out) {
    if ($l -match 'G4ProcessManager: particle\[') { $on = $true }
    if ($on) { $keep += $l; if ($l -match '^\s*$' -and $keep.Count -gt 3) { break } }
  }
  return $keep
}
function Invoke-Run([scriptblock]$launch) {
  $sw = [Diagnostics.Stopwatch]::StartNew(); $out = & $launch; $sw.Stop()
  return @{ Ms = $sw.Elapsed.TotalMilliseconds; Out = $out }
}

$rows = @(); $dumps = @()
foreach ($b in $beams) {
  $n = [int][math]::Round($b.Events * $Scale)
  Write-Output ("=== {0}: {1} events per run" -f $b.Name, $n)
  # port
  $mac = New-Macro $b "port" $n
  $r = Invoke-Run { & $portExe $mac 2>&1 | ForEach-Object { "$_" } }
  $d = Get-Dose $r.Out
  if ($null -eq $d) { Write-Output "  port produced no dose line; last lines:"; $r.Out | Select-Object -Last 15; exit 1 }
  $rows += [pscustomobject]@{ Beam = $b.Name; Particle = $b.Particle; EnergyMeV = $b.Energy; Events = $n; Code = "port"
                              Dose = $d.Dose; Rms = $d.Rms; WallMs = $r.Ms; LoopMs = (Get-PortLoopMs $r.Out); GpuMs = (Get-PortGpuMs $r.Out) }
  Write-Output ("  port        {0,12:E5} Gy +/- {1,9:E2}   loop {2,9:N0} ms   wall {3,9:N0} ms" -f $d.Dose, $d.Rms, (Get-PortLoopMs $r.Out), $r.Ms)
  # Geant4, like-for-like then as shipped
  $sides = @("em"); if (-not $SkipQBBC) { $sides += "qbbc" }
  foreach ($side in $sides) {
    $mac = New-Macro $b $side $n
    $r = Invoke-Run { & cmd /c "`"$runb1`" `"$mac`"" 2>&1 | ForEach-Object { "$_" } }
    $d = Get-Dose $r.Out
    if ($null -eq $d) { Write-Output "  Geant4 ($side) produced no dose line; last lines:"; $r.Out | Select-Object -Last 15; exit 1 }
    foreach ($l in $r.Out) { if ($l -match 'starts on worker thread') { Write-Output "FAIL: Geant4 ran multithreaded"; exit 1 } }
    $label = if ($side -eq "em") { "G4 EM-only" } else { "G4 QBBC" }
    $rows += [pscustomobject]@{ Beam = $b.Name; Particle = $b.Particle; EnergyMeV = $b.Energy; Events = $n; Code = $label
                                Dose = $d.Dose; Rms = $d.Rms; WallMs = $r.Ms; LoopMs = (Get-G4LoopMs $r.Out); GpuMs = [double]::NaN }
    Write-Output ("  {0,-11} {1,12:E5} Gy +/- {2,9:E2}   loop {3,9:N0} ms   wall {4,9:N0} ms" -f $label, $d.Dose, $d.Rms, (Get-G4LoopMs $r.Out), $r.Ms)
    $dumps += ("--- {0} {1}" -f $b.Name, $label); $dumps += (Get-ProcessDump $r.Out)
  }
  $rows | Export-Csv -NoTypeInformation -Path "$Out.csv"
}

# ---- the table
$md = @()
$md += "| beam | events | port (Gy) | G4 EM-only (Gy) | diff | sigma | G4 QBBC (Gy) | QBBC vs EM-only | port loop | G4 EM loop | ratio | port wall | G4 EM wall |"
$md += "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"
foreach ($b in $beams) {
  $p = $rows | Where-Object { $_.Beam -eq $b.Name -and $_.Code -eq "port" }
  $e = $rows | Where-Object { $_.Beam -eq $b.Name -and $_.Code -eq "G4 EM-only" }
  $q = $rows | Where-Object { $_.Beam -eq $b.Name -and $_.Code -eq "G4 QBBC" }
  $diff = ($p.Dose - $e.Dose) / $e.Dose; $sig = ($p.Dose - $e.Dose) / [math]::Sqrt($p.Rms * $p.Rms + $e.Rms * $e.Rms)
  $qcol = if ($q) { ("{0:E4}" -f $q.Dose) } else { "-" }
  $qd = if ($q) { ("{0:+0.00%;-0.00%}" -f (($q.Dose - $e.Dose) / $e.Dose)) } else { "-" }
  $ratio = $e.LoopMs / $p.LoopMs
  $md += ("| {0} | {1:N0} | {2:E4} ± {3:E1} | {4:E4} ± {5:E1} | {6:+0.00%;-0.00%} | {7:0.0} | {8} | {9} | {10:N0} ms | {11:N0} ms | {12:N0}x | {13:N0} ms | {14:N0} ms |" -f `
    $b.Name, $p.Events, $p.Dose, $p.Rms, $e.Dose, $e.Rms, $diff, $sig, $qcol, $qd, $p.LoopMs, $e.LoopMs, $ratio, $p.WallMs, $e.WallMs)
}
$md | Set-Content -Path "$Out.md" -Encoding UTF8
$dumps | Set-Content -Path "$Out.dumps.txt" -Encoding UTF8
Write-Output ""; $md | ForEach-Object { Write-Output $_ }
Write-Output ""; Write-Output "written: $Out.csv, $Out.md, $Out.dumps.txt"
