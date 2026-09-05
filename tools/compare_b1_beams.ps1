# Example B1 with a proton and an alpha beam, this port against a real Geant4 11.1.1 build of
# the same example, on dose and on runtime.
#
#   tools\compare_b1_beams.ps1 [-Events 10000]
#
# Six runs. For each of gamma (B1's own beam, as a control), proton and alpha:
#
#   port          examples\B1\exampleB1.exe            - EM physics only, on the GPU
#   Geant4 QBBC   ref\B1build\Release\exampleB1.exe    - B1 exactly as it ships
#   Geant4 EM     the same, with QBBC's hadronic processes inactivated
#
# The third column is the one the port should be compared against, and the gap between the
# second and third is what the port's missing hadronic physics is worth for this beam. Both
# are reported because either alone would mislead: comparing against QBBC blames the transport
# for a missing process, and comparing only against EM-only hides that the process exists.
#
# **Geant4 runs serial**, forced with G4FORCE_RUN_MANAGER_TYPE=Serial. The install is a
# multithreaded build and G4RunManagerFactory would otherwise hand B1 a G4TaskRunManager that
# uses every core, which makes the timing column a comparison between one GPU and twenty CPU
# cores rather than between two transports. Serial is the like-for-like baseline; multiply by
# the core count for an estimate of what a threaded Geant4 would do.
#
# It also removes a trap. In multithreaded mode every worker calls EndOfRunAction and prints
# its own "End of Local Run" block holding its slice of the events, and only the master's
# "End of Global Run" carries the merged accumulables. Reading the first dose line in that
# output reads one thread's share - which is how the gamma control first came out twenty times
# low, with twenty workers.
#
# Runtime is wall clock for the whole process, initialisation included, because that is what a
# user waits for. The port also reports its own transport time separately - the GPU kernel time
# without table loading - and both are printed, since for a short run the port is dominated by
# reading G4EMLOW off disk and the ratio of those two numbers is the interesting one.
# **The event count is part of the measurement, not a knob for how long you want to wait.**
# The port transports a batch at a time and its batch holds about a million events, so below
# that the GPU is not full: at 200,000 gammas it runs at 1.26e6 events/s and at 2,000,000 it
# runs at 2.7e6, because the second is two saturated batches and the first is one third-empty
# one. Geant4 has no equivalent effect - one core, one event at a time, the same rate at any
# count - so a speed ratio taken at a small count understates the port by about a factor of
# two, and it is the port that changes, not the reference.
#
# Use at least 2,000,000 for anything quoted as a speedup. The dose comparison is fine at any
# count; only the timing needs the batch full.
param(
  [int]$Events = 10000
)

$ErrorActionPreference = "Stop"

# Set here rather than inside the cmd line that launches Geant4. A child process inherits this
# process's environment, so one assignment covers every run; folding it into
# `cmd /c "set VAR=x&& runb1.bat ..."` looks equivalent and is not - PowerShell hands cmd a
# single already-quoted string and cmd splits it somewhere in the middle of the variable name.
# That failed with 'RCE_RUN_MANAGER_TYPE' is not recognized, after a shorter run had already
# printed plausible doses, because the doses were right for an entirely different reason.
$env:G4FORCE_RUN_MANAGER_TYPE = "Serial"
$root = Split-Path -Parent $PSScriptRoot
$tmp  = Join-Path $env:TEMP "g4gpu_b1beams"
New-Item -ItemType Directory -Force -Path $tmp | Out-Null

# Beam definitions. The alpha energy is 210 MeV per nucleon so that its range matches the
# proton's: both stop in the same place inside the scoring volume, so the two runs differ by
# the projectile and not by where it went.
$beams = @(
  @{ Name = "gamma";  Particle = "gamma";  Energy = 6;   Unit = "MeV"; Inactivate = @() },
  @{ Name = "proton"; Particle = "proton"; Energy = 210; Unit = "MeV";
     Inactivate = @("hadElastic", "protonInelastic") },
  @{ Name = "alpha";  Particle = "alpha";  Energy = 840; Unit = "MeV";
     Inactivate = @("hadElastic", "alphaInelastic", "ionElastic", "ionInelastic") }
)

function New-Macro([hashtable]$b, [bool]$emOnly, [int]$n) {
  # /run/verbose 1, not 0. G4RunManager only starts its timer, and only prints the
  # "Run Summary ... User= Real= Sys=" block, when verboseLevel > 0 - so at verbose 0 there is
  # no transport time to read at all. It costs a few extra lines of output and nothing else.
  $lines = @("/run/initialize", "/control/verbose 0", "/run/verbose 1", "")
  if ($emOnly) {
    foreach ($p in $b.Inactivate) { $lines += "/process/inactivate $p" }
    $lines += ""
  }
  $lines += "/gun/particle $($b.Particle)"
  $lines += "/gun/energy $($b.Energy) $($b.Unit)"
  $lines += ""
  $lines += "/run/beamOn $n"
  $suffix = if ($emOnly) { "_em" } else { "" }
  $path = Join-Path $tmp "$($b.Name)$suffix.mac"
  Set-Content -Path $path -Value $lines -Encoding ASCII
  return $path
}

# B1 prints "Cumulated dose per run, in scoring volume : <v> <unit> rms = <v> <unit>", and
# G4BestUnit picks the unit by magnitude - a proton run is in nanoGy where the gamma run is in
# picoGy - so the unit has to be parsed, not assumed. Getting this wrong is docs/RISK.md's
# compare_runs.ps1 lesson repeated.
$doseUnit = @{ "picoGy" = 1e-12; "nanoGy" = 1e-9; "microGy" = 1e-6; "milliGy" = 1e-3;
               "Gy" = 1.0; "kGy" = 1e3 }

# Take the dose from the *global* run and nothing else.
#
# The Geant4 build of B1 is multithreaded. Every worker thread calls EndOfRunAction and prints
# its own "End of Local Run" block with its own share of the events, and only the master's
# "End of Global Run" carries the merged accumulables. Reading the first dose line in the
# output therefore reads one thread's slice: with 20 workers and 2000 events that is 100
# events, and the gamma control came out 20 times low - which is how this was found, and why
# the control is in the table at all.
#
# The port is single-process and prints only the global block, so this finds the same line in
# both.
function Get-Dose([string[]]$out) {
  $start = 0
  for ($i = 0; $i -lt $out.Count; $i++) {
    if ($out[$i] -match 'End of Global Run') { $start = $i }
  }
  foreach ($l in $out[$start..($out.Count-1)]) {
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

# The port's own event-loop time: host wall clock over generating every primary and running
# every batch. Deliberately not the "time ... ms" line below it, which is CUDA-event time
# around the batch loop alone and excludes primary generation - Geant4's number does not.
function Get-PortLoopMs([string[]]$out) {
  foreach ($l in $out) {
    if ($l -match '^event loop\s+([0-9.eE+-]+)\s+ms for') { return [double]$Matches[1] }
  }
  return [double]::NaN
}

# The port's GPU time alone, for the last column.
function Get-PortGpuMs([string[]]$out) {
  foreach ($l in $out) {
    if ($l -match '^time\s+([0-9.eE+-]+)\s+ms for') { return [double]$Matches[1] }
  }
  return [double]::NaN
}

# Geant4's own event-loop time, from the run manager's G4Timer.
#
#   Run Summary
#     Number of events processed : 100000
#     User=12.34s Real=12.56s Sys=0.01s
#
# G4RunManager starts that timer in InitializeEventLoop and stops it in TerminateEventLoop, so
# it brackets exactly the event loop: physics-table building happens in RunInitialization,
# before it starts, and is excluded. That makes it the direct counterpart of the port's event
# loop, and it replaces the two-point wall-clock subtraction this script did first - which was
# an estimate of the same quantity with the start-up scatter of two runs left in it.
#
# Real, not User: wall clock is what the port is measured on, and on a serial run with nothing
# else on the machine the two are the same number anyway.
function Get-G4LoopMs([string[]]$out) {
  $ms = [double]::NaN
  foreach ($l in $out) {
    if ($l -match 'User=([0-9.eE+-]+)s\s+Real=([0-9.eE+-]+)s') { $ms = 1000 * [double]$Matches[2] }
  }
  return $ms
}

# Runs one configuration and returns its wall clock in ms plus whatever it printed.
function Invoke-Run([scriptblock]$launch) {
  $sw = [Diagnostics.Stopwatch]::StartNew()
  $out = & $launch
  $sw.Stop()
  return @{ Ms = $sw.Elapsed.TotalMilliseconds; Out = $out }
}

$results = @()
foreach ($b in $beams) {
  # ---- the port, at both event counts
  $portRun = {
    param($m)
    & (Join-Path $root "examples\B1\exampleB1.exe") $m 2>&1 | ForEach-Object { "$_" }
  }
  $mac = New-Macro $b $false $Events
  $r = Invoke-Run { & $portRun $mac }
  $out = $r.Out
  $d = Get-Dose $out
  if ($null -eq $d) { Write-Host "no dose line from the port for $($b.Name)"; $out | Select-Object -Last 20; exit 1 }
  $results += [pscustomobject]@{ Beam = $b.Name; Code = "port"; Dose = $d.Dose; Rms = $d.Rms
                                 WallMs = $r.Ms
                                 LoopMs = (Get-PortLoopMs $out); GpuMs = (Get-PortGpuMs $out) }

  # ---- real Geant4, twice
  foreach ($em in @($false, $true)) {
    if ($em -and $b.Inactivate.Count -eq 0) { continue }   # nothing to inactivate for a gamma
    $g4Run = {
      param($m)
      & cmd /c "`"$root\ref\run\runb1.bat`" `"$m`"" 2>&1 | ForEach-Object { "$_" }
    }
    $mac = New-Macro $b $em $Events
    $r = Invoke-Run { & $g4Run $mac }
    $out = $r.Out
    $d = Get-Dose $out
    if ($null -eq $d) { Write-Host "no dose line from Geant4 for $($b.Name)"; $out | Select-Object -Last 20; exit 1 }
    # Serial, confirmed rather than assumed: a worker thread announcing itself means the
    # environment variable did not take, and every timing number below would then be a whole
    # machine against one GPU. Cheap to check and it has already been wrong once.
    foreach ($l in $out) {
      if ($l -match 'starts on worker thread') {
        Write-Host "FAIL: Geant4 ran multithreaded despite G4FORCE_RUN_MANAGER_TYPE=Serial"
        exit 1
      }
    }
    $label = if ($em) { "G4 EM-only" } else { "G4 QBBC" }
    $results += [pscustomobject]@{ Beam = $b.Name; Code = $label; Dose = $d.Dose; Rms = $d.Rms
                                   WallMs = $r.Ms
                                   LoopMs = (Get-G4LoopMs $out); GpuMs = [double]::NaN }
  }
}

# ---------------------------------------------------------------- report
Write-Host ""
Write-Host ("Example B1, {0} events per run. Dose in the Shape2 bone trapezoid." -f $Events)
Write-Host  "Geant4 forced to serial (G4FORCE_RUN_MANAGER_TYPE=Serial): one CPU thread against"
Write-Host  "one GPU. The install is a multithreaded build, so a threaded Geant4 on this machine"
Write-Host  "would be roughly the core count faster than the times below."
Write-Host ""
# The per-event rate with start-up removed, and the two codes get it two different ways
# because only one of them can measure itself.
#
#   port    its own reported transport time, which is CUDA-event timing around the stepping
#           kernels and excludes table loading by construction. Subtracting two wall clocks
#           would not work here: at any event count this comparison runs at, the port's
#           transport is a fraction of its ~0.5 s of start-up, so the difference of two wall
#           times is mostly the run-to-run scatter in start-up and comes out negative as often
#           as not.
#   Geant4  the two-point subtraction, since it reports no equivalent. Sound here because for
#           a proton or an alpha its transport dwarfs its start-up; flagged when it does not.
foreach ($r in $results) {
  if ([double]::IsNaN($r.LoopMs)) {
    Write-Host ("FAIL: no event-loop time from {0} for the {1} beam" -f $r.Code, $r.Beam)
    exit 1
  }
}
Write-Host ("{0,-8} {1,-11} {2,16} {3,14} {4,9} {5,10} {6,9} {7,11}" -f
            "beam","code","dose (pGy)","rms (pGy)","wall (s)","loop (s)","gpu (s)","ev/s loop")
foreach ($r in $results) {
  $g = if ([double]::IsNaN($r.GpuMs)) { "-" } else { "{0:n3}" -f ($r.GpuMs/1000) }
  $rate = $Events / ($r.LoopMs / 1000)
  Write-Host ("{0,-8} {1,-11} {2,16:n2} {3,14:n2} {4,9:n2} {5,10:n3} {6,9} {7,11:n0}" -f
              $r.Beam, $r.Code, ($r.Dose*1e12), ($r.Rms*1e12), ($r.WallMs/1000),
              ($r.LoopMs/1000), $g, $rate)
}

Write-Host ""
Write-Host "dose, port vs each Geant4 configuration:"
Write-Host ("{0,-8} {1,-13} {2,10} {3,10} {4,10}" -f "beam","against","ratio","diff %","sigma")
foreach ($b in $beams) {
  $p = $results | Where-Object { $_.Beam -eq $b.Name -and $_.Code -eq "port" }
  foreach ($c in @("G4 QBBC", "G4 EM-only")) {
    $g = $results | Where-Object { $_.Beam -eq $b.Name -and $_.Code -eq $c }
    if ($null -eq $g) { continue }
    $comb = [Math]::Sqrt($p.Rms*$p.Rms + $g.Rms*$g.Rms)
    $sig = if ($comb -gt 0) { [Math]::Abs($p.Dose - $g.Dose) / $comb } else { 0 }
    Write-Host ("{0,-8} {1,-13} {2,10:n5} {3,10:n2} {4,10:n1}" -f
                $b.Name, $c, ($p.Dose/$g.Dose), (100*($p.Dose/$g.Dose - 1)), $sig)
  }
}

Write-Host ""
Write-Host "speed, port vs serial Geant4:"
Write-Host ("  {0,-8} {1,10} {2,11} {3,10}  {4}" -f "beam","wall","event loop","gpu only","against")
foreach ($b in $beams) {
  $p = $results | Where-Object { $_.Beam -eq $b.Name -and $_.Code -eq "port" }
  foreach ($c in @("G4 QBBC", "G4 EM-only")) {
    $g = $results | Where-Object { $_.Beam -eq $b.Name -and $_.Code -eq $c }
    if ($null -eq $g) { continue }
    Write-Host ("  {0,-8} {1,9:n1}x {2,10:n1}x {3,9:n1}x  vs {4}" -f
                $b.Name, ($g.WallMs/$p.WallMs), ($g.LoopMs/$p.LoopMs), ($g.LoopMs/$p.GpuMs), $c)
  }
}
Write-Host ""
Write-Host "  wall        total process time, start-up included - what you actually wait for."
Write-Host "              At these event counts the port is dominated by loading G4EMLOW and"
Write-Host "              the Seltzer-Berger tables, so this understates the transport."
Write-Host "  event loop  the like-for-like number, and each side measures itself: the port's"
Write-Host "              host clock over primary generation plus every batch, and Geant4's own"
Write-Host "              G4Timer from InitializeEventLoop to TerminateEventLoop. Building the"
Write-Host "              physics tables is outside both."
Write-Host "  gpu only    the port's CUDA-event time around the stepping kernels alone. NOT"
Write-Host "              like-for-like - it leaves out the primary generation that Geant4's"
Write-Host "              number includes - and is here to show how much of the port's event"
Write-Host "              loop is still host-side."
Write-Host ""
