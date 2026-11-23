# P27, the acceptance sweep: every species the port transports as a primary, 100 MeV to 10 GeV (per
# nucleon for a nucleus), with example B1's scoring trapezoid in water, bone, air, aluminium,
# polyethylene and lead - this port against Geant4 11.1.1's QBBC as it ships, and both sides timed
# at event counts that mean something. docs/RISK.md V336 is the plan this script carries out;
# docs/B1_SWEEP.md's P27 section is what it measured.
#
#   tools\p27_sweep.ps1 -Tier 1 [-Sides port,box,serial] [-Species proton,C12] [-Materials G4_WATER]
#                       [-Energies 1000] [-Plan] [-Out out\p27]
#
# EVERY RUN IS ONE LINE OF out\p27\runs.csv, appended as it ends, and the script is RESUMABLE: a
# row's next run is decided from the runs the file already holds, so a sweep cut short is
# continued by running the same command again, and a run is never repeated to no purpose.
#
# THE MATERIAL is Shape2's - the 6 cm trapezoid B1 scores in, behind 19 cm of the water envelope
# and the tissue cone - set by `/B1/det/material <NIST name>` before /run/initialize on both
# sides (examples/B1's DetectorConstruction; ref/b1mat on Geant4's), the envelope and the world as
# B1 ships them. V336 says why it is the trapezoid and not the envelope.
#
# THE PORT SIDE (the user's timing rule 1, 2026-11-20). A row's event count is doubled until
# events/s stops rising: the row is SATURATED at count N when the rate at 2N is no more than 3%
# above the rate at N, and N is at or above the class floor (photons, e+-, muons 2,000,000;
# hadrons 200,000; every nucleus 50,000). The row starts at its class's saturated count from
# out\p27\saturation.csv (written from the class curves, V336) or at the floor, and keeps
# doubling past saturation while the dose's rms is above 1% and the next run fits the cap. Every
# run is the gate's binary copied to out\p27\bin, CUDA_VISIBLE_DEVICES=1, G4GPU_HADRONIC_STAGE=final
# and EVERY other G4GPU_ variable removed (the performance chain at its defaults: the flat cascade
# and both factories on, the tail pool, one drain stream), the default live-track count, under
# out\gpu.lock taken before the run and deleted after it whatever happened, the host process's
# working set watched and the run killed past 6 GB. The port's numbers are exampleB1's own lines:
# "event loop ... events/s" (the host clock over the whole loop: the rate quoted), "time ...
# (track-steps/s)", the dose and its rms, the "GB left" line, and the ledgers.
#
# THE GEANT4 SIDE (the user's timing rule 2). ref\b1mat\run.bat - Geant4's own B1 on QBBC, the
# command added - in two configurations:
#   box     G4FORCENUMBEROFTHREADS=10, the i9-10850K's ten physical cores: events / Real of the
#           master's G4Timer (InitializeEventLoop to RunTermination, which INCLUDES the workers'
#           start-up, G4TaskRunManager::CreateAndStartWorkers) - for every row;
#   serial  G4FORCE_RUN_MANAGER_TYPE=Serial: events / User - one energy (1 GeV, per nucleon for a
#           nucleus) per species per material.
# Each macro runs the beam TWICE in one process - N events, then N/4 - so the start-up inside the
# first run's timer is measured, not assumed: init = User1 - (N1/N2) User2, and the row stands
# only if init / User1 < 5%; otherwise it is rerun at the count that makes it so. The row is the
# FIRST run (dose, rms, User, Real), its own start-up included. N is the larger of what puts the
# dose's rms at 0.95% (from the port's rms per event, or a pilot's) and what puts the start-up
# under 4% (from a pilot of P events run twice: its User1 - User2 is the start-up, its User2 / P
# the cost of an event), capped so that the run fits the cap.
#
# TWO SIDES NEVER RUN AT ONCE: this script runs one process at a time, and a port run waits for
# the GPU lock while a Geant4 run needs the whole CPU - the port's host thread is one of its cores.
param(
  [int]$Tier = 1,
  [string]$Sides = "port,box,serial",
  [string]$Species = "",
  [string]$Materials = "",
  [string]$Energies = "",
  [switch]$Plan,
  [switch]$Report,
  [string]$Out = "",
  [double]$PortCapMin = 45,      # a port run doubling towards SATURATION may take this long
  [double]$PortRmsCapMin = 5,    # one doubling past saturation for the 1% rms alone, this long
  [double]$CpuCapMin = 10,       # a box row, Real of both runs
  [double]$SerialCapMin = 20,    # a serial row, User of both runs
  [double]$RmsTarget = 0.0095,
  [long]$StartOverride = 0
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
if ($Out -eq "") { $Out = Join-Path $root "out\p27" }
$logDir = Join-Path $Out "logs"; $macDir = Join-Path $Out "macros"
New-Item -ItemType Directory -Force -Path $logDir, $macDir | Out-Null
$csv = Join-Path $Out "runs.csv"
$satFile = Join-Path $Out "saturation.csv"
$portExe = Join-Path $Out "bin\exampleB1.exe"
$g4Run = Join-Path $root "ref\b1mat\run.bat"
$gpuLock = "D:\g4gpu\out\gpu.lock"
$script:SeedLine = ""

# ---------------------------------------------------------------------------------------- matrix
# Name, the gun commands (the same on both sides: G4ParticleGunMessenger's), A (the energy is per
# nucleon for A > 1 and /gun/energy takes the TOTAL kinetic energy on both sides - V274), class.
function S([string]$n, [string[]]$g, [int]$a, [string]$c) { [pscustomobject]@{ Name = $n; Gun = $g; A = $a; Class = $c } }
$core = @(
  (S "gamma" @("/gun/particle gamma") 1 "em"),
  (S "e-" @("/gun/particle e-") 1 "em"),
  (S "e+" @("/gun/particle e+") 1 "em"),
  (S "mu-" @("/gun/particle mu-") 1 "em"),
  (S "mu+" @("/gun/particle mu+") 1 "em"),
  (S "pi+" @("/gun/particle pi+") 1 "hadron"),
  (S "pi-" @("/gun/particle pi-") 1 "hadron"),
  (S "kaon+" @("/gun/particle kaon+") 1 "hadron"),
  (S "kaon-" @("/gun/particle kaon-") 1 "hadron"),
  (S "proton" @("/gun/particle proton") 1 "hadron"),
  (S "anti_proton" @("/gun/particle anti_proton") 1 "hadron"),
  (S "neutron" @("/gun/particle neutron") 1 "hadron"),
  (S "anti_neutron" @("/gun/particle anti_neutron") 1 "hadron"),
  (S "deuteron" @("/gun/particle deuteron") 2 "lightion"),
  (S "triton" @("/gun/particle triton") 3 "lightion"),
  (S "He3" @("/gun/particle He3") 3 "lightion"),
  (S "alpha" @("/gun/particle alpha") 4 "lightion"),
  (S "C12" @("/gun/particle ion", "/gun/ion 6 12") 12 "gcr"),
  (S "O16" @("/gun/particle ion", "/gun/ion 8 16") 16 "gcr"),
  (S "Si28" @("/gun/particle ion", "/gun/ion 14 28") 28 "gcr"),
  (S "Fe56" @("/gun/particle ion", "/gun/ion 26 56") 56 "gcr")
)
# Tier 3: what else the port's gun fires (species_disposition kStepped, P23a/P23c/P31) - the
# strange species, the light anti-nuclei, and the three neutral mesons that decay where they are
# made. Hadron-class counts; the anti-nuclei per nucleon like their partners.
$extra = @()
foreach ($n in @("kaon0L", "kaon0S", "lambda", "sigma+", "sigma-", "sigma0", "xi-", "xi0", "omega-",
                 "anti_lambda", "anti_sigma+", "anti_sigma0", "anti_sigma-", "anti_xi0", "anti_xi-",
                 "anti_omega-", "pi0", "eta", "eta_prime")) { $extra += (S $n @("/gun/particle $n") 1 "hadron") }
$extra += (S "anti_deuteron" @("/gun/particle anti_deuteron") 2 "hadron")
$extra += (S "anti_triton" @("/gun/particle anti_triton") 3 "hadron")
$extra += (S "anti_He3" @("/gun/particle anti_He3") 3 "hadron")
$extra += (S "anti_alpha" @("/gun/particle anti_alpha") 4 "hadron")

$allE = @(100, 300, 1000, 3000, 10000)
$wb = @("G4_WATER", "G4_BONE_COMPACT_ICRU"); $four = @("G4_AIR", "G4_Al", "G4_POLYETHYLENE", "G4_Pb")
function Pick([string[]]$names) { @($core | Where-Object { $names -contains $_.Name }) }
# A group: species, energies (per nucleon, MeV), materials, the serial row's energy, the gun plane.
function G($sp, $en, $ma, $se, $z) { [pscustomobject]@{ Spec = $sp; Ens = $en; Mats = $ma; SerialE = $se; GunZ = $z } }
# TIER 4 (the user, 2026-11-21: G4GPU is for every Geant4 user, medical physics included): the low
# end, with the gun moved by /B1/gun/z to 1 um upstream of the trapezoid's upstream face (Shape2
# spans z = +40 to +100 mm: B1 places it at +7 cm with a 6 cm length), so that the primary enters
# the scoring volume directly - a 10 keV photon or a 10 MeV proton fired from B1's -15 cm never
# reaches it. Its rows are keyed "z_" so that no tier-1 row (same species, energy and material,
# B1's gun plane) is mistaken for one of them.
$zFace = "39.999 mm"
switch ($Tier) {
  1 { $groups = @(G $core $allE $wb 1000 "") }
  2 { $groups = @(G $core @(300, 1000, 10000) $four 1000 "") }
  3 { $groups = @(G $extra @(1000) $wb 1000 "") }
  4 { $groups = @(
        (G (Pick @("gamma")) @(0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30) $wb 1 $zFace),
        (G (Pick @("e-", "e+")) @(0.1, 0.3, 1, 3, 10, 30) $wb 1 $zFace),
        (G (Pick @("gamma", "e-", "e+")) @(0.1, 1, 10) $four 1 $zFace),
        (G (Pick @("proton", "pi+", "pi-", "kaon+", "kaon-", "deuteron", "triton", "He3", "alpha", "C12", "Fe56")) @(1, 3, 10, 30, 100) $wb 10 $zFace),
        (G (Pick @("neutron")) @(1, 3, 10, 30) $wb 10 $zFace)) }
  default { throw "tier ${Tier}: 1, 2, 3 or 4" }
}
foreach ($g in $groups) {
  if ($Species -ne "")   { $w = $Species -split ","; $g.Spec = @(@($g.Spec) + @($core) + @($extra) | Where-Object { $w -contains $_.Name } | Sort-Object Name -Unique) }
  if ($Materials -ne "") { $g.Mats = $Materials -split "," }
  if ($Energies -ne "")  { $g.Ens = @($Energies -split "," | ForEach-Object { [double]$_ }) }
}
if ($Species -ne "" -or $Materials -ne "" -or $Energies -ne "") { $groups = @($groups[0]) }

$floor = @{ em = 2000000; hadron = 200000; lightion = 50000; gcr = 50000 }
$start = @{} + $floor
if (Test-Path $satFile) { foreach ($r in Import-Csv $satFile) { $start[$r.Class] = [long]$r.Start } }
$pilotN = @{ em = 20000; hadron = 5000; lightion = 2000; gcr = 200 }

$rows = @()
foreach ($g in $groups) { foreach ($s in $g.Spec) { foreach ($e in $g.Ens) { foreach ($m in $g.Mats) {
  $pre = if ($g.GunZ -ne "") { "z_" } else { "" }
  $rows += [pscustomobject]@{ Key = ("{0}{1}_{2}_{3}" -f $pre, $s.Name, $e, $m); Sp = $s; EperA = $e; ETot = $e * $s.A; Mat = $m
                              GunZ = $g.GunZ; Serial = ($e -eq $g.SerialE) }
} } } }
$spec = @($groups | ForEach-Object { $_.Spec }); $ens = @($groups | ForEach-Object { $_.Ens } | Sort-Object -Unique); $mats = @($groups | ForEach-Object { $_.Mats } | Sort-Object -Unique)

# ---------------------------------------------------------------------------------------- the CSV
$cols = "side,tier,key,species,class,e_per_a_mev,e_total_mev,material,events,dose_gy,rms_gy,rel_rms," +
        "wall_s,loop_s,user_s,real_s,ev_s,steps_s,run2_events,run2_user_s,init_s,init_frac," +
        "gb_left,host_peak_gb,refused_species,refused_species_mev,nofs_total,nofs_lines,abandoned," +
        "carried_n,carried_mev,ntime_n,ntime_mev,g4_warnings,status,log,when"
if (-not (Test-Path $csv)) { Set-Content -Path $csv -Value $cols -Encoding ASCII }
function Get-Runs { if (Test-Path $csv) { return @(Import-Csv $csv) } else { return @() } }
function Add-Run([hashtable]$h) {
  $vals = foreach ($c in ($cols -split ",")) {
    $v = if ($h.ContainsKey($c) -and $null -ne $h[$c]) { "$($h[$c])" } else { "" }
    if ($v -match '[,"]') { '"' + ($v -replace '"', '""') + '"' } else { $v }
  }
  Add-Content -Path $csv -Value ($vals -join ",") -Encoding ASCII
}

# ---------------------------------------------------------------------------------------- macros
function New-Macro([pscustomobject]$row, [string]$side, [long[]]$counts) {
  $l = @("# P27 $($row.Key) $side - generated by tools/p27_sweep.ps1", "/B1/det/material $($row.Mat)")
  # THE ONE DEPARTURE FROM THE DEFAULT, measured on the class curve (V337): an EM run of more than
  # 1,000,000 events at the automatic batch (1,678,752 on the Titan V) is refused by the paging
  # guard at the device-stack raise - 4,000,000 1 GeV photons ask 1.875 GB with 1.848 GB free,
  # because what a run holds per event outgrows the margin the automatic batch leaves - so EM
  # runs past a million take batches of a million. The 4M photon run so made reads 436,445 ev/s
  # against 438,577 for 2M at the automatic batch: the batch does not set the rate.
  if ($side -eq "port" -and $row.Sp.Class -eq "em" -and $counts[0] -gt 1000000) { $l += "/run/setBatchSize 1000000" }
  if ($row.GunZ -ne "") { $l += "/B1/gun/z $($row.GunZ)" }
  $l += @("/run/initialize", "/control/verbose 0", "/run/verbose 1")
  $l += $row.Sp.Gun
  $l += ("/gun/energy {0} MeV" -f $row.ETot)
  if ($side -ne "port" -and $script:SeedLine -ne "") { $l += $script:SeedLine }
  foreach ($n in $counts) { $l += ("/run/printProgress {0}" -f [math]::Max(1000, [long]($n / 4))); $l += "/run/beamOn $n" }
  $p = Join-Path $macDir ("{0}_{1}_{2}.mac" -f $row.Key, $side, ($counts -join "-"))
  Set-Content -Path $p -Value $l -Encoding ASCII
  return $p
}

$unit = @{ "femtoGy" = 1e-15; "picoGy" = 1e-12; "nanoGy" = 1e-9; "microGy" = 1e-6; "milliGy" = 1e-3; "Gy" = 1.0; "kGy" = 1e3 }
function Get-Doses([string[]]$t) {
  # Master lines only: a worker's own share is printed with a "G4WTn > " prefix.
  $d = @()
  foreach ($l in $t) {
    if ($l -match '^\s*Cumulated dose per run, in scoring volume :\s*([0-9.eE+-]+)\s+(\S+)\s+rms\s*=\s*([0-9.eE+-]+)\s+(\S+)') {
      $du = $Matches[2]; $ru = $Matches[4]
      $dv = if ($unit.ContainsKey($du)) { [double]$Matches[1] * $unit[$du] } else { [double]::NaN }
      $rv = if ($unit.ContainsKey($ru)) { [double]$Matches[3] * $unit[$ru] } else { [double]::NaN }
      if ([double]$Matches[1] -eq 0) { $dv = 0.0 }
      $d += [pscustomobject]@{ Dose = $dv; Rms = $rv }
    }
  }
  return $d
}
function Get-Timers([string[]]$t) {
  $r = @()
  foreach ($l in $t) { if ($l -match '^\s*User=([0-9.eE+-]+)s\s+Real=([0-9.eE+-]+)s') { $r += [pscustomobject]@{ User = [double]$Matches[1]; Real = [double]$Matches[2] } } }
  return $r
}

# ---------------------------------------------------------------------------------------- the port
function Clear-PortEnv {
  foreach ($v in @(Get-ChildItem Env: | Where-Object { $_.Name -like "G4GPU_*" -or $_.Name -like "G4CASCADE_*" -or $_.Name -eq "G4KCDEBUG" })) {
    Remove-Item ("Env:" + $v.Name) -ErrorAction SilentlyContinue
  }
}
function Invoke-Port([pscustomobject]$row, [long]$n) {
  $mac = New-Macro $row "port" @($n)
  $log = Join-Path $logDir ("{0}_port_{1}.log" -f $row.Key, $n)
  $wait = 0
  while (Test-Path $gpuLock) { $wait++; if ($wait -eq 1) { Write-Host "  waiting for out\gpu.lock: $(Get-Content $gpuLock -TotalCount 1)" }; Start-Sleep -Seconds 30 }
  Set-Content -Path $gpuLock -Value ("P27 {0} {1} {2}" -f $row.Key, $n, (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
  $peak = 0.0; $killed = $false; $sw = [Diagnostics.Stopwatch]::StartNew()
  try {
    Clear-PortEnv
    $env:CUDA_VISIBLE_DEVICES = "1"; $env:G4GPU_HADRONIC_STAGE = "final"
    $p = Start-Process -FilePath $portExe -ArgumentList $mac -WorkingDirectory (Split-Path $portExe) `
         -RedirectStandardOutput $log -RedirectStandardError "$log.err" -NoNewWindow -PassThru
    $null = $p.Handle   # without the handle cached, ExitCode reads null after the exit
    while (-not $p.HasExited) {
      try { $p.Refresh(); $ws = $p.WorkingSet64 / 1GB; if ($ws -gt $peak) { $peak = $ws } } catch {}
      if ($peak -gt 6) { $p.Kill(); $killed = $true }
      Start-Sleep -Milliseconds 500
    }
    $p.WaitForExit()
  } finally {
    $sw.Stop()
    Remove-Item Env:\G4GPU_HADRONIC_STAGE -ErrorAction SilentlyContinue
    if ((Test-Path $gpuLock) -and ((Get-Content $gpuLock -TotalCount 1) -like "P27 *")) { Remove-Item $gpuLock }
  }
  $t = @(Get-Content $log) + @(Get-Content "$log.err" -ErrorAction SilentlyContinue)
  $h = @{ side = "port"; tier = $Tier; key = $row.Key; species = $row.Sp.Name; class = $row.Sp.Class
          e_per_a_mev = $row.EperA; e_total_mev = $row.ETot; material = $row.Mat; events = $n
          wall_s = [math]::Round($sw.Elapsed.TotalSeconds, 2); host_peak_gb = [math]::Round($peak, 2)
          log = $log; when = (Get-Date -Format "yyyy-MM-dd HH:mm:ss") }
  $d = @(Get-Doses $t)
  if ($d.Count -gt 0) { $h.dose_gy = $d[-1].Dose; $h.rms_gy = $d[-1].Rms; if ($d[-1].Dose -gt 0) { $h.rel_rms = $d[-1].Rms / $d[-1].Dose } }
  $nofs = @(); $inNofs = $false; $refS = 0; $refE = 0.0; $carN = 0; $carE = 0.0
  foreach ($l in $t) {
    if ($l -match '^event loop\s+([0-9.eE+-]+)\s+ms for\s+\d+\s+events\s+=\s+([0-9.eE+-]+)\s+events/s') { $h.loop_s = [double]$Matches[1] / 1000; $h.ev_s = [double]$Matches[2] }
    if ($l -match '^time\s+[0-9.eE+-]+\s+ms for .*\(([0-9.eE+-]+)\s+track-steps/s\)') { $h.steps_s = [double]$Matches[1] }
    if ($l -match '([0-9.]+) GB of ([0-9.]+) GB left') { $h.gb_left = [double]$Matches[1] }
    if ($l -match '^\*\*\* (\d+) SECONDARIES OF SPECIES THIS PORT CANNOT TRANSPORT, ([0-9.eE+-]+) MeV') { $refS = [long]$Matches[1]; $refE = [double]$Matches[2] }
    if ($l -match '^\*\*\* (\d+) HADRONIC INTERACTIONS WITH NO FINAL STATE') { $h.nofs_total = [long]$Matches[1]; $inNofs = $true; continue }
    if ($inNofs) {
      if ($l -match '^\s+(\d+)\s+([0-9.eE+-]+)\s+MeV\s+(\S.*)$') { $nofs += ("{0} {1} MeV {2}" -f $Matches[1], $Matches[2], $Matches[3].Trim()) }
      elseif ($l -match '^\S' -and $nofs.Count -gt 0) { $inNofs = $false }
    }
    if ($l -match '^WARNING: (\d+) tracks abandoned') { $h.abandoned = [long]$Matches[1] }
    if ($l -match '^carried away: (\d+) particles, ([0-9.eE+-]+) MeV') { $carN = [long]$Matches[1]; $carE = [double]$Matches[2] }
    if ($l -match '^neutron time cut: (\d+) neutrons killed past 10 us, ([0-9.eE+-]+) MeV') { $h.ntime_n = [long]$Matches[1]; $h.ntime_mev = [double]$Matches[2] }
  }
  $h.refused_species = $refS; $h.refused_species_mev = $refE; $h.carried_n = $carN; $h.carried_mev = $carE
  if (-not $h.ContainsKey("nofs_total")) { $h.nofs_total = 0 }
  if (-not $h.ContainsKey("abandoned")) { $h.abandoned = 0 }
  $h.nofs_lines = ($nofs -join "; ")
  $status = "ok"
  if ($killed) { $status = "killed-host-6GB" }
  elseif ($p.ExitCode -ne 0) { $status = "exit-$($p.ExitCode)" }
  elseif ($d.Count -eq 0) { $status = "no-dose" }
  elseif ($null -eq $h.ev_s) { $status = "no-rate" }
  $h.status = $status
  Add-Run $h
  Write-Host ("  port {0,9:N0}  {1,12:E5} Gy +/- {2:P2}  {3,10:N1} ev/s  {4} GB left  host {5:N2} GB  nofs {6}  {7}" -f $n, $h.dose_gy, $h.rel_rms, $h.ev_s, $h.gb_left, $peak, $h.nofs_total, $status)
  return $h
}

# The row's port runs, the saturation test, and the next count - from runs.csv alone.
function Get-PortState([pscustomobject]$row) {
  $runs = @(Get-Runs | Where-Object { $_.side -eq "port" -and $_.key -eq $row.Key -and $_.status -eq "ok" } |
            Sort-Object { [long]$_.events })
  $fl = $floor[$row.Sp.Class]
  $sat = $null
  for ($i = 0; $i -lt $runs.Count - 1; $i++) {
    $a = $runs[$i]; $b = $runs[$i + 1]
    if ([long]$a.events -ge $fl -and [long]$b.events -eq 2 * [long]$a.events -and [double]$b.ev_s -le 1.03 * [double]$a.ev_s) { $sat = $a; break }
  }
  $last = if ($runs.Count -gt 0) { $runs[-1] } else { $null }
  return [pscustomobject]@{ Runs = $runs; Sat = $sat; Last = $last }
}

function Invoke-PortRow([pscustomobject]$row) {
  $s0 = if ($StartOverride -gt 0) { $StartOverride } else { $start[$row.Sp.Class] }
  while ($true) {
    $st = Get-PortState $row
    # A FAILED run (the engine refused it: V337's pool FATAL for a 1 GeV electron past ~1.6M
    # events, the paging guard) is never repeated and no count at or above it is tried again. A
    # row with no good run below its failure halves down to a sixteenth of its floor; a row whose
    # doubling reaches the failure while still unsaturated halves BELOW its lowest good run, so
    # that the rate's shape under the failing count is on record (a pair within 3% there is
    # saturation below the floor, and is reported as that).
    $bad = @(Get-Runs | Where-Object { $_.side -eq "port" -and $_.key -eq $row.Key -and $_.status -ne "ok" })
    $failMin = if ($bad.Count -gt 0) { ($bad | ForEach-Object { [long]$_.events } | Measure-Object -Minimum).Minimum } else { [long]::MaxValue }
    $lowest = if ($st.Runs.Count -gt 0) { [long]$st.Runs[0].events } else { [long]::MaxValue }
    $fl16 = [long]($floor[$row.Sp.Class] / 16)
    if ($null -eq $st.Last) {
      $next = [math]::Min($s0, [long]($failMin / 2))
      if ($bad.Count -gt 0 -and $next -lt $fl16) { Write-Host "  port: every count down to $fl16 failed ($($bad[-1].status)) - row closed"; return }
    }
    elseif ($null -eq $st.Sat -and 2 * [long]$st.Last.events -ge $failMin) {
      $next = [long]($lowest / 2)
      $pairBelow = $false
      for ($i = 0; $i -lt $st.Runs.Count - 1; $i++) {
        if ([long]$st.Runs[$i + 1].events -eq 2 * [long]$st.Runs[$i].events -and [double]$st.Runs[$i + 1].ev_s -le 1.03 * [double]$st.Runs[$i].ev_s) { $pairBelow = $true }
      }
      if ($pairBelow -or $next -lt $fl16) {
        Write-Host ("  port: {0:N0} failed ({1}); closed at {2:N0}, {3}" -f $failMin, $bad[-1].status, [long]$st.Last.events, $(if ($pairBelow) { "saturated below the floor" } else { "NOT SATURATED" }))
        return
      }
    }
    else {
      if ($null -ne $st.Sat -and 2 * [long]$st.Last.events -ge $failMin) {
        Write-Host ("  port: saturated at {0:N0}; {1:N0} failed - row closed, rms {2:P2}" -f [long]$st.Sat.events, $failMin, [double]$st.Last.rel_rms); return
      }
      $rel = [double]$st.Last.rel_rms
      if ($null -ne $st.Sat -and $rel -le 0.01) { Write-Host ("  port: saturated at {0:N0} ({1:N1} -> {2:N1} ev/s), rms {3:P2} - done" -f [long]$st.Sat.events, [double]$st.Sat.ev_s, [double]$st.Last.ev_s, $rel); return }
      $next = 2 * [long]$st.Last.events
      $pred = $next / [double]$st.Last.ev_s / 60
      $cap = if ($null -eq $st.Sat) { $PortCapMin } else { $PortRmsCapMin }
      if ($pred -gt $cap) {
        $why = if ($null -eq $st.Sat) { "NOT SATURATED" } else { "rms {0:P2} above 1%" -f $rel }
        Write-Host ("  port: {0} at {1:N0}; the next count would take {2:N0} min, over the {3} min cap - row closed as it stands" -f $why, [long]$st.Last.events, $pred, $cap)
        return
      }
    }
    Invoke-Port $row $next | Out-Null
  }
}

# ---------------------------------------------------------------------------------------- Geant4
function Invoke-G4([pscustomobject]$row, [string]$mode, [long]$n1, [long]$n2, [string]$tag) {
  $mac = New-Macro $row $mode @($n1, $n2)
  $log = Join-Path $logDir ("{0}_{1}_{2}_{3}-{4}.log" -f $row.Key, $mode, $tag, $n1, $n2)
  Remove-Item Env:\G4FORCENUMBEROFTHREADS, Env:\G4FORCE_RUN_MANAGER_TYPE -ErrorAction SilentlyContinue
  if ($mode -eq "box") { $env:G4FORCENUMBEROFTHREADS = "10" } else { $env:G4FORCE_RUN_MANAGER_TYPE = "Serial" }
  $sw = [Diagnostics.Stopwatch]::StartNew()
  try {
    $p = Start-Process -FilePath "cmd.exe" -ArgumentList "/c $g4Run $mac" -RedirectStandardOutput $log `
         -RedirectStandardError "$log.err" -NoNewWindow -PassThru
    $p.WaitForExit()
  } finally {
    $sw.Stop()
    Remove-Item Env:\G4FORCENUMBEROFTHREADS, Env:\G4FORCE_RUN_MANAGER_TYPE -ErrorAction SilentlyContinue
  }
  $t = @(Get-Content $log) + @(Get-Content "$log.err" -ErrorAction SilentlyContinue)
  $d = @(Get-Doses $t); $tm = @(Get-Timers $t)
  $warn = @($t | Where-Object { $_ -match 'G4Exception-START' }).Count
  $h = @{ side = $mode; tier = $Tier; key = $row.Key; species = $row.Sp.Name; class = $row.Sp.Class
          e_per_a_mev = $row.EperA; e_total_mev = $row.ETot; material = $row.Mat; events = $n1
          wall_s = [math]::Round($sw.Elapsed.TotalSeconds, 2); g4_warnings = $warn; log = $log
          when = (Get-Date -Format "yyyy-MM-dd HH:mm:ss"); run2_events = $n2 }
  $status = "ok"
  if (@($t | Where-Object { $_ -match "Fatal Exception" }).Count -gt 0) { $status = "fatal" }
  elseif ($d.Count -lt 2 -or $tm.Count -lt 2) { $status = "parse-fail(doses $($d.Count), timers $($tm.Count))" }
  if ($mode -eq "serial" -and @($t | Where-Object { $_ -match '^G4WT\d' }).Count -gt 0) { $status = "ran-multithreaded" }
  if ($status -eq "ok") {
    $h.dose_gy = $d[0].Dose; $h.rms_gy = $d[0].Rms; if ($d[0].Dose -gt 0) { $h.rel_rms = $d[0].Rms / $d[0].Dose }
    $h.user_s = $tm[0].User; $h.real_s = $tm[0].Real; $h.run2_user_s = $tm[1].User
    $h.ev_s = if ($mode -eq "box") { $n1 / $tm[0].Real } else { $n1 / $tm[0].User }
    $h.init_s = $tm[0].User - ($n1 / $n2) * $tm[1].User
    $h.init_frac = $h.init_s / $tm[0].User
  }
  $h.status = $status
  Add-Run $h
  Write-Host ("  {0,-6} {1,9:N0}  {2,12:E5} Gy +/- {3:P2}  User {4:N1} s  Real {5:N1} s  {6,10:N1} ev/s  init {7:N1} s ({8:P1})  {9}" -f $mode, $n1, $h.dose_gy, $h.rel_rms, $h.user_s, $h.real_s, $h.ev_s, $h.init_s, $h.init_frac, $status)
  return $h
}

function Invoke-G4Row([pscustomobject]$row, [string]$mode) {
  $done = @(Get-Runs | Where-Object { $_.side -eq $mode -and $_.key -eq $row.Key })
  $good = @($done | Where-Object { $_.status -eq "ok" -and $_.log -notlike "*_pilot_*" -and [double]$_.init_frac -lt 0.05 })
  if ($good.Count -gt 0) { Write-Host ("  {0}: on record, {1:N0} events, init {2:P1}" -f $mode, [long]$good[-1].events, [double]$good[-1].init_frac); return }
  $failed = @($done | Where-Object { $_.status -ne "ok" })
  $retrySeed = $false
  if ($failed.Count -gt 0) {
    $onlyFatal = @($failed | Where-Object { $_.status -ne "fatal" }).Count -eq 0
    $seeded = @($done | Where-Object { $_.log -like "*_rowseed_*" }).Count -gt 0
    if ($onlyFatal -and -not $seeded) { $retrySeed = $true }
    else { Write-Host "  ${mode}: a failed run is on record; row left for inspection"; return }
  }
  # The rms per event: the port's largest run, else this row's pilot.
  $port = (Get-PortState $row).Last
  # A serial row is sized from the same row's box run when there is one - its second run's User
  # per event is the cost of an event on a core, and the serial row's own two runs then measure
  # its start-up - so a pilot is needed only where no box run exists.
  $box = @(Get-Runs | Where-Object { $_.side -eq "box" -and $_.key -eq $row.Key -and $_.status -eq "ok" -and $_.log -notlike "*_pilot_*" })
  if ($mode -eq "serial" -and $box.Count -gt 0) {
    $pl = $box[-1]; $fromBox = $true
  } else {
    $fromBox = $false
    $pilot = @($done | Where-Object { $_.log -like "*_pilot_*" -and $_.status -eq "ok" })
    if ($pilot.Count -eq 0) {
      $pn = $pilotN[$row.Sp.Class]
      Invoke-G4 $row $mode $pn $pn "pilot" | Out-Null
      $pilot = @(Get-Runs | Where-Object { $_.side -eq $mode -and $_.key -eq $row.Key -and $_.log -like "*_pilot_*" -and $_.status -eq "ok" })
      if ($pilot.Count -eq 0) { return }
    }
    $pl = $pilot[-1]
  }
  # A pilot whose second run took under the timer's 0.05 s resolution (200 nuclei at 100 MeV/n, which
  # stop in a centimetre) is read at that resolution, not as zero: a zero cost made the count infinite.
  $c = [math]::Max(0.05, [double]$pl.run2_user_s) / [double]$pl.run2_events   # core-seconds an event
  # The start-up the 4% is taken of. Measured, the box's is ~0 on the first rows (the workers are
  # built at /run/initialize, before the timer starts) - so the floor is what keeps a box row
  # from being a second long, where ten workers' last events and the timer's resolution are the
  # measurement: 8 core-s (200 core-s of User, ~20 s of Real); a serial row, 2.4 (60 s of User).
  $init = if ($fromBox) { 2.4 } elseif ($mode -eq "box") { [math]::Max(8.0, [double]$pl.user_s - [double]$pl.run2_user_s) } else { [math]::Max(2.4, [double]$pl.user_s - [double]$pl.run2_user_s) }
  $relVar = if ($null -ne $port -and [double]$port.rel_rms -gt 0) { [math]::Pow([double]$port.rel_rms, 2) * [double]$port.events } `
            elseif ([double]$pl.rel_rms -gt 0) { [math]::Pow([double]$pl.rel_rms, 2) * [double]$pl.events } else { 1e9 }
  $nRms = [math]::Min(1e12, [math]::Ceiling($relVar / ($RmsTarget * $RmsTarget)))
  $nInit = [math]::Ceiling(25.0 * $init / $c)                      # start-up at 4% of User
  $cores = if ($mode -eq "box") { 10.0 * [math]::Max(0.5, [double]$pl.user_s / [double]$pl.real_s / 10.0) } else { 1.0 }
  $capMin = if ($mode -eq "box") { $CpuCapMin } else { $SerialCapMin }
  $nCap = [math]::Floor($capMin * 60 * $cores / $c / 1.25)         # both runs inside the cap
  $nMin = if ($fromBox) { 400 } else { 4 * [long]$pl.events }
  $n = [long][math]::Max([math]::Max($nRms, $nInit), $nMin)
  $capped = $false
  if ($n -gt $nCap) { $n = [long]$nCap; $capped = $true }
  $n = [long]([math]::Ceiling($n / 40.0) * 40)
  Write-Host ("  {0}: pilot {1:N4} core-s an event, start-up {2:N1} core-s; rms needs {3:N0}, start-up {4:N0}, cap {5:N0} -> {6:N0}{7}" -f $mode, $c, $init, $nRms, $nInit, $nCap, $n, $(if ($capped) { " (CAPPED)" } else { "" }))
  $tag = if ($retrySeed) { "rowseed" } else { "row" }
  $script:SeedLine = if ($retrySeed) { "/random/setSeeds 20261121 27" } else { "" }
  $h = Invoke-G4 $row $mode $n ([long]($n / 4)) $tag
  if ($h.status -eq "fatal" -and -not $retrySeed) {
    # GEANT4 ITSELF STOPPED: 11.1.1's G4BinaryLightIonReaction "Cannot correct 4-momentum of cascade
    # particles" is a FatalException (had006) for a triton on hydrogen near 3 GeV (V337). One event
    # in some hundred thousand; the same count on another seed is the row.
    Write-Host "  ${mode}: Geant4 stopped on a fatal exception - the same count once more on /random/setSeeds 20261121 27"
    $script:SeedLine = "/random/setSeeds 20261121 27"
    $h = Invoke-G4 $row $mode $n ([long]($n / 4)) "rowseed"
  }
  $script:SeedLine = ""
  if ($h.status -eq "ok" -and $h.init_frac -ge 0.05) {
    $n2 = [long]([math]::Ceiling($n * $h.init_frac / 0.04 / 40.0) * 40)
    Write-Host ("  {0}: start-up {1:P1} of User - rerun at {2:N0}" -f $mode, $h.init_frac, $n2)
    Invoke-G4 $row $mode $n2 ([long]($n2 / 4)) "row" | Out-Null
  }
}

# ---------------------------------------------------------------------------------------- run
$sideList = $Sides -split ","
Write-Host ("P27 tier {0}: {1} rows ({2} species x {3} energies x {4} materials), sides {5}" -f $Tier, $rows.Count, @($spec).Count, @($ens).Count, @($mats).Count, $Sides)
foreach ($k in $start.Keys) { Write-Host ("  start count {0}: {1:N0} (floor {2:N0})" -f $k, $start[$k], $floor[$k]) }
if ($Plan) { foreach ($r in $rows) { Write-Host ("  {0,-34} {1,8} MeV total  {2} {3}" -f $r.Key, $r.ETot, $r.Sp.Class, $(if ($r.Serial) { "+serial" } else { "" })) }; exit 0 }

# ---------------------------------------------------------------------------------------- report
# -Report writes out\p27\tier<N>.md from runs.csv and runs nothing: the comparison (the port's
# largest run against the box row; the difference over the two rms in quadrature), the ledgers,
# and the timing (the port's rate at its largest run, the curve it went through, per core from
# the serial row where there is one - else the box run's events / User, marked *, which is a
# core-second shared with nine other workers and a hyperthread-free one only because
# G4FORCENUMBEROFTHREADS is the physical core count - the box, x core, x box).
if ($Report) {
  $all = Get-Runs
  $md = @("| row | port (Gy) | G4 box (Gy) | diff | sigma | port events | G4 events | ledgers |",
          "|---|--:|--:|--:|--:|--:|--:|---|")
  $tm = @("| row | port ev/s | the port's curve (events: ev/s) | per core | box | x core | x box | GB left |",
          "|---|--:|---|--:|--:|--:|--:|--:|")
  foreach ($r in $rows) {
    $st = Get-PortState $r; $p = $st.Last
    $b = @($all | Where-Object { $_.side -eq "box" -and $_.key -eq $r.Key -and $_.status -eq "ok" -and $_.log -notlike "*_pilot_*" })
    $s = @($all | Where-Object { $_.side -eq "serial" -and $_.key -eq $r.Key -and $_.status -eq "ok" -and $_.log -notlike "*_pilot_*" })
    $bx = if ($b.Count -gt 0) { $b[-1] } else { $null }
    $pd = if ($p) { "{0:E4} +/- {1:E1}" -f [double]$p.dose_gy, [double]$p.rms_gy } else { "-" }
    $bd = if ($bx) { "{0:E4} +/- {1:E1}" -f [double]$bx.dose_gy, [double]$bx.rms_gy } else { "-" }
    $diff = "-"; $sig = "-"
    if ($p -and $bx -and [double]$bx.dose_gy -gt 0) {
      $dd = [double]$p.dose_gy - [double]$bx.dose_gy
      $diff = "{0:+0.00%;-0.00%}" -f ($dd / [double]$bx.dose_gy)
      $q = [math]::Sqrt([math]::Pow([double]$p.rms_gy, 2) + [math]::Pow([double]$bx.rms_gy, 2))
      if ($q -gt 0) { $sv = $dd / $q; $sig = if ([math]::Abs($sv) -gt 3) { "**{0:+0.0;-0.0}**" -f $sv } else { "{0:+0.0;-0.0}" -f $sv } }
    }
    $led = @()
    if ($p) {
      if ([long]$p.nofs_total -gt 0) { $led += "no final state $($p.nofs_total): $($p.nofs_lines)" }
      if ([long]$p.refused_species -gt 0) { $led += "species refused $($p.refused_species) ($($p.refused_species_mev) MeV)" }
      if ([long]$p.abandoned -gt 0) { $led += "abandoned $($p.abandoned)" }
    }
    $ledS = if ($led.Count -gt 0) { $led -join "; " } else { "0" }
    $pe = if ($p) { "{0:N0}" -f [long]$p.events } else { "-" }
    $be = if ($bx) { "{0:N0}" -f [long]$bx.events } else { "-" }
    $md += "| $($r.Key) | $pd | $bd | $diff | $sig | $pe | $be | $ledS |"
    $curve = ($st.Runs | ForEach-Object { "{0:N0}: {1:N0}" -f [long]$_.events, [double]$_.ev_s }) -join ", "
    if ($st.Runs.Count -gt 0 -and $null -eq $st.Sat) { $curve += " (NOT SATURATED)" }
    $pr = if ($p) { [double]$p.ev_s } else { [double]::NaN }
    $core = [double]::NaN; $coreS = "-"
    if ($s.Count -gt 0) { $core = [double]$s[-1].ev_s; $coreS = "{0:N1}" -f $core }
    elseif ($bx) { $core = [double]$bx.events / [double]$bx.user_s; $coreS = "{0:N1}*" -f $core }
    $boxR = if ($bx) { [double]$bx.ev_s } else { [double]::NaN }
    $xc = if ($core -gt 0 -and $pr -gt 0) { "{0:N1}" -f ($pr / $core) } else { "-" }
    $xb = if ($boxR -gt 0 -and $pr -gt 0) { "{0:N2}" -f ($pr / $boxR) } else { "-" }
    $boxS = if ($bx) { "{0:N1}" -f $boxR } else { "-" }
    $prS = if ($p) { "{0:N1}" -f $pr } else { "-" }
    $gb = if ($p) { $p.gb_left } else { "-" }
    $tm += "| $($r.Key) | $prS | $curve | $coreS | $boxS | $xc | $xb | $gb |"
  }
  $f = Join-Path $Out ("tier{0}.md" -f $Tier)
  (@("## doses", "") + $md + @("", "## timing", "") + $tm) | Set-Content -Path $f -Encoding UTF8
  Write-Host "written $f"
  exit 0
}
if (-not (Test-Path $portExe) -and $sideList -contains "port") { Write-Host "FATAL: no $portExe"; exit 1 }
if (-not (Test-Path (Join-Path $root "ref\b1matbuild\Release\b1mat.exe")) -and ($sideList -contains "box" -or $sideList -contains "serial")) { Write-Host "FATAL: ref\b1matbuild\Release\b1mat.exe not built"; exit 1 }
foreach ($r in $rows) {
  Write-Host ("=== {0}  ({1} MeV total, {2})  {3}" -f $r.Key, $r.ETot, $r.Sp.Class, (Get-Date -Format "MM-dd HH:mm"))
  if ($sideList -contains "port") { Invoke-PortRow $r }
  if ($sideList -contains "box") { Invoke-G4Row $r "box" }
  if ($sideList -contains "serial" -and $r.Serial) { Invoke-G4Row $r "serial" }
}
Write-Host "DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
