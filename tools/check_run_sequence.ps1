# Two runs in one process must differ; a fresh process must replay both.
#
# This is Geant4's behaviour and it is two claims, not one. Run B1.exe twice and you get the
# same answer - that is what makes a number quotable. Run two /run/beamOn in one session and
# you get two different answers within noise - that is what makes the second run worth doing.
#
# Getting one without the other is the failure mode, and it is quiet either way. A port that
# repeats the run gives an "independent" second sample that reduces no variance and an
# accumulation across runs that multiplies one answer instead of averaging several. A port that
# does not replay cannot be checked against a reference at all.
#
# What made this worth a check: the difference between runs used to come ENTIRELY from the host
# engine that draws the primaries. Every track's shower stream was keyed on its index within
# the run, so run two fired different primaries into identical showers - and a generator that
# drew no random numbers repeated the run exactly, bit for bit. B1 hid it by having a random
# beam spot. See docs/RISK.md V12.
param(
  [Parameter(Mandatory = $true)][string]$TwoRunLog,
  [Parameter(Mandatory = $true)][string]$RepeatLog,
  [Parameter(Mandatory = $true)][string]$SingleLog
)

# Returns one PSObject per run with its dose and its uncertainty, both in picoGy.
#
# The rms is parsed and not assumed, because the threshold below is a number of sigma and the
# run is the only thing that knows its own sigma. Guessing a percentage here is how the
# builder-vs-project check spent its whole life as a coin toss (docs/RISK.md V10), and the
# first version of THIS check repeated it: 5% against a spread of 2.9%.
function Doses([string]$path) {
  if (-not (Test-Path -LiteralPath $path)) {
    Write-Output "FATAL: no log at $path"
    exit 1
  }
  $out = @()
  foreach ($line in Get-Content -LiteralPath $path) {
    # " Cumulated dose per run, in scoring volume : 832.457 picoGy rms = 17.1678 picoGy"
    if ($line -match 'Cumulated dose per run[^:]*:\s+([0-9.eE+-]+)\s+(\S+)\s+rms\s+=\s+([0-9.eE+-]+)\s+(\S+)') {
      $v = [double]$Matches[1]
      $e = [double]$Matches[3]
      # nanoGy and picoGy both appear depending on the run size; normalise to picoGy so a
      # comparison between a 20k run and a 40k one is a comparison of numbers.
      if ($Matches[2] -eq 'nanoGy') { $v = $v * 1000.0 }
      if ($Matches[4] -eq 'nanoGy') { $e = $e * 1000.0 }
      $out += [PSCustomObject]@{ Dose = $v; Rms = $e }
    }
  }
  return $out
}

# @(...) because PowerShell unwraps a one-element array to the element, and the
# 40000-event log holds exactly one run - so $one.Count was $null and the count check
# below reported "found ." rather than "found 1".
$two = @(Doses $TwoRunLog)
$rep = @(Doses $RepeatLog)
$one = @(Doses $SingleLog)

if ($two.Count -ne 2) {
  Write-Output "FATAL: expected two runs in $TwoRunLog, found $($two.Count)."
  exit 1
}
if ($rep.Count -ne 2) {
  Write-Output "FATAL: expected two runs in $RepeatLog, found $($rep.Count)."
  exit 1
}
if ($one.Count -ne 1) {
  Write-Output "FATAL: expected one run in $SingleLog, found $($one.Count)."
  exit 1
}

Write-Output ("  two runs in one process: {0:N3} +/- {1:N3} then {2:N3} +/- {3:N3} pGy" -f `
  $two[0].Dose, $two[0].Rms, $two[1].Dose, $two[1].Rms)

# 1. The second run is a different sample.
if ($two[0].Dose -eq $two[1].Dose) {
  Write-Output "FATAL: the two runs in one process gave the same dose to the last bit."
  Write-Output "       A second run must be an independent sample, as it is in Geant4."
  Write-Output "       G4RunManager::stream_pos_ is what makes that true; check it advances."
  exit 1
}

# 2. ...but only within noise. A second run that differs by a lot is not a second sample of
#    the same physics, it is a bug - a geometry that moved, a scorer that did not reset, an
#    accumulable still holding the first run.
#
#    In SIGMA, from the rms each run reports, not in per cent. B1 at 20000 events has a 2.1%
#    standard error, so two runs are 2.9% apart on average and a 5% threshold - which is what
#    this said first - fails one run in twenty for no reason at all. Four sigma leaves the
#    check able to see what it is for: a geometry that moved or a scorer that did not reset is
#    tens of sigma, not two.
$diff = [Math]::Abs($two[1].Dose - $two[0].Dose)
$sigma = [Math]::Sqrt($two[0].Rms * $two[0].Rms + $two[1].Rms * $two[1].Rms)
$nsig = if ($sigma -gt 0) { $diff / $sigma } else { 0 }
Write-Output ("  they differ by {0:N3} pGy = {1:N2} sigma, which must be noise and not physics" -f $diff, $nsig)
if ($nsig -gt 4.0) {
  Write-Output "FATAL: the two runs differ by more than statistics can explain."
  Write-Output "       Something other than the random stream changed between them."
  exit 1
}

# 3. A fresh process replays. Same macro, run again, same two numbers - exactly.
if ($rep[0].Dose -ne $two[0].Dose -or $rep[1].Dose -ne $two[1].Dose) {
  Write-Output "FATAL: a second process did not replay the same two runs."
  Write-Output ("       first process:  {0} then {1}" -f $two[0].Dose, $two[1].Dose)
  Write-Output ("       second process: {0} then {1}" -f $rep[0].Dose, $rep[1].Dose)
  Write-Output "       A run must be reproducible from its seed and its position in the stream."
  exit 1
}
Write-Output "  a second process replays both runs exactly"

# 4. The stream carries on with no gap and no overlap: two runs of N are one run of 2N.
#
#    This is the sharp form of the claim, and the one that fails if the offset is wrong rather
#    than merely absent. An offset that jumped too far would still make the runs differ and
#    still replay; it would just quietly skip part of the stream. Summed dose is the observable
#    because dose is extensive in the events.
$sum = $two[0].Dose + $two[1].Dose
$rel = [Math]::Abs($sum - $one[0].Dose) / $one[0].Dose
Write-Output ("  two runs of N summed {0:N3} pGy against {1:N3} for one run of 2N (rel {2:E2})" -f $sum, $one[0].Dose, $rel)
if ($rel -gt 1e-5) {
  Write-Output "FATAL: two runs of N are not one run of 2N."
  Write-Output "       The run offset skips or repeats part of the random stream, so a run is"
  Write-Output "       not the sample it would have been as part of a longer run."
  exit 1
}

Write-Output "the run sequence behaves as Geant4's does"
exit 0
