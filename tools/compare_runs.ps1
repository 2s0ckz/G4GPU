# Compares the dose two runs of the same example reported.
#
# Written for one specific gap: the pipeline runs everything in batch, so a difference between
# a batch run and a macro-driven one - the gun left in a state a vis macro put it in, say - was
# invisible to every check. That difference reached a third of the dose before a person noticed
# it. See docs/RISK.md V3.
#
# The two runs use different event counts on purpose (2,000,000 and 10,000), so this compares
# the *scaled* dose, which is what both report per 10k events. The tolerance is on the
# combined statistical uncertainty of the two, because at 10,000 events the uncertainty is
# about 12 pGy on 429 - three per cent - and a fixed percentage would either be too loose to
# catch anything at 2M or fail constantly at 10k.
param(
  [Parameter(Mandatory = $true)][string]$BatchLog,
  [Parameter(Mandatory = $true)][string]$MacroLog,
  [double]$MaxSigma = 4.0
)

function Read-Dose([string]$path, [string]$label) {
  if (-not (Test-Path -LiteralPath $path)) {
    Write-Output "FATAL: no $label log at $path"
    exit 1
  }
  # "scaled to 10k events: 429.256 pGy  +/- 12.2957 (this run)"
  #
  # Not the "Cumulated dose per run" line, which is formatted with G4BestUnit and so switches
  # between picoGy and nanoGy depending on the event count - the first version of this parser
  # matched only picoGy and failed on the 2,000,000-event run. A line whose unit depends on the
  # value is not something to key a check on.
  foreach ($line in Get-Content -LiteralPath $path) {
    if ($line -match 'scaled to 10k events:\s*([0-9.eE+-]+)\s*pGy\s*\+/-\s*([0-9.eE+-]+)') {
      return @{ dose = [double]$Matches[1]; rms = [double]$Matches[2] }
    }
  }
  Write-Output "FATAL: $label log has no 'scaled to 10k events' line."
  exit 1
}

# $batchRun / $macroRun, not $a / $b: PowerShell variables are case-insensitive, so `$a` and the
# parameter `-A` were the same variable. The first assignment overwrote the path it had just
# been given, and the script reported "both runs report zero uncertainty" for two runs that had
# reported it perfectly well.
$batchRun = Read-Dose $BatchLog 'batch'
$macroRun = Read-Dose $MacroLog 'macro'

$sigma = [Math]::Sqrt($batchRun.rms * $batchRun.rms + $macroRun.rms * $macroRun.rms)
if ($sigma -le 0) {
  Write-Output "FATAL: both runs report zero uncertainty; nothing to compare against."
  exit 1
}
$diff = [Math]::Abs($batchRun.dose - $macroRun.dose)
$n = $diff / $sigma
$shown = '{0:N2}' -f $n

if ($n -gt $MaxSigma) {
  Write-Output ("FATAL: the same example disagrees with itself. batch {0} pGy, macro {1} pGy," -f $batchRun.dose, $macroRun.dose)
  Write-Output ("       {0} sigma apart (limit {1}). A macro that touches the gun must not" -f $shown, $MaxSigma)
  Write-Output "       change what the example fires."
  exit 1
}
Write-Output ("batch and macro agree: {0} pGy vs {1} pGy ({2} sigma)" -f $batchRun.dose, $macroRun.dose, $shown)
exit 0
