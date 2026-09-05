# Enforces example B1's agreement with the Geant4 11.1.1 reference.
#
# The headline claim of this project is that B1 agrees with a 2-million-event Geant4 run to a
# fraction of a sigma. Until now the pipeline *printed* that number and never tested it, so a
# change that drifted the dose to five sigma would have passed every check here and been
# reported as ALL OK. The number was in the log for a person to read, which is not a check.
#
# What this is not: a regression test on the exact value. Two things legitimately move it - a
# different random stream, and any change to where the randomness is drawn - and demanding a
# fixed number would make every such change look like a physics error. Moving primary
# generation from the device to the host shifted B1 from 0.093 to 0.313 sigma, which is not a
# problem; the same shift to 4 sigma would be.
#
# The threshold is on the agreement, in units of the combined uncertainty of the two runs.
param(
  [Parameter(Mandatory = $true)][string]$Log,
  [double]$MaxSigma = 3.0
)

if (-not (Test-Path -LiteralPath $Log)) {
  Write-Output "FATAL: no B1 log at $Log"
  exit 1
}

$sigma = $null
foreach ($line in Get-Content -LiteralPath $Log) {
  # "difference           : 0.114352 pGy = 0.0929394 sigma"
  if ($line -match 'difference\s*:\s*\S+\s*pGy\s*=\s*([0-9.eE+-]+)\s*sigma') {
    $sigma = [double]$Matches[1]
  }
}
if ($null -eq $sigma) {
  Write-Output "FATAL: example B1 did not report its agreement with the Geant4 reference."
  Write-Output "       Expected a line of the form 'difference ... = <n> sigma'."
  exit 1
}

$shown = '{0:N4}' -f $sigma
if ([Math]::Abs($sigma) -gt $MaxSigma) {
  Write-Output "FATAL: example B1 is $shown sigma from the Geant4 11.1.1 reference."
  Write-Output "       The limit is $MaxSigma. Either the physics changed or the reference did;"
  Write-Output "       both need explaining before this passes."
  exit 1
}
Write-Output "B1 agrees with Geant4 11.1.1 to $shown sigma (limit $MaxSigma)"
exit 0
