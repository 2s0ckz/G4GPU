# The dose must not depend on how many live track slots a run was given.
#
# When the output buffer cannot hold what a track would produce, the transport defers that
# track to a later iteration rather than dropping it. That is the whole claim of the throttle,
# and the way to check a claim about something not being lost is to vary the thing that would
# lose it and see whether the answer moves. Three pools, and the answer has to be identical -
# not close, identical: the deferral changes the ORDER tracks are stepped in and nothing else,
# and every random stream is keyed by the track rather than by its slot.
#
# 2.5 slots per event is in the list for a second reason. It is the only configuration in this
# pipeline that makes the pool an ODD number, and a track slot is 236 bytes, so an odd pool put
# the second half of the ping-pong arena at 4 mod 8 and every double in it was misaligned. The
# run died with "CUDA error misaligned address" reported from a cudaMemcpy nowhere near the
# kernel that faulted. tests/test_track_arena.cu checks that arithmetic directly; this checks
# that a real run of it works, which is a different claim.
#
# Labels arrive as ONE comma-separated string rather than as an array, because powershell -File
# does not split a comma-separated argument into an array - it hands the whole thing over as a
# single string, and the first version of this script reported "no output at a,b,c" as though
# that were a filename. It was.
param(
  [Parameter(Mandatory = $true)][string]$Prefix,
  [Parameter(Mandatory = $true)][string]$Labels,
  [string]$Suffix = '.txt'
)

$names = $Labels -split ','
$doses = @()

foreach ($label in $names) {
  $f = "$Prefix$label$Suffix"
  if (-not (Test-Path -LiteralPath $f)) {
    Write-Output "FATAL: no output at $f"
    exit 1
  }
  $dose = $null
  foreach ($line in Get-Content -LiteralPath $f) {
    # "dose10k 429.8198 pGy  sigma 2.7598 pGy"
    if ($line -match '^\s*dose10k\s+([0-9.eE+-]+)\s') { $dose = $Matches[1] }
  }
  if ($null -eq $dose) {
    Write-Output "FATAL: $f has no dose10k line, so the run at $label slots per event did not finish."
    Get-Content -LiteralPath $f | Select-Object -Last 5 | ForEach-Object { Write-Output "       $_" }
    exit 1
  }
  $doses += $dose
}

$pairs = for ($i = 0; $i -lt $doses.Count; $i++) { "$($names[$i]): $($doses[$i])" }
Write-Output ("  dose10k by slots per event - " + ($pairs -join ', '))

if (($doses | Select-Object -Unique).Count -ne 1) {
  Write-Output "FATAL: the dose depends on the size of the track pool."
  Write-Output "       The throttle is meant to defer a track it has no room for, not drop it,"
  Write-Output "       so a smaller pool may cost iterations and must not cost energy."
  exit 1
}
Write-Output "  the pool size does not change the answer"
exit 0
