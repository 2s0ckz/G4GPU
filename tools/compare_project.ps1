# Compares what the builder measured with what the project it generated measures.
#
# The whole design principle of builder/build_scene.hh is that running a model in the builder
# and saving it as C++ go through one description, so the two cannot disagree about what the
# model means. The *source* did not go through it: the emitter in write_project.cc wrote its
# own primary generator, and nothing compared the results. A generated generator that fired one
# of a two-beam model's sources for the entire run therefore passed every check there was - it
# built, it ran, and its dose was not zero.
#
# So: same model, same event count, both scorers, and they have to agree.
#
# The tolerance is on the *relative* difference and is deliberately loose. These are two
# independent Monte Carlo runs with different seeds; a few per cent apart at 200k events is
# ordinary. What this is looking for is the tens of per cent that a missing source, a missing
# volume, or a mis-emitted material produces - a systematic difference, not a statistical one.
param(
  [Parameter(Mandatory = $true)][string]$Builder,
  [Parameter(Mandatory = $true)][string]$Project,
  [double]$Tolerance = 0.06
)

if (-not (Test-Path -LiteralPath $Builder)) {
  Write-Output "FATAL: no builder log at $Builder"
  exit 1
}
if (-not (Test-Path -LiteralPath $Project)) {
  Write-Output "FATAL: no project log at $Project"
  exit 1
}

# "selftest: compare <name> <value> MeV over <n> events"
$want = @{}
foreach ($line in Get-Content -LiteralPath $Builder) {
  if ($line -match '^selftest: compare (\S+) ([0-9.eE+-]+) MeV over (\d+) events') {
    $want[$Matches[1]] = [double]$Matches[2]
  }
}
if ($want.Count -eq 0) {
  Write-Output "FATAL: the builder selftest logged no comparison line."
  Write-Output "       SelftestRunForComparison should print 'selftest: compare <scorer> ...'."
  exit 1
}

# " <name>            <value> MeV  rms <r>" from the generated RunAction.
$got = @{}
foreach ($line in Get-Content -LiteralPath $Project) {
  if ($line -match '^\s+(\S+)\s+([0-9.eE+-]+) MeV\s+rms') {
    # The generated RunAction also prints a "stepping" line, which is the SteppingAction's own
    # accumulation rather than a scorer. It is checked against its scorer inside the project;
    # here it would just be a duplicate name with no counterpart in the builder.
    if ($Matches[1] -ne 'stepping') { $got[$Matches[1]] = [double]$Matches[2] }
  }
}
if ($got.Count -eq 0) {
  Write-Output "FATAL: the generated project reported no scorer totals."
  exit 1
}

$bad = 0
foreach ($name in $want.Keys) {
  if (-not $got.ContainsKey($name)) {
    Write-Output "FATAL: the generated project has no scorer named '$name'."
    $bad = 1
    continue
  }
  $b = $want[$name]
  $p = $got[$name]
  if ($b -le 0) {
    Write-Output "FATAL: the builder measured $b MeV for '$name'; nothing to compare against."
    $bad = 1
    continue
  }
  $dev = [Math]::Abs($p - $b) / $b
  $pct = '{0:N2}' -f (100 * $dev)
  if ($dev -gt $Tolerance) {
    Write-Output "FATAL: '$name' disagrees. builder $b MeV, generated project $p MeV ($pct%)."
    Write-Output "       The saved project and the builder must describe the same model."
    $bad = 1
  } else {
    Write-Output "  $name  builder $b MeV, project $p MeV  ($pct%)"
  }
}
if ($bad -ne 0) { exit 1 }
Write-Output "the generated project agrees with the builder"
exit 0
