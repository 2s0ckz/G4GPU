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
# independent Monte Carlo runs with different seeds. What this is looking for is the tens of
# per cent that a missing source, a missing volume, or a mis-emitted material produces - a
# systematic difference, not a statistical one.
#
# A loose tolerance is only loose relative to the noise, and that was never measured until it
# had to be. The project prints its own rms: dose1 is +/-1.39% at a million events, so one side
# at 200000 is 3.12% and two independent sides are 4.4%. Against a 6% tolerance that is 1.36
# sigma - roughly a one-in-six chance of failing on any given pair, every time the pipeline
# ran. Every pass this check had ever given was luck rather than evidence, and the failure that
# finally exposed that was not a defect in either side. End to end the two differ by 6.4% at
# 200000 histories, 0.80% at a million and -0.08% at four million: 1/sqrt(N), which is what two
# independent Monte Carlos agreeing looks like.
#
# Both sides run a million now (build_all.bat, and kEvents in g4builder_panels.inc). That puts
# the noise near 1% and leaves this tolerance three sigma away, while still catching what it
# exists for - tens of per cent.
#
# The general rule, and the reason this paragraph exists: a threshold is not a number you pick,
# it is a number you compare against the spread of the thing being thresholded. See
# docs/RISK.md V10, and V1 for the last time a comparison had statistics too weak to mean
# anything.
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
