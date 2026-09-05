# Compares two proton depth-dose curves - this port's and the real Geant4's - and scores the
# four things a proton calculation is judged on.
#
#   compare_depth.ps1 -Reference out\g4_depth.csv -Port out\port_depth.csv [-Limits ...]
#
# Both files come from ref/proton/proton_depth.cc, compiled twice: once against Geant4 11.1.1
# and once against this port's headers. One source, so the geometry, the beam and the binning
# cannot differ; what differs is the transport, which is the point.
#
# The four numbers, and why each is separate rather than one aggregate:
#
#   total       energy in vs energy deposited. A conservation check, not a physics one, and it
#               should be exact on both sides - the phantom is deeper than the range.
#   plateau     the integral over the entrance half, which is the restricted stopping power
#               and nothing else. Insensitive to where the peak is.
#   R80         the distal 80% depth, the standard proton range metric and the number a
#               treatment plan is built on. Sensitive to the range table and to anything that
#               biases the energy loss.
#   width       the distal 80%-to-20% falloff, which is range straggling and therefore a test
#               of G4UniversalFluctuation alone. A model with no fluctuations gets R80 right
#               and this badly wrong, which is exactly what happened before one was written.
#
# Aggregating those into a single "agrees to X%" would let a right answer for the wrong reason
# through: a stopping power 1% high and a range table 1% long cancel in the plateau and add in
# R80.
param(
  [Parameter(Mandatory=$true)][string]$Reference,
  [Parameter(Mandatory=$true)][string]$Port,
  [double]$PlateauLimit = 0.01,   # fraction
  [double]$R80Limit     = 0.5,    # mm
  [double]$WidthLimit   = 0.15,   # mm
  [double]$TotalLimit   = 1e-6    # fraction
)

function Read-Curve([string]$path) {
  if (-not (Test-Path $path)) { throw "no such file: $path" }
  $lines = Get-Content $path
  $head = $lines[0]
  $z = New-Object System.Collections.Generic.List[double]
  $d = New-Object System.Collections.Generic.List[double]
  foreach ($l in $lines[2..($lines.Count-1)]) {
    if ($l -notmatch '^\d') { continue }
    $f = $l.Split(',')
    $z.Add([double]$f[1])
    $d.Add([double]$f[3])
  }
  $slab = 0.5
  if ($head -match 'slab_mm=([0-9.eE+-]+)') { $slab = [double]$Matches[1] }
  $ev = 0; $en = 0
  if ($head -match 'events=(\d+)') { $ev = [int]$Matches[1] }
  if ($head -match 'energy_MeV=([0-9.eE+-]+)') { $en = [double]$Matches[1] }
  # Bin centres, not lower edges: R80 is read off an interpolated curve and a half-bin offset
  # would show up as a range difference that is really a labelling difference.
  $zc = New-Object System.Collections.Generic.List[double]
  foreach ($v in $z) { $zc.Add($v + 0.5 * $slab) }
  [pscustomobject]@{ Z = $zc; D = $d; Slab = $slab; Events = $ev; Energy = $en }
}

# Depth at which the curve falls through $frac of its peak, on the distal side, linearly
# interpolated between the two bins that straddle it.
function Distal([object]$c, [double]$frac) {
  $pk = 0; $pi = 0
  for ($i = 0; $i -lt $c.D.Count; $i++) { if ($c.D[$i] -gt $pk) { $pk = $c.D[$i]; $pi = $i } }
  $t = $frac * $pk
  for ($i = $pi; $i -lt $c.D.Count; $i++) {
    if ($c.D[$i] -lt $t) {
      $z0 = $c.Z[$i-1]; $z1 = $c.Z[$i]; $d0 = $c.D[$i-1]; $d1 = $c.D[$i]
      return $z0 + ($d0 - $t) / ($d0 - $d1) * ($z1 - $z0)
    }
  }
  return [double]::NaN
}

$ref = Read-Curve $Reference
$prt = Read-Curve $Port

# The beam and the binning must match; the event count need not. The reference is a checked-in
# oracle file with high statistics (ref/oracle/proton_depth.csv), and the pipeline runs a short
# job against it - so every dose below is divided by the event count and compared per proton.
if ($ref.Slab -ne $prt.Slab -or $ref.Energy -ne $prt.Energy) {
  Write-Host "FAIL: the two curves describe different runs"
  Write-Host ("  reference: {0} MeV, {1} mm slabs" -f $ref.Energy, $ref.Slab)
  Write-Host ("  port:      {0} MeV, {1} mm slabs" -f $prt.Energy, $prt.Slab)
  exit 1
}
if ($ref.Z.Count -ne $prt.Z.Count) {
  Write-Host ("FAIL: {0} bins in the reference, {1} in the port run" -f $ref.Z.Count, $prt.Z.Count)
  exit 1
}

$fails = 0
Write-Host ("proton depth-dose: {0} events of {1} MeV in water, {2} mm slabs" -f $ref.Events, $ref.Energy, $ref.Slab)

# ---- total
$tr = ($ref.D | Measure-Object -Sum).Sum
$tp = ($prt.D | Measure-Object -Sum).Sum
foreach ($p in @(@("Geant4", $tr, $ref.Events), @("port", $tp, $prt.Events))) {
  $want = $p[2] * $ref.Energy
  $dev = [Math]::Abs($p[1] / $want - 1)
  Write-Host ("  total {0,-8} {1,14:g8} MeV of {2:g8} in  ({3:p4})" -f $p[0], $p[1], $want, ($p[1]/$want))
  if ($dev -gt $TotalLimit) {
    Write-Host ("  FAIL: {0} did not deposit the beam energy - {1:p4} of it" -f $p[0], ($p[1]/$want))
    $fails++
  }
}

# ---- plateau: the entrance half, well proximal of the peak
$half = 0
for ($i = 0; $i -lt $ref.Z.Count; $i++) { if ($ref.Z[$i] -lt 0.6 * $ref.Z[$ref.Z.Count-1]) { $half = $i } }
$pr = 0.0; $pp = 0.0
for ($i = 0; $i -le $half; $i++) { $pr += $ref.D[$i]; $pp += $prt.D[$i] }
# Per proton on both sides, so the reference's event count need not match the port run's.
$pdev = ($pp / $prt.Events) / ($pr / $ref.Events) - 1
Write-Host ("  plateau (0-{0:n1} mm)  port/G4 per proton = {1:n5}  ({2:p3})" -f $ref.Z[$half], (1 + $pdev), $pdev)
if ([Math]::Abs($pdev) -gt $PlateauLimit) {
  Write-Host ("  FAIL: plateau dose off by {0:p3}, limit {1:p3}" -f $pdev, $PlateauLimit)
  $fails++
}

# ---- range and falloff
$r80r = Distal $ref 0.8; $r80p = Distal $prt 0.8
$r20r = Distal $ref 0.2; $r20p = Distal $prt 0.2
$wr = $r20r - $r80r; $wp = $r20p - $r80p
Write-Host ("  R80          G4 {0,8:n3} mm   port {1,8:n3} mm   diff {2,7:n3} mm" -f $r80r, $r80p, ($r80p-$r80r))
Write-Host ("  80-20 width  G4 {0,8:n3} mm   port {1,8:n3} mm   diff {2,7:n3} mm" -f $wr, $wp, ($wp-$wr))
if ([Math]::Abs($r80p - $r80r) -gt $R80Limit) {
  Write-Host ("  FAIL: R80 off by {0:n3} mm, limit {1:n2} mm" -f ($r80p-$r80r), $R80Limit)
  $fails++
}
if ([Math]::Abs($wp - $wr) -gt $WidthLimit) {
  Write-Host ("  FAIL: distal falloff width off by {0:n3} mm, limit {1:n2} mm" -f ($wp-$wr), $WidthLimit)
  $fails++
}

if ($fails -gt 0) { Write-Host "FAILED ($fails)"; exit 1 }
Write-Host "OK"
exit 0
