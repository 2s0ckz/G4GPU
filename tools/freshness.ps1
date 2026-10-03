# Is a build product up to date with what the compiler read to make it?
#
#   tools\freshness.ps1 -Target out\X.obj -Deps out\X.d -Recipe out\X.recipe -RecipeNow "<flags>"
#       prints "fresh", or "stale: <reason>" on one line; always exits 0 (the caller reads the text)
#   tools\freshness.ps1 -Recipe out\X.recipe -RecipeNow "<flags>" -Stamp
#       writes the recipe file after a successful compile; prints nothing
#
# THE DEPENDENCY FILE IS NVCC'S, NOT A GUESS. `-MD -MF <file>` makes nvcc write, in make syntax,
# every file it read while compiling the unit: the source, every header reached through every
# include, the toolkit's and the host compiler's own. So the question "does this object need
# recompiling" has an exact answer - is any of THOSE files newer than the object, or missing -
# and this script asks exactly that. The version before 2026-10-02 (docs/RISK.md V221) took the
# newest header anywhere under src as the stale test, which rebuilt all twenty-three engine
# units for a one-line change to one cascade file, and build_all.bat compiled every test every
# time because it had no dependency information at all.
#
# THE RECIPE IS PART OF THE PRODUCT. An object compiled with -arch=sm_86 is not the one a
# -gencode sm_70 build wants, and an object from CUDA 11.6 is not one from 12.9 (docs/RISK.md
# V203), and neither shows in a timestamp. So the caller passes what it is about to compile with
# (the arch flags; anything else that changes the output), this script appends which nvcc is on
# the PATH, and the product is stale whenever that string differs from the one stored beside it
# when it was built. A missing recipe is stale: a product built before this script existed has
# no claim to being current.
#
# WHAT IS DELIBERATELY NOT HERE. No "anything under src newer than the product" fallback, no
# environment variable that says "trust it" - docs/RISK.md S4 is what an un-invalidated cache did
# to this project once. A dependency file that cannot be parsed is stale, a dependency that no
# longer exists is stale, and a product with no dependency file is stale. Every doubt is a
# recompile, never a skipped one.
#
# THE OLDER QUESTION IS STILL ANSWERED. build_vis.bat and build_hook_engine.bat ask the form this
# script had before - `-Obj <product> -Unit <source> -SrcDir <dir>`: is the product older than the
# source or than the newest header under the directory? - and they keep getting that answer until
# they too write dependency files. It over-rebuilds and never under-rebuilds, which is the safe
# side of wrong.
param(
  [string]$Target,
  [string]$Deps,
  [string]$Recipe,
  [string]$RecipeNow = "",
  [switch]$Stamp,
  [string]$Obj,
  [string]$Unit,
  [string]$SrcDir
)
$ErrorActionPreference = "Stop"

if ($Obj) {
  try {
    if (-not (Test-Path -LiteralPath $Obj)) { 'stale'; exit 0 }
    if (-not (Test-Path -LiteralPath $Unit)) { 'stale'; exit 0 }
    if (-not (Test-Path -LiteralPath $SrcDir)) { 'stale'; exit 0 }
    $o = (Get-Item -LiteralPath $Obj).LastWriteTimeUtc
    $u = (Get-Item -LiteralPath $Unit).LastWriteTimeUtc
    $h = Get-ChildItem -LiteralPath $SrcDir -Recurse -File |
         Where-Object { $_.Extension -in '.cuh', '.hh', '.h', '.inc' } |
         Measure-Object -Property LastWriteTimeUtc -Maximum
    if ($null -eq $h.Maximum) { 'stale' }
    elseif ($h.Maximum -gt $o -or $u -gt $o) { 'stale' }
    else { 'fresh' }
  } catch { 'stale' }
  exit 0
}

function Get-RecipeString([string]$now) {
  $nvcc = Get-Command nvcc -ErrorAction SilentlyContinue
  $path = if ($nvcc) { $nvcc.Source } else { "?" }
  return "$now | nvcc=$path"
}

if ($Stamp) {
  if (-not $Recipe) { Write-Output "stale: -Stamp needs -Recipe"; exit 0 }
  $dir = Split-Path -Parent $Recipe
  if ($dir -and -not (Test-Path -LiteralPath $dir)) { New-Item -ItemType Directory -Force $dir | Out-Null }
  Set-Content -LiteralPath $Recipe -Value (Get-RecipeString $RecipeNow) -Encoding ASCII
  exit 0
}

try {
  if (-not $Target -or -not (Test-Path -LiteralPath $Target)) { Write-Output "stale: no product"; exit 0 }
  if (-not $Deps -or -not (Test-Path -LiteralPath $Deps)) { Write-Output "stale: no dependency file"; exit 0 }
  if (-not $Recipe -or -not (Test-Path -LiteralPath $Recipe)) { Write-Output "stale: no recipe"; exit 0 }
  $stored = (Get-Content -LiteralPath $Recipe -Raw).Trim()
  $now = (Get-RecipeString $RecipeNow).Trim()
  if ($stored -ne $now) { Write-Output "stale: recipe changed"; exit 0 }

  $built = (Get-Item -LiteralPath $Target).LastWriteTimeUtc
  # make syntax: "target : dep dep \" with continuation lines, spaces in a path escaped as "\ ",
  # forward slashes throughout. The target is the exe or obj and carries no ": ", so the first
  # ": " ends it; a drive letter's colon is followed by a slash, not a space.
  $text = Get-Content -LiteralPath $Deps -Raw
  $text = $text -replace "\\\r?\n", " "
  $cut = $text.IndexOf(": ")
  if ($cut -lt 0) { Write-Output "stale: dependency file unreadable"; exit 0 }
  $sentinel = [string][char]1
  $body = $text.Substring($cut + 2) -replace '\\ ', $sentinel
  $root = Split-Path -Parent (Resolve-Path -LiteralPath $Deps).Path
  foreach ($tok in ($body -split '\s+')) {
    if (-not $tok) { continue }
    $p = ($tok -replace [regex]::Escape($sentinel), ' ') -replace '/', '\'
    if (-not [System.IO.Path]::IsPathRooted($p)) { $p = Join-Path (Get-Location).Path $p }
    if (-not (Test-Path -LiteralPath $p)) { Write-Output ("stale: missing " + ($p -replace '[()]', '')); exit 0 }
    if ((Get-Item -LiteralPath $p).LastWriteTimeUtc -gt $built) { Write-Output ("stale: newer " + ($p -replace '[()]', '')); exit 0 }
  }
  Write-Output "fresh"
} catch {
  Write-Output ("stale: " + (($_.Exception.Message) -replace '[()\r\n]', ' '))
}
exit 0
