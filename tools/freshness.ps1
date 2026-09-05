# Prints "fresh" if the object is newer than every source it depends on, "stale" otherwise.
#
# This exists because the alternative was a guard on an environment variable - "if defined
# G4GPU_ENGINE_OBJ, skip" - which is a cache with no invalidation. It is right within one
# pipeline run and wrong the moment anyone sets the variable by hand to save four minutes, at
# which point the build silently links an engine compiled before the change under test. That
# happened; see docs/RISK.md S4.
#
# The dependency set is the translation unit itself plus every header under src\. Other .cu
# files are separate translation units and cannot affect this object, which matters: without
# that exclusion, editing the model builder would make the transport engine look stale and the
# check would save nothing. A header nobody includes still forces a rebuild, which is the
# harmless direction.
#
# Conservative in the only direction that matters: anything unexpected prints "stale".
# Rebuilding when it was not needed costs four minutes; not rebuilding when it was needed
# costs a wrong answer that looks right.
param([string]$Obj, [string]$Unit, [string]$SrcDir)
try {
  if (-not (Test-Path -LiteralPath $Obj)) { 'stale'; exit 0 }
  if (-not (Test-Path -LiteralPath $Unit)) { 'stale'; exit 0 }
  if (-not (Test-Path -LiteralPath $SrcDir)) { 'stale'; exit 0 }
  $o = (Get-Item -LiteralPath $Obj).LastWriteTimeUtc
  $t = (Get-Item -LiteralPath $Unit).LastWriteTimeUtc
  $h = Get-ChildItem -LiteralPath $SrcDir -Recurse -File |
       Where-Object { $_.Extension -in '.cuh', '.hh', '.h', '.inc' } |
       Measure-Object -Property LastWriteTimeUtc -Maximum
  if ($null -eq $h.Maximum) { 'stale' }
  elseif ($h.Maximum -gt $o -or $t -gt $o) { 'stale' }
  else { 'fresh' }
} catch { 'stale' }
