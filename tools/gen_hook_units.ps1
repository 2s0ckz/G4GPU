# Writes one translation unit per kernel for a project's own step hook - the eighteen stepping
# kernels, P15's five interaction kernels and P19's two drains - plus the header of
# `extern template` declarations that keeps every one of them out of the project's own object.
#
# WHY THIS EXISTS. docs/RISK.md V65 split the engine because one translation unit holding all
# eighteen stepping kernels takes 25 minutes and, with the Urban ion branch live, kills ptxas
# outright. A project with its own hook type has exactly the same file: `tests/test_custom_hook.cu`
# instantiates `TransportEngine<double, QualityFactorScoring>` and the eighteen launches inside
# BeamOn instantiate eighteen kernels into it. Splitting the ENGINE did nothing for that unit,
# because its kernels are a different specialisation - the hook type is a template parameter -
# so it died the same way, from the same cause, through a different door.
#
# AND IT SPLIT ONLY THE STEPPING KERNELS UNTIL P21, which is docs/RISK.md V210's last finding.
# The line below read `G4GPU_STEP_\w+` and nothing else, so the seven kernels that carry the
# hadronic models - `run_interaction` for FTFP, Bertini, the Binary cascade, the light-ion
# reaction and the at-rest chain, and `run_emextra_drain` for the photo- and lepto-nuclear
# models - had no unit and no declaration, and every one of them was instantiated in the
# project's own object: one module with all seven models in it, which is the arrangement
# docs/RISK.md V189 measured to be past ptxas for the ENGINE and split one model to a unit.
# For a hook project it compiled, slowly: 49.4 minutes of ptxas at 26.3 GB for either project on
# b937164's source under CUDA 12.9 (V210 measured 44.8 at 20.7 under P19), 50.5 and 49.8 minutes of
# nvcc in that commit's own build_all.
# And the module was not just slow. ptxas allocates per module, so the two drains came out at
# 18,672 and 19,424 bytes of frame where the engine's own one-kernel units give 11,936 and
# 4,112 - past the stepping kernels' 16,384-byte stack, so the driver raised the device stack for
# the whole card at the first drain launch of every run (docs/RISK.md V196), and both projects
# printed that they had. Every `extern template` line is a kernel now, whatever its macro, so a
# kernel family added to the engine's block is split for a hook the day it is added.
#
# THE KERNEL LIST IS READ OUT OF transport_run_impl.cuh RATHER THAN WRITTEN HERE, and that is
# the point of doing this in a script at all. The stock engine's `extern template` block is the
# one place that knows which kernels exist; a copy of that list here would be right on the day
# it was written and silently wrong the first time a species was added - the new kernel would
# have no unit, no declaration, and would quietly go back to being instantiated in the project's
# own object, which is the failure this whole mechanism exists to prevent. So the block is
# parsed, and the only substitution is the hook type for StepTap<double>. And the parse is
# COUNTED against every `extern template` line in the header: a declaration this script could
# not read would otherwise be dropped without a word - which is exactly how the seven kernels
# above went unsplit, read past by a pattern that was right the day it was written.
#
# THE UNIT NAMES SAY WHICH PASS COMPILES THEM. A stepping kernel's unit is `hook_<kind>_<arg>.cu`,
# as before. Every other kernel's is `hook_int_<kind>_<arg>.cu`, and build_hook_engine.bat
# compiles `hook_int_*` ONE AT A TIME before the stepping units, as build_engine.bat compiles the
# engine's `transport_run_int_*`: one of them alone peaks at 8 to 25 GB of ptxas (docs/RISK.md
# V189, V222), so six at once is past this machine. A family that is not a stepping kernel is
# treated as heavy without being named here, which is the direction that cannot run the machine
# out of memory.
#
#   -ImplHeader   src\host\transport_run_impl.cuh - the source of truth for the kernel list
#   -HookHeader   the project's header that DEFINES the hook class (included by every unit)
#   -HookType     the type name, exactly as the project writes it
#   -OutDir       where to write the units and hook_kernels.cuh
param(
  [Parameter(Mandatory = $true)][string]$ImplHeader,
  [Parameter(Mandatory = $true)][string]$HookHeader,
  [Parameter(Mandatory = $true)][string]$HookType,
  [Parameter(Mandatory = $true)][string]$OutDir
)
$ErrorActionPreference = 'Stop'

if (-not (Test-Path -LiteralPath $ImplHeader)) { Write-Output "FATAL: no $ImplHeader"; exit 1 }
if (-not (Test-Path -LiteralPath $HookHeader)) { Write-Output "FATAL: no $HookHeader"; exit 1 }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

# The hook header is included by absolute path. The units live in out\, not beside the project,
# and giving build_engine_unit.bat a second -I for every hook would mean a different compile
# line per project; an absolute path in a generated file that is regenerated on every build has
# no such cost and cannot resolve to the wrong header.
$hookAbs = (Resolve-Path -LiteralPath $HookHeader).Path -replace '\\', '/'

$kernels = @()
$declared = 0
foreach ($line in Get-Content -LiteralPath $ImplHeader) {
  # Every declaration, parsed or not: the count the parse is held to. A comment that quotes one
  # starts with `//` and is not counted.
  if ($line -match '^\s*extern\s+template\s') { $declared++ }
  if ($line -match '^\s*extern template\s+(G4GPU_\w+)\((.*)\);\s*$') {
    $macro = $Matches[1]
    $karg  = $Matches[2]
    # The stock hook is the last argument. Replacing it by name rather than by position keeps
    # this honest if an argument is ever added in front of it.
    if ($karg -notmatch 'StepTap<\s*double\s*>') {
      Write-Output "FATAL: '$line' does not name StepTap<double>; this script cannot rewrite it."
      exit 1
    }
    $hookArgs = $karg -replace 'StepTap<\s*double\s*>', $HookType
    # A unit name that says what is in it: gamma, lepton_true, hadron_kGenericIon,
    # int_interaction_kBinary, int_emx_drain_kPhotoNuclear. The argument is the last component of
    # its qualified name, so `ParticleType::kProton` and `had::InteractionBucket::kFtfp` read
    # alike and the stepping units keep the names they had before P21.
    $rest = ($karg -replace ',?\s*StepTap<\s*double\s*>\s*$', '').Trim()
    $rest = (($rest -split '::')[-1]) -replace '[^A-Za-z0-9_]', ''
    $heavy = ($macro -notlike 'G4GPU_STEP_*')
    $tag = if ($heavy) { 'int_' + ($macro -replace '^G4GPU_', '').ToLower() }
           else { ($macro -replace '^G4GPU_STEP_', '').ToLower() }
    $name = if ($rest -eq '') { $tag } else { ($tag + '_' + $rest) }
    $kernels += [pscustomobject]@{ Name = $name; Macro = $macro; Args = $hookArgs; Heavy = $heavy }
  }
}

# A list that came out short is the failure mode this script must not have: fewer units than
# kernels links cleanly for the ones it has and fails at link time for the rest, which is
# survivable - but an EMPTY list would emit no declarations at all and put every kernel back in
# the project's object, which is the 25-minute unit this exists to abolish, silently. And a list
# one SHORT puts one kernel back, as silently: P21 found seven that way.
if ($kernels.Count -lt 10) {
  Write-Output "FATAL: parsed only $($kernels.Count) kernels out of $ImplHeader."
  Write-Output "       The 'extern template G4GPU_*(...);' block is what this reads."
  exit 1
}
if ($kernels.Count -ne $declared) {
  Write-Output "FATAL: $ImplHeader declares $declared kernels extern and this script parsed"
  Write-Output "       $($kernels.Count) of them. A declaration it cannot read would be compiled"
  Write-Output "       into the project's own object - see the header of this script."
  exit 1
}
$names = $kernels | ForEach-Object { $_.Name }
if (($names | Sort-Object -Unique).Count -ne $kernels.Count) {
  Write-Output "FATAL: two kernels map to one unit name; the unit naming above needs extending."
  exit 1
}

foreach ($k in $kernels) {
  $why = if ($k.Heavy) {
@"
// One of the kernels that carry the hadronic models, so build_hook_engine.bat compiles it ONE AT
// A TIME before the stepping units, as build_engine.bat compiles the engine's own
// transport_run_int_*.cu: one model to a unit is docs/RISK.md V189's rule, and these units peak
// at 8 to 25 GB of ptxas each (V222). Before P21 all of them were compiled into the project's own
// object together, which is the module V189 measured to be past ptxas for the engine.
"@
  } else {
@"
// One stepping kernel for one project hook type, in a translation unit of its own, which is
// the shape docs/RISK.md V65 measured to be the only one ptxas compiles for every arrangement
// of this physics.
"@
  }
  $body = @"
// GENERATED by tools\gen_hook_units.ps1 - do not edit; every build overwrites it.
//
$why
// The project's own .cu declares the same kernel `extern template` (see hook_kernels.cuh beside
// this file) and therefore no longer instantiates it.
#include "$hookAbs"
#include "host/transport_run_impl.cuh"

namespace g4gpu::host {

template $($k.Macro)($($k.Args));

}  // namespace g4gpu::host
"@
  $unit = Join-Path $OutDir ("hook_" + $k.Name + ".cu")
  Set-Content -LiteralPath $unit -Value $body -Encoding ASCII
}

$decls = ($kernels | ForEach-Object { "extern template $($_.Macro)($($_.Args));" }) -join "`n"
$nheavy = @($kernels | Where-Object { $_.Heavy }).Count
$header = @"
// GENERATED by tools\gen_hook_units.ps1 - do not edit; every build overwrites it.
//
// The declarations that keep this project's kernels out of the object that launches them: the
// $($kernels.Count - $nheavy) stepping kernels and the $nheavy that carry the hadronic models. Include it AFTER
// host/transport_run_impl.cuh and after the hook class is defined: it is written against that
// header's G4GPU_* macros, so that a parameter added to a kernel cannot leave a declaration here
// disagreeing with the definition in the unit beside it.
//
// Without this include the project compiles - and instantiates all $($kernels.Count) kernels into its own
// translation unit, which takes 25 minutes when it works and kills ptxas when it does not
// (docs/RISK.md V65), and puts the models' kernels in one module whose frames are not the
// engine's (V210).
#ifndef G4GPU_STEP_GAMMA
#error "include host/transport_run_impl.cuh before hook_kernels.cuh"
#endif

namespace g4gpu::host {

$decls

}  // namespace g4gpu::host
"@
Set-Content -LiteralPath (Join-Path $OutDir 'hook_kernels.cuh') -Value $header -Encoding ASCII
Write-Output "$($kernels.Count)"
