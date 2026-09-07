# `G4Step` and `G4Track` on the device: what is available, what is missing, and what it cost

A `G4VUserDeviceSteppingAction` is handed a `G4DeviceStep`, this port's answer to Geant4's
[`G4Step`](https://apc.u-paris.fr/~franco/g4doxy/html/classG4Step.html), which reaches a
`G4DeviceStepPoint` and a `G4DeviceTrack` — the answers to `G4StepPoint` and
[`G4Track`](https://apc.u-paris.fr/~franco/g4doxy/html/classG4Track.html).

Two structural differences to know before the tables, both forced by the machine:

**Objects come back by value, not by pointer.** A pointer would have to point at something, and
there is no per-step object in device memory to point at — the values live in the kernel's
registers. Both spellings compile anyway, because the returned objects define `operator->`, so
a line ported from a Geant4 project needs no edit:

```cpp
const G4double ke = step.GetPreStepPoint()->GetKineticEnergy();   // works
const G4double kd = step.GetPreStepPoint().GetKineticEnergy();    // also works
```

**`GetTrack()` is a handle onto the live track, so its setters are real.** `SetTrackStatus`,
`SetWeight`, `SetPolarization`, `SetKineticEnergy` and the rest write through to the track the
transport is stepping, exactly as they would in Geant4 where `GetTrack()` hands back a mutable
`G4Track*`. Const on the step does not propagate through the pointer, deliberately: Geant4's
`UserSteppingAction` receives a `const G4Step*` and can still call
`step->GetTrack()->SetTrackStatus(fStopAndKill)`, which is the single most common reason to
write a stepping action at all.

---

## `G4Step`

| method | status | note |
|---|---|---|
| `GetTotalEnergyDeposit()` | **yes** | |
| `GetNonIonizingEnergyDeposit()` | **yes** | nuclear recoil; part of the deposit, not additional to it |
| `GetStepLength()` | **yes** | the **true** path length, as in Geant4 — not `\|post − pre\|` |
| `GetDeltaPosition()` | **yes** | |
| `GetDeltaEnergy()` | **yes** | |
| `GetDeltaTime()` | **yes** | derived: the true path over the pre-step velocity, which is what advanced the clock |
| `GetPreStepPoint()`, `GetPostStepPoint()` | **yes** | by value; see above |
| `GetTrack()` | **yes** | a live handle; see above |
| `GetNumberOfSecondariesInCurrentStep()` | **yes** | free — the emitter counts them anyway, to key their RNG streams |
| `IsFirstStepInVolume()`, `IsLastStepInVolume()` | **yes** | |
| `GetSecondaryInCurrentStep()` | **yes** | the real secondaries, not copies. **Newest first**, where Geant4's vector is in creation order — see below |
| `GetSecondary()`, `NewSecondaryVector()` | no | those return the *tracking manager's* secondary vector, accumulated over a track's whole history and owned by `G4TrackingManager`. There is no tracking manager here and no such accumulation: a secondary is appended to the pool and stepped by the next iteration |
| `GetControlFlag()`, `SetControlFlag()` | no | there is no stepping-control mechanism to steer |
| `SetTotalEnergyDeposit()`, `SetStepLength()`, `SetTrack()`, `InitializeStep()`, `UpdateTrack()`, `CopyPostToPreStepPoint()`, `AddTotalEnergyDeposit()`, `Reset*()`, `Set*StepFlag()` | deliberately absent | Geant4's framework API. `UserSteppingAction` gets a `const G4Step*` and cannot call them; they exist for `G4SteppingManager` to build and recycle one heap-allocated step across millions of steps. There is no persistent step object here to build. Worse, a `SetTotalEnergyDeposit` would compile, look like it worked, and change nothing — the deposit has already reached the scorer by the time a hook runs |
| `GetDeltaMomentum()` | deliberately absent | deprecated in Geant4, and derivable from the two step points |
| `CreatePolyline()`, `GetPointerToVectorOfAuxiliaryPoints()` | not meaningful | auxiliary points exist for curved trajectories in a magnetic field. There is no field here, so every step is a straight line. `vis::TrajectoryBuffer` already does the visualization job |

## `G4StepPoint`

| method | status | note |
|---|---|---|
| `GetPosition()`, `GetMomentumDirection()`, `GetMomentum()` | **yes** | |
| `GetKineticEnergy()`, `GetTotalEnergy()`, `GetMass()`, `GetCharge()` | **yes** | |
| `GetBeta()` | **yes** | exactly 1 for a massless particle, not 0/0 |
| `GetGamma()` | **yes** | 0 for a massless particle. Geant4 returns `DBL_MAX`; 0 is returned here so a value that is not meaningful cannot silently poison an average it is folded into |
| `GetStepStatus()` | **yes** | `fGeomBoundary`, `fWorldBoundary`, `fPostStepDoItProc`, `fAlongStepDoItProc`, `fStopAndKill` |
| `GetProcessDefinedStep()` | **yes** | a `G4ProcessId` enum, not a `G4VProcess*` — see below |
| `GetSafety()` | **yes** | pre-step; negative when the stepper had no reason to compute one |
| `GetMaterial()` | **yes** | a material **index**, on the pre-step point |
| `GetPhysicalVolume()` → `GetVolume()` | **yes** | a volume **index** |
| `GetGlobalTime()`, `GetLocalTime()`, `GetProperTime()` | via `GetTrack()` | the clocks live on the track; see **Time** |
| `GetTouchable()`, `GetTouchableHandle()` | not meaningful | a touchable is a host object holding a volume hierarchy. Volumes here are indices into a flat array |
| `GetMaterialCutsCouple()` | not meaningful | cuts are per material; the material index carries the same information |
| `GetSensitiveDetector()` | not meaningful | `step.GetScoreSlot()` is the analogue — the scorer the pre-step volume belongs to |
| `GetPolarization()` | via `GetTrack()` | carried on the track |
| `GetWeight()` | via `GetTrack()` | carried on the track |

### `GetProcessDefinedStep()` is an enum

Geant4 hands back a `G4VProcess*` and you compare `GetProcessName()`. A process object is a host
object, so what is returned here is the identity alone. None of it is inferred after the fact —
the steppers already decide which process won the competition for the step, and the values were
simply being discarded.

`g4dose -verify-step-hook` asserts that **no step reports `fNotDefined`**, which turns "did every
branch get annotated" into something the machine answers rather than something someone
remembered. The enum also carries the processes this port does not have yet — hadronic elastic
and inelastic, decay, photonuclear, capture, Cerenkov and the rest — so a process written
tomorrow has a value to report on the day it is written, rather than needing the enum widened
first. `fNotDefined` means one thing only: a branch nobody annotated.

## `G4Track`

| method | status | note |
|---|---|---|
| `GetTrackID()` | **yes** | the RNG key. Stable across a run, but not a small dense integer the way Geant4's is |
| `GetParentID()` | **yes** | the parent's key; 0 for a primary, as in Geant4 |
| `GetDefinition()`, `GetParticleDefinition()` | **yes** | a `ParticleType` |
| `GetKineticEnergy()` / `SetKineticEnergy()` | **yes** | |
| `GetTotalEnergy()`, `GetMomentum()`, `GetMass()`, `GetCharge()` | **yes** | |
| `GetPosition()` / `SetPosition()` | **yes** | |
| `GetMomentumDirection()` / `SetMomentumDirection()` | **yes** | |
| `GetVelocity()`, `CalculateVelocity()` | **yes** | `c_light * GetBeta()`, as Geant4 |
| `GetGlobalTime()` / `SetGlobalTime()` | **yes** | see **Time** |
| `GetLocalTime()` / `SetLocalTime()` | **yes** | |
| `GetProperTime()` / `SetProperTime()` | **yes** | |
| `GetTrackLength()`, `AddTrackLength()` | **yes** | |
| `GetVertexPosition()` / `SetVertexPosition()` | **yes** | |
| `GetVertexMomentumDirection()` / `Set…` | **yes** | |
| `GetVertexKineticEnergy()` / `Set…` | **yes** | |
| `GetLogicalVolumeAtVertex()` → `GetVolumeAtVertex()` | **yes** | a volume **index** |
| `GetCreatorProcess()` / `SetCreatorProcess()` | **yes** | a `G4ProcessId` |
| `GetCurrentStepNumber()` | **yes** | 1 for a track's first step, as in Geant4 |
| `GetTrackStatus()` / `SetTrackStatus()` | **yes** | see **Killing a track** |
| `IsBelowThreshold()` / `SetBelowThresholdFlag()` | **yes** | nothing in the transport sets it; a stepping action may, and it is then carried across steps |
| `IsGoodForTracking()` / `SetGoodForTrackingFlag()` | **yes** | same |
| `GetWeight()` / `SetWeight()` | **yes** | there is no variance reduction, so nothing but a user changes it |
| `GetPolarization()` / `SetPolarization()` | **yes** | no process here produces or consumes it, so it is user-owned per-track state |
| `GetUserInformation()` / `SetUserInformation()` | **as a POD slot** | `G4VUserTrackInformation*` is a host pointer and cannot exist. The *capability* — attach your own data to a track, read it back next step — is meaningful, so what is offered is an integer slot the engine never touches |
| `GetVolume()` | **as an index** | Geant4 returns a pointer into a host hierarchy |
| `GetNextVolume()`, `GetMaterial()` | on the **step**, not the track | a track handle does not know where it is going or what it is in; the step does. `step.GetPostStepPoint()->GetVolume()` and `step.GetPreStepPoint()->GetMaterial()` |
| `GetTouchable()`, `GetNextTouchable()`, `GetOriginTouchable()` | not meaningful | host objects |
| `GetDynamicParticle()`, `GetStep()` | not meaningful | those objects do not exist; their contents are on this handle and on the step that owns it |
| `GetCreatorModelID()`, `GetCreatorModelName()` | not meaningful | Geant4's model registry has no analogue here |
| `GetParentResonance*()` | not meaningful | no resonance decay exists yet to set one |
| `CalculateVelocityForOpticalPhoton()`, `UseGivenVelocity()`, `SetVelocity()` | not meaningful | no optical photons |
| `CopyTrackInfo()`, constructors, `operator=`, `operator==` | implicit | the handle is an aggregate |

---

## Time

**Implemented, and transcribed rather than derived.** The formula is
`G4Transportation::AlongStepDoIt`'s, read out of the Geant4 11.1.1 source on this machine:

```cpp
initialVelocity = stepData.GetPreStepPoint()->GetVelocity();
if (initialVelocity > 0.0) deltaTime = stepLength / initialVelocity;
fCandidateEndGlobalTime = startTime + deltaTime;
ProposeLocalTime(track.GetLocalTime() + deltaTime);
deltaProperTime = deltaTime * (restMass / track.GetTotalEnergy());
```

Three things in that are easy to get wrong by guessing and all three matter: the velocity is the
**pre-step** one and not a mean over the step; `stepLength` is the **true** path length, so
multiple scattering is already accounted for; and proper time is `dt·m/E` against the pre-step
energy, which for a massless particle is exactly zero rather than a division by zero. The
velocity is `c_light * G4DynamicParticle::GetBeta()`, whose beta this port already transcribed
for `G4IonFluctuations` and which is now defined once and shared.

A secondary inherits its parent's global time at the moment of creation and starts its local and
proper clocks at zero. A primary starts at the vertex time the generator asked for.

## Killing a track

`GetTrack()->SetTrackStatus(...)` is honoured by the kernel the instant the hook returns. The
enum is Geant4's complete `G4TrackStatus`, not the subset this transport can act on:

| status | what happens |
|---|---|
| `fAlive`, `fStopButAlive` | the physics decision stands |
| `fStopAndKill` | the track is not requeued |
| `fKillTrackAndSecondaries` | the track is not requeued, **but its secondaries survive** — they reached their buffers before the hook ran |
| `fSuspend`, `fPostponeToNextEvent` | cannot be honoured: every track in flight is in a species buffer being stepped in lockstep, and there is nowhere to set one aside. The track is killed |

The last two, and the secondaries half of the third, are **counted** in
`RunStats::unsupported_track_status`. A run that asked for something it did not get says so,
rather than leaving someone to find out from a dose that is quietly wrong.

## The secondaries of a step

`GetSecondaryInCurrentStep()` returns a `SecondaryList`, and
`GetNumberOfSecondariesInCurrentStep()` returns the count — the same split Geant4 has, so a hook
that only wants to know how many touches none of the rest.

```cpp
const int n = step.GetNumberOfSecondariesInCurrentStep();
for (auto it = step.GetSecondaryInCurrentStep().begin(); it.valid(); it.advance()) {
  const G4DeviceTrack sec = it.get();          // a handle onto the real secondary
  if (sec.GetDefinition() == ParticleType::kElectron) { /* ... */ }
}
```

Each element is the track the transport will step next iteration, read out of the pool — not a
copy made for the hook's benefit.

**There is no per-step limit.** The list is a backward-linked chain: each secondary stores the
index of the previous one from the same step, so a step's list costs one arena slot and one
`int` per secondary, and no number was ever chosen in advance about how many secondaries a
process may produce. That matters more than it sounds. The alternative — an array per step —
needs a capacity, and a capacity picked from what today's EM physics happens to produce is a
silent truncation waiting for the first model that fragments a nucleus.

Two consequences, both named rather than hidden:

- **Order is newest first.** Geant4's vector is in creation order. Reversing it would need
  somewhere to put the reversed copy, which is an array with a capacity again.
- **The arena is a fixed pool** — sized by `SetSecondaryArenaCapacity`, defaulting to the track
  pool. It is the *chain* that is unbounded per step, not the total across a launch. When the
  pool fills, `SecondaryArena::overflow` counts what did not fit, and
  `GetNumberOfSecondariesInCurrentStep()` still returns the true number the step made — so a
  walk that comes up short against the count is visible to the hook itself rather than looking
  like a step that made fewer particles.

`tests/test_custom_hook.cu` walks the chain over a real run and checks the walk against the
count: 3436 reached, 3436 reported.

## What it cost

Measured, interleaved, six runs of 2M B1 events each way:

```
HEAD   823.7  824.1  824.8  827.3  832.4  862.9 ms   median 824.8
full   889.9  890.3  890.7  892.6  893.4  895.1 ms   median 890.7
```

**+8.0%** — 2.30e6 → 2.13e6 events/s. The two sets do not overlap, so this is not noise. The
dose is bit-identical across three seeds with identical step counts: the physics is untouched
and this is pure bandwidth. The block is 136 bytes, taking a track slot from 100 to 236 - 96 to
232 on the day this was measured, the four bytes since being the species field the pooled
scheduler added - and every one of those bytes
is loaded and stored for every track on every step, whether or not a hook reads it.

There is a known way to get most of it back, not yet done. Ten of the fifteen added fields are
immutable for a track's whole life — parent, vertex position, direction, energy and volume,
creator process — and five more change only if a user writes them. None of them need to ride the
ping-pong buffers. Keyed by a dense track id assigned at creation, they would leave only the
three clocks and the path length in the hot path: roughly +36 bytes instead of +144.

## What is still missing

- **A non-zero `GetNonIonizingEnergyDeposit()` in any test.** It is implemented and it compiles,
  but B1 is a gamma beam and nuclear recoil is a hadron-only deposit, so nothing in the suite
  currently produces a non-zero value for it. Implemented-but-unverified is a weaker claim than
  everything else on this page, and it is listed here rather than in the tables above until a
  proton run checks it.
