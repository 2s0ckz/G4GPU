# G4GPU

A from-scratch CUDA port of Geant4, validated against a real Geant4 11.1.1 install.

The goal is not "a GPU Monte Carlo inspired by Geant4". It is Geant4's own answers, from
Geant4's own models, transcribed rather than re-derived — and checked against the library
itself rather than against textbook formulae or published tables. Where this port and Geant4
disagree, the disagreement is measured, written down, and given a reason.

It is a work in progress. The electromagnetic physics for photons, electrons, positrons,
protons and alphas is complete and agrees with Geant4 to machine precision in most places;
there is no hadronic physics and no decay yet. The [status tables](#physics) below say exactly
what is there and what is not.

---

## Where it stands

| | Geant4 11.1.1 | G4GPU | |
|---|---|---|---|
| **Example B1**, 2M gammas of 6 MeV, dose in the scoring volume | 427.385 ± 0.87 pGy | 425.847 ± 0.87 pGy | **1.25 σ** |
| **Bragg peak**, 100 MeV protons in water, R80 | 77.798 mm | 77.783 mm | **−0.015 mm** |
| plateau dose, 0–60 mm | — | +0.075% | per proton |
| distal 80–20 width | 1.126 mm | 1.115 mm | −0.011 mm |

The B1 figure is **one sample**, and reading it as a constant is the mistake it invites. Four
other seeds give +1.44, −0.26, +0.37 and +0.32 σ: the spread is about 1 σ and straddles zero,
which is what agreement between two Monte Carlos looks like and is the only thing a single
number here can mean. The gate is 3 σ.

### Throughput

Events per second in real time, excluding initialisation - geometry, physics tables, data
loading - and excluding the final dose report. This is host wall clock over generating every
primary and running every batch, the same scope as Geant4's own `G4Timer`, which brackets
`InitializeEventLoop` to `TerminateEventLoop`.

| example B1, 6 MeV gammas, 2M events | events/s | 2M events |
|---|--:|--:|
| **real time, primaries generated host-side through `GeneratePrimaries()`** | **1.71 × 10⁶** | 1172 ms |
| reference driver, primaries generated on the device | 2.40 × 10⁶ | 832 ms |
| *GPU kernels alone, B1 (measures less - see below)* | *2.09 × 10⁶* | *959 ms* |

The first row is **10.4% slower** than the same program built at the commit before the step
hook and the pooled scheduler - 1.91 × 10⁶ - measured interleaved, five runs each way, with no
overlap between the two sets. Two changes bought that, and both are per-track bandwidth:

- The **`G4Track` state** - the accessors behind `GetParentID`, the three clocks,
  `GetTrackLength`, the vertex, `GetCreatorProcess`, `GetWeight`, `GetPolarization` and
  `GetTrackStatus` - is 136 of the 236 bytes a track slot costs, and measured **8.0%** on its
  own. It is on by default because a `GetParentID()` that does not work is a bigger defect than
  a slower run. [`docs/G4STEP.md`](docs/G4STEP.md) has that measurement and the way to get most
  of it back.
- The **species-agnostic pool** costs the remaining ~2.5%: four more bytes on every track for
  its species, and five index arrays that hand each kernel its own species contiguous at launch.

The dose is bit-identical across three seeds with identical step counts through both, so this is
bandwidth and not physics. The reference driver, which keeps its own per-species buffers on
purpose, is where it was.

The gap between the first and third rows is **~105 ns per event** (103 to 110 across four runs),
flat from 1M to 2M events - which is the check that says the cost above is on the device and not
in the API:
the virtual call into your `GeneratePrimaries()`, two `G4UniformRand()` draws, `sample_primary`,
and the readback into the primary array. That is the price of the Geant4-shaped API, not of the
transport, and it is 18% of a B1 run only because a 6 MeV gamma is cheap - 13 track-steps per
event. On a heavier problem the same 105 ns is a smaller share.

It was 198 ns until `G4Event` stopped allocating. A `G4PrimaryVertex` owns a
`std::vector<G4PrimaryParticle>`, so a generator that built one on the stack and added it did
two mallocs and two frees per event for a single primary. `G4Event::NewPrimaryVertex` hands
back a reused slot whose particle vector is cleared but not freed; `AddPrimaryVertex` assigns
into a slot rather than pushing a new one. Nothing was removed from the API and the dose did
not move by a digit.

**The GPU-only number is not the one to compare against Geant4.** It is measured with CUDA
events from the first kernel, so it cannot see primary generation, and quoting it against
Geant4's `Real=` would be comparing a smaller thing to a larger one. `examples/B1` prints both,
labelled, for that reason. `src/host/transport_run.cuh` documents which is which.

Bit-reproducible across batch sizes and thread counts: the RNG is counter-based and keyed on
`(rng_key, step)` carried by the track, never on its buffer slot.

*(RTX 3070, FP64, best of three runs. `G4GPU_FP32` compiles the whole transport in single
precision.)*

---

## How it is validated

Two mechanisms, and the project's whole claim rests on them.

### The oracle

`ref/dump/g4dump.cc` links the real Geant4 11.1.1 library and dumps its models — cross
sections, stopping powers, correction terms, tables — to CSV at `%.17g`. Every physics test in
`tests/` reads those CSVs and compares. Nothing is checked against a paper or a remembered
formula; the reference is the library.

`tools/g4src.sh` refuses to run against a Geant4 whose version does not match the oracle's, so
a silent upgrade cannot quietly invalidate every number in the repository.

The CSVs themselves are **not** checked in — they are 34 MB of Geant4's answers. What is
checked in is the dumper and `ref/oracle/run.bat`, which regenerates them.

### The dual build

`ref/proton/proton_depth.cc` and `examples/B1/` are each **one source file compiled twice**:
once against the real Geant4 install, once against this port's Geant4-shaped headers in
`src/g4`. The geometry, the beam, the binning, the scorers and the physics list are one
description rather than two that have to be kept in step, so the only thing that can differ
between the two runs is the transport.

### The pipeline

`build_all.bat` is the only thing that decides whether the port is correct. It builds every
driver, test, the viewer and the GUI; runs 48 tests; runs example B1 against a 2M-event Geant4
reference with a σ gate; runs the proton depth-dose comparison; checks batch-vs-macro
equivalence, per-voxel dose, mesh import, and that a project generated by the builder GUI
compiles and agrees with the builder.

`tools/quick.ps1` runs a named subset in ~15 seconds for iterating.

---

## Physics

Accuracies below are the **worst case over the whole comparison grid**, against Geant4's own
model output. `< 1e-6` means the test prints `0.0000%` — the deviation is below its print
resolution. Machine-precision figures are quoted as they are printed.

### Implemented and validated

#### Photons

| Geant4 model | what | test | worst deviation |
|---|---|---|---|
| `G4KleinNishinaCompton` | Compton cross section | `test_vs_oracle` | < 1e-6 |
| `G4KleinNishinaCompton::SampleSecondaries` | Compton final state | `test_vs_oracle` | < 1e-6 |
| `G4PairProductionRelModel` | pair production, cross section and final state, with LPM | `test_vs_oracle` | < 1e-6 |
| `G4LivermorePhotoElectricModel` | photoelectric, EPICS2017 data | `test_photoelectric` | < 1e-6 (21,022 points) |
| `G4SauterGavrilaAngularDistribution` | photoelectron direction | `test_photoelectric` | < 1e-6 |
| `G4LivermoreRayleighModel` | Rayleigh cross section, EPICS2017 | `test_rayleigh` | < 1e-6 (3,400 points) |
| `G4RayleighAngularGenerator` | Rayleigh deflection, Cullen three-term fit | `test_rayleigh_angular` | 2.9 sigma over 154 comparisons ² |

#### Electrons and positrons

| Geant4 model | what | test | worst deviation |
|---|---|---|---|
| `G4MollerBhabhaModel` | restricted and unrestricted dE/dx, delta-ray cross section | `test_vs_oracle` | < 1e-6 (966 pts) |
| `G4SeltzerBergerModel` | bremsstrahlung below 1 GeV, from the G4EMLOW tables | `test_brems` | < 1e-6 (726 pts) |
| `G4eBremsstrahlungRelModel` | bremsstrahlung above 1 GeV, with LPM | `test_brems_rel` | 0.108% (1,284 pts, 1 MeV–100 TeV) |
| `G4eeToTwoGammaModel` | positron annihilation, at rest and in flight | `test_annihilation` | < 1e-6 (1,764 pts) |
| `G4UrbanMscModel` | multiple scattering | `test_urban_general`, `test_msc` | **6.7e-16** |
| `G4UniversalFluctuation` | energy-loss fluctuations | `test_fluctuation` | mean preserved to < 0.1% |
| `G4VRangeToEnergyConverter` | production cuts | `test_cuts` | 0.018% |

#### Hadrons and ions

| Geant4 model | what | test | worst deviation |
|---|---|---|---|
| `G4BraggModel` | proton dE/dx below 2 MeV, PSTAR / ICRU / Ziegler branches | `test_bragg` | < 1e-6 |
| `G4BraggIonModel` | ion dE/dx below 2 MeV/u, ASTAR branch | `test_bragg` | 0.001% |
| `G4BetheBlochModel` | dE/dx and delta rays above the boundary | `test_hadron` | < 1e-6 |
| `G4ICRU73QOModel` | dE/dx for **negative** hadrons below the boundary | `test_icru73qo` | < 1e-6 (1,212 pts) |
| `G4MuBetheBlochModel` | muon dE/dx and delta rays | `test_muon` | < 1e-6 |
| `G4EmCorrections` | shell, Barkas, Bloch, Mott, high-order | `test_corrections` | 0.0003% |
| `G4ionEffectiveCharge` | effective charge and its correction | `test_ion_charge` | < 1e-6 |
| `G4ICRU49NuclearStoppingModel` | nuclear stopping | `test_nuclear_stopping` | 0.0017% |
| `G4IonFluctuations` | fluctuations for alpha, He3 and ions (Yang + Geissel) | `test_ion_fluctuation` | **6.3e-16** (2,058 pts) |
| `G4MuBremsstrahlungModel`, `G4MuPairProductionModel` | muon radiative losses | `test_muon` | < 1e-6 |
| `G4hBremsstrahlungModel`, `G4hPairProductionModel` | hadron radiative losses | `test_hadron_radiative` | < 1e-6 |
| `G4WentzelOKandVIxSection` | the Wentzel single-scattering engine | `test_wentzel` | 0.0002% (12,266 pts) |
| `G4WentzelVIModel` | multiple scattering for hadrons | `test_wentzel_msc` | round trip < 1e-6 |
| `G4ScreeningMottCrossSection` | the Mott/Rutherford ratio | `test_mott` | see note ¹ |
| `G4ComponentBarNucleonNucleusXsc` | Barashenkov nucleon–nucleus cross sections, Z = 2…92 | `test_nucleon_xs` | **2.0e-15** (10,738 pts) |

² A rejection sampler has no closed form to diff, so this compares the first two moments of
cos(theta) over a million draws per point against a million of Geant4's. 2.9 standard errors
over 154 comparisons is sampling noise. It found a real bug on its first run - see RISK.md V9.

¹ The Mott table is interpolated; `test_mott` reports the interpolation residual against
Geant4's own tabulated points, which is 0.28–4.6% depending on energy — that is the table's
own coarseness, and the port reproduces Geant4's interpolated value exactly.

#### Tables, materials and framework

| Geant4 class | what | test | worst deviation |
|---|---|---|---|
| `G4VEnergyLossProcess` + `G4LossTableBuilder` | dE/dx, range and inverse-range tables on Geant4's own grid (100 eV–100 TeV, 7 bins/decade, cubic spline, 100-substep midpoint integration) | `test_hadron_range` | see the table below |
| `G4IonisParamMat` | Zeff, Fermi energy, mean excitation, ⟨A^(−2/3)⟩ | `test_material_build` | < 1e-6 |
| `G4DensityEffectData` + Sternheimer | density correction | `test_density_effect` | 1.4e-14 absolute |
| `G4NistManager` | all 309 NIST materials | `test_all_materials` | built and checked |
| `G4Pow` | `A13`, `A23`, `powA`, `logX`, `expA` — Geant4's expansions, not `std::pow` | (used throughout) | exact |
| `G4PhysicsVector` spline | the cubic spline Geant4 interpolates every table with | `test_bragg`, `test_icru90` | < 1e-6 |
| `G4ICRU90StoppingData` | ICRU90 stopping, when enabled | `test_icru90` | < 1e-6 |

**Per-species dE/dx and range**, `test_hadron_range`, 6,204 points against
`G4EmCalculator::GetDEDX` and `::GetRange` in four materials:

| species | dE/dx, 1–10 keV | dE/dx, 0.1–2 MeV | range, 1–10 keV | range, 0.1–2 MeV |
|---|--:|--:|--:|--:|
| proton | 0.000% | 0.038% | 0.000% | 0.018% |
| alpha | 0.000% | 0.000% | 0.000% | 0.000% |
| π⁺ | 0.000% | 1.21% | 0.000% | 0.63% |
| K⁺ | 0.000% | 1.10% | 0.000% | 0.56% |
| μ⁺ | 0.000% | 6.40% | 0.000% | 3.21% |
| GenericIon | 0.001% | 0.13% | 0.001% | 0.067% |
| He3 (scaled from GenericIon) | 0.001% | 0.001% | 0.001% | 0.001% |
| p̄, π⁻, K⁻, μ⁻ | 2–6% | 1.4–15% | 0.8–1.6% | 0.9–7.9% |

The zeros are not rounding. On Geant4's grid this port evaluates the same models at the same
points, so away from a model boundary the two tables *are* the same table. The 0.1–2 MeV column
for the light species straddles a model boundary, where Geant4's 7-bin-per-decade spline rings
across the discontinuity. The negative hadrons carry a separate, undiagnosed discrepancy — see
[open questions](#open-questions).

#### Geometry, navigation and framework

| | test |
|---|---|
| 30 `G4VSolid` primitives + boolean operations, `Inside`/`DistanceToIn`/`DistanceToOut` against Geant4's own | `test_solids` |
| layered navigation, safety, voxel grids, tessellated meshes | `test_navigation`, `test_voxels`, `test_mesh` |
| `G4ParticleGun` position and direction sampling | `test_gun_position` |
| the Geant4-shaped API — `G4NistManager`, `G4Box`, `G4LogicalVolume`, `G4PVPlacement`, `G4MultiFunctionalDetector`, the action chain, `G4UImanager` macros | example B1 builds and runs unmodified |

### Particles transported

| | status |
|---|---|
| γ, e⁻, e⁺ | **transported** |
| proton, alpha | **transported** |
| μ±, π±, K±, p̄, He3, GenericIon | dE/dx, range, delta rays, radiative losses and fluctuations all validated; **not transported** — they need a track buffer in `transport_run.cu`, and π/K/μ need decay to be physically meaningful |
| deuteron, triton | in the oracle, not yet in `ParticleType` |
| neutrons | need hadronic physics |

### Not implemented

| Geant4 | what | why it matters |
|---|---|---|
| **`G4HadronElasticProcess`, `G4HadronInelasticProcess`** | the hadronic processes themselves | no hadronic physics at all; a proton's nuclear interactions are absent |
| `G4BinaryCascade`, `G4CascadeInterface` (Bertini), FTF/QGS, `G4PreCompoundModel`, `G4ExcitationHandler` | inelastic final states | ~450 headers between them |
| `G4ChipsElasticModel`, `G4HadronElastic`, `G4ElasticHadrNucleusHE` | elastic final states | the Barashenkov *rate* is ported; nothing acts on it |
| `G4BGGNucleonElasticXS` Z=1 / <14 MeV / >91 GeV branches | the rest of the nucleon elastic cross section | Z=1 matters most for water |
| `G4ParticleInelasticXS`, `G4NeutronInelasticXS`, `G4BGGPionInelasticXS` | inelastic cross sections | data-driven, from `G4PARTICLEXS4.0` |
| **`G4Decay`** | all of `processes/decay` | invisible for γ/e±/p/α; a hard blocker for π±, K±, μ± |
| **`G4CoulombScattering`** | the discrete single-scattering process | registered next to WentzelVI for every hadron and for e± above 100 MeV. Its cross-section engine *is* ported; the process is not |
| `G4UAtomicDeexcitation` and friends | fluorescence and Auger | constructed unconditionally by `G4EmBuilder`; emits only when the deexcitation flags are on, which QBBC leaves off |
| `G4UrbanMscModel::ComputeTruePathLengthLimit`, ion branch | the `fMinimal` step limit and the `mass ≥ masslimite` path | the Urban *cross section* is general and exact; the *stepping* half is still the electron's, so alpha and He3 currently scatter by WentzelVI, which is the wrong model for them |
| `G4StoppingPhysics`, `G4NeutronKiller`, `G4EmExtraPhysics` | capture at rest, neutron cut, gamma-/electro-/muon-nuclear, neutrinos, synchrotron | all of QBBC's remaining constructors |
| Penelope, PAI, Goudsmit-Saunderson, Livermore polarised, DNA, adjoint, PIXE, optical, transition radiation | alternative and specialist models | none are in QBBC's chain |

A class-by-class inventory of all three Geant4 process trees — 568 electromagnetic headers,
1,235 hadronic, 6 decay — is in [`docs/PORTED.md`](docs/PORTED.md).

### Open questions

Three things are measured, documented and unresolved rather than unknown:

1. **Negative hadrons in air.** At 10 MeV, above every model boundary, Geant4's own
   `ComputeDEDX` gives an anti-proton 4% *more* ionisation than a proton in air, and the same
   to 0.01% in water. The Barkas term is odd in the charge and would make it less; the dumped
   correction terms account for a sixth of the gap; ICRU90 is off by default and is never
   loaded for antiprotons. Left visible in `test_hadron_range` rather than absorbed into a
   tolerance.
2. **The ion step limit** above.
3. **The shared hadron track buffer** that six validated species are waiting on.

---

## Building

Windows, CUDA 11.6+, an NVIDIA GPU of compute capability 8.6 (edit `-arch` otherwise).

```
setupenv.bat        # MSVC + CUDA on PATH
build_all.bat       # everything: drivers, tests, viewer, GUI, example B1, the full pipeline
```

Individual pieces:

```
build_engine.bat            transport_run.obj, the templated stepping kernels
build_view.bat              the OpenGL viewer
build_gui.bat               the geometry builder
examples\B1\build.bat       example B1
tools\quick.ps1 -List       what the targeted test runner can run
```

To regenerate the oracle you need a Geant4 11.1.1 install; point `ref/oracle/run.bat` at it,
then `ref\dump\build.bat && ref\oracle\run.bat`.

## Running

```
examples\B1\exampleB1.exe -n 2000000        # batch
examples\B1\exampleB1.exe run1.mac          # macro
g4view.exe                                  # OpenGL viewer, tracks and dose
g4builder.exe                               # geometry builder; writes a compilable project
```

## Step-level scoring

Geant4 calls `UserSteppingAction` once per step of every track. This transport runs about
3x10^7 track-steps per second, so a host callback per step would cost more than the physics.
The `G4UserSteppingAction` here is therefore handed an **aggregate** - one pseudo-step per
volume per event - and [`src/g4/G4Step.hh`](src/g4/G4Step.hh) says so in as many words: a
stepping action that sums gets the right answer, and one that does anything else does not.

That is enough for dose and useless for anything that needs a step to be a step - the kinetic
energy of a particle as it crosses a boundary, the LET of the step that deposited the energy,
a spectrum, a per-step threshold, a coincidence between volumes.

[`src/core/step_hook.cuh`](src/core/step_hook.cuh) is the general answer. A **StepHook** is a
device functor called once per real step of every track:

```cpp
struct MyHook {
  __device__ void operator()(const DeviceStep<double>& s) const {
    if (s.entered(kDetector)) { /* s.ekin_pre is the energy at the boundary */ }
    const double let = s.let();   // MeV/mm, this step
  }
};
```

`DeviceStep` carries the species, the pre- and post-step kinetic energy and position, the
deposit, the **true** path length (not the chord - MSC deflects within a step, so a LET taken
from the displacement is biased high), both volumes, the event, the track and whether the
track died on this step.

### Reduce on the device; do not materialise steps

This is the part that decides whether a step-level analysis scales. A 2M-event B1 run is
2.6x10^7 track-steps; at 104 bytes a record that is 2.7 GB - and B1 is four volumes with one
pencil beam. A patient CT or a detector with 10^5 volumes is worse by orders of magnitude, and
worse in the direction that matters: the cost grows with the amount of transport, which is the
one quantity a Monte Carlo exists to increase.

So the primitive is a reducer, `StepTally`: a fixed device array, `atomicAdd`-ed into, with the
bin and the weight both supplied as device functors of the step. Its footprint is chosen at
setup and never moves - a 200-bin LET spectrum is 1.6 kB whether it sees a thousand steps or a
trillion. Nothing about it is new here: it is what `score` and `voxel_score` in
[`src/host/transport_run.cu`](src/host/transport_run.cu) already are, and `voxel_score` already
runs at ~10^6 cells. It is also what Geant4 users do - nobody `push_back`s every `G4Step`; they
fill a histogram from `UserSteppingAction` and keep the histogram.

| question | bin | weight | memory |
|---|---|---|---|
| LET spectrum of the dose in a volume | `LogBins` on `s.let()` | `s.edep` | bins |
| dose-averaged quality factor per event | `BinByEvent` (x2) | `Q(s.let())*s.edep`, `s.edep` | events |
| fluence spectrum entering a detector | `LogBins` on `s.ekin_pre` | `1` | bins |
| dose from one species | `BinByScoreSlot` | `s.edep` if species matches | volumes |

`StepTap`, which copies whole steps out for the host to read, also exists and is the stock
instantiation - it is the one hook that answers an arbitrary question without a rebuild, it is
gated at runtime by a null pointer, and it is capped and counts what it drops. It is for
debugging and small runs. It is the one thing here whose cost scales with the simulation.

### Writing one, in the shape of a Geant4 project

A stepping action that sees real steps is an ordinary class in the project's own files, with a
base class, member data and a named method - the same shape as a Geant4 `G4UserSteppingAction`:

```cpp
class QualityFactorScoring : public G4VUserDeviceSteppingAction<QualityFactorScoring> {
 public:
  __device__ void UserSteppingAction(const G4DeviceStep& step) const {
    const G4double q = Q(step.let());               // needs a real step
    atomicAdd(&weighted_[step.event], q * step.edep);
    atomicAdd(&plain_[step.event],        step.edep);
  }
 private:
  G4double* weighted_; G4double* plain_;            // the host allocates; this only points
};
```

Two things differ from Geant4 and both are forced by the machine. The base takes the derived
type (CRTP) so the call **inlines** instead of going through a vtable - at 3x10^7 steps a
second an indirect call per step is not free. And the method is `__device__`, so it cannot call
host functions, allocate or do I/O; the host allocates its buffers and hands it the pointers,
much as a `G4VPrimitiveScorer` is given its hits collection.

The project names its hook and instantiates the engine for it, in **one** `.cu` of its own:

```cpp
#define G4STEP_HOOK QualityFactorScoring
#include "g4/G4RunManager.hh"
#include "host/transport_run_impl.cuh"
namespace g4gpu::host { template class TransportEngine<double, QualityFactorScoring>; }
```

then `rm->SetStepHook(QualityFactorScoring(d_w, d_p, n, slot));` before `BeamOn`.
[`tests/test_custom_hook.cu`](tests/test_custom_hook.cu) is exactly this, built and run by
`build_all.bat`.

### What that costs, precisely

**g4gpu is not rebuilt and not edited.** The library is built once and a project compiles
against it, which is the arrangement Geant4 has. A project with a custom hook does not even
link `transport_run.obj` - it instantiates its own specialization and owns its own build.

**But the compile time does not go away, it moves.** Measured: a project with its own hook
takes **208 s** to build; rebuilding the engine takes **202 s**. The transport kernels are
templated on the hook, so a new hook type is a new set of kernels and something has to compile
them. This is genuinely unlike Geant4, where your project builds in seconds because the
transport is already compiled and dispatch is virtual. The price of a virtual call there is
nothing against Geant4's per-step cost; here it would defeat inlining at every one of 3x10^7
steps a second.

Note what is *not* affected: a project using the **stock** hook links the prebuilt object and
compiles in seconds, paying one predicated load per step. The slow path is opt-in, and only for
projects that write their own device code.

The distinction is *not* C++ versus CUDA. Every project here already goes through nvcc -
example B1 keeps Geant4's exact layout and file names (`exampleB1.cc`, `src/`, `include/`) and
`build.bat` compiles them with `-x cu`. What decides whether a build takes seconds or minutes
is whether one of the project's own translation units *instantiates the kernels*. B1's files
launch nothing (`-c`, not `-dc`), so they do not.

Getting a fast project build as well would mean runtime indirection - a device function
pointer, NVRTC, or an on-device expression bytecode - each of which buys it with either
per-step performance or a great deal of machinery. None of that is here.

### What it costs, and how that is known

The hook is wired into the three stepping kernels. With the stock null-gated `StepTap`:

- B1 dose is **bit-identical** across three seeds, and the track-step counts are exactly equal.
- Throughput moved **+0.14%** on the median of six interleaved 2M-event runs, inside a ±2.4%
  run-to-run spread. There is no measurable cost.

`g4dose.exe -verify-step-hook` asserts the claim the whole mechanism rests on - that a hook
sees every step exactly once, charged to the right event - against the scorer, which counts the
same energy by a completely different route (device `atomicAdd` as the steps happen, versus a
host sum over the records afterwards):

```
sum over steps of edep                  == score_sum[0]
sum over events of (event's sum)^2      == score_sum_sq[0]
```

Both hold exactly. The check was falsified before being trusted: making the gamma kernel skip
the steps that tracks die on - 0.038% of the energy - fails it at 3.8x10^-4 against a 10^-12
gate.

## Layout

```
src/
  core/       tracks, RNG, units, particle definitions
  data/       transcribed Geant4 tables (Seltzer-Berger, PSTAR/ASTAR, Mott, Barashenkov, ...)
  physics/    em/ and hadronic/ models, and stepper.cuh which orders them as Geant4 does
  geometry/   solids, boolean ops, navigation, voxels, meshes, BVH
  g4/         the Geant4-shaped API: G4Box, G4LogicalVolume, G4RunManager, ...
  host/       the transport engine, the dose driver, the viewer and builder hosts
  render/     OpenGL viewer and trajectory store
tests/        48 tests; test_custom_hook is a whole project, built as a user would build one
tools/        extractors, generators, comparison scripts, quick.ps1
ref/          the Geant4-linked oracle dumper and reference programs
examples/B1/  Geant4's example B1, unmodified in shape
docs/         RESULT.md (what is measured), RISK.md (what went wrong and why), PORTED.md
```

## Documentation

- [`docs/RESULT.md`](docs/RESULT.md) — every validation number and how it was obtained.
- [`docs/G4STEP.md`](docs/G4STEP.md) — the `G4Step`, `G4StepPoint` and `G4Track` interfaces
  method by method: what is available, what is missing, what each gap would cost, and the
  measured price of the per-track state that is already there.
- [`docs/RISK.md`](docs/RISK.md) — a log of defects found, with the reasoning that missed them
  the first time. Includes V5/V7 (the reference is a table, not a model — worth reading if you
  are porting anything against Geant4), V8 (a self-consistency test cannot see a scale error in
  its own input), and S12 (a bisect that was sound and whose conclusion was wrong).
- [`docs/PORTED.md`](docs/PORTED.md) — the class-by-class inventory.

## Licence and provenance

This port transcribes physics models from Geant4 11.1.1, which is distributed under the
[Geant4 Software Licence](https://geant4.web.cern.ch/download/license). Data tables are read
from, or extracted from, the Geant4 distribution and its `G4EMLOW` dataset. Geant4 is not
vendored here; you need your own install to regenerate the oracle or build the reference
programs.
