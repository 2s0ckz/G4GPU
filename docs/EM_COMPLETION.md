# Completing G4EmStandardPhysics

B1 exercises gamma / e- / e+ below 6 MeV. This is the rest of the physics list: the models
that exist for particles B1 never produces and for energy ranges it never reaches.

Definitive list, from `G4EmStandardPhysics::ConstructProcess` plus
`G4EmBuilder::ConstructCharged` (11.1.1).

| particle | process | model | status |
|---|---|---|---|
| gamma | PhotoElectric | Livermore | done |
| gamma | Compton | Klein-Nishina | done |
| gamma | Conversion | BetheHeitler5D | partial: no Coulomb corr >50 MeV, no LPM |
| gamma | Rayleigh | Livermore | done |
| e-/e+ | msc < 100 MeV | Urban | done |
| e-/e+ | msc > 100 MeV | WentzelVI | MISSING |
| e-/e+ | CoulombScattering | eCoulombScattering | MISSING |
| e-/e+ | Ionisation | MollerBhabha | done |
| e-/e+ | Bremsstrahlung < 1 GeV | SeltzerBerger | done |
| e-/e+ | Bremsstrahlung > 1 GeV | eBremsstrahlungRel + LPM | MISSING |
| e+ | Annihilation at rest | - | done |
| e+ | Annihilation in flight | eeToTwoGamma | MISSING |
| mu+/mu- | msc | WentzelVI | MISSING |
| mu+/mu- | Ionisation | Bragg / BetheBloch / MuBetheBloch | MISSING |
| mu+/mu- | Bremsstrahlung | MuBremsstrahlung | MISSING |
| mu+/mu- | PairProduction | MuPairProduction | MISSING |
| pi+/-, K+/-, p, pbar | msc | WentzelVI | MISSING |
| pi+/-, K+/-, p, pbar | Ionisation | Bragg / BetheBloch | MISSING |
| pi+/-, K+/-, p, pbar | Brems / PairProd | hBrems / hPairProduction | MISSING |
| p | NuclearStopping | ICRU49 | MISSING |
| ions | Ionisation | BraggIon / BetheBloch + effective charge | MISSING |
| ions | NuclearStopping | ICRU49 | MISSING |

Plus two exactness gaps left over from the Urban transcription:
- positron correction in `ComputeTheta0` (`posa`..`pose`)
- `tlimitmin` frozen with `rangeinit` rather than recomputed each step

## Validation

`G4EmCalculator` computes dE/dx and cross sections for *any* particle, material and energy,
so every model below can be diffed against the same oracle that validated the e-/e+ work.
`ref/dump/g4dump.cc` is extended per stage rather than trusting a transcription.

---

# Progress

## Done

| model | validated against | agreement |
|---|---|---|
| Urban MSC positron correction (`posa`..`pose`) | - | widens theta0 by 17.5%, no table exists |
| Urban `tlimitmin` frozen with `rangeinit` | - | matches Geant4's refresh points |
| `G4eeToTwoGammaModel` (in-flight annihilation) | model class, 644 points | **0.0003%** |
| `G4eBremsstrahlungRelModel` + LPM | model class, 1284 points, 1 MeV - 100 TeV | xs **0.11%**, dE/dx **0.0003%** |
| `G4PairProductionRelModel` Coulomb corr. >50 MeV | - | structural |
| `G4PairProductionRelModel` LPM >100 GeV | - | structural |
| `G4ModifiedTsai::SamplePairDirections` | - | was independently sampled, now coplanar |
| `G4BetheBlochModel` (mu, pi, K, p, pbar, alpha, He3) | model class, 5772 points, 1 keV - 100 TeV | dE/dx **0.000%**, delta xs **0.002%** |
| `G4MuBetheBlochModel` | model class, 2560 points | dE/dx **0.0003%**, delta xs **0.0003%** |
| `G4MuBremsstrahlungModel` | model class, 1920 points | **0.0003%** |
| `G4MuPairProductionModel` | model class, 1632 points | **0.0003%** |
| `G4hBremsstrahlungModel` | model class, 3840 points | **0.0004%** |
| `G4hPairProductionModel` | model class, 2840 points | **0.0003%** |
| `G4EmCorrections::ShellCorrection` | `G4EmCorrections` directly | **0.014%** |
| Barkas / Bloch / Mott / IonBarkas | same | **0.0001% / 0.0000% / 0.0000%** |
| `G4ionEffectiveCharge` (Zi <= 2) | same | **0.0000%** |
| Sternheimer density effect | `G4IonisParamMat::DensityCorrection` | **1.4e-14 absolute** |

Two of these were long-standing bugs rather than new features:

**The water density effect.** `G4_WATER`'s Sternheimer coefficients were taken from the
published `G4DensityEffectData` table (Cbar 3.5017). Geant4 does not use that row for water
- it computes the coefficients analytically and gets Cbar 3.5801. The 0.078 difference was
invisible in the electron work but showed up as a flat 0.39% error in *every* heavy-particle
dE/dx at high energy. Fixing it also closed the ~0.3% water discrepancy in the electron
tables that had been open since the ionisation work: those are now **0.0003%** across the
whole grid, and the B1 dose moved from +0.09% to **+0.05%**.

**Alpha and the ion branch.** `G4BetheBlochModel::SetupParameters` computes
`isIon = (!isAlpha && q > 1.1)`, which reads as "alpha is not an ion". But that line only
runs for a particle *other* than the one the model was initialised with; `Initialise` sets
`isIon` for any charge above one unit, alpha included. So alpha takes
`IonBarkasCorrection`, not `HighOrderCorrections`. Reading the first line alone left alpha
1.6% off; the term-by-term dump showed every correction matching individually, which is what
localised it to the assembly.

## Still missing

| model | why it matters |
|---|---|
| `G4WentzelVIModel` stepping | the cross sections are transcribed; its own step limiter, path conversion and sampling are not |
| `G4BraggIonModel` | ions below 2 MeV/nucleon; `G4BraggModel` covers protons and mesons |
| `G4ICRU73QOModel` | negative hadrons below 2 MeV/nucleon |
| `G4PSTARStopping` tabulated data | the 74 NIST materials; the port uses the Ziegler fallback for them, differing by up to 29% below 2 MeV |
| ICRU90 tabulated stopping | off by default (`G4EmParameters::fICRU90 = false`) |
| `G4ionEffectiveCharge` Zi > 2 branch | needs the material Fermi energy |
| `G4ScreeningMottCrossSection` | the full Mott correction for electrons; its fMottFactor companion IS transcribed |

## Second pass

| model | validated against | agreement |
|---|---|---|
| `G4MuBetheBlochModel` dE/dx and delta xs | model class, 2560 points | **0.0003%** |
| `G4MuBremsstrahlungModel` dE/dx and xs | model class, 1920 points | **0.0003%** |
| `G4MuPairProductionModel` dE/dx and xs | model class, 1632 points | **0.0003%** |
| `G4hBremsstrahlungModel` dE/dx and xs | model class, 3840 points | **0.0004%** |
| `G4hPairProductionModel` dE/dx and xs | model class, 2840 points | **0.0003%** |
| `G4WentzelOKandVIxSection` (nuclear, electron, transport) | engine class, 12266 points | **0.0003%** |
| `G4BraggModel`, Ziegler branch | model class, 100 points | **0.0000%** |
| `G4ICRU49NuclearStoppingModel` | model class, 1392 points | **0.0017%** |
| `G4NistManager` atomic masses, Z = 1..98 | dumped from Geant4 | exact |

Two more bugs surfaced:

**The atomic mass table.** The port hardcoded the ten elements B1 needs and returned 0 for
everything else, so any user material outside that set produced an *infinite* atom density -
silently, because nothing divided by it checked. It also rounded carbon to 12.0107 where
Geant4 has 12.010736, which was the source of a 3e-6 error in the material `<A^(-2/3)>` that
the Wentzel cut-off angle depends on. Both fixed by dumping all 98 from Geant4.

**Z23 is not Z^(2/3).** `G4ICRU49NuclearStoppingModel::InitialiseArray` fills an array named
`Z23` with `powZ(i, 0.23)`. Transcribing it as the name suggests gave a wrong reduced energy.

## What is library-level and what is wired into transport

The port's stepper tracks gamma, e- and e+ only. Everything transcribed for muons, hadrons
and ions is a validated model library, not yet connected to a transport loop: there is no
heavy-particle stepper to call it. The gamma / e- / e+ chain is both transcribed and wired,
which is why the B1 dose exercises it end to end.

## Third pass: the WentzelVI stepping and the remaining Bragg models

| model | validated against | agreement |
|---|---|---|
| `G4WentzelVIModel` step limit / path conversion / sampler | algorithm invariants (see below) | round trip **0.0000%** |
| `G4BraggModel`, Ziegler branch | model class, 484 points | **0.0001%** |
| `G4BraggIonModel`, Ziegler branch | model class, 484 points | **0.0010%** |
| `G4ICRU73QOModel` | model class, 1212 points | **0.0003%** |

**WentzelVI stepping.** Unlike everything else here, this has no Geant4 accessor that can be
called in isolation: `ComputeGeomPathLength` and `ComputeTrueStepLength` mutate model state
across calls and read a `G4Track`. `tests/test_wentzel_msc.cu` therefore checks the
invariants the algorithm is built on - the path conversion round-trips exactly, the geometric
length never exceeds the true length, the step limit stays inside `(0, min(request, range)]`,
the sampler returns unit directions and a displacement no longer than the step - and confirms
that both the multiple-scattering and pure single-scattering modes are actually exercised
(48 and 32 of 80 conversions). The cross sections it drives are diffed against
`G4WentzelOKandVIxSection` separately, to 0.0003%.

The call order is load-bearing and now documented in the header:
`wv_step_limit -> wv_geom_path -> (geometry) -> wv_true_path -> wv_sample_scattering`.
`wv_true_path` is what lowers `cos_theta_min` from 1 and recomputes the single-scattering
cross section above it. Skipping it leaves that cross section at its full forward value, and
the sampler then tries to generate millions of discrete scatters per step. The first version
of the test skipped it and appeared to hang.

**The Bragg family is now complete**: `G4BraggModel` for positive hadrons, `G4BraggIonModel`
for ions, `G4ICRU73QOModel` for negative ones. The last is not a variant of the first two -
its Barkas term is odd in the charge, which is precisely why an antiproton stops differently
from a proton at the same velocity, and why Geant4 gives negative hadrons their own model.

All three implement the per-element Ziegler / oscillator branch, which is what a
user-defined material gets. For the 74 NIST materials in `G4PSTARStopping` / `G4ASTARStopping`
Geant4 prefers its tabulated stopping power instead; `tests/test_bragg.cu` measures that gap
rather than hiding it (29.3% vs PSTAR, 19.3% vs ASTAR, both at sub-keV energies where the
two parameterisations diverge most) and validates the transcription against a material
deliberately built outside both tables.

## Coverage after three passes

Every model `G4EmStandardPhysics` registers now has a transcription, except the pieces listed
below. What remains is data-substitution and two structural gaps, not missing physics:

| still missing | why |
|---|---|
| `G4PSTARStopping` / `G4ASTARStopping` tabulated data | 74 NIST materials; the port uses the Ziegler branch for them, quantified above |
| ICRU90 tabulated stopping | off by default (`G4EmParameters::fICRU90 = false`) |
| `G4ScreeningMottCrossSection` | the full Mott correction for electrons; its `fMottFactor` companion IS transcribed |
| `G4ionEffectiveCharge` Zi > 2 branch | needs the material Fermi energy, which the material table does not carry |
| WentzelVI second moment, `fUseDistanceToBoundary` limit | `useSecondMoment` is false by default; B1 selects `fUseSafety` |
| Ziegler 1988 molecular stopping data | a per-molecule table Geant4 prefers over per-element summation for ~50 compounds |
