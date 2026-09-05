# Full-fidelity physics: what B1 actually uses

Authoritative chain, read from source: `QBBC` -> `G4EmStandardPhysics` -> per-particle
process and model assignment. **Geant4 11.1.1**, because that is the version installed and
built on this machine, so it is the version I can run as an oracle. Transcription source is
`D:\Documents\Geant4\Windows\geant4-v11.1.1\source`, not the 11.5.0 clone - the code must
match the oracle.

## The oracle

Geant4 11.1.1 is fully built (34 libs + DLLs). `exampleB1.exe` builds and runs against it:

```
2,000,000 gamma of 6 MeV -> 85.4769 nGy, rms 173.943 pGy   (+/-0.20%)
                          = 427.384 +/- 0.870 pGy per 10k events
wall time 9.3 s on 20 threads = 215,606 events/s
```

This replaces the shipped `exampleB1.out` as the reference. Two consequences:

1. **The reference is now tight (0.20%), not 2.9%.** Systematics below 1% become measurable.
2. **11.5.0's shipped 434.324 pGy and 11.1.1's actual 427.384 differ by 1.6%.** Geant4's own
   version-to-version spread on this example is comparable to the band I had been working in.
   Corrected comparison for the current code: **-0.27%**, about 1.1 sigma - not the 1.6% I
   had been reporting against the wrong version.

Beyond the dose, the build lets me dump Geant4's own cross sections, dE/dx and range tables
and diff them against each transcription directly. That is a far stronger check than reading
source.

## Model list (default `G4EmStandardPhysics`, no options)

### gamma

| process | model | data | status |
|---|---|---|---|
| G4PhotoElectricEffect | **G4LivermorePhotoElectricModel** (set explicitly) | G4EMLOW/livermore | **missing entirely** - currently a 20 keV absorb cut |
| G4ComptonScattering | **G4KleinNishinaCompton** | none, parameterized | **done, faithful** |
| G4GammaConversion | **G4PairProductionRelModel** | none, analytic | wrong model (using G4BetheHeitlerModel param) + crude energy split |
| G4RayleighScattering | **G4LivermoreRayleighModel** | G4EMLOW/livermore | omitted |

Note the conversion model is `G4PairProductionRelModel`, **not** `G4BetheHeitler5DModel`.

### e- / e+

| process | model | data | status |
|---|---|---|---|
| G4eIonisation | **G4MollerBhabhaModel** + **G4UniversalFluctuation** | none | dE/dx done; **no delta rays, no fluctuations** |
| G4eBremsstrahlung | **G4SeltzerBergerModel** (< 1 GeV) | G4EMLOW/brem_SB | radiative loss currently just discarded |
| MSC | **G4UrbanMscModel** (< 100 MeV) | none | using Highland instead |
| G4eplusAnnihilation | **G4eeToTwoGammaModel** | none | at-rest only, no in-flight |
| G4CoulombScattering | G4eCoulombScatteringModel | none | irrelevant: only above 100 MeV |

Available: **G4EMLOW 8.2** with `livermore`, `brem_SB`, `epics2017`, `doppler`, `fluor`.

## Order of work

Ranked by expected effect on the B1 dose, not by ease.

1. **Production cuts.** Geant4 converts the 1 mm default range cut into a per-material
   energy threshold (`G4RToEConvForElectron` / `ForGamma`). Everything downstream needs it:
   it sets the delta-ray and bremsstrahlung generation thresholds, and it is why Geant4
   deposits some energy continuously and some as explicit secondaries.
2. **Livermore photoelectric.** The largest missing process. Also builds the G4EMLOW reader
   that items 3 and 5 need.
3. **Livermore Rayleigh.** Cheap once the reader exists. No energy transfer, so small.
4. **Delta rays + universal fluctuations** in ionisation. Currently all collision loss is
   deposited locally; Geant4 emits delta rays above the cut which can leave the volume.
5. **Seltzer-Berger bremsstrahlung.** Real photons from the data tables, replacing the
   discard-the-radiative-fraction hack.
6. **G4PairProductionRelModel**, cross section and final state, replacing both the wrong
   model and the uniform energy split.
7. **Urban MSC.** Hardest by far - 638 real lines, 148 branches, 15 tuned constants - and
   subtle errors distort every shower silently. Last, with the oracle to check against.
8. **In-flight annihilation** via G4eeToTwoGammaModel.

## Ground rule

No approximations unless the source shows no alternative. Where Geant4 itself parameterizes
or tabulates, transcribe the parameterization or read the table. Each item gets checked
against the oracle before moving on, and the throughput cost recorded - fidelity will cost
speed, and the honest comparison needs both tracked.

## Throughput baseline, honestly

| | events/s |
|---|---|
| g4gpu track-parallel, RTX 3070 | 2,490,000 |
| Geant4 11.1.1 MT, 20 threads | 215,606 |

**11.6x** - but not yet apples to apples. Geant4 is running full QBBC: Livermore
photoelectric, Rayleigh, Urban MSC, Seltzer-Berger, fluorescence, plus hadronic
initialization. This number will fall as the physics above is added, and that is the point:
the comparison only becomes meaningful when the physics matches.

---

# Audit against the oracle

First run of `tests/test_vs_oracle.cu`, which diffs the port against Geant4's own dumped
numbers. Findings reorder the work.

## Already exact (6+ significant figures)

| quantity | result |
|---|---|
| electron density, Zeff, mean excitation energy | exact, all four materials |
| **Compton cross section** (G4KleinNishinaCompton) | **exact**, 1 keV - 10 GeV, all materials |
| **Pair production cross section** | **exact** |
| Moller-Bhabha unrestricted dE/dx | 0.00 - 0.26% |

The pair result was unexpected, since Geant4's default conversion model is
`G4PairProductionRelModel` while the port transcribed the `G4BetheHeitlerModel`
parameterization. The source explains it: `G4PairProductionRelModel` carries
`fParametrizedXSectionThreshold = 30 GeV` and below that uses **the same parameterization**.
The relativistic treatment only engages above 30 GeV, far outside B1. So the cross section
needs no change; only the final-state sampling does.

Matched-energy check at 5.95662143 MeV:

```
water  pair  mine 0.000320762397   G4 0.000320763    ratio 1.00000
water  compt mine 0.00244114436    G4 0.00244115     ratio 1.00000
bone   pair  mine 0.000736246141   G4 0.000736245    ratio 1.00000
bone   compt mine 0.00431398276    G4 0.00431398     ratio 1.00000
```

## Real gaps, quantified

| # | gap | size |
|---|---|---|
| 1 | **Photoelectric absent.** | Up to **92-97%** of total attenuation at 1 keV; 0.001-0.006% at 6 MeV. Matters only through the degraded photon spectrum - but that is where the low-energy deposit comes from. |
| 2 | **Rayleigh absent.** | Up to **9-13%** at low energy, 0.005-0.01% at 6 MeV. No energy transfer, so it only redirects. |
| 3 | **Range is 4-8% short** at MeV energies (0.9096 at 6 MeV in water, 0.9179 in bone). | Not a bug in the integral: Geant4's range comes from **restricted** dE/dx, since energy above the production cut leaves as delta rays. Ours is unrestricted. This is a symptom of gap 4, not an independent error. |
| 4 | **No delta rays, no fluctuations.** | Production cuts are now known exactly: e- **277.6 keV** in water, **301.3 keV** in tissue, **398.4 keV** in bone, 0.99 keV in air (floor). Above these Geant4 emits transportable secondaries; the port deposits everything continuously. |
| 5 | **Radiation length 0.7-1.05% high.** | My Tsai-form approximation vs Geant4's own computation. Feeds MSC. Fix by transcribing `G4Material::ComputeRadiationLength` rather than approximating. |
| 6 | Bremsstrahlung photons, Urban MSC, in-flight annihilation | unchanged from the plan above |

## Revised order

Cross sections for Compton and pair are done and exact, so items 3 and 6 of the original
plan drop away. What remains, reordered by measured impact:

1. **Restricted dE/dx + delta-ray production** (gaps 3 and 4 together). Largest measurable
   discrepancy, and the production cut values are now known.
2. **Radiation length** exact (gap 5). Small and quick, and MSC depends on it.
3. **Livermore photoelectric** (gap 1), with the G4EMLOW reader.
4. **Seltzer-Berger bremsstrahlung**, real photons.
5. **Livermore Rayleigh** (gap 2).
6. **Pair final-state sampling** - cross section already exact, only the energy split is crude.
7. **Urban MSC**.
8. In-flight annihilation.

## Correction to the dumper

`G4EmCalculator::ComputeDEDX` was called without a cut argument, so it returned
**unrestricted** dE/dx - which is why ours matched to 0.00%. That validates the
Moller-Bhabha transcription but says nothing about the restricted/delta-ray split. The
dumper needs a second pass that passes the real production cut per material, and the range
comparison should use `csda_range_mm` for the unrestricted check.

---

# Progress: items 1 and 2 done

## Item 2 - exact radiation length

Replaced the Tsai-form approximation with `G4Element::ComputeLradTsaiFactor` and
`ComputeCoulombFactor` transcribed verbatim, including the light-element table for Z<=4, and
`X0 = 1 / sum(n_i * radTsai_i)` per `G4Material::ComputeRadiationLength`.

| material | before | after |
|---|---|---|
| air | 1.0067 | **1.0000** |
| water | 1.0068 | **1.0000** |
| A-150 tissue | 1.0080 | **1.0000** |
| bone | 1.0105 | **1.0000** |

## Item 1 - restricted dE/dx and delta rays

Production cuts wired in from Geant4's own values. `collision_dedx` now takes a cut and
defaults to the material threshold, matching `G4MollerBhabhaModel::ComputeDEDXPerVolume`
(the formula already had the cut term; it was being passed Tmax). Added
`delta_ray_xs` from `ComputeCrossSectionPerElectron` and `sample_delta_ray` from
`SampleSecondaries` - both rejection branches, and the primary recoil taken from momentum
conservation exactly as Geant4 does rather than from an angle formula.

The stepper now competes three step limits: distance to boundary, a fraction of the residual
range, and the sampled distance to the next delta-ray emission.

Worst deviation over the full 10 keV - 100 MeV grid, e- and e+, all materials:

| quantity | worst | n |
|---|---|---|
| **delta-ray cross section** | **0.0003%** | 324 |
| restricted dE/dx | 0.3223% | 726 |
| unrestricted dE/dx | 0.2840% | 726 |

Range, which was 4-8% short because it had been integrating the unrestricted stopping power:

| material | E (MeV) | ours | Geant4 | ratio |
|---|---|---|---|---|
| water | 5.9566 | 33.5137 | 33.8507 | 0.9900 |
| tissue | 5.9566 | 29.7366 | 30.0916 | 0.9882 |
| bone | 5.9566 | 19.0893 | 19.4051 | 0.9837 |

Residual 1-1.6% at high energy is the crude radiative dE/dx, which item 4
(Seltzer-Berger) replaces.

## Effect on the B1 dose

| | dose per 10k events | vs Geant4 427.384 +/- 0.870 |
|---|---|---|
| before | 426.250 pGy | -0.27% |
| **after** | **426.818 pGy** | **-0.13%** |

## Cost

| | before | after |
|---|---|---|
| throughput | 2.49M events/s | **2.00M events/s** (-20%) |
| track-steps per event | 39.0 | 45.4 |

Delta rays are real transportable tracks, so both the step count and the track population
rise. Reproducibility is unaffected - still bit-identical across runs.

## Open: water-only 0.3% in restricted dE/dx

Tissue and bone are exact (ratio 1.0000); water alone reads 0.997-0.998, growing with
energy, which is the density-correction signature. Checked and excluded: the Sternheimer
coefficients are identical between 11.1.1 and 11.5.0, and Geant4 uses the chemical-formula
mean excitation energy (78 eV, matching ours) rather than the Sternheimer table's 75 eV.
Water is the only B1 material where those two values disagree, which is suspicious but not
yet explained. It predates this change - the unrestricted value was equally off. Water is
the envelope, not the scoring volume, so the dose sensitivity is second order.

---

# Progress: item 3 done - Livermore photoelectric

Transcribed `G4LivermorePhotoElectricModel` with EPICS2017 data from G4EMLOW 8.2.

**Cross section**, four branches exactly as Geant4 has them: two 5th-order polynomials in
1/E (high and low coefficient sets) covering everything above ~5 keV, and two tabulated
branches below. Loader reads `pe-high-Z.dat`, `pe-low-Z.dat`, `pe-cs-Z.dat`,
`pe-le-cs-Z.dat` for the ten elements in B1's materials and flattens them into device
arrays (10,222 tabulated points).

| band | worst deviation vs Geant4 | n |
|---|---|---|
| E >= 100 keV (high polynomial) | **0.0002%** | 324 |
| 5 - 100 keV (low polynomial) | **0.0003%** | 208 |
| below 5 keV (tabulated) | **0.0004%** | 112 |

The tabulated branches are interpolated linearly, and so are Geant4's: `ReadData` enables the
spline only when the Livermore data directory is `livermore`, and the default is `epics2017`.
An earlier version of this document called the linear interpolation an approximation of a
spline Geant4 used - it is not, and the 0.0004% in the table above was the clue. A cubic spline
replaced by a straight line on a grid this coarse does not agree to four parts in a million.

**Final state**: shell selection by the per-shell parameterized cross sections, then
Sauter-Gavrila photoelectron direction (transcribed including Geant4's quirk of feeding it
the *photon* energy - the model's electron-energy parameter is unnamed and unused). With
fluorescence off, which is the `G4EmStandardPhysics` default, Geant4 deposits the shell
binding energy locally; that is reproduced exactly, so no deexcitation machinery is needed.

The photon tracking floor dropped from 20 keV to **990 eV**, Geant4's own lowest gamma cut.
The old 20 keV blanket absorption existed only to stand in for the missing process.

## Effect on the B1 dose

| stage | dose per 10k | vs Geant4 427.384 +/- 0.870 |
|---|---|---|
| baseline | 426.250 pGy | -0.27% |
| + delta rays, exact radiation length | 426.818 pGy | -0.13% |
| **+ photoelectric** | **427.583 pGy** | **+0.05%** |

0.05% is about a fifth of the reference's own statistical error.

## Cost

| stage | events/s |
|---|---|
| before any of this | 2,490,000 |
| + delta rays | 2,000,000 |
| + photoelectric | **1,940,000** |

Photoelectric costs only 3% - it is a cheap cross section and it *removes* work by absorbing
photons that used to keep Compton-scattering. Against Geant4 MT on 20 threads (215,606
events/s) this is still **9.0x**, and now with substantially closer physics.

## Remaining

4. Seltzer-Berger bremsstrahlung (real photons; also fixes the residual 1-1.6% in range)
5. Livermore Rayleigh
6. Pair production final state (cross section already exact)
7. Urban MSC
8. In-flight annihilation

---

# Progress: item 4 done - Seltzer-Berger bremsstrahlung

Transcribed `G4SeltzerBergerModel` (the tabulated scaled DCS, including the positron
suppression factor) and the two Gauss-Legendre integrations from
`G4eBremsstrahlungRelModel` - `ComputeBremLoss` and `ComputeXSectionPerAtom` - with their
assembly over elements (Z^2 * n_atoms * gBremFactor).

Data: G4EMLOW 8.2 `brem_SB/br<Z>`, a 32 x 57 grid in (kappa, ln T) per element.

**Architecture**: the integrals run on the host at initialization onto a log-spaced energy
grid per material; the device interpolates. That is what Geant4 itself does - it builds
dE/dx and lambda tables at init and interpolates while tracking - so it is the same design,
not a shortcut.

| quantity | worst vs Geant4 | n |
|---|---|---|
| restricted brems dE/dx | **0.072%** | 486 |
| brems photon cross section | **0.054%** | 486 |

Both include positrons, so the suppression factor is confirmed.

## The range prediction held

The residual 1-1.6% in range after item 1 was attributed to the crude radiative term.
Replacing it closed exactly that gap:

| material @ 5.9566 MeV | before | after |
|---|---|---|
| water | 0.9900 | **1.0028** |
| A-150 tissue | 0.9882 | **1.0004** |
| bone | 0.9837 | **1.0004** |

## Photon emission

`SampleEnergyTransfer` transcribed: rejection in ln(k^2 + densityCorr) against a majorant
from the DCS at the lower edge, with Geant4's peak-limit and low-x majorant boosts, and the
positron correction inside the loop. Emission is a third discrete competitor in the lepton
step, alongside the boundary and delta-ray production.

With a keV-scale gamma production cut the restricted radiative dE/dx is ~3e-5 MeV/mm against
~0.29 for collision - essentially **all** bremsstrahlung now leaves as explicit photons
rather than being deposited locally, which is the physically correct behaviour and what the
old radiative-fraction hack was standing in for.

## Effect on the B1 dose

| stage | dose per 10k | vs Geant4 427.384 +/- 0.870 |
|---|---|---|
| + photoelectric | 427.583 pGy | +0.05% |
| **+ bremsstrahlung** | **428.168 pGy** | **+0.18%** |

Both are inside the reference's own +/-0.20%. The dose can no longer discriminate between
these versions - the per-process cross sections are the meaningful check now, and those are
at 0.05-0.07%.

## Cost

| stage | events/s | steps/event |
|---|---|---|
| + delta rays | 2,000,000 | 45.4 |
| + photoelectric | 1,940,000 | 45.4 |
| + bremsstrahlung | **1,850,000** | 47.8 |

Still **8.6x** Geant4 MT on 20 threads.

## Remaining

5. Livermore Rayleigh
6. Pair production final state (cross section already exact)
7. Urban MSC
8. In-flight annihilation

## Known approximation introduced here

Photon emission angle uses the standard mean angle m_e c^2 / E rather than Geant4's default
`G4DipBustGenerator`. At these energies the photon attenuation length in bone (~20 cm) far
exceeds the 6 cm scoring volume, so the emission angle has very little leverage on the
deposit - but it is a real difference and is logged in docs/RISK.md.

---

# Generality pass

B1 is the test case, not the target. Removed the main hardcodings so the same code takes
user-specified materials.

## Production cuts, computed not copied

Transcribed `G4VRangeToEnergyConverter` (the range inversion) with
`G4RToEConvForElectron::ComputeValue` and `G4RToEConvForGamma::ComputeValue`. A material now
derives its own thresholds from a range cut; nothing is hand-entered.

| material | gamma ours | gamma G4 | e- ours | e- G4 |
|---|---|---|---|---|
| air | 0.00099 | 0.00099 | 0.00099 | 0.00099 |
| water | 0.00252562 | 0.00252521 | 0.27764 | 0.277633 |
| tissue | 0.00228365 | 0.00228343 | 0.301335 | 0.301331 |
| bone | 0.00393535 | 0.00393604 | 0.398373 | 0.39836 |

**Worst 0.018%.** The scan also settled a documentation error: **QBBC's default range cut is
0.7 mm, not 1 mm.** The cut *values* used until now were dumped from Geant4 so no physics was
affected, but earlier notes in this file saying "1 mm" were wrong.

## Analytic Sternheimer fallback

Transcribed `G4IonisParamMat::ComputeDensityEffectParameters`'s analytic branch - condensed
and gas ladders, hydrogen and helium special cases - so a material absent from Geant4's
tabulated database still gets density-effect parameters. Reproduces the tabulated C exactly
for air (10.5961), tissue (3.1100) and bone (3.3390).

## Runtime material count

`kNumMaterials = 4` became `kMaxMaterials = 32` plus a runtime `MaterialTable::count`, with
`add_material()` as the entry point and `MaterialTable::elements()` discovering the distinct
atomic numbers a scene needs - replacing the hardcoded B1 element list that the
photoelectric and bremsstrahlung loaders had been given.

## The water discrepancy, narrowed

The unexplained water-only 0.3% in restricted dE/dx now has a concrete lead. The analytic
Sternheimer C for water is **3.5802**; Geant4's tabulated value is **3.5017**, and
1 + 2 ln(75/21.47) = 3.502 - so the tabulated coefficient was derived with **I = 75 eV**
while the dE/dx logarithm uses **I = 78 eV**. Water is the only B1 material where the two
disagree, which matches the pattern exactly.

It is not the whole story, though. Substituting I = 75 eV overshoots by as much as I = 78
undershoots:

| E (MeV) | I=78 | I=75 |
|---|---|---|
| 1.0 | 0.9982 | 1.0025 |
| 2.985 | 0.9974 | 1.0015 |
| 5.957 | 0.9970 | 1.0010 |

Geant4's effective value lies between. Bracketed, not solved; left in docs/RISK.md.

## Still hardcoded

- **Geometry.** `build_b1_volumes()` is still a fixed four-volume tree. A general scene API
  (arbitrary volume trees, rotations, more solid types) is the largest remaining piece of
  the generality work.
- **Primary particle.** 6 MeV gamma is baked into the drivers; a general gun needs particle
  type, energy spectrum and position/direction distributions.
- **Data paths**, defaulted to this machine but overridable via `G4GPU_PHOT_DIR` and
  `G4GPU_SB_DIR`.
- **NIST material shorthand.** Users must supply composition and mean excitation energy
  directly; there is no `FindOrBuildMaterial("G4_WATER")` equivalent.

---

# Progress: item 5 done - Livermore Rayleigh

Transcribed `G4LivermoreRayleighModel::ComputeCrossSectionPerAtom` with EPICS2017
`re-cs-Z.dat`. The files store E^2 * sigma, so the cross section is table(E)/E^2, with the
flat extrapolation above the last point Geant4 uses.

**Worst deviation 0.0004%** over 644 points, 1 keV - 100 MeV, all four materials.

Coherent scattering transfers no energy, so it affects the dose only through attenuation.
Measured effect: **+0.18% -> +0.19%**, i.e. none - as expected, since Rayleigh is 0.01% of
total attenuation at 6 MeV.

The angular deflection uses the Thomson dipole form rather than Geant4's tabulated atomic
form factors (`re-ff-Z.dat`), which are more forward-peaked. Since no energy is transferred
this only changes where the photon goes next. Logged in docs/RISK.md.

## Status after item 5

| process | status | agreement with Geant4 |
|---|---|---|
| Compton | done | exact (6 digits) |
| Pair production, cross section | done | exact (6 digits) |
| Photoelectric | done | 0.0004% |
| Rayleigh | done | 0.0004% |
| Ionisation dE/dx, restricted | done | 0.32% (water only; tissue/bone exact) |
| Delta-ray production | done | 0.0003% |
| Bremsstrahlung dE/dx and XS | done | 0.072% / 0.054% |
| Radiation length | done | exact |
| Production cuts | done, computed | 0.018% |
| Range | done | 0.04 - 0.28% |

| | dose per 10k | vs Geant4 427.384 +/- 0.870 |
|---|---|---|
| current | 428.193 pGy | **+0.19%** |

| | events/s |
|---|---|
| before the physics work | 2,490,000 |
| now, with all of the above | **1,770,000** |
| Geant4 MT, 20 threads | 215,606 |

**8.2x** Geant4 on a full CPU node, with the EM physics now matching process by process.

## Remaining

6. **Pair production final state.** Cross section is exact; the e-/e+ energy split is still
   sampled uniformly rather than from the Bethe-Heitler differential distribution. Small
   effect: both leptons range out inside the scoring volume.
7. **Urban MSC.** The substantial one. Currently Highland.
8. **In-flight annihilation.** Positrons annihilate at rest only.

Plus the generality work that matters more for the stated goal: a scene API for arbitrary
volume trees and a configurable particle gun.

---

# Exactness pass: every approximation replaced

Went back over everything that had been approximated rather than transcribed. All six are
now exact.

| # | was | now |
|---|---|---|
| 1 | brems angle: mean theta ~ m_e c^2 / E | **G4ModifiedTsai::SampleCosTheta**, verbatim |
| 2 | Rayleigh angle: Thomson dipole | **G4RayleighAngularGenerator**, the Cullen three-term fit |
| 3 | positron cut = electron cut (~2% off) | **G4RToEConvForPositron::ComputeValue** |
| 4 | photoelectric shell below 5 keV: innermost | **per-shell pe-ss-cs walk**, verbatim |
| 5 | pair e-/e+ split: uniform in available energy | **screened Bethe-Heitler rejection** |
| 6 | Sternheimer gas inferred from density | **explicit MaterialState** |

## Two corrections to my own earlier notes

**The brems angular generator is `G4ModifiedTsai`, not `G4DipBustGenerator`.** RISK.md had
said DipBust. `G4eBremsstrahlungRelModel` sets ModifiedTsai in its constructor, and
G4SeltzerBergerModel inherits it. Both take the *pre-emission electron kinetic energy* - the
model's second parameter is unnamed and unused, the same quirk as Sauter-Gavrila.

**The Rayleigh parameter extraction was initially wrong.** Inline `// 11-20` comments in the
Geant4 source shifted my table indexing by one, silently. Caught by the sum rule
**PP0 + PP1 + PP2 = Z^2**, which holds exactly for every element (Z=6 -> 36.0, Z=8 -> 64.0,
Z=20 -> 400.0) and failed under the first extraction. Worth recording as the kind of error
that produces plausible numbers and no error message.

## Positron cuts, now exact

| material | e+ ours | e+ Geant4 | ratio |
|---|---|---|---|
| water | 0.270815 | 0.270823 | 1.0000 |
| A-150 tissue | 0.293734 | 0.293733 | 1.0000 |
| bone | 0.386308 | 0.386306 | 1.0000 |

## Effect

| stage | dose per 10k | vs 427.384 +/- 0.870 |
|---|---|---|
| before the exactness pass | 428.193 pGy | +0.19% |
| after | **428.151 pGy** | **+0.18%** |

No significant change, which is the expected outcome: each of these was argued to be a
small effect, and measurement confirms it. The point was not to move the dose but to remove
places where the port and Geant4 differ by construction rather than by statistics.

Throughput 1.78M events/s. All ten test suites pass, all five drivers build, results
bit-identical across runs.

## What is still not transcribed, and why

- **LPM suppression** in pair production and bremsstrahlung. Activates above 100 GeV;
  B1 runs at 6 MeV. Structurally absent, not approximated.
- **Coulomb correction** in the pair sampler, above 50 MeV. Same reasoning.
- **Fluorescence and Auger** after photoelectric absorption. Geant4 has these *disabled* in
  `G4EmStandardPhysics`, and the binding energy is deposited locally - which is what the
  port does, so this is exact for this physics list, not an omission.

---

# Urban multiple scattering

Highland was the last genuine approximation. It is now replaced by a transcription of
`G4UrbanMscModel`, which is not one function but four interacting pieces:

| piece | Geant4 source | what it decides |
|---|---|---|
| `urban_xs_per_atom` / `urban_lambda` | `ComputeCrossSectionPerAtom` | transport mean free path |
| `urban_step_limit` | `ComputeTruePathLengthLimit`, `fUseSafety` branch | how long a step may be |
| `urban_geom_path` / `urban_true_path` | `ComputeGeomPathLength` / `ComputeTrueStepLength` | true vs geometric length |
| `urban_sample_scattering` | `SampleScattering` + `SampleCosineTheta` + `SampleDisplacement` | deflection and lateral shift |

`fUseSafety` is the right branch because `G4EmParameters` defaults `mscStepLimit` to it and
`G4EmStandardPhysics` - the EM constructor QBBC uses - never overrides it.

The two 15x22 correction tables (330 entries each) were extracted from the Geant4 source
**programmatically** into `src/data/urban_msc_tables.cuh`, with the element counts asserted.
An earlier hand extraction of a different Geant4 table was silently off by one because of
inline comments in the source; that is not a mistake worth making twice.

Supporting work: an isotropic safety (`src/geometry/safety.cuh`, Geant4's `DistanceToOut(p)`
/ `DistanceToIn(p)` for box, cone and trapezoid) because the step limiter needs distance to
the nearest boundary in *any* direction, and `G4VEnergyLossProcess`'s real continuous-step
limit `range*dRoverRange + finR*(1-dRoverRange)*(2-finR/range)`, which replaced a plain
`0.2*range`.

## Validating a model with no reference table

Every other process here was checked against a Geant4-generated table. There is none for
MSC. But `SampleCosineTheta` is *constructed* to hit a known mean: it mixes a core branch
(mean `xmean1`), a tail branch (`xmean2`) and an isotropic branch (0), weighting the
isotropic one by `1-qprob` so the total comes out at `xmeanth = exp(-tau)`.

That construction only works when `qprob <= 1`, and Geant4 does not clamp it. When
`qprob > 1` the isotropic branch never fires and the achieved mean is `xmeanth/qprob`. So
the invariant that actually holds in every regime is

    <cos(theta)> == xmeanth / max(1, qprob)

`tests/test_msc.cu` asserts exactly that, and it pins down tau, theta0, the xsi polynomial,
`xmean1`, `xmean2`, `prob`, `qprob` and both branch samplers simultaneously. It agrees to
within 3.3 standard errors over 45 (material, energy, tau) combinations - the largest of 45
standard normals sits near 2.7 on its own, so that is noise.

Finding this was itself the check: the first version of the test asserted `<cos> == exp(-tau)`
and failed by 4.5% at tau = 0.3. The measured ratios matched `1/qprob` to four decimals in
every failing case, and every case that passed had fallen back to `SimpleScattering`, which
*is* exactly mean-preserving. The transcription was right; the assertion was wrong.

## The statistics, which turned out to matter more than the physics

B1 reports `rms` as the uncertainty on the **accumulated total**, not the per-event spread.
A 10,000-event run therefore carries **+/-2.9%** - most 6 MeV gammas deposit nothing in the
scoring volume and a few deposit a lot, so the distribution is very skewed.

The reference 427.385 +/- 0.870 pGy is a **2,000,000**-event run scaled to 10k (+/-0.20%).
Comparing a 10k GPU run against it and reading a 1% difference as signal is a mistake - and
one made here, at the cost of a long and fruitless hunt through the MSC pieces for a deficit
that was noise. Both drivers now print the run's own uncertainty and say so when it is too
loose to compare.

Run 2,000,000 events before drawing any conclusion from the dose.

## Effect

All figures below are 2,000,000-event runs scaled to 10k, so +/-0.20% each.

| | dose per 10k | vs Geant4 427.385 +/- 0.870 |
|---|---|---|
| Geant4 11.1.1, 2M events | 427.385 pGy | - |
| this port, 2M events | **428.372 pGy** | **+0.23%** (0.80 sigma) |

Combined uncertainty is 1.23 pGy against a 0.99 pGy difference: statistically
indistinguishable.

Throughput went **up**, from 1.78M to 2.9M events/s, despite MSC costing roughly 2x per
step: Geant4's real step function cut steps per event from ~45 to 13.1. Against Geant4 MT
on 20 threads (2M events in 9.3 s, ~215k events/s) that is **~13.5x**.

Results stay bit-identical across batch size (8k-1M) and thread count (64-256).

## What is still not transcribed

- **`fUseDistanceToBoundary` and `fUseSafetyPlus`** step limiters. B1 uses `fUseSafety`;
  the others need per-track skin-depth state that this port does not carry.
- **The positron correction in `ComputeTheta0`** (the `posa`..`pose` cache entries).
  Positrons are ~7% of leptons here.
- **`SampleDisplacementNew`**. Geant4 defaults `dispAlg96 = true`, selecting the
  transcribed one; the alternative is unreachable at default settings.
- **`tlimitmin` is recomputed each step** rather than frozen with `rangeinit`. It is a
  floor of order 1e-5 mm, orders of magnitude below the limit it floors.

## WentzelVI and the ion tables, as of 2026-09-04

Closing out the six items that were on `docs/ROADMAP.md`. Four are done and measured, one
turned out not to exist, and one needs an option nothing selects.

**`G4ionEffectiveCharge`, both branches.** The helium fit is exact - 0.000e+00 over 606 points
- and Ziegler's heavy-ion form agrees to 1e-14 for nine ions from lithium to uranium, 1 keV to
100 GeV, six materials. It needed two new things: the material's Fermi energy
(`25 keV * vF^2`, with `vF` the atom-density-weighted mean of Ziegler's per-element table), and
`G4Pow`'s cube root, which is a third-order Taylor expansion about a tabulated point and not
`std::cbrt`. An exact power is 1e-5 away from what Geant4 computes, and the screening length
goes through `A23(1-q)`, so `src/data/g4pow.hh` reproduces the approximation.

The charge correction has no consumer here yet - Geant4 uses it only in
`G4EmCorrections::BuildCorrectionVector`, which needs ion transport - and the code says so
where it is defined. It is measured anyway, through `EffectiveChargeSquareRatio`, which is how
it can be measured at all: 11.1.1 keeps `chargeCorrection` private.

**ICRU 90.** `G4EmParameters::SetUseICRU90Data(true)` replaces PSTAR/ASTAR with the ICRU
Report 90 tables for G4_AIR, G4_WATER and G4_GRAPHITE, in the order `G4BraggModel` resolves
them - ICRU 90 first, PSTAR only if that comes back negative. Off by default, as in Geant4. The
six tables agree to 2e-16 over 1206 points; the switch moves water by up to 1.5%.

**The Mott/Rutherford ratio.** `G4WentzelOKandVIxSection` builds a
`G4ScreeningMottCrossSection` for e- and e+ unconditionally and uses its ratio as the
single-scattering rejection function. For those two species it is the angular distribution, not
a correction to it. Exact: 0.000e+00 over 17136 points, eight elements, both charges, 10 keV to
1 GeV. Its `beta` is the projectile-nucleus relative-system one, so it needed 92 target nuclear
masses; those come from the oracle rather than from a transcription of `G4NucleiProperties`, and
are checked through the ratio.

What is still absent from that class is the *cross section* half - `NuclearCrossSection` and its
form-factor variants - which no standard physics list reaches through WentzelVI. The transport
cross section here is `G4WentzelOKandVIxSection`'s own and agrees with it to 0.0003%.

**The second moment does not exist.** `useSecondMoment` has a setter, a getter, a per-material
physics table built at initialisation and an interpolating accessor - around
`ComputeSecondTransportMoment`, which returns `0.0`. With `z2 == 0`, `prob2` comes out
negative, and the `prob2 > 0.0` guard makes the branch it selects unreachable. Turning the flag
on in 11.1.1 changes nothing. See `docs/RISK.md` O10; this is the third time a "gap" has turned
out to be dead code in Geant4, and the lesson each time is to follow the value to its consumer
before transcribing the formula.

**`fUseDistanceToBoundary`** remains the one live item: `min(tlimit, ComputeGeomLimit(...)/facgeom)`
at a geometry boundary, where `ComputeGeomLimit` is the distance to the next boundary along the
track. WentzelVI's default is `fUseSafety`, so this needs a stepping-algorithm switch before
anything selects it.
