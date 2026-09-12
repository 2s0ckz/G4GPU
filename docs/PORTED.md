# What is ported, and what is not

An inventory of Geant4 11.1.1's three process trees against this port, class by class.

```
source/processes/electromagnetic     568 headers, 10 subdirectories
source/processes/hadronic           1235 headers,  6 subdirectories
source/processes/decay                 6 headers
```

Header count is a bad denominator - most of those are data holders, messengers and
interpolation helpers, and a third of the hadronic tree is models no reference physics list
instantiates. The denominator that means something is **what QBBC actually builds**, so every
table below is marked with whether the class is in QBBC's chain at all.

## Legend

| | |
|---|---|
| **T** | Transported. Transcribed, checked against `ref/oracle`, and reached by a particle in `stepper.cuh`. |
| **V** | Validated, not wired. Transcribed and checked by a registered test, but no species that uses it is transported yet. |
| **P** | Partial - a named sub-band or sub-case is done and the rest is refused rather than approximated. |
| **-** | Not ported. |
| **QBBC** | Whether QBBC instantiates it. `y` = yes, `opt` = only under a non-default option, `n` = no. |

---

## 1. electromagnetic/

### 1.1 standard/ (79 headers) - the core of the port

| Geant4 class | QBBC | | Where |
|---|:--:|:--:|---|
| G4KleinNishinaCompton | y | **T** | `em/gamma_processes.cuh`, `em/klein_nishina.cuh` |
| G4PairProductionRelModel | y | **T** | `em/pair_production.cuh` |
| G4MollerBhabhaModel | y | **T** | `em/electron_processes.cuh` |
| G4SeltzerBergerModel | y | **T** | `em/electron_processes.cuh`, `data/brems_data.cuh` |
| G4eBremsstrahlungRelModel | y | **T** | `em/brems_rel.cuh`. **Dispatched since P14c, and it was ported and unreachable before that.** `G4eBremsstrahlung::InitialiseEnergyLossProcess` splits at `min(SB high, 1 GeV)` and `G4RegionModels::SelectIndex` tests `e <= lowKineticEnergy[idx]`, so 1 GeV itself belongs to Seltzer-Berger and the split is a strict `>` everywhere in this port - the discrete cross section, the final-state sampling and the restricted radiative dE/dx of the e+- energy-loss table. What was missing beside the dispatch is that **the tabulated quantity is not the model**: `G4EmModelManager::FillDEDXVector`/`FillLambdaVector` fix `del = (xs_below/xs_above - 1)*elow` once per material from the ratio of the two models AT the boundary and carry `1 + del/E` above it (`em::model_boundary_del`). It is worth 2.0% of a 1 GeV electron's bremsstrahlung rate in water, 0.21% at 10 GeV, and exactly nothing at or below 1 GeV |
| G4ModifiedTsai | y | **T** | `em/electron_processes.cuh` |
| G4eeToTwoGammaModel | y | **T** | `em/annihilation.cuh` |
| G4UrbanMscModel | y | **T** | `em/urban_msc.cuh`, `data/urban_msc_tables.cuh`. The cross section is general across mass and charge and exact for all eight species dumped (`tests/test_urban_general.cu`, 41,952 points, 6.7e-16). **The stepping half is general too since P14b**: `ComputeTruePathLengthLimit`'s `fMinimal` branch with `facrange` 0.2, no lateral displacement and the `mass >= masslimite` distance estimate beside the electron's `fUseSafety` branch, `ComputeGeomPathLength`'s `currentKinEnergy < mass` test (it was `< electron_mass_c2`, right for the only species that reached it and a factor of 7300 out for an alpha), `SampleCosineTheta` and `SampleScattering` with the mass and the BARE charge the definition carries, and the `extremesmallstep` branch, which was missing on both paths and since P8e is live on both - `kLeptonExtremeSmallStep` is true, the lepton path passes `min(tlimitmin, lambdalimit)` off the tlimitmin the track carries, and the gate it moves by 0.0004 pGy is in 2.1.9. `tests/test_ion_msc.cu` against three new CSVs: transport mfp 6.9e-16 on 1,200 points, the step limit and both path conversions **exactly 0** on 750 rows, the randomised limit at 1.1 sigma on the 7 of 150 cells it is reached in, and the angle within 3.4 sigma at chi²/bin 2.18 over 300 cells x 400,000 draws with the lateral displacement exactly zero on both sides. The mean free path is **not tabulated** and that is Geant4's choice, not a shortcut - see 4.4. **`T` for the ion since P8e**, which was `V` for one package for a reason that was not the physics: `kUrbanIonMscWired` in `stepper.cuh` held the branch off because `transport_run.cu` - the one translation unit with all eighteen kernels - did not survive ptxas with it on (docs/RISK.md **V63**). With the engine split one unit per kernel it compiles, cheaper than the WentzelVI it replaces, and a transported ion is scattered by the model Geant4 gives it: docs/RISK.md **V65**, **V66** and section 2.1.9. `fUseDistanceToBoundary`/`fUseSafetyPlus` are still not transcribed, because option0 selects neither. |
| G4WentzelVIModel | y | **T** | `em/wentzel_msc.cuh`. **P14c wired it for e+- above `G4EmParameters::MscEnergyLimit()` = 100 MeV**, where `G4EmStandardPhysics::ConstructProcess` calls `msc2->SetLowEnergyLimit(MscEnergyLimit())` on the model it hands `ConstructElectronMscProcess`; `SelectIndex`'s `<=` puts 100 MeV itself on Urban, so `step_lepton`'s branch is a strict `>`, and `tests/test_electron_hi.cu` checks that against Geant4's own answer at the boundary. The step limit reads `G4VMscModel::xSectionTable` and not the model (4.4, and `em::WentzelLeptonTable` - 43 nodes on `BuildTableForModel`'s own grid holding `E^2 * CrossSectionPerVolume(..., 0.0, DBL_MAX)`, so at a cut of ZERO: docs/RISK.md V79), and this is the one path in the transport where the lateral displacement is non-zero, because `G4VMscModel::InitialiseParameters` sets `latDisplasment` from `LateralDisplacement()` for `abs(PDGEncoding) == 11` and from `MuHadLateralDisplacement()` - false - for everything `step_hadron` steps. Four `__device__ __noinline__` wrappers, because inlining it four more times killed ptxas: docs/RISK.md V81 - and the wrappers did not save it, so for one package the lepton branch shipped behind `em::kWentzelLeptonMscWired = false` with e+- stepped by Urban at every energy (docs/RISK.md V83). **`T` for e+- as well since P14d**, which was `V` for a reason that was not the physics, exactly as `kUrbanIonMscWired` was: the flag is `true`, the seventeen split units compile with it first time, and the second msc model costs `run_step_lepton` 448 bytes of stack frame (3040 -> 3488 B) at the same 255-register cap. B1's 6 MeV gamma gate and its 100 MeV electron row are IDENTICAL across the flag to every printed digit and every track-step - the branch is a strict `>` and neither beam reaches above it - and the 1 GeV electron row moves +0.201%, from -0.53% to -0.33% against a 1,000,000-event Geant4 reference. It does not close the row: docs/RISK.md **V95** and section 2.1.9 have what is left and which of the two candidates for it is already excluded. The STEPPING half has no Geant4 accessor that can be called in isolation - `ComputeGeomPathLength` and `ComputeTrueStepLength` mutate model state across calls and read a `G4Track` - so it is checked as properties (`tests/test_wentzel_msc.cu` blocks 1-3: the path conversion inverts, the geometric length never exceeds the true one, the limit is bounded, both scattering modes are reached). The single-scattering SAMPLER is checked as a distribution against `G4WentzelOKandVIxSection::SampleSingleScattering` **in this model's configuration of it**, which is not `G4CoulombScattering`'s: `ref/oracle/wentzel_msc_sample.csv`, 240 cells over (species, Z, energy, step fraction) at 400,000 draws each, worst 3.3σ on the accepted fraction, 3.2σ on `<1-cos>` and 2.01 χ²/bin, with the angular interval, `cosTetMaxElec`, the electron/nucleus split and the target mass `factD` is built from all bit-exact. That comparison is what docs/RISK.md **V47** - a dropped `1/(1 + z1*factD)` - had nothing to fail against; without it 52 of the 240 cells fail, to 118σ. Alpha is a dumped species because `uses_wentzel_msc` sends it here where Geant4 sends it to Urban (row above), so the substitution has an oracle of its own. |
| G4WentzelOKandVIxSection | y | **T** | `em/wentzel_xs.cuh` |
| G4ScreeningMottCrossSection | y | **T** | `data/mott.hh` (the Mott/Rutherford ratio inside WentzelVI) |
| G4BraggModel | y | **T** | `em/bragg.cuh` |
| G4BraggIonModel | y | **T** | `em/bragg.cuh` |
| G4BetheBlochModel | y | **T** | `em/hadron_ionisation.cuh` |
| G4UniversalFluctuation | y | **T** | `em/fluctuation.cuh` |
| G4NuclearStopping / G4ICRU49NuclearStoppingModel | **opt** | **T** | `em/nuclear_stopping.cuh`. **The QBBC column was `y` and is wrong.** `G4EmStandardPhysics::ConstructProcess` builds the process only `if(param->MaxNIELEnergy() > 0.0)`, and `G4EmParameters::Initialise` sets `maxNIELEnergy = 0.0` - so `pnuc` is null and no species gets it. `ref/oracle/species_processes.csv`, one row per process on each species' own manager in the constructed QBBC, carries no `nuclearStopping` row for any particle. `/process/em/setMaxNIEL <E>` turns it on, which is why this is `opt`. `step_hadron` applied it unconditionally; `uses_nuclear_stopping` in `core/particle.cuh` is now the one place that decides, and it says no. |
| G4PSTARStopping, G4ASTARStopping, G4NISTStoppingData | y | **T** | `data/nist_stopping.hh` |
| G4SauterGavrilaAngularDistribution | y | **T** | `data/photoelectric_data.cuh` |
| G4MottData, G4SBBremTable | y | **T** | `data/mott.hh`, `data/brems_data.cuh` |
| G4IonFluctuations | y | **T** | `em/ion_fluctuation.cuh`, `data/yang_fluctuation.hh` - the alpha's and every ion's fluctuation model |
| G4Pow (powA / logX / expA) | y | **T** | `data/g4pow.hh` - Geant4's own expansions, not `std::pow` |
| G4ICRU73QOModel | y | **T** | `em/icru73qo.cuh`. In the dispatch, the range tables and now the transport: pi-, K-, pbar and mu- have kernels. The plumbing that was missing is done. **The charge-odd DOSE split P1 left open is CLOSED** (docs/RISK.md V46), and it was not this model. Example B1 at 200 MeV, 500,000 events a side (`ref/b1hadron/`), before and after: mu- +3.78% → **-0.073%** (24.7 → 0.49 sigma), pi- +5.01% → **+0.007%** (32.7 → 0.04), kaon- +9.52% → **+0.069%** (61.0 → 0.46), and mu+, pi+, kaon+ **bit-identical** at -0.34%, -0.17% and +0.18% (2.25, 1.16, 1.21 sigma). The port's own charge splits are now 4.09%, 5.24% and 9.74% against Geant4's 4.36%, 5.43% and 9.62%. The mechanism is `G4VEnergyLossProcess::AlongStepDoIt`'s long-step branch, which fires on every step at these energies and reads the RANGE table rather than the dE/dx table both sides kept comparing - and Geant4 interpolates that range table linearly for the second-registered member of every charge pair, because `G4EmBuilder` shares one `G4hBremsstrahlung` between the two and `G4MuBremsstrahlung`'s constructor calls `SetSpline(false)`. Every one of P1's exclusions holds; they all tested a value, and the defect was in an interpolation rule. `em::hadron_table_uses_spline` is the one place that decides. What this row still carries is the charge-odd air anomaly in `tests/test_hadron_range.cu` (pbar 25% and K- 20% in one band, in `G4_AIR` only), which is a separate and still undiagnosed disagreement about the model's own numbers. |
| **G4CoulombScattering** | **y** | **V** | `em/coulomb_scattering.cuh`. The last EM process in QBBC's chain. Cross section per atom exact against `G4eCoulombScatteringModel::ComputeCrossSectionPerAtom` over 10 species x Z ∈ {1, 6, 8, 13, 26, 82} x 73 energies x 4 materials: **0** for e±, mu±, pi±, p, pbar and 5.5e-15 for K±, with the angular interval to 4.4e-16 and both `MinPrimaryEnergy` forms to 8e-15 (`tests/test_coulomb_scattering.cu`, `ref/oracle/coulomb_xs.csv`). 6098 rows agree on the process being OFF and none disagrees, so the material-dependent threshold is checked too. The same 1752 points are compared again at the **proton** cut, which is the cut the transport passes (docs/RISK.md V48), and the two agree to the last bit because the only quantity a cut can move here is `cosTetMaxElec`. `V` and not `T` because the discrete process needs a stepper hook and `stepper.cuh` is P7/P8's this phase; the device function and its call-site contract are ready. Two findings in the header: **there is no angular handover between msc and single scattering in option0** - `MscThetaLimit` is π, so both models' `cosThetaMin`/`cosThetaMax` are -1 and they cover the same range, separated only by energy (e±, 100 MeV) or by the table threshold (hadrons) - and the light hadrons get **no** explicit energy limit, so their table starts at `G4CoulombScattering::MinPrimaryEnergy`, which is material-dependent (0.87 MeV for a proton in air against 40.0 MeV for an electron). |
| **G4eCoulombScatteringModel** | **y** | **P** | same file. `ComputeCrossSectionPerAtom` and `MinPrimaryEnergy` exact; `SampleSingleScattering` statistically, 20,000 draws x 240 cells against `ref/oracle/coulomb_sample.csv`, worst chi²/bin 2.87 and every scattered fraction inside 5σ; and **`SampleSecondaries`' recoil exactly**, one row per real call in `ref/oracle/coulomb_recoil.csv` (75,000 calls, three passes on the recoil threshold), so the recoil energy, the primary's final energy, the local deposit, the threshold branch and the recoil ion's direction from momentum balance are compared point by point rather than statistically: worst 4.4e-16 on `trec`, 3.6e-16 on `finalT`, 0 on `edep`, 3.3e-15 on the ion direction. Geant4's own `min(trec, kinEnergy)` clamp fires there - for an **antiproton** only, since `SampleSecondaries`' `1 == iz && particle == theProton` exception keeps a proton from backscattering off hydrogen and a pbar is not in that test - and is checked on those rows. `P` for one refused sub-case: **`SelectIsotopeNumber`**. `SampleSecondaries` overrides the element's mean atomic mass with the sampled isotope's *nuclear* mass between the cross section and the sampler, and reproducing the draw needs per-element natural abundances - `data/natural_isotopes.hh` carries the nuclide set only, and `data::Material` has no isotope list. `coulomb_sample_secondaries` takes (Z, A, target mass) and `coulomb_refuse_isotope_selection` names what is missing. The element draw is by direct partial cross section rather than `G4EmElementSelector`'s table, as everywhere in this port. Two facts about the CUT are in docs/RISK.md V48: it is the **proton** production cut that reaches this model, not the electron one, and it changes nothing here because the electron-scattering channel it gates is closed at every energy - so `wentzel_electron_xs` and the sampler's electron branch are transcribed but not exercised through this process. |
| G4hCoulombScatteringModel | **n** | - | **The QBBC column was `y` and is wrong.** Nothing in 11.1.1 constructs it: `grep -rn "new G4hCoulombScatteringModel" source/` has no match, only `#include` lines in `G4EmStandardPhysicsWVI.cc` and `G4EmStandardPhysicsSS.cc`. Every species with the process - hadrons included - gets `G4eCoulombScatteringModel`, created inside `G4CoulombScattering::InitialiseProcess`. Dead code. |
| G4IonCoulombScatteringModel / G4IonCoulombCrossSection | **n** | - | reachable only through `G4EmBuilder::ConstructIonEmPhysicsSS`, which option0 never calls; `ConstructIonEmPhysics` (G4EmBuilder.cc:119-145) takes no `isWVI` argument and registers no `G4CoulombScattering` at all. So no ion in QBBC has single Coulomb scattering, and `ref/oracle/coulomb_limits.csv` carries a row per species saying so. |
| G4eIonisation, G4hIonisation, G4ionIonisation | y | **P** | process classes; their `AlongStepDoIt`, model selection, base-particle scaling and per-step effective charge are inlined in `stepper.cuh` and `em/hadron_range.cuh` rather than existing as classes. All three processes' model dispatch is complete: Bragg / BraggIon / ICRU73QO / BetheBloch / MuBetheBloch, chosen by process rather than by charge magnitude. **The e+- energy-loss TABLE is `G4LossTableBuilder`'s since P14c** - `em::RangeTable`, 85 nodes from 100 eV to 100 TeV, one per species, restricted dE/dx summed over `G4eIonisation` and `G4eBremsstrahlung` as `G4LossTableManager::BuildTables` sums them, with the range and inverse-range vectors built by the same integration the hadron tables use (4.3). Before that it was 128 linear bins of range alone, from 1 keV to 100 MeV, clamped at both ends: docs/RISK.md V64 and V77. `AlongStepDoIt` deposits the WHOLE continuous loss, which it did not until an energy balance on the device found the missing half (V82). What is still `P` for a LEPTON, beyond the class structure: `PostStepGetPhysicalInteractionLength`'s integral approach - the cached `preStepLambda` and `PostStepDoIt`'s rejection against the post-step lambda - so this port draws its delta-ray and bremsstrahlung interaction lengths from the models at the pre-step energy rather than from Geant4's lambda vectors. Refused by name and measured: docs/RISK.md V78. **And the loss itself is `AlongStepDoIt`'s SHAPE since P14d**, not only its total: `length*GetDEDXForScaledEnergy` first and the range inversion only when that exceeds `linLossLimit` = 0.01 of the kinetic energy (`G4VEnergyLossProcess.cc:811-835`), then the `finalT <= lowestKinEnergy` balance at `:913-917`. The port inverted the range on every step and guarded the result, which in a vacuum - where `range - step_len` is a no-op in double - took the WHOLE kinetic energy and killed every lepton on its first step (docs/RISK.md V84, closed by **V96**). `em::kLinearLossLimit` is a lepton constant because `G4ionIonisation` sets 0.02; `step_hadron` has had this structure since it was written and is measurably untouched by the change. The 6 MeV gamma gate moved -0.0710 +/- 0.0045 pGy on five seeds, 0.0166%. |
| G4eMultipleScattering, G4hMultipleScattering | y | **P** | `G4VMultipleScattering::AlongStepDoIt` is inlined, the class is not. `G4hMultipleScattering` defaults to **Urban**, and `G4EmBuilder::ConstructIonEmPhysics` gives alpha, He3, deuteron, triton and GenericIon a fresh one with no model set - so every ion scatters by Urban in Geant4, and only the light hadrons and muons get WentzelVI. Since P14b so does every ion here. What the process itself contributes beyond the model is `G4EmTableUtil::PrepareMscProcess`, and its one decision is that **the step limit type, `facrange` and the lateral-displacement flag are chosen by the PARTICLE and not by the model**: `part.GetPDGMass() > CLHEP::MeV` takes `MscMuHadStepLimitType` (`fMinimal`), `MscMuHadRangeFactor` (0.2) and `MuHadLateralDisplacement` (false), and `G4VMscModel::InitialiseParameters` makes the same split on `abs(PDGEncoding) == 11`. A proton's WentzelVI and an alpha's Urban therefore share all three. `P` for the `AlongStepDoIt` displacement-relocation branch, which is reachable only for a lepton (a heavy particle's displacement is identically zero) and is inlined in `step_lepton` alone. |
| G4ComptonScattering, G4GammaConversion, G4PhotoElectricEffect | y | **P** | the discrete-process bookkeeping is inlined in `stepper.cuh` |
| G4KleinNishinaModel | opt | - | polarised Compton only |
| G4BetheHeitlerModel, G4BetheHeitler5DModel | opt | - | not the default conversion model |
| G4PEEffectFluoModel | opt | - | Livermore is the default photoelectric model |
| G4XrayRayleighModel | opt | - | Livermore is the default Rayleigh model |
| G4UrbanFluctuation | opt | - | `fUrbanFluctuation` is opt-in; default is `fUniversalFluctuation` |
| G4LossFluctuationDummy | opt | - | |
| G4GoudsmitSaundersonMscModel, G4GoudsmitSaundersonTable, G4GSMottCorrection, G4GSPWACorrections | opt | - | EM opt3/opt4 |
| G4LindhardSorensenIonModel, G4LindhardSorensenData | opt | - | ion physics variants |
| G4AtimaEnergyLossModel, G4AtimaFluctuations | opt | - | |
| G4BetheBlochIonGasModel, G4BraggIonGasModel | opt | - | gaseous-detector variants |
| G4PAIModel, G4PAIPhotModel, G4PAIxSection, G4PAIySection, G4PAIModelData, G4PAIPhotData, G4InitXscPAI | opt | - | PAI energy loss, 7 classes |
| G4eDPWACoulombScatteringModel, G4eDPWAElasticDCS, G4eSingleCoulombScatteringModel, G4IonCoulombScatteringModel, G4IonCoulombCrossSection | opt | - | |
| G4WentzelVIRelModel, G4WentzelVIRelXSection | opt | - | relativistic WentzelVI |
| G4eBremParametrizedModel, G4DipBustGenerator | opt | - | |
| G4eplusTo2GammaOKVIModel, G4eplusTo3GammaOKVIModel | opt | - | |
| G4DeltaAngle, G4DeltaAngleFreeScat | opt | - | only reached with atomic deexcitation on; the standard models sample their own delta direction, which **is** ported |
| G4ESTARStopping, G4WaterStopping, G4IonICRU73Data | opt | - | |
| G4EmStandUtil | y | - | a factory; its one decision (which fluctuation model) is the finding below |

### 1.2 lowenergy/ (117 headers)

| Geant4 class | QBBC | | Where |
|---|:--:|:--:|---|
| G4LivermorePhotoElectricModel | y | **T** | `data/photoelectric_data.cuh` (EPICS2017) |
| G4LivermoreRayleighModel + G4RayleighScattering | y | **T** | `data/rayleigh_data.cuh` |
| G4RayleighAngularGenerator | y | **T** | `data/rayleigh_angular_tables.cuh` |
| **G4UAtomicDeexcitation, G4AtomicTransitionManager, G4FluoData, G4AugerData, G4ShellData, G4AtomicShell** | **y** | **-** | fluorescence and Auger. `G4EmBuilder` constructs `G4UAtomicDeexcitation` unconditionally; whether it *emits* depends on the deexcitation flags, off by default in QBBC. Not ported. |
| the other ~105: Penelope (14), MicroElec (11), ecpssr/PIXE (13), polarized Livermore (4), G4IonParametrisedLossModel and its scaling, G4hICRU49*, G4hZiegler1985*, interpolation helpers | n/opt | - | none are in QBBC's chain |

### 1.3 muons/ (14 headers)

| Geant4 class | QBBC | | Where |
|---|:--:|:--:|---|
| G4MuBetheBlochModel | y | **T** | `em/hadron_ionisation.cuh` (`tests/test_muon.cu`). mu± are transported. The model was exact all along; what was not was the *dispatch* - `hadron_ioni_dedx` did not forward the shell tables to it, so the muon's range table and every transported dE/dx lacked the shell correction and `G4EmCorrections::HighOrderCorrections`. Since the high-order term is charge-odd (Barkas ~ z³), the port gave mu+ and mu- a bit-identical B1 dose where Geant4 separates them by 4.36% (re-measured at 500,000 events; the 4.6% first quoted was a 20,000-event figure). Fixed; mu+ dE/dx 6.397% → 0.861%, mu- 10.176% → 2.209%, and the port now separates its two muons by 0.223%. What it did **not** fix is the reference's 4.36%, and the 0.223% it left was the *wrong* answer for a different reason: the reference's own range table for a mu- is interpolated linearly, so its transported loss is not the integral of its own dE/dx. Closed in docs/RISK.md V46 - the port's two muons now differ by 4.09% and its mu- agrees with Geant4 to 0.073%. docs/RISK.md V38 and V46. |
| G4MuBremsstrahlungModel | y | **V** | `em/muon_radiative.cuh`. dE/dx and cross section per volume, exact; **no `SampleSecondaries`**, so the discrete process is not wired - see the row below. |
| G4MuPairProductionModel | y | **V** | `em/muon_radiative.cuh`. Same: the numbers, not the final state. |
| G4MuIonisation, G4MuMultipleScattering | y | **T** | mu± have their own dE/dx and range tables on Geant4's grid, with the flat 200 keV model boundary and G4MuBetheBlochModel above it, a kernel each (`kSpeciesMuonMinus`, `kSpeciesMuonPlus`) and the gun accepting them. |
| **G4MuBremsstrahlung, G4MuPairProduction** (and **G4hBremsstrahlung, G4hPairProduction**, 1.4) | **y** | **P** | The **discrete** radiative processes, and the one gap the muon's transport makes reachable. `step_hadron` samples no discrete radiative interaction for any species, and the range table is ionisation only (`hadron_total_dedx`). Measured, in `G4_WATER`, from `ref/oracle/muon_models.csv` and `hadron_radiative.csv`: the omitted *continuous* restricted share of dE/dx for mu- is 5e-5 at 1.6 GeV, 3.6e-4 at 10 GeV and 2.7e-3 at 100 TeV - below every tolerance in the port. The omitted *discrete* process is the one that matters: mu- at 1 GeV has a 649 m brem and 1.8 km pair mean free path against a 6 m range (P ~ 1%), and at 10 GeV an 86 m pair mean free path against a 58 m range (P ~ 0.8). So a muon above about 10 GeV in water is transported without its dominant loss channel. For pi± the brem mean free path is 1.8e6 mm at 200 MeV and pair production does not start until max(850 MeV, 8m) = 1.12 GeV; for the proton, 7.5 GeV. **Pre-existing** - the proton has carried `hBrems`/`hPairProd` unwired since it was transported - and not P1's to close: what is missing is two `SampleSecondaries`, which is physics. See docs/RISK.md. |
| G4ModifiedMephi | y | - | angular generator for muon secondaries |
| G4ePairProduction, G4MuonToMuonPairProduction, G4MuonToMuonPairProductionModel | y | - | |
| G4EnergyLossForExtrapolator, G4TablesForExtrapolator, G4ErrorEnergyLoss | n | - | error propagation, not transport |

### 1.4 highenergy/ (23 headers)

| Geant4 class | QBBC | | Where |
|---|:--:|:--:|---|
| G4hBremsstrahlungModel | y | **V** | `em/muon_radiative.cuh` (`tests/test_hadron_radiative.cu`) |
| G4hPairProductionModel | y | **V** | `em/muon_radiative.cuh` |
| G4hBremsstrahlung, G4hPairProduction | y | **P** | process wrappers |
| G4AnnihiToMuPair, G4GammaConversionToMuons, G4eeToHadrons + G4eeCrossSections + 6 `G4ee*Model` + G4Vee2hadrons | y | - | registered by `G4EmExtraPhysics`; 11 classes, none ported |
| G4BetheBlochNoDeltaModel, G4BraggNoDeltaModel, G4ICRU73NoDeltaModel | opt | - | |
| G4hhIonisation, G4mplIonisation + 2 models | n | - | monopoles |

### 1.5 xrays/ (18 headers)

| Geant4 class | QBBC | | |
|---|:--:|:--:|---|
| **G4SynchrotronRadiation** | **y** | **-** | registered by `G4EmExtraPhysics`. Inert here anyway - this port has no magnetic field. |
| G4SynchrotronRadiationInMat | n | - | |
| G4Cerenkov, G4Scintillation, G4ScintillationTrackInformation | n | - | optical; QBBC has no `G4OpticalPhysics` |
| the 12 transition-radiation classes (G4*XTRadiator, G4XTR*Model, G4VXTRenergyLoss, ...) | n | - | |

### 1.6 utils/ (47 headers) - framework, not processes

Ported as behaviour rather than as classes: `G4EmParameters` (the defaults that differ between
leptons and hadrons - `finalRange`, `facrange`, `LateralDisplacement`, `mscStepLimit`, the
lowest-energy floors), `G4EmCorrections` (`em/em_corrections.cuh`), `G4ionEffectiveCharge`
(`tests/test_ion_charge.cu`), `G4VEnergyLossProcess::AlongStepDoIt` and
`G4VMultipleScattering::AlongStepDoIt` (inlined in `stepper.cuh` in Geant4's order),
`G4LossTableBuilder`/`G4LossTableManager` range-table construction, `G4VRangeToEnergyConverter`
(`data/production_cuts.cuh`). `G4EmCalculator` is used only to *drive the oracle*, never in
transport.

Not ported: `G4VAtomicDeexcitation`, `G4EmBiasingManager`, `G4EmSaturation`, `G4ElectronIonPair`,
`G4NIELCalculator`, `G4TransportationWithMsc`, `G4EmConfigurator`, `G4EmMultiModel`,
`G4EmElementSelector` (the port sums over elements directly), all messengers.

### 1.7 polarisation/ (24), adjoint/ (24), pii/ (10), dna/ (212)

**Nothing ported, and nothing in QBBC.** 270 headers. Polarised EM needs Stokes vectors the
track store does not carry; adjoint is reverse Monte Carlo; PIXE and Geant4-DNA are separate
physics lists.

---

## 2. hadronic/

**1 component of 1235 headers.** The honest summary is that this tree is where the port ends.

### 2.1 What is ported

| Geant4 class | | Where |
|---|:--:|---|
| G4ComponentBarNucleonNucleusXsc | **P** | `hadronic/barashenkov_xs.cuh` |
| G4PiData (the interpolation it calls) | **T** | same |
| G4BarashenkovData (the tables) | **T** | `data/barashenkov.hh`, 17 elements, 776 points |
| G4NuclearRadii (all 7 radii + both CoulombFactor) | **V** | `hadronic/xs/nuclear_radii.cuh` |
| G4NucleiPropertiesTableAME12 / G4NucleiProperties | **P** | `data/nuclei_mass_ame12.hh`, 3353 nuclides |
| G4PhysicsVector / LogVector / LinearVector (evaluation) | **V** | `hadronic/xs/physics_vector.cuh` |
| G4HadronNucleonXsc (PDG, NS, all three Kaon* forms) | **P** | `hadronic/xs/hadron_nucleon_xsc.cuh` |
| G4ComponentGGHadronNucleusXsc | **V** | `hadronic/xs/gg_hadron_nucleus_xsc.cuh` |
| G4ComponentGGNuclNuclXsc | **V** | `hadronic/xs/gg_nucl_nucl_xsc.cuh` |
| G4UPiNuclearCrossSection | **V** | `hadronic/xs/upi_nuclear_xs.cuh`, `data/upi_nuclear.hh` |
| G4BGGNucleonElasticXS | **V** | `hadronic/xs/bgg_nucleon_xs.cuh` |
| G4BGGNucleonInelasticXS | **V** | same |
| G4BGGPionElasticXS | **V** | `hadronic/xs/bgg_pion_xs.cuh` |
| G4BGGPionInelasticXS | **V** | same |
| G4IsotopeList (amin/amax/aeff) | **T** | `data/isotope_list.hh` |
| G4PhysicsVector::Retrieve (the G4PARTICLEXS format) | **T** | `data/particlexs_data.cuh` |
| G4ParticleInelasticXS (p, d, t, He3, alpha) | **V** | `hadronic/xs/particlexs.cuh` |
| G4NeutronInelasticXS | **V** | same |
| G4NeutronElasticXS | **V** | same |
| G4NeutronCaptureXS | **V** | same |
| G4GammaNuclearXS | **P** | same - below 130 MeV only, see below |
| G4NeutronGeneralProcess (its five tables and grid) | **V** | `hadronic/xs/neutron_general_xs.cuh` |
| G4CrossSectionDataStore (ComputeCrossSection, SampleZandA) | **V** | `hadronic/xs/sample_za.cuh` |
| G4HadXSHelper::FillPeaksStructure | **V** | checked in `tests/test_particlexs.cu`; P5 owns the port |
| G4ComponentSAIDTotalXS | **-** | not reachable - see below |
| G4ComponentAntiNuclNuclearXS | **-** | refused by name in `hadronic/xs/refusal.cuh` |
| G4NeutronGeneralProcess | **P** | `hadronic/neutron_general_xs.cuh`, `hadronic/neutron_wiring.cuh` and `step_neutral` in `physics/stepper.cuh`. **WIRED END TO END SINCE P8d - see 2.1.8**, and `P` rather than `T` for exactly one reason: the inelastic sub-process is selectable and has no final state, so it is refused by name (`HadronicRefusal::kNeutronInelastic`) with the kinetic energy it costs. `EnableNeutronGeneralProcess` is 1 in 11.1.1, so a neutron has one discrete interaction length over elastic + inelastic + capture summed and picks the sub-process from cumulative partials afterwards, and `G4NeutronTrackingCut::ConstructProcess` returns early so the 10 us cut lives inside it. Ported: the grid `PreparePhysicsTable` builds (400 log bins 1 keV - 20 MeV, 70 more to 100 TeV, **linear** interpolation - the spline flag is `false`), the `G4PhysicsVector::LogVectorValue` lookup, the sub-process choice including the **order swap** either side of 20 MeV, the time cut with its energy half correctly inert, the tables on the device per material, and the elastic (P5) and capture (P7) final states. `Upload` still refuses a table that arrives without its final states; what it checks is now the two sub-process data sets and the level scheme rather than the table itself. |

Nucleon-nucleus total, inelastic and elastic, Z = 2..92, 14 MeV - 1 TeV, protons and neutrons,
agreeing with Geant4 to **2e-15 over 10,738 points** (`tests/test_nucleon_xs.cu`).

`P` and not `T` because it is a *cross section with no process attached to it* - nothing calls
it during transport yet - and because `G4BGGNucleonElasticXS` has three branches this does not
cover, each refused loudly in the file header rather than approximated:

- **Z = 1** -> `G4HadronNucleonXsc::HadronNucleonXscNS`, x 1.0115. A different 436-line
  parameterisation, not a limiting case of this one. Matters most for water.
- **below 14 MeV** -> the Coulomb-barrier form.
- **above 91 GeV** -> Glauber-Gribov.

**Those three branches are now complete**, in `hadronic/xs/bgg_nucleon_xs.cuh`, which sits
above `barashenkov_xs.cuh` and calls it for the middle band. The file header of
`barashenkov_xs.cuh` still refuses them because *that file* still does not have them; the class
Geant4 puts above it does. `tests/test_hadronic_xs.cu` compares all four BGG data sets end to
end over Z = 1..92 and 1 keV to 100 TeV.

#### 2.1.1 P2 - the hadronic cross sections (added by the P2 branch)

**1,109,871 points against Geant4 11.1.1 at 1e-12 relative**, two tests:

| test | oracle | points | worst |
|---|---|--:|--:|
| `tests/test_hadronic_xs.cu` | `had_radii`, `had_masses`, `had_coulomb`, `had_hnxsc`, `had_ggcomp`, `had_bgg` | 906,700 | 2.0e-15 |
| `tests/test_particlexs.cu` | `had_particlexs`, `had_particlexs_iso`, `had_neutron_general`, `had_matelem`, `had_xspeaks` | 203,171 | **0** (bit-exact) |

Of the 33 comparison buckets, **29 read 0.000e+00** - all 3,279 AME2012 masses, all three
Coulomb factors, all six `G4HadronNucleonXsc` entry points for eleven projectiles, both
Glauber-Gribov components in all five columns, `G4UPiNuclearCrossSection`, both BGG pion
classes, all five G4PARTICLEXS data sets element and isotope, all five neutron-general tables
with their 9,415 grid nodes, and `G4HadXSHelper`'s peak structure. The four that do not:

| bucket | worst | what it is |
|---|--:|---|
| `RadiusNNGG`, `RadiusHNGG`, `RadiusKNGG`, `RadiusCB` | 2.1e-16 - 3.5e-16 | one ulp; these four multiply `G4Pow::Z13(A)` by an exponential where the others do not |
| `BGGNucleonElasticXS` | 2.0e-15 | `barashenkov_xs.cuh`'s own pre-existing residual - `tests/test_nucleon_xs.cu` reports the same 2e-15 over its 10,738 points |
| `BGGNucleonInelasticXS` | 8.6e-16 | the same |

Bit-exactness took three fixes, all of them constants and none of them physics; see docs/RISK.md
V44.

**V and not T** for all of it, for the same reason `barashenkov_xs.cuh` is: these are cross
sections with no process attached. P5 and P8 wire them.

**`G4GammaNuclearXS` is P.** Below its data files' top energy - 130 MeV for most elements -
element and isotope are exact. Above it, and for hydrogen at any energy, Geant4 needs
`G4PhotoNuclearCrossSection`, the 1821-line CHIPS parameterisation, which is not ported; the
transition region between the table top and 150 MeV needs it too, because it is a straight line
to `xs150[Z]` = CHIPS at 150 MeV. Refused by name (`XsRefusal::kPhotoNuclearCrossSection`) at
the point it would have been needed. 1,128 of the test's element points and 760 of its isotope
points land in that gap and are counted as refusals rather than passes.

**`G4HadronNucleonXsc` is P** for two branches nothing in QBBC reaches: `HyperonNucleonXscNS`
(s/c/b hyperons) and `SCBMesonNucleonXscNS` (s/c/b mesons). Both are refused by name.
`G4NucleiProperties` is **P** because `G4NucleiPropertiesTheoreticalTable` and the Cameron
formula behind it are not ported, so a nuclide outside AME2012 refuses rather than being given
a neighbour's mass. What IS there is the whole AME2012 table: `had_masses.csv` dumps
`GetNuclearMass` for every (Z, A) `IsInStableTable` admits - 3,279 nuclides, the 3353 table
entries less the 74 outside `IsInTable`'s own A <= 273 / Z <= 110 - and all 3,279 agree to the
last bit, with the membership sets equal in both directions. Both directions matter: an extra
nuclide the port claims to know is a mass it invented. P3 has committed its own AME12 table as
`src/data/ame12_masses.hh`; the two want unifying at integration, and `had_masses.csv` is the
oracle for whichever survives.

**`G4ComponentSAIDTotalXS` is not reachable and no `G4SAIDDATA` reader is needed.** Checked
rather than assumed: in the whole 11.1.1 source tree the class is *mentioned* in exactly one
place outside its own two files - an `#include` at line 47 of `G4BGGNucleonInelasticXS.cc` -
and never constructed. Nothing in `physics_lists` names it. `G4SAIDXSDATA` is read only by
`G4ComponentSAIDTotalXS.cc` itself.

**`G4HadronicProcess` does not tabulate the cross section.** It holds no `G4PhysicsVector` and
no `G4PhysicsTable` of it - grep the header and the source, there are none -
`PostStepGetPhysicalInteractionLength` calls `UpdateCrossSectionAndMFP`, and every arm of that
ends in `theCrossSectionDataStore->ComputeCrossSection(dp, currentMat)`. So for a charged
hadron the transport evaluates the model, not a table: the exact opposite of the neutron
(section 2.1.2) and of every EM process.

What `BuildPhysicsTable` does build, when the integral method is on
(`EnableIntegralInelasticXS` and `EnableIntegralElasticXS` both default true) and the particle
is charged, is the *shape* of the cross section per material: `G4HadXSHelper::FillPeaksStructure`
scans `nbin = G4lrint(log(emax/emin)*10/log(10))` points - ten per decade - from the process's
`minKinEnergy` (1 MeV, a private member with no getter) to 100 TeV and records up to three peak
energies and two dip energies. `UpdateCrossSectionAndMFP` uses them only to decide whether the
cached cross section may be reused, with `lambdaFactor = 0.8`. `fXSType` is `fHadTwoPeaks` for
pi+-, pi- and protons, `fHadOnePeak` for K+, and increasing/decreasing by charge otherwise.
`ref/oracle/had_xspeaks.csv` carries those five energies per material for protonInelastic,
hadElastic on a proton, and pi+-Inelastic; P5 owns using them.

The protonInelastic rows are **checked**, and they are the strongest single check in this
package. A peak energy is where a ten-per-decade scan of the *material-level* cross section
stops rising over 1 MeV to 100 TeV, so it is a nonlinear functional of ~800 evaluations: one
wrong point anywhere moves it, a uniform scale error does not move it at all, and it cannot be
right by accident. It is also the only end-to-end check here - element cross sections, atom
densities, the `max(xs, 0)` and the accumulation order at once. 35 points, seven materials by
five energies, all bit-exact. Only protonInelastic is reachable from `tests/test_particlexs.cu`
because QBBC gives that process exactly one data set, `G4ParticleInelasticXS`
(`G4HadronInelasticQBBC.cc:153`); the other three processes are BGG classes.

#### 2.1.2 THE NEUTRON GRID

`ref/oracle/hadronic_params.csv` says `EnableNeutronGeneralProcess = 1`, so a neutron in QBBC
has ONE discrete process. `G4NeutronGeneralProcess` builds a combined per-material table and
the transport reads that, not the three data sets:

*(P8d: the sentence below that attributes the flag to `G4HadronInelasticQBBC::ConstructProcess`
is wrong - the CONSTRUCTOR sets it, and the difference is a whole stage-1 configuration. See
2.1.8 and docs/RISK.md V60. The grid, the node counts and everything else in this section stand.)*

| table | grid | contents |
|--:|---|---|
| 0 | 401 nodes, 1 keV - 20 MeV | `sigEl + sigInel + sigCap`, macroscopic, 1/mm |
| 1 | same | `sigEl / sum` |
| 2 | same | `(sigEl + sigInel) / sum` |
| 3 | 71 nodes, 20 MeV - 100 TeV | `sigEl + sigInel` (capture is exactly zero there) |
| 4 | same | `sigInel / sum` |

The node counts are the load-bearing part: `nLowE = 100*G4lrint(log10(20 MeV / 1 keV))` is
`100*lrint(4.301) = 400` **bins** over 4.301 decades - 93 per decade, not 100 - and
`nHighE = 10*G4lrint(log10(100 TeV / 20 MeV)) = 10*lrint(6.699) = 70`. Reading the 100 and the
10 as bins-per-decade gives 431 and 67 nodes, which leaves both ends exact and moves every
interior node. `tests/test_particlexs.cu` compares the 9,415 node energies as well as the 6,608
values, and asserts 401 + 71 on the arithmetic as well, so it fails even if a future dumper
drops the grid column. This is docs/RISK.md V5 applied before a number was compared.

The oracle for it is the process's own `StorePhysicsTable` output, read back in binary
(`ref/dump/dump_hadronic_xs.cc`) - Geant4's `binVector`, not a second implementation of the
same arithmetic that could be wrong the same way.

`fTimeLimit = 10 us` lives in this process too: `G4NeutronTrackingCut::ConstructProcess`
returns immediately when a `G4NeutronGeneralProcess` exists, so QBBC's neutron time cut is that
member and not a `G4NeutronKiller`. Carried as a constant in `neutron_general_xs.cuh` for
P1/P8, which is where the row above says `G4NeutronTrackingCut` is absent.

#### 2.1.3 de_excitation - the shared tail of every inelastic model (P3)

`G4ExcitationHandler` and everything 11.1.1's defaults dispatch to, from the AME2012 mass table
up to a device-callable `deex::deexcite()`. The defaults are not assumed: they are dumped from
the install into `ref/oracle/deex_params.csv` and every dispatch decision below is made from
one of them. `V` and not `T` throughout because no inelastic model hands this module a fragment
during transport yet - the entry point exists, its caller does not.

| Geant4 class | | Where |
|---|:--:|---|
| G4DeexPrecoParameters (`SetDefaults`) | **V** | `deexcitation/deex_params.cuh`; 32 values compared exactly |
| G4Fragment | **V** | `deexcitation/fragment.cuh` |
| G4NucleiProperties, G4NucleiPropertiesTableAME12, G4NucleiPropertiesTheoreticalTable | **V** | `deexcitation/nuclear_masses.cuh`, `data/ame12_masses.hh` (3,353), `data/nuclei_theoretical.hh` (8,979); 18,407 (Z, A), all four dispatch branches, worst 0 |
| G4PairingCorrection, G4ShellCorrection, G4CameronGilbertPairingCorrections, G4CameronGilbertShellCorrections, G4CookPairingCorrections, G4CookShellCorrections, G4CameronTruranHilfPairingCorrections | **V** | `deexcitation/corrections.cuh`, `data/shell_pairing.hh`; 18,407 (Z, A), worst 0 |
| G4CameronTruranHilfShellCorrections, G4CameronShellPlusPairingCorrections | **V** | same files. Their tables are private statics behind a header-inline accessor this Windows Geant4 does not export, so neither can be dumped directly; the shell-plus-pairing table is checked through `G4FissionBarrier`, its only default consumer, over 4,963 (Z, A) at worst 2.1e-14 |
| G4EvaporationLevelDensityParameter, G4NuclearLevelData::GetLevelDensity | **V** | `deexcitation/corrections.cuh` |
| G4NuclearRadii, G4VCoulombBarrier, G4CoulombBarrier + its six ejectile subclasses, G4FermiCoulombBarrier, G4GEMCoulombBarrier | **V** | `deexcitation/coulomb_barrier.cuh`; 612 points, worst 4.0e-15 |
| G4NuclearLevelData, G4LevelReader, G4LevelManager, G4NucLevel | **V** | `data/level_data.cuh`, `data/level_index.hh`; the whole PhotonEvaporation5.7 dataset - 3,110 files, 3,108 managers, 174,411 levels, 268,190 transitions - reproduced bit for bit |
| G4KalbachCrossSection, G4ChatterjeeCrossSection, G4VEmissionProbability, G4EvaporationProbability + its six subclasses, G4EvaporationChannel | **V** | `deexcitation/evaporation.cuh`; 2,600 points, worst 4.4e-16 |
| G4GEMChannel, G4GEMProbability and the 60 `G4<Xx>GEMChannel` / `G4<Xx>GEMProbability` pairs | **V** | `deexcitation/gem.cuh`, `data/gem_levels.hh` (60 channels, 1,617 excited states); 4,320 probabilities, worst 0 |
| G4CompetitiveFission, G4FissionBarrier, G4FissionProbability, G4FissionParameters, G4FissionLevelDensityParameter | **V** | `deexcitation/fission.cuh`; 20,000-odd points, worst 2.1e-14. `EmittedFragment`'s exception after 100 failed splits is refused by name on the output, because a kernel cannot throw |
| G4PhotonEvaporation, G4GammaTransition | **V** | `deexcitation/photon_evaporation.cuh`; 112 probabilities, worst 5.2e-16 |
| G4FermiBreakUpVI, G4FermiFragmentsPoolVI, G4FermiFragment, G4FermiPair, G4FermiChannels, G4FermiDecayProbability | **V** | `deexcitation/fermi_breakup.cuh`; pool of 991 fragments, 450 pairs and 4,679 channels, 2,802 predicate rows and every cumulative probability at worst 0 |
| G4UnstableFragmentBreakUp | **V** | `deexcitation/excitation_handler.cuh` |
| G4Evaporation, G4EvaporationDefaultGEMFactory (the 68-channel order) | **V** | `deexcitation/excitation_handler.cuh` |
| G4NistManager::GetIsotopeAbundance, as the predicate the handler and G4Evaporation use it as | **V** | `data/natural_isotopes.hh`; 311 (Z, A) of 2,908 tabulated |
| **G4ExcitationHandler** | **P** | `deexcitation/excitation_handler.cuh`, entry point `deex::deexcite()`. Statistically validated on 17 campaigns x 20,000 events, 2.48M products, worst 4.1 sigma. **Refused by name:** the hyper-nucleus path (`nL != 0`), and the PDG code of an excited heavy ion - `G4IonTable::GetIon` snaps E* to G4ENSDFSTATE and puts a *run-dependent* isomer index in the code's last digit, so the module emits (Z, A, E*, floating level) and a PDG code only for the eight species with a fixed definition |
| G4StatMF and its nine helpers | **-** | unreachable: `fMinExPerNucleounForMF` is 200 GeV per nucleon. The handler's condition is reproduced and refuses by name, so raising the parameter is reported rather than silently evaporated |
| G4NuclearPolarization, G4PolarizationTransition | **-** | `fCorrelatedGamma` is false by default; refused by name in `photon_evaporation.cuh` |
| G4FermiPhaseSpaceDecay | **-** | not a gap: `G4FermiBreakUpVI` is a cascade of two-body decays and never constructs it. Only `G4BinaryCascade` does |

Tests: `test_deex_nuclear.cu`, `test_deex_levels.cu`, `test_deex_probs.cu`,
`test_deex_models.cu` (all exact) and `test_deex_breakup.cu` (statistical).

#### 2.1.4 pre_equilibrium/exciton_model - the exciton stage and the cascade hand-over (P6)

`G4PreCompoundModel::DeExcite` and everything under it, plus `G4GeneratorPrecompoundInterface`,
the class every cascade and string model in QBBC hands its residual to. The entry point is
`preco::deexcite(fragment, excitons, ...)`, and it writes **one** product list: the
pre-equilibrium ejectiles first, then P3's whole equilibrium cascade, with
`PrecoStatus::n_preco_products` marking the boundary - so a caller sees a single list and does
not have to know where the stages met. `V` and not `T` throughout because the callers do not
exist yet: P9 (binary cascade), P10 (Bertini) and P11 (FTFP) are what reach this, and the plain
data types they fill in - `CascadeTrack`, `HitNucleon`, `WoundedNucleus` - are defined here
because this is the file that reads them.

**Why the exact half is exact.** Geant4 lets a random engine be installed, so
`ref/dump/dump_precompound.cc` drives every sampler in the package under an eight-value uniform
cycle. `SampleKineticEnergy`, `ChooseFragment`, `PerformTransition` and `PerformEmission` become
deterministic functions of (inputs, phase), and each row compares kinetic energies,
four-momenta and exciton counts **and the number of deviates the call consumed** - the part a
transcription can get wrong while still producing a plausible answer.

| Geant4 class | | Where |
|---|:--:|---|
| **G4PreCompoundModel** (`DeExcite`, `PerformEquilibriumEmission`, `ApplyYourself`) | **V** | `precompound/precompound_model.cuh`, entry point `preco::deexcite()`. 640 equilibrium exciton numbers, 1,424 gate verdicts including the four tokens the entry gate and the loop gate differ in; 15 campaigns x 20,000 events, 2.35M products, worst 3.8 sigma. `ApplyYourself` has no exact oracle and cannot have one - the initial fragment is a local - so it is checked through its consequences over 6 cases x 20,000 events, worst 2.7 sigma. The 1000-iteration guard is a JustWarning in Geant4, so it is REPORTED and the products are still produced |
| G4VPreCompoundFragment (+ `.icc`), G4PreCompoundFragment, G4PreCompoundNucleon, G4PreCompoundIon, G4PreCompoundNeutron / Proton / Deuteron / Triton / He3 / Alpha | **V** | `precompound_fragment.cuh`; 19,200 emission-probability integrals over all five `OPTxs`, 33,280 `Initialize` outputs, 8,456 sampled kinetic energies with their draw counts - all at worst 0. `SetOPTxs` is public and unlocked, which is what makes `GetAlpha`, `GetBeta` and all five inverse cross sections measured rather than transcribed-and-hoped |
| G4PreCompoundEmission (`rho`, `AngularDistribution`, `PerformEmission`), G4PreCompoundFragmentVector, G4PreCompoundEmissionFactory, G4VPreCompoundEmissionFactory | **V** | `precompound_emission.cuh`; 5,120 channel choices, ejectile four-momenta and residual states, worst 1.2e-14. The factory's order is n, p, d, **alpha, t, He3** - not the evaporation module's - and `ChooseFragment` walks it, so it is observable and is checked |
| G4VPreCompoundTransitions, G4PreCompoundTransitions | **V** | `precompound_transitions.cuh`; all four (`fUseCEM`, `fNeverGoBack`) combinations, 102,400 rates and 76,800 post-transition exciton configurations with both draw counts, worst 0 |
| G4GNASHTransitions | **V** | same file - and it is a **dead branch**. It never assigns `TransitionProb1/2/3`, the model reads all three back as the 0.0 the base constructor left, and its first equilibrium test is then `0 <= 0`: setting `fUseGNASH` switches pre-equilibrium emission off entirely. Measured on 5,120 oracle rows, transcribed as written, and reported by name rather than silently producing pure equilibrium emission. docs/RISK.md V49 |
| G4GeneratorPrecompoundInterface (`Propagate`, `PropagateNuclNucl`, `MakeCoalescence`) | **P** | `generator_interface.cuh`; four wounded nuclei x 8 phases, exact on which tracks escaped, on their summed four-momentum and on the FTF/QGS branch verdict. **Refused by name:** `G4DecayKineticTracks`, the anti-nucleus and hypernucleus arms of `PropagateNuclNucl`, and `Propagate`'s silent drop of a residual with Z > A, which is reported instead. Its QGS arm dereferences a null `GetPrimaryProjectile()` for any bare wounded nucleus - docs/RISK.md V50 |
| G4HETCEmissionFactory and the ten `G4HETC*` classes | **-** | refused by name in `precompound_emission.cuh`. `fUseHETC` is false and `G4DeexPrecoParameters`' setters return early outside `G4State_PreInit`, so no run reaches them and there is no oracle a transcription could be checked against - P3's Weisskopf-width case |
| G4LowEIonFragmentation | **-** | not a gap: named in five places in the whole 11.1.1 tree - its own two files, its package's CMake, the History file and `G4PhysicsModelCatalog`'s name list - and constructed nowhere, so QBBC cannot reach it |
| `useSICB` (G4VPreCompoundFragment, G4PreCompoundEmission::UseSICB, G4PreCompoundFragmentVector::UseSICB) and `DeltaR` (G4GeneratorPrecompoundInterface) | **-** | nothing to port: both are written, plumbed and never read. The barrier `useSICB` gated is now the unconditional `elim = theCoulombBarrier*0.5` in `Initialize`, so every charged channel's integral starts at HALF the barrier whatever the flag says. docs/RISK.md V49 |

Tests: `test_precompound.cu` - 374,345 exact comparisons at worst 1.2e-14, plus 21 statistical
campaigns of 20,000 events.
#### 2.1.5 Neutron radiative capture (P7), and the first wiring (P8)

`nCapture` end to end - `G4NeutronCaptureProcess` through P5's framework, on P2's
`G4NeutronCaptureXS`, with `G4NeutronRadCapture` as its one model and P3's photon evaporation
as that model's cascade. Plus the wiring layer P8 adds around it: the stage switch, the
per-process refusal ledger, and `G4Decay` in the steppers.

| Geant4 class / function | QBBC | | Where |
|---|:--:|:--:|---|
| G4NeutronRadCapture::ApplyYourself, both branches (A <= 1 two-body, A >= 5 compound) | y | **V** | `hadronic/capture/neutron_rad_capture.cuh` |
| `lowestEnergyLimit` (the A <= 1 branch's "emit nothing and still kill the neutron") and `minExcitation` | y | **V** | same. H3 + n is the case in the oracle grid where it fires: Q = -1.60 MeV, so the capture is unbound and emits nothing |
| G4DynamicParticle's four-momentum constructor / `Set4Momentum` (the `EnergyMRA2` shell test) | y | **V** | same. The comparison is on mass SQUARED with an allowance of 1e-10 MeV^2, and one ulp of a deuteron's mass squared is 7.8e-10 - so the middle branch always runs for a nucleus and a residual's kinetic energy is `t - sqrt(t^2-|p|^2)`, never `t - M(Z,A)`. Writing it the other way is 4.15e-10 out, measured |
| G4NeutronCaptureProcess (the composition: `fHadNoIntegral`, the evaluated cross section, SampleZandA, one model, CheckResult, FillResult) | y | **V** | `hadronic/capture/capture_process.cuh` |
| G4NucleiProperties::GetNuclearMass, as the capture mass balance uses it | y | **V** | `deexcitation/nuclear_masses.cuh`; 75 numbers over 15 targets and their compounds, worst 0 |
| G4PhotonEvaporation::BreakUpChain as the capture cascade | y | **V** | `deexcitation/photon_evaporation.cuh`, reused unchanged except for the two fixes below |
| **G4Decay in the transport** - in flight as a discrete competitor, at rest on the dying branch | y | **T** | `physics/stepper.cuh` (`step_hadron`, `step_neutral`) through `hadronic/wiring.cuh`. See section 3 for the channel-by-channel table, now `T` |
| `theNumberOfInteractionLengthLeft` (`G4VDiscreteProcess`'s carried count) | y | **-** | Not carried, and not an approximation: over one step lambda is a constant (Geant4 evaluates it at the pre-step energy too), so `-log(u)*lambda` re-drawn per step and `n*lambda` decremented per step are the SAME distribution - the exponential is memoryless. `decay_in_flight_length`'s header has the argument and docs/RISK.md V22 has the reason (a wider `TrackState` is what the register budget cannot afford). Every EM discrete process in this stepper already works this way |
| G4HadronStoppingProcess::AtRestGetPhysicalInteractionLength as a COMPETITOR | y | **P** | `hadronic/wiring.cuh`. It returns 0.0, which is a pre-emption and not a race: no `-log(u)*tau` beats zero, so a stopped pi-, K-, mu- or pbar is captured in Geant4 and never decayed. The port refuses it by name (`HadronicRefusal::kStopped*`) with the rest mass it costs - 139.6 MeV for a pi-, 1876 for a pbar - and in `kStage1`, where Geant4's own three at-rest captures are inactivated, both sides decay instead |

**What is refused, by name.**

- **The isomer index of an excited residual.** Geant4 ends a capture with
  `G4IonTable::GetIon(Z, A, eexc, noFloat, 0)`, which snaps E* onto G4ENSDFSTATE and writes a
  run-dependent isomer digit into the PDG code's last place (1000260572 for Fe57 at 136 keV).
  P3 refuses that digit and this module does too; 101 of the 600 deterministic oracle points
  have a non-zero one, so it is not a corner. The MASS is a separate question and was measured
  rather than refused - all 568 residuals agree with Geant4's snapped mass to the last bit - so
  `CaptureRefusal::kIsomerIonMass` reads "this row went through the snapping", not "this number
  is wrong". docs/RISK.md V40 is why it had to be asked.
- **A stale `fIndex` across captures.** `G4NeutronRadCapture` owns one `G4PhotonEvaporation` for
  the life of the run, and `G4LevelManager::NearestLevelIndex(energy, index)` short-circuits when
  the hinted level is within 10 eV of the energy asked for - so the answer can depend on the
  previous capture. It agrees with a fresh state whenever the predecessor's cascade reached the
  ground state (`GenerateGamma` leaves `fIndex = 0` there); only an isomer-terminated cascade
  leaves it elsewhere. `neutron_rad_capture_apply` takes the state as an optional argument, the
  deterministic oracle uses a fresh model per call and the statistical one uses a single model
  for 20,000 calls as a run does, and the 3.49 sigma the second passes at is the bound on the
  difference.
- **Everything P3 refuses inside photon evaporation**, unchanged: the correlated-gamma angular
  correlation, the hyper-nucleus path, and the PDG code of an excited heavy ion.
- **A cascade longer than the final state can hold** (`kSecondaryOverflow`), and an unphysical
  target (`kUnphysicalTarget`).

**Two ulps that only a recoil could see, fixed here.** `deexcitation/fragment.cuh`'s
`LorentzVector::boost_vector` divided by the energy; `HepLorentzVector::boostVector()` is
`pp * (1./ee)` (LorentzVector.cc:189), which is docs/RISK.md V37's finding for the elastic
models arriving a second time. And P3's `generate_gamma` discarded the sampled level lifetime.
Neither was visible to P3's own five tests, whose fragments are at rest - the boost vector is
then zero and the arithmetic never runs. Putting the division back moves 5493 direction
components from exactly 0 to 1.4e-13 and the kinetic energies to 2.0e-16; discarding the
lifetime again makes every secondary time wrong by 100%.

**The numbers.** `ref/oracle/capture_masses.csv` 75 masses, worst 0. `capture_det.csv`
`ApplyYourself` under a prescribed eight-value uniform cycle, so the model is a deterministic
function of (target, energy, phase): 600 calls, 1831 secondaries, and secondary count, uniforms
consumed, primary energy change, both secondary masses, kinetic energy, direction and time all
worst **0**. `capture_stat.csv` 20,000 captures per (target, energy) at a fixed seed: 1162
comparisons of moments and histogram bins, worst **3.49 sigma** against a stated limit of 5.
Tests: `test_capture.cu`, `test_wiring.cu`.

**The stage-1 like-for-like.** Geant4 QBBC B1 with every `*Inelastic`, the three at-rest
captures, `muonNuclear`, `hBrems`/`hPairProd`/`muBrems`/`muPairProd`, `CoulombScat` and
`NeutronGeneralProc` inactivated - `Decay` and `hadElastic` left active - against the port in
`HadronicStage::kStage1`, 500,000 events per run per side, nine species, 27 runs. Macros,
script, the full table and the Geant4 process dump of the stage are in `ref/b1hadron/`
(`stage1_README.md`). Against the column that is the true like-for-like for a port with decay
and no elastic scattering:

| | proton | alpha | mu+ | pi+ | K+ | mu- | pi- | K- | neutron |
|---|---|---|---|---|---|---|---|---|---|
| diff | -0.29% | -0.08% | -0.24% | -0.25% | +0.30% | +3.72% | +4.86% | +9.83% | 0 vs 0 |
| sigma | 1.9 | 0.5 | 1.6 | 1.7 | 1.9 | 24.3 | 31.7 | 58.3 | - |

The five positive-or-neutral species agree at 0.5 to 1.9 sigma with `Decay` active on both
sides, which is what P8 was for. The three negatives carry docs/RISK.md **V44** - a range-table
interpolation rule that differs for the negative of each charge pair, in this port's EM code,
found and fixed by P14 on its own branch - and not a decay defect: the port's own pi+/pi- pair
agrees with itself to 0.2% and Geant4's differs by 5%. **mu+ is the one row with nothing
missing**: `G4HadronElasticPhysics::ConstructProcess` registers an elastic process for no
lepton, so the two Geant4 columns for mu+ are bit-identical and its 1.6 sigma is a complete
comparison. What `hadElastic` is worth, from the difference between the two Geant4 columns, is
-2.42% for the proton, -4.08% for pi+, -2.16% for pi-, -0.22% for alpha, +0.22% for K+ and
+0.72% for K-.

**What P8 wired, and what it did not.** `G4Decay` is reached, for every species Geant4 gives
one and P4 has a table for, in flight and at rest. `hadElastic` and the neutron general
process are NOT reached, and the reason in both cases is data that is not on the device rather
than a model that is not written:

- **hadElastic** needs `SampleZandA` to pick an isotope, and no table in this port holds
  isotope ABUNDANCES - `data/natural_isotopes.hh` is deliberately the set and not the weights,
  `data/isotope_list.hh` is amin/amax/aeff, and `data::Material` has no isotope field to
  upload one into. The recoil is (Z, A)-resolved (G4ChipsElasticModel's tables are
  per-isotope), so there is no version that draws an element and stops. Its size per species is
  the difference between the two Geant4 columns of `ref/b1hadron/stage1_compare.ps1`.
- **The neutron general process** needs P2's five tables uploaded per material, and its capture
  sub-process needs P3's PhotonEvaporation5.7 level data on the device - 174,411 levels and
  268,190 transitions, with no upload path today. `TransportEngine::Upload` refuses a
  cross-section table that arrives without its final states, so the state is enforced and not
  merely current.
  *(P8b: the level data has an upload path and a device test now - section 2.1.6 - so what is
  left of this item is P2's five combined tables and the sub-process branch in `step_neutral`.
  The refusal in `Upload` still stands and is still the thing that keeps them from arriving
  separately.)*
  *(P8d: CLOSED. `host/neutron_upload.cuh` puts the five combined tables and the two
  per-process data sets on the device - 7.43 MB for a five-material scene - the level data is
  uploaded unconditionally now, and `step_neutral` selects and applies. Section 2.1.8.)*

#### 2.1.6 The isotope abundances, hadElastic and CoulombScat in the steppers (P8b)

The two things P8 named as blocking `hadElastic` were the isotope abundances and, for the
neutron only, P3's level data. The first is closed and the process it blocked is wired for every
charged hadron Geant4 gives one to; the second is not, and the neutron's own section below says
what is left.

| Geant4 class / function | QBBC | | Where |
|---|:--:|:--:|---|
| `G4NistElementBuilder`'s isotope table - `relAbundance`, `nIsotopes`, `nFirstIsotope`, `idxIsotopes`, and the two normalisations `AddElement` and `G4Element::AddIsotope` apply | y | **V** | `data/isotope_abundance.hh`, generated by `tools/extract_isotope_abundance.pl`. 2908 tabulated isotopes over 107 elements, 311 with abundance > 0. Compared bit for bit against a QBBC-initialised `G4NistManager` for all 104 buildable elements: worst **0** on `GetRelativeAbundanceVector` and **0** on `GetIsotopeAbundance` |
| `G4EmUtility::SampleRandomIsotope` / `G4VEmModel::SelectIsotopeNumber` | y | **V** | same file, `nist_sample_isotope_n`. The subtractive loop as written, and no uniform for a single-isotope element |
| `G4VCrossSectionDataSet::SelectIsotope` (the base, abundance-weighted) through `G4CrossSectionDataStore::SampleZandA` | y | **V** | `hadronic/xs/sample_za.cuh`, now with `store_sample_za_rng` so the DRAW COUNT is Geant4's: no element uniform for a one-element material, no isotope uniform for a one-isotope element |
| `G4HadronElasticProcess` in the transport, for p, pi+-, K+-, d, t, He3, alpha | y | **T** | `physics/stepper.cuh` (`step_hadron`) through `hadronic/elastic_wiring.cuh`. The (cross section, model) pair per species is `had::elastic_channel`, transcribed from `G4HadronElasticPhysics::ConstructProcess` |
| `G4CoulombScattering` in the transport, for e+- above 100 MeV and every singly-charged hadron | y | **T** | `physics/stepper.cuh` (`step_lepton`, `step_hadron`) through `em::coulomb_fire`. P14's model, unchanged |
| `G4VEmProcess`'s integral-approach rejection for `CoulombScat` (`fEmIncreasing`) | y | **P** | `coulomb_apply`. The rejection is there and consumes its uniform; Geant4's CACHING of `preStepLambda` across steps (`ComputeIntegralLambda`'s 20%-of-energy window) is not, because this transport re-draws every interaction length every step. Both give the same net rate |
| `G4HadronicProcess`'s integral rejection for `hadElastic` (`fHadTwoPeaks` for p and pi+-, `fHadOnePeak` for K+) | y | **V** | `had::elastic_apply`, through P5's `integral_xs_rejects`. 27 of 132 direct calls rejected in `tests/test_step_hadron.cu` |
| `G4ElasticHadrNucleusHE`'s `G4ElasticData` tables, built and uploaded | y | **V** | `host/hadronic_upload.cuh`. Per (pion, Z) for the Z the scene contains - 18 tables and 0.35 MB for B1's elements - because the model reads one only for Z > 1 |
| `G4BGGNucleonElasticXS` / `G4BGGPionElasticXS` `BuildPhysicsTable` output, uploaded | y | **V** | same file, 3352 B and 4600 B |
| `G4AntiNuclElastic` + `G4ComponentAntiNuclNuclearXS` (the antiproton) | y | **-** | Refused by name. `had::elastic_channel(kAntiProton)` is `kAntiNucleusRefused` and the cross section is zero, so an antiproton draws no hadronic interaction length at all |
| `G4NuclNuclDiffuseElastic` (`G4IonElasticPhysics`, GenericIon) | y | **-** | Reachable since P8c transports the ion, and refused by name rather than left unreachable: `had::elastic_channel(kGenericIon)` is `kIonDiffuseNotWired` and the cross section is zero, so an ion draws no hadronic interaction length. Both halves exist - `xs::ggnn_elastic_element` and `elastic/nucl_nucl_diffuse_elastic.cuh` - and what is missing is the channel, which needs the projectile to be `xs::generic_ion(Z, A)`. Bounded at ~1e-9 per recoil: a micrometre of range against a metre of mean free path. See section 2.1.7 |
| `G4NuclearLevelData::UploadNuclearLevelData` - PhotonEvaporation5.7 on the device | y | **V** | `host/level_upload.cuh`. 3108 managers, 174,411 levels and 268,190 transitions, **9.52 MB**. Every manager's level count, level energies, lifetimes, spins and transitions are read through the cascade's own accessors on the host and on the device and compared **exactly**: 0 disagreements. The cascade on top of it - `G4NeutronRadCapture::ApplyYourself` through `G4PhotonEvaporation::BreakUpChain` - runs 448 captures over 14 targets and 4 energies on both sides: 1783 secondaries, worst **5.82e-11 MeV** absolute on an energy and **2.23e-11** on a direction component. `tests/test_capture_device.cu` |
| the same, at initialisation, whether a neutron arrives or not | y | **T** | `TransportEngine::SetNuclearLevelData` was OFF by default, which Geant4's `G4ExcitationHandler::SetParameters` is not, because the one consumer - the capture sub-process of `G4NeutronGeneralProcess` - was not wired and `read_all_level_data` opens 3110 files against a B1 run whose whole transport is 750 ms. *(P8d: ON by default now, which is what this row said would happen "the day the neutron is wired". The setter survives for a gamma- or electron-only run that cannot make a neutron, and `Upload` refuses the combination of a neutron cross section with no level scheme - measured, that is not a missing gamma but three times as many, docs/RISK.md V60.)* |

**What was found on the way, and all three are in docs/RISK.md.**

- **V54**: `G4NistManager::FindOrBuildElement(105)` aborts the process. The NIST table carries
  107 elements and gives every trans-uranic a fabricated 100% abundance on one isotope, so
  `BuildElement` really does build a `G4Element` for it - and `G4Element::AddIsotope` then indexes
  `G4AtomicShells`' `[105]` tables. 104 is the highest Z at which a material can exist in 11.1.1.
- **V55**: inlining the elastic package into `run_step_hadron` killed ptxas with an access
  violation. Four functions are `__noinline__` now, which is also the right answer for the hot
  path: the elastic branch fires 16 times in 900 steps and `CoulombScat`'s mean free path is
  158 m. The entry carries both the one-kernel reproducer series and the engine's own.
- **V56**: both kaons are 0.6% high in the stage-1 table, 3.4 and 3.7 sigma, and by the same
  amount for both charges - which is what rules out V44. Two candidates, neither excluded.

**The register and stack cost**, `transport_run.cu` with `-Xptxas -v`, against the same file on
main. No change in register count and +816 bytes on the hadron kernel's frame against the
16384-byte limit `Upload` sets:

| kernel | registers | stack frame | spill st/ld |
|---|---|---|---|
| `run_step_hadron` | 255 -> 255 | 3696 -> **4512** B | 68/36 -> 96/52 |
| `run_step_neutral` | 255 -> 255 | 3072 -> **3728** B | 80/36 -> 92/52 |
| `run_step_lepton` | 255 -> 255 | 2992 -> **3056** B | 56/20 -> 76/28 |
| `run_step_gamma` | 255 -> 255 | 3024 -> **2400** B | 44/20 -> 404/868 |

**The numbers.** `ref/oracle/isotopes.csv` 308 isotopes over 104 elements, worst **0** twice.
`ref/oracle/isotope_zanda.csv` 36 blocks, 135 element and 378 isotope frequencies from 200,000
`SampleZandA` draws each, worst **2.60** and **3.55 sigma** against a 5 sigma gate.
`tests/test_step_hadron.cu` runs `step_hadron` on the device and on the host for 900 steps of six
species and compares every field: worst relative deviation **3.20e-13**, with the step status,
the process that defined the step, the secondary count and every secondary's species compared
EXACTLY. The two new final states are called directly on both sides for 132 (species, energy)
cases each: `elastic_apply` worst **5.80e-11** (a recoil energy is a difference of two
target-mass-sized numbers - see the tolerance's own note), `coulomb_fire` worst **3.80e-13**.
Tests: `test_isotopes.cu`, `test_step_hadron.cu`, and `test_hadronic_process.cu` unchanged.

**The mean free paths, because they are what the counts mean.** In water:
`hadElastic` 2013 mm for a 200 MeV proton, 466 mm for a 200 MeV pi+, 836 mm for pi-, 6207 mm for
a 400 MeV K+, 829 mm for an 840 MeV alpha, and no process at all for a muon. `CoulombScat`
158 m for the proton, 434 m for the pion, 483 m for a 1 GeV muon - against B1's 300 mm envelope,
so **`G4CoulombScattering` cannot move a hadron's B1 dose however it is wired**, which is why
Geant4's stage-1 numbers barely change when it is inactivated. That is a prediction the port now
makes rather than an assumption it rests on.

#### 2.1.7 The recoil ion transported, and the depth-dose reference made like for like (P8c)

P8b wired `hadElastic` for every charged hadron, and `build_all.bat`'s proton depth-dose gate
then failed on what that produced:

```
*** 929 SECONDARIES OF SPECIES THIS PORT CANNOT TRANSPORT, 515.136 MeV ***
      GenericIon          929        515.136 MeV
  total port          599484.86 MeV of 600000 in  (99.9141%)
  FAIL: port did not deposit the beam energy - 99.9141% of it
  plateau (0-59.8 mm)  port/G4 per proton = 1.01322  (1.322%)
  FAIL: plateau dose off by 1.322%, limit 1.000%
```

Two failures and two different causes: the recoil nucleus of an elastic scatter had no kernel,
and the reference had been generated with no hadronic process at all.

**`ParticleType::kGenericIon` names two particles, and separating them is the whole of the
first fix.** `G4GenericIon` is a placeholder definition - 938.2723 MeV, charge 1 - whose
`G4ionIonisation` owns the dE/dx and range TABLES. An oxygen recoil is a different particle:
`G4IonTable::CreateIon(8, 16, 0)` builds a `G4Ions` with `GetNucleusMass(8,16)` and charge 8 and
gives it G4GenericIon's own process manager by copying its `g4particleDefinitionInstanceID`
(`G4IonTable::AddProcessManager` - it never calls `SetProcessManager`), and
`G4EmTableUtil::CheckIon` then makes `G4VEnergyLossProcess::PreparePhysicsTable` return early
for every concrete ion, so GenericIon owns the only tables that exist. So the SPECIES selects
the processes and the DEFINITION is the kinematics, and `em::SteppedHadron` is the pair.

| Geant4 class / function | QBBC | | Where |
|---|:--:|:--:|---|
| `G4IonTable::CreateIon`'s definition - mass `GetNucleusMass(Z,A)`, charge `Z*eplus`, `isIon` | y | **V** | `em::ion_particle_def` in `em/hadron_range.cuh`, on `data::nuclear_mass` - the same function the elastic recoil's own kinematics used, so a transported recoil is the particle that was emitted. Checked against `ref/oracle/ion_definitions.csv` |
| `G4IonTable::CreateIon`'s spin and magnetic moment (`FindIsotope(Z,A,E)->GetiSpin()/2`, `->GetMagneticMoment()`, i.e. columns 6 and 7 of `$G4ENSDFSTATEDATA/ENSDFSTATE.dat`) | y | **-** | **Refused by name.** This port reads PhotonEvaporation's level scheme and not ENSDFSTATE's ground-state spin and moment table, so `ion_particle_def` reports spin 0 and `mag_moment2` -1 - exactly right for an even-even nuclide (C12, O16, Ca40 all have 2J = 0) and wrong for one with spin (N14 has 2J = 2). The two are read in one place that changes an answer, the projectile form-factor rejection inside `em::sample_hadron_delta`, and `step_hadron` refuses the ion's whole delta-ray channel there rather than sampling it with a spin it cannot state - `had::HadronicRefusal::kIonDeltaRay`. Unreachable for anything this port makes: an ion's window needs `tmax > cut`, so water's 350 keV cut needs beta^2 gamma^2 > 342, above ~17 GeV per nucleon |
| `G4VEnergyLossProcess`'s `massRatio` / `chargeSqRatio` / `reduceFactor` for an ion, and `PostStepGetPhysicalInteractionLength`'s `if(isIon)` refresh of the charge from `currentModel->ChargeSquareRatio(track)` | y | **V** | `em::hadron_mass_ratio` and `em::hadron_charge_sq_ratio` on a `SteppedHadron`. `massRatio` is `m(G4GenericIon)/m(ion)` - the literal `0.9382723*GeV` and NOT `units::proton_mass_c2`, which differ by 3e-7 - and `chargeSqRatio` is `G4ionEffectiveCharge`'s effective charge times its `chargeCorrection`, squared, recomputed at the pre-step energy |
| `G4BraggIonModel` / `G4BetheBlochModel` for an ion, split at 2 MeV per nucleon | y | **V** | Not called: the table IS the split. `G4ionIonisation::InitialiseEnergyLossProcess` sets `eth = 2 MeV * m(GenericIon)/m_p` and selects the model at the SCALED energy, so the boundary in the ion's own energy is `2 MeV * m_ion/m_p`, which is what reading GenericIon's row at `E*massRatio` reproduces exactly. docs/PORTED.md 4.3's rule: Geant4 does not run the model, it runs a table built from it |
| `G4ionIonisation`'s `SetLinearLossLimit(0.02)` | y | **T** | `step_hadron`. `G4EmParameters::LinearLossLimit` is 0.01 and `G4ionIonisation`'s constructor overrides it, so the alpha, He3 and a generic ion invert the range table at twice the fractional loss a proton does. This was a flat 0.01 for all of them; it decides which of `AlongStepDoIt`'s two expressions computes a step's loss and cannot move a total |
| `G4IonFluctuations` with the ion's effective charge (`SetParticleAndCharge(part, q2)` under `if(isIon)`) | y | **T** | `step_hadron`. The ALPHA keeps the bare 4 - `G4EmTableUtil::CheckIon` excludes deuteron, triton, alpha+ and alpha by name, so `SetParticleAndCharge` is never called for it and `G4IonFluctuations::InitialiseMe` sets `effChargeSquare = charge*charge`. He3 and every real nucleus are not on that list and get the dynamic ratio; He3 was passing a bare 4 until now |
| `G4hMultipleScattering("ionmsc")` with `G4UrbanMscModel`, `fMinimal`, `facrange` 0.2, no lateral displacement | y | **T** | **No longer substituted, since P8e.** The model is general and oracled (section 1.1), the dispatch in `step_hadron` is live - `kUrbanIonMscWired` is true - and the three configuration numbers are on every row of `ref/oracle/ion_msc_step.csv`, read off the process on the particle's own manager rather than assumed. What held it off for a package was a compiler wall in `transport_run.cu` and not physics: docs/RISK.md **V63**, **V65** for the split that removed it, **V66** for what the substitution cost. P8c's own bound on that cost turned out to be the right shape and the wrong reason - a recoil's msc really does nothing, but not because it "dies on its first step": `ComputeTruePathLengthLimit` returns before the step limit when `currentRange*doverrb < presafety`, which for a range under 10 um is true anywhere but a hair from a boundary |
| `G4VEmModel::CorrectionsAlongStep` under `if(isIon)` - the `q2(E_mid)/q2(E_pre)` correction `G4VEnergyLossProcess::AlongStepDoIt` applies to an ion and not to an alpha | y | **-** | Not applied. Both models return immediately unless `eloss >= 5%` of the pre-step energy, so it is a correction on the long steps of a slowing ion. Absent rather than approximated, and it is why the ion's `GetDEDX` column of `ion_tables.csv` is the right thing to compare a TABLE against: `G4EmCalculator::GetDEDX` runs the same call over a 1 nm step, where it is inert by its own guard |
| `G4RadioactiveDecay` for an unstable nuclide | n | **-** | Not in QBBC's chain at all (docs/HADRONIC_PLAN.md section 2), so a real nucleus never decays here and `step_hadron` says so rather than relying on G4GenericIon's placeholder PDG code being absent from every decay table |
| `TrackSpeciesIndex` entries and kernel instantiations for He3 and GenericIon | - | **T** | `core/track_buffer.cuh`, `host/transport_run_impl.cuh`. He3 was pure plumbing - its dE/dx, effective charge, fluctuation model and elastic channel were all in place and it had no index |
| the nuclide on the track | - | **T** | `TrackState::ion_za`, `z*512 + a` in an `unsigned short`. **It costs the struct nothing**: `species` is an int at offset 0 and `pos` is a `Vec3<double>` needing 8-byte alignment, so four bytes of padding already sat at offset 4. `sizeof(TrackState<double>)` is 248 before and after, asserted in `tests/test_ion_transport.cu` section 3 rather than claimed. The buffer pays two bytes a slot out of 236. In a float build `Vec3<float>` aligns to 4 and the struct would grow; that build is not the default (docs/RISK.md N2) |
| a primary GenericIon | - | **-** | Refused by name in `G4RunManager::CheckSpecies`, ahead of the disposition test: a primary ion needs (Z, A) and there is no `/gun/ion` here. Every ion this transport steps arrives through `BufferEmitter::push_nucleus` |

**The second fix is the reference.** `ref/proton/proton_depth.cc` ran `G4EmStandardPhysics` and
nothing else, with a note explaining that comparing against QBBC "would measure that absence
rather than the stepper". That stopped being true when P8 and P8b wired `G4Decay`, `hadElastic`
and `CoulombScat`. It is QBBC on both sides now, with **only what the port lacks inactivated on
the Geant4 side** - the plan's own rule (section 4) - and the run prints
`/particle/process/dump` for the proton and for GenericIon so the configuration is recorded by
what ran rather than by what was intended.

What the dump reads, and it is the whole statement of the comparison:

```
G4ProcessManager: particle[proton]                G4ProcessManager: particle[GenericIon]
[0] Transportation      Active                    [0] Transportation      Active
[1] msc                 Active                    [1] msc                 Active
[2] hIoni               Active                    [2] ionIoni             Active
[3] hBrems              InActive                  [3] ionInelastic        InActive
[4] hPairProd           InActive                  [4] ionElastic          InActive
[5] CoulombScat         Active
[6] hadElastic          Active
[7] protonInelastic     InActive
```

**Two names the UI refuses, and both are V53's mechanism.** `/process/inactivate
neutronInelastic` and `/process/inactivate photonNuclear` both answer `illegal process (or
type) name`, because each is a sub-process inside a general process and is on no manager:
`G4HadProcesses::BuildNeutronInelasticAndCapture` hands the neutron's to
`G4NeutronGeneralProcess::SetInelasticProcess`, and
`G4EmExtraPhysics::ConstructGammaElectroNuclear` hands the gamma's to
`gproc->AddHadProcess(gnuc)` because `G4EmStandardPhysics::ConstructProcess` calls
`SetGeneralProcessActive(true)`. The neutron's comes off with `NeutronGeneralProc`, which is in
the list. The gamma's cannot come off at all without taking Compton, the photoelectric effect,
Rayleigh and conversion with it - processes this port HAS - so it is left on, and the arithmetic
that makes it inert is in the source: nothing in this configuration makes a photon above about
0.5 MeV, and `G4GammaNuclearXS` is a giant-resonance cross section starting near 10 MeV.

**And the phantom is 150 mm wide rather than 50.** `tools/compare_depth.ps1`'s first metric is
energy in against energy deposited, limit 1e-6, justified as "it should be exact on both sides -
the phantom is deeper than the range". Deeper is not wider: an elastic scatter off oxygen leaves
a 100 MeV proton nearly all of its energy at any angle, so one scattered near 90 degrees runs
its whole 77 mm range sideways and out of a 50 mm half-width box. Measured on the new reference,
100,000 events: **112 MeV of 10,000,000 left the phantom, 1.1e-5, eleven times the limit**. That
is energy Geant4 genuinely transported out of the box, so the phantom was widened to hold it
rather than the limit widened to excuse it - the same rule this package applied to the plateau.
At 150 mm the deposited total is 100.0000% of the beam energy on both sides.

**The numbers.** `tests/test_ion_transport.cu`, six sections:

* the scaling is EXACTLY the identity for the ten species that own a table and is not for the
  three that do not - `range_for/lookup` is 0.5722 for the deuteron, 0.4139 for the triton and
  **0.1032** for He3, whose `chargeSqRatio` is the dynamic 4.0007 at 50 MeV in water;
* a stopping track's total true path length against `range_for`: proton +0.152%, alpha +0.055%,
  deuteron +0.020%, triton -0.011%, and the energy balance closes to 0 in every case;
* `sizeof(TrackState<double>)` 248 with `ion_za` at offset 4 and `pos` at offset 8, and 308
  natural isotopes round-tripping through the 16-bit field;
* `ion_particle_def` against `ref/oracle/ion_definitions.csv` for eleven nuclides, mass bit for
  bit - and that file is also where the spin refusal is bounded: 2J is 0 for C12, O16, O18,
  Mg24, S32, Ca40, Fe56 and Pb208 and non-zero for Li7 (1.5), N14 (1) and P31 (0.5);
* a nucleus stepped to a stop in water - an O16 of 0.55 MeV has a range of **1.35 um**, dies on
  its first step and deposits everything where the scatter happened, which is where Geant4 puts
  it too;
* against `ref/oracle/ion_tables.csv` - 2684 rows, eleven nuclides in B1's four materials, 61
  energies each - **`q2_eff` worst 2.30e-15**, dE/dx worst 3.67e-2 (Ca40 in air at 121 MeV,
  which is GenericIon's own table in that band), range worst 1.97e-2.

`tests/test_step_hadron.cu` on the device against the host, unchanged at 900 steps and a worst
relative deviation of **3.197e-13**, now also comparing each secondary's NUCLIDE exactly: 16
elastic scatters produced 13 recoil tracks, **7 of them GenericIon**, drawn as (Z, A) = (8, 16)
and (1, 1) - oxygen recoils in water, which is exactly the population the gate was losing.

**THE GATE, before and after.** `proton_depth.exe 6000 100 out/port_depth.csv 0.7 0.5` against
the regenerated `ref/oracle/proton_depth.csv` (100,000 events), through
`tools/compare_depth.ps1` with its limits untouched:

| metric | before (P8b, `build_all.bat`) | after | limit |
|---|---|---|---|
| total, Geant4 | exact, but see the note above | **100.0000%** | 1e-6 |
| total, port | 99.9141% **FAIL** | **100.0000%** | 1e-6 |
| plateau, port/G4 per proton | 1.01322 (+1.322%) **FAIL** | **1.00158** (+0.158%) | 1.000% |
| R80 | G4 77.798, port 77.743, diff -0.054 mm | G4 **77.730**, port **77.742**, diff **+0.012** mm | 0.5 mm |
| 80-20 falloff width | G4 1.126, port 1.166, diff +0.040 mm | G4 **1.152**, port **1.166**, diff **+0.014** mm | 0.15 mm |

Both failures closed, and by different halves. The TOTAL is item 1: 929 GenericIon secondaries
carrying 515.136 MeV of 600,000 became tracks, and the banner
`*** 929 SECONDARIES OF SPECIES THIS PORT CANNOT TRANSPORT ***` is gone. The PLATEAU is item 2:
the reference's own plateau per proton rose by about 1.2% when `hadElastic` was switched on in
it, because a deflected proton takes a longer path through each slab and its recoils deposit
where they were made - and the same effect shortened Geant4's R80 by 0.068 mm, which is why R80
and the falloff width both improved as well without anything in the port changing them.

**AND THE SAME HARNESS NOW RUNS AN ELECTRON, which is P14d.** `ref/oracle/run.bat` has had the
line since P14c - 100,000 e- of 1 GeV into 4,000 mm of water in 20 mm slabs, 400 mm of
transverse half-width, the phantom sized for a shower rather than for a track - and the Geant4
half was produced then. The PORT half could not be quoted, because every lepton died in the
`G4_Galactic` world on its first step: docs/RISK.md V84, closed by **V96**.

| 100,000 e- of 1 GeV, 4 m of water | Geant4 11.1.1 | port | diff |
|---|--:|--:|--:|
| contained in the phantom | 97.6448% | **97.8498%** | +0.205 pp, **+0.21%** |
| entrance half, 0-2,390 mm, per electron | - | 0.99989 | **-0.011%** |
| R80, the distal 80% of the peak | 1,079.258 mm | **1,081.695 mm** | **+2.437 mm, +0.23%** |
| 80%-to-20% distal falloff | 1,152.746 mm | 1,172.994 mm | +20.248 mm, +1.76% |
| shower maximum, parabolic through three 20 mm bins | 683.6 mm = 1.90 X0 | 652.9 mm = 1.81 X0 | -30.7 mm |

The last row is the weakest number in the table and not the strongest: the curve is flat to 0.4%
over +/- 80 mm about its maximum, so a parabola through three bins at 100,000 events is fitting
noise, and bin for bin through that region the two agree to 0.2-0.6%. What is well determined is
the integral, the entrance half and R80, and they agree to about two parts in a thousand. The
port keeps slightly more of the shower, consistent with its 1.76%-longer distal tail. docs/RISK.md
V96 has the before-column - 0.0000% contained - and the five-seed gamma gate the fix was judged
on.

**The register and stack cost**, `transport_run.cu` with `-Xptxas -v`, measured on this branch
before and after. `run_step_hadron` goes from 11 instantiations to 13 - He3 and GenericIon - and
docs/RISK.md V55 is the entry about what happened the last time that translation unit grew:

| kernel | instantiations | registers | stack frame | spill st/ld | cmem[0] |
|---|---|---|---|---|---|
| `run_step_hadron` | 11 -> **13** | 255 -> 255 | 4512 -> **4576** B | 96/52 -> 100/52 | 1584 -> 1600 |
| `run_step_neutral` | 2 | 255 -> 255 | 3728 -> **3744** B | 92/52 -> 96/52 | 1608 -> 1624 |
| `run_step_lepton` | 2 | 255 -> 255 | 3056 -> **3040** B | 76/28 -> 80/28 | 1448 -> 1464 |
| `run_step_gamma` | 1 | 255 -> 255 | 2400 -> **2416** B | 404/868 -> 368/676 | 1448 -> 1464 |

+64 bytes on the hadron kernel's frame against the 16384-byte limit `Upload` sets, no change in
register count, and ptxas survived. The **+16 bytes of cmem[0] on every kernel is the `ion_za`
pointer** - two `TrackBuffer`s passed by value, eight bytes each - which is a check on the claim
that the field itself is free: the STRUCT did not grow (`sizeof(TrackState<double>)` is 248
before and after), and the only thing that did is the argument list. One of the thirteen hadron
entries spills 184/140 rather than 100/52; that is the instantiation carrying the ion branch.

Compile time is the thing to watch rather than the frame. The translation unit took about
ninety minutes on this machine for the thirteen-plus-four entry points, against the eight
minutes V55 records for eleven - with three Geant4 cmake builds running beside it, so it is not
a clean measurement, but the direction is the one V55 warns about and the next package to add a
kernel should measure it with V55's one-kernel reproducer before it adds one.

*(P8d re-measured it clean, on the same seventeen entry points and with nothing else running:
**22 minutes 13 seconds**. So the ninety was the three cmake builds and not the translation unit,
and V55's "wrong direction by an order of magnitude" is a factor of 2.8 rather than 11. Section
2.1.8 has the after figure.)*

#### 2.1.8 The neutron general process in the neutral stepper (P8d)

The last open item of Phase 2's first wiring, and the row that had been `0.0000 +/- 0.0000`
against `0.0000 +/- 0.0000` since P8.

| Geant4 class / function | QBBC | | Where |
|---|:--:|:--:|---|
| `G4NeutronGeneralProcess::PostStepGetPhysicalInteractionLength` and `CurrentCrossSection` - one interaction length off table 0 below 20 MeV and table 3 above it, with the zone test `energy <= fMiddleEnergy` | y | **T** | `step_neutral` in `physics/stepper.cuh`, through `had::NeutronGeneralXs::total`. One `log(ekin)` per step, handed to every lookup, as Geant4 computes one `fLogEnergy` per step and `ComputeGeneralLambda` and `GetProbability` both read it |
| `PostStepDoIt`'s sub-process choice - one uniform, `q <= GetProbability(1)` then `(2)` below the middle energy and `q <= GetProbability(4)` above it, with the **order swap** between the zones | y | **T** | same, through `::select`. One uniform, drawn where `G4double q = G4UniformRand()` is |
| the elastic sub-process: `G4HadronElasticProcess` with `G4NeutronElasticXS` and `G4ChipsElasticModel` | y | **T** | `hadronic/neutron_wiring.cuh`, `neutron_elastic_apply`. The neutron shares the MODEL with the proton and not the data set - `G4HadronElasticPhysics::ConstructProcess` line 147 builds `new G4HadronElasticProcess()` with a Chips model and hands it to `G4HadProcesses::BuildNeutronElastic`, which supplies `G4NeutronElasticXS` |
| `fXSType == fHadNoIntegral` for a neutral particle, and what follows from it | y | **V** | `G4HadronicProcess::BuildPhysicsTable` guards the whole integral-approach setup with `charge != 0.0`, so the neutron's elastic and capture sub-processes draw NO rejection uniform and do NOT recompute the cross section - the element is drawn from the partial sums the caller left. Both halves are in `neutron_elastic_apply`'s header |
| `fCurrentXSS->ComputeCrossSection` before delegating, **and only for a material with more than one element** | y | **T** | `step_neutral`. With one element `SampleZandA` takes element 0 and draws no uniform, so the call has no answer attached; with more, the partials are what the target draw reads |
| the capture sub-process: `G4NeutronCaptureProcess` with `G4NeutronCaptureXS` and `G4NeutronRadCapture` through `G4PhotonEvaporation::BreakUpChain` | y | **T** | `neutron_capture_apply`, over P7's `capture_final_state` and P3's cascade. The secondaries are emitted inside the `__noinline__` function so that its two 16-entry arrays never cross into the kernel's frame |
| the inelastic sub-process | n | **-** | **Refused by name**, `had::HadronicRefusal::kNeutronInelastic`, with the kinetic energy it costs. It is SELECTABLE - `BuildPhysicsTable` sums all three unconditionally, so leaving the term out of table 0 would be a different cross section - and above 1 MeV it is a large share: measured, **18.2% of interactions in water at 10 MeV, 38.6% in air, 25.5% in bone, 50.7% in lead**. The neutron is killed with its energy deposited locally, which is the conservative disposal and is NOT what Geant4 does with it |
| `HadronicStage::kStage1` - the neutron with `EnableNeutronGeneralProcess` FALSE, i.e. `hadElastic` and `nCapture` as separate processes on their own data stores with their own interaction lengths | y | **T** | `step_neutral`. A DIFFERENT COMPETITION and not the general table minus a term: docs/RISK.md V60 for the finding that made the configuration reachable at all, and `ref/b1neutron/` for the reference binary |
| the five tables and the two per-process data sets on the device | - | **T** | `host/neutron_upload.cuh`. **7.43 MB** for a five-material scene: `ParticleXsTable` copied with its `std::vector`s empty and its three pointers repointed, which is the arrangement `data/particlexs_data.cuh`'s own comment describes |
| P3's level scheme uploaded whether a neutron arrives or not | y | **T** | `SetNuclearLevelData` is ON by default now (9.52 MB), which is Geant4's answer in `G4ExcitationHandler::SetParameters` and what section 2.1.6's row said would happen "the day the neutron is wired" |

**THE NUMBERS.** `tests/test_neutron_general.cu`, five sections, on a scene of water, air,
bone-compact and G4_Pb - lead because B1's four materials stop at Z = 20 and have no large
capture cross section:

* **the upload, exactly.** 61,230 EXACT comparisons and 9,420 tolerant ones over 4,710
  (material, energy) probes - every node and every bin midpoint of both grids for five
  materials - read through the transport's own accessors (`::total`, `::select` at nine q
  values, `neutron_sub_xs_per_volume`). **0 disagreements.** Worst deviation 1.949e-16 on the
  five combined tables and 1.896e-15 on the two per-process sums. The split is not cosmetic: at
  a NODE the value is the uploaded double and is compared exactly, between nodes the
  interpolation is a multiply-add that nvcc contracts to an FMA and MSVC does not, and the
  per-process sums accumulate `total += n_atoms[i]*sigma_i` over up to nine elements.
* **the sub-process frequencies against the partials, both configurations.** 200,000 draws of
  `step_neutral` itself per cell, 40 cells (2 stages x 4 materials x thermal/1 keV/100 keV/1
  MeV/10 MeV), in a 1e9 mm box so geometry never wins: **worst z = 2.57** against a 5-sigma
  gate. The two stages are predicted by two different formulas - the general table's cumulative
  partials in `kFinal`, `sigma_el/(sigma_el+sigma_cap)` in `kStage1` - so passing both is a
  statement that the two competitions are the two competitions.
* **`step_neutral` host against device**, 1600 tracks x three configurations, every field
  including each secondary's species, nuclide and energy: **0 disagreements.** The elastic energy
  balance closes to **2.810e-11 MeV** absolute, which is two ulps of a lead target's mass - the
  recoil energy is `lv.e() - mass2` with both terms of order 1.9e5 MeV, so the limit on that
  field is absolute and not relative.
* **the capture cascade's length against the capacity it is given.** 35,590 captures over every
  element of the four materials at five energies: **3.10 secondaries per capture** (P7's oracle
  measured 3.05 over 600 calls) and a longest cascade of **10** against a capacity of 16. The
  capacity costs 353 bytes of `run_step_neutral`'s frame per unit, so it is a number that has to
  be justified by the physics rather than chosen for comfort.
* **and what a null level scheme does**, which is the claim `Upload`'s refusal rests on: 2000
  thermal captures in lead give 26,485 secondaries carrying 14,609.9069 MeV with a null table
  against 9,021 carrying 13,694.7198 MeV with the real one. Three times the multiplicity, not
  none. docs/RISK.md V60.

**THE STAGE-1 LIKE-FOR-LIKE, AND THE ROW IS NOT A ZERO ANY MORE.** 500,000 neutrons of 100 MeV
into B1, the Geant4 side through `ref/b1neutron/` - Geant4's own example B1 with
`SetEnableNeutronGeneralProcess(false)` in a main of ours, which docs/RISK.md V60 is the entry
about - and `neutronInelastic` inactivated by name:

| | port | G4 stage 1 | diff | sigma | G4 no elastic |
|---|---|---|---|---|---|
| neutron | 54.4664 +/- 0.5408 | 54.4573 +/- 0.5361 | **+0.02%** | **0.01** | **0.0000 +/- 0.0000** |

The third column is the whole of the physics: with `hadElastic` off and `nCapture` left ACTIVE,
500,000 neutrons deposit `0 picoGy  rms = 0 picoGy`. A neutron itself has no ionisation process,
so every gray in the first column is carried by something elastic scattering made - the recoil
protons of hydrogen and the recoil nuclei of oxygen, carbon, nitrogen and calcium, which are
tracks because of 2.1.7 - and `nCapture` contributes nothing at 100 MeV on its own because
without elastic scattering the neutron never slows to where its cross section matters. So the
row tests `G4NeutronElasticXS`, `G4ChipsElasticModel` and the recoil transport, and nothing else.
`ref/b1hadron/stage1_README.md` has the process dump and what the run reports beside the dose
(7.42 MB of tables, 4453 neutrons killed past 10 us discarding 3.7e-5 MeV, and an empty refusal
ledger).

Plus, in section 3, **determinism across the launch geometry**: the same 4,800 tracks at 64 and
at 256 threads a block, tolerance ZERO, every field including each secondary's species, nuclide,
energy and direction - **0 disagreements**. `step_neutral` had no such check before, because
until P8d it emitted nothing.

**THE REGISTER AND STACK COST**, `transport_run.cu` with `-Xptxas -v`, measured on this branch
before and after. The neutral kernel's two instantiations are listed separately for the first
time, and the reason is the finding in the row below them:

| kernel | inst. | registers | stack frame | spill st/ld | cmem[0] |
|---|---|---|---|---|---|
| `run_step_hadron` | 13 | 255 -> 255 | 4576 -> **4592** B | 100/52, unchanged | 1600 -> 1616 |
| `run_step_neutral<kNeutron>` | 1 | 255 -> 255 | 3744 -> **7264** B | 96/52 -> 248/540 | 1624 -> 1640 |
| `run_step_neutral<kPiZero>` | 1 | 255 -> 255 | 3744 -> **3152** B | 96/52 -> 524/848 | 1624 -> 1640 |
| `run_step_lepton` | 2 | 255 -> 255 | 3040 B, unchanged | 80/28, unchanged | 1464, unchanged |
| `run_step_gamma` | 1 | 255 -> 255 | 2416 B, unchanged | 368/676, unchanged | 1464, unchanged |

+3520 bytes on the neutron's frame against the 16384-byte limit `Upload` sets, no change in
register count, ptxas survived, and **the PI0's frame went DOWN by 592 bytes.** That last one is
not ptxas reallocating: the species is a TEMPLATE parameter of `run_step_neutral`, so
`type == ParticleType::kNeutron` is a compile-time constant inside it and every branch this
package added folds away for the pi0 - the tables, the two cross sections, the sub-process
choice, both final states. A pi0 pays nothing for the neutron's physics, which is the property
that made one kernel for two neutral species affordable in the first place and is measured here
rather than assumed. The +16 bytes of cmem[0] on the two kernels that take a `HadronicWiring` by
value is `had::NeutronSubTables`' two pointers; the lepton and gamma kernels do not take one and
did not move.

**THE COMPILE TIME**, which docs/RISK.md V55 says is the thing to watch and which P8c could only
measure dirty. Same machine, same seventeen entry points, nothing else running, `nvcc -O2
-arch=sm_86 -Xptxas -v -c`:

| | `transport_run.cu` |
|---|---|
| main at 2a6b379 | **22 min 13 s** |
| with the neutron general process wired | **24 min 44 s** |

**+2 min 31 s, 11%**, for two sub-processes, a 2.5-decade cross-section evaluation and P3's
whole de-excitation chain arriving in the translation unit. That is what the `__noinline__` on
`neutron_sub_xs_per_volume`, `neutron_elastic_apply` and `neutron_capture_apply` bought, and it
is the number to compare the next package against. It also closes V55's own open question: P8c
recorded "about ninety minutes" for this file with three Geant4 cmake builds beside it, and the
clean figure for the same source is 22 - so the ninety was the contention, and V55's "wrong
direction by an order of magnitude" is a factor of 2.8 against the eight minutes it records for
eleven entry points.

**And the iteration was all on the reproducer**, which is V55's own advice taken: the one-kernel
translation unit of `run_step_neutral<double, kNeutron>` compiles in **75 seconds**, and it is
what found the capture capacity's 353 bytes a slot and what showed 13120 bytes at capacity 32.
The 25 minutes was paid twice - once for the baseline, once at the end - and not once per
iteration.

#### 2.1.9 One translation unit per kernel, and the two switches it freed (P8e)

Nothing in this section is new physics. It is the compiler wall that `2.1.7`'s ion transport and
section `1.1`'s Urban model had been waiting behind, and the numbers that came out once it was
gone. docs/RISK.md **V65**.

**What the engine is now.** `src/host/transport_run.cu` instantiated `TransportEngine<double,
StepTap<double>>`, and the eighteen `<<<>>>` launches inside `BeamOn` implicitly instantiated
eighteen stepping kernels into that one file: **25 min 02 s** of nvcc, and `ptxas died with
status 0xC0000005` as soon as P14b's Urban branch was live in it (V63). The kernels are declared
`extern template` under their definitions in `transport_run_impl.cuh` and defined one per
translation unit beside it - sixteen of them - which `build_engine.bat` compiles six at a time
and archives into `out/transport_run.lib`. `transport_run.cu` keeps the engine's host code and
the three utility kernels; its object went from 21 device functions to 3.

**The granularity is a measurement, and it corrects V55 and V63.** One unit per kernel FAMILY put
pi+, pi-, K+ and K- together, and ptxas died on that unit - with the Urban flag still off, on
code that compiles as four of the eighteen kernels in the single unit. Alone on an idle machine
it died in 99.7 s; one `run_step_hadron` on its own compiles in 90 s. So a translation unit does
not get safer by being made smaller, and "the translation unit is the variable" is not the whole
story: its shape is, and not monotonically in its size.

| | wall |
|---|---|
| one unit, eighteen kernels | **25 min 02 s** |
| eight family units at once (the meson unit died) | 8 min 54 s |
| seventeen units, six at a time | **6 min 31 s** |

**Registers: 255 in every stepping kernel, before and after.** The stack frames and the spill
counts move in both directions, because ptxas allocates per module - `run_step_gamma` grows 624
bytes of frame and drops from 368/676 to 52/20 bytes of spill without one character of its code
changing, which is exactly the movement V55 recorded in reverse when the unit grew. V65 has the
table. The check that none of it reaches the answer is B1's 2,000,000-event gate, which reads
**425.847 pGy +/- 0.867682 against Geant4's 427.385 +/- 0.87, 1.25138 sigma** - main's recorded
number to every digit it prints. Four more seeds: 426.195, 426.917, 427.489, 427.288 pGy.

**`kUrbanIonMscWired` is true.** Section 1.1's `V` for the ion branch becomes `T`, and the
substitution recorded since the alpha was first transported - every ion scattered by WentzelVI
because this port's Urban had only the electron's stepping half - is over. All five Urban units
compiled first time and the whole engine took 376.4 s with the flag true against 391.1 s with it
false, because Urban's stepping half is cheaper than the WentzelVI it replaces: alpha 3920 ->
3760 bytes of frame and 252/344 -> 148/208 of spill, GenericIon 4000 -> 3760 and 360/432 ->
228/296, with the proton's kernel identical in every column. Example B1's stage-1 alpha at 840
MeV over 500,000 events moves from **12,313.6 +/- 13.0** to **12,316.4 +/- 13.0 nGy** against
Geant4's **12,336.9 +/- 13.0** - from 1.27 to **1.12 sigma**, +0.023%, which is a fifth of a
sigma toward Geant4 and not resolvable. The step counts are where the model is visible: alpha
16.5 -> 16.0 steps, deuteron 14.5 -> 14.0, He3 3.5 -> 3.0, O16 1.6 -> 1.0. docs/RISK.md V66.

**The electron's `extremesmallstep` branch is on too**, which is V62's named gap and README's
open question 3. `SampleCosineTheta`'s sub-case for a step below `tsmall = min(tlimitmin,
lambdalimit)` has been in `em::urban_sample_cos_theta` since P14b generalised it; what the lepton
path lacked was the THRESHOLD, which it passed as zero and now takes from the tlimitmin the TRACK
carries - `urban_step_limit` freezes it at the first step and after each boundary, exactly as
Geant4 holds its member, and one recomputed at the sampling site would be a different number
wherever the branch fires. Decided by the gate and not by preference: five seeds of B1's
2,000,000-event run with one `constexpr` different move the dose by at most **0.0023 pGy**, and
by **-0.00036 +/- 0.00053 pGy** on the mean of the five, against a per-run uncertainty of 0.87 -
so 1.25138 sigma from Geant4 becomes **1.25195**, the shipped gate reads **425.847 pGy +/-
0.86768**, and the stage-1 alpha row is **12,316.4 +/- 12.9790 nGy** to every printed digit
either way. The kernel costs nothing for it: 3024 bytes of stack frame, 84/32 spill, 255
registers and cmem[0] 1464 in both instantiations, which is V65's flag-off column to the byte.
docs/RISK.md V66.

#### 2.1.10 binary_cascade - the nucleus model, the fields, the propagator and both entry points (P9)

**P9b has started on the cascade.** The cross sections and the angular distributions its channels
sample from are in and bitwise; `Propagate`, the collision channels above them, `G4Scatterer` and
the resonance tables are not, and the two refusals below still stand. The paragraph P9 wrote is
kept as it was written, because what it says about the ion arm above 50 MeV/nucleon is still true.

**The number to read first: `G4BinaryCascade`'s cascade proper is NOT here.** Everything below is
the machinery the two binary models are built on, plus the two branches of them that reach a
compound nucleus without a cascade. What is missing is `Propagate` and the whole `im_r_matrix`
collision tree under it - `G4Scatterer`, `G4CollisionManager`, the `G4Collision*` channels, the
`G4X*` cross sections, the angular distributions and the resonance widths - and with it the ion
arm above 50 MeV/nucleon, which is where a galactic cosmic ray actually is. Section 2.2's row for
`G4HadronInelasticQBBC` still says "the cascades and the strings: none", and it is still right.

What is here is refused BY NAME at the point it would have been needed - `BicRefusal::cascade`,
`BlirRefusal::cascade`, `RkPropagation::is_refused_field_species` - and never approximated.

The nucleus model under `bic/nucleus/` is **shared with P11 (FTF)**. `nucleus_model.cuh` is the
contract header: include that one, not the four it pulls in. It states what `Init` guarantees (A
nucleons, the hard-core exclusion, zero total three-momentum) and the two things it does not -
every nucleon is off its mass shell downwards, which selects the QGS arm of
`preco::propagate_residual` (docs/RISK.md V50), and the total energy is `Nucleus3D::mass()` and
not `G4NucleiProperties::GetNuclearMass` nor `G4IonTable::GetIonMass`, three answers 11.1.1 uses
in three places.

| Geant4 class | QBBC | | Where |
|---|:--:|:--:|---|
| **G4Fancy3DNucleus** (`Init`, `ChooseNucleons`, `ChoosePositions` incl. the A=12 alpha-cluster branch, `ChooseFermiMomenta`, `ReduceSum`, `CenterNucleons`, `GetNuclearRadius`, `GetOuterRadius`, `GetMass`, `CoulombBarrier`, `StartLoop`/`GetNextNucleon`, both `DoLorentzBoost`/`DoLorentzContraction` overloads, both sorts) | y | **V** | `bic/nucleus/fancy_3d_nucleus.cuh`. Radii, mass, barrier and binding energy exact on 12 nuclides; 5 replayed configurations checked for the hard core, the proton count, the per-nucleon binding energy, the off-shell invariant and the outer radius, all exact; the SAMPLING checked at 20,000 nuclei x 5 nuclides against radial and momentum histograms and five moments, worst 3.20 sigma of 400 bins and 2.15 of 25 moments. C12's cluster spread is a variance used as a sigma and its stream depends on a CLHEP thread-local latch: docs/RISK.md V68. `ReduceSum`'s verdict is returned rather than discarded: V74 |
| G4NuclearFermiDensity, G4NuclearShellModelDensity, G4VNuclearDensity | y | **V** | `bic/nucleus/nuclear_density.cuh`; 360 points of rho0, `GetRelativeDensity`, `GetDensity`, `GetDeriv` and 120 of `GetRadius` including both ends of its guard, all at 3.2e-16 against a 1e-15 bucket. The A < 17 dispatch is asserted per nuclide, not assumed. `theRsquare`'s association order is load-bearing: V67 |
| G4FermiMomentum | y | **V** | `bic/nucleus/fermi_momentum.cuh`; 130 points, 3.8e-15. Not 1e-15, and the reason is `src/data/g4pow.hh`'s A13, not this file: V73 |
| G4Nucleon | y | **V** | `bic/nucleus/nucleon.cuh`. The two `Boost` overloads boost in OPPOSITE directions - the `G4LorentzVector` one is CERNLIB's U101 form and transforms INTO the argument's rest frame - and `G4Fancy3DNucleus::DoLorentzBoost` forwards each to the matching one |
| G4KineticTrack (the two constructors BIC uses, both momentum pairs, `GetActualMass`, the four `Update*Momentum`, `CascadeState`) | y | **P** | `bic/kinetic_track.cuh`. `theFermi3Momentum` is loaded from the nucleon and discarded two lines later, in every event: V69. **Refused by name:** the resonance-width machinery (`G4SampleResonance`, `G4Integrator`, `IntegrateCMMomentum`) and the K0 -> K0S/K0L coin flip, both of which belong with `G4BCDecay` |
| G4ProtonField, G4NeutronField, G4PionPlus/Minus/ZeroField, G4VNuclearField | y | **P** | `bic/nuclear_field.cuh`; 750 field points at 2.4e-14 and 750 barriers exact, on the replayed nuclei. The nucleon fields are a 0.3 fm TABLE and not a formula, and its tail returns a Fermi momentum where a field belongs; the pion fields build a nucleus mass by ADDING the binding energy. Both reproduced, both pinned by the extractor: V70. **Refused by name:** G4AntiProtonField, the three kaon and the three sigma fields - their optical coefficients are recorded as named constants and no channel this package reaches produces one |
| G4RKPropagation (`Init`, `Transport`, `FieldTransport`, `FreeTransport`, both `GetSphereIntersectionTimes`), G4KM_NucleonEqRhs, G4KM_OpticalEqRhs, and the field machinery they drive - G4ClassicalRK4, G4MagErrorStepper, G4MagInt_Driver (`AccurateAdvance`, `OneGoodStep`, `QuickAdvance`, `ComputeNewStepSize`) | y | **P** | `bic/rk_propagation.cuh`; 11 initial states x 5 nuclei x 12 steps = 7,260 position and momentum comparisons at 2.8e-13, 660 cascade states exact, and the per-step momentum transfer at 1.3e-11 MeV absolute. The exit test's short circuit is load-bearing - hoisting the intersection call out of the `||` moves the C12 neutron 2.807 relative in x. The position-error tolerance compares a time to a length and is never binding: V71. `QuickAdvance` is reached on 17 of the 55 trajectories. **Refused by name:** `G4RKFieldIntegrator`, `G4Absorber`, and the spin terms, which are dead for `nvar = 6` |
| **G4BinaryCascade::ApplyYourself**, the `theBCminP` branch | y | **P** | `bic/binary_cascade.cuh`, entry point `bic::apply_yourself()`. A nucleon below 45 MeV never enters the cascade: the whole reaction is `G4PreCompoundModel::ApplyYourself`, which is P6's. 18 cases of {p, n} on {C, O, Al, Fe, Pb} at 5-46 MeV x 5,000 events: compound (Z, A) exact, energy balance 2e-10 MeV/event, species yields worst 3.44 sigma. The 44/46 MeV pair straddles the threshold and the REFUSAL at 46 is asserted against what Geant4 did instead. **Refused by name:** the cascade proper, every pion at every energy (the species test is an `&&`), any projectile that is not a nucleon or a charged pion, and the per-secondary creator model id, which P3's product does not carry |
| **G4BinaryLightIonReaction::ApplyYourself**, the fusion arm | y | **P** | `bic/light_ion_reaction.cuh`, entry point `bic::blir_apply_yourself()`, with `SetLighterAsProjectile`, `FuseNucleiAndPrompound` and `EnergyAndMomentumCorrector`. 20 cases of {d, alpha, C12} on {C, O, Al, Fe, Pb, H} at 1-45 MeV/nucleon x 5,000 events: the fusion gate's verdict exact in all 20 including the one that returns the primary ALIVE (alpha on H at 1 MeV/nucleon - Li5 is unbound), compound (Z, A) exact in 100,000 events, energy balance 2e-10 MeV/event, species yields 3.01 sigma and kinetic energies 3.19. The rotate-to-lab block is the identity and is not carried: V75. **Refused by name:** `Interact` and everything under it, and with it every ion at or above 50 MeV/nucleon; `GetProjectileExcitation`, `SortResult` and `DeExciteSpectatorNucleus`, whose arithmetic is recorded in comments and runs nowhere |
| The `im_r_matrix` collision tree: G4Scatterer, G4CollisionManager, every `G4Collision*` and `G4X*`, G4AngularDistribution and its tables, G4ResonanceNames, G4ResonanceWidth, G4PartialWidthTable, G4BaryonWidth, G4BaryonPartialWidth | y | **P** | **partly, P9b - see the rows below.** The cross sections and the angular distributions are in; the collision channels, the resonance tables, `G4Scatterer` and `G4CollisionManager` are not |
| **G4XNNTotal, G4XnpTotal, G4XNNElastic, G4XnpElastic** and every arm under them - G4XNNTotalLowE, G4XnpTotalLowE, G4XNNElasticLowE, G4XnpElasticLowE, G4XPDGTotal, G4XPDGElastic - with G4CrossSectionPatch, G4CrossSectionComposite, G4LowEXsection, G4VCrossSectionSource::FindKeyParticle and the G4PhysicsLogVector shape the four tables are poured into (P9b) | y | **V** | `bic/im_r/xsec_nn.cuh`. 6,280 points: ten sources x four nucleon pairs x 157 values of sqrt(s), including both sides of both patch boundaries at one part in 1e9. **Bitwise**, all of them. The np grid is stretched by 1% and the pp grid is not: docs/RISK.md V91. Four tables carry a zero 102nd node and one energy grid is short by one: V92. **Refused by name:** the kaon, antinucleon and gamma rows of the two PDG fits, which no channel `G4Scatterer` registers can reach; `G4XpnTotal` and `G4XpnElastic`, which nothing in 11.1.1 includes; `G4CollisionPN`, which `G4Scatterer.cc` includes and never registers |
| **G4AngularDistributionNP, G4AngularDistributionPP** and their float tables, **G4AngularDistribution** (the one-boson-exchange formula, `DifferentialCrossSection`, `Cross`, `CosTheta` in both symmetric and asymmetric forms) and `Phi` (P9b) | y | **V** | `bic/im_r/angular.cuh`, tables by `tools/extract_bic_imr.pl` into `im_r/imr_tables.hh`. **Bitwise** on 43,244 points, driven by a prescribed uniform on both sides: 31,044 of them sweep 199 values of the sample across every one of the 39 and 40 tabulated energies and the midpoints between them, so the bisection walks the whole cumulative. The narrower eight-phase set was MEASURED not to be enough - one table entry moved by 1e-5 changed none of its 1,600 angles and 75 of the sweep's. The tables are `G4float` and that is load-bearing: one float ulp in one of 7,020 entries moves a cosine by 7.5e-5. **Refused by name:** `pcm`, `dsigmax` and `sigtot`, appended to both data files and read by nothing; `NENERGYC`, a leftover of the pre-2010 22-energy shape |
| **G4CollisionNN** (`CrossSection` with its on-shell recast, the eight-component list and its order), G4GeneralNNCollision::IsInCharge, G4CollisionComposite (`CrossSection`, `IsInCharge`, the `FinalState` selection), G4VCollision::CrossSection, **G4CollisionNNElastic**, **G4CollisionnpElastic**, **G4VElasticCollision::FinalState**, and CLHEP::HepLorentzRotation (`set`, `rotateY`, `rotateZ`, `inverse`, both multiplications) (P9b) | y | **P** | `bic/im_r/collision_nn.cuh` and `im_r/lorentz_rotation.cuh`. 3,528 `IsInCharge` verdicts and 2,352 cross sections exact; 73,792 outgoing four-momentum components over 4 pairs x 2 mass shells x 3 kinematic tilts x 157 energies x 8 phases, worst 2.3e-16 with both tracks along +z, 2.5e-16 with the projectile tilted, 2.4e-15 when the target carries its own momentum - eleven ulps through the boost matrix, its inverse, two rotations and two 4x4 products. The tilt is not decoration: with everything along +z both rotations are the identity (docs/RISK.md V75) and composing with `toZ` instead of `toZ.inverse()` was MEASURED to change none of 24,576 components. **Refused by name:** `G4CollisionComposite::FinalState`'s selection over `G4CollisionNN`, because six of its eight partial cross sections are the resonance channels this package does not have - `CollisionRefusal::resonance_channels`, at the point the partial would have been needed |
| **G4Scatterer** (`GetTimeToInteraction`, `GetCrossSection`, `FindCollision`), **G4CollisionManager** (`AddCollision`, `RemoveCollision`, `RemoveTracksCollisions`, `GetNextCollision`, `ClearAndDestroy`, `Entries`) and **G4CollisionInitialState** (P9b) | y | **P** | `bic/im_r/scatterer.cuh`. 8,448 verdicts and 8,448 cross sections exact, 1,954 interaction times bitwise, over 4 pairs x 2 momentum directions x 11 energies x 24 impact parameters x 4 longitudinal separations - the impact parameters straddling all three distance thresholds (2.5231, 3.9894 and 4.7683 fm) from both sides, because five of the six exits are decided by a distance. The manager's ordering replayed over a prescribed add/remove sequence with two exact ties, compared by WHICH collision won and not by its time. The 200 mb charged cap suppresses 128 of 1,056 forward pp configurations below 15 MeV: docs/RISK.md V93, which also shows that two of the four gates provably cannot change a verdict. **Refused by name:** `Scatter` and `GetFinalState` for a nucleon pair, which need the resonance partials; every meson-baryon pair, because `G4CollisionMesonBaryon` is not ported - and that is why this file cannot serve a pion, which is what `theBCminP`'s `&&` lets into the cascade at every energy |
| **The six `G4X*Table` resonance-production cross sections** (G4XNDeltaTable, G4XNDeltastarTable, G4XNNstarTable, G4XDeltaDeltaTable, G4XDeltaDeltastarTable, G4XDeltaNstarTable, through the `G4PhysicsFreeVector` each `CrossSectionTable()` builds) and **G4DetailedBalancePhaseSpaceIntegral** (P9b) | y | **V** | `bic/im_r/resonance_tables.cuh`; 50 columns x 121 and 25 x 120 extracted by `tools/extract_bic_imr.pl`. 12,100 cross sections and 9,000 phase-space integrals, **bitwise**, on every tabulated energy and the midpoint between each pair. Five of the six tables halve their cross section and `G4XNNstarTable` does not - docs/RISK.md V94, found because the port applied the half uniformly and the first `nnstar` point came back at exactly 0.5 relative. Two NNstar columns are short by eight entries and `delta(1940)` has no column at all; both pinned by the extractor. The grid's entries 1 and 2 are both 2.014 GeV, a zero-width interval, and it is unobservable because every column is still zero there. **Refused by name:** `G4XResonance::CrossSection` itself and `G4VXResonance`'s isospin, degeneracy and detailed-balance factors, which need `G4Clebsch` and the quantum numbers of thirty resonances - the TABLE half of that cross section is here and checked, the two factors are not |
| **G4Clebsch** (`TriangleCoeff`, `ClebschGordanCoeff`, `ClebschGordan`, `Weight`) and `G4Pow::logfactorial` (P9c) | y | **P** | `bic/im_r/clebsch.cuh`. 39,291 points **bitwise** - the whole (2J1, 2M1, 2J2, 2M2, 2J) box with the M values running two units past their J, so the two guards that reject them are exercised. `ClebschGordan` returns the SQUARE of the coefficient, which is the class's own convention and not a port error. The three range guards inside `ClebschGordanCoeff` are unreachable and the test asserts the count is zero; removing the M guard makes it 900. **Refused by name:** `GenerateIso3`'s sampling branch, which reads 384 uninitialised array entries, cannot select its last row or column, and returns an empty vector its only caller indexes at [0] and [1] - docs/RISK.md V106. Its three answerable early returns are kept. `Wigner3J` and `NormalizedClebschGordan`, called by nothing |
| **G4XAqmTotal, G4XAqmElastic, G4XMesonBaryonElastic**, **G4CollisionMesonBaryonElastic** (`IsInCharge` by parton count, `CrossSection`, `FinalState` with `G4AngularDistribution(false)`) and `G4CollisionMesonBaryon`'s component order (P9c) | y | **P** | `bic/im_r/xsec_meson.cuh` and `im_r/collision_meson.cuh`. 1,050 in-charge verdicts, 2,100 AQM cross sections and 900 meson-baryon elastic cross sections exact; 57,600 outgoing four-momentum components, worst 2.2e-15 with a moving target. The first pion channel in this package - and it is IDENTICALLY ZERO for a pion on a nucleon over the whole of QBBC's BIC window, because the PDG pi+p elastic fit starts at pLab = 2 GeV and a pion reaches that only at 1870 MeV of kinetic energy: docs/RISK.md V107. The only non-zero rows are a pion on a resonance. This is also the only place the one-boson-exchange formula's ASYMMETRIC branch runs. **Refused by name:** `G4CollisionMesonBaryonToResonance` and its 25 `G4ConcreteMesonBaryonToResonance` channels, and with them the composite's buffered total - returning the elastic partial alone would hand `G4Scatterer` an interaction radius several times too small where the Delta dominates. The kaon and hyperon channels are not refused: they are commented out in 11.1.1 |
| **G4XResonance::CrossSection** and **G4VXResonance::IsospinCorrection** (P9c) | y | **P** | `bic/im_r/resonance_tables.cuh`. 4,950 cross sections **bitwise** through the six concrete channel families, built in the dump exactly as the six `G4CollisionNNTo*` constructors build them, and 144 isospin and IsShortLived flags checked species by species against Geant4's own definitions. The function is shorter than it looks: both entrance tracks are always nucleons, so `DetailedBalance`, `DegeneracyFactor` and `GenerateIso3` are all unreachable - docs/RISK.md V108 - and the correction reduces to `weight/pWeight`. **Refused by name:** `pWeight == 0`, which is a G4HadronicException there; the outgoing spins and masses, which nothing reads |
| `G4Scatterer::Scatter`, `GetCollisions`, `GetFinalState`, the 306 `G4Concrete*` channel instantiations and the six `G4CollisionNNTo*` composites over them, G4CollisionComposite::BufferCrossSection, G4VScatteringCollision, G4XAnnihilationChannel, G4CollisionMesonBaryonToResonance and its two components, the six `G4CollisionNNTo*` channels and the `G4Concrete*` tree under them, G4VScatteringCollision, G4ResonanceNames, G4BaryonWidth, G4BaryonPartialWidth, G4PartialWidthTable | y | **-** | not ported |
| G4BCDecay, G4BCLateParticle, G4BCAction, G4RKFieldIntegrator, G4Absorber, G4MesonAbsorption | y | **-** | not ported; they are reached only from `Propagate` |

Constants no run can be asked for - `theBCminP`, the four `theCutOnP` assignments and the mass
thresholds they are compared against, `theCutOnPAbsorb`, the ten optical coefficients, the field
table's 0.3 fm step, the driver's safety factor and step budget, the C12 cluster geometry - are
checked by `tools/extract_bic_constants.pl`, which asserts the Geant4 SOURCE still says exactly
them. 93 checks. A test that compared the port's copy against a literal in the test would be
comparing a copy with itself, which is docs/RISK.md V52; this is V41's form instead.

Tests: `test_bic_nucleus.cu` (8.6 s), `test_bic_apply.cu` (11 s) and `test_bic_imr.cu` (P9b, 0.5 s;
315,185 comparisons; every cross section, every angle and every interaction time
bitwise, the elastic final states to 2.4e-15 at worst). Device probes, never
launched, `-arch=sm_86`: `bic_nucleus_probe` 82 registers / 168 bytes stack, `bic_rk_probe` 156 /
720, `bic_apply_probe` 255 / 10,080, `bic_blir_probe` 255 / 10,112, `bic_imr_probe` 122 / 48 with
no spills. The two at 255 are P6's `preco::deexcite` inlined whole - it alone reports 10,812 bytes
of spill stores - and are the first time the full de-excitation chain has been compiled for a
device. The cross sections and the angular distributions cost almost nothing beside it: the whole
im_r_matrix arithmetic in one kernel is 48 bytes of frame, because the tables are `static const`
arrays in constant/global memory and nothing is copied into a local.

One finding here belongs to P3 and is filed where P9's oracle found it: `G4PhotonEvaporation`
creates one electron rest mass out of nothing per conversion electron, 511 keV, because the
atomic binding energy that should pay for it is a local initialised to zero. docs/RISK.md V76.

#### 2.1.11 FTFP: the tuned parameters and the Lund string fragmentation (P11, first half)

The bottom half of QBBC's FTFP arm: `G4FTFParameters`, which is a large table of tuned constants
plus the hadron-nucleon cross sections every FTF interaction samples its impact parameter from;
the whole of `G4LundStringFragmentation` - one excited string in, a list of hadrons out; and
`G4ExcitedStringDecay` over it, which fragments a LIST of strings, redraws every short-lived
product's mass from a Breit-Wigner and corrects the energy-momentum balance that redrawing
breaks. The entry point of the package, `ftf::apply_yourself`, does **not** exist yet:
`G4FTFModel`, `G4FTFParticipants`, `G4DiffractiveExcitation` and `G4TheoFSGenerator` are the top
half and are not written, so nothing in the port yet PRODUCES a string for this code to
fragment. What is here is exact against Geant4 11.1.1 and has no caller.

**Why the exact half is exact.** The same technique as 2.1.4's: `ref/dump/dump_ftf.cc` installs
an eight-value uniform cycle as CLHEP's engine, so every sampler and the whole fragmentation
chain become deterministic functions of (inputs, phase), and every row carries **the number of
deviates the call consumed** as well as its answer. That count is half the comparison - a
transcription can produce the right hadron from the wrong number of random numbers, and then
every later hadron in the event is wrong. It is how the K0 substitution in `G4KineticTrack`'s
constructor was found: a `kaon0` becomes a K0S or a K0L on a coin toss, that toss spends a
deviate, and a port that substitutes the species without spending it shifts the entire remaining
stream by one.

| Geant4 class | | Where |
|---|:--:|---|
| **G4FTFParameters** (constructor for all five projectile classes, every getter, `ProcParams[5][7]`, `GetProcProb`, the profile functions, the nuclear-destruction parameters) | **V** | `ftf/ftf_parameters.cuh`. `InitForInteraction` draws no random number, so the oracle has no phase axis: 1,716 rows x 66 columns (22 projectiles x 6 targets x 13 momenta), 21,780 `GetProcProb` points over a rapidity grid that crosses every `Ymin` including the -100 and 1000 sentinels, 8,748 profile-function points. Worst 4.4e-16. **Refused by name:** hyperon projectiles (Lambda, Sigma+-, Xi-), because `G4HadronNucleonXsc::HyperonNucleonXscNS` is `P` in 2.1.1 - 4,920 oracle rows tallied as refusals. An ANTI-hyperon is not refused: Geant4 overwrites both cross sections in the Arkhipov block, so the refused branch never reaches the answer, and the port defers the refusal until after that block. docs/RISK.md V86 |
| **G4VLongitudinalStringDecay** (`SetMinMasses` and its six tables, `SampleQuarkFlavor`, `SampleQuarkPt`, `CreatePartonPair`, `QuarkSplitup`, `ProduceOneHadron`, `PossibleHadronMass`, `SetMinimalStringMass`, `IsItFragmentable`) | **V** | `ftf/lund_tables.cuh`, `ftf/string_decay.cuh`, `ftf/lund_fragment.cuh`. All 1,510 table entries, 9,000 `SetMinimalStringMass` triples including the DiQuark-AntiDiquark re-arrangement arm, the samplers at 8 phases each with their draw counts. The three index errors in `SetMinMasses` are transcribed as written and asserted by name - docs/RISK.md V85 - and `Kappa` is 1e30 for the reason V87 gives |
| **G4HadronBuilder** (`Build`, `BuildLowSpin`, `BuildHighSpin`, `Meson`, `Barion`) | **V** | `ftf/hadron_builder.cuh`; all 8,800 (parton, parton, phase) triples the tables can reach x 3 entry points, PDG code and draw count, worst 0. The 61 charmed and bottom substitutions are REACHABLE - `EnableBCParticles` is 1, so `SampleQuarkFlavor` draws a c or a b once in four thousand splits - and are checked |
| **G4LundStringFragmentation** (`FragmentString`, `Loop_toFragmentString`, `Splitup`, `SplitEandP`, `SplitLast`, `SampleState`, `GetLightConeZ`, `Sample4Momentum`, `StopFragmenting`, `DiQuarkSplitup`, the four last-splitting enumerations, `lambda`) | **V** | `ftf/lund_fragment.cuh`, entry point `ftf::ftf_fragment_string()`. 19 string cases x 2 directions x 8 phases, hadron by hadron: species, four-momentum, formation time and the event's total draw count, worst 9.1e-14. The final-state ENUMERATION is dumped as well as the sampled index (`ftf_laststates.csv`), because `SampleState` collapses up to 350 weighted states to one number and a weight change of a few per cent moves no index - measured, and invisible at 20,000 events too. Statistically: 19 cases x 20,000 strings, species counts and multiplicities, worst 3.6 sigma against a 5-sigma gate; at 200,000 a side every case agrees to 0.16% (docs/RISK.md V88) |
| G4FragmentingString (all four constructors, `TransformToAlignedCms`, `TransformToCenterOfMass`, `LorentzRotate`, the light-cone accessors) | **V** | `ftf/fragmenting_string.cuh`. The redundancy in the state is reproduced rather than derived: `Pplus` after a `SetPleft` is `(Pleft+Pright).plus()`, which is not the same double as `Pleft.plus()+Pright.plus()` |
| CLHEP `HepLorentzRotation` (`set(bx,by,bz)`, `rotateY`, `rotateZ`, `inverse`, `vectorMultiplication`), `Hep3Vector::phi`/`theta` | **V** | `ftf/lorentz.cuh`. Transcribed as a MATRIX and not as `boost()` + rotations: composing then applying is a different order of products, and the difference is at the last bit of every hadron momentum, which is the size the oracle compares at |
| `G4KineticTrack`'s constructor, the one line of it that is physics | **V** | the kaon0 / anti_kaon0 -> K0S or K0L coin toss, in `ftf/lund_fragment.cuh`. The masses are equal to the last bit, so it changes no kinematics - only the species and the stream |
| **G4ExcitedStringDecay** (`FragmentStrings`, `FragmentString`, `EnergyAndMomentumCorrector`) | **V** | `ftf/string_fragmentation.cuh`, entry point `ftf::ftf_fragment_strings()`. 8 string-vector cases x 8 phases: every hadron's species, four-momentum, invariant mass and formation time, and the event's draw count, all exact but the momenta at 3.9e-13. The corrector is dumped SEPARATELY and driven directly (`ftf_corrector.csv`, 8 cases including all four of its early returns, worst 0) because it is a 500-iteration fixed point and a transcription that reaches the same answer by a different route is right while one that never ran is not. Statistically 8 cases x 20,000 events on species, multiplicity and **the energy balance** - a histogram of log10 of the relative energy error, which is where a port that skipped the correction shows up: 99% of Geant4's events land in the -6 bin, and the perturbation that raises the `perMillion` trigger to 1e-2 moves 9,070 of them to -3 (95 sigma) |
| G4SampleResonance (`SampleMass`, `BrWigInt0`, `BrWigInv`), and `GetMinimumMass` as a column | **V** | same file; 2,728 sampled masses with their draw counts at worst 0, over the zero-width arm, the `minMass > maxMass` protection and every short-lived particle in the table with the arguments `FragmentStrings` passes. `GetMinimumMass` is recursive over decay tables and is **not** transcribed: it is `minmass` in `data/ftf_hadrons.hh`, dumped from Geant4 and compared here against the oracle's own argument - 1,670 points - so the column is load-bearing rather than decorative |
| `G4ExcitedString` (`Get4Momentum`, `LorentzRotate`, `IsExcited`) | **V** | `ftf::ExcitedString` in the same file, with the NOT-EXCITED case (a string that carries a hadron rather than a parton pair) and the kinky-string third parton refused by name (`kKinkyStrings`) |
| **G4FTFModel**, G4FTFParticipants, G4DiffractiveExcitation, G4ElasticHNScattering, G4DiffractiveSplitableHadron, G4FTFAnnihilation, G4VPartonStringModel, G4InteractionContent, **G4TheoFSGenerator** | **P** | written by P11b - see 2.1.11b below. `G4FTFModel::AdjustNucleons` is what is NOT written, and it is only on the path below 1 GeV/c per nucleon, i.e. only for an anti-baryon beam |
| G4QGSModel and the QGS string arm | **-** | refused by name: not in QBBC's FTFP chain, and 2.1.4's note on the null `GetPrimaryProjectile()` (docs/RISK.md V50) is why it must stay refused rather than half-built |

Tests: `test_ftf_params.cu` - 152,576 exact comparisons, worst 4.4e-16, 4,920 rows refused by
name. `test_ftf_lund.cu` - 70,555 comparisons, of which 26,400 are draw counts and 889 are
statistical z-scores. `test_ftf_strings.cu` - 11,254 comparisons over the layer above it, of
which 500 are statistical.

Device probes, never launched, `-Xptxas -v` on sm_52. `ftf_lund_device_probe`: **210 registers,
152-byte stack frame, 0 bytes spilled**, 213,047 bytes gmem for the tables, 6,160 bytes cmem[2];
`ftf_fragment_string` itself inlines to a 0-byte frame. `ftf_strings_device_probe`, which
instantiates the layer above: **236 registers, 56-byte stack frame, 0 bytes spilled**, 3,988
bytes cmem[2]. The per-string workspace is 33,320 bytes and the string-decay workspace that
contains it 53,816, both behind a pointer and one per TRACK rather than per string: the
350-entry final-state enumeration is 5,600 of the first and the three hadron lists 27,648, and
on the stack either would blow `Upload`'s 16,384-byte frame limit before the first string
fragmented.

#### 2.1.11b FTFP: the model that makes the strings, and the entry point (P11b)

The half P11 could not write. `ftf::apply_yourself` exists: a projectile and a nucleus in, a
vector of excited strings and a wounded nucleus out, then through P11's fragmentation and into
2.1.4's `Propagate`. The three stages are worth keeping apart because only the first two are
what anyone means by "the Glauber model":

1. **`G4FTFParticipants::GetList`** samples an impact parameter uniformly in a disc of radius
   `GetOuterRadius() + 2 fm` and marks every nucleon within `sqrt(Xtotal/pi/10)` of it. The
   whole test is `RadiusOfHNinteractions2 > b^2` - a STEP function, which is why
   `impact2/fermi/fermi` has to be two divisions and not one by fermi squared.
2. **`ReggeonCascade`** marks MORE nucleons - neighbours of the participants, at
   `Cnd exp(-b^2/R2nd)` - and gives them status 3. They collide with nothing; they exist so the
   residual is left with holes and an excitation. For a 10 GeV proton on lead they are 10.9 of
   the 14.6 wounded nucleons, so most of the residual's excitation comes from nucleons that
   never interacted.
3. **`PutOnMassShell`** gives every marked nucleon a transverse momentum and a light-cone
   fraction and rebuilds the whole system's kinematics in the c.m.s. This is where the nucleons
   go back ON their PDG mass shells - P9's nucleus leaves every one of them off-shell downwards
   (docs/RISK.md V50), and `G4DiffractiveExcitation::ExciteParticipants`' unconditional
   `toBePutOnMassShell` is the step that repairs it. A caller of `preco::propagate_residual`
   that skips it selects the QGS arm.

| Geant4 class | | Where |
|---|:--:|---|
| **G4DiffractiveSplitableHadron** (`SplitUp`, `ChooseStringEnds`, `Diquark`, `GetNextParton`), G4VSplitableHadron's two constructors | **V** | `ftf/splitable_hadron.cuh`. 37 hadron codes x 8 phases: both parton codes and the draw count, worst 0. **G4Parton's CONSTRUCTOR SAMPLES** - one uniform for the colour and, when `GetPDGiSpin()` is non-zero, a second for the spin projection - so a proton's SplitUp costs five deviates and not two. Nothing in FTF reads `theColour` or `theSpinZ`; only the draws matter, and only the draw-count column could have found it. docs/RISK.md V98 |
| **G4FTFParticipants** (`GetList` both arms, `SortInteractionsIncT`, `ShiftInteractionTime`), G4VParticipants (`Init`, `InitProjectileNucleus`), `G4V3DNucleus::ChooseImpactXandY`, G4InteractionContent | **V** | `ftf/participants.cuh`. Five REPLAYED nuclei (C, O, Al, Fe, Pb) x five projectiles x 8 phases: the impact parameter, the participant count, each participant's identity and interaction time, and the draw count - 1,460 comparisons, worst 0. The nucleus is replayed because `G4Fancy3DNucleus`'s rejection sampling shares no stream with Philox; P9's pattern. An interaction whose target nucleon was already hit gets `SetTarget(0)` in Geant4 and is reported here rather than dereferenced |
| **G4ElasticHNScattering** (`ElasticScattering`, `GaussianPt`) | **V** | `ftf/elastic_hn.cuh`. 12 constructed collisions x 8 phases: both four-momenta, both collision counts, the projectile's inherited creation time and position, the draw count - 1,536 comparisons, worst 0. The two collision counts are incremented BEFORE any test that can return false, so a failed elastic scattering still makes both hadrons look like participants |
| **G4DiffractiveExcitation** (`ExciteParticipants`, `_doChargeExchange`, `_doDiffraction`, `_doNonDiffraction`, `CreateStrings`, `ChooseP`, `GaussianPt`, `UnpackMeson`, `UnpackBaryon`, `NewNucleonId`) | **V** | `ftf/diffractive_excitation.cuh`. The same grid: both PDG codes (a charge exchange changes them), four-momenta, statuses, collision counts and draw count - 1,536 comparisons, worst 0, reaching all four arms. `M0projectile` is the PDG mass and not `Pprojectile.mag()` (the two `Uzhi Aug.2019` comments), which is what puts the nucleons back on shell. **Refused by name:** the kinky-string arm, unreachable because `Pt2Kink` is 0 |
| **G4FTFModel** (`Init`, `GetStrings`, `StoreInvolvedNucleon`, `ReggeonCascade`, `PutOnMassShell`, `ComputeNucleusProperties`, `GenerateDeltaIsobar`, `SamplingNucleonKinematics`, `CheckKinematics`, `FinalizeKinematics`, `ExciteParticipants`, `BuildStrings`, `GetResiduals`, `GaussianPt`) | **P** | `ftf/ftf_model.cuh`. No exact oracle - every path starts from a sampled nucleus - so it is checked statistically, below. `ExciteParticipants`' inelastic-rejection factor is an INTEGER division and reads as a smooth suppression: docs/RISK.md V99. **Refused by name:** `AdjustNucleons` and its three algorithm methods (`kAdjustNucleons`), reached only below 1 GeV/c per nucleon, i.e. only by an anti-baryon; an anti-NUCLEUS projectile, because P9's `bic::Nucleon` has no anti types to re-type into. The annihilation branch of `ExciteParticipants` is written, `theAdditionalString` with it, and the participant pool carries 64 slots for it past the two nuclei |
| **G4VPartonStringModel::Scatter** (the rotation to z, the 1000-attempt loop, the string-vector assembly, the wounded-nucleus rotation to the lab, the unphysical-residual table, the `SumMass > InvMass` check) | **V** | `ftf/theo_fs_generator.cuh`. The retry loop is the model and not a safety net: each attempt REBUILDS BOTH NUCLEI, so one `apply_yourself` call's deviate count is not a function of the physics alone. It is also a CONDITIONING and not a filter on failures - the unphysical-residual table rejects the attempts with the most hit nucleons, 7.2% of them for C12 on carbon and 1.6% for a proton, which moves every distribution measured through it (docs/RISK.md V105). `ftf_prescatter.csv` measures the model with the loop taken off and that table's own four-way outcome: 18,569/673/711/47 against Geant4's 18,573/675/697/55 out of 20,000. After 1000 attempts Geant4 returns the primary unchanged at `z = 2*OuterRadius` - an elastic-looking final state out of an inelastic process - and `kScatterAttemptsExhausted` says so |
| **G4TheoFSGenerator::ApplyYourself** | **P** | same file, entry point `ftf::apply_yourself()`. The two dummy branches below 100 MeV (a charm/bottom hadron, a hypernucleus) are reproduced rather than refused, because returning the primary IS what Geant4 does. **Refused by name:** `PropagateNuclNucl` (2.1.4 ported only the hadron-nucleus arm), `G4DecayStrongResonances` (taken when EVERY target nucleon was hit), `G4QuasiElasticChannel` and `G4CRCoalescence`, both unreachable in QBBC |
| **G4FTFAnnihilation** (`Annihilate`, `Create3QuarkAntiQuarkStrings`, `Create1DiquarkAntiDiquarkString`, `Create2QuarkAntiQuarkStrings`, `Create1QuarkAntiQuarkString`, `UnpackBaryon`, `GaussianPt`) | **V** | `ftf/annihilation.cuh`. All four channels and the nine-by-two weight table that picks among them. Nine collisions x 8 phases - both PDG codes, both four-momenta, statuses, collision counts, the projectile's time and position, both partons AND their momenta, the additional string with its own two partons, and the draw count: **2,862 comparisons, worst 0**, reaching all four channels. `GetProbabilityOfAnnihilation()` is non-zero only for an anti-baryon projectile, so no other beam reaches this at all. It is also the only producer of `theAdditionalString`, which is what makes `CreateStrings`' `HadronIsString` arm reachable - and that arm rebuilds the string from the parton objects as they are, so `SplitableHadron` has to store the parton momenta that nothing else in FTF reads back |

**The energy windows, read from the CONSTRUCTED processes and not from the builders.**
`ref/oracle/ftf_windows.csv` walks every hadronic process QBBC registered and writes one row per
model it holds, with the energy range that model is asked for - so the paragraph below is a
measurement rather than a reading of the source, and `test_ftf_model.cu` runs `apply_yourself`
at three energies inside every FTFP row, on carbon and on lead, and reports for each point
whether it ran, refused by name, or did neither (which would fail).
`G4HadronInelasticQBBC::ConstructProcess` gives FTFP the range above
`G4HadronicParameters::GetMinEnergyTransitionFTF_Cascade()` = 3 GeV for p, n, pi+- and, through
`G4HadronicBuilder::BuildKaonsFTFP_BERT`, for K+-; `BuildFTFP_BERT(..., bert=false)` gives it
**all** energies for anti-nucleons, anti-light-ions and anti-hyperons, which is the only way the
sub-GeV arm is reachable and therefore the only reason `AdjustNucleons` matters;
`G4IonPhysicsXS` gives it 3 GeV/nucleon and above for d, t, He3, alpha and GenericIon, on the
same process as `G4BinaryLightIonReaction` below 6 GeV/nucleon. The quasi-elastic channel is
absent because no QBBC builder calls `SetQuasiElasticChannel`.

The table adds three things reading the builders did not give. **The LAMBDA gets FTFP too**, on
`lambdaInelastic` from 3 GeV, and the **anti-lambda at every energy** - a hyperon beam is not
something this document mentioned, and it is the one FTFP row this package cannot run, because
P5's `G4HadronNucleonXsc` has no lambda branch and reports `kHadronNucleonXscRefused`. **The
NEUTRON has no FTFP row at all**: QBBC gives it `NeutronGeneralProc`, whose
`GetHadronicInteractionList()` is empty because the models live inside the sub-processes it
wraps - so a neutron reaches FTFP through a door this table cannot see, and P8d's note about
`EnableNeutronGeneralProcess` applies here too. And every ceiling is **100 TeV**, not the 50 GeV
the statistical cases stop at; the sweep caps itself at 100 GeV.

**What that means for what is usable today.** p, n, pi+- and K+- above 3 GeV go all the way from
`apply_yourself` to the hand-over. An **ANTI-NUCLEON above 1 GeV/c** does too: its annihilations
are written, 20,000 anti-protons and 20,000 anti-neutrons on carbon and on lead all produce
strings, and 0.24 to 0.30 additional strings per event come out of the three-string channel.
Below 1 GeV/c it is refused at `AdjustNucleons` (`kAdjustNucleons`), which is where Geant4 would
strip nucleons off the nucleus before the collision. An ION beam builds its projectile nucleus,
its participants and its strings - the nucleus-nucleus arms of `GetList`, `ReggeonCascade`,
`PutOnMassShell` and `BuildStrings` are written and take the same code as the hadron arms - and
is refused at the hand-over, because `PropagateNuclNucl` is 2.1.4's gap and not this package's.
Every ion case agrees statistically once the port is measured the way the oracle was:
docs/RISK.md V105.

**And one gap that is not this package's but is on its main path.** 2.1.4 refuses
`G4DecayKineticTracks`, the first line of both `Propagate` entry points, which decays every
short-lived track through `G4KineticTrack::Decay()`. It is written there as an edge case. It is
not: the FTF fragmentation resamples every short-lived product's mass and hands the rho, omega,
K* and Delta on undecayed, so **essentially every FTFP event contains one** and
`preco::propagate_residual` refuses it. Everything above the hand-over is validated; the
hand-over itself is reached and reports. docs/RISK.md V100.

Tests: `test_ftf_model.cu`, **20,301 comparisons**. **Exact, worst 0.000e+00 in all 33 buckets**
(12,365 comparisons) under the eight-value cycle engine. **Statistical**, 24 cases x 20,000
events - {p, n, pi+, pi-, K+} on {C, O, Al, Fe, Pb} at 4, 10 and 50 GeV, {alpha, C12} on {C, Pb}
at 8 and 20 GeV/nucleon, and {pbar, nbar} on {C, Pb} at 5 GeV - thirteen histograms against a
5-sigma gate; plus the rapidity/xF and pT spectra as their first moments per species (`<E>`,
`<pz>`, `<pt2>` in the lab) and the energy balance as `<E_total>` per event, one number per case.
Every one of the 23 statistical rows is under 5 sigma, worst **4.13** (`p_O_10` multiplicity),
and 20,301 comparisons in all, in **156 s**.

**The generality sweep.** `ftf_windows.csv` has 17 FTFP rows; the test runs `apply_yourself` at
three energies inside each, on carbon and on lead - 96 (beam, energy, target) points. **11 ran
to a final state, 85 refused BY NAME, none was silent.** The 85 break down as 31
`preco::GeneratorRefusal` (P6's `Propagate`, and V100 says why every FTFP event reaches it), 24
`kAdjustNucleons` (an anti-baryon below 1 GeV/c), 24 `kPropagateNuclNucl` (d, t, He3, alpha at
the hand-over) and 6 `kHadronNucleonXscRefused` (the lambda beam, which has no cross-section
branch below this package).

Three of the thirteen exist to LOCALISE a disagreement rather than to find one: the impact
parameter is the geometry alone, `participants` is `A - NumberOfTargetSpectatorNucleons` (the
Glauber count, separately from the reggeon cascade) and `nncoll` is how many of those were
actually excited. The per-object histograms - species, string mass, excited/not-excited - use a
MEASURED variance and not a Poisson one, because a 4 GeV proton on lead makes a dozen strings at
once and the per-event bin count has Var/Mean of 5.7; assuming Var = Mean reported a 1.45%
difference in a quarter-million-count bin as 5.09 sigma.

**V88's rule is applied, and it answered in both directions.** The three hadron cases carrying
the worst z at 20,000 events - n_Fe_10's NN-collision count (3.49), pip_Al_10's multiplicity
(3.43) and pim_C_10's string count (3.31) - are re-run at **200,000 events on both sides**
(`ftf_modelbig_*.csv`). Every one of them FELL: 1.74, 2.42 and 2.54. **C12 on carbon at 8
GeV/nucleon did the opposite**: 5.34 sigma on the participant count at 20,000 became **12.81 at
200,000**, and the NN-collision count 4.92 became 11.05. Growing by about sqrt(10) is what a
real difference does - and the difference was real, but it was not in the physics. It was that
`dump_modelstat` measures the string-level counters from `Init` + `GetStrings` and the port was
measuring them through `ftf_scatter`, whose unphysical-residual table re-samples 7.2% of C12 +
C attempts and 1.6% of proton ones. Both sides run the same two passes now and the row is 2.95.
docs/RISK.md V105. The same rule applied to the C12 NUCLEUS - 200,000 sampled configurations of
its outer radius, transverse RMS and radial RMS against Geant4's (`C12big` in
`ftf_nucstat.csv`) - left it under 3 sigma, which is how the nucleus was ruled out first. That
is the distinction V88 says one 20,000-event number cannot make, and it is worth recording that
it also cannot tell a difference in the model from a difference in what the two sides measured:
growth at sqrt(10) says "real", not "physics".

Device probe, never launched, `-arch=sm_86`: `ftf_apply_device_probe`, which instantiates the
whole of `ftf::apply_yourself`, is **255 registers, an 864-byte stack frame and 0 bytes spilled**
in the entry function; `ftf_scatter` compiled on its own spills 24 bytes of stores and 48 of
loads, which the entry's own frame absorbs. 400 bytes cmem[0], 5,968 bytes cmem[2]. The
hand-over into P6 is behind a `__noinline__` and costs a 0-byte frame, because it stops at
`propagate_residual` and does not pull in `preco::deexcite`'s 10 kB.
`sizeof(FtfWorkspace<250,64,512,320,256,96>)` is **332,368 bytes** - two nuclei, a 565-slot
splitable-hadron pool (the 64 extra slots are G4FTFAnnihilation's `theAdditionalString`), the
interaction list, 320 excited strings and P11's 53,816-byte string-decay workspace - one per
TRACK in flight and every byte behind a pointer. That is six times P11's and it is the number to
look at before this runs on a device: docs/RISK.md V101.

### 2.2 What QBBC needs and is not there

| QBBC constructor | needs | status |
|---|---|:--:|
| `G4HadronElasticPhysicsXS` | process `G4HadronElasticProcess`; cross sections `G4BGGNucleonElasticXS`, `G4NeutronElasticXS`, `G4BGGPionElasticXS`, `G4ChipsProtonElasticXS`, `G4ComponentGGHadronNucleusXsc`; final states `G4HadronElastic`, `G4ChipsElasticModel`, `G4ElasticHadrNucleusHE`, `G4AntiNuclElastic` | XS partial (above), **final state absent** |
| `G4HadronInelasticQBBC` | `G4HadronInelasticProcess`; `G4ParticleInelasticXS`, `G4BGGPionInelasticXS`, `G4NeutronInelasticXS`; models `G4BinaryCascade`, `G4CascadeInterface` (Bertini), `G4TheoFSGenerator` + `G4FTFModel` + `G4ExcitedStringDecay` + `G4QGSModel`, `G4PreCompoundModel`, `G4GeneratorPrecompoundInterface`, `G4ExcitationHandler` | cross sections **V** (2.1.1); `G4ExcitationHandler` **V** (2.1.3); `G4PreCompoundModel` **V**, `G4GeneratorPrecompoundInterface` **P** (2.1.4); the FTFP arm's string FRAGMENTATION and its tuned parameters **V** (2.1.11), the model above them **none**; the process and the cascades: **none** |
| `G4IonPhysicsXS` | `G4ParticleInelasticXS`, `G4BinaryLightIonReaction` | **none** |
| `G4IonElasticPhysics` | `G4ComponentGGNuclNuclXsc`, `G4NuclNuclDiffuseElastic` | **none** |
| `G4StoppingPhysics` | `G4HadronStoppingProcess`, `G4HadronicAbsorptionBertini`, `G4HadronicAbsorptionFritiof`, `G4MuonMinusCapture`, `G4EmCaptureCascade` | **none** |
| `G4NeutronTrackingCut` | `G4NeutronKiller` | **T** - and the class is not the answer. With `EnableNeutronGeneralProcess = 1` this constructor `return`s without creating a `G4NeutronKiller`; the cut is the two lines at the top of `G4NeutronGeneralProcess::PostStepGetPhysicalInteractionLength`. Ported in `step_neutral`, before geometry and before the cross section, and it **deposits nothing** - `theTotalResult->Initialize(track)` zeroes both energy deposits, so the neutron's kinetic energy is discarded rather than given to the volume. The engine books it (`RunStats::neutron_killed_energy`) because Geant4 does not conserve energy across this either and the only way anyone finds out is a printed number. *(P8d: with the flag OFF - which is the stage-1 reference, and is reachable, docs/RISK.md V60 - `G4NeutronTrackingCut` does build a real `G4NeutronKiller`, and the run prints `TimeCut(ns)= 10000  KinEnergyCut(MeV)= 0`: the same two numbers. So the one line in `step_neutral` is right for both configurations and `P` became `T` without any code changing.)* |
| `G4EmExtraPhysics` | gamma-nuclear (`G4GammaNuclearXS`, `G4LowEGammaNuclearModel`, LEND), electro- and muon-nuclear (`G4ElectroVDNuclearModel`, `G4MuonVDNuclearModel`), 18 neutrino classes, `G4SynchrotronRadiation`, `G4AnnihiToMuPair`, `G4GammaConversionToMuons`, `G4eeToHadrons`, `G4MuonToMuonPairProduction` | **none** |

### 2.3 The model tree, by size

`de_excitation` is section 2.1.3 - its default-configuration subset is transcribed and
validated. Every other one of these is at zero. `de_excitation`, `cascade` and
`binary_cascade` are the ones QBBC actually runs for a proton in water; `particle_hp`,
`inclxx`, `lend`, `im_r_matrix`, `qmd`, `abla`, `parton_string` are alternatives other lists
select.

```
  de_excitation      232      particle_hp        250      inclxx             163
  cascade            119      im_r_matrix        108      parton_string       37
  pre_equilibrium     29      binary_cascade      26      radioactive_decay   24
  lend                23      coherent_elastic    19      lepto_nuclear       16
  qmd                  8      abla                 6      fission              5
  theo_high_energy     3      abrasion             2      em_dissociation      1
  gamma_nuclear        1      quasi_elastic        1
```

`cross_sections/` 65 headers: 3 ported (above), 62 not.
`management/` 15, `processes/` 18, `stopping/` 11, `util/` 53: **0 ported.**

### 2.4 The process framework and the elastic final states (package P5)

Everything here is `V`: transcribed, checked by a registered test, and not yet reached by a
particle, because no species is stepped through a hadronic process until P8 wires one in.

| Geant4 class / function | QBBC | | Where |
|---|:--:|:--:|---|
| G4HadronicProcess::PostStepDoIt / FillResult / CheckResult / CheckEnergyMomentumConservation | y | **V** | `hadronic/process.cuh` |
| G4HadronicProcess::BuildPhysicsTable (the G4HadXSType choice and the integral-approach rejection) | y | **V** | same |
| G4EnergyRangeManager::GetHadronicInteraction | y | **V** | same |
| G4CrossSectionDataStore::ComputeCrossSection / SampleZandA | y | **V** | same |
| G4HadProjectile, G4Nucleus (what the elastic models read of it) | y | **V** | same |
| G4HadFinalState, G4HadSecondary | y | **V** | same |
| G4HadronicInteraction::GetFatalEnergyCheckLevels | y | **V** | same |
| G4HadronElasticProcess::PostStepDoIt | y | **P** | `hadronic/elastic/elastic_process.cuh` |
| G4RToEConvForProton::Convert | y | **V** | same |
| G4HadronElastic::ApplyYourself / SampleInvariantT / GetSlopeCof | y | **V** | `hadronic/elastic/hadron_elastic.cuh` |
| G4ChipsElasticModel::SampleInvariantT | y | **V** | `hadronic/elastic/chips_elastic.cuh` |
| G4ChipsProtonElasticXS (GetChipsCrossSection, CalculateCrossSection, GetPTables, GetTabValues, GetQ2max, GetExchangeT, GetSlope) | y | **V** | same |
| G4ChipsNeutronElasticXS (the same seven) | y | **V** | same, tables in `elastic/chips_neutron_lowe.hh` (422 isotope rows) |
| G4ElasticHadrNucleusHE (SampleInvariantT, FillData, FillFq2, HadrNucDifferCrSec, DefineHadronValues, GetHadronNucleonXsc*) | y | **V** | `hadronic/elastic/elastic_hadr_nucleus_he.cuh` |
| G4NuclNuclDiffuseElastic::SampleInvariantT / SampleCoulombMuCMS / InitDynParameters | y | **P** | `hadronic/elastic/nucl_nucl_diffuse_elastic.cuh` |

`tests/test_hadronic_process.cu` and `tests/test_elastic_models.cu`, against
`ref/oracle/elastic_*.csv` from `ref/dump/dump_elastic.cc`.

**How the G4ElasticHadrNucleusHE tables are checked, and why not directly.** The plan asks for
its initialisation tables to be dumped and matched. They cannot be: `G4ElasticData`'s `R1`, `R2`,
`Pnucl`, `Aeff` and its `fCumProb[NENERGY]` are all **private**, `fElasticData[NHADRONS][ZMAX]`
is a private static, and every function that builds or reads them - `FillData`, `FillFq2`,
`HadrNucDifferCrSec`, `DefineHadronValues`, `GetLightFq2`, `HadronNucleusQ2_2`,
`HadronProtonQ2` - is private too. `SampleInvariantT` is the class's only public door, and
editing the Geant4 install to widen it would invalidate the oracle it is the oracle of. So the
port builds the same tables (`he_fill_data`) and they are validated *through* the sampler:
bitwise on `-t` over 1408 points, which is sensitive enough that changing the tables' mb->GeV^-2
constant from 2.568 to 2.5681 - four parts in 100,000 - moves `-t` by 1.2e-5 and fails five
assertions. That is measured, not assumed. It is nonetheless coverage of the table entries those
1408 points reach and not of every entry, and it is the reason this row says so.

**The numbers.** The four elastic samplers are compared under a prescribed eight-value uniform
cycle, which makes `SampleInvariantT` and `ApplyYourself` deterministic functions of their
inputs, at 1408 points each (2 projectiles x 8 targets from H1 to Pb208 x 11 energies from 1 MeV
to 20 GeV x 8 phases). `-t`, cos(theta_cm), the primary's final energy, the recoil energy and
both direction vectors agree **bitwise** for G4HadronElastic, G4ChipsElasticModel and
G4NuclNuclDiffuseElastic, and to one ulp on `-t` for G4ElasticHadrNucleusHE; the tolerance is
1e-15. The CHIPS cross section is within 1e-16 over 832 points and its (-t)max and GetExchangeT
are bitwise. Statistically, 20,000 samples per point over 176 points: the worst moment and worst
histogram bin are 3.7 and 3.6 sigma. The framework: the overlap model choice over 200,000 draws
at 28 points agrees with Geant4 to 0.0016 and with the closed form to 0.0011; element selection
frequencies to 0.0025 and isotope frequencies to 0.0048.

**What is refused, by name.**

- `G4NuclNuclDiffuseElastic`'s **angle table** - `BuildAngleTable`, `SampleTableThetaCMS`,
  `SampleTableT`, `GetScatteringAngle`, the Bessel functions and the Legendre integrations. In
  11.1.1 none of it is reachable: `SampleInvariantT` has the `SampleTableT` call commented out,
  and `BuildAngleTable` is only called from `Initialise()`, which nothing in the source tree
  calls and which is not the `InitialiseModel` hook a physics list would use. So QBBC's ion
  elastic scattering IS screened Rutherford scattering truncated at the Coulomb grazing angle,
  and that is what is ported. 1500 lines that the oracle's own Geant4 never executes cannot be
  validated by any oracle. `P` for that reason and not for a gap.
- `G4HadronElasticProcess`'s **diffraction branch**. `fDiffraction` is null unless
  `SetDiffraction` is called, and only `G4ChargeExchangePhysics` calls it. `P`.
- `G4ChipsElasticModel`'s five other cross-section classes - `G4ChipsAntiBaryonElasticXS`,
  `G4ChipsPionPlus/MinusElasticXS`, `G4ChipsKaonPlus/MinusElasticXS`. No QBBC particle reaches
  them: QBBC gives pions `G4ElasticHadrNucleusHE` and kaons `G4HadronicBuilder::BuildElastic`.
  A pbar, pi or K handed to `chips_sample_invariant_t` sets `unsupported_pdg`.
- A **Z = 0 target** for the CHIPS proton class, which in Geant4 reaches
  `G4IonTable::GetIon(0, N, 0)` and has no ion. Refused rather than given a plausible mass.
- `G4AntiNuclElastic`, not started. `GetSlopeCof`, which only it calls, is transcribed.
- The **kaon0/anti_kaon0 -> kaon0S/kaon0L substitution** in `G4HadronicProcess::PostStepDoIt`.
  It needs the K0 species, which is P1's refused set; `fill_result` counts and reports a K0
  rather than substituting silently. No elastic model can produce one.
- The **`lastTH` latch** in both CHIPS classes - see `docs/RISK.md` V37.

**What the default energy-momentum check does: nothing.** `G4HadronElasticProcess::PostStepDoIt`
does not call `CheckResult` at all, and the source says why ("cannot be applied because is not
guranteed that recoil nucleus is created"). And `epReportLevel`, which gates
`CheckEnergyMomentumConservation`, is a `G4HadronicProcess` data member initialised to 0 and
changed only by `G4Hadronic_epReportLevel`, `G4HadronicEPTestMessenger` or an explicit setter -
none of which QBBC touches. There is no `G4HadronicParameters::GetEpReportLevel` in 11.1.1. So
QBBC's elastic scattering runs with **no** energy-momentum check; both are transcribed anyway,
returning a verdict instead of printing, so a test can assert on the arithmetic.

---

## 3. decay/ (6 headers) and the channel classes in particles/management/

QBBC registers `G4DecayPhysics`, which builds ONE `G4Decay` and attaches it to every particle
for which `IsApplicable` is true. Invisible for gamma / e± / p / alpha - all stable.

| Geant4 class | QBBC | | Where |
|---|---|---|---|
| G4Decay | y | **T** | `physics/decay/decay.cuh`, wired by P8 in `physics/stepper.cuh` (in flight for pi+-, pi0, K+-, mu+- and the neutron; at rest for pi+, K+, mu+ always and for the negatives under the stage switch) |
| G4DecayTable (`SelectADecayChannel`, `Insert`'s order) | y | **T** | `physics/decay/decay.cuh`, `physics/decay/decay_tables.hh` |
| G4DecayProducts (`Boost`) | y | **T** | `physics/decay/decay_products.cuh` |
| G4DynamicParticle (`Set4Momentum`, `SetMomentum`, `Get4Momentum`, the mass snap) | y | **T** | `physics/decay/decay_products.cuh`; and `Set4Momentum` again in `hadronic/capture/neutron_rad_capture.cuh`, where the `EnergyMRA2` branch that the decay path never reaches decides a capture residual's kinetic energy - see 2.1.5 |
| G4VDecayChannel (`IsOKWithParentMass`, `rangeMass`) | y | **P** | `physics/decay/decay_channels.cuh` - `DynamicalMass`'s Breit-Wigner resampling is refused; no daughter of any ported table has a width above 1e-3 of its mass |
| G4PhaseSpaceDecayChannel (1-, 2-, 3- and N-body) | y | **T** | `physics/decay/decay_channels.cuh` |
| G4MuonDecayChannel | y | **T** | `physics/decay/decay_channels.cuh` - the plain channel is what `G4MuonPlus.cc`/`G4MuonMinus.cc` install |
| G4KL3DecayChannel (+ `DalitzDensity`) | y | **T** | `physics/decay/decay_channels.cuh` |
| G4DalitzDecayChannel | y | **T** | `physics/decay/decay_channels.cuh` - reached by 1.2% of pi0 decays, and a pi0 decays inside its first step wherever it is made (`beta*gamma*c*tau` is 4e-5 mm at 100 MeV) |
| G4NeutronBetaDecayChannel | y | **V** | `physics/decay/decay_channels.cuh`. The neutron HAS the process now and this channel is the only one in its table, so it is reachable in principle and unreachable in practice: 880 s of proper lifetime is 2.6e11 mm of decay length at 100 MeV against a 300 mm phantom. `V` and not `T` for that reason, which is a statement about the geometry rather than about the code. |
| G4MuonDecayChannelWithSpin | n | **-** | exists in 11.1.1; only `G4SpinDecayPhysics` installs it, and QBBC does not register that |
| G4MuonRadiativeDecayChannelWithSpin | n | **-** | same |
| G4PionRadiativeDecayChannel | n | **-** | in the release, in no table |
| G4TauLeptonicDecayChannel | n | **-** | tau is not transported; refused by PDG code |
| G4DecayWithSpin, G4UnknownDecay, G4PionDecayMakeSpin, G4VExtDecayer | n | **-** | QBBC registers none of them |
| G4DecayProcessType | y | **-** | an enum of process sub-types; `core/step_report.cuh` already reserves `fDecay` |

Species with a transcribed table: pi+, pi-, pi0, mu+, mu-, K+, K-, neutron. The refused set is
K0L, K0S, K0, the hyperons and the tau - each refused by PDG code with a message that names it,
never treated as stable; 413 species Geant4 gives a decay table come back `kNoTable`.

The at-rest branch is `V` in a narrower sense than the rest, and it is worth stating: measured
(`ref/oracle/decay_atrest.csv`), `G4HadronicAbsorptionBertini` on pi-/K- and `G4MuonMinusCapture`
on mu- all offer an at-rest interaction length of exactly 0.0, so G4Decay's at-rest branch never
runs for any negative species. It is reachable for pi+, K+, mu+, pi0 and the neutron only. The
muon's bound decay lives inside `G4MuonMinusBoundDecay` with its own K-shell Michel sampler and
belongs to P12, not here. `decay_at_rest_competitor` names the species so a wiring package
cannot forget silently.

Wiring (P8) HAS DELIVERED the three things this paragraph asked for, and section 2.1.5 says how:
a `ParticleType` for each PDG code (`pdg_code` / `particle_type_of_pdg` /
`particle_type_of_nucleus` in `core/particle.cuh`), the process in the stepper's post-step and
at-rest paths, and the competition - which turned out to be a stage SWITCH rather than a race,
because Geant4's own answer depends on whether the at-rest captures are inactivated.

**That moment has arrived.** pi±, K±, mu±, the triton and the neutron all have stepping kernels
now and all five are unstable (`ref/oracle/species_tables.csv` carries each one's lifetime and
`stable` flag; `species_processes.csv` shows the `Decay` process on each). In this port they
stop and stay stopped. It is loud rather than silent - `QBBC.hh`'s banner says "no decay" for
every species it lists, and the `ProcessId::fDecay` slot has been reserved since before any of
them was transported - but it is a real difference from Geant4 for any run that stops one of
them, and it is the largest of the gaps P1 leaves behind. P4 is the package; P8 wires it.

---

## 4. Four discrepancies this inventory found

### 4.1 Pair production above 30 GeV

`gamma_processes.cuh`'s `pair_xs_per_atom` is a line-for-line transcription of
`G4PairProductionRelModel::ComputeParametrizedXSectionPerAtom` - the same twelve coefficients,
the same 1.5 MeV floor and `(E - 2mc^2)^2` scaling. It was *labelled* `G4BetheHeitlerModel`,
which is where those coefficients also appear, and that mislabelling hid the real boundary:

```cpp
fParametrizedXSectionThreshold = 30.0*CLHEP::GeV;   // G4PairProductionRelModel.cc:122
...
if ( gammaEnergy < fParametrizedXSectionThreshold) {
  crossSection = ComputeParametrizedXSectionPerAtom(gammaEnergy, Z);
} else {
  crossSection = ComputeXSectionPerAtom(gammaEnergy, Z);   // numerical DCS integration + LPM
```

Below 30 GeV the port and Geant4 evaluate the same expression. **Above 30 GeV the port keeps
extrapolating the fit while Geant4 integrates the differential cross section with LPM
suppression.** Geant4's own comment puts the fit's validity at 100 GeV and warns it diverges
above 80-90 GeV, so the error is small at 30 GeV and grows from there. The *final state* above
that energy is already correct - `em/pair_production.cuh` transcribes the real
`G4PairProductionRelModel` sampler, LPM and all. It is only the rate.

The comments in `gamma_processes.cuh` now name the right class and record the boundary.

### 4.2 The alpha's fluctuation model - found here, now fixed

`G4EmBuilder::ConstructIonEmPhysics` splits the light nuclei two ways, and the split is not
a function of charge or mass:

```cpp
part = G4Deuteron::Deuteron();  ph->RegisterProcess(new G4hIonisation(),   part);
part = G4Triton::Triton();      ph->RegisterProcess(new G4hIonisation(),   part);
part = G4Alpha::Alpha();        ph->RegisterProcess(new G4ionIonisation(), part);
part = G4He3::He3();            ph->RegisterProcess(new G4ionIonisation(), part);
```

`G4ionIonisation` calls `SetFluctModel(G4EmStandUtil::ModelOfFluctuations(true))`, and that
factory returns **`G4IonFluctuations`**, not `G4UniversalFluctuation`. `G4hIonisation` passes
`(pname == "GenericIon" || pname == "alpha")` and so reaches the same answer for an alpha by a
second route; `G4MuIonisation` passes `true` for its low-energy model only.

| species | Geant4's model | this port, before | now |
|---|---|---|---|
| e-, e+ | G4UniversalFluctuation | Universal | Universal |
| proton, pi±, K±, pbar | G4UniversalFluctuation | Universal | Universal |
| deuteron, triton | G4UniversalFluctuation | not transported | not transported |
| **alpha, He3, GenericIon** | **G4IonFluctuations** | **Universal - wrong** | **Ion** |
| mu± below the Bragg boundary | G4IonFluctuations | not transported | not transported |

The two models share a mean by construction, so the alpha B1 comparison in `RESULT.md` was
right with either and the error lived entirely in the width of the straggling - the one thing
a single scoring volume structurally cannot see.

**What the model is.** Above `10 MeV x charge x mass/m_p` (79.45 MeV for an alpha) the ion is
fast enough that its charge state is fixed and G4IonFluctuations defers to
G4UniversalFluctuation. Below it, a variance is computed in closed form and sampled - Gaussian
by rejection above `meanLoss/sigma = 2`, Gamma between 0.1 and 2, uniform below. The variance
is Bohr's, times two empirical corrections: Q. Yang et al., NIM B61 (1991) 149-155 for the
straggling a fluctuating charge state adds (96 elements x 4 coefficients, extracted by
`tools/extract_yang.sh`), and H. Geissel et al., NIM B195 (2002) 3 for the Fermi-gas factor.

**Two things it needed that the port did not have:**

- `Material::state`. Yang's second term reads a different parameter row and divides the reduced
  energy differently for a gas, and Geant4 asks `kStateGas == GetState()` rather than looking at
  the density. The state is now carried on the material and checked against Geant4's own for
  every material in the oracle.
- `G4Pow::powA`. Not `std::pow`: like `A13` before it, it is a third-order expansion about a
  tabulated point, and the difference is ~1e-7 relative - a thousand times the tolerance this
  is checked at. `logX`, `logBase` and `expA` are transcribed in `data/g4pow.hh`.

**Result**: `tests/test_ion_fluctuation.cu`, 2,058 dispersion points across 7 materials, alpha
and proton, gas and condensed, Yang-active and not, agreeing with
`G4IonFluctuations::Dispersion` to **6.3e-16**. Plus the Vavilov handover to 0, the gas flag
for every material, and the sampled mean in all three sampling branches.

### 4.3 The reference is a table, not a model

Less a discrepancy between this port and Geant4 than between two readings of what "Geant4"
means. `G4VEnergyLossProcess` never evaluates a stopping-power model during transport. At
initialisation it builds a dE/dx vector on a log grid - `MinKinEnergy` 100 eV to `MaxKinEnergy`
100 TeV, `NumberOfBinsPerDecade` 7, so 85 points - splines it (`G4LossTableBuilder::splineFlag`
defaults to true), and integrates that for a range: seed `2*E0/dedx0`, then 100 midpoint
sub-steps per bin against the spline.

This port evaluated the models exactly, on 256 points from 1 keV, with 16 trapezoids. Finer
physics, and a different table from the one Geant4 transports on.

Rebuilt on Geant4's grid, with Geant4's spline and Geant4's integration:

| | before | after |
|---|--:|--:|
| proton dE/dx, 1-10 keV | 0.407% | **0.000%** |
| proton range, 0.1-2 MeV | 0.167% | **0.018%** |
| mu- range, 1-10 keV | 52.5% | **0.836%** |
| GenericIon range, 1-10 keV | 3.43% | **0.001%** |
| **Bragg peak R80 vs Geant4** | **-0.232 mm** | **-0.014 mm** |
| plateau dose | +0.324% | **+0.052%** |

The zeros are not rounding: on the same grid this port evaluates the same models at the same
points, so away from a model boundary the two tables are the same table.

This was `RISK.md` V5, open for months behind a long and entirely correct list of physics it
had been ruled out to be. It was found by adding muons - a 1 keV table floor is invisible for a
proton and is most of a muon's range, so the same defect that cost the proton 0.3% cost the
muon 52%. Widening the species set was a better debugging tool for the proton than any further
work on the proton.

**The e+- table was the last place this rule had not been applied, and it cost 54% of a 1 GeV
electron's dose.** It was 128 linear bins of RANGE alone - no dE/dx vector, no inverse-range
vector, the inverse done by binary search and a log-linear interpolation - from 1 keV to
100 MeV, returning the last bin above the ceiling. `RISK.md` V64 is what that was worth and V77
is what rebuilding it on Geant4's grid found underneath. The same table on the same 85 nodes:

| | before | after |
|---|--:|--:|
| e- range, 1 GeV to 100 TeV | clamped to the 100 MeV value | **3e-9** |
| e- dE/dx at the nodes, above 1 MeV | no dE/dx table existed | **exact** |
| e- range, 0.1-1 keV (nodes) | below the table | **0.0022%** |
| e+ range, 1-10 MeV | the ELECTRON's range | **exact** |
| **B1, 1 GeV electron beam vs Geant4** | **-53.96%** | see docs/B1_SWEEP.md |

The widening that found it was not a species this time but an ENERGY: eleven of the sweep's
twelve beams agreed, and the twelfth was the only one that had ever asked the table a question
above 100 MeV.

---

### 4.4 ...except where it is a model, and the condition is a species name

4.3's rule has held everywhere it has been applied, which is why the place it inverts is worth
its own section rather than a footnote on the row. `G4VMscModel::GetTransportMeanFreePath`
reads a `G4PhysicsVector` when one exists and evaluates `CrossSectionPerVolume` when it does
not, and the one line that decides is `G4VMscModel::GetParticleChangeForMSC`
(G4VMscModel.cc:94):

```
    if(p->GetParticleName() != "GenericIon" &&
       (p->GetPDGMass() < CLHEP::GeV || ForceBuildTableFlag()) ) { ...build xSectionTable... }
```

`SetForceBuildTable` is called nowhere in 11.1.1 - `grep -rn "SetForceBuildTable" source/`
finds its declaration, its definition and the flag it sets, and no caller - so the flag is
always false. Every species `G4UrbanMscModel` serves therefore fails the test: **GenericIon by
name**, and alpha (3727.4 MeV), He3 and triton (2808), deuteron (1875.6) **by mass**. e-, e+,
the muons and the singly charged hadrons pass it and do read a table.

So `em/urban_msc.cuh` has both: `UrbanTable`'s 240-bin log grid for the lepton, because Geant4
builds one, and `urban_heavy_lambda`'s direct evaluation for the ion, because Geant4 does not.
Giving the ion the table would have been a discrepancy dressed up as an optimisation - wrong by
its interpolation error rather than right by its physics - and giving the LEPTON the direct
evaluation would be 4.3's defect over again. The rule is not "tabulate" or "evaluate"; it is
"do what the reference does", and the reference is allowed to do both.

---

---

## 5. Totals

| tree | headers | ported | in QBBC's chain and not ported |
|---|--:|--:|---|
| electromagnetic | 568 | 35 T/V + 9 P | atomic deexcitation (6), G4SynchrotronRadiation, G4EmExtraPhysics' 11 EM classes. **G4CoulombScattering and G4eCoulombScatteringModel are now V/P** (`em/coulomb_scattering.cuh`); what they still need is a stepper hook, not physics. `G4hCoulombScatteringModel` and `G4IonCoulombScatteringModel` came off this list because nothing in 11.1.1 reaches them - see their rows in 1.1. |
| hadronic | 1235 | 3 (1 partial) | effectively all of it |
| decay | 6 | 1 V (`G4Decay`) + 9 V/P channel and product classes from `particles/management` | wiring only: no species carries the process in `stepper.cuh` yet (P8) |
