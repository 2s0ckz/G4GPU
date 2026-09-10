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
| G4eBremsstrahlungRelModel | y | **T** | `em/brems_rel.cuh` |
| G4ModifiedTsai | y | **T** | `em/electron_processes.cuh` |
| G4eeToTwoGammaModel | y | **T** | `em/annihilation.cuh` |
| G4UrbanMscModel | y | **P** | `em/urban_msc.cuh`, `data/urban_msc_tables.cuh`. The cross section is general across mass and charge and exact for all eight species dumped (`tests/test_urban_general.cu`, 41,952 points, 6.7e-16). The *stepping* half is still the electron's: `ComputeTruePathLengthLimit`'s `fUseSafety` branch only, with the lepton `facrange` of 0.04. An ion needs `fMinimal`, `facrange` 0.2, no lateral displacement, and the `mass >= masslimite` path - so alpha and He3 still scatter here by WentzelVI, which is the wrong model for them. |
| G4WentzelVIModel | y | **T** | `em/wentzel_msc.cuh` |
| G4WentzelOKandVIxSection | y | **T** | `em/wentzel_xs.cuh` |
| G4ScreeningMottCrossSection | y | **T** | `data/mott.hh` (the Mott/Rutherford ratio inside WentzelVI) |
| G4BraggModel | y | **T** | `em/bragg.cuh` |
| G4BraggIonModel | y | **T** | `em/bragg.cuh` |
| G4BetheBlochModel | y | **T** | `em/hadron_ionisation.cuh` |
| G4UniversalFluctuation | y | **T** | `em/fluctuation.cuh` |
| G4NuclearStopping / G4ICRU49NuclearStoppingModel | y | **T** | `em/nuclear_stopping.cuh` |
| G4PSTARStopping, G4ASTARStopping, G4NISTStoppingData | y | **T** | `data/nist_stopping.hh` |
| G4SauterGavrilaAngularDistribution | y | **T** | `data/photoelectric_data.cuh` |
| G4MottData, G4SBBremTable | y | **T** | `data/mott.hh`, `data/brems_data.cuh` |
| G4IonFluctuations | y | **T** | `em/ion_fluctuation.cuh`, `data/yang_fluctuation.hh` - the alpha's and every ion's fluctuation model |
| G4Pow (powA / logX / expA) | y | **T** | `data/g4pow.hh` - Geant4's own expansions, not `std::pow` |
| G4ICRU73QOModel | y | **V** | `em/icru73qo.cuh`, and now in the dispatch and the range tables for every negative hadron. Needs pi-/K-/pbar *transport*, which is buffer plumbing rather than physics. |
| **G4CoulombScattering** | **y** | **-** | discrete single scattering, registered next to WentzelVI for every hadron and for e± above 100 MeV |
| **G4eCoulombScatteringModel / G4hCoulombScatteringModel** | **y** | **-** | its models; the shared `G4WentzelOKandVIxSection` engine under them *is* ported |
| G4eIonisation, G4hIonisation, G4ionIonisation | y | **P** | process classes; their `AlongStepDoIt`, model selection, base-particle scaling and per-step effective charge are inlined in `stepper.cuh` and `em/hadron_range.cuh` rather than existing as classes. All three processes' model dispatch is complete: Bragg / BraggIon / ICRU73QO / BetheBloch / MuBetheBloch, chosen by process rather than by charge magnitude. |
| G4eMultipleScattering, G4hMultipleScattering | y | **P** | `G4VMultipleScattering::AlongStepDoIt` is inlined, the class is not. `G4hMultipleScattering` defaults to **Urban**, and `G4EmBuilder::ConstructIonEmPhysics` gives alpha, He3, deuteron, triton and GenericIon a fresh one with no model set - so every ion scatters by Urban in Geant4, and only the light hadrons and muons get WentzelVI. |
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
| G4MuBetheBlochModel | y | **V** | `em/hadron_ionisation.cuh` (`tests/test_muon.cu`) |
| G4MuBremsstrahlungModel | y | **V** | `em/muon_radiative.cuh` |
| G4MuPairProductionModel | y | **V** | `em/muon_radiative.cuh` |
| G4MuIonisation, G4MuBremsstrahlung, G4MuPairProduction, G4MuMultipleScattering | y | **P** | process wrappers. mu± now have their own dE/dx and range tables on Geant4's grid, with the flat 200 keV model boundary and G4MuBetheBlochModel above it; they are still refused at the gun, for want of a track buffer rather than for want of physics. |
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

Nucleon-nucleus total, inelastic and elastic, Z = 2..92, 14 MeV - 1 TeV, protons and neutrons,
agreeing with Geant4 to **2e-15 over 10,738 points** (`tests/test_nucleon_xs.cu`).

`P` and not `T` because it is a *cross section with no process attached to it* - nothing calls
it during transport yet - and because `G4BGGNucleonElasticXS` has three branches this does not
cover, each refused loudly in the file header rather than approximated:

- **Z = 1** -> `G4HadronNucleonXsc::HadronNucleonXscNS`, x 1.0115. A different 436-line
  parameterisation, not a limiting case of this one. Matters most for water.
- **below 14 MeV** -> the Coulomb-barrier form.
- **above 91 GeV** -> Glauber-Gribov.

### 2.2 What QBBC needs and is not there

| QBBC constructor | needs | status |
|---|---|:--:|
| `G4HadronElasticPhysicsXS` | process `G4HadronElasticProcess`; cross sections `G4BGGNucleonElasticXS`, `G4NeutronElasticXS`, `G4BGGPionElasticXS`, `G4ChipsProtonElasticXS`, `G4ComponentGGHadronNucleusXsc`; final states `G4HadronElastic`, `G4ChipsElasticModel`, `G4ElasticHadrNucleusHE`, `G4AntiNuclElastic` | XS partial (above), **final state absent** |
| `G4HadronInelasticQBBC` | `G4HadronInelasticProcess`; `G4ParticleInelasticXS`, `G4BGGPionInelasticXS`, `G4NeutronInelasticXS`; models `G4BinaryCascade`, `G4CascadeInterface` (Bertini), `G4TheoFSGenerator` + `G4FTFModel` + `G4ExcitedStringDecay` + `G4QGSModel`, `G4PreCompoundModel`, `G4GeneratorPrecompoundInterface`, `G4ExcitationHandler` | **none** |
| `G4IonPhysicsXS` | `G4ParticleInelasticXS`, `G4BinaryLightIonReaction` | **none** |
| `G4IonElasticPhysics` | `G4ComponentGGNuclNuclXsc`, `G4NuclNuclDiffuseElastic` | **none** |
| `G4StoppingPhysics` | `G4HadronStoppingProcess`, `G4HadronicAbsorptionBertini`, `G4HadronicAbsorptionFritiof`, `G4MuonMinusCapture`, `G4EmCaptureCascade` | **none** |
| `G4NeutronTrackingCut` | `G4NeutronKiller` | **none** (trivial, but absent) |
| `G4EmExtraPhysics` | gamma-nuclear (`G4GammaNuclearXS`, `G4LowEGammaNuclearModel`, LEND), electro- and muon-nuclear (`G4ElectroVDNuclearModel`, `G4MuonVDNuclearModel`), 18 neutrino classes, `G4SynchrotronRadiation`, `G4AnnihiToMuPair`, `G4GammaConversionToMuons`, `G4eeToHadrons`, `G4MuonToMuonPairProduction` | **none** |

### 2.3 The model tree, by size

Every one of these is at zero. `de_excitation`, `cascade` and `binary_cascade` are the ones
QBBC actually runs for a proton in water; `particle_hp`, `inclxx`, `lend`, `im_r_matrix`,
`qmd`, `abla`, `parton_string` are alternatives other lists select.

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
| G4Decay | y | **V** | `physics/decay/decay.cuh` |
| G4DecayTable (`SelectADecayChannel`, `Insert`'s order) | y | **V** | `physics/decay/decay.cuh`, `physics/decay/decay_tables.hh` |
| G4DecayProducts (`Boost`) | y | **V** | `physics/decay/decay_products.cuh` |
| G4DynamicParticle (`Set4Momentum`, `SetMomentum`, `Get4Momentum`, the mass snap) | y | **V** | `physics/decay/decay_products.cuh` |
| G4VDecayChannel (`IsOKWithParentMass`, `rangeMass`) | y | **P** | `physics/decay/decay_channels.cuh` - `DynamicalMass`'s Breit-Wigner resampling is refused; no daughter of any ported table has a width above 1e-3 of its mass |
| G4PhaseSpaceDecayChannel (1-, 2-, 3- and N-body) | y | **V** | `physics/decay/decay_channels.cuh` |
| G4MuonDecayChannel | y | **V** | `physics/decay/decay_channels.cuh` - the plain channel is what `G4MuonPlus.cc`/`G4MuonMinus.cc` install |
| G4KL3DecayChannel (+ `DalitzDensity`) | y | **V** | `physics/decay/decay_channels.cuh` |
| G4DalitzDecayChannel | y | **V** | `physics/decay/decay_channels.cuh` |
| G4NeutronBetaDecayChannel | y | **V** | `physics/decay/decay_channels.cuh` |
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

Wiring (P8) still owes: a `ParticleType` for each PDG code, the process in the stepper's
at-rest and post-step queues, and the competition above.

---

## 4. Three discrepancies this inventory found

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

---

---

## 5. Totals

| tree | headers | ported | in QBBC's chain and not ported |
|---|--:|--:|---|
| electromagnetic | 568 | 34 T/V + 8 P | G4CoulombScattering (+2 models), atomic deexcitation (6), G4SynchrotronRadiation, G4EmExtraPhysics' 11 EM classes |
| hadronic | 1235 | 3 (1 partial) | effectively all of it |
| decay | 6 | 1 V (`G4Decay`) + 9 V/P channel and product classes from `particles/management` | wiring only: no species carries the process in `stepper.cuh` yet (P8) |
