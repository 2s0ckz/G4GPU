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
| G4UrbanMscModel | y | **P** | `em/urban_msc.cuh`, `data/urban_msc_tables.cuh`. The cross section is general across mass and charge and exact for all eight species dumped (`tests/test_urban_general.cu`, 41,952 points, 6.7e-16). The *stepping* half is still the electron's: `ComputeTruePathLengthLimit`'s `fUseSafety` branch only, with the lepton `facrange` of 0.04. An ion needs `fMinimal`, `facrange` 0.2, no lateral displacement, and the `mass >= masslimite` path - so the species Geant4 scatters by Urban are scattered here by WentzelVI instead. `uses_wentzel_msc` in `core/particle.cuh` is where that substitution is now decided and named: correct for mu±, pi±, K±, p, pbar; a substitution for alpha, He3, GenericIon and - since they gained kernels - deuteron and triton. |
| G4WentzelVIModel | y | **T** | `em/wentzel_msc.cuh` |
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
| **G4CoulombScattering** | **y** | **V** | `em/coulomb_scattering.cuh`. The last EM process in QBBC's chain. Cross section per atom exact against `G4eCoulombScatteringModel::ComputeCrossSectionPerAtom` over 10 species x Z ∈ {1, 6, 8, 13, 26, 82} x 73 energies x 4 materials: **0** for e±, mu±, pi±, p, pbar and 5.5e-15 for K±, with the angular interval to 4.4e-16 and both `MinPrimaryEnergy` forms to 8e-15 (`tests/test_coulomb_scattering.cu`, `ref/oracle/coulomb_xs.csv`). 6098 rows agree on the process being OFF and none disagrees, so the material-dependent threshold is checked too. `V` and not `T` because the discrete process needs a stepper hook and `stepper.cuh` is P7/P8's this phase; the device function and its call-site contract are ready. Two findings in the header: **there is no angular handover between msc and single scattering in option0** - `MscThetaLimit` is π, so both models' `cosThetaMin`/`cosThetaMax` are -1 and they cover the same range, separated only by energy (e±, 100 MeV) or by the table threshold (hadrons) - and the light hadrons get **no** explicit energy limit, so their table starts at `G4CoulombScattering::MinPrimaryEnergy`, which is material-dependent (0.87 MeV for a proton in air against 40.0 MeV for an electron). |
| **G4eCoulombScatteringModel** | **y** | **P** | same file. `ComputeCrossSectionPerAtom` and `MinPrimaryEnergy` exact; `SampleSingleScattering` statistically, 20,000 draws x 240 cells against `ref/oracle/coulomb_sample.csv`, worst chi²/bin 2.87 and every scattered fraction inside 5σ. `P` for one refused sub-case: **`SelectIsotopeNumber`**. `SampleSecondaries` overrides the element's mean atomic mass with the sampled isotope's *nuclear* mass between the cross section and the sampler, and reproducing the draw needs per-element natural abundances - `data/natural_isotopes.hh` carries the nuclide set only, and `data::Material` has no isotope list. `coulomb_sample_secondaries` takes (Z, A, target mass) and `coulomb_refuse_isotope_selection` names what is missing. The element draw is by direct partial cross section rather than `G4EmElementSelector`'s table, as everywhere in this port. |
| G4hCoulombScatteringModel | **n** | - | **The QBBC column was `y` and is wrong.** Nothing in 11.1.1 constructs it: `grep -rn "new G4hCoulombScatteringModel" source/` has no match, only `#include` lines in `G4EmStandardPhysicsWVI.cc` and `G4EmStandardPhysicsSS.cc`. Every species with the process - hadrons included - gets `G4eCoulombScatteringModel`, created inside `G4CoulombScattering::InitialiseProcess`. Dead code. |
| G4IonCoulombScatteringModel / G4IonCoulombCrossSection | **n** | - | reachable only through `G4EmBuilder::ConstructIonEmPhysicsSS`, which option0 never calls; `ConstructIonEmPhysics` (G4EmBuilder.cc:119-145) takes no `isWVI` argument and registers no `G4CoulombScattering` at all. So no ion in QBBC has single Coulomb scattering, and `ref/oracle/coulomb_limits.csv` carries a row per species saying so. |
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
<<<<<<< HEAD
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
=======
| G4NeutronGeneralProcess | **P** | `hadronic/neutron_general_xs.cuh` and `step_neutral` in `physics/stepper.cuh`. The *shape* of the process, not its numbers: `EnableNeutronGeneralProcess` is 1 in 11.1.1, so a neutron has one discrete interaction length over elastic + inelastic + capture summed and picks the sub-process from cumulative partials afterwards, and `G4NeutronTrackingCut::ConstructProcess` returns early so the 10 us cut lives inside it. Ported: the grid `PreparePhysicsTable` builds (400 log bins 1 keV - 20 MeV, 70 more to 100 TeV, **linear** interpolation - the spline flag is `false`), the `G4PhysicsVector::LogVectorValue` lookup, the sub-process choice including the **order swap** either side of 20 MeV, and the time cut with its energy half correctly inert. Not ported: the table's contents (P2) and the final states (P8). The pointer is null, the cross section is zero, and a neutron streams to the world boundary or dies on the clock - which is what a Geant4 neutron does with `NeutronGeneralProc` inactivated. `Upload` refuses a table without final states. |

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

### 2.2 What QBBC needs and is not there

| QBBC constructor | needs | status |
|---|---|:--:|
| `G4HadronElasticPhysicsXS` | process `G4HadronElasticProcess`; cross sections `G4BGGNucleonElasticXS`, `G4NeutronElasticXS`, `G4BGGPionElasticXS`, `G4ChipsProtonElasticXS`, `G4ComponentGGHadronNucleusXsc`; final states `G4HadronElastic`, `G4ChipsElasticModel`, `G4ElasticHadrNucleusHE`, `G4AntiNuclElastic` | XS partial (above), **final state absent** |
| `G4HadronInelasticQBBC` | `G4HadronInelasticProcess`; `G4ParticleInelasticXS`, `G4BGGPionInelasticXS`, `G4NeutronInelasticXS`; models `G4BinaryCascade`, `G4CascadeInterface` (Bertini), `G4TheoFSGenerator` + `G4FTFModel` + `G4ExcitedStringDecay` + `G4QGSModel`, `G4PreCompoundModel`, `G4GeneratorPrecompoundInterface`, `G4ExcitationHandler` | **none** |
| `G4IonPhysicsXS` | `G4ParticleInelasticXS`, `G4BinaryLightIonReaction` | **none** |
| `G4IonElasticPhysics` | `G4ComponentGGNuclNuclXsc`, `G4NuclNuclDiffuseElastic` | **none** |
| `G4StoppingPhysics` | `G4HadronStoppingProcess`, `G4HadronicAbsorptionBertini`, `G4HadronicAbsorptionFritiof`, `G4MuonMinusCapture`, `G4EmCaptureCascade` | **none** |
| `G4NeutronTrackingCut` | `G4NeutronKiller` | **P** - and the class is not the answer. With `EnableNeutronGeneralProcess = 1` this constructor `return`s without creating a `G4NeutronKiller`; the cut is the two lines at the top of `G4NeutronGeneralProcess::PostStepGetPhysicalInteractionLength`. Ported in `step_neutral`, before geometry and before the cross section, and it **deposits nothing** - `theTotalResult->Initialize(track)` zeroes both energy deposits, so the neutron's kinetic energy is discarded rather than given to the volume. The engine books it (`RunStats::neutron_killed_energy`) because Geant4 does not conserve energy across this either and the only way anyone finds out is a printed number. |
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

**That moment has arrived.** pi±, K±, mu±, the triton and the neutron all have stepping kernels
now and all five are unstable (`ref/oracle/species_tables.csv` carries each one's lifetime and
`stable` flag; `species_processes.csv` shows the `Decay` process on each). In this port they
stop and stay stopped. It is loud rather than silent - `QBBC.hh`'s banner says "no decay" for
every species it lists, and the `ProcessId::fDecay` slot has been reserved since before any of
them was transported - but it is a real difference from Geant4 for any run that stops one of
them, and it is the largest of the gaps P1 leaves behind. P4 is the package; P8 wires it.

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
| electromagnetic | 568 | 35 T/V + 9 P | atomic deexcitation (6), G4SynchrotronRadiation, G4EmExtraPhysics' 11 EM classes. **G4CoulombScattering and G4eCoulombScatteringModel are now V/P** (`em/coulomb_scattering.cuh`); what they still need is a stepper hook, not physics. `G4hCoulombScatteringModel` and `G4IonCoulombScatteringModel` came off this list because nothing in 11.1.1 reaches them - see their rows in 1.1. |
| hadronic | 1235 | 3 (1 partial) | effectively all of it |
| decay | 6 | 1 V (`G4Decay`) + 9 V/P channel and product classes from `particles/management` | wiring only: no species carries the process in `stepper.cuh` yet (P8) |
