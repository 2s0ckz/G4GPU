# Porting the rest of QBBC: hadronic, decay, and the last EM items

The plan for finishing the physics. It is written for several people working at once, so most
of it is about boundaries: what each package owns, what it may not touch, how it proves itself,
and how the pieces come back together. Read `docs/PORTED.md` first for what exists; this file
says what comes next and in what order.

**The target is Geant4 11.1.1's `QBBC`, exactly** - the same list example B1 runs, the same
list the oracle links, and the list every existing dose comparison is against. Not a better
physics list, not a space-radiation list; a faithful one, so that every number here can be
checked against a number Geant4 produces. `tools/g4src.sh` names the source tree to transcribe
from and refuses any other.

---

## 1. Why this order

`docs/RESULT.md` puts a figure on what is missing. With QBBC's hadronic processes switched off,
Geant4 and this port agree on a 210 MeV proton's dose to 0.15% and an 840 MeV alpha's to 0.32%.
With them on, Geant4's answer is **19% lower for the proton and 33% lower for the alpha**.
Inelastic reactions remove primaries before they reach the scoring volume and replace them with
neutrons and fragments that deposit somewhere else. That gap is the whole reason for this work,
and it orders it:

1. **Nucleon inelastic + neutron transport.** The dominant term in the gap. Protons and
   neutrons at 0 - 6 GeV are where QBBC spends BIC and Bertini, and the secondary neutrons are
   what carries dose deep into a phantom and out through a shield. Neutrons do not exist in
   this port yet, as a species or as physics.
2. **Ion inelastic.** For galactic cosmic rays it is projectile fragmentation that decides what
   arrives behind a shield - it is the reason a shield is worth designing. QBBC runs
   `G4BinaryLightIonReaction` here, below 6 GeV/n, and FTFP above.
3. **Everything that closes the shower**: pion/kaon/muon transport and decay, hadron elastic
   scattering, absorption at rest, neutron capture, the neutron time cut, gamma-nuclear.
   Individually small; together the difference between a physics list and a collection of
   processes.

Two facts about the code shape it further. Every inelastic model in QBBC - Binary, Bertini,
FTFP, the light-ion reaction, even low-energy gamma-nuclear - ends by handing an excited residual
nucleus to **the same de-excitation chain** (`G4ExcitationHandler`), most of them through
`G4PreCompoundModel`. So de-excitation is not one model among many; it is the shared dependency
of all of them and it goes first. And every hadronic process, elastic or inelastic, goes through
one **process framework** - cross section per element from data, target selection, model
selection by energy with a random choice in the overlaps, final-state application - which is
written once and then reused.

---

## 2. The target, constructor by constructor

Read from `QBBC.cc` and the constructors it registers, in 11.1.1. Transition energies are
`G4HadronicParameters` defaults: FTF from **3 GeV**, cascade-to-FTF handover complete at
**6 GeV**, heavy-hadron threshold 1.1 GeV, `ApplyFactorXS` off. Default range cut 0.7 mm.

| QBBC constructor | particle | cross section | final state (energy range) |
|---|---|---|---|
| `G4HadronElasticPhysicsXS` | p | `G4BGGNucleonElasticXS` | `G4ChipsElasticModel` |
| | n | `G4NeutronElasticXS` | `G4ChipsElasticModel` |
| | pi± | `G4BGGPionElasticXS` | `G4ElasticHadrNucleusHE` |
| | K± | `G4HadronicBuilder::BuildElastic` (agent reads) | |
| | d, t, He3, alpha | `G4ComponentGGNuclNuclXsc` | `G4HadronElastic` |
| `G4IonElasticPhysics` | GenericIon | `G4ComponentGGNuclNuclXsc` | `G4NuclNuclDiffuseElastic` |
| `G4HadronInelasticQBBC` | p | `G4ParticleInelasticXS` | BIC 0-1.5 GeV, BERT 1-6 GeV, FTFP 3 GeV+ |
| | n | `G4NeutronInelasticXS`; capture `G4NeutronCaptureXS` + `G4NeutronRadCapture` - **inside `G4NeutronGeneralProcess` with the elastic**, see below | same three |
| | pi± | `G4BGGPionInelasticXS` | BIC 0-1.5, BERT 1-12 GeV, FTFP 3 GeV+ |
| | K± | `G4HadronicBuilder::BuildKaonsFTFP_BERT` | |
| `G4IonPhysicsXS` | d, t, He3, alpha | `G4ParticleInelasticXS` | `G4BinaryLightIonReaction` 0-6 GeV, FTFP 3 GeV+ |
| | GenericIon | `G4ComponentGGNuclNuclXsc` | same |
| `G4StoppingPhysics` | pi-, K-, Sigma-, Xi-, Omega- (at rest) | - | `G4HadronicAbsorptionBertini` |
| | pbar, nbar, anti-hyperons, anti-nuclei (at rest) | - | `G4HadronicAbsorptionFritiof` |
| | mu- (at rest) | - | `G4MuonMinusCapture` (agent confirms the default flag) |
| `G4NeutronTrackingCut` | n | - | `G4NeutronKiller`: 10 us, 0 MeV |
| `G4DecayPhysics` | every unstable particle | - | `G4Decay` + each particle's decay table |
| `G4EmExtraPhysics` | gamma | `G4GammaNuclearXS` | `G4LowEGammaNuclearModel` (<200 MeV, via PreCompound), Bertini above |
| | e± | | `G4ElectroVDNuclearModel` |
| | mu± | | `G4MuonVDNuclearModel` |
| | (off by default) | | synchrotron, gamma->mumu, e+e-->mumu/hadrons, neutrinos |
| `G4EmStandardPhysics` | hadrons, e± >100 MeV | | **`G4CoulombScattering`** - the one EM process in the chain still unported |

Where a cascade's range overlaps the next model's - BIC to 1.5 GeV and BERT from 1 GeV, BERT to
6 GeV and FTFP from 3 GeV - Geant4 picks **at random with a probability linear across the
overlap** (`G4EnergyRangeManager::GetHadronicInteraction`). That is part of the physics list and
has to be reproduced, not resolved.

**The neutron's three processes are one process.** `ref/oracle/hadronic_params.csv` - the first
dump through the new registry - says `EnableNeutronGeneralProcess = 1` in 11.1.1. So
`G4HadProcesses::BuildNeutronElastic` and `BuildNeutronInelasticAndCapture` take the
`useNeutronGeneral` branch: elastic, inelastic and capture are sub-processes of a single
`G4NeutronGeneralProcess`, which builds a combined cross-section table per material and samples
which sub-process fired from the partials - and `G4NeutronTrackingCut::ConstructProcess` returns
early when that process exists, so the time and energy cut live inside it. Whoever wires the
neutron (P1, P8) reads `G4NeutronGeneralProcess.cc` first, and P2's neutron cross sections have to
come out on **that process's table grid**, not the raw data set's - the V5 lesson (section 8)
applies before a single number is compared. The same file records `EnableBCParticles = 1`: b- and
c-hadrons get FTFP_BERT in QBBC. They are reachable only through FTF at many GeV; P11 refuses
them by name until it does not.

**Data on this machine, and what reads it.** `G4PARTICLEXS4.0` (per-element and per-isotope
files: `proton/inel<Z>`, `neutron/{el,inel,cap}<Z>`, `<Z>_<A>` isotopes, `alpha/`, `gamma/`),
`G4ENSDFSTATE2.3` and `PhotonEvaporation5.7` (nuclear levels, 3110 files - de-excitation),
`G4SAIDDATA2.0` (hadron-nucleon totals). `G4NDL`, `G4ABLA`, `G4INCL` and `RadioactiveDecay` are
present but **not in QBBC's chain** and are out of scope. Every dataset is resolved through
`src/host/g4data.cuh`, whose rule is that missing data is fatal, never a silent zero.

---

## 3. What exists, in four facts

`docs/PORTED.md` has the class-by-class inventory. The facts that shape the packages:

1. **The neutron is not a species.** `core/particle.cuh` has fourteen types and none is neutral
   and heavy. Nor are pi0, deuteron or triton. Nothing hadronic can be transported until it is.
2. **mu±, pi±, K±, pbar have validated physics and no transport.** dE/dx, deltas, radiative
   losses and MSC are transcribed and checked (`V` in PORTED.md); what is missing is a range
   table, a buffer and a kernel instantiation - "buffer plumbing rather than physics".
3. **One hadronic cross section exists and nothing calls it.** Barashenkov nucleon-nucleus,
   Z = 2..92, 14 MeV - 91 GeV, exact to 2e-15. Z = 1, below 14 MeV and above 91 GeV are refused
   in the file header, not approximated.
4. **The process ids are already reserved.** `core/step_report.cuh` names `fHadronElastic`,
   `fHadronInelastic`, `fNeutronCapture`, `fDecay`, `fPhotoNuclear`, `fCoulombScattering` and the
   rest, so a new process reports what it did on the day it is written. `fNotDefined` is
   load-bearing - `g4dose -verify-step-hook` fails the build on it.

---

## 4. How every stage is checked

Three disciplines, and they are the same ones the EM port used.

**Deterministic functions get an exact oracle.** `ref/dump/` links the real Geant4 and writes
CSVs into `ref/oracle/`; a test in `tests/` reads the CSV and compares to a tolerance near
machine precision. Cross sections, level densities, Coulomb barriers, emission widths, branching
ratios, kinematic limits, the parameter tables inside a sampler - all deterministic, all dumped
exactly. **A dump goes in its own file**, `ref/dump/dump_<package>.cc`, registered with
`G4GPU_REGISTER_DUMP`; `CMakeLists.txt` globs the directory and `main` runs whatever registered,
so adding a package touches no shared file. See `ref/dump/dump_registry.hh`.

**Samplers get a statistical oracle, from the same class.** For a final-state model there is no
formula to compare against, but the model itself can be run: construct the `G4HadProjectile`
and `G4Nucleus`, call `ApplyYourself` N times under a fixed seed, and dump moments and histograms
of what comes out - multiplicities by species, energy spectra, angular moments, the residual's
(Z, A, E*). The port samples the same inputs N times and the comparison has a statistical
tolerance it states. `test_urban_msc` and `test_ion_fluctuation` are the pattern.

**Transport gets a like-for-like Geant4 run at every stage.** Geant4 can switch any process off
by UI command - `/process/inactivate protonInelastic` and so on - which is exactly how
`docs/RESULT.md` isolated the EM agreement. So each phase has an oracle that matches what has
been ported so far: after elastic and decay are wired, run QBBC with inelastic and capture
inactivated and compare doses; after neutrons transport, switch their processes on. Nothing is
ever compared against a Geant4 that is running physics the port does not have.

And **anti-vacuity, every time**: each new assertion is run once with its fix removed or its
input perturbed, and the commit message says what failed. `docs/RISK.md` V29 and V32 are what
happens otherwise.

---

## 5. The packages

Each names what it owns, the Geant4 sources, the deliverables, the oracle, and what it refuses.
A package may finish with `P` (partial) status in PORTED.md - a named sub-case done and the rest
refused loudly - but never with an approximation.

### Phase 1 - independent foundations, in parallel

#### P1 Species and transport plumbing

The one package that edits the transport core, and the one every other package eventually
needs to be *reached* by a particle.

- **Owns**: `src/core/particle.cuh`, `src/core/track_buffer.cuh`, `src/physics/stepper.cuh`,
  `src/host/transport_run*.cuh`, `src/g4/G4RunManager.hh` (`CheckSpecies`), `src/g4/QBBC.hh`,
  the particle definitions under `src/g4/`.
- **Deliverables**
  - New species: `kNeutron`, `kPiZero`, `kDeuteron`, `kTriton`. Their `ParticleDef` rows from
    Geant4's PDG table, checked against `ref/oracle/hadron_tables.csv` as the others are.
  - Transport for the charged species whose physics is already validated: mu±, pi±, K±, pbar,
    d, t (the last two by `G4hIonisation`'s scaling, which `hadron_range.cuh` already does for
    GenericIon). Range tables on Geant4's grid, buffers, kernel instantiations, the gun
    accepting them. The muon range table that landed in PORTED.md 4.3 is the template.
  - A **neutral-hadron step**, `step_neutral`: no continuous loss, no MSC; the step is geometry
    against a discrete-process interaction length, with the `G4NeutronKiller` time and energy
    cut applied first. Until P8 wires a process in, its cross section is zero and a neutron
    streams to the world boundary - which is what a neutron does in Geant4 with hadronics off.
  - **Neutrinos**: a species that is created, counted per event as energy carried away, and
    never stepped. Not dropped - counted. QBBC does not transport them either.
  - `SecondaryArena` sized and reported for hadronic multiplicities: an inelastic event at a
    few GeV emits tens of secondaries in one step. Overflow is already never silent; confirm
    the capacity default and the report.
- **Oracle**: run QBBC with `Decay`, `hadElastic`, `*Inelastic`, `nCapture` inactivated and
  compare mu±, pi±, K± depth-dose and range in water against the port - the RESULT.md method,
  the EM-only column. Species tables exact against `hadron_tables.csv`.
- **Refuses**: hyperons, K0L/K0S, anti-nuclei - emitted by cascades later; a stub that counts
  and reports them at emission, fatal if a primary. Not a silent `default:`.
- **Hazard**: `run_step_hadron` is at 255 registers with spill; a fourth kernel instantiation
  per species is fine, a wider `TrackState` is not. Measure `-Xptxas -v` before and after.

#### P2 Hadronic cross sections and the G4PARTICLEXS reader

Everything QBBC's hadronic chain asks for a number per element per energy. Pure transcription
plus one dataset reader; nothing here samples anything.

- **Owns**: `src/physics/hadronic/xs/` (new), `src/data/` additions, `src/host/g4data.cuh`
  extension for `G4PARTICLEXSDATA`.
- **Deliverables**, each with its Geant4 source named at the top of the file:
  - The `G4PARTICLEXS` reader: `G4ParticleInelasticXS`, `G4NeutronInelasticXS`,
    `G4NeutronElasticXS`, `G4NeutronCaptureXS`, `G4GammaNuclearXS` - per-element vectors,
    per-isotope where the file exists, the high-energy hand-over each makes above its table
    (to Glauber-Gribov or Barashenkov), the isotope selection each does.
  - `G4BGGNucleonElasticXS` / `G4BGGNucleonInelasticXS` **completed**: the three refused
    branches of `barashenkov_xs.cuh` - Z = 1 through `G4HadronNucleonXsc::HadronNucleonXscNS`
    with its 1.0115, below 14 MeV Coulomb barrier, above 91 GeV `G4ComponentGGHadronNucleusXsc`.
  - `G4BGGPionElasticXS` / `G4BGGPionInelasticXS` and `G4UPiNuclearCrossSection` under them.
  - `G4ComponentGGNuclNuclXsc` (ion-ion), `G4ComponentGGHadronNucleusXsc` (hadron-nucleus GG),
    `G4ComponentAntiNuclNuclearXS` if time allows (pbar).
  - `G4ComponentSAIDTotalXS` and the `G4SAIDDATA` reader if any of the above reaches it.
  - The dataset-side of target selection: `G4CrossSectionDataStore::SampleZandA` semantics -
    element by macroscopic partial XS, isotope by abundance times isotope XS.
- **Oracle**: `dump_hadronic_xs.cc` - every Z = 1..92, twelve points per decade over each
  dataset's stated range, for each particle; compare exactly. For the isotope branch, every
  isotope Geant4's element list carries.
- **Refuses**: nothing in scope should need refusing; a dataset file that is missing is fatal
  through `g4data.cuh`, as everywhere.

#### P3 De-excitation

The shared tail of every inelastic model, and the largest Phase-1 package. It takes an excited
fragment (Z, A, E*, momentum) and returns fragments and gammas.

- **Owns**: `src/physics/hadronic/deexcitation/` (new), nuclear-level data readers in
  `src/data/` and `src/host/g4data.cuh` for `G4ENSDFSTATEDATA` / `G4LEVELGAMMADATA`.
- **Deliverables**, in the order Geant4 depends on them:
  - Nuclear properties: `G4NucleiProperties` binding energies and masses (check what
    `data/atomic_masses.cuh` already carries), `G4PairingCorrection`, `G4ShellCorrection`,
    `G4CameronTruranHilfShellCorrections` and friends, level-density (`G4EvaporationLevelDensityParameter`).
  - `G4NuclearLevelData` / `G4LevelReader` / `G4LevelManager`: the ENSDFSTATE and
    PhotonEvaporation readers, exact.
  - `G4ExcitationHandler::BreakItUp` and the models it dispatches to with 11.1.1's defaults:
    `G4Evaporation` with its default channel set, `G4FermiBreakUpVI` and its fragment pool,
    `G4PhotonEvaporation` (discrete and continuum), `G4UnstableFragmentBreakUp`,
    `G4CompetitiveFission`. `G4StatMF` only if the default flags reach it (read the handler).
- **Oracle**: `dump_deexcitation.cc`. Deterministic: masses, corrections, level data, emission
  probabilities and Coulomb barriers for a grid of (Z, A, E*). Statistical: `BreakItUp` on a
  fixed set of fragments (light, medium, heavy; E* from 1 to 200 MeV) N times under a fixed
  seed - multiplicity by species, energy spectra, residual (Z, A) distribution.
- **Refuses**: whatever channel is not finished, by name, with the handler reporting it rather
  than skipping it silently.

#### P4 Decay

- **Owns**: `src/physics/decay/` (new), decay tables under `src/g4/` particle definitions.
- **Deliverables**: `G4Decay` - mean life, the at-rest and in-flight interaction length, the
  boost; `G4DecayTable` channel selection; the channels QBBC's transported species use:
  `G4PhaseSpaceDecayChannel` (2- and 3-body), `G4MuonDecayChannel` (or the spin variant if the
  definition selects it - read `G4MuonMinus.cc`), `G4KL3DecayChannel`, `G4DalitzDecayChannel`
  (pi0), `G4PionRadiativeDecayChannel` if in the table. Species: pi±, pi0, mu±, K±. K0L/K0S and
  hyperons are P1's refused set and follow it.
- **Oracle**: `dump_decay.cc`. Deterministic: lifetimes, branching ratios, the kinematic
  limits per channel. Statistical: product energy spectra and angular moments for each channel
  N times, fixed seed.
- **Interaction**: a stopped pi-, K- or mu- is captured, not decayed, and not by winning a
  competition. `G4HadronStoppingProcess::AtRestGetPhysicalInteractionLength` returns 0.0 -
  `G4HadronicAbsorptionBertini` and `G4MuonMinusCapture` both derive from it - while `G4Decay`'s
  at-rest length is a sampled lifetime, and the stepping manager invokes the at-rest process with
  the shortest lifetime. The stopping process pre-empts the decay outright; the negative muon's
  bound decay happens inside `G4MuonMinusBoundDecay`, P12's model, not in `G4Decay`. So P4's
  at-rest branch is reached only by species without a stopping process: pi+, K+, mu+. (Corrected
  2026-09-10 from P4's measurement; the plan had described the two as competing.)

#### P5 The hadronic process framework, and elastic final states

The simplest complete hadronic process, used to write the framework every other one reuses.

- **Owns**: `src/physics/hadronic/process.cuh` (new), `src/physics/hadronic/elastic/` (new).
- **Deliverables**
  - `G4HadronicProcess::PostStepDoIt` semantics as device code: element and isotope selection
    from P2's partial cross sections (`SampleZandA`), `G4EnergyRangeManager` model selection
    including the random choice in an overlap, `G4HadProjectile` / `G4Nucleus` construction,
    `G4HadFinalState` application (`FillResult`: secondaries, the primary's new state or its
    death, local energy deposit), the energy-momentum check as Geant4 configures it.
  - Elastic final states: `G4HadronElastic` (the generic `SampleInvariantT`),
    `G4ChipsElasticModel` and the Chips t-distribution parameterisations it carries,
    `G4ElasticHadrNucleusHE`, `G4NuclNuclDiffuseElastic` (with its table build),
    `G4AntiNuclElastic` last.
- **Oracle**: `dump_elastic.cc`. Deterministic: every parameter table, and the recoil
  kinematics for a given t. Statistical: `SampleInvariantT` N times per (particle, Z, A, E)
  under a fixed seed - the distribution of t and of the scattering angle, moments and
  histogram.
- **Depends on P2 for the numbers** but not for the code: the framework takes a cross section
  functor, and the tests can use Barashenkov, which exists.

### Phase 2 - the layer that needs Phase 1

- **P6 PreCompound**: `G4PreCompoundModel`, its emission channels, `G4GNASHTransitions`,
  `G4PreCompoundEmission`. Exit is P3's handler. Also `G4GeneratorPrecompoundInterface`, which
  FTFP uses to hand its residual over. Statistical oracle on fixed (Z, A, E*, exciton) inputs.
- **P7 Neutron capture**: `G4NeutronRadCapture` -> P3's photon evaporation, with P2's
  `G4NeutronCaptureXS`.
- **P8 First wiring**: P1 + P2 + P4 + P5 + P7 into the steppers - hadron elastic for every
  charged hadron and the neutron, capture and the killer for the neutron, decay for every
  unstable species. Then the first staged like-for-like: QBBC with only `*Inelastic`
  inactivated. This is the first commit in which Geant4's answer moves and the port follows.
- **P14 `G4CoulombScattering`**: the last EM process in QBBC's chain. Small, independent,
  reuses `wentzel_xs.cuh`; a Phase-1 agent that finishes early takes it.

### Phase 3 - the cascades and the string model

Each is large, each is independent of the others, each ends in P6/P3.

- **P9 Binary cascade**: `G4BinaryCascade` (p, n, pi to 1.5 GeV) and
  `G4BinaryLightIonReaction` (every ion QBBC transports, to 6 GeV/n). The ion reaction is the
  one that matters most for the user's problem and it is the harder of the two.
- **P10 Bertini**: `G4CascadeInterface` and the INUCL tree under it, 1 - 6 GeV for nucleons,
  to 12 GeV for pions, and `G4HadronicAbsorptionBertini` for stopping (P12).
- **P11 FTFP**: `G4FTFModel`, `G4ExcitedStringDecay` (Lund fragmentation), the
  parton-string common code, `G4TheoFSGenerator`, above 3 GeV. And `G4HadronicAbsorptionFritiof`.
- **P12 Stopping**: the at-rest processes, once P10/P11 exist.
- **P13 Gamma/electro/muon-nuclear**: `G4GammaNuclearXS` (P2), `G4LowEGammaNuclearModel` ->
  P6, Bertini above 200 MeV (P10), `G4ElectroVDNuclearModel`, `G4MuonVDNuclearModel`.
- **P15 Second and third wiring**, each followed by a like-for-like run with fewer processes
  inactivated, until nothing is.

```
P1 species/transport ──┐
P2 cross sections ─────┼──> P8 wiring (elastic, capture, decay, killer) ──> like-for-like #1
P4 decay ──────────────┤
P5 framework+elastic ──┘
P3 de-excitation ──> P6 precompound ──┬──> P9 BIC + light-ion ──┐
                 └──> P7 n-capture     ├──> P10 Bertini ─────────┼──> P15 wiring ──> like-for-like #2, #3
                                       └──> P11 FTFP ────────────┘
                                                   └──> P12 stopping, P13 gamma-nuclear
```

---

## 6. Working rules for a package

The point of every rule is that five people can work at once and the result can be merged.

1. **Branch and worktree.** Each package works in its own git worktree on `phys/<package>`,
   branched from `main`. Commit there in the house style (a title that is a finding, then the
   body; `Co-Authored-By` line). Do not push; the lead integrates.
2. **Do not run `build_all.bat`.** It takes 33 minutes when nothing changed and about two hours when the engine rebuilds (and `build_extras.bat` over two more), it uses the one GPU, and it writes
   its logs to fixed `%TEMP%\g4gpu_*.txt` names, so two at once corrupt each other. A package's
   tests are host-only translation units: build them with `build_one_test.bat <name>` or
   `nvcc -std=c++17 -O2 -I src -I src/g4 -o tests/<name>.exe tests/<name>.cu`, run them, and run
   the existing tests your change could affect (`tools/quick.ps1 <pattern>`). The lead runs
   `build_all.bat` once per merge, serially. Nothing lands on `main` without it green.
3. **Read from the right Geant4.** `G4SRC=$(sh tools/g4src.sh)`. Every transcribed function
   names its Geant4 file and function in a comment. The data directories are the ones
   `ref/oracle/run.bat` sets.
4. **No approximations, no surrogates.** A sub-case that is not done is refused with a message
   that names it, at the point it would have been needed, and appears in PORTED.md as `P`.
   General across particle, energy and material: nothing hard-coded to a species or a Z.
5. **Shared files are append-only, and only these**: your `TESTS` entry in `build_all.bat` (or in `build_extras.bat`, for a viewer, GUI or hook-project test);
   your rows in `docs/PORTED.md`; your `RISK.md` entry if you found something worth one. New
   code goes in the directories the package owns. Only P1 edits `stepper.cuh`,
   `transport_run*.cuh`, `particle.cuh`, `track_buffer.cuh`.
6. **Oracle dumps go in `ref/dump/dump_<package>.cc`**, registered with
   `G4GPU_REGISTER_DUMP`. Rebuild with `ref/dump/build.bat`, run with `ref/oracle/run.bat`,
   commit the CSV. Fixed seeds for anything sampled.
7. **Every assertion runs once with its fix removed**, and the commit message records what
   failed. A check that cannot fail is not a check.
8. **Report** at the end: branch and worktree path; what is `T`/`V`/`P` and what is refused;
   the numbers (points compared, worst relative error, statistical tolerances met); files
   changed; anything found in Geant4 or in this port worth a RISK.md entry.

---

## 7. Integration

Serial, by the lead, one branch at a time: merge (or cherry-pick, to keep `main` linear as it
has always been), resolve the append-only conflicts, `build_all.bat` (and `build_extras.bat` when the change touches what it covers), commit to `main`. The
`%TEMP%` collision in `build_all.bat` is worth fixing before two integrations ever run at once -
a per-invocation suffix on the log names - but it is not needed while integration is serial.

The like-for-like runs of section 4 are the integration test that matters, and they need a
Geant4 macro per stage that inactivates exactly the processes not yet ported. Those macros live
in `ref/b1hadron/` beside the comparison scripts, one per stage, so that the stage a number was
measured at is recorded with the number.

---

## 8. What to expect to find

Three kinds of thing the EM port found repeatedly, listed so that finding them again costs a
minute and not a day (`docs/RISK.md` has the histories):

- **Geant4 does not run the model; it runs a table built from the model.** Cross sections
  are `G4PhysicsVector`s on a log grid with a spline, and the transport reads the table. Match
  the grid and the spline, or the numbers will be right and the transport wrong (V5).
- **`G4Pow`, not `std::pow`.** `powA`, `A13`, `logX`, `expA` are expansions about tabulated
  points and differ from the exact function at 1e-7 - a thousand times the tolerance the
  oracle is checked at. `data/g4pow.hh` has them.
- **A comment that explains why information is discarded** is the one to re-read when
  something needs that information (V35). And a comment that argues for a weaker claim than the
  code makes is protecting a gap (V32, V35).

---

## 9. P11d - DecayStrongResonances, and the decay engine it shares with P9

Written by P11c after the FTFP entry point was finished, so that **P9d builds the engine once,
for both callers.** Everything below is measured, not remembered: the species list and its
fractions come from 14,000 `ftf::apply_yourself` events over seven beams, and every decay table
was read out of `G4ParticleTable` in a running Geant4 through `ref/dump/dump_ftf.cc`.

### 9.1 Why one module, and where it goes

FTFP is blocked on exactly one thing. Its 96-point generality sweep - `apply_yourself` at three
energies inside every FTFP row of `ref/oracle/ftf_windows.csv`, on carbon and on lead - runs 44
points and refuses 52, **and all 52 are the same refusal**:
`preco::GeneratorRefusal::short_lived_track`, which is P6's by-name refusal of
`G4DecayKineticTracks` (docs/RISK.md V100). It also costs 80% of ion events and 32% of sub-GeV
anti-nucleon ones. The Binary cascade reaches the same code through the same door, and its
cascade makes the same Deltas and N*, so the engine belongs under `bic/` behind a contract
header in the shape of `bic/nucleus/nucleus_model.cuh`: plain structs in, plain structs out, no
knowledge of either caller.

### 9.2 The two entry points, and they are the same three lines

```
G4DecayStrongResonances::Propagate(secondaries, nucleus)     [hadronic/util]
    -> G4DecayKineticTracks decay(secondaries)               // changes the list IN SITU
    -> then copies every surviving track into a G4ReactionProduct and returns

G4GeneratorPrecompoundInterface::Propagate(secondaries, nucleus)          [binary_cascade]
G4GeneratorPrecompoundInterface::PropagateNuclNucl(secondaries, t, p)
    -> G4DecayKineticTracks decay(secondaries)               // the FIRST line of both
    -> then the capture loop and the residual
```

`G4DecayStrongResonances::Propagate` is `G4DecayKineticTracks` plus a type conversion and
nothing else - 20 lines, no physics of its own. So the engine is:

```
G4DecayKineticTracks::Decay(G4KineticTrackVector*)           [hadronic/util]
    for i over a list that GROWS as it goes:
      if (track->GetDefinition()->IsShortLived()) daughters = track->Decay();
      append daughters, delete the parent, null its slot
    then erase the null slots
    - the growing loop is the recursion: a daughter that is itself short-lived is decayed too
    - each daughter inherits the parent's CreatorModelID, ParentResonanceDef, and a
      ParentResonanceID that is `round(parent 4-momentum .mag() / keV)`

G4KineticTrack::Decay()                                      [hadronic/util]
    1. sample a channel from theActualWidth[] (see 9.4), repeat up to 10,000 times until the
       daughter masses fit under the parent's ACTUAL mass
    2. for a channel with 1 or 2 short-lived daughters, draw those daughters' masses with
       G4SampleResonance::SampleMass - P11 already has this as ftf_sample_resonance_mass
    3. G4GeneralPhaseSpaceDecay(parentName, parentMass, BR, nDaughters, names, masses).DecayIt()
    4. boost every product by the parent's boostVector, keep position, formationTime = 0
```

### 9.3 What FTFP actually produces - 27 species, and the shape of the problem

14,000 events over {p on C at 10 GeV, p on Pb at 50 GeV, pi+ on Al at 10 GeV, pbar on C at 5
GeV, pbar on C at 50 MeV, C12 on C at 8 GeV/n, Fe56 on Pb at 8 GeV/n}: **312,632 tracks, 44,759
of them short-lived (14.3%), and 72.6% of events carry at least one.** That last number is why
this is not an edge case.

| PDG | name | tracks | share | mass (MeV) | width (MeV) | channels |
|---:|---|---:|---:|---:|---:|---:|
| 2114 | delta0 | 8797 | 19.7% | 1232.0 | 120.0 | 3 |
| 2214 | delta+ | 8013 | 17.9% | 1232.0 | 120.0 | 3 |
| 1114 | delta- | 6625 | 14.8% | 1232.0 | 117.0 | 1 |
| 2224 | delta++ | 5337 | 11.9% | 1232.0 | 120.0 | 1 |
| 113 | rho0 | 3615 | 8.1% | 775.26 | 149.1 | 1 |
| 223 | omega | 3565 | 8.0% | 782.65 | 8.49 | 3 |
| -213 | rho- | 3515 | 7.9% | 775.8 | 150.3 | 1 |
| 213 | rho+ | 3499 | 7.8% | 775.8 | 150.3 | 1 |
| 323 | k_star+ | 364 | 0.8% | 891.76 | 50.3 | 2 |
| 313 | k_star0 | 331 | 0.7% | 895.55 | 47.3 | 2 |
| 3214 | sigma(1385)0 | 280 | 0.6% | 1383.7 | 36.0 | 3 |
| -313 | anti_k_star0 | 163 | 0.4% | 895.55 | 47.3 | 2 |
| -323 | k_star- | 135 | 0.3% | 891.76 | 50.3 | 2 |
| -2214 | anti_delta+ | 113 | 0.3% | 1232.0 | 120.0 | 2 |
| -2114 | anti_delta0 | 83 | 0.2% | 1232.0 | 120.0 | 2 |
| -2224 | anti_delta++ | 68 | 0.2% | 1232.0 | 120.0 | 1 |
| 3224 | sigma(1385)+ | 68 | 0.2% | 1382.8 | 36.0 | 3 |
| 3114 | sigma(1385)- | 51 | 0.1% | 1387.2 | 39.4 | 3 |
| 333 | phi | 32 | 0.07% | 1019.46 | 4.247 | 3 |
| 225 | f2(1270) | 30 | 0.07% | 1275.5 | 186.7 | 6 |
| 115 | a2(1320)0 | 26 | 0.06% | 1318.3 | 107.0 | 7 |
| -3214 | anti_sigma(1385)0 | 24 | 0.05% | 1383.7 | 36.0 | 3 |
| -3224 | anti_sigma(1385)+ | 14 | 0.03% | 1382.8 | 36.0 | 3 |
| -1114 | anti_delta- | 6 | 0.01% | 1232.0 | 117.0 | 1 |
| -3114 | anti_sigma(1385)- | 3 | 0.01% | 1387.2 | 39.4 | 3 |
| 3314 | xi(1530)- | 1 | 0.002% | 1535.0 | 9.9 | 3 |
| 3324 | xi(1530)0 | 1 | 0.002% | 1531.8 | 9.1 | 3 |

**Four facts that size the job**, all of them from the tables and not from the list above:

1. **68 channels in all, and EVERY ONE of them is `Phase Space`** - `G4PhaseSpaceDecayChannel`,
   `GetKinematicsName() == "Phase Space"`. No Dalitz, no KL3, no muon-decay form; the decay
   forms P4 ported for the long-lived species are not needed here.
2. **63 of the 68 are two-body**, 3 are three-body (omega -> pi+ pi- pi0, and two of a2(1320)0's)
   and **2 are FOUR-body** (f2(1270) -> 4pi). `G4GeneralPhaseSpaceDecay::DecayIt` therefore
   needs its two-body closed form, its three-body form, and the N-body form for N = 4.
3. **Only FIVE channels of the 68 have a SHORT-LIVED daughter**: a2(1320)0's pi rho (x2) and
   omega pi pi (x2), and phi -> rho0 pi0. Those five are the only ones that need
   `G4SampleResonance::SampleMass` at decay time, and the only ones whose `theActualWidth[]`
   needs `IntegrateCMMomentum` (see 9.4). They belong to two species that are together 0.13% of
   the short-lived tracks, so a first cut that REFUSES them by name and covers the other 63
   channels reaches 99.87% of what FTFP makes.
4. Every daughter of the other 63 is stable or long-lived: nucleons, pions, kaons, gammas,
   lambdas, sigmas, xis - all of them species P1 already transports.

### 9.4 `theActualWidth[]`, which is the only hard arithmetic

`G4KineticTrack`'s CONSTRUCTOR fills one width per channel, at the track's ACTUAL (sampled) mass
rather than at the pole, and `Decay()` samples the channel from those. For a channel with
`nDaughters` 2 or 3:

```
thePoleWidth   = channel->GetBR() * parent->GetPDGWidth()
theMassRatio   = thePoleMass / theActualMass
theMomRatio    = theActualMom / thePoleMom
theActualWidth = thePoleWidth * theMassRatio * theMomRatio
```

and for `nDaughters == 1` (none of the 27 has one) it is just `BR * width`. What differs is how
the two momenta are got:

* **both daughters long-lived** (63 of the 68 channels): `EvaluateCMMomentum(mass, m[2])`, the
  closed two-body form, at the actual mass and at the pole mass. No integration.
* **one daughter short-lived** (5 channels): `IntegrateCMMomentum(lowerLimit)` and
  `IntegrateCMMomentum(lowerLimit, polemass)` - a numerical integral of the two-body momentum
  over the resonant daughter's Breit-Wigner, from `G4SampleResonance::GetMinimumMass(daughter)`
  upwards. The daughters are SWAPPED so the short-lived one is in slot 1.
* **both daughters short-lived** (none of the 68): `IntegrateCMMomentum2()`, a double integral.
* **three daughters, one short-lived** (none of the 68): the two long-lived masses are ADDED and
  `IntegrateCMMomentum(0.0)` is used.
* **three daughters, none short-lived** (3 channels): `theDaughterMass[1] += theDaughterMass[2]`
  and then the closed two-body form on the pair.

So the integrator is needed by five channels out of 68 and by no channel at all if those five
are refused by name. `G4SampleResonance` itself - `SampleMass`, `BrWigInt0`, `BrWigInv`,
`GetMinimumMass` - is already ported in `ftf/string_fragmentation.cuh` and
`data/ftf_hadrons.hh`'s `minmass` column.

### 9.5 What the contract has to carry

A track in, a list of tracks out, and nothing that knows about strings or cascades:

```
struct DecayTrack { int pdg; LorentzVector momentum; Vec3d position; double formation_time;
                    int creator_model_id; int parent_resonance_pdg; int parent_resonance_id; };
decay_kinetic_tracks(DecayTrack* list, int& n, int capacity, const DecayTables*, Rng&) -> refusal
```

`preco::CascadeTrack` already carries `pdg`, `momentum`, `position`, `formation_time`,
`creator_model_id` and `is_short_lived`, so the natural shape is for the engine to take a
`CascadeTrack` list in place and for P6 to call it where `G4DecayKineticTracks` is called - the
first line of both `Propagate` entry points - after which `GeneratorRefusal::short_lived_track`
can be deleted rather than reported. The two `parent_resonance_*` fields are new; Geant4 sets
them on every daughter and nothing in QBBC's scoring reads them, so they are bookkeeping to
carry rather than physics to reproduce.

The tables are data and belong in `src/data/`, generated the way `data/ftf_hadrons.hh` is: a
dump that walks `G4ParticleTable`, a Perl generator under `tools/`, and a test that diffs the
two so that a Geant4 release which changes a branching ratio fails a row instead of quietly
changing a shower. 27 species and 68 channels is a small file; the same generator should take
the species list as an argument, because the Binary cascade's set is not identical to FTFP's
and the union is what the module needs.

### 9.6 What P11d does once the module exists

Wire `G4DecayStrongResonances::Propagate` - the `hitCount == GetMassNumber()` exit of
`G4TheoFSGenerator::ApplyYourself`, which QBBC reaches on hydrogen and on any target whose every
nucleon was hit - to the shared engine, delete `kDecayStrongResonances`, and re-run the 96-point
sweep. On the numbers above that is the last refusal in it.

---

## 10. Phase 4 - the rest of QBBC

Decided with the user on 2026-10-03, after P21 landed (main b40c836, RISK V225). The target is
every capability of QBBC in 11.1.1, for any user and any beam, not only the ion and neutron
beams the sweep has carried so far. Measured against QBBC's constructor list, what main has and
what each remaining package closes:

| constructor | on main (b40c836) | package that closes the rest |
|---|---|---|
| G4EmStandardPhysics (option 0) | complete; the gamma general process's zones and table 9 are Geant4's (P21) | P26: the zone totals and EM fractions as tables (V225), the ion along-step charge correction, ion spin, delta rays above 17 GeV/u, the e+- residual above 100 MeV |
| G4EmExtraPhysics | gamma-, electro- and muon-nuclear wired (P19, P21) | P26: the lepton models' FTF arm from 10 GeV. QGS for photons above 3 GeV stays out: no 11.1.1 QBBC photon reaches it (V207) |
| G4DecayPhysics | complete for every transported species | P23b: the decay tables of K0S, K0L, the anti-hyperons and, if QBBC reaches them, the b/c hadrons |
| G4HadronElasticPhysicsXS | every charged hadron and the neutron | P23c: the antiproton and the anti-light ions (`G4ComponentAntiNuclNuclearXS`, `G4AntiNuclElastic`) |
| G4HadronInelasticQBBC | Binary, Bertini and FTFP in Geant4's windows for p, n, pi, K+-; hydrogen target; destroyed nucleus | P23b: hyperon, anti-hyperon and neutral-kaon projectiles and cross sections; P23a: those species on tracks; P23c: the anti-nucleon family |
| G4IonPhysicsXS | the light-ion reaction below 3 GeV/u, FTF above, every target | P24: the 128-entry final state, FTF's 512-track list, Bertini's secondary buffer, the AME2012 fallback |
| G4StoppingPhysics | every arm | P23c: the anti-nucleus hand-over behind Fritiof |
| G4IonElasticPhysics | wired for GenericIon (P25, V267-V271) | - |
| G4NeutronTrackingCut | complete | - |
| the neutron general process | wired; the 11.1.1 charging order is P22 (V219) | P22 |

### The packages

| package | scope | validation | after |
|---|---|---|---|
| **P22** | the neutron's interaction length on its track, 11.1.1's charging order, today's order selectable | the two neutron beams against shipped QBBC; the four ion beams at full counts | - |
| **lead: header split** | the interaction and drain kernel bodies out of `host/transport_run_impl.cuh` into a header only their units include; a sizes-only header for what the host reads | **done, V233** (2026-11-05): a stepping unit reads 43 hadronic headers where it read 133; a touched cascade header rebuilds 8 of 24 units in 83 min where it rebuilt 23 in 1 h 34 | P22 |
| **lead: the second card** | every unit compiled for the cards `setupenv.bat` detects, the compute_70 PTX beside them so any card from Volta up runs the same binary, the Titan V gated beside the 3070, a fresh clone from README | **done, V239** (2026-11-15): the Titan V reproduces the 3070's gates to every printed digit at 1.7x (hadrons) to 2.2x (photons) the rate; the JIT path costs about 25 min once per set of binaries; the stack reservation takes all of its 12 GB, which the Binary-frame slice is to shrink; a fresh clone builds and runs from the four commands | - |
| **lead: the Binary-frame slice** | the cascade's 256-entry working lists out of `run_interaction<kBinary>`'s stack frame into the slot; a run that would page refused by name | the frames from the units' ptxas logs; beams bit-identical before and after; the Titan V proton depth-dose gate | **done, V256-V258** (2026-11-16): kBinary 69,920 -> 15,648 bytes, kLightIon 61,728 -> 20,368 (now the largest), the raise 77,824 -> 24,576 bytes a thread on the package's base and 28,672 on main after P23c (the light-ion frame 20,592 there; the 3070's depth gate 4.65 GB left, the Titan V 6.04 GB of 12 where it had 0.00), the Titan V proton gate 48 min -> seconds and its whole gate stage 37 min, beams and the depth CSV identical to the digit on both cards; the paging guard reads the card's free memory before the raise, the pool and both queues (`G4GPU_ALLOW_PAGING=1` overrides); next cut: the light-ion frame | - |
| **P23b** | hyperon and s/c/b-meson cross-section branches; K0S/K0L/anti-hyperon (and b/c, if reached) decay tables; Bertini accepting K0L/K0S; FTF's hyperon, anti-hyperon and neutral-kaon rows | tapes and campaigns against the oracle, as P9d and P13 did; host-only | none (model-only) |
| **P23a** | the strange species on tracks: rows, ionisation and ranges for the charged ones, decay in flight, the neutral step, interaction lengths from P23b's cross sections | tapes against the loss tables and `G4Decay`; beams: 10 GeV protons and pions, where they are made | **done, V240-V244** (2026-11-15): the sixteen species on tracks through two group kernels, a Bertini hyperon that was a neutron fixed (V241), the antiproton's four derived species on splined tables (V242); open: the two 10 GeV beams' excess over QBBC, which predates P23a (V244, README open question 9) |
| **P23c** | the anti-nucleon family: `G4ComponentAntiNuclNuclearXS`, `G4AntiNuclElastic`, FTF's anti-nucleus arm, the Fritiof at-rest hand-over, the antiproton's processes | campaigns against FTF and the at-rest oracle; an antiproton beam | **done, V250-V255** (2026-11-16): the component and the elastic model exact (8,658 cross sections to 0.0; 9,408 sampled t to 2.7e-13), the hand-over's anti-nucleus arm (97.5% of 18,000 calls run), five species on tracks with no unit added; open: the antiproton beam +20.5% over QBBC, the FTFP annihilation final states (README open question 10, with question 9) |
| **P24** | no capacity refusals: the light-ion reaction's final state, FTF's track list, Bertini's secondary buffer, the AME2012 fallback mass formula | Fe56 on Fe56 at 1 GeV/u and Fe56 on Pb at 20 GeV/u refuse nothing; memory cost measured | **done, V234-V238** (2026-11-06): both grids refuse nothing, the slot 1,566,472 B and FTF's workspace 671,256 B pinned, the mass formula exact over 49,225 nuclides; open: the FTFP ion arm's multiplicity (V236) |
| **P25** | ionElastic wired for GenericIon | an ion beam's recoil spectrum against Geant4 | **done, V267-V271** (2026-11-17): the cross section per volume to 4.8e-16 at 5,112 points, Geant4's own `PostStepDoIt` replayed exactly over 2,728 calls and 120,000 campaign calls, 36 grid cases and a C12 beam host against device; no sweep row moved by a digit, every recoil of 650,000 primaries under the cut. The old 1e-9-per-recoil bound was wrong by two to four orders (the mean free path is 55-300 mm). Refused by name: a GenericIon primary (the gun carries no nuclide), a switch of its own (it runs under `hadron_elastic`) |
| **P26** | the EM remainders listed above | the gamma gate, the ion beams, a 10 GeV electron beam | **done in part, V245-V249** (2026-11-15): the general process's own tables read by the photon (V246, 6e-16; the B1 gate +0.033% toward Geant4), conversion's rate past 30 GeV, CorrectionsAlongStep (V247), GetA27 (V248); the ion's spin and delta rays **closed by P26b, V259-V261** (2026-11-17: ENSDFSTATE's 2,977 ground states on the device, the rejection exact, `kIonDeltaRay` at zero on the five ion and neutron rows, the doses unchanged within noise); refused by name still: the lepton models' FTF arm (V249, unreachable from a 10 GeV beam) and a nuclide with no ENSDFSTATE row; the four ion sweep rows taken by P26b at the default live-track count (V261); open question 5 narrowed to the integral approach |
| **P28** | the FTFP excess behind README's questions 9 and 10 | a per-interaction energy-balance tape against QBBC's own FTFP instances; the three FTFP beams | **done, V264-V266** (2026-11-17): the residual nucleus was never de-excited - its excitation (40 MeV for an antiproton at rest, 294-534 MeV for a 10 GeV proton) became an ion's kinetic energy; de-excited as Geant4 does it, 1,540 species comparisons inside 5 sigma, the three beams inside 1.9 sigma of Geant4 with the eta and eta' dropped on both sides; what stays below QBBC as it ships is the eta/eta' `ParticleType` gap (V240) |
| **P29** | the FTFP ion arm re-measured with the de-excited residual (README question 8) | V236's grid on the same instrument; per-species rows for the ion arm; the four ion sweep rows | **done, V272-V273** (2026-11-17): every grid point inside one sigma of Geant4 (the undecayed residuals reproduce V236's column to the digit), 171 per-species comparisons with none past 5 sigma (`tests/test_ion_arm.cu`, `ref/dump/dump_ion_arm.cc`), no deficit left so no stage tape; the four ion rows unchanged to the digit, their primaries being below ion FTFP's 3 GeV per nucleon; nothing under src changed |
| **P30** | a GenericIon primary: `/gun/particle ion` + `/gun/ion Z A [Q E]` as G4ParticleGunMessenger defines them, the nuclide carried by the seeding | the seeded nucleus against the same nucleus as a secondary, exact; C12 at 12 and 48 GeV and Fe56 at 56 GeV against QBBC | **done, V274-V277** (2026-11-18): the gun and the seeding exact (host-device 2.4e-13); its rows found two host defects older than P30 - every GenericIon projectile carried the placeholder's charge +1 and PDG code (since P8c), and the drain loop's bound of 1,000 abandoned a 1 GeV/u Fe56 - fixed by the lead (V277); the official rows -2.1, +0.5 and -1.6 sigma; refused by name: an excited-state ion, `/gun/ionL`, the charge Q on the track (nothing in QBBC reads it without a field) |
| **P31** | the eta and the eta' as tracks (V240's gap): G4Eta's and G4EtaPrime's decay tables, the eta' decay's short-lived rho0 and omega as 11.1.1 decays them | the no-row ledger named by a tape on six beams; 1,000,000 G4Decay decays a species against the port's; the three FTFP beams against QBBC AS IT SHIPS | **done, V278-V282** (2026-11-18): the ledger group held only the eta and the eta' (no light anti-nuclei); 108 decay channels exact, the campaign's worst row 3.0 sigma over 40 bins; the three FTFP beams -0.16, +0.75 and -1.14 sigma of shipped QBBC with nothing dropped on either side; the 200,000-event C12 12 GeV pair +0.12 sigma (V277's -1.20% was a draw). Refused by name: the rho0/omega track step itself (decayed in place, V280), DynamicalMass's three-body width branches (unreached) |
| **P27** | the acceptance sweep: every transported species as a primary, 100 MeV to 10 GeV, in water, bone, air, aluminium, polyethylene and lead | every row inside 3 sigma against shipped QBBC; every ledger at zero except the documented unreachable items | **done, V336-V341** (2026-11-21/22): 702 rows a side in four tiers (21 core species x 100 MeV-10 GeV x water, bone; the 23 other species the gun fires at 1 GeV; the low end, 10 keV-100 MeV, with `/B1/gun/z` at the trapezoid; 21 x 300 MeV, 1, 10 GeV x air, Al, PE, lead), `/B1/det/material` and `/B1/gun/z` on both B1s the only code (ref/b1mat for Geant4), the default proved untouched; 549 of 630 compared rows inside 3 sigma, the 81 outside named: lead unrunnable (the photoelectric reader holds 12 shells, Pb 24), hadronic dose in a gas 3-26% low (neutrons -85% at 1 MeV), a nucleus at 1 MeV/n deposited in place (fixed by P45, V342-V344: a 10 um residual-range floor Geant4 does not have, dropped for GenericIon; the four rows now -0.4 to +0.6 sigma, 553 of 630 inside), muons -0.3/-0.6% at 3/10 GeV, e+- 100 keV in air -1.6%, the triton at 100 MeV/n -1.4%, an Fe56 run abandoned at the drain bound; the ledgers named group by group; README's throughput table replaced (port at saturation, per core, ten-core box: EM 5-18x the box, protons 1.8x, Fe56 1.6x, pions/kaons/neutrons/light ions/C12 0.7-0.9x, antiprotons 0.47x) |
| **P45** | a nucleus at 1 MeV/n deposited where it starts: C12 and Fe56 fired 1 um before the trapezoid left all their energy in the envelope (P27, V340 item 3) | Geant4's step tape for the same primaries; the ion rows at the face rerun; the gates to the digit | **done, V342-V345** (2026-11-23): `step_hadron`'s 10 um residual-range floor (not Geant4's: `G4VEnergyLossProcess` stops only at `lowestKinEnergy` = 1 keV or the end of the range) deposited a GenericIon where it stood; lifted for the nucleus first, then by P44 for every species; C12/Fe56 at 1 MeV/n from -839 sigma to -0.4..+0.6; every other ion row inside 1.9 sigma; the eleven gates match, the proton/pion/Fe56 beams moved 0.003-0.04 sigma through recoils that now cross a boundary |
| **P46** | the muon above 1 GeV: P27's -0.6% at 10 GeV, every material | a per-process deposit tape against QBBC; each new model's final state replayed on Geant4's recorded deviates; P27's 38 muon rows rerun | **done, V346-V349** (2026-11-23): muIoni's delta ray was G4BetheBlochModel's (rate -1.42% at 10 GeV), muBrems/muPairProd were never drawn (0.13% of the 10 GeV dose); G4MuBetheBlochModel's, G4MuBremsstrahlungModel's and G4MuPairProductionModel's SampleSecondaries ported exact (pair tables bit for bit), wired in `step_hadron`; 36 of 38 muon rows inside 3 sigma (16 outside before; the 2 left are 300 MeV air, the gas); the eleven gates to the digit; the muon rate 0.57-0.76x above 1 GeV, the remedy (per-couple lambda tables) named |
| **P44** | the dose in a gas: why hadron and neutron beams read low in an air scoring volume (neutrons -85% at 1 MeV) and e+- at 100 keV -1.5% | a per-species deposit tape on both sides names the cause; fixed with Geant4's formulas; P27's air rows inside 3 sigma | **done, V350-V353** (2026-11-23): two causes, both the port's - a 10 um residual-range guard that deposited slow protons, nuclei and electrons where they were born (in an air cavity, on its water wall: Geant4 makes 51,552 recoil protons there per 4M 1 MeV neutrons and they cross into the air), removed; the e+- delta-ray, brems and annihilation rates drawn at the pre-step energy, replaced by eIoni's and eBrem's lambda tables and the integral approach (V78's refusal closed for e+-); the cuts audited (equal in the gas, 1e-5 to 3e-4 off in condensed materials, left); tier 2 in air 61 of 63 inside (the two: 10 GeV muons), every tier-4 air row inside (neutron 1 MeV -84.7% -> +1.4%, e+- 100 keV -1.5/-1.7% -> -0.05/-0.01%); the eleven gates rc 0 with every moved digit named |
| **P43** | every element up to Z = 92 runs: the photoelectric table's 12-shell cap and every per-scene element cap lifted, the refusals named; lead and uranium in the trapezoid | the Z scan against Geant4's own models; the gates bit-identical; P27's 72 lead rows and a uranium scan | **done, V354-V357** (2026-11-23): 29 shells (Pa, U), 100 elements a scene in the photoelectric, Rayleigh and Seltzer-Berger tables, every refusal printing the file and the reason; Z = 1-98 to 4.4e-16 and the shell walk exact against Geant4 (which itself stops at Es: G4ElementData holds Z < 99); 108 tests and the eleven gates to the digit; lead 70 of 72 rows run, 55 inside 3 sigma (every photon row inside), uranium's photons inside 0.2%; outside, for other owners: e- at 100 keV entering a high-Z face (+5.6% lead, +15.6% uranium), muons at 10 GeV -3.1%, Bertini refusals disposed inside the scorer (kaons +4-6%), the pool FATAL at 10 GeV, the drain bound on lead's neutron rows |
| **performance phase** | the transport-loop plan (persistent drain, per-model slot pools, warp-cooperative cascade, batch refill) | throughput against Geant4 on the sweep; every dose to the digit (the generator is keyed per primary) | **P32, part 1 (the loop), V283-V286** (2026-11-19): the profile (the hadron and ion beams are the interaction kernels, the host under 2.2%); the loop's host code its own header (19 s to rebuild); the batch refill (score window apart from the events in flight); the drain one thread a block, its slots sized to the card, FTFP's workspaces a count of their own - proton 1 GeV 6.7x, pi+ 10 GeV 3.9x, Fe56 1.7x, all to the digit. Refused by name: the persistent drain (2% to win), the warp-cooperative cascade, a per-bucket slot type; the second-stream drain built on perf/async-wip and not bit-identical on pi+ (V286). **P33, part 2 (the stepping kernels' occupancy), V287-V292** (2026-11-19): every stepping kernel at 255 registers and 8 warps an SM, the navigator spilling at that cap; `__launch_bounds__` at 3-8 blocks an SM measured 5-16% slower on the photon gate and not kept; the live list sorted by material and energy octave within a species - `run_step_lepton<e->` 1.37x, the photon gate 1.22x, proton 1.05x, pi+ 1.06x, Fe56 unchanged, all to the digit. Refused by name: the continuous/discrete kernel split (its lever is warps an SM, measured against), scratch out of the frame (a frame costs memory, not warps) **P34, part 3 (the interaction kernels and the lockstep), V293-V298** (2026-11-19): V286's 201 track-steps were P19's photo-/lepto-nuclear grid running past the slot count (640 threads over 628 slots; P32's ring put those entries behind the hadronic ones) - fixed; the 20.8 ms stall is this card's driver running two streams one after the other, so P32's win was batching, kept on one stream as the deferred drain (a queued interaction waits for a full-width drain, at most 256 iterations) - Fe56 11.1x; each interaction launch strides its bucket in its own slots instead of chunk by chunk - proton 1 GeV 1.62x, pi+ 10 GeV 1.47x over P33; all to the digit, the gates unchanged. Refused by name: the second stream (no overlap on this card), the per-bucket slot type and the frame cuts (slots past the 640 resident blocks add no width; FTFP's width tripled moved nothing), the field maps kept across tries and the warp-cooperative cascade (V297: `Propagate` is 85% of a reaction's thread time) **P37, part 4 (the ion step), V303-V306** (2026-11-19): Geant4's Fe56 rate is 40.3 events a core-second (2,000 serial) and 31 on ten workers (the lead's cpu_rates.md); an ion step costs 270 us of ONE thread's time and a proton step 430 us (per-phase clock64 timers in `step_hadron`: ionElastic's cross section 45 us, msc 62 us, the delta ray 37 us, ionInelastic's length 29 us, the loss 25 us - no phase is a tenth of it), so the "forty times a photon's" is the card's emptiness, not the ion's physics: 2,000 live ions a launch on 16 of 80 SMs, five to fifteen species launched one after another, three 62 us read-backs an iteration, and 71% of the Fe56 5k GPU time in the light-ion and Binary drains (P36's). Fixed, all to the digit on the four beams, the 20k ion beam, the eleven gates and the eleven GPU tests: the species' stepping launches run at once on their own streams (this card does run them together: ten 200 us kernels in 0.27 ms) - Fe56 5k 1.19x, 20k 1.11x; the loop's three read-backs through mapped host memory with the sort's prefix sum on the card - Fe56 5k 49.7 -> 40.6 s (123.2 ev/s), 20k 171.9 -> 153.7 s (130.1 ev/s), photon gate 425 -> 405 ms, proton 1.05x, pi+ 1.05x. Refused by measurement: the launch's block shape (32-256, no effect), a one-launch sort (no faster), the deferral bound (128-1024 within 1.5% of 256). Named and left: caching the ion's interaction lengths (not the identical double at a continuous energy), the duplicate nuclide lookups in the GenericIon kernel (10 us of a 270 us thread, behind the proton's 430 us in most iterations; every unit stale for it), several steps a launch for the ion tail |
| **performance phase** | the transport-loop plan (persistent drain, per-model slot pools, warp-cooperative cascade, batch refill) | throughput against Geant4 on the sweep; every dose to the digit (the generator is keyed per primary) | **P32, part 1 (the loop), V283-V286** (2026-11-19): the profile (the hadron and ion beams are the interaction kernels, the host under 2.2%); the loop's host code its own header (19 s to rebuild); the batch refill (score window apart from the events in flight); the drain one thread a block, its slots sized to the card, FTFP's workspaces a count of their own - proton 1 GeV 6.7x, pi+ 10 GeV 3.9x, Fe56 1.7x, all to the digit. Refused by name: the persistent drain (2% to win), the warp-cooperative cascade, a per-bucket slot type; the second-stream drain built on perf/async-wip and not bit-identical on pi+ (V286). **P33, part 2 (the stepping kernels' occupancy), V287-V292** (2026-11-19): every stepping kernel at 255 registers and 8 warps an SM, the navigator spilling at that cap; `__launch_bounds__` at 3-8 blocks an SM measured 5-16% slower on the photon gate and not kept; the live list sorted by material and energy octave within a species - `run_step_lepton<e->` 1.37x, the photon gate 1.22x, proton 1.05x, pi+ 1.06x, Fe56 unchanged, all to the digit. Refused by name: the continuous/discrete kernel split (its lever is warps an SM, measured against), scratch out of the frame (a frame costs memory, not warps) **P34, part 3 (the interaction kernels and the lockstep), V293-V298** (2026-11-19): V286's 201 track-steps were P19's photo-/lepto-nuclear grid running past the slot count (640 threads over 628 slots; P32's ring put those entries behind the hadronic ones) - fixed; the 20.8 ms stall is this card's driver running two streams one after the other, so P32's win was batching, kept on one stream as the deferred drain (a queued interaction waits for a full-width drain, at most 256 iterations) - Fe56 11.1x; each interaction launch strides its bucket in its own slots instead of chunk by chunk - proton 1 GeV 1.62x, pi+ 10 GeV 1.47x over P33; all to the digit, the gates unchanged. Refused by name: the second stream (no overlap on this card), the per-bucket slot type and the frame cuts (slots past the 640 resident blocks add no width; FTFP's width tripled moved nothing), the field maps kept across tries and the warp-cooperative cascade (V297: `Propagate` is 85% of a reaction's thread time) **P36, part 5 (the cascade as data), V307-V312** (2026-11-20, under V302's relaxed rule): the Binary cascade and the light-ion reaction cut at `Propagate`'s time steps into stages over 16,384 and 2,048 lanes, the field transport one item per secondary, every draw in the thread path's order from the cascade's own key, the tail on the 667 slots, the hand-backs (K = 64/160, capacity, refusals) rerun on the thread path; bic/ untouched, the thread path to the digit, the campaign within 5 sigma (an absorption dropped is caught at 19) - Fe56 20k 2.53x, Fe56 5k 1.64x, proton 1.14x, pi+ 1.06x, the photon gate 0.93x (the pools' memory). Left by name: the set-up and the de-excitation (now two thirds of a drain), lazy pools, a hook project on the flat path. Refused: SoA across lanes, the nucleus in shared memory and the sort by target, the per-pair phase, a cooperative loop **P40 (the Bertini cascade and the FTF string model as data), V313-V317** (2026-11-20): measured first and both REFUSED - on integ/p36 the pion beam is the flat Binary drain 67%, FTFP 11%, Bertini 7%; a Bertini drain is as long as its slowest cascade (76 ms, the slowest call 81 ms; 0.21 s of work in 0.61 s; one particle a fate, 40 fates, p99.9 515) and an FTF interaction has 2.06 strings and 1.9 splits with 68-72% of its time in the residual's de-excitation. The flat pools built only in the final hadronic stage: the photon gate 433 -> 408 ms, every digit kept. Left by name: the FTF workspace count (384: -2% at the card's last 0.08 GB), the Bertini and FTFP drains overlapped with the Binary drain, the shared de-excitation |
| **performance phase** | the transport-loop plan (persistent drain, per-model slot pools, warp-cooperative cascade, batch refill) | throughput against Geant4 on the sweep; every dose to the digit (the generator is keyed per primary) | **P32, part 1 (the loop), V283-V286** (2026-11-19): the profile (the hadron and ion beams are the interaction kernels, the host under 2.2%); the loop's host code its own header (19 s to rebuild); the batch refill (score window apart from the events in flight); the drain one thread a block, its slots sized to the card, FTFP's workspaces a count of their own - proton 1 GeV 6.7x, pi+ 10 GeV 3.9x, Fe56 1.7x, all to the digit. Refused by name: the persistent drain (2% to win), the warp-cooperative cascade, a per-bucket slot type; the second-stream drain built on perf/async-wip and not bit-identical on pi+ (V286). **P33, part 2 (the stepping kernels' occupancy), V287-V292** (2026-11-19): every stepping kernel at 255 registers and 8 warps an SM, the navigator spilling at that cap; `__launch_bounds__` at 3-8 blocks an SM measured 5-16% slower on the photon gate and not kept; the live list sorted by material and energy octave within a species - `run_step_lepton<e->` 1.37x, the photon gate 1.22x, proton 1.05x, pi+ 1.06x, Fe56 unchanged, all to the digit. Refused by name: the continuous/discrete kernel split (its lever is warps an SM, measured against), scratch out of the frame (a frame costs memory, not warps) **P34, part 3 (the interaction kernels and the lockstep), V293-V298** (2026-11-19): V286's 201 track-steps were P19's photo-/lepto-nuclear grid running past the slot count (640 threads over 628 slots; P32's ring put those entries behind the hadronic ones) - fixed; the 20.8 ms stall is this card's driver running two streams one after the other, so P32's win was batching, kept on one stream as the deferred drain (a queued interaction waits for a full-width drain, at most 256 iterations) - Fe56 11.1x; each interaction launch strides its bucket in its own slots instead of chunk by chunk - proton 1 GeV 1.62x, pi+ 10 GeV 1.47x over P33; all to the digit, the gates unchanged. Refused by name: the second stream (no overlap on this card), the per-bucket slot type and the frame cuts (slots past the 640 resident blocks add no width; FTFP's width tripled moved nothing), the field maps kept across tries and the warp-cooperative cascade (V297: `Propagate` is 85% of a reaction's thread time) **P36, part 5 (the cascade as data), V307-V312** (2026-11-20, under V302's relaxed rule): the Binary cascade and the light-ion reaction cut at `Propagate`'s time steps into stages over 16,384 and 2,048 lanes, the field transport one item per secondary, every draw in the thread path's order from the cascade's own key, the tail on the 667 slots, the hand-backs (K = 64/160, capacity, refusals) rerun on the thread path; bic/ untouched, the thread path to the digit, the campaign within 5 sigma (an absorption dropped is caught at 19) - Fe56 20k 2.53x, Fe56 5k 1.64x, proton 1.14x, pi+ 1.06x, the photon gate 0.93x (the pools' memory). Left by name: the set-up and the de-excitation (now two thirds of a drain), lazy pools, a hook project on the flat path. Refused: SoA across lanes, the nucleus in shared memory and the sort by target, the per-pair phase, a cooperative loop **P39, part 6 (the nucleus factory in the flat set-up), V318-V324** (2026-11-20, under V302): P38's factory cherry-picked (V318/V319); the Binary set-up asks the factory for its nucleus each turn (keyed on the cascade's rng_key, step and nucleus count, written into the record), the set-up every round, and a straggler finished on its own lane - a thread-path rerun from the key is no longer the same cascade (the first campaign lost the K+ and Lambda events at 9-11 sigma); proton 1 GeV 1.50x, pi+ 10 GeV 1.19x, Fe56 5k 1.11x, 20k 1.06x, the photon gate unchanged, every dose inside 0.5 sigma of the thread path, the campaign inside 3.6 sigma, the anti-vacuity at 17.5, the gates ba73's (the depth gate inside its tolerance), the thread path V299's to the digit. The light-ion factory measured (Fe56 1.14x / 1.17x) and OFF by default: its capacity hand-backs (5.7% of the Fe56-on-Al27 campaign, the 256-track pool) are reruns of other reactions (-12 sigma). Left by name: the flat scratch lists' cap apart from the pool (to turn the light-ion factory on), the impact-point loop, the wide de-excitation queue (V323: the tail is now the largest stage of every drain) **P41 (the de-excitation tail off the interaction slots), V325-V329** (2026-11-20): the tail measured (a Binary residual 0.6 ms, 1-2 BreakFragment passes, 11 draws; a light-ion residual 1.8 ms with a 32-96 ms hot tail; the handler writes 5/27/26/27/27 of its 536 kB); a tail pool (4,096 Binary entries of 88 kB, 1,024 light-ion of 177 kB) with only the de-excitation workspace compact and a binding one CONTINUED from its state on a slot-sized workspace, never rerun; launched full-or-idle, one lane a block (both measured) - proton 1.16x, pi+ 1.04x, proton 10 GeV 1.09x, Fe56 5k 1.09x, 20k 1.21x, every dose and step count integ/p40's to the digit, the photon gate unchanged. The thread-path drains beside the flat drain: built with the slot ranges apart, measured no gain (the flat rounds queue behind them), off by default. Left by name: the set-up (P39), a persistent flat drain **P42 (the light-ion factory on by default), V331-V334** (2026-11-20, under V302 and the hand-back rule): every one of the campaign's 1,139 Fe56-on-Al27 overflows is in the TAIL (the final decays re-add every final track; 317 tracks and 381 collisions at most, 1 in 19,587 beam reactions); `FlatScratch`'s lists apart from the pool (the thread path's 256); a reaction whose next stage could outgrow its record stops at that boundary (exact bounds before a time step; in the tail the generator put back before the adds) and is CONTINUED on one of 64 records with the thread path's capacities (1,024 tracks, 8,192 collisions) on one thread, its tail on a slot - never rerun; the continued campaign is the larger-record campaign to the bit (0 of 120,000 differ), 0 capacity hand-backs, worst +3.05 / -3.09 sigma, the light-ion factory ON by default - Fe56 5k 1.05x, 20k 1.11x, C12 12 GeV 50k 1.07x, every dose inside 1 sigma of the thread path; `G4GPU_FLAT_FACTORY_ION=0` is V330 to the digit |

Two things stay out on purpose and are recorded, not forgotten: QGS for photons above 3 GeV
(unreachable in 11.1.1's QBBC, V207; the Geant4-version-update plan picks it up when a later
release is the target) and the EM families that are not QBBC (Penelope, PAI, DNA, optical,
adjoint, PIXE).

### Order and load

Integration stays serial, one gate per landing, the lead restacking each branch onto main. The
machine takes one heavy job at a time (section 6, rule 2, and the user's rule since P15), and the
packages are of two kinds: model-only packages (P23b, P24, P26) compile host-only tests and run
nothing on the GPU, transport packages (P22, P23a, P23c, P25) compile engine units and run beams.
One of each kind may run side by side; never two transport packages, never two ptxas, never a
compile beside a GPU run. The sequence:

1. P22 with P23b beside it (started 2026-10-03).
2. The header split as soon as P22 lands, so every gate after costs one unit, not 23 - landed
   2026-11-05 as V233: eight model-carrying units, not one (see the entry for why), and no
   stepping unit.
3. P23a with P24 beside it; then P23c with P26 beside it; P25 in any free transport slot.
4. The performance phase once the kernel code has stopped moving; P27 last, as the acceptance.

RISK numbering: every brief starts at a provisional number - P22 was briefed from V280, P23b from
V300, later packages in steps of 20 - and the lead renumbers on integration: P23b landed as
V226-V229, P22 as V230-V232. Briefs follow the P15-P22 template: measurement first, transcription from 11.1.1,
anti-vacuity, refusal by name, report format.
