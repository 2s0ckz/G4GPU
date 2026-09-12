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
2. **Do not run `build_all.bat`.** It takes twenty minutes, it uses the one GPU, and it writes
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
5. **Shared files are append-only, and only these**: your `TESTS` entry in `build_all.bat`;
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
has always been), resolve the append-only conflicts, `build_all.bat`, commit to `main`. The
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
