# g4gpu — GPU re-architecture of Geant4, full physics parity

> **Status against this plan.** Delivered: the core architecture (track-parallel SoA
> scheduler, counter-based RNG, table formats, dataset loaders), the complete
> `G4EmStandardPhysics` model set, 18 of the ~31 solids plus the three boolean operations and a
> voxel volume, a Geant4-shaped host API, scoring through `G4MultiFunctionalDetector`, a
> viewer and a model-building GUI. Not started: hadronic physics (83 models, 58 cross-section
> sets), the remaining solids, and multi-GPU. The bucket estimates below are unchanged and
> still look about right for what remains; see docs/ROADMAP.md for the near-term list and
> docs/RESULT.md for what is measured.


Reference: Geant4 11.5.0 (1,627,188 LOC in `source/`, 3,509 G4 classes, 484 abstract interfaces).
Transport-relevant subset targeted: ~1.32M LOC. Estimated deliverable: ~600k LOC new CUDA/C++.

## Non-negotiable conventions

These exist so that any future session can resume mid-stream without re-deriving the design.
Violating them silently is the main failure mode of this project.

1. **No virtual functions in device code, ever.** Process/model dispatch is an explicit
   `switch` over a `ModelId` enum, or a sorted-partition kernel launch. Never a vtable.
2. **Every physics function is `__host__ __device__`.** A host-only build of byte-identical
   logic must exist for every sampler, so it is debuggable in a scalar debugger and diffable
   against Geant4 with no GPU involved. This is the primary validation affordance.
3. **SoA everywhere.** No `G4Track`-style struct-of-pointers. Tracks live in parallel arrays
   indexed by slot id. No pointer chasing in a kernel.
4. **No device-side allocation.** Arena/pool allocators sized at init. Secondaries append to
   a preallocated pool with an atomic bump; overflow is a detected, reported condition, never
   a silent truncation.
5. **Scalar type is a template parameter** (`real_t`). FP32 default, FP64 via compile switch.
   Rationale: consumer cards (RTX 3070 = 1/32 FP64 rate) force FP32; datacenter cards do not.
   Never hardcode `double` or `float` in physics code.
6. **No exceptions, no RTTI, no `std::` containers, no I/O in device code.**
7. **Counter-based RNG** (Philox), seeded by (event_id, track_id, interaction_counter), so
   results are reproducible independent of scheduling order. Never a sequential engine.
8. **Every judgment call gets an entry in `docs/RISK.md`** with file, line, and what I was
   unsure about. Targeted debugging beats a search.

## Architecture

Track-level parallelism with many events in flight. NOT one-thread-per-event
(catastrophic divergence, load imbalance, register pressure).

    upload primaries for N events  ->  [ device-resident loop ]  ->  download hits
                                          |
                                          v
      partition live tracks by (particle, winning process, region)
      launch one specialized kernel per partition
      append secondaries to pool (atomic bump)
      stream-compact the pool
      repeat until pool drains

Events are a *tag* on each track, never a scheduling unit. The stepping loop never returns
to the host. Scheduler runs on device (persistent kernel or CUDA graphs) so even launch
overhead is not a round trip.

## Phases

Unit of work is **distinct algorithm classes (~437)**, not LOC. LOC overstates the job:
em/standard is 61,407 lines but only 27,789 are non-noise (55% blank/comment/brace/preproc),
and a typical model (G4KleinNishinaCompton, 281 lines) is ~40 lines of real sampling math.

Measured rate: Klein-Nishina took 2 turns including reading the Geant4 source.

Inventory: 203 EM models, 62 EM processes, 83 hadronic models, 58 hadronic cross-section
sets, 31 solids.

| Bucket | Count | Turns ea | Turns | Notes |
|---|---|---|---|---|
| Simple models (closed form / simple rejection) | ~260 | 2 | ~520 | Parallelizes across subagents |
| Moderate (table-driven, interpolated, multi-branch) | ~130 | 5 | ~650 | Parallelizes |
| Hard (Urban MSC, Navigator, ExcitationHandler, cascades, booleans, INCL++, parton string) | ~45 | 25 | ~1125 | Partly critical path |
| **Transcription subtotal** | 437 | | **~2300** | |
| Core architecture (SoA pool, device scheduler, sort/compact, RNG, table format) | - | - | 400-700 | **Critical path, not transcription** |
| Table generation + 22 dataset loaders | - | - | 200-400 | |
| Host API, scoring, hit buffers, device run/event mgmt | - | - | 200-400 | |
| Build system across ~437 modules + self-consistency tests | - | - | 300-500 | |
| Performance (occupancy, register pressure, kernel splitting, CUDA graphs, multi-GPU) | - | - | 400-800 | **The actual goal, not free** |
| Compile-error fixing | - | - | 200-400 | |
| **Total** | | | **~4000-5500** | |

**~300-550M tokens. 60-120h agent operation -> 2-4 weeks calendar**, or ~1-2 weeks run hard
with aggressive subagent fan-out on the ~1170 turns of independent model transcription.

### Where the cost concentrates
Not in the models. In (a) the core architecture, because all 437 modules are written against
it and design errors propagate everywhere, and (b) performance work, because transcribing
437 models correctly yields a *working* GPU engine, not an optimal one.

### The ~45 hard classes
G4UrbanMscModel: 638 real lines, 148 branches, 15 hardcoded tuned constants. "Correct" means
reproducing Geant4's accumulated empirical tuning; subtle errors silently distort every
shower downstream. G4Navigator: 937 real lines, 255 branches, 61 state variables.
inclxx / cascade / parton_string / im_r_matrix use deep recursion and dynamic-size final
states, needing explicit-stack reformulation. Research, not transcription.

### Rate caveat
The 2-turn rate holds for self-contained models. It degrades for models reading shared state
(material/element composition loops, production cuts, region parameters) - exactly what the
core architecture must get right first.

## Toolchain (verified on this machine, 2026-09-03)

- GPU: RTX 3070, 8GB, CC 8.6. FP64 at 1/32 FP32 rate -> FP32 default.
- CUDA 11.6 (also 11.2, 10.0 installed). Driver supports up to CUDA 13.3.
- **Host compiler must be MSVC 14.29 (VS 2019).** CUDA 11.6 does not support MSVC 14.44 (VS 2022).
- 20 cores, 64GB RAM. D: has 1.7TB free.
- Geant4 11.5.0 reference clone: `D:/g4gpu/reference-geant4` (249MB, shallow). **Not the
  oracle's version** - see `reference-geant4/WRONG_VERSION.md`. The tree that matches the
  oracle is `D:/Documents/Geant4/Windows/geant4-v11.1.1`, and `tools/g4src.sh` is what decides
  which is which; it refuses a mismatch. `docs/RISK.md` O7.

## Division of labour

I write and compile. I do not run physics validation. Handoff is code that builds and links,
with self-consistency tests (sampler normalization, energy-momentum conservation, analytic
limiting cases, RNG reproducibility) that need no Geant4 oracle, plus `docs/RISK.md`.
