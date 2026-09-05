# Progress and measured rate

> **Historical.** This is the log of the first 22 turns, kept for the reasoning. Two things in
> it are now wrong. The reference numbers quoted below are from a single 10,000-event Geant4
> run carrying +/-2.9%, which is the comparison docs/RISK.md S1 is about - the reference is now
> 2,000,000 events. And the component table describes the geometry before the layer rewrite.
> For the current state read docs/RESULT.md, and for what is unfinished docs/ROADMAP.md.


Target: reproduce Geant4 example B1 on GPU. Reference numbers ship in
`reference-geant4/examples/basic/B1/exampleB1.out`, so no Geant4 build is needed:

```
10000 gamma of 6 MeV
cumulated edep       = 4.69408 GeV
mass scoring volume  = 1.7316 kg
Absorbed dose        = 434.324 pGy   rms = 12.4101 pGy
```

B1 as specified: 6 MeV gamma, direction +z, launched from z = -150 mm with x,y uniform
in +/-80 mm. Scoring volume is **Shape2** (the bone trapezoid), not the envelope.
Physics needed is EM only - B1 nominally uses QBBC, but 6 MeV photons in water/tissue/bone
never reach hadronic channels.

## Done and validated (22 turns)

| Component | LOC | Validation |
|---|---|---|
| `core/units.cuh`, `vec3.cuh`, `rng.cuh`, `particle.cuh`, `secondary_pool.cuh` | ~330 | Philox stream reproducibility/decorrelation, rotate_uz identity + length preservation, water number density |
| `physics/em/klein_nishina.cuh` | 110 | Exact per-interaction energy conservation over 20k samples; sampled E1 minimum 0.2451 MeV vs analytic Compton edge E0/(1+2E0/me) = 0.2451 MeV |
| `geometry/solids.cuh` (Box, Cons, Trd) | 250 | MC volume: Trd 936.07 vs 936.00 cm3, Cons 175.90 vs 175.93 cm3. **Trd mass 1.7317 kg vs B1 reported 1.7316 kg.** 628k rays, zero entry/exit inconsistencies |
| `geometry/navigator.cuh` | 90 | Axial ray traverses exactly 60.000 mm of bone; 45,691 random boundary crossings, zero locate/step disagreements |
| `data/materials.cuh` | 110 | Verbatim NIST compositions from G4NistMaterialBuilder.cc; water n_e = 3.343e20 e/mm3 |
| tests | ~380 | 3 suites, all passing |

Toolchain: `nvcc -std=c++17 -O2` with CUDA 11.6 + MSVC 14.29 (VS 2019). C++14 fails -
`namespace a::b` is C++17.

## Honest gaps

- **Nothing has run on the GPU yet.** Every test above is host-side. The
  `__host__ __device__` discipline means it should port, but device compilation,
  memory management and kernel launches are untested.
- Navigator is translations-only, no rotations, non-overlapping daughters assumed.
  True for B1, unchecked at runtime.
- Cons supports rmin=0 and full phi only.

## Remaining for GPU B1

| Work | Turns |
|---|---|
| Gamma cross sections: Compton total (formula in hand), pair production, photoelectric, total mean free path | 15-25 |
| e-/e+ transport: Berger-Seltzer dE/dx, CSDA range, condensed-history step, MSC, brems, annihilation | 30-50 |
| Device kernels: port to CUDA, memory, wave-based scheduler | 25-40 |
| Scoring in Shape2, host main, B1-format output | 10-20 |
| End-to-end tuning to reach 434 pGy | 25-50 |
| **Remaining** | **105-185** |

## Revised estimates (measured, not projected)

| Scope | Earlier estimate | Now |
|---|---|---|
| **GPU B1** | not separately costed | **130-210 turns, ~7M tokens, ~$20-40** |
| Minimum useful deliverable (general EM engine) | 1,050-1,750 turns | **700-900 turns** |
| Full parity, all physics | 4,000-5,500 turns | **2,500-4,000 turns** |

Rate driver: foundational math with clean analytic validation runs ~10x faster than
estimated. The full-parity revision is the least reliable of the three - none of the
~45 hard classes (Urban MSC, cascade reformulation) has been touched yet, and MSC is
the next real test of the rate.
