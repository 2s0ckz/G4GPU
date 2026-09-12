# B1 sweep: four species, three energies, the port against Geant4 11.1.1

Run on 2026-09-11 from main at `8597c5a` (Phase 2 of the hadronic port complete except the two
items named at the end) with `tools/b1_sweep.ps1`. Example B1's geometry - a 20 x 20 x 30 cm water
envelope, the tissue cone, the bone trapezoid at z = +7 cm that is the scoring volume - and its
gun on the -z face. Twelve beams, three runs each:

- **port**: this port's `examples/B1/exampleB1.exe`, on an NVIDIA GeForce RTX 3070.
- **G4 EM-only**: the real Geant4 11.1.1, QBBC, with ONLY what the port still lacks inactivated:
  the inelastic processes (`protonInelastic`, `alphaInelastic`, `ionInelastic`), the hadron
  radiative processes `hBrems`/`hPairProd`, `photonNuclear`/`electronNuclear`/`positronNuclear`,
  and `ionElastic` for the recoil nuclei. Elastic, capture, decay and single Coulomb scattering
  are active on both sides. This is the like-for-like column. Serial run manager, one thread of
  an Intel i9-10850K.
- **G4 QBBC**: the same Geant4 as QBBC ships, nothing inactivated. Its distance from the EM-only
  column is what the physics the port does not have yet is worth for that beam.

The photon like-for-like runs switch the gamma general process off before initialisation, because
`photonNuclear` lives inside it and cannot be inactivated alone; the QBBC column keeps it on, so
the two Geant4 photon columns bracket that switch as well. `/particle/process/dump` for every
Geant4 run is in the sweep's `dumps.txt`; the process lists are what ran, not what was intended.

Doses are the cumulated dose in the trapezoid for the whole run, in Gy, with B1's printed rms,
which is the standard error. Event counts were chosen so that the statistical error is at or
under 0.2% for every row except the 20 MeV electron (0.9%). "loop" is each side's own
event-loop time: the port's host clock over primary generation and every batch, Geant4's
`G4Timer` from `InitializeEventLoop` to `TerminateEventLoop`. "wall" is the whole process,
start-up and table building included. No other process used the GPU or more than one CPU core
while the Geant4 runs were timed, except four ten-second diagnostic runs of the port during the
210 MeV proton beam.

## The table

| beam | events | port (Gy) | G4 EM-only (Gy) | diff | sigma | G4 QBBC (Gy) | QBBC vs EM-only | port loop | G4 EM loop | ratio | port wall | G4 EM wall |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| gamma 1 MeV | 2,000,000 | 1.2724E-008 ± 2.6E-011 | 1.2745E-008 ± 2.6E-011 | -0.16% | -0.6 | 1.2752E-008 | +0.05% | 959 ms | 20,692 ms | 22x | 3,791 ms | 21,827 ms |
| gamma 6 MeV | 2,000,000 | 8.5170E-008 ± 1.7E-010 | 8.5615E-008 ± 1.7E-010 | -0.52% | -1.8 | 8.5640E-008 | +0.03% | 1,211 ms | 31,745 ms | 26x | 2,749 ms | 32,746 ms |
| gamma 100 MeV | 1,000,000 | 5.4649E-007 ± 1.1E-009 | 5.4796E-007 ± 1.1E-009 | -0.27% | -1.0 | 5.4903E-007 | +0.20% | 1,945 ms | 63,376 ms | 33x | 3,493 ms | 64,362 ms |
| e- 20 MeV | 500,000 | 4.6289E-009 ± 4.0E-011 | 4.5247E-009 ± 4.0E-011 | +2.30% | 1.8 | 4.6343E-009 | +2.42% | 821 ms | 24,208 ms | 29x | 2,349 ms | 25,190 ms |
| e- 100 MeV | 300,000 | 3.3107E-007 ± 6.0E-010 | 3.3204E-007 ± 6.0E-010 | -0.29% | -1.1 | 3.3342E-007 | +0.41% | 1,577 ms | 46,893 ms | 30x | 3,096 ms | 47,868 ms |
| **e- 1000 MeV** | 100,000 | 1.1062E-007 ± 3.5E-010 | 2.4027E-007 ± 7.6E-010 | **-53.96%** | **-154.9** | 2.3919E-007 | -0.45% | 626 ms | 24,529 ms | 39x | 2,159 ms | 25,493 ms |
| proton 210 MeV | 500,000 | 3.0061E-006 ± 3.2E-009 | 3.0072E-006 ± 3.2E-009 | -0.04% | -0.2 | 2.5015E-006 | -16.82% | 2,345 ms | 24,677 ms | 11x | 3,892 ms | 25,651 ms |
| proton 400 MeV | 300,000 | 5.9549E-007 ± 8.8E-010 | 5.9691E-007 ± 8.8E-010 | -0.24% | -1.1 | 6.5863E-007 | +10.34% | 1,142 ms | 16,341 ms | 14x | 2,678 ms | 17,318 ms |
| proton 1000 MeV | 200,000 | 2.6281E-007 ± 4.9E-010 | 2.6217E-007 ± 4.9E-010 | +0.24% | 0.9 | 3.9165E-007 | +49.39% | 822 ms | 11,510 ms | 14x | 2,334 ms | 12,482 ms |
| alpha 840 MeV | 300,000 | 7.3833E-006 ± 1.0E-008 | 7.3867E-006 ± 1.0E-008 | -0.05% | -0.2 | 4.9323E-006 | -33.23% | 2,218 ms | 33,778 ms | 15x | 3,746 ms | 34,747 ms |
| alpha 1600 MeV | 200,000 | 1.5483E-006 ± 2.6E-009 | 1.5522E-006 ± 2.6E-009 | -0.25% | -1.1 | 1.5233E-006 | -1.86% | 1,994 ms | 37,426 ms | 19x | 3,530 ms | 38,391 ms |
| alpha 4000 MeV | 100,000 | 5.1270E-007 ± 1.2E-009 | 5.1150E-007 ± 1.2E-009 | +0.23% | 0.7 | 6.6342E-007 | +29.70% | 1,230 ms | 19,535 ms | 16x | 2,768 ms | 20,500 ms |

"sigma" is the difference over the quadrature sum of the two errors. The raw rows are in
`docs/B1_SWEEP.csv`.

## What the table says

**Eleven rows agree with Geant4 within statistics.** Nine are inside 1.1 sigma; the 6 MeV
photon (-0.52%, 1.8 sigma) and the 20 MeV electron (+2.3%, 1.8 sigma) are single samples at the
edge, and the electron row's two Geant4 columns differ from each other by the same 2.4% - at 0.9%
per run that row is noise, not a bias. The 6 MeV photon row is the port's standing B1 figure
(85.1695 nGy, the gate's 425.847 pGy per 10,000 events) against a fresh Geant4 seed; both Geant4
columns today sit 0.2% above the gate's stored reference from the same configuration, which is
the seed-to-seed spread README records. Every proton and alpha row - including the two Bragg-peak
rows at 210 MeV and 840 MeV, the most range-sensitive kind - is inside 0.25% and 1.1 sigma, with
`hadElastic`, `CoulombScat`, decay and the recoil ions all acting on both sides.

**One row is a defect, and it is not a hadronic one.** At 1 GeV the port deposits 46% of
Geant4's dose. Four diagnostic runs at 150, 300, 600 and 1000 MeV give 110.6, 110.8, 110.7 and
110.6 nGy per 100,000 electrons - identical to the 100 MeV row's 110.4 - while Geant4 rises to
240. The port transports every electron above 100 MeV as a 100 MeV electron: the e+- dE/dx,
range and inverse-range table in `src/physics/em/electron_processes.cuh` is built from 1 keV to
100 MeV, `range()` clamps a higher energy to its last bin and `energy_from_range()` returns the
table's ceiling, so after one step the excess is gone, deposited nowhere. Geant4's tables run to
100 TeV. A 2,000,000-event 6 MeV gamma gate could never see it; the first beam that could, did.
RISK V64; the fix (tables on Geant4's grid, `G4eBremsstrahlungRelModel` above 1 GeV,
`G4WentzelVIModel` for e+- above 100 MeV) is package P14c. Until it lands, the port is not to be
used for electrons or positrons above 100 MeV.

**What the missing physics is worth.** The QBBC column against EM-only: photons within 0.2%
(photonuclear), electrons within 0.4% (electro-nuclear and Coulomb recoils), and the hadrons
where the inelastic processes live: -16.8% for the stopping 210 MeV proton, +10.3% at 400 MeV,
+49% at 1 GeV, -33% for the stopping 840 MeV alpha, -1.9% at 1.6 GeV, +30% at 4 GeV. The sign
flips with whether the primary stops in the trapezoid (inelastic removes protons before the
peak) or crosses it (fragments and secondaries deposit more than the primary alone). These are
the numbers Phase 3 - Binary cascade, Bertini, FTFP - is judged against.

**Timing.** The port's event loop is 11 to 39 times Geant4's EM-only loop, and the ratio grows
with energy because Geant4's per-event cost grows with the number of steps while the port's
per-iteration cost is set by how many tracks are live. The wall-clock ratio is 6 to 18x: the
port spends 1.5 to 2.5 s per run on start-up, table building and upload that a 2,000,000-event
run amortises and a 100,000-event run does not. Neither number is a speedup claim against QBBC:
the port is not yet doing the inelastic work, and a comparison against physics you did not
implement is not a comparison (RESULT.md says this at length). The like-for-like column is the
one to quote, and it says: same answer, one GPU against one core, 11 to 39 times sooner.

## Caveats

- Alpha, He3 and the recoil ions are still scattered with WentzelVI where Geant4 uses Urban
  (RISK V63: the Urban ion branch is transcribed and matches Geant4, but enabling it crashes the
  compiler until the transport translation unit is split). The alpha rows above carry that
  substitution and agree within 0.25% regardless.
  **Both halves of that caveat ended the same day** — P8e split the unit and `kUrbanIonMscWired`
  is true (RISK V65, V66). The rows above were taken before it and are left as they were taken;
  what the substitution was worth on the stage-1 alpha is +0.023%, well inside the 0.25%.
- The electron path lacks Urban's `extremesmallstep` branch (RISK V62), left off pending a
  measurement of what it does to the gamma gate; the photon and low-energy electron rows above
  carry that too. **The measurement was taken by P8e and the branch is on** (RISK V66): it moves
  the 2,000,000-event gamma gate by 0.0004 pGy on the mean of five seeds, so these rows stand.
- e+- above 100 MeV are scattered by Urban where Geant4 uses WentzelVI, with a transport mean
  free path clamped at 100 MeV (RISK V83: the model and its table are transcribed and oracled,
  and `transport_run.cu` does not survive ptxas with the branch instantiated - V63 again). The
  1 GeV electron row carries that substitution and is 1.67% low; that is the leading candidate
  for the residual rather than a proven attribution.
- The `e- 20 MeV` row deposits only through bremsstrahlung photons (the electron's own range is
  10 cm and the trapezoid starts 19 cm in), so it tests the radiative chain at 0.9% statistics.
- Geant4 was single-threaded by choice: the timing is one core against one GPU, and the EM-only
  loop time is the honest denominator.

---

## The electron rows again, after P14c (2026-09-11)

The 1 GeV electron row above is what the sweep was run to find: docs/RISK.md V64, an e+- range
table that stopped at 100 MeV and clamped, so every electron above it was transported as a
100 MeV electron. P14c rebuilt those tables on Geant4's own grid - 100 eV to 100 TeV, 7 bins per
decade, one per species, `G4LossTableBuilder`'s spline and integration - dispatched
`G4eBremsstrahlungRelModel` above 1 GeV with `G4EmModelManager`'s continuity factor across the
boundary, and fixed a collision/radiative split that was discarding part of every electron's
continuous loss (V82). The three electron beams and the 6 MeV photon gate were re-run from
`phys/electron_hi` with the same script, the same event counts and the same inactivation list.

| beam | events | port before | port after | G4 EM-only | diff before | diff after | sigma before | sigma after |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| gamma 6 MeV | 2,000,000 | 8.5170E-008 | 8.5189E-008 | 8.5615E-008 | -0.52% | -0.50% | -1.8 | -1.7 |
| e- 20 MeV | 500,000 | 4.6289E-009 | 4.6259E-009 | 4.5247E-009 | +2.30% | +2.24% | 1.8 | 1.8 |
| e- 100 MeV | 300,000 | 3.3107E-007 | 3.3174E-007 | 3.3204E-007 | -0.29% | -0.09% | -1.1 | -0.4 |
| **e- 1000 MeV** | 100,000 | 1.1062E-007 | **2.3626E-007** | 2.4027E-007 | **-53.96%** | **-1.67%** | **-154.9** | **-3.7** |

The Geant4 columns are a fresh run of the same configuration rather than the numbers copied
down: the EM-only 1 GeV electron and 6 MeV photon read 2.4027E-007 and 8.5615E-008 today
against 2.4027E-007 and 8.5615E-008 in the table above - the same to the five digits this table
prints, because `ref/run/runb1.bat` seeds Geant4. The reference has not moved under the port.

**The 1 GeV row went from -54% to -1.7%, and the residual has a name.** What is not in this
build is `G4WentzelVIModel` for e+- above 100 MeV: it is transcribed, its transport mean free
path table is compared against Geant4's at 5.1e-16, its dispatch is written - and
`src/host/transport_run.cu` does not survive ptxas with it instantiated, which is docs/RISK.md
V63's wall on the other stepper and V81/V83 here. So an electron above 100 MeV is stepped by
`G4UrbanMscModel` with a transport mean free path clamped at 100 MeV, which is a substitution in
the step length and the deflection of exactly the part of the track this row is sensitive to:
the trapezoid is 6 cm of bone 19 cm inside a 20 x 20 x 30 cm water envelope, so how far
sideways a 1 GeV electron's shower spreads on the way in is most of the answer. That is the
leading candidate for the -1.67% and it is NOT proven to be all of it: the discrete rates are
drawn from the models rather than from `G4VEnergyLossProcess`'s lambda tables and its integral
approach (V78), and Urban's `extremesmallstep` branch is off on the lepton path (V62). The way
to separate them is to re-take this row with the switch flipped, which is what P8e's split of
the transport translation unit makes possible.

**The 6 MeV photon gate did not move.** 8.5170E-008 to 8.5189E-008 Gy over 2,000,000 events, or
425.847 to 425.946 pGy per 10,000 - +0.023%, against a per-run standard error of 0.87 pGy, so
0.11 sigma. It is not zero and it should not be: V82's fix deposits the sub-cut radiative share
of the continuous loss that was being discarded, which is +5.3e-5 of a 1 MeV electron's energy,
and the range table under the gate's few-MeV secondaries changed from 128 linear bins to
Geant4's 85-node spline. Both move the dose up, and the size is what the arithmetic says.

**The 100 MeV electron row improved without being the target**: -0.29% to -0.09%, 1.1 to 0.4
sigma. 100 MeV is the last energy the old table covered without clamping, so what changed there
is the table's construction - spline against linear interpolation, and an inverse-range vector
where there had been a binary search - rather than its ceiling. The 20 MeV row is unchanged
within its 0.9% statistics.

---

## The electron rows a third time, after P14d (2026-09-12)

P14c's section above ends "the way to separate them is to re-take this row with the switch
flipped, which is what P8e's split of the transport translation unit makes possible." P14d
flipped it: `em::kWentzelLeptonMscWired` is `true`, so an e+- above `G4EmParameters::
MscEnergyLimit()` = 100 MeV is stepped by `G4WentzelVIModel` and one below it by
`G4UrbanMscModel`, which is what `G4EmStandardPhysics::ConstructProcess` builds. docs/RISK.md
V95 has the compile, the kernel and the two inversions; this is the dose.

| beam | events | port before | port after | G4 EM-only | diff before | diff after | sigma before | sigma after |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| gamma 6 MeV | 2,000,000 | 8.5189E-008 | 8.5189E-008 | 8.5615E-008 | -0.50% | -0.50% | -1.7 | -1.7 |
| e- 100 MeV | 300,000 | 3.3174E-007 | 3.3174E-007 | 3.3204E-007 | -0.09% | -0.09% | -0.4 | -0.4 |
| **e- 1000 MeV** | 100,000 | 2.3626E-007 | **2.3720E-007** | 2.4027E-007 | -1.67% | **-1.28%** | -3.7 | **-2.9** |

The first two rows are IDENTICAL, not merely consistent - every digit the run prints, and
25,993,577 track-steps either way for the gamma gate. `G4RegionModels::SelectIndex` tests
`e <= lowKineticEnergy[idx]`, so 100 MeV belongs to Urban and `step_lepton`'s branch is a strict
`>`; a 6 MeV photon makes no secondary above 100 MeV and a 100 MeV electron beam starts exactly
at the boundary. A branch that draws no uniform cannot move a seeded run.

**The 100,000-event 1 GeV row is not precise enough to carry the finding, and the reason is the
reference.** Both sides re-taken at 1,000,000 events:

| 1 GeV e-, dose per 10,000 events | | vs Geant4 (1M) |
|---|--:|--:|
| port, WentzelVI off | 23,734.2 ± 24.06 pGy | -0.532%, -3.74 σ |
| **port, WentzelVI on (ships)** | **23,782.0 ± 23.97 pGy** | **-0.332%, -2.33 σ** |
| Geant4 11.1.1 EM-only, 1,000,000 events | 23,861.2 ± 24.04 pGy | — |
| *Geant4 11.1.1 EM-only, 100,000 events (this table's)* | *24,027.3 ± 76.2 pGy* | |

Geant4's two samples of its own configuration differ by -0.69%, which is 2.3 σ of the
100,000-event run's quoted rms on a pair that shares its first 100,000 events. The dose a 1 GeV
electron puts in a 6 cm trapezoid 19 cm inside the envelope is heavy-tailed, so that rms
converges slowly. **Part of the -1.67% recorded above was the reference and not the port**, and
this row wants 1,000,000 events on both sides.

**What the switch was worth: +47.8 pGy, +0.201% of the row, -0.53% to -0.33%.** A fifth of the
deficit, and not all of it. Of the two other terms V83 named, `extremesmallstep` (RISK V62) is
excluded by measurement - forced off, the same 1,000,000-event row moves by 0.1 pGy and 209 of
205,579,159 track-steps - and the discrete rates (RISK V78) are the largest candidate left,
together with the possibility that the residual is the photon rows' own -0.27%/-0.52% arriving
through a shower rather than anything on the lepton path. docs/RISK.md V95.
