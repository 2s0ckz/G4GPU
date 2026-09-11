# Stage 1: the first configuration in which Geant4's answer moves and the port follows

`docs/HADRONIC_PLAN.md` section 4 says every stage of this port is checked against a Geant4 run
with **exactly the processes the port has**, inactivated one at a time as the port gains them.
P1's `*_emonly.mac` macros are stage 0: everything hadronic and every decay switched off, so the
comparison is of electromagnetic transport alone. These are stage 1.

## What stage 1 is

Geant4 QBBC, example B1, with

* every `*Inelastic` process inactivated (P9-P11 are not written),
* `muonNuclear` inactivated (P13),
* the three at-rest captures inactivated - `hBertiniCaptureAtRest`, `hFritiofCaptureAtRest`,
  `muMinusCaptureAtRest` (P12),
* `hBrems`, `hPairProd`, `muBrems`, `muPairProd` inactivated (no `SampleSecondaries` for either
  radiative model - docs/PORTED.md 1.3),
* and **`Decay`, `hadElastic` and `CoulombScat` LEFT ACTIVE**.

`CoulombScat` was inactivated here until P8b wired it (docs/PORTED.md 2.1.6). Nothing is
inactivated on the Geant4 side any more except processes this port does not have, which is what
the plan's staged method asks for (docs/HADRONIC_PLAN.md section 4) and what makes the first
column below the like-for-like one.

The port runs in `HadronicStage::kStage1`, whose one effect is that a stopped pi-, K- or mu-
DECAYS: on the Geant4 side its at-rest capture has just been switched off and `G4Decay` is the
only at-rest process it has left, so both sides decay it. In the final configuration neither
does, and the port refuses the capture by name - see `physics/hadronic/wiring.cuh`.

## Two macros per species, and why the second one is not redundant

`stage1_<species>.mac` is the configuration above. `stage1_<species>_noelastic.mac` is the same
with `hadElastic` inactivated as well.

**Since P8b the FIRST one is the like-for-like**, because the port has `hadElastic` now. The
second is kept, and it is not redundant: the difference between the two columns is what
`hadElastic` is worth for that species and that geometry, and that number is now the size of
what the port GAINED rather than of what it was missing. It is the same measurement read in the
other direction, and keeping it is what lets "the port followed Geant4's move" be checked
against how big the move was. P8's table below has both columns for the state before the wiring;
P8b's has both for the state after it.

## The neutron is a special case and it is not this port's fault

There is no stage-1 configuration for the neutron in which elastic and capture are active and
inelastic is not, **and no UI command can make one**. With `EnableNeutronGeneralProcess = 1` -
which `G4HadronInelasticQBBC::ConstructProcess` sets unconditionally, with no messenger anywhere
in 11.1.1 - the neutron's process manager holds `Transportation`, `Decay` and
`NeutronGeneralProc` and nothing else. Elastic, inelastic and capture are sub-processes inside
that one object, reachable only through its own summed cross-section table, so
`/process/inactivate` can take all three or none.

So `stage1_neutron.mac` inactivates `NeutronGeneralProc`, which is stage 0 for a neutron:
both sides give exactly zero in B1's scoring volume, and that zero is a prediction rather than
an absence (see `neutron_nogeneral.mac`). The neutron's hadronic transport cannot be validated
by a dose comparison until P9-P11 land; until then it is validated by its cross-section table
(bit-exact, `tests/test_particlexs.cu`), its sub-process selection, and its final states
(`tests/test_elastic_models.cu`, `tests/test_capture.cu`). docs/RISK.md V53.

## How to run

    ref\b1hadron\stage1_compare.ps1 -Events 500000

500,000 events per run per side, and the count is part of the measurement: B1's printed rms is
the standard error and scales as 1/sqrt(N), so a 2,000-event run is +/-1.7% and can neither
confirm nor exclude a 3% effect. docs/RISK.md V44 is the half day that cost.

## The table, as measured

`ref\b1hadron\stage1_compare.ps1 -Events 500000`, on branch `phys/wiring`, against
`D:\g4gpu\ref\B1build` (Geant4 11.1.1, serial run manager). 500,000 events per run per side,
27 runs, doses in nGy. The **like-for-like column is "G4 no elastic"**, because the port has
decay and does not have `hadElastic`.

```
species                     port        G4 no elastic      diff   sigma           G4 stage 1      diff   sigma
proton       3,077.2600 +/- 3.2246  3,086.1000 +/- 3.2325    -0.29%     1.9  3,011.3400 +/- 3.2448     2.19%    14.4
alpha       12,355.1000 +/- 13.0230 12,364.4000 +/- 13.0412  -0.08%     0.5 12,336.9000 +/- 13.0077     0.15%     1.0
muon_plus      598.9640 +/- 0.6341    600.4280 +/- 0.6337    -0.24%     1.6    600.4280 +/- 0.6337    -0.24%     1.6
muon_minus     597.6490 +/- 0.6328    576.2410 +/- 0.6112     3.72%    24.3    576.2410 +/- 0.6112     3.72%    24.3
pion_plus      632.8880 +/- 0.6686    634.4790 +/- 0.6682    -0.25%     1.7    608.6010 +/- 0.8880     3.99%    21.8
pion_minus     631.4470 +/- 0.6671    602.1990 +/- 0.6372     4.86%    31.7    589.1720 +/- 0.6863     7.18%    44.2
kaon_plus    1,210.1400 +/- 1.3853  1,206.4900 +/- 1.3480     0.30%     1.9  1,209.1000 +/- 1.3917     0.09%     0.5
kaon_minus   1,207.1000 +/- 1.3822  1,099.0400 +/- 1.2340     9.83%    58.3  1,106.9500 +/- 1.3246     9.05%    52.3
neutron          0.0000 +/- 0.0000                    -         -       -      0.0000 +/- 0.0000     0.00%     0.0
```

### Reading it

**The five positive-or-neutral species agree: 0.5 to 1.9 sigma.** proton 1.9, alpha 0.5, mu+
1.6, pi+ 1.7, K+ 1.9 - every one of them with `Decay` active on the Geant4 side, which is the
process P8 added. This is the row of the plan P8 exists for: Geant4's answer moved from stage 0
and the port followed.

**The three negatives do not, and the offset is not P8's.** mu- 3.72%, pi- 4.86%, K- 9.83%, all
with the port HIGH. That is docs/RISK.md V44 - the negative of each charge pair had its range
table interpolated by a different rule in this port's EM code - and P14 has the fix on its own
branch, after which mu-, pi- and K- agree with Geant4's EM-only runs to 0.49, 0.04 and 0.46
sigma. These runs are on main's EM code, so the V44 term is present by construction and is
listed here rather than chased. Compare the pairs in the port's own column: pi+ 632.89 against
pi- 631.45 and K+ 1210.14 against K- 1207.10 - the port's two charges agree with each other to
0.2%, as the physics of stage 1 says they should, while Geant4's differ by 5% and 9%. The
disagreement is one-sided.

**What `hadElastic` is worth**, from the two Geant4 columns - the number the next package is
judged against, measured rather than asserted:

```
proton     -2.42%      pi+   -4.08%      K+   +0.22%      alpha  -0.22%
                       pi-   -2.16%      K-   +0.72%      mu+-    0.00%
```

mu+ and mu- are **exactly** 0.00% - 600.4280 and 576.2410 in both columns, rms included,
bit-identical runs. That is not a coincidence to be explained away, it is the source being
right: `G4HadronElasticPhysics::ConstructProcess` registers `G4HadronElasticProcess` for p, n,
pi+-, kaons, the light ions and the anti-nuclei, and **for no lepton**. A muon has no
`hadElastic` to inactivate, `/process/inactivate hadElastic` is silently ignored for it
(docs/RISK.md V43's trap, harmless here), and the process dump below confirms it: the muon rows
list `muonNuclear` and no `hadElastic`. So for mu+ the port's stage-1 physics is COMPLETE, and
its 1.6 sigma is a like-for-like with nothing missing.

The proton's -2.42% and the pion's -4.08% are the gap that remains, and they are negative:
`hadElastic` moves dose OUT of B1's scoring volume in this geometry.

### The stage, recorded by what ran

`/particle/process/dump` output for all nine species is printed by the script after the table.
The three inactivations that make stage 1 what it is appear as `InActive` in it:
`hBertiniCaptureAtRest` on pi- and K-, `muMinusCaptureAtRest` on mu-, every `*Inelastic`, and
`NeutronGeneralProc` on the neutron. `Decay` and `hadElastic` are `Active` wherever the species
has them.
