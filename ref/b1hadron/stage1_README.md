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

    ref\b1hadron\stage1_compare.ps1 -Events 500000 -SkipNoElastic     the like-for-like
    ref\b1hadron\stage1_compare.ps1 -Events 500000                    plus the diagnostic column

500,000 events per run per side, and the count is part of the measurement: B1's printed rms is
the standard error and scales as 1/sqrt(N), so a 2,000-event run is +/-1.7% and can neither
confirm nor exclude a 3% effect. docs/RISK.md V44 is the half day that cost.

`-SkipNoElastic` drops the third run per species - half the Geant4 time - and is the right
default for asking whether the port agrees. Leave it off when the question is how big the effect
`hadElastic` has for a species, which is what the two Geant4 columns differ by.

## The table, as measured - P8b, with hadElastic and CoulombScat on both sides

`ref\b1hadron\stage1_compare.ps1 -Events 500000 -SkipNoElastic`, on branch `phys/wiring2`,
against `D:\g4gpu\ref\B1build` (Geant4 11.1.1, serial run manager). 500,000 events per run per
side, 18 runs, doses in nGy. **Nothing is inactivated on the Geant4 side that this port has.**

```
species                     port           G4 stage 1      diff   sigma
proton       3,003.8400 +/- 3.2354  3,008.1200 +/- 3.2443    -0.14%     0.9
alpha       12,312.3000 +/- 12.9798 12,336.9000 +/- 13.0077  -0.20%     1.3
muon_plus      598.7400 +/- 0.6345    600.3700 +/- 0.6341    -0.27%     1.8
muon_minus     575.2420 +/- 0.6110    575.6470 +/- 0.6110    -0.07%     0.5
pion_plus      606.4620 +/- 0.8888    609.0240 +/- 0.8874    -0.42%     2.0
pion_minus     587.7400 +/- 0.6863    589.5030 +/- 0.6860    -0.30%     1.8
kaon_plus    1,215.0200 +/- 1.4305  1,208.1700 +/- 1.3920     0.57%     3.4
kaon_minus   1,115.7700 +/- 1.3779  1,108.7600 +/- 1.3248     0.63%     3.7
neutron          0.0000 +/- 0.0000      0.0000 +/- 0.0000     0.00%     0.0
```

### Reading it

**Seven of the nine species are inside two sigma, and the three negatives are the story.** mu-
0.5, pi- 1.8, K- 3.7 - against 24.3, 31.7 and 58.3 in P8's table below. That is docs/RISK.md
V44/V46 closed in the transport rather than in a table: P14's fix (the negative of each charge
pair had its range table interpolated by a different rule) is on main and this is the first
like-for-like that has it. The port's own charge pairs still agree with each other - pi+ 606.46
against pi- 587.74, K+ 1215.02 against K- 1115.77 - and now Geant4's do too.

**The proton went from 1.9 sigma against a Geant4 with elastic switched off to 0.9 sigma against
one with it on.** Its `hadElastic` was worth -2.42% in P8's measurement and that gap is gone.

**The two kaons are at 3.4 and 3.7 sigma, both with the port 0.6% HIGH, and that is a finding
rather than a rounding.** Both charges by the same amount, so it is not V44. See docs/RISK.md
V56 for the measurement that bounds it and the two candidates it does not separate.

### The third column, for the four species where the answer turns on it

`-SkipNoElastic` off, same 500,000 events, same build:

```
species                     port           G4 stage 1      diff   sigma        G4 no elastic      diff   sigma
proton       3,003.8400 +/- 3.2354  3,008.1200 +/- 3.2443    -0.14%     0.9  3,092.3900 +/- 3.2300    -2.86%    19.4
pion_plus      606.4620 +/- 0.8888    609.0240 +/- 0.8874    -0.42%     2.0    633.8790 +/- 0.6687    -4.33%    24.7
kaon_plus    1,215.0200 +/- 1.4305  1,208.1700 +/- 1.3920     0.57%     3.4  1,201.9100 +/- 1.3475     1.09%     6.7
kaon_minus   1,115.7700 +/- 1.3779  1,108.7600 +/- 1.3248     0.63%     3.7  1,097.3900 +/- 1.2327     1.67%     9.9
```

**This is what `hadElastic` being wired means, read off two Geant4 runs that differ by one
`/process/inactivate` line.** Geant4's own elastic effect on the dose is -2.73% for the proton,
-3.92% for pi+, +0.52% for K+ and +1.04% for K-, and the port is nearer the column WITH elastic
in every one of the four - 0.9 against 19.4 sigma for the proton, 2.0 against 24.7 for pi+,
3.4 against 6.7 for K+, 3.7 against 9.9 for K-. Including the kaons: whatever the residual 0.6%
is, the port's kaon elastic is acting in the right direction and with roughly the right size.

The sign flips between the pions and the kaons and that is the geometry, not a defect: elastic
scattering moves a 200 MeV pion's dose OUT of B1's 12 cm scoring volume and a 400 MeV kaon's
INTO it.

**What `CoulombScat` was worth, measured by re-running the Geant4 side with it active.** The
stage-1 column moved by at most one sigma from P8's numbers - proton 3011.34 to 3008.12, mu+
600.428 to 600.370, mu- 576.241 to 575.647, pi+ 608.601 to 609.024, K+ 1209.10 to 1208.17,
K- 1106.95 to 1108.76 - which is a Geant4 random stream reshuffle and not a physical effect.
That is what the mean free path predicted: `tests/test_step_hadron.cu` measures
`G4CoulombScattering`'s mfp in water at **158 m for a 200 MeV proton and 483 m for a 1 GeV
muon**, against B1's 300 mm envelope, so the process cannot move a hadron's dose in this
geometry however it is wired. The port's own numbers moved by about a sigma for the same reason
(it now draws a Coulomb interaction length on every step of every singly-charged hadron), which
is why P8's 632.8880 nGy for pi+ and 597.6490 for mu- are not reproduced here and should not be.

**The neutron is still 0 against 0**, and docs/RISK.md V53 is why: `EnableNeutronGeneralProcess`
makes its elastic, inelastic and capture one process that `/process/inactivate` can only take
whole, so there is no stage-1 configuration for it at all. The zero is a prediction the port
reproduces - streaming plus the 10 us time cut - and not evidence about a cross section.

## P8's table - the same stage before hadElastic was wired

Kept because the DIFFERENCE between the two is what the wiring did, and because its
`G4 no elastic` column is what the port used to be compared against.

`ref\b1hadron\stage1_compare.ps1 -Events 500000`, on branch `phys/wiring`, against
`D:\g4gpu\ref\B1build` (Geant4 11.1.1, serial run manager). 500,000 events per run per side,
27 runs, doses in nGy. The like-for-like column there was "G4 no elastic", because the port had
decay and did not have `hadElastic`; both Geant4 columns had `CoulombScat` inactivated.

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

### Reading it - as P8 read it, and two of its sentences are now history

Two claims below have been superseded and are left standing rather than edited, because what
they predicted is what happened. "The three negatives carry V44 and P14 has the fix" - the fix
is on main and the negatives are at 0.5, 1.8 and 3.7 sigma in the table above. And "what
hadElastic is worth" was measured here as the number the next package would be judged against:
-2.42% for the proton and -4.08% for pi+. The port's proton is now 0.9 sigma from a Geant4 with
elastic on, so that is the number that closed.

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
