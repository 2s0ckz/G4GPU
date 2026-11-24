# P43: lead (and uranium) in the B1 trapezoid

The port (this package's engine, 29 photoelectric shells) against Geant4 11.1.1 QBBC as it ships,
per event, by `tools/p27_sweep.ps1 -Report`. Geant4's lead rows are P27's (docs/sweep/p27/runs.csv);
every run of this package is in docs/sweep/p27/runs_p43.csv. docs/RISK.md V356 reads them. The
port's rates ran beside other packages' compiles and sweeps on a shared machine: quoted, not
headlined. A dose with "abandoned" in its ledger is of a run the drain loop's bound stopped part-way
(V356): neutron 1 and 10 GeV, He3 10 GeV/n.

# Tier 2: 21 species x 300 MeV, 1, 10 GeV (per nucleon), G4_Pb

﻿## doses

| row | port (Gy an event) | G4 box (Gy an event) | diff | sigma | port events | G4 events | ledgers |
|---|--:|--:|--:|--:|--:|--:|---|
| gamma_300_G4_Pb | 2.2919E-012 +/- 1.7E-015 | 2.2947E-012 +/- 3.0E-015 | -0.13% | -0.8 | 1,000,000 | 327,360 | 0 |
| gamma_1000_G4_Pb | 7.2774E-012 +/- 8.0E-015 | 7.2718E-012 +/- 1.7E-014 | +0.08% | +0.3 | 500,000 | 108,880 | 0 |
| gamma_10000_G4_Pb | - | 5.8086E-011 +/- 1.7E-013 | - | - | - | 80,000 | 0 |
| e-_300_G4_Pb | 2.1113E-012 +/- 2.1E-015 | 2.1149E-012 +/- 2.9E-015 | -0.17% | -1.0 | 500,000 | 279,520 | 0 |
| e-_1000_G4_Pb | 7.4399E-012 +/- 1.1E-014 | 7.4499E-012 +/- 1.8E-014 | -0.13% | -0.5 | 250,000 | 95,400 | 0 |
| e-_10000_G4_Pb | - | 6.3877E-011 +/- 1.8E-013 | - | - | - | 80,000 | 0 |
| e+_300_G4_Pb | 2.1284E-012 +/- 3.8E-016 | 2.1279E-012 +/- 3.0E-015 | +0.02% | +0.2 | 16,000,000 | 263,400 | 0 |
| e+_1000_G4_Pb | 7.4531E-012 +/- 7.8E-015 | 7.4480E-012 +/- 1.8E-014 | +0.07% | +0.3 | 500,000 | 95,640 | 0 |
| e+_10000_G4_Pb | 6.3919E-011 +/- 1.4E-013 | 6.3901E-011 +/- 1.8E-013 | +0.03% | +0.1 | 125,000 | 80,000 | 0 |
| mu-_300_G4_Pb | 6.8952E-013 +/- 2.6E-016 | 6.8874E-013 +/- 3.8E-016 | +0.11% | +1.7 | 4,000,000 | 1,802,640 | 0 |
| mu-_1000_G4_Pb | 7.7362E-013 +/- 3.0E-016 | 7.7522E-013 +/- 5.1E-016 | -0.21% | -2.7 | 4,000,000 | 1,368,960 | 0 |
| mu-_10000_G4_Pb | 9.5209E-013 +/- 5.5E-016 | 9.8231E-013 +/- 1.1E-015 | -3.08% | **-23.9** | 4,000,000 | 1,127,720 | 0 |
| mu+_300_G4_Pb | 6.9567E-013 +/- 2.6E-016 | 6.9487E-013 +/- 3.8E-016 | +0.12% | +1.7 | 4,000,000 | 1,828,160 | 0 |
| mu+_1000_G4_Pb | 7.6739E-013 +/- 2.1E-016 | 7.6884E-013 +/- 5.0E-016 | -0.19% | -2.7 | 8,000,000 | 1,368,960 | 0 |
| mu+_10000_G4_Pb | 9.5424E-013 +/- 3.9E-016 | 9.8488E-013 +/- 1.2E-015 | -3.11% | **-25.0** | 8,000,000 | 1,094,120 | 0 |
| pi+_300_G4_Pb | 8.6054E-013 +/- 8.5E-016 | 8.5907E-013 +/- 2.0E-015 | +0.17% | +0.7 | 1,600,000 | 280,680 | 0 |
| pi+_1000_G4_Pb | 1.5812E-012 +/- 1.8E-015 | 1.5765E-012 +/- 6.0E-015 | +0.30% | +0.8 | 1,600,000 | 152,000 | 0 |
| pi+_10000_G4_Pb | 8.7805E-012 +/- 2.7E-014 | 8.7965E-012 +/- 7.9E-014 | -0.18% | -0.2 | 400,000 | 47,280 | no final state 50: 17 88211.5 MeV a charged hadron's inelastic interaction with no final state [SIZE - do not add to the WHY group]; 17 88211.5 MeV WHY: bert::apply_yourself refused by name (P10); 16 84184.4 MeV WHERE: of the refused in-flight interactions above, the kinetic energy deposited INSIDE a scoring volume by the disposal - part of the scored dose; do not add; abandoned 1 |
| pi-_300_G4_Pb | 7.5262E-013 +/- 7.1E-016 | 7.4307E-013 +/- 1.7E-015 | +1.29% | **+5.2** | 1,600,000 | 268,920 | 0 |
| pi-_1000_G4_Pb | 1.4586E-012 +/- 1.7E-015 | 1.4494E-012 +/- 5.3E-015 | +0.63% | +1.7 | 1,600,000 | 162,040 | no final state 3: 1 282.049 MeV a charged hadron's inelastic interaction with no final state [SIZE - do not add to the WHY group]; 1 282.049 MeV WHY: bert::apply_yourself refused by name (P10); 1 282.049 MeV WHERE: of the refused in-flight interactions above, the kinetic energy deposited INSIDE a scoring volume by the disposal - part of the scored dose; do not add |
| pi-_10000_G4_Pb | 8.8834E-012 +/- 2.7E-014 | 8.9041E-012 +/- 8.0E-014 | -0.23% | -0.2 | 400,000 | 46,720 | no final state 21: 7 42680.3 MeV a charged hadron's inelastic interaction with no final state [SIZE - do not add to the WHY group]; 7 42680.3 MeV WHY: bert::apply_yourself refused by name (P10); 7 42680.3 MeV WHERE: of the refused in-flight interactions above, the kinetic energy deposited INSIDE a scoring volume by the disposal - part of the scored dose; do not add |
| kaon+_300_G4_Pb | 1.2367E-012 +/- 1.1E-015 | 1.2291E-012 +/- 1.5E-015 | +0.62% | **+4.2** | 1,600,000 | 845,080 | no final state 113: 38 1329.89 MeV a charged hadron's inelastic interaction with no final state [SIZE - do not add to the WHY group]; 38 1329.89 MeV WHY: bert::apply_yourself refused by name (P10); 37 1321.14 MeV WHERE: of the refused in-flight interactions above, the kinetic energy deposited INSIDE a scoring volume by the disposal - part of the scored dose; do not add |
| kaon+_1000_G4_Pb | 1.5939E-012 +/- 1.9E-015 | 1.5284E-012 +/- 4.9E-015 | +4.29% | **+12.4** | 1,600,000 | 210,560 | no final state 177: 61 7497.19 MeV a charged hadron's inelastic interaction with no final state [SIZE - do not add to the WHY group]; 61 7497.19 MeV WHY: bert::apply_yourself refused by name (P10); 55 6933.85 MeV WHERE: of the refused in-flight interactions above, the kinetic energy deposited INSIDE a scoring volume by the disposal - part of the scored dose; do not add |
| kaon+_10000_G4_Pb | 7.6456E-012 +/- 2.3E-014 | 7.6299E-012 +/- 6.4E-014 | +0.21% | +0.2 | 400,000 | 54,200 | no final state 51: 17 20733.5 MeV a charged hadron's inelastic interaction with no final state [SIZE - do not add to the WHY group]; 17 20733.5 MeV WHY: bert::apply_yourself refused by name (P10); 17 20733.5 MeV WHERE: of the refused in-flight interactions above, the kinetic energy deposited INSIDE a scoring volume by the disposal - part of the scored dose; do not add |
| kaon-_300_G4_Pb | 1.5154E-012 +/- 2.0E-015 | 1.4566E-012 +/- 4.1E-015 | +4.04% | **+12.8** | 800,000 | 169,800 | no final state 1192: 404 82973.9 MeV a charged hadron's inelastic interaction with no final state [SIZE - do not add to the WHY group]; 404 82973.9 MeV WHY: bert::apply_yourself refused by name (P10); 384 77687.8 MeV WHERE: of the refused in-flight interactions above, the kinetic energy deposited INSIDE a scoring volume by the disposal - part of the scored dose; do not add |
| kaon-_1000_G4_Pb | 2.0379E-012 +/- 3.3E-015 | 1.9170E-012 +/- 7.3E-015 | +6.30% | **+15.1** | 800,000 | 141,600 | no final state 871: 294 249616 MeV a charged hadron's inelastic interaction with no final state [SIZE - do not add to the WHY group]; 294 249616 MeV WHY: bert::apply_yourself refused by name (P10); 283 239873 MeV WHERE: of the refused in-flight interactions above, the kinetic energy deposited INSIDE a scoring volume by the disposal - part of the scored dose; do not add |
| kaon-_10000_G4_Pb | 9.7221E-012 +/- 2.8E-014 | 9.6630E-012 +/- 8.4E-014 | +0.61% | +0.7 | 400,000 | 44,360 | no final state 102: 34 49180.5 MeV a charged hadron's inelastic interaction with no final state [SIZE - do not add to the WHY group]; 34 49180.5 MeV WHY: bert::apply_yourself refused by name (P10); 34 49180.5 MeV WHERE: of the refused in-flight interactions above, the kinetic energy deposited INSIDE a scoring volume by the disposal - part of the scored dose; do not add |
| proton_300_G4_Pb | 1.5250E-012 +/- 1.1E-015 | 1.5211E-012 +/- 2.2E-015 | +0.26% | +1.6 | 1,600,000 | 450,680 | 0 |
| proton_1000_G4_Pb | 1.3108E-012 +/- 1.3E-015 | 1.3017E-012 +/- 3.9E-015 | +0.70% | +2.2 | 1,600,000 | 184,440 | 0 |
| proton_10000_G4_Pb | 7.6979E-012 +/- 2.2E-014 | 7.6216E-012 +/- 6.2E-014 | +1.00% | +1.2 | 400,000 | 49,720 | no final state 3: 1 2042.91 MeV a charged hadron's inelastic interaction with no final state [SIZE - do not add to the WHY group]; 1 2042.91 MeV WHY: bert::apply_yourself refused by name (P10); 1 2042.91 MeV WHERE: of the refused in-flight interactions above, the kinetic energy deposited INSIDE a scoring volume by the disposal - part of the scored dose; do not add |
| anti_proton_300_G4_Pb | 4.0973E-012 +/- 5.5E-015 | 4.0956E-012 +/- 1.5E-014 | +0.04% | +0.1 | 800,000 | 100,200 | no final state 8: 3 427.39 MeV a charged hadron's inelastic interaction with no final state [SIZE - do not add to the WHY group]; 3 427.39 MeV WHY: bert::apply_yourself refused by name (P10); 2 378.82 MeV WHERE: of the refused in-flight interactions above, the kinetic energy deposited INSIDE a scoring volume by the disposal - part of the scored dose; do not add |
| anti_proton_1000_G4_Pb | 3.1931E-012 +/- 5.8E-015 | 3.1661E-012 +/- 1.5E-014 | +0.85% | +1.7 | 800,000 | 121,240 | no final state 20: 7 2026.83 MeV a charged hadron's inelastic interaction with no final state [SIZE - do not add to the WHY group]; 7 2026.83 MeV WHY: bert::apply_yourself refused by name (P10); 6 1404.51 MeV WHERE: of the refused in-flight interactions above, the kinetic energy deposited INSIDE a scoring volume by the disposal - part of the scored dose; do not add |
| anti_proton_10000_G4_Pb | 1.0704E-011 +/- 3.0E-014 | 1.0628E-011 +/- 9.4E-014 | +0.72% | +0.8 | 400,000 | 40,240 | no final state 29: 6 11731 MeV a charged hadron's inelastic interaction with no final state [SIZE - do not add to the WHY group]; 6 11731 MeV WHY: bert::apply_yourself refused by name (P10); 5 8694.49 MeV WHERE: of the refused in-flight interactions above, the kinetic energy deposited INSIDE a scoring volume by the disposal - part of the scored dose; do not add; 12 35438 MeV BY NAME: a charm or bottom hadron made by FTFP or a decay - no ParticleType, so no track (P23a leaves them refused; their decay tables are P23b's) |
| neutron_300_G4_Pb | 3.4795E-013 +/- 5.5E-016 | 3.3541E-013 +/- 1.5E-015 | +3.74% | **+7.8** | 1,600,000 | 209,200 | abandoned 2 |
| neutron_1000_G4_Pb | 8.6911E-013 +/- 1.2E-015 | 1.1728E-012 +/- 5.9E-015 | -25.89% | **-50.4** | 1,600,000 | 101,120 | abandoned 6697339 |
| neutron_10000_G4_Pb | 2.9063E-012 +/- 7.2E-015 | 1.2182E-011 +/- 1.0E-013 | -76.14% | **-90.4** | 800,000 | 29,000 | no final state 9: 3 6585.03 MeV a charged hadron's inelastic interaction with no final state [SIZE - do not add to the WHY group]; 3 6585.03 MeV WHY: bert::apply_yourself refused by name (P10); 3 6585.03 MeV WHERE: of the refused in-flight interactions above, the kinetic energy deposited INSIDE a scoring volume by the disposal - part of the scored dose; do not add; abandoned 6698383 |
| anti_neutron_300_G4_Pb | 2.1653E-012 +/- 3.3E-015 | 2.1454E-012 +/- 1.1E-014 | +0.93% | +1.8 | 1,600,000 | 150,960 | no final state 24: 8 1669.48 MeV a charged hadron's inelastic interaction with no final state [SIZE - do not add to the WHY group]; 8 1669.48 MeV WHY: bert::apply_yourself refused by name (P10); 8 1669.48 MeV WHERE: of the refused in-flight interactions above, the kinetic energy deposited INSIDE a scoring volume by the disposal - part of the scored dose; do not add |
| anti_neutron_1000_G4_Pb | 2.7714E-012 +/- 2.9E-015 | 2.7416E-012 +/- 1.5E-014 | +1.09% | +2.0 | 3,200,000 | 125,000 | no final state 87: 29 5963.13 MeV a charged hadron's inelastic interaction with no final state [SIZE - do not add to the WHY group]; 29 5963.13 MeV WHY: bert::apply_yourself refused by name (P10); 29 5963.13 MeV WHERE: of the refused in-flight interactions above, the kinetic energy deposited INSIDE a scoring volume by the disposal - part of the scored dose; do not add |
| anti_neutron_10000_G4_Pb | 1.0150E-011 +/- 1.5E-014 | 9.9522E-012 +/- 9.2E-014 | +1.99% | +2.1 | 1,600,000 | 40,760 | no final state 163: 39 52041.2 MeV a charged hadron's inelastic interaction with no final state [SIZE - do not add to the WHY group]; 39 52041.2 MeV WHY: bert::apply_yourself refused by name (P10); 37 51107.7 MeV WHERE: of the refused in-flight interactions above, the kinetic energy deposited INSIDE a scoring volume by the disposal - part of the scored dose; do not add; 48 132281 MeV BY NAME: a charm or bottom hadron made by FTFP or a decay - no ParticleType, so no track (P23a leaves them refused; their decay tables are P23b's) |
| deuteron_300_G4_Pb | 1.3869E-012 +/- 1.4E-015 | 1.3834E-012 +/- 2.8E-015 | +0.25% | +1.1 | 800,000 | 209,880 | 0 |
| deuteron_1000_G4_Pb | 2.0632E-012 +/- 3.2E-015 | 2.0228E-012 +/- 9.8E-015 | +2.00% | **+3.9** | 800,000 | 82,600 | 0 |
| deuteron_10000_G4_Pb | 1.4902E-011 +/- 5.4E-014 | 1.4480E-011 +/- 1.4E-013 | +2.91% | +2.8 | 200,000 | 29,360 | no final state 6: 2 1473.97 MeV a charged hadron's inelastic interaction with no final state [SIZE - do not add to the WHY group]; 2 1473.97 MeV WHY: bert::apply_yourself refused by name (P10); 2 1473.97 MeV WHERE: of the refused in-flight interactions above, the kinetic energy deposited INSIDE a scoring volume by the disposal - part of the scored dose; do not add |
| triton_300_G4_Pb | 1.5698E-012 +/- 1.7E-015 | 1.5533E-012 +/- 4.1E-015 | +1.06% | **+3.7** | 800,000 | 140,720 | 0 |
| triton_1000_G4_Pb | 2.8181E-012 +/- 4.4E-015 | 2.8239E-012 +/- 1.6E-014 | -0.21% | -0.4 | 800,000 | 62,160 | 0 |
| triton_10000_G4_Pb | 2.2229E-011 +/- 7.6E-014 | 2.1777E-011 +/- 2.1E-013 | +2.08% | +2.0 | 200,000 | 24,680 | no final state 25: 8 11613.7 MeV a charged hadron's inelastic interaction with no final state [SIZE - do not add to the WHY group]; 8 11613.7 MeV WHY: bert::apply_yourself refused by name (P10); 1 0.2318 MeV a stopped negative hadron's at-rest capture with no final state (stopping::at_rest; P12's own refusals); 8 11613.7 MeV WHERE: of the refused in-flight interactions above, the kinetic energy deposited INSIDE a scoring volume by the disposal - part of the scored dose; do not add; abandoned 1 |
| He3_300_G4_Pb | 4.2440E-012 +/- 4.2E-015 | 4.2427E-012 +/- 8.6E-015 | +0.03% | +0.1 | 800,000 | 189,680 | 0 |
| He3_1000_G4_Pb | 4.0355E-012 +/- 4.6E-015 | 3.9989E-012 +/- 1.6E-014 | +0.92% | +2.1 | 800,000 | 59,680 | 0 |
| He3_10000_G4_Pb | 1.7446E-011 +/- 4.6E-014 | 2.2739E-011 +/- 2.2E-013 | -23.28% | **-23.5** | 400,000 | 21,960 | no final state 12: 1 4753.69 MeV a neutron's inelastic interaction with no final state [SIZE - do not add to the WHY group]; 3 8191.06 MeV a charged hadron's inelastic interaction with no final state [SIZE - do not add to the WHY group]; 4 12944.8 MeV WHY: bert::apply_yourself refused by name (P10); 4 12944.8 MeV WHERE: of the refused in-flight interactions above, the kinetic energy deposited INSIDE a scoring volume by the disposal - part of the scored dose; do not add; abandoned 6687796 |
| alpha_300_G4_Pb | 5.3854E-012 +/- 5.8E-015 | 5.3921E-012 +/- 1.4E-014 | -0.12% | -0.4 | 800,000 | 131,280 | abandoned 1 |
| alpha_1000_G4_Pb | 4.7134E-012 +/- 7.9E-015 | 4.6784E-012 +/- 2.4E-014 | +0.75% | +1.4 | 400,000 | 44,160 | no final state 1: 1 2888.56 MeV an ion's delta-ray channel (G4ionIonisation above ~147 MeV/u) - counted PER STEP, not per interaction |
| alpha_10000_G4_Pb | 2.9420E-011 +/- 9.2E-014 | 2.9300E-011 +/- 2.8E-013 | +0.41% | +0.4 | 200,000 | 21,160 | no final state 29: 2 5589.5 MeV a neutron's inelastic interaction with no final state [SIZE - do not add to the WHY group]; 8 9820.25 MeV a charged hadron's inelastic interaction with no final state [SIZE - do not add to the WHY group]; 10 15409.8 MeV WHY: bert::apply_yourself refused by name (P10); 9 13374.8 MeV WHERE: of the refused in-flight interactions above, the kinetic energy deposited INSIDE a scoring volume by the disposal - part of the scored dose; do not add |
| C12_300_G4_Pb | 3.8355E-012 +/- 1.4E-014 | 3.8119E-012 +/- 2.1E-014 | +0.62% | +0.9 | 200,000 | 91,560 | no final state 79: 79 58506.7 MeV an ion's delta-ray channel (G4ionIonisation above ~147 MeV/u) - counted PER STEP, not per interaction |
| C12_1000_G4_Pb | 2.0174E-011 +/- 3.9E-014 | 2.0046E-011 +/- 1.3E-013 | +0.64% | +0.9 | 200,000 | 16,880 | no final state 280: 280 751304 MeV an ion's delta-ray channel (G4ionIonisation above ~147 MeV/u) - counted PER STEP, not per interaction |
| C12_10000_G4_Pb | 9.7682E-011 +/- 3.1E-013 | 9.7356E-011 +/- 9.3E-013 | +0.33% | +0.3 | 100,000 | 10,800 | no final state 31: 1 4460.36 MeV a neutron's inelastic interaction with no final state [SIZE - do not add to the WHY group]; 8 19952.8 MeV a charged hadron's inelastic interaction with no final state [SIZE - do not add to the WHY group]; 2 77136.8 MeV an ion's delta-ray channel (G4ionIonisation above ~147 MeV/u) - counted PER STEP, not per interaction; 9 24413.2 MeV WHY: bert::apply_yourself refused by name (P10); 9 24413.2 MeV WHERE: of the refused in-flight interactions above, the kinetic energy deposited INSIDE a scoring volume by the disposal - part of the scored dose; do not add; 2 6644.66 MeV BY NAME: a charm or bottom hadron made by FTFP or a decay - no ParticleType, so no track (P23a leaves them refused; their decay tables are P23b's) |
| O16_300_G4_Pb | 2.5102E-012 +/- 1.9E-014 | 2.4855E-012 +/- 2.0E-014 | +0.99% | +0.9 | 100,000 | 82,680 | no final state 9: 9 6398.41 MeV an ion's delta-ray channel (G4ionIonisation above ~147 MeV/u) - counted PER STEP, not per interaction |
| O16_1000_G4_Pb | 3.2864E-011 +/- 6.6E-014 | 3.2779E-011 +/- 2.7E-013 | +0.26% | +0.3 | 200,000 | 12,200 | no final state 152: 152 379801 MeV an ion's delta-ray channel (G4ionIonisation above ~147 MeV/u) - counted PER STEP, not per interaction; abandoned 1 |
| O16_10000_G4_Pb | 1.2807E-010 +/- 3.8E-013 | 1.2633E-010 +/- 1.2E-012 | +1.37% | +1.4 | 100,000 | 9,480 | no final state 48: 1 3687.5 MeV a neutron's inelastic interaction with no final state [SIZE - do not add to the WHY group]; 14 29862.9 MeV a charged hadron's inelastic interaction with no final state [SIZE - do not add to the WHY group]; 3 1806.74 MeV an ion's delta-ray channel (G4ionIonisation above ~147 MeV/u) - counted PER STEP, not per interaction; 15 33550.4 MeV WHY: bert::apply_yourself refused by name (P10); 15 33550.4 MeV WHERE: of the refused in-flight interactions above, the kinetic energy deposited INSIDE a scoring volume by the disposal - part of the scored dose; do not add |
| Si28_300_G4_Pb | 1.5610E-012 +/- 1.0E-014 | 1.5453E-012 +/- 1.3E-014 | +1.02% | +0.9 | 200,000 | 113,320 | no final state 12: 12 9640.68 MeV an ion's delta-ray channel (G4ionIonisation above ~147 MeV/u) - counted PER STEP, not per interaction; abandoned 1 |
| Si28_1000_G4_Pb | 8.7328E-011 +/- 2.1E-013 | 8.7455E-011 +/- 8.7E-013 | -0.15% | -0.1 | 200,000 | 11,760 | no final state 659: 83 1.44493e+06 MeV a charged hadron's inelastic interaction with no final state [SIZE - do not add to the WHY group]; 410 918016 MeV an ion's delta-ray channel (G4ionIonisation above ~147 MeV/u) - counted PER STEP, not per interaction; 83 1.44493e+06 MeV WHY: the Binary cascade or the light-ion reaction refused for a reason that is not the cascade arm - a tripwire on the model table, a final state past its capacity (since P20, 0.2% of Fe56 on Fe56 at 1 GeV/n), or Propagate1H1's null Decay(); 83 1.44493e+06 MeV WHERE: of the refused in-flight interactions above, the kinetic energy deposited INSIDE a scoring volume by the disposal - part of the scored dose; do not add |
| Si28_10000_G4_Pb | 2.3533E-010 +/- 8.5E-013 | 2.3486E-010 +/- 2.2E-012 | +0.20% | +0.2 | 50,000 | 7,240 | no final state 54: 1 2695.63 MeV a neutron's inelastic interaction with no final state [SIZE - do not add to the WHY group]; 17 22787.2 MeV a charged hadron's inelastic interaction with no final state [SIZE - do not add to the WHY group]; 18 25482.8 MeV WHY: bert::apply_yourself refused by name (P10); 18 25482.8 MeV WHERE: of the refused in-flight interactions above, the kinetic energy deposited INSIDE a scoring volume by the disposal - part of the scored dose; do not add |
| Fe56_300_G4_Pb | 1.5827E-012 +/- 1.1E-014 | 1.5416E-012 +/- 1.3E-014 | +2.67% | +2.4 | 200,000 | 141,520 | no final state 8: 8 5838.35 MeV an ion's delta-ray channel (G4ionIonisation above ~147 MeV/u) - counted PER STEP, not per interaction |
| Fe56_1000_G4_Pb | 1.6497E-010 +/- 4.4E-013 | 1.6615E-010 +/- 1.6E-012 | -0.71% | -0.7 | 100,000 | 7,240 | no final state 840: 270 5.04664e+06 MeV a charged hadron's inelastic interaction with no final state [SIZE - do not add to the WHY group]; 30 201609 MeV an ion's delta-ray channel (G4ionIonisation above ~147 MeV/u) - counted PER STEP, not per interaction; 270 5.04664e+06 MeV WHY: the Binary cascade or the light-ion reaction refused for a reason that is not the cascade arm - a tripwire on the model table, a final state past its capacity (since P20, 0.2% of Fe56 on Fe56 at 1 GeV/n), or Propagate1H1's null Decay(); 270 5.04664e+06 MeV WHERE: of the refused in-flight interactions above, the kinetic energy deposited INSIDE a scoring volume by the disposal - part of the scored dose; do not add |
| Fe56_10000_G4_Pb | 4.9867E-010 +/- 1.6E-012 | 5.0479E-010 +/- 4.8E-012 | -1.21% | -1.2 | 50,000 | 5,600 | no final state 81: 6 26620.9 MeV a neutron's inelastic interaction with no final state [SIZE - do not add to the WHY group]; 18 22347.7 MeV a charged hadron's inelastic interaction with no final state [SIZE - do not add to the WHY group]; 7 196496 MeV an ion's delta-ray channel (G4ionIonisation above ~147 MeV/u) - counted PER STEP, not per interaction; 24 48968.6 MeV WHY: bert::apply_yourself refused by name (P10); 24 48968.6 MeV WHERE: of the refused in-flight interactions above, the kinetic energy deposited INSIDE a scoring volume by the disposal - part of the scored dose; do not add; 2 9131.55 MeV BY NAME: a charm or bottom hadron made by FTFP or a decay - no ParticleType, so no track (P23a leaves them refused; their decay tables are P23b's); abandoned 3 |

## timing

| row | port ev/s | the port's curve (events: ev/s) | per core | box | x core | x box | GB left |
|---|--:|---|--:|--:|--:|--:|--:|
| gamma_300_G4_Pb | 78,085.6 | 500,000: 96,554, 1,000,000: 78,086 (NOT SATURATED) | 1,559.1* | 15,509.9 | 50.1 | 5.03 | 0.25 |
| gamma_1000_G4_Pb | 24,595.6 | 250,000: 31,370, 500,000: 24,596 (NOT SATURATED) | 672.4 | 5,110.5 | 36.6 | 4.81 | 0.22 |
| gamma_10000_G4_Pb | - |  | 65.7* | 651.2 | - | - | - |
| e-_300_G4_Pb | 75,552.7 | 250,000: 85,151, 500,000: 75,553 (NOT SATURATED) | 1,388.7* | 13,750.4 | 54.4 | 5.49 | 0.23 |
| e-_1000_G4_Pb | 24,641.6 | 125,000: 28,737, 250,000: 24,642 (NOT SATURATED) | 605.0 | 4,618.2 | 40.7 | 5.34 | 0.23 |
| e-_10000_G4_Pb | - |  | 59.2* | 581.9 | - | - | - |
| e+_300_G4_Pb | 73,650.6 | 2,000,000: 68,480, 4,000,000: 70,808, 8,000,000: 73,651, 16,000,000: 75,250 | 1,326.8* | 13,118.8 | 55.5 | 5.61 | 0.03 |
| e+_1000_G4_Pb | 29,173.7 | 250,000: 30,816, 500,000: 29,174 (NOT SATURATED) | 607.9 | 4,724.6 | 48.0 | 6.17 | 0.23 |
| e+_10000_G4_Pb | 3,718.8 | 125,000: 3,719 (NOT SATURATED) | 58.9* | 580.1 | 63.1 | 6.41 | 0.23 |
| mu-_300_G4_Pb | 530,487.0 | 2,000,000: 530,487, 4,000,000: 538,770 | 8,387.4* | 83,343.5 | 63.2 | 6.37 |  |
| mu-_1000_G4_Pb | 342,995.0 | 2,000,000: 342,995, 4,000,000: 352,019 | 11,068.2 | 67,790.7 | 31.0 | 5.06 | 0.04 |
| mu-_10000_G4_Pb | 178,281.0 | 2,000,000: 178,281, 4,000,000: 183,063 | 5,147.9* | 51,152.4 | 34.6 | 3.49 | 0.02 |
| mu+_300_G4_Pb | 527,173.0 | 2,000,000: 527,173, 4,000,000: 527,129 | 8,269.3* | 82,104.4 | 63.8 | 6.42 |  |
| mu+_1000_G4_Pb | 356,393.0 | 2,000,000: 341,922, 4,000,000: 356,393, 8,000,000: 352,715 | 10,866.8 | 67,890.5 | 32.8 | 5.25 | 0.04 |
| mu+_10000_G4_Pb | 184,668.0 | 2,000,000: 177,891, 4,000,000: 184,668, 8,000,000: 188,139 | 5,174.7* | 51,446.7 | 35.7 | 3.59 | 0.02 |
| pi+_300_G4_Pb | 7,790.7 | 800,000: 7,791, 1,600,000: 7,659 | 1,499.1* | 14,809.7 | 5.2 | 0.53 | 0.27 |
| pi+_1000_G4_Pb | 3,662.8 | 800,000: 3,663, 1,600,000: 3,505 | 945.0 | 7,935.9 | 3.9 | 0.46 | 0.27 |
| pi+_10000_G4_Pb | 1,117.5 | 12,500: 771, 25,000: 740, 50,000: 911, 100,000: 1,019, 200,000: 1,064, 400,000: 1,118 (NOT SATURATED) | 236.8* | 2,310.3 | 4.7 | 0.48 | 0.27 |
| pi-_300_G4_Pb | 6,147.9 | 800,000: 6,148, 1,600,000: 5,459 | 1,293.7* | 12,774.0 | 4.8 | 0.48 | 0.27 |
| pi-_1000_G4_Pb | 3,544.0 | 800,000: 3,544, 1,600,000: 2,632 | 900.0 | 7,392.4 | 3.9 | 0.48 | 0.27 |
| pi-_10000_G4_Pb | 1,046.3 | 200,000: 1,046, 400,000: 1,033 | 232.2* | 2,266.8 | 4.5 | 0.46 | 0.27 |
| kaon+_300_G4_Pb | 10,372.8 | 800,000: 10,373, 1,600,000: 10,455 | 1,617.2* | 16,003.2 | 6.4 | 0.65 | 0.24 |
| kaon+_1000_G4_Pb | 4,219.9 | 800,000: 4,220, 1,600,000: 3,871 | 1,350.8 | 10,569.8 | 3.1 | 0.40 | 0.25 |
| kaon+_10000_G4_Pb | 1,044.7 | 200,000: 1,045, 400,000: 1,064 | 269.9* | 2,615.5 | 3.9 | 0.40 | 0.25 |
| kaon-_300_G4_Pb | 2,640.4 | 400,000: 2,640, 800,000: 2,609 | 835.4* | 8,230.0 | 3.2 | 0.32 | 0.25 |
| kaon-_1000_G4_Pb | 2,169.7 | 400,000: 2,170, 800,000: 2,103 | 863.3 | 6,933.4 | 2.5 | 0.31 | 0.25 |
| kaon-_10000_G4_Pb | 890.0 | 200,000: 890, 400,000: 858 | 217.4* | 2,114.2 | 4.1 | 0.42 | 0.25 |
| proton_300_G4_Pb | 19,421.5 | 800,000: 19,422, 1,600,000: 18,487 | 2,279.0* | 22,474.9 | 8.5 | 0.86 | 0.28 |
| proton_1000_G4_Pb | 4,705.9 | 800,000: 4,706, 1,600,000: 4,147 | 1,043.7 | 8,577.4 | 4.5 | 0.55 | 0.28 |
| proton_10000_G4_Pb | 1,022.3 | 200,000: 1,022, 400,000: 998 | 246.9* | 2,428.1 | 4.1 | 0.42 | 0.29 |
| anti_proton_300_G4_Pb | 1,736.8 | 400,000: 1,737, 800,000: 1,480 | 499.6* | 4,920.6 | 3.5 | 0.35 | 0.28 |
| anti_proton_1000_G4_Pb | 1,871.0 | 400,000: 1,871, 800,000: 1,686 | 693.3 | 5,616.7 | 2.7 | 0.33 | 0.28 |
| anti_proton_10000_G4_Pb | 811.8 | 200,000: 812, 400,000: 761 | 199.5* | 1,945.7 | 4.1 | 0.42 | 0.28 |
| neutron_300_G4_Pb | 5,630.9 | 800,000: 5,631, 1,600,000: 5,434 | 956.3* | 9,408.8 | 5.9 | 0.60 | 0.3 |
| neutron_1000_G4_Pb | 1,815.8 | 800,000: 1,816, 1,600,000: 1,013 | 548.6 | 4,540.5 | 3.3 | 0.40 | 0.3 |
| neutron_10000_G4_Pb | 497.5 | 800,000: 497 (NOT SATURATED) | 141.1* | 1,373.8 | 3.5 | 0.36 | 0.3 |
| anti_neutron_300_G4_Pb | 2,300.7 | 800,000: 2,301, 1,600,000: 2,224 | 768.2* | 7,574.4 | 3.0 | 0.30 | 0.3 |
| anti_neutron_1000_G4_Pb | 1,562.2 | 800,000: 1,508, 1,600,000: 1,562, 3,200,000: 1,560 | 741.8 | 5,761.7 | 2.1 | 0.27 | 0.05 |
| anti_neutron_10000_G4_Pb | 714.3 | 800,000: 714, 1,600,000: 716 | 209.4* | 2,038.9 | 3.4 | 0.35 | 0.3 |
| deuteron_300_G4_Pb | 5,763.4 | 400,000: 5,763, 800,000: 5,705 | 985.7* | 9,712.2 | 5.8 | 0.59 | 0.28 |
| deuteron_1000_G4_Pb | 1,699.6 | 400,000: 1,700, 800,000: 1,592 | 464.8 | 4,193.2 | 3.7 | 0.41 | 0.28 |
| deuteron_10000_G4_Pb | 462.8 | 100,000: 463, 200,000: 457 | 135.6* | 1,331.3 | 3.4 | 0.35 | 0.28 |
| triton_300_G4_Pb | 3,501.2 | 400,000: 3,501, 800,000: 3,390 | 646.0* | 6,361.8 | 5.4 | 0.55 | 0.28 |
| triton_1000_G4_Pb | 1,018.9 | 400,000: 1,019, 800,000: 895 | 324.6 | 2,679.1 | 3.1 | 0.38 | 0.3 |
| triton_10000_G4_Pb | 322.5 | 100,000: 322, 200,000: 290 | 87.7* | 853.1 | 3.7 | 0.38 | 0.3 |
| He3_300_G4_Pb | 5,215.9 | 400,000: 5,216, 800,000: 5,118 | 945.7* | 9,304.9 | 5.5 | 0.56 | 0.28 |
| He3_1000_G4_Pb | 828.1 | 400,000: 828, 800,000: 803 | 327.8 | 2,867.8 | 2.5 | 0.29 | 0.28 |
| He3_10000_G4_Pb | 204.0 | 400,000: 204 (NOT SATURATED) | 89.2* | 870.5 | 2.3 | 0.23 | 0.27 |
| alpha_300_G4_Pb | 3,290.9 | 400,000: 3,291, 800,000: 3,228 | 590.3* | 5,798.9 | 5.6 | 0.57 | 0.28 |
| alpha_1000_G4_Pb | 903.5 | 200,000: 903, 400,000: 914 | 256.2 | 2,199.1 | 3.5 | 0.41 | 0.28 |
| alpha_10000_G4_Pb | 267.0 | 100,000: 267, 200,000: 217 | 68.9* | 666.0 | 3.9 | 0.40 | 0.27 |
| C12_300_G4_Pb | 2,545.9 | 50,000: 2,335, 100,000: 2,546, 200,000: 2,594 | 426.7* | 4,205.9 | 6.0 | 0.61 | 0.29 |
| C12_1000_G4_Pb | 217.4 | 50,000: 211, 100,000: 217, 200,000: 220 | 89.2 | 752.7 | 2.4 | 0.29 | 0.28 |
| C12_10000_G4_Pb | 67.1 | 50,000: 67, 100,000: 65 | 21.2* | 205.6 | 3.2 | 0.33 | 0.27 |
| O16_300_G4_Pb | 2,865.3 | 50,000: 2,865, 100,000: 2,904 | 412.0* | 4,016.9 | 7.0 | 0.71 | 0.29 |
| O16_1000_G4_Pb | 172.7 | 50,000: 159, 100,000: 173, 200,000: 178 | 68.3 | 565.1 | 2.5 | 0.31 | 0.28 |
| O16_10000_G4_Pb | 51.1 | 50,000: 51, 100,000: 50 | 16.4* | 157.2 | 3.1 | 0.33 | 0.27 |
| Si28_300_G4_Pb | 2,980.1 | 50,000: 2,854, 100,000: 2,980, 200,000: 2,821 | 308.4* | 3,040.2 | 9.7 | 0.98 | 0.29 |
| Si28_1000_G4_Pb | 113.4 | 50,000: 99, 100,000: 109, 200,000: 113 (NOT SATURATED) | 39.2 | 327.9 | 2.9 | 0.35 | 0.28 |
| Si28_10000_G4_Pb | 28.2 | 50,000: 28 (NOT SATURATED) | 8.9* | 83.9 | 3.2 | 0.34 | 0.27 |
| Fe56_300_G4_Pb | 1,973.6 | 50,000: 1,890, 100,000: 1,974, 200,000: 1,897 | 179.3* | 1,777.0 | 11.0 | 1.11 | 0.29 |
| Fe56_1000_G4_Pb | 69.3 | 50,000: 59, 100,000: 69 (NOT SATURATED) | 21.5 | 177.3 | 3.2 | 0.39 | 0.28 |
| Fe56_10000_G4_Pb | 14.0 | 50,000: 14 (NOT SATURATED) | 4.2* | 39.0 | 3.4 | 0.36 | 0.25 |

# Tier 4: photons, e-, e+ at 100 keV, 1, 10 MeV from 1 um before the face, G4_Pb

﻿## doses

| row | port (Gy an event) | G4 box (Gy an event) | diff | sigma | port events | G4 events | ledgers |
|---|--:|--:|--:|--:|--:|--:|---|
| z_e-_0.1_G4_Pb | 4.6448E-016 +/- 2.3E-019 | 4.3979E-016 +/- 3.0E-019 | +5.61% | **+66.0** | 8,000,000 | 4,343,120 | 0 |
| z_e-_1_G4_Pb | 4.9258E-015 +/- 2.2E-018 | 4.9231E-015 +/- 3.0E-018 | +0.05% | +0.7 | 8,000,000 | 4,338,400 | 0 |
| z_e-_10_G4_Pb | 7.4899E-014 +/- 1.1E-017 | 7.4946E-014 +/- 3.7E-017 | -0.06% | -1.2 | 32,000,000 | 2,875,640 | 0 |
| z_e+_0.1_G4_Pb | 5.7525E-015 +/- 1.0E-018 | 5.7245E-015 +/- 2.2E-018 | +0.49% | **+11.5** | 16,000,000 | 3,413,000 | 0 |
| z_e+_1_G4_Pb | 1.1011E-014 +/- 4.6E-018 | 1.1022E-014 +/- 5.2E-018 | -0.10% | -1.7 | 4,000,000 | 3,200,000 | 0 |
| z_e+_10_G4_Pb | 8.4894E-014 +/- 3.3E-017 | 8.5005E-014 +/- 4.2E-017 | -0.13% | -2.1 | 4,000,000 | 2,559,200 | 0 |
| z_gamma_0.1_G4_Pb | 1.0695E-015 +/- 1.6E-019 | 1.0687E-015 +/- 2.6E-019 | +0.07% | +2.5 | 16,000,000 | 6,240,280 | 0 |
| z_gamma_1_G4_Pb | 9.5221E-015 +/- 3.5E-018 | 9.5199E-015 +/- 2.8E-018 | +0.02% | +0.5 | 4,000,000 | 6,400,000 | 0 |
| z_gamma_10_G4_Pb | 8.7145E-014 +/- 2.4E-017 | 8.7200E-014 +/- 3.7E-017 | -0.06% | -1.2 | 8,000,000 | 3,505,720 | 0 |

## timing

| row | port ev/s | the port's curve (events: ev/s) | per core | box | x core | x box | GB left |
|---|--:|---|--:|--:|--:|--:|--:|
| z_e-_0.1_G4_Pb | 2,544,300.0 | 2,000,000: 2,316,130, 4,000,000: 2,544,300, 8,000,000: 2,376,200 | 20,830.3* | 206,876.5 | 122.1 | 12.30 |  |
| z_e-_1_G4_Pb | 2,439,520.0 | 2,000,000: 2,236,910, 4,000,000: 2,439,520, 8,000,000: 2,134,140 | 66,832.9 | 202,706.3 | 36.5 | 12.03 |  |
| z_e-_10_G4_Pb | 1,498,930.0 | 2,000,000: 1,269,660, 4,000,000: 1,388,820, 8,000,000: 1,449,380, 16,000,000: 1,498,930, 32,000,000: 1,408,700 | 13,047.0* | 129,835.5 | 114.9 | 11.54 |  |
| z_e+_0.1_G4_Pb | 1,701,510.0 | 2,000,000: 1,272,020, 4,000,000: 1,595,800, 8,000,000: 1,701,510, 16,000,000: 1,723,220 | 16,111.0* | 160,283.1 | 105.6 | 10.62 |  |
| z_e+_1_G4_Pb | 2,439,460.0 | 2,000,000: 2,439,460, 4,000,000: 2,376,240 | 39,279.5 | 158,081.1 | 62.1 | 15.43 |  |
| z_e+_10_G4_Pb | 1,524,620.0 | 2,000,000: 1,524,620, 4,000,000: 1,553,090 | 11,477.8* | 114,049.6 | 132.8 | 13.37 |  |
| z_gamma_0.1_G4_Pb | 5,461,880.0 | 2,000,000: 4,939,930, 4,000,000: 5,254,440, 8,000,000: 5,461,880, 16,000,000: 5,476,440 | 26,751.8* | 266,177.1 | 204.2 | 20.52 |  |
| z_gamma_1_G4_Pb | 4,415,810.0 | 2,000,000: 4,415,810, 4,000,000: 4,389,690 | 166,945.6 | 258,903.4 | 26.5 | 17.06 |  |
| z_gamma_10_G4_Pb | 2,136,280.0 | 2,000,000: 1,650,200, 4,000,000: 1,936,780, 8,000,000: 2,136,280 (NOT SATURATED) | 16,893.7* | 168,092.7 | 126.5 | 12.71 | 0.07 |

# Uranium (G4_U): photons and e- at 100 keV, 1, 10 MeV from the face; both sides run by P43

﻿## doses

| row | port (Gy an event) | G4 box (Gy an event) | diff | sigma | port events | G4 events | ledgers |
|---|--:|--:|--:|--:|--:|--:|---|
| z_e-_0.1_G4_U | 2.9460E-016 +/- 2.0E-019 | 2.5496E-016 +/- 2.3E-019 | +15.55% | **+131.3** | 4,000,000 | 2,612,680 | 0 |
| z_e-_1_G4_U | 2.8383E-015 +/- 1.3E-018 | 2.8381E-015 +/- 2.4E-018 | +0.01% | +0.1 | 8,000,000 | 2,509,440 | 0 |
| z_e-_10_G4_U | 4.5044E-014 +/- 6.9E-018 | 4.5044E-014 +/- 3.0E-017 | +0.00% | +0.0 | 32,000,000 | 1,706,520 | 0 |
| z_gamma_0.1_G4_U | 6.3910E-016 +/- 2.0E-019 | 6.3776E-016 +/- 2.2E-019 | +0.21% | **+4.6** | 4,000,000 | 3,241,520 | 0 |
| z_gamma_1_G4_U | 6.0152E-015 +/- 2.1E-018 | 6.0137E-015 +/- 2.2E-018 | +0.03% | +0.5 | 4,000,000 | 3,460,240 | 0 |
| z_gamma_10_G4_U | 6.3302E-014 +/- 3.9E-017 | 6.3214E-014 +/- 7.5E-017 | +0.14% | +1.0 | 8,000,000 | 2,098,640 | 0 |

## timing

| row | port ev/s | the port's curve (events: ev/s) | per core | box | x core | x box | GB left |
|---|--:|---|--:|--:|--:|--:|--:|
| z_e-_0.1_G4_U | 4,012,160.0 | 2,000,000: 4,012,160, 4,000,000: 2,842,960 | 13,540.5* | 85,026.9 | 296.3 | 47.19 |  |
| z_e-_1_G4_U | 2,296,650.0 | 2,000,000: 2,220,900, 4,000,000: 2,296,650, 8,000,000: 2,101,010 | 33,748.3 | 80,793.3 | 68.1 | 28.43 |  |
| z_e-_10_G4_U | 1,387,510.0 | 2,000,000: 687,611, 4,000,000: 1,225,230, 8,000,000: 1,299,060, 16,000,000: 1,387,510, 32,000,000: 1,359,310 | 8,552.0* | 53,773.9 | 162.2 | 25.80 | 0.08 |
| z_gamma_0.1_G4_U | 2,937,180.0 | 2,000,000: 2,937,180, 4,000,000: 2,384,320 | 16,802.2* | 106,846.6 | 174.8 | 27.49 |  |
| z_gamma_1_G4_U | 2,814,110.0 | 2,000,000: 2,814,110, 4,000,000: 2,852,520 | 86,820.9 | 103,747.9 | 32.4 | 27.12 |  |
| z_gamma_10_G4_U | 1,121,660.0 | 2,000,000: 889,205, 4,000,000: 1,073,510, 8,000,000: 1,121,660 (NOT SATURATED) | 10,503.1* | 66,041.5 | 106.8 | 16.98 | 0.07 |
