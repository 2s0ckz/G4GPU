# P44: P27's air rows (and the water/bone controls) rerun on phys/gas_dose

Per event; Geant4 = P27's ten-worker box row (docs/sweep/p27/runs.csv), port = the largest ok run of
the rerun (runs.csv here), before = P27's port row. 3 sigma combined. docs/RISK.md V352.

## Tier 2, air (21 species x 300 MeV, 1 GeV, 10 GeV per nucleon)

```
alpha_1000_G4_AIR                  G4 6.9475E-012 port 6.7897E-012 (1,600,000)   -2.27%   -2.1 sigma  | before  -11.62%  -10.2 sigma
alpha_10000_G4_AIR                 G4 1.3959E-011 port 1.3984E-011 (  800,000)   +0.18%   +0.2 sigma  | before  -10.49%  -10.0 sigma
alpha_300_G4_AIR                   G4 8.8167E-012 port 8.7996E-012 (  800,000)   -0.19%   -0.3 sigma  | before   -5.44%   -8.3 sigma
anti_neutron_1000_G4_AIR           G4 2.0982E-012 port 2.0649E-012 (1,600,000)   -1.59%   -1.3 sigma  | before  -15.33%  -12.8 sigma
anti_neutron_10000_G4_AIR          G4 3.1437E-012 port 3.1031E-012 (1,600,000)   -1.29%   -1.1 sigma  | before  -12.18%  -10.9 sigma
anti_neutron_300_G4_AIR            G4 1.2160E-012 port 1.1964E-012 (3,200,000)   -1.62%   -1.4 sigma  | before  -12.13%  -10.8 sigma
anti_proton_1000_G4_AIR            G4 2.9154E-012 port 2.9168E-012 (1,600,000)   +0.05%   +0.0 sigma  | before  -10.05%   -9.4 sigma
anti_proton_10000_G4_AIR           G4 4.0755E-012 port 4.0547E-012 (1,600,000)   -0.51%   -0.5 sigma  | before   -9.20%   -8.3 sigma
anti_proton_300_G4_AIR             G4 2.6792E-012 port 2.6855E-012 (1,600,000)   +0.24%   +0.2 sigma  | before   -5.61%   -6.0 sigma
C12_1000_G4_AIR                    G4 3.2381E-011 port 3.2444E-011 (  200,000)   +0.19%   +0.2 sigma  | before   -5.40%   -6.3 sigma
C12_10000_G4_AIR                   G4 5.1932E-011 port 5.1286E-011 (  100,000)   -1.24%   -1.1 sigma  | before   -9.67%   -9.0 sigma
C12_300_G4_AIR                     G4 1.9782E-011 port 2.0128E-011 (  800,000)   +1.75%   +1.9 sigma  | before   -1.62%   -1.8 sigma
deuteron_1000_G4_AIR               G4 2.9351E-012 port 2.9548E-012 (1,600,000)   +0.67%   +0.5 sigma  | before  -13.47%  -11.5 sigma
deuteron_10000_G4_AIR              G4 6.9670E-012 port 7.0264E-012 (  800,000)   +0.85%   +0.7 sigma  | before  -11.57%  -10.1 sigma
deuteron_300_G4_AIR                G4 2.8769E-012 port 2.8681E-012 (  800,000)   -0.30%   -0.3 sigma  | before   -8.71%   -8.9 sigma
e-_1000_G4_AIR                     G4 2.1984E-012 port 2.2021E-012 (8,000,000)   +0.17%   +1.1 sigma  | before   +0.03%   +0.2 sigma
e-_10000_G4_AIR                    G4 3.1031E-012 port 3.1051E-012 (2,000,000)   +0.06%   +0.4 sigma  | before   -0.13%   -0.8 sigma
e-_300_G4_AIR                      G4 1.7650E-012 port 1.7661E-012 (16,000,000)   +0.06%   +0.5 sigma  | before   -0.13%   -1.0 sigma
e+_1000_G4_AIR                     G4 2.1654E-012 port 2.1693E-012 (8,000,000)   +0.18%   +1.2 sigma  | before   +0.09%   +0.6 sigma
e+_10000_G4_AIR                    G4 3.0916E-012 port 3.0959E-012 (16,000,000)   +0.14%   +0.9 sigma  | before   +0.03%   +0.2 sigma
e+_300_G4_AIR                      G4 1.7078E-012 port 1.7062E-012 (8,000,000)   -0.09%   -0.6 sigma  | before   -0.23%   -1.4 sigma
Fe56_1000_G4_AIR                   G4 5.2499E-010 port 5.1908E-010 (  200,000)   -1.12%   -1.2 sigma  | before   -2.76%   -2.9 sigma
Fe56_10000_G4_AIR                  G4 3.5284E-010 port 3.4712E-010 (  100,000)   -1.62%   -1.6 sigma  | before   -7.45%   -7.6 sigma
Fe56_300_G4_AIR                    G4 5.5995E-012 port 5.6487E-012 (  400,000)   +0.88%   +0.7 sigma  | before   -6.29%   -5.1 sigma
gamma_1000_G4_AIR                  G4 1.0008E-012 port 1.0039E-012 (32,000,000)   +0.30%   +1.8 sigma  | before   +0.04%   +0.2 sigma
gamma_10000_G4_AIR                 G4 1.2784E-012 port 1.2822E-012 (8,000,000)   +0.29%   +1.5 sigma  | before   +0.18%   +0.9 sigma
gamma_300_G4_AIR                   G4 7.7451E-013 port 7.7597E-013 (16,000,000)   +0.19%   +1.1 sigma  | before   +0.09%   +0.5 sigma
He3_1000_G4_AIR                    G4 5.9980E-012 port 5.9786E-012 (1,600,000)   -0.32%   -0.3 sigma  | before   -9.05%   -7.9 sigma
He3_10000_G4_AIR                   G4 1.1799E-011 port 1.1768E-011 (  800,000)   -0.26%   -0.3 sigma  | before   -9.97%   -9.8 sigma
He3_300_G4_AIR                     G4 8.6198E-012 port 8.7119E-012 (1,600,000)   +1.07%   +2.0 sigma  | before   -2.61%   -4.7 sigma
kaon-_1000_G4_AIR                  G4 1.9049E-012 port 1.8843E-012 (1,600,000)   -1.08%   -0.9 sigma  | before  -10.55%   -9.4 sigma
kaon-_10000_G4_AIR                 G4 2.9598E-012 port 2.9366E-012 (1,600,000)   -0.78%   -0.7 sigma  | before   -8.66%   -8.5 sigma
kaon-_300_G4_AIR                   G4 1.9695E-012 port 1.9602E-012 (1,600,000)   -0.47%   -0.4 sigma  | before   -7.47%   -7.2 sigma
kaon+_1000_G4_AIR                  G4 1.4750E-012 port 1.4896E-012 (1,600,000)   +0.99%   +1.1 sigma  | before   -4.02%   -4.6 sigma
kaon+_10000_G4_AIR                 G4 2.5835E-012 port 2.5556E-012 (1,600,000)   -1.08%   -1.1 sigma  | before   -7.54%   -7.7 sigma
kaon+_300_G4_AIR                   G4 1.5149E-012 port 1.5118E-012 (1,600,000)   -0.20%   -0.4 sigma  | before   -1.75%   -3.7 sigma
mu-_1000_G4_AIR                    G4 1.1515E-012 port 1.1505E-012 (8,000,000)   -0.09%   -0.9 sigma  | before   -0.23%   -2.3 sigma
mu-_10000_G4_AIR                   G4 1.4000E-012 port 1.3933E-012 (8,000,000)   -0.48%   -4.8 sigma  | before   -0.63%   -5.9 sigma
mu-_300_G4_AIR                     G4 1.0405E-012 port 1.0387E-012 (4,000,000)   -0.17%   -1.7 sigma  | before   -0.35%   -3.3 sigma
mu+_1000_G4_AIR                    G4 1.1537E-012 port 1.1529E-012 (8,000,000)   -0.06%   -0.6 sigma  | before   -0.22%   -2.1 sigma
mu+_10000_G4_AIR                   G4 1.4023E-012 port 1.3955E-012 (8,000,000)   -0.49%   -4.9 sigma  | before   -0.63%   -6.0 sigma
mu+_300_G4_AIR                     G4 1.0422E-012 port 1.0404E-012 (4,000,000)   -0.18%   -1.7 sigma  | before   -0.36%   -3.4 sigma
neutron_1000_G4_AIR                G4 9.9898E-013 port 9.9566E-013 (6,400,000)   -0.33%   -0.3 sigma  | before  -23.74%  -21.7 sigma
neutron_10000_G4_AIR               G4 3.0748E-012 port 3.0268E-012 (1,600,000)   -1.56%   -1.3 sigma  | before  -16.07%  -13.7 sigma
neutron_300_G4_AIR                 G4 6.7352E-013 port 6.7525E-013 (6,400,000)   +0.26%   +0.2 sigma  | before  -26.13%  -25.9 sigma
O16_1000_G4_AIR                    G4 5.3965E-011 port 5.4334E-011 (  100,000)   +0.68%   +0.8 sigma  | before   -3.98%   -5.5 sigma
O16_10000_G4_AIR                   G4 7.1365E-011 port 7.2112E-011 (  200,000)   +1.05%   +1.1 sigma  | before   -7.05%   -7.3 sigma
O16_300_G4_AIR                     G4 1.0694E-011 port 1.0799E-011 (  200,000)   +0.99%   +0.8 sigma  | before   -3.27%   -2.6 sigma
pi-_1000_G4_AIR                    G4 1.6978E-012 port 1.7130E-012 (1,600,000)   +0.90%   +0.7 sigma  | before   -9.98%   -8.2 sigma
pi-_10000_G4_AIR                   G4 3.1560E-012 port 3.1537E-012 (1,600,000)   -0.07%   -0.1 sigma  | before   -7.75%   -7.7 sigma
pi-_300_G4_AIR                     G4 1.4399E-012 port 1.4604E-012 (1,600,000)   +1.42%   +1.1 sigma  | before  -10.15%   -8.4 sigma
pi+_1000_G4_AIR                    G4 1.6905E-012 port 1.6780E-012 (3,200,000)   -0.74%   -0.7 sigma  | before   -7.58%   -6.4 sigma
pi+_10000_G4_AIR                   G4 3.0900E-012 port 3.0684E-012 (1,600,000)   -0.70%   -0.7 sigma  | before   -7.79%   -7.9 sigma
pi+_300_G4_AIR                     G4 1.4325E-012 port 1.4337E-012 (1,600,000)   +0.08%   +0.1 sigma  | before   -4.17%   -4.4 sigma
proton_1000_G4_AIR                 G4 1.8293E-012 port 1.8292E-012 (3,200,000)  +0.00%  +0.0 sigma  | before   -8.04%   -7.6 sigma
proton_10000_G4_AIR                G4 4.0599E-012 port 4.0608E-012 (1,600,000)   +0.02%   +0.0 sigma  | before  -10.96%  -10.0 sigma
proton_300_G4_AIR                  G4 2.3388E-012 port 2.3333E-012 (1,600,000)   -0.23%   -0.4 sigma  | before   -3.36%   -5.6 sigma
Si28_1000_G4_AIR                   G4 1.3916E-010 port 1.3953E-010 (  200,000)   +0.26%   +0.3 sigma  | before   -2.68%   -3.0 sigma
Si28_10000_G4_AIR                  G4 1.4364E-010 port 1.4570E-010 (  200,000)   +1.43%   +1.5 sigma  | before   -5.51%   -5.5 sigma
Si28_300_G4_AIR                    G4 6.2790E-012 port 6.3213E-012 (  400,000)   +0.67%   +0.5 sigma  | before   -4.32%   -3.6 sigma
triton_1000_G4_AIR                 G4 3.8227E-012 port 3.8663E-012 (1,600,000)   +1.14%   +1.0 sigma  | before  -14.80%  -13.2 sigma
triton_10000_G4_AIR                G4 9.8271E-012 port 9.5797E-012 (  800,000)   -2.52%   -2.2 sigma  | before  -14.45%  -13.1 sigma
triton_300_G4_AIR                  G4 3.5376E-012 port 3.5244E-012 (  800,000)   -0.37%   -0.3 sigma  | before  -12.23%  -11.0 sigma
```

## Tier 4 rerun (the gun 1 um before the trapezoid): air, and the controls in water and bone

```
z_alpha_10_G4_AIR                  G4 7.9016E-011 port 7.9226E-011 (1,600,000)   +0.27%   +2.0 sigma  | before   +0.22%   +1.7 sigma
z_e-_0.1_G4_AIR                    G4 3.2440E-012 port 3.2424E-012 (8,000,000)   -0.05%   -0.6 sigma  | before   -1.53%  -22.1 sigma
z_e-_0.1_G4_BONE_COMPACT_ICRU      G4 4.1803E-015 port 4.1769E-015 (16,000,000)   -0.08%   -1.6 sigma  | before   +0.02%   +0.2 sigma
z_e-_0.1_G4_WATER                  G4 7.8217E-015 port 7.8123E-015 (4,000,000)   -0.12%   -1.7 sigma  | before   -0.11%   -1.6 sigma
z_e-_1_G4_AIR                      G4 8.1162E-013 port 8.1069E-013 (4,000,000)   -0.11%   -1.1 sigma  | before   -0.17%   -1.7 sigma
z_e-_1_G4_BONE_COMPACT_ICRU        G4 4.3438E-014 port 4.3422E-014 (16,000,000)   -0.04%   -0.7 sigma  | before   -0.09%   -1.3 sigma
z_e-_1_G4_WATER                    G4 8.0757E-014 port 8.0720E-014 (16,000,000)   -0.05%   -0.9 sigma  | before   -0.05%   -0.9 sigma
z_e-_10_G4_AIR                     G4 1.1595E-012 port 1.1585E-012 (4,000,000)   -0.09%   -1.0 sigma  | before   -0.22%   -3.1 sigma
z_e-_10_G4_BONE_COMPACT_ICRU       G4 5.1234E-013 port 5.1205E-013 (4,000,000)   -0.06%   -0.9 sigma  | before   -0.18%   -3.0 sigma
z_e-_10_G4_WATER                   G4 9.4000E-013 port 9.3975E-013 (16,000,000)   -0.03%   -0.6 sigma  | before   -0.15%   -2.5 sigma
z_e+_0.1_G4_AIR                    G4 3.3827E-012 port 3.3824E-012 (4,000,000)   -0.01%   -0.1 sigma  | before   -1.74%  -20.0 sigma
z_e+_0.1_G4_BONE_COMPACT_ICRU      G4 1.4890E-014 port 1.4893E-014 (4,000,000)   +0.02%   +0.3 sigma  | before   +0.11%   +1.6 sigma
z_e+_0.1_G4_WATER                  G4 1.9855E-014 port 1.9866E-014 (8,000,000)   +0.06%   +0.8 sigma  | before   +0.00%   +0.0 sigma
z_e+_1_G4_AIR                      G4 8.4648E-013 port 8.4517E-013 (4,000,000)   -0.15%   -1.4 sigma  | before   -0.35%   -3.2 sigma
z_e+_1_G4_BONE_COMPACT_ICRU        G4 5.4101E-014 port 5.4071E-014 (8,000,000)   -0.06%   -0.9 sigma  | before   -0.06%   -0.8 sigma
z_e+_1_G4_WATER                    G4 9.2474E-014 port 9.2361E-014 (4,000,000)   -0.12%   -1.7 sigma  | before   -0.06%   -0.9 sigma
z_e+_10_G4_AIR                     G4 1.1569E-012 port 1.1577E-012 (8,000,000)   +0.07%   +0.8 sigma  | before   +0.04%   +0.5 sigma
z_e+_10_G4_BONE_COMPACT_ICRU       G4 5.0675E-013 port 5.0620E-013 (4,000,000)   -0.11%   -1.7 sigma  | before   +0.12%   +1.9 sigma
z_e+_10_G4_WATER                   G4 9.1094E-013 port 9.1034E-013 (4,000,000)   -0.07%   -1.1 sigma  | before   +0.18%   +2.9 sigma
z_neutron_1_G4_AIR                 G4 3.9741E-013 port 4.0313E-013 (1,600,000)   +1.44%   +1.7 sigma  | before  -84.71% -292.8 sigma
z_neutron_1_G4_BONE_COMPACT_ICRU   G4 4.0590E-014 port 4.0584E-014 (1,600,000)   -0.01%   -0.1 sigma  | before   +0.23%   +2.4 sigma
z_neutron_1_G4_WATER               G4 7.2105E-014 port 7.2133E-014 (1,600,000)   +0.04%   +0.4 sigma  | before   +0.04%   +0.4 sigma
z_neutron_10_G4_AIR                G4 5.8192E-013 port 5.7813E-013 (3,200,000)   -0.65%   -0.6 sigma  | before  -33.74%  -41.1 sigma
z_neutron_10_G4_BONE_COMPACT_ICRU  G4 1.5084E-013 port 1.5245E-013 (1,600,000)   +1.07%   +6.1 sigma  | before   +1.18%   +6.8 sigma
z_neutron_10_G4_WATER              G4 2.2782E-013 port 2.2798E-013 (1,600,000)   +0.07%   +0.3 sigma  | before   +0.07%   +0.3 sigma
z_neutron_30_G4_AIR                G4 5.2710E-012 port 5.2672E-012 (3,200,000)   -0.07%   -0.1 sigma  | before  -10.94%  -16.3 sigma
z_proton_30_G4_AIR                 G4 9.0520E-012 port 9.0169E-012 (1,600,000)   -0.39%   -2.8 sigma  | before   -0.52%   -3.8 sigma
```
