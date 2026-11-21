// P41 V325: the residual classes of the proton 1 GeV and Fe56 56 GeV beams' tapes (the pool's
// tails of integ/p40 + P41's build): name, kind (0 a cascade residual through PreCompound, 1 a
// projectile spectator through the handler alone), Z, A, the E* bin in MeV and the class's tape
// mean exciton numbers (particles, holes, charged particles), rounded.
// The proton beam's Binary residuals: the six commonest classes (29% of 81,567), two of the
// costly E* bins, and the Ca40 target's heaviest.
{"p:N15 E*[0,5)", 0, 7, 15, 0.0, 5.0, 0, 1, 0},
{"p:O15 E*[0,5)", 0, 8, 15, 0.0, 5.0, 0, 1, 0},
{"p:N15 E*[10,20)", 0, 7, 15, 10.0, 20.0, 0, 1, 0},
{"p:O16 E*[20,40)", 0, 8, 16, 20.0, 40.0, 2, 2, 1},
{"p:N14 E*[20,40)", 0, 7, 14, 20.0, 40.0, 1, 3, 0},
{"p:N14 E*[40,60)", 0, 7, 14, 40.0, 60.0, 1, 3, 1},
{"p:O15 E*[60,100)", 0, 8, 15, 60.0, 100.0, 1, 2, 1},
{"p:K39 E*[20,40)", 0, 19, 39, 20.0, 40.0, 1, 2, 0},
// The Fe56 beam's light-ion residuals: the projectile-like ones, cold and hot - the hot ones
// (E* >= 100 MeV, 23% of the reactions) are 77% of the light-ion tail's lane time.
{"fe:Fe55 E*[0,5)", 0, 26, 55, 0.0, 5.0, 0, 1, 0},
{"fe:Mn54 E*[20,40)", 0, 25, 54, 20.0, 40.0, 1, 3, 0},
{"fe:Cr51 E*[150,250)", 0, 24, 51, 150.0, 250.0, 6, 11, 3},
{"fe:Mn52 E*[150,250)", 0, 25, 52, 150.0, 250.0, 6, 10, 3},
// The light-ion arm's spectators (`blir_deexcite_spectator`: the handler without PreCompound).
{"fe:spec Cr50 E*[20,60)", 1, 24, 50, 20.0, 60.0, 0, 0, 0},
{"fe:spec Ca40 E*[60,150)", 1, 20, 40, 60.0, 150.0, 0, 0, 0},
