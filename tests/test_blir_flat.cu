// The flat light-ion reaction (P36, bic/flat/ion_flat.cuh) against the one-thread
// `blir_apply_yourself` and against Geant4 (ref/oracle/bic_blirapply.csv), statistically -
// tests/test_bic_flat.cu's campaign, its rows, its anti-vacuity and its determinism checks, on six
// of the oracle's ion cases at 20,000 reactions each (docs/RISK.md V310).
#define G4GPU_TEST_BLIR_FLAT
#include "test_bic_flat.cu"
