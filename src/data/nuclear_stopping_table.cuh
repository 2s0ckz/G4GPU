// ICRU49 nuclear stopping table, 104 (reduced energy, reduced stopping) pairs, extracted
// programmatically from Geant4 11.1.1 G4ICRU49NuclearStoppingModel::NuclearStoppingPower.
#pragma once
#include "core/units.cuh"

namespace g4gpu::data {

constexpr int kNucaPoints = 104;

/// Reduced energy grid, descending.
template <typename real_t> __host__ __device__ inline const real_t* nuca_energy() {
  static const real_t v[kNucaPoints] = {
    real_t(1.0E+8), real_t(8.0E+7), real_t(6.0E+7), real_t(5.0E+7), real_t(4.0E+7), real_t(3.0E+7)
    , real_t(2.0E+7), real_t(1.5E+7), real_t(1.0E+7), real_t(8.0E+6), real_t(6.0E+6), real_t(5.0E+6)
    , real_t(4.0E+6), real_t(3.0E+6), real_t(2.0E+6), real_t(1.5E+6), real_t(1.0E+6), real_t(8.0E+5)
    , real_t(6.0E+5), real_t(5.0E+5), real_t(4.0E+5), real_t(3.0E+5), real_t(2.0E+5), real_t(1.5E+5)
    , real_t(1.0E+5), real_t(8.0E+4), real_t(6.0E+4), real_t(5.0E+4), real_t(4.0E+4), real_t(3.0E+4)
    , real_t(2.0E+4), real_t(1.5E+4), real_t(1.0E+4), real_t(8.0E+3), real_t(6.0E+3), real_t(5.0E+3)
    , real_t(4.0E+3), real_t(3.0E+3), real_t(2.0E+3), real_t(1.5E+3), real_t(1.0E+3), real_t(8.0E+2)
    , real_t(6.0E+2), real_t(5.0E+2), real_t(4.0E+2), real_t(3.0E+2), real_t(2.0E+2), real_t(1.5E+2)
    , real_t(1.0E+2), real_t(8.0E+1), real_t(6.0E+1), real_t(5.0E+1), real_t(4.0E+1), real_t(3.0E+1)
    , real_t(1.5E+1), real_t(1.0E+1), real_t(8.0E+0), real_t(6.0E+0), real_t(5.0E+0), real_t(4.0E+0)
    , real_t(3.0E+0), real_t(2.0E+0), real_t(1.5E+0), real_t(1.0E+0), real_t(8.0E-1), real_t(6.0E-1)
    , real_t(5.0E-1), real_t(4.0E-1), real_t(3.0E-1), real_t(2.0E-1), real_t(1.5E-1), real_t(1.0E-1)
    , real_t(8.0E-2), real_t(6.0E-2), real_t(5.0E-2), real_t(4.0E-2), real_t(3.0E-2), real_t(2.0E-2)
    , real_t(1.5E-2), real_t(1.0E-2), real_t(8.0E-3), real_t(6.0E-3), real_t(5.0E-3), real_t(4.0E-3)
    , real_t(3.0E-3), real_t(2.0E-3), real_t(1.5E-3), real_t(1.0E-3), real_t(8.0E-4), real_t(6.0E-4)
    , real_t(5.0E-4), real_t(4.0E-4), real_t(3.0E-4), real_t(2.0E-4), real_t(1.5E-4), real_t(1.0E-4)
    , real_t(8.0E-5), real_t(6.0E-5), real_t(5.0E-5), real_t(4.0E-5), real_t(3.0E-5), real_t(2.0E-5)
    , real_t(1.5E-5), real_t(0.0)};
  return v;
}

/// Reduced stopping power at each grid point.
template <typename real_t> __host__ __device__ inline const real_t* nuca_loss() {
  static const real_t v[kNucaPoints] = {
    real_t(5.831E-8), real_t(7.288E-8), real_t(9.719E-8), real_t(1.166E-7), real_t(1.457E-7), real_t(1.942E-7)
    , real_t(2.916E-7), real_t(3.887E-7), real_t(5.833E-7), real_t(7.287E-7), real_t(9.712E-7), real_t(1.166E-6)
    , real_t(1.457E-6), real_t(1.941E-6), real_t(2.911E-6), real_t(3.878E-6), real_t(5.810E-6), real_t(7.262E-6)
    , real_t(9.663E-6), real_t(1.157E-5), real_t(1.442E-5), real_t(1.913E-5), real_t(2.845E-5), real_t(3.762E-5)
    , real_t(5.554E-5), real_t(6.866E-5), real_t(9.020E-5), real_t(1.070E-4), real_t(1.319E-4), real_t(1.722E-4)
    , real_t(2.499E-4), real_t(3.248E-4), real_t(4.688E-4), real_t(5.729E-4), real_t(7.411E-4), real_t(8.718E-4)
    , real_t(1.063E-3), real_t(1.370E-3), real_t(1.955E-3), real_t(2.511E-3), real_t(3.563E-3), real_t(4.314E-3)
    , real_t(5.511E-3), real_t(6.430E-3), real_t(7.756E-3), real_t(9.855E-3), real_t(1.375E-2), real_t(1.736E-2)
    , real_t(2.395E-2), real_t(2.850E-2), real_t(3.552E-2), real_t(4.073E-2), real_t(4.802E-2), real_t(5.904E-2)
    , real_t(9.426E-2), real_t(1.210E-1), real_t(1.377E-1), real_t(1.611E-1), real_t(1.768E-1), real_t(1.968E-1)
    , real_t(2.235E-1), real_t(2.613E-1), real_t(2.871E-1), real_t(3.199E-1), real_t(3.354E-1), real_t(3.523E-1)
    , real_t(3.609E-1), real_t(3.693E-1), real_t(3.766E-1), real_t(3.803E-1), real_t(3.788E-1), real_t(3.711E-1)
    , real_t(3.644E-1), real_t(3.530E-1), real_t(3.444E-1), real_t(3.323E-1), real_t(3.144E-1), real_t(2.854E-1)
    , real_t(2.629E-1), real_t(2.298E-1), real_t(2.115E-1), real_t(1.883E-1), real_t(1.741E-1), real_t(1.574E-1)
    , real_t(1.372E-1), real_t(1.116E-1), real_t(9.559E-2), real_t(7.601E-2), real_t(6.668E-2), real_t(5.605E-2)
    , real_t(5.008E-2), real_t(4.352E-2), real_t(3.617E-2), real_t(2.768E-2), real_t(2.279E-2), real_t(1.723E-2)
    , real_t(1.473E-2), real_t(1.200E-2), real_t(1.052E-2), real_t(8.950E-3), real_t(7.246E-3), real_t(5.358E-3)
    , real_t(4.313E-3), real_t(3.166E-3)};
  return v;
}

}  // namespace g4gpu::data
