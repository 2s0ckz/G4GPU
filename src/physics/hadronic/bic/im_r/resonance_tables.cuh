// The six resonance-production cross-section tables, and the detailed-balance phase-space
// integral that turns a resonance in the entrance channel into one in the exit channel.
//
// Transcribed from G4XNDeltaTable, G4XNDeltastarTable, G4XNNstarTable, G4XDeltaDeltaTable,
// G4XDeltaDeltastarTable, G4XDeltaNstarTable and G4DetailedBalancePhaseSpaceIntegral
// (im_r_matrix, 11.1.1), with their data in `imr_tables.hh` by `tools/extract_bic_imr.pl`.
//
// These are what `G4XResonance::CrossSection` reads before it applies the isospin correction and
// the detailed balance: `table->GetValue(sqrtS, dummy)` on a `G4PhysicsFreeVector` built by
// `CrossSectionTable()`. So the whole NN -> N Delta, N N*, Delta Delta, Delta Delta* and
// Delta N* family of channels is the table lookup here times two factors that are not in this
// file.
//
// ## The factor of a half, which FIVE of the six apply and the sixth does not
//
// Five of the six `CrossSectionTable()` methods write
//
//     G4double value = *(sigmaPointer + i) * 0.5 * millibarn;
//
// so the vector holds HALF the tabulated number. `G4XNNstarTable`'s writes
//
//     G4double value = *(sigmaPointer + i) * millibarn;
//
// with no 0.5, so every NN -> N N* cross section is twice its five siblings' convention.
// docs/RISK.md V94. The half is undocumented in all six; it reads as the
// isospin-averaged-to-one-charge-state factor that `G4VXResonance::IsospinCorrection` divides
// back out through `pWeight`, and if that is what it is then the N N* channel is not divided by
// it. Reproduced per table, where Geant4 puts it - in the table build and not in the caller -
// and `tools/extract_bic_imr.pl` asserts which five have it, because a port that applied the
// factor uniformly would agree with nothing and look like a table error.
//
// ## The energy grid is shared, and it has a duplicate in it
//
// All six classes declare their own `energyTable[121]` and all six hold the same 121 numbers,
// from 0 to 49.244 GeV. `tools/extract_bic_imr.pl` asserts that, so a release that moved one of
// them fails there rather than reading five tables off the sixth's energies.
//
// Entries 1 and 2 are BOTH 2.014, so one interval of the grid has zero width. That is the only
// point at which `lower_bound` and `upper_bound` would choose different bins - everywhere else
// the two give the same interpolated value, because a node is the right end of one interval and
// the left end of the next and the straight line through it is the same. MEASURED: swapping the
// port's `lower_bound` for an `upper_bound` changes none of the 12,100 comparisons, and the
// reason is that every one of the six tables is still exactly zero at 2.014 GeV - the earliest
// non-zero entry in any of them is `nnstar` at 2.088 GeV. So the zero-width interval sits inside
// the dead region of every column, and the bin rule is unobservable through this class.
// Transcribed as `lower_bound` anyway, because that is what `G4PhysicsVector::GetBin` does for a
// free vector and the next release's table may not be zero there.
//
// ## Two columns are short and one resonance has no column at all
//
// `G4XNNstarTable::sigmaNN1535` and `::sigmaNN2190` are declared `[121]` and initialised with
// 113 values, so their last EIGHT entries are zero - a cross section that falls to nothing above
// 39 GeV where its neighbours are still 0.005 mb. Unreachable in QBBC, whose BIC stops at
// 1.5 GeV; reproduced, and pinned by the extractor as an exact set, exactly as the same shape in
// `G4XNNTotalLowE::ss` is (docs/RISK.md V92).
//
// And `G4XNDeltastarTable.hh` carries the comment `// 40 is missing... @@@@@@@` against
// `sigmaND1930`: there is no `sigmaND1940` column, so `delta(1940)` - which exists as a particle
// and is produced by nothing here - has no N Delta* cross section. The extractor asserts the
// absence, so a release that adds the column is visible rather than silently widening the map.
//
// ## REFUSED, by name
//
//   * **`G4XResonance::CrossSection` itself**, and with it `G4VXResonance::IsospinCorrection`,
//     `DetailedBalance` and `DegeneracyFactor`. They need `G4Clebsch` - 643 lines of
//     Clebsch-Gordan machinery - and the isospin, spin and quark content of some thirty baryon
//     resonances, neither of which is in this package. `ResonanceTableRefusal::no_cross_section`
//     is what a caller gets, at the point the correction would have been applied. The TABLE half
//     of that cross section is here and is checked; the two factors are not.
//   * **the name-keyed maps**. Geant4 keys each table by `G4String` - "delta(1600)++" and the
//     three other charge states onto one column - and a kernel has no strings, so the port keys
//     by the resonance's nominal mass in MeV (1600, 1620, ...), which is what the column names
//     say and what the four names share. The charge state chooses nothing: all four map to the
//     same column, which is the whole content of the map.
#ifndef G4GPU_BIC_IMR_RESONANCE_TABLES_CUH
#define G4GPU_BIC_IMR_RESONANCE_TABLES_CUH

#include <cmath>

#include "core/units.cuh"
#include "physics/hadronic/bic/im_r/imr_tables.hh"
#include "physics/hadronic/bic/im_r/xsec_nn.cuh"

namespace g4gpu::bic::imr {

/// Which of the six tables a channel reads, in the order the six `G4CollisionNNTo*` composites
/// are registered in `G4CollisionNN`'s `GROUP8`.
enum ResonanceTable : int {
  kResNDelta = 0,        ///< G4XNDeltaTable,         NN -> N Delta(1232)
  kResDeltaDelta = 1,    ///< G4XDeltaDeltaTable,     NN -> Delta Delta
  kResNDeltastar = 2,    ///< G4XNDeltastarTable,     NN -> N Delta*
  kResDeltaDeltastar = 3,///< G4XDeltaDeltastarTable, NN -> Delta Delta*
  kResNNstar = 4,        ///< G4XNNstarTable,         NN -> N N*
  kResDeltaNstar = 5,    ///< G4XDeltaNstarTable,     NN -> Delta N*
  kResTableCount = 6
};

/// What a table lookup could not do.
struct ResonanceTableRefusal {
  bool no_column = false;        ///< this table has no column for that resonance mass
  bool no_cross_section = false; ///< G4XResonance's two correction factors; see the file header
  int table = -1;
  int mass = 0;
};

/// The column data and the mass list for one table.
struct ResonanceTableView {
  const double* sigma = nullptr;  ///< n_columns * 121, column-major by resonance
  const int* masses = nullptr;    ///< the nominal resonance mass of each column, MeV
  int n_columns = 0;
};

/// The factor `CrossSectionTable()` scales its column by, which is 0.5 in five of the six and
/// 1.0 in `G4XNNstarTable`. See the file header and docs/RISK.md V94.
__host__ __device__ inline double resonance_table_scale(int which) {
  return (which == kResNNstar) ? 1.0 : 0.5;
}

/// `G4XNDeltaTable` and its five siblings, by index.
__host__ __device__ inline ResonanceTableView resonance_table(int which) {
  ResonanceTableView v;
  switch (which) {
    case kResNDelta:
      v.sigma = res_sigma_nd(); v.masses = res_masses_nd(); v.n_columns = kResColsNd; break;
    case kResDeltaDelta:
      v.sigma = res_sigma_dd(); v.masses = res_masses_dd(); v.n_columns = kResColsDd; break;
    case kResNDeltastar:
      v.sigma = res_sigma_ndstar(); v.masses = res_masses_ndstar();
      v.n_columns = kResColsNdstar; break;
    case kResDeltaDeltastar:
      v.sigma = res_sigma_ddstar(); v.masses = res_masses_ddstar();
      v.n_columns = kResColsDdstar; break;
    case kResNNstar:
      v.sigma = res_sigma_nnstar(); v.masses = res_masses_nnstar();
      v.n_columns = kResColsNnstar; break;
    case kResDeltaNstar:
      v.sigma = res_sigma_dnstar(); v.masses = res_masses_dnstar();
      v.n_columns = kResColsDnstar; break;
    default: break;
  }
  return v;
}

/// The column index for a resonance's nominal mass, or -1. Geant4's `xMap.find(particleName)`,
/// with the four charge states collapsed - see the file header.
__host__ __device__ inline int resonance_column(int which, int mass_mev) {
  const ResonanceTableView v = resonance_table(which);
  for (int i = 0; i < v.n_columns; ++i) {
    if (v.masses[i] == mass_mev) { return i; }
  }
  return -1;
}

/// `G4PhysicsFreeVector::GetValue(e, dummy)` on the vector `CrossSectionTable()` returns.
///
/// A free vector's bin lookup is `std::lower_bound(binVector, e) - begin - 1` and its
/// interpolation is the straight line between the two nodes, with `Value()`'s three-way guard:
/// below `edgeMin` the first value, above `edgeMax` the last, and interpolation in between.
/// `useSpline` is false - `G4PhysicsFreeVector`'s constructor defaults it so and none of the six
/// `CrossSectionTable()` methods calls `FillSecondDerivatives`.
///
/// The energy grid's FIRST entry is 0.0 and its second is 2.014 GeV, so the first interval is
/// 2 GeV wide and every sqrt(s) below the two-nucleon threshold interpolates across it from a
/// zero cross section. That is Geant4's grid, not a padding choice here.
__host__ __device__ inline double resonance_cross_section_table(int which, int mass_mev,
                                                                double sqrt_s,
                                                                ResonanceTableRefusal& ref) {
  const ResonanceTableView v = resonance_table(which);
  const int col = resonance_column(which, mass_mev);
  if (v.n_columns == 0 || col < 0) {
    ref.no_column = true;
    ref.table = which;
    ref.mass = mass_mev;
    return 0.0;
  }
  const double* e = res_energy();
  const double* s = v.sigma + static_cast<long>(col) * kResonanceTableSize;
  const double gev = u::GeV<double>();
  const int n = kResonanceTableSize;
  // The scale and the millibarn are applied where `CrossSectionTable()` applies them - as the
  // vector is filled - so the interpolation runs on the scaled values. The scale is 0.5 for
  // five of the six tables and 1.0 for G4XNNstarTable.
  const double scale = resonance_table_scale(which);
  const double lo_v = s[0] * scale * millibarn();
  const double hi_v = s[n - 1] * scale * millibarn();
  if (sqrt_s <= e[0] * gev) { return lo_v; }
  if (sqrt_s >= e[n - 1] * gev) { return hi_v; }
  // lower_bound over the energy grid, then minus one.
  int lo = 0;
  int hi = n;
  while (lo < hi) {
    const int mid = lo + (hi - lo) / 2;
    if (e[mid] * gev < sqrt_s) { lo = mid + 1; } else { hi = mid; }
  }
  const int idx = lo - 1;
  const double x1 = e[idx] * gev;
  const double dl = e[idx + 1] * gev - x1;
  const double y1 = s[idx] * scale * millibarn();
  const double dy = s[idx + 1] * scale * millibarn() - y1;
  return y1 + ((sqrt_s - x1) / dl) * dy;
}

// =============================================================================================
// G4DetailedBalancePhaseSpaceIntegral
// =============================================================================================

/// The 25 resonances `G4DetailedBalancePhaseSpaceIntegral`'s constructor dispatches on, in the
/// order of its `if`/`else if` chain: Delta(1232) first, then the nine Delta*, then the fifteen
/// N*. The order matters only in that the port's column index has to match the extractor's.
enum DbiColumn : int {
  kDbiDelta1232 = 0, kDbiDelta1600, kDbiDelta1620, kDbiDelta1700, kDbiDelta1900, kDbiDelta1905,
  kDbiDelta1910, kDbiDelta1920, kDbiDelta1930, kDbiDelta1950,
  kDbiN1440, kDbiN1520, kDbiN1535, kDbiN1650, kDbiN1675, kDbiN1680, kDbiN1700, kDbiN1710,
  kDbiN1720, kDbiN1900, kDbiN1990, kDbiN2090, kDbiN2190, kDbiN2220, kDbiN2250
};

/// The nominal mass of each column, in the same order, so a caller can look one up by mass the
/// way `resonance_column` does. Delta(1232) is the `delta` array, whose name carries no mass.
__host__ __device__ inline int dbi_mass(int column) {
  const int m[kDbiColumns] = {1232, 1600, 1620, 1700, 1900, 1905, 1910, 1920, 1930, 1950,
                              1440, 1520, 1535, 1650, 1675, 1680, 1700, 1710, 1720, 1900,
                              1990, 2090, 2190, 2220, 2250};
  return (column >= 0 && column < kDbiColumns) ? m[column] : 0;
}

/// `G4DetailedBalancePhaseSpaceIntegral::GetPhaseSpaceIntegral(sqs)`.
///
/// A linear search for the last grid point at or below `sqs`, then a straight line to the next.
/// Three things are transcribed rather than improved:
///
///   * the loop is `for (ie = 0; ie < 119; ie++)`, so index 119 - the LAST grid point - is never
///     taken as a left edge. Above it the function extrapolates along the last interval instead
///     of clamping, and `sqrts[119]` is 3.0 GeV, which a 1.5 GeV projectile on a moving target
///     can exceed.
///   * `it` starts at 0 and is only assigned inside the loop, so an `sqs` below `sqrts[0]`
///     extrapolates BACKWARDS along the first interval and can return a negative integral. The
///     first 29 entries of every column are zero, so in practice it returns zero there.
///   * the comparison is `sqrts[ie]*GeV > sqs` with the multiplication inside the loop, done 119
///     times per call.
__host__ __device__ inline double dbi_phase_space_integral(int column, double sqs) {
  if (column < 0 || column >= kDbiColumns) { return 0.0; }
  const double* e = dbi_sqrts();
  const double* d = dbi_integral() + static_cast<long>(column) * kDbiSize;
  const double gev = u::GeV<double>();
  int it = 0;
  for (int ie = 0; ie < kDbiSize - 1; ++ie) {
    if (e[ie] * gev > sqs) { break; }
    it = ie;
  }
  const double x1 = e[it] * gev;
  const double x2 = e[it + 1] * gev;
  const double y1 = d[it];
  const double y2 = d[it + 1];
  return y1 + (sqs - x1) * (y2 - y1) / (x2 - x1);
}

}  // namespace g4gpu::bic::imr

#endif
