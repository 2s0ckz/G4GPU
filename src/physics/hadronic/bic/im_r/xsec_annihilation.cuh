// G4XAnnihilationChannel - the pion-nucleon resonance production cross section, which is the
// ONLY thing a pion in the binary cascade can do (docs/RISK.md V107: the elastic half is zero
// over the whole window).
//
// Transcribed from G4XAnnihilationChannel.cc, G4BaryonWidth.cc and G4BaryonPartialWidth.cc
// (im_r_matrix, 11.1.1), with the two 25 x 120 width tables in `imr_tables.hh`.
//
// A relativistic Breit-Wigner in the entrance channel:
//
//     sigma = (2J_R + 1)/((2J_1 + 1)(2J_2 + 1))
//             * pi/p_CM^2
//             * branch * Gamma(s)^2 / ((sqrt(s) - m_R)^2 + Gamma(s)^2/4)
//             * clebsch * hbarc^2
//
// with `Gamma(s)` the resonance's mass-dependent TOTAL width, `branch` the ratio of its partial
// width into N pi to that same total, and `clebsch` the normalised isospin decomposition.
//
// ## TWO OF THE WIDTH TABLES' MAP KEYS ARE WRONG, and the effect is silent
//
// Both widths come from a `G4String`-keyed map, and `MassDependentWidth` returns **0** for a key
// it does not hold - after which `G4XAnnihilationChannel` uses the constant
// `resonance->GetPDGWidth()` instead, with nothing said.
//
//   * `G4BaryonPartialWidth`'s constructor writes `wMap["D1700_Npi"]` TWICE: at line 939 with
//     `pwN1700_Npi` - the N(1700) data under the DELTA's label - and again at line 1007 with
//     `pwD1700_Npi`. The second overwrites the first, so **there is no `N1700_Npi` key** and the
//     `pwN1700_Npi` array is compiled in and unreachable. `N(1700)`'s branching ratio is
//     therefore `150 MeV / Gamma(s)`, a constant over a function, which is not a branching ratio
//     and is not bounded by 1.
//   * `G4BaryonWidth`'s map stops at `N(2220)`: **there is no `N(2250)` key**, though the
//     particle exists and `G4CollisionMesonBaryonToResonance` builds a channel for it. Its total
//     width is the constant 500 MeV, so its Breit-Wigner has a fixed rather than a
//     threshold-suppressed shape.
//
// docs/RISK.md V111. Both are reproduced - `partial_width_column` and `total_width_column` return
// -1 for exactly those two, and the callers fall back to the PDG width as Geant4 does - and
// `tools/extract_bic_imr.pl` asserts both, so a release that fixes either fails there first. The
// `pwN1700_Npi` data IS extracted and carried, so that such a release finds it already checked.
//
// ## REFUSED, by name
//
//   * the **strange** resonances. `G4BaryonWidth` and `G4BaryonPartialWidth` carry Lambda, Sigma
//     and Xi columns; the port extracts the 25 non-strange ones, because
//     `G4CollisionMesonBaryonToResonance`'s Lambda and Sigma channels are commented out in
//     11.1.1 (docs/RISK.md V107) and nothing else reaches them.
// ## The isospin factor is a ratio that is always one, and three of its parts are inert
//
// `NormalizedClebsch` reduces to `NormalizedClebschGordan(isoRes, iso3, isoPion, isoNucleon,
// iso3Pion, iso3Nucleon)`, which sums the squared Clebsch-Gordan coefficients over every pion
// projection the pair allows and divides the chosen one by that sum. For a pion on a nucleon the
// sum runs over a COMPLETE set, so unitarity makes it exactly 1 and the division is a no-op -
// MEASURED: removing it changes none of the 2,695 cross sections. Swapping the two constituents
// is a no-op too, by the symmetry of the squares.
//
// The `isoRes < iso3` guard above it is a shortcut and not a filter. It DOES fire now that
// `collision_meson.cuh` sums all 25 channels for every pion-nucleon pair - a pi+ on a proton is
// iso3 = +3 against an N*'s isoRes of 1 - but MEASURED: disabling it changes none of the 3,000
// buffered partials or totals, because the Clebsch-Gordan coefficient underneath is zero for
// exactly the projections the guard rejects and reaches the same answer by a longer route. It is
// transcribed because a release that changed either half would have to change both.
//
//   * the **particle-antiparticle halving** in `NormalizedClebsch`:
//     `if (def1->GetPDGEncoding() != -(def2->GetPDGEncoding())) cleb = 0.5*cleb;` inside a test on
//     `anti < 0` and two equal particle TYPES. A pion and a nucleon are a meson and a baryon, so
//     the outer test is false for every pair this channel sees and the branch is dead. It is
//     transcribed as a flag rather than carried, because reaching it would mean the caller
//     changed.
#ifndef G4GPU_BIC_IMR_XSEC_ANNIHILATION_CUH
#define G4GPU_BIC_IMR_XSEC_ANNIHILATION_CUH

#include <cmath>

#include "physics/hadronic/bic/im_r/clebsch.cuh"
#include "physics/hadronic/bic/im_r/resonance_fs.cuh"

namespace g4gpu::bic::imr {

/// What an annihilation-channel cross section could not do.
struct AnnihRefusal {
  bool unknown_resonance = false;  ///< no column and no PDG width for this code
  bool antiparticle_branch = false;///< the dead halving in NormalizedClebsch; see the header
  int refused_pdg = 0;
  ClebschRefusal clebsch;
  XsecRefusal clebsch_xsec;        ///< what the elastic partial beside it refused, if anything
};

/// The 25 columns of `G4BaryonWidth` and `G4BaryonPartialWidth`, in the order the extractor lays
/// them out: the fifteen N* by increasing name, then the ground-state Delta, then the nine
/// Delta*.
__host__ __device__ inline int width_column_for_mass(int mass, bool is_delta) {
  if (is_delta) {
    if (mass == 1232) { return 15; }
    for (int i = 0; i < 9; ++i) {
      if (deltastar_mass_list()[i] == mass) { return 16 + i; }
    }
    return -1;
  }
  for (int i = 0; i < 15; ++i) {
    if (nstar_mass_list()[i] == mass) { return i; }
  }
  return -1;
}

/// `G4BaryonWidth::MassDependentWidth(shortName)` - the column, or -1 when the map has no key.
///
/// The ONLY key the map lacks among the non-strange baryons is `N(2250)`; see the file header.
__host__ __device__ inline int total_width_column(int mass, bool is_delta) {
  if (!is_delta && mass == 2250) { return -1; }  // the missing key
  return width_column_for_mass(mass, is_delta);
}

/// `G4BaryonPartialWidth::MassDependentWidth(label)` - likewise, and the only missing key here is
/// `N1700_Npi`, clobbered by the second `D1700_Npi` assignment.
__host__ __device__ inline int partial_width_column(int mass, bool is_delta) {
  if (!is_delta && mass == 1700) { return -1; }  // the clobbered key
  return width_column_for_mass(mass, is_delta);
}

/// `G4PhysicsFreeVector::GetValue` on one of the two width tables: `lower_bound` on the energy
/// grid, the straight line between the two nodes, and the three-way guard at the ends.
__host__ __device__ inline double width_table_value(const double* grid, const double* value,
                                                    int column, double sqrt_s) {
  const int n = kBaryonWidthSize;
  const double* v = value + static_cast<long>(column) * n;
  const double gev = u::GeV<double>();
  if (sqrt_s <= grid[0] * gev) { return v[0]; }
  if (sqrt_s >= grid[n - 1] * gev) { return v[n - 1]; }
  int lo = 0;
  int hi = n;
  while (lo < hi) {
    const int mid = lo + (hi - lo) / 2;
    if (grid[mid] * gev < sqrt_s) { lo = mid + 1; } else { hi = mid; }
  }
  const int idx = lo - 1;
  const double x1 = grid[idx] * gev;
  const double dl = grid[idx + 1] * gev - x1;
  const double y1 = v[idx];
  const double dy = v[idx + 1] - y1;
  return y1 + ((sqrt_s - x1) / dl) * dy;
}

/// `G4XAnnihilationChannel::VariableWidth` - the mass-dependent total width, or the constant PDG
/// width when the map has no key.
__host__ __device__ inline double annih_variable_width(int mass, bool is_delta, double pdg_width,
                                                       double sqrt_s) {
  const int col = total_width_column(mass, is_delta);
  if (col < 0) { return pdg_width; }
  return width_table_value(baryon_width_grid(), baryon_width(), col, sqrt_s);
}

/// `G4XAnnihilationChannel::VariablePartialWidth` - likewise for the N pi partial width.
__host__ __device__ inline double annih_variable_partial_width(int mass, bool is_delta,
                                                               double pdg_width, double sqrt_s) {
  const int col = partial_width_column(mass, is_delta);
  if (col < 0) { return pdg_width; }
  return width_table_value(baryon_partial_width_grid(), baryon_partial_width(), col, sqrt_s);
}

/// `G4XAnnihilationChannel::NormalizedClebsch`.
///
/// `isoRes < iso3` returns 0 - note the comparison is against the SIGNED iso3, so a negative iso3
/// always passes it however large its magnitude; a pi- on a neutron is iso3 = -3 against an N*'s
/// isoRes of 1, and `1 < -3` is false, so it goes on to the Clebsch-Gordan, which returns 0 for
/// the same reason by a longer route.
__host__ __device__ inline double annih_normalized_clebsch(int iso1, int iso31, int iso2,
                                                           int iso32, int iso_res,
                                                           AnnihRefusal& ref) {
  const int iso3 = iso31 + iso32;
  if (iso_res < iso3) { return 0.0; }
  if ((iso1 * iso2) == 0) { return 1.0; }
  return normalized_clebsch_gordan(iso_res, iso3, iso1, iso2, iso31, iso32, ref.clebsch);
}

/// `G4XAnnihilationChannel::CrossSection`.
///
/// `hbarc_squared` is CLHEP's `hbarc*hbarc`, and `hbarc` is DERIVED - `hbar_Planck * c_light`
/// with `hbar_Planck = h_Planck/twopi` - not a pasted decimal.
///
/// The first version of this file pasted 197.32696812e-12, which is the value the Particle Data
/// Group quotes and is 6e-8 below what CLHEP computes. Squared, that is 1.2e-7, and the oracle
/// reported exactly 1.251e-07 on the first channel it reached. core/units.cuh already carries
/// CLHEP's derived value, for the same reason its note on `barn()` gives; this uses that one.
__host__ __device__ inline constexpr double hbarc_squared() {
  return u::hbarc<double>() * u::hbarc<double>();
}

__host__ __device__ inline double x_annihilation_channel(
    int spin1, double m1, int iso1, int iso31, int spin2, double m2, int iso2, int iso32,
    int res_mass_label, bool res_is_delta, int res_spin, double res_mass, double res_pdg_width,
    int res_iso, double sqrt_s, AnnihRefusal& ref) {
  const double S = sqrt_s * sqrt_s;
  if (S == 0.0) {
    ref.unknown_resonance = true;
    return 0.0;
  }
  const double width = annih_variable_width(res_mass_label, res_is_delta, res_pdg_width, sqrt_s);
  double branch = 0.0;
  if (width != 0.0) {
    branch = annih_variable_partial_width(res_mass_label, res_is_delta, res_pdg_width, sqrt_s) /
             width;
  }
  const double cleb = annih_normalized_clebsch(iso1, iso31, iso2, iso32, res_iso, ref);
  const double p_cm2_num = (S - (m1 + m2) * (m1 + m2)) * (S - (m1 - m2) * (m1 - m2));
  const double p_cm = std::sqrt(p_cm2_num / (4.0 * S));
  return ((res_spin + 1.0) / ((spin1 + 1) * (spin2 + 1)) * u::pi<double>() / (p_cm * p_cm) *
          branch * width * width /
          ((sqrt_s - res_mass) * (sqrt_s - res_mass) + width * width / 4.0) * cleb *
          hbarc_squared());
}

}  // namespace g4gpu::bic::imr

#endif
