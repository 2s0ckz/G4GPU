// The nucleon-nucleon cross sections the binary cascade's collision tree is built on.
//
// Transcribed from im_r_matrix in 11.1.1: G4XNNTotal, G4XnpTotal, G4XNNElastic, G4XnpElastic
// (the four G4CrossSectionPatch composites), G4XNNTotalLowE, G4XnpTotalLowE, G4XNNElasticLowE,
// G4XnpElasticLowE (the tabulated low-energy arms), G4XPDGTotal and G4XPDGElastic (the
// high-energy fits), G4CrossSectionPatch and G4CrossSectionComposite (how the arms are joined),
// G4LowEXsection (the log-log interpolator), G4VCrossSectionSource::FindKeyParticle, and the
// G4PhysicsLogVector shape the four 101-value tables are poured into.
//
// ## Which of these the cascade actually reaches
//
// `G4Scatterer`'s channel list is `GROUP2(G4CollisionNN, G4CollisionMesonBaryon)`, and
// `G4CollisionNN`'s own cross-section source is `G4XNNTotal` - used for pp, nn AND np, because
// `G4XNNTotalLowE` picks the arm by `FindKeyParticle` and `G4XPDGTotal` by the mass-ordered
// definition pair. The elastic channels take `G4XNNElastic` (pp and nn) and `G4XnpElastic` (np).
//
// **`G4XnpTotal` is reached by `G4CollisionPN` alone, and `G4CollisionPN` is reached by
// nothing.** `G4Scatterer.cc` includes `G4CollisionPN.hh` and does not register it; the class is
// in no GROUPn. `G4XpnTotal` and `G4XpnElastic` are included by nothing at all in the whole
// hadronic tree. All three are transcribed anyway - `G4XnpTotal` because it is one line of
// composition over arms that are here for other reasons, the other two not at all - and the
// reason the distinction is written down is that a reader comparing this file against the
// directory listing would otherwise think three classes were missed.
//
// ## Two things in here are Geant4 arithmetic that a reader would call a bug
//
// **The np energy grid is stretched by 1% and the pp one is not.** `G4XNNElasticLowE`'s
// constructor builds the pp vector with `_eMin = _eMinTable*GeV`, then REASSIGNS
// `_eMin = exp(log(_eMinTable) - _eStepLog)*GeV` and builds the np vector with the new one -
// while both use the same `_eMax = exp(log(_eMinTable) + 101*_eStepLog)*GeV`. A
// `G4PhysicsLogVector(Emin, Emax, 101)` spreads its 102 nodes evenly in log over
// `log(Emax/Emin)`, so the pp nodes come out at the intended 0.01 log steps and the np nodes at
// 1.02/101 = 0.0100990 - the table's 101 values, which were tabulated on a 0.01 grid, land 1%
// too far apart, and the top of the table sits at `_eMinTable*e^0.9999` where the data means
// `e^0.99`. `G4XnpElasticLowE` and `G4XnpTotalLowE` do the same thing (their first `_eMin`
// assignment is dead), so every np cross section in the cascade is read off the stretched grid
// and every pp one off the correct grid. docs/RISK.md V91.
//
// **Every one of the four log vectors has a 102nd node holding zero.** `G4PhysicsLogVector(Emin,
// Emax, Nbin)` sets `numberOfNodes = Nbin + 1 = 102` and zero-fills `dataVector`; the
// constructors then `PutValue` 101 times, leaving `dataVector[101] = 0`. So the top bin
// interpolates from the last tabulated cross section down to zero over the last 1% of the range,
// and at `sqrtS == edgeMax` exactly `Value()` returns 0. Neither is reachable through the patch,
// which stops the low-energy arm at 3 GeV while `edgeMax` is 5.2 GeV - but it is reproduced,
// because "unreachable" is a property of the composition and not of this class, and because the
// same shape appears in `G4XNNTotalLowE::ss`, where the 29th slot is also an uninitialised zero.
// docs/RISK.md V92.
//
// ## REFUSED, by name
//
//   * `G4XPDGTotal`'s and `G4XPDGElastic`'s **kaon, antiproton, antineutron and gamma** rows.
//     The fits are in `imr_tables.hh` for the three pairs a nucleon or a charged pion can form;
//     the K+p, K-p, ppbar, npbar, gamma-p and gamma-gamma rows are NOT, because no channel
//     `G4Scatterer` registers puts a kaon, an antinucleon or a photon into a collision, and a
//     transcribed fit nothing calls is a fit nothing checks. `XsecRefusal::pair` names the pair.
//   * `G4XpnTotal` and `G4XpnElastic`, dead in 11.1.1 (above).
//   * `G4CollisionPN` and with it `G4XnpTotal`'s only caller (above); the composite itself is
//     here because it costs one function.
#ifndef G4GPU_BIC_IMR_XSEC_NN_CUH
#define G4GPU_BIC_IMR_XSEC_NN_CUH

#include <cfloat>
#include <cmath>

#include "core/units.cuh"
#include "data/g4pow.hh"
#include "physics/hadronic/bic/im_r/imr_tables.hh"

namespace g4gpu::bic::imr {

namespace u = g4gpu::units;

/// CLHEP's `millibarn = 1.e-3*barn`, derived the way CLHEP derives it rather than written as a
/// decimal - core/units.cuh's note on `barn()` says what the difference cost once. Local to this
/// namespace because core/units.cuh is shared with the EM port and this package does not own it.
__host__ __device__ inline constexpr double millibarn() { return 1.e-3 * u::barn<double>(); }

/// PDG codes this file answers for. Anything else is refused by code, not defaulted.
enum : int { kPdgProton = 2212, kPdgNeutron = 2112, kPdgPiPlus = 211, kPdgPiMinus = -211 };

/// What a cross-section call could not answer. A refusal returns 0 as Geant4's map miss does,
/// but says so rather than letting a zero cross section read as "no interaction".
struct XsecRefusal {
  bool pair = false;      ///< no row for this (pdg1, pdg2) - see the file header
  bool key_particle = false;  ///< G4VCrossSectionSource::FindKeyParticle's G4HadronicException
  int pdg1 = 0;
  int pdg2 = 0;
  __host__ __device__ bool any() const { return pair || key_particle; }
};

// =============================================================================================
// G4PhysicsLogVector, in the exact shape the four im_r_matrix tables are poured into.
// =============================================================================================

/// One of the four 101-value low-energy tables, as the G4PhysicsLogVector that holds it.
///
/// `n_nodes` is 102 and `n_filled` is 101: see the file header for why the last node is zero.
/// The node energies are not stored - `G4PhysicsLogVector`'s constructor computes
/// `binVector[i] = edgeMin*G4Exp(i/invdBin)` for `1 <= i <= idxmax` and pins `binVector[0]` and
/// `binVector[idxmax+1]` to the two limits, so they are recomputed here the same way. Storing
/// them would be storing a copy of a formula and would hide the stretched np grid.
struct LogVec101 {
  const double* data = nullptr;  ///< the 101 tabulated values, millibarn
  double edge_min = 0.0;
  double edge_max = 0.0;
  double inv_dbin = 0.0;

  static constexpr int kNodes = 102;
  static constexpr int kIdxMax = 100;  ///< numberOfNodes - 2

  /// G4PhysicsLogVector::Initialise, from the two limits the constructor was given.
  __host__ __device__ static LogVec101 make(const double* values, double emin, double emax) {
    LogVec101 v;
    v.data = values;
    v.edge_min = emin;
    v.edge_max = emax;
    // invdBin = (idxmax + 1) / G4Log(edgeMax/edgeMin), with idxmax = 100.
    v.inv_dbin = static_cast<double>(kIdxMax + 1) / std::log(emax / emin);
    return v;
  }

  /// `binVector[i]`. Written exactly as the constructor's loop writes it, endpoints included -
  /// `binVector[0] = Emin` and `binVector[numberOfNodes-1] = Emax` are ASSIGNED before the
  /// loop, and the loop runs only over `1 <= i <= idxmax`, so node 101 is `Emax` and not
  /// `edgeMin*exp(101/invdBin)`. Those two differ by rounding and node 101 is the top of the
  /// last interpolation interval.
  __host__ __device__ double node_energy(int i) const {
    if (i <= 0) { return edge_min; }
    if (i >= kNodes - 1) { return edge_max; }
    return edge_min * std::exp(static_cast<double>(i) / inv_dbin);
  }

  /// `dataVector[i]` - zero past the 101 values `PutValue` wrote.
  ///
  /// The `* millibarn()` is where the four constructors put it: `PutValue(i, table[i]*millibarn)`.
  /// So the vector holds cross sections in Geant4's internal units and the linear interpolation
  /// runs on scaled values. Scaling after the interpolation instead would be the same number to
  /// within a rounding and is not what the source does; `imr_tables.hh` keeps the raw millibarn
  /// figures because that is what the Geant4 files say and a table of 1e-23s is unreadable.
  __host__ __device__ double node_value(int i) const {
    return ((i >= 0 && i < kNNLowETableSize) ? data[i] : 0.0) * millibarn();
  }

  /// G4PhysicsVector::ComputeLogVectorBin, truncating toward zero on a G4int and clamped to
  /// idxmax.
  __host__ __device__ int bin(double e) const {
    const int b = static_cast<int>((std::log(e) - std::log(edge_min)) * inv_dbin);
    return (b < kIdxMax) ? b : kIdxMax;
  }

  /// G4PhysicsVector::Interpolation with `useSpline` false - which it is, because
  /// G4PhysicsLogVector's spline argument defaults to false and none of the four constructors
  /// passes it or calls FillSecondDerivatives.
  __host__ __device__ double interpolate(int idx, double e) const {
    const double x1 = node_energy(idx);
    const double dl = node_energy(idx + 1) - x1;
    const double y1 = node_value(idx);
    const double dy = node_value(idx + 1) - y1;
    return y1 + ((e - x1) / dl) * dy;
  }

  /// G4PhysicsVector::Value(e), which `GetValue(e, isOutRange)` forwards to unchanged.
  __host__ __device__ double value(double e) const {
    if (e > edge_min && e < edge_max) { return interpolate(bin(e), e); }
    if (e <= edge_min) { return node_value(0); }
    return node_value(kNodes - 1);  // zero; see the file header
  }
};

/// The shared grid constants, as the four classes write them. `_eMinTable` and `_eStepLog` are
/// pinned in the Geant4 source by `tools/extract_bic_imr.pl`.
__host__ __device__ inline constexpr double lowe_emin_table() { return 1.8964808; }
__host__ __device__ inline constexpr double lowe_estep_log() { return 0.01; }

/// `_eMax = G4Exp(G4Log(_eMinTable) + tableSize*_eStepLog)*GeV`, the same in all four.
__host__ __device__ inline double lowe_emax() {
  return std::exp(std::log(lowe_emin_table()) + kNNLowETableSize * lowe_estep_log()) *
         u::GeV<double>();
}
/// The pp vector's `Emin`: `_eMinTable*GeV`, used before the reassignment.
__host__ __device__ inline double lowe_emin_pp() {
  return lowe_emin_table() * u::GeV<double>();
}
/// The np vector's `Emin`: one log step BELOW, which is also the `_eMin` every one of the four
/// `CrossSection` methods compares `sqrtS` against - including the pp one, whose vector was
/// built with the other value. See the file header.
__host__ __device__ inline double lowe_emin_np() {
  return std::exp(std::log(lowe_emin_table()) - lowe_estep_log()) * u::GeV<double>();
}

// =============================================================================================
// G4LowEXsection - the 29-pair log-log interpolator behind G4XNNTotalLowE.
// =============================================================================================

/// `G4LowEXsection::CrossSection(aX)`: find the last pair whose energy is <= aX, then
/// interpolate linearly in log-log between it and the next, and multiply by millibarn.
///
/// `n` is 29, and the 29th pair's energy is ZERO - `G4XNNTotalLowE::ss[29]` is initialised with
/// 28 values. Reaching it needs `aX >= 3002.71 MeV`, which the patch's 3 GeV gate forbids; if it
/// were reached, `it` would land on index 28 and `*(it+1)` would read one past the end of the
/// vector. The port stops at index 27 instead of reading out of bounds and reports it, because
/// a kernel that reads past an array does not throw, it returns a number.
__host__ __device__ inline double lowe_xsection(const double* energy, const double* sigma, int n,
                                                double ax, bool& out_of_range) {
  out_of_range = false;
  if (ax < energy[0]) { return 0.0; }
  int it = -1;
  for (int i = 0; i < n; ++i) {
    if (energy[i] > ax) { break; }
    it = i;
  }
  if (it < 0 || it + 1 >= n) {
    // Geant4 would dereference `end()` here; see above.
    out_of_range = true;
    return 0.0;
  }
  const double x1 = std::log(energy[it]);
  const double x2 = std::log(energy[it + 1]);
  const double y1 = std::log(sigma[it]);
  const double y2 = std::log(sigma[it + 1]);
  const double x = std::log(ax);
  const double y = y1 + (x - x1) * (y2 - y1) / (x2 - x1);
  return std::exp(y) * millibarn();
}

// =============================================================================================
// G4VCrossSectionSource::FindKeyParticle
// =============================================================================================

/// The two-particle key the tabulated NN classes index by: PROTON for pp and nn, NEUTRON for
/// np and pn. Geant4 throws a G4HadronicException for any other pair; a kernel cannot throw, so
/// the refusal is returned.
__host__ __device__ inline int find_key_particle(int pdg1, int pdg2, XsecRefusal& ref) {
  if ((pdg1 == kPdgProton && pdg2 == kPdgProton) ||
      (pdg1 == kPdgNeutron && pdg2 == kPdgNeutron)) {
    return kPdgProton;
  }
  if ((pdg1 == kPdgNeutron && pdg2 == kPdgProton) ||
      (pdg2 == kPdgNeutron && pdg1 == kPdgProton)) {
    return kPdgNeutron;
  }
  ref.key_particle = true;
  ref.pdg1 = pdg1;
  ref.pdg2 = pdg2;
  return 0;
}

// =============================================================================================
// The four low-energy arms.
// =============================================================================================

/// G4XNNTotalLowE::CrossSection. The pp table is keyed by proton, the np table by neutron, and
/// the pair that reaches neither is the exception FindKeyParticle throws.
__host__ __device__ inline double x_nn_total_lowe(int pdg1, int pdg2, double sqrt_s,
                                                  XsecRefusal& ref) {
  const int key = find_key_particle(pdg1, pdg2, ref);
  if (ref.key_particle) { return 0.0; }
  const double* sig = (key == kPdgProton) ? nn_total_lowe_pp() : nn_total_lowe_np();
  bool oor = false;
  return lowe_xsection(nn_total_lowe_ss(), sig, kNNTotalLowESize, sqrt_s, oor);
}
/// `IsValid` for the same class - `e > 0 && e < 3*GeV`, a STRICT inequality at both ends where
/// the other three arms use `InLimits`, which is inclusive. The difference decides which arm
/// answers at exactly 3 GeV.
__host__ __device__ inline bool x_nn_total_lowe_valid(double e) {
  return e > 0.0 && e < 3.0 * u::GeV<double>();
}
__host__ __device__ inline double x_nn_total_lowe_high_limit() { return 3.0 * u::GeV<double>(); }

/// G4XNNElasticLowE::CrossSection - the pp/nn table on the un-shifted grid, the np table on the
/// shifted one, and BOTH compared against the shifted `_eMin`.
__host__ __device__ inline double x_nn_elastic_lowe(int pdg1, int pdg2, double sqrt_s,
                                                    XsecRefusal& ref) {
  const int key = find_key_particle(pdg1, pdg2, ref);
  if (ref.key_particle) { return 0.0; }
  const LogVec101 v = (key == kPdgProton)
                          ? LogVec101::make(nn_elastic_lowe_pp(), lowe_emin_pp(), lowe_emax())
                          : LogVec101::make(nn_elastic_lowe_np(), lowe_emin_np(), lowe_emax());
  // The member `_eMin` at the time CrossSection runs is the SHIFTED one - the constructor's
  // second assignment - for both tables.
  const double e_min = lowe_emin_np();
  if (sqrt_s >= e_min && sqrt_s <= lowe_emax()) { return v.value(sqrt_s); }
  if (sqrt_s < e_min) { return v.value(e_min); }
  return 0.0;
}
__host__ __device__ inline bool x_nn_elastic_lowe_valid(double e) {
  return e >= 0.0 && e <= 3.0 * u::GeV<double>();
}
__host__ __device__ inline double x_nn_elastic_lowe_high_limit() {
  return 3.0 * u::GeV<double>();
}

/// G4XnpElasticLowE::CrossSection - np only, and a pair that is not np returns zero rather than
/// refusing, because that is the class's own `if`.
__host__ __device__ inline double x_np_elastic_lowe(int pdg1, int pdg2, double sqrt_s) {
  const bool is_np = (pdg1 == kPdgProton && pdg2 == kPdgNeutron) ||
                     (pdg1 == kPdgNeutron && pdg2 == kPdgProton);
  if (!is_np) { return 0.0; }
  const LogVec101 v =
      LogVec101::make(nn_elastic_lowe_np(), lowe_emin_np(), lowe_emax());
  const double e_min = lowe_emin_np();
  if (sqrt_s >= e_min && sqrt_s <= lowe_emax()) { return v.value(sqrt_s); }
  if (sqrt_s < e_min) { return v.value(e_min); }
  return 0.0;
}

/// G4XnpTotalLowE::CrossSection - np only, same shape, its own table.
__host__ __device__ inline double x_np_total_lowe(int pdg1, int pdg2, double sqrt_s) {
  const bool is_np = (pdg1 == kPdgProton && pdg2 == kPdgNeutron) ||
                     (pdg1 == kPdgNeutron && pdg2 == kPdgProton);
  if (!is_np) { return 0.0; }
  const LogVec101 v = LogVec101::make(np_total_lowe(), lowe_emin_np(), lowe_emax());
  const double e_min = lowe_emin_np();
  if (sqrt_s >= e_min && sqrt_s <= lowe_emax()) { return v.value(sqrt_s); }
  if (sqrt_s < e_min) { return v.value(e_min); }
  return 0.0;
}

// =============================================================================================
// G4XPDGTotal and G4XPDGElastic.
// =============================================================================================

/// The PDG fit rows this file carries, in the order `G4XPDGTotal`'s and `G4XPDGElastic`'s maps
/// would be searched. `kPdgPairNone` is a refusal, not a default.
enum PdgPair { kPdgPairNone = 0, kPdgPairPP, kPdgPairPN, kPdgPairPiP };

/// The mass-ordered pair `G4XPDGTotal` and `G4XPDGElastic` both key by: `trkPair(def1, def2)`
/// swapped when `def1` is the heavier. Both maps hold `(proton, proton)`, `(neutron, neutron)`
/// and `(proton, neutron)` - note the last is stored PROTON FIRST even though the neutron is
/// heavier, so an (n, p) pair is swapped to (p, n) and hits it, and an (n, n) pair hits the
/// separate `nn` key, which was given `ppPDGFit` in both classes.
__host__ __device__ inline PdgPair pdg_pair_of(int pdg1, int pdg2, double m1, double m2) {
  int a = pdg1;
  int b = pdg2;
  if (m1 > m2) {
    a = pdg2;
    b = pdg1;
  }
  if (a == kPdgProton && b == kPdgProton) { return kPdgPairPP; }
  if (a == kPdgNeutron && b == kPdgNeutron) { return kPdgPairPP; }
  if (a == kPdgProton && b == kPdgNeutron) { return kPdgPairPN; }
  if ((a == kPdgPiPlus || a == kPdgPiMinus) && b == kPdgProton) { return kPdgPairPiP; }
  return kPdgPairNone;
}

/// G4XPDGTotal::CrossSection - the 1998 Review of Particle Properties Regge fit
/// `X s^eps + Y1 s^eta1 +- Y2 s^eta2`, with the sign of the third term set by whether the two
/// PDG encodings have OPPOSITE signs (particle-antiparticle) or not.
///
/// Geant4 prints a warning outside [eMinFit, eMaxFit] and computes the fit anyway; the warning
/// is dropped and the arithmetic kept, because the composite calls this at every energy above
/// 3 GeV and eMaxFit is 40 GeV for np - so the "outside the fit range" branch is the normal
/// one above 40 GeV and printing it would be printing per collision.
__host__ __device__ inline double x_pdg_total(int pdg1, int pdg2, double m1, double m2,
                                              double sqrt_s, XsecRefusal& ref) {
  const PdgPair pair = pdg_pair_of(pdg1, pdg2, m1, m2);
  if (pair == kPdgPairNone) {
    ref.pair = true;
    ref.pdg1 = pdg1;
    ref.pdg2 = pdg2;
    return 0.0;
  }
  const double* fit = (pair == kPdgPairPP)   ? pdg_total_pp()
                      : (pair == kPdgPairPN) ? pdg_total_np()
                                             : pdg_total_pip();
  // `coeff` is +1 only when exactly one encoding is negative. pi- has encoding -211 and the
  // proton +2212, so pi-p takes the PLUS sign and pi+p the minus - which is the whole of the
  // pi+/pi- difference in this class, since both share `pipPDGFit`.
  const double enc1 = static_cast<double>(pdg1);
  const double enc2 = static_cast<double>(pdg2);
  double coeff = -1.0;
  if ((enc1 < 0 && enc2 > 0) || (enc2 < 0 && enc1 > 0)) { coeff = 1.0; }
  const double x_fit = fit[2];
  const double y1_fit = fit[3];
  const double y2_fit = fit[4];
  constexpr double epsilon = 0.095;
  constexpr double eta1 = -0.34;
  constexpr double eta2 = -0.55;
  const double gev = u::GeV<double>();
  const double s = (sqrt_s * sqrt_s) / (gev * gev);
  // `G4Pow::powA`, not `std::pow`: powA is a third-order expansion about a tabulated point
  // (src/data/g4pow.hh says why it is not std::pow and what the difference is worth), and the
  // three exponents here are all non-integer, so every one of the three terms takes it.
  using g4gpu::data::g4pow_pow_a;
  double sigma = (x_fit * g4pow_pow_a<double>(s, epsilon) +
                  y1_fit * g4pow_pow_a<double>(s, eta1) +
                  coeff * y2_fit * g4pow_pow_a<double>(s, eta2)) *
                 millibarn();
  if (sigma < 0.0) { sigma = 0.0; }  // Geant4 warns and clamps
  return sigma;
}
__host__ __device__ inline bool x_pdg_total_valid(double e) { return e >= 3.0 * u::GeV<double>(); }
__host__ __device__ inline double x_pdg_total_low_limit() { return 3.0 * u::GeV<double>(); }

/// G4XPDGElastic::CrossSection - `a + b p^n + c log^2 p + d log p` in the LAB momentum, where
/// "lab" is defined by dividing by twice the HEAVIER mass and not by the target's.
///
/// The parameter order in the stored row is not the order it is read in: `aFit = data[2]`,
/// `bFit = data[3]`, `cFit = data[5]`, `dFit = data[6]`, `nFit = data[4]`. Transcribed by index
/// rather than by name for that reason.
__host__ __device__ inline double x_pdg_elastic(int pdg1, int pdg2, double m1, double m2,
                                                double sqrt_s, XsecRefusal& ref) {
  double sigma = 0.0;
  const double m_max = (m1 > m2) ? m1 : m2;
  if (!(m_max > 0.0 && sqrt_s > (m1 + m2))) { return 0.0; }
  double p_lab = std::sqrt((sqrt_s * sqrt_s - (m1 + m2) * (m1 + m2)) *
                           (sqrt_s * sqrt_s - (m1 - m2) * (m1 - m2))) /
                 (2.0 * m_max);
  const PdgPair pair = pdg_pair_of(pdg1, pdg2, m1, m2);
  if (pair == kPdgPairNone) {
    ref.pair = true;
    ref.pdg1 = pdg1;
    ref.pdg2 = pdg2;
    return 0.0;
  }
  // `xMap[pn] = ppData` in this class - there is no separate np elastic fit, unlike the total.
  const double* fit = (pair == kPdgPairPiP)
                          ? ((pdg1 == kPdgPiPlus || pdg2 == kPdgPiPlus) ? pdg_elastic_pip()
                                                                        : pdg_elastic_pim())
                          : pdg_elastic_pp();
  const double p_min_fit = fit[0] * u::GeV<double>();
  const double a_fit = fit[2];
  const double b_fit = fit[3];
  const double n_fit = fit[4];
  const double c_fit = fit[5];
  const double d_fit = fit[6];
  if (p_lab < p_min_fit) { return 0.0; }
  p_lab /= u::GeV<double>();
  if (p_lab > 0.0) {
    const double log_p = std::log(p_lab);
    sigma = a_fit + b_fit * g4gpu::data::g4pow_pow_a<double>(p_lab, n_fit) +
            c_fit * log_p * log_p + d_fit * log_p;
    sigma = sigma * millibarn();
  }
  if (sigma < 0.0) { sigma = 0.0; }  // Geant4 warns and clamps
  return sigma;
}
__host__ __device__ inline bool x_pdg_elastic_valid(double e) {
  return e >= 5.0 * u::GeV<double>();
}
__host__ __device__ inline double x_pdg_elastic_low_limit() { return 5.0 * u::GeV<double>(); }

// =============================================================================================
// The four G4CrossSectionPatch composites.
// =============================================================================================

/// `G4CrossSectionPatch::CrossSection` for a two-component patch: the low-energy arm if it is
/// valid, the high-energy arm if IT is, and a linear blend in between when the gap is real.
///
/// Written out for two components rather than as a loop, because the loop ASSIGNS rather than
/// accumulates - `crossSection = component->CrossSection(...)` - so with two components it is
/// exactly this `if`/`else if`/`else if`, and a reader who saw a `+=` in
/// `G4CrossSectionComposite` next door would otherwise assume the wrong one.
///
/// The blend is `(1-r) sigma_low + r sigma_high` with `r = (ecm - lowHigh)/(highLow - lowHigh)`,
/// and BOTH arms are evaluated outside their own validity to get there: the tabulated arm runs
/// off the top of its table (which is why the table's own range extends to 5.2 GeV) and the PDG
/// arm below its fit range (where `G4XPDGElastic` returns 0 for `pLab < pMinFit`).
///
/// Which arm is EVALUATED matters and not only which is returned, because an arm that is asked
/// for a pair it has no row for sets a refusal. So the four composites below call an arm exactly
/// when Geant4's loop calls it, rather than computing both and choosing.
enum PatchArm { kPatchNone = 0, kPatchLow, kPatchHigh, kPatchBlend };

/// Which arm `G4CrossSectionPatch::CrossSection` would take, for a two-component patch.
__host__ __device__ inline PatchArm patch_arm(double ecm, bool low_valid, double low_high_limit,
                                              bool high_valid, double high_low_limit) {
  if (high_valid) { return kPatchHigh; }  // i = 1's assignment overwrites i = 0's
  if (low_valid) { return kPatchLow; }
  if (ecm > low_high_limit && ecm < high_low_limit) {
    const double denom = high_low_limit - low_high_limit;
    const double diff = ecm - low_high_limit;
    if (denom > 0.0 && diff > 0.0) { return kPatchBlend; }
  }
  return kPatchNone;
}

/// `G4CrossSectionPatch::Transition`'s weighted sum, once both arms have been evaluated.
__host__ __device__ inline double patch_blend(double ecm, double low_high_limit,
                                              double high_low_limit, double sigma_low,
                                              double sigma_high) {
  const double ratio = (ecm - low_high_limit) / (high_low_limit - low_high_limit);
  return (1.0 - ratio) * sigma_low + ratio * sigma_high;
}

/// G4XNNTotal - `G4CollisionNN`'s own cross-section source, and therefore the total NN cross
/// section `G4Scatterer::GetTimeToInteraction` turns into an impact-parameter disc.
__host__ __device__ inline double x_nn_total(int pdg1, int pdg2, double m1, double m2,
                                             double sqrt_s, XsecRefusal& ref) {
  const double lo_hi = x_nn_total_lowe_high_limit();
  const double hi_lo = x_pdg_total_low_limit();
  switch (patch_arm(sqrt_s, x_nn_total_lowe_valid(sqrt_s), lo_hi, x_pdg_total_valid(sqrt_s),
                    hi_lo)) {
    case kPatchLow: return x_nn_total_lowe(pdg1, pdg2, sqrt_s, ref);
    case kPatchHigh: return x_pdg_total(pdg1, pdg2, m1, m2, sqrt_s, ref);
    case kPatchBlend:
      return patch_blend(sqrt_s, lo_hi, hi_lo, x_nn_total_lowe(pdg1, pdg2, sqrt_s, ref),
                         x_pdg_total(pdg1, pdg2, m1, m2, sqrt_s, ref));
    default: return 0.0;
  }
}

/// G4XNNElastic - `G4CollisionNNElastic`'s, i.e. pp and nn.
__host__ __device__ inline double x_nn_elastic(int pdg1, int pdg2, double m1, double m2,
                                               double sqrt_s, XsecRefusal& ref) {
  const double lo_hi = x_nn_elastic_lowe_high_limit();
  const double hi_lo = x_pdg_elastic_low_limit();
  switch (patch_arm(sqrt_s, x_nn_elastic_lowe_valid(sqrt_s), lo_hi,
                    x_pdg_elastic_valid(sqrt_s), hi_lo)) {
    case kPatchLow: return x_nn_elastic_lowe(pdg1, pdg2, sqrt_s, ref);
    case kPatchHigh: return x_pdg_elastic(pdg1, pdg2, m1, m2, sqrt_s, ref);
    case kPatchBlend:
      return patch_blend(sqrt_s, lo_hi, hi_lo, x_nn_elastic_lowe(pdg1, pdg2, sqrt_s, ref),
                         x_pdg_elastic(pdg1, pdg2, m1, m2, sqrt_s, ref));
    default: return 0.0;
  }
}

/// G4XnpElastic - `G4CollisionnpElastic`'s. Its low-energy arm is `G4XnpElasticLowE`, whose
/// `IsValid` and `HighLimit` are the same 3 GeV as `G4XNNElasticLowE`'s.
__host__ __device__ inline double x_np_elastic(int pdg1, int pdg2, double m1, double m2,
                                               double sqrt_s, XsecRefusal& ref) {
  const double lo_hi = x_nn_elastic_lowe_high_limit();
  const double hi_lo = x_pdg_elastic_low_limit();
  switch (patch_arm(sqrt_s, x_nn_elastic_lowe_valid(sqrt_s), lo_hi,
                    x_pdg_elastic_valid(sqrt_s), hi_lo)) {
    case kPatchLow: return x_np_elastic_lowe(pdg1, pdg2, sqrt_s);
    case kPatchHigh: return x_pdg_elastic(pdg1, pdg2, m1, m2, sqrt_s, ref);
    case kPatchBlend:
      return patch_blend(sqrt_s, lo_hi, hi_lo, x_np_elastic_lowe(pdg1, pdg2, sqrt_s),
                         x_pdg_elastic(pdg1, pdg2, m1, m2, sqrt_s, ref));
    default: return 0.0;
  }
}

/// G4XnpTotal. Reached by `G4CollisionPN` alone, which `G4Scatterer` never registers; here
/// because it is one line over arms that exist anyway, and so that a future package that wires
/// `G4CollisionPN` up does not have to re-derive it. Nothing in this port calls it.
__host__ __device__ inline double x_np_total(int pdg1, int pdg2, double m1, double m2,
                                             double sqrt_s, XsecRefusal& ref) {
  const double lo_hi = x_nn_elastic_lowe_high_limit();  // G4XnpTotalLowE's InLimits(0, 3 GeV)
  const double hi_lo = x_pdg_total_low_limit();
  switch (patch_arm(sqrt_s, x_nn_elastic_lowe_valid(sqrt_s), lo_hi, x_pdg_total_valid(sqrt_s),
                    hi_lo)) {
    case kPatchLow: return x_np_total_lowe(pdg1, pdg2, sqrt_s);
    case kPatchHigh: return x_pdg_total(pdg1, pdg2, m1, m2, sqrt_s, ref);
    case kPatchBlend:
      return patch_blend(sqrt_s, lo_hi, hi_lo, x_np_total_lowe(pdg1, pdg2, sqrt_s),
                         x_pdg_total(pdg1, pdg2, m1, m2, sqrt_s, ref));
    default: return 0.0;
  }
}

}  // namespace g4gpu::bic::imr

#endif
