// G4ElectroNuclearCrossSection - the CHIPS electro-nuclear parameterisation, and the three
// equivalent-photon functions G4ElectroVDNuclearModel samples from.
//
// Transcribed from Geant4 11.1.1, source/processes/hadronic/cross_sections/src/
// G4ElectroNuclearCrossSection.cc:
//   GetElementCrossSection, ThresholdEnergy, GetFunctions,
//   GetEquivalentPhotonEnergy, SolveTheEquation, GetEquivalentPhotonQ2, GetVirtualFactor,
//   HighEnergyJ1 / HighEnergyJ2 / HighEnergyJ3, Fun, DFun
//
// THE CLASS IS STATEFUL AND THE STATE IS THE INTERFACE
//
// `GetElementCrossSection` leaves `lastE`, `lastG`, `lastSig`, `lastL` and a pointer to the
// per-Z J-function cache behind, and `GetEquivalentPhotonEnergy()` takes NO arguments and reads
// all five. G4ElectroVDNuclearModel.cc says so in a comment - "Need to call
// GetElementCrossSection before calling GetEquivalentPhotonEnergy" - and it is not an
// optimisation: the sampled photon energy is a function of the electron energy and of the
// nucleus, and neither is passed. So the state is an explicit `ElnState` here, filled by
// `eln_element_xs` and consumed by the three samplers, and a sampler called on a default
// `ElnState` returns zero exactly as Geant4's `lastSig <= 0` guards do.
//
// THE A LIST IS ROUNDED HERE AND IS NOT IN THE PHOTO-NUCLEAR CLASS
//
// `GetFunctions` does `iA = static_cast<G4int>(a + .499)` before comparing against `A[nN]`, so
// carbon's NIST mean 12.010736 becomes 12 and matches `A[7]` exactly; the same element misses
// `G4PhotoNuclearCrossSection`'s `LA` entry of 12 by 0.0107 and is interpolated there. Same
// author, same year, two different rules, and the difference is visible in the fifth digit of
// a carbon cross section. Both are transcribed as written.
//
// `H` IS COMPUTED FROM THE UNROUNDED A. `lastUsedCacheEl->H = alop*Aa*(1.-.072*G4Log(Aa))` uses
// the NIST mean in amu, while `GetFunctions(Aa, ...)` rounds its own copy. The rounding is
// local to GetFunctions - `a` is passed by value - so H sees 12.010736 and the table lookup
// sees 12. That is one line apart in the source and it is the kind of thing a tidy-up breaks.
//
// WHAT IS REFUSED, BY NAME
//
//   * Z outside 1..98: no NIST mean atomic mass in this port (`kPhotoNuclearNoAtomicMass`).
//     Geant4's own `IsElementApplicable` is `Z>0 && Z<120`.
//   * Nothing else. `ThresholdEnergy` returning `infEn` for a nuclide AME2012 does not carry
//     is Geant4's answer, not a gap, and it makes the cross section identically zero.
//
// WHAT IS TRANSCRIBED AND CANNOT BE REACHED. Four `G4cerr` arms - the *HP* warning, the
// `lastL < mLL` warning, the `phLE > lastLE` correction and `SolveTheEquation`'s two - print
// and continue in Geant4. They change results only through the `phLE > lastLE` arm, which
// REWRITES the answer, so that one is ported; the other three are diagnostics and are recorded
// in `ElnSampleStatus` rather than printed, so a test can assert they did not fire.
#ifndef G4GPU_HADRONIC_XS_CHIPS_ELECTRONUCLEAR_CUH
#define G4GPU_HADRONIC_XS_CHIPS_ELECTRONUCLEAR_CUH

#include <cmath>

#include "core/units.cuh"
#include "data/atomic_masses.cuh"
#include "data/chips_electronuclear.hh"
#include "data/nuclei_mass_ame12.hh"
#include "physics/hadronic/xs/chips_photonuclear.cuh"
#include "physics/hadronic/xs/refusal.cuh"

namespace g4gpu::hadronic::xs {

namespace chips {

// ---------------------------------------------------------------------------------------------
// The file-static constants of G4ElectroNuclearCrossSection.cc
//
// Every derived constant is derived here the same way it is derived there. `lmel = G4Log(mel)`
// with mel = 0.5109989 - NOTE that this is not CLHEP's electron_mass_c2 (0.510998910); the
// class writes its own seven-digit copy and the two differ in the eighth digit, which is a
// 2e-8 relative shift in every `lastG` and therefore in every sampled photon energy. The
// class's own number is what is used.
// ---------------------------------------------------------------------------------------------

__host__ __device__ inline constexpr double eln_mel() { return 0.5109989; }
__host__ __device__ inline constexpr double eln_mel2() { return eln_mel() * eln_mel(); }
__host__ __device__ inline double eln_lmel() { return std::log(eln_mel()); }

__host__ __device__ inline constexpr int eln_nE() { return data::kChipsElnN; }
__host__ __device__ inline constexpr int eln_mLL() { return data::kChipsElnN - 1; }
__host__ __device__ inline constexpr double eln_EMi() { return 2.0612; }
__host__ __device__ inline constexpr double eln_EMa() { return 50000.0; }
__host__ __device__ inline constexpr double eln_EMa2() { return eln_EMa() * eln_EMa(); }
__host__ __device__ inline double eln_lEMi() { return std::log(eln_EMi()); }
__host__ __device__ inline double eln_lEMa() { return std::log(eln_EMa()); }
__host__ __device__ inline double eln_lEMa2() { return eln_lEMa() * eln_lEMa(); }
__host__ __device__ inline double eln_dlnE() { return (eln_lEMa() - eln_lEMi()) / eln_mLL(); }
/// alop = 1/137.036/pi, with the class's own seven-digit pi (3.14159265) and not CLHEP's.
__host__ __device__ inline constexpr double eln_alop() {
  return 1.0 / 137.036 / 3.14159265;
}
__host__ __device__ inline double eln_le1() { return (eln_lEMa() - 1.0) * eln_EMa(); }
__host__ __device__ inline double eln_leh() { return (eln_lEMa() - 0.5) * eln_EMa2(); }

__host__ __device__ inline constexpr double eln_ha() { return poc() * 0.5; }
__host__ __device__ inline constexpr double eln_hab() { return eln_ha() * pos(); }
__host__ __device__ inline constexpr double eln_ab() { return poc() * pos(); }
__host__ __device__ inline constexpr double eln_d1() { return 1.0 - reg(); }
__host__ __device__ inline constexpr double eln_d2() { return 2.0 - reg(); }
__host__ __device__ inline constexpr double eln_cd() { return shd() / reg(); }
__host__ __device__ inline constexpr double eln_cd1() { return shd() / eln_d1(); }
__host__ __device__ inline constexpr double eln_cd2() { return shd() / eln_d2(); }
__host__ __device__ inline double eln_ele() { return std::exp(-reg() * eln_lEMa()); }
__host__ __device__ inline double eln_ele1() { return std::exp(eln_d1() * eln_lEMa()); }
__host__ __device__ inline double eln_ele2() { return std::exp(eln_d2() * eln_lEMa()); }
/// phte - the cross section on the high table edge, used for SolveTheEquation's first guess.
__host__ __device__ inline double eln_phte() {
  return poc() * (eln_lEMa() - pos()) + shd() * eln_ele();
}
/// imax / eps - SolveTheEquation's Newton iteration limit and accuracy.
__host__ __device__ inline constexpr int eln_imax() { return 27; }
__host__ __device__ inline constexpr double eln_eps() { return 0.001; }

/// dM = 938.27 + 939.57 - the "mean double nucleon mass" GetVirtualFactor divides by. It is a
/// pair of five-digit decimals in the source and NOT CLHEP's proton and neutron masses; the
/// difference is 3e-6 relative and it scales x in the form factor.
__host__ __device__ inline constexpr double eln_dM() { return 938.27 + 939.57; }
__host__ __device__ inline constexpr double eln_Q0() { return 843.0; }
__host__ __device__ inline constexpr double eln_Q02() { return eln_Q0() * eln_Q0(); }
__host__ __device__ inline double eln_blK0() { return std::log(185.0); }
__host__ __device__ inline constexpr double eln_bp() { return 0.85; }
__host__ __device__ inline double eln_clK0() { return std::log(1390.0); }
__host__ __device__ inline constexpr double eln_cp() { return 3.0; }

// ---------------------------------------------------------------------------------------------
// The per-Z J-function rows, as a mix rather than a materialised array
// ---------------------------------------------------------------------------------------------

/// `cacheEl_t` without the three `new G4double[nE]` arrays: which two table rows the J
/// functions are mixed from, plus the three scalars the cache carries.
struct ElnNucleus {
  int k1 = 0;          ///< lower row of A[]
  int k = 0;           ///< upper row
  double b = 0.0;      ///< weight of the upper row
  int exact = -1;      ///< >= 0: that row verbatim
  int F = 0;           ///< `r`, the low channel - where the cumulative walk starts
  double TH = 0.0;     ///< ThresholdEnergy(Z, N)
  double H = 0.0;      ///< alop*Aa*(1 - .072*ln(Aa)), from the UNROUNDED Aa
  bool known = false;  ///< false only when this port could not build it
};

/// G4ElectroNuclearCrossSection::ThresholdEnergy(Z, N).
///
/// NOT the same function as G4PhotoNuclearCrossSection's, although the two are three lines
/// apart in shape: the A == 1 value is the pi0 mass 134.9766 here and 144.6821 there (the
/// lab-frame pi0 production threshold), and BOTH the proton-separation and neutron-separation
/// branches `return infEn` when the daughter is not in AME2012, where the photo-nuclear
/// version leaves the other branch to answer. So a nuclide with one missing daughter is
/// infinite here and finite there.
template <typename real_t>
__host__ __device__ inline real_t eln_threshold_energy(int Z, int N) {
  const int Aa = Z + N;
  if (Aa < 1) { return real_t(inf_en()); }
  if (Aa == 1) { return real_t(134.9766); }

  if (!data::ame12::is_in_table(Z, Aa)) { return real_t(inf_en()); }
  const real_t mT = data::nuclear_mass<real_t>(Aa, Z);

  if (!(Z != 0 && data::ame12::is_in_table(Z - 1, Aa - 1))) { return real_t(inf_en()); }
  const real_t mP = data::nuclear_mass<real_t>(Aa - 1, Z - 1);
  if (!(N != 0 && data::ame12::is_in_table(Z, Aa - 1))) { return real_t(inf_en()); }
  const real_t mN = data::nuclear_mass<real_t>(Aa - 1, Z);

  const real_t dP = mP + units::proton_mass_c2<real_t>() - mT;
  real_t dN = mN + units::neutron_mass_c2<real_t>() - mT;
  if (dP < dN) { dN = dP; }
  return dN;
}

/// G4ElectroNuclearCrossSection::GetFunctions, reduced to "which rows and with what weight",
/// plus the low channel `r` it returns.
///
/// The source's loop runs the exact-match test for EVERY i and the interpolation once, inside
/// the same loop body, guarded by `r < 0`. So a later exact match overwrites an earlier
/// interpolation and the net rule is: the LAST exact match if there is one, otherwise the
/// interpolation between the bracketing rows with `r = min(LL[k], LL[k1])`.
__host__ __device__ inline ElnNucleus eln_rows(double a_amu) {
  ElnNucleus n;
  // `iA = static_cast<G4int>(a+.499); if(a!=ai) a=ai;` - rounds to the nearest integer, so the
  // table match below is against an integer A and not against the NIST mean.
  const int iA = static_cast<int>(a_amu + 0.499);
  const double a = static_cast<double>(iA);

  int r = -1;
  for (int i = 0; i < data::kChipsElnA; ++i) {
    if (std::fabs(a - data::chips_eln_a_list()[i]) < 0.0005) {
      n.exact = i;
      r = data::chips_eln_low()[i];
    }
    if (r < 0) {
      int k = 0;
      for (k = 1; k < data::kChipsElnA; ++k) {
        if (a < data::chips_eln_a_list()[k]) { break; }
      }
      if (k < 1) { k = 1; }
      if (k >= data::kChipsElnA) { k = data::kChipsElnA - 1; }
      n.k = k;
      n.k1 = k - 1;
      const double xi = data::chips_eln_a_list()[n.k1];
      n.b = (a - xi) / (data::chips_eln_a_list()[k] - xi);
      r = data::chips_eln_low()[k];
      if (data::chips_eln_low()[n.k1] < r) { r = data::chips_eln_low()[n.k1]; }
    }
  }
  n.F = r;
  n.known = true;
  return n;
}

/// J1[i], J2[i], J3[i] of the mixed row - the `xx`, `yy`, `zz` arrays GetFunctions would fill.
__host__ __device__ inline double eln_J1(const ElnNucleus& n, int i) {
  if (n.exact >= 0) { return data::chips_eln_j1()[(n.exact) * data::kChipsElnN + (i)]; }
  const double xi = data::chips_eln_j1()[(n.k1) * data::kChipsElnN + (i)];
  return xi + (data::chips_eln_j1()[(n.k) * data::kChipsElnN + (i)] - xi) * n.b;
}
__host__ __device__ inline double eln_J2(const ElnNucleus& n, int i) {
  if (n.exact >= 0) { return data::chips_eln_j2()[(n.exact) * data::kChipsElnN + (i)]; }
  const double yi = data::chips_eln_j2()[(n.k1) * data::kChipsElnN + (i)];
  return yi + (data::chips_eln_j2()[(n.k) * data::kChipsElnN + (i)] - yi) * n.b;
}
__host__ __device__ inline double eln_J3(const ElnNucleus& n, int i) {
  if (n.exact >= 0) { return data::chips_eln_j3()[(n.exact) * data::kChipsElnN + (i)]; }
  const double zi = data::chips_eln_j3()[(n.k1) * data::kChipsElnN + (i)];
  return zi + (data::chips_eln_j3()[(n.k) * data::kChipsElnN + (i)] - zi) * n.b;
}

/// Build the per-Z cache entry Geant4 keeps in `cache[ZZ]`.
__host__ __device__ inline ElnNucleus eln_nucleus(int Z, bool& refused) {
  refused = false;
  ElnNucleus n;
  if (Z < 1 || Z > 98) { refused = true; return n; }
  const double Aa = data::nist_atomic_mass_table()[Z];
  const int N = static_cast<int>(Aa) - Z;
  n = eln_rows(Aa);
  n.H = eln_alop() * Aa * (1.0 - 0.072 * std::log(Aa));
  n.TH = double(eln_threshold_energy<double>(Z, N));
  return n;
}

// ---------------------------------------------------------------------------------------------
// The three high-energy integrals and the two functions Newton's method uses
// ---------------------------------------------------------------------------------------------

__host__ __device__ inline double eln_high_J1(double lE) {
  return eln_ha() * (lE * lE - eln_lEMa2()) - eln_ab() * (lE - eln_lEMa())
         - eln_cd() * (std::exp(-reg() * lE) - eln_ele());
}
__host__ __device__ inline double eln_high_J2(double lE, double E) {
  return poc() * ((lE - 1.0) * E - eln_le1()) - eln_ab() * (E - eln_EMa())
         + eln_cd1() * (std::exp(eln_d1() * lE) - eln_ele1());
}
__host__ __device__ inline double eln_high_J3(double lE, double E2) {
  return eln_ha() * ((lE - 0.5) * E2 - eln_leh()) - eln_hab() * (E2 - eln_EMa2())
         + eln_cd2() * (std::exp(eln_d2() * lE) - eln_ele2());
}

// ---------------------------------------------------------------------------------------------
// The state GetElementCrossSection leaves behind
// ---------------------------------------------------------------------------------------------

/// `lastE`, `lastG`, `lastSig`, `lastL` and the nucleus, as one object the caller owns.
struct ElnState {
  ElnNucleus nuc;
  double lastE = 0.0;    ///< the electron kinetic energy in MeV
  double lastG = 0.0;    ///< ln(E) - ln(mel), the electron's gamma in logarithmic form
  double lastSig = 0.0;  ///< the cross section in mb, BEFORE the millibarn scaling
  int lastL = 0;         ///< the top channel of the cumulative walk
  int Z = 0;
};

/// What a sampler had to say beyond its number - the G4cerr arms, so that a test can assert
/// none of them fired instead of a test that cannot see them.
struct ElnSampleStatus {
  bool hp_warning = false;        ///< the "*HP*G4ElNucCS::GetEqPhotE" arm
  bool func_region = false;       ///< the draw landed above Y[lastL] and SolveTheEquation ran
  bool lastL_below_mLL = false;   ///< "**G4EleNucCS::GetEfPhE:L=" - in the function region
  bool phle_corrected = false;    ///< the `phLE > lastLE` rewrite, which CHANGES the answer
  bool newton_exhausted = false;  ///< imax iterations without |d| < eps
  bool newton_clamped = false;    ///< "*G4ElNCS::SolveTheEq:*Correction*" - x pushed to topLim
  int q2_tries = 0;               ///< GetEquivalentPhotonQ2's while-loop count
  /// The cumulative walk ended at j == F == 0 and Geant4 read `Y[-1]`, one double off the
  /// front of a stack array. Reachable only for beryllium, whose `LL[6]` is 0, and only when
  /// the draw lands at or below `Y[0]`. Recorded rather than reproduced: this port uses the
  /// 0.0 the rest of that zero-initialised array holds, which is what the read would give on
  /// any layout where the preceding stack slot happens to be zero, and says that it did.
  bool y_index_underflow = false;
};

/// G4ElectroNuclearCrossSection::GetElementCrossSection, in the port's cross-section units.
/// Fills `st` with everything the three samplers below read.
template <typename real_t>
__host__ __device__ inline XsValue<real_t> eln_element_xs(ElnState& st, real_t ekin, int Z) {
  const double Energy = double(ekin) / double(units::MeV<real_t>());
  if (Energy <= eln_EMi()) { return {real_t(0), XsRefusal::kNone}; }

  if (Z != st.Z) {
    bool refused = false;
    st.nuc = eln_nucleus(Z, refused);
    if (refused) { return {real_t(0), XsRefusal::kPhotoNuclearNoAtomicMass}; }
    st.Z = Z;
    st.lastE = 0.0;
    st.lastG = 0.0;
  }

  st.lastE = Energy;
  if (Energy <= st.nuc.TH) {
    st.lastSig = 0.0;
    return {real_t(0), XsRefusal::kNone};
  }

  const double lE = std::log(Energy);
  st.lastG = lE - eln_lmel();
  const double dlg1 = st.lastG + st.lastG - 1.0;
  const double lgoe = st.lastG / st.lastE;

  if (lE < eln_lEMa()) {
    // "Linear fit is made explicitly to fix the last bin for the randomization"
    double shift = (lE - eln_lEMi()) / eln_dlnE();
    int blast = static_cast<int>(shift);
    if (blast < 0) { blast = 0; }
    if (blast >= eln_mLL()) { blast = eln_mLL() - 1; }
    shift -= blast;
    st.lastL = blast + 1;
    const double YNi = dlg1 * eln_J1(st.nuc, blast)
                       - lgoe * (eln_J2(st.nuc, blast) + eln_J2(st.nuc, blast)
                                 - eln_J3(st.nuc, blast) / st.lastE);
    const double YNj = dlg1 * eln_J1(st.nuc, st.lastL)
                       - lgoe * (eln_J2(st.nuc, st.lastL) + eln_J2(st.nuc, st.lastL)
                                 - eln_J3(st.nuc, st.lastL) / st.lastE);
    st.lastSig = YNi + shift * (YNj - YNi);
    if (st.lastSig > YNj) { st.lastSig = YNj; }
  } else {
    st.lastL = eln_mLL();
    const double term1 = eln_J1(st.nuc, eln_mLL()) + st.nuc.H * eln_high_J1(lE);
    const double term2 = eln_J2(st.nuc, eln_mLL()) + st.nuc.H * eln_high_J2(lE, Energy);
    const double En2 = Energy * Energy;
    const double term3 = eln_J3(st.nuc, eln_mLL()) + st.nuc.H * eln_high_J3(lE, En2);
    st.lastSig = dlg1 * term1 - lgoe * (term2 + term2 - term3 / st.lastE);
  }

  if (st.lastSig < 0.0) { st.lastSig = 0.0; }
  return {static_cast<real_t>(st.lastSig) * millibarn<real_t>(), XsRefusal::kNone};
}

/// `Y[i]` of GetEquivalentPhotonEnergy - the cumulative integrated cross section at channel i,
/// with the source's two properties that are not in the formula: the `Y[i] < 0 -> 0` clamp,
/// and the fact that `G4double Y[nE] = {0.0}` is only WRITTEN over `[F, lastL]`, so every index
/// outside that range reads the zero the array was initialised with rather than the formula.
/// Below F the two agree anyway - both J rows are zero there, F being the smaller of their two
/// low channels - but "agrees by arithmetic" and "is the same value" are different claims and
/// this is the second one.
__host__ __device__ inline double eln_Y(const ElnState& st, int i) {
  if (i < st.nuc.F || i > st.lastL) { return 0.0; }
  const double dlg1 = st.lastG + st.lastG - 1.0;
  const double lgoe = st.lastG / st.lastE;
  const double y = dlg1 * eln_J1(st.nuc, i)
                   - lgoe * (eln_J2(st.nuc, i) + eln_J2(st.nuc, i)
                             - eln_J3(st.nuc, i) / st.lastE);
  return (y < 0.0) ? 0.0 : y;
}

/// `Fun(x)` and `DFun(x)` - the integrated cross section above the table and its derivative,
/// both functions of the state SolveTheEquation is called under.
__host__ __device__ inline double eln_DFun(const ElnState& st, double x) {
  const double y = std::exp(x - st.lastG - eln_lmel());
  const double flux = st.lastG * (2.0 - y * (2.0 - y)) - 1.0;
  return (poc() * (x - pos()) + shd() * std::exp(-reg() * x)) * flux;
}
__host__ __device__ inline double eln_Fun(const ElnState& st, double x) {
  const double dlg1 = st.lastG + st.lastG - 1.0;
  const double lgoe = st.lastG / st.lastE;
  const double HE2 = eln_high_J2(x, std::exp(x));
  return dlg1 * eln_high_J1(x) - lgoe * (HE2 + HE2 - eln_high_J3(x, std::exp(2 * x)) / st.lastE);
}

/// G4ElectroNuclearCrossSection::SolveTheEquation - Newton's method on `Fun(x) = f`.
__host__ __device__ inline double eln_solve(const ElnState& st, double f,
                                            ElnSampleStatus& stat) {
  const double lastLE = st.lastG + eln_lmel();
  const double topLim = lastLE - 0.001;
  const double rE = eln_EMa() / std::exp(lastLE);
  double x = eln_lEMa() + f / eln_phte() / (st.lastG * (2.0 - rE * (2.0 - rE)) - 1.0);
  if (x > topLim) { x = topLim; }
  for (int i = 0; i < eln_imax(); ++i) {
    const double fx = eln_Fun(st, x);
    const double df = eln_DFun(st, x);
    const double d = (f - fx) / df;
    x = x + d;
    if (x >= lastLE) {
      stat.newton_clamped = true;
      x = topLim;
    }
    if (std::fabs(d) < eln_eps()) { break; }
    if (i + 1 >= eln_imax()) { stat.newton_exhausted = true; }
  }
  return x;
}

/// G4ElectroNuclearCrossSection::GetEquivalentPhotonEnergy(), MeV.
///
/// One uniform deviate, then either a cumulative walk over `Y[F..lastL]` or - when the draw
/// lands above the table's top - Newton's method on the analytic continuation. `Y[i]` is
/// recomputed from the three J functions rather than stored, for the reason
/// chips_photonuclear.cuh's header gives; the walk is `while (ris > Yj && j < lastL)`, so it
/// reads at most `lastL - F + 1` of them and never materialises the array.
template <typename Rng>
__host__ __device__ inline double eln_equivalent_photon_energy(const ElnState& st, Rng& rng,
                                                               ElnSampleStatus& stat) {
  if (st.lastSig <= 0.0) { return 0.0; }
  const double lastLE = st.lastG + eln_lmel();

  const double Y_lastL = eln_Y(st, st.lastL);

  if (st.lastSig > 0.99 * Y_lastL && st.lastL < eln_mLL() && Y_lastL < 1.E-30) {
    stat.hp_warning = true;
    // The source's `if(lastSig <= 0.0) return 0.0;` inside the warning cannot fire here:
    // lastSig > 0.99*Y_lastL >= 0 and lastSig > 0 is the function's own entry guard.
  }

  const double ris = st.lastSig * rng.uniform();
  double phLE = 0.0;

  if (ris < Y_lastL) {
    int j = st.nuc.F;
    double Yj = eln_Y(st, j);
    while (ris > Yj && j < st.lastL) {
      ++j;
      Yj = eln_Y(st, j);
    }
    // `Yi = Y[j-1]` read from the array, NOT the previous loop value - the two differ when
    // the loop never ran, which is when j is still F. `eln_Y` answers 0 for j-1 < F because
    // the source's `G4double Y[nE] = {0.0}` is only written over the range [F, lastL].
    const int j1 = j - 1;
    if (j1 < 0) { stat.y_index_underflow = true; }
    const double Yi = eln_Y(st, j1);
    phLE = eln_lEMi() + (j1 + (ris - Yi) / (Yj - Yi)) * eln_dlnE();
  } else {
    stat.func_region = true;
    if (st.lastL < eln_mLL()) { stat.lastL_below_mLL = true; }
    const double f = (ris - Y_lastL) / st.nuc.H;
    phLE = eln_solve(st, f, stat);
  }

  if (phLE > lastLE) {
    stat.phle_corrected = true;
    if (lastLE < 7.2) { phLE = std::log(std::exp(lastLE) - 0.511); }
    else { phLE = 7.0; }
  }
  return std::exp(phLE);
}

/// G4ElectroNuclearCrossSection::GetEquivalentPhotonQ2(nu), MeV^2.
///
/// Up to three uniform deviates: the `while(cond && cntTry<maxTry)` loop rejects Q2 > 1878*nu
/// and gives up after three, returning the last draw whether it passed or not. The give-up is
/// Geant4's behaviour and not a failure, so it is counted in `q2_tries` rather than refused.
template <typename Rng>
__host__ __device__ inline double eln_equivalent_photon_q2(const ElnState& st, double nu,
                                                           Rng& rng, ElnSampleStatus& stat) {
  if (st.lastG <= 0.0 || st.lastE <= 0.0) { return 0.0; }
  if (st.lastSig <= 0.0) { return 0.0; }
  const double y = nu / st.lastE;
  if (y >= 1.0 - 1.0 / (st.lastG + st.lastG)) { return 0.0; }
  const double y2 = y * y;
  const double ye = 1.0 - y;
  const double Qi2 = eln_mel2() * y2 / ye;
  const double Qa2 = 4 * st.lastE * st.lastE * ye;
  const double iar = Qi2 / Qa2;
  const double Dy = ye + 0.5 * y2;
  const double Py = ye / Dy;
  const double ePy = 1.0 - std::exp(Py);
  const double Uy = Py * (1.0 - iar);
  const double Fy = (ye + ye) * (1.0 + ye) * iar / y2;
  const double fr = iar / (1.0 - ePy * iar);
  if (Fy <= -fr) { return 0.0; }
  const double LyQa2 = std::log(Fy + fr);

  const int maxTry = 3;
  int cntTry = 0;
  bool cond = true;
  double Q2 = Qi2;
  while (cond && cntTry < maxTry) {
    const double R = rng.uniform();
    Q2 = Qi2 * (ePy + 1.0 / (std::exp(R * LyQa2 - (1.0 - R) * Uy) - Fy));
    ++cntTry;
    cond = Q2 > 1878.0 * nu;
  }
  stat.q2_tries = cntTry;
  if (Q2 < Qi2) { return Qi2; }
  if (Q2 > Qa2) { return Qa2; }
  return Q2;
}

/// G4ElectroNuclearCrossSection::GetVirtualFactor(nu, Q2) - dimensionless, no state, no draw.
__host__ __device__ inline double eln_virtual_factor(double nu, double Q2) {
  if (nu <= 0.0 || Q2 <= 0.0) { return 0.0; }
  const double K = nu - Q2 / eln_dM();
  if (K <= 0.0) { return 0.0; }
  const double lK = std::log(K);
  const double x = 1.0 - K / nu;
  const double GD = 1.0 + Q2 / eln_Q02();
  const double b = std::exp(eln_bp() * (lK - eln_blK0()));
  const double c = std::exp(eln_cp() * (lK - eln_clK0()));
  const double r = 0.5 * std::log(Q2 + nu * nu) - lK;
  const double ef = std::exp(r * (b - c * r * r));
  return (1.0 - x) * ef / GD / GD;
}

}  // namespace chips

}  // namespace g4gpu::hadronic::xs

#endif
