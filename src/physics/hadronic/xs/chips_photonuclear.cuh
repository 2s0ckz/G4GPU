// G4PhotoNuclearCrossSection - the CHIPS photo-nuclear parameterisation, and the two numbers
// G4GammaNuclearXS needs from it.
//
// Transcribed from Geant4 11.1.1, source/processes/hadronic/cross_sections/src/
// G4PhotoNuclearCrossSection.cc:
//   G4PhotoNuclearCrossSection::GetElementCrossSection
//   G4PhotoNuclearCrossSection::GetIsoCrossSection   (the D, T and He3 arms)
//   G4PhotoNuclearCrossSection::ThresholdEnergy
//   G4PhotoNuclearCrossSection::EquLinearFit
//   G4PhotoNuclearCrossSection::GetFunctions
//
// WHY THIS FILE IS IN P2's DIRECTORY AND NOT P13's
//
// docs/PORTED.md 2.1.1 records `G4GammaNuclearXS` as `P` with 1,128 element points and 760
// isotope points refused, all of them `XsRefusal::kPhotoNuclearCrossSection`: the IAEA data
// files stop at 130 MeV for most elements, and above that - and for hydrogen at any energy,
// and in the straight line between the table's top and 150 MeV, which is anchored on
// `xs150[Z]` = CHIPS at 150 MeV - Geant4 evaluates this class. So this is P2's row finished
// rather than a new one, and it goes beside the reader it completes. The P13 brief names this
// as one of two allowed exceptions to that directory's ownership.
//
// THE CLASS IS A CACHE AROUND A PURE FUNCTION, AND THE CACHE IS NOT PORTED
//
// Geant4 keeps `GDR[120]` and `HEN[120]` arrays of `new G4double[nL]` / `new G4double[nH]`,
// built the first time a Z is asked for by `GetFunctions(Aa, ...)`, plus `lastZ/lastE/lastSig`
// scalars. Every one of those is a memo: `GetFunctions` is a linear interpolation between two
// rows of the static tables with ONE coefficient `b` shared by all 105 (or 224) entries, so
//
//     y[q] = SL[k1][q] + (SL[k][q] - SL[k1][q]) * b
//
// and `EquLinearFit` then reads exactly two of those entries. Building the whole row to read
// two of them is what a host cache is for and what a device kernel must not do, so the two
// entries are built where they are needed. That is bit-identical and not an approximation:
// each `y[j]` is a double in Geant4 too, computed by the same expression in the same order,
// and the final `yi + (Y[j+1]-yi)*d` sees the same two doubles. It is asserted, not asserted
// to: `emextra_photonuc.csv` dumps Geant4's answer at 8,832 (Z, E) points and the test
// compares at 1e-15 relative - a fused or reordered form would not survive that.
//
// WHAT A "NUCLEUS" IS HERE: `nistmngr->GetAtomicMassAmu(ZZ)`, the NIST mean atomic mass in
// amu, NOT an isotope's A and NOT an integer. `GetFunctions` matches it against the tabulated
// A list with a +-0.0005 window, so carbon's 12.010736 does NOT match the table's 12 and takes
// the interpolating branch between 9 and 12. (The electro-nuclear class rounds first and does
// match - see chips_electronuclear.cuh. Same author, same year, different rule.)
//
// WHAT IS REFUSED, BY NAME
//
//   * Z outside 1..98 - `data::nist_atomic_mass_table()` stops at 98 and Geant4 would read a
//     NIST entry this port does not carry. `XsRefusal::kPhotoNuclearNoAtomicMass`.
//   * A nuclide whose mass `ThresholdEnergy` needs and AME2012 does not have. Geant4 returns
//     `infEn` = 9e27 for it, which makes the cross section identically zero, and this port
//     returns the same `infEn` - so that IS ported, not refused. The refusal above it is
//     `data::nuclear_mass_known`, which distinguishes "Geant4 says infinity" from "this port
//     cannot say", and only the second is a refusal.
#ifndef G4GPU_HADRONIC_XS_CHIPS_PHOTONUCLEAR_CUH
#define G4GPU_HADRONIC_XS_CHIPS_PHOTONUCLEAR_CUH

#include <cmath>

#include "core/units.cuh"
#include "data/atomic_masses.cuh"
#include "data/chips_photonuclear.hh"
#include "data/nuclei_mass_ame12.hh"
#include "physics/hadronic/xs/refusal.cuh"

namespace g4gpu::hadronic::xs {

namespace chips {

// ---------------------------------------------------------------------------------------------
// The file-static constants at the top of G4PhotoNuclearCrossSection.cc, with their own names.
//
// `Emin`, `milE`, `malE` and `dlE` are computed there from THmin, nL, dE and Emax, and they are
// computed here the same way rather than pasted as decimals: `milE = G4Log(Emin)` with
// Emin = 2 + 104*1 = 106 is a transcendental of an exact integer, so any decimal spelling of it
// is a rounding this port would have chosen and Geant4 would not.
// ---------------------------------------------------------------------------------------------

/// THmin - the minimum energy at which any photo-nuclear cross section is non-zero, MeV.
__host__ __device__ inline constexpr double th_min() { return 2.0; }
/// dE - the GDR table's step in E, MeV.
__host__ __device__ inline constexpr double d_e() { return 1.0; }
/// Emax - the top of the high-energy table, MeV.
__host__ __device__ inline constexpr double e_max() { return 50000.0; }
/// Emin = THmin + (nL-1)*dE - where the GDR table ends and the ln(E) table begins, MeV.
__host__ __device__ inline constexpr double e_min() {
  return th_min() + (data::kChipsGdrN - 1) * d_e();
}
/// milE = G4Log(Emin), and the step dlE = (malE - milE)/(nH-1).
__host__ __device__ inline double mil_e() { return std::log(e_min()); }
__host__ __device__ inline double mal_e() { return std::log(e_max()); }
__host__ __device__ inline double dl_e() {
  return (mal_e() - mil_e()) / (data::kChipsHenN - 1);
}

/// The four ultra-high-energy Regge/Pomeron constants. `shd` is written as the decimal 1.0734
/// in the source with the exact expression `1.075-.0023*G4Log(2.)` commented out beside it;
/// the decimal is what runs and is what is here.
__host__ __device__ inline constexpr double shd() { return 1.0734; }
__host__ __device__ inline constexpr double shc() { return 0.072; }
__host__ __device__ inline constexpr double poc() { return 0.0375; }
__host__ __device__ inline constexpr double pos() { return 16.5; }
__host__ __device__ inline constexpr double reg() { return 0.11; }
/// `infEn` - what ThresholdEnergy returns for a nuclide with no mass. Not an error code in
/// Geant4: it is compared against an energy and makes the cross section zero.
__host__ __device__ inline constexpr double inf_en() { return 9.e27; }

/// CLHEP's millibarn, derived the way CLHEP derives it - see core/units.cuh on why the decimal
/// spelling of barn is one ulp away and where that mattered.
template <typename real_t> __host__ __device__ inline constexpr real_t millibarn() {
  return real_t(1.e-3) * units::barn<real_t>();
}

// ---------------------------------------------------------------------------------------------
// G4PhotoNuclearCrossSection::ThresholdEnergy(Z, N)
// ---------------------------------------------------------------------------------------------

/// Lab-frame threshold, MeV. The four light cases are hard numbers in the source; everything
/// else is `min(mP + m_proton, mN + m_neutron) - mT` over AME2012 masses, and `infEn` when any
/// of the three is not in the table.
///
/// `known` is set false only when this port cannot evaluate what Geant4 would have evaluated.
/// A nuclide Geant4 itself calls `infEn` for comes back as `infEn` with `known` true, because
/// that is an answer and not a gap.
template <typename real_t>
__host__ __device__ inline real_t threshold_energy(int Z, int N, bool& known) {
  known = true;
  const int A = Z + N;
  if (A < 1) { return real_t(inf_en()); }
  if (A == 1) { return real_t(144.6821); }            // pi0 production off a nucleon
  if (Z == 1 && N == 1) { return real_t(2.2263); }    // deuteron disintegration
  if (Z == 1 && N == 2) { return real_t(6.2650); }    // triton, n separation
  if (Z == 2 && N == 1) { return real_t(5.4994); }    // He3, p separation

  if (!data::ame12::is_in_table(Z, A)) { return real_t(inf_en()); }
  const real_t mT = data::nuclear_mass<real_t>(A, Z);

  real_t mP = real_t(inf_en());
  if (Z != 0 && data::ame12::is_in_table(Z - 1, A - 1)) {
    mP = data::nuclear_mass<real_t>(A - 1, Z - 1);
  }
  real_t mN = real_t(inf_en());
  if (N != 0 && data::ame12::is_in_table(Z, A - 1)) {
    mN = data::nuclear_mass<real_t>(A - 1, Z);
  }
  const real_t dP = mP + units::proton_mass_c2<real_t>() - mT;
  real_t dN = mN + units::neutron_mass_c2<real_t>() - mT;
  if (dP < dN) { dN = dP; }
  return dN;
}

// ---------------------------------------------------------------------------------------------
// G4PhotoNuclearFunctions - two table entries instead of a whole row
// ---------------------------------------------------------------------------------------------

/// Which two rows of `SL`/`SH` `GetFunctions` mixes, and with what weight.
///
/// `exact >= 0` means the A matched a tabulated nucleus within 0.0005 and the row is copied,
/// not mixed. Note that Geant4 runs the exact-match test for EVERY i and the interpolation for
/// the first i at which `r` is still negative, so a later exact match overwrites an earlier
/// interpolation; the net result is "the LAST exact match if there is one, otherwise the
/// interpolation", which is what this returns.
struct RowMix {
  int k1 = 0;      ///< lower row
  int k = 0;       ///< upper row
  double b = 0.0;  ///< weight of the upper row
  int exact = -1;  ///< >= 0: copy that row verbatim
  bool zero = false;  ///< the `a <= 1.5` arm of the GDR branch, which writes zeros
};

/// The GDR half of GetFunctions: `SL` indexed by the A list `LA`.
__host__ __device__ inline RowMix gdr_rows(double a) {
  RowMix m;
  for (int i = 0; i < data::kChipsGdrA; ++i) {
    if (std::fabs(a - data::chips_gdr_a_list()[i]) < 0.0005) { m.exact = i; }
  }
  if (m.exact >= 0) { return m; }
  int k = 0;
  for (k = 1; k < data::kChipsGdrA; ++k) {
    if (a < data::chips_gdr_a_list()[k]) { break; }
  }
  if (k < 1) { k = 1; }
  if (k >= data::kChipsGdrA) { k = data::kChipsGdrA - 1; }
  m.k = k;
  m.k1 = k - 1;
  const double xi = data::chips_gdr_a_list()[m.k1];
  m.b = (a - xi) / (data::chips_gdr_a_list()[k] - xi);
  // `if(a>1.5) {...} else y[q]=0.;` - hydrogen's 1.0079 takes the else and gets a GDR table of
  // zeros.
  //
  // IT IS DEAD CODE, and only a perturbation could say so: setting `m.zero = false` - i.e.
  // extrapolating the GDR rows down to A = 1.0079 instead of zeroing them - changed NOT ONE of
  // the 10,388 element and 530 isotope comparisons. The reason is that hydrogen is the only
  // element with A <= 1.5, and hydrogen's `ThresholdEnergy` is the 144.6821 MeV pi0-production
  // threshold, which is ABOVE the GDR table's ceiling of Emin = 106 MeV. So a photon on
  // hydrogen is either below the threshold (cross section exactly zero) or above 106 MeV (the
  // ln(E) table, which has no such guard). Transcribed as written and recorded as unreachable
  // rather than dropped, because "the else can be removed" and "the else cannot be reached"
  // are different claims and only the second is true.
  m.zero = !(a > 1.5);
  return m;
}

/// The high-energy half: `SH` indexed by `HA`. It has no `a > 1.5` arm.
__host__ __device__ inline RowMix hen_rows(double a) {
  RowMix m;
  for (int j = 0; j < data::kChipsHenA; ++j) {
    if (std::fabs(a - data::chips_hen_a_list()[j]) < 0.0005) { m.exact = j; }
  }
  if (m.exact >= 0) { return m; }
  int k = 0;
  for (k = 1; k < data::kChipsHenA; ++k) {
    if (a < data::chips_hen_a_list()[k]) { break; }
  }
  if (k < 1) { k = 1; }
  if (k >= data::kChipsHenA) { k = data::kChipsHenA - 1; }
  m.k = k;
  m.k1 = k - 1;
  const double xi = data::chips_hen_a_list()[m.k1];
  m.b = (a - xi) / (data::chips_hen_a_list()[k] - xi);
  return m;
}

/// One entry of the GDR row `gdr_rows` describes - the `y[q]` Geant4 would have stored.
struct GdrEntry {
  RowMix m;
  __host__ __device__ double operator()(int q) const {
    if (m.exact >= 0) { return data::chips_gdr()[(m.exact) * data::kChipsGdrN + (q)]; }
    if (m.zero) { return 0.0; }
    const double yi = data::chips_gdr()[(m.k1) * data::kChipsGdrN + (q)];
    return yi + (data::chips_gdr()[(m.k) * data::kChipsGdrN + (q)] - yi) * m.b;
  }
};

/// The same for the high-energy row.
struct HenEntry {
  RowMix m;
  __host__ __device__ double operator()(int q) const {
    if (m.exact >= 0) { return data::chips_hen()[(m.exact) * data::kChipsHenN + (q)]; }
    const double zi = data::chips_hen()[(m.k1) * data::kChipsHenN + (q)];
    return zi + (data::chips_hen()[(m.k) * data::kChipsHenN + (q)] - zi) * m.b;
  }
};

/// A single tabulated row, for the deuteron/triton/He3 arms of GetIsoCrossSection, which copy
/// `SL[0]` and `SH[1]`/`SH[2]` verbatim rather than mixing.
struct GdrRowEntry {
  int row;
  __host__ __device__ double operator()(int q) const { return data::chips_gdr()[(row) * data::kChipsGdrN + (q)]; }
};
struct HenRowEntry {
  int row;
  __host__ __device__ double operator()(int q) const { return data::chips_hen()[(row) * data::kChipsHenN + (q)]; }
};

/// G4PhotoNuclearCrossSection::EquLinearFit, with the two Y entries supplied by a functor
/// instead of read from an array - see this file's header for why the row is not materialised.
/// `N` is the table length, `X0` its first abscissa and `DX` the step; the clamp on `j` is
/// Geant4's own and is what makes the last bin a flat extrapolation rather than an
/// out-of-range read.
template <typename Entry>
__host__ __device__ inline double equ_linear_fit(double X, int N, double X0, double DX,
                                                 const Entry& entry) {
  // The `DX<=0 || N<2` arm prints and returns Y[0]; both tables here are fixed and pass it.
  const int N2 = N - 2;
  double d = (X - X0) / DX;
  int j = static_cast<int>(d);
  if (j < 0) { j = 0; }
  else if (j > N2) { j = N2; }
  d -= j;
  const double yi = entry(j);
  return yi + (entry(j + 1) - yi) * d;
}

/// The ultra-high-energy form, above Emax: `SP*(poc*(lE-pos) + shd*exp(-reg*lE))`.
__host__ __device__ inline double uhe(double sp, double lE) {
  return sp * (poc() * (lE - pos()) + shd() * std::exp(-reg() * lE));
}

/// `lastSP` - the Reggeon shadowing coefficient, a function of the mean A alone.
__host__ __device__ inline double shadowing(double Aa) {
  if (Aa == 1.0) { return 1.0; }
  return Aa * (1.0 - shc() * std::log(Aa));
}

// ---------------------------------------------------------------------------------------------
// The two public entry points
// ---------------------------------------------------------------------------------------------

/// G4PhotoNuclearCrossSection::GetElementCrossSection(gamma at `ekin`, Z).
///
/// Returns the port's cross-section units (mm^2 through `millibarn`), like every other class in
/// this directory, so that a caller can add it to a G4PARTICLEXS value without a conversion.
template <typename real_t>
__host__ __device__ inline XsValue<real_t> photo_element_xs(real_t ekin, int Z) {
  const double Energy = double(ekin) / double(units::MeV<real_t>());
  if (Energy < th_min()) { return {real_t(0), XsRefusal::kNone}; }
  if (Z < 1 || Z > 98) { return {real_t(0), XsRefusal::kPhotoNuclearNoAtomicMass}; }

  const double Aa = data::nist_atomic_mass_table()[Z];
  const int N = static_cast<int>(Aa) - Z;
  bool known = true;
  const double lastTH = double(threshold_energy<double>(Z, N, known));
  if (!known) { return {real_t(0), XsRefusal::kNuclearMassNotTabulated}; }

  double sigma = 0.0;
  if (Energy < lastTH) {
    sigma = 0.0;
  } else if (Energy < e_min()) {
    const GdrEntry e{gdr_rows(Aa)};
    sigma = equ_linear_fit(Energy, data::kChipsGdrN, th_min(), d_e(), e);
  } else if (Energy < e_max()) {
    const HenEntry e{hen_rows(Aa)};
    const double lE = std::log(Energy);
    sigma = equ_linear_fit(lE, data::kChipsHenN, mil_e(), dl_e(), e);
  } else {
    sigma = uhe(shadowing(Aa), std::log(Energy));
  }
  if (sigma < 0.0) { sigma = 0.0; }        // std::max(sigma, 0.)
  return {static_cast<real_t>(sigma) * millibarn<real_t>(), XsRefusal::kNone};
}

/// G4PhotoNuclearCrossSection::GetIsoCrossSection(gamma at `ekin`, Z, A).
///
/// Only (1,2), (1,3) and (2,3) have their own arms; every other (Z, A) falls through to
/// `GetElementCrossSection` with the isotope's A DISCARDED. That fall-through is the whole
/// isotope dependence of this class above the three light nuclides, and it is what makes
/// `G4GammaNuclearXS::GetIsoCrossSection` above 150 MeV independent of A for Z > 2.
///
/// The three light arms use `SL[0]` (the A = 2 GDR row) for all three and `SH[1]`/`SH[2]` for
/// the high-energy row, with `SP` = 1, 1, 2. The comment in the source - "same as for deuteron
/// since no A = 3 entry" - is why the triton and He3 share the deuteron's GDR table, and their
/// thresholds are the 6.2650 and 5.4994 above, so the two are not the same function.
template <typename real_t>
__host__ __device__ inline XsValue<real_t> photo_iso_xs(real_t ekin, int Z, int A) {
  const double Energy = double(ekin) / double(units::MeV<real_t>());
  if (Energy < th_min()) { return {real_t(0), XsRefusal::kNone}; }

  int gdr_row = -1, hen_row = -1, sp = 0;
  double th = 0.0;
  bool known = true;
  if (Z == 1 && A == 2) {
    th = double(threshold_energy<double>(1, 1, known));  gdr_row = 0; hen_row = 1; sp = 1;
  } else if (Z == 1 && A == 3) {
    th = double(threshold_energy<double>(1, 2, known));  gdr_row = 0; hen_row = 2; sp = 1;
  } else if (Z == 2 && A == 3) {
    th = double(threshold_energy<double>(2, 1, known));  gdr_row = 0; hen_row = 2; sp = 2;
  } else {
    return photo_element_xs<real_t>(ekin, Z);
  }
  if (!known) { return {real_t(0), XsRefusal::kNuclearMassNotTabulated}; }

  double sigma = 0.0;
  if (Energy < th) {
    sigma = 0.0;
  } else if (Energy < e_min()) {
    const GdrRowEntry e{gdr_row};
    sigma = equ_linear_fit(Energy, data::kChipsGdrN, th_min(), d_e(), e);
  } else if (Energy < e_max()) {
    const HenRowEntry e{hen_row};
    const double lE = std::log(Energy);
    sigma = equ_linear_fit(lE, data::kChipsHenN, mil_e(), dl_e(), e);
  } else {
    sigma = uhe(double(sp), std::log(Energy));
  }
  // `if(sigma < 0.) sigma = 0.;` - the same clamp, spelled differently from the element arm's
  // `std::max`, and reached by a different path.
  if (sigma < 0.0) { sigma = 0.0; }
  return {static_cast<real_t>(sigma) * millibarn<real_t>(), XsRefusal::kNone};
}

}  // namespace chips

}  // namespace g4gpu::hadronic::xs

#endif
