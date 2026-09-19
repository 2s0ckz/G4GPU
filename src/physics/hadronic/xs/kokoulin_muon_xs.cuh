// G4KokoulinMuonNuclearXS - the muon-nuclear cross section QBBC gives `muonNuclear`, and the
// double-differential form G4MuonVDNuclearModel builds its sampling table out of.
//
// Transcribed from Geant4 11.1.1, source/processes/hadronic/cross_sections/src/
// G4KokoulinMuonNuclearXS.cc:
//   ComputeDDMicroscopicCrossSection - R.P. Kokoulin's (18/01/98) approximation of the Borog
//     and Petrukhin double-differential cross section, with the `aeff` shadowing
//   ComputeMicroscopicCrossSection - an eight-point Gauss-Legendre integration of it over the
//     energy loss, in `kkk` logarithmic sub-intervals
//   BuildCrossSectionTable / GetElementCrossSection - a G4PhysicsLogVector per element,
//     61 nodes from 1 GeV to 1 PeV, filled with the integral
//
// `emextra_xs.csv` says this is the ONLY data set on `muonNuclear` for both mu- and mu+, and
// `emextra_windows.csv` that the one model is `G4MuonVDNuclearModel` over 0 to 1 PeV.
//
// WHAT THE TABLE IS, AND WHY THE PORT BUILDS ONE TOO
//
// `GetElementCrossSection` does not evaluate the integral: it reads
// `theCrossSection[Z]->Value(T)` off a 61-node log vector. So the transport sees the TABLE and
// the integral is only ever the table's contents - the V5 lesson of docs/RISK.md, and the same
// shape as P2's G4PARTICLEXS classes. The port therefore builds the same vector with the same
// `G4PhysicsLogVector(1 GeV, 1 PeV, 60)` nodes and reads it with P2's `phys_vec_value`, rather
// than evaluating `ComputeMicroscopicCrossSection` per step and being smooth where Geant4 is
// piecewise linear.
//
// THREE THINGS IN THE SOURCE THAT ARE NOT IN THE FORMULA
//
//   * `GetElementCrossSection` clamps Z > 92 to 92 - "Switch to treat transuranic elements as
//     uranium", `isHeavyElementAllowed` being a local `const G4bool = true`. The clamp is on
//     the LOOKUP only; the table itself was built with the same clamp in
//     BuildCrossSectionTable, so Z = 93 and Z = 92 are the same row rather than a missing one.
//   * `theCrossSection[Z]` is a **null pointer for every Z not in the geometry's element
//     table**, and GetElementCrossSection dereferences it without a test. Geant4 crashes on a
//     Z it was never initialised for. This port builds every Z from 1 to 92 because it can -
//     `A` comes from `data/atomic_masses.cuh` and not from the run's materials - and
//     `emextra_kokoulin.csv` can only check the elements the dump program's materials contain.
//     The difference is recorded rather than hidden: `kokoulin_build_table` reports which Z it
//     filled, and the test asserts that every Z the oracle has is one of them.
//   * `ComputeMicroscopicCrossSection` is **private**. It is checked through the table, whose
//     node values ARE its return values, so the eight-point integration is compared exactly at
//     61 energies per element and the interpolation on top of it at 60 more.
//
// WHAT IS REFUSED, BY NAME
//
//   * Z outside 1..92 after the clamp - `kPhotoNuclearNoAtomicMass`, the same refusal the two
//     CHIPS classes use, because it is the same missing NIST mean atomic mass.
#ifndef G4GPU_HADRONIC_XS_KOKOULIN_MUON_XS_CUH
#define G4GPU_HADRONIC_XS_KOKOULIN_MUON_XS_CUH

#include <cmath>

#include "core/units.cuh"
#include "data/atomic_masses.cuh"
#include "physics/hadronic/xs/physics_vector.cuh"
#include "physics/hadronic/xs/refusal.cuh"

namespace g4gpu::hadronic::xs {

namespace kokoulin {

/// MAXZMUN - the size of `theCrossSection[]`; Z = 92 is the last row that is ever filled.
inline constexpr int kMaxZ = 93;
/// LowestKineticEnergy, HighestKineticEnergy and TotBin, from the constructor.
template <typename real_t> __host__ __device__ constexpr real_t e_lowest() {
  return real_t(1) * units::GeV<real_t>();
}
template <typename real_t> __host__ __device__ constexpr real_t e_highest() {
  return real_t(1e9) * units::MeV<real_t>();  // 1 PeV
}
inline constexpr int kTotBin = 60;
inline constexpr int kNodes = kTotBin + 1;
/// CutFixed = 0.2 GeV - the lower limit of the energy-loss integral and the threshold below
/// which the double-differential form is identically zero. The SAME number appears in
/// G4MuonVDNuclearModel as its own `CutFixed`, and the two are separate members that happen to
/// be equal; both are written out where they are used.
template <typename real_t> __host__ __device__ constexpr real_t cut_fixed() {
  return real_t(0.2) * units::GeV<real_t>();
}

/// `alam2 = 0.400 GeV^2`, `alam = 0.632456 GeV`, `coeffn = alpha/pi`.
template <typename real_t> __host__ __device__ constexpr real_t alam2() {
  return real_t(0.400) * units::GeV<real_t>() * units::GeV<real_t>();
}
template <typename real_t> __host__ __device__ constexpr real_t alam() {
  return real_t(0.632456) * units::GeV<real_t>();
}

/// G4KokoulinMuonNuclearXS::ComputeDDMicroscopicCrossSection(T, Z, A, epsilon).
///
/// `Z` is declared and unnamed in the signature - the cross section depends on A alone through
/// the shadowing `aeff = 0.22*A + 0.78*A^0.89`, and on nothing else about the nucleus. `A` is
/// passed in Geant4's `g/mole`, which is numerically the amu value, and the formula uses it as
/// a pure number: `G4Log(A)` of a dimensioned quantity, which only works because g/mole is 1 in
/// the internal unit system. Passed here as the plain amu number for that reason.
///
/// `mass` is the MU-MINUS mass for both charges: the source reads
/// `G4MuonMinus::MuonMinus()->GetPDGMass()` unconditionally, so a mu+ gets the mu- mass. They
/// are equal in the PDG table, and the line is transcribed as written rather than generalised.
template <typename real_t>
__host__ __device__ inline real_t dd_microscopic_xs(real_t kin_energy, real_t A_amu,
                                                    real_t epsilon, real_t muon_mass) {
  const real_t total_energy = kin_energy + muon_mass;
  if (epsilon >= total_energy - real_t(0.5) * units::proton_mass_c2<real_t>()
      || epsilon <= cut_fixed<real_t>()) {
    return real_t(0);
  }
  const real_t ep = epsilon / units::GeV<real_t>();
  // The shadowing exponent is written `G4Exp(0.89*G4Log(A))` and not `pow(A, 0.89)`; both are
  // the same value to the last bit on this platform but the expression is kept as the source
  // has it, for the reason docs/HADRONIC_PLAN.md section 8 gives about G4Pow.
  const real_t aeff = real_t(0.22) * A_amu + real_t(0.78) * exp(real_t(0.89) * log(A_amu));
  // `*microbarn`, which is 1e-6 barn. It was 1e-3 for one build and the element cross section
  // came out a factor of 999 high at every node - the oracle is what said so.
  const real_t sigph = (real_t(49.2) + real_t(11.1) * log(ep) + real_t(151.8) / sqrt(ep))
                       * real_t(1e-6) * units::barn<real_t>();  // microbarn

  const real_t v = epsilon / total_energy;
  const real_t v1 = real_t(1) - v;
  const real_t v2 = v * v;
  const real_t mass2 = muon_mass * muon_mass;

  const real_t up = total_energy * total_energy * v1 / mass2
                    * (real_t(1) + mass2 * v2 / (alam2<real_t>() * v1));
  const real_t down = real_t(1)
                      + epsilon / alam<real_t>()
                            * (real_t(1)
                               + alam<real_t>() / (real_t(2) * units::proton_mass_c2<real_t>())
                               + epsilon / alam<real_t>());

  const real_t coeffn = units::fine_structure_const<real_t>() / units::pi<real_t>();
  real_t d = coeffn * aeff * sigph / epsilon
             * (-v1
                + (v1 + real_t(0.5) * v2 * (real_t(1) + 2 * mass2 / alam2<real_t>()))
                      * log(up / down));
  if (d < real_t(0)) { d = real_t(0); }
  return d;
}

/// G4KokoulinMuonNuclearXS::ComputeMicroscopicCrossSection(T, A) - the eight-point
/// Gauss-Legendre integration of the above over ln(epsilon), in `kkk` sub-intervals.
///
/// `ak1 = 6.9` and `ak2 = 1.0` set the number of sub-intervals: `kkk = max(1, int((bbb-aaa)/ak1
/// + ak2))`, so one sub-interval per 6.9 natural logarithms of energy-loss range, which is one
/// for everything below about 200 GeV and two above.
template <typename real_t>
__host__ __device__ inline real_t microscopic_xs(real_t kin_energy, real_t A_amu,
                                                 real_t muon_mass) {
  const real_t xgi[8] = {real_t(0.0199), real_t(0.1017), real_t(0.2372), real_t(0.4083),
                         real_t(0.5917), real_t(0.7628), real_t(0.8983), real_t(0.9801)};
  const real_t wgi[8] = {real_t(0.0506), real_t(0.1112), real_t(0.1569), real_t(0.1813),
                         real_t(0.1813), real_t(0.1569), real_t(0.1112), real_t(0.0506)};
  const real_t ak1 = real_t(6.9);
  const real_t ak2 = real_t(1.0);

  real_t cross_section = real_t(0);
  if (kin_energy <= cut_fixed<real_t>()) { return cross_section; }

  const real_t epmin = cut_fixed<real_t>();
  const real_t epmax =
      kin_energy + muon_mass - real_t(0.5) * units::proton_mass_c2<real_t>();
  if (epmax <= epmin) { return cross_section; }  // "NaN bug correction"

  const real_t aaa = log(epmin);
  const real_t bbb = log(epmax);
  int kkk = static_cast<int>((bbb - aaa) / ak1 + ak2);
  if (kkk < 1) { kkk = 1; }
  const real_t hhh = (bbb - aaa) / static_cast<real_t>(kkk);

  for (int l = 0; l < kkk; ++l) {
    const real_t x = aaa + hhh * static_cast<real_t>(l);
    for (int ll = 0; ll < 8; ++ll) {
      const real_t epln = x + xgi[ll] * hhh;
      const real_t ep = exp(epln);
      cross_section += ep * wgi[ll] * dd_microscopic_xs(kin_energy, A_amu, ep, muon_mass);
    }
  }
  cross_section *= hhh;
  if (cross_section < real_t(0)) { cross_section = real_t(0); }
  return cross_section;
}

/// The per-element table `BuildCrossSectionTable` fills: one shared energy grid and one value
/// row per Z. `filled[Z]` says whether the row exists, so that a Z this port could not build
/// is a refusal and not a zero.
template <typename real_t>
struct KokoulinTable {
  real_t energy[kNodes] = {real_t(0)};
  real_t value[kMaxZ][kNodes] = {{real_t(0)}};
  bool filled[kMaxZ] = {false};
};

/// Host-side build: the 61 nodes of `G4PhysicsLogVector(1 GeV, 1 PeV, 60)` and
/// `ComputeMicroscopicCrossSection` at each.
///
/// `G4PhysicsLogVector`'s nodes are `binVector[i] = emin * exp(i*ln(emax/emin)/nbin)` in
/// 11.1.1 - built by `G4PhysicsVector::PrepareVectors` as `edgeMin * G4Exp(i * dbin)` with
/// `dbin = log(emax/emin)/nbin` - and NOT `pow(emax/emin, i/nbin)`; the two differ in the last
/// bits and the difference moves a node, which moves every interpolated value near it.
template <typename real_t>
__host__ inline void build_table(KokoulinTable<real_t>& t, real_t muon_mass) {
  const real_t emin = e_lowest<real_t>();
  const real_t emax = e_highest<real_t>();
  const real_t dbin = log(emax / emin) / static_cast<real_t>(kTotBin);
  for (int i = 0; i < kNodes; ++i) {
    t.energy[i] = emin * exp(static_cast<real_t>(i) * dbin);
  }
  for (int Z = 1; Z < kMaxZ; ++Z) {
    if (Z > 98) { continue; }
    const real_t A = static_cast<real_t>(data::nist_atomic_mass_table()[Z]);
    if (!(A > real_t(0))) { continue; }
    for (int i = 0; i < kNodes; ++i) {
      t.value[Z][i] = microscopic_xs<real_t>(t.energy[i], A, muon_mass);
    }
    t.filled[Z] = true;
  }
}

/// A PhysVec view of one Z's row, so P2's `phys_vec_value` does the lookup.
template <typename real_t>
__host__ __device__ inline PhysVec<real_t> view(const KokoulinTable<real_t>& t, int Z) {
  PhysVec<real_t> pv;
  if (Z < 1 || Z >= kMaxZ || !t.filled[Z]) { return pv; }
  pv.e = t.energy;
  pv.v = t.value[Z];
  pv.n = kNodes;
  pv.type = kLogVector;
  pv.edge_min = t.energy[0];
  pv.edge_max = t.energy[kNodes - 1];
  const real_t idxmax_plus1 = static_cast<real_t>(kNodes - 1);
  pv.inv_dbin = idxmax_plus1 / log(pv.edge_max / pv.edge_min);
  pv.log_emin = log(pv.edge_min);
  return pv;
}

/// G4KokoulinMuonNuclearXS::GetElementCrossSection - the Z > 92 clamp and the table lookup.
template <typename real_t>
__host__ __device__ inline XsValue<real_t> element_xs(const KokoulinTable<real_t>& t,
                                                      real_t kin_energy, int ZZ) {
  // "AR-24Apr2018 Switch to treat transuranic elements as uranium"
  const int Z = (ZZ > 92) ? 92 : ZZ;
  const PhysVec<real_t> pv = view(t, Z);
  if (pv.empty()) { return {real_t(0), XsRefusal::kPhotoNuclearNoAtomicMass}; }
  return {phys_vec_value(pv, kin_energy), XsRefusal::kNone};
}

}  // namespace kokoulin

}  // namespace g4gpu::hadronic::xs

#endif
