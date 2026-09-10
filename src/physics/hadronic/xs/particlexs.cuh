// The five G4PARTICLEXS-based cross-section data sets, element-wise and isotope-wise, with
// each one's high-energy hand-over.
//
// Transcribed from (11.1.1, source/processes/hadronic/cross_sections/src/):
//   G4ParticleInelasticXS.cc   p, d, t, He3, alpha inelastic - QBBC's proton inelastic
//                              (G4HadronInelasticQBBC) and its ion inelastic (G4IonPhysicsXS)
//   G4NeutronInelasticXS.cc    neutron inelastic
//   G4NeutronElasticXS.cc      neutron elastic
//   G4NeutronCaptureXS.cc      neutron capture
//   G4GammaNuclearXS.cc        gamma-nuclear (P13's cross section, delivered here)
//
// They look like one class written five times and they are not. Every one of the five differs
// from the others in at least one of: the hand-over model above the table, the energy below
// which isotope data are used, what happens below the table's first node, and how an isotope
// is chosen. The differences are the physics, so they are listed here rather than factored
// away:
//
//   class                 above the table            isotope data used when   below the table
//   G4ParticleInelasticXS coeff[Z][idx] * GG         E <= 20 MeV              flat at v[0]
//                         (GGHadronNucleus for p,
//                          GGNuclNucl for d/t/He3/a)
//   G4NeutronInelasticXS  coeff[Z] * GGHadronNucleus E <= 20 MeV              flat at v[0]
//   G4NeutronElasticXS    coeff[Z] * GGHadronNucleus never (IsIsoApplicable   flat at v[0]
//                                                     returns false)
//   G4NeutronCaptureXS    element: zero at and above E <= 20 MeV, if the      v[1]*sqrt(E1/E)
//                         20 MeV; isotope: zero      file exists              below node 1,
//                         ABOVE 20 MeV only                                   E floored at
//                                                                             1e-10 eV
//   G4GammaNuclearXS      CHIPS above 150 MeV, and   E < 150 MeV              flat at v[0]
//                         a straight line from the
//                         table's top to 150 MeV
//
// The capture row is two rows because the same ceiling is spelled with two different operators
// twenty lines apart: `if(ekin < emax)` guards GetElementCrossSection and `if(eKin > emax)
// { return xs; }` guards IsoCrossSection. At exactly 20 MeV the element cross section is zero
// and the isotope one is evaluated. Both are transcribed as written; had_particlexs_iso.csv
// includes E = 20 MeV exactly, so the asymmetry is compared and not just described.
//
// `coeff[Z]` is the continuity factor, computed at initialisation as the table's *last value*
// divided by the hand-over model's value *at the table's last energy*. It is not 1, and not
// close to 1 for every Z; it is what makes the cross section continuous at the table's top.
// A port that evaluated Glauber-Gribov directly above the table would be smooth, wrong by
// per cents, and would look like a Glauber-Gribov transcription error.
//
// ISOTOPE SELECTION IS FOUR DIFFERENT FUNCTIONS, AND IT IS IN sample_za.cuh
//
// SelectIsotope is per class:
//   G4ParticleInelasticXS / G4NeutronInelasticXS / G4NeutronCaptureXS: abundance times
//     IsoCrossSection, cumulated, then a uniform draw - unless amax[Z] == amin[Z] or Z is at
//     the class's ceiling, in which case abundance alone.
//   G4NeutronElasticXS: abundance alone, always. It has no isotope data.
//   G4GammaNuclearXS: abundance times IsoCrossSection, but abundance alone above 150 MeV as
//     well as in the other two cases.
// They live with G4CrossSectionDataStore::SampleZandA, which is the only thing that calls
// them.
//
// WHAT IS REFUSED
//
// G4GammaNuclearXS is PARTIAL. Below the data files' top energy - 130 MeV for most elements -
// it is exact, element and isotope. Above it Geant4 needs G4PhotoNuclearCrossSection, the
// 1821-line CHIPS parameterisation, which is not ported: the transition region between the
// table's top and 150 MeV is a straight line to `xs150[Z]`, which is CHIPS at 150 MeV, so even
// the transition needs it. Refused by name (XsRefusal::kPhotoNuclearCrossSection) at the point
// it would have been needed - which is what makes the low-energy branch usable and the gap
// visible.
//
// HYDROGEN IS AN ISOTOPE-PATH SPECIAL CASE ONLY
//
// GetElementCrossSection has no `Z == 1` branch at all: for `ekin <= emax` it returns
// `pv->Value(ekin)` for hydrogen like any other Z, and `gamma/inel1` exists - it declares
// `0 130 2` with both values zero, so Geant4 returns a tabulated exact zero and never reaches
// CHIPS. The special case is in GetIsoCrossSection, which has `ekin <= emax && Z != 1` on the
// element-scaled path and sends `Z == 1` to CHIPS with everything above 150 MeV. An earlier
// version of this comment said "and for hydrogen at any energy", which was wrong for the
// element path; the code was right on both.
//
// G4GammaNuclearXS DOES have a coeff array, and it is not the one the other classes have.
// `static G4double coeff[3][3]` is indexed [Z][A - amin[Z]] - by ISOTOPE, not by particle -
// filled only for Z <= 2 as CHIPS-iso(10 GeV) / CHIPS-element(10 GeV), and read only in the
// `Z <= 2 && ekin > 10 GeV` isotope branch. That branch needs CHIPS on both sides of the
// division, so it is inside the refusal and the array is not transcribed. Named here because
// "G4GammaNuclearXS has xs150 instead of a coeff" is not the whole truth.
#pragma once
#include <cmath>
#include <string>

#include "data/isotope_list.hh"
#include "data/particlexs_data.cuh"
#include "physics/hadronic/xs/gg_hadron_nucleus_xsc.cuh"
#include "physics/hadronic/xs/gg_nucl_nucl_xsc.cuh"
#include "physics/hadronic/xs/physics_vector.cuh"
#include "physics/hadronic/xs/projectile.cuh"
#include "physics/hadronic/xs/refusal.cuh"

namespace g4gpu::hadronic::xs {

/// Which of the five data sets, and so which hand-over and which ceiling.
enum class PxsKind {
  kParticleInelastic,  ///< G4ParticleInelasticXS - p, d, t, He3, alpha
  kNeutronInelastic,   ///< G4NeutronInelasticXS
  kNeutronElastic,     ///< G4NeutronElasticXS
  kNeutronCapture,     ///< G4NeutronCaptureXS
  kGammaNuclear        ///< G4GammaNuclearXS
};

/// MAXZINELP / MAXZINEL / MAXZEL / MAXZCAPTURE = 93, MAXZGAMMAXS = 95.
__host__ __device__ inline int pxs_maxz(PxsKind k) {
  return (k == PxsKind::kGammaNuclear) ? 95 : 93;
}

/// `elimit` - the energy below which isotope data are used. 20 MeV for the two inelastic
/// classes; G4GammaNuclearXS uses rTransitionBound = 150 MeV. G4NeutronCaptureXS has no such
/// limit (20 MeV is already its ceiling) and G4NeutronElasticXS has no isotope data.
template <typename real_t> __host__ __device__ inline real_t pxs_iso_limit(PxsKind k) {
  if (k == PxsKind::kGammaNuclear) { return real_t(150.) * units::MeV<real_t>(); }
  return real_t(20.) * units::MeV<real_t>();
}

/// G4NeutronCaptureXS's `emax` - it returns exactly zero at and above this.
template <typename real_t> __host__ __device__ constexpr real_t pxs_capture_emax() {
  return real_t(20.) * units::MeV<real_t>();
}
/// G4NeutronCaptureXS's `elimit`, the energy floor: 1e-10 eV. `logElimit` is its logarithm,
/// taken once in the constructor.
template <typename real_t> __host__ __device__ constexpr real_t pxs_capture_elimit() {
  return real_t(1.0e-10) * units::eV<real_t>();
}
/// G4GammaNuclearXS::rTransitionBound. Declared `static const G4int` in the header, so
/// `150.*CLHEP::MeV` is truncated to the integer 150 - the same number, and worth noticing
/// before assuming it is a G4double.
template <typename real_t> __host__ __device__ constexpr real_t pxs_gamma_transition() {
  return real_t(150.) * units::MeV<real_t>();
}

/// One dataset, ready to evaluate: the vectors plus the per-Z continuity factor.
///
/// `data` is a pointer so the struct stays copyable to the device with the table living
/// wherever the host put it; on the host it points at a data::ParticleXsTable that must
/// outlive the dataset.
template <typename real_t>
struct PxsDataSet {
  PxsKind kind = PxsKind::kNeutronInelastic;
  Projectile<real_t> particle{};
  const data::ParticleXsTable<real_t>* data = nullptr;
  real_t coeff[data::kPxsMaxZ] = {real_t(0)};
};

/// A PhysVec view of one slice of the flat table. No logarithms: the slice already carries
/// the Initialise() results, so this is device-callable.
template <typename real_t>
__host__ __device__ inline PhysVec<real_t> pxs_view(const data::ParticleXsTable<real_t>& t,
                                                    const data::PxsSlice<real_t>& s) {
  PhysVec<real_t> pv;
  if (s.n < 2 || t.e_data == nullptr) { return pv; }
  pv.e = t.e_data + s.off;
  pv.v = t.v_data + s.off;
  pv.n = s.n;
  pv.type = s.type;
  pv.edge_min = s.edge_min;
  pv.edge_max = s.edge_max;
  pv.inv_dbin = s.inv_dbin;
  pv.log_emin = s.log_emin;
  return pv;
}

/// The hand-over model above the table, for one dataset and one Z.
///
/// G4ParticleInelasticXS's constructor picks the component by particle name: "Glauber-Gribov"
/// (G4ComponentGGHadronNucleusXsc) for a proton and "Glauber-Gribov Nucl-nucl"
/// (G4ComponentGGNuclNuclXsc) for d, t, He3 and alpha. The two neutron classes always use
/// G4ComponentGGHadronNucleusXsc, elastic or inelastic as their own name says. Both are
/// called through the `G4double A` overload with A = aeff[Z], so A is rounded with G4lrint.
template <typename real_t>
__host__ __device__ inline XsValue<real_t> pxs_high_energy(const PxsDataSet<real_t>& ds,
                                                            real_t ekin, int Z) {
  const real_t aeff = data::isotope_aeff_of<real_t>(Z);
  switch (ds.kind) {
    case PxsKind::kNeutronElastic:
      return ggh_elastic_element<real_t>(ds.particle, ekin, Z, aeff);
    case PxsKind::kNeutronInelastic:
      return ggh_inelastic_element<real_t>(ds.particle, ekin, Z, aeff);
    case PxsKind::kParticleInelastic:
      if (ds.particle.pdg == pdg::kProton) {
        return ggh_inelastic_element<real_t>(ds.particle, ekin, Z, aeff);
      }
      return ggnn_inelastic_element<real_t>(ds.particle, ekin, Z, aeff);
    case PxsKind::kNeutronCapture:
      // There is none: GetElementCrossSection returns 0 at and above emax.
      return {real_t(0), XsRefusal::kNone};
    case PxsKind::kGammaNuclear:
      return {real_t(0), XsRefusal::kPhotoNuclearCrossSection};
  }
  return {real_t(0), XsRefusal::kNone};
}

/// The 1/sqrt(E) extrapolation G4NeutronCaptureXS uses below its second node: `(*pv)[1] *
/// sqrt(E1/E)` with E1 = pv->Energy(1). Note it is node ONE, not node zero - the first node
/// of every capture file holds the value at 1e-6 MeV and this form is anchored to the second.
template <typename real_t>
__host__ __device__ inline real_t pxs_capture_value(const PhysVec<real_t>& pv, real_t e,
                                                    real_t loge) {
  const real_t e1 = pv.energy(1);
  return (e >= e1) ? phys_vec_log_value(pv, e, loge) : pv.value_at(1) * sqrt(e1 / e);
}

/// The per-element cross section: G4ParticleInelasticXS::ElementCrossSection,
/// G4NeutronInelasticXS::ElementCrossSection, G4NeutronElasticXS::ElementCrossSection,
/// G4NeutronCaptureXS::GetElementCrossSection, G4GammaNuclearXS::GetElementCrossSection.
///
/// @param loge  log(ekin), as the caller of the real class supplies it from
///              G4DynamicParticle::GetLogKineticEnergy(). Ignored by the gamma class, which
///              uses Value(e) and not LogVectorValue.
template <typename real_t>
__host__ __device__ inline XsValue<real_t> pxs_element_xs(const PxsDataSet<real_t>& ds,
                                                           real_t ekin, real_t loge, int ZZ) {
  const data::ParticleXsTable<real_t>& t = *ds.data;
  const int maxz = pxs_maxz(ds.kind);

  if (ds.kind == PxsKind::kNeutronCapture) {
    // GetElementCrossSection: zero at and above emax, then ElementCrossSection, which floors
    // the energy at elimit and extrapolates as 1/sqrt(E) below the second node.
    if (!(ekin < pxs_capture_emax<real_t>())) { return {real_t(0), XsRefusal::kNone}; }
    const int Z = (ZZ < maxz - 1) ? ZZ : maxz - 1;  // std::min(ZZ, MAXZCAPTURE-1)
    const PhysVec<real_t> pv = pxs_view<real_t>(t, t.element[Z]);
    if (pv.empty()) { return {real_t(0), XsRefusal::kMissingParticleXSData}; }
    real_t e = ekin;
    real_t le = loge;
    if (ekin < pxs_capture_elimit<real_t>()) {
      e = pxs_capture_elimit<real_t>();
      le = log(pxs_capture_elimit<real_t>());
    }
    return {pxs_capture_value<real_t>(pv, e, le), XsRefusal::kNone};
  }

  const int Z = (ZZ >= maxz) ? maxz - 1 : ZZ;
  const PhysVec<real_t> pv = pxs_view<real_t>(t, t.element[Z]);

  if (ds.kind == PxsKind::kGammaNuclear) {
    if (pv.empty()) { return {real_t(0), XsRefusal::kPhotoNuclearCrossSection}; }
    const real_t emax = pv.max_energy();
    if (ekin <= emax) { return {phys_vec_value(pv, ekin), XsRefusal::kNone}; }
    // Both remaining branches need CHIPS: above 150 MeV directly, and between emax and
    // 150 MeV through xs150[Z], which is CHIPS evaluated at 150 MeV.
    return {real_t(0), XsRefusal::kPhotoNuclearCrossSection};
  }

  if (pv.empty()) { return {real_t(0), XsRefusal::kMissingParticleXSData}; }
  if (ekin <= pv.max_energy()) {
    return {phys_vec_log_value(pv, ekin, loge), XsRefusal::kNone};
  }
  const XsValue<real_t> hi = pxs_high_energy<real_t>(ds, ekin, Z);
  return {ds.coeff[Z] * hi.value, hi.refused};
}

/// The per-isotope cross section: G4ParticleInelasticXS::IsoCrossSection,
/// G4NeutronInelasticXS::IsoCrossSection, G4NeutronElasticXS::ComputeIsoCrossSection,
/// G4NeutronCaptureXS::IsoCrossSection, G4GammaNuclearXS::GetIsoCrossSection.
///
/// The two inelastic classes share a shape: use the isotope vector when it exists and the
/// energy is at or below 20 MeV, otherwise the element cross section scaled by A/aeff[Z]. The
/// other three do not:
///   G4NeutronElasticXS has no isotope data at all - ComputeIsoCrossSection is
///     `ElementCrossSection(...)*A/aeff[Z]` with no window and no vector lookup, and it
///     divides by `aeff[Z]` at the *unclamped* Z, unlike the two inelastic classes.
///   G4NeutronCaptureXS uses the isotope vector at any energy below 20 MeV, and does NOT
///     scale by A/aeff when it falls back to the element vector.
///   G4GammaNuclearXS scales by A/aeff[Z] only on the element path, and sends hydrogen and
///     everything above 150 MeV to CHIPS.
template <typename real_t>
__host__ __device__ inline XsValue<real_t> pxs_iso_xs(const PxsDataSet<real_t>& ds,
                                                       real_t ekin, real_t loge, int ZZ,
                                                       int A) {
  const data::ParticleXsTable<real_t>& t = *ds.data;
  const int maxz = pxs_maxz(ds.kind);
  const int* amin = data::isotope_amin();
  const int* amax = data::isotope_amax();
  const real_t af = static_cast<real_t>(A);

  if (ds.kind == PxsKind::kNeutronElastic) {
    // `ElementCrossSection(...)*A/aeff[Z]` at the UNCLAMPED Z, as G4NeutronElasticXS.cc has
    // it - unlike the two inelastic classes, which divide by aeff at the clamped Z. Above
    // Z = 94 there is no aeff entry and Geant4 reads off the end of the array; refused by name
    // rather than dividing by the zero isotope_aeff_of returns, which would be an inf.
    if (ZZ > data::kIsotopeListMaxZ || ZZ < 1) {
      return {real_t(0), XsRefusal::kIsotopeListOutOfRange};
    }
    const XsValue<real_t> el = pxs_element_xs<real_t>(ds, ekin, loge, ZZ);
    return {el.value * af / data::isotope_aeff_of<real_t>(ZZ), el.refused};
  }

  if (ds.kind == PxsKind::kNeutronCapture) {
    if (ekin > pxs_capture_emax<real_t>()) { return {real_t(0), XsRefusal::kNone}; }
    const int Z = (ZZ < maxz - 1) ? ZZ : maxz - 1;
    real_t e = ekin;
    real_t le = loge;
    if (e < pxs_capture_elimit<real_t>()) {
      e = pxs_capture_elimit<real_t>();
      le = log(pxs_capture_elimit<real_t>());
    }
    const PhysVec<real_t> pv = pxs_view<real_t>(t, t.element[Z]);
    if (pv.empty()) { return {real_t(0), XsRefusal::kMissingParticleXSData}; }
    if (Z <= data::kIsotopeListMaxZ && amin[Z] < amax[Z] && A >= amin[Z] && A <= amax[Z]) {
      const PhysVec<real_t> pvi = pxs_view<real_t>(t, data::pxs_isotope<real_t>(t, Z, A));
      if (!pvi.empty()) { return {pxs_capture_value<real_t>(pvi, e, le), XsRefusal::kNone}; }
    }
    return {pxs_capture_value<real_t>(pv, e, le), XsRefusal::kNone};
  }

  if (ds.kind == PxsKind::kGammaNuclear) {
    const int Z = (ZZ >= maxz) ? maxz - 1 : ZZ;
    const PhysVec<real_t> pv = pxs_view<real_t>(t, t.element[Z]);
    if (pv.empty()) { return {real_t(0), XsRefusal::kPhotoNuclearCrossSection}; }
    const real_t emax = pv.max_energy();
    if (Z <= data::kIsotopeListMaxZ && amin[Z] < amax[Z] && A >= amin[Z] && A <= amax[Z]
        && ekin < pxs_gamma_transition<real_t>()) {
      const PhysVec<real_t> pvi = pxs_view<real_t>(t, data::pxs_isotope<real_t>(t, Z, A));
      if (!pvi.empty()) {
        const real_t emaxiso = pvi.max_energy();
        if (ekin <= emaxiso) { return {phys_vec_value(pvi, ekin), XsRefusal::kNone}; }
        // The straight line from the isotope table's top to CHIPS at 150 MeV.
        return {real_t(0), XsRefusal::kPhotoNuclearCrossSection};
      }
    }
    if (ekin <= emax && Z != 1) {
      return {phys_vec_value(pv, ekin) * af / data::isotope_aeff_of<real_t>(Z),
              XsRefusal::kNone};
    }
    return {real_t(0), XsRefusal::kPhotoNuclearCrossSection};
  }

  // G4ParticleInelasticXS / G4NeutronInelasticXS
  const int Z = (ZZ >= maxz) ? maxz - 1 : ZZ;
  const PhysVec<real_t> pv = pxs_view<real_t>(t, t.element[Z]);
  if (pv.empty()) { return {real_t(0), XsRefusal::kMissingParticleXSData}; }
  if (ekin <= pxs_iso_limit<real_t>(ds.kind) && Z <= data::kIsotopeListMaxZ
      && amin[Z] < amax[Z] && A >= amin[Z] && A <= amax[Z]) {
    const PhysVec<real_t> pvi = pxs_view<real_t>(t, data::pxs_isotope<real_t>(t, Z, A));
    if (!pvi.empty()) { return {phys_vec_log_value(pvi, ekin, loge), XsRefusal::kNone}; }
  }
  real_t xs;
  XsRefusal ref = XsRefusal::kNone;
  if (ekin <= pv.max_energy()) {
    xs = phys_vec_log_value(pv, ekin, loge);
  } else {
    const XsValue<real_t> hi = pxs_high_energy<real_t>(ds, ekin, Z);
    xs = ds.coeff[Z] * hi.value;
    ref = hi.refused;
  }
  xs *= af / data::isotope_aeff_of<real_t>(Z);
  return {xs, ref};
}

/// Initialise's "smooth transition": coeff[Z] = last table value / hand-over value at the
/// table's last energy, or 1 when the hand-over is zero.
///
/// Host-side, once per dataset. G4NeutronCaptureXS has no coeff at all; G4GammaNuclearXS's
/// coeff[3][3] is a per-isotope CHIPS ratio used only above 10 GeV for Z <= 2, inside the
/// refused region (see the file header). Both are left at 1 and the evaluation never
/// multiplies by them.
///
/// G4ParticleInelasticXS's array is `coeff[MAXZINELP][5]` and the second index is the PARTICLE
/// - 0 proton, 1 deuteron, 2 triton, 3 He3, 4 alpha, set once in its constructor - not an
/// isotope. One PxsDataSet is one particle, so `coeff[Z]` here IS `coeff[Z][index]` there.
template <typename real_t>
__host__ inline void pxs_build_coeff(PxsDataSet<real_t>& ds) {
  const data::ParticleXsTable<real_t>& t = *ds.data;
  for (int z = 0; z < data::kPxsMaxZ; ++z) { ds.coeff[z] = real_t(1.0); }
  if (ds.kind == PxsKind::kNeutronCapture || ds.kind == PxsKind::kGammaNuclear) { return; }
  for (int z = 1; z <= t.max_z; ++z) {
    const PhysVec<real_t> pv = pxs_view<real_t>(t, t.element[z]);
    if (pv.empty()) { continue; }
    const real_t sig1 = pv.value_at(pv.length() - 1);
    const real_t ehigh = pv.max_energy();
    const XsValue<real_t> sig2 = pxs_high_energy<real_t>(ds, ehigh, z);
    xs_fatal_if_refused(sig2.refused, "pxs_build_coeff");
    ds.coeff[z] = (sig2.value > real_t(0.)) ? sig1 / sig2.value : real_t(1.0);
  }
}

/// Builds one dataset from a G4PARTICLEXS particle subdirectory. Host-only.
///
/// @return false if an element file is missing, which the caller must treat as fatal.
template <typename real_t>
__host__ inline bool pxs_load(PxsKind kind, const Projectile<real_t>& particle,
                              const std::string& dir, data::ParticleXsTable<real_t>& table,
                              PxsDataSet<real_t>& ds) {
  const char* prefix = "inel";
  if (kind == PxsKind::kNeutronElastic) {
    prefix = "el";
  } else if (kind == PxsKind::kNeutronCapture) {
    prefix = "cap";
  }
  const bool is_gamma = (kind == PxsKind::kGammaNuclear);
  // G4NeutronElasticXS::IsIsoApplicable returns false and it never opens an isotope file.
  const bool with_iso = (kind != PxsKind::kNeutronElastic);
  // Files exist to Z = 92 for the nucleon and ion sets and to Z = 94 for gamma, which is
  // MAXZGAMMAXS - 1.
  const int zmax = is_gamma ? 94 : 92;
  if (!data::load_particlexs<real_t>(dir, prefix, zmax, is_gamma, with_iso, table)) {
    return false;
  }
  ds.kind = kind;
  ds.particle = particle;
  ds.data = &table;
  pxs_build_coeff<real_t>(ds);
  return true;
}

}  // namespace g4gpu::hadronic::xs
