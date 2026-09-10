// G4CrossSectionDataStore::ComputeCrossSection and SampleZandA - how a material's macroscopic
// cross section is assembled and how the target element and isotope are drawn from it.
//
// Transcribed from source/processes/hadronic/cross_sections/src/G4CrossSectionDataStore.cc
// (11.1.1): ComputeCrossSection, GetCrossSection(dp, elm, mat), GetIsoCrossSection,
// SampleZandA - and the five SelectIsotope implementations the data sets provide, which are
// here because SampleZandA is their only caller.
//
// THE TWO HALVES ARE ONE OBJECT AND THAT IS THE POINT
//
// SampleZandA does not recompute anything. It reads `matCrossSection` and `xsecelm[]`, the
// running partial sums ComputeCrossSection left behind on the last call:
//
//     matCrossSection += max(natom_i * GetCrossSection(dp, element_i, mat), 0)
//     xsecelm[i] = matCrossSection                       // cumulative, not per-element
//
// and then `cross = matCrossSection*rand(); first i with cross <= xsecelm[i]`. So the element
// is drawn from *macroscopic partial cross sections* - atom density times microscopic cross
// section - and the draw is only valid at the energy and in the material ComputeCrossSection
// was last called with. G4HadronicProcess::PostStepDoIt relies on that: it calls
// ComputeCrossSection for the integral-method check and then SampleZandA, in that order. A
// port that drew an element without having filled the partial sums at the same energy would
// sample the previous step's material, and every individual cross section would still be
// right.
//
// That is why the state lives in one struct here (`MaterialXs`) rather than in two functions.
//
// WHICH CROSS SECTION AN ELEMENT GETS
//
// GetCrossSection(dp, elm, mat) takes the LAST registered data set (`i = nDataSetList-1`) and
// asks it two questions in this order:
//
//   1. `elm->GetNaturalAbundanceFlag() && dataSetList[i]->IsElementApplicable(dp, Z, mat)`
//      -> the element cross section, and no isotope is touched.
//   2. otherwise, sum over isotopes of `abundance_j * GetIsoCrossSection(...)`.
//
// For every data set this package has, IsElementApplicable returns true (G4ParticleInelasticXS,
// the three neutron classes and G4GammaNuclearXS all `return true`; the four BGG classes too),
// and NIST elements carry the natural-abundance flag - so branch 1 is what QBBC takes and
// branch 2 is reached only for a user-built element with custom abundances. Both are here;
// branch 2 needs the per-isotope abundances, which data::Material does not carry, so it takes
// them as an argument.
//
// ISOTOPE SELECTION, PER DATA SET
//
// SelectIsotope is not shared. The five differ in exactly one thing - when they fall back to
// sampling by abundance alone:
//
//   G4ParticleInelasticXS   amax[Z] == amin[Z] || Z >= MAXZINELP
//   G4NeutronInelasticXS    amax[Z] == amin[Z] || Z >= MAXZINEL
//   G4NeutronCaptureXS      amax[Z] == amin[Z] || Z >= MAXZCAPTURE
//   G4NeutronElasticXS      always - it has no isotope data
//   G4GammaNuclearXS        amax[Z] == amin[Z] || kinEnergy > 150 MeV || Z >= MAXZGAMMAXS
//
// and otherwise all four of the non-elastic ones do the same thing: cumulate
// `abundance_j * IsoCrossSection(E, Z, A_j)`, multiply the total by a uniform draw, and take
// the first j whose partial sum reaches it. Note the comparison is `temp[j] >= sum` on the
// cumulative array and `q <= sum` on the abundance-only path - two different directions of
// the same test, both transcribed as written.
//
// The random number is drawn ONCE, before the loop, in every one of them. That matters for
// reproducing a stream: `q = G4UniformRand()` comes before the cross sections are summed, so
// a port that drew it after would consume the same numbers in a different order.
#pragma once
#include <cmath>

#include "data/isotope_list.hh"
#include "data/materials.cuh"
#include "physics/hadronic/xs/particlexs.cuh"
#include "physics/hadronic/xs/refusal.cuh"

namespace g4gpu::hadronic::xs {

/// One element's isotope composition, as G4Element carries it. `n` may be 1.
///
/// data::Material has no isotope list - it carries Z and an atom density per element, which is
/// all the EM port needed - so the caller supplies this. For a NIST element it is
/// G4NistManager's natural abundances; `natural_abundance` is G4Element's own flag, and it
/// decides which of the two branches of GetCrossSection is taken.
template <typename real_t>
struct ElementIsotopes {
  int n = 0;
  const int* a = nullptr;              ///< mass numbers
  const real_t* abundance = nullptr;   ///< relative abundance, summing to 1
  bool natural_abundance = true;
};

/// The running state ComputeCrossSection leaves for SampleZandA: the material's macroscopic
/// cross section and the cumulative partial sums per element.
template <typename real_t>
struct MaterialXs {
  static constexpr int kMax = data::kMaxElements;
  real_t total = 0;               ///< matCrossSection, 1/mm
  /// xsecelm[], cumulative and in element order. Zero-initialised: `total` was and this was
  /// not, and the two are read together. Only [0, n_elements) is ever meaningful, but a struct
  /// that is memcpy'd to a device should not carry uninitialised bytes.
  real_t cumulative[kMax] = {};
  int n_elements = 0;
};

/// The result of a target draw: which element of the material, and which isotope of it.
struct TargetZA {
  int element_index = 0;
  int z = 0;
  int a = 0;
};

/// A DATA SET, AS THE STORE ACTUALLY USES ONE - THREE FUNCTIONS AND NOTHING ELSE
///
/// G4CrossSectionDataStore holds `G4VCrossSectionDataSet*` and calls three things on it:
/// GetElementCrossSection, GetIsoCrossSection and (through SampleZandA) SelectIsotope. So the
/// functions below are templated on a struct providing exactly those three, rather than on
/// PxsDataSet - which is what docs/HADRONIC_PLAN.md section 5 asks for ("a device-callable
/// function taking the material and the per-element/isotope functions") and what lets the same
/// code serve the four BGG classes, which are not G4PARTICLEXS data sets at all.
///
/// The contract, all three `__host__ __device__` and const:
///
///     XsValue<real_t> element(int Z) const;             // GetElementCrossSection
///     XsValue<real_t> isotope(int Z, int A) const;      // GetIsoCrossSection
///     bool abundance_only(int Z) const;                 // SelectIsotope's fallback test
///
/// Energy is bound into the functor rather than passed, because every real data set needs both
/// `ekin` and `G4Log(ekin)` and threading two energies through four call sites is how a port
/// ends up evaluating a cross section at the wrong one. A template and not a virtual: a vtable
/// pointer built on the host is not dereferenceable on the device.
///
/// pxs_xs_functions() below is the adapter for a PxsDataSet.
template <typename real_t>
struct PxsXsFunctions {
  const PxsDataSet<real_t>* ds = nullptr;
  real_t ekin = 0;
  real_t loge = 0;

  __host__ __device__ XsValue<real_t> element(int Z) const {
    return pxs_element_xs<real_t>(*ds, ekin, loge, Z);
  }
  __host__ __device__ XsValue<real_t> isotope(int Z, int A) const {
    return pxs_iso_xs<real_t>(*ds, ekin, loge, Z, A);
  }
  __host__ __device__ bool abundance_only(int Z) const;  // defined below the predicate
};

/// G4CrossSectionDataStore::GetCrossSection(dp, elm, mat) - the microscopic cross section of
/// one element, mm^2, from any data set satisfying the three-function contract above.
///
/// @param iso  the element's isotopes. When `natural_abundance` is false the isotope sum is
///             taken instead of the element cross section, which is branch 2 of the comment
///             at the top of this file.
template <typename real_t, typename XsFn>
__host__ __device__ inline XsValue<real_t> store_element_xs_fn(
    const XsFn& xs, int Z, const ElementIsotopes<real_t>& iso) {
  if (iso.natural_abundance || iso.n <= 0) { return xs.element(Z); }
  real_t sigma = real_t(0.0);
  XsRefusal ref = XsRefusal::kNone;
  for (int j = 0; j < iso.n; ++j) {
    const XsValue<real_t> x = xs.isotope(Z, iso.a[j]);
    if (!x.ok()) { ref = x.refused; }
    sigma += iso.abundance[j] * x.value;
  }
  return {sigma, ref};
}

/// G4CrossSectionDataStore::ComputeCrossSection for any such data set.
template <typename real_t, typename XsFn>
__host__ __device__ inline XsValue<real_t> store_compute_cross_section_fn(
    const XsFn& xs, const data::Material<real_t>& mat, const ElementIsotopes<real_t>* isos,
    MaterialXs<real_t>& out) {
  out.total = real_t(0.0);
  out.n_elements = mat.n_elements;
  XsRefusal ref = XsRefusal::kNone;
  for (int i = 0; i < mat.n_elements; ++i) {
    const int z = static_cast<int>(mat.z[i]);
    const XsValue<real_t> e = store_element_xs_fn<real_t>(xs, z, isos[i]);
    if (!e.ok()) { ref = e.refused; }
    const real_t x = mat.n_atoms[i] * e.value;
    out.total += (x > real_t(0.0)) ? x : real_t(0.0);
    out.cumulative[i] = out.total;
  }
  return {out.total, ref};
}

/// G4CrossSectionDataStore::SampleZandA for any such data set. See the PxsDataSet overload
/// below for the parameter meanings; this is that function with the data set abstracted out.
template <typename real_t, typename XsFn>
__host__ __device__ inline TargetZA store_sample_za_fn(const XsFn& xs,
                                                       const data::Material<real_t>& mat,
                                                       const ElementIsotopes<real_t>* isos,
                                                       const MaterialXs<real_t>& mxs,
                                                       real_t q_elm, real_t q_iso) {
  TargetZA t;
  t.element_index = 0;
  if (mat.n_elements > 1) {
    const real_t cross = mxs.total * q_elm;
    for (int i = 0; i < mat.n_elements; ++i) {
      if (cross <= mxs.cumulative[i]) {
        t.element_index = i;
        break;
      }
    }
  }
  const int Z = static_cast<int>(mat.z[t.element_index]);
  const ElementIsotopes<real_t>& iso = isos[t.element_index];
  t.z = Z;
  t.a = (iso.n > 0) ? iso.a[0] : 0;
  if (iso.n <= 1) { return t; }

  if (xs.abundance_only(Z)) {
    real_t sum = real_t(0.0);
    for (int j = 0; j < iso.n; ++j) {
      sum += iso.abundance[j];
      if (q_iso <= sum) {
        t.a = iso.a[j];
        break;
      }
    }
    return t;
  }
  real_t temp[64];
  const int niso = (iso.n < 64) ? iso.n : 64;
  real_t sum = real_t(0.0);
  for (int j = 0; j < niso; ++j) {
    sum += iso.abundance[j] * xs.isotope(Z, iso.a[j]).value;
    temp[j] = sum;
  }
  sum *= q_iso;
  for (int j = 0; j < niso; ++j) {
    if (temp[j] >= sum) {
      t.a = iso.a[j];
      break;
    }
  }
  return t;
}


/// True when the data set samples an isotope by abundance alone - the per-class fallback test
/// listed in the file header.
template <typename real_t>
__host__ __device__ inline bool store_abundance_only(const PxsDataSet<real_t>& ds, int Z,
                                                     real_t ekin) {
  if (ds.kind == PxsKind::kNeutronElastic) { return true; }
  if (Z >= pxs_maxz(ds.kind)) { return true; }
  if (Z > data::kIsotopeListMaxZ) { return true; }
  if (data::isotope_amax()[Z] == data::isotope_amin()[Z]) { return true; }
  if (ds.kind == PxsKind::kGammaNuclear && ekin > pxs_gamma_transition<real_t>()) {
    return true;
  }
  return false;
}

template <typename real_t>
__host__ __device__ inline bool PxsXsFunctions<real_t>::abundance_only(int Z) const {
  return store_abundance_only<real_t>(*ds, Z, ekin);
}

/// The adapter: a PxsDataSet at one energy, as the three functions the store needs.
template <typename real_t>
__host__ __device__ inline PxsXsFunctions<real_t> pxs_xs_functions(const PxsDataSet<real_t>& ds,
                                                                   real_t ekin, real_t loge) {
  return {&ds, ekin, loge};
}

/// G4CrossSectionDataStore::ComputeCrossSection and SampleZandA for a G4PARTICLEXS data set
/// at one energy. Both are wrappers over the functor forms above, so there is ONE
/// implementation of the accumulation and of the two draws, and every test that touches either
/// covers it. The parameters are the functor forms', documented above.
///
/// The caller must have run store_compute_cross_section at this energy and material before
/// store_sample_za: `mxs` is where the element partial sums come from, and nothing here
/// recomputes them.
template <typename real_t>
__host__ __device__ inline XsValue<real_t> store_compute_cross_section(
    const PxsDataSet<real_t>& ds, real_t ekin, real_t loge,
    const data::Material<real_t>& mat, const ElementIsotopes<real_t>* isos,
    MaterialXs<real_t>& out) {
  return store_compute_cross_section_fn<real_t>(pxs_xs_functions<real_t>(ds, ekin, loge), mat,
                                               isos, out);
}

template <typename real_t>
__host__ __device__ inline TargetZA store_sample_za(const PxsDataSet<real_t>& ds, real_t ekin,
                                                    real_t loge,
                                                    const data::Material<real_t>& mat,
                                                    const ElementIsotopes<real_t>* isos,
                                                    const MaterialXs<real_t>& mxs,
                                                    real_t q_elm, real_t q_iso) {
  return store_sample_za_fn<real_t>(pxs_xs_functions<real_t>(ds, ekin, loge), mat, isos, mxs,
                                    q_elm, q_iso);
}

}  // namespace g4gpu::hadronic::xs
