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
//
// WHERE THE ABUNDANCES COME FROM (P8b)
//
// `ElementIsotopes` below used to have no producer at all - "the caller supplies this" - and
// that was what blocked the elastic and neutron wiring: no table in the port held abundances.
// `data/isotope_abundance.hh` is G4NistElementBuilder's table now, and `NistIsotopeView` is the
// adapter that turns a `data::Material`'s element list into the per-element isotope lists these
// functions index. It is a VIEW rather than an array because the alternative is an
// `ElementIsotopes[kMaxElements]` on the stack of every step - 16 entries of 32 bytes, 512 bytes
// on a kernel already measured at 3696 B of stack against a 3072 B limit (docs/RISK.md V22) - so
// the functions below are templated on the container and index it with `[]`, which a raw
// pointer, an array and a view all satisfy.
#pragma once
#include <cmath>

#include "data/isotope_abundance.hh"
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

/// The isotope composition of a NIST-built element, from `data/isotope_abundance.hh`.
///
/// G4NistElementBuilder::BuildElement keeps the isotopes with a non-zero abundance and
/// G4Element::AddIsotope re-normalises them; both are in the generated table, so this is a pair
/// of pointers into it rather than any arithmetic. `natural_abundance` is true because that is
/// what BuildElement's last line sets - `theElement->SetNaturalAbundanceFlag(true)` - and it is
/// the flag that decides which branch of `store_element_xs_fn` is taken.
///
/// A Z with no NIST element returns `n = 0`, which every function below treats as "no isotope
/// information": `store_element_xs_fn` falls through to the element cross section and
/// `store_sample_za_fn` returns A = 0. That is refused rather than guessed - see
/// `data::IsotopeRefusal` - because an A of 0 is not a nucleus and a caller that uses it will
/// produce a recoil with no mass.
template <typename real_t>
__host__ __device__ inline ElementIsotopes<real_t> nist_element_isotopes(int Z) {
  ElementIsotopes<real_t> e;
  const int n = data::nist_element_n_isotopes(Z);
  if (n <= 0) { return e; }
  const int off = data::nist_element_iso_offset()[Z];
  e.n = n;
  e.a = data::nist_element_iso_a() + off;
  e.abundance = data::NistIsotopeAbundance<real_t>::values() + off;
  e.natural_abundance = true;
  return e;
}

/// A material's per-element isotope lists, indexed by element index, without materialising them.
///
/// Satisfies the `isos[i]` that `store_compute_cross_section_fn` and `store_sample_za_fn` ask
/// for. `operator[]` returns by value and the callee binds it to a `const&`, which extends the
/// temporary's lifetime for the duration of the reference - so there is no array and no
/// dangling pointer, and the 512 bytes of stack an `ElementIsotopes[16]` would cost are not
/// spent. The pointers INSIDE the returned struct are into the compiled-in table, which has
/// static storage duration on the host and lives in the module's constant/global data on the
/// device, so they outlive every caller.
template <typename real_t>
struct NistIsotopeView {
  const data::Material<real_t>* mat = nullptr;
  __host__ __device__ ElementIsotopes<real_t> operator[](int i) const {
    return nist_element_isotopes<real_t>(static_cast<int>(mat->z[i] + real_t(0.5)));
  }
};

template <typename real_t>
__host__ __device__ inline NistIsotopeView<real_t> nist_isotopes_of(
    const data::Material<real_t>& mat) {
  return {&mat};
}

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
///
/// `isos` is anything indexable by element number - an `ElementIsotopes` array, a pointer to
/// one, or a `NistIsotopeView`. See the note at the top of this file for why it is not a
/// pointer any more.
template <typename real_t, typename XsFn, typename IsoArray>
__host__ __device__ inline XsValue<real_t> store_compute_cross_section_fn(
    const XsFn& xs, const data::Material<real_t>& mat, const IsoArray& isos,
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
template <typename real_t, typename XsFn, typename IsoArray>
__host__ __device__ inline TargetZA store_sample_za_fn(const XsFn& xs,
                                                       const data::Material<real_t>& mat,
                                                       const IsoArray& isos,
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
  // TWO PASSES OVER THE ISOTOPES RATHER THAN A `real_t temp[64]`, AND THE NUMBERS ARE THE SAME.
  //
  // Geant4 fills `G4double temp[nIso]` with the running sum and then walks it. That array was
  // 64 doubles here - 512 bytes of kernel stack, in a function `step_neutral` calls on every
  // neutron capture, against the 3072-byte limit that docs/RISK.md V22 already measures
  // `run_step_neutral` exceeding at 3696. The second pass re-accumulates the identical partial
  // sums, because it adds the identical terms in the identical order, so `temp[j] >= sum` and
  // the re-accumulated `acc >= sum` are the same comparison on the same doubles - not an
  // equivalent one. What it costs instead is a second evaluation of each isotope cross section,
  // which is a log-vector lookup; what it buys is that the choice of data set cannot push a
  // kernel over its stack.
  //
  // `tests/test_hadronic_process.cu` section 4 compares this against Geant4's counted
  // frequencies AND against the analytic weights, and its numbers did not move when the array
  // went - which is the check that the two passes agree, measured rather than argued.
  real_t sum = real_t(0.0);
  for (int j = 0; j < iso.n; ++j) { sum += iso.abundance[j] * xs.isotope(Z, iso.a[j]).value; }
  sum *= q_iso;
  real_t acc = real_t(0.0);
  for (int j = 0; j < iso.n; ++j) {
    acc += iso.abundance[j] * xs.isotope(Z, iso.a[j]).value;
    if (acc >= sum) {
      t.a = iso.a[j];
      break;
    }
  }
  return t;
}


/// `store_sample_za_fn` with the two uniforms drawn where Geant4 draws them - which is the
/// number of them as well as the order (P8b).
///
/// THE TWO-ARGUMENT FORM CANNOT EXPRESS "NO DRAW", AND SampleZandA HAS TWO OF THOSE.
///
///     if (mat->GetNumberOfElements() > 1) { ... G4UniformRand() ... }      // the element
///     std::size_t nIso = anElement->GetNumberOfIsotopes();
///     iso = anElement->GetIsotope(0);
///     if (1 < nIso) { ... }                                               // the isotope
///
/// A single-element material consumes NO element uniform and a single-isotope element consumes
/// NO isotope uniform, and the second test is on the element that was just chosen - so a caller
/// of the two-argument form has to know the chosen element's isotope count before it picks the
/// element. Which is to say it cannot, and every existing caller is a test feeding fixed
/// uniforms. In transport the draw count IS the random stream: aluminium and sodium are
/// single-isotope elements, so a port that always drew would put every subsequent sample in a
/// water-plus-aluminium detector on a different number from Geant4's.
///
/// The element index is decided from the partial sums first, exactly as the two-argument form
/// does, and only then is the isotope uniform drawn - so this consumes 0, 1 or 2 uniforms and
/// Geant4 consumes the same count on the same material and element.
template <typename real_t, typename XsFn, typename IsoArray, typename Rng>
__host__ __device__ inline TargetZA store_sample_za_rng(const XsFn& xs,
                                                        const data::Material<real_t>& mat,
                                                        const IsoArray& isos,
                                                        const MaterialXs<real_t>& mxs,
                                                        Rng& rng) {
  const real_t q_elm = (mat.n_elements > 1) ? rng.uniform() : real_t(0);
  int index = 0;
  if (mat.n_elements > 1) {
    const real_t cross = mxs.total * q_elm;
    for (int i = 0; i < mat.n_elements; ++i) {
      if (cross <= mxs.cumulative[i]) {
        index = i;
        break;
      }
    }
  }
  const ElementIsotopes<real_t> iso = isos[index];
  const real_t q_iso = (iso.n > 1) ? rng.uniform() : real_t(0);
  return store_sample_za_fn<real_t>(xs, mat, isos, mxs, q_elm, q_iso);
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
template <typename real_t, typename IsoArray>
__host__ __device__ inline XsValue<real_t> store_compute_cross_section(
    const PxsDataSet<real_t>& ds, real_t ekin, real_t loge,
    const data::Material<real_t>& mat, const IsoArray& isos,
    MaterialXs<real_t>& out) {
  return store_compute_cross_section_fn<real_t>(pxs_xs_functions<real_t>(ds, ekin, loge), mat,
                                               isos, out);
}

template <typename real_t, typename IsoArray>
__host__ __device__ inline TargetZA store_sample_za(const PxsDataSet<real_t>& ds, real_t ekin,
                                                    real_t loge,
                                                    const data::Material<real_t>& mat,
                                                    const IsoArray& isos,
                                                    const MaterialXs<real_t>& mxs,
                                                    real_t q_elm, real_t q_iso) {
  return store_sample_za_fn<real_t>(pxs_xs_functions<real_t>(ds, ekin, loge), mat, isos, mxs,
                                    q_elm, q_iso);
}

}  // namespace g4gpu::hadronic::xs
