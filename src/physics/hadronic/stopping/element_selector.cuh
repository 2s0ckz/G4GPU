// G4ElementSelector: which atom in the material captures the stopped negative particle.
//
// Transcribed from Geant4 11.1.1,
//   processes/hadronic/stopping/src/G4ElementSelector.cc::SelectZandA
//
// ---------------------------------------------------------------------------------------------
// **The Fermi-Teller Z-law, with two exceptions that are not in the law.** A stopped negative
// hadron or muon is captured by an atom with probability proportional to Z times the number
// density - that is Fermi and Teller - and Geant4 overrides it for exactly two cases:
//
//     halogens (Z = 9, 17, 35, 53, 85)   weight 0.66 * Z * n
//     oxygen   (Z = 8)                   weight 0.56 * Z * n
//     everything else                    weight        Z * n
//
// with `N.C.Mukhopadhyay Phys. Rep. 30 (1977) 1` named beside them. The numbers matter here
// rather than anywhere else in this port: **water is 2 hydrogens and 1 oxygen**, so the oxygen
// exception moves the capture fraction in the one material a medical or space-dosimetry run is
// mostly made of. Without it a stopped pi- in water captures on oxygen 8/(8+2) = 80.0% of the
// time; with it, 0.56*8/(0.56*8 + 2) = 69.1%. Every capture product downstream - the EM cascade's
// Z^4 gamma weighting, the nuclear model's target, the residual - follows that choice.
//
// **One uniform for the element, one more for the isotope.** The element loop accumulates into a
// running sum and compares `sum * G4UniformRand()` against the cumulative; the isotope loop
// subtracts relative abundances from a second uniform. Two draws, in that order, and a
// single-element material spends NEITHER: `if (1 < numberOfElements)` guards the first, and
// `if (1 < ni)` the second. That is why a dump over carbon, aluminium, iron and lead consumes no
// deviates here and water consumes one, and why a draw-count comparison sees the difference.
//
// **Both loops can fall off their arrays, and only one of them is safe.** The element loop ends
// `for (i=0; ...) { if (sum <= prob[i]) break; }` with `sum` already multiplied by a uniform in
// [0,1): `prob[n-1]` is the unmultiplied total, so the test is `total*u <= total` and cannot
// fail. The ISOTOPE loop is `y -= ab[i]` until `y <= 0` with `y` a uniform in [0,1), and it
// relies on the abundances summing to at least 1. Geant4 normalises them when an element is
// built from natural abundances, so it holds for every NIST element - but it is an assumption
// about the data and not a property of the loop, and a hand-built element whose abundances sum
// to 0.999 would read `GetIsotope(ni)` off the end. This port REFUSES that case by name rather
// than reading past the end; see `kAbundanceShort`.
#ifndef G4GPU_STOPPING_ELEMENT_SELECTOR_CUH
#define G4GPU_STOPPING_ELEMENT_SELECTOR_CUH

#include "physics/hadronic/process.cuh"

namespace g4gpu::physics::hadronic::stopping {

/// What the element selector could not do.
enum class SelectorRefusal : int {
  kNone = 0,
  /// The isotope loop ran off the end because the relative abundances of the chosen element sum
  /// to less than the uniform it was handed. Geant4 would read `GetIsotope(ni)` past the end of
  /// the vector; what it would find there is not defined by the source, so this is refused.
  kAbundanceShort,
  /// The material has no elements at all.
  kEmptyMaterial
};

/// The (Z, A) a capture happened on.
struct CaptureTarget {
  int z = 0;
  int a = 0;
  int element_index = -1;   ///< which entry of the material it was
  SelectorRefusal refusal = SelectorRefusal::kNone;
};

/// G4ElementSelector's per-element weight: the Fermi-Teller Z times the two exceptions.
///
/// Written as its own function because it is the whole of the physics in this file and because
/// `tests/test_stopping.cu` asserts it element by element against the dump, where a factor that
/// is only ever applied to two of a hundred values is otherwise invisible.
__host__ __device__ inline double element_capture_weight(int z) {
  if (z == 9 || z == 17 || z == 35 || z == 53 || z == 85) { return 0.66 * double(z); }
  if (z == 8) { return 0.56 * double(z); }
  return double(z);
}

/// G4ElementSelector::SelectZandA.
///
/// `rng` is drawn from exactly as Geant4 draws: once for the element when the material has more
/// than one, once for the isotope when the element has more than one, and not at all otherwise.
template <typename real_t, typename Rng>
__host__ __device__ inline CaptureTarget select_z_and_a(const MaterialComposition<real_t>& mat,
                                                        Rng& rng) {
  CaptureTarget t;
  if (mat.n_elements <= 0) {
    t.refusal = SelectorRefusal::kEmptyMaterial;
    return t;
  }

  int i = 0;
  if (mat.n_elements > 1) {
    // The running sum, and then one uniform against it. Geant4 keeps `prob` as a member vector
    // resized on demand; here the cumulative is formed twice rather than stored, because a
    // hundred-element material does not exist and a second pass is cheaper than a buffer.
    double sum = 0.0;
    for (int k = 0; k < mat.n_elements; ++k) {
      sum += element_capture_weight(mat.element_z[k]) * double(mat.n_atoms_per_volume[k]);
    }
    const double pick = sum * double(rng.uniform());
    double run = 0.0;
    for (i = 0; i < mat.n_elements; ++i) {
      run += element_capture_weight(mat.element_z[i]) * double(mat.n_atoms_per_volume[i]);
      if (pick <= run) { break; }
    }
    // `pick <= total` always holds for a uniform in [0,1), so the loop always breaks; the clamp
    // is here so that a denormal total cannot index off the end.
    if (i >= mat.n_elements) { i = mat.n_elements - 1; }
  }

  t.element_index = i;
  t.z = mat.element_z[i];

  const int ni = mat.n_isotopes[i];
  const int off = mat.isotope_offset[i];
  int j = 0;
  if (ni > 1) {
    double y = double(rng.uniform());
    for (j = 0; j < ni; ++j) {
      y -= double(mat.isotope_abundance[off + j]);
      if (y <= 0.0) { break; }
    }
    if (j >= ni) {
      // See the header: Geant4 reads off the end here. Refused by name.
      t.refusal = SelectorRefusal::kAbundanceShort;
      return t;
    }
  }
  t.a = mat.isotope_a[off + j];
  return t;
}

}  // namespace g4gpu::physics::hadronic::stopping

#endif  // G4GPU_STOPPING_ELEMENT_SELECTOR_CUH
