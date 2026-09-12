// G4CollisionMesonBaryon - the channel a PION collides through, and the half of it that is here.
//
// Transcribed from G4CollisionMesonBaryon, G4CollisionMesonBaryonElastic and
// G4CollisionMesonBaryonToResonance (im_r_matrix, 11.1.1).
//
// `G4Scatterer` registers exactly two channels, `GROUP2(G4CollisionNN, G4CollisionMesonBaryon)`,
// and the second is a `G4CollisionComposite` of two components added in this order:
//
//     G4CollisionComposite::AddComponent(new G4CollisionMesonBaryonToResonance());   // 0
//     G4CollisionComposite::AddComponent(new G4CollisionMesonBaryonElastic());       // 1
//
// The ELASTIC one is complete here. The TO-RESONANCE one is not: it is 25
// `G4ConcreteMesonBaryonToResonance` channels over `G4XAnnihilationChannel`, and it needs the
// quantum numbers of 25 resonances and `G4XResonance`'s isospin correction, neither of which is
// in this package. `MesonRefusal::to_resonance` is set at the point its partial cross section
// would have been needed.
//
// ## The composite has no cross-section source, so its total is BUFFERED
//
// `G4CollisionMesonBaryon` overrides nothing: `GetCrossSectionSource()` returns 0 from
// `G4CollisionComposite`, so `G4CollisionComposite::CrossSection` takes its other branch -
//
//     G4AutoLock l(&bufferMutex);
//     const_cast<G4CollisionComposite *>(this)->BufferCrossSection(trk1.GetDefinition(),
//                                                                  trk2.GetDefinition());
//     crossSect = BufferedCrossSection(trk1,trk2);
//
// - which sums its components' cross sections ONCE on a fixed 32-point kinetic-energy grid
// (`G4CollisionComposite::theT`, `imr_tables.hh`'s `composite_T()`), caches the result against
// the two particle definitions, and interpolates that cache forever after. So a pion-nucleon
// total cross section in the cascade is not evaluated at the pair's own energy at all; it is
// read off a 32-point spline of the sum, built at energies from 10 MeV to 100 GeV.
//
// **And the buffer is built with the kinetic energy on the LIGHTER particle.** The comment in
// `BufferCrossSection` says so - "A.R. 28-Sep-2012 Fix reproducibility problem / Assign the
// kinetic energy to the lightest of the two particles, instead to the first one always" - so the
// grid point `theT[i]` is the PION's kinetic energy for a pion-nucleon pair, and `sqrt(s)` at
// that point is `(a4Momentum + b4Momentum).mag()` of a moving pion on a nucleon at rest. The
// port reproduces that; it is why `composite_T()` was extracted with the other tables.
//
// ## REFUSED, by name
//
//   * **`G4CollisionMesonBaryonToResonance`** and the 25 `G4ConcreteMesonBaryonToResonance`
//     channels under it, with `G4XAnnihilationChannel` and `G4XResonance`. Without them the
//     composite's buffered total is incomplete, so `meson_baryon_cross_section` refuses rather
//     than returning the elastic partial alone: at 300 MeV the Delta(1232) resonance dominates
//     the pion-nucleon cross section and an elastic-only total would be low by a factor of
//     several and would then be handed to `G4Scatterer` as an interaction radius.
//   * **the kaon and hyperon channels.** They are in `G4CollisionMesonBaryonToResonance`'s
//     constructor and they are COMMENTED OUT in 11.1.1 - 11 Lambda and 7 Sigma channels inside a
//     `/* ... */` block. So the meson-baryon resonance tree is pion-nucleon only, which answers
//     the question of whether QBBC's species set reaches them: nothing does, because the code
//     that would is not compiled. Recorded here rather than as a refusal, because there is
//     nothing to refuse.
#ifndef G4GPU_BIC_IMR_COLLISION_MESON_CUH
#define G4GPU_BIC_IMR_COLLISION_MESON_CUH

#include <cmath>

#include "physics/hadronic/bic/im_r/collision_nn.cuh"
#include "physics/hadronic/bic/im_r/xsec_meson.cuh"

namespace g4gpu::bic::imr {

/// The two components of `G4CollisionMesonBaryon`, in the order its constructor adds them - which
/// is the order `G4CollisionComposite::FinalState` accumulates their partial cross sections in.
enum MesonBaryonChannel : int {
  kMesonBaryonToResonance = 0,
  kMesonBaryonElastic = 1,
  kMesonBaryonChannelCount = 2
};

/// What a meson-baryon collision could not do.
struct MesonRefusal {
  /// `G4CollisionMesonBaryonToResonance` - 25 channels over G4XAnnihilationChannel and
  /// G4XResonance, not in this package. See the file header for why the elastic partial alone is
  /// not offered as a total.
  bool to_resonance = false;
  bool no_channel = false;
  int pdg1 = 0;
  int pdg2 = 0;
  XsecRefusal xsec;
  AngularRefusal angular;
  __host__ __device__ bool any() const { return to_resonance || no_channel || xsec.any(); }
};

/// `G4CollisionMesonBaryonElastic::IsInCharge` - by PARTON COUNT and not by species: one track
/// with two partons and one with three, in either order.
///
/// That is wider than pion-nucleon. A Delta, an N* and every Delta* has three partons, so a pion
/// scattering elastically off a resonance already in the cascade is in charge here - which is the
/// only way a short-lived particle is ever an INCOMING track in `G4Scatterer`'s tree.
/// `G4GeneralNNCollision::IsInCharge` requires two nucleons, and
/// `G4ConcreteMesonBaryonToResonance::IsInCharge` compares generic types, so neither of those
/// accepts one.
__host__ __device__ inline bool meson_baryon_elastic_is_in_charge(int pdg1, int pdg2,
                                                                  XsecRefusal& ref) {
  int nq1 = 0, ns1 = 0, nq2 = 0, ns2 = 0;
  XsecRefusal a;
  XsecRefusal b;
  const bool ok1 = parton_counts(pdg1, nq1, ns1, a);
  const bool ok2 = parton_counts(pdg2, nq2, ns2, b);
  if (!ok1 || !ok2) {
    ref = ok1 ? b : a;
    return false;
  }
  const int p1 = nq1 + ns1;
  const int p2 = nq2 + ns2;
  return (p1 == 2 && p2 == 3) || (p2 == 2 && p1 == 3);
}

/// `G4CollisionMesonBaryonElastic::CrossSection` - `G4VCollision::CrossSection` over
/// `G4XMesonBaryonElastic`, on the tracks as given.
__host__ __device__ inline double meson_baryon_elastic_cross_section(
    int pdg1, int pdg2, double m1, double m2, const LorentzVector& p1, const LorentzVector& p2,
    double m_pi_plus, double m_proton, XsecRefusal& ref) {
  return x_meson_baryon_elastic(pdg1, pdg2, m1, m2, p1, p2, m_pi_plus, m_proton, ref);
}

/// `G4CollisionMesonBaryonElastic::FinalState` - `G4VElasticCollision`'s, with
/// `G4AngularDistribution(false)`.
///
/// This is the ONLY place in the binary cascade where the one-boson-exchange formula's
/// asymmetric branch runs: `G4VScatteringCollision` builds its distribution with `true` and both
/// nucleon-nucleon elastic channels use a table. The branch is the `else` of the one that is
/// everywhere else, and the difference is not cosmetic - the symmetric form returns
/// `(Cross(t') - Cross(tMax - t'))/(2 norm) + 0.5`, which is antisymmetric about cos = 0 by
/// construction, and this one returns `Cross(t')/norm`, which is not.
template <typename Rng>
__host__ __device__ inline ElasticFinalState meson_baryon_elastic_final_state(
    const LorentzVector& p1, const LorentzVector& p2, double actual1, double actual2, double m10,
    double m20, Rng& rng, AngularRefusal& ref) {
  return elastic_final_state(kAngularObeAsym, p1, p2, actual1, actual2, m10, m20, rng, ref);
}

/// `G4CollisionComposite::CrossSection` for `G4CollisionMesonBaryon` - REFUSED.
///
/// The composite has no cross-section source, so its total is the BUFFERED sum of both partials
/// over the 32-point grid, and one of the two partials is not in this package. Returning the
/// elastic one alone would hand `G4Scatterer` an interaction radius several times too small at
/// exactly the energy where the Delta resonance dominates.
///
/// `elastic_partial_out` is filled with what this package can supply, so a test can exercise the
/// elastic channel through the same call.
__host__ __device__ inline double meson_baryon_cross_section(
    int pdg1, int pdg2, double m1, double m2, const LorentzVector& p1, const LorentzVector& p2,
    double m_pi_plus, double m_proton, MesonRefusal& ref, double* elastic_partial_out) {
  if (elastic_partial_out != nullptr) { *elastic_partial_out = 0.0; }
  if (!meson_baryon_elastic_is_in_charge(pdg1, pdg2, ref.xsec)) {
    ref.no_channel = true;
    ref.pdg1 = pdg1;
    ref.pdg2 = pdg2;
    return 0.0;
  }
  if (elastic_partial_out != nullptr) {
    *elastic_partial_out = meson_baryon_elastic_cross_section(pdg1, pdg2, m1, m2, p1, p2,
                                                              m_pi_plus, m_proton, ref.xsec);
  }
  ref.to_resonance = true;
  ref.pdg1 = pdg1;
  ref.pdg2 = pdg2;
  return 0.0;
}

}  // namespace g4gpu::bic::imr

#endif
