// G4CollisionMesonBaryon - the channel a PION collides through, and both halves of it.
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
// Both are here now. The TO-RESONANCE one is 25 `G4ConcreteMesonBaryonToResonance` channels over
// `G4XAnnihilationChannel` (`xsec_annihilation.cuh`), and it is itself a `G4CollisionComposite`
// with no cross-section source, so the tree is TWO buffers deep - see `MesonBaryonBuffers`.
//
// docs/RISK.md V107 measured that the elastic half is ZERO for every pion below 2 GeV/c lab
// momentum, which is T = 1865 MeV for a pion. Over the whole of the cascade window, therefore,
// the pion's entire cross section - the number `G4Scatterer` turns into an interaction radius,
// and the only branch `FinalState` can take - is resonance production.
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
// ## Nothing is refused here, and the reason the list is short is upstream
//
// The kaon and hyperon channels are in `G4CollisionMesonBaryonToResonance`'s constructor and they
// are COMMENTED OUT in 11.1.1 - 11 Lambda and 7 Sigma channels inside one `/* ... */` block. So
// the meson-baryon resonance tree is pion-nucleon only, which answers the question of whether
// QBBC's species set reaches them: nothing does, because the code that would is not compiled.
// Recorded here rather than as a refusal, because there is nothing to refuse.
#ifndef G4GPU_BIC_IMR_COLLISION_MESON_CUH
#define G4GPU_BIC_IMR_COLLISION_MESON_CUH

#include <cmath>

#include "physics/hadronic/bic/im_r/channels.cuh"
#include "physics/hadronic/bic/im_r/collision_nn.cuh"
#include "physics/hadronic/bic/im_r/xsec_annihilation.cuh"
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
  /// The 32-point buffer for this pair had not been built, so the composite's total could not be
  /// read off it. A device kernel cannot build it inside the call the way Geant4 does.
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

/// The 25 `G4ConcreteMesonBaryonToResonance` channels, in the order
/// `G4CollisionMesonBaryonToResonance`'s constructor adds them: the ten Deltas first, then the
/// fifteen N*, each by increasing nominal mass. That order is what
/// `G4CollisionComposite::FinalState` accumulates in.
///
/// Every one of them has the SAME `IsInCharge` - a generic-type comparison against
/// (proton, pi+), and `G4ParticleTypeConverter` maps all three pions to `PION` and both nucleons
/// to `NUCLEON` - so all 25 are in charge of every pion-nucleon pair and the composite sums all
/// 25. What separates them is the cross section: `NormalizedClebsch` returns 0 when
/// `isoRes < iso3`, which is what makes a pi+ on a proton give zero for every N* channel. That is
/// also what keeps `GetOutgoingParticle` from throwing on a channel it cannot make
/// (docs/RISK.md V110) - the protection is the cross section and nothing else.
struct MesonBaryonChannelSpec {
  int res_pdg;
  int mass;
  bool is_delta;
};

inline constexpr int kMesonBaryonToResonanceCount = 25;

__host__ __device__ inline const MesonBaryonChannelSpec* meson_baryon_channels() {
  static const MesonBaryonChannelSpec v[kMesonBaryonToResonanceCount] = {
      {2214, 1232, true},   {32214, 1600, true},  {2122, 1620, true},  {12214, 1700, true},
      {12122, 1900, true},  {2126, 1905, true},   {22122, 1910, true}, {22214, 1920, true},
      {12126, 1930, true},  {2218, 1950, true},
      {12212, 1440, false}, {2124, 1520, false},  {22212, 1535, false}, {32212, 1650, false},
      {2216, 1675, false},  {12216, 1680, false}, {22124, 1700, false}, {42212, 1710, false},
      {32124, 1720, false}, {42124, 1900, false}, {12218, 1990, false}, {52214, 2090, false},
      {2128, 2190, false},  {100002210, 2220, false}, {100012210, 2250, false}};
  return v;
}

/// The buffers `G4CollisionMesonBaryon` and its to-resonance child answer from. Same two-level
/// shape as the nucleon side (docs/RISK.md V109): the to-resonance child has no cross-section
/// source, so ITS total is a 32-point buffer of the sum of its 25 partials, and the parent
/// buffers the sum of that buffered value and the elastic partial.
///
/// The ELASTIC child is NOT buffered. `G4CollisionMesonBaryonElastic` is a `G4VElasticCollision`
/// with a real `crossSectionSource` (`G4XMesonBaryonElastic`), so `G4VCollision::CrossSection`
/// evaluates it at the tracks as given. `parent[]` therefore holds the raw elastic AT THE NODE
/// TRACKS, which is what `BufferCrossSection` sums there, while `meson_baryon_partials` has to
/// go back to the raw formula at the collision's own energy. The two differ: the nodes are 10
/// MeV, 15.2 MeV, ... and the elastic turns on at 2 GeV/c, so a linear interpolation across the
/// node that straddles the threshold is not the function.
///
/// The middle buffer reproduces its own node exactly at 31 of the 32 nodes. The one it does not
/// is the LAST: `G4CrossSectionBuffer::CrossSection` searches for the first grid point ABOVE
/// sqrt(s), finds none at the top of the grid, and returns with `x1,y1` still at their
/// initialisers 1 and 0 - so the 0.01 mb floor on `y1` forces zero. **The composite's 32nd node
/// is therefore elastic-only for every pair**, whatever the resonance sum is there (7.946e-3 mb
/// at sqrt(s) = 13.74 GeV for pi+ p). docs/RISK.md V112; asserted in tests/test_bic_imr.cu,
/// because no energy the cascade reaches interpolates across that node.
struct MesonBaryonBuffers {
  double grid[kBufferPoints] = {};       ///< sqrt(s) at the 32 grid points, pion kinetic energy
  double to_resonance[kBufferPoints] = {};  ///< raw sum of the 25 concrete channels at the nodes
  double parent[kBufferPoints] = {};     ///< child-buffered to-resonance + raw elastic, per node
  bool built = false;
};

/// `G4CollisionComposite::BufferCrossSection` for the meson-baryon tree, bottom up.
///
/// The grid puts the kinetic energy on the LIGHTER particle, which for a pion-nucleon pair is
/// always the pion.
__host__ __device__ inline void build_meson_baryon_buffers(
    int pion_pdg, int baryon_pdg, double m_pion, double m_baryon, int iso3_pion, int iso3_baryon,
    double m_pi_plus, double m_proton, MesonBaryonBuffers& buf, AnnihRefusal& ref) {
  buffer_sqrt_s_grid(m_pion, m_baryon, buf.grid);
  double elastic_node[kBufferPoints];
  for (int t = 0; t < kBufferPoints; ++t) {
    buf.to_resonance[t] = 0.0;
    buf.parent[t] = 0.0;
    elastic_node[t] = 0.0;
  }
  for (int t = 0; t < kBufferPoints; ++t) {
    // The grid's tracks are the pion along +z and the baryon at rest, so the four-momenta are
    // rebuilt from the grid's own sqrt(s) rather than carried.
    const double sqrt_s = buf.grid[t];
    double sum = 0.0;
    for (int c = 0; c < kMesonBaryonToResonanceCount; ++c) {
      const MesonBaryonChannelSpec& sp = meson_baryon_channels()[c];
      const SpeciesProperties res = species_properties(sp.res_pdg, m_proton, m_baryon);
      if (!res.known) {
        ref.unknown_resonance = true;
        ref.refused_pdg = sp.res_pdg;
        continue;
      }
      sum += x_annihilation_channel(0, m_pion, 2, iso3_pion, 1, m_baryon, 1, iso3_baryon,
                                    sp.mass, sp.is_delta, res.two_spin, res.mass, res.width,
                                    sp.is_delta ? 3 : 1, sqrt_s, ref);
    }
    buf.to_resonance[t] = sum;
    // The node tracks are built from the grid energy itself, not reconstructed from sqrt(s):
    // `BufferCrossSection` puts `theT[tt]*GeV` on the pion and leaves the nucleon at rest, and
    // inverting the invariant mass to get it back costs the last few digits (the same mistake
    // the nucleon-side buffer dump made first).
    const double a_e = m_pion + composite_T()[t] * u::GeV<double>();
    const LorentzVector p1(Vec3d{0.0, 0.0, std::sqrt(a_e * a_e - m_pion * m_pion)}, a_e);
    const LorentzVector p2(Vec3d{0.0, 0.0, 0.0}, m_baryon);
    elastic_node[t] = x_meson_baryon_elastic(pion_pdg, baryon_pdg, m_pion, m_baryon, p1, p2,
                                             m_pi_plus, m_proton, ref.clebsch_xsec);
  }
  // The parent's node value is what `BufferCrossSection` pushes: the sum over the components of
  // `components[i]->CrossSection(a,b)` at the node tracks - and for the to-resonance child that
  // call is already an interpolation of its own buffer, floor and all.
  for (int t = 0; t < kBufferPoints; ++t) {
    buf.parent[t] = buffered_cross_section(buf.grid, buf.to_resonance, kBufferPoints,
                                           buf.grid[t]) +
                    elastic_node[t];
  }
  buf.built = true;
}

/// The two partial cross sections `G4CollisionComposite::FinalState` selects on for a
/// meson-baryon pair, in the order the constructor adds them - the to-resonance child's BUFFERED
/// total and the elastic child's RAW one. See the note on `MesonBaryonBuffers` for why only one
/// of the two comes from a table.
__host__ __device__ inline void meson_baryon_partials(const MesonBaryonBuffers& buf, int pion_pdg,
                                                      int baryon_pdg, double m_pion,
                                                      double m_baryon, const LorentzVector& p1,
                                                      const LorentzVector& p2, double m_pi_plus,
                                                      double m_proton, double* partial_out,
                                                      XsecRefusal& ref) {
  const double sqrt_s = (p1 + p2).mag();
  partial_out[kMesonBaryonToResonance] =
      buffered_cross_section(buf.grid, buf.to_resonance, kBufferPoints, sqrt_s);
  partial_out[kMesonBaryonElastic] = x_meson_baryon_elastic(pion_pdg, baryon_pdg, m_pion,
                                                            m_baryon, p1, p2, m_pi_plus,
                                                            m_proton, ref);
}

/// `G4CollisionComposite::CrossSection` for `G4CollisionMesonBaryon` - the BUFFERED total, which
/// is what `G4Scatterer` turns into an interaction radius for a pion.
///
/// The buffer has to have been built for this pair first; `built` is false otherwise and the
/// call refuses rather than returning the sum of an empty table. That is not Geant4's behaviour
/// - Geant4 builds the buffer lazily inside the call, under a mutex - but a device kernel cannot
/// allocate one, so the caller builds it and this checks.
__host__ __device__ inline double meson_baryon_cross_section(int pdg1, int pdg2, double sqrt_s,
                                                             const MesonBaryonBuffers& buf,
                                                             MesonRefusal& ref) {
  if (!meson_baryon_elastic_is_in_charge(pdg1, pdg2, ref.xsec)) {
    ref.no_channel = true;
    ref.pdg1 = pdg1;
    ref.pdg2 = pdg2;
    return 0.0;
  }
  if (!buf.built) {
    ref.to_resonance = true;
    ref.pdg1 = pdg1;
    ref.pdg2 = pdg2;
    return 0.0;
  }
  return buffered_cross_section(buf.grid, buf.parent, kBufferPoints, sqrt_s);
}

/// `G4CollisionMesonBaryon::FinalState` - `G4CollisionComposite::FinalState`'s draw between the
/// two components, and the index it lands on. -1 when the running sum never passes the draw,
/// which is what Geant4 returns NULL for (the throw below it is commented out in 11.1.1).
template <typename Rng>
__host__ __device__ inline int meson_baryon_select(const double* partial, Rng& rng) {
  double sum = 0.0;
  for (int i = 0; i < kMesonBaryonChannelCount; ++i) { sum += partial[i]; }
  const double random = rng.uniform() * sum;
  double running = 0.0;
  for (int i = 0; i < kMesonBaryonChannelCount; ++i) {
    running += partial[i];
    if (running > random) { return i; }
  }
  return -1;
}

}  // namespace g4gpu::bic::imr

#endif
