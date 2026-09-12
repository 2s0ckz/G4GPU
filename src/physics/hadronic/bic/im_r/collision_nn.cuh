// The nucleon-nucleon collision channels: which one is in charge of a pair, what its cross
// section is, and - for the two elastic ones - what comes out.
//
// Transcribed from im_r_matrix in 11.1.1: G4CollisionNN (its constructor's channel list and its
// own `CrossSection`), G4GeneralNNCollision::IsInCharge, G4CollisionComposite (`CrossSection`,
// `FinalState`, `IsInCharge`), G4VCollision::CrossSection, G4CollisionNNElastic,
// G4CollisionnpElastic and G4VElasticCollision::FinalState.
//
// ## What `G4Scatterer` asks of this file and in what order
//
//   `GetTimeToInteraction` -> `FindCollision` -> `G4CollisionNN::IsInCharge`, then
//   `G4CollisionNN::CrossSection` - the TOTAL, from `G4XNNTotal` - turned into a disc of area
//   sigma and compared against the impact parameter.
//   `Scatter` -> the same `CrossSection`, then `G4CollisionComposite::FinalState`, which
//   evaluates each of the eight components' partial cross sections, throws one uniform against
//   their sum, and hands the collision to whichever component the running sum passes.
//
// So the elastic final states here are reachable only through a selection that needs all eight
// partials. Six of the eight are the resonance-production channels, which are not in this
// package: `CollisionRefusal::resonance_channels` is set by `collision_nn_final_state` and the
// final state is left empty. The two elastic channels are complete and are exercised directly.
//
// ## The recast in G4CollisionNN::CrossSection, which is the whole reason it is overridden
//
// A cascade nucleon is off its mass shell - `G4Fancy3DNucleus` puts every one of them below its
// PDG mass (docs/PORTED.md 2.1.10) - and the NN cross-section tables were measured on free
// nucleons. `G4CollisionNN::CrossSection` therefore rebuilds both tracks before asking:
//
//     t1 = p1.e() - trk1.GetActualMass();      // the KINETIC energy, off-shell
//     p1.setE(t1 + trk1.GetDefinition()->GetPDGMass());   // the same kinetic energy, on-shell
//
// keeping the three-momentum and moving the energy. That is not a Lorentz transformation and the
// resulting four-vector is not on shell either - its invariant mass is
// `sqrt((T+m_PDG)^2 - p^2)`, which equals `m_PDG` only if `p` was the on-shell momentum for `T`.
// It is what the source does, and the `sqrt(s)` the tables are then read at is the one built from
// these two recast vectors.
//
// The guard below it - `if ((p1+p2).mag() < m1_PDG + m2_PDG) return 0.` - uses the RECAST vectors
// and the PDG masses, so a pair whose recast invariant mass falls below threshold gets zero
// rather than an extrapolated table read.
//
// ## G4CollisionNN::GetComponents() returns a null pointer, and it does not matter
//
// `G4CollisionNN` declares its own `G4CollisionVector* components`, sets it to 0 in the
// constructor, never fills it, and overrides `GetComponents()` to return it - while
// `G4CollisionComposite::AddComponent`, which the constructor's `G4ForEach` calls eight times,
// pushes into the BASE class's by-value `components`. So the virtual accessor answers "no
// components" for an object with eight.
//
// Nothing reads it on a live path: `G4CollisionComposite::FinalState` uses the base member
// directly, `CrossSection` takes the `GetCrossSectionSource()` branch because `G4XNNTotal` is not
// null, and `IsInCharge` is overridden by `G4GeneralNNCollision` with a direct nucleon-pair test
// instead of the component scan that would have seen the null. Only `G4VCollision::Print` reads
// it, and it prints "has 0 components". Recorded here rather than in docs/RISK.md because it
// changes no number - but it is one override away from making `G4CollisionNN::IsInCharge` false
// for every pair in the cascade, which is why it is written down at all.
#ifndef G4GPU_BIC_IMR_COLLISION_NN_CUH
#define G4GPU_BIC_IMR_COLLISION_NN_CUH

#include <cmath>

#include "physics/hadronic/bic/im_r/angular.cuh"
#include "physics/hadronic/bic/im_r/lorentz_rotation.cuh"
#include "physics/hadronic/bic/im_r/xsec_nn.cuh"
#include "physics/hadronic/bic/kinetic_track.cuh"

namespace g4gpu::bic::imr {

/// The eight components of `G4CollisionNN`, in the order its constructor's
/// `GROUP8(G4CollisionnpElastic, G4CollisionNNElastic, G4CollisionNNToNDelta,
/// G4CollisionNNToDeltaDelta, G4CollisionNNToNDeltastar, G4CollisionNNToDeltaDeltastar,
/// G4CollisionNNToNNstar, G4CollisionNNToDeltaNstar)` registers them.
///
/// The ORDER is load-bearing and not cosmetic: `G4CollisionComposite::FinalState` accumulates the
/// partial cross sections in this order and returns the first component whose running sum passes
/// one uniform, so a permutation changes which channel a given random selects even when every
/// partial is right.
enum NNChannel : int {
  kNpElastic = 0,
  kNNElastic = 1,
  kNNToNDelta = 2,
  kNNToDeltaDelta = 3,
  kNNToNDeltastar = 4,
  kNNToDeltaDeltastar = 5,
  kNNToNNstar = 6,
  kNNToDeltaNstar = 7,
  kNNChannelCount = 8
};

/// What a collision could not do.
struct CollisionRefusal {
  /// The six resonance-production components of `G4CollisionNN` - every `G4CollisionNNTo*` and
  /// the `G4ConcreteNNTwoBodyResonance` tree, the `G4X*Table` cross sections and the resonance
  /// widths under them. Not in this package. Set wherever one of their partial cross sections
  /// would have been needed.
  bool resonance_channels = false;
  /// No component of `G4CollisionNN` is in charge of this pair.
  bool no_channel = false;
  int pdg1 = 0;
  int pdg2 = 0;
  XsecRefusal xsec;
  AngularRefusal angular;
  __host__ __device__ bool any() const {
    return resonance_channels || no_channel || xsec.any();
  }
};

/// `G4GeneralNNCollision::IsInCharge` - which `G4CollisionNN` inherits, overriding
/// `G4CollisionComposite`'s component scan. Two nucleons, either order, either species.
__host__ __device__ inline bool collision_nn_is_in_charge(int pdg1, int pdg2) {
  const bool a = (pdg1 == kPdgProton || pdg1 == kPdgNeutron);
  const bool b = (pdg2 == kPdgProton || pdg2 == kPdgNeutron);
  return a && b;
}

/// `G4CollisionNNElastic::IsInCharge` - pp or nn, the SAME species on both sides.
__host__ __device__ inline bool nn_elastic_is_in_charge(int pdg1, int pdg2) {
  return (pdg1 == kPdgProton && pdg2 == kPdgProton) ||
         (pdg1 == kPdgNeutron && pdg2 == kPdgNeutron);
}

/// `G4CollisionnpElastic::IsInCharge` - one of each, either order.
__host__ __device__ inline bool np_elastic_is_in_charge(int pdg1, int pdg2) {
  return (pdg1 == kPdgNeutron && pdg2 == kPdgProton) ||
         (pdg1 == kPdgProton && pdg2 == kPdgNeutron);
}

/// `G4CollisionNN::CrossSection` - the total NN cross section, on the recast on-shell pair. See
/// the file header for what the recast is and what it is not.
///
/// The two tracks are passed as their four-momenta and their two masses rather than as
/// `KineticTrack`s, so that the caller can see which mass goes where: `actual1`/`actual2` are
/// `GetActualMass()` (off shell) and `pdg1_mass`/`pdg2_mass` are `GetPDGMass()`.
__host__ __device__ inline double collision_nn_cross_section(
    int pdg1, int pdg2, const LorentzVector& p1_in, const LorentzVector& p2_in, double actual1,
    double actual2, double pdg1_mass, double pdg2_mass, XsecRefusal& ref) {
  LorentzVector p1 = p1_in;
  LorentzVector p2 = p2_in;
  const double t1 = p1.e - actual1;
  const double t2 = p2.e - actual2;
  p1.e = t1 + pdg1_mass;
  p2.e = t2 + pdg2_mass;
  const LorentzVector sum = p1 + p2;
  if (sum.mag() < pdg1_mass + pdg2_mass) { return 0.0; }
  return x_nn_total(pdg1, pdg2, pdg1_mass, pdg2_mass, sum.mag(), ref);
}

/// The two elastic channels' own cross sections. Unlike the composite's total above, these are
/// evaluated on the tracks AS GIVEN - `G4VCollision::CrossSection` passes `trk1` and `trk2`
/// straight to the source - so an off-shell cascade nucleon is read at its off-shell `sqrt(s)`
/// here and at its recast one there. Two different energies for the same pair in the same event,
/// and both are Geant4's.
__host__ __device__ inline double nn_elastic_cross_section(int pdg1, int pdg2,
                                                           const LorentzVector& p1,
                                                           const LorentzVector& p2,
                                                           double pdg1_mass, double pdg2_mass,
                                                           XsecRefusal& ref) {
  return x_nn_elastic(pdg1, pdg2, pdg1_mass, pdg2_mass, (p1 + p2).mag(), ref);
}
__host__ __device__ inline double np_elastic_cross_section(int pdg1, int pdg2,
                                                           const LorentzVector& p1,
                                                           const LorentzVector& p2,
                                                           double pdg1_mass, double pdg2_mass,
                                                           XsecRefusal& ref) {
  return x_np_elastic(pdg1, pdg2, pdg1_mass, pdg2_mass, (p1 + p2).mag(), ref);
}

/// The two outgoing four-momenta of a two-body elastic collision.
struct ElasticFinalState {
  LorentzVector p1;
  LorentzVector p2;
  bool empty = false;  ///< `S - (m10+m20)^2 < 0`, where Geant4 returns an empty vector
};

/// `G4VElasticCollision::FinalState`.
///
/// Four things in it a reader should not have to reconstruct:
///
///   * the outgoing masses are the PDG masses `m10` and `m20`, not the actual masses. So an
///     elastic collision between two off-shell nucleons puts two ON-shell nucleons out, and the
///     four-momentum is not conserved by the difference. That is the source; the binary cascade
///     repairs it afterwards in `CorrectFinalPandE`, which is not in this package.
///   * `cosTheta` is sampled with the ACTUAL masses (`m_1`, `m_2`) and the momentum `pInCM` is
///     built with the PDG ones. Two different mass pairs, four lines apart.
///   * the empty-vector branch is `S - (m10+m20)^2 < 0` and is tested BEFORE the actual masses
///     are read, so a pair below the PDG threshold returns nothing even if it is above the
///     off-shell one.
///   * between `cosTheta` and `phi` sits a block of nested `if`s over the species with EMPTY
///     bodies - `if (trk1.GetDefinition() == G4Proton::Proton()) { } else { }` - left over from a
///     debug printout. It consumes no randoms and changes nothing, and is not carried.
/// Which angular distribution a `G4VElasticCollision` subclass was constructed with. Three
/// subclasses reach this function and each passes a different one: `G4CollisionnpElastic` the NP
/// table, `G4CollisionNNElastic` the PP table, and `G4CollisionMesonBaryonElastic`
/// `G4AngularDistribution(false)` - the one-boson-exchange formula in its ASYMMETRIC branch,
/// which is the only place in the binary cascade that branch is live.
enum ElasticAngular : int { kAngularNp = 0, kAngularPp = 1, kAngularObeAsym = 2 };

template <typename Rng>
__host__ __device__ inline ElasticFinalState elastic_final_state(
    ElasticAngular which_angular, const LorentzVector& p1_in, const LorentzVector& p2_in,
    double actual1, double actual2, double m10, double m20, Rng& rng, AngularRefusal& ref) {
  ElasticFinalState out;
  const LorentzVector pcm = p1_in + p2_in;
  LorentzRotation to_lab = LorentzRotation::from_boost(pcm.boost_vector());
  const LorentzVector ptmp = to_lab.inverse() * p1_in;  // trk1 in the CM frame
  LorentzRotation to_z;
  to_z.rotate_z(-lv_phi(ptmp));
  to_z.rotate_y(-lv_theta(ptmp));
  to_lab = to_lab * to_z.inverse();

  const double S = pcm.e * pcm.e - g4gpu::mag2(pcm.v);  // pCM.mag2(), which may be negative
  if (S - (m10 + m20) * (m10 + m20) < 0.0) {
    out.empty = true;
    return out;
  }
  // The OBE branch builds its forty constants on every call, exactly as `G4VScatteringCollision`
  // constructs a fresh `G4AngularDistribution(true)` per object and `G4CollisionMesonBaryonElastic`
  // one `G4AngularDistribution(false)` per object - the constants are a function of nine fixed
  // numbers, so a cached copy and a rebuilt one are the same doubles.
  const double cos_theta =
      (which_angular == kAngularNp)
          ? angular_np_cos_theta(S, actual1, actual2, rng, ref)
          : ((which_angular == kAngularPp)
                 ? angular_pp_cos_theta(S, actual1, actual2, rng, ref)
                 : angular_obe_cos_theta(angular_obe_constants(), false, S, actual1, actual2,
                                         rng, ref));
  const double phi = angular_phi(rng);
  const double theta = std::acos(cos_theta);
  Vec3d p_final1{std::sin(theta) * std::cos(phi), std::sin(theta) * std::sin(phi), cos_theta};
  const double p_in_cm =
      std::sqrt((S - (m10 + m20) * (m10 + m20)) * (S - (m10 - m20) * (m10 - m20)) / (4.0 * S));
  p_final1 = p_final1 * p_in_cm;
  const Vec3d p_final2 = -1.0 * p_final1;
  const double e_final1 = std::sqrt(g4gpu::mag2(p_final1) + m10 * m10);
  const double e_final2 = std::sqrt(g4gpu::mag2(p_final2) + m20 * m20);
  out.p1 = to_lab * LorentzVector(p_final1, e_final1);
  out.p2 = to_lab * LorentzVector(p_final2, e_final2);
  return out;
}

/// `G4CollisionComposite::FinalState` for `G4CollisionNN` - the eight partial cross sections, one
/// uniform against their sum, and the component the running sum passes.
///
/// **REFUSED:** six of the eight partials are the resonance-production channels, and this package
/// does not have them. The selection cannot be performed without them - not even approximately,
/// because at 400 MeV the inelastic channels carry a comparable share of the total and dropping
/// them would renormalise the elastic one to 1 - so the function refuses by name and returns an
/// empty state. It is written out rather than left absent so that the missing partials have a
/// place to be plugged into and so that the ORDER is recorded now, while the constructor that
/// fixes it is in front of the reader.
///
/// `partial_out` is filled with whatever partials this package can supply, so a caller (and the
/// test) can see the two elastic ones even while the selection is refused.
template <typename Rng>
__host__ __device__ inline ElasticFinalState collision_nn_final_state(
    int pdg1, int pdg2, const LorentzVector& p1, const LorentzVector& p2, double actual1,
    double actual2, double pdg1_mass, double pdg2_mass, Rng& rng, CollisionRefusal& ref,
    double* partial_out) {
  ElasticFinalState out;
  out.empty = true;
  for (int i = 0; i < kNNChannelCount; ++i) { partial_out[i] = 0.0; }
  if (!collision_nn_is_in_charge(pdg1, pdg2)) {
    ref.no_channel = true;
    ref.pdg1 = pdg1;
    ref.pdg2 = pdg2;
    return out;
  }
  if (np_elastic_is_in_charge(pdg1, pdg2)) {
    partial_out[kNpElastic] =
        np_elastic_cross_section(pdg1, pdg2, p1, p2, pdg1_mass, pdg2_mass, ref.xsec);
  }
  if (nn_elastic_is_in_charge(pdg1, pdg2)) {
    partial_out[kNNElastic] =
        nn_elastic_cross_section(pdg1, pdg2, p1, p2, pdg1_mass, pdg2_mass, ref.xsec);
  }
  // Components 2 to 7: G4CollisionNNToNDelta, ...ToDeltaDelta, ...ToNDeltastar,
  // ...ToDeltaDeltastar, ...ToNNstar, ...ToDeltaNstar.
  ref.resonance_channels = true;
  ref.pdg1 = pdg1;
  ref.pdg2 = pdg2;
  (void)rng;
  (void)actual1;
  (void)actual2;
  return out;
}

}  // namespace g4gpu::bic::imr

#endif
