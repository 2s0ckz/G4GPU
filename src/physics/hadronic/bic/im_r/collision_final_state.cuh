// G4Scatterer::Scatter - the final state of one collision, through the whole tree.
//
// Transcribed from G4Scatterer.cc, G4CollisionComposite::FinalState,
// G4ConcreteMesonBaryonToResonance::GetOutgoingParticle and G4VAnnihilationCollision::FinalState
// (im_r_matrix, 11.1.1). The leaves - `G4VElasticCollision::FinalState`,
// `G4VScatteringCollision::FinalState` and the annihilation one - are in `collision_nn.cuh` and
// `resonance_fs.cuh` and were checked against the oracle on their own; this file is the DISPATCH
// that decides which of them runs and with what.
//
// ## THE SELECTION IS THREE DEEP, AND EACH LEVEL DRAWS ITS OWN UNIFORM
//
// `G4CollisionComposite::FinalState` sums its components' `CrossSection`, draws one uniform
// against the sum, and calls the winner's `FinalState` - and the winner is often another
// composite. For a nucleon pair:
//
//     G4CollisionNN                    8 components, one uniform
//       -> G4CollisionNNToNDeltastar   9 middle composites, one uniform
//            -> the concrete channels  one uniform
//                 -> G4VScatteringCollision::FinalState, which draws the masses and the angles
//
// and the cross sections each level sums are NOT the same kind of number. A composite without a
// cross-section source answers from its 32-point buffer (docs/RISK.md V109), so the top level and
// the middle level are both reading caches; a concrete channel has a source and is evaluated at
// the tracks as given. `channels.cuh` carries both.
//
// So the number of uniforms a collision consumes depends on which branch it takes, and the ORDER
// is: component, then middle, then concrete, then whatever the leaf draws. A port that flattened
// the tree - selecting one of the 306 concrete channels with a single uniform against their sum -
// would produce the same DISTRIBUTION and a different event.
//
// ## `Scatter` IS A WRAPPER AROUND `FinalState` AND ONE FATAL EXCEPTION
//
// Everything else in `G4Scatterer::Scatter` is `#ifdef debug_G4Scatterer` or an energy balance
// printed only under an environment variable - except the charge balance, which is NOT optional:
//
//     if(chargeBalance !=0 ) { ... G4Exception("G4Scatterer", "im_r_matrix001", FatalException,
//                                              "Problem in ChargeBalance"); }
//
// A collision whose products do not carry the entrance charge kills the run. `ScatterRefusal::
// charge_imbalance` is where the port puts it; nothing in the tree should be able to reach it,
// and that is worth checking rather than assuming.
//
// ## REFUSED, by name
//
//   * nothing. The eight nucleon components, the six resonance families with their two middle
//     layers, the 306 concrete channels, and both meson-baryon components all reach a leaf here.
#ifndef G4GPU_BIC_IMR_COLLISION_FINAL_STATE_CUH
#define G4GPU_BIC_IMR_COLLISION_FINAL_STATE_CUH

#include <cmath>

#include "physics/hadronic/bic/im_r/collision_meson.cuh"
#include "physics/hadronic/bic/im_r/resonance_fs.cuh"
#include "physics/hadronic/bic/im_r/scatterer.cuh"

namespace g4gpu::bic::imr {

/// What one collision produced, and which way it went to get there.
struct ScatterFinalState {
  int n = 0;              ///< 0 is Geant4's NULL or empty vector
  int pdg[2] = {0, 0};
  LorentzVector p[2];
  int channel = -1;       ///< 0 for G4CollisionNN, 1 for G4CollisionMesonBaryon
  int component = -1;     ///< which component of that composite
  int middle = -1;        ///< which middle-layer child, or -1
  int concrete = -1;      ///< index into the 306 concrete channels (or the 25), or -1
};

/// `G4ConcreteMesonBaryonToResonance::GetOutgoingParticle` and the same mapping for the nucleon
/// side: the charge state of a multiplet picked by the SUM of the two entrance iso3 values.
///
/// `G4ParticleTypeConverter::FindIso3State` walks the converter's map for an entry whose generic
/// type matches and whose `GetPDGiIsospin3()` equals the sum, so the lookup is by VALUE and the
/// port has to turn it into an index - and the two code lists in `channels.cuh` are ordered the
/// OPPOSITE WAY ROUND. `delta_codes` runs (minus, zero, plus, plusplus), increasing iso3, because
/// that is the order the `MakeNNTo*` templates take a Delta multiplet in; `nstar_codes` runs
/// (plus, zero), DECREASING iso3, because that is the order they take an N* multiplet in. So the
/// Delta index is `(iso3+3)/2` and the N* index is `(1-iso3)/2`, and getting the second one
/// backwards costs exactly one charge state: MEASURED, a pi- on a proton then produces N(1440)+
/// where Geant4 produces N(1440)0, and nothing else in 1,970 products moves.
///
/// When the sum is outside the multiplet's range Geant4 prints
/// "for <type> <iso3>" and THROWS (docs/RISK.md V110); here that is `ResonanceFsRefusal::
/// outgoing_state_missing` and a zero code. It should be unreachable: the cross section of a
/// channel whose isospin cannot reach the state is zero, so the selection never lands on it.
__host__ __device__ inline int iso3_state_of_multiplet(int mass, bool is_delta, int iso3_sum,
                                                       bool& missing) {
  missing = false;
  if (is_delta) {
    if (iso3_sum < -3 || iso3_sum > 3 || ((iso3_sum + 3) % 2) != 0) {
      missing = true;
      return 0;
    }
    return delta_codes(mass)[(iso3_sum + 3) / 2];
  }
  if (iso3_sum < -1 || iso3_sum > 1 || ((iso3_sum + 1) % 2) != 0) {
    missing = true;
    return 0;
  }
  return nstar_codes(mass)[(1 - iso3_sum) / 2];
}

/// `G4CollisionComposite::FinalState`'s selection over an arbitrary list of partials: the running
/// sum against one uniform, and -1 for Geant4's fall-through `return NULL`.
template <typename Rng>
__host__ __device__ inline int composite_select(const double* partial, int n, Rng& rng) {
  double sum = 0.0;
  for (int i = 0; i < n; ++i) { sum += partial[i]; }
  const double random = rng.uniform() * sum;
  double running = 0.0;
  for (int i = 0; i < n; ++i) {
    running += partial[i];
    if (running > random) { return i; }
  }
  return -1;
}

/// The second and third levels of a nucleon pair's resonance branch.
///
/// `group` is one of the six `G4CollisionNNTo*` components. Four of them hold their concrete
/// channels directly; `kNNToNDeltastar` and `kNNToDeltaDeltastar` hold nine middle composites
/// each, one per Delta* multiplet, and those answer from their own buffers. Returns the index of
/// the chosen concrete channel in `chans`, or -1.
template <typename Rng>
__host__ __device__ inline int select_concrete_channel(const ConcreteChannel* chans, int n_chan,
                                                       int group, int pdg1, int pdg2,
                                                       double sqrt_s, const NNChannelBuffers& buf,
                                                       Rng& rng, int& middle_out,
                                                       ResonanceTableRefusal& ref) {
  middle_out = -1;
  const bool has_middle = (group == kNNToNDeltastar || group == kNNToDeltaDeltastar);
  if (has_middle) {
    double child_partial[9];
    for (int i = 0; i < 9; ++i) {
      const double* nodes = (group == kNNToNDeltastar) ? buf.ndeltastar_child[i]
                                                       : buf.deltadeltastar_child[i];
      child_partial[i] = buffered_cross_section(buf.grid, nodes, kBufferPoints, sqrt_s);
    }
    const int middle = composite_select(child_partial, 9, rng);
    if (middle < 0) { return -1; }
    middle_out = middle;
  }
  // The concrete channels under the chosen parent, in the order the constructor added them.
  double partial[64];
  int index[64];
  int count = 0;
  for (int c = 0; c < n_chan && count < 64; ++c) {
    const ConcreteChannel& ch = chans[c];
    if (ch.group != group) { continue; }
    if (ch.child != middle_out) { continue; }
    if (!concrete_is_in_charge(ch, pdg1, pdg2)) {
      // `G4CollisionComposite::FinalState` still gives an out-of-charge component a slot in
      // its cache, with a partial of zero - it does not skip it. MEASURED: skipping them instead
      // changes none of the 1,970 products, because a zero slot can never win the running-sum
      // comparison - `running > random` needs the running sum to INCREASE past the draw, and a
      // zero partial does not increase it. The two are the same selection and the slot is kept
      // because the cache Geant4 builds has that shape and a reader should not have to prove the
      // equivalence again.
      partial[count] = 0.0;
    } else {
      partial[count] = concrete_cross_section(ch, pdg1, pdg2, sqrt_s, ref);
    }
    index[count] = c;
    ++count;
  }
  const int pick = composite_select(partial, count, rng);
  return (pick < 0) ? -1 : index[pick];
}

/// `G4Scatterer::Scatter` for a NUCLEON pair - `G4CollisionNN::FinalState` and everything under
/// it.
template <typename Rng>
__host__ __device__ inline ScatterFinalState nn_scatter_final_state(
    const ConcreteChannel* chans, int n_chan, int pdg1, int pdg2, const LorentzVector& p1,
    const LorentzVector& p2, double actual1, double actual2, double pdg1_mass, double pdg2_mass,
    double proton_mass, double neutron_mass, double pion_mass, const NNChannelBuffers& buf,
    Rng& rng, ScatterRefusal& ref) {
  ScatterFinalState out;
  out.channel = 0;
  double partial[kNNChannelCount];
  nn_partial_cross_sections(pdg1, pdg2, p1, p2, pdg1_mass, pdg2_mass, buf, partial,
                            ref.collision.xsec);
  const int component = nn_select_channel(partial, rng);
  out.component = component;
  if (component < 0) { return out; }
  if (component == kNpElastic || component == kNNElastic) {
    // `G4CollisionnpElastic` was built with the NP table and `G4CollisionNNElastic` with the PP
    // one; both are `G4VElasticCollision::FinalState`, and the outgoing species are the entrance
    // species unchanged.
    const ElasticAngular which = (component == kNpElastic) ? kAngularNp : kAngularPp;
    const ElasticFinalState fs = elastic_final_state(which, p1, p2, actual1, actual2, pdg1_mass,
                                                     pdg2_mass, rng, ref.collision.angular);
    if (fs.empty) { return out; }
    out.n = 2;
    out.pdg[0] = pdg1;
    out.pdg[1] = pdg2;
    out.p[0] = fs.p1;
    out.p[1] = fs.p2;
    return out;
  }
  const double sqrt_s = (p1 + p2).mag();
  int middle = -1;
  ResonanceTableRefusal tref;
  const int c = select_concrete_channel(chans, n_chan, component, pdg1, pdg2, sqrt_s, buf, rng,
                                        middle, tref);
  out.middle = middle;
  out.concrete = c;
  if (c < 0) { return out; }
  const ConcreteChannel& ch = chans[c];
  const SpeciesProperties o1 = species_properties(ch.out1, proton_mass, neutron_mass);
  const SpeciesProperties o2 = species_properties(ch.out2, proton_mass, neutron_mass);
  if (!o1.known || !o2.known) {
    ref.final_state = true;
    ref.pdg1 = ch.out1;
    ref.pdg2 = ch.out2;
    return out;
  }
  ResonanceFsRefusal rref;
  const ElasticFinalState fs = scattering_final_state(p1, p2, actual1, actual2, o1, o2,
                                                      neutron_mass, pion_mass, rng, rref);
  if (fs.empty) { return out; }
  out.n = 2;
  out.pdg[0] = ch.out1;
  out.pdg[1] = ch.out2;
  out.p[0] = fs.p1;
  out.p[1] = fs.p2;
  return out;
}

/// `G4Scatterer::Scatter` for a MESON-BARYON pair - `G4CollisionMesonBaryon::FinalState`.
///
/// The to-resonance branch produces ONE track: `G4VAnnihilationCollision::FinalState` puts the
/// whole invariant mass into a single resonance at rest in the CM and boosts it back.
template <typename Rng>
__host__ __device__ inline ScatterFinalState meson_scatter_final_state(
    int pdg1, int pdg2, const LorentzVector& p1, const LorentzVector& p2, double actual1,
    double actual2, double pdg1_mass, double pdg2_mass, double m_pion, double m_baryon,
    double m_pi_plus, double proton_mass, int iso3_1, int iso3_2, const MesonBaryonBuffers& buf,
    Rng& rng, ScatterRefusal& ref) {
  ScatterFinalState out;
  out.channel = 1;
  double partial[kMesonBaryonChannelCount];
  meson_baryon_partials(buf, (pdg1 == kPdgProton || pdg1 == kPdgNeutron) ? pdg2 : pdg1,
                        (pdg1 == kPdgProton || pdg1 == kPdgNeutron) ? pdg1 : pdg2, m_pion,
                        m_baryon, p1, p2, m_pi_plus, proton_mass, partial, ref.meson.xsec);
  const int component = meson_baryon_select(partial, rng);
  out.component = component;
  if (component < 0) { return out; }
  if (component == kMesonBaryonElastic) {
    // The outgoing masses are the PDG ones and the angle is sampled with the actual ones;
    // see the note on `elastic_final_state`.
    const ElasticFinalState fs = meson_baryon_elastic_final_state(
        p1, p2, actual1, actual2, pdg1_mass, pdg2_mass, rng, ref.collision.angular);
    if (fs.empty) { return out; }
    out.n = 2;
    out.pdg[0] = pdg1;
    out.pdg[1] = pdg2;
    out.p[0] = fs.p1;
    out.p[1] = fs.p2;
    return out;
  }
  // The 25 concrete channels, each evaluated at the tracks as given - a concrete channel has a
  // cross-section source, so nothing here is buffered.
  const double sqrt_s = (p1 + p2).mag();
  double cpartial[kMesonBaryonToResonanceCount];
  AnnihRefusal aref;
  for (int c = 0; c < kMesonBaryonToResonanceCount; ++c) {
    const MesonBaryonChannelSpec& sp = meson_baryon_channels()[c];
    const SpeciesProperties res = species_properties(sp.res_pdg, proton_mass, m_baryon);
    if (!res.known) {
      cpartial[c] = 0.0;
      continue;
    }
    cpartial[c] = x_annihilation_channel(0, m_pion, 2, iso3_1, 1, m_baryon, 1, iso3_2, sp.mass,
                                         sp.is_delta, res.two_spin, res.mass, res.width,
                                         sp.is_delta ? 3 : 1, sqrt_s, aref);
  }
  const int c = composite_select(cpartial, kMesonBaryonToResonanceCount, rng);
  out.concrete = c;
  if (c < 0) { return out; }
  const MesonBaryonChannelSpec& sp = meson_baryon_channels()[c];
  bool missing = false;
  const int outgoing = iso3_state_of_multiplet(sp.mass, sp.is_delta, iso3_1 + iso3_2, missing);
  if (missing) {
    ref.final_state = true;
    ref.pdg1 = sp.res_pdg;
    return out;
  }
  out.n = 1;
  out.pdg[0] = outgoing;
  out.p[0] = annihilation_final_state(p1, p2);
  return out;
}

/// `G4Scatterer::Scatter`'s charge balance, which is a FatalException and not a warning.
///
/// `charge_of` is the caller's, because the port has no particle table; the cascade knows the
/// charge of every track it holds.
__host__ __device__ inline bool scatter_charge_balanced(int q_in1, int q_in2,
                                                        const ScatterFinalState& fs,
                                                        const int* q_out) {
  int balance = q_in1 + q_in2;
  for (int i = 0; i < fs.n; ++i) { balance -= q_out[i]; }
  return balance == 0;
}

}  // namespace g4gpu::bic::imr

#endif
