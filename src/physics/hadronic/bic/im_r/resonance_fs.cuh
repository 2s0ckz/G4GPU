// What comes out of a resonance-production channel: the two outgoing four-momenta, with the
// resonance's mass sampled from a Breit-Wigner.
//
// Transcribed from G4VScatteringCollision::FinalState and ::SampleResonanceMass, the three
// `BrWig*` inlines in G4VScatteringCollision.hh, and G4VAnnihilationCollision::FinalState
// (im_r_matrix, 11.1.1), with the resonance masses and widths from
// G4ExcitedDeltaConstructor, G4ExcitedNucleonConstructor and G4ShortLivedConstructor.
//
// `G4ConcreteNNTwoBodyResonance` inherits the first of these, so it is what every one of the 306
// NN channels produces once `channels.cuh` has selected one. `G4ConcreteMesonBaryonToResonance`
// inherits the second, which puts ONE particle out carrying the whole CM energy.
//
// ## Two mass pairs, four lines apart
//
// `FinalState` reads a mass in three places and means a different one each time:
//
//     outm1/outm2   the OUTGOING PDG masses, resampled for a short-lived product
//     cosTheta      sampled with trk1.GetActualMass() and trk2.GetActualMass() - the INCOMING
//                   off-shell masses
//     pCM           built from outm1 and outm2
//
// and the two-shortlived case is not an error but a `G4cerr` under an environment variable:
//
//     if (OutputDefinitions[0]->IsShortLived() && OutputDefinitions[1]->IsShortLived())
//       { if(std::getenv("G4KCDEBUG")) G4cerr << "two shortlived for Type = " ... }
//
// which is every Delta-Delta, Delta-Delta* and Delta-N* channel - 186 of the 306. Both masses are
// then sampled, the second with `maxMass = sqrtS - outm1`, so the first sample constrains the
// second and the ORDER matters.
//
// ## The minimum mass is a nucleon plus a pion, and it is not the product's threshold
//
// Both `SampleResonanceMass` calls pass
//
//     G4Neutron::NeutronDefinition()->GetPDGMass() + G4PionPlus::PionPlus()->GetPDGMass()
//
// as the minimum - 1079.1 MeV - whatever the resonance is. For a Delta(1232) that is about right;
// for an N(2250) it is 1.2 GeV below the pole. And the FIRST call's maximum is
// `sqrtS - (m_n + m_pi)`, not `sqrtS - outm2`, so the two calls use different conventions: the
// first reserves a nucleon-plus-pion for its partner and the second reserves the partner's
// actual sampled mass.
//
// ## SampleResonanceMass's three-step fallback when the window is empty
//
//     if (minMass > maxMass) G4cerr << "##### ... particle out of mass range" << G4endl;
//     if (minMass > maxMass) minMass -= G4PionPlus::PionPlus()->GetPDGMass();
//     if (minMass > maxMass) minMass = 0;
//
// - print, then subtract a pion mass, then give up and allow anything down to zero. The printout
// is unconditional (no environment variable, no verbose flag), so a cascade that reaches it
// prints one line per collision. The port returns the fact instead.
//
// Below a width of 1e-10 GeV the sampler is `max(minMass, min(maxMass, poleMass))` and draws NO
// uniform; above it, exactly one. That matters for the random stream and is asserted.
//
// ## REFUSED, by name
//
//   * **the `chargeBalance` printout** at the end of `FinalState`, which prints four particle
//     names to `G4cout` when the charge does not balance and changes nothing. The port's test
//     asserts the balance over all 306 channels instead (channels.cuh).
//   * **`BrWigInt1`**, declared beside the two that are used and called by nothing.
//   * **`G4ConcreteMesonBaryonToResonance`'s cross section**, `G4XAnnihilationChannel`. Its
//     FINAL STATE is here, because it is six lines and is shared with nothing; its cross section
//     is a separate class this package does not have, so `MesonRefusal::to_resonance` still
//     stands and the channel cannot be selected.
#ifndef G4GPU_BIC_IMR_RESONANCE_FS_CUH
#define G4GPU_BIC_IMR_RESONANCE_FS_CUH

#include <cmath>

#include "physics/hadronic/bic/im_r/channels.cuh"

namespace g4gpu::bic::imr {

/// What a resonance final state could not do.
struct ResonanceFsRefusal {
  /// `SampleResonanceMass`'s `minMass > maxMass`, which Geant4 prints and then works around by
  /// subtracting a pion mass and, failing that, by allowing zero.
  bool mass_window_empty = false;
  /// The second fallback was needed too - `minMass` was set to 0.
  bool mass_window_zeroed = false;
  /// `S - (m10+m20)^2 < 0`: Geant4 returns an empty vector.
  bool below_threshold = false;
  /// No mass or width is known for this PDG code.
  bool unknown_species = false;
  int refused_pdg = 0;
  AngularRefusal angular;
};

/// The PDG mass and width of any species the collision tree puts out, and whether it is
/// short-lived. Nucleons come from the caller because they are the only two the rest of this
/// package already has; everything else is a resonance and comes from the extracted tables.
///
/// The four ground-state Delta charge states do NOT share a width: delta- is 117 MeV where the
/// other three are 120. That is `G4ShortLivedConstructor::ConstructResonances`, pinned by
/// `tools/extract_bic_imr.pl`, and `SampleResonanceMass` reads it - so a delta- comes out of the
/// cascade with a 2.5% narrower mass distribution than a delta0. docs/RISK.md V110.
struct SpeciesProperties {
  double mass = 0.0;
  double width = 0.0;
  /// `GetPDGiSpin()`, twice the spin. Taken from the PDG code, whose LAST DIGIT is 2J+1 for
  /// every baryon here - the rule the encodings are built on, and checked against Geant4's own
  /// definitions species by species rather than assumed.
  int two_spin = 0;
  bool short_lived = false;
  bool known = false;
};

/// 2J from a baryon PDG code: the last digit is 2J+1 - EXCEPT for the four highest N*, whose
/// codes are 100002210, 100002110, 100012210 and 100012110.
///
/// Those four do not follow the rule. A PDG code's last digit is 2J+1 and theirs is 0, which
/// would be 2J = -1; Geant4 gives all four `GetPDGiSpin() == 9`, and the codes are the
/// nine-digit "extended" form the standard reserves for nuclei, used here because the ordinary
/// encoding has no room for another N* at that spin. So the rule is applied where it holds and
/// the four are named - and the test compares every species' 2J against Geant4's own definition,
/// which is what turns this from an assumption into a checked table.
__host__ __device__ inline int two_spin_from_pdg(int pdg) {
  const int c = (pdg < 0) ? -pdg : pdg;
  if (c == 100002210 || c == 100002110 || c == 100012210 || c == 100012110) { return 9; }
  return (c % 10) - 1;
}

__host__ __device__ inline SpeciesProperties species_properties(int pdg, double proton_mass,
                                                                double neutron_mass) {
  SpeciesProperties s;
  s.two_spin = two_spin_from_pdg(pdg);
  if (pdg == kPdgProton) {
    s.mass = proton_mass;
    s.width = 0.0;
    s.known = true;
    return s;
  }
  if (pdg == kPdgNeutron) {
    // G4Neutron's PDG width is 7.478e-25 MeV, which is below SampleResonanceMass's 1e-10 GeV
    // threshold - but a neutron is not short-lived and is never resampled anyway.
    s.mass = neutron_mass;
    s.width = 7.478e-25;
    s.known = true;
    return s;
  }
  // Delta(1232): four charge states, three sharing a width.
  const int* g = delta_codes(1232);
  for (int i = 0; i < 4; ++i) {
    if (pdg == g[i]) {
      s.mass = 1232.0;
      s.width = (i == 0) ? 117.0 : 120.0;  // index 0 is the delta-
      s.short_lived = true;
      s.known = true;
      return s;
    }
  }
  for (int k = 0; k < 9; ++k) {
    const int m = deltastar_mass_list()[k];
    const int* c = delta_codes(m);
    for (int i = 0; i < 4; ++i) {
      if (pdg == c[i]) {
        s.mass = deltastar_mass()[k];
        s.width = deltastar_width()[k];
        s.short_lived = true;
        s.known = true;
        return s;
      }
    }
  }
  for (int k = 0; k < 15; ++k) {
    const int m = nstar_mass_list()[k];
    const int* c = nstar_codes(m);
    for (int i = 0; i < 2; ++i) {
      if (pdg == c[i]) {
        s.mass = nstar_mass()[k];
        s.width = nstar_width()[k];
        s.short_lived = true;
        s.known = true;
        return s;
      }
    }
  }
  return s;
}

/// `G4VScatteringCollision::BrWigInt0` - the integral of a Breit-Wigner, `2 gamma atan(2(x-m0)/gamma)`.
__host__ __device__ inline double brwig_int0(double x, double gamma, double m0) {
  return 2.0 * gamma * std::atan(2.0 * (x - m0) / gamma);
}

/// `G4VScatteringCollision::BrWigInv` - its inverse, `0.5 gamma tan(0.5 x/gamma) + m0`.
__host__ __device__ inline double brwig_inv(double x, double gamma, double m0) {
  return 0.5 * gamma * std::tan(0.5 * x / gamma) + m0;
}

/// `G4VScatteringCollision::SampleResonanceMass`.
///
/// The three-step fallback when the window is empty is transcribed as written - print (reported
/// here), subtract a pion mass, then allow zero - and the zero-width shortcut draws NO uniform.
template <typename Rng>
__host__ __device__ inline double sample_resonance_mass(double pole_mass, double gamma,
                                                        double a_min_mass, double max_mass,
                                                        double pion_mass, Rng& rng,
                                                        ResonanceFsRefusal& ref) {
  double min_mass = a_min_mass;
  if (min_mass > max_mass) { ref.mass_window_empty = true; }
  if (min_mass > max_mass) { min_mass -= pion_mass; }
  if (min_mass > max_mass) {
    min_mass = 0.0;
    ref.mass_window_zeroed = true;
  }
  // `1E-10*GeV` is 1e-7 MeV. Written as the source writes it.
  if (gamma < 1.0e-10 * u::GeV<double>()) {
    const double hi = (max_mass < pole_mass) ? max_mass : pole_mass;
    return (min_mass > hi) ? min_mass : hi;
  }
  const double fmin = brwig_int0(min_mass, gamma, pole_mass);
  const double fmax = brwig_int0(max_mass, gamma, pole_mass);
  const double f = fmin + (fmax - fmin) * rng.uniform();
  return brwig_inv(f, gamma, pole_mass);
}

/// `G4VScatteringCollision::FinalState` - the two outgoing four-momenta.
///
/// The order of the random draws is the order the cascade's stream depends on: the first
/// resonance mass, then the second, then `CosTheta` (one or two uniforms depending on the
/// distribution), then `Phi`. A stable product draws none for its mass.
template <typename Rng>
__host__ __device__ inline ElasticFinalState scattering_final_state(
    const LorentzVector& p1_in, const LorentzVector& p2_in, double actual1, double actual2,
    const SpeciesProperties& out1, const SpeciesProperties& out2, double neutron_mass,
    double pion_mass, Rng& rng, ResonanceFsRefusal& ref) {
  ElasticFinalState out;
  const LorentzVector p = p1_in + p2_in;
  const double sqrt_s = p.mag();
  const double S = sqrt_s * sqrt_s;

  double outm1 = out1.mass;
  double outm2 = out2.mass;
  // The minimum is a neutron plus a pi+, whatever the resonance is - see the file header.
  const double min_mass = neutron_mass + pion_mass;
  if (out1.short_lived) {
    outm1 = sample_resonance_mass(out1.mass, out1.width, min_mass, sqrt_s - min_mass, pion_mass,
                                  rng, ref);
  }
  if (out2.short_lived) {
    outm2 = sample_resonance_mass(out2.mass, out2.width, min_mass, sqrt_s - outm1, pion_mass,
                                  rng, ref);
  }

  // Sampled with the INCOMING actual masses, while pCM below uses the outgoing ones.
  const double cos_theta = angular_obe_cos_theta(angular_obe_constants(), true, S, actual1,
                                                 actual2, rng, ref.angular);
  const double phi = angular_phi(rng);

  const LorentzRotation from_cms = LorentzRotation::from_boost(p.boost_vector());
  const LorentzRotation to_cms_frame = from_cms.inverse();
  const LorentzVector temp = to_cms_frame * p1_in;
  LorentzRotation to_z;
  to_z.rotate_z(-lv_phi(temp));
  to_z.rotate_y(-lv_theta(temp));
  const LorentzRotation to_cms = to_z.inverse();

  const double st = std::sin(std::acos(cos_theta));
  Vec3d p_final1{st * std::cos(phi), st * std::sin(phi), cos_theta};
  const double num = (S - (outm1 + outm2) * (outm1 + outm2)) *
                     (S - (outm1 - outm2) * (outm1 - outm2));
  if (num < 0.0) {
    // `pCM` is a sqrt of a negative number here; Geant4 produces a NaN and carries on. The port
    // refuses instead, because a NaN four-momentum in a cascade is not recoverable and the
    // caller can reject the collision.
    ref.below_threshold = true;
    out.empty = true;
    return out;
  }
  const double p_cm = std::sqrt(num / (4.0 * S));
  p_final1 = p_final1 * p_cm;
  const Vec3d p_final2 = -1.0 * p_final1;
  const double e_final1 = std::sqrt(g4gpu::mag2(p_final1) + outm1 * outm1);
  const double e_final2 = std::sqrt(g4gpu::mag2(p_final2) + outm2 * outm2);
  LorentzVector p4_final1(p_final1, e_final1);
  LorentzVector p4_final2(p_final2, e_final2);
  p4_final1 = to_cms * p4_final1;
  p4_final2 = to_cms * p4_final2;
  const LorentzRotation to_lab = LorentzRotation::from_boost(p.boost_vector());
  out.p1 = to_lab * p4_final1;
  out.p2 = to_lab * p4_final2;
  return out;
}

/// `G4VAnnihilationCollision::FinalState` - ONE particle carrying the whole CM energy, at rest in
/// the CM and boosted to the lab. `G4ConcreteMesonBaryonToResonance` inherits it, so a
/// pion-nucleon collision that makes a resonance makes exactly one track and no angle is sampled:
/// the function draws no uniform at all.
__host__ __device__ inline LorentzVector annihilation_final_state(const LorentzVector& p1,
                                                                  const LorentzVector& p2) {
  const LorentzVector p = p1 + p2;
  const double sqrt_s = p.mag();
  LorentzVector p4_final(Vec3d{0.0, 0.0, 0.0}, sqrt_s);
  const LorentzRotation to_lab = LorentzRotation::from_boost(p.boost_vector());
  return to_lab * p4_final;
}

}  // namespace g4gpu::bic::imr

#endif
