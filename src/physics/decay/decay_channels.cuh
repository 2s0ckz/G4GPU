// The G4VDecayChannel subclasses the decay tables of QBBC's transported unstable species use,
// each transcribed from its own 11.1.1 source file.
//
//   G4PhaseSpaceDecayChannel    pi+, pi-, pi0 -> gamma gamma, and four of K+/K-'s six channels
//   G4MuonDecayChannel          mu+, mu-  (the plain one - see below)
//   G4KL3DecayChannel           K+/K- -> pi0 l nu, both Ke3 and Kmu3
//   G4DalitzDecayChannel        pi0 -> gamma e+ e-
//   G4NeutronBetaDecayChannel   neutron -> e- anti_nu_e p
//
// WHICH MUON CHANNEL. `G4MuonPlus.cc` and `G4MuonMinus.cc` in 11.1.1 both do exactly
//
//     G4VDecayChannel* mode = new G4MuonDecayChannel("mu+",1.00);
//
// so it is the PLAIN channel: no spin correlation and no radiative mode.
// G4MuonDecayChannelWithSpin and G4MuonRadiativeDecayChannelWithSpin exist in the release and
// nothing in QBBC's chain installs either - `G4DecayPhysics::ConstructProcess` builds one
// shared `G4Decay` and registers it against every applicable particle without touching a
// decay table. (`G4DecayWithSpin` is a different process, and QBBC does not register it.) The
// consequence is physical and worth stating: a decaying muon's electron is emitted isotropically
// in the muon rest frame, uncorrelated with the muon's spin, and there is no mu -> e nu nu gamma.
//
// EVERY SAMPLER DRAWS THE SAME RANDOM NUMBERS IN THE SAME ORDER as the Geant4 function it
// comes from. The port's RNG is Philox and Geant4's is MixMax, so no comparison can be
// stream-exact - but keeping the order means a rejection loop's acceptance rate, a
// distribution's shape and the number of draws per decay all match, and a transposed pair of
// draws inside a loop shows up as a shifted distribution rather than as nothing at all.
//
// MAX_LOOP. Every rejection loop in these classes is bounded (1000 in the muon channel, 10000
// elsewhere) and every one FALLS THROUGH with whatever value it last computed rather than
// failing. That is Geant4's behaviour, it is reproduced exactly, and it is the reason the
// muon sampler's inner loop assigns `x = xmax` on a rejection: the last rejected value is the
// answer if the loop runs out. None of these bounds is reached in practice - the worst
// acceptance is the muon inner loop's 25% - but a transcription that "cleaned up" the
// fall-through would be a different sampler at the tail.
#pragma once
#include <cmath>

#include "core/units.cuh"
#include "physics/decay/decay_products.cuh"
#include "physics/decay/decay_tables.hh"

namespace g4gpu::decay {

/// G4PhaseSpaceDecayChannel::Pmx - the momentum of either daughter of a two-body decay of
/// mass `e` into `p1` and `p2`. Returns -1 rather than a NaN when the decay is closed, and
/// the callers test for that.
template <typename real_t>
__host__ __device__ inline real_t pmx(real_t e, real_t p1, real_t p2) {
  const real_t ppp = (e + p1 + p2) * (e + p1 - p2) * (e - p1 + p2) * (e - p1 - p2) /
                     (real_t(4) * e * e);
  return (ppp > real_t(0)) ? sqrt(ppp) : real_t(-1);
}

/// G4VDecayChannel::rangeMass - the default, and nothing in the ported tables changes it.
/// It is how many daughter widths below the sum of daughter masses a parent mass may be and
/// still open the channel, and it also bounds the Breit-Wigner in DynamicalMass.
__host__ __device__ inline constexpr double range_mass() { return 2.5; }

/// G4VDecayChannel::IsOKWithParentMass. A one-body channel is always open; otherwise the
/// parent must reach the sum of (daughter mass - 2.5 * daughter width).
///
/// For every channel in decay_tables.hh the width term is negligible (the largest daughter
/// width is pi0's 7.73e-6 MeV) so this reduces to the sum of masses - but it is the widths
/// that make the test a >= rather than a > and they are kept.
__host__ __device__ inline bool channel_ok_with_parent_mass(const ChannelRow& ch,
                                                            double parent_mass) {
  if (ch.n_daughters == 1) { return true; }
  double sum_min = 0.0;
  for (int i = 0; i < ch.n_daughters; ++i) {
    sum_min += particle_mass(ch.daughter[i]) - range_mass() * particle_width(ch.daughter[i]);
  }
  return parent_mass >= sum_min;
}

/// The largest kinetic energy daughter `d` can carry away, MeV - the channel's kinematic
/// limit for that daughter.
///
/// It is the two-body formula with the OTHER daughters lumped into one system of mass equal
/// to the sum of theirs, which is exact for any number of daughters: a daughter's energy is
/// largest when everything else recoils together, and the lightest that "everything else" can
/// be is the sum of its rest masses. So one expression covers the two-body case (where it is
/// the only value the daughter can have, min == max) and the three-body case (where it is the
/// top of a continuum starting at zero).
///
/// Geant4 does not have this function - it is the analytic bound on what Geant4's samplers
/// produce, and the oracle dumps it from G4PhaseSpaceDecayChannel::Pmx so that the port's
/// version is checked rather than trusted. The sampled maxima are then required to approach
/// it from below, which is what catches a sampler that shares the released energy wrongly.
template <typename real_t>
__host__ __device__ inline real_t channel_slot_max_kinetic_energy(const ChannelRow& ch,
                                                                  real_t parent_mass, int d) {
  const real_t m = static_cast<real_t>(particle_mass(ch.daughter[d]));
  real_t rest = real_t(0);
  for (int i = 0; i < ch.n_daughters; ++i) {
    if (i != d) { rest += static_cast<real_t>(particle_mass(ch.daughter[i])); }
  }
  const real_t p = pmx(parent_mass, m, rest);
  if (p <= real_t(0)) { return real_t(0); }
  return sqrt(p * p + m * m) - m;
}

/// True when G4PhaseSpaceDecayChannel would take its `withWidth` branch and resample a
/// daughter mass from a Breit-Wigner through G4VDecayChannel::DynamicalMass. No daughter of
/// any ported table reaches it; the samplers refuse rather than approximate if one ever does.
__host__ __device__ inline bool channel_needs_dynamical_mass(const ChannelRow& ch) {
  for (int i = 0; i < ch.n_daughters; ++i) {
    const double m = particle_mass(ch.daughter[i]);
    const double w = particle_width(ch.daughter[i]);
    if (w > 1.0e-3 * m) { return true; }
  }
  return false;
}

// ---------------------------------------------------------------------------------------
// G4PhaseSpaceDecayChannel
// ---------------------------------------------------------------------------------------

/// G4PhaseSpaceDecayChannel::OneBodyDecayIt - the daughter is at rest in the parent frame.
/// Unreachable from any ported table (no channel has one daughter); here because DecayIt's
/// switch has the case and leaving it out would make `n_daughters == 1` fall into the
/// two-body branch and read a daughter that does not exist.
///
/// One deliberate departure, and it is the only one in this file. Geant4 builds the daughter
/// as `new G4DynamicParticle(G4MT_daughters[0], dummy, 0.0)` with `dummy` a DEFAULT-
/// CONSTRUCTED G4ThreeVector, so the product's momentum direction is the ZERO vector, not a
/// unit vector. This buffer's contract says directions are unit vectors and every consumer
/// of it may divide by one, so the direction is +z here. It is observable only through a
/// direction that carries no momentum, and taking Geant4's zero would put a non-unit vector
/// into a field nothing else in the port allows to be non-unit.
///
/// `out.parent_mass` is NOT set here, nor in any other sampler in this file: it is the mass
/// G4DecayProducts stores for its parent, which is the SNAPPED dynamical mass and not the
/// mass the kinematics use, and only `channel_decay_it` knows the parent's PDG code to snap
/// against. See `snapped_dynamical_mass`.
template <typename real_t>
__host__ __device__ inline void phase_space_one_body(const ChannelRow& ch, real_t parent_mass,
                                                     DecayProducts<real_t>& out) {
  (void)parent_mass;
  out.push(ch.daughter[0], 0, static_cast<real_t>(particle_mass(ch.daughter[0])), real_t(0),
           real_t(0), real_t(0), real_t(1));
}

/// G4PhaseSpaceDecayChannel::TwoBodyDecayIt.
///
/// Random draws, in order: costheta, phi. The daughters are back to back, the first along
/// +direction and the second along -direction, and each is built through
/// `G4DynamicParticle(def, direction, Ekin, mass)` from `Ekin = sqrt(p^2+m^2) - m` - so the
/// kinetic energy is the stored quantity and the momentum is recomputed from it, which is
/// what DecayProduct does.
template <typename real_t, typename rng_t>
__host__ __device__ inline void phase_space_two_body(const ChannelRow& ch, real_t parent_mass,
                                                    rng_t& rng, DecayProducts<real_t>& out) {
  const real_t m0 = static_cast<real_t>(particle_mass(ch.daughter[0]));
  const real_t m1 = static_cast<real_t>(particle_mass(ch.daughter[1]));
  if (parent_mass < m0 + m1) {
    // G4Exception PART112, JustWarning, and `return products` with nothing in it. The parent
    // is then killed with no secondaries and its kinetic energy deposited locally. Reported
    // rather than silently produced.
    out.fail(DecayStatus::kDaughterMassTooLarge);
    return;
  }
  const real_t p = pmx(parent_mass, m0, m1);
  const real_t costheta = real_t(2) * rng.uniform() - real_t(1);
  const real_t sintheta = sqrt((real_t(1) - costheta) * (real_t(1) + costheta));
  const real_t phi = units::twopi<real_t>() * rng.uniform();
  const real_t dx = sintheta * cos(phi);
  const real_t dy = sintheta * sin(phi);
  const real_t dz = costheta;
  out.push(ch.daughter[0], 0, m0, sqrt(p * p + m0 * m0) - m0, dx, dy, dz);
  out.push(ch.daughter[1], 1, m1, sqrt(p * p + m1 * m1) - m1, -dx, -dy, -dz);
}

/// G4PhaseSpaceDecayChannel::ThreeBodyDecayIt - "originally written in GDECA3 of GEANT3".
///
/// Random draws, in order: rd1, rd2 (repeated until the three momenta close a triangle), then
/// costheta, phi for daughter 0's direction, then phin for the azimuth of daughter 2 about it.
///
/// The energies are shared as (rd2, 1-rd1, rd1-rd2) of the released energy with rd1 >= rd2,
/// which is flat in the Dalitz plot; the loop rejects the shares whose momenta cannot form a
/// closed triangle. Daughter 2's polar angle relative to daughter 0 is then FIXED by momentum
/// conservation (the cosine on line 487 of the Geant4 file), only its azimuth is free, and
/// daughter 1 is minus the sum of the other two. That is why the products come out in the
/// order 0, 2, 1 - and the port pushes them in that order too.
template <typename real_t, typename rng_t>
__host__ __device__ inline void phase_space_three_body(const ChannelRow& ch,
                                                       real_t parent_mass, rng_t& rng,
                                                       DecayProducts<real_t>& out) {
  real_t dm[3];
  real_t sum_dm = real_t(0);
  for (int i = 0; i < 3; ++i) {
    dm[i] = static_cast<real_t>(particle_mass(ch.daughter[i]));
    sum_dm += dm[i];
  }
  if (sum_dm > parent_mass) {
    out.fail(DecayStatus::kDaughterMassTooLarge);
    return;
  }

  real_t dp[3] = {real_t(0), real_t(0), real_t(0)};
  const int kMaxLoop = 10000;
  for (int loop = 0; loop < kMaxLoop; ++loop) {
    real_t rd1 = rng.uniform();
    real_t rd2 = rng.uniform();
    if (rd2 > rd1) {
      const real_t rd = rd1;
      rd1 = rd2;
      rd2 = rd;
    }
    real_t pmax = real_t(0);
    real_t psum = real_t(0);
    real_t energy = rd2 * (parent_mass - sum_dm);
    dp[0] = sqrt(energy * energy + real_t(2) * energy * dm[0]);
    if (dp[0] > pmax) { pmax = dp[0]; }
    psum += dp[0];
    energy = (real_t(1) - rd1) * (parent_mass - sum_dm);
    dp[1] = sqrt(energy * energy + real_t(2) * energy * dm[1]);
    if (dp[1] > pmax) { pmax = dp[1]; }
    psum += dp[1];
    energy = (rd1 - rd2) * (parent_mass - sum_dm);
    dp[2] = sqrt(energy * energy + real_t(2) * energy * dm[2]);
    if (dp[2] > pmax) { pmax = dp[2]; }
    psum += dp[2];
    if (pmax <= psum - pmax) { break; }
  }

  const real_t costheta = real_t(2) * rng.uniform() - real_t(1);
  const real_t sintheta = sqrt((real_t(1) - costheta) * (real_t(1) + costheta));
  const real_t phi = units::twopi<real_t>() * rng.uniform();
  const real_t sinphi = sin(phi);
  const real_t cosphi = cos(phi);

  const real_t d0x = sintheta * cosphi;
  const real_t d0y = sintheta * sinphi;
  const real_t d0z = costheta;
  out.push(ch.daughter[0], 0, dm[0], sqrt(dp[0] * dp[0] + dm[0] * dm[0]) - dm[0], d0x, d0y,
           d0z);

  const real_t costhetan =
      (dp[1] * dp[1] - dp[2] * dp[2] - dp[0] * dp[0]) / (real_t(2) * dp[2] * dp[0]);
  const real_t sinthetan = sqrt((real_t(1) - costhetan) * (real_t(1) + costhetan));
  const real_t phin = units::twopi<real_t>() * rng.uniform();
  const real_t sinphin = sin(phin);
  const real_t cosphin = cos(phin);
  real_t d2x = sinthetan * cosphin * costheta * cosphi - sinthetan * sinphin * sinphi +
               costhetan * sintheta * cosphi;
  real_t d2y = sinthetan * cosphin * costheta * sinphi + sinthetan * sinphin * cosphi +
               costhetan * sintheta * sinphi;
  real_t d2z = -sinthetan * cosphin * sintheta + costhetan * costheta;
  // Geant4 divides by direction2.mag() twice - once to build pmom and once again to take its
  // unit vector - because the rotated vector is only a unit vector up to rounding. Kept, so
  // that the same cancellation happens here.
  const real_t d2mag = sqrt(d2x * d2x + d2y * d2y + d2z * d2z);
  const real_t p2x = dp[2] * d2x / d2mag;
  const real_t p2y = dp[2] * d2y / d2mag;
  const real_t p2z = dp[2] * d2z / d2mag;
  const real_t p2m = sqrt(p2x * p2x + p2y * p2y + p2z * p2z);
  out.push(ch.daughter[2], 2, dm[2], sqrt(p2m * p2m + dm[2] * dm[2]) - dm[2], p2x / p2m,
           p2y / p2m, p2z / p2m);

  const real_t p1x = -(d0x * dp[0] + d2x * (dp[2] / d2mag));
  const real_t p1y = -(d0y * dp[0] + d2y * (dp[2] / d2mag));
  const real_t p1z = -(d0z * dp[0] + d2z * (dp[2] / d2mag));
  const real_t p1m = sqrt(p1x * p1x + p1y * p1y + p1z * p1z);
  out.push(ch.daughter[1], 1, dm[1], sqrt(p1m * p1m + dm[1] * dm[1]) - dm[1], p1x / p1m,
           p1y / p1m, p1z / p1m);
}

/// G4PhaseSpaceDecayChannel::ManyBodyDecayIt - "originally written in FORTRAN by M.Asai",
/// NBODY, 19/Apr/1995. Four or five daughters.
///
/// No decay table of any species QBBC transports reaches it: the widest channel in
/// decay_tables.hh has three daughters. It is transcribed anyway because it is the `default:`
/// arm of G4PhaseSpaceDecayChannel::DecayIt and leaving it out would make an added four-body
/// row silently wrong rather than loudly missing, and because it is testable without a real
/// table - the oracle constructs a synthetic four-body G4PhaseSpaceDecayChannel and this
/// samples the same one.
///
/// Two things in it that look like mistakes and are not to be corrected:
///
///   The acceptance is `if (weight < G4UniformRand()) break;` - the loop exits when the
///   weight is BELOW the uniform, which is the opposite sense to a textbook rejection. It is
///   what the code does, and since the weight is a product of p_i/sm_i factors that are all
///   below one, the effect is a mild bias towards low-weight configurations rather than
///   nonsense.
///
///   `daughtermomentum[N-1]` is computed as Pmx(sm[N-2], m[N-2], sm[N-1]) and then never
///   used: the last particle's momentum is minus the second-to-last's. The line is kept
///   because it draws no randoms and removing it would be a silent divergence if a later
///   Geant4 starts using the value.
template <typename real_t, typename rng_t>
__host__ __device__ inline void phase_space_many_body(const ChannelRow& ch, real_t parent_mass,
                                                      rng_t& rng,
                                                      DecayProducts<real_t>& out) {
  const int nd = ch.n_daughters;
  real_t dm[DecayProducts<real_t>::kMaxProducts];
  real_t sum_dm = real_t(0);
  for (int i = 0; i < nd; ++i) {
    dm[i] = static_cast<real_t>(particle_mass(ch.daughter[i]));
    sum_dm += dm[i];
  }
  if (sum_dm > parent_mass) {
    out.fail(DecayStatus::kDaughterMassTooLarge);
    return;
  }

  real_t dp[DecayProducts<real_t>::kMaxProducts];
  real_t sm[DecayProducts<real_t>::kMaxProducts];
  real_t rd[DecayProducts<real_t>::kMaxProducts];
  int number_of_try = 0;
  const int kMaxLoop = 10000;
  for (int loop = 0; loop < kMaxLoop; ++loop) {
    rd[0] = real_t(1);
    for (int i = 1; i < nd - 1; ++i) { rd[i] = rng.uniform(); }
    rd[nd - 1] = real_t(0);
    for (int i = 1; i < nd - 1; ++i) {
      for (int j = i + 1; j < nd; ++j) {
        if (rd[i] < rd[j]) {
          const real_t t = rd[i];
          rd[i] = rd[j];
          rd[j] = t;
        }
      }
    }
    real_t tmas = parent_mass - sum_dm;
    real_t temp = sum_dm;
    for (int i = 0; i < nd; ++i) {
      sm[i] = rd[i] * tmas + temp;
      temp -= dm[i];
    }

    real_t weight = real_t(1);
    bool sm_ok = true;
    for (int i = 0; i < nd - 1 && sm_ok; ++i) {
      sm_ok = (sm[i] - dm[i] - sm[i + 1] >= real_t(0));
    }
    if (!sm_ok) { continue; }

    dp[nd - 1] = pmx(sm[nd - 2], dm[nd - 2], sm[nd - 1]);
    bool illegal = false;
    for (int i = nd - 2; i >= 0; --i) {
      dp[i] = pmx(sm[i], dm[i], sm[i + 1]);
      if (dp[i] < real_t(0)) {
        illegal = true;
        break;
      }
      weight *= dp[i] / sm[i];
    }
    if (illegal) {
      // G4Exception PART112 and `return nullptr` - not an empty product list, a null one.
      out.fail(DecayStatus::kDaughterMassTooLarge);
      return;
    }
    if (++number_of_try > 100) {
      // G4Exception PART113, `return nullptr`: "Decay Kinematics cannot be calculated".
      out.fail(DecayStatus::kKinematicsFailed);
      return;
    }
    if (weight < rng.uniform()) { break; }
  }

  // The products are built IN PLACE in the caller's buffer, in daughter order, because that
  // is the order Geant4 finally pushes them in and because the boost has to go through
  // G4DynamicParticle's own Get4Momentum / Set4Momentum on each pass: Geant4 creates each
  // daughter from a three-momentum (which forces it onto the PDG mass shell) and then boosts
  // the already-created ones through those two accessors, so each boost round-trips the state
  // through (direction, kinetic energy, mass) and Set4Momentum's dynamical-mass rule applies.
  //
  // THIS CHANNEL DOES NOT CLOSE ITS FOUR-MOMENTUM to double precision and the reason is not
  // the round trip. `beta = dp[i]/sqrt(dp[i]^2 + sm[i+1]^2)` is within 1e-11 of one whenever
  // the subsystem being boosted is nearly massless - two photons, say - and at that point
  // CLHEP's `1/sqrt(1-b2)` has lost most of its significant digits to the cancellation in
  // `1-b2`. The residual is 5.9e-7 of the parent mass for the four-body channel here; Geant4
  // measures 2.9e-7 for the same channel (ref/oracle/decay_closure.csv), and the first version
  // of this function, which carried the four-vectors in locals and forced them back onto the
  // mass shell at the end, measured the same 4.9e-7 - so that restructure is a faithfulness
  // fix and not a precision one, and it is recorded here as such rather than credited with a
  // number it did not move.
  out.n = nd;
  for (int i = 0; i < nd; ++i) {
    DecayProduct<real_t>& q = out.p[i];
    q.pdg = ch.daughter[i];
    q.daughter = i;
    q.mass = dm[i];
    q.ekin = real_t(0);
  }

  {
    const int i = nd - 2;
    const real_t costheta = real_t(2) * rng.uniform() - real_t(1);
    const real_t sintheta = sqrt((real_t(1) - costheta) * (real_t(1) + costheta));
    const real_t phi = units::twopi<real_t>() * rng.uniform();
    const real_t dz = costheta;
    const real_t dy = sintheta * sin(phi);
    const real_t dx = sintheta * cos(phi);
    out.p[i].set_momentum(dx * dp[i], dy * dp[i], dz * dp[i]);
    out.p[i + 1].set_momentum(-dx * dp[i], -dy * dp[i], -dz * dp[i]);
  }

  for (int i = nd - 3; i >= 0; --i) {
    const real_t costheta = real_t(2) * rng.uniform() - real_t(1);
    const real_t sintheta = sqrt((real_t(1) - costheta) * (real_t(1) + costheta));
    const real_t phi = units::twopi<real_t>() * rng.uniform();
    const real_t dz = costheta;
    const real_t dy = sintheta * sin(phi);
    const real_t dx = sintheta * cos(phi);
    real_t beta = dp[i];
    beta /= sqrt(dp[i] * dp[i] + sm[i + 1] * sm[i + 1]);
    for (int j = i + 1; j < nd; ++j) {
      FourVector<real_t> p4 = out.p[j].four_momentum();
      lorentz_boost(p4, dx * beta, dy * beta, dz * beta);
      out.p[j].set_four_momentum(p4);
    }
    out.p[i].set_momentum(-dx * dp[i], -dy * dp[i], -dz * dp[i]);
  }
}

// ---------------------------------------------------------------------------------------
// G4MuonDecayChannel
// ---------------------------------------------------------------------------------------

/// G4MuonDecayChannel::DecayIt. "This version neglects muon polarization, and electron mass;
/// assumes the pure V-A coupling; the Neutrinos are correctly V-A."
///
/// Note the signature: Geant4's takes a parent mass and IGNORES IT, using
/// `G4MT_parent->GetPDGMass()` instead. So a muon whose dynamic mass has drifted decays with
/// its PDG mass. The port takes no parent mass for the same reason - passing one would imply
/// it was used.
///
/// The sampling, and why the marginal electron spectrum is Michel's:
///
///   `Ee` is drawn UNIFORM on (0,1) and `Ene` from a density proportional to x(1-x) on
///   (0, xmax) by rejection, and the pair is accepted only if Ee + Ene >= 1. Integrating the
///   accepted joint density over Ene gives, for x = Ee,
///
///       p(x) ∝ ∫_{1-x}^{1} y(1-y) dy = x^2/2 - x^3/3 = (x^2/6)(3 - 2x)
///
///   which is exactly the massless Michel spectrum x^2(3-2x). That identity is the second,
///   independent oracle for this channel in tests/test_decay.cu: it does not come from
///   Geant4 at all, so agreement with it cannot be an agreement between two copies of the
///   same mistake.
///
///   xmax is 1 + (m_e/m_mu)^2, slightly ABOVE one, so x can be drawn where x(1-x) is negative
///   and is then always rejected. Kept, because the width of the sampled interval sets the
///   acceptance rate.
///
///   The electron's momentum is sqrt(Ee^2 EMax^2 + 2 Ee EMax m_e) with
///   EMax = m_mu/2 - m_e: the momentum of a particle of kinetic energy Ee*EMax. So Ee is a
///   reduced KINETIC energy, not a reduced total energy, and the endpoint is m_mu/2 - m_e.
///
///   The three momenta close exactly despite the electron mass being neglected in the angles:
///   with costheta = 1 - 2/Ee - 2/Ene + 2/(Ee Ene), the third direction vector
///   (-Ene/Enm sinθ, 0, -Ee/Enm - Ene/Enm cosθ) has unit length identically, because
///   Ee^2 + Ene^2 + 2 Ee Ene cosθ = (2 - Ee - Ene)^2 = Enm^2.
template <typename real_t, typename rng_t>
__host__ __device__ inline void muon_decay(const ChannelRow& ch, int parent_pdg, rng_t& rng,
                                           DecayProducts<real_t>& out) {
  const real_t parent_mass = static_cast<real_t>(particle_mass(parent_pdg));
  real_t dm[3];
  for (int i = 0; i < 3; ++i) { dm[i] = static_cast<real_t>(particle_mass(ch.daughter[i])); }

  const real_t xmax = real_t(1) + dm[0] * dm[0] / parent_mass / parent_mass;
  const real_t emax = parent_mass / real_t(2) - dm[0];

  real_t ee = real_t(0);
  real_t ene = real_t(0);
  const int kMaxLoop = 1000;
  for (int loop1 = 0; loop1 < kMaxLoop; ++loop1) {
    ee = rng.uniform();
    real_t x = real_t(0);
    for (int loop2 = 0; loop2 < kMaxLoop; ++loop2) {
      x = xmax * rng.uniform();
      const real_t gam = rng.uniform();
      if (gam <= x * (real_t(1) - x)) { break; }
      x = xmax;
    }
    ene = x;
    if (ene >= (real_t(1) - ee)) { break; }
    ene = real_t(1) - ee;
  }
  const real_t enm = real_t(2) - ee - ene;

  const real_t costheta =
      real_t(1) - real_t(2) / ee - real_t(2) / ene + real_t(2) / ene / ee;
  const real_t sintheta = sqrt(real_t(1) - costheta * costheta);

  const real_t rphi = units::twopi<real_t>() * rng.uniform();
  const real_t rtheta = acos(real_t(2) * rng.uniform() - real_t(1));
  const real_t rpsi = units::twopi<real_t>() * rng.uniform();

  // G4RotationMatrix::set(phi, theta, psi), CLHEP RotationE.cc:32 - the Euler-angle form.
  // `direction *= rot` is `direction = rot * direction` (CLHEP ThreeVectorR.cc:16), so these
  // are the matrix rows.
  const real_t sp = sin(rphi), cp = cos(rphi);
  const real_t st = sin(rtheta), ct = cos(rtheta);
  const real_t ss = sin(rpsi), cs = cos(rpsi);
  const real_t rxx = cs * cp - ct * sp * ss;
  const real_t rxy = cs * sp + ct * cp * ss;
  const real_t rxz = ss * st;
  const real_t ryx = -ss * cp - ct * sp * cs;
  const real_t ryy = -ss * sp + ct * cp * cs;
  const real_t ryz = cs * st;
  const real_t rzx = st * sp;
  const real_t rzy = -st * cp;
  const real_t rzz = ct;

  // daughter 0: the electron, along +z before the rotation
  {
    const real_t p = sqrt(ee * ee * emax * emax + real_t(2) * ee * emax * dm[0]);
    const real_t vx = real_t(0), vy = real_t(0), vz = real_t(1);
    const real_t dx = rxx * vx + rxy * vy + rxz * vz;
    const real_t dy = ryx * vx + ryy * vy + ryz * vz;
    const real_t dz = rzx * vx + rzy * vy + rzz * vz;
    out.push_momentum(ch.daughter[0], 0, dx * p, dy * p, dz * p);
  }
  // daughter 1: nu_e (mu+) / anti_nu_e (mu-), at costheta from the electron
  {
    const real_t p = sqrt(ene * ene * emax * emax + real_t(2) * ene * emax * dm[1]);
    const real_t vx = sintheta, vy = real_t(0), vz = costheta;
    const real_t dx = rxx * vx + rxy * vy + rxz * vz;
    const real_t dy = ryx * vx + ryy * vy + ryz * vz;
    const real_t dz = rzx * vx + rzy * vy + rzz * vz;
    out.push_momentum(ch.daughter[1], 1, dx * p, dy * p, dz * p);
  }
  // daughter 2: anti_nu_mu (mu+) / nu_mu (mu-), closing the momentum
  {
    const real_t p = sqrt(enm * enm * emax * emax + real_t(2) * enm * emax * dm[2]);
    const real_t vx = -ene / enm * sintheta;
    const real_t vy = real_t(0);
    const real_t vz = -ee / enm - ene / enm * costheta;
    const real_t dx = rxx * vx + rxy * vy + rxz * vz;
    const real_t dy = ryx * vx + ryy * vy + ryz * vz;
    const real_t dz = rzx * vx + rzy * vy + rzz * vz;
    out.push_momentum(ch.daughter[2], 2, dx * p, dy * p, dz * p);
  }
}

// ---------------------------------------------------------------------------------------
// G4KL3DecayChannel
// ---------------------------------------------------------------------------------------

/// G4KL3DecayChannel::PhaseSpace - the same GDECA3 energy sharing as
/// G4PhaseSpaceDecayChannel::ThreeBodyDecayIt, but it also returns the kinetic energies
/// because DalitzDensity needs them. `E` comes back as KINETIC energy and `P` as momentum.
template <typename real_t, typename rng_t>
__host__ __device__ inline void kl3_phase_space(real_t parent_mass, const real_t m[3],
                                                rng_t& rng, real_t e[3], real_t p[3]) {
  real_t sum_dm = real_t(0);
  for (int i = 0; i < 3; ++i) { sum_dm += m[i]; }
  const int kMaxLoop = 10000;
  for (int loop = 0; loop < kMaxLoop; ++loop) {
    real_t rd1 = rng.uniform();
    real_t rd2 = rng.uniform();
    if (rd2 > rd1) {
      const real_t rd = rd1;
      rd1 = rd2;
      rd2 = rd;
    }
    real_t pmax = real_t(0);
    real_t psum = real_t(0);
    real_t energy = rd2 * (parent_mass - sum_dm);
    p[0] = sqrt(energy * energy + real_t(2) * energy * m[0]);
    e[0] = energy;
    if (p[0] > pmax) { pmax = p[0]; }
    psum += p[0];
    energy = (real_t(1) - rd1) * (parent_mass - sum_dm);
    p[1] = sqrt(energy * energy + real_t(2) * energy * m[1]);
    e[1] = energy;
    if (p[1] > pmax) { pmax = p[1]; }
    psum += p[1];
    energy = (rd1 - rd2) * (parent_mass - sum_dm);
    p[2] = sqrt(energy * energy + real_t(2) * energy * m[2]);
    e[2] = energy;
    if (p[2] > pmax) { pmax = p[2]; }
    psum += p[2];
    if (pmax <= psum - pmax) { break; }
  }
}

/// G4KL3DecayChannel::DalitzDensity - "KL3 decay Dalitz Plot Density, see Chounet et al
/// Phys. Rep. 4, 201". Returns Rho/RhoMax, which is compared against a uniform.
///
/// Arguments arrive as KINETIC energies and the first three lines add the masses, so the body
/// works in total energies. pLambda is the linear energy dependence of f+ and pXi0 is
/// f+(0)/f-; both come from the channel row, chosen by (parent, lepton) in the constructor.
///
/// RhoMax is `Fmax^2 * massK^3/8` with `Fmax = 1 + pLambda*(massK^2/massPi^2 + 1)` when
/// pLambda > 0. It is an ENVELOPE, not the true maximum of Rho: Rho/RhoMax comes out around
/// 1e-2 over the physical region, so the loop accepts about one draw in fifty for Ke3. That
/// inefficiency is Geant4's and is reproduced, because changing the envelope changes the
/// number of draws per decay and nothing else.
template <typename real_t>
__host__ __device__ inline real_t kl3_dalitz_density(real_t mass_k, real_t epi, real_t el,
                                                     real_t enu, real_t mass_pi, real_t mass_l,
                                                     real_t mass_nu, real_t p_lambda,
                                                     real_t p_xi0) {
  epi = epi + mass_pi;
  el = el + mass_l;
  enu = enu + mass_nu;

  const real_t epi_max =
      (mass_k * mass_k + mass_pi * mass_pi - mass_l * mass_l) / real_t(2) / mass_k;
  const real_t e = epi_max - epi;
  const real_t q2 = mass_k * mass_k + mass_pi * mass_pi - real_t(2) * mass_k * epi;

  const real_t f = real_t(1) + p_lambda * q2 / mass_pi / mass_pi;
  real_t fmax = real_t(1);
  if (p_lambda > real_t(0)) {
    fmax = real_t(1) + p_lambda * (mass_k * mass_k / mass_pi / mass_pi + real_t(1));
  }
  const real_t xi = p_xi0 * (real_t(1) + p_lambda * q2 / mass_pi / mass_pi);

  const real_t coeff_a = mass_k * (real_t(2) * el * enu - mass_k * e) +
                         mass_l * mass_l * (e / real_t(4) - enu);
  const real_t coeff_b = mass_l * mass_l * (enu - e / real_t(2));
  const real_t coeff_c = mass_l * mass_l * e / real_t(4);

  const real_t rho_max = (fmax * fmax) * (mass_k * mass_k * mass_k / real_t(8));
  const real_t rho = (f * f) * (coeff_a + coeff_b * xi + coeff_c * xi * xi);
  return rho / rho_max;
}

/// G4KL3DecayChannel::DecayIt. "This version neglects muon polarization, assumes the pure V-A
/// coupling, gives incorrect energy spectrum for Neutrinos."
///
/// Random draws, in order: r (the acceptance uniform, drawn BEFORE the phase space), then
/// PhaseSpace's own draws, then - once accepted - costheta and phi for the pion and phin for
/// the neutrino's azimuth. Drawing r first matters: it fixes how many uniforms a rejected
/// configuration consumes.
///
/// Like the three-body phase space, the pion's direction is free, the neutrino's polar angle
/// relative to it is fixed by momentum conservation and only its azimuth is free, and the
/// LEPTON is minus the sum of the other two - so its momentum is not P[1] from PhaseSpace at
/// all. Push order is therefore pion, neutrino, lepton: daughters 0, 2, 1.
///
/// The parent mass argument is ignored here too, exactly as in the muon channel: DecayIt
/// takes `G4double` and reads `G4MT_parent->GetPDGMass()`.
template <typename real_t, typename rng_t>
__host__ __device__ inline void kl3_decay(const ChannelRow& ch, int parent_pdg, rng_t& rng,
                                          DecayProducts<real_t>& out) {
  const real_t mass_k = static_cast<real_t>(particle_mass(parent_pdg));
  real_t m[3];
  for (int i = 0; i < 3; ++i) { m[i] = static_cast<real_t>(particle_mass(ch.daughter[i])); }

  real_t e[3] = {real_t(0), real_t(0), real_t(0)};
  real_t p[3] = {real_t(0), real_t(0), real_t(0)};
  const int kMaxLoop = 10000;
  for (int loop = 0; loop < kMaxLoop; ++loop) {
    const real_t r = rng.uniform();
    kl3_phase_space(mass_k, m, rng, e, p);
    const real_t w = kl3_dalitz_density<real_t>(mass_k, e[0], e[1], e[2], m[0], m[1], m[2],
                                                static_cast<real_t>(ch.kl3_lambda),
                                                static_cast<real_t>(ch.kl3_xi0));
    if (r <= w) { break; }
  }

  const real_t costheta = real_t(2) * rng.uniform() - real_t(1);
  const real_t sintheta = sqrt((real_t(1) - costheta) * (real_t(1) + costheta));
  const real_t phi = units::twopi<real_t>() * rng.uniform();
  const real_t sinphi = sin(phi);
  const real_t cosphi = cos(phi);
  const real_t m0x = sintheta * cosphi * p[0];
  const real_t m0y = sintheta * sinphi * p[0];
  const real_t m0z = costheta * p[0];
  out.push_momentum(ch.daughter[0], 0, m0x, m0y, m0z);

  const real_t costhetan = (p[1] * p[1] - p[2] * p[2] - p[0] * p[0]) / (real_t(2) * p[2] * p[0]);
  const real_t sinthetan = sqrt((real_t(1) - costhetan) * (real_t(1) + costhetan));
  const real_t phin = units::twopi<real_t>() * rng.uniform();
  const real_t sinphin = sin(phin);
  const real_t cosphin = cos(phin);
  const real_t d2x = sinthetan * cosphin * costheta * cosphi - sinthetan * sinphin * sinphi +
                     costhetan * sintheta * cosphi;
  const real_t d2y = sinthetan * cosphin * costheta * sinphi + sinthetan * sinphin * cosphi +
                     costhetan * sintheta * sinphi;
  const real_t d2z = -sinthetan * cosphin * sintheta + costhetan * costheta;
  // Not normalised: Geant4 multiplies the rotated direction by P[2] directly here, unlike
  // ThreeBodyDecayIt which divides by its magnitude. The vector is a unit vector up to
  // rounding, so the difference is in the last bits - but it is a different expression and it
  // is kept as the different expression.
  const real_t m2x = d2x * p[2];
  const real_t m2y = d2y * p[2];
  const real_t m2z = d2z * p[2];
  out.push_momentum(ch.daughter[2], 2, m2x, m2y, m2z);

  out.push_momentum(ch.daughter[1], 1, -(m0x + m2x), -(m0y + m2y), -(m0z + m2z));
}

// ---------------------------------------------------------------------------------------
// G4DalitzDecayChannel
// ---------------------------------------------------------------------------------------

/// G4DalitzDecayChannel::DecayIt - pi0 -> gamma e+ e-, through a virtual photon of mass^2 = t.
///
/// Random draws, in order: (x, w) until accepted, then costheta and phi for the gamma, then
/// costheta and phi for the lepton in the (l+ l-) rest frame.
///
/// t is sampled log-uniformly between (2 m_l)^2 and m_pi0^2 - `x = 2 log(2 m_l) .. 2 log M`
/// with t = exp(x) - against the Kroll-Wada shape
///
///     (1 - 4 m^2/t) ^ 1/2  *  (1 + 2 m^2/t)  *  (1 - t/M^2)^3
///
/// with envelope wmax = 1.5. Note the envelope is above the shape's maximum of 1 at t -> 0,
/// so nothing is clipped; the acceptance is about 20%.
///
/// The gamma is built from `G4DynamicParticle(def, gdirection, Pgamma)` - the (direction,
/// KINETIC energy) constructor - so its kinetic energy IS Pgamma and its mass is zero, which
/// is right for a photon and is the reason no separate massless case is needed.
///
/// The leptons are made back to back in the (l+ l-) rest frame and boosted by
/// beta = Pgamma/(M - Pgamma) along -gdirection: the pair's energy is M - Pgamma and its
/// momentum is Pgamma against the photon. They go through Get4Momentum / boost /
/// Set4Momentum, so their masses pass through Set4Momentum's dynamical-mass rule.
template <typename real_t, typename rng_t>
__host__ __device__ inline void dalitz_decay(const ChannelRow& ch, int parent_pdg, rng_t& rng,
                                             DecayProducts<real_t>& out) {
  const real_t parent_mass = static_cast<real_t>(particle_mass(parent_pdg));
  const real_t lepton_mass = static_cast<real_t>(particle_mass(ch.daughter[1]));

  const real_t xmin = real_t(2) * log(real_t(2) * lepton_mass);
  const real_t xmax = real_t(2) * log(parent_mass);
  const real_t wmax = real_t(1.5);
  real_t t = real_t(0);
  const int kMaxLoop = 10000;
  for (int loop = 0; loop < kMaxLoop; ++loop) {
    const real_t x = rng.uniform() * (xmax - xmin) + xmin;
    const real_t w = rng.uniform() * wmax;
    t = exp(x);
    real_t ww = real_t(0);
    const real_t w1 = real_t(1) - real_t(4) * lepton_mass * lepton_mass / t;
    if (w1 > real_t(0)) {
      const real_t w2 = real_t(1) + real_t(2) * lepton_mass * lepton_mass / t;
      real_t w3 = real_t(1) - t / parent_mass / parent_mass;
      w3 = w3 * w3 * w3;
      ww = w3 * w2 * sqrt(w1);
    }
    if (w <= ww) { break; }
  }

  const real_t p_gamma = pmx(parent_mass, real_t(0), sqrt(t));
  real_t costheta = real_t(2) * rng.uniform() - real_t(1);
  real_t sintheta = sqrt((real_t(1) - costheta) * (real_t(1) + costheta));
  real_t phi = units::twopi<real_t>() * rng.uniform();
  const real_t gx = sintheta * cos(phi);
  const real_t gy = sintheta * sin(phi);
  const real_t gz = costheta;

  const real_t beta = p_gamma / (parent_mass - p_gamma);

  const real_t p_lepton = pmx(sqrt(t), lepton_mass, lepton_mass);
  const real_t e_lepton = sqrt(p_lepton * p_lepton + lepton_mass * lepton_mass);
  costheta = real_t(2) * rng.uniform() - real_t(1);
  sintheta = sqrt((real_t(1) - costheta) * (real_t(1) + costheta));
  phi = units::twopi<real_t>() * rng.uniform();
  const real_t lx = sintheta * cos(phi);
  const real_t ly = sintheta * sin(phi);
  const real_t lz = costheta;

  out.push(ch.daughter[0], 0, real_t(0), p_gamma, gx, gy, gz);
  out.push(ch.daughter[1], 1, lepton_mass, e_lepton - lepton_mass, lx, ly, lz);
  out.push(ch.daughter[2], 2, lepton_mass, e_lepton - lepton_mass, -lx, -ly, -lz);

  for (int i = 1; i <= 2; ++i) {
    FourVector<real_t> p4 = out.p[i].four_momentum();
    lorentz_boost(p4, -gx * beta, -gy * beta, -gz * beta);
    out.p[i].set_four_momentum(p4);
  }
}

// ---------------------------------------------------------------------------------------
// G4NeutronBetaDecayChannel
// ---------------------------------------------------------------------------------------

/// G4NeutronBetaDecayChannel::DecayIt - "free neutron beta decay kinematics, neglects
/// neutron/electron polarization, without Coulomb effect".
///
/// Random draws, in order: (x, w, r0) until accepted, then costheta and phi for the overall
/// rotation, then phin for the neutrino's azimuth about the electron.
///
/// The acceptance is `if (r > r0) break;` - the loop exits when the WEIGHT EXCEEDS the
/// uniform envelope draw, which is the normal sense; note that it is written the other way
/// round from G4KL3DecayChannel's `if (r <= w)`, and both are kept as written.
///
/// aENuCorr = -0.102 is the electron-neutrino angular correlation coefficient, declared as a
/// local const in the header's DecayIt and not settable.
///
/// This channel is unreachable in a QBBC run: `G4NeutronTrackingCut` kills a neutron at 10 us
/// and the neutron's lifetime is 880.2 s, so the decay probability over a tracked neutron's
/// life is about 1e-11. It is transcribed because the neutron IS an unstable transported
/// species and G4Decay is registered on it, so the process has to answer - and because a
/// future run without the tracking cut would otherwise silently produce nothing.
template <typename real_t, typename rng_t>
__host__ __device__ inline void neutron_beta_decay(const ChannelRow& ch, int parent_pdg,
                                                   rng_t& rng, DecayProducts<real_t>& out) {
  const real_t parent_mass = static_cast<real_t>(particle_mass(parent_pdg));
  real_t dm[3];
  real_t sum_dm = real_t(0);
  for (int i = 0; i < 3; ++i) {
    dm[i] = static_cast<real_t>(particle_mass(ch.daughter[i]));
    sum_dm += dm[i];
  }
  const real_t xmax = parent_mass - sum_dm;
  const real_t a_enu_corr = real_t(-0.102);

  real_t x = real_t(0);
  real_t p = real_t(0);
  real_t w = real_t(0);
  const real_t m_e = dm[0];
  const int kMaxLoop = 10000;
  for (int loop = 0; loop < kMaxLoop; ++loop) {
    x = xmax * rng.uniform();
    p = sqrt(x * (x + real_t(2) * m_e));
    w = real_t(1) - real_t(2) * rng.uniform();
    const real_t r = p * (x + m_e) * (xmax - x) * (xmax - x) *
                     (real_t(1) + a_enu_corr * p / (x + m_e) * w);
    const real_t r0 = rng.uniform() * (xmax + m_e) * (xmax + m_e) * xmax * xmax *
                      (real_t(1) + a_enu_corr);
    if (r > r0) { break; }
  }

  const real_t costheta = real_t(2) * rng.uniform() - real_t(1);
  const real_t theta = acos(costheta);
  const real_t phi = units::twopi<real_t>() * rng.uniform();
  // rm = Rz(phi) * Ry(theta): CLHEP's HepRotation::rotateY/rotateZ LEFT-multiply
  // (Rotation.cc:74, :87), so rotateY then rotateZ composes in that order.
  const real_t cth = cos(theta), sth = sin(theta);
  const real_t cph = cos(phi), sph = sin(phi);
  const real_t rxx = cph * cth, rxy = -sph, rxz = cph * sth;
  const real_t ryx = sph * cth, ryy = cph, ryz = sph * sth;
  const real_t rzx = -sth, rzy = real_t(0), rzz = cth;

  // daughter 0: the electron, along +z before the rotation
  {
    const real_t vx = real_t(0), vy = real_t(0), vz = real_t(1);
    const real_t dx = rxx * vx + rxy * vy + rxz * vz;
    const real_t dy = ryx * vx + ryy * vy + ryz * vz;
    const real_t dz = rzx * vx + rzy * vy + rzz * vz;
    out.push_momentum(ch.daughter[0], 0, dx * p, dy * p, dz * p);
  }
  // daughter 1: the antineutrino, at cos = w from the electron. Its energy comes from the
  // exact two-body-plus-recoil solution, not from the phase-space share.
  real_t e_nu = (parent_mass - dm[2]) * (parent_mass + dm[2]) + (m_e * m_e) -
                real_t(2) * parent_mass * (x + m_e);
  e_nu /= real_t(2) * (parent_mass + p * w - (x + m_e));
  const real_t cosn = w;
  const real_t phin = units::twopi<real_t>() * rng.uniform();
  const real_t sinn = sqrt((real_t(1) - cosn) * (real_t(1) + cosn));
  {
    const real_t vx = sinn * cos(phin), vy = sinn * sin(phin), vz = cosn;
    const real_t dx = rxx * vx + rxy * vy + rxz * vz;
    const real_t dy = ryx * vx + ryy * vy + ryz * vz;
    const real_t dz = rzx * vx + rzy * vy + rzz * vz;
    out.push_momentum(ch.daughter[1], 1, dx * e_nu, dy * e_nu, dz * e_nu);
  }
  // daughter 2: the proton, closing the momentum. Geant4 builds its direction from
  // (pPx cos(phin), pPx sin(phin), pPz) / pP with pP from its own kinetic energy eP, and
  // that is a unit vector only to the accuracy of the energy bookkeeping - kept as written.
  {
    const real_t e_p = parent_mass - e_nu - (x + m_e) - dm[2];
    const real_t ppx = -e_nu * sinn;
    const real_t ppz = -p - e_nu * cosn;
    const real_t pp = sqrt(e_p * (e_p + real_t(2) * dm[2]));
    const real_t vx = ppx / pp * cos(phin), vy = ppx / pp * sin(phin), vz = ppz / pp;
    const real_t dx = rxx * vx + rxy * vy + rxz * vz;
    const real_t dy = ryx * vx + ryy * vy + ryz * vz;
    const real_t dz = rzx * vx + rzy * vy + rzz * vz;
    out.push_momentum(ch.daughter[2], 2, dx * pp, dy * pp, dz * pp);
  }
}

// ---------------------------------------------------------------------------------------
// The dispatcher
// ---------------------------------------------------------------------------------------

/// G4VDecayChannel::DecayIt through the vtable, plus G4PhaseSpaceDecayChannel::DecayIt's own
/// switch on the number of daughters.
///
/// `parent_mass` is the DYNAMIC mass G4Decay passes (`aParticle->GetMass()`). Only
/// G4PhaseSpaceDecayChannel uses it - and it uses it only when positive, falling back to the
/// PDG mass otherwise. G4MuonDecayChannel, G4KL3DecayChannel, G4DalitzDecayChannel and
/// G4NeutronBetaDecayChannel all take the argument and ignore it, reading the PDG mass
/// instead, so passing a perturbed mass to one of those changes nothing. That asymmetry is
/// Geant4's and it is why this function takes both the mass and the parent's PDG code.
///
/// AND THE MASS THE KINEMATICS USE IS NOT THE MASS THE BOOST USES. `out.parent_mass` is what
/// G4DecayProducts stores for its parent and what `G4DecayProducts::Boost` builds beta from;
/// every phase-space channel creates that parent through the four-argument
/// G4DynamicParticle constructor, which SNAPS the mass it is given back to the PDG value
/// unless the two differ by more than 1e-5 MeV (G4DynamicParticle.cc:92). So a dynamic parent
/// mass 1e-6 MeV off the PDG value is used for the daughter momenta and NOT for the boost,
/// and this function is the only place that knows both numbers. The samplers therefore leave
/// `out.parent_mass` alone; it is set here, once, for every channel kind.
template <typename real_t, typename rng_t>
__host__ __device__ inline void channel_decay_it(const ChannelRow& ch, int parent_pdg,
                                                 real_t parent_mass, rng_t& rng,
                                                 DecayProducts<real_t>& out) {
  // Geant4 returns a FRESH G4DecayProducts from every DecayIt, so the buffer starts empty
  // whether or not the caller cleared it. Without this the second decay through a reused
  // buffer appends to the first, `push` silently refuses past kMaxProducts, and the products
  // are a mixture of two decays - which is exactly what happened, and what the "only 1 of
  // 400000 samples produced products" line in tests/test_decay.cu was reporting.
  out.clear();
  const double pdg_parent_mass = particle_mass(parent_pdg);
  switch (ch.kind) {
    case ChannelKind::kPhaseSpace: {
      if (channel_needs_dynamical_mass(ch)) {
        out.fail(DecayStatus::kDynamicalMassNotPorted);
        return;
      }
      // G4PhaseSpaceDecayChannel::DecayIt: `if (parentMass > 0) use it, else the PDG mass`.
      const real_t m = (parent_mass > real_t(0)) ? parent_mass
                                                 : static_cast<real_t>(pdg_parent_mass);
      out.parent_mass = static_cast<real_t>(
          snapped_dynamical_mass(pdg_parent_mass, static_cast<double>(m)));
      switch (ch.n_daughters) {
        case 1: phase_space_one_body(ch, m, out); return;
        case 2: phase_space_two_body(ch, m, rng, out); return;
        case 3: phase_space_three_body(ch, m, rng, out); return;
        default: phase_space_many_body(ch, m, rng, out); return;
      }
    }
    // These four build their rest-frame parent with the THREE-argument G4DynamicParticle
    // constructor, which takes no mass at all, so the stored parent mass is the PDG mass
    // however far the dynamic mass has drifted - the same fact that makes them ignore the
    // `parent_mass` argument for their kinematics.
    case ChannelKind::kMuonDecay:
      out.parent_mass = static_cast<real_t>(pdg_parent_mass);
      muon_decay<real_t>(ch, parent_pdg, rng, out);
      return;
    case ChannelKind::kKL3:
      out.parent_mass = static_cast<real_t>(pdg_parent_mass);
      kl3_decay<real_t>(ch, parent_pdg, rng, out);
      return;
    case ChannelKind::kDalitz:
      out.parent_mass = static_cast<real_t>(pdg_parent_mass);
      dalitz_decay<real_t>(ch, parent_pdg, rng, out);
      return;
    case ChannelKind::kNeutronBeta:
      out.parent_mass = static_cast<real_t>(pdg_parent_mass);
      neutron_beta_decay<real_t>(ch, parent_pdg, rng, out);
      return;
  }
  out.fail(DecayStatus::kChannelNotPorted);
}

}  // namespace g4gpu::decay
