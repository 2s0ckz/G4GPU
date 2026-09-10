// The output buffer a decay writes into, and the G4DynamicParticle arithmetic it carries.
//
// THE CONTRACT
//
//   `DecayProducts<real_t>` is a fixed-size, caller-owned struct. `sample_decay` fills it and
//   never allocates, so it is usable from a kernel with the buffer in registers or in shared
//   memory. On return:
//
//     status == DecayStatus::kOK      `n` products are valid, in the CHANNEL'S OWN PUSH
//                                     order (which is not the daughter order - see
//                                     DecayProduct::daughter), and `channel` says which
//                                     channel of the parent's table fired.
//     status != kOK                   `n == 0`. Nothing was produced; the status names what
//                                     happened and `decay_status_message` prints it.
//
//   WHAT THE CALLER DOES WITH A NON-OK STATUS IS PER STATUS, not one rule. `kStable` is
//   G4Decay::DecayIt returning an untouched particle change (G4Decay.cc:195): the track
//   lives on. `kNoTable` is the DECAY101 path, and Geant4 KILLS the parent there -
//   `SetNumberOfSecondaries(0); ProposeTrackStatus(fStopAndKill);
//   ProposeLocalEnergyDeposit(0.0);` (G4Decay.cc:224) - so a refused species must not be
//   left transporting. `kNoChannel` is a FatalException in Geant4 (DECAY003) and has no
//   defined continuation at all. This paragraph replaces a blanket "the caller must not kill
//   the parent", which was right for one of the three.
//
//   THE SECONDARY ORDER IS REVERSED ON THE WAY OUT OF GEANT4. G4Decay::DecayIt builds its
//   secondary tracks with `products->PopProducts()` (G4Decay.cc:364), and PopProducts pops
//   from the BACK (G4DecayProducts.cc:141). So Geant4's list of secondary TRACKS is the
//   reverse of the product vector this buffer reproduces. Nothing physical depends on it,
//   but anything compared against a dump of G4Track secondaries has to reverse one side.
//
//   Each product carries (pdg, mass, direction, kinetic energy) - NOT a four-vector as its
//   primary form - because that is what G4DynamicParticle stores and what the transport wants.
//   `four_momentum()` derives the four-vector the way `G4DynamicParticle::Get4Momentum()`
//   does, and `set_four_momentum()` inverts it the way `G4DynamicParticle::Set4Momentum()`
//   does, including its dynamical-mass rule. Round-tripping through those two is lossy in the
//   last bits and Geant4 does it too - twice for the Dalitz leptons, once for everything in
//   `G4DecayProducts::Boost` - so the port does it in the same places rather than carrying a
//   four-vector cleanly and disagreeing in the sixteenth digit.
//
//   The four-momentum is in MeV throughout (momentum "in energy equivalent", as
//   G4DynamicParticle::GetTotalMomentum's own comment puts it), and directions are unit
//   vectors. Positions and times are the caller's: `sample_decay` returns kinematics only.
//
//   `kMaxProducts` is 5. That is the arity of `G4VDecayChannel`'s daughter-name constructor
//   (G4VDecayChannel.hh:53) and the size of `G4PhaseSpaceDecayChannel::givenDaughterMasses`
//   (`enum { MAX_N_DAUGHTERS = 5 }`, G4PhaseSpaceDecayChannel.hh:46) - it is NOT a limit
//   inside G4VDecayChannel, which allocates its daughter arrays dynamically and would take
//   more if a constructor offered them. No channel in any table this package transcribes has
//   more than three, so the two spare slots exist only so that the general N-body sampler has
//   somewhere to put a table that does.
#pragma once
#include <cstdint>

#include "physics/decay/decay_tables.hh"

namespace g4gpu::decay {

/// G4DynamicParticle's EnergyMomentumRelationAllowance, MeV: 1.0e-2*keV. The tolerance inside
/// which a four-vector's invariant mass is taken to BE the PDG mass rather than replacing it.
__host__ __device__ inline constexpr double energy_momentum_allowance() { return 1.0e-5; }

/// `G4DynamicParticle(definition, direction, kineticEnergy, dynamicalMass)`,
/// G4DynamicParticle.cc:79. The four-argument constructor does NOT store the mass it is
/// given: it starts from the PDG mass and replaces it only if the two differ by more than
/// EnergyMomentumRelationAllowance, and then by zero if the given mass is itself below the
/// allowance.
///
/// This matters in exactly one place and it is easy to miss. Every G4PhaseSpaceDecayChannel
/// builds its rest-frame parent with `new G4DynamicParticle(G4MT_parent, dummy, 0.0,
/// parentmass)`, so the mass G4DecayProducts stores - and therefore the mass `Boost` builds
/// beta from - is the SNAPPED one, while the kinematics inside the channel use the raw
/// `current_parent_mass`. A dynamic parent mass within 1e-5 MeV of the PDG value is used for
/// the daughter momenta and NOT for the boost.
__host__ __device__ inline double snapped_dynamical_mass(double pdg_mass, double given_mass) {
  if (fabs(pdg_mass - given_mass) > energy_momentum_allowance()) {
    return (given_mass > energy_momentum_allowance()) ? given_mass : 0.0;
  }
  return pdg_mass;
}

enum class DecayStatus : int {
  kOK = 0,
  /// The parent's PDG code has no transcribed table. decay_refusal_reason(pdg) says why.
  kNoTable,
  /// G4ParticleDefinition::GetPDGStable() is true. G4Decay::DecayIt returns an untouched
  /// particle change in this case, which is a no-op and not an error.
  kStable,
  /// G4DecayTable::SelectADecayChannel returned nullptr: every channel failed
  /// IsOKWithParentMass at this parent mass, so sumBR was zero. Geant4 raises a
  /// FatalException here (G4Decay.cc, "DECAY003").
  kNoChannel,
  /// A channel kind this package has not transcribed. Cannot happen for the tables in
  /// decay_tables.hh; kept so that adding a row without adding a sampler fails loudly.
  kChannelNotPorted,
  /// The sum of the daughters' masses exceeds the parent mass, which G4PhaseSpaceDecayChannel
  /// reports as a JustWarning "PART112" and answers with an EMPTY product list - the parent is
  /// then killed with no secondaries. Reproduced as a status rather than as silence.
  kDaughterMassTooLarge,
  /// G4PhaseSpaceDecayChannel's `withWidth` branch, where a daughter's own width is more than
  /// a thousandth of its mass and its mass is resampled from a Breit-Wigner. Unreachable for
  /// every daughter in decay_tables.hh - the widest is pi0 at 7.73e-6 MeV against 135 MeV,
  /// which is 6e-8 of its mass - so the branch is refused rather than left half-written.
  kDynamicalMassNotPorted,
  /// G4PhaseSpaceDecayChannel::ManyBodyDecayIt gave up: more than 100 tries produced no set
  /// of virtual subsystem masses that closes. Geant4 raises PART113 (JustWarning) and returns
  /// a NULL product list, which is a different outcome from PART112's empty one, and it is a
  /// different outcome from SelectADecayChannel finding nothing - so it has its own status
  /// rather than borrowing kNoChannel's.
  kKinematicsFailed,
};

__host__ __device__ inline const char* decay_status_message(DecayStatus s) {
  switch (s) {
    case DecayStatus::kOK: return "ok";
    case DecayStatus::kNoTable: return "no decay table for this PDG code (refused, not stable)";
    case DecayStatus::kStable: return "particle is flagged stable; G4Decay does nothing";
    case DecayStatus::kNoChannel:
      return "no channel passes IsOKWithParentMass at this parent mass "
             "(G4Decay.cc raises FatalException DECAY003)";
    case DecayStatus::kChannelNotPorted: return "channel kind not transcribed";
    case DecayStatus::kDaughterMassTooLarge:
      return "sum of daughter masses exceeds the parent mass "
             "(G4PhaseSpaceDecayChannel PART112: empty product list)";
    case DecayStatus::kDynamicalMassNotPorted:
      return "daughter width exceeds 1e-3 of its mass: G4VDecayChannel::DynamicalMass "
             "resampling is not transcribed (no daughter in any ported table reaches it)";
    case DecayStatus::kKinematicsFailed:
      return "N-body kinematics did not close in 100 tries "
             "(G4PhaseSpaceDecayChannel PART113: null product list)";
  }
  return "unknown";
}

template <typename real_t>
struct FourVector {
  real_t x, y, z, t;
};

/// `HepLorentzVector::boost(bx,by,bz)`, CLHEP LorentzVector.cc:54, verbatim - including the
/// `b2 > 0 ? (gamma-1)/b2 : 0` guard and the order the four components are written in. The
/// comment in CLHEP explains why the inaccuracy of (gamma-1)/b2 at small beta does not matter:
/// it multiplies O(beta^2) and is added to an O(beta) term.
template <typename real_t>
__host__ __device__ inline void lorentz_boost(FourVector<real_t>& p, real_t bx, real_t by,
                                              real_t bz) {
  const real_t b2 = bx * bx + by * by + bz * bz;
  const real_t ggamma = real_t(1) / sqrt(real_t(1) - b2);
  const real_t bp = bx * p.x + by * p.y + bz * p.z;
  const real_t gamma2 = (b2 > real_t(0)) ? (ggamma - real_t(1)) / b2 : real_t(0);
  const real_t t = p.t;
  p.x = p.x + gamma2 * bp * bx + ggamma * bx * t;
  p.y = p.y + gamma2 * bp * by + ggamma * by * t;
  p.z = p.z + gamma2 * bp * bz + ggamma * bz * t;
  p.t = ggamma * (t + bp);
}

/// One decay product, with G4DynamicParticle's state: what it is, what mass it is carrying
/// (the DYNAMICAL mass, which Set4Momentum can move off the PDG value), where it is going and
/// how much kinetic energy it has.
template <typename real_t>
struct DecayProduct {
  int pdg = 0;
  /// Which daughter of the channel this is, 0-based, in G4VDecayChannel's daughter order.
  ///
  /// It is carried explicitly because the product LIST order is not the daughter order and
  /// cannot be: G4PhaseSpaceDecayChannel::ThreeBodyDecayIt pushes daughter 0, then 2, then 1,
  /// because it derives daughter 1's momentum from the other two, and G4KL3DecayChannel does
  /// the same. The port pushes in Geant4's push order so that a comparison against the oracle
  /// is a straight index join, and this field is how a caller that wants "the pion" finds it
  /// - K+ -> pi+ pi+ pi- has two daughters of the same species, so PDG code alone will not do.
  int daughter = 0;
  real_t mass = 0;   ///< MeV. G4DynamicParticle::GetMass(), i.e. theDynamicalMass.
  real_t ekin = 0;   ///< MeV
  real_t dir[3] = {real_t(0), real_t(0), real_t(1)};  ///< unit vector

  /// G4DynamicParticle::Get4Momentum(), G4DynamicParticle.icc:163. The momentum is recomputed
  /// from the kinetic energy as sqrt(E*E + 2*m*E) - not sqrt((E+2m)*E), which is what
  /// GetTotalMomentum uses and which differs in the last bit.
  __host__ __device__ FourVector<real_t> four_momentum() const {
    const real_t p = sqrt(ekin * ekin + real_t(2) * mass * ekin);
    return FourVector<real_t>{dir[0] * p, dir[1] * p, dir[2] * p, ekin + mass};
  }

  /// G4DynamicParticle::GetTotalMomentum(), G4DynamicParticle.icc:175.
  __host__ __device__ real_t total_momentum() const {
    return sqrt((ekin + real_t(2) * mass) * ekin);
  }

  __host__ __device__ real_t total_energy() const { return ekin + mass; }

  /// G4DynamicParticle::Set4Momentum, G4DynamicParticle.cc:410. Three branches, and the
  /// middle one is the one that is easy to miss: the dynamical mass is replaced ONLY if the
  /// four-vector's invariant mass-squared differs from the PDG mass-squared by more than
  /// EnergyMRA2. Inside that window the stored mass is left alone and the kinetic energy is
  /// derived from it, so a boost that perturbs the invariant mass at the 1e-10 MeV level does
  /// not drift the mass; outside it, the particle really does change mass.
  __host__ __device__ void set_four_momentum(const FourVector<real_t>& p) {
    const real_t p2 = p.x * p.x + p.y * p.y + p.z * p.z;
    if (p2 > real_t(0)) {
      const real_t pm = sqrt(p2);
      dir[0] = p.x / pm;
      dir[1] = p.y / pm;
      dir[2] = p.z / pm;
      const real_t mass2 = p.t * p.t - p2;
      const real_t pdg_mass = static_cast<real_t>(particle_mass(pdg));
      const real_t pdg_mass2 = pdg_mass * pdg_mass;
      const real_t mra2 = static_cast<real_t>(energy_momentum_allowance() *
                                              energy_momentum_allowance());
      if (mass2 < mra2) {
        mass = real_t(0);
      } else if (fabs(pdg_mass2 - mass2) > mra2) {
        mass = sqrt(mass2);
      }
      ekin = p.t - mass;
    } else {
      dir[0] = real_t(1);
      dir[1] = real_t(0);
      dir[2] = real_t(0);
      ekin = real_t(0);
    }
  }

  /// G4DynamicParticle::SetMomentum, G4DynamicParticle.cc:393 - the three-vector constructor's
  /// body. The kinetic energy is p^2/(sqrt(p^2+m^2)+m) rather than sqrt(p^2+m^2)-m: the same
  /// number, without the cancellation at p << m. G4MuonDecayChannel, G4KL3DecayChannel and
  /// G4NeutronBetaDecayChannel all reach their daughters through this constructor, so a
  /// low-momentum neutrino's energy comes out of this form and not the subtraction.
  __host__ __device__ void set_momentum(real_t px, real_t py, real_t pz) {
    const real_t p2 = px * px + py * py + pz * pz;
    if (p2 > real_t(0)) {
      const real_t pm = sqrt(p2);
      dir[0] = px / pm;
      dir[1] = py / pm;
      dir[2] = pz / pm;
      ekin = p2 / (sqrt(p2 + mass * mass) + mass);
    } else {
      dir[0] = real_t(1);
      dir[1] = real_t(0);
      dir[2] = real_t(0);
      ekin = real_t(0);
    }
  }
};

template <typename real_t>
struct DecayProducts {
  /// G4VDecayChannel's own daughter limit; see the file comment.
  static constexpr int kMaxProducts = 5;

  DecayStatus status = DecayStatus::kOK;
  int n = 0;
  /// Index into the parent's slice of kChannels, or -1 when no channel was selected. The
  /// caller reports it so that a wrong branching ratio shows up as the wrong channel firing.
  int channel = -1;
  /// The parent mass the products were made in the rest frame of. G4DecayProducts keeps this
  /// as its parent G4DynamicParticle's mass and `Boost` reads it back to build beta, so it is
  /// part of the buffer and not a local.
  real_t parent_mass = 0;
  DecayProduct<real_t> p[kMaxProducts];

  __host__ __device__ void clear() {
    status = DecayStatus::kOK;
    n = 0;
    channel = -1;
    parent_mass = real_t(0);
  }

  __host__ __device__ bool push(int pdg, int daughter, real_t mass, real_t ekin, real_t dx,
                                real_t dy, real_t dz) {
    if (n >= kMaxProducts) { return false; }
    DecayProduct<real_t>& q = p[n];
    q.pdg = pdg;
    q.daughter = daughter;
    q.mass = mass;
    q.ekin = ekin;
    q.dir[0] = dx;
    q.dir[1] = dy;
    q.dir[2] = dz;
    ++n;
    return true;
  }

  /// Push with a three-momentum, through G4DynamicParticle's SetMomentum. The mass is the
  /// daughter's PDG mass, as every such constructor call in Geant4 uses.
  __host__ __device__ bool push_momentum(int pdg, int daughter, real_t px, real_t py,
                                         real_t pz) {
    if (n >= kMaxProducts) { return false; }
    DecayProduct<real_t>& q = p[n];
    q.pdg = pdg;
    q.daughter = daughter;
    q.mass = static_cast<real_t>(particle_mass(pdg));
    q.set_momentum(px, py, pz);
    ++n;
    return true;
  }

  __host__ __device__ void fail(DecayStatus s) {
    status = s;
    n = 0;
  }
};

/// G4DecayProducts::Boost(totalEnergy, momentumDirection), G4DecayProducts.cc:177.
///
/// Two things about it that a rewrite would get wrong. The total momentum is built as
/// sqrt((E-m)(E+m)) and is taken to be ZERO when E <= m, so a parent at rest boosts by
/// nothing rather than by a NaN. And the two-argument form calls the three-argument one,
/// which FIRST boosts each product into the stored parent's rest frame - except that the
/// stored parent is always at rest already (every channel builds it with kinetic energy 0:
/// G4PhaseSpaceDecayChannel.cc:130, G4MuonDecayChannel.cc:154, G4KL3DecayChannel.cc:202,
/// G4DalitzDecayChannel.cc:117, G4NeutronBetaDecayChannel.cc:153), so
/// `energy - mass > DBL_MIN` is false and that first boost is skipped. The port asserts the
/// same thing by construction: the products are in the rest frame, so there is one boost.
///
/// CALL THIS AT MOST ONCE PER PRODUCT SET. Geant4's Boost ends by boosting the stored PARENT
/// too (`G4LorentzVector parent4(0,0,0,mass); parent4.boost(...);
/// theParentParticle->Set4Momentum(parent4)`, G4DecayProducts.cc:243), which is what makes a
/// second call de-boost before re-boosting. This buffer carries only the parent's MASS - an
/// invariant, so it stays correct - and not its momentum, so a second call here would boost
/// a second time from the wrong frame. `sample_decay` calls it once and there is no reason
/// for a caller to call it at all; the alternative would be storing a parent four-vector
/// nothing else needs.
template <typename real_t>
__host__ __device__ inline void boost_products(DecayProducts<real_t>& products,
                                               real_t total_energy, const real_t dir[3]) {
  const real_t mass = products.parent_mass;
  real_t total_momentum = real_t(0);
  if (total_energy > mass) {
    total_momentum = sqrt((total_energy - mass) * (total_energy + mass));
  }
  const real_t bx = dir[0] * total_momentum / total_energy;
  const real_t by = dir[1] * total_momentum / total_energy;
  const real_t bz = dir[2] * total_momentum / total_energy;
  for (int i = 0; i < products.n; ++i) {
    FourVector<real_t> p4 = products.p[i].four_momentum();
    lorentz_boost(p4, bx, by, bz);
    products.p[i].set_four_momentum(p4);
  }
}

}  // namespace g4gpu::decay
