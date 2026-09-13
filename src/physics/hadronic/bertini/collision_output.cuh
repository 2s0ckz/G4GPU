// G4CollisionOutput, G4InuclNuclei, G4CascadeCheckBalance, G4CascadeRecoilMaker and
// G4CascadeCoalescence: the bookkeeping that turns a list of cascade products into an event.
//
// Transcribed from Geant4 11.1.1, cascade/cascade/src/:
//   G4CollisionOutput::reset / add / addOutgoingParticle(s) / addOutgoingNucleus /
//     addRecoilFragment / removeOutgoingParticle / removeRecoilFragment /
//     getTotalOutputMomentum / getTotalCharge / getTotalBaryonNumber / getTotalStrangeness /
//     trivialise / setOnShell / setRemainingExitationEnergy / selectPairToTune /
//     tuneSelectedPair / boostToLabFrame
//   G4InuclNuclei::fill / setExitationEnergy / getNucleiMass / makeG4Fragment
//   G4CascadeCheckBalance::collide (the four overloads the cascade uses) / okay and the
//     six per-quantity tests
//   G4CascadeRecoilMaker::collide / fillRecoil / deltaM / goodFragment / goodRecoil /
//     wholeEvent / goodNucleus / makeRecoilFragment
//   G4CascadeCoalescence::FindClusters / selectCandidates / tryClusters / createNuclei /
//     removeNucleons / getClusterMomentum / maxDeltaP / clusterType / goodCluster /
//     makeLightIon
//
// ---------------------------------------------------------------------------------------------
// **The recoil nucleus is not computed; it is what did not come out.** `G4CascadeRecoilMaker`
// has no model of a residual at all: it asks `G4CascadeCheckBalance` for the difference between
// the initial state and everything in the output list, and calls the NEGATIVE of that difference
// the recoil. `recoilZ = -deltaQ`, `recoilA = -deltaB`, `recoilMomentum = -deltaLV`. So the
// residual's mass number, charge and four-momentum are conservation identities, and its
// EXCITATION is `recoilMomentum.m() - getNucleiMass(A, Z)` - the amount by which the leftover
// four-momentum is heavier than a ground-state nucleus of the same (A, Z). Every check the
// cascader then applies (`goodFragment`, `goodRecoil`, `wholeEvent`, `goodNucleus`) is a check
// on that difference, and an event that fails one is regenerated from scratch.
//
// ---------------------------------------------------------------------------------------------
// Four things in `setOnShell` that a reading does not give you, and two of them are bugs that
// change what QBBC produces.
//
// **1. `encMeV` is a million times too small.** The line is
//
//     G4double encMeV = mom_non_cons.e() / GeV;   // Excitation below is in MeV
//
// and `mom_non_cons.e()` is already in Bertini's GeV. Converting GeV to MeV is `* GeV` in
// Geant4's unit system (where MeV == 1 and GeV == 1000); `/ GeV` divides by a thousand instead,
// leaving a number 1e6 below the MeV value it is compared against. Two lines below,
// `eex + encMeV >= 0.0` is therefore `eex >= 0` for any fragment with positive excitation, and
// `need_hard_tuning` is set false - the routine declares the energy balanced by nuclear
// excitation when it has moved essentially nothing. The neighbouring conversions are right:
// `setRemainingExitationEnergy` divides MeV by GeV to get GeV, and `fillRecoil` multiplies GeV
// by GeV to get MeV.
//
// **2. The fragment arm of that same block writes to a local and throws it away.**
//
//     G4LorentzVector fragMom = recoilFragments[i].GetMomentum();
//     G4double newMass = fragMom.m() + encMeV;
//     fragMom.setVectM(fragMom.vect(), newMass);
//     need_hard_tuning = false;
//
// `fragMom` is a copy. Nothing is assigned back to `recoilFragments[i]`. The arm's entire effect
// is to set `need_hard_tuning = false`, i.e. to skip the pair tuning that would have balanced
// the event. The loop also reads `recoilFragments[0]` where it indexes `i`, which only matters
// when there is more than one fragment - there never is.
//
// **3. So the only arm that actually does anything is the outgoing-NUCLEI one**, and it reaches
// `outgoingNuclei[0].setExitationEnergy(encMeV)` with `need_hard_tuning && encMeV > 0` - which,
// given (1), sets an excitation of order a millionth of the imbalance.
//
// **4. The hard tuning that is left picks a PAIR of outgoing particles with opposite momentum
// components along one axis and shifts that component between them.** `selectPairToTune`
// maximises `|p1[l]| + |p2[l]|` over pairs and axes subject to `p1[l]*p2[l] < 0` and both above
// `0.3*sqrt(1.88*|de|)`; `tuneSelectedPair` solves a quadratic for the shift. When no pair
// qualifies the event is left unbalanced and `on_shell` stays false, which is what makes
// `G4InuclCollider` retry.
//
// Reproduced as written, all four. A port that "fixes" (1) and (2) balances events Geant4 does
// not and rejects events Geant4 accepts, and `bertini_apply.csv`'s energy non-conservation
// column is the measurement of exactly that.
#ifndef G4GPU_BERTINI_COLLISION_OUTPUT_CUH
#define G4GPU_BERTINI_COLLISION_OUTPUT_CUH

#include <cmath>

#include "physics/hadronic/bertini/cascade_model.cuh"
#include "physics/hadronic/bertini/workspace.cuh"
#include "physics/hadronic/deexcitation/fragment.cuh"

namespace g4gpu::physics::hadronic::bert {

/// G4ExitonConfiguration.
struct ExitonConfiguration {
  int proton_quasi_particles = 0;
  int neutron_quasi_particles = 0;
  int proton_holes = 0;
  int neutron_holes = 0;

  __host__ __device__ void clear() {
    proton_quasi_particles = neutron_quasi_particles = 0;
    proton_holes = neutron_holes = 0;
  }
  __host__ __device__ bool empty() const {
    return proton_quasi_particles == 0 && neutron_quasi_particles == 0 &&
           proton_holes == 0 && neutron_holes == 0;
  }
  /// `incrementQP(ip)` and `incrementHoles(ip)` are both no-ops for anything but 1 and 2, and
  /// the cascader calls the second one with `holes.first`, which can be 0 when nothing was
  /// consumed. That silence is deliberate in Geant4 and reproduced.
  __host__ __device__ void increment_qp(int ip) {
    if (ip == 1) { ++proton_quasi_particles; }
    else if (ip == 2) { ++neutron_quasi_particles; }
  }
  __host__ __device__ void increment_holes(int ip) {
    if (ip == 1) { ++proton_holes; }
    else if (ip == 2) { ++neutron_holes; }
  }
};

/// G4InuclNuclei, reduced to its data.
///
/// It carries a four-momentum AS STORED - direction, kinetic energy and dynamical mass, exactly
/// as `G4InuclParticle` does (docs/RISK.md V125) - so the stored kinetic energy is kept beside it
/// rather than recomputed: `G4ParticleLargerEkin` and `setOnShell`'s
/// `getKineticEnergy() + enc > 0` test both read the STORED one.
struct InuclNucleus {
  int a = 0;
  int z = 0;
  LV momentum;
  double exc_MeV = 0.0;     ///< G4InuclNuclei::getExitationEnergy, MeV
  double mass = 0.0;        ///< the dynamical mass in GeV: getNucleiMass(a,z) + exc/1000
  double ekin = 0.0;        ///< the STORED kinetic energy, GeV
  ExitonConfiguration excitons;
};

/// G4InuclNuclei::setExitationEnergy. `ekin_new` is built from the OLD mass (`getMass()` is read
/// before `setMass`), so what is preserved across the call is |p|, not the energy - the momentum
/// is put on the new mass shell.
__host__ __device__ inline void nuclei_set_excitation(InuclNucleus& n, double e_MeV) {
  const double emass = inucl_nuclei_mass(n.a, n.z) + e_MeV * 0.001;
  const double ekin_new =
      (n.ekin == 0.0) ? 0.0
                      : std::sqrt(emass * emass + n.ekin * (2.0 * n.mass + n.ekin)) - emass;
  n.mass = emass;
  n.exc_MeV = e_MeV;
  n.ekin = ekin_new;
  // setKineticEnergy then getMomentum(): |p| = sqrt(T^2 + 2mT) along the stored direction.
  const Vec3d dir = clhep_unit(n.momentum.v);
  const double pm = std::sqrt(ekin_new * ekin_new + 2.0 * emass * ekin_new);
  n.momentum = LV(dir * pm, ekin_new + emass);
}

/// G4InuclNuclei::fill(mom, a, z, exc). The definition is installed first, so the store below
/// uses the GROUND-state ion mass; the excitation is applied afterwards and moves the mass.
__host__ __device__ inline InuclNucleus nuclei_fill(const LV& mom, int a, int z,
                                                    double exc_MeV = 0.0) {
  InuclNucleus n;
  n.a = a;
  n.z = z;
  n.mass = inucl_nuclei_mass(a, z);
  n.momentum = inucl_store_momentum_mass(mom, n.mass, &n.ekin);
  n.excitons.clear();
  nuclei_set_excitation(n, exc_MeV);
  return n;
}

/// G4InuclNuclei::setMomentum on an existing nucleus - `setOnShell`'s rebalancing calls it, and
/// the dynamical mass it stores against is the one the nucleus already has (ground state plus
/// whatever excitation was set), not the ground-state mass.
__host__ __device__ inline void nuclei_set_momentum(InuclNucleus& n, const LV& mom) {
  n.momentum = inucl_store_momentum_mass(mom, n.mass, &n.ekin);
}

// =============================================================================================
// G4CollisionOutput
// =============================================================================================

/// One elementary particle on the output list: a type code and a four-momentum as stored.
struct OutgoingParticle {
  int type = 0;
  LV momentum;
  double ekin = 0.0;   ///< the STORED kinetic energy, which is what G4ParticleLargerEkin reads
};

/// G4CollisionOutput's three lists, with capacities.
///
/// Geant4 keeps `outgoingParticles`, `outgoingNuclei` and `recoilFragments` as `std::vector`s
/// with no bound. Each capacity here is a REFUSAL, reported by name through `overflow` rather
/// than truncating a final state.
struct CollisionOutput {
  OutgoingParticle particles[kMaxOutgoingParticles];
  int n_particles = 0;
  InuclNucleus nuclei[kMaxOutgoingNuclei];
  int n_nuclei = 0;
  /// `recoilFragments` is a vector in Geant4 and holds exactly one entry everywhere in the
  /// cascade: `G4IntraNucleiCascader::finishCascade` adds one and `G4InuclCollider` removes it
  /// again after de-excitation. One slot, and a second `addRecoilFragment` is an overflow.
  deex::Fragment recoil_fragment;
  bool has_recoil_fragment = false;
  /// The exciton configuration the recoil carries.
  ///
  /// In Geant4 this lives ON the G4Fragment - `SetNumberOfHoles`, `SetNumberOfExcitedParticle` -
  /// and `G4NonEquilibriumEvaporator` reads it back with `G4ExitonConfiguration config(target)`.
  /// P3's `deex::Fragment` has no room for it (P6 carries its `Excitons` beside the fragment for
  /// the same reason), so it travels beside the fragment here. Dropping it would not fail any
  /// four-momentum comparison and would silently disable the whole non-equilibrium stage, whose
  /// only entry condition is `QP + QH > 0`.
  ExitonConfiguration recoil_excitons;
  double eex_rest = 0.0;    ///< GeV
  bool on_shell = false;
  LV mom_non_cons;
  CascadeOverflow overflow = CascadeOverflow::kNone;
  /// Set when `setOnShell` had to reach its last resort, the pair tuning. Geant4 has no such
  /// flag; it is here because that path is the one numerical amplifier in the whole cascade -
  /// `tuneSelectedPair` takes the root `-(W - sqrt(W*W+V))`, which for a small shift is a
  /// cancellation of two nearly equal numbers and carries a relative error of order
  /// `eps * W / x`. Measured: of 384 oracle events, the only four four-momentum components
  /// that disagree with Geant4 by more than 1e-12 all belong to one tuned event, and they
  /// disagree by up to 1.4e-10 while every other component in the same event is at 1e-14.
  bool hard_tuned = false;
};

__host__ __device__ inline void co_reset(CollisionOutput& o) {
  o.n_particles = 0;
  o.n_nuclei = 0;
  o.has_recoil_fragment = false;
  o.recoil_excitons.clear();
  o.eex_rest = 0.0;
  o.on_shell = false;
  o.mom_non_cons = LV();
  o.overflow = CascadeOverflow::kNone;
  o.hard_tuned = false;
}

__host__ __device__ inline bool co_add_particle(CollisionOutput& o, int type, const LV& mom) {
  if (o.n_particles >= kMaxOutgoingParticles) {
    o.overflow = CascadeOverflow::kOutgoingParticles;
    return false;
  }
  OutgoingParticle& p = o.particles[o.n_particles++];
  p.type = type;
  p.momentum = mom;
  p.ekin = inucl_stored_kinetic_energy(mom, type);
  return true;
}

/// `addOutgoingParticle(cparticle)` - the cascade particle's INUCL particle, four-momentum and
/// type only. Everything geometric about it is dropped here, which is the point: a particle that
/// leaves the nucleus stops being a cascade particle.
__host__ __device__ inline bool co_add_particle(CollisionOutput& o, const CascadeParticle& cp) {
  return co_add_particle(o, cp.type, cp.momentum);
}

__host__ __device__ inline bool co_add_nucleus(CollisionOutput& o, const InuclNucleus& n) {
  if (o.n_nuclei >= kMaxOutgoingNuclei) {
    o.overflow = CascadeOverflow::kOutgoingNuclei;
    return false;
  }
  o.nuclei[o.n_nuclei++] = n;
  return true;
}

/// `G4CollisionOutput::add(right)` - the two particle lists are APPENDED and the fragment list is
/// REPLACED, not combined. Geant4's own comment says so, and it is what lets
/// `G4InuclCollider::collide` accumulate de-excitation products into one buffer while the recoil
/// fragment stays a single object that the last writer owns.
__host__ __device__ inline void co_add(CollisionOutput& o, const CollisionOutput& r) {
  for (int i = 0; i < r.n_particles; ++i) {
    if (!co_add_particle(o, r.particles[i].type, r.particles[i].momentum)) { return; }
  }
  for (int i = 0; i < r.n_nuclei; ++i) {
    if (!co_add_nucleus(o, r.nuclei[i])) { return; }
  }
  o.recoil_fragment = r.recoil_fragment;
  o.has_recoil_fragment = r.has_recoil_fragment;
  o.recoil_excitons = r.recoil_excitons;
  o.eex_rest = 0.0;
  o.on_shell = false;
}

__host__ __device__ inline void co_remove_particle(CollisionOutput& o, int index) {
  if (index < 0 || index >= o.n_particles) { return; }
  for (int i = index; i + 1 < o.n_particles; ++i) { o.particles[i] = o.particles[i + 1]; }
  --o.n_particles;
}

/// getTotalOutputMomentum. The recoil fragment is in Geant4's MeV and is scaled here, as
/// `GetMomentum()/GeV` does there.
__host__ __device__ inline LV co_total_momentum(const CollisionOutput& o) {
  LV tot;
  for (int i = 0; i < o.n_particles; ++i) { tot += o.particles[i].momentum; }
  for (int i = 0; i < o.n_nuclei; ++i) { tot += o.nuclei[i].momentum; }
  if (o.has_recoil_fragment) {
    tot += LV(o.recoil_fragment.momentum.v * 0.001, o.recoil_fragment.momentum.e * 0.001);
  }
  return tot;
}

__host__ __device__ inline int co_total_charge(const CollisionOutput& o) {
  int q = 0;
  for (int i = 0; i < o.n_particles; ++i) {
    q += static_cast<int>(inucl_type_row(o.particles[i].type).charge);
  }
  for (int i = 0; i < o.n_nuclei; ++i) { q += o.nuclei[i].z; }
  if (o.has_recoil_fragment) { q += o.recoil_fragment.z; }
  return q;
}

__host__ __device__ inline int co_total_baryon(const CollisionOutput& o) {
  int b = 0;
  // `baryon()` on G4InuclElementaryParticle asks the DEFINITION, not G4InuclParticleNames::
  // baryon - the two disagree for a deuteron and the definition is what this total uses.
  for (int i = 0; i < o.n_particles; ++i) { b += inucl_type_row(o.particles[i].type).baryon; }
  for (int i = 0; i < o.n_nuclei; ++i) { b += o.nuclei[i].a; }
  if (o.has_recoil_fragment) { b += o.recoil_fragment.a; }
  return b;
}

/// getTotalStrangeness - and note that it sums ONLY the elementary particles. A hypernucleus
/// would carry strangeness and is not counted; Geant4's comment in CheckBalance says the same
/// thing about the initial state ("Currently we ignore possibility of hypernucleus target").
__host__ __device__ inline int co_total_strangeness(const CollisionOutput& o) {
  int s = 0;
  for (int i = 0; i < o.n_particles; ++i) { s += inucl_type_row(o.particles[i].type).strangeness; }
  return s;
}

/// setRemainingExitationEnergy - the total excitation still carried by the output, in GeV.
__host__ __device__ inline void co_set_remaining_excitation(CollisionOutput& o) {
  o.eex_rest = 0.0;
  for (int i = 0; i < o.n_nuclei; ++i) { o.eex_rest += o.nuclei[i].exc_MeV * 0.001; }
  if (o.has_recoil_fragment) { o.eex_rest += o.recoil_fragment.excitation * 0.001; }
}

// =============================================================================================
// G4CascadeCheckBalance
// =============================================================================================

/// What the balance checker measured. Geant4 keeps it on the object; here it is a value so that
/// the recoil maker and the interface can each hold their own.
struct CascadeBalance {
  LV initial;
  LV final_lv;
  int initial_baryon = 0, final_baryon = 0;
  int initial_charge = 0, final_charge = 0;
  int initial_strange = 0, final_strange = 0;
  double relative_limit = 1.0e-6;
  double absolute_limit = 1.0e-6;

  __host__ __device__ LV delta_lv() const { return final_lv - initial; }
  __host__ __device__ double delta_e() const { return final_lv.e - initial.e; }
  __host__ __device__ double relative_e() const {
    return (initial.e == 0.0) ? delta_e() : delta_e() / initial.e;
  }
  __host__ __device__ double delta_ke() const {
    return (final_lv.e - final_lv.mag()) - (initial.e - initial.mag());
  }
  __host__ __device__ double relative_ke() const {
    const double ki = initial.e - initial.mag();
    return (ki == 0.0) ? delta_ke() : delta_ke() / ki;
  }
  __host__ __device__ double delta_p() const { return g4gpu::mag(delta_lv().v); }
  __host__ __device__ double relative_p() const {
    const double pi = g4gpu::mag(initial.v);
    return (pi == 0.0) ? delta_p() : delta_p() / pi;
  }
  __host__ __device__ int delta_b() const { return final_baryon - initial_baryon; }
  __host__ __device__ int delta_q() const { return final_charge - initial_charge; }
  __host__ __device__ int delta_s() const { return final_strange - initial_strange; }

  __host__ __device__ bool energy_okay() const {
    return std::fabs(relative_e()) < relative_limit && std::fabs(delta_e()) < absolute_limit;
  }
  __host__ __device__ bool ekin_okay() const {
    return std::fabs(relative_ke()) < relative_limit && std::fabs(delta_ke()) < absolute_limit;
  }
  /// **Momentum gets ten times the slack.** `momentumOkay` compares against `10.*relativeLimit`
  /// and `10.*absoluteLimit` where energy compares against the limits themselves - so with the
  /// interface's (5%, 10 MeV) the momentum test is (50%, 100 MeV). Written out because it looks
  /// like a typo and is not: it is in 11.1.1 and it is what decides whether an event is retried.
  __host__ __device__ bool momentum_okay() const {
    return std::fabs(relative_p()) < 10.0 * relative_limit &&
           std::fabs(delta_p()) < 10.0 * absolute_limit;
  }
  __host__ __device__ bool baryon_okay() const { return delta_b() == 0; }
  __host__ __device__ bool charge_okay() const { return delta_q() == 0; }
  __host__ __device__ bool strange_okay() const { return delta_s() == 0; }
  /// G4CascadeCheckBalance::okay() - `(energyOkay() && momentumOkay() && baryonOkay() &&
  /// chargeOkay() && strangeOkay())`. `ekinOkay` is NOT in it.
  __host__ __device__ bool okay() const {
    return energy_okay() && momentum_okay() && baryon_okay() && charge_okay() && strange_okay();
  }
};

/// The initial state of a balance check: a bullet that is either an elementary particle or a
/// nucleus, plus a target nucleus.
struct BalanceInitial {
  LV bullet;
  int bullet_type = 0;     ///< 0 when the bullet is a nucleus
  int bullet_a = 0, bullet_z = 0;
  LV target;
  int target_a = 0, target_z = 0;
  bool has_bullet = true;
};

/// G4CascadeCheckBalance::collide(bullet, target, output) - the initial-state half.
///
/// **The electron correction.** Before comparing, the checker scans the OUTPUT for electrons and
/// adds their rest masses to the INITIAL four-momentum while subtracting their charge. That is
/// internal conversion during photon evaporation: the electron did not come from the collision,
/// it came out of the atomic shell, so the event only balances if the initial state is credited
/// with it. Bertini's own de-excitation never makes one; P3's photon evaporation does.
__host__ __device__ inline void balance_collide(CascadeBalance& b, const BalanceInitial& in,
                                                const CollisionOutput& o) {
  b.initial = LV();
  b.initial_charge = 0;
  b.initial_baryon = 0;
  b.initial_strange = 0;

  if (in.has_bullet) {
    b.initial += in.bullet;
    if (in.bullet_type != 0) {
      const InuclTypeRow r = inucl_type_row(in.bullet_type);
      b.initial_charge += static_cast<int>(r.charge);
      b.initial_baryon += r.baryon;
      b.initial_strange += r.strangeness;
    } else {
      b.initial_charge += in.bullet_z;
      b.initial_baryon += in.bullet_a;
    }
  }
  b.initial += in.target;
  b.initial_charge += in.target_z;
  b.initial_baryon += in.target_a;

  int nelec = 0;
  double el_mass = 0.0;
  for (int i = 0; i < o.n_particles; ++i) {
    if (o.particles[i].type == kElectron) {
      el_mass += inucl_particle_mass(kElectron) * 1000.0;   // GetPDGMass(), MeV
      ++nelec;
    }
  }
  if (nelec > 0) {
    b.initial += LV(Vec3d{0.0, 0.0, 0.0}, el_mass * 0.001);
    b.initial_charge -= nelec;
  }

  b.final_lv = co_total_momentum(o);
  b.final_baryon = co_total_baryon(o);
  b.final_charge = co_total_charge(o);
  b.final_strange = co_total_strangeness(o);
}

/// The `collide(<EP>, <CP>)` overload: the output list PLUS a list of cascade particles still in
/// flight, which is how `G4IntraNucleiCascader` asks for the recoil in mid-cascade. Geant4 builds
/// a temporary G4CollisionOutput for it; here the two contributions are summed directly, which
/// is the same arithmetic without the copy.
__host__ __device__ inline void balance_collide(CascadeBalance& b, const BalanceInitial& in,
                                                const CollisionOutput& o,
                                                const CascadeParticle* cparticles,
                                                int n_cparticles) {
  balance_collide(b, in, o);
  for (int i = 0; i < n_cparticles; ++i) {
    const int t = cparticles[i].type;
    b.final_lv += cparticles[i].momentum;
    b.final_baryon += inucl_type_row(t).baryon;
    b.final_charge += static_cast<int>(inucl_type_row(t).charge);
    b.final_strange += inucl_type_row(t).strangeness;
  }
}

// =============================================================================================
// G4CascadeRecoilMaker
// =============================================================================================

/// What is left over, and whether it is a nucleus.
struct RecoilState {
  int a = 0;
  int z = 0;
  LV momentum;
  double excitation_MeV = 0.0;
  double input_ekin = 0.0;      ///< the bullet's kinetic energy, for goodNucleus's ceiling
  double tolerance = 0.001;     ///< excTolerance, MeV - G4IntraNucleiCascader's small_ekin
};

/// G4CascadeRecoilMaker::deltaM - how much heavier the leftover four-momentum is than a
/// ground-state nucleus of the same (A, Z). GeV.
__host__ __device__ inline double recoil_delta_m(const RecoilState& r) {
  return r.momentum.mag() - inucl_nuclei_mass(r.a, r.z);
}

__host__ __device__ inline bool recoil_good_fragment(const RecoilState& r) {
  return r.a > 0 && r.z >= 0 && r.a >= r.z;
}

__host__ __device__ inline bool recoil_good_recoil(const RecoilState& r) {
  return recoil_good_fragment(r) && r.excitation_MeV > -r.tolerance;
}

/// wholeEvent - nothing left at all. The momentum test is against `excTolerance/GeV`, i.e. the
/// MeV tolerance reinterpreted as GeV, which is a thousand times tighter than it reads.
__host__ __device__ inline bool recoil_whole_event(const RecoilState& r) {
  return r.a == 0 && r.z == 0 && g4gpu::mag(r.momentum.v) < r.tolerance * 0.001 &&
         std::fabs(r.momentum.e) < r.tolerance * 0.001;
}

/// G4CascadeRecoilMaker::fillRecoil - the recoil IS the conservation difference, negated.
__host__ __device__ inline void recoil_fill(RecoilState& r, const CascadeBalance& b) {
  r.z = -b.delta_q();
  r.a = -b.delta_b();
  const LV d = b.delta_lv();
  r.momentum = LV(Vec3d{-d.v.x, -d.v.y, -d.v.z}, -d.e);
  // "Bertini uses MeV for excitation energy"
  r.excitation_MeV = recoil_good_fragment(r) ? recoil_delta_m(r) * 1000.0 : 0.0;
  if (std::fabs(r.excitation_MeV) < r.tolerance) { r.excitation_MeV = 0.0; }
}

/// G4CascadeRecoilMaker::goodNucleus - is this residual something the de-excitation chain can be
/// asked to handle?
///
/// The ceiling is the LARGER of a fifth of the bullet's kinetic energy and seven times the
/// residual's binding energy, which for a low-energy projectile on a heavy target is the second
/// one. `minExcitation` is 0.1 keV and anything under it is "effectively zero" and accepted
/// without further test.
__host__ __device__ inline bool recoil_good_nucleus(const RecoilState& r) {
  const double min_excitation = 1.0e-4;          // 0.1 keV in MeV
  const double reasonable_excitation = 7.0;
  const double fractional_excitation = 0.2;
  if (!recoil_good_recoil(r)) { return false; }
  if (r.excitation_MeV <= min_excitation) { return true; }
  const double dm = inucl_binding_energy(r.a, r.z);
  const double exc_max0z = fractional_excitation * r.input_ekin * 1000.0;
  const double exc_dm = reasonable_excitation * dm;
  const double exc_max = (exc_max0z > exc_dm) ? exc_max0z : exc_dm;
  return r.excitation_MeV < exc_max;
}

/// G4CascadeRecoilMaker::makeRecoilFragment.
///
/// The four-momentum is FORCED to match the excitation rather than carried over: the mass is
/// `getNucleiMass(A, Z) + excitation/GeV` and the momentum is `setVectM(recoil.vect(), mass)`, so
/// the three-momentum survives and the energy is rebuilt. Geant4's comment says why - "User may
/// have overridden excitation energy" - and `finishCascade` does exactly that for a quasi-elastic
/// scatter.
__host__ __device__ inline deex::Fragment recoil_make_fragment(const RecoilState& r,
                                                               const ExitonConfiguration& ex) {
  (void)ex;   // the excitons travel beside the fragment, in CollisionOutput::recoil_excitons
  deex::Fragment f;
  f.z = r.z;
  f.a = r.a;
  const double frag_mass = inucl_nuclei_mass(r.a, r.z) + r.excitation_MeV * 0.001;
  const LV mom = lv_set_vect_m(r.momentum.v, frag_mass);
  // `SetMomentum(fragMom*GeV)` - the fragment is in Geant4's MeV from here on.
  f.set_za_and_momentum(LV(mom.v * 1000.0, mom.e * 1000.0), r.z, r.a);
  return f;
}

// =============================================================================================
// G4CascadeCoalescence
// =============================================================================================

/// The three momentum cuts, from the dumped parameters. GeV/c.
struct CoalescenceCuts {
  double dp_max_doublet;
  double dp_max_triplet;
  double dp_max_alpha;
};

__host__ __device__ inline CoalescenceCuts coalescence_cuts(const CascadeParams& p) {
  return CoalescenceCuts{p.dp_max_doublet, p.dp_max_triplet, p.dp_max_alpha};
}

/// `maxDeltaP` - the largest momentum any member has in the CLUSTER's rest frame.
__host__ __device__ inline double coal_max_delta_p(const CollisionOutput& o, const int* idx,
                                                   int n) {
  LV p_cluster;
  for (int i = 0; i < n; ++i) { p_cluster += o.particles[idx[i]].momentum; }
  const Vec3d boost = p_cluster.boost_vector();
  double max_dp = -1.0;
  for (int i = 0; i < n; ++i) {
    LV m = o.particles[idx[i]].momentum;
    m.boost(Vec3d{-boost.x, -boost.y, -boost.z});
    const double dp = g4gpu::mag(m.v);
    if (dp > max_dp) { max_dp = dp; }
  }
  return max_dp;
}

/// `clusterType` - the SUM of the nucleon type codes, so pn is 3, ppn is 4, pnn is 5, ppnn is 6.
__host__ __device__ inline int coal_cluster_type(const CollisionOutput& o, const int* idx,
                                                 int n) {
  int type = 0;
  for (int i = 0; i < n; ++i) {
    const int t = o.particles[idx[i]].type;
    type += inucl_is_nucleon(t) ? t : 0;
  }
  return type;
}

/// `goodCluster` - the right nucleon content for the size, and every member within the size's
/// momentum cut of the cluster's rest frame.
///
/// `allNucleons` is not transcribed as a call: its loop body is `result &= getHadron(clus[0])`,
/// index **0** where every other line in the class indexes `i`, so it only ever tests the first
/// member. It cannot matter - `selectCandidates` skips non-nucleons before a candidate is ever
/// built - and reproducing an index bug that is provably unreachable would be reproducing
/// nothing, so the test is written the way it is meant, with a note that it is the way it is
/// meant and not the way it is written.
__host__ __device__ inline bool coal_good_cluster(const CollisionOutput& o, const int* idx,
                                                  int n, const CoalescenceCuts& cuts) {
  for (int i = 0; i < n; ++i) {
    if (!inucl_is_nucleon(o.particles[idx[i]].type)) { return false; }
  }
  const int type = coal_cluster_type(o, idx, n);
  if (n == 2) { return type == 3 && coal_max_delta_p(o, idx, n) < cuts.dp_max_doublet; }
  if (n == 3) {
    return (type == 4 || type == 5) && coal_max_delta_p(o, idx, n) < cuts.dp_max_triplet;
  }
  if (n == 4) { return type == 6 && coal_max_delta_p(o, idx, n) < cuts.dp_max_alpha; }
  return false;
}

/// G4CascadeCoalescence::FindClusters - the whole of it, in one pass over the output list.
///
/// **The search order is what decides the answer.** `selectCandidates` walks four nested index
/// loops in increasing order and calls `tryClusters` for the four-body combination FIRST, then
/// the three-body one after the innermost loop ends, then the two-body one - and a nucleon that
/// is taken by an accepted cluster is struck out of every later combination. So the first alpha
/// found wins over the deuteron that would have used one of its nucleons, and the earliest
/// indices win over later ones. The output list is in descending kinetic-energy order at this
/// point (`finishCascade` sorts it just before), so "earliest" means "most energetic".
///
/// Two lists are kept for a reason Geant4 makes explicit: `selectCandidates` marks a nucleon used
/// as soon as a candidate is ACCEPTED, and `createNuclei` then CLEARS the used set and re-marks
/// only the nucleons of candidates that actually became ions. `makeLightIon` can still refuse one
/// (an invalid Z for the size), and a nucleon in a refused candidate has to go back on the output
/// list rather than vanish.
__host__ __device__ inline void coal_find_clusters(CollisionOutput& o,
                                                   const CoalescenceCuts& cuts) {
  // At most one cluster per four nucleons, and the output list is bounded, so the candidate
  // list is bounded by a quarter of it. `used` is a bitmask over the particle list.
  constexpr int kMaxClusters = kMaxOutgoingParticles / 2;
  int cand_idx[kMaxClusters][4];
  int cand_n[kMaxClusters];
  int n_cand = 0;
  bool used[kMaxOutgoingParticles];
  for (int i = 0; i < o.n_particles; ++i) { used[i] = false; }

  const int nh = o.n_particles;
  int c[4];
  for (int i1 = 0; i1 < nh; ++i1) {
    if (!inucl_is_nucleon(o.particles[i1].type)) { continue; }
    for (int i2 = i1 + 1; i2 < nh; ++i2) {
      if (!inucl_is_nucleon(o.particles[i2].type)) { continue; }
      for (int i3 = i2 + 1; i3 < nh; ++i3) {
        if (!inucl_is_nucleon(o.particles[i3].type)) { continue; }
        for (int i4 = i3 + 1; i4 < nh; ++i4) {
          if (!inucl_is_nucleon(o.particles[i4].type)) { continue; }
          if (used[i1] || used[i2] || used[i3] || used[i4]) { continue; }
          c[0] = i1; c[1] = i2; c[2] = i3; c[3] = i4;
          if (coal_good_cluster(o, c, 4, cuts) && n_cand < kMaxClusters) {
            for (int k = 0; k < 4; ++k) { cand_idx[n_cand][k] = c[k]; }
            cand_n[n_cand++] = 4;
            used[i1] = used[i2] = used[i3] = used[i4] = true;
          }
        }
        if (used[i1] || used[i2] || used[i3]) { continue; }
        c[0] = i1; c[1] = i2; c[2] = i3;
        if (coal_good_cluster(o, c, 3, cuts) && n_cand < kMaxClusters) {
          for (int k = 0; k < 3; ++k) { cand_idx[n_cand][k] = c[k]; }
          cand_n[n_cand++] = 3;
          used[i1] = used[i2] = used[i3] = true;
        }
      }
      if (used[i1] || used[i2]) { continue; }
      c[0] = i1; c[1] = i2;
      if (coal_good_cluster(o, c, 2, cuts) && n_cand < kMaxClusters) {
        for (int k = 0; k < 2; ++k) { cand_idx[n_cand][k] = c[k]; }
        cand_n[n_cand++] = 2;
        used[i1] = used[i2] = true;
      }
    }
  }

  // createNuclei: build each candidate, and re-mark used only on success.
  for (int i = 0; i < o.n_particles; ++i) { used[i] = false; }
  for (int i = 0; i < n_cand; ++i) {
    const int n = cand_n[i];
    const int type = coal_cluster_type(o, cand_idx[i], n);
    int z = -1;
    if (n == 2 && type == 3) { z = 1; }        // deuteron (pn)
    if (n == 3 && type == 5) { z = 1; }        // triton (pnn)
    if (n == 3 && type == 4) { z = 2; }        // He-3 (ppn)
    if (n == 4 && type == 6) { z = 2; }        // alpha (ppnn)
    if (z < 0) { continue; }
    LV p_cluster;
    for (int k = 0; k < n; ++k) { p_cluster += o.particles[cand_idx[i][k]].momentum; }
    if (!co_add_nucleus(o, nuclei_fill(p_cluster, n, z, 0.0))) { return; }
    for (int k = 0; k < n; ++k) { used[cand_idx[i][k]] = true; }
  }

  // removeNucleons: from the highest index down, so the earlier indices stay valid.
  for (int i = o.n_particles - 1; i >= 0; --i) {
    if (used[i]) { co_remove_particle(o, i); }
  }
}

// =============================================================================================
// G4CollisionOutput::setOnShell and its two helpers
// =============================================================================================

/// `selectPairToTune` - find the pair of outgoing particles and the axis along which the largest
/// pair of opposite momentum components can absorb the imbalance.
///
/// Returns false when no pair qualifies, which leaves the event unbalanced and `on_shell` false.
__host__ __device__ inline bool co_select_pair_to_tune(const CollisionOutput& o, double de,
                                                       int& first, int& second, int& axis) {
  first = second = -1;
  axis = -1;
  if (o.n_particles < 2) { return false; }

  int ibest1 = -1, ibest2 = -1, i3 = -1;
  double pbest = 0.0;
  const double pcut = 0.3 * std::sqrt(1.88 * std::fabs(de));
  double p1 = 0.0;

  for (int i = 0; i < o.n_particles - 1; ++i) {
    const Vec3d& m1 = o.particles[i].momentum.v;
    for (int j = i + 1; j < o.n_particles; ++j) {
      const Vec3d& m2 = o.particles[j].momentum.v;
      for (int l = 0; l < 3; ++l) {
        const double a = (l == 0) ? m1.x : (l == 1) ? m1.y : m1.z;
        const double b = (l == 0) ? m2.x : (l == 1) ? m2.y : m2.z;
        if (a * b < 0.0 && std::fabs(a) > pcut && std::fabs(b) > pcut) {
          const double psum = std::fabs(a) + std::fabs(b);
          if (psum > pbest) {
            ibest1 = i;
            ibest2 = j;
            i3 = l;
            p1 = a;
            pbest = psum;
          }
        }
      }
    }
  }
  if (i3 < 0) { return false; }
  axis = i3;
  // The sign of `de` decides the order when p1 is exactly zero - Geant4's own note.
  if (de > 0.0) {
    first = (p1 > 0.0) ? ibest1 : ibest2;
    second = (p1 > 0.0) ? ibest2 : ibest1;
  } else {
    first = (p1 < 0.0) ? ibest2 : ibest1;
    second = (p1 < 0.0) ? ibest1 : ibest2;
  }
  return true;
}

__host__ __device__ inline double lv_component(const LV& v, int i) {
  return (i == 0) ? v.v.x : (i == 1) ? v.v.y : (i == 2) ? v.v.z : v.e;
}
__host__ __device__ inline void lv_set_component(LV& v, int i, double x) {
  if (i == 0) { v.v.x = x; } else if (i == 1) { v.v.y = x; } else if (i == 2) { v.v.z = x; }
  else { v.e = x; }
}

/// `tuneSelectedPair` - solve for the shift `x` along `axis` that makes the pair absorb
/// `mom_non_cons.e()`, and pick the root with the right sign.
__host__ __device__ inline bool co_tune_selected_pair(LV& mom1, LV& mom2, int axis,
                                                      const LV& mom_non_cons) {
  const double new_e12 = mom1.e + mom2.e + mom_non_cons.e;
  const double R = 0.5 * (new_e12 * new_e12 + mom2.e * mom2.e - mom1.e * mom1.e) / new_e12;
  const double Q = -(lv_component(mom1, axis) + lv_component(mom2, axis)) / new_e12;
  const double UDQ = 1.0 / (Q * Q - 1.0);
  const double W = (R * Q + lv_component(mom2, axis)) * UDQ;
  const double V = (mom2.e * mom2.e - R * R) * UDQ;
  const double DET = W * W + V;
  if (DET < 0.0) { return false; }

  const double x1 = -(W + std::sqrt(DET));
  const double x2 = -(W - std::sqrt(DET));
  bool xset = false;
  double x = 0.0;
  if (mom_non_cons.e > 0.0) {
    if (x1 > 0.0 && R + Q * x1 >= 0.0) { x = x1; xset = true; }
    if (!xset && x2 > 0.0 && R + Q * x2 >= 0.0) { x = x2; xset = true; }
  } else {
    if (x1 < 0.0 && R + Q * x1 >= 0.0) { x = x1; xset = true; }
    if (!xset && x2 < 0.0 && R + Q * x2 >= 0.0) { x = x2; xset = true; }
  }
  if (!xset) { return false; }
  lv_set_component(mom1, axis, lv_component(mom1, axis) + x);
  lv_set_component(mom2, axis, lv_component(mom2, axis) - x);
  return true;
}

/// G4CollisionOutput::setOnShell - make the final state balance the initial one, or say it could
/// not. See the four numbered notes in this file's header for what it really does.
__host__ __device__ inline void co_set_on_shell(CollisionOutput& o, const LV& bullet,
                                                const LV& target) {
  const double accuracy = 0.00001;    // 10 keV, in Bertini's GeV
  o.on_shell = false;

  LV ini_mom = bullet;
  LV momt = target;
  // The internal-conversion credit again: an electron on the output list means the initial state
  // has to be given its rest mass before the two are compared.
  const LV el4mom(Vec3d{0.0, 0.0, 0.0}, inucl_particle_mass(kElectron));
  for (int i = 0; i < o.n_particles; ++i) {
    if (o.particles[i].type == kElectron) { momt += el4mom; }
  }
  ini_mom += momt;

  LV out_mom = co_total_momentum(o);
  o.mom_non_cons = ini_mom - out_mom;
  double pnc = g4gpu::mag(o.mom_non_cons.v);
  double enc = o.mom_non_cons.e;

  co_set_remaining_excitation(o);

  if (std::fabs(enc) <= accuracy && pnc <= accuracy) {
    o.on_shell = true;
    return;
  }

  // Give the whole imbalance to the LAST particle that can still be physical afterwards,
  // searching backwards. Only one of the three lists is tried: particles, else nuclei, else the
  // fragment.
  if (o.n_particles > 0) {
    for (int ip = o.n_particles - 1; ip >= 0; --ip) {
      if (o.particles[ip].ekin + enc > 0.0) {
        const LV last = o.particles[ip].momentum + o.mom_non_cons;
        o.particles[ip].momentum = inucl_store_momentum(last, o.particles[ip].type,
                                                        &o.particles[ip].ekin);
        break;
      }
    }
  } else if (o.n_nuclei > 0) {
    for (int in = o.n_nuclei - 1; in >= 0; --in) {
      if (o.nuclei[in].ekin + enc > 0.0) {
        nuclei_set_momentum(o.nuclei[in], o.nuclei[in].momentum + o.mom_non_cons);
        break;
      }
    }
  } else if (o.has_recoil_fragment) {
    LV last(o.recoil_fragment.momentum.v * 0.001, o.recoil_fragment.momentum.e * 0.001);
    if ((last.e - last.mag()) + enc > 0.0) {
      last += o.mom_non_cons;
      o.recoil_fragment.set_momentum(LV(last.v * 1000.0, last.e * 1000.0));
    }
  }

  out_mom = co_total_momentum(o);
  o.mom_non_cons = ini_mom - out_mom;
  pnc = g4gpu::mag(o.mom_non_cons.v);
  enc = o.mom_non_cons.e;

  bool need_hard_tuning = true;
  // See note 1: `/ GeV` where the comment asks for MeV, so this is a MILLIONTH of the imbalance
  // in MeV. Transcribed as written - `* 1000.0` would be the intended conversion.
  const double encMeV = o.mom_non_cons.e * 0.001;
  if (o.has_recoil_fragment) {
    const double eex = o.recoil_fragment.excitation;
    if (eex > 0.0 && eex + encMeV >= 0.0) {
      // Note 2: Geant4 modifies a LOCAL copy of the fragment's four-momentum here and never
      // writes it back, so the only effect of this arm is the flag below. Reproduced by doing
      // nothing, which is what it does.
      need_hard_tuning = false;
    }
  } else if (o.n_nuclei > 0) {
    for (int i = 0; i < o.n_nuclei; ++i) {
      const double eex = o.nuclei[i].exc_MeV;
      if (eex > 0.0 && eex + encMeV >= 0.0) {
        nuclei_set_excitation(o.nuclei[i], eex + encMeV);
        need_hard_tuning = false;
        break;
      }
    }
    if (need_hard_tuning && encMeV > 0.0) {
      nuclei_set_excitation(o.nuclei[0], encMeV);
      need_hard_tuning = false;
    }
  }

  if (!need_hard_tuning) {
    o.on_shell = true;
    return;
  }

  o.hard_tuned = true;
  int t1 = -1, t2 = -1, axis = -1;
  if (!co_select_pair_to_tune(o, enc, t1, t2, axis)) { return; }

  LV mom1 = o.particles[t1].momentum;
  LV mom2 = o.particles[t2].momentum;
  if (!co_tune_selected_pair(mom1, mom2, axis, o.mom_non_cons)) { return; }
  o.particles[t1].momentum = inucl_store_momentum(mom1, o.particles[t1].type,
                                                  &o.particles[t1].ekin);
  o.particles[t2].momentum = inucl_store_momentum(mom2, o.particles[t2].type,
                                                  &o.particles[t2].ekin);

  out_mom = co_total_momentum(o);
  // std::sort(outgoingParticles, G4ParticleLargerEkin()) - descending STORED kinetic energy.
  for (int i = 1; i < o.n_particles; ++i) {
    const OutgoingParticle key = o.particles[i];
    int j = i - 1;
    while (j >= 0 && o.particles[j].ekin < key.ekin) {
      o.particles[j + 1] = o.particles[j];
      --j;
    }
    o.particles[j + 1] = key;
  }
  o.mom_non_cons = ini_mom - out_mom;
  pnc = g4gpu::mag(o.mom_non_cons.v);
  enc = o.mom_non_cons.e;
  // **An OR, not an AND.** The event is called on shell if EITHER the energy or the momentum
  // came out under 10 keV, where the early return at the top of the function required both.
  o.on_shell = (std::fabs(enc) < accuracy || pnc < accuracy);
}

/// G4CollisionOutput::trivialise - the "no interaction" answer: the target and then the bullet,
/// unchanged, in that order.
__host__ __device__ inline void co_trivialise(CollisionOutput& o, int bullet_type,
                                              const LV& bullet, int bullet_a, int bullet_z,
                                              int target_a, int target_z, const LV& target) {
  co_reset(o);
  if (target_a > 1) {
    co_add_nucleus(o, nuclei_fill(target, target_a, target_z, 0.0));
  } else {
    co_add_particle(o, (target_z == 1) ? kProton : kNeutron, target);
  }
  if (bullet_type == 0) {
    co_add_nucleus(o, nuclei_fill(bullet, bullet_a, bullet_z, 0.0));
  } else {
    co_add_particle(o, bullet_type, bullet);
  }
}

/// The per-vector half of `boostToLabFrame`: reflect, rotate, boost, in that order.
///
/// All three, and the first two are easy to lose. `reflectionNeeded()` negates z when the SCM
/// axis points backwards - it is a REFLECTION, not a rotation, and Geant4 throws rather than
/// answer when `v2 < small` and the frame is not degenerate; `undefined` carries that state out.
/// `rotate` then maps the vector off +z onto the collision's own axes and is the identity for a
/// degenerate frame (docs/RISK.md V124 is about exactly that identity hiding the call).
__host__ __device__ inline LV co_boost_one(const LV& mom, const LorentzConvertor& lc,
                                           bool& undefined) {
  LV m = mom;
  bool und = false;
  if (lc_reflection_needed(lc, und)) { m.v.z = -m.v.z; }
  undefined = undefined || und;
  m = lc_rotate(lc, m);
  return lc_back_to_the_lab(lc, m);
}

/// G4CollisionOutput::boostToLabFrame - every particle, every nucleus and the recoil fragment,
/// through the same converter, with the particle list re-sorted by kinetic energy in between.
/// The fragment is in MeV and is scaled both ways around the boost, which Geant4's own comment
/// flags ("Fragment momentum must be converted to and from Bertini units").
__host__ __device__ inline void co_boost_to_lab(CollisionOutput& o, const LorentzConvertor& lc,
                                                bool& undefined) {
  undefined = false;
  for (int i = 0; i < o.n_particles; ++i) {
    o.particles[i].momentum =
        inucl_store_momentum(co_boost_one(o.particles[i].momentum, lc, undefined),
                             o.particles[i].type, &o.particles[i].ekin);
  }
  // std::sort(outgoingParticles, G4ParticleLargerEkin()) - here, between the two loops.
  for (int i = 1; i < o.n_particles; ++i) {
    const OutgoingParticle key = o.particles[i];
    int j = i - 1;
    while (j >= 0 && o.particles[j].ekin < key.ekin) {
      o.particles[j + 1] = o.particles[j];
      --j;
    }
    o.particles[j + 1] = key;
  }
  for (int i = 0; i < o.n_nuclei; ++i) {
    nuclei_set_momentum(o.nuclei[i], co_boost_one(o.nuclei[i].momentum, lc, undefined));
  }
  if (o.has_recoil_fragment) {
    const LV in(o.recoil_fragment.momentum.v * 0.001, o.recoil_fragment.momentum.e * 0.001);
    const LV out = co_boost_one(in, lc, undefined);
    o.recoil_fragment.set_momentum(LV(out.v * 1000.0, out.e * 1000.0));
  }
}

}  // namespace g4gpu::physics::hadronic::bert

#endif  // G4GPU_BERTINI_COLLISION_OUTPUT_CUH
