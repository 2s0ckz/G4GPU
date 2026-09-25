// G4BinaryLightIonReaction - the model QBBC gives every ion it transports, below 6 GeV/nucleon.
//
// Transcribed from G4BinaryLightIonReaction.{hh,cc} (models/binary_cascade) in 11.1.1.
//
// This is the piece that matters most for a galactic-cosmic-ray shielding calculation: it is
// what fragments the projectile, and projectile fragmentation is what decides what arrives
// behind a shield. `G4IonPhysics::ConstructProcess` registers it for d, t, He3, alpha and
// GenericIon from 0 to `GetMaxEnergyTransitionFTF_Cascade()` = 6 GeV, and
// `G4EnergyRangeManager` divides by the baryon number, so that window is **per nucleon**.
//
// ## The model has two arms and they share almost nothing
//
// `ApplyYourself`'s first decision is
//
//     if( (mom.t()-mom.mag())/pA < 50*MeV )  cascaders = FuseNucleiAndPrompound(mom);
//     else                                   result = Interact(mom, toBreit);
//
// Below **50 MeV per nucleon** there is no cascade at all: the two nuclei are fused into one
// compound nucleus of (pZ+tZ, pA+tA) with `pA` particle excitons of which `pZ` are charged and
// ZERO holes, and the whole reaction is `G4PreCompoundModel::DeExcite` on that - which is P6's
// `preco::deexcite`, which is on main.
//
// That arm is complete, and `tests/test_bic_apply.cu` is where it is checked against
// `G4BinaryLightIonReaction::ApplyYourself` itself - 20 cases of {d, alpha, C12} on
// {C, O, Al, Fe, Pb, H} at 1 to 45 MeV/nucleon, 5,000 events each. The fusion gate's verdict
// and the compound's (Z, A) come out exact in 100,000 events, the energy balance is exact to
// 2e-10 MeV per event once the conversion electrons are paid for, and the species yields and
// kinetic energies agree at 3.0 and 3.2 sigma against a 5 sigma band. An earlier draft of this
// header said the arm was "validated end to end" while no translation unit included this file
// at all; the claim above is the test's output, not a reading.
//
// Above it, `Interact` builds a `G4Fancy3DNucleus` for each nucleus, turns the projectile's
// nucleons into `G4KineticTrack`s aimed at the target with an impact parameter, and calls
// `G4BinaryCascade::Propagate`. **That arm is REFUSED by name**: `G4BinaryCascade` needs the
// `im_r_matrix` collision tree, which this package has not transcribed. `BlirRefusal::cascade`
// says so at the point it would have been needed, rather than producing a fusion answer for an
// energy where Geant4 would have produced a cascade.
//
// ## What the fusion arm actually is
//
// `FuseNucleiAndPrompound` is twenty lines and three of them are worth reading twice.
//
// **The fusion gate is kinematic, not energetic.** `mFused = GetIonMass(pZ+tZ, pA+tA)` is the
// compound's ground-state mass and `m2Compound = (mom.e()+mTarget, mom.vect()).m2()` is the
// invariant mass squared of the two nuclei together. If the second is below the first the nuclei
// cannot fuse and the method returns null - and `ApplyYourself` then returns the PRIMARY
// UNCHANGED, `isAlive` with its original energy and direction. So a sub-barrier ion-ion
// encounter in QBBC is an elastic-looking non-event, not an interaction, even though the
// inelastic cross section said one happened. That is the only place in this model where the
// final state is not `stopAndKill`.
//
// **The exciton configuration is the projectile.** `SetNumberOfParticles(pA)`,
// `SetNumberOfCharged(pZ)`, `SetNumberOfHoles(0)` - so every nucleon of the projectile is a
// particle exciton and there are no holes at all. `G4Fragment::GetNumberOfExcitons()` is
// `particles + holes = pA`, and P6's entry gate and loop gate both read it. For an alpha that is
// 4 excitons of which 2 are charged; for a C12 it is 12 of which 6 are.
//
// **`GetIonMass` is `G4NucleiProperties::GetNuclearMass` and that is not obvious.**
// `G4BinaryCascade::GetIonMass(Z, A)` calls `G4IonTable::GetIonMass`, which is
// `GetNucleusMass`, which asks `GetLightIon(Z, A)` for a PDG mass and otherwise calls
// `G4NucleiProperties::GetNuclearMass(A, Z)`. `GetLightIon` covers exactly (1,1), (1,2), (1,3),
// (2,3) and (2,4) - and `G4NucleiProperties::GetNuclearMass` special-cases those same five plus
// the neutron. So for every (Z, A) with `Z <= A` and `A >= 1` the two agree exactly, and
// `deex::nuclear_mass(A, Z)` is the right function. It is NOT the same as
// `G4Fancy3DNucleus::GetMass()`, which computes `Z m_p + (A-Z) m_n - BE` and differs from the
// table by the atomic electron binding - three answers to one question, all three used in
// 11.1.1, and `nucleus/fancy_3d_nucleus.cuh`'s `mass()` says where each is used.
//
// ## The rotation to the lab frame is the identity
//
// `ApplyYourself` ends with
//
//     G4LorentzRotation toZ;
//     toZ.rotateZ(-1*mom.phi());
//     toZ.rotateY(-1*mom.theta());
//     G4LorentzRotation toLab(toZ.inverse());
//
// and then `tmp *= toLab` on every secondary. Those are the same five lines as
// `G4HadProjectile::InitialiseLocal` (G4HadProjectile.cc:72-76) - and that method has ALREADY
// applied them: it stores `theMom.set(0, 0, sqrt(T(T+2m)), T+m)`, a four-momentum along +z, and
// keeps the inverse rotation in `toLabFrame` for `G4HadronicProcess::FillResult` to apply.
// `aTrack.Get4Momentum()` therefore returns a vector with `phi() == 0` and `theta() == 0`, both
// rotations are the identity, and so is `toLab`. The swapped case is the same: `toBreit*it`
// boosts an at-rest nucleus along the original projectile's velocity, which is +z.
//
// So the block is a no-op, and the port does not carry it. It is written out here because the
// only way to know that is to read `G4HadProjectile`, and because a caller that hands this
// model a projectile NOT along +z would need it. docs/RISK.md V72.
//
// ## REFUSED, by name
//
//   * `Interact` and everything it reaches - `G4BinaryCascade::Propagate`, the two
//     `G4Fancy3DNucleus`-to-`G4KineticTrack` conversions, the 150-try loop, the impact
//     parameter. `BlirRefusal::cascade`.
//   * `GetProjectileExcitation`, `SortResult` and `DeExciteSpectatorNucleus`, which take
//     `Interact`'s output and have no input without it. Their arithmetic is recorded in the
//     comments on `blir_projectile_excitation_term` below so that the next agent transcribing
//     them has the reading; none of it runs.
//   * the anti-nucleus and hyper-nucleus cases, which reach P3's and P6's own refusals.
#ifndef G4GPU_BIC_LIGHT_ION_REACTION_CUH
#define G4GPU_BIC_LIGHT_ION_REACTION_CUH

#include <cmath>

#include "core/units.cuh"
#include "core/vec3.cuh"
#include "physics/hadronic/bic/bic_params.cuh"
#include "physics/hadronic/bic/cascade_propagate.cuh"
#include "physics/hadronic/bic/rk_propagation.cuh"
#include "physics/hadronic/capture/neutron_rad_capture.cuh"
#include "physics/hadronic/deexcitation/nuclear_masses.cuh"
#include "physics/hadronic/precompound/precompound_model.cuh"
#include "physics/hadronic/process.cuh"

namespace g4gpu::bic {

using deex::LorentzVector;
using deex::Vec3d;

/// `G4DynamicParticle`'s `(definition, totalEnergy, momentum)` constructor, which is the shape
/// `ApplyYourself` builds every secondary with - and which is NOT `T = E - m`.
///
/// P7 already transcribed it, onto this exact struct, as
/// `capture::set_four_momentum(HadSecondary&, LorentzVector, pdg_mass)`: the `EnergyMRA2` shell
/// test with its three branches, the `unit()` rather than a division, and the +x direction for a
/// zero momentum. It is REUSED and not copied. P7's own comment explains why it has two copies
/// (P4's is on a different struct); this would be a third copy on the SAME struct, which is one
/// too many - the three branches decide whether a secondary's kinetic energy is `E - M_PDG` or
/// `E - sqrt(E^2-p^2)`, and those differ by half an ulp of E on a recoil, amplified by the
/// cancellation. docs/RISK.md V37.
///
/// The one difference between the two constructors is cosmetic: the `(def, E, p3)` form reads
/// `aParticleMomentum.mag2()` where `Set4Momentum` reads `aMomentum.vect().mag2()`. Same number.
namespace capture = g4gpu::physics::hadronic::capture;

/// What `ApplyYourself` refused.
struct BlirRefusal {
  /// The projectile's kinetic energy per nucleon is at or above 50 MeV, so Geant4 would have
  /// run `Interact` and this package has no cascade. Named, not approximated.
  /// The cascade arm refused - `Propagate` could not finish. `cascade_ref` says which of its
  /// refusals it was. Until P9e this flag meant "the cascade is not here at all".
  bool cascade = false;
  bool no_fusion = false;      ///< the kinematic gate failed; the primary is returned unchanged
  bool capacity = false;       ///< the caller's product or secondary buffer
  bool anti_or_hyper = false;  ///< a negative baryon number or a lambda
  bool nucleus = false;        ///< one of the two `G4Fancy3DNucleus::Init` calls failed
  /// 150 impact parameters and no final state. Geant4 prints "no final state for:" and returns
  /// the primary unchanged, which is what the port does; the flag is here because an event that
  /// comes back alive after 150 tries is a different thing from one that missed.
  bool no_final_state = false;
  /// The momentum non-conservation exit: `momentum.vect().mag() - momentum.e() >= 10 keV` after
  /// the correction loop, which Geant4 calls "invalid final state" and answers with the primary.
  bool momentum_not_conserved = false;
  int refused_pdg = 0;
  double refused_kin_per_nucleon = 0.0;
  CascadeRefusal cascade_ref;

  __host__ __device__ bool any() const {
    return cascade || no_fusion || capacity || anti_or_hyper || nucleus || no_final_state ||
           momentum_not_conserved;
  }
};

/// The projectile and target (A, Z) after `SetLighterAsProjectile`, and the four-momentum that
/// goes with the (possibly swapped) projectile.
struct BlirFrame {
  int pa = 0, pz = 0;
  int ta = 0, tz = 0;
  bool swapped = false;
  LorentzVector mom;              ///< the projectile four-momentum, along +z
  Vec3d breit_boost{0.0, 0.0, 0.0};  ///< `mom.boostVector()` of the ORIGINAL projectile
};

/// `G4BinaryLightIonReaction::SetLighterAsProjectile`.
///
/// If the target is lighter than the projectile the two are swapped and the new projectile's
/// four-momentum is rebuilt from scratch: `G4LorentzVector it(m1, (0,0,0))` - CLHEP's
/// `(t, p)` constructor, so an at-rest nucleus of the new projectile's mass - boosted by
/// `toBreit`, the original projectile's own `boostVector()`. So the heavier nucleus becomes the
/// projectile with the SAME velocity the lighter one had, and the secondaries are boosted back
/// and MIRRORED (`tmp.setVect(-tmp.vect())`) at the end.
///
/// `m1` is `GetIonMass` of the new (pZ, pA), i.e. of the original target.
__host__ __device__ inline BlirFrame blir_set_lighter_as_projectile(int pa, int pz, int ta,
                                                                    int tz,
                                                                    const LorentzVector& mom) {
  BlirFrame f;
  f.pa = pa;
  f.pz = pz;
  f.ta = ta;
  f.tz = tz;
  f.mom = mom;
  f.breit_boost = mom.boost_vector();
  if (ta < pa) {
    f.swapped = true;
    const int t = f.ta; f.ta = f.pa; f.pa = t;
    const int z = f.tz; f.tz = f.pz; f.pz = z;
    const double m1 = deex::nuclear_mass(f.pa, f.pz);
    // `toBreit * G4LorentzVector(m1, (0,0,0))`. The boost matrix applied to an at-rest
    // four-vector is (gamma*m1*beta, gamma*m1), and beta is along +z, so the result is along
    // +z as well - which is why the rotate-to-lab block stays the identity after a swap.
    const double b2 = g4gpu::mag2(f.breit_boost);
    const double gamma = 1.0 / std::sqrt(1.0 - b2);
    f.mom = LorentzVector(gamma * m1 * f.breit_boost, gamma * m1);
  }
  return f;
}

/// `(mom.t() - mom.mag())/pA`, the quantity the 50 MeV gate is applied to.
///
/// It is the projectile's kinetic energy in the frame `mom` is expressed in, divided by its
/// baryon number AFTER the swap - so for a heavy projectile on a light target it is the
/// ORIGINAL TARGET's mass number that divides, and the two orderings of the same collision are
/// gated on different numbers. `mom.mag()` is the invariant mass, not the PDG mass, so a
/// projectile whose four-momentum drifted from its mass shell is measured against its own.
__host__ __device__ inline double blir_kin_per_nucleon(const BlirFrame& f) {
  return (f.mom.e - f.mom.mag()) / static_cast<double>(f.pa);
}

/// One product of the fusion arm, in the shape `EnergyAndMomentumCorrector` needs.
struct BlirProduct {
  LorentzVector momentum;
  double pdg_mass = 0.0;
  int z = 0, a = 0, pdg = 0;
  bool newly_added = true;   ///< G4ReactionProduct::SetNewlyAdded
};

/// `G4BinaryLightIonReaction::FuseNucleiAndPrompound`, complete.
///
/// Returns false when the nuclei cannot fuse, which `ApplyYourself` turns into "return the
/// primary unchanged". `products`/`n_products` receive P6's output; `status` is P6's report,
/// which carries its own refusals and the pre-equilibrium/equilibrium boundary.
template <typename Rng>
__host__ __device__ inline bool blir_fuse_nuclei_and_precompound(
    const BlirFrame& f, const data::LevelTable& lt, const deex::FermiPool& pool,
    const preco::PrecoWorkspace& ws, Rng& rng, preco::PrecoStatus& status,
    BlirRefusal& ref) {
  const double m_fused = deex::nuclear_mass(f.pa + f.ta, f.pz + f.tz);
  const double m_target = deex::nuclear_mass(f.ta, f.tz);

  // `G4LorentzVector pCompound(mom.e()+mTarget, mom.vect())` - the (t, p) constructor again -
  // and `m2()`, which is E^2 - |p|^2 and can be negative.
  const LorentzVector p_compound(f.mom.v, f.mom.e + m_target);
  const double m2_compound =
      p_compound.e * p_compound.e - g4gpu::mag2(p_compound.v);
  if (m2_compound < m_fused * m_fused) {
    ref.no_fusion = true;
    return false;
  }

  // `aL = G4LorentzVector(mom.t()+mTarget, mom.vect())` - the same four-vector as pCompound,
  // built a second time. The commented-out line above it in Geant4 put the momentum along z
  // with `plop(0,0,mom.vect().mag())` and its own comment asks "why using plop in z direction?
  // this will not conserve momentum?"; the live line does not, and conserves it.
  const deex::Fragment frag = deex::make_fragment(f.pa + f.ta, f.pz + f.tz, p_compound);
  preco::Excitons ex;
  ex.particles = f.pa;   // SetNumberOfParticles(pA)
  ex.charged = f.pz;     // SetNumberOfCharged(pZ)
  ex.holes = 0;          // SetNumberOfHoles(0)

  status = preco::deexcite(frag, ex, lt, pool, ws, rng);
  return true;
}

/// `G4BinaryLightIonReaction::EnergyAndMomentumCorrector`.
///
/// Rescales a product list's three-momenta by a common factor until their total energy in their
/// own centre-of-mass frame equals the invariant mass of the four-momentum they are supposed to
/// add up to, then boosts them into that frame. It is the only thing in this model that enforces
/// energy conservation, and it is used twice: once on the cascade products against
/// `pInitialState - pspectators`, and once on them against `pInitialState - pFragments`.
///
/// Five things about it:
///
///   * it returns TRUE when the iteration did NOT converge. `success` is set only for the
///     printout, and the products are boosted and returned either way. The port returns the
///     convergence verdict separately so a caller can see it.
///   * the two early FALSE returns are the real failures: `SumMass > TotalCollisionMass` (the
///     products' rest masses do not fit inside the available invariant mass) and
///     `SumMom.m2() < 0` (the products are collectively spacelike).
///   * `SumMass` is reused: it is the sum of PDG masses for the first test and then overwritten
///     with `sqrt(SumMom.m2())`, which nothing reads. The second assignment is dead.
///   * the scale is a fixed point, not a solve: `Scale = TotalCollisionMass/Sum - 1` and the
///     next iteration multiplies every momentum by `1 + factor*Scale`. `factor` is 1 for the
///     first eleven attempts and then `max(1, log|OldScale/(OldScale-Scale)|)`, an acceleration
///     that can only speed it up.
///   * `OldScale == Scale` is an exact double comparison and it is the loop's real exit: a
///     frozen iteration stops rather than spinning 2500 times.
///
/// `total_collision_mom` is the four-momentum the products must add up to. `mass_scale` is
/// `TotalCollisionMom.m()`, CLHEP's signed root.
struct BlirCorrectorReport {
  bool ok = true;            ///< the function's own return value
  bool converged = false;    ///< `success`, which Geant4 only prints
  int attempts = 0;
  double final_scale = 0.0;
};

__host__ __device__ inline BlirCorrectorReport blir_energy_and_momentum_corrector(
    BlirProduct* out, int n_out, const LorentzVector& total_collision_mom) {
  BlirCorrectorReport rep;
  const int n_attempt_scale = 2500;
  const double err_limit = 1.0e-6;
  if (n_out == 0) { return rep; }

  LorentzVector sum_mom;
  double sum_mass = 0.0;
  const double total_collision_mass = total_collision_mom.mag();
  for (int i = 0; i < n_out; ++i) {
    sum_mom += out[i].momentum;
    sum_mass += out[i].pdg_mass;
  }
  if (sum_mass > total_collision_mass) {
    rep.ok = false;
    return rep;
  }
  const double m2 = sum_mom.e * sum_mom.e - g4gpu::mag2(sum_mom.v);
  if (m2 < 0.0) {
    rep.ok = false;
    return rep;
  }
  sum_mass = std::sqrt(m2);   // dead: nothing reads it after this

  // `G4ThreeVector Beta = -SumMom.boostVector(); ... mom *= Beta;` - and `mom *= Beta` with a
  // three-vector goes through `HepLorentzRotation`'s converting constructor, so it is the
  // rotation form and not `HepLorentzVector::boost`. See `rk_propagation.cuh`'s
  // `boost_by_rotation`.
  const Vec3d bv = sum_mom.boost_vector();
  const Vec3d beta{-bv.x, -bv.y, -bv.z};
  for (int i = 0; i < n_out; ++i) {
    out[i].momentum = boost_by_rotation(out[i].momentum, beta);
  }

  double scale = 0.0, old_scale = 0.0, factor = 1.0, sum = 0.0;
  int attempt = 0;
  for (attempt = 0; attempt < n_attempt_scale; ++attempt) {
    sum = 0.0;
    for (int i = 0; i < n_out; ++i) {
      LorentzVector h = out[i].momentum;
      h.v = h.v + (factor * scale) * h.v;
      const double e = std::sqrt(g4gpu::mag2(h.v) + out[i].pdg_mass * out[i].pdg_mass);
      h.e = e;
      out[i].momentum = h;
      sum += e;
    }
    old_scale = scale;
    scale = total_collision_mass / sum - 1.0;
    if (std::abs(scale) <= err_limit || old_scale == scale) {
      rep.converged = true;
      break;
    }
    if (attempt > 10) {
      const double f = std::log(std::abs(old_scale / (old_scale - scale)));
      factor = (f > 1.0) ? f : 1.0;
    }
  }
  rep.attempts = attempt;
  rep.final_scale = scale;

  const Vec3d back = total_collision_mom.boost_vector();
  for (int i = 0; i < n_out; ++i) {
    out[i].momentum = boost_by_rotation(out[i].momentum, back);
  }
  return rep;
}

/// The arithmetic of `GetProjectileExcitation`, for the agent that finishes the cascade arm.
///
/// Per HIT projectile nucleon: the local Fermi energy at its position minus its own kinetic
/// energy, summed. `localFermiEnergy = sqrt(m^2 + p_F(rho)^2) - m` with `rho` the projectile
/// nucleus's own density at that nucleon's position, and the nucleon's kinetic energy is
/// `GetMomentum().t() - GetMomentum().mag()` - its own off-shell mass, not the PDG mass. The
/// sum can be negative and `DeExciteSpectatorNucleus` clamps it with `std::max(0., ...)`.
///
/// It is a `__host__ __device__` function and not a refusal because it needs only a nucleon and
/// a nucleus, both of which this package has; what it does not have is a cascade to mark the
/// nucleons hit. Called by nothing until `Interact` exists.
__host__ __device__ inline double blir_projectile_excitation_term(const Nucleon& nucleon,
                                                                  const NuclearDensity& density,
                                                                  const FermiMomentum& fermi) {
  const double local_density = density.density(nucleon.position);
  const double local_pf = fermi.fermi_momentum(local_density);
  const double m = nucleon.pdg_mass();
  const double local_fermi_energy = std::sqrt(m * m + local_pf * local_pf) - m;
  return local_fermi_energy - (nucleon.momentum.e - nucleon.momentum.mag());
}

// =============================================================================================
// The entry point
// =============================================================================================


// ---------------------------------------------------------------------------------------------
// The CASCADE arm: G4BinaryLightIonReaction::Interact and the three functions that take its
// output apart again.
// ---------------------------------------------------------------------------------------------
//
// ## TWO NUCLEI, AND THE PROJECTILE'S NUCLEONS ARE THE SECONDARIES
//
// `Interact` builds a `G4Fancy3DNucleus` for the PROJECTILE as well as the target, centres it,
// picks an impact parameter in the disc of radius `tOuter + pOuter`, and hands
// `G4BinaryCascade::Propagate` one `G4KineticTrack` per projectile nucleon - each built from its
// own `G4Nucleon`, each `outside`, each with the SAME four-momentum `(0, 0, |p|/pA, E/pA)`. So
// the cascade this package already has runs unchanged; what is new is who the secondaries are.
//
// **That is also the whole of `SortResult`.** A projectile nucleon nothing hit comes back with
// `IsParticipant()` false (docs/RISK.md V163: false means "a nucleon that was not touched") and
// is a SPECTATOR; everything else is a cascader. The spectators are counted into (spectatorA,
// spectatorZ) and become a fragment for the de-excitation handler. A port that stored the hit
// flag on the track instead of on the nucleon, or that read the predicate the obvious way round,
// would sort every product into the wrong pile.
//
// ## THE NUCLEUS BUILDS SHARE ONE SCRATCH, AND THAT IS NOT AN OPTIMISATION
//
// `projectile3dNucleus->Init(pA,pZ)` and `target3dNucleus->Init(tA,tZ)` run one after the other
// on the same random stream, and `G4Fancy3DNucleus::ChooseFermiMomenta` draws through
// `CLHEP::RandGauss`, whose cached second value is a THREAD-LOCAL static that survives from one
// Init to the next (see `Nucleus3DScratch` in nucleus/fancy_3d_nucleus.cuh, note 5). Giving the
// two builds separate scratches puts the port half a Gaussian out of step on the second one.
//
// ## REFUSED, by name
//
//   * nothing new. `Propagate`'s own refusals travel out in `BlirRefusal::cascade_ref`, and
//     `Propagate1H1` cannot be reached from here: `Interact` always calls `Propagate`, whatever
//     the target's A, because `G4BinaryLightIonReaction` does not have the A == 1 branch
//     `G4BinaryCascade::ApplyYourself` has.

/// Every array the ion reaction needs and cannot allocate. `scratch` is shared by BOTH nucleus
/// builds on purpose; see the note above.
struct BlirStorage {
  Nucleon* projectile_nucleons = nullptr;
  Nucleon* target_nucleons = nullptr;
  Nucleus3DScratch scratch;
  double* proton_field = nullptr;
  double* neutron_field = nullptr;
  int field_capacity = 0;
  CascadeWorkspace cascade;
  /// `SortResult`'s two output vectors, and the de-excitation's own products.
  BlirProduct* spectators = nullptr;
  int spectator_capacity = 0;
  BlirProduct* cascaders = nullptr;
  int cascader_capacity = 0;
  /// `Interact`'s own secondary list - one track per projectile nucleon.
  ///
  /// Caller-owned like everything else here, and that is not only the rule: as a LOCAL of
  /// `blir_interact` it is 256 `CascadeTrack`s, 38 kilobytes, on the frame that then calls
  /// `propagate` - which has its own 256-entry `DecayTrack` and three 256-entry index arrays,
  /// and which calls the de-excitation below that. MEASURED: with it on the stack the test
  /// died with an access violation (0xC0000005) before it printed a line.
  CascadeTrack* initial = nullptr;
  int initial_capacity = 0;
};

/// `G4BinaryLightIonReaction::GetProjectileExcitation`.
///
/// The excitation the projectile remnant is left with, summed over the projectile nucleons that
/// WERE hit: for each one, the local Fermi energy at its own position minus the kinetic energy
/// it had in the nucleus. `aNuc->GetMomentum()` is the nucleus's own record and the cascade
/// never writes to it, so this reads the nucleon as it was built - the only thing the cascade
/// changed about it is the hit flag.
///
/// It can come out NEGATIVE, and `DeExciteSpectatorNucleus` clamps it with `std::max(0., ...)`
/// rather than here.
__host__ __device__ inline double blir_projectile_excitation(const Nucleus3D& proj) {
  double total = 0.0;
  for (int i = 0; i < proj.my_a; ++i) {
    const Nucleon& n = proj.nucleons[i];
    if (!n.hit) { continue; }
    const double density = proj.density.density(n.position);
    const double p_fermi = proj.fermi.fermi_momentum(density);
    const double mass = n.pdg_mass();
    const double e_fermi = std::sqrt(mass * mass + p_fermi * p_fermi) - mass;
    // `aNuc->GetMomentum().t() - aNuc->GetMomentum().mag()`: the nucleon's kinetic energy as the
    // nucleus stored it, with `mag()` its own invariant mass and not the PDG mass.
    total += e_fermi - (n.momentum.e - n.momentum.mag());
  }
  return total;
}

/// What one `Interact` did.
struct BlirInteractReport {
  bool ok = false;           ///< a non-empty product list came back
  int tries = 0;             ///< `tryCount`, at most 150
  int n_products = 0;
  double projectile_excitation = 0.0;
  deex::Vec3d position{0.0, 0.0, 0.0};   ///< the impact parameter of the LAST try
  double impact_max = 0.0;
  CascadeRefusal cascade_ref;
  NucleusReport projectile_rep;
  NucleusReport target_rep;
  PropagateResult propagate;
};

/// `G4BinaryLightIonReaction::Interact(mom, toBreit)`.
///
/// `proj` and `tgt` are the caller's two `Nucleus3D` objects; both are rebuilt on every try, as
/// Geant4 rebuilds both (it `new`s them inside the loop and deletes them again whenever the
/// result is empty). `de_excite` is `Propagate`'s exit into the precompound model.
///
/// `it = toBreit * G4LorentzVector(projectileMass, G4ThreeVector(0,0,0))` is computed on every
/// try and never read; `projectileMass` is only used to build it. Both are here in the comment
/// and not in the code, because a variable nothing reads is not part of the answer - but the
/// `GetIonMass(Z, A)` call behind it is, in the sense that a release which made it throw would
/// change this loop's behaviour.
template <typename Prop, typename Rng, typename DeExcite>
__host__ __device__ inline BlirInteractReport blir_interact(
    const BlirFrame& f, Nucleus3D& proj, Nucleus3D& tgt, Prop& propagator, BlirStorage& store,
    BicCascadeState& st, const CascadeSpecies& sp, Rng& rng, DeExcite&& de_excite,
    BlirRefusal& ref) {
  BlirInteractReport rep;
  CascadeTrack* initial = store.initial;

  int try_count = 0;
  do {
    ++try_count;
    proj.nucleons = store.projectile_nucleons;
    proj.capacity = store.scratch.capacity;
    rep.projectile_rep = nucleus_init(proj, store.scratch, f.pa, f.pz, rng);
    if (rep.projectile_rep.fatal()) {
      ref.nucleus = true;
      break;
    }
    proj.center_nucleons();

    tgt.nucleons = store.target_nucleons;
    tgt.capacity = store.scratch.capacity;
    rep.target_rep = nucleus_init(tgt, store.scratch, f.ta, f.tz, rng);
    if (rep.target_rep.fatal()) {
      ref.nucleus = true;
      break;
    }

    const double impact_max = tgt.outer_radius() + proj.outer_radius();
    rep.impact_max = impact_max;
    const double ax = (2.0 * rng.uniform() - 1.0) * impact_max;
    const double ay = (2.0 * rng.uniform() - 1.0) * impact_max;
    // "-2.*impactMax-5.*fermi": far enough upstream that every nucleon starts outside.
    const deex::Vec3d pos{ax, ay, -2.0 * impact_max - 5.0 * deex::fermi()};
    rep.position = pos;

    // `G4LorentzVector nucleonMom(1./pA*mom); nucleonMom.setZ(nucleonMom.vect().mag());
    //  nucleonMom.setX(0); nucleonMom.setY(0);` - the order is load-bearing: `setZ` runs FIRST,
    // so the magnitude it stores is that of the whole scaled three-vector, and only then are x
    // and y zeroed. Every projectile nucleon gets this same four-momentum.
    const double inv_pa = 1.0 / static_cast<double>(f.pa);
    const deex::Vec3d scaled = inv_pa * f.mom.v;
    const imr::LorentzVector nucleon_mom(deex::Vec3d{0.0, 0.0, std::sqrt(g4gpu::mag2(scaled))},
                                         inv_pa * f.mom.e);

    int n_initial = 0;
    for (int i = 0; i < proj.my_a && n_initial < store.initial_capacity; ++i) {
      const Nucleon& n = proj.nucleons[i];
      CascadeTrack t;
      t.pdg = n.pdg();
      t.pdg_mass = n.pdg_mass();
      t.charge = n.charge();
      t.baryon = 1;
      t.momentum = nucleon_mom;
      t.position = n.position + pos;
      t.formation_time = 0.0;
      t.state = kOutside;
      t.nucleon_index = i;
      t.nucleon_owner = 1;       ///< the PROJECTILE's nucleon; see `CascadeTrack::nucleon_owner`
      t.creator_model_id = blir_model_id();
      // `SetProjectilePotential(-Efermi)` with the Fermi energy at the nucleon's own position in
      // the PROJECTILE, before the impact parameter is added.
      const double density = proj.density.density(n.position);
      const double p_fermi = proj.fermi.fermi_momentum(density);
      const double e_fermi =
          std::sqrt(t.pdg_mass * t.pdg_mass + p_fermi * p_fermi) - t.pdg_mass;
      t.projectile_potential = -e_fermi;
      initial[n_initial++] = t;
    }
    if (n_initial != proj.my_a) {
      ref.capacity = true;
      break;
    }

    // `thePropagator->Init(the3DNucleus)` happens inside Propagate in Geant4; the port builds
    // the field maps here because they need the caller's tables.
    NucleusReport frep;
    propagator = make_rk_propagation(tgt, frep, store.proton_field, store.neutron_field,
                                     store.field_capacity);
    CascadeRefusal cref;
    st.projectile_nucleons = proj.nucleons;
    const PropagateResult pr =
        propagate(st, tgt, propagator, sp, store.cascade, initial, n_initial, tgt.density,
                  coulomb_barrier_mev(f.ta, f.tz), de_excite, rng, cref);
    rep.propagate = pr;
    rep.cascade_ref = cref;
    if (cref.any()) {
      ref.cascade = true;
      ref.refused_pdg = cref.refused_pdg;
      break;
    }
    // "if( result && result->size()==0) { delete result; result=0; }" - an empty vector is a
    // NULL here, so both of `Propagate`'s failure returns send this loop round again. That is
    // the one place `G4BinaryLightIonReaction` does NOT distinguish them.
    if (pr.outcome != kPropagateNoCollision && pr.n_products > 0) {
      rep.ok = true;
      rep.n_products = pr.n_products;
    }
  } while (!rep.ok && try_count < 150);

  rep.tries = try_count;
  if (rep.ok) { rep.projectile_excitation = blir_projectile_excitation(proj); }
  return rep;
}

/// `G4BinaryLightIonReaction::SortResult`.
///
/// Splits the cascade's products on `GetNewlyAdded()`, which `ProductsAddFinalState` set from
/// `IsParticipant()`. Returns the spectators' summed four-momentum and fills `spectator_a` and
/// `spectator_z`; `p_final_state` receives the cascaders' sum, which `ApplyYourself` reads.
///
/// `spectatorA` counts one per spectator PRODUCT, not its baryon number - every spectator is a
/// single projectile nucleon that was never touched, so the two agree, and the source counts
/// products.
__host__ __device__ inline imr::LorentzVector blir_sort_result(
    const CascadeProduct* result, int n_result, BlirProduct* spectators, int spectator_capacity,
    BlirProduct* cascaders, int cascader_capacity, int& n_spectators, int& n_cascaders,
    int& spectator_a, int& spectator_z, imr::LorentzVector& p_final_state, BlirRefusal& ref) {
  imr::LorentzVector p_spectators;
  p_final_state = imr::LorentzVector();
  n_spectators = 0;
  n_cascaders = 0;
  spectator_a = 0;
  spectator_z = 0;
  for (int i = 0; i < n_result; ++i) {
    BlirProduct p;
    p.momentum = result[i].momentum;
    p.pdg_mass = result[i].pdg_mass;
    p.pdg = result[i].pdg;
    p.z = result[i].nucleus_z;
    p.a = result[i].nucleus_a;
    p.newly_added = result[i].newly_added;
    if (result[i].newly_added) {
      if (n_cascaders >= cascader_capacity) {
        ref.capacity = true;
        return p_spectators;
      }
      p_final_state = p_final_state + p.momentum;
      cascaders[n_cascaders++] = p;
    } else {
      if (n_spectators >= spectator_capacity) {
        ref.capacity = true;
        return p_spectators;
      }
      p_spectators = p_spectators + p.momentum;
      spectators[n_spectators++] = p;
      ++spectator_a;
      // `G4lrint(GetDefinition()->GetPDGCharge()/eplus)` - the real charge, so a spectator
      // proton counts and a spectator neutron does not.
      spectator_z += (result[i].pdg == imr::kPdgProton) ? 1 : 0;
    }
  }
  return p_spectators;
}

/// `G4BinaryLightIonReaction::DeExciteSpectatorNucleus`.
///
/// The projectile remnant. Three things about it are worth naming:
///
///   * the fragment is built with **zero particles and zero charged excitons** and
///     `holes = pA - spectatorA`, and its four-momentum is AT REST:
///     `(0, 0, 0, mFragment + max(0, theStatisticalExEnergy))`. So the excitation
///     `GetProjectileExcitation` computed is the whole of what the handler is told, the
///     spectators' own momenta are thrown away, and the products are boosted back afterwards by
///     `pSpectators.boostVector()`.
///   * it is the EXCITATION HANDLER, `theHandler->BreakItUp`, and not the precompound model -
///     which is why the exciton counts are zero and why this is P3's entry point and not P6's.
///   * when the spectators cannot make a fragment - `spectatorZ == 0` or `spectatorA == 1` -
///     they are not de-excited at all: each one is marked `SetNewlyAdded(true)` and pushed onto
///     the CASCADERS, and its momentum joins `pFinalState`. A single spectator neutron leaves
///     the reaction as a neutron.
///
/// Then the cascaders are corrected twice: once against `pInitialState - pFragments`, and again
/// against `pInitialState` if the first did not converge. The de-excitation products are appended
/// AFTER the first correction and are not themselves corrected.
template <typename Rng>
__host__ __device__ inline void blir_deexcite_spectator(
    BlirProduct* spectators, int n_spectators, BlirProduct* cascaders, int& n_cascaders,
    int cascader_capacity, int spectator_a, int spectator_z, int pa,
    double statistical_ex_energy, const imr::LorentzVector& p_spectators,
    const imr::LorentzVector& p_initial_state, imr::LorentzVector& p_final_state,
    const data::LevelTable& lt, const deex::FermiPool& pool, const deex::DeexWorkspace& dws,
    Rng& rng, deex::DeexStatus& dstatus, BlirRefusal& ref, BlirCorrectorReport& last) {
  int n_frag = 0;
  imr::LorentzVector p_fragments;
  const int first_frag = n_cascaders;   // where the de-excitation products will go

  if (spectator_z > 0 && spectator_a > 1) {
    deex::Fragment pro_res;
    const double m_fragment = deex::nuclear_mass(spectator_a, spectator_z);
    const double e = m_fragment + ((statistical_ex_energy > 0.0) ? statistical_ex_energy : 0.0);
    pro_res.set_za_and_momentum(imr::LorentzVector(deex::Vec3d{0.0, 0.0, 0.0}, e), spectator_z,
                                spectator_a);
    // SetNumberOfParticles(0), SetNumberOfCharged(0), SetNumberOfHoles(pA-spectatorA): the
    // handler reads none of the three, and they are written out because the fragment Geant4
    // hands over carries them.
    (void)pa;
    dstatus = deex::deexcite(pro_res, lt, pool, dws, rng);
    n_frag = dstatus.n_products;
  } else if (spectator_a != 0) {
    for (int i = 0; i < n_spectators; ++i) {
      if (n_cascaders >= cascader_capacity) {
        ref.capacity = true;
        return;
      }
      BlirProduct p = spectators[i];
      p.newly_added = true;
      p_final_state = p_final_state + p.momentum;
      cascaders[n_cascaders++] = p;
    }
  }

  // The de-excitation products, boosted into the spectators' frame. `boost_fragments` is
  // `G4LorentzRotation(pSpectators.boostVector())` - the forward boost, not its inverse.
  if (n_frag > 0) {
    const imr::LorentzRotation boost =
        imr::LorentzRotation::from_boost(p_spectators.boost_vector());
    for (int i = 0; i < n_frag; ++i) {
      const deex::DeexProduct& d = dws.products[i];
      BlirProduct p;
      p.pdg = d.pdg;
      p.z = d.z;
      p.a = d.a;
      p.newly_added = true;
      p.pdg_mass = (d.a > 0) ? (deex::nuclear_mass(d.a, d.z) + d.excitation)
                             : ((d.pdg == deex::kPdgElectron)
                                    ? u::electron_mass_c2<double>() : 0.0);
      p.momentum = boost * d.momentum;
      p_fragments = p_fragments + p.momentum;
      if (first_frag + i >= cascader_capacity) {
        ref.capacity = true;
        return;
      }
      // Held back: they are appended AFTER the correction below, which is the source's order.
      cascaders[first_frag + i] = p;
    }
  }

  // "the creation of excited fragment did violate E/p, so correct cascaders to get overall
  // conservation" - on the cascaders ONLY, with the de-excitation products not yet in the list.
  const imr::LorentzVector p_cas = p_initial_state - p_fragments;
  const BlirCorrectorReport r1 = blir_energy_and_momentum_corrector(cascaders, n_cascaders,
                                                                    p_cas);
  last = r1;
  n_cascaders += n_frag;
  if (!r1.ok) {
    last = blir_energy_and_momentum_corrector(cascaders, n_cascaders, p_initial_state);
  }
}

/// The `HadFinalState` width this model instantiates. A 50 MeV/nucleon C12 on Pb208 fuses into
/// a compound of A = 220 and E* of a few hundred MeV, and P3's evaporation cascade on that emits
/// tens of fragments and gammas; 128 is generous and the overflow is reported, never silent.
inline constexpr int kBlirMaxSecondaries = 128;
using BlirFinalState = physics::hadronic::HadFinalState<double, kBlirMaxSecondaries>;

/// `G4BinaryLightIonReaction::ApplyYourself`, both arms.
///
/// `projectile` and `target` are P5's shapes. The projectile's four-momentum is reconstructed as
/// `(0, 0, sqrt(T(T+2m)), T+m)` because that is what `G4HadProjectile::Get4Momentum()` returns -
/// already rotated into +z, which is what makes this model's rotate-to-lab block the identity
/// (see the file header and docs/RISK.md V72).
///
/// Three outcomes, and they are not interchangeable:
///
///   * **kinetic energy per nucleon at or above 50 MeV**: `Interact` - the 150-try loop, the
///     projectile nucleus whose nucleons become `outside` tracks at a sampled impact parameter,
///     the same `propagate`, then `SortResult`, the correction loop and
///     `DeExciteSpectatorNucleus`. Until P9e this was `ref.cascade` and an empty final state.
///   * **below 50 MeV/nucleon and the nuclei cannot fuse**: `ref.no_fusion`, and the final state
///     is the PRIMARY UNCHANGED - `isAlive`, its own kinetic energy, its own direction. That is
///     Geant4's answer and it is the only branch of this model that does not kill the primary.
///   * **below 50 MeV/nucleon and they fuse**: `stopAndKill` and P6's whole product list, with
///     the swap undone and mirrored if `SetLighterAsProjectile` swapped.
/// What the cascade arm did, for a caller that wants to see the loop counts rather than only
/// the products. Every field is zero on the fusion arm.
struct BlirReport {
  bool cascade_arm = false;
  int interact_tries = 0;      ///< `tryCount`, at most 150
  int spectator_a = 0;
  int spectator_z = 0;
  int n_cascaders = 0;
  int n_spectators = 0;
  double projectile_excitation = 0.0;
  int correction_loops = 0;    ///< `loopcount` in the E/p loop, at most 11
  bool correction_gave_up = false;
  /// The LAST `EnergyAndMomentumCorrector` call of the event - the one that decides what the
  /// final state adds up to - and it is here because the ion arm does NOT conserve energy
  /// exactly and cannot be asserted as if it did.
  ///
  /// `EnergyAndMomentumCorrector` is a FIXED-POINT iteration with `ErrLimit = 1.E-6` - see
  /// docs/RISK.md V184 for the whole of it - and
  /// `Scale = TotalCollisionMass/Sum - 1` at exit is the relative error still left in the
  /// products' centre-of-mass energy. It has two exits: `|Scale| <= ErrLimit`, and
  /// `OldScale == Scale` - an exact double comparison that stops a frozen iteration wherever it
  /// froze, with no bound on the error at all. It returns TRUE either way and Geant4 only
  /// prints the difference under a debug flag. So the energy the event is short by is
  /// `|Scale|` times the invariant mass the products were corrected to, and that is a quantity
  /// a test can ASSERT - which is the difference between reproducing Geant4 and losing energy.
  ///
  /// MEASURED on the campaign: an alpha at 50 MeV/nucleon on Fe56 leaves 0.0529 MeV on a
  /// converged event, against `1e-6 * 55,800 MeV = 0.0558`; the worst event of that case is
  /// 2.11 MeV, which is a frozen exit.
  /// Whether that last corrector call happened AT ALL. It does not when `spectatorA == 0`:
  /// Geant4 guards `DeExciteSpectatorNucleus` with `if (spectatorA > 0)`, so an event whose
  /// projectile cascaded entirely away is left wherever the while loop above put it - and that
  /// loop exits on `|momentum.e() - pspectators.e()| <= 10*MeV`, so TEN MEV is the bound on
  /// those events and there is nothing sharper to say about them.
  /// `gamma` of `boost_fragments`, the boost `DeExciteSpectatorNucleus` applies to the
  /// spectator fragment's de-excitation products - `G4LorentzRotation(pSpectators.boostVector())`.
  ///
  /// It is reported because a test comparing this arm's energy balance has to know what the
  /// spectator's products were boosted by. It is NOT the explanation of V76's conversion
  /// electron here, which is what it was added for: on the nucleon path
  /// `G4PhotonEvaporation::GenerateGamma` CREATES one electron rest mass per internal
  /// conversion and a test subtracts it flat, and the first guess was that this arm sees
  /// `gamma * m_e` instead. It does not. MEASURED on ic_a50_Fe56 ev 10628: the surplus is not
  /// scaled by gamma (1.05 there), it is ABSORBED - every de-excitation product of this arm
  /// goes through `EnergyAndMomentumCorrector`, which rescales the cascaders until the total
  /// matches `pInitialState - pFragments`. So the ion event balances against its initial energy
  /// with no electron term at all, and subtracting one is what creates a discrepancy of exactly
  /// `m_e`. The wrong guess is written down because the right answer is only interesting
  /// against it.
  double spectator_gamma = 1.0;
  bool last_correction_ran = false;
  bool last_correction_converged = false;
  double last_correction_scale = 0.0;
  int last_correction_attempts = 0;
  PropagateResult propagate;
};

/// **This lives here and not in `binary_cascade.cuh` because BOTH entry points need it and
/// that file includes this one.** `G4BinaryCascade::ApplyYourself` reaches it through its own
/// `Propagate`, and `G4BinaryLightIonReaction::Interact` reaches the same `Propagate` with the
/// projectile nucleons as its secondaries. Putting it in `cascade_propagate.cuh` instead would
/// make that file depend on P6, and the whole point of the de-excitation being a parameter
/// there is that it does not.
/// The cascade's exit into the precompound model - `theDeExcitation->DeExcite(*fragment)` with
/// `G4Fragment(a, z, GetFinalNucleusMomentum())` and the three exciton counters FindFragments set.
///
/// `__noinline__` for the reason docs/RISK.md V55 gives and one more that is specific here: P6's
/// `deexcite` drives P3's whole evaporation cascade, and inlining it into `propagate` - which is
/// already the deepest frame in this package - puts both stack frames live at once for the entire
/// cascade loop, when the de-excitation runs exactly once and at the very end.
template <typename Rng>
__host__ __device__ __noinline__ int bic_deexcite_fragment(
    const CascadeFragment& frag, CascadeProduct* out, int capacity, const data::LevelTable& lt,
    const deex::FermiPool& pool, const preco::PrecoWorkspace& ws, Rng& rng,
    preco::PrecoStatus& status, bool& overflow) {
  deex::Fragment f;
  f.set_za_and_momentum(frag.momentum, frag.z, frag.a);
  preco::Excitons ex;
  ex.particles = frag.particles;
  ex.charged = frag.charged;
  ex.holes = frag.holes;
  status = preco::deexcite(f, ex, lt, pool, ws, rng);
  int n = 0;
  for (int i = 0; i < status.n_products; ++i) {
    if (n >= capacity) {
      overflow = true;
      break;
    }
    const deex::DeexProduct& p = ws.products[i];
    out[n] = CascadeProduct{};
    out[n].pdg = p.pdg;
    // The same two-branch mass rule `deex::deex_kinetic_energy` uses: for a nucleus the table
    // mass plus whatever excitation P3 left on it, for A = 0 the electron's mass or nothing.
    const double pdg_mass =
        (p.a > 0) ? (deex::nuclear_mass(p.a, p.z) + p.excitation)
                  : ((p.pdg == deex::kPdgElectron) ? u::electron_mass_c2<double>() : 0.0);
    // P3 hands back a full four-momentum, so there is nothing to rebuild - only the definition
    // mass to record alongside it.
    out[n].momentum = p.momentum;
    // `G4PreCompoundModel` products carry their own creator ids, which P3 does not keep; -1 says
    // so rather than claiming theBIC_ID for something the cascade did not emit.
    out[n].creator_model_id = -1;
    out[n].nucleus_z = p.z;
    out[n].nucleus_a = p.a;
    out[n].pdg_mass = pdg_mass;
    ++n;
  }
  return n;
}

/// `G4BinaryLightIonReaction::ApplyYourself`'s cascade branch, from `Interact` to the secondary
/// list. It is a separate function only because the fusion branch is already long; Geant4 has it
/// all in one.
///
/// ## THE CORRECTION LOOP IS ON THE ENERGY AND THE EXIT IS ON THE MOMENTUM
///
///     while (std::abs(momentum.e()-pspectators.e()) > 10*MeV)
///     {  pCorrect = pInitialState - pspectators;
///        EnergyAndMomentumCorrector(cascaders, pCorrect);
///        pFinalState = sum over cascaders;
///        momentum = pInitialState - pFinalState;
///        if (++loopcount > 10) {
///           if (momentum.vect().mag() - momentum.e() > 10*keV) throw;
///           else break; } }
///
/// The `while` tests the ENERGY against 10 MeV and the give-up test inside it compares a
/// three-momentum magnitude against an energy - `|p| - E`, which for a timelike remnant is
/// negative and for a spacelike one is positive. So the loop runs until the cascaders carry the
/// energy the spectators left them, and gives up after eleven passes; the throw is for the case
/// where what is left over is spacelike, and a kernel cannot throw, so it is
/// `BlirRefusal::momentum_not_conserved` and the primary comes back alive - which is also what
/// the `spectatorA > 0` branch below does with the same test.
///
/// ## `pInitialState` IS THE PROJECTILE PLUS THE TARGET'S MASS AND NOTHING ELSE
///
///     pInitialState = mom;
///     pInitialState.setT(pInitialState.getT() + GetIonMass(tZ,tA));
///
/// The target's REST MASS is added to the energy and not as a four-vector, which is the same
/// thing for a target at rest and is what makes the corrector's target four-momentum exact.
template <typename Rng>
__host__ __device__ inline preco::PrecoStatus blir_cascade_arm(
    const physics::hadronic::HadProjectile<double>& projectile, const BlirFrame& f,
    const data::LevelTable& lt, const deex::FermiPool& pool, const preco::PrecoWorkspace& ws,
    BlirStorage& store, Rng& rng, BlirFinalState& result, BlirRefusal& ref, BlirReport& rep) {
  preco::PrecoStatus status;
  rep.cascade_arm = true;

  Nucleus3D proj;
  Nucleus3D tgt;
  RkPropagation propagator;
  BicCascadeState st;
  CascadeSpecies sp;
  sp.proton_mass = u::proton_mass_c2<double>();
  sp.neutron_mass = u::neutron_mass_c2<double>();
  sp.pi_plus_mass = pdg_mass_pion_charged();
  sp.pi_zero_mass = pdg_mass_pion_zero();

  bool preco_overflow = false;
  auto de = [&](const CascadeFragment& frag, CascadeProduct* out, int capacity) {
    return bic_deexcite_fragment(frag, out, capacity, lt, pool, ws, rng, status, preco_overflow);
  };

  const BlirInteractReport ir =
      blir_interact(f, proj, tgt, propagator, store, st, sp, rng, de, ref);
  rep.interact_tries = ir.tries;
  rep.propagate = ir.propagate;
  rep.projectile_excitation = ir.projectile_excitation;
  ref.cascade_ref = ir.cascade_ref;
  if (preco_overflow) { ref.capacity = true; }
  if (ref.any()) { return status; }
  if (!ir.ok) {
    // "G4BinaryLightIonReaction no final state for:" - 150 impact parameters and nothing came
    // back. Geant4 prints and returns the primary unchanged.
    ref.no_final_state = true;
    result.status = physics::hadronic::HadFinalStateStatus::kIsAlive;
    result.energy_change = projectile.kin_energy;
    result.momentum_change = Vec3<double>{0.0, 0.0, 1.0};
    return status;
  }

  imr::LorentzVector p_initial_state = f.mom;
  p_initial_state.e += deex::nuclear_mass(f.ta, f.tz);

  imr::LorentzVector p_final_state;
  int n_spectators = 0, n_cascaders = 0, spectator_a = 0, spectator_z = 0;
  const imr::LorentzVector p_spectators = blir_sort_result(
      store.cascade.products, ir.n_products, store.spectators, store.spectator_capacity,
      store.cascaders, store.cascader_capacity, n_spectators, n_cascaders, spectator_a,
      spectator_z, p_final_state, ref);
  rep.spectator_a = spectator_a;
  {
    const double m_spec = p_spectators.mag();
    rep.spectator_gamma = (m_spec > 0.0) ? (p_spectators.e / m_spec) : 1.0;
  }
  rep.spectator_z = spectator_z;
  rep.n_spectators = n_spectators;
  if (ref.any()) { return status; }

  imr::LorentzVector momentum = p_initial_state - p_final_state;
  int loopcount = 0;
  while (std::fabs(momentum.e - p_spectators.e) > 10.0 * u::MeV<double>()) {
    const imr::LorentzVector p_correct = p_initial_state - p_spectators;
    blir_energy_and_momentum_corrector(store.cascaders, n_cascaders, p_correct);
    p_final_state = imr::LorentzVector();
    for (int i = 0; i < n_cascaders; ++i) { p_final_state = p_final_state + store.cascaders[i].momentum; }
    momentum = p_initial_state - p_final_state;
    if (++loopcount > 10) {
      if (std::sqrt(g4gpu::mag2(momentum.v)) - momentum.e > 10.0 * u::keV<double>()) {
        // `throw G4HadronicException(... "G4BinaryCasacde::ApplyCollision()")` - the typo and
        // the wrong function name are Geant4's. A kernel cannot throw.
        ref.momentum_not_conserved = true;
        result.status = physics::hadronic::HadFinalStateStatus::kIsAlive;
        result.energy_change = projectile.kin_energy;
        result.momentum_change = Vec3<double>{0.0, 0.0, 1.0};
        rep.correction_loops = loopcount;
        rep.correction_gave_up = true;
        return status;
      }
      break;
    }
  }
  rep.correction_loops = loopcount;

  if (spectator_a > 0) {
    if (std::sqrt(g4gpu::mag2(momentum.v)) - momentum.e < 10.0 * u::keV<double>()) {
      deex::DeexStatus dstatus;
      BlirCorrectorReport last;
      blir_deexcite_spectator(store.spectators, n_spectators, store.cascaders, n_cascaders,
                              store.cascader_capacity, spectator_a, spectator_z, f.pa,
                              ir.projectile_excitation, p_spectators, p_initial_state,
                              p_final_state, lt, pool, ws.deex, rng, dstatus, ref, last);
      rep.last_correction_ran = true;
      rep.last_correction_converged = last.converged;
      rep.last_correction_scale = last.final_scale;
      rep.last_correction_attempts = last.attempts;
      status.deex = dstatus;
      if (ref.any()) { return status; }
    } else {
      // "G4BinaryLightIonReaction invalid final state for:" - the primary, unchanged.
      ref.momentum_not_conserved = true;
      result.status = physics::hadronic::HadFinalStateStatus::kIsAlive;
      result.energy_change = projectile.kin_energy;
      result.momentum_change = Vec3<double>{0.0, 0.0, 1.0};
      return status;
    }
  }
  rep.n_cascaders = n_cascaders;

  // `toZ.rotateZ(-mom.phi()); toZ.rotateY(-mom.theta()); toLab = toZ.inverse()` - the identity
  // for a projectile already along +z, which is what P5 hands over. See the file header.
  result.status = physics::hadronic::HadFinalStateStatus::kStopAndKill;
  const deex::Vec3d inv_boost{-f.breit_boost.x, -f.breit_boost.y, -f.breit_boost.z};
  for (int i = 0; i < n_cascaders; ++i) {
    // `if((*iter)->GetNewlyAdded())` - and ONLY those. A spectator that was de-excited is gone,
    // and one that could not be is in this list with the flag set.
    if (!store.cascaders[i].newly_added) { continue; }
    imr::LorentzVector q = store.cascaders[i].momentum;
    if (f.swapped) {
      q = boost_by_rotation(q, inv_boost);
      q.v = deex::Vec3d{-q.v.x, -q.v.y, -q.v.z};
    }
    physics::hadronic::HadSecondary<double> s;
    s.pdg = store.cascaders[i].pdg;
    s.z = store.cascaders[i].z;
    s.a = store.cascaders[i].a;
    s.time = 0.0;   // `G4double time = 0;` with the creation time commented out
    s.weight = 1.0;
    s.creator_model_id = blir_model_id();
    capture::set_four_momentum(s, q, store.cascaders[i].pdg_mass);
    if (!result.add_secondary(s)) {
      ref.capacity = true;
      break;
    }
  }
  return status;
}


template <typename Rng>
__host__ __device__ inline preco::PrecoStatus blir_apply_yourself(
    const physics::hadronic::HadProjectile<double>& projectile,
    const physics::hadronic::HadNucleus& target, const data::LevelTable& lt,
    const deex::FermiPool& pool, const preco::PrecoWorkspace& ws, BlirStorage& store,
    Rng& rng, BlirFinalState& result, BlirRefusal& ref, BlirReport& rep) {
  preco::PrecoStatus status;
  result.clear();

  if (projectile.baryon_number < 1 || target.l != 0) {
    ref.anti_or_hyper = true;
    ref.refused_pdg = projectile.pdg;
    return status;
  }

  const double m = projectile.mass;
  const double t = projectile.kin_energy;
  const LorentzVector p4(Vec3d{0.0, 0.0, std::sqrt(t * (t + 2.0 * m))}, t + m);

  const BlirFrame f = blir_set_lighter_as_projectile(
      projectile.baryon_number, static_cast<int>(std::lrint(projectile.charge)), target.a,
      target.z, p4);

  const double kin_per_nucleon = blir_kin_per_nucleon(f);
  ref.refused_kin_per_nucleon = kin_per_nucleon;
  if (!(kin_per_nucleon < blir_fusion_threshold_per_nucleon())) {
    return blir_cascade_arm(projectile, f, lt, pool, ws, store, rng, result, ref, rep);
  }

  if (!blir_fuse_nuclei_and_precompound(f, lt, pool, ws, rng, status, ref)) {
    // "abort!! happens for too low energy for nuclei to fuse": isAlive, unchanged.
    result.status = physics::hadronic::HadFinalStateStatus::kIsAlive;
    result.energy_change = projectile.kin_energy;
    result.momentum_change = Vec3<double>{0.0, 0.0, 1.0};
    return status;
  }

  result.status = physics::hadronic::HadFinalStateStatus::kStopAndKill;

  // The swap undone: `tmp *= toBreit.inverse(); tmp.setVect(-tmp.vect());`. The inverse of a
  // pure boost by beta is a boost by -beta, and the mirror is applied AFTER it.
  const Vec3d inv_boost{-f.breit_boost.x, -f.breit_boost.y, -f.breit_boost.z};

  for (int i = 0; i < status.n_products; ++i) {
    const deex::DeexProduct& p = ws.products[i];
    LorentzVector q = p.momentum;
    if (f.swapped) {
      q = boost_by_rotation(q, inv_boost);
      q.v = Vec3d{-q.v.x, -q.v.y, -q.v.z};
    }
    // `tmp *= toLab` is the identity - see the file header.
    physics::hadronic::HadSecondary<double> s;
    s.pdg = p.pdg;
    s.z = p.z;
    s.a = p.a;
    s.time = 0.0;   // `G4double time = 0;` with the creation time commented out
    s.weight = 1.0;
    s.creator_model_id = blir_model_id();
    // The mass the emitting model used, for FillResult's own 1 keV shell test. For a nucleus
    // it is `G4NucleiProperties::GetNuclearMass` plus the excitation P3 left on it; for one of
    // the eight fixed species it is the PDG mass. `deex::nuclear_mass` is both, because the
    // light branch of that function IS the PDG mass.
    //
    // **A = 0 is two species, not one**, and the electron branch is written out even though it
    // changes no number today. P3's photon evaporation emits a gamma AND a conversion electron
    // and both carry `a == 0`; only the gamma is massless, and Geant4 hands
    // `G4DynamicParticle(definition, totalEnergy, momentum)` the electron's own definition.
    // Passing 0 here still produced 0.511 MeV, because `set_four_momentum`'s middle branch
    // takes `|PDGmass^2 - mass2| > EnergyMRA2` and recovers `sqrt(mass2)` from the
    // four-momentum - which for P3's electron is exactly `m_e`, since it builds the momentum as
    // `sqrt((E-m_e)(E+m_e))`. Measured: `tests/test_bic_apply.cu` is byte-identical either way.
    // So this is not a fix; it is the same rule `deex::deex_kinetic_energy` has had since P3,
    // written where a reader looks, so that a product whose four-momentum is ever off shell
    // does not silently change species mass.
    const double pdg_mass =
        (p.a > 0) ? (deex::nuclear_mass(p.a, p.z) + p.excitation)
                  : ((p.pdg == deex::kPdgElectron) ? u::electron_mass_c2<double>() : 0.0);
    capture::set_four_momentum(s, q, pdg_mass);
    if (!result.add_secondary(s)) {
      ref.capacity = true;
      break;
    }
  }
  return status;
}

}  // namespace g4gpu::bic

#endif
