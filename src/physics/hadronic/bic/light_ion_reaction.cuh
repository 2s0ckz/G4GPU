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
// That arm is complete, and `tests/test_bic_ion.cu` is where it is checked against
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
// model a projectile NOT along +z would need it. docs/RISK.md V75.
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
  bool cascade = false;
  bool no_fusion = false;      ///< the kinematic gate failed; the primary is returned unchanged
  bool capacity = false;       ///< the caller's product or secondary buffer
  bool anti_or_hyper = false;  ///< a negative baryon number or a lambda
  int refused_pdg = 0;
  double refused_kin_per_nucleon = 0.0;

  __host__ __device__ bool any() const {
    return cascade || no_fusion || capacity || anti_or_hyper;
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

/// The `HadFinalState` width this model instantiates. A 50 MeV/nucleon C12 on Pb208 fuses into
/// a compound of A = 220 and E* of a few hundred MeV, and P3's evaporation cascade on that emits
/// tens of fragments and gammas; 128 is generous and the overflow is reported, never silent.
inline constexpr int kBlirMaxSecondaries = 128;
using BlirFinalState = physics::hadronic::HadFinalState<double, kBlirMaxSecondaries>;

/// `G4BinaryLightIonReaction::ApplyYourself`, with the cascade arm refused.
///
/// `projectile` and `target` are P5's shapes. The projectile's four-momentum is reconstructed as
/// `(0, 0, sqrt(T(T+2m)), T+m)` because that is what `G4HadProjectile::Get4Momentum()` returns -
/// already rotated into +z, which is what makes this model's rotate-to-lab block the identity
/// (see the file header and docs/RISK.md V75).
///
/// Three outcomes, and they are not interchangeable:
///
///   * **kinetic energy per nucleon at or above 50 MeV**: `ref.cascade` and an EMPTY final
///     state. Geant4 would have run a cascade; this package has none, and returning the fusion
///     answer for an energy where the cascade runs would be an approximation rather than a
///     refusal.
///   * **below 50 MeV/nucleon and the nuclei cannot fuse**: `ref.no_fusion`, and the final state
///     is the PRIMARY UNCHANGED - `isAlive`, its own kinetic energy, its own direction. That is
///     Geant4's answer and it is the only branch of this model that does not kill the primary.
///   * **below 50 MeV/nucleon and they fuse**: `stopAndKill` and P6's whole product list, with
///     the swap undone and mirrored if `SetLighterAsProjectile` swapped.
template <typename Rng>
__host__ __device__ inline preco::PrecoStatus blir_apply_yourself(
    const physics::hadronic::HadProjectile<double>& projectile,
    const physics::hadronic::HadNucleus& target, const data::LevelTable& lt,
    const deex::FermiPool& pool, const preco::PrecoWorkspace& ws, Rng& rng,
    BlirFinalState& result, BlirRefusal& ref) {
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
    ref.cascade = true;
    ref.refused_pdg = projectile.pdg;
    return status;
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
    // `sqrt((E-m_e)(E+m_e))`. Measured: `tests/test_bic_ion.cu` is byte-identical either way.
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
