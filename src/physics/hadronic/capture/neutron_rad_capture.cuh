// G4NeutronRadCapture - the only capture final state QBBC registers, and the whole of
// `nCapture`'s physics.
//
// Transcribed from Geant4 11.1.1,
//   processes/hadronic/models/de_excitation/photon_evaporation/src/G4NeutronRadCapture.cc
//     G4NeutronRadCapture::G4NeutronRadCapture   (the two energy constants)
//     G4NeutronRadCapture::InitialiseModel       (which of them the parameters overwrite)
//     G4NeutronRadCapture::ApplyYourself         (both branches)
//   models/de_excitation/management/include/G4VEvaporationChannel.hh
//     BreakUpFragment  - `new G4FragmentVector(); BreakUpChain(results, nucleus); return it`
//   models/de_excitation/photon_evaporation/src/G4PhotonEvaporation.cc
//     BreakUpChain     - `do { gamma = GenerateGamma(nucleus); ... } while(gamma)`
//
// THE MODEL IS TWO MODELS, SPLIT ON THE COMPOUND'S MASS NUMBER
//
// `ApplyYourself` increments A first and then tests `A <= 4`, so the split is on the COMPOUND
// and not on the target: a target with A of 1, 2 or 3 takes the "simplified method of 1 gamma
// emission" and everything from A = 4 up (He4 + n -> He5, and every heavier nuclide) goes
// through photon evaporation. In water that is exactly the two hydrogen isotopes on one side
// and the three oxygen isotopes on the other, which is why both branches matter for the one
// material this project measures a dose in.
//
//   A <= 4   two-body kinematics, one gamma. The compound is never formed as a G4Fragment and
//            no level scheme is consulted: the gamma takes `e1 = (M-mass)(M+mass)/(2M)` in the
//            centre of mass - which is exactly the energy that leaves the residual with
//            invariant mass `mass` - in an isotropic direction, boosted to the lab, and the
//            residual is what is left of the four-momentum.
//   A >= 5   a G4Fragment(A, Z, lab4mom) whose excitation is M - mass by construction, handed
//            to G4PhotonEvaporation::BreakUpFragment. The gammas come out in cascade order and
//            the residual - which BreakUpChain has been modifying in place - is pushed LAST.
//
// WHAT `aTrack.GetGlobalTime()` IS HERE, BECAUSE IT READS LIKE A DOUBLE COUNT
//
// `ApplyYourself` takes a `G4HadProjectile`, not a `G4Track`, and
// `G4HadProjectile::Initialise` sets `theTime = 0.0` with the comment "time of interaction
// starts from zero, not global time of a track". So `time` in this function is ZERO, every
// secondary's `SetTime` carries only the de-excitation cascade's own delay, and
// `G4HadronicProcess::FillResult` is the one place the track's global time is added
// (`max(secTime, 0) + time0`). Reading `GetGlobalTime` as the track's would add it twice.
//
// THE CASCADE DELAY IS REAL AND THIS IS THE FIRST CALLER THAT NEEDS IT
//
// `G4PhotonEvaporation::GenerateGamma` samples `-ltime*G4Log(G4UniformRand())` from the level's
// lifetime and writes the running total onto both the gamma and the residual;
// `G4NeutronRadCapture` then gives every secondary `time + max(f->GetCreationTime(), 0.0)`.
// P3's `generate_gamma` drew the number and discarded it, with a comment saying a neutron time
// cut would need it. It does: a neutron's 10 us clock is the one cut in this transport that a
// de-excitation delay can push a track past. `generate_gamma` now takes an optional
// accumulator; passing null leaves it bit-identical, because the draw was always made.
//
// WHAT IS REFUSED, BY NAME
//
//   * The PDG CODE of an excited residual ion, and only the code. Geant4 asks
//     `G4IonTable::GetIon(Z, A, eexc, noFloat, 0)`, which puts an isomer index in the code's
//     last digit - `1000260572` for Fe57 at 136 keV where the ground state is `1000260570` -
//     and P3 refuses that index because it is a property of the run's ion table (docs/PORTED.md
//     2.1.3). This module returns the ground-state encoding and carries (Z, A, E*) beside it.
//     Not rare: 101 of the 600 deterministic oracle points have a non-zero isomer digit.
//
//     The MASS is a different question and it is MEASURED rather than refused. GetIon also
//     snaps E* onto the G4ENSDFSTATE isomer table, so Geant4's residual mass is
//     `M(Z,A) + snapped(E*)` and this module's is `M(Z,A) + E*` with the PhotonEvaporation5.7
//     level energy the cascade actually stopped on. Over the whole deterministic grid - 568
//     residuals, H through Pb, thermal to 10 MeV - the two agree to the last bit at every
//     point, so `CaptureRefusal::kIsomerIonMass` is a flag saying "this row went through the
//     snapping" and not a claim that the number is wrong. docs/RISK.md V40 is why that had to
//     be measured: a compiled table and a data file answering the same question disagreed for
//     952 nuclides once already.
//   * Everything P3 refuses inside photon evaporation, unchanged: the correlated-gamma angular
//     distribution (`fCorrelatedGamma` is false, so every discrete gamma is isotropic) and the
//     internal-conversion SHELL, which `fStoreAllLevels = false` makes unsamplable - so an
//     internal-conversion electron carries the whole transition energy, which is what this
//     configuration computes rather than an approximation of it.
//   * A hyper-fragment cannot arise here: a neutron adds no lambda.
#pragma once

#include <cmath>

#include "core/units.cuh"
#include "core/vec3.cuh"
#include "data/level_data.cuh"
#include "physics/hadronic/deexcitation/deex_params.cuh"
#include "physics/hadronic/deexcitation/fragment.cuh"
#include "physics/hadronic/deexcitation/nuclear_masses.cuh"
#include "physics/hadronic/deexcitation/photon_evaporation.cuh"
#include "physics/hadronic/process.cuh"

namespace g4gpu::physics::hadronic::capture {

/// `lowestEnergyLimit`, G4NeutronRadCapture's constructor: 10 eV. It is NOT overwritten by
/// InitialiseModel - only `minExcitation` is - and it gates the A <= 4 branch alone: a compound
/// whose mass defect is below it emits nothing at all and the process returns an untouched
/// particle change with `stopAndKill` already proposed, so the neutron disappears and no gamma
/// is made. The A >= 5 branch has no such test; photon evaporation's own 10 eV tolerance is
/// what stops it there.
template <typename real_t> __host__ __device__ constexpr real_t capture_lowest_energy() {
  return real_t(10.0) * units::eV<real_t>();
}

/// `minExcitation` AFTER InitialiseModel: `G4DeexPrecoParameters::GetMinExcitation()`, 10 eV.
///
/// The constructor sets 0.1 keV and InitialiseModel replaces it, so the constant in the
/// constructor is dead in every configuration that calls InitialiseModel - which every physics
/// list does, through G4HadronicProcess::PreparePhysicsTable. Both values are written down
/// because 0.1 keV is what a reader of the constructor would take, and it is ten times the
/// value that runs. The two agree with `deex_params().min_excitation`, which is where this
/// reads it from rather than duplicating the number.
__host__ __device__ inline double capture_min_excitation() {
  return deex::deex_params().min_excitation;
}

/// `CLHEP::Hep3Vector::unit()`, which returns the ZERO VECTOR for a zero vector rather than a
/// default axis - and that is the case this function is here for. A residual nucleus left at
/// rest by a symmetric cascade has zero momentum, `f->GetMomentum().vect().unit()` is (0,0,0),
/// and `G4DynamicParticle(theDef, direction, ekin)` stores that direction as given. `normalize`
/// in core/vec3.cuh returns +z instead, which would give a nucleus at rest a direction.
///
/// IT MULTIPLIES BY THE RECIPROCAL, and this model needs both conventions. `unit()` is
/// `p *= (1.0/std::sqrt(mag2()))` and `G4DynamicParticle::Set4Momentum` is
/// `theMomentumDirection.setX(momentum.px()/pp)` - a division. The A <= 4 branch builds its two
/// secondaries with the four-momentum constructor and so takes the DIVISION; the A >= 5 branch
/// calls `f->GetMomentum().vect().unit()` explicitly and takes the RECIPROCAL. They differ by an
/// ulp, and the residual's direction is a cancellation, so the two are written out separately
/// below rather than routed through one helper. docs/RISK.md V37 is the entry about the ulp in
/// `boostVector` that this is the same class of.
///
/// Written here rather than reused: `deex::clhep_unit` in ../deexcitation/excitation_handler.cuh
/// is the same three lines, and including that header for them would pull the sixty GEM
/// channels, the Fermi pool and the fission tables into every translation unit that captures a
/// neutron - which, after P8, includes the transport kernels. `elastic/hadron_elastic.cuh` has
/// a third copy for a `Vec3<real_t>` for the same reason. Three copies of `p *= 1/sqrt(mag2)`
/// is worse than one; three copies that each say which CLHEP function they are and why they are
/// not the shared `normalize` is better than one shared function nobody can find.
__host__ __device__ inline deex::Vec3d capture_unit(const deex::Vec3d& v) {
  const double tot = g4gpu::mag2(v);
  if (!(tot > 0.0)) { return deex::Vec3d{0.0, 0.0, 0.0}; }
  const double inv = 1.0 / std::sqrt(tot);
  return deex::Vec3d{v.x * inv, v.y * inv, v.z * inv};
}

/// `G4DynamicParticle::EnergyMomentumRelationAllowance` squared: `(1.0e-2*keV)^2` = 1e-10 MeV^2.
///
/// Squared, and that is the whole trap. It is compared against `|PDGmass^2 - mass2|`, a
/// difference of two MASS-SQUARED numbers - so for a deuteron at 1875.613 MeV the PDG mass
/// squared is 3.518e6 MeV^2 and one ulp of it is 7.8e-10 MeV^2, ALREADY LARGER THAN THE
/// ALLOWANCE. The test therefore fails for any nucleus heavier than about 300 MeV even when the
/// four-momentum is on the mass shell to the last bit, and the dynamical mass really does become
/// `sqrt(mass2)` rather than the PDG mass. That is not an edge case; it is every capture
/// residual on the A <= 4 branch.
template <typename real_t> __host__ __device__ constexpr double capture_energy_mra2() {
  const double allowance = 1.0e-2 * 1.0e-3;  // 1.0e-2 * keV, in MeV
  return allowance * allowance;
}

/// `G4DynamicParticle::Set4Momentum` and the `(definition, G4LorentzVector)` constructor, which
/// are the same three branches - transcribed onto a HadSecondary.
///
/// WHY THIS IS NOT `ekin = E - m`, WHICH IS WHAT IT LOOKS LIKE
///
///     mass2 = t*t - |p|^2
///     if (mass2 < EnergyMRA2)                      dynamical mass = 0
///     else if (|PDGmass^2 - mass2| > EnergyMRA2)   dynamical mass = sqrt(mass2)
///     else                                         dynamical mass = PDG mass  (unchanged)
///     kinetic energy = t - dynamical mass
///
/// For the capture residual the middle branch is the one that runs, for the reason
/// `capture_energy_mra2` gives, so the kinetic energy is `t - sqrt(t*t - |p|^2)` and NOT
/// `t - M(Z,A)`. The two differ by half an ulp of `t`, which for a 1 keV neutron on hydrogen is
/// 4e-10 RELATIVE on a 0.55 keV deuteron recoil - a cancellation of 3.4e6 amplifying it. That
/// is what tests/test_capture.cu measured before this function existed, and it is the same
/// shape as docs/RISK.md V37: a recoil is a difference of two large numbers and every ulp
/// upstream of it arrives multiplied.
///
/// `unit()` and not a division: `SetMomentumDirection(momentum.vect().unit())` in both the
/// constructor and Set4Momentum. See capture_unit.
///
/// P4's `DecayProduct::set_four_momentum` in physics/decay/decay_products.cuh is the same
/// transcription onto a different struct, and its comment records the same three branches. Two
/// copies because the two structs are different and neither package owns the other's.
template <typename real_t>
__host__ __device__ inline void set_four_momentum(HadSecondary<real_t>& s,
                                                  const deex::LorentzVector& p4,
                                                  double pdg_mass) {
  const double p2 = g4gpu::mag2(p4.v);
  if (!(p2 > 0.0)) {
    // `SetMomentumDirection(1,0,0); SetKineticEnergy(0)` - and it is +x, not +z.
    s.direction = Vec3<real_t>{real_t(1), real_t(0), real_t(0)};
    s.kin_energy = real_t(0);
    s.mass = static_cast<real_t>(pdg_mass);
    return;
  }
  const deex::Vec3d u = capture_unit(p4.v);
  s.direction = Vec3<real_t>{static_cast<real_t>(u.x), static_cast<real_t>(u.y),
                             static_cast<real_t>(u.z)};
  const double total = p4.e;
  const double mass2 = total * total - p2;
  const double pdg2 = pdg_mass * pdg_mass;
  double dyn = pdg_mass;
  if (mass2 < capture_energy_mra2<real_t>()) {
    dyn = 0.0;
  } else if (std::fabs(pdg2 - mass2) > capture_energy_mra2<real_t>()) {
    dyn = std::sqrt(mass2);
  }
  // `SetKineticEnergy` stores whatever it is given; there is no clamp at zero in
  // G4DynamicParticle, so a spacelike four-momentum really would produce a negative kinetic
  // energy. Reproduced, because clamping here would hide it.
  s.kin_energy = static_cast<real_t>(total - dyn);
  s.mass = static_cast<real_t>(dyn);
}

/// What the call could not do faithfully. Never a silent fallback.
enum class CaptureRefusal : int {
  kNone = 0,
  /// The residual came out on an isomeric level, so `G4IonTable::GetIon(Z, A, eexc, ...)`
  /// snapped E* to G4ENSDFSTATE and returned an ion whose PDG code carries an isomer index
  /// this module does not reproduce - and whose mass came from the snapped excitation rather
  /// than the raw one. The two masses agree at every point of the oracle grid; see the file
  /// header for the measurement and for what is actually refused.
  kIsomerIonMass,
  /// The final state ran out of room. `HadFinalState::secondary_overflow` carries the count.
  kSecondaryOverflow,
  /// A target with Z > A, A < 1 or Z < 1: `G4NucleiProperties::GetNuclearMass` returns 0 for it
  /// and every energy below would be built from that zero.
  kUnphysicalTarget,
};

__host__ __device__ inline const char* capture_refusal_name(CaptureRefusal r) {
  switch (r) {
    case CaptureRefusal::kNone: return "none";
    case CaptureRefusal::kIsomerIonMass:
      return "G4IonTable::GetIon isomer snapping for an excited residual ion (the isomer index "
             "is run-dependent; the excitation is reported unsnapped)";
    case CaptureRefusal::kSecondaryOverflow:
      return "the gamma cascade produced more secondaries than the final state can hold";
    case CaptureRefusal::kUnphysicalTarget:
      return "target with A < 1, Z < 1 or Z > A: G4NucleiProperties::GetNuclearMass is zero";
  }
  return "unknown";
}

/// Everything one call of `ApplyYourself` did, beyond the secondaries themselves.
///
/// These are the quantities the oracle compares deterministically: the mass balance is a pure
/// function of (target Z, target A, neutron energy) and does not depend on a single random
/// number, so it can be checked to the last bit before any sampler is exercised.
template <typename real_t>
struct CaptureInfo {
  real_t target_mass = real_t(0);       ///< GetNuclearMass(A, Z), the target at rest
  real_t invariant_mass = real_t(0);    ///< M = |lab4mom|, target at rest plus the neutron
  real_t compound_mass = real_t(0);     ///< GetNuclearMass(A+1, Z)
  real_t excitation = real_t(0);        ///< M - compound_mass: the compound's E*
  int compound_z = 0;
  int compound_a = 0;
  /// The A <= 4 branch's centre-of-mass gamma energy, `(M-mass)(M+mass)/(2M)`. Zero on the
  /// other branch.
  real_t cm_gamma_energy = real_t(0);
  bool one_gamma_branch = false;
  /// `M - mass <= lowestEnergyLimit` on the A <= 4 branch: nothing is emitted and the neutron
  /// still dies, which is a real outcome and not an error.
  bool below_lowest_energy = false;
  /// The `M < mass` protection on the A >= 5 branch fired: lab4mom was rebuilt from
  /// `max(mass, lab4mom.e())`. It is a genuine energy non-conservation of up to `mass - M`,
  /// and Geant4's, so it is reported rather than hidden.
  bool kinematics_fixed = false;
  int n_gammas = 0;
  int n_conversion_electrons = 0;
  int n_cascade_steps = 0;              ///< calls to GenerateGamma that emitted something
  int residual_z = 0;
  int residual_a = 0;
  real_t residual_excitation = real_t(0);   ///< after the handler's `<= minExcitation` zeroing
  int residual_level = 0;                   ///< G4Fragment::GetFloatingLevelNumber
  real_t cascade_time = real_t(0);          ///< the residual's creation time, ns
  CaptureRefusal refused = CaptureRefusal::kNone;
};

/// The species of a nucleus, as G4NeutronRadCapture resolves it: the five light definitions by
/// name and `G4IonTable::GetIon` for everything else.
///
/// `pdg` is the PDG nuclear code `10LZZZAAAI` with I = 0 for every case, which is what
/// `G4IonTable::GetNucleusEncoding` builds for a ground-state ion. An EXCITED ion's real code
/// carries a run-dependent isomer digit in I - see CaptureRefusal::kIsomerIonMass - so the
/// ground-state encoding is returned and the excitation travels beside it in the CaptureInfo
/// rather than being folded into a code that would be wrong.
__host__ __device__ inline int capture_residual_pdg(int z, int a) {
  if (a == 1 && z == 0) { return 2112; }
  if (a == 1 && z == 1) { return 2212; }
  return pdg_nuclear_code(z, a);
}

/// `part->GetPDGMass()` for one secondary of a capture - what `FillResult` and `CheckResult`
/// compare the DYNAMICAL mass in `HadSecondary::mass` against.
///
/// The two are different numbers and `HadSecondary` has one field, which carries the dynamical
/// one because that is what the four-momentum implies and what a transport has to propagate.
/// The definition's mass is recomputed here from (pdg, Z, A) exactly as Geant4 reads it off the
/// definition, rather than being stored: a fifth field on P5's struct is not this package's to
/// add, and the mapping is three lines.
///
/// @param residual_excitation the E* the residual ion was built at, which is part of an excited
///        ion's PDG mass. Zero for a gamma, an electron and a ground-state residual.
template <typename real_t>
__host__ __device__ inline double capture_secondary_pdg_mass(const HadSecondary<real_t>& s,
                                                              double residual_excitation) {
  if (s.a == 0) {
    return (s.pdg == deex::kPdgElectron) ? units::electron_mass_c2<double>() : 0.0;
  }
  return deex::nuclear_mass(s.a, s.z) + residual_excitation;
}

/// `theDef->GetPDGMass()` for a residual, which is the mass its kinetic energy is measured
/// against.
///
/// For the six light nuclei `G4NucleiProperties::GetNuclearMass` returns the particle
/// definition's own PDG mass by construction (see deex::nuclear_mass's Z <= 2 branch), so the
/// two agree exactly and nothing has to special-case them. For an excited heavy ion Geant4's
/// answer is the snapped isomer's mass; this is the unsnapped one, and it is refused by name.
__host__ __device__ inline double capture_residual_pdg_mass(int z, int a, double excitation) {
  return deex::nuclear_mass(a, z) + excitation;
}

/// G4NeutronRadCapture::ApplyYourself.
///
/// @param projectile the neutron, in G4HadProjectile's frame: its four-momentum is
///        `(0, 0, sqrt(T(T+2m)), m+T)`, i.e. rotated onto +z. Every direction below is
///        therefore relative to the neutron's own, and the process rotates the result back with
///        `rotateUz` - which is what `fill_result` in ../process.cuh does.
/// @param target the (Z, A) `SampleZandA` drew. The target is at rest: `lab4mom` is seeded with
///        `(0, 0, 0, GetNuclearMass(A, Z))` and nothing else of G4Nucleus is read.
/// @param levels P3's PhotonEvaporation5.7 table. Only the A >= 5 branch touches it.
/// @param out the final state. `status` is set to stopAndKill FIRST, exactly as
///        `theParticleChange.SetStatusChange(stopAndKill)` is the second line of the function -
///        so the neutron dies even on the paths that emit nothing.
/// @param persistent the photon-evaporation state to carry ACROSS calls, or null for a fresh
///        one per capture.
///
///        This is a real difference and it is a one-in-many-captures one, so it is a parameter
///        rather than a comment. Geant4's `G4NeutronRadCapture` owns one `G4PhotonEvaporation`
///        for the life of the run, and `fIndex` survives from one capture to the next.
///        `G4LevelManager::NearestLevelIndex(energy, index)` uses that index as a hint AND
///        short-circuits on it - `if(ntrans == 0 || std::abs(energy - fLevelEnergy[idx]) <=
///        tolerance) return idx` - so a stale hint can change the answer, not just the search.
///        It almost never does: `GenerateGamma` ends a completed cascade with `fIndex = 0`, the
///        same value a fresh state starts at, so the two agree for every capture whose
///        predecessor reached the ground state. They can differ only after a cascade that
///        stopped on an ISOMER, which returns early and leaves `fIndex` on that level.
///
///        The transport passes null, because a per-track G4PhotonEvaporation is state this
///        device port has nowhere to keep. The oracle's deterministic table constructs a fresh
///        model per call so that the comparison is about the transcription; its statistical
///        table uses one long-lived model, as a run does, so the difference is measured rather
///        than assumed away.
template <typename real_t, int kCap, typename Rng>
__host__ __device__ inline CaptureInfo<real_t> neutron_rad_capture_apply(
    const HadProjectile<real_t>& projectile, const HadNucleus& target,
    const data::LevelTable& levels, Rng& rng, HadFinalState<real_t, kCap>* out,
    deex::PhotonEvaporationState* persistent = nullptr) {
  CaptureInfo<real_t> info;
  out->clear();
  out->status = HadFinalStateStatus::kStopAndKill;

  int A = target.a;
  const int Z = target.z;
  if (A < 1 || Z < 1 || Z > A) {
    info.refused = CaptureRefusal::kUnphysicalTarget;
    return info;
  }

  // "time of interaction starts from zero": G4HadProjectile::theTime. See the file header.
  const double interaction_time = 0.0;

  // Create initial state. Doubles, not real_t: these are ~2e5 MeV numbers whose difference is
  // the whole answer - the capture Q value is 2 to 9 MeV out of a compound mass of up to
  // 2e5 MeV, and the residual's kinetic energy is keV out of that. deex/nuclear_masses.cuh
  // says the same thing about its own arithmetic.
  const double target_mass = deex::nuclear_mass(A, Z);
  const double kin = static_cast<double>(projectile.kin_energy);
  const double m_n = static_cast<double>(projectile.mass);
  const double p_neutron = std::sqrt(kin * (kin + 2.0 * m_n));
  // `target_mass + (m_n + kin)` AND NOT `(target_mass + m_n) + kin`, and the parentheses are
  // measured rather than tidy. Geant4 builds this in two statements -
  // `lab4mom.set(0,0,0,GetNuclearMass(A,Z))` and then `lab4mom += aTrack.Get4Momentum()`, whose
  // energy component `G4HadProjectile::InitialiseLocal` already formed as `theMass +
  // theKinEnergy` - so the neutron's mass and kinetic energy are added to each other FIRST and
  // the target's mass last. Associating it the other way rounds differently at the last bit,
  // and everything downstream is a cancellation: `M = |lab4mom|`, then the gamma energy
  // `(M-mass)(M+mass)/(2M)`, then the residual's kinetic energy. MEASURED, by writing it the
  // other way and rerunning tests/test_capture.cu: `(target_mass + m_n) + kin` puts the
  // secondary kinetic energies 1.0e-13 out and the directions 1.1e-13 out, where Geant4's
  // associativity puts every one of 1831 of them at exactly zero.
  deex::LorentzVector lab4mom(0.0, 0.0, p_neutron, target_mass + (m_n + kin));

  const double M = lab4mom.mag();
  ++A;
  const double mass = deex::nuclear_mass(A, Z);

  info.target_mass = static_cast<real_t>(target_mass);
  info.invariant_mass = static_cast<real_t>(M);
  info.compound_mass = static_cast<real_t>(mass);
  info.excitation = static_cast<real_t>(M - mass);
  info.compound_z = Z;
  info.compound_a = A;

  // ---------------------------------------------------------------------------------------
  // "simplified method of 1 gamma emission" - the compound has four nucleons or fewer.
  // ---------------------------------------------------------------------------------------
  if (A <= 4) {
    info.one_gamma_branch = true;
    const deex::Vec3d bst = lab4mom.boost_vector();

    // The test is on the mass DEFECT and not on the neutron's energy, and it is `<=`. For a
    // proton target M - mass is the deuteron's 2.2246 MeV binding energy plus the neutron's
    // kinetic energy, so it can only fail for a target whose compound is unbound.
    if (M - mass <= static_cast<double>(capture_lowest_energy<real_t>())) {
      info.below_lowest_energy = true;
      return info;
    }

    // Exactly the energy that leaves the residual with invariant mass `mass`:
    // E_res = M - e1 = (M^2 + mass^2)/(2M) and p_res = e1, so E_res^2 - p_res^2 = mass^2.
    // Written as Geant4 writes it - the product of the sum and the difference - rather than as
    // (M*M - mass*mass)/(2*M), because for a 2.2 MeV defect on a 1876 MeV compound the second
    // form cancels three and a half digits and the first cancels none.
    const double e1 = (M - mass) * (M + mass) / (2.0 * M);
    info.cm_gamma_energy = static_cast<real_t>(e1);

    const deex::Vec3d dir = deex::random_direction(rng);
    deex::LorentzVector lv2(dir * e1, e1);
    lv2.boost(bst);

    {
      HadSecondary<real_t> g;
      g.pdg = deex::kPdgGamma;
      g.z = 0;
      g.a = 0;
      g.mass = real_t(0);
      g.time = static_cast<real_t>(interaction_time);
      // A gamma's `mass2 = t*t - |p|^2` is not exactly zero - the direction is a unit vector
      // only to rounding - but it is far below EnergyMRA2, so the first branch of
      // Set4Momentum takes the dynamical mass to zero and the kinetic energy to the total
      // energy. That is why this and the residual go through the same function.
      set_four_momentum(g, lv2, 0.0);
      if (!out->add_secondary(g)) { info.refused = CaptureRefusal::kSecondaryOverflow; }
      ++info.n_gammas;
    }

    lab4mom -= lv2;

    {
      HadSecondary<real_t> r;
      r.z = Z;
      r.a = A;
      r.pdg = capture_residual_pdg(Z, A);
      r.time = static_cast<real_t>(interaction_time);
      // A compound Geant4 has no definition for - (Z=1, A=4), say - reaches
      // G4IonTable::GetIon(1, 4, 0.0), whose mass is built from GetNuclearMass, which is the
      // same number `mass` already holds.
      set_four_momentum(r, lab4mom, mass);
      if (!out->add_secondary(r)) { info.refused = CaptureRefusal::kSecondaryOverflow; }
      info.residual_z = Z;
      info.residual_a = A;
    }
    return info;
  }

  // ---------------------------------------------------------------------------------------
  // Photon evaporation - the compound has five nucleons or more.
  // ---------------------------------------------------------------------------------------

  // "protection against wrong kinematic". M < mass means the compound is below its own ground
  // state, which happens when the mass table's Q value for this capture is negative - it is
  // for a handful of nuclides. Geant4 does not reject it: it rebuilds the four-momentum at the
  // ground-state mass, keeping the larger of `mass` and the lab energy, and the momentum is
  // recomputed from that. Energy is conserved only when lab4mom.e() already exceeded `mass`;
  // otherwise up to `mass - lab4mom.e()` is created. Reported, not repaired.
  if (M < mass) {
    info.kinematics_fixed = true;
    const double etot = (mass > lab4mom.e) ? mass : lab4mom.e;
    const double ptot = std::sqrt((etot - mass) * (etot + mass));
    const deex::Vec3d v = capture_unit(lab4mom.v);
    lab4mom = deex::LorentzVector(v * ptot, etot);
  }

  deex::Fragment nucleus = deex::make_fragment(A, Z, lab4mom);

  // G4VEvaporationChannel::BreakUpFragment is BreakUpChain into a fresh vector, and
  // BreakUpChain is `do { gamma = GenerateGamma(nucleus); if(gamma) products->push_back }
  // while(gamma)`. Written out here rather than through P3's photon_break_up_chain because
  // this caller needs each gamma converted to a secondary as it appears - the alternative is a
  // second buffer of Fragments the size of the longest cascade, on a kernel's stack - and
  // because it needs the creation time, which the chain helper has nowhere to put.
  //
  // ONE PhotonEvaporationState FOR THE WHOLE CASCADE, matching the single G4PhotonEvaporation
  // the model owns: its `level_index` is a search hint that must survive from one gamma to the
  // next. Whether it also survives from one CAPTURE to the next is the caller's choice - see
  // the `persistent` parameter.
  deex::PhotonEvaporationState local_ps;
  deex::PhotonEvaporationState& ps = (persistent != nullptr) ? *persistent : local_ps;
  double creation_time = 0.0;
  for (;;) {
    const deex::GammaEmission g =
        deex::generate_gamma(ps, nucleus, levels, rng, &creation_time);
    if (!g.emitted) { break; }
    ++info.n_cascade_steps;
    HadSecondary<real_t> s;
    const bool is_gamma = (g.product.pdg_if_not_nucleus == deex::kPdgGamma);
    s.pdg = g.product.pdg_if_not_nucleus;
    s.z = 0;
    s.a = 0;
    s.mass = static_cast<real_t>(g.product.ground_state_mass);
    // `ekin = std::max(0.0, etot - theDef->GetPDGMass())`, with etot the fragment's total
    // energy. For a gamma the mass is zero; for a conversion electron it is m_e.
    const double etot = g.product.momentum.e;
    const double ekin = etot - g.product.ground_state_mass;
    s.kin_energy = static_cast<real_t>((ekin > 0.0) ? ekin : 0.0);
    const deex::Vec3d u = capture_unit(g.product.momentum.v);
    s.direction = Vec3<real_t>{static_cast<real_t>(u.x), static_cast<real_t>(u.y),
                               static_cast<real_t>(u.z)};
    // `timeF = f->GetCreationTime(); if(timeF < 0.0) timeF = 0.0; SetTime(time + timeF)`.
    const double tf = (creation_time > 0.0) ? creation_time : 0.0;
    s.time = static_cast<real_t>(interaction_time + tf);
    if (!out->add_secondary(s)) { info.refused = CaptureRefusal::kSecondaryOverflow; }
    if (is_gamma) { ++info.n_gammas; } else { ++info.n_conversion_electrons; }
  }

  // `fv->push_back(aFragment)` - the residual goes in LAST, after every gamma, and it is the
  // fragment BreakUpChain has been modifying in place rather than a copy of the compound.
  {
    double eexc = nucleus.excitation;
    int level = nucleus.floating_level;
    if (eexc <= capture_min_excitation()) {
      eexc = 0.0;
      // Geant4 does not zero the floating level here (G4ExcitationHandler does); GetIon is
      // called with `noFloat` regardless, so the level index never reaches the ion table from
      // this model at all.
    } else {
      info.refused = CaptureRefusal::kIsomerIonMass;
    }
    const double res_mass = capture_residual_pdg_mass(nucleus.z, nucleus.a, eexc);
    HadSecondary<real_t> r;
    r.z = nucleus.z;
    r.a = nucleus.a;
    r.pdg = capture_residual_pdg(nucleus.z, nucleus.a);
    r.mass = static_cast<real_t>(res_mass);
    const double ekin = nucleus.momentum.e - res_mass;
    r.kin_energy = static_cast<real_t>((ekin > 0.0) ? ekin : 0.0);
    const deex::Vec3d u = capture_unit(nucleus.momentum.v);
    r.direction = Vec3<real_t>{static_cast<real_t>(u.x), static_cast<real_t>(u.y),
                               static_cast<real_t>(u.z)};
    const double tf = (creation_time > 0.0) ? creation_time : 0.0;
    r.time = static_cast<real_t>(interaction_time + tf);
    if (!out->add_secondary(r)) { info.refused = CaptureRefusal::kSecondaryOverflow; }
    info.residual_z = nucleus.z;
    info.residual_a = nucleus.a;
    info.residual_excitation = static_cast<real_t>(eexc);
    info.residual_level = level;
    info.cascade_time = static_cast<real_t>(tf);
  }
  return info;
}

}  // namespace g4gpu::physics::hadronic::capture
