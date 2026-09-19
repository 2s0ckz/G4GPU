// G4LightTargetCollider - Bertini's photon arm on hydrogen and deuterium, and the only piece of
// the INUCL tree P10 left refused.
//
// Transcribed from Geant4 11.1.1, source/processes/hadronic/models/cascade/cascade/src/
// G4LightTargetCollider.cc:
//   collide, GammaDCrossSection, AbsorptionOnDeuteron, SingleNucleonScattering
//
// WHY IT IS HERE AND NOT UNDER bertini/
//
// `docs/PORTED.md` 2.1.12 lists `G4LightTargetCollider` among P10's refusals - "a photon on
// A < 3 - P13's" - and `bert::apply_yourself` returns `InterfaceRefusal::kLightTargetCollider`
// at the point `G4CascadeInterface::ApplyYourself` would have branched. That branch is
//
//     if (aTrack.GetDefinition() == G4Gamma::Gamma() && theNucleus.GetA_asInt() < 3)
//
// and it is the FIRST thing ApplyYourself does after IsApplicable, before `createBullet`. So
// the arm is a sibling of `bert::apply_yourself` rather than something inside it, and this file
// is that sibling: it includes P10's headers and P10's headers do not include it. Nothing under
// `bertini/` changes.
//
// WHY IT MATTERS: WATER. Two of the three atoms in water are hydrogen, and `G4GammaNuclearXS`
// sends every hydrogen target to CHIPS at every energy, so a photo-nuclear interaction on the
// hydrogen in water is a real and frequent event above the 144.68 MeV pi0 threshold. Without
// this file every one of them is a refusal.
//
// THE THREE BRANCHES, AND THE ONE THAT IS NOT A CASCADE
//
//   target = proton (A = 1)
//     ke < 0.1447 GeV   -> `globalOutput.trivialise(bullet, target)`: the photon and the proton
//                          come back unchanged, which the interface then treats as a final
//                          state, not as "no interaction". Below the lab threshold for pi0
//                          production off a proton.
//     otherwise         -> G4ElementaryParticleCollider on the bare nucleon, i.e. P10's
//                          `ep_collide`. If it produces nothing, trivialise as well.
//
//   target = deuteron (A = 2)   three channels, chosen by cross section:
//     gamma-p           -> Fermi momentum sampled on the Hulthen potential (a FIXED magnitude
//                          of 0.045 GeV in a random direction, not a distribution), scattering
//                          on the moving proton, the neutron kept as an unmodified spectator
//     gamma-n           -> the mirror
//     absorption        -> the deuteron breaks into p + n back to back in the CM
//
// `GammaDCrossSection` is in MILLIBARN and the two channel cross sections it competes against -
// `G4CascadeChannelTables::GetTable(9)->getCrossSection(ke)` and table 18 - are the Bertini
// gamma-p and gamma-n totals, also in millibarn. Only their RATIO is used, so the unit never
// leaves the function.
//
// WHAT IS REFUSED, BY NAME
//
//   * A non-photon bullet. Geant4's `collide` accepts pions too - `AbsorptionOnDeuteron` has
//     pi+, pi- and pi0 arms and `SingleNucleonScattering` is generic - but the only caller in
//     11.1.1 is the gamma branch of `G4CascadeInterface::ApplyYourself`, so the pion arms have
//     no oracle and are refused rather than transcribed on speculation.
//   * A target that is neither a proton nor a deuteron. Geant4 raises a FatalException
//     ("Scattering from this target not implemented") for A = 1 with Z = 0 (a bare neutron) and
//     for anything else that reaches it; A < 3 with Z = 1 is the whole reachable set.
//   * Energy below the deuteron break-up threshold, where Geant4 issues a JustWarning
//     ("Projectile energy below reaction threshold") and trivialises. Reproduced as the
//     trivialisation it is, with the warning recorded rather than printed.
#ifndef G4GPU_HADRONIC_EMEXTRA_LIGHT_TARGET_CUH
#define G4GPU_HADRONIC_EMEXTRA_LIGHT_TARGET_CUH

#include <cmath>

#include "physics/hadronic/bertini/cascade_params.cuh"
#include "physics/hadronic/bertini/channel_tables.cuh"
#include "physics/hadronic/bertini/collision_output.cuh"
#include "physics/hadronic/bertini/ep_collider.cuh"
#include "physics/hadronic/bertini/inucl_particle.cuh"
#include "data/nuclei_mass_ame12.hh"
#include "physics/hadronic/emextra/config.cuh"
#include "physics/hadronic/process.cuh"

namespace g4gpu::physics::hadronic::emextra {

using bert::LV;
using bert::Vec3d;

/// Which arm of `G4LightTargetCollider::collide` ran, so a campaign can report the mix rather
/// than only the products.
enum class LightTargetArm : int {
  kNone = 0,
  kProtonBelowThreshold,   ///< ke < 0.1447 GeV: trivialised
  kProtonCollider,         ///< G4ElementaryParticleCollider on a bare proton
  kProtonColliderEmpty,    ///< the collider produced nothing: trivialised
  kDeuteronBelowThreshold, ///< ke < mP + mN - mD: JustWarning, then trivialised
  kDeuteronProton,         ///< scattering off the proton, neutron spectator
  kDeuteronNeutron,        ///< scattering off the neutron, proton spectator
  kDeuteronAbsorption      ///< gamma d -> p n
};

/// What one `light_target_collide` did.
struct LightTargetResult {
  LightTargetArm arm = LightTargetArm::kNone;
  int n_products = 0;
  /// `SingleNucleonScattering` failed to generate a final state after 200 tries at every
  /// multiplicity down to 2, and Geant4's JustWarning path returns the projectile and the
  /// target as "dummies" that are NOT in the CM frame. Reproduced, and flagged, because it is
  /// a final state that does not conserve anything.
  bool scattering_failed = false;
  EmExtraRefusal refusal = EmExtraRefusal::kNone;
  bert::ColliderRefusal collider_refusal = bert::ColliderRefusal::kNone;
};

/// `mP`, `mN`, `mD` and `pFermiD` as the constructor sets them - Bertini's GeV throughout.
/// The three masses come from the particle table, NOT from `G4InuclElementaryParticle::
/// getParticleMass`, and for the deuteron that is `G4Deuteron::Deuteron()->GetPDGMass()`,
/// which is the same number `bert::inucl_nuclei_mass(2, 1, 0)` returns only because
/// G4NucleiProperties answers the light nuclides from the particle table too.
__host__ __device__ inline constexpr double lt_mP() {
  return units::proton_mass_c2<double>() * 0.001;
}
__host__ __device__ inline constexpr double lt_mN() {
  return units::neutron_mass_c2<double>() * 0.001;
}
__host__ __device__ inline constexpr double lt_mD() {
  return data::deuteron_mass_c2<double>() * 0.001;
}
/// "Fermi momentum of nucleon in deuteron Hulthen potential" - a CONSTANT 45 MeV/c, given a
/// random direction. Not sampled from a Hulthen distribution; the comment names the potential
/// the number came from and the code uses the number.
__host__ __device__ inline constexpr double lt_p_fermi_d() { return 0.045; }

/// `G4LightTargetCollider::GammaDCrossSection(E_gamma)`, millibarn, E in GeV.
///
/// Three pieces: a flat 1000 mb below the pi0 threshold ("no parameterization needed below pi0
/// threshold where cross section is 100% disintegration"), a Gaussian in the resonance region,
/// and an inverse fourth power above 420 MeV. Note the first branch is the INITIALISER, so an
/// energy in neither of the two tested ranges - i.e. `E <= 0.144` - keeps 1000.
__host__ __device__ inline double gamma_d_cross_section(double gammaEnergy) {
  double sigma = 1000.0;
  double term = 0.0;
  if (gammaEnergy > 0.144 && gammaEnergy < 0.42) {
    term = (gammaEnergy - 0.24) / 0.155;
    sigma = 0.065 * std::exp(-term * term);
  } else if (gammaEnergy >= 0.42) {
    sigma = 0.000526 / gammaEnergy / gammaEnergy / gammaEnergy / gammaEnergy;
  }
  return sigma;
}

/// `G4LightTargetCollider::SingleNucleonScattering`, in the CM of the projectile and the
/// (moving) target nucleon.
///
/// NOT the same function as `G4ElementaryParticleCollider::collide`, although both end in
/// `G4CascadeFinalStateGenerator`. Three differences, all of which change the random stream:
///
///   1. the multiplicity is drawn ONCE and then walked DOWN - `while (mult > 1) { ... if
///      (itry == itry_max) mult--; else break; }` - where `generateSCMfinalState` redraws it
///      every attempt;
///   2. `itry_max` is 200, not 10;
///   3. there is no SCM conversion, no pion-absorption arm and no sort by kinetic energy: the
///      momenta are returned in the CM frame the caller supplied and the caller boosts them.
///
/// `ke` is the projectile's kinetic energy in the nucleon's rest frame, which is where the
/// caller has already boosted both.
template <typename Rng>
__host__ __device__ inline bool single_nucleon_scattering(int proj_type, const LV& proj_mom,
                                                          int nucleon_type,
                                                          const LV& nucleon_mom,
                                                          const bert::CascadeParams& par,
                                                          bert::ColliderOutput& out,
                                                          bert::BertiniWorkspace& ws,
                                                          Rng& rng) {
  out.n = 0;
  out.refusal = bert::ColliderRefusal::kNone;
  out.channel_refusal = bert::ChannelRefusal::kNone;
  out.sample_fell_off = false;

  const int is = proj_type * nucleon_type;
  const bert::ChannelTable t = bert::channel_table(is);
  if (!t.valid()) {
    out.refusal = bert::ColliderRefusal::kNoChannelTable;
    return false;
  }
  // `G4double ke = projectile.getKineticEnergy();` - the INUCL stored kinetic energy of the
  // projectile as the caller handed it in, not a recomputed `e - m`.
  double ke = 0.0;
  bert::inucl_store_momentum(proj_mom, proj_type, &ke);
  const LV total = proj_mom + nucleon_mom;
  const double Ecm = total.mag();

  bool fell = false;
  int mult = bert::channel_multiplicity(t, ke, rng, fell);
  out.sample_fell_off = out.sample_fell_off || fell;

  const int itry_max = 200;
  bert::FinalStateConfig& cfg = ws.fs_cfg;
  cfg = bert::FinalStateConfig();
  bool generated = false;

  while (mult > 1) {
    if (mult > t.max_multiplicity() || mult > bert::kMaxFinalStateSize) {
      out.refusal = bert::ColliderRefusal::kMultiplicityTooLarge;
      return false;
    }
    int itry = 0;
    bool generate = true;
    while (generate && itry < itry_max) {
      int chan = -1;
      fell = false;
      const bert::ChannelRefusal cr = bert::outgoing_particle_types(
          t, mult, ke, rng, cfg.kinds, chan, fell, ws.sigma_buf);
      out.sample_fell_off = out.sample_fell_off || fell;
      if (cr != bert::ChannelRefusal::kNone) {
        out.channel_refusal = cr;
        out.refusal = bert::ColliderRefusal::kChannelRefused;
        return false;
      }
      cfg.multiplicity = mult;
      for (int i = 0; i < mult; ++i) {
        cfg.masses[i] = bert::inucl_particle_mass(cfg.kinds[i]);
        cfg.masses2[i] = cfg.masses[i] * cfg.masses[i];
      }
      const int fs = (mult == 2) ? cfg.kinds[0] * cfg.kinds[1] : 0;
      bert::fs_choose_generators(cfg, is, fs, par);
      bert::fs_save_kinematics(cfg, proj_type, proj_mom, nucleon_type, nucleon_mom);
      bool bad = false;
      generate = !bert::fs_generate(cfg, Ecm, out.momenta, rng, bad);
      ++itry;
    }
    // `if (itry == itry_max) mult--; else break;` - the post-increment above makes `itry` equal
    // to `itry_max` on exhaustion AND on a success at the 200th attempt, so a final state
    // generated on the last try is DISCARDED and the multiplicity drops. The same off-by-one
    // `generateSCMfinalState` has (docs/RISK.md V121), with a different limit.
    if (itry == itry_max) { --mult; }
    else { generated = true; break; }
  }

  if (!generated || mult < 2) { return false; }
  out.n = mult;
  for (int i = 0; i < mult; ++i) { out.kinds[i] = cfg.kinds[i]; }
  return true;
}

/// `G4LightTargetCollider::AbsorptionOnDeuteron` for a photon: gamma d -> p n, isotropic in the
/// CM ("assuming 100% S wave"), then boosted to the lab along the bullet's own direction.
///
/// `betacm(0, 0, p/(E + mD))` is built from the MAGNITUDE of the bullet momentum on the z axis,
/// which is correct only because the caller's bullet is along +z - which it is, because
/// `createBullet` puts it there. Written out rather than generalised.
template <typename Rng>
__host__ __device__ inline void absorption_on_deuteron(const LV& bullet, double bullet_mass,
                                                       int kinds[2], LV momenta[2], Rng& rng) {
  const double S = bullet_mass * bullet_mass + lt_mD() * lt_mD() + 2.0 * lt_mD() * bullet.e;
  const double sp = lt_mP() + lt_mN();
  const double sm = lt_mP() - lt_mN();
  const double qcm = std::sqrt((S - sp * sp) * (S - sm * sm) / S / 4.0);

  LV m1, m2;
  m1.e = std::sqrt(lt_mP() * lt_mP() + qcm * qcm);
  m2.e = std::sqrt(lt_mN() * lt_mN() + qcm * qcm);
  kinds[0] = bert::kProton;
  kinds[1] = bert::kNeutron;

  const Vec3d d = deex::random_direction(rng);
  m1.v = Vec3d{qcm * d.x, qcm * d.y, qcm * d.z};
  m2.v = Vec3d{-m1.v.x, -m1.v.y, -m1.v.z};

  const double pmod = std::sqrt(g4gpu::mag2(bullet.v));
  const Vec3d betacm{0.0, 0.0, pmod / (bullet.e + lt_mD())};
  m1.boost(betacm);
  m2.boost(betacm);
  momenta[0] = m1;
  momenta[1] = m2;
}

/// `G4CascadeInterface::copyOutputToHadronicResult`, for a light-target output.
///
/// The same function `bert::apply_yourself` runs inline at its end; it is written again here
/// because P10's copy is inside that function and this arm never enters it. The one branch this
/// output can reach that the cascade's cannot is nothing - no K0, no dibaryon, no nucleus above
/// A = 1 comes out of a proton or deuteron target - but the K0 mixing draw is kept because a
/// gamma-p channel CAN produce a K0 (the gamma-p table has strange channels above 1 GeV) and
/// dropping the deviate would desynchronise the stream.
template <typename real_t, int kCap, typename Rng>
__host__ __device__ inline int copy_light_output(const bert::CollisionOutput& out,
                                                 HadFinalState<real_t, kCap>& fs, Rng& rng) {
  fs.status = HadFinalStateStatus::kStopAndKill;
  fs.energy_change = real_t(0);
  int n = 0;
  for (int i = 0; i < out.n_particles; ++i) {
    const int type = out.particles[i].type;
    if (bert::inucl_is_quasideuteron(type)) { continue; }
    HadSecondary<real_t> s;
    int pdg = bert::inucl_type_row(type).pdg;
    if (type == bert::kKaonZero || type == bert::kKaonZeroBar) {
      pdg = (rng.uniform() > 0.5) ? 130 : 310;
    }
    s.pdg = pdg;
    s.z = static_cast<int>(bert::inucl_type_row(type).charge);
    s.a = bert::inucl_names_baryon(type);
    s.mass = real_t(bert::inucl_particle_mass(type) * 1000.0);
    s.kin_energy = real_t(out.particles[i].ekin * 1000.0);
    s.direction = Vec3<real_t>{real_t(out.particles[i].momentum.v.x),
                               real_t(out.particles[i].momentum.v.y),
                               real_t(out.particles[i].momentum.v.z)};
    const real_t m = sqrt(s.direction.x * s.direction.x + s.direction.y * s.direction.y +
                          s.direction.z * s.direction.z);
    if (m > real_t(0)) {
      s.direction.x /= m;
      s.direction.y /= m;
      s.direction.z /= m;
    }
    if (fs.add_secondary(s)) { ++n; }
  }
  for (int i = 0; i < out.n_nuclei; ++i) {
    HadSecondary<real_t> s;
    s.z = out.nuclei[i].z;
    s.a = out.nuclei[i].a;
    s.pdg = pdg_nuclear_code(s.z, s.a);
    s.mass = real_t(out.nuclei[i].mass * 1000.0);
    s.kin_energy = real_t(out.nuclei[i].ekin * 1000.0);
    s.direction = Vec3<real_t>{real_t(out.nuclei[i].momentum.v.x),
                               real_t(out.nuclei[i].momentum.v.y),
                               real_t(out.nuclei[i].momentum.v.z)};
    const real_t m = sqrt(s.direction.x * s.direction.x + s.direction.y * s.direction.y +
                          s.direction.z * s.direction.z);
    if (m > real_t(0)) {
      s.direction.x /= m;
      s.direction.y /= m;
      s.direction.z /= m;
    }
    if (fs.add_secondary(s)) { ++n; }
  }
  return n;
}

/// `G4LightTargetCollider::collide` plus the `copyOutputToHadronicResult` that follows it in
/// `G4CascadeInterface::ApplyYourself`, for a photon on A < 3.
///
/// Units: the projectile arrives in the port's MeV, everything inside is Bertini's GeV, and the
/// secondaries go back out in MeV - the same two conversions `createBullet` and
/// `makeDynamicParticle` do, and the same ones `bert::apply_yourself` does.
///
/// NOTE what ApplyYourself does NOT do on this branch: there is no retry loop, no
/// `balance->collide`, no `retryInelasticProton` and no `checkFinalResult` gate on the result -
/// `checkFinalResult()` is called but it only reports. So a light-target event is never
/// regenerated and never becomes `NoInteraction`.
template <typename real_t, int kCap, typename Rng>
__host__ __device__ inline LightTargetResult light_target_collide(
    const HadProjectile<real_t>& projectile, const HadNucleus& target,
    HadFinalState<real_t, kCap>& fs, const bert::CascadeParams& par,
    bert::BertiniWorkspace* ws, bert::ColliderOutput& epo, bert::CollisionOutput& out,
    Rng& rng) {
  LightTargetResult r;
  fs.clear();
  if (projectile.pdg != 22) {
    r.refusal = EmExtraRefusal::kLightTargetCollider;
    return r;
  }
  if (ws == nullptr) {
    r.refusal = EmExtraRefusal::kCapacity;
    return r;
  }
  if (!(target.z == 1 && (target.a == 1 || target.a == 2))) {
    // "Scattering from this target not implemented" - a FatalException in Geant4. A bare
    // neutron target (Z = 0, A = 1) lands here too.
    r.refusal = EmExtraRefusal::kLightTargetCollider;
    return r;
  }

  // createBullet: along +z, Bertini's GeV, through the INUCL store.
  const double plab = double(projectile.momentum()) * 0.001;
  const double elab = double(projectile.total_energy()) * 0.001;
  const LV bullet = bert::inucl_store_momentum(LV(Vec3d{0.0, 0.0, plab}, elab), bert::kPhoton);
  double ke = 0.0;
  bert::inucl_store_momentum(bullet, bert::kPhoton, &ke);
  const double bullet_mass = bert::inucl_particle_mass(bert::kPhoton);

  bert::co_reset(out);

  if (target.a == 1) {
    const LV tgt(Vec3d{0.0, 0.0, 0.0}, bert::inucl_particle_mass(bert::kProton));
    if (ke < 0.1447) {
      r.arm = LightTargetArm::kProtonBelowThreshold;
      bert::co_trivialise(out, bert::kPhoton, bullet, 0, 0, 1, 1, tgt);
    } else {
      bert::ep_collide(bert::kPhoton, bullet, bert::kProton, tgt, par, epo, *ws, rng);
      r.collider_refusal = epo.refusal;
      // `kKinematicsFailed` IS NOT A REFUSAL HERE. It is `generateSCMfinalState` exhausting its
      // ten attempts, after which Geant4's `collide` does `if (out.n == 0) return;` - "failed
      // to collide: bullet passes through" - and `G4LightTargetCollider` then trivialises.
      // So it is an empty output and the empty-output branch below is Geant4's answer to it.
      // It is REACHABLE and frequent: a photon just above the 144.7 MeV gate is below the
      // pi0-p threshold in the channel tables' own binning, which is the effect the two
      // comments in `G4CascadeInterface::ApplyYourself` describe and neither of them fixes.
      // Every other `ColliderRefusal` is a genuine boundary of this port and is reported.
      if (epo.refusal != bert::ColliderRefusal::kNone
          && epo.refusal != bert::ColliderRefusal::kKinematicsFailed) {
        r.refusal = EmExtraRefusal::kSubModel;
        return r;
      }
      if (epo.n == 0) {
        r.arm = LightTargetArm::kProtonColliderEmpty;
        bert::co_trivialise(out, bert::kPhoton, bullet, 0, 0, 1, 1, tgt);
      } else {
        r.arm = LightTargetArm::kProtonCollider;
        for (int i = 0; i < epo.n; ++i) {
          if (!bert::co_add_particle(out, epo.kinds[i], epo.momenta[i])) {
            r.refusal = EmExtraRefusal::kCapacity;
            return r;
          }
        }
      }
    }
  } else {
    const LV tgt(Vec3d{0.0, 0.0, 0.0}, lt_mD());
    if (ke < lt_mP() + lt_mN() - lt_mD()) {
      // "Should not happen as long as inelastic cross section is zero" - a JustWarning.
      r.arm = LightTargetArm::kDeuteronBelowThreshold;
      bert::co_trivialise(out, bert::kPhoton, bullet, 0, 0, 2, 1, tgt);
    } else {
      const double gammaPXS = bert::channel_cross_section(bert::channel_table(9), ke);
      const double gammaNXS = bert::channel_cross_section(bert::channel_table(18), ke);
      const double gammaDXS = gamma_d_cross_section(ke);
      double probP = 0.0;
      double probN = 0.0;
      // "Highest threshold is 0.152 (for gamma p -> n pi+). Because of Fermi momentum in
      // deuteron, raise this to 0.159" - below it the two scattering channels have zero
      // probability and every event is absorption, whatever the cross sections say.
      if (ke > 0.159) {
        const double totalDXS = gammaPXS + gammaNXS + gammaDXS;
        probP = gammaPXS / totalDXS;
        probN = (gammaPXS + gammaNXS) / totalDXS;
      }
      const double rndm = rng.uniform();
      if (rndm < probP || rndm < probN) {
        const bool on_proton = (rndm < probP);
        r.arm = on_proton ? LightTargetArm::kDeuteronProton : LightTargetArm::kDeuteronNeutron;
        // The Fermi momentum: one random DIRECTION, a fixed magnitude, and the two nucleons
        // back to back. Note both nucleons are put on shell with `sqrt(m^2 + pFermi^2)`, so
        // their energies sum to more than the deuteron mass - the binding energy is not
        // subtracted anywhere in this function.
        const Vec3d fd = deex::random_direction(rng);
        const Vec3d fermi{lt_p_fermi_d() * fd.x, lt_p_fermi_d() * fd.y, lt_p_fermi_d() * fd.z};
        const double ep = std::sqrt(lt_mP() * lt_mP() + lt_p_fermi_d() * lt_p_fermi_d());
        const double en = std::sqrt(lt_mN() * lt_mN() + lt_p_fermi_d() * lt_p_fermi_d());
        LV protonMomentum(fermi, ep);
        LV neutronMomentum(Vec3d{-fermi.x, -fermi.y, -fermi.z}, en);

        LV targetNucleon = on_proton ? protonMomentum : neutronMomentum;
        const LV spectator = on_proton ? neutronMomentum : protonMomentum;
        const int target_type = on_proton ? bert::kProton : bert::kNeutron;
        const int spectator_type = on_proton ? bert::kNeutron : bert::kProton;

        LV bulletMomentum = bullet;
        // `findBoostToCM(w)` is `-(this + w).boostVector()`, so `-betacm` below is the boost
        // FROM the CM back to the lab.
        const LV sum = bulletMomentum + targetNucleon;
        const Vec3d bsum = sum.boost_vector();
        const Vec3d betacm{-bsum.x, -bsum.y, -bsum.z};

        const Vec3d torest = targetNucleon.boost_vector();
        const Vec3d toRest{-torest.x, -torest.y, -torest.z};
        targetNucleon.boost(toRest);
        bulletMomentum.boost(toRest);

        // G4InuclElementaryParticle(mom, definition) - the INUCL store, again, and for a photon
        // it is not the identity (docs/RISK.md V125).
        const LV projStored = bert::inucl_store_momentum(bulletMomentum, bert::kPhoton);
        const LV nucStored = bert::inucl_store_momentum(targetNucleon, target_type);
        const bool ok = single_nucleon_scattering(bert::kPhoton, projStored, target_type,
                                                  nucStored, par, epo, *ws, rng);
        r.collider_refusal = epo.refusal;
        if (epo.refusal != bert::ColliderRefusal::kNone) {
          r.refusal = EmExtraRefusal::kSubModel;
          return r;
        }
        if (!ok) {
          // "Failed to generate final state" - the projectile and the target nucleon are
          // pushed back as dummies, NOT in the CM frame, and are then boosted as though they
          // were. Transcribed, and flagged.
          r.scattering_failed = true;
          epo.n = 2;
          epo.kinds[0] = bert::kPhoton;
          epo.momenta[0] = projStored;
          epo.kinds[1] = target_type;
          epo.momenta[1] = nucStored;
        }
        for (int i = 0; i < epo.n; ++i) {
          LV m = epo.momenta[i];
          m.boost(Vec3d{-betacm.x, -betacm.y, -betacm.z});
          if (!bert::co_add_particle(out, epo.kinds[i], m)) {
            r.refusal = EmExtraRefusal::kCapacity;
            return r;
          }
        }
        // "Add the recoil nucleon unmodified" - in the LAB frame, where it was built, and with
        // no boost at all.
        if (!bert::co_add_particle(out, spectator_type, spectator)) {
          r.refusal = EmExtraRefusal::kCapacity;
          return r;
        }
      } else {
        r.arm = LightTargetArm::kDeuteronAbsorption;
        int kinds[2];
        LV momenta[2];
        absorption_on_deuteron(bullet, bullet_mass, kinds, momenta, rng);
        for (int i = 0; i < 2; ++i) {
          if (!bert::co_add_particle(out, kinds[i], momenta[i])) {
            r.refusal = EmExtraRefusal::kCapacity;
            return r;
          }
        }
      }
    }
  }

  // copyOutputToHadronicResult, as G4CascadeInterface does it for every branch.
  r.n_products = copy_light_output(out, fs, rng);
  if (fs.secondary_overflow > 0) { r.refusal = EmExtraRefusal::kCapacity; }
  return r;
}

}  // namespace g4gpu::physics::hadronic::emextra

#endif
