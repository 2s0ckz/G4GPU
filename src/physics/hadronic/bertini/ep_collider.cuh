// G4ElementaryParticleCollider and G4CascadeFinalStateAlgorithm: one hadron-hadron collision,
// from a channel-table lookup to a list of four-vectors in the lab frame.
//
// Transcribed from Geant4 11.1.1:
//   G4ElementaryParticleCollider::collide / generateMultiplicity /
//     generateOutgoingPartTypes / generateSCMfinalState / fillOutgoingMasses /
//     generateSCMpionAbsorption / generateSCMpionNAbsorption / pionNucleonAbsorption
//                                   (cascade/cascade/src/G4ElementaryParticleCollider.cc)
//   G4CascadeFinalStateAlgorithm::Configure / SaveKinematics / ChooseGenerators /
//     GenerateTwoBody / GenerateMultiBody / FillMagnitudes / satisfyTriangle /
//     FillDirections / FillDirThreeBody / FillDirManyBody / GenerateCosTheta /
//     FillUsingKopylov / BetaKopylov
//                                   (cascade/cascade/src/G4CascadeFinalStateAlgorithm.cc)
//   G4VHadDecayAlgorithm::Generate / IsDecayAllowed / TwoBodyMomentum / UniformTheta /
//     UniformPhi                    (hadronic/util/src/G4VHadDecayAlgorithm.cc)
//   G4HadDecayGenerator::Generate / GenerateOneBody
//                                   (hadronic/util/src/G4HadDecayGenerator.cc)
//
// ---------------------------------------------------------------------------------------------
// **Which particle is "the bullet" is decided twice, by two different rules, and they disagree.**
//
// `collide` sets the frame with
//     if (particle2->nucleon() || particle2->quasi_deutron()) { bullet=1; target=2; }
//     else                                                    { bullet=2; target=1; }
// and `G4CascadeFinalStateAlgorithm::SaveKinematics`, called a few lines later through
// `Configure`, sets it again with
//     if (target->nucleon()) { bullet=1; target=2; } else { bullet=2; target=1; }
// - no quasi-deuteron arm. For a pion on a dibaryon the two therefore choose OPPOSITE
// assignments, and since `toSCM` is what `rotate()` builds the final state around, the axis is
// reversed between the frame `collide` boosts back from and the frame the algorithm rotated in.
// It does not matter for the absorption channels, because those never reach
// `generateSCMfinalState` - a dibaryon target goes to `generateSCMpionAbsorption`, which builds
// its own back-to-back pair and does not rotate. Transcribed as two separate decisions, under
// their own names, so that the disagreement stays visible.
//
// **`GenerateCosTheta`'s two arguments swap meaning by overload resolution, and that is
// deliberate.** For multiplicity 3 it calls `angDist->GetCosTheta(bullet_ekin, ptype)` with a
// double and an int, against the two overloads `(G4int, G4double)` and
// `(const G4double&, const G4double&)`. The second wins - one exact match beats none - and
// `G4VThreeBodyAngDst` implements it as `GetCosTheta((G4int)pcm, ekin)`, i.e. it swaps them
// back. The comment in that header says "re-interpret 'pcm' as ptype". So the three-body
// angular distribution is called with (ptype, ekin) after all, through a double round trip. The
// port calls the (ptype, ekin) form directly and this comment is the reason it is allowed to.
//
// **The multi-body polar angle is a rejection sampler on sin(theta), not cos(theta).** Its
// `while (|sinth| > maxCosTheta && ...)` guard uses maxCosTheta (0.9999) as a SIN limit, and on
// failure after ten tries it falls back to `sinth = 0.5*u` - a half-uniform sine, not an
// isotropic angle. Then `costh = +-sqrt(1-sinth^2)` with the sign thrown separately, so the
// distribution is symmetric about 90 degrees by construction and the generator has no way to
// prefer forward over backward. That is Bertini's high-energy multi-body angular model.
//
// **A final state generated on the TENTH attempt is thrown away.** `generateSCMfinalState`'s
// loop is `while (generate && itry++ < itry_max)` and its exit test is `if (itry >= itry_max)
// return;`. When the tenth pass succeeds, `generate` goes false and the loop condition
// short-circuits BEFORE the post-increment, so `itry` stops at exactly `itry_max` - and the
// exit test, which was written to catch exhaustion, fires on the success too. The particles
// buffer is never filled and `collide` passes the bullet through. Nine usable attempts, not
// ten. Transcribed as written: `ep_generate_scm_final_state` counts with Geant4's own
// post-increment and asks `fs_retry_exhausted`, which is a separate named function precisely so
// that the `>=` can be asserted on its own. It has to be, because **the oracle grid cannot see
// this one**: no case among the 10,752 succeeds on its tenth pass, so every empty case is plain
// exhaustion (itry = 11) and the perturbation `>=` -> `>` passes all of them. V52's distinction
// - a question, pinned by construction, not a hole. docs/RISK.md V121.
#ifndef G4GPU_BERTINI_EP_COLLIDER_CUH
#define G4GPU_BERTINI_EP_COLLIDER_CUH

#include <cmath>

#include "physics/hadronic/bertini/angular_dist.cuh"
#include "physics/hadronic/bertini/channel_tables.cuh"
#include "physics/hadronic/bertini/inucl_particle.cuh"
#include "physics/hadronic/bertini/lorentz_convertor.cuh"
#include "physics/hadronic/bertini/nuclei_model.cuh"
#include "physics/hadronic/bertini/workspace.cuh"

namespace g4gpu::physics::hadronic::bert {

/// G4CascadeFinalStateAlgorithm's four constants.
constexpr double kFsMaxCosTheta = 0.9999;
constexpr double kFsOneOverE = 0.3678794;   ///< NOT 1/e to full precision: seven digits, and
                                            ///< it multiplies the rejection envelope, so the
                                            ///< truncation is part of the sampled distribution
constexpr double kFsSmall = 1.0e-10;
constexpr int kFsItryMax = 10;

/// `generateSCMfinalState`'s exit test, on its own so that it can be asserted on its own.
///
/// `itry` is Geant4's counter after the loop `while (generate && itry++ < itry_max)`, so it is
/// 11 when all ten passes failed and exactly `itry_max` when the TENTH pass succeeded - the
/// short-circuit on `generate` skips that last increment. Geant4 rejects both with one `>=`.
/// See the header comment: the second case is invisible in the oracle, and this is where it is
/// pinned instead.
__host__ __device__ inline bool fs_retry_exhausted(int itry) { return itry >= kFsItryMax; }

/// What a collision could not do. Every one is reported; none is a silent empty final state.
enum class ColliderRefusal {
  kNone = 0,
  kNotHadronHadron,         ///< useEPCollider's sanity check
  kNeutrinoProjectile,      ///< collide() returns immediately for a neutrino
  kNoChannelTable,          ///< no table and neither particle is a dibaryon
  kIllegalDibaryonPartner,  ///< useQuasiDeuteron rejects the pair
  kMuonAbsorption,          ///< generateSCMmuonAbsorption - P12's, refused by name
  kIllegalAbsorption,       ///< the type product matches none of the nine absorption cases
  kIllegalPionNAbsorption,  ///< pi-p / pi+n charge exchange expected and not found
  kPionNAbsorptionNucleus,  ///< the pi-N-on-a-bound-nucleon arm fired but no (A, Z) was given
  kKinematicsFailed,        ///< itry_max attempts without a valid final state
  kMultiplicityTooLarge,    ///< more than kMaxFinalStateSize products
  kChannelRefused           ///< the channel table itself refused - see ChannelRefusal
};

/// Which of the twelve verdicts above are THIS PORT declining to do something, and which are
/// Geant4 itself producing no final state.
///
/// The distinction matters to every caller inside the cascade: Geant4's `collide` ends with
/// `if (particles.empty()) return;` and the cascade treats that as "the collision did not
/// happen" - it breaks out of the partner loop and propagates the particle instead. A neutrino
/// projectile, a missing channel table, an illegal dibaryon partner and ten failed kinematic
/// attempts are all that case, and a cascade that stopped on them would be a different model. A
/// muon absorption, a pi-N absorption without a nucleus, an out-of-range multiplicity and a
/// refused channel table are not: they are places where Geant4 would have produced something
/// and this port will not, and they have to reach the caller by name.
__host__ __device__ inline bool collider_refusal_is_port_limit(ColliderRefusal r) {
  return r == ColliderRefusal::kMuonAbsorption ||
         r == ColliderRefusal::kPionNAbsorptionNucleus ||
         r == ColliderRefusal::kMultiplicityTooLarge ||
         r == ColliderRefusal::kChannelRefused;
}

/// One collision's output, sized by the largest final state a channel table can name.
struct ColliderOutput {
  int n = 0;
  int kinds[kMaxFinalStateSize];
  LV momenta[kMaxFinalStateSize];
  ColliderRefusal refusal = ColliderRefusal::kNone;
  ChannelRefusal channel_refusal = ChannelRefusal::kNone;
  bool sample_fell_off = false;   ///< G4CascadeSampler::sampleFlat's "Is this right?" branch
};

/// G4VHadDecayAlgorithm::TwoBodyMomentum. Geant4 THROWS when PSQ < -eV and clamps to zero
/// otherwise; a kernel cannot throw, so a significantly negative PSQ is reported through
/// `bad_kinematics` and the momentum is the clamped zero.
__host__ __device__ inline double two_body_momentum(double m0, double m1, double m2,
                                                    bool& bad_kinematics) {
  double psq = (m0 + m1 + m2) * (m0 + m1 - m2) * (m0 - m1 + m2) * (m0 - m1 - m2);
  bad_kinematics = false;
  if (psq < 0.0) {
    // CLHEP::eV in Bertini's GeV units.
    bad_kinematics = (psq < -1.0e-9);
    psq = 0.0;
  }
  return std::sqrt(psq) / (2.0 * m0);
}

template <typename Rng>
__host__ __device__ inline double uniform_theta(Rng& rng) {
  return std::acos(2.0 * rng.uniform() - 1.0);
}
template <typename Rng>
__host__ __device__ inline double uniform_phi(Rng& rng) {
  return 6.283185307179586476925286766559 * rng.uniform();
}

// `FinalStateConfig` - the state G4CascadeFinalStateAlgorithm::Configure builds and the
// generators then read - lives in workspace.cuh, because it is 264 bytes of arrays and the
// workspace is where this package puts those. In Geant4 it is not a struct at all: `kinds`,
// `masses`, `masses2`, `modules`, `toSCM` and `bullet_ekin` are DATA MEMBERS of
// G4CascadeFinalStateAlgorithm and G4ElementaryParticleCollider, allocated once and reused, so
// a per-call struct on the stack would be a change of shape as well as a cost. Measured: the
// device probe in tests/test_bertini_collide.cu goes from an 880-byte frame and 184 registers
// to 416 bytes and 158 registers when it moves.

/// G4CascadeFinalStateAlgorithm::ChooseGenerators.
///
/// `angDist` is null for every multiplicity above three, which is why `GenerateCosTheta` has a
/// separate many-body branch; and it is null for a two-body state whose final state is not in
/// the table (`fs > 0` fails), which cannot happen because `fs` is a product of two non-zero
/// type codes. `kw` is 1 for elastic (fs == is) and 2 otherwise, and that is the ONLY place kw
/// comes from - the "strangeness production" arms of ChooseDist are reached exactly when the
/// two-body final state differs from the initial state.
__host__ __device__ inline void fs_choose_generators(FinalStateConfig& cfg, int is, int fs,
                                                     const CascadeParams& par) {
  cfg.use_phase_space = par.use_phase_space;
  cfg.mom_index = par.use_phase_space
                      ? -1
                      : choose_multibody_momdst(is, cfg.multiplicity, par);
  if (fs > 0 && cfg.multiplicity == 2) {
    const int kw = (fs == is) ? 1 : 2;
    cfg.ang = choose_two_body_angdst(is, fs, kw);
  } else if (cfg.multiplicity == 3) {
    cfg.ang = choose_two_body_angdst(is, 0, 0);
  } else {
    cfg.ang = AngDstChoice();
  }
}

/// G4CascadeFinalStateAlgorithm::SaveKinematics. See the header comment: this is the SECOND,
/// different, bullet/target decision, and it has no quasi-deuteron arm.
__host__ __device__ inline void fs_save_kinematics(FinalStateConfig& cfg, int type1,
                                                   const LV& mom1, int type2, const LV& mom2) {
  if (inucl_is_nucleon(type2)) {
    cfg.to_scm.bullet = mom1;
    cfg.to_scm.target = mom2;
  } else {
    cfg.to_scm.bullet = mom2;
    cfg.to_scm.target = mom1;
  }
  lc_to_the_center_of_mass(cfg.to_scm);
  cfg.bullet_ekin = lc_kin_energy_in_trs(cfg.to_scm);
}

/// G4CascadeFinalStateAlgorithm::GenerateCosTheta.
///
/// `p0` is 0.36 for a nucleon and 0.25 for everything else, selected by `ptype < 3` - which is
/// true for the proton (1) and the neutron (2) and false for every other code including the
/// negative lepton ones, so a muon gets 0.25.
template <typename Rng>
__host__ __device__ inline double fs_generate_cos_theta(const FinalStateConfig& cfg, int ptype,
                                                        double pmod, Rng& rng,
                                                        bool& used_fallback) {
  used_fallback = false;
  if (cfg.multiplicity == 3) {
    bool flat = false;
    return paramang_cos_theta(data::bertini_paramang_angdst()[cfg.ang.index], ptype,
                              cfg.bullet_ekin, rng, flat);
  }
  const double p0 = (ptype < 3) ? 0.36 : 0.25;
  const double alf = 1.0 / p0 / (p0 - (pmod + p0) * std::exp(-pmod / p0));
  double sinth = 2.0;
  int itry1 = -1;
  while (std::fabs(sinth) > kFsMaxCosTheta && ++itry1 < kFsItryMax) {
    const double s1 = pmod * rng.uniform();
    const double s2 = alf * kFsOneOverE * p0 * rng.uniform();
    const double salf = s1 * alf * std::exp(-s1 / p0);
    if (salf > s2) { sinth = s1 / pmod; }
  }
  if (itry1 == kFsItryMax) {
    used_fallback = true;
    sinth = 0.5 * rng.uniform();
  }
  double costh = std::sqrt(1.0 - sinth * sinth);
  if (rng.uniform() > 0.5) { costh = -costh; }
  return costh;
}

/// G4CascadeFinalStateAlgorithm::GenerateTwoBody.
template <typename Rng>
__host__ __device__ inline bool fs_generate_two_body(const FinalStateConfig& cfg,
                                                     double initial_mass, LV* final_state,
                                                     Rng& rng, bool& bad_kinematics) {
  if (cfg.multiplicity != 2) { return false; }
  const double pscm =
      two_body_momentum(initial_mass, cfg.masses[0], cfg.masses[1], bad_kinematics);
  double costh;
  if (cfg.ang.kind == AngDstKind::kNumInt) {
    costh = numint_cos_theta(data::bertini_numint_angdst()[cfg.ang.index], cfg.bullet_ekin,
                             pscm, rng);
  } else if (cfg.ang.kind == AngDstKind::kParamExp) {
    costh = paramexp_cos_theta(data::bertini_paramexp_angdst()[cfg.ang.index], cfg.bullet_ekin,
                               pscm, rng);
  } else if (cfg.ang.kind == AngDstKind::kParamAng3Body) {
    bool flat = false;
    costh = paramang_cos_theta(data::bertini_paramang_angdst()[cfg.ang.index], cfg.kinds[0],
                               cfg.bullet_ekin, rng, flat);
  } else {
    // ChooseDist returned null: an isotropic angle, which is the `(2.*G4UniformRand() - 1.)`
    // fallback in GenerateTwoBody. Reachable only for an initial state ChooseDist has no arm
    // for, which the channel tables should make impossible - but the fallback is what runs if
    // it happens, not an error.
    costh = 2.0 * rng.uniform() - 1.0;
  }
  // `mom.setRThetaPhi(pscm, acos(costh), UniformPhi())`.
  const double theta = std::acos(costh);
  const double phi = uniform_phi(rng);
  const double st = std::sin(theta);
  const Vec3d p{pscm * st * std::cos(phi), pscm * st * std::sin(phi), pscm * std::cos(theta)};
  final_state[0] = lc_rotate(cfg.to_scm, lv_set_vect_m(p, cfg.masses[0]));
  final_state[1] = lv_set_vect_m(Vec3d{-final_state[0].v.x, -final_state[0].v.y,
                                       -final_state[0].v.z},
                                 cfg.masses[1]);
  return true;
}

/// G4CascadeFinalStateAlgorithm::FillMagnitudes.
///
/// The loop is subtle in two ways. It breaks out of the inner `for` on either a too-small
/// momentum or a `eleft <= mass_last`, and then `if (i < multiplicity-1) continue` sends it
/// round again - so a partial set of moduli is discarded rather than patched. And the LAST
/// momentum is not sampled: it is whatever energy is left, `sqrt(eleft^2 - m_last^2)`, so the
/// final particle carries all the accumulated rounding of the others.
template <typename Rng>
__host__ __device__ inline bool fs_fill_magnitudes(FinalStateConfig& cfg, double initial_mass,
                                                   Rng& rng) {
  if (cfg.mom_index < 0) { return false; }
  const int mult = cfg.multiplicity;
  const double mass_last = cfg.masses[mult - 1];
  int itry = -1;
  while (++itry < kFsItryMax) {
    double eleft = initial_mass;
    int i = 0;
    for (; i < mult - 1; ++i) {
      const double pmod = parammom_momentum(data::bertini_parammom_momdst()[cfg.mom_index],
                                            cfg.kinds[i], cfg.bullet_ekin, rng);
      if (pmod < kFsSmall) { break; }
      eleft -= std::sqrt(pmod * pmod + cfg.masses[i] * cfg.masses[i]);
      if (eleft <= mass_last) { break; }
      cfg.modules[i] = pmod;
    }
    if (i < mult - 1) { continue; }
    double plast = eleft * eleft - mass_last * mass_last;
    if (plast <= kFsSmall) { continue; }
    plast = std::sqrt(plast);
    cfg.modules[mult - 1] = plast;
    // satisfyTriangle, for three-body states only. Its return is `(size != 3) || !(...)`, so a
    // multiplicity above three passes trivially - and the caller's `multiplicity > 3 ||` in
    // front of it means the function is only ever called with exactly three.
    if (mult > 3) { return true; }
    const double* p = cfg.modules;
    const bool ok = !(p[0] < std::fabs(p[1] - p[2]) || p[0] > p[1] + p[2] ||
                      p[1] < std::fabs(p[0] - p[2]) || p[1] > p[0] + p[2] ||
                      p[2] < std::fabs(p[0] - p[1]) || p[2] > p[1] + p[0]);
    if (ok) { return true; }
  }
  return false;
}

/// G4CascadeFinalStateAlgorithm::FillDirThreeBody.
///
/// The third particle is thrown first, at an angle from the three-body distribution; the FIRST
/// is then placed at the angle the momentum triangle forces, relative to the third; and the
/// second is pure recoil, `(0,0,0,M) - p0 - p2`, so it is the only one whose mass is not
/// imposed. `|costh| >= maxCosTheta` aborts the whole generation rather than clamping.
template <typename Rng>
__host__ __device__ inline bool fs_fill_dir_three_body(const FinalStateConfig& cfg,
                                                       double initial_mass, LV* final_state,
                                                       Rng& rng) {
  bool fallback = false;
  double costh = fs_generate_cos_theta(cfg, cfg.kinds[2], cfg.modules[2], rng, fallback);
  final_state[2] =
      lc_rotate(cfg.to_scm, inucl_with_fixed_theta(costh, cfg.modules[2], cfg.masses[2], rng));
  costh = -0.5 *
          (cfg.modules[2] * cfg.modules[2] + cfg.modules[0] * cfg.modules[0] -
           cfg.modules[1] * cfg.modules[1]) /
          cfg.modules[2] / cfg.modules[0];
  if (std::fabs(costh) >= kFsMaxCosTheta) { return false; }
  final_state[0] = lc_rotate_about(
      cfg.to_scm, final_state[2],
      inucl_with_fixed_theta(costh, cfg.modules[0], cfg.masses[0], rng));
  final_state[1] = LV(Vec3d{0.0, 0.0, 0.0}, initial_mass);
  final_state[1] -= final_state[0] + final_state[2];
  return true;
}

/// G4CascadeFinalStateAlgorithm::FillDirManyBody.
template <typename Rng>
__host__ __device__ inline bool fs_fill_dir_many_body(const FinalStateConfig& cfg,
                                                      double initial_mass, LV* final_state,
                                                      Rng& rng) {
  const int mult = cfg.multiplicity;
  bool fallback = false;
  for (int i = 0; i < mult - 2; ++i) {
    const double costh = fs_generate_cos_theta(cfg, cfg.kinds[i], cfg.modules[i], rng, fallback);
    final_state[i] = lc_rotate(
        cfg.to_scm, inucl_with_fixed_theta(costh, cfg.modules[i], cfg.masses[i], rng));
  }
  LV psum;
  for (int i = 0; i < mult - 2; ++i) { psum += final_state[i]; }
  const double pmod = lv_rho(psum);
  const double costh = -0.5 *
                       (pmod * pmod + cfg.modules[mult - 2] * cfg.modules[mult - 2] -
                        cfg.modules[mult - 1] * cfg.modules[mult - 1]) /
                       pmod / cfg.modules[mult - 2];
  if (std::fabs(costh) >= kFsMaxCosTheta) { return false; }
  final_state[mult - 2] = lc_rotate_about(
      cfg.to_scm, psum,
      inucl_with_fixed_theta(costh, cfg.modules[mult - 2], cfg.masses[mult - 2], rng));
  final_state[mult - 1] = LV(Vec3d{0.0, 0.0, 0.0}, initial_mass);
  final_state[mult - 1] -= psum + final_state[mult - 2];
  return true;
}

/// G4CascadeFinalStateAlgorithm::BetaKopylov, the N-body phase-space beta variable.
template <typename Rng>
__host__ __device__ inline double fs_beta_kopylov(int k, Rng& rng) {
  const int n = 3 * k - 5;
  const double xn = double(n);
  const double fmax = std::sqrt(g4pow_n(xn / (xn + 1.0), n) / (xn + 1.0));
  double f, chi;
  do {
    chi = rng.uniform();
    f = std::sqrt(g4pow_n(chi, n) * (1.0 - chi));
  } while (fmax * rng.uniform() > f);
  return chi;
}

/// G4CascadeFinalStateAlgorithm::FillUsingKopylov - the `usePhaseSpace` branch.
///
/// Off by default (the dumped `usePhaseSpace` is 0) and transcribed anyway because it is cheap:
/// it is a chain of two-body decays with the recoil system's mass shrinking at each step. Its
/// own FIXME says the theta distribution "should use Bertini fit function" and does not.
template <typename Rng>
__host__ __device__ inline void fs_fill_using_kopylov(int n, double initial_mass,
                                                      const double* masses, LV* final_state,
                                                      Rng& rng, bool& bad_kinematics) {
  double mtot = 0.0;
  for (int i = 0; i < n; ++i) { mtot += masses[i]; }
  double mu = mtot;
  double mass = initial_mass;
  double t = mass - mtot;
  LV recoil(Vec3d{0.0, 0.0, 0.0}, mass);
  bad_kinematics = false;
  for (int k = n - 1; k > 0; --k) {
    mu -= masses[k];
    t *= (k > 1) ? fs_beta_kopylov(k, rng) : 0.0;
    const double recoil_mass = mu + t;
    const Vec3d boost = recoil.boost_vector();
    bool bad = false;
    const double p = two_body_momentum(mass, masses[k], recoil_mass, bad);
    bad_kinematics = bad_kinematics || bad;
    // **phi is drawn BEFORE theta, and the C++ standard does not say so.** Geant4 writes
    //
    //     momV.setRThetaPhi(TwoBodyMomentum(Mass,masses[k],recoilMass),
    //                       UniformTheta(), UniformPhi());
    //
    // - three function calls in one argument list, two of which consume a random deviate. Their
    // order is indeterminately sequenced, and the compiler that built this oracle (MSVC 19.29,
    // x64) evaluates the list RIGHT TO LEFT, so `UniformPhi` takes the first deviate and
    // `UniformTheta` the second. Drawing theta first gives the same NUMBER of deviates and
    // different angles, which is why this cost a bisection rather than a reading: the draw
    // counts matched on all 2,688 phase-space cases and the four-momenta did not. Reproduced as
    // the oracle does it. docs/RISK.md V123 - and note that this makes the phase-space branch
    // compiler-dependent in Geant4 itself, so a Linux build of 11.1.1 may sample it the other
    // way round.
    const double phi = uniform_phi(rng);
    const double theta = uniform_theta(rng);
    const double st = std::sin(theta);
    const Vec3d mv{p * st * std::cos(phi), p * st * std::sin(phi), p * std::cos(theta)};
    final_state[k] = lv_set_vect_m(mv, masses[k]);
    recoil = lv_set_vect_m(Vec3d{-mv.x, -mv.y, -mv.z}, recoil_mass);
    final_state[k].boost(boost);
    recoil.boost(boost);
    mass = recoil_mass;
  }
  final_state[0] = recoil;
}

/// G4VHadDecayAlgorithm::IsDecayAllowed - the gate in front of BOTH generators, and the only
/// thing that stops a final state heavier than the available energy.
///
/// Without it `TwoBodyMomentum` would clamp a negative PSQ to zero and hand back a pair of
/// particles at rest, which `generateSCMfinalState` would then ACCEPT: `Generate` returns
/// `!finalState.empty()`, and a clamped-to-zero two-body state is not empty. With it the
/// generator returns an empty state, the retry loop draws a new multiplicity and a new final
/// state, and the collision is resampled. The distinction is not academic - near a channel's
/// threshold the sampler happily proposes states it cannot pay for. docs/RISK.md V122, which
/// also records where this had to be read from: two base classes up, in hadronic/util, reached
/// from the cascade code only as `fsGenerator.Generate(...)`.
__host__ __device__ inline bool fs_is_decay_allowed(const FinalStateConfig& cfg,
                                                    double initial_mass) {
  if (!(initial_mass > 0.0) || cfg.multiplicity < 2) { return false; }
  double msum = 0.0;
  for (int i = 0; i < cfg.multiplicity; ++i) { msum += cfg.masses[i]; }
  return initial_mass >= msum;
}

/// G4CascadeFinalStateAlgorithm::GenerateMultiBody plus Generate's two-body arm.
///
/// `G4HadDecayGenerator::Generate` also has a one-body arm (`masses.size() == 1U`), which is
/// unreachable here: every channel table's lowest multiplicity is 2 and `getMultiplicity`
/// returns `sampled + 2`, so `cfg.multiplicity` is never 1. `fs_is_decay_allowed` refuses it
/// rather than falling through to the two-body branch.
template <typename Rng>
__host__ __device__ inline bool fs_generate(FinalStateConfig& cfg, double initial_mass,
                                            LV* final_state, Rng& rng, bool& bad_kinematics) {
  bad_kinematics = false;
  if (!fs_is_decay_allowed(cfg, initial_mass)) { return false; }
  if (cfg.multiplicity == 2) {
    return fs_generate_two_body(cfg, initial_mass, final_state, rng, bad_kinematics);
  }
  if (cfg.use_phase_space) {
    fs_fill_using_kopylov(cfg.multiplicity, initial_mass, cfg.masses, final_state, rng,
                          bad_kinematics);
    return true;
  }
  if (cfg.multiplicity < 3 || cfg.mom_index < 0) { return false; }
  int itry = -1;
  while (++itry < kFsItryMax) {
    if (!fs_fill_magnitudes(cfg, initial_mass, rng)) { continue; }
    const bool ok = (cfg.multiplicity == 3)
                        ? fs_fill_dir_three_body(cfg, initial_mass, final_state, rng)
                        : fs_fill_dir_many_body(cfg, initial_mass, final_state, rng);
    if (ok) { return true; }
  }
  return false;
}

// =============================================================================================
// G4ElementaryParticleCollider
// =============================================================================================

/// G4ElementaryParticleCollider::pionNucleonAbsorption. With the dumped `piNAbsorption` of 0
/// the last conjunct `u < absProb` is false for every deviate, so the branch never fires - but
/// it still DRAWS the deviate whenever the first two conjuncts hold, which is a pi- p or pi+ n
/// collision below 50 MeV. That draw is part of the random stream and is reproduced.
template <typename Rng>
__host__ __device__ inline bool ep_pion_nucleon_absorption(int is, double ekin,
                                                           const CascadeParams& par,
                                                           Rng& rng) {
  if (!(is == kPionMinus * kProton || is == kPionPlus * kNeutron)) { return false; }
  if (!(ekin < 0.05)) { return false; }
  return rng.uniform() < par.pin_absorption;
}

/// G4ElementaryParticleCollider::generateSCMpionAbsorption - pion or photon on a dibaryon.
///
/// Nine type products map to three nucleon pairs; anything else is `Illegal absorption`. The
/// two nucleons are back to back and isotropic in the SCM, with `pmod` from the standard
/// two-body form written out rather than through TwoBodyMomentum - `a = (E^2-m1^2-m2^2)/2` and
/// `p = sqrt((a^2 - m1^2 m2^2)/E^2)` - which is the same number in exact arithmetic and a
/// different one in floating point, so it is written the way Geant4 writes it.
template <typename Rng>
__host__ __device__ inline void ep_generate_scm_pion_absorption(double etot_scm, int type1,
                                                                int type2, ColliderOutput& out,
                                                                Rng& rng) {
  const int tp = type1 * type2;
  int k0, k1;
  if (tp == kPionZero * kDiproton || tp == kPionPlus * kUnboundPN ||
      tp == kPhoton * kDiproton) {
    k0 = kProton; k1 = kProton;
  } else if (tp == kPionMinus * kDiproton || tp == kPionPlus * kDineutron ||
             tp == kPionZero * kUnboundPN || tp == kPhoton * kUnboundPN) {
    k0 = kProton; k1 = kNeutron;
  } else if (tp == kPionZero * kDineutron || tp == kPionMinus * kUnboundPN ||
             tp == kPhoton * kDineutron) {
    k0 = kNeutron; k1 = kNeutron;
  } else {
    out.refusal = ColliderRefusal::kIllegalAbsorption;
    out.n = 0;
    return;
  }
  const double m0 = inucl_particle_mass(k0);
  const double m1 = inucl_particle_mass(k1);
  const double m02 = m0 * m0;
  const double m12 = m1 * m1;
  const double a = 0.5 * (etot_scm * etot_scm - m02 - m12);
  const double pmod = std::sqrt((a * a - m02 * m12) / (etot_scm * etot_scm));
  const LV mom1 = inucl_with_random_angles(pmod, m0, rng);
  out.n = 2;
  out.kinds[0] = k0;
  out.kinds[1] = k1;
  out.momenta[0] = mom1;
  out.momenta[1] = lv_set_vect_m(Vec3d{-mom1.v.x, -mom1.v.y, -mom1.v.z}, m1);
}

/// G4ElementaryParticleCollider::generateSCMpionNAbsorption - pi- p -> n or pi+ n -> p with the
/// REST OF THE NUCLEUS taking the recoil. It needs (A, Z), which is why
/// `setNucleusState` exists on the collider at all, and the recoil mass is
/// `G4InuclNuclei::getNucleiMass(A-1, Z-(2-ntype))` - the nucleus minus the struck nucleon.
///
/// Only the ejected nucleon is filled: the recoiling nucleus is NOT added to the output, so the
/// event does not conserve momentum at this level and the cascader's recoil bookkeeping is what
/// picks it up. Unreachable with the dumped piNAbsorption of 0.
template <typename Rng>
__host__ __device__ inline void ep_generate_scm_pion_n_absorption(int type1, const LV& mom1,
                                                                  int type2, const LV& mom2,
                                                                  int nucleus_a, int nucleus_z,
                                                                  double recoil_mass,
                                                                  ColliderOutput& out,
                                                                  Rng& rng) {
  const int tp = type1 * type2;
  if (tp != kPionMinus * kProton && tp != kPionPlus * kNeutron) {
    out.refusal = ColliderRefusal::kIllegalPionNAbsorption;
    out.n = 0;
    return;
  }
  (void)nucleus_a;
  (void)nucleus_z;
  const int ntype = inucl_is_nucleon(type2) ? type2 : type1;
  const int out_type = 3 - ntype;    // proton is 1, neutron is 2, so 3-# swaps them
  const double m0 = inucl_particle_mass(out_type);
  const double m02 = m0 * m0;
  const double mr2 = recoil_mass * recoil_mass;
  const LV pi_n = mom1 + mom2;
  LV vsum(Vec3d{0.0, 0.0, 0.0}, recoil_mass);
  vsum += pi_n;
  const double esq_scm = vsum.e * vsum.e - g4gpu::mag2(vsum.v);
  const double a = 0.5 * (esq_scm - m02 - mr2);
  const double pmod = std::sqrt((a * a - m02 * mr2) / esq_scm);
  LV m = inucl_with_random_angles(pmod, m0, rng);
  const Vec3d bv = pi_n.boost_vector();
  m.boost(Vec3d{-bv.x, -bv.y, -bv.z});
  out.n = 1;
  out.kinds[0] = out_type;
  out.momenta[0] = m;
}

/// G4ElementaryParticleCollider::generateSCMfinalState - the retry loop around the channel
/// tables and the kinematics.
///
/// Ten attempts; each one re-samples the multiplicity AND the final state, so a channel whose
/// kinematics are impossible at this energy is not retried with the same products - it is
/// retried from the multiplicity up. On failure the output is EMPTY and `collide` passes the
/// bullet through unchanged.
///
/// The loop is written with Geant4's own counting, post-increment inside the condition, because
/// the exit test `itry >= itry_max` cannot be simplified to "did it generate?" - see the header
/// comment: a success on the tenth pass leaves `itry == itry_max` and is discarded.
template <typename Rng>
__host__ __device__ inline void ep_generate_scm_final_state(double ekin, double etot_scm,
                                                            int type1, const LV& mom1,
                                                            int type2, const LV& mom2,
                                                            const CascadeParams& par,
                                                            ColliderOutput& out,
                                                            BertiniWorkspace& ws, Rng& rng) {
  const int is = type1 * type2;
  const ChannelTable t = channel_table(is);
  if (!t.valid()) {
    out.refusal = ColliderRefusal::kNoChannelTable;
    out.n = 0;
    return;
  }
  FinalStateConfig& cfg = ws.fs_cfg;
  cfg = FinalStateConfig();
  int itry = 0;
  bool generate = true;
  while (generate && itry++ < kFsItryMax) {
    out.n = 0;
    bool fell = false;
    const int mult = channel_multiplicity(t, ekin, rng, fell);
    out.sample_fell_off = out.sample_fell_off || fell;
    // A multiplicity above the table's maximum is a THIRD out-of-bounds read in this tree, and
    // it is refused rather than reproduced. `getOutgoingParticleTypes` prints "Illegal
    // multiplicity" and clamps, so `particle_kinds` and `masses` come back at the clamped size
    // and `Configure` gives the algorithm that size - but `generateSCMfinalState` keeps its own
    // unclamped `multiplicity` and ends with `particles.resize(multiplicity)` and a loop that
    // reads `scm_momentums[i]` and `particle_kinds[i]` past the end of both. Unreachable from a
    // sampler: `getMultiplicity` returns a row index plus 2 and the index is bounded by the
    // multiplicity table's own row count, so `mult <= maxMultiplicity()` always. (The "Illegal
    // multiplicity" lines the oracle run prints come from dump_chsample, which asks every
    // channel for every multiplicity 2..9 on purpose, and that clamp IS reproduced - see
    // `outgoing_particle_types`. This is the caller's half of it, which is not.)
    if (mult > t.max_multiplicity() || mult > kMaxFinalStateSize) {
      out.refusal = ColliderRefusal::kMultiplicityTooLarge;
      return;
    }
    int chan = -1;
    const ChannelRefusal cr =
        outgoing_particle_types(t, mult, ekin, rng, cfg.kinds, chan, fell, ws.sigma_buf);
    out.sample_fell_off = out.sample_fell_off || fell;
    if (cr != ChannelRefusal::kNone) {
      out.channel_refusal = cr;
      out.refusal = ColliderRefusal::kChannelRefused;
      return;
    }
    cfg.multiplicity = mult;
    // fillOutgoingMasses
    for (int i = 0; i < mult; ++i) {
      cfg.masses[i] = inucl_particle_mass(cfg.kinds[i]);
      cfg.masses2[i] = cfg.masses[i] * cfg.masses[i];
    }
    // Configure: ChooseGenerators, then SaveKinematics.
    const int fs = (mult == 2) ? cfg.kinds[0] * cfg.kinds[1] : 0;
    fs_choose_generators(cfg, is, fs, par);
    fs_save_kinematics(cfg, type1, mom1, type2, mom2);
    bool bad = false;
    generate = !fs_generate(cfg, etot_scm, out.momenta, rng, bad);
  }
  // `if (itry >= itry_max) return;` - exhaustion AND a tenth-pass success, both.
  if (fs_retry_exhausted(itry)) {
    out.refusal = ColliderRefusal::kKinematicsFailed;
    out.n = 0;
    return;
  }
  out.n = cfg.multiplicity;
  for (int i = 0; i < cfg.multiplicity; ++i) { out.kinds[i] = cfg.kinds[i]; }
}

/// G4ElementaryParticleCollider::collide.
///
/// The three arms are, in order: a nucleon target (the channel tables), a dibaryon target with a
/// muon (refused - that is P12's muon capture), and a dibaryon target with a pion or photon
/// (absorption). Note that the first two tests are `if`, not `else if`, so a collision between a
/// nucleon and a dibaryon would enter both - which cannot happen because a dibaryon is never a
/// `nucleon()`.
///
/// **`generateSCMmuonAbsorption` is refused by name.** It needs G4GDecay3's three-body phase
/// space and it is reached only from muon capture at rest, which is P12's package and whose
/// caller does not exist in this port. The refusal is returned where the call would be, so a
/// mu- on a dibaryon is reported rather than silently producing nothing.
///
/// `have_nucleus` is `G4ElementaryParticleCollider::setNucleusState` having been called: the
/// collider is a member of G4NucleiModel's caller and the cascade sets (A, Z) on it before every
/// collision, but the bare two-body entry point below has no nucleus to set. The ONE place it
/// matters is `generateSCMpionNAbsorption`, which needs the residual mass
/// `getNucleiMass(A-1, Z-(2-ntype))`; with no nucleus that arm is refused by name instead of
/// approximated, which is what P10's oracle grid measures.
template <typename Rng>
__host__ __device__ inline void ep_collide_in_nucleus(int type1, const LV& mom1, int type2,
                                                      const LV& mom2, int nucleus_a,
                                                      int nucleus_z, bool have_nucleus,
                                                      const CascadeParams& par,
                                                      ColliderOutput& out,
                                                      BertiniWorkspace& ws, Rng& rng) {
  out.n = 0;
  out.refusal = ColliderRefusal::kNone;
  out.channel_refusal = ChannelRefusal::kNone;
  out.sample_fell_off = false;

  if (inucl_is_neutrino(type1) || inucl_is_neutrino(type2)) {
    out.refusal = ColliderRefusal::kNeutrinoProjectile;
    return;
  }
  const int is = type1 * type2;
  const bool qd1 = inucl_is_quasideuteron(type1);
  const bool qd2 = inucl_is_quasideuteron(type2);
  if (!channel_table(is).valid() && !qd1 && !qd2) {
    out.refusal = ColliderRefusal::kNoChannelTable;
    return;
  }

  // The FIRST bullet/target decision, the one with the quasi-deuteron arm.
  LorentzConvertor to_scm;
  if (inucl_is_nucleon(type2) || qd2) {
    to_scm.bullet = mom1;
    to_scm.target = mom2;
  } else {
    to_scm.bullet = mom2;
    to_scm.target = mom1;
  }
  lc_to_the_center_of_mass(to_scm);
  const double etot_scm = to_scm.ecm_tot;

  if (inucl_is_nucleon(type1) || inucl_is_nucleon(type2)) {
    const double ekin = lc_kin_energy_in_trs(to_scm);
    if (ep_pion_nucleon_absorption(is, ekin, par, rng)) {
      // Geant4 calls generateSCMpionNAbsorption here, which needs the RESIDUAL NUCLEUS mass,
      // `G4InuclNuclei::getNucleiMass(A-1, Z-(2-ntype))`, held on the collider by
      // `setNucleusState`. Without a nucleus the arm is REFUSED BY NAME rather than
      // approximated. Unreachable with the dumped piNAbsorption of 0 - the deviate above is
      // still drawn, because Geant4 draws it.
      if (!have_nucleus) {
        out.refusal = ColliderRefusal::kPionNAbsorptionNucleus;
        return;
      }
      // G4ElementaryParticleCollider::generateSCMpionNAbsorption's own argument:
      // `ntype` is whichever of the pair is the nucleon, and the residual is the nucleus minus
      // that nucleon - one proton fewer for a proton (ntype 1 -> 2-1 = 1), none for a neutron.
      const int ntype = inucl_is_nucleon(type2) ? type2 : type1;
      const double recoil_mass =
          inucl_nuclei_mass(nucleus_a - 1, nucleus_z - (2 - ntype), 0.0);
      // NOT a return: Geant4 falls through to the common `backToTheLab` loop at the end of
      // collide(), exactly as the ordinary final state does. The routine's own
      // `mom1.boost(-piN4.boostVector())` leaves the nucleon in the pi-N frame, which is the
      // frame `convertToSCM` boosts out of, so the two boosts compose.
      ep_generate_scm_pion_n_absorption(type1, mom1, type2, mom2, nucleus_a, nucleus_z,
                                        recoil_mass, out, rng);
    } else {
      ep_generate_scm_final_state(ekin, etot_scm, type1, mom1, type2, mom2, par, out, ws, rng);
    }
  }

  if (qd1 || qd2) {
    // Geant4's two `if`s are not `else if`, so a nucleon-and-dibaryon pair would enter both.
    // It cannot happen - a dibaryon is never `nucleon()` - but if it ever did, the absorption
    // arm's answer is the one that survives, so any refusal the block above left is dropped
    // here rather than reported against a final state that exists.
    out.refusal = ColliderRefusal::kNone;
    if (!nm_use_quasideuteron(type1, type2) && !nm_use_quasideuteron(type2, type1)) {
      out.refusal = ColliderRefusal::kIllegalDibaryonPartner;
      return;
    }
    if (inucl_is_muon(type1) || inucl_is_muon(type2)) {
      out.refusal = ColliderRefusal::kMuonAbsorption;
      return;
    }
    ep_generate_scm_pion_absorption(etot_scm, type1, type2, out, rng);
  }

  if (out.n == 0) { return; }   // "failed to collide": bullet passes through

  // Every four-vector here has been through an INUCL particle twice by the time the caller sees
  // it, and neither pass is the identity - see `inucl_store_momentum`. `particles[i].fill(...)`
  // stores the SCM momentum, `backToTheLab` reads it back, and `ipart->setMomentum(mom)` stores
  // the lab momentum. Both stores are reproduced, in Geant4's order, because a photon's
  // four-vector is measurably different on the other side of one. docs/RISK.md V125.
  for (int i = 0; i < out.n; ++i) {
    out.momenta[i] =
        lc_back_to_the_lab(to_scm, inucl_store_momentum(out.momenta[i], out.kinds[i]));
  }
  double ekin[kMaxFinalStateSize];
  for (int i = 0; i < out.n; ++i) {
    out.momenta[i] = inucl_store_momentum(out.momenta[i], out.kinds[i], &ekin[i]);
  }

  // std::sort(particles.begin(), particles.end(), G4ParticleLargerEkin()) - descending kinetic
  // energy, and the comparator reads `getKineticEnergy()`, the STORED one, not `e - m`. An
  // insertion sort here: the list is at most nine long and a comparison sort's branch
  // divergence costs more than the moves.
  for (int i = 1; i < out.n; ++i) {
    const LV m = out.momenta[i];
    const int k = out.kinds[i];
    const double e = ekin[i];
    int j = i - 1;
    while (j >= 0 && ekin[j] < e) {
      out.momenta[j + 1] = out.momenta[j];
      out.kinds[j + 1] = out.kinds[j];
      ekin[j + 1] = ekin[j];
      --j;
    }
    out.momenta[j + 1] = m;
    out.kinds[j + 1] = k;
    ekin[j + 1] = e;
  }
}

/// The bare two-body entry point: `collide` on a collider whose `setNucleusState` was never
/// called. Identical to the above in every arm except `generateSCMpionNAbsorption`, which is
/// refused by name because it needs a residual-nucleus mass that does not exist here.
template <typename Rng>
__host__ __device__ inline void ep_collide(int type1, const LV& mom1, int type2, const LV& mom2,
                                           const CascadeParams& par, ColliderOutput& out,
                                           BertiniWorkspace& ws, Rng& rng) {
  ep_collide_in_nucleus(type1, mom1, type2, mom2, 0, 0, false, par, out, ws, rng);
}

}  // namespace g4gpu::physics::hadronic::bert

#endif  // G4GPU_BERTINI_EP_COLLIDER_CUH
