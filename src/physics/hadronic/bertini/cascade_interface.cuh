// G4InuclCollider and G4CascadeInterface: the two retry loops above the cascade, and the entry
// point the transport calls.
//
// Transcribed from Geant4 11.1.1, cascade/cascade/src/:
//   G4InuclCollider::collide / deexcite / useCascadeDeexcitation / usePreCompoundDeexcitation /
//     photonuclearOkay
//   G4PreCompoundDeexcitation::deExcite
//   G4CascadeInterface::ApplyYourself / IsApplicable / createBullet / createTarget /
//     copyOutputToHadronicResult / makeDynamicParticle / checkFinalResult /
//     retryInelasticProton / retryInelasticNucleus / throwNonConservationFailure /
//     coulombBarrierViolation / NoInteraction
//
// ---------------------------------------------------------------------------------------------
// **There are THREE retry loops stacked on top of each other, and they count differently.**
//
//   G4CascadeInterface::ApplyYourself   up to 20, `numberOfTries`, condition is a PREDICATE
//   G4InuclCollider::collide            up to 100, plain counter, exits on `acceptable()`
//   G4IntraNucleiCascader::collide      up to 100, plain counter, exits on `finishCascade()`
//
// so one call can in principle run two hundred thousand cascades. The middle one ends by
// TRIVIALISING - bullet and target returned unchanged - and the outer one ends by returning
// `NoInteraction`, which is a different thing: the first is an elastic-looking final state, the
// second tells the process that the model did nothing at all and leaves the track alone.
//
// **The outer loop's condition is not "did it work".** `retryInelasticNucleus()` is
//
//     numberOfTries < maximumTries &&
//     ( (npart != 0 && npart+nfrag < 3 && firstOut == bullet->getDefinition())
//       || !balance->okay() )
//
// - it retries a SUCCESSFUL interaction whose final state looks elastic: two or fewer products
// with the leading one the same species as the bullet. So Bertini's inelastic model refuses to
// return an elastic scatter, and does it by regenerating the whole event up to twenty times.
// `retryInelasticProton` is the hydrogen version and tests a two-body final state against the
// bullet in EITHER slot.
//
// **`throwNonConservationFailure` ends the job.** After twenty attempts, if the balance is still
// bad, `G4CascadeInterface` throws a `G4HadronicException` and Geant4 stops. A kernel cannot
// throw and must not silently continue, so `ApplyResult::would_throw` carries it out by name and
// the caller decides; `ref/dump/dump_bertini.cc` catches it and counts it in the `thrown` column
// of `bertini_apply.csv`, which is what this port's number is compared against.
//
// ---------------------------------------------------------------------------------------------
// **Which de-excitation an instance uses is not `G4CascadeParameters::usePreCompound()`.**
// That flag is false, so the constructor calls `useCascadeDeexcitation()` - and then
// `G4HadronInelasticQBBC::ConstructProcess` calls `usePreCompoundDeexcitation()` on the p/n
// instance and on the pi+/pi- instance, overriding it for four of QBBC's eleven Bertini species.
// Kaons and hyperons, built by `G4HadronicBuilder::BuildFTFP_BERT`, keep the cascade's own
// evaporators. Both arms are here and `deexcite_choice` selects; `qbbc_bertini_range(pdg)` in
// cascade_params.cuh is what a caller asks. docs/RISK.md V118.
//
// **The PreCompound arm's `explosion()` branch is dead in QBBC.** `G4PreCompoundDeexcitation`
// takes it only when `theExcitationHandler` is non-null, and that member is filled only when the
// constructor fails to find a "PRECO" model in `G4HadronicInteractionRegistry` - which QBBC
// always registers. So the branch that would hand a small hot fragment straight to
// `G4ExcitationHandler::BreakItUp` cannot run, and every fragment goes to
// `G4PreCompoundModel::DeExcite`. Reproduced by not having the branch, with this note in place
// of it.
//
// It is an ORDER dependence and not a configuration flag, which is why it is worth the paragraph:
// `G4HadronInelasticQBBC::ConstructProcess` creates `thePreCompound` before it creates any
// `G4CascadeInterface`, so the registry lookup succeeds. A physics list that built Bertini first
// would give each Bertini instance its own `G4ExcitationHandler`, the branch would be live, and
// every residual with A <= 20 or Z == 0 would go to Fermi break-up instead of pre-equilibrium -
// a different model for exactly the light residuals this package's grid is full of. The oracle
// runs the same order as QBBC (`ref/dump/dump_bertini.cc` constructs its interfaces long after
// the run manager has initialised the list), so the port and the oracle agree because they make
// the same choice, not because the choice does not matter.
#ifndef G4GPU_BERTINI_CASCADE_INTERFACE_CUH
#define G4GPU_BERTINI_CASCADE_INTERFACE_CUH

#include <cmath>

#include "physics/hadronic/bertini/deexcite.cuh"
#include "physics/hadronic/bertini/intra_cascader.cuh"
#include "physics/hadronic/precompound/precompound_model.cuh"
#include "physics/hadronic/process.cuh"

namespace g4gpu::physics::hadronic::bert {

/// What the interface could not do.
enum class InterfaceRefusal : int {
  kNone = 0,
  kNotApplicable,          ///< IsApplicable: no channel table for this projectile
  kLightTargetCollider,    ///< a photon on A < 3 goes to G4LightTargetCollider - P13's
  kHyperNucleus,           ///< a target with L != 0
  kCascader,               ///< the intra-nuclear cascade refused; see CascaderRefusal
  kDeexcite,               ///< the de-excitation refused; see DeexciteRefusal
  kOverflow,               ///< an output capacity
  kSecondaryOverflow,      ///< HadFinalState's capacity
  kPrecompoundRefused      ///< P6 reported a refusal of its own
};

/// What one `apply_yourself` did beyond the final state.
struct ApplyResult {
  int n_tries = 0;                  ///< G4CascadeInterface::numberOfTries
  int n_collider_tries = 0;         ///< the last G4InuclCollider::collide attempt count
  bool no_interaction = false;      ///< twenty attempts failed: the track is left alone
  bool would_throw = false;         ///< throwNonConservationFailure would have ended the job
  bool trivialised = false;         ///< the collider gave up and returned bullet + target
  InterfaceRefusal refusal = InterfaceRefusal::kNone;
  CascaderRefusal cascader_refusal = CascaderRefusal::kNone;
  /// When `cascader_refusal` is `kFate`, WHICH fate refusal. The cascader forwards
  /// G4NucleiModel's own outcome and the two enumerations are separate, so a campaign that
  /// recorded only the cascader's kFate could not say what the model had refused.
  FateRefusal fate_refusal = FateRefusal::kNone;
  DeexciteRefusal deexcite_refusal = DeexciteRefusal::kNone;
  CascadeBalance balance;           ///< what checkFinalResult measured, in Bertini's GeV
};

/// G4CascadeInterface::IsApplicable(const G4ParticleDefinition*).
///
/// "Nuclei are okay" without further test - `GetAtomicMass() > 1` returns true immediately - and
/// everything else needs a channel table on the proton. So an alpha is applicable and a muon is
/// not, and neither statement mentions energy.
__host__ __device__ inline bool interface_is_applicable(int pdg) {
  if (pdg > 1000000000) { return true; }          // a nucleus
  const int type = inucl_type_from_pdg(pdg);
  return type != 0 && channel_table(type * kProton).valid();
}

/// G4InuclCollider::deexcite - the dispatch, plus its own ten-attempt retry.
///
/// The retry is `do { ... } while (!validateOutput(fragment, DEXoutput) && ++itry < itry_max)`
/// and `validateOutput` returns TRUE immediately when `G4CascadeParameters::checkConservation()`
/// is false - which it is - because the balance checker is never allocated. So with the dumped
/// parameters this loop runs EXACTLY ONCE, and the `check_conservation` flag is what decides
/// that; a run with `G4CASCADE_CHECK_ECONS` set would take the other path, which is why the
/// parameter is read rather than folded away.
template <typename Rng>
__host__ __device__ inline void collider_deexcite(
    const deex::Fragment& fragment, const ExitonConfiguration& excitons,
    DeexciteChoice choice, const CascadeParams& par, CollisionOutput& global_out,
    CollisionOutput& dex_out, CollisionOutput& tmp, BertiniWorkspace& ws,
    const data::LevelTable& lt, const deex::FermiPool& pool, const preco::PrecoWorkspace& pws,
    Rng& rng, ApplyResult& res) {
  if (fragment.a <= 1) { return; }       // "Nothing real to be de-excited"

  const int itry_max = 10;
  int itry = 0;
  bool ok = false;
  do {
    co_reset(dex_out);
    if (choice == DeexciteChoice::kCascade) {
      DeexciteRefusal dr = DeexciteRefusal::kNone;
      cascade_deexcite(fragment, excitons, dex_out, tmp, ws, rng, dr);
      if (deexcite_refusal_is_port_limit(dr)) {
        res.refusal = InterfaceRefusal::kDeexcite;
        res.deexcite_refusal = dr;
        return;
      }
    } else {
      // G4PreCompoundDeexcitation::deExcite -> G4PreCompoundModel::DeExcite, which is P6's.
      // The exciton configuration crosses the package boundary here: Bertini counts protons and
      // neutrons separately and PreCompound counts particles and charged particles, which is
      // the same information under two names.
      preco::Excitons ex;
      ex.particles = excitons.proton_quasi_particles + excitons.neutron_quasi_particles;
      ex.charged = excitons.proton_quasi_particles;
      ex.holes = excitons.proton_holes + excitons.neutron_holes;
      const preco::PrecoStatus st = preco::deexcite(fragment, ex, lt, pool, pws, rng);
      for (int i = 0; i < st.n_products; ++i) {
        const deex::DeexProduct& p = pws.products[i];
        // `addOutgoingParticles(G4ReactionProductVector*)` splits on whether the definition has
        // an INUCL type code: nucleons, pions, kaons and gammas become outgoing PARTICLES and
        // everything with A > 1 becomes an outgoing NUCLEUS. The momenta arrive in Geant4's MeV.
        const LV mom(p.momentum.v * 0.001, p.momentum.e * 0.001);
        const int type = inucl_type_from_pdg(p.pdg);
        if (p.a > 1 || (p.a == 1 && type == 0)) {
          if (!co_add_nucleus(global_out, nuclei_fill(mom, p.a, p.z, 0.0))) {
            res.refusal = InterfaceRefusal::kOverflow;
            return;
          }
        } else if (type != 0) {
          if (!co_add_particle(dex_out, type, inucl_store_momentum(mom, type))) {
            res.refusal = InterfaceRefusal::kOverflow;
            return;
          }
        } else {
          if (!co_add_nucleus(dex_out, nuclei_fill(mom, p.a, p.z, 0.0))) {
            res.refusal = InterfaceRefusal::kOverflow;
            return;
          }
        }
      }
    }
    // validateOutput: true when the balance checker was never allocated.
    ok = !par.check_conservation;
    if (!ok) {
      CascadeBalance b;
      b.relative_limit = 1.0e-6;
      b.absolute_limit = 1.0e-6;
      BalanceInitial in;
      in.has_bullet = false;
      in.target = LV(fragment.momentum.v * 0.001, fragment.momentum.e * 0.001);
      in.target_a = fragment.a;
      in.target_z = fragment.z;
      balance_collide(b, in, dex_out);
      ok = b.okay();
    }
  } while (!ok && ++itry < itry_max);

  co_add(global_out, dex_out);
}

/// G4InuclCollider::collide - a hundred attempts at a whole event, in the target's rest frame.
///
/// The bullet is REALIGNED before the loop: `toTheTargetRestFrame` is filled once, and the
/// cascade is run on a bullet whose momentum is `(0, 0, getTRSMomentum())` - purely along the
/// local z, with the frame's own rotation restored afterwards by `boostToLabFrame`. That is why
/// the cascade never sees an off-axis projectile and why `G4LorentzConvertor::rotate` matters at
/// the end rather than the beginning (docs/RISK.md V124).
template <typename Rng>
__host__ __device__ inline void inucl_collider_collide(
    const CascadeSetup& lab, DeexciteChoice choice, const CascadeParams& par,
    const NucleiModelParams& nmp, NucleiModel& model, CollisionOutput& global_out,
    CollisionOutput& out, CollisionOutput& dex_out, CollisionOutput& tmp, ColliderOutput& epo,
    BertiniWorkspace& ws, const data::LevelTable& lt, const deex::FermiPool& pool,
    const preco::PrecoWorkspace& pws, Rng& rng, ApplyResult& res) {
  const int itry_max = 100;

  LorentzConvertor to_trs;
  to_trs.bullet = lab.bullet;
  to_trs.target = lab.target;
  lc_to_the_target_rest_frame(to_trs);

  // `inelasticInteractionPossible` computes a Coulomb barrier and then IGNORES it:
  // `G4bool possible = true; // Force inelastic; should be (ekin >= VCOL)`. Transcribed as the
  // unconditional true it is, with the comment, because a port that implemented the barrier
  // would refuse events Geant4 runs.
  CascadeSetup zframe = lab;
  zframe.bullet = LV(Vec3d{0.0, 0.0, lc_trs_momentum(to_trs)}, 0.0);
  // `G4InuclElementaryParticle(bmom, btype)` - the energy component of `bmom` is left at ZERO
  // and the constructor's store rebuilds it from the mass shell. Not an approximation: the
  // four-vector Geant4 hands the cascader genuinely has e = 0 going in.
  zframe.bullet = inucl_store_momentum(zframe.bullet, lab.bullet_type);

  int itry = 0;
  while (itry < itry_max) {
    ++itry;
    co_reset(global_out);
    co_reset(out);

    ExitonConfiguration excitons;
    RecoilState recoil;
    CascaderResult cres;
    cascader_collide(model, nmp, par, zframe, out, excitons, recoil, epo, ws, rng, cres);
    if (cres.refusal != CascaderRefusal::kNone) {
      res.refusal = InterfaceRefusal::kCascader;
      res.cascader_refusal = cres.refusal;
      res.fate_refusal = cres.fate_refusal;
      return;
    }
    res.n_collider_tries = cres.n_tries;
    res.trivialised = cres.trivialised;

    if (out.has_recoil_fragment) {
      collider_deexcite(out.recoil_fragment, out.recoil_excitons, choice, par, out, dex_out,
                        tmp, ws, lt, pool, pws, rng, res);
      if (res.refusal != InterfaceRefusal::kNone) { return; }
      out.has_recoil_fragment = false;      // removeRecoilFragment()
    }

    // `photonuclearOkay` is behind `std::getenv("G4CASCADE_CHECK_PHOTONUCLEAR")`, which is not
    // set on this install, so the check never runs. It is the one place in the cascade tree
    // gated by a raw getenv rather than by G4CascadeParameters.

    bool undefined = false;
    co_boost_to_lab(out, to_trs, undefined);
    co_add(global_out, out);
    co_set_on_shell(global_out, lab.bullet, lab.target);
    if (global_out.on_shell) { return; }
  }

  co_trivialise(global_out, lab.bullet_type, lab.bullet, lab.bullet_a, lab.bullet_z,
                lab.target_a, lab.target_z, lab.target);
  res.trivialised = true;
}

/// G4CascadeInterface::retryInelasticNucleus - retry a final state that looks ELASTIC.
///
/// `firstOut == bullet->getDefinition()` compares particle DEFINITIONS, so it is a species test
/// and not a type-code one; for the species Bertini handles the two agree. The `npart != 0`
/// guard means an EMPTY final state is NOT retried on this path - only the balance test can
/// force that - which is the opposite of the hydrogen version below.
__host__ __device__ inline bool retry_inelastic_nucleus(int n_tries, int max_tries,
                                                        const CollisionOutput& out,
                                                        int bullet_type,
                                                        const CascadeBalance& balance) {
  const int npart = out.n_particles;
  const int nfrag = out.n_nuclei;
  const int first_out = (npart == 0) ? 0 : out.particles[0].type;
  return (n_tries < max_tries) &&
         (((npart != 0) && (npart + nfrag < 3 && first_out == bullet_type)) || !balance.okay());
}

/// G4CascadeInterface::retryInelasticProton - the hydrogen version, and it retries an EMPTY
/// final state as well as an elastic-looking two-body one. No balance test at all.
__host__ __device__ inline bool retry_inelastic_proton(int n_tries, int max_tries,
                                                       const CollisionOutput& out,
                                                       int bullet_type) {
  const int n = out.n_particles;
  return (n_tries < max_tries) &&
         (n == 0 || (n == 2 && (out.particles[0].type == bullet_type ||
                                out.particles[1].type == bullet_type)));
}

/// G4CascadeInterface::coulombBarrierViolation - built and never used in this build.
///
/// It is called only under `G4CASCADE_COULOMB_DEV`, a compiler flag this install does not set,
/// so `retryInelasticNucleus` takes the other arm. Transcribed so that a run that defines the
/// flag has it, and named so that the fact it is unused is visible.
__host__ __device__ inline bool coulomb_barrier_violation(const CollisionOutput& out) {
  const double coulomb_barrier = 8.7e-3;      // 8.7 MeV in Bertini's GeV
  bool violated = false;
  for (int i = 0; i < out.n_particles; ++i) {
    if (out.particles[i].type == kProton) {
      violated = violated || (out.particles[i].ekin < coulomb_barrier);
    }
  }
  return violated;
}

/// G4CascadeInterface::ApplyYourself, and the entry point the transport calls.
///
/// Units: the projectile arrives in the port's MeV, everything inside is Bertini's GeV, and the
/// secondaries go back out in MeV - the same two conversions `createBullet` and
/// `makeDynamicParticle` do.
///
/// **A K0 or anti-K0 on the output list is mixed to K0S or K0L with one deviate**, drawn INSIDE
/// `makeDynamicParticle` and therefore after the whole event is generated. That draw is part of
/// the random stream and its position in it is what a draw-count comparison sees.
template <typename real_t, int kCap, typename Rng>
__host__ __device__ inline ApplyResult apply_yourself(
    const HadProjectile<real_t>& projectile, const HadNucleus& target,
    HadFinalState<real_t, kCap>& fs, DeexciteChoice choice, const CascadeParams& par,
    const InterfaceLimits& lim, NucleiModel& model, CollisionOutput& global_out,
    CollisionOutput& out, CollisionOutput& dex_out, CollisionOutput& tmp, ColliderOutput& epo,
    BertiniWorkspace& ws, const data::LevelTable& lt, const deex::FermiPool& pool,
    const preco::PrecoWorkspace& pws, int secondary_model_id, Rng& rng) {
  ApplyResult res;
  fs.clear();

  if (target.l != 0) {
    res.refusal = InterfaceRefusal::kHyperNucleus;
    return res;
  }
  if (!interface_is_applicable(projectile.pdg)) {
    res.refusal = InterfaceRefusal::kNotApplicable;
    res.no_interaction = true;
    return res;
  }
  // "If target A < 3 skip all cascade machinery and do scattering on nucleons" - a photon on
  // hydrogen or deuterium goes to G4LightTargetCollider, which is P13's gamma-nuclear package
  // and is not in this port. Refused by name at the point it would have been needed.
  if (projectile.pdg == 22 && target.a < 3) {
    res.refusal = InterfaceRefusal::kLightTargetCollider;
    return res;
  }

  const NucleiModelParams nmp = nuclei_model_params(par);

  // createBullet: along +z with |p| and E from the track, in GeV, then stored.
  CascadeSetup lab;
  lab.bullet_type = inucl_type_from_pdg(projectile.pdg);
  const double plab = double(projectile.momentum()) * 0.001;
  const double elab = double(projectile.total_energy()) * 0.001;
  lab.bullet = inucl_store_momentum(LV(Vec3d{0.0, 0.0, plab}, elab), lab.bullet_type);
  lab.bullet_a = 0;
  lab.bullet_z = 0;

  // createTarget: A > 1 is a nucleus at rest, A == 1 is a bare nucleon.
  lab.target_a = target.a;
  lab.target_z = target.z;
  if (target.a > 1) {
    lab.target = LV(Vec3d{0.0, 0.0, 0.0}, inucl_nuclei_mass(target.a, target.z));
  } else {
    const int t = (target.z == 1) ? kProton : kNeutron;
    lab.target = LV(Vec3d{0.0, 0.0, 0.0}, inucl_particle_mass(t));
  }

  const bool is_hydrogen = (target.a == 1);
  CascadeBalance balance;
  balance.relative_limit = lim.balance_relative;
  balance.absolute_limit = lim.balance_absolute_GeV;
  const BalanceInitial bin = cascader_initial(lab);

  res.n_tries = 0;
  do {
    co_reset(global_out);
    if (target.a > 1) {
      inucl_collider_collide(lab, choice, par, nmp, model, global_out, out, dex_out, tmp, epo,
                             ws, lt, pool, pws, rng, res);
    } else {
      // A bare nucleon target is `useEPCollider`'s case: G4InuclCollider hands it straight to
      // G4ElementaryParticleCollider with no nucleus, no cascade and no de-excitation.
      ep_collide(lab.bullet_type, lab.bullet, (target.z == 1) ? kProton : kNeutron, lab.target,
                 par, epo, ws, rng);
      co_reset(global_out);
      for (int i = 0; i < epo.n; ++i) {
        if (!co_add_particle(global_out, epo.kinds[i], epo.momenta[i])) {
          res.refusal = InterfaceRefusal::kOverflow;
          return res;
        }
      }
    }
    if (res.refusal != InterfaceRefusal::kNone) { return res; }

    balance_collide(balance, bin, global_out);
    ++res.n_tries;
  } while (is_hydrogen
               ? retry_inelastic_proton(res.n_tries, lim.maximum_tries, global_out,
                                        lab.bullet_type)
               : retry_inelastic_nucleus(res.n_tries, lim.maximum_tries, global_out,
                                         lab.bullet_type, balance));

  res.balance = balance;

  if (res.n_tries >= lim.maximum_tries) {
    // "Null event if unsuccessful" - NoInteraction, which leaves the track alive with its
    // energy unchanged. NOT the same as a trivialised collider output.
    res.no_interaction = true;
    fs.status = HadFinalStateStatus::kIsAlive;
    fs.energy_change = (projectile.kin_energy > real_t(0)) ? projectile.kin_energy : real_t(0);
    return res;
  }

  if (!balance.okay()) {
    // throwNonConservationFailure() - a G4HadronicException that ENDS THE JOB. A kernel cannot
    // throw; the fact is carried out by name and counted, and the final state is the same
    // NoInteraction that follows the throw in the source.
    res.would_throw = true;
    res.no_interaction = true;
    fs.status = HadFinalStateStatus::kIsAlive;
    fs.energy_change = (projectile.kin_energy > real_t(0)) ? projectile.kin_energy : real_t(0);
    return res;
  }

  // copyOutputToHadronicResult.
  fs.status = HadFinalStateStatus::kStopAndKill;
  fs.energy_change = real_t(0);
  for (int i = 0; i < global_out.n_particles; ++i) {
    const int type = global_out.particles[i].type;
    if (inucl_is_quasideuteron(type)) {
      // "ERROR: G4CascadeInterface incompatible particle type" - Geant4 returns a null pointer
      // and `AddSecondary(0)` puts a null in the list. A dibaryon cannot leave the cascade, so
      // this is unreachable; it is skipped here rather than propagated as a null.
      continue;
    }
    HadSecondary<real_t> s;
    int pdg = inucl_type_row(type).pdg;
    if (type == kKaonZero || type == kKaonZeroBar) {
      // The K0/K0bar mixing, one deviate, AFTER the event: 310 (K0S) or 130 (K0L).
      pdg = (rng.uniform() > 0.5) ? 130 : 310;
    }
    s.pdg = pdg;
    s.z = static_cast<int>(inucl_type_row(type).charge);
    s.a = inucl_names_baryon(type);
    s.mass = real_t(inucl_particle_mass(type) * 1000.0);
    s.kin_energy = real_t(global_out.particles[i].ekin * 1000.0);
    s.direction = Vec3<real_t>{real_t(global_out.particles[i].momentum.v.x),
                               real_t(global_out.particles[i].momentum.v.y),
                               real_t(global_out.particles[i].momentum.v.z)};
    const real_t m = sqrt(s.direction.x * s.direction.x + s.direction.y * s.direction.y +
                          s.direction.z * s.direction.z);
    if (m > real_t(0)) {
      s.direction.x /= m;
      s.direction.y /= m;
      s.direction.z /= m;
    }
    s.creator_model_id = secondary_model_id;
    if (!fs.add_secondary(s)) { res.refusal = InterfaceRefusal::kSecondaryOverflow; }
  }
  for (int i = 0; i < global_out.n_nuclei; ++i) {
    HadSecondary<real_t> s;
    s.z = global_out.nuclei[i].z;
    s.a = global_out.nuclei[i].a;
    s.pdg = pdg_nuclear_code(s.z, s.a);
    s.mass = real_t(global_out.nuclei[i].mass * 1000.0);
    s.kin_energy = real_t(global_out.nuclei[i].ekin * 1000.0);
    s.direction = Vec3<real_t>{real_t(global_out.nuclei[i].momentum.v.x),
                               real_t(global_out.nuclei[i].momentum.v.y),
                               real_t(global_out.nuclei[i].momentum.v.z)};
    const real_t m = sqrt(s.direction.x * s.direction.x + s.direction.y * s.direction.y +
                          s.direction.z * s.direction.z);
    if (m > real_t(0)) {
      s.direction.x /= m;
      s.direction.y /= m;
      s.direction.z /= m;
    }
    s.creator_model_id = secondary_model_id;
    if (!fs.add_secondary(s)) { res.refusal = InterfaceRefusal::kSecondaryOverflow; }
  }

  // checkFinalResult() re-runs the balance and prints; with verbose 0 it changes nothing, and
  // the numbers it would have printed are in `res.balance` already.
  return res;
}

}  // namespace g4gpu::physics::hadronic::bert

#endif  // G4GPU_BERTINI_CASCADE_INTERFACE_CUH
