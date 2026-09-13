// G4IntraNucleiCascader: the loop that runs the cascade until the nucleus is empty or the
// particles have left, and the retry that throws the whole event away when what is left is not a
// nucleus.
//
// Transcribed from Geant4 11.1.1, cascade/cascade/src/G4IntraNucleiCascader.cc:
//   collide, initialize, newCascade, setupCascade, generateCascade, finishCascade, finalize,
//   processTrappedParticle, decayTrappedParticle, particleCanInteract
// and cascade/cascade/src/G4CascadeColliderBase.cc (useEPCollider, inelasticInteractionPossible,
// validateOutput).
//
// ---------------------------------------------------------------------------------------------
// **The cascade is a STACK, and it is worked from the back.** `generateCascade` reads
// `cascad_particles.back()`, gives it one step, pops it, and pushes whatever came out. So the
// most recently created particle is always the next one stepped - depth first, not breadth
// first - and the order products are pushed in (descending beta, from `generateParticleFate`)
// decides which of them goes deepest. That is not a detail of bookkeeping: the nucleus is being
// eaten as the cascade runs, so a particle stepped early meets a fuller nucleus than one stepped
// late, and reversing the order changes the event.
//
// **The loop stops when the nucleus is empty, not when the particles are gone.** `while
// (!cascad_particles.empty() && !model->empty())` - and `empty()` is
// `neutronNumberCurrent < 1 && protonNumberCurrent < 1`. Whatever is still in flight when the
// last nucleon is consumed is moved to the output by `finishCascade` without being stepped
// again.
//
// **Three things end a particle**, and only the first is "it left":
//   * it reached the surface (`stillInside` false) and either has enough energy to climb the
//     Coulomb barrier or tunnelled through it - `output.addOutgoingParticle`;
//   * it is inside, has reflected fewer than fifty times AND is `worthToPropagate` - it goes
//     back on the stack;
//   * anything else is TRAPPED: a nucleon becomes an exciton quasi-particle, a hyperon is
//     decayed in flight and its daughters go back on the stack, and anything else - a pion - is
//     released to the output with Geant4's own FIXME beside it ("this is a meson, so need to
//     absorb it").
//
// **The Coulomb barrier is a tunnelling probability, not a cut.**
//   CBP = exp(-0.0181 * 0.5 * Z * (1/KE - 1/barrier) * sqrt(mass*(barrier-KE)))
// with `barrier = 0.00126*Z/(1 + cbrt(A))` in GeV, and `KE > 0.0001` guarding the 1/KE. A proton
// below the barrier escapes with probability CBP and is trapped otherwise. The commented-out
// line above it is the older, simpler form; the one that runs has the `sqrt(mass*(barrier-KE))`
// WKB factor.
//
// ---------------------------------------------------------------------------------------------
// **`finishCascade` is where an event is accepted or thrown away.** In order: leftover cascade
// particles to the output; coalescence into light ions; recompute the recoil; reject unless the
// recoil is a physical fragment or the event fragmented completely; add a bare nucleon to the
// particle list if A = 1; zero a sub-MeV excitation for a quasi-elastic scatter; make the recoil
// fragment; sort; `setOnShell`; accept if `acceptable()`.
//
// **`minimum_recoil_A` is dead code, and it takes three lines of the source to see it.** It is
// meant to be a feedback loop - raise the bar after a failure so the next attempt gives up
// earlier - and it cannot fire. It starts at 0. The raise is
// `if (afin <= minimum_recoil_A && minimum_recoil_A < tnuclei->getA()) ++minimum_recoil_A;`,
// reached only at the END of `finishCascade`; but `afin <= 0` is impossible there, because
// `!goodFragment() && !wholeEvent()` has already returned false for `afin <= 0`, and `afin == 0`
// has already returned TRUE. So `afin >= 1 > 0` at the raise and the counter never moves. Its
// only reader, `generateCascade`'s `if (aresid <= minimum_recoil_A) return;`, therefore means
// "stop when the residual has no baryons left", which is reachable, and nothing else. Both are
// transcribed as written - a port that "fixed" the guard would abandon cascades Geant4 runs to
// the end - and the perturbation that deletes the raise entirely passes every comparison,
// because deleting dead code changes nothing.
//
// **`itry_max` is 100 and `finalize` does not report failure; it TRIVIALISES.** After a hundred
// attempts the output is replaced by the bullet and the target, unchanged - an elastic
// scatter with no scattering - and the caller has no way to tell that apart from a real
// pass-through except by the counter this port returns.
#ifndef G4GPU_BERTINI_INTRA_CASCADER_CUH
#define G4GPU_BERTINI_INTRA_CASCADER_CUH

#include <cmath>
#include "physics/hadronic/bertini/cascade_model.cuh"
#include "physics/hadronic/bertini/collision_output.cuh"
#include "physics/hadronic/bertini/workspace.cuh"

namespace g4gpu::physics::hadronic::bert {

/// G4IntraNucleiCascader's four constants.
constexpr int kCascaderItryMax = 100;
constexpr int kReflectionCut = 50;
constexpr double kCascaderSmallEkin = 0.001;   ///< 0.001*MeV, i.e. 1 keV expressed in MeV
constexpr double kQuasielastCut = 1.0;         ///< 1*MeV

/// What the cascader could not do.
enum class CascaderRefusal : int {
  kNone = 0,
  kFate,                  ///< generateParticleFate refused; its FateRefusal says which
  kOverflow,              ///< a workspace or output capacity
  kTrappedHyperonDecay,   ///< decayTrappedParticle needs a G4DecayTable - see the note below
  kBulletNotUsable,       ///< neither an elementary particle nor a nucleus
  kTargetNotNucleus,      ///< the target must be a nucleus
  kNucleusRefused         ///< generateModel refused this (A, Z)
};

/// What one `collide` did, beyond the output itself.
struct CascaderResult {
  int n_tries = 0;               ///< the `itry` finalize() compares against itry_max
  bool trivialised = false;      ///< a hundred attempts failed; bullet and target returned
  CascaderRefusal refusal = CascaderRefusal::kNone;
  FateRefusal fate_refusal = FateRefusal::kNone;
  int n_iterations = 0;          ///< generateCascade's `iloop`, summed over attempts
  /// How many particles were still in flight when the LAST attempt finished, i.e. how many
  /// `finishCascade` had to move to the output itself. Non-zero only when the cascade stopped
  /// because the NUCLEUS emptied rather than because the particles left. Measured over the 576
  /// oracle events, He-4 included: up to four, and every one of them on an event that then
  /// trivialised, whose output is replaced wholesale - so no ACCEPTED event has leftovers and
  /// the perturbation that drops the transfer passes every comparison. Reported rather than
  /// assumed, which is what made the difference between those two statements visible.
  int n_leftover = 0;
};

/// `finalize`'s exhaustion test, on its own so that it can be asserted on its own.
///
/// `itry` is Geant4's counter after `do { newCascade(++itry); ... } while (!finishCascade() &&
/// itry<itry_max)`, so it is `itry_max` both when the hundredth attempt FAILED and when the
/// hundredth attempt SUCCEEDED - the loop cannot run again either way. `finalize` rejects both
/// with one `>=`, so a cascade that finally worked on the last allowed attempt is thrown away
/// and replaced by the bullet and the target. Exactly the shape of the collider's tenth-attempt
/// discard, docs/RISK.md V121.
///
/// **No oracle case can see this.** The two events in `bertini_cascader.csv` that trivialise
/// both failed all hundred attempts, so `>=` and `&& !finished` agree on every row and the
/// perturbation passed all 575,992 comparisons. `tests/test_bertini_cascade.cu` asserts this
/// function at 99, 100 and 101 instead.
__host__ __device__ inline bool cascader_exhausted(int itry) {
  return itry >= kCascaderItryMax;
}

/// G4IntraNucleiCascader::particleCanInteract - "if we have a lookup table for particle type on
/// proton, it interacts". Nothing about energy: a 1 keV kaon "can interact" and a 5 GeV deuteron
/// cannot, because the tables are indexed by type alone. The argument is a bare type code and the
/// lookup key is a PRODUCT of two, which works only because the proton's code is 1.
///
/// **Always true on this entry point.** Everything the cascade can put on its stack has a table:
/// the collider's products come out of those same tables, and `decayTrappedParticle` - the one
/// other source - tests for a table itself and releases the daughters that have none. The branch
/// exists for `Propagate`'s pre-loaded secondaries, which are P11's and are not in this port.
/// Forcing it true passes every comparison, so it is asserted directly in
/// tests/test_bertini_cascade.cu instead.
__host__ __device__ inline bool cascader_particle_can_interact(int type) {
  return channel_table(type * kProton).valid();
}

/// The bullet and target a cascade runs on, in Bertini's units.
struct CascadeSetup {
  int bullet_type = 0;      ///< 0 when the bullet is a nucleus
  LV bullet;
  int bullet_a = 0, bullet_z = 0;
  int target_a = 0, target_z = 0;
  LV target;                ///< the target nucleus at rest: (0, 0, 0, getNucleiMass(A, Z))
};

/// `BalanceInitial` for a setup - the same four numbers the recoil maker needs.
__host__ __device__ inline BalanceInitial cascader_initial(const CascadeSetup& s) {
  BalanceInitial in;
  in.bullet = s.bullet;
  in.bullet_type = s.bullet_type;
  in.bullet_a = s.bullet_a;
  in.bullet_z = s.bullet_z;
  in.target = s.target;
  in.target_a = s.target_a;
  in.target_z = s.target_z;
  in.has_bullet = true;
  return in;
}

/// G4IntraNucleiCascader::processTrappedParticle.
///
/// A nucleon becomes an exciton quasi-particle and disappears from the event - its energy is
/// now the residual's excitation, by the recoil maker's subtraction. A hyperon is decayed in
/// flight, which needs `G4DecayTable::SelectADecayChannel` and `DecayIt` - P4's machinery, in a
/// different package, reached through a `G4ParticleDefinition` this module does not carry - so
/// it is REFUSED BY NAME rather than approximated. It is reachable: a lambda or sigma produced
/// deep in a heavy nucleus below the barrier gets here. Anything else, which in practice means a
/// pion, is released to the output unchanged, with Geant4's own FIXME beside it.
__host__ __device__ inline bool cascader_process_trapped(const CascadeParticle& trapped,
                                                         CollisionOutput& out,
                                                         ExitonConfiguration& excitons,
                                                         CascaderRefusal& refusal) {
  const int xtype = trapped.type;
  if (inucl_is_nucleon(xtype)) {
    excitons.increment_qp(xtype);
    return true;
  }
  if (inucl_names_hyperon(xtype)) {
    refusal = CascaderRefusal::kTrappedHyperonDecay;
    return false;
  }
  return co_add_particle(out, trapped.type, trapped.momentum);
}

/// G4IntraNucleiCascader::generateCascade - the depth-first loop over the stack.
template <typename Rng>
__host__ __device__ inline void cascader_generate(NucleiModel& m, const NucleiModelParams& p,
                                                  const CascadeParams& par,
                                                  const CascadeSetup& s, CollisionOutput& out,
                                                  ExitonConfiguration& excitons,
                                                  RecoilState& recoil, double coulomb_barrier,
                                                  double minimum_recoil_a, ColliderOutput& epo,
                                                  BertiniWorkspace& ws, Rng& rng,
                                                  CascaderResult& res) {
  const BalanceInitial in = cascader_initial(s);
  CascadeBalance bal;
  bal.relative_limit = 1.0e-6;
  bal.absolute_limit = 1.0e-6;

  while (ws.n_cascade > 0 && !(m.neutron_number_current < 1 && m.proton_number_current < 1)) {
    ++res.n_iterations;

    CascadeParticle current = ws.cascade[ws.n_cascade - 1];

    // A particle with no channel table cannot interact and goes straight out. This is the only
    // exit that does not test the Coulomb barrier.
    if (!cascader_particle_can_interact(current.type)) {
      if (!co_add_particle(out, current.type, current.momentum)) {
        res.refusal = CascaderRefusal::kOverflow;
        return;
      }
      --ws.n_cascade;
      continue;
    }

    FateRefusal fr = FateRefusal::kNone;
    nm_generate_particle_fate(m, p, par, current, epo, ws, rng, fr);
    if (fr != FateRefusal::kNone) {
      res.refusal = CascaderRefusal::kFate;
      res.fate_refusal = fr;
      return;
    }
    --ws.n_cascade;      // "Discarding last cparticle from list"

    if (ws.n_new_cascade == 1) {
      const CascadeParticle& cp = ws.new_cascade[0];
      if (cp.current_zone < m.number_of_zones) {          // stillInside
        if (cp.reflection_counter < kReflectionCut &&
            nm_worth_to_propagate(m, cp.reflected, cp.type, cp.current_zone,
                                  inucl_stored_kinetic_energy(cp.momentum, cp.type))) {
          if (!ws_push_cascade(ws, cp)) { res.refusal = CascaderRefusal::kOverflow; return; }
        } else {
          CascaderRefusal tr = CascaderRefusal::kNone;
          if (!cascader_process_trapped(cp, out, excitons, tr)) {
            res.refusal = (tr != CascaderRefusal::kNone) ? tr : CascaderRefusal::kOverflow;
            return;
          }
        }
      } else {                                            // at the surface: escape or tunnel
        const double ke = inucl_stored_kinetic_energy(cp.momentum, cp.type);
        const double mass = inucl_particle_mass(cp.type);
        const double q = inucl_type_row(cp.type).charge;
        if (ke < q * coulomb_barrier) {
          double cbp = 0.0;
          if (ke > 0.0001) {
            cbp = std::exp(-0.0181 * 0.5 * double(s.target_z) *
                           (1.0 / ke - 1.0 / coulomb_barrier) *
                           std::sqrt(mass * (coulomb_barrier - ke)));
          }
          if (rng.uniform() < cbp) {
            // "Tunnelling through barrier leaves KE unchanged"
            if (!co_add_particle(out, cp.type, cp.momentum)) {
              res.refusal = CascaderRefusal::kOverflow;
              return;
            }
          } else {
            CascaderRefusal tr = CascaderRefusal::kNone;
            if (!cascader_process_trapped(cp, out, excitons, tr)) {
              res.refusal = (tr != CascaderRefusal::kNone) ? tr : CascaderRefusal::kOverflow;
              return;
            }
          }
        } else {
          if (!co_add_particle(out, cp.type, cp.momentum)) {
            res.refusal = CascaderRefusal::kOverflow;
            return;
          }
        }
      }
    } else {                                              // an interaction
      for (int i = 0; i < ws.n_new_cascade; ++i) {
        if (!ws_push_cascade(ws, ws.new_cascade[i])) {
          res.refusal = CascaderRefusal::kOverflow;
          return;
        }
      }
      // The nucleons the collision consumed become HOLES. `holes.second` is only incremented
      // when it is non-zero, which is the dibaryon case; `holes.first` is passed unconditionally
      // and `incrementHoles` ignores a zero, so a step that consumed nothing adds nothing.
      excitons.increment_holes(m.current_nucl1);
      if (m.current_nucl2 > 0) { excitons.increment_holes(m.current_nucl2); }
    }

    // The residual after this step, from everything in the output PLUS everything still in
    // flight. This is the only place the cascade can stop early.
    balance_collide(bal, in, out, ws.cascade, ws.n_cascade);
    recoil_fill(recoil, bal);
    if (double(recoil.a) <= minimum_recoil_a) { return; }
  }
}

/// G4IntraNucleiCascader::finishCascade - accept the event, or say it has to be regenerated.
///
/// Returns true when the cascade is finished and acceptable. `minimum_recoil_a` is raised in
/// place on the failure path, which is what makes the next attempt give up earlier.
template <typename Rng>
__host__ __device__ inline bool cascader_finish(NucleiModel& m, const CascadeParams& par,
                                                const CascadeSetup& s, CollisionOutput& out,
                                                ExitonConfiguration& excitons,
                                                RecoilState& recoil, double& minimum_recoil_a,
                                                BertiniWorkspace& ws, Rng& rng,
                                                CascaderResult& res) {
  (void)rng;
  const BalanceInitial in = cascader_initial(s);
  CascadeBalance bal;
  bal.relative_limit = 1.0e-6;
  bal.absolute_limit = 1.0e-6;

  // Everything still in flight leaves as it is.
  res.n_leftover = ws.n_cascade;
  for (int i = 0; i < ws.n_cascade; ++i) {
    if (!co_add_particle(out, ws.cascade[i].type, ws.cascade[i].momentum)) {
      res.refusal = CascaderRefusal::kOverflow;
      return false;
    }
  }
  ws.n_cascade = 0;

  if (par.do_coalescence) {
    coal_find_clusters(out, coalescence_cuts(par));
    if (out.overflow != CascadeOverflow::kNone) {
      res.refusal = CascaderRefusal::kOverflow;
      return false;
    }
    // "Update recoil fragment after generating light ions" - the coalesced ions carry less mass
    // than the nucleons that made them, so the residual moves.
    balance_collide(bal, in, out);
    recoil_fill(recoil, bal);
  }

  const int afin = recoil.a;
  const int zfin = recoil.z;
  if (!recoil_good_fragment(recoil) && !recoil_whole_event(recoil)) { return false; }
  if (afin == 0) { return true; }        // the whole target fragmented

  if (afin == 1) {
    const int last_type = (zfin == 1) ? kProton : kNeutron;
    const double mass = inucl_particle_mass(last_type);
    const double mres = recoil.momentum.mag();
    // Not enough energy to be that nucleon: the event is unphysical and is regenerated.
    //
    // **`small_ekin` is `0.001*MeV` and this comparison is in GeV**, so the threshold that
    // actually runs is one MeV, not one keV - a thousand times looser than the constant reads.
    // The SAME constant is handed to the recoil maker as `excTolerance`, where it IS compared
    // against an excitation in MeV and does mean a keV, and again as `excTolerance/GeV` against
    // a momentum in GeV, where it means an eV. One number, three meanings, and only the middle
    // one matches its declaration.
    //
    // Found by adding He-4 to the oracle grid: a residual of A = 1 needs a target light enough
    // to lose three of its four nucleons, and on C, Al, Fe and Pb it never happens. With the keV
    // reading, the port accepted a pi+ event on He-4 that Geant4 rejected a hundred times and
    // then trivialised.
    if (mres - mass < -kCascaderSmallEkin) { return false; }
    // Too much energy: Geant4 adds it anyway "and lets SetOnShell fudge things", with a FIXME.
    //
    // **The nucleon is CONSTRUCTED here, so it goes through the INUCL store**, and that is what
    // discards the excess. `G4InuclElementaryParticle(presid, last_type, INCascader)` forces the
    // residual four-vector onto the nucleon's mass shell; the 12 MeV that made `mres - mass`
    // positive is simply gone, the event no longer balances, `wholeEvent` is false and - with no
    // fragment either, because the residual is now (0,0) - `setOnShell` is never reached and the
    // cascade is REGENERATED. Every other `addOutgoingParticle` in the cascader hands over a
    // particle that was stored when it was made; this one is new, and passing the raw
    // four-vector instead made the port accept a pi+ on He-4 that Geant4 rejects a hundred times
    // and then trivialises.
    if (!co_add_particle(out, last_type,
                         inucl_store_momentum(recoil.momentum, last_type))) {
      res.refusal = CascaderRefusal::kOverflow;
      return false;
    }
    balance_collide(bal, in, out);
    recoil_fill(recoil, bal);
  }

  // A quasi-elastic scatter - one particle out - with a sub-MeV excitation is declared exactly
  // elastic, so that the recoil fragment is not handed to de-excitation for a keV.
  if (out.n_particles == 1) {
    if (std::fabs(recoil.excitation_MeV) < kQuasielastCut) { recoil.excitation_MeV = 0.0; }
  }

  if (recoil_good_nucleus(recoil)) {
    out.recoil_fragment = recoil_make_fragment(recoil, excitons);
    out.has_recoil_fragment = true;
  }

  // "Put final-state particles in leading order for return" - descending STORED kinetic energy.
  for (int i = 1; i < out.n_particles; ++i) {
    const OutgoingParticle key = out.particles[i];
    int j = i - 1;
    while (j >= 0 && out.particles[j].ekin < key.ekin) {
      out.particles[j + 1] = out.particles[j];
      --j;
    }
    out.particles[j + 1] = key;
  }

  if (recoil_whole_event(recoil) || recoil_good_nucleus(recoil)) {
    co_set_on_shell(out, s.bullet, s.target);
    if (out.on_shell) { return true; }
  }

  // Raise the bar for the next attempt: abandon the cascade as soon as the residual drops this
  // low, so that a cascade that ate the whole nucleus is not tried again the same way.
  if (double(afin) <= minimum_recoil_a && minimum_recoil_a < double(m.a)) {
    minimum_recoil_a += 1.0;
  }
  return false;
}

/// G4IntraNucleiCascader::collide - initialize, then up to a hundred attempts.
///
/// `coulombBarrier = 0.00126*Z/(1 + cbrt(A))` uses the INTEGER cube root, `G4Pow::Z13`, because
/// `getA()` returns a `G4int` - the same overload selection that decides the nuclear radius in
/// `generateModel` (see nuclei_model.cuh's note 1).
template <typename Rng>
__host__ __device__ inline void cascader_collide(NucleiModel& m, const NucleiModelParams& p,
                                                 const CascadeParams& par,
                                                 const CascadeSetup& s, CollisionOutput& out,
                                                 ExitonConfiguration& excitons,
                                                 RecoilState& recoil, ColliderOutput& epo,
                                                 BertiniWorkspace& ws, Rng& rng,
                                                 CascaderResult& res) {
  res = CascaderResult();
  if (s.target_a <= 1) { res.refusal = CascaderRefusal::kTargetNotNucleus; return; }
  if (s.bullet_type == 0 && s.bullet_a <= 0) {
    res.refusal = CascaderRefusal::kBulletNotUsable;
    return;
  }

  if (nm_generate_model(m, s.target_a, s.target_z, p) != NucleiModelRefusal::kNone) {
    res.refusal = CascaderRefusal::kNucleusRefused;
    return;
  }
  const double coulomb_barrier =
      0.00126 * double(s.target_z) / (1.0 + inucl_cbrt_int(s.target_a));

  // `inputEkin = bullet->getKineticEnergy()` - the STORED kinetic energy, which for a nucleus
  // bullet means the four-vector has to go through the ion's own mass shell first.
  recoil.tolerance = kCascaderSmallEkin;
  if (s.bullet_type != 0) {
    recoil.input_ekin = inucl_stored_kinetic_energy(s.bullet, s.bullet_type);
  } else {
    inucl_store_momentum_mass(s.bullet, inucl_nuclei_mass(s.bullet_a, s.bullet_z),
                              &recoil.input_ekin);
  }
  double minimum_recoil_a = 0.0;

  int itry = 0;
  bool finished = false;
  do {
    ++itry;
    // newCascade: the model's census and hit list, the output, both cascade buffers and the
    // exciton configuration. NOT `current_nucl1`/`current_nucl2`, which Geant4 leaves alone.
    nm_reset(m);
    ws.n_collision_points = 0;
    co_reset(out);
    ws.n_cascade = 0;
    ws.n_new_cascade = 0;
    excitons.clear();

    // setupCascade: one cascade particle for a hadron projectile, or the nucleon cluster of an
    // ion projectile plus whatever missed the target.
    if (s.bullet_type != 0) {
      FateRefusal fr = FateRefusal::kNone;
      const CascadeParticle cp =
          nm_initialize_cascad(m, p, s.bullet_type, s.bullet, rng, fr);
      if (fr != FateRefusal::kNone) {
        res.refusal = CascaderRefusal::kFate;
        res.fate_refusal = fr;
        return;
      }
      if (!ws_push_cascade(ws, cp)) { res.refusal = CascaderRefusal::kOverflow; return; }
    } else {
      int n_cascade = 0, n_released = 0;
      int rel_types[BertiniWorkspace::kMaxIonBulletA];
      LV rel_mom[BertiniWorkspace::kMaxIonBulletA];
      FateRefusal fr = FateRefusal::kNone;
      nm_initialize_cascad_nucleus(m, s.bullet_a, s.bullet_z, s.bullet,
                                   inucl_nuclei_mass(s.bullet_a, s.bullet_z),
                                   recoil.input_ekin, s.target_a, s.target_z, s.target, ws,
                                   &n_cascade, rel_types, rel_mom, &n_released,
                                   BertiniWorkspace::kMaxIonBulletA, rng, fr);
      if (fr != FateRefusal::kNone) {
        res.refusal = CascaderRefusal::kFate;
        res.fate_refusal = fr;
        return;
      }
      for (int i = 0; i < n_released; ++i) {
        if (!co_add_particle(out, rel_types[i], rel_mom[i])) {
          res.refusal = CascaderRefusal::kOverflow;
          return;
        }
      }
      if (n_cascade == 0) {
        // A compound nucleus: every nucleon of the bullet becomes a quasi-particle, and a random
        // number of holes is thrown on top. `G4int(2*(ab-zb)*u + 0.5)` is a TRUNCATION of a
        // uniform on [0, 2N] plus a half, i.e. a rounded uniform - not a Poisson and not a
        // binomial, and it can exceed the number of nucleons of that kind.
        for (int i = 0; i < s.bullet_a; ++i) { excitons.increment_qp(i < s.bullet_z ? 1 : 2); }
        const int ihn = int(2.0 * double(s.bullet_a - s.bullet_z) * rng.uniform() + 0.5);
        const int ihz = int(2.0 * double(s.bullet_z) * rng.uniform() + 0.5);
        for (int i = 0; i < ihn; ++i) { excitons.increment_holes(2); }
        for (int i = 0; i < ihz; ++i) { excitons.increment_holes(1); }
      }
    }

    cascader_generate(m, p, par, s, out, excitons, recoil, coulomb_barrier, minimum_recoil_a,
                      epo, ws, rng, res);
    if (res.refusal != CascaderRefusal::kNone) { return; }

    finished = cascader_finish(m, par, s, out, excitons, recoil, minimum_recoil_a, ws, rng, res);
    if (res.refusal != CascaderRefusal::kNone) { return; }
  } while (!finished && itry < kCascaderItryMax);

  res.n_tries = itry;
  // finalize(itry, ...): `if (itry >= itry_max) output.trivialise(bullet, target);`
  //
  // **Note what that does NOT test.** The loop is `while (!finishCascade() && itry<itry_max)`,
  // so a cascade that SUCCEEDS on the hundredth attempt exits with `itry == itry_max` - and
  // `finalize` throws it away and returns the bullet and the target instead. Ninety-nine usable
  // attempts, not a hundred, and it is the same shape as the collider's own tenth-attempt
  // discard (docs/RISK.md V121): an exhaustion test written with `>=` that also fires on the
  // success. Transcribed as written; `res.trivialised` says when it happened so that a
  // statistical campaign can count it rather than wonder where the elastic events came from.
  if (cascader_exhausted(itry)) {
    co_trivialise(out, s.bullet_type, s.bullet, s.bullet_a, s.bullet_z, s.target_a, s.target_z,
                  s.target);
    res.trivialised = true;
  }
}

}  // namespace g4gpu::physics::hadronic::bert

#endif  // G4GPU_BERTINI_INTRA_CASCADER_CUH
