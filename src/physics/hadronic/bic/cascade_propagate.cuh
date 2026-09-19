// G4BinaryCascade::Propagate and G4BinaryCascade::DeExcite - the cascade loop, and the assembly
// of every layer under it.
//
// Transcribed from G4BinaryCascade.cc (models/binary_cascade, 11.1.1), lines 386-680 and 929-999.
// Every piece this calls has its own file and its own oracle; what is here is the ORDER, which is
// the one part no per-piece comparison can check.
//
//     BuildTargetList                      the nucleus becomes theTargetList
//     thePropagator->Init                  the field maps, from that same nucleus
//     BuildLateParticleCollisions          the projectile becomes theSecondaryList
//     FindCollisions(&theSecondaryList)    every action, for every secondary
//     Entries() == 0            ->  return 0
//     while (Entries() && currentZ && --collisionLoopMaxCount>0):
//         Absorb(); Capture();
//         if (Entries()) { DoTimeStep(next time - now); ApplyCollision(next) or Remove(next); }
//     no target list, or no proton in it   ->  FillVoidNucleusProducts
//     Absorb(); Capture();
//     !haveProducts             ->  return an EMPTY vector
//     StepParticlesOut()
//     leftover secondaries  ->  theFinalState;  leftover collisions  ->  removed
//     ExcitationEnergy < 0      ->  CorrectFinalPandE, up to five rounds
//     still < 0                 ->  return empty
//     DeExcite()                    the fragment goes to the precompound model
//     G4DecayKineticTracks decay(&theFinalState)
//     ProductsAddFinalState, then ProductsAddPrecompound
//
// ## `return 0` AND `return an empty vector` MEAN DIFFERENT THINGS
//
// `ApplyYourself` has two nested loops around `Propagate` and they read its two failures
// differently. A NULL return - `theCollisionMgr->Entries() == 0`, no collision was found at all -
// means the sampled impact parameter missed, and the INNER loop resamples the position up to 200
// times. An EMPTY vector - collisions were found and every one was vetoed, or the excitation came
// out negative - means the geometry was fine and the physics refused, and the OUTER loop rebuilds
// the nucleus from scratch and tries again up to 100 times. A port that returned one thing for
// both would spend 200 tries resampling a position that was never the problem, and would rebuild
// a nucleus that was never wrong. `PropagateOutcome` is therefore part of the result and not a
// diagnostic.
//
// ## THE COLLISION LOOP'S GUARD IS A MILLION
//
//     G4int collisionLoopMaxCount=1000000;
//     while(theCollisionMgr->Entries() > 0 && currentZ && --collisionLoopMaxCount>0)
//
// A cascade makes tens of collisions, so this is not a safety net at a plausible number - it is a
// guard against a cycle, and it is carried at its own size because a smaller one would change
// which events complete.
//
// ## REFUSED, by name
//
//   * **G4BinaryCascade::FillVoidNucleusProducts**, the branch for a nucleus whose target list is
//     empty or which has no proton left in it. It is 180 lines of ad-hoc corrections - including
//     a `G4UniformRand()`-drawn kinetic energy handed to the leftover nucleons when the balance
//     comes out negative - and it is reached only once the cascade has destroyed the nucleus
//     outright. `CascadeRefusal::void_nucleus` is set at the point it would have been needed and
//     the product list is left as it stands, so that a caller sees an incomplete event rather
//     than a plausible one.
//   * **G4BinaryCascade::HighEnergyModelFSProducts**, and the late-particle arm with it. Both are
//     reachable only when a high-energy generator owns this cascade and hands it secondaries
//     whose state is `undefined` plus a `GetPrimaryProjectile()`;
//     `CascadeRefusal::high_energy_primary` is set when an input secondary has that state.
//   * **the `G4HadronicException` in `DeExcite`'s A <= 1 arm** ("G4BinaryCasacde:: Invalid
//     Fragment", the typo is Geant4's), thrown when a one-nucleon fragment has more than one
//     track left between the target and captured lists. A kernel cannot throw, so it is
//     `CascadeRefusal::invalid_nucleus`.
#ifndef G4GPU_BIC_CASCADE_PROPAGATE_CUH
#define G4GPU_BIC_CASCADE_PROPAGATE_CUH

#include <cfloat>
#include <cmath>

#include "physics/hadronic/bic/cascade_deexcite.cuh"

namespace g4gpu::bic {

/// The most tracks `Propagate` will hold in one of its scratch index arrays. These index the
/// SECONDARY and FINAL lists, not the target list, so they are far smaller than A; 256 is the
/// bound every other file in this package uses and the overflow is a refusal, never a silent
/// truncation.
inline constexpr int kCascadeMaxScratch = 256;

/// How `Propagate` ended. `ApplyYourself` branches on exactly this.
enum PropagateOutcome : int {
  kPropagateOk = 0,
  kPropagateNoCollision = 1,          ///< Geant4's `return 0`: resample the impact parameter
  kPropagateNoProducts = 2,           ///< `!haveProducts`: rebuild the nucleus and try again
  kPropagateNegativeExcitation = 3,   ///< five rounds of CorrectFinalPandE were not enough
  kPropagateVoidNucleus = 4,          ///< FillVoidNucleusProducts, refused; see the file header
  kPropagateRefused = 5               ///< see `CascadeRefusal`
};

/// Everything the cascade needs and does not own: the pool, the collision list, the channel
/// table, the cross-section buffers and the two product arrays. All caller-allocated, which is
/// the rule for every kernel-side structure in this port.
struct CascadeWorkspace {
  CascadeTrack* pool = nullptr;
  int pool_capacity = 0;
  imr::CollisionInitialState* collisions = nullptr;
  int collision_capacity = 0;
  const imr::ConcreteChannel* channels = nullptr;
  int n_channels = 0;
  CascadeBuffers* buffers = nullptr;
  /// theFinalState's products followed by the precompound's, which is the order Geant4 pushes
  /// them and therefore the order the secondaries come out in.
  CascadeProduct* products = nullptr;
  int product_capacity = 0;
  /// `DeExcite`'s return, before `ProductsAddPrecompound` boosts it.
  CascadeProduct* preco_products = nullptr;
  int preco_capacity = 0;
};

/// What one `Propagate` produced, beyond the product list itself.
struct PropagateResult {
  int outcome = kPropagateOk;
  int n_products = 0;
  int n_final_state = 0;       ///< how many of `n_products` came from ProductsAddFinalState
  double excitation_energy = 0.0;
  /// `FindFragments`' answer, which is what the precompound model was asked about. `fragment_a`
  /// is zero when there was no fragment and `DecayVoidNucleus` ran instead.
  int fragment_a = 0;
  int fragment_z = 0;
  int fragment_holes = 0;
  int fragment_particles = 0;
  int fragment_charged = 0;
  imr::LorentzVector fragment_momentum;
  int collisions_applied = 0;  ///< `collisionCount` under debug_BIC_Propagate_Collisions
  int correct_rounds = 0;      ///< how many times CorrectFinalPandE ran, 0..5
  int step_out_resets = 0;     ///< StepParticlesOut's `countreset`
  int loop_iterations = 0;     ///< 1000000 - collisionLoopMaxCount
  /// What `Capture`'s gate saw the LAST time it said yes, and how many times it did. The gate
  /// is a MEAN over every nucleon inside - `capturedEnergy/particlesBelowCut < 0.2*theCutOnP`
  /// - and a cascade that ends with everything captured looks exactly like a cascade that
  /// never started, so the number the decision turned on is worth keeping.
  int captures = 0;
  int capture_count = 0;
  double capture_energy = 0.0;
};

/// `G4BinaryCascade::GetSpherePoint(G4double r, const G4LorentzVector& mom4)`.
///
/// A point uniform in the disc of radius `r` orthogonal to `mom`, displaced `1.5 r` back along
/// it. The disc is sampled by rejection from the square - `x1, x2` each `2*(U()-0.5)`, redrawn
/// while `x1^2 + x2^2 > 1` - which consumes TWO uniforms per attempt and, for a well-behaved
/// engine, 1.27 attempts on average. Geant4's own loop comment is "or random is badly broken";
/// this carries a 10,000-attempt guard because a kernel cannot spin forever, and reports it.
///
/// `o1 = mom.orthogonal()` is CLHEP's `Hep3Vector::orthogonal()`, transcribed below: it is not
/// any orthogonal vector, it is a specific one chosen by which component is smallest in
/// magnitude, and which one it picks decides the ORIENTATION of the sampled disc. Two vectors
/// that differ by a rotation about `mom` give the same physics only because the disc is sampled
/// isotropically; the individual draw is not the same, so it is transcribed and not replaced.
__host__ __device__ inline Vec3<double> clhep_orthogonal(const Vec3<double>& v) {
  const double xx = v.x < 0.0 ? -v.x : v.x;
  const double yy = v.y < 0.0 ? -v.y : v.y;
  const double zz = v.z < 0.0 ? -v.z : v.z;
  if (xx < yy) {
    return (xx < zz) ? Vec3<double>{0.0, v.z, -v.y} : Vec3<double>{v.y, -v.x, 0.0};
  }
  return (yy < zz) ? Vec3<double>{-v.z, 0.0, v.x} : Vec3<double>{v.y, -v.x, 0.0};
}

template <typename Rng>
__host__ __device__ inline Vec3<double> get_sphere_point(double r, const Vec3<double>& mom,
                                                         Rng& rng, bool* gave_up = nullptr) {
  const Vec3<double> o1 = clhep_orthogonal(mom);
  const Vec3<double> o2 = g4gpu::cross(mom, o1);
  double x1 = 0.0;
  double x2 = 0.0;
  int guard = 10000;
  do {
    x1 = (rng.uniform() - 0.5) * 2.0;
    x2 = (rng.uniform() - 0.5) * 2.0;
  } while (x1 * x1 + x2 * x2 > 1.0 && --guard > 0);
  if (guard <= 0 && gave_up != nullptr) { *gave_up = true; }
  // The 1.5 is the source's, and its own comment beside it is not: "point is random in plane
  // (circle of radius r) orthogonal to mom, plus -1*r*mom->vect()->unit()" says ONE radius back
  // and the line says one and a half. With r = 1.1*(outerRadius + 3 fermi) that puts the track
  // about 1.65 outer radii upstream, which is why its `outside` state is never in doubt. The
  // code is what runs; docs/RISK.md V162.
  return r * (x1 * g4gpu::normalize(o1) + x2 * g4gpu::normalize(o2) -
              1.5 * g4gpu::normalize(mom));
}

/// `G4BinaryCascade::Capture`'s move, once `capture_decision` has said yes.
///
/// `UpdateTracksAndCollisions(&captured, NULL, NULL)` erases each captured track from
/// theSecondaryList and drops its collisions; the track has already been pushed to
/// theCapturedList and had `Hit()` called on it. In the pool that is one retag plus
/// `remove_tracks`, and `Hit()` is the `hit` flag - which matters, because
/// `ProductsAddFinalState` reads it back as `IsParticipant()`.
template <typename Prop>
__host__ __device__ inline bool capture_secondaries(BicCascadeState& st,
                                                    imr::CollisionList& colls,
                                                    const Prop& propagator,
                                                    CascadeRefusal& ref,
                                                    CaptureDecision* seen = nullptr) {
  const CaptureDecision d = capture_decision(st.lists.pool, st.lists.n_pool, st.cut_on_p,
                                             propagator);
  if (seen != nullptr) { *seen = d; }
  if (!d.capture) { return false; }
  int captured[kCascadeMaxScratch];
  int n_cap = 0;
  for (int i = 0; i < st.lists.n_pool; ++i) {
    CascadeTrack& t = st.lists.pool[i];
    if (t.list != kListSecondary || t.state != kInside) { continue; }
    if (t.pdg != imr::kPdgProton && t.pdg != imr::kPdgNeutron) { continue; }
    if (n_cap >= kCascadeMaxScratch) {
      ref.capacity = true;
      break;
    }
    t.list = kListCaptured;
    mark_hit(st, i);
    captured[n_cap++] = i;
  }
  colls.remove_tracks(captured, n_cap);
  return true;
}

/// `G4BinaryCascade::Propagate(secondaries, aNucleus)`.
///
/// `secondaries` is the list `ApplyYourself` built - for a standalone cascade exactly one track,
/// the projectile, with its state already set to `outside` - and it is MODIFIED in place, because
/// `BuildLateParticleCollisions` shifts every formation time so that the earliest is zero.
///
/// `propagator` must already have been built from THIS nucleus: `Propagate`'s
/// `thePropagator->Init(the3DNucleus)` is `make_rk_propagation`, which needs the caller's two
/// field tables and so cannot be done from in here. Geant4 calls `Init` twice on the same
/// nucleus - once in `ApplyYourself` and again here - so one call before this one is the same
/// state.
///
/// `de_excite` is the caller's exit into the precompound model: it is called once, with the
/// fragment, and must fill `ws.preco_products` and return how many it filled. It is a parameter
/// so that this file does not depend on P6, and so that the call can be `__noinline__` at the
/// caller, which is what keeps the precompound model's frame out of this one's.
template <typename Prop, typename Rng, typename DeExcite>
__host__ __device__ inline PropagateResult propagate(
    BicCascadeState& st, Nucleus3D& nucleus, Prop& propagator, const CascadeSpecies& sp,
    const CascadeWorkspace& ws, CascadeTrack* secondaries, int n_secondaries,
    const NuclearDensity& density, double coulomb_barrier, DeExcite&& de_excite, Rng& rng,
    CascadeRefusal& ref) {
  PropagateResult out;
  imr::CollisionList colls;
  colls.items = ws.collisions;
  colls.capacity = ws.collision_capacity;
  colls.n = 0;

  // `theOuterRadius = the3DNucleus->GetOuterRadius(); theCurrentTime=0;
  //  theProjectile4Momentum=G4LorentzVector(0,0,0,0); theMomentumTransfer=G4ThreeVector(0,0,0);`
  // then ClearAndDestroy on the captured, secondary and final lists and on the manager.
  st.lists.pool = ws.pool;
  st.nucleons = nucleus.nucleons;
  st.lists.capacity = ws.pool_capacity;
  st.lists.n_pool = 0;
  st.outer_radius = nucleus.outer_radius();
  st.current_time = 0.0;
  st.projectile_4mom = imr::LorentzVector(deex::Vec3d{0.0, 0.0, 0.0}, 0.0);
  st.momentum_transfer = deex::Vec3d{0.0, 0.0, 0.0};

  // `theCutOnP` from `the3DNucleus->GetMass()`, which is `Z m_p + (A-Z) m_n - BE` IN MeV and so
  // above 120 for every nucleus that exists - docs/RISK.md V72, and `bic_params.cuh` writes out
  // both readings. Set BEFORE BuildTargetList, as the source sets it.
  NucleusReport nrep;
  st.cut_on_p = cut_on_p(nucleus.mass(nrep));
  if (nrep.hyper_nucleus) {
    ref.invalid_nucleus = true;
    out.outcome = kPropagateRefused;
    return out;
  }

  build_target_list(nucleus, st, sp.proton_mass, sp.neutron_mass, ref);
  if (ref.any()) {
    out.outcome = kPropagateRefused;
    return out;
  }

  // `BuildLateParticleCollisions(secondaries)`. A secondary whose state is `undefined` is a
  // high-energy generator's product and takes the late-particle arm, refused by name.
  for (int i = 0; i < n_secondaries; ++i) {
    if (secondaries[i].state == kUndefined) {
      ref.high_energy_primary = true;
      ref.refused_pdg = secondaries[i].pdg;
      out.outcome = kPropagateRefused;
      return out;
    }
  }
  if (!build_late_particle_collisions(secondaries, n_secondaries, st, ref)) {
    out.outcome = kPropagateRefused;
    return out;
  }

  // The closure `UpdateTracksAndCollisions` and `StepParticlesOut` both need, so that
  // FindCollisions always runs where Geant4 runs it - `G4BCDecay::GetCollisions` draws a uniform,
  // so finding collisions one statement earlier is a different event.
  auto find = [&](const int* idx, int n) {
    find_collisions(st, colls, idx, n, ws.channels, ws.n_channels, *ws.buffers, sp, propagator,
                    rng, ref);
  };

  // `FindCollisions(&theSecondaryList)` - "if called stand alone find first collisions".
  {
    int initial[kCascadeMaxScratch];
    int n_initial = 0;
    for (int i = 0; i < st.lists.n_pool; ++i) {
      if (st.lists.pool[i].list != kListSecondary) { continue; }
      if (n_initial >= kCascadeMaxScratch) {
        ref.capacity = true;
        out.outcome = kPropagateRefused;
        return out;
      }
      initial[n_initial++] = i;
    }
    find(initial, n_initial);
  }

  if (colls.size() == 0) {
    // "late particles ALWAYS create Entries", so an empty manager here is no collision at all.
    out.outcome = kPropagateNoCollision;
    return out;
  }

  auto apply = [&](int ci) {
    const bool ok = apply_collision(st, colls, ci, ws.channels, ws.n_channels, *ws.buffers, sp,
                                    propagator, density, coulomb_barrier, rng, find, ref);
    if (ok) { ++out.collisions_applied; }
    return ok;
  };

  bool have_products = false;
  int loop_guard = 1000000;
  while (colls.size() > 0 && st.current_z && --loop_guard > 0) {
    AbsorbRefusal abs_ref;
    if (absorb(st.lists.pool, st.lists.n_pool, cut_on_p_absorb(), abs_ref)) {
      have_products = true;
    }
    {
      CaptureDecision seen;
      if (capture_secondaries(st, colls, propagator, ref, &seen)) {
        have_products = true;
        ++out.captures;
        out.capture_count = seen.particles_below_cut;
        out.capture_energy = seen.captured_energy;
      }
    }
    // "propagate to the next collision if any (collisions could have been deleted by previous
    // absorption or capture)".
    if (colls.size() > 0) {
      int next = colls.next_collision();
      if (next >= 0) {
        const double step = colls.items[next].collision_time - st.current_time;
        const TimeStepReport tsr = do_time_step(st, colls, propagator, step, sp.proton_mass,
                                                sp.neutron_mass, find, ref);
        if (!tsr.success) {
          // "Check if nextCollision is still valid, ie. particle did not leave nucleus"
          if (colls.next_collision() != next) { next = -1; }
        }
        if (next >= 0) {
          if (apply(next)) {
            have_products = true;
          } else {
            colls.remove(next);
          }
        }
      }
    }
  }
  out.loop_iterations = 1000000 - loop_guard;

  // "nucleus completely destroyed" - `! theTargetList.size() || ! nProtons`, where nProtons
  // counts G4Proton in theTargetList ONLY, and not in theCapturedList.
  {
    int n_target = 0;
    int n_protons = 0;
    for (int i = 0; i < st.lists.n_pool; ++i) {
      if (st.lists.pool[i].list != kListTarget) { continue; }
      ++n_target;
      if (st.lists.pool[i].pdg == imr::kPdgProton) { ++n_protons; }
    }
    if (n_target == 0 || n_protons == 0) {
      ref.void_nucleus = true;
      out.outcome = kPropagateVoidNucleus;
      return out;
    }
  }

  // "No more collisions: absorb, capture and propagate the secondaries out of the nucleus"
  {
    AbsorbRefusal abs_ref;
    if (absorb(st.lists.pool, st.lists.n_pool, cut_on_p_absorb(), abs_ref)) {
      have_products = true;
    }
    {
      CaptureDecision seen;
      if (capture_secondaries(st, colls, propagator, ref, &seen)) {
        have_products = true;
        ++out.captures;
        out.capture_count = seen.particles_below_cut;
        out.capture_energy = seen.captured_energy;
      }
    }
  }
  if (!have_products) {
    out.outcome = kPropagateNoProducts;
    return out;
  }

  const StepOutReport sor = step_particles_out(st, colls, propagator, sp, apply, find, ref);
  out.step_out_resets = sor.resets;

  // "add left secondaries to FinalSate" - a warning path in Geant4, and it fires.
  for (int i = 0; i < st.lists.n_pool; ++i) {
    if (st.lists.pool[i].list == kListSecondary) { push_final(st, i); }
  }
  // "Warning: remove left over collision(s)"
  while (colls.size() > 0) {
    const int next = colls.next_collision();
    if (next < 0) { break; }
    colls.remove(next);
  }

  double excitation = get_excitation_energy(st, sp.neutron_mass);
  if (excitation < 0.0) {
    // `do { CorrectFinalPandE(); ExcitationEnergy=GetExcitationEnergy(); }
    //  while (++ntry < maxtry && ExcitationEnergy < 0);` with maxtry = 5, so the correction runs
    // at least once and at most five times - and each round is capped at two per cent by the
    // 0.98 floor in `correct_final_p_and_e`.
    int ntry = 0;
    do {
      correct_final_p_and_e(st, sp.neutron_mass);
      excitation = get_excitation_energy(st, sp.neutron_mass);
      ++out.correct_rounds;
    } while (++ntry < 5 && excitation < 0.0);
  }
  out.excitation_energy = excitation;
  if (excitation < 0.0) {
    // `ClearAndDestroy(products); return products;` - empty, and "FixMe" in the source.
    out.outcome = kPropagateNegativeExcitation;
    return out;
  }

  // ------------------------------------------------------------------------------------------
  // `DeExcite()`
  // ------------------------------------------------------------------------------------------
  int n_preco = 0;
  const CascadeFragment frag = find_fragments(st);
  out.fragment_a = frag.a;
  out.fragment_z = frag.z;
  out.fragment_holes = frag.holes;
  out.fragment_particles = frag.particles;
  out.fragment_charged = frag.charged;
  out.fragment_momentum = frag.momentum;
  if (frag.a > 1) {
    n_preco = de_excite(frag, ws.preco_products, ws.preco_capacity);
  } else if (frag.a == 1) {
    // "fragment->GetA_asInt() <= 1, so a single proton, as a fragment must have Z>0". The product
    // is made AT REST - `SetTotalEnergy(GetPDGMass())` and `SetMomentum(G4ThreeVector(0))`, with
    // the comment "see boost for preCompoundProducts below" - and `ProductsAddPrecompound` gives
    // it its momentum. `precompoundLorentzboost` is the one `GetExcitationEnergy` stored a few
    // lines above, because this arm does not call `GetFinalNucleusMomentum` itself.
    int found = -1;
    int n_left = 0;
    for (int i = 0; i < st.lists.n_pool; ++i) {
      const int l = st.lists.pool[i].list;
      if (l != kListTarget && l != kListCaptured) { continue; }
      ++n_left;
      found = i;
    }
    if (n_left != 1) {
      ref.invalid_nucleus = true;
      out.outcome = kPropagateRefused;
      return out;
    }
    if (ws.preco_capacity < 1) {
      ref.capacity = true;
      out.outcome = kPropagateRefused;
      return out;
    }
    const CascadeTrack& t = st.lists.pool[found];
    ws.preco_products[0] = CascadeProduct{};
    ws.preco_products[0].pdg = t.pdg;
    ws.preco_products[0].pdg_mass = t.pdg_mass;
    ws.preco_products[0].nucleus_z = t.charge;
    ws.preco_products[0].nucleus_a = t.baryon;
    ws.preco_products[0].momentum = imr::LorentzVector(deex::Vec3d{0.0, 0.0, 0.0}, t.pdg_mass);
    ws.preco_products[0].creator_model_id = bic_model_id();
    ws.preco_products[0].parent_resonance_pdg = t.parent_resonance_pdg;
    ws.preco_products[0].parent_resonance_id = t.parent_resonance_id;
    n_preco = 1;
  } else {
    // "No fragment, can be neutrons only" - FindFragments returned null because Z < 1.
    n_preco = decay_void_nucleus(st, ws.preco_products, ws.preco_capacity, bic_model_id(), rng,
                                 ref);
  }

  // ------------------------------------------------------------------------------------------
  // `G4DecayKineticTracks decay(&theFinalState)` - the shared engine, on the final state, AFTER
  // DeExcite and BEFORE the products are made. A resonance that left the nucleus decays here.
  // ------------------------------------------------------------------------------------------
  {
    DecayTrack dl[kCascadeMaxScratch];
    int n = 0;
    // In theFinalState's own order, which is `final_seq` - see `push_final`.
    for (int seq = 0; seq < st.n_final_pushed; ++seq) {
      int i = -1;
      for (int k = 0; k < st.lists.n_pool; ++k) {
        if (st.lists.pool[k].list == kListFinal && st.lists.pool[k].final_seq == seq) {
          i = k;
          break;
        }
      }
      if (i < 0) { continue; }
      const CascadeTrack& t = st.lists.pool[i];
      if (n >= kCascadeMaxScratch) {
        ref.capacity = true;
        out.outcome = kPropagateRefused;
        return out;
      }
      dl[n] = DecayTrack{};
      dl[n].pdg = t.pdg;
      dl[n].momentum = t.momentum;
      dl[n].position = t.position;
      dl[n].formation_time = t.formation_time;
      dl[n].creator_model_id = t.creator_model_id;
      dl[n].parent_resonance_pdg = t.parent_resonance_pdg;
      dl[n].parent_resonance_id = t.parent_resonance_id;
      // The pool index, so that a survivor can be matched back to the track it was and keep the
      // `IsParticipant()` that `ProductsAddFinalState` asks it for. A daughter comes back -1.
      dl[n].caller_tag = i;
      ++n;
    }
    KineticDecayRefusal kref;
    int m = n;
    decay_kinetic_tracks(dl, m, kCascadeMaxScratch, rng, kref);
    if (kref.any()) {
      ref.unknown_species = ref.unknown_species || kref.unknown_species;
      ref.capacity = ref.capacity || kref.list_full;
      ref.refused_pdg = kref.refused_pdg;
    }
    // The engine handed back the surviving list in Geant4's order. Rebuild theFinalState from
    // it: every old member off the list first, then the survivors appended in that order, so
    // that `ProductsAddFinalState`'s walk sees what Geant4's walk would see.
    for (int i = 0; i < st.lists.n_pool; ++i) {
      if (st.lists.pool[i].list == kListFinal) { st.lists.pool[i].list = kListNone; }
    }
    st.n_final_pushed = 0;
    for (int k = 0; k < m; ++k) {
      CascadeTrack t;
      if (dl[k].caller_tag >= 0) {
        // A survivor is the same `G4KineticTrack*`, so everything the pool entry carried is still
        // true of it - including its nucleon and its `Hit()`. Only its place in the list changes.
        t = st.lists.pool[dl[k].caller_tag];
      } else {
        // A daughter is a `new G4KineticTrack` with a null `theNucleon`, so `IsParticipant()` is
        // false on it whatever its parent was.
        const int si = imr::decay_species_index(dl[k].pdg);
        if (si < 0) {
          ref.unknown_species = true;
          ref.refused_pdg = dl[k].pdg;
          out.outcome = kPropagateRefused;
          return out;
        }
        t.pdg = dl[k].pdg;
        t.pdg_mass = imr::decay_species_mass()[si];
        t.charge = imr::decay_species_charge()[si];
        t.baryon = imr::decay_species_baryon()[si];
        t.state = kGoneOut;
        t.nucleon_index = -1;
        t.hit = false;
      }
      t.momentum = dl[k].momentum;
      t.position = dl[k].position;
      t.formation_time = dl[k].formation_time;
      t.creator_model_id = dl[k].creator_model_id;
      t.parent_resonance_pdg = dl[k].parent_resonance_pdg;
      t.parent_resonance_id = dl[k].parent_resonance_id;
      const int added = st.lists.add(t, kListNone);
      if (added < 0) {
        ref.capacity = true;
        out.outcome = kPropagateRefused;
        return out;
      }
      push_final(st, added);
    }
  }

  int n_products = products_add_final_state(st, ws.products, 0, ws.product_capacity,
                                            bic_model_id(), ref);
  out.n_final_state = n_products;
  n_products = products_add_precompound(st, ws.preco_products, n_preco, ws.products, n_products,
                                        ws.product_capacity, ref);
  out.n_products = n_products;
  return out;
}

}  // namespace g4gpu::bic

#endif
