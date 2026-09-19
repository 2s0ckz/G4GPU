// G4BinaryCascade's endgame: StepParticlesOut and CorrectFinalPandE.
//
// Transcribed from G4BinaryCascade.cc (models/binary_cascade, 11.1.1). When the collision loop
// has nothing left to do, whatever is still inside the nucleus has to be got out, and then the
// outgoing set has to be made to fit inside the energy the reaction actually had.
//
// ## `StepParticlesOut` STEPS BY THE SOONEST EXIT, TIMES 1.2
//
// Each turn it asks every `inside` secondary for the time at which it would leave the nuclear
// sphere - `GetSphereIntersectionTimes`'s SECOND root - takes the smallest, multiplies it by 1.2,
// and steps by that unless a collision comes sooner. The 1.2 is what guarantees progress: a step
// to exactly the intersection time leaves the track on the surface, where
// `GetSphereIntersectionTimes` reports a MISS (`sqrtArg == 0`), and the loop would not advance.
//
// It gives up in two ways and they are different. After 100 steps with an EMPTY collision manager
// it re-runs `FindCollisions` over everything still inside and resets the step counter - that is
// `countreset`, and it is how a particle trapped in the potential gets another chance at a
// collision. After 100 of those resets it abandons the attempt, moves every remaining secondary
// into the final state as it stands, and breaks. Then, outside the loop, `DoTimeStep(DBL_MAX)`
// takes everything that is left arbitrarily far.
//
// ## `CorrectFinalPandE` NEVER CORRECTS BY MORE THAN TWO PER CENT
//
//     if (pFinals.vect().mag() > pInCM)
//     { G4double factor = std::max(0.98, pInCM/pFinals.vect().mag());
//       for each final: p3 = factor * (toCMS*p).vect(); ... }
//
// `pInCM` is the two-body momentum for (residual nucleus at its nominal mass) + (the outgoing set
// at its own invariant mass), and the outgoing momenta are scaled to it. But the scale factor is
// floored at 0.98, so when the outgoing set carries more than about 2% too much momentum the
// correction stops there and the imbalance survives - the comment in the source calls it a "small
// correction" and it is not a repair. `Propagate` calls it up to five times in a row while the
// excitation energy is negative, which is how a large imbalance is worked down; if five rounds of
// 2% are not enough the event is dropped with an empty product list.
//
// The energy each final gets is `sqrt(Get4Momentum().mag2() + p3.mag2())` with `mag2()` the
// SIGNED invariant of the track's lab four-momentum, so a track that was off shell keeps exactly
// the amount by which it was.
//
// ## REFUSED, by name
//
//   * nothing.
#ifndef G4GPU_BIC_CASCADE_FINISH_CUH
#define G4GPU_BIC_CASCADE_FINISH_CUH

#include <cfloat>
#include <cmath>

#include "physics/hadronic/bic/cascade_find.cuh"

namespace g4gpu::bic {

/// What one `StepParticlesOut` did, for a caller that wants to see why it stopped.
struct StepOutReport {
  int steps = 0;         ///< `counter`, reset every time FindCollisions is re-run
  int resets = 0;        ///< `countreset`
  int abandoned = 0;     ///< secondaries moved to the final state by the countreset>100 exit
  bool state_error = false;  ///< a secondary that was neither inside nor outside: Geant4 throws
};

/// `G4BinaryCascade::StepParticlesOut`.
///
/// `apply` and `find` are the two callables the loop needs - `apply_collision` and
/// `find_collisions` bound to this cascade's arguments - passed in rather than templated on the
/// whole argument list, because the loop's shape is the transcription and the plumbing is not.
template <typename Prop, typename Apply, typename Find>
__host__ __device__ inline StepOutReport step_particles_out(
    BicCascadeState& st, imr::CollisionList& colls, Prop& propagator, const CascadeSpecies& sp,
    Apply&& apply, Find&& find, CascadeRefusal& ref) {
  StepOutReport rep;
  while (st.lists.count(kListSecondary) > 0) {
    // "about 30*fermi/(0.1*c_light)", i.e. a big step.
    double min_time_step = 1.0e-12 * u::ns<double>();
    for (int i = 0; i < st.lists.n_pool; ++i) {
      const CascadeTrack& t = st.lists.pool[i];
      if (t.list != kListSecondary) { continue; }
      if (t.state == kInside) {
        const KineticTrack kt = as_kinetic_track(t);
        double t_dummy = 0.0;
        double t_step = 0.0;
        const bool intersect = propagator.sphere_intersection_times(kt, t_dummy, t_step);
        if (intersect && t_step < min_time_step && t_step > 0.0) { min_time_step = t_step; }
      } else if (t.state != kOutside) {
        // Geant4 throws a G4HadronicException here. A kernel cannot, so it is reported and the
        // track is left alone; the loop still terminates through `countreset`.
        rep.state_error = true;
      }
    }
    min_time_step *= 1.2;
    double time_to_collision = DBL_MAX;
    int next = -1;
    if (colls.size() > 0) {
      next = colls.next_collision();
      if (next >= 0) {
        time_to_collision = colls.items[next].collision_time - st.current_time;
      }
    }
    if (time_to_collision > min_time_step) {
      do_time_step(st, colls, propagator, min_time_step, sp.proton_mass, sp.neutron_mass, find,
                   ref);
      ++rep.steps;
    } else {
      const TimeStepReport tsr = do_time_step(st, colls, propagator, time_to_collision,
                                              sp.proton_mass, sp.neutron_mass, find, ref);
      if (!tsr.success) {
        // "Check if nextCollision is still valid, ie. particle did not leave nucleus"
        if (colls.next_collision() != next) { next = -1; }
      }
      if (next >= 0) {
        if (!apply(next)) { colls.remove(next); }
      }
    }
    if (rep.resets > 100) {
      // "add left secondaries to FinalSate" - as they stand, uncorrected.
      for (int i = 0; i < st.lists.n_pool; ++i) {
        if (st.lists.pool[i].list == kListSecondary) {
          push_final(st, i);
          ++rep.abandoned;
        }
      }
      break;
    }
    AbsorbRefusal abs_ref;
    int secondaries[256];
    int n_sec = 0;
    for (int i = 0; i < st.lists.n_pool && n_sec < 256; ++i) {
      if (st.lists.pool[i].list == kListSecondary) { secondaries[n_sec++] = i; }
    }
    absorb(st.lists.pool, st.lists.n_pool, cut_on_p_absorb(), abs_ref);
    const CaptureDecision cap = capture_decision(st.lists.pool, st.lists.n_pool, st.cut_on_p,
                                                 propagator);
    if (cap.capture) {
      int captured[256];
      int n_cap = 0;
      for (int i = 0; i < st.lists.n_pool && n_cap < 256; ++i) {
        CascadeTrack& t = st.lists.pool[i];
        if (t.list != kListSecondary || t.state != kInside) { continue; }
        if (t.pdg != imr::kPdgProton && t.pdg != imr::kPdgNeutron) { continue; }
        t.list = kListCaptured;
        mark_hit(st, i);
        captured[n_cap++] = i;
      }
      colls.remove_tracks(captured, n_cap);
    }
    if (rep.steps > 100 && colls.size() == 0) {
      // No collision and stepping for some time: look again, and count a reset.
      n_sec = 0;
      for (int i = 0; i < st.lists.n_pool && n_sec < 256; ++i) {
        if (st.lists.pool[i].list == kListSecondary) { secondaries[n_sec++] = i; }
      }
      find(secondaries, n_sec);
      rep.steps = 0;
      ++rep.resets;
    }
  }
  do_time_step(st, colls, propagator, DBL_MAX, sp.proton_mass, sp.neutron_mass, find, ref);
  return rep;
}

/// `G4BinaryCascade::CorrectFinalPandE`.
///
/// Returns the factor it applied, or 1.0 when it did nothing - which is the case when the final
/// state is empty, when the remnant's four-momentum came back as an explicit zero from
/// `GetFinal4Momentum`'s superluminal guard, when there is not enough invariant mass for the
/// two-body split, or when the outgoing set is already inside the limit.
__host__ __device__ inline double correct_final_p_and_e(BicCascadeState& st,
                                                        double neutron_mass) {
  if (st.lists.count(kListFinal) == 0) { return 1.0; }
  const imr::LorentzVector p_nucleus = get_final_4momentum(st);
  if (p_nucleus.e == 0.0) { return 1.0; }
  imr::LorentzVector p_finals(deex::Vec3d{0.0, 0.0, 0.0}, 0.0);
  for (int i = 0; i < st.lists.n_pool; ++i) {
    if (st.lists.pool[i].list == kListFinal) {
      p_finals = p_finals + st.lists.pool[i].momentum;
    }
  }
  const imr::LorentzVector p_cm = p_nucleus + p_finals;
  const imr::LorentzRotation to_cms = imr::LorentzRotation::from_boost(-1.0 * p_cm.boost_vector());
  const imr::LorentzVector p_finals_cm = to_cms * p_finals;
  const imr::LorentzRotation to_lab = to_cms.inverse();
  const double s0 = p_cm.e * p_cm.e - g4gpu::mag2(p_cm.v);
  const double m10 = get_ion_mass(st.current_z, st.current_a, neutron_mass);
  const double m20 = p_finals_cm.mag();
  if (s0 - (m10 + m20) * (m10 + m20) < 0.0) { return 1.0; }
  const double p_in_cm = std::sqrt((s0 - (m10 + m20) * (m10 + m20)) *
                                   (s0 - (m10 - m20) * (m10 - m20)) / (4.0 * s0));
  const double mag_finals = std::sqrt(g4gpu::mag2(p_finals_cm.v));
  if (!(mag_finals > p_in_cm)) { return 1.0; }
  // "small correction" - and the 0.98 floor is what makes it small whatever the imbalance is.
  const double ratio = p_in_cm / mag_finals;
  const double factor = (0.98 > ratio) ? 0.98 : ratio;
  for (int i = 0; i < st.lists.n_pool; ++i) {
    CascadeTrack& t = st.lists.pool[i];
    if (t.list != kListFinal) { continue; }
    const deex::Vec3d p3 = factor * (to_cms * t.momentum).v;
    // `Get4Momentum().mag2()` is the SIGNED invariant of the LAB four-momentum, taken before the
    // boost, so a track that was off shell keeps exactly the amount by which it was.
    const double m2 = t.momentum.e * t.momentum.e - g4gpu::mag2(t.momentum.v);
    const imr::LorentzVector p(p3, std::sqrt(m2 + g4gpu::mag2(p3)));
    t.momentum = to_lab * p;
  }
  return factor;
}

}  // namespace g4gpu::bic

#endif
