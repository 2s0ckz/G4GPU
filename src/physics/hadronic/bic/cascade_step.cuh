// G4BinaryCascade's time step: UpdateTracksAndCollisions, CorrectBarionsOnBoundary, DoTimeStep.
//
// Transcribed from G4BinaryCascade.cc (models/binary_cascade, 11.1.1). This is the half of the
// cascade loop that is not a collision: the propagator moves every secondary by one time step,
// and then the bookkeeping works out who crossed the nuclear surface in which direction and what
// that costs in energy.
//
// ## A BARYON CROSSING THE SURFACE IS PAID FOR OUT OF THE NUCLEUS'S MASS
//
// `CorrectBarionsOnBoundary` is called with the tracks that went IN and the tracks that went OUT.
// For each group it works out what the nucleus's mass was before and after the crossing, and
// shares the difference equally over the crossing tracks:
//
//     in:   correction = secondaryMass_in  + mass_initial - mass_final
//     out:  correction = mass_initial - mass_final - secondaryMass_out
//     if (secondaries > 1) correction /= secondaries
//     if (e + correction > actualMass) UpdateTrackingMomentum(e + correction)
//
// where `mass_initial` and `mass_final` are `GetIonMass(currentZ, currentA)` before and after
// `currentA` and `currentZ` are moved by the crossing baryons. So the energy a nucleon gains
// entering a nucleus is the binding energy the nucleus gains by having it, divided among however
// many crossed together in that step.
//
// **A RESONANCE ENTERING COUNTS AS A PROTON AND LEAVING COUNTS AS A NEUTRON.** The two branches
// differ by one word:
//
//     in:   if (neutron || proton) secondaryMass_in  += definition->GetPDGMass();
//           else                   secondaryMass_in  += G4Proton::Proton()->GetPDGMass();
//     out:  if (neutron || proton) secondaryMass_out += definition->GetPDGMass();
//           else                   secondaryMass_out += G4Neutron::Neutron()->GetPDGMass();
//
// A Delta++ crossing in contributes 938.272 MeV and the same Delta++ crossing out contributes
// 939.565 - a 1.293 MeV asymmetry in a term that is then divided by the number of crossings and
// added to each crossing track's energy. docs/RISK.md V156. Reproduced.
//
// And when the correction is not enough to get the track over its own mass, the two branches
// undo the barrier in OPPOSITE directions, both under the comment "Undo correction for Colomb
// Barrier": the one that cannot get in has the barrier ADDED and is sent to `miss_nucleus`, the
// one that cannot get out has it SUBTRACTED and is `captured`. Only a proton or a neutron is
// captured that way; a pion or a resonance that cannot get out is left alone entirely, with its
// energy uncorrected and its state still `gone_out`.
//
// ## `DoTimeStep`'s RETURN VALUE IS ABOUT THE NEXT COLLISION AND NOTHING ELSE
//
// It returns false when the track that the collision manager's NEXT collision is about has just
// left the nucleus or been captured - which tells `Propagate` to check whether that collision is
// still the next one and to drop it if it is not. Everything else about the step succeeds.
//
// ## REFUSED, by name
//
//   * nothing.
#ifndef G4GPU_BIC_CASCADE_STEP_CUH
#define G4GPU_BIC_CASCADE_STEP_CUH

#include <cmath>

#include "physics/hadronic/bic/cascade_capture.cuh"
#include "physics/hadronic/bic/rk_propagation.cuh"

namespace g4gpu::bic {

/// `G4KineticTrack::UpdateTrackingMomentum(G4double aEnergy)` - the energy is set and the
/// momentum rebuilt from it at FIXED MASS, along the same direction.
///
/// The mass kept is the ACTUAL mass, so a track that was off shell stays off shell by the same
/// amount. A new energy below that mass would give an imaginary momentum; Geant4 does not guard
/// it and neither does this, because every caller tests `e + correction > GetActualMass()` first.
__host__ __device__ inline void update_tracking_momentum(CascadeTrack& t, double new_e) {
  const double mass = t.actual_mass();
  const double p2 = new_e * new_e - mass * mass;
  const deex::Vec3d dir = g4gpu::normalize(t.momentum.v);
  t.momentum = imr::LorentzVector(dir * ((p2 > 0.0) ? std::sqrt(p2) : 0.0), new_e);
}

/// `G4BinaryCascade::UpdateTracksAndCollisions`, the three list operations.
///
/// `old_secondary`, `old_target` and `new_secondary` are pool indices. Removing a track from a
/// list is a retag to `kListNone` plus `RemoveTracksCollisions` on the collision manager; adding
/// one is a retag to `kListSecondary`. The caller must then run `FindCollisions` on exactly the
/// new secondaries, which is what `find_collisions` is for - it is a parameter rather than a
/// separate call so that the ORDER cannot be got wrong: `G4BCDecay::GetCollisions` draws a
/// uniform, so finding collisions before or after the list is updated is a different event.
template <typename FindColl>
__host__ __device__ inline void update_tracks_and_collisions(
    BicCascadeState& st, imr::CollisionList& collisions, const int* old_secondary, int n_old_sec,
    const int* old_target, int n_old_tgt, const int* new_secondary, int n_new,
    FindColl&& find_collisions) {
  if (n_old_sec > 0) {
    for (int i = 0; i < n_old_sec; ++i) {
      if (st.lists.pool[old_secondary[i]].list == kListSecondary) {
        st.lists.pool[old_secondary[i]].list = kListNone;
      }
    }
    collisions.remove_tracks(old_secondary, n_old_sec);
  }
  if (n_old_tgt > 0) {
    for (int i = 0; i < n_old_tgt; ++i) { st.lists.pool[old_target[i]].list = kListNone; }
    collisions.remove_tracks(old_target, n_old_tgt);
  }
  if (n_new > 0) {
    for (int i = 0; i < n_new; ++i) { st.lists.pool[new_secondary[i]].list = kListSecondary; }
    find_collisions(new_secondary, n_new);
  }
}

/// What `CorrectBarionsOnBoundary` could not correct, and had to redirect instead.
struct BoundaryFailures {
  int index[32] = {};
  int n = 0;
  bool overflow = false;
};

/// `G4BinaryCascade::CorrectBarionsOnBoundary`.
///
/// `in` and `out` are pool indices of the tracks that crossed the surface this step. Returns the
/// ones that could not, which `DoTimeStep` reads as "recompute who went in and out".
template <typename Prop>
__host__ __device__ inline BoundaryFailures correct_barions_on_boundary(
    BicCascadeState& st, const int* in, int n_in, const int* out, int n_out,
    const Prop& propagator, double proton_mass, double neutron_mass, CascadeRefusal& ref) {
  BoundaryFailures fail;
  if (n_in > 0) {
    int secondaries_in = 0;
    int barions_in = 0;
    int charge_in = 0;
    double mass_in = 0.0;
    for (int i = 0; i < n_in; ++i) {
      const CascadeTrack& t = st.lists.pool[in[i]];
      ++secondaries_in;
      charge_in += t.charge;
      if (t.baryon != 0) {
        barions_in += t.baryon;
        // A resonance entering counts as a PROTON; see the file header and docs/RISK.md V156.
        mass_in += (t.pdg == imr::kPdgNeutron || t.pdg == imr::kPdgProton) ? t.pdg_mass
                                                                          : proton_mass;
      }
    }
    const double mass_initial = get_ion_mass(st.current_z, st.current_a, neutron_mass);
    st.current_z += charge_in;
    st.current_a += barions_in;
    const double mass_final = get_ion_mass(st.current_z, st.current_a, neutron_mass);
    double correction = mass_in + mass_initial - mass_final;
    if (secondaries_in > 1) { correction /= static_cast<double>(secondaries_in); }
    for (int i = 0; i < n_in; ++i) {
      CascadeTrack& t = st.lists.pool[in[i]];
      if (t.momentum.e + correction > t.actual_mass()) {
        update_tracking_momentum(t, t.momentum.e + correction);
      } else {
        // Cannot get in. The barrier is ADDED back - "Undo correction for Colomb Barrier".
        t.state = kMissNucleus;
        update_tracking_momentum(t, t.momentum.e + propagator.barrier(t.pdg));
        if (fail.n < 32) {
          fail.index[fail.n++] = in[i];
        } else {
          fail.overflow = true;
        }
        st.current_z -= t.charge;
        st.current_a -= t.baryon;
      }
    }
  }
  if (n_out > 0) {
    int secondaries_out = 0;
    int barions_out = 0;
    int charge_out = 0;
    double mass_out = 0.0;
    for (int i = 0; i < n_out; ++i) {
      const CascadeTrack& t = st.lists.pool[out[i]];
      ++secondaries_out;
      charge_out += t.charge;
      if (t.baryon != 0) {
        barions_out += t.baryon;
        // And a resonance LEAVING counts as a NEUTRON. One word apart from the branch above.
        mass_out += (t.pdg == imr::kPdgNeutron || t.pdg == imr::kPdgProton) ? t.pdg_mass
                                                                           : neutron_mass;
      }
    }
    const double mass_initial = get_ion_mass(st.current_z, st.current_a, neutron_mass);
    st.current_a -= barions_out;
    st.current_z -= charge_out;
    if (st.current_a < 0) {
      // Geant4 throws here. `currentZ < 0` is NOT checked - the source's own comment says "a
      // delta minus will do currentZ < 0 in light nuclei" and the test for it is commented out.
      ref.invalid_nucleus = true;
      return fail;
    }
    const double mass_final = get_ion_mass(st.current_z, st.current_a, neutron_mass);
    double correction = mass_initial - mass_final - mass_out;
    if (secondaries_out > 1) { correction /= static_cast<double>(secondaries_out); }
    for (int i = 0; i < n_out; ++i) {
      CascadeTrack& t = st.lists.pool[out[i]];
      if (t.momentum.e + correction > t.actual_mass()) {
        update_tracking_momentum(t, t.momentum.e + correction);
      } else if (t.pdg == imr::kPdgProton || t.pdg == imr::kPdgNeutron) {
        // Cannot get out. The barrier is SUBTRACTED - the same comment, the other sign.
        t.state = kCaptured;
        update_tracking_momentum(t, t.momentum.e - propagator.barrier(t.pdg));
        if (fail.n < 32) {
          fail.index[fail.n++] = out[i];
        } else {
          fail.overflow = true;
        }
        st.current_z += t.charge;
        st.current_a += t.baryon;
      }
      // A pion or a resonance that cannot get out is left ENTIRELY alone - not corrected, not
      // captured, still `gone_out`. That is the source's `else` with only a debug print in it.
    }
  }
  return fail;
}

/// A `CascadeTrack` seen as the `KineticTrack` the propagator takes, and back again.
///
/// `bic/kinetic_track.cuh`'s `KineticTrack` is `G4KineticTrack` with all three of its
/// four-vectors; `CascadeTrack` is the cascade's own slimmer record. The propagator reads and
/// writes the tracking momentum, the position and the state and nothing else, so the round trip
/// is those three plus the identity - and `set_tracking_momentum` is used rather than a member
/// assignment so that the `Get4Momentum`/`GetTrackingMomentum` round trip of docs/RISK.md V69
/// happens exactly where Geant4 has it.
__host__ __device__ inline KineticTrack as_kinetic_track(const CascadeTrack& t) {
  KineticTrack kt;
  kt.pdg = t.pdg;
  kt.pdg_mass = t.pdg_mass;
  kt.charge = t.charge;
  kt.baryon_number = t.baryon;
  kt.formation_time = t.formation_time;
  kt.position = t.position;
  kt.nucleon_index = t.nucleon_index;
  kt.state = t.state;
  kt.projectile_potential = t.projectile_potential;
  kt.creator_model_id = t.creator_model_id;
  kt.set_tracking_momentum(t.momentum);
  return kt;
}

__host__ __device__ inline void from_kinetic_track(const KineticTrack& kt, CascadeTrack& t) {
  t.position = kt.position;
  t.state = kt.state;
  t.momentum = kt.tracking_momentum();
  t.projectile_potential = kt.projectile_potential;
}

/// `G4RKPropagation::Transport(active, dummy, timeStep)` over the secondary list.
///
/// The momentum transfer is RESET at the top of every call - "reset momentum transfer to field"
/// is the source's own comment - and `DoTimeStep` then ADDS the call's total to the cascade's
/// running `theMomentumTransfer`. So a port that accumulated across calls would hand the
/// precompound stage a transfer summed over every step of the cascade instead of the last one.
template <typename Prop>
__host__ __device__ inline void transport_secondaries(BicCascadeState& st, Prop& propagator,
                                                      double time_step) {
  propagator.momentum_transfer = deex::Vec3d{0.0, 0.0, 0.0};
  for (int i = 0; i < st.lists.n_pool; ++i) {
    CascadeTrack& t = st.lists.pool[i];
    if (t.list != kListSecondary) { continue; }
    KineticTrack kt = as_kinetic_track(t);
    RkAdvanceReport rep;
    propagator.transport_one(kt, time_step, rep);
    from_kinetic_track(kt, t);
  }
}

/// What one `DoTimeStep` did, beyond changing the lists.
struct TimeStepReport {
  bool success = true;      ///< false when the next collision's primary has just gone
  int n_gone_out = 0;       ///< moved to theFinalState this step
  int n_captured = 0;       ///< moved to theCapturedList this step
  int n_gone_in = 0;
  bool boundary_overflow = false;
};

/// `G4BinaryCascade::DoTimeStep`.
///
/// The order is load-bearing and is the source's: take the `outside` and `inside` sets BEFORE
/// transporting, transport, add the propagator's transfer to the running one, then work out who
/// crossed by looking at the state of the tracks that WERE outside and inside. A track that was
/// outside and is now inside went in; one that was inside and is now `gone_out` went out.
/// `miss_nucleus` and `gone_out` among the ones that were OUTSIDE are appended to the final state
/// as well - those are the ones that missed the nucleus altogether or passed straight through.
template <typename Prop, typename FindColl>
__host__ __device__ inline TimeStepReport do_time_step(BicCascadeState& st,
                                                       imr::CollisionList& collisions,
                                                       Prop& propagator, double time_step,
                                                       double proton_mass, double neutron_mass,
                                                       FindColl&& find_collisions,
                                                       CascadeRefusal& ref) {
  TimeStepReport out;
  int was_outside[256];
  int was_inside[256];
  int n_was_out = 0;
  int n_was_in = 0;
  for (int i = 0; i < st.lists.n_pool; ++i) {
    const CascadeTrack& t = st.lists.pool[i];
    if (t.list != kListSecondary) { continue; }
    if (t.state == kOutside && n_was_out < 256) { was_outside[n_was_out++] = i; }
    if (t.state == kInside && n_was_in < 256) { was_inside[n_was_in++] = i; }
  }

  transport_secondaries(st, propagator, time_step);
  st.momentum_transfer = st.momentum_transfer + propagator.momentum_transfer;

  int gone_in[256];
  int gone_out[256];
  int n_in = 0;
  int n_out = 0;
  for (int k = 0; k < n_was_out; ++k) {
    if (st.lists.pool[was_outside[k]].state == kInside && n_in < 256) {
      gone_in[n_in++] = was_outside[k];
    }
  }
  for (int k = 0; k < n_was_in; ++k) {
    if (st.lists.pool[was_inside[k]].state == kGoneOut && n_out < 256) {
      gone_out[n_out++] = was_inside[k];
    }
  }
  const BoundaryFailures fail = correct_barions_on_boundary(st, gone_in, n_in, gone_out, n_out,
                                                            propagator, proton_mass,
                                                            neutron_mass, ref);
  out.boundary_overflow = fail.overflow;
  if (fail.n > 0) {
    // Some tracks that were supposed to cross were sent to miss_nucleus or captured instead, so
    // both sets are rebuilt from the states as they now are.
    n_in = 0;
    n_out = 0;
    for (int k = 0; k < n_was_out; ++k) {
      if (st.lists.pool[was_outside[k]].state == kInside && n_in < 256) {
        gone_in[n_in++] = was_outside[k];
      }
    }
    for (int k = 0; k < n_was_in; ++k) {
      if (st.lists.pool[was_inside[k]].state == kGoneOut && n_out < 256) {
        gone_out[n_out++] = was_inside[k];
      }
    }
  }
  out.n_gone_in = n_in;
  // The ones that were OUTSIDE and missed, or went straight through, join the same list.
  for (int k = 0; k < n_was_out; ++k) {
    const int s = st.lists.pool[was_outside[k]].state;
    if ((s == kMissNucleus || s == kGoneOut) && n_out < 256) { gone_out[n_out++] = was_outside[k]; }
  }
  for (int k = 0; k < n_out; ++k) { push_final(st, gone_out[k]); }
  out.n_gone_out = n_out;

  int captured[256];
  int n_cap = 0;
  for (int i = 0; i < st.lists.n_pool; ++i) {
    const CascadeTrack& t = st.lists.pool[i];
    if (t.list == kListSecondary && t.state == kCaptured && n_cap < 256) { captured[n_cap++] = i; }
  }

  // Is the collision the manager would run next about a track that has just left or been caught?
  if (collisions.size() > 0) {
    const int next = collisions.next_collision();
    if (next >= 0) {
      const int primary = collisions.items[next].primary;
      for (int k = 0; k < n_out; ++k) {
        if (gone_out[k] == primary) { out.success = false; }
      }
      for (int k = 0; k < n_cap; ++k) {
        if (captured[k] == primary) { out.success = false; }
      }
    }
  }

  // `UpdateTracksAndCollisions(kt_gone_out, 0, 0)`: the ones that left are already tagged
  // kListFinal, so this is the collision removal and nothing else.
  collisions.remove_tracks(gone_out, n_out);

  if (n_cap > 0) {
    for (int k = 0; k < n_cap; ++k) {
      st.lists.pool[captured[k]].list = kListCaptured;
      mark_hit(st, captured[k]);
    }
    collisions.remove_tracks(captured, n_cap);
  }
  out.n_captured = n_cap;
  (void)find_collisions;  // no new secondaries are made by a time step
  st.current_time += time_step;
  return out;
}

}  // namespace g4gpu::bic

#endif
