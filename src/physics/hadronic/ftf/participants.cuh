// G4FTFParticipants: the impact parameter, and which nucleons the projectile hits.
//
// Transcribed from
//   source/processes/hadronic/models/parton_string/diffraction/src/G4FTFParticipants.cc
//     GetList (both arms), SortInteractionsIncT, ShiftInteractionTime, Clean
//   .../management/src/G4VParticipants.cc          Init, InitProjectileNucleus
//   .../management/include/G4InteractionContent.hh the six setters GetList calls
//   source/processes/hadronic/util/include/G4V3DNucleus.hh
//     ChooseImpactXandY - the rejection sampler, inline in the header
//
// WHAT A POINTER IS HERE. Geant4's participants are a vector of `G4InteractionContent*`, each
// holding four raw pointers - two `G4VSplitableHadron*` and two `G4Nucleon*` - into objects
// owned by three different places. On the device they are indices into the workspace's two
// arrays, and the one pointer value that carries information is the NULL: GetList's
// hadron-nucleus arm leaves `SetTarget(0)` when the nucleon it just found was already hit, and
// every later reader of that interaction would dereference it. `kNullSplitable` is that null,
// and the model reports it rather than reading slot -1. (It is unreachable in a single event
// because `Init` rebuilds the nucleus with no marks and each nucleon is visited once per pass -
// but "unreachable" is the claim this port is not allowed to make silently.)
//
// THE RESAMPLING LOOP SPENDS DEVIATES AND KEEPS THE MARKS. `do { ... } while
// (theInteractions.size() == 0 && ++loopCounter < 1000)` re-samples the impact parameter when
// nothing was hit. Nothing was hit means no nucleon was marked, so the retry is clean; but the
// projectile-nucleus arm also calls `DoTranslation(theBeamPosition)` INSIDE the loop, and only
// when at least one interaction happened, so a retry does not undo a translation that never
// happened. Both are reproduced as written.
//
// THE INTERACTION TIME IS A LENGTH. `(z_projectile + z_nucleon)/betta_z` has a length over a
// dimensionless velocity, i.e. mm, and `ShiftInteractionTime` then assigns it to
// `SetTimeOfCreation`, which the fragmentation reads as ns. Geant4 carries the inconsistency
// (the comment in G4FTFParticipants.cc's ShiftInteractionTime says "To put correct times and
// z-coordinates"); this port carries it too, because changing it would move every hadron's
// formation time and nothing downstream in QBBC divides by c.
#pragma once
#include <cmath>

#include "core/units.cuh"
#include "physics/hadronic/bic/nucleus/nucleus_model.cuh"
#include "physics/hadronic/ftf/ftf_parameters.cuh"
#include "physics/hadronic/ftf/lorentz.cuh"
#include "physics/hadronic/ftf/refusal.cuh"
#include "physics/hadronic/ftf/splitable_hadron.cuh"

namespace g4gpu::hadronic::ftf {

/// Geant4's `G4VSplitableHadron* = 0`, as an index.
inline constexpr int kNullSplitable = -1;

/// G4InteractionContent, reduced to the seven fields G4FTFModel and G4DiffractiveExcitation
/// read. `theNumberOfHard` / `theNumberOfSoft` / `theNumberOfDiffractive` are set by nothing in
/// the FTF path (the soft-collision count that ExciteParticipants tests lives on the splitable
/// hadron, not here) and are not carried.
struct Interaction {
  int projectile = kNullSplitable;        ///< GetProjectile()
  int target = kNullSplitable;            ///< GetTarget()
  int projectile_nucleon = -1;            ///< GetProjectileNucleon(), -1 for a hadron projectile
  int target_nucleon = -1;                ///< GetTargetNucleon()
  double interaction_time = 0.0;
  int status = 0;                         ///< GetStatus(); 0 means "skipped after annihilation"
};

/// G4FTFParticipants' state plus the two splitable-hadron pools it fills.
///
/// The pool is indexed rather than allocated: slot 0 is the primary, slots
/// `[1, 1+kMaxA)` belong to target nucleon i, and `[1+kMaxA, 1+2*kMaxA)` to projectile nucleon
/// i. `bic::Nucleon::hit_by` carries the slot, which is exactly what `G4Nucleon::Hit(ptr)` and
/// `GetSplitableHadron()` carry in Geant4.
template <int kMaxA = bic::kMaxNucleons, int kMaxInteractions = 1024>
struct FtfParticipants {
  static constexpr int kPoolSize = 1 + 2 * kMaxA;
  static constexpr int kPrimarySlot = 0;
  static constexpr int kTargetBase = 1;
  static constexpr int kProjectileBase = 1 + kMaxA;

  SplitableHadron pool[kPoolSize];
  Interaction interactions[kMaxInteractions];
  int n_interactions = 0;
  int current_interaction = -1;  ///< G4FTFParticipants::currentInteraction

  double b_impact = 0.0;         ///< Bimpact
  bool bin_interval = false;     ///< BinInterval - false in every QBBC run; see the note below
  double bmin2 = -1.0;           ///< Bmin2
  double bmax2 = -1.0;           ///< Bmax2

  bool interaction_capacity = false;  ///< reported, never truncated
  bool null_target = false;           ///< an interaction came out with SetTarget(0)
  bool loops_exhausted = false;       ///< the 1000-attempt impact-parameter loop gave up

  __host__ __device__ void start_loop() { current_interaction = -1; }
  __host__ __device__ bool next() { return ++current_interaction < n_interactions; }
  __host__ __device__ Interaction& interaction() { return interactions[current_interaction]; }
  __host__ __device__ void clean() {
    n_interactions = 0;
    current_interaction = -1;
    for (int i = 0; i < kPoolSize; ++i) { pool[i].alive = false; }
  }
};

/// G4V3DNucleus::ChooseImpactXandY - a point uniform in the unit disc, scaled.
///
/// Two deviates per rejection iteration and an acceptance of pi/4, so the mean cost is 8/pi
/// deviates. The rejection is `x*x + y*y > 1`, strictly greater, so the boundary is accepted.
template <typename Rng>
__host__ __device__ inline void ftf_choose_impact_xy(double max_impact, double* impact_x,
                                                     double* impact_y, Rng& rng) {
  double x = 0.0, y = 0.0;
  do {
    x = 2.0 * rng.uniform() - 1.0;
    y = 2.0 * rng.uniform() - 1.0;
  } while (x * x + y * y > 1.0);
  *impact_x = x * max_impact;
  *impact_y = y * max_impact;
}

/// G4FTFParticipants::ShiftInteractionTime.
///
/// The FIRST interaction is left alone - the loop starts at 1 - so interaction 0 keeps its
/// absolute time and every other one is measured from it. The projectile's z is then copied
/// from the target's, which is what makes the two ends of a collision share a point.
template <int kMaxA, int kMaxInteractions>
__host__ __device__ inline void ftf_shift_interaction_time(
    FtfParticipants<kMaxA, kMaxInteractions>* p) {
  const double initial_time = p->interactions[0].interaction_time;
  for (int i = 1; i < p->n_interactions; ++i) {
    const double inter_time = p->interactions[i].interaction_time - initial_time;
    p->interactions[i].interaction_time = inter_time;
    const int pr = p->interactions[i].projectile;
    const int tr = p->interactions[i].target;
    if (pr == kNullSplitable || tr == kNullSplitable) {
      p->null_target = true;
      continue;
    }
    Vec3d pr_position = p->pool[pr].position;
    pr_position.z = p->pool[tr].position.z;
    p->pool[pr].position = pr_position;
    p->pool[pr].time_of_creation = inter_time;
    p->pool[tr].time_of_creation = inter_time;
  }
}

/// G4FTFParticipants::SortInteractionsIncT.
///
/// Geant4 uses `std::sort` with a strict `<` on the interaction time, which is unspecified for
/// equal keys; an insertion sort is used here and IS stable. Two interactions with bitwise
/// equal times means two nucleon pairs with equal `z_p + z_t`, which for sampled positions is a
/// measure-zero event - the same argument P9 makes for `SortNucleonsIncZ`
/// (bic/nucleus/fancy_3d_nucleus.cuh).
template <int kMaxA, int kMaxInteractions>
__host__ __device__ inline void ftf_sort_interactions_inc_t(
    FtfParticipants<kMaxA, kMaxInteractions>* p) {
  if (p->n_interactions < 2) { return; }  // Geant4's "Avoid unnecesary work"
  for (int i = 1; i < p->n_interactions; ++i) {
    Interaction key = p->interactions[i];
    int j = i - 1;
    while (j >= 0 && key.interaction_time < p->interactions[j].interaction_time) {
      p->interactions[j + 1] = p->interactions[j];
      --j;
    }
    p->interactions[j + 1] = key;
  }
}

/// Append one interaction, or report the capacity.
template <int kMaxA, int kMaxInteractions>
__host__ __device__ inline bool ftf_push_interaction(
    FtfParticipants<kMaxA, kMaxInteractions>* p, const Interaction& in) {
  if (p->n_interactions >= kMaxInteractions) {
    p->interaction_capacity = true;
    return false;
  }
  p->interactions[p->n_interactions++] = in;
  return true;
}

/// G4FTFParticipants::GetList - the hadron-nucleus (and anti-baryon-nucleus) arm.
///
/// `thePrimary` is the projectile as `Scatter` left it: rotated onto +z, so `betta_z` is
/// `|p|/E` up to rounding and the 1e-10 floor never fires for a real beam. It fires for a
/// projectile at rest, which the anti-nucleon arm of QBBC can produce, and then every
/// interaction time is 1e10 times a length.
///
/// The impact-parameter RANGE is `GetOuterRadius() + 2 fm`, and `GetOuterRadius` is a property
/// of the sampled configuration (the furthest nucleon plus one hard-core distance), so it
/// changes event to event - which is why the participant list cannot be replayed from (A, Z)
/// alone and the oracle has to replay the nucleus as well.
template <int kMaxA, int kMaxInteractions, typename Rng>
__host__ __device__ inline void ftf_participants_get_list_hadron(
    FtfParticipants<kMaxA, kMaxInteractions>* p, bic::Nucleus3D* target,
    const FtfParameters<double>* params, int primary_pdg, const Vec4& primary_p4, Rng& rng) {

  double betta_z = primary_p4.v.z / primary_p4.e;
  if (betta_z < 1.0e-10) { betta_z = 1.0e-10; }

  p->start_loop();
  p->n_interactions = 0;

  const double deltaxy = 2.0 * deex::fermi();  // Extra nuclear radius

  const int primary = FtfParticipants<kMaxA, kMaxInteractions>::kPrimarySlot;
  p->pool[primary] = splitable_from_primary(primary_pdg, primary_p4);

  const double xyradius = target->outer_radius() + deltaxy;
  // Geant4 writes `impact2/fermi/fermi` - TWO divisions, not one by fermi squared. The two differ
  // by an ulp, and the consumer is `GetProbabilityOfInteraction`, which is a STEP function:
  // `RadiusOfHNinteractions2 > impact_square ? 1 : 0`. An ulp either side of that step is a
  // whole participant. Measured: with `impact2 / (fermi*fermi)` the port found two
  // participants where Geant4 found one for a pi+ on Al27 at phase 0 of the cycle, and every
  // other column of ref/oracle/ftf_getlist.csv - the impact parameter, the draw count, the
  // identity and time of the participant they agreed on - was still exact.
  const double fermi_unit = deex::fermi();

  const int max_number_of_loops = 1000;
  int loop_counter = 0;
  double impact_x = 0.0, impact_y = 0.0;
  do {
    if (p->bin_interval) {
      const double b2 = p->bmin2 + rng.uniform() * (p->bmax2 - p->bmin2);
      const double b = (b2 > 0.0) ? std::sqrt(b2) : 0.0;
      const double phi = units::twopi<double>() * rng.uniform();
      impact_x = b * std::cos(phi);
      impact_y = b * std::sin(phi);
      p->b_impact = b;
    } else {
      ftf_choose_impact_xy(xyradius, &impact_x, &impact_y, rng);
      p->b_impact = std::sqrt(impact_x * impact_x + impact_y * impact_y);
    }

    p->pool[primary].position = Vec3d{impact_x, impact_y, 0.0};

    target->start_loop();
    for (int i = 0; i < target->my_a; ++i) {
      bic::Nucleon* nucleon = target->next_nucleon();
      const double dx = impact_x - nucleon->position.x;
      const double dy = impact_y - nucleon->position.y;
      const double impact2 = dx * dx + dy * dy;

      if (ftf_get_probability_of_interaction(params, impact2 / fermi_unit / fermi_unit) > rng.uniform()) {
        p->pool[primary].status = 1;  // It takes part in the interaction
        int target_slot = kNullSplitable;
        if (!nucleon->hit) {
          target_slot = FtfParticipants<kMaxA, kMaxInteractions>::kTargetBase + i;
          p->pool[target_slot] =
              splitable_from_nucleon(nucleon->pdg(), nucleon->momentum, nucleon->position);
          nucleon->hit = true;
          nucleon->hit_by = target_slot;
          p->pool[target_slot].status = 1;
        } else {
          p->null_target = true;
        }
        Interaction in;
        in.projectile = primary;
        in.projectile_nucleon = -1;  // SetProjectileNucleon( 0 ) - explicit in the original
        in.target = target_slot;
        in.target_nucleon = i;
        in.status = 1;
        in.interaction_time =
            (p->pool[primary].position.z + nucleon->position.z) / betta_z;
        if (!ftf_push_interaction(p, in)) { return; }
      }
    }
  } while ((p->n_interactions == 0) && ++loop_counter < max_number_of_loops);

  if (loop_counter >= max_number_of_loops) {
    p->loops_exhausted = true;
    return;  // Geant4 returns with an empty list, which Scatter reads as a failed attempt
  }

  // SortInteractionsIncT() is commented out in the original with the reason
  // "Not need because nucleons are sorted in increasing z-coordinates" - G4VParticipants::Init
  // calls SortNucleonsIncZ, so the loop above already visited them in z order.
  ftf_shift_interaction_time(p);
}

/// G4FTFParticipants::GetList - the nucleus-nucleus (and antinucleus-nucleus) arm.
///
/// The projectile nucleus has been boosted and Lorentz-contracted by `Init` before this runs,
/// and it is sorted in DECREASING z (`SortNucleonsDecZ`) against the target's increasing z, so
/// the double loop walks the two nuclei towards each other.
///
/// `DoTranslation( theBeamPosition )` moves the whole projectile nucleus to the impact point
/// and is inside the loop, guarded by `theInteractions.size() != 0`: a pass that found nothing
/// leaves the projectile where it was and re-samples, so the translation happens exactly once.
template <int kMaxA, int kMaxInteractions, typename Rng>
__host__ __device__ inline void ftf_participants_get_list_nucleus(
    FtfParticipants<kMaxA, kMaxInteractions>* p, bic::Nucleus3D* target,
    bic::Nucleus3D* projectile, const FtfParameters<double>* params, const Vec4& primary_p4,
    Rng& rng) {

  double betta_z = primary_p4.v.z / primary_p4.e;
  if (betta_z < 1.0e-10) { betta_z = 1.0e-10; }

  p->start_loop();
  p->n_interactions = 0;

  const double deltaxy = 2.0 * deex::fermi();
  const double xyradius = projectile->outer_radius() + target->outer_radius() + deltaxy;
  // Geant4 writes `impact2/fermi/fermi` - TWO divisions, not one by fermi squared. The two differ
  // by an ulp, and the consumer is `GetProbabilityOfInteraction`, which is a STEP function:
  // `RadiusOfHNinteractions2 > impact_square ? 1 : 0`. An ulp either side of that step is a
  // whole participant. Measured: with `impact2 / (fermi*fermi)` the port found two
  // participants where Geant4 found one for a pi+ on Al27 at phase 0 of the cycle, and every
  // other column of ref/oracle/ftf_getlist.csv - the impact parameter, the draw count, the
  // identity and time of the participant they agreed on - was still exact.
  const double fermi_unit = deex::fermi();

  const int max_number_of_loops = 1000;
  int loop_counter = 0;
  double impact_x = 0.0, impact_y = 0.0;
  do {
    if (p->bin_interval) {
      const double b2 = p->bmin2 + rng.uniform() * (p->bmax2 - p->bmin2);
      const double b = (b2 > 0.0) ? std::sqrt(b2) : 0.0;
      const double phi = units::twopi<double>() * rng.uniform();
      impact_x = b * std::cos(phi);
      impact_y = b * std::sin(phi);
      p->b_impact = b;
    } else {
      ftf_choose_impact_xy(xyradius, &impact_x, &impact_y, rng);
      p->b_impact = std::sqrt(impact_x * impact_x + impact_y * impact_y);
    }

    const Vec3d beam_position{impact_x, impact_y, 0.0};

    for (int ip = 0; ip < projectile->my_a; ++ip) {
      bic::Nucleon* proj_nucleon = &projectile->nucleons[ip];
      int projectile_slot = kNullSplitable;
      for (int it = 0; it < target->my_a; ++it) {
        bic::Nucleon* targ_nucleon = &target->nucleons[it];
        const double dx = impact_x + proj_nucleon->position.x - targ_nucleon->position.x;
        const double dy = impact_y + proj_nucleon->position.y - targ_nucleon->position.y;
        const double impact2 = dx * dx + dy * dy;
        int target_slot = kNullSplitable;
        if (ftf_get_probability_of_interaction(params, impact2 / fermi_unit / fermi_unit) > rng.uniform()) {
          if (!proj_nucleon->hit) {
            projectile_slot =
                FtfParticipants<kMaxA, kMaxInteractions>::kProjectileBase + ip;
            p->pool[projectile_slot] = splitable_from_nucleon(
                proj_nucleon->pdg(), proj_nucleon->momentum, proj_nucleon->position);
            proj_nucleon->hit = true;
            proj_nucleon->hit_by = projectile_slot;
            p->pool[projectile_slot].status = 1;
          } else {
            projectile_slot = proj_nucleon->hit_by;
          }
          if (!targ_nucleon->hit) {
            target_slot = FtfParticipants<kMaxA, kMaxInteractions>::kTargetBase + it;
            p->pool[target_slot] = splitable_from_nucleon(
                targ_nucleon->pdg(), targ_nucleon->momentum, targ_nucleon->position);
            targ_nucleon->hit = true;
            targ_nucleon->hit_by = target_slot;
            p->pool[target_slot].status = 1;
          } else {
            target_slot = targ_nucleon->hit_by;
          }

          Interaction in;
          in.projectile = projectile_slot;
          in.target = target_slot;
          in.projectile_nucleon = ip;
          in.target_nucleon = it;
          in.interaction_time =
              (proj_nucleon->position.z + targ_nucleon->position.z) / betta_z;
          in.status = 1;
          if (!ftf_push_interaction(p, in)) { return; }
        }
      }
    }

    if (p->n_interactions != 0) { projectile->do_translation(beam_position); }

  } while ((p->n_interactions == 0) && ++loop_counter < max_number_of_loops);

  if (loop_counter >= max_number_of_loops) {
    p->loops_exhausted = true;
    return;
  }

  ftf_sort_interactions_inc_t(p);
  ftf_shift_interaction_time(p);
}

}  // namespace g4gpu::hadronic::ftf
