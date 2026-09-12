// The cascade's scheduler: when two tracks will collide, and the list of collisions waiting.
//
// Transcribed from G4Scatterer.cc (`GetTimeToInteraction`, `GetCrossSection`, `FindCollision`,
// `GetCollisions`, `GetFinalState`, `Scatter`), G4CollisionInitialState.{hh,cc} and
// G4CollisionManager.cc (im_r_matrix, 11.1.1).
//
// `G4BinaryCascade::FindCollisions` walks every pair of a participant and a candidate target and
// asks `GetTimeToInteraction`; whatever comes back below DBL_MAX becomes a `G4CollisionInitialState`
// in the manager, and `DoTimeStep` repeatedly takes the earliest one. So this file is the loop
// control of the whole cascade, and every one of its four early exits is a collision that does
// not happen.
//
// ## The five gates, in the order they are applied
//
//   1. `collisionTime > 0`. The target nucleons are FROZEN - only track 1 moves - so the time is
//      `(dx . v)/|v|^2` and a target behind the projectile is simply not hit.
//   2. `0.7*pi*distance_fast > 500 mb`, on the LAB transverse distance. A cheap rejection before
//      any boost, and the 0.7 is a margin against the CM impact parameter being larger.
//   3. `pi*distance > 500 mb`, on the CM impact parameter squared.
//   4. `pi*distance > 200 mb` if BOTH tracks are charged.
//   5. `pi*distance > 200 mb` if EITHER track is a neutron and sqrt(s) > 1.91 GeV - the comment
//      says pn is the largest cross section and is under 200 mb above 1.91 GeV.
//   Then `(m1_actual + m2_actual) > sqrt(s)` returns DBL_MAX, and only then is the channel found
//   and its cross section compared against `pi*distance`.
//
// ## Three things in `GetTimeToInteraction` that a reader would not predict
//
// **The target is put at rest for the geometry and left moving for the energy.** The CM boost is
// built from `mom1 + G4LorentzVector(0,0,0, trk2.Get4Momentum().mag())` - track 2 replaced by a
// particle of its own INVARIANT MASS at rest - while `sqrtS` two lines later is
// `(trk1.Get4Momentum() + trk2.Get4Momentum()).mag()`, the real one, and the cross section is
// evaluated on the real tracks too. So the impact parameter is computed in a frame that is not
// the pair's CM frame whenever the target is moving, which in the cascade it is: a target nucleon
// carries Fermi momentum. Transcribed as written.
//
// **The two positions are given a time of 100 before being boosted.** `G4LorentzVector
// coordinate1(trk1.GetPosition(), 100.)` - an undocumented 100 ns, the same for both - and the
// boost mixes that time into the spatial components. Because it is the SAME for both and only
// the difference is used, `(gamma*100*beta - gamma*100*beta)` cancels exactly and the value does
// not matter; what matters is that it is identical in the two, so a port that used 0 in one and
// 100 in the other would be wrong by `gamma*beta*100`.
//
// **The fast z-aligned branch and the general branch compute different things.** The first is
// taken when `|mom1.unit().z() - 1| < 1e-6` and uses `deltaz/velocity_z` with
// `distance_fast = dx^2 + dy^2`; the second projects out the velocity direction. They agree when
// the momentum is along +z and the second is a strict generalisation, but the first divides by
// `mom1.z()` where the second divides by `|mom1|^2`, so they round differently. Both are here.
//
// ## REFUSED, by name
//
//   * **`G4Scatterer::Scatter` and `GetFinalState`** for a nucleon pair: they call
//     `G4CollisionComposite::FinalState`, whose selection needs the six resonance-production
//     partial cross sections this package does not have. `ScatterRefusal::final_state`.
//   * a meson-baryon pair whose 32-point BUFFER the caller did not supply.
//     `G4CollisionMesonBaryon` has no cross-section source, so its total is a cache built once
//     per pair of particle definitions inside `CrossSection` itself, under `bufferMutex`. A
//     device kernel cannot allocate one, so the cascade owns it and builds it up front;
//     `ScatterRefusal::meson_baryon` fires if it arrives null or unbuilt. The channel itself is
//     no longer refused: `collision_meson.cuh` has both components.
//   * **`G4CollisionManager::Print`** and the `debug_G4CollisionManager` block in
//     `GetNextCollision`, which print and change nothing.
#ifndef G4GPU_BIC_IMR_SCATTERER_CUH
#define G4GPU_BIC_IMR_SCATTERER_CUH

#include <cfloat>
#include <cmath>

#include "physics/hadronic/bic/im_r/collision_meson.cuh"
#include "physics/hadronic/bic/im_r/collision_nn.cuh"
#include "physics/hadronic/bic/kinetic_track.cuh"

namespace g4gpu::bic::imr {

/// What the scatterer could not do.
struct ScatterRefusal {
  bool meson_baryon = false;  ///< a meson-baryon pair arrived without a built buffer
  bool final_state = false;   ///< Scatter/GetFinalState needs the resonance partials
  bool list_full = false;     ///< the collision list's capacity
  int pdg1 = 0;
  int pdg2 = 0;
  CollisionRefusal collision;
  MesonRefusal meson;
  __host__ __device__ bool any() const { return meson_baryon || final_state || list_full; }
};

/// The three constants `GetTimeToInteraction` rejects on, as `static const G4double` locals.
__host__ __device__ inline double max_cross_section() { return 500.0 * millibarn(); }
__host__ __device__ inline double max_charged_cross_section() { return 200.0 * millibarn(); }
/// The neutron rule's threshold, `sqrtS > 1.91*GeV`.
__host__ __device__ inline double neutron_special_sqrt_s() { return 1.91 * u::GeV<double>(); }

/// `G4Scatterer::FindCollision` - the first registered channel that is in charge, in the order
/// `GROUP2(G4CollisionNN, G4CollisionMesonBaryon)` registers them.
///
/// Returns 0 for G4CollisionNN, 1 for G4CollisionMesonBaryon and -1 for neither. Both tests are
/// the channels' own `IsInCharge`: `G4GeneralNNCollision`'s two-nucleon test and
/// `G4CollisionComposite`'s scan over the meson-baryon composite's two components.
__host__ __device__ inline int scatterer_find_collision(int pdg1, int pdg2,
                                                        ScatterRefusal& ref) {
  if (collision_nn_is_in_charge(pdg1, pdg2)) { return 0; }
  if (meson_baryon_is_in_charge(pdg1, pdg2, ref.meson.xsec)) { return 1; }
  return -1;
}

/// `G4Scatterer::GetCrossSection` - the in-charge channel's cross section, or zero if none is.
///
/// `mb` is the meson-baryon composite's 32-point buffer for THIS pair of definitions, or null.
/// Geant4 builds that buffer lazily inside the call, under `bufferMutex`, and keeps one per pair
/// on the composite for the life of the run; a device kernel cannot allocate, so the caller owns
/// it and passing null for a meson-baryon pair is a refusal rather than a zero.
__host__ __device__ inline double scatterer_cross_section(
    int pdg1, int pdg2, const LorentzVector& p1, const LorentzVector& p2, double actual1,
    double actual2, double pdg1_mass, double pdg2_mass, ScatterRefusal& ref,
    const MesonBaryonBuffers* mb = nullptr) {
  const int channel = scatterer_find_collision(pdg1, pdg2, ref);
  if (channel == 0) {
    return collision_nn_cross_section(pdg1, pdg2, p1, p2, actual1, actual2, pdg1_mass, pdg2_mass,
                                      ref.collision.xsec);
  }
  if (channel != 1) { return 0.0; }
  if (mb == nullptr || !mb->built) {
    ref.meson_baryon = true;
    ref.pdg1 = pdg1;
    ref.pdg2 = pdg2;
    return 0.0;
  }
  return meson_baryon_cross_section(pdg1, pdg2, (p1 + p2).mag(), *mb, ref.meson);
}

/// The two intermediate quantities `GetTimeToInteraction` computes before it decides, returned
/// so a test can compare them separately from the verdict. `time` is DBL_MAX for no collision.
struct TimeToInteraction {
  double time = DBL_MAX;
  double collision_time = 0.0;   ///< before the gates, and negative for a receding target
  double distance_fast = 0.0;    ///< the LAB transverse distance squared, mm^2
  double distance = 0.0;         ///< the CM impact parameter squared, mm^2
  double sqrt_s = 0.0;
  double cross_section = 0.0;    ///< only filled if the gates were passed
  int gate = 0;                  ///< which gate stopped it; 0 = a collision, see below
};

/// Which of `GetTimeToInteraction`'s exits was taken. Reported rather than collapsed into
/// DBL_MAX, because "no collision" is five different physical statements and a cascade that
/// stops early is diagnosed by which one.
enum TimeGate : int {
  kGateCollision = 0,
  kGateNegativeTime = 1,     ///< collisionTime <= 0: the target is behind
  kGateFastDistance = 2,     ///< 0.7*pi*distance_fast > 500 mb
  kGateCmDistance = 3,       ///< pi*distance > 500 mb
  kGateCharged = 4,          ///< both charged and pi*distance > 200 mb
  kGateNeutron = 5,          ///< a neutron above 1.91 GeV and pi*distance > 200 mb
  kGateBelowThreshold = 6,   ///< m1 + m2 > sqrt(s)
  kGateNoChannel = 7,        ///< FindCollision returned nothing
  kGateZeroCrossSection = 8, ///< the channel's cross section is zero
  kGateTooFar = 9            ///< distance > sigma/pi
};

/// `G4Scatterer::GetTimeToInteraction(trk1, trk2)`.
///
/// `pos1`, `pos2` are the two positions; the rest is the two tracks' kinematics. Written against
/// the pieces rather than against `KineticTrack` so the caller can see which momentum is which:
/// Geant4 uses `GetTrackingMomentum()` for the velocity and `Get4Momentum()` for everything else,
/// and those are the same four-vector only because `theFermi3Momentum` is inert (docs/RISK.md V69).
__host__ __device__ inline TimeToInteraction scatterer_time_to_interaction(
    int pdg1, int pdg2, int charge1, int charge2, const Vec3d& pos1, const Vec3d& pos2,
    const LorentzVector& tracking1, const LorentzVector& p1, const LorentzVector& p2,
    double actual1, double actual2, double pdg1_mass, double pdg2_mass, ScatterRefusal& ref,
    const MesonBaryonBuffers* mb = nullptr) {
  TimeToInteraction out;
  const LorentzVector mom1_in = tracking1;
  double collision_time = 0.0;
  double distance_fast = 0.0;

  const Vec3d unit1 = g4gpu::normalize(mom1_in.v);
  if (std::fabs(unit1.z - 1.0) < 1e-6) {
    // The z-aligned fast path: the time is deltaz over the z velocity alone.
    const Vec3d position = pos2 - pos1;
    const double deltaz = position.z;
    const double velocity = mom1_in.v.z / mom1_in.e * u::c_light<double>();
    collision_time = deltaz / velocity;
    distance_fast = position.x * position.x + position.y * position.y;
  } else {
    // The nucleons of the nucleus are FROZEN, i.e. do not move.
    Vec3d position = pos2 - pos1;
    const Vec3d velocity = (1.0 / mom1_in.e) * mom1_in.v * u::c_light<double>();
    collision_time = g4gpu::dot(position, velocity) / g4gpu::mag2(velocity);
    position = position - velocity * collision_time;
    distance_fast = g4gpu::mag2(position);
  }
  out.collision_time = collision_time;
  out.distance_fast = distance_fast;

  if (!(collision_time > 0.0)) {
    out.gate = kGateNegativeTime;
    return out;
  }
  if (0.7 * u::pi<double>() * distance_fast > max_cross_section()) {
    out.gate = kGateFastDistance;
    return out;
  }

  // The target replaced by a particle of its own invariant mass AT REST - see the file header.
  LorentzVector mom1 = mom1_in;
  LorentzVector mom2(Vec3d{0.0, 0.0, 0.0}, p2.mag());
  const Vec3d to_cms = -1.0 * (mom1 + mom2).boost_vector();
  const LorentzRotation to_cms_frame = LorentzRotation::from_boost(to_cms);
  mom1 = to_cms_frame * mom1;
  mom2 = to_cms_frame * mom2;

  // The 100 in the fourth component is Geant4's and is the same in both, so it cancels.
  const LorentzVector coordinate1(pos1, 100.0);
  const LorentzVector coordinate2(pos2, 100.0);
  const Vec3d pos = (to_cms_frame * coordinate1).v - (to_cms_frame * coordinate2).v;
  const Vec3d mom = mom1.v - mom2.v;

  const double dot_pm = g4gpu::dot(pos, mom);
  const double distance = g4gpu::dot(pos, pos) - dot_pm * dot_pm / g4gpu::mag2(mom);
  out.distance = distance;

  if (u::pi<double>() * distance > max_cross_section()) {
    out.gate = kGateCmDistance;
    return out;
  }
  // `std::abs(GetPDGCharge()) > 0.1` on both.
  if (charge1 != 0 && charge2 != 0 &&
      u::pi<double>() * distance > max_charged_cross_section()) {
    out.gate = kGateCharged;
    return out;
  }
  const double sqrt_s = (p1 + p2).mag();
  out.sqrt_s = sqrt_s;
  if ((pdg1 == kPdgNeutron || pdg2 == kPdgNeutron) && sqrt_s > neutron_special_sqrt_s() &&
      u::pi<double>() * distance > max_charged_cross_section()) {
    out.gate = kGateNeutron;
    return out;
  }
  if ((actual1 + actual2) > sqrt_s) {
    out.gate = kGateBelowThreshold;
    return out;
  }

  const int channel = scatterer_find_collision(pdg1, pdg2, ref);
  if (channel < 0) {
    out.gate = kGateNoChannel;
    return out;
  }
  const double sigma = scatterer_cross_section(pdg1, pdg2, p1, p2, actual1, actual2, pdg1_mass,
                                               pdg2_mass, ref, mb);
  if (ref.meson_baryon) {
    out.gate = kGateNoChannel;
    return out;
  }
  out.cross_section = sigma;
  if (!(sigma > 0.0)) {
    out.gate = kGateZeroCrossSection;
    return out;
  }
  if (distance <= sigma / u::pi<double>()) {
    out.time = collision_time;
    out.gate = kGateCollision;
  } else {
    out.gate = kGateTooFar;
  }
  return out;
}

// =============================================================================================
// G4CollisionInitialState and G4CollisionManager.
// =============================================================================================

/// `G4CollisionInitialState`, with the two track pointers as indices into the cascade's track
/// array. `theTs` - the target COLLECTION - is one entry for every collision `G4Scatterer`
/// makes, because `GetCollisions` pushes exactly one target into it; the vector exists for
/// `G4MesonAbsorption`, which makes two-target collisions and which this package does not have.
/// So the collection is a single index here and the multi-target case is refused where it would
/// be needed rather than carried empty.
struct CollisionInitialState {
  double collision_time = DBL_MAX;
  int primary = -1;   ///< index of thePrimary
  int target = -1;    ///< index of theTarget, or of theTs[0] when built by GetCollisions
  int generator = -1; ///< which G4BCAction made it: 0 = G4Scatterer; -1 = none
  bool alive = false;
};

/// `G4CollisionManager`'s list, in a caller-owned array. The cascade's own state never lives in
/// a local: a nucleus of 208 nucleons against a handful of participants is several hundred
/// pending collisions, and the brief's rule is that the list travels by pointer.
struct CollisionList {
  CollisionInitialState* items = nullptr;
  int capacity = 0;
  int n = 0;  ///< one past the highest slot ever used; dead slots inside are `alive == false`

  /// `G4CollisionManager::ClearAndDestroy`.
  __host__ __device__ void clear() { n = 0; }

  /// `G4CollisionManager::AddCollision(time, proj, target)`. Geant4 throws a
  /// `G4HadronicException` for `time == DBL_MAX` after printing both tracks; a kernel cannot
  /// throw, so the collision is simply not added and the caller is told - which is also what the
  /// only caller, `G4BinaryCascade::FindCollisions`, would want, since it has already filtered
  /// on `GetTimeToInteraction` returning less than DBL_MAX.
  __host__ __device__ bool add(double time, int primary, int target, int generator,
                               ScatterRefusal& ref) {
    if (!(time < DBL_MAX)) { return false; }
    if (n >= capacity) {
      ref.list_full = true;
      return false;
    }
    CollisionInitialState& c = items[n++];
    c.collision_time = time;
    c.primary = primary;
    c.target = target;
    c.generator = generator;
    c.alive = true;
    return true;
  }

  /// `G4CollisionManager::RemoveCollision`.
  __host__ __device__ void remove(int index) {
    if (index >= 0 && index < n) { items[index].alive = false; }
  }

  /// `G4CollisionManager::RemoveTracksCollisions(toBeCaned)` - every collision whose primary,
  /// whose `theTarget` or whose target collection contains one of the listed tracks.
  ///
  /// Geant4 collects them into a second vector and erases afterwards, with the comment "cannot
  /// remove the collision from the list inside the loop". Marking is the same thing without the
  /// second vector, and the ORDER of what is left is preserved either way - which matters,
  /// because `GetNextCollision` breaks ties by taking the FIRST of equal times.
  __host__ __device__ void remove_tracks(const int* tracks, int n_tracks) {
    if (tracks == nullptr || n_tracks <= 0) { return; }
    for (int i = 0; i < n; ++i) {
      if (!items[i].alive) { continue; }
      for (int j = 0; j < n_tracks; ++j) {
        if (items[i].primary == tracks[j] || items[i].target == tracks[j]) {
          items[i].alive = false;
          break;
        }
      }
    }
  }

  /// `G4CollisionManager::GetNextCollision` - the earliest, with a STRICT `>` so that the first
  /// of several equal times wins. Returns -1 for an empty list.
  __host__ __device__ int next_collision() const {
    int the_next = -1;
    double next_time = DBL_MAX;
    for (int i = 0; i < n; ++i) {
      if (!items[i].alive) { continue; }
      if (next_time > items[i].collision_time) {
        next_time = items[i].collision_time;
        the_next = i;
      }
    }
    return the_next;
  }

  /// How many collisions are pending - `theCollisionList->size()`, which
  /// `G4BinaryCascade::Propagate` reads to decide whether the cascade is over.
  __host__ __device__ int size() const {
    int c = 0;
    for (int i = 0; i < n; ++i) {
      if (items[i].alive) { ++c; }
    }
    return c;
  }
};

}  // namespace g4gpu::bic::imr

#endif
