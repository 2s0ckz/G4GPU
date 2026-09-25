// G4BinaryCascade's state, and the bookkeeping every part of Propagate reads.
//
// Transcribed from G4BinaryCascade.{hh,cc} (models/binary_cascade, 11.1.1): the four track lists,
// the running (A, Z), `BuildTargetList`, `BuildLateParticleCollisions`, `GetIonMass`,
// `GetFinal4Momentum`, `GetFinalNucleusMomentum` and `GetExcitationEnergy`.
//
// `G4BinaryCascade` keeps its cascade in four `G4KineticTrackVector`s and moves tracks between
// them; every decision in `Propagate` is about which list a track is in.
//
//     theSecondaryList   the cascade's own particles - the projectile, then everything a
//                        collision or a decay puts back into the nucleus
//     theTargetList      the nucleus's nucleons that have not been hit by a higher-energy model
//     theCapturedList    nucleons a secondary dropped below `theCutOnP` inside the nucleus
//     theFinalState      everything that has left, or is leaving
//
// and two counters, `currentA` and `currentZ`, that track what is left of the nucleus. The whole
// excitation energy at the end is the difference between what went in and what came out, so a
// track counted in the wrong list is an excitation energy that is wrong by a nucleon mass.
//
// ## THE EXCITATION ENERGY IS A DIFFERENCE OF TWO LARGE NUMBERS
//
// `GetExcitationEnergy` is `GetFinalNucleusMomentum().mag() - GetIonMass(currentZ, currentA)`,
// and the first term is the initial nucleus plus the projectile minus every track in
// theFinalState, boosted so that its momentum equals the captured nucleons'. For a 1 GeV proton
// on lead both terms are near 194 GeV and the answer is tens of MeV, so this is a cancellation of
// four significant figures and every term in it has to be built the same way Geant4 builds it -
// including `GetIonMass`'s three-way fallback and the `boost.mag2() > 1` guard that zeroes the
// whole thing rather than throwing.
//
// ## `theCutOnP` IS ALWAYS 45 MeV, AND THE OBVIOUS READING OF IT IS WRONG
//
//     theCutOnP=90*MeV;
//     if(the3DNucleus->GetMass()>30) theCutOnP = 70*MeV;
//     if(the3DNucleus->GetMass()>60) theCutOnP = 50*MeV;
//     if(the3DNucleus->GetMass()>120) theCutOnP = 45*MeV;
//
// Those thresholds look like mass numbers and are not. `G4Fancy3DNucleus::GetMass()` returns
// `myZ*m_p + (myA-myZ)*m_n - BindingEnergy()` - a mass in MeV - so it is above 120 for every
// nucleus that exists, a single neutron included, and all three branches always fire. P9 measured
// this and it is docs/RISK.md V72; `bic_params.cuh` carries `cut_on_p(nucleus_mass_mev)` with the
// mass-number reading beside it as the function nothing calls. This file uses that one, and does
// not have a second copy of the arithmetic.
//
// ## REFUSED, by name
//
//   * **the `GetPrimaryProjectile()` branch of `BuildLateParticleCollisions`.** It is non-null
//     only when a high-energy generator handed this cascade its final state - `G4TheoFSGenerator`
//     with `G4BinaryCascade` as the transport - and then the function computes an excitation and
//     returns false if it is negative, which sends `Propagate` into
//     `HighEnergyModelFSProducts`. QBBC's BIC is a standalone `G4HadronicInteraction` with no
//     primary set, so the branch is dead for every call this port serves;
//     `CascadeRefusal::high_energy_primary` names it.
#ifndef G4GPU_BIC_CASCADE_STATE_CUH
#define G4GPU_BIC_CASCADE_STATE_CUH

#include <cfloat>
#include <cmath>

#include "physics/hadronic/bic/bic_params.cuh"
#include "physics/hadronic/bic/im_r/collision_final_state.cuh"
#include "physics/hadronic/bic/kinetic_decay.cuh"
#include "physics/hadronic/bic/nucleus/nucleus_model.cuh"

namespace g4gpu::bic {

/// Which of `G4BinaryCascade`'s four `G4KineticTrackVector`s a track is currently in.
///
/// Geant4 moves a `G4KineticTrack*` from one vector to another and identity is the pointer. The
/// port keeps every track in ONE pool and moves it by changing this tag, so that identity is the
/// pool index and stays valid for the whole cascade. That is not cosmetic: `G4CollisionManager`
/// holds a primary and a target for every scheduled collision and `RemoveTracksCollisions` finds
/// them by pointer, so a list that shifted its elements on removal would invalidate every
/// scheduled collision in it.
enum TrackList : int {
  kListNone = 0,       ///< removed from the cascade altogether (Geant4 deletes the track)
  kListSecondary = 1,  ///< theSecondaryList
  kListTarget = 2,     ///< theTargetList
  kListCaptured = 3,   ///< theCapturedList
  kListFinal = 4       ///< theFinalState
};

/// One track of the cascade. `state` is `G4KineticTrack::CascadeState`, the enum in
/// `kinetic_track.cuh`, which also has the class this shape comes from and the note on why its two
/// momenta are the same four-vector (docs/RISK.md V69).
struct CascadeTrack {
  int pdg = 0;
  double pdg_mass = 0.0;     ///< the DEFINITION's mass, which the elastic final state puts out
  int charge = 0;
  int baryon = 0;
  imr::LorentzVector momentum;
  deex::Vec3d position{0.0, 0.0, 0.0};
  double formation_time = 0.0;
  int state = kUndefined;
  int nucleon_index = -1;    ///< the G4Nucleon this track is, or -1 for a cascade particle
  /// WHICH nucleus that G4Nucleon belongs to: 0 for the target, 1 for the PROJECTILE.
  ///
  /// `G4KineticTrack::theNucleon` is a pointer and does not need to say; the port indexes an
  /// array and does. It matters because `G4BinaryLightIonReaction::Interact` builds a projectile
  /// nucleus too and hands `Propagate` one track per projectile nucleon, each constructed from
  /// its own `G4Nucleon` - and the whole of `SortResult` is that the projectile nucleons NOTHING
  /// hit come back with `IsParticipant()` false and become the spectator fragment.
  int nucleon_owner = 0;
  double projectile_potential = 0.0;
  int creator_model_id = -1;
  int parent_resonance_pdg = 0;
  int parent_resonance_id = 0;
  int list = kListNone;      ///< which of the four G4KineticTrackVectors; see `TrackList`
  int final_seq = -1;        ///< position in theFinalState; see `push_final`
  bool hit = false;          ///< `G4KineticTrack::Hit()`, which marks the G4Nucleon

  /// `G4KineticTrack::GetActualMass` - `sqrt(|the4Momentum.mag2()|)`.
  __host__ __device__ double actual_mass() const {
    return std::sqrt(std::fabs(momentum.e * momentum.e - g4gpu::mag2(momentum.v)));
  }
};

/// What the cascade could not do.
struct CascadeRefusal {
  bool high_energy_primary = false;  ///< the dead BuildLateParticleCollisions branch; see above
  bool capacity = false;             ///< one of the four lists is full
  bool invalid_nucleus = false;      ///< BuildTargetList's (A,Z) throw
  bool unknown_species = false;
  bool void_nucleus = false;         ///< FillVoidNucleusProducts; see cascade_propagate.cuh
  int refused_pdg = 0;
  __host__ __device__ bool any() const {
    return high_energy_primary || capacity || invalid_nucleus || unknown_species || void_nucleus;
  }
};

/// The one pool the four lists are views of, caller-owned. A cascade on a lead nucleus holds 208
/// target nucleons plus whatever it makes, so the capacity is the caller's problem and
/// `CascadeRefusal::capacity` says when it was not enough.
///
/// The four `n_*` counters are not sizes - they are recomputed by `count_list` - they are the
/// running totals `Propagate` reads, kept so that a caller can see them without a scan.
struct CascadeLists {
  CascadeTrack* pool = nullptr;
  int n_pool = 0;
  int capacity = 0;

  /// Append a track and return its pool index, or -1 when the pool is full.
  __host__ __device__ int add(const CascadeTrack& t, int list) {
    if (n_pool >= capacity) { return -1; }
    pool[n_pool] = t;
    pool[n_pool].list = list;
    return n_pool++;
  }
  __host__ __device__ int count(int list) const {
    int n = 0;
    for (int i = 0; i < n_pool; ++i) {
      if (pool[i].list == list) { ++n; }
    }
    return n;
  }
};

/// Everything `Propagate` carries between its helpers. It is NOT `G4KineticTrack::CascadeState`,
/// which is the per-track enum in `kinetic_track.cuh` and already has that name.
struct BicCascadeState {
  CascadeLists lists;
  /// The PROJECTILE nucleus's nucleon array, for `G4BinaryLightIonReaction` only. Null for a
  /// hadron projectile, which has no nucleus of its own.
  Nucleon* projectile_nucleons = nullptr;
  /// The target nucleus's own nucleon array, so that `Hit()` can mark what Geant4 marks.
  ///
  /// `G4KineticTrack::Hit()` is `if (theNucleon) theNucleon->Hit(1, 1.)` - it sets a flag on the
  /// **G4Nucleon**, not on the track - and `IsParticipant()` reads it back through the same
  /// pointer. Two things depend on that indirection and neither is obvious. An ELASTIC product is
  /// `new G4KineticTrack(trk2)`, a COPY of the target track, so it shares the nucleon; `Hit()` is
  /// called on the ENTRANCE track AFTER the products have been made, and the product sees it
  /// anyway. And `BuildTargetList` skips a hit nucleon, so a nucleus that has been through a
  /// cascade is not the nucleus it was. A port with a `hit` flag on the track reproduces neither:
  /// MEASURED, the first product of the first case of `bic_imr_prop.csv` came back with
  /// `IsParticipant()` false where Geant4 has it true, because the copy was taken one statement
  /// before the flag was set.
  Nucleon* nucleons = nullptr;
  int initial_a = 0;
  int initial_z = 0;
  int current_a = 0;
  int current_z = 0;
  int projectile_a = 0;
  int projectile_z = 0;
  int late_a = 0;
  int late_z = 0;
  double initial_nuclear_mass = 0.0;
  double mass_in_nucleus = 0.0;
  double current_initial_energy = 0.0;
  double current_time = 0.0;
  double cut_on_p = 0.0;
  double outer_radius = 0.0;
  imr::LorentzVector initial_4mom;        ///< theInitial4Mom: (0,0,0,initial_nuclear_mass)
  imr::LorentzVector projectile_4mom;     ///< theProjectile4Momentum
  deex::Vec3d momentum_transfer{0.0, 0.0, 0.0};
  deex::Vec3d precompound_boost{0.0, 0.0, 0.0};  ///< precompoundLorentzboost, set by the below
  int n_final_pushed = 0;    ///< the next sequence number `push_final` will hand out
};

/// `G4KineticTrack::Hit()` - the flag goes on the NUCLEON when there is one, and on the track
/// either way. See the note on `BicCascadeState::nucleons` for why both.
__host__ __device__ inline Nucleon* owning_nucleons(const BicCascadeState& st, int owner) {
  return (owner == 1) ? st.projectile_nucleons : st.nucleons;
}

__host__ __device__ inline void mark_hit(BicCascadeState& st, int pool_index) {
  CascadeTrack& t = st.lists.pool[pool_index];
  t.hit = true;
  Nucleon* n = owning_nucleons(st, t.nucleon_owner);
  if (t.nucleon_index >= 0 && n != nullptr) { n[t.nucleon_index].hit = true; }
}

/// `G4KineticTrack::IsParticipant()`, and it is NOT what the name says:
///
///     G4bool G4KineticTrack::IsParticipant() const
///     { if(!theNucleon) return true;
///       return theNucleon->AreYouHit(); }
///
/// **A track with no nucleon is a participant.** Every particle the cascade MAKES - every pion,
/// every resonance product, the projectile itself - has a null `theNucleon` and so answers true;
/// the only tracks that answer false are the nucleus's own nucleons that nothing has touched.
/// Read as "not a spectator" it is right, and read as its own name it is backwards. It reaches
/// the outside world as `G4ReactionProduct::SetNewlyAdded`, which the process uses to decide
/// whether a secondary is new, so a port that took the name at face value would mark every pion
/// it produced as NOT newly added. MEASURED before the fix: the pi+ and the pi0 of case 28 of
/// `bic_imr_prop.csv` came back false where Geant4 has them true. docs/RISK.md V163.
__host__ __device__ inline bool is_participant(const BicCascadeState& st,
                                               const CascadeTrack& t) {
  if (t.nucleon_index < 0) { return true; }
  const Nucleon* n = owning_nucleons(st, t.nucleon_owner);
  return n != nullptr && n[t.nucleon_index].hit;
}

/// Move a track into theFinalState, keeping the ORDER it was pushed in.
///
/// Geant4's `theFinalState` is a vector and `ProductsAddFinalState` walks it front to back, so
/// the product order is the order tracks left the cascade - which is not the order they were
/// created in. The port's lists are tags on one pool, and a tag has no order, so the sequence
/// number is what carries it. MEASURED without it: case 28 of `bic_imr_prop.csv` put its proton
/// where Geant4 has its pi0, and every product after the first was a different particle.
__host__ __device__ inline void push_final(BicCascadeState& st, int pool_index) {
  st.lists.pool[pool_index].list = kListFinal;
  st.lists.pool[pool_index].final_seq = st.n_final_pushed++;
}

/// `G4BinaryCascade::GetIonMass(Z, A)` and its three fallbacks.
///
/// The first is `G4IonTable::GetIonMass(Z,A)`, which is `G4NucleiProperties::GetNuclearMass(A,Z)`
/// - P3's `deex::nuclear_mass`, already validated. The second is the one worth naming: when the
/// CHARGE EXCEEDS the mass number, which "will happen for light nuclei with pions involved", the
/// mass asked for is `GetIonMass(A,A)` - a nucleus of A protons. The third is `A` neutron masses
/// for anything neutral, and an empty nucleus is zero.
__host__ __device__ inline double get_ion_mass(int z, int a, double neutron_mass) {
  if (z > 0 && a >= z) { return deex::nuclear_mass(a, z); }
  if (a > 0 && z > 0) { return deex::nuclear_mass(a, a); }
  if (a >= 0 && z <= 0) { return a * neutron_mass; }
  return 0.0;
}


/// `G4BinaryCascade::BuildTargetList`.
///
/// Every nucleon the higher-energy model did NOT hit becomes a target track, put ON MASS SHELL -
/// `mom.setE(sqrt(p^2 + m_PDG^2))` - with the comment "the potential inside the nucleus is taken
/// into account, and nucleons are on mass shell". The nucleus model's own energies carry the
/// potential; this discards it and keeps the momentum.
__host__ __device__ inline void build_target_list(Nucleus3D& nucleus, BicCascadeState& st,
                                                  double proton_mass, double neutron_mass,
                                                  CascadeRefusal& ref) {
  if (!nucleus.start_loop()) { return; }
  st.initial_z = nucleus.charge();
  st.initial_a = nucleus.mass_number();
  st.initial_nuclear_mass = get_ion_mass(st.initial_z, st.initial_a, neutron_mass);
  st.initial_4mom = imr::LorentzVector(deex::Vec3d{0.0, 0.0, 0.0}, st.initial_nuclear_mass);
  st.current_a = 0;
  st.current_z = 0;
  Nucleon* nucleon = nullptr;
  while ((nucleon = nucleus.next_nucleon()) != nullptr) {
    if (nucleon->hit) { continue; }
    CascadeTrack t;
    t.pdg = nucleon->pdg();
    t.pdg_mass = nucleon->pdg_mass();
    t.charge = nucleon->charge();
    t.baryon = 1;
    t.position = nucleon->position;
    t.momentum = imr::LorentzVector(
        nucleon->momentum.v,
        std::sqrt(g4gpu::mag2(nucleon->momentum.v) + t.pdg_mass * t.pdg_mass));
    t.state = kInside;
    t.nucleon_index = nucleus.current - 1;
    if (st.lists.add(t, kListTarget) < 0) {
      ref.capacity = true;
      return;
    }
    ++st.current_a;
    if (t.charge > 0) { ++st.current_z; }
  }
  if (st.current_z > 0) {
    st.mass_in_nucleus = get_ion_mass(st.current_z, st.current_a, neutron_mass);
  } else if (st.current_z == 0 && st.current_a >= 1) {
    st.mass_in_nucleus = st.current_a * neutron_mass;
  } else {
    ref.invalid_nucleus = true;
    return;
  }
  st.current_initial_energy = st.initial_4mom.e + st.projectile_4mom.e;
  (void)proton_mass;
}

/// `G4BinaryCascade::BuildLateParticleCollisions`, minus the dead high-energy branch.
///
/// The formation times are shifted so the EARLIEST is zero, and a track whose state is
/// `undefined` - which only a high-energy generator's products are - goes to
/// `FindLateParticleCollision` instead of into the cascade. A standalone BIC's one secondary is
/// the projectile, whose state `ApplyYourself` set to `outside`, so it takes the other branch.
__host__ __device__ inline bool build_late_particle_collisions(CascadeTrack* secondaries, int n,
                                                               BicCascadeState& st,
                                                               CascadeRefusal& ref) {
  st.late_a = 0;
  st.late_z = 0;
  st.projectile_a = 0;
  st.projectile_z = 0;
  double starting_time = DBL_MAX;
  for (int i = 0; i < n; ++i) {
    if (secondaries[i].formation_time < starting_time) {
      starting_time = secondaries[i].formation_time;
    }
  }
  for (int i = 0; i < n; ++i) {
    secondaries[i].formation_time -= starting_time;
    if (secondaries[i].state == kUndefined) {
      st.late_a += secondaries[i].baryon;
      st.late_z += secondaries[i].charge;
      // The track stays where the caller put it; `find_late_particle_collision` schedules it.
      continue;
    }
    if (st.lists.add(secondaries[i], kListSecondary) < 0) {
      ref.capacity = true;
      return false;
    }
    st.projectile_4mom = st.projectile_4mom + secondaries[i].momentum;
    st.projectile_a += secondaries[i].baryon;
    st.projectile_z += secondaries[i].charge;
  }
  // `GetPrimaryProjectile()` is null for a standalone cascade, so `success = true`.
  return true;
}

/// `G4BinaryCascade::GetFinal4Momentum` - what is left when everything that has gone is taken
/// away, and the superluminal guard that zeroes it.
///
/// The guard is `final4Momentum.e() > 0 && (vect()/e()).mag() > 1.0 && currentA > 0`, i.e. a
/// remnant whose velocity would exceed c. Geant4 prints under a debug flag and returns a zero
/// four-vector, which then makes the excitation energy exactly minus the ion mass and sends
/// `Propagate` into `CorrectFinalPandE`.
__host__ __device__ inline imr::LorentzVector get_final_4momentum(const BicCascadeState& st) {
  imr::LorentzVector f = st.initial_4mom + st.projectile_4mom;
  for (int i = 0; i < st.lists.n_pool; ++i) {
    if (st.lists.pool[i].list == kListFinal) { f = f - st.lists.pool[i].momentum; }
  }
  if (f.e > 0.0 && st.current_a > 0) {
    const deex::Vec3d beta = (1.0 / f.e) * f.v;
    if (std::sqrt(g4gpu::mag2(beta)) > 1.0) {
      return imr::LorentzVector(deex::Vec3d{0.0, 0.0, 0.0}, 0.0);
    }
  }
  return f;
}

/// `G4BinaryCascade::GetFinalNucleusMomentum` - the remnant, boosted into the frame where its
/// momentum equals the captured nucleons', and the boost kept for the precompound products.
///
/// `boost = (NucleusMomentum.vect() - CapturedMomentum.vect())/NucleusMomentum.e()` and the
/// rotation applied is `G4LorentzRotation(-boost)`, so the sign stored in
/// `precompoundLorentzboost` is the one that takes a precompound product BACK.
__host__ __device__ inline imr::LorentzVector get_final_nucleus_momentum(BicCascadeState& st) {
  imr::LorentzVector captured(deex::Vec3d{0.0, 0.0, 0.0}, 0.0);
  for (int i = 0; i < st.lists.n_pool; ++i) {
    if (st.lists.pool[i].list == kListCaptured) {
      captured = captured + st.lists.pool[i].momentum;
    }
  }
  imr::LorentzVector nucleus = get_final_4momentum(st);
  if (nucleus.e > 0.0) {
    deex::Vec3d boost = (1.0 / nucleus.e) * (nucleus.v - captured.v);
    if (g4gpu::mag2(boost) > 1.0) {
      boost = deex::Vec3d{0.0, 0.0, 0.0};
      nucleus = imr::LorentzVector(deex::Vec3d{0.0, 0.0, 0.0}, 0.0);
    }
    st.precompound_boost = boost;
    nucleus = imr::LorentzRotation::from_boost(-1.0 * boost) * nucleus;
  }
  return nucleus;
}

/// `G4BinaryCascade::GetExcitationEnergy`.
///
/// The `currentZ == 0` arm subtracts `3 MeV per nucleon` from the remnant's invariant mass for a
/// neutral remnant of more than one nucleon - a made-up binding, not a table lookup - and returns
/// zero for an invalid (A, Z) rather than throwing.
__host__ __device__ inline double get_excitation_energy(BicCascadeState& st, double neutron_mass) {
  double nucleus_mass = 0.0;
  if (st.current_z > 0) {
    nucleus_mass = get_ion_mass(st.current_z, st.current_a, neutron_mass);
  } else if (st.current_z == 0) {
    if (st.current_a == 1) {
      nucleus_mass = neutron_mass;
    } else {
      nucleus_mass = get_final_nucleus_momentum(st).mag() -
                     3.0 * u::MeV<double>() * static_cast<double>(st.current_a);
    }
  } else {
    return 0.0;
  }
  return get_final_nucleus_momentum(st).mag() - nucleus_mass;
}

}  // namespace g4gpu::bic

#endif
