// G4KineticTrack - a particle inside the cascade, with its position relative to the nucleus.
//
// Transcribed from G4KineticTrack.{hh,cc} (hadronic/util) in 11.1.1: the two constructors the
// binary cascade uses, the four-momentum accessors, the cascade state, and `IsParticipant`.
//
// **The dual momentum is inert, and that is a measured fact rather than a reading.**
// G4KineticTrack carries THREE four-vectors - `the4Momentum` (returned by
// `GetTrackingMomentum`), `theFermi3Momentum`, and `theTotal4Momentum` (returned by
// `Get4Momentum`) - tied together by `SetTrackingMomentum`:
//
//     the4Momentum = aMomentum;
//     theTotal4Momentum = the4Momentum + theFermi3Momentum;
//     theTotal4Momentum.setE(sqrt(aMomentum.mag2() + theTotal4Momentum.vect().mag2()));
//
// so the two differ exactly when `theFermi3Momentum` is non-zero. It is assigned a non-zero
// value in ONE place in the whole class - the `G4Nucleon*` constructor's initialiser list,
// `theFermi3Momentum(nucleon->GetMomentum())` at G4KineticTrack.cc:430 - and that constructor's
// body is
//
//     theFermi3Momentum.setE(0);
//     Set4Momentum(a4Momentum);
//
// where `Set4Momentum` ends with `theFermi3Momentum = G4LorentzVector(0)`. The assignment is
// discarded two lines after it is made, and no other code writes the member. So for every
// kinetic track in every event, `Get4Momentum()` and `GetTrackingMomentum()` return the same
// four-vector, up to the `sqrt(m^2 + p^2)` round trip `SetTrackingMomentum` puts the energy
// through. docs/RISK.md V69.
//
// Both are kept here, with the round trip, because BIC reads one in `Capture` and the other in
// `G4RKPropagation::Transport` and a port that collapsed them would lose the ulp - and because
// a Geant4 release that removes the `Set4Momentum` call would bring the Fermi momentum to life
// in the propagator without touching a line of the propagator.
//
// **REFUSED, by name:** the resonance machinery. The `(definition, time, position, momentum)`
// constructor's body builds `theActualWidth[nChannels]` by integrating a Breit-Wigner over each
// decay channel's daughter masses (`G4SampleResonance`, `G4Integrator`, `IntegrateCMMomentum`,
// `IntegrandFunction1..4`), and `SampleResidualLifetime` and `Decay()` are what read it. Those
// belong with `G4BCDecay`, which this package refuses, so `KineticTrack` carries
// `n_channels = 0` and `KtRefusal::resonance_widths` is set for a projectile whose definition
// has a decay table. A nucleon has none; a pion and every resonance do.
//
// **REFUSED, by name:** the kaon0/anti-kaon0 coin flip at the top of the same constructor,
// which substitutes K0S or K0L with probability 1/2 each and consumes one uniform doing it.
// K0 is P1's refused species and no QBBC channel this package reaches produces one.
#ifndef G4GPU_BIC_KINETIC_TRACK_CUH
#define G4GPU_BIC_KINETIC_TRACK_CUH

#include <cmath>

#include "core/units.cuh"
#include "core/vec3.cuh"
#include "physics/hadronic/bic/nucleus/nucleon.cuh"

namespace g4gpu::bic {

/// `G4KineticTrack::CascadeState` (G4KineticTrack.hh:117), in Geant4's order - the integer
/// values matter because `SelectFromKTV` compares them and because the oracle dumps them.
enum CascadeState : int {
  kUndefined = 0,
  kOutside = 1,
  kGoingIn = 2,
  kInside = 3,
  kGoingOut = 4,
  kGoneOut = 5,
  kCaptured = 6,
  kMissNucleus = 7
};

/// What a track construction refused.
struct KtRefusal {
  bool resonance_widths = false;  ///< the definition has a decay table; see the file header
  bool neutral_kaon = false;      ///< the K0 -> K0S/K0L coin flip
  int refused_pdg = 0;
};

/// G4KineticTrack, restricted to what G4BinaryCascade, G4RKPropagation and
/// G4BinaryLightIonReaction read of it.
///
/// `nucleon_index` is Geant4's `theNucleon` pointer as an index into the target nucleus's
/// nucleon array: `Hit()` sets that nucleon's hit flag and `IsParticipant()` reads it, with
/// "no nucleon" meaning "is a participant" - so a projectile track, which has no nucleon, is a
/// participant from the moment it is made, and a target nucleon is one only once it is hit.
/// That predicate is what `ProductsAddFinalState` turns into `SetNewlyAdded`, which is what
/// `G4BinaryLightIonReaction::SortResult` splits cascaders from spectators on. An index of -1
/// is the null pointer.
struct KineticTrack {
  int pdg = 0;
  double pdg_mass = 0.0;
  int charge = 0;         ///< G4lrint(GetPDGCharge()/eplus)
  int baryon_number = 0;

  double formation_time = 0.0;
  Vec3d position{0.0, 0.0, 0.0};

  LorentzVector the4_momentum;     ///< G4KineticTrack::the4Momentum, GetTrackingMomentum()
  LorentzVector fermi3_momentum;   ///< theFermi3Momentum - always zero, see the file header
  LorentzVector total4_momentum;   ///< theTotal4Momentum, Get4Momentum()

  int nucleon_index = -1;          ///< theNucleon, as an index; -1 is null
  int state = kUndefined;
  double projectile_potential = 0.0;
  int creator_model_id = -1;
  int n_channels = 0;              ///< always 0 here; see the refusal in the file header

  __host__ __device__ const LorentzVector& tracking_momentum() const { return the4_momentum; }
  __host__ __device__ const LorentzVector& momentum4() const { return total4_momentum; }

  /// `G4KineticTrack::GetActualMass` - `sqrt(|the4Momentum.mag2()|)`. Note the absolute value:
  /// a spacelike tracking momentum gives a positive mass here rather than the signed root
  /// `LorentzVector::mag()` returns, and every energy test in `G4RKPropagation` compares
  /// against THIS.
  __host__ __device__ double actual_mass() const {
    const double m2 = the4_momentum.e * the4_momentum.e - g4gpu::mag2(the4_momentum.v);
    return std::sqrt(std::abs(m2));
  }

  /// G4KineticTrack::Set4Momentum - and note that it zeroes `theFermi3Momentum`, which is the
  /// line that makes the member inert.
  __host__ __device__ void set4_momentum(const LorentzVector& p4) {
    total4_momentum = p4;
    the4_momentum = total4_momentum;
    fermi3_momentum = LorentzVector();
  }

  /// G4KineticTrack::SetTrackingMomentum.
  __host__ __device__ void set_tracking_momentum(const LorentzVector& p4) {
    the4_momentum = p4;
    total4_momentum = the4_momentum + fermi3_momentum;
    const double mass2 = p4.e * p4.e - g4gpu::mag2(p4.v);
    const double p2 = g4gpu::mag2(total4_momentum.v);
    total4_momentum.e = std::sqrt(mass2 + p2);
  }

  /// G4KineticTrack::Update4Momentum(G4double) - change the energy at constant mass.
  ///
  /// The `else` branch assigns to the local parameter `aEnergy` and leaves `newP` at zero, so a
  /// requested energy below the mass produces a particle AT REST with energy `sqrt(mass2)` -
  /// not a failure and not the requested energy. Both callers in BIC test the energy against
  /// `GetActualMass()` first, so the branch should be unreachable from them; it is transcribed
  /// because `mass2` here is `theTotal4Momentum.mag2()` while their test uses
  /// `the4Momentum.mag2()`, and those two differ by the round trip above.
  __host__ __device__ void update4_momentum(double energy) {
    double new_p = 0.0;
    const double mass2 =
        total4_momentum.e * total4_momentum.e - g4gpu::mag2(total4_momentum.v);
    if (energy * energy > mass2) {
      new_p = std::sqrt(energy * energy - mass2);
    } else {
      energy = std::sqrt(mass2);
    }
    set4_momentum(LorentzVector(new_p * g4gpu::normalize(the4_momentum.v), energy));
  }

  /// G4KineticTrack::Update4Momentum(const G4ThreeVector&).
  __host__ __device__ void update4_momentum(const Vec3d& p3) {
    const double mass2 =
        total4_momentum.e * total4_momentum.e - g4gpu::mag2(total4_momentum.v);
    const double new_e = std::sqrt(mass2 + g4gpu::mag2(p3));
    set4_momentum(LorentzVector(p3, new_e));
  }

  /// G4KineticTrack::UpdateTrackingMomentum(G4double).
  __host__ __device__ void update_tracking_momentum(double energy) {
    double new_p = 0.0;
    const double mass2 =
        total4_momentum.e * total4_momentum.e - g4gpu::mag2(total4_momentum.v);
    if (energy * energy > mass2) {
      new_p = std::sqrt(energy * energy - mass2);
    } else {
      energy = std::sqrt(mass2);
    }
    set_tracking_momentum(LorentzVector(new_p * g4gpu::normalize(the4_momentum.v), energy));
  }

  /// G4KineticTrack::UpdateTrackingMomentum(const G4ThreeVector&).
  __host__ __device__ void update_tracking_momentum(const Vec3d& p3) {
    const double mass2 =
        total4_momentum.e * total4_momentum.e - g4gpu::mag2(total4_momentum.v);
    const double new_e = std::sqrt(mass2 + g4gpu::mag2(p3));
    set_tracking_momentum(LorentzVector(p3, new_e));
  }
};

/// `G4KineticTrack(const G4ParticleDefinition*, formationTime, position, 4momentum)`, with the
/// resonance-width block and the K0 coin flip refused (see the file header).
///
/// `pdg_mass`, `charge` and `baryon_number` are what the definition would have supplied; they
/// are parameters here because there is no particle table in a kernel.
__host__ __device__ inline KineticTrack make_kinetic_track(int pdg, double pdg_mass, int charge,
                                                           int baryon_number,
                                                           double formation_time,
                                                           const Vec3d& position,
                                                           const LorentzVector& p4,
                                                           bool has_decay_table,
                                                           KtRefusal& ref) {
  KineticTrack kt;
  kt.pdg = pdg;
  kt.pdg_mass = pdg_mass;
  kt.charge = charge;
  kt.baryon_number = baryon_number;
  kt.formation_time = formation_time;
  kt.position = position;
  kt.the4_momentum = p4;
  kt.total4_momentum = p4;
  kt.fermi3_momentum = LorentzVector();
  kt.state = kUndefined;
  if (pdg == 311 || pdg == -311) {
    ref.neutral_kaon = true;
    ref.refused_pdg = pdg;
  }
  if (has_decay_table) {
    ref.resonance_widths = true;
    ref.refused_pdg = pdg;
  }
  return kt;
}

/// `G4KineticTrack(G4Nucleon*, position, 4momentum)` - the one BIC's `BuildTargetList` and
/// `G4BinaryLightIonReaction::Interact` use.
///
/// `nChannels` is 0 and `theActualMass` is set to the nucleon's PDG mass by the initialiser
/// list, which no accessor ever reads back (`GetActualMass()` recomputes from the momentum), so
/// the member is as inert as `theFermi3Momentum`. The two lines that matter are in the body:
/// the Fermi momentum is loaded and then thrown away. Written out here, in that order, so the
/// discard is visible rather than assumed.
__host__ __device__ inline KineticTrack make_nucleon_kinetic_track(
    const Nucleon& nucleon, int nucleon_index, const Vec3d& position, const LorentzVector& p4) {
  KineticTrack kt;
  kt.pdg = nucleon.pdg();
  kt.pdg_mass = nucleon.pdg_mass();
  kt.charge = nucleon.charge();
  kt.baryon_number = (nucleon.type == kLambda) ? 1 : 1;
  kt.formation_time = 0.0;
  kt.position = position;
  kt.the4_momentum = p4;
  kt.nucleon_index = nucleon_index;
  kt.state = kUndefined;
  kt.n_channels = 0;
  // theFermi3Momentum(nucleon->GetMomentum()); theFermi3Momentum.setE(0);
  kt.fermi3_momentum = nucleon.momentum;
  kt.fermi3_momentum.e = 0.0;
  // ... and Set4Momentum throws it away.
  kt.set4_momentum(p4);
  return kt;
}

}  // namespace g4gpu::bic

#endif
