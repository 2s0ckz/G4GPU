// G4Step, G4StepPoint, G4Track and the touchable, for a G4UserSteppingAction.
//
// READ THIS BEFORE WRITING A STEPPING ACTION. A step here is not one step.
//
// Geant4 calls UserSteppingAction once per step of every track, on the CPU, and a stepping
// action can therefore see each step individually. This transport takes about 370 nanoseconds
// per event on the whole GPU, with tens of thousands of tracks in flight at once; a host
// virtual call per step would cost more than the physics and would serialise the device.
//
// So the step handed to a stepping action here is an *aggregate*: one per event, per volume
// that has a sensitive detector, carrying the totals for that event in that volume. The
// numbers a stepping action reads off it are:
//
//   GetTotalEnergyDeposit()      the energy deposited in this volume during this event
//   GetStepLength()              the track length in this volume during this event, if the
//                                detector has a G4PSTrackLength primitive; 0 otherwise
//   GetNumberOfAggregatedSteps() how many real steps were summed, if the detector has a
//                                G4PSNofStep primitive; 0 otherwise
//
// The consequence is worth being blunt about. A stepping action that *adds* what it reads -
// which is what B1's does, and what almost every dose or energy-deposit example does - gets
// exactly the right answer, because the sum over an event of the per-step deposits is the
// event's deposit. A stepping action that does anything else - takes a maximum, tests a
// threshold per step, looks at where within the volume a step happened, counts steps without
// asking for a G4PSNofStep - does not, and no amount of care inside the callback will fix it.
// IsAggregated() returns true so that such an action can detect the situation and say so.
//
// The device-side alternative for per-step quantities is a primitive scorer: G4PSTrackLength,
// G4PSNofStep, G4PSCellFlux and G4PSDoseDeposit are computed step by step on the GPU where
// the steps are, and read back per event. Anything a stepping action would compute by
// accumulating over steps belongs there.
#pragma once
#include "g4/G4Material.hh"
#include "g4/G4PVPlacement.hh"
#include "g4/G4ThreeVector.hh"
#include "g4/G4Types.hh"

class G4ParticleDefinition;

/// What Geant4's touchable does for a stepping action: name the volume a step was in.
///
/// Geant4's touchable also carries the placement history that distinguishes two copies of the
/// same physical volume. There is no history here - the scene is flattened to absolute
/// transforms before it reaches the device - so a touchable is one placement, and GetVolume()
/// ignores its depth argument.
class G4VTouchable {
 public:
  explicit G4VTouchable(G4PVPlacement* pv = nullptr) : pv_(pv) {}

  G4VPhysicalVolume* GetVolume(G4int depth = 0) const {
    (void)depth;
    return pv_;
  }
  G4int GetCopyNumber(G4int depth = 0) const {
    (void)depth;
    return (pv_ != nullptr) ? pv_->GetCopyNo() : 0;
  }
  const G4ThreeVector& GetTranslation(G4int depth = 0) const {
    (void)depth;
    static const G4ThreeVector zero;
    return (pv_ != nullptr) ? pv_->GetTranslation() : zero;
  }

 private:
  G4PVPlacement* pv_;
};

/// Geant4's G4TouchableHandle is a reference-counted handle; the arrow operator is what
/// examples use it through, so a pointer serves the same code.
using G4TouchableHandle = const G4VTouchable*;

class G4StepPoint {
 public:
  const G4VTouchable* GetTouchableHandle() const { return &touchable_; }
  G4VPhysicalVolume* GetPhysicalVolume() const { return touchable_.GetVolume(); }
  G4Material* GetMaterial() const { return material_; }
  const G4ThreeVector& GetPosition() const { return position_; }
  G4double GetKineticEnergy() const { return kinetic_energy_; }
  G4double GetGlobalTime() const { return global_time_; }

  void SetTouchable(G4PVPlacement* pv) { touchable_ = G4VTouchable(pv); }
  void SetMaterial(G4Material* m) { material_ = m; }
  void SetPosition(const G4ThreeVector& p) { position_ = p; }
  void SetKineticEnergy(G4double e) { kinetic_energy_ = e; }
  void SetGlobalTime(G4double t) { global_time_ = t; }

 private:
  G4VTouchable touchable_;
  G4Material* material_ = nullptr;
  G4ThreeVector position_;
  G4double kinetic_energy_ = 0;
  G4double global_time_ = 0;
};

/// The track a step belongs to. For an aggregated step this is the event's primary, and
/// GetTrackID() is -1 to say that it stands for every track in the event rather than one.
class G4Track {
 public:
  G4int GetTrackID() const { return track_id_; }
  G4int GetParentID() const { return parent_id_; }
  const G4ParticleDefinition* GetDefinition() const { return definition_; }
  G4double GetKineticEnergy() const { return kinetic_energy_; }
  const G4ThreeVector& GetPosition() const { return position_; }
  const G4ThreeVector& GetMomentumDirection() const { return direction_; }
  G4VPhysicalVolume* GetVolume() const { return volume_; }

  void SetTrackID(G4int id) { track_id_ = id; }
  void SetParentID(G4int id) { parent_id_ = id; }
  void SetDefinition(const G4ParticleDefinition* d) { definition_ = d; }
  void SetKineticEnergy(G4double e) { kinetic_energy_ = e; }
  void SetPosition(const G4ThreeVector& p) { position_ = p; }
  void SetMomentumDirection(const G4ThreeVector& d) { direction_ = d; }
  void SetVolume(G4VPhysicalVolume* v) { volume_ = v; }

 private:
  G4int track_id_ = -1;
  G4int parent_id_ = 0;
  const G4ParticleDefinition* definition_ = nullptr;
  G4double kinetic_energy_ = 0;
  G4ThreeVector position_;
  G4ThreeVector direction_;
  G4VPhysicalVolume* volume_ = nullptr;
};

class G4Step {
 public:
  G4StepPoint* GetPreStepPoint() { return &pre_; }
  const G4StepPoint* GetPreStepPoint() const { return &pre_; }
  G4StepPoint* GetPostStepPoint() { return &post_; }
  const G4StepPoint* GetPostStepPoint() const { return &post_; }
  G4Track* GetTrack() const { return track_; }

  G4double GetTotalEnergyDeposit() const { return edep_; }
  G4double GetStepLength() const { return step_length_; }
  G4double GetDeltaTime() const { return 0; }

  /// How many real device-side steps this one stands for, or 0 if the detector has no
  /// G4PSNofStep primitive to count them. See the note at the top of this file.
  G4int GetNumberOfAggregatedSteps() const { return n_aggregated_; }

  /// Always true here. A stepping action that needs individual steps should test this and
  /// say so rather than quietly producing a number that means something else.
  G4bool IsAggregated() const { return true; }

  void SetTotalEnergyDeposit(G4double e) { edep_ = e; }
  void SetStepLength(G4double l) { step_length_ = l; }
  void SetNumberOfAggregatedSteps(G4int n) { n_aggregated_ = n; }
  void SetTrack(G4Track* t) { track_ = t; }

 private:
  G4StepPoint pre_;
  G4StepPoint post_;
  G4Track* track_ = nullptr;
  G4double edep_ = 0;
  G4double step_length_ = 0;
  G4int n_aggregated_ = 0;
};
