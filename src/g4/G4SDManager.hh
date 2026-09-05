// Sensitive detectors and primitive scorers, as Geant4 arranges them.
//
// In Geant4 a logical volume becomes sensitive by being given a G4VSensitiveDetector, and the
// common case is a G4MultiFunctionalDetector carrying one or more G4VPrimitiveScorer objects
// (G4PSDoseDeposit, G4PSEnergyDeposit, ...). That structure is kept here, because it is what a
// veteran user reaches for and because it already expresses what the GUI needs: a named
// scorer, of a chosen quantity, attached to chosen volumes.
//
// What reaches the device is much simpler: each volume carries a `score_index`, and the
// stepping kernels add into that slot. The manager's job is to turn detector objects into
// those indices and to hold the results afterwards.
#pragma once
#include <cstdio>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "g4/G4LogicalVolume.hh"
#include "g4/G4SystemOfUnits.hh"
#include "g4/G4Types.hh"

/// What a scorer accumulates.
enum class G4ScoreQuantity {
  kEnergyDeposit,  ///< MeV summed in the volume
  kDoseDeposit,    ///< Gy: energy deposit divided by the volume's mass
  kFlatSurfaceFlux,
  kTrackLength,    ///< mm of track summed over the volume
  kCellFlux,       ///< track length / volume, i.e. fluence
  kNofStep,
};

/// Base of the primitive scorers. Geant4's G4VPrimitiveScorer has a much larger interface;
/// what matters here is the name and the quantity, because the accumulation itself happens in
/// the stepping kernels rather than in a Fill() callback.
class G4VPrimitiveScorer {
 public:
  G4VPrimitiveScorer(const G4String& name, G4ScoreQuantity q) : name_(name), quantity_(q) {}
  virtual ~G4VPrimitiveScorer() = default;

  const G4String& GetName() const { return name_; }
  G4ScoreQuantity GetQuantity() const { return quantity_; }

  /// Whether this scorer wants a value per voxel rather than one per volume. Only
  /// G4PSEnergyDeposit3D says yes, and only a voxel volume can honour it.
  virtual G4bool IsPerVoxel() const { return false; }

  /// Index into the run's score array, assigned when the detector is flattened.
  G4int device_index = -1;
  /// Accumulated value after a run, in the quantity's own unit.
  G4double total = 0;
  /// Sum of squares over events, for the statistical uncertainty.
  G4double total_sq = 0;

  /// Whether Accept() below does anything, and so whether the per-event path has to run.
  ///
  /// False for every stock scorer, and that is load-bearing rather than an optimisation. A
  /// stock scorer's total comes straight from the device's per-event array, summed in a fixed
  /// order - and those totals are what example B1's 0.09-sigma agreement with Geant4 is
  /// measured on. Routing them through a host-side loop, even one that changes nothing, would
  /// re-associate the additions and move the last digits of a validated number. So the filtered
  /// path is a separate path, taken only by scorers that say they need it.
  virtual G4bool IsFiltered() const { return false; }

  /// A per-event hook: whether this event's contribution counts, and how much of it.
  ///
  /// READ THIS BEFORE OVERRIDING. The value handed over is **one event's total in this
  /// scorer's volumes**, not one step's deposit. Individual steps happen inside a device
  /// kernel tens of thousands at a time and are never on the host; the same aggregation, and
  /// the same consequences, as the step given to a G4UserSteppingAction - the whole argument
  /// is at the top of g4/G4Step.hh.
  ///
  /// So a threshold written here is a threshold on the *event*, not on the step. That is a
  /// meaningful thing to want - it is a coincidence gate, a detector trigger - but it is not
  /// the same thing as "ignore deposits below 10 keV", and an override that means the second
  /// will quietly compute the first.
  ///
  /// Return false to drop the event. Scale @p value to weight it.
  virtual G4bool Accept(G4double& value, G4int event_id) {
    (void)value;
    (void)event_id;
    return true;
  }

  /// Accumulated through Accept(), when IsFiltered() is true. Ignored otherwise.
  G4double filtered_total = 0;
  G4double filtered_total_sq = 0;

 private:
  G4String name_;
  G4ScoreQuantity quantity_;
};

class G4PSEnergyDeposit : public G4VPrimitiveScorer {
 public:
  explicit G4PSEnergyDeposit(const G4String& name)
      : G4VPrimitiveScorer(name, G4ScoreQuantity::kEnergyDeposit) {}
};

/// Per-cell energy deposit in a voxel volume, as G4PSEnergyDeposit3D scores it.
///
/// Geant4's 3D scorers key a hits map by the copy number of the replica the step was in, which
/// for a parameterised voxel phantom is the cell. The same thing happens here, with the cell
/// index in the voxel store standing in for the copy number - it is the same number, computed
/// the same way from the same grid.
///
/// Attaching one changes the *stepping* of the volume it is attached to: steps then end at
/// every cell boundary rather than only where the material changes, so a deposit belongs to
/// exactly one cell. Geant4 pays the same cost, for the same reason, through the boundary at
/// every replica.
///
/// `total` and `total_sq` still hold the volume's sum, so the same scorer reports both the
/// distribution and the number it would have reported without this.
class G4PSEnergyDeposit3D : public G4VPrimitiveScorer {
 public:
  explicit G4PSEnergyDeposit3D(const G4String& name)
      : G4VPrimitiveScorer(name, G4ScoreQuantity::kEnergyDeposit) {}

  G4bool IsPerVoxel() const override { return true; }

  /// Energy deposit per cell in MeV, indexed by the cell's index in the voxel store. Filled
  /// by the run manager after a run; empty before one.
  std::vector<G4double> cells;
};

class G4PSDoseDeposit : public G4VPrimitiveScorer {
 public:
  explicit G4PSDoseDeposit(const G4String& name)
      : G4VPrimitiveScorer(name, G4ScoreQuantity::kDoseDeposit) {}
};

class G4PSTrackLength : public G4VPrimitiveScorer {
 public:
  explicit G4PSTrackLength(const G4String& name)
      : G4VPrimitiveScorer(name, G4ScoreQuantity::kTrackLength) {}
};

class G4PSCellFlux : public G4VPrimitiveScorer {
 public:
  explicit G4PSCellFlux(const G4String& name)
      : G4VPrimitiveScorer(name, G4ScoreQuantity::kCellFlux) {}
};

class G4PSNofStep : public G4VPrimitiveScorer {
 public:
  explicit G4PSNofStep(const G4String& name)
      : G4VPrimitiveScorer(name, G4ScoreQuantity::kNofStep) {}
};

class G4VSensitiveDetector {
 public:
  explicit G4VSensitiveDetector(const G4String& name) : name_(name) {}
  virtual ~G4VSensitiveDetector() = default;
  const G4String& GetName() const { return name_; }

  virtual std::vector<G4VPrimitiveScorer*> GetScorers() const { return {}; }

 private:
  G4String name_;
};

/// A detector that holds primitive scorers, exactly as G4MultiFunctionalDetector does.
class G4MultiFunctionalDetector : public G4VSensitiveDetector {
 public:
  explicit G4MultiFunctionalDetector(const G4String& name) : G4VSensitiveDetector(name) {}

  void RegisterPrimitive(G4VPrimitiveScorer* s) { scorers_.push_back(s); }
  void RemovePrimitive(G4VPrimitiveScorer* s) {
    for (std::size_t i = 0; i < scorers_.size(); ++i) {
      if (scorers_[i] == s) {
        scorers_.erase(scorers_.begin() + static_cast<long>(i));
        return;
      }
    }
  }
  std::vector<G4VPrimitiveScorer*> GetScorers() const override { return scorers_; }

 private:
  std::vector<G4VPrimitiveScorer*> scorers_;
};

class G4SDManager {
 public:
  static G4SDManager* GetSDMpointer() {
    static G4SDManager inst;
    return &inst;
  }

  /// Forgets every detector and attachment.
  ///
  /// The GUI rebuilds the whole G4 graph on every edit, and clears G4PVPlacement's,
  /// G4LogicalVolume's, G4Material's and G4Element's registries to do it - because they are
  /// global and additive, and a rebuild that did not clear them would keep the previous
  /// version's volumes in the scene. This one was missed.
  ///
  /// Keeping it was not merely a leak. `attached_` is keyed by G4LogicalVolume*, and a freed
  /// volume's address is reused by the next allocation - so a *fresh* logical volume could
  /// find a *stale* detector under its own address, and with it a stale device_index left over
  /// from a run with a different number of scorers. That index is what the kernels write
  /// through. The symptom was every score in the scene reading exactly zero, transport and
  /// geometry both provably correct, after adding a second scorer.
  void Reset() {
    for (G4VSensitiveDetector* sd : detectors_) {
      for (G4VPrimitiveScorer* ps : sd->GetScorers()) { ps->device_index = -1; }
    }
    detectors_.clear();
    scorers_.clear();
    attached_.clear();
  }

  void AddNewDetector(G4VSensitiveDetector* sd) { detectors_.push_back(sd); }

  G4VSensitiveDetector* FindSensitiveDetector(const G4String& name) const {
    for (G4VSensitiveDetector* sd : detectors_) {
      if (sd->GetName() == name) { return sd; }
    }
    return nullptr;
  }

  /// Attaches @p sd to @p lv. Geant4 spells this
  /// G4VUserDetectorConstruction::SetSensitiveDetector; both routes end up here.
  void Attach(G4LogicalVolume* lv, G4VSensitiveDetector* sd) {
    if (lv == nullptr || sd == nullptr) { return; }
    attached_[lv] = sd;
    lv->SetSensitive(true);
  }

  G4VSensitiveDetector* DetectorFor(G4LogicalVolume* lv) const {
    const auto it = attached_.find(lv);
    return (it == attached_.end()) ? nullptr : it->second;
  }

  const std::vector<G4VSensitiveDetector*>& Detectors() const { return detectors_; }

  /// Every scorer of every attached detector, in the order their indices were assigned.
  const std::vector<G4VPrimitiveScorer*>& Scorers() const { return scorers_; }

  /// Assigns device indices. Called once by the flattener; a scorer shared between volumes
  /// keeps one index, so the volumes sum into the same slot - which is what "one scorer over
  /// several volumes" has to mean.
  ///
  /// The walk is over `detectors_`, which is in registration order, and *not* over the
  /// attachment map, which is keyed by G4LogicalVolume* and so iterates in address order.
  /// Address order is arbitrary: with two scorers, whether the first one registered got slot 0
  /// or slot 1 depended on where the allocator happened to put two logical volumes. The run's
  /// score array is indexed by slot, so anything that assumed "result i belongs to the scorer I
  /// declared i-th" was right by luck and wrong on the next build. Registration order makes
  /// that assumption true.
  ///
  /// AddNewDetector is called for every SetSensitiveDetector, so a detector attached to three
  /// volumes appears three times; the device_index >= 0 test below is what makes that a no-op
  /// rather than three slots.
  void AssignIndices() {
    scorers_.clear();
    for (G4VSensitiveDetector* sd : detectors_) {
      for (G4VPrimitiveScorer* ps : sd->GetScorers()) {
        if (ps->device_index >= 0) { continue; }
        ps->device_index = static_cast<G4int>(scorers_.size());
        ps->total = 0;
        ps->total_sq = 0;
        scorers_.push_back(ps);
      }
    }
  }

  /// The scorer slot a logical volume feeds, or -1. A detector with several scorers uses the
  /// first; the kernels accumulate energy deposit once per volume and the host converts it
  /// into whatever each scorer reports, so the distinction between dose and energy deposit is
  /// made after the run, not during it.
  G4int SlotFor(G4LogicalVolume* lv) const {
    G4VSensitiveDetector* sd = DetectorFor(lv);
    if (sd == nullptr) { return -1; }
    const auto ps = sd->GetScorers();
    return ps.empty() ? -1 : ps.front()->device_index;
  }

  /// Whether the scorer a logical volume feeds wants a value per voxel. Same "the first one
  /// decides" rule as SlotFor, and for the same reason: the kernels accumulate once per step,
  /// so what the volume feeds is one slot however many scorers read it afterwards.
  G4bool PerVoxelFor(G4LogicalVolume* lv) const {
    G4VSensitiveDetector* sd = DetectorFor(lv);
    if (sd == nullptr) { return false; }
    const auto ps = sd->GetScorers();
    return !ps.empty() && ps.front()->IsPerVoxel();
  }

 private:
  std::vector<G4VSensitiveDetector*> detectors_;
  std::vector<G4VPrimitiveScorer*> scorers_;
  std::map<G4LogicalVolume*, G4VSensitiveDetector*> attached_;
};
