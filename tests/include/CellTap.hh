/// \file CellTap.hh
///
/// test_voxel_scoring's device stepping action, in a header for the reason
/// tests/include/QualityFactorScoring.hh gives: since P8e the transport kernels for a project
/// hook are compiled one per translation unit by build_hook_engine.bat, and every one of those
/// units has to see the class. docs/RISK.md V65.
///
/// `CellRec` came out of an anonymous namespace to get here, and that is a fix rather than a
/// move. An anonymous namespace in a header gives every translation unit its own type of that
/// name, so `CellTap` - whose member is a `CellRec*` - would have had one definition per unit
/// with a different member type in each, which is an ODR violation that links cleanly because
/// nothing about `CellRec` reaches the mangled name of the kernels.
#ifndef CellTap_h
#define CellTap_h 1

#include "g4/G4VUserDeviceSteppingAction.hh"

/// What one step contributed, and where both its ends were.
struct CellRec {
  double pre_z;
  double post_z;
  double edep;
};

/// Records both ends of every step in the scoring volume.
class CellTap : public G4VUserDeviceSteppingAction<CellTap> {
 public:
  CellTap() = default;
  CellTap(CellRec* recs, int* ctl, int cap) : recs_(recs), ctl_(ctl), cap_(cap) {}

  __device__ void UserSteppingAction(const G4DeviceStep& step) const {
    if (recs_ == nullptr || step.GetScoreSlot() != 0) { return; }
    const int i = atomicAdd(ctl_, 1);
    if (i >= cap_) {
      atomicAdd(ctl_ + 1, 1);
      return;
    }
    recs_[i].pre_z = step.GetPreStepPoint()->GetPosition().z;
    recs_[i].post_z = step.GetPostStepPoint()->GetPosition().z;
    recs_[i].edep = step.GetTotalEnergyDeposit();
  }

 private:
  CellRec* recs_ = nullptr;
  int* ctl_ = nullptr;      ///< [0] written, [1] dropped
  int cap_ = 0;
};

#endif  // CellTap_h
