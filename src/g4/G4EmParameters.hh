// G4EmParameters: the EM options a Geant4 application sets before the run.
//
// Geant4's class carries some eighty parameters. This holds the ones that change an answer
// this port computes, and nothing else - a setter that silently does nothing is worse than a
// missing one, because it reads as configured.
//
//   SetUseICRU90Data(bool)      ICRU Report 90 stopping powers for air, water and graphite,
//                               in place of PSTAR/ASTAR. Off by default, as in Geant4.
//
// Everything else Geant4's messenger accepts is left out on purpose. `G4UImanager` reports an
// unhandled `/process/em/...` command rather than swallowing it, so a macro that sets a
// parameter this port does not implement says so instead of appearing to work.
//
// The flag is read when a material's device record is built - `G4Material` calls
// `set_icru90` at that point, exactly as `G4BraggModel` resolves `iICRU90` once per material -
// so it has to be set **before** `G4RunManager::Initialize()`. That ordering is Geant4's own:
// `G4EmParameters` is consulted in `G4VEmModel::Initialise`, which runs at initialisation.
#pragma once
#include "G4Types.hh"

/// The EM parameter set. A singleton, as in Geant4.
class G4EmParameters {
 public:
  static G4EmParameters* Instance() {
    static G4EmParameters inst;
    return &inst;
  }

  /// ICRU 90 stopping powers instead of PSTAR/ASTAR, for the three materials ICRU 90 covers.
  ///
  /// Off by default, which is Geant4's default too. When it is on, G4BraggModel resolves the
  /// ICRU 90 index *before* the PSTAR index and returns as soon as it has one - so this
  /// replaces the stopping power for G4_AIR, G4_WATER and G4_GRAPHITE and leaves every other
  /// material on PSTAR.
  void SetUseICRU90Data(G4bool v) { use_icru90_ = v; }
  G4bool UseICRU90Data() const { return use_icru90_; }

  G4EmParameters(const G4EmParameters&) = delete;
  G4EmParameters& operator=(const G4EmParameters&) = delete;

 private:
  G4EmParameters() = default;
  G4bool use_icru90_ = false;
};
