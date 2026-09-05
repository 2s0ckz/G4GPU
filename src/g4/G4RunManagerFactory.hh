// G4RunManagerFactory.
//
// Geant4 11 examples do not construct a run manager directly; they ask the factory for one and
// let it decide between the serial, multithreaded and tasking managers based on how Geant4 was
// built and on G4FORCE_RUN_MANAGER_TYPE. exampleB1.cc's first real line is that call, so it
// has to exist here for the file to be the file.
//
// There is one run manager here, and the type argument is accepted and ignored: the
// parallelism is the GPU's, and it is not a choice made at this level. GetRunManagerType()
// reports Serial, which is the truth about the host side.
#pragma once
#include "g4/G4RunManager.hh"

enum class G4RunManagerType {
  Default,
  Serial,
  SerialOnly,
  MT,
  MTOnly,
  Tasking,
  TaskingOnly,
  SubEventMT,
  SubEventMTOnly
};

class G4RunManagerFactory {
 public:
  static G4RunManager* CreateRunManager(G4RunManagerType type = G4RunManagerType::Default,
                                        G4bool fail_if_unsupported = true,
                                        G4int n_threads = 0) {
    (void)type;
    (void)fail_if_unsupported;
    (void)n_threads;
    return new G4RunManager;
  }
  /// Geant4 also accepts the type as a string, from G4FORCE_RUN_MANAGER_TYPE.
  static G4RunManager* CreateRunManager(const G4String& type, G4bool fail_if_unsupported = true,
                                        G4int n_threads = 0) {
    (void)type;
    (void)fail_if_unsupported;
    (void)n_threads;
    return new G4RunManager;
  }

  static G4RunManagerType GetRunManagerType() { return G4RunManagerType::Serial; }
};
