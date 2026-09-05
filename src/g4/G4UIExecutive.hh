// G4UIExecutive and G4VisExecutive.
//
// These are the two objects a Geant4 example's main() creates to get an interactive session:
// the vis manager, which owns the viewer and the /vis/ command set, and the UI executive,
// whose SessionStart() hands control to the user until they quit.
//
//     G4UIExecutive* ui = nullptr;
//     if (argc == 1) { ui = new G4UIExecutive(argc, argv); }
//     G4VisManager* visManager = new G4VisExecutive;
//     visManager->Initialize();
//     ...
//     if (ui) { UImanager->ApplyCommand("/control/execute init_vis.mac"); ui->SessionStart(); }
//
// Both are real here. G4VisExecutive::Initialize() registers the visualisation manager in
// render/vis_manager.h as the handler for /vis/ commands, so `/vis/open OGL 600x600` in an
// example's vis.mac opens the same window g4view.exe opens, with the same run control bar.
// SessionStart() then runs that window's event loop.
//
// If no /vis/open has been issued by the time SessionStart() is reached, there is no window to
// pump, and the session falls back to reading commands from stdin with an `Idle>` prompt -
// which is what a Geant4 terminal session does.
#pragma once
#include <cstdio>
#include <iostream>
#include <string>
#include "g4/G4UImanager.hh"
#include "g4/G4ios.hh"
#include "render/vis_manager.h"

/// Geant4's base class name; examples declare the variable as G4VisManager*.
class G4VisManager {
 public:
  virtual ~G4VisManager() = default;
  virtual void Initialize() = 0;
  void SetVerboseLevel(G4int v) { verbose_ = v; }
  void SetVerboseLevel(const G4String&) {}
  G4int GetVerboseLevel() const { return verbose_; }

 private:
  G4int verbose_ = 1;
};

class G4VisExecutive : public G4VisManager {
 public:
  G4VisExecutive() = default;
  /// Geant4 takes a verbosity string here ("quiet", "errors", "warnings", ...).
  explicit G4VisExecutive(const G4String& verbosity) { (void)verbosity; }

  void Initialize() override {
    G4UImanager::GetUIpointer()->SetVisHandler(g4gpu::vis::viewer::ApplyVisCommand);
  }
};

class G4UIExecutive {
 public:
  G4UIExecutive(G4int argc, char** argv, const G4String& session = "") {
    (void)argc;
    (void)argv;
    (void)session;
  }

  G4bool IsGUI() const { return g4gpu::vis::viewer::IsOpen(); }

  /// Hands control over until the user quits.
  ///
  /// With a window open this is the viewer's event loop: the run control bar, the camera and
  /// the text panel. Without one it is a command prompt on stdin, which is how a Geant4
  /// terminal session behaves - and which is also what makes an example scriptable by piping
  /// commands into it.
  void SessionStart() {
    if (g4gpu::vis::viewer::IsOpen()) {
      g4gpu::vis::viewer::Loop();
      return;
    }
    auto* ui = G4UImanager::GetUIpointer();
    std::string line;
    std::printf("No viewer open (/vis/open was not issued). Type commands, or `exit`.\n");
    while (true) {
      std::printf("Idle> ");
      std::fflush(stdout);
      if (!std::getline(std::cin, line)) { break; }
      if (line == "exit" || line == "quit") { break; }
      if (line.empty()) { continue; }
      ui->ApplyCommand(line);
    }
  }
};
