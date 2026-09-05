// The visualisation manager: an interactive window, a run control bar, and the /vis/ command
// set, in one object that any program can open.
//
// This is the header Geant4's G4VisExecutive stands in front of. It exists as a separate
// translation unit - compiled once by build_vis.bat into out/vis_manager.obj and linked into
// whatever needs it - for the same reason Geant4 ships libG4vis rather than a header: the
// viewer pulls in Win32, WGL, GDI and the CUDA renderer, and every program that included it
// would pay to compile all of that again.
//
// A viewer needs a geometry, so Open() requires G4RunManager::Instance() to have been
// Initialize()d. In a Geant4 macro that ordering is already the convention:
//
//     /run/initialize
//     /control/execute vis.mac      # which starts with /vis/open
#pragma once
#include <string>

namespace g4gpu::vis::viewer {

struct Options {
  int width = 1440;
  int height = 880;
  /// Non-zero drives the camera and one run for this many frames, saves a PNG and exits.
  /// The only way to smoke-test an interactive window with no human in front of it.
  int selftest_frames = 0;
  int initial_events = 0;
  /// Holds the viewpoint instead of orbiting, so the saved image is comparable between runs.
  bool fixed_view = false;
  double shot_theta = 70;
  double shot_phi = 20;
  std::string png_path = "out/g4view_selftest.png";
  /// Shown in the title bar.
  std::string title = "g4gpu - viewer";
};

/// Creates the window, the GL context, the fonts and the device buffers, sizing the camera
/// from the geometry's extent. Returns false if there is no initialised run manager, or if
/// the window, the GL context or the font atlas could not be created. Calling it twice is a
/// no-op that returns true, so `/vis/open` in a macro is harmless after -w/-h on the command
/// line have already opened one.
bool Open(const Options& opt);

/// True once Open() has succeeded.
bool IsOpen();

/// Pumps messages and draws until the window closes, or until the selftest frame count is
/// reached. Returns 0 on a clean exit.
int Loop();

/// Handles the /vis/ subset a Geant4 visualisation macro uses. `/vis/open` opens the viewer if
/// it is not open yet, with the geometry the run manager holds. Returns false for a command
/// the viewer does not implement, which the UI manager reports.
bool ApplyVisCommand(const std::string& cmd);

/// Runs `n` events, appending trajectories and updating the panel. What the Run button does.
void BeamOnEvents(int n);

/// Appends a line to the on-screen log, and to stdout.
void LogLine(const std::string& s);

/// The options Open() was called with, so a caller can read back the size actually used.
const Options& CurrentOptions();

}  // namespace g4gpu::vis::viewer
