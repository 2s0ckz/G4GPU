// g4view: the viewer as a standalone program.
//
// All of the work is in render/vis_manager.cu, compiled once into out/vis_manager.obj and
// linked into anything that wants a window - this program, and any example whose macro says
// /vis/open. What is left here is the part that differs between the two: g4view picks a scene
// from the registry by name, where an example builds its own detector in main().
//
//   g4view.exe                       the B1 scene
//   g4view.exe -scene B1 vis.mac     run a visualisation macro
//   g4view.exe -n 500                run 500 events at startup
//   g4view.exe -shot 70 20           save one frame from a fixed viewpoint and exit
//   g4view.exe -selftest             drive the camera and a run for 120 frames, save, exit
//
// Controls
//   left drag    orbit                    SPACE   run the event count in the box
//   right drag   pan                      C       clear trajectories
//   wheel        zoom                      X      x-ray (tracks through solids)
//   R            reset camera              G      solids on/off
//   S            save a PNG                ESC    quit
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "g4/G4RunManager.hh"
#include "g4/G4UImanager.hh"
#include "render/vis_manager.h"
#include "scenes/scene_registry.hh"

int main(int argc, char** argv) {
  namespace viewer = g4gpu::vis::viewer;
  viewer::Options opt;
  std::string scene_name = "B1";
  std::string macro;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "-selftest") == 0) {
      opt.selftest_frames = 120;
    } else if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
      opt.initial_events = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "-shot") == 0 && i + 2 < argc) {
      // -shot <theta> <phi>: render a few frames from a fixed viewpoint, save, exit.
      opt.selftest_frames = 4;
      opt.fixed_view = true;
      opt.shot_theta = std::atof(argv[++i]);
      opt.shot_phi = std::atof(argv[++i]);
    } else if (std::strcmp(argv[i], "-scene") == 0 && i + 1 < argc) {
      scene_name = argv[++i];
    } else if (std::strcmp(argv[i], "-w") == 0 && i + 1 < argc) {
      opt.width = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "-h") == 0 && i + 1 < argc) {
      opt.height = std::atoi(argv[++i]);
    } else if (argv[i][0] != '-') {
      macro = argv[i];
    }
  }
  opt.png_path = "D:\\g4gpu\\out\\g4view_selftest.png";

  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
    std::printf("FATAL: no CUDA device\n");
    return 2;
  }
  std::printf("GPU: %s, CC %d.%d\n", prop.name, prop.major, prop.minor);

  // The scene, through the same API an example uses.
  auto* runManager = new G4RunManager;
  if (!g4gpu::scenes::Install(scene_name, runManager)) {
    std::printf("no scene named \"%s\"; available: %s\n", scene_name.c_str(),
                g4gpu::scenes::Names().c_str());
    return 2;
  }
  runManager->Initialize();

  auto* uim = G4UImanager::GetUIpointer();
  uim->SetVisHandler(viewer::ApplyVisCommand);
  if (!viewer::Open(opt)) { return 2; }

  viewer::LogLine("scene \"" + scene_name + "\"");
  if (!macro.empty()) { uim->ExecuteMacroFile(macro); }
  if (opt.initial_events > 0) { viewer::BeamOnEvents(opt.initial_events); }

  return viewer::Loop();
}
