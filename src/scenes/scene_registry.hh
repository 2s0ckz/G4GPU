// Named scenes the viewer can load.
//
// A scene is just a detector construction plus a primary generator plus a run action - the
// same three objects an example's main() hands to the run manager. Registering them by name
// lets the viewer show any of them without a separate executable per detector, and lets the
// model-building GUI hand its own generated scene in through the same door.
#pragma once
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "g4/G4Colour.hh"
#include "g4/G4NistManager.hh"
#include "g4/G4RunManager.hh"
#include "g4/G4SDManager.hh"
#include "g4/G4Solids.hh"
#include "g4/G4UserActions.hh"

namespace g4gpu::scenes {

using Installer = std::function<void(G4RunManager*)>;

inline std::map<std::string, Installer>& Registry() {
  static std::map<std::string, Installer> r;
  return r;
}

inline void Register(const std::string& name, Installer fn) { Registry()[name] = fn; }

inline bool Install(const std::string& name, G4RunManager* rm) {
  const auto it = Registry().find(name);
  if (it == Registry().end()) { return false; }
  it->second(rm);
  return true;
}

inline std::string Names() {
  std::string s;
  for (const auto& kv : Registry()) { s += (s.empty() ? "" : ", ") + kv.first; }
  return s;
}

/// Registers a scene at static-initialisation time, so a translation unit that defines one
/// only has to declare a namespace-scope object of this type.
struct AutoRegister {
  AutoRegister(const std::string& name, Installer fn) { Register(name, std::move(fn)); }
};

}  // namespace g4gpu::scenes
