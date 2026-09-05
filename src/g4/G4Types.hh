// The scalar typedefs every Geant4 source file starts from.
//
// Geant4 defines these so that the whole toolkit can be retyped at once; the same is true
// here, and G4double is the type the whole device side is templated on. Keeping the names
// means a DetectorConstruction written for Geant4 compiles here unchanged.
#pragma once
#include <string>

using G4double = double;
using G4float = float;
using G4int = int;
using G4long = long;
using G4bool = bool;
using G4String = std::string;

/// Geant4's own alias for a complex-free "no value" in optional arguments.
inline constexpr G4int kInvalidCopyNo = -1;
