// G4LogicalVolumeStore: the registry of every logical volume, by name.
//
// Geant4 examples use it to find a volume they did not keep a pointer to - most often the
// envelope, in a PrimaryGeneratorAction that wants its size. G4LogicalVolume keeps the same
// registry; this is Geant4's interface onto it.
#pragma once
#include <cstdio>
#include <vector>
#include "g4/G4LogicalVolume.hh"

class G4LogicalVolumeStore {
 public:
  static G4LogicalVolumeStore* GetInstance() {
    static G4LogicalVolumeStore inst;
    return &inst;
  }

  /// Geant4 returns nullptr and prints a warning when the name is not found, unless `verbose`
  /// is false. Same here: a null return is the caller's to handle, and silence would turn a
  /// misspelled volume name into a default-constructed geometry.
  G4LogicalVolume* GetVolume(const G4String& name, G4bool verbose = true) const {
    for (G4LogicalVolume* lv : G4LogicalVolume::Registry()) {
      if (lv->GetName() == name) { return lv; }
    }
    if (verbose) {
      std::printf("WARNING: G4LogicalVolumeStore has no volume named \"%s\"\n", name.c_str());
    }
    return nullptr;
  }

  std::size_t size() const { return G4LogicalVolume::Registry().size(); }
  G4LogicalVolume* operator[](std::size_t i) const { return G4LogicalVolume::Registry()[i]; }

  std::vector<G4LogicalVolume*>::const_iterator begin() const {
    return G4LogicalVolume::Registry().begin();
  }
  std::vector<G4LogicalVolume*>::const_iterator end() const {
    return G4LogicalVolume::Registry().end();
  }
};

class G4PhysicalVolumeStore {
 public:
  static G4PhysicalVolumeStore* GetInstance() {
    static G4PhysicalVolumeStore inst;
    return &inst;
  }
  std::size_t size() const { return G4PVPlacement::Registry().size(); }
  G4PVPlacement* operator[](std::size_t i) const { return G4PVPlacement::Registry()[i]; }
};
