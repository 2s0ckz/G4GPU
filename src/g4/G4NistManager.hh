// G4NistManager: builds materials by their NIST name, the way `FindOrBuildMaterial("G4_WATER")`
// does in Geant4.
//
// The compositions come from src/data/nist_materials.hh, generated from Geant4's own
// G4NistManager by ref/dump/g4dump.cc - including the tabulated Sternheimer density-effect
// parameters, which matter: for water the analytic fallback gives Cbar = 3.5017 where Geant4
// computes 3.5801, and that is a visible shift in the electron stopping power.
//
// An unknown name is a hard error rather than a silently empty material. A detector built on a
// material that does not exist would otherwise run and produce a dose.
#pragma once
#include <cstdio>
#include <cstdlib>
#include "data/nist_materials.hh"
#include "g4/G4Material.hh"

class G4NistManager {
 public:
  static G4NistManager* Instance() {
    static G4NistManager inst;
    return &inst;
  }

  G4Material* FindOrBuildMaterial(const G4String& name) {
    if (G4Material* existing = G4Material::GetMaterial(name)) { return existing; }
    const auto* entry = Find(name);
    if (entry == nullptr) {
      std::printf("\nFATAL: no NIST material named \"%s\".\n"
                  "  %d NIST materials are available; names are Geant4's, e.g. G4_WATER,\n"
                  "  G4_AIR, G4_BONE_COMPACT_ICRU. Define a custom material with G4Material\n"
                  "  and G4Element if this one is not in the NIST list.\n",
                  name.c_str(), g4gpu::g4::nist::kNumNistMaterials);
      std::exit(2);
    }
    const auto state = (entry->state == 1)   ? kStateSolid
                       : (entry->state == 2) ? kStateLiquid
                       : (entry->state == 3) ? kStateGas
                                             : kStateUndefined;
    auto* mat = new G4Material(name, entry->density_g_cm3 * g / cm3,
                               entry->n_components, state);
    for (int i = 0; i < entry->n_components; ++i) {
      const int z = entry->components[i].z;
      mat->AddElement(FindOrBuildElement(z), entry->components[i].fraction);
    }
    mat->SetMeanExcitationEnergy(entry->mean_excitation_eV * eV);
    if (entry->has_sternheimer) {
      mat->SetDensityEffectParameters(entry->cbar, entry->x0, entry->x1, entry->a, entry->m,
                                      entry->delta0);
    }
    return mat;
  }

  /// An element by atomic number, with its standard atomic weight.
  G4Element* FindOrBuildElement(G4int z) {
    char name[16];
    std::snprintf(name, sizeof name, "Z%d", z);
    if (G4Element* e = G4Element::GetElement(name)) { return e; }
    const G4double a = g4gpu::data::atomic_mass<G4double>(z);
    return new G4Element(name, name, static_cast<G4double>(z), a * g / mole);
  }

  /// How many NIST materials this build knows about.
  G4int GetNumberOfNistMaterials() const { return g4gpu::g4::nist::kNumNistMaterials; }
  const g4gpu::g4::nist::NistMaterial* GetNistMaterials() const {
    return g4gpu::g4::nist::kNistMaterials;
  }

 private:
  static const g4gpu::g4::nist::NistMaterial* Find(const G4String& name) {
    for (int i = 0; i < g4gpu::g4::nist::kNumNistMaterials; ++i) {
      if (name == g4gpu::g4::nist::kNistMaterials[i].name) {
        return &g4gpu::g4::nist::kNistMaterials[i];
      }
    }
    return nullptr;
  }
};
