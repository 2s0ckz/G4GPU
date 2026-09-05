// G4Element, G4Material and G4NistManager.
//
// Materials are described the way Geant4 describes them - by element, by mass fraction or
// atom count, with a density and a state - and converted to the flat device Material record
// only when the detector is finished. The conversion goes through data::add_material, which
// also derives the production thresholds from the range cut, so a user-defined material needs
// no hand-entered cut energies.
#pragma once
#include <cmath>
#include <map>
#include <memory>
#include <vector>
#include "data/materials.cuh"
#include "data/nist_excitation.hh"
#include "g4/G4EmParameters.hh"
#include "g4/G4SystemOfUnits.hh"
#include "g4/G4Types.hh"

enum G4State { kStateUndefined = 0, kStateSolid, kStateLiquid, kStateGas };

class G4Element {
 public:
  /// Geant4's signature: name, chemical symbol, atomic number, molar mass.
  G4Element(const G4String& name, const G4String& symbol, G4double zeff, G4double aeff)
      : name_(name), symbol_(symbol), z_(zeff), a_(aeff) {
    Registry().push_back(this);
  }

  const G4String& GetName() const { return name_; }
  const G4String& GetSymbol() const { return symbol_; }
  G4double GetZ() const { return z_; }
  G4int GetZasInt() const { return static_cast<G4int>(z_ + 0.5); }
  G4double GetA() const { return a_; }
  G4double GetAtomicMassAmu() const { return a_ / (g / mole); }

  static std::vector<G4Element*>& Registry() {
    static std::vector<G4Element*> r;
    return r;
  }
  static G4Element* GetElement(const G4String& name) {
    for (G4Element* e : Registry()) {
      if (e->name_ == name) { return e; }
    }
    return nullptr;
  }

 private:
  G4String name_, symbol_;
  G4double z_, a_;
};

class G4Material {
 public:
  /// A single-element material.
  G4Material(const G4String& name, G4double z, G4double a, G4double density,
             G4State state = kStateUndefined, G4double temp = 0, G4double pressure = 0)
      : name_(name), density_(density), state_(state) {
    components_.push_back({nullptr, static_cast<G4int>(z + 0.5), a, 1.0, /*by_atoms=*/false});
    (void)temp;
    (void)pressure;
    Registry().push_back(this);
  }

  /// A compound, to be filled with AddElement / AddMaterial.
  G4Material(const G4String& name, G4double density, G4int ncomponents,
             G4State state = kStateUndefined, G4double temp = 0, G4double pressure = 0)
      : name_(name), density_(density), state_(state) {
    components_.reserve(static_cast<std::size_t>(ncomponents));
    (void)temp;
    (void)pressure;
    Registry().push_back(this);
  }

  void AddElement(G4Element* el, G4double fraction) {
    components_.push_back({el, el->GetZasInt(), el->GetA(), fraction, false});
  }
  void AddElement(G4Element* el, G4int natoms) {
    components_.push_back({el, el->GetZasInt(), el->GetA(), static_cast<G4double>(natoms), true});
  }
  void AddMaterial(G4Material* mat, G4double fraction) {
    for (const Component& c : mat->components_) {
      Component out = c;
      out.amount = c.amount * fraction;
      out.by_atoms = false;
      components_.push_back(out);
    }
  }

  const G4String& GetName() const { return name_; }

  /// Geant4's chemical formula, which is not decoration: G4BraggModel matches on it to choose
  /// the ICRU 49 molecular stopping power for a compound that is not one of the 74 NIST
  /// materials. Set it to "H_2O", "SiO_2", "CO_2" and so on for a hand-built compound and the
  /// low-energy proton stopping power becomes Geant4's; leave it empty and the per-element
  /// Ziegler fit is used, which for water differs by up to 29% at the Bragg peak.
  ///
  /// The recognised strings are in data/nist_stopping_names.hh.
  void SetChemicalFormula(const G4String& f) { formula_ = f; }
  const G4String& GetChemicalFormula() const { return formula_; }
  G4double GetDensity() const { return density_; }
  G4State GetState() const { return state_; }
  std::size_t GetNumberOfElements() const { return components_.size(); }

  /// A tabulated mean excitation energy overrides the Bragg-rule estimate. Geant4 exposes the
  /// same override through G4IonisParamMat::SetMeanExcitationEnergy.
  void SetMeanExcitationEnergy(G4double i) { mean_excitation_ = i; }
  G4double GetMeanExcitationEnergy() const { return mean_excitation_; }

  /// The mean excitation energy actually in use: the one that was set, or the one Build()
  /// derived by Bragg additivity when none was.
  ///
  /// Zero until Build() has run, because until then there is nothing to report. Deriving it
  /// here on demand instead would be a second implementation of the same rule, correct right
  /// up until the day G4IonisParamMat changed and the two disagreed.
  G4double GetEffectiveMeanExcitationEnergy() const {
    return (mean_excitation_ > 0) ? mean_excitation_ : derived_excitation_;
  }

  /// Sternheimer density-effect parameters, when tabulated ones are known. Without them the
  /// analytic fallback is used, which for water is off by 2% in Cbar - enough to shift the
  /// electron stopping power in the fourth digit.
  void SetDensityEffectParameters(G4double C, G4double x0, G4double x1, G4double a, G4double mm_,
                                  G4double delta0 = 0) {
    has_sternheimer_ = true;
    ster_[0] = C; ster_[1] = x0; ster_[2] = x1; ster_[3] = a; ster_[4] = mm_; ster_[5] = delta0;
  }

  /// Emits this material into the device table, returning its index.
  G4int Build(g4gpu::data::MaterialTable<G4double>& table, G4double range_cut_mm) const;

  static std::vector<G4Material*>& Registry() {
    static std::vector<G4Material*> r;
    return r;
  }
  static G4Material* GetMaterial(const G4String& name) {
    for (G4Material* m : Registry()) {
      if (m->name_ == name) { return m; }
    }
    return nullptr;
  }

  /// Index in the device material table, filled in by the detector flattener.
  G4int device_index = -1;

 private:
  struct Component {
    G4Element* element;
    G4int z;
    G4double a;       ///< molar mass, Geant4 units
    G4double amount;  ///< mass fraction, or an atom count when by_atoms
    G4bool by_atoms;
  };

  G4String name_;
  G4String formula_;
  G4double density_;
  G4State state_;
  std::vector<Component> components_;
  G4double mean_excitation_ = 0;
  /// What Build() derived when mean_excitation_ was zero. Mutable because Build() is const:
  /// it fills a table rather than changing the material, and this records what it computed
  /// rather than changing what the material is.
  mutable G4double derived_excitation_ = 0;
  G4bool has_sternheimer_ = false;
  G4double ster_[6] = {0, 0, 0, 0, 0, 0};
};

inline G4int G4Material::Build(g4gpu::data::MaterialTable<G4double>& table,
                               G4double range_cut_mm) const {
  namespace data = g4gpu::data;
  std::vector<G4int> zs;
  std::vector<G4double> w;

  // Atom counts become mass fractions, which is the form the device record wants.
  G4double total = 0;
  const G4bool by_atoms = !components_.empty() && components_[0].by_atoms;
  for (const Component& c : components_) {
    total += by_atoms ? c.amount * c.a / (g / mole) : c.amount;
  }
  if (total <= 0) { total = 1; }
  for (const Component& c : components_) {
    zs.push_back(c.z);
    w.push_back((by_atoms ? c.amount * c.a / (g / mole) : c.amount) / total);
  }

  const auto state = (state_ == kStateGas)      ? data::MaterialState::kGas
                     : (state_ == kStateSolid)  ? data::MaterialState::kSolid
                     : (state_ == kStateLiquid) ? data::MaterialState::kLiquid
                                                : data::MaterialState::kAuto;

  // A material with no usable components cannot be built. Refused loudly rather than passed
  // on: `add_material` would produce a record with zero electron density, and the first
  // dE/dx would divide by it.
  if (zs.empty()) {
    std::printf("\nFATAL: material \"%s\" has no components.\n"
                "  Give it a NIST name (G4_WATER, ...), or add at least one element with a\n"
                "  non-zero fraction. A material with nothing in it has no stopping power,\n"
                "  and a run using it would score a NaN dose rather than fail.\n",
                name_.c_str());
    std::exit(2);
  }

  // The mean excitation energy, derived if it was not given.
  //
  // Geant4 derives it in G4IonisParamMat::ComputeMeanParameters when the material has no
  // tabulated value, as Bragg additivity in the logarithm over the elements. Passing zero
  // through instead put `log(0)` into the density-effect parameters, and every dose computed
  // with a user-built material came back NaN. See data/nist_excitation.hh.
  G4double excitation_eV = mean_excitation_ / eV;
  if (excitation_eV <= 0) {
    excitation_eV = data::derive_mean_excitation_eV(
        static_cast<int>(zs.size()), zs.data(), w.data(),
        [](int z) { return static_cast<double>(data::atomic_mass<G4double>(z)); });
    if (excitation_eV <= 0) {
      std::printf("\nFATAL: material \"%s\" has no mean excitation energy and none could be\n"
                  "  derived - no component has a tabulated value. Set one explicitly with\n"
                  "  SetMeanExcitationEnergy, or use elements with Z between 1 and 98.\n",
                  name_.c_str());
      std::exit(2);
    }
  }

  const G4int idx = data::add_material<G4double>(
      table, density_ / (g / cm3), static_cast<int>(zs.size()), zs.data(), w.data(),
      excitation_eV, range_cut_mm, state);
  derived_excitation_ = excitation_eV * eV;
  if (idx >= 0) {
    // Which tabulated stopping-power data this material has, by name and then by chemical
    // formula, exactly as G4BraggModel decides it. Done here because this is the last place a
    // material still has a name: the device record does not carry one.
    data::set_nist_stopping<G4double>(table[idx], name_.c_str(), formula_.c_str());
    // ICRU 90 replaces PSTAR/ASTAR for air, water and graphite when the user asked for it.
    // Read here rather than in the physics because this is where the material still has a
    // name, and consulted once per material because that is what G4BraggModel does.
    if (G4EmParameters::Instance()->UseICRU90Data()) {
      data::set_icru90<G4double>(table[idx], name_.c_str());
    }
  }
  if (idx >= 0 && has_sternheimer_) {
    data::set_sternheimer<G4double>(table[idx], ster_[0], ster_[1], ster_[2], ster_[3], ster_[4],
                                    ster_[5]);
  }
  return idx;
}
