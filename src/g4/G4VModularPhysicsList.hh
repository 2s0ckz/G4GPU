// Physics lists: G4VUserPhysicsList, G4VModularPhysicsList, G4VPhysicsConstructor.
//
// A Geant4 example chooses its physics by handing the run manager a list, and B1 hands it
// QBBC. The list is not decoration here - it decides which processes the device stepper runs,
// through g4gpu::ProcessFlags - but it is coarser than Geant4's: a list here is a set of
// switches, not a registry of process objects with their own models and tables. What the
// switches do is in src/physics/stepper.cuh; which models implement them is a property of
// this codebase, not of the list.
//
// Where a Geant4 list promises physics this transport does not have, the class says so out
// loud at construction. QBBC's hadronic component is the case that matters: B1's run1.mac
// fires 210 MeV protons through it, and a run that silently transported them as something
// else would be the worst possible outcome.
#pragma once
#include <cstdio>
#include <memory>
#include <string>
#include <vector>
#include "g4/G4SystemOfUnits.hh"
#include "g4/G4Types.hh"
#include "physics/scene.cuh"

class G4VUserPhysicsList;

/// A component of a modular list. Geant4's G4VPhysicsConstructor builds processes; here it
/// turns switches on.
class G4VPhysicsConstructor {
 public:
  explicit G4VPhysicsConstructor(const G4String& name = "") : name_(name) {}
  virtual ~G4VPhysicsConstructor() = default;

  const G4String& GetPhysicsName() const { return name_; }
  void SetVerboseLevel(G4int v) { verbose_ = v; }
  G4int GetVerboseLevel() const { return verbose_; }

  /// Geant4's two build methods. Overriding them is how a user's own constructor works, so
  /// they are virtual and called; the default does nothing.
  virtual void ConstructParticle() {}
  virtual void ConstructProcess() {}

  /// What this component switches on. ORed into the list's flags.
  virtual void Apply(g4gpu::ProcessFlags& f) const { (void)f; }

 private:
  G4String name_;
  G4int verbose_ = 0;
};

/// The EM standard process set: what B1 actually runs, and what this transport is validated
/// against - see docs/RESULT.md.
class G4EmStandardPhysics : public G4VPhysicsConstructor {
 public:
  explicit G4EmStandardPhysics(G4int ver = 1, const G4String& name = "G4EmStandard")
      : G4VPhysicsConstructor(name) {
    SetVerboseLevel(ver);
  }

  void Apply(g4gpu::ProcessFlags& f) const override {
    f.photoelectric = true;
    f.compton = true;
    f.rayleigh = true;
    f.pair_production = true;
    f.bremsstrahlung = true;
    f.annihilation = true;
    f.multiple_scattering = true;
  }
};

/// The option4 variant differs in Geant4 by using more accurate models and finer binning. The
/// models here are one set, so this is the same physics as G4EmStandardPhysics; it exists so
/// that a detector description naming it compiles, and it says what it is.
class G4EmStandardPhysics_option4 : public G4EmStandardPhysics {
 public:
  explicit G4EmStandardPhysics_option4(G4int ver = 1)
      : G4EmStandardPhysics(ver, "G4EmStandard_opt4") {
    std::printf(
        "note: G4EmStandardPhysics_option4 selects the same models as G4EmStandardPhysics\n"
        "      here; there is one set of EM models, not four options over them.\n");
  }
};

class G4VUserPhysicsList {
 public:
  virtual ~G4VUserPhysicsList() = default;

  virtual void ConstructParticle() {}
  virtual void ConstructProcess() {}

  void SetVerboseLevel(G4int v) { verbose_ = v; }
  G4int GetVerboseLevel() const { return verbose_; }

  /// Geant4's default production cut, applied to every particle and region.
  void SetDefaultCutValue(G4double c) {
    default_cut_ = c;
    cut_set_ = true;
  }
  G4double GetDefaultCutValue() const { return default_cut_; }
  G4bool HasCutValue() const { return cut_set_; }

  /// Per-particle cuts. Accepted; there is one range cut here, so the last one set wins and
  /// a per-particle value prints what it did.
  void SetCutValue(G4double c, const G4String& particle) {
    if (cut_set_ && c != default_cut_) {
      std::printf("note: /run/setCut is one range cut for every species here; \"%s\" sets it\n"
                  "      for all of them.\n",
                  particle.c_str());
    }
    SetDefaultCutValue(c);
  }
  void SetCutsWithDefault() {}
  void DumpCutValuesTable(G4int = 0) {}

  /// The switches this list asks the stepper for.
  virtual g4gpu::ProcessFlags Flags() const = 0;

 private:
  G4double default_cut_ = 0.7 * mm;
  G4bool cut_set_ = false;
  G4int verbose_ = 0;
};

class G4VModularPhysicsList : public G4VUserPhysicsList {
 public:
  void RegisterPhysics(G4VPhysicsConstructor* c) {
    if (c != nullptr) { parts_.emplace_back(c); }
  }
  G4int GetNumberOfPhysicsConstructor() const { return static_cast<G4int>(parts_.size()); }
  const G4VPhysicsConstructor* GetPhysics(G4int i) const {
    return (i >= 0 && i < static_cast<G4int>(parts_.size())) ? parts_[i].get() : nullptr;
  }

  /// Every registered component's switches, ORed. Nothing is on that no component asked for,
  /// so an empty list transports nothing and says so when the run produces no interactions.
  g4gpu::ProcessFlags Flags() const override {
    g4gpu::ProcessFlags f{};
    f.photoelectric = f.compton = f.rayleigh = f.pair_production = false;
    f.bremsstrahlung = f.annihilation = f.multiple_scattering = false;
    for (const auto& p : parts_) { p->Apply(f); }
    return f;
  }

 private:
  std::vector<std::unique_ptr<G4VPhysicsConstructor>> parts_;
};
