// G4Accumulable and G4AccumulableManager.
//
// In Geant4 these exist for multithreading: each worker thread keeps its own copy of a
// quantity, and Merge() sums them into the master's at end of run. An example registers its
// accumulables once in the constructor, Reset()s them at begin of run and Merge()s at end,
// and that is the shape B1's RunAction has.
//
// Here there is one process and one accumulator, so Merge() has nothing to merge. The classes
// exist anyway, with the same interface, because an example moved from Geant4 uses them and
// because Reset() at begin of run is load-bearing either way: without it a second /run/beamOn
// in the same session would add to the first run's total.
#pragma once
#include <string>
#include <vector>
#include "g4/G4Types.hh"

/// The type-erased half, so the manager can hold accumulables of different types.
class G4VAccumulable {
 public:
  explicit G4VAccumulable(const G4String& name = "") : name_(name) {}
  virtual ~G4VAccumulable() = default;
  virtual void Reset() = 0;
  virtual void Merge() = 0;
  const G4String& GetName() const { return name_; }

 private:
  G4String name_;
};

template <typename T>
class G4Accumulable : public G4VAccumulable {
 public:
  G4Accumulable(T initial = T{}) : value_(initial), initial_(initial) {}
  G4Accumulable(const G4String& name, T initial = T{})
      : G4VAccumulable(name), value_(initial), initial_(initial) {}

  G4Accumulable& operator=(T v) {
    value_ = v;
    return *this;
  }
  G4Accumulable& operator+=(T v) {
    value_ += v;
    return *this;
  }
  G4Accumulable& operator*=(T v) {
    value_ *= v;
    return *this;
  }
  G4Accumulable& operator++() {
    ++value_;
    return *this;
  }

  operator T() const { return value_; }

  T GetValue() const { return value_; }
  void SetValue(T v) { value_ = v; }

  void Reset() override { value_ = initial_; }
  /// Nothing to merge: one process, one copy. Kept so that an example's EndOfRunAction reads
  /// the way it does in Geant4.
  void Merge() override {}

 private:
  T value_;
  T initial_;
};

class G4AccumulableManager {
 public:
  static G4AccumulableManager* Instance() {
    static G4AccumulableManager inst;
    return &inst;
  }

  /// Geant4 takes a reference and keeps a pointer to it; the accumulable is a member of the
  /// user's RunAction and outlives the run.
  void RegisterAccumulable(G4VAccumulable& a) {
    for (G4VAccumulable* p : registry_) {
      if (p == &a) { return; }
    }
    registry_.push_back(&a);
  }
  void RegisterAccumulable(G4VAccumulable* a) {
    if (a != nullptr) { RegisterAccumulable(*a); }
  }

  void Reset() {
    for (G4VAccumulable* p : registry_) { p->Reset(); }
  }
  void Merge() {
    for (G4VAccumulable* p : registry_) { p->Merge(); }
  }

  std::size_t GetNofAccumulables() const { return registry_.size(); }

 private:
  std::vector<G4VAccumulable*> registry_;
};
