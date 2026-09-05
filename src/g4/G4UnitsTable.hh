// The units table, and G4BestUnit.
//
// Geant4 keeps a registry of named units grouped into categories - Length, Energy, Dose - and
// G4BestUnit picks the one that makes a number readable: 427.4 picoGy rather than
// 4.274e-10 gray. Examples use it in every print, and B1's RunAction registers the dose
// sub-multiples itself, so the registry has to accept new units into a category at runtime
// rather than being a fixed table.
#pragma once
#include <cmath>
#include <cstdio>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>
#include "g4/G4SystemOfUnits.hh"
#include "g4/G4Types.hh"
#include "g4/G4ios.hh"

class G4UnitDefinition {
 public:
  struct Entry {
    G4String name;
    G4String symbol;
    G4String category;
    G4double value;
  };

  /// Registering a unit is the whole purpose of constructing one, exactly as in Geant4 where
  /// examples write `new G4UnitDefinition("picogray", "picoGy", "Dose", picogray);` and never
  /// touch the returned object. A category that does not exist yet is created.
  G4UnitDefinition(const G4String& name, const G4String& symbol, const G4String& category,
                   G4double value) {
    for (Entry& e : Table()) {
      if (e.name == name && e.category == category) {
        e.symbol = symbol;
        e.value = value;
        return;
      }
    }
    Table().push_back(Entry{name, symbol, category, value});
  }

  static std::vector<Entry>& Table() {
    static std::vector<Entry> t = BuiltIn();
    return t;
  }

  /// The value of a named unit, or 0 if there is no such name. Geant4 returns the value and
  /// warns; a caller that multiplies by 0 gets an obviously wrong answer rather than a
  /// plausible one, which is the intent.
  static G4double GetValueOf(const G4String& name) {
    for (const Entry& e : Table()) {
      if (e.name == name || e.symbol == name) { return e.value; }
    }
    std::printf("WARNING: unknown unit \"%s\"\n", name.c_str());
    return 0;
  }

  static G4String GetCategory(const G4String& name) {
    for (const Entry& e : Table()) {
      if (e.name == name || e.symbol == name) { return e.category; }
    }
    return "";
  }

  static void BuildUnitsTable() { (void)Table(); }

  static void PrintUnitsTable() {
    G4cout << "----- The units table -----" << G4endl;
    for (const Entry& e : Table()) {
      G4cout << "  " << e.category << ": " << e.name << " (" << e.symbol << ") = " << e.value
             << G4endl;
    }
  }

 private:
  /// The categories every Geant4 program has before an example adds anything. Deliberately
  /// not exhaustive: what is here is what the units table is asked for in practice, and an
  /// unknown name warns rather than being silently treated as 1.
  static std::vector<Entry> BuiltIn() {
    return {
        {"nanometer", "nm", "Length", nanometer},
        {"micrometer", "um", "Length", micrometer},
        {"millimeter", "mm", "Length", millimeter},
        {"centimeter", "cm", "Length", centimeter},
        {"meter", "m", "Length", meter},
        {"kilometer", "km", "Length", kilometer},

        {"electronvolt", "eV", "Energy", electronvolt},
        {"kiloelectronvolt", "keV", "Energy", kiloelectronvolt},
        {"megaelectronvolt", "MeV", "Energy", megaelectronvolt},
        {"gigaelectronvolt", "GeV", "Energy", gigaelectronvolt},
        {"teraelectronvolt", "TeV", "Energy", teraelectronvolt},
        {"joule", "J", "Energy", joule},

        {"nanosecond", "ns", "Time", nanosecond},
        {"microsecond", "us", "Time", microsecond},
        {"millisecond", "ms", "Time", millisecond},
        {"second", "s", "Time", second},

        {"milligram", "mg", "Mass", milligram},
        {"gram", "g", "Mass", gram},
        {"kilogram", "kg", "Mass", kilogram},

        {"gray", "Gy", "Dose", gray},

        {"radian", "rad", "Angle", radian},
        {"milliradian", "mrad", "Angle", milliradian},
        {"degree", "deg", "Angle", degree},

        {"mm3", "mm3", "Volume", mm3},
        {"cm3", "cm3", "Volume", cm3},
        {"m3", "m3", "Volume", m3},

        {"g/cm3", "g/cm3", "Volumic Mass", gram / cm3},
        {"mg/cm3", "mg/cm3", "Volumic Mass", milligram / cm3},
        {"kg/m3", "kg/m3", "Volumic Mass", kilogram / m3},

        {"becquerel", "Bq", "Activity", becquerel},
        {"kilobecquerel", "kBq", "Activity", kilobecquerel},
        {"megabecquerel", "MBq", "Activity", megabecquerel},
        {"gigabecquerel", "GBq", "Activity", gigabecquerel},
        {"curie", "Ci", "Activity", curie},
        {"millicurie", "mCi", "Activity", millicurie},
        {"microcurie", "uCi", "Activity", microcurie},
    };
  }
};

/// A value plus the category to print it in. Streaming it writes the number in whichever unit
/// of that category makes it readable, with the symbol.
///
///     G4cout << G4BestUnit(dose, "Dose") << G4endl;    ->    427.385 picoGy
class G4BestUnit {
 public:
  G4BestUnit(G4double value, const G4String& category)
      : value_(value), category_(category) {}

  /// Geant4 defines this, which is what makes `someG4String += G4BestUnit(e, "Energy")`
  /// compile in an example. Defined below, after the stream operator it uses.
  operator G4String() const;

  G4double Value() const { return value_; }
  const G4String& Category() const { return category_; }

  /// The largest unit of the category in which the value is at least 1, or the smallest unit
  /// of the category if the value is smaller than all of them. This is Geant4's choice, and
  /// it is what keeps 4.274e-10 gray from ever being printed.
  const G4UnitDefinition::Entry* Best() const {
    const G4UnitDefinition::Entry* best = nullptr;
    const G4UnitDefinition::Entry* smallest = nullptr;
    const G4double v = std::fabs(value_);
    for (const auto& e : G4UnitDefinition::Table()) {
      if (e.category != category_) { continue; }
      if (smallest == nullptr || e.value < smallest->value) { smallest = &e; }
      if (v >= e.value && (best == nullptr || e.value > best->value)) { best = &e; }
    }
    return (best != nullptr) ? best : smallest;
  }

 private:
  G4double value_;
  G4String category_;
};

inline std::ostream& operator<<(std::ostream& os, const G4BestUnit& b) {
  const auto* u = b.Best();
  if (u == nullptr) {
    // No unit in that category: print the raw internal number rather than inventing a symbol.
    return os << b.Value() << " [no unit registered for category \"" << b.Category() << "\"]";
  }
  return os << (b.Value() / u->value) << " " << u->symbol;
}

inline G4BestUnit::operator G4String() const {
  std::ostringstream os;
  os << *this;
  return os.str();
}
