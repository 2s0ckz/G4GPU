// Every physical constant in the port, against the ones CLHEP defines in Geant4 11.1.1.
//
// This test exists because a constant is the one kind of transcription error that looks like
// a physics bug. `src/physics/em/em_corrections.cuh` held amu_c2 = 931.49410242 - the current
// CODATA value, and more accurate than Geant4's - which put G4ionEffectiveCharge 1.5e-8 off
// Geant4's own answer. Invisible in a dose, fatal to a 1e-9 comparison, and every plausible
// explanation was a formula. `src/physics/em/nuclear_stopping.cuh` had the same value while
// `bragg.cuh` two files away had Geant4's, so the port disagreed with itself as well.
//
// CLHEP pins these per release and several of 11.1.1's are older than what a person would look
// up. The rule this test enforces is: the reference implementation's value is the correct one,
// whatever the literature says, because the whole point is to reproduce its answers.
//
// Two sets are checked, because the port has two:
//   - `core/units.cuh`, which the device physics uses;
//   - `g4/G4PhysicalConstants.hh`, the Geant4-shaped header an example includes.
// They are separate definitions of the same numbers, so they are both compared - a drift
// between them is exactly as harmful as a drift from Geant4.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "core/units.cuh"
#include "g4/G4PhysicalConstants.hh"
#include "g4/G4SystemOfUnits.hh"

using real_t = double;

namespace {

struct Entry {
  const char* name;      ///< the row in constants.csv
  double ours;           ///< core/units.cuh, or a derived value
  double g4_api;         ///< g4/G4PhysicalConstants.hh, or NAN if it has no such constant
  double tol;            ///< relative
  const char* why;       ///< why the tolerance is not zero
};

}  // namespace

int main() {
  const char* env = std::getenv("G4GPU_ORACLE");
  const std::string dir = (env != nullptr) ? env : "ref/oracle";
  FILE* f = std::fopen((dir + "/constants.csv").c_str(), "r");
  if (f == nullptr) {
    std::printf("cannot read %s/constants.csv\n", dir.c_str());
    return 1;
  }

  namespace u = g4gpu::units;
  // g4/G4PhysicalConstants.hh works in Geant4's internal units, so a value in mm or MeV is
  // already the number CLHEP prints; the divisions below are written out anyway so the unit of
  // each comparison is visible rather than implied.
  const Entry table[] = {
      {"electron_mass_c2", u::electron_mass_c2<real_t>(), electron_mass_c2 / MeV, 0,
       "an exact literal in both"},
      {"proton_mass_c2", u::proton_mass_c2<real_t>(), proton_mass_c2 / MeV, 0,
       "an exact literal in both"},
      {"neutron_mass_c2", u::neutron_mass_c2<real_t>(), neutron_mass_c2 / MeV, 0,
       "an exact literal in both"},
      {"amu_c2", u::amu_c2<real_t>(), amu_c2 / MeV, 0, "an exact literal in both"},
      {"Avogadro", u::avogadro<real_t>(), Avogadro * mole, 0, "an exact literal in both"},
      {"pi", u::pi<real_t>(), pi, 0, "an exact literal in both"},
      // CLHEP derives these three rather than tabulating them - out of e, mu0, h and c - so
      // the literature value of the same quantity is not the same number. The port now holds
      // CLHEP's own derived values, read out of ref/oracle/constants.csv, which is why the
      // tolerance is one bit and not the 8e-8 the CODATA figures would need.
      {"classic_electr_radius", u::classic_electron_radius<real_t>(),
       classic_electr_radius / mm, 1e-15,
       "pinned from the oracle, so only the last bit can differ"},
      {"fine_structure_const", u::fine_structure_const<real_t>(), fine_structure_const, 1e-15,
       "pinned from the oracle, so only the last bit can differ"},
      {"barn", u::barn<real_t>(), barn / (mm * mm), 1e-15,
       "CLHEP derives it through the unit system: 9.9999999999999993e-23, not 1e-22"},
  };
  constexpr int kN = int(sizeof table / sizeof table[0]);

  double from_csv[kN];
  bool seen[kN] = {};
  for (int i = 0; i < kN; ++i) { from_csv[i] = 0; }

  char line[512];
  if (std::fgets(line, sizeof line, f) == nullptr) {
    std::fclose(f);
    return 1;
  }
  int rows = 0;
  while (std::fgets(line, sizeof line, f) != nullptr) {
    char name[128] = {0}, unit[64] = {0};
    double v = 0;
    if (std::sscanf(line, "%127[^,],%lf,%63[^,\n]", name, &v, unit) < 2) { continue; }
    ++rows;
    for (int i = 0; i < kN; ++i) {
      if (std::strcmp(name, table[i].name) == 0) {
        from_csv[i] = v;
        seen[i] = true;
      }
    }
  }
  std::fclose(f);

  std::printf("== CLHEP constants, Geant4 11.1.1 (%d rows in constants.csv) ==\n", rows);
  std::printf("  %-24s %22s %12s %12s\n", "constant", "Geant4", "units.cuh", "G4 header");

  int fails = 0;
  for (int i = 0; i < kN; ++i) {
    const Entry& e = table[i];
    if (!seen[i]) {
      std::printf("  %-24s %22s\n", e.name, "MISSING from the oracle");
      std::printf("    FAIL: nothing to compare against\n");
      ++fails;
      continue;
    }
    const double dev_ours = std::fabs(e.ours / from_csv[i] - 1);
    const double dev_api =
        std::isnan(e.g4_api) ? 0.0 : std::fabs(e.g4_api / from_csv[i] - 1);
    std::printf("  %-24s %22.17g %12.2e %12.2e\n", e.name, from_csv[i], dev_ours, dev_api);
    if (dev_ours > e.tol) {
      std::printf("    FAIL: core/units.cuh is %.3e away (tolerance %.0e - %s)\n", dev_ours,
                  e.tol, e.why);
      ++fails;
    }
    if (dev_api > e.tol) {
      std::printf("    FAIL: g4/G4PhysicalConstants.hh is %.3e away (tolerance %.0e - %s)\n",
                  dev_api, e.tol, e.why);
      ++fails;
    }
  }

  // A constants file that lost a row would otherwise shrink this test silently.
  if (rows < kN) {
    std::printf("\n  FAIL: constants.csv has %d rows but %d constants are checked\n", rows,
                kN);
    ++fails;
  }

  std::printf("\n%s (%d failures)\n", fails ? "FAILED" : "PASSED", fails);
  return fails ? 1 : 0;
}
