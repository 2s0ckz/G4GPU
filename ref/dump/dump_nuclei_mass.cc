// nuclei_mass_levels.csv - G4NucleiProperties::GetNuclearMass over every nuclide a transport can
// make, and which of its four levels answered (P24).
//
// `had_masses.csv` (dump_hadronic_xs.cc) covers the AME2012 table and nothing past it, because
// until P24 the port refused everything past it. src/data/nuclei_mass_ame12.hh now carries all
// four levels - the particle-table masses, AME2012, the Moller-Nix theoretical table, and
// `Z == A`, `Z == 0` and Weizsaecker's formula - and this file is what it is compared with,
// exactly, by tests/test_hadronic_xs.cu.
//
// THE GRID: every 1 <= A <= 400 and 0 <= Z <= min(A, 150). That contains every (Z, A) an ion
// beam's ledger can name - a GenericIon is a fragment of a projectile or a target, and no QBBC
// target or beam the port runs is heavier than U238, so a fused or fragmented system stays under
// A = 300 - and it runs past both tables' edges (AME2012's IsInTable stops at A = 273 and
// Z = 110, the theoretical table's at A = 339 and Z = 136), so every level and every boundary
// between two levels is on it.
//
// THE COLUMNS
//   level      1 particle table, 2 AME2012, 3 theoretical, 4 Z == A, 5 Z == 0, 6 formula - read
//              off Geant4's own IsInTable calls in GetNuclearMass's order, not off the port's
//   mass       G4NucleiProperties::GetNuclearMass(A, Z)
//   ion_mass   G4IonTable::GetIonMass(Z, A) - what G4IonTable::CreateIon gives a GenericIon,
//              through GetNucleusMass; it must be the same number, and the column is the proof
//   binding    G4NucleiProperties::GetBindingEnergy(A, Z)
#include <cstdio>

#include "G4IonTable.hh"
#include "G4NucleiProperties.hh"
#include "G4NucleiPropertiesTableAME12.hh"
#include "G4NucleiPropertiesTheoreticalTable.hh"
#include "G4SystemOfUnits.hh"
#include "dump_registry.hh"

namespace {

// `G4NucleiPropertiesTheoreticalTable::IsInTable` is private ("can be used only by
// G4NucleiProperties"), so it is reached the way dump_bic.cc reaches the cascade's members: an
// explicit instantiation may name a private member, and the friend it defines hands the pointer
// out. The AME2012 test is public as G4NucleiProperties::IsInStableTable.
struct TheoInTableTag {
  using type = G4bool (*)(G4int, G4int);
  friend type theo_in_table_ptr(TheoInTableTag);
};
template <typename Tag, typename Tag::type M>
struct NucleiPeek {
  friend typename Tag::type theo_in_table_ptr(Tag) { return M; }
};
template struct NucleiPeek<TheoInTableTag, &G4NucleiPropertiesTheoreticalTable::IsInTable>;

int geant4_level(int a, int z) {
  if (a < 1 || z < 0 || z > a) { return 0; }
  if ((z == 1 && a == 1) || (z == 0 && a == 1) || (z == 1 && a == 2) || (z == 1 && a == 3) ||
      (z == 2 && a == 4) || (z == 2 && a == 3)) {
    return 1;
  }
  if (G4NucleiProperties::IsInStableTable(a, z)) { return 2; }
  if (theo_in_table_ptr(TheoInTableTag{})(z, a)) { return 3; }
  if (z == a) { return 4; }
  if (z == 0) { return 5; }
  return 6;
}

void dump_nuclei_mass(const DumpContext&) {
  FILE* f = std::fopen("nuclei_mass_levels.csv", "w");
  std::fprintf(f, "a,z,level,mass_MeV,ion_mass_MeV,binding_MeV\n");
  G4IonTable* ions = G4IonTable::GetIonTable();
  for (int a = 1; a <= 400; ++a) {
    for (int z = 0; z <= a && z <= 150; ++z) {
      const double m = G4NucleiProperties::GetNuclearMass(a, z) / MeV;
      const double im = ions->GetIonMass(z, a) / MeV;
      const double b = G4NucleiProperties::GetBindingEnergy(a, z) / MeV;
      std::fprintf(f, "%d,%d,%d,%.17g,%.17g,%.17g\n", a, z, geant4_level(a, z), m, im, b);
    }
  }
  std::fclose(f);
}

}  // namespace

G4GPU_REGISTER_DUMP("nuclei_mass", "nuclei_mass_levels.csv", dump_nuclei_mass);
