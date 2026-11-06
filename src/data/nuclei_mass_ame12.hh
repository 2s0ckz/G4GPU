// G4NucleiProperties - the nuclear mass, all four levels, ONE copy for the whole port.
//
// Transcribed from G4NucleiProperties, G4NucleiPropertiesTableAME12 and
// G4NucleiPropertiesTheoreticalTable (source/particles/management, 11.1.1).
// `G4NucleiProperties::GetNuclearMass(A, Z)` is a four-way dispatch and not a formula:
//
//   1. Z <= 2 and one of (1,1) (0,1) (1,2) (1,3) (2,4) (2,3) -> the PDG mass of proton,
//      neutron, deuteron, triton, alpha, He3, read off the PARTICLE TABLE, not the AME table.
//      The two differ: AME gives the deuteron 1875.612793 MeV and G4Deuteron declares
//      1875.613 MeV, a 0.2 keV choice every emission threshold of P3's de-excitation sees.
//   2. G4NucleiPropertiesTableAME12 - measured, 3353 nuclides, IsInTable narrows to A <= 273,
//      Z <= 110 (data/ame12_masses.hh).
//   3. G4NucleiPropertiesTheoreticalTable - Moller-Nix, 8979 nuclides, Z 8..136, A 16..339
//      (data/nuclei_theoretical.hh).
//   4. otherwise `Z == A` gives A * m_p, `Z == 0` gives A * m_n, and everything else is
//      `G4NucleiProperties::NuclearMass`, Weizsaecker's formula with Geant4's five coefficients
//      and the AME03/AME12 electron correction.
//
// and the mass is floored at zero. Geant4 returns ZERO for A < 1, Z < 0 or Z > A, and for a
// formula mass below zero; for nothing else.
//
// WHY THIS WAS TWO FILES, AND WHY IT IS ONE NOW (P24, docs/RISK.md V238)
//
// P2 wrote this header for the cross sections - two Coulomb barriers need the target's mass,
// `G4NuclearRadii::CoulombFactor` and `G4ComponentGGNuclNuclXsc::ComputeCoulombBarier` - with
// levels 1 and 2 only and its OWN 3353-entry transcription of the AME2012 table, and it refused
// levels 3 and 4 by returning zero, on the argument that a QBBC target is a naturally occurring
// isotope. P3 wrote `deexcitation/nuclear_masses.cuh` for de-excitation, which walks off
// stability by construction, with all four levels and a SECOND transcription of AME2012
// (data/ame12_masses.hh, written by tools/extract_deex_tables.pl). The two AME transcriptions
// were compared entry by entry before P2's was deleted: 3353 Z codes, 296 `shortTable` entries
// and 3353 mass excesses, identical as text and as numbers.
//
// The argument for refusing was right about targets and wrong about the transport. A GenericIon
// is a fragment, and `G4IonTable::CreateIon` gives every one `GetNucleusMass(Z, A)`, i.e. this
// function - so a projectile fragment past the AME edge has the theoretical table's or the
// formula's mass in Geant4, and in the port it had zero, which the stepper refuses by name
// (`kIonWithoutNuclide`: 1 to 9 GenericIons per ion beam, docs/RISK.md V217, V232). One
// implementation now answers every caller: the four levels below are P3's code moved here,
// `deex::` keeps its names through using-declarations, and `data::nuclear_mass<real_t>` is
// Geant4's G4double answer converted once at the end, whatever `real_t` is.
//
// Everything is `double` inside. A nuclear mass is ~2e5 MeV and every emission threshold in P3 is
// a difference of two of them at the 10 eV level - 1e-11 relative.
//
// WHAT IS STILL REFUSED
//
// Only what Geant4 itself answers with zero: `nuclear_mass_known(a, z)` is true exactly when
// `GetNuclearMass` returns a positive mass, and the callers that refuse on it (the ion-ion
// Coulomb barrier, the stepper's GenericIon test) keep their refusals for those cases alone.
// No formula mass is below zero on P24's grid, 1 <= A <= 400 and 0 <= Z <= A
// (`nuclei_mass_levels.csv`, every row compared with Geant4's); A < 1, Z < 0 and Z > A are not
// nuclides.
#ifndef G4GPU_DATA_NUCLEI_MASS_AME12_HH
#define G4GPU_DATA_NUCLEI_MASS_AME12_HH

#include <cmath>

#include "core/units.cuh"
#include "data/ame12_masses.hh"
#include "data/nuclei_theoretical.hh"

namespace g4gpu::data {

// ---------------------------------------------------------------------------------------------
// G4NucleiPropertiesTableAME12
// ---------------------------------------------------------------------------------------------

/// G4NucleiPropertiesTableAME12::GetIndex. Two-level: shortTable[A-1] is the first entry with
/// mass number A, and the scan inside that A is linear over at most ~21 entries.
///
/// Geant4 raises an exception for A > MaxA, A < 1 or Z > A and then returns -1; -1 is returned
/// directly here, because every caller treats -1 as "not in the table" and the next level of
/// G4NucleiProperties is the right answer for those arguments.
///
/// A = 295 returns -1 even though the entry exists. shortTable is declared [MaxA+1] with
/// MaxA = 295 and initialised with 295 values, so shortTable[295] is a language-supplied zero
/// and the scan `for (i = shortTable[294]; i < shortTable[295]; ++i)` is `3352 .. -1`, empty.
/// Reproduced, not fixed. It is the second of two reasons the top of the table is unreachable,
/// and not the one that fires: ame12_in_table() rejects A > 273 and Z > 110 first, and
/// G4NucleiProperties consults nothing without asking IsInTable.
__host__ __device__ inline int ame12_index(int Z, int A) {
  if (A > kAme12MaxA || A < 1 || Z > A) { return -1; }
  const short* st = ame12_short_table();
  const short* zz = ame12_z();
  for (int i = st[A - 1]; i < st[A]; ++i) {
    if (zz[i] == Z) { return i; }
  }
  return -1;
}

/// G4NucleiPropertiesTableAME12::IsInTable.
__host__ __device__ inline bool ame12_in_table(int Z, int A) {
  return (Z <= A && A >= 1 && A <= 273 && Z >= 0 && Z <= 110 && ame12_index(Z, A) >= 0);
}

/// G4NucleiPropertiesTableAME12::GetMassExcess - the tabulated value, keV in the table.
__host__ __device__ inline double ame12_mass_excess(int Z, int A) {
  const int i = ame12_index(Z, A);
  return (i >= 0) ? ame12_mass_excess_keV()[i] * units::keV<double>() : 0.0;
}

/// G4NucleiPropertiesTableAME12::GetAtomicMass = mass excess + A * amu_c2.
__host__ __device__ inline double ame12_atomic_mass(int Z, int A) {
  const int i = ame12_index(Z, A);
  if (i < 0) { return 0.0; }
  return ame12_mass_excess_keV()[i] * units::keV<double>() +
         static_cast<double>(A) * units::amu_c2<double>();
}

/// G4NucleiPropertiesTableAME12::GetBindingEnergy. Entry 0 is the neutron's mass excess and
/// entry 1 the hydrogen atom's - the table's own first two rows, used as constants.
__host__ __device__ inline double ame12_binding_energy(int Z, int A) {
  const int i = ame12_index(Z, A);
  if (i < 0) { return 0.0; }
  const double* me = ame12_mass_excess_keV();
  return (static_cast<double>(A - Z) * me[0] + static_cast<double>(Z) * me[1] - me[i]) *
         units::keV<double>();
}

/// G4NucleiPropertiesTableAME12::GetBetaDecayEnergy.
__host__ __device__ inline double ame12_beta_decay_energy(int Z, int A) {
  const int i = ame12_index(Z, A);
  return (i >= 0) ? ame12_beta_energy_keV()[i] * units::keV<double>() : 0.0;
}

/// The electron mass in orbit G4NucleiPropertiesTableAME12::GetNuclearMass subtracts: Z * m_e
/// less the electronic binding, the two-term fit of AME03/AME12. Geant4 tabulates it once per Z
/// in a thread-local array (`electronMass[iz]`, iz from 1, so entry 0 keeps its static zero)
/// with std::pow, not G4Pow::powZ, and writes the second coefficient `1.55468*1e-6`.
__host__ __device__ inline double ame12_electron_mass(int Z) {
  if (Z <= 0) { return 0.0; }
  return static_cast<double>(Z) * units::electron_mass_c2<double>() -
         (14.4381 * std::pow(static_cast<double>(Z), 2.39)) * units::eV<double>() -
         (1.55468 * 1e-6 * std::pow(static_cast<double>(Z), 5.35)) * units::eV<double>();
}

/// G4NucleiPropertiesTableAME12::GetNuclearMass - atomic mass less the bound electrons.
__host__ __device__ inline double ame12_nuclear_mass(int Z, int A) {
  const double m = ame12_atomic_mass(Z, A) - ame12_electron_mass(Z);
  return (m < 0.0) ? 0.0 : m;
}

// ---------------------------------------------------------------------------------------------
// G4NucleiPropertiesTheoreticalTable - P.Moller, J.R.Nix, W.D.Myers, W.J.Swiatecki,
// At. Data Nucl. Data Tables 59 (1995) 185.
// ---------------------------------------------------------------------------------------------

/// G4NucleiPropertiesTheoreticalTable::GetIndex. Indexed by Z, scanned over A - the opposite
/// way round from the AME table.
__host__ __device__ inline int theo_index(int Z, int A) {
  if (A > 339 || A < 16 || Z > 136 || Z < 8 || Z > A) { return -1; }
  const short* st = theo_short_table();
  const short* aa = theo_a();
  for (int i = st[Z - 8]; i < st[Z - 8 + 1]; ++i) {
    if (aa[i] == A) { return i; }
  }
  return -1;
}

/// G4NucleiPropertiesTheoreticalTable::IsInTable.
__host__ __device__ inline bool theo_in_table(int Z, int A) {
  return (Z <= A && A >= 16 && A <= 339 && Z <= 136 && Z >= 8 && theo_index(Z, A) >= 0);
}

/// The table is in MeV, unlike the AME one which is in keV.
__host__ __device__ inline double theo_mass_excess(int Z, int A) {
  const int i = theo_index(Z, A);
  return (i >= 0) ? theo_mass_excess_MeV()[i] * units::MeV<double>() : 0.0;
}

/// G4NucleiPropertiesTheoreticalTable::GetBindingEnergy. Its own two constants, 7.289034 and
/// 8.071431 MeV, which are the AME table's first two rows rounded to six decimals - a
/// difference of ~0.1 eV that is Geant4's and is kept.
__host__ __device__ inline double theo_binding_energy(int Z, int A) {
  const int i = theo_index(Z, A);
  if (i < 0) { return 0.0; }
  const double Mh = 7.289034 * units::MeV<double>();
  const double Mn = 8.071431 * units::MeV<double>();
  return static_cast<double>(Z) * Mh + static_cast<double>(A - Z) * Mn -
         theo_mass_excess_MeV()[i] * units::MeV<double>();
}

/// G4NucleiPropertiesTheoreticalTable::GetAtomicMass.
__host__ __device__ inline double theo_atomic_mass(int Z, int A) {
  const int i = theo_index(Z, A);
  if (i < 0) { return 0.0; }
  return theo_mass_excess_MeV()[i] * units::MeV<double>() +
         static_cast<double>(A) * units::amu_c2<double>();
}

/// G4NucleiPropertiesTheoreticalTable::ElectronicBindingEnergy - a one-term fit, and a
/// different one from the AME table's two-term form above.
__host__ __device__ inline double theo_electronic_binding_energy(int Z) {
  const double ael = 1.433e-5 * units::MeV<double>();
  return ael * std::pow(static_cast<double>(Z), 2.39);
}

/// G4NucleiPropertiesTheoreticalTable::GetNuclearMass.
__host__ __device__ inline double theo_nuclear_mass(int Z, int A) {
  const int i = theo_index(Z, A);
  if (i < 0) { return 0.0; }
  return theo_atomic_mass(Z, A) - static_cast<double>(Z) * units::electron_mass_c2<double>() +
         theo_electronic_binding_energy(Z);
}

// ---------------------------------------------------------------------------------------------
// The six light nuclei, from the particle table
// ---------------------------------------------------------------------------------------------

/// The masses G4NucleiProperties::GetNuclearMass reads off the particle table when Z <= 2:
/// G4Proton.cc and G4Neutron.cc (CLHEP proton_mass_c2 / neutron_mass_c2), and G4Deuteron.cc
/// 1.875613 GeV, G4Triton.cc 2.808921 GeV, G4Alpha.cc 3.727379 GeV, G4He3.cc 2.808391 GeV.
/// Written as `x.yz * GeV`, as Geant4 writes them, and not as the MeV literal: see the note over
/// the particle list in physics/hadronic/xs/projectile.cuh for the kaon that cost 1.4e-12 by
/// being written the other way. These four happen to be the same double either way.
template <typename real_t> __host__ __device__ constexpr real_t deuteron_mass_c2() {
  return real_t(1.875613) * units::GeV<real_t>();
}
template <typename real_t> __host__ __device__ constexpr real_t triton_mass_c2() {
  return real_t(2.808921) * units::GeV<real_t>();
}
template <typename real_t> __host__ __device__ constexpr real_t alpha_mass_c2() {
  return real_t(3.727379) * units::GeV<real_t>();
}
template <typename real_t> __host__ __device__ constexpr real_t he3_mass_c2() {
  return real_t(2.808391) * units::GeV<real_t>();
}

__host__ __device__ inline constexpr double pdg_mass_proton() {
  return units::proton_mass_c2<double>();
}
__host__ __device__ inline constexpr double pdg_mass_neutron() {
  return units::neutron_mass_c2<double>();
}
__host__ __device__ inline constexpr double pdg_mass_deuteron() {
  return deuteron_mass_c2<double>();
}
__host__ __device__ inline constexpr double pdg_mass_triton() { return triton_mass_c2<double>(); }
__host__ __device__ inline constexpr double pdg_mass_he3() { return he3_mass_c2<double>(); }
__host__ __device__ inline constexpr double pdg_mass_alpha() { return alpha_mass_c2<double>(); }

// ---------------------------------------------------------------------------------------------
// G4NucleiProperties - the formula, and the dispatch over all four levels
// ---------------------------------------------------------------------------------------------

/// G4NucleiProperties::BindingEnergy - Weizsaecker's formula with Geant4's coefficients.
/// Returned negative of the binding, as Geant4 does, because AtomicMass subtracts it.
///
/// std::pow and not G4Pow: this function is in source/particles and does not include G4Pow.
__host__ __device__ inline double weizsaecker_binding_energy(double A, double Z) {
  const int Npairing = static_cast<int>(A - Z) % 2;
  const int Zpairing = static_cast<int>(Z) % 2;
  double binding = -15.67 * A                                     // nuclear volume
                   + 17.23 * std::pow(A, 2.0 / 3.0)               // surface energy
                   + 93.15 * ((A / 2. - Z) * (A / 2. - Z)) / A    // asymmetry
                   + 0.6984523 * Z * Z * std::pow(A, -1.0 / 3.0); // coulomb
  if (Npairing == Zpairing) {
    binding += (Npairing + Zpairing - 1) * 12.0 / std::sqrt(A);   // pairing
  }
  return -binding * units::MeV<double>();
}

/// G4NucleiProperties::AtomicMass, the formula level. Reads the neutron and hydrogen mass
/// excesses through G4NucleiPropertiesTableAME12::GetMassExcess(0,1) and (1,1), which are the
/// table's first two rows.
__host__ __device__ inline double formula_atomic_mass(double A, double Z) {
  const double* me = ame12_mass_excess_keV();
  const double hydrogen_mass_excess = me[1] * units::keV<double>();
  const double neutron_mass_excess = me[0] * units::keV<double>();
  return (A - Z) * neutron_mass_excess + Z * hydrogen_mass_excess -
         weizsaecker_binding_energy(A, Z) + A * units::amu_c2<double>();
}

/// G4NucleiProperties::NuclearMass, the formula level - atomic mass converted to nuclear mass by
/// the AME03/AME12 electron formula. Here the two fit terms are ADDED; in
/// G4NucleiPropertiesTableAME12::GetNuclearMass they are subtracted from Z*m_e and the whole
/// subtracted, which is the same arithmetic written twice. `1.55468*1e-6` as Geant4 writes it.
__host__ __device__ inline double formula_nuclear_mass(double A, double Z) {
  if (A < 1 || Z < 0 || Z > A) { return 0.0; }
  double mass = formula_atomic_mass(A, Z);
  mass -= Z * units::electron_mass_c2<double>();
  mass += (14.4381 * std::pow(Z, 2.39) + 1.55468 * 1e-6 * std::pow(Z, 5.35)) *
          units::eV<double>();
  return mass;
}

/// Which of GetNuclearMass's levels answers for (A, Z). Not a Geant4 type: it is what
/// `nuclei_mass_levels.csv` records beside the mass, so that a disagreement names its level.
enum class NuclearMassLevel : int {
  kInvalid = 0,    ///< A < 1, Z < 0 or Z > A - Geant4 returns 0
  kLight = 1,      ///< the six particle-table masses
  kAme12 = 2,      ///< G4NucleiPropertiesTableAME12
  kTheoretical = 3,  ///< G4NucleiPropertiesTheoreticalTable
  kAllProtons = 4,   ///< Z == A: A * proton mass
  kAllNeutrons = 5,  ///< Z == 0: A * neutron mass
  kFormula = 6       ///< G4NucleiProperties::NuclearMass, Weizsaecker
};

__host__ __device__ inline NuclearMassLevel nuclear_mass_level(int A, int Z) {
  if (A < 1 || Z < 0 || Z > A) { return NuclearMassLevel::kInvalid; }
  if ((Z == 1 && A == 1) || (Z == 0 && A == 1) || (Z == 1 && A == 2) || (Z == 1 && A == 3) ||
      (Z == 2 && A == 4) || (Z == 2 && A == 3)) {
    return NuclearMassLevel::kLight;
  }
  if (ame12_in_table(Z, A)) { return NuclearMassLevel::kAme12; }
  if (theo_in_table(Z, A)) { return NuclearMassLevel::kTheoretical; }
  if (Z == A) { return NuclearMassLevel::kAllProtons; }
  if (Z == 0) { return NuclearMassLevel::kAllNeutrons; }
  return NuclearMassLevel::kFormula;
}

/// G4NucleiProperties::GetNuclearMass(G4int A, G4int Z), MeV, in Geant4's G4double arithmetic.
/// The argument order is (A, Z), Geant4's here and the reverse of the two tables'.
__host__ __device__ inline double get_nuclear_mass(int A, int Z) {
  if (A < 1 || Z < 0 || Z > A) { return 0.0; }

  double mass = -1.0;
  if (Z <= 2) {
    if (Z == 1 && A == 1) { mass = pdg_mass_proton(); }
    else if (Z == 0 && A == 1) { mass = pdg_mass_neutron(); }
    else if (Z == 1 && A == 2) { mass = pdg_mass_deuteron(); }
    else if (Z == 1 && A == 3) { mass = pdg_mass_triton(); }
    else if (Z == 2 && A == 4) { mass = pdg_mass_alpha(); }
    else if (Z == 2 && A == 3) { mass = pdg_mass_he3(); }
  }
  if (mass < 0.0) {
    if (ame12_in_table(Z, A)) {
      mass = ame12_nuclear_mass(Z, A);
    } else if (theo_in_table(Z, A)) {
      mass = theo_nuclear_mass(Z, A);
    } else if (Z == A) {
      mass = A * pdg_mass_proton();
    } else if (Z == 0) {
      mass = A * pdg_mass_neutron();
    } else {
      mass = formula_nuclear_mass(static_cast<double>(A), static_cast<double>(Z));
    }
  }
  return (mass < 0.0) ? 0.0 : mass;
}

/// G4NucleiProperties::GetMassExcess(G4int A, G4int Z).
__host__ __device__ inline double get_mass_excess(int A, int Z) {
  if (A < 1 || Z < 0 || Z > A) { return 0.0; }
  if (ame12_in_table(Z, A)) { return ame12_mass_excess(Z, A); }
  if (theo_in_table(Z, A)) { return theo_mass_excess(Z, A); }
  // G4NucleiProperties::MassExcess - atomic mass less A amu, both from the formula level.
  return formula_atomic_mass(static_cast<double>(A), static_cast<double>(Z)) -
         static_cast<double>(A) * units::amu_c2<double>();
}

/// G4NucleiProperties::GetBindingEnergy(G4int A, G4int Z).
__host__ __device__ inline double get_binding_energy(int A, int Z) {
  if (A < 1 || Z < 0 || Z > A) { return 0.0; }
  if (ame12_in_table(Z, A)) { return ame12_binding_energy(Z, A); }
  if (theo_in_table(Z, A)) { return theo_binding_energy(Z, A); }
  return weizsaecker_binding_energy(static_cast<double>(A), static_cast<double>(Z));
}

/// G4NucleiProperties::GetAtomicMass(G4double A, G4double Z), integer arguments.
__host__ __device__ inline double get_atomic_mass(int A, int Z) {
  if (A < 1 || Z < 0 || Z > A) { return 0.0; }
  if (ame12_in_table(Z, A)) { return ame12_atomic_mass(Z, A); }
  if (theo_in_table(Z, A)) { return theo_atomic_mass(Z, A); }
  return formula_atomic_mass(static_cast<double>(A), static_cast<double>(Z));
}

/// G4NucleiProperties::IsInStableTable - "is this nuclide in AME2012".
__host__ __device__ inline bool is_in_stable_table(int A, int Z) {
  if (A < 1 || Z < 0 || Z > A) { return false; }
  return ame12_in_table(Z, A);
}

/// The AME2012 membership test under its P2 name, which the CHIPS photo- and electro-nuclear
/// cross sections ask.
namespace ame12 {
__host__ __device__ inline bool is_in_table(int z, int a) { return ame12_in_table(z, a); }
}  // namespace ame12

/// True when G4NucleiProperties::GetNuclearMass(A, Z) returns a mass - a positive one.
///
/// Until P24 this meant "levels 1 or 2", and a GenericIon past the AME edge was refused for want
/// of a mass Geant4 has. Now it is false only where Geant4's own answer is zero.
__host__ __device__ inline bool nuclear_mass_known(int a, int z) {
  return get_nuclear_mass(a, z) > 0.0;
}

/// G4NucleiProperties::GetNuclearMass(A, Z), MeV, as `real_t`. Computed in double - Geant4's
/// G4double - and converted once.
template <typename real_t>
__host__ __device__ inline real_t nuclear_mass(int a, int z) {
  return static_cast<real_t>(get_nuclear_mass(a, z));
}

}  // namespace g4gpu::data

#endif  // G4GPU_DATA_NUCLEI_MASS_AME12_HH
