// Nuclear masses, mass excesses and binding energies.
//
// Transcribed from G4NucleiProperties (11.1.1, source/particles/management), which is a
// four-way dispatch and not a formula:
//
//   1. six light nuclei - p, n, d, t, He3, alpha - take the PDG mass off the particle table,
//      NOT the AME2012 value. The two differ: AME2012 gives the deuteron 1875.612793 MeV and
//      G4Deuteron declares 1875.613 MeV, so the choice is worth 0.2 keV and is visible at the
//      tolerance every emission threshold in this module is compared at.
//   2. AME2012, if the nuclide is in it (G4NucleiPropertiesTableAME12, 3353 entries).
//   3. the Moller-Nix theoretical table, if it is in that (8979 entries, Z 8..136, A 16..339).
//   4. otherwise Z == A gives A*m_p, Z == 0 gives A*m_n, and everything else is Weizsaecker's
//      semi-empirical formula with Geant4's own five coefficients.
//
// De-excitation walks off stability by construction - an evaporation chain removes nucleons
// one at a time from whatever a cascade left behind - so all four branches are reachable and
// all four are here. Branch 4 in particular is not a fallback that never fires: any (Z, A)
// with Z < 8 outside AME2012, and anything past A = 339, lands in it.
//
// Everything is `double`, not the project's `real_t` template. A nuclear mass is ~2e5 MeV and
// the quantity every threshold in this module tests is a difference of two of them at the
// 10 eV level - 1e-11 relative. Single precision has 1e-7, so a float instantiation would not
// be a less accurate version of this file, it would be one in which no emission threshold
// works at all. The type is therefore fixed rather than parameterised.
#ifndef G4GPU_DEEX_NUCLEAR_MASSES_CUH
#define G4GPU_DEEX_NUCLEAR_MASSES_CUH

#include <cmath>

#include "core/units.cuh"
#include "data/ame12_masses.hh"
#include "data/nuclei_theoretical.hh"

namespace g4gpu::deex {

namespace u = g4gpu::units;

// ---------------------------------------------------------------------------------------------
// G4NucleiPropertiesTableAME12
// ---------------------------------------------------------------------------------------------

/// G4NucleiPropertiesTableAME12::GetIndex. Two-level: shortTable[A-1] is the first entry with
/// mass number A, and the scan inside that A is linear over at most ~21 entries.
///
/// Geant4 raises an exception for A > MaxA, A < 1 or Z > A and then returns -1; -1 is returned
/// directly here, because every caller in this module already treats -1 as "not in the table"
/// and the next branch of G4NucleiProperties is the right answer for those arguments.
///
/// A = 295 returns -1 even though the entry exists. shortTable is declared [MaxA+1] with
/// MaxA = 295 and initialised with 295 values, so shortTable[295] is a language-supplied zero
/// and the scan `for (i = shortTable[294]; i < shortTable[295]; ++i)` is `3352 .. -1`, empty.
/// Reproduced, not fixed: this port's job is to return Geant4's number.
///
/// It is the second of two reasons the top of the table is unreachable, and not the one that
/// fires: ame12_in_table() below rejects A > 273 and Z > 110 first, and G4NucleiProperties
/// consults nothing without asking IsInTable. So AME2012's 3353 entries are 3353 rows of which
/// the ones above A = 273 are dead in Geant4, and a fragment there gets the theoretical table
/// or Weizsaecker. Both cut-offs are here because both are Geant4's and neither is implied by
/// the other.
__host__ __device__ inline int ame12_index(int Z, int A) {
  if (A > data::kAme12MaxA || A < 1 || Z > A) { return -1; }
  const short* st = data::ame12_short_table();
  const short* zz = data::ame12_z();
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
  return (i >= 0) ? data::ame12_mass_excess_keV()[i] * u::keV<double>() : 0.0;
}

/// G4NucleiPropertiesTableAME12::GetAtomicMass = mass excess + A * amu_c2.
__host__ __device__ inline double ame12_atomic_mass(int Z, int A) {
  const int i = ame12_index(Z, A);
  if (i < 0) { return 0.0; }
  return data::ame12_mass_excess_keV()[i] * u::keV<double>() +
         static_cast<double>(A) * u::amu_c2<double>();
}

/// G4NucleiPropertiesTableAME12::GetBindingEnergy. Entry 0 is the neutron's mass excess and
/// entry 1 the hydrogen atom's - the table's own first two rows, used as constants.
__host__ __device__ inline double ame12_binding_energy(int Z, int A) {
  const int i = ame12_index(Z, A);
  if (i < 0) { return 0.0; }
  const double* me = data::ame12_mass_excess_keV();
  return (static_cast<double>(A - Z) * me[0] + static_cast<double>(Z) * me[1] - me[i]) *
         u::keV<double>();
}

/// G4NucleiPropertiesTableAME12::GetBetaDecayEnergy.
__host__ __device__ inline double ame12_beta_decay_energy(int Z, int A) {
  const int i = ame12_index(Z, A);
  return (i >= 0) ? data::ame12_beta_energy_keV()[i] * u::keV<double>() : 0.0;
}

/// G4NucleiPropertiesTableAME12::GetNuclearMass - atomic mass less the bound electrons.
///
/// The electron mass is not Z*m_e: it is Z*m_e minus the electronic binding energy of the
/// atom, with the two-term fit of AME03/AME12. Geant4 tabulates it once per Z in a
/// thread-local array using std::pow, not G4Pow::powZ, so std::pow is what is used here.
__host__ __device__ inline double ame12_electron_mass(int Z) {
  if (Z <= 0) { return 0.0; }
  return static_cast<double>(Z) * u::electron_mass_c2<double>() -
         (14.4381 * std::pow(static_cast<double>(Z), 2.39)) * u::eV<double>() -
         (1.55468e-6 * std::pow(static_cast<double>(Z), 5.35)) * u::eV<double>();
}

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
  const short* st = data::theo_short_table();
  const short* aa = data::theo_a();
  for (int i = st[Z - 8]; i < st[Z - 8 + 1]; ++i) {
    if (aa[i] == A) { return i; }
  }
  return -1;
}

__host__ __device__ inline bool theo_in_table(int Z, int A) {
  return (Z <= A && A >= 16 && A <= 339 && Z <= 136 && Z >= 8 && theo_index(Z, A) >= 0);
}

/// The table is in MeV, unlike the AME one which is in keV.
__host__ __device__ inline double theo_mass_excess(int Z, int A) {
  const int i = theo_index(Z, A);
  return (i >= 0) ? data::theo_mass_excess_MeV()[i] * u::MeV<double>() : 0.0;
}

/// G4NucleiPropertiesTheoreticalTable::GetBindingEnergy. Its own two constants, 7.289034 and
/// 8.071431 MeV, which are the AME table's first two rows rounded to six decimals - a
/// difference of ~0.1 eV that is Geant4's and is kept.
__host__ __device__ inline double theo_binding_energy(int Z, int A) {
  const int i = theo_index(Z, A);
  if (i < 0) { return 0.0; }
  const double Mh = 7.289034 * u::MeV<double>();
  const double Mn = 8.071431 * u::MeV<double>();
  return static_cast<double>(Z) * Mh + static_cast<double>(A - Z) * Mn -
         data::theo_mass_excess_MeV()[i] * u::MeV<double>();
}

__host__ __device__ inline double theo_atomic_mass(int Z, int A) {
  const int i = theo_index(Z, A);
  if (i < 0) { return 0.0; }
  return data::theo_mass_excess_MeV()[i] * u::MeV<double>() +
         static_cast<double>(A) * u::amu_c2<double>();
}

/// G4NucleiPropertiesTheoreticalTable::ElectronicBindingEnergy - a one-term fit, and a
/// different one from the AME table's two-term form above.
__host__ __device__ inline double theo_electronic_binding_energy(int Z) {
  const double ael = 1.433e-5 * u::MeV<double>();
  return ael * std::pow(static_cast<double>(Z), 2.39);
}

__host__ __device__ inline double theo_nuclear_mass(int Z, int A) {
  const int i = theo_index(Z, A);
  if (i < 0) { return 0.0; }
  return theo_atomic_mass(Z, A) - static_cast<double>(Z) * u::electron_mass_c2<double>() +
         theo_electronic_binding_energy(Z);
}

// ---------------------------------------------------------------------------------------------
// G4NucleiProperties - the dispatch, and the formula branch under it.
// ---------------------------------------------------------------------------------------------

/// The six PDG masses G4NucleiProperties reads off the particle table. From G4Proton.cc,
/// G4Neutron.cc (CLHEP proton_mass_c2 / neutron_mass_c2) and G4Deuteron.cc, G4Triton.cc,
/// G4He3.cc, G4Alpha.cc, which declare theirs literally in GeV.
__host__ __device__ inline constexpr double pdg_mass_proton() {
  return u::proton_mass_c2<double>();
}
__host__ __device__ inline constexpr double pdg_mass_neutron() {
  return u::neutron_mass_c2<double>();
}
__host__ __device__ inline constexpr double pdg_mass_deuteron() { return 1.875613 * u::GeV<double>(); }
__host__ __device__ inline constexpr double pdg_mass_triton() { return 2.808921 * u::GeV<double>(); }
__host__ __device__ inline constexpr double pdg_mass_he3() { return 2.808391 * u::GeV<double>(); }
__host__ __device__ inline constexpr double pdg_mass_alpha() { return 3.727379 * u::GeV<double>(); }

/// G4NucleiProperties::BindingEnergy - Weizsaecker's formula with Geant4's coefficients.
/// Returned negative of the binding, as Geant4 does, because AtomicMass subtracts it.
///
/// std::pow and not G4Pow: this function is in source/particles and does not include G4Pow.
/// The 2/3 and -1/3 powers are therefore exact ones, unlike everywhere else in this module.
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
  return -binding * u::MeV<double>();
}

/// G4NucleiProperties::AtomicMass, the formula branch. Reads the neutron and hydrogen mass
/// excesses straight out of the AME table rather than from the class constants.
__host__ __device__ inline double formula_atomic_mass(double A, double Z) {
  const double* me = data::ame12_mass_excess_keV();
  const double hydrogen_mass_excess = me[1] * u::keV<double>();
  const double neutron_mass_excess = me[0] * u::keV<double>();
  return (A - Z) * neutron_mass_excess + Z * hydrogen_mass_excess -
         weizsaecker_binding_energy(A, Z) + A * u::amu_c2<double>();
}

/// G4NucleiProperties::NuclearMass, the formula branch - atomic mass converted to nuclear mass
/// by the AME03/AME12 electron formula. Note the sign: here the two fit terms are ADDED, in
/// G4NucleiPropertiesTableAME12::GetNuclearMass they are subtracted from Z*m_e and the whole
/// thing subtracted, which is the same arithmetic written twice.
__host__ __device__ inline double formula_nuclear_mass(double A, double Z) {
  if (A < 1 || Z < 0 || Z > A) { return 0.0; }
  double mass = formula_atomic_mass(A, Z);
  mass -= Z * u::electron_mass_c2<double>();
  mass += (14.4381 * std::pow(Z, 2.39) + 1.55468e-6 * std::pow(Z, 5.35)) * u::eV<double>();
  return mass;
}

/// G4NucleiProperties::GetNuclearMass(G4int A, G4int Z). The dispatch.
__host__ __device__ inline double nuclear_mass(int A, int Z) {
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

/// G4NucleiProperties::GetMassExcess(G4int A, G4int Z). Note the argument order, which is
/// (A, Z) here and (Z, A) in the two table classes - Geant4's, kept so a transcription can be
/// read against the source line by line.
__host__ __device__ inline double mass_excess(int A, int Z) {
  if (A < 1 || Z < 0 || Z > A) { return 0.0; }
  if (ame12_in_table(Z, A)) { return ame12_mass_excess(Z, A); }
  if (theo_in_table(Z, A)) { return theo_mass_excess(Z, A); }
  // G4NucleiProperties::MassExcess - atomic mass less A amu, both from the formula branch.
  return formula_atomic_mass(static_cast<double>(A), static_cast<double>(Z)) -
         static_cast<double>(A) * u::amu_c2<double>();
}

/// G4NucleiProperties::GetBindingEnergy(G4int A, G4int Z).
__host__ __device__ inline double binding_energy(int A, int Z) {
  if (A < 1 || Z < 0 || Z > A) { return 0.0; }
  if (ame12_in_table(Z, A)) { return ame12_binding_energy(Z, A); }
  if (theo_in_table(Z, A)) { return theo_binding_energy(Z, A); }
  return weizsaecker_binding_energy(static_cast<double>(A), static_cast<double>(Z));
}

/// G4NucleiProperties::GetAtomicMass(G4double A, G4double Z), integer arguments.
__host__ __device__ inline double atomic_mass(int A, int Z) {
  if (A < 1 || Z < 0 || Z > A) { return 0.0; }
  if (ame12_in_table(Z, A)) { return ame12_atomic_mass(Z, A); }
  if (theo_in_table(Z, A)) { return theo_atomic_mass(Z, A); }
  return formula_atomic_mass(static_cast<double>(A), static_cast<double>(Z));
}

/// G4NucleiProperties::IsInStableTable - "is this nuclide in AME2012", which is what
/// G4ExcitationHandler and G4Evaporation ask before deciding a fragment is finished.
__host__ __device__ inline bool is_in_stable_table(int A, int Z) {
  if (A < 1 || Z < 0 || Z > A) { return false; }
  return ame12_in_table(Z, A);
}

/// G4Fragment::ComputeGroundStateMass, the nLambdas <= 0 branch: nothing but
/// G4NucleiProperties::GetNuclearMass(A, Z). Written as a named function anyway because every
/// threshold in this module is a difference of ground-state masses and reading it as
/// "ground state mass" rather than "nuclear_mass with the arguments the other way round" is
/// worth one line of indirection.
///
/// Hyper-fragments (nLambdas > 0) go to G4HyperNucleiProperties and are refused: see
/// excitation_handler.cuh, which reports them by name.
__host__ __device__ inline double ground_state_mass(int Z, int A) {
  return (A <= 0) ? 0.0 : nuclear_mass(A, Z);
}

}  // namespace g4gpu::deex

#endif
