// Nuclear masses, mass excesses and binding energies - P3's names for G4NucleiProperties.
//
// G4NucleiProperties (11.1.1, source/particles/management) is a four-way dispatch: the six
// light nuclei from the particle table, AME2012, the Moller-Nix theoretical table, and then
// `Z == A`, `Z == 0` and Weizsaecker's formula. De-excitation walks off stability by
// construction - an evaporation chain removes nucleons one at a time from whatever a cascade
// left behind - so all four levels are reachable from here, and P3 transcribed all four in this
// file.
//
// THE TRANSCRIPTION NOW LIVES IN data/nuclei_mass_ame12.hh (P24, docs/RISK.md V238). P2 had a
// second G4NucleiProperties there for the cross sections, with levels 1 and 2 only, its own copy
// of the AME2012 table and a refusal for the rest, and the transport's GenericIon took its mass
// from P2's - so a fragment past the AME edge was refused for a mass this file had all along.
// The code below this comment used to be the four levels; it is now the same functions, moved
// verbatim into `g4gpu::data`, re-exported under the names every caller in the hadronic tree
// already uses. Nothing a de-excitation caller reads has changed: the moved arithmetic is the
// same, except `1.55468*1e-6`, written as Geant4 writes it where P3 had folded it to
// `1.55468e-6` (the two are the same double; asserted in tests/test_deex_nuclear.cu).
#ifndef G4GPU_DEEX_NUCLEAR_MASSES_CUH
#define G4GPU_DEEX_NUCLEAR_MASSES_CUH

#include "data/nuclei_mass_ame12.hh"

namespace g4gpu::deex {

// G4NucleiPropertiesTableAME12
using data::ame12_atomic_mass;
using data::ame12_beta_decay_energy;
using data::ame12_binding_energy;
using data::ame12_electron_mass;
using data::ame12_in_table;
using data::ame12_index;
using data::ame12_mass_excess;
using data::ame12_nuclear_mass;

// G4NucleiPropertiesTheoreticalTable
using data::theo_atomic_mass;
using data::theo_binding_energy;
using data::theo_electronic_binding_energy;
using data::theo_in_table;
using data::theo_index;
using data::theo_mass_excess;
using data::theo_nuclear_mass;

// The particle-table masses and the formula level
using data::formula_atomic_mass;
using data::formula_nuclear_mass;
using data::pdg_mass_alpha;
using data::pdg_mass_deuteron;
using data::pdg_mass_he3;
using data::pdg_mass_neutron;
using data::pdg_mass_proton;
using data::pdg_mass_triton;
using data::weizsaecker_binding_energy;

/// G4NucleiProperties::GetNuclearMass(G4int A, G4int Z). Note the argument order, which is
/// (A, Z) here and (Z, A) in the two table classes - Geant4's, kept so a transcription can be
/// read against the source line by line.
__host__ __device__ inline double nuclear_mass(int A, int Z) { return data::get_nuclear_mass(A, Z); }

/// G4NucleiProperties::GetMassExcess(G4int A, G4int Z).
__host__ __device__ inline double mass_excess(int A, int Z) { return data::get_mass_excess(A, Z); }

/// G4NucleiProperties::GetBindingEnergy(G4int A, G4int Z).
__host__ __device__ inline double binding_energy(int A, int Z) {
  return data::get_binding_energy(A, Z);
}

/// G4NucleiProperties::GetAtomicMass(G4double A, G4double Z), integer arguments.
__host__ __device__ inline double atomic_mass(int A, int Z) { return data::get_atomic_mass(A, Z); }

/// G4NucleiProperties::IsInStableTable - "is this nuclide in AME2012", which is what
/// G4ExcitationHandler and G4Evaporation ask before deciding a fragment is finished.
__host__ __device__ inline bool is_in_stable_table(int A, int Z) {
  return data::is_in_stable_table(A, Z);
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
