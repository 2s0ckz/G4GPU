// Pairing corrections, shell corrections and the nuclear level-density parameter.
//
// Transcribed from G4PairingCorrection, G4ShellCorrection, the five G4Cameron*/G4Cook* table
// classes under de_excitation/util, G4NuclearLevelData::GetLevelDensity /
// GetPairingCorrection, G4EvaporationLevelDensityParameter and the level-density block of the
// G4LevelManager constructor (11.1.1).
//
// Two things about this layer are worth reading before the code:
//
// **The tables are windows, and the windows are the choice.** Every G4Cameron*/G4Cook* class
// has the same shape - a Z table plus an N table, summed - and differs only in the (Z, N)
// rectangle it claims. So `G4ShellCorrection::GetShellCorrection` is not "look up the shell
// correction", it is "Cook if Cook has it, else Cameron-Gilbert if it has it, else zero", and
// the answer changes discontinuously at Z = 28 and Z = 96. Same for pairing, where the fall
// through is not zero but a formula.
//
// **With 11.1.1's defaults the level density is not a table at all.** `fLD` defaults to true,
// and G4NuclearLevelData::GetLevelDensity then returns `A * 0.075/MeV` without ever asking the
// level manager. The A > 20 four-way parameterisation in the G4LevelManager constructor - the
// one that reads a shell correction - is dead code in the default configuration. It is ported
// because the flag exists and because dumping both is the only way to show which one Geant4
// ran; deex_params().level_density_flag picks between them exactly where Geant4 does.
#ifndef G4GPU_DEEX_CORRECTIONS_CUH
#define G4GPU_DEEX_CORRECTIONS_CUH

#include <cmath>

#include "core/units.cuh"
#include "data/g4pow.hh"
#include "data/shell_pairing.hh"
#include "physics/hadronic/deexcitation/deex_params.cuh"

namespace g4gpu::deex {

namespace u = g4gpu::units;

// ---------------------------------------------------------------------------------------------
// The five table classes. Each returns false and leaves `result` untouched outside its window,
// which is what makes the fall-through chains below readable as chains.
// ---------------------------------------------------------------------------------------------

/// G4CameronGilbertPairingCorrections: Z = 11..98, N = 11..150.
/// A.Gilbert and A.G.W.Cameron, Can. J. Phys. 43 (1965) 1446.
__host__ __device__ inline bool cameron_gilbert_pairing(int N, int Z, double& result) {
  if (Z >= 11 && Z <= 98 && N >= 11 && N <= 150) {
    result = data::cameron_gilbert_pairing_corrections_z()[Z - 11] +
             data::cameron_gilbert_pairing_corrections_n()[N - 11];
    return true;
  }
  return false;
}

/// G4CameronGilbertShellCorrections: Z = 11..98, N = 11..150.
__host__ __device__ inline bool cameron_gilbert_shell(int N, int Z, double& result) {
  if (Z >= 11 && Z <= 98 && N >= 11 && N <= 150) {
    result = data::cameron_gilbert_shell_corrections_z()[Z - 11] +
             data::cameron_gilbert_shell_corrections_n()[N - 11];
    return true;
  }
  return false;
}

/// G4CookShellCorrections: Z = 28..95, N = 33..150.
/// J.L.Cook, H.Ferguson, A.R.de L.Musgrove, Aust. J. Phys. 20 (1967) 477.
__host__ __device__ inline bool cook_shell(int N, int Z, double& result) {
  if (Z >= 28 && Z <= 95 && N >= 33 && N <= 150) {
    result = data::cook_shell_corrections_z()[Z - 28] +
             data::cook_shell_corrections_n()[N - 33];
    return true;
  }
  return false;
}

/// G4CookPairingCorrections: Z = 28..95, N = 33..150. Constructed by nothing in the default
/// configuration - G4PairingCorrection holds Cameron-Gilbert and Cameron shell-plus-pairing,
/// not this - and here so that the set of tables is complete rather than the set that is read.
__host__ __device__ inline bool cook_pairing(int N, int Z, double& result) {
  if (Z >= 28 && Z <= 95 && N >= 33 && N <= 150) {
    result = data::cook_pairing_corrections_z()[Z - 28] +
             data::cook_pairing_corrections_n()[N - 33];
    return true;
  }
  return false;
}

/// G4CameronTruranHilfShellCorrections: Z = 10..102, N = 10..155.
__host__ __device__ inline bool cameron_truran_hilf_shell(int N, int Z, double& result) {
  if (Z >= 10 && Z <= 102 && N >= 10 && N <= 155) {
    result = data::cameron_truran_hilf_shell_corrections_z()[Z - 10] +
             data::cameron_truran_hilf_shell_corrections_n()[N - 10];
    return true;
  }
  return false;
}

/// G4CameronTruranHilfPairingCorrections: Z = 10..102, N = 10..155. Its N table is declared
/// [146] and initialised with 145 values, so N = 155 reads a zero the language supplied; see
/// tools/extract_deex_tables.pl, which reproduces the zero deliberately.
__host__ __device__ inline bool cameron_truran_hilf_pairing(int N, int Z, double& result) {
  if (Z >= 10 && Z <= 102 && N >= 10 && N <= 155) {
    result = data::cameron_truran_hilf_pairing_corrections_z()[Z - 10] +
             data::cameron_truran_hilf_pairing_corrections_n()[N - 10];
    return true;
  }
  return false;
}

/// G4CameronShellPlusPairingCorrections: 1..200 in both, and the bounds test is `<=` only -
/// Geant4 does not check for Z or N below 1, so `SPZTable[Z-1]` reads before the array for
/// Z = 0. Guarded here at 1, which changes the answer for exactly the arguments on which
/// Geant4's is undefined.
/// A.G.W.Cameron, Can. J. Phys. 35 (1957) 1021, Table 1.
__host__ __device__ inline bool cameron_shell_plus_pairing(int N, int Z, double& result) {
  if (Z >= 1 && Z <= 200 && N >= 1 && N <= 200) {
    result = data::cameron_shell_plus_pairing_corrections_z()[Z - 1] +
             data::cameron_shell_plus_pairing_corrections_n()[N - 1];
    return true;
  }
  return false;
}

// ---------------------------------------------------------------------------------------------
// G4PairingCorrection and G4ShellCorrection - the dispatchers.
// ---------------------------------------------------------------------------------------------

/// G4PairingCorrection::GetPairingCorrection(A, Z). Cameron-Gilbert inside its window;
/// otherwise 12 MeV / sqrt(A) times the number of unpaired-nucleon groups, clamped at zero.
///
/// The `(1 - Z + 2*(Z/2))` is integer division: 1 for even Z, 0 for odd. So the multiplier is
/// 2 for even-even, 1 for odd-A and 0 for odd-odd, which is the usual pairing term written
/// without a conditional.
///
/// The final `max(., 0)` is DEAD CODE and is known to be: every value in both Cameron-Gilbert
/// pairing tables is >= 0 (the minimum of each is exactly 0.00) and the fall-through formula's
/// multiplier is 0, 1 or 2, so neither branch can be negative. Removing the clamp does not
/// change one of the 18,407 answers tests/test_deex_nuclear.cu compares - that was measured,
/// not assumed. It is kept because it is Geant4's, and it is labelled because an assertion
/// that cannot fail is not an assertion: nothing here validates the clamp.
__host__ __device__ inline double pairing_correction(int A, int Z) {
  double pair_corr = 0.0;
  const int N = A - Z;
  if (!cameron_gilbert_pairing(N, Z, pair_corr)) {
    const double kPairingConstant = 12.0 * u::MeV<double>();
    pair_corr = ((1 - Z + 2 * (Z / 2)) + (1 - N + 2 * (N / 2))) * kPairingConstant /
                std::sqrt(static_cast<double>(A));
  }
  return (pair_corr > 0.0) ? pair_corr : 0.0;
}

/// G4PairingCorrection::GetFissionPairingCorrection - the formula unconditionally, and not
/// clamped at zero. Different function, not the max() removed: a fission barrier wants the
/// smooth term even where a table exists.
__host__ __device__ inline double fission_pairing_correction(int A, int Z) {
  const int N = A - Z;
  const double kPairingConstant = 12.0 * u::MeV<double>();
  return ((1 - Z + 2 * (Z / 2)) + (1 - N + 2 * (N / 2))) * kPairingConstant /
         std::sqrt(static_cast<double>(A));
}

/// G4ShellCorrection::GetShellCorrection(A, Z). Cook, else Cameron-Gilbert, else zero -
/// and note that the second call's return value is discarded in Geant4, so "else zero" is
/// the initialiser surviving rather than an explicit branch.
__host__ __device__ inline double shell_correction(int A, int Z) {
  double shell_corr = 0.0;
  const int N = A - Z;
  if (!cook_shell(N, Z, shell_corr)) {
    cameron_gilbert_shell(N, Z, shell_corr);
  }
  return shell_corr;
}

// ---------------------------------------------------------------------------------------------
// Level density.
// ---------------------------------------------------------------------------------------------

/// The level-density block of the G4LevelManager constructor: a four-way parameterisation in
/// the parity of Z and N for A > 20, and G4NuclearLevelData::GetLevelDensity(Z, A, 0) below
/// that - which, with fLD true, is A*0.075 again.
///
/// J. Nucl. Sci. Tech. 31(2) 151-162 (1994), as G4LevelManager.cc cites it. Reached only when
/// deex_params().level_density_flag is false.
__host__ __device__ inline double level_manager_level_density(int Z, int A) {
  double ld = static_cast<double>(A) * deex_params().level_density;
  if (A > 20) {
    const int N = A - Z;
    const int In = N - (N / 2) * 2;
    const int Iz = Z - (Z / 2) * 2;
    const double a13 = 1.0 / data::g4pow_z13<double>(A);
    if (In == 0 && Iz == 0) {
      ld = 0.067946 * A * (1.0 + 4.1277 * a13);
    } else if (In == 0 && Iz == 1) {
      ld = 0.053061 * A * (1.0 + 7.1862 * a13);
    } else if (In == 1 && Iz == 0) {
      ld = 0.060920 * A * (1.0 + 3.8767 * a13);
    } else {
      ld = 0.065291 * A * (1.0 + 4.4505 * a13);
    }
  }
  return ld;
}

/// G4NuclearLevelData::GetLevelDensity(Z, A, U). `U` is accepted and ignored - by Geant4 too:
/// the flag branch does not use it and G4LevelManager::LevelDensity(U) discards its argument.
/// Kept in the signature because every caller passes an excitation energy and dropping it
/// would hide that the level density in this configuration does not depend on one.
///
/// `has_levels` is whether G4NuclearLevelData::GetLevelManager(Z, A) would return non-null,
/// i.e. whether Z and A are inside the AMIN/AMAX window AND the PhotonEvaporation file parsed.
/// The no-level fall-back `0.058025*A*(1 + 5.9059/A^(1/3))` is a third formula again.
__host__ __device__ inline double level_density(int Z, int A, double U, bool has_levels) {
  if (deex_params().level_density_flag) {
    return static_cast<double>(A) * deex_params().level_density;
  }
  (void)U;
  if (has_levels) { return level_manager_level_density(Z, A); }
  return 0.058025 * A * (1.0 + 5.9059 / data::g4pow_z13<double>(A));
}

/// G4NuclearLevelData::GetPairingCorrection(Z, A) - NOT the same function as
/// G4PairingCorrection::GetPairingCorrection(A, Z) it delegates to when fLD is set. The other
/// branch uses sqrt(A) only above A = 36 and a flat 6 below it, and its multiplier counts from
/// 2 rather than from 1 - so the two agree only where the table wins.
__host__ __device__ inline double level_data_pairing_correction(int Z, int A) {
  if (deex_params().level_density_flag) { return pairing_correction(A, Z); }
  const int N = A - Z;
  const double par = 12.0 * u::MeV<double>();
  const double x = (A <= 36) ? 6.0 : std::sqrt(static_cast<double>(A));
  return (2 - Z + (Z / 2) * 2 - N + (N / 2) * 2) * par / x;
}

/// G4EvaporationLevelDensityParameter::LevelDensityParameter(A, Z, U) - a one-line forward to
/// G4NuclearLevelData::GetLevelDensity. Named separately because it is the object the
/// evaporation channels hold, and because the commented-out Iljinov parameterisation that
/// used to be in it (alpha 0.072, beta 0.257, gamma 0.059, f = 2.31) is not what runs.
__host__ __device__ inline double evaporation_level_density_parameter(int A, int Z, double U,
                                                                      bool has_levels) {
  return level_density(Z, A, U, has_levels);
}

}  // namespace g4gpu::deex

#endif
