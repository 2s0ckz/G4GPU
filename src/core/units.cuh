// Geant4-compatible internal units: mm, MeV, ns. Values from CLHEP SystemOfUnits.
#pragma once
namespace g4gpu::units {

template <typename T> __host__ __device__ constexpr T mm()  { return T(1); }
template <typename T> __host__ __device__ constexpr T cm()  { return T(10); }
template <typename T> __host__ __device__ constexpr T m()   { return T(1000); }
template <typename T> __host__ __device__ constexpr T MeV() { return T(1); }
template <typename T> __host__ __device__ constexpr T keV() { return T(1e-3); }
template <typename T> __host__ __device__ constexpr T eV()  { return T(1e-6); }
template <typename T> __host__ __device__ constexpr T GeV() { return T(1e3); }

// 1 barn = 1e-28 m^2 = 1e-22 mm^2
template <typename T> __host__ __device__ constexpr T barn() { return T(1e-22); }

// Time, and the speed that ties it to length. CLHEP has nanosecond as the unit of time and
// derives c_light from the SI value; both are done the same way here rather than pasting
// 299.792458, so that mm and ns remain the only places a length or a time is defined.
// docs/RISK.md V8 is what a hand-typed physical constant cost last time.
template <typename T> __host__ __device__ constexpr T ns() { return T(1); }
template <typename T> __host__ __device__ constexpr T s()  { return T(1e9) * ns<T>(); }
template <typename T> __host__ __device__ constexpr T c_light() {
  return T(2.99792458e8) * m<T>() / s<T>();  // 299.792458 mm/ns
}

template <typename T> __host__ __device__ constexpr T pi()    { return T(3.14159265358979323846); }
template <typename T> __host__ __device__ constexpr T twopi() { return T(6.28318530717958647692); }

// The masses and constants CLHEP pins per release. These are 11.1.1's values, which is what
// the oracle links; several are older than the CODATA numbers a person would look up, and
// substituting a newer one is a transcription error however much more accurate it is. The
// amu_c2 here was 931.49410242 for a day and put G4ionEffectiveCharge 1.5e-8 off Geant4's own
// answer - invisible in a dose, fatal to a 1e-9 comparison, and indistinguishable from a
// formula bug until the constants themselves were checked. tests/test_constants.cu does that.
template <typename T> __host__ __device__ constexpr T electron_mass_c2() { return T(0.510998910); }
template <typename T> __host__ __device__ constexpr T proton_mass_c2()   { return T(938.272013); }
template <typename T> __host__ __device__ constexpr T neutron_mass_c2()  { return T(939.56536); }
template <typename T> __host__ __device__ constexpr T amu_c2()           { return T(931.494028); }
template <typename T> __host__ __device__ constexpr T avogadro()         { return T(6.02214076e23); }

/// Classical electron radius, mm.
///
/// CLHEP does not tabulate this: it derives it, as `elm_coupling/electron_mass_c2` with
/// `elm_coupling = e^2/(4 pi eps0)`, out of the unit system's own e, mu0 and c. The result is
/// 2.817940545232519e-12 mm, which is 7.8e-8 away from the CODATA value 2.8179403262e-12 fm
/// that this used to hold. Every ionisation prefactor carries that factor, so the port was
/// biased by 8e-8 everywhere - far below any physics tolerance, and still wrong.
///
/// The value below is Geant4 11.1.1's own, read out of ref/oracle/constants.csv, and
/// tests/test_constants.cu keeps it there.
template <typename T> __host__ __device__ constexpr T classic_electron_radius() {
  return T(2.817940545232519e-12);
}

/// Fine structure constant, as CLHEP derives it (elm_coupling/hbarc), not 1/137.035999139.
template <typename T> __host__ __device__ constexpr T fine_structure_const() {
  return T(0.007297352565305215);
}

/// twopi * electron_mass_c2 * classic_electron_radius^2, MeV mm^2 - the Bethe-Bloch and
/// Moller prefactor. Derived in CLHEP, so pinned here rather than recomputed: recomputing it
/// from the three factors above gives a different last bit.
template <typename T> __host__ __device__ constexpr T twopi_mc2_rcl2() {
  return T(2.5495497670537043e-23);
}

/// hbar * c, MeV mm.
template <typename T> __host__ __device__ constexpr T hbarc() {
  return T(1.9732698045930245e-10);
}

/// Electron Compton wavelength hbarc/(m_e c^2), mm, and the Bohr radius, mm.
template <typename T> __host__ __device__ constexpr T electron_compton_length() {
  return T(3.8615929818578764e-10);
}
template <typename T> __host__ __device__ constexpr T bohr_radius() {
  return T(5.2917725261316923e-08);
}

/// Atoms per mm^3 for density in g/cm^3 and molar mass in g/mol.
/// n[1/mm^3] = rho[g/cm^3] * N_A / (A[g/mol] * 1000 mm^3/cm^3)
template <typename T>
__host__ __device__ inline T number_density(T density_g_cm3, T molar_mass_g_mol) {
  return density_g_cm3 * T(6.02214076e20) / molar_mass_g_mol;
}

}  // namespace g4gpu::units
