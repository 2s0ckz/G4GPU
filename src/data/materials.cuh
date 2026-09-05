// Flat material table: per-element atom number densities, precomputed on the host so
// device code does no composition arithmetic.
//
// Compositions and densities are transcribed verbatim from Geant4 11.5.0
// G4NistMaterialBuilder.cc for the four materials example B1 uses. Water is given there
// by atom count (H2 O1) and is converted to weight fractions here.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "core/units.cuh"
#include "data/nist_stopping_names.hh"
#include "data/atomic_masses.cuh"
#include "data/icru90.hh"
#include "data/fermi_velocity.hh"
#include "data/g4pow.hh"
#include "data/production_cuts.cuh"

namespace g4gpu::data {

/// Elements one material may contain.
///
/// 16, and the number is bounded from both sides. Below: the shipped NIST database has
/// materials with ten - G4_BLOOD_ICRP and G4_CONCRETE - and nine more with nine, among them
/// G4_TISSUE_SOFT_ICRP, G4_MUSCLE_SKELETAL_ICRP and G4_SKIN_ICRP, which are exactly what a
/// medical calculation reaches for. This was 8, which is what example B1's materials need, and
/// `from_weight_fractions` had no bound check: every one of those eleven materials wrote past
/// the end of `z[]` into the fields after it. Above: `WentzelElementXs` carries two arrays of
/// this length as a *per-thread local* in step_hadron, so every slot is 16 bytes of local
/// memory on every thread whether the material uses it or not.
///
/// A material with more than this is refused rather than truncated - see from_weight_fractions.
constexpr int kMaxElements = 16;


/// Number densities are in atoms per mm^3, matching the mm-based unit system.
template <typename real_t>
struct Material {
  int n_elements;
  real_t z[kMaxElements];         ///< atomic number, as real for use in Z-parameterizations
  real_t n_atoms[kMaxElements];   ///< atoms / mm^3
  real_t density;                 ///< g/cm^3, kept for dose normalization
  real_t electron_density;        ///< electrons / mm^3, = sum(Z_i * n_i)
  real_t z_eff;
  /// <A^(-2/3)> weighted by atom density (G4IonisParamMat::GetInvA23), which the
  /// Wentzel scattering cut-off angle needs.
  real_t inv_a23;
  /// Fermi energy, MeV: 25 keV * vF^2 with vF the atom-density-weighted mean of Ziegler's
  /// per-element Fermi velocity (G4IonisParamMat::GetFermiEnergy). The heavy-ion effective
  /// charge is a function of the ion's velocity in units of this, and carrying it is what
  /// made G4ionEffectiveCharge's Zi > 2 branch transcribable at all.
  real_t fermi_energy;
  real_t mean_excitation;         ///< MeV, tabulated (4th arg of Geant4 AddMaterial)
  real_t radiation_length;        ///< mm, for the Highland MSC formula

  // Sternheimer density-effect parameters, taken verbatim from Geant4's
  // G4DensityEffectData table (Sternheimer, At. Data Nucl. Data Tables 30:261, 1984).
  // All four B1 materials have table entries, so no analytic fallback is needed.
  real_t c_density, x0_density, x1_density, a_density, m_density, delta0_density;

  // Production thresholds, MeV. Geant4 converts the 1 mm default range cut into these
  // per-material energies; values are Geant4 11.1.1's own, dumped by ref/dump/g4dump.cc.
  // Above them a transportable secondary is created; below, the energy is deposited
  // continuously. This is what makes dE/dx *restricted* rather than total.
  real_t cut_gamma, cut_electron, cut_positron;

  // The row of the 74-material PSTAR and ASTAR tables this material's low-energy heavy-
  // particle stopping power comes from, or -1 for none. Resolved from the material's name -
  // and failing that its chemical formula - when it is built, because the device has no names.
  // See data/nist_stopping_names.hh and set_nist_stopping below.
  int nist_stopping = -1;
  /// Matter state, in G4State's encoding: 0 undefined, 1 solid, 2 liquid, 3 gas.
  ///
  /// Carried because two models branch on it rather than on density. G4IonFluctuations::Factor
  /// picks a different reduced-energy scaling and a different parameter row for a gas, and
  /// G4IonisParamMat::ComputeDensityEffectParameters uses a different Sternheimer ladder.
  /// Geant4 tests `kStateGas == material->GetState()` in both, so an exact state and not a
  /// density threshold is what reproduces it. See material_is_gas.
  int state = 0;
  /// Row in the ICRU 90 tables, or -1. Set only when G4EmParameters::SetUseICRU90Data(true)
  /// was called before initialisation - so a default run has -1 here for every material and
  /// takes the PSTAR path, which is Geant4's default too. When it is set it *replaces* PSTAR:
  /// G4BraggModel resolves iICRU90 first and returns as soon as it has one.
  int icru90 = -1;
};

/// 2*ln(10), the Sternheimer slope.
template <typename real_t> __host__ __device__ constexpr real_t twoln10() {
  return real_t(4.60517018598809136804);
}

/// Density-effect correction delta(x), with x = log10(beta*gamma).
/// Transcribed from G4IonisParamMat::GetDensityCorrection.
template <typename real_t>
__host__ __device__ inline real_t density_correction(const Material<real_t>& m, real_t x) {
  if (x < m.x0_density) {
    return (m.delta0_density > real_t(0))
               ? m.delta0_density * exp(twoln10<real_t>() * (x - m.x0_density))
               : real_t(0);
  }
  if (x >= m.x1_density) { return twoln10<real_t>() * x - m.c_density; }
  return twoln10<real_t>() * x - m.c_density
         + m.a_density * exp(log(m.x1_density - x) * m.m_density);
}

/// Compile-time ceiling on distinct materials in a scene. The actual count is a runtime
/// value carried by MaterialTable, so a user scene is not limited to B1s four.
constexpr int kMaxMaterials = 32;

/// Convenience ids for the B1 test scene only. User scenes index by their own order.
enum B1MaterialId : int { kAir = 0, kWater = 1, kA150Tissue = 2, kBoneCompact = 3 };

/// Retained so existing B1 code keeps compiling; new code should use MaterialTable::count.
constexpr int kNumMaterials = 4;

/// Standard atomic weight, g/mol, for any element Geant4 knows (Z = 1..98).
///
/// Returns 0 outside that range, which callers must treat as an error rather than as a
/// mass: dividing by it silently produced an infinite atom density before this table
/// covered more than B1s ten elements.
template <typename real_t>
__host__ __device__ inline real_t atomic_mass(int z) {
  if (z < 1 || z > 98) { return real_t(0); }
  return real_t(nist_atomic_mass_table()[z]);
}

/// Coulomb correction factor, transcribed from G4Element::ComputeCoulombFactor
/// (Phys. Rev. D50 3-1 (1994) p.1254).
template <typename real_t>
__host__ __device__ inline real_t coulomb_factor(real_t Z) {
  constexpr real_t k1 = real_t(0.0083), k2 = real_t(0.20206), k3 = real_t(0.0020),
                   k4 = real_t(0.0369);
  constexpr real_t alpha = units::fine_structure_const<real_t>();
  const real_t az2 = (alpha * Z) * (alpha * Z);
  const real_t az4 = az2 * az2;
  return (k1 * az4 + k2 + real_t(1) / (real_t(1) + az2)) * az2 - (k3 * az4 + k4) * az4;
}

/// Tsai radiation-length factor per atom, mm^2. Transcribed verbatim from
/// G4Element::ComputeLradTsaiFactor, including the light-element table for Z <= 4.
/// The material radiation length is then 1 / sum(n_i * radTsai_i), matching
/// G4Material::ComputeRadiationLength. This replaces an earlier approximation that ran
/// 0.7-1.05% high against Geant4.
template <typename real_t>
__host__ inline real_t element_rad_tsai(int z) {
  constexpr real_t alpha = units::fine_structure_const<real_t>();
  const real_t re = units::classic_electron_radius<real_t>();
  const real_t alpha_rcl2 = alpha * re * re;

  const real_t Z = real_t(z);
  static const real_t Lrad_light[4] = {real_t(5.31), real_t(4.79), real_t(4.74), real_t(4.71)};
  static const real_t Lprad_light[4] = {real_t(6.144), real_t(5.621), real_t(5.805),
                                        real_t(5.924)};
  const real_t logZ3 = std::log(Z) / real_t(3);
  real_t Lrad, Lprad;
  const int iz = z - 1;
  if (iz <= 3) {
    Lrad = Lrad_light[iz];
    Lprad = Lprad_light[iz];
  } else {
    Lrad = std::log(real_t(184.15)) - logZ3;
    Lprad = std::log(real_t(1194.0)) - real_t(2) * logZ3;
  }
  return real_t(4) * alpha_rcl2 * Z * (Z * (Lrad - coulomb_factor<real_t>(Z)) + Lprad);
}

/// Builds one material from weight fractions. Host-only: device code reads the result.
/// @param mean_excitation_eV  tabulated mean excitation energy (Geant4 AddMaterial arg 4)
template <typename real_t>
__host__ inline Material<real_t> from_weight_fractions(real_t density_g_cm3, int n,
                                                       const int* zs, const real_t* w,
                                                       real_t mean_excitation_eV) {
  Material<real_t> m{};
  // Refused, not truncated, and not silently overrun. A material is a fixed-size device record,
  // so this bound is real; what it must never do is write past it. The eleven NIST materials
  // with nine or ten elements did exactly that until kMaxElements was raised, and the symptom
  // would have been a corrupted density or excitation energy rather than a crash.
  if (n > kMaxElements) {
    std::printf(
        "\nFATAL: a material with %d elements, and this build stores at most %d.\n"
        "  Raise kMaxElements in src/data/materials.cuh and rebuild. It is a device-side\n"
        "  array bound, not a physics limit; the cost is that WentzelElementXs carries two\n"
        "  arrays of that length per thread.\n",
        n, kMaxElements);
    std::exit(2);
  }
  m.n_elements = n;
  m.density = density_g_cm3;
  m.electron_density = real_t(0);
  m.mean_excitation = mean_excitation_eV * real_t(1e-6);  // eV -> MeV
  real_t atom_density = real_t(0);
  real_t inv_a23 = real_t(0);
  real_t vf = real_t(0);  // atom-density-weighted Fermi velocity, per G4IonisParamMat
  real_t radinv = real_t(0);  // 1/X0 = sum(n_i * radTsai_i), per G4Material
  // Geant4 normalises the mass fractions - G4Material::FillProperties computes
  // `coeff = 1/wtSum` and scales every fraction by it - and the NIST compositions do not sum
  // to one. G4_AIR's four tabulated fractions sum to 0.999999, so without this the port's
  // electron density, and with it every macroscopic cross section, was 1e-6 low in air. That
  // deviation had been visible in test_vs_oracle's printout for as long as the printout
  // existed, and nothing asserted on it.
  real_t wt_sum = real_t(0);
  for (int i = 0; i < n; ++i) { wt_sum += w[i]; }
  const real_t wt_coeff = (wt_sum > real_t(0)) ? real_t(1) / wt_sum : real_t(1);
  for (int i = 0; i < n; ++i) {
    m.z[i] = real_t(zs[i]);
    const real_t wi = w[i] * wt_coeff;
    // n_i = rho * w_i * N_A / A_i, with the cm^3 -> mm^3 factor folded into number_density
    m.n_atoms[i] = units::number_density<real_t>(density_g_cm3 * wi, atomic_mass<real_t>(zs[i]));
    m.electron_density += m.z[i] * m.n_atoms[i];
    atom_density += m.n_atoms[i];
    radinv += m.n_atoms[i] * element_rad_tsai<real_t>(zs[i]);
    // G4IonisParamMat divides by G4Element::GetN()^(2/3). GetN() returns fNeff, the
    // abundance-weighted nucleon number, which for a NIST element is the standard atomic
    // weight - not an integer. Rounding it left invA23 0.25-0.5% off and shifted every
    // Wentzel cut-off angle. It also goes through G4Pow::A23, which is a Taylor expansion of
    // the cube root and not the cube root: an exact power is ~1e-5 away from what Geant4
    // computes, so g4pow_a23 reproduces the approximation rather than improving on it.
    inv_a23 += m.n_atoms[i] / g4pow_a23<real_t>(atomic_mass<real_t>(zs[i]));
    vf += m.n_atoms[i] * static_cast<real_t>(fermi_velocity_table(zs[i]));
  }
  m.z_eff = (atom_density > real_t(0)) ? m.electron_density / atom_density : real_t(1);
  m.inv_a23 = (atom_density > real_t(0)) ? inv_a23 / atom_density : real_t(1);
  // fFermiEnergy = 25*keV * vF*vF, with vF averaged the same way. 25 keV = 0.025 MeV here.
  const real_t vf_mean = (atom_density > real_t(0)) ? vf / atom_density : real_t(0);
  m.fermi_energy = real_t(0.025) * vf_mean * vf_mean;
  m.radiation_length = (radinv > real_t(0)) ? real_t(1) / radinv : real_t(1e30);
  return m;
}

/// Analytic Sternheimer parameters, for materials with no entry in Geant4's tabulated
/// database. Transcribed from G4IonisParamMat::ComputeDensityEffectParameters (the `else`
/// branch), including the hydrogen and helium special cases and the gas ladder.
///
/// Geant4 knows each material state explicitly, so  is_gas is passed in rather than
/// inferred. The density fallback below only applies when a caller does not say.
enum class MaterialState : int { kAuto = 0, kSolid = 1, kLiquid = 2, kGas = 3 };

/// Is this material a gas, in the sense Geant4's models ask?
///
/// Geant4 asks `kStateGas == material->GetState()` and nothing else - kStateUndefined takes
/// the non-gas branch there. This adds one fallback: a Material built without a state at all
/// falls back to a density threshold, because this port can construct materials a real
/// G4Material never is - a detector entered by hand in the builder, with no state given.
/// The fallback is checked against every NIST material in tests/test_all_materials.cu, where
/// it agrees with the tabulated state for all of them, so it changes no answer that Geant4
/// also has an answer for.
template <typename real_t>
__host__ __device__ inline bool material_is_gas(const Material<real_t>& m) {
  if (m.state != 0) { return m.state == static_cast<int>(MaterialState::kGas); }
  return m.density < real_t(0.01);
}

/// Records the matter state. Call before compute_sternheimer, which reads it.
template <typename real_t>
__host__ inline void set_state(Material<real_t>& m, MaterialState state) {
  m.state = static_cast<int>(state);
}

template <typename real_t>
__host__ inline void compute_sternheimer(Material<real_t>& m,
                                        MaterialState state = MaterialState::kAuto) {
  // Cd2 = 4*pi*hbarc^2*r_e; plasma energy = sqrt(Cd2 * n_e).
  constexpr real_t hbarc = units::hbarc<real_t>();  // MeV*mm, CLHEP's derived value
  const real_t Cd2 = real_t(4) * units::pi<real_t>() * hbarc * hbarc
                     * units::classic_electron_radius<real_t>();
  const real_t plasma = std::sqrt(Cd2 * m.electron_density);
  // Geant4 branches on solid/liquid vs gas. An explicit argument wins and is recorded on the
  // material; otherwise whatever set_state put there is used, and only a material with no
  // state at all falls back to the density test - see material_is_gas.
  if (state != MaterialState::kAuto) { m.state = static_cast<int>(state); }
  const bool is_gas = material_is_gas<real_t>(m);

  m.c_density = real_t(1) + real_t(2) * std::log(m.mean_excitation / plasma);
  m.delta0_density = real_t(0);
  const bool single_h = (m.n_elements == 1 && static_cast<int>(m.z[0] + real_t(0.5)) == 1);
  const bool single_he = (m.n_elements == 1 && static_cast<int>(m.z[0] + real_t(0.5)) == 2);

  if (!is_gas) {
    constexpr real_t E100eV = real_t(100e-6);
    const real_t ClimiS[2] = {real_t(3.681), real_t(5.215)};
    const real_t X0valS[2] = {real_t(1.0), real_t(1.5)};
    const real_t X1valS[2] = {real_t(2.0), real_t(3.0)};
    const int icase = (m.mean_excitation < E100eV) ? 0 : 1;
    m.x0_density = (m.c_density < ClimiS[icase]) ? real_t(0.2)
                                                 : real_t(0.326) * m.c_density - X0valS[icase];
    m.x1_density = X1valS[icase];
    m.m_density = real_t(3.0);
    if (single_h) { m.x0_density = real_t(0.425); m.x1_density = real_t(2.0);
                    m.m_density = real_t(5.949); }
  } else {
    m.m_density = real_t(3.0);
    m.x1_density = real_t(4.0);
    if (m.c_density <= real_t(10.0))        { m.x0_density = real_t(1.6); }
    else if (m.c_density <= real_t(10.5))   { m.x0_density = real_t(1.7); }
    else if (m.c_density <= real_t(11.0))   { m.x0_density = real_t(1.8); }
    else if (m.c_density <= real_t(11.5))   { m.x0_density = real_t(1.9); }
    else if (m.c_density <= real_t(12.25))  { m.x0_density = real_t(2.0); }
    else if (m.c_density <= real_t(13.804)) { m.x0_density = real_t(2.0); m.x1_density = real_t(5.0); }
    else { m.x0_density = real_t(0.326) * m.c_density - real_t(2.5); m.x1_density = real_t(5.0); }
    if (single_h)  { m.x0_density = real_t(1.837); m.x1_density = real_t(3.0);
                     m.m_density = real_t(4.754); }
    if (single_he) { m.x0_density = real_t(2.191); m.x1_density = real_t(3.0);
                     m.m_density = real_t(3.297); }
  }
  // a is fixed by requiring delta to be continuous at x0.
  const real_t dx = m.x1_density - m.x0_density;
  m.a_density = (dx > real_t(0))
                    ? (m.c_density - twoln10<real_t>() * m.x0_density) / std::pow(dx, m.m_density)
                    : real_t(0);
}

/// Sternheimer density-effect coefficients, in G4DensityEffectData array order
/// (indices 2..7 of the Mnn arrays): C, x0, x1, a, m, delta0.
template <typename real_t>
__host__ inline void set_sternheimer(Material<real_t>& m, real_t C, real_t x0, real_t x1,
                                     real_t a, real_t mexp, real_t d0) {
  m.c_density = C;
  m.x0_density = x0;
  m.x1_density = x1;
  m.a_density = a;
  m.m_density = mexp;
  m.delta0_density = d0;
}

/// Production thresholds for the default 1 mm range cut, from Geant4 11.1.1 itself
/// (ref/oracle/cuts.csv). Air sits at the 990 eV floor because 1 mm of air is almost no
/// material at all.
template <typename real_t>
__host__ inline void set_cuts(Material<real_t>& m, real_t g, real_t e, real_t p) {
  m.cut_gamma = g;
  m.cut_electron = e;
  m.cut_positron = p;
}

/// A scene's materials. `count` is a runtime value; `kMaxMaterials` only bounds storage.
template <typename real_t>
struct MaterialTable {
  Material<real_t> m[kMaxMaterials];
  int count = 0;

  __host__ __device__ const Material<real_t>& operator[](int i) const { return m[i]; }
  __host__ __device__ Material<real_t>& operator[](int i) { return m[i]; }

  /// Distinct atomic numbers across every material, which is what the data loaders need.
  /// Replaces the hardcoded B1 element list.
  __host__ int elements(int* zs_out, int max_out) const {
    int n = 0;
    bool seen[101] = {};
    for (int i = 0; i < count; ++i) {
      for (int j = 0; j < m[i].n_elements; ++j) {
        const int z = static_cast<int>(m[i].z[j] + real_t(0.5));
        if (z >= 1 && z <= 100 && !seen[z] && n < max_out) {
          seen[z] = true;
          zs_out[n++] = z;
        }
      }
    }
    return n;
  }
};


/// The ICRU 90 row for a material, by name. Three materials, and only when the caller asked.
///
/// Separate from set_nist_stopping and gated by the flag at the call site rather than here,
/// because "which table does this material have" and "did the user turn ICRU 90 on" are two
/// different questions and only one of them is about the material.
///
/// Called from G4Material's device-record build, which is the path an example or a generated
/// project takes: G4NistManager::FindOrBuildMaterial -> G4Material -> here. The hard-coded
/// build_b1_materials below does not call it, so the reference driver's own four materials
/// stay on PSTAR whatever the flag says - they are a fixture, not a user's detector.
template <typename real_t>
__host__ inline void set_icru90(Material<real_t>& m, const char* g4_name) {
  m.icru90 = icru90_index(g4_name);
}

/// Resolves which tabulated stopping-power data @p m has, from its Geant4 name and chemical
/// formula. Host only, and called once per material.
///
/// This is `G4PSTARStopping::Initialise`'s decision: the material's name against the 74 NIST
/// names first, and if that misses, its chemical formula against the twelve formulae that
/// resolve to one of those rows. A material called "MyWater" with the formula "H_2O" therefore
/// gets G4_WATER's tabulated stopping power - which is what Geant4 gives it.
///
/// It is worth knowing what is *not* here. `G4BraggModel` carries its own eleven-compound ICRU
/// 49 parameterisation, selected by the same formulae, for when PSTAR has nothing. Its list is
/// a strict subset of the twelve above, so PSTAR always resolves first and that branch is dead
/// code - in Geant4 as much as here. It was transcribed, measured to be unreachable, and
/// deleted; see docs/DAY.md.
///
/// Doing this here rather than at use is not an optimisation. The device has no material
/// names to match on.
///
/// @param g4_name  the material's Geant4 name, e.g. "G4_WATER"
/// @param formula  its chemical formula, e.g. "H_2O"; empty or null if it has none
template <typename real_t>
__host__ inline void set_nist_stopping(Material<real_t>& m, const char* g4_name,
                                       const char* formula = nullptr) {
  m.nist_stopping = nist_stopping_index(g4_name);
  if (m.nist_stopping < 0) { m.nist_stopping = nist_stopping_index_by_formula(formula); }
}

/// Adds a material from weight fractions and derives its production thresholds from the
/// range cut, so a user-supplied material needs no hand-entered cut values.
///
/// @param range_cut_mm  production range cut; Geant4's QBBC default is 0.7 mm
/// @return index of the new material, or -1 if the table is full
template <typename real_t>
__host__ inline int add_material(MaterialTable<real_t>& t, real_t density_g_cm3, int n,
                                 const int* zs, const real_t* w, real_t mean_excitation_eV,
                                 real_t range_cut_mm = real_t(0.7),
                                 MaterialState state = MaterialState::kAuto) {
  if (t.count >= kMaxMaterials) { return -1; }
  const int idx = t.count;
  Material<real_t>& m = t.m[idx];
  m = from_weight_fractions<real_t>(density_g_cm3, n, zs, w, mean_excitation_eV);

  // Thresholds computed from the range cut, matching G4VRangeToEnergyConverter.
  m.cut_gamma = convert_cut_gamma<real_t>(range_cut_mm, m.n_elements, m.z, m.n_atoms);
  m.cut_electron =
      convert_cut_electron<real_t>(range_cut_mm, m.n_elements, m.z, m.n_atoms, m.density);
  m.cut_positron = convert_cut_electron<real_t>(range_cut_mm, m.n_elements, m.z, m.n_atoms,
                                                m.density, /*is_positron=*/true);

  // Sternheimer parameters stay zeroed; callers with tabulated values call set_sternheimer,
  // otherwise compute_sternheimer supplies Geant4's analytic fallback.
  compute_sternheimer<real_t>(m, state);
  ++t.count;
  return idx;
}

/// The four B1 materials, in B1MaterialId order.
template <typename real_t>
__host__ inline void build_b1_materials(Material<real_t>* out) {
  {  // G4_AIR: density 0.00120479, weight fractions
    const int zs[4] = {6, 7, 8, 18};
    const real_t w[4] = {real_t(0.000124), real_t(0.755267), real_t(0.231781), real_t(0.012827)};
    out[kAir] = from_weight_fractions<real_t>(real_t(0.00120479), 4, zs, w, real_t(85.7));
    // G4DensityEffectData M101
    set_sternheimer<real_t>(out[kAir], real_t(10.5961), real_t(1.7418), real_t(4.2759), real_t(0.10914089377455813), real_t(3.3994), real_t(0));
    // G4NistMaterialBuilder declares air a gas; the other three take its default,
    // which is kStateSolid - water included. Only gas-or-not is ever asked, and the
    // tabulated state is used rather than a guess because G4IonFluctuations picks a
    // different reduced-energy scaling for a gas.
    set_state<real_t>(out[kAir], MaterialState::kGas);
    set_cuts<real_t>(out[kAir], real_t(0.00099), real_t(0.00099), real_t(0.00099));
    set_nist_stopping<real_t>(out[kAir], "G4_AIR");
  }
  {  // G4_WATER: H2O by atom count -> weight fractions
    const real_t aH = atomic_mass<real_t>(1), aO = atomic_mass<real_t>(8);
    const real_t total = real_t(2) * aH + aO;
    const int zs[2] = {1, 8};
    const real_t w[2] = {real_t(2) * aH / total, aO / total};
    out[kWater] = from_weight_fractions<real_t>(real_t(1.0), 2, zs, w, real_t(78.0));
    // NOT the G4DensityEffectData M273 row (Cbar 3.5017, x0 0.24, x1 2.8004). For G4_WATER,
    // Geant4 computes the Sternheimer coefficients analytically instead of reading that
    // table, and the two disagree: Cbar 3.5801 vs 3.5017. Using the tabulated row made every
    // heavy particle 0.39% low in water at high energy, and left a long-standing ~0.3%
    // discrepancy in the electron dE/dx. These are G4IonisParamMat's own values, dumped at
    // full precision into ref/oracle/ionisation_params.csv.
    set_sternheimer<real_t>(out[kWater], real_t(3.5801414263065614),
                            real_t(0.25703333929878008), real_t(2.81743333929878),
                            real_t(0.091150961477294901), real_t(3.4773), real_t(0));
    set_state<real_t>(out[kWater], MaterialState::kSolid);
    set_cuts<real_t>(out[kWater], real_t(0.00252520505), real_t(0.277632595), real_t(0.270822571));
    set_nist_stopping<real_t>(out[kWater], "G4_WATER", "H_2O");
  }
  {  // G4_A-150_TISSUE: density 1.127
    const int zs[6] = {1, 6, 7, 8, 9, 20};
    const real_t w[6] = {real_t(0.101327), real_t(0.775501), real_t(0.035057),
                         real_t(0.052316), real_t(0.017422), real_t(0.018378)};
    out[kA150Tissue] = from_weight_fractions<real_t>(real_t(1.127), 6, zs, w, real_t(65.1));
    // G4DensityEffectData M96
    set_sternheimer<real_t>(out[kA150Tissue], real_t(3.1099999999999999), real_t(0.13289999999999999), real_t(2.6234000000000002), real_t(0.10781957049030769), real_t(3.4441999999999999), real_t(0));
    set_state<real_t>(out[kA150Tissue], MaterialState::kSolid);
    set_cuts<real_t>(out[kA150Tissue], real_t(0.00228342803), real_t(0.301330518), real_t(0.293732854));
    set_nist_stopping<real_t>(out[kA150Tissue], "G4_A-150_TISSUE");
  }
  {  // G4_BONE_COMPACT_ICRU: density 1.85
    const int zs[8] = {1, 6, 7, 8, 12, 15, 16, 20};
    const real_t w[8] = {real_t(0.064), real_t(0.278), real_t(0.027), real_t(0.410),
                         real_t(0.002), real_t(0.070), real_t(0.002), real_t(0.147)};
    out[kBoneCompact] = from_weight_fractions<real_t>(real_t(1.85), 8, zs, w, real_t(91.9));
    // G4DensityEffectData M116
    set_sternheimer<real_t>(out[kBoneCompact], real_t(3.3390), real_t(0.0944), real_t(3.0201), real_t(0.058220375161520004), real_t(3.6419000000000001), real_t(0));
    set_state<real_t>(out[kBoneCompact], MaterialState::kSolid);
    set_cuts<real_t>(out[kBoneCompact], real_t(0.00393604038), real_t(0.398359696), real_t(0.386305646));
    set_nist_stopping<real_t>(out[kBoneCompact], "G4_BONE_COMPACT_ICRU");
  }
}

}  // namespace g4gpu::data
