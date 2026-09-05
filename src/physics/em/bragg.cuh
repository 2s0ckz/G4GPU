// Low-energy heavy-particle ionisation, transcribed from G4BraggModel (11.1.1).
//
// G4hIonisation, G4MuIonisation and G4ionIonisation all use a Bragg-family model below
// 2 MeV per nucleon and Bethe-Bloch above it. This is the branch that produces the Bragg
// peak, and Bethe-Bloch is simply wrong there - it diverges as the velocity falls.
//
// IMPORTANT, and measured rather than assumed by tests/test_bragg.cu: G4BraggModel::DEDX has
// a four-way decision. For any of the 74 NIST materials in G4PSTARStopping it uses that
// tabulated stopping power; for a molecule in the Ziegler 1988 list it uses molecular data;
// otherwise it falls back to the per-element Ziegler parameterisation transcribed here.
// This port implements the Ziegler fallback, which is the path a user-defined material
// takes. Where Geant4 would reach for PSTAR instead, the two differ, and the test reports
// by how much rather than hiding it.
#pragma once
#include <cmath>
#include "core/particle.cuh"
#include "core/units.cuh"
#include "data/materials.cuh"
#include "data/icru90.hh"
#include "data/nist_stopping.hh"
#include "data/ziegler_ion_tables.cuh"
#include "data/ziegler_tables.cuh"
#include "physics/em/hadron_ionisation.cuh"

namespace g4gpu::em {

/// Proton mass in atomic mass units, as G4BraggModel uses it to scale the energy.
template <typename real_t> __host__ __device__ constexpr real_t proton_mass_amu() {
  return real_t(1.007276);
}
/// eV * cm^2 * 1e-15, the unit the Ziegler parameterisation returns.
/// In this port's units (MeV, mm): 1 eV = 1e-6 MeV, 1 cm^2 = 100 mm^2.
template <typename real_t> __host__ __device__ constexpr real_t ziegler_factor() {
  return real_t(1e-6) * real_t(100.0) * real_t(1e-15);
}
/// G4BraggModel::lowestKinEnergy, below which the model scales as sqrt(T).
template <typename real_t> __host__ __device__ constexpr real_t bragg_lowest_energy() {
  return real_t(0.25e-3);  // 0.25 keV
}

/// Electronic stopping power for one element, in eV cm^2 / 1e15 atoms.
/// Verbatim from G4BraggModel::ElectronicStoppingPower.
///
/// @param kinetic proton-equivalent kinetic energy, MeV
template <typename real_t>
__host__ __device__ inline real_t ziegler_stopping_power(int z, real_t kinetic) {
  int i = z - 1;
  if (i < 0) { i = 0; }
  if (i > 91) { i = 91; }
  const real_t* a = data::ziegler_a<real_t>();
  // T in keV per amu.
  real_t T = kinetic / (real_t(1e-3) * proton_mass_amu<real_t>());

  real_t fac = real_t(1);
  if (T < real_t(40) && i == 5) {  // carbon has its own low-energy cut
    fac = sqrt(T * real_t(0.025));
    T = real_t(40);
  } else if (T < real_t(10)) {
    fac = sqrt(T * real_t(0.1));
    T = real_t(10);
  }
  const real_t x1 = a[i * 5 + 1];
  const real_t x2 = a[i * 5 + 2];
  const real_t x3 = a[i * 5 + 3];
  const real_t x4 = a[i * 5 + 4];
  const real_t slow = x1 * exp(log(T) * real_t(0.45));
  const real_t shigh = log(real_t(1) + x3 / T + x4 * T) * x2 / T;
  const real_t loss = slow * shigh * fac / (slow + shigh);
  return fmax(loss, real_t(0));
}

/// Unrestricted electronic stopping power of a material, MeV/mm, for a proton.
///
/// Two branches, which is G4BraggModel::DEDX with one branch of its own removed as dead:
///
///   1. a material G4PSTARStopping can resolve - by NIST name, or failing that by chemical
///      formula - gets the tabulated PSTAR stopping power;
///   2. anything else gets the per-element Ziegler fit.
///
/// G4BraggModel has a third, between those two: its own eleven-compound ICRU 49
/// parameterisation, chosen by chemical formula. Its formula list is a strict subset of the
/// twelve formulae G4PSTARStopping resolves, so branch 1 always wins and that branch is
/// unreachable - in Geant4 as much as here. It was transcribed, measured to be unreachable
/// (a material named "CustomMolecularWater" with the formula "H_2O" gets G4_WATER's *table*,
/// and tests/test_bragg.cu checks that against Geant4's own answer), and deleted.
///
/// Which branch a material takes is not a detail: for water at the Bragg peak, PSTAR and the
/// per-element fit differ by 29%.
template <typename real_t>
__host__ __device__ inline real_t bragg_dedx_unrestricted(const data::Material<real_t>& m,
                                                          real_t kinetic) {
  // The tables are a mass stopping power in MeV cm2/g; density is g/cm3, and 1 cm = 10 mm.
  //
  // ICRU 90 before PSTAR, and returning as soon as it has an answer, because that is the
  // order G4BraggModel::ElectronicDEDX resolves them in: iICRU90 is looked up first and iPSTAR
  // only when it comes back negative. m.icru90 is -1 unless
  // G4EmParameters::SetUseICRU90Data(true) was called before initialisation, so a default run
  // never takes this branch.
  if (m.icru90 >= 0) {
    return data::icru90_mass_stopping<real_t>(m.icru90, kinetic, false) * m.density
           / real_t(10);
  }
  if (m.nist_stopping >= 0) {
    return data::pstar_mass_stopping<real_t>(m.nist_stopping, kinetic) * m.density
           / real_t(10);
  }
  real_t eloss = real_t(0);
  for (int i = 0; i < m.n_elements; ++i) {
    const int z = static_cast<int>(m.z[i] + real_t(0.5));
    eloss += ziegler_stopping_power(z, kinetic) * m.n_atoms[i];
  }
  return eloss * ziegler_factor<real_t>();
}

/// Restricted dE/dx, MeV/mm.
/// Verbatim from G4BraggModel::ComputeDEDXPerVolume: the tabulated (here parameterised)
/// stopping power is for a proton, scaled by the charge squared, with a correction
/// subtracting the delta rays above the production cut.
template <typename real_t>
__host__ __device__ inline real_t bragg_dedx(const data::Material<real_t>& m,
                                             ParticleType type, real_t kinetic, real_t cut) {
  const ParticleDef<real_t> pd = particle_def<real_t>(type);
  if (pd.mass <= real_t(0) || kinetic <= real_t(0)) { return real_t(0); }
  constexpr real_t kProtonMass = units::proton_mass_c2<real_t>();
  const real_t mass_rate = pd.mass / kProtonMass;

  const real_t tmax = hadron_max_secondary_energy(pd, kinetic);
  const real_t tkin = kinetic / mass_rate;  // proton-equivalent energy
  const real_t cut_energy = fmax(cut, bragg_lowest_energy<real_t>() * mass_rate);

  real_t dedx;
  if (tkin < bragg_lowest_energy<real_t>()) {
    dedx = bragg_dedx_unrestricted(m, bragg_lowest_energy<real_t>())
           * sqrt(tkin / bragg_lowest_energy<real_t>());
  } else {
    dedx = bragg_dedx_unrestricted(m, tkin);
    if (cut_energy < tmax) {
      const real_t tau = kinetic / pd.mass;
      const real_t x = cut_energy / tmax;
      dedx += (log(x) * (tau + real_t(1)) * (tau + real_t(1))
                   / (tau * (tau + real_t(2)))
               + real_t(1) - x)
              * twopi_mc2_rcl2<real_t>() * m.electron_density;
    }
  }
  return fmax(dedx, real_t(0)) * pd.charge * pd.charge;
}


// ------------------------------------------------------------------ ions (BraggIon)

/// massFactor = 1000 * amu_c2 / HeMass, from the G4BraggIonModel constructor.
template <typename real_t> __host__ __device__ inline real_t bragg_ion_mass_factor() {
  constexpr real_t kAmu = units::amu_c2<real_t>();
  constexpr real_t kHeMass = real_t(3727.379);
  return real_t(1000) * kAmu / kHeMass;
}
/// rateMassHe2p = HeMass / proton_mass_c2.
template <typename real_t> __host__ __device__ inline real_t rate_mass_he2p() {
  constexpr real_t kProtonMass = units::proton_mass_c2<real_t>();
  constexpr real_t kHeMass = real_t(3727.379);
  return kHeMass / kProtonMass;
}

/// Effective charge squared of a helium ion.
/// Verbatim from G4BraggIonModel::HeEffChargeSquare. Distinct from the
/// G4ionEffectiveCharge polynomial used by the Bethe-Bloch corrections, though it shares
/// the same six coefficients.
///
/// @param z    material effective atomic number
/// @param e_he helium kinetic energy in MeV
template <typename real_t>
__host__ __device__ inline real_t he_eff_charge_square(real_t z, real_t e_he) {
  constexpr real_t c[6] = {real_t(0.2865),  real_t(0.1266),  real_t(-0.001429),
                           real_t(0.02402), real_t(-0.01135), real_t(0.001475)};
  const real_t e = fmax(real_t(0), log(e_he * bragg_ion_mass_factor<real_t>()));
  real_t x = c[0], y = real_t(1);
  for (int i = 1; i < 6; ++i) {
    y *= e;
    x += y * c[i];
  }
  real_t w = real_t(7.6) - e;
  w = real_t(1) + (real_t(0.007) + real_t(0.00005) * z) * exp(-w * w);
  return real_t(4) * (real_t(1) - exp(-x)) * w * w;
}

/// Alpha electronic stopping power for one element, in eV cm^2 / 1e15 atoms.
/// Verbatim from G4BraggIonModel::ElectronicStoppingPower. A different parameterisation
/// from the proton one: five coefficients all used, and a separate low-energy branch.
///
/// @param t_he helium kinetic energy in MeV
template <typename real_t>
__host__ __device__ inline real_t ziegler_ion_stopping_power(int z, real_t t_he) {
  int i = z - 1;
  if (i < 0) { i = 0; }
  if (i > 91) { i = 91; }
  const real_t* a = data::ziegler_ion_a<real_t>();
  const real_t slow0 = a[i * 5 + 0];
  const real_t x1 = a[i * 5 + 1];
  const real_t x2 = a[i * 5 + 2];
  const real_t x3 = a[i * 5 + 3];
  const real_t x4 = a[i * 5 + 4];
  real_t ionloss;
  if (t_he < real_t(0.001)) {
    const real_t shigh =
        log(real_t(1) + x3 * real_t(1000) + x4 * real_t(0.001)) * x2 * real_t(1000);
    ionloss = slow0 * shigh * sqrt(t_he * real_t(1000)) / (slow0 + shigh);
  } else {
    const real_t slow = slow0 * exp(log(t_he * real_t(1000)) * x1);
    const real_t shigh = log(real_t(1) + x3 / t_he + x4 * t_he) * x2 / t_he;
    ionloss = slow * shigh / (slow + shigh);
  }
  return fmax(ionloss, real_t(0));
}

/// Unrestricted electronic stopping power for a helium-like ion at @p t_he, MeV/mm.
///
/// @p t_he is the *helium-equivalent* energy: the caller scales a non-alpha to the energy an
/// alpha of the same velocity would have and divides the result by the helium effective charge
/// squared, which is how G4BraggIonModel's own Ziegler branch is parameterised.
///
/// Two branches. One of the 74 NIST materials gets the tabulated ASTAR stopping power;
/// anything else gets the per-element Ziegler helium fit.
///
/// Geant4 arrives at a non-alpha's answer by a different route - `HeDEDX` reads the *proton*
/// PSTAR table at the ion's own energy, with no scaling at all - and the two agree to about
/// one part in a hundred thousand on every point in ref/oracle/bragg.csv. That is not luck:
/// at these energies the ASTAR and PSTAR tabulations differ by exactly the effective charge
/// squared, which is the factor being divided out. tests/test_bragg.cu asserts the alpha path
/// to a part in a million and the non-alpha path to five in a hundred thousand, separately,
/// so that the looser of the two claims cannot hide behind the tighter one.
///
/// G4BraggIonModel has a third branch - HasMaterialForHe, its own eleven-compound list - which
/// is not here. Every compound in it is also a NIST material, so branch 1 takes those; a
/// user-defined material with one of those chemical formulae falls to Ziegler here where
/// Geant4 would use the compound fit. That is the only real gap.
template <typename real_t>
__host__ __device__ inline real_t bragg_ion_dedx_unrestricted(const data::Material<real_t>& m,
                                                              real_t t_he) {
  // ICRU 90 first, as in G4BraggIonModel; -1 unless the user asked for it. The alpha table is
  // indexed by the *scaled* kinetic energy, which is what t_he already is.
  if (m.icru90 >= 0) {
    return data::icru90_mass_stopping<real_t>(m.icru90, t_he, true) * m.density / real_t(10);
  }
  if (m.nist_stopping >= 0) {
    // The table is a mass stopping power in MeV cm2/g; density is g/cm3, and 1 cm = 10 mm.
    return data::astar_mass_stopping<real_t>(m.nist_stopping, t_he) * m.density / real_t(10);
  }
  real_t eloss = real_t(0);
  for (int i = 0; i < m.n_elements; ++i) {
    const int z = static_cast<int>(m.z[i] + real_t(0.5));
    eloss += ziegler_ion_stopping_power(z, t_he) * m.n_atoms[i];
  }
  return eloss * ziegler_factor<real_t>();
}

/// Restricted dE/dx for an ion below the Bragg/Bethe-Bloch boundary, MeV/mm.
/// Verbatim from G4BraggIonModel::ComputeDEDXPerVolume.
///
/// The parameterisation is for helium, so a non-alpha ion has its energy scaled to the
/// helium equivalent and the result divided by the helium effective charge squared; the
/// caller's own charge then multiplies back in through the delta-ray correction.
template <typename real_t>
__host__ __device__ inline real_t bragg_ion_dedx(const data::Material<real_t>& m,
                                                 ParticleType type, real_t kinetic,
                                                 real_t min_kin_energy) {
  const ParticleDef<real_t> pd = particle_def<real_t>(type);
  if (pd.mass <= real_t(0) || kinetic <= real_t(0)) { return real_t(0); }
  constexpr real_t kProtonMass = units::proton_mass_c2<real_t>();
  const real_t mass_rate = pd.mass / kProtonMass;
  const bool is_alpha = pd.is_alpha;

  const real_t tmax = hadron_max_secondary_energy(pd, kinetic);
  const real_t tmin = fmax(bragg_lowest_energy<real_t>() * mass_rate, min_kin_energy);

  real_t T = kinetic;
  // Geant4 uses <Z> = total electrons / total atoms, which is exactly z_eff.
  const real_t he_charge_square = he_eff_charge_square(m.z_eff, T);
  if (!is_alpha) { T *= rate_mass_he2p<real_t>(); }

  real_t dedx;
  if (T < bragg_lowest_energy<real_t>()) {
    dedx = bragg_ion_dedx_unrestricted(m, bragg_lowest_energy<real_t>())
           * sqrt(T / bragg_lowest_energy<real_t>());
  } else {
    dedx = bragg_ion_dedx_unrestricted(m, T);
  }
  if (!is_alpha) { dedx /= he_charge_square; }

  if (tmin < tmax) {
    const real_t tau = kinetic / pd.mass;
    const real_t x = tmin / tmax;
    real_t del = (log(x) * (tau + real_t(1)) * (tau + real_t(1)) / (tau * (tau + real_t(2)))
                  + real_t(1) - x)
                 * twopi_mc2_rcl2<real_t>() * m.electron_density;
    if (is_alpha) { del *= he_charge_square; }
    dedx += del;
  }
  return fmax(dedx, real_t(0));
}

}  // namespace g4gpu::em
