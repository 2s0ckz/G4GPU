// Nuclear stopping, transcribed from G4ICRU49NuclearStoppingModel (11.1.1).
//
// G4EmStandardPhysics registers G4NuclearStopping for protons and for every ion. It is the
// energy a slow heavy particle loses to elastic recoils of whole nuclei rather than to
// atomic electrons - negligible at high energy, but comparable to the electronic stopping
// power at the very end of an ion track.
//
// The process is active only below z1^2 MeV per nucleon, which is where G4BraggIonModel is
// also in play, so this is strictly a low-energy correction.
//
// Not transcribed: the Gaussian fluctuation on the loss. G4VEmModel::lossFlucFlag defaults
// to true, so Geant4 applies straggling inside ComputeDEDXPerVolume itself and returns a
// different number every call. This port computes the mean and leaves straggling to the
// transport layer, which is why tests/test_nuclear_stopping.cu turns the flag off in the
// oracle before comparing.
#pragma once
#include <cmath>
#include "core/particle.cuh"
#include "core/units.cuh"
#include "data/materials.cuh"
#include "data/nuclear_stopping_table.cuh"

namespace g4gpu::em {

/// Nuclear stopping power for one (projectile, target) pair, in eV cm^2 / 1e15 atoms.
/// Verbatim from G4ICRU49NuclearStoppingModel::NuclearStoppingPower.
///
/// @param kinetic projectile kinetic energy, MeV
/// @param mass1   projectile mass in amu
/// @param mass2   target nucleon number
template <typename real_t>
__host__ __device__ inline real_t nuclear_stopping_power(real_t kinetic, real_t z1, real_t z2,
                                                         real_t mass1, real_t mass2) {
  const real_t energy = kinetic * real_t(1e3);  // keV
  const real_t z12 = z1 * z2;
  if (z12 <= real_t(0)) { return real_t(0); }
  const int iz1 = (z1 < real_t(99)) ? static_cast<int>(z1 + real_t(0.5)) : 99;
  const int iz2 = (z2 < real_t(99)) ? static_cast<int>(z2 + real_t(0.5)) : 99;

  real_t rm;
  if (z1 > real_t(1.5)) {
    // Z23[Z] is Z^0.23, not Z^(2/3): G4ICRU49NuclearStoppingModel::InitialiseArray fills it
    // with powZ(i, 0.23) despite the name, and Z23[1] = 1.
    const real_t a = (iz1 == 1) ? real_t(1) : pow(real_t(iz1), real_t(0.23));
    const real_t b = (iz2 == 1) ? real_t(1) : pow(real_t(iz2), real_t(0.23));
    rm = (mass1 + mass2) * (a + b);
  } else {
    rm = (mass1 + mass2) * pow(real_t(iz2), real_t(1) / real_t(3));
  }
  if (rm <= real_t(0)) { return real_t(0); }
  const real_t er = real_t(32.536) * mass2 * energy / (z12 * rm);  // reduced energy

  const real_t* ne = data::nuca_energy<real_t>();
  const real_t* nl = data::nuca_loss<real_t>();
  real_t nloss = real_t(0);
  if (er >= ne[0]) {
    nloss = nl[0];
  } else {
    // The grid descends, so walk it from the bottom as Geant4 does.
    for (int i = data::kNucaPoints - 2; i >= 0; --i) {
      const real_t edi = ne[i];
      if (er <= edi) {
        const real_t edi1 = ne[i + 1];
        nloss = (nl[i] - nl[i + 1]) * (er - edi1) / (edi - edi1) + nl[i + 1];
        break;
      }
    }
  }
  nloss *= real_t(8.462) * z12 * mass1 / rm;  // back to eV / (1e15 atoms/cm^2)
  return fmax(nloss, real_t(0));
}

/// Nuclear stopping dE/dx per volume, MeV/mm.
/// Verbatim from G4ICRU49NuclearStoppingModel::ComputeDEDXPerVolume, including the
/// z1^2 MeV per nucleon cut-off above which the process returns zero.
template <typename real_t>
__host__ __device__ inline real_t nuclear_stopping_dedx(const data::Material<real_t>& m,
                                                        ParticleType type, real_t kinetic) {
  const ParticleDef<real_t> pd = particle_def<real_t>(type);
  if (kinetic <= real_t(0) || pd.mass <= real_t(0)) { return real_t(0); }
  constexpr real_t kProtonMass = units::proton_mass_c2<real_t>();
  constexpr real_t kAmu = units::amu_c2<real_t>();
  const real_t z1 = fabs(pd.charge);
  if (kinetic * kProtonMass / pd.mass > z1 * z1) { return real_t(0); }  // > z1^2 MeV
  const real_t mass1 = pd.mass / kAmu;

  real_t nloss = real_t(0);
  for (int i = 0; i < m.n_elements; ++i) {
    const real_t z2 = m.z[i];
    // Geant4 uses G4Element::GetN(), the abundance-weighted nucleon number.
    const real_t mass2 = data::atomic_mass<real_t>(static_cast<int>(z2 + real_t(0.5)));
    nloss += nuclear_stopping_power(kinetic, z1, z2, mass1, mass2) * m.n_atoms[i];
  }
  // eV * cm^2 * 1e-15 expressed in MeV and mm.
  constexpr real_t kZieglerFactor = real_t(1e-6) * real_t(100.0) * real_t(1e-15);
  return nloss * kZieglerFactor;
}

}  // namespace g4gpu::em
