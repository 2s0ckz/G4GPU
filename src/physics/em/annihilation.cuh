// Positron annihilation into two photons, transcribed from G4eeToTwoGammaModel (11.1.1).
//
// Two regimes:
//   at rest    - two back-to-back 511 keV photons, isotropic
//   in flight  - Heitler's differential cross section, sampled by rejection in log(eps),
//                with the second photon fixed by momentum conservation
//
// Polarisation is not tracked by this port, so the polarisation vectors Geant4 assigns are
// not computed. Positronium formation (fSampleAtomicPDF) is off by default in
// G4eplusAnnihilation and is not transcribed.
#pragma once
#include <cmath>
#include "core/units.cuh"
#include "core/vec3.cuh"
#include "data/materials.cuh"

namespace g4gpu::em {

/// Annihilation cross section per electron, mm^2. Verbatim from
/// G4eeToTwoGammaModel::ComputeCrossSectionPerElectron (the Heitler formula).
template <typename real_t>
__host__ __device__ inline real_t annihilation_xs_per_electron(real_t kinetic) {
  const real_t me = units::electron_mass_c2<real_t>();
  const real_t ekin = fmax(real_t(1e-6), kinetic);  // Geant4 floors at 1 eV
  const real_t tau = ekin / me;
  const real_t gam = tau + real_t(1);
  const real_t gamma2 = gam * gam;
  const real_t bg2 = tau * (tau + real_t(2));
  const real_t bg = sqrt(bg2);
  const real_t re = units::classic_electron_radius<real_t>();
  const real_t pi_rcl2 = real_t(3.14159265358979323846) * re * re;
  return pi_rcl2
         * ((gamma2 + real_t(4) * gam + real_t(1)) * log(gam + bg) - (gam + real_t(3)) * bg)
         / (bg2 * (gam + real_t(1)));
}

/// Macroscopic annihilation cross section, 1/mm. Geant4 scales the per-electron cross
/// section by the electron density of the material.
template <typename real_t>
__host__ __device__ inline real_t annihilation_xs(const data::Material<real_t>& m,
                                                  real_t kinetic) {
  return m.electron_density * annihilation_xs_per_electron(kinetic);
}

template <typename real_t>
struct AnnihilationResult {
  Vec3<real_t> dir1, dir2;
  real_t energy1, energy2;
};

/// Two photons from a positron annihilating at rest: back to back, isotropic.
template <typename real_t, typename Rng>
__host__ __device__ inline AnnihilationResult<real_t> sample_annihilation_at_rest(Rng& rng) {
  AnnihilationResult<real_t> out;
  out.energy1 = out.energy2 = units::electron_mass_c2<real_t>();
  const real_t ct = real_t(2) * rng.uniform() - real_t(1);
  const real_t st = sqrt(fmax(real_t(0), real_t(1) - ct * ct));
  const real_t phi = units::twopi<real_t>() * rng.uniform();
  out.dir1 = Vec3<real_t>{st * cos(phi), st * sin(phi), ct};
  out.dir2 = Vec3<real_t>{-out.dir1.x, -out.dir1.y, -out.dir1.z};
  return out;
}

/// Two photons from a positron annihilating in flight. Transcribed from the
/// posiKinEnergy != 0 branch of G4eeToTwoGammaModel::SampleSecondaries.
///
/// eps is the fraction of the total energy carried by the first photon; it is sampled
/// uniformly in log(eps) over [epsilmin, epsilmax] and rejected against Heitler's
/// distribution. The second photon takes the rest, in the direction that conserves momentum.
template <typename real_t, typename Rng>
__host__ __device__ inline AnnihilationResult<real_t> sample_annihilation_in_flight(
    real_t kinetic, const Vec3<real_t>& dir, Rng& rng) {
  const real_t me = units::electron_mass_c2<real_t>();
  const real_t tau = kinetic / me;
  const real_t gam = tau + real_t(1);
  const real_t tau2 = tau + real_t(2);
  const real_t sqgrate = sqrt(tau / tau2) * real_t(0.5);
  const real_t sqg2m1 = sqrt(tau * tau2);
  const real_t epsilmin = real_t(0.5) - sqgrate;
  const real_t epsilmax = real_t(0.5) + sqgrate;
  const real_t epsilqot = epsilmax / epsilmin;

  real_t epsil = epsilmin;
  for (int guard = 0; guard < 1000; ++guard) {
    epsil = epsilmin * exp(log(epsilqot) * rng.uniform());
    const real_t greject =
        real_t(1) - epsil + (real_t(2) * gam * epsil - real_t(1)) / (epsil * tau2 * tau2);
    if (greject >= rng.uniform()) { break; }
  }

  real_t cost = (epsil * tau2 - real_t(1)) / (epsil * sqg2m1);
  if (cost > real_t(1)) { cost = real_t(1); }
  if (cost < real_t(-1)) { cost = real_t(-1); }
  const real_t sint = sqrt((real_t(1) + cost) * (real_t(1) - cost));
  const real_t phi = units::twopi<real_t>() * rng.uniform();

  AnnihilationResult<real_t> out;
  const real_t total_energy = kinetic + real_t(2) * me;
  out.energy1 = epsil * total_energy;
  out.energy2 = (real_t(1) - epsil) * total_energy;
  out.dir1 = normalize(rotate_uz(Vec3<real_t>{sint * cos(phi), sint * sin(phi), cost}, dir));

  // Second photon direction from momentum conservation, as Geant4 does it.
  const real_t posi_p = sqrt(kinetic * (kinetic + real_t(2) * me));
  const Vec3<real_t> d{posi_p * dir.x - out.energy1 * out.dir1.x,
                       posi_p * dir.y - out.energy1 * out.dir1.y,
                       posi_p * dir.z - out.energy1 * out.dir1.z};
  const real_t dn = sqrt(d.x * d.x + d.y * d.y + d.z * d.z);
  out.dir2 = (dn > real_t(0)) ? Vec3<real_t>{d.x / dn, d.y / dn, d.z / dn}
                              : Vec3<real_t>{-out.dir1.x, -out.dir1.y, -out.dir1.z};
  return out;
}

}  // namespace g4gpu::em
