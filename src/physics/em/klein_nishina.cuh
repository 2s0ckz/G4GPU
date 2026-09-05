// Compton scattering, Klein-Nishina, Butcher & Messel sampling (Nucl Phys 20 (1960) 15).
// Transcribed from Geant4 11.5.0 G4KleinNishinaCompton::SampleSecondaries.
// Atomic binding neglected, matching the Geant4 model.
#pragma once
#include "core/rng.cuh"
#include "core/secondary_pool.cuh"
#include "core/units.cuh"
#include "core/vec3.cuh"

namespace g4gpu::em {

/// Result of one Compton interaction. Caller applies it to the track store.
template <typename real_t>
struct ComptonResult {
  Vec3<real_t> gamma_dir;   ///< new gamma direction (unit)
  real_t gamma_energy;      ///< new gamma kinetic energy; 0 => gamma is killed
  real_t local_deposit;     ///< energy deposited at the interaction site
  bool gamma_survives;
};

/// Samples one Compton interaction on a free electron.
///
/// @param gamma_energy0  incident gamma kinetic energy
/// @param gamma_dir0     incident direction, must be unit
/// @param rng            per-track counter-based stream
/// @param secondaries    pool the recoil electron is appended to
/// @param lowest_sec_e   below this, a secondary is not created and its energy is deposited
/// @param low_e_limit    below this, no interaction occurs
///
/// Returns the gamma's new state. The recoil electron, if any, is pushed to @p secondaries.
template <typename real_t, typename Rng, typename Pool>
__host__ __device__ inline ComptonResult<real_t> sample_klein_nishina(
    real_t gamma_energy0, const Vec3<real_t>& gamma_dir0, Rng& rng, Pool& secondaries,
    int event_id, real_t lowest_sec_e, real_t low_e_limit)
{
  ComptonResult<real_t> out{gamma_dir0, gamma_energy0, real_t(0), true};
  if (gamma_energy0 <= low_e_limit) { return out; }

  const real_t E0_m = gamma_energy0 / units::electron_mass_c2<real_t>();

  // Sample the energy fraction epsilon = E1/E0 by the two-branch Butcher & Messel method.
  const real_t eps0 = real_t(1) / (real_t(1) + real_t(2) * E0_m);
  const real_t eps0sq = eps0 * eps0;
  const real_t alpha1 = -log(eps0);
  const real_t alpha2 = alpha1 + real_t(0.5) * (real_t(1) - eps0sq);

  real_t epsilon = real_t(0), epsilonsq = real_t(0);
  real_t onecost = real_t(0), sint2 = real_t(0), greject = real_t(0);

  // Geant4 caps the rejection loop at 1000 tries and treats overflow as a null interaction.
  constexpr int kMaxLoop = 1000;
  int nloop = 0;
  do {
    if (++nloop > kMaxLoop) { return out; }

    const real_t r0 = rng.uniform();
    const real_t r1 = rng.uniform();
    const real_t r2 = rng.uniform();

    if (alpha1 > alpha2 * r0) {
      epsilon = exp(-alpha1 * r1);           // eps0^r1
      epsilonsq = epsilon * epsilon;
    } else {
      epsilonsq = eps0sq + (real_t(1) - eps0sq) * r1;
      epsilon = sqrt(epsilonsq);
    }

    onecost = (real_t(1) - epsilon) / (epsilon * E0_m);
    sint2 = onecost * (real_t(2) - onecost);
    greject = real_t(1) - epsilon * sint2 / (real_t(1) + epsilonsq);
    if (greject >= r2) { break; }
  } while (true);

  // Scattered gamma direction, z-axis along the parent gamma.
  if (sint2 < real_t(0)) { sint2 = real_t(0); }
  const real_t cos_theta = real_t(1) - onecost;
  const real_t sin_theta = sqrt(sint2);
  const real_t phi = units::twopi<real_t>() * rng.uniform();

  Vec3<real_t> gamma_dir1{sin_theta * cos(phi), sin_theta * sin(phi), cos_theta};
  gamma_dir1 = rotate_uz(gamma_dir1, gamma_dir0);
  const real_t gamma_energy1 = epsilon * gamma_energy0;

  real_t edep = real_t(0);
  if (gamma_energy1 > lowest_sec_e) {
    out.gamma_dir = gamma_dir1;
    out.gamma_energy = gamma_energy1;
    out.gamma_survives = true;
  } else {
    out.gamma_energy = real_t(0);
    out.gamma_survives = false;
    edep = gamma_energy1;
  }

  // Recoil electron takes the balance; its direction follows from momentum conservation.
  const real_t e_kin = gamma_energy0 - gamma_energy1;
  if (e_kin > lowest_sec_e) {
    const Vec3<real_t> e_dir =
        normalize(gamma_energy0 * gamma_dir0 - gamma_energy1 * gamma_dir1);
    // Overflow is reported by the pool, never silently dropped.
    secondaries.push(ParticleType::kElectron, e_dir, e_kin, event_id);
  } else {
    edep += e_kin;
  }

  out.local_deposit = edep;
  return out;
}

}  // namespace g4gpu::em
