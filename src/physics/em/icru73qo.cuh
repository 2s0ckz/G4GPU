// Low-energy ionisation for NEGATIVELY charged hadrons, transcribed from G4ICRU73QOModel
// (11.1.1).
//
// G4hIonisation and G4MuIonisation select this instead of G4BraggModel when the charge is
// negative, below 2 MeV per nucleon. It is a quantum-oscillator model: the stopping power is
// summed over atomic shells, each contributing a Bethe term L0 plus Barkas (L1) and Bloch
// (L2) corrections that are odd and even in the charge respectively. That charge-odd L1 term
// is the whole reason negative hadrons need a separate model - it is what makes an
// antiproton stop differently from a proton at the same velocity.
//
// Shell data comes from ICRU Report 73 for 25 elements; anything else falls back to an
// oscillator energy built from the atomic binding energies and the material plasma energy,
// which is transcribed here too.
#pragma once
#include <cmath>
#include "core/particle.cuh"
#include "core/units.cuh"
#include "data/atomic_masses.cuh"
#include "data/em_correction_tables.cuh"  // atomic shell occupancies
#include "data/icru73qo_tables.cuh"
#include "data/materials.cuh"
#include "physics/em/hadron_ionisation.cuh"

namespace g4gpu::em {

/// G4ICRU73QOModel::lowestKinEnergy.
template <typename real_t> __host__ __device__ constexpr real_t qo_lowest_energy() {
  return real_t(5e-3);  // 5 keV
}

/// Linear interpolation over one of the L tables, verbatim from GetL0 / GetL1 / GetL2.
/// The scan runs forward and the index is clamped, which for L0 means energies above the
/// last tabulated point extrapolate through the zero-filled 67th entry - reproduced.
template <typename real_t>
__host__ __device__ inline real_t qo_interp(const real_t* x, const real_t* y, int size,
                                            real_t norm_energy) {
  int n;
  for (n = 0; n < size; ++n) {
    if (norm_energy < x[n]) { break; }
  }
  if (n == 0) { n = 1; }
  if (n >= size) { n = size - 1; }
  const real_t d = x[n] - x[n - 1];
  if (d == real_t(0)) { return y[n - 1]; }
  return y[n - 1] + (y[n] - y[n - 1]) * (norm_energy - x[n - 1]) / d;
}

/// Number of shells used for this element.
template <typename real_t>
__host__ __device__ inline int qo_number_of_shells(int z) {
  const int idx = data::qo::qo_index_z(z);
  if (idx >= 0) { return data::qo::n_shells_for_element()[idx]; }
  return (z >= 0 && z <= 104) ? data::emcorr::n_shells()[z] : 1;
}

/// Oscillator energy for an element outside the ICRU 73 set.
/// Verbatim from G4ICRU73QOModel::GetOscillatorEnergy.
template <typename real_t>
__host__ __device__ inline real_t qo_oscillator_energy(int z, int shell) {
  const real_t plasma =
      (z >= 1 && z <= 98) ? real_t(data::nist_plasma_energy_eV()[z]) : real_t(0);
  const real_t plasma2 = plasma * plasma;
  const int* ish = data::emcorr::shell_index();
  const int* nel = data::emcorr::n_electrons();
  const int ne = (z >= 0 && z <= 104) ? nel[ish[z] + shell] : 0;
  const real_t plasmon_term =
      real_t(0.66667) * real_t(ne) * plasma2 / (real_t(z) * real_t(z));
  const real_t exphalf = exp(real_t(0.5));
  const real_t ion_term = exphalf * real_t(data::qo::binding_energy_eV()[ish[z] + shell]);
  return sqrt(ion_term * ion_term + plasmon_term);  // eV
}

/// Shell energy, eV. Verbatim from G4ICRU73QOModel::GetShellEnergy.
template <typename real_t>
__host__ __device__ inline real_t qo_shell_energy(int z, int shell) {
  const int idx = data::qo::qo_index_z(z);
  if (idx >= 0) {
    return real_t(data::qo::shell_energy_eV()[data::qo::start_elem_index()[idx] + shell]);
  }
  return qo_oscillator_energy<real_t>(z, shell);
}

/// Shell strength (occupation / Z). Verbatim from G4ICRU73QOModel::GetShellStrength.
template <typename real_t>
__host__ __device__ inline real_t qo_shell_strength(int z, int shell) {
  const int idx = data::qo::qo_index_z(z);
  if (idx >= 0) {
    return real_t(data::qo::subshell_occupation()[data::qo::start_elem_index()[idx] + shell])
           / real_t(z);
  }
  const int* ish = data::emcorr::shell_index();
  const int* nel = data::emcorr::n_electrons();
  const int ne = (z >= 0 && z <= 104) ? nel[ish[z] + shell] : 0;
  return real_t(ne) / real_t(z);
}

/// Stopping power per element, MeV/mm per atom-density unit.
/// Verbatim from G4ICRU73QOModel::DEDXPerElement.
///
/// @param kinetic proton-equivalent kinetic energy, MeV
/// @param charge  signed charge in units of e (negative for the particles this model serves)
template <typename real_t>
__host__ __device__ inline real_t qo_dedx_per_element(int atomic_number, real_t kinetic,
                                                      real_t charge) {
  const int z = (atomic_number < 97) ? atomic_number : 97;
  int n_shells = qo_number_of_shells<real_t>(z);
  if (n_shells < 1) { n_shells = 1; }
  constexpr real_t kProtonMass = units::proton_mass_c2<real_t>();
  constexpr real_t alpha = units::fine_structure_const<real_t>();
  const real_t me = units::electron_mass_c2<real_t>();

  // v/c = sqrt(2T/Mp); fBetheVelocity = alpha / (v/c).
  const real_t beta_nr = sqrt(real_t(2) * kinetic / kProtonMass);
  const real_t f_bethe_velocity = alpha / beta_nr;

  const real_t tau = kinetic / kProtonMass;
  const real_t gam = tau + real_t(1);
  const real_t bg2 = tau * (tau + real_t(2));
  const real_t beta2 = bg2 / (gam * gam);

  const real_t* l0x = data::qo::l0_x<real_t>();
  const real_t* l0y = data::qo::l0_y<real_t>();
  const real_t* l1x = data::qo::l1_x<real_t>();
  const real_t* l1y = data::qo::l1_y<real_t>();
  const real_t* l2x = data::qo::l2_x<real_t>();
  const real_t* l2y = data::qo::l2_y<real_t>();

  real_t l0_term = real_t(0), l1_term = real_t(0), l2_term = real_t(0);
  for (int s = 0; s < n_shells; ++s) {
    const real_t shell_e = qo_shell_energy<real_t>(z, s);  // eV
    if (shell_e <= real_t(0)) { continue; }
    // 2 m_e c^2 beta^2 / E_shell, with both in the same units.
    const real_t norm_energy = (real_t(2) * me * real_t(1e6) * beta2) / shell_e;
    const real_t strength = qo_shell_strength<real_t>(z, s);
    l0_term += strength * qo_interp(l0x, l0y, data::qo::kSizeL0, norm_energy);
    l1_term += strength * qo_interp(l1x, l1y, data::qo::kSizeL1, norm_energy);
    l2_term += strength * qo_interp(l2x, l2y, data::qo::kSizeL2, norm_energy);
  }

  const real_t charge_square = charge * charge;
  const real_t fb = real_t(data::qo::factor_bethe()[(z < 99) ? z : 98]);
  return real_t(2) * twopi_mc2_rcl2<real_t>() * charge_square * fb
         * (l0_term + charge * f_bethe_velocity * l1_term
            + charge_square * f_bethe_velocity * f_bethe_velocity * l2_term)
         / beta2;
}

/// Unrestricted stopping power of a material, MeV/mm.
/// Verbatim from G4ICRU73QOModel::DEDX - note the extra factor of Z per element, which is
/// there because the shell strengths were divided by Z.
template <typename real_t>
__host__ __device__ inline real_t qo_dedx_unrestricted(const data::Material<real_t>& m,
                                                       real_t kinetic, real_t charge) {
  real_t eloss = real_t(0);
  for (int i = 0; i < m.n_elements; ++i) {
    const int z = static_cast<int>(m.z[i] + real_t(0.5));
    eloss += qo_dedx_per_element(z, kinetic, charge) * m.n_atoms[i] * m.z[i];
  }
  return eloss;
}

/// Restricted dE/dx, MeV/mm.
/// Verbatim from G4ICRU73QOModel::ComputeDEDXPerVolume.
template <typename real_t>
__host__ __device__ inline real_t qo_dedx(const data::Material<real_t>& m, ParticleType type,
                                          real_t kinetic, real_t cut) {
  const ParticleDef<real_t> pd = particle_def<real_t>(type);
  if (pd.mass <= real_t(0) || kinetic <= real_t(0)) { return real_t(0); }
  constexpr real_t kProtonMass = units::proton_mass_c2<real_t>();
  const real_t mass_rate = pd.mass / kProtonMass;
  const real_t charge_square = pd.charge * pd.charge;

  const real_t tmax = hadron_max_secondary_energy(pd, kinetic);
  const real_t tkin = kinetic / mass_rate;
  const real_t cut_energy = fmax(cut, qo_lowest_energy<real_t>() * mass_rate);

  real_t dedx;
  if (tkin > qo_lowest_energy<real_t>()) {
    dedx = qo_dedx_unrestricted(m, tkin, pd.charge);
  } else {
    dedx = qo_dedx_unrestricted(m, qo_lowest_energy<real_t>(), pd.charge)
           * sqrt(tkin / qo_lowest_energy<real_t>());
  }
  if (cut_energy < tmax) {
    const real_t tau = kinetic / pd.mass;
    const real_t x = cut_energy / tmax;
    dedx += (log(x) * (tau + real_t(1)) * (tau + real_t(1)) / (tau * (tau + real_t(2)))
             + real_t(1) - x)
            * twopi_mc2_rcl2<real_t>() * charge_square * m.electron_density;
  }
  return fmax(dedx, real_t(0));
}

}  // namespace g4gpu::em
