// The Wentzel single-scattering cross section, transcribed from G4WentzelOKandVIxSection
// (11.1.1).
//
// This is the engine shared by G4WentzelVIModel (multiple scattering above 100 MeV for
// e-/e+, and for every charged hadron at all energies) and G4eCoulombScatteringModel (the
// single-scattering process that runs alongside it). Both are registered by
// G4EmStandardPhysics, so B1's physics list contains them even though B1 never reaches the
// energies where they act on electrons.
//
// The model is a screened Rutherford cross section: nuclear scattering with a Moliere-style
// screening angle, plus scattering off the atomic electrons, each cut off at a maximum
// angle. Above that angle the process is handled discretely by G4CoulombScattering rather
// than by multiple scattering, which is what makes the pair "combined".
//
// The Mott/Rutherford ratio (G4ScreeningMottCrossSection) is in data/mott.hh and is used by
// wentzel_msc.cuh's single-scattering sampler for e+-, as Geant4 uses it. What that leaves out
// is the *cross section* half of that class - NuclearCrossSection and its form-factor
// variants - which no standard physics list reaches through WentzelVI: the transport cross
// section here is G4WentzelOKandVIxSection's own, and it is checked against it to 0.0003%.
//
// The nuclear form factor beyond the exponential one is absent. G4EmParameters defaults
// NuclearFormfactorType to fExponentialNF, so the Gaussian and flat variants need a setter
// nothing here has.
#pragma once
#include <cmath>
#include "core/particle.cuh"
#include "core/units.cuh"
#include "data/materials.cuh"

namespace g4gpu::em {

/// Per-element screening and form-factor constants, from
/// G4WentzelOKandVIxSection::InitialiseA. G4EmParameters defaults ScreeningFactor to 1.
template <typename real_t>
struct WentzelElement {
  real_t screen_r2;       ///< ScreenRSquare[Z], used for heavy particles
  real_t screen_r2_elec;  ///< ScreenRSquareElec[Z], used for e-/e+
  real_t form_factor;     ///< FormFactor[Z]
};

template <typename real_t>
__host__ __device__ inline WentzelElement<real_t> wentzel_element(int z) {
  constexpr real_t alpha2 = units::fine_structure_const<real_t>() * units::fine_structure_const<real_t>();
  const real_t me = units::electron_mass_c2<real_t>();
  const real_t a0 = me / real_t(0.88534);
  constexpr real_t constn = real_t(6.937e-6);  // 1/MeV^2
  constexpr real_t fct = real_t(1.0);          // G4EmParameters::ScreeningFactor
  const real_t afact = real_t(0.5) * fct * alpha2 * a0 * a0;

  WentzelElement<real_t> e;
  if (z <= 1) {
    e.screen_r2 = afact;
    e.screen_r2_elec = afact;
    e.form_factor = real_t(3.097e-6);
    return e;
  }
  const real_t x = pow(real_t(z), real_t(1) / real_t(3));
  e.screen_r2 = afact * (real_t(1) + exp(-real_t(z) * real_t(z) * real_t(0.001))) * x * x;
  e.screen_r2_elec = afact * x * x;
  const real_t a27 = pow(data::atomic_mass<real_t>(z), real_t(0.27));
  e.form_factor = constn * a27 * a27;
  return e;
}

/// The kinematic state SetupKinematic and SetupTarget build up.
template <typename real_t>
struct WentzelState {
  real_t mom2;           ///< p^2 c^2, MeV^2
  real_t inv_beta2;      ///< 1 + m^2/p^2
  real_t fact_b;         ///< spin / inv_beta2
  real_t cos_tet_max_nuc;
  real_t cos_tet_max_elec;
  real_t screen_z;
  real_t kin_factor;
  real_t form_fact_a;
  /// G4WentzelOKandVIxSection sets this to 1 + 2e-4 Z^2 for electrons only (it is a crude
  /// normalisation stand-in applied alongside G4ScreeningMottCrossSection) and leaves it at
  /// 1 for every other particle, positrons included.
  real_t mott_factor;
  /// Lab kinetic energy and mass, carried because the Mott rejection needs them per element:
  /// G4ScreeningMottCrossSection::SetupKinematic works in the projectile-nucleus relative
  /// system, so its beta depends on the target Z and cannot be precomputed here.
  real_t tkin;
  real_t mass;
  /// True for e- and e+. G4WentzelOKandVIxSection builds a G4ScreeningMottCrossSection for
  /// those two and no others - "Mott corrections always added" - and uses its
  /// Mott/Rutherford ratio as the single-scattering rejection function in their place of the
  /// analytic Rutherford-plus-spin expression.
  bool use_mott;
};

/// 2 pi (m_e c^2 r_e)^2.
template <typename real_t> __host__ __device__ inline real_t wentzel_coeff() {
  const real_t p0 = units::electron_mass_c2<real_t>() * units::classic_electron_radius<real_t>();
  return units::twopi<real_t>() * p0 * p0;
}

/// 0.5 * (FactorForAngleLimit * hbarc / fermi)^2, with the parameter at its default of 1.
template <typename real_t> __host__ __device__ inline real_t wentzel_factor_a2() {
  // CLHEP's own hbarc, not 197.3269804e-12: it derives the value from h, c and the unit
  // system and comes out at 1.9732698045930245e-10 MeV*mm. See core/units.cuh (RISK.md O8).
  constexpr real_t hbarc = units::hbarc<real_t>();
  constexpr real_t fermi = real_t(1e-12);            // mm
  const real_t a = hbarc / fermi;
  return real_t(0.5) * a * a;
}

/// Verbatim from G4WentzelOKandVIxSection::ComputeMaxElectronScattering.
template <typename real_t>
__host__ __device__ inline real_t wentzel_cos_max_elec(const ParticleDef<real_t>& pd,
                                                       ParticleType type, real_t tkin,
                                                       real_t mom2, real_t cut) {
  const real_t me = units::electron_mass_c2<real_t>();
  real_t cos_max = real_t(1);
  if (pd.mass > real_t(1)) {  // heavier than 1 MeV: the hadron/muon branch
    const real_t ratio = me / pd.mass;
    const real_t tau = tkin / pd.mass;
    const real_t tmax = real_t(2) * me * tau * (tau + real_t(2))
                        / (real_t(1) + real_t(2) * ratio * (tau + real_t(1)) + ratio * ratio);
    cos_max = real_t(1) - fmin(cut, tmax) * me / mom2;
  } else {
    const bool is_electron = (type == ParticleType::kElectron);
    const real_t tmax = is_electron ? real_t(0.5) * tkin : tkin;
    const real_t t = fmin(cut, tmax);
    const real_t mom21 = t * (t + real_t(2) * me);
    const real_t t1 = tkin - t;
    if (t1 > real_t(0)) {
      const real_t mom22 = t1 * (t1 + real_t(2) * pd.mass);
      const real_t ctm = (mom2 + mom22 - mom21) * real_t(0.5) / sqrt(mom2 * mom22);
      if (ctm < real_t(1)) { cos_max = ctm; }
      if (is_electron && cos_max < real_t(0)) { cos_max = real_t(0); }
    }
  }
  return cos_max;
}

/// SetupKinematic followed by SetupTarget, for one element.
/// Verbatim from G4WentzelOKandVIxSection, with isCombined true (which is how
/// G4WentzelVIModel constructs it).
///
/// @param inv_a23 material <A^(-2/3)>, G4IonisParamMat::GetInvA23
template <typename real_t>
__host__ __device__ inline WentzelState<real_t> wentzel_setup(const ParticleDef<real_t>& pd,
                                                              ParticleType type, real_t tkin,
                                                              real_t inv_a23, int z,
                                                              real_t cut,
                                                              real_t cos_theta_lim) {
  constexpr real_t alpha2 = units::fine_structure_const<real_t>() * units::fine_structure_const<real_t>();
  WentzelState<real_t> s;
  const real_t spin = (pd.spin != real_t(0)) ? real_t(0.5) : real_t(0);
  const real_t charge_square = pd.charge * pd.charge;

  s.mom2 = tkin * (tkin + real_t(2) * pd.mass);
  s.inv_beta2 = real_t(1) + pd.mass * pd.mass / s.mom2;
  s.fact_b = spin / s.inv_beta2;
  // isCombined: the nuclear cut-off angle is the tighter of the model limit and the
  // form-factor limit.
  s.cos_tet_max_nuc =
      fmax(cos_theta_lim, real_t(1) - wentzel_factor_a2<real_t>() * inv_a23 / s.mom2);

  const int zt = (z < 99) ? z : 99;
  const WentzelElement<real_t> el = wentzel_element<real_t>(zt);
  s.kin_factor = wentzel_coeff<real_t>() * real_t(zt) * charge_square * s.inv_beta2 / s.mom2;
  s.mott_factor = (type == ParticleType::kElectron)
                      ? real_t(1) + real_t(2.0e-4) * real_t(zt) * real_t(zt)
                      : real_t(1);
  s.tkin = tkin;
  s.mass = pd.mass;
  s.use_mott = (type == ParticleType::kElectron || type == ParticleType::kPositron);

  if (zt == 1) {
    s.screen_z = el.screen_r2 / s.mom2;
  } else if (pd.mass > real_t(1)) {
    s.screen_z = fmin(real_t(zt) * real_t(1.13),
                      real_t(1.13) + real_t(3.76) * real_t(zt) * real_t(zt) * s.inv_beta2
                                         * alpha2 * charge_square)
                 * el.screen_r2 / s.mom2;
    } else {
    const real_t tau = tkin / pd.mass;
    const real_t z23 = pow(real_t(zt), real_t(2) / real_t(3));
    s.screen_z = fmin(real_t(zt) * real_t(1.13),
                      real_t(1.13)
                          + real_t(3.76) * real_t(zt) * real_t(zt) * s.inv_beta2 * alpha2
                                * sqrt(tau / (tau + z23)))
                 * el.screen_r2_elec / s.mom2;
  }
  // A proton on hydrogen cannot scatter backwards in the CM frame.
  if (zt == 1 && type == ParticleType::kProton && s.cos_tet_max_nuc < real_t(0)) {
    s.cos_tet_max_nuc = real_t(0);
  }
  s.form_fact_a = el.form_factor * s.mom2;
  s.cos_tet_max_elec = wentzel_cos_max_elec(pd, type, tkin, s.mom2, cut);
  return s;
}

/// Nuclear cross section between two angles.
/// Verbatim from G4WentzelOKandVIxSection::ComputeNuclearCrossSection (fMottFactor = 1,
/// since the Mott correction is not transcribed).
template <typename real_t>
__host__ __device__ inline real_t wentzel_nuclear_xs(const WentzelState<real_t>& s, int z,
                                                     real_t cos_min, real_t cos_max) {
  return real_t(z) * s.kin_factor * s.mott_factor * (cos_min - cos_max)
         / ((real_t(1) - cos_min + s.screen_z) * (real_t(1) - cos_max + s.screen_z));
}

/// Electron cross section between two angles.
/// Verbatim from G4WentzelOKandVIxSection::ComputeElectronCrossSection.
template <typename real_t>
__host__ __device__ inline real_t wentzel_electron_xs(const WentzelState<real_t>& s,
                                                      real_t cos_min, real_t cos_max) {
  const real_t c1 = fmax(cos_min, s.cos_tet_max_elec);
  const real_t c2 = fmax(cos_max, s.cos_tet_max_elec);
  return (c1 <= c2) ? real_t(0)
                    : s.kin_factor * s.mott_factor * (c1 - c2)
                          / ((real_t(1) - c1 + s.screen_z) * (real_t(1) - c2 + s.screen_z));
}

/// Transport cross section per atom, integrated from cos_max up to 1.
/// Verbatim from G4WentzelOKandVIxSection::ComputeTransportCrossSectionPerAtom: the electron
/// and nuclear contributions use the same integral, evaluated at different cut-off angles,
/// with a series expansion where the argument is small.
template <typename real_t>
__host__ __device__ inline real_t wentzel_transport_xs_per_atom(const WentzelState<real_t>& s,
                                                                int z, real_t cos_max) {
  constexpr real_t numlimit = real_t(0.1);
  if (cos_max >= real_t(1)) { return real_t(0); }
  const real_t fb = s.screen_z * s.fact_b;

  auto integral = [&](real_t cost) {
    const real_t x = (real_t(1) - cost) / s.screen_z;
    real_t y;
    if (x < numlimit) {
      const real_t x2 = real_t(0.5) * x * x;
      y = x2 * ((real_t(1) - real_t(1.3333333) * x + real_t(3) * x2)
                - fb * x * (real_t(0.6666667) - x));
    } else {
      const real_t x1 = x / (real_t(1) + x);
      const real_t xlog = log(real_t(1) + x);
      y = xlog - x1 - fb * (x + x1 - real_t(2) * xlog);
    }
    return fmax(y, real_t(0));
  };

  real_t xs = real_t(0);
  const real_t costm = fmax(cos_max, s.cos_tet_max_elec);
  if (costm < real_t(1)) { xs = integral(costm); }        // scattering off atomic electrons
  if (cos_max < real_t(1)) { xs += integral(cos_max) * real_t(z); }  // off the nucleus
  return xs * s.kin_factor;
}

/// Transport cross section per volume, 1/mm.
/// Mirrors G4WentzelVIModel::ComputeTransportXSectionPerVolume.
template <typename real_t>
__host__ __device__ inline real_t wentzel_transport_xs(const data::Material<real_t>& m,
                                                       ParticleType type, real_t tkin,
                                                       real_t cut, real_t cos_theta_lim) {
  const ParticleDef<real_t> pd = particle_def<real_t>(type);
  if (pd.mass <= real_t(0) || tkin <= real_t(0)) { return real_t(0); }
  real_t xs = real_t(0);
  for (int i = 0; i < m.n_elements; ++i) {
    const int z = static_cast<int>(m.z[i] + real_t(0.5));
    const WentzelState<real_t> s =
        wentzel_setup(pd, type, tkin, m.inv_a23, z, cut, cos_theta_lim);
    if (s.cos_tet_max_nuc < real_t(1)) {
      xs += m.n_atoms[i] * wentzel_transport_xs_per_atom(s, z, s.cos_tet_max_nuc);
    }
  }
  return xs;
}


/// Transport mean free path, mm - the Wentzel counterpart of urban_lambda.
///
/// This is the quantity G4WentzelVIModel builds its step limitation on. The stepping
/// algorithm itself (G4WentzelVIModel::ComputeTruePathLengthLimit, ComputeGeomPathLength and
/// SampleScattering) is NOT transcribed; only the cross sections are. See docs/RISK.md.
template <typename real_t>
__host__ __device__ inline real_t wentzel_lambda(const data::Material<real_t>& m,
                                                 ParticleType type, real_t tkin, real_t cut,
                                                 real_t cos_theta_lim) {
  const real_t xs = wentzel_transport_xs(m, type, tkin, cut, cos_theta_lim);
  return (xs > real_t(0)) ? real_t(1) / xs : real_t(1e30);
}

}  // namespace g4gpu::em
