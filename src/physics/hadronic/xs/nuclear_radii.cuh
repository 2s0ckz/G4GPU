// G4NuclearRadii - the nuclear radii and the two Coulomb factors every hadronic cross
// section in this directory is built out of.
//
// Transcribed from source/processes/hadronic/util/src/G4NuclearRadii.cc (11.1.1):
// ExplicitRadius, Radius, RadiusRMS, RadiusNNGG, RadiusECS, RadiusHNGG, RadiusKNGG, RadiusND,
// RadiusCB, ParticleRadius, both CoulombFactor overloads, and the r0[93] table.
//
// There are seven different radii here and they are not variants of one formula - each is
// fitted for the model that calls it, and picking the wrong one is a silent per-cent error:
//
//   RadiusHNGG   hadron-nucleus Glauber-Gribov (G4ComponentGGHadronNucleusXsc)
//   RadiusKNGG   kaon-nucleus Glauber-Gribov, 1.3 fm * A^(1/3) flat
//   Radius       nucleus-nucleus Glauber-Gribov (G4ComponentGGNuclNuclXsc)
//   RadiusCB     the Coulomb-barrier radius, r0[Z] * A^(1/3), with its own per-Z table
//   RadiusNNGG / RadiusRMS / RadiusECS / RadiusND
//                used by the elastic final-state models and the electromagnetic-dissociation
//                cross section, transcribed here because they are the same file and the same
//                ExplicitRadius special cases, not because this package calls them.
//
// ExplicitRadius is shared by five of them and is what makes the light nuclei not follow any
// A^(1/3) law: p 0.895, d 2.13, t 1.80, He3 1.96, He4 1.68, Li7 2.40, Be9 2.51 fm. The test
// covers Z <= 4 for that reason.
#pragma once
#include <cmath>

#include "core/units.cuh"
#include "data/g4pow.hh"
#include "data/nuclei_mass_ame12.hh"
#include "physics/hadronic/xs/g4pow_extra.cuh"
#include "physics/hadronic/xs/projectile.cuh"

namespace g4gpu::hadronic::xs {

/// CLHEP fermi, in this port's mm-based units: 1e-15 m = 1e-12 mm.
template <typename real_t> __host__ __device__ constexpr real_t fermi() {
  return real_t(1e-12);
}
/// CLHEP millibarn = 1e-3 * barn, derived the way CLHEP derives it rather than written as
/// 1e-25, so that the last bit matches whatever `barn` is in core/units.cuh.
template <typename real_t> __host__ __device__ constexpr real_t millibarn() {
  return real_t(1e-3) * units::barn<real_t>();
}

/// `fAlpha` at the top of G4NuclearRadii.cc: half the fine-structure constant times hbar*c.
/// It is a half here and a whole in G4HadronNucleonXsc::CoulombBarrier, which divides by
/// 2*(pR+tR) instead - the same number written two ways, and worth noticing before deciding
/// one of them is a factor-of-two bug.
template <typename real_t> __host__ __device__ constexpr real_t nuclear_radii_alpha() {
  return real_t(0.5) * units::fine_structure_const<real_t>() * units::hbarc<real_t>();
}

/// G4NuclearRadii::r0 - the Coulomb-barrier radius parameter, fm, indexed by Z (0..92).
__host__ __device__ inline const double* coulomb_r0_fm() {
  static const double v[93] = {
    1.2, 1.3, 1.3, 1.3, 1.3, 1.17, 1.54, 1.65, 1.71, 1.7,
    1.75, 1.7, 1.57, 1.53, 1.4, 1.3, 1.30, 1.44, 1.4, 1.4,
    1.4, 1.4, 1.4, 1.46, 1.4, 1.4, 1.46, 1.55, 1.5, 1.38,
    1.48, 1.4, 1.4, 1.4, 1.46, 1.4, 1.4, 1.4, 1.4, 1.4,
    1.45, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.45, 1.48, 1.4,
    1.52, 1.46, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4,
    1.5, 1.4, 1.4, 1.4, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3,
    1.4, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.33,
    1.43, 1.3, 1.32, 1.34, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3,
    1.3, 1.3, 1.3};
  return v;
}

/// G4NuclearRadii::ExplicitRadius - measured rms radii for the seven light nuclei that do not
/// follow any A^(1/3) law. Returns 0 when there is no explicit value, which is the signal
/// every caller below tests for.
template <typename real_t>
__host__ __device__ inline real_t nr_explicit_radius(int z, int a) {
  real_t R = real_t(0);
  if (z <= 4) {
    if (a == 1) {
      R = real_t(0.895) * fermi<real_t>();
    } else if (a == 2) {
      R = real_t(2.13) * fermi<real_t>();
    } else if (z == 1 && a == 3) {
      R = real_t(1.80) * fermi<real_t>();
    } else if (z == 2 && a == 3) {
      R = real_t(1.96) * fermi<real_t>();
    } else if (z == 2 && a == 4) {
      R = real_t(1.68) * fermi<real_t>();
    } else if (z == 3) {
      R = real_t(2.40) * fermi<real_t>();
    } else if (z == 4) {
      R = real_t(2.51) * fermi<real_t>();
    }
  }
  return R;
}

/// G4NuclearRadii::Radius - the nucleus-nucleus radius, used by G4ComponentGGNuclNuclXsc.
///
/// The `y*(x - 1/x)` form below A = 50 with four different y, then A^0.27 above it. powZ and
/// not powA for that power: see g4pow_extra.cuh.
template <typename real_t>
__host__ __device__ inline real_t nr_radius(int z, int a) {
  real_t R = nr_explicit_radius<real_t>(z, a);
  if (R == real_t(0)) {
    if (a <= 50) {
      real_t y = real_t(1.1);
      if (a <= 15) {
        y = real_t(1.26);
      } else if (a <= 20) {
        y = real_t(1.19);
      } else if (a <= 30) {
        y = real_t(1.12);
      }
      const real_t x = data::g4pow_z13<real_t>(a);
      R = y * (x - real_t(1) / x);
    } else {
      R = g4pow_pow_z<real_t>(a, real_t(0.27));
    }
    R *= fermi<real_t>();
  }
  return R;
}

/// G4NuclearRadii::RadiusRMS.
template <typename real_t>
__host__ __device__ inline real_t nr_radius_rms(int z, int a) {
  real_t R = nr_explicit_radius<real_t>(z, a);
  if (R == real_t(0)) {
    R = real_t(1.24) * g4pow_pow_z<real_t>(a, real_t(0.28)) * fermi<real_t>();
  }
  return R;
}

/// G4NuclearRadii::RadiusNNGG.
template <typename real_t>
__host__ __device__ inline real_t nr_radius_nngg(int z, int a) {
  real_t R = nr_explicit_radius<real_t>(z, a);
  if (R == real_t(0)) {
    if (a > 20) {
      R = real_t(1.08) * data::g4pow_z13<real_t>(a)
          * (real_t(0.85) + real_t(0.15) * exp(-static_cast<real_t>(a - 21) / real_t(40.)));
    } else {
      R = real_t(1.08) * data::g4pow_z13<real_t>(a)
          * (real_t(1.0) + real_t(0.3) * exp(-static_cast<real_t>(a - 21) / real_t(10.)));
    }
    R *= fermi<real_t>();
  }
  return R;
}

/// G4NuclearRadii::RadiusECS. Note it has no ExplicitRadius branch and returns 0 above A = 50.
template <typename real_t>
__host__ __device__ inline real_t nr_radius_ecs(int z, int a) {
  real_t R = real_t(0);
  constexpr real_t c1 = real_t(0.77329745);
  constexpr real_t c2 = real_t(1.38206072);
  constexpr real_t c3 = real_t(30.28295235);
  if (a <= 30) {
    const real_t af = static_cast<real_t>(a);
    const real_t vn = real_t(0.5) * af + g4pow_pow_n<real_t>(real_t(0.028) * af, 2)
                      - g4pow_pow_n<real_t>(real_t(0.011) * af, 3);
    const real_t dev = vn - static_cast<real_t>(a - z);
    R = c1 * data::g4pow_z13<real_t>(a) + c2 / data::g4pow_z13<real_t>(a)
        + c3 * dev * dev / static_cast<real_t>(a * a);
  } else if (a <= 50) {
    const real_t y = real_t(1.1);
    const real_t x = data::g4pow_z13<real_t>(a);
    R = y * (x - real_t(1) / x);
  }
  return R * fermi<real_t>();
}

/// G4NuclearRadii::RadiusHNGG - the hadron-nucleus Glauber-Gribov radius. No ExplicitRadius
/// branch: even for A = 1 it is 1.08 fm * (1 + 0.1*exp(19/20)).
template <typename real_t>
__host__ __device__ inline real_t nr_radius_hngg(int a) {
  real_t R = fermi<real_t>();
  if (a > 20) {
    R *= real_t(1.08) * data::g4pow_z13<real_t>(a)
         * (real_t(0.8) + real_t(0.2) * exp(-static_cast<real_t>(a - 20) / real_t(20.)));
  } else {
    R *= real_t(1.08) * data::g4pow_z13<real_t>(a)
         * (real_t(1.0) + real_t(0.1) * exp(-static_cast<real_t>(a - 20) / real_t(20.)));
  }
  return R;
}

/// G4NuclearRadii::RadiusKNGG.
template <typename real_t>
__host__ __device__ inline real_t nr_radius_kngg(int a) {
  return real_t(1.3) * fermi<real_t>() * data::g4pow_z13<real_t>(a);
}

/// G4NuclearRadii::RadiusND. One fermi for every A except A = 1, where it is 0.895 - the two
/// A^(1/3) branches that used to be here are commented out in 11.1.1 and are not reinstated.
template <typename real_t>
__host__ __device__ inline real_t nr_radius_nd(int a) {
  const real_t R = fermi<real_t>();
  return (a == 1) ? R * real_t(0.895) : R;
}

/// G4NuclearRadii::RadiusCB - the Coulomb-barrier radius. Z is clamped to 92 for the r0 table
/// but A^(1/3) uses the unclamped A.
template <typename real_t>
__host__ __device__ inline real_t nr_radius_cb(int z, int a) {
  real_t R = nr_explicit_radius<real_t>(z, a);
  if (R == real_t(0)) {
    const int zz = (z < 92) ? z : 92;
    R = static_cast<real_t>(coulomb_r0_fm()[zz]) * data::g4pow_z13<real_t>(a)
        * fermi<real_t>();
  }
  return R;
}

/// G4NuclearRadii::ParticleRadius - keyed on |PDG|, so pi- and K- get the pi+ and K+ radius.
template <typename real_t>
__host__ __device__ inline real_t nr_particle_radius(const Projectile<real_t>& p) {
  real_t R = fermi<real_t>();
  const int apdg = (p.pdg < 0) ? -p.pdg : p.pdg;
  if (apdg == pdg::kNeutron || apdg == pdg::kProton) {
    R *= real_t(0.895);
  } else if (apdg == pdg::kPiPlus) {
    R *= real_t(0.663);
  } else if (apdg == pdg::kKaonPlus) {
    R *= real_t(0.340);
  } else {
    R *= real_t(0.5);
  }
  return R;
}

/// G4NuclearRadii::CoulombFactor(particle, nucleon, ekin) - the hadron-on-free-nucleon form,
/// with the target radius fixed at the proton's 0.895 fm.
///
/// Returns exactly zero below the barrier, which is a threshold and not a suppression: a
/// wrong target mass here does not scale the cross section, it moves where it becomes zero.
template <typename real_t>
__host__ __device__ inline real_t nr_coulomb_factor(const Projectile<real_t>& p,
                                                    const Projectile<real_t>& nucleon,
                                                    real_t ekin) {
  const real_t tR = real_t(0.895) * fermi<real_t>();
  const real_t pR = nr_particle_radius<real_t>(p);
  const real_t pZ = p.charge;
  const real_t tZ = nucleon.charge;
  const real_t pM = p.mass;
  const real_t tM = nucleon.mass;
  const real_t pElab = ekin + pM;
  const real_t totTcm = sqrt(pM * pM + tM * tM + real_t(2.) * pElab * tM) - pM - tM;
  const real_t bC = nuclear_radii_alpha<real_t>() * pZ * tZ / (pR + tR);
  return (totTcm > bC) ? real_t(1.) - bC / totTcm : real_t(0.0);
}

/// G4NuclearRadii::CoulombFactor(Z, A, particle, ekin) - the hadron-on-nucleus form, used by
/// all four BGG classes below their table's lower edge.
///
/// The nuclear mass comes from data::nuclear_mass; a nuclide that table does not carry gives
/// zero, and the caller must have tested data::nuclear_mass_known rather than let a zero mass
/// through - a zero target mass makes totTcm zero and the factor zero, i.e. it silently turns
/// a cross section off instead of reporting that the mass is missing.
template <typename real_t>
__host__ __device__ inline real_t nr_coulomb_factor_nucleus(int z, int a,
                                                            const Projectile<real_t>& p,
                                                            real_t ekin) {
  const real_t tR = nr_radius_cb<real_t>(z, a);
  const real_t pR = nr_particle_radius<real_t>(p);
  const real_t pZ = p.charge;
  const real_t pM = p.mass;
  const real_t tM = data::nuclear_mass<real_t>(a, z);
  const real_t pElab = ekin + pM;
  const real_t totTcm = sqrt(pM * pM + tM * tM + real_t(2.) * pElab * tM) - pM - tM;
  const real_t bC = nuclear_radii_alpha<real_t>() * pZ * static_cast<real_t>(z) / (pR + tR);
  return (totTcm > bC) ? real_t(1.) - bC / totTcm : real_t(0.0);
}

}  // namespace g4gpu::hadronic::xs
