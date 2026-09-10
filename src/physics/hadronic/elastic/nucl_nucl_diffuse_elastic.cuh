// G4NuclNuclDiffuseElastic - the elastic final state QBBC's G4IonElasticPhysics gives GenericIon.
//
// Transcribed from Geant4 11.1.1,
//   processes/hadronic/models/coherent_elastic/src/G4NuclNuclDiffuseElastic.cc
//     G4NuclNuclDiffuseElastic::SampleInvariantT
//     G4NuclNuclDiffuseElastic::SampleCoulombMuCMS
//     G4NuclNuclDiffuseElastic::InitDynParameters
//   processes/hadronic/models/coherent_elastic/include/G4NuclNuclDiffuseElastic.hh
//     CalculateNuclearRad, CalculateZommerfeld, CalculateAm, CalculateRutherfordAnglePar
//
// ---------------------------------------------------------------------------------------------
// The angle table this class builds is not the table it samples from. Read this before adding it.
//
// The name and the class's 2135 lines promise a diffuse-elastic angular distribution built from
// Bessel functions and a 300 x 200 momentum/angle table. In 11.1.1 none of that is reached:
//
//   - `SampleInvariantT` computes the CMS momentum and then calls `SampleCoulombMuCMS`. The line
//     that would have used the table,
//         // t = SampleTableT( aParticle, momentumCMS, G4double(Z), G4double(A) );
//     is commented out immediately above it.
//   - `BuildAngleTable` is called only from `Initialise()`, and `Initialise()` is called from
//     nowhere in the whole source tree. The class does not override
//     `G4HadronicInteraction::InitialiseModel`, which is the hook a physics list would use, so
//     no table is ever built - `fAngleTable` stays null for the life of the run.
//   - `SampleTableThetaCMS`, `GetScatteringAngle`, `BesselJzero`, `BesselJone`,
//     `GetDiffElasticSumProbA`, the Legendre96/Legendre10 integrations, `SampleThetaCMS`,
//     `TestAngleTable` and `InitParametersGla` are therefore all dead in the sampling path.
//
// So what QBBC's ion elastic scattering actually IS, is pure screened Rutherford scattering
// truncated at the Coulomb grazing angle, and that is what is transcribed here. The table
// machinery is REFUSED by name rather than ported: porting 1500 lines that 11.1.1 cannot reach
// would be 1500 lines of unverifiable code, since no oracle can be built from a function the
// oracle's own Geant4 never calls. `docs/PORTED.md` records it as such, and if a later Geant4
// re-enables `SampleTableT` the refusal is the thing to come back to.
//
// This is the third time in this project that a commented-out line was the whole answer
// (docs/RISK.md V31, V32): a gap with a rationale over it reads as a decision.
//
// ---------------------------------------------------------------------------------------------
// One thing to know about the state.
//
// `InitDynParameters` recomputes fBeta, fZommerfeld and fAm only `if( z )` - if the projectile
// is charged. For a neutral projectile they keep whatever the previous call left, and
// `fCoulombMuC` is then built from another particle's Sommerfeld parameter. The process this
// model is registered on is attached to GenericIon only, so z >= 1 always and the branch is
// never taken; the port has no such state to leak, and `sample_invariant_t` below refuses z == 0
// by name instead of silently reproducing a stale number.
#pragma once

#include <cmath>

#include "core/units.cuh"
#include "data/g4pow.hh"
#include "physics/hadronic/elastic/hadron_elastic.cuh"
#include "physics/hadronic/process.cuh"

namespace g4gpu::physics::hadronic::elastic {

/// The empirical constructor parameters of G4NuclNuclDiffuseElastic that the live path reads.
/// The others (fCofAlphaMax, fCofAlphaCoulomb, fProfileDelta, fProfileAlpha, fCofDelta,
/// fCofAlpha, fEnergyBin = 300, fAngleBin = 200, lowEnergyRecoilLimit = 100 keV) belong to the
/// dead table path and are listed here so that a reader knows they were looked at, not missed.
template <typename real_t>
struct NuclNuclDiffuseParams {
  /// fCofLambda, 1.0. `lambda = fCofLambda * k * R`.
  real_t cof_lambda = real_t(1);
  /// fNuclearRadiusCof, 1.0. The radius is `fNuclearRadiusCof * 1 fermi * A^(1/3)`.
  real_t nuclear_radius_cof = real_t(1);
};

/// The model's registered energy window: SetMinEnergy(50 MeV) in the constructor, then
/// SetMinEnergy(0.0) by G4IonElasticPhysics::ConstructProcess, and SetMaxEnergy from
/// G4HadronicParameters::GetMaxEnergy() (100 TeV). With one model registered the range is never
/// consulted (see `choose_hadronic_interaction`), but it is what the range manager would read.
template <typename real_t>
__host__ __device__ constexpr real_t nucl_nucl_diffuse_min_energy_as_registered() {
  return real_t(0);
}

/// G4NuclNuclDiffuseElastic::CalculateNuclearRad, in mm.
///
/// The body is `radius = r0 * A13(A)` with `r0 = fNuclearRadiusCof * 1 fermi`. Everything else in
/// that function - the A < 50 / A >= 50 split with 1.16, 1.1 and 1.7 fermi and `powA(A, 0.27)` -
/// is inside a `/* */` block. So the radius is just A^(1/3) femtometres, through G4Pow::A13 and
/// not std::cbrt.
template <typename real_t>
__host__ __device__ real_t nucl_nucl_nuclear_rad(real_t a, real_t nuclear_radius_cof) {
  const real_t fermi = real_t(1e-12) * units::mm<real_t>();  // CLHEP fermi = 1e-12 mm
  const real_t r0 = nuclear_radius_cof * fermi;
  return r0 * data::g4pow_a13<real_t>(a);
}

/// G4NuclNuclDiffuseElastic::CalculateZommerfeld - the Sommerfeld parameter,
/// `alpha * Z1 * Z2 / beta`, with Z1 the projectile's charge in units of eplus.
template <typename real_t>
__host__ __device__ real_t nucl_nucl_zommerfeld(real_t beta, real_t z1, real_t z2) {
  return units::fine_structure_const<real_t>() * z1 * z2 / beta;
}

/// G4NuclNuclDiffuseElastic::CalculateAm - the Wentzel screening parameter.
///
/// `ch = 1.13 + 3.76*n^2`, `zn = 1.77 * k * Bohr_radius / Z^(1/3)`, `Am = ch/zn^2`, with
/// `k = momentum/hbarc` and Z the TARGET atomic number through G4Pow::A13 - note A13, applied to
/// a G4double Z, not Z13 applied to an integer. The two differ: A13 is a Taylor expansion about
/// a tabulated quarter-integer or integer, Z13 is a tabulated exact cube root.
template <typename real_t>
__host__ __device__ real_t nucl_nucl_am(real_t momentum, real_t n, real_t z) {
  const real_t k = momentum / units::hbarc<real_t>();
  const real_t ch = real_t(1.13) + real_t(3.76) * n * n;
  const real_t zn =
      real_t(1.77) * k * (real_t(1) / data::g4pow_a13<real_t>(z)) * units::bohr_radius<real_t>();
  return ch / (zn * zn);
}

/// What `SampleCoulombMuCMS` computes on the way to the answer, exposed so a deterministic
/// oracle comparison can check each piece rather than only the sampled t.
template <typename real_t>
struct NuclNuclDiffuseState {
  real_t nuclear_radius = real_t(0);   ///< R1(A_projectile) + R2(A_target), mm
  real_t wave_vector = real_t(0);      ///< k = p_cms/hbarc, mm^-1
  real_t beta = real_t(0);
  real_t zommerfeld = real_t(0);
  real_t am = real_t(0);
  real_t profile_lambda = real_t(0);   ///< fCofLambda * k * R
  real_t half_rut_theta_tg = real_t(0);
  real_t half_rut_theta_tg2 = real_t(0);
  real_t coulomb_mu_c = real_t(0);     ///< tg2/(1+tg2), the grazing-angle cut on (1-cos)/2
  bool refused_neutral = false;
};

/// G4NuclNuclDiffuseElastic::InitDynParameters plus CalculateRutherfordAnglePar, for the values
/// SampleCoulombMuCMS reads.
///
/// `fNuclearRadius` must already hold R1 + R2 - SampleCoulombMuCMS sets it just before calling,
/// and InitDynParameters uses it without recomputing, so the two are a pair.
///
/// `CalculateCoulombPhaseZero()` is called here in Geant4 and sets `fCoulombPhase0` from the
/// imaginary part of `GammaLogB2n(1 + i*n)`. Nothing in the live path reads it - only the dead
/// AmplitudeNear/AmplitudeFar do - so the complex log-gamma is not ported. Named, not dropped.
template <typename real_t>
__host__ __device__ NuclNuclDiffuseState<real_t> nucl_nucl_init_dyn_parameters(
    real_t projectile_charge, real_t projectile_mass, real_t target_z, real_t part_mom,
    real_t nuclear_radius, const NuclNuclDiffuseParams<real_t>& par) {
  NuclNuclDiffuseState<real_t> st;
  st.nuclear_radius = nuclear_radius;
  st.wave_vector = part_mom / units::hbarc<real_t>();
  const real_t lambda = par.cof_lambda * st.wave_vector * nuclear_radius;
  if (projectile_charge != real_t(0)) {
    const real_t a = part_mom / projectile_mass;  // beta*gamma
    st.beta = a / sqrt(real_t(1) + a * a);
    st.zommerfeld = nucl_nucl_zommerfeld<real_t>(st.beta, projectile_charge, target_z);
    st.am = nucl_nucl_am<real_t>(part_mom, st.zommerfeld, target_z);
  } else {
    st.refused_neutral = true;
  }
  st.profile_lambda = lambda;
  st.half_rut_theta_tg = st.zommerfeld / st.profile_lambda;
  st.half_rut_theta_tg2 = st.half_rut_theta_tg * st.half_rut_theta_tg;
  st.coulomb_mu_c = st.half_rut_theta_tg2 / (real_t(1) + st.half_rut_theta_tg2);
  return st;
}

/// G4NuclNuclDiffuseElastic::SampleCoulombMuCMS.
///
/// Samples mu = (1 - cos(theta_cms))/2 from the screened Rutherford distribution truncated at
/// the grazing angle, whose inverse CDF is
///
///     mu = muC * r * Am / (Am + muC*(1 - r)),      r uniform on (0,1)
///
/// and returns `t = 4 * p_cms^2 * mu`, which is -t in MeV^2. Exactly one uniform is consumed.
///
/// `muC = tan^2(theta_R/2) / (1 + tan^2(theta_R/2))` with `tan(theta_R/2) = n/(kR)`, so the cut
/// is where the Coulomb trajectory grazes the sum of the two nuclear radii: below the grazing
/// angle the scattering is Coulomb, above it the nuclear part takes over and this model - as
/// 11.1.1 runs it - does not produce those angles at all.
template <typename real_t, typename Rng>
__host__ __device__ real_t nucl_nucl_sample_coulomb_mu_cms(const NuclNuclDiffuseState<real_t>& st,
                                                           real_t p_cms, Rng& rng) {
  const real_t rand = rng.uniform();
  real_t mu = st.coulomb_mu_c * rand * st.am;
  mu /= st.am + st.coulomb_mu_c * (real_t(1) - rand);
  return real_t(4) * p_cms * p_cms * mu;
}

/// G4NuclNuclDiffuseElastic::SampleInvariantT.
///
/// The CMS momentum is obtained by BOOSTING, not by the closed form G4HadronElastic uses:
/// `lv1 = (p, 0, 0, sqrt(m1^2+p^2))`, `lv = lv1 + (0,0,0,M(A,Z))`, `lv1.boost(-lv.boostVector())`
/// and then `|lv1.vect()|`. Note the projectile momentum is put on the **x** axis here, which is
/// harmless because only the magnitude is used, and note the two routes to p_cms are the same
/// number to rounding but not bitwise - `G4HadronElastic::ApplyYourself` computes its own
/// `momentumCMS` for pLocalTmax and this function computes another for the t it returns.
///
/// The radii are R1 = A_projectile^(1/3) fm and R2 = A_target^(1/3) fm, ADDED (not added in
/// quadrature - the quadrature line is commented out in InitParameters), and `fAtomicWeight` is
/// the target A as an integer cast to double, which is the isotope's A from SampleZandA and not
/// the NIST atomic mass. (`Initialise()`, in the dead path, uses GetAtomicMassAmu instead.)
///
/// Returns -t in MeV^2. `refused` is set when the projectile is neutral, which this model cannot
/// handle - see the header note.
template <typename real_t, typename NuclearMassFn, typename Rng>
__host__ __device__ real_t nucl_nucl_diffuse_sample_invariant_t(
    const HadProjectile<real_t>& projectile, int z, int a, const NuclearMassFn& nuclear_mass,
    const NuclNuclDiffuseParams<real_t>& par, Rng& rng, NuclNuclDiffuseState<real_t>* state_out,
    bool* refused) {
  if (refused) { *refused = false; }
  const real_t m1 = projectile.mass;
  const real_t p = projectile.momentum();
  const real_t tot_e_lab = sqrt(m1 * m1 + p * p);
  const real_t mass2 = nuclear_mass(z, a);

  // lv = lv1 + target at rest; boost lv1 by -beta to the CMS. With lv1 along one axis the boost
  // is one-dimensional: beta = p/(E1 + M), and |p_cms| = gamma*(p - beta*E1).
  //
  // `HepLorentzVector::boostVector()` is `pp * (1./ee)` (LorentzVector.cc:189) - a multiplication
  // by the reciprocal, not a division. It is an ulp, and it propagates through the sampled t into
  // the recoil direction, which is compared against Geant4's near machine precision.
  const real_t e_tot = tot_e_lab + mass2;
  const real_t b = p * (real_t(1) / e_tot);
  const real_t b2 = b * b;
  const real_t ggamma = real_t(1) / sqrt(real_t(1) - b2);
  const real_t bp = -b * p;  // boost(-bst): the boost vector is negated
  const real_t gamma2 = (b2 > real_t(0)) ? (ggamma - real_t(1)) / b2 : real_t(0);
  const real_t p_cms_signed = p + gamma2 * bp * (-b) + ggamma * (-b) * tot_e_lab;
  const real_t p_cms = (p_cms_signed > real_t(0)) ? p_cms_signed : -p_cms_signed;

  const real_t r1 = nucl_nucl_nuclear_rad<real_t>(real_t(projectile.baryon_number),
                                                  par.nuclear_radius_cof);
  const real_t r2 = nucl_nucl_nuclear_rad<real_t>(real_t(a), par.nuclear_radius_cof);
  const real_t radius = r1 + r2;

  NuclNuclDiffuseState<real_t> st = nucl_nucl_init_dyn_parameters<real_t>(
      projectile.charge, m1, real_t(z), p_cms, radius, par);
  if (state_out) { *state_out = st; }
  if (st.refused_neutral) {
    if (refused) { *refused = true; }
    return real_t(0);
  }
  return nucl_nucl_sample_coulomb_mu_cms<real_t>(st, p_cms, rng);
}

}  // namespace g4gpu::physics::hadronic::elastic
