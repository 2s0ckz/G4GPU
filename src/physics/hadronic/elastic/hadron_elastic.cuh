// G4HadronElastic - the base class of every hadron-nucleus elastic final state, and the model
// QBBC itself uses for d, t, He3 and alpha.
//
// Transcribed from Geant4 11.1.1,
//   processes/hadronic/models/coherent_elastic/src/G4HadronElastic.cc
//     G4HadronElastic::ApplyYourself
//     G4HadronElastic::SampleInvariantT   (the Gheisha two-exponential parameterisation)
//     G4HadronElastic::GetSlopeCof
//
// ApplyYourself is shared by all four elastic models: G4ChipsElasticModel,
// G4ElasticHadrNucleusHE and G4NuclNuclDiffuseElastic all override only SampleInvariantT and
// inherit the kinematics, the recoil and the threshold handling from here. So this file is where
// "what an elastic scatter does" lives, and the other three files are just t-samplers.
//
// The target is at rest. G4HadronElastic builds `lv(0, 0, plab, e1 + mass2)` with the target
// contributing nothing but its mass: no Fermi motion, no thermal motion, nothing from
// G4Nucleus beyond (Z, A). See the HadNucleus comment in ../process.cuh.
#pragma once

#include <cmath>

#include "core/units.cuh"
#include "core/vec3.cuh"
#include "data/g4pow.hh"
#include "physics/hadronic/process.cuh"

namespace g4gpu::physics::hadronic::elastic {

/// G4Pow::Z23 - Z13 squared, with Z13 the tabulated cube root of an integer. `data/g4pow.hh`
/// has A13/A23/Z13; Z23 and powZ are added here rather than there because `data/g4pow.hh` is
/// shared with the packages running in parallel and this package does not own it.
template <typename real_t>
__host__ __device__ inline real_t g4pow_z23(int z) {
  const real_t x = data::g4pow_z13<real_t>(z);
  return x * x;
}

/// G4Pow::powZ(Z, y) = expA(y * lz[Z]), with lz[Z] = log(Z). NOT std::pow: `expA` is a
/// third-order expansion about a tabulated point, and the difference from the exact function is
/// about 1e-7 relative - a thousand times the tolerance an oracle comparison is checked at.
/// docs/HADRONIC_PLAN.md section 8 and the header of data/g4pow.hh are the history.
template <typename real_t>
__host__ __device__ inline real_t g4pow_pow_z(int z, real_t y) {
  if (z <= 0) { return real_t(0); }
  return data::g4pow_exp_a<real_t>(y * log(static_cast<real_t>(z)));
}

/// CLHEP Hep3Vector::unit(), which is NOT a division by the magnitude.
///
///     double tot = mag2();
///     Hep3Vector p(x(),y(),z());
///     return tot > 0.0 ? p *= (1.0/std::sqrt(tot)) : p;
///
/// It multiplies by the RECIPROCAL of the magnitude, and `core/vec3.cuh`'s `normalize` divides by
/// it. Written here rather than by changing `normalize`, which is shared with the packages running
/// in parallel and whose callers were written against division.
///
/// Honestly: on the 1408 comparison points of tests/test_elastic_models.cu this makes NO
/// difference - putting the division back leaves every column bitwise identical, so no assertion
/// in this package can currently fail because of it. It is here because it is what CLHEP does,
/// not because it was measured to matter. The ulp that WAS measurable, and that this file used to
/// get wrong, is `boostVector`'s: see the boost below, where `pp/ee` instead of `pp*(1./ee)` puts
/// the recoil direction 5.8e-13 and the alpha-on-Pb recoil energy 2.3e-10 out.
///
/// The zero-length case also differs: CLHEP returns the zero vector, `normalize` returns +z.
/// Neither can be reached from an elastic final state (the primary's momentum is non-zero
/// whenever `eFinal > 0`, and a recoil is only emitted when `erec > 0`), and CLHEP's is what is
/// reproduced.
template <typename real_t>
__host__ __device__ inline Vec3<real_t> clhep_unit(const Vec3<real_t>& v) {
  const real_t tot = v.x * v.x + v.y * v.y + v.z * v.z;
  if (!(tot > real_t(0))) { return Vec3<real_t>{real_t(0), real_t(0), real_t(0)}; }
  const real_t inv = real_t(1) / sqrt(tot);
  return Vec3<real_t>{v.x * inv, v.y * inv, v.z * inv};
}

/// The lowest energy at which G4HadronElastic scatters at all: 1e-6 eV. Below it the primary is
/// returned unchanged along +z, which after `rotateUz` is its original direction.
template <typename real_t>
__host__ __device__ constexpr real_t hadron_elastic_lowest_energy() {
  return real_t(1e-6) * units::eV<real_t>();
}

/// G4HadronElastic::SampleInvariantT - the Gheisha two-exponential momentum-transfer
/// parameterisation, in GeV^2 converted back to MeV^2.
///
/// Structure: two exponentials in -t with slopes `bb` and `dd` and weights `aa` and `cc`, each
/// truncated at `tmax`; one uniform chooses which exponential, a second samples from it. The
/// coefficients are piecewise in A (<= 62 or > 62), in whether the projectile is a pion, and for
/// a pion in whether plab is above 400 MeV/c.
///
/// Three transcription traps, all of them in the source as written:
///   - `tmax` is `pLocalTmax/GeV^2`, and pLocalTmax is set by ApplyYourself to `4*pcms^2` before
///     SampleInvariantT is called. It is a member, not an argument, so a sampler called without
///     ApplyYourself having run first would use a stale tmax. Here it is an explicit argument.
///   - `aa = (A*A)/bb` in the high-energy pion and the "other particles" branches, where the
///     commented-out line above it says `powZ(A, 1.93)/bb`. A*A is what runs.
///   - the low-energy pion branch really is `29.*z07in13*z07in13*Z23(A)` with
///     z07in13 = pow(0.7, 1/3) computed with std::pow, not G4Pow - so 29*0.7^(2/3), and the
///     0.7^(1/3) factors are std::pow's value, not expA's.
///
/// `numLimit = 18` caps the exponent, so `q1`/`q2` saturate at 1 rather than underflowing.
///
/// A is the target mass number; Z is unused by this sampler (Geant4's signature names it and
/// then ignores it).
template <typename real_t, typename Rng>
__host__ __device__ real_t hadron_elastic_sample_invariant_t(int pdg, real_t mom, int /*z*/,
                                                             int a, real_t p_local_tmax,
                                                             Rng& rng) {
  const real_t plab_low_limit = real_t(400) * units::MeV<real_t>();
  const real_t gev2 = units::GeV<real_t>() * units::GeV<real_t>();
  const real_t z07in13 = pow(real_t(0.7), real_t(0.3333333333));
  const real_t num_limit = real_t(18);

  const int apdg = (pdg < 0) ? -pdg : pdg;
  const real_t tmax = p_local_tmax / gev2;

  real_t aa, bb, cc, dd;
  if (a <= 62) {
    if (apdg == 211) {  // pions
      if (mom >= plab_low_limit) {
        bb = real_t(14.5) * g4pow_z23<real_t>(a);
        dd = real_t(10);
        cc = real_t(0.075) * data::g4pow_z13<real_t>(a) / dd;
        aa = real_t(a) * real_t(a) / bb;
      } else {
        bb = real_t(29) * z07in13 * z07in13 * g4pow_z23<real_t>(a);
        dd = real_t(15);
        cc = real_t(0.04) * data::g4pow_z13<real_t>(a) / dd;
        aa = g4pow_pow_z<real_t>(a, real_t(1.63)) / bb;
      }
    } else {
      bb = real_t(14.5) * g4pow_z23<real_t>(a);
      dd = real_t(20);
      aa = real_t(a) * real_t(a) / bb;
      cc = real_t(1.4) * data::g4pow_z13<real_t>(a) / dd;
    }
  } else {
    if (apdg == 211) {
      if (mom >= plab_low_limit) {
        bb = real_t(60) * z07in13 * data::g4pow_z13<real_t>(a);
        dd = real_t(30);
        aa = real_t(0.5) * real_t(a) * real_t(a) / bb;
        cc = real_t(4) * g4pow_pow_z<real_t>(a, real_t(0.4)) / dd;
      } else {
        bb = real_t(120) * z07in13 * data::g4pow_z13<real_t>(a);
        dd = real_t(30);
        aa = real_t(2) * g4pow_pow_z<real_t>(a, real_t(1.33)) / bb;
        cc = real_t(4) * g4pow_pow_z<real_t>(a, real_t(0.4)) / dd;
      }
    } else {
      bb = real_t(60) * data::g4pow_z13<real_t>(a);
      dd = real_t(25);
      aa = g4pow_pow_z<real_t>(a, real_t(1.33)) / bb;
      cc = real_t(0.2) * g4pow_pow_z<real_t>(a, real_t(0.4)) / dd;
    }
  }
  const real_t bt = bb * tmax, dt = dd * tmax;
  real_t q1 = real_t(1) - exp(-((bt < num_limit) ? bt : num_limit));
  const real_t q2 = real_t(1) - exp(-((dt < num_limit) ? dt : num_limit));
  const real_t s1 = q1 * aa;
  const real_t s2 = q2 * cc;
  if ((s1 + s2) * rng.uniform() < s2) {
    q1 = q2;
    bb = dd;
  }
  return -gev2 * log(real_t(1) - rng.uniform() * q1) / bb;
}

/// G4HadronElastic::GetSlopeCof - the ds/dt slope coefficients for strange, charmed and bottom
/// hadrons.
///
/// Called by G4AntiNuclElastic and by nothing else in the elastic chain; kept because it is a
/// pure table and the alternative to transcribing it is to have it missing when P11's FTFP
/// secondaries start reaching an elastic process. The default is 1.0, which is what a nucleon or
/// a pion gets.
///
/// Geant4's own two bugs are reproduced, not fixed, because reproducing them is the job: the
/// Omega is matched on `pdg == 3324`, which is Xi*0, not 3334 (Omega-); and the kaon test runs
/// AFTER the baryon chain in a second `if`, so a baryon code that also matched a meson code
/// would be overwritten - none do.
template <typename real_t>
__host__ __device__ real_t hadron_elastic_slope_cof(int pdg) {
  real_t coeff = real_t(1);

  constexpr real_t lBarCof1S = real_t(0.88);
  constexpr real_t lBarCof2S = real_t(0.76);
  constexpr real_t lBarCof3S = real_t(0.64);
  constexpr real_t lBarCof1C = real_t(0.784378);
  constexpr real_t lBarCofSC = real_t(0.664378);
  constexpr real_t lBarCof2SC = real_t(0.544378);
  constexpr real_t lBarCof1B = real_t(0.740659);
  constexpr real_t lBarCofSB = real_t(0.620659);
  constexpr real_t lBarCof2SB = real_t(0.500659);

  if (pdg == 3122 || pdg == 3222 || pdg == 3112 || pdg == 3212) {
    coeff = lBarCof1S;
  } else if (pdg == 3322 || pdg == 3312) {
    coeff = lBarCof2S;
  } else if (pdg == 3324) {
    coeff = lBarCof3S;
  } else if (pdg == 4122 || pdg == 4212 || pdg == 4222 || pdg == 4112) {
    coeff = lBarCof1C;
  } else if (pdg == 4332) {
    coeff = lBarCof2SC;
  } else if (pdg == 4232 || pdg == 4132) {
    coeff = lBarCofSC;
  } else if (pdg == 5122 || pdg == 5222 || pdg == 5112 || pdg == 5212) {
    coeff = lBarCof1B;
  } else if (pdg == 5332) {
    coeff = lBarCof2SB;
  } else if (pdg == 5132 || pdg == 5232) {
    coeff = lBarCofSB;
  }

  constexpr real_t lMesCof1S = real_t(0.82);
  constexpr real_t llMesCof1C = real_t(0.676568);
  constexpr real_t llMesCof1B = real_t(0.610989);
  constexpr real_t llMesCof2C = real_t(0.353135);
  constexpr real_t llMesCof2B = real_t(0.221978);
  constexpr real_t llMesCofSC = real_t(0.496568);
  constexpr real_t llMesCofSB = real_t(0.430989);
  constexpr real_t llMesCofCB = real_t(0.287557);
  constexpr real_t llMesCofEtaP = real_t(0.88);
  constexpr real_t llMesCofEta = real_t(0.76);

  if (pdg == 321 || pdg == 311 || pdg == 310) {
    coeff = lMesCof1S;
  } else if (pdg == 511 || pdg == 521) {
    coeff = llMesCof1B;
  } else if (pdg == 421 || pdg == 411) {
    coeff = llMesCof1C;
  } else if (pdg == 531) {
    coeff = llMesCofSB;
  } else if (pdg == 541) {
    coeff = llMesCofCB;
  } else if (pdg == 431) {
    coeff = llMesCofSC;
  } else if (pdg == 441 || pdg == 443) {
    coeff = llMesCof2C;
  } else if (pdg == 553) {
    coeff = llMesCof2B;
  } else if (pdg == 221) {
    coeff = llMesCofEta;
  } else if (pdg == 331) {
    coeff = llMesCofEtaP;
  }
  return coeff;
}

/// What ApplyYourself needs to know about the target and the recoil, from tables this package
/// does not own.
///
/// `nuclear_mass(Z, A)` is G4NucleiProperties::GetNuclearMass(A, Z) - package P3's. It appears
/// as a functor for the same reason the cross sections do: the elastic kinematics are exact
/// given the target mass, and inventing a mass formula here would make them approximate.
///
/// `recoil_pdg(Z, A)` is the species of the recoil. Geant4 names proton, deuteron, triton, He3
/// and alpha explicitly and takes everything else from G4IonTable::GetIon(Z,A,0); the PDG code
/// is enough for this package (species enums are P1's), so the recoil is reported as
/// (pdg, Z, A, mass) and P1 maps it.
struct TargetTablesContract {};

/// The result of one elastic ApplyYourself, before the process turns it into a step result.
template <typename real_t>
struct ElasticSample {
  real_t t = real_t(0);              ///< -t, MeV^2, as SampleInvariantT returned it
  real_t cos_theta_cms = real_t(1);
  real_t momentum_cms = real_t(0);
  real_t p_local_tmax = real_t(0);   ///< 4*pcms^2
  bool resampled = false;            ///< t was out of [0, tmax] and the generic sampler was used
};

/// G4HadronElastic::ApplyYourself.
///
/// The sampler is a template parameter, so this one function serves all four models exactly as
/// the virtual SampleInvariantT does in Geant4. `sampler(pdg, plab, Z, A, pLocalTmax, rng)`
/// returns -t in MeV^2.
///
/// Faithful points that are easy to lose:
///   - `momentumCMS = plab*mass2/sqrt(m1^2 + mass2^2 + 2*mass2*e1)`, written that way rather
///     than by boosting, and `pLocalTmax = 4*pcms^2`.
///   - if the sampler returns t outside [0, pLocalTmax], Geant4 warns twice and then RESAMPLES
///     **with G4HadronElastic::SampleInvariantT specifically** - not with the model's own
///     sampler. So a Chips t out of range falls back to Gheisha, and it costs the random numbers
///     of both attempts.
///   - phi is drawn AFTER the resample check, so the number of uniforms consumed depends on
///     whether the resample happened.
///   - cost is clamped to [-1, 1] and sint = sqrt((1-cost)(1+cost)) - written as the product,
///     not as sqrt(1-cost^2), which matters at cost near +-1.
///   - the recoil four-momentum is `lv - nlv1` where lv is the TOTAL initial four-momentum, so
///     the recoil carries the momentum imbalance exactly; erec = max(lv.e() - mass2, 0).
///   - the recoil is emitted only if `erec > GetRecoilEnergyThreshold()`, otherwise its energy
///     becomes a LOCAL DEPOSIT. The threshold is set per step by the process (see
///     elastic_process.cuh) and defaults to 0 on the model.
///   - if the primary's final energy is <= 0 the direction is set to +z and the energy to 0,
///     which the process then reads as "the primary stopped".
template <typename real_t, int kCap, typename Sampler, typename NuclearMassFn,
          typename RecoilSpeciesFn, typename Rng>
__host__ __device__ ElasticSample<real_t> hadron_elastic_apply_yourself(
    const HadProjectile<real_t>& projectile, const HadNucleus& target, const Sampler& sampler,
    const NuclearMassFn& nuclear_mass, const RecoilSpeciesFn& recoil_species,
    real_t recoil_energy_threshold, int secondary_model_id, Rng& rng,
    HadFinalState<real_t, kCap>* out) {
  ElasticSample<real_t> info;
  out->clear();

  const real_t ekin = projectile.kin_energy;
  if (ekin <= hadron_elastic_lowest_energy<real_t>()) {
    out->energy_change = ekin;
    out->momentum_change = Vec3<real_t>{real_t(0), real_t(0), real_t(1)};
    info.cos_theta_cms = real_t(1);
    return info;
  }

  const int a = target.a, z = target.z;
  const real_t m1 = projectile.mass;
  const real_t plab = sqrt(ekin * (ekin + real_t(2) * m1));

  const real_t mass2 = nuclear_mass(z, a);
  const real_t e1 = m1 + ekin;
  // lv = the total initial four-momentum, target at rest.
  const real_t lv_pz = plab, lv_e = e1 + mass2;
  const real_t momentum_cms =
      plab * mass2 / sqrt(m1 * m1 + mass2 * mass2 + real_t(2) * mass2 * e1);
  const real_t p_local_tmax = real_t(4) * momentum_cms * momentum_cms;
  info.momentum_cms = momentum_cms;
  info.p_local_tmax = p_local_tmax;

  real_t t = sampler(projectile.pdg, plab, z, a, p_local_tmax, rng);
  if (t < real_t(0) || t > p_local_tmax) {
    info.resampled = true;
    t = hadron_elastic_sample_invariant_t<real_t>(projectile.pdg, plab, z, a, p_local_tmax, rng);
  }
  info.t = t;

  const real_t phi = rng.uniform() * units::twopi<real_t>();
  real_t cost = real_t(1) - real_t(2) * t / p_local_tmax;
  if (cost > real_t(1)) { cost = real_t(1); }
  else if (cost < real_t(-1)) { cost = real_t(-1); }
  const real_t sint = sqrt((real_t(1) - cost) * (real_t(1) + cost));
  info.cos_theta_cms = cost;

  // nlv1 in the CMS, then boosted to the lab along +z with beta = lv_pz/lv_e.
  const real_t n_px = momentum_cms * sint * cos(phi);
  const real_t n_py = momentum_cms * sint * sin(phi);
  const real_t n_pz_cms = momentum_cms * cost;
  const real_t n_e_cms = sqrt(momentum_cms * momentum_cms + m1 * m1);

  // CLHEP HepLorentzVector::boost, written in CLHEP's own form rather than the algebraically
  // equal `gamma*(pz + b*E)`: it computes `gamma2 = (gamma-1)/b2` and
  // `z' = z + gamma2*bp*bz + gamma*bz*t`. The two agree to the last bit only sometimes, and the
  // deterministic recoil comparison in tests/test_elastic_models.cu is checked near machine
  // precision, so the arithmetic is copied rather than simplified. The boost vector is
  // `HepLorentzVector::boostVector()`, which is along +z because the target is at rest and which
  // is `pp * (1./ee)` - a multiplication by the reciprocal, not `pp/ee`. That is an ulp, and this
  // ulp propagates into the recoil direction, so it is copied too (LorentzVector.cc:189).
  const real_t bz = lv_pz * (real_t(1) / lv_e);
  const real_t b2 = bz * bz;
  const real_t ggamma = real_t(1) / sqrt(real_t(1) - b2);
  const real_t bp = bz * n_pz_cms;
  const real_t gamma2 = (b2 > real_t(0)) ? (ggamma - real_t(1)) / b2 : real_t(0);
  const real_t n_pz = n_pz_cms + gamma2 * bp * bz + ggamma * bz * n_e_cms;
  const real_t n_e = ggamma * (n_e_cms + bp);

  const real_t e_final = n_e - m1;
  if (e_final <= real_t(0)) {
    out->momentum_change = Vec3<real_t>{real_t(0), real_t(0), real_t(1)};
    out->energy_change = real_t(0);
  } else {
    out->momentum_change = clhep_unit(Vec3<real_t>{n_px, n_py, n_pz});
    out->energy_change = e_final;
  }

  // lv -= nlv1
  const real_t r_px = -n_px, r_py = -n_py, r_pz = lv_pz - n_pz, r_e = lv_e - n_e;
  const real_t erec = (r_e - mass2 > real_t(0)) ? (r_e - mass2) : real_t(0);

  if (erec > recoil_energy_threshold) {
    HadSecondary<real_t> sec;
    recoil_species(z, a, &sec.pdg, &sec.mass);
    sec.z = z;
    sec.a = a;
    sec.kin_energy = erec;
    sec.direction = clhep_unit(Vec3<real_t>{r_px, r_py, r_pz});
    sec.creator_model_id = secondary_model_id;
    out->add_secondary(sec);
  } else {
    out->local_energy_deposit = erec;
  }
  return info;
}

}  // namespace g4gpu::physics::hadronic::elastic
