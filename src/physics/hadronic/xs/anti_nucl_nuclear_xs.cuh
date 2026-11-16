// G4ComponentAntiNuclNuclearXS ("AntiAGlauber") - anti-nucleon and light anti-nucleus on a
// nucleus in the Glauber approximation of Galoyan, Uzhinsky et al., Phys. Lett. B705 (2011) 235.
//
// Transcribed from source/processes/hadronic/cross_sections/src/G4ComponentAntiNuclNuclearXS.cc
// and its header (11.1.1): GetTotalElementCrossSection, GetInelasticElementCrossSection,
// GetElasticElementCrossSection, the three isotope forwards, GetAntiHadronNucleonTotCrSc and
// GetAntiHadronNucleonElCrSc, with the ReffTot / ReffInel tables of the header.
//
// In QBBC this one component is the cross section of BOTH the hadElastic process
// (`G4HadronElasticPhysics::ConstructProcess`, `ElasticXS("AntiAGlauber")`) and the inelastic
// process (`G4HadronicBuilder::BuildAntiLightIonsFTFP`, `InelasticXS("AntiAGlauber")`) of the
// six species `G4HadParticles::GetLightAntiIons()` lists: anti_proton, anti_neutron,
// anti_deuteron, anti_triton, anti_He3, anti_alpha. P2 refused it by name
// (`XsRefusal::kComponentAntiNuclNuclearXS`); P23c ports it and that refusal now means only what
// `ggnn_compute_cross_sections` uses it for - an anti-nucleus handed to the WRONG component.
//
// WHAT A CALLER MUST HAND IT, AND WHAT IT DOES WITH IT
//
//   * `A` is a REAL: `G4CrossSectionInelastic/Elastic::GetElementCrossSection` pass
//     `nist->GetAtomicMassAmu(Z)`, the NIST mean atomic mass, so hydrogen is 1.00794 and carbon
//     12.0107. The component truncates it to `intA = static_cast<G4int>(A)` for the radius and
//     the light-target table, but the Glauber product `ApAt = |B| * A` takes the UNTRUNCATED
//     value - two different A's in one formula, both transcribed.
//   * `kinEnergy` is the projectile's TOTAL kinetic energy; the momentum is divided by |B| inside
//     `GetAntiHadronNucleonTotCrSc`, so the NN cross section is at the per-nucleon MOMENTUM (not
//     at the per-nucleon kinetic energy, which is what the nucleus-nucleus GG component uses).
//   * The NN cross sections are in MILLIBARN as plain numbers (the `xsection *= millibarn` lines
//     are commented out in the source), and only the nuclear results carry `millibarn`. The
//     inelastic one on hydrogen is `(sigmaTotal - sigmaElastic) * millibarn` and the total one
//     `sigmaTotal * millibarn` - the pbar-p row of the table is never read (its 0.000 entry).
//   * The radii are in FERMI as plain numbers and `10.*pi*R^2` is the fm^2 -> mb conversion.
//
// TWO THINGS IN THE SOURCE THAT ARE NOT WHAT A READER EXPECTS, transcribed as they stand:
//
//   1. The table rows are not symmetric where a reader would expect symmetry: ReffTot's anti-t
//      row has 3.589 against anti-alpha, its anti-He3 row 2.589, and the anti-alpha row reads
//      3.589 against t and 3.598 against He3. Four different numbers for two pairs of mirror
//      collisions; the table is data and is copied digit for digit.
//   2. `R0` is a MEMBER that `GetAntiHadronNucleonTotCrSc` overwrites (`R0 = sqrt(0.40874044*
//      SigAss - B)`) and `GetAntiHadronNucleonElCrSc` reads after calling it - so the elastic
//      NN cross section's R0 is the one its own call to the total computed. Here R0 is a local
//      returned alongside, which is the same value by construction (the elastic function calls
//      the total one first, with the same arguments).
//
// THE PROJECTILE INDEX. `i` is chosen by POINTER identity against the six singletons; this port
// keys it on the PDG code. An anti-hypernucleus (`i < 0`, `IsAntiHypernucleus()`) is treated as
// an anti-alpha by the source; no QBBC list in this port reaches it, and a projectile that is not
// one of the six is refused by name (`kComponentAntiNuclNuclearXS`) rather than given the
// anti-alpha's radius, because the source prints a JustWarning for it and carries on with
// whatever `fRadiusEff` the previous call left - a stale member, not a cross section.
#pragma once
#include <cmath>

#include "core/units.cuh"
#include "data/g4pow.hh"
#include "physics/hadronic/xs/g4pow_extra.cuh"
#include "physics/hadronic/xs/projectile.cuh"
#include "physics/hadronic/xs/refusal.cuh"

namespace g4gpu::hadronic::xs {

/// CLHEP's `millibarn = 1.e-3*barn`, built the way CLHEP builds it (units.cuh's note on barn).
template <typename real_t>
__host__ __device__ inline real_t anu_millibarn() {
  return real_t(1.e-3) * units::barn<real_t>();
}

/// The projectile row `i` of the source: 0 for pbar and nbar, 1 anti-d, 2 anti-t, 3 anti-He3,
/// 4 anti-alpha; -1 for anything else (refused by the callers below).
__host__ __device__ inline int anu_projectile_index(int pdg) {
  switch (pdg) {
    case -2212:
    case -2112: return 0;
    case -1000010020: return 1;
    case -1000010030: return 2;
    case -1000020030: return 3;
    case -1000020040: return 4;
    default: return -1;
  }
}

/// The target column `j`: the five light nuclei by (Z, intA), -1 otherwise.
__host__ __device__ inline int anu_target_index(int Z, int intA) {
  if (Z == 1 && intA == 1) { return 0; }
  if (Z == 1 && intA == 2) { return 1; }
  if (Z == 1 && intA == 3) { return 2; }
  if (Z == 2 && intA == 3) { return 3; }
  if (Z == 2 && intA == 4) { return 4; }
  return -1;
}

/// G4ComponentAntiNuclNuclearXS.hh's ReffTot[5][5] and ReffInel[5][5], fm.
template <typename real_t>
__host__ __device__ inline real_t anu_reff_tot(int i, int j) {
  constexpr double t[5][5] = {{0.000, 3.800, 3.300, 3.300, 2.376},
                              {3.800, 3.238, 3.144, 3.144, 2.544},
                              {3.300, 3.144, 3.075, 3.075, 3.589},
                              {3.300, 3.144, 3.075, 3.075, 2.589},
                              {2.376, 2.544, 3.589, 3.598, 2.241}};
  return real_t(t[i][j]);
}
template <typename real_t>
__host__ __device__ inline real_t anu_reff_inel(int i, int j) {
  constexpr double t[5][5] = {{0.000, 3.582, 3.105, 3.105, 2.209},
                              {3.582, 3.169, 3.066, 3.066, 2.498},
                              {3.105, 3.066, 2.973, 2.973, 2.508},
                              {3.105, 3.066, 2.973, 2.973, 2.508},
                              {2.209, 2.498, 2.508, 2.508, 2.158}};
  return real_t(t[i][j]);
}

/// What `GetAntiHadronNucleonTotCrSc` leaves in its members: the total pbar-p cross section in
/// mb (a plain number), and the kinematic members the elastic one reads afterwards.
template <typename real_t>
struct AnuNucleon {
  real_t total_mb = 0;    ///< fAntiHadronNucleonTotXsc
  real_t S = 0;           ///< GeV^2
  real_t sqrt_s = 0;      ///< GeV
  real_t R0 = 0;          ///< GeV^-1, the member the elastic function reads
};

/// G4ComponentAntiNuclNuclearXS::GetAntiHadronNucleonTotCrSc. `kin_energy` is the projectile's
/// whole kinetic energy, MeV; its momentum is divided by |B| before anything else.
template <typename real_t>
__host__ __device__ inline AnuNucleon<real_t> anu_nucleon_total(const Projectile<real_t>& p,
                                                              real_t kin_energy) {
  constexpr real_t Mn = real_t(0.93827231);   // GeV
  constexpr real_t b0 = real_t(11.92);        // GeV^(-2)
  constexpr real_t b2 = real_t(0.3036);       // GeV^(-2)
  constexpr real_t SqrtS0 = real_t(20.74);    // GeV
  constexpr real_t S0 = real_t(33.0625);      // GeV^2
  const int ab = (p.baryon_number < 0) ? -p.baryon_number : p.baryon_number;
  const real_t Pmass = p.mass;
  const real_t Energy = Pmass + kin_energy;
  const real_t momentum = sqrt(Energy * Energy - Pmass * Pmass) / static_cast<real_t>(ab);
  const real_t Plab = momentum / units::GeV<real_t>();

  AnuNucleon<real_t> o;
  const real_t Elab = sqrt(Mn * Mn + Plab * Plab);
  o.S = real_t(2.) * Mn * Mn + real_t(2.) * Mn * Elab;
  o.sqrt_s = sqrt(o.S);
  const real_t B = b0 + b2 * log(o.sqrt_s / SqrtS0) * log(o.sqrt_s / SqrtS0);
  const real_t SigAss = real_t(36.04) + real_t(0.304) * log(o.S / S0) * log(o.S / S0);
  o.R0 = sqrt(real_t(0.40874044) * SigAss - B);
  constexpr real_t C = real_t(13.55);
  constexpr real_t d1 = real_t(-4.47);
  constexpr real_t d2 = real_t(12.38);
  constexpr real_t d3 = real_t(-12.43);
  o.total_mb = SigAss * (real_t(1) + real_t(1.) / (sqrt(o.S - real_t(4.) * Mn * Mn))
                                         / (g4pow_pow_n<real_t>(o.R0, 3)) * C
                                         * (real_t(1) + d1 / o.sqrt_s
                                            + d2 / (g4pow_pow_n<real_t>(o.sqrt_s, 2))
                                            + d3 / (g4pow_pow_n<real_t>(o.sqrt_s, 3))));
  return o;
}

/// G4ComponentAntiNuclNuclearXS::GetAntiHadronNucleonElCrSc, mb as a plain number.
template <typename real_t>
__host__ __device__ inline real_t anu_nucleon_elastic(const Projectile<real_t>& p,
                                                      real_t kin_energy) {
  constexpr real_t Mn = real_t(0.93827231);
  constexpr real_t S0 = real_t(33.0625);
  const AnuNucleon<real_t> n = anu_nucleon_total<real_t>(p, kin_energy);
  const real_t SigAss = real_t(4.5) + real_t(0.101) * log(n.S / S0) * log(n.S / S0);
  constexpr real_t C = real_t(59.27);
  constexpr real_t d1 = real_t(-6.95);
  constexpr real_t d2 = real_t(23.54);
  constexpr real_t d3 = real_t(-25.34);
  return SigAss * (real_t(1) + real_t(1.) / (sqrt(n.S - real_t(4.) * Mn * Mn))
                                   / (g4pow_pow_n<real_t>(n.R0, 3)) * C
                                   * (real_t(1) + d1 / n.sqrt_s
                                      + d2 / (g4pow_pow_n<real_t>(n.sqrt_s, 2))
                                      + d3 / (g4pow_pow_n<real_t>(n.sqrt_s, 3))));
}

/// The heavy-target radius branch (`j < 0`), fm. `tot` selects GetTotal's coefficients,
/// otherwise GetInelastic's.
template <typename real_t>
__host__ __device__ inline real_t anu_heavy_radius(int i, int intA, bool tot) {
  const real_t z13 = data::g4pow_z13<real_t>(intA);
  if (tot) {
    switch (i) {
      case 0: return real_t(1.34) * g4pow_pow_z<real_t>(intA, real_t(0.23)) + real_t(1.35) / z13;
      case 1: return real_t(1.46) * g4pow_pow_z<real_t>(intA, real_t(0.21)) + real_t(1.45) / z13;
      case 2: return real_t(1.40) * g4pow_pow_z<real_t>(intA, real_t(0.21)) + real_t(1.63) / z13;
      case 3: return real_t(1.40) * g4pow_pow_z<real_t>(intA, real_t(0.21)) + real_t(1.63) / z13;
      default: return real_t(1.35) * g4pow_pow_z<real_t>(intA, real_t(0.21)) + real_t(1.10) / z13;
    }
  }
  switch (i) {
    case 0: return real_t(1.31) * g4pow_pow_z<real_t>(intA, real_t(0.22)) + real_t(0.90) / z13;
    case 1: return real_t(1.38) * g4pow_pow_z<real_t>(intA, real_t(0.21)) + real_t(1.55) / z13;
    case 2: return real_t(1.34) * g4pow_pow_z<real_t>(intA, real_t(0.21)) + real_t(1.51) / z13;
    case 3: return real_t(1.34) * g4pow_pow_z<real_t>(intA, real_t(0.21)) + real_t(1.51) / z13;
    default: return real_t(1.30) * g4pow_pow_z<real_t>(intA, real_t(0.21)) + real_t(1.05) / z13;
  }
}

/// G4ComponentAntiNuclNuclearXS::GetTotalElementCrossSection, mm^2.
template <typename real_t>
__host__ __device__ inline XsValue<real_t> anu_total_element(const Projectile<real_t>& p,
                                                            real_t kin_energy, int Z, real_t A) {
  const int i = anu_projectile_index(p.pdg);
  if (i < 0) { return {real_t(0), XsRefusal::kComponentAntiNuclNuclearXS}; }
  const real_t sigmaTotal = anu_nucleon_total<real_t>(p, kin_energy).total_mb;
  const int intA = static_cast<int>(A);
  const int j = anu_target_index(Z, intA);
  if (i == 0 && j == 0) { return {sigmaTotal * anu_millibarn<real_t>(), XsRefusal::kNone}; }
  const real_t r = (j >= 0) ? anu_reff_tot<real_t>(i, j) : anu_heavy_radius<real_t>(i, intA, true);
  const real_t R2 = r * r;
  const int ab = (p.baryon_number < 0) ? -p.baryon_number : p.baryon_number;
  const real_t ApAt = static_cast<real_t>(ab) * A;
  const real_t pi = units::pi<real_t>();
  return {anu_millibarn<real_t>() * real_t(2.) * pi * R2 * real_t(10.)
              * log(real_t(1.) + (ApAt * sigmaTotal / (real_t(2.) * pi * R2 * real_t(10.)))),
          XsRefusal::kNone};
}

/// G4ComponentAntiNuclNuclearXS::GetInelasticElementCrossSection, mm^2.
template <typename real_t>
__host__ __device__ inline XsValue<real_t> anu_inelastic_element(const Projectile<real_t>& p,
                                                                real_t kin_energy, int Z,
                                                                real_t A) {
  const int i = anu_projectile_index(p.pdg);
  if (i < 0) { return {real_t(0), XsRefusal::kComponentAntiNuclNuclearXS}; }
  const real_t sigmaTotal = anu_nucleon_total<real_t>(p, kin_energy).total_mb;
  const real_t sigmaElastic = anu_nucleon_elastic<real_t>(p, kin_energy);
  const int intA = static_cast<int>(A);
  const int j = anu_target_index(Z, intA);
  if (i == 0 && j == 0) {
    return {(sigmaTotal - sigmaElastic) * anu_millibarn<real_t>(), XsRefusal::kNone};
  }
  const real_t r =
      (j >= 0) ? anu_reff_inel<real_t>(i, j) : anu_heavy_radius<real_t>(i, intA, false);
  const real_t R2 = r * r;
  const int ab = (p.baryon_number < 0) ? -p.baryon_number : p.baryon_number;
  const real_t ApAt = static_cast<real_t>(ab) * A;
  const real_t pi = units::pi<real_t>();
  return {anu_millibarn<real_t>() * pi * R2 * real_t(10.)
              * log(real_t(1.) + (ApAt * sigmaTotal / (pi * R2 * real_t(10.)))),
          XsRefusal::kNone};
}

/// G4ComponentAntiNuclNuclearXS::GetElasticElementCrossSection: total minus inelastic, clamped
/// at zero, mm^2.
template <typename real_t>
__host__ __device__ inline XsValue<real_t> anu_elastic_element(const Projectile<real_t>& p,
                                                              real_t kin_energy, int Z, real_t A) {
  const XsValue<real_t> t = anu_total_element<real_t>(p, kin_energy, Z, A);
  if (!t.ok()) { return t; }
  const XsValue<real_t> in = anu_inelastic_element<real_t>(p, kin_energy, Z, A);
  real_t el = t.value - in.value;
  if (el < real_t(0.)) { el = real_t(0.); }
  return {el, XsRefusal::kNone};
}

}  // namespace g4gpu::hadronic::xs
