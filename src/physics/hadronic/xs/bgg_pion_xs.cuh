// G4BGGPionElasticXS and G4BGGPionInelasticXS - QBBC's pi+ and pi- elastic and inelastic
// cross sections, at every energy.
//
// Transcribed from source/processes/hadronic/cross_sections/src/G4BGGPionElasticXS.cc and
// G4BGGPionInelasticXS.cc (11.1.1): GetElementCrossSection, GetIsoCrossSection,
// CoulombFactorPiPlus, FactorPiMinus, and BuildPhysicsTable's four per-Z tables.
//
// Same three-piece shape as the nucleon classes and four different boundaries:
//
//   Z = 1, any energy           1.0115 * A * HadronNucleonXscNS   (identical to the nucleons')
//   Z > 1, E below 20 MeV       theCoulombFac[Z] * (Coulomb factor for pi+, 1/sqrt(E) for pi-)
//   Z > 1, 20 MeV to 91 GeV     G4UPiNuclearCrossSection
//   Z > 1, above 91 GeV         theGlauberFac[Z] * Glauber-Gribov
//
// and the energy is floored at fLowestEnergy = 1 MeV before any of it - so below 1 MeV the
// cross section is *constant*, at its 1 MeV value, rather than following 1/sqrt(E) down.
//
// THE BOUNDARY IS `<=` FOR ELASTIC AND `<` FOR INELASTIC
//
//   G4BGGPionElasticXS:    if(ekin <= fLowEnergy) { ... } else if(ekin > fGlauberEnergy)
//   G4BGGPionInelasticXS:  if(ekin <  fLowEnergy) { ... } else if(ekin > fGlauberEnergy)
//
// At exactly 20 MeV the elastic class takes the Coulomb branch and the inelastic class takes
// the G4UPiNuclearCrossSection branch. The two agree there by construction - the factor is
// built as `UPi(20 MeV) / factor(20 MeV)` - so this is a difference of a few ulps and not of
// physics, which is exactly why it would survive a port that "tidied" one of them: nothing
// would fail. Transcribed as two different comparisons, and the test compares at 20 MeV.
//
// Note also that the inelastic class's low-energy array is called theLowEPiPlus /
// theLowEPiMinus rather than theCoulombFacPiPlus / theCoulombFacPiMinus. Same construction,
// different name, and the inelastic class has no `theCoulombFac...[1] = 1` line at all
// (it sets theLowEPiPlus[1] and theLowEPiMinus[1] to 1, which Z = 1 never reads).
#pragma once
#include <cmath>

#include "core/units.cuh"
#include "data/materials.cuh"
#include "physics/hadronic/xs/bgg_nucleon_xs.cuh"
#include "physics/hadronic/xs/gg_hadron_nucleus_xsc.cuh"
#include "physics/hadronic/xs/hadron_nucleon_xsc.cuh"
#include "physics/hadronic/xs/nuclear_radii.cuh"
#include "physics/hadronic/xs/projectile.cuh"
#include "physics/hadronic/xs/refusal.cuh"
#include "physics/hadronic/xs/upi_nuclear_xs.cuh"

namespace g4gpu::hadronic::xs {

/// fLowEnergy for the two pion classes - 20 MeV, not the nucleons' 14.
template <typename real_t> __host__ __device__ constexpr real_t bgg_pion_low_energy() {
  return real_t(20.) * units::MeV<real_t>();
}
/// fLowestEnergy - the floor applied to the kinetic energy before anything else.
template <typename real_t> __host__ __device__ constexpr real_t bgg_pion_lowest_energy() {
  return real_t(1.) * units::MeV<real_t>();
}

template <typename real_t>
struct BggPionTable {
  bool is_elastic = true;
  int theA[93] = {0};
  real_t glauber_pip[93] = {real_t(0)};
  real_t glauber_pim[93] = {real_t(0)};
  real_t low_pip[93] = {real_t(0)};  ///< theCoulombFacPiPlus / theLowEPiPlus
  real_t low_pim[93] = {real_t(0)};  ///< theCoulombFacPiMinus / theLowEPiMinus
  UPiNuclearTable<real_t> upi;
};

/// G4BGGPion{Elastic,Inelastic}XS::CoulombFactorPiPlus. Both classes have it, identical.
template <typename real_t>
__host__ __device__ inline real_t bgg_pion_coulomb_factor_piplus(const BggPionTable<real_t>& t,
                                                                 real_t kin_energy, int Z) {
  if (!(kin_energy > real_t(0.0))) { return real_t(0.0); }
  return nr_coulomb_factor_nucleus<real_t>(Z, t.theA[Z], pi_plus<real_t>(), kin_energy);
}

/// G4BGGPion{Elastic,Inelastic}XS::FactorPiMinus - 1/sqrt(E), with E in MeV. Not a Coulomb
/// factor at all: a pi- has no barrier, and this is the s-wave 1/v shape of an attractive
/// channel.
template <typename real_t>
__host__ __device__ inline real_t bgg_pion_factor_piminus(real_t kin_energy) {
  return real_t(1.0) / sqrt(kin_energy);
}

/// G4BGGPion{Elastic,Inelastic}XS::GetElementCrossSection.
template <typename real_t>
__host__ __device__ inline XsValue<real_t> bgg_pion_element_xs(const BggPionTable<real_t>& t,
                                                                const Projectile<real_t>& p,
                                                                real_t kin_energy, int ZZ) {
  const real_t lowest = bgg_pion_lowest_energy<real_t>();
  const real_t ekin = (kin_energy > lowest) ? kin_energy : lowest;
  const int Z = (ZZ < 92) ? ZZ : 92;
  const bool is_piplus = (p.pdg == pdg::kPiPlus);

  if (1 == Z) {
    // The hydrogen branch is the nucleon classes' verbatim, including the 1.0115, except that
    // GetIsoCrossSection is called with the *unfloored* kinetic energy: Geant4 passes
    // `dp->GetKineticEnergy()`, not the `ekin` it just computed. Kept.
    const XsValue<real_t> iso =
        bgg_nucleon_iso_hydrogen<real_t>(t.is_elastic, p, kin_energy, 1);
    return {bgg_hydrogen_factor<real_t>() * iso.value, iso.refused};
  }

  const bool below = t.is_elastic ? (ekin <= bgg_pion_low_energy<real_t>())
                                  : (ekin < bgg_pion_low_energy<real_t>());
  if (below) {
    const real_t cross =
        is_piplus ? t.low_pip[Z] * bgg_pion_coulomb_factor_piplus<real_t>(t, ekin, Z)
                  : t.low_pim[Z] * bgg_pion_factor_piminus<real_t>(ekin);
    return {cross, XsRefusal::kNone};
  }
  if (ekin > bgg_glauber_energy<real_t>()) {
    const real_t fac = is_piplus ? t.glauber_pip[Z] : t.glauber_pim[Z];
    const XsValue<real_t> gg = t.is_elastic ? ggh_elastic<real_t>(p, ekin, Z, t.theA[Z])
                                            : ggh_inelastic<real_t>(p, ekin, Z, t.theA[Z]);
    return {fac * gg.value, gg.refused};
  }
  return t.is_elastic ? upi_elastic<real_t>(t.upi, p, Z, t.theA[Z], ekin)
                      : upi_inelastic<real_t>(t.upi, p, Z, t.theA[Z], ekin);
}

/// BuildPhysicsTable for one of the two pion classes, in Geant4's order.
template <typename real_t>
__host__ inline void bgg_build_pion_table(bool is_elastic, BggPionTable<real_t>& t) {
  t.is_elastic = is_elastic;
  upi_build_table<real_t>(t.upi);
  t.theA[0] = t.theA[1] = 1;

  const Projectile<real_t> pip = pi_plus<real_t>();
  const Projectile<real_t> pim = pi_minus<real_t>();
  const real_t eg = bgg_glauber_energy<real_t>();
  const real_t el = bgg_pion_low_energy<real_t>();

  for (int iz = 2; iz < 93; ++iz) {
    t.theA[iz] = g4lrint<real_t>(data::atomic_mass<real_t>(iz));
  }
  for (int iz = 2; iz < 93; ++iz) {
    const XsValue<real_t> csup = is_elastic ? ggh_elastic<real_t>(pip, eg, iz, t.theA[iz])
                                            : ggh_inelastic<real_t>(pip, eg, iz, t.theA[iz]);
    xs_fatal_if_refused(csup.refused, "bgg_build_pion_table (pi+, Glauber-Gribov)");
    const XsValue<real_t> csdn =
        is_elastic ? upi_elastic<real_t>(t.upi, pip, iz, t.theA[iz], eg)
                   : upi_inelastic<real_t>(t.upi, pip, iz, t.theA[iz], eg);
    xs_fatal_if_refused(csdn.refused, "bgg_build_pion_table (pi+, G4UPiNuclearCrossSection)");
    t.glauber_pip[iz] = csdn.value / csup.value;
  }
  for (int iz = 2; iz < 93; ++iz) {
    const XsValue<real_t> csup = is_elastic ? ggh_elastic<real_t>(pim, eg, iz, t.theA[iz])
                                            : ggh_inelastic<real_t>(pim, eg, iz, t.theA[iz]);
    xs_fatal_if_refused(csup.refused, "bgg_build_pion_table (pi-, Glauber-Gribov)");
    const XsValue<real_t> csdn =
        is_elastic ? upi_elastic<real_t>(t.upi, pim, iz, t.theA[iz], eg)
                   : upi_inelastic<real_t>(t.upi, pim, iz, t.theA[iz], eg);
    xs_fatal_if_refused(csdn.refused, "bgg_build_pion_table (pi-, G4UPiNuclearCrossSection)");
    t.glauber_pim[iz] = csdn.value / csup.value;
  }

  t.low_pip[1] = real_t(1.0);
  t.low_pim[1] = real_t(1.0);

  for (int iz = 2; iz < 93; ++iz) {
    const XsValue<real_t> csdn =
        is_elastic ? upi_elastic<real_t>(t.upi, pip, iz, t.theA[iz], el)
                   : upi_inelastic<real_t>(t.upi, pip, iz, t.theA[iz], el);
    xs_fatal_if_refused(csdn.refused, "bgg_build_pion_table (pi+, low energy)");
    t.low_pip[iz] = csdn.value / bgg_pion_coulomb_factor_piplus<real_t>(t, el, iz);
  }
  for (int iz = 2; iz < 93; ++iz) {
    const XsValue<real_t> csdn =
        is_elastic ? upi_elastic<real_t>(t.upi, pim, iz, t.theA[iz], el)
                   : upi_inelastic<real_t>(t.upi, pim, iz, t.theA[iz], el);
    xs_fatal_if_refused(csdn.refused, "bgg_build_pion_table (pi-, low energy)");
    t.low_pim[iz] = csdn.value / bgg_pion_factor_piminus<real_t>(el);
  }
}

}  // namespace g4gpu::hadronic::xs
