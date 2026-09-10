// G4BGGNucleonElasticXS and G4BGGNucleonInelasticXS, complete - the three branches
// barashenkov_xs.cuh refuses in its header, and the dispatcher that joins them to the middle
// band it already has.
//
// Transcribed from source/processes/hadronic/cross_sections/src/G4BGGNucleonElasticXS.cc and
// G4BGGNucleonInelasticXS.cc (11.1.1): GetElementCrossSection, GetIsoCrossSection,
// CoulombFactor, and the four per-Z tables BuildPhysicsTable fills.
//
// These two are QBBC's proton and neutron elastic and inelastic cross sections
// (G4HadronElasticPhysicsXS for the proton, G4HadronInelasticQBBC via G4ParticleInelasticXS
// for the proton's inelastic - see below), and they are three cross sections stitched
// together at two energies:
//
//   Z = 1, any energy      1.0115 * A * G4HadronNucleonXsc::HadronNucleonXscNS
//   Z > 1, E <= 14 MeV     theCoulombFac[Z] * CoulombFactor(E, Z)
//   Z > 1, 14 MeV < E <= 91 GeV   the Barashenkov table (barashenkov_xs.cuh)
//   Z > 1, E > 91 GeV      theGlauberFac[Z] * Glauber-Gribov
//
// The two `Fac` arrays are what make it continuous, and they are not constants: each is the
// Barashenkov value at the boundary divided by the other model's value at the same boundary,
// per Z and per projectile. So they have to be *built* from the two models rather than
// tabulated, which is why this file has a host-side table builder and the callers take the
// table as an argument. bgg_build_nucleon_table is BuildPhysicsTable, in order.
//
// FOUR ASYMMETRIES BETWEEN THE ELASTIC AND INELASTIC CLASSES
//
// The two files look like copies of each other and are not:
//
// 1. `CoulombFactor` is a different function. Elastic: `G4NuclearRadii::CoulombFactor(Z,
//    theA[Z], proton, E)` for a proton and a flat 1.0 for a neutron, so a neutron's elastic
//    cross section is simply *constant* below 14 MeV at its 14 MeV value. Inelastic: the
//    proton gets that factor times a three-parameter sigmoid from G4ProtonInelasticCrossSection,
//    and the neutron gets a five-parameter form from G4NeutronInelasticCrossSection - a real
//    energy dependence, not a constant.
// 2. The inelastic class's proton branch returns 0 when `G4NuclearRadii::CoulombFactor` returns
//    0 (`if(res > 0.0)` guards the shape factors), so below the barrier it is exactly zero.
// 3. `theCoulombFacP[0]` and `theCoulombFacN[0]` are set only in the elastic class; the
//    inelastic one sets index 1 and leaves index 0 at zero. Neither is read - Z is at least 1
//    - but it is the kind of difference that makes a diff of the two files look wrong.
// 4. The elastic class's GetElementCrossSection carries the comment "this method should be
//    called only for Z > 1" directly above a branch on `1 == Z`. The comment is stale; the
//    branch is live, and it is how hydrogen is answered.
//
// THE 1.0115
//
// A bare literal in all four BGG classes, with no comment anywhere in Geant4. It multiplies
// the hydrogen cross section only. Transcribed as a literal, in one place, named.
//
// WHO ACTUALLY CALLS THESE IN QBBC
//
// G4HadronElasticPhysicsXS gives the proton G4BGGNucleonElasticXS and the neutron
// G4NeutronElasticXS, so the nucleon *elastic* BGG class is a proton-only path in QBBC even
// though it is written for both. G4HadronInelasticQBBC gives the proton G4ParticleInelasticXS
// and the neutron G4NeutronInelasticXS, so G4BGGNucleonInelasticXS is not in QBBC's chain at
// all in 11.1.1 - it is what several other reference lists use, and it is the class the
// Barashenkov component exists for. Both are ported because the plan asks for them completed,
// and because they are the only consumer of the Coulomb-barrier branch this package has.
#pragma once
#include <cmath>

#include "core/units.cuh"
#include "data/materials.cuh"
#include "physics/hadronic/barashenkov_xs.cuh"
#include "physics/hadronic/xs/gg_hadron_nucleus_xsc.cuh"
#include "physics/hadronic/xs/hadron_nucleon_xsc.cuh"
#include "physics/hadronic/xs/nuclear_radii.cuh"
#include "physics/hadronic/xs/projectile.cuh"
#include "physics/hadronic/xs/refusal.cuh"

namespace g4gpu::hadronic::xs {

/// The 1.0115 the hydrogen branch of all four BGG classes multiplies by.
template <typename real_t> __host__ __device__ constexpr real_t bgg_hydrogen_factor() {
  return real_t(1.0115);
}
/// fGlauberEnergy - above this the Glauber-Gribov component takes over.
template <typename real_t> __host__ __device__ constexpr real_t bgg_glauber_energy() {
  return real_t(91.) * units::GeV<real_t>();
}
/// fLowEnergy for the two nucleon classes (the pion ones use 20 MeV).
template <typename real_t> __host__ __device__ constexpr real_t bgg_nucleon_low_energy() {
  return real_t(14.0) * units::MeV<real_t>();
}

/// The per-Z tables BuildPhysicsTable fills, for one of the two classes.
///
/// `built_for_proton` is not decoration. See bgg_build_nucleon_table.
template <typename real_t>
struct BggNucleonTable {
  bool is_elastic = true;
  bool built_for_proton = true;
  int theA[93] = {0};
  real_t glauber_p[93] = {real_t(0)};
  real_t glauber_n[93] = {real_t(0)};
  real_t coulomb_p[93] = {real_t(0)};
  real_t coulomb_n[93] = {real_t(0)};
};

/// `const G4double llog10 = G4Log(10.)` at the top of G4BGGNucleonInelasticXS.cc, used to turn
/// a natural logarithm into a base-10 one.
template <typename real_t> __host__ __device__ inline real_t bgg_llog10() {
  return log(real_t(10.));
}

/// G4BGGNucleonElasticXS::CoulombFactor - a proton gets the nuclear Coulomb factor, a neutron
/// gets exactly 1.
template <typename real_t>
__host__ __device__ inline real_t bgg_elastic_coulomb_factor(const BggNucleonTable<real_t>& t,
                                                             const Projectile<real_t>& p,
                                                             real_t kin_energy, int Z) {
  if (p.pdg != pdg::kProton) { return real_t(1.0); }
  return nr_coulomb_factor_nucleus<real_t>(Z, t.theA[Z], p, kin_energy);
}

/// G4BGGNucleonInelasticXS::CoulombFactor - two entirely different forms, one per projectile.
///
/// The proton's extra factors are lifted from the retired G4ProtonInelasticCrossSection and
/// the neutron's whole expression from G4NeutronInelasticCrossSection; Geant4 says so in
/// comments. `aa` is theA[Z] as a double, and `elog` is log10(E/GeV).
template <typename real_t>
__host__ __device__ inline real_t bgg_inelastic_coulomb_factor(
    const BggNucleonTable<real_t>& t, const Projectile<real_t>& p, real_t kin_energy, int Z) {
  real_t res = real_t(0.0);
  if (kin_energy <= real_t(0.0)) { return res; }

  const real_t elog = log(kin_energy / units::GeV<real_t>()) / bgg_llog10<real_t>();
  const real_t aa = static_cast<real_t>(t.theA[Z]);

  if (p.pdg == pdg::kProton) {
    res = nr_coulomb_factor_nucleus<real_t>(Z, t.theA[Z], p, kin_energy);
    // from G4ProtonInelasticCrossSection
    if (res > real_t(0.0)) {
      real_t ff1 = real_t(5.6) - real_t(0.016) * aa;   // slope of the drop at medium energies
      real_t ff2 = real_t(1.37) + real_t(1.37) / aa;   // start of the slope
      const real_t ff3 =
          real_t(0.8) + real_t(18.) / aa - real_t(0.002) * aa;  // stephight
      res *= (real_t(1.0)
              + ff3 * (real_t(1.0) - (real_t(1.0) / (real_t(1) + exp(-ff1 * (elog + ff2))))));
      ff1 = real_t(8.) - real_t(8.) / aa - real_t(0.008) * aa;      // slope of the rise
      ff2 = real_t(2.34) - real_t(5.4) / aa - real_t(0.0028) * aa;  // start of the rise
      res /= (real_t(1.0) + exp(-ff1 * (elog + ff2)));
    }
  } else {
    // from G4NeutronInelasticCrossSection
    const real_t p3 = real_t(0.6) + real_t(13.) / aa - real_t(0.0005) * aa;
    const real_t p4 = real_t(7.2449) - real_t(0.018242) * aa;
    const real_t p5 = real_t(1.36) + real_t(1.8) / aa + real_t(0.0005) * aa;
    const real_t p6 = real_t(1.) + real_t(200.) / aa + real_t(0.02) * aa;
    const real_t p7 = real_t(3.0) - (aa - real_t(70.)) * (aa - real_t(200.)) / real_t(11000.);

    const real_t firstexp = exp(-p4 * (elog + p5));
    const real_t secondexp = exp(-p6 * (elog + p7));

    res = (real_t(1.) + p3 * firstexp / (real_t(1.) + firstexp)) / (real_t(1.) + secondexp);
  }
  return res;
}

/// The CoulombFactor of whichever class the table belongs to.
template <typename real_t>
__host__ __device__ inline real_t bgg_nucleon_coulomb_factor(const BggNucleonTable<real_t>& t,
                                                             const Projectile<real_t>& p,
                                                             real_t kin_energy, int Z) {
  return t.is_elastic ? bgg_elastic_coulomb_factor<real_t>(t, p, kin_energy, Z)
                      : bgg_inelastic_coulomb_factor<real_t>(t, p, kin_energy, Z);
}

/// G4BGGNucleon{Elastic,Inelastic}XS::GetIsoCrossSection - "this method should be called only
/// for Z = 1", and it is: `A * HadronNucleonXscNS`, elastic or inelastic component.
template <typename real_t>
__host__ __device__ inline XsValue<real_t> bgg_nucleon_iso_hydrogen(bool is_elastic,
                                                                    const Projectile<real_t>& p,
                                                                    real_t kin_energy, int A) {
  const HadXs<real_t> hn = hn_xsc_ns<real_t>(p, proton<real_t>(), kin_energy);
  if (!hn.ok()) { return {real_t(0), hn.refused}; }
  const real_t x = is_elastic ? hn.elastic : hn.inelastic;
  return {static_cast<real_t>(A) * x, XsRefusal::kNone};
}

/// G4BGGNucleon{Elastic,Inelastic}XS::GetElementCrossSection.
///
/// The two classes differ in the middle band only by which Barashenkov column they read, and
/// at the top only by which Glauber-Gribov component member they read - so one function with
/// the table's `is_elastic` flag reproduces both, and the asymmetries listed in the file
/// header live in bgg_nucleon_coulomb_factor.
template <typename real_t>
__host__ __device__ inline XsValue<real_t> bgg_nucleon_element_xs(
    const BggNucleonTable<real_t>& t, const Projectile<real_t>& p, real_t kin_energy, int ZZ) {
  const int Z = (ZZ < 92) ? ZZ : 92;
  const bool is_proton = (p.pdg == pdg::kProton);
  const real_t ekin = kin_energy;

  if (1 == Z) {
    const XsValue<real_t> iso =
        bgg_nucleon_iso_hydrogen<real_t>(t.is_elastic, p, ekin, 1);
    return {bgg_hydrogen_factor<real_t>() * iso.value, iso.refused};
  }

  if (ekin <= bgg_nucleon_low_energy<real_t>()) {
    real_t cross = is_proton ? t.coulomb_p[Z] : t.coulomb_n[Z];
    cross *= bgg_nucleon_coulomb_factor<real_t>(t, p, ekin, Z);
    return {cross, XsRefusal::kNone};
  }
  if (ekin > bgg_glauber_energy<real_t>()) {
    const real_t fac = is_proton ? t.glauber_p[Z] : t.glauber_n[Z];
    const XsValue<real_t> gg =
        t.is_elastic ? ggh_elastic<real_t>(p, ekin, Z, t.theA[Z])
                     : ggh_inelastic<real_t>(p, ekin, Z, t.theA[Z]);
    return {fac * gg.value, gg.refused};
  }
  const NucleonNucleusXs<real_t> b = barashenkov_xs<real_t>(Z, ekin, is_proton);
  return {t.is_elastic ? b.elastic : b.inelastic, XsRefusal::kNone};
}

/// BuildPhysicsTable, for one of the two classes. Host-only: it evaluates both neighbouring
/// models 4 x 91 times.
///
/// The order matters and is Geant4's: theA[] first (the Glauber-Gribov loop for the proton
/// fills it, and the neutron loop reads it), then the two Glauber factors at 91 GeV, then
/// theCoulombFac[1] = 1, then the two Coulomb factors at 14 MeV. Every division is
/// `barashenkov / other`, so the port must not reorder it into `other / barashenkov`.
///
/// WHY THIS TAKES A PROJECTILE AND FILLS BOTH PROJECTILES' TABLES WITH IT
///
/// `built_for_proton` is which particle BuildPhysicsTable was called with, and it changes the
/// NEUTRON's cross section below 14 MeV. That is not a port artefact; it is what Geant4 does,
/// and the mechanism is worth spelling out because it looks like a bug in the port until it is
/// traced:
///
///   * theCoulombFacP[], theCoulombFacN[], theGlauberFacP[], theGlauberFacN[] and theA[] are
///     `static` members of G4BGGNucleonElasticXS, shared by every instance.
///   * BuildPhysicsTable returns immediately when they are already filled (`if(0 != theA[0])`),
///     so exactly ONE instance ever fills them - the first one initialised.
///   * `CoulombFactor(kinEnergy, Z)` is a member function that branches on `isProton`, a
///     per-INSTANCE flag, and takes no particle argument.
///   * BuildPhysicsTable divides BOTH tables by it:
///         theCoulombFacP[iz] = barashenkov_p(14 MeV) / CoulombFactor(14 MeV, iz);
///         theCoulombFacN[iz] = barashenkov_n(14 MeV) / CoulombFactor(14 MeV, iz);
///     with the one `isProton` of whichever instance got there first.
///   * GetElementCrossSection then multiplies by `CoulombFactor(ekin, Z)` using its OWN
///     instance's isProton.
///
/// So a neutron instance whose tables were built by a proton instance returns
/// `barashenkov_n(14 MeV) / CoulombFactor_proton(14 MeV, Z) * 1.0` below 14 MeV - the elastic
/// class's neutron CoulombFactor being a flat 1. For Z = 92 that is a factor 2.13 above the
/// naive `barashenkov_n(14 MeV)`, and for the inelastic class's neutron at 1.8 keV it is a
/// factor of ten. Both were wrong here until had_bgg.csv said so.
///
/// The dump builds the proton instance first for both classes, which is also QBBC's order -
/// G4HadronElasticPhysicsXS constructs G4BGGNucleonElasticXS for the proton and gives the
/// neutron G4NeutronElasticXS instead, so in QBBC the proton is the only instance there is.
template <typename real_t>
__host__ inline void bgg_build_nucleon_table(bool is_elastic, BggNucleonTable<real_t>& t,
                                             bool built_for_proton = true) {
  t.is_elastic = is_elastic;
  t.built_for_proton = built_for_proton;
  t.theA[0] = t.theA[1] = 1;

  const Projectile<real_t> pr = proton<real_t>();
  const Projectile<real_t> ne = neutron<real_t>();
  const real_t eg = bgg_glauber_energy<real_t>();
  const real_t el = bgg_nucleon_low_energy<real_t>();

  for (int iz = 2; iz < 93; ++iz) {
    // G4lrint(G4NistManager::GetAtomicMassAmu(iz)) - data/materials.cuh's atomic_mass is that
    // same table, dumped from Geant4 (see its header).
    t.theA[iz] = g4lrint<real_t>(data::atomic_mass<real_t>(iz));
  }
  for (int iz = 2; iz < 93; ++iz) {
    const XsValue<real_t> csup = is_elastic ? ggh_elastic<real_t>(pr, eg, iz, t.theA[iz])
                                            : ggh_inelastic<real_t>(pr, eg, iz, t.theA[iz]);
    xs_fatal_if_refused(csup.refused, "bgg_build_nucleon_table (proton, Glauber-Gribov)");
    const NucleonNucleusXs<real_t> b = barashenkov_xs<real_t>(iz, eg, true);
    const real_t csdn = is_elastic ? b.elastic : b.inelastic;
    t.glauber_p[iz] = csdn / csup.value;
  }
  for (int iz = 2; iz < 93; ++iz) {
    const XsValue<real_t> csup = is_elastic ? ggh_elastic<real_t>(ne, eg, iz, t.theA[iz])
                                            : ggh_inelastic<real_t>(ne, eg, iz, t.theA[iz]);
    xs_fatal_if_refused(csup.refused, "bgg_build_nucleon_table (neutron, Glauber-Gribov)");
    const NucleonNucleusXs<real_t> b = barashenkov_xs<real_t>(iz, eg, false);
    const real_t csdn = is_elastic ? b.elastic : b.inelastic;
    t.glauber_n[iz] = csdn / csup.value;
  }

  // The elastic class sets index 0 as well; the inelastic one does not. Both set index 1.
  t.coulomb_p[1] = t.coulomb_n[1] = real_t(1.0);
  if (is_elastic) { t.coulomb_p[0] = t.coulomb_n[0] = real_t(1.0); }

  // The divisor is the BUILDING instance's CoulombFactor for both loops, not the projectile
  // whose column is being filled - see the comment above this function.
  const Projectile<real_t>& builder = built_for_proton ? pr : ne;
  for (int iz = 2; iz < 93; ++iz) {
    const NucleonNucleusXs<real_t> b = barashenkov_xs<real_t>(iz, el, true);
    const real_t csdn = is_elastic ? b.elastic : b.inelastic;
    t.coulomb_p[iz] = csdn / bgg_nucleon_coulomb_factor<real_t>(t, builder, el, iz);
  }
  for (int iz = 2; iz < 93; ++iz) {
    const NucleonNucleusXs<real_t> b = barashenkov_xs<real_t>(iz, el, false);
    const real_t csdn = is_elastic ? b.elastic : b.inelastic;
    t.coulomb_n[iz] = csdn / bgg_nucleon_coulomb_factor<real_t>(t, builder, el, iz);
  }
}

}  // namespace g4gpu::hadronic::xs
