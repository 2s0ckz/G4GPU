// G4ComponentGGNuclNuclXsc - nucleus on nucleus in the Glauber-Gribov approximation.
//
// Transcribed from source/processes/hadronic/cross_sections/src/G4ComponentGGNuclNuclXsc.cc
// (11.1.1): ComputeCrossSections and ComputeCoulombBarier.
//
// In QBBC this is the elastic cross section for d, t, He3 and alpha (G4HadronElasticPhysicsXS),
// the elastic *and* inelastic cross section for GenericIon (G4IonElasticPhysics and
// G4IonPhysicsXS), and the high-energy hand-over of G4ParticleInelasticXS for every ion -
// G4ParticleInelasticXS's constructor asks the registry for "Glauber-Gribov Nucl-nucl" for
// anything that is not a proton.
//
// FOUR THINGS THAT ARE NOT THE HADRON-NUCLEUS VERSION
//
// 1. The nucleon-nucleon cross section is evaluated at the *per-nucleon* kinetic energy,
//    `pTkin = kinEnergy/pA`. A 200 MeV alpha is four 50 MeV nucleons here.
// 2. Only two of the four nucleon-nucleon combinations are computed - pp and np, both against
//    a proton target - and the isospin counting `(pZ*Z + pN*tN)` and `(pZ*tN + pN*Z)` supplies
//    the rest. So nn is taken to equal pp and pn to equal np, which is what isospin symmetry
//    says and not an approximation introduced here.
// 3. There is no bar-correction table and no `A > 1` special case; instead the *hydrogen*
//    target is handed to G4ComponentGGHadronNucleusXsc with projectile and target swapped:
//
//      G4double e = kinEnergy*proton_mass_c2/aParticle->GetPDGMass();
//      fHadrNucl->ComputeCrossSections(theProton, e, pZ, pA, pL);
//
//    a proton of the same velocity striking the ion. Transcribed verbatim, including the
//    velocity scaling, because it is the only path an ion-on-hydrogen cross section has - and
//    hydrogen is half the atoms in water.
// 4. `ppInXsc`, which feeds the production cross section, is read from the shared
//    G4HadronNucleonXsc member AFTER the lambda call rather than from the pp one - so for a
//    hypernucleus projectile it is 0.88 times the pp inelastic and not the pp inelastic. See
//    the comment at the read itself. The hadron-nucleus version has no such call to be
//    overwritten by.
//
// THE COULOMB BARRIER GATES EVERYTHING
//
// `if (cB > 0.)` wraps the whole calculation, and all five cross sections are set to exactly
// zero when it is not. cB is ComputeCoulombBarier, which needs the target's nuclear mass; a
// nuclide outside data/nuclei_mass_ame12.hh is refused by name rather than given a zero mass,
// which would produce a barrier that is never passed and a cross section that is always zero.
#pragma once
#include <cmath>

#include "core/units.cuh"
#include "physics/hadronic/xs/g4pow_extra.cuh"
#include "physics/hadronic/xs/gg_hadron_nucleus_xsc.cuh"
#include "physics/hadronic/xs/hadron_nucleon_xsc.cuh"
#include "physics/hadronic/xs/nuclear_radii.cuh"
#include "physics/hadronic/xs/projectile.cuh"
#include "physics/hadronic/xs/refusal.cuh"

namespace g4gpu::hadronic::xs {

/// G4ComponentGGNuclNuclXsc::ComputeCoulombBarier.
///
/// Not the same expression as G4NuclearRadii::CoulombFactor(Z, A, ...): the barrier here is
/// `alpha*hbarc*pZ*Z*0.5/(pR+tR)` with radii from G4NuclearRadii::Radius, where that one uses
/// `0.5*alpha*hbarc*pZ*Z/(pR+tR)` with the target radius from RadiusCB. Same algebraic factor,
/// different radii - RadiusCB carries the per-Z r0 table and Radius does not.
///
/// Returns a negative value to mean "the target mass is not known", which the caller turns
/// into a refusal. Geant4 cannot express that because its mass lookup always answers.
///
/// @param p_tkin  Geant4 names this parameter `pTkin` and its one caller passes `kinEnergy`,
///                the projectile's total kinetic energy - NOT the per-nucleon `pTkin` the
///                same name means twenty lines earlier in ComputeCrossSections. Kept as the
///                source has it, because dividing by A here would move the barrier.
template <typename real_t>
__host__ __device__ inline real_t ggnn_coulomb_barrier(const Projectile<real_t>& p,
                                                       real_t p_tkin, int Z, int A, real_t pR,
                                                       real_t tR) {
  if (!data::nuclear_mass_known(A, Z)) { return real_t(-1); }
  // `G4int pZ = aParticle->GetPDGCharge()*inve;` - a truncation, not a rint, unlike the one
  // in ComputeCrossSections. Identical for an integer charge; kept as the source has it.
  const int pZ = static_cast<int>(p.charge);
  const real_t pM = p.mass;
  const real_t tM = data::nuclear_mass<real_t>(A, Z);
  const real_t pElab = p_tkin + pM;
  const real_t totEcm = sqrt(pM * pM + tM * tM + real_t(2.) * pElab * tM);
  const real_t totTcm = totEcm - pM - tM;

  const real_t qfact = units::fine_structure_const<real_t>() * units::hbarc<real_t>();
  const real_t bC = qfact * static_cast<real_t>(pZ) * static_cast<real_t>(Z) * real_t(0.5)
                    / (pR + tR);
  return (totTcm <= bC) ? real_t(0.) : real_t(1.) - bC / totTcm;
}

/// G4ComponentGGNuclNuclXsc::ComputeCrossSections.
///
/// @param kin_energy  the projectile's TOTAL kinetic energy, MeV - not per nucleon. The
///                    division by A happens inside, as it does in Geant4.
template <typename real_t>
__host__ __device__ inline HadXs<real_t> ggnn_compute_cross_sections(
    const Projectile<real_t>& p, real_t kin_energy, int Z, int A) {
  HadXs<real_t> out;

  const int pZ = g4lrint<real_t>(p.charge);
  const int pA = p.baryon_number;
  const int pL = p.n_lambdas;
  const bool pHN = p.is_hypernucleus();
  constexpr real_t cHN = real_t(0.88);

  // An ANTI-nucleus is not this model's projectile, and the failure is silent rather than
  // loud: `pTkin = kinEnergy/pA` below would divide by a negative baryon number and
  // `nr_radius(pZ, pA)` would be asked for a negative Z and A. Geant4 never calls
  // G4ComponentGGNuclNuclXsc for one - an anti-nucleus goes to G4ComponentAntiNuclNuclearXS
  // through "AntiAGlauber" - and it has no guard here because nothing in a physics list
  // reaches it. This port is callable with any Projectile, so it refuses by name.
  if (pA < 1) {
    out.refused = XsRefusal::kComponentAntiNuclNuclearXS;
    return out;
  }

  // hydrogen: a proton of the same velocity on the projectile nucleus
  if (1 == Z && 1 == A) {
    const real_t e = kin_energy * units::proton_mass_c2<real_t>() / p.mass;
    return ggh_compute_cross_sections<real_t>(proton<real_t>(), e, pZ, pA, pL);
  }

  constexpr real_t cofInelastic = real_t(2.4);
  constexpr real_t cofTotal = real_t(2.0);

  const real_t pTkin = kin_energy / static_cast<real_t>(pA);

  const int pN = pA - pZ;
  const int tN = A - Z;

  const real_t tR = nr_radius<real_t>(Z, A);
  real_t pR = nr_radius<real_t>(pZ, pA);

  if (pHN) {
    pR *= sqrt(g4pow_z23<real_t>(pA - pL) + cHN * g4pow_z23<real_t>(pL))
          / data::g4pow_z13<real_t>(pA);
  }

  const real_t cB = ggnn_coulomb_barrier<real_t>(p, kin_energy, Z, A, pR, tR);
  if (cB < real_t(0)) {
    out.refused = XsRefusal::kNuclearMassNotTabulated;
    return out;
  }

  if (cB > real_t(0.)) {
    const Projectile<real_t> thePr = proton<real_t>();
    const HadXs<real_t> pp = hn_xsc_ns<real_t>(thePr, thePr, pTkin);
    real_t sigma = static_cast<real_t>(pZ * Z + pN * tN) * pp.total;
    // `ppInXsc` IS NOT THE pp INELASTIC WHEN THE PROJECTILE IS A HYPERNUCLEUS
    //
    // Geant4 reads it from the shared G4HadronNucleonXsc member AFTER the lambda call:
    //
    //     sigma = (pZ*Z+pN*tN)*fHNXsc->HadronNucleonXscNS(theProton, theProton, pTkin);
    //     if(pHN) sigma += pL*A*fHNXsc->HadronNucleonXsc(theLambda, theProton, pTkin);
    //     G4double ppInXsc = fHNXsc->GetInelasticHadronNucleonXsc();
    //
    // and `HadronNucleonXsc(theLambda, ...)` dispatches to HyperonNucleonXscNS, which does
    // `fTotalXsc = coeff*HadronNucleonXscNS(theProton, nucleon, ekin); fInelasticXsc *= coeff;`
    // with coeff = 0.88 for a lambda. So for a hypernucleus Geant4's ppInXsc is the LAMBDA
    // call's inelastic - 0.88 times the pp one - and that is what feeds `xratio` and
    // fProductionXsc below. This read `pp.inelastic` unconditionally, which is right for every
    // projectile that is not a hypernucleus and wrong for one that is.
    //
    // NOT TESTABLE TODAY, AND SAID SO HERE RATHER THAN LEFT TO LOOK COVERED: the whole
    // hypernucleus arm is refused three lines down, because HyperonNucleonXscNS is not ported
    // (XsRefusal::kHyperonNucleonXscNS), and G4HadronicParameters::EnableHyperNuclei is off by
    // default so nothing in QBBC produces one. Whoever ports HyperonNucleonXscNS makes this
    // line live; it is written the way Geant4 reads it so that they do not have to find it.
    real_t ppInXsc = pp.inelastic;
    if (pHN) {
      const HadXs<real_t> lp = hadron_nucleon_xsc<real_t>(lambda<real_t>(), thePr, pTkin);
      if (!lp.ok()) {
        out.refused = lp.refused;
        return out;
      }
      sigma += static_cast<real_t>(pL * A) * lp.total;
      ppInXsc = lp.inelastic;
    }

    const HadXs<real_t> np = hn_xsc_ns<real_t>(neutron<real_t>(), thePr, pTkin);
    sigma += static_cast<real_t>(pZ * tN + pN * Z) * np.total;
    const real_t npInXsc = np.inelastic;

    const real_t nucleusSquare =
        cofTotal * units::pi<real_t>() * (pR * pR + tR * tR);  // basically 2piRR

    const real_t ratio = sigma / nucleusSquare;
    out.total = nucleusSquare * log(real_t(1.) + ratio) * cB;
    out.inelastic =
        nucleusSquare * log(real_t(1.) + cofInelastic * ratio) * cB / cofInelastic;
    out.elastic =
        (out.total - out.inelastic > real_t(0.0)) ? out.total - out.inelastic : real_t(0.0);

    const real_t difratio = ratio / (real_t(1.) + ratio);
    out.diffraction = real_t(0.5) * nucleusSquare * (difratio - log(real_t(1.) + difratio));

    const real_t xratio = (static_cast<real_t>(pZ * Z + pN * tN) * ppInXsc
                           + static_cast<real_t>(pZ * tN + pN * Z) * npInXsc)
                          / nucleusSquare;
    out.production =
        nucleusSquare * log(real_t(1.) + cofInelastic * xratio) * cB / cofInelastic;
    out.production = (out.production < out.inelastic) ? out.production : out.inelastic;
  }
  // else: every field stays at its zero initialiser, which is what Geant4's else branch does.
  return out;
}

/// The G4double-A element overloads, which round A with G4lrint first.
template <typename real_t>
__host__ __device__ inline XsValue<real_t> ggnn_inelastic_element(const Projectile<real_t>& p,
                                                                  real_t ekin, int Z,
                                                                  real_t A) {
  const HadXs<real_t> r =
      ggnn_compute_cross_sections<real_t>(p, ekin, Z, g4lrint<real_t>(A));
  return {r.inelastic, r.refused};
}
template <typename real_t>
__host__ __device__ inline XsValue<real_t> ggnn_elastic_element(const Projectile<real_t>& p,
                                                                real_t ekin, int Z,
                                                                real_t A) {
  const HadXs<real_t> r =
      ggnn_compute_cross_sections<real_t>(p, ekin, Z, g4lrint<real_t>(A));
  return {r.elastic, r.refused};
}

}  // namespace g4gpu::hadronic::xs
