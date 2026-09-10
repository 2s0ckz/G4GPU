// G4HadronNucleonXsc - hadron on a single free nucleon.
//
// Transcribed from source/processes/hadronic/cross_sections/src/G4HadronNucleonXsc.cc
// (11.1.1): HadronNucleonXsc (the dispatcher), HadronNucleonXscPDG, HadronNucleonXscNS,
// KaonNucleonXscNS, KaonNucleonXscGG, KaonNucleonXscVG, CoulombBarrier, and the file-scope
// constants at its top.
//
// This is the bottom of the hadronic cross-section stack. Two things sit on it:
//
//   * the Z = 1 branch of all four BGG classes. G4BGGNucleonElasticXS::GetElementCrossSection
//     answers hydrogen with `1.0115*GetIsoCrossSection(dp,1,1)`, and GetIsoCrossSection is
//     `A * fHadron->HadronNucleonXscNS(...)` on the elastic or inelastic component. The 1.0115
//     is a bare literal in Geant4 with no comment; it appears identically in all four classes.
//     Hydrogen is the branch docs/PORTED.md 2.1 lists first among the three barashenkov_xs.cuh
//     refuses, and it "matters most for water".
//   * every Glauber-Gribov component. G4ComponentGGHadronNucleusXsc sums
//     `Z*HadronNucleonXsc(p) + N*HadronNucleonXsc(n)` and G4ComponentGGNuclNuclXsc sums four
//     nucleon-nucleon combinations, so the nucleus cross sections are this function folded
//     with a nuclear radius.
//
// WHAT IS TRANSCRIBED AND WHAT IS REFUSED
//
// p, n, pbar, nbar, pi+, pi-, K+, K-, K0S, K0L and gamma are complete. Two branches of the
// dispatcher are refused by name in refusal.cuh:
//   HyperonNucleonXscNS   (|pdg| > 3000 and in the hyperon list) - Lambda, Sigma, Xi, Omega
//                         and their charm and bottom partners. Reachable in QBBC only through
//                         a cascade emitting one, i.e. after P9/P10/P11 exist; and through
//                         G4ComponentGGNuclNuclXsc's hypernucleus branch, which
//                         G4HadronicParameters::EnableHyperNuclei gates off by default.
//   SCBMesonNucleonXscNS  (|pdg| > 220 and in the s/c/b meson list) - D, B, eta, eta', J/psi.
//                         `EnableBCParticles = 1` in 11.1.1 puts b- and c-hadrons in QBBC's
//                         chain, so this is a real gap and not a hypothetical one; it is
//                         reachable only above several GeV through FTF (plan section 2).
// Sigma- is NOT in that refusal: pdg 3112 has its own row in HadronNucleonXscPDG's parameter
// list, and the dispatcher's hyperon test catches it first, so 3112 reaches
// HyperonNucleonXscNS and the PDG row is dead code. Transcribed as dead code, with the
// dispatcher's order preserved, because changing the order would "fix" a branch Geant4 does
// not take.
//
// A TYPO THAT IS PART OF THE ANSWER
//
// The pLab >= 373 GeV/c elastic form appears twice, once for a neutron projectile and once
// for a proton, and the two are not the same expression:
//
//   neutron:  6.5 + 0.308*G4Exp(G4Log(G4Log(sMand/400.)*1.65)) + ...
//   proton:   6.5 + 0.308*G4Exp(G4Log(G4Log(sMand/400.))*1.65) + ...
//
// The 1.65 is inside the outer logarithm for the neutron and outside it for the proton. Since
// exp(log(x*1.65)) is 1.65*x and exp(log(x)*1.65) is x^1.65, the SUB-EXPRESSION differs by a
// factor of 2.4 at 373 GeV/c - 0.924 against 0.384.
//
// The cross section does not, and the difference is worth stating in the right size, because
// an earlier version of this comment said "a factor of two" and that is the kind of claim a
// later reader checks by deciding it must be a bug. The 0.308*(...) term is one of three
// additive terms in an elastic cross section dominated by the constant 6.5, so measured on a
// neutron-proton pair:
//
//     pLab        neutron form   proton form   ratio
//     373 GeV/c     7.2425 mb     7.0759 mb    1.024
//       1 TeV/c     7.5770 mb     7.4235 mb    1.021
//      10 TeV/c     8.5572 mb     9.4476 mb    0.906
//
// 2.4% at the boundary, and it REVERSES SIGN by 10 TeV/c rather than diverging. Transcribed
// verbatim, both of them: this is upstream's, the oracle contains it, and a port that
// "corrected" it would be 2.4% away from Geant4 for a neutron above 373 GeV/c and right about
// the physics.
#pragma once
#include <cmath>

#include "core/units.cuh"
#include "physics/hadronic/xs/g4pow_extra.cuh"
#include "physics/hadronic/xs/nuclear_radii.cuh"
#include "physics/hadronic/xs/projectile.cuh"
#include "physics/hadronic/xs/refusal.cuh"

namespace g4gpu::hadronic::xs {

// The file-scope constants of G4HadronNucleonXsc.cc, with its own names.
template <typename real_t> __host__ __device__ constexpr real_t hn_inv_gev() {
  return real_t(1) / units::GeV<real_t>();
}
template <typename real_t> __host__ __device__ constexpr real_t hn_inv_gev2() {
  return real_t(1) / (units::GeV<real_t>() * units::GeV<real_t>());
}
template <typename real_t> __host__ __device__ constexpr real_t hn_min_log_p() {
  return real_t(3.5);  ///< min of (lnP-minLogP)^2
}
template <typename real_t> __host__ __device__ constexpr real_t hn_cof_log_e() {
  return real_t(.0557);  ///< elastic (lnP-minLogP)^2
}
template <typename real_t> __host__ __device__ constexpr real_t hn_cof_log_t() {
  return real_t(.3);  ///< total (lnP-minLogP)^2
}
template <typename real_t> __host__ __device__ constexpr real_t hn_p_min() {
  return real_t(.1);  ///< fast LE calculation, GeV/c
}
template <typename real_t> __host__ __device__ constexpr real_t hn_p_max() {
  return real_t(1000.);  ///< fast HE calculation, GeV/c
}
/// Protection against zero kinetic energy.
template <typename real_t> __host__ __device__ constexpr real_t hn_ekin_min() {
  return real_t(0.1) * units::MeV<real_t>();
}
/// Highest kinetic energy at which the Coulomb barrier is applied.
template <typename real_t> __host__ __device__ constexpr real_t hn_ekin_max_qb() {
  return real_t(100) * units::MeV<real_t>();
}

/// G4HadronNucleonXsc::CalcMandelstamS - inline in the header.
template <typename real_t>
__host__ __device__ inline real_t hn_mandelstam_s(real_t ekin1, real_t mass1, real_t mass2) {
  return mass1 * mass1 + mass2 * mass2 + real_t(2) * mass2 * (ekin1 + mass1);
}

/// G4HadronNucleonXsc::CoulombBarrier.
///
/// Not the same function as G4NuclearRadii::CoulombFactor(particle, nucleon, ekin), though it
/// computes the same quantity: this one has its own radius ladder (0.5 fm default, 0.895 for
/// the proton, 0.663 for pi+, 0.340 for K+ - keyed on *pointer identity with the positive
/// particle*, so pi- and K- get 0.5 fm here and 0.663 / 0.340 there) and divides by
/// 2*(pR+tR) with the full fine-structure constant rather than by (pR+tR) with half of it.
/// HadronNucleonXscPDG calls this one; HadronNucleonXscNS and KaonNucleonXscVG call the other.
template <typename real_t>
__host__ __device__ inline real_t hn_coulomb_barrier(const Projectile<real_t>& p,
                                                     const Projectile<real_t>& nucleon,
                                                     real_t ekin) {
  const real_t tR = real_t(0.895) * fermi<real_t>();
  real_t pR = real_t(0.5) * fermi<real_t>();
  if (p.pdg == pdg::kProton) {
    pR = real_t(0.895) * fermi<real_t>();
  } else if (p.pdg == pdg::kPiPlus) {
    pR = real_t(0.663) * fermi<real_t>();
  } else if (p.pdg == pdg::kKaonPlus) {
    pR = real_t(0.340) * fermi<real_t>();
  }
  const real_t pZ = p.charge;
  const real_t tZ = nucleon.charge;
  const real_t pM = p.mass;
  const real_t tM = nucleon.mass;
  const real_t pElab = ekin + pM;
  const real_t totEcm = sqrt(pM * pM + tM * tM + real_t(2.) * pElab * tM);
  const real_t totTcm = totEcm - pM - tM;
  const real_t bC = units::fine_structure_const<real_t>() * units::hbarc<real_t>() * pZ * tZ
                    / (real_t(2.) * (pR + tR));
  return (totTcm > bC) ? real_t(1.) - bC / totTcm : real_t(0.0);
}

/// G4HadronNucleonXsc::HadronNucleonXscPDG - the PDG 2017 Regge fit,
/// sigma = del*(H*L^2 + P) + R1*exp(-eta1*L) + R2*exp(-eta2*L) with L = log(s/x^2).
///
/// The inelastic split is a flat 0.75 of the total, which is the model and not an assumption.
template <typename real_t>
__host__ __device__ inline HadXs<real_t> hn_xsc_pdg(const Projectile<real_t>& p,
                                                    const Projectile<real_t>& nucleon,
                                                    real_t ekin) {
  constexpr real_t M = real_t(2.1206);  // GeV
  constexpr real_t eta1 = real_t(0.4473);
  constexpr real_t eta2 = real_t(0.5486);
  constexpr real_t H = real_t(0.272);

  const int pdgc = p.pdg;
  // A photon is given the rho mass here, not zero.
  const real_t mass1 = (pdgc == pdg::kGamma) ? real_t(770.) : p.mass;
  const real_t mass2 = nucleon.mass;

  const real_t sMand = hn_mandelstam_s<real_t>(ekin, mass1, mass2) * hn_inv_gev2<real_t>();
  const real_t x = (mass1 + mass2) * hn_inv_gev<real_t>() + M;
  const real_t blog = log(sMand / (x * x));

  real_t P = real_t(0.0), R1 = real_t(0.0), R2 = real_t(0.0), del = real_t(1.0);

  const bool proton = (nucleon.pdg == pdg::kProton);
  const bool neutron = (nucleon.pdg == pdg::kNeutron);

  if (pdgc == pdg::kNeutron) {
    if (proton) {
      P = real_t(34.71); R1 = real_t(12.52); R2 = real_t(-6.66);
    } else {
      P = real_t(34.41); R1 = real_t(13.07); R2 = real_t(-7.394);
    }
  } else if (pdgc == pdg::kProton) {
    if (neutron) {
      P = real_t(34.71); R1 = real_t(12.52); R2 = real_t(-6.66);
    } else {
      P = real_t(34.41); R1 = real_t(13.07); R2 = real_t(-7.394);
    }
  } else if (pdgc == pdg::kAntiProton) {
    if (neutron) {
      P = real_t(34.71); R1 = real_t(12.52); R2 = real_t(6.66);
    } else {
      P = real_t(34.41); R1 = real_t(13.07); R2 = real_t(7.394);
    }
  } else if (pdgc == pdg::kAntiNeutron) {
    if (proton) {
      P = real_t(34.71); R1 = real_t(12.52); R2 = real_t(6.66);
    } else {
      P = real_t(34.41); R1 = real_t(13.07); R2 = real_t(7.394);
    }
  } else if (pdgc == pdg::kPiPlus) {
    P = real_t(18.75); R1 = real_t(9.56); R2 = real_t(-1.767);
  } else if (pdgc == pdg::kPiMinus) {
    P = real_t(18.75); R1 = real_t(9.56); R2 = real_t(1.767);
  } else if (pdgc == pdg::kKaonPlus) {
    if (proton) {
      P = real_t(16.36); R1 = real_t(4.29); R2 = real_t(-3.408);
    } else {
      P = real_t(16.31); R1 = real_t(3.7); R2 = real_t(-1.826);
    }
  } else if (pdgc == pdg::kKaonMinus) {
    if (proton) {
      P = real_t(16.36); R1 = real_t(4.29); R2 = real_t(3.408);
    } else {
      P = real_t(16.31); R1 = real_t(3.7); R2 = real_t(1.826);
    }
  } else if (pdgc == pdg::kKaonZeroShort || pdgc == pdg::kKaonZeroLong) {
    P = real_t(16.36); R1 = real_t(2.5); R2 = real_t(0.);
  } else if (pdgc == pdg::kSigmaMinus) {
    // Dead in practice - the dispatcher sends |pdg| > 3000 to HyperonNucleonXscNS first.
    P = real_t(34.7); R1 = real_t(-46.); R2 = real_t(48.);
  } else if (pdgc == pdg::kGamma) {
    del = real_t(0.003063);
    P = real_t(34.71) * del;
    R1 = neutron ? real_t(0.0231) : real_t(0.0139);
    R2 = real_t(0.);
  } else {
    // "as proton ???" in the source, comment and all.
    if (neutron) {
      P = real_t(34.71); R1 = real_t(12.52); R2 = real_t(-6.66);
    } else {
      P = real_t(34.41); R1 = real_t(13.07); R2 = real_t(-7.394);
    }
  }

  HadXs<real_t> r;
  r.total = millibarn<real_t>()
            * (del * (H * blog * blog + P) + R1 * exp(-eta1 * blog) + R2 * exp(-eta2 * blog));
  r.inelastic = real_t(0.75) * r.total;
  r.elastic = r.total - r.inelastic;

  if (proton && p.charge > real_t(0.) && ekin < hn_ekin_max_qb<real_t>()) {
    const real_t cB = hn_coulomb_barrier<real_t>(p, nucleon, ekin);
    r.total *= cB;
    r.elastic *= cB;
    r.inelastic *= cB;
  }
  return r;
}

/// G4HadronNucleonXsc::HadronNucleonXscNS - N. Starkov's parameterisation of the IHEP
/// hadron-nucleon database. Everything is in millibarn until the two multiplications at the
/// end, and `fTotalXsc` starts at zero below 10 GeV/c and at the PDG total above it.
template <typename real_t>
__host__ __device__ inline HadXs<real_t> hn_xsc_ns(const Projectile<real_t>& p,
                                                   const Projectile<real_t>& nucleon,
                                                   real_t ekin0) {
  const real_t ekin = (ekin0 > hn_ekin_min<real_t>()) ? ekin0 : hn_ekin_min<real_t>();
  const int pdgc = p.pdg;

  if (pdgc == pdg::kAntiProton || pdgc == pdg::kAntiNeutron) {
    return hn_xsc_pdg<real_t>(p, nucleon, ekin);
  }

  const real_t pM = p.mass;
  const real_t tM = nucleon.mass;
  real_t pE = ekin + pM;
  real_t pLab = sqrt(ekin * (ekin + real_t(2) * pM));

  const real_t sMand = hn_mandelstam_s<real_t>(ekin, pM, tM) * hn_inv_gev2<real_t>();

  pLab *= hn_inv_gev<real_t>();
  pE *= hn_inv_gev<real_t>();

  HadXs<real_t> r;
  if (pLab >= real_t(10.)) {
    r.total = hn_xsc_pdg<real_t>(p, nucleon, ekin).total / millibarn<real_t>();
  } else {
    r.total = real_t(0.0);
  }
  r.elastic = real_t(0.0);
  const real_t logP = log(pLab);

  const bool proton = (nucleon.pdg == pdg::kProton);
  const bool neutron = (nucleon.pdg == pdg::kNeutron);

  if (pdgc == pdg::kNeutron) {
    if (pLab >= real_t(373.)) {
      // See the file header: the 1.65 is inside the outer log here and outside it in the
      // proton branch below. Verbatim.
      r.elastic = real_t(6.5)
                  + real_t(0.308) * exp(log(log(sMand / real_t(400.)) * real_t(1.65)))
                  + real_t(9.19) * exp(-log(sMand) * real_t(0.458));
    } else if (pLab >= real_t(100.)) {
      r.elastic = real_t(5.53)
                  + real_t(0.308) * exp(log(log(sMand / real_t(28.9))) * real_t(1.1))
                  + real_t(9.19) * exp(-log(sMand) * real_t(0.458));
    } else if (pLab >= real_t(10.)) {
      r.elastic = real_t(6)
                  + real_t(20) / ((logP - real_t(0.182)) * (logP - real_t(0.182)) + real_t(1.0));
    } else {  // pLab < 10 GeV/c
      if (neutron) {  // nn to be pp
        const real_t x = log(pLab / real_t(0.73));
        if (pLab < real_t(0.4)) {
          r.total = real_t(23) + real_t(50) * sqrt(g4pow_pow_n<real_t>(-x, 7));
          r.elastic = r.total;
        } else if (pLab < real_t(0.73)) {
          r.total = real_t(23) + real_t(50) * sqrt(g4pow_pow_n<real_t>(-x, 7));
          r.elastic = r.total;
        } else if (pLab < real_t(1.05)) {
          r.total = real_t(23) + real_t(40) * x * x;
          r.elastic = real_t(23) + real_t(20) * x * x;
        } else {  // 1.05 - 10 GeV/c
          r.total = real_t(39.0)
                    + real_t(75) * (pLab - real_t(1.2))
                          / (g4pow_pow_n<real_t>(pLab, 3) + real_t(0.15));
          r.elastic =
              real_t(6)
              + real_t(20) / ((logP - real_t(0.182)) * (logP - real_t(0.182)) + real_t(1.0));
        }
      }
      if (proton) {  // pn to be np
        if (pLab < real_t(0.02)) {
          r.total = real_t(4100)
                    + real_t(30) * exp(log(log(real_t(1.3) / pLab)) * real_t(3.6));
          r.elastic = r.total;
        } else if (pLab < real_t(0.8)) {
          r.total = real_t(33)
                    + real_t(30) * g4pow_pow_n<real_t>(log(pLab / real_t(1.3)), 4);
          r.elastic = r.total;
        } else if (pLab < real_t(1.4)) {
          r.total = real_t(33)
                    + real_t(30) * g4pow_pow_n<real_t>(log(pLab / real_t(0.95)), 2);
          const real_t x = log(real_t(0.511) / pLab);
          r.elastic = real_t(6) + real_t(52) / (x * x + real_t(1.6));
        } else {  // 1.4 < pLab < 10
          r.total = real_t(33.3)
                    + real_t(20.8) * (pLab * pLab - real_t(1.35))
                          / (sqrt(g4pow_pow_n<real_t>(pLab, 5)) + real_t(0.95));
          r.elastic =
              real_t(6)
              + real_t(20) / ((logP - real_t(0.182)) * (logP - real_t(0.182)) + real_t(1.0));
        }
      }
    }
  } else if (pdgc == pdg::kProton) {
    if (pLab >= real_t(373.)) {  // pdg due to TOTEM data
      r.elastic = real_t(6.5)
                  + real_t(0.308) * exp(log(log(sMand / real_t(400.))) * real_t(1.65))
                  + real_t(9.19) * exp(-log(sMand) * real_t(0.458));
    } else if (pLab >= real_t(100.)) {
      r.elastic = real_t(5.53)
                  + real_t(0.308) * exp(log(log(sMand / real_t(28.9))) * real_t(1.1))
                  + real_t(9.19) * exp(-log(sMand) * real_t(0.458));
    } else if (pLab >= real_t(10.)) {
      r.elastic =
          real_t(6.)
          + real_t(20.) / ((logP - real_t(0.182)) * (logP - real_t(0.182)) + real_t(1.0));
    } else {
      if (proton) {  // pp
        if (pLab < real_t(0.73)) {
          r.total = real_t(23)
                    + real_t(50)
                          * sqrt(g4pow_pow_n<real_t>(log(real_t(0.73) / pLab), 7));
          r.elastic = r.total;
        } else if (pLab < real_t(1.05)) {
          const real_t x = log(pLab / real_t(0.73));
          r.total = real_t(23) + real_t(40) * x * x;
          r.elastic = real_t(23) + real_t(20) * x * x;
        } else {  // 1.05 - 10 GeV/c
          r.total = real_t(39.0)
                    + real_t(75) * (pLab - real_t(1.2))
                          / (g4pow_pow_n<real_t>(pLab, 3) + real_t(0.15));
          r.elastic =
              real_t(6.)
              + real_t(20.) / ((logP - real_t(0.182)) * (logP - real_t(0.182)) + real_t(1.0));
        }
      } else if (neutron) {  // pn to be np
        if (pLab < real_t(0.02)) {
          r.total = real_t(4100)
                    + real_t(30) * exp(log(log(real_t(1.3) / pLab)) * real_t(3.6));
          r.elastic = r.total;
        } else if (pLab < real_t(0.8)) {
          r.total = real_t(33)
                    + real_t(30) * g4pow_pow_n<real_t>(log(pLab / real_t(1.3)), 4);
          r.elastic = r.total;
        } else if (pLab < real_t(1.4)) {
          const real_t x1 = log(pLab / real_t(0.95));
          const real_t x2 = log(real_t(0.511) / pLab);
          r.total = real_t(33) + real_t(30) * x1 * x1;
          r.elastic = real_t(6) + real_t(52) / (x2 * x2 + real_t(1.6));
        } else {  // 1.4 < pLab < 10
          r.total = real_t(33.3)
                    + real_t(20.8) * (pLab * pLab - real_t(1.35))
                          / (sqrt(g4pow_pow_n<real_t>(pLab, 5)) + real_t(0.95));
          r.elastic =
              real_t(6.)
              + real_t(20.) / ((logP - real_t(0.182)) * (logP - real_t(0.182)) + real_t(1.0));
        }
      }
    }
  } else if ((pdgc == pdg::kPiPlus && proton) || (pdgc == pdg::kPiMinus && neutron)) {
    // pi+ p; pi- n
    if (pLab < real_t(0.28)) {
      r.total = real_t(10.)
                / ((logP + real_t(1.273)) * (logP + real_t(1.273)) + real_t(0.05));
      r.elastic = r.total;
    } else if (pLab < real_t(0.68)) {
      r.total = real_t(14.)
                / ((logP + real_t(1.273)) * (logP + real_t(1.273)) + real_t(0.07));
      r.elastic = r.total;
    } else if (pLab < real_t(0.85)) {
      const real_t x = log(pLab / real_t(0.77));
      r.total = real_t(88.) * x * x + real_t(14.9);
      r.elastic = r.total * exp(real_t(-3.) * (pLab - real_t(0.68)));
    } else if (pLab < real_t(1.15)) {
      const real_t x = log(pLab / real_t(0.77));
      r.total = real_t(88.) * x * x + real_t(14.9);
      r.elastic = real_t(6.0)
                  + real_t(1.4) / ((pLab - real_t(1.4)) * (pLab - real_t(1.4)) + real_t(0.1));
    } else if (pLab < real_t(1.4)) {  // ns original
      const real_t Ex1 =
          real_t(3.2)
          * exp(-(pLab - real_t(2.55)) * (pLab - real_t(2.55)) / real_t(0.55) / real_t(0.55));
      const real_t Ex2 =
          real_t(12)
          * exp(-(pLab - real_t(1.47)) * (pLab - real_t(1.47)) / real_t(0.225) / real_t(0.225));
      r.total = Ex1 + Ex2 + real_t(27.5);
      r.elastic = real_t(6.0)
                  + real_t(1.4) / ((pLab - real_t(1.4)) * (pLab - real_t(1.4)) + real_t(0.1));
    } else if (pLab < real_t(2.0)) {  // ns original
      const real_t Ex1 =
          real_t(3.2)
          * exp(-(pLab - real_t(2.55)) * (pLab - real_t(2.55)) / real_t(0.55) / real_t(0.55));
      const real_t Ex2 =
          real_t(12)
          * exp(-(pLab - real_t(1.47)) * (pLab - real_t(1.47)) / real_t(0.225) / real_t(0.225));
      r.total = Ex1 + Ex2 + real_t(27.5);
      r.elastic = real_t(3.0)
                  + real_t(1.36)
                        / ((logP - real_t(0.336)) * (logP - real_t(0.336)) + real_t(0.08));
    } else if (pLab < real_t(3.5)) {  // ns original
      const real_t Ex1 =
          real_t(3.2)
          * exp(-(pLab - real_t(2.55)) * (pLab - real_t(2.55)) / real_t(0.55) / real_t(0.55));
      const real_t Ex2 =
          real_t(12)
          * exp(-(pLab - real_t(1.47)) * (pLab - real_t(1.47)) / real_t(0.225) / real_t(0.225));
      r.total = Ex1 + Ex2 + real_t(27.5);
      r.elastic = real_t(3.0)
                  + real_t(6.20)
                        / ((logP - real_t(0.336)) * (logP - real_t(0.336)) + real_t(0.8));
    } else if (pLab < real_t(10.)) {  // my
      r.total = real_t(10.6) + real_t(2.) * log(pE)
                + real_t(25) * exp(-log(pE) * real_t(0.43));
      r.elastic = real_t(3.0)
                  + real_t(6.20)
                        / ((logP - real_t(0.336)) * (logP - real_t(0.336)) + real_t(0.8));
    } else {  // pLab > 10, my
      r.elastic = real_t(3.0)
                  + real_t(6.20)
                        / ((logP - real_t(0.336)) * (logP - real_t(0.336)) + real_t(0.8));
    }
  } else if ((pdgc == pdg::kPiPlus && neutron) || (pdgc == pdg::kPiMinus && proton)) {
    // pi+ n; pi- p
    if (pLab < real_t(0.28)) {
      r.total = real_t(0.288)
                / ((pLab - real_t(0.28)) * (pLab - real_t(0.28)) + real_t(0.004));
      r.elastic = real_t(1.8)
                  / ((logP + real_t(1.273)) * (logP + real_t(1.273)) + real_t(0.07));
    } else if (pLab < real_t(0.395676)) {  // first peak
      r.total = real_t(0.648)
                / ((pLab - real_t(0.28)) * (pLab - real_t(0.28)) + real_t(0.009));
      r.elastic = real_t(0.257)
                  / ((pLab - real_t(0.28)) * (pLab - real_t(0.28)) + real_t(0.01));
    } else if (pLab < real_t(0.5)) {
      const real_t y = log(pLab / real_t(0.48));
      r.total = real_t(26) + real_t(110) * y * y;
      r.elastic = real_t(0.37) * r.total;
    } else if (pLab < real_t(0.65)) {
      const real_t x = log(pLab / real_t(0.48));
      r.total = real_t(26.) + real_t(110.) * x * x;
      r.elastic = real_t(0.95)
                  / ((pLab - real_t(0.72)) * (pLab - real_t(0.72)) + real_t(0.049));
    } else if (pLab < real_t(0.72)) {
      r.total = real_t(36.1)
                + real_t(10)
                      * exp(-(pLab - real_t(0.72)) * (pLab - real_t(0.72)) / real_t(0.06)
                            / real_t(0.06))
                + real_t(24)
                      * exp(-(pLab - real_t(1.015)) * (pLab - real_t(1.015)) / real_t(0.075)
                            / real_t(0.075));
      r.elastic = real_t(0.95)
                  / ((pLab - real_t(0.72)) * (pLab - real_t(0.72)) + real_t(0.049));
    } else if (pLab < real_t(0.88)) {
      r.total = real_t(36.1)
                + real_t(10.)
                      * exp(-(pLab - real_t(0.72)) * (pLab - real_t(0.72)) / real_t(0.06)
                            / real_t(0.06))
                + real_t(24)
                      * exp(-(pLab - real_t(1.015)) * (pLab - real_t(1.015)) / real_t(0.075)
                            / real_t(0.075));
      r.elastic = real_t(0.95)
                  / ((pLab - real_t(0.72)) * (pLab - real_t(0.72)) + real_t(0.049));
    } else if (pLab < real_t(1.03)) {
      r.total = real_t(36.1)
                + real_t(10.)
                      * exp(-(pLab - real_t(0.72)) * (pLab - real_t(0.72)) / real_t(0.06)
                            / real_t(0.06))
                + real_t(24)
                      * exp(-(pLab - real_t(1.015)) * (pLab - real_t(1.015)) / real_t(0.075)
                            / real_t(0.075));
      r.elastic = real_t(2.0)
                  + real_t(0.4)
                        / ((pLab - real_t(1.03)) * (pLab - real_t(1.03)) + real_t(0.016));
    } else if (pLab < real_t(1.15)) {
      r.total = real_t(36.1)
                + real_t(10.)
                      * exp(-(pLab - real_t(0.72)) * (pLab - real_t(0.72)) / real_t(0.06)
                            / real_t(0.06))
                + real_t(24)
                      * exp(-(pLab - real_t(1.015)) * (pLab - real_t(1.015)) / real_t(0.075)
                            / real_t(0.075));
      r.elastic = real_t(2.0)
                  + real_t(0.4)
                        / ((pLab - real_t(1.03)) * (pLab - real_t(1.03)) + real_t(0.016));
    } else if (pLab < real_t(1.3)) {
      r.total = real_t(36.1)
                + real_t(10.)
                      * exp(-(pLab - real_t(0.72)) * (pLab - real_t(0.72)) / real_t(0.06)
                            / real_t(0.06))
                + real_t(24)
                      * exp(-(pLab - real_t(1.015)) * (pLab - real_t(1.015)) / real_t(0.075)
                            / real_t(0.075));
      r.elastic = real_t(3.) + real_t(13.) / pLab;
    } else if (pLab < real_t(10.)) {  // < 3.0 in the ns original
      r.total = real_t(36.1) + real_t(0.079) - real_t(4.313) * logP
                + real_t(3)
                      * exp(-(pLab - real_t(2.1)) * (pLab - real_t(2.1)) / real_t(0.4)
                            / real_t(0.4))
                + real_t(1.5)
                      * exp(-(pLab - real_t(1.4)) * (pLab - real_t(1.4)) / real_t(0.12)
                            / real_t(0.12));
      r.elastic = real_t(3.) + real_t(13.) / pLab;
    } else {  // mb
      r.elastic = real_t(3.) + real_t(13.) / pLab;
    }
  } else if (pdgc == pdg::kKaonMinus && proton) {  // K-p
    if (pLab < hn_p_min<real_t>()) {
      const real_t psp = pLab * sqrt(pLab);
      r.elastic = real_t(5.2) / psp;
      r.total = real_t(14.) / psp;
    } else if (pLab > hn_p_max<real_t>()) {
      const real_t ld = logP - hn_min_log_p<real_t>();
      const real_t ld2 = ld * ld;
      r.elastic = hn_cof_log_e<real_t>() * ld2 + real_t(2.23);
      r.total = real_t(1.1) * hn_cof_log_t<real_t>() * ld2 + real_t(19.7);
    } else {
      const real_t ld = logP - hn_min_log_p<real_t>();
      const real_t ld2 = ld * ld;
      const real_t sp = sqrt(pLab);
      const real_t psp = pLab * sp;
      const real_t p2 = pLab * pLab;
      const real_t p4 = p2 * p2;
      const real_t lh = pLab - real_t(1.01);
      const real_t hd = lh * lh + real_t(.011);
      const real_t lm = pLab - real_t(.39);
      const real_t md = lm * lm + real_t(.000356);
      const real_t lh1 = pLab - real_t(0.78);
      const real_t hd1 = lh1 * lh1 + real_t(.00166);
      const real_t lh2 = pLab - real_t(1.63);
      const real_t hd2 = lh2 * lh2 + real_t(.007);
      r.elastic = real_t(5.2) / psp
                  + (real_t(1.1) * hn_cof_log_e<real_t>() * ld2 + real_t(2.23))
                        / (real_t(1.) - real_t(.7) / sp + real_t(.075) / p4)
                  + real_t(.004) / md + real_t(0.005) / hd1 + real_t(0.01) / hd2
                  + real_t(.15) / hd;
      r.total = real_t(14.) / psp
                + (real_t(1.1) * hn_cof_log_t<real_t>() * ld2 + real_t(19.5))
                      / (real_t(1.) - real_t(.21) / sp + real_t(.52) / p4)
                + real_t(.006) / md + real_t(0.01) / hd1 + real_t(0.02) / hd2
                + real_t(.20) / hd;
    }
  } else if (pdgc == pdg::kKaonMinus && neutron) {  // K-n
    if (pLab > hn_p_max<real_t>()) {
      const real_t ld = logP - hn_min_log_p<real_t>();
      const real_t ld2 = ld * ld;
      r.elastic = hn_cof_log_e<real_t>() * ld2 + real_t(2.23);
      r.total = real_t(1.1) * hn_cof_log_t<real_t>() * ld2 + real_t(19.7);
    } else {
      const real_t lh = pLab - real_t(0.98);
      const real_t hd = lh * lh + real_t(.021);
      const real_t sqrLogPlab = logP * logP;
      r.elastic = real_t(5.0) + real_t(8.1) * exp(-logP * real_t(1.8))
                  + real_t(0.16) * sqrLogPlab - real_t(1.3) * logP + real_t(.15) / hd;
      r.total = real_t(25.2) + real_t(0.38) * sqrLogPlab - real_t(2.9) * logP
                + real_t(0.30) / hd;
    }
  } else if (pdgc == pdg::kKaonPlus && proton) {  // K+p
    if (pLab < real_t(0.631)) {  // VI: modified low-energy part
      r.elastic = r.total = real_t(12.03);
    } else if (pLab > hn_p_max<real_t>()) {
      const real_t ld = logP - hn_min_log_p<real_t>();
      const real_t ld2 = ld * ld;
      r.elastic = hn_cof_log_e<real_t>() * ld2 + real_t(2.23);
      r.total = hn_cof_log_t<real_t>() * ld2 + real_t(19.2);
    } else {
      const real_t ld = logP - hn_min_log_p<real_t>();
      const real_t ld2 = ld * ld;
      const real_t lr = pLab - real_t(.38);
      const real_t LE = real_t(.7) / (lr * lr + real_t(.076));
      const real_t sp = sqrt(pLab);
      const real_t p2 = pLab * pLab;
      const real_t p4 = p2 * p2;
      r.elastic = LE
                  + (hn_cof_log_e<real_t>() * ld2 + real_t(2.23))
                        / (real_t(1.) - real_t(.7) / sp + real_t(.1) / p4)
                  + real_t(2.)
                        / ((pLab - real_t(0.8)) * (pLab - real_t(0.8)) + real_t(0.652));
      r.total = LE
                + (hn_cof_log_t<real_t>() * ld2 + real_t(19.5))
                      / (real_t(1.) + real_t(.46) / sp + real_t(1.6) / p4)
                + real_t(2.6) / ((pLab - real_t(1.)) * (pLab - real_t(1.)) + real_t(0.392));
    }
  } else if (pdgc == pdg::kKaonPlus && neutron) {  // K+n
    if (pLab < hn_p_min<real_t>()) {
      const real_t lm = pLab - real_t(0.94);
      const real_t md = lm * lm + real_t(.392);
      r.elastic = real_t(2.) / md;
      r.total = real_t(4.6) / md;
    } else if (pLab > hn_p_max<real_t>()) {
      const real_t ld = logP - hn_min_log_p<real_t>();
      const real_t ld2 = ld * ld;
      r.elastic = hn_cof_log_e<real_t>() * ld2 + real_t(2.23);
      r.total = hn_cof_log_t<real_t>() * ld2 + real_t(19.2);
    } else {
      const real_t ld = logP - hn_min_log_p<real_t>();
      const real_t ld2 = ld * ld;
      const real_t sp = sqrt(pLab);
      const real_t p2 = pLab * pLab;
      const real_t p4 = p2 * p2;
      const real_t lm = pLab - real_t(0.94);
      const real_t md = lm * lm + real_t(.392);
      r.elastic = (hn_cof_log_e<real_t>() * ld2 + real_t(2.23))
                      / (real_t(1.) - real_t(.7) / sp + real_t(.1) / p4)
                  + real_t(2.) / md;
      r.total = (hn_cof_log_t<real_t>() * ld2 + real_t(19.5))
                    / (real_t(1.) + real_t(.46) / sp + real_t(1.6) / p4)
                + real_t(4.6) / md;
    }
  }

  r.total *= millibarn<real_t>();
  r.elastic *= millibarn<real_t>();
  r.elastic = (r.elastic < r.total) ? r.elastic : r.total;

  if (proton && p.charge > real_t(0.) && ekin < hn_ekin_max_qb<real_t>()) {
    const real_t cB = nr_coulomb_factor<real_t>(p, nucleon, ekin);
    r.total *= cB;
    r.elastic *= cB;
  }
  r.inelastic = (r.total - r.elastic > real_t(0.0)) ? r.total - r.elastic : real_t(0.0);
  return r;
}

/// G4HadronNucleonXsc::KaonNucleonXscVG - the "smoothed NS" kaon fit, which is NOT the kaon
/// part of HadronNucleonXscNS. Both are live: NS through KaonNucleonXscNS for hydrogen, VG
/// through KaonNucleonXscGG for everything heavier.
///
/// TWO of the four channels differ, not three. This said three, and the count matters because
/// it is the sort of thing a later reader uses to decide the two functions can be merged. The
/// K+p and K+n arms of the two are BYTE-IDENTICAL in Geant4 (G4HadronNucleonXsc.cc:693-751
/// against :909-967, zero lines changed). The differences are all in the K- arms:
///
///   K-p (.cc:631-672 vs :859-888)  VG drops the three small resonance peaks, uses 0.60/hd
///                                  where NS uses 0.20/hd, AND drops the 1.1 from the elastic
///                                  log coefficient - NS has (1.1*cofLogE*ld2 + 2.23) and VG
///                                  has (cofLogE*ld2 + 2.23). Three differences, not two; the
///                                  1.1 was missing from this list and the code has it.
///   K-n (.cc:673-692 vs :889-908)  VG uses .045 and 0.60 where NS uses .021 and 0.30.
///
/// Note its Coulomb-barrier test has no upper energy limit, unlike NS's `ekin < 100 MeV`.
template <typename real_t>
__host__ __device__ inline HadXs<real_t> hn_kaon_xsc_vg(const Projectile<real_t>& p,
                                                        const Projectile<real_t>& nucleon,
                                                        real_t ekin) {
  const real_t pM = p.mass;
  real_t pLab = sqrt(ekin * (ekin + real_t(2) * pM));
  pLab *= hn_inv_gev<real_t>();
  const real_t logP = log(pLab);

  HadXs<real_t> r;
  r.total = real_t(0.0);

  const bool proton = (nucleon.pdg == pdg::kProton);
  const bool neutron = (nucleon.pdg == pdg::kNeutron);

  if (p.pdg == pdg::kKaonMinus && proton) {  // K-p
    if (pLab < hn_p_min<real_t>()) {
      const real_t psp = pLab * sqrt(pLab);
      r.elastic = real_t(5.2) / psp;
      r.total = real_t(14.) / psp;
    } else if (pLab > hn_p_max<real_t>()) {
      const real_t ld = logP - hn_min_log_p<real_t>();
      const real_t ld2 = ld * ld;
      r.elastic = hn_cof_log_e<real_t>() * ld2 + real_t(2.23);
      r.total = real_t(1.1) * hn_cof_log_t<real_t>() * ld2 + real_t(19.7);
    } else {
      const real_t ld = logP - hn_min_log_p<real_t>();
      const real_t ld2 = ld * ld;
      const real_t sp = sqrt(pLab);
      const real_t psp = pLab * sp;
      const real_t p2 = pLab * pLab;
      const real_t p4 = p2 * p2;
      const real_t lh = pLab - real_t(1.01);
      const real_t hd = lh * lh + real_t(.011);
      r.elastic = real_t(5.2) / psp
                  + (hn_cof_log_e<real_t>() * ld2 + real_t(2.23))
                        / (real_t(1.) - real_t(.7) / sp + real_t(.075) / p4)
                  + real_t(.15) / hd;
      r.total = real_t(14.) / psp
                + (real_t(1.1) * hn_cof_log_t<real_t>() * ld2 + real_t(19.5))
                      / (real_t(1.) - real_t(.21) / sp + real_t(.52) / p4)
                + real_t(.60) / hd;
    }
  } else if (p.pdg == pdg::kKaonMinus && neutron) {  // K-n
    if (pLab > hn_p_max<real_t>()) {
      const real_t ld = logP - hn_min_log_p<real_t>();
      const real_t ld2 = ld * ld;
      r.elastic = hn_cof_log_e<real_t>() * ld2 + real_t(2.23);
      r.total = real_t(1.1) * hn_cof_log_t<real_t>() * ld2 + real_t(19.7);
    } else {
      const real_t lh = pLab - real_t(0.98);
      const real_t hd = lh * lh + real_t(.045);  // vg version
      const real_t sqrLogPlab = logP * logP;
      r.elastic = real_t(5.0) + real_t(8.1) * exp(-logP * real_t(1.8))
                  + real_t(0.16) * sqrLogPlab - real_t(1.3) * logP + real_t(.15) / hd;
      r.total = real_t(25.2) + real_t(0.38) * sqrLogPlab - real_t(2.9) * logP
                + real_t(0.60) / hd;  // vg version
    }
  } else if (p.pdg == pdg::kKaonPlus && proton) {  // K+p
    if (pLab < real_t(0.631)) {
      r.elastic = r.total = real_t(12.03);
    } else if (pLab > hn_p_max<real_t>()) {
      const real_t ld = logP - hn_min_log_p<real_t>();
      const real_t ld2 = ld * ld;
      r.elastic = hn_cof_log_e<real_t>() * ld2 + real_t(2.23);
      r.total = hn_cof_log_t<real_t>() * ld2 + real_t(19.2);
    } else {
      const real_t ld = logP - hn_min_log_p<real_t>();
      const real_t ld2 = ld * ld;
      const real_t lr = pLab - real_t(.38);
      const real_t LE = real_t(.7) / (lr * lr + real_t(.076));
      const real_t sp = sqrt(pLab);
      const real_t p2 = pLab * pLab;
      const real_t p4 = p2 * p2;
      r.elastic = LE
                  + (hn_cof_log_e<real_t>() * ld2 + real_t(2.23))
                        / (real_t(1.) - real_t(.7) / sp + real_t(.1) / p4)
                  + real_t(2.)
                        / ((pLab - real_t(0.8)) * (pLab - real_t(0.8)) + real_t(0.652));
      r.total = LE
                + (hn_cof_log_t<real_t>() * ld2 + real_t(19.5))
                      / (real_t(1.) + real_t(.46) / sp + real_t(1.6) / p4)
                + real_t(2.6) / ((pLab - real_t(1.)) * (pLab - real_t(1.)) + real_t(0.392));
    }
  } else if (p.pdg == pdg::kKaonPlus && neutron) {  // K+n
    if (pLab < hn_p_min<real_t>()) {
      const real_t lm = pLab - real_t(0.94);
      const real_t md = lm * lm + real_t(.392);
      r.elastic = real_t(2.) / md;
      r.total = real_t(4.6) / md;
    } else if (pLab > hn_p_max<real_t>()) {
      const real_t ld = logP - hn_min_log_p<real_t>();
      const real_t ld2 = ld * ld;
      r.elastic = hn_cof_log_e<real_t>() * ld2 + real_t(2.23);
      r.total = hn_cof_log_t<real_t>() * ld2 + real_t(19.2);
    } else {
      const real_t ld = logP - hn_min_log_p<real_t>();
      const real_t ld2 = ld * ld;
      const real_t sp = sqrt(pLab);
      const real_t p2 = pLab * pLab;
      const real_t p4 = p2 * p2;
      const real_t lm = pLab - real_t(0.94);
      const real_t md = lm * lm + real_t(.392);
      r.elastic = (hn_cof_log_e<real_t>() * ld2 + real_t(2.23))
                      / (real_t(1.) - real_t(.7) / sp + real_t(.1) / p4)
                  + real_t(2.) / md;
      r.total = (hn_cof_log_t<real_t>() * ld2 + real_t(19.5))
                    / (real_t(1.) + real_t(.46) / sp + real_t(1.6) / p4)
                + real_t(4.6) / md;
    }
  }

  r.total *= millibarn<real_t>();
  r.elastic *= millibarn<real_t>();

  if (proton && p.charge > real_t(0.)) {
    const real_t cB = nr_coulomb_factor<real_t>(p, nucleon, ekin);
    r.total *= cB;
    r.elastic *= cB;
  }
  r.elastic = (r.elastic < r.total) ? r.elastic : r.total;
  r.inelastic = (r.total - r.elastic > real_t(0.0)) ? r.total - r.elastic : real_t(0.0);
  return r;
}

/// G4HadronNucleonXsc::KaonNucleonXscGG - VG for a charged kaon, the mean of the K- and K+ VG
/// results for a neutral one.
template <typename real_t>
__host__ __device__ inline HadXs<real_t> hn_kaon_xsc_gg(const Projectile<real_t>& p,
                                                        const Projectile<real_t>& nucleon,
                                                        real_t ekin) {
  HadXs<real_t> r;
  if (p.pdg == pdg::kKaonMinus || p.pdg == pdg::kKaonPlus) {
    r = hn_kaon_xsc_vg<real_t>(p, nucleon, ekin);
  } else if (p.pdg == pdg::kKaonZeroShort || p.pdg == pdg::kKaonZeroLong) {
    const HadXs<real_t> m = hn_kaon_xsc_vg<real_t>(kaon_minus<real_t>(), nucleon, ekin);
    const HadXs<real_t> pl = hn_kaon_xsc_vg<real_t>(kaon_plus<real_t>(), nucleon, ekin);
    r.total = (m.total + pl.total) * real_t(0.5);
    r.elastic = (m.elastic + pl.elastic) * real_t(0.5);
    r.inelastic = (m.inelastic + pl.inelastic) * real_t(0.5);
  }
  return r;
}

/// G4HadronNucleonXsc::KaonNucleonXscNS - HadronNucleonXscNS for a charged kaon; for a neutral
/// one, the K-/K+ mean, evaluated at 100 MeV and scaled by sqrt(100 MeV / ekin) below 100 MeV.
template <typename real_t>
__host__ __device__ inline HadXs<real_t> hn_kaon_xsc_ns(const Projectile<real_t>& p,
                                                        const Projectile<real_t>& nucleon,
                                                        real_t ekin) {
  HadXs<real_t> r;
  if (p.pdg == pdg::kKaonMinus || p.pdg == pdg::kKaonPlus) {
    r = hn_xsc_ns<real_t>(p, nucleon, ekin);
  } else if (p.pdg == pdg::kKaonZeroShort || p.pdg == pdg::kKaonZeroLong) {
    real_t fact = real_t(0.5);
    HadXs<real_t> m, pl;
    if (ekin > hn_ekin_max_qb<real_t>()) {
      m = hn_xsc_ns<real_t>(kaon_minus<real_t>(), nucleon, ekin);
      pl = hn_xsc_ns<real_t>(kaon_plus<real_t>(), nucleon, ekin);
    } else {
      const real_t e = (ekin > hn_ekin_min<real_t>()) ? ekin : hn_ekin_min<real_t>();
      fact *= sqrt(hn_ekin_max_qb<real_t>() / e);
      m = hn_xsc_ns<real_t>(kaon_minus<real_t>(), nucleon, hn_ekin_max_qb<real_t>());
      pl = hn_xsc_ns<real_t>(kaon_plus<real_t>(), nucleon, hn_ekin_max_qb<real_t>());
    }
    r.total = (m.total + pl.total) * fact;
    r.elastic = (m.elastic + pl.elastic) * fact;
    r.inelastic = (m.inelastic + pl.inelastic) * fact;
  }
  return r;
}

/// G4HadronNucleonXsc::HadronNucleonXsc - the dispatcher, in its own order.
///
/// The order is load-bearing: `pdg > 3000` is tested before `pdg > 220`, and both are tested
/// on |pdg|, so an anti-hyperon takes the hyperon branch. The two refused branches are the
/// ones named in the file header.
template <typename real_t>
__host__ __device__ inline HadXs<real_t> hadron_nucleon_xsc(const Projectile<real_t>& p,
                                                            const Projectile<real_t>& nucleon,
                                                            real_t ekin) {
  const int apdg = (p.pdg < 0) ? -p.pdg : p.pdg;

  if (apdg == pdg::kProton || apdg == pdg::kNeutron || apdg == pdg::kPiPlus) {
    return hn_xsc_ns<real_t>(p, nucleon, ekin);
  }
  if (apdg == pdg::kGamma) { return hn_xsc_pdg<real_t>(p, nucleon, ekin); }
  if (apdg == pdg::kKaonPlus || apdg == pdg::kKaonZeroShort || apdg == pdg::kKaonZeroLong) {
    return hn_kaon_xsc_ns<real_t>(p, nucleon, ekin);
  }
  if (apdg > 3000) {
    // The hyperon list of G4HadronNucleonXsc::HadronNucleonXsc, s- then c- then b-.
    const bool hyperon =
        (apdg == 3122 || apdg == 3222 || apdg == 3112 || apdg == 3212 || apdg == 3322 ||
         apdg == 3312 || apdg == 3324 || apdg == 4122 || apdg == 4332 || apdg == 4212 ||
         apdg == 4222 || apdg == 4112 || apdg == 4232 || apdg == 4132 || apdg == 5122 ||
         apdg == 5332 || apdg == 5112 || apdg == 5222 || apdg == 5212 || apdg == 5132 ||
         apdg == 5232);
    if (hyperon) {
      HadXs<real_t> r;
      r.refused = XsRefusal::kHyperonNucleonXscNS;
      return r;
    }
    return hn_xsc_pdg<real_t>(p, nucleon, ekin);
  }
  if (apdg > 220) {
    const bool scb = (apdg == 511 || apdg == 421 || apdg == 531 || apdg == 541 ||
                      apdg == 431 || apdg == 411 || apdg == 521 || apdg == 221 ||
                      apdg == 331 || apdg == 441 || apdg == 443 || apdg == 543);
    if (scb) {
      HadXs<real_t> r;
      r.refused = XsRefusal::kSCBMesonNucleonXscNS;
      return r;
    }
    return hn_xsc_pdg<real_t>(p, nucleon, ekin);
  }
  return hn_xsc_pdg<real_t>(p, nucleon, ekin);
}

}  // namespace g4gpu::hadronic::xs
