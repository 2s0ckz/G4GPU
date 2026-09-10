// The projectile, as every hadronic cross section in Geant4 actually asks about it.
//
// The classes in this directory are transcribed from Geant4 code that branches on
// `theParticle == theProton`, `pdg == 211`, `aParticle->GetBaryonNumber()` and so on - pointer
// identity against the singleton particle definitions, or the PDG encoding. There is no
// species enum to switch on, and the branches are not all reachable from one: G4HadronNucleonXsc
// asks for the *target nucleon* too, and G4ComponentGGNuclNuclXsc needs the projectile's Z, A
// and its number of lambdas.
//
// So the interface is this struct rather than `core/particle.cuh`'s ParticleType. Three
// reasons, and the third is the one that decided it:
//
//   1. core/particle.cuh belongs to P1 (docs/HADRONIC_PLAN.md section 6 rule 5) and does not
//      yet have a neutron, a pi0, a deuteron or a triton.
//   2. A cross section is also needed for the *target* nucleon, which is not a transported
//      species at all.
//   3. GenericIon is not one particle. Its charge, baryon number and mass are the target
//      nucleus's, and G4ComponentGGNuclNuclXsc reads all three off the definition. An enum
//      would have to be widened to a (Z, A) pair for that one case anyway.
//
// The fields are exactly the G4ParticleDefinition accessors the transcribed code calls, so a
// reader can check the transcription against the Geant4 line without a translation step.
#pragma once

#include "core/units.cuh"
#include "data/nuclei_mass_ame12.hh"

namespace g4gpu::hadronic::xs {

/// PDG encodings the branches below test against, spelled out so that a comparison in a
/// transcribed function reads like the Geant4 line it came from.
namespace pdg {
constexpr int kGamma = 22;
constexpr int kProton = 2212;
constexpr int kNeutron = 2112;
constexpr int kAntiProton = -2212;
constexpr int kAntiNeutron = -2112;
constexpr int kPiPlus = 211;
constexpr int kPiMinus = -211;
constexpr int kPiZero = 111;
constexpr int kKaonPlus = 321;
constexpr int kKaonMinus = -321;
constexpr int kKaonZeroShort = 310;
constexpr int kKaonZeroLong = 130;
constexpr int kLambda = 3122;
constexpr int kSigmaMinus = 3112;
}  // namespace pdg

/// What the Geant4 cross sections read off a G4ParticleDefinition.
///
/// @c charge is in units of eplus, which is 1 in this unit system, so it is the signed integer
/// charge written as a real - Geant4 divides by eplus wherever it needs the number.
template <typename real_t>
struct Projectile {
  int pdg = 0;
  real_t mass = 0;            ///< GetPDGMass(), MeV
  real_t charge = 0;          ///< GetPDGCharge()/eplus
  int baryon_number = 0;      ///< GetBaryonNumber()
  int n_lambdas = 0;          ///< GetNumberOfLambdasInHypernucleus()

  __host__ __device__ bool is_hypernucleus() const { return n_lambdas > 0; }
};

// The particle definitions QBBC's hadronic chain uses, from their own Geant4 source files.
// Masses: G4Proton.cc and G4Neutron.cc take CLHEP's proton_mass_c2 / neutron_mass_c2;
// G4Deuteron.cc 1.875613 GeV, G4Triton.cc 2.808921 GeV, G4Alpha.cc 3.727379 GeV, G4He3.cc
// 2.808391 GeV, G4PionPlus/Minus.cc 0.1395701 GeV, G4KaonPlus/Minus.cc 0.493677 GeV,
// G4Lambda.cc 1.115683 GeV, G4KaonZeroShort/Long.cc 0.497614 GeV.
//
// EVERY ONE IS WRITTEN AS Geant4 WRITES IT - `x.yz * GeV` AND NOT THE MeV LITERAL
//
// This looks like a style choice and is not. `0.493677 * 1000.0` and `493.677` are different
// doubles: the first is 493.67699999999996, the second 493.67700000000002, one ulp apart. The
// kaon was `real_t(493.677)` here and every Coulomb factor for a kaon disagreed with Geant4 by
// 1.4e-12 - a thousand times the tolerance - because `totTcm` in
// G4NuclearRadii::CoulombFactor is `sqrt(pM^2 + tM^2 + 2*pElab*tM) - pM - tM`, which at 1 MeV
// subtracts 1432 MeV from 1432 MeV to get 0.65. That cancellation multiplies a relative error
// in the mass by about 2200, and the amplification is a further 1/(1-bC/totTcm) = 8 where the
// factor is near its threshold. One ulp in, 1.4e-12 out.
//
// So the arithmetic is Geant4's, not the shortest spelling of the same number. Only the kaon
// actually differed; the others are written this way so that the next one cannot.

template <typename real_t> __host__ __device__ inline Projectile<real_t> proton() {
  return {pdg::kProton, units::proton_mass_c2<real_t>(), real_t(1), 1, 0};
}
template <typename real_t> __host__ __device__ inline Projectile<real_t> neutron() {
  return {pdg::kNeutron, units::neutron_mass_c2<real_t>(), real_t(0), 1, 0};
}
template <typename real_t> __host__ __device__ inline Projectile<real_t> anti_proton() {
  return {pdg::kAntiProton, units::proton_mass_c2<real_t>(), real_t(-1), -1, 0};
}
template <typename real_t> __host__ __device__ inline Projectile<real_t> anti_neutron() {
  return {pdg::kAntiNeutron, units::neutron_mass_c2<real_t>(), real_t(0), -1, 0};
}
template <typename real_t> __host__ __device__ inline Projectile<real_t> pi_plus() {
  return {pdg::kPiPlus, real_t(0.1395701) * units::GeV<real_t>(), real_t(1), 0, 0};
}
template <typename real_t> __host__ __device__ inline Projectile<real_t> pi_minus() {
  return {pdg::kPiMinus, real_t(0.1395701) * units::GeV<real_t>(), real_t(-1), 0, 0};
}
template <typename real_t> __host__ __device__ inline Projectile<real_t> kaon_plus() {
  return {pdg::kKaonPlus, real_t(0.493677) * units::GeV<real_t>(), real_t(1), 0, 0};
}
template <typename real_t> __host__ __device__ inline Projectile<real_t> kaon_minus() {
  return {pdg::kKaonMinus, real_t(0.493677) * units::GeV<real_t>(), real_t(-1), 0, 0};
}
// G4KaonZeroShort.cc and G4KaonZeroLong.cc both give 0.497614 GeV, so the two differ here only
// in the PDG code - and the code is what G4HadronNucleonXsc's dispatcher and
// G4ComponentGGHadronNucleusXsc's `is_kaon` test branch on. They are not transported by QBBC
// (P1 refuses them by name) but G4HadronNucleonXsc::KaonNucleonXscNS and KaonNucleonXscGG both
// have a neutral-kaon arm - the K-/K+ mean - and nothing else reaches it, so without these two
// that arm is transcribed and never compared.
template <typename real_t> __host__ __device__ inline Projectile<real_t> kaon_zero_short() {
  return {pdg::kKaonZeroShort, real_t(0.497614) * units::GeV<real_t>(), real_t(0), 0, 0};
}
template <typename real_t> __host__ __device__ inline Projectile<real_t> kaon_zero_long() {
  return {pdg::kKaonZeroLong, real_t(0.497614) * units::GeV<real_t>(), real_t(0), 0, 0};
}
template <typename real_t> __host__ __device__ inline Projectile<real_t> lambda() {
  return {pdg::kLambda, real_t(1.115683) * units::GeV<real_t>(), real_t(0), 1, 0};
}
template <typename real_t> __host__ __device__ inline Projectile<real_t> gamma() {
  return {pdg::kGamma, real_t(0), real_t(0), 0, 0};
}
template <typename real_t> __host__ __device__ inline Projectile<real_t> deuteron() {
  return {1000010020, data::deuteron_mass_c2<real_t>(), real_t(1), 2, 0};
}
template <typename real_t> __host__ __device__ inline Projectile<real_t> triton() {
  return {1000010030, data::triton_mass_c2<real_t>(), real_t(1), 3, 0};
}
template <typename real_t> __host__ __device__ inline Projectile<real_t> he3() {
  return {1000020030, data::he3_mass_c2<real_t>(), real_t(2), 3, 0};
}
template <typename real_t> __host__ __device__ inline Projectile<real_t> alpha() {
  return {1000020040, data::alpha_mass_c2<real_t>(), real_t(2), 4, 0};
}

/// An arbitrary ion, as G4IonTable builds it: the PDG code is 10LZZZAAAI with L = I = 0, the
/// charge is Z, the baryon number A, and the mass is G4IonTable::GetNucleusMass, which is
/// G4NucleiProperties::GetNuclearMass(A, Z) except for the six light ions it answers from the
/// particle table - which is exactly what data::nuclear_mass does.
///
/// Returns a projectile with mass 0 when the nuclide is outside the AME12 table; see
/// data/nuclei_mass_ame12.hh for what is refused there and why a zero must not be used.
template <typename real_t> __host__ __device__ inline Projectile<real_t> generic_ion(int z,
                                                                                     int a) {
  return {1000000000 + 10000 * z + 10 * a, data::nuclear_mass<real_t>(a, z),
          static_cast<real_t>(z), a, 0};
}

}  // namespace g4gpu::hadronic::xs
