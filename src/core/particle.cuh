#pragma once
#include "core/units.cuh"
namespace g4gpu {

/// Every particle G4EmStandardPhysics registers EM processes for.
/// The first three are what example B1 produces; the rest exist because the physics list
/// covers them (via G4EmBuilder::ConstructCharged), and their models are transcribed for
/// completeness even though B1 never creates one.
enum class ParticleType : int {
  kGamma = 0,
  kElectron = 1,
  kPositron = 2,
  kMuonMinus = 3,
  kMuonPlus = 4,
  kPionPlus = 5,
  kPionMinus = 6,
  kKaonPlus = 7,
  kKaonMinus = 8,
  kProton = 9,
  kAntiProton = 10,
  kAlpha = 11,
  kHe3 = 12,
  kGenericIon = 13,
  kNumTypes = 14
};

/// Static properties the EM models need: PDG mass, charge and spin.
/// Values are Geant4 11.1.1's own PDG table; the mass and charge columns of
/// ref/oracle/hadron_tables.csv carry the same numbers and tests/test_hadron.cu checks them.
template <typename real_t>
struct ParticleDef {
  real_t mass;    ///< MeV
  real_t charge;  ///< in units of the positron charge
  real_t spin;    ///< in units of hbar
  bool is_ion;    ///< selects IonBarkasCorrection rather than HighOrderCorrections
  bool is_alpha;
  bool is_lepton;      ///< G4ParticleDefinition::GetLeptonNumber() != 0
  /// G4BetheBlochModel's `magMoment2`: the magnetic moment in units of the particle's *own*
  /// magneton, squared, minus one - zero for a Dirac particle, the anomaly for a real one.
  /// It appears only in the delta-ray form-factor rejection and only for a particle with
  /// spin, so the -1 that spin-0 particles carry here is never read.
  ///
  /// Geant4 computes it as `PDGMagneticMoment*mass/(0.5*eplus*hbar_Planck*c_squared)`, then
  /// squares and subtracts one. For the proton, whose moment is quoted as g*mN with mN built
  /// from the proton mass, that reduces to g exactly; for He3, quoted against the same
  /// (proton) mN but carrying its own mass, it is g*m/m_p. Those reductions are folded into
  /// the constants below rather than recomputed, because the magneton a moment was quoted
  /// against is not a property this struct carries.
  real_t mag_moment2;
};

template <typename real_t>
__host__ __device__ inline ParticleDef<real_t> particle_def(ParticleType t) {
  switch (t) {
    case ParticleType::kGamma:
      return {real_t(0), real_t(0), real_t(1), false, false, false, real_t(-1)};
    case ParticleType::kElectron:
      return {real_t(0.510998910), real_t(-1), real_t(0.5), false, false, true,
              real_t(1.00115965218076 * 1.00115965218076 - 1)};
    case ParticleType::kPositron:
      return {real_t(0.510998910), real_t(1), real_t(0.5), false, false, true,
              real_t(1.00115965218076 * 1.00115965218076 - 1)};
    case ParticleType::kMuonMinus:
      return {real_t(105.6583715), real_t(-1), real_t(0.5), false, false, true,
              real_t(1.0011659209 * 1.0011659209 - 1)};
    case ParticleType::kMuonPlus:
      return {real_t(105.6583715), real_t(1), real_t(0.5), false, false, true,
              real_t(1.0011659209 * 1.0011659209 - 1)};
    // 139.5701, not the PDG 139.57018: G4PionPlus is constructed with a literal
    // `0.1395701*GeV`, and matching Geant4 is the point rather than matching the PDG. The two
    // differ by 6e-7 relative, which is nothing in a stopping power and everything in a table
    // compared at 1e-9 - and it is the mass every pion energy is scaled by.
    case ParticleType::kPionPlus:
      return {real_t(139.5701), real_t(1), real_t(0), false, false, false, real_t(-1)};
    case ParticleType::kPionMinus:
      return {real_t(139.5701), real_t(-1), real_t(0), false, false, false, real_t(-1)};
    case ParticleType::kKaonPlus:
      return {real_t(493.677), real_t(1), real_t(0), false, false, false, real_t(-1)};
    case ParticleType::kKaonMinus:
      return {real_t(493.677), real_t(-1), real_t(0), false, false, false, real_t(-1)};
    case ParticleType::kProton:
      return {units::proton_mass_c2<real_t>(), real_t(1), real_t(0.5), false, false, false,
              real_t(2.792847351 * 2.792847351 - 1)};
    case ParticleType::kAntiProton:
      return {units::proton_mass_c2<real_t>(), real_t(-1), real_t(0.5), false, false, false,
              real_t(2.792847351 * 2.792847351 - 1)};
    case ParticleType::kAlpha:
      // is_ion is true despite G4BetheBlochModel::SetupParameters computing
      // isIon = (!isAlpha && q > 1.1). That line only runs for a particle other than the
      // one the model was initialised with; Initialise itself sets isIon for any charge
      // above one unit, alpha included, so alpha takes the IonBarkasCorrection branch.
      return {real_t(3727.379), real_t(2), real_t(0), true, true, false, real_t(-1)};
    case ParticleType::kHe3:
      // The moment is quoted against the *proton* magneton yet scaled by He3's own mass,
      // so the cancellation that leaves the proton with a bare g does not happen here.
      return {real_t(2808.391), real_t(2), real_t(0.5), true, false, false,
              real_t(2.12762485 * 2808.391 / 938.272013
                     * (2.12762485 * 2808.391 / 938.272013) - 1)};
    case ParticleType::kGenericIon:
      // 938.2723 MeV, and it is not units::proton_mass_c2 (938.272013). G4GenericIon is
      // constructed with a literal `0.9382723*GeV` of its own, and since every non-alpha,
      // non-He3 ion scales its dE/dx and range from GenericIon's table by a mass ratio, that
      // literal is what the ratio is against. The two differ by 3e-7 relative, which is
      // nothing physically and everything to a table compared at 1e-9.
      return {real_t(938.2723), real_t(1), real_t(0.5), true, false, false,
              real_t(2.792847351 * 2.792847351 - 1)};
    default:
      return {real_t(0), real_t(0), real_t(0), false, false, false, real_t(-1)};
  }
}

/// Is G4ionIonisation the ionisation process Geant4 registers for this species, rather than
/// G4hIonisation or G4MuIonisation?
///
/// This one predicate decides three things - the low-energy dE/dx model, the fluctuation
/// model, and whether a base particle's table is scaled - so it is written once here rather
/// than re-derived at each of them.
///
/// It is a table and not a rule over charge or mass, because Geant4's own choice is neither.
/// G4EmBuilder::ConstructIonEmPhysics splits the light nuclei:
///
///     alpha, He3        -> G4ionIonisation    (charge 2, masses 3727 and 2808)
///     deuteron, triton  -> G4hIonisation      (charge 1, masses 1876 and 2809)
///     GenericIon        -> G4ionIonisation
///     p, pbar, pi, K    -> G4hIonisation
///     mu+, mu-          -> G4MuIonisation
///
/// A triton and a He3 have almost the same mass and different processes; a deuteron and an
/// alpha differ in both charge and process but not in the direction a charge rule would
/// guess. `charge > 1.5` happens to be right for every species transported today and is wrong
/// for the first deuteron.
__host__ __device__ inline bool uses_ion_ionisation(ParticleType t) {
  return t == ParticleType::kAlpha || t == ParticleType::kHe3
         || t == ParticleType::kGenericIon;
}

/// G4MuIonisation's species. Separate from the two above because the muon's model boundary is
/// a flat 200 keV rather than the mass-scaled 2 MeV, and its high-energy model is
/// G4MuBetheBlochModel rather than G4BetheBlochModel.
__host__ __device__ inline bool is_muon(ParticleType t) {
  return t == ParticleType::kMuonMinus || t == ParticleType::kMuonPlus;
}

/// Does this species' ionisation process attach G4IonFluctuations rather than
/// G4UniversalFluctuation?
///
/// The same question as uses_ion_ionisation, because both answers come from which process is
/// registered: G4ionIonisation calls ModelOfFluctuations(true) unconditionally and
/// G4hIonisation calls it with false for everything it is given here.
///
/// One case this cannot express: G4MuIonisation attaches G4IonFluctuations to its *low-energy*
/// model and G4UniversalFluctuation above 200 keV, so a muon's choice is per-model rather than
/// per-particle. When muons are transported that choice belongs at the model selection in
/// hadron_range.cuh, not here.
__host__ __device__ inline bool uses_ion_fluctuations(ParticleType t) {
  return uses_ion_ionisation(t);
}

/// The particle whose dE/dx and range tables this species is scaled from, or itself.
///
/// G4VEnergyLossProcess does not build a table per particle. Each ionisation process names a
/// base particle - G4hIonisation.cc and G4ionIonisation.cc both do it in
/// InitialiseEnergyLossProcess - and everything else looks that base particle's table up at a
/// *scaled* energy and rescales the answer:
///
///     scaledE = E * massRatio,   massRatio     = m_base / m_this
///     dE/dx   = chargeSqRatio * table_dedx(scaledE)
///     range   = table_range(scaledE) / (chargeSqRatio * massRatio)
///
/// The species with no base particle - the ones that get their own table - are exactly those
/// G4hIonisation lists by name (proton, anti_proton, pi+, pi-, kaon+, kaon-, GenericIon,
/// alpha), plus mu+ and mu- which G4MuIonisation never gives a base to. That leaves He3, whose
/// base is GenericIon, and deuteron and triton, whose base is the proton.
__host__ __device__ inline ParticleType hadron_base_particle(ParticleType t) {
  return (t == ParticleType::kHe3) ? ParticleType::kGenericIon : t;
}

/// True for the particles that go through the heavy-charged-particle ionisation chain
/// (Bragg / Bethe-Bloch) rather than Moller-Bhabha.
__host__ __device__ inline bool is_heavy_charged(ParticleType t) {
  return static_cast<int>(t) >= static_cast<int>(ParticleType::kMuonMinus)
         && static_cast<int>(t) < static_cast<int>(ParticleType::kNumTypes);
}

}  // namespace g4gpu
