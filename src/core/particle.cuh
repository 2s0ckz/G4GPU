#pragma once
#include "core/units.cuh"
namespace g4gpu {

/// beta = v/c, exactly as G4DynamicParticle::ComputeBeta computes it - including its
/// shortcut to exactly 1 for a massless particle and above 1000 rest masses.
///
/// Defined once because it is now read from two unrelated places: G4Track::GetVelocity,
/// which sets the time of flight over a step, and G4IonFluctuations, which needs beta^2.
/// Writing `kinetic*(kinetic+2m)/E^2` in one of them - algebraically the same thing - would
/// not be bit-identical to this form, and would put a 1e-16 wobble between two numbers
/// Geant4 keeps exactly equal. docs/RISK.md V8 is what a duplicated constant cost before.
template <typename real_t>
__host__ __device__ inline real_t dynamic_particle_beta(real_t kinetic, real_t mass) {
  if (mass <= real_t(0) || kinetic >= real_t(1000) * mass) { return real_t(1); }
  const real_t t = kinetic / mass;
  return sqrt(t * (t + real_t(2))) / (t + real_t(1));
}

/// Every particle QBBC creates and this port has a name for.
///
/// It began as "every particle G4EmStandardPhysics registers EM processes for", which is why
/// the first fourteen are all charged or a photon. That was the right set while the only
/// physics here was electromagnetic, and it is the wrong set now: a hadronic cascade emits
/// neutrons and pi0s, and a decay emits neutrinos, and none of the three has an EM process
/// at all. So the list is what QBBC can PRODUCE, not what it registers a dE/dx for.
///
/// Three groups, and which group a species is in is answered by classify_species() in
/// core/track_buffer.cuh rather than by its position here:
///
///   * stepped     - a kernel exists (gamma, e+-, and the eleven charged hadrons and the two
///                   neutral hadrons below)
///   * counted     - the six neutrinos: created, their energy booked as carried out of the
///                   event, never stepped. QBBC gives them Transportation and nothing else,
///                   so in Geant4 they leave the world with their energy; the accounting here
///                   is that escape, done at emission because there is no point moving them.
///   * refused     - everything else a cascade or a decay can make. Named and counted at
///                   emission and fatal as a primary, never a silent default.
///
/// APPEND ONLY, and nothing may be inserted before kNumTypes. The value is stored on every
/// track in a device buffer and written into trajectory records the viewer reads back, so a
/// renumbering silently reinterprets both.
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
  // Added with the hadronic transport plumbing. The neutron and pi0 are the first neutral
  // hadrons; the deuteron and triton are the first species whose dE/dx comes entirely from a
  // base particle's table by G4hIonisation's scaling.
  kNeutron = 14,
  kPiZero = 15,
  kDeuteron = 16,
  kTriton = 17,
  // The six neutrinos, as separate species rather than one "neutrino".
  //
  // Their ParticleDef rows are identical - massless, uncharged, spin 1/2, lepton number +-1 -
  // so one type would give the same numbers everywhere in this file. What it would throw away
  // is the only thing a neutrino carries: which one it is. G4DecayPhysics' channels name them
  // individually (mu- -> e- + anti_nu_e + nu_mu, pi+ -> mu+ + nu_mu), so a decay that had to
  // report "a neutrino" could not be checked against the channel it came from. docs/RISK.md
  // V35 is what discarding the one distinguishing bit costs later.
  kNeutrinoE = 18,
  kAntiNeutrinoE = 19,
  kNeutrinoMu = 20,
  kAntiNeutrinoMu = 21,
  kNeutrinoTau = 22,
  kAntiNeutrinoTau = 23,
  // Refused, with real rows. QBBC transports all six - species_processes.csv shows the charged
  // hyperons carrying hIoni scaled from the proton and an Urban msc, and all six carrying
  // hadElastic, an inelastic and a decay - so they are species this port is MISSING rather
  // than species that do not arise. They are here so that the refusal can name what it
  // refused, and so that `/gun/particle lambda` gets an answer about the lambda instead of
  // "unknown particle".
  //
  // This is not the complete refused set. sigma0, xi0, omega-, the anti-hyperons, the light
  // anti-nuclei and the b/c hadrons `EnableBCParticles` turns on are all reachable through a
  // cascade and none of them is here - because nothing in this port can produce one yet, and a
  // row nobody can reach is a row nobody checks. The MECHANISM is what P1 owes and what is
  // finished: species_disposition refuses anything without a kernel, BufferEmitter::push counts
  // it under its own name, and the run reports it. Whichever package first emits an omega-
  // appends four lines - an enumerator, a name, a ParticleDef row, and a dump entry - and gets
  // the refusal for nothing.
  kKaonZeroLong = 24,
  kKaonZeroShort = 25,
  kLambda = 26,
  kSigmaPlus = 27,
  kSigmaMinus = 28,
  kXiMinus = 29,
  kNumTypes = 30
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
      //
      // mag_moment2 is -1 and NOT the proton's 6.8, which is what this row carried until
      // ref/oracle/species_tables.csv was dumped and disagreed. G4GenericIon never calls
      // SetPDGMagneticMoment, so its moment is zero, so magmom is zero and magMoment2 is
      // 0 - 1. The proton's value was here because GenericIon borrows the proton's mass, and
      // borrowing one constant is not borrowing the other. It is read - the delta-ray
      // rejection in em/hadron_delta.cuh uses it whenever spin > 0, and GenericIon's spin is
      // 1/2 - so this was a wrong spectrum waiting for the first transported ion.
      return {real_t(938.2723), real_t(1), real_t(0.5), true, false, false, real_t(-1)};
    // ---- neutral hadrons.
    case ParticleType::kNeutron:
      // G4Neutron.cc: `new G4Ions(name, neutron_mass_c2, ...)` with 2*spin = 1, lepton
      // number 0, baryon number +1.
      //
      // The mass is CLHEP's neutron_mass_c2 = 939.56536 MeV. The comment sitting directly
      // above that line in G4Neutron.cc says 939.56563 - two digits transposed - and it is
      // commented-out documentation of an older CLHEP, not the value used. Taking the comment
      // would have put the mass 2.9e-7 out, which is nothing in a kinematic limit and three
      // hundred times the tolerance species_tables.csv is compared at.
      //
      // The moment is quoted against the PROTON magneton while the neutron carries its own
      // mass, so the cancellation that leaves the proton a bare g does not happen: see the
      // note on ParticleDef::mag_moment2 and the identical arithmetic for He3.
      return {units::neutron_mass_c2<real_t>(), real_t(0), real_t(0.5), false, false, false,
              real_t(-1.9130427 * 939.56536 / 938.272013
                     * (-1.9130427 * 939.56536 / 938.272013) - 1)};
    case ParticleType::kPiZero:
      // G4PionZero.cc: `0.1349766*GeV`, 2*spin = 0, no magnetic moment. Spin zero, so
      // mag_moment2 is never read and -1 is also the value Geant4 computes from a zero moment.
      return {real_t(134.9766), real_t(0), real_t(0), false, false, false, real_t(-1)};
    // ---- the two light ions G4EmBuilder gives G4hIonisation rather than G4ionIonisation.
    //
    // is_ion is FALSE for both, and that is not an oversight. G4BetheBlochModel::Initialise
    // sets isIon on `particle->GetPDGCharge() > CLHEP::eplus || pname == "GenericIon"`, and a
    // deuteron's charge is exactly eplus - strictly greater is strictly greater. So a deuteron
    // takes the HighOrderCorrections branch like a proton, not the IonBarkasCorrection branch
    // like an alpha, even though it is a nucleus. A rule written on "is it a nucleus" gets
    // both of these wrong; see uses_ion_ionisation below for the same split by process.
    case ParticleType::kDeuteron:
      // G4Deuteron.cc: `1.875613*GeV`, 2*spin = 2 (so spin 1), moment 0.857438230 nuclear
      // magnetons against the proton's magneton.
      return {real_t(1875.613), real_t(1), real_t(1), false, false, false,
              real_t(0.857438230 * 1875.613 / 938.272013
                     * (0.857438230 * 1875.613 / 938.272013) - 1)};
    case ParticleType::kTriton:
      // G4Triton.cc: `2.808921*GeV`, 2*spin = 1 (so spin 1/2), moment 2.97896248.
      //
      // 2808.921, and He3 above is 2808.391. The two differ in the third decimal and in
      // nothing else a glance would catch - and they have different charges, different spins,
      // different processes and different base particles. Nothing here is shared between them.
      return {real_t(2808.921), real_t(1), real_t(0.5), false, false, false,
              real_t(2.97896248 * 2808.921 / 938.272013
                     * (2.97896248 * 2808.921 / 938.272013) - 1)};
    // ---- neutrinos. Massless, uncharged, spin 1/2, lepton number +-1, no moment.
    //
    // is_lepton is true, which is the one field that does any work for them: it is
    // `GetLeptonNumber() != 0`, and G4BetheBlochModel::SetupParameters leaves tlimit at
    // infinity for a lepton. Nothing here ever asks a neutrino for a stopping power, but the
    // row has to be right rather than absent, because the alternative is a zero that reads
    // like a value.
    case ParticleType::kNeutrinoE:
    case ParticleType::kAntiNeutrinoE:
    case ParticleType::kNeutrinoMu:
    case ParticleType::kAntiNeutrinoMu:
    case ParticleType::kNeutrinoTau:
    case ParticleType::kAntiNeutrinoTau:
      return {real_t(0), real_t(0), real_t(0.5), false, false, true, real_t(-1)};
    // ---- refused, but with the real numbers. See the enum.
    //
    // Every moment below is quoted in G4*.cc against the proton magneton while the particle
    // carries its own mass, so each is scaled by mass/m_p exactly as the neutron's and He3's
    // are. Transcribed from the constructors rather than copied out of the oracle CSV, which
    // would make the test that compares them a comparison of a number with itself.
    case ParticleType::kKaonZeroLong:
    case ParticleType::kKaonZeroShort:
      // G4KaonZeroLong.cc / G4KaonZeroShort.cc: both `0.497614*GeV`, 2*spin = 0, no moment.
      // Same mass and spin, different lifetimes (1.287e-14 and 7.3508e-12 MeV of width) and
      // different decay tables - so they are two species and not one, and P4 will need both.
      return {real_t(497.614), real_t(0), real_t(0), false, false, false, real_t(-1)};
    case ParticleType::kLambda:
      // G4Lambda.cc: `1.115683*GeV`, 2*spin = 1, moment -0.613 mN.
      return {real_t(1115.683), real_t(0), real_t(0.5), false, false, false,
              real_t(-0.613 * 1115.683 / 938.272013
                     * (-0.613 * 1115.683 / 938.272013) - 1)};
    case ParticleType::kSigmaPlus:
      // G4SigmaPlus.cc: `1.18937*GeV`, +eplus, 2*spin = 1, moment 2.458 mN.
      return {real_t(1189.37), real_t(1), real_t(0.5), false, false, false,
              real_t(2.458 * 1189.37 / 938.272013 * (2.458 * 1189.37 / 938.272013) - 1)};
    case ParticleType::kSigmaMinus:
      // G4SigmaMinus.cc: `1.197449*GeV`, -eplus, 2*spin = 1, moment -1.160 mN.
      return {real_t(1197.449), real_t(-1), real_t(0.5), false, false, false,
              real_t(-1.160 * 1197.449 / 938.272013
                     * (-1.160 * 1197.449 / 938.272013) - 1)};
    case ParticleType::kXiMinus:
      // G4XiMinus.cc: `1.32171*GeV`, -eplus, 2*spin = 1, moment -0.6507 mN.
      return {real_t(1321.71), real_t(-1), real_t(0.5), false, false, false,
              real_t(-0.6507 * 1321.71 / 938.272013
                     * (-0.6507 * 1321.71 / 938.272013) - 1)};
    case ParticleType::kNumTypes:
      break;
  }
  // A ParticleType with no row above.
  //
  // NOT a row of zeros, which is what this returned. A zero mass makes beta exactly 1, a zero
  // charge makes every stopping power zero, and a zero spin turns the delta-ray form factor
  // off - so an unhandled species came back as a massless neutral that traverses the geometry
  // depositing nothing, which is a perfectly plausible-looking gamma and is the shape of
  // failure this project keeps writing up: the wrong answer that declines to announce itself.
  //
  // A NEGATIVE mass announces itself. Every guard in the ionisation chain is `pd.mass <= 0`
  // and fires; dynamic_particle_beta's `mass <= 0` returns 1 as it does for a photon; and the
  // value prints as -1 in any report rather than as a number somebody might believe. There is
  // no case that reaches here today - every enumerator but kNumTypes has a row - and
  // tests/test_species.cu asserts the poison rather than trusting this comment.
  return {real_t(-1), real_t(0), real_t(0), false, false, false, real_t(-1)};
}

/// Geant4's own name for a species, so that a refusal or a report can say what it refused.
///
/// One table, here, next to the masses. There were three before - G4ParticleTable's `add`
/// list, the `type_of` helper each test writes, and the prose in G4RunManager::CheckSpecies'
/// message - and the third had drifted: it described the missing muon transport as "the
/// smallest gap of the three" long after the range table had landed.
///
/// Returns "?" for a value outside the enum rather than an empty string, so that a message
/// built from it reads as a message rather than as a gap.
__host__ __device__ inline const char* particle_name(ParticleType t) {
  switch (t) {
    case ParticleType::kGamma: return "gamma";
    case ParticleType::kElectron: return "e-";
    case ParticleType::kPositron: return "e+";
    case ParticleType::kMuonMinus: return "mu-";
    case ParticleType::kMuonPlus: return "mu+";
    case ParticleType::kPionPlus: return "pi+";
    case ParticleType::kPionMinus: return "pi-";
    case ParticleType::kKaonPlus: return "kaon+";
    case ParticleType::kKaonMinus: return "kaon-";
    case ParticleType::kProton: return "proton";
    case ParticleType::kAntiProton: return "anti_proton";
    case ParticleType::kAlpha: return "alpha";
    case ParticleType::kHe3: return "He3";
    case ParticleType::kGenericIon: return "GenericIon";
    case ParticleType::kNeutron: return "neutron";
    case ParticleType::kPiZero: return "pi0";
    case ParticleType::kDeuteron: return "deuteron";
    case ParticleType::kTriton: return "triton";
    case ParticleType::kNeutrinoE: return "nu_e";
    case ParticleType::kAntiNeutrinoE: return "anti_nu_e";
    case ParticleType::kNeutrinoMu: return "nu_mu";
    case ParticleType::kAntiNeutrinoMu: return "anti_nu_mu";
    case ParticleType::kNeutrinoTau: return "nu_tau";
    case ParticleType::kAntiNeutrinoTau: return "anti_nu_tau";
    case ParticleType::kKaonZeroLong: return "kaon0L";
    case ParticleType::kKaonZeroShort: return "kaon0S";
    case ParticleType::kLambda: return "lambda";
    case ParticleType::kSigmaPlus: return "sigma+";
    case ParticleType::kSigmaMinus: return "sigma-";
    case ParticleType::kXiMinus: return "xi-";
    default: return "?";
  }
}

/// The PDG code of a species, and its inverse.
///
/// P8 needs both because every hadronic process and every decay channel names its products by
/// PDG code - P4's `decay_tables.hh` and P5's `HadSecondary` both do, deliberately, because
/// `core/particle.cuh` belongs to a different package from either - and the transport has to
/// turn that code into a species with a kernel, a refusal, or a neutrino to book.
///
/// A NUCLEUS IS NOT A PDG CODE THIS TABLE CAN LIST. `10LZZZAAAI` encodes (Z, A) and a
/// run-dependent isomer index, so it is DECODED rather than matched: see
/// `particle_type_of_nucleus`. The six light nuclei have fixed codes and are listed anyway,
/// because a model that emits an alpha writes `1000020040` and a reader that only decoded would
/// have to agree with the encoding in two places.
__host__ __device__ inline int pdg_code(ParticleType t) {
  switch (t) {
    case ParticleType::kGamma: return 22;
    case ParticleType::kElectron: return 11;
    case ParticleType::kPositron: return -11;
    case ParticleType::kMuonMinus: return 13;
    case ParticleType::kMuonPlus: return -13;
    case ParticleType::kPionPlus: return 211;
    case ParticleType::kPionMinus: return -211;
    case ParticleType::kKaonPlus: return 321;
    case ParticleType::kKaonMinus: return -321;
    case ParticleType::kProton: return 2212;
    case ParticleType::kAntiProton: return -2212;
    case ParticleType::kAlpha: return 1000020040;
    case ParticleType::kHe3: return 1000020030;
    // G4GenericIon's own code is 1000000000 - Z = 0, A = 0, which is not a nuclide. It is a
    // placeholder definition whose tables every real ion scales from, and no process emits one.
    case ParticleType::kGenericIon: return 1000000000;
    case ParticleType::kNeutron: return 2112;
    case ParticleType::kPiZero: return 111;
    case ParticleType::kDeuteron: return 1000010020;
    case ParticleType::kTriton: return 1000010030;
    case ParticleType::kNeutrinoE: return 12;
    case ParticleType::kAntiNeutrinoE: return -12;
    case ParticleType::kNeutrinoMu: return 14;
    case ParticleType::kAntiNeutrinoMu: return -14;
    case ParticleType::kNeutrinoTau: return 16;
    case ParticleType::kAntiNeutrinoTau: return -16;
    case ParticleType::kKaonZeroLong: return 130;
    case ParticleType::kKaonZeroShort: return 310;
    case ParticleType::kLambda: return 3122;
    case ParticleType::kSigmaPlus: return 3222;
    case ParticleType::kSigmaMinus: return 3112;
    case ParticleType::kXiMinus: return 3312;
    case ParticleType::kNumTypes: break;
  }
  return 0;
}

/// The species of a nucleus (Z, A), as every hadronic model in Geant4 resolves one: six light
/// nuclei by name and `G4IonTable::GetIon` for the rest.
///
/// Returns `kGenericIon` for everything heavier, which in this port is a REFUSED species - it
/// has no kernel, so `species_disposition` reports it and the run counts it. That is the honest
/// answer and not a shortcut: `particle_def(kGenericIon)` carries G4GenericIon's own
/// 938.2723 MeV placeholder mass, so transporting a carbon recoil as one would step a nucleus
/// twelve times too light. Whoever gives this port real ion transport replaces this line, not
/// the ParticleDef.
///
/// (Z, A) = (0, 1) is a neutron and (1, 1) a proton, which are not ions at all - the mapping is
/// here rather than at the call sites because a de-excitation product list holds all of them.
__host__ __device__ inline ParticleType particle_type_of_nucleus(int z, int a) {
  if (a == 1 && z == 0) { return ParticleType::kNeutron; }
  if (a == 1 && z == 1) { return ParticleType::kProton; }
  if (a == 2 && z == 1) { return ParticleType::kDeuteron; }
  if (a == 3 && z == 1) { return ParticleType::kTriton; }
  if (a == 3 && z == 2) { return ParticleType::kHe3; }
  if (a == 4 && z == 2) { return ParticleType::kAlpha; }
  return ParticleType::kGenericIon;
}

/// A PDG code back to a species. `kNumTypes` for a code this port has no row for at all, which
/// is distinct from a code it has a row for and refuses to transport - the first is "I do not
/// know what that is" and the second is "I know and cannot".
///
/// The nuclear branch decodes rather than matches, and it ignores the isomer digit: an excited
/// ion and its ground state are the same SPECIES, and the excitation is carried beside the code
/// by whichever module produced it (see `capture/neutron_rad_capture.cuh`).
///
/// IT DOES NOT IGNORE L. `10LZZZAAAI` has a lambda count in it, and Geant4 11.1.1's particle
/// table really does hold hypertriton (1010010030), hyperalpha, hyperH4, doublehyperH4 and
/// doublehyperdoubleneutron - `ref/oracle/decay_applicable.csv` lists all five. Decoding only
/// (Z, A) turns a hypertriton into a triton and a hyperalpha into an alpha, which is a species
/// with the wrong mass and the wrong lifetime rather than an unknown one. P3's excitation
/// handler refuses `nL != 0` by name for the same reason; this is the same refusal at the
/// species boundary, and `tests/test_wiring.cu` found it by asking whether the round trip
/// closes for every code in the oracle rather than for the ones this port emits.
__host__ __device__ inline ParticleType particle_type_of_pdg(int pdg) {
  if (pdg > 1000000000) {
    const int l = (pdg / 10000000) % 100;
    if (l != 0) { return ParticleType::kNumTypes; }
    const int z = (pdg / 10000) % 1000;
    const int a = (pdg / 10) % 1000;
    return particle_type_of_nucleus(z, a);
  }
  switch (pdg) {
    case 22: return ParticleType::kGamma;
    case 11: return ParticleType::kElectron;
    case -11: return ParticleType::kPositron;
    case 13: return ParticleType::kMuonMinus;
    case -13: return ParticleType::kMuonPlus;
    case 211: return ParticleType::kPionPlus;
    case -211: return ParticleType::kPionMinus;
    case 111: return ParticleType::kPiZero;
    case 321: return ParticleType::kKaonPlus;
    case -321: return ParticleType::kKaonMinus;
    case 2212: return ParticleType::kProton;
    case -2212: return ParticleType::kAntiProton;
    case 2112: return ParticleType::kNeutron;
    case 12: return ParticleType::kNeutrinoE;
    case -12: return ParticleType::kAntiNeutrinoE;
    case 14: return ParticleType::kNeutrinoMu;
    case -14: return ParticleType::kAntiNeutrinoMu;
    case 16: return ParticleType::kNeutrinoTau;
    case -16: return ParticleType::kAntiNeutrinoTau;
    case 130: return ParticleType::kKaonZeroLong;
    case 310: return ParticleType::kKaonZeroShort;
    case 3122: return ParticleType::kLambda;
    case 3222: return ParticleType::kSigmaPlus;
    case 3112: return ParticleType::kSigmaMinus;
    case 3312: return ParticleType::kXiMinus;
    default: break;
  }
  return ParticleType::kNumTypes;
}

/// A species whose energy leaves the event without being transported.
///
/// QBBC registers nothing but G4Transportation for a neutrino, so in Geant4 one is created,
/// streamed across the world in a single step, and leaves. Doing that here would cost a track
/// slot, a kernel launch and a boundary search per neutrino to reach a conclusion that is
/// known at emission - so the energy is booked as escaping at the point of creation instead,
/// and no track is made.
///
/// That is an OPTIMISATION of Geant4's answer and not a different answer, but only while the
/// two conditions behind it hold, so they are written down: a neutrino deposits nothing on the
/// way out (no process to deposit with), and nothing downstream reads a neutrino track. The
/// day either changes - a neutrino-nucleus process, or a stepping action that wants to see
/// one - this becomes a species with a kernel like any other, and the accounting below is what
/// has to be removed.
__host__ __device__ inline bool is_neutrino(ParticleType t) {
  return t == ParticleType::kNeutrinoE || t == ParticleType::kAntiNeutrinoE
         || t == ParticleType::kNeutrinoMu || t == ParticleType::kAntiNeutrinoMu
         || t == ParticleType::kNeutrinoTau || t == ParticleType::kAntiNeutrinoTau;
}

/// A hadron with no charge: no continuous energy loss, no multiple scattering, no delta rays.
/// Its step is geometry against a discrete interaction length and nothing else. See
/// step_neutral in physics/stepper.cuh.
__host__ __device__ inline bool is_neutral_hadron(ParticleType t) {
  return t == ParticleType::kNeutron || t == ParticleType::kPiZero;
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
///
/// The deuteron and triton rows are what this function was missing while they were not
/// transported: without them both fell through to `return t`, and hadron_species_of would
/// have sent them to HadronSpecies::kProton anyway by ITS default - the right table for the
/// wrong reason, with hadron_mass_ratio then returning 1 and the lookup done at the deuteron's
/// own energy instead of at E * m_p/m_d. A deuteron's range would have come out as a proton's
/// of the same kinetic energy, which is a factor of about two.
/// Transcribed rather than tabulated, because Geant4's `else` branch is a rule and the rule
/// covers species this port has not met yet. `G4hIonisation::InitialiseEnergyLossProcess`:
///
///     if (pname == "proton" || "anti_proton" || "pi+" || "pi-" || "kaon+" || "kaon-"
///         || "GenericIon" || "alpha")        theBaseParticle = nullptr;
///     else if (GetPDGSpin() == 0.0)          q > 0 ? kaon+ : kaon-
///     else                                   q > 0 ? proton : anti_proton
///
/// - so the choice turns on SPIN and then on the sign of the charge, which is why a deuteron
/// (spin 1) and a triton (spin 1/2) both scale from the proton while a hypothetical spin-0
/// singly-charged hadron would scale from the kaon. mu+- are excluded because G4MuIonisation
/// never sets a base particle, and He3 because G4ionIonisation hands it GenericIon explicitly.
///
/// Checked against `ref/oracle/species_processes.csv`'s `base_particle` column - which is
/// `G4VEnergyLossProcess::BaseParticle()` on the process the constructed QBBC registered - for
/// every species that has an ionisation process at all. It gets sigma+, sigma- and xi- right
/// as a side effect of being the rule rather than a list, and those three are species this
/// port refuses to transport, so the rule is checked further than it is used.
///
/// For a NEUTRAL hadron the answer is unreachable: G4hIonisation is registered by charge, so
/// the neutron and the pi0 have no ionisation process to have a base particle. `q > 0` is false
/// for them and the expression below returns anti_proton or kaon-, which is what Geant4's own
/// expression would return if it were ever evaluated. Neither is read.
__host__ __device__ inline ParticleType hadron_base_particle(ParticleType t) {
  switch (t) {
    // G4ionIonisation's two: alpha has no base, He3's is set explicitly to GenericIon.
    case ParticleType::kAlpha:
    case ParticleType::kGenericIon:
    // G4MuIonisation never sets one.
    case ParticleType::kMuonMinus:
    case ParticleType::kMuonPlus:
    // G4hIonisation's by-name list.
    case ParticleType::kProton:
    case ParticleType::kAntiProton:
    case ParticleType::kPionPlus:
    case ParticleType::kPionMinus:
    case ParticleType::kKaonPlus:
    case ParticleType::kKaonMinus:
      return t;
    case ParticleType::kHe3:
      return ParticleType::kGenericIon;
    default: break;
  }
  const ParticleDef<double> pd = particle_def<double>(t);
  if (pd.spin == 0.0) {
    return (pd.charge > 0.0) ? ParticleType::kKaonPlus : ParticleType::kKaonMinus;
  }
  return (pd.charge > 0.0) ? ParticleType::kProton : ParticleType::kAntiProton;
}

/// True for the particles that go through the heavy-charged-particle ionisation chain
/// (Bragg / Bethe-Bloch) rather than Moller-Bhabha.
///
/// Written out rather than as an index range. It USED to be
/// `t >= kMuonMinus && t < kNumTypes`, which was true of every species in the enum at the time
/// and became wrong the moment a neutral one was appended: a neutron and a neutrino both
/// answered yes. Nothing called it, so nothing broke - which is the only reason this is a note
/// and not an incident. An enum-range predicate is a claim about the ORDER of an enum, and the
/// enum above is append-only precisely because its order is a storage format rather than a
/// classification.
__host__ __device__ inline bool is_heavy_charged(ParticleType t) {
  return t == ParticleType::kMuonMinus || t == ParticleType::kMuonPlus
         || t == ParticleType::kPionPlus || t == ParticleType::kPionMinus
         || t == ParticleType::kKaonPlus || t == ParticleType::kKaonMinus
         || t == ParticleType::kProton || t == ParticleType::kAntiProton
         || t == ParticleType::kAlpha || t == ParticleType::kHe3
         || t == ParticleType::kGenericIon || t == ParticleType::kDeuteron
         || t == ParticleType::kTriton;
}

/// Does QBBC register G4NuclearStopping for this species?
///
/// **No. For none of them.** Which is not what reading G4EmBuilder suggests, and is why this
/// is a predicate answered by the oracle rather than by the source.
///
/// The source reads as a four-way split: `G4EmStandardPhysics::ConstructProcess` makes one
/// `G4NuclearStopping* pnuc` and hands it to GenericIon, then `G4EmBuilder::ConstructCharged`
/// gives it to the proton and `ConstructIonEmPhysics` to the alpha and He3 - passing over the
/// deuteron and triton two lines above them, and never reaching mu+-, pi+-, K+- or pbar at
/// all. That split is real, and it is downstream of a line four functions up:
///
///     G4double nielEnergyLimit = param->MaxNIELEnergy();
///     G4NuclearStopping* pnuc = nullptr;
///     if(nielEnergyLimit > 0.0) { pnuc = new G4NuclearStopping(); ... }
///
/// and `G4EmParameters::Initialise` sets `maxNIELEnergy = 0.0`. So `pnuc` is null, every
/// `if(nullptr != pnuc)` fails, and the process is registered for nobody.
/// `ref/oracle/species_processes.csv` - one row per process on each species' own process
/// manager, from the constructed QBBC - carries no `nuclearStopping` row for any particle,
/// which is the fact rather than the derivation of it.
///
/// step_hadron applied nuclear stopping unconditionally, so this port has been running a
/// process the reference does not. It is small - G4ICRU49NuclearStoppingModel returns zero
/// above about z1^2 MeV per nucleon, so it touches only the last microns of a track, and the
/// energy it removed was deposited locally in the same volume it would otherwise have been
/// deposited in by the dying track. What it changed is the STEP the deposit happened on and
/// the non-ionising share of it, not the total. The measured size is in the commit that made
/// this false; see docs/RISK.md.
///
/// Kept as a predicate rather than deleted, because it is a physics-list parameter and not a
/// property of Geant4: `/process/em/setMaxNIEL <E>` turns it on, and a list that does so wants
/// exactly the four-species split the source describes. One place to change.
__host__ __device__ inline bool uses_nuclear_stopping(ParticleType /*t*/) {
  return false;
}

/// Is WentzelVI the multiple-scattering model Geant4 gives this species, rather than Urban?
///
/// Another table rather than a rule, and it splits the charged hadrons in a place no property
/// of the particle would. `G4EmStandardPhysics::ConstructProcess` makes ONE
/// `G4hMultipleScattering("ionmsc")` with **no model set**, and G4hMultipleScattering's
/// default model is Urban. Then:
///
///     mu+-                G4MuMultipleScattering + SetEmModel(G4WentzelVIModel)
///     pi+-, K+-, p, pbar  G4EmBuilder::ConstructLightHadrons - a fresh
///                         G4hMultipleScattering + SetEmModel(G4WentzelVIModel)
///     deuteron, triton    the shared "ionmsc", no model     -> URBAN
///     alpha, He3          a fresh G4hMultipleScattering(), no model -> URBAN
///     GenericIon          the shared "ionmsc"                -> URBAN
///
/// So the deuteron and the triton scatter by Urban in Geant4, sitting two lines above the
/// alpha in the same function, and the pion - lighter, same charge - scatters by WentzelVI.
///
/// THE `false` ANSWER IS NOW A DISPATCH AND NOT A SUBSTITUTION (P14b). It used to be the
/// latter: `em/urban_msc.cuh`'s stepping half was the electron's - `is_positron` in its step
/// limit and its sampler, and an e-/e+ transport-mfp table - so an ion could not be run
/// through it and `step_hadron` ran WentzelVI for these five species too. That model is now
/// general across mass and charge and `step_hadron` branches on this predicate, at compile
/// time, because `type` is a template parameter of `run_step_hadron`. What the substitution
/// was worth is in docs/RISK.md V61.
///
/// The predicate is checked against Geant4 rather than against the source it was read from:
/// `ref/oracle/species_processes.csv`'s `models` column is model 0's name off each species'
/// own process manager in a constructed QBBC, and `tests/test_species.cu` compares this
/// function against it species by species. It reads `UrbanMsc` for alpha, He3, deuteron,
/// triton and GenericIon and `WentzelVIUni` for the eight below.
__host__ __device__ inline bool uses_wentzel_msc(ParticleType t) {
  return t == ParticleType::kMuonMinus || t == ParticleType::kMuonPlus
         || t == ParticleType::kPionPlus || t == ParticleType::kPionMinus
         || t == ParticleType::kKaonPlus || t == ParticleType::kKaonMinus
         || t == ParticleType::kProton || t == ParticleType::kAntiProton;
}

}  // namespace g4gpu
