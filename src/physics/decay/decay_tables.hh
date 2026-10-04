// Decay tables for the unstable species QBBC transports, exactly as Geant4 11.1.1's own
// particle definitions build them.
//
// This is DATA, and it is data because the tables are not a physics model: they are the
// literal contents of `G4MuonPlus::Definition()`, `G4PionPlus::Definition()` and friends,
// which each `new` a `G4DecayTable` and insert channels with hard-coded branching ratios.
// Every row below names the file it came from. ref/dump/dump_decay.cc reads the same tables
// out of a running Geant4 and tests/test_decay.cu diffs them, so a Geant4 release that
// changes a branching ratio shows up as a failing row rather than as a shower that is subtly
// the wrong shape.
//
// Two things about the ORDER of the channels, because both are load-bearing.
//
//   `G4DecayTable::Insert` is an insertion sort: a new channel goes before the first existing
//   channel with a STRICTLY smaller BR, else at the end. So the stored order is descending BR
//   with insertion order preserved among equals - which is not the order the definitions
//   write the channels in. G4KaonPlus.cc inserts 0.6355, 0.2066, 0.0559, 0.01761, 0.0507,
//   0.0335 and the table ends up 0.6355, 0.2066, 0.0559, 0.0507, 0.0335, 0.01761: Ke3 and
//   Kmu3 move ahead of pi+ pi0 pi0. The tables below are in the SORTED order, and the dump
//   compares channel by channel BY INDEX so that a wrong order fails.
//
//   `SelectADecayChannel` walks the table accumulating BRs and stops at the first one past
//   `sumBR * rand`, so which channel a given uniform selects depends on that order. The order
//   is therefore part of the physics, not a presentation detail.
//
// The branching ratios do NOT sum to 1. K+/K- sum to 0.99981 and SelectADecayChannel
// normalises by the sum of the channels that pass `IsOKWithParentMass`, so the effective
// ratios are the tabulated ones divided by 0.99981. That is Geant4's behaviour and it is
// reproduced rather than corrected.
//
// EVERY MASS AND WIDTH IS WRITTEN AS THE GEANT4 SOURCE WRITES IT, product and all -
// `0.493677 * gev()` and not `493.677`. Those are not the same double: 0.493677*1000 rounds
// to 493.67699999999996 and the decimal literal 493.677 to 493.67700000000002, one ulp
// apart. The kaon rows said 493.677 and the table comparison reported a worst relative
// deviation of 1.15e-16 - inside its 1e-15 tolerance, so it passed, and it was the only
// nonzero number in the whole block. Multiplying as Geant4 multiplies takes it to zero.
//
// What is NOT here is refused by PDG code, not approximated: see `decay_refusal` below. The
// refusal is structural: a PDG code with no row gets `kNoTable` and a message that names the
// code, at the point the decay would have happened.
//
// P23b ADDED THE STRANGE AND THE HEAVY ONES, for the transport P23a will give them: K0S and K0L
// (whose four semileptonic channels are G4KL3DecayChannel with K0L's OWN form factors), the
// seven anti-hyperons, and the thirty b/c hadrons of G4HadParticles::sBCHadrons together with
// etac, J/psi and Upsilon. The last group is NOT from the particle definitions - those pass a
// null decay table - but from `G4HadronicBuilder::BuildDecayTableForBCHadrons`, which QBBC
// reaches through `BuildBCHadronsFTFP_BERT` because its constructor sets EnableBCParticles: one
// G4PhaseSpaceDecayChannel per species with BR 1.0, "dummy" by that function's own comment, and
// the only four-body channels in this file (lambda_b and its antiparticle). The eta's table
// came with them because etac -> eta pi+ pi- puts one on the stack. ref/dump/dump_decay.cc
// dumps every one of these species from the running Geant4, so each row is an oracle row.
//
// P10c ADDED THE SEVEN PARTICLE HYPERONS, and the reason is worth stating because it is not
// "transport now needs them". `G4IntraNucleiCascader::decayTrappedParticle` decays a hyperon the
// Bertini cascade has trapped below the nuclear potential, inside the nucleus, and puts the
// daughters back on the cascade stack; that caller is in a different package and needs the table
// whether or not anything transports a free lambda. The rows are read the same way as every
// other row here - out of the species' own G4*.cc - and dumped by ref/dump/dump_decay.cc, so
// they are oracle rows and not an extension on trust. All eleven channels are
// G4PhaseSpaceDecayChannel with two daughters, so no new sampler came with them.
#pragma once
#include <cstdint>

namespace g4gpu::decay {

// ---------------------------------------------------------------------------------------
// PDG codes. Geant4's own encodings, from the `PDG encoding` argument of each
// G4ParticleDefinition constructor.
// ---------------------------------------------------------------------------------------
enum : int {
  kPdgGamma = 22,
  kPdgElectron = 11,
  kPdgPositron = -11,
  kPdgNuE = 12,
  kPdgAntiNuE = -12,
  kPdgNuMu = 14,
  kPdgAntiNuMu = -14,
  kPdgMuMinus = 13,
  kPdgMuPlus = -13,
  kPdgPiPlus = 211,
  kPdgPiMinus = -211,
  kPdgPiZero = 111,
  kPdgKaonPlus = 321,
  kPdgKaonMinus = -321,
  kPdgNeutron = 2112,
  kPdgProton = 2212,
  // The seven hyperons, added by P10c. They are here not because QBBC transports them - it does,
  // through G4HadronicBuilder::BuildFTFP_BERT, but that is not what needs them - but because
  // `G4IntraNucleiCascader::decayTrappedParticle` decays a hyperon that the Bertini cascade
  // trapped below the nuclear potential, INSIDE the nucleus, and feeds the daughters back onto
  // the cascade stack. Without these rows that path had to be refused, and the refusal cost 18%
  // of a K- event and 47% of a lambda one (docs/RISK.md V135). Every row is from the species'
  // own G4*.cc, and ref/dump/dump_decay.cc dumps them so each is an oracle row.
  kPdgLambda = 3122,
  kPdgSigmaPlus = 3222,
  kPdgSigmaZero = 3212,
  kPdgSigmaMinus = 3112,
  kPdgXiZero = 3322,
  kPdgXiMinus = 3312,
  kPdgOmegaMinus = 3334,
  // ---- P23b: the neutral kaons, the anti-baryons their tables and the anti-hyperons' need,
  // the anti-hyperons, and the b/c hadrons with every daughter BuildDecayTableForBCHadrons
  // names.
  kPdgKaonZeroLong = 130,
  kPdgKaonZeroShort = 310,
  kPdgAntiProton = -2212,
  kPdgAntiNeutron = -2112,
  kPdgAntiLambda = -3122,
  kPdgAntiSigmaPlus = -3222,
  kPdgAntiSigmaZero = -3212,
  kPdgAntiSigmaMinus = -3112,
  kPdgAntiXiZero = -3322,
  kPdgAntiXiMinus = -3312,
  kPdgAntiOmegaMinus = -3334,
  kPdgEta = 221,
  kPdgEtaPrime = 331,
  kPdgDPlus = 411,
  kPdgDMinus = -411,
  kPdgDZero = 421,
  kPdgAntiDZero = -421,
  kPdgDsPlus = 431,
  kPdgDsMinus = -431,
  kPdgBPlus = 521,
  kPdgBMinus = -521,
  kPdgBZero = 511,
  kPdgAntiBZero = -511,
  kPdgBsZero = 531,
  kPdgAntiBsZero = -531,
  kPdgBcPlus = 541,
  kPdgBcMinus = -541,
  kPdgLambdacPlus = 4122,
  kPdgAntiLambdacPlus = -4122,
  kPdgXicPlus = 4232,
  kPdgAntiXicPlus = -4232,
  kPdgXicZero = 4132,
  kPdgAntiXicZero = -4132,
  kPdgOmegacZero = 4332,
  kPdgAntiOmegacZero = -4332,
  kPdgLambdab = 5122,
  kPdgAntiLambdab = -5122,
  kPdgXibZero = 5232,
  kPdgAntiXibZero = -5232,
  kPdgXibMinus = 5132,
  kPdgAntiXibMinus = -5132,
  kPdgOmegabMinus = 5332,
  kPdgAntiOmegabMinus = -5332,
  kPdgEtac = 441,
  kPdgJPsi = 443,
  kPdgUpsilon = 553,
  // The charmed and bottom sigmas G4HadParticles::sBCHadrons leaves out "because they decay
  // very quickly" - which in 11.1.1 means lifetime 0.0 and NOT short-lived, so they are tracks
  // that G4Decay decays where they are made, with their definitions' own tables.
  kPdgSigmacPlusPlus = 4222,
  kPdgSigmacPlus = 4212,
  kPdgSigmacZero = 4112,
  kPdgSigmabPlus = 5222,
  kPdgSigmabZero = 5212,
  kPdgSigmabMinus = 5112,
  kPdgAntiSigmacPlusPlus = -4222,
  kPdgAntiSigmacPlus = -4212,
  kPdgAntiSigmacZero = -4112,
  kPdgAntiSigmabPlus = -5222,
  kPdgAntiSigmabZero = -5212,
  kPdgAntiSigmabMinus = -5112,
};

/// CLHEP's unit prefixes, so that a mass can be written exactly as its G4ParticleDefinition
/// constructor writes it. `megaelectronvolt = 1`, `gigaelectronvolt = 1e3*MeV`,
/// `nanosecond = 1`, `second = 1e9*ns` (CLHEP SystemOfUnits.h), which is also what
/// core/units.cuh pins - repeated here because this file is included by device code that has
/// no other reason to pull in the unit system.
__host__ __device__ inline constexpr double gev() { return 1.0e3; }
__host__ __device__ inline constexpr double mev() { return 1.0; }
/// CLHEP's `kiloelectronvolt = 1.e-3*megaelectronvolt`. G4Eta.cc and G4Upsilon.cc give their
/// widths in keV (`1.31*keV`, `54.00*keV`), and `54.00*1.e-3` is not the double `0.054`.
__host__ __device__ inline constexpr double kev() { return 1.0e-3; }
__host__ __device__ inline constexpr double nanosecond() { return 1.0; }
__host__ __device__ inline constexpr double second() { return 1.0e9 * nanosecond(); }

/// hbar, MeV ns. CLHEP defines `hbar_Planck = h_Planck/twopi` and `hbarc = hbar_Planck *
/// c_light`, so dividing 11.1.1's own hbarc by its own c_light returns hbar_Planck - and it
/// returns it BIT-IDENTICALLY, difference exactly zero, which was measured rather than hoped
/// for. It is derived here rather than typed because docs/RISK.md V8 is what a hand-typed
/// constant cost. Only pi0 needs it, and it needs it because G4PionZero.cc throws away the
/// lifetime it was constructed with and recomputes it from the width.
__host__ __device__ inline constexpr double hbar_planck_MeV_ns() {
  return 1.9732698045930245e-10 / (2.99792458e8 * 1000.0 / 1e9);
}

// ---------------------------------------------------------------------------------------
// Particle rows. mass and width in MeV, lifetime in ns.
//
// `stable` is the G4ParticleDefinition `stable` flag, and it is NOT `lifetime < 0`. The
// triton is flagged stable AND carries a 17.774 year lifetime, and G4Decay reads the two
// differently: `IsApplicable` looks at the lifetime (so the triton gets a G4Decay process),
// while `GetMeanFreePath`, `GetMeanLifeTime` and `DecayIt` all look at the flag (so it never
// decays). Both are kept so the port can answer either question.
// ---------------------------------------------------------------------------------------
struct ParticleRow {
  int pdg;
  double mass;      ///< MeV, G4ParticleDefinition's PDG mass
  double width;     ///< MeV, G4ParticleDefinition's PDG width
  double lifetime;  ///< ns, G4ParticleDefinition::GetPDGLifeTime(); -1 for the stable ones
  bool stable;      ///< G4ParticleDefinition::GetPDGStable()
  const char* name; ///< Geant4's own particle name, for messages and for the oracle join
};

/// Every particle that appears as a parent or a daughter in the tables below, from its own
/// G4*.cc definition file in 11.1.1. Nothing else is here - a code that is not in this list
/// is refused, which is how K0L and the hyperons are refused.
///
/// G4Electron.cc/G4Positron.cc pass `electron_mass_c2`, which is CLHEP's 0.510998910 MeV in
/// 11.1.1; G4Neutron.cc passes `neutron_mass_c2`, CLHEP's 939.56536 MeV. Both are the values
/// src/core/units.cuh pins and tests/test_constants.cu checks, and they are repeated here
/// rather than included so that this file stays pure data.
///
/// A function returning a function-local static rather than a namespace-scope array, because
/// nvcc cannot see a namespace-scope `constexpr` object from device code - the same idiom
/// data/barashenkov.hh uses, and for the same reason.
constexpr int kNumParticles = 81;   // 16, the seven hyperons P10c added, and P23b's 58

__host__ __device__ inline const ParticleRow* particle_rows() {
  static const ParticleRow v[kNumParticles] = {
    // G4Gamma.cc
    {kPdgGamma, 0.0, 0.0, -1.0, true, "gamma"},
    // G4Electron.cc, G4Positron.cc
    {kPdgElectron, 0.510998910, 0.0, -1.0, true, "e-"},
    {kPdgPositron, 0.510998910, 0.0, -1.0, true, "e+"},
    // G4NeutrinoE.cc, G4AntiNeutrinoE.cc, G4NeutrinoMu.cc, G4AntiNeutrinoMu.cc - massless,
    // widthless, stable. QBBC does not transport them; P1 counts them as energy carried away.
    {kPdgNuE, 0.0, 0.0, -1.0, true, "nu_e"},
    {kPdgAntiNuE, 0.0, 0.0, -1.0, true, "anti_nu_e"},
    {kPdgNuMu, 0.0, 0.0, -1.0, true, "nu_mu"},
    {kPdgAntiNuMu, 0.0, 0.0, -1.0, true, "anti_nu_mu"},
    // G4MuonMinus.cc / G4MuonPlus.cc: 0.1056583715*GeV, 2.99598e-16*MeV, 2196.98*ns
    {kPdgMuMinus, 0.1056583715 * gev(), 2.99598e-16 * mev(), 2196.98 * nanosecond(), false,
     "mu-"},
    {kPdgMuPlus, 0.1056583715 * gev(), 2.99598e-16 * mev(), 2196.98 * nanosecond(), false,
     "mu+"},
    // G4PionPlus.cc / G4PionMinus.cc: 0.1395701*GeV, 2.5284e-14*MeV, 26.033*ns
    {kPdgPiPlus, 0.1395701 * gev(), 2.5284e-14 * mev(), 26.033 * nanosecond(), false, "pi+"},
    {kPdgPiMinus, 0.1395701 * gev(), 2.5284e-14 * mev(), 26.033 * nanosecond(), false, "pi-"},
    // G4PionZero.cc: 0.1349766*GeV, 7.73e-06*MeV, and then
    //   anInstance->SetPDGLifeTime( hbar_Planck/(anInstance->GetPDGWidth()) );
    // which OVERWRITES the 8.52e-8*ns the constructor was given. The computed value is
    // 8.51503178e-8 ns.
    {kPdgPiZero, 0.1349766 * gev(), 7.73e-06 * mev(),
     hbar_planck_MeV_ns() / (7.73e-06 * mev()), false, "pi0"},
    // G4KaonPlus.cc / G4KaonMinus.cc: 0.493677*GeV, 5.317e-14*MeV, 12.380*ns
    {kPdgKaonPlus, 0.493677 * gev(), 5.317e-14 * mev(), 12.380 * nanosecond(), false, "kaon+"},
    {kPdgKaonMinus, 0.493677 * gev(), 5.317e-14 * mev(), 12.380 * nanosecond(), false,
     "kaon-"},
    // G4Neutron.cc: neutron_mass_c2, 7.478e-28*GeV, 880.2*second. Unstable, and QBBC does
    // attach G4Decay to it - but G4NeutronTrackingCut kills a neutron at 10 us, eleven orders
    // below 880 s, so the channel below has never fired in a QBBC run and still has to exist
    // for the process to answer GetMeanFreePath.
    // 7.478e-28 GeV is 7.478e-25 MeV: this row said 7.478e-19 for an afternoon, which is the
    // GeV-to-MeV factor applied the wrong way, and the table comparison is what found it.
    // Written as the product now, so the factor cannot be applied by hand at all.
    {kPdgNeutron, 939.56536, 7.478e-28 * gev(), 880.2 * second(), false, "neutron"},
    // G4Proton.cc: stable, lifetime -1. Present as the neutron beta decay daughter.
    {kPdgProton, 938.272013, 0.0, -1.0, true, "proton"},
    // ---- the hyperons, for G4IntraNucleiCascader::decayTrappedParticle (P10c) ----
    // G4Lambda.cc: 1.115683*GeV, 2.501e-12*MeV, 0.2631*ns
    {kPdgLambda, 1.115683 * gev(), 2.501e-12 * mev(), 0.2631 * nanosecond(), false, "lambda"},
    // G4SigmaPlus.cc: 1.18937*GeV, 8.209e-12*MeV, 0.08018*ns
    {kPdgSigmaPlus, 1.18937 * gev(), 8.209e-12 * mev(), 0.08018 * nanosecond(), false,
     "sigma+"},
    // G4SigmaZero.cc: 1.192642*GeV, 8.9e-3*MeV, and then - exactly as G4PionZero.cc does -
    //   anInstance->SetPDGLifeTime( hbar_Planck/(anInstance->GetPDGWidth()) );
    // which OVERWRITES the 7.4e-11*ns the constructor was given. The sigma0's width is eight
    // orders above every other hyperon's because it decays electromagnetically, and that is
    // what makes the overwrite matter here rather than being a curiosity: 7.4e-11 ns and the
    // computed 7.3989e-11 ns differ in the fourth digit, not the tenth.
    {kPdgSigmaZero, 1.192642 * gev(), 8.9e-3 * mev(),
     hbar_planck_MeV_ns() / (8.9e-3 * mev()), false, "sigma0"},
    // G4SigmaMinus.cc: 1.197449*GeV, 4.45e-12*MeV, 0.1479*ns
    {kPdgSigmaMinus, 1.197449 * gev(), 4.45e-12 * mev(), 0.1479 * nanosecond(), false,
     "sigma-"},
    // G4XiZero.cc: 1.31486*GeV, 2.27e-12*MeV, 0.290*ns
    {kPdgXiZero, 1.31486 * gev(), 2.27e-12 * mev(), 0.290 * nanosecond(), false, "xi0"},
    // G4XiMinus.cc: 1.32171*GeV, 4.02e-12*MeV, 0.1639*ns
    {kPdgXiMinus, 1.32171 * gev(), 4.02e-12 * mev(), 0.1639 * nanosecond(), false, "xi-"},
    // G4OmegaMinus.cc: 1.67245*GeV, 8.07e-12*MeV, 0.0821*ns
    {kPdgOmegaMinus, 1.67245 * gev(), 8.07e-12 * mev(), 0.0821 * nanosecond(), false,
     "omega-"},
    // ---- P23b. Appended, never inserted: the original 23 stay first because particle_index
    // is a linear scan and the pions, muons, electrons and kaons it is asked about on every
    // transported decay must keep their short walk.
    //
    // G4KaonZeroLong.cc / G4KaonZeroShort.cc: the same 0.497614*GeV, and lifetimes 51.16 ns and
    // 0.08954 ns - a factor of 571, which is the whole reason they are two species.
    {kPdgKaonZeroLong, 0.497614 * gev(), 1.287e-14 * mev(), 51.16 * nanosecond(), false,
     "kaon0L"},
    {kPdgKaonZeroShort, 0.497614 * gev(), 7.3508e-12 * mev(), 0.08954 * nanosecond(), false,
     "kaon0S"},
    // G4AntiProton.cc: proton_mass_c2, 0.0*MeV, stable, lifetime -1.0.
    {kPdgAntiProton, 938.272013, 0.0, -1.0, true, "anti_proton"},
    // G4AntiNeutron.cc: neutron_mass_c2, 7.478e-28*GeV, 880.2*second - the neutron's numbers.
    // A daughter of the anti-lambda, the anti-sigma+ and the anti-sigma-. Its OWN table (a
    // G4NeutronBetaDecayChannel on "anti_neutron") is the anti-light-ion package's, P23c, and
    // is refused by name below until it lands.
    {kPdgAntiNeutron, 939.56536, 7.478e-28 * gev(), 880.2 * second(), false, "anti_neutron"},
    // The seven anti-hyperons, each from its G4Anti*.cc, with its particle's numbers.
    {kPdgAntiLambda, 1.115683 * gev(), 2.501e-12 * mev(), 0.2631 * nanosecond(), false,
     "anti_lambda"},
    {kPdgAntiSigmaPlus, 1.18937 * gev(), 8.209e-12 * mev(), 0.08018 * nanosecond(), false,
     "anti_sigma+"},
    // G4AntiSigmaZero.cc overwrites its 7.4e-11 ns with hbar_Planck/width, as G4SigmaZero.cc does.
    {kPdgAntiSigmaZero, 1.192642 * gev(), 8.9e-3 * mev(),
     hbar_planck_MeV_ns() / (8.9e-3 * mev()), false, "anti_sigma0"},
    {kPdgAntiSigmaMinus, 1.197449 * gev(), 4.45e-12 * mev(), 0.1479 * nanosecond(), false,
     "anti_sigma-"},
    {kPdgAntiXiZero, 1.31486 * gev(), 2.27e-12 * mev(), 0.290 * nanosecond(), false,
     "anti_xi0"},
    {kPdgAntiXiMinus, 1.32171 * gev(), 4.02e-12 * mev(), 0.1639 * nanosecond(), false,
     "anti_xi-"},
    {kPdgAntiOmegaMinus, 1.67245 * gev(), 8.07e-12 * mev(), 0.0821 * nanosecond(), false,
     "anti_omega-"},
    // G4Eta.cc: 0.547862*GeV, 1.31*keV, 0.0*ns. G4EtaPrime.cc: 0.95778*GeV, 0.196*MeV, 0.0*ns.
    // Lifetime ZERO, not unknown: G4Decay::GetMeanFreePath answers DBL_MIN, so both decay where
    // they are made. Daughters of etac and Upsilon here.
    {kPdgEta, 0.547862 * gev(), 1.31 * kev(), 0.0 * nanosecond(), false, "eta"},
    {kPdgEtaPrime, 0.95778 * gev(), 0.196 * mev(), 0.0 * nanosecond(), false, "eta_prime"},
    // The charmed mesons: G4DMesonPlus.cc, G4DMesonMinus.cc, G4DMesonZero.cc,
    // G4AntiDMesonZero.cc, G4DsMesonPlus.cc, G4DsMesonMinus.cc.
    {kPdgDPlus, 1.86958 * gev(), 6.33e-10 * mev(), 1.040e-3 * nanosecond(), false, "D+"},
    {kPdgDMinus, 1.86958 * gev(), 6.33e-10 * mev(), 1.040e-3 * nanosecond(), false, "D-"},
    {kPdgDZero, 1.86484 * gev(), 1.605e-9 * mev(), 0.4101e-3 * nanosecond(), false, "D0"},
    {kPdgAntiDZero, 1.86484 * gev(), 1.605e-9 * mev(), 0.4101e-3 * nanosecond(), false,
     "anti_D0"},
    {kPdgDsPlus, 1.96827 * gev(), 1.317e-9 * mev(), 0.500e-3 * nanosecond(), false, "Ds+"},
    {kPdgDsMinus, 1.96827 * gev(), 1.317e-9 * mev(), 0.500e-3 * nanosecond(), false, "Ds-"},
    // The bottom mesons: G4BMesonPlus.cc, G4BMesonMinus.cc, G4BMesonZero.cc,
    // G4AntiBMesonZero.cc, G4BsMesonZero.cc, G4AntiBsMesonZero.cc, G4BcMesonPlus.cc,
    // G4BcMesonMinus.cc.
    {kPdgBPlus, 5.27929 * gev(), 4.018e-10 * mev(), 1.638e-3 * nanosecond(), false, "B+"},
    {kPdgBMinus, 5.27929 * gev(), 4.018e-10 * mev(), 1.638e-3 * nanosecond(), false, "B-"},
    {kPdgBZero, 5.27961 * gev(), 4.33e-10 * mev(), 1.520e-3 * nanosecond(), false, "B0"},
    {kPdgAntiBZero, 5.27961 * gev(), 4.33e-10 * mev(), 1.520e-3 * nanosecond(), false,
     "anti_B0"},
    {kPdgBsZero, 5.36689 * gev(), 4.374e-10 * mev(), 1.505e-3 * nanosecond(), false, "Bs0"},
    {kPdgAntiBsZero, 5.36689 * gev(), 4.374e-10 * mev(), 1.505e-3 * nanosecond(), false,
     "anti_Bs0"},
    {kPdgBcPlus, 6.2749 * gev(), 1.298e-9 * mev(), 0.507e-3 * nanosecond(), false, "Bc+"},
    {kPdgBcMinus, 6.2749 * gev(), 1.298e-9 * mev(), 0.507e-3 * nanosecond(), false, "Bc-"},
    // The charmed baryons: G4LambdacPlus.cc, G4AntiLambdacPlus.cc, G4XicPlus.cc,
    // G4AntiXicPlus.cc, G4XicZero.cc, G4AntiXicZero.cc, G4OmegacZero.cc, G4AntiOmegacZero.cc.
    {kPdgLambdacPlus, 2.28646 * gev(), 3.30e-9 * mev(), 0.200e-3 * nanosecond(), false,
     "lambda_c+"},
    {kPdgAntiLambdacPlus, 2.28646 * gev(), 3.30e-9 * mev(), 0.200e-3 * nanosecond(), false,
     "anti_lambda_c+"},
    {kPdgXicPlus, 2.46787 * gev(), 1.49e-9 * mev(), 0.442e-3 * nanosecond(), false, "xi_c+"},
    {kPdgAntiXicPlus, 2.46787 * gev(), 1.49e-9 * mev(), 0.442e-3 * nanosecond(), false,
     "anti_xi_c+"},
    {kPdgXicZero, 2.47085 * gev(), 5.9e-9 * mev(), 0.112e-3 * nanosecond(), false, "xi_c0"},
    {kPdgAntiXicZero, 2.47085 * gev(), 5.9e-9 * mev(), 0.112e-3 * nanosecond(), false,
     "anti_xi_c0"},
    {kPdgOmegacZero, 2.6952 * gev(), 9.6e-9 * mev(), 0.0268e-3 * nanosecond(), false,
     "omega_c0"},
    {kPdgAntiOmegacZero, 2.6952 * gev(), 9.6e-9 * mev(), 0.0268e-3 * nanosecond(), false,
     "anti_omega_c0"},
    // The bottom baryons: G4Lambdab.cc, G4AntiLambdab.cc, G4XibZero.cc, G4AntiXibZero.cc,
    // G4XibMinus.cc, G4AntiXibMinus.cc, G4OmegabMinus.cc, G4AntiOmegabMinus.cc.
    {kPdgLambdab, 5.61958 * gev(), 0.4478e-9 * mev(), 1.470e-3 * nanosecond(), false,
     "lambda_b"},
    {kPdgAntiLambdab, 5.61958 * gev(), 0.4478e-9 * mev(), 1.470e-3 * nanosecond(), false,
     "anti_lambda_b"},
    {kPdgXibZero, 5.7919 * gev(), 0.445e-9 * mev(), 1.479e-3 * nanosecond(), false, "xi_b0"},
    {kPdgAntiXibZero, 5.7919 * gev(), 0.445e-9 * mev(), 1.479e-3 * nanosecond(), false,
     "anti_xi_b0"},
    {kPdgXibMinus, 5.7945 * gev(), 0.419e-9 * mev(), 1.571e-3 * nanosecond(), false, "xi_b-"},
    {kPdgAntiXibMinus, 5.7945 * gev(), 0.419e-9 * mev(), 1.571e-3 * nanosecond(), false,
     "anti_xi_b-"},
    {kPdgOmegabMinus, 6.0461 * gev(), 0.40e-9 * mev(), 1.640e-3 * nanosecond(), false,
     "omega_b-"},
    {kPdgAntiOmegabMinus, 6.0461 * gev(), 0.40e-9 * mev(), 1.640e-3 * nanosecond(), false,
     "anti_omega_b-"},
    // The three quarkonia BuildDecayTableForBCHadrons adds by name after its loop: G4Etac.cc
    // 2.9834*GeV and 31.8*MeV, G4JPsi.cc 3.096916*GeV and 0.0929*MeV, G4Upsilon.cc 9.46030*GeV
    // and 54.00*keV - all three with lifetime 0.0*ns, like the eta.
    {kPdgEtac, 2.9834 * gev(), 31.8 * mev(), 0.0 * nanosecond(), false, "etac"},
    {kPdgJPsi, 3.096916 * gev(), 0.0929 * mev(), 0.0 * nanosecond(), false, "J/psi"},
    {kPdgUpsilon, 9.46030 * gev(), 54.00 * kev(), 0.0 * nanosecond(), false, "Upsilon"},
    // The charmed and bottom sigmas and their antiparticles: G4SigmacPlusPlus.cc,
    // G4SigmacPlus.cc, G4SigmacZero.cc, G4SigmabPlus.cc, G4SigmabZero.cc, G4SigmabMinus.cc and
    // the six G4Anti*.cc, lifetime 0.0*ns each, the antiparticle with its particle's numbers.
    {kPdgSigmacPlusPlus, 2.45397 * gev(), 1.89 * mev(), 0.0 * nanosecond(), false,
     "sigma_c++"},
    {kPdgSigmacPlus, 2.4529 * gev(), 2.2 * mev(), 0.0 * nanosecond(), false, "sigma_c+"},
    {kPdgSigmacZero, 2.45375 * gev(), 1.83 * mev(), 0.0 * nanosecond(), false, "sigma_c0"},
    {kPdgSigmabPlus, 5.8113 * gev(), 9.7 * mev(), 0.0 * nanosecond(), false, "sigma_b+"},
    {kPdgSigmabZero, 5.8078 * gev(), 5.0 * mev(), 0.0 * nanosecond(), false, "sigma_b0"},
    {kPdgSigmabMinus, 5.8155 * gev(), 4.9 * mev(), 0.0 * nanosecond(), false, "sigma_b-"},
    {kPdgAntiSigmacPlusPlus, 2.45397 * gev(), 1.89 * mev(), 0.0 * nanosecond(), false,
     "anti_sigma_c++"},
    {kPdgAntiSigmacPlus, 2.4529 * gev(), 2.2 * mev(), 0.0 * nanosecond(), false,
     "anti_sigma_c+"},
    {kPdgAntiSigmacZero, 2.45375 * gev(), 1.83 * mev(), 0.0 * nanosecond(), false,
     "anti_sigma_c0"},
    {kPdgAntiSigmabPlus, 5.8113 * gev(), 9.7 * mev(), 0.0 * nanosecond(), false,
     "anti_sigma_b+"},
    {kPdgAntiSigmabZero, 5.8078 * gev(), 5.0 * mev(), 0.0 * nanosecond(), false,
     "anti_sigma_b0"},
    {kPdgAntiSigmabMinus, 5.8155 * gev(), 4.9 * mev(), 0.0 * nanosecond(), false,
     "anti_sigma_b-"},
  };
  return v;
}

/// Index into particle_rows(), or -1. A linear scan: this is called once per decay and once per
/// daughter, and for the species transported before P23b - all in the first sixteen rows -
/// the walk is as short as it always was. A b/c daughter costs up to 81 compares, once per
/// decay of a species FTF makes once in four thousand string splits.
__host__ __device__ inline int particle_index(int pdg) {
  const ParticleRow* p = particle_rows();
  for (int i = 0; i < kNumParticles; ++i) {
    if (p[i].pdg == pdg) { return i; }
  }
  return -1;
}

__host__ __device__ inline double particle_mass(int pdg) {
  const int i = particle_index(pdg);
  return (i >= 0) ? particle_rows()[i].mass : -1.0;
}

__host__ __device__ inline double particle_width(int pdg) {
  const int i = particle_index(pdg);
  return (i >= 0) ? particle_rows()[i].width : -1.0;
}

// ---------------------------------------------------------------------------------------
// Channels
// ---------------------------------------------------------------------------------------

/// Which G4VDecayChannel subclass the definition installed. The kind is not derivable from
/// the daughter list: pi0 -> gamma e+ e- is a G4DalitzDecayChannel and K+ -> pi0 e+ nu_e is a
/// G4KL3DecayChannel, and both are three-body with a lepton pair. It is transcribed from the
/// `new G4...DecayChannel(...)` line, and the dump reports G4VDecayChannel::GetKinematicsName()
/// so the mapping is checked and not assumed.
enum class ChannelKind : int {
  kPhaseSpace = 0,   ///< G4PhaseSpaceDecayChannel, "Phase Space"
  kMuonDecay = 1,    ///< G4MuonDecayChannel, "Muon Decay"
  kKL3 = 2,          ///< G4KL3DecayChannel, "KL3 Decay"
  kDalitz = 3,       ///< G4DalitzDecayChannel, "Dalitz Decay"
  kNeutronBeta = 4,  ///< G4NeutronBetaDecayChannel, "Neutron Decay"
};

struct ChannelRow {
  ChannelKind kind;
  double br;         ///< G4VDecayChannel::GetBR(), as the definition passed it
  int n_daughters;
  /// PDG codes, in G4VDecayChannel's own daughter order. Five slots because that is
  /// G4VDecayChannel's constructor arity and the bound on `givenDaughterMasses`; every row
  /// below fills two or three and leaves the rest zero, which is why the aggregate
  /// initialisers are short. Four and five are reachable only through
  /// G4PhaseSpaceDecayChannel::ManyBodyDecayIt, which no real table needs and which is
  /// therefore validated against a synthetic channel - see decay_channels.cuh.
  int daughter[5];
  /// G4KL3DecayChannel's two form-factor parameters, chosen in its constructor by (parent,
  /// lepton) pair. Zero for every other kind; read only when kind == kKL3.
  double kl3_lambda;
  double kl3_xi0;
};

/// One parent's table: a contiguous slice of kChannels.
struct DecayTableRow {
  int parent_pdg;
  int first;  ///< index into kChannels
  int count;
};

/// Every channel, grouped by parent, each group in G4DecayTable::Insert's descending-BR
/// order. The comment on each row is the line in the Geant4 definition file it came from.
constexpr int kNumChannels = 98;   // 19, the eleven hyperon channels P10c added, P23b's 68

__host__ __device__ inline const ChannelRow* channel_rows() {
  static const ChannelRow v[kNumChannels] = {
    // ---- mu+ : G4MuonPlus.cc
    //   G4VDecayChannel* mode = new G4MuonDecayChannel("mu+",1.00);
    // ONE channel, BR 1.00, and it is G4MuonDecayChannel - not
    // G4MuonDecayChannelWithSpin and not G4MuonRadiativeDecayChannelWithSpin. Those two
    // exist in 11.1.1 and neither muon definition installs them; nothing in QBBC's
    // G4DecayPhysics replaces a decay table either (it only registers one shared G4Decay
    // over every applicable particle). So the muon has no radiative channel here and no spin
    // correlation: G4MuonDecayChannel's own comment is "this version neglects muon
    // polarization, and electron mass; assumes the pure V-A coupling".
    {ChannelKind::kMuonDecay, 1.00, 3, {kPdgPositron, kPdgNuE, kPdgAntiNuMu}, 0.0, 0.0},
    // ---- mu- : G4MuonMinus.cc
    {ChannelKind::kMuonDecay, 1.00, 3, {kPdgElectron, kPdgAntiNuE, kPdgNuMu}, 0.0, 0.0},
    // ---- pi+ : G4PionPlus.cc
    //   new G4PhaseSpaceDecayChannel("pi+",1.00,2,"mu+","nu_mu");
    // One channel at BR 1.00. pi -> e nu (1.2e-4) and the radiative channel are NOT in
    // 11.1.1's table, so G4PionRadiativeDecayChannel is never constructed for pi+/pi- and
    // there is nothing here to transcribe from it.
    {ChannelKind::kPhaseSpace, 1.00, 2, {kPdgMuPlus, kPdgNuMu, 0}, 0.0, 0.0},
    // ---- pi- : G4PionMinus.cc
    {ChannelKind::kPhaseSpace, 1.00, 2, {kPdgMuMinus, kPdgAntiNuMu, 0}, 0.0, 0.0},
    // ---- pi0 : G4PionZero.cc, inserted 0.988 then 0.012, already descending
    //   new G4PhaseSpaceDecayChannel("pi0",0.988,2,"gamma","gamma");
    //   new G4DalitzDecayChannel("pi0",0.012,"e-","e+");
    // The Dalitz constructor puts gamma FIRST (idGamma=0), then the lepton, then the
    // antilepton, whatever order the names were given in - so the daughter list is
    // (gamma, e-, e+) and not (e-, e+, gamma).
    {ChannelKind::kPhaseSpace, 0.988, 2, {kPdgGamma, kPdgGamma, 0}, 0.0, 0.0},
    {ChannelKind::kDalitz, 0.012, 3, {kPdgGamma, kPdgElectron, kPdgPositron}, 0.0, 0.0},
    // ---- kaon+ : G4KaonPlus.cc, six channels. Insertion order 0.6355, 0.2066, 0.0559,
    // 0.01761, 0.0507, 0.0335; stored order below is what Insert leaves behind.
    //   mode[0] G4PhaseSpaceDecayChannel("kaon+",0.6355,2,"mu+","nu_mu")
    //   mode[1] G4PhaseSpaceDecayChannel("kaon+",0.2066,2,"pi+","pi0")
    //   mode[2] G4PhaseSpaceDecayChannel("kaon+",0.0559,3,"pi+","pi+","pi-")
    //   mode[4] G4KL3DecayChannel("kaon+",0.0507,"pi0","e+","nu_e")          <- Ke3
    //   mode[5] G4KL3DecayChannel("kaon+",0.0335,"pi0","mu+","nu_mu")        <- Kmu3
    //   mode[3] G4PhaseSpaceDecayChannel("kaon+",0.01761,3,"pi+","pi0","pi0")
    // The KL3 parameters come from G4KL3DecayChannel's constructor: (K+, e+) and (K-, e-)
    // -> lambda 0.0286, xi0 -0.35; (K+, mu+) and (K-, mu-) -> lambda 0.033, xi0 -0.35. The
    // same constructor holds two K0L pairs that are not reached from here - 0.0300 / -0.11
    // for Ke3 and 0.034 / -0.11 for Kmu3 - and an `else` arm for illegal arguments that
    // silently reuses K0L's Ke3 pair rather than refusing. That last one is worth knowing:
    // a mistyped parent or lepton name in Geant4 gives a channel with the wrong form
    // factors and no diagnostic above verbose level 2.
    {ChannelKind::kPhaseSpace, 0.6355, 2, {kPdgMuPlus, kPdgNuMu, 0}, 0.0, 0.0},
    {ChannelKind::kPhaseSpace, 0.2066, 2, {kPdgPiPlus, kPdgPiZero, 0}, 0.0, 0.0},
    {ChannelKind::kPhaseSpace, 0.0559, 3, {kPdgPiPlus, kPdgPiPlus, kPdgPiMinus}, 0.0, 0.0},
    {ChannelKind::kKL3, 0.0507, 3, {kPdgPiZero, kPdgPositron, kPdgNuE}, 0.0286, -0.35},
    {ChannelKind::kKL3, 0.0335, 3, {kPdgPiZero, kPdgMuPlus, kPdgNuMu}, 0.033, -0.35},
    {ChannelKind::kPhaseSpace, 0.01761, 3, {kPdgPiPlus, kPdgPiZero, kPdgPiZero}, 0.0, 0.0},
    // ---- kaon- : G4KaonMinus.cc, the charge conjugate of the above, same BRs and same
    // KL3 parameters (the constructor tests kaon- with e-/mu- into the same two branches).
    {ChannelKind::kPhaseSpace, 0.6355, 2, {kPdgMuMinus, kPdgAntiNuMu, 0}, 0.0, 0.0},
    {ChannelKind::kPhaseSpace, 0.2066, 2, {kPdgPiMinus, kPdgPiZero, 0}, 0.0, 0.0},
    {ChannelKind::kPhaseSpace, 0.0559, 3, {kPdgPiMinus, kPdgPiPlus, kPdgPiMinus}, 0.0, 0.0},
    {ChannelKind::kKL3, 0.0507, 3, {kPdgPiZero, kPdgElectron, kPdgAntiNuE}, 0.0286, -0.35},
    {ChannelKind::kKL3, 0.0335, 3, {kPdgPiZero, kPdgMuMinus, kPdgAntiNuMu}, 0.033, -0.35},
    {ChannelKind::kPhaseSpace, 0.01761, 3, {kPdgPiMinus, kPdgPiZero, kPdgPiZero}, 0.0, 0.0},
    // ---- neutron : G4Neutron.cc
    //   G4VDecayChannel* mode = new G4NeutronBetaDecayChannel("neutron",1.00);
    // and that constructor sets the daughters to e-, anti_nu_e, proton in that order.
    {ChannelKind::kNeutronBeta, 1.00, 3, {kPdgElectron, kPdgAntiNuE, kPdgProton}, 0.0, 0.0},
    // ---- the hyperons (P10c). EVERY ONE IS G4PhaseSpaceDecayChannel WITH TWO DAUGHTERS, which
    // is why this extension needed no new channel kind: the whole of the added physics is eleven
    // rows of data through a sampler that pi+ -> mu+ nu_mu already validated.
    //
    // Each group is in G4DecayTable::Insert order. All seven definitions happen to insert in
    // descending BR already, so Insert's sort is the identity here - stated because the kaon
    // group above is the counterexample and a reader should not have to check.
    // ---- lambda : G4Lambda.cc
    //   mode[0] = new G4PhaseSpaceDecayChannel("lambda",0.639,2,"proton","pi-");
    //   mode[1] = new G4PhaseSpaceDecayChannel("lambda",0.358,2,"neutron","pi0");
    // The two sum to 0.997, not 1: SelectADecayChannel normalises by the sum, as for the kaons.
    {ChannelKind::kPhaseSpace, 0.639, 2, {kPdgProton, kPdgPiMinus, 0}, 0.0, 0.0},
    {ChannelKind::kPhaseSpace, 0.358, 2, {kPdgNeutron, kPdgPiZero, 0}, 0.0, 0.0},
    // ---- sigma+ : G4SigmaPlus.cc, 0.516 + 0.483 = 0.999
    {ChannelKind::kPhaseSpace, 0.516, 2, {kPdgProton, kPdgPiZero, 0}, 0.0, 0.0},
    {ChannelKind::kPhaseSpace, 0.483, 2, {kPdgNeutron, kPdgPiPlus, 0}, 0.0, 0.0},
    // ---- sigma0 : G4SigmaZero.cc, one channel, and it is ELECTROMAGNETIC - the only hyperon
    // channel that emits a photon, and the reason a trapped sigma0 puts a gamma back on the
    // cascade stack rather than a nucleon.
    {ChannelKind::kPhaseSpace, 1.000, 2, {kPdgLambda, kPdgGamma, 0}, 0.0, 0.0},
    // ---- sigma- : G4SigmaMinus.cc
    {ChannelKind::kPhaseSpace, 1.00, 2, {kPdgNeutron, kPdgPiMinus, 0}, 0.0, 0.0},
    // ---- xi0 : G4XiZero.cc
    {ChannelKind::kPhaseSpace, 1.000, 2, {kPdgLambda, kPdgPiZero, 0}, 0.0, 0.0},
    // ---- xi- : G4XiMinus.cc
    {ChannelKind::kPhaseSpace, 1.000, 2, {kPdgLambda, kPdgPiMinus, 0}, 0.0, 0.0},
    // ---- omega- : G4OmegaMinus.cc, three channels. Its first daughter is a LAMBDA and its
    // second a K-, so an omega- trapped in a nucleus puts an unstable daughter back on the
    // cascade stack, which may itself be trapped and decayed again. That recursion is the
    // cascade's own loop and not a recursive call; see intra_cascader.cuh.
    {ChannelKind::kPhaseSpace, 0.678, 2, {kPdgLambda, kPdgKaonMinus, 0}, 0.0, 0.0},
    {ChannelKind::kPhaseSpace, 0.236, 2, {kPdgXiZero, kPdgPiMinus, 0}, 0.0, 0.0},
    {ChannelKind::kPhaseSpace, 0.086, 2, {kPdgXiMinus, kPdgPiZero, 0}, 0.0, 0.0},
    // ---- P23b, channels 30 onward.
    //
    // ---- kaon0S : G4KaonZeroShort.cc, inserted 0.6920 then 0.3069 - already descending.
    //   mode[0] = new G4PhaseSpaceDecayChannel("kaon0S",0.6920,2,"pi+","pi-");
    //   mode[1] = new G4PhaseSpaceDecayChannel("kaon0S",0.3069,2,"pi0","pi0");
    {ChannelKind::kPhaseSpace, 0.6920, 2, {kPdgPiPlus, kPdgPiMinus, 0}, 0.0, 0.0},
    {ChannelKind::kPhaseSpace, 0.3069, 2, {kPdgPiZero, kPdgPiZero, 0}, 0.0, 0.0},
    // ---- kaon0L : G4KaonZeroLong.cc, six channels, inserted
    //   mode[0] PhaseSpace 0.1952 pi0 pi0 pi0
    //   mode[1] PhaseSpace 0.1254 pi0 pi+ pi-
    //   mode[2] KL3        0.2027 pi- e+ nu_e          (Ke3)
    //   mode[3] KL3        0.2027 pi+ e- anti_nu_e     (Ke3)
    //   mode[4] KL3        0.1352 pi- mu+ nu_mu        (Kmu3)
    //   mode[5] KL3        0.1352 pi+ mu- anti_nu_mu   (Kmu3)
    // and Insert's strict `br > GetBR()` leaves them as below: each Ke3 goes in front of the
    // 0.1952 and the SECOND Ke3 behind the first, because an equal BR does not move ahead; the
    // two Kmu3 land between 3pi0 and pi0 pi+ pi- in their own insertion order. So the pi0 pi+ pi-
    // channel, written second, is selected last.
    //
    // The KL3 form factors are K0L's own and not the charged kaon's, which is what the
    // constructor's (parent, lepton) ladder gives "kaon0L": lambda 0.0300 and xi0 -0.11 for
    // either Ke3, 0.034 and -0.11 for either Kmu3 - and the daughter order is (pion, lepton,
    // neutrino) whatever the charges.
    {ChannelKind::kKL3, 0.2027, 3, {kPdgPiMinus, kPdgPositron, kPdgNuE}, 0.0300, -0.11},
    {ChannelKind::kKL3, 0.2027, 3, {kPdgPiPlus, kPdgElectron, kPdgAntiNuE}, 0.0300, -0.11},
    {ChannelKind::kPhaseSpace, 0.1952, 3, {kPdgPiZero, kPdgPiZero, kPdgPiZero}, 0.0, 0.0},
    {ChannelKind::kKL3, 0.1352, 3, {kPdgPiMinus, kPdgMuPlus, kPdgNuMu}, 0.034, -0.11},
    {ChannelKind::kKL3, 0.1352, 3, {kPdgPiPlus, kPdgMuMinus, kPdgAntiNuMu}, 0.034, -0.11},
    {ChannelKind::kPhaseSpace, 0.1254, 3, {kPdgPiZero, kPdgPiPlus, kPdgPiMinus}, 0.0, 0.0},
    // ---- the anti-hyperons, each the charge conjugate of its particle's table with the same
    // BRs and the same insertion order (already descending). G4AntiLambda.cc,
    // G4AntiSigmaPlus.cc, G4AntiSigmaZero.cc, G4AntiSigmaMinus.cc, G4AntiXiZero.cc,
    // G4AntiXiMinus.cc, G4AntiOmegaMinus.cc.
    {ChannelKind::kPhaseSpace, 0.639, 2, {kPdgAntiProton, kPdgPiPlus, 0}, 0.0, 0.0},
    {ChannelKind::kPhaseSpace, 0.358, 2, {kPdgAntiNeutron, kPdgPiZero, 0}, 0.0, 0.0},
    {ChannelKind::kPhaseSpace, 0.516, 2, {kPdgAntiProton, kPdgPiZero, 0}, 0.0, 0.0},
    {ChannelKind::kPhaseSpace, 0.483, 2, {kPdgAntiNeutron, kPdgPiMinus, 0}, 0.0, 0.0},
    {ChannelKind::kPhaseSpace, 1.000, 2, {kPdgAntiLambda, kPdgGamma, 0}, 0.0, 0.0},
    {ChannelKind::kPhaseSpace, 1.00, 2, {kPdgAntiNeutron, kPdgPiPlus, 0}, 0.0, 0.0},
    {ChannelKind::kPhaseSpace, 1.000, 2, {kPdgAntiLambda, kPdgPiZero, 0}, 0.0, 0.0},
    {ChannelKind::kPhaseSpace, 1.000, 2, {kPdgAntiLambda, kPdgPiPlus, 0}, 0.0, 0.0},
    {ChannelKind::kPhaseSpace, 0.678, 2, {kPdgAntiLambda, kPdgKaonPlus, 0}, 0.0, 0.0},
    {ChannelKind::kPhaseSpace, 0.236, 2, {kPdgAntiXiZero, kPdgPiPlus, 0}, 0.0, 0.0},
    {ChannelKind::kPhaseSpace, 0.086, 2, {kPdgAntiXiMinus, kPdgPiZero, 0}, 0.0, 0.0},
    // ---- the b/c hadrons: G4HadronicBuilder::BuildDecayTableForBCHadrons, one
    // `new G4PhaseSpaceDecayChannel(name, 1.0, n, ...)` each, in the order of its switch.
    // Charmed mesons:
    {ChannelKind::kPhaseSpace, 1.0, 3, {kPdgKaonMinus, kPdgPiPlus, kPdgPiPlus}, 0.0, 0.0},
    {ChannelKind::kPhaseSpace, 1.0, 3, {kPdgKaonPlus, kPdgPiMinus, kPdgPiMinus}, 0.0, 0.0},
    {ChannelKind::kPhaseSpace, 1.0, 3, {kPdgKaonMinus, kPdgPiPlus, kPdgPiZero}, 0.0, 0.0},
    {ChannelKind::kPhaseSpace, 1.0, 3, {kPdgKaonPlus, kPdgPiMinus, kPdgPiZero}, 0.0, 0.0},
    {ChannelKind::kPhaseSpace, 1.0, 3, {kPdgKaonPlus, kPdgKaonMinus, kPdgPiPlus}, 0.0, 0.0},
    {ChannelKind::kPhaseSpace, 1.0, 3, {kPdgKaonMinus, kPdgKaonPlus, kPdgPiMinus}, 0.0, 0.0},
    // Bottom mesons:
    {ChannelKind::kPhaseSpace, 1.0, 3, {kPdgAntiDZero, kPdgPiPlus, kPdgPiZero}, 0.0, 0.0},
    {ChannelKind::kPhaseSpace, 1.0, 3, {kPdgDZero, kPdgPiMinus, kPdgPiZero}, 0.0, 0.0},
    {ChannelKind::kPhaseSpace, 1.0, 3, {kPdgDMinus, kPdgPiPlus, kPdgPiZero}, 0.0, 0.0},
    {ChannelKind::kPhaseSpace, 1.0, 3, {kPdgDPlus, kPdgPiMinus, kPdgPiZero}, 0.0, 0.0},
    {ChannelKind::kPhaseSpace, 1.0, 3, {kPdgDsMinus, kPdgPiPlus, kPdgPiZero}, 0.0, 0.0},
    {ChannelKind::kPhaseSpace, 1.0, 3, {kPdgDsPlus, kPdgPiMinus, kPdgPiZero}, 0.0, 0.0},
    {ChannelKind::kPhaseSpace, 1.0, 2, {kPdgJPsi, kPdgPiPlus, 0}, 0.0, 0.0},
    {ChannelKind::kPhaseSpace, 1.0, 2, {kPdgJPsi, kPdgPiMinus, 0}, 0.0, 0.0},
    // Charmed baryons and anti-baryons:
    {ChannelKind::kPhaseSpace, 1.0, 3, {kPdgProton, kPdgKaonMinus, kPdgPiPlus}, 0.0, 0.0},
    {ChannelKind::kPhaseSpace, 1.0, 3, {kPdgAntiProton, kPdgKaonPlus, kPdgPiMinus}, 0.0, 0.0},
    {ChannelKind::kPhaseSpace, 1.0, 3, {kPdgSigmaPlus, kPdgKaonMinus, kPdgPiPlus}, 0.0, 0.0},
    {ChannelKind::kPhaseSpace, 1.0, 3, {kPdgAntiSigmaPlus, kPdgKaonPlus, kPdgPiMinus}, 0.0,
     0.0},
    {ChannelKind::kPhaseSpace, 1.0, 3, {kPdgLambda, kPdgKaonMinus, kPdgPiPlus}, 0.0, 0.0},
    {ChannelKind::kPhaseSpace, 1.0, 3, {kPdgAntiLambda, kPdgKaonPlus, kPdgPiMinus}, 0.0, 0.0},
    {ChannelKind::kPhaseSpace, 1.0, 3, {kPdgXiZero, kPdgKaonMinus, kPdgPiPlus}, 0.0, 0.0},
    {ChannelKind::kPhaseSpace, 1.0, 3, {kPdgAntiXiZero, kPdgKaonPlus, kPdgPiMinus}, 0.0, 0.0},
    // Bottom baryons and anti-baryons. lambda_b's is the only FOUR-body channel any real table
    // here has, so it is the first real caller of G4PhaseSpaceDecayChannel::ManyBodyDecayIt -
    // validated until now against a synthetic channel only (decay_channels.cuh).
    {ChannelKind::kPhaseSpace, 1.0, 4, {kPdgLambdacPlus, kPdgPiPlus, kPdgPiMinus, kPdgPiMinus},
     0.0, 0.0},
    {ChannelKind::kPhaseSpace, 1.0, 4,
     {kPdgAntiLambdacPlus, kPdgPiMinus, kPdgPiPlus, kPdgPiPlus}, 0.0, 0.0},
    {ChannelKind::kPhaseSpace, 1.0, 3, {kPdgLambdacPlus, kPdgKaonMinus, kPdgPiZero}, 0.0, 0.0},
    {ChannelKind::kPhaseSpace, 1.0, 3, {kPdgAntiLambdacPlus, kPdgKaonPlus, kPdgPiZero}, 0.0,
     0.0},
    {ChannelKind::kPhaseSpace, 1.0, 3, {kPdgLambdacPlus, kPdgKaonMinus, kPdgPiMinus}, 0.0,
     0.0},
    {ChannelKind::kPhaseSpace, 1.0, 3, {kPdgAntiLambdacPlus, kPdgKaonPlus, kPdgPiPlus}, 0.0,
     0.0},
    {ChannelKind::kPhaseSpace, 1.0, 3, {kPdgXicPlus, kPdgKaonMinus, kPdgPiMinus}, 0.0, 0.0},
    {ChannelKind::kPhaseSpace, 1.0, 3, {kPdgAntiXicPlus, kPdgKaonPlus, kPdgPiPlus}, 0.0, 0.0},
    // The three quarkonia, added by name after the loop:
    //   etac    -> eta pi+ pi-        J/psi -> pi0 pi+ pi-        Upsilon -> eta_prime pi+ pi-
    {ChannelKind::kPhaseSpace, 1.0, 3, {kPdgEta, kPdgPiPlus, kPdgPiMinus}, 0.0, 0.0},
    {ChannelKind::kPhaseSpace, 1.0, 3, {kPdgPiZero, kPdgPiPlus, kPdgPiMinus}, 0.0, 0.0},
    {ChannelKind::kPhaseSpace, 1.0, 3, {kPdgEtaPrime, kPdgPiPlus, kPdgPiMinus}, 0.0, 0.0},
    // ---- eta : G4Eta.cc, four phase-space channels, inserted already descending. Here because
    // etac -> eta pi+ pi- makes one; it is also what FTFP's string fragmentation makes most of
    // after the pions and kaons, and transport is what needs it there.
    //   mode[0] ("eta",0.3942,2,"gamma","gamma")  mode[1] ("eta",0.3256,3,"pi0","pi0","pi0")
    //   mode[2] ("eta",0.226,3,"pi0","pi+","pi-")  mode[3] ("eta",0.0468,3,"gamma","pi+","pi-")
    // The eta_prime's own table is NOT here: two of its five channels have a rho0 and an omega
    // daughter, whose widths take G4PhaseSpaceDecayChannel into DynamicalMass, which this port
    // does not have. It is refused by name below.
    {ChannelKind::kPhaseSpace, 0.3942, 2, {kPdgGamma, kPdgGamma, 0}, 0.0, 0.0},
    {ChannelKind::kPhaseSpace, 0.3256, 3, {kPdgPiZero, kPdgPiZero, kPdgPiZero}, 0.0, 0.0},
    {ChannelKind::kPhaseSpace, 0.226, 3, {kPdgPiZero, kPdgPiPlus, kPdgPiMinus}, 0.0, 0.0},
    {ChannelKind::kPhaseSpace, 0.0468, 3, {kPdgGamma, kPdgPiPlus, kPdgPiMinus}, 0.0, 0.0},
    // ---- the charmed and bottom sigmas, 86 through 97: one channel each, from the definition,
    // to the lambda_c+ or lambda_b and the pion that carries the charge.
    //   ("sigma_c++",1.000,2,"lambda_c+","pi+")  ("sigma_c+",1.000,2,"lambda_c+","pi0")
    //   ("sigma_c0",1.000,2,"lambda_c+","pi-")   ("sigma_b+",1.000,2,"lambda_b","pi+") ...
    {ChannelKind::kPhaseSpace, 1.000, 2, {kPdgLambdacPlus, kPdgPiPlus, 0}, 0.0, 0.0},
    {ChannelKind::kPhaseSpace, 1.000, 2, {kPdgLambdacPlus, kPdgPiZero, 0}, 0.0, 0.0},
    {ChannelKind::kPhaseSpace, 1.000, 2, {kPdgLambdacPlus, kPdgPiMinus, 0}, 0.0, 0.0},
    {ChannelKind::kPhaseSpace, 1.000, 2, {kPdgLambdab, kPdgPiPlus, 0}, 0.0, 0.0},
    {ChannelKind::kPhaseSpace, 1.000, 2, {kPdgLambdab, kPdgPiZero, 0}, 0.0, 0.0},
    {ChannelKind::kPhaseSpace, 1.000, 2, {kPdgLambdab, kPdgPiMinus, 0}, 0.0, 0.0},
    {ChannelKind::kPhaseSpace, 1.000, 2, {kPdgAntiLambdacPlus, kPdgPiMinus, 0}, 0.0, 0.0},
    {ChannelKind::kPhaseSpace, 1.000, 2, {kPdgAntiLambdacPlus, kPdgPiZero, 0}, 0.0, 0.0},
    {ChannelKind::kPhaseSpace, 1.000, 2, {kPdgAntiLambdacPlus, kPdgPiPlus, 0}, 0.0, 0.0},
    {ChannelKind::kPhaseSpace, 1.000, 2, {kPdgAntiLambdab, kPdgPiMinus, 0}, 0.0, 0.0},
    {ChannelKind::kPhaseSpace, 1.000, 2, {kPdgAntiLambdab, kPdgPiZero, 0}, 0.0, 0.0},
    {ChannelKind::kPhaseSpace, 1.000, 2, {kPdgAntiLambdab, kPdgPiPlus, 0}, 0.0, 0.0},
  };
  return v;
}

/// The parents, and where each one's channels start. Order here is irrelevant to the physics
/// (the lookup is by PDG code); the channel order WITHIN a group is not.
constexpr int kNumTables = 70;   // 8, the seven hyperons P10c added, and P23b's 55

__host__ __device__ inline const DecayTableRow* table_rows() {
  static const DecayTableRow v[kNumTables] = {
      {kPdgMuPlus, 0, 1},     {kPdgMuMinus, 1, 1}, {kPdgPiPlus, 2, 1},
      {kPdgPiMinus, 3, 1},    {kPdgPiZero, 4, 2},  {kPdgKaonPlus, 6, 6},
      {kPdgKaonMinus, 12, 6}, {kPdgNeutron, 18, 1},
      // The hyperons, channels 19 through 29.
      {kPdgLambda, 19, 2},    {kPdgSigmaPlus, 21, 2}, {kPdgSigmaZero, 23, 1},
      {kPdgSigmaMinus, 24, 1}, {kPdgXiZero, 25, 1},   {kPdgXiMinus, 26, 1},
      {kPdgOmegaMinus, 27, 3},
      // P23b: the neutral kaons, channels 30 through 37.
      {kPdgKaonZeroShort, 30, 2}, {kPdgKaonZeroLong, 32, 6},
      // The anti-hyperons, 38 through 48.
      {kPdgAntiLambda, 38, 2},     {kPdgAntiSigmaPlus, 40, 2}, {kPdgAntiSigmaZero, 42, 1},
      {kPdgAntiSigmaMinus, 43, 1}, {kPdgAntiXiZero, 44, 1},    {kPdgAntiXiMinus, 45, 1},
      {kPdgAntiOmegaMinus, 46, 3},
      // The b/c hadrons, one channel each, 49 through 81.
      {kPdgDPlus, 49, 1},           {kPdgDMinus, 50, 1},         {kPdgDZero, 51, 1},
      {kPdgAntiDZero, 52, 1},       {kPdgDsPlus, 53, 1},         {kPdgDsMinus, 54, 1},
      {kPdgBPlus, 55, 1},           {kPdgBMinus, 56, 1},         {kPdgBZero, 57, 1},
      {kPdgAntiBZero, 58, 1},       {kPdgBsZero, 59, 1},         {kPdgAntiBsZero, 60, 1},
      {kPdgBcPlus, 61, 1},          {kPdgBcMinus, 62, 1},        {kPdgLambdacPlus, 63, 1},
      {kPdgAntiLambdacPlus, 64, 1}, {kPdgXicPlus, 65, 1},        {kPdgAntiXicPlus, 66, 1},
      {kPdgXicZero, 67, 1},         {kPdgAntiXicZero, 68, 1},    {kPdgOmegacZero, 69, 1},
      {kPdgAntiOmegacZero, 70, 1},  {kPdgLambdab, 71, 1},        {kPdgAntiLambdab, 72, 1},
      {kPdgXibZero, 73, 1},         {kPdgAntiXibZero, 74, 1},    {kPdgXibMinus, 75, 1},
      {kPdgAntiXibMinus, 76, 1},    {kPdgOmegabMinus, 77, 1},    {kPdgAntiOmegabMinus, 78, 1},
      {kPdgEtac, 79, 1},            {kPdgJPsi, 80, 1},           {kPdgUpsilon, 81, 1},
      // The eta, 82 through 85.
      {kPdgEta, 82, 4},
      // The charmed and bottom sigmas, 86 through 97.
      {kPdgSigmacPlusPlus, 86, 1},     {kPdgSigmacPlus, 87, 1},     {kPdgSigmacZero, 88, 1},
      {kPdgSigmabPlus, 89, 1},         {kPdgSigmabZero, 90, 1},     {kPdgSigmabMinus, 91, 1},
      {kPdgAntiSigmacPlusPlus, 92, 1}, {kPdgAntiSigmacPlus, 93, 1}, {kPdgAntiSigmacZero, 94, 1},
      {kPdgAntiSigmabPlus, 95, 1},     {kPdgAntiSigmabZero, 96, 1}, {kPdgAntiSigmabMinus, 97, 1},
  };
  return v;
}

/// The parent's table, or count == 0 if there is none. A count of zero is the refusal: see
/// decay_refusal_reason.
__host__ __device__ inline DecayTableRow decay_table_for(int pdg) {
  const DecayTableRow* t = table_rows();
  for (int i = 0; i < kNumTables; ++i) {
    if (t[i].parent_pdg == pdg) { return t[i]; }
  }
  return DecayTableRow{pdg, 0, 0};
}

/// Why a PDG code has no table, in words, for the message at the point the decay would have
/// happened. Named codes are the ones Geant4 DOES have a table for and this package does not
/// - so that "refused" is never confused with "stable".
///
/// The unnamed rest fall through to one string rather than a per-code list, because the point
/// of the message is that a code arrived that nothing transports; the code itself is printed
/// by the caller.
__host__ __device__ inline const char* decay_refusal_reason(int pdg) {
  switch (pdg) {
    // K0L, K0S and the seven anti-hyperons are ported (P23b) and no longer reach this function.
    // They were refused here while "nothing in this port can produce one" - and that stopped
    // being true when FTFP was wired: its string fragmentation emits anti-hyperons and neutral
    // kaons on every few events, which is what P23a's transport will need these tables for.
    case 311:
    case -311:
      return "K0/anti-K0: G4KaonZero's table is two ONE-BODY channels into K0L and K0S; no "
             "generator in QBBC emits a K0 (G4KineticTrack and Bertini both mix it first) - "
             "refused";
    case -2112:
      return "anti_neutron: G4NeutronBetaDecayChannel(\"anti_neutron\") - the anti-light-ion "
             "package's (P23c), refused here; 880 s against an annihilation length of cm";
    case 331:
      return "eta_prime: two of its five G4PhaseSpaceDecayChannels have a rho0 or an omega "
             "daughter, wide enough for G4VDecayChannel::DynamicalMass, which is not ported";
    case 15:
    case -15:
      return "tau: G4TauLeptonicDecayChannel and friends - refused, not transported";
    default:
      return "no decay table transcribed for this PDG code";
  }
}

/// G4ParticleDefinition::GetPDGStable() for a code the tables know, and `true` for one they
/// do not - so a species with no row is never handed to a sampler by accident.
__host__ __device__ inline bool particle_is_stable(int pdg) {
  const int i = particle_index(pdg);
  return (i >= 0) ? particle_rows()[i].stable : true;
}

/// G4ParticleDefinition::GetPDGLifeTime(), ns. -1 for a stable particle, and -1 for a code
/// with no row (which is what `G4Decay::IsApplicable` reads to decline the process).
__host__ __device__ inline double particle_lifetime(int pdg) {
  const int i = particle_index(pdg);
  return (i >= 0) ? particle_rows()[i].lifetime : -1.0;
}

}  // namespace g4gpu::decay
