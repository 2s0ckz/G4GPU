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
// What is NOT here is refused by PDG code, not approximated: see `decay_refusal` below.
// K0L, K0S, the hyperons and the anti-hyperons have tables in Geant4 and no row here, and
// P1's species stub refuses them at emission for the same reason - nothing transports them
// yet. The refusal is structural: a PDG code with no row gets `kNoTable` and a message that
// names the code, at the point the decay would have happened.
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
};

/// CLHEP's unit prefixes, so that a mass can be written exactly as its G4ParticleDefinition
/// constructor writes it. `megaelectronvolt = 1`, `gigaelectronvolt = 1e3*MeV`,
/// `nanosecond = 1`, `second = 1e9*ns` (CLHEP SystemOfUnits.h), which is also what
/// core/units.cuh pins - repeated here because this file is included by device code that has
/// no other reason to pull in the unit system.
__host__ __device__ inline constexpr double gev() { return 1.0e3; }
__host__ __device__ inline constexpr double mev() { return 1.0; }
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
constexpr int kNumParticles = 16;

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
  };
  return v;
}

/// Index into particle_rows(), or -1. Linear scan over sixteen rows: this is called once per
/// decay and once per daughter, and a sixteen-entry scan is cheaper on a GPU than any
/// structure that would replace it.
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
constexpr int kNumChannels = 19;

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
  };
  return v;
}

/// The parents, and where each one's channels start. Order here is irrelevant to the physics
/// (the lookup is by PDG code); the channel order WITHIN a group is not.
constexpr int kNumTables = 8;

__host__ __device__ inline const DecayTableRow* table_rows() {
  static const DecayTableRow v[kNumTables] = {
      {kPdgMuPlus, 0, 1},     {kPdgMuMinus, 1, 1}, {kPdgPiPlus, 2, 1},
      {kPdgPiMinus, 3, 1},    {kPdgPiZero, 4, 2},  {kPdgKaonPlus, 6, 6},
      {kPdgKaonMinus, 12, 6}, {kPdgNeutron, 18, 1},
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
    case 130:
      return "K0L: G4KaonZeroLong installs two G4PhaseSpaceDecayChannels and four "
             "G4KL3DecayChannels with K0L's own form factors - refused, P1 refuses the species";
    case 310:
      return "K0S: G4KaonZeroShort's table - refused, P1 refuses the species";
    case 311:
    case -311:
      return "K0/anti-K0: short-lived, decays through K0S/K0L - refused";
    case 3122:
    case -3122:
    case 3222:
    case 3212:
    case 3112:
    case -3222:
    case -3212:
    case -3112:
    case 3322:
    case 3312:
    case -3322:
    case -3312:
    case 3334:
    case -3334:
      return "hyperon: has a Geant4 decay table - refused, P1 refuses the species";
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
