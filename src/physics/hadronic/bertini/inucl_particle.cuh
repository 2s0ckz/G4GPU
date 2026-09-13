// G4InuclParticleNames, G4InuclElementaryParticle and G4InuclSpecialFunctions: the type codes
// Bertini speaks, and the free functions the whole INUCL tree calls.
//
// Transcribed from Geant4 11.1.1:
//   G4InuclParticleNames                  (cascade/cascade/include/G4InuclParticleNames.hh)
//   G4InuclElementaryParticle::makeDefinition / type / getStrangeness / getParticleMass
//                                         (cascade/cascade/src/G4InuclElementaryParticle.cc)
//   G4InuclSpecialFunctions::randomInuclPowers / getAL / csNN / csPN / FermiEnergy / G4cbrt /
//     randomGauss / randomPHI / randomCOS_SIN / generateWithFixedTheta /
//     generateWithRandomAngles  (cascade/cascade/src/G4InuclSpecialFunctions.cc)
//   G4InuclSpecialFunctions::bindingEnergy (cascade/cascade/src/bindingEnergy.cc)
//   G4InuclSpecialFunctions::nucleiLevelDensity (cascade/cascade/src/nucleiLevelDensity.cc)
//   paraMaker::getParams / getTruncated    (cascade/cascade/src/paraMaker.cc)
//
// **Bertini's type codes are not PDG codes and its units are not the port's.** A type code is a
// small signed integer whose PRODUCT with another identifies a two-body initial state - that is
// why the codes are 1, 2, 3, 5, 7, 9, 11, ... and not 1, 2, 3, 4: `pim*pro` (5) and `pip*neu`
// (6) have to be distinct, and so do all 34 pairs G4CascadeChannelTables registers. Masses are
// in **GeV**, because `getParticleMass` returns `pd->GetPDGMass()*MeV/GeV` and every INUCL class
// works in GeV/mm from there on. The conversion to the port's MeV happens once, at the
// interface, exactly where G4CascadeInterface does it.
//
// **Why the masses are a table here and not a call into core/particle.cuh.** P1's ParticleType
// has fourteen species and none of the ones this needs beyond the nucleons and pions: no K0, no
// anti-K0, no hyperon, no anti-nucleus, and above all none of the three unbound dibaryons
// (G4Diproton, G4UnboundPN, G4Dineutron), which are Bertini's own G4ParticleDefinition
// subclasses and exist nowhere else in Geant4. These are not transported species - they live and
// die inside one cascade - so they belong to this module. Every value is checked at 1e-15
// against `ref/oracle/bertini_particles.csv`, which reads them out of the install's own PDG
// table, so a mistyped digit fails rather than shifting a spectrum.
#ifndef G4GPU_BERTINI_INUCL_PARTICLE_CUH
#define G4GPU_BERTINI_INUCL_PARTICLE_CUH

#include <cmath>

#include "data/g4pow.hh"
#include "physics/hadronic/bertini/lorentz_convertor.cuh"
#include "physics/hadronic/deexcitation/nuclear_masses.cuh"

namespace g4gpu::physics::hadronic::bert {

// =============================================================================================
// G4InuclParticleNames
// =============================================================================================

/// G4InuclParticleNames::Long, verbatim. The `Short` enum is the same values under other
/// spellings, so it is not duplicated.
enum InuclType : int {
  kNuclei = 0, kProton = 1, kNeutron = 2,
  kPionPlus = 3, kPionMinus = 5, kPionZero = 7, kPhoton = 9,
  kKaonPlus = 11, kKaonMinus = 13, kKaonZero = 15, kKaonZeroBar = 17,
  kLambda = 21, kSigmaPlus = 23, kSigmaZero = 25, kSigmaMinus = 27,
  kXiZero = 29, kXiMinus = 31, kOmegaMinus = 33,
  kDeuteron = 41, kTriton = 43, kHe3 = 45, kAlpha = 47,
  kAntiProton = 51, kAntiNeutron = 53,
  kAntiDeuteron = 61, kAntiTriton = 63, kAntiHe3 = 65, kAntiAlpha = 67,
  kDiproton = 111, kUnboundPN = 112, kDineutron = 122,
  kElectronNu = -1, kMuonNu = -3, kTauNu = -5,
  kAntiElectronNu = -7, kAntiMuonNu = -9, kAntiTauNu = -11,
  kWMinus = -13, kWPlus = -15, kZzero = -17,
  kElectron = -21, kMuonMinus = -23, kTauMinus = -25,
  kPositron = -27, kMuonPlus = -29, kTauPlus = -31
};

__host__ __device__ inline bool inucl_is_photon(int t) { return t == kPhoton; }
__host__ __device__ inline bool inucl_is_muon(int t) { return t == kMuonMinus || t == kMuonPlus; }
__host__ __device__ inline bool inucl_is_electron(int t) {
  return t == kElectron || t == kPositron;
}
__host__ __device__ inline bool inucl_is_neutrino(int t) {
  return t == kElectronNu || t == kMuonNu || t == kTauNu || t == kAntiElectronNu ||
         t == kAntiMuonNu || t == kAntiTauNu;
}
__host__ __device__ inline bool inucl_is_pion(int t) {
  return t == kPionPlus || t == kPionMinus || t == kPionZero;
}
__host__ __device__ inline bool inucl_is_nucleon(int t) {
  return t == kProton || t == kNeutron;
}
__host__ __device__ inline bool inucl_is_antinucleon(int t) {
  return t == kAntiProton || t == kAntiNeutron;
}
/// G4InuclParticleNames::quasi_deutron - literally `ityp > 100`, which the three dibaryon codes
/// (111, 112, 122) are and nothing else is.
__host__ __device__ inline bool inucl_is_quasideuteron(int t) { return t > 100; }

/// G4InuclParticleNames::baryon. The header calls it an emulation of
/// G4ParticleDefinition::GetBaryonNumber, and it is a plain lookup - note that the four light
/// nuclei (deuteron 41, triton 43, He3 45, alpha 47) are NOT in it, so it returns 0 for a
/// deuteron the cascade coalesced while returning 2 for an unbound pn pair. G4InuclElementary-
/// Particle::baryon() does not use this function; it asks the G4ParticleDefinition, which gives
/// the deuteron 2. Both are transcribed, under their own names, because the two disagree and
/// different call sites use different ones.
__host__ __device__ inline int inucl_names_baryon(int t) {
  if (t == kProton || t == kNeutron || t == kLambda || t == kSigmaPlus || t == kSigmaZero ||
      t == kSigmaMinus || t == kXiZero || t == kXiMinus || t == kOmegaMinus) {
    return 1;
  }
  if (t == kDeuteron || t == kDiproton || t == kUnboundPN || t == kDineutron) { return 2; }
  if (t == kAntiProton || t == kAntiNeutron) { return -1; }
  if (t == kAntiDeuteron) { return -2; }
  if (t == kAntiTriton || t == kAntiHe3) { return -3; }
  if (t == kAntiAlpha) { return -4; }
  return 0;
}

__host__ __device__ inline bool inucl_names_hyperon(int t) {
  return t == kLambda || t == kSigmaPlus || t == kSigmaZero || t == kSigmaMinus ||
         t == kXiZero || t == kXiMinus || t == kOmegaMinus;
}

// =============================================================================================
// G4InuclElementaryParticle's static tables
// =============================================================================================

/// One row of the type table: mass in GeV, charge in eplus, strangeness, baryon number as the
/// G4ParticleDefinition reports it, and the PDG code for the species that have one.
struct InuclTypeRow {
  double mass_GeV;
  double charge;
  int strangeness;
  int baryon;       ///< G4ParticleDefinition::GetBaryonNumber, which is 2 for a deuteron
  int pdg;          ///< 0 for the three dibaryons, which have no PDG code
};

/// G4InuclElementaryParticle::getParticleMass / getStrangeness through makeDefinition, for
/// every type code that maps to a definition. The masses are `GetPDGMass()*MeV/GeV`, i.e.
/// Geant4's own PDG table divided by 1000, and they are compared against
/// ref/oracle/bertini_particles.csv at 1e-15.
///
/// The three dibaryons are Bertini's own definitions: G4Diproton (2*proton), G4UnboundPN
/// (proton+neutron) and G4Dineutron (2*neutron), each built with the SUM of its constituents'
/// masses and no binding energy - they are unbound by construction, which is the whole point of
/// a quasi-deuteron - so their masses are exact sums of the nucleon masses above and are written
/// that way rather than as independent constants.
__host__ __device__ inline InuclTypeRow inucl_type_row(int t) {
  // CLHEP's proton_mass_c2 and neutron_mass_c2 in GeV, which is what G4Proton and G4Neutron
  // are constructed with: 938.272013 and 939.56536 MeV.
  const double mp = 0.93827201299999996;
  const double mn = 0.93956536000000002;
  // `strangeness` is G4InuclElementaryParticle::getStrangeness, which is
  // `GetQuarkContent(3) - GetAntiQuarkContent(3)` - the NUMBER OF s QUARKS, not the PDG
  // strangeness quantum number. So K+ (u sbar) is -1 and K- (ubar s) is +1, and the lambda
  // (uds) is +1 where the PDG writes S = -1. Every sign below is the dumped one; getting the
  // convention "right" would break the two places the cascade uses it (the hyperon() predicate
  // and G4ElementaryParticleCollider's strangeness bookkeeping).
  switch (t) {
    case kProton:       return {mp, 1.0, 0, 1, 2212};
    case kNeutron:      return {mn, 0.0, 0, 1, 2112};
    case kPionPlus:     return {0.1395701, 1.0, 0, 0, 211};
    case kPionMinus:    return {0.1395701, -1.0, 0, 0, -211};
    case kPionZero:     return {0.1349766, 0.0, 0, 0, 111};
    case kPhoton:       return {0.0, 0.0, 0, 0, 22};
    case kKaonPlus:     return {0.49367699999999998, 1.0, -1, 0, 321};
    case kKaonMinus:    return {0.49367699999999998, -1.0, 1, 0, -321};
    case kKaonZero:     return {0.497614, 0.0, -1, 0, 311};
    case kKaonZeroBar:  return {0.497614, 0.0, 1, 0, -311};
    case kLambda:       return {1.115683, 0.0, 1, 1, 3122};
    case kSigmaPlus:    return {1.18937, 1.0, 1, 1, 3222};
    case kSigmaZero:    return {1.192642, 0.0, 1, 1, 3212};
    case kSigmaMinus:   return {1.197449, -1.0, 1, 1, 3112};
    case kXiZero:       return {1.3148599999999999, 0.0, 2, 1, 3322};
    case kXiMinus:      return {1.3217099999999999, -1.0, 2, 1, 3312};
    case kOmegaMinus:   return {1.67245, -1.0, 3, 1, 3334};
    case kDeuteron:     return {1.875613, 1.0, 0, 2, 1000010020};
    case kTriton:       return {2.8089209999999998, 1.0, 0, 3, 1000010030};
    case kHe3:          return {2.8083909999999999, 2.0, 0, 3, 1000020030};
    case kAlpha:        return {3.727379, 2.0, 0, 4, 1000020040};
    case kAntiProton:   return {mp, -1.0, 0, -1, -2212};
    case kAntiNeutron:  return {mn, 0.0, 0, -1, -2112};
    case kAntiDeuteron: return {1.875613, -1.0, 0, -2, -1000010020};
    case kAntiTriton:   return {2.8089209999999998, -1.0, 0, -3, -1000010030};
    case kAntiHe3:      return {2.8083909999999999, -2.0, 0, -3, -1000020030};
    case kAntiAlpha:    return {3.727379, -2.0, 0, -4, -1000020040};
    // The unbound dibaryons: sums, not measured masses. The dumped values are exactly these
    // sums (1.8765440259999999, 1.877837373, 1.87913072), which is the evidence that
    // G4Diproton, G4UnboundPN and G4Dineutron carry no binding energy.
    case kDiproton:     return {2.0 * mp, 2.0, 0, 2, 0};
    case kUnboundPN:    return {mp + mn, 1.0, 0, 2, 0};
    case kDineutron:    return {2.0 * mn, 0.0, 0, 2, 0};
    case kElectron:     return {0.00051099890999999997, -1.0, 0, 0, 11};
    case kPositron:     return {0.00051099890999999997, 1.0, 0, 0, -11};
    case kMuonMinus:    return {0.1056583715, -1.0, 0, 0, 13};
    case kMuonPlus:     return {0.1056583715, 1.0, 0, 0, -13};
    case kTauMinus:     return {1.7768600000000001, -1.0, 0, 0, 15};
    case kTauPlus:      return {1.7768600000000001, 1.0, 0, 0, -15};
    case kElectronNu:   return {0.0, 0.0, 0, 0, 12};
    case kAntiElectronNu: return {0.0, 0.0, 0, 0, -12};
    case kMuonNu:       return {0.0, 0.0, 0, 0, 14};
    case kAntiMuonNu:   return {0.0, 0.0, 0, 0, -14};
    case kTauNu:        return {0.0, 0.0, 0, 0, 16};
    case kAntiTauNu:    return {0.0, 0.0, 0, 0, -16};
    // G4InuclElementaryParticle::makeDefinition prints an error and returns 0 for anything
    // else, and getParticleMass then returns 0.0. A zero mass is indistinguishable from a
    // photon's, so this returns a NEGATIVE mass instead and every caller checks it - the same
    // convention core/particle.cuh adopted for its unhandled species, for the same reason.
    default:            return {-1.0, 0.0, 0, 0, 0};
  }
}

__host__ __device__ inline double inucl_particle_mass(int t) {
  return inucl_type_row(t).mass_GeV;
}
__host__ __device__ inline bool inucl_valid(int t) { return t != kNuclei; }

/// G4InuclElementaryParticle::type(const G4ParticleDefinition*), the inverse map, by PDG code.
///
/// Two properties of the Geant4 function do not survive into a pure lookup and are handled by
/// the callers rather than hidden here:
///   * K0S and K0L (310, 130) are mapped to kaonZero or kaonZeroBar **by a coin flip**
///     (`G4UniformRand() > 0.5`), because Bertini's tables are indexed by strangeness states and
///     the weak states are mixtures. That draws a random number, so it cannot live in a pure
///     function: `inucl_type_from_pdg_weak_kaon` below takes the deviate.
///   * anything unknown - including every nucleus - returns 0, which `valid()` rejects.
__host__ __device__ inline int inucl_type_from_pdg(int pdg) {
  switch (pdg) {
    case 2212: return kProton;
    case 2112: return kNeutron;
    case 211: return kPionPlus;
    case -211: return kPionMinus;
    case 111: return kPionZero;
    case 22: return kPhoton;
    case 321: return kKaonPlus;
    case -321: return kKaonMinus;
    case 311: return kKaonZero;
    case -311: return kKaonZeroBar;
    case 3122: return kLambda;
    case 3222: return kSigmaPlus;
    case 3212: return kSigmaZero;
    case 3112: return kSigmaMinus;
    case 3322: return kXiZero;
    case 3312: return kXiMinus;
    case 3334: return kOmegaMinus;
    case 1000010020: return kDeuteron;
    case 1000010030: return kTriton;
    case 1000020030: return kHe3;
    case 1000020040: return kAlpha;
    case -2212: return kAntiProton;
    case -2112: return kAntiNeutron;
    case -1000010020: return kAntiDeuteron;
    case -1000010030: return kAntiTriton;
    case -1000020030: return kAntiHe3;
    case -1000020040: return kAntiAlpha;
    case 11: return kElectron;
    case -11: return kPositron;
    case 12: return kElectronNu;
    case -12: return kAntiElectronNu;
    case 13: return kMuonMinus;
    case -13: return kMuonPlus;
    case 14: return kMuonNu;
    case -14: return kAntiMuonNu;
    case 15: return kTauMinus;
    case -15: return kTauPlus;
    case 16: return kTauNu;
    case -16: return kAntiTauNu;
    default: return kNuclei;   // 0, which valid() rejects
  }
}

/// The K0S/K0L arm of G4InuclElementaryParticle::type, with its deviate made explicit.
/// `u > 0.5` selects kaonZero, matching `(G4UniformRand() > 0.5) ? kaonZero : kaonZeroBar`.
__host__ __device__ inline int inucl_type_from_pdg_weak_kaon(int pdg, double u) {
  if (pdg == 310 || pdg == 130) { return (u > 0.5) ? kKaonZero : kKaonZeroBar; }
  return inucl_type_from_pdg(pdg);
}

// =============================================================================================
// G4InuclSpecialFunctions
// =============================================================================================

/// G4InuclSpecialFunctions::G4cbrt(G4double) - NOT std::cbrt. It is
/// `sign(x) * G4Exp(G4Log(|x|)/3)`, which differs from the exact cube root by the difference
/// between exp(log(x)/3) and x^(1/3) - about one ulp of intermediate rounding, not the 1e-5
/// that G4Pow::A13's Taylor expansion costs.
///
/// **G4Exp and G4Log are std::exp and std::log on this platform.** `G4Exp.hh` opens with
/// `#ifdef WIN32 / # define G4Exp std::exp` and closes the VDT implementation with
/// `#endif /* WIN32 */`; `G4Log.hh` does the same. The oracle is built on Windows, so the
/// port's std::exp/std::log are EXACTLY what the oracle computed - which is why every angular
/// distribution below compares bitwise rather than to a few ulp. On Linux Geant4 the same
/// numbers would come from VDT's polynomial and would differ in the last bit or two; the
/// comparison would then need a tolerance, and this comment is where to look when it does.
/// Nothing here uses G4Pow's expA/logX: the whole INUCL tree includes G4Exp.hh and G4Log.hh,
/// and reaches G4Pow only for powN, Z13 and Z23.
__host__ __device__ inline double inucl_cbrt(double x) {
  if (x == 0.0) { return 0.0; }
  const double s = (x < 0.0) ? -1.0 : 1.0;
  return s * std::exp(std::log(std::fabs(x)) / 3.0);
}

/// G4Pow::powN(x, n) for the n = 0..3 the INUCL power series use - and it is **not** x^n.
///
///     if (0.0 == x) { return 0.0; }
///
/// is its first line, before any test on n, so `powN(0, 0)` is 0 and not 1. That single line
/// changes what `randomInuclPowers` and `G4InuclParamMomDst::GetMomentum` compute at zero
/// kinetic energy: every term of the series, including the constant one, vanishes. A
/// transcription that wrote `ekin^0 = 1` gives a non-zero momentum for a particle at rest and
/// a different number of rejection-loop iterations, which is how this was found - the draw
/// COUNT disagreed with the oracle at ekin = 0 before the value did. docs/RISK.md V120.
__host__ __device__ inline double g4pow_n(double x, int n) {
  if (x == 0.0) { return 0.0; }
  double res = 1.0;
  for (int i = 0; i < n; ++i) { res *= x; }
  return res;
}

/// G4InuclSpecialFunctions::G4cbrt(G4int) - a DIFFERENT function from the double overload: it
/// uses `G4Pow::Z13`, the tabulated integer cube root, not the exp/log expansion. The two
/// disagree, and which one a call site gets depends on whether its argument is an int, which
/// makes `G4cbrt(A)` with an `G4int A` and `G4cbrt(nuclearRadius)` two different functions in
/// the same expression - G4NucleiModel::generateModel has both.
__host__ __device__ inline double inucl_cbrt_int(int n) {
  if (n == 0) { return 0.0; }
  const double s = (n < 0) ? -1.0 : 1.0;
  const int an = (n < 0) ? -n : n;
  return s * data::g4pow_z13<double>(an);
}

/// G4Pow::Z23(Z) = Z13(Z)^2, which is what FermiEnergy uses. Not A23: the argument is an int
/// and goes through the tabulated cube root.
__host__ __device__ inline double inucl_z23(int z) {
  const double x = data::g4pow_z13<double>(z);
  return x * x;
}

/// G4InuclSpecialFunctions::getAL - the level-density-ish coefficient G4EquilibriumEvaporator
/// and G4Fissioner use. `0.76 + 2.2/G4cbrt(A)`, with the INTEGER overload of G4cbrt.
__host__ __device__ inline double inucl_get_al(int a) {
  return 0.76 + 2.2 / inucl_cbrt_int(a);
}

/// G4InuclSpecialFunctions::csNN - the neutron-neutron total cross section parametrisation, mb,
/// with e in MeV. Note the discontinuity at 40 MeV is in Geant4: the two branches do not meet
/// (-1174.8/1600 + 3088.5/40 + 5.3107 = 82.5 against 93074/1600 - 11.148/40 + 22.429 = 80.3).
__host__ __device__ inline double inucl_cs_nn(double e) {
  if (e < 40.0) { return -1174.8 / (e * e) + 3088.5 / e + 5.3107; }
  return 93074.0 / (e * e) - 11.148 / e + 22.429;
}

/// G4InuclSpecialFunctions::csPN - the proton-neutron one, same shape.
__host__ __device__ inline double inucl_cs_pn(double e) {
  if (e < 40.0) { return -5057.4 / (e * e) + 9069.2 / e + 6.9466; }
  return 239380.0 / (e * e) + 1802.0 / e + 27.147;
}

/// G4InuclSpecialFunctions::FermiEnergy. `ntype == 0` is the NEUTRON branch (it uses A - Z), so
/// the argument is not a type code from the enum above - it is an index. That is worth saying
/// because `ntype` is called with 0 and 1 at the call sites and reads as if it were a particle.
__host__ __device__ inline double inucl_fermi_energy(int a, int z, int ntype) {
  const double c = 55.4 / inucl_z23(a);
  const double arg = (ntype == 0) ? inucl_z23(a - z) : inucl_z23(z);
  return c * arg;
}

/// G4InuclSpecialFunctions::bindingEnergy - `G4NucleiProperties::GetBindingEnergy(A, Z)` with
/// the same guard G4NucleiProperties has, copied rather than encapsulated (the comment in
/// bindingEnergy.cc says so). MeV, which is why every caller divides by GeV.
__host__ __device__ inline double inucl_binding_energy(int a, int z) {
  if (a < 1 || z < 0 || z > a) { return 0.0; }
  return deex::binding_energy(a, z);
}

/// G4InuclNuclei::getNucleiMass(a, z, exc) - `(G4NucleiProperties::GetNuclearMass(a,z) + exc)`
/// converted from Geant4's MeV to Bertini's GeV. The excitation is added in MeV BEFORE the
/// conversion, which is what makes an excited fragment's four-momentum `setVectM(p, m + E*/1000)`
/// rather than `m + E*`.
///
/// Bertini calls this for a residual nucleus, for a recoil, and for every evaporation daughter,
/// so the whole de-excitation chain's energetics hang off P3's `deex::nuclear_mass`, which is
/// the AME12 table with Geant4's own formula fallback. `(0, 0)` is G4InuclNuclei's "dummy
/// without definition" case and returns 0 here as GetNuclearMass does.
__host__ __device__ inline double inucl_nuclei_mass(int a, int z, double exc_MeV = 0.0) {
  return (deex::nuclear_mass(a, z) + exc_MeV) * 0.001;
}

/// G4InuclSpecialFunctions::bindingEnergyAsymptotic - the smooth liquid-drop formula, used only
/// by G4Fissioner. MeV.
__host__ __device__ inline double inucl_binding_energy_asymptotic(int a, int z) {
  double x = 1.0 - 2.0 * double(z) / double(a);
  x *= x;
  const double x1 = inucl_cbrt_int(a);
  const double x2 = x1 * x1;
  const double x3 = 1.0 / x1;
  const double x4 = 1.0 / x2;
  double x5 = 1.0 - 0.62025 * x4;
  x5 *= x5;
  const double z13 = inucl_cbrt_int(z);
  return 17.035 * (1.0 - 1.846 * x) * a -
         25.8357 * (1.0 - 1.712 * x) * x2 * x5 -
         0.779 * z * (z - 1) * x3 *
             (1.0 - 1.5849 * x4 + 1.2273 / a + 1.5772 * x4 * x4) +
         0.4328 * z13 * z13 * z13 * z13 * x3 *
             (1.0 - 0.57811 * x3 - 0.14518 * x4 + 0.496 / a);
}

/// G4InuclSpecialFunctions::nucleiLevelDensity (nucleiLevelDensity.cc): a measured table from
/// A = 20 to A = 245, and `0.1*A` below 20. There is no upper guard - `NLD[a-20]` for A > 245
/// reads past the end of a 226-element array - so this port clamps and REPORTS instead, through
/// the sentinel below. A = 246 is reachable in principle (a uranium target plus a heavy
/// projectile) though not from any QBBC beam, so it is refused by name rather than approximated.
__host__ __device__ inline double inucl_nuclei_level_density(int a, bool& out_of_range) {
  out_of_range = false;
  if (a < 20) { return 0.1 * double(a); }
  if (a > 245) { out_of_range = true; return 0.0; }
  static const double nld[226] = {
      3.94, 3.84, 3.74, 3.64, 3.55, 4.35, 4.26, 4.09, 3.96, 4.18,
      4.39, 4.61, 4.82, 4.44, 4.44, 4.43, 4.42, 5.04, 5.66, 5.8,
      5.95, 5.49, 6.18, 7.11, 6.96, 7.2, 7.73, 6.41, 6.85, 6.77,
      6.91, 7.3, 7.2, 6.86, 8.06, 7.8, 7.82, 8.41, 8.13, 7.19,
      8.35, 8.13, 8.02, 8.93, 8.9, 9.7, 9.65, 10.55, 9.38, 9.72,
      10.66, 11.98, 12.76, 12.1, 12.86, 13.0, 12.81, 12.8, 12.65, 12.0,
      12.69, 14.05, 13.33, 13.28, 13.22, 13.17, 8.66, 11.03, 10.4, 13.47,
      10.17, 12.22, 11.62, 12.95, 13.15, 13.57, 12.87, 16.2, 14.71, 15.69,
      14.1, 18.56, 16.22, 16.7, 17.13, 17.0, 16.86, 16.2, 15.61, 16.8,
      17.93, 17.5, 16.97, 17.3, 17.6, 15.78, 16.8, 17.49, 16.03, 15.08,
      16.74, 17.74, 17.43, 18.1, 17.1, 19.01, 17.02, 17.0, 17.02, 18.51,
      17.2, 16.8, 16.97, 16.14, 16.91, 17.69, 15.5, 14.56, 14.35, 16.5,
      18.29, 17.8, 17.05, 21.31, 19.15, 19.5, 19.78, 20.3, 20.9, 21.9,
      22.89, 25.68, 24.6, 24.91, 23.24, 22.9, 22.46, 21.98, 21.64, 21.8,
      21.85, 21.7, 21.69, 23.7, 21.35, 23.03, 20.66, 21.81, 20.77, 22.2,
      22.58, 22.55, 21.45, 21.16, 21.02, 20.87, 22.09, 22.0, 21.28, 23.05,
      21.7, 21.18, 22.28, 23.0, 22.11, 23.56, 22.83, 24.88, 22.6, 23.5,
      23.89, 23.9, 23.94, 21.16, 22.3, 21.7, 21.19, 20.7, 20.29, 21.32,
      19.0, 17.93, 17.85, 15.7, 13.54, 11.9, 10.02, 10.48, 10.28, 11.72,
      13.81, 14.7, 15.5, 16.3, 17.2, 18.0, 18.9, 19.7, 20.6, 21.4,
      22.3, 23.1, 24.0, 24.8, 25.6, 26.5, 27.3, 28.2, 29.0, 29.9,
      30.71, 30.53, 31.45, 29.6, 30.2, 30.65, 30.27, 29.52, 30.08, 29.8,
      29.87, 30.25, 30.5, 29.8, 29.17, 28.67};
  return nld[a - 20];
}

// =============================================================================================
// G4InuclSpecialFunctions::paraMaker
// =============================================================================================

/// paraMaker's four Z-interpolated tables and the six-element parameter vectors it builds.
///
/// The interpolator here is constructed with `extrapolate = false`, which is the ONLY place in
/// the INUCL tree that turns extrapolation off - every G4CascadeSampler and every
/// G4ParamExpTwoBodyAngDst uses the default `true`. So a Z below 10 or above 70 is CLAMPED here
/// and extrapolated everywhere else, and getting that backwards would move every heavy target's
/// evaporation parameters.
struct ParaMakerParams {
  double ak[6];
  double cpa[6];
};

/// G4CascadeInterpolator<5>::getBin / interpolate on paraMaker's Z1 scale, with extrapolation
/// OFF. Written out here rather than sharing channel_tables.cuh's interpolator because the flag
/// differs and because this scale has five points, not thirty.
__host__ __device__ inline double para_interpolate(double z, const double (&yb)[5]) {
  static const double z1[5] = {10.0, 20.0, 30.0, 50.0, 70.0};
  const int last = 4;
  double xindex, xdiff, xbin;
  if (z < z1[0]) {
    xindex = 0.0;
    xbin = z1[1] - z1[0];
    xdiff = 0.0;                   // doExtrapolation == false
  } else if (z >= z1[last]) {
    xindex = double(last);
    xbin = z1[last] - z1[last - 1];
    xdiff = 0.0;
  } else {
    int i = 1;
    for (; i < last && z > z1[i]; ++i) {}
    xindex = double(i - 1);
    xbin = z1[i] - z1[i - 1];
    xdiff = z - z1[i - 1];
  }
  const double val = xindex + xdiff / xbin;
  const int i = (val < 0.0) ? 0 : (val > double(last)) ? last - 1 : int(val);
  const double frac = val - double(i);
  return (i == last) ? yb[last] : (yb[i] + frac * (yb[i + 1] - yb[i]));
}

__host__ __device__ inline ParaMakerParams para_maker_get_params(double z) {
  static const double AP[5] = {0.42, 0.58, 0.68, 0.77, 0.80};
  static const double CP[5] = {0.50, 0.28, 0.20, 0.15, 0.10};
  static const double AA[5] = {0.68, 0.82, 0.91, 0.97, 0.98};
  static const double CA[5] = {0.10, 0.10, 0.10, 0.08, 0.06};
  ParaMakerParams p;
  p.ak[0] = 0.0;
  p.cpa[0] = 0.0;
  p.ak[1] = para_interpolate(z, AP);
  p.ak[5] = para_interpolate(z, AA);
  p.cpa[1] = para_interpolate(z, CP);
  p.cpa[5] = para_interpolate(z, CA);
  p.ak[2] = p.ak[1] + 0.06;
  p.ak[3] = p.ak[1] + 0.12;
  p.ak[4] = p.ak[5] - 0.06;
  p.cpa[2] = p.cpa[1] * 0.5;
  p.cpa[3] = p.cpa[1] / 3.0;
  p.cpa[4] = 4.0 * p.cpa[5] / 3.0;
  return p;
}

/// paraMaker::getTruncated - the same two interpolations without the four derived entries.
__host__ __device__ inline void para_maker_get_truncated(double z, double& ak2, double& cp2) {
  static const double AP[5] = {0.42, 0.58, 0.68, 0.77, 0.80};
  static const double CP[5] = {0.50, 0.28, 0.20, 0.15, 0.10};
  ak2 = para_interpolate(z, AP);
  cp2 = para_interpolate(z, CP);
}

/// G4InuclSpecialFunctions::randomInuclPowers. A double power series in (ekin, S) with a
/// uniform S, used by both G4InuclParamAngDst and G4InuclParamMomDst.
///
/// Both powers go through `g4pow_n` above, which is zero at a zero base for EVERY exponent -
/// so at ekin = 0 the whole (ekin) series collapses to zero rather than to its constant term.
/// The (S) series is safe from that only because S is a uniform deviate and the engines here
/// never return exactly 0.
template <typename Rng>
__host__ __device__ inline double inucl_random_powers(double ekin, const double (&coeff)[4][4],
                                                      Rng& rng) {
  const double s = rng.uniform();
  double pq = 0.0;
  double pr = 0.0;
  for (int i = 0; i < 4; ++i) {
    double v = 0.0;
    for (int k = 0; k < 4; ++k) { v += coeff[i][k] * g4pow_n(ekin, k); }
    pq += v;
    pr += v * g4pow_n(s, i);
  }
  return std::sqrt(s) * (pr + (1.0 - pq) * (s * s * s * s));
}

/// G4InuclSpecialFunctions::randomGauss. Box-Muller with both deviates floored at 1e-6 and the
/// second also capped at 1 - 1e-6, so it never takes log(0). Note it returns only the SINE
/// branch and throws the cosine one away: two uniforms per Gaussian, not two Gaussians per two
/// uniforms.
template <typename Rng>
__host__ __device__ inline double inucl_random_gauss(double sigma, Rng& rng) {
  const double eps = 1.0e-6;
  const double twopi = 6.283185307179586476925286766559;
  double r1 = rng.uniform();
  r1 = (r1 > eps) ? r1 : eps;
  double r2 = rng.uniform();
  r2 = (r2 > eps) ? r2 : eps;
  r2 = (r2 < 1.0 - eps) ? r2 : 1.0 - eps;
  return sigma * std::sin(twopi * r1) * std::sqrt(-2.0 * std::log(r2));
}

template <typename Rng>
__host__ __device__ inline double inucl_random_phi(Rng& rng) {
  const double twopi = 6.283185307179586476925286766559;
  return twopi * rng.uniform();
}

/// G4InuclSpecialFunctions::randomCOS_SIN. `CT = 1 - 2u`, then sin from `sqrt(1 - CT^2)` -
/// which is always non-negative, so the polar angle is isotropic over the full sphere only
/// because the azimuth is thrown separately.
template <typename Rng>
__host__ __device__ inline void inucl_random_cos_sin(Rng& rng, double& ct, double& st) {
  ct = 1.0 - 2.0 * rng.uniform();
  st = std::sqrt(1.0 - ct * ct);
}

// =============================================================================================
// How an INUCL particle stores its momentum - which is NOT as a four-vector
// =============================================================================================
//
// `G4InuclParticle` holds a `G4DynamicParticle`, and a G4DynamicParticle keeps a **unit
// direction, a kinetic energy and a dynamical mass**. Every four-vector handed to one is taken
// apart into those three and rebuilt on every read:
//
//   G4InuclParticle::setMomentum(mom)                     [G4InuclParticle.cc, and its own
//     mass = getMass();                                    comment says "WARNING! Bertini code
//     if (|mass - mom.m()| <= 1e-5)                        doesn't do four-vectors; repair mass
//       pDP.Set4Momentum(mom*GeV/MeV);                     before use!"]
//     else
//       pDP.SetMomentum(mom.vect()*GeV/MeV);
//
//   G4DynamicParticle::Set4Momentum(p)                    [G4DynamicParticle.cc]
//     direction = p.vect().unit();
//     mass2 = t*t - |p|^2;
//     if      (mass2 < EnergyMRA2)              dynamicalMass = 0;
//     else if (|PDGmass^2 - mass2| > EnergyMRA2) dynamicalMass = sqrt(mass2);
//     else                                       dynamicalMass stays the PDG mass;
//     kineticEnergy = t - dynamicalMass;
//
//   G4DynamicParticle::Get4Momentum()
//     |p| = sqrt(Ekin^2 + 2*m*Ekin);  p4 = (direction*|p|, Ekin + m)
//
// with `EnergyMomentumRelationAllowance = 1e-2 keV`, so `EnergyMRA2 = 1e-10 MeV^2`.
//
// **It is not the identity, and for a photon it is not even close.** Rebuilding from
// (direction, Ekin, mass) re-imposes the mass shell exactly: a four-vector whose
// `e^2 - |p|^2` had drifted to a rounding residual comes back with the residual removed. For a
// massive particle that is a one-ulp change. For a PHOTON, whose `m()` after a boost is
// `sqrt(of a cancellation)` and therefore of order 1e-8 times its energy rather than 1e-16, it
// changes `getKinEnergyInTheTRS()` - which is `e - m` - in the ninth digit, and that energy is
// the abscissa of every angular distribution the collision then samples.
//
// Measured, not argued: with this round trip left out, the gamma-nucleon rows of
// `ref/oracle/bertini_epcollide.csv` disagree by up to 2e-7 relative while every other pair
// agrees to 2e-13 - and only when the target has a momentum, because a target at rest makes the
// boost the identity and the residual exactly zero. docs/RISK.md V125.
//
// The `*GeV/MeV` and `*MeV/GeV` round trip on top of that is real too (Geant4 stores in MeV and
// Bertini works in GeV) and is reproduced here for the same reason.

/// `EnergyMomentumRelationAllowance^2`, in MeV^2: (1e-2 keV)^2 = 1e-10.
constexpr double kInuclEnergyMRA2 = 1.0e-10;

/// G4InuclParticle::setMomentum followed by getMomentum(): what an INUCL particle gives back
/// after being handed @p mom (in GeV) as a particle of type @p type.
///
/// `pdg_mass_mev` is the dynamical mass the definition installed, in Geant4's own units. The
/// port's mass table is in GeV because every INUCL formula is, so the MeV value is the GeV one
/// scaled - which is what the comparison `|PDGmass2 - mass2| > EnergyMRA2` is done against, and
/// the allowance is 1e-10 MeV^2, twenty orders of magnitude above any scaling rounding.
__host__ __device__ inline LV inucl_store_momentum(const LV& mom, int type,
                                                   double* stored_ekin_gev = nullptr) {
  const double mass_gev = inucl_particle_mass(type);
  const Vec3d p{mom.v.x * 1000.0, mom.v.y * 1000.0, mom.v.z * 1000.0};
  const double t = mom.e * 1000.0;
  const double pmod2 = g4gpu::mag2(p);
  double dyn = mass_gev * 1000.0;      // theDynamicalMass, MeV, as SetDefinition left it
  double ekin = 0.0;
  Vec3d dir{1.0, 0.0, 0.0};            // both zero-momentum branches set (1,0,0)
  if (pmod2 > 0.0) {
    // CLHEP's Hep3Vector::unit() is `p *= 1/sqrt(mag2)` - a reciprocal multiply, not three
    // divisions. The difference is one ulp per component and this whole function exists
    // because of ulps, so it is written CLHEP's way.
    const double inv = 1.0 / std::sqrt(pmod2);
    dir = Vec3d{p.x * inv, p.y * inv, p.z * inv};
    if (std::fabs(mass_gev - mom.mag()) <= 1.0e-5) {   // Set4Momentum
      const double mass2 = t * t - pmod2;
      const double pdg2 = dyn * dyn;
      if (mass2 < kInuclEnergyMRA2) {
        dyn = 0.0;
      } else if (std::fabs(pdg2 - mass2) > kInuclEnergyMRA2) {
        dyn = std::sqrt(mass2);
      }
      ekin = t - dyn;
    } else {                                            // SetMomentum, mass left alone
      ekin = pmod2 / (std::sqrt(pmod2 + dyn * dyn) + dyn);
    }
  }
  if (stored_ekin_gev != nullptr) { *stored_ekin_gev = ekin * 0.001; }
  const double pm = std::sqrt(ekin * ekin + 2.0 * dyn * ekin);
  return LV(Vec3d{dir.x * pm * 0.001, dir.y * pm * 0.001, dir.z * pm * 0.001},
            (ekin + dyn) * 0.001);
}

/// `G4InuclParticle::getKineticEnergy()` - the STORED kinetic energy, which after the round trip
/// above is not `e - m` of the four-vector that went in: it is `t - dynamicalMass` in MeV,
/// scaled back. `G4ParticleLargerEkin`, the comparator `collide` sorts its final state with,
/// reads this and not the four-vector.
__host__ __device__ inline double inucl_stored_kinetic_energy(const LV& mom, int type) {
  double ekin = 0.0;
  inucl_store_momentum(mom, type, &ekin);
  return ekin;
}

}  // namespace g4gpu::physics::hadronic::bert

#endif  // G4GPU_BERTINI_INUCL_PARTICLE_CUH
