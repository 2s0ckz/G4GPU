// The six pre-equilibrium emission channels, and the emission-probability integral under them.
//
// Transcribed from source/processes/hadronic/models/pre_equilibrium/exciton_model/ in 11.1.1:
// G4VPreCompoundFragment (+ .icc), G4PreCompoundFragment, G4PreCompoundNucleon,
// G4PreCompoundIon, and G4PreCompoundNeutron / Proton / Deuteron / Triton / He3 / Alpha.
//
// Geant4 has this as a six-deep virtual hierarchy with one long-lived object per ejectile.
// A kernel cannot dispatch on a vtable, so the shape here is one state struct plus free
// functions that switch on the ejectile index. The switch is not an optimisation: the six
// differ in eight separate functions - GetRj, GetAlpha, GetBeta, FactorialFactor,
// CoalescenceFactor, the Coulomb-barrier object, the Kalbach/Chatterjee parameter row, and
// which of the two ProbabilityDistributionFunction bodies runs - and every one of them is
// written out below beside the Geant4 class it comes from.
//
// **The index order is the FACTORY's, and it is not (n, p, d, t, He3, alpha).**
// G4PreCompoundEmissionFactory::CreateFragmentVector pushes
//
//     neutron, proton, deuteron, ALPHA, triton, He3
//
// and G4PreCompoundFragmentVector::ChooseFragment walks a cumulative array in exactly that
// order. So the order is observable - it decides which channel a given uniform deviate
// selects - and it is the order used here and in the oracle's columns. The evaporation module
// next door (deexcitation/evaporation.cuh) uses n, p, d, t, He3, alpha, which is
// G4EvaporationDefaultGEMFactory's order; the two are different orders of the same six
// ejectiles and confusing them silently swaps tritons for alphas.
//
// **What is shared with P3's de-excitation, and what is not.** The inverse cross sections are
// literally the same functions: `kalbach_cross_section`, `chatterjee_cross_section` and
// `kalbach_power_parameter` in deexcitation/evaporation.cuh, and `coulomb_barrier` in
// deexcitation/coulomb_barrier.cuh, are Geant4's G4KalbachCrossSection,
// G4ChatterjeeCrossSection and G4CoulombBarrier, and Geant4 shares them between the two
// modules, so the port shares them too. What is NOT shared is how they are called:
//
//   evaporation  G4EvaporationProbability::CrossSection pivots Kalbach at 0.6*CB and
//                multiplies by the penetration factor (1 - 0.6*CB/K).
//   pre-compound G4PreCompoundFragment::CrossSection pivots at the FULL CB and applies no
//                penetration factor at all.
//
// The barrier itself is the same object (G4NeutronCoulombBarrier and friends, one per
// ejectile), so `coulomb_barrier(ejA, ejZ, resA, resZ, U)` serves both.
//
// **The barrier pointer is written before the object exists here too - and it is harmless.**
// Every one of the six declares its G4*CoulombBarrier as a member AFTER passing its address to
// the base initialiser, which is the exact shape of docs/RISK.md V42. It costs nothing here
// because G4VPreCompoundFragment's constructor only STORES the pointer; it does not write
// through it, as G4GEMChannel's does. Recorded because the two look identical at the
// constructor and only one of them is a defect.
//
// **theMass is the PDG mass, not the AME mass** - `particle->GetPDGMass()` in
// G4VPreCompoundFragment's constructor. For all six of these it happens to equal
// `G4NucleiProperties::GetNuclearMass`, because that function special-cases Z <= 2 to the
// same six CLHEP/G4 constants (see deexcitation/nuclear_masses.cuh's `nuclear_mass`), so
// `deex::nuclear_mass(a, z)` is used and the coincidence is recorded rather than relied on
// silently.
//
// **useSICB is dead.** G4VPreCompoundFragment initialises it to `true`, five comments in this
// package say its default is false, `G4PreCompoundEmission::UseSICB` and
// `G4PreCompoundFragmentVector::UseSICB` plumb it down to every channel - and nothing in
// 11.1.1 ever reads it. The superimposed barrier it used to gate is now the unconditional
// `elim = theCoulombBarrier*0.5` line in Initialize below. It is not ported, because there is
// nothing to port; see docs/RISK.md V49.
#ifndef G4GPU_PRECO_FRAGMENT_CUH
#define G4GPU_PRECO_FRAGMENT_CUH

#include <cmath>

#include "core/units.cuh"
#include "data/level_data.cuh"
#include "physics/hadronic/deexcitation/corrections.cuh"
#include "physics/hadronic/deexcitation/coulomb_barrier.cuh"
#include "physics/hadronic/deexcitation/deex_params.cuh"
#include "physics/hadronic/deexcitation/evaporation.cuh"
#include "physics/hadronic/deexcitation/fragment.cuh"
#include "physics/hadronic/deexcitation/nuclear_masses.cuh"

namespace g4gpu::preco {

namespace u = g4gpu::units;
using deex::Fragment;
using deex::LorentzVector;

/// G4PreCompoundEmissionFactory's order. See the file header: this is NOT the evaporation
/// module's order, and ChooseFragment makes it observable.
enum PreFragKind {
  kPreNeutron = 0,
  kPreProton = 1,
  kPreDeuteron = 2,
  kPreAlpha = 3,
  kPreTriton = 4,
  kPreHe3 = 5,
};
constexpr int kNumPreFragments = 6;

/// (A, Z) and the Kalbach/Chatterjee parameter row for each channel.
///
/// `index` is G4PreCompoundFragment's constructor, verbatim:
///     theZ == 0 -> 0 ;  theZ == 1 -> theA ;  otherwise -> theA + 1
/// which for these six gives n=0, p=1, d=2, t=3, He3=4, alpha=5 - the row order of
/// G4KalbachCrossSection::paramK, which is a DIFFERENT order again from the factory's. Both
/// are written out so that neither has to be re-derived.
struct PreFragSpec {
  int a, z, index;
};

__host__ __device__ inline const PreFragSpec* pre_frag_specs() {
  static const PreFragSpec s[kNumPreFragments] = {
    {1, 0, 0},   // G4PreCompoundNeutron
    {1, 1, 1},   // G4PreCompoundProton
    {2, 1, 2},   // G4PreCompoundDeuteron
    {4, 2, 5},   // G4PreCompoundAlpha    - factory slot 3, Kalbach row 5
    {3, 1, 3},   // G4PreCompoundTriton   - factory slot 4, Kalbach row 3
    {3, 2, 4},   // G4PreCompoundHe3      - factory slot 5, Kalbach row 4
  };
  return s;
}

/// True for the four G4PreCompoundIon subclasses (d, alpha, t, He3) and false for the two
/// G4PreCompoundNucleon ones. It selects which ProbabilityDistributionFunction body runs, and
/// they are not limiting cases of each other.
__host__ __device__ inline bool pre_frag_is_ion(int kind) {
  return kind != kPreNeutron && kind != kPreProton;
}

// ---------------------------------------------------------------------------------------------
// G4VPreCompoundFragment::Initialize, as the state it leaves behind
// ---------------------------------------------------------------------------------------------

/// Everything G4VPreCompoundFragment::Initialize computes, plus the two members
/// G4PreCompoundFragment mutates during the integral (`muu`, `probmax`).
///
/// Geant4's objects are long-lived and these are data members, so a field left over from the
/// previous fragment can be read by the next call. That staleness is reproduced exactly where
/// it is observable and headed off where it is not:
///
///   * `possible == false` corresponds to Initialize's early `return`, which leaves
///     theResMass, theReducedMass, theBindingEnergy and theResA13 at the PREVIOUS fragment's
///     values. They cannot be read, because that path also leaves theMaxKinEnergy at 0 and
///     IsItPossible requires theMaxKinEnergy > 0. So this struct is rebuilt per fragment and
///     nothing stale is carried; the reason it is safe to do so is that test.
///   * `theEmissionProbability` is likewise NOT reset when IsItPossible is false, and Geant4
///     leaves the previous fragment's value in it. Nothing in the pre-compound path reads
///     GetEmissionProbability() - ChooseFragment uses G4PreCompoundFragmentVector's own
///     cumulative array - so the stale value is unobservable and is not reproduced.
///
/// `optxs` is a per-CHANNEL member in Geant4 (`G4VPreCompoundFragment::OPTxs`), initialised to
/// 3 in the constructor and overwritten by `SetOPTxs` from fPrecoType. It is carried here
/// rather than read from the parameter block on every call because SetOPTxs is public and
/// unlocked, so the oracle can and does dump all five branches from one run.
///
/// `parent_has_levels` / `res_has_levels` are `G4NuclearLevelData::GetLevelManager(Z,A) !=
/// nullptr`, which is the third argument P3's `level_density` needs. Computed once here: both
/// nuclides are fixed for the life of the state, and with fLD = true neither is read at all.
struct PreFragState {
  int kind = 0;
  int a = 0, z = 0;           ///< theA, theZ
  int index = 0;              ///< the Kalbach/Chatterjee row
  int optxs = 3;              ///< OPTxs
  int frag_a = 0, frag_z = 0; ///< theFragA, theFragZ - the DECAYING nucleus
  int res_a = 0, res_z = 0;   ///< theResA, theResZ
  double res_a13 = 0.0;       ///< theResA13 = G4Pow::Z13(theResA)
  double coulomb_barrier = 0.0;
  double binding_energy = 0.0;
  double min_kin_energy = 0.0;
  double max_kin_energy = 0.0;
  double res_mass = 0.0;      ///< theResMass
  double reduced_mass = 0.0;  ///< theReducedMass
  double mass = 0.0;          ///< theMass, the ejectile's PDG mass
  double muu = 0.0;           ///< ComputePowerParameter; stays 0 for a neutron, as in Geant4
  double probmax = 0.0;       ///< the rejection majorant IntegrateEmissionProbability leaves
  double emission_probability = 0.0;
  bool parent_has_levels = false;
  bool res_has_levels = false;
  bool possible = false;      ///< Initialize did not take its early return
};

/// G4VPreCompoundFragment::Initialize.
///
/// The `elim` line is where the superimposed Coulomb barrier ended up: with OPTxs != 0 the
/// lower limit of the kinetic-energy integral is HALF the barrier, not the barrier and not
/// zero. It is unconditional - no flag reaches it - and it is the reason a charged ejectile's
/// spectrum starts below its barrier.
///
/// `theMaxKinEnergy` is the exact two-body asymptotic value in the fragment's rest frame:
/// with Ecm the parent's invariant mass, T_max = (Ecm^2 - Mres^2 + m^2)/(2 Ecm) - m, written
/// by Geant4 as `((Ecm-Mres)*(Ecm+Mres) + m*m)/(2Ecm) - m` so that the difference of two
/// ~2e5 MeV numbers is taken before the sum.
__host__ __device__ inline PreFragState pre_frag_initialize(int kind, const Fragment& frag,
                                                            const data::LevelTable& lt,
                                                            int optxs) {
  const PreFragSpec& s = pre_frag_specs()[kind];
  PreFragState st;
  st.kind = kind;
  st.a = s.a;
  st.z = s.z;
  st.index = s.index;
  st.optxs = optxs;
  st.mass = deex::nuclear_mass(s.a, s.z);   // == particle->GetPDGMass() for these six
  st.frag_a = frag.a;
  st.frag_z = frag.z;
  st.res_a = frag.a - s.a;
  st.res_z = frag.z - s.z;

  // theMinKinEnergy = theMaxKinEnergy = theCoulombBarrier = 0.0 before the test, so an
  // impossible channel reports a zero maximum and IsItPossible rejects it.
  if ((st.res_a < st.res_z) || (st.res_a < st.a) || (st.res_z < st.z)) { return st; }
  st.possible = true;

  st.parent_has_levels = (data::find_manager(lt, frag.z, frag.a) >= 0);
  st.res_has_levels = (data::find_manager(lt, st.res_z, st.res_a) >= 0);

  st.res_a13 = data::g4pow_z13<double>(st.res_a);
  st.coulomb_barrier = deex::coulomb_barrier(s.a, s.z, st.res_a, st.res_z, frag.excitation);

  const double elim = (0 == optxs) ? st.coulomb_barrier : st.coulomb_barrier * 0.5;

  st.res_mass = deex::nuclear_mass(st.res_a, st.res_z);
  st.reduced_mass = st.res_mass * st.mass / (st.res_mass + st.mass);
  st.binding_energy = st.res_mass + st.mass - frag.ground_state_mass;

  const double ecm = frag.momentum.mag();
  const double two_ecm = ecm + ecm;
  const double tmax =
      ((ecm - st.res_mass) * (ecm + st.res_mass) + st.mass * st.mass) / two_ecm - st.mass;
  st.max_kin_energy = (tmax > 0.0) ? tmax : 0.0;
  if (elim == 0.0) {
    st.min_kin_energy = 0.0;
  } else {
    const double tmin =
        ((st.mass + elim) * (two_ecm - st.mass - elim) + st.mass * st.mass) / two_ecm - st.mass;
    st.min_kin_energy = (tmin > 0.0) ? tmin : 0.0;
  }
  return st;
}

/// G4VPreCompoundFragment::IsItPossible (the .icc). The exciton bookkeeping test: there must
/// be enough neutron-like and proton-like PARTICLE excitons to build the ejectile out of, and
/// the kinetic-energy window must be open.
__host__ __device__ inline bool pre_frag_is_possible(const PreFragState& st, int n_particles,
                                                     int n_charged) {
  const int pplus = n_charged;
  const int pneut = n_particles - pplus;
  return (pneut >= st.a - st.z) && (pplus >= st.z) && (st.max_kin_energy > 0.0);
}

/// G4VPreCompoundFragment::GetEnergyThreshold - theMaxKinEnergy - theCoulombBarrier. Used only
/// by G4HETCFragment, which is refused (see precompound_emission.cuh); kept because it is the
/// one public number of this class that says whether the FULL barrier is surmountable, and a
/// reader comparing against HETC needs it to exist.
__host__ __device__ inline double pre_frag_energy_threshold(const PreFragState& st) {
  return st.max_kin_energy - st.coulomb_barrier;
}

// ---------------------------------------------------------------------------------------------
// The per-ejectile parameter functions
// ---------------------------------------------------------------------------------------------

/// GetRj(nParticles, nCharged) for each of the six, from the six .cc files.
///
/// This is the combinatorial probability of finding the ejectile's nucleons among the particle
/// excitons: for a nucleon a ratio, for a deuteron 2 p_c p_n / (P(P-1)), and so on. Every one
/// of them returns 0.0 rather than a fraction when there are not enough of the right kind -
/// and the guards are NOT the same expressions as IsItPossible's, which is why both exist:
/// the alpha's `nCharged >= 2 && (P - nCharged) >= 2` and IsItPossible's `pplus >= theZ &&
/// pneut >= theA - theZ` happen to agree, but the deuteron's and the triton's are written out
/// separately in each class and a difference would be silent.
__host__ __device__ inline double pre_frag_rj(int kind, int n_particles, int n_charged) {
  const double np = static_cast<double>(n_particles);
  const double nc = static_cast<double>(n_charged);
  const double nn = np - nc;
  switch (kind) {
    case kPreNeutron:
      // G4PreCompoundNeutron::GetRj
      return (n_particles > 0) ? nn / np : 0.0;
    case kPreProton:
      // G4PreCompoundProton::GetRj
      return (n_particles > 0) ? nc / np : 0.0;
    case kPreDeuteron:
      // G4PreCompoundDeuteron::GetRj
      if (n_charged >= 1 && (n_particles - n_charged) >= 1) {
        return 2.0 * nc * nn / (np * (np - 1.0));
      }
      return 0.0;
    case kPreTriton:
      // G4PreCompoundTriton::GetRj
      if (n_charged >= 1 && (n_particles - n_charged) >= 2) {
        return (3.0 * nc * nn * (nn - 1.0)) / (np * (np - 1.0) * (np - 2.0));
      }
      return 0.0;
    case kPreHe3:
      // G4PreCompoundHe3::GetRj
      if (n_charged >= 2 && (n_particles - n_charged) >= 1) {
        return (3.0 * nc * (nc - 1.0) * nn) / (np * (np - 1.0) * (np - 2.0));
      }
      return 0.0;
    case kPreAlpha:
      // G4PreCompoundAlpha::GetRj
      if (n_charged >= 2 && (n_particles - n_charged) >= 2) {
        return (6.0 * nc * (nc - 1.0)) * (nn * (nn - 1.0)) /
               ((np * (np - 1.0)) * ((np - 2.0) * (np - 3.0)));
      }
      return 0.0;
    default:
      return 0.0;
  }
}

/// The quartic in Z that four of the six alphas share, from G4PreCompoundProton::GetAlpha:
///     C(Z) = (((0.15417e-06 Z - 0.29875e-04) Z + 0.21071e-02) Z - 0.66612e-01) Z + 0.98375
/// clamped to 0.10 at Z >= 70. Written once because it is one expression appearing four times
/// with four different multipliers - and, crucially, evaluated at two DIFFERENT charges: the
/// proton uses theResZ (the residual) and the deuteron and triton use theFragZ (the decaying
/// nucleus). That asymmetry is Geant4's and it is not a typo the port may tidy: for the proton
/// the two differ by exactly one unit of charge.
__host__ __device__ inline double pre_frag_alpha_c(int Z) {
  if (Z >= 70) { return 0.10; }
  const double z = static_cast<double>(Z);
  return ((((0.15417e-06 * z) - 0.29875e-04) * z + 0.21071e-02) * z - 0.66612e-01) * z + 0.98375;
}

/// The piecewise-linear C(Z) that He3 and the alpha use instead - four bands rather than a
/// quartic, and Z <= 30 gives a flat 0.10 where the quartic is still falling. A different
/// parameterisation, not a limiting case.
__host__ __device__ inline double pre_frag_alpha_c_he(int Z) {
  if (Z <= 30) { return 0.10; }
  if (Z <= 50) { return 0.1 - (Z - 30) * 0.001; }
  if (Z < 70) { return 0.08 - (Z - 50) * 0.001; }
  return 0.06;
}

/// GetAlpha() for each of the six.
///
/// **Reached only by the OPTxs == 0 inverse cross section** (G4PreCompoundFragment::GetOpt0),
/// and fPrecoType is 3, so nothing in QBBC's configuration evaluates it. Transcribed rather
/// than refused because it IS observable: `G4VPreCompoundFragment::SetOPTxs` is public and is
/// not locked by the run state, so the oracle sets OPTxs = 0 on a channel object and dumps the
/// resulting emission probability - which makes GetAlpha and GetBeta checked numbers rather
/// than unvalidated transcription.
__host__ __device__ inline double pre_frag_get_alpha(const PreFragState& st) {
  switch (st.kind) {
    case kPreNeutron:
      // G4PreCompoundNeutron::GetAlpha - the only one built from a RADIUS and not a charge.
      return 0.76 + 2.2 / st.res_a13;
    case kPreProton:
      return 1.0 + pre_frag_alpha_c(st.res_z);          // theResZ
    case kPreDeuteron:
      return 1.0 + 0.5 * pre_frag_alpha_c(st.frag_z);   // theFragZ
    case kPreTriton:
      return 1.0 + pre_frag_alpha_c(st.frag_z) / 3.0;   // theFragZ
    case kPreHe3:
      return 1.0 + pre_frag_alpha_c_he(st.frag_z) * (4.0 / 3.0);
    case kPreAlpha:
      return 1.0 + pre_frag_alpha_c_he(st.frag_z);
    default:
      return 1.0;
  }
}

/// GetBeta() for each of the six. Only the neutron has a formula; the proton overrides with
/// -theCoulombBarrier and the four ions inherit G4PreCompoundIon::GetBeta, which is the same
/// thing. So `beta` is the negated barrier for five of the six, and the neutron's expression
/// is divided by its own GetAlpha - which means GetOpt0's `1 + beta/ekin` carries a 1/alpha
/// the other five do not.
__host__ __device__ inline double pre_frag_get_beta(const PreFragState& st) {
  if (st.kind == kPreNeutron) {
    // G4PreCompoundNeutron::GetBeta
    const double a = pre_frag_get_alpha(st);
    return (2.12 / (st.res_a13 * st.res_a13) - 0.05) * u::MeV<double>() / a;
  }
  return -st.coulomb_barrier;
}

/// G4PreCompoundIon::FactorialFactor(N, P) for the four ions. The 2013 fix J.M. Quesada's
/// comment names ("FactorialFactor fixed", 05.07.2013) is in the triton, He3 and alpha forms:
/// they are products of six or eight consecutive integers over 12 or 144, and the deuteron's
/// is (N-1)(N-2)(P-1)P/2.
///
/// These go NEGATIVE for small exciton numbers - the alpha's has factors (N-4) and (P-3), so
/// N = 3 or P = 2 makes it negative - and Geant4 does not guard it. It cannot escape into the
/// probability, because IsItPossible already required four particle excitons for an alpha, and
/// N >= P always. The arithmetic is transcribed as written and the guard is left where Geant4
/// put it.
__host__ __device__ inline double pre_frag_factorial_factor(int kind, int N, int P) {
  switch (kind) {
    case kPreDeuteron:
      return static_cast<double>((N - 1) * (N - 2) * (P - 1) * P) * 0.5;
    case kPreTriton:
    case kPreHe3:
      // G4PreCompoundTriton and G4PreCompoundHe3 have the identical expression.
      return static_cast<double>(((N - 3) * (P - 2) * (N - 2)) * ((P - 1) * (N - 1) * P)) / 12.0;
    case kPreAlpha:
      return static_cast<double>(((N - 4) * (P - 3) * (N - 3) * (P - 2)) *
                                 ((N - 2) * (P - 1) * (N - 1) * P)) / 144.0;
    default:
      return 0.0;   // never called for a nucleon: G4PreCompoundNucleon has no such member
  }
}

/// G4PreCompoundIon::CoalescenceFactor(A) - `A` here is the DECAYING nucleus's mass number,
/// theFragA, not the ejectile's, because G4PreCompoundIon calls it as
/// `CoalescenceFactor(theFragA)`. 16/A for a deuteron, 243/A^2 for a triton or He3, 4096/A^3
/// for an alpha, i.e. (2 A_ej)^... - written out, because the pattern 2^4, 3^5, 4^6 is not a
/// formula anyone should re-derive.
__host__ __device__ inline double pre_frag_coalescence_factor(int kind, int fragA) {
  const double A = static_cast<double>(fragA);
  switch (kind) {
    case kPreDeuteron: return 16.0 / A;
    case kPreTriton:
    case kPreHe3:      return 243.0 / (A * A);
    case kPreAlpha:    return 4096.0 / (A * A * A);
    default:           return 0.0;
  }
}

// ---------------------------------------------------------------------------------------------
// G4PreCompoundFragment::CrossSection and GetOpt0
// ---------------------------------------------------------------------------------------------

/// G4PreCompoundFragment::GetOpt0 - Dostrovsky's inverse cross section, in millibarn.
///
/// `1.e+25 * pi * r0*r0 * theResA13` with `r0 = fR0 * theResA13`: that is
/// pi * fR0^2 * A_res, NOT pi * fR0^2 * A_res^(2/3), because theResA13 appears three times.
/// Whether that third factor is intended is not a question this port can answer - it is what
/// the installed Geant4 computes, and the OPTxs == 0 emission probability the oracle dumps
/// contains it. The 1e+25 converts mm^2 to millibarn (1 mb = 1e-25 mm^2), which is why fR0
/// being in mm matters.
__host__ __device__ inline double pre_frag_opt0(const PreFragState& st, double ekin) {
  const double r0 = deex::deex_params().r0 * st.res_a13;
  return 1.e+25 * u::pi<double>() * r0 * r0 * st.res_a13 * pre_frag_get_alpha(st) *
         (1.0 + pre_frag_get_beta(st) / ekin);
}

/// G4PreCompoundFragment::CrossSection(ekin), all five OPTxs branches.
///
/// The default is OPTxs = 3 -> Kalbach, and the two differences from the evaporation module's
/// call of the same function are in the file header: the pivot is the FULL Coulomb barrier and
/// there is no (1 - elim/K) penetration factor.
///
/// `OPTxs == 4 && theMaxKinEnergy < 10.` uses the Dostrovsky form for a channel whose window
/// is narrower than 10 MeV and Kalbach above it - a hybrid, and the bare `10.` is 10 MeV
/// because MeV is 1 in these units.
__host__ __device__ inline double pre_frag_cross_section(const PreFragState& st, double ekin) {
  if (st.optxs == 0 || (st.optxs == 4 && st.max_kin_energy < 10.0 * u::MeV<double>())) {
    return pre_frag_opt0(st, ekin);
  }
  if (st.optxs <= 2) {
    return deex::chatterjee_cross_section(ekin, st.coulomb_barrier, st.res_a13, st.muu,
                                          st.index, st.z, st.res_a);
  }
  return deex::kalbach_cross_section(ekin, st.coulomb_barrier, st.res_a13, st.muu, st.index,
                                     st.z, st.a, st.res_a);
}

// ---------------------------------------------------------------------------------------------
// The two ProbabilityDistributionFunction bodies
// ---------------------------------------------------------------------------------------------

/// The exciton configuration a PDF reads off the fragment: G4Fragment's numberOfParticles,
/// numberOfCharged and numberOfHoles. Carried as a separate struct because P3's `Fragment`
/// does not have them - the de-excitation module never looks at an exciton count - and adding
/// three fields to a struct another package owns is not this package's to do.
struct Excitons {
  int particles = 0;
  int charged = 0;
  int holes = 0;

  __host__ __device__ int total() const { return particles + holes; }  ///< GetNumberOfExcitons
};

/// G4PreCompoundNucleon::ProbabilityDistributionFunction - the neutron and proton.
///
/// The exciton-model emission rate of Gudima, Mashnik and Toneev: the density of (P, H)
/// states at U in the parent times the density of (P-1, H) states at U - T - B in the
/// residual, with the inverse cross section supplying the detailed-balance factor. Written by
/// Geant4 as a ratio of two single-particle level densities g0 (parent, at its own excitation)
/// and g1 (residual, at ZERO excitation - `GetLevelDensity(theResZ, theResA, 0.0)`), raised to
/// the power N-2.
///
/// A0 and A1 are the Pauli-blocking energies. Note A1 = (A0 - 0.5*P)/g1: it is A0 in units of
/// the parent's density rescaled by the residual's, and the -0.5*P is the one particle
/// removed. Both `E0 <= 0` and `E1 <= 0` return exactly zero, which is how the kinematic edge
/// of the spectrum is imposed - not by clipping the kinetic energy.
///
/// `fact = 2 mb / (pi^2 hbarc^3)` and the trailing `g1/(E0 g0 g0)` are Geant4's grouping.
__host__ __device__ inline double pre_nucleon_pdf(const PreFragState& st, double ekin,
                                                  const Fragment& frag, const Excitons& ex) {
  const double U = frag.excitation;
  const int P = ex.particles;
  const int H = ex.holes;
  const int N = P + H;

  const double sixoverpi2 = 6.0 / deex::pi2();
  const double g0 =
      sixoverpi2 * deex::level_density(st.frag_z, st.frag_a, U, st.parent_has_levels);
  const double g1 =
      sixoverpi2 * deex::level_density(st.res_z, st.res_a, 0.0, st.res_has_levels);

  const double A0 = (P * P + H * H + P - 3 * H) / (4.0 * g0);
  const double A1 = (A0 - 0.5 * P) / g1;

  const double E0 = U - A0;
  if (E0 <= 0.0) { return 0.0; }

  const double E1 = U - ekin - st.binding_energy - A1;
  if (E1 <= 0.0) { return 0.0; }

  const double rj = pre_frag_rj(st.kind, P, ex.charged);
  const double xs = pre_frag_cross_section(st, ekin);
  if (rj < 0.0 || xs < 0.0) { return 0.0; }

  const double hc = u::hbarc<double>();
  const double fact = 2.0 * deex::millibarn() / (deex::pi2() * hc * hc * hc);
  return fact * st.reduced_mass * rj * xs * ekin * P * (N - 1) *
         deex::g4pow_pow_n(g1 * E1 / (g0 * E0), N - 2) * g1 / (E0 * g0 * g0);
}

/// G4PreCompoundIon::ProbabilityDistributionFunction - the deuteron, triton, He3 and alpha.
///
/// Same exciton model with the ejectile treated as a cluster of A nucleons pulled out of the
/// particle excitons: hence FactorialFactor (the number of ways to choose them),
/// CoalescenceFactor (the phase-space penalty for their being close in momentum), and a THIRD
/// density gj at the cluster's own energy `efinal = T + B`. `gj` is set to `g1` on the line
/// above, so the third density is the residual's and the separate name is bookkeeping - but it
/// is kept, because it appears three times and reading it as g1 hides which factor is which.
///
/// Four differences from the nucleon form that are easy to read past:
///   * E1 is built from `theMaxKinEnergy - eKin - A1`, not from U - eKin - B - A1. The
///     kinematic maximum has the binding energy in it already.
///   * A1 is floored at zero, and so are E1 and Ej. The nucleon form floors nothing and
///     returns zero instead, so the two treat the edge of the spectrum differently: an ion's
///     probability goes to zero smoothly through powN(0, n), a nucleon's is cut off.
///   * the exponents are N-A-1 and A-1, and for a nucleon (A = 1) they would be N-2 and 0 -
///     so the nucleon form is the A = 1 case of this only if gj/g1 and the factorials are
///     dropped, which they are not. Two functions, not one with a parameter.
///   * the prefactor is `0.75 mb/(pi fR0^3)` times sqrt(2/(mu efinal)), a non-relativistic
///     velocity, where the nucleon form has 2 mb/(pi^2 hbarc^3) times mu.
__host__ __device__ inline double pre_ion_pdf(const PreFragState& st, double ekin,
                                              const Fragment& frag, const Excitons& ex) {
  const double efinal = ekin + st.binding_energy;
  if (efinal <= 0.0) { return 0.0; }

  const double U = frag.excitation;
  const int P = ex.particles;
  const int H = ex.holes;
  const int A = st.a;
  const int N = P + H;

  const double sixoverpi2 = 6.0 / deex::pi2();
  const double g0 =
      sixoverpi2 * deex::level_density(st.frag_z, st.frag_a, U, st.parent_has_levels);
  const double g1 =
      sixoverpi2 * deex::level_density(st.res_z, st.res_a, 0.0, st.res_has_levels);
  const double gj = g1;

  const double A0 = (P * P + H * H + P - 3 * H) / (4.0 * g0);
  double A1 = (A0 * g0 + A * (A - 2 * P - 1) * 0.25) / g1;
  if (A1 < 0.0) { A1 = 0.0; }

  const double E0 = U - A0;
  if (E0 <= 0.0) { return 0.0; }

  double E1 = st.max_kin_energy - ekin - A1;
  if (E1 < 0.0) { E1 = 0.0; }

  const double Aj = A * (A + 1) / (4.0 * gj);
  double Ej = efinal - Aj;
  if (Ej < 0.0) { Ej = 0.0; }

  const double rj = pre_frag_rj(st.kind, P, ex.charged);
  const double xs = pre_frag_cross_section(st, ekin);

  const double r0 = deex::deex_params().r0;
  const double fact = 0.75 * deex::millibarn() / (u::pi<double>() * r0 * r0 * r0);

  return fact * ekin * xs * rj * pre_frag_coalescence_factor(st.kind, st.frag_a) *
         pre_frag_factorial_factor(st.kind, N, P) *
         std::sqrt(2.0 / (st.reduced_mass * efinal)) *
         deex::g4pow_pow_n(g1 * E1 / (g0 * E0), N - A - 1) *
         deex::g4pow_pow_n(gj * Ej / (g0 * E0), A - 1) * gj * g1 /
         (g0 * g0 * E0 * st.res_a);
}

/// The virtual dispatch of ProbabilityDistributionFunction, as a branch.
__host__ __device__ inline double pre_frag_pdf(const PreFragState& st, double ekin,
                                               const Fragment& frag, const Excitons& ex) {
  return pre_frag_is_ion(st.kind) ? pre_ion_pdf(st, ekin, frag, ex)
                                  : pre_nucleon_pdf(st, ekin, frag, ex);
}

// ---------------------------------------------------------------------------------------------
// G4PreCompoundFragment::IntegrateEmissionProbability and CalcEmissionProbability
// ---------------------------------------------------------------------------------------------

/// G4PreCompoundFragment::IntegrateEmissionProbability.
///
/// A midpoint rule with ONE MEV bins and a floor of four bins - `G4int nbins = del*den` with
/// `den = 1/MeV` is a truncating double-to-int conversion, so a 7.9 MeV window gets 7 bins of
/// 1.129 MeV and not 8 of 1 MeV. The bin width is therefore not 1 MeV; it is
/// (up - low)/floor(up - low in MeV), and a window under 4 MeV always gets exactly 4 bins.
///
/// The `if (y < sum*0.01) break;` is a convergence exit, not an error path, and it is
/// observable twice over: it truncates the integral (so the returned probability is not the
/// full integral of the PDF), and it stops `probmax` from seeing the rest of the curve. Since
/// the integrand rises from the barrier, falls, and the break can only fire on the falling
/// side, the majorant it leaves is still an upper bound over the sampled region - but only
/// because of that shape, and `SampleKineticEnergy` multiplies it by 1.25 anyway.
///
/// `probmax` is a member in Geant4, mutated here through the state struct for the same reason.
__host__ __device__ inline double pre_frag_integrate(PreFragState& st, double low, double up,
                                                     const Fragment& frag, const Excitons& ex) {
  const double den = 1.0 / u::MeV<double>();
  double del = up - low;
  int nbins = static_cast<int>(del * den);
  if (nbins < 4) { nbins = 4; }
  del /= static_cast<double>(nbins);
  double e = low + 0.5 * del;
  st.probmax = pre_frag_pdf(st, e, frag, ex);
  double sum = st.probmax;
  for (int i = 1; i < nbins; ++i) {
    e += del;
    const double y = pre_frag_pdf(st, e, frag, ex);
    if (y > st.probmax) { st.probmax = y; }
    sum += y;
    if (y < sum * 0.01) { break; }
  }
  return sum * del;
}

/// G4PreCompoundFragment::CalcEmissionProbability.
///
/// The `theMaxKinEnergy <= theMinKinEnergy` test is what closes a charged channel whose half
/// barrier already exceeds the two-body maximum, and it returns before `muu` is computed - so
/// a closed channel leaves the previous residual's power parameter behind. Unobservable, for
/// the reason PreFragState's comment gives.
__host__ __device__ inline double pre_frag_emission_probability(PreFragState& st,
                                                                const Fragment& frag,
                                                                const Excitons& ex) {
  st.emission_probability = 0.0;
  if (st.max_kin_energy <= st.min_kin_energy) { return 0.0; }
  if (0 < st.index) { st.muu = deex::kalbach_power_parameter(st.res_a, st.index); }
  st.emission_probability =
      pre_frag_integrate(st, st.min_kin_energy, st.max_kin_energy, frag, ex);
  return st.emission_probability;
}

/// G4PreCompoundFragment::SampleKineticEnergy - rejection against `1.25 * probmax` over the
/// whole window, at most 100 tries, and on the 100th failure it returns the last T anyway
/// rather than reporting anything. `tries` records how many draws it took so a test can see
/// the failure rate; a run that exhausts the 100 is reported by the caller.
///
/// `probmax *= toler` mutates the majorant, exactly as Geant4 mutates the member. It matters
/// that this happens once per emission and not once per try.
template <typename Rng>
__host__ __device__ inline double pre_frag_sample_kinetic_energy(PreFragState& st,
                                                                 const Fragment& frag,
                                                                 const Excitons& ex, Rng& rng,
                                                                 int* tries = nullptr) {
  const double delta = st.max_kin_energy - st.min_kin_energy;
  const double toler = 1.25;
  st.probmax *= toler;
  double T = 0.0;
  int i = 0;
  for (; i < 100; ++i) {
    T = st.min_kin_energy + delta * rng.uniform();
    const double prob = pre_frag_pdf(st, T, frag, ex);
    if (st.probmax * rng.uniform() <= prob) { break; }
  }
  if (tries != nullptr) { *tries = i + 1; }
  return T;
}

}  // namespace g4gpu::preco

#endif
