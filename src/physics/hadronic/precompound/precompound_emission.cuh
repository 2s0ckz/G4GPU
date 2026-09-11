// Choosing which of the six channels emits, and the kinematics of the emission.
//
// Transcribed from G4PreCompoundEmission (+ .hh's inline GetTotalProbability),
// G4PreCompoundFragmentVector, G4PreCompoundEmissionFactory and
// G4VPreCompoundEmissionFactory (11.1.1).
//
// Three steps, and each has something in it that a reading of the physics would not predict:
//
//   1. CalculateProbabilities  Initialize every channel on this fragment, ask IsItPossible,
//                              integrate the ones that pass, and accumulate a CUMULATIVE
//                              array. The return value is the total.
//   2. ChooseFragment          `x = probabilities[n-1]*rand()`, then the first index with
//                              `x <= probabilities[i]`. The array is cumulative and the
//                              closed inequality means a channel with zero probability can be
//                              selected when the total is zero - see below.
//   3. PerformEmission         sample T from the chosen channel, give it an isotropic
//                              direction in the fragment's rest frame (or the angular
//                              generator's, if fUseAngularGen), boost, and subtract.
//
// **The order the channels sit in is G4PreCompoundEmissionFactory's**: neutron, proton,
// deuteron, ALPHA, triton, He3. Step 2 walks that order, so it is observable, and
// precompound_fragment.cuh's PreFragKind is in it.
//
// **REFUSED, by name: G4HETCEmissionFactory and the ten G4HETC* classes.** fUseHETC is false
// in 11.1.1, and unlike the OPTxs branches this one cannot be reached from the oracle at all:
// `G4PreCompoundEmission::SetHETCModel()` is public, but the flag that calls it lives in
// G4DeexPrecoParameters, whose setters all return early unless the run state is
// G4State_PreInit - and the dump program runs after initialisation. So HETC's own
// CalcEmissionProbability (a closed-form product over Pf, Hf, Nf and a spin factor, not an
// integral), its GetSpinFactor and K, and its GetAlpha/GetBeta - about 800 lines over ten
// classes - would be transcription with no possible oracle, which is the case P3 refused the
// Weisskopf width for. `preco_refusal()` names it at the point it would be selected.
#ifndef G4GPU_PRECO_EMISSION_CUH
#define G4GPU_PRECO_EMISSION_CUH

#include <cmath>

#include "core/units.cuh"
#include "core/vec3.cuh"
#include "data/g4pow.hh"
#include "data/level_data.cuh"
#include "physics/hadronic/deexcitation/corrections.cuh"
#include "physics/hadronic/deexcitation/deex_params.cuh"
// For deex::clhep_unit: CLHEP's Hep3Vector::unit(), which returns the ZERO vector for a zero
// vector where src/core/vec3.cuh's normalize() returns (0, 0, 1). The angular generator needs
// the CLHEP semantics and P3 already wrote them down, so they are used rather than repeated.
#include "physics/hadronic/deexcitation/excitation_handler.cuh"
#include "physics/hadronic/deexcitation/fragment.cuh"
#include "physics/hadronic/precompound/precompound_fragment.cuh"

namespace g4gpu::preco {

namespace u = g4gpu::units;
using deex::Fragment;
using deex::LorentzVector;
using deex::Vec3d;

/// Everything this package can refuse, so that a refusal is a name and not a silent branch.
/// Mirrors deex::DeexStatus' shape.
struct PrecoRefusal {
  /// fUseHETC would have selected G4HETCEmissionFactory. See the file header.
  bool hetc = false;
  /// fUseGNASH selected G4GNASHTransitions. Transcribed, but its own dead-branch behaviour
  /// (precompound_transitions.cuh's header) means no pre-equilibrium emission happens, so a
  /// caller is told rather than silently getting pure equilibrium emission.
  bool gnash = false;
  /// A fragment with lambdas != 0 - G4PreCompoundModel::DeExcite sends it straight to the
  /// handler, which refuses it in turn. Reported here so the reason is visible at this level.
  bool hyper_fragment = false;
  /// ChooseFragment was reached with a zero total probability. Geant4 returns channel 0 with
  /// probability zero and then samples a kinetic energy from an empty window.
  bool zero_total_probability = false;
  /// SampleKineticEnergy used all 100 rejection tries and returned the last draw anyway.
  bool sampler_exhausted = false;
  /// The exciton bookkeeping went inconsistent; G4Fragment's setters throw here.
  bool exciton_count = false;
  /// The output buffer was too small for the pre-equilibrium products.
  bool capacity = false;
  /// G4PreCompoundModel's own 1000-iteration guard, where Geant4 prints a JustWarning and
  /// falls through to equilibrium emission - so this one is a report, not a failure.
  bool loop_limit = false;

  int refused_z = 0, refused_a = 0;

  __host__ __device__ bool any() const {
    return hetc || gnash || hyper_fragment || zero_total_probability || sampler_exhausted ||
           exciton_count || capacity || loop_limit;
  }
};

/// The six channels' states plus the cumulative probability array, i.e. what
/// G4PreCompoundFragmentVector holds between CalculateProbabilities and ChooseFragment.
struct EmissionChannels {
  PreFragState st[kNumPreFragments];
  double cumulative[kNumPreFragments] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  double total = 0.0;
};

/// G4PreCompoundFragmentVector::CalculateProbabilities, which is what
/// G4PreCompoundEmission::GetTotalProbability forwards to.
///
/// Note that `Initialize` runs on EVERY channel including the impossible ones - it has to,
/// because IsItPossible reads theMaxKinEnergy, which Initialize sets. The cumulative array is
/// written for every index, so a zero-probability channel repeats its predecessor's value and
/// the closed `x <= probabilities[i]` in ChooseFragment can never select it unless the
/// preceding total is also zero.
__host__ __device__ inline EmissionChannels
emission_probabilities(const Fragment& frag, const Excitons& ex, const data::LevelTable& lt,
                       int optxs) {
  EmissionChannels ch;
  double probtot = 0.0;
  for (int i = 0; i < kNumPreFragments; ++i) {
    ch.st[i] = pre_frag_initialize(i, frag, lt, optxs);
    const double prob = pre_frag_is_possible(ch.st[i], ex.particles, ex.charged)
                            ? pre_frag_emission_probability(ch.st[i], frag, ex)
                            : 0.0;
    probtot += prob;
    ch.cumulative[i] = probtot;
  }
  ch.total = probtot;
  return ch;
}

/// G4PreCompoundFragmentVector::ChooseFragment.
///
/// `x = probabilities[nChannels-1]*G4UniformRand()` and then the first `x <= probabilities[i]`.
/// The loop has no fallthrough guard: if every cumulative entry were below x it would return
/// `(*theChannels)[nChannels]`, one past the end - which cannot happen, because x is the total
/// times a deviate below 1 and the last entry IS the total. With a total of exactly zero, x is
/// zero and index 0 is returned with `0 <= 0`; the caller reports that, because the neutron
/// channel then samples from a window it was never given.
///
/// One uniform deviate, always.
template <typename Rng>
__host__ __device__ inline int choose_fragment(const EmissionChannels& ch, Rng& rng,
                                               PrecoRefusal& ref) {
  const double x = ch.cumulative[kNumPreFragments - 1] * rng.uniform();
  int i = 0;
  for (; i < kNumPreFragments; ++i) {
    if (x <= ch.cumulative[i]) { break; }
  }
  if (i >= kNumPreFragments) { i = kNumPreFragments - 1; }
  if (!(ch.total > 0.0)) { ref.zero_total_probability = true; }
  return i;
}

// ---------------------------------------------------------------------------------------------
// G4PreCompoundEmission::rho and AngularDistribution
// ---------------------------------------------------------------------------------------------

/// G4Pow::logfactorial(n) = sum_{k=1..n} G4Log(k), accumulated in that order.
///
/// Geant4 builds a 512-entry table in G4Pow's constructor by accumulating `lz[i] = G4Log(i)`,
/// so it is a running sum of 512 fast logs and NOT lgamma(n+1): the two differ by the
/// accumulated rounding of up to 512 additions, about 1e-14 relative at the top of the table.
/// Reproduced by accumulating in the same order and direction, which is why this is a loop and
/// not a call to std::lgamma. `n` is bounded by 511 in Geant4 (the table's size); above that
/// Geant4 reads out of bounds, and this returns the sum it would have had, which is a
/// divergence recorded rather than hidden - no exciton number in this model comes close.
__host__ __device__ inline double log_factorial(int n) {
  double logf = 0.0;
  for (int i = 1; i <= n; ++i) { logf += std::log(static_cast<double>(i)); }
  return logf;
}

/// G4PreCompoundEmission::rho(p, h, gg, E, Ef) - the Ericson state density of a (p, h)
/// configuration at energy E, with the finite-well correction as an alternating sum over the
/// holes.
///
/// `Aph` is the Pauli energy; `E - Aph < 0` returns zero. The sum's j-th term subtracts one
/// more Fermi energy from the effective energy and stops as soon as that goes negative, and
/// each term carries a sign (-1)^j and a binomial coefficient built multiplicatively. The
/// `logmax = 200` clamp on the exponent is Geant4's protection against overflow and it BIASES
/// the result rather than reporting - a clamped term is 2.7e86 instead of whatever it was.
///
/// **Reached only through AngularDistribution, i.e. only when fUseAngularGen is true.** That
/// flag is false in 11.1.1 and cannot be changed after initialisation, so neither this nor
/// AngularDistribution below can be dumped from the installed Geant4 - `rho` and
/// `AngularDistribution` are private members and `PerformEmission`'s angular branch is chosen
/// in G4PreCompoundEmission's constructor from the locked parameter. What IS checkable is
/// `G4Pow::logfactorial`, which is public, so the one table this function depends on is
/// compared exactly and the arithmetic around it is transcription. That division is stated
/// here rather than left for a reader to discover.
__host__ __device__ inline double preco_rho(int p, int h, double gg, double E, double Ef) {
  const double Aph = (p * p + h * h + p - 3.0 * h) / (4.0 * gg);
  if (E - Aph < 0.0) { return 0.0; }

  const double log_const = (p + h) * std::log(gg) - log_factorial(p + h - 1) -
                           log_factorial(p) - log_factorial(h);

  double t1 = 1.0;
  double t2 = 1.0;
  const double logmax = 200.0;
  double logt3 = (p + h - 1) * std::log(E - Aph) + log_const;
  if (logt3 > logmax) { logt3 = logmax; }
  double tot = std::exp(logt3);

  double eeff = E - Aph;
  for (int j = 1; j <= h; ++j) {
    eeff -= Ef;
    if (eeff < 0.0) { break; }
    t1 *= -1.0;
    t2 *= static_cast<double>(h + 1 - j) / static_cast<double>(j);
    logt3 = (p + h - 1) * std::log(eeff) + log_const;
    if (logt3 > logmax) { logt3 = logmax; }
    tot += t1 * t2 * std::exp(logt3);
  }
  return tot;
}

/// G4PreCompoundEmission::AngularDistribution - the fUseAngularGen branch of PerformEmission.
///
/// Kalbach's systematics: the emission is forward-peaked with an exponential in cos(theta)
/// about the INCIDENT direction, `an` setting the slope. Four things in it are worth naming:
///
///   * `ProjEnergy = aFragment.GetExcitationEnergy()`. The comment above it admits this is not
///     the projectile energy - "If I would know which is the projectile ... I could remove the
///     binding energy" - and uses the excitation instead. That is a documented substitution
///     inside Geant4, not a transcription liberty here.
///   * `Eav` starts as the average exciton energy `2p(p+1)/((p+h) gg)` and is then multiplied
///     by rho(p+1,h)/rho(p,h) and shifted; if either rho is non-positive it is replaced
///     wholesale by the Fermi energy.
///   * `an` is divided by `ne = GetNumberOfExcitons() - 1` only when that is above 1, and
///     clamped at 10.
///   * the direction is `theIncidentDirection = aFragment.GetMomentum().vect().unit()`, so a
///     fragment AT REST has no incident direction. CLHEP's `unit()` returns the zero vector
///     there and `rotateUz` about a zero axis leaves the momentum along +z - which is why the
///     port uses P3's `clhep_unit` semantics and not `normalize`'s (0,0,1) default. Every
///     validated fragment in this package's grid is at rest, so this is the branch that would
///     be taken if the flag were ever set.
///
/// Returns the momentum three-vector, already rotated onto the fragment's own direction, so
/// that it is a drop-in replacement for the isotropic branch.
///
/// Consumes exactly two uniform deviates (cos theta, then phi), where the isotropic branch
/// consumes 2 on average 4/pi of the time - so the two branches cannot be compared draw for
/// draw against each other, only against Geant4 with the same branch selected.
template <typename Rng>
__host__ __device__ inline Vec3d angular_momentum(const PreFragState& st, const Fragment& frag,
                                                  const Excitons& ex, double ekin,
                                                  const data::LevelTable& lt, Rng& rng) {
  const int p = ex.particles;
  const int h = ex.holes;
  const double U = frag.excitation;
  const double fermi_energy = deex::deex_params().fermi_energy;
  const double bemission = st.binding_energy;

  const bool has_levels = (data::find_manager(lt, frag.z, frag.a) >= 0);
  const double gg = (6.0 / deex::pi2()) * deex::level_density(frag.z, frag.a, U, has_levels);

  // Average exciton energy relative to the bottom of the nuclear well.
  double eav = 2.0 * p * (p + 1) / ((p + h) * gg);

  // Excitation relative to the Fermi level. The commented-out alternative in the Geant4
  // source, `U - KineticEnergyOfEmittedFragment - Bemission`, is a different quantity and is
  // not what runs.
  double uf = U - (p - h) * fermi_energy;
  if (uf < 0.0) { uf = 0.0; }

  const double w_num = preco_rho(p + 1, h, gg, uf, fermi_energy);
  const double w_den = preco_rho(p, h, gg, uf, fermi_energy);
  if (w_num > 0.0 && w_den > 0.0) {
    eav *= (w_num / w_den);
    eav += -uf / (p + h) + fermi_energy;
  } else {
    eav = fermi_energy;
  }

  double an = 0.0;
  const double eeff = ekin + bemission + fermi_energy;
  // Geant4's guard is `> DBL_MIN`, i.e. 2.2e-308 - a positivity test written as a
  // denormal threshold, and in double it behaves as one. Kept as written.
  const double dbl_min = 2.2250738585072014e-308;
  if (ekin > dbl_min && eeff > dbl_min) {
    double zeta = 9.3 / std::sqrt(ekin / u::MeV<double>());
    if (zeta < 1.0) { zeta = 1.0; }
    // Not the projectile energy: see the fourth bullet in this function's header comment.
    const double proj_energy = frag.excitation;
    an = 3.0 * std::sqrt((proj_energy + fermi_energy) * eeff) / (zeta * eav);
    const int ne = ex.total() - 1;
    if (ne > 1) { an /= static_cast<double>(ne); }
    if (an > 10.0) { an = 10.0; }
  }

  const double random = rng.uniform();
  double cost;
  if (an < 0.1) {
    cost = 1.0 - 2.0 * random;
  } else {
    const double exp2an = std::exp(-2.0 * an);
    cost = 1.0 + std::log(1.0 - random * (1.0 - exp2an)) / an;
    if (cost > 1.0) { cost = 1.0; }
    else if (cost < -1.0) { cost = -1.0; }
  }

  const double phi = u::twopi<double>() * rng.uniform();
  const double pmag = std::sqrt(ekin * (ekin + 2.0 * st.mass));
  const double sint = std::sqrt((1.0 - cost) * (1.0 + cost));
  const Vec3d local{pmag * std::cos(phi) * sint, pmag * std::sin(phi) * sint, pmag * cost};

  // CLHEP's unit(), not normalize()'s: a fragment at rest has NO incident direction and must
  // keep the local frame, rather than being given +z as an axis. rotate_uz about a zero
  // vector is the identity, which is what CLHEP's rotateUz does with a zero axis too.
  const Vec3d incident = deex::clhep_unit(frag.momentum.v);
  return g4gpu::rotate_uz(local, incident);
}

// ---------------------------------------------------------------------------------------------
// G4PreCompoundEmission::PerformEmission
// ---------------------------------------------------------------------------------------------

/// One emitted pre-equilibrium particle: what G4ReactionProduct carries out of PerformEmission.
///
/// (Z, A) and a four-momentum, with `pdg` filled for all six because all six have a fixed
/// particle definition - so unlike P3's DeexProduct there is no ion-table boundary here.
struct PrecoProduct {
  LorentzVector momentum;
  int z = 0;
  int a = 0;
  int pdg = 0;
  double mass = 0.0;   ///< the PDG mass the emitter used, so kinetic energy is unambiguous

  __host__ __device__ double kinetic_energy() const {
    const double k = momentum.e - mass;
    return (k > 0.0) ? k : 0.0;
  }
};

/// PDG codes of the six pre-equilibrium ejectiles, in the factory order.
__host__ __device__ inline int pre_frag_pdg(int kind) {
  switch (kind) {
    case kPreNeutron:  return 2112;
    case kPreProton:   return 2212;
    case kPreDeuteron: return 1000010020;
    case kPreAlpha:    return 1000020040;
    case kPreTriton:   return 1000010030;
    case kPreHe3:      return 1000020030;
    default:           return 0;
  }
}

/// G4PreCompoundEmission::PerformEmission.
///
/// The fragment is updated IN PLACE, as Geant4 does: (Z, A) to the residual's, the particle
/// and charge exciton counts reduced by the ejectile's own A and Z, and the momentum to the
/// remainder. `SetMomentum` recomputes the ground-state mass and the excitation from the new
/// (Z, A) - that is the identity P3's `set_za_and_momentum` carries - so the residual's E*
/// falls out of the subtraction and is never computed independently.
///
/// **The exciton subtraction cannot go negative**, and the reason is IsItPossible: it required
/// `pneut >= A - Z` and `pplus >= Z`, so `(P - A) - (Pc - Z) = (P - Pc) - (A - Z) >= 0`. That
/// is what keeps `SetNumberOfCharged` from throwing on the line after `SetNumberOfParticles`,
/// and it is why the order of those two calls matters (see precompound_transitions.cuh).
///
/// The emission is isotropic in the fragment's REST frame and then boosted by the fragment's
/// own velocity; `Rest4Momentum.boostVector()` is p/E of the parent, so a parent at rest
/// leaves the ejectile isotropic in the lab.
template <typename Rng>
__host__ __device__ inline PrecoProduct perform_emission(int kind, EmissionChannels& ch,
                                                         Fragment& frag, Excitons& ex,
                                                         const data::LevelTable& lt, Rng& rng,
                                                         PrecoRefusal& ref) {
  PreFragState& st = ch.st[kind];

  int tries = 0;
  double kin_energy = pre_frag_sample_kinetic_energy(st, frag, ex, rng, &tries);
  if (kin_energy < 0.0) { kin_energy = 0.0; }
  if (tries >= 100) {
    ref.sampler_exhausted = true;
    ref.refused_z = frag.z;
    ref.refused_a = frag.a;
  }

  Vec3d pvec;
  if (deex::deex_params().use_angular_gen) {
    pvec = angular_momentum(st, frag, ex, kin_energy, lt, rng);
  } else {
    const double pmag = std::sqrt(kin_energy * (kin_energy + 2.0 * st.mass));
    const Vec3d dir = deex::random_direction(rng);
    pvec = pmag * dir;
  }

  const double emitted_mass = st.mass;
  LorentzVector emitted(pvec, emitted_mass + kin_energy);

  LorentzVector rest = frag.momentum;
  emitted.boost(rest.boost_vector());
  rest -= emitted;

  // Update the residual: Z, A, exciton counts, then the momentum (which recomputes E*).
  const int new_particles = ex.particles - st.a;
  const int new_charged = ex.charged - st.z;
  ex.particles = new_particles;
  ex.charged = new_charged;
  if (new_charged > new_particles || new_particles < 0 || new_charged < 0) {
    ref.exciton_count = true;
    ref.refused_z = frag.z;
    ref.refused_a = frag.a;
  }
  frag.set_za_and_momentum(rest, st.res_z, st.res_a);

  PrecoProduct out;
  out.momentum = emitted;
  out.z = st.z;
  out.a = st.a;
  out.pdg = pre_frag_pdg(kind);
  out.mass = emitted_mass;
  return out;
}

}  // namespace g4gpu::preco

#endif
