// The exciton-number transitions: Delta n = +2, 0 and -2, and the change they make.
//
// Transcribed from G4VPreCompoundTransitions, G4PreCompoundTransitions and
// G4GNASHTransitions (11.1.1, pre_equilibrium/exciton_model).
//
// A pre-equilibrium state either emits a particle or rearranges its excitons, and this file is
// the second half. `CalculateProbability` returns the SUM of three rates and leaves the three
// separately in members that the model then reads back - and the order matters: Geant4's own
// comment says "WARNING: CalculateProbability MUST be called prior to Get!! (0 values would be
// returned otherwise)". Here they are returned in one struct, so the ordering hazard cannot
// exist; the warning is recorded because it is what the shape of the original is protecting
// against.
//
// **Two flags, four configurations, and only one of them is QBBC's.** fUseCEM is true and
// fNeverGoBack is false in 11.1.1, so the default is Gudima's CEM rates with all three
// transitions live. The other three combinations are reachable - `UseCEMtr` and `UseNGB` are
// public inline setters on G4VPreCompoundTransitions with no state lock on them, unlike
// G4DeexPrecoParameters' setters - so all four are transcribed and all four are dumped.
//
//   useCEMtr = true            Gudima/Mashnik/Toneev: a nucleon-nucleon cross section at the
//                              relative energy, a Pauli factor, and an interaction volume.
//   useCEMtr = false           Gupta's two-term polynomial in U, and NO Delta n = 0 rate at
//                              all - TransitionProb3 stays zero, so the exciton number never
//                              stays put.
//   useNGB = true              TransitionProb2 and TransitionProb3 are left at zero in both
//                              branches, i.e. "never go back": Delta n = +2 only. In the
//                              model's loop that immediately satisfies `P1 <= P2+P3` being
//                              FALSE, so it does not stop pre-equilibrium; it removes the
//                              downward and sideways moves.
//
// **G4GNASHTransitions never sets TransitionProb1/2/3.** It computes a probability, returns
// it, and leaves the three base-class members at the 0.0 the base constructor gave them. The
// model reads P1, P2, P3 straight afterwards and its first equilibrium test is
// `P1 <= P2+P3` - which is `0 <= 0`, true - so **fUseGNASH = true sends every fragment to
// equilibrium emission on the first iteration and no pre-equilibrium emission ever happens.**
// That is transcribed rather than corrected, and the oracle dumps the three getters after a
// GNASH CalculateProbability call so the fact is measured and not argued. See docs/RISK.md.
#ifndef G4GPU_PRECO_TRANSITIONS_CUH
#define G4GPU_PRECO_TRANSITIONS_CUH

#include <cmath>

#include "core/units.cuh"
#include "data/g4pow.hh"
#include "data/level_data.cuh"
#include "physics/hadronic/deexcitation/corrections.cuh"
#include "physics/hadronic/deexcitation/deex_params.cuh"
#include "physics/hadronic/deexcitation/fragment.cuh"
#include "physics/hadronic/precompound/precompound_fragment.cuh"

namespace g4gpu::preco {

namespace u = g4gpu::units;

/// CLHEP's h_Planck in MeV ns, derived the way CLHEP derives the chain rather than pasted:
/// CLHEP pins h_Planck in SI, sets hbar_Planck = h_Planck/twopi and hbarc = hbar_Planck *
/// c_light, and src/core/units.cuh pins hbarc and c_light, so h_Planck = twopi hbarc/c_light
/// inverts that. One multiply and one divide of pinned doubles, i.e. agreement to the last
/// couple of bits, and docs/RISK.md V8 is what a second hand-typed constant cost. Used only by
/// G4GNASHTransitions, which the file header shows is a dead branch.
__host__ __device__ inline constexpr double h_planck() {
  return u::twopi<double>() * u::hbarc<double>() / u::c_light<double>();
}

/// std::log and std::exp stand in for G4Log and G4Exp throughout, as everywhere else in this
/// port: they agree to within an ulp (see the note in src/data/g4pow.hh). G4Pow's TABULATED
/// logX/powA are a different matter and are not substituted anywhere.
///
/// The three rates and their sum, i.e. what CalculateProbability returns plus what
/// GetTransitionProb1/2/3 would return afterwards.
struct TransitionProbs {
  double p1 = 0.0;   ///< TransitionProb1, Delta n = +2
  double p2 = 0.0;   ///< TransitionProb2, Delta n = -2
  double p3 = 0.0;   ///< TransitionProb3, Delta n =  0
  double total = 0.0;

  __host__ __device__ double sum() const { return p1 + p2 + p3; }
};

/// Which transition model is in use. The default is `kCEM`; `kGupta` is the same class with
/// useCEMtr false, and `kGNASH` is a different class entirely.
enum TransitionModel { kCEM = 0, kGupta = 1, kGNASH = 2 };

/// G4PreCompoundTransitions::CalculateProbability, both branches.
///
/// Returns 0.0 for U < 10 eV or N == 0 without touching p1/p2/p3 - so a caller that reuses a
/// TransitionProbs from a previous fragment would read stale rates. Geant4 has exactly that
/// hazard (the members persist); the port returns a fresh struct, so it cannot.
///
/// **The CEM branch consumes one random number**, and it is not a rejection: the projectile
/// nucleon's charge is sampled as `G4int(P*rand()) <= GetNumberOfCharged()`. Two things about
/// that line are worth writing down because they are not what "with probability Z/A" would
/// give:
///   * `G4int(P*rand())` is uniform on 0..P-1, so the test is `U(0..P-1) <= Pc`, which is
///     (Pc+1)/P and not Pc/P. With Pc = 0 the nucleon is still charged one time in P.
///   * it is the number of CHARGED PARTICLE EXCITONS, not the nucleus's Z/A, that decides.
/// The consequence is only which nucleon mass divides the relative energy, so the effect is
/// small - but the random draw is consumed either way, which is what makes a stream-for-stream
/// comparison against Geant4 impossible for a different engine and is why this package's
/// campaign checks are statistical.
///
/// `xx = 2 r0 + hbarc/(m_p v_rel)` is a de Broglie-broadened interaction radius, `Vint` its
/// sphere volume written as `pi xx^3 / 0.75` (that is 4/3 pi xx^3), and TransitionProb1 is
/// sigma * PauliFactor * v_rel / Vint - a collision rate per unit time.
///
/// `GE = 6/pi^2 * U * g(Z,A,U)` is the exciton-model g*E, and `Fph` the Pauli energy. The
/// `GE > Fph1` guard is J.M. Quesada and A. Howard's fix; the comment beside it records that
/// the original test was `U - Fph < 0`, which is a different quantity.
template <typename Rng>
__host__ __device__ inline TransitionProbs
transition_probability_cem(const deex::Fragment& frag, const Excitons& ex, bool use_ngb,
                           const data::LevelTable& lt, Rng& rng) {
  TransitionProbs tp;
  const int H = ex.holes;
  const int P = ex.particles;
  const int N = P + H;
  const int A = frag.a;
  const int Z = frag.z;
  const double U = frag.excitation;

  if (U < 10.0 * u::eV<double>() || 0 == N) { return tp; }

  const double fermi_energy = deex::deex_params().fermi_energy;
  const double r0 = deex::deex_params().transitions_r0;

  const bool has_levels = (data::find_manager(lt, Z, A) >= 0);
  const double sixdpi2 = 6.0 / deex::pi2();
  const double GE = sixdpi2 * U * deex::level_density(Z, A, U, has_levels);

  // T_rel: 1.6 Fermi energies plus the excitation shared over the excitons.
  const double rel_energy = 1.6 * fermi_energy + U / static_cast<double>(N);

  bool charged_nucleon = false;
  if (static_cast<int>(P * rng.uniform()) <= ex.charged) { charged_nucleon = true; }

  const double m_nucleon = charged_nucleon ? u::proton_mass_c2<double>()
                                           : u::neutron_mass_c2<double>();
  const double rel_v_sqr = 2.0 * rel_energy / m_nucleon;
  const double rel_v = std::sqrt(rel_v_sqr);

  // The two free-nucleon cross sections, in millibarn, as functions of v_rel in units of c.
  const double pp_xs = (10.63 / rel_v_sqr - 29.92 / rel_v + 42.9) * deex::millibarn();
  const double np_xs = (34.10 / rel_v_sqr - 82.20 / rel_v + 82.2) * deex::millibarn();

  // The isospin average over the target's other A-1 nucleons. JMQ's "small bug fixed" is the
  // (Z-1) and (A-Z-1): the projectile nucleon is not its own scattering partner.
  double avg_xs;
  if (charged_nucleon) {
    avg_xs = ((Z - 1) * pp_xs + (A - Z) * np_xs) / static_cast<double>(A - 1);
  } else {
    avg_xs = ((A - Z - 1) * pp_xs + Z * np_xs) / static_cast<double>(A - 1);
  }

  // The Pauli suppression: linear in E_F/T_rel, with a 3/2-power correction above 0.5.
  const double fermi_rel_ratio = fermi_energy / rel_energy;
  double pauli = 1.0 - 1.4 * fermi_rel_ratio;
  if (fermi_rel_ratio > 0.5) {
    const double x = 2.0 - 1.0 / fermi_rel_ratio;
    pauli += 0.4 * fermi_rel_ratio * x * x * std::sqrt(x);
  }

  const double xx = 2.0 * r0 + u::hbarc<double>() / (u::proton_mass_c2<double>() * rel_v);
  const double vint = u::pi<double>() * xx * xx * xx / 0.75;

  // Note the sqrt is recomputed from the PROTON mass whatever the sampled nucleon was, so the
  // velocity in the numerator and the one in `rel_v` above differ for a neutron.
  const double p1 = avg_xs * pauli *
                    std::sqrt(2.0 * rel_energy / u::proton_mass_c2<double>()) / vint;
  tp.p1 = (p1 > 0.0) ? p1 : 0.0;

  const double Fph = static_cast<double>(P * P + H * H + P - 3 * H) * 0.25;

  if (!use_ngb) {
    const double Fph1 = Fph + N * 0.5;      // F(p+1, h+1)
    const double plimit = 100.0;
    if (GE > Fph1) {
      const double x0 = GE - Fph;
      double x1 = (N + 1) * std::log(x0 / (GE - Fph1));
      if (x1 < plimit) {
        x1 = std::exp(x1) * tp.p1 / x0;
        const double p2 = (P * H * (N + 1) * (N - 2)) * x1 / x0;
        tp.p2 = (p2 > 0.0) ? p2 : 0.0;
        const double p3 =
            ((N + 1) * (P * (P - 1) + 4 * P * H + H * (H - 1))) * x1 / static_cast<double>(N);
        tp.p3 = (p3 > 0.0) ? p3 : 0.0;
      }
    }
  }
  tp.total = tp.sum();
  return tp;
}

/// G4PreCompoundTransitions::CalculateProbability with useCEMtr = false - Gupta's form.
///
/// `U*(4.2e12 - 3.6e10*U/(N+1))/(16 c_light)`: a rate in inverse time built from two
/// hand-fitted coefficients with implicit units, divided by 16 times the speed of light. It
/// goes NEGATIVE for U/(N+1) above 116.67 MeV and is then clamped to zero by the max, which
/// means a hot fragment with few excitons has NO upward transition and the model's
/// `P1 <= P2+P3` test sends it to equilibrium.
///
/// TransitionProb3 is never set in this branch: Gupta's parameterisation has no Delta n = 0
/// term, so with useCEMtr false the exciton number always changes. And TransitionProb2 needs
/// N > 1 as well as !useNGB.
///
/// Consumes NO random numbers, unlike the CEM branch.
__host__ __device__ inline TransitionProbs
transition_probability_gupta(const deex::Fragment& frag, const Excitons& ex, bool use_ngb,
                             const data::LevelTable& lt) {
  TransitionProbs tp;
  const int H = ex.holes;
  const int P = ex.particles;
  const int N = P + H;
  const double U = frag.excitation;
  if (U < 10.0 * u::eV<double>() || 0 == N) { return tp; }

  const bool has_levels = (data::find_manager(lt, frag.z, frag.a) >= 0);
  const double sixdpi2 = 6.0 / deex::pi2();
  const double GE = sixdpi2 * U * deex::level_density(frag.z, frag.a, U, has_levels);

  const double p1 = U * (4.2e+12 - 3.6e+10 * U / static_cast<double>(N + 1)) /
                    (16.0 * u::c_light<double>());
  tp.p1 = (p1 > 0.0) ? p1 : 0.0;

  if (!use_ngb && N > 1) {
    tp.p2 = ((N - 1) * (N - 2) * P * H) * tp.p1 / (GE * GE);
  }
  tp.total = tp.sum();
  return tp;
}

/// G4GNASHTransitions::CalculateProbability.
///
/// Returns a single rate and leaves p1 = p2 = p3 = 0, which is the whole point of transcribing
/// it: see the file header. `k = 135 MeV^3` is the squared matrix element's normalisation,
/// `x` a piecewise energy correction with three breakpoints (2, 7 and 15 MeV per exciton), and
/// the h_Planck in the denominator makes the result a rate.
///
/// The Pauli energy here is `((P+1)^2 + (H+1)^2 + (P+1) - 3(H-1))/4` - note `H-1`, where
/// G4PreCompoundTransitions and both PDFs use `-3H`. One of the two is a typo and this port
/// cannot say which; it reproduces what is installed.
__host__ __device__ inline TransitionProbs
transition_probability_gnash(const deex::Fragment& frag, const Excitons& ex,
                             const data::LevelTable& lt) {
  TransitionProbs tp;   // p1 = p2 = p3 = 0.0, and they stay that way - Geant4's too
  const double k = 135.0 * u::MeV<double>() * u::MeV<double>() * u::MeV<double>();
  const double E = frag.excitation;
  const double P = static_cast<double>(ex.particles);
  const double H = static_cast<double>(ex.holes);
  const double N = P + H;
  const int Z = frag.z;
  const int A = frag.a;

  double matrix_element = k * N / ((static_cast<double>(A) * A * A) * E);
  double x = E / (N * u::MeV<double>());
  const double xf = std::sqrt(2.0 / 7.0);
  if (x < 2.0) { x *= xf; }
  else if (x < 7.0) { x *= std::sqrt(x / 7.0); }
  else if (x > 15.0) { x *= std::sqrt(15.0 / x); }
  matrix_element *= x;

  const bool has_levels = (data::find_manager(lt, Z, A) >= 0);
  const double gg = (6.0 / deex::pi2()) * deex::level_density(Z, A, E, has_levels);

  const double epauli = ((P + 1.0) * (P + 1.0) + (H + 1.0) * (H + 1.0) + (P + 1.0) -
                         3.0 * (H - 1.0)) * 0.25;

  double probability = gg * gg * gg * (E - epauli) * (E - epauli);
  probability *= matrix_element / (2.0 * (N + 1.0) * h_planck());

  tp.total = probability;
  return tp;
}

/// The virtual dispatch of CalculateProbability across the three models.
template <typename Rng>
__host__ __device__ inline TransitionProbs
transition_probability(int model, const deex::Fragment& frag, const Excitons& ex, bool use_ngb,
                       const data::LevelTable& lt, Rng& rng) {
  if (model == kGNASH) { return transition_probability_gnash(frag, ex, lt); }
  if (model == kGupta) { return transition_probability_gupta(frag, ex, use_ngb, lt); }
  return transition_probability_cem(frag, ex, use_ngb, lt, rng);
}

/// What PerformTransition did, so a caller can report the case Geant4 throws on.
struct TransitionResult {
  Excitons ex;
  int delta_n = 0;              ///< +1, 0 or -1 - the HALVED Delta n, as Geant4 applies it
  bool refused_exciton_count = false;  ///< G4Fragment would have thrown; see below
};

/// G4PreCompoundTransitions::PerformTransition.
///
/// Selects one of the three by cumulative comparison against `rand()*(P1+P2+P3)`, then halves
/// deltaN (so +2 becomes +1 particle and +1 hole) and applies it. Three details:
///
///   * **the particle and hole counts are written before the charge count**, and Geant4's own
///     comment says why: `SetNumberOfCharged` THROWS if the charge exceeds the particle count,
///     so the particle count has to be raised first. The port keeps the order and reports
///     rather than throwing.
///   * the Delta n = -2 branch removes a charge with probability (Pc+1)/P - the same
///     `G4int(Npart*rand()) <= Ncharged` form as the CEM nucleon choice, off by one from
///     Pc/P - or unconditionally when every particle is charged.
///   * the Delta n = +2 branch adds a charge with weight Z/A computed over the nucleons NOT
///     already excited: `A = GetA_asInt() - Npart`, `Z = GetZ_asInt() - Ncharged`. And it uses
///     `G4lrint` where the other uses `G4int`, so this one ROUNDS and that one TRUNCATES: for
///     A = 10 and Z = 5 the test is `lrint(10 u) <= 5`, which passes for u < 0.55, not 0.5.
///
/// **The final clamp reads the PRE-transition counts.** `if (Npart < Ncharged)` uses the local
/// copies taken before the update, not the values just written - so it fires only when the
/// fragment ARRIVED with more charges than particles, which G4Fragment's own setters would
/// have thrown on. It is therefore dead in any state this model can reach, and it is
/// transcribed as written (with the same locals) rather than "fixed" to read the new values,
/// because fixing it would change the Delta n = +2 result whenever a charge was just added to
/// a saturated configuration.
/// The Delta n = +2 charge decision needs the fragment's Z and A, which `Excitons` does not
/// carry, so this takes both.
template <typename Rng>
__host__ __device__ inline TransitionResult perform_transition(const TransitionProbs& tp,
                                                               const deex::Fragment& frag,
                                                               const Excitons& in, Rng& rng) {
  TransitionResult r;
  r.ex = in;
  const double chosen = rng.uniform() * (tp.p1 + tp.p2 + tp.p3);
  int delta_n = 0;
  const int npart = in.particles;
  const int ncharged = in.charged;
  const int nholes = in.holes;
  if (chosen <= tp.p1) {
    delta_n = 2;
  } else if (chosen <= tp.p1 + tp.p2) {
    delta_n = -2;
  }
  delta_n /= 2;
  r.delta_n = delta_n;

  // SetNumberOfParticles / SetNumberOfHoles first - see the comment above.
  r.ex.particles = npart + delta_n;
  r.ex.holes = nholes + delta_n;

  if (delta_n < 0) {
    if ((ncharged == npart) ||
        (ncharged >= 1 && static_cast<int>(npart * rng.uniform()) <= ncharged)) {
      r.ex.charged = ncharged + delta_n;   // delta_n is negative
    }
  } else if (delta_n > 0) {
    const int A = frag.a - npart;
    const int Z = frag.z - ncharged;
    // G4lrint - round half to even, which is what std::lrint does under the default rounding
    // mode. The exact half is measure-zero on a uniform deviate, so the choice of tie rule is
    // not observable here; std::lrint is used because it is what G4lrint is.
    const long rounded = std::lrint(A * rng.uniform());
    if ((Z == A) || (Z > 0 && rounded <= static_cast<long>(Z))) {
      r.ex.charged = ncharged + delta_n;
    }
  }

  // The clamp, with the PRE-transition locals Geant4 uses.
  if (npart < ncharged) { r.ex.charged = npart; }

  // G4Fragment::SetNumberOfCharged throws when the charge exceeds the particle count, and
  // SetNumberOfHoles throws when the charged-hole count exceeds the hole count. Neither can
  // happen from the arithmetic above - the -2 branch lowers the charge only when it can and
  // the +2 branch raises the particle count first - so this reports rather than guards, and a
  // report means the transcription or the caller's initial state is wrong.
  if (r.ex.charged > r.ex.particles || r.ex.particles < 0 || r.ex.holes < 0) {
    r.refused_exciton_count = true;
  }
  return r;
}

/// G4GNASHTransitions::PerformTransition - a DIFFERENT function from the one above, not a
/// special case of it: it always adds one particle and one hole (there is no Delta n = -2 or 0
/// in GNASH), adds a charge with weight Z/A over the WHOLE nucleus rather than the unexcited
/// part, and its final clamp reads the POST-transition counts where
/// G4PreCompoundTransitions' reads the pre-transition ones. Unreachable in practice, because
/// the model exits to equilibrium before ever calling it (file header), and transcribed so
/// that a run with fUseGNASH set is refused with a number rather than a guess.
template <typename Rng>
__host__ __device__ inline TransitionResult
perform_transition_gnash(const deex::Fragment& frag, const Excitons& in, Rng& rng) {
  TransitionResult r;
  r.ex = in;
  r.delta_n = 1;
  r.ex.particles = in.particles + 1;
  r.ex.holes = in.holes + 1;
  if (rng.uniform() * frag.a <= static_cast<double>(frag.z)) { r.ex.charged = in.charged + 1; }
  if (r.ex.particles < r.ex.charged) { r.ex.charged = r.ex.particles; }
  return r;
}

}  // namespace g4gpu::preco

#endif
