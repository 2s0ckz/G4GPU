// G4PreCompoundModel: the exciton-model stage between a cascade and the de-excitation chain.
//
// Transcribed from G4PreCompoundModel (+ .hh's inline PerformEquilibriumEmission) in 11.1.1,
// pre_equilibrium/exciton_model.
//
// One excited fragment plus an exciton configuration in; a list of pre-equilibrium ejectiles
// followed by P3's whole equilibrium cascade out. The loop is a competition between emitting
// a particle and rearranging the excitons, and it ends the first time the fragment fails any
// of six equilibrium tests - at which point the residual is handed to
// G4ExcitationHandler::BreakItUp, which is `deex::deexcite`.
//
// **The gate at the top and the gate in the loop are not the same test.** Both are written out
// below; the differences are:
//
//   entry   `U < fLowLimitExc*A`     strict
//   loop    `U <= fLowLimitExc*A`    inclusive
//   entry   `(Z < minZ && A < minA)` AND - a fragment needs BOTH a small Z and a small A to
//                                    skip pre-equilibrium
//   loop    `Z < minZ || A < minA`   OR  - either one ends it
//
// So Li6 (Z = 3, A = 6) enters pre-equilibrium and then leaves it on the first iteration
// because `Z < 3` is false but `A < 5` is false too... no: Z = 3 is not < 3 and A = 6 is not
// < 5, so it stays. He4 (Z = 2, A = 4) fails the entry gate (2 < 3 AND 4 < 5) and never
// enters. But Be7 (Z = 4, A = 7) enters by the AND and stays by the OR, while H3 (Z = 1,
// A = 3) is stopped at the entry. The pair of fragments the two forms disagree about is
// (Z < 3, A >= 5) or (Z >= 3, A < 5): a Li4 (Z = 3, A = 4) passes the entry gate - 3 is not
// < 3 - and is then sent to equilibrium by the loop's `A < minA`. That fragment therefore
// costs one full CalculateProbability call and one uniform deviate before going to the
// handler, and reproducing the difference is what makes the random stream line up.
//
// **The equilibrium exciton number.**
//     n_eq = G4lrint(sqrt((12/pi^2) U g(Z,A,U)))
// with g = A*0.075/MeV under fLD, so n_eq = lrint(sqrt(1.21585 * 0.075 * A * U/MeV)). It can
// be ZERO - Al27 at 0.5 MeV gives sqrt(1.23) = 1.1 -> 1, and a lighter or colder fragment
// gives 0 - and then `ne <= n_eq` is false for any ne >= 1 and the fragment goes to
// equilibrium on the first pass. That is not a degenerate case to guard; it is how the low
// end of the pre-equilibrium window closes.
//
// **The soft cutoff divides by n_eq** (`x = (ne - n_eq)/n_eq`), and is guarded by `go_ahead`,
// which is `ne <= n_eq` - so it is only evaluated when n_eq >= ne >= 0. n_eq = ne = 0 would
// be 0/0, and the following test `GetNumberOfExcitons() <= 0` catches that fragment anyway.
// fUseSoftCutoff is false in 11.1.1 and cannot be changed after initialisation, so this whole
// branch is unreachable from the oracle; it is transcribed and its unreachability recorded.
//
// **The 1000-iteration guard is a JustWarning, not a FatalException.** Geant4 prints and then
// performs equilibrium emission on whatever the fragment has become. So it is a REPORT here
// and not a refusal that abandons the event - `PrecoRefusal::loop_limit` is set and the
// products are still produced. That distinction matters: the de-excitation module's
// equivalent guard (deex::DeexStatus::refused_loop_limit) sits where Geant4 aborts.
#ifndef G4GPU_PRECO_MODEL_CUH
#define G4GPU_PRECO_MODEL_CUH

#include <cmath>

#include "core/units.cuh"
#include "data/level_data.cuh"
#include "physics/hadronic/deexcitation/corrections.cuh"
#include "physics/hadronic/deexcitation/deex_params.cuh"
#include "physics/hadronic/deexcitation/excitation_handler.cuh"
#include "physics/hadronic/deexcitation/fermi_breakup.cuh"
#include "physics/hadronic/deexcitation/fragment.cuh"
#include "physics/hadronic/precompound/precompound_emission.cuh"
#include "physics/hadronic/precompound/precompound_transitions.cuh"

namespace g4gpu::preco {

namespace u = g4gpu::units;

/// What one DeExcite call did, and every refusal it hit.
struct PrecoStatus {
  int n_preco_products = 0;  ///< the pre-equilibrium ejectiles, before the handler's products
  int n_products = 0;        ///< the total written to the caller's product buffer
  int n_iterations = 0;      ///< G4PreCompoundModel's `count` - transitions plus emissions
  int n_transitions = 0;
  int n_emissions = 0;
  bool reached_equilibrium_without_emitting = false;
  /// Set when the entry gate at the top of DeExcite sent the fragment straight to the handler,
  /// so no pre-equilibrium stage ran at all.
  bool skipped_precompound = false;
  PrecoRefusal ref;
  deex::DeexStatus deex;    ///< what the equilibrium hand-over reported
};

/// Caller-owned buffers. The product list is shared with P3: one `DeexProduct` array holds
/// both the pre-equilibrium ejectiles and the equilibrium cascade's output, in emission order,
/// so a caller sees ONE list and does not have to know where the boundary was. `n_preco_products`
/// in PrecoStatus is where it was, for a test that wants to separate them.
struct PrecoWorkspace {
  deex::DeexWorkspace deex;          ///< the handler's three scratch buffers and its products
  deex::DeexProduct* products = nullptr;
  int products_capacity = 0;
};

/// A pre-equilibrium ejectile in P3's product form, so one list holds both stages.
__host__ __device__ inline deex::DeexProduct to_deex_product(const PrecoProduct& p) {
  deex::DeexProduct d;
  d.momentum = p.momentum;
  d.z = p.z;
  d.a = p.a;
  d.pdg = p.pdg;
  d.excitation = 0.0;      // all six ejectiles are emitted in their ground state
  d.floating_level = 0;
  d.long_lived = false;
  return d;
}

// ---------------------------------------------------------------------------------------------
// The three decisions DeExcite makes about (Z, A, U), as named functions
// ---------------------------------------------------------------------------------------------
//
// These were expressions inline in `deexcite` below, and an expression has no address a test can
// aim at: the test compared its OWN copy of each formula against the oracle column, agreed with
// itself to the last bit over 1,920 points, and went on agreeing when the port's copy of all
// three was changed underneath it. docs/RISK.md V52 has the measurement. They are functions now
// so that `tests/test_precompound.cu` compares the port and not a transcription of the port, and
// `deexcite` is their only other caller.

/// The entry gate at the top of G4PreCompoundModel::DeExcite, as a predicate on (Z, A, U): true
/// when pre-equilibrium is SKIPPED and the fragment goes straight to the handler. The `!isActive`
/// and `GetNumberOfLambdas() > 0` clauses stay with the caller, being functions of neither.
///
/// Note the AND in the (Z, A) clause and the STRICT `<` on the low limit. The loop's version of
/// both is different - see `preco_loop_gate` and this file's header.
__host__ __device__ inline bool preco_entry_gate(int Z, int A, double U) {
  const deex::DeexParameters& par = deex::deex_params();
  return (Z < par.min_z_for_preco && A < par.min_a_for_preco) ||
         U < par.preco_low_energy * A || U > A * par.preco_high_energy;
}

/// The (Z, A) and excitation clauses of the six-way equilibrium test inside DeExcite's loop:
/// true when the loop hands the fragment to the equilibrium handler. The other three clauses -
/// `!go_ahead`, `P1 <= P2+P3` and `GetNumberOfExcitons() <= 0` - stay in the loop, being
/// functions of the transition probabilities and the exciton configuration rather than of
/// (Z, A, U). None of the six has a side effect, so factoring four of them out cannot change
/// what the short circuit evaluates.
///
/// OR where the entry gate has AND, and `<=` where the entry gate has `<`.
__host__ __device__ inline bool preco_loop_gate(int Z, int A, double U) {
  const deex::DeexParameters& par = deex::deex_params();
  return Z < par.min_z_for_preco || A < par.min_a_for_preco ||
         U <= par.preco_low_energy * A || U > A * par.preco_high_energy;
}

/// n_eq = G4lrint(sqrt((12/pi^2) U g(Z, A, U))), the critical exciton number DeExcite computes
/// at the top of each outer iteration and compares `GetNumberOfExcitons()` against.
///
/// `has_levels` is P3's level-density dispatch input, `data::find_manager(lt, Z, A) >= 0`; it is
/// a parameter rather than a lookup here so the function needs no level table and the test can
/// drive both branches. It can return ZERO - see the file header.
__host__ __device__ inline int preco_equilibrium_exciton_number(int Z, int A, double U,
                                                                 bool has_levels) {
  const double ldfact = 12.0 / deex::pi2();
  return static_cast<int>(
      std::lrint(std::sqrt(ldfact * U * deex::level_density(Z, A, U, has_levels))));
}

/// G4PreCompoundModel::PerformEquilibriumEmission - `GetExcitationHandler()->BreakItUp()` and
/// splice the result onto the end.
///
/// The handler writes into its own product buffer, so the products are copied across into the
/// caller's single list here. That copy is the whole cost of presenting one list.
template <typename Rng>
__host__ __device__ inline void perform_equilibrium_emission(const deex::Fragment& frag,
                                                             const data::LevelTable& lt,
                                                             const deex::FermiPool& pool,
                                                             const PrecoWorkspace& ws,
                                                             PrecoStatus& st, Rng& rng) {
  st.deex = deex::deexcite(frag, lt, pool, ws.deex, rng);
  for (int i = 0; i < st.deex.n_products; ++i) {
    if (st.n_products >= ws.products_capacity) {
      st.ref.capacity = true;
      st.ref.refused_z = frag.z;
      st.ref.refused_a = frag.a;
      return;
    }
    ws.products[st.n_products++] = ws.deex.products[i];
  }
}

/// G4PreCompoundModel::DeExcite.
///
/// `frag` and `ex` are taken by value: Geant4 mutates the caller's G4Fragment in place through
/// the whole loop and the caller does not use it again, and a copy is what makes this callable
/// on a const input from a device kernel.
///
/// `transition_model` selects G4PreCompoundTransitions (kCEM or kGupta, by fUseCEM) or
/// G4GNASHTransitions (kGNASH, by fUseGNASH). The default is kCEM. Passing kGNASH sets
/// `ref.gnash` and the loop then exits to equilibrium on its first test, which is exactly what
/// Geant4 does and is why the refusal is a report rather than an error.
template <typename Rng>
__host__ __device__ inline PrecoStatus deexcite(deex::Fragment frag, Excitons ex,
                                                const data::LevelTable& lt,
                                                const deex::FermiPool& pool,
                                                const PrecoWorkspace& ws, Rng& rng,
                                                int transition_model = kCEM,
                                                int optxs = -1) {
  PrecoStatus st;
  const deex::DeexParameters& par = deex::deex_params();
  if (optxs < 0) { optxs = par.preco_type; }
  if (transition_model == kGNASH) { st.ref.gnash = true; }
  if (par.use_hetc) {
    // G4PreCompoundEmission::SetHETCModel would have replaced all six channels. Refused by
    // name: see precompound_emission.cuh's header.
    st.ref.hetc = true;
    st.ref.refused_z = frag.z;
    st.ref.refused_a = frag.a;
    return st;
  }

  const bool is_active = !par.preco_dummy;

  double U = frag.excitation;
  int Z = frag.z;
  int A = frag.a;

  if (frag.lambdas > 0) { st.ref.hyper_fragment = true; }

  // The entry gate; `preco_entry_gate` is where the AND and the strict `<` are written out.
  if (!is_active || preco_entry_gate(Z, A, U) || frag.lambdas > 0) {
    st.skipped_precompound = true;
    st.reached_equilibrium_without_emitting = true;
    perform_equilibrium_emission(frag, lt, pool, ws, st, rng);
    return st;
  }

  int count = 0;
  const int countmax = 1000;

  for (;;) {
    U = frag.excitation;
    Z = frag.z;
    A = frag.a;
    const bool has_levels = (data::find_manager(lt, Z, A) >= 0);
    const int eq_exciton_number = preco_equilibrium_exciton_number(Z, A, U, has_levels);

    bool is_transition = false;
    do {
      ++count;
      const int ne = ex.total();
      bool go_ahead = (ne <= eq_exciton_number);

      if (par.use_soft_cutoff && go_ahead) {
        // Unreachable with 11.1.1's defaults; see the file header.
        const double x = static_cast<double>(ne - eq_exciton_number) /
                         static_cast<double>(eq_exciton_number);
        if (rng.uniform() < 1.0 - std::exp(-x * x / 0.32)) { go_ahead = false; }
      }

      const TransitionProbs tp =
          transition_probability(transition_model, frag, ex, par.never_go_back, lt, rng);

      // The six-way equilibrium test. `P1 <= P2+P3` is the physical criterion - Quesada's
      // comment says it PREVAILS over the critical-exciton-number approximation - and it is
      // also what makes fUseGNASH a dead branch, because GNASH leaves all three at zero.
      if (!go_ahead || tp.p1 <= tp.p2 + tp.p3 || preco_loop_gate(Z, A, U) ||
          ex.total() <= 0) {
        if (st.n_emissions == 0) { st.reached_equilibrium_without_emitting = true; }
        st.n_iterations = count;
        perform_equilibrium_emission(frag, lt, pool, ws, st, rng);
        return st;
      }

      EmissionChannels ch = emission_probabilities(frag, ex, lt, optxs);
      const double emission_probability = ch.total;
      const double total_probability = emission_probability + tp.total;

      // Select the subprocess. Geant4's inequality is `> emissionProbability`, so the
      // emission branch is the `<=` one - which is why a zero emission probability still
      // reaches PerformEmission if the deviate comes out exactly 0.
      if (total_probability * rng.uniform() > emission_probability) {
        is_transition = true;
        ++st.n_transitions;
        const TransitionResult tr = perform_transition(tp, frag, ex, rng);
        ex = tr.ex;
        if (tr.refused_exciton_count) {
          st.ref.exciton_count = true;
          st.ref.refused_z = frag.z;
          st.ref.refused_a = frag.a;
        }
      } else {
        is_transition = false;
        ++st.n_emissions;
        const int kind = choose_fragment(ch, rng, st.ref);
        const PrecoProduct p = perform_emission(kind, ch, frag, ex, lt, rng, st.ref);
        if (st.n_products >= ws.products_capacity) {
          st.ref.capacity = true;
          st.ref.refused_z = frag.z;
          st.ref.refused_a = frag.a;
          st.n_iterations = count;
          return st;
        }
        ws.products[st.n_products++] = to_deex_product(p);
        ++st.n_preco_products;
      }
    } while (is_transition);

    if (count >= countmax) {
      // JustWarning in Geant4, then equilibrium emission on the current fragment.
      st.ref.loop_limit = true;
      st.ref.refused_z = frag.z;
      st.ref.refused_a = frag.a;
      st.n_iterations = count;
      perform_equilibrium_emission(frag, lt, pool, ws, st, rng);
      return st;
    }
  }
}

// ---------------------------------------------------------------------------------------------
// G4PreCompoundModel::ApplyYourself
// ---------------------------------------------------------------------------------------------

/// The initial fragment G4PreCompoundModel::ApplyYourself builds for a nucleon on a nucleus.
///
/// (A + 1, Z + Zp) with the projectile's four-momentum added to the target at rest, and
/// **two particle excitons of which exactly one is charged, plus one hole** -
/// `SetNumberOfExcitedParticle(2, 1)` and `SetNumberOfHoles(1, 0)`.
///
/// The `1` is unconditional. A NEUTRON projectile on a nucleus gets one charged particle
/// exciton in Geant4, the same as a proton, even though the two nucleons in the initial state
/// are the projectile neutron and a knocked-out nucleon. That is not a reading of the source
/// this port is free to improve: GetRj for the proton channel is Pc/P = 1/2 for both
/// projectiles, so a neutron-induced reaction emits protons at the same relative rate as a
/// proton-induced one at the first step. Recorded in docs/RISK.md V51, with the measurement.
///
/// `Zp` is 1 for a proton and 0 for a neutron; `Ap` is 1 for both. Any other projectile is a
/// FatalException in Geant4 ("G4PreCompoundModel is used for <name>") and is refused here.
__host__ __device__ inline deex::Fragment apply_yourself_initial_fragment(int projectile_pdg,
                                                                          double kin_energy,
                                                                          const Vec3d& dir,
                                                                          int targetZ,
                                                                          int targetA,
                                                                          Excitons& ex,
                                                                          bool& refused) {
  refused = false;
  int zp = 0;
  double mp = 0.0;
  if (projectile_pdg == 2212) {
    zp = 1;
    mp = deex::pdg_mass_proton();
  } else if (projectile_pdg == 2112) {
    zp = 0;
    mp = deex::pdg_mass_neutron();
  } else {
    refused = true;
    return deex::Fragment{};
  }
  const int ap = 1;

  const double etot = mp + kin_energy;
  const double pmag = std::sqrt(kin_energy * (kin_energy + 2.0 * mp));
  const double target_mass = deex::nuclear_mass(targetA, targetZ);
  const deex::LorentzVector p(pmag * dir.x, pmag * dir.y, pmag * dir.z, etot + target_mass);

  ex.particles = 2;
  ex.charged = 1;   // unconditional in Geant4 - see this function's header
  ex.holes = 1;

  return deex::make_fragment(targetA + ap, targetZ + zp, p);
}

}  // namespace g4gpu::preco

#endif
