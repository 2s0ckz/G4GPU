// G4DiffractiveExcitation: one nucleon-nucleon collision becomes two excited hadrons, and then
// two strings.
//
// Transcribed from
//   source/processes/hadronic/models/parton_string/diffraction/src/G4DiffractiveExcitation.cc
//     ExciteParticipants, ExciteParticipants_doChargeExchange,
//     ExciteParticipants_doDiffraction, ExciteParticipants_doNonDiffraction,
//     CreateStrings, ChooseP, GaussianPt, GetQuarkFractionOfKink,
//     UnpackMeson, UnpackBaryon, NewNucleonId
//   source/processes/hadronic/util/src/G4SampleResonance.cc
//     SampleMass(const G4ParticleDefinition*, maxMass) and GetMinimumMass
//   source/processes/hadronic/util/src/G4ExcitedString.cc
//     the (Color, AntiColor, Direction) constructor
//
// THE FOUR OUTCOMES OF ONE COLLISION, and they are not four branches of one switch:
//
//   1. QUARK EXCHANGE (`QeExc + QeNoExc`, sampled first). The two hadrons swap a constituent
//      quark and become different particles - a proton can become a neutron and the target a
//      Delta+. This is `_doChargeExchange`, and it can END the collision three ways: with an
//      elastic scattering of the two NEW hadrons (return 0), with a request to continue into
//      the diffraction/non-diffraction sampling (return 1), or with a failure (return 99).
//   2. PROJECTILE DIFFRACTION and 3. TARGET DIFFRACTION, chosen against each other inside
//      `_doDiffraction` by `ProbProjectileDiffraction` after both were normalised by their sum.
//   4. NON-DIFFRACTION, `_doNonDiffraction`, which excites both.
//
// The probabilities come from `G4FTFParameters::GetProcProb(proc, y_proj - y_targ)` for proc =
// 0..4, and they are renormalised THREE TIMES on the way: once against a sum that may exceed 1
// (`QeNoExc = 1 - QeExc - Pd_proj - Pd_targ`), once by `1 - QeExc - QeNoExc`, and once by their
// own sum. Each renormalisation is conditional, so a probability that came out zero stays zero
// rather than becoming 0/0.
//
// `M0projectile` IS THE PDG MASS, NOT THE CURRENT ONE. The two `Uzhi Aug.2019` comments mark
// where `Pprojectile.mag()` was replaced by `GetDefinition()->GetPDGMass()` and where the
// "put it on shell first" block was commented out in favour of an unconditional
// `toBePutOnMassShell = true`. The consequence is load-bearing: the participants arrive OFF
// their mass shells (P9's nucleus guarantees `mag() < PDGMass` for every nucleon), and this
// method puts them back ON the PDG shell in the collision c.m.s. before doing anything else.
// That is the step docs/RISK.md V50 says a caller of `preco::propagate_residual` must perform,
// and G4FTFModel performs it here and only here.
//
// THE KINKY-STRING ARM OF CreateStrings IS REFUSED, not half-built. `Pt2Kink` is set to 0 in
// the G4FTFParameters constructor with the comment "To switch off kinky strings (bad results
// obtained with 6.0*GeV*GeV)", so `Pt = 0` always and `Pt > 500 MeV` is never true; the arm is
// unreachable in 11.1.1. `kKinkyStrings` is set if a future parameter set ever reaches it,
// because a kink makes TWO strings from one hadron and silently making one would lose a parton.
#pragma once
#include <cmath>

#include "core/units.cuh"
#include "data/ftf_hadrons.hh"
#include "data/g4pow.hh"
#include "physics/hadronic/ftf/elastic_hn.cuh"
#include "physics/hadronic/ftf/ftf_parameters.cuh"
#include "physics/hadronic/ftf/lorentz.cuh"
#include "physics/hadronic/ftf/refusal.cuh"
#include "physics/hadronic/ftf/splitable_hadron.cuh"
#include "physics/hadronic/ftf/string_fragmentation.cuh"

namespace g4gpu::hadronic::ftf {

/// G4SampleResonance::GetMinimumMass( p ).
///
/// `DBL_MAX` never leaves this function: the `else` branch of `IsShortLived()` returns the PDG
/// mass, and the short-lived branch always finds at least the most probable channel. The one
/// value Geant4 cannot produce is the one for a short-lived particle with NO decay table - the
/// diquarks - where it dereferences a null pointer; `ftf_hadrons.csv` writes -1 there and this
/// reports `kResonanceMinimumMassMissing` rather than substituting a number.
__host__ __device__ inline double ftf_get_minimum_mass(int pdg, FtfRefusal* refused) {
  const data::FtfHadron* h = data::ftf_find_hadron(pdg);
  if (h == nullptr) {
    if (refused != nullptr) { *refused = FtfRefusal::kUnknownHadronCode; }
    return 0.0;
  }
  if (!h->shortlived) { return h->mass; }
  if (h->minmass < 0.0) {
    if (refused != nullptr) { *refused = FtfRefusal::kResonanceMinimumMassMissing; }
    return 0.0;
  }
  return h->minmass;
}

/// G4SampleResonance::SampleMass( p, maxMass ) = SampleMass(PDGMass, PDGWidth,
/// GetMinimumMass(p), maxMass). The four-argument form is P11's
/// `ftf_sample_resonance_mass` in string_fragmentation.cuh, deviate for deviate.
template <typename Rng>
__host__ __device__ inline double ftf_sample_mass_of(int pdg, double max_mass,
                                                     FtfRefusal* refused, Rng& rng) {
  const data::FtfHadron* h = data::ftf_find_hadron(pdg);
  if (h == nullptr) {
    if (refused != nullptr) { *refused = FtfRefusal::kUnknownHadronCode; }
    return 0.0;
  }
  const double min_mass = ftf_get_minimum_mass(pdg, refused);
  return ftf_sample_resonance_mass<double>(h->mass, h->width, min_mass, max_mass, rng);
}

/// G4DiffractiveExcitation::ChooseP - `Pmin * (Pmax/Pmin)^u`, i.e. P(x) ~ 1/x on [Pmin, Pmax].
///
/// `G4Pow::powA` and NOT `std::pow`: powA is an expansion about tabulated points and differs
/// from the exact power at 1e-7 (docs/HADRONIC_PLAN.md section 8), which is four orders above
/// this port's tolerance and is spent on every non-diffractive collision.
///
/// Geant4 THROWS for `Pmin <= 0 || Pmax <= Pmin`. A kernel cannot throw; every caller here has
/// already established `Pmax > Pmin > 0` by a `SqrtS < ProjMassT + TargMassT` test, so the
/// guard returns Pmin and sets the caller's refusal rather than inventing a range.
template <typename Rng>
__host__ __device__ inline double ftf_choose_p(double p_min, double p_max, bool* ok, Rng& rng) {
  const double range = p_max - p_min;
  if (p_min <= 0.0 || range <= 0.0) {
    *ok = false;
    return p_min;
  }
  *ok = true;
  return p_min * data::g4pow_pow_a<double>(p_max / p_min, rng.uniform());
}

/// G4DiffractiveExcitation::UnpackMeson. Note that the six flavour-neutral codes it special-
/// cases are NOT the same six G4DiffractiveSplitableHadron::ChooseStringEnds special-cases:
/// this one adds eta_c (441), J/psi (443) and Upsilon (553) and resolves them to c-cbar and
/// b-bbar WITHOUT a deviate, while pi0/eta/eta' still cost one.
template <typename Rng>
__host__ __device__ inline void ftf_unpack_meson(int id_pdg, int* q1, int* q2, Rng& rng) {
  const int abs_id = (id_pdg < 0) ? -id_pdg : id_pdg;
  if (!(abs_id == 111 || abs_id == 221 || abs_id == 331 || abs_id == 441 || abs_id == 443 ||
        abs_id == 553)) {
    *q1 = abs_id / 100;
    *q2 = (abs_id % 100) / 10;
    const int mx = (*q1 > *q2) ? *q1 : *q2;
    int anti = 1 - 2 * (mx % 2);
    if (id_pdg < 0) { anti *= -1; }
    *q1 *= anti;
    *q2 *= -1 * anti;
  } else {
    if (abs_id == 441 || abs_id == 443) {
      *q1 = 4;
      *q2 = -4;
    } else if (abs_id == 553) {
      *q1 = 5;
      *q2 = -5;
    } else {
      if (rng.uniform() < 0.5) {
        *q1 = 1;
        *q2 = -1;
      } else {
        *q1 = 2;
        *q2 = -2;
      }
    }
  }
}

/// G4DiffractiveExcitation::UnpackBaryon - three integer divisions on the SIGNED code, so an
/// anti-baryon comes out with three negative quark indices.
__host__ __device__ inline void ftf_unpack_baryon(int id_pdg, int* q1, int* q2, int* q3) {
  *q1 = id_pdg / 1000;
  *q2 = (id_pdg % 1000) / 100;
  *q3 = (id_pdg % 100) / 10;
}

/// G4DiffractiveExcitation::NewNucleonId - sort the three flavours descending and build
/// `Q1*1000 + Q2*100 + Q3*10 + 2`, i.e. always the spin-1/2 code; the callers add 2 to make a
/// Delta. The sort is NOT a full sort: the first `else if` makes it three comparisons where
/// four would be needed in general, and for the (a < b < c) case it still lands ordered because
/// the second swap catches it. Transcribed as written.
__host__ __device__ inline int ftf_new_nucleon_id(int q1, int q2, int q3) {
  int tmp = 0;
  if (q3 > q2) {
    tmp = q2;
    q2 = q3;
    q3 = tmp;
  } else if (q3 > q1) {
    tmp = q1;
    q1 = q3;
    q3 = tmp;
  }
  if (q2 > q1) {
    tmp = q1;
    q1 = q2;
    q2 = tmp;
  }
  return q1 * 1000 + q2 * 100 + q3 * 10 + 2;
}

/// `GetPDGiIsospin() == 3` - "is this hadron a Delta". The column comes from the oracle
/// (ref/oracle/ftf_hadrons.csv), not from a rule about PDG codes.
__host__ __device__ inline bool ftf_is_isospin_three_halves(int pdg) {
  const data::FtfHadron* h = data::ftf_find_hadron(pdg);
  return (h != nullptr) && (h->iisospin == 3);
}

/// G4DiffractiveExcitation::CommonVariables, field for field.
struct ExciteCommon {
  int projectile_pdg = 0, abs_projectile_pdg = 0, target_pdg = 0, abs_target_pdg = 0;
  double m0_projectile = 0.0, m0_projectile2 = 0.0, m0_target = 0.0, m0_target2 = 0.0;
  double proj_mass_t = 0.0, proj_mass_t2 = 0.0, targ_mass_t = 0.0, targ_mass_t2 = 0.0;
  double mmin_projectile = 0.0, mmin_target = 0.0;
  double projectile_diff_min = 0.0, projectile_diff_min2 = 0.0;
  double projectile_nondiff_min = 0.0, projectile_nondiff_min2 = 0.0;
  double target_diff_min = 0.0, target_diff_min2 = 0.0;
  double target_nondiff_min = 0.0, target_nondiff_min2 = 0.0;
  double s = 0.0, sqrt_s = 0.0, pt2 = 0.0, pz_cms = 0.0, pz_cms2 = 0.0;
  double max_pt_square = 0.0;
  double prob_exc = 0.0, qminus = 0.0, qplus = 0.0;
  double pminus_new = 0.0, pplus_new = 0.0, tminus_new = 0.0, tplus_new = 0.0;
  double pminus_min = 0.0, pminus_max = 0.0, tplus_min = 0.0, tplus_max = 0.0;
  double prob_projectile_diffraction = 0.0, prob_target_diffraction = 0.0;
  double prob_of_diffraction = 0.0;
  Vec4 p_projectile, p_target, q_momentum;
  LorentzRot to_cms, to_lab;
  FtfRefusal refused = FtfRefusal::kNone;
};

/// G4DiffractiveExcitation::ExciteParticipants_doChargeExchange.
///
/// Returns 0 (done - an elastic scattering of the two new hadrons was performed), 1 (continue
/// into the diffraction sampling) or 99 (failed).
///
/// The meson arm and the baryon arm are almost disjoint code, and the ONE thing they share is
/// what "exchange" means: the target is always a NUCLEON, never an antinucleon, so only a
/// QUARK can cross. An anti-baryon projectile has only anti-quarks and returns 1 immediately -
/// which is why an anti-proton beam never charge-exchanges and always goes to the
/// diffraction/annihilation branches.
///
/// THE MESON ARM'S `shootInt(N)+1` IS A UNIFORM INTEGER IN [1, N] AND COSTS ONE DEVIATE.
/// `G4RandFlat::shootInt(n)` is `(long)(n*flat())` in CLHEP, so `1 + (int)(N*u)`.
template <typename Rng>
__host__ __device__ inline int ftf_excite_do_charge_exchange(SplitableHadron* projectile,
                                                             SplitableHadron* target,
                                                             const FtfParameters<double>* params,
                                                             ExciteCommon* c, Rng& rng) {
  const int failed = 99;
  const double delta_prob_at_quark_exchange = params->delta_prob_at_quark_exchange;
  double mtest_pr = 0.0, mtest_tr = 0.0;

  int new_proj_code = 0, new_targ_code = 0;
  int proj_q1 = 0, proj_q2 = 0, proj_q3 = 0;
  if (c->abs_projectile_pdg < 1000) {
    ftf_unpack_meson(c->projectile_pdg, &proj_q1, &proj_q2, rng);
  } else {
    ftf_unpack_baryon(c->projectile_pdg, &proj_q1, &proj_q2, &proj_q3);
  }
  int targ_q1 = 0, targ_q2 = 0, targ_q3 = 0;
  ftf_unpack_baryon(c->target_pdg, &targ_q1, &targ_q2, &targ_q3);

  if (c->abs_projectile_pdg < 1000) {  // ---------------- projectile is a meson ---------------
    bool is_proj_q1_quark = false;
    int proj_exchange_q = proj_q2;
    if (proj_q1 > 0) {
      is_proj_q1_quark = true;
      proj_exchange_q = proj_q1;
    }
    int n_possible_states = 0;
    if (proj_exchange_q != targ_q1) { n_possible_states++; }
    if (proj_exchange_q != targ_q2) { n_possible_states++; }
    if (proj_exchange_q != targ_q3) { n_possible_states++; }
    const int n_sampled =
        static_cast<int>(static_cast<long>(n_possible_states * rng.uniform())) + 1;
    n_possible_states = 0;
    int targ_exchange_q = 0;
    if (proj_exchange_q != targ_q1) {
      if (++n_possible_states == n_sampled) {
        targ_exchange_q = targ_q1;
        targ_q1 = proj_exchange_q;
        if (is_proj_q1_quark) { proj_q1 = targ_exchange_q; } else { proj_q2 = targ_exchange_q; }
      }
    }
    if (proj_exchange_q != targ_q2) {
      if (++n_possible_states == n_sampled) {
        targ_exchange_q = targ_q2;
        targ_q2 = proj_exchange_q;
        if (is_proj_q1_quark) { proj_q1 = targ_exchange_q; } else { proj_q2 = targ_exchange_q; }
      }
    }
    if (proj_exchange_q != targ_q3) {
      if (++n_possible_states == n_sampled) {
        targ_exchange_q = targ_q3;
        targ_q3 = proj_exchange_q;
        if (is_proj_q1_quark) { proj_q1 = targ_exchange_q; } else { proj_q2 = targ_exchange_q; }
      }
    }

    const int a_proj_q1 = (proj_q1 < 0) ? -proj_q1 : proj_q1;
    const int a_proj_q2 = (proj_q2 < 0) ? -proj_q2 : proj_q2;
    bool proj_excited = false;
    const int max_number_of_attempts = 50;
    int attempts = 0;
    while (attempts++ < max_number_of_attempts) {
      const double prob_spin0 = 0.5;
      const double ksi = rng.uniform();
      if (a_proj_q1 == a_proj_q2) {
        if (rng.uniform() < prob_spin0) {  // Meson spin = 0 (pseudo-scalar)
          if (a_proj_q1 < 3) {
            new_proj_code = 111;  // pi0
            if (ksi < 0.5) {
              new_proj_code = 221;  // eta
              if (ksi < 0.25) { new_proj_code = 331; }  // eta'
            }
          } else if (a_proj_q1 == 3) {
            new_proj_code = 221;
            if (ksi < 0.5) { new_proj_code = 331; }
          } else if (a_proj_q1 == 4) {
            new_proj_code = 441;  // eta_c(1S)
          } else if (a_proj_q1 == 5) {
            new_proj_code = 551;  // eta_b(1S)
          }
        } else {  // Meson spin = 1 (vector meson)
          if (a_proj_q1 < 3) {
            new_proj_code = 113;  // rho0
            if (ksi < 0.5) { new_proj_code = 223; }  // omega
          } else if (a_proj_q1 == 3) {
            new_proj_code = 333;  // phi
          } else if (a_proj_q1 == 4) {
            new_proj_code = 443;  // J/psi(1S)
          } else if (a_proj_q1 == 5) {
            new_proj_code = 553;  // Upsilon(1S)
          }
        }
      } else {
        if (a_proj_q1 > a_proj_q2) {
          new_proj_code = a_proj_q1 * 100 + a_proj_q2 * 10 + 1;
        } else {
          new_proj_code = a_proj_q2 * 100 + a_proj_q1 * 10 + 1;
        }
      }
      proj_excited = false;
      if (a_proj_q1 <= 3 && a_proj_q2 <= 3 && rng.uniform() < 0.5) {
        new_proj_code += 2;  // Excited meson: last PDG digit 3 instead of 1
        proj_excited = true;
      }

      // The sign of the meson code from the SIGNED quark indices: u and c count +2, d, s and b
      // count -1, in e/3 units.
      int value = proj_q1, abs_value = a_proj_q1, qquarks = 0;
      for (int iq = 0; iq < 2; ++iq) {
        if (iq == 1) {
          value = proj_q2;
          abs_value = a_proj_q2;
        }
        if (abs_value == 2 || abs_value == 4) {
          qquarks += 2 * value / abs_value;
        } else {
          qquarks -= value / abs_value;
        }
      }
      if (qquarks < 0 || (qquarks == 0 && a_proj_q1 != a_proj_q2 && a_proj_q1 % 2 == 0)) {
        new_proj_code *= -1;
      }

      // Projectile
      if (data::ftf_find_hadron(new_proj_code) == nullptr) { continue; }
      c->mmin_projectile = ftf_get_minimum_mass(new_proj_code, &c->refused);
      if (c->sqrt_s - c->m0_target < c->mmin_projectile) { continue; }
      {
        const data::FtfHadron* tp = data::ftf_find_hadron(new_proj_code);
        mtest_pr = ftf_sample_mass_of(new_proj_code, tp->mass + 5.0 * tp->width, &c->refused, rng);
      }

      // Target
      new_targ_code = ftf_new_nucleon_id(targ_q1, targ_q2, targ_q3);

      if (targ_q1 <= 3 && targ_q2 <= 3 && targ_q3 <= 3) {
        if (targ_q1 != targ_q2 && targ_q1 != targ_q3 && targ_q2 != targ_q3) {
          // Lambda or Sigma0 ? Two deviates in the worst case, and the second is drawn only
          // when the first failed - so the branch taken is visible in the draw count.
          if (rng.uniform() < 0.5) {
            new_targ_code += 2;
          } else if (rng.uniform() < 0.75) {
            new_targ_code = 3122;  // Lambda
          }
        } else if (targ_q1 == targ_q2 && targ_q1 == targ_q3) {
          new_targ_code += 2;
          proj_excited = true;  // Create Delta isobar
        } else if (ftf_is_isospin_three_halves(target->pdg)) {  // Delta was the target
          if (rng.uniform() > delta_prob_at_quark_exchange) {
            new_targ_code += 2;
            proj_excited = true;
          }
        } else if (!proj_excited && rng.uniform() < delta_prob_at_quark_exchange &&
                   c->sqrt_s > c->m0_projectile + data::ftf_find_hadron(2224)->mass) {
          new_targ_code += 2;  // Create Delta isobar
        }
      }

      // Excited Lambda, Sigma and Xi states do not exist in Geant4 (nor, for 3124, in the PDG),
      // so the code is walked back to the ground state.
      if (new_targ_code == 3124 || new_targ_code == 3224 || new_targ_code == 3214 ||
          new_targ_code == 3114 || new_targ_code == 3324 || new_targ_code == 3314) {
        new_targ_code -= 2;
      }
      // Geant4 has no Xi_c' or Xi_b': they are mapped onto Xi_c and Xi_b by hand.
      if (new_targ_code == 4322) {
        new_targ_code = 4232;
      } else if (new_targ_code == 4312) {
        new_targ_code = 4132;
      } else if (new_targ_code == 5312) {
        new_targ_code = 5132;
      } else if (new_targ_code == 5322) {
        new_targ_code = 5232;
      }

      const data::FtfHadron* tt = data::ftf_find_hadron(new_targ_code);
      if (tt == nullptr) { continue; }
      c->mmin_target = ftf_get_minimum_mass(new_targ_code, &c->refused);
      if (c->sqrt_s - mtest_pr < c->mmin_target) { continue; }
      mtest_tr = ftf_sample_mass_of(new_targ_code, tt->mass + 5.0 * tt->width, &c->refused, rng);
      if (c->sqrt_s > mtest_pr + mtest_tr) { break; }
    }
    if (attempts >= max_number_of_attempts) { return failed; }

    if (mtest_pr >= c->p_projectile.mag() || projectile->status != 0) {
      c->m0_projectile = mtest_pr;
    }
    c->m0_projectile2 = c->m0_projectile * c->m0_projectile;
    c->projectile_diff_min = c->m0_projectile + 220.0 * units::MeV<double>();
    c->projectile_nondiff_min = c->m0_projectile + 220.0 * units::MeV<double>();
    if (mtest_tr >= c->p_target.mag() || target->status != 0) { c->m0_target = mtest_tr; }
    c->m0_target2 = c->m0_target * c->m0_target;
    c->target_diff_min = c->m0_target + 220.0 * units::MeV<double>();
    c->target_nondiff_min = c->m0_target + 220.0 * units::MeV<double>();

  } else {  // ---------------- projectile is a baryon or an anti-baryon ----------------

    // An anti-baryon has only anti-quarks and the target only quarks: nothing can cross without
    // making a q-q-qbar, so the exchange is skipped and the collision continues.
    if (c->projectile_pdg < 0) { return 1; }

    bool is_projectile_exchanged_q = false;
    int first_q = targ_q1, second_q = targ_q2, third_q = targ_q3;
    int other_first_q = proj_q1, other_second_q = proj_q2, other_third_q = proj_q3;
    if (rng.uniform() < 0.5) {
      is_projectile_exchanged_q = true;
      first_q = proj_q1;
      second_q = proj_q2;
      third_q = proj_q3;
      other_first_q = targ_q1;
      other_second_q = targ_q2;
      other_third_q = targ_q3;
    }
    int exchanged_q = 0;
    const double ksi = rng.uniform();
    if (ksi < 0.333333) {
      exchanged_q = first_q;
    } else if (0.333333 <= ksi && ksi < 0.666667) {
      exchanged_q = second_q;
    } else {
      exchanged_q = third_q;
    }

    const double prob_same = params->prob_of_same_quark_exchange;
    const int max_count = 100;
    int count = 0, other_exchanged_q = 0;
    do {
      if (exchanged_q != other_first_q || rng.uniform() < prob_same) {
        other_exchanged_q = other_first_q;
        other_first_q = exchanged_q;
        exchanged_q = other_exchanged_q;
      } else {
        if (exchanged_q != other_second_q || rng.uniform() < prob_same) {
          other_exchanged_q = other_second_q;
          other_second_q = exchanged_q;
          exchanged_q = other_exchanged_q;
        } else {
          if (exchanged_q != other_third_q || rng.uniform() < prob_same) {
            other_exchanged_q = other_third_q;
            other_third_q = exchanged_q;
            exchanged_q = other_exchanged_q;
          }
        }
      }
    } while (other_exchanged_q == 0 && ++count < max_count);
    if (count >= max_count) { return failed; }

    if (ksi < 0.333333) {
      first_q = exchanged_q;
    } else if (0.333333 <= ksi && ksi < 0.666667) {
      second_q = exchanged_q;
    } else {
      third_q = exchanged_q;
    }
    if (is_projectile_exchanged_q) {
      proj_q1 = first_q;       proj_q2 = second_q;       proj_q3 = third_q;
      targ_q1 = other_first_q; targ_q2 = other_second_q; targ_q3 = other_third_q;
    } else {
      targ_q1 = first_q;       targ_q2 = second_q;       targ_q3 = third_q;
      proj_q1 = other_first_q; proj_q2 = other_second_q; proj_q3 = other_third_q;
    }

    new_proj_code = ftf_new_nucleon_id(proj_q1, proj_q2, proj_q3);
    new_targ_code = ftf_new_nucleon_id(targ_q1, targ_q2, targ_q3);

    for (int i_hadron = 0; i_hadron < 2; i_hadron++) {
      int code_q1 = proj_q1, code_q2 = proj_q2, code_q3 = proj_q3;
      int new_had_code = new_proj_code;
      double mass_constraint = c->m0_target;
      bool is_hadron_a_delta = ftf_is_isospin_three_halves(projectile->pdg);
      if (i_hadron == 1) {
        code_q1 = targ_q1;
        code_q2 = targ_q2;
        code_q3 = targ_q3;
        new_had_code = new_targ_code;
        mass_constraint = c->m0_projectile;
        is_hadron_a_delta = ftf_is_isospin_three_halves(target->pdg);
      }
      if (code_q1 > 3 || code_q2 > 3 || code_q3 > 3) { continue; }
      if (code_q1 == code_q2 && code_q1 == code_q3) {
        new_had_code += 2;  // Delta++ (uuu) or Delta- (ddd)
      } else if (is_hadron_a_delta) {
        if (rng.uniform() > delta_prob_at_quark_exchange) {
          new_had_code += 2;
        } else {
          new_had_code += 0;
        }
      } else {
        if (rng.uniform() < delta_prob_at_quark_exchange &&
            c->sqrt_s > data::ftf_find_hadron(2224)->mass + mass_constraint) {
          new_had_code += 2;
        } else {
          new_had_code += 0;
        }
      }
      if (i_hadron == 0) { new_proj_code = new_had_code; } else { new_targ_code = new_had_code; }
    }

    if (new_proj_code == 3124 || new_proj_code == 3224 || new_proj_code == 3214 ||
        new_proj_code == 3114 || new_proj_code == 3324 || new_proj_code == 3314) {
      new_proj_code -= 2;
    }
    if (new_targ_code == 3124 || new_targ_code == 3224 || new_targ_code == 3214 ||
        new_targ_code == 3114 || new_targ_code == 3324 || new_targ_code == 3314) {
      new_targ_code -= 2;
    }
    if (new_proj_code == 4322) {
      new_proj_code = 4232;
    } else if (new_proj_code == 4312) {
      new_proj_code = 4132;
    } else if (new_proj_code == 5312) {
      new_proj_code = 5132;
    } else if (new_proj_code == 5322) {
      new_proj_code = 5232;
    }
    if (new_targ_code == 4322) {
      new_targ_code = 4232;
    } else if (new_targ_code == 4312) {
      new_targ_code = 4132;
    } else if (new_targ_code == 5312) {
      new_targ_code = 5132;
    } else if (new_targ_code == 5322) {
      new_targ_code = 5232;
    }

    // The ORDER of the two mass samplings is itself sampled, because energy conservation makes
    // the second one's range depend on the first one's answer.
    int first_hadron_code = new_targ_code, second_hadron_code = new_proj_code;
    int first_hadron_status = target->status, second_hadron_status = projectile->status;
    double mass_constraint = c->m0_projectile;
    bool is_first_target = true;
    if (rng.uniform() < 0.5) {
      first_hadron_code = new_proj_code;
      second_hadron_code = new_targ_code;
      first_hadron_status = projectile->status;
      second_hadron_status = target->status;
      mass_constraint = c->m0_target;
      is_first_target = false;
    }
    double mtest_1st = 0.0, mtest_2nd = 0.0, mmin_1st = 0.0, mmin_2nd = 0.0;
    for (int i_sampling_case = 0; i_sampling_case < 2; i_sampling_case++) {
      int a_hadron_code = first_hadron_code;
      int a_hadron_status = first_hadron_status;
      if (i_sampling_case == 1) {
        a_hadron_code = second_hadron_code;
        a_hadron_status = second_hadron_status;
        mass_constraint = mtest_1st;
      }
      double mtest_hadron = 0.0, mmin_hadron = 0.0;
      if (a_hadron_status == 1 || a_hadron_status == 2) {
        const data::FtfHadron* tp = data::ftf_find_hadron(a_hadron_code);
        if (tp == nullptr) { return failed; }
        mmin_hadron = ftf_get_minimum_mass(a_hadron_code, &c->refused);
        if (c->sqrt_s - mass_constraint < mmin_hadron) { return failed; }
        if (tp->width == 0.0) {
          mtest_hadron = ftf_sample_mass_of(a_hadron_code, tp->mass, &c->refused, rng);
        } else {
          const int max_number_of_attempts = 50;
          int attempts = 0;
          while (attempts < max_number_of_attempts) {
            attempts++;
            mtest_hadron = ftf_sample_mass_of(a_hadron_code, tp->mass + 5.0 * tp->width,
                                              &c->refused, rng);
            if (c->sqrt_s < mtest_hadron + mass_constraint) {
              continue;
            } else {
              break;
            }
          }
          if (attempts >= max_number_of_attempts) { return failed; }
        }
      }
      if (i_sampling_case == 0) {
        mtest_1st = mtest_hadron;
        mmin_1st = mmin_hadron;
      } else {
        mtest_2nd = mtest_hadron;
        mmin_2nd = mmin_hadron;
      }
    }
    if (is_first_target) {
      mtest_tr = mtest_1st;
      mtest_pr = mtest_2nd;
      c->mmin_target = mmin_1st;
      c->mmin_projectile = mmin_2nd;
    } else {
      mtest_tr = mtest_2nd;
      mtest_pr = mtest_1st;
      c->mmin_target = mmin_2nd;
      c->mmin_projectile = mmin_1st;
    }

    if (mtest_pr != 0.0) {
      c->m0_projectile = mtest_pr;
      c->m0_projectile2 = c->m0_projectile * c->m0_projectile;
      c->projectile_diff_min = c->m0_projectile + 220.0 * units::MeV<double>();
      c->projectile_nondiff_min = c->m0_projectile + 220.0 * units::MeV<double>();
    }
    if (mtest_tr != 0.0) {
      c->m0_target = mtest_tr;
      c->m0_target2 = c->m0_target * c->m0_target;
      c->target_diff_min = c->m0_target + 220.0 * units::MeV<double>();
      c->target_nondiff_min = c->m0_target + 220.0 * units::MeV<double>();
    }
  }

  if (c->sqrt_s < c->m0_projectile + c->m0_target) { return failed; }

  c->pz_cms2 = (c->s * c->s + c->m0_projectile2 * c->m0_projectile2 +
                c->m0_target2 * c->m0_target2 -
                2.0 * (c->s * (c->m0_projectile2 + c->m0_target2) +
                       c->m0_projectile2 * c->m0_target2)) /
               4.0 / c->s;
  if (c->pz_cms2 < 0.0) { return failed; }  // not enough energy for a Delta

  projectile->pdg = new_proj_code;
  target->pdg = new_targ_code;
  c->pz_cms = std::sqrt(c->pz_cms2);
  c->p_projectile.v.z = c->pz_cms;
  c->p_projectile.e = std::sqrt(c->m0_projectile2 + c->pz_cms2);
  c->p_target.v.z = -c->pz_cms;
  c->p_target.e = std::sqrt(c->m0_target2 + c->pz_cms2);

  if (projectile->status != 0) { projectile->status = 2; }
  if (target->status != 0) { target->status = 2; }

  if (c->sqrt_s < c->m0_projectile + c->target_diff_min ||
      c->sqrt_s < c->projectile_diff_min + c->m0_target || c->prob_of_diffraction == 0.0) {
    c->prob_exc = 0.0;
  }

  if (rng.uniform() > c->prob_exc) {  // Make elastic scattering of the two NEW hadrons
    c->p_projectile = lorentz_apply(c->to_lab, c->p_projectile);
    c->p_target = lorentz_apply(c->to_lab, c->p_target);
    projectile->momentum = c->p_projectile;
    target->momentum = c->p_target;
    const bool result = ftf_elastic_scattering(projectile, target, params, rng);
    return result ? 0 : failed;
  }

  c->prob_of_diffraction = c->prob_projectile_diffraction + c->prob_target_diffraction;
  if (c->prob_of_diffraction != 0.0) {
    c->prob_projectile_diffraction /= c->prob_of_diffraction;
    c->prob_target_diffraction /= c->prob_of_diffraction;
  }
  return 1;
}

/// G4DiffractiveExcitation::ExciteParticipants_doDiffraction.
///
/// One of the two hadrons is replaced by its minimum DIFFRACTIVE state and the other keeps its
/// mass; a transverse momentum is sampled at 1.2 times the elastic average pt^2, and then a
/// light-cone component is drawn from ChooseP's 1/x distribution. The `while (loopCondition)`
/// repeats the whole sampling when the recoiling hadron did not actually end up excited, and
/// answers `Qmomentum = 0` and false after 1000 tries - which ExciteParticipants reads as a
/// failed excitation and turns into an elastic scattering.
template <typename Rng>
__host__ __device__ inline bool ftf_excite_do_diffraction(SplitableHadron* projectile,
                                                          SplitableHadron* target,
                                                          const FtfParameters<double>* params,
                                                          ExciteCommon* c, Rng& rng) {
  bool is_projectile_diffraction = false;
  if (rng.uniform() < c->prob_projectile_diffraction) {
    is_projectile_diffraction = true;
    c->proj_mass_t2 = c->projectile_diff_min2;
    c->proj_mass_t = c->projectile_diff_min;
    c->targ_mass_t2 = c->m0_target2;
    c->targ_mass_t = c->m0_target;
  } else {
    c->proj_mass_t2 = c->m0_projectile2;
    c->proj_mass_t = c->m0_projectile;
    c->targ_mass_t2 = c->target_diff_min2;
    c->targ_mass_t = c->target_diff_min;
  }

  if (c->sqrt_s < c->proj_mass_t + c->targ_mass_t) { return false; }

  c->pz_cms2 = (c->s * c->s + c->proj_mass_t2 * c->proj_mass_t2 +
                c->targ_mass_t2 * c->targ_mass_t2 -
                2.0 * (c->s * (c->proj_mass_t2 + c->targ_mass_t2) +
                       c->proj_mass_t2 * c->targ_mass_t2)) /
               4.0 / c->s;
  if (c->pz_cms2 < 0.0) { return false; }
  c->max_pt_square = c->pz_cms2;

  const double diffr_average_pt2 = params->avarage_pt2_of_elastic_scattering * 1.2;
  bool loop_condition = true;
  int whilecount = 0;
  do {
    whilecount++;
    if (whilecount > 1000) {
      c->q_momentum = Vec4(0.0, 0.0, 0.0, 0.0);
      return false;  // Ignore this interaction
    }

    c->q_momentum =
        Vec4(ftf_gaussian_pt_excitation(diffr_average_pt2, c->max_pt_square, rng), 0.0);
    c->pt2 = g4gpu::mag2(c->q_momentum.v);
    if (is_projectile_diffraction) {
      c->proj_mass_t2 = c->projectile_diff_min2 + c->pt2;
      c->targ_mass_t2 = c->m0_target2 + c->pt2;
    } else {
      c->proj_mass_t2 = c->m0_projectile2 + c->pt2;
      c->targ_mass_t2 = c->target_diff_min2 + c->pt2;
    }
    c->proj_mass_t = std::sqrt(c->proj_mass_t2);
    c->targ_mass_t = std::sqrt(c->targ_mass_t2);
    if (c->sqrt_s < c->proj_mass_t + c->targ_mass_t) { continue; }

    c->pz_cms2 = (c->s * c->s + c->proj_mass_t2 * c->proj_mass_t2 +
                  c->targ_mass_t2 * c->targ_mass_t2 -
                  2.0 * (c->s * (c->proj_mass_t2 + c->targ_mass_t2) +
                         c->proj_mass_t2 * c->targ_mass_t2)) /
                 4.0 / c->s;
    if (c->pz_cms2 < 0.0) { continue; }

    c->pz_cms = std::sqrt(c->pz_cms2);
    bool ok = true;
    if (is_projectile_diffraction) {
      c->pminus_min = std::sqrt(c->proj_mass_t2 + c->pz_cms2) - c->pz_cms;
      c->pminus_max = c->sqrt_s - c->targ_mass_t;
      c->pminus_new = ftf_choose_p(c->pminus_min, c->pminus_max, &ok, rng);
      if (!ok) {
        c->q_momentum = Vec4(0.0, 0.0, 0.0, 0.0);
        return false;
      }
      c->tminus_new = c->sqrt_s - c->pminus_new;
      c->qminus = lv_minus(c->p_target) - c->tminus_new;
      c->tplus_new = c->targ_mass_t2 / c->tminus_new;
      c->qplus = lv_plus(c->p_target) - c->tplus_new;
      c->q_momentum.v.z = (c->qplus - c->qminus) / 2.0;
      c->q_momentum.e = (c->qplus + c->qminus) / 2.0;
      const Vec4 pn = c->p_projectile + c->q_momentum;
      loop_condition = (pn.e * pn.e - g4gpu::mag2(pn.v)) < c->projectile_diff_min2;
    } else {
      c->tplus_min = std::sqrt(c->targ_mass_t2 + c->pz_cms2) - c->pz_cms;
      c->tplus_max = c->sqrt_s - c->proj_mass_t;
      c->tplus_new = ftf_choose_p(c->tplus_min, c->tplus_max, &ok, rng);
      if (!ok) {
        c->q_momentum = Vec4(0.0, 0.0, 0.0, 0.0);
        return false;
      }
      c->pplus_new = c->sqrt_s - c->tplus_new;
      c->qplus = c->pplus_new - lv_plus(c->p_projectile);
      c->pminus_new = c->proj_mass_t2 / c->pplus_new;
      c->qminus = c->pminus_new - lv_minus(c->p_projectile);
      c->q_momentum.v.z = (c->qplus - c->qminus) / 2.0;
      c->q_momentum.e = (c->qplus + c->qminus) / 2.0;
      const Vec4 tn = c->p_target - c->q_momentum;
      loop_condition = (tn.e * tn.e - g4gpu::mag2(tn.v)) < c->target_diff_min2;
    }
  } while (loop_condition);

  if (is_projectile_diffraction) {
    // `SetStatus(0)` and then `if (GetStatus() == 2) SetStatus(1)` - the second test reads the
    // value the first line just wrote, so it can never fire. Transcribed as the dead line it
    // is, because removing it would make the reader think status 2 survives here.
    projectile->status = 0;
    if (projectile->status == 2) { projectile->status = 1; }
    if (target->status == 1 && target->collision_count == 0) { target->status = 2; }
  } else {
    target->status = 0;
  }
  return true;
}

/// The `while` condition of ExciteParticipants_doNonDiffraction's sampling loop, as a function
/// because it reads two four-vectors that are built from the current Qmomentum.
///
/// IT IS RE-EVALUATED AFTER EVERY `continue`, and that is not the same as caching a bool: a
/// `continue` from the `SqrtS < ProjMassT + TargMassT` or `PZcms2 < 0` test leaves Qmomentum as
/// the pure transverse vector just sampled (pz and E still zero), and the condition is then
/// evaluated on THAT - which is why the loop can exit through a Qmomentum whose longitudinal
/// components were never assigned. The diffractive arm one function up uses a stored
/// `loopCondition` variable instead and therefore does NOT re-evaluate; the two arms differ,
/// and each is transcribed as it is written.
__host__ __device__ inline bool ftf_nondiff_loop_condition(const ExciteCommon* c) {
  const Vec4 pn = c->p_projectile + c->q_momentum;
  if ((pn.e * pn.e - g4gpu::mag2(pn.v)) < c->projectile_nondiff_min2) { return true; }
  const Vec4 tn = c->p_target - c->q_momentum;
  return (tn.e * tn.e - g4gpu::mag2(tn.v)) < c->target_nondiff_min2;
}

/// G4DiffractiveExcitation::ExciteParticipants_doNonDiffraction.
///
/// Both hadrons are excited to at least their non-diffractive minimum masses. The two
/// light-cone variables are drawn independently, each either from ChooseP's 1/x law or
/// uniformly, with `GetProbLogDistrPrD()` and `GetProbLogDistr()` deciding which - and the
/// ORDER of the two draws is itself a coin toss, which costs one deviate and changes nothing
/// about the distribution, only about the stream.
template <typename Rng>
__host__ __device__ inline bool ftf_excite_do_non_diffraction(SplitableHadron* projectile,
                                                              SplitableHadron* target,
                                                              const FtfParameters<double>* params,
                                                              ExciteCommon* c, Rng& rng) {
  c->proj_mass_t2 = c->projectile_nondiff_min2;
  c->proj_mass_t = c->projectile_nondiff_min;
  c->targ_mass_t2 = c->target_nondiff_min2;
  c->targ_mass_t = c->target_nondiff_min;
  if (c->sqrt_s < c->proj_mass_t + c->targ_mass_t) { return false; }

  c->pz_cms2 = (c->s * c->s + c->proj_mass_t2 * c->proj_mass_t2 +
                c->targ_mass_t2 * c->targ_mass_t2 -
                2.0 * (c->s * (c->proj_mass_t2 + c->targ_mass_t2) +
                       c->proj_mass_t2 * c->targ_mass_t2)) /
               4.0 / c->s;
  // NOTE: there is no `PZcms2 < 0` test here, unlike in the diffractive arm - which has BOTH
  // this guard and the `SqrtS < ProjMassT + TargMassT` one above. Algebraically the second
  // implies the first, because PZcms2 is `lambda(S, Mp2, Mt2)/4S` and the mass test is exactly
  // `sqrt(S) >= sqrt(Mp2) + sqrt(Mt2)`. So the omission is safe except at the rounding level,
  // where a lambda of -1e-9 makes `maxPtSquare` negative, `exp(-maxPt2/<pt2>)` slightly greater
  // than 1, and Pt2 slightly NEGATIVE. This arm's GaussianPt has no `Pt2 > 0` guard on its
  // square root (G4ElasticHNScattering's does), so Pt would be a NaN, every later comparison
  // against it would be false, and the do-while would EXIT rather than retry - handing the
  // caller a NaN Qmomentum which it then adds to the projectile's four-momentum. Transcribed as
  // written; adding a guard here would hide the case rather than fix it.
  c->max_pt_square = c->pz_cms2;

  int whilecount = 0;
  do {
    whilecount++;
    if (whilecount > 1000) {
      c->q_momentum = Vec4(0.0, 0.0, 0.0, 0.0);
      return false;
    }

    c->q_momentum =
        Vec4(ftf_gaussian_pt_excitation(params->average_pt2, c->max_pt_square, rng), 0.0);
    c->pt2 = g4gpu::mag2(c->q_momentum.v);
    c->proj_mass_t2 = c->projectile_nondiff_min2 + c->pt2;
    c->proj_mass_t = std::sqrt(c->proj_mass_t2);
    c->targ_mass_t2 = c->target_nondiff_min2 + c->pt2;
    c->targ_mass_t = std::sqrt(c->targ_mass_t2);
    if (c->sqrt_s < c->proj_mass_t + c->targ_mass_t) { continue; }

    c->pz_cms2 = (c->s * c->s + c->proj_mass_t2 * c->proj_mass_t2 +
                  c->targ_mass_t2 * c->targ_mass_t2 -
                  2.0 * (c->s * (c->proj_mass_t2 + c->targ_mass_t2) +
                         c->proj_mass_t2 * c->targ_mass_t2)) /
                 4.0 / c->s;
    if (c->pz_cms2 < 0.0) { continue; }

    c->pz_cms = std::sqrt(c->pz_cms2);
    c->pminus_min = std::sqrt(c->proj_mass_t2 + c->pz_cms2) - c->pz_cms;
    c->pminus_max = c->sqrt_s - c->targ_mass_t;
    c->tplus_min = std::sqrt(c->targ_mass_t2 + c->pz_cms2) - c->pz_cms;
    c->tplus_max = c->sqrt_s - c->proj_mass_t;

    bool ok = true;
    if (rng.uniform() <= 0.5) {
      if (rng.uniform() < params->prob_log_distr_prd) {
        c->pminus_new = ftf_choose_p(c->pminus_min, c->pminus_max, &ok, rng);
      } else {
        c->pminus_new = (c->pminus_max - c->pminus_min) * rng.uniform() + c->pminus_min;
      }
      if (!ok) { return false; }
      if (rng.uniform() < params->prob_log_distr) {
        c->tplus_new = ftf_choose_p(c->tplus_min, c->tplus_max, &ok, rng);
      } else {
        c->tplus_new = (c->tplus_max - c->tplus_min) * rng.uniform() + c->tplus_min;
      }
      if (!ok) { return false; }
    } else {
      if (rng.uniform() < params->prob_log_distr) {
        c->tplus_new = ftf_choose_p(c->tplus_min, c->tplus_max, &ok, rng);
      } else {
        c->tplus_new = (c->tplus_max - c->tplus_min) * rng.uniform() + c->tplus_min;
      }
      if (!ok) { return false; }
      if (rng.uniform() < params->prob_log_distr_prd) {
        c->pminus_new = ftf_choose_p(c->pminus_min, c->pminus_max, &ok, rng);
      } else {
        c->pminus_new = (c->pminus_max - c->pminus_min) * rng.uniform() + c->pminus_min;
      }
      if (!ok) { return false; }
    }

    c->qminus = c->pminus_new - lv_minus(c->p_projectile);
    c->qplus = -(c->tplus_new - lv_plus(c->p_target));
    c->q_momentum.v.z = (c->qplus - c->qminus) / 2.0;
    c->q_momentum.e = (c->qplus + c->qminus) / 2.0;

  } while (ftf_nondiff_loop_condition(c));

  projectile->status = 0;
  target->status = 0;
  return true;
}

/// G4DiffractiveExcitation::ExciteParticipants.
template <typename Rng>
__host__ __device__ inline bool ftf_excite_participants(SplitableHadron* projectile,
                                                        SplitableHadron* target,
                                                        const FtfParameters<double>* params,
                                                        ExciteCommon* c, Rng& rng) {
  *c = ExciteCommon();

  c->p_projectile = projectile->momentum;
  if (c->p_projectile.v.z < 0.0) { return false; }
  c->projectile_pdg = projectile->pdg;
  c->abs_projectile_pdg = (c->projectile_pdg < 0) ? -c->projectile_pdg : c->projectile_pdg;
  {
    const data::FtfHadron* hp = data::ftf_find_hadron(c->projectile_pdg);
    if (hp == nullptr) {
      c->refused = FtfRefusal::kUnknownHadronCode;
      return false;
    }
    c->m0_projectile = hp->mass;  // Uzhi Aug.2019: the PDG mass, NOT Pprojectile.mag()
  }

  c->p_target = target->momentum;
  c->target_pdg = target->pdg;
  c->abs_target_pdg = (c->target_pdg < 0) ? -c->target_pdg : c->target_pdg;
  {
    const data::FtfHadron* ht = data::ftf_find_hadron(c->target_pdg);
    if (ht == nullptr) {
      c->refused = FtfRefusal::kUnknownHadronCode;
      return false;
    }
    c->m0_target = ht->mass;
  }

  const Vec4 psum = c->p_projectile + c->p_target;
  c->s = psum.e * psum.e - g4gpu::mag2(psum.v);
  c->sqrt_s = std::sqrt(c->s);

  // `toBePutOnMassShell` is an unconditional true since Aug 2019 (the commented-out blocks
  // above it are where it used to depend on M0 < Mmin). It is what puts the off-shell nuclear
  // nucleons back on their PDG shells - docs/RISK.md V50's requirement, met here.
  const bool to_be_put_on_mass_shell = true;
  c->mmin_projectile = ftf_get_minimum_mass(c->projectile_pdg, &c->refused);
  c->m0_projectile2 = c->m0_projectile * c->m0_projectile;
  c->projectile_diff_min = params->proj_min_diff_mass;
  c->projectile_nondiff_min = params->proj_min_non_diff_mass;
  if (c->m0_projectile > c->projectile_diff_min) {
    c->projectile_diff_min = c->mmin_projectile + 220.0 * units::MeV<double>();
    c->projectile_nondiff_min = c->mmin_projectile + 220.0 * units::MeV<double>();
    if (c->abs_projectile_pdg > 3000) {  // Strange baryon
      c->projectile_diff_min += 140.0 * units::MeV<double>();
      c->projectile_nondiff_min += 140.0 * units::MeV<double>();
    }
  }
  c->mmin_target = ftf_get_minimum_mass(c->target_pdg, &c->refused);
  c->m0_target2 = c->m0_target * c->m0_target;
  c->target_diff_min = params->tar_min_diff_mass;
  c->target_nondiff_min = params->tar_min_non_diff_mass;
  if (c->m0_target > c->target_diff_min) {
    c->target_diff_min = c->mmin_target + 220.0 * units::MeV<double>();
    c->target_nondiff_min = c->mmin_target + 220.0 * units::MeV<double>();
    if (c->abs_target_pdg > 3000) {
      c->target_diff_min += 140.0 * units::MeV<double>();
      c->target_nondiff_min += 140.0 * units::MeV<double>();
    }
  }

  const Vec3d bv = psum.boost_vector();
  c->to_cms = lorentz_boost(Vec3d{-bv.x, -bv.y, -bv.z});
  Vec4 ptmp = lorentz_apply(c->to_cms, c->p_projectile);
  if (ptmp.v.z <= 0.0) { return false; }  // "String" moving backwards in CMS
  lorentz_rotate_z(&c->to_cms, -lv_phi(ptmp));
  lorentz_rotate_y(&c->to_cms, -lv_theta(ptmp));
  c->to_lab = lorentz_inverse(c->to_cms);
  c->p_projectile = lorentz_apply(c->to_cms, c->p_projectile);
  c->p_target = lorentz_apply(c->to_cms, c->p_target);

  const double sum_masses = c->m0_projectile + c->m0_target;
  if (c->sqrt_s < sum_masses) { return false; }  // The model cannot work at low energy

  c->pz_cms2 = (c->s * c->s + c->m0_projectile2 * c->m0_projectile2 +
                c->m0_target2 * c->m0_target2 -
                2.0 * (c->s * (c->m0_projectile2 + c->m0_target2) +
                       c->m0_projectile2 * c->m0_target2)) /
               4.0 / c->s;
  if (c->pz_cms2 < 0.0) { return false; }  // can happen with an off-shell nuclear nucleon

  c->pz_cms = std::sqrt(c->pz_cms2);
  if (to_be_put_on_mass_shell) {
    if (c->p_projectile.v.z > 0.0) {
      c->p_projectile.v.z = c->pz_cms;
      c->p_target.v.z = -c->pz_cms;
    } else {
      c->p_projectile.v.z = -c->pz_cms;
      c->p_target.v.z = c->pz_cms;
    }
    c->p_projectile.e =
        std::sqrt(c->m0_projectile2 + c->p_projectile.v.x * c->p_projectile.v.x +
                  c->p_projectile.v.y * c->p_projectile.v.y + c->pz_cms2);
    c->p_target.e = std::sqrt(c->m0_target2 + c->p_target.v.x * c->p_target.v.x +
                              c->p_target.v.y * c->p_target.v.y + c->pz_cms2);
  }

  // Process probabilities as a function of the rapidity gap.
  const double projectile_rapidity = lv_rapidity(c->p_projectile);
  const double target_rapidity = lv_rapidity(c->p_target);
  const double dy = projectile_rapidity - target_rapidity;
  double qe_no_exc = ftf_get_proc_prob(params, 0, dy);
  const double qe_exc = ftf_get_proc_prob(params, 1, dy) * ftf_get_proc_prob(params, 4, dy);
  c->prob_projectile_diffraction = ftf_get_proc_prob(params, 2, dy);
  c->prob_target_diffraction = ftf_get_proc_prob(params, 3, dy);
  c->prob_of_diffraction = c->prob_projectile_diffraction + c->prob_target_diffraction;

  if (qe_no_exc + qe_exc + c->prob_projectile_diffraction + c->prob_target_diffraction > 1.0) {
    qe_no_exc = 1.0 - qe_exc - c->prob_projectile_diffraction - c->prob_target_diffraction;
  }
  if (qe_exc + qe_no_exc != 0.0) { c->prob_exc = qe_exc / (qe_exc + qe_no_exc); }
  if (1.0 - qe_exc - qe_no_exc > 0.0) {
    c->prob_projectile_diffraction /= (1.0 - qe_exc - qe_no_exc);
    c->prob_target_diffraction /= (1.0 - qe_exc - qe_no_exc);
  }

  int return_code = 1;
  if (rng.uniform() < qe_exc + qe_no_exc) {
    return_code = ftf_excite_do_charge_exchange(projectile, target, params, c, rng);
  }

  bool return_result = false;
  if (return_code == 0) {
    return_result = true;  // Successfully ended by an elastic scattering; nothing else to do
  } else if (return_code == 1) {
    c->prob_of_diffraction = c->prob_projectile_diffraction + c->prob_target_diffraction;
    if (c->prob_of_diffraction != 0.0) {
      c->prob_projectile_diffraction /= c->prob_of_diffraction;
    } else {
      c->prob_projectile_diffraction = 0.0;
    }
    c->projectile_diff_min2 = c->projectile_diff_min * c->projectile_diff_min;
    c->projectile_nondiff_min2 = c->projectile_nondiff_min * c->projectile_nondiff_min;
    c->target_diff_min2 = c->target_diff_min * c->target_diff_min;
    c->target_nondiff_min2 = c->target_nondiff_min * c->target_nondiff_min;
    if (rng.uniform() < c->prob_of_diffraction) {
      return_result = ftf_excite_do_diffraction(projectile, target, params, c, rng);
    } else {
      return_result = ftf_excite_do_non_diffraction(projectile, target, params, c, rng);
    }
    if (return_result) {
      c->p_projectile = c->p_projectile + c->q_momentum;
      c->p_target = c->p_target - c->q_momentum;
      c->p_projectile = lorentz_apply(c->to_lab, c->p_projectile);
      c->p_target = lorentz_apply(c->to_lab, c->p_target);
      projectile->momentum = c->p_projectile;
      target->momentum = c->p_target;
      projectile->collision_count += 1;
      target->collision_count += 1;
    }
  }
  return return_result;
}

/// G4DiffractiveExcitation::CreateStrings - an excited hadron becomes a string.
///
/// `first` and `second` are Geant4's `FirstString` and `SecondString`; `second` is produced
/// only by the kinky arm, which is unreachable. `*n_out` is 0, 1 or 2.
///
/// THE STRING ENDS ARE (end, start), IN THAT ORDER. `new G4ExcitedString( end, start, dir )`
/// pushes `end` first, so `GetLeftParton()` is the STRING END and `GetRightParton()` is the
/// string START - the opposite of what the names suggest. P11's `ExcitedString::left` is
/// `thePartons.front()`, so `left = end` and `right = start`, and the momenta follow.
///
/// THE NO-KINK MOMENTUM SPLIT is a two-body decay of the hadron along z with the transverse
/// momentum shared EQUALLY (`px/2`, `py/2` to each end) rather than by the light-cone fraction.
/// `Exp = sqrt(pz^2 + (mt2^2 - 4 E^2 pt^2/4)/mt2)/2` is the half-difference of the two z
/// momenta; note the `4.0*sqr(Eh)*Pt2/4.0`, which is `E^2 pt^2` written with a 4 that cancels,
/// and is transcribed as written because the cancellation is exact only in that order.
template <typename Rng>
__host__ __device__ inline void ftf_create_strings(SplitableHadron* hadron, bool is_projectile,
                                                   const FtfParameters<double>* params,
                                                   ExcitedString* out, int* n_out,
                                                   FtfRefusal* refused, Rng& rng) {
  *n_out = 0;
  const bool hadron_is_string = hadron->is_split;
  if (!hadron_is_string) { splitable_split_up(hadron, refused, rng); }

  bool ok = false;
  const int start = splitable_next_parton(hadron, &ok);
  if (!ok) { return; }  // "No start parton found"
  const int end = splitable_next_parton(hadron, &ok);
  if (!ok) { return; }  // "No end parton found"

  if (hadron_is_string) {
    // The hadron was already split, which in FTF means G4FTFAnnihilation split it: the string
    // is rebuilt from the two parton objects AS THEY ARE, momenta included. `start` is
    // Parton[0] and `end` is Parton[1], and `new G4ExcitedString(end, start, dir)` pushes `end`
    // first - so `left` is Parton[1] and `right` is Parton[0], and the momenta follow the codes.
    ExcitedString s;
    s.left = end;
    s.right = start;
    s.direction = is_projectile ? +1 : -1;
    s.pleft = hadron->parton_mom[1];
    s.pright = hadron->parton_mom[0];
    s.excited = true;
    s.time_of_creation = hadron->time_of_creation;
    s.position = hadron->position;
    out[0] = s;
    *n_out = 1;
    return;
  }

  const int pdg_start_q = (start < 0) ? -start : start;
  const int pdg_end_q = (end < 0) ? -end : end;
  (void)pdg_start_q;
  (void)pdg_end_q;

  const double wmin =
      is_projectile ? params->proj_min_diff_mass : params->tar_min_diff_mass;
  const double w = hadron->momentum.mag();
  const double w2 = w * w;

  const data::FtfSubType st_start = parton_sub_type(start, refused);
  const data::FtfSubType st_end = parton_sub_type(end, refused);
  const bool both_diquark = (st_start == data::FtfSubType::kDiQuark &&
                             st_end == data::FtfSubType::kDiQuark);
  const bool both_quark =
      (st_start == data::FtfSubType::kQuark && st_end == data::FtfSubType::kQuark);

  if (!(both_diquark || both_quark)) {
    // Kinky strings are allowed only for qq-q strings.
    if (w > wmin) {
      double pt = 0.0;
      if (hadron->status == 0) {
        const double pt2_kink = params->pt2_kink;
        if (pt2_kink != 0.0) {
          pt = std::sqrt(pt2_kink *
                         (data::g4pow_pow_a<double>(w2 / 16.0 / pt2_kink + 1.0, rng.uniform()) -
                          1.0));
        } else {
          pt = 0.0;
        }
      } else {
        pt = 0.0;
      }
      if (pt > 500.0 * units::MeV<double>()) {
        // Unreachable in 11.1.1: Pt2Kink is 0 in the G4FTFParameters constructor, so `pt` is
        // exactly 0 above. A kink makes TWO strings out of one hadron, so building one would
        // lose a parton and the energy with it.
        if (refused != nullptr) { *refused = FtfRefusal::kKinkyStrings; }
        return;
      }
    }
  }

  // Kink is impossible: one string, and the two ends share the hadron's momentum.
  ExcitedString s;
  s.left = end;
  s.right = start;
  s.direction = is_projectile ? +1 : -1;
  s.excited = true;
  s.time_of_creation = hadron->time_of_creation;
  s.position = hadron->position;

  const Vec4 hm = hadron->momentum;
  Vec4 p_start(hm.v.x / 2.0, hm.v.y / 2.0, 0.0, 0.0);  // quark
  Vec4 p_end(hm.v.x / 2.0, hm.v.y / 2.0, 0.0, 0.0);    // di-quark
  const double pz = hm.v.z;
  const double eh = hm.e;
  const double pt2 = hm.v.x * hm.v.x + hm.v.y * hm.v.y;
  const double mt2 = lv_mt2(hm);
  const double exp_term = std::sqrt(pz * pz + (mt2 * mt2 - 4.0 * eh * eh * pt2 / 4.0) / mt2) / 2.0;
  const double pzq = pz / 2.0 - exp_term;
  p_start.v.z = pzq;
  p_start.e = std::sqrt(pzq * pzq + pt2 / 4.0);
  const double pzqq = pz / 2.0 + exp_term;
  p_end.v.z = pzqq;
  p_end.e = std::sqrt(pzqq * pzqq + pt2 / 4.0);

  s.pleft = p_end;    // thePartons.front() is `end`
  s.pright = p_start;
  out[0] = s;
  *n_out = 1;
}

}  // namespace g4gpu::hadronic::ftf
