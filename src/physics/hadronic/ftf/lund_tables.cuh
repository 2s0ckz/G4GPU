// The Lund string-decay tables: G4VLongitudinalStringDecay's constructor and SetMinMasses(),
// and G4LundStringFragmentation's constructor over the top of them.
//
// Transcribed from
//   source/processes/hadronic/models/parton_string/hadronization/src/G4VLongitudinalStringDecay.cc
//     G4VLongitudinalStringDecay::G4VLongitudinalStringDecay, SetMinMasses, SetProbCCbar,
//     SetProbBBbar, SetStringTensionParameter
//   .../src/G4LundStringFragmentation.cc
//     G4LundStringFragmentation::G4LundStringFragmentation
//
// WHAT THIS FILE IS. Six tables and about thirty scalars. Five of the six tables are not
// constants - they are COMPUTED by SetMinMasses from the particle table, out of PDG masses and
// the mixing parameters - so this file transcribes the function rather than pasting its
// output. That is deliberate and it is what makes ref/oracle/ftf_lund_tables.csv a check of
// anything: a pasted table agrees with itself (docs/RISK.md V52), while a transcribed
// SetMinMasses agrees with Geant4 only if data/ftf_hadrons.hh, the mixing vectors and the
// index arithmetic are all right at once.
//
// THREE INDEX MISMATCHES IN SetMinMasses, TRANSCRIBED AS WRITTEN. See docs/RISK.md V85. In
// short: the d-dbar meson row writes Meson[0][0][2] = 221 (eta) but its weight into
// MesonWeight[0][0][3], and Meson[0][0][3] = 331 (eta') but its weight into
// MesonWeight[0][0][4], which the omega line then overwrites. So from a d-dbar string the eta
// has weight 0, the eta' carries the eta's weight, and the row sums to 0.875 instead of 1. The
// u-ubar row two blocks below is written with the same five lines and consistent indices, so
// the two light-quark rows are not the same table. And the c-cbar / b-bbar lines multiply
// Meson[3][3][*] and Meson[4][4][*] - the PDG CODES - by a probability ratio instead of
// MesonWeight, turning code 441 into the integer 147. Neither is reachable in QBBC's default
// configuration (Prob_QQbar[3] = Prob_QQbar[4] = 0 gates the c and b columns of every
// SplitLast weight), but the d-dbar row is reached on every light string, and it is measured.
//
// THE ONE NUMBER NO RUN CAN BE ASKED FOR. G4LundStringFragmentation::Tmt = 190 MeV is a
// private member with no getter and no setter, so ref/dump/dump_ftf.cc cannot dump it the way
// it dumps MassCut or SigmaQT. It is carried here as the literal from the constructor and is
// checked only through the Pt spectrum of ftf_fragment.csv, where SplitEandP samples
// `HadronMass - TmtCur*log(u)` - which is sensitive to it at the first draw. Said out loud
// because "transcribed and hoped" is exactly what P3's Weisskopf case and P6's HETC case were.
#pragma once
#include <cmath>

#include "core/units.cuh"
#include "data/ftf_hadrons.hh"
#include "physics/hadronic/ftf/refusal.cuh"

namespace g4gpu::hadronic::ftf {

/// CLHEP's fermi, derived the way CLHEP derives it (`1.e-15*meter`) rather than pasted as
/// 1e-12 mm - the doctrine core/units.cuh's `barn()` comment sets out, for the same reason.
template <typename T> __host__ __device__ constexpr T fermi() {
  return T(1.e-15) * units::m<T>();
}
/// CLHEP's millibarn, `1.e-3*barn`.
template <typename T> __host__ __device__ constexpr T millibarn() {
  return T(1.e-3) * units::barn<T>();
}

/// The tables SetMinMasses builds plus the scalars the two constructors set. ~12 kB of
/// doubles and ints, which is why it is a struct passed by pointer and never a local: a
/// kernel that put this on the stack would spill before it fragmented anything.
template <typename real_t>
struct LundTables {
  // ---- SetMinMasses' six tables ----
  real_t min_mass_qqbar[5][5];      ///< minMassQQbarStr
  real_t min_mass_qdiq[5][5][5];    ///< minMassQDiQStr
  int meson[5][5][7];               ///< Meson
  real_t meson_weight[5][5][7];     ///< MesonWeight
  int baryon[5][5][5][4];           ///< Baryon
  real_t baryon_weight[5][5][5][4]; ///< BaryonWeight
  int qcharge[5];                   ///< Qcharge, in units of e/3
  real_t prob_qqbar[5];             ///< Prob_QQbar

  // ---- G4VLongitudinalStringDecay's constructor, as G4LundStringFragmentation leaves it ----
  real_t mass_cut;             ///< MassCut, 210 MeV (Mpi + Delta)
  real_t sigma_qt;             ///< SigmaQT, 0.435 GeV after the Lund constructor
  real_t strange_suppress;     ///< StrangeSuppress, (1 - 0.12)/2
  real_t diquark_suppress;     ///< DiquarkSuppress, 0.07
  real_t diquark_break_prob;   ///< DiquarkBreakProb, 0.3 after the Lund constructor
  int string_loop_interrupt;   ///< 1000
  int cluster_loop_interrupt;  ///< 500
  real_t pspin_meson[3];       ///< probability of a pseudo-scalar meson, by strangeness count
  real_t pspin_barion;         ///< probability of a spin-1/2 baryon
  real_t vector_meson_mix[6];
  real_t scalar_meson_mix[6];
  real_t prob_ccbar;
  real_t prob_bbbar;
  real_t prob_cb;              ///< ProbCCbar + ProbBBbar, the SampleQuarkFlavor gate
  real_t prob_eta_c;
  real_t prob_eta_b;
  real_t max_mass;             ///< MaxMass, -350 GeV - the "no such particle" sentinel
  real_t kappa;                ///< string tension, 1 GeV/fermi
  real_t tmt;                  ///< G4LundStringFragmentation::Tmt, 190 MeV - see file header
  real_t mass_of_light_quark;
  real_t mass_of_s_quark;
  real_t mass_of_c_quark;
  real_t mass_of_b_quark;
  real_t mass_of_string_junction;
};

/// G4VLongitudinalStringDecay::G4VLongitudinalStringDecay followed by
/// G4LundStringFragmentation::G4LundStringFragmentation, in that order, because the derived
/// constructor overwrites four of the base's values and the order is observable: SigmaQT is
/// 0.5 GeV in the base and 0.435 in the derived one, StrangeSuppress 0.44 then (1-0.12)/2 =
/// 0.44 exactly (the same number written two ways, and NOT the same double - 0.44 is
/// 0.44000000000000000222 and (1.0-0.12)/2.0 is 0.44000000000000000222 as well here, but the
/// oracle carries whichever the install ends with), DiquarkBreakProb 0.1 then 0.3, and MassCut
/// 210 MeV in both.
///
/// @param enable_bc_particles `G4HadronicParameters::EnableBCParticles()`, which is 1 in
///        11.1.1 and is what switches ProbCCbar and ProbBBbar from 0 to 2e-4 and 5e-5. Passed
///        rather than hard-coded because it is a run-time flag, and because a port that
///        hard-codes it cannot report the case where it is off.
template <typename real_t>
__host__ __device__ inline void lund_set_scalars(LundTables<real_t>* t, bool enable_bc_particles) {
  // --- G4VLongitudinalStringDecay::G4VLongitudinalStringDecay ---
  t->mass_cut = real_t(210.0) * units::MeV<real_t>();
  t->string_loop_interrupt = 1000;
  t->cluster_loop_interrupt = 500;
  t->sigma_qt = real_t(0.5) * units::GeV<real_t>();
  t->strange_suppress = real_t(0.44);
  t->diquark_suppress = real_t(0.07);
  t->diquark_break_prob = real_t(0.1);

  t->pspin_meson[0] = real_t(0.5);  // u or d + anti-u or anti-d
  t->pspin_meson[1] = real_t(0.4);  // one quark strange, charm or bottom
  t->pspin_meson[2] = real_t(0.3);  // both quarks strange, charm or bottom
  t->pspin_barion = real_t(0.5);

  t->vector_meson_mix[0] = real_t(0.0);
  t->vector_meson_mix[1] = real_t(0.5);
  t->vector_meson_mix[2] = real_t(0.0);
  t->vector_meson_mix[3] = real_t(0.5);
  t->vector_meson_mix[4] = real_t(1.0);
  t->vector_meson_mix[5] = real_t(1.0);

  t->scalar_meson_mix[0] = real_t(0.5);
  t->scalar_meson_mix[1] = real_t(0.25);
  t->scalar_meson_mix[2] = real_t(0.5);
  t->scalar_meson_mix[3] = real_t(0.25);
  t->scalar_meson_mix[4] = real_t(1.0);
  t->scalar_meson_mix[5] = real_t(0.5);

  t->prob_ccbar = real_t(0.0);
  t->prob_eta_c = real_t(0.1);
  t->prob_bbbar = real_t(0.0);
  t->prob_eta_b = real_t(0.0);
  t->prob_cb = t->prob_ccbar + t->prob_bbbar;

  t->max_mass = real_t(-350.0) * units::GeV<real_t>();
  // SetStringTensionParameter(aValue) is `Kappa = aValue * GeV/fermi`, and the base
  // constructor calls `Kappa = 1.0 * GeV/fermi` DIRECTLY rather than through the setter -
  // while G4LundStringFragmentation's constructor calls `SetStringTensionParameter(1.*GeV/fermi)`,
  // i.e. the setter with an argument that is already GeV/fermi, so the multiplication happens
  // twice and Kappa ends at (GeV/fermi)^2. That is not a transcription slip here; it is what
  // the install does, and ref/oracle/ftf_lund_params.csv carries the value it ends with.
  t->kappa = real_t(1.0) * units::GeV<real_t>() / fermi<real_t>();

  t->mass_of_light_quark = real_t(140.0) * units::MeV<real_t>();
  t->mass_of_s_quark = real_t(500.0) * units::MeV<real_t>();
  t->mass_of_c_quark = real_t(1600.0) * units::MeV<real_t>();
  t->mass_of_b_quark = real_t(4500.0) * units::MeV<real_t>();
  t->mass_of_string_junction = real_t(720.0) * units::MeV<real_t>();

  // --- G4LundStringFragmentation::G4LundStringFragmentation ---
  t->mass_cut = real_t(210.0) * units::MeV<real_t>();
  t->sigma_qt = real_t(0.435) * units::GeV<real_t>();
  t->tmt = real_t(190.0) * units::MeV<real_t>();
  t->kappa = real_t(1.0) * units::GeV<real_t>() / fermi<real_t>() * units::GeV<real_t>() /
             fermi<real_t>();
  t->diquark_break_prob = real_t(0.3);
  t->strange_suppress = (real_t(1.0) - real_t(0.12)) / real_t(2.0);
  t->diquark_suppress = real_t(0.07);
  if (enable_bc_particles) {
    t->prob_ccbar = real_t(0.0002);
    t->prob_bbbar = real_t(5.0e-5);
  } else {
    t->prob_ccbar = real_t(0.0);
    t->prob_bbbar = real_t(0.0);
  }
  t->prob_cb = t->prob_ccbar + t->prob_bbbar;
}

/// G4VLongitudinalStringDecay::SetMinMasses, transcribed line for line.
///
/// It runs twice in Geant4 - once from the base constructor and once from
/// G4LundStringFragmentation's, after the mixing parameters have been changed - and the second
/// run is the one whose output survives. Only the second is reproduced here, because
/// lund_set_scalars above has already applied both constructors; the first run's output is
/// overwritten entry for entry (every table is either zeroed at the top of the function or
/// assigned unconditionally), which is checked by the fact that the oracle's tables are the
/// tables this produces.
template <typename real_t>
__host__ __device__ inline void lund_set_min_masses(LundTables<real_t>* t) {
  using data::ftf_find_hadron;
  const real_t max_mass = t->max_mass;

  // ---- minimal mass of q-qbar strings ----
  //
  // Geant4 leaves minMassQQbarStr[i][j] UNWRITTEN when FindParticle(100*i+11) is null, and the
  // array is a bare member of a heap object, so the entry would be indeterminate. In 11.1.1 no
  // entry is skipped - the five codes 111, 211, 311, 411, 511 and the five 111..511 all exist -
  // so the loop below initialises the array to max_mass first and the oracle then shows that
  // nothing kept it. Initialising is not an approximation of Geant4's behaviour; reading
  // uninitialised memory has no behaviour to approximate.
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) { t->min_mass_qqbar[i][j] = max_mass; }
  }
  for (int i = 1; i < 6; ++i) {
    const int code1 = 100 * i + 10 * 1 + 1;
    const data::FtfHadron* h1 = ftf_find_hadron(code1);
    if (h1 != nullptr) {
      for (int j = 1; j < 6; ++j) {
        const int code2 = 100 * j + 10 * 1 + 1;
        const data::FtfHadron* h2 = ftf_find_hadron(code2);
        if (h2 != nullptr) {
          t->min_mass_qqbar[i - 1][j - 1] = static_cast<real_t>(h1->mass) +
                                            static_cast<real_t>(h2->mass) +
                                            real_t(70.0) * units::MeV<real_t>();
        }
      }
    }
  }
  // "u-ubar = 0.5 Pi0 + 0.24 Eta + 0.25 Eta'" - Geant4's comment for this one assignment.
  t->min_mass_qqbar[1][1] = t->min_mass_qqbar[0][0];

  // ---- minimal mass of qq-q strings ----
  for (int i = 1; i < 6; ++i) {
    const int code1 = 100 * i + 10 * 1 + 1;
    const data::FtfHadron* h1 = ftf_find_hadron(code1);
    for (int j = 1; j < 6; ++j) {
      for (int k = 1; k < 6; ++k) {
        const int kfla = (j > k) ? j : k;
        const int kflb = (j < k) ? j : k;
        // Add a d quark - except for (1,1), where a u quark is added instead.
        int code2 = 1000 * kfla + 100 * kflb + 10 * 1 + 2;
        if (j == 1 && k == 1) { code2 = 1000 * 2 + 100 * 1 + 10 * 1 + 2; }

        const data::FtfHadron* h2 = ftf_find_hadron(code2);
        const data::FtfHadron* h3 = ftf_find_hadron(code2 + 2);

        if (h2 == nullptr && h3 == nullptr) {
          t->min_mass_qdiq[i - 1][j - 1][k - 1] = max_mass;
          continue;
        }
        if (h2 != nullptr && h3 != nullptr) {
          if (h2->mass > h3->mass) { h2 = h3; }
        }
        if (h2 == nullptr && h3 != nullptr) { h2 = h3; }

        // h1 is dereferenced unconditionally here in Geant4 - there is no null test on
        // `hadron1` in this loop, unlike the q-qbar loop above. All five codes exist in
        // 11.1.1; a missing one would be a null dereference there and is a refusal here,
        // which the caller sees as max_mass in the table and a report at the entry point.
        if (h1 == nullptr) {
          t->min_mass_qdiq[i - 1][j - 1][k - 1] = max_mass;
          continue;
        }
        t->min_mass_qdiq[i - 1][j - 1][k - 1] = static_cast<real_t>(h1->mass) +
                                                static_cast<real_t>(h2->mass) +
                                                real_t(70.0) * units::MeV<real_t>();
      }
    }
  }

  // q charges, in units of e/3:  d  u  s  c  b
  t->qcharge[0] = -1;
  t->qcharge[1] = 2;
  t->qcharge[2] = -1;
  t->qcharge[3] = 2;
  t->qcharge[4] = -1;

  // ---- the meson table for a small string's last two-body decay ----
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) {
      for (int k = 0; k < 7; ++k) {
        t->meson[i][j][k] = 0;
        t->meson_weight[i][j][k] = real_t(0.0);
      }
    }
  }
  {
    int strange_q = 0;
    int strange_aq = 0;
    for (int i = 0; i < 5; ++i) {
      if (i >= 2) { strange_q = 1; }
      for (int j = 0; j < 5; ++j) {
        strange_aq = 0;
        if (j >= 2) { strange_aq = 1; }
        const int mx = (i > j) ? i : j;
        const int mn = (i < j) ? i : j;
        t->meson[i][j][0] = 100 * (mx + 1) + 10 * (mn + 1) + 1;  // scalar
        t->meson_weight[i][j][0] = t->pspin_meson[strange_q + strange_aq];
        t->meson[i][j][1] = 100 * (mx + 1) + 10 * (mn + 1) + 3;  // vector
        t->meson_weight[i][j][1] = real_t(1.0) - t->pspin_meson[strange_q + strange_aq];
      }
    }
  }

  // d-dbar. The weights of the eta and the eta' land one index high - see the file header and
  // docs/RISK.md V85. Transcribed with Geant4's indices, not with the indices the comments in
  // G4VLongitudinalStringDecay.cc describe.
  t->meson[0][0][0] = 111;
  t->meson_weight[0][0][0] = t->pspin_meson[0] * t->scalar_meson_mix[0];  // Pi0
  t->meson[0][0][2] = 221;
  t->meson_weight[0][0][3] =
      t->pspin_meson[0] * (real_t(1) - t->scalar_meson_mix[0] - t->scalar_meson_mix[1]);  // Eta
  t->meson[0][0][3] = 331;
  t->meson_weight[0][0][4] = t->pspin_meson[0] * t->scalar_meson_mix[1];  // Eta'

  t->meson[0][0][1] = 113;
  t->meson_weight[0][0][1] =
      (real_t(1.0) - t->pspin_meson[0]) * (real_t(1) - t->vector_meson_mix[1]);  // Rho
  t->meson[0][0][4] = 223;
  t->meson_weight[0][0][4] =
      (real_t(1.0) - t->pspin_meson[0]) * t->vector_meson_mix[1];  // omega

  // u-ubar. Same five lines, consistent indices.
  t->meson[1][1][0] = 111;
  t->meson_weight[1][1][0] = t->pspin_meson[0] * t->scalar_meson_mix[0];
  t->meson[1][1][2] = 221;
  t->meson_weight[1][1][2] =
      t->pspin_meson[0] * (real_t(1) - t->scalar_meson_mix[0] - t->scalar_meson_mix[1]);
  t->meson[1][1][3] = 331;
  t->meson_weight[1][1][3] = t->pspin_meson[0] * t->scalar_meson_mix[1];

  t->meson[1][1][1] = 113;
  t->meson_weight[1][1][1] =
      (real_t(1.0) - t->pspin_meson[0]) * (real_t(1) - t->vector_meson_mix[1]);
  t->meson[1][1][4] = 223;
  t->meson_weight[1][1][4] = (real_t(1.0) - t->pspin_meson[0]) * t->vector_meson_mix[1];

  // s-sbar
  t->meson[2][2][0] = 221;
  t->meson_weight[2][2][0] = t->pspin_meson[2] * (real_t(1) - t->scalar_meson_mix[5]);
  t->meson[2][2][2] = 331;
  t->meson_weight[2][2][2] = t->pspin_meson[2] * t->scalar_meson_mix[5];
  t->meson[2][2][1] = 333;
  t->meson_weight[2][2][1] = (real_t(1.0) - t->pspin_meson[2]) * t->vector_meson_mix[5];

  // c-cbar and b-bbar: Geant4 multiplies the PDG CODE, not the weight. 441 * (0.1/0.3) is the
  // integer 147 and 551 * (0.0/0.3) is 0, neither of which is a particle. Transcribed as
  // written; unreachable while Prob_QQbar[3] and Prob_QQbar[4] are zero, which is every QBBC
  // run, and reported through kUnknownHadronCode if it ever is reached.
  if (t->pspin_meson[2] != real_t(0.0)) {
    t->meson[3][3][0] = static_cast<int>(static_cast<real_t>(t->meson[3][3][0]) *
                                         (t->prob_eta_c) / (t->pspin_meson[2]));
    t->meson[3][3][1] = static_cast<int>(static_cast<real_t>(t->meson[3][3][1]) *
                                         (real_t(1.0) - t->prob_eta_c) /
                                         (real_t(1.) - t->pspin_meson[2]));
    t->meson[4][4][0] = static_cast<int>(static_cast<real_t>(t->meson[4][4][0]) *
                                         (t->prob_eta_b) / (t->pspin_meson[2]));
    t->meson[4][4][1] = static_cast<int>(static_cast<real_t>(t->meson[4][4][1]) *
                                         (real_t(1.0) - t->prob_eta_b) /
                                         (real_t(1.) - t->pspin_meson[2]));
  }

  // ---- the baryon table ----
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) {
      for (int k = 0; k < 5; ++k) {
        for (int l = 0; l < 4; ++l) {
          t->baryon[i][j][k][l] = 0;
          t->baryon_weight[i][j][k][l] = real_t(0.0);
        }
      }
    }
  }
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) {
      for (int k = 0; k < 5; ++k) {
        const int kfla = i + 1, kflb = j + 1, kflc = k + 1;
        int kfld = (kfla > kflb) ? kfla : kflb;
        kfld = (kfld > kflc) ? kfld : kflc;
        int kflf = (kfla < kflb) ? kfla : kflb;
        kflf = (kflf < kflc) ? kflf : kflc;
        const int kfle = kfla + kflb + kflc - kfld - kflf;

        t->baryon[i][j][k][0] = 1000 * kfld + 100 * kfle + 10 * kflf + 2;  // spin 1/2
        t->baryon_weight[i][j][k][0] = t->pspin_barion;
        t->baryon[i][j][k][1] = 1000 * kfld + 100 * kfle + 10 * kflf + 4;  // spin 3/2
        t->baryon_weight[i][j][k][1] = real_t(1.0) - t->pspin_barion;
      }
    }
  }

  // The five same-flavour triples have only the spin-3/2 state.
  t->baryon[0][0][0][0] = 1114;  t->baryon_weight[0][0][0][0] = real_t(1.0);  // Delta-
  t->baryon[0][0][0][1] = 0;     t->baryon_weight[0][0][0][1] = real_t(0.0);
  t->baryon[1][1][1][0] = 2224;  t->baryon_weight[1][1][1][0] = real_t(1.0);  // Delta++
  t->baryon[1][1][1][1] = 0;     t->baryon_weight[1][1][1][1] = real_t(0.0);
  t->baryon[2][2][2][0] = 3334;  t->baryon_weight[2][2][2][0] = real_t(1.0);  // Omega-
  t->baryon[2][2][2][1] = 0;     t->baryon_weight[2][2][2][1] = real_t(0.0);
  t->baryon[3][3][3][0] = 4444;  t->baryon_weight[3][3][3][0] = real_t(1.0);  // Omega_cc++
  t->baryon[3][3][3][1] = 0;     t->baryon_weight[3][3][3][1] = real_t(0.0);
  t->baryon[4][4][4][0] = 5554;  t->baryon_weight[4][4][4][0] = real_t(1.0);  // Omega_bb-
  t->baryon[4][4][4][1] = 0;     t->baryon_weight[4][4][4][1] = real_t(0.0);

  // The six flavour permutations of each mixed triple: the spin-1/2 slot becomes the
  // Lambda-like state at HALF its previous weight (`*= 0.5`, so pspin_barion/2), and slot 2 -
  // which the loop above never wrote - becomes the Sigma-like state at 0.5*pspin_barion. Slot
  // 3 is never written for any triple, which is why the Meson table's seven slots and the
  // Baryon table's four are both larger than anything that fills them.
  struct Triple { int a, b, c; int lambda_like, sigma_like; };
  const Triple triples[] = {
      {0, 1, 2, 3122, 3212},  // sud: Lambda / Sigma0
      {0, 1, 3, 4122, 4212},  // cud: Lambda_c+ / Sigma_c+
      {1, 2, 3, 4232, 4322},  // cus: Xi_c+ / Xi_c+'
      {0, 2, 3, 4132, 4312},  // cds: Xi_c0 / Xi_c0'
      {0, 1, 4, 5122, 5212},  // bud: Lambda_b0 / Sigma_b0
      {1, 2, 4, 5232, 5322},  // bus: Xi_b0 / Xi_b0'
      {0, 2, 4, 5132, 5312},  // bds: Xi_b- / Xi_b-'
  };
  for (const Triple& tr : triples) {
    const int perm[6][3] = {{tr.a, tr.b, tr.c}, {tr.a, tr.c, tr.b}, {tr.b, tr.a, tr.c},
                            {tr.b, tr.c, tr.a}, {tr.c, tr.a, tr.b}, {tr.c, tr.b, tr.a}};
    for (const auto& p : perm) {
      t->baryon[p[0]][p[1]][p[2]][0] = tr.lambda_like;
      t->baryon_weight[p[0]][p[1]][p[2]][0] *= real_t(0.5);
      t->baryon[p[0]][p[1]][p[2]][2] = tr.sigma_like;
      t->baryon_weight[p[0]][p[1]][p[2]][2] = real_t(0.5) * t->pspin_barion;
    }
  }

  // Geant4's final pass: any Baryon code with no particle behind it becomes 0, so that the
  // do-while loops in SplitLast that terminate on `Baryon[...] != 0` stop at it. Note that
  // the same pass is NOT applied to the Meson table, which is how the 147 above survives.
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) {
      for (int k = 0; k < 5; ++k) {
        for (int l = 0; l < 4; ++l) {
          if (t->baryon[i][j][k][l] != 0 &&
              ftf_find_hadron(t->baryon[i][j][k][l]) == nullptr) {
            t->baryon[i][j][k][l] = 0;
          }
        }
      }
    }
  }

  // Probabilities of q-qbar pair production for a kink or a gluon. The c and b entries are
  // zero here whatever EnableBCParticles says - they are the gate that keeps the two broken
  // Meson rows above unreachable, and they are NOT the ProbCCbar / ProbBBbar that
  // SampleQuarkFlavor uses.
  const real_t prob_uubar = real_t(0.33);
  t->prob_qqbar[0] = prob_uubar;
  t->prob_qqbar[1] = prob_uubar;
  t->prob_qqbar[2] = real_t(1.0) - real_t(2.) * prob_uubar;
  t->prob_qqbar[3] = real_t(0.0);
  t->prob_qqbar[4] = real_t(0.0);
}

/// The two constructors and SetMinMasses, in Geant4's order.
template <typename real_t>
__host__ __device__ inline void lund_init(LundTables<real_t>* t, bool enable_bc_particles) {
  lund_set_scalars(t, enable_bc_particles);
  lund_set_min_masses(t);
}

// ---------------------------------------------------------------------------------------------
// G4VLondgitudinalStringDecay::SetMinimalStringMass, and the predicate under it.
// ---------------------------------------------------------------------------------------------

/// Whether a (left, right) parton pair is one SetMinimalStringMass will accept.
///
/// The two conditions are Geant4's own, written as one predicate so that a test can address
/// it: partons of the SAME sub-type (q-qbar or qq-qqbar) must have opposite-sign codes, and
/// partons of DIFFERENT sub-types (q-qq or qbar-qqbar) must have same-sign codes. Anything
/// else throws "Illegal quark content as input" there and is kIllegalPartonPair here.
__host__ __device__ inline bool ftf_parton_pair_is_legal(int left_code, int right_code) {
  const bool l_di = (left_code > 1000) || (left_code < -1000);
  const bool r_di = (right_code > 1000) || (right_code < -1000);
  const long long product = static_cast<long long>(left_code) * right_code;
  if (l_di == r_di) { return product < 0; }
  return product > 0;
}

/// What SetMinimalStringMass leaves behind: the minimal mass and its square.
template <typename real_t> struct MinimalStringMass {
  real_t mass = 0;
  real_t mass2 = 0;
  FtfRefusal refused = FtfRefusal::kNone;
};

/// G4VLongitudinalStringDecay::SetMinimalStringMass.
///
/// @param string_mass  the string's own invariant mass, which only the DiQuark-AntiDiQuark arm
///                     reads (to decide whether two baryons fit). Passed explicitly because
///                     that arm is the one place a *table* lookup depends on the kinematics.
template <typename real_t>
__host__ __device__ inline MinimalStringMass<real_t> ftf_minimal_string_mass(
    const LundTables<real_t>* t, int left_code, int right_code, real_t string_mass) {
  MinimalStringMass<real_t> out;
  if (!ftf_parton_pair_is_legal(left_code, right_code)) {
    out.refused = FtfRefusal::kIllegalPartonPair;
    return out;
  }
  const int qleft = (left_code < 0) ? -left_code : left_code;
  const int qright = (right_code < 0) ? -right_code : right_code;

  auto finish = [&out](real_t m) {
    out.mass = m;
    out.mass2 = m * m;
    return out;
  };

  if (qleft < 6 && qright < 6) {  // Q-Qbar
    return finish(t->min_mass_qqbar[qleft - 1][qright - 1]);
  }
  if (qleft < 6 && qright > 1000) {  // Q - DiQ; can be negative
    const int q1 = qright / 1000;
    const int q2 = (qright / 100) % 10;
    return finish(t->min_mass_qdiq[qleft - 1][q1 - 1][q2 - 1]);
  }
  if (qleft > 1000 && qright < 6) {  // DiQ - Q
    const int q1 = qleft / 1000;
    const int q2 = (qleft / 100) % 10;
    return finish(t->min_mass_qdiq[qright - 1][q1 - 1][q2 - 1]);
  }

  // DiQuark - AntiDiQuark. Note that the two `if (qleft < 6 && qright > 1000)` tests above
  // leave a gap: a parton code between 6 and 1000 - which no diquark or quark code is -
  // reaches here and indexes min_mass_qdiq with q1 = 0, i.e. [-1]. Geant4 has the same gap and
  // no code can land in it, because CreatePartonPair builds only |code| <= 5 and
  // (max*1000 + min*100 + spin).
  const int q1 = qleft / 1000;
  const int q2 = (qleft / 100) % 10;
  const int q3 = qright / 1000;
  const int q4 = (qright / 100) % 10;

  const real_t m1 = t->min_mass_qdiq[q1 - 1][q2 - 1][0];
  const real_t m2 = t->min_mass_qdiq[q3 - 1][q4 - 1][0];
  // A negative entry means "no such particle" (max_mass = -350 GeV).
  if (m1 > real_t(0.) && m2 > real_t(0.)) {
    const real_t est = m1 + m2;
    if (string_mass > est) {  // two baryons fit
      out.mass = m1 + m2;
      out.mass2 = est * est;  // Geant4 squares `EstimatedMass`, which is the same sum
      return out;
    }
  }
  if (m1 < real_t(0.) && m2 > real_t(0.)) { return finish(t->max_mass); }
  if (m1 > real_t(0.) && m2 < real_t(0.)) { return finish(m1); }

  // Re-arrangement into two mesons.
  const real_t a = t->min_mass_qqbar[q1 - 1][q3 - 1] + t->min_mass_qqbar[q2 - 1][q4 - 1];
  const real_t b = t->min_mass_qqbar[q1 - 1][q4 - 1] + t->min_mass_qqbar[q2 - 1][q3 - 1];
  return finish((a < b) ? a : b);
}

/// G4FTFParameters::GetMinMass - the minimal diffractive-dissociation mass of a hadron, read
/// out of the two tables above by the hadron's own PDG code. `Qleft` and `Qright` are
/// `max(pdg/100, 1)` and `max((pdg/10)%10, 1)`, which for a baryon gives (thousands+hundreds,
/// tens) - so a proton, 2212, gives Qleft = 22 and Qright = 1, taking the DiQ-Q arm with the
/// diquark read out of Qleft = 22: q1 = 2, q2 = 2. The clamps to [1,5] are Geant4's, added to
/// stop the array indices going out of range, and the comment above them says so.
template <typename real_t>
__host__ __device__ inline real_t ftf_get_min_mass(const LundTables<real_t>* t, int pdg) {
  const int part_id = (pdg < 0) ? -pdg : pdg;
  const int qleft_raw = part_id / 100;
  const int qright_raw = (part_id / 10) % 10;
  const int qleft = (qleft_raw > 1) ? qleft_raw : 1;
  const int qright = (qright_raw > 1) ? qright_raw : 1;
  real_t estimated = real_t(0.0);
  auto clamp15 = [](int v) {
    const int lo = (v < 5) ? v : 5;
    return (lo > 1) ? lo : 1;
  };
  if (qleft < 6 && qright < 6) {
    estimated = t->min_mass_qqbar[qleft - 1][qright - 1];
  } else if (qleft < 6 && qright > 6) {
    const int q1 = clamp15(qright / 10);
    const int q2 = clamp15(qright % 10);
    estimated = t->min_mass_qdiq[qleft - 1][q1 - 1][q2 - 1];
  } else if (qleft > 6 && qright < 6) {
    const int q1 = clamp15(qleft / 10);
    const int q2 = clamp15(qleft % 10);
    estimated = t->min_mass_qdiq[qright - 1][q1 - 1][q2 - 1];
  }
  return estimated;
}

}  // namespace g4gpu::hadronic::ftf
