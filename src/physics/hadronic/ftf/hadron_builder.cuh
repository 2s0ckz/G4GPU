// G4HadronBuilder: a quark pair or a diquark-quark pair becomes a hadron.
//
// Transcribed from
//   source/processes/hadronic/models/parton_string/hadronization/src/G4HadronBuilder.cc
//     G4HadronBuilder::Build, BuildLowSpin, BuildHighSpin, Meson, Barion
//   .../include/G4HadronBuilder.hh   (the Spin enum, whose VALUES are the PDG spin digit)
//
// This is where the meson and baryon mixing tables are actually spent. `SetMinMasses`' tables
// (lund_tables.cuh) are used only by SplitLast's final two-body decay; every hadron the
// fragmentation LOOP produces comes through here, so these ~40 lines are most of the species
// composition of an FTFP event.
//
// THE SPIN ENUM IS ARITHMETIC, NOT A TAG. `SpinZero = 1, SpinHalf = 2, SpinOne = 3,
// SpinThreeHalf = 4`, and the value is added to the PDG code as its last digit: a
// pseudo-scalar meson is `110*n + 1`, a vector meson `110*n + 3`, a spin-1/2 baryon
// `...  + 2`, a spin-3/2 baryon `... + 4`. So the enum cannot be renumbered and is written out
// here with its values.
//
// WHAT `BuildLowSpin` DOES FOR A DIQUARK, which its own comment admits: "will return a
// SpinThreeHalf Barion if all quarks the same". Barion() overrides the requested spin with
// SpinThreeHalf whenever the three flavours are equal, because there is no spin-1/2 uuu. That
// is why `PossibleHadronMass`, which calls BuildLowSpin to estimate a string's minimum mass,
// can come back with a Delta.
//
// THE HEAVY-FLAVOUR SUBSTITUTIONS ARE REACHABLE AND ARE PORTED. `EnableBCParticles` is 1 in
// 11.1.1 (ref/oracle/ftf_lund_params.csv), so G4LundStringFragmentation sets ProbCCbar = 2e-4
// and ProbBBbar = 5e-5 and SampleQuarkFlavor draws a c or a b from the vacuum at that rate on
// every split. The 21 charmed-meson, 40 bottom-meson, 16 charmed-baryon and 25 bottom-baryon
// substitutions below are therefore reached by an ordinary proton beam, roughly once in four
// thousand quark pairs, and they are not refused. Geant4's own comment explains them: there
// are no excited charmed or bottom hadrons in Geant4, so the builder maps them onto ground
// states, "prefer[ring] to conserve the electric charge rather than other flavor numbers" -
// which means a few of them violate charm or bottom number, and the ones that do are marked.
//
// THE ONE CASE THAT THROWS. Meson() throws "Illegal Quark content as input" for |id| > 5 and
// Barion() for |id1| < 1000 or |id2| > 5. Build() dispatches a diquark-diquark pair to
// Barion(), which then throws on `|id2| > 5`. A kernel cannot throw, so it is refused by name
// (kHadronBuilderIllegalContent). No caller in FTFP produces that pair: Splitup dispatches on
// DecayIsQuark and both DiQuarkSplitup arms build from a quark pair.
#pragma once
#include "data/ftf_hadrons.hh"
#include "physics/hadronic/ftf/lund_tables.cuh"
#include "physics/hadronic/ftf/refusal.cuh"

namespace g4gpu::hadronic::ftf {

/// G4HadronBuilder::Spin. The values ARE the PDG code's last digit - see the file header.
enum class Spin : int { kSpinZero = 1, kSpinHalf = 2, kSpinOne = 3, kSpinThreeHalf = 4 };

/// What a Build* call returns: the hadron's PDG code, 0 for "no such particle" (Geant4's null
/// G4ParticleDefinition*, which every caller in this package tests for), and a refusal for the
/// two cases Geant4 answers with an exception.
struct BuiltHadron {
  int pdg = 0;
  FtfRefusal refused = FtfRefusal::kNone;
};

/// `G4ParticleTable::FindParticle(code)` reduced to "is there one", used where Geant4 keeps a
/// pointer only to test it against null.
__host__ __device__ inline int ftf_existing_code(int pdg) {
  return (data::ftf_find_hadron(pdg) != nullptr) ? pdg : 0;
}

/// The charmed- and bottom-hadron substitution table of G4HadronBuilder::Meson.
///
/// Geant4 writes it as a 60-arm if/else chain on `std::abs(PDGEncoding)`, sign-preserving for
/// the flavoured mesons and sign-DROPPING for the quarkonia (`PDGEncoding = 441` with no
/// ternary, because a c-cbar state is its own antiparticle). Both shapes are reproduced: the
/// `signed` flag says which.
__host__ __device__ inline int ftf_meson_heavy_substitution(int pdg_encoding) {
  const int a = (pdg_encoding < 0) ? -pdg_encoding : pdg_encoding;
  const int s = (pdg_encoding > 0) ? 1 : -1;
  // Charmed mesons -> D+, D0, Ds+
  if (a == 10411 || a == 413 || a == 10413 || a == 20413 || a == 415) { return s * 411; }
  if (a == 10421 || a == 423 || a == 10423 || a == 20423 || a == 425) { return s * 421; }
  if (a == 10431 || a == 433 || a == 10433 || a == 20433 || a == 435) { return s * 431; }
  // Charmonia -> eta_c or J/psi, unsigned
  if (a == 10441 || a == 100441) { return 441; }
  if (a == 10443 || a == 20443 || a == 100443 || a == 30443 || a == 9000443 ||
      a == 9010443 || a == 9020443 || a == 445 || a == 100445) {
    return 443;
  }
  // Bottom mesons -> B0, B+, Bs0, Bc+
  if (a == 10511 || a == 513 || a == 10513 || a == 20513 || a == 515) { return s * 511; }
  if (a == 10521 || a == 523 || a == 10523 || a == 20523 || a == 525) { return s * 521; }
  if (a == 10531 || a == 533 || a == 10533 || a == 20533 || a == 535) { return s * 531; }
  if (a == 10541 || a == 543 || a == 10543 || a == 20543 || a == 545) { return s * 541; }
  // Bottomonia -> Upsilon, unsigned. Note that 551 (eta_b) is in this list: the builder maps
  // eta_b onto the Upsilon, so `Meson[4][4][0] = 551` in SetMinMasses names a state the
  // builder can never produce.
  if (a == 551 || a == 10551 || a == 100551 || a == 110551 || a == 200551 || a == 210551 ||
      a == 10553 || a == 20553 || a == 30553 || a == 100553 || a == 110553 || a == 120553 ||
      a == 130553 || a == 200553 || a == 210553 || a == 220553 || a == 300553 ||
      a == 9000553 || a == 9010553 || a == 555 || a == 10555 || a == 20555 || a == 100555 ||
      a == 110555 || a == 120555 || a == 200555 || a == 557 || a == 100557) {
    return 553;
  }
  return pdg_encoding;
}

/// The charmed- and bottom-baryon substitution table of G4HadronBuilder::Barion. Every arm is
/// sign-preserving. The ones Geant4 marks as violating charm or bottom number are noted.
__host__ __device__ inline int ftf_barion_heavy_substitution(int pdg_encoding) {
  const int a = (pdg_encoding < 0) ? -pdg_encoding : pdg_encoding;
  const int s = (pdg_encoding > 0) ? 1 : -1;
  // Charmed baryons
  if (a == 4224) { return s * 4222; }
  if (a == 4214) { return s * 4212; }
  if (a == 4114) { return s * 4112; }
  if (a == 4322 || a == 4324) { return s * 4232; }
  if (a == 4312 || a == 4314) { return s * 4132; }
  if (a == 4334) { return s * 4332; }
  if (a == 4412 || a == 4414 || a == 4432 || a == 4434) { return s * 4232; }  // charm -1
  if (a == 4422 || a == 4424) { return s * 4222; }                           // charm -1
  if (a == 4444) { return s * 4222; }                                        // charm -2
  // Bottom baryons
  if (a == 5114) { return s * 5112; }
  if (a == 5214) { return s * 5212; }
  if (a == 5224) { return s * 5222; }
  if (a == 5312 || a == 5314) { return s * 5132; }
  if (a == 5322 || a == 5324) { return s * 5232; }
  if (a == 5334) { return s * 5332; }
  if (a == 5142 || a == 5412 || a == 5414) { return s * 5232; }  // charm -1
  if (a == 5242 || a == 5422 || a == 5424) { return s * 5222; }  // charm -1
  if (a == 5342 || a == 5432 || a == 5434) { return s * 5232; }  // charm -1
  if (a == 5442 || a == 5444) { return s * 5222; }               // charm -2
  if (a == 5512 || a == 5514) { return s * 5132; }               // bottom -1
  if (a == 5522 || a == 5524) { return s * 5232; }               // bottom -1
  if (a == 5532 || a == 5534) { return s * 5332; }               // bottom -1
  if (a == 5542 || a == 5544) { return s * 5232; }               // charm -1, bottom -1
  if (a == 5554) { return s * 5332; }                            // bottom -2
  return pdg_encoding;
}

/// G4HadronBuilder::Meson.
///
/// @tparam Rng anything with `double uniform()`.
template <typename real_t, typename Rng>
__host__ __device__ inline BuiltHadron ftf_build_meson(const LundTables<real_t>* t, int black,
                                                       int white, Spin the_spin, Rng& rng) {
  BuiltHadron out;
  int id1 = black;
  int id2 = white;
  const int a1 = (id1 < 0) ? -id1 : id1;
  const int a2 = (id2 < 0) ? -id2 : id2;
  if (a1 < a2) {
    const int xchg = id1;
    id1 = id2;
    id2 = xchg;
  }
  const int abs_id1 = (id1 < 0) ? -id1 : id1;
  if (abs_id1 > 5) {
    out.refused = FtfRefusal::kHadronBuilderIllegalContent;
    return out;
  }

  int pdg_encoding = 0;
  if (id1 + id2 == 0) {
    if (abs_id1 < 4) {  // light quarks: u, d or s
      const real_t rmix = static_cast<real_t>(rng.uniform());
      const int imix = 2 * abs_id1 - 1;
      // `(G4int)(rmix + mix)` is a TRUNCATION of a positive sum, so it is 1 exactly when
      // rmix + mix >= 1. The two mixings are read at imix-1 and imix, i.e. a sliding pair of
      // the six-element vector - which is why scalarMesonMix has six entries for three
      // light-quark pairs.
      if (the_spin == Spin::kSpinZero) {
        pdg_encoding =
            110 * (1 + static_cast<int>(rmix + t->scalar_meson_mix[imix - 1]) +
                   static_cast<int>(rmix + t->scalar_meson_mix[imix])) +
            static_cast<int>(the_spin);
      } else {
        pdg_encoding =
            110 * (1 + static_cast<int>(rmix + t->vector_meson_mix[imix - 1]) +
                   static_cast<int>(rmix + t->vector_meson_mix[imix])) +
            static_cast<int>(the_spin);
      }
    } else {  // c-cbar or b-bbar
      pdg_encoding = abs_id1 * 100 + abs_id1 * 10;
      if (pdg_encoding == 440) {
        pdg_encoding += (static_cast<real_t>(rng.uniform()) < t->prob_eta_c) ? 1 : 3;
      }
      if (pdg_encoding == 550) {
        pdg_encoding += (static_cast<real_t>(rng.uniform()) < t->prob_eta_b) ? 1 : 3;
      }
    }
  } else {
    pdg_encoding = 100 * abs_id1 + 10 * ((id2 < 0) ? -id2 : id2) + static_cast<int>(the_spin);
    const bool is_up = ((abs_id1 & 1) == 0);  // quark 1 is an up-type quark (u or c)
    const bool is_anti = (id1 < 0);
    if ((is_up && is_anti) || (!is_up && !is_anti)) { pdg_encoding = -pdg_encoding; }
  }

  pdg_encoding = ftf_meson_heavy_substitution(pdg_encoding);
  out.pdg = ftf_existing_code(pdg_encoding);
  return out;
}

/// G4HadronBuilder::Barion.
template <typename real_t, typename Rng>
__host__ __device__ inline BuiltHadron ftf_build_barion(const LundTables<real_t>* t, int black,
                                                        int white, Spin the_spin, Rng& rng) {
  BuiltHadron out;
  int id1 = black;
  int id2 = white;
  {
    const int a1 = (id1 < 0) ? -id1 : id1;
    const int a2 = (id2 < 0) ? -id2 : id2;
    if (a1 < a2) {
      const int xchg = id1;
      id1 = id2;
      id2 = xchg;
    }
  }
  const int abs1 = (id1 < 0) ? -id1 : id1;
  const int abs2 = (id2 < 0) ? -id2 : id2;
  if (abs1 < 1000 || abs2 > 5) {
    out.refused = FtfRefusal::kHadronBuilderIllegalContent;
    return out;
  }

  int ifl1 = abs1 / 1000;
  int ifl2 = (abs1 - ifl1 * 1000) / 100;
  const int diquark_spin = abs1 % 10;
  const int ifl3 = id2;
  if (id1 < 0) {
    ifl1 = -ifl1;
    ifl2 = -ifl2;
  }
  const int kfla = (ifl1 < 0) ? -ifl1 : ifl1;
  const int kflb = (ifl2 < 0) ? -ifl2 : ifl2;
  const int kflc = (ifl3 < 0) ? -ifl3 : ifl3;

  int kfld = (kfla > kflb) ? kfla : kflb;
  kfld = (kfld > kflc) ? kfld : kflc;
  int kflf = (kfla < kflb) ? kfla : kflb;
  kflf = (kflf < kflc) ? kflf : kflc;
  const int kfle = kfla + kflb + kflc - kfld - kflf;

  // uuu, ddd or sss is always spin 3/2 - there is no spin-1/2 state of three identical
  // flavours. This overrides the caller's request, including BuildLowSpin's.
  Spin spin = (kfla == kflb && kflb == kflc) ? Spin::kSpinThreeHalf : the_spin;

  int kfll = 0;
  if (kfld < 6) {
    if (spin == Spin::kSpinHalf && kfld > kfle && kfle > kflf) {
      // Spin 1/2 with three different flavours: two states exist, Lambda-like (the two
      // lighter quarks reversed in the code) and Sigma-like.
      if (diquark_spin == 1) {
        if (kfla == kfld) {  // the heaviest quark is in the diquark
          kfll = 1;
        } else {
          kfll = static_cast<int>(real_t(0.25) + static_cast<real_t>(rng.uniform()));
        }
      }
      if (diquark_spin == 3 && kfla != kfld) {
        kfll = static_cast<int>(real_t(0.75) + static_cast<real_t>(rng.uniform()));
      }
    }
  }

  int pdg_encoding;
  if (kfll == 1) {
    pdg_encoding = 1000 * kfld + 100 * kflf + 10 * kfle + static_cast<int>(spin);
  } else {
    pdg_encoding = 1000 * kfld + 100 * kfle + 10 * kflf + static_cast<int>(spin);
  }
  if (id1 < 0) { pdg_encoding = -pdg_encoding; }

  pdg_encoding = ftf_barion_heavy_substitution(pdg_encoding);
  out.pdg = ftf_existing_code(pdg_encoding);
  return out;
}

/// G4HadronBuilder::Build - one draw for the spin, then Meson or Barion.
template <typename real_t, typename Rng>
__host__ __device__ inline BuiltHadron ftf_hadron_build(const LundTables<real_t>* t, int black,
                                                        int white, Rng& rng) {
  const data::FtfHadron* b = data::ftf_find_hadron(black);
  const data::FtfHadron* w = data::ftf_find_hadron(white);
  if (b == nullptr || w == nullptr) {
    BuiltHadron out;
    out.refused = FtfRefusal::kUnknownHadronCode;
    return out;
  }
  if (b->subtype == data::FtfSubType::kDiQuark || w->subtype == data::FtfSubType::kDiQuark) {
    const Spin spin = (static_cast<real_t>(rng.uniform()) < t->pspin_barion)
                          ? Spin::kSpinHalf
                          : Spin::kSpinThreeHalf;
    return ftf_build_barion(t, black, white, spin, rng);
  }
  // `>= 3` counts the s, c and b quarks, so a c-cbar pair lands on mesonSpinMix[2] just as an
  // s-sbar pair does.
  int strange_q = 0;
  if (((black < 0) ? -black : black) >= 3) { ++strange_q; }
  if (((white < 0) ? -white : white) >= 3) { ++strange_q; }
  const Spin spin = (static_cast<real_t>(rng.uniform()) < t->pspin_meson[strange_q])
                        ? Spin::kSpinZero
                        : Spin::kSpinOne;
  return ftf_build_meson(t, black, white, spin, rng);
}

/// G4HadronBuilder::BuildLowSpin - no spin draw at all.
template <typename real_t, typename Rng>
__host__ __device__ inline BuiltHadron ftf_hadron_build_low_spin(const LundTables<real_t>* t,
                                                                 int black, int white,
                                                                 Rng& rng) {
  const data::FtfHadron* b = data::ftf_find_hadron(black);
  const data::FtfHadron* w = data::ftf_find_hadron(white);
  if (b == nullptr || w == nullptr) {
    BuiltHadron out;
    out.refused = FtfRefusal::kUnknownHadronCode;
    return out;
  }
  if (b->subtype == data::FtfSubType::kQuark && w->subtype == data::FtfSubType::kQuark) {
    return ftf_build_meson(t, black, white, Spin::kSpinZero, rng);
  }
  return ftf_build_barion(t, black, white, Spin::kSpinHalf, rng);
}

/// G4HadronBuilder::BuildHighSpin.
template <typename real_t, typename Rng>
__host__ __device__ inline BuiltHadron ftf_hadron_build_high_spin(const LundTables<real_t>* t,
                                                                  int black, int white,
                                                                  Rng& rng) {
  const data::FtfHadron* b = data::ftf_find_hadron(black);
  const data::FtfHadron* w = data::ftf_find_hadron(white);
  if (b == nullptr || w == nullptr) {
    BuiltHadron out;
    out.refused = FtfRefusal::kUnknownHadronCode;
    return out;
  }
  if (b->subtype == data::FtfSubType::kQuark && w->subtype == data::FtfSubType::kQuark) {
    return ftf_build_meson(t, black, white, Spin::kSpinOne, rng);
  }
  return ftf_build_barion(t, black, white, Spin::kSpinThreeHalf, rng);
}

}  // namespace g4gpu::hadronic::ftf
