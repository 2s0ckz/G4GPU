// G4FTFAnnihilation: an anti-baryon and a nucleon annihilate into one, two or three strings.
//
// Transcribed from
//   source/processes/hadronic/models/parton_string/diffraction/src/G4FTFAnnihilation.cc
//     Annihilate, Create3QuarkAntiQuarkStrings, Create1DiquarkAntiDiquarkString,
//     Create2QuarkAntiQuarkStrings, Create1QuarkAntiQuarkString, GaussianPt, UnpackBaryon,
//     ChooseX
//
// ## Why an anti-nucleon beam needs this and no other beam does
//
// `G4FTFParameters::GetProbabilityOfAnnihilation()` is `Xannihilation / (Xtotal - Xelastic)`
// and `Xannihilation` is non-zero only in InitForInteraction's anti-baryon branch. So
// G4FTFModel::ExciteParticipants' third arm is unreachable for p, n, pi+-, K+- and every ion,
// and is taken on a large fraction of collisions for an anti-nucleon - which QBBC gives FTFP at
// EVERY energy through `G4HadronicBuilder::BuildFTFP_BERT(..., bert=false)`.
//
// ## The four channels, and the cross sections that choose between them
//
//   a  `X_a` - the "3-shirt diagram": three anti-quark-quark strings, one of which is an
//      ADDITIONAL splitable hadron that did not exist before. 625.1 mb at rest.
//   b  `X_b` - one quark-antiquark pair annihilates and the rest form a diquark-antidiquark
//      string. Exactly 0 mb at rest, and `3.13 + 140*(threshold - sqrt(s))^2.5` below the
//      two-pion threshold in flight - the one place in FTF that uses `G4Pow::powA` with a
//      non-integer exponent.
//   c  `X_c` - two quark-antiquark strings. 49.989 mb at rest.
//   d  `X_d` - one quark-antiquark string; the target nucleon is GONE (status 4). 6.614 mb.
//
// Each is then multiplied by a weight that depends on WHICH anti-baryon meets WHICH nucleon -
// the same nine-by-two table `ftf_annihilation_weights` carries in ftf_parameters.cuh, written
// out again here because Geant4 writes it out again here, with `X_a` NOT scaled.
//
// ## Every SetFirstParton is two deviates
//
// `SetFirstParton`/`SetSecondParton` each `delete` a parton and `new` another, and a
// `G4Parton` constructor draws a colour and (for a non-zero `GetPDGiSpin()`) a spin
// projection - docs/RISK.md V98. Channel (a) calls them six times and also constructs an
// `AdditionalString` with the DEFAULT `G4DiffractiveSplitableHadron` constructor, which makes a
// d and an anti-d before they are replaced: 4 more. None of those values is ever read. The
// draw-count column of ref/oracle/ftf_annih.csv is the only thing that can check them, and the
// order they happen in is the order below.
//
// ## The rapidity ordering in channel (a) has a transcribed bug
//
// `Ystring` is computed inside `if ( Pstring.e() > 1.0e-30 )` and then inside
// `if ( Pstring.e() + Pstring.pz() < 1.0e-30 )` - so the `else` that assigns
// `Ystring = Pstring.rapidity()` is reached only when the numerator is small AND the
// denominator is not. For an ordinary string, with E + pz well above 1e-30, NOTHING assigns
// Ystring and it stays 0.0. All three strings therefore have rapidity 0 and the "keep ordering
// in rapidity" block below sorts on ties. Reproduced exactly: the ordering it produces
// (`QuarkOrder`) decides which parton pair goes to which string, so a port that computed the
// rapidity properly would build different strings from the same quarks.
#pragma once
#include <cmath>

#include "core/units.cuh"
#include "data/ftf_hadrons.hh"
#include "data/g4pow.hh"
#include "physics/hadronic/ftf/diffractive_excitation.cuh"
#include "physics/hadronic/ftf/ftf_parameters.cuh"
#include "physics/hadronic/ftf/lorentz.cuh"
#include "physics/hadronic/ftf/refusal.cuh"
#include "physics/hadronic/ftf/splitable_hadron.cuh"

namespace g4gpu::hadronic::ftf {

/// G4FTFAnnihilation::CommonVariables.
struct AnnihCommon {
  int aq[3] = {0, 0, 0};
  int q[3] = {0, 0, 0};
  bool rotate_strings = false;
  double s = 0.0, sqrt_s = 0.0;
  Vec4 p_projectile, p_target;
  LorentzRot to_lab, random_rotation;
  FtfRefusal refused = FtfRefusal::kNone;
};

/// G4FTFAnnihilation::UnpackBaryon - the ABSOLUTE code is divided and the signs are restored
/// afterwards, which is NOT what G4DiffractiveExcitation::UnpackBaryon does (that one divides
/// the signed code). The two give the same answer; only this one is written so that it
/// obviously does.
__host__ __device__ inline void ftf_annih_unpack_baryon(int id_pdg, int* q1, int* q2, int* q3) {
  const int abs_id = (id_pdg < 0) ? -id_pdg : id_pdg;
  *q1 = abs_id / 1000;
  *q2 = (abs_id % 1000) / 100;
  *q3 = (abs_id % 100) / 10;
  if (id_pdg < 0) {
    *q1 = -*q1;
    *q2 = -*q2;
    *q3 = -*q3;
  }
}

/// `G4DiffractiveSplitableHadron::SetFirstParton` / `SetSecondParton`, with the deviates the
/// replacement `G4Parton` spends (docs/RISK.md V98) and nothing else - the parton's colour and
/// spin are never read.
template <typename Rng>
__host__ __device__ inline void ftf_set_first_parton(SplitableHadron* h, int pdg,
                                                     FtfRefusal* refused, Rng& rng) {
  h->parton[0] = pdg;
  ftf_parton_construct_draws(pdg, refused, rng);
}
template <typename Rng>
__host__ __device__ inline void ftf_set_second_parton(SplitableHadron* h, int pdg,
                                                      FtfRefusal* refused, Rng& rng) {
  h->parton[1] = pdg;
  ftf_parton_construct_draws(pdg, refused, rng);
}

/// `new G4DiffractiveSplitableHadron()` - the DEFAULT constructor, which is the only place in
/// FTF that uses it. It pre-fills the two partons with a d (1) and an anti-d (-1), spending
/// four deviates on two objects that channel (a) then replaces.
template <typename Rng>
__host__ __device__ inline SplitableHadron ftf_default_splitable(FtfRefusal* refused, Rng& rng) {
  SplitableHadron h;
  h.pdg = 0;
  h.parton[0] = 1;
  h.parton[1] = -1;
  h.parton_index = -1;
  h.alive = true;
  ftf_parton_construct_draws(1, refused, rng);
  ftf_parton_construct_draws(-1, refused, rng);
  return h;
}

/// `G4RandFlat::shootInt(n)` = `long(flat()*double(n))` (CLHEP RandFlat.icc:57).
template <typename Rng>
__host__ __device__ inline int ftf_shoot_int(int n, Rng& rng) {
  return static_cast<int>(static_cast<long>(rng.uniform() * static_cast<double>(n)));
}

/// The anti-quark-quark pair's meson code, which all four channels build the same way and each
/// writes out separately. `aKsi` is ONE deviate and is spent even when the flavours differ and
/// it is not used - which is why it is drawn before the branch here too.
__host__ __device__ inline int ftf_annih_meson_code(int anti_quark, int quark, double a_ksi) {
  const int abs_aq = (anti_quark < 0) ? -anti_quark : anti_quark;
  const int abs_q = (quark < 0) ? -quark : quark;
  int new_code = 0;
  if (abs_aq == abs_q) {
    if (abs_aq != 3) {
      new_code = 111;  // Pi0
      if (a_ksi < 0.5) {
        new_code = 221;  // Eta
        if (a_ksi < 0.25) { new_code = 331; }  // Eta'
      }
    } else {
      new_code = 221;
      if (a_ksi < 0.5) { new_code = 331; }
    }
  } else {
    // The sign is carried by `absX/X`, which is +1 or -1 - integer division of two ints of the
    // same magnitude, so it is exact and not a rounding.
    if (abs_aq > abs_q) {
      new_code = abs_aq * 100 + abs_q * 10 + 1;
      new_code *= abs_aq / anti_quark;
    } else {
      new_code = abs_q * 100 + abs_aq * 10 + 1;
      new_code *= abs_q / quark;
    }
  }
  return new_code;
}

/// G4FTFAnnihilation::GaussianPt - the same eight lines as G4DiffractiveExcitation's, without
/// the `Pt2 > 0` guard on the square root.
template <typename Rng>
__host__ __device__ inline Vec3d ftf_annih_gaussian_pt(double average_pt2, double max_pt_square,
                                                       Rng& rng) {
  return ftf_gaussian_pt_excitation(average_pt2, max_pt_square, rng);
}

/// The rapidity as `Create3QuarkAntiQuarkStrings` and `Create2QuarkAntiQuarkStrings` compute it.
///
/// Transcribed WITH its nesting, which is the bug the file header describes: `rapidity()` is
/// reached only when `E + pz` is below 1e-30 and `E - pz` is not, so for every string a real
/// collision produces this returns 0.0 and the ordering below it sorts on ties.
__host__ __device__ inline double ftf_annih_ystring(const Vec4& p) {
  double ystring = 0.0;
  if (p.e > 1.0e-30) {
    if (p.e + p.v.z < 1.0e-30) {
      ystring = -1.0e30;
      if (p.e - p.v.z < 1.0e-30) {
        ystring = 1.0e30;
      } else {
        ystring = lv_rapidity(p);
      }
    }
  }
  return ystring;
}

/// G4FTFAnnihilation::Create3QuarkAntiQuarkStrings - channel (a), the "3-shirt diagram".
///
/// `additional` receives the third string's splitable hadron, which did not exist before;
/// `*made_additional` says whether it was written. Geant4 allocates it inside the `iString == 2`
/// arm of the loop, AFTER that string's `aKsi` deviate and BEFORE its FindParticle.
template <typename Rng>
__host__ __device__ inline bool ftf_annih_create_3q_aq_strings(SplitableHadron* projectile,
                                                               SplitableHadron* target,
                                                               SplitableHadron* additional,
                                                               bool* made_additional,
                                                               FtfParameters<double>* params,
                                                               AnnihCommon* c, Rng& rng) {
  const int max_number_of_loops = 1000;
  const double mass_q2 = 0.0;  // "Simplest case is considered with Mass_Q = 0.0"
  double quark_xs[6] = {0, 0, 0, 0, 0, 0};
  Vec3d quark_mom[6];

  const double alfa_r = 0.5;
  double average_pt2 = 200.0 * 200.0;
  const double max_pt_square = c->s;
  double scale_factor = 1.0;
  double alfa = 0.0, beta = 0.0;

  int number_of_tries = 0, loop_counter = 0;
  do {
    double x1 = 0.0, x2 = 0.0, x3 = 0.0;
    double product = 1.0;
    for (int i_case = 0; i_case < 2; ++i_case) {  // anti-baryon, then baryon
      const double r1 = rng.uniform();
      const double r2 = rng.uniform();
      if (alfa_r == 1.0) {
        // Dead: alfa_r is a local initialised to 0.5 and never written. Transcribed because
        // the `else` below is only meaningful next to it.
        x1 = 1.0 - std::sqrt(r1);
        x2 = (1.0 - x1) * r2;
      } else {
        x1 = r1 * r1;
        // `(1-x1) * sqr(sin(...))` and NOT `(1-x1) * sin(...) * sin(...)`: C++ associates the
        // second left to right, so it multiplies `(1-x1)` by the sine first and rounds in a
        // different place. `sqr` is a macro over one expression, so the square is formed first.
        const double sn = std::sin(units::pi<double>() / 2.0 * r2);
        x2 = (1.0 - x1) * (sn * sn);
      }
      x3 = 1.0 - x1 - x2;
      const int index = i_case * 3;
      quark_xs[index] = x1;
      quark_xs[index + 1] = x2;
      quark_xs[index + 2] = x3;
      product *= (x1 * x2 * x3);
    }

    if (product == 0.0) { continue; }

    ++number_of_tries;
    if (number_of_tries == 100 * (number_of_tries / 100)) {
      scale_factor /= 2.0;
      average_pt2 *= scale_factor;
    }

    Vec3d pt_sum{0.0, 0.0, 0.0};
    for (int i = 0; i < 6; ++i) {
      quark_mom[i] = ftf_annih_gaussian_pt(average_pt2, max_pt_square, rng);
      pt_sum = pt_sum + quark_mom[i];
    }
    // `PtSum /= 6.0` is Hep3Vector::operator/=, i.e. `*= 1.0/6.0`.
    const double inv6 = 1.0 / 6.0;
    pt_sum = Vec3d{pt_sum.x * inv6, pt_sum.y * inv6, pt_sum.z * inv6};

    alfa = 0.0;
    beta = 0.0;
    for (int i = 0; i < 6; ++i) {
      quark_mom[i] = quark_mom[i] - pt_sum;
      const double val = (g4gpu::mag2(quark_mom[i]) + mass_q2) / quark_xs[i];
      if (i < 3) { alfa += val; } else { beta += val; }
    }
  } while ((std::sqrt(alfa) + std::sqrt(beta) > c->sqrt_s) &&
           ++loop_counter < max_number_of_loops);

  if (loop_counter >= max_number_of_loops) { return false; }

  const double decay_momentum2 = c->s * c->s + alfa * alfa + beta * beta -
                                 2.0 * (c->s * (alfa + beta) + alfa * beta);
  const double wminus_target =
      (c->s - alfa + beta + std::sqrt(decay_momentum2)) / 2.0 / c->sqrt_s;
  const double wplus_projectile = c->sqrt_s - beta / wminus_target;

  for (int i_case = 0; i_case < 2; ++i_case) {
    const int index = i_case * 3;
    const double w = (i_case == 1) ? -wminus_target : wplus_projectile;
    for (int i = 0; i < 3; ++i) {
      const double pz = w * quark_xs[index + i] / 2.0 -
                        (g4gpu::mag2(quark_mom[index + i]) + mass_q2) /
                            (2.0 * w * quark_xs[index + i]);
      quark_mom[index + i].z = pz;
    }
  }

  // Sampling of the anti-quark order in the projectile. One deviate, six permutations, and
  // case 0 is the identity - so a sixth of the draws change nothing.
  const int sampled_case = ftf_shoot_int(6, rng);
  int tmp1 = 0, tmp2 = 0;
  switch (sampled_case) {
    case 1: tmp1 = c->aq[1]; c->aq[1] = c->aq[2]; c->aq[2] = tmp1; break;
    case 2: tmp1 = c->aq[0]; c->aq[0] = c->aq[1]; c->aq[1] = tmp1; break;
    case 3: tmp1 = c->aq[0]; tmp2 = c->aq[1]; c->aq[0] = c->aq[2];
            c->aq[1] = tmp1; c->aq[2] = tmp2; break;
    case 4: tmp1 = c->aq[0]; tmp2 = c->aq[1]; c->aq[0] = tmp2;
            c->aq[1] = c->aq[2]; c->aq[2] = tmp1; break;
    case 5: tmp1 = c->aq[0]; tmp2 = c->aq[1]; c->aq[0] = c->aq[2];
            c->aq[1] = tmp2; c->aq[2] = tmp1; break;
    default: break;
  }

  *made_additional = false;
  int anti_quark = 0, quark = 0;
  for (int i_string = 0; i_string < 3; ++i_string) {
    if (i_string == 0) {
      anti_quark = c->aq[0];
      quark = c->q[0];
      ftf_set_first_parton(projectile, anti_quark, &c->refused, rng);
      ftf_set_second_parton(projectile, quark, &c->refused, rng);
      projectile->status = 0;
    } else if (i_string == 1) {
      quark = c->q[1];
      anti_quark = c->aq[1];
      ftf_set_first_parton(target, quark, &c->refused, rng);
      ftf_set_second_parton(target, anti_quark, &c->refused, rng);
      target->status = 0;
    } else {
      anti_quark = c->aq[2];
      quark = c->q[2];
    }
    const double a_ksi = rng.uniform();
    const int new_code = ftf_annih_meson_code(anti_quark, quark, a_ksi);
    if (i_string == 2) {
      *additional = ftf_default_splitable(&c->refused, rng);
      *made_additional = true;
    }
    const data::FtfHadron* test = data::ftf_find_hadron(new_code);
    if (test == nullptr) { return false; }
    if (i_string == 0) {
      projectile->pdg = new_code;
      params->proj_min_diff_mass = 0.5 * units::GeV<double>();
      params->proj_min_non_diff_mass = 0.5 * units::GeV<double>();
    } else if (i_string == 1) {
      target->pdg = new_code;
      params->tar_min_diff_mass = 0.5 * units::GeV<double>();
      params->tar_min_non_diff_mass = 0.5 * units::GeV<double>();
    } else {
      additional->pdg = new_code;
      ftf_set_first_parton(additional, c->aq[2], &c->refused, rng);
      ftf_set_second_parton(additional, c->q[2], &c->refused, rng);
      additional->status = 0;
    }
  }

  // The three strings' four-momenta, ordered by a rapidity that is always 0 (see the header).
  Vec4 pstring1, pstring2, pstring3;
  int quark_order[3] = {0, 0, 0};
  double ystring_max = 0.0, ystring_min = 0.0;
  for (int i = 0; i < 3; ++i) {
    const Vec3d tmp = quark_mom[i] + quark_mom[i + 3];
    const Vec4 pstring(tmp, std::sqrt(g4gpu::mag2(quark_mom[i]) + mass_q2) +
                                std::sqrt(g4gpu::mag2(quark_mom[i + 3]) + mass_q2));
    const double ystring = ftf_annih_ystring(pstring);
    if (i == 0) {
      pstring1 = pstring;
      ystring_max = ystring;
      quark_order[0] = 0;
    } else if (i == 1) {
      if (ystring > ystring_max) {
        pstring2 = pstring1;
        ystring_min = ystring_max;
        pstring1 = pstring;
        ystring_max = ystring;
        quark_order[0] = 1;
        quark_order[1] = 0;
      } else {
        pstring2 = pstring;
        ystring_min = ystring;
        quark_order[1] = 1;
      }
    } else {
      if (ystring > ystring_max) {
        pstring3 = pstring2;
        pstring2 = pstring1;
        pstring1 = pstring;
        // NOTE the order: QuarkOrder[1] is written from QuarkOrder[0] and then QuarkOrder[2]
        // from the JUST-WRITTEN QuarkOrder[1], so both end up holding the old QuarkOrder[0].
        // Transcribed as written.
        quark_order[1] = quark_order[0];
        quark_order[2] = quark_order[1];
        quark_order[0] = 2;
      } else if (ystring > ystring_min) {
        pstring3 = pstring2;
        pstring2 = pstring;
      } else {
        pstring3 = pstring;
        quark_order[2] = 2;
      }
    }
  }

  Vec4 quark_4mom[6];
  for (int i = 0; i < 6; ++i) {
    quark_4mom[i] = Vec4(quark_mom[i], std::sqrt(g4gpu::mag2(quark_mom[i]) + mass_q2));
    if (c->rotate_strings) { quark_4mom[i] = lorentz_apply(c->random_rotation, quark_4mom[i]); }
    quark_4mom[i] = lorentz_apply(c->to_lab, quark_4mom[i]);
  }

  // `Splitting()` then two GetNext calls, which cycle the index: AntiParton is Parton[0] and
  // Parton is Parton[1]. This port keeps the parton MOMENTA on the string rather than on the
  // splitable hadron (splitable_hadron.cuh's header), so they are handed to the caller.
  projectile->is_split = true;
  target->is_split = true;
  if (*made_additional) { additional->is_split = true; }

  c->p_projectile = pstring1;
  c->p_target = pstring3;
  Vec4 left_string = pstring2;

  if (c->rotate_strings) {
    c->p_projectile = lorentz_apply(c->random_rotation, c->p_projectile);
    c->p_target = lorentz_apply(c->random_rotation, c->p_target);
    left_string = lorentz_apply(c->random_rotation, left_string);
  }
  c->p_projectile = lorentz_apply(c->to_lab, c->p_projectile);
  c->p_target = lorentz_apply(c->to_lab, c->p_target);
  left_string = lorentz_apply(c->to_lab, left_string);

  projectile->time_of_creation = target->time_of_creation;
  projectile->position = target->position;
  if (*made_additional) {
    additional->time_of_creation = target->time_of_creation;
    additional->position = target->position;
  }

  projectile->momentum = c->p_projectile;
  if (*made_additional) { additional->momentum = left_string; }
  target->momentum = c->p_target;

  projectile->collision_count += 1;
  if (*made_additional) { additional->collision_count += 1; }
  target->collision_count += 1;

  // The parton four-momenta, in the order Geant4 assigns them.
  projectile->parton_mom[0] = quark_4mom[quark_order[0]];
  projectile->parton_mom[1] = quark_4mom[quark_order[0] + 3];
  target->parton_mom[0] = quark_4mom[quark_order[2]];
  target->parton_mom[1] = quark_4mom[quark_order[2] + 3];
  if (*made_additional) {
    additional->parton_mom[0] = quark_4mom[quark_order[1]];
    additional->parton_mom[1] = quark_4mom[quark_order[1] + 3];
  }
  return true;
}

/// The candidate anti-quark/quark pairs that can annihilate, and what is left over. Shared by
/// channels (b) and (c), which build the same table twice in Geant4.
__host__ __device__ inline int ftf_annih_candidates(const AnnihCommon* c, int cand_aq[9][2],
                                                    int cand_q[9][2]) {
  int n = 0;
  for (int i_aq = 0; i_aq < 3; ++i_aq) {
    for (int i_q = 0; i_q < 3; ++i_q) {
      if (-c->aq[i_aq] == c->q[i_q]) {
        if (i_aq == 0) { cand_aq[n][0] = 1; cand_aq[n][1] = 2; }
        if (i_aq == 1) { cand_aq[n][0] = 0; cand_aq[n][1] = 2; }
        if (i_aq == 2) { cand_aq[n][0] = 0; cand_aq[n][1] = 1; }
        if (i_q == 0) { cand_q[n][0] = 1; cand_q[n][1] = 2; }
        if (i_q == 1) { cand_q[n][0] = 0; cand_q[n][1] = 2; }
        if (i_q == 2) { cand_q[n][0] = 0; cand_q[n][1] = 1; }
        ++n;
      }
    }
  }
  return n;
}

/// G4FTFAnnihilation::Create1DiquarkAntiDiquarkString - channel (b).
///
/// Returns 0 (done), 1 (continue to the next channel) or 99 (failed). Note that the "continue"
/// answer is what happens when NO anti-quark-quark pair can annihilate, and Geant4's own
/// comment says the string it would have made is not implemented: "If we allow the string to
/// interact with other nuclear nucleons, we have to set up MinDiffrMass in Parameters, and
/// ascribe a PDGEncoding. To be done yet!"
template <typename Rng>
__host__ __device__ inline int ftf_annih_create_1dq_adq_string(SplitableHadron* projectile,
                                                               SplitableHadron* target,
                                                               AnnihCommon* c, Rng& rng) {
  int cand_aq[9][2] = {};
  int cand_q[9][2] = {};
  const int candidats_n = ftf_annih_candidates(c, cand_aq, cand_q);
  if (candidats_n == 0) { return 1; }

  const int sampled_case = ftf_shoot_int(candidats_n, rng);
  const int left_aq1 = c->aq[cand_aq[sampled_case][0]];
  const int left_aq2 = c->aq[cand_aq[sampled_case][1]];
  const int left_q1 = c->q[cand_q[sampled_case][0]];
  const int left_q2 = c->q[cand_q[sampled_case][1]];

  // The last digit is 3 for both - "for simplicity, only 3 is considered" - and the
  // anti-diquark's is built by SUBTRACTING 3 from a negative code, which is how it comes out as
  // -xy03 rather than -xy01.
  const int abs_aq1 = (left_aq1 < 0) ? -left_aq1 : left_aq1;
  const int abs_aq2 = (left_aq2 < 0) ? -left_aq2 : left_aq2;
  const int abs_q1 = (left_q1 < 0) ? -left_q1 : left_q1;
  const int abs_q2 = (left_q2 < 0) ? -left_q2 : left_q2;
  const int anti_dq = (abs_aq1 > abs_aq2) ? (1000 * left_aq1 + 100 * left_aq2 - 3)
                                          : (1000 * left_aq2 + 100 * left_aq1 - 3);
  const int dq = (abs_q1 > abs_q2) ? (1000 * left_q1 + 100 * left_q2 + 3)
                                   : (1000 * left_q2 + 100 * left_q1 + 3);

  ftf_set_first_parton(projectile, dq, &c->refused, rng);
  ftf_set_second_parton(projectile, anti_dq, &c->refused, rng);

  // "It is assumed that quark and di-quark masses are 0."
  Vec4 pquark(0.0, 0.0, -c->sqrt_s / 2.0, c->sqrt_s / 2.0);
  Vec4 paquark(0.0, 0.0, c->sqrt_s / 2.0, c->sqrt_s / 2.0);
  if (c->rotate_strings) {
    pquark = lorentz_apply(c->random_rotation, pquark);
    paquark = lorentz_apply(c->random_rotation, paquark);
  }
  pquark = lorentz_apply(c->to_lab, pquark);
  paquark = lorentz_apply(c->to_lab, paquark);
  projectile->parton_mom[0] = pquark;
  projectile->parton_mom[1] = paquark;
  projectile->is_split = true;

  projectile->status = 0;
  target->status = 4;  // The target nucleon has annihilated; the comment says "3->4"
  c->p_projectile = Vec4(0.0, 0.0, 0.0, c->sqrt_s);
  c->p_projectile = lorentz_apply(c->to_lab, c->p_projectile);

  projectile->time_of_creation = target->time_of_creation;
  projectile->position = target->position;
  projectile->momentum = c->p_projectile;

  projectile->collision_count += 1;
  target->collision_count += 1;
  return 0;
}

/// G4FTFAnnihilation::Create2QuarkAntiQuarkStrings - channel (c).
template <typename Rng>
__host__ __device__ inline int ftf_annih_create_2q_aq_strings(SplitableHadron* projectile,
                                                              SplitableHadron* target,
                                                              FtfParameters<double>* params,
                                                              AnnihCommon* c, Rng& rng) {
  Vec3d quark_mom[4];
  double quark_xs[4] = {0, 0, 0, 0};
  double average_pt2 = 200.0 * 200.0;
  const double max_pt_square = c->s;
  const double mass_q2 = 0.0;
  double scale_factor = 1.0;
  int number_of_tries = 0, loop_counter = 0;
  const int max_number_of_loops = 1000;
  double alfa = 0.0, beta = 0.0;
  const double alfa_r = 0.5;

  do {
    double product = 1.0;
    for (int i_case = 0; i_case < 2; ++i_case) {
      const double r = rng.uniform();
      double x = 0.0;
      if (alfa_r == 1.0) {
        x = (i_case == 0) ? std::sqrt(r) : (1.0 - std::sqrt(r));
      } else {
        const double sn = std::sin(units::pi<double>() / 2.0 * r);
        x = sn * sn;
      }
      const int index = i_case * 2;
      quark_xs[index] = x;
      quark_xs[index + 1] = 1.0 - x;
      product *= x * (1.0 - x);
    }
    if (product == 0.0) { continue; }

    ++number_of_tries;
    if (number_of_tries == 100 * (number_of_tries / 100)) {
      scale_factor /= 2.0;
      average_pt2 *= scale_factor;
    }

    Vec3d pt_sum{0.0, 0.0, 0.0};
    for (int i = 0; i < 4; ++i) {
      quark_mom[i] = ftf_annih_gaussian_pt(average_pt2, max_pt_square, rng);
      pt_sum = pt_sum + quark_mom[i];
    }
    const double inv4 = 1.0 / 4.0;
    pt_sum = Vec3d{pt_sum.x * inv4, pt_sum.y * inv4, pt_sum.z * inv4};
    for (int i = 0; i < 4; ++i) { quark_mom[i] = quark_mom[i] - pt_sum; }

    alfa = 0.0;
    beta = 0.0;
    for (int i_case = 0; i_case < 2; ++i_case) {
      const int index = i_case * 2;
      for (int i = 0; i < 2; ++i) {
        const double val = (g4gpu::mag2(quark_mom[index + i]) + mass_q2) / quark_xs[index + i];
        if (i_case == 0) { alfa += val; } else { beta += val; }
      }
    }
  } while ((std::sqrt(alfa) + std::sqrt(beta) > c->sqrt_s) &&
           ++loop_counter < max_number_of_loops);

  if (loop_counter >= max_number_of_loops) { return 99; }

  const double decay_momentum2 = c->s * c->s + alfa * alfa + beta * beta -
                                 2.0 * (c->s * (alfa + beta) + alfa * beta);
  const double wminus_target =
      (c->s - alfa + beta + std::sqrt(decay_momentum2)) / 2.0 / c->sqrt_s;
  const double wplus_projectile = c->sqrt_s - beta / wminus_target;

  for (int i_case = 0; i_case < 2; ++i_case) {
    const int index = i_case * 2;
    for (int i = 0; i < 2; ++i) {
      const double w = (i_case == 1) ? -wminus_target : wplus_projectile;
      const double pz = w * quark_xs[index + i] / 2.0 -
                        (g4gpu::mag2(quark_mom[index + i]) + mass_q2) /
                            (2.0 * w * quark_xs[index + i]);
      quark_mom[index + i].z = pz;
    }
  }

  int cand_aq[9][2] = {};
  int cand_q[9][2] = {};
  const int candidats_n = ftf_annih_candidates(c, cand_aq, cand_q);
  if (candidats_n == 0) { return 1; }

  const int sampled_case = ftf_shoot_int(candidats_n, rng);
  const int left_aq1 = c->aq[cand_aq[sampled_case][0]];
  const int left_aq2 = c->aq[cand_aq[sampled_case][1]];
  int left_q1 = 0, left_q2 = 0;
  if (rng.uniform() < 0.5) {
    left_q1 = c->q[cand_q[sampled_case][0]];
    left_q2 = c->q[cand_q[sampled_case][1]];
  } else {
    left_q2 = c->q[cand_q[sampled_case][0]];
    left_q1 = c->q[cand_q[sampled_case][1]];
  }

  int anti_quark = 0, quark = 0;
  for (int i_string = 0; i_string < 2; ++i_string) {
    if (i_string == 0) {
      anti_quark = left_aq1;
      quark = left_q1;
      ftf_set_first_parton(projectile, anti_quark, &c->refused, rng);
      ftf_set_second_parton(projectile, quark, &c->refused, rng);
      projectile->status = 0;
    } else {
      quark = left_q2;
      anti_quark = left_aq2;
      ftf_set_first_parton(target, quark, &c->refused, rng);
      ftf_set_second_parton(target, anti_quark, &c->refused, rng);
      target->status = 0;
    }
    const double a_ksi = rng.uniform();
    const int new_code = ftf_annih_meson_code(anti_quark, quark, a_ksi);
    if (data::ftf_find_hadron(new_code) == nullptr) { return 99; }
    if (i_string == 0) {
      projectile->pdg = new_code;
      params->proj_min_diff_mass = 0.5 * units::GeV<double>();
      params->proj_min_non_diff_mass = 0.5 * units::GeV<double>();
    } else {
      target->pdg = new_code;
      params->tar_min_diff_mass = 0.5 * units::GeV<double>();
      params->tar_min_non_diff_mass = 0.5 * units::GeV<double>();
    }
  }

  int quark_order[2] = {0, 0};
  Vec4 pstring1, pstring2;
  double ystring1 = 0.0, ystring2 = 0.0;
  for (int i_case = 0; i_case < 2; ++i_case) {
    const Vec3d tmp = quark_mom[i_case] + quark_mom[i_case + 2];
    const Vec4 pstring(tmp, std::sqrt(g4gpu::mag2(quark_mom[i_case]) + mass_q2) +
                                std::sqrt(g4gpu::mag2(quark_mom[i_case + 2]) + mass_q2));
    const double ystring = ftf_annih_ystring(pstring);
    if (i_case == 0) {
      pstring1 = pstring;
      ystring1 = ystring;
    } else {
      pstring2 = pstring;
      ystring2 = ystring;
    }
  }
  if (ystring1 > ystring2) {
    c->p_projectile = pstring1;
    c->p_target = pstring2;
    quark_order[0] = 0;
    quark_order[1] = 1;
  } else {
    c->p_projectile = pstring2;
    c->p_target = pstring1;
    quark_order[0] = 1;
    quark_order[1] = 0;
  }

  if (c->rotate_strings) {
    c->p_projectile = lorentz_apply(c->random_rotation, c->p_projectile);
    c->p_target = lorentz_apply(c->random_rotation, c->p_target);
  }
  c->p_projectile = lorentz_apply(c->to_lab, c->p_projectile);
  c->p_target = lorentz_apply(c->to_lab, c->p_target);

  Vec4 quark_4mom[4];
  for (int i = 0; i < 4; ++i) {
    quark_4mom[i] = Vec4(quark_mom[i], std::sqrt(g4gpu::mag2(quark_mom[i]) + mass_q2));
    if (c->rotate_strings) { quark_4mom[i] = lorentz_apply(c->random_rotation, quark_4mom[i]); }
    quark_4mom[i] = lorentz_apply(c->to_lab, quark_4mom[i]);
  }

  projectile->is_split = true;
  projectile->parton_mom[0] = quark_4mom[quark_order[0]];
  projectile->parton_mom[1] = quark_4mom[quark_order[0] + 2];
  target->is_split = true;
  target->parton_mom[0] = quark_4mom[quark_order[1]];
  target->parton_mom[1] = quark_4mom[quark_order[1] + 2];

  projectile->time_of_creation = target->time_of_creation;
  projectile->position = target->position;
  projectile->momentum = c->p_projectile;
  target->momentum = c->p_target;

  projectile->collision_count += 1;
  target->collision_count += 1;
  return 0;
}

/// G4FTFAnnihilation::Create1QuarkAntiQuarkString - channel (d). The target nucleon is gone.
template <typename Rng>
__host__ __device__ inline bool ftf_annih_create_1q_aq_string(SplitableHadron* projectile,
                                                              SplitableHadron* target,
                                                              FtfParameters<double>* params,
                                                              AnnihCommon* c, Rng& rng) {
  // TWO pairs must annihilate, so the candidate table is over ordered pairs of pairs: at most
  // 36 entries, and the survivor is the one index neither pair used.
  int cand_aq[36] = {};
  int cand_q[36] = {};
  int candidats_n = 0;
  for (int iaq1 = 0; iaq1 < 3; ++iaq1) {
    for (int iaq2 = 0; iaq2 < 3; ++iaq2) {
      if (iaq1 == iaq2) { continue; }
      for (int iq1 = 0; iq1 < 3; ++iq1) {
        for (int iq2 = 0; iq2 < 3; ++iq2) {
          if (iq1 == iq2) { continue; }
          if (-c->aq[iaq1] == c->q[iq1] && -c->aq[iaq2] == c->q[iq2]) {
            if ((iaq1 == 0 && iaq2 == 1) || (iaq1 == 1 && iaq2 == 0)) {
              cand_aq[candidats_n] = 2;
            } else if ((iaq1 == 0 && iaq2 == 2) || (iaq1 == 2 && iaq2 == 0)) {
              cand_aq[candidats_n] = 1;
            } else if ((iaq1 == 1 && iaq2 == 2) || (iaq1 == 2 && iaq2 == 1)) {
              cand_aq[candidats_n] = 0;
            }
            if ((iq1 == 0 && iq2 == 1) || (iq1 == 1 && iq2 == 0)) {
              cand_q[candidats_n] = 2;
            } else if ((iq1 == 0 && iq2 == 2) || (iq1 == 2 && iq2 == 0)) {
              cand_q[candidats_n] = 1;
            } else if ((iq1 == 1 && iq2 == 2) || (iq1 == 2 && iq2 == 1)) {
              cand_q[candidats_n] = 0;
            }
            ++candidats_n;
          }
        }
      }
    }
  }

  if (candidats_n == 0) { return true; }

  const int sampled_case = ftf_shoot_int(candidats_n, rng);
  const int left_aq = c->aq[cand_aq[sampled_case]];
  const int left_q = c->q[cand_q[sampled_case]];

  ftf_set_first_parton(projectile, left_q, &c->refused, rng);
  ftf_set_second_parton(projectile, left_aq, &c->refused, rng);
  projectile->status = 0;
  const double a_ksi = rng.uniform();
  const int new_code = ftf_annih_meson_code(left_aq, left_q, a_ksi);
  if (data::ftf_find_hadron(new_code) == nullptr) { return false; }
  projectile->pdg = new_code;
  params->proj_min_diff_mass = 0.5 * units::GeV<double>();
  params->proj_min_non_diff_mass = 0.5 * units::GeV<double>();

  target->status = 4;  // The target nucleon has annihilated
  c->p_projectile = Vec4(0.0, 0.0, 0.0, c->sqrt_s);
  c->p_projectile = lorentz_apply(c->to_lab, c->p_projectile);

  Vec4 pquark(0.0, 0.0, -c->sqrt_s / 2.0, c->sqrt_s / 2.0);
  Vec4 paquark(0.0, 0.0, c->sqrt_s / 2.0, c->sqrt_s / 2.0);
  if (c->rotate_strings) {
    pquark = lorentz_apply(c->random_rotation, pquark);
    paquark = lorentz_apply(c->random_rotation, paquark);
  }
  pquark = lorentz_apply(c->to_lab, pquark);
  paquark = lorentz_apply(c->to_lab, paquark);
  projectile->parton_mom[0] = pquark;
  projectile->parton_mom[1] = paquark;
  projectile->is_split = true;

  projectile->time_of_creation = target->time_of_creation;
  projectile->position = target->position;
  projectile->momentum = c->p_projectile;

  projectile->collision_count += 1;
  target->collision_count += 1;
  return true;
}

/// G4FTFAnnihilation::Annihilate.
template <typename Rng>
__host__ __device__ inline bool ftf_annihilate(SplitableHadron* projectile,
                                               SplitableHadron* target,
                                               SplitableHadron* additional,
                                               bool* made_additional,
                                               FtfParameters<double>* params, AnnihCommon* c,
                                               Rng& rng) {
  *c = AnnihCommon();
  *made_additional = false;

  c->p_projectile = projectile->momentum;
  const int projectile_pdg = projectile->pdg;
  if (projectile_pdg > 0) {
    // Not an anti-baryon: the target is marked as reggeon-involved and nothing else happens.
    target->status = 3;
    return false;
  }
  const double m0_projectile2 =
      c->p_projectile.e * c->p_projectile.e - g4gpu::mag2(c->p_projectile.v);

  const int target_pdg = target->pdg;
  c->p_target = target->momentum;
  const double m0_target2 = c->p_target.e * c->p_target.e - g4gpu::mag2(c->p_target.v);

  const Vec4 psum = c->p_projectile + c->p_target;
  c->s = psum.e * psum.e - g4gpu::mag2(psum.v);
  c->sqrt_s = std::sqrt(c->s);

  const Vec3d bv = psum.boost_vector();
  LorentzRot to_cms = lorentz_boost(Vec3d{-bv.x, -bv.y, -bv.z});
  const Vec4 ptmp = lorentz_apply(to_cms, c->p_projectile);
  lorentz_rotate_z(&to_cms, -lv_phi(ptmp));
  lorentz_rotate_y(&to_cms, -lv_theta(ptmp));
  c->to_lab = lorentz_inverse(to_cms);

  // `(1880/sqrt(s))^4` through G4Pow::powA, not std::pow. At rest sqrt(s) is about 1880 MeV -
  // the two nucleon masses - so the probability is near 1 and falls as the fourth power of the
  // energy: an annihilation at rest is isotropic and one at 10 GeV is not.
  if (rng.uniform() <=
      data::g4pow_pow_a<double>(1880.0 / c->sqrt_s, 4.0)) {
    c->rotate_strings = true;
    lorentz_rotate_z(&c->random_rotation, 2.0 * units::pi<double>() * rng.uniform());
    lorentz_rotate_y(&c->random_rotation, std::acos(2.0 * rng.uniform() - 1.0));
    lorentz_rotate_z(&c->random_rotation, 2.0 * units::pi<double>() * rng.uniform());
  }

  const data::FtfHadron* pdef = data::ftf_find_hadron(projectile_pdg);
  const data::FtfHadron* tdef = data::ftf_find_hadron(target_pdg);
  if (pdef == nullptr || tdef == nullptr) {
    c->refused = FtfRefusal::kUnknownHadronCode;
    return false;
  }
  const double meson_prod_threshold =
      pdef->mass + tdef->mass + (2.0 * 140.0 + 16.0) * units::MeV<double>();
  double prel2 = c->s * c->s + m0_projectile2 * m0_projectile2 + m0_target2 * m0_target2 -
                 2.0 * (c->s * (m0_projectile2 + m0_target2) + m0_projectile2 * m0_target2);
  prel2 /= c->s;

  double x_a = 0.0, x_b = 0.0, x_c = 0.0, x_d = 0.0;
  if (prel2 <= 0.0) {
    // Annihilation at rest. "Values are copied from Parameters" - and they are NOT the same
    // numbers InitForInteraction computes, which is why they are written out again.
    x_a = 625.1;
    x_b = 0.0;
    x_c = 49.989;
    x_d = 6.614;
  } else {
    const double flow_f = 1.0 / std::sqrt(prel2) * units::GeV<double>();
    x_a = 25.0 * flow_f;
    if (c->sqrt_s < meson_prod_threshold) {
      x_b = 3.13 + 140.0 * data::g4pow_pow_a<double>(
                              (meson_prod_threshold - c->sqrt_s) / units::GeV<double>(), 2.5);
    } else {
      x_b = 6.8 * units::GeV<double>() / c->sqrt_s;
    }
    if (pdef->mass + tdef->mass > c->sqrt_s) { x_b = 0.0; }
    const double msum = pdef->mass + tdef->mass;
    x_c = 2.0 * flow_f * msum * msum / c->s;
    x_d = 23.3 * units::GeV<double>() * units::GeV<double>() / c->s;
  }

  // The nine-by-two weight table. `X_a` is NOT scaled - only b, c and d are.
  bool is_unknown = false;
  if (target_pdg == 2212 || target_pdg == 2214) {
    if (projectile_pdg == -2212 || projectile_pdg == -2214) {
      x_b *= 5.0; x_c *= 5.0; x_d *= 6.0;
    } else if (projectile_pdg == -2112 || projectile_pdg == -2114) {
      x_b *= 4.0; x_c *= 4.0; x_d *= 4.0;
    } else if (projectile_pdg == -3122) {
      x_b *= 3.0; x_c *= 3.0; x_d *= 2.0;
    } else if (projectile_pdg == -3112) {
      x_b *= 2.0; x_c *= 2.0; x_d *= 0.0;
    } else if (projectile_pdg == -3212) {
      x_b *= 3.0; x_c *= 3.0; x_d *= 2.0;
    } else if (projectile_pdg == -3222) {
      x_b *= 4.0; x_c *= 4.0; x_d *= 2.0;
    } else if (projectile_pdg == -3312) {
      x_b *= 1.0; x_c *= 1.0; x_d *= 0.0;
    } else if (projectile_pdg == -3322) {
      x_b *= 2.0; x_c *= 2.0; x_d *= 0.0;
    } else if (projectile_pdg == -3334) {
      x_b *= 0.0; x_c *= 0.0; x_d *= 0.0;
    } else {
      is_unknown = true;
    }
  } else if (target_pdg == 2112 || target_pdg == 2114) {
    if (projectile_pdg == -2212 || projectile_pdg == -2214) {
      x_b *= 4.0; x_c *= 4.0; x_d *= 4.0;
    } else if (projectile_pdg == -2112 || projectile_pdg == -2114) {
      x_b *= 5.0; x_c *= 5.0; x_d *= 6.0;
    } else if (projectile_pdg == -3122) {
      x_b *= 3.0; x_c *= 3.0; x_d *= 2.0;
    } else if (projectile_pdg == -3112) {
      x_b *= 4.0; x_c *= 4.0; x_d *= 2.0;
    } else if (projectile_pdg == -3212) {
      x_b *= 3.0; x_c *= 3.0; x_d *= 2.0;
    } else if (projectile_pdg == -3222) {
      x_b *= 2.0; x_c *= 2.0; x_d *= 0.0;
    } else if (projectile_pdg == -3312) {
      x_b *= 2.0; x_c *= 2.0; x_d *= 0.0;
    } else if (projectile_pdg == -3322) {
      x_b *= 1.0; x_c *= 1.0; x_d *= 0.0;
    } else if (projectile_pdg == -3334) {
      x_b *= 0.0; x_c *= 0.0; x_d *= 0.0;
    } else {
      is_unknown = true;
    }
  } else {
    is_unknown = true;
  }
  if (is_unknown) {
    // Geant4 prints "Unknown anti-baryon for FTF annihilation" and CONTINUES with the unscaled
    // b, c and d. A silent continuation with the wrong weights is exactly what
    // docs/HADRONIC_PLAN.md section 6 rule 4 forbids, so it is reported instead.
    c->refused = FtfRefusal::kUndefinedProjectileNucleonAssumed;
    return false;
  }

  const double x_annihilation = x_a + x_b + x_c + x_d;

  ftf_annih_unpack_baryon(projectile_pdg, &c->aq[0], &c->aq[1], &c->aq[2]);
  ftf_annih_unpack_baryon(target_pdg, &c->q[0], &c->q[1], &c->q[2]);

  const double ksi = rng.uniform();

  if (ksi < x_a / x_annihilation) {
    return ftf_annih_create_3q_aq_strings(projectile, target, additional, made_additional,
                                          params, c, rng);
  }

  int result_code = 99;
  if (ksi < (x_a + x_b) / x_annihilation) {
    result_code = ftf_annih_create_1dq_adq_string(projectile, target, c, rng);
    if (result_code == 0) { return true; }
    if (result_code == 99) { return false; }
  }

  if (ksi < (x_a + x_b + x_c) / x_annihilation) {
    result_code = ftf_annih_create_2q_aq_strings(projectile, target, params, c, rng);
    if (result_code == 0) { return true; }
    if (result_code == 99) { return false; }
  }

  if (ksi < (x_a + x_b + x_c + x_d) / x_annihilation) {
    return ftf_annih_create_1q_aq_string(projectile, target, params, c, rng);
  }

  return true;
}

}  // namespace g4gpu::hadronic::ftf
