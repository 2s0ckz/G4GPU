// The meson-baryon elastic cross section - the first channel in this package a PION can reach.
//
// Transcribed from G4XAqmTotal, G4XAqmElastic and G4XMesonBaryonElastic (im_r_matrix, 11.1.1),
// with `G4Pow::powN` and `G4VCollision::GetNumberOfPartons`.
//
// QBBC gives `G4BinaryCascade` every pi+ and pi- from 0 to 1.5 GeV, and
// `G4BinaryCascade::ApplyYourself`'s first statement is an `&&` on the species, so a pion enters
// the cascade at EVERY energy (bic/binary_cascade.cuh). `G4Scatterer`'s second channel is
// `G4CollisionMesonBaryon`, a composite of `G4CollisionMesonBaryonToResonance` and
// `G4CollisionMesonBaryonElastic`; this file is the second of those two.
//
// ## G4XMesonBaryonElastic does not use its own tracks' identities
//
// It builds a DUMMY pi+ and a DUMMY proton carrying the real four-momenta, evaluates
// `G4XPDGElastic` on that pair, and scales the answer by the ratio of two AQM cross sections:
//
//     sigma = G4XPDGElastic(pi+ dummy, p dummy) * G4XAqmElastic(trk1, trk2)
//                                              / G4XAqmElastic(pi+ dummy, p dummy)
//
// The dummy tracks carry the real momenta and the PDG masses of a pi+ and a proton, so the
// `pLab` the PDG fit is read at is built from the real kinematics and the reference masses. The
// species enter only through the ratio.
//
// **And the ratio is exactly 1 for every pair this channel can see.** `G4XAqmTotal` depends on
// nothing but the quark content: the number of strange quarks, the number of non-strange ones,
// and whether each particle has exactly two. Every meson the cascade produces is a pion (2
// non-strange partons) and every baryon is a nucleon, a Delta or an N* (3 non-strange partons),
// so `nMesons` is 1 and both strangeness ratios are 0 for the real pair and for the dummy pair
// alike, and the two AQM cross sections are the same number. MEASURED against the oracle over
// every species pair the channel accepts. The ratio is still computed here, because it is what
// the class computes and because a release that gave BIC a kaon would make it stop being 1.
//
// ## Two things in G4XAqmTotal that are wrong and cannot bite here
//
// **The strangeness ratio is an INTEGER division.** `G4int sTrk1`, `G4int qTrk1`, and
// `G4double sRatio1 = sTrk1 / qTrk1;` - so a Lambda (one strange quark, two non-strange) gets
// 1/2 = 0 where the physical ratio is 0.5, and the `(1 - 0.4*sRatio)` suppression is skipped
// entirely. Only a particle with at least as many strange quarks as non-strange - a kaon, a Xi,
// an Omega - gets a non-zero ratio. Every species this cascade reaches has `sTrk = 0`, so the
// division is 0/3 either way.
//
// **`G4XAqmElastic` raises a dimensioned quantity to a non-integer power.**
// `sigma = 0.39 * powA(sigmaTot, 1.5)` with `sigmaTot` in Geant4's internal area units, about
// 2.7e-24 mm^2, so the result is about 4e-36 - nine orders below the cross section it is
// compared against in the `if (sigma > sigmaTot) throw` that follows, which is why the throw
// never fires. The number is meaningless on its own and is only ever used as a ratio, where the
// units and the 0.39 cancel. Reproduced exactly, because the ratio is only exactly 1 if both
// sides are computed the same way.
//
// ## REFUSED, by name
//
//   * **every strange or heavy species.** `parton_counts` answers for the nucleons, the three
//     pions and the non-strange baryon resonances, and refuses anything else by PDG code rather
//     than guessing a quark content. A kaon reaching this file would otherwise silently get
//     `nq = 0, ns = 0` and an AQM cross section of 40 mb.
//   * `G4XAqmTotal`'s **gamma** case, which `G4XMesonBaryonElastic` tests for separately and
//     answers with a zero cross section. That branch IS transcribed, because the test is on the
//     LIGHTER of the two tracks and a photon is lighter than a pion.
#ifndef G4GPU_BIC_IMR_XSEC_MESON_CUH
#define G4GPU_BIC_IMR_XSEC_MESON_CUH

#include <cmath>

#include "physics/hadronic/bic/im_r/xsec_nn.cuh"

namespace g4gpu::bic::imr {

/// PDG codes this file knows the quark content of.
enum : int { kPdgPiZero = 111, kPdgGamma = 22 };

/// `G4VCollision::GetNumberOfPartons` split into the two counts `G4XAqmTotal` reads: `ns` is
/// `GetQuarkContent(3) + GetAntiQuarkContent(3)` (strange) and `nq` is the same sum over
/// flavours 1, 2, 4, 5 and 6.
///
/// Returns false and sets the refusal for a species it does not know, rather than answering
/// (0, 0) - which `G4XAqmTotal` would turn into a perfectly plausible 40 mb.
__host__ __device__ inline bool parton_counts(int pdg, int& nq, int& ns, XsecRefusal& ref) {
  nq = 0;
  ns = 0;
  switch (pdg) {
    case kPdgProton:
    case kPdgNeutron:
      nq = 3;
      return true;
    case kPdgPiPlus:
    case kPdgPiMinus:
    case kPdgPiZero:
      nq = 2;
      return true;
    case kPdgGamma:
      return true;  // no quarks at all; G4XMesonBaryonElastic tests for it before it gets here
    default:
      break;
  }
  // The non-strange baryon resonances: Delta(1232) and every Delta* and N* the collision tree
  // produces. They are three light quarks like the nucleon, and their PDG codes are the ones
  // G4CollisionMesonBaryonToResonance looks up by number. Recognised by the rule the encodings
  // follow rather than by a list of fifty: a baryon's code has |code| >= 1000 with the last
  // digit 2J+1, and the three quark digits are all 1 or 2 for a non-strange one.
  int c = (pdg < 0) ? -pdg : pdg;
  if (c >= 1000) {
    c /= 10;  // drop 2J+1
    const int q3 = c % 10;
    const int q2 = (c / 10) % 10;
    const int q1 = (c / 100) % 10;
    if (q1 >= 1 && q1 <= 2 && q2 >= 1 && q2 <= 2 && q3 >= 1 && q3 <= 2) {
      nq = 3;
      return true;
    }
  }
  ref.pair = true;
  ref.pdg1 = pdg;
  return false;
}

/// `G4Pow::powN(x, n)` for the small non-negative `n` this file uses - a loop of multiplications
/// below |n| = 9 and `std::pow` above it, with zero base giving zero.
__host__ __device__ inline double g4pow_pow_n(double x, int n) {
  if (x == 0.0) { return 0.0; }
  const int an = (n >= 0) ? n : -n;
  if (an > 8) { return std::pow(x, static_cast<double>(n)); }
  double res = 1.0;
  if (n >= 0) {
    for (int i = 0; i < n; ++i) { res *= x; }
  } else {
    const double y = 1.0 / x;
    for (int i = 0; i < an; ++i) { res *= y; }
  }
  return res;
}

/// `G4XAqmTotal::CrossSection` - the additive quark model's total cross section, which depends on
/// nothing but the two quark contents.
///
/// `s_ratio` is computed as Geant4 computes it: an INTEGER division assigned to a double. See the
/// file header.
__host__ __device__ inline double x_aqm_total(int nq1, int ns1, int nq2, int ns2) {
  double s_ratio1 = 0.0;
  if (nq1 != 0) { s_ratio1 = static_cast<double>(ns1 / nq1); }
  double s_ratio2 = 0.0;
  if (nq2 != 0) { s_ratio2 = static_cast<double>(ns2 / nq2); }
  int n_mesons = 0;
  if (ns1 + nq1 == 2) { ++n_mesons; }
  if (ns2 + nq2 == 2) { ++n_mesons; }
  return 40.0 * g4pow_pow_n(2.0 / 3.0, n_mesons) * (1.0 - 0.4 * s_ratio1) *
         (1.0 - 0.4 * s_ratio2) * millibarn();
}

/// `G4XAqmElastic::CrossSection` - `0.39 * powA(sigmaTot, 1.5)`, in whatever units that leaves.
///
/// Geant4 throws a `G4HadronicException` if the result exceeds the total; it cannot, by nine
/// orders of magnitude, for the reason the file header gives. The comparison is transcribed as a
/// flag rather than a throw.
__host__ __device__ inline double x_aqm_elastic(int nq1, int ns1, int nq2, int ns2,
                                                bool& exceeds_total) {
  const double sigma_tot = x_aqm_total(nq1, ns1, nq2, ns2);
  const double sigma = 0.39 * g4gpu::data::g4pow_pow_a<double>(sigma_tot, 1.5);
  exceeds_total = (sigma > sigma_tot);
  return sigma;
}

/// `G4XMesonBaryonElastic::CrossSection`.
///
/// `m_pi_plus` and `m_proton` are the two PDG masses the dummy tracks are built with; they are
/// parameters because there is no particle table in a kernel, and they are what the PDG elastic
/// fit is then read at.
__host__ __device__ inline double x_meson_baryon_elastic(int pdg1, int pdg2, double m1, double m2,
                                                         const LorentzVector& p1,
                                                         const LorentzVector& p2,
                                                         double m_pi_plus, double m_proton,
                                                         XsecRefusal& ref) {
  // `FindLightParticle` compares the two PDG masses; a photon is lighter than a pion and gives
  // a zero cross section.
  const int light_pdg = (m1 < m2) ? pdg1 : pdg2;
  if (light_pdg == kPdgGamma) { return 0.0; }

  int nq1 = 0, ns1 = 0, nq2 = 0, ns2 = 0;
  if (!parton_counts(pdg1, nq1, ns1, ref) || !parton_counts(pdg2, nq2, ns2, ref)) { return 0.0; }
  // The dummy pair is always (pi+, proton): two non-strange partons and three.
  bool dummy_exceeds = false;
  bool real_exceeds = false;
  const double x_aqm_dummy = x_aqm_elastic(2, 0, 3, 0, dummy_exceeds);
  const double x_aqm = x_aqm_elastic(nq1, ns1, nq2, ns2, real_exceeds);
  double factor = 1.0;
  if (x_aqm_dummy != 0.0) { factor = x_aqm / x_aqm_dummy; }

  // The two dummy tracks carry the REAL four-momenta and the reference masses, so `sqrt(s)` is
  // built from the real kinematics and the PDG fit is keyed on (pi+, p).
  const double sqrt_s = (p1 + p2).mag();
  XsecRefusal pdg_ref;
  const double sigma =
      x_pdg_elastic(kPdgPiPlus, kPdgProton, m_pi_plus, m_proton, sqrt_s, pdg_ref);
  if (pdg_ref.any()) { ref = pdg_ref; }
  return sigma * factor;
}

}  // namespace g4gpu::bic::imr

#endif
