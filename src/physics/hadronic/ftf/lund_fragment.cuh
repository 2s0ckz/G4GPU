// G4LundStringFragmentation::FragmentString and everything under it: one excited string in,
// a list of hadrons out.
//
// Transcribed from
//   source/processes/hadronic/models/parton_string/hadronization/src/G4LundStringFragmentation.cc
//     FragmentString, Loop_toFragmentString, Splitup, SplitEandP, SplitLast, SampleState,
//     DiQuarkSplitup, Diquark_AntiDiquark_belowThreshold_lastSplitting,
//     Diquark_AntiDiquark_aboveThreshold_lastSplitting, Quark_AntiQuark_lastSplitting,
//     Quark_Diquark_lastSplitting
//   .../src/G4VLongitudinalStringDecay.cc
//     ProduceOneHadron, PossibleHadronMass, QuarkSplitup, CreatePartonPair
//
// THE WORKSPACE IS A STRUCT PASSED BY POINTER. SplitLast enumerates up to 350 final states with
// a PDG code pair and a weight each, and the hadron lists have to be built up before they can be
// joined: `sizeof(FragmentWorkspace<double,96>)` is 33,320 bytes - 5,600 for the enumeration and
// 27,648 for the three lists of 72-byte hadrons (96 left, 96 right, 192 joined). A kernel that
// put that on the stack would blow the 16,384-byte frame `Upload` allows for a whole step before
// it fragmented anything, which is the constraint docs/HADRONIC_PLAN.md's P1 hazard note and the
// P11 brief both name. The same shape as precompound/precompound_model.cuh's PrecoWorkspace, and
// the cost is one workspace per track in flight, not per string. Measured with it in global
// memory, the probe kernel in tests/test_ftf_lund.cu is 210 registers and a 152-byte frame with
// nothing spilled.
//
// WHAT IS MUTABLE AND WHY IT IS NOT IN THE TABLES. Splitup rewrites DiquarkSuppress and
// StrangeSuppress from the string's own mass before every split and restores them afterwards,
// and DiQuarkSplitup rewrites StrangeSuppress again inside that. In Geant4 they are members of
// the model; here the tables are const - they are shared, and on the device they belong in
// constant memory - so the two live in the workspace as `SamplerState`. The restore is
// reproduced exactly, including the one that does not restore: DiQuarkSplitup's break arm
// calls `SetStrangenessSuppression((1.0-StrSup)/2.0)`, which treats a SUPPRESSION as if it
// were a probability and so does not put back what was there; nothing reads it between that
// line and the `StrangeSuppress = StrSup` three lines later, so it is dead, and it is
// transcribed as written rather than silently corrected.
//
// FOUR MORE DEAD LINES, all in this file's originals:
//
//  1. FragmentString brackets its ProduceOneHadron call with `SetMassCut(10000.*MeV)` and
//     `SetMassCut(Mcut)`. ProduceOneHadron reads MassCut in exactly one place - a
//     `#ifdef debug_VStringDecay` printout - so the pair has no effect on the answer. MassCut
//     itself is read nowhere in the Lund path at all.
//  2. SplitLast ends `string->LorentzRotate(toObserverFrame)` on a string it never transformed
//     INTO that frame, and Loop_toFragmentString deletes the string on the next statement.
//  3. Loop_toFragmentString's `toCms`/`toObserverFrame` are recomputed inside the loop and the
//     `toCmsI`/`toObserverFrameI` computed before it are the ones the final rotation uses; the
//     inner pair is used only for the hadron just produced.
//  4. G4FragmentingString::TransformToAlignedCms leaves Ptright's z component alone where
//     every other maintainer of that member zeroes it - and the Lund path never reads either
//     Ptleft or Ptright. See fragmenting_string.cuh.
#pragma once
#include <cmath>

#include "core/units.cuh"
#include "data/ftf_hadrons.hh"
#include "physics/hadronic/ftf/fragmenting_string.cuh"
#include "physics/hadronic/ftf/hadron_builder.cuh"
#include "physics/hadronic/ftf/lorentz.cuh"
#include "physics/hadronic/ftf/lund_tables.cuh"
#include "physics/hadronic/ftf/refusal.cuh"
#include "physics/hadronic/ftf/string_decay.cuh"

namespace g4gpu::hadronic::ftf {

/// The two sampler parameters Splitup and DiQuarkSplitup rewrite per split. Geant4 keeps them
/// as members of the model; the tables here are const, so they live with the event.
template <typename real_t>
struct SamplerState {
  real_t strange_suppress = 0;
  real_t diquark_suppress = 0;
};

/// One produced hadron, in the form P6's hand-over wants it.
struct FragHadron {
  int pdg = 0;
  Vec4 momentum;
  Vec3d position{0.0, 0.0, 0.0};
  double formation_time = 0.0;
};

/// Everything one string's fragmentation needs to write down. 33,320 bytes at kMaxHadrons = 96;
/// see the file header for where they go and why it is a pointer.
template <typename real_t, int kMaxHadrons = 96>
struct FragmentWorkspace {
  // SplitLast's final-state enumeration - Geant4's FS_LeftHadron / FS_RightHadron / FS_Weight
  // and NumberOf_FS, with Geant4's own 350 capacity.
  static constexpr int kMaxFinalStates = 350;
  int fs_left[kMaxFinalStates];
  int fs_right[kMaxFinalStates];
  real_t fs_weight[kMaxFinalStates];
  int number_of_fs = 0;

  // The two hadron lists the loop fills and FragmentString joins.
  FragHadron left[kMaxHadrons];
  int n_left = 0;
  FragHadron right[kMaxHadrons];
  int n_right = 0;

  // The joined result.
  FragHadron out[2 * kMaxHadrons];
  int n_out = 0;

  SamplerState<real_t> sampler;
  real_t minimal_string_mass = 0;
  real_t minimal_string_mass2 = 0;
  FtfRefusal refused = FtfRefusal::kNone;
};

/// G4KineticTrack's CONSTRUCTOR, the one line of it that is physics.
///
/// `G4KineticTrack::G4KineticTrack(definition, formationTime, position, 4momentum)` opens with
///
///     if (G4KaonZero::KaonZero() == theDefinition || G4AntiKaonZero::AntiKaonZero() == ...)
///       { theDefinition = (G4UniformRand()<0.5) ? KaonZeroShort : KaonZeroLong; }
///
/// so a kaon0 or an anti_kaon0 becomes a K0S or a K0L on a coin toss, and the toss consumes a
/// uniform deviate. That is not a detail: the Lund fragmentation produces kaon0 and anti_kaon0
/// freely - `Meson[0][2][0]` is 311 and a d-s pair from the vacuum makes one - and every
/// produced hadron goes through this constructor, in Splitup, in SplitLast and in
/// ProduceOneHadron. A port that substitutes the species without spending the deviate gets the
/// right particle and then the WRONG everything else, because the whole remaining random
/// stream is shifted by one. Found exactly that way: ref/oracle/ftf_fragment.csv disagreed on
/// a d-dbar string at 1.5 GeV with 4 draws against 6, and the two missing draws were these.
///
/// The masses are equal to the last bit (497.614 MeV for all three), so the substitution
/// changes no kinematics - only the species and the stream.
template <typename real_t, typename Rng>
__host__ __device__ inline int ftf_kinetic_track_pdg(int pdg, Rng& rng) {
  if (pdg == 311 || pdg == -311) {
    return (static_cast<real_t>(rng.uniform()) < real_t(0.5)) ? 310 : 130;
  }
  return pdg;
}

/// A PDG mass, or 0 with a refusal for a code the table does not carry.
template <typename real_t>
__host__ __device__ inline real_t ftf_pdg_mass(int pdg, FtfRefusal* refused) {
  const data::FtfHadron* h = data::ftf_find_hadron(pdg);
  if (h == nullptr) {
    if (refused != nullptr) { *refused = FtfRefusal::kUnknownHadronCode; }
    return real_t(0);
  }
  return static_cast<real_t>(h->mass);
}

/// G4VLongitudinalStringDecay::SetMinimalStringMass applied to a FragmentingString, storing
/// the result in the workspace as Geant4 stores it in the model.
template <typename real_t, int N>
__host__ __device__ inline void ftf_set_minimal_string_mass(const LundTables<real_t>* t,
                                                            FragmentWorkspace<real_t, N>* ws,
                                                            const FragmentingString& s) {
  const MinimalStringMass<real_t> mm =
      ftf_minimal_string_mass(t, s.left, s.right, static_cast<real_t>(ftf_string_mass(s)));
  if (mm.refused != FtfRefusal::kNone) { ws->refused = mm.refused; }
  ws->minimal_string_mass = mm.mass;
  ws->minimal_string_mass2 = mm.mass2;
}

/// G4VLongitudinalStringDecay::SampleQuarkFlavor with the mutable suppression.
template <typename real_t, typename Rng>
__host__ __device__ inline int ftf_sample_quark_flavor_s(const LundTables<real_t>* t,
                                                         const SamplerState<real_t>& st,
                                                         Rng& rng) {
  int quark = 1;
  const real_t ksi = static_cast<real_t>(rng.uniform());
  if (ksi < t->prob_cb) {
    quark = (ksi < t->prob_ccbar) ? 4 : 5;
  } else {
    quark = 1 + static_cast<int>(static_cast<real_t>(rng.uniform()) / st.strange_suppress);
  }
  return quark;
}

/// G4VLongitudinalStringDecay::CreatePartonPair with the mutable suppressions.
template <typename real_t, typename Rng>
__host__ __device__ inline PartonPair ftf_create_parton_pair_s(const LundTables<real_t>* t,
                                                               const SamplerState<real_t>& st,
                                                               int need_particle,
                                                               bool allow_diquarks, Rng& rng) {
  PartonPair out;
  if (allow_diquarks && static_cast<real_t>(rng.uniform()) < st.diquark_suppress) {
    const int q1 = ftf_sample_quark_flavor_s(t, st, rng);
    const int q2 = ftf_sample_quark_flavor_s(t, st, rng);
    const int spin = (q1 != q2 && static_cast<real_t>(rng.uniform()) <= real_t(0.5)) ? 1 : 3;
    const int hi = (q1 > q2) ? q1 : q2;
    const int lo = (q1 < q2) ? q1 : q2;
    const int code = (hi * 1000 + lo * 100 + spin) * need_particle;
    out.first = ftf_existing_code(-code);
    out.second = ftf_existing_code(code);
    return out;
  }
  const int code = ftf_sample_quark_flavor_s(t, st, rng) * need_particle;
  out.first = ftf_existing_code(code);
  out.second = ftf_existing_code(-code);
  return out;
}

/// G4VLongitudinalStringDecay::QuarkSplitup. `IsParticle` is -1 for a quark, so the pair
/// created is an antiquark-or-antidiquark and its partner; `QuarkPair.first` goes into the
/// hadron with the decaying quark and `QuarkPair.second` becomes the new string end.
template <typename real_t, typename Rng>
__host__ __device__ inline BuiltHadron ftf_quark_splitup(const LundTables<real_t>* t,
                                                          const SamplerState<real_t>& st,
                                                          int decay, int* created, Rng& rng) {
  const int is_particle = (decay > 0) ? -1 : +1;
  const PartonPair pair = ftf_create_parton_pair_s(t, st, is_particle, true, rng);
  *created = pair.second;
  return ftf_hadron_build(t, pair.first, decay, rng);
}

/// G4LundStringFragmentation::DiQuarkSplitup.
template <typename real_t, typename Rng>
__host__ __device__ inline BuiltHadron ftf_diquark_splitup(const LundTables<real_t>* t,
                                                           SamplerState<real_t>* st, int decay,
                                                           int* created, Rng& rng) {
  const real_t str_sup = st->strange_suppress;
  const real_t prob_qqbar = (real_t(1.0) - real_t(2.0) * str_sup) * real_t(1.25);

  if (static_cast<real_t>(rng.uniform()) < t->diquark_break_prob) {
    int stable_quark = decay / 1000;
    int decay_quark = (decay / 100) % 10;
    if (static_cast<real_t>(rng.uniform()) < real_t(0.5)) {
      const int swap = stable_quark;
      stable_quark = decay_quark;
      decay_quark = swap;
    }
    const int is_particle = (decay_quark > 0) ? -1 : +1;
    st->strange_suppress = (real_t(1.0) - prob_qqbar) / real_t(2.0);
    const PartonPair pair = ftf_create_parton_pair_s(t, *st, is_particle, false, rng);
    // Geant4's `SetStrangenessSuppression((1.0-StrSup)/2.0)` - a suppression fed to a setter
    // that expects a probability. Dead: nothing reads StrangeSuppress between here and the
    // restore below. Written as Geant4 writes it.
    st->strange_suppress = (real_t(1.0) - str_sup) / real_t(2.0);

    const int quark_encoding = pair.second;
    const int aq = (quark_encoding < 0) ? -quark_encoding : quark_encoding;
    const int as = (stable_quark < 0) ? -stable_quark : stable_quark;
    const int i10 = (aq > as) ? aq : as;
    const int i20 = (aq < as) ? aq : as;
    const int spin =
        (i10 != i20 && static_cast<real_t>(rng.uniform()) <= real_t(0.5)) ? 1 : 3;
    const int new_decay = -1 * is_particle * (i10 * 1000 + i20 * 100 + spin);
    *created = ftf_existing_code(new_decay);
    const BuiltHadron had = ftf_hadron_build(t, pair.first, decay_quark, rng);
    st->strange_suppress = str_sup;
    return had;
  }

  const int is_particle = (decay > 0) ? +1 : -1;
  st->strange_suppress = (real_t(1.0) - prob_qqbar) / real_t(2.0);
  const PartonPair pair = ftf_create_parton_pair_s(t, *st, is_particle, false, rng);
  *created = pair.second;
  const BuiltHadron had = ftf_hadron_build(t, pair.first, decay, rng);
  st->strange_suppress = str_sup;
  return had;
}

/// G4VLongitudinalStringDecay::PossibleHadronMass.
///
/// Returns the estimated mass and, through `h1`/`h2`, the hadrons it built. `h2` stays 0 for
/// anything that is not a four-quark string, which is what tells ProduceOneHadron whether to
/// emit one hadron or two.
template <typename real_t, typename Rng>
__host__ __device__ inline real_t ftf_possible_hadron_mass(const LundTables<real_t>* t,
                                                           const FragmentingString& s,
                                                           bool use_build, int* h1, int* h2,
                                                           FtfRefusal* refused, Rng& rng) {
  real_t mass = real_t(0.0);
  *h1 = 0;
  *h2 = 0;
  if (!ftf_is_four_quark_string(s)) {
    // build == 0 means BuildLowSpin; `use_build` selects Build instead, which is what the
    // `Pcreate build` argument is for. No caller in 11.1.1 passes anything but 0.
    const BuiltHadron b = use_build ? ftf_hadron_build(t, s.left, s.right, rng)
                                    : ftf_hadron_build_low_spin(t, s.left, s.right, rng);
    if (b.refused != FtfRefusal::kNone && refused != nullptr) { *refused = b.refused; }
    *h1 = b.pdg;
    mass = (b.pdg != 0) ? ftf_pdg_mass<real_t>(b.pdg, nullptr) : t->max_mass;
    return mass;
  }

  const real_t string_mass = static_cast<real_t>(ftf_string_mass(s));
  int cluster_interrupt = 0;
  int hh1 = 0, hh2 = 0;
  for (;;) {
    if (cluster_interrupt++ >= t->cluster_loop_interrupt) {
      // Geant4's `return false`, i.e. 0.0 - and it returns from INSIDE the loop, before the
      // `pdefs->first = Hadron1` at the end of the function. So the caller's pair keeps the
      // (nullptr, nullptr) it was constructed with and ProduceOneHadron emits NOTHING, even
      // though the last iteration built two hadrons. h1/h2 stay 0 here for the same reason.
      // Unreachable from FragmentString, which calls ProduceOneHadron only for a string that
      // is NOT a four-quark string, so no oracle point can distinguish this from the
      // alternative; it is written to match the source rather than to pass a test.
      return real_t(0.0);
    }
    const int lq1 = s.left / 1000;
    const int lq2 = (s.left / 100) % 10;
    const int rq1 = s.right / 1000;
    const int rq2 = (s.right / 100) % 10;
    if (static_cast<real_t>(rng.uniform()) < real_t(0.5)) {
      hh1 = ftf_hadron_build(t, lq1, rq1, rng).pdg;
      hh2 = ftf_hadron_build(t, lq2, rq2, rng).pdg;
    } else {
      hh1 = ftf_hadron_build(t, lq1, rq2, rng).pdg;
      hh2 = ftf_hadron_build(t, lq2, rq1, rng).pdg;
    }
    if (hh1 == 0 || hh2 == 0) { continue; }
    const real_t m1 = ftf_pdg_mass<real_t>(hh1, nullptr);
    const real_t m2 = ftf_pdg_mass<real_t>(hh2, nullptr);
    if (string_mass <= m1 + m2) { continue; }
    mass = m1 + m2;
    break;
  }
  *h1 = hh1;
  *h2 = hh2;
  return mass;
}

/// G4VLongitudinalStringDecay::ProduceOneHadron - a string too light to fragment becomes one
/// hadron, or two if it is a four-quark string.
///
/// The one-hadron arm does NOT conserve energy and Geant4's comment says so: the hadron keeps
/// the string's three-momentum and its energy is recomputed from its own mass.
template <typename real_t, int N, typename Rng>
__host__ __device__ inline void ftf_produce_one_hadron(const LundTables<real_t>* t,
                                                       FragmentWorkspace<real_t, N>* ws,
                                                       const FragmentingString& s,
                                                       const Vec4& string_mom,
                                                       const Vec3d& string_pos, Rng& rng) {
  ws->n_out = 0;
  ftf_set_minimal_string_mass(t, ws, s);
  int h1 = 0, h2 = 0;
  (void)ftf_possible_hadron_mass(t, s, false, &h1, &h2, &ws->refused, rng);
  if (h1 == 0) { return; }
  if (h2 == 0) {
    const real_t m = ftf_pdg_mass<real_t>(h1, &ws->refused);
    const Vec3d p3 = string_mom.v;
    const Vec4 mom(p3, std::sqrt(g4gpu::mag2(p3) + static_cast<double>(m) * m));
    // The track is constructed AFTER the four-momentum is built from the ORIGINAL
    // definition's mass, so the K0 substitution cannot change the kinematics here either.
    h1 = ftf_kinetic_track_pdg<real_t>(h1, rng);
    ws->out[0].pdg = h1;
    ws->out[0].momentum = mom;
    ws->out[0].position = string_pos;
    ws->out[0].formation_time = 0.0;
    ws->n_out = 1;
    return;
  }
  const real_t m1 = ftf_pdg_mass<real_t>(h1, &ws->refused);
  const real_t m2 = ftf_pdg_mass<real_t>(h2, &ws->refused);
  const TwoBodyMomenta<real_t> tb =
      ftf_sample_4momentum(t, m1, m2, static_cast<real_t>(string_mom.mag()), rng);
  Vec4 mom1(tb.px, tb.py, tb.pz, tb.e);
  Vec4 mom2(tb.apx, tb.apy, tb.apz, tb.ae);
  h1 = ftf_kinetic_track_pdg<real_t>(h1, rng);
  h2 = ftf_kinetic_track_pdg<real_t>(h2, rng);
  const Vec3d bv = string_mom.boost_vector();
  mom1.boost(bv);
  mom2.boost(bv);
  ws->out[0].pdg = h1;
  ws->out[0].momentum = mom1;
  ws->out[0].position = string_pos;
  ws->out[0].formation_time = 0.0;
  ws->out[1].pdg = h2;
  ws->out[1].momentum = mom2;
  ws->out[1].position = string_pos;
  ws->out[1].formation_time = 0.0;
  ws->n_out = 2;
}

/// G4LundStringFragmentation::SplitEandP. Returns false when the string cannot give the hadron
/// its four-momentum, which is Geant4's null return and means "start all over".
template <typename real_t, int N, typename Rng>
__host__ __device__ inline bool ftf_split_e_and_p(const LundTables<real_t>* t,
                                                  FragmentWorkspace<real_t, N>* ws,
                                                  int hadron_pdg, const FragmentingString& s,
                                                  const FragmentingString& new_string,
                                                  Vec4* out, Rng& rng) {
  Vec4 string_4mom = s.pstring;
  const real_t string_mt2 = static_cast<real_t>(ftf_string_mass_t2(s));
  const real_t string_mt = std::sqrt(string_mt2);

  const real_t hadron_mass = ftf_pdg_mass<real_t>(hadron_pdg, &ws->refused);
  ftf_set_minimal_string_mass(t, ws, new_string);
  const real_t minimal = ws->minimal_string_mass;

  if (minimal < real_t(0.0)) { return false; }
  if ((hadron_mass + minimal > static_cast<real_t>(ftf_string_mass(s))) ||
      minimal < real_t(0.)) {
    return false;
  }

  string_4mom.v.z = 0.0;
  Vec3d string_pt = string_4mom.v;
  string_pt.z = 0.0;

  // TmtCur: the "temperature" of the hadron's transverse mass, by what is decaying into what.
  // The q->M and qq->M arms are Geant4's empty blocks - Tmt unchanged, with `Tmt*0.89`
  // commented out in the second.
  real_t tmt_cur = t->tmt;
  const bool decay_is_quark = ftf_is_quark(s.decay);
  const bool decay_is_diquark = ftf_is_diquark(s.decay);
  const data::FtfHadron* hd = data::ftf_find_hadron(hadron_pdg);
  const int hadron_baryon = (hd != nullptr) ? hd->baryon : 0;
  if (decay_is_quark && hadron_baryon != 0) {
    tmt_cur = t->tmt * real_t(0.37);  // q -> B
  } else if (decay_is_quark && hadron_baryon == 0) {
    // q -> M: Tmt
  } else if (decay_is_diquark && hadron_baryon == 0) {
    // qq -> M: Tmt, with Tmt*0.89 commented out in the original
  } else if (decay_is_diquark && hadron_baryon != 0) {
    tmt_cur = t->tmt * real_t(1.35);  // qq -> B
  }

  real_t hadron_mass_t2 = real_t(0), residual_mass_t2 = real_t(0);
  Vec3d hadron_pt{0.0, 0.0, 0.0};
  int attempt = 0;
  for (;;) {
    ++attempt;
    if (attempt > t->string_loop_interrupt) { return false; }
    const real_t hadron_mt =
        hadron_mass - tmt_cur * std::log(static_cast<real_t>(rng.uniform()));
    const real_t pt2 = hadron_mt * hadron_mt - hadron_mass * hadron_mass;
    const real_t pt = std::sqrt(pt2);
    const real_t phi =
        real_t(2.) * units::pi<real_t>() * static_cast<real_t>(rng.uniform());
    hadron_pt = Vec3d{pt * std::cos(phi), pt * std::sin(phi), 0.0};
    const Vec3d rem_sys_pt = string_pt - hadron_pt;
    hadron_mass_t2 = hadron_mass * hadron_mass + static_cast<real_t>(g4gpu::mag2(hadron_pt));
    residual_mass_t2 = minimal * minimal + static_cast<real_t>(g4gpu::mag2(rem_sys_pt));
    if (!(std::sqrt(hadron_mass_t2) + std::sqrt(residual_mass_t2) > string_mt)) { break; }
  }

  const real_t d = string_mt2 - hadron_mass_t2 - residual_mass_t2;
  const real_t pz2 =
      (d * d - real_t(4) * hadron_mass_t2 * residual_mass_t2) / real_t(4.) / string_mt2;
  if (pz2 < real_t(0)) { return false; }

  const real_t pz = std::sqrt(pz2);
  const real_t zmin = (std::sqrt(hadron_mass_t2 + pz2) - pz) / std::sqrt(string_mt2);
  const real_t zmax = (std::sqrt(hadron_mass_t2 + pz2) + pz) / std::sqrt(string_mt2);
  if (zmin >= zmax) { return false; }

  const real_t z = ftf_get_light_cone_z(t, zmin, zmax, s.decay, hadron_pdg,
                                        static_cast<real_t>(hadron_pt.x),
                                        static_cast<real_t>(hadron_pt.y), rng);

  const real_t lcd = static_cast<real_t>(ftf_light_cone_decay(s));
  const real_t dir = static_cast<real_t>(ftf_decay_direction(s));
  hadron_pt.z = 0.5 * static_cast<double>(dir) *
                (static_cast<double>(z) * lcd - hadron_mass_t2 / (z * lcd));
  const double hadron_e =
      0.5 * (static_cast<double>(z) * lcd + hadron_mass_t2 / (static_cast<double>(z) * lcd));
  *out = Vec4(hadron_pt, hadron_e);
  return true;
}

/// G4LundStringFragmentation::Splitup - one hadron off one end of the string.
///
/// Returns the hadron's PDG code (0 for "no hadron this time") and, on success, the remaining
/// string. The two probability rescalings are the string's own mass feeding back into the
/// flavour composition: a light string suppresses diquark and strange production almost
/// completely, and `NumberOfpossibleBaryons` counts how many baryons the string could already
/// make, so a qq-qqbar string is harder to break than a q-qbar one.
template <typename real_t, int N, typename Rng>
__host__ __device__ inline int ftf_splitup(const LundTables<real_t>* t,
                                           FragmentWorkspace<real_t, N>* ws,
                                           FragmentingString* s, FragmentingString* new_string,
                                           bool* have_new_string, Vec4* hadron_mom, Rng& rng) {
  *have_new_string = false;
  const int side_of_decay = (static_cast<real_t>(rng.uniform()) < real_t(0.5)) ? 1 : -1;
  if (side_of_decay < 0) {
    ftf_set_left_parton_stable(s);
  } else {
    ftf_set_right_parton_stable(s);
  }

  const real_t string_mass = static_cast<real_t>(ftf_string_mass(*s));
  const real_t prob_dq_adq = ws->sampler.diquark_suppress;
  const real_t prob_sas = real_t(1.0) - real_t(2.0) * ws->sampler.strange_suppress;

  int n_possible_baryons = 2;
  if (!ftf_is_quark(s->left)) { ++n_possible_baryons; }
  if (!ftf_is_quark(s->right)) { ++n_possible_baryons; }

  real_t actual = prob_dq_adq;
  actual *= (real_t(1.0) - data::g4pow_pow_a<real_t>(
                               static_cast<real_t>(n_possible_baryons) * real_t(1400.0) /
                                   string_mass,
                               real_t(8.0)));
  if (actual < real_t(0.0)) { actual = real_t(0.); }
  ws->sampler.diquark_suppress = actual;

  real_t mth = real_t(1250.0);                              // 2 Mk + Mpi
  if (n_possible_baryons == 3) { mth = real_t(2520.0); }     // Mlambda/Msigma + Mk + Mpi
  else if (n_possible_baryons == 4) { mth = real_t(2380.0); }  // 2 Mlambda/Msigma + Mk + Mpi

  actual = prob_sas;
  actual *= (real_t(1.0) - data::g4pow_pow_a<real_t>(mth / string_mass, real_t(2.5)));
  if (actual < real_t(0.0)) { actual = real_t(0.0); }
  ws->sampler.strange_suppress = (real_t(1.0) - actual) / real_t(2.0);

  int new_string_end = 0;
  BuiltHadron had;
  if (ftf_decay_is_quark(*s)) {
    had = ftf_quark_splitup(t, ws->sampler, s->decay, &new_string_end, rng);
  } else {
    had = ftf_diquark_splitup(t, &ws->sampler, s->decay, &new_string_end, rng);
  }

  ws->sampler.diquark_suppress = prob_dq_adq;
  ws->sampler.strange_suppress = (real_t(1.0) - prob_sas) / real_t(2.0);

  if (had.refused != FtfRefusal::kNone) { ws->refused = had.refused; }
  if (had.pdg == 0) { return 0; }

  // The content-only string, whose minimum mass SplitEandP needs before its momentum exists.
  const FragmentingString content = ftf_string_content_only(*s, new_string_end, &ws->refused);
  if (!ftf_split_e_and_p(t, ws, had.pdg, *s, content, hadron_mom, rng)) { return 0; }

  // Geant4 constructs the G4KineticTrack here, between SplitEandP and the new
  // G4FragmentingString - so the K0 coin toss falls between those two and not after them.
  const int track_pdg = ftf_kinetic_track_pdg<real_t>(had.pdg, rng);

  *new_string = ftf_string_after_hadron(*s, new_string_end, *hadron_mom, &ws->refused);
  *have_new_string = true;
  return track_pdg;
}

/// G4LundStringFragmentation::SampleState - a cumulative walk over the enumerated final
/// states.
///
/// `Sum += FS_Weight[i]/SumWeights` with SumWeights = 0 gives NaN, the comparison
/// `Sum >= ksi` is then false for every i, and the function returns the LAST index. Geant4
/// does exactly that and it is reachable: a string whose only enumerated states all have
/// weight zero - which the d-dbar meson row's zero eta weight (docs/RISK.md V85) can produce
/// when it is the only state that fits. Transcribed rather than guarded.
template <typename real_t, int N, typename Rng>
__host__ __device__ inline int ftf_sample_state(FragmentWorkspace<real_t, N>* ws, Rng& rng) {
  if (ws->number_of_fs > 349) {
    ws->refused = FtfRefusal::kFinalStateCapacity;
    ws->number_of_fs = 349;
  }
  real_t sum_weights = real_t(0.);
  for (int i = 0; i < ws->number_of_fs; ++i) { sum_weights += ws->fs_weight[i]; }
  const real_t ksi = static_cast<real_t>(rng.uniform());
  real_t sum = real_t(0.);
  int index = 0;
  for (int i = 0; i < ws->number_of_fs; ++i) {
    sum += (ws->fs_weight[i] / sum_weights);
    index = i;
    if (sum >= ksi) { break; }
  }
  return index;
}

/// Appends one enumerated final state, with Geant4's 350 clamp and its JustWarning turned
/// into a report.
template <typename real_t, int N>
__host__ __device__ inline void ftf_push_final_state(FragmentWorkspace<real_t, N>* ws, int left,
                                                     int right, real_t weight) {
  if (ws->number_of_fs > 349) {
    ws->refused = FtfRefusal::kFinalStateCapacity;
    ws->number_of_fs = 349;
  }
  ws->fs_left[ws->number_of_fs] = left;
  ws->fs_right[ws->number_of_fs] = right;
  ws->fs_weight[ws->number_of_fs] = weight;
  ++ws->number_of_fs;
}

/// G4LundStringFragmentation::Quark_AntiQuark_lastSplitting.
template <typename real_t, int N>
__host__ __device__ inline bool ftf_quark_antiquark_last_splitting(
    const LundTables<real_t>* t, FragmentWorkspace<real_t, N>* ws, const FragmentingString& s) {
  const real_t string_mass = static_cast<real_t>(ftf_string_mass(s));
  const real_t string_mass_sqr = string_mass * string_mass;

  const int quark = (s.left > 0) ? s.left : s.right;
  const int anti_quark = (s.left > 0) ? s.right : s.left;
  const int abs_q = (quark < 0) ? -quark : quark;
  const int abs_aq = (anti_quark < 0) ? -anti_quark : anti_quark;
  // `Qcharge[IDquark-1]` with the SIGNED code: for a quark on the left this is the right
  // index, and the function is only ever reached with `quark > 0` because of the swap above.
  const int quark_charge = t->qcharge[quark - 1];
  const int anti_quark_charge = -t->qcharge[abs_aq - 1];

  ws->number_of_fs = 0;
  for (int prod_q = 1; prod_q < 4; ++prod_q) {  // u-ubar, d-dbar, s-sbar only
    const int left_charge = quark_charge - t->qcharge[prod_q - 1];
    int sign_q = left_charge / 3;
    if (sign_q == 0) { sign_q = 1; }
    if (quark == 1 && prod_q == 3) { sign_q = 1; }   // K0    (d, sbar)
    if (quark == 3 && prod_q == 1) { sign_q = -1; }  // K0bar (s, dbar)
    if (quark == 4 && prod_q == 2) { sign_q = 1; }   // D0    (c, ubar)
    if (quark == 5 && prod_q == 1) { sign_q = -1; }  // anti_B0
    if (quark == 5 && prod_q == 3) { sign_q = -1; }  // anti_Bs0

    const int right_charge = anti_quark_charge + t->qcharge[prod_q - 1];
    int sign_aq = right_charge / 3;
    if (sign_aq == 0) { sign_aq = 1; }
    if (anti_quark == -1 && prod_q == 3) { sign_aq = -1; }  // K0bar
    if (anti_quark == -3 && prod_q == 1) { sign_aq = 1; }   // K0
    if (anti_quark == -4 && prod_q == 2) { sign_aq = -1; }  // anti_D0
    if (anti_quark == -5 && prod_q == 1) { sign_aq = 1; }   // B0
    if (anti_quark == -5 && prod_q == 3) { sign_aq = 1; }   // Bs0

    int state_q = 0;
    const int max_loops = 1000;
    int loop = 0;
    do {
      const int lcode = sign_q * t->meson[abs_q - 1][prod_q - 1][state_q];
      const data::FtfHadron* lh = data::ftf_find_hadron(lcode);
      if (lh == nullptr) {
        ++state_q;
        continue;
      }
      const real_t lm = static_cast<real_t>(lh->mass);

      int state_aq = 0;
      const int max_inner = 1000;
      int inner = 0;
      do {
        const int rcode = sign_aq * t->meson[abs_aq - 1][prod_q - 1][state_aq];
        const data::FtfHadron* rh = data::ftf_find_hadron(rcode);
        if (rh == nullptr) {
          ++state_aq;
          continue;
        }
        const real_t rm = static_cast<real_t>(rh->mass);
        if (string_mass > lm + rm) {
          const real_t psqr = ftf_lambda(string_mass_sqr, lm * lm, rm * rm);
          const real_t w = std::sqrt(psqr) *
                           t->meson_weight[abs_q - 1][prod_q - 1][state_q] *
                           t->meson_weight[abs_aq - 1][prod_q - 1][state_aq] *
                           t->prob_qqbar[prod_q - 1];
          // The LEFT/RIGHT assignment depends on which end of the string the quark was on.
          if (s.left > 0) {
            ftf_push_final_state(ws, rcode, lcode, w);
          } else {
            ftf_push_final_state(ws, lcode, rcode, w);
          }
        }
        ++state_aq;
      } while ((t->meson[abs_aq - 1][prod_q - 1][state_aq] != 0) && ++inner < max_inner);
      if (inner >= max_inner) { return false; }
      ++state_q;
    } while ((t->meson[abs_q - 1][prod_q - 1][state_q] != 0) && ++loop < max_loops);
    if (loop >= max_loops) { return false; }
  }
  return true;
}

/// G4LundStringFragmentation::Quark_Diquark_lastSplitting.
///
/// NOTE the difference from the function above: its inner `if (LeftHadron == NULL) continue;`
/// does NOT increment StateQ, so a non-zero Meson entry with no particle behind it spins the
/// loop until the 1000-iteration guard and the function returns false. The quark-antiquark
/// version was fixed (`{ StateQ++; continue; }`) and this one was not. Transcribed as written,
/// which is why the `state_q` increment below is inside the accepting path only.
template <typename real_t, int N>
__host__ __device__ inline bool ftf_quark_diquark_last_splitting(
    const LundTables<real_t>* t, FragmentWorkspace<real_t, N>* ws, const FragmentingString& s) {
  const real_t string_mass = static_cast<real_t>(ftf_string_mass(s));
  const real_t string_mass_sqr = string_mass * string_mass;

  const bool left_is_quark = ftf_is_quark(s.left);
  const int quark = left_is_quark ? s.left : s.right;
  const int di_quark = left_is_quark ? s.right : s.left;
  const int abs_q = (quark < 0) ? -quark : quark;
  const int abs_dq = (di_quark < 0) ? -di_quark : di_quark;
  const int dq1 = abs_dq / 1000;
  const int dq2 = (abs_dq - dq1 * 1000) / 100;
  const int sign_dq = (di_quark < 0) ? -1 : 1;

  ws->number_of_fs = 0;
  for (int prod_q = 1; prod_q < 4; ++prod_q) {
    int sign_q;
    if (quark > 0) {
      sign_q = -1;
      if (quark == 2) { sign_q = 1; }
      if (quark == 1 && prod_q == 3) { sign_q = 1; }   // K0
      if (quark == 3 && prod_q == 1) { sign_q = -1; }  // K0bar
      if (quark == 4) { sign_q = 1; }                  // D+, D0, Ds+
      if (quark == 5) { sign_q = -1; }                 // B-, anti_B0, anti_Bs0
    } else {
      sign_q = 1;
      if (quark == -2) { sign_q = -1; }
      if (quark == -1 && prod_q == 3) { sign_q = -1; }  // K0bar
      if (quark == -3 && prod_q == 1) { sign_q = 1; }   // K0
      if (quark == -4) { sign_q = -1; }
      if (quark == -5) { sign_q = 1; }
    }
    if (abs_q == prod_q) { sign_q = 1; }

    int state_q = 0;
    const int max_loops = 1000;
    int loop = 0;
    // Geant4's do-while with a BARE `continue` on a null hadron, written as a for(;;) whose
    // tail is the do-while's condition - so the null path and the accepting path evaluate
    // `Meson[...][StateQ] != 0 && ++loopCounter < 1000` identically, short circuit included.
    // Writing it as `if (null) { ++loop; continue; }` would increment the counter on the
    // normal exit too, which Geant4's short circuit does not.
    for (;;) {
      const int lcode = sign_q * t->meson[abs_q - 1][prod_q - 1][state_q];
      const data::FtfHadron* lh = data::ftf_find_hadron(lcode);
      if (lh != nullptr) {
        const real_t lm = static_cast<real_t>(lh->mass);

        int state_dq = 0;
        const int max_inner = 1000;
        int inner = 0;
        for (;;) {
          const int rcode = sign_dq * t->baryon[dq1 - 1][dq2 - 1][prod_q - 1][state_dq];
          const data::FtfHadron* rh = data::ftf_find_hadron(rcode);
          if (rh != nullptr) {
            const real_t rm = static_cast<real_t>(rh->mass);
            if (string_mass > lm + rm) {
              const real_t psqr = ftf_lambda(string_mass_sqr, lm * lm, rm * rm);
              const real_t w = std::sqrt(psqr) *
                               t->meson_weight[abs_q - 1][prod_q - 1][state_q] *
                               t->baryon_weight[dq1 - 1][dq2 - 1][prod_q - 1][state_dq] *
                               t->prob_qqbar[prod_q - 1];
              ftf_push_final_state(ws, lcode, rcode, w);
            }
            ++state_dq;
          }
          if (!((t->baryon[dq1 - 1][dq2 - 1][prod_q - 1][state_dq] != 0) &&
                ++inner < max_inner)) {
            break;
          }
        }
        if (inner >= max_inner) { return false; }
        ++state_q;
      }
      if (!((t->meson[abs_q - 1][prod_q - 1][state_q] != 0) && ++loop < max_loops)) { break; }
    }
    if (loop >= max_loops) { return false; }
  }
  return true;
}

/// G4LundStringFragmentation::Diquark_AntiDiquark_aboveThreshold_lastSplitting - two baryons.
template <typename real_t, int N>
__host__ __device__ inline bool ftf_diquark_antidiquark_above_threshold(
    const LundTables<real_t>* t, FragmentWorkspace<real_t, N>* ws, const FragmentingString& s) {
  const real_t string_mass = static_cast<real_t>(ftf_string_mass(s));
  const real_t string_mass_sqr = string_mass * string_mass;

  const int anti_di = (s.left < 0) ? s.left : s.right;
  const int di = (s.left < 0) ? s.right : s.left;
  const int abs_adi = (anti_di < 0) ? -anti_di : anti_di;
  const int abs_di = (di < 0) ? -di : di;
  const int adi_q1 = abs_adi / 1000;
  const int adi_q2 = (abs_adi - adi_q1 * 1000) / 100;
  const int di_q1 = abs_di / 1000;
  const int di_q2 = (abs_di - di_q1 * 1000) / 100;

  ws->number_of_fs = 0;
  for (int prod_q = 1; prod_q < 6; ++prod_q) {  // NOTE: 1..5 here, 1..3 in the other two
    int state_adiq = 0;
    const int max_loops = 1000;
    int loop = 0;
    // Bare `continue` on a null hadron in both loops - the same shape as
    // Quark_Diquark_lastSplitting's and not the fixed shape of
    // Quark_AntiQuark_lastSplitting's.
    for (;;) {
      const int lcode = -t->baryon[adi_q1 - 1][adi_q2 - 1][prod_q - 1][state_adiq];
      const data::FtfHadron* lh = data::ftf_find_hadron(lcode);
      if (lh != nullptr) {
        const real_t lm = static_cast<real_t>(lh->mass);

        int state_diq = 0;
        const int max_inner = 1000;
        int inner = 0;
        for (;;) {
          const int rcode = +t->baryon[di_q1 - 1][di_q2 - 1][prod_q - 1][state_diq];
          const data::FtfHadron* rh = data::ftf_find_hadron(rcode);
          if (rh != nullptr) {
            const real_t rm = static_cast<real_t>(rh->mass);
            if (string_mass > lm + rm) {
              const real_t psqr = ftf_lambda(string_mass_sqr, lm * lm, rm * rm);
              // sqrt(P^2)*P^2, i.e. |p|^3 - unlike the other two functions, which use |p|.
              const real_t w =
                  std::sqrt(psqr) * psqr *
                  t->baryon_weight[adi_q1 - 1][adi_q2 - 1][prod_q - 1][state_adiq] *
                  t->baryon_weight[di_q1 - 1][di_q2 - 1][prod_q - 1][state_diq] *
                  t->prob_qqbar[prod_q - 1];
              ftf_push_final_state(ws, lcode, rcode, w);
            }
            ++state_diq;
          }
          if (!((t->baryon[di_q1 - 1][di_q2 - 1][prod_q - 1][state_diq] != 0) &&
                ++inner < max_inner)) {
            break;
          }
        }
        if (inner >= max_inner) { return false; }
        ++state_adiq;
      }
      if (!((t->baryon[adi_q1 - 1][adi_q2 - 1][prod_q - 1][state_adiq] != 0) &&
            ++loop < max_loops)) {
        break;
      }
    }
    if (loop >= max_loops) { return false; }
  }
  return true;
}

/// G4LundStringFragmentation::Diquark_AntiDiquark_belowThreshold_lastSplitting - two mesons.
template <typename real_t, typename Rng>
__host__ __device__ inline bool ftf_diquark_antidiquark_below_threshold(
    const LundTables<real_t>* t, const FragmentingString& s, int* left, int* right, Rng& rng) {
  const real_t string_mass = static_cast<real_t>(ftf_string_mass(s));
  int cluster_interrupt = 0;
  bool is_ok = false;
  do {
    const int lq1 = s.left / 1000;
    const int lq2 = (s.left / 100) % 10;
    const int rq1 = s.right / 1000;
    const int rq2 = (s.right / 100) % 10;
    if (static_cast<real_t>(rng.uniform()) < real_t(0.5)) {
      *left = ftf_hadron_build(t, lq1, rq1, rng).pdg;
      *right = (*left == 0) ? 0 : ftf_hadron_build(t, lq2, rq2, rng).pdg;
    } else {
      *left = ftf_hadron_build(t, lq1, rq2, rng).pdg;
      *right = (*left == 0) ? 0 : ftf_hadron_build(t, lq2, rq1, rng).pdg;
    }
    is_ok = (*left != 0) && (*right != 0);
    if (is_ok) {
      is_ok = (string_mass >
               ftf_pdg_mass<real_t>(*left, nullptr) + ftf_pdg_mass<real_t>(*right, nullptr));
    }
    ++cluster_interrupt;
  } while (!is_ok && cluster_interrupt < t->cluster_loop_interrupt);
  return is_ok;
}

/// G4LundStringFragmentation::SplitLast - the remaining string becomes two hadrons.
template <typename real_t, int N, typename Rng>
__host__ __device__ inline bool ftf_split_last(const LundTables<real_t>* t,
                                               FragmentWorkspace<real_t, N>* ws,
                                               FragmentingString* s, Rng& rng) {
  ftf_set_minimal_string_mass(t, ws, *s);
  if (ws->minimal_string_mass < real_t(0.)) { return false; }

  const Vec4 str4 = s->pstring;
  const Vec3d bv = str4.boost_vector();
  LorentzRot to_cms = lorentz_boost(Vec3d{-bv.x, -bv.y, -bv.z});
  const Vec4 pleft_cms = lorentz_apply(to_cms, s->pleft);
  lorentz_rotate_z(&to_cms, -1.0 * lv_phi(pleft_cms));
  lorentz_rotate_y(&to_cms, -1.0 * lv_theta(pleft_cms));
  const LorentzRot to_observer = lorentz_inverse(to_cms);

  const real_t string_mass = static_cast<real_t>(ftf_string_mass(*s));

  int left_hadron = 0, right_hadron = 0;
  ws->number_of_fs = 0;
  for (int i = 0; i < FragmentWorkspace<real_t, N>::kMaxFinalStates; ++i) {
    ws->fs_weight[i] = real_t(0.);
  }

  ftf_set_left_parton_stable(s);  // to query quark contents

  if (ftf_is_four_quark_string(*s)) {
    const int idl = (s->left < 0) ? -s->left : s->left;
    const int idr = (s->right < 0) ? -s->right : s->right;
    if (idl > 3000 || idr > 3000) {
      if (!ftf_diquark_antidiquark_below_threshold(t, *s, &left_hadron, &right_hadron, rng)) {
        return false;
      }
    } else if (string_mass - ws->minimal_string_mass < real_t(0.)) {
      if (!ftf_diquark_antidiquark_below_threshold(t, *s, &left_hadron, &right_hadron, rng)) {
        return false;
      }
    } else {
      ftf_diquark_antidiquark_above_threshold(t, ws, *s);
      if (ws->number_of_fs == 0) { return false; }
      const int st = ftf_sample_state(ws, rng);
      if (s->left < 0) {
        left_hadron = ws->fs_left[st];
        right_hadron = ws->fs_right[st];
      } else {
        left_hadron = ws->fs_right[st];
        right_hadron = ws->fs_left[st];
      }
    }
  } else if (ftf_decay_is_quark(*s) && ftf_stable_is_quark(*s)) {
    ftf_quark_antiquark_last_splitting(t, ws, *s);
    if (ws->number_of_fs == 0) { return false; }
    const int st = ftf_sample_state(ws, rng);
    if (s->left < 0) {
      left_hadron = ws->fs_right[st];
      right_hadron = ws->fs_left[st];
    } else {
      left_hadron = ws->fs_left[st];
      right_hadron = ws->fs_right[st];
    }
  } else {
    ftf_quark_diquark_last_splitting(t, ws, *s);
    if (ws->number_of_fs == 0) { return false; }
    const int st = ftf_sample_state(ws, rng);
    if (ftf_is_quark(s->left)) {
      left_hadron = ws->fs_left[st];
      right_hadron = ws->fs_right[st];
    } else {
      left_hadron = ws->fs_right[st];
      right_hadron = ws->fs_left[st];
    }
  }

  const Vec4 p_left = s->pleft;
  const Vec4 p_right = s->pright;
  const real_t lm = ftf_pdg_mass<real_t>(left_hadron, &ws->refused);
  const real_t rm = ftf_pdg_mass<real_t>(right_hadron, &ws->refused);
  const TwoBodyMomenta<real_t> tb = ftf_sample_4momentum(t, lm, rm, string_mass, rng);
  Vec4 left_mom(tb.px, tb.py, tb.pz, tb.e);
  Vec4 right_mom(tb.apx, tb.apy, tb.apz, tb.ae);

  // Sample4Momentum puts the FIRST hadron along +z, which is wrong when the string is moving
  // the other way; for anything but a q-qbar string the two are swapped on a coin toss
  // conditioned on the sign of the string ends' z.
  if (!(ftf_decay_is_quark(*s) && ftf_stable_is_quark(*s))) {
    if (static_cast<real_t>(rng.uniform()) <= real_t(0.5)) {
      if (p_left.v.z <= 0.0) {
        const Vec4 tmp = left_mom;
        left_mom = right_mom;
        right_mom = tmp;
      }
    } else {
      if (p_right.v.z >= 0.0) {
        const Vec4 tmp = left_mom;
        left_mom = right_mom;
        right_mom = tmp;
      }
    }
  }

  left_mom = lorentz_apply(to_observer, left_mom);
  right_mom = lorentz_apply(to_observer, right_mom);

  if (ws->n_left >= N || ws->n_right >= N) {
    ws->refused = FtfRefusal::kStringHadronCapacity;
    return false;
  }
  // The two G4KineticTrack constructions, in Geant4's order: LeftVector first.
  left_hadron = ftf_kinetic_track_pdg<real_t>(left_hadron, rng);
  ws->left[ws->n_left].pdg = left_hadron;
  ws->left[ws->n_left].momentum = left_mom;
  ws->left[ws->n_left].position = Vec3d{0.0, 0.0, 0.0};
  ws->left[ws->n_left].formation_time = 0.0;
  ++ws->n_left;
  right_hadron = ftf_kinetic_track_pdg<real_t>(right_hadron, rng);
  ws->right[ws->n_right].pdg = right_hadron;
  ws->right[ws->n_right].momentum = right_mom;
  ws->right[ws->n_right].position = Vec3d{0.0, 0.0, 0.0};
  ws->right[ws->n_right].formation_time = 0.0;
  ++ws->n_right;

  // Geant4's `string->LorentzRotate(toObserverFrame)` goes here. The string was never
  // transformed into that frame and is deleted by the caller on the next statement; see the
  // file header.
  return true;
}

/// G4LundStringFragmentation::Loop_toFragmentString.
template <typename real_t, int N, typename Rng>
__host__ __device__ inline bool ftf_loop_to_fragment_string(
    const LundTables<real_t>* t, FragmentWorkspace<real_t, N>* ws, const FragmentingString& the_string,
    int direction, double time_of_creation, const Vec3d& position_of_creation, Rng& rng) {
  LorentzRot to_observer_frame_i;
  bool final_success = false;
  int attempt = 0;

  while (!final_success && attempt++ < t->string_loop_interrupt) {
    FragmentingString current = the_string;
    const LorentzRot to_cms_i = ftf_transform_to_aligned_cms(&current);
    to_observer_frame_i = lorentz_inverse(to_cms_i);

    ws->n_left = 0;
    ws->n_right = 0;
    bool inner_success = true;
    const int max_loops = 1000;
    int loop_counter = -1;

    for (;;) {
      ftf_set_minimal_string_mass(t, ws, current);
      if (ftf_stop_fragmenting(ws->minimal_string_mass,
                               static_cast<real_t>(ftf_string_mass(current)),
                               ftf_is_four_quark_string(current), rng)) {
        break;
      }
      if (++loop_counter >= max_loops) { break; }

      const LorentzRot to_cms = ftf_transform_to_aligned_cms(&current);
      const LorentzRot to_observer = lorentz_inverse(to_cms);

      FragmentingString new_string;
      bool have_new = false;
      Vec4 hadron_mom;
      const int hadron_pdg =
          ftf_splitup(t, ws, &current, &new_string, &have_new, &hadron_mom, rng);

      if (hadron_pdg != 0) {
        const Vec4 lab_mom = lorentz_apply(to_observer, hadron_mom);
        // The hadron's position and formation time start at zero (Splitup constructs the
        // kinetic track with a default position and a formation time of 0), so the
        // transformed `Coordinate` is zero and the formation time is
        // `t0 - fermi/c_light` for every hadron the LOOP produces - about -3.34e-15 ns. The
        // two hadrons SplitLast produces keep 0. That difference is visible in
        // ref/oracle/ftf_fragment.csv and is how the two paths can be told apart in it.
        const double formation_time =
            time_of_creation + 0.0 - fermi<double>() / units::c_light<double>();
        FragHadron h;
        h.pdg = hadron_pdg;
        h.momentum = lab_mom;
        h.position = position_of_creation;
        h.formation_time = formation_time;
        if (ftf_decay_direction(current) > 0) {
          if (ws->n_left >= N) {
            ws->refused = FtfRefusal::kStringHadronCapacity;
            return false;
          }
          ws->left[ws->n_left++] = h;
        } else {
          if (ws->n_right >= N) {
            ws->refused = FtfRefusal::kStringHadronCapacity;
            return false;
          }
          ws->right[ws->n_right++] = h;
        }
        current = new_string;
      }
      ftf_string_lorentz_rotate(&current, to_observer);
    }

    if (loop_counter >= max_loops) { inner_success = false; }

    if (inner_success && ftf_split_last(t, ws, &current, rng)) { final_success = true; }
  }

  if (!final_success) { ws->refused = FtfRefusal::kFragmentLoopExhausted; }

  const int sign = (direction < 0) ? -1 : +1;
  for (int i = 0; i < ws->n_left; ++i) {
    Vec4 tmp = ws->left[i].momentum;
    tmp.v.z = sign * tmp.v.z;
    ws->left[i].momentum = lorentz_apply(to_observer_frame_i, tmp);
  }
  for (int i = 0; i < ws->n_right; ++i) {
    Vec4 tmp = ws->right[i].momentum;
    tmp.v.z = sign * tmp.v.z;
    ws->right[i].momentum = lorentz_apply(to_observer_frame_i, tmp);
  }
  return final_success;
}

/// G4LundStringFragmentation::FragmentString - the entry point for one string.
///
/// Fills `ws->out` with `ws->n_out` hadrons. An empty result is Geant4's empty
/// G4KineticTrackVector, which the caller reads as "retry the whole interaction".
template <typename real_t, int N, typename Rng>
__host__ __device__ inline void ftf_fragment_string(const LundTables<real_t>* t,
                                                    FragmentWorkspace<real_t, N>* ws,
                                                    int left_code, int right_code,
                                                    const Vec4& left_mom, const Vec4& right_mom,
                                                    int direction, double time_of_creation,
                                                    const Vec3d& position, Rng& rng) {
  ws->n_out = 0;
  ws->n_left = 0;
  ws->n_right = 0;
  ws->refused = FtfRefusal::kNone;
  ws->sampler.strange_suppress = t->strange_suppress;
  ws->sampler.diquark_suppress = t->diquark_suppress;

  const FragmentingString the_string =
      ftf_string_from_excited(left_code, right_code, left_mom, right_mom, direction);

  FragmentingString a_string = the_string;
  ftf_set_minimal_string_mass(t, ws, a_string);

  if (!ftf_is_four_quark_string(a_string) &&
      !ftf_is_it_fragmentable(ws->minimal_string_mass,
                              static_cast<real_t>(ftf_string_mass(a_string)))) {
    // Geant4 raises the mass cut to 10 GeV around this call and puts it back afterwards;
    // ProduceOneHadron never reads it. See the file header.
    ftf_produce_one_hadron(t, ws, a_string, the_string.pstring, position, rng);
    for (int i = 0; i < ws->n_out; ++i) {
      ws->out[i].formation_time = time_of_creation;
      ws->out[i].position = position;
    }
    return;
  }

  const bool success = ftf_loop_to_fragment_string(t, ws, the_string, direction,
                                                   time_of_creation, position, rng);
  if (!success) {
    ws->n_out = 0;
    return;
  }

  // Join: the left list in order, then the right list REVERSED.
  int n = 0;
  for (int i = 0; i < ws->n_left; ++i) { ws->out[n++] = ws->left[i]; }
  for (int i = ws->n_right - 1; i >= 0; --i) { ws->out[n++] = ws->right[i]; }
  ws->n_out = n;
}

}  // namespace g4gpu::hadronic::ftf
