// G4FragmentingString: the string as the fragmentation loop mutates it.
//
// Transcribed from
//   source/processes/hadronic/models/parton_string/hadronization/src/G4FragmentingString.cc
//     all four constructors, SetLeftPartonStable, SetRightPartonStable, GetDecayDirection,
//     IsAFourQuarkString, DecayIsQuark, StableIsQuark, StablePt, DecayPt, LightConePlus,
//     LightConeMinus, LightConeDecay, Get4Momentum, Mass, Mass2, MassT2, GetPstring, GetPleft,
//     GetPright, TransformToAlignedCms, TransformToCenterOfMass, LorentzRotate, Boost
//   .../include/G4FragmentingString.hh   (SetPleft, SetPright, LorentzRotate, the inlines)
//
// The state is eleven numbers and two PDG codes, and the redundancy in it is deliberate on
// Geant4's part and load-bearing here: `Pstring` is always `Pleft + Pright`, `Pplus`/`Pminus`
// are always Pstring's light-cone components, and `Ptleft`/`Ptright` are Pleft/Pright with z
// zeroed - but each is stored, and each SETTER recomputes only some of them. SetPleft
// recomputes Ptleft, Pstring, Pplus and Pminus but NOT Ptright; the (old, newdecay, momentum)
// constructor recomputes all of them by hand. Reproducing the redundancy rather than deriving
// the dependents on read is what makes the arithmetic bit-identical: `Pplus` after a SetPleft
// is `(Pleft + Pright).plus()`, which is not the same double as `Pleft.plus() +
// Pright.plus()`.
//
// `decaying` IS three-valued, not a bool. `None` is not an error state - the two
// "quark content only" constructors leave it at None for exactly one statement before setting
// it - but GetDecayDirection, StablePt, DecayPt and LightConeDecay all throw on it in Geant4.
// A kernel cannot throw, so they return a sentinel and set a refusal; the one caller that
// could hit it (Splitup, between the two constructors) cannot, because the second constructor
// assigns `decaying` from the old string before returning.
#pragma once
#include <cmath>

#include "data/ftf_hadrons.hh"
#include "physics/hadronic/ftf/lorentz.cuh"
#include "physics/hadronic/ftf/refusal.cuh"

namespace g4gpu::hadronic::ftf {

/// G4FragmentingString::DecaySide.
enum class DecaySide : int { kNone = 0, kLeft = 1, kRight = 2 };

/// G4FragmentingString. `left`/`right` and `stable`/`decay` are PDG codes, which is what the
/// G4ParticleDefinition pointers are used as everywhere in the fragmentation.
struct FragmentingString {
  int left = 0;
  int right = 0;
  Vec3d pt_left{0.0, 0.0, 0.0};
  Vec3d pt_right{0.0, 0.0, 0.0};
  double pplus = 0.0;
  double pminus = 0.0;
  int stable = 0;
  int decay = 0;
  Vec4 pstring;
  Vec4 pleft;
  Vec4 pright;
  DecaySide decaying = DecaySide::kNone;
};

/// G4FragmentingString(const G4ExcitedString&). `direction` is the string's PROJECTILE(+1) /
/// TARGET(-1) tag, and it is what decides which end decays first.
__host__ __device__ inline FragmentingString ftf_string_from_excited(int left_code,
                                                                    int right_code,
                                                                    const Vec4& left_mom,
                                                                    const Vec4& right_mom,
                                                                    int direction) {
  FragmentingString s;
  s.left = left_code;
  s.right = right_code;
  s.pt_left = left_mom.v;
  s.pt_left.z = 0.0;
  s.pt_right = right_mom.v;
  s.pt_right.z = 0.0;
  s.stable = 0;
  s.decay = 0;
  s.decaying = (direction > 0) ? DecaySide::kLeft : DecaySide::kRight;
  s.pleft = left_mom;
  s.pright = right_mom;
  s.pstring = left_mom + right_mom;
  s.pplus = lv_plus(s.pstring);
  s.pminus = lv_minus(s.pstring);
  return s;
}

/// G4FragmentingString(old, newdecay, momentum) - the string after a hadron of four-momentum
/// `momentum` has been taken off the decaying end.
///
/// Note that `Ptleft = old.Ptleft - momentum->vect()` subtracts the hadron's FULL three-vector
/// and then zeroes z, while `Pleft = old.Pleft - Momentum` subtracts the four-vector. The two
/// are consistent only because Ptleft's z is discarded afterwards.
__host__ __device__ inline FragmentingString ftf_string_after_hadron(
    const FragmentingString& old, int new_decay, const Vec4& momentum, FtfRefusal* refused) {
  FragmentingString s;
  s.decaying = DecaySide::kNone;
  const Vec4 mom(momentum.v, momentum.e);
  if (old.decaying == DecaySide::kLeft) {
    s.right = old.right;
    s.pt_right = old.pt_right;
    s.pright = old.pright;
    s.left = new_decay;
    s.pt_left = old.pt_left - momentum.v;
    s.pt_left.z = 0.0;
    s.pleft = old.pleft - mom;
    s.pstring = s.pleft + s.pright;
    s.pplus = lv_plus(s.pstring);
    s.pminus = lv_minus(s.pstring);
    s.decay = s.left;
    s.stable = s.right;
    s.decaying = DecaySide::kLeft;
  } else if (old.decaying == DecaySide::kRight) {
    s.right = new_decay;
    s.pt_right = old.pt_right - momentum.v;
    s.pt_right.z = 0.0;
    s.pright = old.pright - mom;
    s.left = old.left;
    s.pt_left = old.pt_left;
    s.pleft = old.pleft;
    s.pstring = s.pleft + s.pright;
    s.pplus = lv_plus(s.pstring);
    s.pminus = lv_minus(s.pstring);
    s.decay = s.right;
    s.stable = s.left;
    s.decaying = DecaySide::kRight;
  } else {
    // Geant4 throws "no decay Direction defined" here.
    if (refused != nullptr) { *refused = FtfRefusal::kIllegalPartonPair; }
  }
  return s;
}

/// G4FragmentingString(old, newdecay) - quark content only, every momentum zeroed. Used by
/// Splitup to ask SetMinimalStringMass what the REMAINING string's minimum mass would be
/// before its momentum is known.
__host__ __device__ inline FragmentingString ftf_string_content_only(
    const FragmentingString& old, int new_decay, FtfRefusal* refused) {
  FragmentingString s;
  s.decaying = DecaySide::kNone;
  s.pt_left = Vec3d{0.0, 0.0, 0.0};
  s.pt_right = Vec3d{0.0, 0.0, 0.0};
  s.pplus = 0.0;
  s.pminus = 0.0;
  s.stable = 0;
  s.decay = 0;
  s.pstring = Vec4(0.0, 0.0, 0.0, 0.0);
  s.pleft = Vec4(0.0, 0.0, 0.0, 0.0);
  s.pright = Vec4(0.0, 0.0, 0.0, 0.0);
  if (old.decaying == DecaySide::kLeft) {
    s.right = old.right;
    s.left = new_decay;
    s.decaying = DecaySide::kLeft;
  } else if (old.decaying == DecaySide::kRight) {
    s.right = new_decay;
    s.left = old.left;
    s.decaying = DecaySide::kRight;
  } else {
    if (refused != nullptr) { *refused = FtfRefusal::kIllegalPartonPair; }
  }
  return s;
}

__host__ __device__ inline void ftf_set_left_parton_stable(FragmentingString* s) {
  s->stable = s->left;
  s->decay = s->right;
  s->decaying = DecaySide::kRight;
}

__host__ __device__ inline void ftf_set_right_parton_stable(FragmentingString* s) {
  s->stable = s->right;
  s->decay = s->left;
  s->decaying = DecaySide::kLeft;
}

/// +1 for Left, -1 for Right, and 0 where Geant4 throws.
__host__ __device__ inline int ftf_decay_direction(const FragmentingString& s) {
  if (s.decaying == DecaySide::kLeft) { return +1; }
  if (s.decaying == DecaySide::kRight) { return -1; }
  return 0;
}

__host__ __device__ inline bool ftf_is_quark(int pdg) {
  const data::FtfHadron* h = data::ftf_find_hadron(pdg);
  return (h != nullptr) && (h->subtype == data::FtfSubType::kQuark);
}
__host__ __device__ inline bool ftf_is_diquark(int pdg) {
  const data::FtfHadron* h = data::ftf_find_hadron(pdg);
  return (h != nullptr) && (h->subtype == data::FtfSubType::kDiQuark);
}

__host__ __device__ inline bool ftf_is_four_quark_string(const FragmentingString& s) {
  return ftf_is_diquark(s.left) && ftf_is_diquark(s.right);
}
__host__ __device__ inline bool ftf_decay_is_quark(const FragmentingString& s) {
  return ftf_is_quark(s.decay);
}
__host__ __device__ inline bool ftf_stable_is_quark(const FragmentingString& s) {
  return ftf_is_quark(s.stable);
}

__host__ __device__ inline Vec3d ftf_stable_pt(const FragmentingString& s) {
  return (s.decaying == DecaySide::kLeft) ? s.pt_right : s.pt_left;
}
__host__ __device__ inline Vec3d ftf_decay_pt(const FragmentingString& s) {
  return (s.decaying == DecaySide::kLeft) ? s.pt_left : s.pt_right;
}

/// LightConeDecay: p+ for a left decay, p- for a right one. Geant4 throws for None and falls
/// off the end of the function without returning; 0 here.
__host__ __device__ inline double ftf_light_cone_decay(const FragmentingString& s) {
  if (s.decaying == DecaySide::kLeft) { return s.pplus; }
  if (s.decaying == DecaySide::kRight) { return s.pminus; }
  return 0.0;
}

__host__ __device__ inline double ftf_string_mass(const FragmentingString& s) {
  return s.pstring.mag();
}
__host__ __device__ inline double ftf_string_mass2(const FragmentingString& s) {
  const double m = s.pstring.mag();
  // Mass2 is `Pstring.mag2()`, which is SIGNED and is not `mag()*mag()` for a spacelike
  // string - mag() takes the signed square root, so squaring it loses the sign. Computed
  // directly, as Geant4 does.
  (void)m;
  return s.pstring.e * s.pstring.e - g4gpu::mag2(s.pstring.v);
}
/// MassT2 = Pplus*Pminus. Not `m^2 + pt^2`: the two differ once Pplus and Pminus have been
/// recomputed from a sum while Ptleft/Ptright have not.
__host__ __device__ inline double ftf_string_mass_t2(const FragmentingString& s) {
  return s.pplus * s.pminus;
}

__host__ __device__ inline void ftf_set_pleft(FragmentingString* s, const Vec4& p) {
  s->pleft = p;
  s->pt_left = s->pleft.v;
  s->pt_left.z = 0.0;
  s->pstring = s->pleft + s->pright;
  s->pplus = lv_plus(s->pstring);
  s->pminus = lv_minus(s->pstring);
}

__host__ __device__ inline void ftf_set_pright(FragmentingString* s, const Vec4& p) {
  s->pright = p;
  s->pt_right = s->pright.v;
  s->pt_right.z = 0.0;
  s->pstring = s->pleft + s->pright;
  s->pplus = lv_plus(s->pstring);
  s->pminus = lv_minus(s->pstring);
}

/// G4FragmentingString::LorentzRotate. Note the ORDER: SetPleft is called first, which already
/// recomputes Pstring from the OLD Pright, and SetPright then recomputes it again. The
/// intermediate Pstring is never read, but `Pstring = Pleft + Pright` is assigned a third time
/// explicitly afterwards - so the header's inline does the sum twice and then a third time.
/// Written the same way; the result is the same double, and the point of saying so is that a
/// reader comparing this with the header will find the same three assignments.
__host__ __device__ inline void ftf_string_lorentz_rotate(FragmentingString* s,
                                                          const LorentzRot& r) {
  ftf_set_pleft(s, lorentz_apply(r, s->pleft));
  ftf_set_pright(s, lorentz_apply(r, s->pright));
  s->pstring = s->pleft + s->pright;
  s->pt_left = s->pleft.v;
  s->pt_left.z = 0.0;
  s->pt_right = s->pright.v;
  s->pt_right.z = 0.0;
  s->pplus = lv_plus(s->pstring);
  s->pminus = lv_minus(s->pstring);
}

/// G4FragmentingString::TransformToAlignedCms - boost to the string's rest frame, then rotate
/// so that the LEFT parton is along +z. Returns the transform and applies it to the string.
__host__ __device__ inline LorentzRot ftf_transform_to_aligned_cms(FragmentingString* s) {
  const Vec3d bv = s->pstring.boost_vector();
  LorentzRot m = lorentz_boost(Vec3d{-bv.x, -bv.y, -bv.z});
  const Vec4 momentum = lorentz_apply(m, s->pleft);
  lorentz_rotate_z(&m, -1.0 * lv_phi(momentum));
  lorentz_rotate_y(&m, -1.0 * lv_theta(momentum));
  s->pleft = lorentz_apply(m, s->pleft);
  s->pright = lorentz_apply(m, s->pright);
  s->pstring = lorentz_apply(m, s->pstring);
  s->pt_left = s->pleft.v;
  s->pt_left.z = 0.0;
  // `Ptright = G4ThreeVector(Pright.vect());` with NO `setZ(0.)`, unlike every other place
  // the two are maintained - TransformToCenterOfMass, LorentzRotate, SetPright and all four
  // constructors zero it. Transcribed as written. It is dead in QBBC: `Ptleft` and `Ptright`
  // are read only through StablePt() and DecayPt(), and the whole 11.1.1 tree calls DecayPt
  // in exactly one place - G4QGSMFragmentation.cc:504 - and StablePt nowhere. So the Lund
  // fragmentation maintains two members it never reads, one of them inconsistently.
  s->pt_right = s->pright.v;
  s->pplus = lv_plus(s->pstring);
  s->pminus = lv_minus(s->pstring);
  return m;
}

/// G4FragmentingString::TransformToCenterOfMass - the boost without the alignment.
__host__ __device__ inline LorentzRot ftf_transform_to_cms(FragmentingString* s) {
  const Vec3d bv = s->pstring.boost_vector();
  const LorentzRot m = lorentz_boost(Vec3d{-bv.x, -bv.y, -bv.z});
  s->pleft = lorentz_apply(m, s->pleft);
  s->pright = lorentz_apply(m, s->pright);
  s->pstring = lorentz_apply(m, s->pstring);
  s->pt_left = s->pleft.v;
  s->pt_left.z = 0.0;
  s->pt_right = s->pright.v;
  s->pt_right.z = 0.0;
  s->pplus = lv_plus(s->pstring);
  s->pminus = lv_minus(s->pstring);
  return m;
}

}  // namespace g4gpu::hadronic::ftf
