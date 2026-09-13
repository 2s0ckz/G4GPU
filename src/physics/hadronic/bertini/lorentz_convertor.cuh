// G4LorentzConvertor: the frame transformation every Bertini collision goes through.
//
// Transcribed from Geant4 11.1.1:
//   G4LorentzConvertor::toTheCenterOfMass / toTheTargetRestFrame / fillKinematics /
//     backToTheLab / getKinEnergyInTheTRS / getTRSMomentum / rotate (both overloads) /
//     reflectionNeeded            (cascade/cascade/src/G4LorentzConvertor.cc)
//
// It is a small class with one subtlety that matters everywhere: **`scm_momentum` means two
// different things depending on which frame was selected.** After `toTheCenterOfMass` it is the
// REVERSED TARGET momentum in the CM frame; after `toTheTargetRestFrame` it is the BULLET
// momentum in the target's frame. Both are then used as "the reference z axis" by `rotate`, so
// the axis a final state is built around flips sign between the two calls. Geant4's own comments
// say so ("SCM is reverse target momentum in the CM frame" / "SCM is bullet momentum in the
// target's frame") and the second one calls the result a "pseudo-pscm".
//
// `small` is 1e-10 and it gates three different decisions: whether the boost is worth applying
// in `backToTheLab`, whether the frame is `degenerated` (already along z, so `rotate` is the
// identity), and whether `reflectionNeeded` can answer at all - it THROWS when `v2 < small` and
// the frame is not degenerate, which is a state the caller is supposed to have made impossible.
// A kernel cannot throw, so `reflection_needed` returns a verdict plus an `undefined` flag and
// the caller reports it.
#ifndef G4GPU_BERTINI_LORENTZ_CONVERTOR_CUH
#define G4GPU_BERTINI_LORENTZ_CONVERTOR_CUH

#include <cmath>

#include "core/vec3.cuh"
#include "physics/hadronic/deexcitation/fragment.cuh"

namespace g4gpu::physics::hadronic::bert {

/// P3 named the two types this module lives on; they are used unqualified here so that the
/// transcribed formulas read like the Geant4 they came from.
using LV = deex::LorentzVector;
using deex::Vec3d;

/// `HepLorentzVector::setVectM(p, m)` - energy from the mass shell, which is how every INUCL
/// class builds a four-vector from a three-momentum. Not a member of P3's LorentzVector, so it
/// is a free function here rather than an edit to a file this package does not own.
__host__ __device__ inline LV lv_set_vect_m(const Vec3d& p, double m) {
  return LV(p, std::sqrt(g4gpu::mag2(p) + m * m));
}

/// `HepLorentzVector::rho()` - the magnitude of the three-momentum.
__host__ __device__ inline double lv_rho(const LV& p) { return g4gpu::mag(p.v); }

/// G4LorentzConvertor's state, as a plain struct: a device port cannot carry a stateful
/// converter object around, and every call site in the INUCL tree does
/// `setBullet; setTarget; toTheCenterOfMass; ...` in that order, so the state is built in one
/// place and read in several.
struct LorentzConvertor {
  LV bullet;
  LV target;
  LV scm;               ///< see the header comment: its MEANING depends on the frame chosen
  Vec3d scm_dir{0.0, 0.0, 1.0};
  Vec3d velocity{0.0, 0.0, 0.0};
  double v2 = 0.0;
  double ecm_tot = 0.0;
  double valong = 0.0;
  bool degenerated = false;
};

/// G4LorentzConvertor::small.
constexpr double kLcSmall = 1.0e-10;

/// G4LorentzConvertor::fillKinematics.
__host__ __device__ inline void lc_fill_kinematics(LorentzConvertor& lc) {
  lc.ecm_tot = (lc.target + lc.bullet).mag();
  lc.scm_dir = g4gpu::normalize(lc.scm.v);
  lc.valong = g4gpu::dot(lc.velocity, lc.scm_dir);
  lc.v2 = g4gpu::mag2(lc.velocity);
  const double pvsq = lc.v2 - lc.valong * lc.valong;
  lc.degenerated = (pvsq < kLcSmall);
}

/// G4LorentzConvertor::toTheCenterOfMass.
__host__ __device__ inline void lc_to_the_center_of_mass(LorentzConvertor& lc) {
  lc.velocity = (lc.target + lc.bullet).boost_vector();
  lc.scm = lc.target;
  lc.scm.boost(Vec3d{-lc.velocity.x, -lc.velocity.y, -lc.velocity.z});
  lc.scm.v = Vec3d{-lc.scm.v.x, -lc.scm.v.y, -lc.scm.v.z};
  lc_fill_kinematics(lc);
}

/// G4LorentzConvertor::toTheTargetRestFrame.
__host__ __device__ inline void lc_to_the_target_rest_frame(LorentzConvertor& lc) {
  lc.velocity = lc.target.boost_vector();
  lc.scm = lc.bullet;
  lc.scm.boost(Vec3d{-lc.velocity.x, -lc.velocity.y, -lc.velocity.z});
  lc_fill_kinematics(lc);
}

/// G4LorentzConvertor::backToTheLab. The `v2 > small` guard is Geant4's: below it the boost is
/// skipped entirely rather than applied as a near-identity, so the answer is bit-for-bit the
/// input and not the input plus rounding.
__host__ __device__ inline LV lc_back_to_the_lab(const LorentzConvertor& lc, const LV& mom) {
  LV out = mom;
  if (lc.v2 > kLcSmall) { out.boost(lc.velocity); }
  return out;
}

/// G4LorentzConvertor::getKinEnergyInTheTRS. Note it does NOT read the stored frame: it boosts
/// the bullet into the target's rest frame from scratch, so it gives the same answer whether
/// `toTheCenterOfMass` or `toTheTargetRestFrame` was called - or neither.
__host__ __device__ inline double lc_kin_energy_in_trs(const LorentzConvertor& lc) {
  LV b = lc.bullet;
  const Vec3d bv = lc.target.boost_vector();
  b.boost(Vec3d{-bv.x, -bv.y, -bv.z});
  return b.e - b.mag();
}

__host__ __device__ inline double lc_trs_momentum(const LorentzConvertor& lc) {
  LV b = lc.bullet;
  const Vec3d bv = lc.target.boost_vector();
  b.boost(Vec3d{-bv.x, -bv.y, -bv.z});
  return lv_rho(b);
}

__host__ __device__ inline double lc_scm_momentum(const LorentzConvertor& lc) {
  return lv_rho(lc.scm);
}

/// G4LorentzConvertor::rotate(mom) - map a four-vector built about +z onto the collision's own
/// axes. The double check on `vscm.mag()` and `vxcm.mag()` is Geant4's; when it fails Geant4
/// prints and returns the UNROTATED vector, which is what this does.
__host__ __device__ inline LV lc_rotate(const LorentzConvertor& lc, const LV& mom) {
  LV out = mom;
  if (lc.degenerated) { return out; }
  const Vec3d vscm = lc.velocity - lc.valong * lc.scm_dir;
  const Vec3d vxcm = g4gpu::cross(lc.scm_dir, lc.velocity);
  if (g4gpu::mag(vscm) > kLcSmall && g4gpu::mag(vxcm) > kLcSmall) {
    out.v = mom.v.x * g4gpu::normalize(vscm) + mom.v.y * g4gpu::normalize(vxcm) +
            mom.v.z * lc.scm_dir;
  }
  return out;
}

/// G4LorentzConvertor::rotate(mom1, mom) - the same, about an arbitrary first axis. Its guard
/// is `vperp > small` rather than `!degenerated`, so the two overloads can disagree about
/// whether to rotate for the same converter state.
__host__ __device__ inline LV lc_rotate_about(const LorentzConvertor& lc, const LV& mom1,
                                              const LV& mom) {
  LV out = mom;
  const Vec3d dir = g4gpu::normalize(mom1.v);
  const double pv = g4gpu::dot(lc.velocity, dir);
  const double vperp = lc.v2 - pv * pv;
  if (vperp > kLcSmall) {
    const Vec3d vmom1 = lc.velocity - pv * dir;
    const Vec3d vxm1 = g4gpu::cross(dir, lc.velocity);
    if (g4gpu::mag(vmom1) > kLcSmall && g4gpu::mag(vxm1) > kLcSmall) {
      out.v = mom.v.x * g4gpu::normalize(vmom1) + mom.v.y * g4gpu::normalize(vxm1) +
              mom.v.z * dir;
    }
  }
  return out;
}

/// G4LorentzConvertor::reflectionNeeded. Geant4 THROWS a G4HadronicException when
/// `v2 < small && !degenerated` - "return value undefined" - so `undefined` carries that state
/// out instead of ending the event, and the caller reports it.
__host__ __device__ inline bool lc_reflection_needed(const LorentzConvertor& lc,
                                                     bool& undefined) {
  undefined = (lc.v2 < kLcSmall && !lc.degenerated);
  return (lc.v2 >= kLcSmall && (!lc.degenerated || lc.scm.v.z < 0.0));
}

}  // namespace g4gpu::physics::hadronic::bert

#endif  // G4GPU_BERTINI_LORENTZ_CONVERTOR_CUH
