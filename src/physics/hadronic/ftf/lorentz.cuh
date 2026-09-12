// The four-vector operations and the 4x4 Lorentz rotation the string decay is written in.
//
// Transcribed from CLHEP 2.4.6.2 as Geant4 11.1.1 ships it
// (source/externals/clhep/src/LorentzRotation.cc, include/CLHEP/Vector/LorentzRotation.icc,
// include/CLHEP/Vector/ThreeVector.icc, include/CLHEP/Vector/LorentzVector.icc):
//   HepLorentzRotation::set(bx,by,bz), rotateY, rotateZ, inverse, matrixMultiplication,
//   vectorMultiplication;  Hep3Vector::phi, theta;  HepLorentzVector::plus, minus, perp2.
//
// WHY A MATRIX AND NOT boost(). The de-excitation module's four-vector
// (deexcitation/fragment.cuh) already has `boost()` and `boost_vector()`, and this file reuses
// that type rather than defining a second one - what P6's hand-over reads is
// `deex::LorentzVector`, so a separate FTF four-vector would need a conversion at exactly the
// place a rounding difference would be invisible.
//
// But the string decay does NOT boost: it builds a G4LorentzRotation, composes a boost with two
// rotations into it, inverts it, and applies the same matrix to several vectors
// (G4FragmentingString::TransformToAlignedCms, G4LundStringFragmentation::SplitLast,
// Loop_toFragmentString). Composing a boost and two rotations into a matrix and then applying
// it is NOT the same arithmetic as boosting and then rotating - the products are accumulated
// in a different order - and the difference is at the last bit of every hadron momentum, which
// is exactly the size ref/oracle/ftf_fragment.csv compares at. So the matrix is transcribed.
//
// THE ONE PLACE CLHEP'S OWN GUARD IS COMMENTED OUT. `HepLorentzRotation::set` computes
// `gamma = 1/sqrt(1 - b^2)` with the `b^2 >= 1` diagnostic commented out in the CLHEP source
// shipped with 11.1.1, so a boost vector at or above the speed of light gives a NaN rather
// than a message. Reproduced: a NaN that propagates is what Geant4 does, and inventing a clamp
// here would hide the caller that produced it. The callers in this package cannot produce one -
// every boost vector is `P/E` of a timelike four-momentum.
#pragma once
#include <cmath>

#include "core/units.cuh"
#include "core/vec3.cuh"
#include "physics/hadronic/deexcitation/fragment.cuh"

namespace g4gpu::hadronic::ftf {

using Vec4 = deex::LorentzVector;
using Vec3d = deex::Vec3d;

/// HepLorentzVector::plus() = t + z and minus() = t - z, the light-cone components the string
/// decay's whole longitudinal bookkeeping is in.
__host__ __device__ inline double lv_plus(const Vec4& p) { return p.e + p.v.z; }
__host__ __device__ inline double lv_minus(const Vec4& p) { return p.e - p.v.z; }

/// Hep3Vector::perp2 / HepLorentzVector::perp2 - the transverse momentum squared.
__host__ __device__ inline double lv_perp2(const Vec3d& v) { return v.x * v.x + v.y * v.y; }
__host__ __device__ inline double lv_perp(const Vec3d& v) { return std::sqrt(lv_perp2(v)); }

/// Hep3Vector::phi() - `atan2(y, x)`, and EXACTLY 0 when x and y are both zero rather than
/// whatever atan2(0,0) gives. The guard is CLHEP's, and it matters: SplitLast rotates by
/// -Pleft.phi() and a string aligned with z has Pt = 0 there.
__host__ __device__ inline double lv_phi(const Vec3d& v) {
  return (v.x == 0.0 && v.y == 0.0) ? 0.0 : std::atan2(v.y, v.x);
}

/// Hep3Vector::theta() - `atan2(perp(), z)`, and exactly 0 for the null vector.
__host__ __device__ inline double lv_theta(const Vec3d& v) {
  return (v.x == 0.0 && v.y == 0.0 && v.z == 0.0) ? 0.0 : std::atan2(lv_perp(v), v.z);
}

__host__ __device__ inline double lv_phi(const Vec4& p) { return lv_phi(p.v); }
__host__ __device__ inline double lv_theta(const Vec4& p) { return lv_theta(p.v); }

/// CLHEP::HepLorentzRotation, in CLHEP's own member order: rows x, y, z, t.
struct LorentzRot {
  double xx = 1, xy = 0, xz = 0, xt = 0;
  double yx = 0, yy = 1, yz = 0, yt = 0;
  double zx = 0, zy = 0, zz = 1, zt = 0;
  double tx = 0, ty = 0, tz = 0, tt = 1;
};

/// HepLorentzRotation::set(bx, by, bz) - a pure boost. `bgamma = gamma^2/(1+gamma)` is CLHEP's
/// singular-free form of `(gamma-1)/b^2`.
__host__ __device__ inline LorentzRot lorentz_boost(const Vec3d& b) {
  LorentzRot m;
  const double bp2 = b.x * b.x + b.y * b.y + b.z * b.z;
  const double gamma = 1.0 / std::sqrt(1.0 - bp2);
  const double bgamma = gamma * gamma / (1.0 + gamma);
  m.xx = 1.0 + bgamma * b.x * b.x;
  m.yy = 1.0 + bgamma * b.y * b.y;
  m.zz = 1.0 + bgamma * b.z * b.z;
  m.xy = m.yx = bgamma * b.x * b.y;
  m.xz = m.zx = bgamma * b.x * b.z;
  m.yz = m.zy = bgamma * b.y * b.z;
  m.xt = m.tx = gamma * b.x;
  m.yt = m.ty = gamma * b.y;
  m.zt = m.tz = gamma * b.z;
  m.tt = gamma;
  return m;
}

/// HepLorentzRotation::rotateZ(delta) - LEFT-multiplication by R_z(delta), written as CLHEP
/// writes it: rows x and y are replaced by `c*rowx - s*rowy` and `s*rowx + c*rowy`.
__host__ __device__ inline void lorentz_rotate_z(LorentzRot* m, double delta) {
  const double c1 = std::cos(delta);
  const double s1 = std::sin(delta);
  const double r1x = c1 * m->xx - s1 * m->yx;
  const double r1y = c1 * m->xy - s1 * m->yy;
  const double r1z = c1 * m->xz - s1 * m->yz;
  const double r1t = c1 * m->xt - s1 * m->yt;
  const double r2x = s1 * m->xx + c1 * m->yx;
  const double r2y = s1 * m->xy + c1 * m->yy;
  const double r2z = s1 * m->xz + c1 * m->yz;
  const double r2t = s1 * m->xt + c1 * m->yt;
  m->xx = r1x; m->xy = r1y; m->xz = r1z; m->xt = r1t;
  m->yx = r2x; m->yy = r2y; m->yz = r2z; m->yt = r2t;
}

/// HepLorentzRotation::rotateY(delta). Note the SIGNS differ from rotateZ's: rows x and z
/// become `c*rowx + s*rowz` and `-s*rowx + c*rowz`.
__host__ __device__ inline void lorentz_rotate_y(LorentzRot* m, double delta) {
  const double c1 = std::cos(delta);
  const double s1 = std::sin(delta);
  const double r1x = c1 * m->xx + s1 * m->zx;
  const double r1y = c1 * m->xy + s1 * m->zy;
  const double r1z = c1 * m->xz + s1 * m->zz;
  const double r1t = c1 * m->xt + s1 * m->zt;
  const double r3x = -s1 * m->xx + c1 * m->zx;
  const double r3y = -s1 * m->xy + c1 * m->zy;
  const double r3z = -s1 * m->xz + c1 * m->zz;
  const double r3t = -s1 * m->xt + c1 * m->zt;
  m->xx = r1x; m->xy = r1y; m->xz = r1z; m->xt = r1t;
  m->zx = r3x; m->zy = r3y; m->zz = r3z; m->zt = r3t;
}

/// HepLorentzRotation::inverse() - the metric-adjoint, not a numerical inverse: transpose the
/// spatial block and negate the mixed space-time entries.
__host__ __device__ inline LorentzRot lorentz_inverse(const LorentzRot& m) {
  LorentzRot r;
  r.xx = m.xx;  r.xy = m.yx;  r.xz = m.zx;  r.xt = -m.tx;
  r.yx = m.xy;  r.yy = m.yy;  r.yz = m.zy;  r.yt = -m.ty;
  r.zx = m.xz;  r.zy = m.yz;  r.zz = m.zz;  r.zt = -m.tz;
  r.tx = -m.xt; r.ty = -m.yt; r.tz = -m.zt; r.tt = m.tt;
  return r;
}

/// HepLorentzRotation::vectorMultiplication - `m * p`.
__host__ __device__ inline Vec4 lorentz_apply(const LorentzRot& m, const Vec4& p) {
  const double x = p.v.x, y = p.v.y, z = p.v.z, t = p.e;
  return Vec4(m.xx * x + m.xy * y + m.xz * z + m.xt * t,
              m.yx * x + m.yy * y + m.yz * z + m.yt * t,
              m.zx * x + m.zy * y + m.zz * z + m.zt * t,
              m.tx * x + m.ty * y + m.tz * z + m.tt * t);
}

}  // namespace g4gpu::hadronic::ftf
