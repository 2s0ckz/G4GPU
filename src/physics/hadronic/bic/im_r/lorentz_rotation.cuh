// CLHEP::HepLorentzRotation, only the parts the binary cascade's two-body final states use.
//
// Transcribed from CLHEP 2.4.6.0 as shipped inside Geant4 11.1.1
// (source/externals/clhep/src/LorentzRotation.cc, include/CLHEP/Vector/LorentzRotation.icc,
// src/LorentzVectorL.cc): the default constructor, `set(bx,by,bz)`, `rotateY`, `rotateZ`,
// `inverse()`, `matrixMultiplication` and `vectorMultiplication`.
//
// ## Why a 4x4 matrix and not a boost plus two rotations
//
// `G4VElasticCollision::FinalState` and `G4VScatteringCollision::FinalState` both build their
// transform by ACCUMULATING into one matrix:
//
//     G4LorentzRotation toLabFrame(pCM.boostVector());
//     G4LorentzVector Ptmp = toLabFrame.inverse() * trk1.Get4Momentum();
//     G4LorentzRotation toZ;  toZ.rotateZ(-Ptmp.phi());  toZ.rotateY(-Ptmp.theta());
//     toLabFrame *= toZ.inverse();
//     ... p4Final1 *= toLabFrame;
//
// and a boost composed with a rotation is not a boost. Applying the three pieces in sequence to
// the four-vector would give the same answer in exact arithmetic and a different one in double,
// because the sixteen matrix entries are formed once and then reused - so the matrix is what is
// transcribed. The same reasoning docs/RISK.md V67 gives for association order in an expression.
//
// ## The one thing worth knowing about `set`
//
// `mtt = gamma` and `bgamma = gamma*gamma/(1+gamma)`, not the `(gamma-1)/beta^2` a textbook
// writes. They are the same number and the second is singular-free at beta = 0, which is why
// CLHEP writes it that way and why this does too - a cascade nucleon at rest is a boost vector of
// exactly zero and it happens on the first collision of every event.
//
// ## REFUSED, by name
//
// `rotateX`, `boostX`, `boostY`, `boostZ`, `rotate(angle, axis)`, the `HepBoost`/`HepRotation`
// constructors and `decompose` - nothing in the binary cascade's two-body kinematics calls them.
// They are not here rather than here-and-untested.
#ifndef G4GPU_BIC_IMR_LORENTZ_ROTATION_CUH
#define G4GPU_BIC_IMR_LORENTZ_ROTATION_CUH

#include <cmath>

#include "core/vec3.cuh"
#include "physics/hadronic/deexcitation/fragment.cuh"

namespace g4gpu::bic::imr {

using g4gpu::deex::LorentzVector;
using Vec3d = Vec3<double>;

/// `HepLorentzVector::phi()` - `Hep3Vector::phi()` of the spatial part, with CLHEP's guard at
/// the origin. `atan2(0,0)` is 0 on IEEE anyway; the guard is transcribed because CLHEP has it
/// and because a compiler that contracted it away would be making a different function.
__host__ __device__ inline double lv_phi(const LorentzVector& p) {
  if (p.v.x == 0.0 && p.v.y == 0.0) { return 0.0; }
  return std::atan2(p.v.y, p.v.x);
}

/// `HepLorentzVector::theta()` - `atan2(perp(), z)`, zero at the origin.
__host__ __device__ inline double lv_theta(const LorentzVector& p) {
  if (p.v.x == 0.0 && p.v.y == 0.0 && p.v.z == 0.0) { return 0.0; }
  return std::atan2(std::sqrt(p.v.x * p.v.x + p.v.y * p.v.y), p.v.z);
}

/// CLHEP::HepLorentzRotation. Row-major, in CLHEP's own member order: the four rows are
/// (x, y, z, t) and the four columns likewise, so `m[3][3]` is `mtt`.
struct LorentzRotation {
  double m[4][4] = {{1.0, 0.0, 0.0, 0.0},
                    {0.0, 1.0, 0.0, 0.0},
                    {0.0, 0.0, 1.0, 0.0},
                    {0.0, 0.0, 0.0, 1.0}};

  /// `HepLorentzRotation::set(bx, by, bz)`, which is also what the `Hep3Vector` constructor
  /// forwards to. `bp2 >= 1` is a commented-out warning in CLHEP and not a guard, so a boost
  /// vector at or past the speed of light produces a NaN there and produces one here.
  __host__ __device__ static LorentzRotation from_boost(const Vec3d& b) {
    LorentzRotation r;
    const double bp2 = b.x * b.x + b.y * b.y + b.z * b.z;
    const double gamma = 1.0 / std::sqrt(1.0 - bp2);
    const double bgamma = gamma * gamma / (1.0 + gamma);
    r.m[0][0] = 1.0 + bgamma * b.x * b.x;
    r.m[1][1] = 1.0 + bgamma * b.y * b.y;
    r.m[2][2] = 1.0 + bgamma * b.z * b.z;
    r.m[0][1] = r.m[1][0] = bgamma * b.x * b.y;
    r.m[0][2] = r.m[2][0] = bgamma * b.x * b.z;
    r.m[1][2] = r.m[2][1] = bgamma * b.y * b.z;
    r.m[0][3] = r.m[3][0] = gamma * b.x;
    r.m[1][3] = r.m[3][1] = gamma * b.y;
    r.m[2][3] = r.m[3][2] = gamma * b.z;
    r.m[3][3] = gamma;
    return r;
  }

  /// `HepLorentzRotation::rotateY(delta)` - rows 1 and 3 mixed, in CLHEP's sign convention:
  /// `r1 = c*rowx + s*rowz`, `r3 = -s*rowx + c*rowz`. The two rows are read into temporaries
  /// first, because the second assignment would otherwise use the first's result.
  __host__ __device__ void rotate_y(double delta) {
    const double c1 = std::cos(delta);
    const double s1 = std::sin(delta);
    double rx[4], rz[4];
    for (int j = 0; j < 4; ++j) {
      rx[j] = m[0][j];
      rz[j] = m[2][j];
    }
    for (int j = 0; j < 4; ++j) {
      m[0][j] = c1 * rx[j] + s1 * rz[j];
      m[2][j] = -s1 * rx[j] + c1 * rz[j];
    }
  }

  /// `HepLorentzRotation::rotateZ(delta)` - rows 1 and 2, `r1 = c*rowx - s*rowy`,
  /// `r2 = s*rowx + c*rowy`.
  __host__ __device__ void rotate_z(double delta) {
    const double c1 = std::cos(delta);
    const double s1 = std::sin(delta);
    double rx[4], ry[4];
    for (int j = 0; j < 4; ++j) {
      rx[j] = m[0][j];
      ry[j] = m[1][j];
    }
    for (int j = 0; j < 4; ++j) {
      m[0][j] = c1 * rx[j] - s1 * ry[j];
      m[1][j] = s1 * rx[j] + c1 * ry[j];
    }
  }

  /// `HepLorentzRotation::inverse()` - the transpose with the time row and time column negated,
  /// except `tt`. It is the exact inverse of a Lorentz transformation and CLHEP builds it by
  /// permuting the stored entries rather than by solving, so no arithmetic happens here and the
  /// inverse of a matrix is bit-for-bit its entries rearranged.
  __host__ __device__ LorentzRotation inverse() const {
    LorentzRotation r;
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) { r.m[i][j] = m[j][i]; }
    }
    for (int i = 0; i < 3; ++i) {
      r.m[i][3] = -m[3][i];
      r.m[3][i] = -m[i][3];
    }
    r.m[3][3] = m[3][3];
    return r;
  }

  /// `HepLorentzRotation::matrixMultiplication(m1)`, i.e. `(*this) * m1` - and the term order
  /// inside each sum is CLHEP's: `mxx*m1.xx + mxy*m1.yx + mxz*m1.zx + mxt*m1.tx`, left index
  /// running over this matrix's columns in x, y, z, t order.
  __host__ __device__ LorentzRotation operator*(const LorentzRotation& o) const {
    LorentzRotation r;
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        r.m[i][j] = m[i][0] * o.m[0][j] + m[i][1] * o.m[1][j] + m[i][2] * o.m[2][j] +
                    m[i][3] * o.m[3][j];
      }
    }
    return r;
  }

  /// `HepLorentzRotation::vectorMultiplication(p)`, which is what both `operator*(HepLorentzVector)`
  /// and `HepLorentzVector::operator*=(HepLorentzRotation)` call.
  __host__ __device__ LorentzVector operator*(const LorentzVector& p) const {
    const double x = p.v.x, y = p.v.y, z = p.v.z, t = p.e;
    return LorentzVector(m[0][0] * x + m[0][1] * y + m[0][2] * z + m[0][3] * t,
                         m[1][0] * x + m[1][1] * y + m[1][2] * z + m[1][3] * t,
                         m[2][0] * x + m[2][1] * y + m[2][2] * z + m[2][3] * t,
                         m[3][0] * x + m[3][1] * y + m[3][2] * z + m[3][3] * t);
  }
};

}  // namespace g4gpu::bic::imr

#endif
