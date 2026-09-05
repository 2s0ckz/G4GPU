// Rigid placement of a solid in the world.
//
// Stored as the global -> local map directly, because that is the direction every navigation
// query needs: local = rot * (global - trans). Geant4 stores a G4RotationMatrix describing
// the daughter frame and inverts it at use; keeping the inverse is the same information with
// one fewer transpose in the inner loop.
#pragma once
#include <cmath>
#include "core/vec3.cuh"

namespace g4gpu::geom {

/// Row-major 3x3 rotation plus a translation. `identity` lets the common unrotated case skip
/// nine multiplies per query, which matters because this sits inside the stepping loop.
template <typename real_t>
struct Transform {
  real_t rot[9];
  Vec3<real_t> trans;
  bool identity;
};

template <typename real_t>
__host__ __device__ inline Transform<real_t> make_translation(const Vec3<real_t>& t) {
  Transform<real_t> x{};
  x.rot[0] = real_t(1); x.rot[1] = real_t(0); x.rot[2] = real_t(0);
  x.rot[3] = real_t(0); x.rot[4] = real_t(1); x.rot[5] = real_t(0);
  x.rot[6] = real_t(0); x.rot[7] = real_t(0); x.rot[8] = real_t(1);
  x.trans = t;
  x.identity = true;
  return x;
}

/// Rotation by `deg_x`, `deg_y`, `deg_z` about the global axes, applied X then Y then Z to
/// the *solid*, followed by a translation. The stored matrix is the inverse of that, so a
/// global point maps straight into solid-local coordinates.
template <typename real_t>
__host__ __device__ inline Transform<real_t> make_placement(const Vec3<real_t>& t, real_t rad_x,
                                                            real_t rad_y, real_t rad_z) {
  const real_t cx = cos(rad_x), sx = sin(rad_x);
  const real_t cy = cos(rad_y), sy = sin(rad_y);
  const real_t cz = cos(rad_z), sz = sin(rad_z);

  // Forward rotation R = Rz * Ry * Rx (solid frame -> world frame).
  const real_t r00 = cz * cy;
  const real_t r01 = cz * sy * sx - sz * cx;
  const real_t r02 = cz * sy * cx + sz * sx;
  const real_t r10 = sz * cy;
  const real_t r11 = sz * sy * sx + cz * cx;
  const real_t r12 = sz * sy * cx - cz * sx;
  const real_t r20 = -sy;
  const real_t r21 = cy * sx;
  const real_t r22 = cy * cx;

  Transform<real_t> x{};
  // Stored matrix is R^T, the world -> solid map. R is orthonormal, so the transpose is the
  // inverse exactly, with no accumulation of round-off from a general inversion.
  x.rot[0] = r00; x.rot[1] = r10; x.rot[2] = r20;
  x.rot[3] = r01; x.rot[4] = r11; x.rot[5] = r21;
  x.rot[6] = r02; x.rot[7] = r12; x.rot[8] = r22;
  x.trans = t;
  x.identity = (rad_x == real_t(0) && rad_y == real_t(0) && rad_z == real_t(0));
  return x;
}

/// Global point -> solid-local point.
template <typename real_t>
__host__ __device__ inline Vec3<real_t> to_local(const Transform<real_t>& x,
                                                 const Vec3<real_t>& p) {
  const Vec3<real_t> d = p - x.trans;
  if (x.identity) { return d; }
  return Vec3<real_t>{x.rot[0] * d.x + x.rot[1] * d.y + x.rot[2] * d.z,
                      x.rot[3] * d.x + x.rot[4] * d.y + x.rot[5] * d.z,
                      x.rot[6] * d.x + x.rot[7] * d.y + x.rot[8] * d.z};
}

/// Global direction -> solid-local direction. No translation, and no renormalisation: the
/// matrix is orthonormal, so a unit direction stays unit.
template <typename real_t>
__host__ __device__ inline Vec3<real_t> dir_to_local(const Transform<real_t>& x,
                                                     const Vec3<real_t>& d) {
  if (x.identity) { return d; }
  return Vec3<real_t>{x.rot[0] * d.x + x.rot[1] * d.y + x.rot[2] * d.z,
                      x.rot[3] * d.x + x.rot[4] * d.y + x.rot[5] * d.z,
                      x.rot[6] * d.x + x.rot[7] * d.y + x.rot[8] * d.z};
}

/// Solid-local direction (e.g. a surface normal) -> global direction.
template <typename real_t>
__host__ __device__ inline Vec3<real_t> dir_to_global(const Transform<real_t>& x,
                                                      const Vec3<real_t>& d) {
  if (x.identity) { return d; }
  return Vec3<real_t>{x.rot[0] * d.x + x.rot[3] * d.y + x.rot[6] * d.z,
                      x.rot[1] * d.x + x.rot[4] * d.y + x.rot[7] * d.z,
                      x.rot[2] * d.x + x.rot[5] * d.y + x.rot[8] * d.z};
}

}  // namespace g4gpu::geom
