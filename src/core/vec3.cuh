#pragma once
#include <cmath>
namespace g4gpu {

template <typename T>
struct Vec3 {
  T x, y, z;
};

template <typename T> __host__ __device__ inline Vec3<T> operator+(const Vec3<T>& a, const Vec3<T>& b)
{ return {a.x + b.x, a.y + b.y, a.z + b.z}; }
template <typename T> __host__ __device__ inline Vec3<T> operator-(const Vec3<T>& a, const Vec3<T>& b)
{ return {a.x - b.x, a.y - b.y, a.z - b.z}; }
template <typename T> __host__ __device__ inline Vec3<T> operator*(T s, const Vec3<T>& v)
{ return {s * v.x, s * v.y, s * v.z}; }
template <typename T> __host__ __device__ inline Vec3<T> operator*(const Vec3<T>& v, T s)
{ return {s * v.x, s * v.y, s * v.z}; }
template <typename T> __host__ __device__ inline T dot(const Vec3<T>& a, const Vec3<T>& b)
{ return a.x * b.x + a.y * b.y + a.z * b.z; }
template <typename T> __host__ __device__ inline Vec3<T> cross(const Vec3<T>& a, const Vec3<T>& b)
{ return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
template <typename T> __host__ __device__ inline T mag2(const Vec3<T>& v) { return dot(v, v); }
template <typename T> __host__ __device__ inline T mag(const Vec3<T>& v) { return sqrt(dot(v, v)); }

template <typename T> __host__ __device__ inline Vec3<T> normalize(const Vec3<T>& v) {
  const T m = mag(v);
  return (m > T(0)) ? Vec3<T>{v.x / m, v.y / m, v.z / m} : Vec3<T>{T(0), T(0), T(1)};
}

/// Rotates a vector expressed in a frame whose z-axis is @p u into the global frame.
/// Faithful to CLHEP Hep3Vector::rotateUz, including the u3<0 degenerate branch.
template <typename T>
__host__ __device__ inline Vec3<T> rotate_uz(const Vec3<T>& v, const Vec3<T>& u) {
  const T u1 = u.x, u2 = u.y, u3 = u.z;
  T up = u1 * u1 + u2 * u2;
  if (up > T(0)) {
    up = sqrt(up);
    const T px = v.x, py = v.y, pz = v.z;
    return {(u1 * u3 * px - u2 * py) / up + u1 * pz,
            (u2 * u3 * px + u1 * py) / up + u2 * pz,
            -up * px + u3 * pz};
  }
  if (u3 < T(0)) { return {-v.x, v.y, -v.z}; }
  return v;
}

}  // namespace g4gpu
