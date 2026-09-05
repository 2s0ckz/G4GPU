// G4ThreeVector and G4RotationMatrix, the CLHEP types Geant4 uses for geometry.
//
// The accessor spelling matters as much as the arithmetic: Geant4 code says v.x(), v.mag2(),
// v.unit(), rot.rotateY(a). A vector class that only offered .x would compile in this project
// and in no one's existing detector description.
#pragma once
#include <cmath>
#include "g4/G4Types.hh"

class G4ThreeVector {
 public:
  G4ThreeVector() : dx_(0), dy_(0), dz_(0) {}
  G4ThreeVector(G4double x, G4double y, G4double z) : dx_(x), dy_(y), dz_(z) {}

  G4double x() const { return dx_; }
  G4double y() const { return dy_; }
  G4double z() const { return dz_; }
  void setX(G4double v) { dx_ = v; }
  void setY(G4double v) { dy_ = v; }
  void setZ(G4double v) { dz_ = v; }
  void set(G4double x, G4double y, G4double z) { dx_ = x; dy_ = y; dz_ = z; }

  G4double mag2() const { return dx_ * dx_ + dy_ * dy_ + dz_ * dz_; }
  G4double mag() const { return std::sqrt(mag2()); }
  G4double perp2() const { return dx_ * dx_ + dy_ * dy_; }
  G4double perp() const { return std::sqrt(perp2()); }
  G4double theta() const { return std::atan2(perp(), dz_); }
  G4double phi() const { return std::atan2(dy_, dx_); }
  G4double dot(const G4ThreeVector& v) const { return dx_ * v.dx_ + dy_ * v.dy_ + dz_ * v.dz_; }
  G4ThreeVector cross(const G4ThreeVector& v) const {
    return {dy_ * v.dz_ - dz_ * v.dy_, dz_ * v.dx_ - dx_ * v.dz_, dx_ * v.dy_ - dy_ * v.dx_};
  }
  G4ThreeVector unit() const {
    const G4double m2 = mag2();
    if (m2 <= 0) { return {0, 0, 0}; }
    const G4double inv = 1.0 / std::sqrt(m2);
    return {dx_ * inv, dy_ * inv, dz_ * inv};
  }

  G4ThreeVector operator+(const G4ThreeVector& v) const {
    return {dx_ + v.dx_, dy_ + v.dy_, dz_ + v.dz_};
  }
  G4ThreeVector operator-(const G4ThreeVector& v) const {
    return {dx_ - v.dx_, dy_ - v.dy_, dz_ - v.dz_};
  }
  G4ThreeVector operator-() const { return {-dx_, -dy_, -dz_}; }
  G4ThreeVector operator*(G4double a) const { return {dx_ * a, dy_ * a, dz_ * a}; }
  G4ThreeVector operator/(G4double a) const { return {dx_ / a, dy_ / a, dz_ / a}; }
  G4ThreeVector& operator+=(const G4ThreeVector& v) {
    dx_ += v.dx_; dy_ += v.dy_; dz_ += v.dz_;
    return *this;
  }
  G4ThreeVector& operator-=(const G4ThreeVector& v) {
    dx_ -= v.dx_; dy_ -= v.dy_; dz_ -= v.dz_;
    return *this;
  }
  G4ThreeVector& operator*=(G4double a) { dx_ *= a; dy_ *= a; dz_ *= a; return *this; }
  G4bool operator==(const G4ThreeVector& v) const {
    return dx_ == v.dx_ && dy_ == v.dy_ && dz_ == v.dz_;
  }
  G4bool operator!=(const G4ThreeVector& v) const { return !(*this == v); }

 private:
  G4double dx_, dy_, dz_;
};

inline G4ThreeVector operator*(G4double a, const G4ThreeVector& v) { return v * a; }

/// A 3x3 rotation, built the way CLHEP builds one: start from the identity and apply
/// rotateX/rotateY/rotateZ in turn.
///
/// Beware the convention, which this reproduces because Geant4 code depends on it: the matrix
/// handed to G4PVPlacement or a boolean solid is the *world to solid* map, so the solid ends
/// up rotated by its inverse. `rot->rotateY(35*deg)` tilts the frame by +35 degrees, which
/// tilts the object by -35. This is one of the oldest tripwires in Geant4 and it is
/// deliberately not "fixed" here - code moved from Geant4 would then place things differently.
class G4RotationMatrix {
 public:
  G4RotationMatrix() { setIdentity(); }

  /// From nine row-major elements.
  ///
  /// Not part of CLHEP's interface, and here only so that a matrix computed elsewhere - the
  /// anchor chain in builder/model.hh, which has no G4 types - can be handed over without
  /// being decomposed back into three Euler angles and rebuilt, which for a composed rotation
  /// is neither exact nor unique.
  explicit G4RotationMatrix(const G4double* row_major) {
    for (int i = 0; i < 9; ++i) { m_[i] = row_major[i]; }
  }

  void setIdentity() {
    for (int i = 0; i < 9; ++i) { m_[i] = (i % 4 == 0) ? 1.0 : 0.0; }
  }

  G4double xx() const { return m_[0]; }
  G4double xy() const { return m_[1]; }
  G4double xz() const { return m_[2]; }
  G4double yx() const { return m_[3]; }
  G4double yy() const { return m_[4]; }
  G4double yz() const { return m_[5]; }
  G4double zx() const { return m_[6]; }
  G4double zy() const { return m_[7]; }
  G4double zz() const { return m_[8]; }

  G4RotationMatrix& rotateX(G4double a) {
    const G4double c = std::cos(a), s = std::sin(a);
    const G4double r[9] = {1, 0, 0, 0, c, -s, 0, s, c};
    return preMultiply(r);
  }
  G4RotationMatrix& rotateY(G4double a) {
    const G4double c = std::cos(a), s = std::sin(a);
    const G4double r[9] = {c, 0, s, 0, 1, 0, -s, 0, c};
    return preMultiply(r);
  }
  G4RotationMatrix& rotateZ(G4double a) {
    const G4double c = std::cos(a), s = std::sin(a);
    const G4double r[9] = {c, -s, 0, s, c, 0, 0, 0, 1};
    return preMultiply(r);
  }

  G4RotationMatrix inverse() const {
    G4RotationMatrix r;
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) { r.m_[3 * i + j] = m_[3 * j + i]; }
    }
    return r;
  }

  G4ThreeVector operator*(const G4ThreeVector& v) const {
    return {m_[0] * v.x() + m_[1] * v.y() + m_[2] * v.z(),
            m_[3] * v.x() + m_[4] * v.y() + m_[5] * v.z(),
            m_[6] * v.x() + m_[7] * v.y() + m_[8] * v.z()};
  }

  /// Ordinary matrix product, so `(a * b) * v` is `a * (b * v)` as CLHEP's is.
  ///
  /// Needed to compose a chain of placements - a volume anchored to another, itself anchored
  /// to a third - into the single world-to-solid map G4PVPlacement takes.
  G4RotationMatrix operator*(const G4RotationMatrix& o) const {
    G4RotationMatrix r;
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        r.m_[3 * i + j] = m_[3 * i + 0] * o.m_[0 * 3 + j] + m_[3 * i + 1] * o.m_[1 * 3 + j]
                          + m_[3 * i + 2] * o.m_[2 * 3 + j];
      }
    }
    return r;
  }

  /// Row-major access, for the flattener.
  const G4double* data() const { return m_; }

 private:
  G4RotationMatrix& preMultiply(const G4double* r) {
    G4double out[9];
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        out[3 * i + j] = r[3 * i + 0] * m_[0 * 3 + j] + r[3 * i + 1] * m_[1 * 3 + j]
                         + r[3 * i + 2] * m_[2 * 3 + j];
      }
    }
    for (int i = 0; i < 9; ++i) { m_[i] = out[i]; }
    return *this;
  }
  G4double m_[9];
};

using G4Transform3D = G4RotationMatrix;  // placeholder alias; full type not needed yet
