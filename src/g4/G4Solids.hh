// Every concrete solid, with Geant4's constructor signatures.
//
// Include this, or one of the per-solid forwarding headers (G4Box.hh, G4Tubs.hh, ...) that
// exist so that `#include "G4Box.hh"` works the way it does in Geant4.
#pragma once
#include <algorithm>
#include <cstdio>
#include <cstdlib>

#include "g4/G4Material.hh"   // G4VoxelGrid translates its cells through G4Material::device_index
#include "g4/G4VSolid.hh"

// ---------------------------------------------------------------- flat-faced

class G4Box : public G4VSolid {
 public:
  G4Box(const G4String& name, G4double pX, G4double pY, G4double pZ)
      : G4VSolid(name), dx_(pX), dy_(pY), dz_(pZ) {}
  G4int Build(g4gpu::g4::SolidPool& pool) const override {
    return pool.add_solid(make(g4gpu::geom::SolidType::kBox, {dx_, dy_, dz_}));
  }
  G4double Extent() const override { return std::sqrt(dx_ * dx_ + dy_ * dy_ + dz_ * dz_); }
  G4double GetXHalfLength() const { return dx_; }
  G4double GetYHalfLength() const { return dy_; }
  G4double GetZHalfLength() const { return dz_; }

 private:
  G4double dx_, dy_, dz_;
};

class G4Trd : public G4VSolid {
 public:
  G4Trd(const G4String& name, G4double dx1, G4double dx2, G4double dy1, G4double dy2,
        G4double dz)
      : G4VSolid(name), dx1_(dx1), dx2_(dx2), dy1_(dy1), dy2_(dy2), dz_(dz) {}
  G4int Build(g4gpu::g4::SolidPool& pool) const override {
    return pool.add_solid(make(g4gpu::geom::SolidType::kTrd, {dx1_, dx2_, dy1_, dy2_, dz_}));
  }
  G4double Extent() const override {
    const G4double r = std::max(std::max(dx1_, dx2_), std::max(dy1_, dy2_));
    return std::sqrt(2 * r * r + dz_ * dz_);
  }

 private:
  G4double dx1_, dx2_, dy1_, dy2_, dz_;
};

class G4Para : public G4VSolid {
 public:
  G4Para(const G4String& name, G4double dx, G4double dy, G4double dz, G4double alpha,
         G4double theta, G4double phi)
      : G4VSolid(name), dx_(dx), dy_(dy), dz_(dz), alpha_(alpha), theta_(theta), phi_(phi) {}
  G4int Build(g4gpu::g4::SolidPool& pool) const override {
    return pool.add_solid(make(g4gpu::geom::SolidType::kPara,
                               {dx_, dy_, dz_, std::tan(alpha_),
                                std::tan(theta_) * std::cos(phi_),
                                std::tan(theta_) * std::sin(phi_)}));
  }
  G4double Extent() const override {
    return (dx_ + dy_ + dz_) * (1.0 + std::fabs(std::tan(theta_)) + std::fabs(std::tan(alpha_)));
  }

 private:
  G4double dx_, dy_, dz_, alpha_, theta_, phi_;
};

/// A general trapezoid. Its six faces are computed from the eight corners at construction and
/// stored as planes, which is what the device engine consumes.
///
/// Geant4 requires the side faces to be planar and raises a fatal exception when they are not;
/// the same condition is checked here (a non-planar G4Trap is a specification error, not a
/// shape), but it is reported rather than fatal, and the planes are fitted through three
/// corners regardless so that the caller sees something sensible.
class G4Trap : public G4VSolid {
 public:
  G4Trap(const G4String& name, G4double dz, G4double theta, G4double phi, G4double dy1,
         G4double dx1, G4double dx2, G4double alp1, G4double dy2, G4double dx3, G4double dx4,
         G4double alp2)
      : G4VSolid(name), dz_(dz), theta_(theta), phi_(phi), dy1_(dy1), dx1_(dx1), dx2_(dx2),
        alp1_(alp1), dy2_(dy2), dx3_(dx3), dx4_(dx4), alp2_(alp2) {}

  /// The simple constructor: a right trapezoid, no tilts.
  G4Trap(const G4String& name, G4double dz, G4double dy1, G4double dx1, G4double dx2,
         G4double dy2, G4double dx3, G4double dx4)
      : G4Trap(name, dz, 0, 0, dy1, dx1, dx2, 0, dy2, dx3, dx4, 0) {}

  G4int Build(g4gpu::g4::SolidPool& pool) const override {
    G4double planes[24];
    Planes(planes);
    g4gpu::g4::Sol s = make(g4gpu::geom::SolidType::kTrap, {});
    s.a = pool.add_aux(planes, 24);
    s.b = 6;
    return pool.add_solid(s);
  }
  G4double Extent() const override {
    const G4double r = std::max(std::max(dx1_, dx2_), std::max(dx3_, dx4_));
    const G4double q = std::max(dy1_, dy2_);
    return std::sqrt(r * r + q * q + dz_ * dz_)
           + dz_ * std::fabs(std::tan(theta_));
  }

  /// The six face planes, as (nx, ny, nz, d) with the interior at n.x <= d.
  void Planes(G4double* out) const {
    const G4double tthx = std::tan(theta_) * std::cos(phi_);
    const G4double tthy = std::tan(theta_) * std::sin(phi_);
    const G4double ta1 = std::tan(alp1_), ta2 = std::tan(alp2_);
    G4ThreeVector c[8];
    for (int iz = 0; iz < 2; ++iz) {
      const G4double z = iz ? dz_ : -dz_;
      const G4double dy = iz ? dy2_ : dy1_;
      const G4double ta = iz ? ta2 : ta1;
      const G4double xm = iz ? dx3_ : dx1_;
      const G4double xp = iz ? dx4_ : dx2_;
      for (int iy = 0; iy < 2; ++iy) {
        const G4double y = iy ? dy : -dy;
        const G4double hx = iy ? xp : xm;
        for (int ix = 0; ix < 2; ++ix) {
          c[iz + 2 * iy + 4 * ix] =
              G4ThreeVector((ix ? hx : -hx) + ta * y + tthx * z, y + tthy * z, z);
        }
      }
    }
    const int faces[6][3] = {{0, 2, 6}, {1, 5, 7}, {0, 4, 5}, {2, 3, 7}, {0, 1, 3}, {4, 6, 7}};
    for (int f = 0; f < 6; ++f) {
      const G4ThreeVector& a = c[faces[f][0]];
      const G4ThreeVector& b = c[faces[f][1]];
      const G4ThreeVector& d = c[faces[f][2]];
      G4ThreeVector n = (b - a).cross(d - a).unit();
      G4double off = n.dot(a);
      if (-off > 0) { n = -n; off = -off; }  // centre must satisfy n.x <= d
      out[4 * f + 0] = n.x();
      out[4 * f + 1] = n.y();
      out[4 * f + 2] = n.z();
      out[4 * f + 3] = off;
    }
  }

 private:
  G4double dz_, theta_, phi_, dy1_, dx1_, dx2_, alp1_, dy2_, dx3_, dx4_, alp2_;
};

class G4Tet : public G4VSolid {
 public:
  G4Tet(const G4String& name, const G4ThreeVector& p0, const G4ThreeVector& p1,
        const G4ThreeVector& p2, const G4ThreeVector& p3)
      : G4VSolid(name), p0_(p0), p1_(p1), p2_(p2), p3_(p3) {}
  G4int Build(g4gpu::g4::SolidPool& pool) const override {
    const G4ThreeVector tri[4][3] = {{p0_, p1_, p2_}, {p0_, p1_, p3_}, {p0_, p2_, p3_},
                                     {p1_, p2_, p3_}};
    const G4ThreeVector opp[4] = {p3_, p2_, p1_, p0_};
    G4double planes[16];
    for (int f = 0; f < 4; ++f) {
      G4ThreeVector n = (tri[f][1] - tri[f][0]).cross(tri[f][2] - tri[f][0]).unit();
      G4double off = n.dot(tri[f][0]);
      if (n.dot(opp[f]) - off > 0) { n = -n; off = -off; }
      planes[4 * f + 0] = n.x();
      planes[4 * f + 1] = n.y();
      planes[4 * f + 2] = n.z();
      planes[4 * f + 3] = off;
    }
    g4gpu::g4::Sol s = make(g4gpu::geom::SolidType::kTet, {});
    s.a = pool.add_aux(planes, 16);
    s.b = 4;
    return pool.add_solid(s);
  }
  G4double Extent() const override {
    return std::max(std::max(p0_.mag(), p1_.mag()), std::max(p2_.mag(), p3_.mag()));
  }

 private:
  G4ThreeVector p0_, p1_, p2_, p3_;
};

// ---------------------------------------------------------------- curved

class G4Tubs : public G4VSolid {
 public:
  G4Tubs(const G4String& name, G4double rmin, G4double rmax, G4double dz, G4double sphi,
         G4double dphi)
      : G4VSolid(name), rmin_(rmin), rmax_(rmax), dz_(dz), sphi_(sphi), dphi_(dphi) {}
  G4int Build(g4gpu::g4::SolidPool& pool) const override {
    return pool.add_solid(
        make(g4gpu::geom::SolidType::kTubs, {rmin_, rmax_, dz_, sphi_, dphi_}));
  }
  G4double Extent() const override { return std::sqrt(rmax_ * rmax_ + dz_ * dz_); }
  G4double GetInnerRadius() const { return rmin_; }
  G4double GetOuterRadius() const { return rmax_; }
  G4double GetZHalfLength() const { return dz_; }

 private:
  G4double rmin_, rmax_, dz_, sphi_, dphi_;
};

class G4Cons : public G4VSolid {
 public:
  G4Cons(const G4String& name, G4double rmin1, G4double rmax1, G4double rmin2, G4double rmax2,
         G4double dz, G4double sphi, G4double dphi)
      : G4VSolid(name), rmin1_(rmin1), rmax1_(rmax1), rmin2_(rmin2), rmax2_(rmax2), dz_(dz),
        sphi_(sphi), dphi_(dphi) {}
  G4int Build(g4gpu::g4::SolidPool& pool) const override {
    // The B1-validated closed form covers exactly the solid cone with a full phi sweep; use it
    // where it applies, both because it is the faster path and because it is the one the dose
    // comparison exercises.
    const G4bool simple = (rmin1_ == 0 && rmin2_ == 0
                           && dphi_ >= 2 * 3.14159265358979323846 - 1e-12 && sphi_ == 0);
    if (simple) {
      return pool.add_solid(make(g4gpu::geom::SolidType::kCons, {rmax1_, rmax2_, dz_}));
    }
    return pool.add_solid(make(g4gpu::geom::SolidType::kConeSection,
                               {rmin1_, rmax1_, rmin2_, rmax2_, dz_, sphi_, dphi_}));
  }
  G4double Extent() const override {
    const G4double r = std::max(rmax1_, rmax2_);
    return std::sqrt(r * r + dz_ * dz_);
  }

 private:
  G4double rmin1_, rmax1_, rmin2_, rmax2_, dz_, sphi_, dphi_;
};

class G4Orb : public G4VSolid {
 public:
  G4Orb(const G4String& name, G4double r) : G4VSolid(name), r_(r) {}
  G4int Build(g4gpu::g4::SolidPool& pool) const override {
    return pool.add_solid(make(g4gpu::geom::SolidType::kOrb, {r_}));
  }
  G4double Extent() const override { return r_; }

 private:
  G4double r_;
};

class G4Sphere : public G4VSolid {
 public:
  G4Sphere(const G4String& name, G4double rmin, G4double rmax, G4double sphi, G4double dphi,
           G4double stheta, G4double dtheta)
      : G4VSolid(name), rmin_(rmin), rmax_(rmax), sphi_(sphi), dphi_(dphi), stheta_(stheta),
        dtheta_(dtheta) {}
  G4int Build(g4gpu::g4::SolidPool& pool) const override {
    return pool.add_solid(make(g4gpu::geom::SolidType::kSphere,
                               {rmin_, rmax_, sphi_, dphi_, stheta_, dtheta_}));
  }
  G4double Extent() const override { return rmax_; }

 private:
  G4double rmin_, rmax_, sphi_, dphi_, stheta_, dtheta_;
};

class G4Torus : public G4VSolid {
 public:
  G4Torus(const G4String& name, G4double rmin, G4double rmax, G4double rtor, G4double sphi,
          G4double dphi)
      : G4VSolid(name), rmin_(rmin), rmax_(rmax), rtor_(rtor), sphi_(sphi), dphi_(dphi) {}
  G4int Build(g4gpu::g4::SolidPool& pool) const override {
    return pool.add_solid(
        make(g4gpu::geom::SolidType::kTorus, {rmin_, rmax_, rtor_, sphi_, dphi_}));
  }
  G4double Extent() const override { return rtor_ + rmax_; }

 private:
  G4double rmin_, rmax_, rtor_, sphi_, dphi_;
};

class G4EllipticalTube : public G4VSolid {
 public:
  G4EllipticalTube(const G4String& name, G4double dx, G4double dy, G4double dz)
      : G4VSolid(name), dx_(dx), dy_(dy), dz_(dz) {}
  G4int Build(g4gpu::g4::SolidPool& pool) const override {
    return pool.add_solid(make(g4gpu::geom::SolidType::kEllipticalTube, {dx_, dy_, dz_}));
  }
  G4double Extent() const override {
    const G4double r = std::max(dx_, dy_);
    return std::sqrt(r * r + dz_ * dz_);
  }

 private:
  G4double dx_, dy_, dz_;
};

class G4Ellipsoid : public G4VSolid {
 public:
  G4Ellipsoid(const G4String& name, G4double ax, G4double by, G4double cz,
              G4double zcut1 = 0, G4double zcut2 = 0)
      : G4VSolid(name), ax_(ax), by_(by), cz_(cz),
        z1_((zcut1 == 0 && zcut2 == 0) ? -cz : zcut1),
        z2_((zcut1 == 0 && zcut2 == 0) ? cz : zcut2) {}
  G4int Build(g4gpu::g4::SolidPool& pool) const override {
    return pool.add_solid(make(g4gpu::geom::SolidType::kEllipsoid, {ax_, by_, cz_, z1_, z2_}));
  }
  G4double Extent() const override { return std::max(ax_, std::max(by_, cz_)); }

 private:
  G4double ax_, by_, cz_, z1_, z2_;
};

class G4EllipticalCone : public G4VSolid {
 public:
  G4EllipticalCone(const G4String& name, G4double xSemiAxis, G4double ySemiAxis, G4double zMax,
                   G4double zTopCut)
      : G4VSolid(name), ax_(xSemiAxis), ay_(ySemiAxis), zmax_(zMax), zcut_(zTopCut) {}
  G4int Build(g4gpu::g4::SolidPool& pool) const override {
    return pool.add_solid(
        make(g4gpu::geom::SolidType::kEllipticalCone, {ax_, ay_, zmax_, zcut_}));
  }
  G4double Extent() const override {
    const G4double r = std::max(ax_, ay_) * (zmax_ + zcut_);
    return std::sqrt(r * r + zcut_ * zcut_);
  }

 private:
  G4double ax_, ay_, zmax_, zcut_;
};

class G4Paraboloid : public G4VSolid {
 public:
  G4Paraboloid(const G4String& name, G4double dz, G4double rlo, G4double rhi)
      : G4VSolid(name), dz_(dz), rlo_(rlo), rhi_(rhi) {}
  G4int Build(g4gpu::g4::SolidPool& pool) const override {
    return pool.add_solid(make(g4gpu::geom::SolidType::kParaboloid, {dz_, rlo_, rhi_}));
  }
  G4double Extent() const override {
    const G4double r = std::max(rlo_, rhi_);
    return std::sqrt(r * r + dz_ * dz_);
  }

 private:
  G4double dz_, rlo_, rhi_;
};

class G4Hype : public G4VSolid {
 public:
  G4Hype(const G4String& name, G4double rmin, G4double rmax, G4double stIn, G4double stOut,
         G4double halfLenZ)
      : G4VSolid(name), rmin_(rmin), rmax_(rmax), stin_(stIn), stout_(stOut), dz_(halfLenZ) {}
  G4int Build(g4gpu::g4::SolidPool& pool) const override {
    return pool.add_solid(make(g4gpu::geom::SolidType::kHype,
                               {rmin_, rmax_, std::tan(stin_), std::tan(stout_), dz_}));
  }
  G4double Extent() const override {
    const G4double r = rmax_ + std::fabs(std::tan(stout_)) * dz_;
    return std::sqrt(r * r + dz_ * dz_);
  }

 private:
  G4double rmin_, rmax_, stin_, stout_, dz_;
};

// ---------------------------------------------------------------- swept

/// z-sections given as (z, rInner, rOuter) triples, exactly as G4Polycone takes them.
class G4Polycone : public G4VSolid {
 public:
  G4Polycone(const G4String& name, G4double sphi, G4double dphi, G4int numZPlanes,
             const G4double* z, const G4double* rInner, const G4double* rOuter)
      : G4VSolid(name), sphi_(sphi), dphi_(dphi) {
    for (G4int i = 0; i < numZPlanes; ++i) {
      sec_.push_back(z[i]);
      sec_.push_back(rInner[i]);
      sec_.push_back(rOuter[i]);
    }
  }
  G4int Build(g4gpu::g4::SolidPool& pool) const override {
    g4gpu::g4::Sol s = make(g4gpu::geom::SolidType::kPolycone, {sphi_, dphi_});
    s.a = pool.add_aux(sec_.data(), static_cast<int>(sec_.size()));
    s.b = static_cast<int>(sec_.size() / 3) - 1;
    return pool.add_solid(s);
  }
  G4double Extent() const override {
    G4double e = 0;
    for (std::size_t i = 0; i + 2 < sec_.size(); i += 3) {
      e = std::max(e, std::sqrt(sec_[i] * sec_[i] + sec_[i + 2] * sec_[i + 2]));
    }
    return e;
  }

 private:
  G4double sphi_, dphi_;
  std::vector<G4double> sec_;
};

class G4Polyhedra : public G4VSolid {
 public:
  G4Polyhedra(const G4String& name, G4double sphi, G4double dphi, G4int numSide,
              G4int numZPlanes, const G4double* z, const G4double* rInner,
              const G4double* rOuter)
      : G4VSolid(name), sphi_(sphi), dphi_(dphi), sides_(numSide) {
    for (G4int i = 0; i < numZPlanes; ++i) {
      sec_.push_back(z[i]);
      sec_.push_back(rInner[i]);
      sec_.push_back(rOuter[i]);
    }
  }
  G4int Build(g4gpu::g4::SolidPool& pool) const override {
    g4gpu::g4::Sol s =
        make(g4gpu::geom::SolidType::kPolyhedra, {sphi_, dphi_, static_cast<G4double>(sides_)});
    s.a = pool.add_aux(sec_.data(), static_cast<int>(sec_.size()));
    s.b = static_cast<int>(sec_.size() / 3) - 1;
    return pool.add_solid(s);
  }
  G4double Extent() const override {
    G4double e = 0;
    for (std::size_t i = 0; i + 2 < sec_.size(); i += 3) {
      e = std::max(e, std::sqrt(sec_[i] * sec_[i] + sec_[i + 2] * sec_[i + 2]));
    }
    return e / std::cos(3.14159265358979323846 / std::max(3, sides_));
  }

 private:
  G4double sphi_, dphi_;
  G4int sides_;
  std::vector<G4double> sec_;
};

// ---------------------------------------------------------------- booleans

/// Common base for the three boolean solids. The transform is Geant4's: the rotation given is
/// the world-to-solid map, so the second operand ends up rotated by its inverse.
class G4BooleanSolid : public G4VSolid {
 public:
  G4BooleanSolid(const G4String& name, G4VSolid* a, G4VSolid* b, G4RotationMatrix* rot,
                 const G4ThreeVector& trans)
      : G4VSolid(name), a_(a), b_(b), trans_(trans), has_rot_(rot != nullptr) {
    if (rot != nullptr) {
      const G4double* m = rot->data();
      for (int i = 0; i < 9; ++i) { rot_[i] = m[i]; }
    }
  }

  G4double Extent() const override {
    return std::max(a_->Extent(), b_->Extent() + trans_.mag());
  }

 protected:
  G4int BuildWith(g4gpu::geom::SolidType op, g4gpu::g4::SolidPool& pool) const {
    const G4int ia = a_->Build(pool);
    const G4int ib = b_->Build(pool);
    // A mesh operand is refused here rather than produced wrong. The boolean engine decides
    // which intervals along a ray are inside from the list of surface crossings its operands
    // report, and geom::surface_candidates has a fixed capacity that a mesh can exceed - so
    // the result would be a solid of quietly the wrong shape, with nothing to indicate it.
    // Build time is the right place to say so: it is /run/initialize, before any physics.
    for (const G4int child : {ia, ib}) {
      if (pool.solids[child].type != g4gpu::geom::SolidType::kMesh) { continue; }
      std::printf(
          "\nFATAL: \"%s\" uses a tessellated solid as a boolean operand.\n"
          "  A mesh may be placed, and it is transported against its triangles, but it cannot\n"
          "  be unioned, subtracted or intersected here: the boolean engine works from a\n"
          "  bounded list of surface crossings and a mesh can have more of them than fits.\n"
          "  Do the boolean in the CAD tool and import the result, or approximate the mesh\n"
          "  with primitives.\n",
          GetName().c_str());
      std::exit(2);
    }
    g4gpu::g4::Xf x{};
    if (has_rot_) {
      for (int i = 0; i < 9; ++i) { x.rot[i] = rot_[i]; }
      x.identity = false;
    } else {
      x = g4gpu::geom::make_translation<G4double>({trans_.x(), trans_.y(), trans_.z()});
    }
    x.trans = g4gpu::Vec3<G4double>{trans_.x(), trans_.y(), trans_.z()};
    pool.solids[ib].xform = pool.add_xform(x);
    g4gpu::g4::Sol s = make(op, {});
    s.a = ia;
    s.b = ib;
    return pool.add_solid(s);
  }

 private:
  G4VSolid* a_;
  G4VSolid* b_;
  G4ThreeVector trans_;
  G4double rot_[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  G4bool has_rot_;
};

class G4UnionSolid : public G4BooleanSolid {
 public:
  G4UnionSolid(const G4String& name, G4VSolid* a, G4VSolid* b,
               G4RotationMatrix* rot = nullptr, const G4ThreeVector& trans = G4ThreeVector())
      : G4BooleanSolid(name, a, b, rot, trans) {}
  G4int Build(g4gpu::g4::SolidPool& pool) const override {
    return BuildWith(g4gpu::geom::SolidType::kUnion, pool);
  }
};

class G4SubtractionSolid : public G4BooleanSolid {
 public:
  G4SubtractionSolid(const G4String& name, G4VSolid* a, G4VSolid* b,
                     G4RotationMatrix* rot = nullptr,
                     const G4ThreeVector& trans = G4ThreeVector())
      : G4BooleanSolid(name, a, b, rot, trans) {}
  G4int Build(g4gpu::g4::SolidPool& pool) const override {
    return BuildWith(g4gpu::geom::SolidType::kSubtraction, pool);
  }
};

class G4IntersectionSolid : public G4BooleanSolid {
 public:
  G4IntersectionSolid(const G4String& name, G4VSolid* a, G4VSolid* b,
                      G4RotationMatrix* rot = nullptr,
                      const G4ThreeVector& trans = G4ThreeVector())
      : G4BooleanSolid(name, a, b, rot, trans) {}
  G4int Build(g4gpu::g4::SolidPool& pool) const override {
    return BuildWith(g4gpu::geom::SolidType::kIntersection, pool);
  }
};

// ---------------------------------------------------------------- voxel volumes

/// A regular grid of cells, each with its own material.
///
/// Geant4 has no such solid: it expresses a voxelised phantom as a parameterised placement
/// (G4NestedPhantomParameterisation), which is a scheme for making many volumes cheaply. Here
/// it is one *solid* instead, because the layer navigator scans volumes - a 512^3 CT as
/// 134 million volumes would be unusable, whereas as one volume with 134 million cell indices
/// it costs a single entry in the scan and an array lookup per step.
///
/// The materials array is indexed x + nx*(y + ny*z), which is the order a .raw dump and a
/// MATLAB array both use, so an import is a copy rather than a transpose.
class G4VoxelGrid : public G4VSolid {
 public:
  /// @param half_x,half_y,half_z  half extent of the whole grid
  /// @param nx,ny,nz              cell counts
  /// @param materials             nx*ny*nz device material indices, or null for uniform
  G4VoxelGrid(const G4String& name, G4double half_x, G4double half_y, G4double half_z,
              G4int nx, G4int ny, G4int nz, const short* materials = nullptr)
      : G4VSolid(name), hx_(half_x), hy_(half_y), hz_(half_z), nx_(nx), ny_(ny), nz_(nz) {
    const std::size_t n = static_cast<std::size_t>(nx) * ny * nz;
    cells_.assign(n, -1);
    if (materials != nullptr) {
      for (std::size_t i = 0; i < n; ++i) { cells_[i] = materials[i]; }
    }
  }

  /// Sets one cell's material. Used by the importers, which classify after reading.
  void SetCell(G4int i, G4int j, G4int k, short material) {
    if (i < 0 || j < 0 || k < 0 || i >= nx_ || j >= ny_ || k >= nz_) { return; }
    cells_[static_cast<std::size_t>(i) + nx_ * (j + static_cast<std::size_t>(ny_) * k)] =
        material;
  }
  short GetCell(G4int i, G4int j, G4int k) const {
    if (i < 0 || j < 0 || k < 0 || i >= nx_ || j >= ny_ || k >= nz_) { return -1; }
    return cells_[static_cast<std::size_t>(i) + nx_ * (j + static_cast<std::size_t>(ny_) * k)];
  }
  std::vector<short>& Cells() { return cells_; }
  /// The class index per cell, for the renderer. Same length as Cells(), or empty.
  std::vector<short>& ClassCells() { return class_cells_; }

  /// What the numbers in Cells() MEAN, when they are not device material indices.
  ///
  /// Two callers fill a grid and they were numbering its cells differently.
  ///
  /// A hand-built grid (a test, a scene) writes `mat->device_index` straight into a cell,
  /// which is what the transport reads, and leaves this empty.
  ///
  /// A grid built from a model - the builder importing a segmentation - cannot: device
  /// indices do not exist yet. They are handed out in G4Flatten, from the materials of the
  /// PLACED volumes, and Construct() runs first. So the builder wrote its own material
  /// indices and the transport read them as device indices, which agree only when every
  /// model material, in order, is also some placed volume's material. Add one material the
  /// user has not placed anywhere and the numbering slides; assign a class a material no
  /// ordinary volume uses - the normal case for a segmented phantom, where the volume has one
  /// material and its classes have others - and that material is never even built, because
  /// nothing walks voxel classes looking for materials to build.
  ///
  /// What that looked like: a gamma read an out-of-range material and flew through the
  /// phantom depositing nothing, and an electron read one and dumped its whole energy in
  /// three steps. See docs/RISK.md V20.
  ///
  /// So a grid may now say what its cell numbers refer to. Non-empty means Cells() holds
  /// indices into THIS list; Build() translates them to device indices, and G4Flatten builds
  /// every material in it so that they have one.
  void SetCellMaterials(std::vector<G4Material*> mats) { cell_materials_ = std::move(mats); }
  const std::vector<G4Material*>& CellMaterials() const { return cell_materials_; }
  /// One colour per class, 0xAARRGGBB, for the renderer. Empty draws the volume as one box.
  std::vector<unsigned int>& ClassColours() { return class_rgba_; }

  G4int Build(g4gpu::g4::SolidPool& pool) const override {
    g4gpu::g4::Sol s = make(g4gpu::geom::SolidType::kVoxelGrid,
                            {hx_, hy_, hz_, static_cast<G4double>(nx_),
                             static_cast<G4double>(ny_), static_cast<G4double>(nz_)});
    // Cells go to the device as DEVICE material indices, whatever they are numbered as here.
    // See SetCellMaterials.
    std::vector<short> device_cells;
    const short* cells = cells_.data();
    if (!cell_materials_.empty()) {
      device_cells.resize(cells_.size());
      for (std::size_t i = 0; i < cells_.size(); ++i) {
        const short v = cells_[i];
        if (v < 0 || static_cast<std::size_t>(v) >= cell_materials_.size()) {
          device_cells[i] = -1;   // unassigned; material_at falls back to the volume's own
          continue;
        }
        const G4Material* m = cell_materials_[static_cast<std::size_t>(v)];
        if (m == nullptr) {
          device_cells[i] = -1;
          continue;
        }
        if (m->device_index < 0) {
          // Only reachable if G4Flatten did not build this grid's materials before flattening
          // its solid. Fatal rather than silently -1: a class the user assigned, transported
          // as something else, is the failure this whole mechanism exists to end.
          std::printf("\nFATAL: voxel material \"%s\" was never built for the device.\n"
                      "       G4Flatten must build a grid's CellMaterials before its solid.\n",
                      m->GetName().c_str());
          std::exit(2);
        }
        device_cells[i] = static_cast<short>(m->device_index);
      }
      cells = device_cells.data();
    }
    s.a = pool.add_voxels(cells, static_cast<int>(cells_.size()),
                          (class_cells_.size() == cells_.size()) ? class_cells_.data()
                                                                 : nullptr);
    // Where this volume's class colours are, for the renderer to look up per cell. Zero
    // classes leaves p[7] at 0, which is what tells the renderer to draw the box and nothing
    // more - every voxel volume that is not a segmentation, and every one whose values are
    // physical properties rather than indices.
    if (!class_rgba_.empty() && class_cells_.size() == cells_.size()) {
      s.p[6] = static_cast<G4double>(
          pool.add_class_colours(class_rgba_.data(), static_cast<int>(class_rgba_.size())));
      s.p[7] = static_cast<G4double>(class_rgba_.size());
    } else {
      s.p[6] = 0;
      s.p[7] = 0;
    }
    s.b = static_cast<int>(cells_.size());
    return pool.add_solid(s);
  }
  G4double Extent() const override { return std::sqrt(hx_ * hx_ + hy_ * hy_ + hz_ * hz_); }

  G4int GetNx() const { return nx_; }
  G4int GetNy() const { return ny_; }
  G4int GetNz() const { return nz_; }

 private:
  G4double hx_, hy_, hz_;
  G4int nx_, ny_, nz_;
  std::vector<short> cells_;
  /// See SetCellMaterials. Empty means cells_ already holds device indices.
  std::vector<G4Material*> cell_materials_;
  /// Parallel to cells_, holding which voxel class each cell came from. Render-only; see
  /// SolidPool::voxel_class_cells.
  std::vector<short> class_cells_;
  std::vector<unsigned int> class_rgba_;
};

// ---------------------------------------------------------------- tessellated

/// Geant4's flag for whether a facet's vertices are absolute positions or offsets from the
/// first. Both are accepted; RELATIVE is what an exported CAD facet usually is not.
///
/// ABSOLUTE and RELATIVE are macros in wingdi.h - polygon fill modes - so any translation unit
/// that has included windows.h turns this enum into a syntax error. Undefining them here is
/// what G4SystemOfUnits.hh already does for `pascal`, and for the same reason: Geant4's name
/// is the one that has to survive, because it is the one in every existing detector
/// description. Nothing in this project draws polygons through GDI.
#ifdef ABSOLUTE
#undef ABSOLUTE
#endif
#ifdef RELATIVE
#undef RELATIVE
#endif
enum G4FacetVertexType { ABSOLUTE, RELATIVE };

/// Base of the facets a G4TessellatedSolid is built from.
class G4VFacet {
 public:
  virtual ~G4VFacet() = default;
  virtual G4int GetNumberOfVertices() const = 0;
  virtual G4ThreeVector GetVertex(G4int i) const = 0;
  /// Appends this facet's triangles to @p tri, nine reals each.
  virtual void Triangulate(std::vector<G4double>& tri) const = 0;
};

class G4TriangularFacet : public G4VFacet {
 public:
  G4TriangularFacet(const G4ThreeVector& p0, const G4ThreeVector& p1, const G4ThreeVector& p2,
                    G4FacetVertexType type = ABSOLUTE) {
    v_[0] = p0;
    v_[1] = (type == ABSOLUTE) ? p1 : p0 + p1;
    v_[2] = (type == ABSOLUTE) ? p2 : p0 + p2;
  }
  G4int GetNumberOfVertices() const override { return 3; }
  G4ThreeVector GetVertex(G4int i) const override { return v_[(i < 0 || i > 2) ? 0 : i]; }
  void Triangulate(std::vector<G4double>& tri) const override {
    for (int k = 0; k < 3; ++k) {
      tri.push_back(v_[k].x());
      tri.push_back(v_[k].y());
      tri.push_back(v_[k].z());
    }
  }

 private:
  G4ThreeVector v_[3];
};

class G4QuadrangularFacet : public G4VFacet {
 public:
  G4QuadrangularFacet(const G4ThreeVector& p0, const G4ThreeVector& p1, const G4ThreeVector& p2,
                      const G4ThreeVector& p3, G4FacetVertexType type = ABSOLUTE) {
    v_[0] = p0;
    v_[1] = (type == ABSOLUTE) ? p1 : p0 + p1;
    v_[2] = (type == ABSOLUTE) ? p2 : p0 + p2;
    v_[3] = (type == ABSOLUTE) ? p3 : p0 + p3;
  }
  G4int GetNumberOfVertices() const override { return 4; }
  G4ThreeVector GetVertex(G4int i) const override { return v_[(i < 0 || i > 3) ? 0 : i]; }
  /// Fanned from the first vertex: correct for a planar convex quad, which is what a
  /// quadrangular facet is meant to be.
  void Triangulate(std::vector<G4double>& tri) const override {
    const int idx[6] = {0, 1, 2, 0, 2, 3};
    for (int k = 0; k < 6; ++k) {
      tri.push_back(v_[idx[k]].x());
      tri.push_back(v_[idx[k]].y());
      tri.push_back(v_[idx[k]].z());
    }
  }

 private:
  G4ThreeVector v_[4];
};

/// A triangle mesh, transported against its triangles.
///
/// Geant4's usage, which CADMesh.hh also produces:
///
///     auto* solid = new G4TessellatedSolid("part");
///     solid->AddFacet(new G4TriangularFacet(p0, p1, p2, ABSOLUTE));
///     ...
///     solid->SetSolidClosed(true);
///
/// SetSolidClosed(true) is what triggers the work: a BVH over the triangles, the bounding box,
/// and the enclosed volume by the divergence theorem. Geant4 uses the call to finalise its own
/// vertex and edge structures, so an existing detector description already makes it.
///
/// Two limits, both stated rather than worked around:
///
/// The mesh must be closed. Containment is decided by counting ray crossings, and an open
/// surface makes that count meaningless - a point can be "inside" along one ray and outside
/// along another. Nothing here can detect an open mesh reliably (a watertightness check needs
/// edge adjacency, which an STL's duplicated vertices destroy), so this is the caller's to
/// guarantee. A mesh whose volume comes out at zero or at the full bounding box is the usual
/// symptom.
///
/// A tessellated solid may not be an operand of a boolean. The boolean engine works from the
/// list of surface crossings its operands report along a ray, and a mesh can have more than
/// that list holds; truncating it would give a solid that is quietly the wrong shape.
/// G4UnionSolid and its siblings refuse one at construction.
class G4TessellatedSolid : public G4VSolid {
 public:
  explicit G4TessellatedSolid(const G4String& name = "tessellated") : G4VSolid(name) {}

  /// Takes ownership, as Geant4's does.
  G4bool AddFacet(G4VFacet* facet) {
    if (facet == nullptr) { return false; }
    facets_.emplace_back(facet);
    closed_ = false;
    return true;
  }

  /// Adds triangles directly, nine reals each. What the mesh readers and the model builder
  /// use; equivalent to a G4TriangularFacet per triangle without the object per facet.
  void AddTriangles(const std::vector<G4double>& tri) {
    tri_.insert(tri_.end(), tri.begin(), tri.end());
    closed_ = false;
  }

  void SetSolidClosed(G4bool closed) {
    if (!closed) {
      closed_ = false;
      return;
    }
    for (const auto& f : facets_) { f->Triangulate(tri_); }
    facets_.clear();
    if (tri_.size() < 9) {
      std::printf("\nFATAL: G4TessellatedSolid \"%s\" was closed with no facets.\n",
                  GetName().c_str());
      std::exit(2);
    }
    const int n = static_cast<int>(tri_.size() / 9);
    volume_ = g4gpu::geom::mesh_volume(tri_.data(), n);
    area_ = g4gpu::geom::mesh_area(tri_.data(), n);
    for (int k = 0; k < 3; ++k) {
      lo_[k] = tri_[static_cast<std::size_t>(k)];
      hi_[k] = lo_[k];
    }
    for (std::size_t i = 0; i + 2 < tri_.size(); i += 3) {
      for (int k = 0; k < 3; ++k) {
        lo_[k] = std::min(lo_[k], tri_[i + k]);
        hi_[k] = std::max(hi_[k], tri_[i + k]);
      }
    }
    closed_ = true;
  }
  G4bool GetSolidClosed() const { return closed_; }

  G4int GetNumberOfFacets() const {
    return static_cast<G4int>(facets_.size() + tri_.size() / 9);
  }
  /// The enclosed volume, exact for a closed mesh. Zero until SetSolidClosed(true).
  G4double GetCubicVolume() const { return volume_; }
  G4double GetSurfaceArea() const { return area_; }
  G4ThreeVector GetMinExtent() const { return G4ThreeVector(lo_[0], lo_[1], lo_[2]); }
  G4ThreeVector GetMaxExtent() const { return G4ThreeVector(hi_[0], hi_[1], hi_[2]); }
  const std::vector<G4double>& Triangles() const { return tri_; }

  G4int Build(g4gpu::g4::SolidPool& pool) const override {
    if (!closed_) {
      std::printf(
          "\nFATAL: G4TessellatedSolid \"%s\" was placed without SetSolidClosed(true).\n"
          "  Closing it is what builds the BVH and computes the volume; without it the solid\n"
          "  has no surface to intersect and every track would pass straight through.\n",
          GetName().c_str());
      std::exit(2);
    }
    const int n = static_cast<int>(tri_.size() / 9);
    G4double bmin[3] = {0, 0, 0};
    G4double bmax[3] = {0, 0, 0};
    const int root =
        g4gpu::geom::build_bvh(tri_.data(), n, pool.tri, pool.bvh, bmin, bmax);

    g4gpu::g4::Sol s = make(g4gpu::geom::SolidType::kMesh,
                            {0.5 * (bmax[0] - bmin[0]), 0.5 * (bmax[1] - bmin[1]),
                             0.5 * (bmax[2] - bmin[2]), 0.5 * (bmax[0] + bmin[0]),
                             0.5 * (bmax[1] + bmin[1]), 0.5 * (bmax[2] + bmin[2]), volume_});
    s.a = root;
    s.b = n;
    return pool.add_solid(s);
  }

  G4double Extent() const override {
    G4double reach = 0;
    for (int k = 0; k < 3; ++k) {
      reach = std::max(reach, std::max(std::fabs(lo_[k]), std::fabs(hi_[k])));
    }
    return reach * std::sqrt(3.0);
  }

 private:
  std::vector<std::unique_ptr<G4VFacet>> facets_;
  std::vector<G4double> tri_;
  G4double volume_ = 0;
  G4double area_ = 0;
  G4double lo_[3] = {0, 0, 0};
  G4double hi_[3] = {0, 0, 0};
  G4bool closed_ = false;
};
