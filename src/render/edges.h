// The wireframe edge list, built on the host and drawn by vis::render_edges.
//
// SHARED BY THE VIEWER AND THE BUILDER, and it was not: this lived inside vis_manager.cu, so
// the builder had no way to draw an edge and never launched the pass. Its own styles said
// `solid = visible && !wireframe`, which is right - a wireframe volume must not be ray cast as
// a surface - so a volume set to wireframe there was simply invisible, and the default world,
// which arrives with wireframe on, had no outline at all.
//
// One implementation rather than two, for the reason that keeps coming up in this project: a
// second copy is a second thing to fix, and the one nobody remembers is the one that is wrong.
//
// EVERY SOLID, NOT JUST BOXES. The first version emitted the twelve edges of a box and nothing
// else, behind a comment of mine claiming a wireframe for a curved solid would have to be a
// bounding cube and so "a lie about its shape". That was an argument against drawing a CUBE,
// not an argument for drawing NOTHING - and the consequence was that a sphere set to wireframe
// disappeared, which is what "wireframe is not working for round primitives" was. A sphere has
// a wireframe, and it is the one every CAD tool draws: rings of latitude and a few meridians.
//
// Almost all of it is one function. A cylinder, a cone, a sphere, an ellipsoid, a paraboloid
// and a hyperboloid are all surfaces of revolution about z, so each is a table of (z, rx, ry)
// levels handed to AddProfile - rings at the levels, meridians between them - and they differ
// only in the table they build. The exceptions are the flat-faced solids: kTrd has its eight
// corners in p[] directly, and kPara, kTrap and kTet arrive as HALF-SPACES rather than as
// dimensions, so their corners have to be recovered by intersecting plane triples. See
// AddPlanes.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "core/vec3.cuh"
#include "geometry/navigator.cuh"
#include "geometry/transform.cuh"

namespace g4gpu::vis {

/// Line segments in world coordinates, one colour each, as vis::render_edges wants them: six
/// parallel float arrays rather than an array of structs, so the pass reads them coalesced.
struct EdgeList {
  std::vector<float> x0, y0, z0, x1, y1, z1;
  std::vector<unsigned int> rgb;

  /// How finely a curve is drawn. The cost is one GPU thread per segment and a line pass that
  /// was already running, so these are set by legibility rather than by budget: 32 segments
  /// and 5 levels drew a sphere as three loose hoops, which is what "not enough vertices and
  /// edges, e.g. the sphere and the torus" was.
  static constexpr int kRingSegments = 48;
  /// Rings along the length of a curved solid, and the meridians joining them. Nine levels
  /// puts seven rings on a sphere (its two poles have no radius), which reads as a surface.
  static constexpr int kProfileLevels = 9;
  static constexpr int kMeridians = 12;
  /// A torus is swept round its own circle, so it needs both families: cross-sections at
  /// intervals of the major angle, and rings at intervals of the minor one.
  static constexpr int kTorusMajor = 16;
  static constexpr int kTorusMinor = 8;

  void Add(float ax, float ay, float az, float bx, float by, float bz, unsigned int c) {
    x0.push_back(ax); y0.push_back(ay); z0.push_back(az);
    x1.push_back(bx); y1.push_back(by); z1.push_back(bz);
    rgb.push_back(c);
  }

  /// Local to world. The stored matrix is world -> local, so the inverse is its transpose.
  template <typename real_t>
  static Vec3<real_t> ToWorld(const geom::Transform<real_t>& xf, real_t x, real_t y, real_t z) {
    const real_t* r = xf.rot;
    return Vec3<real_t>{r[0] * x + r[3] * y + r[6] * z + xf.trans.x,
                        r[1] * x + r[4] * y + r[7] * z + xf.trans.y,
                        r[2] * x + r[5] * y + r[8] * z + xf.trans.z};
  }

  template <typename real_t>
  void AddSegment(const geom::Transform<real_t>& xf, real_t x0l, real_t y0l, real_t z0l,
                  real_t x1l, real_t y1l, real_t z1l, unsigned int c) {
    const auto a = ToWorld(xf, x0l, y0l, z0l);
    const auto b = ToWorld(xf, x1l, y1l, z1l);
    Add(static_cast<float>(a.x), static_cast<float>(a.y), static_cast<float>(a.z),
        static_cast<float>(b.x), static_cast<float>(b.y), static_cast<float>(b.z), c);
  }

  /// The twelve edges of an axis-aligned box, transformed by a volume's placement.
  template <typename real_t>
  void AddBox(const geom::Transform<real_t>& xf, real_t hx, real_t hy, real_t hz,
              unsigned int c) {
    const real_t sx[2] = {-hx, hx}, sy[2] = {-hy, hy}, sz[2] = {-hz, hz};
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
        AddSegment(xf, sx[0], sy[i], sz[j], sx[1], sy[i], sz[j], c);
        AddSegment(xf, sx[i], sy[0], sz[j], sx[i], sy[1], sz[j], c);
        AddSegment(xf, sx[i], sy[j], sz[0], sx[i], sy[j], sz[1], c);
      }
    }
  }

  /// A closed ring at height @p z with semi-axes @p rx and @p ry, in @p n segments, starting at
  /// azimuth @p phi0 so that a wedge's drawn meridians sit on the solid.
  template <typename real_t>
  void AddRing(const geom::Transform<real_t>& xf, real_t z, real_t rx, real_t ry, int n,
               real_t phi0, unsigned int c) {
    if (rx <= real_t(0) && ry <= real_t(0)) { return; }
    if (n < 3) { n = 3; }
    const double two_pi = 6.283185307179586;
    for (int i = 0; i < n; ++i) {
      const double a0 = phi0 + two_pi * i / n, a1 = phi0 + two_pi * (i + 1) / n;
      AddSegment(xf, static_cast<real_t>(rx * std::cos(a0)),
                 static_cast<real_t>(ry * std::sin(a0)), z,
                 static_cast<real_t>(rx * std::cos(a1)),
                 static_cast<real_t>(ry * std::sin(a1)), z, c);
    }
  }

  /// A surface of revolution given as a table of (z, rx, ry) levels: a ring at each level, and
  /// meridians joining consecutive levels.
  ///
  /// This is the whole of the curved-solid wireframe. A cylinder is two levels of equal radius,
  /// a cone two of different radius, a sphere five of sqrt(R^2 - z^2) - the solids differ only
  /// in the table they hand over.
  ///
  /// @param meridians how many generators to draw between the rings. A prism wants one per
  ///        side, so the drawn lines are the solid's real edges; a round solid wants a fixed
  ///        few, since any choice of meridian on it is arbitrary.
  template <typename real_t>
  void AddProfile(const geom::Transform<real_t>& xf, const std::vector<real_t>& z,
                  const std::vector<real_t>& rx, const std::vector<real_t>& ry, int ring_n,
                  int meridians, real_t phi0, unsigned int c) {
    const std::size_t n = z.size();
    if (n == 0 || rx.size() != n || ry.size() != n) { return; }
    for (std::size_t k = 0; k < n; ++k) { AddRing(xf, z[k], rx[k], ry[k], ring_n, phi0, c); }
    const int nm = (meridians >= 1) ? meridians : 1;
    const double two_pi = 6.283185307179586;
    for (int m = 0; m < nm; ++m) {
      const double a = phi0 + two_pi * m / nm;
      const double ca = std::cos(a), sa = std::sin(a);
      for (std::size_t k = 0; k + 1 < n; ++k) {
        AddSegment(xf, static_cast<real_t>(rx[k] * ca), static_cast<real_t>(ry[k] * sa), z[k],
                   static_cast<real_t>(rx[k + 1] * ca), static_cast<real_t>(ry[k + 1] * sa),
                   z[k + 1], c);
      }
    }
  }

  /// The edges of a convex solid that arrives as HALF-SPACES rather than as dimensions.
  ///
  /// kPara, kTrap and kTet are all stored as a set of planes n.q <= d - a trapezoid's eight
  /// corners are turned into six planes at construction and the dimensions are not kept - so
  /// there is no corner to read off. The corners are recovered instead: every triple of planes
  /// meets at a point, the points satisfying all the planes are the solid's vertices, and two
  /// vertices lying on the same two planes are the ends of an edge. Exact for any convex
  /// polyhedron, and it does not care which solid the planes came from.
  template <typename real_t>
  void AddPlanes(const geom::Transform<real_t>& xf, const real_t* pl, int n, unsigned int c) {
    constexpr int kMaxPlanes = 8;
    if (pl == nullptr || n < 4 || n > kMaxPlanes) { return; }

    // Unit normals, so that one tolerance serves every plane. G4Tet normalises its own and
    // para_planes does too; doing it here means AddPlanes does not depend on that.
    double p[4 * kMaxPlanes];
    double scale = 0;
    for (int i = 0; i < n; ++i) {
      const double nx = pl[4 * i], ny = pl[4 * i + 1], nz = pl[4 * i + 2];
      const double len = std::sqrt(nx * nx + ny * ny + nz * nz);
      if (!(len > 0)) { return; }
      p[4 * i] = nx / len;
      p[4 * i + 1] = ny / len;
      p[4 * i + 2] = nz / len;
      p[4 * i + 3] = static_cast<double>(pl[4 * i + 3]) / len;
      scale = std::max(scale, std::fabs(p[4 * i + 3]));
    }
    if (!(scale > 0)) { scale = 1; }
    const double on_plane = 1e-6 * scale;  ///< "this vertex lies on that face"
    const double slack = 1e-6 * scale;     ///< tolerance on the containment test

    struct Vertex {
      double x, y, z;
    };
    std::vector<Vertex> vs;
    for (int i = 0; i < n; ++i) {
      for (int j = i + 1; j < n; ++j) {
        for (int k = j + 1; k < n; ++k) {
          const double* a = p + 4 * i;
          const double* b = p + 4 * j;
          const double* e = p + 4 * k;
          const double det = a[0] * (b[1] * e[2] - b[2] * e[1])
                             - a[1] * (b[0] * e[2] - b[2] * e[0])
                             + a[2] * (b[0] * e[1] - b[1] * e[0]);
          // The normals are unit, so the determinant is a dimensionless measure of how far
          // from parallel the three planes are. Three planes through one line have no vertex.
          if (std::fabs(det) < 1e-6) { continue; }
          const double v[3] = {
              (a[3] * (b[1] * e[2] - b[2] * e[1]) - a[1] * (b[3] * e[2] - b[2] * e[3])
               + a[2] * (b[3] * e[1] - b[1] * e[3])) / det,
              (a[0] * (b[3] * e[2] - b[2] * e[3]) - a[3] * (b[0] * e[2] - b[2] * e[0])
               + a[2] * (b[0] * e[3] - b[3] * e[0])) / det,
              (a[0] * (b[1] * e[3] - b[3] * e[1]) - a[1] * (b[0] * e[3] - b[3] * e[0])
               + a[3] * (b[0] * e[1] - b[1] * e[0])) / det};
          bool contained = true;
          for (int m = 0; m < n && contained; ++m) {
            contained = (p[4 * m] * v[0] + p[4 * m + 1] * v[1] + p[4 * m + 2] * v[2]
                         - p[4 * m + 3]) <= slack;
          }
          if (!contained) { continue; }
          // A corner where more than three faces meet is produced once per triple through it;
          // keep it once, or every edge leaving it would be drawn several times over.
          bool seen = false;
          for (const Vertex& q : vs) {
            const double dx = q.x - v[0], dy = q.y - v[1], dz = q.z - v[2];
            if (dx * dx + dy * dy + dz * dz <= on_plane * on_plane) {
              seen = true;
              break;
            }
          }
          if (!seen) { vs.push_back(Vertex{v[0], v[1], v[2]}); }
        }
      }
    }

    // Two vertices sharing two faces lie on those faces' common line, which is an edge.
    for (std::size_t u = 0; u < vs.size(); ++u) {
      for (std::size_t w = u + 1; w < vs.size(); ++w) {
        int shared = 0;
        for (int m = 0; m < n; ++m) {
          const double du = p[4 * m] * vs[u].x + p[4 * m + 1] * vs[u].y
                            + p[4 * m + 2] * vs[u].z - p[4 * m + 3];
          const double dw = p[4 * m] * vs[w].x + p[4 * m + 1] * vs[w].y
                            + p[4 * m + 2] * vs[w].z - p[4 * m + 3];
          if (std::fabs(du) <= on_plane && std::fabs(dw) <= on_plane) { ++shared; }
        }
        if (shared >= 2) {
          AddSegment(xf, static_cast<real_t>(vs[u].x), static_cast<real_t>(vs[u].y),
                     static_cast<real_t>(vs[u].z), static_cast<real_t>(vs[w].x),
                     static_cast<real_t>(vs[w].y), static_cast<real_t>(vs[w].z), c);
        }
      }
    }
  }

  /// The edges a volume contributes.
  ///
  /// @param aux the solid store's aux pool, which is where a polycone's z-sections and a
  ///        trapezoid's planes live. Those solids keep no shape at all in `p[]`, so without it
  ///        they draw nothing - the same silent zero that bounding_radius carries a note about.
  ///
  /// The curved solids are drawn over the full 2*pi even where a phi wedge or a theta cut makes
  /// them partial, though a wedge's start angle is honoured so the drawn meridians lie on the
  /// solid. That over-draws a wedge, and it is the right way round: a wireframe is a statement
  /// about where a volume is, and one covering a little too much is worth more than the nothing
  /// a sphere used to get.
  template <typename real_t>
  void AddVolume(const geom::Volume<real_t>& v, unsigned int c, const real_t* aux = nullptr) {
    const auto& s = v.solid;
    const real_t* p = s.p;
    std::vector<real_t> z, rx, ry;
    real_t phi0 = 0;
    auto level = [&](real_t zz, real_t a, real_t b) {
      z.push_back(zz);
      rx.push_back(a);
      ry.push_back(b);
    };
    // A curve sampled at kProfileLevels heights between two ends.
    auto sweep = [&](real_t z_lo, real_t z_hi, auto radius) {
      for (int k = 0; k < kProfileLevels; ++k) {
        const real_t t = static_cast<real_t>(k) / static_cast<real_t>(kProfileLevels - 1);
        const real_t zz = z_lo + (z_hi - z_lo) * t;
        real_t a = 0, b = 0;
        radius(zz, a, b);
        level(zz, a, b);
      }
    };
    // A SPHERE'S LATITUDES ARE SPACED BY ANGLE, not by height. Equal steps in z put the rings
    // where the surface is turning fastest - two of them within a few degrees of a pole, where
    // they are tiny and add nothing - and leave the equator bare. Equal steps in the polar
    // angle is what a globe does, and it is the difference between a sphere and a stack of
    // hoops. Returns the rings for radius @p R.
    auto polar = [&](real_t R) {
      const double pi = 3.141592653589793;
      for (int k = 0; k < kProfileLevels; ++k) {
        const double th = pi * k / (kProfileLevels - 1);
        // THE POLES ARE EXACTLY ZERO, and sin(pi) is not: it is 1.2e-16, so the south pole
        // came out as a ring of radius 5e-15 and AddRing - which only rejects a radius that is
        // not positive - drew all 48 segments of it. Invisible in the picture and not
        // invisible in the count, which is how it was found.
        const bool pole = (k == 0 || k == kProfileLevels - 1);
        const real_t rr = pole ? real_t(0) : static_cast<real_t>(R * std::sin(th));
        level(static_cast<real_t>(R * std::cos(th)), rr, rr);
      }
    };

    switch (s.type) {
      case geom::SolidType::kBox:
      case geom::SolidType::kVoxelGrid:
        AddBox(v.xform, p[0], p[1], p[2], c);
        return;

      case geom::SolidType::kTrd: {
        // A rectangular frustum: dx1, dx2 at -dz and +dz, dy1, dy2, dz. Its eight corners are
        // in p[] directly, so there is no reason to go round through planes for it.
        const real_t hz = p[4];
        const real_t hx[2] = {p[0], p[1]}, hy[2] = {p[2], p[3]}, fz[2] = {-hz, hz};
        for (int f = 0; f < 2; ++f) {
          AddSegment(v.xform, -hx[f], -hy[f], fz[f],  hx[f], -hy[f], fz[f], c);
          AddSegment(v.xform,  hx[f], -hy[f], fz[f],  hx[f],  hy[f], fz[f], c);
          AddSegment(v.xform,  hx[f],  hy[f], fz[f], -hx[f],  hy[f], fz[f], c);
          AddSegment(v.xform, -hx[f],  hy[f], fz[f], -hx[f], -hy[f], fz[f], c);
        }
        for (int sx = 0; sx < 2; ++sx) {
          for (int sy = 0; sy < 2; ++sy) {
            const real_t ux = sx ? real_t(1) : real_t(-1), uy = sy ? real_t(1) : real_t(-1);
            AddSegment(v.xform, ux * hx[0], uy * hy[0], -hz, ux * hx[1], uy * hy[1], hz, c);
          }
        }
        return;
      }

      case geom::SolidType::kPara: {
        // A parallelepiped: its six planes are derived from p[], not stored in the aux pool.
        real_t pl[24];
        const int n = geom::para_planes(p, pl);
        AddPlanes(v.xform, pl, n, c);
        return;
      }

      case geom::SolidType::kTrap:
      case geom::SolidType::kTet:
        AddPlanes(v.xform, (aux != nullptr) ? aux + s.a : nullptr, s.b, c);
        return;

      case geom::SolidType::kOrb:
        polar(p[0]);
        break;

      case geom::SolidType::kSphere:
        // p[1] is rmax; a hollow sphere's inner surface gets its own rings, on the same
        // latitudes so the two read as one shell with a hole rather than as two objects.
        phi0 = p[2];
        polar(p[1]);
        if (p[0] > real_t(0)) {
          const double pi = 3.141592653589793;
          for (int k = 1; k + 1 < kProfileLevels; ++k) {   // not the poles: no ring there
            const double th = pi * k / (kProfileLevels - 1);
            const real_t r = static_cast<real_t>(p[0] * std::sin(th));
            AddRing(v.xform, static_cast<real_t>(p[0] * std::cos(th)), r, r, kRingSegments,
                    phi0, c);
          }
        }
        break;

      case geom::SolidType::kTubs:
        phi0 = p[3];
        level(-p[2], p[1], p[1]);
        level(p[2], p[1], p[1]);
        if (p[0] > real_t(0)) {
          AddRing(v.xform, -p[2], p[0], p[0], kRingSegments, phi0, c);
          AddRing(v.xform, p[2], p[0], p[0], kRingSegments, phi0, c);
        }
        break;

      case geom::SolidType::kCons:
        level(-p[2], p[0], p[0]);
        level(p[2], p[1], p[1]);
        break;

      case geom::SolidType::kConeSection:
        phi0 = p[5];
        level(-p[4], p[1], p[1]);
        level(p[4], p[3], p[3]);
        if (p[0] > real_t(0) || p[2] > real_t(0)) {
          AddRing(v.xform, -p[4], p[0], p[0], kRingSegments, phi0, c);
          AddRing(v.xform, p[4], p[2], p[2], kRingSegments, phi0, c);
        }
        break;

      case geom::SolidType::kEllipticalTube:
        level(-p[2], p[0], p[1]);
        level(p[2], p[0], p[1]);
        break;

      case geom::SolidType::kEllipsoid: {
        // Semi-axes p[0..2], cut between p[3] and p[4]. An uncut ellipsoid leaves both zero,
        // which is not a range - the full solid is what that means.
        real_t lo = p[3], hi = p[4];
        if (!(hi > lo)) {
          lo = -p[2];
          hi = p[2];
        }
        lo = (lo < -p[2]) ? -p[2] : lo;
        hi = (hi > p[2]) ? p[2] : hi;
        sweep(lo, hi, [&](real_t zz, real_t& a, real_t& b) {
          const real_t u = zz / p[2];
          const real_t q = real_t(1) - u * u;
          const real_t f = (q > real_t(0)) ? static_cast<real_t>(std::sqrt(q)) : real_t(0);
          a = p[0] * f;
          b = p[1] * f;
        });
        break;
      }

      case geom::SolidType::kEllipticalCone:
        // Semi-axis slopes p[0], p[1]; the apex is at z = p[2] and the solid is cut at +-p[3],
        // so a radius is the slope times the distance below the apex.
        sweep(-p[3], p[3], [&](real_t zz, real_t& a, real_t& b) {
          const real_t h = p[2] - zz;
          a = (h > real_t(0)) ? p[0] * h : real_t(0);
          b = (h > real_t(0)) ? p[1] * h : real_t(0);
        });
        break;

      case geom::SolidType::kParaboloid:
        // Half length p[0], radius p[1] at -dz and p[2] at +dz. It is r^2 that is linear in z,
        // which is what makes it a paraboloid rather than a cone.
        sweep(-p[0], p[0], [&](real_t zz, real_t& a, real_t& b) {
          const real_t t = (zz + p[0]) / (real_t(2) * p[0]);
          const real_t r2 = p[1] * p[1] + (p[2] * p[2] - p[1] * p[1]) * t;
          a = b = (r2 > real_t(0)) ? static_cast<real_t>(std::sqrt(r2)) : real_t(0);
        });
        break;

      case geom::SolidType::kHype:
        // Outer waist radius p[1] with stereo tangent p[3]; inner p[0] with p[2].
        sweep(-p[4], p[4], [&](real_t zz, real_t& a, real_t& b) {
          const real_t t = p[3] * zz;
          a = b = static_cast<real_t>(std::sqrt(p[1] * p[1] + t * t));
        });
        if (p[0] > real_t(0)) {
          for (int k = 0; k < kProfileLevels; ++k) {
            const real_t f = static_cast<real_t>(k) / static_cast<real_t>(kProfileLevels - 1);
            const real_t zz = -p[4] + real_t(2) * p[4] * f;
            const real_t t = p[2] * zz;
            const real_t r = static_cast<real_t>(std::sqrt(p[0] * p[0] + t * t));
            AddRing(v.xform, zz, r, r, kRingSegments, real_t(0), c);
          }
        }
        break;

      case geom::SolidType::kTorus: {
        // Not a surface of revolution in the profile sense - its own axis is a circle - so it
        // does not go through AddProfile, and it needs BOTH families of curve to read as a
        // torus. Cross-sections alone are a fan of loose loops, which is what it was: two
        // rings in the z = 0 plane and eight tube circles.
        //
        // Every point on it is (rtor + rmin_t cos(b)) * (cos(a), sin(a)) + rmin_t sin(b) z,
        // with a the major angle round the axis and b the minor angle round the tube. Holding
        // b fixed and sweeping a gives a ring the long way round; holding a fixed and sweeping
        // b gives a cross-section. Both are drawn.
        const double two_pi = 6.283185307179586;
        const double rt = static_cast<double>(p[2]), rm = static_cast<double>(p[1]);
        auto at = [&](double a, double b, real_t& x, real_t& y, real_t& z) {
          const double r = rt + rm * std::cos(b);
          x = static_cast<real_t>(r * std::cos(a));
          y = static_cast<real_t>(r * std::sin(a));
          z = static_cast<real_t>(rm * std::sin(b));
        };
        // Rings the long way round, at kTorusMinor stations of the minor angle - so the outer
        // equator, the inner one, the top and the bottom are all among them.
        for (int j = 0; j < kTorusMinor; ++j) {
          const double b = two_pi * j / kTorusMinor;
          for (int i = 0; i < kTorusMajor; ++i) {
            real_t x0, y0, z0v, x1, y1, z1v;
            at(p[3] + two_pi * i / kTorusMajor, b, x0, y0, z0v);
            at(p[3] + two_pi * (i + 1) / kTorusMajor, b, x1, y1, z1v);
            AddSegment(v.xform, x0, y0, z0v, x1, y1, z1v, c);
          }
        }
        // And the cross-sections, at kTorusMajor stations of the major angle.
        for (int i = 0; i < kTorusMajor; ++i) {
          const double a = p[3] + two_pi * i / kTorusMajor;
          for (int j = 0; j < kTorusMinor; ++j) {
            real_t x0, y0, z0v, x1, y1, z1v;
            at(a, two_pi * j / kTorusMinor, x0, y0, z0v);
            at(a, two_pi * (j + 1) / kTorusMinor, x1, y1, z1v);
            AddSegment(v.xform, x0, y0, z0v, x1, y1, z1v, c);
          }
        }
        return;
      }

      case geom::SolidType::kPolycone:
      case geom::SolidType::kPolyhedra: {
        // Its shape is the (z, rmin, rmax) table in the aux pool, not p[] - so without the
        // pool there is nothing to draw, which is why AddVolume asks for one.
        //
        // s.b is the number of SEGMENTS, so there are s.b + 1 planes. Reading only the
        // segments would drop the last plane, which on the builder's own bicone is one of the
        // two widest - the same off-by-one that bounding_radius carries a note about.
        if (aux == nullptr || s.b < 0) { return; }
        const bool prism = (s.type == geom::SolidType::kPolyhedra);
        const int sides = static_cast<int>(p[2] + real_t(0.5));
        const int ring_n = prism ? ((sides >= 3) ? sides : 3) : kRingSegments;
        // A PRISM'S RADIUS IS THE APOTHEM, NOT THE CORNER. geom::polyhedra_planes puts face f
        // at distance r from the axis along a normal at sphi + step*(f + 0.5) - so the faces
        // are centred BETWEEN the drawn vertices, and a corner is r / cos(pi/sides) out.
        // Taken as a corner radius the outline sits inside the solid it is outlining, by 15%
        // at six sides. Read off the engine's own planes rather than assumed, which is also
        // where the vertex azimuths come from: sphi + step*k, hence phi0 = sphi below.
        const real_t rs =
            prism ? static_cast<real_t>(1.0 / std::cos(3.141592653589793 / ring_n)) : real_t(1);
        phi0 = p[0];
        for (int i = 0; i <= s.b; ++i) {
          const real_t* sec = aux + s.a + 3 * i;
          level(sec[0], sec[2] * rs, sec[2] * rs);
          if (sec[1] > real_t(0)) {
            AddRing(v.xform, sec[0], sec[1] * rs, sec[1] * rs, ring_n, phi0, c);
          }
        }
        AddProfile(v.xform, z, rx, ry, ring_n, prism ? ring_n : kMeridians, phi0, c);
        return;
      }

      default:
        // A MESH and a BOOLEAN get nothing, and for once that is not an oversight. A CAD import
        // is tens of thousands of triangles: drawing its edges would put more line segments in
        // the frame than the ray cast costs, and its bounding box really is no statement about
        // its shape. A boolean's shape is not its children's either - a subtraction drawn as
        // both of its operands is drawn wrong.
        return;
    }
    AddProfile(v.xform, z, rx, ry, kRingSegments, kMeridians, phi0, c);
  }

  void clear() {
    x0.clear(); y0.clear(); z0.clear(); x1.clear(); y1.clear(); z1.clear(); rgb.clear();
  }
  std::size_t Size() const { return rgb.size(); }
};

}  // namespace g4gpu::vis
