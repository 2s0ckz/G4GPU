// The wireframe edge list: does an outline actually lie on the solid it claims to outline?
//
// There is no oracle for a wireframe - it is a drawing, not a physical quantity - but there is
// one fact it has to satisfy, and it is enough to catch every way this has been wrong: EVERY
// DRAWN VERTEX IS ON THE SURFACE OF THE SOLID. The engine already answers "is this point in the
// solid", so the outline can be checked against the same containment test the ray cast uses,
// and any disagreement about a solid's dimensions shows up immediately.
//
// What that catches, in the order these were found:
//
//   * A SOLID WITH NO OUTLINE AT ALL. The list drew boxes and voxel grids and nothing else, so
//     a sphere set to wireframe was invisible - and the only check on it was "turning the
//     wireframe off changes the picture", which the world satisfies because the world is a box.
//     A per-type segment count cannot be satisfied by a pass that draws nothing.
//   * AN OUTLINE OF THE WRONG SIZE. A polyhedra's rmax is the APOTHEM, not the corner radius:
//     geom::polyhedra_planes puts face f at distance r along a normal at sphi + step*(f + 0.5),
//     so the faces are centred between the vertices and a corner is r / cos(pi/sides) out. Drawn
//     at r, a hexagonal prism's outline sits 15% inside the solid. Nothing about the count
//     changes when that is wrong, and nothing about the picture says which of the two it is.
//   * AN OUTLINE OF THE WRONG SHAPE. The plane-triple reconstruction that recovers a kPara,
//     kTrap or kTet's corners could put a vertex anywhere; a shear read the wrong way round
//     gives eight plausible corners of the wrong parallelepiped.
//
// BRACKETED, NOT TESTED AT THE POINT. EdgeList stores floats, so an exact corner round-tripped
// through float lands about 5e-7 off, which is outside kSurfTolerance in double and says nothing
// about the geometry. A vertex is required to be inside when pulled 0.1% toward the axis and
// outside when pushed 2% away from it. At six sides the apothem and the corner differ by 15%,
// so the bracket is far tighter than the error it is there to catch.
#include <cmath>
#include <cstdio>
#include <vector>

#include "geometry/navigator.cuh"
#include "geometry/solids.cuh"
#include "render/edges.h"

using namespace g4gpu;
using real_t = double;
using E = vis::EdgeList;

static int fails = 0;
static void check(bool ok, const char* what) {
  printf(ok ? "  ok:   %s\n" : "  FAIL: %s\n", what);
  if (!ok) { ++fails; }
}

static geom::Volume<real_t> identity_volume() {
  geom::Volume<real_t> v{};
  v.xform.rot[0] = v.xform.rot[4] = v.xform.rot[8] = real_t(1);
  return v;
}

/// Every vertex the list holds, bracketed against the solid's own containment test.
///
/// @param inward   the direction "into the solid" from a vertex, as a scale on the position.
///        Radially inward for a solid built about the z axis, which is every case here.
/// @param expect   how many segments the type should produce, or 0 not to ask.
static void on_surface(const char* what, const vis::EdgeList& e,
                       const geom::SolidStore<real_t>& st, const geom::Solid<real_t>& s,
                       std::size_t expect) {
  int tested = 0, off_surface = 0, not_bounded = 0;
  for (std::size_t i = 0; i < e.Size(); ++i) {
    const real_t x = e.x0[i], y = e.y0[i], z = e.z0[i];
    const real_t r = std::sqrt(x * x + y * y);
    // A pole has no radius to scale, so the bracket has no direction to point in.
    if (r < real_t(1e-9)) { continue; }
    ++tested;
    // SCALED ABOUT THE ORIGIN, all three coordinates together, which is the radial direction
    // for every solid here - each is centred on its own origin and contains it. Scaling only x
    // and y was wrong near a POLE, where the outward direction is almost entirely z: pushing
    // the radius out 2% while shrinking z by 0.1% moves such a point INWARD, and it read as 48
    // vertices of a sphere not being bounded by their own surface. That appeared only once the
    // profile was sampled finely enough to have vertices near a pole at all.
    if (!geom::inside(st, s, Vec3<real_t>{x * real_t(0.999), y * real_t(0.999),
                                          z * real_t(0.999)})) {
      ++off_surface;
    }
    if (geom::inside(st, s, Vec3<real_t>{x * real_t(1.02), y * real_t(1.02),
                                         z * real_t(1.02)})) {
      ++not_bounded;
    }
  }
  printf("  %-14s %zu segments, %d vertices: %d off the surface, %d not bounded by it\n", what,
         e.Size(), tested, off_surface, not_bounded);
  if (expect > 0) {
    check(e.Size() == expect, "the segment count is what the shape needs");
  }
  check(tested > 0, "the solid has an outline at all");
  check(off_surface == 0, "every drawn vertex is inside the solid when pulled in");
  check(not_bounded == 0, "and outside it when pushed out");
}

/// The longest segment in the list, as a fraction of the solid's size.
///
/// WHETHER A CURVE IS FOLLOWED OR CHORDED. `on_surface` puts every drawn VERTEX on the surface,
/// and a chord straight across a curve has both ends on the surface too - so it cannot tell a
/// smooth meridian from a straight line between two rings, which is exactly what a sphere had:
/// latitudes of 48 segments and longitudes of eight chords from pole to pole.
///
/// A segment's length is the test. On a curve of radius R sampled every angle t, a segment is
/// about R*t long, so requiring every segment to be under a small fraction of R bounds the
/// angle it turns through - and a chord across a whole meridian is half a circumference, which
/// no bound like that admits.
static real_t longest_segment(const vis::EdgeList& e, real_t scale) {
  real_t worst = 0;
  for (std::size_t i = 0; i < e.Size(); ++i) {
    const real_t dx = real_t(e.x1[i]) - e.x0[i];
    const real_t dy = real_t(e.y1[i]) - e.y0[i];
    const real_t dz = real_t(e.z1[i]) - e.z0[i];
    const real_t d = std::sqrt(dx * dx + dy * dy + dz * dz);
    if (d > worst) { worst = d; }
  }
  return (scale > 0) ? worst / scale : worst;
}

int main() {
  const real_t kFullTurn = real_t(6.283185307179586);

  // ---------------------------------------------------------------- 1. surfaces of revolution
  //
  // One table of (z, rx, ry) levels per solid and one function that draws them, so what is
  // being checked per type is the TABLE: a radius law read out of the wrong p[] entry puts
  // every vertex off the surface at once.
  printf("\n== a curved solid's outline lies on it ==\n");
  {
    geom::SolidStore<real_t> st{};
    geom::Volume<real_t> v = identity_volume();

    // An orb. Its latitudes are spaced by ANGLE, not by height - equal steps in z crowd two
    // rings within a few degrees of each pole, where they are tiny, and leave the equator bare
    // - so the two poles are the only levels with no radius and every other one is a ring.
    v.solid = geom::Solid<real_t>{};
    v.solid.type = geom::SolidType::kOrb;
    v.solid.p[0] = 40;
    vis::EdgeList orb;
    orb.AddVolume(v, 0u);
    on_surface("orb", orb, st, v.solid,
               7 * E::kRingSegments + E::kMeridians * (E::kProfileSamples - 1));
    // EVERY LINE ON IT FOLLOWS THE SURFACE, latitudes and longitudes alike. A sphere of radius
    // R sampled at 48 round and 32 along turns at most 2*pi/32 per segment, which is 0.2 R -
    // so 0.25 is a bound with a little room and nowhere near the 2 R a pole-to-pole chord
    // would be. This is the check `on_surface` cannot make: a chord has both ends on the
    // surface too.
    {
      const real_t worst = longest_segment(orb, 40);
      printf("  %-14s longest segment %.3f of the radius\n", "orb", worst);
      check(worst < real_t(0.25), "every line on a sphere follows it, meridians included");
    }

    // A hollow cylinder. Its inner rings are on the rmin surface, where the bracket points the
    // wrong way by construction - so they are checked separately, against rmin's own solid.
    v.solid = geom::Solid<real_t>{};
    v.solid.type = geom::SolidType::kTubs;
    v.solid.p[0] = 8;    // rmin
    v.solid.p[1] = 25;   // rmax
    v.solid.p[2] = 30;   // dz
    v.solid.p[4] = kFullTurn;
    vis::EdgeList tubs;
    tubs.AddVolume(v, 0u);
    int outer = 0, inner = 0, neither = 0;
    for (std::size_t i = 0; i < tubs.Size(); ++i) {
      const real_t r = std::sqrt(real_t(tubs.x0[i]) * tubs.x0[i]
                                 + real_t(tubs.y0[i]) * tubs.y0[i]);
      if (std::fabs(r - 25) < real_t(1e-3)) { ++outer; }
      else if (std::fabs(r - 8) < real_t(1e-3)) { ++inner; }
      else { ++neither; }
    }
    printf("  %-14s %zu segments: %d on rmax, %d on rmin, %d on neither\n", "hollow tubs",
           tubs.Size(), outer, inner, neither);
    check(neither == 0, "every vertex of a hollow tube is on one of its two radii");
    check(inner == 2 * E::kRingSegments, "the inner surface gets a ring at each end");
    check(outer > 0, "and the outer surface is drawn too");

    // A cone: rmax differs at the two ends, so a table that reads one radius for both puts
    // half the vertices off the surface.
    v.solid = geom::Solid<real_t>{};
    v.solid.type = geom::SolidType::kCons;
    v.solid.p[0] = 30;   // rmax at -dz
    v.solid.p[1] = 10;   // rmax at +dz
    v.solid.p[2] = 25;   // dz
    vis::EdgeList cons;
    cons.AddVolume(v, 0u);
    on_surface("cons", cons, st, v.solid, 2 * E::kRingSegments + E::kMeridians);

    // A paraboloid, whose r^2 is linear in z rather than r. Drawn as a cone, the mid levels
    // sit inside the surface - the one case where only the INTERMEDIATE levels are wrong, so
    // it is what says the levels are sampled from the real law and not just at the ends.
    v.solid = geom::Solid<real_t>{};
    v.solid.type = geom::SolidType::kParaboloid;
    v.solid.p[0] = 30;   // dz
    v.solid.p[1] = 5;    // r at -dz
    v.solid.p[2] = 25;   // r at +dz
    vis::EdgeList par;
    par.AddVolume(v, 0u);
    on_surface("paraboloid", par, st, v.solid,
               9 * E::kRingSegments + E::kMeridians * (E::kProfileSamples - 1));

    // An elliptical tube: rx and ry differ, so a table that carries one radius per level puts
    // every vertex but four off the surface.
    v.solid = geom::Solid<real_t>{};
    v.solid.type = geom::SolidType::kEllipticalTube;
    v.solid.p[0] = 30;
    v.solid.p[1] = 12;
    v.solid.p[2] = 20;
    vis::EdgeList etub;
    etub.AddVolume(v, 0u);
    on_surface("elliptical tube", etub, st, v.solid, 2 * E::kRingSegments + E::kMeridians);

    // A TORUS GETS ITS OWN CHECK, because on_surface's bracket cannot express it. That bracket
    // scales a vertex radially, which is "outward" only for a solid built about the z axis; on
    // a torus, scaling a point on the INNER half of the tube moves it INTO the material. So the
    // implicit function is used directly: (sqrt(x^2 + y^2) - rtor)^2 + z^2 = rmin^2, which is
    // exact and needs no direction at all.
    //
    // It also needs both FAMILIES of curve. Cross-sections alone are a fan of loose loops -
    // which is what it was, two rings in the z = 0 plane and eight tube circles - so the count
    // is what says the rings the long way round are there.
    v.solid = geom::Solid<real_t>{};
    v.solid.type = geom::SolidType::kTorus;
    v.solid.p[0] = 0;    // rmin of the tube: solid
    v.solid.p[1] = 8;    // tube radius
    v.solid.p[2] = 40;   // rtor
    v.solid.p[4] = kFullTurn;
    vis::EdgeList tor;
    tor.AddVolume(v, 0u);
    int off_tube = 0;
    real_t worst = 0;
    for (std::size_t i = 0; i < tor.Size(); ++i) {
      const real_t x = tor.x0[i], y = tor.y0[i], z = tor.z0[i];
      const real_t q = std::sqrt(x * x + y * y) - real_t(40);
      const real_t d = std::sqrt(q * q + z * z) - real_t(8);
      if (std::fabs(d) > worst) { worst = std::fabs(d); }
      if (std::fabs(d) > real_t(1e-3)) { ++off_tube; }
    }
    printf("  %-14s %zu segments, %d off the tube, worst %.2e mm\n", "torus", tor.Size(),
           off_tube, static_cast<double>(worst));
    check(tor.Size() == (E::kTorusMajor + E::kTorusMinor) * E::kRingSegments,
          "a torus is drawn as rings the long way round AND cross-sections");
    check(off_tube == 0, "every drawn vertex of a torus is on its tube");
    // And both families are curves. Each used to be drawn with the OTHER family's station
    // count, so a cross-section was an octagon and a long-way ring a 16-gon - the largest
    // segment on the tube of radius 8 was about 6 mm, three quarters of the tube's own radius.
    // Against rtor + rmin, not the tube radius: a ring the long way round has that radius, so
    // it is the largest curve on the solid and the one whose segments are longest.
    {
      const real_t worst = longest_segment(tor, 48);
      printf("  %-14s longest segment %.3f of its outer radius\n", "torus", worst);
      check(worst < real_t(0.25),
            "and both families of curve on a torus are drawn as curves");
    }
  }

  // ---------------------------------------------------------------- 2. the aux-pool solids
  //
  // A polycone and a polyhedra keep no shape at all in p[] - their (z, rmin, rmax) sections are
  // in the aux pool - so without the pool there is nothing to draw, which is the same silent
  // zero that bounding_radius carries a note about.
  printf("\n== an outline read from the aux pool lies on the solid too ==\n");
  {
    // The builder's own bicone: three sections, widest in the middle.
    std::vector<real_t> aux = {-30, 0, 10, 0, 0, 25, 30, 0, 5};
    geom::SolidStore<real_t> st{};
    st.aux = aux.data();
    geom::Volume<real_t> v = identity_volume();
    v.solid = geom::Solid<real_t>{};
    v.solid.type = geom::SolidType::kPolycone;
    v.solid.a = 0;
    v.solid.b = 2;   // SEGMENTS, so there are three planes - the off-by-one that drops the last
    v.solid.p[1] = kFullTurn;
    vis::EdgeList pc;
    pc.AddVolume(v, 0u, aux.data());
    on_surface("polycone", pc, st, v.solid, 3 * E::kRingSegments + E::kMeridians * 2);

    vis::EdgeList none;
    none.AddVolume(v, 0u);
    check(none.Size() == 0, "and without the pool it draws nothing rather than a wrong shape");

    // A straight hexagonal prism. rmax is the APOTHEM: drawn as a corner radius the outline is
    // 15% inside the solid, and every vertex fails the bracket.
    std::vector<real_t> hex = {-30, 0, 20, 0, 0, 20, 30, 0, 20};
    st.aux = hex.data();
    v.solid = geom::Solid<real_t>{};
    v.solid.type = geom::SolidType::kPolyhedra;
    v.solid.a = 0;
    v.solid.b = 2;
    v.solid.p[1] = kFullTurn;
    v.solid.p[2] = 6;   // sides
    vis::EdgeList ph;
    ph.AddVolume(v, 0u, hex.data());
    on_surface("polyhedra", ph, st, v.solid, 3 * 6 + 6 * 2);
    real_t rmax = 0;
    for (std::size_t i = 0; i < ph.Size(); ++i) {
      const real_t r =
          std::sqrt(real_t(ph.x0[i]) * ph.x0[i] + real_t(ph.y0[i]) * ph.y0[i]);
      if (r > rmax) { rmax = r; }
    }
    const real_t corner = real_t(20) / std::cos(real_t(3.141592653589793) / 6);
    printf("  a prism's corner is at %.4f, and the apothem it is built from is 20\n", rmax);
    check(std::fabs(rmax - corner) < real_t(1e-3),
          "a prism is drawn to its corner radius, not to its apothem");
    // And the meridians are the prism's real edges, not an arbitrary eight.
    check(ph.Size() == 3 * 6 + 6 * 2, "a prism gets one generator per side");
  }

  // ---------------------------------------------------------------- 3. the half-space solids
  //
  // kPara, kTrap and kTet keep no dimensions - only planes n.q <= d - so their corners are
  // recovered by intersecting plane triples and keeping the points that satisfy every plane.
  // Two vertices lying on the same two planes are the ends of an edge.
  //
  // Checked by COUNT and by containment: a convex polyhedron with V vertices and E edges is
  // wrong in a way the count sees if the reconstruction drops a vertex or invents one, and
  // wrong in a way containment sees if it puts one in the wrong place.
  printf("\n== a solid stored as half-spaces has its corners recovered ==\n");
  {
    geom::SolidStore<real_t> st{};
    geom::Volume<real_t> v = identity_volume();

    // A parallelepiped with a real shear in both directions: 8 corners, 12 edges. Its planes
    // come from p[] through geom::para_planes, which is also what the ray cast uses - so the
    // outline cannot disagree with the surface about the shear.
    v.solid = geom::Solid<real_t>{};
    v.solid.type = geom::SolidType::kPara;
    v.solid.p[0] = 20;
    v.solid.p[1] = 15;
    v.solid.p[2] = 10;
    v.solid.p[3] = real_t(0.5);   // tan(alpha)
    v.solid.p[4] = real_t(0.3);   // tan(theta) cos(phi)
    v.solid.p[5] = real_t(0.2);   // tan(theta) sin(phi)
    vis::EdgeList para;
    para.AddVolume(v, 0u);
    check(para.Size() == 12, "a sheared parallelepiped has twelve edges");
    // Its corners are x = +-dx + tan(alpha) y0 + tan(theta)cos(phi) z, where y0 is the
    // UNSHEARED y - which is why the planes are read rather than the corners derived. Checked
    // against the solid rather than against that formula, for the same reason.
    int outside = 0;
    for (std::size_t i = 0; i < para.Size(); ++i) {
      const Vec3<real_t> a{para.x0[i], para.y0[i], para.z0[i]};
      const Vec3<real_t> b{para.x1[i], para.y1[i], para.z1[i]};
      // A midpoint is strictly inside a convex solid when both ends are on it, which is the
      // one statement that does not need a tolerance on the surface itself.
      const Vec3<real_t> mid{(a.x + b.x) / 2, (a.y + b.y) / 2, (a.z + b.z) / 2};
      if (!geom::inside(st, v.solid, Vec3<real_t>{mid.x * real_t(0.99), mid.y * real_t(0.99),
                                                  mid.z * real_t(0.99)})) {
        ++outside;
      }
    }
    printf("  %-14s %zu segments, %d midpoints not in the solid\n", "para", para.Size(),
           outside);
    check(outside == 0, "every drawn edge of a para runs through the solid");

    // A tetrahedron: 4 planes, 4 vertices, 6 edges - and the reconstruction has to find
    // exactly the four points where three of the four planes meet.
    const real_t c0[3] = {0, 0, 0}, c1[3] = {30, 0, 0}, c2[3] = {0, 30, 0}, c3[3] = {0, 0, 30};
    const real_t* tri[4][3] = {{c0, c1, c2}, {c0, c1, c3}, {c0, c2, c3}, {c1, c2, c3}};
    const real_t* opp[4] = {c3, c2, c1, c0};
    std::vector<real_t> tet(16);
    for (int f = 0; f < 4; ++f) {
      real_t u[3], w[3], n[3];
      for (int k = 0; k < 3; ++k) {
        u[k] = tri[f][1][k] - tri[f][0][k];
        w[k] = tri[f][2][k] - tri[f][0][k];
      }
      n[0] = u[1] * w[2] - u[2] * w[1];
      n[1] = u[2] * w[0] - u[0] * w[2];
      n[2] = u[0] * w[1] - u[1] * w[0];
      const real_t len = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
      for (int k = 0; k < 3; ++k) { n[k] /= len; }
      real_t off = n[0] * tri[f][0][0] + n[1] * tri[f][0][1] + n[2] * tri[f][0][2];
      if (n[0] * opp[f][0] + n[1] * opp[f][1] + n[2] * opp[f][2] - off > 0) {
        for (int k = 0; k < 3; ++k) { n[k] = -n[k]; }
        off = -off;
      }
      tet[4 * f + 0] = n[0];
      tet[4 * f + 1] = n[1];
      tet[4 * f + 2] = n[2];
      tet[4 * f + 3] = off;
    }
    st.aux = tet.data();
    v.solid = geom::Solid<real_t>{};
    v.solid.type = geom::SolidType::kTet;
    v.solid.a = 0;
    v.solid.b = 4;
    vis::EdgeList t;
    t.AddVolume(v, 0u, tet.data());
    check(t.Size() == 6, "a tetrahedron has six edges");
    // The four corners are known here, so this one can be exact: every drawn endpoint is one
    // of them, and each of them is an endpoint.
    const real_t* want[4] = {c0, c1, c2, c3};
    int hits[4] = {0, 0, 0, 0}, unknown = 0;
    for (std::size_t i = 0; i < t.Size(); ++i) {
      const real_t ends[2][3] = {{t.x0[i], t.y0[i], t.z0[i]}, {t.x1[i], t.y1[i], t.z1[i]}};
      for (int k = 0; k < 2; ++k) {
        int found = -1;
        for (int q = 0; q < 4; ++q) {
          const real_t dx = ends[k][0] - want[q][0], dy = ends[k][1] - want[q][1];
          const real_t dz = ends[k][2] - want[q][2];
          if (dx * dx + dy * dy + dz * dz < real_t(1e-6)) { found = q; }
        }
        if (found < 0) { ++unknown; } else { ++hits[found]; }
      }
    }
    printf("  %-14s %zu segments, endpoints per corner %d %d %d %d, %d elsewhere\n", "tet",
           t.Size(), hits[0], hits[1], hits[2], hits[3], unknown);
    check(unknown == 0, "every endpoint of a tet's outline is one of its four corners");
    check(hits[0] == 3 && hits[1] == 3 && hits[2] == 3 && hits[3] == 3,
          "and each corner is met by exactly three edges");
  }

  // ---------------------------------------------------------------- 4. what is left out
  //
  // A mesh and a boolean get no outline, and that is a decision rather than an oversight: a CAD
  // import is tens of thousands of triangles, and a boolean's shape is not its operands'. Worth
  // a check because the alternative failure - drawing a bounding box and calling it an outline -
  // is exactly what the curved solids used to get, and it looks like it works.
  printf("\n== a mesh and a boolean draw nothing rather than something wrong ==\n");
  {
    geom::Volume<real_t> v = identity_volume();
    v.solid = geom::Solid<real_t>{};
    v.solid.type = geom::SolidType::kMesh;
    v.solid.p[0] = v.solid.p[1] = v.solid.p[2] = 50;
    vis::EdgeList m;
    m.AddVolume(v, 0u);
    check(m.Size() == 0, "a mesh has no outline");

    v.solid = geom::Solid<real_t>{};
    v.solid.type = geom::SolidType::kSubtraction;
    v.solid.a = 0;
    v.solid.b = 1;
    vis::EdgeList b;
    b.AddVolume(v, 0u);
    check(b.Size() == 0, "and neither does a boolean");
  }

  printf(fails == 0 ? "\nALL OK\n" : "\n%d FAILED\n", fails);
  return fails == 0 ? 0 : 1;
}
