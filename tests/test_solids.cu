// Every solid primitive and boolean operation, against Geant4's own G4VSolid answers.
//
// ref/oracle/solids.csv holds 4000 pseudo-random (point, direction) pairs per solid, with
// Geant4's Inside, DistanceToIn(p, v) and DistanceToOut(p, v) for each. This test rebuilds the
// same 30 solids in the port's representation and compares.
//
// Points Geant4 reports as kSurface are skipped for the distance comparison: both codes are
// entitled to answer either way there, and a surface point's entry distance is 0 or the far
// crossing depending on which side of a tolerance the point fell. They are still counted, and
// the count is printed, because a large surface fraction would mean the sample is degenerate.
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "core/rng.cuh"
#include "geometry/solids.cuh"
#include "geometry/volume_of.cuh"

using namespace g4gpu;
using namespace g4gpu::geom;
using real_t = double;

namespace {

constexpr real_t kDeg = 3.14159265358979323846 / 180.0;

/// One solid under test, together with whatever pooled data it needs.
struct Case {
  std::string name;
  Solid<real_t> solid;
  std::vector<Solid<real_t>> pool;      ///< boolean children
  std::vector<Transform<real_t>> xf;    ///< their transforms
  std::vector<real_t> aux;              ///< planes and z-sections
};

Solid<real_t> mk(SolidType t, std::initializer_list<real_t> ps) {
  Solid<real_t> s{};
  s.type = t;
  int i = 0;
  for (real_t v : ps) {
    if (i < 8) { s.p[i++] = v; }
  }
  s.xform = -1;
  return s;
}

/// G4Trap's six faces. Geant4 builds them from the corner vertices, so this does too: the
/// eight corners are generated from the half-lengths and the three tilt angles, then each face
/// gets the plane through three of its corners. Doing it by corners rather than by an
/// analytic normal is what makes theta/phi/alpha all fall out without special cases.
void trap_planes(real_t dz, real_t theta, real_t phi, real_t dy1, real_t dx1, real_t dx2,
                 real_t alp1, real_t dy2, real_t dx3, real_t dx4, real_t alp2,
                 std::vector<real_t>& out) {
  const real_t tthx = tan(theta) * cos(phi);
  const real_t tthy = tan(theta) * sin(phi);
  const real_t ta1 = tan(alp1), ta2 = tan(alp2);

  // Corners: index bit 0 = +z, bit 1 = +y, bit 2 = +x.
  Vec3<real_t> c[8];
  for (int iz = 0; iz < 2; ++iz) {
    const real_t z = iz ? dz : -dz;
    const real_t dy = iz ? dy2 : dy1;
    const real_t ta = iz ? ta2 : ta1;
    const real_t xm = iz ? dx3 : dx1;  // half-x at -dy
    const real_t xp = iz ? dx4 : dx2;  // half-x at +dy
    for (int iy = 0; iy < 2; ++iy) {
      const real_t y = iy ? dy : -dy;
      const real_t hx = iy ? xp : xm;
      for (int ix = 0; ix < 2; ++ix) {
        const real_t x = (ix ? hx : -hx) + ta * y + tthx * z;
        c[iz + 2 * iy + 4 * ix] = Vec3<real_t>{x, y + tthy * z, z};
      }
    }
  }
  // Face corner triples, wound so the normal points outward.
  const int faces[6][3] = {
      {0, 2, 6},  // -z  (iz=0)
      {1, 5, 7},  // +z
      {0, 4, 5},  // -y  (iy=0)
      {2, 3, 7},  // +y
      {0, 1, 3},  // -x  (ix=0)
      {4, 6, 7},  // +x
  };
  const Vec3<real_t> centre{0, 0, 0};
  for (int f = 0; f < 6; ++f) {
    const Vec3<real_t>& a = c[faces[f][0]];
    const Vec3<real_t>& b = c[faces[f][1]];
    const Vec3<real_t>& d = c[faces[f][2]];
    Vec3<real_t> n = normalize(cross(b - a, d - a));
    real_t off = dot(n, a);
    if (dot(n, centre) - off > 0) { n = real_t(-1) * n; off = -off; }  // face the outside
    out.push_back(n.x);
    out.push_back(n.y);
    out.push_back(n.z);
    out.push_back(off);
  }
}

void tet_planes(const Vec3<real_t>& a, const Vec3<real_t>& b, const Vec3<real_t>& c,
                const Vec3<real_t>& d, std::vector<real_t>& out) {
  const Vec3<real_t> tri[4][3] = {{a, b, c}, {a, b, d}, {a, c, d}, {b, c, d}};
  const Vec3<real_t> opp[4] = {d, c, b, a};
  for (int f = 0; f < 4; ++f) {
    Vec3<real_t> n = normalize(cross(tri[f][1] - tri[f][0], tri[f][2] - tri[f][0]));
    real_t off = dot(n, tri[f][0]);
    if (dot(n, opp[f]) - off > 0) { n = real_t(-1) * n; off = -off; }
    out.push_back(n.x);
    out.push_back(n.y);
    out.push_back(n.z);
    out.push_back(off);
  }
}

void add_polycone(Case& c, SolidType type, real_t sphi, real_t dphi, int sides, int n,
                  const real_t* z, const real_t* ri, const real_t* ro) {
  c.solid = mk(type, {sphi, dphi, real_t(sides)});
  c.solid.a = 0;
  c.solid.b = n - 1;  // number of frusta
  for (int i = 0; i < n; ++i) {
    c.aux.push_back(z[i]);
    c.aux.push_back(ri[i]);
    c.aux.push_back(ro[i]);
  }
}

/// The 30 cases, in the same order and with the same parameters the oracle used.
std::vector<Case> build_cases() {
  std::vector<Case> v;
  const real_t twopi = units::twopi<real_t>();
  const real_t pi = units::pi<real_t>();

  auto simple = [&](const char* n, Solid<real_t> s) {
    Case c;
    c.name = n;
    c.solid = s;
    v.push_back(std::move(c));
  };

  simple("box", mk(SolidType::kBox, {30, 40, 50}));
  simple("tubs_full", mk(SolidType::kTubs, {0, 40, 50, 0, twopi}));
  simple("tubs_hollow", mk(SolidType::kTubs, {15, 40, 50, 0, twopi}));
  simple("tubs_wedge", mk(SolidType::kTubs, {10, 40, 50, 20 * kDeg, 200 * kDeg}));
  simple("cons_full", mk(SolidType::kConeSection, {0, 20, 0, 40, 30, 0, twopi}));
  simple("cons_hollow", mk(SolidType::kConeSection, {5, 20, 10, 40, 30, 0, twopi}));
  simple("cons_wedge",
         mk(SolidType::kConeSection, {5, 20, 10, 40, 30, 30 * kDeg, 150 * kDeg}));
  simple("orb", mk(SolidType::kOrb, {45}));
  simple("sphere_full", mk(SolidType::kSphere, {0, 45, 0, twopi, 0, pi}));
  simple("sphere_shell", mk(SolidType::kSphere, {20, 45, 0, twopi, 0, pi}));
  simple("sphere_wedge",
         mk(SolidType::kSphere, {20, 45, 20 * kDeg, 200 * kDeg, 30 * kDeg, 90 * kDeg}));
  simple("torus", mk(SolidType::kTorus, {0, 12, 40, 0, twopi}));
  simple("torus_hollow", mk(SolidType::kTorus, {5, 12, 40, 0, twopi}));
  simple("torus_wedge", mk(SolidType::kTorus, {5, 12, 40, 30 * kDeg, 200 * kDeg}));
  simple("trd", mk(SolidType::kTrd, {30, 20, 40, 25, 50}));
  simple("para", mk(SolidType::kPara, {30, 40, 50, tan(20 * kDeg),
                                       tan(15 * kDeg) * cos(25 * kDeg),
                                       tan(15 * kDeg) * sin(25 * kDeg)}));
  {
    Case c;
    c.name = "trap";
    trap_planes(50, 10 * kDeg, 20 * kDeg, 30, 20, 26, 0, 35, 22, 29, 0, c.aux);
    c.solid = mk(SolidType::kTrap, {});
    c.solid.a = 0;
    c.solid.b = 6;
    v.push_back(std::move(c));
  }
  simple("eltube", mk(SolidType::kEllipticalTube, {30, 45, 50}));
  simple("ellipsoid", mk(SolidType::kEllipsoid, {30, 45, 60, -40, 50}));
  simple("elcone", mk(SolidType::kEllipticalCone, {0.5, 0.8, 50, 30}));
  simple("paraboloid", mk(SolidType::kParaboloid, {40, 10, 35}));
  simple("hype", mk(SolidType::kHype, {10, 25, tan(15 * kDeg), tan(30 * kDeg), 45}));
  {
    Case c;
    c.name = "tet";
    tet_planes({0, 0, 40}, {40, 0, -20}, {-25, 30, -20}, {-25, -30, -20}, c.aux);
    c.solid = mk(SolidType::kTet, {});
    c.solid.a = 0;
    c.solid.b = 4;
    v.push_back(std::move(c));
  }
  {
    const real_t z[4] = {-50, -10, 20, 50};
    const real_t ri[4] = {0, 8, 8, 0};
    const real_t ro[4] = {20, 40, 25, 35};
    Case c;
    c.name = "polycone";
    add_polycone(c, SolidType::kPolycone, 0, twopi, 0, 4, z, ri, ro);
    v.push_back(std::move(c));
    Case c2;
    c2.name = "polycone_wedge";
    add_polycone(c2, SolidType::kPolycone, 30 * kDeg, 180 * kDeg, 0, 4, z, ri, ro);
    v.push_back(std::move(c2));
    Case c3;
    c3.name = "polyhedra";
    add_polycone(c3, SolidType::kPolyhedra, 0, twopi, 6, 4, z, ri, ro);
    v.push_back(std::move(c3));
  }
  // Booleans. Child 0 is the left operand, child 1 the right; the right carries the offset.
  {
    auto boolean = [&](const char* name, SolidType op, Solid<real_t> lhs, Solid<real_t> rhs,
                       Transform<real_t> rhs_xf) {
      Case c;
      c.name = name;
      c.xf.push_back(rhs_xf);
      rhs.xform = 0;
      c.pool.push_back(lhs);
      c.pool.push_back(rhs);
      c.solid = mk(op, {});
      c.solid.a = 0;
      c.solid.b = 1;
      v.push_back(std::move(c));
    };
    const Solid<real_t> ba = mk(SolidType::kBox, {30, 30, 30});
    const Solid<real_t> bb = mk(SolidType::kBox, {20, 20, 50});
    const Solid<real_t> cy = mk(SolidType::kTubs, {0, 18, 60, 0, twopi});
    boolean("union_box_box", SolidType::kUnion, ba, bb,
            make_translation<real_t>({15, 10, 0}));
    boolean("subtract_box_cyl", SolidType::kSubtraction, ba, cy,
            make_translation<real_t>({10, 0, 0}));
    boolean("intersect_box_box", SolidType::kIntersection, ba, bb,
            make_translation<real_t>({15, 10, 0}));
    // The oracle built this one with rot->rotateY(35 deg). Geant4 uses that matrix as the
    // world -> solid map (G4DisplacedSolid::Inside applies the *inverse* of the transform it
    // was handed), so the solid itself ends up rotated by -35 degrees. make_placement takes
    // the rotation of the solid, which is the opposite sign - the same trap as G4PVPlacement's
    // rotation argument. Passing +35 here silently misplaces the cut and nothing else notices.
    boolean("subtract_rotated_cyl", SolidType::kSubtraction, ba, cy,
            make_placement<real_t>({5, 0, 0}, 0, -35 * kDeg, 0));
  }
  return v;
}

struct Stats {
  int n = 0, surface = 0;
  int inside_bad = 0, din_bad = 0, dout_bad = 0;
  int g4_wrong = 0;  ///< rows where Geant4's DistanceToIn contradicts Geant4's own Inside()
  double worst_din = 0, worst_dout = 0;
};

}  // namespace

int main() {
  const char* env = std::getenv("G4GPU_ORACLE");
  const std::string dir = (env != nullptr) ? env : "ref/oracle";
  FILE* f = std::fopen((dir + "/solids.csv").c_str(), "r");
  if (f == nullptr) {
    std::printf("cannot read %s/solids.csv\n", dir.c_str());
    return 1;
  }
  char line[512];
  if (std::fgets(line, sizeof line, f) == nullptr) { std::fclose(f); return 1; }

  std::vector<Case> cases = build_cases();
  std::vector<Stats> stats(cases.size());

  // The independent entry distances the oracle derived from Geant4's Inside() alone, keyed by
  // "solid:ray". Used only to adjudicate a disagreement - see the g4_wrong column.
  std::vector<std::string> scan_key;
  std::vector<double> scan_t;
  {
    FILE* g = std::fopen((dir + "/solid_scan.csv").c_str(), "r");
    if (g != nullptr) {
      char l[256];
      if (std::fgets(l, sizeof l, g) != nullptr) {
        while (std::fgets(l, sizeof l, g) != nullptr) {
          char nm[64];
          int ray;
          double t;
          if (std::sscanf(l, "%63[^,],%d,%lf", nm, &ray, &t) == 3) {
            scan_key.push_back(std::string(nm) + ":" + std::to_string(ray));
            scan_t.push_back(t);
          }
        }
      }
      std::fclose(g);
    }
  }
  auto scan_lookup = [&](const std::string& name, int ray, double& out) {
    const std::string k = name + ":" + std::to_string(ray);
    for (std::size_t i = 0; i < scan_key.size(); ++i) {
      if (scan_key[i] == k) { out = scan_t[i]; return true; }
    }
    return false;
  };
  std::vector<int> ray_index(cases.size(), 0);

  // Distances agree to this tolerance in mm. The port finds a crossing then probes kProbe()
  // past it to classify the far side, so an answer can differ from Geant4's by at most that
  // probe; 1e-4 mm leaves two decades of headroom over kProbe() = 1e-6 mm.
  const double kDistTol = 1e-4;

  int total = 0, unmatched = 0;
  while (std::fgets(line, sizeof line, f) != nullptr) {
    char name[64];
    double px, py, pz, dx, dy, dz, din, dout;
    int icode;
    if (std::sscanf(line, "%63[^,],%lf,%lf,%lf,%lf,%lf,%lf,%d,%lf,%lf", name, &px, &py, &pz,
                    &dx, &dy, &dz, &icode, &din, &dout) != 10) {
      continue;
    }
    std::size_t ci = cases.size();
    for (std::size_t i = 0; i < cases.size(); ++i) {
      if (cases[i].name == name) { ci = i; break; }
    }
    if (ci == cases.size()) { ++unmatched; continue; }
    ++total;

    const int this_ray = ray_index[ci]++;
    const Case& c = cases[ci];
    SolidStore<real_t> st{};
    st.solids = c.pool.empty() ? nullptr : c.pool.data();
    st.xforms = c.xf.empty() ? nullptr : c.xf.data();
    st.aux = c.aux.empty() ? nullptr : c.aux.data();

    const Vec3<real_t> p{px, py, pz};
    const Vec3<real_t> d{dx, dy, dz};
    Stats& s = stats[ci];
    ++s.n;
    if (icode == 1) {
      ++s.surface;
      continue;  // both codes may answer either way on a surface
    }

    const bool ours_in = inside(st, c.solid, p);
    const bool ref_in = (icode == 2);
    if (ours_in != ref_in) {
      ++s.inside_bad;
      continue;  // a containment disagreement makes the distance comparison meaningless
    }

    if (!ref_in) {
      const double ours = dist_in(st, c.solid, p, d);
      const double ref = (din < -1.5) ? kInfinity<real_t>() : din;
      const auto missed = [](double v) { return v >= 0.5 * kInfinity<real_t>(); };
      if (missed(ours) && missed(ref)) { continue; }
      const double err = std::fabs(ours - ref);
      if (err <= kDistTol) {
        if (err > s.worst_din) { s.worst_din = err; }
        continue;
      }
      // Disagreement. Adjudicate with the entry distance the oracle derived from Geant4's
      // Inside() alone; Inside() defines the solid, DistanceToIn is only meant to agree with
      // it. G4Hype fails that agreement for rays starting inside its inner hyperboloid.
      double scan = 0;
      bool g4_self_inconsistent = false;
      if (scan_lookup(c.name, this_ray, scan)) {
        const double ref_scan = (scan < -1.5) ? kInfinity<real_t>() : scan;
        // The scan bisects a 0.5 mm bracket, so it locates the surface to about a nanometre.
        const bool ours_matches_scan = (missed(ours) && missed(ref_scan))
                                       || std::fabs(ours - ref_scan) < 1e-6;
        g4_self_inconsistent = ours_matches_scan;
      }
      if (g4_self_inconsistent) {
        ++s.g4_wrong;
      } else {
        ++s.din_bad;
        if (err > s.worst_din) { s.worst_din = err; }
      }
    } else {
      const double ours = dist_out(st, c.solid, p, d);
      const double ref = (dout < -1.5) ? kInfinity<real_t>() : dout;
      const double err = std::fabs(ours - ref);
      if (err > s.worst_dout) { s.worst_dout = err; }
      if (err > kDistTol) { ++s.dout_bad; }
    }
  }
  std::fclose(f);

  std::printf("== solids vs Geant4 G4VSolid ==\n");
  std::printf("  %-22s %6s %6s %8s %8s %8s %8s %11s %11s\n", "solid", "n", "surf", "in_bad",
              "din_bad", "dout_bad", "g4_wrong", "worst_din", "worst_dout");
  int fails = 0;
  int g4_wrong_total = 0;
  for (std::size_t i = 0; i < cases.size(); ++i) {
    const Stats& s = stats[i];
    const bool bad = (s.inside_bad != 0 || s.din_bad != 0 || s.dout_bad != 0 || s.n == 0);
    g4_wrong_total += s.g4_wrong;
    std::printf("  %-22s %6d %6d %8d %8d %8d %8d %11.3g %11.3g%s\n", cases[i].name.c_str(), s.n,
                s.surface, s.inside_bad, s.din_bad, s.dout_bad, s.g4_wrong, s.worst_din,
                s.worst_dout, bad ? "   <-- FAIL" : "");
    if (bad) { ++fails; }
  }
  if (g4_wrong_total > 0) {
    std::printf("\n  %d row(s) where Geant4's DistanceToIn contradicts Geant4's own Inside();\n"
                "  the port agrees with Inside(), so those are not counted as failures.\n",
                g4_wrong_total);
  }
  std::printf("\n  %d rays compared across %d solids", total, static_cast<int>(cases.size()));
  if (unmatched > 0) { std::printf("; %d rows had no matching case", unmatched); }
  std::printf("\n");
  if (total == 0) {
    std::printf("  FAIL: nothing compared\n");
    ++fails;
  }
  if (unmatched > 0) {
    std::printf("  FAIL: the oracle dumped solids this test does not build\n");
    ++fails;
  }
  // ---------------------------------------------------------------- the extent contract
  //
  // `solid_half_extent` returns a bound on each *coordinate* - `max(dx, dy, dz)` for a box -
  // not a bounding radius. A box of (30, 40, 50) reaches 71.4 mm from its origin at the
  // corner and the function returns 50. That is exactly right for what it is for, sizing an
  // axis-aligned sampling box, and it is why nothing may use it as a radius without
  // multiplying by sqrt(3).
  //
  // Nothing checked either reading. This checks the one the function actually promises, for
  // every solid the test builds - any point the solid claims to contain must lie inside the
  // cube of half-side `extent` - and prints the measured circumradius alongside, because a
  // bounding sphere per volume is the obvious next use and the conversion factor should be a
  // measurement rather than an assumption.
  //
  // Two cases in that function are admitted guesses: `kTrap`/`kTet` return four times the
  // largest plane offset ("a plane offset bounds the inradius, not the circumradius") and
  // `kPolycone`/`kPolyhedra` multiply by 1.2. Guesses are what this section is for.
  {
    std::printf("\n== solid_half_extent bounds every coordinate ==\n");
    std::printf("  %-16s %8s %10s %10s %10s %8s\n", "solid", "inside", "extent", "worst |x|",
                "worst |p|", "|p|/ext");
    constexpr int kSamples = 200000;
    int bad_cases = 0;
    for (std::size_t ci = 0; ci < cases.size(); ++ci) {
      const Case& c = cases[ci];
      SolidStore<real_t> st{};
      st.solids = c.pool.empty() ? nullptr : c.pool.data();
      st.xforms = c.xf.empty() ? nullptr : c.xf.data();
      st.aux = c.aux.empty() ? nullptr : c.aux.data();

      const real_t extent = solid_half_extent(st, c.solid);
      // Sample in a box comfortably larger than the extent, so a solid that reaches further
      // than it claims has somewhere to be found.
      const real_t box = extent * real_t(3) + real_t(10);
      Philox<real_t> rng(static_cast<uint32_t>(ci) + 1u, 11u, 3u);
      int n_in = 0;
      real_t worst_coord = 0, worst_r = 0;
      for (int k = 0; k < kSamples; ++k) {
        const Vec3<real_t> p{(rng.uniform() * real_t(2) - real_t(1)) * box,
                             (rng.uniform() * real_t(2) - real_t(1)) * box,
                             (rng.uniform() * real_t(2) - real_t(1)) * box};
        if (!inside(st, c.solid, p)) { continue; }
        ++n_in;
        worst_coord = fmax(worst_coord, fmax(fabs(p.x), fmax(fabs(p.y), fabs(p.z))));
        worst_r = fmax(worst_r, sqrt(dot(p, p)));
      }
      // 1 + 1e-9: a point exactly on the surface is inside by the solid's own tolerance, and
      // the extent is the exact half-side for a box, so the comparison has to allow the last
      // bit rather than demand a strict inequality.
      const bool over = (worst_coord > extent * real_t(1 + 1e-9));
      std::printf("  %-16s %8d %10.4g %10.4g %10.4g %8.3f%s\n", c.name.c_str(), n_in,
                  double(extent), double(worst_coord), double(worst_r),
                  (extent > 0) ? double(worst_r / extent) : 0.0,
                  over ? "   <-- FAIL: outside the claimed extent" : "");
      if (n_in == 0) {
        std::printf("    FAIL: no sampled point was inside - the extent is untested here\n");
        ++bad_cases;
      }
      if (over) { ++bad_cases; }
      // sqrt(3) is the most a cube's corner can exceed its half-side by. If a solid beat that
      // it would mean the coordinate bound had failed, which the check above already catches -
      // this is here so the printed ratio can be read as "how loose a bounding sphere would be".
      if (extent > 0 && worst_r > extent * real_t(1.7320509)) {
        std::printf("    FAIL: |p| exceeds extent*sqrt(3), which is arithmetically impossible\n"
                    "          if the coordinate bound holds\n");
        ++bad_cases;
      }
    }
    if (bad_cases > 0) {
      std::printf("  FAIL: %d solid(s) reach beyond solid_half_extent\n", bad_cases);
      fails += bad_cases;
    }
  }

  std::printf("\n%s (%d failures)\n", fails ? "FAILED" : "PASSED", fails);
  return fails ? 1 : 0;
}
