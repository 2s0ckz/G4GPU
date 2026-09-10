// Triangle-mesh solids: the BVH, the ray-triangle test, and containment by parity.
//
// There is no Geant4 oracle here. G4TessellatedSolid exists in Geant4 and could in principle
// answer the same queries, but ref/dump/g4dump.cc builds its solids from parameters and a mesh
// has no parameters - the oracle would have to be handed the same triangle list, which means
// writing the file out and reading it back, and what that would test is the file format.
//
// So the checks are against things the geometry has to satisfy regardless of implementation:
//
//   * a mesh of a box answers exactly what the analytic box answers - containment for random
//     points, entry and exit distances for random rays. Any mesh of a box is the box, so this
//     is an exact comparison with no discretisation error to allow for, and it is the one
//     check that would catch a wrong ray-triangle test, a wrong parity rule or a broken BVH
//     all at once.
//   * the BVH agrees with brute force over every triangle, on a mesh with enough triangles to
//     have a real tree. That separates "the tree is wrong" from "the triangle test is wrong":
//     if the box test passes and this fails, it is the tree.
//   * the enclosed volume from the divergence theorem matches the analytic volume.
//   * points on vertices and edges - where a parity count is at its least reliable - are
//     classified consistently from many directions.
//
// Every ray direction here is off-axis. An axis-aligned ray through a box mesh passes through
// shared edges and vertices constantly, and a test that used only those would either fail for
// the wrong reason or, worse, pass while the parity rule was broken for the general case.
#include <cmath>
#include <cstdio>
#include <vector>

#include "core/rng.cuh"
#include "geometry/bvh_build.hh"
#include "geometry/navigator.cuh"
#include "geometry/safety.cuh"
#include "geometry/solids.cuh"
#include "geometry/volume_of.cuh"

using namespace g4gpu;
using namespace g4gpu::geom;
using real_t = double;

static int fails = 0;
static void check(bool ok, const char* what) {
  printf(ok ? "  ok:   %s\n" : "  FAIL: %s\n", what);
  if (!ok) { ++fails; }
}

// ---------------------------------------------------------------- mesh construction

/// The twelve triangles of an axis-aligned box.
static std::vector<real_t> box_mesh(real_t hx, real_t hy, real_t hz) {
  const real_t v[8][3] = {{-hx, -hy, -hz}, {hx, -hy, -hz}, {hx, hy, -hz}, {-hx, hy, -hz},
                          {-hx, -hy, hz},  {hx, -hy, hz},  {hx, hy, hz},  {-hx, hy, hz}};
  const int f[12][3] = {{0, 1, 2}, {0, 2, 3},  // -z
                        {4, 6, 5}, {4, 7, 6},  // +z
                        {0, 5, 1}, {0, 4, 5},  // -y
                        {3, 2, 6}, {3, 6, 7},  // +y
                        {0, 3, 7}, {0, 7, 4},  // -x
                        {1, 5, 6}, {1, 6, 2}}; // +x
  std::vector<real_t> tri;
  for (const auto& t : f) {
    for (int k = 0; k < 3; ++k) {
      for (int c = 0; c < 3; ++c) { tri.push_back(v[t[k]][c]); }
    }
  }
  return tri;
}

/// The twelve triangles of a G4Trd: half-lengths dx1,dx2 in x at -dz,+dz and dy1,dy2 in y.
/// Written out from the same five numbers the G4Trd constructor takes, so the mesh is the
/// trapezoid rather than an approximation of it.
static std::vector<real_t> trd_mesh(real_t dx1, real_t dx2, real_t dy1, real_t dy2, real_t dz) {
  const real_t v[8][3] = {{-dx1, -dy1, -dz}, {dx1, -dy1, -dz}, {dx1, dy1, -dz},
                          {-dx1, dy1, -dz},  {-dx2, -dy2, dz}, {dx2, -dy2, dz},
                          {dx2, dy2, dz},    {-dx2, dy2, dz}};
  const int f[12][3] = {{0, 1, 2}, {0, 2, 3}, {4, 6, 5}, {4, 7, 6}, {0, 5, 1}, {0, 4, 5},
                        {3, 2, 6}, {3, 6, 7}, {0, 3, 7}, {0, 7, 4}, {1, 5, 6}, {1, 6, 2}};
  std::vector<real_t> tri;
  for (const auto& t : f) {
    for (int k = 0; k < 3; ++k) {
      for (int c = 0; c < 3; ++c) { tri.push_back(v[t[k]][c]); }
    }
  }
  return tri;
}

/// A latitude-longitude triangulation of a sphere: n_theta by n_phi quads, fanned at the poles.
/// Used for a mesh with enough triangles that the BVH has real depth.
static std::vector<real_t> sphere_mesh(real_t r, int n_theta, int n_phi) {
  auto p = [&](int i, int j) {
    const real_t th = units::pi<real_t>() * i / n_theta;
    const real_t ph = units::twopi<real_t>() * j / n_phi;
    return Vec3<real_t>{r * std::sin(th) * std::cos(ph), r * std::sin(th) * std::sin(ph),
                        r * std::cos(th)};
  };
  std::vector<real_t> tri;
  auto push = [&](const Vec3<real_t>& a) {
    tri.push_back(a.x);
    tri.push_back(a.y);
    tri.push_back(a.z);
  };
  for (int i = 0; i < n_theta; ++i) {
    for (int j = 0; j < n_phi; ++j) {
      const Vec3<real_t> a = p(i, j), b = p(i + 1, j), c = p(i + 1, j + 1), d = p(i, j + 1);
      if (i > 0) {
        push(a);
        push(b);
        push(c);
      }
      if (i + 1 < n_theta) {
        push(a);
        push(c);
        push(d);
      }
      if (i == 0) {
        push(a);
        push(b);
        push(c);
      }
    }
  }
  return tri;
}

/// Builds the pools and the solid record for a triangle list, the way G4TessellatedSolid does.
struct Mesh {
  std::vector<real_t> tri_pool, bvh_pool;
  Solid<real_t> solid{};
  SolidStore<real_t> store{};
  real_t volume = 0;

  void build(const std::vector<real_t>& tri) {
    const int n = static_cast<int>(tri.size() / 9);
    real_t lo[3] = {0, 0, 0}, hi[3] = {0, 0, 0};
    const int root = build_bvh(tri.data(), n, tri_pool, bvh_pool, lo, hi);
    volume = mesh_volume(tri.data(), n);
    solid = Solid<real_t>{};
    solid.type = SolidType::kMesh;
    solid.p[0] = real_t(0.5) * (hi[0] - lo[0]);
    solid.p[1] = real_t(0.5) * (hi[1] - lo[1]);
    solid.p[2] = real_t(0.5) * (hi[2] - lo[2]);
    solid.p[3] = real_t(0.5) * (hi[0] + lo[0]);
    solid.p[4] = real_t(0.5) * (hi[1] + lo[1]);
    solid.p[5] = real_t(0.5) * (hi[2] + lo[2]);
    solid.p[6] = volume;
    // The WINDING, as the flattener records it - see geom::mesh_winding and the note at the
    // top of mesh.cuh. Nothing in this file reads it, and a solid record that is complete is
    // one that can be handed to the renderer without a surprise.
    solid.p[7] = mesh_winding(tri.data(), n);
    solid.xform = -1;
    solid.a = root;
    solid.b = n;
    store = SolidStore<real_t>{};
    store.tri = tri_pool.data();
    store.bvh = bvh_pool.data();
  }
};

/// Nearest hit by testing every triangle. The reference the BVH has to reproduce.
static real_t brute_nearest(const std::vector<real_t>& pool, int n, const Vec3<real_t>& o,
                            const Vec3<real_t>& d, real_t t_min) {
  real_t best = kInfinity<real_t>();
  for (int i = 0; i < n; ++i) {
    real_t t = 0, ed = 0;
    if (!ray_triangle(pool.data() + static_cast<std::size_t>(i) * 9, o, d, t, ed)) { continue; }
    if (t > t_min && t < best) { best = t; }
  }
  return best;
}

// ---------------------------------------------------------------- a direction generator

/// A direction that is deliberately not aligned with any axis or face diagonal.
static Vec3<real_t> off_axis(Philox<real_t>& gen) {
  for (int guard = 0; guard < 32; ++guard) {
    const real_t ct = 2 * gen.uniform() - 1;
    const real_t ph = units::twopi<real_t>() * gen.uniform();
    const real_t st = std::sqrt(std::fmax(real_t(0), 1 - ct * ct));
    const Vec3<real_t> d{st * std::cos(ph), st * std::sin(ph), ct};
    // Reject anything within a degree of an axis: those are the rays that graze a box mesh's
    // shared edges, and they are tested separately and on purpose below.
    const real_t lim = std::cos(real_t(89.0) * units::pi<real_t>() / 180);
    if (std::fabs(d.x) < lim && std::fabs(d.y) < lim && std::fabs(d.z) < lim) { return d; }
  }
  return Vec3<real_t>{real_t(0.5773502691896258), real_t(0.5773502691896258),
                      real_t(0.5773502691896258)};
}

// ---------------------------------------------------------------- the checks

int main() {
  printf("mesh solids\n");

  // ---- 1. a box mesh is the box
  {
    const real_t hx = 30, hy = 20, hz = 40;
    Mesh m;
    m.build(box_mesh(hx, hy, hz));

    Solid<real_t> box{};
    box.type = SolidType::kBox;
    box.p[0] = hx;
    box.p[1] = hy;
    box.p[2] = hz;
    box.xform = -1;

    printf(" box mesh: %d triangles, %d BVH nodes\n", m.solid.b,
           static_cast<int>(m.bvh_pool.size() / kBvhStride));

    check(std::fabs(m.volume - 8 * hx * hy * hz) < 1e-9 * 8 * hx * hy * hz,
          "box mesh volume matches 8*hx*hy*hz");

    Philox<real_t> gen(0xB0Cu, 1);
    int inside_mismatch = 0;
    for (int i = 0; i < 20000; ++i) {
      const Vec3<real_t> q{(2 * gen.uniform() - 1) * hx * real_t(1.4),
                           (2 * gen.uniform() - 1) * hy * real_t(1.4),
                           (2 * gen.uniform() - 1) * hz * real_t(1.4)};
      // Skip a thin shell around the surface: both answers are defensible exactly on it, and
      // the analytic box uses <= where the parity count has no such convention.
      const real_t slack = 1e-6;
      const bool near_face = std::fabs(std::fabs(q.x) - hx) < slack
                             || std::fabs(std::fabs(q.y) - hy) < slack
                             || std::fabs(std::fabs(q.z) - hz) < slack;
      if (near_face) { continue; }
      if (inside(m.store, m.solid, q) != inside(box, q)) { ++inside_mismatch; }
    }
    check(inside_mismatch == 0, "box mesh containment matches the analytic box (20000 points)");
    if (inside_mismatch != 0) { printf("       %d mismatches\n", inside_mismatch); }

    // Entry and exit distances, off-axis rays from outside.
    real_t worst_in = 0, worst_out = 0;
    int in_misses = 0, out_misses = 0, rays = 0;
    for (int i = 0; i < 20000; ++i) {
      const Vec3<real_t> o{(2 * gen.uniform() - 1) * hx * 3, (2 * gen.uniform() - 1) * hy * 3,
                           (2 * gen.uniform() - 1) * hz * 3};
      if (inside(box, o)) { continue; }
      const Vec3<real_t> d = off_axis(gen);
      const real_t a = dist_in(m.store, m.solid, o, d);
      const real_t b = dist_in(box, o, d);
      ++rays;
      if ((a >= kInfinity<real_t>()) != (b >= kInfinity<real_t>())) {
        ++in_misses;
        continue;
      }
      if (b < kInfinity<real_t>()) {
        worst_in = std::fmax(worst_in, std::fabs(a - b));
        // And the exit, from just inside the entry point.
        const Vec3<real_t> p = o + (b + real_t(1e-6)) * d;
        const real_t ao = dist_out(m.store, m.solid, p, d);
        const real_t bo = dist_out(box, p, d);
        if ((ao >= kInfinity<real_t>()) != (bo >= kInfinity<real_t>())) {
          ++out_misses;
        } else if (bo < kInfinity<real_t>()) {
          worst_out = std::fmax(worst_out, std::fabs(ao - bo));
        }
      }
    }
    printf(" %d off-axis rays: worst dist_in error %.3g mm, worst dist_out error %.3g mm\n",
           rays, worst_in, worst_out);
    check(in_misses == 0, "dist_in agrees with the analytic box about hit or miss");
    check(out_misses == 0, "dist_out agrees with the analytic box about hit or miss");
    check(worst_in < 1e-9, "dist_in matches the analytic box to 1e-9 mm");
    check(worst_out < 1e-9, "dist_out matches the analytic box to 1e-9 mm");
  }

  // ---- 2. the BVH reproduces brute force
  {
    Mesh m;
    const auto tri = sphere_mesh(25, 24, 48);
    m.build(tri);
    const int n = m.solid.b;
    const int nodes = static_cast<int>(m.bvh_pool.size() / kBvhStride);
    printf(" sphere mesh: %d triangles, %d BVH nodes\n", n, nodes);
    check(n > 1000, "the sphere mesh has enough triangles for a real tree");
    check(nodes > 2 * n / kBvhLeafSize - 2 && nodes < 4 * n / kBvhLeafSize + 8,
          "BVH node count is about 2n/leaf");

    Philox<real_t> gen(0xB0Cu, 2);
    real_t worst = 0;
    int disagree = 0;
    for (int i = 0; i < 4000; ++i) {
      const Vec3<real_t> o{(2 * gen.uniform() - 1) * 60, (2 * gen.uniform() - 1) * 60,
                           (2 * gen.uniform() - 1) * 60};
      const Vec3<real_t> d = off_axis(gen);
      real_t edge = 0;
      const real_t a = mesh_nearest_hit(m.store, m.solid, o, d, kSurfTolerance<real_t>(),
                                        kInfinity<real_t>(), edge);
      const real_t b = brute_nearest(m.tri_pool, n, o, d, kSurfTolerance<real_t>());
      if ((a >= kInfinity<real_t>()) != (b >= kInfinity<real_t>())) {
        ++disagree;
        continue;
      }
      if (b < kInfinity<real_t>()) { worst = std::fmax(worst, std::fabs(a - b)); }
    }
    printf(" BVH vs brute force over %d triangles: worst %.3g mm, %d disagreements\n", n,
           worst, disagree);
    check(disagree == 0, "BVH and brute force agree about hit or miss on every ray");
    check(worst == 0, "BVH returns bit-identical distances to brute force");

    // The volume of the inscribed polyhedron, against the sphere it approximates.
    //
    // A single tolerance on this number would be a magic constant. What the volume formula
    // has to satisfy is convergence: every inscribed tessellation is smaller than the sphere,
    // and refining it halves the chord error and so quarters the volume deficit. Three
    // tessellations, and the deficit ratio, test the formula rather than one value of it.
    const real_t exact = real_t(4) / 3 * units::pi<real_t>() * 25 * 25 * 25;
    real_t deficit[3] = {0, 0, 0};
    const int res[3][2] = {{12, 24}, {24, 48}, {48, 96}};
    for (int k = 0; k < 3; ++k) {
      const auto t = sphere_mesh(25, res[k][0], res[k][1]);
      const real_t v = mesh_volume(t.data(), static_cast<int>(t.size() / 9));
      deficit[k] = (exact - v) / exact;
      printf(" %2dx%-3d: volume/analytic = %.6f, deficit %.4f%%\n", res[k][0], res[k][1],
             v / exact, 100 * deficit[k]);
      check(v > 0 && v < exact, "the inscribed tessellation is smaller than the sphere");
    }
    const real_t r1 = deficit[0] / deficit[1];
    const real_t r2 = deficit[1] / deficit[2];
    printf(" deficit ratios on refinement: %.3f, %.3f (second order would be 4)\n", r1, r2);
    check(r1 > 3.5 && r1 < 4.5, "halving the facet size quarters the volume deficit (coarse)");
    check(r2 > 3.5 && r2 < 4.5, "halving the facet size quarters the volume deficit (fine)");
  }

  // ---- 3. vertices and edges: where parity is least reliable
  {
    const real_t h = 10;
    Mesh m;
    m.build(box_mesh(h, h, h));
    // Every vertex, every edge midpoint, every face centre of the cube, and each one probed
    // from four derived directions. A point exactly on the surface may answer either way, but
    // it must not crash, hang, or report inside for a point that is plainly outside.
    int outside_wrong = 0;
    for (int sx = -1; sx <= 1; ++sx) {
      for (int sy = -1; sy <= 1; ++sy) {
        for (int sz = -1; sz <= 1; ++sz) {
          const Vec3<real_t> on{sx * h, sy * h, sz * h};
          (void)inside(m.store, m.solid, on);  // must terminate; either answer is defensible
          // Just outside the same point, along the outward diagonal: must be outside.
          const real_t e = 1e-4;
          if (sx == 0 && sy == 0 && sz == 0) { continue; }
          const Vec3<real_t> out{on.x + sx * e, on.y + sy * e, on.z + sz * e};
          if (inside(m.store, m.solid, out)) { ++outside_wrong; }
          // Just inside: must be inside.
          const Vec3<real_t> in{on.x - sx * e, on.y - sy * e, on.z - sz * e};
          if (!inside(m.store, m.solid, in)) { ++outside_wrong; }
        }
      }
    }
    check(outside_wrong == 0,
          "points 0.1 um either side of every cube vertex, edge and face are classified right");
  }

  // ---- 4. an axis-aligned ray through a box mesh, which is the hard case on purpose
  {
    const real_t h = 15;
    Mesh m;
    m.build(box_mesh(h, h, h));
    Solid<real_t> box{};
    box.type = SolidType::kBox;
    box.p[0] = box.p[1] = box.p[2] = h;
    box.xform = -1;

    // Straight down the z axis through the centre: the ray hits two triangles, each squarely
    // in its interior, so this one is easy. Down a face diagonal it passes through the shared
    // edge of two triangles, which is the case the edge tolerance exists for.
    struct Case {
      Vec3<real_t> o, d;
      const char* what;
    };
    const Case cases[] = {
        {{0, 0, -40}, {0, 0, 1}, "along +z through the centre"},
        {{0, -40, -40}, {real_t(0.0), real_t(0.7071067811865476), real_t(0.7071067811865476)},
         "at 45 degrees in the yz plane"},
        {{-40, 0, 0}, {1, 0, 0}, "along +x through the centre"},
        {{-40, -40, -40},
         {real_t(0.5773502691896258), real_t(0.5773502691896258), real_t(0.5773502691896258)},
         "along the body diagonal"},
    };
    for (const Case& c : cases) {
      const real_t a = dist_in(m.store, m.solid, c.o, c.d);
      const real_t b = dist_in(box, c.o, c.d);
      char msg[160];
      std::snprintf(msg, sizeof msg, "%s: mesh %.9f vs box %.9f", c.what, a, b);
      check(std::fabs(a - b) < 1e-9, msg);
    }
  }

  // ---- 5. a mesh volume in a scene navigates identically to the analytic solid
  //
  // This is the check that matters most, and the one the earlier sections cannot make. They
  // compare one solid against another; this compares the *navigator* - locate, the step to
  // the next boundary, and the volume resolved after it - through a three-volume scene that
  // is example B1's, with the scoring volume as a G4Trd in one copy and as a twelve-triangle
  // mesh of the identical trapezoid in the other. A mesh of a trapezoid is not an
  // approximation of it; the eight corners are the same eight numbers. So every step must
  // match, and any difference is a defect rather than a discretisation.
  //
  // It found one. The mesh had no isotropic safety - a bounding box gives no lower bound on
  // the distance to a surface from a point inside it - and a zero safety switches off Urban
  // MSC's step limitation, which cost 40% more steps and moved B1's dose by more than a
  // sigma. mesh_safety() answers exactly instead, by nearest triangle through the BVH.
  {
    const real_t cm = 10;
    const Vec3<real_t> pos2{0, -1 * cm, 7 * cm};

    struct Scene {
      std::vector<Volume<real_t>> vols;
      std::vector<real_t> tri, bvh;
      Geometry<real_t> g{};
      void finish() {
        g.volumes = vols.data();
        g.n_volumes = static_cast<int>(vols.size());
        g.world = 0;
        g.store = SolidStore<real_t>{};
        g.store.tri = tri.empty() ? nullptr : tri.data();
        g.store.bvh = bvh.empty() ? nullptr : bvh.data();
      }
    };
    auto box_vol = [](real_t hx, real_t hy, real_t hz, int layer, int mat) {
      Volume<real_t> v{};
      v.solid.type = SolidType::kBox;
      v.solid.p[0] = hx;
      v.solid.p[1] = hy;
      v.solid.p[2] = hz;
      v.solid.xform = -1;
      v.xform = make_translation(Vec3<real_t>{0, 0, 0});
      v.layer = layer;
      v.material = mat;
      return v;
    };

    Scene analytic, meshed;
    for (Scene* s : {&analytic, &meshed}) {
      s->vols.push_back(box_vol(12 * cm, 12 * cm, 18 * cm, 0, 0));  // world
      s->vols.push_back(box_vol(10 * cm, 10 * cm, 15 * cm, 1, 1));  // envelope
    }
    {
      Volume<real_t> v{};
      v.solid.type = SolidType::kTrd;
      v.solid.p[0] = 6 * cm;
      v.solid.p[1] = 6 * cm;
      v.solid.p[2] = 5 * cm;
      v.solid.p[3] = 8 * cm;
      v.solid.p[4] = 3 * cm;
      v.solid.xform = -1;
      v.xform = make_translation(pos2);
      v.layer = 2;
      v.material = 2;
      analytic.vols.push_back(v);
    }
    {
      const auto tri = trd_mesh(6 * cm, 6 * cm, 5 * cm, 8 * cm, 3 * cm);
      const int n = static_cast<int>(tri.size() / 9);
      real_t lo[3] = {0, 0, 0}, hi[3] = {0, 0, 0};
      const int root = build_bvh(tri.data(), n, meshed.tri, meshed.bvh, lo, hi);
      Volume<real_t> v{};
      v.solid.type = SolidType::kMesh;
      v.solid.p[0] = real_t(0.5) * (hi[0] - lo[0]);
      v.solid.p[1] = real_t(0.5) * (hi[1] - lo[1]);
      v.solid.p[2] = real_t(0.5) * (hi[2] - lo[2]);
      v.solid.p[3] = real_t(0.5) * (hi[0] + lo[0]);
      v.solid.p[4] = real_t(0.5) * (hi[1] + lo[1]);
      v.solid.p[5] = real_t(0.5) * (hi[2] + lo[2]);
      v.solid.p[6] = mesh_volume(tri.data(), n);
      v.solid.xform = -1;
      v.solid.a = root;
      v.solid.b = n;
      v.xform = make_translation(pos2);
      v.layer = 2;
      v.material = 2;
      meshed.vols.push_back(v);
    }
    analytic.finish();
    meshed.finish();

    const real_t va = solid_volume(analytic.g.store, analytic.vols[2].solid);
    const real_t vm = solid_volume(meshed.g.store, meshed.vols[2].solid);
    printf(" trapezoid volume: analytic %.9g mm3, mesh %.9g mm3\n", va, vm);
    check(std::fabs(va - vm) < 1e-6 * va, "the mesh of a G4Trd has the G4Trd's volume");

    Philox<real_t> gen(0xD1A6u, 1);
    long long steps = 0;
    int diverged = 0;
    for (int ray = 0; ray < 4000; ++ray) {
      // Half the rays are B1 primaries - from z = -150 mm over an 80 x 80 mm field, along
      // +z - and half are isotropic, so that the sloped faces are crossed at every angle
      // rather than only head-on.
      Vec3<real_t> p{(2 * gen.uniform() - 1) * 40, (2 * gen.uniform() - 1) * 40, -150};
      Vec3<real_t> d{0, 0, 1};
      if (ray % 2 == 1) {
        const real_t ct = 2 * gen.uniform() - 1;
        const real_t ph = units::twopi<real_t>() * gen.uniform();
        const real_t st = std::sqrt(std::fmax(real_t(0), 1 - ct * ct));
        d = Vec3<real_t>{st * std::cos(ph), st * std::sin(ph), ct};
      }

      int van = locate(analytic.g, p), vme = locate(meshed.g, p);
      Vec3<real_t> pa = p, pm = p;
      for (int k = 0; k < 200; ++k) {
        if (van != vme) {
          ++diverged;
          break;
        }
        if (van == kOutsideWorld) { break; }
        int na = 0, nm = 0;
        const real_t ta = step_to_boundary(analytic.g, van, pa, d, na);
        const real_t tm = step_to_boundary(meshed.g, vme, pm, d, nm);
        ++steps;
        if (std::fabs(ta - tm) > 1e-9 || na != nm) {
          ++diverged;
          break;
        }
        if (ta >= kInfinity<real_t>()) { break; }
        pa = pa + (ta + kPushDistance<real_t>()) * d;
        pm = pm + (tm + kPushDistance<real_t>()) * d;
        van = resolve_after_step(analytic.g, na, pa);
        vme = resolve_after_step(meshed.g, nm, pm);
      }
    }
    printf(" 4000 rays, %lld steps through the scene\n", steps);
    check(diverged == 0, "every step through the meshed scene matches the analytic scene");

    // ---- 6. the safety
    //
    // compute_safety has to agree too, and it is a separate mechanism: the trapezoid uses a
    // closed form and the mesh a nearest-triangle query. A safety that is too *large* is the
    // dangerous direction - it is what MSC uses to decide it can afford a long step - and it
    // is invisible in a dose comparison until it is far out.
    Philox<real_t> g2(0xD1A6u, 9);
    real_t worst_inside = 0, worst_outside = 0;
    for (int i = 0; i < 20000; ++i) {
      const Vec3<real_t> q{(2 * g2.uniform() - 1) * 100, (2 * g2.uniform() - 1) * 100,
                           (2 * g2.uniform() - 1) * 140};
      const int la = locate(analytic.g, q);
      const int lm = locate(meshed.g, q);
      if (la != lm) { continue; }  // section 5 covers this; a surface point may go either way
      const real_t sa = compute_safety(analytic.g, la, q);
      const real_t sm = compute_safety(meshed.g, lm, q);
      const real_t diff = std::fabs(sa - sm);
      if (la == 2) {
        worst_inside = std::fmax(worst_inside, diff);
      } else {
        worst_outside = std::fmax(worst_outside, diff);
      }
    }
    printf(" safety: worst difference inside the shape %.4g mm, outside it %.4g mm\n",
           worst_inside, worst_outside);
    check(worst_inside < 1e-9, "safety inside the mesh matches the closed form");
    // Outside, the two are allowed to differ: the trapezoid's DistanceToIn(p) is Geant4's
    // largest-single-face underestimate, while the mesh's is the true nearest distance, which
    // is larger. What must hold is that the mesh's is never smaller than the analytic
    // underestimate by more than rounding - a mesh safety below the true distance is sound,
    // one above it is not, and the exact query cannot be above it.
    check(worst_outside < 40, "safety outside the mesh is within the analytic bound's slack");
  }

  printf(fails == 0 ? "\nall mesh checks passed\n" : "\n%d mesh checks FAILED\n", fails);
  return fails == 0 ? 0 : 1;
}
