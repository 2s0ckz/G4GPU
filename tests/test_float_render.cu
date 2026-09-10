// The float copy the render pass walks has to be a VALID geometry, not merely a near one.
//
// WHY THIS EXISTS
//
// The renderer calls the transport's geometry functions so that the picture cannot disagree
// with the physics about where a surface is. The transport is double; on a GeForce card double
// is a throughput cliff, so the render pass is given a float copy of the scene.
//
// The worry worth having about that copy is the BVH. A box converted INWARD by one ulp no
// longer contains its own triangles, and the traversal rejects any subtree whose box the ray
// misses - so the mesh would not be shaded a little wrong, it would have a hole along the
// seam.
//
// IT CANNOT HAPPEN, AND THIS IS THE TEST THAT SAYS SO RATHER THAN A COMMENT THAT HOPES SO.
// `geom::build_bvh` sets every box bound to the min or max of vertex COORDINATES, so each
// bound IS one of the numbers that will be compared against it, and rounding to float is
// monotonic. Containment therefore survives a plain cast exactly - which is why
// render/float_geometry.cuh is a cast and nothing more. The first version of that file spent
// sixty lines recomputing every box from the float triangles; this test is what showed the
// lines were unnecessary, and it is kept so that a change to build_bvh which COMPUTES a bound
// rather than picking one is caught here instead of appearing as holes in a CAD import.
//
// The last check is the rays, and it carries one lesson worth stating. Aimed exactly at the
// sphere's centre from a lattice of directions, 14 of 4000 rays that hit in double missed in
// float - and 21 of those same rays missed in DOUBLE too. Those rays land on the shared edges
// of a UV sphere, where Moller-Trumbore's barycentric test can reject both adjacent triangles
// and leave a hole. That is a property of the intersector at both precisions, not of the
// conversion: jitter the aim by a third of a millimetre, off the lattice, and the two agree on
// every one of 4000 rays to within 0.0007 mm. So the sweep is deliberately jittered, and the
// alignment case is recorded here rather than asserted, because asserting it would be
// asserting a known limit of ray_triangle.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include <vector>

#include "geometry/bvh_build.hh"
#include "geometry/mesh.cuh"
#include "geometry/safety.cuh"
#include "render/float_geometry.cuh"
#include "render/renderer.cuh"

using namespace g4gpu;
using namespace g4gpu::geom;

namespace {

int g_fails = 0;
void Check(bool ok, const char* what) {
  std::printf(ok ? "  ok:   %s\n" : "  FAIL: %s\n", what);
  if (!ok) { ++g_fails; }
}

/// A UV sphere, offset far from the origin so the coordinates are large enough for float
/// rounding to bite. A coordinate of 400 mm has a float ulp of 3e-5 mm; at the origin there is
/// nothing to round, and a fixture at the origin would pass whatever the conversion did.
std::vector<double> Sphere(int bands, double radius, const double centre[3]) {
  std::vector<double> tri;
  const int nu = bands * 2, nv = bands;
  auto at = [&](int i, int j, double out[3]) {
    const double u = 2.0 * 3.14159265358979 * i / nu;
    const double v = 3.14159265358979 * j / nv;
    out[0] = centre[0] + radius * std::sin(v) * std::cos(u);
    out[1] = centre[1] + radius * std::sin(v) * std::sin(u);
    out[2] = centre[2] + radius * std::cos(v);
  };
  for (int j = 0; j < nv; ++j) {
    for (int i = 0; i < nu; ++i) {
      double p[4][3];
      at(i, j, p[0]);
      at(i + 1, j, p[1]);
      at(i + 1, j + 1, p[2]);
      at(i, j + 1, p[3]);
      const int idx[2][3] = {{0, 1, 2}, {0, 2, 3}};
      for (const auto& q : idx) {
        for (int k = 0; k < 3; ++k) {
          for (int c = 0; c < 3; ++c) { tri.push_back(p[q[k]][c]); }
        }
      }
    }
  }
  return tri;
}

/// Walks the tree checking that each node's box holds its subtree's triangles and sits inside
/// the box it was handed. Returns the number of violations.
int CheckSubtree(const std::vector<float>& bvh, const std::vector<float>& tri, int ni,
                 const float* parent_lo, const float* parent_hi, int* nodes) {
  if (ni < 0 || static_cast<std::size_t>(ni + 1) * kBvhStride > bvh.size()) { return 0; }
  const float* node = bvh.data() + static_cast<std::size_t>(ni) * kBvhStride;
  const int first = static_cast<int>(node[6]);
  const int count = static_cast<int>(node[7]);
  ++*nodes;
  int bad = 0;
  if (parent_lo != nullptr) {
    for (int k = 0; k < 3; ++k) {
      if (node[k] < parent_lo[k] || node[3 + k] > parent_hi[k]) { ++bad; }
    }
  }
  if (count == 0) {
    bad += CheckSubtree(bvh, tri, first, node, node + 3, nodes);
    bad += CheckSubtree(bvh, tri, first + 1, node, node + 3, nodes);
    return bad;
  }
  for (int t = 0; t < count; ++t) {
    const std::size_t v = static_cast<std::size_t>(first + t) * 9;
    if (v + 9 > tri.size()) { continue; }
    for (int c = 0; c < 3; ++c) {
      for (int k = 0; k < 3; ++k) {
        const float x = tri[v + static_cast<std::size_t>(c) * 3 + k];
        if (x < node[k] || x > node[3 + k]) { ++bad; }
      }
    }
  }
  return bad;
}

}  // namespace

int main() {
  std::printf("== the float render geometry is a valid geometry ==\n");

  // A sphere 120 mm across, centred 400 mm out on every axis - a phantom's worth of offset in
  // this project's default 500 mm world.
  const double centre[3] = {400.0, -350.0, 420.0};
  const std::vector<double> tri_d = Sphere(40, 60.0, centre);
  const int n_tri = static_cast<int>(tri_d.size() / 9);

  std::vector<double> tri_pool, bvh_pool;
  double lo[3] = {0, 0, 0}, hi[3] = {0, 0, 0};
  const int root = build_bvh(tri_d.data(), n_tri, tri_pool, bvh_pool, lo, hi);

  Solid<double> mesh{};
  mesh.type = SolidType::kMesh;
  for (int k = 0; k < 3; ++k) {
    mesh.p[k] = 0.5 * (hi[k] - lo[k]);
    mesh.p[3 + k] = 0.5 * (hi[k] + lo[k]);
  }
  mesh.p[6] = mesh_volume(tri_d.data(), n_tri);
  mesh.xform = -1;
  mesh.a = root;
  mesh.b = n_tri;

  std::vector<Volume<double>> vols(2);
  vols[0].solid = Solid<double>{SolidType::kBox, {600, 600, 600}};
  vols[0].xform = make_translation<double>({0, 0, 0});
  vols[0].layer = 0;
  vols[0].material = 0;
  vols[1].solid = mesh;
  vols[1].xform = make_translation<double>({0, 0, 0});
  vols[1].layer = 1;
  vols[1].material = 1;

  std::vector<Solid<double>> solids{vols[0].solid, mesh};

  vis::HostGeometry h{};
  h.volumes = vols.data();
  h.n_volumes = 2;
  h.world = 0;
  h.solids = solids.data();
  h.n_solids = static_cast<int>(solids.size());
  h.tri = tri_pool.data();
  h.n_tri = static_cast<int>(tri_pool.size());
  h.bvh = bvh_pool.data();
  h.n_bvh = static_cast<int>(bvh_pool.size());

  vis::FloatPools f;
  vis::ConvertToFloat(h, f);
  std::printf("  %d triangles, %zu nodes, centred at (%g %g %g)\n", n_tri,
              bvh_pool.size() / kBvhStride, centre[0], centre[1], centre[2]);

  // ---- 1. Every box holds its triangles and sits inside its parent's, AFTER THE CAST.
  //
  // This is the property float_geometry.cuh relies on to be a cast and nothing more. It holds
  // because build_bvh picks each bound from the vertices rather than computing it; if that
  // ever changes, this is where it shows.
  {
    int nodes = 0;
    const int bad = CheckSubtree(f.bvh, f.tri, root, nullptr, nullptr, &nodes);
    std::printf("  walked %d nodes, %d containment violations\n", nodes, bad);
    Check(nodes > 100, "the tree has interior nodes, so the walk means something");
    Check(bad == 0,
          "casting the boxes keeps every one of them around its own triangles, because "
          "build_bvh picks its bounds from the vertices");
  }

  // ---- 2. The mesh solid's own box, which mesh_inside tests before it walks anything.
  {
    const Solid<float>& sf = f.volumes[1].solid;
    int bad = 0;
    for (int t = 0; t < n_tri; ++t) {
      for (int c = 0; c < 3; ++c) {
        for (int k = 0; k < 3; ++k) {
          const float x = f.tri[static_cast<std::size_t>(t) * 9 + c * 3 + k];
          if (std::fabs(x - sf.p[3 + k]) > sf.p[k]) { ++bad; }
        }
      }
    }
    Check(bad == 0, "the mesh solid's own float box holds every float triangle");
  }

  // ---- 3. And the whole struct came across: a field left behind is a volume on the wrong
  // layer or with the wrong material, which no ray test would notice.
  {
    bool same = true;
    for (int i = 0; i < 2; ++i) {
      const Volume<double>& d = vols[static_cast<std::size_t>(i)];
      const Volume<float>& o = f.volumes[static_cast<std::size_t>(i)];
      same = same && o.layer == d.layer && o.material == d.material
             && o.score_index == d.score_index && o.score_per_voxel == d.score_per_voxel
             && o.has_class_layers == d.has_class_layers && o.layer_lo == d.layer_lo
             && o.layer_hi == d.layer_hi && o.solid.type == d.solid.type
             && o.solid.a == d.solid.a && o.solid.b == d.solid.b
             && o.solid.xform == d.solid.xform;
    }
    Check(same, "every non-real field of every volume and solid came across unchanged");
  }

  // ---- 4. And the rays agree.
  //
  // Jittered off the mesh's own lattice - see the note at the top of this file for the 14
  // rays that say why, and for the 21 that miss in double as well.
  {
    SolidStore<double> sd{};
    sd.tri = tri_pool.data();
    sd.bvh = bvh_pool.data();
    SolidStore<float> sf{};
    sf.tri = f.tri.data();
    sf.bvh = f.bvh.data();
    const Solid<float>& mf = f.solids[1];

    int hits_d = 0, missing = 0, extra = 0, worst_at = 0;
    double worst = 0;
    for (int i = 0; i < 4000; ++i) {
      // A deterministic sweep of directions and aim points, so a failure is reproducible.
      const double a = 6.28318530718 * (i % 79) / 79.0;
      const double b = 3.14159265359 * ((i / 79) % 51) / 50.0;
      const Vec3<double> dir{std::sin(b) * std::cos(a), std::sin(b) * std::sin(a),
                             std::cos(b)};
      const double tx = centre[0] + 0.37 * std::sin(i * 1.7);
      const double ty = centre[1] + 0.37 * std::cos(i * 2.3);
      const double tz = centre[2] + 0.37 * std::sin(i * 0.9);
      const Vec3<double> o{tx - 500.0 * dir.x, ty - 500.0 * dir.y, tz - 500.0 * dir.z};
      double ed = 0;
      int td = -1;
      const double t_d = mesh_nearest_hit(sd, mesh, o, dir, kSurfTolerance<double>(),
                                          kInfinity<double>(), ed, &td);
      float ef = 0;
      int tf = -1;
      const Vec3<float> of{static_cast<float>(o.x), static_cast<float>(o.y),
                           static_cast<float>(o.z)};
      const Vec3<float> df{static_cast<float>(dir.x), static_cast<float>(dir.y),
                           static_cast<float>(dir.z)};
      const float t_f = mesh_nearest_hit(sf, mf, of, df, kSurfTolerance<float>(),
                                         kInfinity<float>(), ef, &tf);
      const bool hd = t_d < kInfinity<double>();
      const bool hf = t_f < kInfinity<float>();
      if (hd) { ++hits_d; }
      if (hd && !hf) { ++missing; }
      if (!hd && hf) { ++extra; }
      if (hd && hf) {
        const double d = std::fabs(static_cast<double>(t_f) - t_d);
        if (d > worst) {
          worst = d;
          worst_at = i;
        }
      }
    }
    std::printf("  %d of 4000 rays hit in double; %d missing in float, %d extra; worst "
                "distance difference %.3g mm (ray %d)\n",
                hits_d, missing, extra, worst, worst_at);
    Check(hits_d == 4000, "every ray hits the sphere in double, so a miss in float is real");
    Check(missing == 0, "no ray that hits in double misses in float");
    Check(extra == 0, "and none that misses in double hits in float");
    Check(worst < 1e-2, "the two precisions put the surface in the same place, to 0.01 mm");
  }

  // ---- 5. A CURVED SOLID MUST NOT LOSE RAYS IN FLOAT, AT ANY CAMERA DISTANCE.
  //
  // Reported as "sphere and orb render with ray tracing artifacts, and I can't see the
  // ellipsoid at all". Two causes, both float-only, both invisible in double:
  //
  //   * the generic engine solves a quadratic whose discriminant is b^2 - 4ac. For a 60 mm
  //     sphere seen from 1500 mm that is 9.00e6 - 8.99e6 = 1.44e4 - two numbers agreeing to
  //     three digits, which in float leaves four. The root lands about 0.004 mm out,
  //     `is_crossing_to` probes 0.001 mm either side of it, both probes fall on the same side
  //     of the surface, and the crossing is not seen. The renderer therefore starts the ray at
  //     the solid's bounding sphere, where b and c are both O(radius) and nothing cancels.
  //
  //   * `quadric_inside` compared the raw quadric value against a tolerance that is a LENGTH.
  //     An orb is written x^2+y^2+z^2-r^2, gradient 120 per mm at r=60; an ellipsoid is
  //     written x^2/a^2+...-1, gradient 0.04 per mm. One tolerance is a band 8e-7 mm wide on
  //     one and 2.5e-3 mm on the other - wider than the probe - so on the ellipsoid BOTH sides
  //     of a crossing reported "on the surface" and it had no surfaces at all. The residual is
  //     now divided by the gradient, which is what quadric_normal beside it always did.
  //
  // The shape of the failure is what identifies it, so the sweep is over DISTANCE: 100% at
  // 200 mm, 47% at 1500 and 10% at 4000 is a cancelling discriminant, and 0.1% everywhere is a
  // tolerance in the wrong units.
  //
  // EVERY SOLID THE RENDERER CAN DRAW, and not the four that were suspected. The first version
  // of this swept orb, sphere, ellipsoid and tubs, and a note beside it said box, trd and cons
  // use closed forms and "were never affected - which the measurement said before anything was
  // changed". The measurement said nothing of the kind: those three were not in it. A cone with
  // an inner radius or a phi wedge does not take the closed form at all (see G4Cons::Build) and
  // goes through the same generic path the quadrics do - reported as the cone showing the same
  // artifact as the sphere, at some zooms and not others, which is this failure's signature.
  //
  // So the list is the enum, not a hunch.
  {
    std::printf("\n  every solid, rays kept in float against double:\n");
    struct Case {
      const char* name;
      SolidType type;
      double aim_lo, aim_hi;   ///< where to aim, so the shape is hit where it has material
      double p[8];
    };
    // p[] in each solid's own order - see the make(...) calls in src/g4/G4Solids.hh, which is
    // the only place that order is written down. aim_lo/aim_hi matter as much: a ray down the
    // axis of a HOLLOW tube never enters the material, so double misses it too and the row
    // says nothing. Each shape is aimed where it actually is.
    static const Case kCases[] = {
        {"box",          SolidType::kBox,             0, 30, {60, 50, 40}},
        {"cons",         SolidType::kCons,            0, 20, {60, 30, 50}},
        {"trd",          SolidType::kTrd,             0, 20, {60, 30, 50, 25, 40}},
        {"tubs",         SolidType::kTubs,            0, 40, {0, 60, 60, 0, 6.28318530718}},
        {"tubs-hollow",  SolidType::kTubs,           28, 52, {20, 60, 60, 0, 6.28318530718}},
        {"conesection",  SolidType::kConeSection,     0, 20,
                                                     {0, 60, 0, 30, 50, 0, 6.28318530718}},
        {"conesec-ann",  SolidType::kConeSection,    24, 40,
                                                     {20, 60, 10, 30, 50, 0, 6.28318530718}},
        {"orb",          SolidType::kOrb,             0, 40, {60}},
        {"sphere",       SolidType::kSphere,          0, 40, {0, 60, 0, 6.28318530718, 0,
                                                              3.14159265359}},
        {"sphere-shell", SolidType::kSphere,         45, 57, {40, 60, 0, 6.28318530718, 0,
                                                              3.14159265359}},
        {"torus",        SolidType::kTorus,           0, 30, {0, 25, 20, 0, 6.28318530718}},
        {"para",         SolidType::kPara,            0, 30, {50, 40, 45, 0.2, 0.1, 0.15}},
        {"elliptube",    SolidType::kEllipticalTube,  0, 30, {50, 35, 45}},
        {"ellipsoid",    SolidType::kEllipsoid,       0, 30, {60, 40, 50, -50, 50}},
        {"ellipcone",    SolidType::kEllipticalCone,  0, 15, {0.5, 0.4, 100, 40}},
        {"paraboloid",   SolidType::kParaboloid,      0, 30, {45, 10, 55}},
        {"hype",         SolidType::kHype,            0, 30, {0, 40, 0.15, 0.25, 45}},
    };

    // The renderer's own rule, so this measures what it measures. See render_geometry.
    auto hits = [](const Case& c, double cam, int n, bool as_float) {
      int hit = 0;
      for (int i = 0; i < n; ++i) {
        // Uniform over the aim annulus, at the golden angle so the rays do not fall on any
        // lattice the shape might have.
        const double f = (i + 0.5) / n;
        const double r = std::sqrt(c.aim_lo * c.aim_lo
                                   + f * (c.aim_hi * c.aim_hi - c.aim_lo * c.aim_lo));
        const double ang = 2.399963229 * i;
        const double tx = r * std::cos(ang), ty = r * std::sin(ang);
        const double L = std::sqrt(tx * tx + ty * ty + cam * cam);
        if (as_float) {
          Solid<float> s{};
          s.type = c.type;
          for (int k = 0; k < 8; ++k) { s.p[k] = static_cast<float>(c.p[k]); }
          const Vec3<float> o{0.f, 0.f, static_cast<float>(-cam)};
          const Vec3<float> d{static_cast<float>(tx / L), static_cast<float>(ty / L),
                              static_cast<float>(cam / L)};
          const float reach = bounding_radius(s);
          float skip = 0.f;
          if (reach > 0.f) {
            const float approach = -dot(o, d);
            skip = approach - reach * 1.01f - 1.f;
            if (skip < 0.f) { skip = 0.f; }
          }
          if (dist_in(s, o + skip * d, d) < kInfinity<float>()) { ++hit; }
        } else {
          Solid<double> s{};
          s.type = c.type;
          for (int k = 0; k < 8; ++k) { s.p[k] = c.p[k]; }
          const Vec3<double> o{0.0, 0.0, -cam};
          const Vec3<double> d{tx / L, ty / L, cam / L};
          const double reach = bounding_radius(s);
          double skip = 0.0;
          if (reach > 0.0) {
            const double approach = -dot(o, d);
            skip = approach - reach * 1.01 - 1.0;
            if (skip < 0.0) { skip = 0.0; }
          }
          if (dist_in(s, o + skip * d, d) < kInfinity<double>()) { ++hit; }
        }
      }
      return hit;
    };

    constexpr int kRays = 1200;
    double worst_ratio = 1.0;
    const char* worst_shape = "";
    double worst_cam = 0;
    int too_few = 0;
    const char* too_few_shape = "";
    for (double cam : {200.0, 600.0, 1500.0, 4000.0, 12000.0}) {
      std::printf("    %7.0f mm ", cam);
      for (const Case& c : kCases) {
        const int nd = hits(c, cam, kRays, false);
        const int nf = hits(c, cam, kRays, true);
        // FLOAT AGAINST DOUBLE, per shape, and not against the ray count. What double finds
        // is the truth this is measuring against; a shape double only partly hits - a wedge,
        // a shell seen head-on - is still a valid comparison, and demanding double hit
        // everything only tempts the fixture to be tuned until it does.
        //
        // Separately, double has to find ENOUGH for the ratio to mean something. A row where
        // double hits nothing would otherwise pass with float hitting nothing either, which is
        // how a torus aimed down its own hole came to look like a clean result.
        if (nd < kRays / 4) {
          ++too_few;
          too_few_shape = c.name;
        } else {
          const double ratio = static_cast<double>(nf) / nd;
          if (ratio < worst_ratio) {
            worst_ratio = ratio;
            worst_shape = c.name;
            worst_cam = cam;
          }
        }
        std::printf(" %s %d/%d", c.name, nf, nd);
      }
      std::printf("\n");
    }
    if (too_few > 0) {
      std::printf("    %d rows had too few double hits to compare (last: %s)\n", too_few,
                  too_few_shape);
    }
    Check(too_few == 0, "every solid in the sweep is aimed at where it has material, so each "
                        "row is a comparison and not a shrug");
    std::printf("    worst: %s at %.0f mm kept %.4f of what double found\n", worst_shape,
                worst_cam, worst_ratio);
    // 99%, not 100%: a ray exactly tangent to a surface is a genuine coin flip at any
    // precision, and the sweep is dense enough to find a few.
    Check(worst_ratio >= 0.99,
          "float keeps at least 99% of the rays double finds, on every solid at every camera "
          "distance");
  }

  // ---- 5b. AND THE TWO THAT KEEP THEIR SHAPE IN THE AUX POOL.
  //
  // kPolycone and kPolyhedra carry their z-sections outside p[], so they cannot be built as a
  // bare Solid and were left out of the sweep above. That is exactly why they matter here: the
  // builder's "Cone" primitive is a POLYCONE, not a G4Cons - see InsertPrimitive - so the cone
  // in a selftest picture is the one solid family nothing had ever compared in float.
  //
  // Aimed across the whole silhouette out to 98% of the widest section, which is the other
  // half of what the sweep above does not do: its rays are all near the axis, where every
  // surface is nearly head-on. The sphere's cancelling discriminant showed up at the LIMB,
  // where the surface is nearly parallel to the ray and the two roots are nearly equal - so a
  // sweep that never goes near the limb is a sweep that cannot see it.
  {
    std::printf("\n  aux-pool solids, rays kept in float against double:\n");
    // A cone: wide at -z, a point at +z, in three sections so the taper changes slope.
    // THE BUILDER'S OWN Cone PRIMITIVE, which is a bicone: widest in the middle, so the
    // surface has an outward CREASE at z = 0 where the slope reverses. A monotonic taper has
    // no such edge, and a taper was what the first version of this swept.
    const double sec[] = {-30, 0, 10,   0, 0, 25,   30, 0, 5};
    const int n_sec = 3;

    std::vector<double> aux_d(sec, sec + n_sec * 3);
    std::vector<float> aux_f;
    for (double v : aux_d) { aux_f.push_back(static_cast<float>(v)); }

    SolidStore<double> sd{};
    sd.aux = aux_d.data();
    SolidStore<float> sf{};
    sf.aux = aux_f.data();

    struct Case { const char* name; SolidType type; double p2; };
    static const Case kCases[] = {
        {"polycone",  SolidType::kPolycone,  6.28318530718},
        {"polyhedra", SolidType::kPolyhedra, 6.28318530718},
    };

    auto hits = [&](const Case& c, double cam, double aim, int n, bool as_float) {
      int hit = 0;
      for (int i = 0; i < n; ++i) {
        const double r = aim * std::sqrt((i + 0.5) / n);
        const double ang = 2.399963229 * i;
        const double tx = r * std::cos(ang), ty = r * std::sin(ang);
        const double L = std::sqrt(tx * tx + ty * ty + cam * cam);
        if (as_float) {
          Solid<float> s{};
          s.type = c.type;
          s.p[0] = 0.f;
          s.p[1] = static_cast<float>(c.p2);
          s.p[2] = 6.f;   // sides, for polyhedra; ignored by polycone
          s.a = 0;
          s.b = n_sec - 1;
          const Vec3<float> o{0.f, 0.f, static_cast<float>(-cam)};
          const Vec3<float> d{static_cast<float>(tx / L), static_cast<float>(ty / L),
                              static_cast<float>(cam / L)};
          const float reach = bounding_radius(sf, s);
          float skip = 0.f;
          if (reach > 0.f) {
            const float approach = -dot(o, d);
            skip = approach - reach * 1.01f - 1.f;
            if (skip < 0.f) { skip = 0.f; }
          }
          if (dist_in(sf, s, o + skip * d, d) < kInfinity<float>()) { ++hit; }
        } else {
          Solid<double> s{};
          s.type = c.type;
          s.p[0] = 0.0;
          s.p[1] = c.p2;
          s.p[2] = 6.0;
          s.a = 0;
          s.b = n_sec - 1;
          const Vec3<double> o{0.0, 0.0, -cam};
          const Vec3<double> d{tx / L, ty / L, cam / L};
          const double reach = bounding_radius(sd, s);
          double skip = 0.0;
          if (reach > 0.0) {
            const double approach = -dot(o, d);
            skip = approach - reach * 1.01 - 1.0;
            if (skip < 0.0) { skip = 0.0; }
          }
          if (dist_in(sd, s, o + skip * d, d) < kInfinity<double>()) { ++hit; }
        }
      }
      return hit;
    };

    constexpr int kRays = 2000;
    double worst_ratio = 1.0;
    const char* worst_shape = "";
    double worst_cam = 0;
    int too_few = 0;
    for (double cam : {200.0, 600.0, 1500.0, 4000.0, 12000.0}) {
      std::printf("    %7.0f mm ", cam);
      for (const Case& c : kCases) {
        for (double aim : {8.0, 24.5}) {   // near the axis, and out at the limb
          const int nd = hits(c, cam, aim, kRays, false);
          const int nf = hits(c, cam, aim, kRays, true);
          if (nd < kRays / 4) {
            ++too_few;
          } else {
            const double ratio = static_cast<double>(nf) / nd;
            if (ratio < worst_ratio) {
              worst_ratio = ratio;
              worst_shape = c.name;
              worst_cam = cam;
            }
          }
          std::printf(" %s@%.1f %d/%d", c.name, aim, nf, nd);
        }
      }
      std::printf("\n");
    }
    Check(too_few == 0, "both aux-pool solids are hit by enough rays in double to compare");
    std::printf("    worst: %s at %.0f mm kept %.4f of what double found\n", worst_shape,
                worst_cam, worst_ratio);
    Check(worst_ratio >= 0.99,
          "float keeps at least 99% of the rays double finds on a polycone and a polyhedra, "
          "at the limb as well as near the axis");
  }

  // ---- 5c. AND A BOOLEAN, WHICH HANDS ITS CHILDREN THE ORIGIN IT WAS GIVEN.
  //
  // boolean_dist recurses with the same ray, so a union of two orbs solves the orbs'
  // quadratics from wherever the caller started - and a boolean has no p[] extent, so it fell
  // through bounding_radius to zero and the origin shift never happened. Same bug as the
  // polycone above, reached through a different hole in the same function.
  //
  // Two orbs offset along x by more than their radius, so the union has a waist and its
  // silhouette is not a circle: rays near the waist are the grazing ones.
  {
    std::printf("\n  a boolean of two orbs, rays kept in float against double:\n");
    // Two pools rather than one generic builder: a lambda cannot take a container whose
    // element type depends on an earlier `auto` parameter, and spelling it twice is shorter
    // than the template that would.
    auto hits = [&](SolidType op, double cam, double aim, int n, bool as_float) {
      int hit = 0;
      std::vector<Transform<float>> xff{make_translation<float>({-25.f, 0.f, 0.f}),
                                        make_translation<float>({25.f, 0.f, 0.f})};
      std::vector<Transform<double>> xfd{make_translation<double>({-25.0, 0.0, 0.0}),
                                         make_translation<double>({25.0, 0.0, 0.0})};
      std::vector<Solid<float>> pf(2);
      std::vector<Solid<double>> pd(2);
      for (int k = 0; k < 2; ++k) {
        pf[k].type = SolidType::kOrb;
        pf[k].p[0] = 40.f;
        pf[k].xform = k;
        pd[k].type = SolidType::kOrb;
        pd[k].p[0] = 40.0;
        pd[k].xform = k;
      }
      Solid<float> nf{};
      nf.type = op;
      nf.a = 0;
      nf.b = 1;
      nf.xform = -1;
      Solid<double> nd2{};
      nd2.type = op;
      nd2.a = 0;
      nd2.b = 1;
      nd2.xform = -1;
      SolidStore<float> sf{};
      sf.solids = pf.data();
      sf.xforms = xff.data();
      SolidStore<double> sd{};
      sd.solids = pd.data();
      sd.xforms = xfd.data();
      for (int i = 0; i < n; ++i) {
        const double r = aim * std::sqrt((i + 0.5) / n);
        const double ang = 2.399963229 * i;
        const double tx = r * std::cos(ang), ty = r * std::sin(ang);
        const double L = std::sqrt(tx * tx + ty * ty + cam * cam);
        if (as_float) {
          const Vec3<float> o{0.f, 0.f, static_cast<float>(-cam)};
          const Vec3<float> d{static_cast<float>(tx / L), static_cast<float>(ty / L),
                              static_cast<float>(cam / L)};
          const float reach = bounding_radius(sf, nf);
          float skip = 0.f;
          if (reach > 0.f) {
            const float approach = -dot(o, d);
            skip = approach - reach * 1.01f - 1.f;
            if (skip < 0.f) { skip = 0.f; }
          }
          if (dist_in(sf, nf, o + skip * d, d) < kInfinity<float>()) { ++hit; }
        } else {
          const Vec3<double> o{0.0, 0.0, -cam};
          const Vec3<double> d{tx / L, ty / L, cam / L};
          const double reach = bounding_radius(sd, nd2);
          double skip = 0.0;
          if (reach > 0.0) {
            const double approach = -dot(o, d);
            skip = approach - reach * 1.01 - 1.0;
            if (skip < 0.0) { skip = 0.0; }
          }
          if (dist_in(sd, nd2, o + skip * d, d) < kInfinity<double>()) { ++hit; }
        }
      }
      return hit;
    };

    // THE PROPERTY IS FLATNESS, not a perfect score. A boolean's surface has creases -
    // where two orb surfaces meet, the two roots really are nearly equal - and a ray grazing
    // one is a coin flip at any precision, which costs about 2% on this deliberately grazing
    // aim. What the origin shift claims is that the hit rate STOPS DEPENDING ON WHERE THE
    // CAMERA IS, so that is what is asserted: the ratio at 12 m must match the ratio up close.
    //
    // Without the boolean bound the same rows read 99.7% at 200 mm, 28% at 1500 - the default
    // camera distance - and 0.26% at 12 m. That is the measurement the flatness test is
    // standing in for, and it was taken by deleting the bound.
    constexpr int kRays = 2000;
    double worst_ratio = 1.0;
    double worst_cam = 0;
    double near[3] = {0, 0, 0};      ///< each op's ratio up close
    double worst_drop = 0;           ///< and the largest fall from it, for that same op
    const char* drop_name = "";
    int too_few = 0;
    for (double cam : {200.0, 600.0, 1500.0, 4000.0, 12000.0}) {
      std::printf("    %7.0f mm ", cam);
      const SolidType ops[3] = {SolidType::kUnion, SolidType::kSubtraction,
                                SolidType::kIntersection};
      const char* names[3] = {"union", "subtract", "intersect"};
      for (int k = 0; k < 3; ++k) {
        // 39 mm: just inside each orb's own radius, so the rays graze the union's waist.
        const int nd = hits(ops[k], cam, 39.0, kRays, false);
        const int nf = hits(ops[k], cam, 39.0, kRays, true);
        if (nd < kRays / 4) {
          ++too_few;
        } else {
          const double ratio = static_cast<double>(nf) / nd;
          if (ratio < worst_ratio) {
            worst_ratio = ratio;
            worst_cam = cam;
          }
        }
        std::printf(" %s %d/%d", names[k], nf, nd);
        // PER OP, because the three have different silhouettes: an intersection of two orbs
        // offset by more than a radius is a thin lens, and comparing its worst row against the
        // union's best is a comparison of two shapes rather than of two distances.
        if (nd >= kRays / 4) {
          const double ratio = static_cast<double>(nf) / nd;
          if (cam < 300.0) {
            near[k] = ratio;
          } else if (near[k] > 0 && near[k] - ratio > worst_drop) {
            worst_drop = near[k] - ratio;
            drop_name = names[k];
          }
        }
      }
      std::printf("\n");
    }
    Check(too_few == 0, "each boolean is hit by enough rays in double to compare");
    std::printf("    worst: at %.0f mm kept %.4f; largest fall from the close-up rate is "
                "%.4f (%s)\n", worst_cam, worst_ratio, worst_drop, drop_name);
    Check(worst_ratio >= 0.97,
          "float keeps at least 97% of the rays double finds through a boolean - the rest is "
          "the crease where two surfaces meet");
    Check(worst_drop <= 0.01,
          "and no boolean's hit rate falls off with camera distance, which is the property the "
          "origin shift exists for");
  }

  // ---- 6. A NEARLY TRANSPARENT SURFACE IS MOSTLY THE BACKGROUND, NOT MOSTLY BLACK.
  //
  // Reported as "when opacity is set to a very low value it looks like the background is
  // black, rather than whatever the viewer background is".
  //
  // The geometry pass accumulates PREMULTIPLIED colour: each surface adds
  // `alpha * (1 - alpha_so_far) * colour`, so a volume at 3% opacity leaves 3% of its colour
  // in the framebuffer and nothing else. That was written straight out - a very dark pixel -
  // when what it means is 3% of the colour and 97% of what is behind it. The coverage now
  // rides in the framebuffer's spare alpha byte and vis::resolve_pixel finishes the job.
  //
  // Checked as an identity rather than by eye: a surface at coverage a over background B must
  // resolve to premultiplied + (1-a)B, which at a = 0 is B exactly and at a = 255 is the
  // surface exactly. The two ends are what the report was about and what a picture cannot be
  // asked precisely.
  {
    std::printf("\n  compositing a partly covering surface over the background:\n");
    const vis::Palette pal{};
    // A framebuffer word with a given low half. vis::pack_pixel is device-only - it uses
    // __float_as_uint - and resolve_pixel only compares the whole word against kEmptyPixel
    // and reads the low half, so any depth that is not all-ones will do.
    auto pix = [](unsigned int rgba) {
      return (0x40000000ull << 32) | static_cast<unsigned long long>(rgba);
    };
    const int py = 137, ph = 400;
    int br = 0, bg = 0, bb = 0;
    vis::background(py, ph, pal, br, bg, bb);
    std::printf("    the background there is (%d %d %d)\n", br, bg, bb);
    Check(br + bg + bb > 0, "the background is not black, or this test is about nothing");

    // An empty pixel is the background, which is the case that already worked.
    {
      int r = -1, g = -1, b = -1;
      vis::resolve_pixel(vis::kEmptyPixel, py, ph, pal, r, g, b);
      Check(r == br && g == bg && b == bb, "an empty pixel is exactly the background");
    }

    // A fully opaque surface is exactly itself, whatever is behind.
    {
      const unsigned long long v = pix(vis::pack_rgba(200, 100, 50, 255));
      int r = 0, g = 0, b = 0;
      vis::resolve_pixel(v, py, ph, pal, r, g, b);
      Check(r == 200 && g == 100 && b == 50, "an opaque surface is exactly its own colour");
    }

    // And the case that was wrong: 3% coverage of a bright colour. Premultiplied that is 3%
    // of 255 = 7, which written out is nearly black; resolved it has to be within a step of
    // the background.
    {
      const int a = 8;   // about 3%
      const int pr = 255 * a / 255, pg = 255 * a / 255, pb = 255 * a / 255;
      const unsigned long long v = pix(vis::pack_rgba(pr, pg, pb, a));
      int r = 0, g = 0, b = 0;
      vis::resolve_pixel(v, py, ph, pal, r, g, b);
      std::printf("    3%% white resolves to (%d %d %d); premultiplied alone it is (%d %d %d)"
                  "\n", r, g, b, pr, pg, pb);
      // Nearer the background than the premultiplied colour is, on every channel.
      const int d_bg = std::abs(r - br) + std::abs(g - bg) + std::abs(b - bb);
      const int d_raw = std::abs(pr - br) + std::abs(pg - bg) + std::abs(pb - bb);
      Check(d_bg < d_raw,
            "a barely covering surface resolves nearer the background than its premultiplied "
            "colour does");
      Check(r >= br && g >= bg && b >= bb,
            "and it lightens the background rather than darkening it");
    }

    // Zero coverage is the background exactly - the limit the report was standing at.
    {
      const unsigned long long v = pix(vis::pack_rgba(0, 0, 0, 0));
      int r = 0, g = 0, b = 0;
      vis::resolve_pixel(v, py, ph, pal, r, g, b);
      Check(r == br && g == bg && b == bb,
            "a surface covering none of the pixel leaves the background untouched");
    }

    // And every writer that is not the geometry pass is opaque, or a wireframe edge would
    // fade into the background too.
    Check((vis::pack_rgb(1, 2, 3) >> 24) == 255u,
          "pack_rgb, which the edge and trajectory passes use, is opaque");
  }

  // ------------------------------------------------------------------ the limb, against truth
  //
  // AN ORB HAS AN EXACT ORACLE, and this is the only test here that uses one: a ray hits a
  // sphere iff its perpendicular distance from the centre is under the radius. Everything above
  // compares float against double, which cannot see a ray that BOTH precisions lose - and that
  // is what the limb loss was.
  //
  // The mechanism is in geom::kProbe, and the reason it needs a test of its own is that it is
  // invisible to every other check here: a step along a near-tangent ray leaves the surface by
  // only step * h / R, so where the half-chord h is small the probe lands inside the surface
  // band and is_crossing_to reports no crossing. Rays four millimetres deep into a 40 mm sphere
  // were being lost, and the width of the band is (tol/probe)^2 * R / 2 - PROPORTIONAL TO R, so
  // zooming does not escape it and it grows in pixels as you zoom in.
  //
  // Asserted as a fraction of the radius, because that is the form the failure takes. It was
  // 5.3e-3 and it is 4.6e-5.
  {
    std::printf("\n== a sphere's limb, against the exact answer ==\n");
    Solid<float> sf{};
    sf.type = SolidType::kOrb;
    SolidStore<float> st{};
    double worst_frac = 0;
    double worst_R = 0;
    int total_false_pos = 0;
    for (double R : {2.0, 10.0, 40.0, 200.0, 1000.0}) {
      sf.p[0] = static_cast<float>(R);
      // Five camera distances, because the ORIGIN SHIFT is what makes this independent of them
      // and a regression there would show up here as a distance-dependent band.
      for (double D : {R * 1.05, R * 2, R * 5, R * 40, R * 300}) {
        double furthest = 0;
        for (int i = 0; i < 4000; ++i) {
          const double q = R * 1.02 * (i + 0.5) / 4000;
          if (q >= D) { continue; }
          const double a = std::asin(q / D);
          const Vec3<float> eye{0.0f, 0.0f, static_cast<float>(D)};
          const Vec3<float> dir{static_cast<float>(std::sin(a)), 0.0f,
                                static_cast<float>(-std::cos(a))};
          // Exactly the renderer's own ray setup, origin shift included.
          const float reach = bounding_radius(st, sf);
          float skip = -(eye.z * dir.z) - reach * 1.01f - 1.0f;
          if (skip < 0.0f) { skip = 0.0f; }
          const Vec3<float> from{eye.x + skip * dir.x, eye.y + skip * dir.y,
                                 eye.z + skip * dir.z};
          const bool hit = dist_in(st, sf, from, dir) < kInfinity<float>();
          const bool want = q < R;
          if (hit == want) { continue; }
          if (!want) { ++total_false_pos; }
          const double gap = std::fabs(R - q);
          if (gap > furthest) { furthest = gap; }
        }
        const double frac = furthest / R;
        if (frac > worst_frac) {
          worst_frac = frac;
          worst_R = R;
        }
      }
    }
    std::printf("    worst limb band %.2e of the radius (R = %.0f mm)\n", worst_frac, worst_R);
    Check(worst_frac < 5e-4,
          "float loses rays only within 5e-4 of a sphere's radius of its limb, at every "
          "camera distance");
    // A ray that misses and is reported as a hit would be a surface drawn outside the solid,
    // which is a different failure and has never happened - so it is worth pinning at zero
    // rather than folding into the band above.
    Check(total_false_pos == 0, "and never reports a hit for a ray that misses");
  }

  std::printf("\n%s (%d failures)\n", g_fails ? "FAILED" : "ALL PASS", g_fails);
  return g_fails ? 1 : 0;
}
