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
#include <vector>

#include "geometry/bvh_build.hh"
#include "geometry/mesh.cuh"
#include "render/float_geometry.cuh"

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

  std::printf("\n%s (%d failures)\n", g_fails ? "FAILED" : "ALL PASS", g_fails);
  return g_fails ? 1 : 0;
}
