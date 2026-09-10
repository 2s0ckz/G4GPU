// What the renderer draws where volumes overlap, asked one ray at a time on the HOST.
//
// Every renderer bug in this project so far was found by taking a screenshot and looking at it.
// That worked, and it cost three wrong diagnoses on one occasion and a ten-minute GUI rebuild
// per attempt - because nothing below the kernel could be called from a test. vis::trace_pixel
// is __host__ __device__ now, so a ray can be traced against a hand-built scene in a
// millisecond and the answer inspected as a number.
//
// What this file is for is the OVERLAP RULES, which is where the picture and the transport can
// disagree: a higher layer takes the space from a lower one, a translucent volume has to let
// what is behind it through, and a voxel class set to null is not there at all. Each of those
// is one ray with one expected colour, and each of them has been wrong.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "geometry/bvh_build.hh"
#include "render/renderer.cuh"

using namespace g4gpu;
using real_t = float;

static int fails = 0;
static void check(bool ok, const char* what) {
  printf(ok ? "  ok:   %s\n" : "  FAIL: %s\n", what);
  if (!ok) { ++fails; }
}

// ---------------------------------------------------------------- a scene, built by hand

/// Everything trace_pixel needs, owned so the pointers stay alive.
struct Scene {
  std::vector<geom::Volume<real_t>> volumes;
  std::vector<vis::VolumeStyle> styles;
  std::vector<short> cells;        ///< material per cell, and the same offsets as `cls`
  std::vector<short> cls;          ///< class per cell
  std::vector<int> class_layer;    ///< one layer per class
  std::vector<unsigned char> absent;
  std::vector<unsigned int> class_rgba;
  bool any_absent = false;

  geom::Geometry<real_t> geometry() const {
    geom::Geometry<real_t> g{};
    g.volumes = volumes.data();
    g.n_volumes = static_cast<int>(volumes.size());
    g.world = 0;
    g.voxels.material = cells.empty() ? nullptr : cells.data();
    g.voxels.count = static_cast<int>(cells.size());
    g.voxels.cls = cls.empty() ? nullptr : cls.data();
    g.voxels.class_layer = class_layer.empty() ? nullptr : class_layer.data();
    g.voxels.class_absent = any_absent ? absent.data() : nullptr;
    return g;
  }
};

static geom::Transform<real_t> at(real_t x, real_t y, real_t z) {
  geom::Transform<real_t> t{};
  t.rot[0] = t.rot[4] = t.rot[8] = real_t(1);
  t.trans = Vec3<real_t>{x, y, z};
  return t;
}

/// Adds a box, and the style that draws it. @p alpha 255 is opaque.
static int add_box(Scene& s, real_t hx, real_t hy, real_t hz, real_t z, int layer, int r, int g,
                   int b, int alpha) {
  geom::Volume<real_t> v{};
  v.solid.type = geom::SolidType::kBox;
  v.solid.p[0] = hx;
  v.solid.p[1] = hy;
  v.solid.p[2] = hz;
  v.xform = at(0, 0, z);
  v.layer = layer;
  v.layer_lo = layer;
  v.layer_hi = layer;
  s.volumes.push_back(v);
  vis::VolumeStyle st{};
  st.solid = true;
  st.r = static_cast<unsigned char>(r);
  st.g = static_cast<unsigned char>(g);
  st.b = static_cast<unsigned char>(b);
  st.a = static_cast<unsigned char>(alpha);
  s.styles.push_back(st);
  return static_cast<int>(s.volumes.size()) - 1;
}

/// Adds an n^3 voxel grid of half extent @p half, centred on the origin.
///
/// @param class_of  cell k index along z -> class index. Two classes only, which is enough:
///        what these checks turn on is a class's LAYER and its alpha, not how many there are.
template <typename F>
static int add_grid(Scene& s, real_t half, int n, int layer, F class_of,
                    const int layers[2], const unsigned int rgba[2], const bool absent[2]) {
  geom::Volume<real_t> v{};
  v.solid.type = geom::SolidType::kVoxelGrid;
  v.solid.p[0] = v.solid.p[1] = v.solid.p[2] = half;
  v.solid.p[3] = v.solid.p[4] = v.solid.p[5] = static_cast<real_t>(n);
  v.solid.a = static_cast<int>(s.cells.size());
  v.solid.p[6] = static_cast<real_t>(s.class_layer.size());   // where this run starts
  v.solid.p[7] = real_t(2);                                   // and how long it is
  v.xform = at(0, 0, 0);
  v.layer = layer;

  const int base = static_cast<int>(s.class_layer.size());
  for (int c = 0; c < 2; ++c) {
    s.class_layer.push_back(layers[c]);
    s.class_rgba.push_back(rgba[c]);
    s.absent.push_back(absent[c] ? 1u : 0u);
    if (absent[c]) { s.any_absent = true; }
  }
  // has_class_layers / layer_lo / layer_hi are what the ownership scan prunes on, and getting
  // them wrong is a silent wrong answer rather than a crash - so they are derived here the
  // same way the flattener derives them.
  v.has_class_layers = (layers[0] != layer) || (layers[1] != layer);
  v.has_absent_classes = absent[0] || absent[1];
  v.layer_lo = layer;
  v.layer_hi = layer;
  for (int c = 0; c < 2; ++c) {
    if (absent[c]) { continue; }
    if (layers[c] < v.layer_lo) { v.layer_lo = layers[c]; }
    if (layers[c] > v.layer_hi) { v.layer_hi = layers[c]; }
  }

  for (int k = 0; k < n; ++k) {
    for (int j = 0; j < n; ++j) {
      for (int i = 0; i < n; ++i) {
        const int c = class_of(k);
        s.cls.push_back(static_cast<short>(c));
        s.cells.push_back(static_cast<short>(c));
      }
    }
  }
  (void)base;
  s.volumes.push_back(v);
  vis::VolumeStyle st{};
  st.solid = true;
  st.a = 255;
  s.styles.push_back(st);
  return static_cast<int>(s.volumes.size()) - 1;
}

/// The world: a box big enough to hold everything, drawn as a wireframe so it composites
/// nothing - which is what the builder and the viewer both do with it.
static void add_world(Scene& s, real_t half) {
  geom::Volume<real_t> v{};
  v.solid.type = geom::SolidType::kBox;
  v.solid.p[0] = v.solid.p[1] = v.solid.p[2] = half;
  v.xform = at(0, 0, 0);
  v.layer = 0;
  v.layer_lo = 0;
  v.layer_hi = 0;
  s.volumes.push_back(v);
  vis::VolumeStyle st{};
  st.solid = false;   // wireframe
  s.styles.push_back(st);
}

/// Traces the ray down the -z axis through the middle of the frame and resolves it to RGB.
static void trace_axis(const Scene& s, int& r, int& g, int& b, int* alpha = nullptr) {
  const vis::Camera cam = vis::make_camera(vis::Vec3f{0, 0, 400}, vis::Vec3f{0, 0, 0}, vis::Vec3f{0, 1, 0},
                                           40.0f, 64, 64);
  const unsigned long long px =
      vis::trace_pixel<real_t>(s.geometry(), s.styles.data(), cam, 32.0f, 32.0f, false,
                               s.cls.empty() ? nullptr : s.cls.data(),
                               s.class_rgba.empty() ? nullptr : s.class_rgba.data());
  if (alpha != nullptr) {
    *alpha = (px == vis::kEmptyPixel)
                 ? 0
                 : static_cast<int>((static_cast<unsigned int>(px & 0xFFFFFFFFull) >> 24)
                                    & 0xFFu);
  }
  // Palette with a BLACK background, so any colour in the answer came from the geometry.
  vis::Palette pal;
  pal.bg_top = 0;
  pal.bg_bottom = 0;
  vis::resolve_pixel(px, 32, 64, pal, r, g, b);
}

static const char* dominant(int r, int g, int b) {
  if (r < 8 && g < 8 && b < 8) { return "black (nothing drawn)"; }
  if (r > g && r > b) { return "red"; }
  if (g > r && g > b) { return "green"; }
  if (b > r && b > g) { return "blue"; }
  return "grey";
}

int main() {
  // ---------------------------------------------------------------- 1. through a translucent
  // higher layer, to the cells beyond it
  //
  // A phantom on layer 1 with a translucent box on layer 2 over it. Where the box is, the box
  // owns the space and the cells under it are correctly not drawn - but the box ENDS, and the
  // cells beyond its far face are the phantom's again. Reported as not being able to see the
  // phantom through the object at all.
  //
  // Along the axis: the grid is 8 cells of 10 mm from z = -40 to 40, the near half is air
  // (alpha 0) so nothing saturates before the box, the box spans z = 0..20, and the far cells
  // below z = -20 are opaque RED. So a correct trace is red seen through the blue box, and the
  // failure is blue over black.
  printf("\n== a translucent higher layer does not hide the cells behind it ==\n");
  {
    Scene s;
    add_world(s, 500);
    const int layers[2] = {1, 1};
    const unsigned int rgba[2] = {0x00000000u,                       // class 0: air, alpha 0
                                  vis::pack_rgba(230, 40, 40, 255)}; // class 1: opaque red
    const bool absent[2] = {false, false};
    add_grid(s, 40, 8, 1, [](int k) { return (k < 2) ? 1 : 0; }, layers, rgba, absent);
    add_box(s, 20, 20, 10, 10, 2, 40, 80, 230, 77);   // translucent blue, layer 2, z 0..20

    int r = 0, g = 0, b = 0, a = 0;
    trace_axis(s, r, g, b, &a);
    printf("  rgb (%3d %3d %3d) coverage %3d -> %s\n", r, g, b, a, dominant(r, g, b));
    check(r > 60, "the opaque cells beyond a translucent cover are drawn");
    check(b > 20, "and the cover itself is still drawn in front of them");
  }

  // ---------------------------------------------------------------- 2. a null class hands its
  // space to whatever is in it
  //
  // A phantom on layer 2 whose near half is set to the NULL layer, with a box on layer 1 inside
  // that half. Null is not a low layer, it is not there - so the box owns that space whatever
  // its own layer is, and it has to be drawn. Reported as not being able to see it.
  //
  // The box is opaque MAGENTA and sits at z = 12..28, inside the nulled cells; the phantom's
  // far half below z = 0 is opaque GREEN. So a correct trace is magenta, and the two ways to
  // fail are green (the box was skipped and the far cells drawn instead) and black.
  printf("\n== a volume inside a null voxel class is drawn there ==\n");
  {
    Scene s;
    add_world(s, 500);
    const int layers[2] = {2, 2};
    const unsigned int rgba[2] = {vis::pack_rgba(200, 200, 40, 255),   // class 0: nulled
                                  vis::pack_rgba(40, 200, 60, 255)};   // class 1: green
    const bool absent[2] = {true, false};
    add_grid(s, 40, 8, 2, [](int k) { return (k < 4) ? 1 : 0; }, layers, rgba, absent);
    add_box(s, 8, 8, 8, 20, 1, 230, 40, 200, 255);   // opaque magenta, layer 1, z 12..28

    int r = 0, g = 0, b = 0, a = 0;
    trace_axis(s, r, g, b, &a);
    printf("  rgb (%3d %3d %3d) coverage %3d -> %s\n", r, g, b, a, dominant(r, g, b));
    check(r > 60 && b > 40, "a volume in a nulled class's cells is drawn there");
    check(g < r, "and the cells beyond it are not what was drawn instead");
  }

  // ---------------------------------------------------------------- 3. a translucent volume
  // sitting IN the nulled cells is still translucent
  //
  // The two features together, which is where they interfere. A phantom whose air class is
  // nulled, with a translucent box on a higher layer inside that air - the arrangement anyone
  // gets by dropping a box into a CT - and the tissue behind it has to show through the box.
  //
  // What went wrong is the interaction and not either half. `inside_volume` is per CELL now, so
  // a ray standing in a nulled cell is NOT inside the volume - correct for ownership, and the
  // wrong question for "have I already marched this grid". The march's resume branch is gated
  // on it, so a ray that stopped inside the box could not find the grid again: `dist_in` from
  // inside the grid's own box returns zero and the search rejects that as a volume it is
  // already in. The box was composited over the background instead of over the phantom, which
  // with a dark background reads as the box having gone opaque.
  printf("\n== a translucent volume inside nulled cells shows what is behind it ==\n");
  {
    Scene s;
    add_world(s, 500);
    const int layers[2] = {1, 1};
    const unsigned int rgba[2] = {vis::pack_rgba(200, 200, 40, 255),   // class 0: nulled air
                                  vis::pack_rgba(230, 40, 40, 255)};   // class 1: opaque red
    const bool absent[2] = {true, false};
    // Air above z = 0, tissue below it.
    add_grid(s, 40, 8, 1, [](int k) { return (k < 4) ? 1 : 0; }, layers, rgba, absent);
    // Wholly inside the air, and on a HIGHER layer, so it owns that space outright.
    add_box(s, 20, 20, 10, 20, 2, 40, 80, 230, 77);   // translucent blue, z 10..30

    int r = 0, g = 0, b = 0, a = 0;
    trace_axis(s, r, g, b, &a);
    printf("  rgb (%3d %3d %3d) coverage %3d -> %s\n", r, g, b, a, dominant(r, g, b));
    check(r > 60, "the tissue beyond a translucent box in nulled air is drawn");
    check(b > 20, "and the box is still drawn in front of it");
    check(a > 250, "and the pixel ends up fully covered rather than showing background");
  }

  // ---------------------------------------------------------------- 4. AN ORDINARY VOLUME
  // beyond a translucent higher layer
  //
  // Everything above is about a voxel grid, which has a cell march to resume. An ordinary solid
  // has no march, and the search skips any volume the ray is already inside - the test that
  // stops a translucent box being re-entered and painted until it saturates. So a volume whose
  // space a higher layer takes for part of the ray was never drawn again beyond that: the
  // higher volume could be seen through, and what was behind it could not.
  //
  // Reported twice over: two translucent boxes where the far one cannot be seen through the
  // near one, and a phantom under a vest that appears only when the PHANTOM is on the higher
  // layer. That second one is the same statement with the layers named.
  //
  // Along the axis: translucent blue on layer 2 from z = 10 to 40, opaque red on layer 1 from
  // z = -50 to 30. The blue is IN FRONT and owns the overlap, so the red's own front face at
  // z = 30 is correctly not drawn - but the blue ends at z = 10, and the red beyond that is the
  // red's again.
  //
  // The red's blue channel is zero and the cover's is not, so neither colour can be mistaken
  // for the other: cover only reads as (12, 24, 69), red only as (r, g, 0).
  printf("\n== an ordinary volume is drawn beyond a translucent higher layer ==\n");
  {
    Scene s;
    add_world(s, 500);
    add_box(s, 30, 30, 40, -10, 1, 230, 60, 0, 255);   // opaque red, layer 1, z -50..30
    add_box(s, 20, 20, 15, 25, 2, 40, 80, 230, 77);    // translucent blue, layer 2, z 10..40

    int r = 0, g = 0, b = 0, a = 0;
    trace_axis(s, r, g, b, &a);
    printf("  rgb (%3d %3d %3d) coverage %3d -> %s\n", r, g, b, a, dominant(r, g, b));
    check(r > 60, "the volume beyond a translucent cover is drawn");
    check(b > 30, "and the cover is still drawn in front of it");
    check(a > 250, "and the pixel is covered rather than showing background");
  }

  // ---------------------------------------------------------------- 5. A MESH COVER
  //
  // The same claim as 4, with the cover a triangle mesh - which is the case actually reported,
  // twice, as a phantom under a vest. A mesh takes a different path through the search: one BVH
  // walk for the nearest hit and NO containment test, deliberately, because a containment test
  // on a mesh is a parity count and paying one per composited layer is what made a CAD import
  // unusable.
  //
  // No containment test also means the search cannot tell an ENTRY from an EXIT. From inside
  // the mesh the nearest hit ahead is its far wall, offered as though the ray were entering
  // there - and at the same distance as the covered volume's resume, which the mesh then wins
  // on rank because it is the higher layer. The covered volume never resumes, and the next
  // iteration finds it owning the point it is standing in, so it is skipped for good.
  //
  // Which is why a box cover works and a mesh cover does not, and why nothing in case 4 could
  // have caught this.
  printf("\n== and beyond a translucent MESH cover, which is the reported case ==\n");
  {
    Scene s;
    add_world(s, 500);
    add_box(s, 30, 30, 40, -10, 1, 230, 60, 0, 255);   // opaque red, layer 1, z -50..30

    // A closed 12-triangle box, the same shape as the cover in case 4, as a mesh.
    std::vector<real_t> tri;
    {
      const real_t h[3] = {20, 20, 15};
      const int f[6][4] = {{0, 1, 3, 2}, {4, 6, 7, 5}, {0, 4, 5, 1},
                           {2, 3, 7, 6}, {0, 2, 6, 4}, {1, 5, 7, 3}};
      real_t v[8][3];
      for (int i = 0; i < 8; ++i) {
        v[i][0] = ((i & 1) ? h[0] : -h[0]);
        v[i][1] = ((i & 2) ? h[1] : -h[1]);
        v[i][2] = ((i & 4) ? h[2] : -h[2]) + real_t(25);   // centred at z = 25
      }
      for (int q = 0; q < 6; ++q) {
        const int* c = f[q];
        const int t2[2][3] = {{c[0], c[1], c[2]}, {c[0], c[2], c[3]}};
        for (int k = 0; k < 2; ++k) {
          for (int e = 0; e < 3; ++e) {
            for (int a2 = 0; a2 < 3; ++a2) { tri.push_back(v[t2[k][e]][a2]); }
          }
        }
      }
    }
    std::vector<real_t> tri_pool, bvh_pool;
    real_t lo[3] = {0, 0, 0}, hi[3] = {0, 0, 0};
    const int n_tri = static_cast<int>(tri.size() / 9);
    const int root = geom::build_bvh(tri.data(), n_tri, tri_pool, bvh_pool, lo, hi);

    geom::Volume<real_t> m{};
    m.solid.type = geom::SolidType::kMesh;
    for (int k = 0; k < 3; ++k) {
      m.solid.p[k] = real_t(0.5) * (hi[k] - lo[k]);
      m.solid.p[3 + k] = real_t(0.5) * (hi[k] + lo[k]);
    }
    // p[6] the volume and p[7] the WINDING, both from the same sum, as the flattener does. The
    // winding is taken from geom::mesh_winding rather than hand-written: this fixture's
    // triangles turned out to be wound INWARD, and asserting a hand-picked sign would have
    // been asserting the fixture's mistake. Whichever way it is wound, the renderer has to
    // tell an entry from an exit - so the check below is worth more with it read off.
    m.solid.p[6] = geom::mesh_volume(tri.data(), n_tri);
    m.solid.p[7] = geom::mesh_winding(tri.data(), n_tri);
    m.solid.xform = -1;
    m.solid.a = root;
    m.solid.b = n_tri;
    m.xform = at(0, 0, 0);
    m.layer = 2;
    m.layer_lo = 2;
    m.layer_hi = 2;
    s.volumes.push_back(m);
    vis::VolumeStyle ms{};
    ms.solid = true;
    ms.r = 40;
    ms.g = 80;
    ms.b = 230;
    ms.a = 77;   // translucent, as the vest is
    s.styles.push_back(ms);

    // The mesh pools have to reach trace_pixel, and Scene::geometry does not carry them.
    geom::Geometry<real_t> g = s.geometry();
    g.store.tri = tri_pool.data();
    g.store.bvh = bvh_pool.data();

    const vis::Camera cam = vis::make_camera(vis::Vec3f{0, 0, 400}, vis::Vec3f{0, 0, 0},
                                             vis::Vec3f{0, 1, 0}, 40.0f, 64, 64);
    const unsigned long long px =
        vis::trace_pixel<real_t>(g, s.styles.data(), cam, 32.0f, 32.0f, false, nullptr,
                                 nullptr);
    vis::Palette pal;
    pal.bg_top = 0;
    pal.bg_bottom = 0;
    int r = 0, g2 = 0, b = 0;
    vis::resolve_pixel(px, 32, 64, pal, r, g2, b);
    const int a =
        (px == vis::kEmptyPixel)
            ? 0
            : static_cast<int>((static_cast<unsigned int>(px & 0xFFFFFFFFull) >> 24) & 0xFFu);
    printf("  rgb (%3d %3d %3d) coverage %3d -> %s   (%d triangles)\n", r, g2, b, a,
           dominant(r, g2, b), n_tri);
    check(r > 60, "the volume beyond a translucent MESH cover is drawn");
    check(b > 30, "and the mesh is still drawn in front of it");
    check(a > 250, "and the pixel is covered rather than showing background");
  }

  // ---------------------------------------------------------------- 6. the rules that already
  // held, so a fix to the ones above cannot quietly undo them
  printf("\n== and the overlap rules that were already right ==\n");
  {
    // A higher layer still hides a lower one where it covers it.
    Scene s;
    add_world(s, 500);
    const int layers[2] = {1, 1};
    const unsigned int rgba[2] = {vis::pack_rgba(230, 40, 40, 255),
                                  vis::pack_rgba(230, 40, 40, 255)};
    const bool absent[2] = {false, false};
    add_grid(s, 40, 8, 1, [](int) { return 0; }, layers, rgba, absent);
    add_box(s, 20, 20, 10, 60, 2, 40, 80, 230, 255);   // OPAQUE blue in front, layer 2

    int r = 0, g = 0, b = 0;
    trace_axis(s, r, g, b);
    printf("  opaque cover:  rgb (%3d %3d %3d) -> %s\n", r, g, b, dominant(r, g, b));
    check(b > r, "an opaque higher layer hides the phantom behind it");
  }
  {
    // A volume on a LOWER layer inside a phantom whose cells are present stays hidden: the
    // cells are there, so they own the space. This is the other half of check 2, and without
    // it "drawn in the hole" is satisfied by a renderer that has stopped honouring layers.
    Scene s;
    add_world(s, 500);
    const int layers[2] = {2, 2};
    const unsigned int rgba[2] = {vis::pack_rgba(200, 200, 40, 255),
                                  vis::pack_rgba(40, 200, 60, 255)};
    const bool absent[2] = {false, false};   // nothing nulled this time
    add_grid(s, 40, 8, 2, [](int k) { return (k < 4) ? 1 : 0; }, layers, rgba, absent);
    add_box(s, 8, 8, 8, 20, 1, 230, 40, 200, 255);

    int r = 0, g = 0, b = 0;
    trace_axis(s, r, g, b);
    printf("  class present: rgb (%3d %3d %3d) -> %s\n", r, g, b, dominant(r, g, b));
    check(b < 60, "a lower-layer volume inside a phantom whose cells are present is hidden");
  }

  printf(fails == 0 ? "\nALL OK\n" : "\n%d FAILED\n", fails);
  return fails == 0 ? 0 : 1;
}
