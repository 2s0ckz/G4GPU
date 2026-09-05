// Voxel-volume traversal: the DDA that walks cells and stops where the material changes.
//
// There is no Geant4 oracle for this - G4NestedPhantomParameterisation is a placement scheme,
// not a shape, so there is no G4VSolid to compare against. The checks are therefore against
// facts the geometry has to satisfy independently of any implementation:
//
//   * a slab phantom crossed along an axis reports slab boundaries at exactly the right
//     depths, and the sum of the segment lengths is the box thickness;
//   * a homogeneous grid is crossed in a single step, whatever the direction - the property
//     that makes a CT affordable, and the one a naive per-cell stepper would fail;
//   * the material at a point matches the cell it is in, for the whole grid;
//   * a randomised march never loses length: the segments always sum to the chord through the
//     box, which is the invariant that catches an off-by-one in the cell index or a missed
//     boundary.
#include <cmath>
#include <cstdio>
#include <vector>

#include "core/rng.cuh"
#include "data/materials.cuh"
#include "geometry/navigator.cuh"

using namespace g4gpu;
using namespace g4gpu::geom;
using real_t = double;

static int fails = 0;
static void check(bool ok, const char* what) {
  printf(ok ? "  ok:   %s\n" : "  FAIL: %s\n", what);
  if (!ok) { ++fails; }
}

/// A voxel volume with `n` cells per side, half extent `half`, and a material function.
template <typename F>
static void build(int n, real_t half, F material_of, std::vector<short>& cells,
                  Volume<real_t>& v) {
  cells.assign(static_cast<std::size_t>(n) * n * n, 0);
  for (int k = 0; k < n; ++k) {
    for (int j = 0; j < n; ++j) {
      for (int i = 0; i < n; ++i) {
        cells[static_cast<std::size_t>(i) + n * (j + static_cast<std::size_t>(n) * k)] =
            static_cast<short>(material_of(i, j, k));
      }
    }
  }
  v = Volume<real_t>{};
  v.solid.type = SolidType::kVoxelGrid;
  v.solid.p[0] = v.solid.p[1] = v.solid.p[2] = half;
  v.solid.p[3] = v.solid.p[4] = v.solid.p[5] = static_cast<real_t>(n);
  v.solid.a = 0;
  v.solid.xform = -1;
  v.xform = make_translation<real_t>({0, 0, 0});
  v.layer = 1;
  v.material = 0;
}

int main() {
  // ---------------------------------------------------------------- slab phantom
  //
  // 10 cells over 100 mm, so 10 mm cells. Material 1 for cells 0..3, material 2 for 4..6,
  // material 3 for 7..9 along z: boundaries at z = -10 and z = +20 in the grid's frame.
  printf("== slab phantom, boundaries along z ==\n");
  {
    const int n = 10;
    const real_t half = 50;
    std::vector<short> cells;
    Volume<real_t> vols[2];
    // The world has to contain the grid, and the grid outranks it.
    vols[0] = Volume<real_t>{};
    vols[0].solid.type = SolidType::kBox;
    vols[0].solid.p[0] = vols[0].solid.p[1] = vols[0].solid.p[2] = 200;
    vols[0].solid.xform = -1;
    vols[0].xform = make_translation<real_t>({0, 0, 0});
    vols[0].layer = 0;
    vols[0].material = 0;
    build(n, half, [](int, int, int k) { return (k < 4) ? 1 : ((k < 7) ? 2 : 3); }, cells,
          vols[1]);

    Geometry<real_t> g{};
    g.volumes = vols;
    g.n_volumes = 2;
    g.world = 0;
    g.voxels.material = cells.data();
    g.voxels.count = static_cast<int>(cells.size());

    check(material_at(g, 1, Vec3<real_t>{0, 0, -45}) == 1, "cell 0 is material 1");
    check(material_at(g, 1, Vec3<real_t>{0, 0, -15}) == 1, "cell 3 is material 1");
    check(material_at(g, 1, Vec3<real_t>{0, 0, -5}) == 2, "cell 4 is material 2");
    check(material_at(g, 1, Vec3<real_t>{0, 0, 15}) == 2, "cell 6 is material 2");
    check(material_at(g, 1, Vec3<real_t>{0, 0, 25}) == 3, "cell 7 is material 3");
    check(material_at(g, 1, Vec3<real_t>{0, 0, 45}) == 3, "cell 9 is material 3");

    // March along +z from inside the world and record where the material changes.
    Vec3<real_t> p{0, 0, -150};
    const Vec3<real_t> d{0, 0, 1};
    int vol = locate(g, p);
    std::vector<double> boundaries;
    double in_grid = 0;
    for (int step = 0; step < 20 && vol != kOutsideWorld; ++step) {
      int entered = kOutsideWorld;
      const real_t dist = step_to_boundary(g, vol, p, d, entered);
      if (vol == 1) {
        in_grid += dist;
        boundaries.push_back(p.z + dist);
      }
      p = p + (dist + kPushDistance<real_t>()) * d;
      vol = resolve_after_step(g, entered, p);
    }
    printf("  material changes at z =");
    for (double b : boundaries) { printf(" %.4f", b); }
    printf("\n  total path in the grid = %.6f mm (expect 100)\n", in_grid);

    check(boundaries.size() == 3, "three segments: two internal boundaries and the exit");
    if (boundaries.size() == 3) {
      check(std::fabs(boundaries[0] - (-10.0)) < 1e-6, "first boundary at z = -10");
      check(std::fabs(boundaries[1] - 20.0) < 1e-6, "second boundary at z = +20");
      check(std::fabs(boundaries[2] - 50.0) < 1e-6, "exit at z = +50");
    }
    check(std::fabs(in_grid - 100.0) < 1e-5, "the segments sum to the box thickness");
  }

  // ---------------------------------------------------------------- homogeneous grid
  //
  // The reason a voxel volume is affordable: identical cells cost one step, not one per cell.
  printf("== homogeneous grid is crossed in one step ==\n");
  {
    const int n = 64;
    std::vector<short> cells;
    Volume<real_t> vols[2];
    vols[0] = Volume<real_t>{};
    vols[0].solid.type = SolidType::kBox;
    vols[0].solid.p[0] = vols[0].solid.p[1] = vols[0].solid.p[2] = 400;
    vols[0].solid.xform = -1;
    vols[0].xform = make_translation<real_t>({0, 0, 0});
    vols[0].layer = 0;
    vols[0].material = 0;
    build(n, 100, [](int, int, int) { return 1; }, cells, vols[1]);

    Geometry<real_t> g{};
    g.volumes = vols;
    g.n_volumes = 2;
    g.world = 0;
    g.voxels.material = cells.data();
    g.voxels.count = static_cast<int>(cells.size());

    Philox<real_t> rng(31, 7);
    int worst_steps = 0;
    int trials = 0;
    for (int t = 0; t < 2000; ++t) {
      const real_t ct = 2 * rng.uniform() - 1;
      const real_t st = std::sqrt(1 - ct * ct);
      const real_t ph = units::twopi<real_t>() * rng.uniform();
      const Vec3<real_t> d{st * std::cos(ph), st * std::sin(ph), ct};
      // Start on the far side and aim through the centre, so the grid is definitely crossed.
      Vec3<real_t> p{-350 * d.x, -350 * d.y, -350 * d.z};
      int vol = locate(g, p);
      if (vol == kOutsideWorld) { continue; }
      ++trials;
      int steps_in_grid = 0;
      for (int k = 0; k < 40 && vol != kOutsideWorld; ++k) {
        int entered = kOutsideWorld;
        const real_t dist = step_to_boundary(g, vol, p, d, entered);
        if (vol == 1) { ++steps_in_grid; }
        p = p + (dist + kPushDistance<real_t>()) * d;
        vol = resolve_after_step(g, entered, p);
      }
      worst_steps = std::max(worst_steps, steps_in_grid);
    }
    printf("  %d rays, most steps inside a 64^3 uniform grid = %d\n", trials, worst_steps);
    check(trials > 1000, "enough rays crossed the grid");
    check(worst_steps == 1, "a uniform 64^3 grid costs exactly one step");
  }

  // ---------------------------------------------------------------- checkerboard
  //
  // The opposite extreme: every neighbour differs, so every cell boundary is a step. This is
  // what bounds the cost, and it also exercises the DDA on all three axes at once.
  printf("== checkerboard: length is conserved ==\n");
  {
    const int n = 16;
    const real_t half = 80;
    std::vector<short> cells;
    Volume<real_t> vols[2];
    vols[0] = Volume<real_t>{};
    vols[0].solid.type = SolidType::kBox;
    vols[0].solid.p[0] = vols[0].solid.p[1] = vols[0].solid.p[2] = 300;
    vols[0].solid.xform = -1;
    vols[0].xform = make_translation<real_t>({0, 0, 0});
    vols[0].layer = 0;
    vols[0].material = 0;
    build(n, half, [](int i, int j, int k) { return 1 + ((i + j + k) & 1); }, cells, vols[1]);

    Geometry<real_t> g{};
    g.volumes = vols;
    g.n_volumes = 2;
    g.world = 0;
    g.voxels.material = cells.data();
    g.voxels.count = static_cast<int>(cells.size());

    Philox<real_t> rng(97, 3);
    double worst_err = 0;
    int trials = 0;
    long long total_steps = 0;
    for (int t = 0; t < 500; ++t) {
      const real_t ct = 2 * rng.uniform() - 1;
      const real_t st = std::sqrt(1 - ct * ct);
      const real_t ph = units::twopi<real_t>() * rng.uniform();
      const Vec3<real_t> d{st * std::cos(ph), st * std::sin(ph), ct};
      // Aim at a random point inside the grid, from outside it.
      const Vec3<real_t> aim{(2 * rng.uniform() - 1) * half * real_t(0.8),
                             (2 * rng.uniform() - 1) * half * real_t(0.8),
                             (2 * rng.uniform() - 1) * half * real_t(0.8)};
      Vec3<real_t> p = aim - real_t(250) * d;
      int vol = locate(g, p);
      if (vol == kOutsideWorld) { continue; }

      // The exact chord through the box, from the box routine itself: this is the reference
      // the marched segments have to reproduce.
      const real_t t_in = box_dist_in(vols[1].solid.p, p, d);
      if (t_in >= kInfinity<real_t>()) { continue; }
      const Vec3<real_t> entry = p + t_in * d;
      const real_t chord = box_dist_out(vols[1].solid.p, entry, d);
      if (chord <= 0) { continue; }
      ++trials;

      double marched = 0;
      int steps = 0;
      for (int k = 0; k < 400 && vol != kOutsideWorld; ++k) {
        int entered = kOutsideWorld;
        const real_t dist = step_to_boundary(g, vol, p, d, entered);
        if (vol == 1) {
          marched += dist;
          ++steps;
        }
        p = p + (dist + kPushDistance<real_t>()) * d;
        vol = resolve_after_step(g, entered, p);
      }
      total_steps += steps;
      // Each step is nudged forward by kPushDistance, so the marched total is short by that
      // much per step. Accounting for it explicitly, rather than widening the tolerance until
      // it passes, is what keeps this test able to see a real lost segment.
      const double expected = chord - steps * kPushDistance<real_t>();
      worst_err = std::max(worst_err, std::fabs(marched - expected));
    }
    printf("  %d rays, %.1f steps per ray, worst length error %.3g mm\n", trials,
           static_cast<double>(total_steps) / std::max(1, trials), worst_err);
    check(trials > 300, "enough rays crossed the checkerboard");
    check(worst_err < 1e-4, "marched length matches the chord through the box");
  }

  // ---------------------------------------------------------------- material lookup
  printf("== material at a point matches its cell ==\n");
  {
    const int n = 12;
    const real_t half = 60;
    std::vector<short> cells;
    Volume<real_t> v;
    build(n, half, [](int i, int j, int k) { return 1 + (i * 7 + j * 3 + k) % 5; }, cells, v);
    Geometry<real_t> g{};
    g.volumes = &v;
    g.n_volumes = 1;
    g.world = 0;
    g.voxels.material = cells.data();
    g.voxels.count = static_cast<int>(cells.size());

    int bad = 0;
    const real_t cell = 2 * half / n;
    for (int k = 0; k < n; ++k) {
      for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
          // The centre of each cell.
          const Vec3<real_t> q{-half + cell * (i + real_t(0.5)),
                               -half + cell * (j + real_t(0.5)),
                               -half + cell * (k + real_t(0.5))};
          const int want = 1 + (i * 7 + j * 3 + k) % 5;
          if (material_at(g, 0, q) != want) { ++bad; }
        }
      }
    }
    printf("  %d cell centres checked, %d wrong\n", n * n * n, bad);
    check(bad == 0, "every cell centre reports its own material");
  }

  printf("\n%s (%d failures)\n", fails ? "FAILED" : "ALL PASS", fails);
  return fails;
}
