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
//     boundary;
//   * and the grid answers the two questions every solid is asked about its SIZE - how far it
//     reaches and how much space it fills - the same way the box of the same dimensions does.
//     It used to answer zero to both, being absent from the switch in each function, and that
//     is the kind of wrong that no traversal test can see. See section 5.
#include <cmath>
#include <cstdio>
#include <vector>

#include "core/rng.cuh"
#include "data/materials.cuh"
#include "geometry/navigator.cuh"
#include "geometry/volume_of.cuh"

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

  // ---------------------------------------------------------------- how big is it
  //
  // A GRID IS A BOX, AND HAS TO ANSWER LIKE ONE.
  //
  // `solid_half_extent` and `analytic_volume` are switches over solid type, and kVoxelGrid was
  // in neither. It therefore fell through to the default in each: a coordinate bound of ZERO,
  // and no closed form, which sends solid_volume off to sample a cube of side zero and get
  // nothing. Both are load-bearing, and none of the traversal above touches either:
  //
  //   * a bound of zero collapses the bounding sphere the same-layer overlap check compares
  //     centre separations against, so a phantom only registered as overlapping something
  //     when their centres nearly coincided. That is what "the overlap check only fires if
  //     there is substantial overlap" was;
  //   * a volume of zero is a MASS of zero for anything scored on a grid as a whole, and a
  //     dose is energy over mass;
  //   * and the viewer sizes the camera from the same bound, so a phantom framed the view
  //     around every object except itself.
  //
  // Checked against the box of the same half extent, because that is what a grid is: the same
  // shape, with a material per cell inside it. Deliberately anisotropic, so an answer that
  // took one axis for all three would show.
  {
    printf("\n== a grid is sized like the box it is ==\n");
    SolidStore<real_t> st{};
    Solid<real_t> grid{};
    grid.type = SolidType::kVoxelGrid;
    grid.p[0] = 150; grid.p[1] = 100; grid.p[2] = 850;   // a tall phantom, mm
    grid.p[3] = 60;  grid.p[4] = 40;  grid.p[5] = 340;   // cells
    grid.a = 0;
    Solid<real_t> box{};
    box.type = SolidType::kBox;
    box.p[0] = 150; box.p[1] = 100; box.p[2] = 850;

    const real_t he_grid = solid_half_extent(st, grid);
    const real_t he_box = solid_half_extent(st, box);
    printf("  half extent: grid %g, box %g (expected 850)\n", he_grid, he_box);
    check(he_grid == he_box, "the grid's coordinate bound is the box's");
    check(he_grid == real_t(850), "and it is the largest half extent");

    const real_t v_grid = solid_volume(st, grid);
    const real_t v_box = solid_volume(st, box);
    const real_t want = real_t(8) * 150 * 100 * 850;
    printf("  volume: grid %g, box %g (expected %g)\n", v_grid, v_box, want);
    check(v_grid == want, "the grid's volume is its half extents times eight");
    check(v_grid == v_box, "which is the box's volume");

    // The bound has to BOUND: no point the grid contains may lie outside it. Same contract
    // tests/test_solids.cu checks for the other thirty solids, which this type is not in.
    int outside = 0;
    Philox<real_t> rng(0xB0Bu, 0u);
    for (int i = 0; i < 20000; ++i) {
      const Vec3<real_t> q{real_t(1.2) * he_grid * (2 * rng.uniform() - 1),
                           real_t(1.2) * he_grid * (2 * rng.uniform() - 1),
                           real_t(1.2) * he_grid * (2 * rng.uniform() - 1)};
      if (!inside(st, grid, q)) { continue; }
      if (fabs(q.x) > he_grid || fabs(q.y) > he_grid || fabs(q.z) > he_grid) { ++outside; }
    }
    check(outside == 0, "no point inside the grid is outside its coordinate bound");
  }

  // ---------------------------------------------------------------- per-class layers
  //
  // A CLASS OF A PHANTOM CAN OWN THE SPACE ITS VOLUME LOSES.
  //
  // The layer rule gives shared space to the higher layer, and a voxel volume used to have
  // one layer for all of it. So a phantom overlapping a seat, a couch or an implant was
  // either always on top - its air winning over solid metal - or always underneath, with its
  // bone losing to the same. Per-class layers are the third answer, and the whole of the
  // question is per POINT: the same volume outranks its neighbour in one cell and not in the
  // next.
  //
  // Three things have to hold, and they are checked separately because they fail separately.
  {
    printf("\n== a voxel class can be on its own layer ==\n");
    // Sixteen cells along z, alternating class 0 and class 1. Both classes are given the SAME
    // material, which is the case that catches a step that ends only where the material
    // changes: the layer changes at every boundary here and the material never does.
    constexpr int n = 16;
    const real_t half = 80;
    const real_t cell_h = 2 * half / n;
    std::vector<short> cells(static_cast<std::size_t>(n), static_cast<short>(7));
    std::vector<short> cls(static_cast<std::size_t>(n), 0);
    for (int k = 0; k < n; ++k) { cls[static_cast<std::size_t>(k)] = (k % 2) ? 1 : 0; }
    // Class 0 inherits the volume's layer (1); class 1 is on 5. A box on layer 3 covers the
    // whole grid, so it outranks class 0 and loses to class 1.
    const int class_layers[2] = {1, 5};

    std::vector<Volume<real_t>> vols(3);
    vols[0].solid = {SolidType::kBox, {400, 400, 400}};
    vols[0].xform = make_translation<real_t>({0, 0, 0});
    vols[0].layer = 0;
    vols[0].material = 0;
    vols[1].solid = {SolidType::kVoxelGrid, {half, half, half, 1, 1, n}};
    vols[1].solid.a = 0;
    vols[1].solid.p[6] = 0;   // the class run starts at 0 and is two long
    vols[1].solid.p[7] = 2;
    vols[1].xform = make_translation<real_t>({0, 0, 0});
    vols[1].layer = 1;
    vols[1].material = 7;
    vols[1].has_class_layers = true;
    vols[1].layer_lo = 1;
    vols[1].layer_hi = 5;
    vols[2].solid = {SolidType::kBox, {half, half, half}};
    vols[2].xform = make_translation<real_t>({0, 0, 0});
    vols[2].layer = 3;
    vols[2].material = 9;

    Geometry<real_t> g{};
    g.volumes = vols.data();
    g.n_volumes = 3;
    g.world = 0;
    g.voxels.material = cells.data();
    g.voxels.count = n;
    g.voxels.cls = cls.data();
    g.voxels.class_layer = class_layers;

    // 1. WHO OWNS EACH CELL. The property, stated directly, at every cell centre.
    int wrong_owner = 0, wrong_mat = 0;
    for (int k = 0; k < n; ++k) {
      const real_t z = -half + (k + real_t(0.5)) * cell_h;
      const Vec3<real_t> p{0, 0, z};
      const int want = (k % 2) ? 1 : 2;    // class 1 keeps it, class 0 loses to the box
      if (locate(g, p) != want) { ++wrong_owner; }
      const int want_mat = (k % 2) ? 7 : 9;
      if (material_at(g, locate(g, p), p) != want_mat) { ++wrong_mat; }
    }
    printf("  %d cells: %d wrong owner, %d wrong material\n", n, wrong_owner, wrong_mat);
    check(wrong_owner == 0, "each cell is owned by the higher of its class and the box");
    check(wrong_mat == 0, "and reports that owner's material");

    // 2. A STEP ENDS WHERE THE LAYER CHANGES, even though the material does not.
    //
    // Without it, the walk crosses all sixteen cells in one step - the material is 7
    // throughout - and the box covering half of them is never seen.
    {
      const VoxelGrid<real_t> grid = voxel_grid_of(vols[1].solid);
      int changed_to = -1;
      const real_t t = voxel_step(g.voxels, grid, {0, 0, -half + real_t(0.001)}, {0, 0, 1},
                                  changed_to, false);
      printf("  voxel_step from the near face: %g mm (one cell is %g)\n",
             static_cast<double>(t), static_cast<double>(cell_h));
      check(fabs(t - (cell_h - real_t(0.001))) < real_t(1e-6),
            "a step inside the grid ends at the first cell boundary, layer not material");
    }

    // 3. WHERE A GRID STARTS OUTRANKING SOMETHING, which is what the box's own steps need.
    //
    // From inside the box at the near face, the first cell that beats layer 3 is cell 1, one
    // cell in. Asked for a rank the whole grid beats, it is zero; for one nothing beats,
    // never.
    {
      const VoxelGrid<real_t> grid = voxel_grid_of(vols[1].solid);
      const Vec3<real_t> q{0, 0, -half + real_t(0.001)};
      const Vec3<real_t> d{0, 0, 1};
      const long long box_rank = volume_rank(3, 2);
      const real_t t = voxel_first_outranking(g.voxels, grid, q, d, 1, vols[1].layer,
                                              box_rank, kInfinity<real_t>());
      printf("  first cell outranking the box: %g mm\n", static_cast<double>(t));
      check(fabs(t - (cell_h - real_t(0.001))) < real_t(1e-6),
            "a grid starts outranking a box at the first cell of the winning class");
      const real_t t_none =
          voxel_first_outranking(g.voxels, grid, q, d, 1, vols[1].layer,
                                 volume_rank(99, 0), kInfinity<real_t>());
      check(t_none >= kInfinity<real_t>(), "and never, against a layer no class beats");
      const real_t t_all =
          voxel_first_outranking(g.voxels, grid, q, d, 1, vols[1].layer,
                                 volume_rank(-5, 0), kInfinity<real_t>());
      check(t_all == real_t(0), "and at once, against a layer every class beats");
      // The limit is what keeps this from walking a phantom's whole diagonal for a step that
      // was only going a millimetre.
      const real_t t_lim = voxel_first_outranking(g.voxels, grid, q, d, 1, vols[1].layer,
                                                  box_rank, cell_h * real_t(0.5));
      check(t_lim >= kInfinity<real_t>(), "and gives up at the limit it was given");
    }

    // 4. AND NONE OF IT HAPPENS WITHOUT THE ARRAYS. The same geometry with the class arrays
    // withheld is the old behaviour exactly: one layer for the whole grid, so the box on 3
    // outranks all of it.
    {
      Geometry<real_t> g2 = g;
      g2.voxels.cls = nullptr;
      g2.voxels.class_layer = nullptr;
      std::vector<Volume<real_t>> v2(vols);
      v2[1].has_class_layers = false;
      g2.volumes = v2.data();
      int owned_by_grid = 0;
      for (int k = 0; k < n; ++k) {
        const real_t z = -half + (k + real_t(0.5)) * cell_h;
        if (locate(g2, {0, 0, z}) == 1) { ++owned_by_grid; }
      }
      check(owned_by_grid == 0, "with no per-class layers the box owns every cell, as before");
    }

    // 5. THE RENDERER ASKS THE SAME QUESTION A CHEAPER WAY, AND MUST GET THE SAME ANSWER.
    //
    // owns_contained_point exists so the renderer does not call locate once per composited
    // surface - locate would re-test the volume's own containment, which for a mesh is the
    // parity count the nearest-hit walk exists to avoid. The saving is only allowed if the two
    // agree, so that is asserted directly rather than argued: over a lattice, for EVERY volume
    // containing each point (which is the helper's precondition), owning the point and being
    // what locate returns are the same thing.
    //
    // This scene is the one that can tell them apart. A per-volume rank would put the grid at
    // layer 1 everywhere, so the box on 3 would appear to cover the class-1 cells that in fact
    // outrank it - the helper has to ask at the point, and asserting over the lattice is what
    // says it does.
    {
      int checked = 0, disagree = 0;
      unsigned seed = 20260908u;
      auto jitter = [&]() {
        seed = seed * 1664525u + 1013904223u;
        return real_t(seed >> 8) / real_t(1 << 24) - real_t(0.5);
      };
      for (int ix = -6; ix <= 6; ++ix) {
        for (int iy = -6; iy <= 6; ++iy) {
          for (int iz = -12; iz <= 12; ++iz) {
            const Vec3<real_t> p{ix * real_t(20) + jitter() * real_t(3),
                                 iy * real_t(20) + jitter() * real_t(3),
                                 iz * real_t(10) + jitter() * real_t(3)};
            const int owner = locate(g, p);
            for (int v = 0; v < g.n_volumes; ++v) {
              if (!inside_volume(g, v, p)) { continue; }   // the precondition
              ++checked;
              if (owns_contained_point(g, v, p) != (owner == v)) { ++disagree; }
            }
          }
        }
      }
      printf("  %d (point, containing volume) pairs: %d disagree with locate\n", checked,
             disagree);
      check(checked > 3000, "the lattice reaches enough of the scene to be worth asserting");
      check(disagree == 0, "owning a contained point and being what locate returns agree");
    }

    // AND OUTSIDE THE WORLD NOTHING OWNS ANYTHING, which is locate's first answer too. A
    // volume poking out through the world face still has surfaces out there, and without the
    // guard the renderer would draw them.
    {
      std::vector<Volume<real_t>> v3(vols);
      v3[0].solid = {SolidType::kBox, {half, half, half}};        // world shrunk to the grid
      v3[2].solid = {SolidType::kBox, {half, half, half}};        // and the box moved outside
      v3[2].xform = make_translation<real_t>({0, 0, 2 * half});
      Geometry<real_t> g3 = g;
      g3.volumes = v3.data();
      const Vec3<real_t> out{0, 0, half + 10};
      check(inside_volume(g3, 2, out), "the probe point is inside the box");
      check(locate(g3, out) == kOutsideWorld, "and outside the world, so locate owns nothing");
      check(!owns_contained_point(g3, 2, out), "so the box does not own it either");
    }
  }

  printf("\n%s (%d failures)\n", fails ? "FAILED" : "ALL PASS", fails);
  return fails;
}
