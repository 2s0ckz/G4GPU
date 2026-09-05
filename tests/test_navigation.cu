// Builds the full B1 volume set and material table, then checks locate() and
// step_to_boundary() by ray-marching a primary along +z the way B1 fires it.
//
// Also exercises what the layer model adds over a mother/daughter tree: deliberate overlap
// resolved by layer, ties resolved by definition order, a solid straddling what would have
// been a mother boundary, and rotated placements.
#include <cstdio>
#include <cmath>
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

enum VolId : int { kWorld = 0, kEnvelope = 1, kShape1 = 2, kShape2 = 3, kNumVols = 4 };
static const char* vol_name(int v) {
  switch (v) {
    case kWorld: return "World";
    case kEnvelope: return "Envelope";
    case kShape1: return "Shape1";
    case kShape2: return "Shape2";
  }
  return "OUTSIDE";
}

/// B1 geometry in mm. Envelope 20x20x30 cm; world 1.2x that.
/// Three layers: world 0, envelope 1, the two shapes 2.
static void build_b1(Volume<real_t>* v) {
  v[kWorld] = {{SolidType::kBox, {120.0, 120.0, 180.0}},
               make_translation<real_t>({0, 0, 0}), 0, data::kAir};
  v[kEnvelope] = {{SolidType::kBox, {100.0, 100.0, 150.0}},
                  make_translation<real_t>({0, 0, 0}), 1, data::kWater};
  v[kShape1] = {{SolidType::kCons, {20.0, 40.0, 30.0}},
                make_translation<real_t>({0.0, 20.0, -70.0}), 2, data::kA150Tissue};
  v[kShape2] = {{SolidType::kTrd, {60.0, 60.0, 50.0, 80.0, 30.0}},
                make_translation<real_t>({0.0, -10.0, 70.0}), 2, data::kBoneCompact};
}

/// Marches a ray and returns the number of crossings, counting mismatches between what
/// step_to_boundary predicted and what locate() finds just past the boundary.
static int march(const Geometry<real_t>& g, Vec3<real_t> p, const Vec3<real_t>& d, int max_steps,
                 int& mismatches, bool verbose) {
  int vol = locate(g, p);
  int crossings = 0;
  while (vol != kOutsideWorld && crossings < max_steps) {
    int entered = kOutsideWorld;
    const real_t step = step_to_boundary(g, vol, p, d, entered);
    if (step >= kInfinity<real_t>() || step < 0) { ++mismatches; break; }
    p = p + (step + kPushDistance<real_t>()) * d;
    const int predicted = resolve_after_step(g, entered, p);
    const int actual = locate(g, p);
    if (verbose) {
      printf("  %-8s -> %-8s  step %8.3f mm  (z %8.3f)\n", vol_name(vol), vol_name(actual),
             step, p.z);
    }
    if (predicted != actual) {
      if (verbose) {
        printf("    MISMATCH: predicted %s, locate says %s\n", vol_name(predicted),
               vol_name(actual));
      }
      ++mismatches;
    }
    vol = actual;
    ++crossings;
  }
  return crossings;
}

int main() {
  Volume<real_t> vols[kNumVols];
  build_b1(vols);
  Geometry<real_t> g{vols, kNumVols, kWorld};

  data::Material<real_t> mats[data::kNumMaterials];
  data::build_b1_materials<real_t>(mats);

  printf("== material table ==\n");
  for (int i = 0; i < data::kNumMaterials; ++i) {
    printf("  mat %d: rho=%.6g g/cm3, %d elements, n_e=%.4e e/mm3\n",
           i, mats[i].density, mats[i].n_elements, mats[i].electron_density);
  }
  // H2O: 10 electrons per molecule, 3.3456e19 molecules/mm^3 -> 3.3456e20 e/mm^3
  check(std::fabs(mats[data::kWater].electron_density / 3.3456e20 - 1.0) < 2e-3,
        "water electron density ~3.35e20 e/mm^3");
  // Bone is 1.85x denser than water with similar Z/A, so n_e should be ~1.8x
  check(mats[data::kBoneCompact].electron_density > 1.7 * mats[data::kWater].electron_density,
        "bone electron density exceeds water by ~1.8x");

  printf("== locate() ==\n");
  check(locate(g, Vec3<real_t>{0, 0, 0}) == kEnvelope, "origin is in Envelope");
  check(locate(g, Vec3<real_t>{0, -10, 70}) == kShape2, "Shape2 centre resolves to Shape2");
  check(locate(g, Vec3<real_t>{0, 20, -70}) == kShape1, "Shape1 centre resolves to Shape1");
  check(locate(g, Vec3<real_t>{0, 0, 170}) == kWorld, "point past Envelope is in World");
  check(locate(g, Vec3<real_t>{0, 0, 200}) == kOutsideWorld, "point past World is outside");
  check(locate(g, Vec3<real_t>{0, 0, -150}) == kEnvelope, "B1 gun z sits on the Envelope face");

  printf("== ray march along +z through Shape2 (x=0, y=-10) ==\n");
  {
    Vec3<real_t> p{0.0, -10.0, -150.0};
    const Vec3<real_t> d{0, 0, 1};
    int vol = locate(g, p);
    real_t traversed_bone = 0;
    int crossings = 0, mismatches = 0;
    while (vol != kOutsideWorld && crossings < 20) {
      int entered = kOutsideWorld;
      const real_t step = step_to_boundary(g, vol, p, d, entered);
      if (vol == kShape2) { traversed_bone += step; }
      p = p + (step + kPushDistance<real_t>()) * d;
      const int predicted = resolve_after_step(g, entered, p);
      const int actual = locate(g, p);
      printf("  %-8s -> %-8s  step %8.3f mm  (z %8.3f)\n", vol_name(vol), vol_name(actual),
             step, p.z);
      if (predicted != actual) { ++mismatches; }
      vol = actual;
      ++crossings;
    }
    // Shape2 spans z in [40,100] mm, so a central axial ray sees 60 mm of bone.
    printf("  bone traversed = %.4f mm (expect 60.000)\n", traversed_bone);
    check(std::fabs(traversed_bone - 60.0) < 1e-3, "axial ray crosses 60 mm of bone");
    check(crossings < 20, "ray march terminated by leaving the world");
    check(mismatches == 0, "prediction matches locate on every crossing");
  }

  printf("== randomized locate/step consistency ==\n");
  {
    Philox<real_t> rng(11, 3);
    int steps = 0, mismatches = 0;
    for (int trial = 0; trial < 20000; ++trial) {
      Vec3<real_t> p{(2 * rng.uniform() - 1) * 110, (2 * rng.uniform() - 1) * 110,
                     (2 * rng.uniform() - 1) * 170};
      if (locate(g, p) == kOutsideWorld) { continue; }
      const real_t ct = 2 * rng.uniform() - 1;
      const real_t st = std::sqrt(1 - ct * ct);
      const real_t ph = units::twopi<real_t>() * rng.uniform();
      const Vec3<real_t> d{st * std::cos(ph), st * std::sin(ph), ct};
      steps += march(g, p, d, 8, mismatches, false);
    }
    printf("  %d boundary crossings, %d mismatches\n", steps, mismatches);
    check(steps > 20000, "enough crossings to be meaningful");
    check(mismatches == 0, "step_to_boundary and locate agree on every crossing");
  }

  // ------------------------------------------------------------ layer model

  printf("== layer priority ==\n");
  {
    // Two boxes on top of each other at the origin. The higher layer must win regardless of
    // which was defined first, which is the whole point of the model - a mother/daughter tree
    // could not express this at all, because neither box contains the other.
    enum : int { kW = 0, kLow = 1, kHigh = 2, kN = 3 };
    Volume<real_t> v[kN];
    v[kW] = {{SolidType::kBox, {200.0, 200.0, 200.0}}, make_translation<real_t>({0, 0, 0}), 0,
             data::kAir};
    // Defined earlier but on the higher layer.
    v[kHigh] = {{SolidType::kBox, {50.0, 50.0, 50.0}}, make_translation<real_t>({0, 0, 0}), 5,
                data::kBoneCompact};
    v[kLow] = {{SolidType::kBox, {80.0, 80.0, 80.0}}, make_translation<real_t>({0, 0, 0}), 2,
               data::kWater};
    Geometry<real_t> gg{v, kN, kW};

    check(locate(gg, Vec3<real_t>{0, 0, 0}) == kHigh, "overlap resolves to the higher layer");
    check(locate(gg, Vec3<real_t>{0, 0, 65}) == kLow, "outside the high layer, the low one wins");
    check(locate(gg, Vec3<real_t>{0, 0, 120}) == kW, "outside both, the world wins");

    // Marching out along +z must see 50 -> 80 -> 200, i.e. bone then water then air.
    Vec3<real_t> p{0, 0, 0};
    const Vec3<real_t> d{0, 0, 1};
    int vol = locate(gg, p);
    int seq[4] = {-1, -1, -1, -1};
    int n = 0;
    while (vol != kOutsideWorld && n < 4) {
      seq[n++] = vol;
      int entered = kOutsideWorld;
      const real_t step = step_to_boundary(gg, vol, p, d, entered);
      p = p + (step + kPushDistance<real_t>()) * d;
      vol = resolve_after_step(gg, entered, p);
    }
    check(seq[0] == kHigh && seq[1] == kLow && seq[2] == kW,
          "stepping out crosses high -> low -> world in order");
  }

  printf("== tie-break by definition order ==\n");
  {
    enum : int { kW = 0, kA = 1, kB = 2, kN = 3 };
    Volume<real_t> v[kN];
    v[kW] = {{SolidType::kBox, {200.0, 200.0, 200.0}}, make_translation<real_t>({0, 0, 0}), 0,
             data::kAir};
    v[kA] = {{SolidType::kBox, {50.0, 50.0, 50.0}}, make_translation<real_t>({0, 0, 0}), 3,
             data::kWater};
    v[kB] = {{SolidType::kBox, {50.0, 50.0, 50.0}}, make_translation<real_t>({0, 0, 0}), 3,
             data::kBoneCompact};
    Geometry<real_t> gg{v, kN, kW};
    check(locate(gg, Vec3<real_t>{0, 0, 0}) == kB, "equal layers: the later definition wins");
  }

  printf("== solid straddling the world boundary is clipped ==\n");
  {
    enum : int { kW = 0, kStick = 1, kN = 2 };
    Volume<real_t> v[kN];
    v[kW] = {{SolidType::kBox, {100.0, 100.0, 100.0}}, make_translation<real_t>({0, 0, 0}), 0,
             data::kAir};
    // Half in, half out: centred on the +z world face.
    v[kStick] = {{SolidType::kBox, {10.0, 10.0, 50.0}},
                 make_translation<real_t>({0, 0, 100.0}), 4, data::kBoneCompact};
    Geometry<real_t> gg{v, kN, kW};
    check(locate(gg, Vec3<real_t>{0, 0, 80}) == kStick, "the part inside the world is present");
    check(locate(gg, Vec3<real_t>{0, 0, 120}) == kOutsideWorld,
          "the part outside the world is cut off");

    // A track fired along +z from inside must leave the world, not get stuck in the stub.
    Vec3<real_t> p{0, 0, -90};
    const Vec3<real_t> d{0, 0, 1};
    int mismatches = 0;
    const int n = march(gg, p, d, 12, mismatches, false);
    check(n < 12, "a track through the straddling solid still exits the world");
    check(mismatches == 0, "no prediction mismatch around the clipped solid");
  }

  printf("== rotated placement ==\n");
  {
    // A tall thin box laid on its side by a 90 degree rotation about y: its local +z half
    // length of 90 mm must end up along global x.
    enum : int { kW = 0, kBar = 1, kN = 2 };
    Volume<real_t> v[kN];
    v[kW] = {{SolidType::kBox, {200.0, 200.0, 200.0}}, make_translation<real_t>({0, 0, 0}), 0,
             data::kAir};
    const real_t half_pi = units::pi<real_t>() / real_t(2);
    v[kBar] = {{SolidType::kBox, {10.0, 10.0, 90.0}},
               make_placement<real_t>({0, 0, 0}, 0, half_pi, 0), 3, data::kWater};
    Geometry<real_t> gg{v, kN, kW};

    check(locate(gg, Vec3<real_t>{80, 0, 0}) == kBar, "rotated bar extends along +x");
    check(locate(gg, Vec3<real_t>{-80, 0, 0}) == kBar, "rotated bar extends along -x");
    check(locate(gg, Vec3<real_t>{0, 0, 80}) == kW, "rotated bar no longer extends along z");
    check(locate(gg, Vec3<real_t>{0, 5, 0}) == kBar, "rotated bar still 10 mm half-width in y");

    // Crossing it along x must measure 180 mm, the rotated long axis.
    Vec3<real_t> p{-150, 0, 0};
    const Vec3<real_t> d{1, 0, 0};
    int vol = locate(gg, p);
    real_t in_bar = 0;
    for (int k = 0; k < 8 && vol != kOutsideWorld; ++k) {
      int entered = kOutsideWorld;
      const real_t step = step_to_boundary(gg, vol, p, d, entered);
      if (vol == kBar) { in_bar += step; }
      p = p + (step + kPushDistance<real_t>()) * d;
      vol = resolve_after_step(gg, entered, p);
    }
    printf("  traversed %.4f mm of the rotated bar (expect 180.000)\n", in_bar);
    check(std::fabs(in_bar - 180.0) < 1e-3, "ray crosses the rotated long axis");

    int mismatches = 0;
    Philox<real_t> rng(7, 5);
    int steps = 0;
    for (int trial = 0; trial < 5000; ++trial) {
      Vec3<real_t> q{(2 * rng.uniform() - 1) * 190, (2 * rng.uniform() - 1) * 190,
                     (2 * rng.uniform() - 1) * 190};
      if (locate(gg, q) == kOutsideWorld) { continue; }
      const real_t ct = 2 * rng.uniform() - 1;
      const real_t st = std::sqrt(1 - ct * ct);
      const real_t ph = units::twopi<real_t>() * rng.uniform();
      steps += march(gg, q, Vec3<real_t>{st * std::cos(ph), st * std::sin(ph), ct}, 8,
                     mismatches, false);
    }
    printf("  %d crossings with rotation, %d mismatches\n", steps, mismatches);
    check(steps > 5000, "enough rotated crossings to be meaningful");
    check(mismatches == 0, "rotation does not break locate/step agreement");
  }

  printf("\n%s (%d failures)\n", fails ? "FAILED" : "ALL PASS", fails);
  return fails;
}
