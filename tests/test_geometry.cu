// Validates the B1 solids without any Geant4 involvement:
//  - Monte Carlo volume vs the analytic formula (Trd volume also pins B1 scoring mass)
//  - ray entry/exit consistency against inside()
#include <cstdio>
#include <cmath>
#include "core/rng.cuh"
#include "core/units.cuh"
#include "geometry/solids.cuh"

using namespace g4gpu;
using namespace g4gpu::geom;
using real_t = double;

static int fails = 0;
static void check(bool ok, const char* what) {
  printf(ok ? "  ok:   %s\n" : "  FAIL: %s\n", what);
  if (!ok) { ++fails; }
}

// B1 geometry, in mm (Geant4 internal length unit).
static Solid<real_t> b1_world()    { return {SolidType::kBox,  {120.0, 120.0, 180.0}}; }
static Solid<real_t> b1_envelope() { return {SolidType::kBox,  {100.0, 100.0, 150.0}}; }
static Solid<real_t> b1_shape1()   { return {SolidType::kCons, {20.0, 40.0, 30.0}}; }
static Solid<real_t> b1_shape2()   { return {SolidType::kTrd,  {60.0, 60.0, 50.0, 80.0, 30.0}}; }

/// Monte Carlo volume in mm^3 over a bounding box of half-extents h.
static real_t mc_volume(const Solid<real_t>& s, real_t hx, real_t hy, real_t hz, int n) {
  Philox<real_t> rng(99, 1);
  int hits = 0;
  for (int i = 0; i < n; ++i) {
    Vec3<real_t> q{(2 * rng.uniform() - 1) * hx, (2 * rng.uniform() - 1) * hy,
                   (2 * rng.uniform() - 1) * hz};
    if (inside(s, q)) { ++hits; }
  }
  return real_t(hits) / real_t(n) * (8 * hx * hy * hz);
}

int main() {
  const real_t cm3 = 1000.0;  // mm^3 per cm^3

  printf("== Trd volume (pins B1 scoring mass) ==\n");
  const real_t v_trd = mc_volume(b1_shape2(), 61, 81, 31, 4000000);
  // dx const at 6cm, dy linear 5->8cm, dz 3cm: V = 4*dx*<dy>*(2dz) = 4*6*6.5*6 = 936 cm^3
  printf("  MC       = %.2f cm^3\n", v_trd / cm3);
  printf("  analytic = 936.00 cm^3\n");
  check(std::fabs(v_trd / cm3 / 936.0 - 1.0) < 2e-3, "Trd volume matches analytic to 0.2%");
  const real_t mass_kg = (v_trd / cm3) * 1.85 / 1000.0;  // G4_BONE_COMPACT_ICRU
  printf("  mass @1.85 g/cm3 = %.4f kg   (B1 reports 1.7316 kg)\n", mass_kg);
  check(std::fabs(mass_kg / 1.7316 - 1.0) < 2e-3, "scoring mass matches B1 output");

  printf("== Cons volume ==\n");
  const real_t v_cons = mc_volume(b1_shape1(), 41, 41, 31, 4000000);
  // frustum: V = pi*h/3*(r1^2+r1*r2+r2^2) = pi*6/3*(4+8+16) = 175.93 cm^3
  const real_t v_cons_ana = units::pi<real_t>() * 6.0 / 3.0 * (4.0 + 8.0 + 16.0);
  printf("  MC = %.2f cm^3, analytic = %.2f cm^3\n", v_cons / cm3, v_cons_ana);
  check(std::fabs(v_cons / cm3 / v_cons_ana - 1.0) < 3e-3, "Cons volume matches frustum formula");

  printf("== ray entry/exit consistency ==\n");
  const Solid<real_t> solids[4] = {b1_world(), b1_envelope(), b1_shape1(), b1_shape2()};
  const char* names[4] = {"World box", "Envelope box", "Shape1 cons", "Shape2 trd"};
  for (int k = 0; k < 4; ++k) {
    Philox<real_t> rng(7, k);
    int entered = 0, entry_on_surface = 0, exit_leaves = 0, exits = 0;
    for (int i = 0; i < 200000; ++i) {
      // launch from well outside toward the origin region
      Vec3<real_t> q{(2 * rng.uniform() - 1) * 400, (2 * rng.uniform() - 1) * 400,
                     (2 * rng.uniform() - 1) * 400};
      if (inside(solids[k], q)) { continue; }
      Vec3<real_t> target{(2 * rng.uniform() - 1) * 60, (2 * rng.uniform() - 1) * 60,
                          (2 * rng.uniform() - 1) * 60};
      const Vec3<real_t> d = normalize(target - q);
      const real_t t_in = dist_in(solids[k], q, d);
      if (t_in >= kInfinity<real_t>()) { continue; }
      ++entered;
      // nudge just past the entry point: must now be inside
      const Vec3<real_t> hit = q + (t_in + 1e-6) * d;
      if (inside(solids[k], hit)) { ++entry_on_surface; }
      const real_t t_out = dist_out(solids[k], hit, d);
      if (t_out < kInfinity<real_t>()) {
        ++exits;
        const Vec3<real_t> out = hit + (t_out + 1e-6) * d;
        if (!inside(solids[k], out)) { ++exit_leaves; }
      }
    }
    printf("  %-13s entered %6d  entry lands inside %6d  exit leaves %6d/%6d\n",
           names[k], entered, entry_on_surface, exit_leaves, exits);
    check(entered > 1000, "enough rays entered to be meaningful");
    check(entry_on_surface == entered, "dist_in lands strictly inside the solid");
    check(exit_leaves == exits, "dist_out lands strictly outside the solid");
  }

  printf("\n%s (%d failures)\n", fails ? "FAILED" : "ALL PASS", fails);
  return fails;
}
