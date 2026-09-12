// Validates electron stopping power and CSDA range against NIST ESTAR.
//
// CAVEAT: the ESTAR reference numbers below are approximate values for water, used with
// loose (10%) tolerances. They are good enough to catch a transcription error or a units
// slip by an order of magnitude, which is their purpose. For a tight comparison, pull the
// real tables from physics.nist.gov/Star and narrow the tolerances.
#include <cstdio>
#include <cmath>
#include "physics/em/electron_processes.cuh"

using namespace g4gpu;
using real_t = double;

static int fails = 0;
static void check(bool ok, const char* what) {
  printf(ok ? "  ok:   %s\n" : "  FAIL: %s\n", what);
  if (!ok) { ++fails; }
}

int main() {
  data::Material<real_t> mats[data::kNumMaterials];
  data::build_b1_materials<real_t>(mats);

  printf("== derived material parameters ==\n");
  const char* names[4] = {"air", "water", "A-150 tissue", "bone compact"};
  for (int i = 0; i < data::kNumMaterials; ++i) {
    printf("  %-13s Zeff = %5.2f  I = %5.1f eV  X0 = %8.2f mm (%6.2f g/cm2)\n", names[i],
           mats[i].z_eff, mats[i].mean_excitation * 1e6, mats[i].radiation_length,
           mats[i].radiation_length * mats[i].density / 10.0);
  }
  // Water X0 is tabulated at 36.08 g/cm2 = 360.8 mm.
  const real_t x0_water = mats[data::kWater].radiation_length;
  check(std::fabs(x0_water / 360.8 - 1.0) < 0.03, "water radiation length ~36 cm");
  // Water Zeff should be close to 10/3 electrons per atom for H2O.
  check(mats[data::kWater].z_eff > 3.0 && mats[data::kWater].z_eff < 3.8, "water Zeff ~3.33");

  printf("== collision stopping power in water, MeV cm2/g ==\n");
  // ESTAR water collision stopping power: ~1.85 at 1 MeV, rising slowly to ~1.97 at 10 MeV.
  const real_t es[5] = {0.1, 1.0, 5.0, 6.0, 10.0};
  const real_t ref[5] = {4.115, 1.849, 1.911, 1.925, 1.968};
  for (int i = 0; i < 5; ++i) {
    const real_t dedx_mm = em::collision_dedx<real_t>(mats[data::kWater], es[i], false, 1e9)  /* unrestricted: ESTAR is */;
    // MeV/mm -> MeV cm2/g:  *10 mm/cm  / rho
    const real_t sp = dedx_mm * 10.0 / mats[data::kWater].density;
    printf("  E = %5.1f MeV : computed %.4f, ESTAR ~%.4f, ratio %.4f\n", es[i], sp, ref[i],
           sp / ref[i]);
    check(std::fabs(sp / ref[i] - 1.0) < 0.10, "collision dE/dx within 10% of ESTAR");
  }

  printf("== positron vs electron dE/dx ==\n");
  const real_t e_el = em::collision_dedx<real_t>(mats[data::kWater], 1.0, false, 1e9);
  const real_t e_po = em::collision_dedx<real_t>(mats[data::kWater], 1.0, true, 1e9);
  printf("  1 MeV in water: electron %.5f, positron %.5f MeV/mm (ratio %.4f)\n", e_el, e_po,
         e_po / e_el);
  // Bhabha differs from Moller by only a few percent at 1 MeV.
  check(std::fabs(e_po / e_el - 1.0) < 0.15, "Bhabha within 15% of Moller at 1 MeV");

  printf("== monotonicity ==\n");
  bool mono = true;
  for (real_t e = 2.5; e < 20.0; e += 0.5) {  // minimum ionization in water is near 1.5 MeV
    if (em::collision_dedx<real_t>(mats[data::kWater], e, false, 1e9)
        < em::collision_dedx<real_t>(mats[data::kWater], e - 0.5, false, 1e9)) {
      mono = false;
    }
  }
  check(mono, "collision dE/dx rises monotonically above 2.5 MeV (relativistic rise)");

  printf("== range table vs Geant4 (restricted, not CSDA) ==\n");
  // Build with the real Seltzer-Berger radiative term, as the drivers do.
  const int zs_sb[10] = {1, 6, 7, 8, 9, 12, 15, 16, 18, 20};
  static data::SBTableSet<real_t> sb{};
  static data::BremsTable<real_t> bt;
  const bool sb_ok = data::load_sb_tables<real_t>(
      "D:\\Documents\\Geant4\\Windows\\geant4-v11.1.1-install\\share\\Geant4\\data"
      "\\G4EMLOW8.2\\brem_SB", zs_sb, 10, sb);
  if (sb_ok) { data::build_brems_tables<real_t>(mats, sb, bt); }
  if (!sb_ok) {
    printf("  cannot read the Seltzer-Berger tables - the e+- dE/dx table is the sum over\n"
           "  G4eIonisation AND G4eBremsstrahlung and cannot be built without them.\n");
    return 1;
  }
  // Static: 212 KiB of table is not a stack object. `false` is e-; the table now carries a
  // species dimension because Geant4 builds one per particle and Bhabha is not Moller.
  static em::RangeTable<real_t> rt;
  em::build_range_table<real_t>(mats, rt, &sb);
  // The table now integrates the RESTRICTED stopping power, matching what Geant4 builds its
  // range table from, so the reference is Geant4's own range - not ESTAR's CSDA range.
  // Values from ref/oracle/electron_tables.csv, Geant4 11.1.1, default 1 mm cut.
  struct RangeRef { int mat; const char* name; real_t e, g4_mm; };
  const RangeRef rrefs[9] = {
      {data::kWater, "water", real_t(1.0000), real_t(4.45932)},
      {data::kWater, "water", real_t(5.0119), real_t(28.2368)},
      {data::kWater, "water", real_t(5.9566), real_t(33.8507)},
      {data::kA150Tissue, "tissue", real_t(1.0000), real_t(3.93578)},
      {data::kA150Tissue, "tissue", real_t(5.0119), real_t(25.0852)},
      {data::kA150Tissue, "tissue", real_t(5.9566), real_t(30.0916)},
      {data::kBoneCompact, "bone", real_t(1.0000), real_t(2.57051)},
      {data::kBoneCompact, "bone", real_t(5.0119), real_t(16.1889)},
      {data::kBoneCompact, "bone", real_t(5.9566), real_t(19.4051)}};
  for (const auto& rr : rrefs) {
    const real_t ours = rt.lookup(rr.mat, false, rr.e);
    printf("  %-7s E = %6.4f MeV : ours %8.4f mm, Geant4 %8.4f mm, ratio %.4f\n", rr.name,
           rr.e, ours, rr.g4_mm, ours / rr.g4_mm);
    check(std::fabs(ours / rr.g4_mm - 1.0) < 0.05, "range within 5% of Geant4");
  }

  printf("== ranges in the B1 materials at 5 MeV ==\n");
  for (int i = 0; i < data::kNumMaterials; ++i) {
    printf("  %-13s range = %8.3f mm\n", names[i], rt.lookup(i, false, 5.0));
  }
  // Shape2 is 60 mm thick; a 5 MeV electron in bone must stop well inside it.
  const real_t r_bone = rt.lookup(data::kBoneCompact, false, 5.0);
  check(r_bone > 5.0 && r_bone < 30.0, "5 MeV electron range in bone is well under 60 mm");
  check(rt.lookup(data::kWater, false, 5.0) > r_bone, "range in water exceeds range in denser bone");

  printf("== range table monotonic and continuous ==\n");
  bool r_mono = true;
  real_t prev = 0;
  for (real_t e = 0.01; e < 50.0; e *= 1.05) {
    const real_t r = rt.lookup(data::kWater, false, e);
    if (r < prev) { r_mono = false; }
    prev = r;
  }
  check(r_mono, "range increases monotonically with energy");

  printf("== MSC (Highland) ==\n");
  // A 5 MeV electron over 1 mm of water: X/X0 ~ 1/360, theta0 should be a few degrees.
  const real_t th = em::msc_theta0<real_t>(mats[data::kWater], 5.0, 1.0);
  printf("  5 MeV e- over 1 mm water: theta0 = %.5f rad = %.3f deg\n", th, th * 180.0 / 3.14159265);
  check(th > 0.001 && th < 0.5, "theta0 is a physically plausible few degrees");
  const real_t th10 = em::msc_theta0<real_t>(mats[data::kWater], 5.0, 10.0);
  check(th10 > th, "theta0 grows with step length");
  const real_t th_hi = em::msc_theta0<real_t>(mats[data::kWater], 50.0, 1.0);
  check(th_hi < th, "theta0 shrinks with increasing energy");
  const real_t th_bone = em::msc_theta0<real_t>(mats[data::kBoneCompact], 5.0, 1.0);
  check(th_bone > th, "denser bone scatters more than water over the same length");

  printf("\n%s (%d failures)\n", fails ? "FAILED" : "ALL PASS", fails);
  return fails;
}
