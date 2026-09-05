// Low-energy heavy-particle ionisation, against G4BraggModel called directly.
//
// G4BraggModel::DEDX has a four-way decision. For any of the 74 NIST materials tabulated in
// G4PSTARStopping it returns that tabulated stopping power; for a molecule in the Ziegler
// 1988 list it uses molecular data; otherwise it falls through to the per-element Ziegler
// parameterisation, which is what this port implements and what any user-defined material
// gets.
//
// All four of B1's materials are in PSTAR, so comparing there would only measure
// Ziegler-versus-PSTAR, not the transcription. The oracle therefore also builds a material
// deliberately outside both lists (60% Si, 40% Ge by weight) whose PSTAR index is -1: that
// is where the transcription is actually checked. The PSTAR difference on B1's materials is
// then reported separately, as a known and quantified gap rather than a hidden one.
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <string>
#include "core/particle.cuh"
#include "data/materials.cuh"
#include "physics/em/bragg.cuh"
#include "physics/em/hadron_ionisation.cuh"

using namespace g4gpu;
using real_t = double;

static ParticleType type_of(const std::string& n) {
  if (n == "proton") { return ParticleType::kProton; }
  if (n == "pi+") { return ParticleType::kPionPlus; }
  if (n == "kaon+") { return ParticleType::kKaonPlus; }
  if (n == "alpha") { return ParticleType::kAlpha; }
  return ParticleType::kNumTypes;
}

int main() {
  // B1's four, plus two materials the oracle builds to reach branches B1 cannot:
  //
  //   CustomSiGe            no NIST name and no chemical formula -> the per-element Ziegler
  //                         fit, which is the branch a user-defined material takes;
  //   CustomMolecularWater  no NIST name but the formula "H_2O" -> G4PSTARStopping resolves
  //                         it by formula to G4_WATER's row. That path is easy to get wrong in
  //                         the direction that looks right: fall back to Ziegler instead and
  //                         the answer is still the right order of magnitude and 29% out.
  data::MaterialTable<real_t> table{};
  table.count = 0;
  data::Material<real_t> b1[data::kNumMaterials];
  data::build_b1_materials<real_t>(b1);
  const char* g4names[6] = {"G4_AIR",               "G4_WATER",
                            "G4_A-150_TISSUE",      "G4_BONE_COMPACT_ICRU",
                            "CustomSiGe",           "CustomMolecularWater"};
  data::Material<real_t> mats[6];
  for (int i = 0; i < 4; ++i) { mats[i] = b1[i]; }
  {
    const int zs[2] = {14, 32};
    const real_t w[2] = {real_t(0.6), real_t(0.4)};
    const int idx = data::add_material<real_t>(table, real_t(4.2), 2, zs, w,
                                               real_t(224.66048772837619));
    if (idx < 0) {
      std::printf("cannot build the custom material\n");
      return 1;
    }
    mats[4] = table.m[idx];
    // No name, no formula: -1 for both, which is what makes this the Ziegler branch.
    data::set_nist_stopping<real_t>(mats[4], "CustomSiGe");
  }
  {
    // H2O by atom count, 1 g/cm3, mean excitation as Geant4 derives it for a hand-built
    // material (ref/oracle/materials.csv) - not water's tabulated 78 eV, because Geant4 does
    // not look that up for a material it was not asked to build by name.
    const real_t aH = data::atomic_mass<real_t>(1), aO = data::atomic_mass<real_t>(8);
    const real_t total = real_t(2) * aH + aO;
    const int zs[2] = {1, 8};
    const real_t w[2] = {real_t(2) * aH / total, aO / total};
    const int idx = data::add_material<real_t>(table, real_t(1.0), 2, zs, w,
                                               real_t(68.9984175));
    if (idx < 0) {
      std::printf("cannot build the molecular material\n");
      return 1;
    }
    mats[5] = table.m[idx];
    data::set_nist_stopping<real_t>(mats[5], "CustomMolecularWater", "H_2O");
    if (mats[5].nist_stopping < 0) {
      std::printf("FAIL: \"H_2O\" did not resolve to a NIST stopping row\n");
      return 1;
    }
  }

  const char* env = std::getenv("G4GPU_ORACLE");
  const std::string dir = (env != nullptr) ? env : "ref/oracle";
  FILE* f = std::fopen((dir + "/bragg.csv").c_str(), "r");
  if (f == nullptr) {
    std::printf("cannot read %s/bragg.csv\n", dir.c_str());
    return 1;
  }
  char line[512];
  if (std::fgets(line, sizeof line, f) == nullptr) { std::fclose(f); return 1; }

  struct Acc { double worst; std::string where; int n; };
  Acc ziegler{0, "", 0}, pstar_gap{0, "", 0}, ziegler_ion{0, "", 0}, astar_alpha{0, "", 0},
      astar_other{0, "", 0};
  auto track = [](Acc& a, double ours, double g4, const char* p, const char* m, double e) {
    if (g4 <= 0) { return; }
    ++a.n;
    const double dev = std::fabs(ours / g4 - 1);
    if (dev > a.worst) {
      a.worst = dev;
      char buf[180];
      std::snprintf(buf, sizeof buf, "%s in %s at %.4g MeV", p, m, e);
      a.where = buf;
    }
  };

  while (std::fgets(line, sizeof line, f) != nullptr) {
    char mat[128], part[32];
    double e, cut, d1, d2, dp;
    int ip, ia;
    if (std::sscanf(line, "%127[^,],%31[^,],%lf,%lf,%lf,%lf,%d,%lf,%d", mat, part, &e, &cut,
                    &d1, &d2, &ip, &dp, &ia) != 9) {
      continue;
    }
    const ParticleType t = type_of(part);
    if (t == ParticleType::kNumTypes) { continue; }
    int mi = -1;
    for (int i = 0; i < 6; ++i) {
      if (std::string(mat) == g4names[i]) { mi = i; break; }
    }
    if (mi < 0) { continue; }
    const real_t ours = em::bragg_dedx(mats[mi], t, real_t(e), real_t(cut));
    const real_t ours_ion = em::bragg_ion_dedx(mats[mi], t, real_t(e), real_t(cut));
    const bool below_boundary = e < em::bragg_bethe_boundary(particle_def<real_t>(t));
    if (ip < 0) {
      track(ziegler, ours, d1, part, mat, e);  // the branch this port implements
    } else if (below_boundary) {
      // Geant4 uses PSTAR here. Only report the gap below the Bragg/Bethe-Bloch boundary,
      // since above it neither model is the one Geant4 would select.
      track(pstar_gap, ours, d1, part, mat, e);
    }
    if (ia < 0) {
      track(ziegler_ion, ours_ion, d2, part, mat, e);
    } else if (below_boundary) {
      track((t == ParticleType::kAlpha) ? astar_alpha : astar_other, ours_ion, d2, part, mat,
            e);
    }
  }
  std::fclose(f);

  printf("== per-element Ziegler branch (PSTAR / ASTAR index -1) ==\n");
  printf("  %-24s %8s %12s   %s\n", "model", "points", "worst dev", "where");
  printf("  %-24s %8d %11.4f%%   %s\n", "G4BraggModel (protons)", ziegler.n,
         100 * ziegler.worst, ziegler.where.c_str());
  printf("  %-24s %8d %11.4f%%   %s\n", "G4BraggIonModel (ions)", ziegler_ion.n,
         100 * ziegler_ion.worst, ziegler_ion.where.c_str());

  printf("\n== tabulated PSTAR and ASTAR, where Geant4 uses them ==\n");
  printf("  (the 74 NIST materials, below the Bragg/Bethe-Bloch boundary)\n");
  printf("  %-24s %8d %11.4f%%   %s\n", "vs PSTAR", pstar_gap.n, 100 * pstar_gap.worst,
         pstar_gap.where.c_str());
  printf("  %-24s %8d %11.4f%%   %s\n", "vs ASTAR (alphas)", astar_alpha.n,
         100 * astar_alpha.worst, astar_alpha.where.c_str());
  printf("  %-24s %8d %11.4f%%   %s\n", "vs ASTAR (other ions)", astar_other.n,
         100 * astar_other.worst, astar_other.where.c_str());

  int fails = 0;
  if (ziegler.n == 0 || ziegler_ion.n == 0) {
    printf("\n  FAIL: a Ziegler branch was never exercised - the oracle has no material\n"
           "        outside PSTAR/ASTAR, so nothing was actually tested\n");
    ++fails;
  }
  if (pstar_gap.n == 0 || astar_alpha.n == 0 || astar_other.n == 0) {
    printf("\n  FAIL: the tabulated branch was never exercised - no oracle material had a\n"
           "        PSTAR or ASTAR index, so the tables were not tested\n");
    ++fails;
  }
  if (ziegler.worst > 0.001) {
    printf("\n  FAIL: G4BraggModel Ziegler branch exceeds 0.1%%\n");
    ++fails;
  }
  if (ziegler_ion.worst > 0.001) {
    printf("\n  FAIL: G4BraggIonModel Ziegler branch exceeds 0.1%%\n");
    ++fails;
  }
  // These two were a *reported gap* of 29% and 19% until the tables were extracted from
  // Geant4's own sources and read through its own spline. They are now a requirement, and a
  // tight one: the same table, the same interpolation, the same extrapolation below 1 keV.
  // Anything above a part in a million means one of those three has drifted.
  if (pstar_gap.worst > 1e-6) {
    printf("\n  FAIL: the tabulated PSTAR branch no longer reproduces Geant4\n");
    ++fails;
  }
  // The alpha path reads the ASTAR table at the alpha's own energy and is asserted tight.
  // The non-alpha path reaches the same answer by a different route than Geant4 - see the
  // comment on bragg_ion_dedx_unrestricted - and agrees to about one part in a hundred
  // thousand, which is asserted separately so that the looser claim cannot hide behind the
  // tighter one.
  if (astar_alpha.worst > 1e-6) {
    printf("\n  FAIL: the tabulated ASTAR branch no longer reproduces Geant4 for alphas\n");
    ++fails;
  }
  if (astar_other.worst > 5e-5) {
    printf("\n  FAIL: the ion model's non-alpha path drifted from Geant4 by more than 5e-5\n");
    ++fails;
  }
  printf("\n%s (%d failures)\n", fails ? "FAILED" : "PASSED", fails);
  return fails ? 1 : 0;
}
