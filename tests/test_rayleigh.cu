// Validates the Livermore Rayleigh cross section against Geant4's own values.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include "data/materials.cuh"
#include "data/rayleigh_data.cuh"

using namespace g4gpu;
using real_t = double;
static const char* kDir = "D:/Documents/Geant4/Windows/geant4-v11.1.1-install/share/Geant4"
                          "/data/G4EMLOW8.2/epics2017/rayl";

static int mat_index(const std::string& n) {
  if (n == "G4_AIR") return data::kAir;
  if (n == "G4_WATER") return data::kWater;
  if (n == "G4_A-150_TISSUE") return data::kA150Tissue;
  if (n == "G4_BONE_COMPACT_ICRU") return data::kBoneCompact;
  return -1;
}

/// P43 (docs/RISK.md V356): every element, Z = 1 .. 100, in ONE table - the scene no geometry
/// can exceed since V355 (kRayleighMaxElements was 16) - against
/// G4LivermoreRayleighModel::ComputeCrossSectionPerAtom at 441 energies a Z (pe_atom.csv, ref/dump's
/// pe_shells dump). To 1e-12 relative: the same points, the same 1/E^2, and two algebraically
/// equal linear interpolations (G4PhysicsVector's y1 + b*dy, this port's y1*(1-f) + y2*f).
static int z_scan() {
  printf("\n== every element, Z = 1 .. 100 (P43, docs/RISK.md V356) ==\n");
  int all[100];
  for (int z = 1; z <= 100; ++z) all[z - 1] = z;
  static data::RayleighTable<real_t> rt{};
  std::vector<real_t> te, tv;
  if (!data::load_rayleigh<real_t>(kDir, all, 100, rt, te, tv)) {
    printf("FAILED: the table of Z = 1 .. 100 did not load (the line above says why)\n");
    return 1;
  }
  rt.table_e = te.data(); rt.table_v = tv.data(); rt.table_n = (int)te.size();
  printf("  loaded %d elements, %d points\n", rt.n_elements, rt.table_n);
  const char* env = std::getenv("G4GPU_ORACLE");
  const std::string dir = (env != nullptr) ? env : "ref/oracle";
  std::ifstream in(dir + "/pe_atom.csv");
  if (!in) {
    printf("FAILED: cannot read %s/pe_atom.csv - ref/dump's pe_shells dump writes it\n",
           dir.c_str());
    return 1;
  }
  std::string line; std::getline(in, line);
  double worst[101] = {0};
  long rows = 0, bad = 0;
  while (std::getline(in, line)) {
    std::vector<std::string> f; std::stringstream ss(line); std::string c;
    while (std::getline(ss, c, ',')) f.push_back(c);
    if (f.size() < 4) continue;
    const int z = std::atoi(f[0].c_str());
    if (z < 1 || z > 100) continue;
    const double e = std::atof(f[1].c_str());
    const double g4 = std::atof(f[3].c_str());
    const double ours = data::rayleigh_xs_per_atom<real_t>(rt, z, e);
    const double rel = (g4 != 0) ? std::fabs(ours / g4 - 1.0) : std::fabs(ours);
    ++rows;
    if (rel > worst[z]) worst[z] = rel;
    if (rel > 1e-12 && ++bad <= 10)
      printf("  MISMATCH Z %d at %.6g MeV: ours %.17g, Geant4 %.17g (rel %.3g)\n", z, e, ours, g4,
             rel);
  }
  double w = 0; int wz = 0;
  for (int z = 1; z <= 100; ++z) if (worst[z] > w) { w = worst[z]; wz = z; }
  printf("  cross section per atom, %ld points: worst %.3g relative (Z = %d); Fe %.3g, Cu %.3g, "
         "Ag %.3g, W %.3g, Pb %.3g, U %.3g\n", rows, w, wz, worst[26], worst[29], worst[47],
         worst[74], worst[82], worst[92]);
  if (rows < 98 * 441 || bad > 0) {
    printf("FAILED: %ld of %ld points outside 1e-12 (expected %d points)\n", bad, rows, 98 * 441);
    return 1;
  }
  printf("  every element agrees\n");
  return 0;
}

int main() {
  data::Material<real_t> mats[data::kNumMaterials];
  data::build_b1_materials<real_t>(mats);
  const int zs[10] = {1, 6, 7, 8, 9, 12, 15, 16, 18, 20};
  data::RayleighTable<real_t> rt{};
  std::vector<real_t> te, tv;
  if (!data::load_rayleigh<real_t>(kDir, zs, 10, rt, te, tv)) { printf("LOAD FAILED\n"); return 1; }
  rt.table_e = te.data(); rt.table_v = tv.data(); rt.table_n = (int)te.size();
  printf("loaded %d elements, %d points\n\n", rt.n_elements, rt.table_n);

  std::ifstream in("D:/g4gpu/ref/oracle/gamma_xs.csv");
  std::string line; std::getline(in, line);
  double worst = 0, worst_e = 0; int n = 0;
  printf("  %-8s %10s %14s %14s %8s\n", "material", "E (MeV)", "ours 1/mm", "G4 1/mm", "ratio");
  const double probe[4] = {0.03, 0.1, 1.0, 6.0};
  while (std::getline(in, line)) {
    std::vector<std::string> f; std::stringstream ss(line); std::string c;
    while (std::getline(ss, c, ',')) f.push_back(c);
    if (f.size() < 6) continue;
    const int m = mat_index(f[0]); if (m < 0) continue;
    const real_t e = std::atof(f[1].c_str());
    const real_t g4 = std::atof(f[5].c_str());
    if (g4 <= 0) continue;
    real_t ours = 0;
    for (int i = 0; i < mats[m].n_elements; ++i)
      ours += mats[m].n_atoms[i]
              * data::rayleigh_xs_per_atom<real_t>(rt, (int)(mats[m].z[i] + 0.5), e);
    const double r = ours / g4;
    if (e >= 1e-3 && e <= 100) { ++n; if (std::fabs(r-1) > worst) { worst = std::fabs(r-1); worst_e = e; } }
    bool want = false;
    for (double p : probe) if (std::fabs(e-p)/p < 0.02) want = true;
    if (want && m == data::kBoneCompact)
      printf("  %-8s %10.4g %14.6g %14.6g %8.4f\n", "bone", e, ours, g4, r);
  }
  printf("\n  worst deviation over 1 keV - 100 MeV, all materials: %.4f%%  (n=%d, at %.4g MeV)\n",
         100*worst, n, worst_e);
  return z_scan();
}
