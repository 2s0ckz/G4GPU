// Validates the Livermore photoelectric cross section against Geant4's own dumped values.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>

#include "data/materials.cuh"
#include "data/photoelectric_data.cuh"

using namespace g4gpu;
using real_t = double;

static const char* kPhotDir =
    "D:\\Documents\\Geant4\\Windows\\geant4-v11.1.1-install\\share\\Geant4\\data\\G4EMLOW8.2"
    "\\epics2017\\phot";
static const char* kOracle = "D:\\g4gpu\\ref\\oracle\\";

static int mat_index(const std::string& n) {
  if (n == "G4_AIR") { return data::kAir; }
  if (n == "G4_WATER") { return data::kWater; }
  if (n == "G4_A-150_TISSUE") { return data::kA150Tissue; }
  if (n == "G4_BONE_COMPACT_ICRU") { return data::kBoneCompact; }
  return -1;
}

int main() {
  data::Material<real_t> mats[data::kNumMaterials];
  data::build_b1_materials<real_t>(mats);

  // Every element appearing in the four B1 materials.
  const int zs[10] = {1, 6, 7, 8, 9, 12, 15, 16, 18, 20};
  data::PhotoElectricTable<real_t> pe{};
  std::vector<real_t> te, tv;
  if (!data::load_photoelectric<real_t>(kPhotDir, zs, 10, pe, te, tv)) {
    std::printf("FAILED to load photoelectric data from %s\n", kPhotDir);
    return 1;
  }
  pe.table_e = te.data();
  pe.table_v = tv.data();
  pe.table_n = static_cast<int>(te.size());
  std::printf("loaded %d elements, %d tabulated points\n\n", pe.n_elements, pe.table_n);

  // Compare the macroscopic photoelectric cross section, 1/mm, against gamma_xs.csv.
  std::ifstream in(std::string(kOracle) + "gamma_xs.csv");
  if (!in) { std::printf("cannot open oracle csv\n"); return 1; }
  std::string line;
  std::getline(in, line);  // header

  struct Band { const char* name; real_t lo, hi; double worst; double worst_e; int n; };
  Band bands[3] = {{"E >= 100 keV (high poly)", 0.1, 1e4, 0, 0, 0},
                   {"5 - 100 keV (low poly)", 5e-3, 0.1, 0, 0, 0},
                   {"below 5 keV (tabulated)", 0.0, 5e-3, 0, 0, 0}};

  while (std::getline(in, line)) {
    std::vector<std::string> f;
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ',')) { f.push_back(cell); }
    if (f.size() < 6) { continue; }
    const int m = mat_index(f[0]);
    if (m < 0) { continue; }
    const real_t e = std::atof(f[1].c_str());
    const real_t g4 = std::atof(f[2].c_str());
    if (g4 <= 0) { continue; }

    real_t ours = 0;
    for (int i = 0; i < mats[m].n_elements; ++i) {
      const int z = static_cast<int>(mats[m].z[i] + 0.5);
      ours += mats[m].n_atoms[i] * data::photoelectric_xs_per_atom<real_t>(pe, z, e);
    }
    const double ratio = ours / g4;
    for (auto& b : bands) {
      if (e >= b.lo && e < b.hi) {
        ++b.n;
        if (std::fabs(ratio - 1.0) > b.worst) { b.worst = std::fabs(ratio - 1.0); b.worst_e = e; }
      }
    }
  }

  std::printf("photoelectric macroscopic cross section vs Geant4:\n");
  for (const auto& b : bands) {
    std::printf("  %-26s worst %8.4f%%   (n=%d, at %.4g MeV)\n", b.name, 100 * b.worst, b.n,
                b.worst_e);
  }

  // A few explicit points, since B1's degraded photons live around 50-200 keV.
  std::printf("\n  %-22s %10s %14s %14s %8s\n", "material", "E (MeV)", "ours 1/mm", "G4 1/mm",
              "ratio");
  std::ifstream in2(std::string(kOracle) + "gamma_xs.csv");
  std::getline(in2, line);
  const double probe[4] = {0.05, 0.1, 0.5, 6.0};
  while (std::getline(in2, line)) {
    std::vector<std::string> f;
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ',')) { f.push_back(cell); }
    if (f.size() < 6) { continue; }
    const int m = mat_index(f[0]);
    if (m != data::kBoneCompact) { continue; }
    const real_t e = std::atof(f[1].c_str());
    bool want = false;
    for (double p : probe) { if (std::fabs(e - p) / p < 0.02) { want = true; } }
    if (!want) { continue; }
    const real_t g4 = std::atof(f[2].c_str());
    real_t ours = 0;
    for (int i = 0; i < mats[m].n_elements; ++i) {
      const int z = static_cast<int>(mats[m].z[i] + 0.5);
      ours += mats[m].n_atoms[i] * data::photoelectric_xs_per_atom<real_t>(pe, z, e);
    }
    std::printf("  %-22s %10.4g %14.6g %14.6g %8.4f\n", "bone", e, ours, g4,
                (g4 > 0 ? ours / g4 : 0));
  }
  return 0;
}
