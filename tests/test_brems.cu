// Validates the Seltzer-Berger bremsstrahlung tables against Geant4's own values.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>

#include "data/materials.cuh"
#include "data/brems_data.cuh"

using namespace g4gpu;
using real_t = double;

static const char* kSbDir =
    "D:\\Documents\\Geant4\\Windows\\geant4-v11.1.1-install\\share\\Geant4\\data\\G4EMLOW8.2"
    "\\brem_SB";
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

  const int zs[10] = {1, 6, 7, 8, 9, 12, 15, 16, 18, 20};
  data::SBTableSet<real_t> sb{};
  if (!data::load_sb_tables<real_t>(kSbDir, zs, 10, sb)) {
    std::printf("FAILED to load Seltzer-Berger tables from %s\n", kSbDir);
    return 1;
  }
  std::printf("loaded %d Seltzer-Berger element tables\n", sb.n_elements);

  data::BremsTable<real_t> bt;
  data::build_brems_tables<real_t>(mats, sb, bt);
  std::printf("built brems tables (%d bins)\n\n", data::kBremsBins);

  struct Stat { double worst = 0, at = 0; int n = 0; };
  Stat dedx_s, xs_s;

  std::ifstream in(std::string(kOracle) + "electron_tables.csv");
  if (!in) { std::printf("cannot open oracle csv\n"); return 1; }
  std::string line;
  std::getline(in, line);

  std::printf("  %-8s %-3s %9s %13s %13s %8s %13s %13s %8s\n", "material", "p", "E (MeV)",
              "dedx ours", "dedx G4", "ratio", "xs ours", "xs G4", "ratio");
  const double probe[4] = {0.5, 1.0, 3.0, 6.0};

  while (std::getline(in, line)) {
    std::vector<std::string> f;
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ',')) { f.push_back(cell); }
    if (f.size() < 14) { continue; }
    const int m = mat_index(f[0]);
    if (m < 0 || m == data::kAir) { continue; }
    const bool pos = (f[1] == "e+");
    const real_t e = std::atof(f[2].c_str());
    const real_t g4_dedx = std::atof(f[12].c_str());
    const real_t g4_xs = std::atof(f[13].c_str());

    const real_t our_dedx = bt.dedx_at(m, pos, e);
    const real_t our_xs = bt.xs_at(m, pos, e);

    if (e >= 0.1 && e <= 100.0) {
      if (g4_dedx > 0) {
        const double r = our_dedx / g4_dedx;
        ++dedx_s.n;
        if (std::fabs(r - 1.0) > dedx_s.worst) { dedx_s.worst = std::fabs(r - 1.0); dedx_s.at = e; }
      }
      if (g4_xs > 0) {
        const double r = our_xs / g4_xs;
        ++xs_s.n;
        if (std::fabs(r - 1.0) > xs_s.worst) { xs_s.worst = std::fabs(r - 1.0); xs_s.at = e; }
      }
    }

    bool want = false;
    for (double p : probe) { if (std::fabs(e - p) / p < 0.02) { want = true; } }
    if (!want || pos) { continue; }
    std::printf("  %-8s %-3s %9.4g %13.6g %13.6g %8.4f %13.6g %13.6g %8.4f\n", f[0].c_str() + 3,
                f[1].c_str(), e, our_dedx, g4_dedx, (g4_dedx > 0 ? our_dedx / g4_dedx : 0),
                our_xs, g4_xs, (g4_xs > 0 ? our_xs / g4_xs : 0));
  }

  std::printf("\n  worst over 0.1 - 100 MeV, e- and e+, water/tissue/bone:\n");
  std::printf("    restricted brems dE/dx  %8.3f%%  (n=%d, at %.4g MeV)\n", 100 * dedx_s.worst,
              dedx_s.n, dedx_s.at);
  std::printf("    brems photon XS         %8.3f%%  (n=%d, at %.4g MeV)\n", 100 * xs_s.worst,
              xs_s.n, xs_s.at);
  return 0;
}
