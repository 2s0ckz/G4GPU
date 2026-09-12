// Diffs the GPU port against Geant4's own numbers, dumped by ref/dump/g4dump.cc.
//
// This is the real validation: not NIST, not an analytic formula I chose, but the values
// the installed Geant4 11.1.1 actually computes for B1's materials and physics list.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <map>

#include "data/materials.cuh"
#include "physics/em/electron_processes.cuh"
#include "physics/em/gamma_processes.cuh"

using namespace g4gpu;
using real_t = double;

static const char* kOracle = "D:\\g4gpu\\ref\\oracle\\";

static std::vector<std::vector<std::string>> read_csv(const std::string& name) {
  std::vector<std::vector<std::string>> rows;
  std::ifstream in(std::string(kOracle) + name);
  if (!in) {
    std::printf("  CANNOT OPEN %s%s\n", kOracle, name.c_str());
    return rows;
  }
  std::string line;
  bool header = true;
  while (std::getline(in, line)) {
    if (header) { header = false; continue; }
    if (line.empty()) { continue; }
    std::vector<std::string> f;
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ',')) { f.push_back(cell); }
    rows.push_back(f);
  }
  return rows;
}

/// Maps Geant4 material names onto our MaterialId ordering.
static int mat_index(const std::string& name) {
  if (name == "G4_AIR") { return data::kAir; }
  if (name == "G4_WATER") { return data::kWater; }
  if (name == "G4_A-150_TISSUE") { return data::kA150Tissue; }
  if (name == "G4_BONE_COMPACT_ICRU") { return data::kBoneCompact; }
  return -1;
}

struct Stat {
  double worst = 0;
  double worst_e = 0;
  int n = 0;
  void add(double ratio, double e) {
    ++n;
    const double d = std::fabs(ratio - 1.0);
    if (d > worst) { worst = d; worst_e = e; }
  }
};

int main() {
  data::Material<real_t> mats[data::kNumMaterials];
  data::build_b1_materials<real_t>(mats);
  const char* names[4] = {"air", "water", "A-150 tissue", "bone"};

  // ---------------------------------------------------------------- material parameters
  std::printf("== material parameters vs Geant4 ==\n");
  std::printf("  %-14s %-22s %12s %12s %8s\n", "material", "quantity", "ours", "Geant4", "ratio");
  for (const auto& r : read_csv("materials.csv")) {
    const int m = mat_index(r[0]);
    if (m < 0) { continue; }
    const double g4_ne = std::atof(r[2].c_str());
    const double g4_zeff = std::atof(r[3].c_str());
    const double g4_iexc = std::atof(r[4].c_str());
    const double g4_x0 = std::atof(r[5].c_str());
    std::printf("  %-14s %-22s %12.6g %12.6g %8.4f\n", names[m], "electron density /mm3",
                mats[m].electron_density, g4_ne, mats[m].electron_density / g4_ne);
    std::printf("  %-14s %-22s %12.6g %12.6g %8.4f\n", "", "Zeff", mats[m].z_eff, g4_zeff,
                mats[m].z_eff / g4_zeff);
    std::printf("  %-14s %-22s %12.6g %12.6g %8.4f\n", "", "mean excitation eV",
                mats[m].mean_excitation * 1e6, g4_iexc, mats[m].mean_excitation * 1e6 / g4_iexc);
    std::printf("  %-14s %-22s %12.6g %12.6g %8.4f\n", "", "radiation length mm",
                mats[m].radiation_length, g4_x0, mats[m].radiation_length / g4_x0);
  }

  // ---------------------------------------------------------------- gamma cross sections
  std::printf("\n== gamma cross sections vs Geant4, 1/mm ==\n");
  Stat compt[4], pair_[4];
  double phot_frac_max[4] = {0, 0, 0, 0}, rayl_frac_max[4] = {0, 0, 0, 0};
  double phot_frac_6[4] = {0, 0, 0, 0}, rayl_frac_6[4] = {0, 0, 0, 0};
  for (const auto& r : read_csv("gamma_xs.csv")) {
    const int m = mat_index(r[0]);
    if (m < 0) { continue; }
    const double e = std::atof(r[1].c_str());
    const double g_phot = std::atof(r[2].c_str());
    const double g_compt = std::atof(r[3].c_str());
    const double g_conv = std::atof(r[4].c_str());
    const double g_rayl = std::atof(r[5].c_str());
    const double total = g_phot + g_compt + g_conv + g_rayl;

    // Ours, same quantities.
    double our_compt = 0, our_pair = 0;
    for (int i = 0; i < mats[m].n_elements; ++i) {
      our_compt += mats[m].n_atoms[i] * em::compton_xs_per_atom<real_t>(e, mats[m].z[i]);
      our_pair += mats[m].n_atoms[i] * em::pair_xs_per_atom<real_t>(e, mats[m].z[i]);
    }
    if (g_compt > 0 && e >= 0.01 && e <= 100) { compt[m].add(our_compt / g_compt, e); }
    if (g_conv > 0 && e >= 2.0 && e <= 100) { pair_[m].add(our_pair / g_conv, e); }
    if (total > 0) {
      const double pf = g_phot / total, rf = g_rayl / total;
      if (e >= 0.01 && pf > phot_frac_max[m]) { phot_frac_max[m] = pf; }
      if (e >= 0.01 && rf > rayl_frac_max[m]) { rayl_frac_max[m] = rf; }
      if (std::fabs(e - 6.0) / 6.0 < 0.03) { phot_frac_6[m] = pf; rayl_frac_6[m] = rf; }
    }
  }
  std::printf("  %-14s %-28s %10s %12s\n", "material", "quantity", "worst dev", "at E (MeV)");
  for (int m = 0; m < 4; ++m) {
    if (compt[m].n == 0) { continue; }
    std::printf("  %-14s %-28s %9.2f%% %12.4g\n", names[m], "Compton (G4KleinNishinaCompton)",
                100 * compt[m].worst, compt[m].worst_e);
    std::printf("  %-14s %-28s %9.2f%% %12.4g\n", "", "pair: ours(BH) vs G4(PairProdRel)",
                100 * pair_[m].worst, pair_[m].worst_e);
  }
  std::printf("\n  processes we do not implement, as a fraction of total attenuation:\n");
  std::printf("  %-14s %14s %14s %14s %14s\n", "material", "phot max", "phot @6MeV", "Rayl max",
              "Rayl @6MeV");
  for (int m = 0; m < 4; ++m) {
    std::printf("  %-14s %13.2f%% %13.4f%% %13.2f%% %13.4f%%\n", names[m],
                100 * phot_frac_max[m], 100 * phot_frac_6[m], 100 * rayl_frac_max[m],
                100 * rayl_frac_6[m]);
  }

  // ---------------------------------------------------------------- electron tables
  std::printf("\n== electron dE/dx and range vs Geant4 ==\n");
  std::printf("  NOTE: Geant4 eIoni dE/dx is RESTRICTED to the production cut - energy above\n");
  std::printf("  the cut leaves as delta rays. Ours is unrestricted, so ours should read high.\n");
  // No range table is built here: every comparison in this block calls the MODELS
  // (`collision_dedx`, `delta_ray_xs`) at a point, and the table it used to build was never
  // read. That distinction is the whole of docs/RISK.md V64 - an oracle that compares a model
  // function at an energy is right for any energy the caller passes, and the ceiling was in
  // the table the transport reads. `tests/test_electron_hi.cu` compares the TABLE.
  std::printf("\n  %-8s %-3s %9s %11s %11s %7s %11s %11s %7s %11s %11s %7s\n", "material",
              "p", "E (MeV)", "restr ours", "restr G4", "ratio", "unrest ours", "unrest G4",
              "ratio", "deltaXS ours", "deltaXS G4", "ratio");
  const double probe[5] = {0.1, 0.5, 1.0, 3.0, 6.0};
  Stat restr, unrestr, dxs;
  for (const auto& r : read_csv("electron_tables.csv")) {
    const int m = mat_index(r[0]);
    if (m < 0 || m == data::kAir) { continue; }
    const bool is_pos = (r[1] == "e+");
    const double e = std::atof(r[2].c_str());
    const double cut = std::atof(r[3].c_str());
    const double g_restr = std::atof(r[4].c_str());
    const double g_unrestr = std::atof(r[5].c_str());
    const double g_dxs = std::atof(r[10].c_str());

    const double o_restr = em::collision_dedx<real_t>(mats[m], e, is_pos, cut);
    const double o_unrestr = em::collision_dedx<real_t>(mats[m], e, is_pos, 1e9);
    const double o_dxs = em::delta_ray_xs<real_t>(mats[m], e, is_pos, cut);

    // Accumulate over the whole grid where the quantity is meaningful.
    if (e >= 0.01 && e <= 100) {
      if (g_restr > 0) { restr.add(o_restr / g_restr, e); }
      if (g_unrestr > 0) { unrestr.add(o_unrestr / g_unrestr, e); }
      if (g_dxs > 1e-12) { dxs.add(o_dxs / g_dxs, e); }
    }

    bool want = false;
    for (double p : probe) { if (std::fabs(e - p) / p < 0.015) { want = true; } }
    if (!want || is_pos) { continue; }
    std::printf("  %-8s %-3s %9.4g %11.6g %11.6g %7.4f %11.6g %11.6g %7.4f %11.6g %11.6g %7.4f\n",
                names[m], r[1].c_str(), e, o_restr, g_restr,
                (g_restr > 0 ? o_restr / g_restr : 0), o_unrestr, g_unrestr,
                (g_unrestr > 0 ? o_unrestr / g_unrestr : 0), o_dxs, g_dxs,
                (g_dxs > 0 ? o_dxs / g_dxs : 0));
  }
  std::printf("\n  worst deviation over the whole 10 keV - 100 MeV grid, e- and e+:\n");
  std::printf("    restricted dE/dx   %8.4f%%  (n=%d, worst at %.4g MeV)\n", 100 * restr.worst,
              restr.n, restr.worst_e);
  std::printf("    unrestricted dE/dx %8.4f%%  (n=%d, worst at %.4g MeV)\n",
              100 * unrestr.worst, unrestr.n, unrestr.worst_e);
  std::printf("    delta-ray XS       %8.4f%%  (n=%d, worst at %.4g MeV)\n", 100 * dxs.worst,
              dxs.n, dxs.worst_e);

  return 0;
}
