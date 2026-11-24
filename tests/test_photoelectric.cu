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

static std::vector<std::string> split_csv(const std::string& line) {
  std::vector<std::string> f;
  std::stringstream ss(line);
  std::string cell;
  while (std::getline(ss, cell, ',')) { f.push_back(cell); }
  return f;
}

/// Replays the photoelectric shell walk with scripted deviates: r0 first, then 0.5 for ever -
/// the scripted engine ref/dump/dump_pe_shells.cc drives Geant4's SampleSecondaries with.
struct ScriptRng {
  double r0;
  int n = 0;
  __host__ __device__ double uniform() { return (n++ == 0) ? r0 : 0.5; }
};

/// P43 (docs/RISK.md V356): every element, not B1's ten. The whole of Z = 1 .. 100 is loaded
/// into ONE table - the scene no geometry can exceed since V355 - and held to Geant4's own
/// models: the cross section per atom at 441 energies a Z (pe_atom.csv), and the shell the walk
/// selects, deviate for deviate, for 20 elements from H to U at 25 energies and 19 deviates
/// (pe_shell.csv). Before V355 the load itself failed at Z = 37, the first element with more
/// than 12 shells.
///
/// The limits: the cross section to 1e-12 relative, because the port and Geant4 evaluate the
/// same polynomial in the same order, and interpolate the same points by two algebraically equal
/// formulas (G4PhysicsVector::Interpolation's y1 + b*dy against this port's y1*(1-f) + y2*f),
/// which can differ in the last bits only; the shell walk EXACTLY, because the selected shell's
/// binding energy is one number read from one file and the photoelectron's energy one
/// subtraction.
static int z_scan() {
  std::printf("\n== every element, Z = 1 .. 100 (P43, docs/RISK.md V356) ==\n");
  std::printf("  table capacity: kPeMaxShells %d, kPeMaxElements %d; %zu B an element, %zu B the "
              "table\n", data::kPeMaxShells, data::kPeMaxElements,
              sizeof(data::PhotoElectricElement<real_t>), sizeof(data::PhotoElectricTable<real_t>));
  int all[100];
  for (int z = 1; z <= 100; ++z) { all[z - 1] = z; }
  static data::PhotoElectricTable<real_t> t{};
  std::vector<real_t> te, tv;
  if (!data::load_photoelectric<real_t>(kPhotDir, all, 100, t, te, tv)) {
    std::printf("FAILED: the table of Z = 1 .. 100 did not load (the line above says why)\n");
    return 1;
  }
  t.table_e = te.data();
  t.table_v = tv.data();
  t.table_n = static_cast<int>(te.size());
  int max_shells = 0, max_z = 0;
  for (int i = 0; i < t.n_elements; ++i) {
    if (t.elements[i].n_shells > max_shells) {
      max_shells = t.elements[i].n_shells;
      max_z = t.elements[i].z;
    }
  }
  std::printf("  loaded %d elements, %d tabulated points; the most shells: %d (Z = %d)\n",
              t.n_elements, t.table_n, max_shells, max_z);

  const char* env = std::getenv("G4GPU_ORACLE");
  const std::string dir = (env != nullptr) ? env : "ref/oracle";
  int failures = 0;

  // ---- the cross section per atom.
  {
    std::ifstream in(dir + "/pe_atom.csv");
    if (!in) {
      std::printf("FAILED: cannot read %s/pe_atom.csv - ref/dump's pe_shells dump writes it\n",
                  dir.c_str());
      return 1;
    }
    std::string line;
    std::getline(in, line);
    double worst[101] = {0};
    double worst_e[101] = {0};
    int n[101] = {0};
    long rows = 0, bad = 0, edges = 0;
    while (std::getline(in, line)) {
      const std::vector<std::string> f = split_csv(line);
      if (f.size() < 4) { continue; }
      const int z = std::atoi(f[0].c_str());
      const double e = std::atof(f[1].c_str());
      const double g4 = std::atof(f[2].c_str());
      if (z < 1 || z > 100) { continue; }
      const double ours = data::photoelectric_xs_per_atom<real_t>(t, z, e);
      const double rel = (g4 != 0) ? std::fabs(ours / g4 - 1.0) : std::fabs(ours);
      ++rows;
      // AN ABSORPTION EDGE HIT EXACTLY. The tabulated files give an edge as two points at one
      // energy (pe-le-cs-55.dat: 1.000000000e-03 twice, Cs's M3 edge, 1.8216e-3 then 2.0865e-3
      // barn). At that energy and only there, G4PhysicsVector::Value returns the value BELOW the
      // edge (its bin search stops at the first node) and interp_block the value ABOVE it (it
      // stops at the last). The grid of 441 energies lands on one such node: Cs at 1 keV. It is
      // measure zero - a transported photon's energy is never a tabulated node - and it is not
      // changed, because this package changes capacities and not arithmetic (V355); it is
      // counted here, by name, and not as agreement. docs/RISK.md V356.
      if (rel > 1e-12) {
        const data::PhotoElectricElement<real_t>& el = t.elements[t.z_to_index[z]];
        const int offs[2] = {el.cs_offset, el.le_offset};
        const int cnts[2] = {el.cs_count, el.le_count};
        bool on_edge = false;
        for (int b = 0; b < 2; ++b) {
          for (int k = 0; k + 1 < cnts[b]; ++k) {
            on_edge |= (te[offs[b] + k] == e && te[offs[b] + k + 1] == e);
          }
        }
        if (on_edge) {
          ++edges;
          std::printf("  AT AN EDGE NODE Z %d, %.10g MeV: ours %.17g (above the edge), Geant4 "
                      "%.17g (below it)\n", z, e, ours, g4);
          continue;
        }
      }
      ++n[z];
      if (rel > worst[z]) { worst[z] = rel; worst_e[z] = e; }
      if (rel > 1e-12) {
        if (++bad <= 10) {
          std::printf("  MISMATCH Z %d at %.6g MeV: ours %.17g, Geant4 %.17g (rel %.3g)\n", z, e,
                      ours, g4, rel);
        }
      }
    }
    double w = 0;
    int wz = 0;
    for (int z = 1; z <= 100; ++z) {
      if (worst[z] > w) { w = worst[z]; wz = z; }
    }
    std::printf("  cross section per atom, %ld points: worst %.3g relative (Z = %d at %.4g MeV) "
                "off the edge nodes; %ld at an edge node\n", rows, w, wz, worst_e[wz], edges);
    const int named[] = {26, 29, 47, 74, 82, 92};
    const char* sym[] = {"Fe", "Cu", "Ag", "W", "Pb", "U"};
    for (int k = 0; k < 6; ++k) {
      const int z = named[k];
      const int i = t.z_to_index[z];
      std::printf("    %-2s (Z = %2d, %2d shells): %3d energies, worst %.3g\n", sym[k], z,
                  t.elements[i].n_shells, n[z], worst[z]);
    }
    if (rows < 98 * 441 || bad > 0) {
      std::printf("FAILED: %ld of %ld points outside 1e-12 (expected %d points)\n", bad, rows,
                  98 * 441);
      ++failures;
    }
  }

  // ---- the shell walk, deviate for deviate.
  {
    std::ifstream in(dir + "/pe_shell.csv");
    if (!in) {
      std::printf("FAILED: cannot read %s/pe_shell.csv - ref/dump's pe_shells dump writes it\n",
                  dir.c_str());
      return 1;
    }
    std::string line;
    std::getline(in, line);
    long rows = 0, bad = 0, absorbed = 0;
    int shells_seen[101][data::kPeMaxShells] = {{0}};
    while (std::getline(in, line)) {
      const std::vector<std::string> f = split_csv(line);
      if (f.size() < 6) { continue; }
      const int z = std::atoi(f[0].c_str());
      const double e = std::atof(f[1].c_str());
      const double r0 = std::atof(f[2].c_str());
      const bool g4_produced = std::atoi(f[3].c_str()) != 0;
      const double g4_te = std::atof(f[4].c_str());
      ScriptRng rng{r0};
      const data::PhotoElectricResult<real_t> r = data::sample_photoelectric<real_t>(
          t, z, e, Vec3<real_t>{0, 0, 1}, rng);
      ++rows;
      if (!g4_produced) { ++absorbed; }
      const bool same = (r.electron_produced == g4_produced)
                        && (!g4_produced || r.electron_ekin == g4_te);
      if (!same) {
        if (++bad <= 10) {
          std::printf("  MISMATCH Z %d at %.6g MeV, r0 %.6g: ours %s %.17g MeV, Geant4 %s %.17g "
                      "MeV\n", z, e, r0, r.electron_produced ? "e-" : "absorbed",
                      r.electron_ekin, g4_produced ? "e-" : "absorbed", g4_te);
        }
      } else if (g4_produced && z >= 1 && z <= 100) {
        // Which shell that was, for the line below: the binding energy names it.
        const data::PhotoElectricElement<real_t>& el = t.elements[t.z_to_index[z]];
        for (int s = 0; s < el.n_shells; ++s) {
          if (el.param_high[s * 7 + 1] == r.local_deposit) { ++shells_seen[z][s]; break; }
        }
      }
    }
    int distinct_pb = 0, distinct_u = 0;
    for (int s = 0; s < data::kPeMaxShells; ++s) {
      distinct_pb += (shells_seen[82][s] > 0);
      distinct_u += (shells_seen[92][s] > 0);
    }
    std::printf("  shell walk, %ld (Z, E, r0) against SampleSecondaries: %ld differ; %ld absorbed "
                "whole on both sides; Pb selected %d distinct shells of 24, U %d of 29\n",
                rows, bad, absorbed, distinct_pb, distinct_u);
    if (rows < 20 * 25 * 19 || bad > 0) {
      std::printf("FAILED: the shell walk differs from Geant4's on %ld of %ld\n", bad, rows);
      ++failures;
    }
  }
  if (failures == 0) { std::printf("  every element agrees\n"); }
  return failures == 0 ? 0 : 1;
}

int main() {
  data::Material<real_t> mats[data::kNumMaterials];
  data::build_b1_materials<real_t>(mats);

  // Every element appearing in the four B1 materials.
  const int zs[10] = {1, 6, 7, 8, 9, 12, 15, 16, 18, 20};
  static data::PhotoElectricTable<real_t> pe{};  // 364 KB since V355: static, not on the stack
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
  return z_scan();
}
