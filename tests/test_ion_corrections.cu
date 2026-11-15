// P26: `em::ion_corrections_along_step` - `G4VEmModel::CorrectionsAlongStep` under `if(isIon)` -
// against the running Geant4's models (`ref/oracle/ion_corrections.csv`, ref/dump/
// dump_ion_corrections.cc): He3, C12, O16 and Fe56 in B1's four materials, 0.1 to 1000 MeV per
// nucleon, mean losses from 1% to 99% of the pre-step energy with the 5% guard on both sides, each
// handed to the model the ion's `ionIoni` selects at that energy. And the alpha, whose rows must
// come back as they went in - its models return at once, and `isIon` is false for it anyway.
//
// The comparison is to the models' agreement in the effective charge (`ion_tables.csv`'s q2_eff,
// which tests/test_ion_transport.cu already holds), so a few ulps; the guard rows exactly.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "data/materials.cuh"
#include "physics/em/hadron_range.cuh"

using namespace g4gpu;
using real_t = double;

namespace {

int g_fails = 0;
void fail(const std::string& msg) {
  std::printf("  FAIL: %s\n", msg.c_str());
  ++g_fails;
}

std::vector<std::string> split(const std::string& s) {
  std::vector<std::string> out;
  std::string cur;
  for (const char c : s) {
    if (c == ',') {
      out.push_back(cur);
      cur.clear();
    } else if (c != '\n' && c != '\r') {
      cur += c;
    }
  }
  out.push_back(cur);
  return out;
}

}  // namespace

int main() {
  const char* env = std::getenv("G4GPU_ORACLE");
  const std::string dir = (env != nullptr && env[0] != '\0') ? env : "ref/oracle";
  static data::Material<real_t> mats[data::kNumMaterials];
  data::build_b1_materials<real_t>(mats);
  const char* names[data::kNumMaterials] = {"G4_AIR", "G4_WATER", "G4_A-150_TISSUE",
                                            "G4_BONE_COMPACT_ICRU"};
  FILE* f = std::fopen((dir + "/ion_corrections.csv").c_str(), "r");
  if (f == nullptr) {
    std::printf("  FAIL: cannot read %s/ion_corrections.csv - run ref/oracle/run.bat\n",
                dir.c_str());
    return 1;
  }
  static char line[4096];
  long long rows = 0, acted = 0, inert = 0, bad = 0, alpha_rows = 0, alpha_bad = 0;
  double worst = 0, smallest_factor = 1;
  while (std::fgets(line, sizeof line, f) != nullptr) {
    const std::vector<std::string> v = split(line);
    if (v.size() != 8 || v[0] == "material") { continue; }
    int m = -1;
    for (int k = 0; k < data::kNumMaterials; ++k) {
      if (v[0] == names[k]) { m = k; }
    }
    if (m < 0) { continue; }
    const int z = std::atoi(v[2].c_str()), a = std::atoi(v[3].c_str());
    const double e = std::strtod(v[4].c_str(), nullptr);
    const double in = std::strtod(v[5].c_str(), nullptr);
    const double out = std::strtod(v[6].c_str(), nullptr);
    ++rows;
    if (a == 4) {
      // The alpha: `isIon` false, so the stepper never calls the correction; Geant4's models
      // return at once as well. Both sides: unchanged.
      ++alpha_rows;
      if (out != in) { ++alpha_bad; }
      continue;
    }
    const em::SteppedHadron<real_t> h = (a == 3) ? em::stepped_hadron<real_t>(ParticleType::kHe3)
                                                 : em::stepped_ion<real_t>(z, a);
    const bool is_ion = h.is_real_ion() || em::uses_dynamic_effective_charge(h.type);
    if (!is_ion) {
      fail("He3 or a real nucleus is not `isIon` in the stepper's test");
      continue;
    }
    const double mine = em::ion_corrections_along_step<real_t>(mats[m], h, e, in);
    if (out == in) { ++inert; } else { ++acted; }
    if (out != in && out / in < smallest_factor) { smallest_factor = out / in; }
    const double d = std::fabs(mine - out) / std::fabs(out);
    if (d > worst) { worst = d; }
    if (d > 1e-12 || ((out == in) != (mine == in))) {
      if (bad < 10) {
        std::printf("  %s %s %.6g MeV, eloss %.17g: %.17g, Geant4 %.17g\n", names[m],
                    v[1].c_str(), e, in, mine, out);
      }
      ++bad;
    }
  }
  std::fclose(f);
  std::printf("== CorrectionsAlongStep against Geant4's models ==\n");
  std::printf("  %lld rows: %lld corrected, %lld returned unchanged by the guard; worst "
              "relative difference %.2e; the largest correction a factor %.5f; alpha rows %lld, "
              "%lld changed\n", rows, acted, inert, worst, smallest_factor, alpha_rows, alpha_bad);
  if (rows == 0) { fail("no rows"); }
  if (bad != 0) { fail("the correction differs from Geant4's"); }
  if (alpha_bad != 0) { fail("Geant4 corrected an alpha"); }
  if (acted == 0 || inert == 0) {
    fail("the rows must have both corrected and guarded steps for this to test anything");
  }
  std::printf("\n%s: %d failure(s)\n", g_fails ? "FAILED" : "passed", g_fails);
  return g_fails ? 1 : 0;
}
