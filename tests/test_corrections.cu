// The individual Bethe-Bloch correction terms, against G4EmCorrections directly.
//
// Diffing the assembled dE/dx only says that something is off. These say which term, which
// is what let the water density-effect error be identified rather than guessed at.
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>
#include "core/particle.cuh"
#include "data/materials.cuh"
#include "physics/em/em_corrections.cuh"

using namespace g4gpu;
using real_t = double;

static ParticleType type_of(const std::string& n) {
  if (n == "mu-") { return ParticleType::kMuonMinus; }
  if (n == "proton") { return ParticleType::kProton; }
  if (n == "anti_proton") { return ParticleType::kAntiProton; }
  if (n == "alpha") { return ParticleType::kAlpha; }
  if (n == "He3") { return ParticleType::kHe3; }
  return ParticleType::kNumTypes;
}

int main() {
  data::Material<real_t> mats[data::kNumMaterials];
  data::build_b1_materials<real_t>(mats);
  const char* g4names[4] = {"G4_AIR", "G4_WATER", "G4_A-150_TISSUE", "G4_BONE_COMPACT_ICRU"};
  static em::ShellTables<real_t> st;
  em::build_shell_tables(st);

  const char* env = std::getenv("G4GPU_ORACLE");
  const std::string dir = (env != nullptr) ? env : "ref/oracle";
  FILE* f = std::fopen((dir + "/corrections.csv").c_str(), "r");
  if (f == nullptr) {
    std::printf("cannot read %s/corrections.csv\n", dir.c_str());
    return 1;
  }
  char line[512];
  if (std::fgets(line, sizeof line, f) == nullptr) { std::fclose(f); return 1; }

  struct Acc { double worst; double at; };
  Acc shell{0, 0}, barkas{0, 0}, bloch{0, 0}, mott{0, 0}, effq{0, 0};
  std::string w_shell, w_barkas, w_bloch, w_mott, w_effq;
  int n = 0;
  auto track = [](Acc& a, std::string& where, double ours, double g4, const char* p,
                  const char* m, double e) {
    // Relative where the term is meaningful, absolute where it is near zero.
    const double denom = std::fabs(g4);
    const double dev = (denom > 1e-6) ? std::fabs(ours / g4 - 1) : std::fabs(ours - g4);
    if (dev > a.worst) {
      a.worst = dev;
      a.at = e;
      char buf[160];
      std::snprintf(buf, sizeof buf, "%s in %s at %.4g MeV", p, m, e);
      where = buf;
    }
  };

  while (std::fgets(line, sizeof line, f) != nullptr) {
    char mat[128], part[32];
    double e, sh, ba, bl, mo, ho, ib, eq;
    if (std::sscanf(line, "%127[^,],%31[^,],%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf", mat, part, &e,
                    &sh, &ba, &bl, &mo, &ho, &ib, &eq) != 10) {
      continue;
    }
    const ParticleType t = type_of(part);
    if (t == ParticleType::kNumTypes) { continue; }
    int mi = -1;
    for (int i = 0; i < data::kNumMaterials; ++i) {
      if (std::string(mat) == g4names[i]) { mi = i; break; }
    }
    if (mi < 0) { continue; }
    const auto pd = particle_def<real_t>(t);
    // Geant4 substitutes the effective charge for |q| > 1.5 before evaluating the terms.
    auto cpd = pd;
    if (std::fabs(pd.charge) > 1.5) {
      cpd.charge = em::ion_effective_charge(mats[mi], pd, real_t(e));
    }
    ++n;
    track(shell, w_shell, em::shell_correction(st, mats[mi], pd, real_t(e)), sh, part, mat, e);
    track(barkas, w_barkas, em::barkas_correction(mats[mi], cpd, real_t(e)), ba, part, mat, e);
    track(bloch, w_bloch, em::bloch_correction(cpd, real_t(e)), bl, part, mat, e);
    track(mott, w_mott, em::mott_correction(cpd, real_t(e)), mo, part, mat, e);
    if (eq > 0) {
      track(effq, w_effq, cpd.charge * cpd.charge, eq, part, mat, e);
    }
  }
  std::fclose(f);

  printf("== Bethe-Bloch correction terms vs G4EmCorrections ==\n");
  printf("  %d rows compared\n\n", n);
  printf("  %-24s %12s   %s\n", "term", "worst dev", "where");
  printf("  %-24s %11.4f%%   %s\n", "shell", 100 * shell.worst, w_shell.c_str());
  printf("  %-24s %11.4f%%   %s\n", "Barkas", 100 * barkas.worst, w_barkas.c_str());
  printf("  %-24s %11.4f%%   %s\n", "Bloch", 100 * bloch.worst, w_bloch.c_str());
  printf("  %-24s %11.4f%%   %s\n", "Mott", 100 * mott.worst, w_mott.c_str());
  printf("  %-24s %11.4f%%   %s\n", "effective charge^2", 100 * effq.worst, w_effq.c_str());

  int fails = 0;
  if (n == 0) { printf("\n  FAIL: nothing compared\n"); ++fails; }
  // Shell and Barkas are table interpolations and should be near exact; the effective
  // charge covers only the Zi <= 2 branch, which is what alpha and He3 need.
  if (shell.worst > 0.01) { printf("\n  FAIL: shell correction exceeds 1%%\n"); ++fails; }
  if (bloch.worst > 0.001) { printf("  FAIL: Bloch correction exceeds 0.1%%\n"); ++fails; }
  if (mott.worst > 0.001) { printf("  FAIL: Mott correction exceeds 0.1%%\n"); ++fails; }
  printf("\n%s (%d failures)\n", fails ? "FAILED" : "PASSED", fails);
  return fails ? 1 : 0;
}
