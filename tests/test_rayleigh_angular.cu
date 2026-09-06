// G4RayleighAngularGenerator::SampleDirection against Geant4's own.
//
// WHY THIS EXISTS
//
// `src/data/rayleigh_angular_tables.cuh` was, until this test, the one baked table in the port
// with neither an extractor nor a test - the only one a Geant4 change could alter silently.
// It is a poor thing to leave unguarded precisely because of how quietly it fails: Rayleigh
// scattering transfers no energy, it only turns a photon, so a wrong table moves a dose by a
// fraction of a per cent and breaks nothing loudly. `tools/refresh_tables.sh` named it for
// that reason. `tools/extract_rayleigh_angular.sh` closes the reproducibility half - and when
// it was first run it reproduced all 909 hand-transcribed parameters exactly, which is the
// strongest thing that could have been said about the transcription. This closes the other
// half: whether the *sampler* built on those parameters is Geant4's.
//
// WHY MOMENTS AND NOT VALUES
//
// SampleDirection is a rejection loop drawing three uniforms per iteration and one more for
// phi. There is no closed form to diff, and reproducing Geant4's random sequence would mean
// installing a CLHEP engine in the dumper and replaying it here - possible, and more machinery
// than the question needs.
//
// So `ref/oracle/rayleigh_angular.csv` carries the first two moments of cos(theta) over a
// million Geant4 draws per point, and this draws a million of its own and compares. That is a
// statistical test and it is stated as one: it cannot see an error smaller than a few standard
// errors. What it can see is every error worth having - a wrong parameter row, a weight and a
// slope transposed, a series expansion mis-transcribed, an acceptance test with the wrong
// power - because each of those moves a mean or a width by percent, and the standard error of
// a million draws is 1e-3.
//
// TWO moments and not one. The distribution is forward-peaked, so <cos> alone is blind to a
// symmetric broadening: an error that widens the distribution while leaving its centre put
// passes on the mean and fails on the second moment.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "core/rng.cuh"
#include "core/vec3.cuh"
#include "data/rayleigh_data.cuh"

using namespace g4gpu;
using real_t = double;

namespace {

struct Row {
  int z = 0;
  double e = 0, mean_cos = 0, mean_cos2 = 0;
  long long n = 0;
};

std::vector<Row> load(const std::string& path) {
  std::vector<Row> out;
  FILE* f = std::fopen(path.c_str(), "r");
  if (f == nullptr) { return out; }
  char line[512];
  if (std::fgets(line, sizeof line, f) == nullptr) {
    std::fclose(f);
    return out;
  }
  while (std::fgets(line, sizeof line, f) != nullptr) {
    Row r;
    if (std::sscanf(line, "%d,%lf,%lld,%lf,%lf", &r.z, &r.e, &r.n, &r.mean_cos,
                    &r.mean_cos2) != 5) {
      continue;
    }
    out.push_back(r);
  }
  std::fclose(f);
  return out;
}

}  // namespace

int main() {
  const char* env = std::getenv("G4GPU_ORACLE");
  const std::string dir = (env != nullptr) ? env : "ref/oracle";
  const auto rows = load(dir + "/rayleigh_angular.csv");
  if (rows.empty()) {
    std::printf("cannot read %s/rayleigh_angular.csv - run ref/oracle/run.bat first\n",
                dir.c_str());
    return 1;
  }

  int fails = 0;
  const int kN = 1000000;
  double worst_sigma = 0;
  std::string worst_where;

  std::printf("== G4RayleighAngularGenerator::SampleDirection ==\n");
  std::printf("  %d points, %d draws each side\n\n", static_cast<int>(rows.size()), kN);
  std::printf("  %3s %10s  %11s %11s %8s   %11s %11s %8s\n", "Z", "E(MeV)", "<cos> ours",
              "<cos> G4", "sigma", "<cos2> ours", "<cos2> G4", "sigma");

  for (const Row& r : rows) {
    Philox<real_t> rng(static_cast<uint32_t>(r.z * 7919 + 13),
                       static_cast<uint32_t>(r.e * 100000), 5u);
    const Vec3<real_t> incident{0, 0, 1};
    double s1 = 0, s2 = 0, s4 = 0;
    for (int i = 0; i < kN; ++i) {
      const Vec3<real_t> d = data::sample_rayleigh_direction(r.z, real_t(r.e), incident, rng);
      const double c = d.z;  // incident is +z, so z is cos(theta)
      s1 += c;
      s2 += c * c;
      s4 += c * c * c * c;
    }
    const double m1 = s1 / kN, m2 = s2 / kN, m4 = s4 / kN;

    // The gate is the combined standard error of the two independent estimates, from the
    // measured variances rather than a bound: sqrt(var/N + var/N). Geant4's variance is
    // recovered from its own two moments; for the second moment the fourth is needed and only
    // this side has it, so that one uses ours twice - the two distributions are the same
    // distribution if the test passes, which makes it self-consistent rather than circular.
    const double var1 = std::fmax(m2 - m1 * m1, 0.0);
    const double var1_g4 = std::fmax(r.mean_cos2 - r.mean_cos * r.mean_cos, 0.0);
    const double se1 = std::sqrt(var1 / kN + var1_g4 / kN);
    const double var2 = std::fmax(m4 - m2 * m2, 0.0);
    const double se2 = std::sqrt(2.0 * var2 / kN);

    const double d1 = std::fabs(m1 - r.mean_cos);
    const double d2 = std::fabs(m2 - r.mean_cos2);
    const double n1 = (se1 > 0) ? d1 / se1 : 0.0;
    const double n2 = (se2 > 0) ? d2 / se2 : 0.0;

    std::printf("  %3d %10.4g  %11.6f %11.6f %7.2fs   %11.6f %11.6f %7.2fs\n", r.z, r.e, m1,
                r.mean_cos, n1, m2, r.mean_cos2, n2);

    const double worst = std::fmax(n1, n2);
    if (worst > worst_sigma) {
      worst_sigma = worst;
      char buf[120];
      std::snprintf(buf, sizeof buf, "Z=%d at %.4g MeV", r.z, r.e);
      worst_where = buf;
    }
  }

  // Six standard errors. With 77 points and two moments each - 154 comparisons - a 5-sigma gate
  // would be tripped by chance about once in three thousand runs; 6 sigma puts that at one in
  // a million while still being a hundred times tighter than any transcription error would be.
  constexpr double kLimit = 6.0;
  std::printf("\n  worst deviation: %.2f standard errors (%s)\n", worst_sigma,
              worst_where.c_str());
  if (worst_sigma > kLimit) {
    std::printf("  FAIL: %.2f sigma exceeds the %.0f sigma limit - this is not sampling noise\n",
                worst_sigma, kLimit);
    ++fails;
  }
  if (rows.size() < 20) {
    std::printf("  FAIL: only %d points in the oracle; expected the full Z and energy grid\n",
                static_cast<int>(rows.size()));
    ++fails;
  }

  std::printf("\n%s (%d failures)\n", fails ? "FAILED" : "PASSED", fails);
  return fails ? 1 : 0;
}
