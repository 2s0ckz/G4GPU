// Barashenkov nucleon-nucleus cross sections against Geant4's own.
//
// ref/oracle/nucleon_xs.csv is G4NucleonNuclearCrossSection - the wrapper around
// G4ComponentBarNucleonNucleusXsc - for every Z from 2 to 92 at twelve points per decade from
// 14 MeV to 1 TeV, for protons and for neutrons. That component is what both
// G4BGGNucleonElasticXS and G4BGGNucleonInelasticXS delegate to over that whole band, so this
// is the hadronic cross section for any nucleon anything is realistically transported at.
//
// Every Z, not the five in example B1's materials, and the reason is structural: seventeen Z
// values are tabulated in Geant4 and the other seventy-four are an interpolation in A between
// the two that bracket them. Checking only tabulated elements would test the table lookup and
// leave the interpolation - the half with an A^(2/3) scaling and two atomic masses in it, and
// so the half more likely to be transcribed wrongly - completely unexercised.
//
// Three columns are compared and not one. Elastic is total minus inelastic on both sides, so a
// sign error, a swapped proton/neutron column, or a Z-interpolation applied to one and not the
// other can cancel in the difference and show up in only one of the three.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "physics/hadronic/barashenkov_xs.cuh"

using namespace g4gpu;
using real_t = double;

namespace {

struct Row {
  int z = 0;
  double e = 0, total = 0, inelastic = 0, elastic = 0;
  bool is_proton = true;
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
    char pname[32];
    const int n = std::sscanf(line, "%d,%lf,%lf,%lf,%lf,%31s", &r.z, &r.e, &r.total,
                              &r.inelastic, &r.elastic, pname);
    if (n != 6) { continue; }
    r.is_proton = (pname[0] == 'p');
    out.push_back(r);
  }
  std::fclose(f);
  return out;
}

struct Cell {
  int n = 0;
  double worst = 0;
  std::string where;
};

void note(Cell& c, double dev, const char* what) {
  ++c.n;
  if (dev > c.worst) {
    c.worst = dev;
    c.where = what;
  }
}

}  // namespace

int main() {
  const char* env = std::getenv("G4GPU_ORACLE");
  const std::string dir = (env != nullptr) ? env : "ref/oracle";
  const auto rows = load(dir + "/nucleon_xs.csv");
  if (rows.empty()) {
    std::printf("cannot read %s/nucleon_xs.csv - run ref/oracle/run.bat first\n", dir.c_str());
    return 1;
  }

  int fails = 0;

  // Split by whether Geant4 has this element tabulated or is interpolating to it, because the
  // two are different code and a single worst-case would report whichever happened to be
  // worse rather than telling you which one is wrong.
  const int tabulated[] = {2, 4, 6, 7, 8, 11, 13, 14, 20, 26, 29, 42, 48, 50, 74, 82, 92};
  auto is_tabulated = [&](int z) {
    for (int t : tabulated) {
      if (t == z) { return true; }
    }
    return false;
  };

  Cell tot[2], inel[2], elas[2];  // [0] tabulated, [1] interpolated
  int compared = 0;
  int zmin = 999, zmax = 0;

  for (const Row& r : rows) {
    const auto x = hadronic::barashenkov_xs<real_t>(r.z, real_t(r.e), r.is_proton);
    const int k = is_tabulated(r.z) ? 0 : 1;
    ++compared;
    if (r.z < zmin) { zmin = r.z; }
    if (r.z > zmax) { zmax = r.z; }

    char buf[200];
    // Relative deviation where the reference is non-zero. The inelastic cross section is
    // genuinely zero at the bottom of several elements' grids - a 14 MeV proton on helium
    // cannot react - so an absolute floor is needed or those rows divide by zero.
    auto cmp = [&](Cell& c, double ours, double g4, const char* what) {
      const double scale = (g4 > 1e-30) ? g4 : 1e-30;
      const double dev = (g4 > 1e-30) ? std::fabs(ours - g4) / scale
                                      : (std::fabs(ours) > 1e-30 ? 1.0 : 0.0);
      std::snprintf(buf, sizeof buf, "%s Z=%d at %.4g MeV (%s: ours %.6g, G4 %.6g mm^2)",
                    r.is_proton ? "p" : "n", r.z, r.e, what, ours, g4);
      note(c, dev, buf);
    };
    cmp(tot[k], x.total, r.total, "total");
    cmp(inel[k], x.inelastic, r.inelastic, "inelastic");
    cmp(elas[k], x.elastic, r.elastic, "elastic");
  }

  std::printf("== Barashenkov nucleon-nucleus cross sections vs G4NucleonNuclearCrossSection ==\n");
  std::printf("  %d points, Z from %d to %d\n\n", compared, zmin, zmax);
  const char* kind[2] = {"tabulated Z", "interpolated Z"};
  Cell* cells[3][2] = {{&tot[0], &tot[1]}, {&inel[0], &inel[1]}, {&elas[0], &elas[1]}};
  const char* names[3] = {"total", "inelastic", "elastic"};

  for (int q = 0; q < 3; ++q) {
    for (int k = 0; k < 2; ++k) {
      Cell& c = *cells[q][k];
      std::printf("  %-10s %-15s %6d points  worst %10.3e  %s\n", names[q], kind[k], c.n,
                  c.worst, c.where.c_str());
      if (c.n == 0) {
        std::printf("  FAIL: no %s / %s points compared\n", names[q], kind[k]);
        ++fails;
      }
      // 1e-12, and it is a rounding limit rather than a physics one. There is nothing between
      // this port's answer and Geant4's but double-precision arithmetic in the same order over
      // the same table, so they agree to 2e-15 - the limit is a thousand times that, which
      // leaves room for a compiler reassociating a sum and none at all for a wrong table entry,
      // a wrong energy grid, or the A^(2/3) scaling applied to the wrong pair.
      //
      // It was 1e-9 first, and everything failed at 5e-9: the oracle was being written with
      // %.9g, so the reference's own write precision was the entire disagreement. The dump is
      // %.17g for this file now. A tolerance that had been left at 1e-9 would have passed and
      // been measuring the CSV formatter.
      if (c.worst > 1e-12) {
        std::printf("  FAIL: %s at %s is off by %.3e, limit 1e-12\n", names[q], kind[k],
                    c.worst);
        ++fails;
      }
    }
  }

  // Both element classes must actually be present. If the oracle were ever dumped for only the
  // tabulated Z values, every check above would pass while testing half the code.
  if (tot[0].n == 0 || tot[1].n == 0) {
    std::printf("  FAIL: the oracle covers only one of tabulated / interpolated Z\n");
    ++fails;
  }

  std::printf("\n%s (%d failures)\n", fails ? "FAILED" : "PASSED", fails);
  return fails ? 1 : 0;
}
