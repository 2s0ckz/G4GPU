// ICRU Report 90 electronic stopping powers, against Geant4's own G4ICRU90StoppingData.
//
// These are the tables Geant4 uses in place of PSTAR for air, water and graphite when
// G4EmParameters::SetUseICRU90Data(true) has been called. They are off by default, which is
// exactly why they need a test of their own: nothing in a default run touches them, so a
// wrong table would sit there indefinitely and then be wrong for the one user who turns the
// switch on.
//
// ref/oracle/icru90.csv is dumped from the data object directly rather than through the model,
// because turning the flag on in the dumper would replace the stopping power in every other
// row of every other file - G4BraggModel resolves iICRU90 before iPSTAR. The number here is
// the same number the model returns with the flag on.
//
// Two claims, two sections. The tables being right, and the model reaching them.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "data/icru90.hh"
#include "data/materials.cuh"
#include "physics/em/bragg.cuh"

using namespace g4gpu;
using real_t = double;

int main() {
  const char* env = std::getenv("G4GPU_ORACLE");
  const std::string dir = (env != nullptr) ? env : "ref/oracle";
  int fails = 0;

  // ---------------------------------------------------------- 1. the tables, against Geant4
  //
  // The energy range is deliberately wider than the tables: 1 eV to 20 GeV, against a proton
  // table that starts at 1 keV and ends at 10 GeV. Below the first point Geant4 extrapolates
  // as sqrt(E/E0) and above the last its spline saturates, and both of those are as much a
  // part of the answer as the 57 tabulated values.
  int total = 0;
  {
    FILE* f = std::fopen((dir + "/icru90.csv").c_str(), "r");
    if (f == nullptr) {
      std::printf("cannot read %s/icru90.csv\n", dir.c_str());
      return 1;
    }
    char line[512];
    if (std::fgets(line, sizeof line, f) == nullptr) {
      std::fclose(f);
      return 1;
    }

    struct Acc {
      double worst = 0;
      double at_e = 0;
      int n = 0;
    };
    // Split by material *and* particle: six tables, six ways to be wrong, and a single worst
    // deviation over all of them would let five right tables hide one wrong one.
    Acc acc[3][2];
    int seen[3] = {0, 0, 0};
    const char* names[3] = {"G4_AIR", "G4_WATER", "G4_GRAPHITE"};

    while (std::fgets(line, sizeof line, f) != nullptr) {
      char mat[128] = {0}, part[64] = {0};
      double e = 0, g4 = 0;
      if (std::sscanf(line, "%127[^,],%63[^,],%lf,%lf", mat, part, &e, &g4) != 4) { continue; }
      int mi = -1;
      for (int i = 0; i < 3; ++i) {
        if (std::strcmp(mat, names[i]) == 0) { mi = i; }
      }
      if (mi < 0) { continue; }
      const bool alpha = (std::strcmp(part, "alpha") == 0);
      seen[mi] = 1;

      // icru90_index is the same name lookup the material record uses, so the test exercises
      // it rather than assuming the enum order.
      const int idx = data::icru90_index(mat);
      const real_t ours =
          data::icru90_mass_stopping<real_t>(idx, static_cast<real_t>(e), alpha);

      if (std::fabs(g4) <= 0) { continue; }
      Acc& a = acc[mi][alpha ? 1 : 0];
      ++a.n;
      const double dev = std::fabs(ours / g4 - 1);
      if (dev > a.worst) {
        a.worst = dev;
        a.at_e = e;
      }
    }
    std::fclose(f);

    std::printf("== ICRU 90 mass stopping power vs G4ICRU90StoppingData ==\n");
    std::printf("  %-14s %-8s %8s %14s %14s\n", "material", "particle", "points", "worst dev",
                "at E/MeV");
    for (int m = 0; m < 3; ++m) {
      for (int p = 0; p < 2; ++p) {
        const Acc& a = acc[m][p];
        std::printf("  %-14s %-8s %8d %13.3e %14.4g\n", names[m], p ? "alpha" : "proton", a.n,
                    a.worst, a.at_e);
        total += a.n;
        if (a.n == 0) {
          std::printf("    FAIL: %s %s was never compared\n", names[m],
                      p ? "alpha" : "proton");
          ++fails;
        } else if (a.worst > 1e-9) {
          // 1e-9: the port reads the same tabulated values and runs the same spline, in
          // double. What is left is the order of a few multiplications. Getting here needed
          // one real fix - Geant4 stores these as G4float, so the number it splines is
          // (double)(float)119.70 and not (double)119.70, which was a uniform 6e-8 out.
          std::printf("    FAIL: %.3e from Geant4 at %.4g MeV, tolerance 1e-9\n", a.worst,
                      a.at_e);
          ++fails;
        }
      }
    }
    if (seen[0] == 0 || seen[1] == 0 || seen[2] == 0) {
      // G4ICRU90StoppingData::Initialise scans the material table once and latches when it
      // has found all three names, so a material built after that call gets no index and its
      // rows never reach this file. That happened to graphite, and it is the kind of gap that
      // leaves a table looking tested.
      std::printf("\n  FAIL: the oracle covers only %d of the 3 ICRU90 materials\n",
                  seen[0] + seen[1] + seen[2]);
      ++fails;
    }
  }

  // ------------------------------------------------ 2. the switch changes the stopping power
  //
  // The tables being right is one claim; the model reaching them is another, and it is the
  // one that would silently fail. `bragg_dedx_unrestricted` consults `m.icru90` before
  // `m.nist_stopping` and returns as soon as it has an answer, which is the order
  // G4BraggModel::ElectronicDEDX resolves them in. If that branch were dead - a stale index,
  // a wrong order - every number in section 1 would still be right and the switch would do
  // nothing.
  //
  // So: build water twice, once with the PSTAR index and once with ICRU 90 on, and require
  // that the ICRU 90 build returns the ICRU 90 table and that the two differ. They should:
  // ICRU 90 revised the water stopping power by about a percent near the Bragg peak, which is
  // the entire reason the option exists.
  {
    std::printf("\n== the switch reaches the model ==\n");
    data::MaterialTable<real_t> t{};
    t.count = 0;
    const int zs[2] = {1, 8};
    const real_t aH = data::atomic_mass<real_t>(1), aO = data::atomic_mass<real_t>(8);
    const real_t sum = real_t(2) * aH + aO;
    const real_t w[2] = {real_t(2) * aH / sum, aO / sum};
    const int i1 = data::add_material<real_t>(t, real_t(1.0), 2, zs, w, real_t(78.0));
    const int i2 = data::add_material<real_t>(t, real_t(1.0), 2, zs, w, real_t(78.0));
    if (i1 < 0 || i2 < 0) {
      std::printf("  FAIL: could not build the two water records\n");
      ++fails;
    } else {
      data::Material<real_t> pstar = t.m[i1];
      data::Material<real_t> icru = t.m[i2];
      data::set_nist_stopping<real_t>(pstar, "G4_WATER");
      data::set_nist_stopping<real_t>(icru, "G4_WATER");
      data::set_icru90<real_t>(icru, "G4_WATER");
      if (pstar.nist_stopping < 0) {
        std::printf("  FAIL: G4_WATER has no PSTAR row - the comparison is meaningless\n");
        ++fails;
      }
      if (icru.icru90 != data::kIcru90Water) {
        std::printf("  FAIL: set_icru90 gave index %d, expected %d\n", icru.icru90,
                    data::kIcru90Water);
        ++fails;
      }
      if (pstar.icru90 >= 0) {
        std::printf("  FAIL: the PSTAR-only record has an ICRU90 index - the flag leaked\n");
        ++fails;
      }
      std::printf("  %12s %14s %14s %10s\n", "E/MeV", "PSTAR", "ICRU90", "differ");
      double worst_gap = 0;
      int wrong_branch = 0;
      // 1 keV to 2 MeV, and not further, because that is where both tables are live. PSTAR's
      // grid ends at 2 MeV - Geant4 hands protons to G4BetheBlochModel above that, so
      // G4PSTARStopping has no reason to tabulate higher - and both Geant4 and this port
      // clamp a spline to its end value outside its range. ICRU 90's proton table runs to
      // 10 GeV. So above 2 MeV the two differ by 70% for a reason that has nothing to do with
      // ICRU 90, and quoting that as "the switch changes the answer" would be measuring the
      // wrong thing.
      const int kLast = 33;  // 1e-3 * 10^3.3 = 1.995 MeV
      for (int i = 0; i <= kLast; ++i) {
        const real_t e = static_cast<real_t>(1e-3 * std::pow(10.0, i / 10.0));
        const real_t a = em::bragg_dedx_unrestricted(pstar, e);
        const real_t b = em::bragg_dedx_unrestricted(icru, e);
        // The ICRU90 branch has to return the ICRU90 table, times density/10.
        const real_t want =
            data::icru90_mass_stopping<real_t>(data::kIcru90Water, e, false) * icru.density
            / real_t(10);
        if (b != want) { ++wrong_branch; }
        if (a > 0) { worst_gap = std::fmax(worst_gap, std::fabs(double(b) / double(a) - 1)); }
        if (i % 8 == 0 || i == kLast) {
          std::printf("  %12.4g %14.6g %14.6g %9.3f%%\n", double(e), double(a), double(b),
                      (a > 0) ? 100 * (double(b) / double(a) - 1) : 0.0);
        }
      }
      if (wrong_branch != 0) {
        std::printf("  FAIL: %d of 41 energies did not return the ICRU90 table - the model\n"
                    "        is not taking the ICRU90 branch\n",
                    wrong_branch);
        ++fails;
      }
      std::printf("  largest difference from PSTAR: %.3f%%\n", 100 * worst_gap);
      // 0.1%: ICRU 90's whole purpose is that it differs from PSTAR. If the two agreed
      // everywhere the switch would be untestable and this test would be measuring nothing.
      if (worst_gap < 1e-3) {
        std::printf("  FAIL: ICRU90 and PSTAR agree to %.1e everywhere - one of the two\n"
                    "        tables is not being read\n",
                    worst_gap);
        ++fails;
      }
    }
  }

  std::printf("\n%d table points compared\n%s (%d failures)\n", total,
              fails ? "FAILED" : "PASSED", fails);
  return fails ? 1 : 0;
}
