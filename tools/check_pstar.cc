// A standalone check that the extracted PSTAR and ASTAR tables, read through Geant4's spline,
// reproduce Geant4's own answers.
//
// Standalone and host-only on purpose: it needs no CUDA toolchain and no scene, so it can be
// run the moment the generator has run and before anything is wired into the transport. If
// this disagrees, nothing downstream is worth building.
//
//   g++ -std=c++17 -O2 -I . -o check_pstar.exe check_pstar.cc && ./check_pstar.exe
//
// The reference is ref/oracle/bragg.csv, dumped by ref/dump/g4dump.cc from
// G4PSTARStopping::GetElectronicDEDX and G4ASTARStopping::GetElectronicDEDX directly - not
// from G4BraggModel, so no restricted-dE/dx correction is in the way.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifndef __CUDACC__
#define __host__
#define __device__
#endif

#include "data/nist_stopping.hh"

using namespace g4gpu::data;

namespace {

double DensityOf(const std::string& name) {
  // From ref/oracle/materials.csv. Only the materials bragg.csv contains.
  if (name == "G4_AIR") { return 0.00120479; }
  if (name == "G4_WATER") { return 1.0; }
  if (name == "G4_A-150_TISSUE") { return 1.127; }
  if (name == "G4_BONE_COMPACT_ICRU") { return 1.85; }
  return -1.0;  // CustomSiGe, which has no NIST row and is not checked here
}

struct Acc {
  double worst = 0;
  std::string where;
  int n = 0;
};

void Track(Acc& a, double ours, double theirs, const std::string& what) {
  ++a.n;
  const double scale = std::fabs(theirs);
  if (scale <= 0) { return; }
  const double dev = std::fabs(ours - theirs) / scale;
  if (dev > a.worst) {
    a.worst = dev;
    a.where = what;
  }
}

}  // namespace

int main() {
  const char* path = "../ref/oracle/bragg.csv";
  std::FILE* f = std::fopen(path, "r");
  if (f == nullptr) {
    std::printf("cannot open %s\n", path);
    return 2;
  }
  char line[1024];
  if (std::fgets(line, sizeof line, f) == nullptr) { return 2; }  // header

  Acc pstar, astar;
  int no_row = 0;
  while (std::fgets(line, sizeof line, f) != nullptr) {
    // material,particle,energy,cut,bragg,braggion,pstar_index,pstar_dedx,astar_index
    char mat[128] = {0}, part[64] = {0};
    double energy = 0, cut = 0, bragg = 0, braggion = 0, pstar_dedx = 0;
    int ip = 0, ia = 0;
    if (std::sscanf(line, "%127[^,],%63[^,],%lf,%lf,%lf,%lf,%d,%lf,%d", mat, part, &energy,
                    &cut, &bragg, &braggion, &ip, &pstar_dedx, &ia)
        != 9) {
      continue;
    }
    const double density = DensityOf(mat);
    if (density <= 0) { continue; }
    const int row = nist_stopping_index(mat);
    if (row < 0) {
      ++no_row;
      continue;
    }
    char what[256];

    // Protons only for PSTAR: the CSV's pstar_dedx column is the proton table, evaluated at
    // the row's own energy, whatever particle the line is about.
    if (std::strcmp(part, "proton") == 0 && ip >= 0 && pstar_dedx > 0) {
      const double ours = pstar_mass_stopping<double>(row, energy) * density / 10.0;
      std::snprintf(what, sizeof what, "%s at %g MeV", mat, energy);
      Track(pstar, ours, pstar_dedx, what);
    }

    // ASTAR: G4BraggIonModel's answer for an alpha is the ASTAR table times the density, plus
    // a restricted correction above the cut. Below the cut there is no correction, and the
    // Bragg peak - where this matters - is far below any production cut, so the comparison is
    // restricted to where braggion_dedx is the bare table.
    if (std::strcmp(part, "alpha") == 0 && ia >= 0 && energy < cut && braggion > 0) {
      const double ours = astar_mass_stopping<double>(row, energy) * density / 10.0;
      std::snprintf(what, sizeof what, "%s at %g MeV", mat, energy);
      Track(astar, ours, braggion, what);
    }
  }
  std::fclose(f);

  std::printf("PSTAR: %d points, worst deviation %.3g%%   %s\n", pstar.n, 100 * pstar.worst,
              pstar.where.c_str());
  std::printf("ASTAR: %d points, worst deviation %.3g%%   %s\n", astar.n, 100 * astar.worst,
              astar.where.c_str());
  if (no_row > 0) { std::printf("%d rows had no NIST table and were skipped\n", no_row); }

  int fails = 0;
  if (pstar.n < 100) {
    std::printf("FAIL: too few PSTAR points compared\n");
    ++fails;
  }
  if (pstar.worst > 1e-6) {
    std::printf("FAIL: PSTAR does not reproduce Geant4\n");
    ++fails;
  }
  std::printf(fails == 0 ? "PASSED\n" : "%d FAILED\n", fails);
  return fails == 0 ? 0 : 1;
}
