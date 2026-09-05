// Turns the raw ICRU 90 stopping tables into the header the transport uses.
//
//   tools/extract_icru90.sh   Geant4's sources  ->  tools/icru90_raw.hh
//   tools/gen_icru90.cc       that              ->  src/data/icru90.hh
//
// Two steps for the same reason as the NIST tables: the second derivatives of Geant4's
// not-a-knot spline are a tridiagonal solve, which the device cannot do and which should not
// be redone at every upload for a quantity that never changes. Computed here, once, and
// emitted as data.
//
// One output header rather than two, unlike gen_stopping.cc: this is 318 values, not 370 kB,
// so there is no compile-time reason to split the names from the tables.
//
// Run tools/regen_icru90.sh rather than this directly: it extracts, recompiles this, and then
// checks the result against Geant4's own answers, in that order. The order matters - the
// generator compiles the raw header *in*, so regenerating the raw header and re-running an old
// generator binary silently produces the old tables from the new data (docs/RISK.md O6).
#include "fmt17.hh"

#include <cstdio>
#include <vector>

// Compiled as plain C++. The spline header is __host__ __device__ because the transport needs
// it on both sides; this generator uses only the host half.
#ifndef __CUDACC__
#define __host__
#define __device__
#endif

#include "data/g4spline.hh"
#include "icru90_raw.hh"

using namespace g4gpu::data;

namespace {

/// Geant4 holds these stopping powers as `G4float` and widens them on the way into the
/// physics vector:
///
///     static const G4float e0_proton[57] = { 119.70f, ... };
///     data->PutValues(i, e[i]*CLHEP::MeV, ((G4double)dedx[i])*fac);
///
/// So the number Geant4 splines is `(double)(float)119.70`, not `(double)119.70`. Emitting the
/// decimal as a double put every one of the six tables 6e-8 from Geant4's answer -
/// consistently, at every energy, in every material, which is the signature of a float and
/// not of an arithmetic difference. The energy grids are `G4double` in Geant4 and are left
/// alone.
double Float32(double v) { return static_cast<double>(static_cast<float>(v)); }

/// One row of `real_t(...)` literals, in the style of the other generated tables: the
/// conversion is written out so a float build gets float constants rather than a narrowing
/// of doubles at runtime.
void EmitRow(std::FILE* f, const double* v, int n, const char* indent) {
  std::fprintf(f, "%s{", indent);
  for (int i = 0; i < n; ++i) {
    char nb[64];
    std::fprintf(f, "real_t(%s)%s", g4gpu::tools::fmt17(v[i], nb, sizeof nb),
                 (i + 1 < n) ? "," : "");
  }
  std::fprintf(f, "}");
}

/// A `template <typename real_t> ... (int i)` accessor over three rows, following
/// data/nist_stopping.hh: a function-local static in a __host__ __device__ function is how
/// this project carries a constant table to both sides of the machine. A namespace-scope
/// __constant__ array cannot be, and an array of pointers to them cannot be gathered on the
/// device.
void EmitTable(std::FILE* f, const char* name, const double* const rows[3], int n,
               const char* doc) {
  std::fprintf(f, "/// %s\ntemplate <typename real_t>\n"
                  "__host__ __device__ inline const real_t* %s(int i) {\n"
                  "  static const real_t v[3][%d] = {\n",
               doc, name, n);
  for (int m = 0; m < 3; ++m) {
    EmitRow(f, rows[m], n, "    ");
    std::fprintf(f, "%s\n", (m < 2) ? "," : "");
  }
  std::fprintf(f, "  };\n"
                  "  return (i >= 0 && i < kNumIcru90) ? v[i] : nullptr;\n"
                  "}\n\n");
}

void EmitGrid(std::FILE* f, const char* name, const double* v, int n, const char* doc) {
  std::fprintf(f, "/// %s\ntemplate <typename real_t>\n"
                  "__host__ __device__ inline const real_t* %s() {\n"
                  "  static const real_t v[%d] = ",
               doc, name, n);
  EmitRow(f, v, n, "");
  std::fprintf(f, ";\n  return v;\n}\n\n");
}

std::vector<double> SecondDerivatives(const double* x, const double* y, int n) {
  std::vector<double> d2(static_cast<std::size_t>(n), 0.0);
  fill_second_derivatives(x, y, n, d2.data());
  return d2;
}

}  // namespace

int main(int argc, char** argv) {
  const char* path = (argc > 1) ? argv[1] : "../src/data/icru90.hh";
  std::FILE* f = std::fopen(path, "w");
  if (f == nullptr) {
    std::printf("cannot write %s\n", path);
    return 2;
  }

  using namespace g4gpu::data::icru90_raw;
  constexpr int kNP = 57, kNA = 49;

  // Rounded through float before anything else, including the spline: Geant4 splines the
  // widened floats, so the second derivatives have to come from the same values.
  std::vector<double> prot_v[3] = {std::vector<double>(kProtonAir, kProtonAir + kNP),
                                   std::vector<double>(kProtonWater, kProtonWater + kNP),
                                   std::vector<double>(kProtonGraphite,
                                                       kProtonGraphite + kNP)};
  std::vector<double> alph_v[3] = {std::vector<double>(kAlphaAir, kAlphaAir + kNA),
                                   std::vector<double>(kAlphaWater, kAlphaWater + kNA),
                                   std::vector<double>(kAlphaGraphite, kAlphaGraphite + kNA)};
  for (int m = 0; m < 3; ++m) {
    for (double& v : prot_v[m]) { v = Float32(v); }
    for (double& v : alph_v[m]) { v = Float32(v); }
  }

  std::fprintf(f, "%s",
      "// ICRU Report 90 electronic stopping powers, with the second derivatives of Geant4's\n"
      "// spline.\n"
      "//\n"
      "// GENERATED by tools/gen_icru90.cc from tools/icru90_raw.hh, which is itself generated\n"
      "// by tools/extract_icru90.sh from Geant4 11.1.1's sources. Do not edit by hand.\n"
      "//\n"
      "// Three materials - G4_AIR, G4_WATER, G4_GRAPHITE - for protons and alphas. Geant4\n"
      "// reads these only when G4EmParameters::SetUseICRU90Data(true) has been called: they\n"
      "// are off by default, and when on they take precedence over PSTAR for those three\n"
      "// materials. G4BraggModel::ElectronicDEDX resolves iICRU90 first and reaches iPSTAR\n"
      "// only when it is negative, so this replaces PSTAR rather than correcting it.\n"
      "//\n"
      "// Values are mass stopping powers in MeV cm2/g, as PSTAR's are:\n"
      "//\n"
      "//     dE/dx [MeV/mm] = table * density[g/cm3] / 10\n"
      "//\n"
      "// Below the first grid point (1 keV) Geant4 extrapolates as sqrt(E/E0) rather than\n"
      "// splining - G4ICRU90StoppingData::GetDEDX - which is the rule PSTAR uses as well.\n"
      "//\n"
      "// The alpha tables are indexed by *scaled* kinetic energy: the alpha's kinetic energy\n"
      "// divided by its mass ratio to a proton, which is what G4BraggIonModel passes in.\n"
      "#pragma once\n"
      "#include <cstring>\n"
      "#include \"data/g4spline.hh\"\n"
      "\n"
      "namespace g4gpu::data {\n"
      "\n"
      "/// The three materials, in the order Geant4's nameNIST_ICRU90 lists them - which is\n"
      "/// also the order G4ICRU90StoppingData::GetIndex returns.\n"
      "enum Icru90Material { kIcru90Air = 0, kIcru90Water = 1, kIcru90Graphite = 2 };\n"
      "constexpr int kNumIcru90 = 3;\n"
      "constexpr int kIcru90ProtonPoints = 57;\n"
      "constexpr int kIcru90AlphaPoints = 49;\n"
      "\n"
      "/// The material index for a Geant4 material name, or -1. Three names, so a chain of\n"
      "/// comparisons rather than a table.\n"
      "inline int icru90_index(const char* g4_name) {\n"
      "  if (g4_name == nullptr) { return -1; }\n"
      "  if (std::strcmp(g4_name, \"G4_AIR\") == 0) { return kIcru90Air; }\n"
      "  if (std::strcmp(g4_name, \"G4_WATER\") == 0) { return kIcru90Water; }\n"
      "  if (std::strcmp(g4_name, \"G4_GRAPHITE\") == 0) { return kIcru90Graphite; }\n"
      "  return -1;\n"
      "}\n"
      "\n");

  EmitGrid(f, "icru90_proton_grid", kProtonGrid, kNP, "Proton kinetic energies, MeV.");
  EmitGrid(f, "icru90_alpha_grid", kAlphaGrid, kNA, "Alpha scaled kinetic energies, MeV.");

  const double* prot[3] = {prot_v[0].data(), prot_v[1].data(), prot_v[2].data()};
  const double* alph[3] = {alph_v[0].data(), alph_v[1].data(), alph_v[2].data()};
  const std::vector<double> pd2[3] = {SecondDerivatives(kProtonGrid, prot[0], kNP),
                                      SecondDerivatives(kProtonGrid, prot[1], kNP),
                                      SecondDerivatives(kProtonGrid, prot[2], kNP)};
  const std::vector<double> ad2[3] = {SecondDerivatives(kAlphaGrid, alph[0], kNA),
                                      SecondDerivatives(kAlphaGrid, alph[1], kNA),
                                      SecondDerivatives(kAlphaGrid, alph[2], kNA)};
  const double* pd2p[3] = {pd2[0].data(), pd2[1].data(), pd2[2].data()};
  const double* ad2p[3] = {ad2[0].data(), ad2[1].data(), ad2[2].data()};

  EmitTable(f, "icru90_proton_row", prot, kNP,
            "Proton mass stopping power, MeV cm2/g, one row per ICRU90 material.");
  EmitTable(f, "icru90_proton_d2", pd2p, kNP,
            "Not-a-knot second derivatives for the proton rows.");
  EmitTable(f, "icru90_alpha_row", alph, kNA,
            "Alpha mass stopping power, MeV cm2/g, against *scaled* kinetic energy.");
  EmitTable(f, "icru90_alpha_d2", ad2p, kNA,
            "Not-a-knot second derivatives for the alpha rows.");

  std::fprintf(f, "%s",
      "/// ICRU 90 mass stopping power, MeV cm2/g, or 0 for a material with no table.\n"
      "///\n"
      "/// Verbatim from G4ICRU90StoppingData::GetDEDX: the spline above the first grid point,\n"
      "/// and sqrt(E/E0) scaling of the first value below it.\n"
      "template <typename real_t>\n"
      "__host__ __device__ inline real_t icru90_mass_stopping(int idx, real_t energy,\n"
      "                                                       bool alpha) {\n"
      "  const real_t* x = alpha ? icru90_alpha_grid<real_t>() : icru90_proton_grid<real_t>();\n"
      "  const real_t* y = alpha ? icru90_alpha_row<real_t>(idx) : icru90_proton_row<real_t>(idx);\n"
      "  const real_t* d2 = alpha ? icru90_alpha_d2<real_t>(idx) : icru90_proton_d2<real_t>(idx);\n"
      "  const int n = alpha ? kIcru90AlphaPoints : kIcru90ProtonPoints;\n"
      "  if (y == nullptr) { return real_t(0); }\n"
      "  if (energy <= x[0]) { return y[0] * sqrt(energy / x[0]); }\n"
      "  return spline_value<real_t>(x, y, d2, n, energy);\n"
      "}\n"
      "\n"
      "}  // namespace g4gpu::data\n");
  std::fclose(f);
  std::printf("wrote %s\n", path);
  return 0;
}
