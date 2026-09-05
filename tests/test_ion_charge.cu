// G4ionEffectiveCharge and G4IonisParamMat's derived quantities, against Geant4 directly.
//
// Two sections, because they are two different kinds of claim.
//
// 1. The per-material ionisation parameters. Zeff, the Fermi energy, <A^-2/3> and the
//    L-factor are the inputs every ion and MSC formula is built on, and nothing checked them
//    directly until now - they were only ever visible through a dE/dx with a dozen other
//    inputs. Two of them are new or changed: fermi_energy did not exist, and inv_a23 was
//    switched from an exact 2/3 power to G4Pow's Taylor approximation of it.
//
// 2. The effective charge itself, both branches, separately. The helium branch (Zi <= 2)
//    covers alpha and He3, the only ions G4EmStandardPhysics registers by name, and it was
//    already exercised through the Bethe-Bloch corrections. The heavy-ion branch was not:
//    nothing in the oracle used an ion heavier than an alpha, so the Zi > 2 formula had
//    nothing behind it and went untranscribed - it needs the material's Fermi energy, which
//    the material record did not carry.
//
// ref/oracle/ion_charge.csv holds both of Geant4 11.1.1's published outputs for nine ions
// from helium to uranium over 1 keV to 100 GeV: EffectiveCharge, and
// EffectiveChargeSquareRatio, which is (charge * chargeCorrection)^2. The ratio is what makes
// the charge correction checkable at all - 11.1.1 keeps chargeCorrection private, with no
// accessor - and it is the quantity G4EmCorrections actually consumes.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "core/particle.cuh"
#include "data/fermi_velocity.hh"
#include "data/materials.cuh"
#include "physics/em/em_corrections.cuh"

using namespace g4gpu;
using real_t = double;

namespace {

// The oracle's materials, in the order ref/dump/g4dump.cc builds them.
const char* kNames[6] = {"G4_AIR",          "G4_WATER",
                         "G4_A-150_TISSUE", "G4_BONE_COMPACT_ICRU",
                         "CustomSiGe",      "CustomMolecularWater"};

struct Acc {
  double worst = 0;
  std::string where;
  int n = 0;
};

void track(Acc& a, double ours, double g4, const char* what) {
  if (std::fabs(g4) <= 0) { return; }
  ++a.n;
  const double dev = std::fabs(ours / g4 - 1);
  if (dev > a.worst) {
    a.worst = dev;
    a.where = what;
  }
}

int material_index(const char* name) {
  for (int i = 0; i < 6; ++i) {
    if (std::strcmp(name, kNames[i]) == 0) { return i; }
  }
  return -1;
}

}  // namespace

int main() {
  data::MaterialTable<real_t> table{};
  table.count = 0;
  data::Material<real_t> b1[data::kNumMaterials];
  data::build_b1_materials<real_t>(b1);
  data::Material<real_t> mats[6];
  for (int i = 0; i < 4; ++i) { mats[i] = b1[i]; }
  {
    const int zs[2] = {14, 32};
    const real_t w[2] = {real_t(0.6), real_t(0.4)};
    const int idx =
        data::add_material<real_t>(table, real_t(4.2), 2, zs, w, real_t(224.66048772837619));
    if (idx < 0) { return 1; }
    mats[4] = table.m[idx];
  }
  {
    const real_t aH = data::atomic_mass<real_t>(1), aO = data::atomic_mass<real_t>(8);
    const real_t total = real_t(2) * aH + aO;
    const int zs[2] = {1, 8};
    const real_t w[2] = {real_t(2) * aH / total, aO / total};
    const int idx =
        data::add_material<real_t>(table, real_t(1.0), 2, zs, w, real_t(68.9984175));
    if (idx < 0) { return 1; }
    mats[5] = table.m[idx];
  }

  const char* env = std::getenv("G4GPU_ORACLE");
  const std::string dir = (env != nullptr) ? env : "ref/oracle";
  int fails = 0;
  char line[512];

  // ---------------------------------------------------------------- 1. material parameters
  {
    FILE* f = std::fopen((dir + "/material_ionis.csv").c_str(), "r");
    if (f == nullptr) {
      std::printf("cannot read %s/material_ionis.csv\n", dir.c_str());
      return 1;
    }
    if (std::fgets(line, sizeof line, f) == nullptr) {
      std::fclose(f);
      return 1;
    }
    std::printf("== G4IonisParamMat, per material ==\n");
    std::printf("  %-22s %14s %14s %14s %10s\n", "material", "Zeff", "E_fermi/MeV", "<A^-2/3>",
                "L-factor");
    Acc zeff, efermi, inva23, lfac, ne;
    while (std::fgets(line, sizeof line, f) != nullptr) {
      char name[128] = {0};
      double z = 0, ef = 0, ia = 0, lf = 0, mex = 0, nel = 0;
      if (std::sscanf(line, "%127[^,],%lf,%lf,%lf,%lf,%lf,%lf", name, &z, &ef, &ia, &lf, &mex,
                      &nel)
          != 7) {
        continue;
      }
      const int mi = material_index(name);
      if (mi < 0) { continue; }
      const data::Material<real_t>& m = mats[mi];

      // The L-factor has no field on Material because nothing consumes it yet, so the
      // extracted table is checked here through the same atom-density weighting Geant4 uses.
      // An extracted table with no value check behind it is exactly what went wrong with the
      // stopping powers (docs/RISK.md O6).
      real_t lf_ours = 0, norm = 0;
      for (int i = 0; i < m.n_elements; ++i) {
        const int zi = static_cast<int>(m.z[i] + real_t(0.5));
        lf_ours += m.n_atoms[i] * static_cast<real_t>(data::ziegler_l_factor(zi));
        norm += m.n_atoms[i];
      }
      lf_ours = (norm > 0) ? lf_ours / norm : real_t(0);

      std::printf("  %-22s %14.6g %14.6g %14.6g %10.5f\n", name, double(m.z_eff),
                  double(m.fermi_energy), double(m.inv_a23), double(lf_ours));
      track(zeff, double(m.z_eff), z, name);
      track(efermi, double(m.fermi_energy), ef, name);
      track(inva23, double(m.inv_a23), ia, name);
      track(lfac, double(lf_ours), lf, name);
      track(ne, double(m.electron_density), nel, name);
    }
    std::fclose(f);

    struct Row {
      const char* what;
      Acc* a;
    };
    // All five are pure arithmetic over the same tabulated constants Geant4 uses, so the only
    // difference should be the order of a few multiplications. 1e-12 is the width of that.
    const Row rows[5] = {{"Zeff", &zeff},
                         {"Fermi energy", &efermi},
                         {"<A^-2/3>", &inva23},
                         {"L-factor", &lfac},
                         {"electron density", &ne}};
    std::printf("\n  %-20s %8s %14s   %s\n", "quantity", "points", "worst dev", "where");
    for (const Row& r : rows) {
      std::printf("  %-20s %8d %13.3e   %s\n", r.what, r.a->n, r.a->worst, r.a->where.c_str());
      if (r.a->n == 0) {
        std::printf("    FAIL: %s was never compared\n", r.what);
        ++fails;
      } else if (r.a->worst > 1e-12) {
        std::printf("    FAIL: %s is %.3e from Geant4, tolerance 1e-12\n", r.what, r.a->worst);
        ++fails;
      }
    }
  }

  // ---------------------------------------------------------------- 2. the effective charge
  {
    FILE* f = std::fopen((dir + "/ion_charge.csv").c_str(), "r");
    if (f == nullptr) {
      std::printf("cannot read %s/ion_charge.csv\n", dir.c_str());
      return 1;
    }
    if (std::fgets(line, sizeof line, f) == nullptr) {
      std::fclose(f);
      return 1;
    }
    // Each branch gets its own accumulator: they are different formulae with different
    // failure modes, and the tighter of the two must not cover for the looser.
    Acc he_q, he_r, heavy_q, heavy_r;
    int ions_seen[93] = {};
    while (std::fgets(line, sizeof line, f) != nullptr) {
      char mat[128] = {0}, ion[64] = {0};
      int z = 0, a = 0;
      double mass = 0, e = 0, q = 0, ratio = 0;
      if (std::sscanf(line, "%127[^,],%63[^,],%d,%d,%lf,%lf,%lf,%lf", mat, ion, &z, &a, &mass,
                      &e, &q, &ratio)
          != 8) {
        continue;
      }
      const int mi = material_index(mat);
      if (mi < 0 || z < 1 || z > 92) { continue; }
      ions_seen[z] = 1;

      // The port's particle record, with Geant4's own mass and charge for this ion.
      ParticleDef<real_t> pd{};
      pd.mass = static_cast<real_t>(mass);
      pd.charge = static_cast<real_t>(z);
      pd.spin = real_t(0);
      pd.is_ion = true;
      pd.is_alpha = (z == 2);

      real_t corr = 1;
      const real_t ours_q = em::ion_effective_charge(mats[mi], pd, static_cast<real_t>(e), corr);
      const double ours_ratio = double(ours_q * corr) * double(ours_q * corr);

      char what[220];
      std::snprintf(what, sizeof what, "%s in %s at %.4g MeV", ion, mat, e);
      if (z <= 2) {
        track(he_q, ours_q, q, what);
        track(he_r, ours_ratio, ratio, what);
      } else {
        track(heavy_q, ours_q, q, what);
        track(heavy_r, ours_ratio, ratio, what);
      }
    }
    std::fclose(f);

    int n_heavy = 0;
    for (int z = 3; z <= 92; ++z) { n_heavy += ions_seen[z]; }

    std::printf("\n== G4ionEffectiveCharge ==\n");
    std::printf("  %-34s %8s %14s   %s\n", "quantity", "points", "worst dev", "where");
    struct Row {
      const char* what;
      Acc* a;
    };
    const Row rows[4] = {{"helium branch, charge", &he_q},
                         {"helium branch, (q*corr)^2", &he_r},
                         {"heavy-ion branch, charge", &heavy_q},
                         {"heavy-ion branch, (q*corr)^2", &heavy_r}};
    for (const Row& r : rows) {
      std::printf("  %-34s %8d %13.3e   %s\n", r.what, r.a->n, r.a->worst, r.a->where.c_str());
    }

    if (he_q.n == 0) {
      std::printf("\n  FAIL: the helium branch was never exercised\n");
      ++fails;
    }
    if (n_heavy < 7) {
      std::printf("\n  FAIL: only %d ions above helium in the oracle - the heavy-ion branch\n"
                  "        needs several, or a formula that is wrong for most of the periodic\n"
                  "        table passes on the one it happens to fit\n",
                  n_heavy);
      ++fails;
    }
    // 1e-9 rather than 0: the port computes in double from Geant4's own double constants, so
    // the only difference should be the order of a few multiplications. It is not 1e-12
    // because the branch takes exp(0.3*log(y)) of a quantity built from G4Pow's approximate
    // cube root, and that amplifies the last couple of bits.
    for (const Row& r : rows) {
      if (r.a->n > 0 && r.a->worst > 1e-9) {
        std::printf("\n  FAIL: %s is %.3e from Geant4, tolerance 1e-9\n    worst at %s\n",
                    r.what, r.a->worst, r.a->where.c_str());
        ++fails;
      }
    }
  }

  std::printf("\n%s (%d failures)\n", fails ? "FAILED" : "PASSED", fails);
  return fails ? 1 : 0;
}
