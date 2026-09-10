// The de-excitation module's deterministic layer against Geant4's own answers.
//
// Four files, four independent classes, all compared at machine precision:
//
//   deex_params.csv       G4DeexPrecoParameters as the install has it against
//                         deex_params.cuh's transcription. This is the one comparison that
//                         cannot be "close": a flag that differs by one selects a different
//                         model, so every value is compared exactly and every flag bit for
//                         bit.
//   deex_masses.csv       G4NucleiProperties nuclear mass / mass excess / binding energy /
//                         IsInStableTable over 18,407 (Z, A) pairs. Split by which of the
//                         four branches answered - AME2012, the Moller-Nix table, the
//                         Z == A / Z == 0 special cases, Weizsaecker - because a single
//                         worst-case would report whichever branch happened to be worse
//                         instead of saying which one is wrong. The nucleus is ~2e5 MeV and
//                         the module's thresholds are differences of two of them at 10 eV, so
//                         the tolerance is relative and near double precision.
//   deex_corrections.csv  G4PairingCorrection, G4ShellCorrection and
//                         G4NuclearLevelData::GetLevelDensity over the same sweep, split by
//                         which table's window the nuclide fell in.
//   deex_coulomb.csv      G4NuclearRadii::RadiusCB, G4CoulombBarrier and
//                         G4FermiCoulombBarrier for all six evaporation ejectiles.
//
// Anti-vacuity: the run recorded in the commit message perturbed, one at a time, the AME
// short-table slicing (off by one row), the Cook shell window (Z >= 28 to Z >= 27), the
// Cameron-Gilbert pairing fall-through (max(x,0) removed) and the Coulomb barrier's rho
// (0.4 R_CB of the ejectile to 0.4 R_CB of the residual). Each perturbation fails this test;
// what each one reported is in the commit body.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "physics/hadronic/deexcitation/corrections.cuh"
#include "physics/hadronic/deexcitation/coulomb_barrier.cuh"
#include "physics/hadronic/deexcitation/deex_params.cuh"
#include "physics/hadronic/deexcitation/nuclear_masses.cuh"

using namespace g4gpu;

namespace {

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

/// Relative deviation, with an absolute floor so that a reference of exactly zero - which
/// happens for every neutral ejectile's Coulomb barrier and for the shell correction outside
/// both tables - is compared as "is ours also zero" rather than dividing by it.
double dev_of(double ours, double g4, double floor_abs) {
  if (std::fabs(g4) > floor_abs) { return std::fabs(ours - g4) / std::fabs(g4); }
  return (std::fabs(ours) > floor_abs) ? 1.0 : 0.0;
}

std::vector<std::string> read_lines(const std::string& path) {
  std::vector<std::string> out;
  FILE* f = std::fopen(path.c_str(), "r");
  if (f == nullptr) { return out; }
  char line[4096];
  while (std::fgets(line, sizeof line, f) != nullptr) { out.push_back(line); }
  std::fclose(f);
  return out;
}

int fails = 0;

/// A bucket is a failure both when it is over tolerance and when it is EMPTY: an empty bucket
/// means the predicate that selects it never fired, which is how a wrong window would hide.
void report(Cell& c, double tol, const char* label) {
  const bool ok = (c.n > 0 && c.worst <= tol);
  std::printf("  %-34s %6d pts  worst %.3g  %s\n", label, c.n, c.worst, ok ? "OK" : "FAIL");
  if (!ok) {
    ++fails;
    std::printf("      worst at %s\n", c.where.c_str());
  }
}
}  // namespace

int main() {
  const char* env = std::getenv("G4GPU_ORACLE");
  const std::string dir = (env != nullptr) ? env : "ref/oracle";

  // ------------------------------------------------------------------- parameters
  {
    const auto lines = read_lines(dir + "/deex_params.csv");
    if (lines.size() < 2) {
      std::printf("cannot read %s/deex_params.csv - run ref/oracle/run.bat tables\n",
                  dir.c_str());
      return 1;
    }
    const deex::DeexParameters& p = deex::deex_params();
    int n = 0, bad = 0;
    for (std::size_t i = 1; i < lines.size(); ++i) {
      char name[64];
      double value = 0;
      if (std::sscanf(lines[i].c_str(), "%63[^,],%lf", name, &value) != 2) { continue; }
      double ours = 0;
      bool known = true;
      // Units: the dump divides energies by MeV and times by ns, and this port's unit system
      // is the same one, so the comparison is of plain numbers.
      if (!std::strcmp(name, "LevelDensity")) { ours = p.level_density; }
      else if (!std::strcmp(name, "R0")) { ours = p.r0; }
      else if (!std::strcmp(name, "TransitionsR0")) { ours = p.transitions_r0; }
      else if (!std::strcmp(name, "FBUEnergyLimit")) { ours = p.fbu_energy_limit; }
      else if (!std::strcmp(name, "FermiEnergy")) { ours = p.fermi_energy; }
      else if (!std::strcmp(name, "PrecoLowEnergy")) { ours = p.preco_low_energy; }
      else if (!std::strcmp(name, "PrecoHighEnergy")) { ours = p.preco_high_energy; }
      else if (!std::strcmp(name, "PhenoFactor")) { ours = p.pheno_factor; }
      else if (!std::strcmp(name, "MinExcitation")) { ours = p.min_excitation; }
      else if (!std::strcmp(name, "MaxLifeTime")) { ours = p.max_life_time; }
      else if (!std::strcmp(name, "MinExPerNucleounForMF")) {
        ours = p.min_ex_per_nucleon_for_mf;
      }
      else if (!std::strcmp(name, "MinZForPreco")) { ours = p.min_z_for_preco; }
      else if (!std::strcmp(name, "MinAForPreco")) { ours = p.min_a_for_preco; }
      else if (!std::strcmp(name, "PrecoModelType")) { ours = p.preco_type; }
      else if (!std::strcmp(name, "DeexModelType")) { ours = p.deex_type; }
      else if (!std::strcmp(name, "TwoJMAX")) { ours = p.two_j_max; }
      else if (!std::strcmp(name, "NeverGoBack")) { ours = p.never_go_back; }
      else if (!std::strcmp(name, "UseSoftCutoff")) { ours = p.use_soft_cutoff; }
      else if (!std::strcmp(name, "UseCEM")) { ours = p.use_cem; }
      else if (!std::strcmp(name, "UseGNASH")) { ours = p.use_gnash; }
      else if (!std::strcmp(name, "UseHETC")) { ours = p.use_hetc; }
      else if (!std::strcmp(name, "UseAngularGen")) { ours = p.use_angular_gen; }
      else if (!std::strcmp(name, "PrecoDummy")) { ours = p.preco_dummy; }
      else if (!std::strcmp(name, "CorrelatedGamma")) { ours = p.correlated_gamma; }
      else if (!std::strcmp(name, "StoreICLevelData")) { ours = p.store_ic_level_data; }
      else if (!std::strcmp(name, "InternalConversionFlag")) { ours = p.internal_conversion; }
      else if (!std::strcmp(name, "LevelDensityFlag")) { ours = p.level_density_flag; }
      else if (!std::strcmp(name, "DiscreteExcitationFlag")) {
        ours = p.discrete_excitation_flag;
      }
      else if (!std::strcmp(name, "IsomerProduction")) { ours = p.isomer_production; }
      else if (!std::strcmp(name, "DeexChannelsType")) { ours = p.deex_channel_type; }
      else if (!std::strcmp(name, "HandlerMaxZForFermiBreakUp")) {
        ours = deex::kMaxZForFermiBreakUp;
      }
      else if (!std::strcmp(name, "HandlerMaxAForFermiBreakUp")) {
        ours = deex::kMaxAForFermiBreakUp;
      }
      else { known = false; }
      if (!known) { continue; }
      ++n;
      // Exact, not close: these are flags and thresholds, and a parameter that is 1e-15 off is
      // a transcription error even where it would not change a dispatch.
      const double d = dev_of(ours, value, 0.0);
      if (d > 1e-15) {
        ++bad;
        std::printf("  param MISMATCH %-28s ours %.17g  G4 %.17g\n", name, ours, value);
      }
    }
    std::printf("parameters: %d compared, %d mismatched  %s\n", n, bad, bad ? "FAIL" : "OK");
    if (bad || n < 30) { ++fails; }
  }

  // ------------------------------------------------------------------- masses
  {
    const auto lines = read_lines(dir + "/deex_masses.csv");
    // Which branch answered, from the port's own predicates - so the split is a claim about
    // the port's dispatch and a wrong dispatch shows up as a whole bucket going bad.
    Cell mass[4], excess[4], binding[4];
    int table_flag_bad = 0, rows = 0;
    for (std::size_t i = 1; i < lines.size(); ++i) {
      int Z = 0, A = 0, stable = 0;
      double m = 0, me = 0, be = 0;
      if (std::sscanf(lines[i].c_str(), "%d,%d,%d,%lf,%lf,%lf", &Z, &A, &stable, &m, &me,
                      &be) != 6) {
        continue;
      }
      ++rows;
      if (deex::is_in_stable_table(A, Z) != (stable != 0)) { ++table_flag_bad; }

      int k;
      const bool light = (Z <= 2) && ((Z == 1 && A <= 3) || (Z == 0 && A == 1) ||
                                      (Z == 2 && (A == 3 || A == 4)));
      if (light || deex::ame12_in_table(Z, A)) { k = 0; }
      else if (deex::theo_in_table(Z, A)) { k = 1; }
      else if (Z == A || Z == 0) { k = 2; }
      else { k = 3; }

      char buf[160];
      std::snprintf(buf, sizeof buf, "Z=%d A=%d (branch %d)", Z, A, k);
      note(mass[k], dev_of(deex::nuclear_mass(A, Z), m, 1e-30), buf);
      // The mass excess passes through zero (C-12 is zero by definition of the amu), so it
      // needs an absolute floor; 1e-9 MeV is a millionth of the smallest excess in the table.
      note(excess[k], dev_of(deex::mass_excess(A, Z), me, 1e-9), buf);
      note(binding[k], dev_of(deex::binding_energy(A, Z), be, 1e-9), buf);
    }
    std::printf("masses: %d rows, IsInStableTable mismatches %d  %s\n", rows, table_flag_bad,
                table_flag_bad ? "FAIL" : "OK");
    if (table_flag_bad) { ++fails; }
    const char* names[4] = {"AME2012 / light PDG", "Moller-Nix table", "Z==A or Z==0",
                            "Weizsaecker formula"};
    for (int k = 0; k < 4; ++k) {
      char lbl[80];
      std::snprintf(lbl, sizeof lbl, "mass, %s", names[k]);
      report(mass[k], 1e-14, lbl);
      std::snprintf(lbl, sizeof lbl, "excess, %s", names[k]);
      report(excess[k], 1e-13, lbl);
      std::snprintf(lbl, sizeof lbl, "binding, %s", names[k]);
      report(binding[k], 1e-12, lbl);
    }
  }

  // ------------------------------------------------------------------- corrections
  {
    const auto lines = read_lines(dir + "/deex_corrections.csv");
    // Split by which table's window the nuclide is in: the tables are chosen by window and a
    // wrong window is the failure mode, so the buckets have to be the windows.
    Cell pair_tab, pair_form, fission_pair, shell_cook, shell_cg, shell_none, ld, ldpair;
    int rows = 0;
    for (std::size_t i = 1; i < lines.size(); ++i) {
      int Z = 0, A = 0;
      double pc = 0, fpc = 0, sc = 0, l = 0, ldp = 0;
      if (std::sscanf(lines[i].c_str(), "%d,%d,%lf,%lf,%lf,%lf,%lf", &Z, &A, &pc, &fpc, &sc,
                      &l, &ldp) != 7) {
        continue;
      }
      ++rows;
      const int N = A - Z;
      char buf[160];
      std::snprintf(buf, sizeof buf, "Z=%d A=%d N=%d", Z, A, N);
      double tmp = 0;
      const bool cg = deex::cameron_gilbert_pairing(N, Z, tmp);
      note(cg ? pair_tab : pair_form, dev_of(deex::pairing_correction(A, Z), pc, 1e-12), buf);
      note(fission_pair, dev_of(deex::fission_pairing_correction(A, Z), fpc, 1e-12), buf);
      const bool ck = deex::cook_shell(N, Z, tmp);
      const bool cgs = deex::cameron_gilbert_shell(N, Z, tmp);
      note(ck ? shell_cook : (cgs ? shell_cg : shell_none),
           dev_of(deex::shell_correction(A, Z), sc, 1e-12), buf);
      // has_levels is irrelevant with level_density_flag set - which is the point: if the
      // flag were read wrongly this bucket would be off by the four-way A > 20 formula.
      note(ld, dev_of(deex::level_density(Z, A, 0.0, true), l, 1e-30), buf);
      note(ldpair, dev_of(deex::level_data_pairing_correction(Z, A), ldp, 1e-12), buf);
    }
    std::printf("corrections: %d rows\n", rows);
    report(pair_tab, 1e-14, "pairing, Cameron-Gilbert window");
    report(pair_form, 1e-14, "pairing, 12/sqrt(A) fall-through");
    report(fission_pair, 1e-14, "fission pairing");
    report(shell_cook, 1e-14, "shell, Cook window");
    report(shell_cg, 1e-14, "shell, Cameron-Gilbert window");
    report(shell_none, 1e-14, "shell, outside both (zero)");
    report(ld, 1e-15, "level density (fLD = true)");
    report(ldpair, 1e-14, "level-data pairing correction");
  }

  // ------------------------------------------------------------------- Coulomb barriers
  {
    const auto lines = read_lines(dir + "/deex_coulomb.csv");
    Cell rad_res, rad_ej, barrier, barrier_u, fermi_b, pen;
    int rows = 0;
    for (std::size_t i = 1; i < lines.size(); ++i) {
      int ejZ = 0, ejA = 0, Zres = 0, Ares = 0;
      double U = 0, rres = 0, rej = 0, cb = 0, fcb = 0, pf = 0;
      if (std::sscanf(lines[i].c_str(), "%d,%d,%d,%d,%lf,%lf,%lf,%lf,%lf,%lf", &ejZ, &ejA,
                      &Zres, &Ares, &U, &rres, &rej, &cb, &fcb, &pf) != 10) {
        continue;
      }
      ++rows;
      char buf[176];
      std::snprintf(buf, sizeof buf, "ejectile (Z=%d,A=%d) on (Z=%d,A=%d) U=%g MeV", ejZ, ejA,
                    Zres, Ares, U);
      // The dump prints radii in fermi; this port's unit of length is mm.
      note(rad_res, dev_of(deex::radius_cb(Zres, Ares) / deex::fermi(), rres, 1e-30), buf);
      note(rad_ej, dev_of(deex::radius_cb(ejZ, ejA) / deex::fermi(), rej, 1e-30), buf);
      note(U > 0 ? barrier_u : barrier,
           dev_of(deex::coulomb_barrier(ejA, ejZ, Ares, Zres, U), cb, 1e-30), buf);
      note(fermi_b, dev_of(deex::fermi_coulomb_barrier(ejA, ejZ, Ares, Zres), fcb, 1e-30), buf);
      note(pen, dev_of(deex::barrier_penetration_factor(ejA, ejZ, Zres), pf, 1e-30), buf);
    }
    std::printf("Coulomb: %d rows\n", rows);
    report(rad_res, 1e-15, "RadiusCB(residual)");
    report(rad_ej, 1e-15, "RadiusCB(ejectile)");
    report(barrier, 1e-14, "G4CoulombBarrier, U = 0");
    report(barrier_u, 1e-14, "G4CoulombBarrier, U > 0");
    report(fermi_b, 1e-14, "G4FermiCoulombBarrier");
    report(pen, 1e-15, "BarrierPenetrationFactor");
  }

  std::printf("%s\n", fails == 0 ? "ALL OK" : "FAILURES ABOVE");
  return fails == 0 ? 0 : 1;
}
