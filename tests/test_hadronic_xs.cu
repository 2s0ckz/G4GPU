// The four model layers under every hadronic cross section in QBBC, against Geant4's own.
//
// Five oracle files, one per layer, because a disagreement in
// G4BGGNucleonInelasticXS at 200 MeV can come from six places and this is what says which:
//
//   had_radii.csv    G4NuclearRadii's seven radii and G4NucleiProperties::GetNuclearMass
//   had_coulomb.csv  the three Coulomb factors - which are thresholds, not scalings
//   had_hnxsc.csv    G4HadronNucleonXsc, all six entry points, eleven projectiles
//   had_ggcomp.csv   G4ComponentGGHadronNucleusXsc and G4ComponentGGNuclNuclXsc
//   had_bgg.csv      the four BGG data sets end to end, and with them
//                    G4UPiNuclearCrossSection (see ref/dump/dump_hadronic_xs.cc section 5 for
//                    why it has no file of its own - its accessors are inlines over private
//                    statics that Geant4's DLL does not export)
//
// WHY EVERY COLUMN AND NOT JUST THE ONE THE TRANSPORT READS
//
// The Glauber-Gribov elastic cross section is total minus inelastic, and the two are built from
// the same `ratio` with two different logarithms. A wrong nucleusSquare scales all three; a
// wrong cofInelastic moves inelastic and elastic in opposite directions; a bar-correction table
// read at the wrong Z moves inelastic only. Comparing the difference alone cannot separate
// them, and production and diffraction are the two columns nothing else in this port reads yet -
// so they are the two most likely to be wrong and never noticed.
//
// THE ENERGIES ARE NOT A LOGARITHMIC SCAN
//
// They are the dump's, which is twelve per decade PLUS every boundary energy each class changes
// model at: 14 MeV, 20 MeV, 91 GeV, the top of each table, and the kinetic energy at each of the
// thirty-five pLab branch points of G4HadronNucleonXsc, computed per particle. A scan alone
// lands on none of them, and every one of them is a branch where a `<` transcribed as a `<=`
// changes the answer at exactly one energy.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdarg>
#include <cstring>
#include <string>
#include <vector>

#include "physics/hadronic/xs/bgg_nucleon_xs.cuh"
#include "physics/hadronic/xs/bgg_pion_xs.cuh"
#include "physics/hadronic/xs/gg_hadron_nucleus_xsc.cuh"
#include "physics/hadronic/xs/gg_nucl_nucl_xsc.cuh"
#include "physics/hadronic/xs/hadron_nucleon_xsc.cuh"
#include "physics/hadronic/xs/nuclear_radii.cuh"

#include "data/isotope_list.hh"
#include "data/materials.cuh"

using namespace g4gpu;
using namespace g4gpu::hadronic::xs;
using real_t = double;

namespace {

// 1e-12 relative, for the same reason test_nucleon_xs.cu uses it: there is nothing between this
// port's answer and Geant4's but double arithmetic in the same order, the CSVs are %.17g, and
// the observed agreement is a few ulps. The limit leaves room for a compiler reassociating a
// sum and none for a wrong table entry or a wrong branch.
constexpr double kTol = 1e-12;

/// A named comparison bucket: how many points, the worst relative deviation, and where.
struct Cell {
  int n = 0;
  double worst = 0;
  std::string where;
  int refused = 0;  ///< points the port refused by name rather than answered
};

void note(Cell& c, double dev, const std::string& what) {
  ++c.n;
  if (dev > c.worst) {
    c.worst = dev;
    c.where = what;
  }
}

/// Relative deviation, with an absolute floor because several of these columns are *genuinely*
/// zero over whole regions - the Coulomb factor below the barrier, the inelastic cross section
/// below threshold, G4NeutronCaptureXS above 20 MeV, RadiusECS above A = 50. A relative
/// comparison there divides by zero; treating "both zero" as agreement and "one zero" as total
/// disagreement is the only test that distinguishes a threshold from a missing branch.
double deviation(double ours, double g4) {
  const double floor = 1e-300;
  if (std::fabs(g4) > floor) { return std::fabs(ours - g4) / std::fabs(g4); }
  return (std::fabs(ours) > floor) ? 1.0 : 0.0;
}

void cmp(Cell& c, double ours, double g4, const char* fmt, ...) {
  char buf[256];
  va_list ap;
  va_start(ap, fmt);
  std::vsnprintf(buf, sizeof buf, fmt, ap);
  va_end(ap);
  char full[400];
  std::snprintf(full, sizeof full, "%s (ours %.17g, G4 %.17g)", buf, ours, g4);
  note(c, deviation(ours, g4), full);
}

/// Opens an oracle CSV and skips its header. Fatal if absent: a test that silently compares
/// nothing passes.
FILE* open_oracle(const std::string& dir, const char* name) {
  const std::string path = dir + "/" + name;
  FILE* f = std::fopen(path.c_str(), "r");
  if (f == nullptr) {
    std::printf("FAIL: cannot read %s - run ref/oracle/run.bat tables in this worktree first\n",
                path.c_str());
    return nullptr;
  }
  char line[1024];
  if (std::fgets(line, sizeof line, f) == nullptr) {
    std::fclose(f);
    return nullptr;
  }
  return f;
}

/// The eleven projectile names the dump uses, mapped onto this port's Projectile builders.
/// Returns false for a name the port has no projectile for, which is a test failure and not a
/// skip - the dump and the port must cover the same set.
bool projectile_by_name(const char* name, Projectile<real_t>& out) {
  if (std::strcmp(name, "proton") == 0) { out = proton<real_t>(); return true; }
  if (std::strcmp(name, "neutron") == 0) { out = neutron<real_t>(); return true; }
  if (std::strcmp(name, "anti_proton") == 0) { out = anti_proton<real_t>(); return true; }
  if (std::strcmp(name, "anti_neutron") == 0) { out = anti_neutron<real_t>(); return true; }
  if (std::strcmp(name, "pi+") == 0) { out = pi_plus<real_t>(); return true; }
  if (std::strcmp(name, "pi-") == 0) { out = pi_minus<real_t>(); return true; }
  if (std::strcmp(name, "kaon+") == 0) { out = kaon_plus<real_t>(); return true; }
  if (std::strcmp(name, "kaon-") == 0) { out = kaon_minus<real_t>(); return true; }
  if (std::strcmp(name, "kaon0S") == 0) { out = kaon_zero_short<real_t>(); return true; }
  if (std::strcmp(name, "kaon0L") == 0) { out = kaon_zero_long<real_t>(); return true; }
  if (std::strcmp(name, "gamma") == 0) { out = gamma<real_t>(); return true; }
  if (std::strcmp(name, "alpha") == 0) { out = alpha<real_t>(); return true; }
  if (std::strcmp(name, "deuteron") == 0) { out = deuteron<real_t>(); return true; }
  if (std::strcmp(name, "triton") == 0) { out = triton<real_t>(); return true; }
  if (std::strcmp(name, "He3") == 0) { out = he3<real_t>(); return true; }
  if (std::strcmp(name, "C12") == 0) { out = generic_ion<real_t>(6, 12); return true; }
  if (std::strcmp(name, "Fe56") == 0) { out = generic_ion<real_t>(26, 56); return true; }
  return false;
}

// ------------------------------------------------------------------- 1. radii and masses

/// Every (Z, A) had_radii.csv carried, so the isotope windows the port and the dump each hold a
/// copy of can be compared against each other.
bool g_seen[96][300] = {};

int check_radii(const std::string& dir, Cell* cells) {
  FILE* f = open_oracle(dir, "had_radii.csv");
  if (f == nullptr) { return 1; }
  char line[1024];
  while (std::fgets(line, sizeof line, f) != nullptr) {
    int z = 0, a = 0, in_ame = 0;
    double expl = 0, rad = 0, rms = 0, nngg = 0, ecs = 0, hngg = 0, kngg = 0, nd = 0, cb = 0,
           mass = 0;
    if (std::sscanf(line, "%d,%d,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%d", &z, &a, &expl,
                    &rad, &rms, &nngg, &ecs, &hngg, &kngg, &nd, &cb, &mass, &in_ame) != 13) {
      continue;
    }
    if (z >= 0 && z < 96 && a >= 0 && a < 300) { g_seen[z][a] = true; }
    cmp(cells[0], nr_explicit_radius<real_t>(z, a), expl, "ExplicitRadius Z=%d A=%d", z, a);
    cmp(cells[1], nr_radius<real_t>(z, a), rad, "Radius Z=%d A=%d", z, a);
    cmp(cells[2], nr_radius_rms<real_t>(z, a), rms, "RadiusRMS Z=%d A=%d", z, a);
    cmp(cells[3], nr_radius_nngg<real_t>(z, a), nngg, "RadiusNNGG Z=%d A=%d", z, a);
    cmp(cells[4], nr_radius_ecs<real_t>(z, a), ecs, "RadiusECS Z=%d A=%d", z, a);
    cmp(cells[5], nr_radius_hngg<real_t>(a), hngg, "RadiusHNGG A=%d", a);
    cmp(cells[6], nr_radius_kngg<real_t>(a), kngg, "RadiusKNGG A=%d", a);
    cmp(cells[7], nr_radius_nd<real_t>(a), nd, "RadiusND A=%d", a);
    cmp(cells[8], nr_radius_cb<real_t>(z, a), cb, "RadiusCB Z=%d A=%d", z, a);

    // The mass table is the one place a refusal is expected and correct: G4NucleiProperties
    // falls back to G4NucleiPropertiesTheoreticalTable outside AME2012, which is not ported.
    // `in_ame` is Geant4's own IsInStableTable, so this checks that the port refuses exactly
    // the nuclides Geant4 computes rather than measures - not one more and not one fewer.
    const bool known = data::nuclear_mass_known(a, z);
    if (known != (in_ame != 0)) {
      char buf[160];
      std::snprintf(buf, sizeof buf,
                    "AME12 coverage Z=%d A=%d: port says %s, G4 IsInStableTable says %s", z, a,
                    known ? "known" : "refused", in_ame ? "known" : "not in table");
      note(cells[10], 1.0, buf);
    } else {
      ++cells[10].n;
    }
    if (known) {
      cmp(cells[9], data::nuclear_mass<real_t>(a, z), mass, "NuclearMass Z=%d A=%d", z, a);
    } else {
      ++cells[9].refused;
    }
  }
  std::fclose(f);
  return 0;
}

// ------------------------------------------------------------------- 2. the Coulomb factors

int check_coulomb(const std::string& dir, Cell* cells) {
  FILE* f = open_oracle(dir, "had_coulomb.csv");
  if (f == nullptr) { return 1; }
  char line[1024];
  while (std::fgets(line, sizeof line, f) != nullptr) {
    char form[32], pname[32];
    int z = 0, a = 0;
    double e = 0, fac = 0;
    if (std::sscanf(line, "%31[^,],%31[^,],%d,%d,%lf,%lf", form, pname, &z, &a, &e, &fac) != 6) {
      continue;
    }
    Projectile<real_t> p;
    if (!projectile_by_name(pname, p)) {
      note(cells[0], 1.0, std::string("no port projectile named ") + pname);
      continue;
    }
    if (std::strcmp(form, "nucleus") == 0) {
      // G4NuclearRadii::CoulombFactor(Z, A, particle, ekin) needs the target's nuclear mass, so
      // a nuclide outside AME2012 is refused here rather than given a zero mass - a zero mass
      // makes totTcm zero and the factor zero, i.e. it turns the cross section off silently.
      if (!data::nuclear_mass_known(a, z)) {
        ++cells[0].refused;
        continue;
      }
      cmp(cells[0], nr_coulomb_factor_nucleus<real_t>(z, a, p, e),
          fac, "CoulombFactor(Z,A) %s Z=%d A=%d at %.6g MeV", pname, z, a, e);
    } else if (std::strcmp(form, "nucleon") == 0) {
      cmp(cells[1], nr_coulomb_factor<real_t>(p, proton<real_t>(), e), fac,
          "CoulombFactor(p,proton) %s at %.6g MeV", pname, e);
    } else if (std::strcmp(form, "nucleon_n") == 0) {
      cmp(cells[1], nr_coulomb_factor<real_t>(p, neutron<real_t>(), e), fac,
          "CoulombFactor(p,neutron) %s at %.6g MeV", pname, e);
    } else if (std::strcmp(form, "barrier") == 0) {
      cmp(cells[2], hn_coulomb_barrier<real_t>(p, proton<real_t>(), e), fac,
          "HadronNucleonXsc::CoulombBarrier %s at %.6g MeV", pname, e);
    }
  }
  std::fclose(f);
  return 0;
}

// ------------------------------------------------------------------- 3. G4HadronNucleonXsc

int check_hnxsc(const std::string& dir, Cell* cells) {
  FILE* f = open_oracle(dir, "had_hnxsc.csv");
  if (f == nullptr) { return 1; }
  char line[1024];
  while (std::fgets(line, sizeof line, f) != nullptr) {
    char method[32], pname[32], nname[32];
    double e = 0, tot = 0, el = 0, inel = 0;
    if (std::sscanf(line, "%31[^,],%31[^,],%31[^,],%lf,%lf,%lf,%lf", method, pname, nname, &e,
                    &tot, &el, &inel) != 7) {
      continue;
    }
    Projectile<real_t> p, n;
    if (!projectile_by_name(pname, p) || !projectile_by_name(nname, n)) {
      note(cells[0], 1.0, std::string("no port projectile named ") + pname);
      continue;
    }
    HadXs<real_t> x;
    int k = 0;
    if (std::strcmp(method, "PDG") == 0) {
      x = hn_xsc_pdg<real_t>(p, n, e);
      k = 0;
    } else if (std::strcmp(method, "NS") == 0) {
      x = hn_xsc_ns<real_t>(p, n, e);
      k = 1;
    } else if (std::strcmp(method, "Dispatch") == 0) {
      x = hadron_nucleon_xsc<real_t>(p, n, e);
      k = 2;
    } else if (std::strcmp(method, "KaonNS") == 0) {
      x = hn_kaon_xsc_ns<real_t>(p, n, e);
      k = 3;
    } else if (std::strcmp(method, "KaonGG") == 0) {
      x = hn_kaon_xsc_gg<real_t>(p, n, e);
      k = 4;
    } else if (std::strcmp(method, "KaonVG") == 0) {
      x = hn_kaon_xsc_vg<real_t>(p, n, e);
      k = 5;
    } else {
      continue;
    }
    if (!x.ok()) {
      ++cells[k].refused;
      continue;
    }
    cmp(cells[k], x.total, tot, "%s %s+%s at %.6g MeV total", method, pname, nname, e);
    cmp(cells[k], x.elastic, el, "%s %s+%s at %.6g MeV elastic", method, pname, nname, e);
    cmp(cells[k], x.inelastic, inel, "%s %s+%s at %.6g MeV inelastic", method, pname, nname, e);
  }
  std::fclose(f);
  return 0;
}

// ------------------------------------------------------------------- 4. the GG components

int check_ggcomp(const std::string& dir, Cell* cells) {
  FILE* f = open_oracle(dir, "had_ggcomp.csv");
  if (f == nullptr) { return 1; }
  char line[1024];
  while (std::fgets(line, sizeof line, f) != nullptr) {
    char comp[32], pname[32];
    int z = 0, a = 0;
    double e = 0, tot = 0, inel = 0, el = 0, prod = 0, dif = 0;
    if (std::sscanf(line, "%31[^,],%31[^,],%d,%d,%lf,%lf,%lf,%lf,%lf,%lf", comp, pname, &z, &a,
                    &e, &tot, &inel, &el, &prod, &dif) != 10) {
      continue;
    }
    Projectile<real_t> p;
    if (!projectile_by_name(pname, p)) {
      note(cells[0], 1.0, std::string("no port projectile named ") + pname);
      continue;
    }
    const bool nucl = (std::strcmp(comp, "GGNuclNucl") == 0);
    const int k = nucl ? 1 : 0;
    // An ion projectile whose own mass is not in AME2012 cannot be built at all; a target
    // outside it cannot have a Coulomb barrier. Both are refusals, and generic_ion returns a
    // zero mass for the first, so it is tested before the call rather than after.
    if (nucl && (p.mass <= 0 || !data::nuclear_mass_known(a, z))) {
      ++cells[k].refused;
      continue;
    }
    const HadXs<real_t> x = nucl ? ggnn_compute_cross_sections<real_t>(p, e, z, a)
                                 : ggh_compute_cross_sections<real_t>(p, e, z, a);
    if (!x.ok()) {
      ++cells[k].refused;
      continue;
    }
    cmp(cells[k], x.total, tot, "%s %s Z=%d A=%d at %.6g MeV total", comp, pname, z, a, e);
    cmp(cells[k], x.inelastic, inel, "%s %s Z=%d A=%d at %.6g MeV inelastic", comp, pname, z, a,
        e);
    cmp(cells[k], x.elastic, el, "%s %s Z=%d A=%d at %.6g MeV elastic", comp, pname, z, a, e);
    cmp(cells[k], x.production, prod, "%s %s Z=%d A=%d at %.6g MeV production", comp, pname, z,
        a, e);
    cmp(cells[k], x.diffraction, dif, "%s %s Z=%d A=%d at %.6g MeV diffraction", comp, pname, z,
        a, e);
  }
  std::fclose(f);
  return 0;
}

// ------------------------------------------------------------------- 5. the four BGG sets

int check_bgg(const std::string& dir, Cell* cells) {
  // Eight instances, because `isProton` / `isPiplus` is a per-instance member and the per-Z
  // tables are per-class: one object answers for one projectile, exactly as in Geant4 where the
  // dump has to build eight of them too.
  BggNucleonTable<real_t> nucEl, nucIn;
  bgg_build_nucleon_table<real_t>(true, nucEl);
  bgg_build_nucleon_table<real_t>(false, nucIn);
  BggPionTable<real_t> piEl, piIn;
  bgg_build_pion_table<real_t>(true, piEl);
  bgg_build_pion_table<real_t>(false, piIn);

  FILE* f = open_oracle(dir, "had_bgg.csv");
  if (f == nullptr) { return 1; }
  char line[1024];
  while (std::fgets(line, sizeof line, f) != nullptr) {
    char dset[40], pname[32];
    int z = 0;
    double e = 0, xs = 0;
    if (std::sscanf(line, "%39[^,],%31[^,],%d,%lf,%lf", dset, pname, &z, &e, &xs) != 5) {
      continue;
    }
    Projectile<real_t> p;
    if (!projectile_by_name(pname, p)) {
      note(cells[0], 1.0, std::string("no port projectile named ") + pname);
      continue;
    }
    XsValue<real_t> x;
    int k = -1;
    if (std::strcmp(dset, "BGGNucleonElastic") == 0) {
      x = bgg_nucleon_element_xs<real_t>(nucEl, p, e, z);
      k = 0;
    } else if (std::strcmp(dset, "BGGNucleonInelastic") == 0) {
      x = bgg_nucleon_element_xs<real_t>(nucIn, p, e, z);
      k = 1;
    } else if (std::strcmp(dset, "BGGPionElastic") == 0) {
      x = bgg_pion_element_xs<real_t>(piEl, p, e, z);
      k = 2;
    } else if (std::strcmp(dset, "BGGPionInelastic") == 0) {
      x = bgg_pion_element_xs<real_t>(piIn, p, e, z);
      k = 3;
    } else {
      continue;
    }
    if (!x.ok()) {
      ++cells[k].refused;
      continue;
    }
    cmp(cells[k], x.value, xs, "%s %s Z=%d at %.10g MeV", dset, pname, z, e);
    // The middle band of the two pion classes IS G4UPiNuclearCrossSection, returned with
    // nothing applied to it, so those rows are also the only oracle this port has for that
    // class. Counted separately so its coverage cannot vanish into the BGG total.
    if (k >= 2 && z > 1) {
      const double ekin = (e > 1.0) ? e : 1.0;  // fLowestEnergy
      const bool in_upi_band = (k == 2) ? (ekin > 20.0 && ekin <= 91000.0)
                                        : (ekin >= 20.0 && ekin <= 91000.0);
      if (in_upi_band) {
        cmp(cells[4], x.value, xs, "G4UPiNuclearCrossSection %s %s Z=%d at %.10g MeV",
            (k == 2) ? "elastic" : "inelastic", pname, z, e);
      }
    }
  }
  std::fclose(f);
  return 0;
}

/// Reports one bucket and counts the failures: a worst deviation over tolerance, and an empty
/// bucket, which is the failure that a test comparing nothing would otherwise pass.
int report(const char* group, const char* name, const Cell& c, bool may_be_empty = false) {
  std::printf("  %-22s %-26s %8d points", group, name, c.n);
  if (c.refused > 0) { std::printf("  %6d refused", c.refused); } else { std::printf("%15s", ""); }
  std::printf("  worst %10.3e\n", c.worst);
  int fails = 0;
  if (c.n == 0 && !may_be_empty) {
    std::printf("    FAIL: no points compared for %s / %s\n", group, name);
    ++fails;
  }
  if (c.worst > kTol) {
    std::printf("    FAIL: %s / %s off by %.3e, limit %.0e\n      %s\n", group, name, c.worst,
                kTol, c.where.c_str());
    ++fails;
  }
  return fails;
}

}  // namespace

int main() {
  const char* env = std::getenv("G4GPU_ORACLE");
  const std::string dir = (env != nullptr) ? env : "ref/oracle";

  Cell radii[11], coul[3], hn[6], gg[2], bgg[5];
  int io = 0;
  io += check_radii(dir, radii);
  io += check_coulomb(dir, coul);
  io += check_hnxsc(dir, hn);
  io += check_ggcomp(dir, gg);
  io += check_bgg(dir, bgg);
  if (io != 0) {
    std::printf("\nFAILED (%d oracle files missing)\n", io);
    return 1;
  }

  std::printf("== hadronic cross-section model layers vs Geant4 11.1.1 ==\n\n");
  int fails = 0;
  const char* rnames[11] = {"ExplicitRadius", "Radius",     "RadiusRMS",  "RadiusNNGG",
                            "RadiusECS",      "RadiusHNGG", "RadiusKNGG", "RadiusND",
                            "RadiusCB",       "NuclearMass", "AME12 coverage"};
  for (int i = 0; i < 11; ++i) { fails += report("G4NuclearRadii", rnames[i], radii[i]); }
  const char* cnames[3] = {"CoulombFactor(Z,A,p,E)", "CoulombFactor(p,nucleon,E)",
                           "HadronNucleonXsc barrier"};
  for (int i = 0; i < 3; ++i) { fails += report("Coulomb", cnames[i], coul[i]); }
  const char* hnames[6] = {"HadronNucleonXscPDG", "HadronNucleonXscNS", "HadronNucleonXsc",
                           "KaonNucleonXscNS",    "KaonNucleonXscGG",   "KaonNucleonXscVG"};
  for (int i = 0; i < 6; ++i) { fails += report("G4HadronNucleonXsc", hnames[i], hn[i]); }
  fails += report("Glauber-Gribov", "GGHadronNucleusXsc", gg[0]);
  fails += report("Glauber-Gribov", "GGNuclNuclXsc", gg[1]);
  const char* bnames[5] = {"BGGNucleonElasticXS", "BGGNucleonInelasticXS",
                           "BGGPionElasticXS", "BGGPionInelasticXS",
                           "UPiNuclearCrossSection"};
  for (int i = 0; i < 5; ++i) { fails += report("BGG data sets", bnames[i], bgg[i]); }

  // Coverage gates. Each of these has been wrong once in this project: an oracle regenerated
  // with a narrower loop, or a projectile silently skipped, leaves every check above passing
  // while testing a fraction of the code.
  //
  // THE ISOTOPE WINDOWS, CHECKED AGAINST A SECOND COPY OF THEM
  //
  // src/data/isotope_list.hh and ref/dump/dump_hadronic_xs.cc each transcribe G4IsotopeList.hh
  // independently - the header is static and not linkable, so the dump cannot use Geant4's own
  // copy. That means a wrong amin[Z] in one of them is detectable and a wrong one in both is
  // not. had_radii.csv is one (Z, A) row per A in the DUMP's window, so requiring the PORT's
  // window to be exactly the set of rows present is the comparison. It also fails loudly if the
  // oracle is ever regenerated by a dumper with a narrower loop.
  {
    int missing = 0, extra = 0;
    for (int z = 1; z <= 92; ++z) {
      for (int a = 1; a < 300; ++a) {
        const bool in_port = (a >= data::isotope_amin()[z] && a <= data::isotope_amax()[z]);
        if (in_port && !g_seen[z][a]) {
          if (missing < 5) {
            std::printf("  FAIL: isotope_list.hh has Z=%d A=%d in [amin,amax] but "
                        "had_radii.csv does not\n", z, a);
          }
          ++missing;
        }
        // The dump also emits the rounded mean A the BGG classes use as theA[Z], which is not
        // always inside [amin, amax] - so a row outside the port's window is only an error if
        // it is not that one.
        if (!in_port && g_seen[z][a] && a != g4lrint<real_t>(data::atomic_mass<real_t>(z))) {
          if (extra < 5) {
            std::printf("  FAIL: had_radii.csv has Z=%d A=%d, outside isotope_list.hh's "
                        "[%d,%d] and not theA[Z]=%d\n", z, a, data::isotope_amin()[z],
                        data::isotope_amax()[z], g4lrint<real_t>(data::atomic_mass<real_t>(z)));
          }
          ++extra;
        }
      }
    }
    if (missing != 0 || extra != 0) {
      std::printf("  FAIL: isotope windows disagree with the oracle: %d missing, %d extra\n",
                  missing, extra);
      ++fails;
    } else {
      std::printf("  %-22s %-26s %8d (Z,A) pairs, windows agree with G4IsotopeList.hh\n",
                  "G4IsotopeList", "amin/amax", radii[1].n);
    }
  }
  if (hn[1].n < 3000) {
    std::printf("  FAIL: HadronNucleonXscNS compared at only %d points\n", hn[1].n);
    ++fails;
  }
  if (bgg[4].n < 10000) {
    std::printf("  FAIL: G4UPiNuclearCrossSection reached at only %d points - the BGG pion "
                "middle band is its only oracle\n", bgg[4].n);
    ++fails;
  }
  // THE REFUSAL ARM OF THE MASS TABLE
  //
  // had_radii.csv cannot exercise it: every (Z, A) with a G4PARTICLEXS file is inside AME2012,
  // so `refused` above is zero and that is correct. The refusal still has to be tested, because
  // a data::nuclear_mass that answered for everything would be a theoretical mass table this
  // port does not have - and the value it returned would be whatever the lookup found next to
  // the missing entry. So it is asserted directly, on both sides.
  {
    struct Probe { int z, a; bool known; const char* why; };
    const Probe probes[] = {
        {26, 56, true, "Fe56, the middle of the table"},
        {1, 1, true, "the proton"},
        {92, 238, true, "U238, the heaviest nuclide with a data file"},
        {1, 20, false, "Z=1 A=20 - twenty neutrons and one proton, not a nuclide"},
        {92, 150, false, "U150 - far proton-rich of the drip line"},
        {110, 290, false, "Z=110 A=290 - past the top of AME2012"},
    };
    for (const Probe& pr : probes) {
      const bool got = data::nuclear_mass_known(pr.a, pr.z);
      if (got != pr.known) {
        std::printf("  FAIL: nuclear_mass_known(%d,%d) is %s, expected %s (%s)\n", pr.a, pr.z,
                    got ? "true" : "false", pr.known ? "true" : "false", pr.why);
        ++fails;
      }
      // A refused nuclide must also return zero rather than a neighbour's mass: every caller
      // in this package tests nuclear_mass_known and a non-zero value there would let a wrong
      // mass through anything that forgot to.
      if (!pr.known && data::nuclear_mass<real_t>(pr.a, pr.z) != real_t(0)) {
        std::printf("  FAIL: nuclear_mass(%d,%d) is %.17g, not 0, for a nuclide outside "
                    "AME2012\n", pr.a, pr.z, data::nuclear_mass<real_t>(pr.a, pr.z));
        ++fails;
      }
    }
  }

  std::printf("\n%s (%d failures)\n", fails ? "FAILED" : "PASSED", fails);
  return fails ? 1 : 0;
}
