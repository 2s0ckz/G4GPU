// The hadronic process framework: model choice, target choice, and what a final state does to a
// track.
//
// `src/physics/hadronic/process.cuh` and `src/physics/hadronic/elastic/elastic_process.cuh`
// against `ref/oracle/elastic_framework.csv`, `elastic_overlap.csv` and `elastic_zanda.csv`.
//
// Six things are checked, and each of them is a place where a plausible rewrite gives a
// different answer:
//
//  1. The recoil threshold. `G4HadronElasticProcess::PostStepDoIt` sets it from the PROTON
//     production cut, and G4RToEConvForProton::Convert is `rangeCut/mm * 100 keV` with no
//     material dependence and no clamping. The oracle has it for seven materials and six range
//     cuts, so a port that made it material-dependent or clamped it to [1 keV, 10 GeV] fails
//     even though it would pass on water at 0.7 mm.
//
//  2. The overlap model choice. Counted over 200,000 draws at seven energies across each of the
//     two overlaps QBBC has, against Geant4's own count and against the closed form
//     P(upper) = (E - E_min,upper)/(E_max,lower - E_min,upper). Both, because agreeing with the
//     formula and disagreeing with Geant4 would mean the formula was read out of the same
//     misreading of the source.
//
//  3. Element selection from the partial cross sections, with Geant4's cross sections fed in, so
//     what is being tested is the selection and not P2's numbers.
//
//  4. Isotope selection, both branches: against Geant4's counted frequencies using its own
//     per-isotope cross sections and abundances, and against the analytic weights.
//
//  5. What `elastic_post_step_do_it` does with a final state: the three primary branches, the
//     recoil threshold, and the non-ionizing deposit. No oracle for this - it is a reading of
//     G4HadronElasticProcess.cc - so it is checked as properties, including the two properties
//     that separate it from the generic FillResult.
//
//  6. `check_result` and `report_energy_momentum` at their 11.1.1 defaults, including the fact
//     that at those defaults neither can fail.
// ---------------------------------------------------------------------------------------------
// This translation unit hands HOST-ONLY functors to __host__ __device__ templates on purpose.
//
// Every function under test is `__host__ __device__` so that the same code runs in a kernel, and
// nvcc instantiates the device side of each template even when nothing launches it. The functors
// this test supplies - the oracle-backed nuclear-mass table, the recoil species map, the CHIPS
// parameter lambdas - read std::vector and std::string and cannot be device code, and they do
// not need to be: what is being checked is the numbers, on the host, against a CSV. So nvcc's
// "calling a __host__ function from a __host__ __device__ function" family is expected here and
// is silenced by number rather than by making the whole build quieter or by adding
// --extended-lambda to the shared build_one_test.bat. A REAL caller of these templates from
// device code supplies device functors and gets the warning if it does not.
//
//   20011 - a __host__ function called from a __host__ __device__ one
//   20013 - the same for a constexpr __host__ function (every C++17 lambda here is one)
//   20015 - the same, reported without a name
#ifdef __CUDACC__
#pragma nv_diag_suppress 20011
#pragma nv_diag_suppress 20013
#pragma nv_diag_suppress 20015
#endif

#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "core/rng.cuh"
#include "physics/hadronic/elastic/elastic_process.cuh"
#include "physics/hadronic/elastic/hadron_elastic.cuh"
#include "physics/hadronic/process.cuh"

using namespace g4gpu;
using namespace g4gpu::physics::hadronic;
using real_t = double;

namespace {

int fails = 0;

void fail(const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  std::printf("  FAIL: ");
  std::vprintf(fmt, ap);
  std::printf("\n");
  va_end(ap);
  ++fails;
}

// ------------------------------------------------------------------------------------------
// CSV helpers
// ------------------------------------------------------------------------------------------

std::string oracle_dir() {
  const char* env = std::getenv("G4GPU_ORACLE");
  return (env != nullptr) ? std::string(env) : std::string("ref/oracle");
}

std::vector<std::string> split(const std::string& s, char sep) {
  std::vector<std::string> out;
  std::size_t a = 0;
  while (true) {
    const std::size_t b = s.find(sep, a);
    if (b == std::string::npos) { out.push_back(s.substr(a)); break; }
    out.push_back(s.substr(a, b - a));
    a = b + 1;
  }
  return out;
}

/// Read a `name,value,unit` file into a lookup.
struct NameValue {
  std::vector<std::string> names;
  std::vector<double> values;
  /// `get`'s `found` flag is how a caller asks whether a key is present, and every caller here
  /// wants the value as well - so there is no separate `has`. One existed and nothing called it.
  double get(const char* n, bool* found = nullptr) const {
    for (std::size_t i = 0; i < names.size(); ++i) {
      if (names[i] == n) { if (found) { *found = true; } return values[i]; }
    }
    if (found) { *found = false; }
    return 0.0;
  }
};

NameValue load_name_value(const std::string& path) {
  NameValue nv;
  FILE* f = std::fopen(path.c_str(), "r");
  if (f == nullptr) { return nv; }
  char line[512];
  if (std::fgets(line, sizeof line, f) == nullptr) { std::fclose(f); return nv; }
  while (std::fgets(line, sizeof line, f) != nullptr) {
    std::string s(line);
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) { s.pop_back(); }
    const auto parts = split(s, ',');
    if (parts.size() < 2) { continue; }
    nv.names.push_back(parts[0]);
    nv.values.push_back(std::atof(parts[1].c_str()));
  }
  std::fclose(f);
  return nv;
}

std::vector<std::vector<std::string>> load_rows(const std::string& path,
                                                std::vector<std::string>* header) {
  std::vector<std::vector<std::string>> out;
  FILE* f = std::fopen(path.c_str(), "r");
  if (f == nullptr) { return out; }
  char line[8192];
  if (std::fgets(line, sizeof line, f) != nullptr && header != nullptr) {
    std::string s(line);
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) { s.pop_back(); }
    *header = split(s, ',');
  }
  while (std::fgets(line, sizeof line, f) != nullptr) {
    std::string s(line);
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) { s.pop_back(); }
    if (s.empty()) { continue; }
    out.push_back(split(s, ','));
  }
  std::fclose(f);
  return out;
}

// ------------------------------------------------------------------------------------------
// 1. The recoil threshold from the proton production cut.
// ------------------------------------------------------------------------------------------

void check_proton_cut(const NameValue& nv) {
  std::printf("== the recoil threshold: G4RToEConvForProton::Convert ==\n");
  const double rcuts[] = {0.0, 0.1, 0.7, 1.0, 10.0, 100.0};
  double worst = 0;
  int n = 0;
  for (double rc : rcuts) {
    char key[128];
    std::snprintf(key, sizeof key, "convertRangeToEnergy_proton_%.4g_mm", rc);
    bool found = false;
    const double g4 = nv.get(key, &found);
    if (!found) { fail("%s missing from elastic_framework.csv", key); continue; }
    const double ours = elastic::proton_recoil_cut_energy<real_t>(rc);
    const double dev = (g4 != 0.0) ? std::fabs(ours / g4 - 1.0) : std::fabs(ours);
    if (dev > worst) { worst = dev; }
    ++n;
    std::printf("  range cut %8.4g mm -> %-14.8g MeV   (G4 %.8g)\n", rc, ours, g4);
  }
  if (n != 6) { fail("expected 6 range cuts, compared %d", n); }
  if (worst > 0.0) { fail("proton cut conversion is not exact: worst %.3g", worst); }

  // Material independence, from the seven materials the dumper builds. A port that read the
  // cut out of a per-material table would pass a one-material check and fail this one.
  int nmat = 0;
  double spread_worst = 0;
  for (std::size_t i = 0; i < nv.names.size(); ++i) {
    if (nv.names[i].rfind("protonCutEnergy_", 0) != 0) { continue; }
    ++nmat;
    const double ours = elastic::proton_recoil_cut_energy<real_t>(0.7);
    const double dev = std::fabs(ours / nv.values[i] - 1.0);
    if (dev > spread_worst) { spread_worst = dev; }
  }
  std::printf("  %d materials, all at the 0.7 mm default: worst deviation %.3g\n", nmat,
              spread_worst);
  if (nmat < 5) { fail("expected at least 5 per-material cuts, found %d", nmat); }
  if (spread_worst > 0.0) { fail("the proton cut is not material-independent in the port"); }
  std::printf("\n");
}

// ------------------------------------------------------------------------------------------
// 2. The overlap model choice.
// ------------------------------------------------------------------------------------------

void check_overlap(const std::string& dir) {
  std::printf("== G4EnergyRangeManager overlap: which model, and how often ==\n");
  std::vector<std::string> hdr;
  const auto rows = load_rows(dir + "/elastic_overlap.csv", &hdr);
  if (rows.empty()) { fail("cannot read elastic_overlap.csv"); return; }

  std::printf("  %-10s %10s %12s %12s %12s %10s\n", "overlap", "E (MeV)", "G4 measured",
              "port measured", "formula", "worst dev");
  double worst_vs_g4 = 0, worst_vs_formula = 0;
  int n = 0;
  for (const auto& r : rows) {
    if (r.size() < 10) { continue; }
    const double lo_min = std::atof(r[1].c_str()), lo_max = std::atof(r[2].c_str());
    const double up_min = std::atof(r[3].c_str()), up_max = std::atof(r[4].c_str());
    const double ekin = std::atof(r[5].c_str());
    const int ntrial = std::atoi(r[6].c_str());
    const double p_g4 = std::atof(r[8].c_str());
    const double p_formula = std::atof(r[9].c_str());

    // The dump registers each pair both ways round, and the row name says which: `_rev` rows put
    // the UPPER model first, which is what makes `emi1 < emi2` and takes the other branch of
    // G4EnergyRangeManager's cou == 2 arithmetic. Both branches have to be walked or half of
    // that function is untested - inverting the port's first ternary changed no test result until
    // these rows existed.
    const bool upper_first = (r[0].size() >= 4 &&
                              r[0].compare(r[0].size() - 4, 4, "_rev") == 0);
    ModelRange<real_t> models[2];
    const int i_lo = upper_first ? 1 : 0, i_up = upper_first ? 0 : 1;
    models[i_lo].min_energy = lo_min; models[i_lo].max_energy = lo_max;
    models[i_up].min_energy = up_min; models[i_up].max_energy = up_max;

    Philox<real_t> rng(7u, 11u, static_cast<uint32_t>(ekin));
    int n_upper = 0;
    for (int i = 0; i < ntrial; ++i) {
      const ModelSelection s = choose_hadronic_interaction<real_t>(models, 2, ekin, 1, rng);
      if (s.status != ModelChoice::kOk) {
        fail("model choice failed with status %d at %g MeV", int(s.status), ekin);
        break;
      }
      if (s.index == i_up) { ++n_upper; }
    }
    const double p_ours = double(n_upper) / double(ntrial);
    const double ours_formula =
        overlap_probability_upper<real_t>(lo_max, up_min, ekin);
    const double d1 = std::fabs(p_ours - p_g4);
    const double d2 = std::fabs(p_ours - ours_formula);
    if (d1 > worst_vs_g4) { worst_vs_g4 = d1; }
    if (d2 > worst_vs_formula) { worst_vs_formula = d2; }
    ++n;
    std::printf("  %-10s %10.4g %12.6f %12.6f %12.6f %10.3g\n", r[0].c_str(), ekin, p_g4,
                p_ours, p_formula, d1);
    if (std::fabs(ours_formula - p_formula) > 1e-14) {
      fail("the port's closed form disagrees with the dump's at %g MeV: %.17g vs %.17g", ekin,
           ours_formula, p_formula);
    }
    // Exact at the ends of the overlap: at E = E_min,upper no draw can select the upper model
    // and at E = E_max,lower every draw must, because the comparison is `< rand*(...)` with a
    // uniform on (0,1). Those two are not statistical and are checked as equalities.
    if (ekin == up_min && n_upper != 0) {
      fail("at the bottom of the overlap (%g MeV) the upper model was picked %d times", ekin,
           n_upper);
    }
    if (ekin == lo_max && n_upper != ntrial) {
      fail("at the top of the overlap (%g MeV) the upper model was picked only %d/%d times",
           ekin, n_upper, ntrial);
    }
    (void)up_max;
  }
  // Two independent 200,000-sample binomials: sigma of the difference is at most
  // sqrt(2*0.25/200000) = 0.0016, so 5 sigma is 0.008.
  const double tol = 0.008;
  std::printf("  %d points; worst |port - G4| %.3g, worst |port - formula| %.3g, tol %.3g\n", n,
              worst_vs_g4, worst_vs_formula, tol);
  if (n != 28) { fail("expected 28 overlap points, compared %d", n); }
  if (worst_vs_g4 > tol) { fail("overlap frequency differs from Geant4 by %.3g", worst_vs_g4); }
  if (worst_vs_formula > tol) {
    fail("overlap frequency differs from the closed form by %.3g", worst_vs_formula);
  }

  // The three non-overlap statuses, which must be reported and never silently defaulted.
  {
    ModelRange<real_t> m3[3];
    for (int i = 0; i < 3; ++i) { m3[i].min_energy = 0; m3[i].max_energy = 1e6; }
    Philox<real_t> rng(1u, 1u, 1u);
    const ModelSelection s = choose_hadronic_interaction<real_t>(m3, 3, 1000.0, 1, rng);
    if (s.status != ModelChoice::kMoreThanTwoCompeting || s.index != -1) {
      fail("three competing models should report kMoreThanTwoCompeting, got %d/%d",
           int(s.status), s.index);
    }
    ModelRange<real_t> m2[2];
    m2[0].min_energy = 0; m2[0].max_energy = 1e6;
    m2[1].min_energy = 10; m2[1].max_energy = 1e5;  // fully inside the first
    const ModelSelection s2 = choose_hadronic_interaction<real_t>(m2, 2, 1000.0, 1, rng);
    if (s2.status != ModelChoice::kFullyOverlapping || s2.index != -1) {
      fail("nested ranges should report kFullyOverlapping, got %d/%d", int(s2.status), s2.index);
    }
    m2[0].min_energy = 0; m2[0].max_energy = 10;
    m2[1].min_energy = 100; m2[1].max_energy = 200;
    const ModelSelection s3 = choose_hadronic_interaction<real_t>(m2, 2, 50.0, 1, rng);
    if (s3.status != ModelChoice::kNoModelInRange || s3.index != -1) {
      fail("a gap should report kNoModelInRange, got %d/%d", int(s3.status), s3.index);
    }
    // One registered model is used whatever the energy: Geant4's shortcut returns it without
    // looking at the range, and that is the branch QBBC's elastic takes for every particle.
    m2[0].min_energy = 1e5; m2[0].max_energy = 1e6;
    const ModelSelection s4 = choose_hadronic_interaction<real_t>(m2, 1, 1.0, 1, rng);
    if (s4.status != ModelChoice::kOk || s4.index != 0) {
      fail("a single registered model must be used out of range, got %d/%d", int(s4.status),
           s4.index);
    }
  }
  // The per-nucleon energy for an ion: a 6 GeV alpha is a 1.5 GeV/n projectile, so it must land
  // at the TOP of a 1 - 1.5 GeV/n overlap and not above it.
  {
    ModelRange<real_t> m2[2];
    m2[0].min_energy = 0;    m2[0].max_energy = 1500;
    m2[1].min_energy = 1000; m2[1].max_energy = 6000;
    Philox<real_t> rng(3u, 3u, 3u);
    int n_upper = 0;
    const int kN = 20000;
    for (int i = 0; i < kN; ++i) {
      const ModelSelection s = choose_hadronic_interaction<real_t>(m2, 2, 6000.0, 4, rng);
      if (s.index == 1) { ++n_upper; }
    }
    if (n_upper != kN) {
      fail("a 6 GeV alpha (1.5 GeV/n) must always pick the upper model; picked it %d/%d times",
           n_upper, kN);
    }
    // And without the per-nucleon division it would be out of both ranges entirely.
    const ModelSelection s = choose_hadronic_interaction<real_t>(m2, 2, 6000.0, 1, rng);
    if (s.status != ModelChoice::kOk || s.index != 1) {
      fail("a 6 GeV proton is at the top of the upper range, expected model 1, got %d/%d",
           int(s.status), s.index);
    }
  }
  std::printf("\n");
}

// ------------------------------------------------------------------------------------------
// 3 and 4. Element and isotope selection.
// ------------------------------------------------------------------------------------------

struct IsoSpec { int a; double abundance; double xs; int count; };

struct ElmRow {
  std::string material;
  double ekin = 0;
  int nelm = 0, i = 0, z = 0;
  double natoms = 0, xs_per_atom = 0, cum = 0;
  int ntrial = 0, n_selected = 0;
  std::vector<IsoSpec> isos;
};

/// A functor over a fixed per-element / per-isotope table, so the selection logic is what is
/// under test and Geant4's own cross sections are the input. `element_wise` picks which branch
/// of SampleZandA is exercised.
struct TableXs {
  const std::vector<ElmRow>* rows = nullptr;
  std::size_t first = 0, n = 0;
  bool element_wise = false;

  __host__ real_t element_xs(int z, real_t, int) const {
    for (std::size_t k = first; k < first + n; ++k) {
      if ((*rows)[k].z == z) { return (*rows)[k].xs_per_atom; }
    }
    return real_t(0);
  }
  __host__ real_t iso_xs(int z, int a, real_t, int) const {
    for (std::size_t k = first; k < first + n; ++k) {
      if ((*rows)[k].z != z) { continue; }
      for (const IsoSpec& s : (*rows)[k].isos) { if (s.a == a) { return s.xs; } }
    }
    return real_t(0);
  }
  __host__ bool is_element_applicable(int, real_t, int) const { return element_wise; }
  template <typename Rng>
  __host__ int select_isotope(int z, int, real_t, int, Rng&) const {
    // Only reached in the element-wise branch, which the isotope test does not use.
    for (std::size_t k = first; k < first + n; ++k) {
      if ((*rows)[k].z == z) { return (*rows)[k].isos[0].a; }
    }
    return 0;
  }
};

void check_zanda(const std::string& dir) {
  std::printf("== G4CrossSectionDataStore: ComputeCrossSection and SampleZandA ==\n");
  std::vector<std::string> hdr;
  const auto raw = load_rows(dir + "/elastic_zanda.csv", &hdr);
  if (raw.empty()) { fail("cannot read elastic_zanda.csv"); return; }

  std::vector<ElmRow> rows;
  bool have_iso_xs = true;
  for (const auto& r : raw) {
    if (r.size() < 14) { continue; }
    ElmRow e;
    e.material = r[0];
    e.ekin = std::atof(r[1].c_str());
    e.nelm = std::atoi(r[2].c_str());
    e.i = std::atoi(r[4].c_str());
    e.z = std::atoi(r[5].c_str());
    e.natoms = std::atof(r[6].c_str());
    e.xs_per_atom = std::atof(r[7].c_str());
    e.cum = std::atof(r[8].c_str());
    e.ntrial = std::atoi(r[9].c_str());
    e.n_selected = std::atoi(r[10].c_str());
    for (const std::string& tok : split(r[13], ';')) {
      const auto f = split(tok, ':');
      IsoSpec s;
      if (f.size() == 4) {
        s.a = std::atoi(f[0].c_str());
        s.abundance = std::atof(f[1].c_str());
        s.xs = std::atof(f[2].c_str());
        s.count = std::atoi(f[3].c_str());
      } else if (f.size() == 2) {
        s.a = std::atoi(f[0].c_str());
        s.count = std::atoi(f[1].c_str());
        have_iso_xs = false;
      } else {
        continue;
      }
      e.isos.push_back(s);
    }
    rows.push_back(e);
  }
  if (!have_iso_xs) {
    fail("elastic_zanda.csv has no per-isotope abundance/cross section columns - regenerate "
         "the oracle with the current ref/dump/dump_elastic.cc");
    return;
  }

  // Group rows into (material, energy) blocks - one per material state.
  std::printf("  %-22s %8s %3s %10s %12s %12s %10s\n", "material", "E (MeV)", "Z", "G4 frac",
              "port frac", "expected", "dev");
  double worst_elm = 0, worst_iso = 0, worst_xs = 0;
  int n_elm = 0, n_iso = 0;
  std::size_t k = 0;
  while (k < rows.size()) {
    const std::size_t first = k;
    const int nelm = rows[first].nelm;
    if (nelm <= 0 || first + std::size_t(nelm) > rows.size()) { break; }
    const std::size_t n = std::size_t(nelm);
    k += n;

    std::vector<int> zs(n);
    std::vector<real_t> dens(n), nat(n, 0);
    std::vector<int> niso(n), isooff(n);
    std::vector<bool> natural(n, true);
    std::vector<int> iso_a;
    std::vector<real_t> iso_ab;
    for (std::size_t j = 0; j < n; ++j) {
      zs[j] = rows[first + j].z;
      dens[j] = rows[first + j].natoms;
      niso[j] = int(rows[first + j].isos.size());
      isooff[j] = int(iso_a.size());
      for (const IsoSpec& s : rows[first + j].isos) {
        iso_a.push_back(s.a);
        iso_ab.push_back(s.abundance);
      }
    }
    std::vector<char> natural_c(n, 0);  // std::vector<bool> has no data()
    MaterialComposition<real_t> mat;
    mat.n_elements = int(n);
    mat.element_z = zs.data();
    mat.n_atoms_per_volume = dens.data();
    mat.n_isotopes = niso.data();
    mat.isotope_offset = isooff.data();
    mat.natural_abundance = reinterpret_cast<const bool*>(natural_c.data());
    mat.isotope_a = iso_a.data();
    mat.isotope_abundance = iso_ab.data();

    TableXs xs;
    xs.rows = &rows;
    xs.first = first;
    xs.n = n;
    xs.element_wise = false;  // the CHIPS data sets are isotope-wise, as Geant4's is here

    // The total cross section and the cumulative array. Compared against Geant4's own, which is
    // a check on ComputeCrossSection and not only on the selection.
    std::vector<real_t> xsecelm(n, 0), xseciso(16, 0);
    const real_t total = compute_cross_section<real_t>(mat, xs, 2212, rows[first].ekin,
                                                       xsecelm.data());
    for (std::size_t j = 0; j < n; ++j) {
      const double d = std::fabs(xsecelm[j] / rows[first + j].cum - 1.0);
      if (d > worst_xs) { worst_xs = d; }
    }
    {
      const double d = std::fabs(total / rows[first].cum * 0 + total / xsecelm[n - 1] - 1.0);
      if (d > worst_xs) { worst_xs = d; }
    }

    // Element and isotope frequencies, counted.
    const int ntrial = rows[first].ntrial;
    std::vector<int> counts(n, 0);
    std::vector<std::vector<int>> iso_counts(n);
    for (std::size_t j = 0; j < n; ++j) { iso_counts[j].assign(niso[j], 0); }
    Philox<real_t> rng(101u, 202u, static_cast<uint32_t>(rows[first].ekin));
    for (int t = 0; t < ntrial; ++t) {
      const ZandA za = sample_z_and_a<real_t>(mat, xs, 2212, rows[first].ekin, total,
                                             xsecelm.data(), rng, xseciso.data());
      counts[za.element_index] += 1;
      for (int j = 0; j < niso[za.element_index]; ++j) {
        if (iso_a[isooff[za.element_index] + j] == za.a) {
          ++iso_counts[za.element_index][j];
          break;
        }
      }
    }
    for (std::size_t j = 0; j < n; ++j) {
      const double g4 = double(rows[first + j].n_selected) / double(ntrial);
      const double ours = double(counts[j]) / double(ntrial);
      const double lo = (j == 0) ? 0.0 : rows[first + j - 1].cum;
      const double expected = (rows[first + j].cum - lo) / rows[first + n - 1].cum;
      const double d = std::fabs(ours - g4);
      if (d > worst_elm) { worst_elm = d; }
      ++n_elm;
      std::printf("  %-22s %8.4g %3d %10.6f %12.6f %12.6f %10.3g\n",
                  rows[first].material.c_str(), rows[first].ekin, zs[j], g4, ours, expected, d);
      if (std::fabs(ours - expected) > 0.01) {
        fail("element %d frequency %.6f is not the cross-section share %.6f", zs[j], ours,
             expected);
      }
      // Isotopes: against Geant4's count and against abundance*xs normalised.
      double wsum = 0;
      for (int q = 0; q < niso[j]; ++q) {
        wsum += rows[first + j].isos[q].abundance * rows[first + j].isos[q].xs;
      }
      for (int q = 0; q < niso[j]; ++q) {
        const double g4i = double(rows[first + j].isos[q].count) / double(counts[j] > 0 ? 1 : 1);
        const double g4frac = (rows[first + j].n_selected > 0)
                                  ? double(rows[first + j].isos[q].count) /
                                        double(rows[first + j].n_selected)
                                  : 0.0;
        const double ourfrac = (counts[j] > 0) ? double(iso_counts[j][q]) / double(counts[j])
                                               : 0.0;
        const double wfrac = (wsum > 0) ? rows[first + j].isos[q].abundance *
                                              rows[first + j].isos[q].xs / wsum
                                        : 0.0;
        (void)g4i;
        const double di = std::fabs(ourfrac - g4frac);
        const double dw = std::fabs(ourfrac - wfrac);
        if (di > worst_iso) { worst_iso = di; }
        ++n_iso;
        if (dw > 0.01) {
          fail("Z=%d A=%d isotope frequency %.6f is not the abundance*xs share %.6f", zs[j],
               rows[first + j].isos[q].a, ourfrac, wfrac);
        }
      }
    }
  }
  // 200,000 draws split across elements: the smallest element share here is 11%, so its
  // binomial sigma is about 0.0007 and 5 sigma over the difference of two counts is 0.005. The
  // isotope fractions are conditional on the element and the rarest (Pb-204, 1.4%) has about
  // 300 counts, so its sigma is 0.006 - the isotope tolerance has to be looser and is stated
  // separately rather than hidden in one number.
  std::printf("  %d element frequencies, %d isotope frequencies\n", n_elm, n_iso);
  std::printf("  cumulative cross sections: worst relative deviation %.3g\n", worst_xs);
  std::printf("  element frequency: worst |port - G4| %.3g (tol 0.005)\n", worst_elm);
  std::printf("  isotope frequency: worst |port - G4| %.3g (tol 0.03)\n", worst_iso);
  if (n_elm != 15) { fail("expected 15 element rows, compared %d", n_elm); }
  if (worst_xs > 1e-12) {
    fail("ComputeCrossSection's cumulative array differs from Geant4's by %.3g", worst_xs);
  }
  if (worst_elm > 0.005) { fail("element selection frequency off by %.3g", worst_elm); }
  if (worst_iso > 0.03) { fail("isotope selection frequency off by %.3g", worst_iso); }

  // The single-element shortcut: Geant4 returns element 0 WITHOUT drawing. A port that always
  // drew would be right on average and desynchronised on every subsequent sample, so the check
  // is on the random stream and not on the answer.
  {
    int z1[1] = {6};
    real_t d1[1] = {1.0};
    int ni[1] = {1};
    int io[1] = {0};
    char nb[1] = {0};
    int ia[1] = {12};
    real_t iab[1] = {1.0};
    MaterialComposition<real_t> one;
    one.n_elements = 1;
    one.element_z = z1;
    one.n_atoms_per_volume = d1;
    one.n_isotopes = ni;
    one.isotope_offset = io;
    one.natural_abundance = reinterpret_cast<const bool*>(nb);
    one.isotope_a = ia;
    one.isotope_abundance = iab;
    TableXs xs;
    xs.rows = &rows;
    xs.first = 0;
    xs.n = 0;
    real_t xe[1] = {1.0}, xi[1] = {0};
    Philox<real_t> a(5u, 5u, 5u), b(5u, 5u, 5u);
    const real_t first_of_b = b.uniform();
    const ZandA za = sample_z_and_a<real_t>(one, xs, 2212, 100.0, real_t(1), xe, a, xi);
    const real_t next_of_a = a.uniform();
    if (za.z != 6 || za.a != 12) {
      fail("a single-element material must return its element, got Z=%d A=%d", za.z, za.a);
    }
    if (next_of_a != first_of_b) {
      fail("a single-element material consumed a random number; the stream is desynchronised");
    }
  }
  std::printf("\n");
}

// ------------------------------------------------------------------------------------------
// 5. What the elastic process does with a final state.
// ------------------------------------------------------------------------------------------

/// A stub model, so the process can be driven to each branch on demand.
struct StubModel {
  real_t efinal = 0;
  real_t edep = 0;
  HadFinalStateStatus status = HadFinalStateStatus::kIsAlive;
  int nsec = 0;
  real_t sec_ekin = 0;
  Vec3<real_t> dir{real_t(0), real_t(0), real_t(1)};

  template <typename Rng, int kCap>
  __host__ void operator()(const HadProjectile<real_t>&, const HadNucleus&, real_t, Rng&,
                           HadFinalState<real_t, kCap>* out) const {
    out->clear();
    out->status = status;
    out->energy_change = efinal;
    out->local_energy_deposit = edep;
    out->momentum_change = dir;
    for (int i = 0; i < nsec; ++i) {
      HadSecondary<real_t> s;
      s.pdg = 2212;
      s.mass = 938.272013;
      s.kin_energy = sec_ekin;
      s.direction = Vec3<real_t>{real_t(0), real_t(0), real_t(1)};
      out->add_secondary(s);
    }
  }
};

void check_elastic_process() {
  std::printf("== G4HadronElasticProcess::PostStepDoIt semantics ==\n");
  HadProjectile<real_t> pro;
  pro.pdg = 2212;
  pro.baryon_number = 1;
  pro.charge = 1;
  pro.mass = 938.272013;
  pro.kin_energy = 100.0;
  HadNucleus tgt{8, 16, 0};
  const Vec3<real_t> indir = normalize(Vec3<real_t>{real_t(0.3), real_t(-0.4), real_t(0.8)});
  HadFinalState<real_t, 8> scratch;
  Philox<real_t> rng(1u, 2u, 3u);
  const real_t tcut = elastic::proton_recoil_cut_energy<real_t>(real_t(0.7));

  // (a) the primary survives: alive, direction rotated, no deposit.
  {
    StubModel m;
    m.efinal = 90.0;
    m.nsec = 1;
    m.sec_ekin = 5.0;
    m.dir = Vec3<real_t>{real_t(0), real_t(0), real_t(1)};
    const auto r = elastic::elastic_post_step_do_it<real_t, 8>(
        pro, tgt, indir, real_t(1), true, false, HadXsType::kNoIntegral, real_t(0), real_t(0),
        tcut, m, rng, &scratch);
    if (!r.interacted) { fail("(a) the interaction was rejected"); }
    if (r.status != TrackStatusChange::kAlive) { fail("(a) status is not alive"); }
    if (r.energy != real_t(90)) { fail("(a) energy %g, expected 90", r.energy); }
    // The model returned +z, so rotateUz(indir) must give back indir exactly.
    if (std::fabs(r.momentum_direction.z - indir.z) > 1e-15 ||
        std::fabs(r.momentum_direction.x - indir.x) > 1e-15) {
      fail("(a) rotateUz of +z did not return the incident direction");
    }
    if (r.n_secondaries != 1) { fail("(a) the recoil above %g MeV was not emitted", tcut); }
    if (r.local_energy_deposit != real_t(0)) { fail("(a) unexpected deposit %g", r.local_energy_deposit); }
    std::printf("  (a) primary alive, recoil %g MeV > cut %g MeV: emitted, deposit %g\n",
                m.sec_ekin, tcut, r.local_energy_deposit);
  }

  // (b) the recoil is below the cut: it becomes a LOCAL and NON-IONIZING deposit, and both
  //     deposits are the same number.
  {
    StubModel m;
    m.efinal = 99.99;
    m.nsec = 1;
    m.sec_ekin = tcut * real_t(0.5);
    const auto r = elastic::elastic_post_step_do_it<real_t, 8>(
        pro, tgt, indir, real_t(1), true, false, HadXsType::kNoIntegral, real_t(0), real_t(0),
        tcut, m, rng, &scratch);
    if (r.n_secondaries != 0) { fail("(b) a sub-threshold recoil was emitted"); }
    if (std::fabs(r.local_energy_deposit - m.sec_ekin) > 1e-15) {
      fail("(b) deposit %g, expected the recoil energy %g", r.local_energy_deposit, m.sec_ekin);
    }
    if (r.non_ionizing_energy_deposit != r.local_energy_deposit) {
      fail("(b) the non-ionizing deposit %g differs from the local deposit %g",
           r.non_ionizing_energy_deposit, r.local_energy_deposit);
    }
    if (!r.recoil_below_threshold) { fail("(b) the below-threshold path was not reported"); }
    std::printf("  (b) recoil %g MeV <= cut: deposited locally, non-ionizing = local = %g\n",
                m.sec_ekin, r.local_energy_deposit);
  }

  // (c) the primary stops. With at-rest processes -> StopButAlive; without -> StopAndKill.
  //     This is the hook a stopped pi- is captured through rather than deleted.
  {
    StubModel m;
    m.efinal = 0.0;
    const auto r1 = elastic::elastic_post_step_do_it<real_t, 8>(
        pro, tgt, indir, real_t(1), true, true, HadXsType::kNoIntegral, real_t(0), real_t(0),
        tcut, m, rng, &scratch);
    const auto r2 = elastic::elastic_post_step_do_it<real_t, 8>(
        pro, tgt, indir, real_t(1), true, false, HadXsType::kNoIntegral, real_t(0), real_t(0),
        tcut, m, rng, &scratch);
    if (r1.status != TrackStatusChange::kStopButAlive) {
      fail("(c) a stopped primary with at-rest processes must be StopButAlive, got %d",
           int(r1.status));
    }
    if (r2.status != TrackStatusChange::kStopAndKill) {
      fail("(c) a stopped primary with no at-rest process must be StopAndKill, got %d",
           int(r2.status));
    }
    std::printf("  (c) zero final energy: StopButAlive with at-rest processes, "
                "StopAndKill without\n");
  }

  // (d) THE DIFFERENCE FROM FillResult: the elastic process ignores GetStatusChange. A model
  //     that asks for stopAndKill while leaving a positive energy is kept alive here and killed
  //     by the generic FillResult. Both are checked, on the same final state.
  {
    StubModel m;
    m.efinal = 50.0;
    m.status = HadFinalStateStatus::kStopAndKill;
    const auto r = elastic::elastic_post_step_do_it<real_t, 8>(
        pro, tgt, indir, real_t(1), true, false, HadXsType::kNoIntegral, real_t(0), real_t(0),
        tcut, m, rng, &scratch);
    if (r.status != TrackStatusChange::kAlive || r.energy != real_t(50)) {
      fail("(d) the elastic process must decide from the final energy, not GetStatusChange: "
           "got status %d energy %g", int(r.status), r.energy);
    }
    m(pro, tgt, tcut, rng, &scratch);
    const auto g = fill_result<real_t, 8, 8>(scratch, indir, real_t(0), real_t(1), false,
                                             nullptr);
    if (g.status != TrackStatusChange::kStopAndKill || g.energy != real_t(0)) {
      fail("(d) the generic FillResult must obey GetStatusChange: got status %d energy %g",
           int(g.status), g.energy);
    }
    std::printf("  (d) stopAndKill with 50 MeV left: elastic keeps it alive at 50 MeV, "
                "FillResult kills it\n");
  }

  // (d2) FillResult's branch ORDER, which (d) cannot see.
  //
  //      Geant4 tests `GetStatusChange() == stopAndKill` FIRST and only then reads a zero final
  //      energy as a stop. The two orders differ in exactly one case: stopAndKill together with
  //      zero energy, on a particle that HAS at-rest processes. Geant4 kills it; the swapped
  //      order makes it fStopButAlive, which hands the track to the at-rest chain a model
  //      explicitly asked to end. Test (d) uses a positive energy, so both orders agree there
  //      and swapping them in the port changed no test result until this case existed. A
  //      stopped pi- is the particle this matters for (P12's absorption competes on that hook).
  {
    StubModel m;
    m.efinal = 0.0;
    m.status = HadFinalStateStatus::kStopAndKill;
    m(pro, tgt, tcut, rng, &scratch);
    const auto g = fill_result<real_t, 8, 8>(scratch, indir, real_t(0), real_t(1), true,
                                             nullptr);
    if (g.status != TrackStatusChange::kStopAndKill) {
      fail("(d2) stopAndKill with zero energy and at-rest processes must still be killed: "
           "got status %d", int(g.status));
    }
    // And the control: the same zero energy WITHOUT stopAndKill is fStopButAlive.
    m.status = HadFinalStateStatus::kIsAlive;
    m(pro, tgt, tcut, rng, &scratch);
    const auto g2 = fill_result<real_t, 8, 8>(scratch, indir, real_t(0), real_t(1), true,
                                              nullptr);
    if (g2.status != TrackStatusChange::kStopButAlive) {
      fail("(d2) zero energy without stopAndKill must be StopButAlive when at-rest processes "
           "exist: got status %d", int(g2.status));
    }
    std::printf("  (d2) zero energy: stopAndKill kills even with at-rest processes; without it, "
                "StopButAlive\n");
  }

  // (e) a zero-energy or non-alive track is returned untouched, before any random number.
  {
    StubModel m;
    m.efinal = 1.0;
    HadProjectile<real_t> dead = pro;
    dead.kin_energy = 0.0;
    Philox<real_t> a(9u, 9u, 9u), b(9u, 9u, 9u);
    const real_t first_of_b = b.uniform();
    const auto r = elastic::elastic_post_step_do_it<real_t, 8>(
        dead, tgt, indir, real_t(1), true, false, HadXsType::kNoIntegral, real_t(0), real_t(0),
        tcut, m, a, &scratch);
    if (r.interacted) { fail("(e) a zero-energy track interacted"); }
    if (a.uniform() != first_of_b) { fail("(e) a zero-energy track consumed a random number"); }
    const auto r2 = elastic::elastic_post_step_do_it<real_t, 8>(
        pro, tgt, indir, real_t(1), false, false, HadXsType::kNoIntegral, real_t(0), real_t(0),
        tcut, m, a, &scratch);
    if (r2.interacted) { fail("(e) a non-alive track interacted"); }
    std::printf("  (e) zero energy and non-alive tracks return unchanged, consuming nothing\n");
  }

  // (f) the integral-cross-section rejection. With xs_now == 0 and a positive xs at the step
  //     start, every draw must reject; with xs_now >= xs_at_step_start none may.
  {
    StubModel m;
    m.efinal = 90.0;
    int rejected = 0;
    for (int i = 0; i < 1000; ++i) {
      const auto r = elastic::elastic_post_step_do_it<real_t, 8>(
          pro, tgt, indir, real_t(1), true, false, HadXsType::kTwoPeaks, real_t(0), real_t(1),
          tcut, m, rng, &scratch);
      if (r.rejected_by_integral_xs) { ++rejected; }
    }
    if (rejected != 1000) { fail("(f) a zero cross section rejected only %d/1000", rejected); }
    int accepted = 0;
    for (int i = 0; i < 1000; ++i) {
      const auto r = elastic::elastic_post_step_do_it<real_t, 8>(
          pro, tgt, indir, real_t(1), true, false, HadXsType::kTwoPeaks, real_t(1), real_t(1),
          tcut, m, rng, &scratch);
      if (r.interacted) { ++accepted; }
    }
    if (accepted != 1000) {
      fail("(f) an unchanged cross section rejected %d/1000", 1000 - accepted);
    }
    // And a neutron never does the rejection at all: hadronic_xs_type is kNoIntegral for a
    // neutral particle whatever the integral flag says, which is what makes the neutron's
    // elastic usable as a sub-process of G4NeutronGeneralProcess.
    const HadXsType nt = hadronic_xs_type<real_t>(2112, real_t(0), real_t(939.56536), 0, false,
                                                  true);
    if (nt != HadXsType::kNoIntegral) {
      fail("(f) a neutron's cross-section type should be kNoIntegral, got %d", int(nt));
    }
    const HadXsType pt = hadronic_xs_type<real_t>(2212, real_t(1), real_t(938.272013), 0, false,
                                                  true);
    if (pt != HadXsType::kTwoPeaks) {
      fail("(f) a proton's cross-section type should be kTwoPeaks, got %d", int(pt));
    }
    const HadXsType pit = hadronic_xs_type<real_t>(-211, real_t(-1), real_t(139.57018), 0, false,
                                                   true);
    if (pit != HadXsType::kTwoPeaks) {
      fail("(f) a pi-'s cross-section type should be kTwoPeaks, got %d", int(pit));
    }
    std::printf("  (f) integral rejection: 1000/1000 rejected at zero xs, 0/1000 at equal xs; "
                "neutron type = kNoIntegral\n");
  }
  std::printf("\n");
}

// ------------------------------------------------------------------------------------------
// 6. CheckResult and the ep report at their defaults.
// ------------------------------------------------------------------------------------------

void check_checks(const NameValue& nv) {
  std::printf("== CheckResult and the energy-momentum report, at 11.1.1's defaults ==\n");
  bool f1 = false, f2 = false;
  const double rel = nv.get("fatalEnergyCheckRelative", &f1);
  const double abs_ = nv.get("fatalEnergyCheckAbsolute", &f2);
  if (!f1 || !f2) { fail("fatal energy-check levels missing from elastic_framework.csv"); }
  const FatalEnergyCheckLevels<real_t> lv;
  if (lv.relative != rel || lv.absolute != abs_) {
    fail("fatal energy-check levels are (%.17g, %.17g), Geant4's are (%.17g, %.17g)",
         lv.relative, lv.absolute, rel, abs_);
  }
  std::printf("  GetFatalEnergyCheckLevels = (%.3g relative, %.4g MeV absolute)\n", lv.relative,
              lv.absolute);
  if (nv.get("epCheckRelativeDefaultIsDBL_MAX") != 1.0 ||
      nv.get("epCheckAbsoluteDefaultIsDBL_MAX") != 1.0) {
    fail("the ep check levels are not both DBL_MAX by default in Geant4");
  }
  if (kDefaultEpReportLevel != 0) { fail("the port's default ep report level is not 0"); }
  std::printf("  epReportLevel default = %d (the report never runs), epCheckLevels = "
              "(inf, inf)\n", kDefaultEpReportLevel);
  if (nv.get("recoilEnergyThresholdDefault") != 0.0) {
    fail("G4HadronicInteraction's default recoil threshold is not 0");
  }

  HadProjectile<real_t> pro;
  pro.pdg = 2212;
  pro.baryon_number = 1;
  pro.charge = 1;
  pro.mass = 938.272013;
  pro.kin_energy = 100.0;
  const real_t mtarget = 14895.081534649964;  // O-16, from elastic_nuclear_mass.csv

  // A conserving elastic final state with a recoil: the primary keeps E - erec and the recoil
  // carries erec, so deltaE is zero to rounding.
  {
    HadFinalState<real_t, 8> r;
    r.clear();
    r.energy_change = 95.0;
    HadSecondary<real_t> s;
    s.pdg = 1000080160;
    s.z = 8; s.a = 16;
    s.mass = mtarget;
    s.kin_energy = 5.0;
    r.add_secondary(s);
    real_t de = 0;
    const real_t masses[1] = {mtarget};
    const auto v = check_result<real_t, 8>(pro, mtarget, r, lv, masses, &de);
    if (v != CheckResultVerdict::kAccept || std::fabs(de) > 1e-9) {
      fail("a conserving elastic state was not accepted: verdict %d, dE %g", int(v), de);
    }
    std::printf("  a conserving recoil: accepted, dE = %.3g MeV\n", de);
  }
  // The suppressed-recoil case: no secondary at all, and the nuclear mass is dropped from the
  // initial side so that a suppressed recoil still balances. Without that rule the imbalance
  // would be the whole target mass, 14.9 GeV, and every elastic scatter would be re-sampled.
  {
    HadFinalState<real_t, 8> r;
    r.clear();
    r.energy_change = 99.95;
    r.local_energy_deposit = 0.05;
    real_t de = 0;
    const auto v = check_result<real_t, 8>(pro, mtarget, r, lv, nullptr, &de);
    if (v != CheckResultVerdict::kAccept || std::fabs(de) > 1e-9) {
      fail("a suppressed recoil was not accepted: verdict %d, dE %g", int(v), de);
    }
    std::printf("  a suppressed recoil (no secondary): accepted, dE = %.3g MeV "
                "(the target mass is dropped from both sides)\n", de);
  }
  // And a real violation, which must exceed BOTH levels. 2% of 100 MeV is 2 MeV and the
  // absolute level is 1 GeV, so at 100 MeV nothing can trip it - the check is inert at QBBC's
  // proton energies, and that is the fact being recorded.
  {
    HadFinalState<real_t, 8> r;
    r.clear();
    r.energy_change = 0.0;
    r.local_energy_deposit = 0.0;
    real_t de = 0;
    const auto v = check_result<real_t, 8>(pro, mtarget, r, lv, nullptr, &de);
    if (v != CheckResultVerdict::kAccept) {
      fail("at 100 MeV even a total energy loss should pass the AND of (2%%, 1 GeV): got %d",
           int(v));
    }
    std::printf("  losing all 100 MeV: still accepted (dE = %.4g MeV is under the 1 GeV "
                "absolute level)\n", de);
    HadProjectile<real_t> hot = pro;
    hot.kin_energy = 100000.0;  // 100 GeV, where 2% is 2 GeV and the absolute level bites
    HadFinalState<real_t, 8> r2;
    r2.clear();
    r2.energy_change = 0.0;
    real_t de2 = 0;
    const auto v2 = check_result<real_t, 8>(hot, mtarget, r2, lv, nullptr, &de2);
    if (v2 != CheckResultVerdict::kResampleEnergyBalance) {
      fail("at 100 GeV a total energy loss must be re-sampled: got %d, dE %g", int(v2), de2);
    }
    std::printf("  losing all 100 GeV: re-sampled (dE = %.4g MeV over both levels)\n", de2);
  }
  // An off-shell secondary: |m_pdg - m_dyn| > 0.1*m_pdg + 1 MeV.
  {
    HadFinalState<real_t, 8> r;
    r.clear();
    r.energy_change = 95.0;
    HadSecondary<real_t> s;
    s.pdg = 2212;
    s.mass = 938.272013 * 1.5;
    s.kin_energy = 5.0;
    r.add_secondary(s);
    const real_t masses[1] = {938.272013};
    real_t de = 0;
    const auto v = check_result<real_t, 8>(pro, mtarget, r, lv, masses, &de);
    if (v != CheckResultVerdict::kResampleOffShellSecondary) {
      fail("a 50%% off-shell secondary must be re-sampled: got %d", int(v));
    }
    std::printf("  a secondary 50%% off its PDG mass: re-sampled\n");
  }
  // The ep report at the defaults: with both levels at infinity every verdict must pass, and a
  // baryon-number imbalance must ALSO pass, because chargePass is forced true when the absolute
  // level is DBL_MAX. That is Geant4's behaviour and it is the reason the default is not a
  // safety net.
  {
    HadronicStepResult<real_t, 8> res;
    res.status = TrackStatusChange::kAlive;
    res.energy = 95.0;
    res.momentum_direction = Vec3<real_t>{real_t(0), real_t(0), real_t(1)};
    res.n_secondaries = 0;
    const Vec3<real_t> dir{real_t(0), real_t(0), real_t(1)};
    const real_t inf = 1e308;
    const auto rep = report_energy_momentum<real_t, 8>(pro, dir, mtarget, 8, 16, res, nullptr,
                                                       nullptr, inf, inf, true);
    if (!rep.conservation_pass) {
      fail("with infinite check levels the ep report must pass; it did not");
    }
    if (rep.relative != real_t(0)) {
      fail("with an infinite absolute level checkRelative is false, so `relative` must be 0; "
           "got %g", rep.relative);
    }
    // The same state with finite levels: an alive primary with no secondary is seeded with the
    // whole initial four-momentum, so it conserves exactly whatever it did.
    const auto rep2 = report_energy_momentum<real_t, 8>(pro, dir, mtarget, 8, 16, res, nullptr,
                                                        nullptr, real_t(0.01), real_t(1), false);
    if (!rep2.conservation_pass || rep2.absolute != real_t(0)) {
      fail("an alive primary with no secondary must conserve exactly: dE %g, pass %d",
           rep2.absolute, int(rep2.conservation_pass));
    }
    std::printf("  the ep report: passes at the (inf, inf) defaults; a suppressed recoil "
                "conserves exactly by construction\n");
  }
  std::printf("\n");
}

}  // namespace

int main() {
  const std::string dir = oracle_dir();
  const NameValue nv = load_name_value(dir + "/elastic_framework.csv");
  if (nv.names.empty()) {
    std::printf("cannot read %s/elastic_framework.csv - run ref/oracle/run.bat tables first\n",
                dir.c_str());
    return 1;
  }
  check_proton_cut(nv);
  check_overlap(dir);
  check_zanda(dir);
  check_elastic_process();
  check_checks(nv);

  if (fails == 0) {
    std::printf("test_hadronic_process: PASS\n");
    return 0;
  }
  std::printf("test_hadronic_process: %d FAILURES\n", fails);
  return 1;
}
