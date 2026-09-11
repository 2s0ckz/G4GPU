// G4NistElementBuilder's isotope abundances, and the target draw that needed them.
//
// Two oracle files, and two different disciplines:
//
//   isotopes.csv        EXACT. Every element G4NistManager builds, its isotope list and its
//                       relative-abundance vector, against data/isotope_abundance.hh. Compared
//                       bit for bit, because both sides are the same arithmetic on the same
//                       literals: `(0.01*W_i/ww)/wtSum` in double, in that order. A tolerance
//                       here would hide exactly the mistake this is guarding - a port that
//                       applied one of the two normalisations, or applied them in the other
//                       order, is out by an ulp or two on every multi-isotope element and by
//                       nothing at all on hydrogen.
//
//   isotope_zanda.csv   STATISTICAL. G4CrossSectionDataStore::SampleZandA's counted element and
//                       isotope frequencies, 200,000 draws per (data set, material, energy) for
//                       water, air, compact bone and lead over three data sets that take the
//                       three different SelectIsotope branches. The port draws the same number
//                       of times against Geant4's OWN cross sections, taken from the CSV, so
//                       what is measured is the draw and the abundances rather than P2's tables
//                       (which tests/test_particlexs.cu already compares bit for bit).
//
// WHY THIS FILE EXISTS WHEN tests/test_hadronic_process.cu ALREADY TESTS SampleZandA
//
// That test feeds Geant4's abundances in from `elastic_zanda.csv` and checks the selection. It
// cannot fail on a missing abundance table, because it never asks the port for one - and until
// P8b there was none: `data/natural_isotopes.hh` is the SET of natural isotopes and
// `data/isotope_list.hh` is amin/amax/aeff. That absence is what blocked `hadElastic` and the
// neutron general process, both of which draw a target. So the thing tested here is the 311
// numbers themselves, over all 107 elements, plus the draw taken through the port's own table.
// This translation unit hands HOST-ONLY functors to `__host__ __device__` templates on purpose,
// for the reason tests/test_hadronic_process.cu states at length: the functions under test are
// device-callable so the same code runs in a kernel, nvcc instantiates their device side even
// when nothing launches them, and the functors here read std::vector and std::string. What is
// checked here is the numbers, on the host, against a CSV; the device side of the same templates
// is exercised by tests/test_step_hadron.cu with real device functors.
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
#include "data/isotope_abundance.hh"
#include "data/isotope_list.hh"
#include "data/materials.cuh"
#include "data/natural_isotopes.hh"
#include "physics/hadronic/xs/sample_za.cuh"

using real_t = double;
using namespace g4gpu;
using namespace g4gpu::hadronic::xs;

namespace {

int g_fails = 0;
void fail(const char* fmt, ...) {
  std::printf("    FAIL: ");
  va_list ap;
  va_start(ap, fmt);
  std::vfprintf(stdout, fmt, ap);
  va_end(ap);
  std::printf("\n");
  ++g_fails;
}

std::vector<std::string> split(const std::string& s, char c) {
  std::vector<std::string> out;
  std::size_t a = 0;
  for (std::size_t i = 0; i <= s.size(); ++i) {
    if (i == s.size() || s[i] == c) {
      out.push_back(s.substr(a, i - a));
      a = i + 1;
    }
  }
  return out;
}

/// Loads a CSV, dropping the header line. Returns an empty vector when the file is absent so
/// the caller can say which file rather than dying inside a parser.
std::vector<std::vector<std::string>> load_rows(const std::string& path) {
  std::vector<std::vector<std::string>> rows;
  FILE* f = std::fopen(path.c_str(), "r");
  if (f == nullptr) { return rows; }
  char line[8192];
  bool first = true;
  while (std::fgets(line, sizeof line, f) != nullptr) {
    if (first) {
      first = false;
      continue;
    }
    std::size_t n = std::strlen(line);
    while (n > 0 && (line[n - 1] == '\n' || line[n - 1] == '\r')) { line[--n] = '\0'; }
    if (n == 0) { continue; }
    rows.push_back(split(line, ','));
  }
  std::fclose(f);
  return rows;
}

// =============================================================================================
// 1. The table, exactly.
// =============================================================================================

int check_table(const std::string& dir) {
  const auto rows = load_rows(dir + "/isotopes.csv");
  if (rows.empty()) {
    fail("cannot read isotopes.csv - regenerate the oracle (ref/dump/build.bat then "
         "ref/oracle/run.bat tables)");
    return 1;
  }

  int n_elements = 0, n_isotopes = 0, last_z = -1;
  double worst_w = 0.0, worst_nist = 0.0;
  std::string where_w, where_nist;
  int n_natural_agreed = 0;

  for (const auto& r : rows) {
    if (r.size() < 11) { continue; }
    const int Z = std::atoi(r[0].c_str());
    const int nnist = std::atoi(r[2].c_str());
    const int n0 = std::atoi(r[3].c_str());
    const int nelem_iso = std::atoi(r[4].c_str());
    const int j = std::atoi(r[5].c_str());
    const int A = std::atoi(r[6].c_str());
    const double w = std::atof(r[7].c_str());
    const double nist_ab = std::atof(r[8].c_str());
    ++n_isotopes;
    if (Z != last_z) {
      ++n_elements;
      last_z = Z;
      // The pre-BuildElement numbers: nIsotopes[Z] and nFirstIsotope[Z].
      if (data::nist_n_isotopes()[Z] != nnist) {
        fail("Z=%d: nIsotopes %d, Geant4 says %d", Z, data::nist_n_isotopes()[Z], nnist);
      }
      if (data::nist_first_isotope_n()[Z] != n0) {
        fail("Z=%d: nFirstIsotope %d, Geant4 says %d", Z, data::nist_first_isotope_n()[Z], n0);
      }
      if (data::nist_element_n_isotopes(Z) != nelem_iso) {
        fail("Z=%d: the element carries %d isotopes, Geant4 says %d", Z,
             data::nist_element_n_isotopes(Z), nelem_iso);
      }
    }
    const int off = data::nist_element_iso_offset()[Z];
    if (j >= data::nist_element_n_isotopes(Z)) {
      fail("Z=%d: Geant4 has an isotope %d the port's element vector does not", Z, j);
      continue;
    }
    if (data::nist_element_iso_a()[off + j] != A) {
      fail("Z=%d j=%d: A %d, Geant4 says %d", Z, j, data::nist_element_iso_a()[off + j], A);
    }
    // Bit for bit. Both sides are `(0.01*W/ww)/wtSum` evaluated in double.
    const double port_w = data::nist_element_iso_w()[off + j];
    if (port_w != w) {
      const double rel = (w != 0.0) ? std::fabs(port_w - w) / std::fabs(w) : std::fabs(port_w);
      if (rel > worst_w) {
        worst_w = rel;
        char b[160];
        std::snprintf(b, sizeof b, "Z=%d A=%d port %.17g G4 %.17g", Z, A, port_w, w);
        where_w = b;
      }
    }
    // GetIsotopeAbundance(Z, A) - the FIRST normalisation, before AddIsotope's. Different
    // numbers from the above on every element whose surviving abundances do not sum to
    // exactly 1.0, so checking one does not check the other.
    const double port_nist = data::nist_isotope_abundance(Z, A);
    if (port_nist != nist_ab) {
      const double rel =
          (nist_ab != 0.0) ? std::fabs(port_nist - nist_ab) / std::fabs(nist_ab)
                           : std::fabs(port_nist);
      if (rel > worst_nist) {
        worst_nist = rel;
        char b[160];
        std::snprintf(b, sizeof b, "Z=%d A=%d port %.17g G4 %.17g", Z, A, port_nist, nist_ab);
        where_nist = b;
      }
    }
    // The predicate data/natural_isotopes.hh carries is the SIGN of this number, so the two
    // tables have to agree about which isotopes exist. They were extracted by two scripts from
    // the same source and nothing else compares them.
    if (data::is_natural_isotope(Z, A) != (nist_ab > 0.0)) {
      fail("Z=%d A=%d: natural_isotopes.hh says %d, abundance is %.17g", Z, A,
           int(data::is_natural_isotope(Z, A)), nist_ab);
    } else {
      ++n_natural_agreed;
    }
  }

  std::printf("  elements %d, isotopes %d, natural-set agreements %d\n", n_elements, n_isotopes,
              n_natural_agreed);
  std::printf("  abundance vector      worst relative %.3e %s\n", worst_w, where_w.c_str());
  std::printf("  GetIsotopeAbundance   worst relative %.3e %s\n", worst_nist,
              where_nist.c_str());
  if (worst_w != 0.0) { fail("the abundance vector is not bit-exact: %s", where_w.c_str()); }
  if (worst_nist != 0.0) {
    fail("GetIsotopeAbundance is not bit-exact: %s", where_nist.c_str());
  }
  // 104, not 107. G4AtomicShells' tables are `[105]` and G4Element::AddIsotope indexes them,
  // so FindOrBuildElement(105) aborts the process - which is how this number was found. See
  // the header of data/isotope_abundance.hh and docs/RISK.md V60.
  if (n_elements != data::kNistBuildableMaxZ) {
    fail("compared %d elements, expected %d", n_elements, data::kNistBuildableMaxZ);
  }
  // The three elements above the buildable ceiling each carry one fabricated isotope, so the
  // oracle sees three fewer than the table holds. Stated as arithmetic rather than as a
  // literal, so a Geant4 that raises either limit fails here with the reason visible.
  int above = 0;
  for (int Z = data::kNistBuildableMaxZ + 1; Z < data::kNistMaxElements; ++Z) {
    above += data::nist_element_n_isotopes(Z);
  }
  if (n_isotopes != data::kNistElementIsotopes - above) {
    fail("compared %d isotopes, expected %d (%d of the table's %d are at Z > %d)", n_isotopes,
         data::kNistElementIsotopes - above, above, data::kNistElementIsotopes,
         data::kNistBuildableMaxZ);
  }
  // Every element vector sums to 1, which is what AddIsotope's second normalisation is for.
  for (int Z = 1; Z < data::kNistMaxElements; ++Z) {
    const int n = data::nist_element_n_isotopes(Z);
    if (n == 0) { continue; }
    const int off = data::nist_element_iso_offset()[Z];
    double s = 0.0;
    for (int j = 0; j < n; ++j) { s += data::nist_element_iso_w()[off + j]; }
    if (std::fabs(s - 1.0) > 1e-15) {
      fail("Z=%d: abundances sum to %.17g", Z, s);
    }
  }
  return 0;
}

// =============================================================================================
// 2. The draw, against Geant4's counted frequencies.
// =============================================================================================

/// One element's row out of isotope_zanda.csv.
struct IsoSpec {
  int a = 0;
  double abundance = 0;
  double xs = 0;   ///< the per-isotope cross section, mm^2
  int count = 0;
};
struct ElmRow {
  std::string dataset, particle, material;
  double ekin = 0;
  int nelm = 0, i = 0, z = 0, ntrial = 0, n_selected = 0;
  double natoms = 0, xs_per_atom = 0, cum = 0;
  std::vector<IsoSpec> isos;
};

/// The three-function contract of xs/sample_za.cuh, backed by the oracle's own numbers.
///
/// `abundance_only` is the data set's SelectIsotope fallback test, spelled out per class rather
/// than read out of a PxsDataSet, because what is under test here is the draw and not the
/// reader. The three rules are the ones in the header of xs/sample_za.cuh.
struct OracleXs {
  const std::vector<ElmRow>* block = nullptr;
  int kind = 0;   ///< 0 neutron elastic, 1 neutron capture, 2 particle inelastic

  const ElmRow* row_for(int Z) const {
    for (const ElmRow& r : *block) {
      if (r.z == Z) { return &r; }
    }
    return nullptr;
  }
  XsValue<real_t> element(int Z) const {
    const ElmRow* r = row_for(Z);
    return {r != nullptr ? r->xs_per_atom : 0.0, XsRefusal::kNone};
  }
  XsValue<real_t> isotope(int Z, int A) const {
    const ElmRow* r = row_for(Z);
    if (r != nullptr) {
      for (const IsoSpec& s : r->isos) {
        if (s.a == A) { return {s.xs, XsRefusal::kNone}; }
      }
    }
    return {0.0, XsRefusal::kNone};
  }
  bool abundance_only(int Z) const {
    if (kind == 0) { return true; }              // G4NeutronElasticXS has no isotope data
    if (Z >= 93) { return true; }                // MAXZCAPTURE / MAXZINELP
    if (Z > data::kIsotopeListMaxZ) { return true; }
    return data::isotope_amax()[Z] == data::isotope_amin()[Z];
  }
};

/// The two-sample proportion z for two counts out of the same N, with the POOLED estimate of p.
///
/// THE POOLED ESTIMATE IS NOT A REFINEMENT, IT IS WHAT MAKES THE TEST WORK ON A RARE ELEMENT.
///
/// The obvious form uses the oracle's own p in the variance - `2*p_g4*(1-p_g4)/N` - and that
/// variance is exactly zero whenever Geant4 selected a bin zero times. Air's carbon at 1 MeV is
/// such a bin: Geant4 drew C-13 zero times in 200,000 and the port drew it once, and the first
/// version of this test reported that as an infinite deviation and failed. The pooled p,
/// `(k1+k2)/2N`, is the standard two-sample estimator and gives the right answers at both ends:
/// 0 against 0 is zero sigma, 0 against 1 is 1.0 sigma, and 0 against 25 out of 200,000 is 5.0 -
/// so a bin the port draws and Geant4 never does is still caught, at the count where it stops
/// being a fluctuation. docs/RISK.md V1 is a test whose tolerance was a coin flip.
double two_sample_z(int k_g4, int k_port, int n) {
  if (k_g4 == k_port) { return 0.0; }
  const double p = double(k_g4 + k_port) / (2.0 * double(n));
  const double var = 2.0 * p * (1.0 - p) / double(n);
  if (!(var > 0.0)) { return 0.0; }
  return std::fabs(double(k_port) - double(k_g4)) / double(n) / std::sqrt(var);
}

int n_elm_strong = 0, n_iso_strong = 0;

int check_draw(const std::string& dir) {
  const auto raw = load_rows(dir + "/isotope_zanda.csv");
  if (raw.empty()) {
    fail("cannot read isotope_zanda.csv");
    return 1;
  }
  std::vector<ElmRow> rows;
  for (const auto& r : raw) {
    if (r.size() < 15) { continue; }
    ElmRow e;
    e.dataset = r[0];
    e.particle = r[1];
    e.material = r[2];
    e.ekin = std::atof(r[3].c_str());
    e.nelm = std::atoi(r[4].c_str());
    e.i = std::atoi(r[6].c_str());
    e.z = std::atoi(r[7].c_str());
    e.natoms = std::atof(r[8].c_str());
    e.xs_per_atom = std::atof(r[9].c_str());
    e.cum = std::atof(r[10].c_str());
    e.ntrial = std::atoi(r[11].c_str());
    e.n_selected = std::atoi(r[12].c_str());
    for (const std::string& tok : split(r[14], ';')) {
      const auto f = split(tok, ':');
      if (f.size() != 4) { continue; }
      IsoSpec s;
      s.a = std::atoi(f[0].c_str());
      s.abundance = std::atof(f[1].c_str());
      s.xs = std::atof(f[2].c_str());
      s.count = std::atoi(f[3].c_str());
      e.isos.push_back(s);
    }
    rows.push_back(e);
  }

  std::printf("  %-22s %-22s %8s %3s %10s %10s %7s %7s\n", "dataset", "material", "E (MeV)",
              "Z", "G4 frac", "port frac", "elm sig", "iso sig");
  double worst_elm_sigma = 0.0, worst_iso_sigma = 0.0, worst_ab = 0.0;
  std::string where_elm, where_iso, where_ab;
  int n_blocks = 0, n_elm = 0, n_iso = 0;

  std::size_t at = 0;
  while (at < rows.size()) {
    // One block is all the element rows sharing (dataset, material, energy).
    std::vector<ElmRow> block;
    const std::string ds = rows[at].dataset, mt = rows[at].material;
    const double ek = rows[at].ekin;
    while (at < rows.size() && rows[at].dataset == ds && rows[at].material == mt
           && rows[at].ekin == ek) {
      block.push_back(rows[at]);
      ++at;
    }
    ++n_blocks;

    // THE ABUNDANCES THE ORACLE USED MUST BE THE PORT'S. Checked before the draw, so a
    // frequency disagreement cannot be blamed on the table and vice versa.
    for (const ElmRow& r : block) {
      const int n = data::nist_element_n_isotopes(r.z);
      if (n != int(r.isos.size())) {
        fail("%s %s Z=%d: %d isotopes, Geant4 used %d", ds.c_str(), mt.c_str(), r.z, n,
             int(r.isos.size()));
        continue;
      }
      const int off = data::nist_element_iso_offset()[r.z];
      for (int j = 0; j < n; ++j) {
        if (data::nist_element_iso_a()[off + j] != r.isos[j].a) {
          fail("%s Z=%d j=%d: A %d, Geant4 used %d", mt.c_str(), r.z, j,
               data::nist_element_iso_a()[off + j], r.isos[j].a);
        }
        const double d = std::fabs(data::nist_element_iso_w()[off + j] - r.isos[j].abundance);
        if (d > worst_ab) {
          worst_ab = d;
          char b[160];
          std::snprintf(b, sizeof b, "%s Z=%d A=%d", mt.c_str(), r.z, r.isos[j].a);
          where_ab = b;
        }
      }
    }

    // A Material carrying only what the draw reads: the element list and the atom densities,
    // out of the CSV. Building it from the NIST material database instead would make this a
    // test of data/nist_materials.hh as well.
    data::Material<real_t> mat{};
    mat.n_elements = block[0].nelm;
    for (int i = 0; i < mat.n_elements && i < data::kMaxElements; ++i) {
      mat.z[i] = block[i].z;
      mat.n_atoms[i] = block[i].natoms;
    }

    OracleXs xs;
    xs.block = &block;
    xs.kind = (ds == "G4NeutronElasticXS") ? 0 : ((ds == "G4NeutronCaptureXS") ? 1 : 2);

    // ComputeCrossSection through the port's own accumulation, with the isotope lists coming
    // from the port's table - which is the whole point of the exercise.
    MaterialXs<real_t> mxs{};
    const auto total =
        store_compute_cross_section_fn<real_t>(xs, mat, nist_isotopes_of<real_t>(mat), mxs);
    // The cumulative array Geant4 left behind, for the same reason test_hadronic_process
    // checks it: agreeing on the frequencies while disagreeing on the partial sums would be
    // two errors cancelling.
    for (int i = 0; i < mat.n_elements; ++i) {
      const double rel = (block[i].cum != 0.0)
                             ? std::fabs(mxs.cumulative[i] - block[i].cum) / block[i].cum
                             : std::fabs(mxs.cumulative[i]);
      if (rel > 1e-12) {
        fail("%s %s E=%g Z=%d: cumulative %.17g, Geant4 %.17g", ds.c_str(), mt.c_str(), ek,
             block[i].z, mxs.cumulative[i], block[i].cum);
      }
    }
    (void)total;

    const int kN = block[0].ntrial;
    std::vector<int> got_elm(mat.n_elements, 0);
    std::vector<std::vector<int>> got_iso(mat.n_elements);
    for (int i = 0; i < mat.n_elements; ++i) {
      got_iso[i].assign(data::nist_element_n_isotopes(block[i].z), 0);
    }
    // A fixed seed, and two uniforms per draw in the order SampleZandA consumes them: the
    // element first, then the isotope. A single-element material draws NO element uniform in
    // Geant4, which is why the port's function takes the two separately instead of an Rng.
    Philox<real_t> rng(0xB8Bu, 0u, 0u);
    for (int k = 0; k < kN; ++k) {
      const real_t q_elm = (mat.n_elements > 1) ? rng.uniform() : real_t(0);
      const real_t q_iso = rng.uniform();
      const TargetZA t = store_sample_za_fn<real_t>(xs, mat, nist_isotopes_of<real_t>(mat), mxs,
                                                    q_elm, q_iso);
      if (t.element_index < 0 || t.element_index >= mat.n_elements) {
        fail("%s %s: element index %d out of range", ds.c_str(), mt.c_str(), t.element_index);
        break;
      }
      ++got_elm[t.element_index];
      const int off = data::nist_element_iso_offset()[t.z];
      const int n = data::nist_element_n_isotopes(t.z);
      bool found = false;
      for (int j = 0; j < n; ++j) {
        if (data::nist_element_iso_a()[off + j] == t.a) {
          ++got_iso[t.element_index][j];
          found = true;
          break;
        }
      }
      if (!found) {
        fail("%s %s: drew Z=%d A=%d, which is not an isotope of that element", ds.c_str(),
             mt.c_str(), t.z, t.a);
        break;
      }
    }

    for (int i = 0; i < mat.n_elements; ++i) {
      const double p_g4 = double(block[i].n_selected) / double(kN);
      const double p_port = double(got_elm[i]) / double(kN);
      const double sig = two_sample_z(block[i].n_selected, got_elm[i], kN);
      ++n_elm;
      if (block[i].n_selected + got_elm[i] >= 100) { ++n_elm_strong; }
      if (sig > worst_elm_sigma) {
        worst_elm_sigma = sig;
        char b[200];
        std::snprintf(b, sizeof b, "%s %s E=%g Z=%d  G4 %.5f port %.5f", ds.c_str(),
                      mt.c_str(), ek, block[i].z, p_g4, p_port);
        where_elm = b;
      }
      double iso_sig = 0.0;
      for (std::size_t j = 0; j < got_iso[i].size() && j < block[i].isos.size(); ++j) {
        const double q_g4 = double(block[i].isos[j].count) / double(kN);
        const double q_port = double(got_iso[i][j]) / double(kN);
        const double s = two_sample_z(block[i].isos[j].count, got_iso[i][j], kN);
        ++n_iso;
        if (block[i].isos[j].count + got_iso[i][j] >= 100) { ++n_iso_strong; }
        if (s > iso_sig) { iso_sig = s; }
        if (s > worst_iso_sigma) {
          worst_iso_sigma = s;
          char b[200];
          std::snprintf(b, sizeof b, "%s %s E=%g Z=%d A=%d  G4 %.5f port %.5f", ds.c_str(),
                        mt.c_str(), ek, block[i].z, block[i].isos[j].a, q_g4, q_port);
          where_iso = b;
        }
      }
      if (i == 0) {
        std::printf("  %-22s %-22s %8g %3d %10.5f %10.5f %7.2f %7.2f\n", ds.c_str(),
                    mt.c_str(), ek, block[i].z, p_g4, p_port, sig, iso_sig);
      }
    }
  }

  // The "well populated" counts are the ones the comparison actually bites on: a bin both
  // sides selected fewer than a hundred times cannot distinguish a 1% error in a weight from
  // Poisson noise, so reporting only the totals would overstate what was checked.
  std::printf("\n  %d blocks, %d element frequencies (%d with >= 100 counts), %d isotope "
              "frequencies (%d with >= 100)\n", n_blocks, n_elm, n_elm_strong, n_iso,
              n_iso_strong);
  std::printf("  abundances used by the oracle vs the port: worst |diff| %.3e %s\n", worst_ab,
              where_ab.c_str());
  std::printf("  worst element  deviation %.2f sigma   %s\n", worst_elm_sigma,
              where_elm.c_str());
  std::printf("  worst isotope  deviation %.2f sigma   %s\n", worst_iso_sigma,
              where_iso.c_str());
  if (worst_ab != 0.0) {
    fail("the oracle's abundances are not the port's: %s", where_ab.c_str());
  }
  // 5 sigma over ~200 frequencies: a one-sided 5-sigma tail is 3e-7, so this is expected to
  // fire once in 2 million runs rather than occasionally. docs/RISK.md V1 is a test whose
  // tolerance was a coin flip.
  if (worst_elm_sigma > 5.0) { fail("element frequency off: %s", where_elm.c_str()); }
  if (worst_iso_sigma > 5.0) { fail("isotope frequency off: %s", where_iso.c_str()); }
  if (n_blocks < 36) { fail("only %d blocks compared, expected 36", n_blocks); }
  return 0;
}

// =============================================================================================
// 3. What a Z with no NIST element does. Refused, not guessed.
// =============================================================================================

void check_refusal() {
  for (const int Z : {0, 108, 120, -3}) {
    const auto e = nist_element_isotopes<real_t>(Z);
    if (e.n != 0 || e.a != nullptr || e.abundance != nullptr) {
      fail("Z=%d: expected no isotope information, got n=%d", Z, e.n);
    }
    if (data::nist_isotope_abundance(Z, 1) != 0.0) {
      fail("Z=%d: GetIsotopeAbundance is not zero outside the table", Z);
    }
  }
  // A material holding such a Z draws A = 0, which is not a nucleus - so the caller must
  // refuse. Asserted here so that the contract in the header is a tested statement.
  data::Material<real_t> mat{};
  mat.n_elements = 1;
  mat.z[0] = 108;
  mat.n_atoms[0] = 1.0;
  struct Unit {
    XsValue<real_t> element(int) const { return {1.0, XsRefusal::kNone}; }
    XsValue<real_t> isotope(int, int) const { return {1.0, XsRefusal::kNone}; }
    bool abundance_only(int) const { return true; }
  } u;
  MaterialXs<real_t> mxs{};
  (void)store_compute_cross_section_fn<real_t>(u, mat, nist_isotopes_of<real_t>(mat), mxs);
  const TargetZA t =
      store_sample_za_fn<real_t>(u, mat, nist_isotopes_of<real_t>(mat), mxs, 0.5, 0.5);
  if (t.a != 0) {
    fail("a Z with no NIST element drew A=%d; it must be 0 so the caller can refuse it", t.a);
  }
  std::printf("  Z outside G4NistElementBuilder: no isotopes, A = 0, %s\n",
              data::isotope_refusal_name(data::IsotopeRefusal::kNoSuchElement));
}

}  // namespace

int main() {
  const char* env = std::getenv("G4GPU_ORACLE");
  const std::string dir = (env != nullptr) ? env : "ref/oracle";
  std::printf("== G4NistElementBuilder isotope abundances vs Geant4 11.1.1 ==\n");
  std::printf("   oracle: %s\n\n", dir.c_str());

  std::printf("-- 1. the table, bit for bit --\n");
  if (check_table(dir) != 0) {
    std::printf("\nFAILED (%d failures)\n", g_fails);
    return 1;
  }
  std::printf("\n-- 2. SampleZandA through the port's abundances --\n");
  check_draw(dir);
  std::printf("\n-- 3. a Z with no NIST element --\n");
  check_refusal();

  std::printf("\n%s (%d failures)\n", g_fails ? "FAILED" : "PASSED", g_fails);
  return g_fails ? 1 : 0;
}
