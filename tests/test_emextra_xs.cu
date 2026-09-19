// The three cross sections QBBC's G4EmExtraPhysics processes carry, against Geant4 11.1.1.
//
//   emextra_photonuc.csv    G4PhotoNuclearCrossSection - the CHIPS photo-nuclear
//                           parameterisation, element for Z = 1..98 and the three light
//                           isotope arms, over a grid that includes every branch boundary
//                           exactly (THmin = 2, Emin = 106, Emax = 50000 MeV).
//   emextra_electronuc.csv  G4ElectroNuclearCrossSection::GetElementCrossSection, Z = 1..98.
//   emextra_eqphoton.csv    GetEquivalentPhotonEnergy, GetEquivalentPhotonQ2 and
//                           GetVirtualFactor under the prescribed eight-value cycle, WITH the
//                           number of deviates each consumed.
//   emextra_kokoulin.csv    G4KokoulinMuonNuclearXS - the double-differential form on the
//                           G4MuonVDNuclearModel sampling grid, and the element cross section
//                           at the 61 nodes of its own log vector and between them.
//
// EVERYTHING HERE IS EXACT. Nothing in these four files samples anything except the three
// equivalent-photon functions, and those are driven under a prescribed engine so that they are
// deterministic functions of (inputs, phase). The tolerance is 1e-13 relative, which is where a
// G4Exp/G4Log-against-std difference would show if there were one; the CHIPS classes use G4Log
// and G4Exp throughout and this port uses std::log and std::exp, so the buckets that come back
// at 1e-16 are also a measurement that those two agree on this platform for these arguments.
//
// WHY THE TWO CHIPS CLASSES ARE COMPARED OVER ALL 98 ELEMENTS AND NOT OVER THE RUN'S SEVEN.
// Neither class needs a data file or a G4Element: both are closed-form in `GetAtomicMassAmu(Z)`,
// so the oracle can ask for any Z and the port must answer for any Z. The Kokoulin table cannot
// - `theCrossSection[Z]` is filled only for elements in the geometry - so its element rows are
// the run's own elements and the coverage is REPORTED rather than assumed.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "physics/hadronic/xs/chips_electronuclear.cuh"
#include "physics/hadronic/xs/chips_photonuclear.cuh"
#include "physics/hadronic/xs/kokoulin_muon_xs.cuh"

using namespace g4gpu;
using namespace g4gpu::hadronic::xs;

/// The deliverables are device code. Never launched; instantiating the kernel is what proves
/// they compile for the device and what `-Xptxas -v` measures. The Kokoulin table is ~45 kB and
/// is reached through a pointer, never by value and never as a local.
__global__ void emextra_xs_probe(const kokoulin::KokoulinTable<double>* tab, double e, int Z,
                                 double* out) {
  const XsValue<double> a = chips::photo_element_xs<double>(e, Z);
  const XsValue<double> b = chips::photo_iso_xs<double>(e, 1, 2);
  chips::ElnState st;
  const XsValue<double> c = chips::eln_element_xs<double>(st, e, Z);
  const XsValue<double> d = kokoulin::element_xs<double>(*tab, e, Z);
  const double f = chips::eln_virtual_factor(100.0, 1000.0);
  out[0] = a.value + b.value + c.value + d.value + f
           + kokoulin::dd_microscopic_xs<double>(e, 12.0, 300.0, 105.6583715);
}

namespace {

int fails = 0;

struct Bucket {
  const char* name;
  long long n = 0;
  double worst = 0.0;
  std::string where;
  double tol = 1e-13;
  long long refused = 0;
};

std::vector<Bucket> buckets;

int new_bucket(const char* name, double tol) {
  Bucket b;
  b.name = name;
  b.tol = tol;
  buckets.push_back(b);
  return static_cast<int>(buckets.size()) - 1;
}

void cmp(int bi, double got, double want, const std::string& where) {
  Bucket& b = buckets[bi];
  ++b.n;
  const double scale = (std::fabs(want) > 0.0) ? std::fabs(want) : 1.0;
  const double rel = std::fabs(got - want) / scale;
  if (rel > b.worst) { b.worst = rel; b.where = where; }
}

void cmp_int(int bi, long long got, long long want, const std::string& where) {
  Bucket& b = buckets[bi];
  ++b.n;
  if (got != want) {
    b.worst = 1.0;
    if (b.where.empty()) {
      b.where = where + " got " + std::to_string(got) + " want " + std::to_string(want);
    }
  }
}

std::string oracle_dir() {
  const char* e = std::getenv("G4GPU_ORACLE");
  return (e != nullptr && e[0] != '\0') ? std::string(e) : std::string("ref/oracle");
}

std::vector<std::string> split(const std::string& s) {
  std::vector<std::string> out;
  std::string cur;
  for (const char c : s) {
    if (c == ',') { out.push_back(cur); cur.clear(); }
    else if (c != '\n' && c != '\r') { cur.push_back(c); }
  }
  out.push_back(cur);
  return out;
}

struct Csv {
  std::map<std::string, std::size_t> ix;
  std::vector<std::vector<std::string>> rows;
  bool load(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "r");
    if (f == nullptr) { return false; }
    static char line[8192];
    bool first = true;
    while (std::fgets(line, sizeof line, f) != nullptr) {
      const std::vector<std::string> v = split(line);
      if (first) {
        for (std::size_t i = 0; i < v.size(); ++i) { ix[v[i]] = i; }
        first = false;
      } else if (!v.empty() && !v[0].empty()) {
        rows.push_back(v);
      }
    }
    std::fclose(f);
    return !first;
  }
  const std::string& s(std::size_t r, const char* col) const {
    static const std::string empty;
    const auto it = ix.find(col);
    if (it == ix.end() || it->second >= rows[r].size()) { return empty; }
    return rows[r][it->second];
  }
  double d(std::size_t r, const char* col) const { return std::atof(s(r, col).c_str()); }
  int i(std::size_t r, const char* col) const { return std::atoi(s(r, col).c_str()); }
};

/// The prescribed eight-value cycle, on the port's side. Identical to the CycleEngine in
/// ref/dump/dump_emextra.cc, and the draw count is what makes a transcription that reaches the
/// right answer through the wrong number of deviates visible.
struct CycleRng {
  int phase = 0;
  int n = 0;
  __host__ __device__ void reset(int p) { phase = p; n = 0; }
  __host__ __device__ double uniform() {
    // A function-scope `static const` array, not a class static: nvcc treats a class static as
    // a host variable and device code referencing it is "undefined in device code" - the same
    // rule src/data/bertini_channels.hh's header records for the generated tables.
    static const double kSeq[8] = {0.05, 0.37, 0.63, 0.91, 0.12, 0.78, 0.29, 0.55};
    const double v = kSeq[(n + phase) % 8];
    ++n;
    return v;
  }
};

/// The port's cross sections come back in mm^2 through `millibarn`; the oracle is in mb.
double to_mb(double v) { return v / double(chips::millibarn<double>()); }

/// `g/mole` as `emextra/lepton_vd.cuh` derives it. Declared here rather than including that
/// header, which would pull the whole Bertini tree into a cross-section test; the derivation is
/// one expression and the oracle's `gmole` row is what checks it in both files.
double ee_g_per_mole() {
  const double e_SI = 1.602176634e-19;
  const double joule = 1.0e-6 / e_SI;
  const double second = 1.0e9;
  const double meter = 1.0e3;
  return 1.0e-3 * joule * second * second / (meter * meter);
}

}  // namespace

int main() {
  const std::string dir = oracle_dir();

  // -------------------------------------------------------------------------------------------
  // 1. emextra_photonuc.csv - G4PhotoNuclearCrossSection
  // -------------------------------------------------------------------------------------------
  const int b_pn_el = new_bucket("PhotoNuclearElement", 1e-13);
  const int b_pn_iso = new_bucket("PhotoNuclearIsotope", 1e-13);
  {
    Csv c;
    if (!c.load(dir + "/emextra_photonuc.csv")) {
      std::printf("FAIL: cannot read %s/emextra_photonuc.csv\n", dir.c_str());
      ++fails;
    } else {
      for (std::size_t r = 0; r < c.rows.size(); ++r) {
        const int Z = c.i(r, "Z");
        const int A = c.i(r, "A");
        const double e = c.d(r, "energy_MeV");
        const double want = c.d(r, "xs_mb");
        const bool is_iso = (c.s(r, "kind") == "isotope");
        const int bi = is_iso ? b_pn_iso : b_pn_el;
        const XsValue<double> v = is_iso ? chips::photo_iso_xs<double>(e, Z, A)
                                         : chips::photo_element_xs<double>(e, Z);
        if (!v.ok()) {
          ++buckets[bi].refused;
          continue;
        }
        cmp(bi, to_mb(v.value), want,
            "Z=" + std::to_string(Z) + " A=" + std::to_string(A) + " E=" + c.s(r, "energy_MeV"));
      }
    }
  }

  // -------------------------------------------------------------------------------------------
  // 2. emextra_electronuc.csv - G4ElectroNuclearCrossSection::GetElementCrossSection
  //
  // A FRESH `ElnState` PER ROW, like the oracle's fresh object per row. The class caches by Z
  // and returns `lastSig` unchanged when the same (Z, E) comes round again; a state carried
  // across the grid would be testing the cache and not the cross section.
  // -------------------------------------------------------------------------------------------
  const int b_eln = new_bucket("ElectroNuclearElement", 1e-13);
  {
    Csv c;
    if (!c.load(dir + "/emextra_electronuc.csv")) {
      std::printf("FAIL: cannot read %s/emextra_electronuc.csv\n", dir.c_str());
      ++fails;
    } else {
      for (std::size_t r = 0; r < c.rows.size(); ++r) {
        const int Z = c.i(r, "Z");
        const double e = c.d(r, "energy_MeV");
        chips::ElnState st;
        const XsValue<double> v = chips::eln_element_xs<double>(st, e, Z);
        if (!v.ok()) {
          ++buckets[b_eln].refused;
          continue;
        }
        cmp(b_eln, to_mb(v.value), c.d(r, "xs_mb"),
            "Z=" + std::to_string(Z) + " E=" + c.s(r, "energy_MeV"));
      }
    }
  }

  // -------------------------------------------------------------------------------------------
  // 3. emextra_eqphoton.csv - the three equivalent-photon functions under the cycle
  // -------------------------------------------------------------------------------------------
  const int b_eq_xs = new_bucket("EqPhotonCrossSection", 1e-13);
  const int b_eq_nu = new_bucket("EqPhotonEnergy", 1e-13);
  const int b_eq_q2 = new_bucket("EqPhotonQ2", 1e-13);
  const int b_eq_vf = new_bucket("EqPhotonVirtualFactor", 1e-13);
  const int b_eq_dn = new_bucket("EqPhotonDrawsNu", 0.0);
  const int b_eq_dq = new_bucket("EqPhotonDrawsQ2", 0.0);
  const int b_eq_dt = new_bucket("EqPhotonDrawsTotal", 0.0);
  long long n_hp = 0, n_func = 0, n_corr = 0, n_newton = 0, n_underflow = 0;
  {
    Csv c;
    if (!c.load(dir + "/emextra_eqphoton.csv")) {
      std::printf("FAIL: cannot read %s/emextra_eqphoton.csv\n", dir.c_str());
      ++fails;
    } else {
      for (std::size_t r = 0; r < c.rows.size(); ++r) {
        const int Z = c.i(r, "Z");
        const double lepE = c.d(r, "lepton_MeV");
        const int phase = c.i(r, "phase");
        const std::string where =
            "Z=" + std::to_string(Z) + " E=" + c.s(r, "lepton_MeV") + " ph=" + c.s(r, "phase");
        CycleRng rng;
        rng.reset(phase);
        chips::ElnState st;
        chips::ElnSampleStatus stat;
        const XsValue<double> xs = chips::eln_element_xs<double>(st, lepE, Z);
        if (!xs.ok()) { ++buckets[b_eq_xs].refused; continue; }
        const int d0 = rng.n;
        cmp(b_eq_xs, to_mb(xs.value), c.d(r, "xs_mb"), where);
        const double nu = chips::eln_equivalent_photon_energy(st, rng, stat);
        const int d1 = rng.n;
        const double q2 = chips::eln_equivalent_photon_q2(st, nu, rng, stat);
        const int d2 = rng.n;
        const double vf = chips::eln_virtual_factor(nu, q2);
        cmp(b_eq_nu, nu, c.d(r, "nu_MeV"), where);
        cmp(b_eq_q2, q2, c.d(r, "Q2_MeV2"), where);
        cmp(b_eq_vf, vf, c.d(r, "virtual_factor"), where);
        cmp_int(b_eq_dn, d1 - d0, c.i(r, "draws_nu"), where);
        cmp_int(b_eq_dq, d2 - d1, c.i(r, "draws_q2"), where);
        cmp_int(b_eq_dt, rng.n, c.i(r, "draws_total"), where);
        if (stat.hp_warning) { ++n_hp; }
        if (stat.func_region) { ++n_func; }
        if (stat.phle_corrected) { ++n_corr; }
        if (stat.newton_exhausted || stat.newton_clamped) { ++n_newton; }
        if (stat.y_index_underflow) { ++n_underflow; }
      }
    }
  }

  // -------------------------------------------------------------------------------------------
  // 4. emextra_kokoulin.csv - the double-differential form and the element table
  // -------------------------------------------------------------------------------------------
  const int b_ko_dd = new_bucket("KokoulinDoubleDifferential", 1e-13);
  const int b_ko_ddg = new_bucket("KokoulinDDatGramPerMole", 1e-13);
  const int b_ko_el = new_bucket("KokoulinElement", 1e-13);
  const int b_gmole = new_bucket("GramPerMoleConstant", 1e-15);
  std::set<int> kokoulin_z;
  {
    // The muon mass G4KokoulinMuonNuclearXS reads: `G4MuonMinus::MuonMinus()->GetPDGMass()`,
    // for BOTH charges. G4MuonMinus.cc declares 0.1056583715*GeV.
    const double mu_mass = 105.6583715;
    auto* tab = new kokoulin::KokoulinTable<double>();
    kokoulin::build_table<double>(*tab, mu_mass);

    Csv c;
    if (!c.load(dir + "/emextra_kokoulin.csv")) {
      std::printf("FAIL: cannot read %s/emextra_kokoulin.csv\n", dir.c_str());
      ++fails;
    } else {
      for (std::size_t r = 0; r < c.rows.size(); ++r) {
        const std::string kind = c.s(r, "kind");
        if (kind == "dd") {
          const double A = c.d(r, "A_amu");
          const double T = c.d(r, "T_MeV");
          const double ep = c.d(r, "eps_MeV");
          const double got = kokoulin::dd_microscopic_xs<double>(T, A, ep, mu_mass);
          cmp(b_ko_dd, to_mb(got), c.d(r, "value"),
              "A=" + c.s(r, "A_amu") + " T=" + c.s(r, "T_MeV") + " eps=" + c.s(r, "eps_MeV"));
        } else if (kind == "dd_gmole") {
          // The SAME function with the argument G4MuonVDNuclearModel actually passes:
          // `adat[iz]*(g/mole)`, which is 6.24e21 times the amu value. It is a different
          // number by twenty-one orders of magnitude and it is the one the muon model's
          // sampling table is built from.
          const double A = c.d(r, "A_amu") * ee_g_per_mole();
          const double T = c.d(r, "T_MeV");
          const double ep = c.d(r, "eps_MeV");
          const double got = kokoulin::dd_microscopic_xs<double>(T, A, ep, mu_mass);
          cmp(b_ko_ddg, to_mb(got), c.d(r, "value"),
              "A=" + c.s(r, "A_amu") + " T=" + c.s(r, "T_MeV") + " eps=" + c.s(r, "eps_MeV"));
        } else if (kind == "gmole") {
          cmp(b_gmole, ee_g_per_mole(), c.d(r, "value"), "g/mole");
        } else if (kind == "elem") {
          const int Z = c.i(r, "Z");
          kokoulin_z.insert(Z);
          const double T = c.d(r, "T_MeV");
          const XsValue<double> v = kokoulin::element_xs<double>(*tab, T, Z);
          if (!v.ok()) { ++buckets[b_ko_el].refused; continue; }
          cmp(b_ko_el, to_mb(v.value), c.d(r, "value"),
              "Z=" + std::to_string(Z) + " T=" + c.s(r, "T_MeV"));
        }
      }
    }
    delete tab;
  }

  // -------------------------------------------------------------------------------------------
  // Report
  // -------------------------------------------------------------------------------------------
  long long total = 0, total_refused = 0;
  for (const Bucket& b : buckets) {
    total += b.n;
    total_refused += b.refused;
    const bool bad = (b.worst > b.tol);
    std::printf("%-28s %8lld pts  worst %.3e  tol %.0e%s%s%s\n", b.name, b.n, b.worst, b.tol,
                bad ? "  FAIL " : "", bad ? b.where.c_str() : "",
                b.refused > 0 ? ("  refused " + std::to_string(b.refused)).c_str() : "");
    if (bad) { ++fails; }
    if (b.n == 0) {
      std::printf("FAIL: bucket %s compared nothing\n", b.name);
      ++fails;
    }
  }
  std::printf("\nKokoulin element coverage: %d of Z = 1..92 (the run's own elements; the port "
              "builds all 92)\n",
              int(kokoulin_z.size()));
  std::printf("equivalent-photon diagnostics over the grid: %lld *HP* warnings, %lld draws in "
              "the function region, %lld phLE corrections, %lld Newton clamps/exhaustions, "
              "%lld Y[-1] underflows\n",
              n_hp, n_func, n_corr, n_newton, n_underflow);
  std::printf("%lld comparisons, %lld refused\n", total, total_refused);

  if (fails == 0) {
    std::printf("\ntest_emextra_xs: OK\n");
    return 0;
  }
  std::printf("\ntest_emextra_xs: %d FAILURES\n", fails);
  return 1;
}
