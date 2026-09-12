// The im_r_matrix collision tree's cross sections and angular distributions, against
// ref/oracle/bic_imr_*.csv.
//
// Everything here is EXACT, and not because the quantities are deterministic - two of the three
// are samplers - but because the samplers are driven by a PRESCRIBED uniform sequence on both
// sides. `ref/dump/dump_bic.cc` installs an eight-value cycle engine into CLHEP and the port
// uses the same eight in the same order, which turns `G4AngularDistributionNP::CosTheta` and
// `G4AngularDistribution::CosTheta` into functions of their arguments. That is a much stronger
// check than a histogram: a histogram of 20,000 angles would not see a table read one row off,
// and this does, at the first row it reaches.
//
//   bic_imr_xsec.csv     ten cross-section sources - the four G4CrossSectionPatch composites
//                        and every arm under them - on 240 values of sqrt(s) for each of the
//                        four nucleon pairs. 6,280 points. The grid includes both sides of both
//                        patch boundaries at one part in 1e9, because which arm answers at
//                        3 GeV and at 5 GeV is decided by a `<` against a `<=`.
//   bic_imr_angular_sweep  the same two tables driven by 199 values of the uniform at every one
//                        of their 39 and 40 tabulated energies and at the midpoint between each
//                        pair - 31,044 points, which is what makes the bisection walk the whole
//                        cumulative. The eight-phase file below reaches eight points of each row
//                        and was MEASURED not to notice one table entry moved by 1e-5.
//   bic_imr_angular.csv  G4AngularDistributionNP::CosTheta, G4AngularDistributionPP::CosTheta,
//                        G4AngularDistribution::CosTheta in both its symmetric and asymmetric
//                        forms, and Phi, at 25 energies x 4 mass pairs x 8 phases. 4,000 points,
//                        and the number of uniforms each call consumed is compared too - a
//                        sampler that draws the wrong number of randoms gives the right answer
//                        here and the wrong one in a cascade.
//   bic_imr_obe.csv      G4AngularDistribution::DifferentialCrossSection on a cos(theta) grid,
//                        so that a disagreement in the one-boson-exchange formula is separated
//                        from a disagreement in the twelve halvings that invert it. 4,200 points.
//
// **Why the tolerance is 1e-15 and not zero.** The port and Geant4 evaluate the same expressions
// in the same order in double, so most of these agree bitwise; what they do not share is
// `G4Pow::powA`, whose port (src/data/g4pow.hh) computes its two tables with `std::log`/`std::exp`
// where Geant4 stores them, and `millibarn`, which both derive from `1e-28*m^2` but through
// different products. Both are last-bit effects. The buckets are set where the measurement puts
// them and the commit message records what moves when a term is perturbed.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "physics/hadronic/bic/im_r/angular.cuh"
#include "physics/hadronic/bic/im_r/xsec_nn.cuh"

using namespace g4gpu;
namespace imr = g4gpu::bic::imr;

/// The device probe. The deliverables are `__host__ __device__` templates, and a template that
/// is only ever instantiated from the host is never compiled for the device at all - so this
/// kernel, which is never launched, is what proves the module is device code and what makes
/// `-Xptxas -v` report its register and stack cost.
struct CycleRngDev {
  int n = 0;
  int phase = 0;
  __host__ __device__ double uniform();
};

__global__ void bic_imr_probe(double* out, double s, double m1, double m2) {
  CycleRngDev rng;
  imr::AngularRefusal ref;
  imr::XsecRefusal xref;
  const imr::AngularObeConstants k = imr::angular_obe_constants();
  out[0] = imr::angular_np_cos_theta(s, m1, m2, rng, ref);
  out[1] = imr::angular_pp_cos_theta(s, m1, m2, rng, ref);
  out[2] = imr::angular_obe_cos_theta(k, true, s, m1, m2, rng, ref);
  out[3] = imr::x_nn_total(2212, 2212, m1, m2, std::sqrt(s), xref);
  out[4] = imr::x_nn_elastic(2212, 2212, m1, m2, std::sqrt(s), xref);
  out[5] = imr::x_np_elastic(2112, 2212, m1, m2, std::sqrt(s), xref);
}

namespace {

int fails = 0;

// ---------------------------------------------------------------------------------------------
// Comparison bookkeeping - the same shape tests/test_bic_nucleus.cu uses
// ---------------------------------------------------------------------------------------------

struct Bucket {
  const char* name;
  long long n = 0;
  double worst = 0.0;
  std::string where;
  double tol = 1e-15;
};

std::vector<Bucket> buckets;

int new_bucket(const char* name, double tol) {
  Bucket b;
  b.name = name;
  b.tol = tol;
  buckets.push_back(b);
  return static_cast<int>(buckets.size()) - 1;
}

/// The plain relative `cmp` other tests in this package have is deliberately absent: every
/// quantity here crosses zero somewhere - a cross section below its threshold, a cosine at 90
/// degrees, the symmetric cumulative at cos = 0 where its two halves cancel exactly - and a
/// relative comparison at those points divides a rounding by a cancellation. Only the floored
/// form below is used.

/// `cmp` with an absolute floor, for a quantity that crosses zero. `cos(theta)` does, and so
/// does the symmetric OBE cumulative at cos(theta) = 0 where it is exactly 0.5 by construction
/// and its two halves cancel to a rounding.
void cmp_scaled(int bi, double got, double want, double scale, const std::string& where) {
  Bucket& b = buckets[bi];
  ++b.n;
  const double denom = (std::fabs(want) > scale) ? std::fabs(want) : scale;
  const double rel = std::fabs(got - want) / denom;
  if (rel > b.worst) {
    b.worst = rel;
    b.where = where;
  }
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

// ---------------------------------------------------------------------------------------------
// CSV reading
// ---------------------------------------------------------------------------------------------

std::string oracle_dir() {
  const char* env = std::getenv("G4GPU_ORACLE");
  return (env != nullptr) ? std::string(env) : std::string("ref/oracle");
}

std::vector<std::vector<std::string>> read_csv(const std::string& name) {
  std::vector<std::vector<std::string>> rows;
  const std::string path = oracle_dir() + "/" + name;
  FILE* f = std::fopen(path.c_str(), "r");
  if (f == nullptr) {
    std::printf("MISSING %s\n", path.c_str());
    ++fails;
    return rows;
  }
  char line[8192];
  bool header = true;
  while (std::fgets(line, sizeof line, f) != nullptr) {
    if (header) { header = false; continue; }
    std::vector<std::string> f2;
    std::string cur;
    for (const char* p = line; *p != '\0'; ++p) {
      if (*p == ',') { f2.push_back(cur); cur.clear(); }
      else if (*p != '\n' && *p != '\r') { cur.push_back(*p); }
    }
    f2.push_back(cur);
    if (!f2.empty()) { rows.push_back(f2); }
  }
  std::fclose(f);
  return rows;
}

double dv(const std::vector<std::string>& f, std::size_t i) {
  return (i < f.size()) ? std::strtod(f[i].c_str(), nullptr) : 0.0;
}
int iv(const std::vector<std::string>& f, std::size_t i) {
  return (i < f.size()) ? std::atoi(f[i].c_str()) : 0;
}
std::string sv(const std::vector<std::string>& f, std::size_t i) {
  return (i < f.size()) ? f[i] : std::string();
}

// ---------------------------------------------------------------------------------------------
// The prescribed uniform cycle, host side: the same eight values in the same order as
// ref/dump/dump_bic.cc's ImrCycleEngine, and a counter so the test can assert that the port
// drew as many uniforms as Geant4 did.
// ---------------------------------------------------------------------------------------------

struct CycleRng {
  int n = 0;
  int phase = 0;
  void reset(int p) { phase = p; n = 0; }
  // `__host__ __device__` although this one only ever runs on the host: the samplers it is
  // handed to are `__host__ __device__` templates, and nvcc warns 20011 about every
  // instantiation otherwise - a warning in every build that would hide a real one.
  __host__ __device__ double uniform() {
    // The eight are written inside the function rather than read from the namespace-scope
    // `kSeq`: a host `const double[]` referenced from `__host__ __device__` code is warning
    // 20014, and the point of the annotation was to have no warnings.
    const double seq[8] = {0.05, 0.37, 0.63, 0.91, 0.12, 0.78, 0.29, 0.55};
    const double v = seq[(n + phase) % 8];
    ++n;
    return v;
  }
};

}  // namespace

/// The device probe's RNG, out of line so the struct above stays a plain aggregate.
__host__ __device__ double CycleRngDev::uniform() {
  const double seq[8] = {0.05, 0.37, 0.63, 0.91, 0.12, 0.78, 0.29, 0.55};
  const double v = seq[(n + phase) % 8];
  ++n;
  return v;
}

int main() {
  const double mp = 938.27201300000002;  // G4Proton's PDG mass, as the oracle prints it
  const double mn = 939.56536000000002;  // G4Neutron's

  // -------------------------------------------------------------------------------------------
  // 1. The cross sections.
  // -------------------------------------------------------------------------------------------
  const int b_patch = new_bucket("XsecPatch (the four composites)", 1e-15);
  const int b_lowe = new_bucket("XsecLowE (the four tabulated arms)", 1e-15);
  const int b_pdg = new_bucket("XsecPDG (the two fits)", 2e-15);

  {
    const auto rows = read_csv("bic_imr_xsec.csv");
    for (const auto& r : rows) {
      const std::string pair = sv(r, 0);
      const std::string src = sv(r, 1);
      const double sqrt_s = dv(r, 2);
      const double want = dv(r, 3);  // millibarn
      int pdg1 = 0, pdg2 = 0;
      double m1 = 0.0, m2 = 0.0;
      if (pair == "pp") { pdg1 = pdg2 = 2212; m1 = m2 = mp; }
      else if (pair == "nn") { pdg1 = pdg2 = 2112; m1 = m2 = mn; }
      else if (pair == "np") { pdg1 = 2112; pdg2 = 2212; m1 = mn; m2 = mp; }
      else { pdg1 = 2212; pdg2 = 2112; m1 = mp; m2 = mn; }

      imr::XsecRefusal ref;
      double got = 0.0;
      int bucket = b_patch;
      if (src == "XNNTotal") { got = imr::x_nn_total(pdg1, pdg2, m1, m2, sqrt_s, ref); }
      else if (src == "XNNElastic") { got = imr::x_nn_elastic(pdg1, pdg2, m1, m2, sqrt_s, ref); }
      else if (src == "XnpElastic") { got = imr::x_np_elastic(pdg1, pdg2, m1, m2, sqrt_s, ref); }
      else if (src == "XnpTotal") { got = imr::x_np_total(pdg1, pdg2, m1, m2, sqrt_s, ref); }
      else if (src == "XNNTotalLowE") {
        got = imr::x_nn_total_lowe(pdg1, pdg2, sqrt_s, ref);
        bucket = b_lowe;
      } else if (src == "XNNElasticLowE") {
        got = imr::x_nn_elastic_lowe(pdg1, pdg2, sqrt_s, ref);
        bucket = b_lowe;
      } else if (src == "XnpElasticLowE") {
        got = imr::x_np_elastic_lowe(pdg1, pdg2, sqrt_s);
        bucket = b_lowe;
      } else if (src == "XnpTotalLowE") {
        got = imr::x_np_total_lowe(pdg1, pdg2, sqrt_s);
        bucket = b_lowe;
      } else if (src == "XPDGTotal") {
        got = imr::x_pdg_total(pdg1, pdg2, m1, m2, sqrt_s, ref);
        bucket = b_pdg;
      } else if (src == "XPDGElastic") {
        got = imr::x_pdg_elastic(pdg1, pdg2, m1, m2, sqrt_s, ref);
        bucket = b_pdg;
      } else {
        continue;
      }
      // Every one of these four pairs is one the arms answer for, so a refusal here is a
      // transcription error and not a species the port declines.
      if (ref.any()) {
        std::printf("REFUSED %s %s at sqrt(s)=%.6g\n", pair.c_str(), src.c_str(), sqrt_s);
        ++fails;
        continue;
      }
      cmp_scaled(bucket, got / imr::millibarn(), want, 1e-6,
                 pair + " " + src + " sqrt(s)=" + std::to_string(sqrt_s));
    }
  }

  // -------------------------------------------------------------------------------------------
  // 2. The angular distributions, under the prescribed cycle.
  // -------------------------------------------------------------------------------------------
  const int b_tbl = new_bucket("AngularTable (NP and PP CosTheta)", 1e-15);
  const int b_obe = new_bucket("AngularOBE (CosTheta, both sym)", 1e-15);
  const int b_phi = new_bucket("AngularPhi", 1e-15);
  const int b_draws = new_bucket("AngularDraws (uniforms consumed)", 0.0);

  {
    const imr::AngularObeConstants k = imr::angular_obe_constants();
    const auto rows = read_csv("bic_imr_angular.csv");
    for (const auto& r : rows) {
      const std::string dist = sv(r, 0);
      const double s = dv(r, 1);
      const double m1 = dv(r, 2);
      const double m2 = dv(r, 3);
      const int phase = iv(r, 4);
      const double want = dv(r, 5);
      const int want_draws = iv(r, 6);
      CycleRng rng;
      rng.reset(phase);
      imr::AngularRefusal ref;
      double got = 0.0;
      int bucket = b_tbl;
      const std::string where =
          dist + " s=" + std::to_string(s) + " m1=" + std::to_string(m1) +
          " m2=" + std::to_string(m2) + " phase=" + std::to_string(phase);
      if (dist == "NP") { got = imr::angular_np_cos_theta(s, m1, m2, rng, ref); }
      else if (dist == "PP") { got = imr::angular_pp_cos_theta(s, m1, m2, rng, ref); }
      else if (dist == "OBEsym") {
        got = imr::angular_obe_cos_theta(k, true, s, m1, m2, rng, ref);
        bucket = b_obe;
      } else if (dist == "OBEasym") {
        got = imr::angular_obe_cos_theta(k, false, s, m1, m2, rng, ref);
        bucket = b_obe;
      } else if (dist == "Phi") {
        got = imr::angular_phi(rng);
        bucket = b_phi;
      } else {
        continue;
      }
      if (ref.bisection_budget) {
        std::printf("REFUSED bisection budget: %s\n", where.c_str());
        ++fails;
      }
      cmp_scaled(bucket, got, want, 1e-3, where);
      cmp_int(b_draws, rng.n, want_draws, where);
    }
  }

  // -------------------------------------------------------------------------------------------
  // 2b. The table sweep - the same two samplers driven by 199 values of the uniform at each of
  //     the 39 and 40 tabulated energies and at the midpoint between each pair, which is what
  //     makes the bisection walk the WHOLE cumulative instead of eight points of it.
  //
  //     This block exists because the eight-phase block above was measured not to be enough:
  //     moving one entry of the 7,020-value NP table by one part in 65,000 changed none of its
  //     1,600 angles. It changes this block's.
  // -------------------------------------------------------------------------------------------
  const int b_sweep = new_bucket("AngularSweep (whole cumulative)", 1e-15);
  {
    const auto rows = read_csv("bic_imr_angular_sweep.csv");
    for (const auto& r : rows) {
      const std::string dist = sv(r, 0);
      const double s = dv(r, 1);
      const double m1 = dv(r, 2);
      const double m2 = dv(r, 3);
      const double sample = dv(r, 4);
      const double want = dv(r, 5);
      // One prescribed value, served to every draw - the port's mirror of ImrFixedEngine.
      struct FixedRng {
        double v;
        __host__ __device__ double uniform() { return v; }
      } rng{sample};
      imr::AngularRefusal ref;
      double got = 0.0;
      if (dist == "NP") { got = imr::angular_np_cos_theta(s, m1, m2, rng, ref); }
      else if (dist == "PP") { got = imr::angular_pp_cos_theta(s, m1, m2, rng, ref); }
      else { continue; }
      cmp_scaled(b_sweep, got, want, 1e-3,
                 dist + " s=" + std::to_string(s) + " sample=" + std::to_string(sample));
    }
  }

  // -------------------------------------------------------------------------------------------
  // 3. The one-boson-exchange cumulative itself.
  // -------------------------------------------------------------------------------------------
  const int b_dsig = new_bucket("ObeDifferentialCrossSection", 1e-14);
  {
    const imr::AngularObeConstants k = imr::angular_obe_constants();
    const auto rows = read_csv("bic_imr_obe.csv");
    for (const auto& r : rows) {
      const bool sym = iv(r, 0) != 0;
      const double s = dv(r, 1);
      const double m1 = dv(r, 2);
      const double m2 = dv(r, 3);
      const double ct = dv(r, 4);
      const double want = dv(r, 5);
      const double got = imr::angular_obe_dsigma(k, sym, s, m1, m2, ct);
      cmp_scaled(b_dsig, got, want, 1e-6,
                 std::string(sym ? "sym" : "asym") + " s=" + std::to_string(s) +
                     " cos=" + std::to_string(ct));
    }
  }

  // -------------------------------------------------------------------------------------------
  // 4. Structural assertions on the extracted tables. These are not oracle comparisons - they
  //    are the invariants the sampling depends on, checked on the port's own copy so that a
  //    mis-sliced table fails here rather than as a wrong angle 4,000 rows later.
  // -------------------------------------------------------------------------------------------
  const int b_tab = new_bucket("TableInvariants", 0.0);
  {
    for (int j = 0; j < imr::kAngularNpEnergies; ++j) {
      const float last = imr::angular_np_sig()[j * imr::kAngularAngles + imr::kAngularAngles - 1];
      cmp_int(b_tab, (std::fabs(static_cast<double>(last) - 1.0) < 2e-5) ? 1 : 0, 1,
              "NP row " + std::to_string(j) + " cumulative ends at 1");
      if (j > 0) {
        cmp_int(b_tab, (imr::angular_np_elab()[j] > imr::angular_np_elab()[j - 1]) ? 1 : 0, 1,
                "NP elab increasing at " + std::to_string(j));
      }
    }
    for (int j = 0; j < imr::kAngularPpEnergies; ++j) {
      const float last = imr::angular_pp_sig()[j * imr::kAngularAngles + imr::kAngularAngles - 1];
      cmp_int(b_tab, (std::fabs(static_cast<double>(last) - 1.0) < 2e-5) ? 1 : 0, 1,
              "PP row " + std::to_string(j) + " cumulative ends at 1");
      if (j > 0) {
        cmp_int(b_tab, (imr::angular_pp_elab()[j] > imr::angular_pp_elab()[j - 1]) ? 1 : 0, 1,
                "PP elab increasing at " + std::to_string(j));
      }
    }
    // The 29th slot of G4XNNTotalLowE::ss is an uninitialised zero (see xsec_nn.cuh). Asserted
    // here rather than assumed, because if a release ever fills it the extractor's count check
    // fires first - and if someone "fixes" the extractor by relaxing the count, this fires.
    cmp_int(b_tab, (imr::nn_total_lowe_ss()[imr::kNNTotalLowESize - 1] == 0.0) ? 1 : 0, 1,
            "G4XNNTotalLowE::ss[28] is the zero-filled 29th slot");
    cmp_int(b_tab,
            (imr::nn_total_lowe_ss()[imr::kNNTotalLowESize - 2] == 3002.71) ? 1 : 0, 1,
            "G4XNNTotalLowE::ss[27] is the last real energy");
    // The stretched np grid: the pp vector's top filled node is at exactly one log unit above
    // its edgeMin and the np vector's is 1% past where its table means it to be. Both measured
    // from the port's own LogVec101, which is what reads them.
    const imr::LogVec101 vpp = imr::LogVec101::make(
        imr::nn_elastic_lowe_pp(), imr::lowe_emin_pp(), imr::lowe_emax());
    const imr::LogVec101 vnp = imr::LogVec101::make(
        imr::nn_elastic_lowe_np(), imr::lowe_emin_np(), imr::lowe_emax());
    const double log_span_pp = std::log(vpp.node_energy(100) / vpp.node_energy(0));
    const double log_span_np = std::log(vnp.node_energy(100) / vnp.node_energy(0));
    cmp_int(b_tab, (std::fabs(log_span_pp - 1.00) < 1e-12) ? 1 : 0, 1,
            "pp grid spans exactly 1.00 in log over its 101 filled nodes");
    cmp_int(b_tab, (std::fabs(log_span_np - 1.0099009900990099) < 1e-12) ? 1 : 0, 1,
            "np grid spans 1.00990 where its table means 0.99");
  }

  // -------------------------------------------------------------------------------------------
  std::printf("\n%-38s %10s %14s  %s\n", "bucket", "points", "worst", "where");
  for (const Bucket& b : buckets) {
    const bool bad = b.worst > b.tol;
    if (bad) { ++fails; }
    std::printf("%-38s %10lld %14.4g  %s%s\n", b.name, b.n, b.worst, bad ? "FAIL " : "",
                b.where.c_str());
  }
  std::printf("\ntest_bic_imr: %s\n", (fails == 0) ? "PASS" : "FAIL");
  return (fails == 0) ? 0 : 1;
}
