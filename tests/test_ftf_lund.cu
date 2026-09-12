// The Lund string decay's samplers against ref/oracle/ftf_build.csv and ftf_samplers.csv.
//
// Exact, under the eight-value uniform cycle ref/dump/dump_ftf.cc installs as Geant4's random
// engine - the same technique and the same eight values as dump_precompound.cc's. Every row
// carries the number of deviates the call consumed as well as its answer, and the count is
// compared: G4HadronBuilder::Meson spends one draw on the spin and a SECOND on the
// scalar/vector mixing only for a neutral light-quark pair, and G4HadronBuilder::Barion spends
// one only for a spin-1/2 baryon of three different flavours whose diquark spin is 1 with the
// heaviest quark outside it. Those are the two draws a transcription omits while still
// producing a plausible hadron, and the count is the only thing that sees it.
//
//   ftf_build.csv      Build / BuildLowSpin / BuildHighSpin for all 8,800 (parton, parton,
//                      phase) triples the tables can reach, with three draw counts.
//   ftf_samplers.csv   SampleQuarkFlavor at four strangeness-suppression values,
//                      SampleQuarkPt at five pt limits, GetLightConeZ over both its arms.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "physics/hadronic/ftf/hadron_builder.cuh"
#include "physics/hadronic/ftf/string_decay.cuh"

using namespace g4gpu;
using namespace g4gpu::hadronic;

/// The eight uniforms ref/dump/dump_ftf.cc's CycleEngine cycles through, in its order. Written
/// out on both sides rather than derived, for the reason tests/test_precompound.cu gives: if
/// the two lists diverge every exact comparison here turns into noise.
///
/// A function-local static is the one form of a constant array nvcc accepts in
/// `__host__ __device__` code, and `CycleRng` has to be device-callable because the samplers
/// it drives are `__host__ __device__` templates.
__host__ __device__ inline const double* cycle_seq() {
  static const double v[8] = {0.05, 0.37, 0.63, 0.91, 0.12, 0.78, 0.29, 0.55};
  return v;
}

struct CycleRng {
  int phase = 0;
  int n = 0;
  __host__ __device__ void reset(int p) { phase = p; n = 0; }
  __host__ __device__ double uniform() {
    const double v = cycle_seq()[(n + phase) % 8];
    ++n;
    return v;
  }
};

/// Never launched; instantiated so that the module is compiled for the device at all, and so
/// that `-Xptxas -v` reports its register and stack cost. See tests/test_ftf_params.cu.
__global__ void ftf_lund_device_probe(ftf::LundTables<double>* lund, int black, int white,
                                      int* out) {
  CycleRng rng;
  rng.reset(0);
  out[0] = ftf::ftf_hadron_build(lund, black, white, rng).pdg;
  out[1] = ftf::ftf_hadron_build_low_spin(lund, black, white, rng).pdg;
  out[2] = ftf::ftf_hadron_build_high_spin(lund, black, white, rng).pdg;
  out[3] = ftf::ftf_sample_quark_flavor(lund, rng);
  double px = 0.0, py = 0.0;
  ftf::ftf_sample_quark_pt(lund, -1.0, rng, &px, &py);
  out[4] = static_cast<int>(px + py);
  out[5] = static_cast<int>(
      ftf::ftf_get_light_cone_z(lund, 0.05, 0.95, 1, 211, 0.0, 0.0, rng) * 1000.0);
}

namespace {

int fails = 0;

struct Bucket {
  const char* name;
  long long n = 0;
  double worst = 0.0;
  std::string where;
  double tol = 1e-13;
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

std::string oracle_dir() {
  const char* e = std::getenv("G4GPU_ORACLE");
  return (e != nullptr && e[0] != '\0') ? std::string(e) : std::string("ref/oracle");
}

std::vector<std::string> read_lines(const std::string& path) {
  std::vector<std::string> out;
  FILE* f = std::fopen(path.c_str(), "r");
  if (f == nullptr) { return out; }
  static char line[65536];
  while (std::fgets(line, sizeof line, f) != nullptr) { out.push_back(line); }
  std::fclose(f);
  return out;
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

double dv(const std::vector<std::string>& f, std::size_t i) {
  return (i < f.size()) ? std::atof(f[i].c_str()) : 0.0;
}
int iv(const std::vector<std::string>& f, std::size_t i) {
  return (i < f.size()) ? std::atoi(f[i].c_str()) : 0;
}

struct Header {
  std::map<std::string, std::size_t> ix;
  std::size_t n = 0;
  void parse(const std::string& line) {
    const std::vector<std::string> f = split(line);
    n = f.size();
    for (std::size_t i = 0; i < f.size(); ++i) { ix[f[i]] = i; }
  }
  std::size_t at(const char* name) const {
    const auto it = ix.find(name);
    if (it == ix.end()) {
      std::printf("FAIL: oracle column '%s' is missing\n", name);
      ++fails;
      return 0;
    }
    return it->second;
  }
};

}  // namespace

int main() {
  const std::string dir = oracle_dir();
  ftf::LundTables<double>* lund = new ftf::LundTables<double>();
  ftf::lund_init(lund, true);

  // -------------------------------------------------------------------------------------------
  // ftf_build.csv
  // -------------------------------------------------------------------------------------------
  const int b_build = new_bucket("HadronBuildPdg", 0.0);
  const int b_draws = new_bucket("HadronBuildDrawCount", 0.0);
  long long n_refused = 0;
  {
    const std::vector<std::string> lines = read_lines(dir + "/ftf_build.csv");
    if (lines.size() < 2) {
      std::printf("FAIL: cannot read %s/ftf_build.csv\n", dir.c_str());
      ++fails;
    }
    Header h;
    h.parse(lines[0]);
    long long rows = 0;
    for (std::size_t li = 1; li < lines.size(); ++li) {
      const std::vector<std::string> f = split(lines[li]);
      if (f.size() != h.n) { continue; }
      ++rows;
      const int black = iv(f, h.at("black"));
      const int white = iv(f, h.at("white"));
      const int phase = iv(f, h.at("phase"));
      char wbuf[96];
      std::snprintf(wbuf, sizeof wbuf, "black=%d white=%d phase=%d", black, white, phase);
      const std::string w = wbuf;

      CycleRng rng;
      rng.reset(phase);
      const ftf::BuiltHadron r1 = ftf::ftf_hadron_build(lund, black, white, rng);
      const int d1 = rng.n;
      rng.reset(phase);
      const ftf::BuiltHadron r2 = ftf::ftf_hadron_build_low_spin(lund, black, white, rng);
      const int d2 = rng.n;
      rng.reset(phase);
      const ftf::BuiltHadron r3 = ftf::ftf_hadron_build_high_spin(lund, black, white, rng);
      const int d3 = rng.n;
      if (r1.refused != ftf::FtfRefusal::kNone || r2.refused != ftf::FtfRefusal::kNone ||
          r3.refused != ftf::FtfRefusal::kNone) {
        ++n_refused;
        continue;
      }
      cmp_int(b_build, r1.pdg, iv(f, h.at("build")), w + " build");
      cmp_int(b_build, r2.pdg, iv(f, h.at("lowspin")), w + " lowspin");
      cmp_int(b_build, r3.pdg, iv(f, h.at("highspin")), w + " highspin");
      cmp_int(b_draws, d1, iv(f, h.at("build_draws")), w + " build_draws");
      cmp_int(b_draws, d2, iv(f, h.at("lowspin_draws")), w + " lowspin_draws");
      cmp_int(b_draws, d3, iv(f, h.at("highspin_draws")), w + " highspin_draws");
    }
    if (rows != 8800) {
      std::printf("FAIL: ftf_build.csv has %lld rows, expected 8800\n", rows);
      ++fails;
    }
  }

  // The two-draw case has to be present, or the draw-count bucket is checking nothing: it is
  // exactly the neutral light-quark meson (id1 + id2 == 0, |id1| < 4), which is what
  // G4HadronBuilder::Meson spends a second deviate on.
  {
    CycleRng rng;
    rng.reset(0);
    (void)ftf::ftf_hadron_build(lund, 1, -1, rng);
    if (rng.n != 2) {
      std::printf("FAIL: Build(d, dbar) consumed %d deviates, expected 2\n", rng.n);
      ++fails;
    }
    rng.reset(0);
    (void)ftf::ftf_hadron_build(lund, 1, 2, rng);
    if (rng.n != 1) {
      std::printf("FAIL: Build(d, u) consumed %d deviates, expected 1\n", rng.n);
      ++fails;
    }
  }

  // BuildLowSpin on three identical flavours must come back spin 3/2, which is the override
  // its own comment admits to. `1103` is dd1 and `1` a d quark: ddd.
  {
    CycleRng rng;
    rng.reset(0);
    const ftf::BuiltHadron r = ftf::ftf_hadron_build_low_spin(lund, 1103, 1, rng);
    if (r.pdg != 1114) {
      std::printf("FAIL: BuildLowSpin(dd1, d) is %d, expected 1114 (Delta-)\n", r.pdg);
      ++fails;
    }
  }

  // A diquark-diquark pair must be refused by name and not answered.
  {
    CycleRng rng;
    rng.reset(0);
    const ftf::BuiltHadron r = ftf::ftf_hadron_build(lund, 2103, -2103, rng);
    if (r.refused != ftf::FtfRefusal::kHadronBuilderIllegalContent) {
      std::printf("FAIL: Build(ud1, anti_ud1) answered %d instead of refusing\n", r.pdg);
      ++fails;
    }
  }

  // -------------------------------------------------------------------------------------------
  // ftf_samplers.csv
  // -------------------------------------------------------------------------------------------
  const int b_flavor = new_bucket("SampleQuarkFlavor", 0.0);
  const int b_pt = new_bucket("SampleQuarkPt", 1e-13);
  const int b_lcz = new_bucket("GetLightConeZ", 1e-13);
  const int b_sdraws = new_bucket("SamplerDrawCount", 0.0);
  {
    const std::vector<std::string> lines = read_lines(dir + "/ftf_samplers.csv");
    if (lines.size() < 2) {
      std::printf("FAIL: cannot read %s/ftf_samplers.csv\n", dir.c_str());
      ++fails;
    }
    Header h;
    h.parse(lines[0]);
    // SampleQuarkPt's x and y come from one call, so the two rows of a (ptmax, phase) pair are
    // compared against one sampling. Keyed on the row's own arg1/phase rather than on row
    // order, so a reordered dump still lines up.
    for (std::size_t li = 1; li < lines.size(); ++li) {
      const std::vector<std::string> f = split(lines[li]);
      if (f.size() != h.n) { continue; }
      const std::string what = f[h.at("what")];
      const double a1 = dv(f, h.at("arg1"));
      const int phase = iv(f, h.at("phase"));
      const double want = dv(f, h.at("value"));
      const int want_draws = iv(f, h.at("draws"));
      char wbuf[128];
      std::snprintf(wbuf, sizeof wbuf, "%s arg1=%g phase=%d", what.c_str(), a1, phase);
      const std::string w = wbuf;

      if (what == "SampleQuarkFlavor") {
        // The oracle sweeps StrangeSuppress because Splitup rewrites it per split from the
        // string mass; the port takes it as a parameter for the same reason.
        CycleRng rng;
        rng.reset(phase);
        const double saved = lund->strange_suppress;
        lund->strange_suppress = a1;
        const int q = ftf::ftf_sample_quark_flavor(lund, rng);
        lund->strange_suppress = saved;
        cmp_int(b_flavor, q, (long long)want, w);
        cmp_int(b_sdraws, rng.n, want_draws, w + " draws");
      } else if (what == "SampleQuarkPt_x" || what == "SampleQuarkPt_y") {
        CycleRng rng;
        rng.reset(phase);
        double px = 0.0, py = 0.0;
        ftf::ftf_sample_quark_pt(lund, a1, rng, &px, &py);
        cmp(b_pt, (what == "SampleQuarkPt_x") ? px : py, want, w);
        cmp_int(b_sdraws, rng.n, want_draws, w + " draws");
      } else if (what == "GetLightConeZ") {
        const int hadron = static_cast<int>(a1);
        const int parton = iv(f, h.at("arg2"));
        const double px = dv(f, h.at("arg3"));
        CycleRng rng;
        rng.reset(phase);
        const double z =
            ftf::ftf_get_light_cone_z(lund, 0.05, 0.95, parton, hadron, px, 0.0, rng);
        char w2[160];
        std::snprintf(w2, sizeof w2, "GetLightConeZ hadron=%d parton=%d px=%g phase=%d",
                      hadron, parton, px, phase);
        cmp(b_lcz, z, want, w2);
        cmp_int(b_sdraws, rng.n, want_draws, std::string(w2) + " draws");
      } else {
        std::printf("FAIL: unknown sampler '%s' in ftf_samplers.csv\n", what.c_str());
        ++fails;
      }
    }
  }

  // -------------------------------------------------------------------------------------------
  // Report
  // -------------------------------------------------------------------------------------------
  std::printf("\n%-32s %10s %14s  %s\n", "bucket", "points", "worst rel", "where");
  long long total = 0;
  for (const Bucket& b : buckets) {
    total += b.n;
    const bool bad = (b.worst > b.tol);
    if (bad) { ++fails; }
    std::printf("%-32s %10lld %14.3e%s %s\n", b.name, b.n, b.worst, bad ? " FAIL" : "",
                b.where.c_str());
    if (b.n == 0) {
      std::printf("FAIL: bucket %s compared nothing\n", b.name);
      ++fails;
    }
  }
  std::printf("\n%lld comparisons, %lld rows refused by name\n", total, n_refused);
  std::printf("%s\n", fails == 0 ? "PASS test_ftf_lund" : "FAIL test_ftf_lund");
  delete lund;
  return fails == 0 ? 0 : 1;
}
