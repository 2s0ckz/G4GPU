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

#include "core/rng.cuh"
#include "physics/hadronic/ftf/hadron_builder.cuh"
#include "physics/hadronic/ftf/lund_fragment.cuh"
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
///
/// BOTH the tables and the workspace are POINTERS, and that is the whole design of this module
/// rather than a convention: `FragmentWorkspace<double>` is 9,048 bytes - a 350-entry
/// final-state enumeration and three hadron lists - and a kernel that put it on the stack would
/// spend more on one string than the 16,384-byte frame `Upload` allows for a whole step. The
/// frame this probe reports is what the fragmentation costs with the workspace in global
/// memory; the workspace itself is the caller's, one per track.
__global__ void ftf_lund_device_probe(ftf::LundTables<double>* lund,
                                      ftf::FragmentWorkspace<double>* ws, int black, int white,
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
  // The whole chain: FragmentString -> Loop_toFragmentString -> Splitup/SplitEandP -> SplitLast
  // and the three last-splitting enumerations, plus the ProduceOneHadron arm through the second
  // call, whose string is too light to fragment.
  ftf::ftf_fragment_string(lund, ws, 1, -1, ftf::Vec4(0.0, 0.0, 2500.0, 2500.0),
                           ftf::Vec4(0.0, 0.0, -2500.0, 2500.0), 1, 0.0,
                           ftf::Vec3d{0.0, 0.0, 0.0}, rng);
  out[6] = ws->n_out;
  ftf::ftf_fragment_string(lund, ws, 1, -1, ftf::Vec4(0.0, 0.0, 150.0, 150.0),
                           ftf::Vec4(0.0, 0.0, -150.0, 150.0), 1, 0.0,
                           ftf::Vec3d{0.0, 0.0, 0.0}, rng);
  out[7] = ws->n_out + ws->number_of_fs + static_cast<int>(ws->refused);
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
      } else if (what == "SampleQuarkFlavorHeavy") {
        // The c/b branch, reachable only because SetProbCCbar and SetProbBBbar are public and
        // unlocked - at the physical 2.5e-4 the cycle's 0.05 never enters it. arg1 is
        // ProbCCbar and arg2 ProbBBbar; ProbCB is their sum, as SetProbCCbar computes it.
        const double pb = dv(f, h.at("arg2"));
        CycleRng rng;
        rng.reset(phase);
        const double sc = lund->prob_ccbar, sb = lund->prob_bbbar, scb = lund->prob_cb;
        lund->prob_ccbar = a1;
        lund->prob_bbbar = pb;
        lund->prob_cb = a1 + pb;
        const int q = ftf::ftf_sample_quark_flavor(lund, rng);
        lund->prob_ccbar = sc;
        lund->prob_bbbar = sb;
        lund->prob_cb = scb;
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
  // ftf_decisions.csv - IsItFragmentable, StopFragmenting and Sample4Momentum on their own
  // -------------------------------------------------------------------------------------------
  const int b_dec = new_bucket("FragmentationDecisions", 0.0);
  const int b_s4m = new_bucket("Sample4Momentum", 1e-13);
  {
    const std::vector<std::string> lines = read_lines(dir + "/ftf_decisions.csv");
    if (lines.size() < 2) {
      std::printf("FAIL: cannot read %s/ftf_decisions.csv\n", dir.c_str());
      ++fails;
    }
    Header h;
    h.parse(lines[0]);
    for (std::size_t li = 1; li < lines.size(); ++li) {
      const std::vector<std::string> f = split(lines[li]);
      if (f.size() != h.n) { continue; }
      const int lc = iv(f, h.at("left"));
      const int rc = iv(f, h.at("right"));
      const double mass = dv(f, h.at("mass"));
      const int phase = iv(f, h.at("phase"));
      const double m1 = dv(f, h.at("m1"));
      const double m2 = dv(f, h.at("m2"));
      char wbuf[160];
      std::snprintf(wbuf, sizeof wbuf, "left=%d right=%d M=%g phase=%d m1=%g m2=%g", lc, rc,
                    mass, phase, m1, m2);
      const std::string w = wbuf;

      const ftf::MinimalStringMass<double> mm =
          ftf::ftf_minimal_string_mass(lund, lc, rc, mass);
      cmp(b_dec, mm.mass, dv(f, h.at("minimal_mass")), w + " minimal_mass");
      cmp_int(b_dec, ftf::ftf_is_it_fragmentable(mm.mass, mass) ? 1 : 0,
              iv(f, h.at("fragmentable")), w + " fragmentable");
      const bool is4q = iv(f, h.at("is4q")) != 0;
      CycleRng rng;
      rng.reset(phase);
      const bool stop = ftf::ftf_stop_fragmenting(mm.mass, mass, is4q, rng);
      cmp_int(b_dec, stop ? 1 : 0, iv(f, h.at("stop")), w + " stop");
      cmp_int(b_dec, rng.n, iv(f, h.at("stop_draws")), w + " stop_draws");

      rng.reset(phase);
      const ftf::TwoBodyMomenta<double> tb =
          ftf::ftf_sample_4momentum(lund, m1, m2, mass, rng);
      cmp(b_s4m, tb.px, dv(f, h.at("s4m_px")), w + " px");
      cmp(b_s4m, tb.py, dv(f, h.at("s4m_py")), w + " py");
      cmp(b_s4m, tb.pz, dv(f, h.at("s4m_pz")), w + " pz");
      cmp(b_s4m, tb.e, dv(f, h.at("s4m_e")), w + " e");
      cmp(b_s4m, tb.apx, dv(f, h.at("s4m_apx")), w + " apx");
      cmp(b_s4m, tb.apy, dv(f, h.at("s4m_apy")), w + " apy");
      cmp(b_s4m, tb.apz, dv(f, h.at("s4m_apz")), w + " apz");
      cmp(b_s4m, tb.ae, dv(f, h.at("s4m_ae")), w + " ae");
      cmp_int(b_s4m, rng.n, iv(f, h.at("s4m_draws")), w + " s4m_draws");
    }
  }

  // -------------------------------------------------------------------------------------------
  // ftf_fragment.csv - the whole FragmentString chain, hadron by hadron
  // -------------------------------------------------------------------------------------------
  const int b_nhad = new_bucket("FragmentMultiplicity", 0.0);
  const int b_hpdg = new_bucket("FragmentHadronPdg", 0.0);
  const int b_hmom = new_bucket("FragmentHadronMomentum", 1e-12);
  const int b_htime = new_bucket("FragmentFormationTime", 1e-13);
  const int b_hdraws = new_bucket("FragmentDrawCount", 0.0);
  {
    const std::vector<std::string> lines = read_lines(dir + "/ftf_fragment.csv");
    if (lines.size() < 2) {
      std::printf("FAIL: cannot read %s/ftf_fragment.csv\n", dir.c_str());
      ++fails;
    }
    Header h;
    h.parse(lines[0]);
    ftf::FragmentWorkspace<double>* ws = new ftf::FragmentWorkspace<double>();
    std::string last_key;
    int n_got = 0;
    int got_draws = 0;
    for (std::size_t li = 1; li < lines.size(); ++li) {
      const std::vector<std::string> f = split(lines[li]);
      if (f.size() != h.n) { continue; }
      const int lc = iv(f, h.at("left"));
      const int rc = iv(f, h.at("right"));
      const double mass = dv(f, h.at("mass"));
      const int direction = iv(f, h.at("direction"));
      const int phase = iv(f, h.at("phase"));
      char kbuf[128];
      std::snprintf(kbuf, sizeof kbuf, "%s dir=%d phase=%d", f[h.at("case")].c_str(),
                    direction, phase);
      const std::string k = kbuf;
      if (k != last_key) {
        CycleRng rng;
        rng.reset(phase);
        const double half = 0.5 * mass;
        const ftf::Vec4 lm(0.0, 0.0, half, half);
        const ftf::Vec4 rm(0.0, 0.0, -half, half);
        ftf::ftf_fragment_string(lund, ws, lc, rc, lm, rm, direction, 0.0,
                                 ftf::Vec3d{0.0, 0.0, 0.0}, rng);
        n_got = ws->n_out;
        got_draws = rng.n;
        last_key = k;
        cmp_int(b_nhad, n_got, iv(f, h.at("nhadrons")), k + " nhadrons");
        cmp_int(b_hdraws, got_draws, iv(f, h.at("draws")), k + " draws");
      }
      const int idx = iv(f, h.at("index"));
      if (idx < 0) { continue; }  // the "no hadrons" marker row
      if (idx >= n_got) { continue; }
      const std::string w = k + " hadron " + std::to_string(idx);
      cmp_int(b_hpdg, ws->out[idx].pdg, iv(f, h.at("pdg")), w + " pdg");
      cmp(b_hmom, ws->out[idx].momentum.v.x, dv(f, h.at("px")), w + " px");
      cmp(b_hmom, ws->out[idx].momentum.v.y, dv(f, h.at("py")), w + " py");
      cmp(b_hmom, ws->out[idx].momentum.v.z, dv(f, h.at("pz")), w + " pz");
      cmp(b_hmom, ws->out[idx].momentum.e, dv(f, h.at("e")), w + " e");
      cmp(b_htime, ws->out[idx].formation_time, dv(f, h.at("formation_time")), w + " time");
    }
    delete ws;
  }

  // -------------------------------------------------------------------------------------------
  // ftf_laststates.csv - the final-state ENUMERATION, not the sampled index
  // -------------------------------------------------------------------------------------------
  //
  // This file exists because the sampled index is a lossy view of the enumeration. SplitLast
  // lists up to 350 (left, right, weight) triples and SampleState collapses them to one
  // number, so a weight change of a few per cent moves no index: measured, two perturbations
  // of this module's campaign - Diquark_AntiDiquark_aboveThreshold's |p|^3 weight made |p|,
  // and Prob_QQbar[3] raised from 0 to 0.33 - changed nothing in ftf_fragment.csv AND nothing
  // at 20,000 events per case (about 0.7 sigma), because the eighteen enumerated states of a
  // qq-qqbar string have weights within a factor 1.2 of each other. Dumping the list makes
  // both exact. Geant4's FS_LeftHadron / FS_RightHadron / FS_Weight and NumberOf_FS are public
  // members; the three functions that fill them are private and are reached in the dump
  // through the explicit-instantiation route ref/dump/dump_ftf.cc's header explains.
  const int b_fsn = new_bucket("LastStateCount", 0.0);
  const int b_fslist = new_bucket("LastStateList", 0.0);
  const int b_fsw = new_bucket("LastStateWeight", 1e-12);
  {
    const std::vector<std::string> lines = read_lines(dir + "/ftf_laststates.csv");
    if (lines.size() < 2) {
      std::printf("FAIL: cannot read %s/ftf_laststates.csv\n", dir.c_str());
      ++fails;
    }
    Header h;
    h.parse(lines[0]);
    ftf::FragmentWorkspace<double>* ws = new ftf::FragmentWorkspace<double>();
    std::string last_key;
    for (std::size_t li = 1; li < lines.size(); ++li) {
      const std::vector<std::string> f = split(lines[li]);
      if (f.size() != h.n) { continue; }
      const std::string which = f[h.at("which")];
      const int lc = iv(f, h.at("left"));
      const int rc = iv(f, h.at("right"));
      const double mass = dv(f, h.at("mass"));
      char kbuf[128];
      std::snprintf(kbuf, sizeof kbuf, "%s %d %d %g", which.c_str(), lc, rc, mass);
      const std::string k = kbuf;
      if (k != last_key) {
        last_key = k;
        const double half = 0.5 * mass;
        ftf::FragmentingString s = ftf::ftf_string_from_excited(
            lc, rc, ftf::Vec4(0.0, 0.0, half, half), ftf::Vec4(0.0, 0.0, -half, half), 1);
        ftf::ftf_set_left_parton_stable(&s);
        ftf::ftf_set_minimal_string_mass(lund, ws, s);
        ws->number_of_fs = 0;
        for (int i = 0; i < ftf::FragmentWorkspace<double>::kMaxFinalStates; ++i) {
          ws->fs_weight[i] = 0.0;
        }
        bool ok = false;
        if (which == "DiQ-ADiQ-above") {
          ok = ftf::ftf_diquark_antidiquark_above_threshold(lund, ws, s);
        } else if (which == "Q-Qbar") {
          ok = ftf::ftf_quark_antiquark_last_splitting(lund, ws, s);
        } else {
          ok = ftf::ftf_quark_diquark_last_splitting(lund, ws, s);
        }
        cmp_int(b_fsn, ok ? 1 : 0, iv(f, h.at("ok")), k + " ok");
        cmp_int(b_fsn, ws->number_of_fs, iv(f, h.at("number_of_fs")), k + " number_of_fs");
      }
      const int idx = iv(f, h.at("index"));
      if (idx < 0 || idx >= ws->number_of_fs) { continue; }
      const std::string w = k + " fs[" + std::to_string(idx) + "]";
      cmp_int(b_fslist, ws->fs_left[idx], iv(f, h.at("fs_left")), w + " left");
      cmp_int(b_fslist, ws->fs_right[idx], iv(f, h.at("fs_right")), w + " right");
      cmp(b_fsw, ws->fs_weight[idx], dv(f, h.at("fs_weight")), w + " weight");
    }
    delete ws;
  }

  // -------------------------------------------------------------------------------------------
  // ftf_fragstat.csv / ftf_fragstat_mult.csv - the statistical half
  // -------------------------------------------------------------------------------------------
  //
  // WHY THIS SECTION EXISTS AND THE EXACT ONE IS NOT ENOUGH. The eight-value cycle pins every
  // function, but it is only eight points: SplitLast's final-state choice for a qq-qqbar
  // string enumerates eighteen states whose cumulative probabilities sit 2 to 5 per cent away
  // from the nearest of the eight deviates, so a weight change of that size moves no sampled
  // index. Two perturbations of this campaign - the |p|^3 in
  // Diquark_AntiDiquark_aboveThreshold's weight made |p|, and Prob_QQbar[3] raised from 0 to
  // 0.33 - are exactly that size and were NOT caught by the exact half. 20,000 events per case
  // are, because a 5% shift in a species fraction at n ~ 2,000 is 30 sigma.
  //
  // The comparison is statistical and not exact because the port's engine is Philox and the
  // oracle's is HepJamesRandom. `z = (n1 - n2)/sqrt(n1 + n2)` is the multinomial count
  // comparison with the variance taken as Poisson, which OVERESTIMATES it for a species that
  // is nearly every hadron (var = N p (1-p) < n) - conservative in the direction of fewer
  // false alarms, which is the direction docs/RISK.md V1 says to be careful about.
  const int b_stat = new_bucket("FragStatSpeciesSigma", 5.0);
  const int b_statmult = new_bucket("FragStatMultiplicitySigma", 5.0);
  {
    const std::vector<std::string> lines = read_lines(dir + "/ftf_fragstat.csv");
    const std::vector<std::string> mlines = read_lines(dir + "/ftf_fragstat_mult.csv");
    if (lines.size() < 2 || mlines.size() < 2) {
      std::printf("FAIL: cannot read %s/ftf_fragstat*.csv\n", dir.c_str());
      ++fails;
    }
    Header h, mh;
    h.parse(lines[0]);
    mh.parse(mlines[0]);

    // The oracle's cases, in its own order, keyed by name.
    struct Case { std::string name; int left, right; double mass; int n; };
    std::vector<Case> cases;
    std::map<std::string, std::map<int, long long>> want_species;
    std::map<std::string, std::map<int, long long>> want_mult;
    // n_events comes from the oracle row rather than from a constant here. The two sides have
    // to run the SAME number of events or the z below is not a z at all, and a 20,000 written
    // in two files is exactly the constant that goes out of step when the campaign is
    // enlarged - which is how this module's 10x run was done.
    std::map<std::string, int> want_n;
    for (std::size_t li = 1; li < lines.size(); ++li) {
      const std::vector<std::string> f = split(lines[li]);
      if (f.size() != h.n) { continue; }
      want_species[f[h.at("case")]][iv(f, h.at("pdg"))] = std::atoll(f[h.at("count")].c_str());
      want_n[f[h.at("case")]] = iv(f, h.at("n_events"));
    }
    for (std::size_t li = 1; li < mlines.size(); ++li) {
      const std::vector<std::string> f = split(mlines[li]);
      if (f.size() != mh.n) { continue; }
      want_mult[f[mh.at("case")]][iv(f, mh.at("multiplicity"))] =
          std::atoll(f[mh.at("count")].c_str());
    }
    // The (left, right, mass) of each case comes from ftf_fragment.csv, which carries them.
    {
      const std::vector<std::string> fl = read_lines(dir + "/ftf_fragment.csv");
      Header fh;
      if (!fl.empty()) { fh.parse(fl[0]); }
      std::map<std::string, Case> seen;
      for (std::size_t li = 1; li < fl.size(); ++li) {
        const std::vector<std::string> f = split(fl[li]);
        if (f.size() != fh.n) { continue; }
        const std::string nm = f[fh.at("case")];
        if (seen.count(nm)) { continue; }
        Case c;
        c.name = nm;
        c.left = iv(f, fh.at("left"));
        c.right = iv(f, fh.at("right"));
        c.mass = dv(f, fh.at("mass"));
        c.n = want_n.count(nm) ? want_n[nm] : 0;
        seen[nm] = c;
        cases.push_back(c);
      }
    }

    ftf::FragmentWorkspace<double>* ws = new ftf::FragmentWorkspace<double>();
    for (const Case& c : cases) {
      if (want_species.count(c.name) == 0) { continue; }
      std::map<int, long long> got_species;
      std::map<int, long long> got_mult;
      for (int ev = 0; ev < c.n; ++ev) {
        Philox<double> rng(0x46544601u, static_cast<uint32_t>(ev), 0u);
        const double half = 0.5 * c.mass;
        ftf::ftf_fragment_string(lund, ws, c.left, c.right, ftf::Vec4(0.0, 0.0, half, half),
                                 ftf::Vec4(0.0, 0.0, -half, half), 1, 0.0,
                                 ftf::Vec3d{0.0, 0.0, 0.0}, rng);
        got_mult[ws->n_out]++;
        for (int i = 0; i < ws->n_out; ++i) { got_species[ws->out[i].pdg]++; }
      }
      // Every species either side contributes one comparison; a species one side produced and
      // the other did not is compared against zero, which is what makes a missing channel
      // visible rather than absent from the tally.
      std::map<int, int> keys;
      for (const auto& kv : want_species[c.name]) { keys[kv.first] = 1; }
      for (const auto& kv : got_species) { keys[kv.first] = 1; }
      for (const auto& kv : keys) {
        const long long n2 = want_species[c.name].count(kv.first)
                                 ? want_species[c.name][kv.first] : 0;
        const long long n1 = got_species.count(kv.first) ? got_species[kv.first] : 0;
        const double denom = std::sqrt(static_cast<double>(n1 + n2));
        const double z = (denom > 0.0) ? std::fabs(double(n1 - n2)) / denom : 0.0;
        char w[128];
        std::snprintf(w, sizeof w, "%s pdg=%d got %lld want %lld", c.name.c_str(), kv.first,
                      n1, n2);
        cmp(b_stat, z, 0.0, w);
      }
      std::map<int, int> mkeys;
      for (const auto& kv : want_mult[c.name]) { mkeys[kv.first] = 1; }
      for (const auto& kv : got_mult) { mkeys[kv.first] = 1; }
      for (const auto& kv : mkeys) {
        const long long n2 =
            want_mult[c.name].count(kv.first) ? want_mult[c.name][kv.first] : 0;
        const long long n1 = got_mult.count(kv.first) ? got_mult[kv.first] : 0;
        const double denom = std::sqrt(static_cast<double>(n1 + n2));
        const double z = (denom > 0.0) ? std::fabs(double(n1 - n2)) / denom : 0.0;
        char w[128];
        std::snprintf(w, sizeof w, "%s mult=%d got %lld want %lld", c.name.c_str(), kv.first,
                      n1, n2);
        cmp(b_statmult, z, 0.0, w);
      }
    }
    delete ws;
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
