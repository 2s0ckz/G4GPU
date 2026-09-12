// G4ExcitedStringDecay::FragmentStrings and everything it adds to the Lund fragmentation, against
// ref/oracle/ftf_resonance.csv, ftf_corrector.csv, ftf_strings.csv and ftf_stringstat*.csv.
//
// The layer this test covers is where a string's hadrons stop being the string's hadrons: every
// short-lived product's mass is redrawn from a Breit-Wigner, which breaks the energy balance the
// fragmentation had, and an iterative corrector then rescales every momentum in the strings'
// c.m.s. until the sum comes back. Both halves are checked separately from the whole -
//
//   ftf_resonance.csv   G4SampleResonance::SampleMass at 8 phases, over its three arms and then
//                       over every short-lived particle in the table with the arguments
//                       FragmentStrings passes. The `min` column is compared against
//                       data/ftf_hadrons.hh's `minmass`, which is what makes that column
//                       load-bearing: a wrong minimum mass shifts every resonance's mass.
//   ftf_corrector.csv   EnergyAndMomentumCorrector alone, on constructed hadron lists, including
//                       all four of its early returns. It draws no random number, so this is
//                       exact with no phase axis.
//   ftf_strings.csv     the whole of FragmentStrings on constructed string vectors under the
//                       eight-value cycle, with the draw count.
//   ftf_stringstat*.csv 20,000 events a case: species, multiplicity, and the distribution of
//                       log10 of the relative energy error - the corrector's own output, in one
//                       histogram.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "core/rng.cuh"
#include "physics/hadronic/ftf/string_fragmentation.cuh"

using namespace g4gpu;
using namespace g4gpu::hadronic;

/// The same eight uniforms as ref/dump/dump_ftf.cc's CycleEngine and tests/test_ftf_lund.cu's
/// CycleRng, written out for the reason that test gives: a second copy that drifts turns every
/// exact comparison into noise, and the two lists are short enough to read.
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

/// Never launched. The point is the same as tests/test_ftf_lund.cu's probe, one layer up: this
/// one instantiates the string-decay workspace, which is the FragmentWorkspace PLUS the output
/// list and the saved masses, and `-Xptxas -v` then says what the whole chain costs.
__global__ void ftf_strings_device_probe(ftf::LundTables<double>* lund,
                                         ftf::StringsWorkspace<double>* ws,
                                         ftf::ExcitedString* strings, int n_strings, int* out) {
  CycleRng rng;
  rng.reset(0);
  ftf::ftf_fragment_strings(lund, ws, strings, n_strings, rng);
  out[0] = ws->n_out;
  out[1] = ws->success ? 1 : 0;
  out[2] = static_cast<int>(ws->refused);
  out[3] = static_cast<int>(
      ftf::ftf_sample_resonance_mass<double>(770.0, 150.0, 290.0, 1520.0, rng));
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

/// One case's strings, rebuilt from ftf_stringspec.csv. FragmentStrings MUTATES them, so every
/// call needs a fresh copy - the same reason the dump rebuilds its G4ExcitedStringVector.
struct StringSpecRow {
  int index, left, right, direction, track_pdg;
  double lpx, lpy, lpz, le, rpx, rpy, rpz, re;
};

std::vector<ftf::ExcitedString> build_case(const std::vector<StringSpecRow>& rows) {
  std::vector<ftf::ExcitedString> out;
  for (const StringSpecRow& r : rows) {
    ftf::ExcitedString s;
    if (r.track_pdg != 0) {
      s.excited = false;
      s.track_pdg = r.track_pdg;
      s.track_mom = ftf::Vec4(r.lpx, r.lpy, r.lpz, r.le);
      s.track_time = 0.0;
      s.track_position = ftf::Vec3d{0.0, 0.0, 0.0};
    } else {
      s.excited = true;
      s.left = r.left;
      s.right = r.right;
      s.pleft = ftf::Vec4(r.lpx, r.lpy, r.lpz, r.le);
      s.pright = ftf::Vec4(r.rpx, r.rpy, r.rpz, r.re);
      s.direction = r.direction;
    }
    out.push_back(s);
  }
  return out;
}

}  // namespace

int main() {
  const std::string dir = oracle_dir();
  ftf::LundTables<double>* lund = new ftf::LundTables<double>();
  ftf::lund_init(lund, true);

  // -------------------------------------------------------------------------------------------
  // ftf_resonance.csv - G4SampleResonance::SampleMass
  // -------------------------------------------------------------------------------------------
  const int b_res = new_bucket("ResonanceMass", 1e-13);
  const int b_resdraw = new_bucket("ResonanceDrawCount", 0.0);
  const int b_resmin = new_bucket("ResonanceMinimumMass", 0.0);
  {
    const std::vector<std::string> lines = read_lines(dir + "/ftf_resonance.csv");
    if (lines.size() < 2) {
      std::printf("FAIL: cannot read %s/ftf_resonance.csv\n", dir.c_str());
      ++fails;
    }
    Header h;
    h.parse(lines[0]);
    for (std::size_t li = 1; li < lines.size(); ++li) {
      const std::vector<std::string> f = split(lines[li]);
      if (f.size() != h.n) { continue; }
      const int pdg = iv(f, h.at("pdg"));
      const double pole = dv(f, h.at("pole"));
      const double gamma = dv(f, h.at("gamma"));
      const double lo = dv(f, h.at("min"));
      const double hi = dv(f, h.at("max"));
      const int phase = iv(f, h.at("phase"));
      char wbuf[192];
      std::snprintf(wbuf, sizeof wbuf, "pdg=%d pole=%g gamma=%g min=%g max=%g phase=%d", pdg,
                    pole, gamma, lo, hi, phase);
      const std::string w = wbuf;

      CycleRng rng;
      rng.reset(phase);
      const double m = ftf::ftf_sample_resonance_mass<double>(pole, gamma, lo, hi, rng);
      cmp(b_res, m, dv(f, h.at("mass")), w);
      cmp_int(b_resdraw, rng.n, iv(f, h.at("draws")), w + " draws");

      // The arguments themselves, for the rows that come from a real particle: the port has to
      // reconstruct (pole, gamma, min, max) from data/ftf_hadrons.hh, and `minmass` is the one
      // of the four that no other test reads.
      if (pdg != 0 && phase == 0) {
        const data::FtfHadron* d = data::ftf_find_hadron(pdg);
        if (d == nullptr) {
          std::printf("FAIL: %s is in ftf_resonance.csv and not in the header\n", w.c_str());
          ++fails;
        } else {
          cmp(b_resmin, d->mass, pole, w + " pole");
          cmp(b_resmin, d->width, gamma, w + " width");
          cmp(b_resmin, d->minmass + 10.0, lo, w + " minmass+10MeV");
          cmp(b_resmin, d->mass + 5.0 * d->width, hi, w + " pole+5width");
          cmp_int(b_resmin, d->shortlived ? 1 : 0, 1, w + " shortlived");
        }
      }
    }
  }

  // -------------------------------------------------------------------------------------------
  // ftf_corrector.csv - EnergyAndMomentumCorrector on its own
  // -------------------------------------------------------------------------------------------
  const int b_corrok = new_bucket("CorrectorVerdict", 0.0);
  const int b_corrmom = new_bucket("CorrectorMomentum", 1e-12);
  {
    const std::vector<std::string> lines = read_lines(dir + "/ftf_corrector.csv");
    if (lines.size() < 2) {
      std::printf("FAIL: cannot read %s/ftf_corrector.csv\n", dir.c_str());
      ++fails;
    }
    Header h;
    h.parse(lines[0]);
    // Group the rows by case, in the file's order.
    std::vector<std::string> order;
    std::map<std::string, std::vector<std::vector<std::string>>> rows;
    for (std::size_t li = 1; li < lines.size(); ++li) {
      const std::vector<std::string> f = split(lines[li]);
      if (f.size() != h.n) { continue; }
      const std::string nm = f[h.at("case")];
      if (rows.count(nm) == 0) { order.push_back(nm); }
      rows[nm].push_back(f);
    }

    ftf::StringsWorkspace<double>* ws = new ftf::StringsWorkspace<double>();
    for (const std::string& nm : order) {
      const std::vector<std::vector<std::string>>& rs = rows[nm];
      ws->n_out = 0;
      for (const std::vector<std::string>& f : rs) {
        if (iv(f, h.at("index")) < 0) { continue; }  // the empty-list marker row
        const int pdg = iv(f, h.at("pdg"));
        const data::FtfHadron* d = data::ftf_find_hadron(pdg);
        const double m = (d != nullptr) ? d->mass : 0.0;
        const ftf::Vec3d p3{dv(f, h.at("in_px")), dv(f, h.at("in_py")), dv(f, h.at("in_pz"))};
        ws->out[ws->n_out].pdg = pdg;
        ws->out[ws->n_out].momentum = ftf::Vec4(p3, std::sqrt(g4gpu::mag2(p3) + m * m));
        ++ws->n_out;
      }
      const std::vector<std::string>& f0 = rs[0];
      const ftf::Vec4 cms(dv(f0, h.at("cms_px")), dv(f0, h.at("cms_py")),
                          dv(f0, h.at("cms_pz")), dv(f0, h.at("cms_e")));
      const bool ok = ftf::ftf_energy_momentum_corrector(ws, cms);
      cmp_int(b_corrok, ok ? 1 : 0, iv(f0, h.at("success")), nm + " success");
      for (const std::vector<std::string>& f : rs) {
        const int i = iv(f, h.at("index"));
        if (i < 0 || i >= ws->n_out) { continue; }
        const std::string w = nm + " hadron " + std::to_string(i);
        cmp(b_corrmom, ws->out[i].momentum.v.x, dv(f, h.at("px")), w + " px");
        cmp(b_corrmom, ws->out[i].momentum.v.y, dv(f, h.at("py")), w + " py");
        cmp(b_corrmom, ws->out[i].momentum.v.z, dv(f, h.at("pz")), w + " pz");
        cmp(b_corrmom, ws->out[i].momentum.e, dv(f, h.at("e")), w + " e");
      }
    }
    delete ws;
  }

  // -------------------------------------------------------------------------------------------
  // ftf_stringspec.csv + ftf_strings.csv - the whole of FragmentStrings
  // -------------------------------------------------------------------------------------------
  std::vector<std::string> case_order;
  std::map<std::string, std::vector<StringSpecRow>> specs;
  {
    const std::vector<std::string> lines = read_lines(dir + "/ftf_stringspec.csv");
    if (lines.size() < 2) {
      std::printf("FAIL: cannot read %s/ftf_stringspec.csv\n", dir.c_str());
      ++fails;
    }
    Header h;
    h.parse(lines[0]);
    for (std::size_t li = 1; li < lines.size(); ++li) {
      const std::vector<std::string> f = split(lines[li]);
      if (f.size() != h.n) { continue; }
      const std::string nm = f[h.at("case")];
      if (specs.count(nm) == 0) { case_order.push_back(nm); }
      StringSpecRow r;
      r.index = iv(f, h.at("index"));
      r.left = iv(f, h.at("left"));
      r.right = iv(f, h.at("right"));
      r.direction = iv(f, h.at("direction"));
      r.track_pdg = iv(f, h.at("track_pdg"));
      r.lpx = dv(f, h.at("lpx"));
      r.lpy = dv(f, h.at("lpy"));
      r.lpz = dv(f, h.at("lpz"));
      r.le = dv(f, h.at("le"));
      r.rpx = dv(f, h.at("rpx"));
      r.rpy = dv(f, h.at("rpy"));
      r.rpz = dv(f, h.at("rpz"));
      r.re = dv(f, h.at("re"));
      specs[nm].push_back(r);
    }
  }

  const int b_nhad = new_bucket("StringsMultiplicity", 0.0);
  const int b_pdg = new_bucket("StringsHadronPdg", 0.0);
  const int b_mom = new_bucket("StringsHadronMomentum", 1e-11);
  const int b_mass = new_bucket("StringsHadronMass", 1e-11);
  const int b_time = new_bucket("StringsFormationTime", 1e-13);
  const int b_draws = new_bucket("StringsDrawCount", 0.0);
  {
    const std::vector<std::string> lines = read_lines(dir + "/ftf_strings.csv");
    if (lines.size() < 2) {
      std::printf("FAIL: cannot read %s/ftf_strings.csv\n", dir.c_str());
      ++fails;
    }
    Header h;
    h.parse(lines[0]);
    ftf::StringsWorkspace<double>* ws = new ftf::StringsWorkspace<double>();
    std::string last_key;
    int n_got = 0;
    for (std::size_t li = 1; li < lines.size(); ++li) {
      const std::vector<std::string> f = split(lines[li]);
      if (f.size() != h.n) { continue; }
      const std::string nm = f[h.at("case")];
      const int phase = iv(f, h.at("phase"));
      const std::string k = nm + " phase=" + std::to_string(phase);
      if (k != last_key) {
        last_key = k;
        std::vector<ftf::ExcitedString> s = build_case(specs[nm]);
        CycleRng rng;
        rng.reset(phase);
        ftf::ftf_fragment_strings(lund, ws, s.data(), (int)s.size(), rng);
        n_got = ws->n_out;
        // Geant4 returns a NULL vector when it gave up; the oracle writes nhadrons = -1 for
        // that and 0 never happens, so the two are compared as they are written.
        const int want_n = iv(f, h.at("nhadrons"));
        cmp_int(b_nhad, ws->success ? n_got : -1, want_n, k + " nhadrons");
        cmp_int(b_draws, rng.n, iv(f, h.at("draws")), k + " draws");
      }
      const int i = iv(f, h.at("index"));
      if (i < 0 || i >= n_got) { continue; }
      const std::string w = k + " hadron " + std::to_string(i);
      cmp_int(b_pdg, ws->out[i].pdg, iv(f, h.at("pdg")), w + " pdg");
      cmp(b_mom, ws->out[i].momentum.v.x, dv(f, h.at("px")), w + " px");
      cmp(b_mom, ws->out[i].momentum.v.y, dv(f, h.at("py")), w + " py");
      cmp(b_mom, ws->out[i].momentum.v.z, dv(f, h.at("pz")), w + " pz");
      cmp(b_mom, ws->out[i].momentum.e, dv(f, h.at("e")), w + " e");
      cmp(b_mass, ws->out[i].momentum.mag(), dv(f, h.at("mass")), w + " mass");
      cmp(b_time, ws->out[i].formation_time, dv(f, h.at("formation_time")), w + " time");
    }
    delete ws;
  }

  // -------------------------------------------------------------------------------------------
  // ftf_stringstat*.csv - species, multiplicity and the energy balance over 20,000 events
  // -------------------------------------------------------------------------------------------
  //
  // The balance histogram is the one that says whether the CORRECTOR ran: Geant4 calls it when
  // any string's hadrons are more than 1e-6 off in energy and iterates to 1e-5, so a port that
  // skipped it would pile up in the -2 and -3 bins of log10|dE/E| while the oracle sits at -5
  // and below. Comparing the histogram rather than a mean is deliberate - a mean over events
  // that mostly agree hides a tail of events that do not.
  const int b_stat = new_bucket("StringStatSpeciesSigma", 5.0);
  const int b_statmult = new_bucket("StringStatMultiplicitySigma", 5.0);
  const int b_statbal = new_bucket("StringStatBalanceSigma", 5.0);
  {
    const std::vector<std::string> slines = read_lines(dir + "/ftf_stringstat.csv");
    const std::vector<std::string> mlines = read_lines(dir + "/ftf_stringstat_mult.csv");
    const std::vector<std::string> blines = read_lines(dir + "/ftf_stringstat_balance.csv");
    if (slines.size() < 2 || mlines.size() < 2 || blines.size() < 2) {
      std::printf("FAIL: cannot read %s/ftf_stringstat*.csv\n", dir.c_str());
      ++fails;
    }
    Header sh, mh, bh;
    sh.parse(slines[0]);
    mh.parse(mlines[0]);
    bh.parse(blines[0]);
    std::map<std::string, std::map<int, long long>> want_species, want_mult, want_bal;
    std::map<std::string, int> want_n;
    for (std::size_t li = 1; li < slines.size(); ++li) {
      const std::vector<std::string> f = split(slines[li]);
      if (f.size() != sh.n) { continue; }
      want_species[f[sh.at("case")]][iv(f, sh.at("pdg"))] =
          std::atoll(f[sh.at("count")].c_str());
      want_n[f[sh.at("case")]] = iv(f, sh.at("n_events"));
    }
    for (std::size_t li = 1; li < mlines.size(); ++li) {
      const std::vector<std::string> f = split(mlines[li]);
      if (f.size() != mh.n) { continue; }
      want_mult[f[mh.at("case")]][iv(f, mh.at("multiplicity"))] =
          std::atoll(f[mh.at("count")].c_str());
    }
    for (std::size_t li = 1; li < blines.size(); ++li) {
      const std::vector<std::string> f = split(blines[li]);
      if (f.size() != bh.n) { continue; }
      want_bal[f[bh.at("case")]][iv(f, bh.at("rel_e_bin"))] =
          std::atoll(f[bh.at("count")].c_str());
    }

    ftf::StringsWorkspace<double>* ws = new ftf::StringsWorkspace<double>();
    for (const std::string& nm : case_order) {
      if (want_n.count(nm) == 0) { continue; }
      const int n_events = want_n[nm];
      std::map<int, long long> got_species, got_mult, got_bal;
      for (int ev = 0; ev < n_events; ++ev) {
        Philox<double> rng(0x46545302u, static_cast<uint32_t>(ev), 0u);
        std::vector<ftf::ExcitedString> s = build_case(specs[nm]);
        ftf::Vec4 total(0.0, 0.0, 0.0, 0.0);
        for (const ftf::ExcitedString& x : s) { total = total + ftf::ftf_string_4momentum(x); }
        ftf::ftf_fragment_strings(lund, ws, s.data(), (int)s.size(), rng);
        const int n = ws->success ? ws->n_out : 0;
        got_mult[n]++;
        if (n > 0) {
          ftf::Vec4 sum(0.0, 0.0, 0.0, 0.0);
          for (int i = 0; i < n; ++i) {
            got_species[ws->out[i].pdg]++;
            sum = sum + ws->out[i].momentum;
          }
          const double rel = std::fabs((sum.e - total.e) / total.e);
          int bin = -12;
          if (rel > 0.0) { bin = (int)std::floor(std::log10(rel)); }
          if (bin < -12) { bin = -12; }
          if (bin > 1) { bin = 1; }
          got_bal[bin]++;
        }
      }
      struct Pair { std::map<int, long long>* got; std::map<int, long long>* want; int bucket;
                    const char* tag; };
      const Pair pairs[3] = {{&got_species, &want_species[nm], b_stat, "pdg"},
                             {&got_mult, &want_mult[nm], b_statmult, "mult"},
                             {&got_bal, &want_bal[nm], b_statbal, "log10dE"}};
      for (const Pair& p : pairs) {
        std::map<int, int> keys;
        for (const auto& kv : *p.want) { keys[kv.first] = 1; }
        for (const auto& kv : *p.got) { keys[kv.first] = 1; }
        for (const auto& kv : keys) {
          const long long n2 = p.want->count(kv.first) ? (*p.want)[kv.first] : 0;
          const long long n1 = p.got->count(kv.first) ? (*p.got)[kv.first] : 0;
          const double denom = std::sqrt(static_cast<double>(n1 + n2));
          const double z = (denom > 0.0) ? std::fabs(double(n1 - n2)) / denom : 0.0;
          char w[160];
          std::snprintf(w, sizeof w, "%s %s=%d got %lld want %lld", nm.c_str(), p.tag,
                        kv.first, n1, n2);
          cmp(p.bucket, z, 0.0, w);
        }
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
  std::printf("\n%lld comparisons\n", total);
  std::printf("%s\n", fails == 0 ? "PASS test_ftf_strings" : "FAIL test_ftf_strings");
  delete lund;
  return fails == 0 ? 0 : 1;
}
