// Bertini's own de-excitation against ref/oracle/bertini_deexcite.csv, exact.
//
// Five entry points, each dumped separately as well as through the chain, because each is
// constructible on its own in Geant4 and has a public `deExcite(const G4Fragment&,
// G4CollisionOutput&)`:
//
//   stage 0  G4CascadeDeexcitation      - the whole chain
//   stage 1  G4BigBanger                - disperse the fragment into free nucleons
//   stage 2  G4NonEquilibriumEvaporator - the exciton walk
//   stage 3  G4EquilibriumEvaporator    - Dostrovsky evaporation, the photon chain, fission
//   stage 4  G4Fissioner                - one binary split
//
// **1,920 cases**: 12 fragments x 4 exciton configurations x 5 stages x 8 phases. Everything is
// bounded - the Box-Muller Gaussian, the Dostrovsky rejection, the exciton Newton solve, the big
// bang's energy-fraction rejection, the fissioner's 4-D minimisation - so the eight-value cycle
// drives all of it and every number is compared value for value.
//
// **The exciton configuration is an input, not a by-product.** The non-equilibrium stage's only
// entry condition is `QP + QH > 0`; with the empty configuration it emits nothing and hands the
// fragment straight on. P3's `deex::Fragment` has no room for the counts - G4Fragment carries
// them - so they travel beside it in `CollisionOutput::recoil_excitons`, and a port that dropped
// them would pass every four-momentum comparison while silently disabling a whole stage. Four
// configurations are run, including the empty one.
//
// **Why the draw count is the assertion that earns its keep, again.** The exciton walk solves
// `X^QEX = R` by Newton iteration inside a rejection loop inside a retry loop, and the number of
// deviates one call costs is a strong function of (A, Z, E*, the exciton counts). The evaporator
// above it draws two per Dostrovsky trial and gives up after a thousand. Nothing else in the
// output would notice a retry loop that ran a different number of times.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include "core/rng.cuh"
#include "physics/hadronic/bertini/deexcite.cuh"

using namespace g4gpu;
using namespace g4gpu::physics::hadronic;
using bert::BertiniWorkspace;
using bert::CollisionOutput;
using bert::DeexciteRefusal;
using bert::ExitonConfiguration;
using bert::LV;
using bert::Vec3d;

/// The deliverable is device code. Never launched; instantiating it is what proves the whole
/// de-excitation chain compiles for the device and what `-Xptxas -v` measures. The workspace and
/// both collision-output buffers are reached through pointers: between them they hold the big
/// bang's per-nucleon arrays, the fission work list and two output lists, which is far more than
/// a thread's stack.
__global__ void bertini_deexcite_probe(int a, int z, double eexs, BertiniWorkspace* ws,
                                       CollisionOutput* out, CollisionOutput* tmp, int* n_out) {
  Philox<double> rng(21u, 22u, 23u);
  DeexciteRefusal r = DeexciteRefusal::kNone;
  ExitonConfiguration ex;
  ex.proton_quasi_particles = 2;
  ex.neutron_quasi_particles = 3;
  ex.proton_holes = 1;
  ex.neutron_holes = 2;
  bert::cascade_deexcite(bert::deex_make_fragment(a, z, eexs), ex, *out, *tmp, *ws, rng, r);
  n_out[0] = out->n_particles;
  n_out[1] = out->n_nuclei;
  n_out[2] = static_cast<int>(r);
}

namespace {

int fails = 0;

__host__ __device__ inline const double* cycle_seq() {
  static const double v[8] = {0.05, 0.37, 0.63, 0.91, 0.12, 0.78, 0.29, 0.55};
  return v;
}

/// dump_bertini.cc's CycleEngine.
struct CycleRng {
  int phase = 0;
  long long n = 0;
  __host__ __device__ void reset(int p) { phase = p; n = 0; }
  __host__ __device__ double uniform() {
    ++n;
    const unsigned i = static_cast<unsigned>(n - 1) + static_cast<unsigned>(phase);
    return cycle_seq()[i % 8u];
  }
};

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

/// A NaN has to be tested for, not compared - `rel > worst` is false for one. See the same note
/// in tests/test_bertini_cascade.cu and docs/RISK.md V128.
void cmp_at(int bi, double got, double want, double scale, const std::string& where) {
  Bucket& b = buckets[bi];
  ++b.n;
  if (std::isnan(got) != std::isnan(want)) {
    b.worst = std::numeric_limits<double>::infinity();
    if (b.where.empty()) { b.where = where + " (NaN)"; }
    return;
  }
  const double rel = std::fabs(got - want) / ((scale > 0.0) ? scale : 1.0);
  if (rel > b.worst) {
    b.worst = rel;
    b.where = where;
  }
}

void cmp(int bi, double got, double want, const std::string& where) {
  cmp_at(bi, got, want, (std::fabs(want) > 0.0) ? std::fabs(want) : 1.0, where);
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
  return (e != nullptr) ? std::string(e) : std::string("ref/oracle");
}

std::vector<std::string> read_lines(const std::string& name) {
  std::vector<std::string> out;
  const std::string path = oracle_dir() + "/" + name;
  FILE* f = std::fopen(path.c_str(), "r");
  if (f == nullptr) {
    std::printf("FAIL: cannot open %s\n", path.c_str());
    ++fails;
    return out;
  }
  char line[65536];
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

/// One case: every row of the file that shares (stage, a, z, eexs, the four exciton counts,
/// phase).
struct Case {
  int stage = 0, a = 0, z = 0, phase = 0;
  double eexs = 0.0;
  int qpp = 0, qnp = 0, qph = 0, qnh = 0;
  long long draws = 0;
  int npart = 0, nnuc = 0, nfrag = 0;
  std::vector<int> ptype;
  std::vector<LV> pmom;
  std::vector<int> na, nz;
  std::vector<double> nexc;
  std::vector<LV> nmom;
  std::vector<int> fa, fz;
  std::vector<double> fexc;
  std::vector<LV> fmom;
};

// Coverage counters, printed with the result: a bucket of comparisons says nothing about which
// branches produced them.
long long n_banged = 0, n_fissioned = 0, n_photons = 0, n_light_ions = 0, n_untouched = 0;
long long n_bang_failed = 0;

void check_deexcite() {
  const int bd = new_bucket("DeexDrawCount", 0.0);
  const int bn = new_bucket("DeexMultiplicity", 0.0);
  const int bk = new_bucket("DeexKinds", 0.0);
  const int bp = new_bucket("DeexMomenta", 1e-12);
  const int bx = new_bucket("DeexExcitation", 1e-11);

  std::vector<Case> cases;
  std::map<std::string, std::size_t> index;
  const std::vector<std::string> lines = read_lines("bertini_deexcite.csv");
  for (std::size_t li = 1; li < lines.size(); ++li) {
    const std::vector<std::string> f = split(lines[li]);
    if (f.size() < 23) { continue; }
    char key[192];
    std::snprintf(key, sizeof key, "%d_%d_%d_%s_%d_%d_%d_%d_%d", iv(f, 0), iv(f, 1), iv(f, 2),
                  f[3].c_str(), iv(f, 4), iv(f, 5), iv(f, 6), iv(f, 7), iv(f, 8));
    auto it = index.find(key);
    if (it == index.end()) {
      Case c;
      c.stage = iv(f, 0);
      c.a = iv(f, 1);
      c.z = iv(f, 2);
      c.eexs = dv(f, 3);
      c.qpp = iv(f, 4);
      c.qnp = iv(f, 5);
      c.qph = iv(f, 6);
      c.qnh = iv(f, 7);
      c.phase = iv(f, 8);
      c.draws = iv(f, 9);
      c.npart = iv(f, 10);
      c.nnuc = iv(f, 11);
      c.nfrag = iv(f, 12);
      index[key] = cases.size();
      cases.push_back(c);
      it = index.find(key);
    }
    Case& c = cases[it->second];
    const int kind = iv(f, 13);
    const LV mom(Vec3d{dv(f, 19), dv(f, 20), dv(f, 21)}, dv(f, 22));
    if (kind == 0) {
      c.ptype.push_back(iv(f, 15));
      c.pmom.push_back(mom);
    } else if (kind == 1) {
      c.na.push_back(iv(f, 16));
      c.nz.push_back(iv(f, 17));
      c.nexc.push_back(dv(f, 18));
      c.nmom.push_back(mom);
    } else if (kind == 2) {
      c.fa.push_back(iv(f, 16));
      c.fz.push_back(iv(f, 17));
      c.fexc.push_back(dv(f, 18));
      c.fmom.push_back(mom);
    }
  }

  BertiniWorkspace* ws = new BertiniWorkspace();
  CollisionOutput* out = new CollisionOutput();
  CollisionOutput* tmp = new CollisionOutput();

  for (const Case& c : cases) {
    bert::ws_reset(*ws);
    bert::co_reset(*out);
    CycleRng rng;
    rng.reset(c.phase);

    ExitonConfiguration ex;
    ex.proton_quasi_particles = c.qpp;
    ex.neutron_quasi_particles = c.qnp;
    ex.proton_holes = c.qph;
    ex.neutron_holes = c.qnh;

    const deex::Fragment frag = bert::deex_make_fragment(c.a, c.z, c.eexs);
    DeexciteRefusal r = DeexciteRefusal::kNone;
    deex::Fragment ff[2];
    int nff = 0;

    switch (c.stage) {
      case 0: bert::cascade_deexcite(frag, ex, *out, *tmp, *ws, rng, r); break;
      case 1: bert::bang_deexcite(frag, *out, *ws, rng, r); break;
      case 2: bert::noneq_deexcite(frag, ex, *out, rng, r); break;
      case 3: bert::eq_deexcite(frag, *out, *ws, rng, r); break;
      default: nff = bert::fission_deexcite(frag, ff, *ws, rng, r); break;
    }

    char w[192];
    std::snprintf(w, sizeof w, "s%d A%d Z%d E%g x%d%d%d%d ph%d", c.stage, c.a, c.z, c.eexs,
                  c.qpp, c.qnp, c.qph, c.qnh, c.phase);
    const std::string where(w);

    // `kBigBangFailed` is Geant4 giving up, not this port declining: `generateBangInSCM`
    // exhausts its thousand attempts, `deExcite` prints "No bang! Don't know why..." and
    // returns with an EMPTY output. Under the cycle engine that is not rare - the rejection
    // sampler sees only four distinct (x, u) pairs - and those cases are compared against the
    // empty output the oracle recorded for them, not skipped. See
    // `deexcite_refusal_is_port_limit`.
    if (bert::deexcite_refusal_is_port_limit(r)) {
      std::printf("FAIL: de-excitation refused (%d) at %s\n", static_cast<int>(r),
                  where.c_str());
      ++fails;
      continue;
    }
    if (r == DeexciteRefusal::kBigBangFailed) { ++n_bang_failed; }

    // Coverage.
    if (c.stage == 0) {
      if (bert::deex_explosion_base(c.a, c.z, c.eexs)) { ++n_banged; }
      if (out->n_particles == 0 && out->has_recoil_fragment) { ++n_untouched; }
      for (int i = 0; i < out->n_particles; ++i) {
        if (out->particles[i].type == bert::kPhoton) { ++n_photons; }
      }
      if (out->n_nuclei > 0) { ++n_light_ions; }
    }
    if (c.stage == 4 && nff == 2) { ++n_fissioned; }

    cmp_int(bd, rng.n, c.draws, where + " draws");

    // The fissioner writes its two fragments to the caller's array, not to a CollisionOutput -
    // Geant4 keeps them in a separate `fission_output` buffer for the same reason - so the
    // comparison for stage 4 is against the returned pair.
    const int got_nfrag = (c.stage == 4) ? nff : (out->has_recoil_fragment ? 1 : 0);
    cmp_int(bn, (c.stage == 4) ? 0 : out->n_particles, c.npart, where + " npart");
    cmp_int(bn, (c.stage == 4) ? 0 : out->n_nuclei, c.nnuc, where + " nnuc");
    cmp_int(bn, got_nfrag, c.nfrag, where + " nfrag");

    if (c.stage != 4) {
      const int np = (out->n_particles < c.npart) ? out->n_particles : c.npart;
      for (int i = 0; i < np; ++i) {
        const std::string wi = where + " p" + std::to_string(i);
        cmp_int(bk, out->particles[i].type, c.ptype[i], wi + " type");
        const double sc = out->particles[i].momentum.e;
        cmp_at(bp, out->particles[i].momentum.v.x, c.pmom[i].v.x, sc, wi + " px");
        cmp_at(bp, out->particles[i].momentum.v.y, c.pmom[i].v.y, sc, wi + " py");
        cmp_at(bp, out->particles[i].momentum.v.z, c.pmom[i].v.z, sc, wi + " pz");
        cmp_at(bp, out->particles[i].momentum.e, c.pmom[i].e, sc, wi + " e");
      }
      const int nn = (out->n_nuclei < c.nnuc) ? out->n_nuclei : c.nnuc;
      for (int i = 0; i < nn; ++i) {
        const std::string wi = where + " n" + std::to_string(i);
        cmp_int(bk, out->nuclei[i].a, c.na[i], wi + " A");
        cmp_int(bk, out->nuclei[i].z, c.nz[i], wi + " Z");
        cmp(bx, out->nuclei[i].exc_MeV, c.nexc[i], wi + " exc");
        const double sc = out->nuclei[i].momentum.e;
        cmp_at(bp, out->nuclei[i].momentum.v.x, c.nmom[i].v.x, sc, wi + " px");
        cmp_at(bp, out->nuclei[i].momentum.v.y, c.nmom[i].v.y, sc, wi + " py");
        cmp_at(bp, out->nuclei[i].momentum.v.z, c.nmom[i].v.z, sc, wi + " pz");
        cmp_at(bp, out->nuclei[i].momentum.e, c.nmom[i].e, sc, wi + " e");
      }
      if (out->has_recoil_fragment && c.nfrag > 0) {
        const deex::Fragment& g = out->recoil_fragment;
        cmp_int(bk, g.a, c.fa[0], where + " frag A");
        cmp_int(bk, g.z, c.fz[0], where + " frag Z");
        cmp(bx, g.excitation, c.fexc[0], where + " frag exc");
        const double sc = g.momentum.e * 0.001;
        cmp_at(bp, g.momentum.v.x * 0.001, c.fmom[0].v.x, sc, where + " frag px");
        cmp_at(bp, g.momentum.v.y * 0.001, c.fmom[0].v.y, sc, where + " frag py");
        cmp_at(bp, g.momentum.v.z * 0.001, c.fmom[0].v.z, sc, where + " frag pz");
        cmp_at(bp, g.momentum.e * 0.001, c.fmom[0].e, sc, where + " frag e");
      }
    } else {
      const int nf = (nff < c.nfrag) ? nff : c.nfrag;
      for (int i = 0; i < nf; ++i) {
        const std::string wi = where + " f" + std::to_string(i);
        cmp_int(bk, ff[i].a, c.fa[i], wi + " A");
        cmp_int(bk, ff[i].z, c.fz[i], wi + " Z");
        cmp(bx, ff[i].excitation, c.fexc[i], wi + " exc");
        const double sc = ff[i].momentum.e * 0.001;
        cmp_at(bp, ff[i].momentum.v.x * 0.001, c.fmom[i].v.x, sc, wi + " px");
        cmp_at(bp, ff[i].momentum.v.y * 0.001, c.fmom[i].v.y, sc, wi + " py");
        cmp_at(bp, ff[i].momentum.v.z * 0.001, c.fmom[i].v.z, sc, wi + " pz");
        cmp_at(bp, ff[i].momentum.e * 0.001, c.fmom[i].e, sc, wi + " e");
      }
    }
  }

  delete tmp;
  delete out;
  delete ws;
}

/// Branches no grid of fragments can reach, pinned by construction.
void check_pinned() {
  const int b = new_bucket("PinnedByConstruction", 0.0);

  // **The two `explosion` rules disagree in both directions**, and which one a fragment meets
  // depends only on where in the chain it is - the base class's for G4CascadeDeexcitation, the
  // evaporator's own for G4EquilibriumEvaporator. A grid can show that each fires; only a
  // direct comparison shows that they are different functions.
  //
  // A = 30, Z = 0 is a neutron ball above the base class's A cut: `A <= 20 || Z == 0` lets it
  // through, `!(A >= 12 && Z < 3*(A-Z))` does not (30 >= 12 and 0 < 90).
  cmp_int(b, bert::deex_explosion_base(30, 0, 1000.0) ? 1 : 0, 1, "base: a neutron ball bangs");
  cmp_int(b, bert::deex_explosion_equilibrium(30, 0, 1000.0) ? 1 : 0, 0,
          "equilibrium: a neutron ball does not");
  // A = 8, Z = 6 is the other direction: below the equilibrium rule's A = 12 and so always a
  // candidate there, and below the base rule's A = 20 as well - so this pair agrees. The
  // disagreeing case in that direction needs A between 12 and 20 with Z above a third of N:
  // A = 16, Z = 13 has N = 3 and 13 >= 9, so the equilibrium rule fires and the base one does
  // too (A <= 20). At A = 30, Z = 25 the base rule does NOT and the equilibrium one does.
  cmp_int(b, bert::deex_explosion_base(30, 25, 1.0e5) ? 1 : 0, 0,
          "base: a proton-rich A=30 does not bang");
  cmp_int(b, bert::deex_explosion_equilibrium(30, 25, 1.0e5) ? 1 : 0, 1,
          "equilibrium: a proton-rich A=30 does");

  // **`xProbability`'s guard is an OR that nothing can fail.** `if (x < 1.0 || x > 0.0)` is true
  // for every real x, including the ones outside [0,1] that would make the expression
  // meaningless. The caller only ever passes a uniform deviate, so no grid can distinguish it
  // from the `&&` that was surely meant; asked directly, at an argument no caller produces.
  cmp_int(b, (bert::bang_x_probability(1.5, 6) != 0.0) ? 1 : 0, 1,
          "xProbability evaluates outside [0,1] rather than returning zero");
  cmp_int(b, (bert::bang_x_probability(-0.5, 6) != 0.0) ? 1 : 0, 1,
          "xProbability evaluates below zero too");
  // The parity split: even A takes an extra sqrt(1-x) and a lower power.
  cmp(b, bert::bang_x_probability(0.25, 6),
      0.25 * 0.25 * std::sqrt(0.75) * std::pow(0.75, 6.0), "even-A probability");
  cmp(b, bert::bang_x_probability(0.25, 7),
      0.25 * 0.25 * std::pow(0.75, 8.0), "odd-A probability");
}

}  // namespace

int main() {
  check_deexcite();
  check_pinned();

  std::printf("  de-excitation: %lld chain cases exploded, %lld fissions, %lld photons,"
              " %lld with a coalesced light ion, %lld left untouched\n",
              n_banged, n_fissioned, n_photons, n_light_ions, n_untouched);

  long long total = 0;
  std::printf("%-40s %10s %14s\n", "bucket", "points", "worst rel");
  for (const Bucket& b : buckets) {
    total += b.n;
    const bool ok = !(b.worst > b.tol);
    if (!ok) { ++fails; }
    std::printf("%-40s %10lld %14.3g %-4s %s\n", b.name, b.n, b.worst, ok ? "ok" : "FAIL",
                b.where.c_str());
  }
  std::printf("%lld comparisons, %d failures\n", total, fails);
  return (fails == 0) ? 0 : 1;
}
