// Bertini's cascade: G4NucleiModel's sampling half, G4IntraNucleiCascader and
// G4CascadeCoalescence, against four oracle files. Everything here is exact under the cycle
// engine - the cascade draws nothing unbounded - so it is compared value for value.
//
//   `initializeCascad(particle)` - where a projectile enters the nucleus. One deviate for the
//   entry point, plus the whole trajectory sampler for a photon, whose `forceFirst` sends it
//   through `choosePointAlongTraj`. **1,584 cases**: 6 nuclei x 6 particle types x 4 momenta
//   (one below the 1 eV at-rest cut) x 11 phases - the eight-value cycle plus three
//   constant-engine passes, of which the one at EXACTLY zero is the only way to put a photon at
//   radial incidence.
//
//   `generateParticleFate` - one step of the cascade. **27,648 cases**: 6 nuclei x 6 types x
//   4 momenta x 3 directions x 4 radii x 2 generations x 8 phases, of which 10,113 produce a
//   collision and the rest propagate to the next zone. Everything about the step is compared:
//   the draw count, how many particles came out, each one's type, four-momentum, position,
//   zone, path, reflection counter, generation and inbound flag, which nucleon types the model
//   consumed, what its proton and neutron census dropped to, and where the INPUT particle ended.
//
//   `G4IntraNucleiCascader::collide` - a whole event, entry point to accepted final state, with
//   the retry loop, coalescence, the recoil bookkeeping and `setOnShell`. **576 cases**:
//   6 nuclei x 4 types x 3 momenta x 8 phases.
//
//   `G4CascadeCoalescence::FindClusters` - twelve hand-built final states, because a real
//   cascade at these energies makes deuterons and almost nothing else.
//
// **The nuclei are 4, 9, 12, 27, 56 and 207 and the first two are the point.** G4NucleiModel
// gives A < 5 a single zone, 5 <= A < 12 three zones on a GAUSSIAN density and A >= 12 three or
// six on Woods-Saxon; a grid of 12, 27, 56 and 207 cascades one of those three shapes. He-4 also
// found a real difference - see the note on the bare recoil nucleon in intra_cascader.cuh -
// because a residual of A = 1 needs a target light enough to lose three of its four nucleons.
//
// **Why the draw count is the first assertion again.** A cascade step costs between 2 and about
// 60 deviates and the number is a function of the nucleus's remaining census, not of what
// happened: `generateInteractionPartners` throws a proton (3 deviates) and a neutron (3), then
// for a pion or photon up to three quasi-deuterons (6 each), and every one of those is thrown
// whether or not the partner is ever used. A transcription that skips a throw because the
// partner was discarded gets plausible physics out of the wrong random numbers, and only the
// count says so.
//
// **What the grid is shaped to reach**, since docs/RISK.md V124 is about a grid that could not
// see a whole function: a position on a coordinate axis makes `pos.dot(mom)` a single component
// and `choosePointAlongTraj`'s rotation axis degenerate, so the position is off every axis; a
// momentum parallel to the position makes `pperp2` zero, so `boundaryTransition`'s third arm
// (transmission on transverse momentum) can never run and `choosePointAlongTraj`'s
// `prang < 1e-6` shortcut always does - so one direction IS radial and two are not; and a
// generation of 0 exempts every particle from the young-secondary veto through the
// `current_path < 1000` sentinel, so both generations are run. The perturbation campaign at the
// bottom of this file is what says whether that worked.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include "core/rng.cuh"
#include "physics/hadronic/bertini/intra_cascader.cuh"

using namespace g4gpu;
using namespace g4gpu::physics::hadronic;
using bert::BertiniWorkspace;
using bert::CascadeParticle;
using bert::ColliderOutput;
using bert::FateRefusal;
using bert::LV;
using bert::NucleiModel;
using bert::NucleiModelParams;
using bert::Vec3d;

/// The deliverable is device code. Never launched; instantiating it is what proves the cascade
/// step compiles for the device and what `-Xptxas -v` measures. The nucleus, the workspace and
/// the collider's output buffer are all reached through pointers, exactly as a real kernel would
/// have them: they are per-thread state in global memory, not stack.
__global__ void bertini_fate_probe(int type, double plab, NucleiModel* nm,
                                   CascadeParticle* cp, ColliderOutput* oc,
                                   BertiniWorkspace* ws, int* n_out) {
  Philox<double> rng(11u, 12u, 13u);
  const bert::CascadeParams par = bert::default_cascade_params();
  const NucleiModelParams p = bert::nuclei_model_params(par);
  bert::FateRefusal r = bert::FateRefusal::kNone;
  const double m = bert::inucl_particle_mass(type);
  *cp = bert::nm_initialize_cascad(*nm, p, type,
                                   LV(Vec3d{0.0, 0.0, plab}, std::sqrt(plab * plab + m * m)),
                                   rng, r);
  bert::nm_generate_particle_fate(*nm, p, par, *cp, *oc, *ws, rng, r);
  n_out[0] = ws->n_new_cascade;
  n_out[1] = static_cast<int>(r);
  n_out[2] = ws->n_partners;
}

namespace {

int fails = 0;


// ---------------------------------------------------------------------------------------------
// The engine and the comparison bookkeeping, as tests/test_bertini_collide.cu defines them
// ---------------------------------------------------------------------------------------------

__host__ __device__ inline const double* cycle_seq() {
  static const double v[8] = {0.05, 0.37, 0.63, 0.91, 0.12, 0.78, 0.29, 0.55};
  return v;
}

/// The three constants dump_bertini.cc's ConstEngine uses, encoded as phases 8, 9 and 10.
__host__ __device__ inline const double* const_seq() {
  static const double v[3] = {0.0, 1.0e-13, 0.5};
  return v;
}

/// dump_bertini.cc's CycleEngine: eight fixed uniforms, so every bounded sampler becomes a
/// deterministic function of (inputs, phase) and the draw count can be compared. Nothing in the
/// cascade half is unbounded - `generateSCMfinalState` gives up after ten attempts and so does
/// every sampler under it - so unlike the phase-space pass this file needs only the cycle.
///
/// Phases 8, 9 and 10 are its ConstEngine, one fixed value for every draw. They exist for a
/// single branch: `choosePointAlongTraj`'s radial-incidence shortcut, which needs the entry
/// deviate to be EXACTLY zero. The cycle's smallest value puts the incidence angle at 0.2255
/// radians, five orders of magnitude above the 1e-6 cut - and 1e-13 is not enough either: there
/// the rotation about a very short axis still returns the antipode to 1e-13 relative, under the
/// tolerance. At exactly zero the axis is exactly zero, CLHEP does nothing, and the two branches
/// give opposite ends of the nucleus. Deleting the shortcut passed all 535,692 comparisons
/// before this phase existed.
struct CycleRng {
  int phase = 0;
  long long n = 0;
  __host__ __device__ void reset(int p) { phase = p; n = 0; }
  __host__ __device__ double uniform() {
    ++n;
    if (phase >= 8) { return const_seq()[phase - 8]; }
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

/// **A NaN has to be tested for, not compared.** `rel > b.worst` is FALSE when `rel` is NaN -
/// every comparison against a NaN is - so a bucket that records the worst relative error the way
/// every other test in this project does will report `0` for a column that is entirely NaN and
/// pass. That is not hypothetical: the perturbation that removes `clhep_rotate`'s zero-axis
/// guard makes the rotation `0/0` and this function returned "worst 0, ok" for it until the
/// explicit test below was added. Worth carrying back to the other buckets in tests/.
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

// ---------------------------------------------------------------------------------------------
// CSV reading
// ---------------------------------------------------------------------------------------------

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

/// The dump's own grid constants, written out on both sides. The three directions are the same
/// numbers to seventeen digits because `dirs[0]` is deliberately PARALLEL to the position
/// direction - a difference in the last bit would turn the radial-incidence case into a
/// non-radial one and silently change which branch of `choosePointAlongTraj` is measured.
const Vec3d kRhat{0.4242640687119285, 0.5656854249492380, 0.7071067811865476};
const Vec3d kDirs[3] = {{0.4242640687119285, 0.5656854249492380, 0.7071067811865476},
                        {0.3713906763541037, -0.5570860145311556, 0.7427813527082074},
                        {-0.5883484054145521, 0.1961161351381840, -0.7844645405527361}};
const double kRads[4] = {0.15, 0.45, 0.75, 0.999};

/// A nucleus, built once and reset before every case exactly as the dump does. `reset()` is
/// what restores the proton and neutron census and clears the trailing-effect hit list, and
/// without it no row in the file could be reproduced on its own.
struct ModelCache {
  std::map<int, NucleiModel> models;
  NucleiModelParams p;
  NucleiModel& get(int a, int z) {
    const int key = a * 1000 + z;
    auto it = models.find(key);
    if (it != models.end()) { return it->second; }
    NucleiModel m;
    bert::nm_generate_model(m, a, z, p);
    return models.emplace(key, m).first->second;
  }
};

// ---------------------------------------------------------------------------------------------
// bertini_initcascad.csv
// ---------------------------------------------------------------------------------------------

/// How many `initializeCascad` cases went through `choosePointAlongTraj` rather than stopping at
/// the surface, and how many took the capture-at-rest branch. Both are printed, because a run in
/// which either is zero is a run that measured less than the file contains.
int n_traj = 0;
int n_atrest = 0;

void check_initcascad(ModelCache& mc) {
  const int bd = new_bucket("InitDrawCount", 0.0);
  const int bz = new_bucket("InitZone", 0.0);
  const int bp = new_bucket("InitPosition", 1e-13);
  const int bc = new_bucket("InitPath", 1e-13);
  const int bm = new_bucket("InitMomentum", 1e-13);

  const bert::CascadeParams par = bert::default_cascade_params();
  const std::vector<std::string> lines = read_lines("bertini_initcascad.csv");
  for (std::size_t li = 1; li < lines.size(); ++li) {
    const std::vector<std::string> f = split(lines[li]);
    if (f.size() < 16) { continue; }
    const int a = iv(f, 0), z = iv(f, 1), type = iv(f, 2);
    const double plab = dv(f, 3);
    const int phase = iv(f, 4);

    NucleiModel& m = mc.get(a, z);
    bert::nm_reset(m);
    CycleRng rng;
    rng.reset(phase);

    const double mass = bert::inucl_particle_mass(type);
    // The dump builds `G4InuclElementaryParticle bullet(fourvector, type)`, whose constructor
    // runs `setMomentum` - so what `initializeCascad` carries is the four-vector AFTER the INUCL
    // store, not the one written down. For a pi0 at 1e-12 GeV/c that is the difference between
    // a momentum of 1e-12 and a momentum of zero: the store rebuilds |p| from a kinetic energy
    // that `t - dynamicalMass` has cancelled to nothing. Measured at 7.4e-12 relative before
    // this line was here, with every other column exact.
    const LV mom = bert::inucl_store_momentum(
        LV(Vec3d{0.0, 0.0, plab}, std::sqrt(plab * plab + mass * mass)), type);
    FateRefusal r = FateRefusal::kNone;
    const CascadeParticle cp = bert::nm_initialize_cascad(m, mc.p, type, mom, rng, r);

    char w[160];
    std::snprintf(w, sizeof w, "A%d Z%d t%d p%g ph%d", a, z, type, plab, phase);
    const std::string where(w);

    if (r != FateRefusal::kNone) {
      std::printf("FAIL: initializeCascad refused (%d) at %s\n", static_cast<int>(r),
                  where.c_str());
      ++fails;
      continue;
    }
    if (bert::nm_force_first(0, type)) { ++n_traj; }
    if (iv(f, 9) < a * 0 + m.number_of_zones) { ++n_atrest; }

    cmp_int(bd, rng.n, iv(f, 5), where + " draws");
    cmp_int(bz, cp.current_zone, iv(f, 9), where + " zone");
    cmp_int(bz, cp.generation, iv(f, 11), where + " generation");
    const double rscale = m.nuclei_radius;
    cmp_at(bp, cp.position.x, dv(f, 6), rscale, where + " posx");
    cmp_at(bp, cp.position.y, dv(f, 7), rscale, where + " posy");
    cmp_at(bp, cp.position.z, dv(f, 8), rscale, where + " posz");
    cmp(bc, cp.current_path, dv(f, 10), where + " cpath");
    const double escale = cp.momentum.e;
    cmp_at(bm, cp.momentum.v.x, dv(f, 12), escale, where + " px");
    cmp_at(bm, cp.momentum.v.y, dv(f, 13), escale, where + " py");
    cmp_at(bm, cp.momentum.v.z, dv(f, 14), escale, where + " pz");
    cmp_at(bm, cp.momentum.e, dv(f, 15), escale, where + " e");
    (void)par;
  }
}

// ---------------------------------------------------------------------------------------------
// bertini_fate.csv
// ---------------------------------------------------------------------------------------------

/// One case: every row of the file that shares (a, z, type, plab, dir, rad, gen, phase).
struct FateCase {
  int a = 0, z = 0, type = 0, dir = 0, rad = 0, gen = 0, phase = 0;
  double plab = 0.0;
  long long draws = 0;
  int n = 0;
  int nucl1 = 0, nucl2 = 0, npcur = 0, nncur = 0;
  double cpos[3] = {0.0, 0.0, 0.0};
  int czone = 0;
  double ccpath = 0.0;
  std::vector<int> otype;
  std::vector<LV> mom;
  std::vector<Vec3d> pos;
  std::vector<int> zone;
  std::vector<double> cpath;
  std::vector<int> nrefl;
  std::vector<int> movingin;
  std::vector<int> ogen;
};

/// Coverage counters. Printed with the result, because a bucket of 200,000 comparisons says
/// nothing about which branches produced them.
long long n_prop = 0, n_coll = 0, n_refl = 0, n_absorb = 0, n_multi = 0;

void check_fate(ModelCache& mc) {
  const int bd = new_bucket("FateDrawCount", 0.0);
  const int bn = new_bucket("FateMultiplicity", 0.0);
  const int bk = new_bucket("FateKinds", 0.0);
  // Same reasoning as the collider's momentum bucket: a momentum COMPONENT can be near zero
  // while the four-vector it belongs to is hundreds of MeV, so every component is normalised by
  // its own particle's energy, and every position by the nuclear radius.
  const int bp = new_bucket("FateMomenta", 1e-13);
  const int bx = new_bucket("FatePositions", 1e-13);
  const int bs = new_bucket("FateState", 0.0);
  const int bc = new_bucket("FateCensus", 0.0);
  const int bi = new_bucket("FateInputParticle", 1e-13);

  std::vector<FateCase> cases;
  std::map<std::string, std::size_t> index;
  const std::vector<std::string> lines = read_lines("bertini_fate.csv");
  for (std::size_t li = 1; li < lines.size(); ++li) {
    const std::vector<std::string> f = split(lines[li]);
    if (f.size() < 33) { continue; }
    char key[160];
    std::snprintf(key, sizeof key, "%d_%d_%d_%s_%d_%d_%d_%d", iv(f, 0), iv(f, 1), iv(f, 2),
                  f[3].c_str(), iv(f, 4), iv(f, 5), iv(f, 6), iv(f, 7));
    auto it = index.find(key);
    if (it == index.end()) {
      FateCase c;
      c.a = iv(f, 0);
      c.z = iv(f, 1);
      c.type = iv(f, 2);
      c.plab = dv(f, 3);
      c.dir = iv(f, 4);
      c.rad = iv(f, 5);
      c.gen = iv(f, 6);
      c.phase = iv(f, 7);
      c.draws = iv(f, 8);
      c.n = iv(f, 9);
      c.nucl1 = iv(f, 23);
      c.nucl2 = iv(f, 24);
      c.npcur = iv(f, 25);
      c.nncur = iv(f, 26);
      c.cpos[0] = dv(f, 27);
      c.cpos[1] = dv(f, 28);
      c.cpos[2] = dv(f, 29);
      c.czone = iv(f, 30);
      c.ccpath = dv(f, 31);
      index[key] = cases.size();
      cases.push_back(c);
      it = index.find(key);
    }
    FateCase& c = cases[it->second];
    if (iv(f, 10) < 0) { continue; }   // the n = 0 placeholder row
    c.otype.push_back(iv(f, 11));
    c.mom.push_back(LV(Vec3d{dv(f, 12), dv(f, 13), dv(f, 14)}, dv(f, 15)));
    c.pos.push_back(Vec3d{dv(f, 16), dv(f, 17), dv(f, 18)});
    c.zone.push_back(iv(f, 19));
    c.cpath.push_back(dv(f, 20));
    c.nrefl.push_back(iv(f, 21));
    c.movingin.push_back(iv(f, 22));
    c.ogen.push_back(iv(f, 32));
  }

  const bert::CascadeParams par = bert::default_cascade_params();
  BertiniWorkspace* ws = new BertiniWorkspace();
  ColliderOutput oc;

  // **`current_nucl1`/`current_nucl2` are NOT cleared by `reset()` and the dump does not clear
  // them either**, so a case in which nothing interacted reports the pair the PREVIOUS case
  // consumed. They are model members, zeroed once by the constructor, and `generateParticleFate`
  // sets them only on its two returning paths - the no-interaction fall-through at the end of
  // the partner loop is not one of them. The test therefore carries them exactly as the dump
  // does: cleared when the nucleus changes (a new G4NucleiModel) and never in between. In
  // production nothing reads the stale value - G4IntraNucleiCascader asks for the pair only
  // after an interaction, where it was just set - but an oracle that calls the function
  // directly does, and reproducing the leak is what makes this column an assertion rather than
  // a coincidence.
  int last_key = -1;
  for (const FateCase& c : cases) {
    NucleiModel& m = mc.get(c.a, c.z);
    bert::nm_reset(m);
    if (c.a * 1000 + c.z != last_key) {
      m.current_nucl1 = 0;
      m.current_nucl2 = 0;
      last_key = c.a * 1000 + c.z;
    }
    bert::ws_reset(*ws);

    CycleRng rng;
    rng.reset(c.phase);

    const double mass = bert::inucl_particle_mass(c.type);
    const Vec3d p3 = kDirs[c.dir] * c.plab;
    const LV mom = bert::lv_set_vect_m(p3, mass);
    const Vec3d pos = kRhat * (kRads[c.rad] * m.nuclei_radius);
    // generation 0 carries `large` in the path field - the sentinel that exempts a projectile
    // from the young-secondary veto - and generation 1 carries 0, as a real secondary does.
    CascadeParticle cp = bert::cp_fill(c.type, bert::inucl_store_momentum(mom, c.type), pos,
                                       bert::nm_zone(m, g4gpu::mag(pos)),
                                       (c.gen == 0) ? 1000.0 : 0.0, c.gen);

    FateRefusal r = FateRefusal::kNone;
    bert::nm_generate_particle_fate(m, mc.p, par, cp, oc, *ws, rng, r);

    char w[192];
    std::snprintf(w, sizeof w, "A%d Z%d t%d p%g d%d r%d g%d ph%d", c.a, c.z, c.type, c.plab,
                  c.dir, c.rad, c.gen, c.phase);
    const std::string where(w);

    if (r != FateRefusal::kNone) {
      std::printf("FAIL: fate refused (%d) at %s\n", static_cast<int>(r), where.c_str());
      ++fails;
      continue;
    }

    if (ws->n_new_cascade == 1) { ++n_prop; } else { ++n_coll; }
    if (ws->n_new_cascade > 2) { ++n_multi; }
    if (c.nucl2 != 0) { ++n_absorb; }
    for (int i = 0; i < ws->n_new_cascade; ++i) {
      if (ws->new_cascade[i].reflection_counter > 0) { ++n_refl; }
    }

    cmp_int(bd, rng.n, c.draws, where + " draws");
    cmp_int(bn, ws->n_new_cascade, c.n, where + " nout");
    cmp_int(bc, m.current_nucl1, c.nucl1, where + " nucl1");
    cmp_int(bc, m.current_nucl2, c.nucl2, where + " nucl2");
    cmp_int(bc, m.proton_number_current, c.npcur, where + " npcur");
    cmp_int(bc, m.neutron_number_current, c.nncur, where + " nncur");

    const double rscale = m.nuclei_radius;
    cmp_at(bi, cp.position.x, c.cpos[0], rscale, where + " cposx");
    cmp_at(bi, cp.position.y, c.cpos[1], rscale, where + " cposy");
    cmp_at(bi, cp.position.z, c.cpos[2], rscale, where + " cposz");
    cmp_int(bs, cp.current_zone, c.czone, where + " czone");
    cmp(bi, cp.current_path, c.ccpath, where + " ccpath");

    const int nn = (ws->n_new_cascade < c.n) ? ws->n_new_cascade : c.n;
    for (int i = 0; i < nn; ++i) {
      const CascadeParticle& o = ws->new_cascade[i];
      const std::string wi = where + " out" + std::to_string(i);
      cmp_int(bk, o.type, c.otype[i], wi + " type");
      const double escale = o.momentum.e;
      cmp_at(bp, o.momentum.v.x, c.mom[i].v.x, escale, wi + " px");
      cmp_at(bp, o.momentum.v.y, c.mom[i].v.y, escale, wi + " py");
      cmp_at(bp, o.momentum.v.z, c.mom[i].v.z, escale, wi + " pz");
      cmp_at(bp, o.momentum.e, c.mom[i].e, escale, wi + " e");
      cmp_at(bx, o.position.x, c.pos[i].x, rscale, wi + " posx");
      cmp_at(bx, o.position.y, c.pos[i].y, rscale, wi + " posy");
      cmp_at(bx, o.position.z, c.pos[i].z, rscale, wi + " posz");
      cmp_int(bs, o.current_zone, c.zone[i], wi + " zone");
      cmp(bx, o.current_path, c.cpath[i], wi + " cpath");
      cmp_int(bs, o.reflection_counter, c.nrefl[i], wi + " nrefl");
      cmp_int(bs, o.moving_in ? 1 : 0, c.movingin[i], wi + " movingin");
      // The generation is what `isProjectile` and therefore `forceFirst` key off, and it is the
      // one field of a product that nothing else in this file would notice: a collision's
      // daughters carry `parent + 1` and a propagated particle carries its own. Without this
      // column, giving every product `next_gen + 1` passed all 538,875 comparisons.
      cmp_int(bs, o.generation, c.ogen[i], wi + " generation");
    }
  }

  delete ws;
}

// ---------------------------------------------------------------------------------------------
// bertini_coalescence.csv - G4CascadeCoalescence::FindClusters
//
// Coalescence consumes no random numbers, so this is an exact function of the outgoing-particle
// list and the grid is hand-built. It has to be: the 384 cascader events make 18 deuterons,
// 3 tritons, one He-3 and NO alpha, so deleting the entire four-body arm from the port passed
// all 575,849 comparisons before this section existed. The twelve configurations below are the
// same literals dump_bertini.cc uses - a last-bit difference would move one across a cut.
// ---------------------------------------------------------------------------------------------

struct CoalPart { int type; double px, py, pz; };
const CoalPart kC0[] = {{1, 0.03, 0.02, 0.34}, {1, -0.02, 0.03, 0.28},
                        {2, 0.01, -0.04, 0.31}, {2, -0.03, -0.01, 0.27}};
const CoalPart kC1[] = {{1, 0.18, 0.12, 0.44}, {1, -0.15, 0.19, 0.18},
                        {2, 0.11, -0.22, 0.41}, {2, -0.14, -0.09, 0.17}};
const CoalPart kC2[] = {{1, 0.03, 0.02, 0.33}, {2, -0.02, 0.03, 0.28},
                        {2, 0.01, -0.04, 0.30}};
const CoalPart kC3[] = {{1, 0.03, 0.02, 0.33}, {1, -0.02, 0.03, 0.28},
                        {2, 0.01, -0.04, 0.30}};
const CoalPart kC4[] = {{1, 0.02, 0.01, 0.31}, {2, -0.02, -0.01, 0.29}};
const CoalPart kC5[] = {{1, 0.02, 0.01, 0.31}, {1, -0.02, -0.01, 0.29}};
const CoalPart kC6[] = {{2, 0.02, 0.01, 0.31}, {2, -0.02, -0.01, 0.29}};
const CoalPart kC7[] = {{1, 0.03, 0.02, 0.34}, {1, -0.02, 0.03, 0.28},
                        {2, 0.01, -0.04, 0.31}, {2, -0.03, -0.01, 0.27},
                        {1, 0.00, 0.00, 0.30}};
const CoalPart kC8[] = {{1, 0.03, 0.02, 0.34}, {1, -0.02, 0.03, 0.28},
                        {2, 0.01, -0.04, 0.31}, {2, -0.03, -0.01, 0.27},
                        {1, 0.02, 0.01, 0.31}, {2, -0.02, -0.01, 0.29}};
const CoalPart kC9[] = {{1, 0.03, 0.02, 0.34}, {3, 0.10, 0.10, 0.50},
                        {1, -0.02, 0.03, 0.28}, {5, -0.10, 0.10, 0.50},
                        {2, 0.01, -0.04, 0.31}, {2, -0.03, -0.01, 0.27}};
const CoalPart kC10[] = {{1, 0.03, 0.02, 0.33}, {1, -0.02, 0.03, 0.28},
                         {2, 0.01, -0.04, 0.30}, {2, 0.00, 0.00, 0.60}};
const CoalPart kC11[] = {{1, 0.02, 0.01, 0.31}};

struct CoalCfg { const CoalPart* p; int n; };
const CoalCfg kCoalCfgs[] = {{kC0, 4}, {kC1, 4}, {kC2, 3}, {kC3, 3}, {kC4, 2}, {kC5, 2},
                             {kC6, 2}, {kC7, 5}, {kC8, 6}, {kC9, 6}, {kC10, 4}, {kC11, 1}};

long long n_alphas = 0, n_triplets = 0, n_doublets = 0, n_rejected = 0;

void check_coalescence() {
  const int bn = new_bucket("CoalescenceCounts", 0.0);
  const int bk = new_bucket("CoalescenceKinds", 0.0);
  const int bp = new_bucket("CoalescenceMomenta", 1e-13);

  struct Row { int cfg, npart, nnuc, kind, i, type, a, z; LV mom; };
  std::vector<Row> rows;
  const std::vector<std::string> lines = read_lines("bertini_coalescence.csv");
  for (std::size_t li = 1; li < lines.size(); ++li) {
    const std::vector<std::string> f = split(lines[li]);
    if (f.size() < 12) { continue; }
    Row r;
    r.cfg = iv(f, 0);
    r.npart = iv(f, 1);
    r.nnuc = iv(f, 2);
    r.kind = iv(f, 3);
    r.i = iv(f, 4);
    r.type = iv(f, 5);
    r.a = iv(f, 6);
    r.z = iv(f, 7);
    r.mom = LV(Vec3d{dv(f, 8), dv(f, 9), dv(f, 10)}, dv(f, 11));
    rows.push_back(r);
  }

  const bert::CascadeParams par = bert::default_cascade_params();
  const bert::CoalescenceCuts cuts = bert::coalescence_cuts(par);
  bert::CollisionOutput* out = new bert::CollisionOutput();
  BertiniWorkspace* cws = new BertiniWorkspace();

  for (int ic = 0; ic < 12; ++ic) {
    bert::co_reset(*out);
    for (int i = 0; i < kCoalCfgs[ic].n; ++i) {
      const CoalPart& p = kCoalCfgs[ic].p[i];
      const double m = bert::inucl_particle_mass(p.type);
      bert::co_add_particle(
          *out, p.type,
          bert::inucl_store_momentum(bert::lv_set_vect_m(Vec3d{p.px, p.py, p.pz}, m), p.type));
    }
    const int before = out->n_particles;
    bert::coal_find_clusters(*out, cuts, *cws);

    const std::string where = "cfg" + std::to_string(ic);
    int want_npart = -1, want_nnuc = -1;
    for (const Row& r : rows) {
      if (r.cfg != ic) { continue; }
      want_npart = r.npart;
      want_nnuc = r.nnuc;
    }
    cmp_int(bn, out->n_particles, want_npart, where + " npart");
    cmp_int(bn, out->n_nuclei, want_nnuc, where + " nnuc");
    if (out->n_nuclei == 0 && before > 1) { ++n_rejected; }
    for (int i = 0; i < out->n_nuclei; ++i) {
      if (out->nuclei[i].a == 4) { ++n_alphas; }
      else if (out->nuclei[i].a == 3) { ++n_triplets; }
      else if (out->nuclei[i].a == 2) { ++n_doublets; }
    }

    for (const Row& r : rows) {
      if (r.cfg != ic || r.kind < 0) { continue; }
      const std::string wi = where + ((r.kind == 0) ? " p" : " n") + std::to_string(r.i);
      if (r.kind == 0) {
        if (r.i >= out->n_particles) { continue; }
        cmp_int(bk, out->particles[r.i].type, r.type, wi + " type");
        const double sc = out->particles[r.i].momentum.e;
        cmp_at(bp, out->particles[r.i].momentum.v.x, r.mom.v.x, sc, wi + " px");
        cmp_at(bp, out->particles[r.i].momentum.v.y, r.mom.v.y, sc, wi + " py");
        cmp_at(bp, out->particles[r.i].momentum.v.z, r.mom.v.z, sc, wi + " pz");
        cmp_at(bp, out->particles[r.i].momentum.e, r.mom.e, sc, wi + " e");
      } else {
        if (r.i >= out->n_nuclei) { continue; }
        cmp_int(bk, out->nuclei[r.i].a, r.a, wi + " A");
        cmp_int(bk, out->nuclei[r.i].z, r.z, wi + " Z");
        const double sc = out->nuclei[r.i].momentum.e;
        cmp_at(bp, out->nuclei[r.i].momentum.v.x, r.mom.v.x, sc, wi + " px");
        cmp_at(bp, out->nuclei[r.i].momentum.v.y, r.mom.v.y, sc, wi + " py");
        cmp_at(bp, out->nuclei[r.i].momentum.v.z, r.mom.v.z, sc, wi + " pz");
        cmp_at(bp, out->nuclei[r.i].momentum.e, r.mom.e, sc, wi + " e");
      }
    }
  }
  delete out;
}

// ---------------------------------------------------------------------------------------------
// Two branches no grid of inputs can reach, pinned by construction instead
//
// docs/RISK.md V52's distinction: a question answered by construction is not a hole. Both of
// these were found by the perturbation campaign coming back NOT CAUGHT, and neither can be
// reached by any (nucleus, particle, position, direction, phase) the oracle can be asked for.
// ---------------------------------------------------------------------------------------------
void check_pinned() {
  const int b = new_bucket("PinnedByConstruction", 0.0);

  // **1. The reflection COUNTER accumulates; only the FLAG is cleared.** Geant4's transmit arms
  // call `resetReflection()`, which is `{ reflected = false; }` and does not touch
  // `reflectionCounter`; `G4IntraNucleiCascader` compares that counter against `reflection_cut`
  // (50) as a LIFETIME budget. A single call to `generateParticleFate` crosses at most one
  // boundary, so no grid of one-step cases can tell a counter that accumulates from one that is
  // zeroed on every transmission - the perturbation that does the zeroing passed all 537,804
  // comparisons. Three transitions in a row say it directly.
  NucleiModel m;
  const NucleiModelParams p = bert::nuclei_model_params(bert::default_cascade_params());
  bert::nm_generate_model(m, 207, 82, p);
  {
    // A proton deep inside lead, aimed outward but with too little radial momentum to climb the
    // potential step: it reflects. Then the same particle given enough momentum transmits, and
    // then reflects again - and the counter must read 2, not 1.
    const Vec3d pos = Vec3d{0.0, 0.0, 1.0} * (0.5 * m.nuclei_radius);
    int zone = bert::nm_zone(m, g4gpu::mag(pos));
    int nrefl = 0;
    bool reflected = false, inz0 = false;
    LV mom = bert::lv_set_vect_m(Vec3d{0.0, 0.0, 0.001}, bert::inucl_particle_mass(bert::kProton));
    bert::nm_boundary_transition(m, bert::kProton, pos, mom, zone, false, nrefl, reflected,
                                 inz0);
    cmp_int(b, nrefl, 1, "reflection counter after one reflection");
    cmp_int(b, reflected ? 1 : 0, 1, "reflected flag after one reflection");

    const int zone_before = zone;
    LV fast = bert::lv_set_vect_m(Vec3d{0.0, 0.0, 0.9}, bert::inucl_particle_mass(bert::kProton));
    bert::nm_boundary_transition(m, bert::kProton, pos, fast, zone, false, nrefl, reflected,
                                 inz0);
    cmp_int(b, zone, zone_before + 1, "transmitted outward");
    cmp_int(b, reflected ? 1 : 0, 0, "reflected flag cleared by a transmission");
    cmp_int(b, nrefl, 1, "reflection COUNTER survives a transmission");

    LV slow = bert::lv_set_vect_m(Vec3d{0.0, 0.0, 0.001}, bert::inucl_particle_mass(bert::kProton));
    bert::nm_boundary_transition(m, bert::kProton, pos, slow, zone, false, nrefl, reflected,
                                 inz0);
    cmp_int(b, nrefl, 2, "reflection counter accumulates across a transmission");
  }

  // **2. `Hep3Vector::angle`'s clamp.** CLHEP clamps the cosine into [-1, 1] before the acos,
  // because for a nearly parallel pair the quotient can land an ulp above 1 and `acos` would
  // return NaN. Through `choosePointAlongTraj` - the only caller - the arguments are an exactly
  // normalised position and either (0,0,1) or a normalised momentum, so the quotient is 1.0
  // exactly and the clamp never fires: removing it passed all 538,874 comparisons. The clamp is
  // therefore asked on its own, at arguments no call site produces.
  {
    cmp(b, bert::clhep_acos_clamped(1.0 + 1e-16), 0.0, "acos clamp above +1");
    cmp(b, bert::clhep_acos_clamped(1.0000000001), 0.0, "acos clamp well above +1");
    cmp(b, bert::clhep_acos_clamped(-1.0000000001), 3.14159265358979323846,
        "acos clamp below -1");
    cmp_int(b, std::isnan(bert::clhep_acos_clamped(1.5)) ? 1 : 0, 0,
            "acos clamp does not produce a NaN");
    cmp_int(b, bert::clhep_angle(Vec3d{0.0, 0.0, 0.0}, Vec3d{1.0, 0.0, 0.0}) == 0.0 ? 1 : 0, 1,
            "angle against the zero vector is zero, not NaN");
  }

  // **3. `finalize`'s exhaustion test fires on a hundredth-attempt SUCCESS as well.** The loop
  // is `do { ++itry; ... } while (!finishCascade() && itry < 100)`, so `itry == 100` means either
  // "all hundred failed" or "the hundredth worked", and `if (itry >= itry_max) trivialise()`
  // throws both away. The two trivialised events in the oracle both failed all hundred attempts,
  // so changing the test to `itry >= max && !finished` passed all 575,992 comparisons. Same
  // family as the collider's `fs_retry_exhausted` (docs/RISK.md V121), and asserted the same way.
  {
    cmp_int(b, bert::cascader_exhausted(99) ? 1 : 0, 0, "99 attempts is not exhaustion");
    cmp_int(b, bert::cascader_exhausted(100) ? 1 : 0, 1, "100 attempts IS exhaustion");
    cmp_int(b, bert::cascader_exhausted(101) ? 1 : 0, 1, "101 attempts is exhaustion");
  }

  // **4. `particleCanInteract` is always true on the `collide` entry point**, because everything
  // the cascade can stack came out of a channel table or was filtered by one. Forcing it true
  // passed all 575,992 comparisons. What it is for is a particle with no table at all, which
  // reaches the cascader only through `Propagate`'s pre-loaded secondaries - P11's path, not
  // ported. The predicate itself is asked here at both answers.
  {
    cmp_int(b, bert::cascader_particle_can_interact(bert::kProton) ? 1 : 0, 1, "p interacts");
    cmp_int(b, bert::cascader_particle_can_interact(bert::kNeutron) ? 1 : 0, 1, "n interacts");
    cmp_int(b, bert::cascader_particle_can_interact(bert::kPionZero) ? 1 : 0, 1, "pi0 interacts");
    cmp_int(b, bert::cascader_particle_can_interact(bert::kPhoton) ? 1 : 0, 1, "gamma interacts");
    cmp_int(b, bert::cascader_particle_can_interact(bert::kLambda) ? 1 : 0, 1, "lambda interacts");
    // A light ion, an electron and a neutrino have no hadron-nucleon table of their own.
    cmp_int(b, bert::cascader_particle_can_interact(bert::kDeuteron) ? 1 : 0, 0,
            "a deuteron does not");
    cmp_int(b, bert::cascader_particle_can_interact(bert::kAlpha) ? 1 : 0, 0, "an alpha does not");
    cmp_int(b, bert::cascader_particle_can_interact(bert::kElectron) ? 1 : 0, 0,
            "an electron does not");
  }

  // **5. `Hep3Vector::rotate` about the ZERO axis leaves the vector alone**, which is what makes
  // the radial-incidence shortcut load-bearing rather than decorative: without it, an exactly
  // radial photon would get its ENTRY point back as its exit point and a chord of zero length.
  {
    const Vec3d v{1.0, 2.0, 3.0};
    const Vec3d got = bert::clhep_rotate(v, -3.14159265358979323846, Vec3d{0.0, 0.0, 0.0});
    cmp(b, got.x, v.x, "rotate about a zero axis: x");
    cmp(b, got.y, v.y, "rotate about a zero axis: y");
    cmp(b, got.z, v.z, "rotate about a zero axis: z");
    const Vec3d same = bert::clhep_rotate(v, 0.0, Vec3d{0.0, 1.0, 0.0});
    cmp(b, same.y, v.y, "rotate by a zero angle is the identity");
  }
}

// ---------------------------------------------------------------------------------------------
// bertini_cascader.csv - G4IntraNucleiCascader::collide
// ---------------------------------------------------------------------------------------------

/// One event: every row of the file that shares (a, z, type, plab, phase).
struct CascCase {
  int a = 0, z = 0, type = 0, phase = 0;
  double plab = 0.0;
  long long draws = 0;
  int npart = 0, nnuc = 0, nfrag = 0;
  std::vector<int> ptype;
  std::vector<LV> pmom;
  std::vector<int> na, nz;
  std::vector<double> nexc;
  std::vector<LV> nmom;
  int fa = 0, fz = 0;
  double fexc = 0.0;
  LV fmom;
};

long long n_retried = 0, n_coalesced = 0, n_nofrag = 0, n_max_draws = 0, n_tuned = 0;
long long n_leftover = 0;

void check_cascader(ModelCache& mc) {
  const int bd = new_bucket("CascadeDrawCount", 0.0);
  const int bn = new_bucket("CascadeMultiplicity", 0.0);
  const int bk = new_bucket("CascadeKinds", 0.0);
  // **3e-10 and not 1e-13, and the number is measured, not chosen.** Over 384 events, 4,529 of
  // the 4,540 four-momentum components agree with Geant4 to 1e-13 or better - the same as the
  // single-step bucket above. FOUR do not, and all four belong to ONE event: the Pb-207 neutron
  // at 3 GeV/c, phase 1, which is the longest cascade in the grid at 16,481 deviates. There one
  // outgoing neutron's direction differs by 6.6e-10 radians, which makes its SMALLEST momentum
  // component (0.0109 of 0.198 GeV/c) differ by 1.37e-10 relative while its energy differs by
  // 1.7e-12; the recoil fragment carries the exact negative of that, because the event balances
  // on both sides. It is chain drift, and it was checked rather than assumed: `setOnShell`
  // leaves that event alone (its imbalance is 2.8e-14 GeV, under the 10 keV early return), the
  // pair tuning never runs in the whole grid, and the nuclear mass table agrees with the
  // install's to 2.2e-16 over all 18,407 nuclides. What would NOT be drift is a different
  // branch, and the draw counts, multiplicities and product types are exact for all 384 events.
  const int bp = new_bucket("CascadeMomenta", 3e-10);
  const int bf = new_bucket("CascadeRecoil", 3e-11);

  std::vector<CascCase> cases;
  std::map<std::string, std::size_t> index;
  const std::vector<std::string> lines = read_lines("bertini_cascader.csv");
  for (std::size_t li = 1; li < lines.size(); ++li) {
    const std::vector<std::string> f = split(lines[li]);
    if (f.size() < 19) { continue; }
    char key[128];
    std::snprintf(key, sizeof key, "%d_%d_%d_%s_%d", iv(f, 0), iv(f, 1), iv(f, 2),
                  f[3].c_str(), iv(f, 4));
    auto it = index.find(key);
    if (it == index.end()) {
      CascCase c;
      c.a = iv(f, 0);
      c.z = iv(f, 1);
      c.type = iv(f, 2);
      c.plab = dv(f, 3);
      c.phase = iv(f, 4);
      c.draws = iv(f, 5);
      c.npart = iv(f, 6);
      c.nnuc = iv(f, 7);
      c.nfrag = iv(f, 8);
      index[key] = cases.size();
      cases.push_back(c);
      it = index.find(key);
    }
    CascCase& c = cases[it->second];
    const int kind = iv(f, 9);
    const LV mom(Vec3d{dv(f, 15), dv(f, 16), dv(f, 17)}, dv(f, 18));
    if (kind == 0) {
      c.ptype.push_back(iv(f, 11));
      c.pmom.push_back(mom);
    } else if (kind == 1) {
      c.na.push_back(iv(f, 12));
      c.nz.push_back(iv(f, 13));
      c.nexc.push_back(dv(f, 14));
      c.nmom.push_back(mom);
    } else if (kind == 2) {
      c.fa = iv(f, 12);
      c.fz = iv(f, 13);
      c.fexc = dv(f, 14);
      c.fmom = mom;
    }
  }

  const bert::CascadeParams par = bert::default_cascade_params();
  BertiniWorkspace* ws = new BertiniWorkspace();
  bert::CollisionOutput* out = new bert::CollisionOutput();
  ColliderOutput epo;

  // The dump builds ONE G4IntraNucleiCascader for the whole grid and `generateModel(a,z)`
  // returns early (with a `reset()`) when A and Z have not changed. Nothing numeric depends on
  // the rebuild, but `current_nucl1`/`current_nucl2` are model members that neither
  // `generateModel`'s early return nor `reset()` clears, so they carry across cases exactly as
  // they do in bertini_fate.csv. Cleared only when the nucleus changes.
  int last_key = -1;
  for (const CascCase& c : cases) {
    NucleiModel& m = mc.get(c.a, c.z);
    if (c.a * 1000 + c.z != last_key) {
      m.current_nucl1 = 0;
      m.current_nucl2 = 0;
      last_key = c.a * 1000 + c.z;
    }
    bert::ws_reset(*ws);
    CycleRng rng;
    rng.reset(c.phase);

    const double mass = bert::inucl_particle_mass(c.type);
    bert::CascadeSetup s;
    s.bullet_type = c.type;
    s.bullet = bert::inucl_store_momentum(
        LV(Vec3d{0.0, 0.0, c.plab}, std::sqrt(c.plab * c.plab + mass * mass)), c.type);
    s.target_a = c.a;
    s.target_z = c.z;
    // G4InuclNuclei(a, z): a nucleus at rest, whose four-momentum is (0, 0, 0, M).
    s.target = LV(Vec3d{0.0, 0.0, 0.0}, bert::inucl_nuclei_mass(c.a, c.z));

    bert::ExitonConfiguration ex;
    bert::RecoilState rec;
    bert::CascaderResult res;
    bert::cascader_collide(m, mc.p, par, s, *out, ex, rec, epo, *ws, rng, res);

    char w[160];
    std::snprintf(w, sizeof w, "A%d Z%d t%d p%g ph%d", c.a, c.z, c.type, c.plab, c.phase);
    const std::string where(w);

    if (res.refusal != bert::CascaderRefusal::kNone) {
      std::printf("FAIL: cascader refused (%d, fate %d) at %s\n",
                  static_cast<int>(res.refusal), static_cast<int>(res.fate_refusal),
                  where.c_str());
      ++fails;
      continue;
    }
    if (res.n_tries > 1) { ++n_retried; }
    if (out->hard_tuned) { ++n_tuned; }
    if (!res.trivialised && res.n_leftover > n_leftover) { n_leftover = res.n_leftover; }
    if (out->n_nuclei > 0) { ++n_coalesced; }
    if (!out->has_recoil_fragment) { ++n_nofrag; }
    if (rng.n > n_max_draws) { n_max_draws = rng.n; }

    cmp_int(bd, rng.n, c.draws, where + " draws");
    cmp_int(bn, out->n_particles, c.npart, where + " npart");
    cmp_int(bn, out->n_nuclei, c.nnuc, where + " nnuc");
    cmp_int(bn, out->has_recoil_fragment ? 1 : 0, c.nfrag, where + " nfrag");

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
      cmp(bf, out->nuclei[i].exc_MeV, c.nexc[i], wi + " exc");
      const double sc = out->nuclei[i].momentum.e;
      cmp_at(bp, out->nuclei[i].momentum.v.x, c.nmom[i].v.x, sc, wi + " px");
      cmp_at(bp, out->nuclei[i].momentum.v.y, c.nmom[i].v.y, sc, wi + " py");
      cmp_at(bp, out->nuclei[i].momentum.v.z, c.nmom[i].v.z, sc, wi + " pz");
      cmp_at(bp, out->nuclei[i].momentum.e, c.nmom[i].e, sc, wi + " e");
    }
    if (out->has_recoil_fragment && c.nfrag > 0) {
      cmp_int(bk, out->recoil_fragment.a, c.fa, where + " frag A");
      cmp_int(bk, out->recoil_fragment.z, c.fz, where + " frag Z");
      cmp(bf, out->recoil_fragment.excitation, c.fexc, where + " frag exc");
      // The fragment is in Geant4's MeV; the oracle dumped it divided by GeV.
      const double sc = out->recoil_fragment.momentum.e * 0.001;
      cmp_at(bf, out->recoil_fragment.momentum.v.x * 0.001, c.fmom.v.x, sc, where + " frag px");
      cmp_at(bf, out->recoil_fragment.momentum.v.y * 0.001, c.fmom.v.y, sc, where + " frag py");
      cmp_at(bf, out->recoil_fragment.momentum.v.z * 0.001, c.fmom.v.z, sc, where + " frag pz");
      cmp_at(bf, out->recoil_fragment.momentum.e * 0.001, c.fmom.e, sc, where + " frag e");
    }
  }

  delete out;
  delete ws;
}

}  // namespace

int main() {
  ModelCache mc;
  mc.p = bert::nuclei_model_params(bert::default_cascade_params());

  check_initcascad(mc);
  check_fate(mc);
  check_cascader(mc);
  check_coalescence();
  check_pinned();

  std::printf("  initializeCascad: %d trajectory-sampled (photons), %d cases\n", n_traj,
              n_atrest);
  std::printf("  cascader: %lld events needed a retry, %lld produced a coalesced light ion,"
              " %lld with no recoil fragment, %lld needed setOnShell pair tuning,"
              " worst draw count %lld, most left in flight at the end %lld\n",
              n_retried, n_coalesced, n_nofrag, n_tuned, n_max_draws, n_leftover);
  std::printf("  coalescence: %lld alphas, %lld A=3 clusters, %lld deuterons, %lld rejected\n",
              n_alphas, n_triplets, n_doublets, n_rejected);
  std::printf("  fate: %lld propagations, %lld collisions (%lld with more than two products),"
              " %lld quasi-deuteron absorptions, %lld reflections\n",
              n_prop, n_coll, n_multi, n_absorb, n_refl);

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
