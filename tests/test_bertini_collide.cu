// G4ElementaryParticleCollider::collide against ref/oracle/bertini_epcollide.csv, exact.
//
// One collision is the whole two-body chain: the multiplicity draw, the final-state draw, the
// mass fill, the choice of angular and momentum generator, the momentum moduli, the polar and
// azimuthal angles, two kinds of rotation, the boost back to the lab and a descending-kinetic-
// energy sort. Under the engines `ref/dump/dump_bertini.cc` installs, all of that is a
// deterministic function of (types, p_lab, target momentum, phase), so it is compared value for
// value rather than as a distribution - **10,752 cases**: 28 particle pairs x 12 lab momenta x
// 2 target states x 8 phases x 2 generator settings.
//
// **Two target states, because a head-on collision hides a function.** With the target at rest
// the CM velocity is parallel to the SCM axis, `G4LorentzConvertor::degenerated` is true and
// `rotate(mom)` returns its argument - so on a grid of head-on collisions the rotation that ends
// every two-body final state is never applied, and deleting it changes nothing. That is not a
// hypothesis; it is what the perturbation campaign found (docs/RISK.md V124). The second state
// gives the target a Fermi momentum off every axis, which is what the cascade presents anyway.
//
// **The draw count is compared first and it is the assertion that earns its keep.** A collision
// costs between 4 and 60 deviates depending on which arm it takes, and almost every mistake in
// this chain is a mistake about how many: a retry loop that runs nine times instead of ten, a
// rejection sampler whose guard is evaluated in the other order, an early return that skips a
// draw. The cycle has period 8, so a one-deviate error usually leaves the FIRST few numbers
// looking plausible and corrupts everything after; the count says so immediately and names the
// case.
//
// The two cases below are not in the oracle grid because no physical input reaches them, and
// both are pinned by construction at the end of this file instead (RISK V52's distinction
// between a hole and a question): the pi-N-on-a-bound-nucleon arm, which needs a residual
// nucleus mass this entry point does not have and is refused by name, and the muon-on-dibaryon
// arm, which is P12's.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "core/rng.cuh"
#include "physics/hadronic/bertini/cascade_params.cuh"
#include "physics/hadronic/bertini/ep_collider.cuh"

using namespace g4gpu;
using namespace g4gpu::physics::hadronic;
using bert::BertiniWorkspace;
using bert::ColliderOutput;
using bert::ColliderRefusal;
using bert::LV;
using bert::Vec3d;

/// The deliverable is device code. This kernel is never launched - the package is validated on
/// the host and the machine's GPU belongs to the integration builds - but instantiating it is
/// what proves the collider compiles for the device: no host-only header, no `std::` call
/// without a device overload, and a workspace reached through a pointer rather than a local.
/// `-Xptxas -v` on this translation unit is where the register and stack numbers in the package
/// report come from.
///
/// `ColliderOutput` is taken BY POINTER for the same reason the workspace is. It holds nine
/// four-vectors and their type codes, and it is the CALLER's buffer - in the cascader it will be
/// what `G4CollisionOutput` is in Geant4, a data member reused across collisions. Declaring it
/// as a local here charges the collider for the caller's storage and reports a frame no real
/// kernel pays: 1,216 bytes, against 880 with it passed in. Moving `FinalStateConfig` into the
/// workspace - where Geant4 also keeps it, as data members of the algorithm object - took that
/// 880 to **416 bytes and 158 registers, 0 spill**, which is the number in the package report.
__global__ void bertini_collide_probe(int type1, int type2, double plab, int* n_out,
                                      double* out, ColliderOutput* out_c,
                                      BertiniWorkspace* ws) {
  Philox<double> rng(7u, 8u, 9u);
  const bert::CascadeParams par = bert::default_cascade_params();
  const double m1 = bert::inucl_particle_mass(type1);
  const double m2 = bert::inucl_particle_mass(type2);
  const LV mom1 = LV(Vec3d{0.0, 0.0, plab}, std::sqrt(plab * plab + m1 * m1));
  const LV mom2 = LV(Vec3d{0.0, 0.0, 0.0}, m2);
  bert::ep_collide(type1, mom1, type2, mom2, par, *out_c, *ws, rng);
  n_out[0] = out_c->n;
  n_out[1] = static_cast<int>(out_c->refusal);
  n_out[2] = static_cast<int>(out_c->channel_refusal);
  for (int i = 0; i < bert::kMaxFinalStateSize; ++i) {
    out[4 * i + 0] = out_c->momenta[i].v.x;
    out[4 * i + 1] = out_c->momenta[i].v.y;
    out[4 * i + 2] = out_c->momenta[i].v.z;
    out[4 * i + 3] = out_c->momenta[i].e;
  }
}

namespace {

int fails = 0;

/// The eight uniforms dump_bertini.cc's CycleEngine cycles through, in its order. Written out on
/// both sides rather than derived, because if the two lists diverge every exact comparison in
/// this file turns into noise.
__host__ __device__ inline const double* cycle_seq() {
  static const double v[8] = {0.05, 0.37, 0.63, 0.91, 0.12, 0.78, 0.29, 0.55};
  return v;
}

/// One engine, two behaviours, because one of Geant4's samplers cannot be driven by the other.
///
/// `kind == 0` is dump_bertini.cc's CycleEngine: eight fixed uniforms, which makes every bounded
/// sampler a deterministic function of (inputs, phase) and lets the draw count be compared.
///
/// `kind == 1` is its LcgEngine, and it exists because `G4CascadeFinalStateAlgorithm::BetaKopylov`
/// is an UNBOUNDED rejection loop drawing two deviates per trial: under a period-8 cycle that is
/// four distinct trials in total, and at the multiplicities the phase-space generator reaches,
/// none of the four is ever accepted - the loop does not terminate. The LCG is written out in
/// exact 64-bit integer arithmetic on both sides, so it is still reproduced bit for bit; it just
/// has a period long enough to be a random number generator.
struct CycleRng {
  int kind = 0;
  int phase = 0;
  long long n = 0;
  unsigned long long s = 0;
  __host__ __device__ void reset(int k, int p) {
    kind = k;
    phase = p;
    n = 0;
    s = 88172645463325252ull + 1442695040888963407ull * static_cast<unsigned long long>(p + 1);
  }
  __host__ __device__ void reset(int p) { reset(0, p); }
  __host__ __device__ double uniform() {
    ++n;
    if (kind == 0) {
      const unsigned i = static_cast<unsigned>(n - 1) + static_cast<unsigned>(phase);
      return cycle_seq()[i % 8u];
    }
    s = 6364136223846793005ull * s + 1442695040888963407ull;
    return (static_cast<double>(s >> 11) + 0.5) * (1.0 / 9007199254740992.0);
  }
};

// ---------------------------------------------------------------------------------------------
// Comparison bookkeeping, as tests/test_bertini_data.cu and tests/test_precompound.cu define it
// ---------------------------------------------------------------------------------------------

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

/// Like `cmp`, but with the scale given rather than taken from `want`.
///
/// A momentum COMPONENT is the wrong thing to divide by. `finalState[0]` is built at a polar
/// angle and then rotated onto the SCM axis, so any one of px, py, pz can land near zero while
/// the vector it belongs to is a few hundred MeV/c - and a component that is 1e-11 of itself is
/// then 1e-13 of the four-vector it is part of, which is the only scale the number actually
/// has. Every momentum comparison below is normalised by the product's own energy, which for a
/// hadron is never small.
void cmp_at(int bi, double got, double want, double scale, const std::string& where) {
  Bucket& b = buckets[bi];
  ++b.n;
  const double rel = std::fabs(got - want) / ((scale > 0.0) ? scale : 1.0);
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

/// The two target momenta the dump uses, in its order. `tm == 0` is a target at rest, which
/// makes the collision frame DEGENERATE - the CM velocity is parallel to the SCM axis and
/// `G4LorentzConvertor::rotate(mom)` returns its argument. `tm == 1` gives the target a Fermi
/// momentum off every axis, which is the case the cascade actually produces and the only one in
/// which the rotation does anything. Both are compared: the degenerate branch is a branch.
inline const double* target_momentum(int tm) {
  static const double v[2][3] = {{0.0, 0.0, 0.0}, {0.15, -0.08, 0.11}};
  return v[tm & 1];
}

/// One case: every row of the CSV that shares (ps, tm, type1, type2, plab, phase).
struct Case {
  int ps = 0;                        ///< 0 the INUCL generators, 1 usePhaseSpace/Kopylov
  int tm = 0;                        ///< 0 target at rest, 1 target with a Fermi momentum
  int type1 = 0, type2 = 0, phase = 0;
  double plab = 0.0;
  int draws = 0;
  int n = 0;
  std::vector<int> kinds;
  std::vector<LV> momenta;
};

// ---------------------------------------------------------------------------------------------
// bertini_epcollide.csv
// ---------------------------------------------------------------------------------------------
void check_collide() {
  const int bd = new_bucket("CollideDrawCount", 0.0);
  const int bn = new_bucket("CollideMultiplicity", 0.0);
  const int bk = new_bucket("CollideKinds", 0.0);
  // 1e-13, not the 1e-12 this started at and not 0. The worst over 104,012 components is
  // 2.5e-14, and what is left is the genuine last bits of `acos`, `exp`, `sqrt` and the
  // Lorentz boost - two independent compilations of the same formulas, not two formulas. The
  // tolerance was 1e-12 while a systematic 2e-7 disagreement on every gamma-nucleon row was
  // sitting under it; that was not what the tolerance was buying, but it is what it bought.
  const int bp = new_bucket("CollideMomenta", 1e-13);
  const int br = new_bucket("CollideRefusal", 0.0);

  // Group the rows. The dump writes one row per outgoing particle, or a single row with
  // n = 0 and i = -1 when the collision produced nothing.
  std::vector<Case> cases;
  std::map<std::string, std::size_t> index;
  const std::vector<std::string> lines = read_lines("bertini_epcollide.csv");
  for (std::size_t li = 1; li < lines.size(); ++li) {
    const std::vector<std::string> f = split(lines[li]);
    if (f.size() < 14) { continue; }
    const std::string key =
        f[0] + "," + f[1] + "," + f[2] + "," + f[3] + "," + f[4] + "," + f[5];
    std::size_t at;
    const std::map<std::string, std::size_t>::iterator it = index.find(key);
    if (it == index.end()) {
      Case c;
      c.ps = iv(f, 0);
      c.tm = iv(f, 1);
      c.type1 = iv(f, 2);
      c.type2 = iv(f, 3);
      c.plab = dv(f, 4);
      c.phase = iv(f, 5);
      c.draws = iv(f, 6);
      c.n = iv(f, 7);
      at = cases.size();
      cases.push_back(c);
      index[key] = at;
    } else {
      at = it->second;
    }
    const int i = iv(f, 8);
    if (i < 0) { continue; }           // the n = 0 placeholder row
    Case& c = cases[at];
    if (static_cast<int>(c.kinds.size()) <= i) {
      c.kinds.resize(i + 1, 0);
      c.momenta.resize(i + 1);
    }
    c.kinds[i] = iv(f, 9);
    c.momenta[i] = LV(Vec3d{dv(f, 10), dv(f, 11), dv(f, 12)}, dv(f, 13));
  }
  if (cases.empty()) {
    std::printf("FAIL: bertini_epcollide.csv produced no cases\n");
    ++fails;
    return;
  }

  // The workspace is tens of kilobytes; it is the object the package exists to keep off the
  // stack, so the test allocates one and reuses it exactly as a kernel would.
  BertiniWorkspace* ws = new BertiniWorkspace;
  const bert::CascadeParams dumped = bert::default_cascade_params();

  // Counters, printed at the end. They are not assertions but they say which arms this grid
  // actually reached, which is what makes the buckets above non-vacuous.
  long long n_empty = 0, n_kin_failed = 0, n_absorption = 0, n_kopylov = 0;
  long long n_tm[2] = {0, 0};
  int max_mult = 0, max_draws = 0, max_mult_ps = 0, max_mult_inucl = 0;

  for (const Case& c : cases) {
    char w[192];
    std::snprintf(w, sizeof w, "ps%d tm%d %d x %d at plab %.3g phase %d", c.ps, c.tm, c.type1,
                  c.type2, c.plab, c.phase);
    // The second pass of the dump ran with `/process/had/cascade/usePhaseSpace true`, which
    // replaces the INUCL multi-body generator with FillUsingKopylov. It is 0 in the install, so
    // without that pass the Kopylov transcription would be a reading that nothing measures.
    bert::CascadeParams par = dumped;
    par.use_phase_space = (c.ps == 1);
    const double m1 = bert::inucl_particle_mass(c.type1);
    const double m2 = bert::inucl_particle_mass(c.type2);
    const double* tp = target_momentum(c.tm);
    // The dump builds a `G4InuclElementaryParticle` from each of these, and that constructor
    // takes the four-vector apart into a direction, a kinetic energy and a mass. `collide` then
    // reads it back. So the collider never sees the four-vector written here - it sees what
    // came back out - and the test has to hand it the same thing. See `inucl_store_momentum`.
    const LV mom1 = bert::inucl_store_momentum(
        LV(Vec3d{0.0, 0.0, c.plab}, std::sqrt(c.plab * c.plab + m1 * m1)), c.type1);
    const LV mom2 =
        bert::inucl_store_momentum(bert::lv_set_vect_m(Vec3d{tp[0], tp[1], tp[2]}, m2), c.type2);

    CycleRng rng;
    rng.reset(c.ps, c.phase);
    bert::ws_reset(*ws);
    ColliderOutput out;
    bert::ep_collide(c.type1, mom1, c.type2, mom2, par, out, *ws, rng);

    cmp_int(bd, rng.n, c.draws, std::string("draws ") + w);
    cmp_int(bn, out.n, c.n, std::string("multiplicity ") + w);
    // Geant4 reaches an empty final state by returning; the port reaches it by refusing, and
    // the two have to agree on WHICH cases those are. A refusal with a non-empty output, or an
    // empty output with no refusal, is a failure here even if the multiplicity matched.
    cmp_int(br, (out.n == 0) ? 1 : 0, (c.n == 0) ? 1 : 0, std::string("empty ") + w);
    if (out.n == 0) {
      ++n_empty;
      if (out.refusal == ColliderRefusal::kKinematicsFailed) { ++n_kin_failed; }
      // Nothing may be refused for a reason this grid is not allowed to produce: every empty
      // case here must be the kinematics retry loop giving up, not a missing table or a
      // capacity.
      cmp_int(br, static_cast<int>(out.refusal),
              static_cast<int>(ColliderRefusal::kKinematicsFailed),
              std::string("refusal ") + w);
      continue;
    }
    if (bert::inucl_is_quasideuteron(c.type1) || bert::inucl_is_quasideuteron(c.type2)) {
      ++n_absorption;
    }
    if (c.ps == 1 && out.n > 2) {
      ++n_kopylov;
      if (out.n > max_mult_ps) { max_mult_ps = out.n; }
    } else if (c.ps == 0 && out.n > max_mult_inucl) {
      max_mult_inucl = out.n;
    }
    ++n_tm[c.tm & 1];
    if (out.n > max_mult) { max_mult = out.n; }
    if (rng.n > max_draws) { max_draws = static_cast<int>(rng.n); }
    cmp_int(br, static_cast<int>(out.refusal), 0, std::string("no refusal ") + w);

    const int n = (out.n < c.n) ? out.n : c.n;
    for (int i = 0; i < n; ++i) {
      char w2[192];
      std::snprintf(w2, sizeof w2, "%s product %d", w, i);
      cmp_int(bk, out.kinds[i], c.kinds[i], std::string("kind ") + w2);
      const double s = c.momenta[i].e;
      cmp_at(bp, out.momenta[i].v.x, c.momenta[i].v.x, s, std::string("px ") + w2);
      cmp_at(bp, out.momenta[i].v.y, c.momenta[i].v.y, s, std::string("py ") + w2);
      cmp_at(bp, out.momenta[i].v.z, c.momenta[i].v.z, s, std::string("pz ") + w2);
      cmp_at(bp, out.momenta[i].e, c.momenta[i].e, s, std::string("e ") + w2);
    }
  }

  std::printf("  collide: %zu cases, %lld empty (%lld from the kinematics retry loop), "
              "%lld absorptions, %lld Kopylov multi-body, max multiplicity %d (%d with phase "
              "space), max draws %d\n",
              cases.size(), n_empty, n_kin_failed, n_absorption, n_kopylov, max_mult,
              max_mult_ps, max_draws);

  // The grid has to have reached every arm this file claims to test, or the buckets above are
  // measuring one code path and reporting on six.
  const int bc = new_bucket("CollideArmsReached", 0.0);
  cmp_int(bc, (n_empty > 0) ? 1 : 0, 1, "at least one collision produced nothing");
  cmp_int(bc, (n_absorption > 0) ? 1 : 0, 1, "at least one dibaryon absorption");
  cmp_int(bc, (max_mult >= 5) ? 1 : 0, 1, "at least one many-body final state");
  cmp_int(bc, (n_kopylov > 0) ? 1 : 0, 1, "at least one Kopylov phase-space final state");
  // Phase space reaches multiplicities the INUCL generator cannot: FillUsingKopylov is a chain
  // of two-body decays with no rejection, so it never fails where FillMagnitudes' triangle
  // condition and FillDirManyBody's |cos| < 0.9999 do.
  cmp_int(bc, (max_mult_ps > max_mult_inucl) ? 1 : 0, 1,
          "phase space produced a larger final state than the INUCL generator's largest");
  // Both target states have to be present, or `G4LorentzConvertor::rotate(mom)` is never
  // applied: with a target at rest the frame is degenerate and the rotation is the identity.
  cmp_int(bc, (n_tm[0] > 0) ? 1 : 0, 1, "the degenerate frame (target at rest) is in the grid");
  cmp_int(bc, (n_tm[1] > 0) ? 1 : 0, 1, "and so is the rotated one (target with Fermi momentum)");

  delete ws;
}

// ---------------------------------------------------------------------------------------------
// The two arms the oracle grid cannot reach, pinned by construction
// ---------------------------------------------------------------------------------------------
//
// Both are refusals, and a refusal that is never exercised is a comment. These assertions are
// what stop the refusal from being deleted or from quietly becoming an empty final state.
void check_refusals() {
  const int b = new_bucket("CollideRefusedByName", 0.0);
  BertiniWorkspace* ws = new BertiniWorkspace;
  bert::CascadeParams par = bert::default_cascade_params();
  CycleRng rng;

  // 1. The pi-N-on-a-bound-nucleon arm. `piNAbsorption` is 0 in the install, so
  // `pionNucleonAbsorption` is false for every deviate and the arm is dead - but it is dead
  // because of a PARAMETER, not because of the code, so the parameter is changed here and the
  // refusal is checked. Geant4 would call generateSCMpionNAbsorption, which needs the residual
  // nucleus mass; this entry point has no nucleus, so it refuses by name.
  par.pin_absorption = 1.0;
  ColliderOutput out;
  const double mpi = bert::inucl_particle_mass(bert::kPionMinus);
  const double mp = bert::inucl_particle_mass(bert::kProton);
  const double plab = 0.05;                 // well under the 50 MeV kinetic-energy gate
  const LV pi(Vec3d{0.0, 0.0, plab}, std::sqrt(plab * plab + mpi * mpi));
  const LV pro(Vec3d{0.0, 0.0, 0.0}, mp);
  rng.reset(0);
  bert::ws_reset(*ws);
  bert::ep_collide(bert::kPionMinus, pi, bert::kProton, pro, par, out, *ws, rng);
  cmp_int(b, static_cast<int>(out.refusal),
          static_cast<int>(ColliderRefusal::kPionNAbsorptionNucleus), "pi- p absorption arm");
  cmp_int(b, out.n, 0, "pi- p absorption produces nothing");
  cmp_int(b, rng.n, 1, "the absorption probability costs exactly one deviate");

  // ... and with the dumped parameter the SAME collision takes the channel-table arm instead,
  // having drawn that one deviate. If it did not, the line above would be testing a constant.
  par.pin_absorption = bert::default_cascade_params().pin_absorption;
  rng.reset(0);
  bert::ws_reset(*ws);
  bert::ep_collide(bert::kPionMinus, pi, bert::kProton, pro, par, out, *ws, rng);
  cmp_int(b, static_cast<int>(out.refusal), static_cast<int>(ColliderRefusal::kNone),
          "pi- p at the dumped piNAbsorption is a normal collision");
  cmp_int(b, (rng.n > 1) ? 1 : 0, 1, "and it still paid for the absorption deviate");

  // 2. The muon-on-dibaryon arm - G4ElementaryParticleCollider::generateSCMmuonAbsorption, which
  // is muon capture and belongs to P12. It is refused by name at the point the call would be.
  const double mmu = bert::inucl_particle_mass(bert::kMuonMinus);
  const LV mu(Vec3d{0.0, 0.0, 0.0}, mmu);
  const LV pn(Vec3d{0.0, 0.0, 0.0}, bert::inucl_particle_mass(bert::kUnboundPN));
  rng.reset(0);
  bert::ws_reset(*ws);
  bert::ep_collide(bert::kMuonMinus, mu, bert::kUnboundPN, pn, par, out, *ws, rng);
  cmp_int(b, static_cast<int>(out.refusal), static_cast<int>(ColliderRefusal::kMuonAbsorption),
          "mu- on a dibaryon");
  cmp_int(b, out.n, 0, "mu- absorption produces nothing");
  // useQuasiDeuteron admits the muon, so the refusal above is the muon test firing and not the
  // partner test. A proton on the same dibaryon is refused for the other reason, which is what
  // makes the two distinguishable.
  rng.reset(0);
  bert::ws_reset(*ws);
  bert::ep_collide(bert::kProton, LV(Vec3d{0.0, 0.0, 1.0}, std::sqrt(1.0 + mp * mp)),
                   bert::kUnboundPN, pn, par, out, *ws, rng);
  cmp_int(b, static_cast<int>(out.refusal),
          static_cast<int>(ColliderRefusal::kIllegalDibaryonPartner), "p on a dibaryon");

  // 3. A neutrino is dropped before anything else happens, without a draw.
  rng.reset(0);
  bert::ws_reset(*ws);
  bert::ep_collide(bert::kMuonNu, LV(Vec3d{0.0, 0.0, 1.0}, 1.0), bert::kProton, pro, par, out,
                   *ws, rng);
  cmp_int(b, static_cast<int>(out.refusal),
          static_cast<int>(ColliderRefusal::kNeutrinoProjectile), "nu_mu on a proton");
  cmp_int(b, rng.n, 0, "a neutrino costs no deviates");

  // 4. An initial state with no channel table and no dibaryon. A proton on a bound DEUTERON:
  // type 41, which is a real INUCL type but not a `quasi_deutron()` (that test is `type > 100`),
  // so neither arm of collide() claims it.
  const LV deu(Vec3d{0.0, 0.0, 0.0}, bert::inucl_particle_mass(bert::kDeuteron));
  rng.reset(0);
  bert::ws_reset(*ws);
  bert::ep_collide(bert::kProton, LV(Vec3d{0.0, 0.0, 1.0}, std::sqrt(1.0 + mp * mp)),
                   bert::kDeuteron, deu, par, out, *ws, rng);
  cmp_int(b, static_cast<int>(out.refusal), static_cast<int>(ColliderRefusal::kNoChannelTable),
          "p on a deuteron");
  cmp_int(b, rng.n, 0, "a missing table costs no deviates");

  // 5. **The channel-table key is a PRODUCT of type codes, and the products are not unique.**
  // `interCase.hadrons()` is `type1 * type2` and the codes are 1, 2, 3, 5, 7, 9, 11, ..., so
  // pi+ x pi- is 15 - which is also K0 x p, a real table. Five of the 34 keys alias this way:
  // 9 (pi+ pi+ / gamma p), 15, 21 (pi+ pi0 / lambda p), 25 (pi- pi- / sigma0 p) and
  // 33 (pi+ K+ / omega- p).
  //
  // It costs nothing, and the reason is worth recording because it is not obvious: the alias
  // only defeats the "cannot collide" guard at the top of `collide`. Both of the arms BELOW
  // that guard test the particles themselves - one needs a `nucleon()`, the other a
  // `quasi_deutron()` - and two pions are neither, so the collision falls through to the empty
  // "failed to collide" path anyway. The alias changes which of two routes leads to the same
  // empty final state. The port reproduces the aliasing rather than adding a check Geant4 does
  // not have, and this asserts the whole of the consequence: no refusal, and nothing produced.
  rng.reset(0);
  bert::ws_reset(*ws);
  bert::ep_collide(bert::kPionPlus, pi, bert::kPionMinus, LV(Vec3d{0.0, 0.0, 0.0}, mpi), par,
                   out, *ws, rng);
  cmp_int(b, static_cast<int>(out.refusal), static_cast<int>(ColliderRefusal::kNone),
          "pi+ on pi- is not refused - the key aliases onto the K0 p table");
  cmp_int(b, out.n, 0, "but neither arm of collide() claims it, so nothing is produced");
  cmp_int(b, rng.n, 0, "and it costs no deviates");
  // The alias is real and not a typo in this test: the same key, asked of the table directly,
  // is the K0-proton channel.
  cmp_int(b, bert::channel_table(bert::kPionPlus * bert::kPionMinus).valid() ? 1 : 0, 1,
          "pi+ x pi- is a valid table key");
  cmp_int(b, bert::channel_table(bert::kPionPlus * bert::kPionMinus).slot,
          bert::channel_table(bert::kKaonZero * bert::kProton).slot, "and it is K0 p's slot");

  delete ws;
}

// ---------------------------------------------------------------------------------------------
// IsDecayAllowed and the tenth-attempt discard, pinned directly
// ---------------------------------------------------------------------------------------------
//
// Both are gates that produce an EMPTY final state, and an empty final state is indistinguishable
// from a dozen other ways of failing when it is only seen through `collide`. Both are therefore
// also asserted on their own, at the level of the function that implements them.
void check_gates() {
  const int b = new_bucket("FinalStateGates", 0.0);

  // G4VHadDecayAlgorithm::IsDecayAllowed. Three ways to fail and one to pass.
  bert::FinalStateConfig cfg;
  cfg.multiplicity = 2;
  cfg.masses[0] = 0.93827;
  cfg.masses[1] = 0.13957;
  cmp_int(b, bert::fs_is_decay_allowed(cfg, 2.0) ? 1 : 0, 1, "2 GeV pays for p + pi");
  cmp_int(b, bert::fs_is_decay_allowed(cfg, 1.0) ? 1 : 0, 0, "1 GeV does not");
  // The test is `>=`, not `>`: a final state at exactly threshold is allowed and comes out at
  // rest. This is the boundary the retry loop turns on, so it is asserted and not assumed.
  cmp_int(b, bert::fs_is_decay_allowed(cfg, cfg.masses[0] + cfg.masses[1]) ? 1 : 0, 1,
          "exactly at threshold is allowed");
  cmp_int(b, bert::fs_is_decay_allowed(cfg, 0.0) ? 1 : 0, 0, "zero initial mass is not");
  cfg.multiplicity = 1;
  cmp_int(b, bert::fs_is_decay_allowed(cfg, 2.0) ? 1 : 0, 0, "multiplicity 1 is not");

  // And that the gate is what stops it, rather than the generator failing later: with the gate
  // passed, the same two-body state at 1 GeV would have produced a pair at rest, because
  // TwoBodyMomentum clamps a negative PSQ to zero instead of throwing.
  bool bad = false;
  const double pscm = bert::two_body_momentum(1.0, cfg.masses[0], cfg.masses[1], bad);
  cmp(b, pscm, 0.0, "TwoBodyMomentum clamps below threshold");
  cmp_int(b, bad ? 1 : 0, 1, "and reports the clamp as bad kinematics");

  // ------------------------------------------------------------------------------------------
  // The two things the oracle grid provably cannot distinguish, pinned directly instead.
  // ------------------------------------------------------------------------------------------
  //
  // 1. **The tenth-attempt discard.** `generateSCMfinalState`'s exit test is `itry >= itry_max`
  // and its loop is `while (generate && itry++ < itry_max)`, so a success on the tenth pass
  // leaves itry at exactly itry_max and is thrown away with the exhaustions. No case in the
  // oracle grid succeeds on its tenth pass - every empty case there is exhaustion, which leaves
  // itry at 11 - so changing the `>=` to `>` passes all 10,752 of them. The comparison itself is
  // therefore asserted, which is the whole of what the off-by-one is.
  const int bx = new_bucket("RetryExhaustionTest", 0.0);
  cmp_int(bx, bert::fs_retry_exhausted(bert::kFsItryMax + 1) ? 1 : 0, 1,
          "ten failed passes leave itry at itry_max+1 and are discarded");
  cmp_int(bx, bert::fs_retry_exhausted(bert::kFsItryMax) ? 1 : 0, 1,
          "a SUCCESS on the tenth pass leaves itry at itry_max and is discarded too");
  cmp_int(bx, bert::fs_retry_exhausted(bert::kFsItryMax - 1) ? 1 : 0, 0,
          "a success on the ninth pass is kept");
  cmp_int(bx, bert::fs_retry_exhausted(1) ? 1 : 0, 0, "and so is one on the first");

  // 2. **The truncated 1/e.** `oneOverE` is 0.3678794 - seven digits, not `std::exp(-1.)` - and
  // it appears once, scaling the rejection threshold in GenerateCosTheta:
  //
  //     s2   = alf * oneOverE * p0 * inuclRndm();
  //     salf = s1 * alf * G4Exp(-s1 / p0);
  //     if (salf > s2) sinth = s1 / pmod;
  //
  // The ACCEPTED value is `s1 / pmod`, which does not contain oneOverE at all: the constant only
  // moves where the accept/reject boundary sits. Replacing it with the full-precision 1/e shifts
  // that boundary by 1.2e-7 relative, and no deviate in either engine's stream lands inside so
  // narrow a window on this grid - the whole file still agrees. What the constant does is
  // therefore asserted as the constant, the way tests/test_bertini_data.cu pins crossSectionUnits.
  const int bo = new_bucket("OneOverETruncation", 0.0);
  cmp(bo, bert::kFsOneOverE, 0.3678794, "oneOverE is Geant4's seven-digit literal");
  cmp_int(bo, (bert::kFsOneOverE != std::exp(-1.0)) ? 1 : 0, 1,
          "and it is NOT 1/e - the truncation is part of the sampled distribution");
  // The relative size of the truncation, so that the number above is a measurement and not a
  // transcription of a transcription: about 1.1e-7, which is the width of the window in which a
  // deviate would have to land for the difference to be visible in a final state.
  cmp_int(bo, (std::fabs(bert::kFsOneOverE - std::exp(-1.0)) / std::exp(-1.0) < 2.0e-7) ? 1 : 0,
          1, "the truncation is about one part in ten million");
}

// ---------------------------------------------------------------------------------------------
// How an INUCL particle stores a momentum, asserted where the oracle grid cannot reach
// ---------------------------------------------------------------------------------------------
//
// `inucl_store_momentum` is what makes the gamma-nucleon rows of the oracle agree at 2.5e-14
// instead of 2.0e-7 (docs/RISK.md V125), and the campaign for this commit shows exactly how much
// of it the grid can see: removing the round trip ENTIRELY is caught, and so is writing CLHEP's
// `unit()` as three divisions instead of a reciprocal multiply - but removing either of the two
// stores inside `collide`, or the PDG-mass snap, or the zero-mass guard, or the `SetMomentum`
// branch, is not. None of those is dead code; each is a case the 10,752 collisions here do not
// contain. So each is asserted directly.
//
// The reason the grid cannot see them is worth stating, because it is the same reason the whole
// round trip mattered in the first place: rebuilding a four-vector from (direction, Ekin, mass)
// moves its COMPONENTS by about one part in 1e17, and this file compares components. It moves
// `e - m` of a massless particle by one part in 1e9, and nothing computes `e - m` of a PRODUCT.
// It is the BULLET whose `e - m` the next collision takes - which is why the stores are here at
// all, and why they will be load-bearing the moment a product becomes a bullet in the cascader.
void check_storage() {
  // 1e-15 and not 0: two of these compare a reconstructed |p| against the |p| that went in, and
  // the reconstruction is `direction * sqrt(Ekin^2 + 2*m*Ekin)` - the same number to the last
  // bits and not the same bits. Every other assertion in the bucket is a yes/no, which a
  // tolerance cannot soften.
  const int b = new_bucket("InuclMomentumStorage", 1e-15);
  const double mp = bert::inucl_particle_mass(bert::kProton);

  // 1. The amplification itself, measured. A photon boosted into a moving nucleon's rest frame
  // has an `m()` that is the square root of a cancellation; the store removes it. This is the
  // mechanism behind V125, and it is asserted as an INEQUALITY because the size of the residual
  // is rounding and not a number to compare against.
  bert::LorentzConvertor lc;
  lc.bullet = LV(Vec3d{0.0, 0.0, 3.0}, 3.0);                       // a 3 GeV photon
  lc.target = bert::lv_set_vect_m(Vec3d{0.15, -0.08, 0.11}, mp);   // a nucleon with Fermi motion
  const double ekin_raw = bert::lc_kin_energy_in_trs(lc);
  lc.bullet = bert::inucl_store_momentum(lc.bullet, bert::kPhoton);
  lc.target = bert::inucl_store_momentum(lc.target, bert::kProton);
  const double ekin_stored = bert::lc_kin_energy_in_trs(lc);
  const double moved = std::fabs(ekin_stored - ekin_raw) / ekin_stored;
  cmp_int(b, (moved > 1.0e-10) ? 1 : 0, 1,
          "storing the photon moves getKinEnergyInTheTRS by more than 1e-10");
  cmp_int(b, (moved < 1.0e-6) ? 1 : 0, 1, "and by less than 1e-6");

  // 2. The store is not the identity for a massive particle either - it is just a one-ulp
  // change there, which is why only the photon rows of the oracle could see it.
  const LV pm = bert::lv_set_vect_m(Vec3d{0.15, -0.08, 0.11}, mp);
  const LV ps = bert::inucl_store_momentum(pm, bert::kProton);
  const double dp = std::fabs(ps.v.x - pm.v.x) + std::fabs(ps.v.y - pm.v.y) +
                    std::fabs(ps.v.z - pm.v.z) + std::fabs(ps.e - pm.e);
  cmp_int(b, (dp > 0.0) ? 1 : 0, 1, "a nucleon's four-vector changes when it is stored");
  cmp_int(b, (dp < 1.0e-14) ? 1 : 0, 1, "but only in the last bits");

  // 3. The stored kinetic energy is not `e - m` of the four-vector that went in. That is what
  // `G4ParticleLargerEkin` sorts on, and the two orderings differ only for products whose
  // energies agree to the last bits - which none of the 10,752 collisions produces.
  const double ekin_stored_p = bert::inucl_stored_kinetic_energy(pm, bert::kProton);
  cmp_int(b, (ekin_stored_p != pm.e - mp) ? 1 : 0, 1,
          "getKineticEnergy() is not e - m of the four-vector handed in");
  cmp_int(b, (std::fabs(ekin_stored_p - (pm.e - mp)) < 1.0e-15) ? 1 : 0, 1,
          "though it is the same number to fourteen digits");

  // 4. `G4InuclParticle::setMomentum`'s OTHER branch. When the four-vector is more than 1e-5 GeV
  // off the definition's mass shell it calls `SetMomentum`, which keeps the three-momentum and
  // the CURRENT mass and throws the energy away - so the vector comes back ON the mass shell
  // with the momentum it was given. Nothing in a collision is ever that far off shell, so the
  // grid never takes this branch; it is what a cascader handing over a bound nucleon would.
  const LV off_shell = LV(Vec3d{0.15, -0.08, 0.11}, 2.0);   // energy twice what the mass allows
  const LV back = bert::inucl_store_momentum(off_shell, bert::kProton);
  cmp(b, bert::lv_rho(back), bert::lv_rho(off_shell), "SetMomentum keeps |p|");
  cmp(b, back.e, std::sqrt(bert::lv_rho(off_shell) * bert::lv_rho(off_shell) + mp * mp),
      "and puts the energy back on the PDG mass shell");
  cmp_int(b, (std::fabs(back.e - off_shell.e) > 0.5) ? 1 : 0, 1,
          "which is nothing like the energy it was handed");

  // 5. The zero-mass guard, `if (mass2 < EnergyMRA2) dynamicalMass = 0;`. Without it a photon
  // whose boosted four-vector has drifted SPACELIKE takes `sqrt(mass2)` of a negative number.
  // The drift has to be small enough that `setMomentum` still chooses Set4Momentum - |m| under
  // 1e-5 GeV - and that is exactly the regime a boost produces.
  const LV spacelike = LV(Vec3d{0.0, 0.0, 1.0}, 1.0 - 1.0e-12);
  cmp_int(b, (std::fabs(spacelike.mag()) < 1.0e-5) ? 1 : 0, 1,
          "the spacelike photon is still within setMomentum's 1e-5 window");
  cmp_int(b, (spacelike.mag() < 0.0) ? 1 : 0, 1, "and its signed mass is negative");
  const LV guarded = bert::inucl_store_momentum(spacelike, bert::kPhoton);
  cmp_int(b, (guarded.e == guarded.e) ? 1 : 0, 1, "it comes back finite, not NaN");
  cmp(b, guarded.e, bert::lv_rho(guarded), "and exactly null, which is what the guard forces");
}

}  // namespace

int main() {
  check_collide();
  check_refusals();
  check_gates();
  check_storage();

  std::printf("%-34s %12s %14s\n", "bucket", "points", "worst rel");
  long long total = 0;
  for (const Bucket& b : buckets) {
    total += b.n;
    const bool bad = (b.n == 0) || (b.worst > b.tol);
    std::printf("%-34s %12lld %14.3g %s%s\n", b.name, b.n, b.worst, bad ? "FAIL " : "ok   ",
                b.where.c_str());
    if (bad) { ++fails; }
  }
  std::printf("%lld comparisons, %d failures\n", total, fails);
  return (fails == 0) ? 0 : 1;
}
