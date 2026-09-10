// G4Decay and the decay channels of QBBC's unstable transported species, against Geant4's own.
//
// Six blocks, and they are separated because a single worst-case would report whichever of
// them happened to be worse rather than saying which one is wrong:
//
//   1  the tables            exact. Branching ratios, channel kinds, daughter lists and their
//                            ORDER, masses, widths, lifetimes, IsOKWithParentMass, and the
//                            two-body momentum. This is the block that fails if a Geant4
//                            release edits a branching ratio.
//   2  the process           exact. GetMeanFreePath across its gamma = 20 branch and its two
//                            sentinel branches, and GetMeanLifeTime.
//   3  applicability         exact, and it is where the REFUSALS are checked: every species
//                            Geant4 gives a decay table and this package does not must come
//                            back refused by PDG code, not stable and not silently absent.
//                            Also where the pre-assigned-decay path is shown unused.
//   4  the boost             exact. Separated from the sampling so that an error in it cannot
//                            arrive mixed with Monte Carlo noise.
//   5  the samplers          statistical. Per channel and per product slot: the first two
//                            moments of the kinetic energy, its sampled range against the
//                            analytic limit, the angular moments, the pairwise opening
//                            angles, and a 40-bin spectrum.
//   6  the Michel spectrum   analytic, and INDEPENDENT OF GEANT4. G4MuonDecayChannel's
//                            sampling reduces exactly to dN/dx ∝ x^2(3-2x), so the first
//                            three moments of the electron's reduced energy must be 0.7,
//                            8/15 and 3/7. Nothing in this check comes from Geant4, so
//                            agreeing with it cannot be two copies of the same mistake
//                            agreeing with each other.
//
// THE STATISTICAL TOLERANCE IS COMPUTED, NOT CHOSEN. Every sampled comparison is against
// `k * sqrt(se_port^2 + se_oracle^2)` with k = 6 and the standard errors taken from the
// port's own sampled variance (the oracle's variance is the same quantity to within the thing
// being tested). A fixed "1%" would be far too loose for a two-body channel, whose spectrum is
// a delta function and whose standard error is zero, and too tight for nothing. Both runs use
// 400,000 samples, so a 6-sigma band over the ~170 moment comparisons here has a false-failure
// rate around 1e-5.
//
// The delta-function channels are the reason for the additive floor: a two-body decay's
// kinetic energies are fixed, both sides compute them to sixteen digits from the same
// formula, and the tolerance there is 1e-12 relative rather than zero because the port and
// Geant4 evaluate `sqrt(p^2+m^2)-m` after a different number of intermediate roundings.
//
// WHY THE HISTOGRAMS AND NOT ONLY THE MOMENTS. Two of these channels are rejection samplers
// against a shape - the muon's Michel spectrum and KL3's Dalitz density - and KL3's two form
// factor parameters (pLambda, pXi0) are PRIVATE in G4KL3DecayChannel with no accessor, so
// they cannot be dumped and diffed. The only thing that can see them is the shape of the
// pion's spectrum, which is why a per-bin Poisson comparison is here and why substituting
// K0L's pLambda = 0.0300 for K+'s 0.0286 has to fail it.
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "core/rng.cuh"
#include "physics/decay/decay.cuh"
#include "physics/decay/decay_channels.cuh"
#include "physics/decay/decay_products.cuh"
#include "physics/decay/decay_tables.hh"

using namespace g4gpu;
using namespace g4gpu::decay;
using real_t = double;

namespace {

// ------------------------------------------------------------------------------------
// A CSV reader that keys columns by NAME rather than by position, because these files have
// up to twenty-six columns and a sscanf format string with twenty-six conversions is a
// transcription error waiting to happen - and because adding a column to a dump should not
// silently shift every field in the test that reads it.
// ------------------------------------------------------------------------------------
struct Csv {
  std::vector<std::string> header;
  std::vector<std::vector<std::string>> rows;

  int col(const char* name) const {
    for (std::size_t i = 0; i < header.size(); ++i) {
      if (header[i] == name) { return static_cast<int>(i); }
    }
    return -1;
  }
  const std::string& get(std::size_t r, int c) const { return rows[r][c]; }
  double num(std::size_t r, int c) const { return std::atof(rows[r][c].c_str()); }
  int i32(std::size_t r, int c) const { return std::atoi(rows[r][c].c_str()); }
};

std::vector<std::string> split(const char* line) {
  std::vector<std::string> out;
  std::string cur;
  for (const char* p = line; *p != '\0'; ++p) {
    if (*p == ',') {
      out.push_back(cur);
      cur.clear();
    } else if (*p != '\n' && *p != '\r') {
      cur.push_back(*p);
    }
  }
  out.push_back(cur);
  return out;
}

bool load_csv(const std::string& path, Csv& out) {
  FILE* f = std::fopen(path.c_str(), "r");
  if (f == nullptr) { return false; }
  char line[4096];
  if (std::fgets(line, sizeof line, f) == nullptr) {
    std::fclose(f);
    return false;
  }
  out.header = split(line);
  while (std::fgets(line, sizeof line, f) != nullptr) {
    std::vector<std::string> r = split(line);
    if (r.size() == out.header.size()) { out.rows.push_back(r); }
  }
  std::fclose(f);
  return !out.rows.empty();
}

// ------------------------------------------------------------------------------------
struct Cell {
  int n = 0;
  double worst = 0;
  std::string where;
  void note(double dev, const std::string& what) {
    ++n;
    if (dev > worst) {
      worst = dev;
      where = what;
    }
  }
};

int g_fails = 0;

// ------------------------------------------------------------------------------------
// ANTI-VACUITY, RE-RUNNABLE.
//
// `test_decay -perturb <name>` breaks one thing and requires the run to FAIL. It exists
// because docs/HADRONIC_PLAN.md section 6 rule 7 says every assertion is run once with its
// fix removed or its input perturbed - and because doing that by editing a header, building,
// running and reverting leaves nothing behind that says it was done. This way `-perturb list`
// enumerates them and each one is a command anybody can repeat.
//
// The mutations reach the port's tables in place, so that a moved branching ratio or mass
// propagates into every downstream check and the record says what all of them measured. That
// takes one piece of machinery. `particle_rows`, `channel_rows` and `table_rows` return
// pointers to `static const` arrays with constant initialisers - which is what lets nvcc
// accept a function-local static in `__host__ __device__` code at all, and is not negotiable
// - and MSVC puts those in a read-only section. Writing through a const_cast therefore fell
// over with an access violation on six of the fourteen perturbations below and quietly
// succeeded on four, depending on which array the linker had put where; the four that
// "worked" were undefined behaviour that happened to land in .data. So the page is made
// writable first, explicitly, and only for the duration of a perturbed run.
//
// What cannot be reached this way is a code path - a swapped push order, a dropped guard -
// and those were inverted by editing the header, building and reverting, with the measured
// failure recorded in the commit message rather than here.
#ifdef _WIN32
extern "C" __declspec(dllimport) int __stdcall VirtualProtect(void*, std::size_t,
                                                              unsigned long,
                                                              unsigned long*);
#endif

/// Make the bytes holding `n` objects at `p` writable. Returns false where the platform
/// offers no way to ask, in which case a perturbed run says so rather than crashing.
template <typename T>
bool make_writable(const T* p, int n) {
#ifdef _WIN32
  unsigned long old = 0;
  const unsigned long kPageReadWrite = 0x04;
  return VirtualProtect(const_cast<void*>(static_cast<const void*>(p)),
                        sizeof(T) * static_cast<std::size_t>(n), kPageReadWrite, &old) != 0;
#else
  (void)p;
  (void)n;
  return false;
#endif
}

const char* g_perturb = nullptr;

bool perturbing(const char* name) {
  return (g_perturb != nullptr) && (std::strcmp(g_perturb, name) == 0);
}

struct Perturbation {
  const char* name;
  const char* what;
};

const Perturbation kPerturbations[] = {
    {"br", "kaon+ channel 0's branching ratio, +1e-10 relative"},
    {"kind", "kaon+ channel 3 (Ke3) becomes a plain phase-space channel"},
    {"daughter-swap", "kaon+ channel 3's lepton and neutrino exchanged"},
    {"channel-order", "kaon+ channels 3 and 5 exchanged in the table"},
    {"ndaughters", "pi0's Dalitz channel claims two daughters"},
    {"table-count", "kaon+'s table claims five channels"},
    {"mass", "kaon+'s mass, +1e-12 relative"},
    {"width", "pi0's width, +1e-3 relative"},
    {"lifetime", "kaon+'s lifetime, +1e-7 relative"},
    {"stable", "the neutron is flagged stable"},
    {"refuse", "the neutron's table is relabelled as K0L's, a refused species"},
    {"kl3-lambda", "Ke3's form factor slope becomes K0L's 0.0300"},
    {"kl3-xi0", "Ke3's f+(0)/f- becomes K0L's -0.11"},
    {"select-normalise", "channel selection normalises over the OPEN channels only"},
};


/// Applied once, before any comparison. Indices are into the port's own tables and are
/// asserted rather than assumed, so a table edit turns a silent no-op perturbation into a
/// loud one.
void apply_perturbation() {
  if (g_perturb == nullptr) { return; }
  if (!make_writable(particle_rows(), kNumParticles) ||
      !make_writable(channel_rows(), kNumChannels) ||
      !make_writable(table_rows(), kNumTables)) {
    std::printf("  FAIL: cannot make the port's tables writable on this platform; the "
                "perturbation harness needs it\n");
    ++g_fails;
    return;
  }
  auto* prow = const_cast<ParticleRow*>(particle_rows());
  auto* crow = const_cast<ChannelRow*>(channel_rows());
  auto* trow = const_cast<DecayTableRow*>(table_rows());
  const DecayTableRow kp = decay_table_for(kPdgKaonPlus);
  if (kp.count != 6) {
    std::printf("  FAIL: kaon+ no longer has six channels; the perturbations are stale\n");
    ++g_fails;
    return;
  }
  if (perturbing("br")) { crow[kp.first].br *= 1.0 + 1e-10; }
  if (perturbing("kind")) { crow[kp.first + 3].kind = ChannelKind::kPhaseSpace; }
  if (perturbing("daughter-swap")) {
    const int t = crow[kp.first + 3].daughter[1];
    crow[kp.first + 3].daughter[1] = crow[kp.first + 3].daughter[2];
    crow[kp.first + 3].daughter[2] = t;
  }
  if (perturbing("channel-order")) {
    const ChannelRow t = crow[kp.first + 3];
    crow[kp.first + 3] = crow[kp.first + 5];
    crow[kp.first + 5] = t;
  }
  if (perturbing("ndaughters")) {
    const DecayTableRow p0 = decay_table_for(kPdgPiZero);
    crow[p0.first + 1].n_daughters = 2;
  }
  if (perturbing("table-count")) {
    for (int i = 0; i < kNumTables; ++i) {
      if (trow[i].parent_pdg == kPdgKaonPlus) { trow[i].count = 5; }
    }
  }
  if (perturbing("mass")) {
    prow[particle_index(kPdgKaonPlus)].mass *= 1.0 + 1e-12;
  }
  if (perturbing("width")) { prow[particle_index(kPdgPiZero)].width *= 1.0 + 1e-3; }
  if (perturbing("lifetime")) {
    prow[particle_index(kPdgKaonPlus)].lifetime *= 1.0 + 1e-7;
  }
  if (perturbing("stable")) { prow[particle_index(kPdgNeutron)].stable = true; }
  if (perturbing("refuse")) {
    for (int i = 0; i < kNumTables; ++i) {
      if (trow[i].parent_pdg == kPdgNeutron) { trow[i].parent_pdg = 130; }
    }
  }
  if (perturbing("kl3-lambda")) { crow[kp.first + 3].kl3_lambda = 0.0300; }
  if (perturbing("kl3-xi0")) { crow[kp.first + 3].kl3_xi0 = -0.11; }
}

/// The `select-normalise` perturbation, which is a code path and not a datum: the cumulative
/// sum advances only over the channels that pass IsOKWithParentMass, which is what a rewrite
/// of SelectADecayChannel naturally produces and what Geant4 does not do.
template <typename rng_t>
int select_a_decay_channel_normalised(const DecayTableRow& table, double parent_mass,
                                      rng_t& rng) {
  double sum_br = 0.0;
  for (int i = 0; i < table.count; ++i) {
    if (channel_ok_with_parent_mass(channel_rows()[table.first + i], parent_mass)) {
      sum_br += channel_rows()[table.first + i].br;
    }
  }
  if (sum_br <= 0.0) { return -1; }
  const double br = sum_br * static_cast<double>(rng.uniform());
  double sum = 0.0;
  for (int i = 0; i < table.count; ++i) {
    const ChannelRow& ch = channel_rows()[table.first + i];
    if (!channel_ok_with_parent_mass(ch, parent_mass)) { continue; }
    sum += ch.br;
    if (br < sum) { return i; }
  }
  return -1;
}

/// Relative deviation with an absolute floor, so a reference of exactly zero (a massless
/// daughter's mass, a stable particle's -1 lifetime) does not divide by zero.
double reldev(double ours, double ref, double floor_abs = 1e-300) {
  const double d = std::fabs(ours - ref);
  const double s = std::fabs(ref);
  if (s > floor_abs) { return d / s; }
  return (d > floor_abs) ? 1.0 : 0.0;
}

void require(bool ok, const char* fmt, ...) {
  if (ok) { return; }
  ++g_fails;
  va_list ap;
  va_start(ap, fmt);
  std::printf("  FAIL: ");
  std::vprintf(fmt, ap);
  std::printf("\n");
  va_end(ap);
}

/// G4VDecayChannel::GetKinematicsName -> ChannelKind. The mapping is checked rather than
/// assumed: the kind is what selects the sampler, and it is not derivable from the daughter
/// list (pi0 -> gamma e+ e- and K+ -> pi0 e+ nu_e are both three-body with a lepton pair and
/// they are different classes).
bool kind_matches(const std::string& kinematics, ChannelKind k) {
  if (kinematics == "Phase Space") { return k == ChannelKind::kPhaseSpace; }
  if (kinematics == "Muon Decay") { return k == ChannelKind::kMuonDecay; }
  if (kinematics == "KL3 Decay") { return k == ChannelKind::kKL3; }
  if (kinematics == "Dalitz Decay") { return k == ChannelKind::kDalitz; }
  if (kinematics == "Neutron Decay") { return k == ChannelKind::kNeutronBeta; }
  return false;
}

// ------------------------------------------------------------------------------------
// Sampling, in the same shape the oracle accumulates it (ref/dump/dump_decay.cc).
// ------------------------------------------------------------------------------------
constexpr int kHistBins = 40;
constexpr int kSamples = 400000;

struct SlotAcc {
  int pdg = 0;
  int daughter = -1;
  double n = 0;
  double s1 = 0, s2 = 0, s3 = 0, s4 = 0;  ///< sums of e, e^2, e^3, e^4
  double min_e = 1e300, max_e = -1e300;
  double cz1 = 0, cz2 = 0, cz4 = 0;
  double emax = 0;
  double hist[kHistBins] = {0};
};

struct PairAcc {
  double c1 = 0, c2 = 0, c3 = 0, c4 = 0;
};

struct ChannelAcc {
  int nd = 0;
  int accepted = 0;
  SlotAcc slot[DecayProducts<real_t>::kMaxProducts];
  PairAcc pair[DecayProducts<real_t>::kMaxProducts][DecayProducts<real_t>::kMaxProducts];
  /// The four-momentum residual: max over samples of |sum of daughter four-momenta - the
  /// parent's|, relative to the parent mass. Not in the oracle at all - it is a property the
  /// transcription must have whatever Geant4 does, and it is the cheapest check that a
  /// direction was assembled wrongly.
  double worst_p_residual = 0;
  double worst_e_residual = 0;
};

/// Sample one channel N times, with the histogram top edges handed in so that the port and
/// the oracle bin identically. `emax` is checked separately against the port's own
/// channel_slot_max_kinetic_energy, so using the oracle's value here does not weaken that.
void sample_channel(const ChannelRow& ch, int parent_pdg, real_t parent_mass, uint32_t seed,
                    const double* emax, ChannelAcc& acc) {
  acc.nd = ch.n_daughters;
  for (int s = 0; s < acc.nd; ++s) { acc.slot[s].emax = emax[s]; }
  Philox<real_t> rng(seed, 7u, 41u);
  DecayProducts<real_t> out;
  for (int i = 0; i < kSamples; ++i) {
    channel_decay_it<real_t>(ch, parent_pdg, parent_mass, rng, out);
    if (out.status != DecayStatus::kOK || out.n != acc.nd) { continue; }
    ++acc.accepted;
    double sx = 0, sy = 0, sz = 0, se = 0;
    for (int s = 0; s < acc.nd; ++s) {
      SlotAcc& st = acc.slot[s];
      const double e = static_cast<double>(out.p[s].ekin);
      st.pdg = out.p[s].pdg;
      st.daughter = out.p[s].daughter;
      st.n += 1;
      st.s1 += e;
      st.s2 += e * e;
      st.s3 += e * e * e;
      st.s4 += e * e * e * e;
      if (e < st.min_e) { st.min_e = e; }
      if (e > st.max_e) { st.max_e = e; }
      const double cz = static_cast<double>(out.p[s].dir[2]);
      st.cz1 += cz;
      st.cz2 += cz * cz;
      st.cz4 += cz * cz * cz * cz;
      int b = static_cast<int>(e / st.emax * kHistBins);
      if (b < 0) { b = 0; }
      if (b >= kHistBins) { b = kHistBins - 1; }
      st.hist[b] += 1;
      const FourVector<real_t> p4 = out.p[s].four_momentum();
      sx += static_cast<double>(p4.x);
      sy += static_cast<double>(p4.y);
      sz += static_cast<double>(p4.z);
      se += static_cast<double>(p4.t);
    }
    const double pres = std::sqrt(sx * sx + sy * sy + sz * sz) / static_cast<double>(parent_mass);
    const double eres =
        std::fabs(se - static_cast<double>(out.parent_mass)) / static_cast<double>(parent_mass);
    if (pres > acc.worst_p_residual) { acc.worst_p_residual = pres; }
    if (eres > acc.worst_e_residual) { acc.worst_e_residual = eres; }
    for (int a = 0; a < acc.nd; ++a) {
      for (int b = a + 1; b < acc.nd; ++b) {
        double c = 0;
        for (int k = 0; k < 3; ++k) {
          c += static_cast<double>(out.p[a].dir[k]) * static_cast<double>(out.p[b].dir[k]);
        }
        PairAcc& pa = acc.pair[a][b];
        pa.c1 += c;
        pa.c2 += c * c;
        pa.c3 += c * c * c;
        pa.c4 += c * c * c * c;
      }
    }
  }
}

/// The standard error of a mean, from the sampled second moment. Returns zero for a
/// degenerate (delta-function) sample, which is what makes the tolerance collapse to the
/// additive floor for a two-body channel.
double stderr_of_mean(double s1, double s2, double n) {
  if (n < 2) { return 0.0; }
  const double m = s1 / n;
  double var = s2 / n - m * m;
  if (var < 0) { var = 0; }
  return std::sqrt(var / n);
}

/// The comparison every sampled quantity goes through: within k combined standard errors, or
/// within a small relative floor. `se_port` is the port's own; the oracle's is taken to be
/// the same, which is exact to within the discrepancy being tested.
bool within_sigma(double ours, double ref, double se_port, double k, double rel_floor,
                  double* out_sigmas) {
  const double combined = se_port * std::sqrt(2.0);
  const double tol = k * combined + rel_floor * std::fabs(ref) + 1e-300;
  const double d = std::fabs(ours - ref);
  *out_sigmas = (combined > 0) ? d / combined : 0.0;
  return d <= tol;
}

}  // namespace

int main(int argc, char** argv) {
  for (int a = 1; a < argc; ++a) {
    if (std::strcmp(argv[a], "-perturb") == 0 && a + 1 < argc) { g_perturb = argv[++a]; }
  }
  if (g_perturb != nullptr && std::strcmp(g_perturb, "list") == 0) {
    std::printf("perturbations (each must make this test FAIL):\n");
    for (const Perturbation& p : kPerturbations) {
      std::printf("  %-18s %s\n", p.name, p.what);
    }
    return 0;
  }
  if (g_perturb != nullptr) {
    bool known = false;
    for (const Perturbation& p : kPerturbations) {
      known = known || (std::strcmp(p.name, g_perturb) == 0);
    }
    if (!known) {
      std::printf("unknown perturbation '%s'; -perturb list names them\n", g_perturb);
      return 2;
    }
  }

  const char* env = std::getenv("G4GPU_ORACLE");
  const std::string dir = (env != nullptr) ? env : "ref/oracle";

  Csv tables, process, applicable, select, atrest, boost_csv, moments, pairs, hist, closure;
  const char* names[] = {"decay_tables.csv",  "decay_process.csv", "decay_applicable.csv",
                         "decay_select.csv",  "decay_atrest.csv",  "decay_boost.csv",
                         "decay_moments.csv", "decay_pairs.csv",   "decay_hist.csv",
                         "decay_closure.csv"};
  Csv* csvs[] = {&tables, &process, &applicable, &select, &atrest, &boost_csv,
                 &moments, &pairs, &hist, &closure};
  constexpr int kNumCsv = 10;
  for (int i = 0; i < kNumCsv; ++i) {
    if (!load_csv(dir + "/" + names[i], *csvs[i])) {
      std::printf("cannot read %s/%s - run ref/dump/build.bat then ref/oracle/run.bat tables\n",
                  dir.c_str(), names[i]);
      return 1;
    }
  }

  apply_perturbation();
  if (g_perturb != nullptr) {
    std::printf("!! PERTURBED: %s - this run is REQUIRED to fail\n\n", g_perturb);
  }

  // ==================================================================================
  // 1. The tables, exactly.
  // ==================================================================================
  {
    std::printf("== decay tables against the 11.1.1 particle definitions ==\n");
    const int c_pdg = tables.col("parent_pdg");
    const int c_pmass = tables.col("parent_mass_MeV");
    const int c_pwidth = tables.col("parent_width_MeV");
    const int c_plife = tables.col("parent_lifetime_ns");
    const int c_pstable = tables.col("parent_stable");
    const int c_nch = tables.col("n_channels");
    const int c_ch = tables.col("channel");
    const int c_kin = tables.col("kinematics");
    const int c_br = tables.col("br");
    const int c_nd = tables.col("n_daughters");
    const int c_ok = tables.col("is_ok_with_pdg_mass");
    const int c_pmx = tables.col("two_body_pmax_MeV");
    const int c_lam = tables.col("kl3_lambda");
    const int c_xi = tables.col("kl3_xi0");
    const int c_dpdg[3] = {tables.col("d0_pdg"), tables.col("d1_pdg"), tables.col("d2_pdg")};
    const int c_dm[3] = {tables.col("d0_mass_MeV"), tables.col("d1_mass_MeV"),
                         tables.col("d2_mass_MeV")};
    const int c_dw[3] = {tables.col("d0_width_MeV"), tables.col("d1_width_MeV"),
                         tables.col("d2_width_MeV")};

    Cell mass, width, life, br, pmxc, dmass, dwidth, kl3;
    int rows_seen = 0;
    std::vector<int> seen_channels(kNumChannels, 0);

    for (std::size_t r = 0; r < tables.rows.size(); ++r) {
      const int pdg = tables.i32(r, c_pdg);
      const int ich = tables.i32(r, c_ch);
      const DecayTableRow tab = decay_table_for(pdg);
      char buf[256];
      std::snprintf(buf, sizeof buf, "%s channel %d", tables.get(r, 0).c_str(), ich);
      if (tab.count < 1) {
        require(false, "%s: Geant4 has a decay table and the port has no row", buf);
        continue;
      }
      require(tab.count == tables.i32(r, c_nch), "%s: %d channels in the port, %d in Geant4",
              buf, tab.count, tables.i32(r, c_nch));
      if (ich >= tab.count) { continue; }
      ++rows_seen;
      seen_channels[tab.first + ich] = 1;
      const ChannelRow& ch = channel_rows()[tab.first + ich];

      // The parent's own row.
      mass.note(reldev(particle_mass(pdg), tables.num(r, c_pmass)), buf);
      width.note(reldev(particle_width(pdg), tables.num(r, c_pwidth)), buf);
      life.note(reldev(particle_lifetime(pdg), tables.num(r, c_plife)), buf);
      require(particle_is_stable(pdg) == (tables.i32(r, c_pstable) != 0),
              "%s: stable flag differs", buf);

      // The channel: kind, branching ratio, daughters IN ORDER.
      require(kind_matches(tables.get(r, c_kin), ch.kind), "%s: kinematics '%s' is not kind %d",
              buf, tables.get(r, c_kin).c_str(), static_cast<int>(ch.kind));
      br.note(reldev(ch.br, tables.num(r, c_br)), buf);
      require(ch.n_daughters == tables.i32(r, c_nd), "%s: %d daughters, Geant4 has %d", buf,
              ch.n_daughters, tables.i32(r, c_nd));
      for (int d = 0; d < ch.n_daughters && d < 3; ++d) {
        require(ch.daughter[d] == tables.i32(r, c_dpdg[d]),
                "%s: daughter %d is %d in the port and %d in Geant4", buf, d, ch.daughter[d],
                tables.i32(r, c_dpdg[d]));
        dmass.note(reldev(particle_mass(ch.daughter[d]), tables.num(r, c_dm[d])), buf);
        dwidth.note(reldev(particle_width(ch.daughter[d]), tables.num(r, c_dw[d])), buf);
      }

      // IsOKWithParentMass and the two-body kinematic limit.
      const double pm = particle_mass(pdg);
      require(channel_ok_with_parent_mass(ch, pm) == (tables.i32(r, c_ok) != 0),
              "%s: IsOKWithParentMass differs at the PDG mass", buf);
      // AT THE BOUNDARY, where `>=` and `>` part company. The oracle can only be asked about
      // the PDG mass (IsOKWithParentMass takes one), and every channel here is either wide
      // open or wide shut there, so the inequality itself is invisible to that comparison.
      // G4VDecayChannel.cc:621 is `return (parentMass >= sumOfDaughterMassMin);`, and this
      // probes it from both sides of the transcribed threshold: exactly at the sum it is
      // open, one part in 1e15 below it is shut.
      {
        double sum_min = 0.0;
        for (int d = 0; d < ch.n_daughters; ++d) {
          sum_min += particle_mass(ch.daughter[d]) - 2.5 * particle_width(ch.daughter[d]);
        }
        if (ch.n_daughters > 1 && sum_min > 0.0) {
          require(channel_ok_with_parent_mass(ch, sum_min),
                  "%s: shut at a parent mass exactly equal to the daughter sum %.17g - the "
                  "test in G4VDecayChannel is >= and not >", buf, sum_min);
          require(!channel_ok_with_parent_mass(ch, sum_min * (1.0 - 1e-15)),
                  "%s: open below the daughter sum %.17g", buf, sum_min);
        }
      }
      if (ch.n_daughters == 2) {
        const double ours =
            pmx<double>(pm, particle_mass(ch.daughter[0]), particle_mass(ch.daughter[1]));
        pmxc.note(reldev(ours, tables.num(r, c_pmx)), buf);
      }

      // The KL3 form factors, straight off G4KL3DecayChannel's public accessors. This
      // package believed for a while that pLambda and pXi0 were unreachable and built a
      // spectrum-shape argument around it; GetDalitzParameterLambda() and
      // GetDalitzParameterXi() are public inline (G4KL3DecayChannel.hh:52). Zero is required
      // for every other channel kind, so a stray parameter on a phase-space row also fails.
      kl3.note(reldev(ch.kl3_lambda, tables.num(r, c_lam)), buf);
      kl3.note(reldev(ch.kl3_xi0, tables.num(r, c_xi)), buf);
    }

    // The other direction: a channel in the port that Geant4 does not have would otherwise
    // never be looked at.
    for (int i = 0; i < kNumChannels; ++i) {
      require(seen_channels[i] != 0, "port channel index %d has no Geant4 row", i);
    }

    struct {
      const char* name;
      Cell* c;
      double tol;
    } cells[] = {{"parent mass", &mass, 1e-15},   {"parent width", &width, 1e-15},
                 {"parent lifetime", &life, 1e-15}, {"branching ratio", &br, 1e-15},
                 {"two-body Pmx", &pmxc, 1e-14},  {"daughter mass", &dmass, 1e-15},
                 {"daughter width", &dwidth, 1e-15},
                 {"KL3 form factors", &kl3, 1e-15}};
    std::printf("  %d channels compared, %d channels in the port\n", rows_seen, kNumChannels);
    for (const auto& e : cells) {
      std::printf("  %-18s %4d values  worst rel %10.3e   %s\n", e.name, e.c->n, e.c->worst,
                  e.c->where.c_str());
      if (e.c->worst > e.tol) {
        std::printf("  FAIL: %s worst %.3e exceeds %.3e\n", e.name, e.c->worst, e.tol);
        ++g_fails;
      }
    }
    std::printf("\n");
  }

  // ==================================================================================
  // 2. G4Decay::GetMeanFreePath and ::GetMeanLifeTime, exactly.
  // ==================================================================================
  {
    std::printf("== G4Decay interaction lengths ==\n");
    const int c_pdg = process.col("pdg");
    const int c_mass = process.col("mass_MeV");
    const int c_ekin = process.col("ekin_MeV");
    const int c_mfp = process.col("mean_free_path_mm");
    const int c_mlt = process.col("mean_life_ns");
    Cell mfp_low, mfp_high, mlt;
    int n_sentinel = 0;
    for (std::size_t r = 0; r < process.rows.size(); ++r) {
      const int pdg = process.i32(r, c_pdg);
      const double m = process.num(r, c_mass);
      const double e = process.num(r, c_ekin);
      char buf[200];
      std::snprintf(buf, sizeof buf, "%s at %.4g MeV (Ekin/m = %.4g)",
                    process.get(r, 0).c_str(), e, e / m);
      const double ours = in_flight_mean_free_path<double>(pdg, m, e);
      const double ref = process.num(r, c_mfp);
      // The DBL_MIN sentinel has to be exact, not close: a caller distinguishes "decay now"
      // from "a very short step" by comparing against it.
      if (ref == decay_zero_length()) {
        ++n_sentinel;
        require(ours == decay_zero_length(), "%s: expected the DBL_MIN sentinel, got %.17g",
                buf, ours);
      } else {
        // Split at the gamma = 20 handover, because the two sides are different expressions
        // and a single worst-case would not say which one moved.
        Cell& c = (e / m > decay_highest_value()) ? mfp_high : mfp_low;
        c.note(reldev(ours, ref), buf);
      }
      mlt.note(reldev(at_rest_mean_life<double>(pdg), process.num(r, c_mlt)), buf);
    }
    std::printf("  mean free path, Ekin/m <= 20   %4d points  worst rel %10.3e   %s\n",
                mfp_low.n, mfp_low.worst, mfp_low.where.c_str());
    std::printf("  mean free path, Ekin/m >  20   %4d points  worst rel %10.3e   %s\n",
                mfp_high.n, mfp_high.worst, mfp_high.where.c_str());
    std::printf("  mean life at rest              %4d points  worst rel %10.3e   %s\n", mlt.n,
                mlt.worst, mlt.where.c_str());
    std::printf("  DBL_MIN sentinel rows          %4d (exact)\n", n_sentinel);
    require(mfp_low.worst <= 1e-14, "mean free path (slow) worst %.3e", mfp_low.worst);
    require(mfp_high.worst <= 1e-14, "mean free path (fast) worst %.3e", mfp_high.worst);
    require(mlt.worst <= 1e-15, "mean life worst %.3e", mlt.worst);
    require(mfp_high.n > 0 && mfp_low.n > 0 && n_sentinel > 0,
            "the grid did not cover all three branches");
    std::printf("\n");
  }

  // ==================================================================================
  // 3. Applicability, the refusals, and the pre-assigned-decay path.
  // ==================================================================================
  {
    std::printf("== which species G4Decay is registered on, and what this package refuses ==\n");
    const int c_pdg = applicable.col("pdg");
    const int c_mass = applicable.col("mass_MeV");
    const int c_width = applicable.col("width_MeV");
    const int c_life = applicable.col("lifetime_ns");
    const int c_stable = applicable.col("stable");
    const int c_app = applicable.col("is_applicable");
    const int c_hastab = applicable.col("has_table");
    const int c_pat = applicable.col("preassigned_proper_time_ns");
    const int c_pap = applicable.col("has_preassigned_products");
    int n_known = 0, n_refused = 0, n_preassigned = 0;
    Cell known;
    std::string refused_examples;
    for (std::size_t r = 0; r < applicable.rows.size(); ++r) {
      const int pdg = applicable.i32(r, c_pdg);
      const bool g4_applicable = applicable.i32(r, c_app) != 0;
      const bool g4_has_table = applicable.i32(r, c_hastab) != 0;

      // The pre-assigned-decay path: G4Decay has one and this package refuses it. It is
      // unreachable only as long as nothing sets these two fields, so every species is
      // checked rather than one.
      if (applicable.num(r, c_pat) >= 0.0 || applicable.i32(r, c_pap) != 0) {
        ++n_preassigned;
      }

      if (particle_index(pdg) >= 0) {
        ++n_known;
        const std::string what = applicable.get(r, 0);
        known.note(reldev(particle_mass(pdg), applicable.num(r, c_mass)), what);
        known.note(reldev(particle_width(pdg), applicable.num(r, c_width)), what);
        known.note(reldev(particle_lifetime(pdg), applicable.num(r, c_life)), what);
        require(particle_is_stable(pdg) == (applicable.i32(r, c_stable) != 0),
                "%s: stable flag differs", what.c_str());
        require(decay_is_applicable(pdg) == g4_applicable,
                "%s: IsApplicable differs (port %d, Geant4 %d)", what.c_str(),
                decay_is_applicable(pdg) ? 1 : 0, g4_applicable ? 1 : 0);
        continue;
      }

      // Not in the port's particle table. If Geant4 gives it a decay table, the port must
      // refuse it BY CODE with a message - never treat it as stable and never fall through
      // to a similar species.
      if (g4_has_table && g4_applicable) {
        ++n_refused;
        const DecayTableRow tab = decay_table_for(pdg);
        require(tab.count == 0, "%s (%d): port claims %d channels for a refused species",
                applicable.get(r, 0).c_str(), pdg, tab.count);
        if (refused_examples.size() < 200) {
          refused_examples += applicable.get(r, 0);
          refused_examples += " ";
        }
      }
    }

    // THE REFUSAL MESSAGE, CHECKED AGAINST THE GENERIC ONE. This used to read
    // `require(why != nullptr && why[0] != '\0')`, which cannot fail: every arm of
    // decay_refusal_reason's switch returns a string literal, so the assertion was asking
    // whether a literal is non-empty. What is actually worth requiring is that the species
    // the plan names as refused are refused BY NAME - so the message differs from the
    // catch-all - and that is falsifiable by deleting a case from the switch.
    {
      const char* generic = decay_refusal_reason(987654321);
      const int named[] = {130, 310, 311, -311, 3122, 3222, 3212, 3112,
                           3322, 3312, 3334, 15, -15};
      for (int pdg : named) {
        const char* why = decay_refusal_reason(pdg);
        require(std::strcmp(why, generic) != 0,
                "pdg %d falls through to the generic refusal message: the plan names it as a "
                "refused species and the port does not name it back", pdg);
      }
      require(std::strcmp(decay_refusal_reason(987654321), generic) == 0,
              "an unnamed PDG code should get the generic message");
      std::printf("  %d refused species named individually, the rest get \"%s\"\n",
                  static_cast<int>(sizeof(named) / sizeof(named[0])), generic);
    }
    std::printf("  %d species the port knows, worst rel %10.3e   %s\n", n_known, known.worst,
                known.where.c_str());
    std::printf("  %d species Geant4 decays and the port refuses by PDG code\n", n_refused);
    std::printf("    e.g. %s\n", refused_examples.c_str());
    std::printf("  %d of %d species carry a pre-assigned decay (expected 0)\n", n_preassigned,
                static_cast<int>(applicable.rows.size()));
    require(known.worst <= 1e-15, "known-species properties worst %.3e", known.worst);
    require(n_refused > 0, "no refused species in the oracle - the refusal path is untested");
    require(n_preassigned == 0,
            "%d species carry a pre-assigned decay: G4Decay's pre-assigned path is REACHABLE "
            "and this package refuses it",
            n_preassigned);

    // The triton: flagged stable AND carrying a 17.774 year lifetime, so IsApplicable accepts
    // it while nothing else in G4Decay will act on it. It is not in the port's table (it does
    // not decay), and the point of naming it here is that "IsApplicable is true" must not be
    // read as "this species decays".
    for (std::size_t r = 0; r < applicable.rows.size(); ++r) {
      if (applicable.get(r, 0) != "triton") { continue; }
      require(applicable.i32(r, c_app) == 1 && applicable.i32(r, c_stable) == 1 &&
                  applicable.num(r, c_life) > 0.0,
              "triton is no longer the stable-flag-versus-lifetime counter-example");
      std::printf("  triton: applicable=%d stable=%d lifetime=%.4g ns - the flag, not the "
                  "lifetime, is what stops it decaying\n",
                  applicable.i32(r, c_app), applicable.i32(r, c_stable),
                  applicable.num(r, c_life));
    }
    std::printf("\n");
  }

  // ==================================================================================
  // 3b. G4DecayTable::SelectADecayChannel, sampled - including at masses that close
  //     channels, which is the only place its two peculiarities are observable.
  // ==================================================================================
  {
    std::printf("== G4DecayTable::SelectADecayChannel, %s ==\n",
                "at the PDG mass and at masses that close channels");
    const int s_pdg = select.col("parent_pdg");
    const int s_frac = select.col("mass_fraction");
    const int s_mass = select.col("parent_mass_MeV");
    const int s_nch = select.col("n_channels");
    const int s_samples = select.col("samples");
    const int s_null = select.col("n_null");
    const int s_ch = select.col("channel");
    const int s_ok = select.col("is_ok");
    const int s_count = select.col("count");

    Cell freq;
    int cases = 0, null_cases = 0, starved = 0, starved_agreed = 0;
    // Walk the oracle's (species, mass) cases. Each case is a contiguous run of channel rows.
    std::size_t r = 0;
    while (r < select.rows.size()) {
      const int pdg = select.i32(r, s_pdg);
      const double frac = select.num(r, s_frac);
      const double pm = select.num(r, s_mass);
      const int nch = select.i32(r, s_nch);
      const int draws = select.i32(r, s_samples);
      const int g4_null = select.i32(r, s_null);
      std::vector<int> g4_count(static_cast<std::size_t>(nch), 0);
      std::vector<int> g4_ok(static_cast<std::size_t>(nch), 0);
      for (int c = 0; c < nch && r < select.rows.size(); ++c, ++r) {
        const int idx = select.i32(r, s_ch);
        if (idx < 0 || idx >= nch) { continue; }
        g4_count[static_cast<std::size_t>(idx)] = select.i32(r, s_count);
        g4_ok[static_cast<std::size_t>(idx)] = select.i32(r, s_ok);
      }
      const DecayTableRow tab = decay_table_for(pdg);
      char who[100];
      std::snprintf(who, sizeof who, "pdg %d at %.2f of its mass (%.3f MeV)", pdg, frac, pm);
      require(tab.count == nch, "%s: %d channels in the port, %d in Geant4", who, tab.count,
              nch);
      if (tab.count != nch) { continue; }
      ++cases;

      // The port, sampled the same number of times. Seeded per case so that adding a case
      // does not move an existing one.
      std::vector<int> count(static_cast<std::size_t>(nch), 0);
      int nulls = 0;
      Philox<real_t> rng(static_cast<uint32_t>(7919 * (pdg + 4000) + int(frac * 100)), 11u,
                         23u);
      for (int i = 0; i < draws; ++i) {
        const int ich = perturbing("select-normalise")
                            ? select_a_decay_channel_normalised(tab, pm, rng)
                            : select_a_decay_channel<real_t>(tab, static_cast<real_t>(pm), rng);
        if (ich < 0) {
          ++nulls;
        } else if (ich < nch) {
          ++count[static_cast<std::size_t>(ich)];
        }
      }
      require(nulls == g4_null, "%s: %d null selections in the port, %d in Geant4", who, nulls,
              g4_null);
      if (g4_null == draws) { ++null_cases; }

      for (int c = 0; c < nch; ++c) {
        const double n = draws;
        const double p_g4 = g4_count[static_cast<std::size_t>(c)] / n;
        const double p_port = count[static_cast<std::size_t>(c)] / n;
        char buf[160];
        std::snprintf(buf, sizeof buf, "%s channel %d", who, c);
        // A CHANNEL THAT IS OPEN AND NEVER SELECTED is the finding this block exists for,
        // and zero is not a frequency to compare with a sigma - it is an exact claim, so it
        // is asserted as one on both sides.
        if (g4_ok[static_cast<std::size_t>(c)] != 0 &&
            g4_count[static_cast<std::size_t>(c)] == 0 && g4_null != draws) {
          ++starved;
          if (count[static_cast<std::size_t>(c)] == 0) { ++starved_agreed; }
          require(count[static_cast<std::size_t>(c)] == 0,
                  "%s: Geant4 never selects this OPEN channel (its cumulative bound is past "
                  "sumBR) and the port selected it %d times",
                  buf, count[static_cast<std::size_t>(c)]);
          continue;
        }
        // Otherwise a binomial comparison of two independent runs of `draws` draws.
        const double pbar = 0.5 * (p_g4 + p_port);
        const double se = std::sqrt(2.0 * pbar * (1.0 - pbar) / n);
        const double sig = (se > 0) ? std::fabs(p_port - p_g4) / se : 0.0;
        freq.note(sig, buf);
        require(sig <= 6.0, "%s: selected %.5f of the time, Geant4 %.5f (%.1f sigma)", buf,
                p_port, p_g4, sig);
      }
    }
    std::printf("  %d (species, mass) cases, %d of them with every channel shut\n", cases,
                null_cases);
    std::printf("  selection frequency    %4d values  worst %6.2f sigma   %s\n", freq.n,
                freq.worst, freq.where.c_str());
    std::printf("  open channels never selected: %d, port agrees on %d\n", starved,
                starved_agreed);
    require(cases >= 40, "only %d selection cases", cases);
    require(null_cases > 0, "no case where every channel is shut: kNoChannel is untested here");
    require(starved > 0,
            "no case where an OPEN channel is never selected - the cumulative sum running "
            "past sumBR is untested, and it is the whole reason this block samples at "
            "reduced masses");
    std::printf("\n");
  }

  // ==================================================================================
  // 3d. G4KL3DecayChannel::DalitzDensity on a grid, exactly.
  // ==================================================================================
  {
    std::printf("== G4KL3DecayChannel::DalitzDensity over the Dalitz plane ==\n");
    // The sampled pion spectrum CANNOT see the form factors, which was measured rather than
    // supposed: substituting K0L's pLambda = 0.0300 for K+'s 0.0286, and K0L's pXi0 = -0.11
    // for -0.35, left every assertion in this file passing. Ke3's pXi0 multiplies
    // m_e^2 = 0.261 MeV^2 against a coefficient of order m_K^3 and is invisible at any sample
    // size. So the density itself is compared, point by point, to machine precision - which
    // pins both parameters and the whole Chounet expression rather than a shape they barely
    // move.
    Csv dalitz;
    if (!load_csv(dir + "/decay_dalitz.csv", dalitz)) {
      std::printf("  FAIL: cannot read decay_dalitz.csv\n");
      ++g_fails;
    } else {
      const int d_par = dalitz.col("parent");
      const int d_lep = dalitz.col("lepton");
      const int d_lam = dalitz.col("lambda");
      const int d_xi = dalitz.col("xi0");
      const int d_mk = dalitz.col("mass_k_MeV");
      const int d_mpi = dalitz.col("mass_pi_MeV");
      const int d_ml = dalitz.col("mass_l_MeV");
      const int d_mnu = dalitz.col("mass_nu_MeV");
      const int d_epi = dalitz.col("epi_MeV");
      const int d_el = dalitz.col("el_MeV");
      const int d_enu = dalitz.col("enu_MeV");
      const int d_rho = dalitz.col("density");
      Cell rho;
      double worst_abs = 0;
      for (std::size_t r = 0; r < dalitz.rows.size(); ++r) {
        const double ours = kl3_dalitz_density<double>(
            dalitz.num(r, d_mk), dalitz.num(r, d_epi), dalitz.num(r, d_el),
            dalitz.num(r, d_enu), dalitz.num(r, d_mpi), dalitz.num(r, d_ml),
            dalitz.num(r, d_mnu), dalitz.num(r, d_lam), dalitz.num(r, d_xi));
        const double ref = dalitz.num(r, d_rho);
        char buf[200];
        std::snprintf(buf, sizeof buf, "%s -> pi0 %s nu at Epi %.4g El %.4g Enu %.4g",
                      dalitz.get(r, d_par).c_str(), dalitz.get(r, d_lep).c_str(),
                      dalitz.num(r, d_epi), dalitz.num(r, d_el), dalitz.num(r, d_enu));
        rho.note(reldev(ours, ref), buf);
        if (std::fabs(ours - ref) > worst_abs) { worst_abs = std::fabs(ours - ref); }
        // The parameters the density was evaluated with must be the ones the port's channel
        // row carries, or the grid would be checking the formula against itself with
        // Geant4's numbers handed in.
        const int kpdg = (dalitz.get(r, d_par) == "kaon+") ? kPdgKaonPlus : kPdgKaonMinus;
        const bool is_mu = dalitz.get(r, d_lep) == "mu+" || dalitz.get(r, d_lep) == "mu-";
        const DecayTableRow tab = decay_table_for(kpdg);
        int found = -1;
        for (int i = 0; i < tab.count; ++i) {
          const ChannelRow& c = channel_rows()[tab.first + i];
          if (c.kind != ChannelKind::kKL3) { continue; }
          const int lep = c.daughter[1];
          const bool c_is_mu = (lep == kPdgMuPlus || lep == kPdgMuMinus);
          if (c_is_mu == is_mu) { found = i; }
        }
        require(found >= 0, "%s: no KL3 channel in the port's table for this lepton", buf);
        if (found >= 0) {
          const ChannelRow& c = channel_rows()[tab.first + found];
          rho.note(reldev(c.kl3_lambda, dalitz.num(r, d_lam)), buf);
          rho.note(reldev(c.kl3_xi0, dalitz.num(r, d_xi)), buf);
        }
      }
      std::printf("  Dalitz density      %5d values  worst rel %10.3e (worst abs %.3e)   %s\n",
                  rho.n, rho.worst, worst_abs, rho.where.c_str());
      require(rho.n >= 1500, "only %d Dalitz points compared", rho.n);
      require(rho.worst <= 1e-14, "Dalitz density worst %.3e", rho.worst);
    }
    std::printf("\n");
  }

  // ==================================================================================
  // 3c. The at-rest queue QBBC actually built, and what it does to this process.
  // ==================================================================================
  {
    std::printf("== the at-rest competition, measured ==\n");
    const int a_pdg = atrest.col("pdg");
    const int a_n = atrest.col("n_atrest");
    const int a_idx = atrest.col("index");
    const int a_name = atrest.col("process_name");
    const int a_len = atrest.col("at_rest_length_ns");

    int n_bertini = 0, n_fritiof = 0, n_mucap = 0, n_decay = 0;
    for (std::size_t r = 0; r < atrest.rows.size(); ++r) {
      const std::string& pname = atrest.get(r, a_name);
      if (pname == "hBertiniCaptureAtRest") { ++n_bertini; }
      if (pname == "hFritiofCaptureAtRest") { ++n_fritiof; }
      if (pname == "muMinusCaptureAtRest") { ++n_mucap; }
      if (pname == "Decay") { ++n_decay; }
      // EVERY at-rest process that is not G4Decay is a G4HadronStoppingProcess (or e+
      // annihilation), and every one of them answers a hard zero. That is the pre-emption.
      if (pname != "Decay" && !pname.empty()) {
        require(atrest.num(r, a_len) == 0.0,
                "%s's at-rest length is %.17g, not zero - it would be a race after all",
                pname.c_str(), atrest.num(r, a_len));
      }
    }
    std::printf("  at-rest processes registered: %d Decay, %d hBertiniCaptureAtRest, "
                "%d hFritiofCaptureAtRest, %d muMinusCaptureAtRest\n",
                n_decay, n_bertini, n_fritiof, n_mucap);
    // G4StoppingPhysics::ConstructProcess names its species explicitly: Bertini on pi-,
    // kaon-, Sigma-, Xi-, Omega-; muon capture on mu-. Counting them pins the physics list.
    require(n_bertini == 5, "%d species carry hBertiniCaptureAtRest, expected 5 "
                            "(pi-, kaon-, Sigma-, Xi-, Omega-)", n_bertini);
    require(n_mucap == 1, "%d species carry muMinusCaptureAtRest, expected 1 (mu-)", n_mucap);

    // Now the port's hook, species by species, against that queue.
    for (int t = 0; t < kNumTables; ++t) {
      const int pdg = table_rows()[t].parent_pdg;
      std::string competitor;
      double decay_len = -1.0;
      bool competitor_first = false;
      bool found = false;
      for (std::size_t r = 0; r < atrest.rows.size(); ++r) {
        if (atrest.i32(r, a_pdg) != pdg) { continue; }
        found = true;
        const std::string& pname = atrest.get(r, a_name);
        if (pname == "Decay") {
          decay_len = atrest.num(r, a_len);
        } else if (!pname.empty()) {
          competitor = pname;
          competitor_first = (atrest.i32(r, a_idx) == 0);
        }
      }
      const AtRestCompetitor ours = decay_at_rest_competitor(pdg);
      AtRestCompetitor theirs = AtRestCompetitor::kNone;
      if (competitor == "hBertiniCaptureAtRest") {
        theirs = AtRestCompetitor::kHadronicAbsorptionBertini;
      } else if (competitor == "muMinusCaptureAtRest") {
        theirs = AtRestCompetitor::kMuonMinusCapture;
      }
      require(found, "pdg %d has no row in decay_atrest.csv: QBBC built it no process "
                     "manager, so the port's hook is unverified for it", pdg);
      require(ours == theirs,
              "pdg %d: the port names competitor %d, Geant4's at-rest queue has '%s'", pdg,
              static_cast<int>(ours), competitor.empty() ? "(only Decay)" : competitor.c_str());
      require(decay_len > 0.0,
              "pdg %d: G4Decay's own at-rest length is %.17g, so 'zero pre-empts it' is not a "
              "statement about anything", pdg, decay_len);
      if (!competitor.empty()) {
        require(competitor_first,
                "pdg %d: '%s' is not first in the at-rest vector; the pre-emption argument "
                "rests on the zero, but the order is what makes it visible", pdg,
                competitor.c_str());
        std::printf("  pdg %-5d %-22s length 0 vs Decay's %.4g ns - decay pre-empted\n", pdg,
                    competitor.c_str(), decay_len);
      }
      // The port's own at-rest length, for the number the wiring package would compare.
      const double ours_len = at_rest_interaction_length<double>(pdg, 1.0);
      require(ours_len > 0.0, "pdg %d: the port's at-rest length is not positive", pdg);
      require(reldev(ours_len, particle_lifetime(pdg)) <= 1e-15,
              "pdg %d: at_rest_interaction_length(1.0) is %.17g, not the lifetime %.17g", pdg,
              ours_len, particle_lifetime(pdg));
    }
    std::printf("  pi+, kaon+, mu+, pi0, neutron: no at-rest process but Decay, so the "
                "at-rest branch is theirs alone\n");
    std::printf("\n");
  }

  // ==================================================================================
  // 4. G4DecayProducts::Boost, exactly.
  // ==================================================================================
  {
    std::printf("== G4DecayProducts::Boost on a fixed rest-frame product set ==\n");
    const int c_pmass = boost_csv.col("parent_mass_MeV");
    const int c_pekin = boost_csv.col("parent_ekin_MeV");
    const int c_dx = boost_csv.col("dirx");
    const int c_dy = boost_csv.col("diry");
    const int c_dz = boost_csv.col("dirz");
    const int c_slot = boost_csv.col("slot");
    const int c_px = boost_csv.col("px_MeV");
    const int c_py = boost_csv.col("py_MeV");
    const int c_pz = boost_csv.col("pz_MeV");
    const int c_e = boost_csv.col("e_MeV");
    const int c_m = boost_csv.col("mass_MeV");

    // The same rest-frame configuration the dump builds: pi+ -> mu+ nu_mu, muon along
    // (2,-3,6)/7 which is a unit vector exactly, so the input carries no rounding of its own.
    const double pm = particle_mass(kPdgPiPlus);
    const double m0 = particle_mass(kPdgMuPlus);
    const double p0 = pmx<double>(pm, m0, 0.0);
    const double rd[3] = {2.0 / 7.0, -3.0 / 7.0, 6.0 / 7.0};

    Cell mom, ene, mas;
    std::size_t r = 0;
    while (r < boost_csv.rows.size()) {
      const double ekin = boost_csv.num(r, c_pekin);
      const double dir[3] = {boost_csv.num(r, c_dx), boost_csv.num(r, c_dy),
                             boost_csv.num(r, c_dz)};
      DecayProducts<double> products;
      products.parent_mass = boost_csv.num(r, c_pmass);
      products.push(kPdgMuPlus, 0, m0, std::sqrt(p0 * p0 + m0 * m0) - m0, rd[0], rd[1], rd[2]);
      products.push(kPdgNuMu, 1, 0.0, p0, -rd[0], -rd[1], -rd[2]);
      const double dirv[3] = {dir[0], dir[1], dir[2]};
      boost_products<double>(products, ekin + products.parent_mass, dirv);
      for (int s = 0; s < 2 && r < boost_csv.rows.size(); ++s, ++r) {
        if (boost_csv.i32(r, c_slot) != s) { break; }
        const FourVector<double> p4 = products.p[s].four_momentum();
        char buf[220];
        std::snprintf(buf, sizeof buf, "ekin %.4g dir (%.3f,%.3f,%.3f) slot %d", ekin, dir[0],
                      dir[1], dir[2], s);
        // The momentum components are compared against the momentum MAGNITUDE, not against
        // themselves: a component that is legitimately near zero (the transverse part after a
        // boost along it) would otherwise divide by nothing.
        const double scale = std::sqrt(boost_csv.num(r, c_px) * boost_csv.num(r, c_px) +
                                       boost_csv.num(r, c_py) * boost_csv.num(r, c_py) +
                                       boost_csv.num(r, c_pz) * boost_csv.num(r, c_pz));
        mom.note(std::fabs(p4.x - boost_csv.num(r, c_px)) / (scale + 1e-300), buf);
        mom.note(std::fabs(p4.y - boost_csv.num(r, c_py)) / (scale + 1e-300), buf);
        mom.note(std::fabs(p4.z - boost_csv.num(r, c_pz)) / (scale + 1e-300), buf);
        ene.note(reldev(p4.t, boost_csv.num(r, c_e)), buf);
        mas.note(reldev(products.p[s].mass, boost_csv.num(r, c_m)), buf);
      }
    }
    std::printf("  boosted momentum   %4d values  worst rel %10.3e   %s\n", mom.n, mom.worst,
                mom.where.c_str());
    std::printf("  boosted energy     %4d values  worst rel %10.3e   %s\n", ene.n, ene.worst,
                ene.where.c_str());
    std::printf("  dynamical mass     %4d values  worst rel %10.3e   %s\n", mas.n, mas.worst,
                mas.where.c_str());
    require(mom.n >= 40, "only %d boosted momenta compared", mom.n);
    require(mom.worst <= 1e-14, "boosted momentum worst %.3e", mom.worst);
    require(ene.worst <= 1e-14, "boosted energy worst %.3e", ene.worst);
    require(mas.worst <= 1e-15, "dynamical mass worst %.3e", mas.worst);
    std::printf("\n");
  }

  // ==================================================================================
  // 5. The samplers, statistically.
  // ==================================================================================
  // The synthetic four-body channel, which exercises
  // G4PhaseSpaceDecayChannel::ManyBodyDecayIt - the `default:` arm that no real table
  // reaches. Built here to match the dump's `G4PhaseSpaceDecayChannel("kaon+",1.0,4,"e+",
  // "nu_e","gamma","gamma")` exactly.
  const ChannelRow kFourBody{ChannelKind::kPhaseSpace,
                             1.0,
                             4,
                             {kPdgPositron, kPdgNuE, kPdgGamma, kPdgGamma, 0},
                             0.0,
                             0.0};
  {
    std::printf("== sampled final states, %d samples per channel ==\n", kSamples);
    const int m_pdg = moments.col("parent_pdg");
    const int m_ch = moments.col("channel");
    const int m_slot = moments.col("slot");
    const int m_daughter = moments.col("daughter");
    const int m_dpdg = moments.col("daughter_pdg");
    const int m_limit = moments.col("ekin_limit");
    const int m_e1 = moments.col("mean_ekin");
    const int m_e2 = moments.col("mean_ekin2");
    const int m_min = moments.col("min_ekin");
    const int m_max = moments.col("max_ekin");
    const int m_cz = moments.col("mean_cosz");
    const int m_cz2 = moments.col("mean_cos2z");
    const int p_pdg = pairs.col("parent_pdg");
    const int p_ch = pairs.col("channel");
    const int p_i = pairs.col("slot_i");
    const int p_j = pairs.col("slot_j");
    const int p_c1 = pairs.col("mean_cos");
    const int p_c2 = pairs.col("mean_cos2");
    const int h_pdg = hist.col("parent_pdg");
    const int h_ch = hist.col("channel");
    const int h_slot = hist.col("slot");
    const int h_bin = hist.col("bin");
    const int h_cnt = hist.col("count");
    const int h_emax = hist.col("emax_MeV");

    // Walk the port's own tables so that a channel with no oracle rows is a failure rather
    // than a silent skip. The synthetic four-body channel is appended as (321, 99).
    struct Job {
      int parent_pdg;
      int channel;
      const ChannelRow* ch;
      double parent_mass;
    };
    std::vector<Job> jobs;
    for (int t = 0; t < kNumTables; ++t) {
      for (int i = 0; i < table_rows()[t].count; ++i) {
        jobs.push_back(Job{table_rows()[t].parent_pdg, i, &channel_rows()[table_rows()[t].first + i],
                           particle_mass(table_rows()[t].parent_pdg)});
      }
    }
    jobs.push_back(Job{kPdgKaonPlus, 99, &kFourBody, particle_mass(kPdgKaonPlus)});

    Cell mean1, mean2, minc, maxc, cosz, cos2z, paircos, paircos2, limit;
    int hist_bins = 0, hist_bad = 0;
    double worst_hist = 0;
    std::string worst_hist_where;
    // Four-momentum closure, against GEANT4'S OWN closure per channel rather than against a
    // chosen tolerance. Two channels do not close anywhere near double precision and the
    // reason is in each case the channel's own arithmetic, not the port's:
    //
    //   G4MuonDecayChannel "neglects ... electron mass" (its own comment): the electron's
    //   momentum is sqrt(T^2 + 2 T m_e) while the three DIRECTIONS are laid out for a
    //   massless electron, so the energies miss by exactly m_e and the momenta by up to it -
    //   4.8e-3 of the muon mass. Geant4 measures 4.8129408e-3 and the port 4.8129408e-3.
    //
    //   ManyBodyDecayIt boosts each nested subsystem by beta = p/sqrt(p^2 + m_sub^2), and a
    //   subsystem of two photons has m_sub small enough that beta is within 1e-11 of one -
    //   at which point CLHEP's `1/sqrt(1-b2)` has lost most of its digits. Geant4's residual
    //   is 2.9e-7 of the parent mass and the port's is 5.9e-7.
    //
    // Picking a tolerance for those would have meant either a 5e-3 bound on every channel -
    // which would hide a real closure error in the other eighteen - or a per-channel guess.
    // Dumping Geant4's own answer makes it a comparison: the port must lose precision where
    // Geant4 loses it and nowhere else. A factor of 30 on the maximum over 400,000 samples of
    // a rounding-dominated quantity, plus a 1e-15 floor for the channels whose residual is
    // exactly zero.
    Cell closure_p, closure_e;
    double worst_p_res[2] = {0, 0};
    double worst_e_res[2] = {0, 0};
    std::string worst_res_where[2];
    int channels_done = 0;

    for (const Job& job : jobs) {
      // Find the oracle's rows for this channel, and its histogram top edges.
      double emax[DecayProducts<real_t>::kMaxProducts] = {0, 0, 0, 0, 0};
      int nslots = 0;
      std::vector<std::size_t> mrows;
      for (std::size_t r = 0; r < moments.rows.size(); ++r) {
        if (moments.i32(r, m_pdg) != job.parent_pdg || moments.i32(r, m_ch) != job.channel) {
          continue;
        }
        const int s = moments.i32(r, m_slot);
        if (s >= 0 && s < DecayProducts<real_t>::kMaxProducts) {
          emax[s] = moments.num(r, m_limit);
          if (s + 1 > nslots) { nslots = s + 1; }
        }
        mrows.push_back(r);
      }
      char who[80];
      std::snprintf(who, sizeof who, "pdg %d channel %d", job.parent_pdg, job.channel);
      require(nslots == job.ch->n_daughters, "%s: %d oracle slots, %d daughters in the port",
              who, nslots, job.ch->n_daughters);
      if (nslots != job.ch->n_daughters) { continue; }
      ++channels_done;

      ChannelAcc acc;
      const uint32_t seed =
          static_cast<uint32_t>(31 * (job.parent_pdg + 4000) + job.channel + 1);
      sample_channel(*job.ch, job.parent_pdg, job.parent_mass, seed, emax, acc);
      require(acc.accepted == kSamples, "%s: only %d of %d samples produced products", who,
              acc.accepted, kSamples);
      const int rk = (job.ch->kind == ChannelKind::kMuonDecay) ? 1 : 0;
      if (acc.worst_p_residual > worst_p_res[rk]) {
        worst_p_res[rk] = acc.worst_p_residual;
        worst_res_where[rk] = who;
      }
      if (acc.worst_e_residual > worst_e_res[rk]) { worst_e_res[rk] = acc.worst_e_residual; }

      // Against Geant4's own closure for this channel.
      for (std::size_t r = 0; r < closure.rows.size(); ++r) {
        if (closure.i32(r, closure.col("parent_pdg")) != job.parent_pdg ||
            closure.i32(r, closure.col("channel")) != job.channel) {
          continue;
        }
        const double gp = closure.num(r, closure.col("worst_p_residual"));
        const double ge = closure.num(r, closure.col("worst_e_residual"));
        char buf[200];
        std::snprintf(buf, sizeof buf, "%s (%s): |sum p|/M %.4e vs G4 %.4e, |sum E - M|/M "
                                       "%.4e vs G4 %.4e",
                      who, closure.get(r, closure.col("kinematics")).c_str(),
                      acc.worst_p_residual, gp, acc.worst_e_residual, ge);
        closure_p.note(acc.worst_p_residual / (30.0 * gp + 1e-15), buf);
        closure_e.note(acc.worst_e_residual / (30.0 * ge + 1e-15), buf);
        require(acc.worst_p_residual <= 30.0 * gp + 1e-15,
                "momentum closure: %s", buf);
        require(acc.worst_e_residual <= 30.0 * ge + 1e-15, "energy closure: %s", buf);
      }

      for (std::size_t r : mrows) {
        const int s = moments.i32(r, m_slot);
        const SlotAcc& st = acc.slot[s];
        const double n = st.n;
        char buf[220];
        std::snprintf(buf, sizeof buf, "%s slot %d (pdg %d)", who, s, st.pdg);

        // Exact: which daughter is in which slot, and that daughter's kinematic limit. The
        // slot order is not the daughter order for a three-body phase space or a KL3 decay
        // (both push 0, 2, 1) and getting it wrong is a real, silent physics error - the
        // neutrino would be created with the lepton's spectrum.
        require(st.pdg == moments.i32(r, m_dpdg), "%s: slot holds pdg %d, Geant4 has %d", buf,
                st.pdg, moments.i32(r, m_dpdg));
        // The daughter index has to agree - EXCEPT when the two daughters in question are the
        // same species, in which case the labelling is unobservable and the oracle's own
        // reading of it is arbitrary. The dump recovers the mapping by matching product
        // definitions against the daughter list greedily, so for K+ -> pi+ pi0 pi0 it calls
        // the first pi0 it sees daughter 1 while the port calls it daughter 2. Nothing
        // measurable distinguishes them: same mass, same width, same limit, same spectrum.
        // K+ -> pi+ pi+ pi- is the case where it IS observable (the pi- pins the middle slot)
        // and there the indices must match, which they do.
        const int od = moments.i32(r, m_daughter);
        require(st.daughter == od || job.ch->daughter[st.daughter] == job.ch->daughter[od],
                "%s: slot holds daughter %d, Geant4 has %d", buf, st.daughter, od);
        limit.note(reldev(channel_slot_max_kinetic_energy<double>(*job.ch, job.parent_mass, od),
                          moments.num(r, m_limit)),
                   buf);

        double sig = 0;
        const double se1 = stderr_of_mean(st.s1, st.s2, n);
        if (!within_sigma(st.s1 / n, moments.num(r, m_e1), se1, 6.0, 1e-12, &sig)) {
          require(false, "%s: <Ekin> %.10g vs %.10g (%.1f sigma)", buf, st.s1 / n,
                  moments.num(r, m_e1), sig);
        }
        mean1.note(sig, buf);
        const double se2 = stderr_of_mean(st.s2, st.s4, n);
        if (!within_sigma(st.s2 / n, moments.num(r, m_e2), se2, 6.0, 1e-12, &sig)) {
          require(false, "%s: <Ekin^2> %.10g vs %.10g (%.1f sigma)", buf, st.s2 / n,
                  moments.num(r, m_e2), sig);
        }
        mean2.note(sig, buf);

        // The support of the spectrum, pinned from BOTH SIDES against the analytic limit
        // rather than against the oracle's own extreme:
        //
        //   no sample may exceed the limit                  - catches a sampler that shares
        //                                                     the released energy wrongly
        //   the largest sample must reach within 5% of it   - catches a sampler that stops
        //                                                     short (the worst gap over every
        //                                                     slot here is 0.9%, the neutron's
        //                                                     electron)
        //   a two-body channel's min must equal its max     - its energies are fixed
        //
        // Comparing the port's extreme directly against the ORACLE's was tried and taken out.
        // An extreme is an order statistic: for a spectrum vanishing like (L-x)^2 at its
        // endpoint the gap scales as (u/N)^(1/3) with u exponential, so the ratio of two
        // independent runs' gaps is (u1/u2)^(1/3) and lands a factor of 8 apart once every
        // 513 comparisons - about one run in ten across the 54 slots here, and it was measured
        // at 90 on the four-body channel on the first attempt. The property it was reaching
        // for is the shape of the spectrum near its edges, and the 40-bin Poisson comparison
        // below tests that far more sharply and without the heavy tail. The ratios are still
        // printed, because a gross disagreement in them is worth seeing.
        const double lim = moments.num(r, m_limit);
        require(st.max_e <= lim * (1 + 1e-12),
                "%s: sampled max %.10g exceeds the kinematic limit %.10g", buf, st.max_e, lim);
        require(lim - st.max_e <= 0.05 * lim,
                "%s: sampled max %.10g does not reach within 5%% of the limit %.10g", buf,
                st.max_e, lim);
        if (job.ch->n_daughters == 2) {
          require(std::fabs(st.max_e - st.min_e) <= 1e-12 * lim,
                  "%s: a two-body channel's energy is not fixed (min %.10g, max %.10g)", buf,
                  st.min_e, st.max_e);
        }
        {
          const double floor = 1e-12 * lim + 1e-300;
          const double ga = lim - st.max_e + floor;
          const double gb = lim - moments.num(r, m_max) + floor;
          maxc.note((ga > gb) ? ga / gb : gb / ga, buf);
          const double ma = st.min_e + floor;
          const double mb = moments.num(r, m_min) + floor;
          minc.note((ma > mb) ? ma / mb : mb / ma, buf);
        }

        // Isotropy, against a fixed axis: 0 and 1/3 for every slot of every channel here.
        const double sez = stderr_of_mean(st.cz1, st.cz2, n);
        if (!within_sigma(st.cz1 / n, moments.num(r, m_cz), sez, 6.0, 0.0, &sig)) {
          require(false, "%s: <cos> %.6g vs %.6g (%.1f sigma)", buf, st.cz1 / n,
                  moments.num(r, m_cz), sig);
        }
        cosz.note(sig, buf);
        const double sez2 = stderr_of_mean(st.cz2, st.cz4, n);
        if (!within_sigma(st.cz2 / n, moments.num(r, m_cz2), sez2, 6.0, 0.0, &sig)) {
          require(false, "%s: <cos^2> %.6g vs %.6g (%.1f sigma)", buf, st.cz2 / n,
                  moments.num(r, m_cz2), sig);
        }
        cos2z.note(sig, buf);
      }

      // Pairwise opening angles: the part of the final state isotropy does not determine.
      for (std::size_t r = 0; r < pairs.rows.size(); ++r) {
        if (pairs.i32(r, p_pdg) != job.parent_pdg || pairs.i32(r, p_ch) != job.channel) {
          continue;
        }
        const int a = pairs.i32(r, p_i);
        const int b = pairs.i32(r, p_j);
        const PairAcc& pa = acc.pair[a][b];
        const double n = acc.accepted;
        char buf[220];
        std::snprintf(buf, sizeof buf, "%s slots %d-%d", who, a, b);
        double sig = 0;
        const double se1 = stderr_of_mean(pa.c1, pa.c2, n);
        if (!within_sigma(pa.c1 / n, pairs.num(r, p_c1), se1, 6.0, 1e-12, &sig)) {
          require(false, "%s: <cos_ij> %.10g vs %.10g (%.1f sigma)", buf, pa.c1 / n,
                  pairs.num(r, p_c1), sig);
        }
        paircos.note(sig, buf);
        const double se2 = stderr_of_mean(pa.c2, pa.c4, n);
        if (!within_sigma(pa.c2 / n, pairs.num(r, p_c2), se2, 6.0, 1e-12, &sig)) {
          require(false, "%s: <cos^2_ij> %.10g vs %.10g (%.1f sigma)", buf, pa.c2 / n,
                  pairs.num(r, p_c2), sig);
        }
        paircos2.note(sig, buf);
      }

      // The spectra, per bin, as a Poisson difference. Five sigma with a +1 floor: over the
      // ~2200 bins in this file that is an expected 1e-3 false failures.
      for (std::size_t r = 0; r < hist.rows.size(); ++r) {
        if (hist.i32(r, h_pdg) != job.parent_pdg || hist.i32(r, h_ch) != job.channel) {
          continue;
        }
        const int s = hist.i32(r, h_slot);
        const int b = hist.i32(r, h_bin);
        if (s < 0 || s >= acc.nd || b < 0 || b >= kHistBins) { continue; }
        require(std::fabs(hist.num(r, h_emax) - acc.slot[s].emax) <=
                    1e-12 * acc.slot[s].emax,
                "%s slot %d: histogram top edge differs", who, s);
        const double ours = acc.slot[s].hist[b];
        const double ref = hist.num(r, h_cnt);
        ++hist_bins;
        const double sigma = std::sqrt(ours + ref + 1.0);
        const double dev = std::fabs(ours - ref) / sigma;
        if (dev > worst_hist) {
          worst_hist = dev;
          char buf[220];
          std::snprintf(buf, sizeof buf, "%s slot %d bin %d (%.0f vs %.0f)", who, s, b, ours,
                        ref);
          worst_hist_where = buf;
        }
        if (dev > 5.0) { ++hist_bad; }
      }
    }

    std::printf("  %d channels sampled (including the synthetic 4-body)\n", channels_done);
    std::printf("  slot kinematic limit   %4d values  worst rel %10.3e   %s\n", limit.n,
                limit.worst, limit.where.c_str());
    std::printf("  <Ekin>                 %4d values  worst %6.2f sigma   %s\n", mean1.n,
                mean1.worst, mean1.where.c_str());
    std::printf("  <Ekin^2>               %4d values  worst %6.2f sigma   %s\n", mean2.n,
                mean2.worst, mean2.where.c_str());
    std::printf("  gap to the limit       %4d values  port/G4 ratio %7.2f (not asserted, see "
                "the comment)   %s\n",
                maxc.n, maxc.worst, maxc.where.c_str());
    std::printf("  sampled min            %4d values  port/G4 ratio %7.2f (not asserted)   "
                "%s\n",
                minc.n, minc.worst, minc.where.c_str());
    std::printf("  <cos> vs fixed axis    %4d values  worst %6.2f sigma   %s\n", cosz.n,
                cosz.worst, cosz.where.c_str());
    std::printf("  <cos^2> vs fixed axis  %4d values  worst %6.2f sigma   %s\n", cos2z.n,
                cos2z.worst, cos2z.where.c_str());
    std::printf("  <cos> between products %4d values  worst %6.2f sigma   %s\n", paircos.n,
                paircos.worst, paircos.where.c_str());
    std::printf("  <cos^2> between        %4d values  worst %6.2f sigma   %s\n", paircos2.n,
                paircos2.worst, paircos2.where.c_str());
    std::printf("  spectra                %4d bins    worst %6.2f sigma   %s\n", hist_bins,
                worst_hist, worst_hist_where.c_str());
    std::printf("  four-momentum closes   worst |sum p|/M %.3e, |sum E - M|/M %.3e   %s\n",
                worst_p_res[0], worst_e_res[0], worst_res_where[0].c_str());
    std::printf("    G4MuonDecayChannel   worst |sum p|/M %.3e, |sum E - M|/M %.3e   "
                "(m_e/m_mu = %.3e, the mass it neglects)\n",
                worst_p_res[1], worst_e_res[1],
                particle_mass(kPdgElectron) / particle_mass(kPdgMuMinus));
    std::printf("  closure vs Geant4's    %4d channels  worst %.2f of the 30x band   %s\n",
                closure_p.n, (closure_p.worst > closure_e.worst) ? closure_p.worst
                                                                 : closure_e.worst,
                (closure_p.worst > closure_e.worst) ? closure_p.where.c_str()
                                                    : closure_e.where.c_str());

    require(limit.worst <= 1e-14, "kinematic limit worst %.3e", limit.worst);
    require(hist_bad == 0, "%d of %d spectrum bins are past 5 sigma", hist_bad, hist_bins);
    // G4MuonDecayChannel's energy residual is not merely BOUNDED by the electron mass, it is
    // EQUAL to it, every time: the three reduced energies sum to 2 by construction
    // (Enm = 2 - Ee - Ene), each daughter's kinetic energy is its reduced energy times
    // EMax = m_mu/2 - m_e, and only the electron carries a rest mass - so
    //   sum E = 2*EMax + m_e = m_mu - m_e
    // exactly, for any sampled point. That makes it an equality to assert rather than a
    // tolerance to choose, and it fails if the sampler stops conserving the reduced energies.
    // The momentum residual is bounded by the same number and approaches it from below
    // (sqrt(T^2+2 T m_e) - T -> m_e as T grows), so that one is an inequality.
    const double me_over_mmu = particle_mass(kPdgElectron) / particle_mass(kPdgMuMinus);
    // 1e-9 rather than 1e-12 because this is the MAXIMUM over 400,000 samples of a quantity
    // that is analytically constant, and each sample's value has been round-tripped through
    // G4DynamicParticle's (direction, kinetic energy, mass) representation four times.
    require(std::fabs(worst_e_res[1] - me_over_mmu) <= 1e-9 * me_over_mmu,
            "the muon channel's energy residual %.17g is not exactly m_e/m_mu = %.17g",
            worst_e_res[1], me_over_mmu);
    require(worst_p_res[1] <= me_over_mmu,
            "the muon channel's momentum residual %.3e exceeds the electron mass it neglects "
            "(%.3e of the parent)",
            worst_p_res[1], me_over_mmu);
    std::printf("\n");
  }

  // ==================================================================================
  // 6. The Michel spectrum, against its analytic form.
  // ==================================================================================
  {
    std::printf("== G4MuonDecayChannel's electron spectrum against the analytic Michel form ==\n");
    // The sampling draws Ee uniform on (0,1) and Ene from y(1-y), accepting only
    // Ee + Ene >= 1. Integrating out Ene leaves p(x) = (x^2/6)(3-2x) for x = Ee, so the
    // moments of the reduced KINETIC energy x = T_e/(m_mu/2 - m_e) are
    //     <x> = 7/10,  <x^2> = 8/15,  <x^3> = 3/7
    // exactly - the electron mass cancels out of the reduction, because x is defined against
    // the endpoint. None of these three numbers comes from Geant4.
    const double analytic[3] = {0.7, 8.0 / 15.0, 3.0 / 7.0};
    const char* names3[3] = {"<x>", "<x^2>", "<x^3>"};
    const int muon_pdgs[2] = {kPdgMuPlus, kPdgMuMinus};
    for (int mi = 0; mi < 2; ++mi) {
      const int pdg = muon_pdgs[mi];
      const DecayTableRow tab = decay_table_for(pdg);
      const ChannelRow& ch = channel_rows()[tab.first];
      const double m_mu = particle_mass(pdg);
      const double emax = m_mu / 2.0 - particle_mass(ch.daughter[0]);
      Philox<real_t> rng(static_cast<uint32_t>(pdg + 1000), 3u, 19u);
      DecayProducts<real_t> out;
      double s[7] = {0, 0, 0, 0, 0, 0, 0};
      int n = 0;
      for (int i = 0; i < kSamples; ++i) {
        channel_decay_it<real_t>(ch, pdg, m_mu, rng, out);
        if (out.status != DecayStatus::kOK) { continue; }
        // Slot 0 is the electron: G4MuonDecayChannel pushes daughter 0 first and daughter 0
        // is e+/e-.
        const double x = static_cast<double>(out.p[0].ekin) / emax;
        double xk = 1.0;
        for (int k = 0; k < 7; ++k) {
          s[k] += xk;
          xk *= x;
        }
        ++n;
      }
      for (int k = 1; k <= 3; ++k) {
        const double m = s[k] / n;
        // The standard error of <x^k> from the sampled 2k-th moment, so the band is the
        // sample's own and not a guess.
        double var = s[2 * k] / n - m * m;
        if (var < 0) { var = 0; }
        const double se = std::sqrt(var / n);
        const double sig = (se > 0) ? std::fabs(m - analytic[k - 1]) / se : 0.0;
        std::printf("  %s %-6s sampled %.8f  analytic %.8f  %5.2f sigma (se %.2e)\n",
                    (pdg == kPdgMuPlus) ? "mu+" : "mu-", names3[k - 1], m, analytic[k - 1],
                    sig, se);
        require(sig <= 6.0, "%s %s is %.2f sigma from the analytic Michel value",
                (pdg == kPdgMuPlus) ? "mu+" : "mu-", names3[k - 1], sig);
      }
    }
    std::printf("\n");
  }

  // ==================================================================================
  // 7. The refusals, from the port's side.
  // ==================================================================================
  {
    std::printf("== refusals ==\n");
    // A species with no table must come back kNoTable with a message, whatever mass and
    // energy it is handed. K0L (130) is the one the plan names.
    const int refused[] = {130, 310, 3122, 3222, 3334, -15, 999999};
    DecayProducts<real_t> out;
    const real_t dir[3] = {0, 0, 1};
    for (int pdg : refused) {
      Philox<real_t> rng(1u, 2u, 3u);
      sample_decay<real_t>(pdg, real_t(500), real_t(100), dir, false, rng, out);
      require(out.status == DecayStatus::kNoTable && out.n == 0,
              "pdg %d: expected kNoTable with no products, got status %d and %d products", pdg,
              static_cast<int>(out.status), out.n);
    }
    std::printf("  %d refused PDG codes return kNoTable with no products\n",
                static_cast<int>(sizeof(refused) / sizeof(refused[0])));

    // A stable species must be a no-op, not a refusal and not a decay.
    for (int pdg : {kPdgProton, kPdgElectron, kPdgGamma}) {
      Philox<real_t> rng(1u, 2u, 3u);
      sample_decay<real_t>(pdg, real_t(particle_mass(pdg)), real_t(10), dir, false, rng, out);
      require(out.status == DecayStatus::kStable && out.n == 0,
              "pdg %d: a stable particle must be a no-op, got status %d", pdg,
              static_cast<int>(out.status));
    }
    std::printf("  stable species (p, e-, gamma) are a no-op, not a refusal\n");

    // The at-rest competition hook. A stopped pi- or K- is CAPTURED by
    // G4HadronicAbsorptionBertini before it can decay; a stopped mu- competes with
    // G4MuonMinusCapture. Neither is implemented here, and the hook is what a wiring package
    // has to consult.
    require(decay_at_rest_competitor(kPdgPiMinus) ==
                AtRestCompetitor::kHadronicAbsorptionBertini,
            "pi- has no at-rest competitor");
    require(decay_at_rest_competitor(kPdgKaonMinus) ==
                AtRestCompetitor::kHadronicAbsorptionBertini,
            "kaon- has no at-rest competitor");
    require(decay_at_rest_competitor(kPdgMuMinus) == AtRestCompetitor::kMuonMinusCapture,
            "mu- has no at-rest competitor");
    require(decay_at_rest_competitor(kPdgPiPlus) == AtRestCompetitor::kNone,
            "pi+ should have no at-rest competitor: nothing captures a positive hadron");
    std::printf("  at-rest competition: pi- and kaon- -> %s\n",
                decay_at_rest_competitor_name(AtRestCompetitor::kHadronicAbsorptionBertini));
    std::printf("                       mu-           -> %s\n",
                decay_at_rest_competitor_name(AtRestCompetitor::kMuonMinusCapture));
    std::printf("  at-rest mean life: pi- %.4g ns, kaon- %.4g ns, mu- %.4g ns\n",
                at_rest_mean_life<double>(kPdgPiMinus), at_rest_mean_life<double>(kPdgKaonMinus),
                at_rest_mean_life<double>(kPdgMuMinus));
    std::printf("\n");
  }

  // ==================================================================================
  // 8. The interaction-length count, which is the process's own state and not the stepper's.
  // ==================================================================================
  {
    std::printf("== the number of interaction lengths left, across steps ==\n");
    // `-log(U)`: mean 1, variance 1, and strictly positive. Checked as a distribution
    // because the value is what every length in this file is multiplied by, and a sign error
    // in it would give negative step limits that a single sample might not show.
    Philox<real_t> rng(20260910u, 13u, 29u);
    double s1 = 0, s2 = 0;
    double smallest = 1e300;
    const int n = 200000;
    for (int i = 0; i < n; ++i) {
      const double x = reset_number_of_interaction_lengths<double>(rng);
      require(x > 0.0 || i > 0, "the first -log(U) draw was %.17g", x);
      if (x < smallest) { smallest = x; }
      s1 += x;
      s2 += x * x;
    }
    const double m1 = s1 / n;
    const double var = s2 / n - m1 * m1;
    const double se = std::sqrt(var / n);
    std::printf("  -log(U): mean %.6f (analytic 1, %.2f sigma), variance %.4f (analytic 1), "
                "smallest of %d draws %.3e\n",
                m1, std::fabs(m1 - 1.0) / se, var, n, smallest);
    require(std::fabs(m1 - 1.0) <= 6.0 * se, "<-log(U)> is %.6f, not 1", m1);
    require(std::fabs(var - 1.0) <= 0.05, "var(-log(U)) is %.4f, not 1", var);
    require(smallest > 0.0, "-log(U) produced a non-positive value");

    // SubtractNumberOfInteractionLengthLeft, and the double floor at perMillion. Three
    // cases, and the middle one is the whole reason the floor exists.
    struct Case {
      double n_left, prev_step, current, expect;
      const char* what;
    };
    const Case cases[] = {
        {1.0, 0.25, 1.0, 0.75, "a quarter of the length consumed"},
        {1.0, 1.0, 1.0, per_million(),
         "the WHOLE length consumed: floored at perMillion, not zero, so the next step's "
         "limit is tiny rather than exactly zero"},
        {1.0, 2.0, 1.0, per_million(), "more than the length consumed: floored, not negative"},
        {0.5, 0.0, 0.0, 0.5, "a zero current length: no subtraction at all"},
    };
    for (const Case& c : cases) {
      const double got = subtract_interaction_lengths<double>(c.n_left, c.prev_step, c.current);
      require(reldev(got, c.expect) <= 1e-15, "%s: got %.17g, expected %.17g", c.what, got,
              c.expect);
    }
    std::printf("  subtract-and-floor: %d cases, including a fully consumed length flooring "
                "at %.0e rather than 0\n\n",
                static_cast<int>(sizeof(cases) / sizeof(cases[0])), per_million());
  }

  // ==================================================================================
  // 9. The two masses a decay has, and the fact that they are not the same one.
  // ==================================================================================
  {
    std::printf("== the dynamical-mass snap in G4DynamicParticle's four-argument ctor ==\n");
    // Every G4PhaseSpaceDecayChannel builds its rest-frame parent with
    // `new G4DynamicParticle(parent, dummy, 0.0, parentmass)`, and that constructor keeps the
    // PDG mass unless the given mass differs by more than 1e-5 MeV. So the kinematics use the
    // dynamic mass and the BOOST uses the snapped one, and a decay handed a mass 1e-6 MeV off
    // the PDG value has two different parent masses in it at once.
    //
    // Not in the oracle, because G4DecayProducts' parent is not reachable from a dumped
    // product list - `G4DecayProducts::GetParentParticle` gives it, but the number that
    // matters is the one Boost divides by, and the way to see that is the identity below:
    // inside the allowance the boost must be IDENTICAL to the PDG-mass boost, and outside it
    // must not be.
    const DecayTableRow pip = decay_table_for(kPdgPiPlus);
    const double m_pdg = particle_mass(kPdgPiPlus);
    struct Case { double delta; bool snaps; };
    const Case cases[] = {{0.0, true}, {1e-6, true}, {-1e-6, true}, {1e-4, false},
                          {-1e-4, false}};
    for (const Case& c : cases) {
      const double given = m_pdg + c.delta;
      const double snapped = snapped_dynamical_mass(m_pdg, given);
      require((snapped == m_pdg) == c.snaps,
              "a parent mass %+.1e MeV off the PDG value snapped to %.17g (expected %s)",
              c.delta, snapped, c.snaps ? "the PDG mass" : "itself");
      DecayProducts<real_t> out;
      Philox<real_t> rng(4242u, 5u, 7u);
      channel_decay_it<real_t>(channel_rows()[pip.first], kPdgPiPlus,
                               static_cast<real_t>(given), rng, out);
      require(out.status == DecayStatus::kOK, "pi+ -> mu+ nu did not sample");
      require(static_cast<double>(out.parent_mass) == snapped,
              "the buffer's parent mass is %.17g, not the snapped %.17g",
              static_cast<double>(out.parent_mass), snapped);
      // And the kinematics did use the UNSNAPPED mass: the two-body momentum is Pmx of the
      // given mass, which differs from Pmx of the PDG mass even inside the allowance.
      const double p_given = pmx<double>(given, particle_mass(kPdgMuPlus), 0.0);
      const double p_got = static_cast<double>(out.p[1].ekin);  // the massless neutrino
      require(reldev(p_got, p_given) <= 1e-14,
              "the daughter momentum %.17g came from neither the given mass (%.17g) nor "
              "anything near it", p_got, p_given);
    }
    std::printf("  five parent masses around pi+'s: the boost mass snaps inside 1e-5 MeV and "
                "the kinematics never do\n\n");
  }

  if (g_perturb != nullptr) {
    // The inversion. A perturbed run that PASSES means the assertions it broke are not
    // assertions, and that is the failure being reported.
    std::printf("perturbation '%s': %d assertion(s) fired\n", g_perturb, g_fails);
    if (g_fails == 0) {
      std::printf("VACUOUS - nothing failed under this perturbation\n");
      return 1;
    }
    std::printf("PASSED (the perturbation was detected)\n");
    return 0;
  }
  std::printf("%s (%d failures)\n", g_fails ? "FAILED" : "PASSED", g_fails);
  return g_fails ? 1 : 0;
}
