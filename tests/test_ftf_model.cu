// G4FTFModel and the classes that make the strings, against ref/oracle/ftf_*.csv.
//
// This is the half of FTFP that P11 could not write: from a projectile and a nucleus to a
// vector of excited strings and a wounded nucleus. Six tables, and the split between them is
// the split between what a shared random stream can check and what it cannot.
//
//   ftf_splitup.csv    G4DiffractiveSplitableHadron::SplitUp, i.e. ChooseStringEnds, for 37
//                      hadron codes x 8 phases of the eight-value cycle. Two parton codes and
//                      the draw count. This is where a proton becomes (u, ud1) or (d, uu1),
//                      and the SuppresUUDDSS rejection is visible only in the count.
//   ftf_hnelastic.csv  G4ElasticHNScattering::ElasticScattering on twelve constructed
//                      collisions x 8 phases: both four-momenta, both collision counts, the
//                      projectile's inherited creation time and position, the draw count.
//   ftf_excite.csv     G4DiffractiveExcitation::ExciteParticipants on the same twelve x 8:
//                      both hadrons' PDG codes (a charge exchange changes them), four-momenta,
//                      statuses and collision counts, and the draw count. All four arms -
//                      quark exchange, projectile diffraction, target diffraction and
//                      non-diffraction - are reached across the case x phase grid.
//   ftf_cstrings.csv   G4DiffractiveExcitation::CreateStrings on the same hadrons, both as
//                      projectile and as target: the two parton codes, both parton momenta,
//                      the direction, time and position.
//   ftf_nucleus.csv    a REPLAYED nucleus configuration per nuclide, and
//   ftf_getlist.csv    G4FTFParticipants::GetList on it x 5 projectiles x 8 phases: the impact
//                      parameter, the participant list and the interaction times, exactly.
//                      The nucleus is replayed because G4Fancy3DNucleus's rejection sampling
//                      shares no stream with this port's Philox - P9's pattern
//                      (docs/PORTED.md 2.1.10).
//   ftf_modelstat_*    the statistical half: 15 cases x 20,000 events of the whole model.
//                      String count and mass at GetStrings exit, species and multiplicity of
//                      the hadrons Scatter returns, and the wounded nucleus's hole count.
//
//   ftf_modelcases.csv the INPUTS of those cases - the projectile's PDG mass and lab momentum
//                      as Geant4 computed them - so that the statistical half does not have to
//                      copy a table and cannot disagree with the dump about what was run.
//   ftf_modelbig_*     three of those cases again at 200,000 events, which is docs/RISK.md
//                      V88's rule applied to the rows that sat at 3 sigma.
//
// WHY TWO CASE TABLES ARE STILL DUPLICATED HERE. The twelve constructed collisions and the five
// GetList projectiles are inputs to the EXACT tables, and those tables are keyed by a case name
// rather than by a row of numbers. P11's test copies the eight-value cycle for the same reason
// and gives the same warning: a second copy that drifts turns every exact comparison into
// noise. Both copies are literal and short enough to read side by side. The STATISTICAL cases
// are not duplicated - they come from ftf_modelcases.csv - because an ion's projectile mass is
// `G4IonTable::GetIonMass(Z, A)` and writing that number twice is exactly the mistake the
// warning describes.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "core/rng.cuh"
#include "physics/hadronic/ftf/theo_fs_generator.cuh"

using namespace g4gpu;
using namespace g4gpu::hadronic;

/// The same eight uniforms as ref/dump/dump_ftf.cc's CycleEngine.
__host__ __device__ inline const double* cycle_seq() {
  static const double v[8] = {0.05, 0.37, 0.63, 0.91, 0.12, 0.78, 0.29, 0.55};
  return v;
}

struct CycleRng {
  int phase = 0;
  int n = 0;
  __host__ __device__ void reset(int p) {
    phase = p;
    n = 0;
  }
  __host__ __device__ double uniform() {
    const double v = cycle_seq()[(n + phase) % 8];
    ++n;
    return v;
  }
};

/// Never launched. The probe for the WHOLE of `ftf::apply_yourself`, which is what the P11b
/// brief asks to be reported: the entry point instantiated for the device, with the P6
/// hand-over behind its `__noinline__` so that `preco::deexcite`'s 10 kB frame
/// (docs/PORTED.md 2.1.10) is not added to every caller's.
using ProbeWS = ftf::FtfWorkspace<250, 64, 512, 320, 256, 96>;

__global__ void ftf_apply_device_probe(ftf::LundTables<double>* lund, ProbeWS* ws,
                                       physics::hadronic::HadProjectile<double>* proj,
                                       physics::hadronic::HadNucleus* nuc,
                                       physics::hadronic::HadFinalState<double, 128>* out,
                                       int* res) {
  Philox<double> rng(1u, 2u);
  ftf::apply_yourself(*proj, *nuc, *out, ws, lund, rng);
  res[0] = out->n_secondaries;
  res[1] = static_cast<int>(ws->report.refused);
  res[2] = ws->report.attempts;
  res[3] = ws->model.n_strings;
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

std::vector<std::string> split_csv(const std::string& s) {
  std::vector<std::string> out;
  std::string cur;
  for (char c : s) {
    if (c == ',') {
      out.push_back(cur);
      cur.clear();
    } else if (c != '\r') {
      cur.push_back(c);
    }
  }
  out.push_back(cur);
  return out;
}

struct Csv {
  std::vector<std::string> cols;
  std::vector<std::vector<std::string>> rows;
  std::map<std::string, int> ix;

  bool load(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "r");
    if (f == nullptr) {
      std::printf("FAIL: cannot open %s\n", path.c_str());
      ++fails;
      return false;
    }
    char line[65536];
    bool first = true;
    while (std::fgets(line, sizeof(line), f) != nullptr) {
      std::string s(line);
      while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) { s.pop_back(); }
      if (s.empty()) { continue; }
      if (first) {
        cols = split_csv(s);
        for (int i = 0; i < static_cast<int>(cols.size()); ++i) { ix[cols[i]] = i; }
        first = false;
        continue;
      }
      rows.push_back(split_csv(s));
    }
    std::fclose(f);
    return true;
  }
  double d(size_t r, const char* c) const { return std::atof(rows[r][ix.at(c)].c_str()); }
  long long i(size_t r, const char* c) const { return std::atoll(rows[r][ix.at(c)].c_str()); }
  const std::string& s(size_t r, const char* c) const { return rows[r][ix.at(c)]; }
};

/// `xs::Projectile<double>` from a PDG code, through the same table the model reads.
hadronic::xs::Projectile<double> projectile_of(int pdg) {
  hadronic::xs::Projectile<double> p;
  const data::FtfHadron* h = data::ftf_find_hadron(pdg);
  p.pdg = pdg;
  p.mass = h->mass;
  p.charge = h->charge;
  p.baryon_number = h->baryon;
  p.n_lambdas = 0;
  return p;
}

// ---------------------------------------------------------------------------------------------
// The case tables, copied from ref/dump/dump_ftf.cc. See the file header.
// ---------------------------------------------------------------------------------------------

struct CollisionCase {
  const char* name;
  int proj_pdg;
  int targ_pdg;
  double plab;
  double targ_px, targ_py, targ_pz, targ_e;
  int targ_a, targ_z;
  int proj_status, targ_status;
  int proj_ncol, targ_ncol;
};

const CollisionCase kCollisions[] = {
    {"p_C_4GeV",     2212,  2212, 4700.0,  0.0,   0.0,  0.0,  920.0, 12,  6, 1, 1, 0, 0},
    {"p_C_4GeV_n",   2212,  2112, 4700.0,  30.0, -20.0, 15.0, 930.0, 12,  6, 1, 1, 0, 0},
    {"p_Pb_10GeV",   2212,  2212, 10800.0, 60.0,  40.0, -25.0, 900.0, 207, 82, 1, 1, 0, 0},
    {"n_O_10GeV",    2112,  2212, 10800.0, 0.0,   0.0,  0.0,  938.272013, 16, 8, 1, 1, 0, 0},
    {"pip_C_4GeV",   211,   2212, 4130.0,  0.0,   0.0,  0.0,  938.272013, 12, 6, 1, 1, 0, 0},
    {"pim_Fe_50GeV", -211,  2112, 50000.0, -10.0, 25.0, 5.0,  925.0, 56, 26, 1, 1, 0, 0},
    {"kp_C_10GeV",   321,   2212, 10500.0, 0.0,   0.0,  0.0,  938.272013, 12, 6, 1, 1, 0, 0},
    {"km_Al_50GeV",  -321,  2112, 50000.0, 12.0, -8.0,  30.0, 935.0, 27, 13, 1, 1, 0, 0},
    {"pbar_C_10GeV", -2212, 2212, 10800.0, 0.0,   0.0,  0.0,  938.272013, 12, 6, 1, 1, 0, 0},
    {"p_C_50GeV",    2212,  2112, 50000.0, 0.0,   0.0,  0.0,  939.56536, 12, 6, 1, 1, 0, 0},
    {"p_C_2ndcol",   2212,  2212, 4700.0,  0.0,   0.0,  0.0,  938.272013, 12, 6, 0, 0, 1, 1},
    {"delta_C_4GeV", 2214,  2112, 4700.0,  0.0,   0.0,  0.0,  939.56536, 12, 6, 2, 2, 1, 0},
};

/// `fermi` in this port's units (mm), as deex::fermi().
const double kFermi = 1e-12;

ftf::SplitableHadron make_projectile_splitable(int pdg, double px, double py, double pz) {
  const data::FtfHadron* d = data::ftf_find_hadron(pdg);
  const double e = std::sqrt(px * px + py * py + pz * pz + d->mass * d->mass);
  return ftf::splitable_from_primary(pdg, ftf::Vec4(px, py, pz, e));
}

ftf::SplitableHadron make_target_splitable(int pdg, double px, double py, double pz, double e,
                                           double x, double y, double z) {
  return ftf::splitable_from_nucleon(pdg, ftf::Vec4(px, py, pz, e), ftf::Vec3d{x, y, z});
}

const CollisionCase* find_case(const std::string& name) {
  for (const CollisionCase& c : kCollisions) {
    if (name == c.name) { return &c; }
  }
  return nullptr;
}

// ---------------------------------------------------------------------------------------------
// Statistical comparison, the shape tests/test_precompound.cu states.
// ---------------------------------------------------------------------------------------------

/// Two Poisson-ish counts out of N trials each, as a z-score on the difference of the two
/// binomial proportions. Bins where BOTH sides are below `min_count` are skipped and counted,
/// because a 5-sigma gate on a bin holding three events is a gate on nothing.
struct ZStat {
  const char* name;
  long long bins = 0;
  long long skipped = 0;
  double worst = 0.0;
  std::string where;
  double gate = 5.0;
};

void z_compare(ZStat& z, long long got, long long want, long long n, const std::string& where,
               long long min_count = 20) {
  if (got < min_count && want < min_count) {
    ++z.skipped;
    return;
  }
  ++z.bins;
  const double p1 = static_cast<double>(got) / n;
  const double p2 = static_cast<double>(want) / n;
  const double pbar = 0.5 * (p1 + p2);
  const double var = pbar * (1.0 - pbar) * 2.0 / n;
  const double s = (var > 0.0) ? std::fabs(p1 - p2) / std::sqrt(var) : 0.0;
  if (s > z.worst) {
    z.worst = s;
    z.where = where + " got " + std::to_string(got) + " want " + std::to_string(want);
  }
}

/// The same comparison for a histogram whose entries are OBJECTS rather than events - species
/// counts, string masses, the excited/not-excited split - where one event contributes several
/// and the count can exceed the number of events.
///
/// THE BINOMIAL FORM ABOVE CANNOT BE USED THERE, and neither can the plain Poisson one.
///  * Binomial: `p = count/N` exceeds 1, `p(1-p)` goes negative, and the z it produces is
///    meaningless - it read 325 sigma for a 14% difference in the pi0 yield.
///  * Poisson (`|a-b|/sqrt(a+b)`): correct only if the entries were independent. They are not.
///    One event of a 4 GeV proton on lead makes a dozen strings at once, so the bin count per
///    event has a variance several times its mean, and a test that assumes Var = Mean reports
///    a 1.45% difference in a quarter-million-count bin as 5.09 sigma.
///
/// So the variance is MEASURED rather than assumed: `overdispersion` is Var/Mean of the
/// per-event bin count, taken from the port's own events (which is the same physics, so the
/// same correlation), floored at 1 so that a bin with fewer entries than events still gets the
/// Poisson width. `z = |a-b| / sqrt(f*(a+b))`.
///
/// Measured on ref/oracle/ftf_modelstat_strings.csv: for p+Pb at 4 GeV the log10-mass bin 29
/// holds 12.4 strings per event with f = 5.7, and the worst z over all 148 bins falls from
/// 5.09 to 2.14. Nothing about the port changed; the 5.09 was a property of the test.
void z_compare_counts(ZStat& z, long long got, long long want, double overdispersion,
                      const std::string& where, long long min_count = 20) {
  if (got < min_count && want < min_count) {
    ++z.skipped;
    return;
  }
  ++z.bins;
  const double f = (overdispersion > 1.0) ? overdispersion : 1.0;
  const double tot = f * (static_cast<double>(got) + static_cast<double>(want));
  const double s =
      (tot > 0.0) ? std::fabs(static_cast<double>(got) - static_cast<double>(want)) /
                        std::sqrt(tot)
                  : 0.0;
  if (s > z.worst) {
    z.worst = s;
    z.where = where + " got " + std::to_string(got) + " want " + std::to_string(want) +
              " f=" + std::to_string(f);
  }
}

/// Two MEANS compared: `|m1 - m2| / sqrt(var/n1 + var/n2)`, with the variance measured on the
/// port side and used for both sides.
///
/// Using one variance for two samples is a real assumption and it is the right one here: the
/// two sides are meant to be the same distribution, so if they are, the port's variance is the
/// oracle's; and if they are not, the difference in the MEANS is what this is looking for and
/// a wrong variance changes the z by a factor, not a sign. The oracle dumps sums and counts but
/// not sums of squares, which is why the variance has to come from one side.
void z_mean_compare(ZStat& z, double sum_port, double sumsq_port, long long n_port,
                    double sum_ref, long long n_ref, const std::string& where) {
  ++z.bins;
  const double m1 = sum_port / n_port;
  const double m2 = sum_ref / n_ref;
  const double var = (sumsq_port / n_port - m1 * m1) * n_port / (n_port - 1.0);
  const double se = (var > 0.0) ? std::sqrt(var / n_port + var / n_ref) : 0.0;
  const double s = (se > 0.0) ? std::fabs(m1 - m2) / se : 0.0;
  if (s > z.worst) {
    z.worst = s;
    char buf[128];
    std::snprintf(buf, sizeof(buf), " got %.6g want %.6g", m1, m2);
    z.where = where + buf;
  }
}

/// Var/Mean of a per-event count, from the running sum and sum-of-squares.
double overdispersion_of(long long sum, long long sum_sq, long long n_events) {
  if (n_events < 2 || sum <= 0) { return 1.0; }
  const double mean = static_cast<double>(sum) / n_events;
  const double mean_sq = static_cast<double>(sum_sq) / n_events;
  const double var = (mean_sq - mean * mean) * n_events / (n_events - 1);
  return (var > 0.0) ? var / mean : 1.0;
}

std::vector<ZStat> zstats;

int new_z(const char* name, double gate = 5.0) {
  ZStat z;
  z.name = name;
  z.gate = gate;
  zstats.push_back(z);
  return static_cast<int>(zstats.size()) - 1;
}

}  // namespace

int main(int argc, char** argv) {
  const bool quick = (argc > 1 && std::strcmp(argv[1], "--quick") == 0);
  // `--means` prints the MEAN of each per-event histogram, port against oracle, case by case.
  // A z-score says a bin disagrees; a mean says by how much and in which direction, which is
  // what localises a disagreement to a projectile species or a target. It is a diagnostic and
  // not an assertion - nothing in it can fail the test.
  const bool means = (argc > 1 && std::strcmp(argv[1], "--means") == 0);
  const std::string dir = oracle_dir();

  ftf::LundTables<double>* lund = new ftf::LundTables<double>();
  ftf::lund_init(lund, true);

  // -------------------------------------------------------------------------------------------
  // 1. ftf_splitup.csv - G4DiffractiveSplitableHadron::SplitUp
  // -------------------------------------------------------------------------------------------
  const int b_split = new_bucket("splitup partons", 0.0);
  const int b_split_draws = new_bucket("splitup draws", 0.0);
  {
    Csv csv;
    if (csv.load(dir + "/ftf_splitup.csv")) {
      for (size_t r = 0; r < csv.rows.size(); ++r) {
        const int pdg = static_cast<int>(csv.i(r, "pdg"));
        const int ph = static_cast<int>(csv.i(r, "phase"));
        CycleRng rng;
        rng.reset(ph);
        ftf::SplitableHadron h = make_projectile_splitable(pdg, 0.0, 0.0, 5000.0);
        ftf::FtfRefusal sref = ftf::FtfRefusal::kNone;
        ftf::splitable_split_up(&h, &sref, rng);
        bool ok0 = false, ok1 = false;
        const int p0 = ftf::splitable_next_parton(&h, &ok0);
        const int p1 = ftf::splitable_next_parton(&h, &ok1);
        const std::string w = "pdg " + std::to_string(pdg) + " ph " + std::to_string(ph);
        cmp_int(b_split, ok0 ? p0 : 0, csv.i(r, "parton0"), w + " p0");
        cmp_int(b_split, ok1 ? p1 : 0, csv.i(r, "parton1"), w + " p1");
        cmp_int(b_split_draws, rng.n, csv.i(r, "draws"), w);
      }
    }
  }

  // -------------------------------------------------------------------------------------------
  // 2. ftf_hnelastic.csv - G4ElasticHNScattering::ElasticScattering
  // -------------------------------------------------------------------------------------------
  const int b_el_res = new_bucket("elastic verdict", 0.0);
  const int b_el_mom = new_bucket("elastic four-momenta", 1e-13);
  const int b_el_state = new_bucket("elastic counts/time/pos", 1e-14);
  const int b_el_draws = new_bucket("elastic draws", 0.0);
  {
    Csv csv;
    if (csv.load(dir + "/ftf_hnelastic.csv")) {
      for (size_t r = 0; r < csv.rows.size(); ++r) {
        const CollisionCase* c = find_case(csv.s(r, "case"));
        if (c == nullptr) {
          std::printf("FAIL: unknown elastic case %s\n", csv.s(r, "case").c_str());
          ++fails;
          continue;
        }
        const int ph = static_cast<int>(csv.i(r, "phase"));
        CycleRng rng;
        rng.reset(ph);
        ftf::FtfParameters<double> par;
        ftf::ftf_init_for_interaction(&par, projectile_of(c->proj_pdg), c->targ_a, c->targ_z,
                                      c->plab, lund);
        ftf::SplitableHadron pr = make_projectile_splitable(c->proj_pdg, 0.0, 0.0, c->plab);
        ftf::SplitableHadron tr =
            make_target_splitable(c->targ_pdg, c->targ_px, c->targ_py, c->targ_pz, c->targ_e,
                                  1.0 * kFermi, -2.0 * kFermi, 0.5 * kFermi);
        pr.status = c->proj_status;
        tr.status = c->targ_status;
        pr.collision_count = c->proj_ncol;
        tr.collision_count = c->targ_ncol;
        tr.time_of_creation = 3.25;
        const bool res = ftf::ftf_elastic_scattering(&pr, &tr, &par, rng);
        const std::string w = csv.s(r, "case") + " ph " + std::to_string(ph);
        cmp_int(b_el_res, res ? 1 : 0, csv.i(r, "result"), w);
        cmp(b_el_mom, pr.momentum.v.x, csv.d(r, "ppx"), w + " ppx");
        cmp(b_el_mom, pr.momentum.v.y, csv.d(r, "ppy"), w + " ppy");
        cmp(b_el_mom, pr.momentum.v.z, csv.d(r, "ppz"), w + " ppz");
        cmp(b_el_mom, pr.momentum.e, csv.d(r, "pe"), w + " pe");
        cmp(b_el_mom, tr.momentum.v.x, csv.d(r, "tpx"), w + " tpx");
        cmp(b_el_mom, tr.momentum.v.y, csv.d(r, "tpy"), w + " tpy");
        cmp(b_el_mom, tr.momentum.v.z, csv.d(r, "tpz"), w + " tpz");
        cmp(b_el_mom, tr.momentum.e, csv.d(r, "te"), w + " te");
        cmp_int(b_el_state, pr.collision_count, csv.i(r, "pncol"), w + " pncol");
        cmp_int(b_el_state, tr.collision_count, csv.i(r, "tncol"), w + " tncol");
        cmp(b_el_state, pr.time_of_creation, csv.d(r, "ptime"), w + " ptime");
        cmp(b_el_state, pr.position.x, csv.d(r, "px_pos"), w + " posx");
        cmp(b_el_state, pr.position.y, csv.d(r, "py_pos"), w + " posy");
        cmp(b_el_state, pr.position.z, csv.d(r, "pz_pos"), w + " posz");
        cmp_int(b_el_draws, rng.n, csv.i(r, "draws"), w);
      }
    }
  }

  // -------------------------------------------------------------------------------------------
  // 3. ftf_excite.csv - G4DiffractiveExcitation::ExciteParticipants
  // -------------------------------------------------------------------------------------------
  const int b_ex_res = new_bucket("excite verdict", 0.0);
  const int b_ex_pdg = new_bucket("excite species", 0.0);
  const int b_ex_mom = new_bucket("excite four-momenta", 1e-12);
  const int b_ex_state = new_bucket("excite status/counts", 0.0);
  const int b_ex_draws = new_bucket("excite draws", 0.0);
  {
    Csv csv;
    if (csv.load(dir + "/ftf_excite.csv")) {
      for (size_t r = 0; r < csv.rows.size(); ++r) {
        const CollisionCase* c = find_case(csv.s(r, "case"));
        if (c == nullptr) { continue; }
        const int ph = static_cast<int>(csv.i(r, "phase"));
        CycleRng rng;
        rng.reset(ph);
        ftf::FtfParameters<double> par;
        ftf::ftf_init_for_interaction(&par, projectile_of(c->proj_pdg), c->targ_a, c->targ_z,
                                      c->plab, lund);
        ftf::SplitableHadron pr = make_projectile_splitable(c->proj_pdg, 0.0, 0.0, c->plab);
        ftf::SplitableHadron tr =
            make_target_splitable(c->targ_pdg, c->targ_px, c->targ_py, c->targ_pz, c->targ_e,
                                  1.0 * kFermi, -2.0 * kFermi, 0.5 * kFermi);
        pr.status = c->proj_status;
        tr.status = c->targ_status;
        pr.collision_count = c->proj_ncol;
        tr.collision_count = c->targ_ncol;
        tr.time_of_creation = 3.25;
        ftf::ExciteCommon common;
        const bool res = ftf::ftf_excite_participants(&pr, &tr, &par, &common, rng);
        const std::string w = csv.s(r, "case") + " ph " + std::to_string(ph);
        cmp_int(b_ex_res, res ? 1 : 0, csv.i(r, "result"), w);
        cmp_int(b_ex_pdg, pr.pdg, csv.i(r, "ppdg"), w + " ppdg");
        cmp_int(b_ex_pdg, tr.pdg, csv.i(r, "tpdg"), w + " tpdg");
        cmp(b_ex_mom, pr.momentum.v.x, csv.d(r, "ppx"), w + " ppx");
        cmp(b_ex_mom, pr.momentum.v.y, csv.d(r, "ppy"), w + " ppy");
        cmp(b_ex_mom, pr.momentum.v.z, csv.d(r, "ppz"), w + " ppz");
        cmp(b_ex_mom, pr.momentum.e, csv.d(r, "pe"), w + " pe");
        cmp(b_ex_mom, tr.momentum.v.x, csv.d(r, "tpx"), w + " tpx");
        cmp(b_ex_mom, tr.momentum.v.y, csv.d(r, "tpy"), w + " tpy");
        cmp(b_ex_mom, tr.momentum.v.z, csv.d(r, "tpz"), w + " tpz");
        cmp(b_ex_mom, tr.momentum.e, csv.d(r, "te"), w + " te");
        cmp_int(b_ex_state, pr.status, csv.i(r, "pstatus"), w + " pstatus");
        cmp_int(b_ex_state, tr.status, csv.i(r, "tstatus"), w + " tstatus");
        cmp_int(b_ex_state, pr.collision_count, csv.i(r, "pncol"), w + " pncol");
        cmp_int(b_ex_state, tr.collision_count, csv.i(r, "tncol"), w + " tncol");
        cmp_int(b_ex_draws, rng.n, csv.i(r, "draws"), w);
      }
    }
  }

  // -------------------------------------------------------------------------------------------
  // 4. ftf_cstrings.csv - G4DiffractiveExcitation::CreateStrings
  // -------------------------------------------------------------------------------------------
  const int b_cs_partons = new_bucket("createstrings partons", 0.0);
  const int b_cs_mom = new_bucket("createstrings parton momenta", 1e-13);
  const int b_cs_state = new_bucket("createstrings dir/time/pos", 1e-14);
  const int b_cs_draws = new_bucket("createstrings draws", 0.0);
  {
    Csv csv;
    if (csv.load(dir + "/ftf_cstrings.csv")) {
      for (size_t r = 0; r < csv.rows.size(); ++r) {
        const CollisionCase* c = find_case(csv.s(r, "case"));
        if (c == nullptr) { continue; }
        const int ph = static_cast<int>(csv.i(r, "phase"));
        const int side = static_cast<int>(csv.i(r, "isproj"));
        CycleRng rng;
        rng.reset(ph);
        ftf::FtfParameters<double> par;
        ftf::ftf_init_for_interaction(&par, projectile_of(c->proj_pdg), c->targ_a, c->targ_z,
                                      c->plab, lund);
        ftf::SplitableHadron h =
            (side == 0) ? make_projectile_splitable(c->proj_pdg, 120.0, -80.0, c->plab)
                        : make_target_splitable(c->targ_pdg, c->targ_px, c->targ_py, c->targ_pz,
                                                c->targ_e + 400.0, 1.0 * kFermi, -2.0 * kFermi,
                                                0.5 * kFermi);
        h.status = (side == 0) ? c->proj_status : c->targ_status;
        h.time_of_creation = 3.25;
        ftf::ExcitedString made[2];
        int n_made = 0;
        ftf::FtfRefusal ref = ftf::FtfRefusal::kNone;
        ftf::ftf_create_strings(&h, side == 0, &par, made, &n_made, &ref, rng);
        const std::string w =
            csv.s(r, "case") + " side " + std::to_string(side) + " ph " + std::to_string(ph);
        cmp_int(b_cs_partons, n_made, csv.i(r, "nstrings"), w + " nstrings");
        if (n_made < 1) { continue; }
        cmp_int(b_cs_partons, made[0].left, csv.i(r, "left"), w + " left");
        cmp_int(b_cs_partons, made[0].right, csv.i(r, "right"), w + " right");
        cmp(b_cs_mom, made[0].pleft.v.x, csv.d(r, "lpx"), w + " lpx");
        cmp(b_cs_mom, made[0].pleft.v.y, csv.d(r, "lpy"), w + " lpy");
        cmp(b_cs_mom, made[0].pleft.v.z, csv.d(r, "lpz"), w + " lpz");
        cmp(b_cs_mom, made[0].pleft.e, csv.d(r, "le"), w + " le");
        cmp(b_cs_mom, made[0].pright.v.x, csv.d(r, "rpx"), w + " rpx");
        cmp(b_cs_mom, made[0].pright.v.y, csv.d(r, "rpy"), w + " rpy");
        cmp(b_cs_mom, made[0].pright.v.z, csv.d(r, "rpz"), w + " rpz");
        cmp(b_cs_mom, made[0].pright.e, csv.d(r, "re"), w + " re");
        cmp_int(b_cs_state, made[0].direction, csv.i(r, "direction"), w + " direction");
        cmp(b_cs_state, made[0].time_of_creation, csv.d(r, "time"), w + " time");
        cmp(b_cs_state, made[0].position.x, csv.d(r, "posx"), w + " posx");
        cmp(b_cs_state, made[0].position.y, csv.d(r, "posy"), w + " posy");
        cmp(b_cs_state, made[0].position.z, csv.d(r, "posz"), w + " posz");
        cmp_int(b_cs_draws, rng.n, csv.i(r, "draws"), w);
      }
    }
  }

  // -------------------------------------------------------------------------------------------
  // 5. ftf_nucleus.csv + ftf_getlist.csv - the replayed nucleus and the participant list
  // -------------------------------------------------------------------------------------------
  const int b_gl_b = new_bucket("getlist impact parameter", 1e-14);
  const int b_gl_n = new_bucket("getlist participant count", 0.0);
  const int b_gl_idx = new_bucket("getlist participant identity", 0.0);
  const int b_gl_t = new_bucket("getlist interaction time", 1e-13);
  const int b_gl_draws = new_bucket("getlist draws", 0.0);
  {
    Csv nuc_csv, list_csv;
    if (nuc_csv.load(dir + "/ftf_nucleus.csv") && list_csv.load(dir + "/ftf_getlist.csv")) {
      // Rebuild each nucleus from the dumped configuration.
      std::map<std::string, std::vector<size_t>> by_nucleus;
      for (size_t r = 0; r < nuc_csv.rows.size(); ++r) {
        by_nucleus[nuc_csv.s(r, "nucleus")].push_back(r);
      }
      struct ListCase {
        const char* name;
        int pdg;
        double plab;
      };
      const ListCase kListCases[] = {{"p4", 2212, 4700.0},
                                     {"p10", 2212, 10800.0},
                                     {"pip4", 211, 4130.0},
                                     {"kp10", 321, 10500.0},
                                     {"pbar10", -2212, 10800.0}};

      static bic::Nucleon nucleons[250];
      using Parts = ftf::FtfParticipants<250, 1024, 64>;
      Parts* parts = new Parts();

      for (size_t r = 0; r < list_csv.rows.size();) {
        const std::string nucname = list_csv.s(r, "nucleus");
        const std::string projname = list_csv.s(r, "proj");
        const int ph = static_cast<int>(list_csv.i(r, "phase"));
        const long long ninter = list_csv.i(r, "ninter");

        const ListCase* lc = nullptr;
        for (const ListCase& l : kListCases) {
          if (projname == l.name) { lc = &l; }
        }
        if (lc == nullptr) {
          ++r;
          continue;
        }

        // Replay the nucleus.
        const std::vector<size_t>& idx = by_nucleus[nucname];
        bic::Nucleus3D nuc;
        nuc.nucleons = nucleons;
        nuc.capacity = 250;
        nuc.my_a = static_cast<int>(idx.size());
        nuc.my_z = static_cast<int>(nuc_csv.i(idx[0], "z"));
        nuc.my_l = 0;
        // The density and the hard-core distance are what `bic::nucleus_init` would have set;
        // the replay reproduces them rather than re-sampling, because `GetOuterRadius` (which
        // GetList's impact-parameter range is) adds `nucleondistance` to the furthest nucleon.
        if (nuc.my_a < 17) {
          nuc.density = bic::make_shell_model_density(nuc.my_a, nuc.my_z);
          nuc.nucleondistance = (nuc.my_a == 12) ? 0.9 * 1e-12 : 0.8 * 1e-12;
        } else {
          nuc.density = bic::make_fermi_density(nuc.my_a, nuc.my_z);
          nuc.nucleondistance = 0.8 * 1e-12;
        }
        for (size_t k = 0; k < idx.size(); ++k) {
          const size_t rr = idx[k];
          bic::Nucleon& n = nucleons[nuc_csv.i(rr, "index")];
          const long long type = nuc_csv.i(rr, "type");
          n.type = (type == 1) ? bic::kProton : ((type == 2) ? bic::kNeutron : bic::kLambda);
          n.position = ftf::Vec3d{nuc_csv.d(rr, "posx"), nuc_csv.d(rr, "posy"), nuc_csv.d(rr, "posz")};
          n.momentum = ftf::Vec4(nuc_csv.d(rr, "px"), nuc_csv.d(rr, "py"), nuc_csv.d(rr, "pz"),
                                 nuc_csv.d(rr, "e"));
          n.binding_energy = nuc_csv.d(rr, "binding");
          n.hit = false;
          n.hit_by = ftf::kNullSplitable;
        }

        ftf::FtfParameters<double> par;
        ftf::ftf_init_for_interaction(&par, projectile_of(lc->pdg),
                                      static_cast<int>(nuc_csv.i(idx[0], "a")), nuc.my_z,
                                      lc->plab, lund);
        CycleRng rng;
        rng.reset(ph);
        parts->clean();
        parts->bin_interval = false;
        const data::FtfHadron* pd = data::ftf_find_hadron(lc->pdg);
        const ftf::Vec4 p4(0.0, 0.0, lc->plab,
                           std::sqrt(lc->plab * lc->plab + pd->mass * pd->mass));
        ftf::ftf_participants_get_list_hadron(parts, &nuc, &par, lc->pdg, p4, rng);

        const std::string w = nucname + " " + projname + " ph " + std::to_string(ph);
        cmp(b_gl_b, parts->b_impact, list_csv.d(r, "b"), w + " b");
        cmp_int(b_gl_n, parts->n_interactions, ninter, w + " ninter");
        cmp_int(b_gl_draws, rng.n, list_csv.i(r, "draws"), w + " draws");

        // Walk the rows of this (nucleus, proj, phase) group.
        size_t r0 = r;
        while (r0 < list_csv.rows.size() && list_csv.s(r0, "nucleus") == nucname &&
               list_csv.s(r0, "proj") == projname && list_csv.i(r0, "phase") == ph) {
          const long long k = list_csv.i(r0, "index");
          if (k >= 0 && k < parts->n_interactions) {
            cmp_int(b_gl_idx, parts->interactions[k].target_nucleon,
                    list_csv.i(r0, "targ_index"), w + " i" + std::to_string(k));
            cmp(b_gl_t, parts->interactions[k].interaction_time, list_csv.d(r0, "time"),
                w + " t" + std::to_string(k));
          }
          ++r0;
        }
        r = r0;
      }
      delete parts;
    }
  }

  // -------------------------------------------------------------------------------------------
  // 5b. ftf_aanucleus.csv + ftf_getlist_aa.csv - GetList's NUCLEUS-NUCLEUS arm, exactly
  //
  // Two replayed nuclei instead of one, the projectile already boosted and Lorentz-contracted
  // as G4FTFModel::Init leaves it. This exists because the ion arm's statistical rows were the
  // only ones outside the gate and a 3% difference in an AA participant count is invisible
  // against the event-to-event spread: here the same quantity is exact or it is not.
  // -------------------------------------------------------------------------------------------
  const int b_aa_b = new_bucket("AA getlist impact parameter", 1e-14);
  const int b_aa_n = new_bucket("AA getlist participant count", 0.0);
  const int b_aa_idx = new_bucket("AA getlist pair identity", 0.0);
  const int b_aa_t = new_bucket("AA getlist interaction time", 1e-13);
  const int b_aa_draws = new_bucket("AA getlist draws", 0.0);
  {
    Csv nuc_csv, list_csv;
    if (nuc_csv.load(dir + "/ftf_aanucleus.csv") && list_csv.load(dir + "/ftf_getlist_aa.csv")) {
      static bic::Nucleon pnucleons[250];
      static bic::Nucleon tnucleons[250];
      static ftf::Vec3d psaved[250];
      using Parts = ftf::FtfParticipants<250, 1024, 64>;
      Parts* parts = new Parts();

      // The case inputs the dump used, keyed by name. Only the kinetic energy is needed here -
      // the two configurations come from the oracle.
      struct AaCase {
        const char* name;
        int proj_a, proj_z, targ_a, targ_z;
        double kin;
      };
      const AaCase kAa[] = {{"He4_C12", 4, 2, 12, 6, 32000.0},
                            {"C12_C12", 12, 6, 12, 6, 96000.0},
                            {"C12_Pb207", 12, 6, 207, 82, 96000.0}};

      for (const AaCase& ac : kAa) {
        // Replay both nuclei.
        auto build = [&](int side, bic::Nucleon* store, bic::Nucleus3D& nuc, int a, int z) {
          nuc.nucleons = store;
          nuc.capacity = 250;
          nuc.my_a = a;
          nuc.my_z = z;
          nuc.my_l = 0;
          if (a < 17) {
            nuc.density = bic::make_shell_model_density(a, z);
            nuc.nucleondistance = (a == 12) ? 0.9 * kFermi : 0.8 * kFermi;
          } else {
            nuc.density = bic::make_fermi_density(a, z);
            nuc.nucleondistance = 0.8 * kFermi;
          }
          for (size_t r = 0; r < nuc_csv.rows.size(); ++r) {
            if (nuc_csv.s(r, "case") != ac.name) { continue; }
            if (nuc_csv.i(r, "side") != side) { continue; }
            bic::Nucleon& n = store[nuc_csv.i(r, "index")];
            const long long type = nuc_csv.i(r, "type");
            n.type = (type == 1) ? bic::kProton : ((type == 2) ? bic::kNeutron : bic::kLambda);
            n.position = ftf::Vec3d{nuc_csv.d(r, "posx"), nuc_csv.d(r, "posy"),
                                    nuc_csv.d(r, "posz")};
            n.momentum = ftf::Vec4(nuc_csv.d(r, "px"), nuc_csv.d(r, "py"), nuc_csv.d(r, "pz"),
                                   nuc_csv.d(r, "e"));
            n.binding_energy = nuc_csv.d(r, "binding");
            n.hit = false;
            n.hit_by = ftf::kNullSplitable;
          }
        };
        bic::Nucleus3D pnuc, tnuc;
        build(0, pnucleons, pnuc, ac.proj_a, ac.proj_z);
        build(1, tnucleons, tnuc, ac.targ_a, ac.targ_z);
        for (int i = 0; i < ac.proj_a; ++i) { psaved[i] = pnucleons[i].position; }

        const double pmass = deex::nuclear_mass(ac.proj_a, ac.proj_z);
        const double p = std::sqrt(ac.kin * (ac.kin + 2.0 * pmass));
        const ftf::Vec4 primary(0.0, 0.0, p, ac.kin + pmass);
        hadronic::xs::Projectile<double> proj;
        proj.pdg = physics::hadronic::pdg_nuclear_code(ac.proj_z, ac.proj_a);
        proj.mass = pmass;
        proj.charge = ac.proj_z;
        proj.baryon_number = ac.proj_a;
        proj.n_lambdas = 0;
        ftf::FtfParameters<double> par;
        ftf::ftf_init_for_interaction(&par, proj, ac.targ_a, ac.targ_z, p / ac.proj_a, lund);

        for (size_t r = 0; r < list_csv.rows.size();) {
          if (list_csv.s(r, "case") != ac.name) {
            ++r;
            continue;
          }
          const int ph = static_cast<int>(list_csv.i(r, "phase"));
          const long long ninter = list_csv.i(r, "ninter");

          for (int i = 0; i < ac.proj_a; ++i) {
            pnucleons[i].position = psaved[i];
            pnucleons[i].hit = false;
            pnucleons[i].hit_by = ftf::kNullSplitable;
          }
          for (int i = 0; i < ac.targ_a; ++i) {
            tnucleons[i].hit = false;
            tnucleons[i].hit_by = ftf::kNullSplitable;
          }
          CycleRng rng;
          rng.reset(ph);
          parts->clean();
          parts->bin_interval = false;
          ftf::ftf_participants_get_list_nucleus(parts, &tnuc, &pnuc, &par, primary, rng);

          const std::string w = std::string(ac.name) + " ph " + std::to_string(ph);
          cmp(b_aa_b, parts->b_impact, list_csv.d(r, "b"), w + " b");
          cmp_int(b_aa_n, parts->n_interactions, ninter, w + " ninter");
          cmp_int(b_aa_draws, rng.n, list_csv.i(r, "draws"), w + " draws");

          size_t r0 = r;
          while (r0 < list_csv.rows.size() && list_csv.s(r0, "case") == ac.name &&
                 list_csv.i(r0, "phase") == ph) {
            const long long k = list_csv.i(r0, "index");
            if (k >= 0 && k < parts->n_interactions) {
              cmp_int(b_aa_idx, parts->interactions[k].projectile_nucleon,
                      list_csv.i(r0, "proj_index"), w + " p" + std::to_string(k));
              cmp_int(b_aa_idx, parts->interactions[k].target_nucleon,
                      list_csv.i(r0, "targ_index"), w + " t" + std::to_string(k));
              cmp(b_aa_t, parts->interactions[k].interaction_time, list_csv.d(r0, "time"),
                  w + " time" + std::to_string(k));
            }
            ++r0;
          }
          r = r0;
        }
      }
      delete parts;
    }
  }

  // -------------------------------------------------------------------------------------------
  // 5c. ftf_annih.csv - G4FTFAnnihilation::Annihilate, all four channels
  // -------------------------------------------------------------------------------------------
  const int b_an_res = new_bucket("annihilation verdict", 0.0);
  const int b_an_pdg = new_bucket("annihilation species", 0.0);
  const int b_an_mom = new_bucket("annihilation four-momenta", 1e-14);
  const int b_an_state = new_bucket("annihilation status/counts/pos", 1e-14);
  const int b_an_part = new_bucket("annihilation partons", 0.0);
  const int b_an_pmom = new_bucket("annihilation parton momenta", 1e-14);
  const int b_an_draws = new_bucket("annihilation draws", 0.0);
  {
    struct AnnihCase {
      const char* name;
      int proj_pdg, targ_pdg;
      double plab, targ_px, targ_py, targ_pz, targ_e;
      int targ_a, targ_z;
    };
    const AnnihCase kAnnih[] = {
        {"pbar_p_rest", -2212, 2212, 1.0, 0.0, 0.0, 0.0, 938.272013, 1, 1},
        {"pbar_p_100", -2212, 2212, 100.0, 0.0, 0.0, 0.0, 938.272013, 12, 6},
        {"pbar_p_1G", -2212, 2212, 1000.0, 0.0, 0.0, 0.0, 938.272013, 12, 6},
        {"pbar_p_10G", -2212, 2212, 10000.0, 0.0, 0.0, 0.0, 938.272013, 12, 6},
        {"pbar_n_1G", -2212, 2112, 1000.0, 20.0, -15.0, 10.0, 930.0, 12, 6},
        {"nbar_p_1G", -2112, 2212, 1000.0, 0.0, 0.0, 0.0, 938.272013, 12, 6},
        {"nbar_n_10G", -2112, 2112, 10000.0, 0.0, 0.0, 0.0, 939.56536, 56, 26},
        {"lbar_p_1G", -3122, 2212, 1000.0, 0.0, 0.0, 0.0, 938.272013, 12, 6},
        {"pbar_d_1G", -2212, 2214, 1000.0, 0.0, 0.0, 0.0, 1232.0, 12, 6},
    };
    Csv csv;
    if (csv.load(dir + "/ftf_annih.csv")) {
      for (size_t r = 0; r < csv.rows.size(); ++r) {
        const AnnihCase* c = nullptr;
        for (const AnnihCase& k : kAnnih) {
          if (csv.s(r, "case") == k.name) { c = &k; }
        }
        if (c == nullptr) {
          std::printf("FAIL: unknown annihilation case %s\n", csv.s(r, "case").c_str());
          ++fails;
          continue;
        }
        const int ph = static_cast<int>(csv.i(r, "phase"));
        CycleRng rng;
        rng.reset(ph);
        ftf::FtfParameters<double> par;
        ftf::ftf_init_for_interaction(&par, projectile_of(c->proj_pdg), c->targ_a, c->targ_z,
                                      c->plab, lund);
        ftf::SplitableHadron pr = make_projectile_splitable(c->proj_pdg, 0.0, 0.0, c->plab);
        ftf::SplitableHadron tr =
            make_target_splitable(c->targ_pdg, c->targ_px, c->targ_py, c->targ_pz, c->targ_e,
                                  1.0 * kFermi, -2.0 * kFermi, 0.5 * kFermi);
        pr.status = 1;
        tr.status = 1;
        tr.time_of_creation = 3.25;
        ftf::SplitableHadron add;
        bool made_add = false;
        ftf::AnnihCommon common;
        const bool res = ftf::ftf_annihilate(&pr, &tr, &add, &made_add, &par, &common, rng);
        const std::string w = csv.s(r, "case") + " ph " + std::to_string(ph);
        cmp_int(b_an_res, res ? 1 : 0, csv.i(r, "result"), w);
        cmp_int(b_an_pdg, pr.pdg, csv.i(r, "ppdg"), w + " ppdg");
        cmp_int(b_an_pdg, tr.pdg, csv.i(r, "tpdg"), w + " tpdg");
        cmp(b_an_mom, pr.momentum.v.x, csv.d(r, "ppx"), w + " ppx");
        cmp(b_an_mom, pr.momentum.v.y, csv.d(r, "ppy"), w + " ppy");
        cmp(b_an_mom, pr.momentum.v.z, csv.d(r, "ppz"), w + " ppz");
        cmp(b_an_mom, pr.momentum.e, csv.d(r, "pe"), w + " pe");
        cmp(b_an_mom, tr.momentum.v.x, csv.d(r, "tpx"), w + " tpx");
        cmp(b_an_mom, tr.momentum.v.y, csv.d(r, "tpy"), w + " tpy");
        cmp(b_an_mom, tr.momentum.v.z, csv.d(r, "tpz"), w + " tpz");
        cmp(b_an_mom, tr.momentum.e, csv.d(r, "te"), w + " te");
        cmp_int(b_an_state, pr.status, csv.i(r, "pstatus"), w + " pstatus");
        cmp_int(b_an_state, tr.status, csv.i(r, "tstatus"), w + " tstatus");
        cmp_int(b_an_state, pr.collision_count, csv.i(r, "pncol"), w + " pncol");
        cmp_int(b_an_state, tr.collision_count, csv.i(r, "tncol"), w + " tncol");
        cmp(b_an_state, pr.time_of_creation, csv.d(r, "ptime"), w + " ptime");
        cmp(b_an_state, pr.position.x, csv.d(r, "pposx"), w + " pposx");
        cmp(b_an_state, pr.position.y, csv.d(r, "pposy"), w + " pposy");
        cmp(b_an_state, pr.position.z, csv.d(r, "pposz"), w + " pposz");
        cmp_int(b_an_part, pr.parton[0], csv.i(r, "pq0"), w + " pq0");
        cmp_int(b_an_part, pr.parton[1], csv.i(r, "pq1"), w + " pq1");
        cmp(b_an_pmom, pr.parton_mom[0].v.x, csv.d(r, "pq0px"), w + " pq0px");
        cmp(b_an_pmom, pr.parton_mom[0].v.y, csv.d(r, "pq0py"), w + " pq0py");
        cmp(b_an_pmom, pr.parton_mom[0].v.z, csv.d(r, "pq0pz"), w + " pq0pz");
        cmp(b_an_pmom, pr.parton_mom[0].e, csv.d(r, "pq0e"), w + " pq0e");
        cmp(b_an_pmom, pr.parton_mom[1].v.x, csv.d(r, "pq1px"), w + " pq1px");
        cmp(b_an_pmom, pr.parton_mom[1].v.y, csv.d(r, "pq1py"), w + " pq1py");
        cmp(b_an_pmom, pr.parton_mom[1].v.z, csv.d(r, "pq1pz"), w + " pq1pz");
        cmp(b_an_pmom, pr.parton_mom[1].e, csv.d(r, "pq1e"), w + " pq1e");
        cmp_int(b_an_part, made_add ? 1 : 0, csv.i(r, "nadd"), w + " nadd");
        if (made_add && csv.i(r, "nadd") == 1) {
          cmp_int(b_an_pdg, add.pdg, csv.i(r, "apdg"), w + " apdg");
          cmp_int(b_an_part, add.parton[0], csv.i(r, "aq0"), w + " aq0");
          cmp_int(b_an_part, add.parton[1], csv.i(r, "aq1"), w + " aq1");
          cmp(b_an_mom, add.momentum.v.x, csv.d(r, "apx"), w + " apx");
          cmp(b_an_mom, add.momentum.v.y, csv.d(r, "apy"), w + " apy");
          cmp(b_an_mom, add.momentum.v.z, csv.d(r, "apz"), w + " apz");
          cmp(b_an_mom, add.momentum.e, csv.d(r, "ae"), w + " ae");
          cmp(b_an_pmom, add.parton_mom[0].v.x, csv.d(r, "aq0px"), w + " aq0px");
          cmp(b_an_pmom, add.parton_mom[0].v.y, csv.d(r, "aq0py"), w + " aq0py");
          cmp(b_an_pmom, add.parton_mom[0].v.z, csv.d(r, "aq0pz"), w + " aq0pz");
          cmp(b_an_pmom, add.parton_mom[0].e, csv.d(r, "aq0e"), w + " aq0e");
          cmp(b_an_pmom, add.parton_mom[1].v.x, csv.d(r, "aq1px"), w + " aq1px");
          cmp(b_an_pmom, add.parton_mom[1].v.y, csv.d(r, "aq1py"), w + " aq1py");
          cmp(b_an_pmom, add.parton_mom[1].v.z, csv.d(r, "aq1pz"), w + " aq1pz");
          cmp(b_an_pmom, add.parton_mom[1].e, csv.d(r, "aq1e"), w + " aq1e");
        }
        cmp_int(b_an_draws, rng.n, csv.i(r, "draws"), w + " draws");
      }
    }
  }

  // -------------------------------------------------------------------------------------------
  // 6. ftf_modelstat_*.csv - the whole model, statistically
  // -------------------------------------------------------------------------------------------
  // The five that describe the final state, and three that say WHERE a disagreement is: the
  // impact parameter is the geometry alone, `participants` is the Glauber count separately
  // from the reggeon cascade, and `nncoll` is how many of those participants were actually
  // excited. The 12% pion deficit that the integer-division bug in
  // `G4FTFModel::ExciteParticipants` caused showed up in `nncoll` and `participants` while the
  // hole count still agreed to 0.2%, and nothing in the first five could have said which.
  /// Every G4FTFParameters number the statistical cases were run with, compared exactly.
  const int b_par = new_bucket("model FTF parameters", 1e-14);
  const int z_nstrings = new_z("model nstrings");
  const int z_smass = new_z("model string log10 mass");
  const int z_species = new_z("model species");
  const int z_mult = new_z("model multiplicity");
  const int z_holes = new_z("model wounded nucleons");
  // The rapidity/xF and pT spectra, as their first moments per species: <E>, <pz> and <pt2> in
  // the lab. A mean is not a spectrum, but it is what a 20,000-event oracle can carry per
  // species without a per-species histogram axis, and a port whose longitudinal or transverse
  // scale is wrong moves it.
  // The energy-momentum balance per event, in one number per case: the mean total energy of
  // everything Scatter returns. It is the sum of the oracle's per-species `sum_e` column over
  // species, divided by the number of events, against the same quantity from the port - so a
  // port that lost or invented energy anywhere between the impact parameter and the last
  // fragmentation moves it, whatever the species composition does.
  const int z_etot = new_z("model <E_total> per event");
  const int z_mean_e = new_z("model <E> per species");
  const int z_mean_pz = new_z("model <pz> per species");
  const int z_mean_pt2 = new_z("model <pt2> per species");
  const int z_b = new_z("model impact parameter");
  const int z_part = new_z("model participants");
  const int z_nn = new_z("model NN collisions");
  const int z_exc = new_z("model excited/not-excited");
  {
    // The case table comes from the ORACLE, not from a copy here: `ftf_modelcases.csv` carries
    // the projectile's PDG mass and lab momentum as Geant4 computed them, which for an ION is
    // `G4IonTable::GetIonMass(Z, A)` - one of the three answers P9's nucleus_model.cuh lists,
    // and not the one `G4Fancy3DNucleus::GetMass()` gives. Reading it removes the only place
    // this test could have disagreed with the dump about what was simulated.
    struct ModelCase {
      std::string name;
      int pdg;
      double kin;
      int a, z, proj_a, proj_z;
      double pmass, plab;
    };
    Csv cs, ch, cm, cw, cc;
    const bool have = cs.load(dir + "/ftf_modelstat_strings.csv") &&
                      ch.load(dir + "/ftf_modelstat_species.csv") &&
                      cm.load(dir + "/ftf_modelstat_mult.csv") &&
                      cw.load(dir + "/ftf_modelstat_wounded.csv") &&
                      cc.load(dir + "/ftf_modelcases.csv");
    std::vector<ModelCase> kModelCases;
    if (have) {
      for (size_t r = 0; r < cc.rows.size(); ++r) {
        ModelCase mc;
        mc.name = cc.s(r, "case");
        mc.pdg = static_cast<int>(cc.i(r, "pdg"));
        mc.kin = cc.d(r, "kin");
        mc.a = static_cast<int>(cc.i(r, "a"));
        mc.z = static_cast<int>(cc.i(r, "z"));
        mc.proj_a = static_cast<int>(cc.i(r, "proj_a"));
        mc.proj_z = static_cast<int>(cc.i(r, "proj_z"));
        mc.pmass = cc.d(r, "pmass");
        mc.plab = cc.d(r, "plab");
        kModelCases.push_back(mc);
      }
    }
    if (have) {
      using WS = ftf::FtfWorkspace<250, 64, 1024, 512, 256, 96>;
      WS* ws = new WS();
      const int n_events = quick ? 2000 : 20000;
      for (size_t ci = 0; ci < kModelCases.size(); ++ci) {
        const ModelCase& c = kModelCases[ci];
        // Reference histograms for this case.
        std::map<int, long long> ref_nstrings, ref_mass, ref_species, ref_mult, ref_holes;
        std::map<int, long long> ref_b, ref_part, ref_nn, ref_exc;
        // The three momentum sums per species, which is how the oracle carries the rapidity/xF
        // and pT spectra: `sum_pz/count` is the mean longitudinal momentum of that species and
        // `sum_pt2/count` its mean transverse momentum squared, both in the lab.
        std::map<int, double> ref_sum_e, ref_sum_pz, ref_sum_pt2;
        long long ref_n = 0;
        for (size_t r = 0; r < cs.rows.size(); ++r) {
          if (cs.s(r, "case") != c.name) { continue; }
          ref_n = cs.i(r, "n_events");
          const std::string& q = cs.s(r, "quantity");
          const int bin = static_cast<int>(cs.i(r, "bin"));
          if (q == "nstrings") {
            ref_nstrings[bin] = cs.i(r, "count");
          } else if (q == "log10mass") {
            ref_mass[bin] = cs.i(r, "count");
          } else if (q == "b_halffm") {
            ref_b[bin] = cs.i(r, "count");
          } else if (q == "participants") {
            ref_part[bin] = cs.i(r, "count");
          } else if (q == "nncoll") {
            ref_nn[bin] = cs.i(r, "count");
          } else if (q == "excited") {
            ref_exc[bin] = cs.i(r, "count");
          }
        }
        for (size_t r = 0; r < ch.rows.size(); ++r) {
          if (ch.s(r, "case") != c.name) { continue; }
          const int sp = static_cast<int>(ch.i(r, "pdg"));
          ref_species[sp] = ch.i(r, "count");
          ref_sum_e[sp] = ch.d(r, "sum_e");
          ref_sum_pz[sp] = ch.d(r, "sum_pz");
          ref_sum_pt2[sp] = ch.d(r, "sum_pt2");
        }
        for (size_t r = 0; r < cm.rows.size(); ++r) {
          if (cm.s(r, "case") != c.name) { continue; }
          ref_mult[static_cast<int>(cm.i(r, "multiplicity"))] = cm.i(r, "count");
        }
        for (size_t r = 0; r < cw.rows.size(); ++r) {
          if (cw.s(r, "case") != c.name || cw.s(r, "quantity") != "holes") { continue; }
          ref_holes[static_cast<int>(cw.i(r, "bin"))] = cw.i(r, "count");
        }
        if (ref_n == 0) {
          std::printf("FAIL: no oracle rows for model case %s\n", c.name.c_str());
          ++fails;
          continue;
        }

        // EXACT: every G4FTFParameters number the model above reads, for this case. P11's
        // `ftf_params.csv` grid is 22 named projectile PARTICLES and an ion is not one of them,
        // so `InitForInteraction`'s `ProjectileIsNucleus` arm had no coverage until here.
        {
          hadronic::xs::Projectile<double> pp;
          if (c.proj_a > 0) {
            pp.pdg = c.pdg;
            pp.mass = c.pmass;
            pp.charge = c.proj_z;
            pp.baryon_number = c.proj_a;
            pp.n_lambdas = 0;
          } else {
            pp = projectile_of(c.pdg);
          }
          const double plab_per_n = (c.proj_a > 0) ? c.plab / c.proj_a : c.plab;
          ftf::FtfParameters<double> par;
          ftf::ftf_init_for_interaction(&par, pp, c.a, c.z, plab_per_n, lund);
          const std::string w = c.name;
          cmp(b_par, plab_per_n, cc.d(ci, "plab_per_n"), w + " plab_per_n");
          cmp(b_par, par.x_total, cc.d(ci, "xtotal"), w + " xtotal");
          cmp(b_par, par.x_elastic, cc.d(ci, "xelastic"), w + " xelastic");
          cmp(b_par, par.x_inelastic, cc.d(ci, "xinel"), w + " xinel");
          cmp(b_par, par.prob_of_elastic_scatt, cc.d(ci, "prob_el"), w + " prob_el");
          cmp(b_par, par.prob_of_annihilation, cc.d(ci, "prob_annih"), w + " prob_annih");
          cmp(b_par, par.cof_nuclear_destruction, cc.d(ci, "cof_nd"), w + " cof_nd");
          cmp(b_par, par.cof_nuclear_destruction_pr, cc.d(ci, "cof_nd_pr"), w + " cof_nd_pr");
          cmp(b_par, par.r2_of_nuclear_destruction, cc.d(ci, "r2_nd"), w + " r2_nd");
          cmp(b_par, par.dof_nuclear_destruction, cc.d(ci, "dof_nd"), w + " dof_nd");
          cmp(b_par, par.pt2_of_nuclear_destruction, cc.d(ci, "pt2_nd"), w + " pt2_nd");
          cmp(b_par, par.max_pt2_of_nuclear_destruction, cc.d(ci, "maxpt2_nd"),
              w + " maxpt2_nd");
          cmp(b_par, par.excitation_energy_per_wounded_nucleon, cc.d(ci, "exc_per_wn"),
              w + " exc_per_wn");
          cmp(b_par, par.max_number_of_collisions, cc.d(ci, "max_ncoll"), w + " max_ncoll");
        }

        std::map<int, long long> got_nstrings, got_mass, got_species, got_mult, got_holes;
        std::map<int, long long> got_b, got_part, got_nn, got_exc;
        // Sums of SQUARES of the per-event bin counts, for the three per-object histograms.
        // They are what turns Var/Mean into a measured number rather than an assumed 1; see
        // `z_compare_counts`.
        std::map<int, long long> sq_mass, sq_species, sq_exc;
        std::map<int, long long> ev_mass, ev_species, ev_exc;
        std::map<int, double> got_sum_e, got_sq_e, got_sum_pz, got_sq_pz, got_sum_pt2,
            got_sq_pt2;
        double got_etot = 0.0, got_etot_sq = 0.0;
        long long sum_attempts = 0;
        long long n_ok = 0;
        // An ION is not in `data/ftf_hadrons.hh` and cannot be; its projectile is built from
        // (A, Z) and the oracle's own PDG mass.
        hadronic::xs::Projectile<double> proj;
        if (c.proj_a > 0) {
          proj.pdg = c.pdg;
          proj.mass = c.pmass;
          proj.charge = c.proj_z;
          proj.baryon_number = c.proj_a;
          proj.n_lambdas = 0;
        } else {
          proj = projectile_of(c.pdg);
        }
        const ftf::Vec4 primary(0.0, 0.0, c.plab, c.kin + c.pmass);
        for (int ev = 0; ev < n_events; ++ev) {
          Philox<double> rng(static_cast<uint32_t>(ev), 7u);
          const bool ok = ftf::ftf_scatter(ws, proj, primary, c.a, c.z, lund, rng);
          if (!ok) { continue; }
          ev_mass.clear();
          ev_species.clear();
          ev_exc.clear();
          ++got_nstrings[ws->model.n_strings];
          for (int i = 0; i < ws->model.n_strings; ++i) {
            const ftf::Vec4 m = ftf::ftf_string_4momentum(ws->model.strings[i]);
            const double mass = m.mag();
            int b = (mass > 0.0) ? static_cast<int>(10.0 * std::log10(mass)) : -1;
            if (b < -1) { b = -1; }
            if (b > 59) { b = 59; }
            ++got_mass[b];
            ++ev_mass[b];
            const int e = ws->model.strings[i].excited ? 1 : 0;
            ++got_exc[e];
            ++ev_exc[e];
          }
          {
            // `b` is in half-fermi bins, as the dump writes it; `participants` is
            // `A - NumberOfTargetSpectatorNucleons`, the counter BuildStrings decrements.
            int bb = static_cast<int>(2.0 * ws->model.participants.b_impact / 1e-12);
            if (bb < 0) { bb = 0; }
            if (bb > 79) { bb = 79; }
            ++got_b[bb];
            ++got_nn[ws->model.n_nn_collisions];
            ++got_part[c.a - ws->model.n_target_spectators];
          }
          ++n_ok;
          sum_attempts += ws->report.attempts;
          ++got_mult[ws->strings.n_out];
          double ev_etot = 0.0;
          for (int i = 0; i < ws->strings.n_out; ++i) {
            ev_etot += ws->strings.out[i].momentum.e;
            const int sp = ws->strings.out[i].pdg;
            ++got_species[sp];
            ++ev_species[sp];
            const ftf::Vec4& m = ws->strings.out[i].momentum;
            const double pt2 = m.v.x * m.v.x + m.v.y * m.v.y;
            got_sum_e[sp] += m.e;
            got_sq_e[sp] += m.e * m.e;
            got_sum_pz[sp] += m.v.z;
            got_sq_pz[sp] += m.v.z * m.v.z;
            got_sum_pt2[sp] += pt2;
            got_sq_pt2[sp] += pt2 * pt2;
          }
          got_etot += ev_etot;
          got_etot_sq += ev_etot * ev_etot;
          int nh = 0;
          for (int i = 0; i < ws->model.target.my_a; ++i) {
            if (ws->model.target.nucleons[i].hit) { ++nh; }
          }
          ++got_holes[nh];
          for (const auto& kv : ev_mass) { sq_mass[kv.first] += kv.second * kv.second; }
          for (const auto& kv : ev_species) { sq_species[kv.first] += kv.second * kv.second; }
          for (const auto& kv : ev_exc) { sq_exc[kv.first] += kv.second * kv.second; }
        }

        // The two sides ran different numbers of events; the z-test compares proportions, so
        // each count is scaled to the smaller N before the comparison.
        const long long n_min = (ref_n < n_events) ? ref_n : n_events;
        auto scale = [&](long long v, long long from) {
          return static_cast<long long>(static_cast<double>(v) * n_min / from + 0.5);
        };
        if (means) {
          auto mean_of = [](const std::map<int, long long>& h) {
            double s = 0.0, n = 0.0;
            for (const auto& kv : h) {
              s += static_cast<double>(kv.first) * kv.second;
              n += kv.second;
            }
            return (n > 0.0) ? s / n : 0.0;
          };
          std::printf("%-10s  nstr %7.3f/%7.3f  part %7.3f/%7.3f  nn %7.3f/%7.3f  "
                      "b/2fm %7.3f/%7.3f  mult %7.3f/%7.3f  holes %7.3f/%7.3f  att %6.3f\n",
                      c.name.c_str(), mean_of(got_nstrings), mean_of(ref_nstrings),
                      mean_of(got_part), mean_of(ref_part), mean_of(got_nn), mean_of(ref_nn),
                      mean_of(got_b), mean_of(ref_b), mean_of(got_mult), mean_of(ref_mult),
                      mean_of(got_holes), mean_of(ref_holes),
                      (n_ok > 0) ? static_cast<double>(sum_attempts) / n_ok : 0.0);
        }
        for (const auto& kv : ref_nstrings) {
          z_compare(zstats[z_nstrings], scale(got_nstrings[kv.first], n_events),
                    scale(kv.second, ref_n), n_min,
                    c.name + " nstrings " + std::to_string(kv.first));
        }
        for (const auto& kv : ref_mass) {
          z_compare_counts(zstats[z_smass], scale(got_mass[kv.first], n_events),
                           scale(kv.second, ref_n),
                           overdispersion_of(got_mass[kv.first], sq_mass[kv.first], n_events),
                           c.name + " logM " + std::to_string(kv.first));
        }
        for (const auto& kv : ref_species) {
          z_compare_counts(zstats[z_species], scale(got_species[kv.first], n_events),
                           scale(kv.second, ref_n),
                           overdispersion_of(got_species[kv.first], sq_species[kv.first],
                                             n_events),
                           c.name + " pdg " + std::to_string(kv.first));
        }
        for (const auto& kv : ref_mult) {
          z_compare(zstats[z_mult], scale(got_mult[kv.first], n_events), scale(kv.second, ref_n),
                    n_min, c.name + " mult " + std::to_string(kv.first));
        }
        for (const auto& kv : ref_holes) {
          z_compare(zstats[z_holes], scale(got_holes[kv.first], n_events),
                    scale(kv.second, ref_n), n_min,
                    c.name + " holes " + std::to_string(kv.first));
        }
        for (const auto& kv : ref_b) {
          z_compare(zstats[z_b], scale(got_b[kv.first], n_events), scale(kv.second, ref_n),
                    n_min, c.name + " b/2fm " + std::to_string(kv.first));
        }
        for (const auto& kv : ref_part) {
          z_compare(zstats[z_part], scale(got_part[kv.first], n_events),
                    scale(kv.second, ref_n), n_min,
                    c.name + " part " + std::to_string(kv.first));
        }
        for (const auto& kv : ref_nn) {
          z_compare(zstats[z_nn], scale(got_nn[kv.first], n_events), scale(kv.second, ref_n),
                    n_min, c.name + " nncoll " + std::to_string(kv.first));
        }
        for (const auto& kv : ref_exc) {
          z_compare_counts(zstats[z_exc], scale(got_exc[kv.first], n_events),
                           scale(kv.second, ref_n),
                           overdispersion_of(got_exc[kv.first], sq_exc[kv.first], n_events),
                           c.name + " excited " + std::to_string(kv.first));
        }
        // The energy balance, one comparison per case.
        {
          double ref_etot = 0.0;
          for (const auto& kv : ref_sum_e) { ref_etot += kv.second; }
          if (n_ok > 1) {
            z_mean_compare(zstats[z_etot], got_etot, got_etot_sq, n_ok, ref_etot, ref_n,
                           c.name + " E_total");
          }
        }
        // The per-species momentum moments. `sum/count` is the mean, and the z is the
        // difference of two means over the standard error of that difference, with the
        // variance measured on the port side (the same physics, so the same spread) and used
        // for both. Species with fewer than 200 entries on either side are skipped: a mean
        // over a handful of hadrons has an error bar the comparison cannot resolve.
        for (const auto& kv : ref_species) {
          const int sp = kv.first;
          const long long n_p = got_species[sp];
          const long long n_r = kv.second;
          if (n_p < 200 || n_r < 200) {
            ++zstats[z_mean_e].skipped;
            ++zstats[z_mean_pz].skipped;
            ++zstats[z_mean_pt2].skipped;
            continue;
          }
          z_mean_compare(zstats[z_mean_e], got_sum_e[sp], got_sq_e[sp], n_p, ref_sum_e[sp], n_r,
                         c.name + " <E> pdg " + std::to_string(sp));
          z_mean_compare(zstats[z_mean_pz], got_sum_pz[sp], got_sq_pz[sp], n_p, ref_sum_pz[sp],
                         n_r, c.name + " <pz> pdg " + std::to_string(sp));
          z_mean_compare(zstats[z_mean_pt2], got_sum_pt2[sp], got_sq_pt2[sp], n_p,
                         ref_sum_pt2[sp], n_r,
                         c.name + " <pt2> pdg " + std::to_string(sp));
        }
      }
      delete ws;
    }
  }

  // -------------------------------------------------------------------------------------------
  // 6b. ftf_nucstat.csv - P9's nucleus, measured the way FTF's geometry uses it
  //
  // `GetList` samples an impact parameter in a disc of radius `GetOuterRadius() + 2 fm` and then
  // asks nucleon by nucleon whether the TRANSVERSE distance is inside the interaction radius.
  // So the two numbers that decide a participant count are an extreme-value statistic and a
  // transverse second moment, and P9's own validation - radial histograms of |r| - constrains
  // neither tightly. This section is here because the ion cases disagreed and the question
  // "is it the model or the nucleus underneath it" has to be answerable.
  // -------------------------------------------------------------------------------------------
  const int z_outer = new_z("nucleus outer radius");
  const int z_rms_t = new_z("nucleus transverse RMS");
  const int z_rms_r = new_z("nucleus radial RMS");
  {
    Csv cn;
    if (cn.load(dir + "/ftf_nucstat.csv")) {
      static bic::Nucleon nucleons[250];
      static Vec3<double> smom[250];
      static double sfermi[250];
      static bic::NucleusSortEntry ssums[250];
      static double sflat[bic::kFlatBlock];
      bic::Nucleus3DScratch sc;
      sc.momentum = smom;
      sc.fermi_p = sfermi;
      sc.test_sums = ssums;
      sc.flat_block = sflat;
      sc.capacity = 250;

      std::map<std::string, int> seen;
      for (size_t r0 = 0; r0 < cn.rows.size(); ++r0) { seen[cn.s(r0, "nucleus")] = 1; }
      for (const auto& nk : seen) {
        const std::string& nname = nk.first;
        int a = 0, z = 0;
        long long ref_n = 0;
        std::map<int, long long> ref_outer, ref_rt, ref_rr;
        for (size_t r0 = 0; r0 < cn.rows.size(); ++r0) {
          if (cn.s(r0, "nucleus") != nname) { continue; }
          a = static_cast<int>(cn.i(r0, "a"));
          z = static_cast<int>(cn.i(r0, "z"));
          ref_n = cn.i(r0, "n_events");
          const std::string& q = cn.s(r0, "quantity");
          const int bin = static_cast<int>(cn.i(r0, "bin"));
          if (q == "outer_qfm") { ref_outer[bin] = cn.i(r0, "count"); }
          else if (q == "rms_t_dfm") { ref_rt[bin] = cn.i(r0, "count"); }
          else if (q == "rms_r_dfm") { ref_rr[bin] = cn.i(r0, "count"); }
        }
        if (ref_n == 0) { continue; }

        const int n_events = quick ? 2000 : 20000;
        std::map<int, long long> got_outer, got_rt, got_rr;
        bic::Nucleus3D nuc;
        nuc.nucleons = nucleons;
        nuc.capacity = 250;
        for (int ev = 0; ev < n_events; ++ev) {
          Philox<double> rng(static_cast<uint32_t>(ev), 31u);
          const bic::NucleusReport rep = bic::nucleus_init(nuc, sc, a, z, rng);
          if (rep.fatal()) { continue; }
          nuc.sort_nucleons_inc_z();
          int b = static_cast<int>(4.0 * nuc.outer_radius() / 1e-12);
          if (b < 0) { b = 0; }
          if (b > 199) { b = 199; }
          ++got_outer[b];
          double sx = 0.0, sr = 0.0;
          for (int i = 0; i < nuc.my_a; ++i) {
            const ftf::Vec3d& p = nuc.nucleons[i].position;
            sx += (p.x * p.x + p.y * p.y) / 1e-24;
            sr += g4gpu::mag2(p) / 1e-24;
          }
          int bt = static_cast<int>(10.0 * std::sqrt(sx / a));
          if (bt < 0) { bt = 0; }
          if (bt > 199) { bt = 199; }
          ++got_rt[bt];
          int br = static_cast<int>(10.0 * std::sqrt(sr / a));
          if (br < 0) { br = 0; }
          if (br > 199) { br = 199; }
          ++got_rr[br];
        }
        const long long n_min = (ref_n < n_events) ? ref_n : n_events;
        auto scale = [&](long long v, long long from) {
          return static_cast<long long>(static_cast<double>(v) * n_min / from + 0.5);
        };
        for (const auto& kv : ref_outer) {
          z_compare(zstats[z_outer], scale(got_outer[kv.first], n_events),
                    scale(kv.second, ref_n), n_min,
                    nname + " outer/4fm " + std::to_string(kv.first));
        }
        for (const auto& kv : ref_rt) {
          z_compare(zstats[z_rms_t], scale(got_rt[kv.first], n_events), scale(kv.second, ref_n),
                    n_min, nname + " rmsT/10fm " + std::to_string(kv.first));
        }
        for (const auto& kv : ref_rr) {
          z_compare(zstats[z_rms_r], scale(got_rr[kv.first], n_events), scale(kv.second, ref_n),
                    n_min, nname + " rmsR/10fm " + std::to_string(kv.first));
        }
        if (means) {
          auto mean_of = [](const std::map<int, long long>& h) {
            double s = 0.0, n = 0.0;
            for (const auto& kv : h) {
              s += static_cast<double>(kv.first) * kv.second;
              n += kv.second;
            }
            return (n > 0.0) ? s / n : 0.0;
          };
          std::printf("%-8s  outer/4fm %8.4f/%8.4f  rmsT/10fm %8.4f/%8.4f  "
                      "rmsR/10fm %8.4f/%8.4f\n",
                      nname.c_str(), mean_of(got_outer), mean_of(ref_outer), mean_of(got_rt),
                      mean_of(ref_rt), mean_of(got_rr), mean_of(ref_rr));
        }
      }
    }
  }

  // -------------------------------------------------------------------------------------------
  // 7. ftf_modelbig_*.csv - docs/RISK.md V88's rule, applied to the three worst rows
  //
  // V88: 20,000 events is the oracle's limit, and a row at 3 sigma there is either a
  // fluctuation or a real sub-percent difference - the two are not distinguishable from one
  // 20,000-event number. So the three cases that carried the worst z at 20,000 are re-run at
  // 200,000 on BOTH sides. A fluctuation comes back at a similar or smaller z; a real
  // difference comes back at roughly sqrt(10) times it.
  // -------------------------------------------------------------------------------------------
  const int z_big_nstr = new_z("200k nstrings");
  const int z_big_part = new_z("200k participants");
  const int z_big_nn = new_z("200k NN collisions");
  const int z_big_mult = new_z("200k multiplicity");
  {
    struct BigCase {
      const char* name;
      int pdg;
      double kin;
      int a, z;
    };
    const BigCase kBig[] = {{"n_Fe_10", 2112, 10000.0, 56, 26},
                            {"pip_Al_10", 211, 10000.0, 27, 13},
                            {"pim_C_10", -211, 10000.0, 12, 6}};
    Csv cs, cm;
    if (cs.load(dir + "/ftf_modelbig_strings.csv") && cm.load(dir + "/ftf_modelbig_mult.csv")) {
      using WS = ftf::FtfWorkspace<250, 64, 1024, 512, 256, 96>;
      WS* ws = new WS();
      const int n_events = quick ? 20000 : 200000;
      for (const BigCase& c : kBig) {
        std::map<int, long long> ref_nstr, ref_part, ref_nn, ref_mult;
        long long ref_n = 0;
        for (size_t r = 0; r < cs.rows.size(); ++r) {
          if (cs.s(r, "case") != c.name) { continue; }
          ref_n = cs.i(r, "n_events");
          const std::string& q = cs.s(r, "quantity");
          const int bin = static_cast<int>(cs.i(r, "bin"));
          if (q == "nstrings") { ref_nstr[bin] = cs.i(r, "count"); }
          else if (q == "participants") { ref_part[bin] = cs.i(r, "count"); }
          else if (q == "nncoll") { ref_nn[bin] = cs.i(r, "count"); }
        }
        for (size_t r = 0; r < cm.rows.size(); ++r) {
          if (cm.s(r, "case") != c.name) { continue; }
          ref_mult[static_cast<int>(cm.i(r, "multiplicity"))] = cm.i(r, "count");
        }
        if (ref_n == 0) {
          std::printf("FAIL: no 200k oracle rows for %s\n", c.name);
          ++fails;
          continue;
        }
        std::map<int, long long> got_nstr, got_part, got_nn, got_mult;
        const hadronic::xs::Projectile<double> proj = projectile_of(c.pdg);
        const data::FtfHadron* pd = data::ftf_find_hadron(c.pdg);
        const double p = std::sqrt(c.kin * (c.kin + 2.0 * pd->mass));
        const ftf::Vec4 primary(0.0, 0.0, p, c.kin + pd->mass);
        for (int ev = 0; ev < n_events; ++ev) {
          Philox<double> rng(static_cast<uint32_t>(ev), 11u);
          if (!ftf::ftf_scatter(ws, proj, primary, c.a, c.z, lund, rng)) { continue; }
          ++got_nstr[ws->model.n_strings];
          ++got_nn[ws->model.n_nn_collisions];
          ++got_part[c.a - ws->model.n_target_spectators];
          ++got_mult[ws->strings.n_out];
        }
        const long long n_min = (ref_n < n_events) ? ref_n : n_events;
        auto scale = [&](long long v, long long from) {
          return static_cast<long long>(static_cast<double>(v) * n_min / from + 0.5);
        };
        for (const auto& kv : ref_nstr) {
          z_compare(zstats[z_big_nstr], scale(got_nstr[kv.first], n_events),
                    scale(kv.second, ref_n), n_min,
                    std::string(c.name) + " nstrings " + std::to_string(kv.first));
        }
        for (const auto& kv : ref_part) {
          z_compare(zstats[z_big_part], scale(got_part[kv.first], n_events),
                    scale(kv.second, ref_n), n_min,
                    std::string(c.name) + " part " + std::to_string(kv.first));
        }
        for (const auto& kv : ref_nn) {
          z_compare(zstats[z_big_nn], scale(got_nn[kv.first], n_events),
                    scale(kv.second, ref_n), n_min,
                    std::string(c.name) + " nncoll " + std::to_string(kv.first));
        }
        for (const auto& kv : ref_mult) {
          z_compare(zstats[z_big_mult], scale(got_mult[kv.first], n_events),
                    scale(kv.second, ref_n), n_min,
                    std::string(c.name) + " mult " + std::to_string(kv.first));
        }
      }
      delete ws;
    }
  }

  // -------------------------------------------------------------------------------------------
  // Report
  // -------------------------------------------------------------------------------------------
  std::printf("\n%-36s %10s %14s  %s\n", "bucket", "points", "worst rel", "where");
  long long total = 0;
  for (const Bucket& b : buckets) {
    total += b.n;
    const bool bad = (b.worst > b.tol);
    if (bad) { ++fails; }
    std::printf("%-36s %10lld %14.3e%s %s\n", b.name, b.n, b.worst, bad ? " FAIL" : "",
                b.where.c_str());
    if (b.n == 0) {
      std::printf("FAIL: bucket %s compared nothing\n", b.name);
      ++fails;
    }
  }
  std::printf("\n%-36s %10s %8s %10s  %s\n", "statistic", "bins", "skipped", "worst z", "where");
  for (const ZStat& z : zstats) {
    total += z.bins;
    const bool bad = (z.worst > z.gate);
    if (bad) { ++fails; }
    std::printf("%-36s %10lld %8lld %10.2f%s %s\n", z.name, z.bins, z.skipped, z.worst,
                bad ? " FAIL" : "", z.where.c_str());
    if (z.bins == 0) {
      std::printf("FAIL: statistic %s compared nothing\n", z.name);
      ++fails;
    }
  }

  std::printf("\n%lld comparisons\n", total);
  std::printf("%s\n", fails == 0 ? "PASS test_ftf_model" : "FAIL test_ftf_model");
  delete lund;
  return fails == 0 ? 0 : 1;
}
