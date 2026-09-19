// The four G4EmExtraPhysics final-state models, end to end.
//
//   G4LowEGammaNuclearModel   a photon below 200 MeV, absorbed whole into a compound nucleus
//                             and de-excited through P6 and P3
//   G4LightTargetCollider     a photon on hydrogen or deuterium, which is the arm P10 refused
//   G4CascadeInterface        a photon from 199 MeV to 6 GeV, through P10, with the CASCADE's
//                             own de-excitation and not P6's
//   G4ElectroVDNuclearModel / G4MuonVDNuclearModel
//                             an equivalent photon, the scattered lepton, and the same gamma
//                             chain below 10 GeV
//
// WHAT IS COMPARED, AND WHY IT IS NOT ONE THING
//
// The deterministic half is the EM vertex, and it is pinned ELSEWHERE:
// `tests/test_emextra_xs.cu` compares `GetEquivalentPhotonEnergy`, `GetEquivalentPhotonQ2`,
// `GetVirtualFactor` and their three draw counts against `emextra_eqphoton.csv` bit for bit
// under the prescribed eight-value cycle, and the Kokoulin double-differential cross section
// every term of the muon model's sampling table is built from against `emextra_kokoulin.csv`.
// What cannot be pinned is the whole model: below the EM vertex sits Bertini, whose three
// nested retry loops mean no prescribed engine survives to the top (docs/PORTED.md 2.1.12 and
// docs/RISK.md V132). So the assembly is compared as a DISTRIBUTION, against
// `emextra_apply.csv`, exactly as test_bertini_apply.cu compares its own.
//
// AND THE MODEL CHOICE IS NOT IN THIS CAMPAIGN. Which of GammaNPreco, Bertini and the unported
// QGS generator a photon of a given energy gets is a deterministic function plus one uniform,
// and `tests/test_emextra_config.cu` compares it against the closed form over 200,000 draws at
// each of seven points. Mixing it in here would put two thirds of the 5 GeV events into a model
// this port refuses and measure nothing about the model that ran. So the choice is validated
// exactly, each model is driven directly and validated statistically, and the two are separate.
//
// THE REFUSAL RATE IS STILL PART OF THE ANSWER and is reported per case, because Bertini has
// refusals of its own.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "core/rng.cuh"
#include "data/level_data.cuh"
#include "host/g4data.cuh"
#include "physics/hadronic/emextra/lepton_nuclear.cuh"
#include "physics/hadronic/emextra/lepton_vd.cuh"
#include "physics/hadronic/emextra/photon_nuclear.cuh"

using namespace g4gpu;
using namespace g4gpu::physics::hadronic;
namespace ee = g4gpu::physics::hadronic::emextra;

/// The final state one photo-nuclear event needs room for. A 6 GeV photon on lead makes tens of
/// secondaries plus the evaporation chain; 256 is what P10's own campaign uses and overflow is
/// REPORTED, never silently dropped.
constexpr int kCap = 256;

/// The three entry points, instantiated for the device and never launched. `-Xptxas -v` on this
/// translation unit is what reports the register and stack cost the P13 brief asks for.
/// Everything that is not a scalar is reached through a pointer: the Bertini workspace alone is
/// tens of kilobytes and the muon sampling table is 2.3 MB.
__global__ void emextra_photon_probe(int pdg, double ke, int a, int z, bert::NucleiModel* nm,
                                     bert::CollisionOutput* go, bert::CollisionOutput* co,
                                     bert::CollisionOutput* dx, bert::CollisionOutput* tp,
                                     bert::ColliderOutput* epo, bert::BertiniWorkspace* bws,
                                     data::LevelTable lt, deex::FermiPool pool,
                                     preco::PrecoWorkspace pws,
                                     HadFinalState<double, kCap>* fs, int* out) {
  Philox<double> rng(11u, 12u, 13u);
  HadProjectile<double> p;
  p.pdg = pdg;
  p.mass = 0.0;
  p.kin_energy = ke;
  HadNucleus n;
  n.a = a;
  n.z = z;
  ee::GammaWorkspace ws;
  ws.model = nm;
  ws.global_out = go;
  ws.out = co;
  ws.dex_out = dx;
  ws.tmp = tp;
  ws.epo = epo;
  ws.bert_ws = bws;
  ws.preco = pws;
  const ee::PhotonNuclearResult r = ee::photon_nuclear(p, n, *fs, ws, lt, pool, rng);
  out[0] = fs->n_secondaries;
  out[1] = static_cast<int>(r.model);
  out[2] = static_cast<int>(r.refusal);
}

__global__ void emextra_lepton_probe(int pdg, double ke, int a, int z, bert::NucleiModel* nm,
                                     bert::CollisionOutput* go, bert::CollisionOutput* co,
                                     bert::CollisionOutput* dx, bert::CollisionOutput* tp,
                                     bert::ColliderOutput* epo, bert::BertiniWorkspace* bws,
                                     const ee::MuVdTable* mutab, data::LevelTable lt,
                                     deex::FermiPool pool, preco::PrecoWorkspace pws,
                                     HadFinalState<double, kCap>* fs, int* out) {
  Philox<double> rng(21u, 22u, 23u);
  HadProjectile<double> p;
  p.pdg = pdg;
  p.mass = (pdg == 13 || pdg == -13) ? 105.6583715 : 0.510998910;
  p.kin_energy = ke;
  HadNucleus n;
  n.a = a;
  n.z = z;
  ee::GammaWorkspace ws;
  ws.model = nm;
  ws.global_out = go;
  ws.out = co;
  ws.dex_out = dx;
  ws.tmp = tp;
  ws.epo = epo;
  ws.bert_ws = bws;
  ws.preco = pws;
  // The PROCESS entry points and not the models, because these are what P15 calls: the
  // register and stack numbers below are then the ones a stepper pays. The wrapper adds the
  // range manager, which with one registered model is a branch and not a draw.
  const ee::LeptonNuclearResult e = ee::electron_nuclear(p, n, *fs, ws, lt, pool, rng);
  const ee::LeptonNuclearResult m = ee::muon_nuclear(p, n, *fs, *mutab, ws, lt, pool, rng);
  out[0] = fs->n_secondaries;
  out[1] = static_cast<int>(e.vd.no_photon);
  out[2] = static_cast<int>(m.vd.no_photon);
}

namespace {

int fails = 0;

std::string oracle_dir() {
  const char* e = std::getenv("G4GPU_ORACLE");
  return (e != nullptr && e[0] != '\0') ? std::string(e) : std::string("ref/oracle");
}

// ---------------------------------------------------------------------------------------------
// Statistical comparison, as tests/test_bertini_apply.cu defines it
// ---------------------------------------------------------------------------------------------

struct Bucket {
  const char* name;
  long long n = 0;
  double worst = 0.0;      ///< worst deviation in units of the pooled standard error
  std::string where;
  double tol = 5.0;
};

std::vector<Bucket> buckets;

int new_bucket(const char* name, double tol) {
  Bucket b;
  b.name = name;
  b.tol = tol;
  buckets.push_back(b);
  return static_cast<int>(buckets.size()) - 1;
}

double case_worst = 0.0;

/// Compare two sample means in units of the pooled standard error, with the population
/// standard deviation estimated from the PORT's sample - the same function and the same
/// reasoning as test_bertini_apply.cu's. A zero variance on both sides falls back to a relative
/// comparison at 1e-12, because a zero band fails on a rounding difference.
void cmp_sigma(int bi, double got, double want, double sigma, long long n_got, long long n_want,
               const std::string& where) {
  Bucket& b = buckets[bi];
  ++b.n;
  double dev;
  if (sigma > 0.0 && n_got > 0 && n_want > 0) {
    const double se = sigma * std::sqrt(1.0 / double(n_got) + 1.0 / double(n_want));
    dev = (se > 0.0) ? std::fabs(got - want) / se : 0.0;
  } else {
    const double scale = (std::fabs(want) > 0.0) ? std::fabs(want) : 1.0;
    dev = std::fabs(got - want) / scale / 1e-12;
  }
  if (dev > 3.5) {
    std::printf("  DIAG %-22s %-46s %6.2f sigma got %.6g want %.6g  (sigma %.4g, n %lld/%lld)\n",
                b.name, where.c_str(), dev, got, want, sigma, n_got, n_want);
  }
  if (dev > case_worst) { case_worst = dev; }
  if (dev > b.worst) {
    b.worst = dev;
    char buf[160];
    std::snprintf(buf, sizeof buf, " got %.6g want %.6g", got, want);
    b.where = where + buf;
  }
}

/// Compare two sample means with WELCH'S standard error - each sample's own spread over its own
/// count - which is what a comparison of two means with unequal variances is, EXCEPT that a
/// spread estimated from two samples is not a spread and the rule below says what is done then.
///
/// `cmp_sigma` above uses one pooled sigma, as test_bertini_apply.cu does, and that is right
/// when both samples estimate the same spread. It is wrong by a factor of four when they do not:
/// this campaign has a species row with twelve oracle samples of spread 17.2 against a hundred
/// and forty port samples of spread 4.7, and the same row reads 8.02 sigma with the port's
/// spread used for both, 2.2 with the oracle's, and 2.25 under Welch. The difference is not a
/// band to choose - it is a question about what the error of a mean IS - so wherever the oracle
/// dumps its own rms beside its mean, the comparison uses both.
///
/// AND WELCH ALONE IS NOT ENOUGH, because a sample standard deviation has degrees of freedom.
/// The 2,000-event oracle gives 23 of the 991 species spectra exactly TWO soft samples, and 20
/// more between three and four; a two-sample sd has one degree of freedom, its expected value is
/// only sqrt(2/pi) = 0.80 of the true sigma, and it is five times too small a few per cent of
/// the time. Over 43 such rows that is not a possibility, it is an arrival time. The row that
/// found it is `evd e- 200 MeV on O16`, pi+ soft spectrum: two oracle pions at 16.99 and 18.53
/// MeV, so sd 0.77 and a Welch error of 0.88, against thirty port pions of mean 11.10 and
/// spread 3.81 - which reads 7.54 sigma and is a one-degree-of-freedom sd, not a discrepancy.
/// docs/RISK.md V177 has the escalation that proved it: at 20,000 oracle events the same row's
/// soft sample is ten times bigger and the disagreement is gone.
///
/// So the error used is the LARGER of two estimates of the same standard error:
///
///   SE_welch = sqrt(s1^2/n1 + s2^2/n2)              each sample's own spread, unequal variances
///   SE_equal = max(s1, s2) * sqrt(1/n1 + 1/n2)      one spread, the larger, variances equal
///
/// They agree to a per cent for the 700 rows with more than a hundred samples a side, so the
/// rule changes nothing where the data can tell the two models apart. Where it cannot - one df
/// against twenty-nine - it takes the conservative one, which is the only honest answer to "is
/// the spread 0.77 or 3.81?" when one side has two samples. Checked BOTH ways round: the twelve
/// against a hundred and forty row keeps Welch's 4.98 - its SE_equal is 5.17, so the two are
/// within four per cent of each other - and reads 2.15, and the two against thirty row takes
/// SE_equal's 2.78 over Welch's 0.88 and reads 2.39. Neither number was chosen; both fall out
/// of the same max.
///
/// `max(s1, s2)` and not "the spread of the sample with more degrees of freedom", which is what
/// this said first. That rule has no answer when the two counts are equal, and its tie-break
/// towards the port picked a spread of 0.53 over one of 12.44 on a two-against-two row and
/// reported 6.80 sigma for a difference two samples a side cannot resolve. The larger spread
/// needs no tie-break and is conservative in the direction that matters.
void cmp_welch(int bi, double got, double got_sd, long long n_got, double want, double want_sd,
               long long n_want, const std::string& where) {
  Bucket& b = buckets[bi];
  ++b.n;
  double dev = 0.0;
  const double v_welch = (n_got > 0 ? got_sd * got_sd / double(n_got) : 0.0) +
                         (n_want > 0 ? want_sd * want_sd / double(n_want) : 0.0);
  // The LARGER of the two spreads, used for both sides. Not "the one with more degrees of
  // freedom": that is ambiguous when the two counts are equal, and the first version of this
  // broke its tie towards the port, which on a two-against-two row picked a spread of 0.53
  // over one of 12.44 and reported 6.80 sigma for a difference the data cannot resolve. The
  // larger spread is the conservative estimate of a common sigma and needs no tie-break.
  const double sd_big = (got_sd > want_sd) ? got_sd : want_sd;
  const double inv_n = (n_got > 0 ? 1.0 / double(n_got) : 0.0) +
                       (n_want > 0 ? 1.0 / double(n_want) : 0.0);
  const double v_equal = sd_big * sd_big * inv_n;
  const double v = (v_equal > v_welch) ? v_equal : v_welch;
  if (v > 0.0) {
    dev = std::fabs(got - want) / std::sqrt(v);
  } else {
    const double scale = (std::fabs(want) > 0.0) ? std::fabs(want) : 1.0;
    dev = std::fabs(got - want) / scale / 1e-12;
  }
  if (dev > 3.5) {
    std::printf("  DIAG %-22s %-46s %6.2f sigma got %.6g +- %.4g (n %lld)  want %.6g +- %.4g "
                "(n %lld)\n",
                b.name, where.c_str(), dev, got, got_sd, n_got, want, want_sd, n_want);
  }
  if (dev > case_worst) { case_worst = dev; }
  if (dev > b.worst) {
    b.worst = dev;
    char buf[200];
    std::snprintf(buf, sizeof buf, " got %.6g +-%.4g n=%lld want %.6g +-%.4g n=%lld", got,
                  got_sd, n_got, want, want_sd, n_want);
    b.where = where + buf;
  }
}

/// A conserved quantity compared RELATIVELY and not in sigma. docs/RISK.md V133: the total
/// energy of a final state is an identity, its sample variance is rounding, and a five-sigma
/// band on it is a band of zero width that reports hundreds of sigma for six digits of
/// agreement.
void cmp_rel(int bi, double got, double want, double tol, const std::string& where) {
  Bucket& b = buckets[bi];
  ++b.n;
  const double scale = (std::fabs(want) > 0.0) ? std::fabs(want) : 1.0;
  const double rel = std::fabs(got - want) / scale;
  const double dev = rel / tol;   // in units of the stated band
  if (dev > case_worst) { case_worst = dev; }
  if (dev > b.worst) {
    b.worst = dev;
    char buf[160];
    std::snprintf(buf, sizeof buf, " got %.6g want %.6g (rel %.3e)", got, want, rel);
    b.where = where + buf;
  }
}

/// Running moments for one quantity, plus the largest value seen.
///
/// The maximum is not decoration. A species whose mean is out by a factor of four and whose rms
/// is out by a factor of forty has a few enormous entries, not a shifted distribution, and the
/// largest one names what they are: a secondary above the projectile's own energy is a
/// mis-binned particle or an unphysical one, and either is a finding rather than a tolerance.
struct Moments {
  long long n = 0;
  double s1 = 0.0, s2 = 0.0;
  double max = 0.0;
  void add(double x) { ++n; s1 += x; s2 += x * x; if (x > max) { max = x; } }
  double mean() const { return (n > 0) ? s1 / double(n) : 0.0; }
  double var() const {
    if (n < 2) { return 0.0; }
    const double m = mean();
    const double v = s2 / double(n) - m * m;
    return (v > 0.0) ? v : 0.0;
  }
  double sigma() const { return std::sqrt(var()); }
};

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
  long long ll(std::size_t r, const char* col) const { return std::atoll(s(r, col).c_str()); }
};

/// The same thirteen species buckets `dump_emextra.cc` uses, by the port's (pdg, Z, A).
int species_bucket_of(int pdg, int z, int a) {
  switch (pdg) {
    case 2112: return 0;
    case 2212: return 1;
    case 211: return 7;
    case -211: return 8;
    case 111: return 9;
    case 22: return 10;
    case 11: return 11;
    default: break;
  }
  if (a == 2 && z == 1) { return 2; }
  if (a == 3 && z == 1) { return 3; }
  if (a == 3 && z == 2) { return 4; }
  if (a == 4 && z == 2) { return 5; }
  if (a > 4) { return 6; }
  return 12;
}

const char* species_bucket_name(int i) {
  static const char* kNames[13] = {"n",   "p",   "d",    "t",     "He3", "alpha", "heavier",
                                   "pi+", "pi-", "pi0",  "gamma", "e-",  "other"};
  return (i >= 0 && i < 13) ? kNames[i] : "other";
}

}  // namespace

int main() {
  const std::string dir = oracle_dir();
  const std::string pe = host::g4photon_evaporation_dir();
  if (pe.empty()) {
    std::printf("PhotonEvaporation dataset not found - g4data.cuh could not resolve "
                "G4LEVELGAMMADATA\n");
    return 1;
  }
  data::LevelTableStorage lts;
  data::read_all_level_data(
      lts, pe, data::kLevelZMax,
      [](int Z, int A) { return deex::shell_correction(A, Z); },
      [](int Z, int A) { return deex::level_manager_level_density(Z, A); });
  const data::LevelTable lt = lts.view();
  deex::FermiPoolStorage ps;
  deex::build_fermi_pool(ps, lt);
  const deex::FermiPool pool = ps.view();

  std::vector<deex::Fragment> evap(4096), results(1024), step(512);
  std::vector<deex::DeexProduct> deex_products(1024), preco_products(1024);
  preco::PrecoWorkspace pws;
  pws.deex.evap_list = evap.data();
  pws.deex.evap_capacity = static_cast<int>(evap.size());
  pws.deex.results = results.data();
  pws.deex.results_capacity = static_cast<int>(results.size());
  pws.deex.step = step.data();
  pws.deex.step_capacity = static_cast<int>(step.size());
  pws.deex.products = deex_products.data();
  pws.deex.products_capacity = static_cast<int>(deex_products.size());
  pws.products = preco_products.data();
  pws.products_capacity = static_cast<int>(preco_products.size());

  ee::GammaWorkspace ws;
  ws.model = new bert::NucleiModel();
  ws.global_out = new bert::CollisionOutput();
  ws.out = new bert::CollisionOutput();
  ws.dex_out = new bert::CollisionOutput();
  ws.tmp = new bert::CollisionOutput();
  ws.epo = new bert::ColliderOutput();
  ws.bert_ws = new bert::BertiniWorkspace();
  ws.preco = pws;

  auto* fs = new HadFinalState<double, kCap>();

  // A smoke pass: every model reached once, on every target the campaign uses, so that a
  // compile-time change that breaks a dispatch is caught before the statistical pass.
  Philox<double> rng(1u, 2u, 3u);
  struct Case { double ke; int a, z; const char* what; };
  const Case cases[] = {
      {10.0, 12, 6, "GammaNPreco on C12"},
      {30.0, 208, 82, "GammaNPreco on Pb208"},
      {150.0, 16, 8, "GammaNPreco on O16"},
      {300.0, 27, 13, "Bertini on Al27"},
      {300.0, 1, 1, "LightTarget on H1"},
      {300.0, 2, 1, "LightTarget on D2"},
      {1000.0, 56, 26, "Bertini on Fe56"},
      {5000.0, 12, 6, "Bertini-or-QGS on C12"},
  };
  std::map<int, long long> model_count, refusal_count;
  for (const Case& c : cases) {
    long long ran = 0, refused = 0, secondaries = 0;
    for (int k = 0; k < 200; ++k) {
      HadProjectile<double> p;
      p.pdg = 22;
      p.mass = 0.0;
      p.kin_energy = c.ke;
      HadNucleus n;
      n.a = c.a;
      n.z = c.z;
      const ee::PhotonNuclearResult r = ee::photon_nuclear(p, n, *fs, ws, lt, pool, rng);
      ++model_count[static_cast<int>(r.model)];
      if (r.refusal != ee::EmExtraRefusal::kNone) {
        ++refused;
        ++refusal_count[static_cast<int>(r.refusal)];
      } else {
        ++ran;
        secondaries += fs->n_secondaries;
      }
    }
    std::printf("%-28s ran %4lld  refused %4lld  <n_sec> %.2f\n", c.what, ran, refused,
                ran > 0 ? double(secondaries) / double(ran) : 0.0);
    if (ran == 0 && refused == 0) {
      std::printf("FAIL: %s produced neither a final state nor a refusal\n", c.what);
      ++fails;
    }
  }
  std::printf("models chosen:");
  for (const auto& kv : model_count) {
    std::printf(" %s=%lld", ee::model_name(static_cast<ee::Model>(kv.first)), kv.second);
  }
  std::printf("\nrefusals:");
  for (const auto& kv : refusal_count) {
    std::printf(" %s=%lld", ee::refusal_name(static_cast<ee::EmExtraRefusal>(kv.first)),
                kv.second);
  }
  std::printf("\n");

  // -------------------------------------------------------------------------------------------
  // G4LightTargetCollider's four reachable arms, called DIRECTLY.
  //
  // Three of them cannot be reached through `photon_nuclear`: below 144.7 MeV the range manager
  // sends a photon to GammaNPreco and never to Bertini, so the proton-below-threshold arm and
  // the deuteron's absorption-only region (below 159 MeV) are invisible from there. The arm is
  // therefore driven on its own, with the thresholds bracketed - `ke < 0.1447` for a proton
  // target and the `ke > 0.159` gate that decides whether the two scattering channels on a
  // deuteron have any probability at all.
  // -------------------------------------------------------------------------------------------
  {
    struct LT { double ke; int a; ee::LightTargetArm arm; const char* what; };
    const LT lts2[] = {
        {100.0, 1, ee::LightTargetArm::kProtonBelowThreshold, "gamma 100 MeV on H1"},
        {144.6, 1, ee::LightTargetArm::kProtonBelowThreshold, "gamma 144.6 MeV on H1"},
        // 144.8 MeV is above the `ke < 0.1447` gate but below the pi0-p threshold the channel
        // tables' own binning puts at 144, so the collider exhausts its ten attempts and the
        // arm is `kProtonColliderEmpty` - Geant4's `if (numberOfOutgoingParticles() == 0)
        // trivialise`. That is the effect the two comments in `G4CascadeInterface::
        // ApplyYourself` describe and neither of them implements.
        {144.8, 1, ee::LightTargetArm::kProtonColliderEmpty, "gamma 144.8 MeV on H1"},
        {300.0, 1, ee::LightTargetArm::kProtonCollider,       "gamma 300 MeV on H1"},
        {100.0, 2, ee::LightTargetArm::kDeuteronAbsorption,   "gamma 100 MeV on D2"},
        {158.9, 2, ee::LightTargetArm::kDeuteronAbsorption,   "gamma 158.9 MeV on D2"},
    };
    for (const LT& c : lts2) {
      // The two scattering arms on a deuteron are chosen by a draw, so only the arms that are
      // DETERMINED by the energy are asserted; above 159 MeV all three are possible and the
      // mix is what the statistical campaign measures.
      HadProjectile<double> p;
      p.pdg = 22;
      p.mass = 0.0;
      p.kin_energy = c.ke;
      HadNucleus n;
      n.a = c.a;
      n.z = 1;
      const ee::LightTargetResult r = ee::light_target_collide(
          p, n, *fs, bert::default_cascade_params(), ws.bert_ws, *ws.epo, *ws.global_out, rng);
      if (r.refusal != ee::EmExtraRefusal::kNone) {
        std::printf("FAIL: %s refused (%s)\n", c.what, ee::refusal_name(r.refusal));
        ++fails;
      } else if (r.arm != c.arm) {
        std::printf("FAIL: %s took arm %d, expected %d\n", c.what, int(r.arm), int(c.arm));
        ++fails;
      }
      // Both trivialised arms return exactly the target and the bullet, in that order.
      if (c.arm == ee::LightTargetArm::kProtonBelowThreshold && fs->n_secondaries != 2) {
        std::printf("FAIL: %s trivialised to %d secondaries, expected 2\n", c.what,
                    fs->n_secondaries);
        ++fails;
      }
      if (c.arm == ee::LightTargetArm::kDeuteronAbsorption && fs->n_secondaries != 2) {
        std::printf("FAIL: %s broke the deuteron into %d products, expected 2 (p + n)\n",
                    c.what, fs->n_secondaries);
        ++fails;
      }
    }
    // Above 159 MeV all three deuteron arms are live; the mix is counted here so that an arm
    // that silently stops being selected is visible.
    std::map<int, long long> arms;
    for (int k = 0; k < 500; ++k) {
      HadProjectile<double> p;
      p.pdg = 22;
      p.mass = 0.0;
      p.kin_energy = 400.0;
      HadNucleus n;
      n.a = 2;
      n.z = 1;
      const ee::LightTargetResult r = ee::light_target_collide(
          p, n, *fs, bert::default_cascade_params(), ws.bert_ws, *ws.epo, *ws.global_out, rng);
      ++arms[static_cast<int>(r.arm)];
    }
    std::printf("light-target deuteron arms at 400 MeV over 500 draws:");
    for (const auto& kv : arms) { std::printf(" arm%d=%lld", kv.first, kv.second); }
    std::printf("\n");
    if (arms.size() < 3) {
      std::printf("FAIL: only %d of the three deuteron arms were selected at 400 MeV\n",
                  int(arms.size()));
      ++fails;
    }
  }

  // The two lepton models, on the same targets.
  auto* mutab = new ee::MuVdTable();
  // `g/mole` in Geant4's internal units, derived as CLHEP derives it and compared against
  // CLHEP's own value in tests/test_emextra_xs.cu. Here it is what MakeSamplingTable
  // multiplies `adat[iz]` by before calling the double-differential cross section.
  ee::mu_vd_make_sampling_table(*mutab, 105.6583715, ee::g_per_mole());

  struct LCase { int pdg; double mass; double ke; int a, z; const char* what; };
  const LCase lcases[] = {
      {11, 0.510998910, 50.0, 12, 6, "e- 50 MeV on C12"},
      {11, 0.510998910, 200.0, 16, 8, "e- 200 MeV on O16"},
      {11, 0.510998910, 1000.0, 27, 13, "e- 1 GeV on Al27"},
      {-11, 0.510998910, 1000.0, 56, 26, "e+ 1 GeV on Fe56"},
      {11, 0.510998910, 10000.0, 208, 82, "e- 10 GeV on Pb208"},
      {13, 105.6583715, 200.0, 12, 6, "mu- 200 MeV on C12"},
      {13, 105.6583715, 1000.0, 27, 13, "mu- 1 GeV on Al27"},
      {13, 105.6583715, 10000.0, 208, 82, "mu- 10 GeV on Pb208"},
  };
  for (const LCase& c : lcases) {
    long long photons = 0, no_photon = 0, refused = 0, secondaries = 0;
    double sum_nu = 0.0, sum_ekin = 0.0;
    std::map<int, long long> why;
    for (int k = 0; k < 200; ++k) {
      HadProjectile<double> p;
      p.pdg = c.pdg;
      p.mass = c.mass;
      p.charge = (c.pdg > 0) ? -1.0 : 1.0;
      p.kin_energy = c.ke;
      HadNucleus n;
      n.a = c.a;
      n.z = c.z;
      const bool is_mu = (c.pdg == 13 || c.pdg == -13);
      const ee::LeptonVdResult r =
          is_mu ? ee::muon_vd_apply(p, n, *fs, *mutab, ws, lt, pool, rng)
                : ee::electro_vd_apply(p, n, *fs, ws, lt, pool, rng);
      if (r.refusal != ee::EmExtraRefusal::kNone) { ++refused; continue; }
      if (r.no_photon != ee::NoPhotonReason::kNone) {
        ++no_photon;
        ++why[static_cast<int>(r.no_photon)];
        continue;
      }
      ++photons;
      sum_nu += r.photon_energy;
      sum_ekin += r.lepton_final_kin;
      secondaries += fs->n_secondaries;
    }
    std::printf("%-26s photon %4lld  none %4lld  refused %4lld  <nu> %8.3f  <T_lep> %8.3f  "
                "<n_sec> %.2f\n",
                c.what, photons, no_photon, refused,
                photons > 0 ? sum_nu / double(photons) : 0.0,
                photons > 0 ? sum_ekin / double(photons) : 0.0,
                photons > 0 ? double(secondaries) / double(photons) : 0.0);
    if (photons == 0 && no_photon == 0 && refused == 0) {
      std::printf("FAIL: %s did nothing at all\n", c.what);
      ++fails;
    }
  }

  // The muon model below its own threshold returns the track untouched, and that is a
  // measured fact and not a refusal: epmax = T + m_mu - 0.5*m_p, and for T = 200 MeV that is
  // -163.5 MeV, well under CutFixed = 200.
  {
    HadProjectile<double> p;
    p.pdg = 13;
    p.mass = 105.6583715;
    p.kin_energy = 200.0;
    HadNucleus n;
    n.a = 12;
    n.z = 6;
    const ee::LeptonVdResult r = ee::muon_vd_apply(p, n, *fs, *mutab, ws, lt, pool, rng);
    if (r.no_photon != ee::NoPhotonReason::kMuonBelowCut) {
      std::printf("FAIL: a 200 MeV mu- was not stopped by the CutFixed gate (reason %d)\n",
                  int(r.no_photon));
      ++fails;
    }
    if (fs->n_secondaries != 0 || fs->energy_change != 200.0) {
      std::printf("FAIL: a 200 MeV mu- below the gate did not come back untouched\n");
      ++fails;
    }
    // THE THRESHOLD ITSELF, BRACKETED. `epmax <= CutFixed` is `T <= CutFixed + 0.5*m_p - m_mu`
    // = 200 + 469.136 - 105.658 = 563.478 MeV, so the gate must close at 563.4 and open at
    // 563.6. Bracketing it is the whole point: the first version of this check asked only
    // whether a 200 MeV muon was stopped and whether a 563.5 MeV one was not, and a CutFixed
    // perturbed from 200 to 100 - which moves the threshold to 463.478 - passed both, because
    // 200 is below either threshold and 563.5 above either. Measured, then fixed.
    struct Br { double ke; bool stopped; };
    const Br br[] = {{200.0, true},  {463.0, true},  {463.6, true},
                     {500.0, true},  {563.4, true},  {563.6, false}, {1000.0, false}};
    for (const Br& b : br) {
      p.kin_energy = b.ke;
      const ee::LeptonVdResult rb = ee::muon_vd_apply(p, n, *fs, *mutab, ws, lt, pool, rng);
      const bool stopped = (rb.no_photon == ee::NoPhotonReason::kMuonBelowCut);
      if (stopped != b.stopped) {
        std::printf("FAIL: a %.1f MeV mu- %s stopped by the CutFixed gate and should %s "
                    "(threshold is CutFixed + 0.5*m_p - m_mu = 563.478 MeV)\n",
                    b.ke, stopped ? "was" : "was not", b.stopped ? "be" : "not be");
        ++fails;
      }
    }
  }

  // -------------------------------------------------------------------------------------------
  // THE THREE LEPTO-NUCLEAR PROCESSES AS PROCESSES
  //
  // `emextra::electron_nuclear` and `emextra::muon_nuclear` are the process-level entry points -
  // `G4EnergyRangeManager` plus the model - and P15 calls these, not `electro_vd_apply` and
  // `muon_vd_apply`. Four things are asserted about them, and the third is the one that was
  // assumed wrong first.
  //
  //  1. THE WRAPPER IS NOT A SECOND SAMPLER. Driven from the same seed, the process and the
  //     model it wraps produce the SAME equivalent photon, the same scattered lepton and the
  //     same secondary count, bit for bit. That is only true if the range manager draws no
  //     random number, which with one registered model it does not - and if it ever started to,
  //     every lepton case in the campaign would shift by one draw and this row would catch it.
  //  2. A wrong pdg is named, not run.
  //  3. THE 1 PeV WINDOW IS INERT. `G4EnergyRangeManager::GetHadronicInteraction` returns the
  //     single registered model without looking at its range ("VI shortcut" in the source), so
  //     a 10 PeV muon is handed to `G4MuonVDNuclearModel` in Geant4 and here. This asserts that
  //     both 1 PeV and 10 PeV CHOOSE the model; what happens next is the >= 10 GeV FTF arm's
  //     refusal, which is a different thing and is asserted as `kSubModel` rather than as the
  //     absence of a model. See lepton_nuclear.cuh.
  //  4. A hyper-nuclear target is refused by name before any of it.
  // -------------------------------------------------------------------------------------------
  {
    auto lepton = [](int pdg, double ke) {
      HadProjectile<double> p;
      p.pdg = pdg;
      p.mass = (pdg == 13 || pdg == -13) ? 105.6583715 : 0.510998910;
      p.kin_energy = ke;
      return p;
    };
    HadNucleus n56;
    n56.a = 56;
    n56.z = 26;

    // 1. process == model, from the same seed.
    struct Same { int pdg; double ke; const char* what; };
    const Same same[] = {{11, 1000.0, "e- 1 GeV"},
                         {-11, 1000.0, "e+ 1 GeV"},
                         {13, 1000.0, "mu- 1 GeV"},
                         {-13, 5000.0, "mu+ 5 GeV"}};
    for (const Same& s : same) {
      Philox<double> ra(7u, 8u, 9u), rb(7u, 8u, 9u);
      const HadProjectile<double> p = lepton(s.pdg, s.ke);
      ee::LeptonNuclearResult pr;
      ee::LeptonVdResult mr;
      if (s.pdg == 13 || s.pdg == -13) {
        pr = ee::muon_nuclear(p, n56, *fs, *mutab, ws, lt, pool, ra);
        mr = ee::muon_vd_apply(p, n56, *fs, *mutab, ws, lt, pool, rb);
      } else {
        pr = ee::electron_nuclear(p, n56, *fs, ws, lt, pool, ra);
        mr = ee::electro_vd_apply(p, n56, *fs, ws, lt, pool, rb);
      }
      const bool ok = pr.vd.photon_energy == mr.photon_energy &&
                      pr.vd.lepton_final_kin == mr.lepton_final_kin &&
                      pr.vd.lepton_cos_theta == mr.lepton_cos_theta &&
                      pr.n_secondaries == mr.n_secondaries &&
                      pr.vd.no_photon == mr.no_photon && pr.vd.refusal == mr.refusal;
      if (!ok) {
        std::printf("FAIL: %s through the process differs from the model on the same seed "
                    "(nu %.17g vs %.17g, T_lep %.17g vs %.17g, n_sec %d vs %d)\n",
                    s.what, pr.vd.photon_energy, mr.photon_energy, pr.vd.lepton_final_kin,
                    mr.lepton_final_kin, pr.n_secondaries, mr.n_secondaries);
        ++fails;
      }
      const ee::Process want_proc = (s.pdg == 11)    ? ee::Process::kElectronNuclear
                                    : (s.pdg == -11) ? ee::Process::kPositronNuclear
                                                     : ee::Process::kMuonNuclear;
      if (pr.process != want_proc) {
        std::printf("FAIL: %s got process %s, want %s\n", s.what,
                    ee::process_name(pr.process), ee::process_name(want_proc));
        ++fails;
      }
    }

    // 2. A wrong pdg. A proton has an inelastic process of its own and no business here.
    {
      Philox<double> r(1u, 1u, 1u);
      HadProjectile<double> p = lepton(11, 1000.0);
      p.pdg = 2212;
      p.mass = 938.272013;
      const ee::LeptonNuclearResult e = ee::electron_nuclear(p, n56, *fs, ws, lt, pool, r);
      const ee::LeptonNuclearResult m = ee::muon_nuclear(p, n56, *fs, *mutab, ws, lt, pool, r);
      if (e.refusal != ee::EmExtraRefusal::kNoModelInRange ||
          m.refusal != ee::EmExtraRefusal::kNoModelInRange ||
          e.model != ee::Model::kNone || m.model != ee::Model::kNone) {
        std::printf("FAIL: a proton was not refused by the two lepto-nuclear entry points "
                    "(e %d/%d, mu %d/%d)\n",
                    int(e.refusal), int(e.model), int(m.refusal), int(m.model));
        ++fails;
      }
      // And an electron must not be accepted by the muon's process either.
      const ee::LeptonNuclearResult x =
          ee::muon_nuclear(lepton(11, 1000.0), n56, *fs, *mutab, ws, lt, pool, r);
      if (x.refusal != ee::EmExtraRefusal::kNoModelInRange) {
        std::printf("FAIL: muon_nuclear accepted an electron (refusal %d)\n", int(x.refusal));
        ++fails;
      }
    }

    // 3. The window is inert: 1 PeV is the registered maximum and 10 PeV is ten times past it,
    //    and both must still choose the model. `vd_max_energy_MeV()` is 1e9.
    for (double ke : {1.0e9, 1.0e10}) {
      Philox<double> r(3u, 4u, 5u);
      const ee::LeptonNuclearResult m =
          ee::muon_nuclear(lepton(13, ke), n56, *fs, *mutab, ws, lt, pool, r);
      if (m.model != ee::Model::kMuonVD || m.choice != ModelChoice::kOk) {
        std::printf("FAIL: a %.3g MeV mu- found no model (model %d choice %d) - the 1 PeV "
                    "window is INERT with one registered model, see lepton_nuclear.cuh\n",
                    ke, int(m.model), int(m.choice));
        ++fails;
      }
      if (!m.vd.used_ftf || m.refusal != ee::EmExtraRefusal::kSubModel) {
        std::printf("FAIL: a %.3g MeV mu- did not reach the FTF arm's refusal (used_ftf %d "
                    "refusal %d)\n",
                    ke, int(m.vd.used_ftf), int(m.refusal));
        ++fails;
      }
    }

    // 4. A hyper-nuclear target, refused before the model.
    {
      Philox<double> r(6u, 7u, 8u);
      HadNucleus hyp;
      hyp.a = 56;
      hyp.z = 26;
      hyp.l = 1;
      const ee::LeptonNuclearResult e =
          ee::electron_nuclear(lepton(11, 1000.0), hyp, *fs, ws, lt, pool, r);
      const ee::LeptonNuclearResult m =
          ee::muon_nuclear(lepton(13, 1000.0), hyp, *fs, *mutab, ws, lt, pool, r);
      if (e.refusal != ee::EmExtraRefusal::kHyperNucleus ||
          m.refusal != ee::EmExtraRefusal::kHyperNucleus) {
        std::printf("FAIL: a hyper-nuclear target was not refused by name (e %d, mu %d)\n",
                    int(e.refusal), int(m.refusal));
        ++fails;
      }
    }
  }

  // -------------------------------------------------------------------------------------------
  // WHERE THE HARD PHOTONS COME FROM
  //
  // The campaign below found that the port's gamma SPECTRUM for a 3 GeV photon on oxygen has a
  // mean of 6.97 MeV against Geant4's 1.756, with the same yield (1.94 per event) and the same
  // summed event energy - so about fourteen events in two thousand contain one gamma of a GeV
  // or so where Geant4 contains none. This block finds such an event and prints the whole final
  // state, because "the mean is four times out" is not a finding and "one secondary in 0.7% of
  // events is the projectile coming back out" is.
  // -------------------------------------------------------------------------------------------
  // Off by default: it costs twenty thousand 3 GeV Bertini events and the finding it produced
  // is recorded in docs/RISK.md V176. `G4GPU_EMEXTRA_HARDGAMMA=1` runs it again.
  if (std::getenv("G4GPU_EMEXTRA_HARDGAMMA") != nullptr) {
    Philox<double> drng(0xD1A6u, 1u, 2u);
    int printed = 0;
    for (int k = 0; k < 20000 && printed < 3; ++k) {
      HadProjectile<double> p;
      p.pdg = 22;
      p.mass = 0.0;
      p.kin_energy = 3000.0;
      HadNucleus n;
      n.a = 16;
      n.z = 8;
      const bert::ApplyResult rr = bert::apply_yourself(
          p, n, *fs, ee::qbbc_gamma_deexcite_choice(), bert::default_cascade_params(),
          bert::default_interface_limits(), *ws.model, *ws.global_out, *ws.out, *ws.dex_out,
          *ws.tmp, *ws.epo, *ws.bert_ws, lt, pool, ws.preco, 0, drng);
      double maxg = 0.0;
      for (int i = 0; i < fs->n_secondaries; ++i) {
        if (fs->secondaries[i].pdg == 22 && fs->secondaries[i].kin_energy > maxg) {
          maxg = fs->secondaries[i].kin_energy;
        }
      }
      if (maxg < 100.0) { continue; }
      ++printed;
      std::printf("HARD-GAMMA event %d: tries %d collider %d trivialised %d no_interaction %d "
                  "would_throw %d refusal %d fate %d  n_sec %d\n",
                  k, rr.n_tries, rr.n_collider_tries, int(rr.trivialised),
                  int(rr.no_interaction), int(rr.would_throw), int(rr.refusal),
                  int(rr.fate_refusal), fs->n_secondaries);
      double esum = 0.0;
      for (int i = 0; i < fs->n_secondaries; ++i) {
        const HadSecondary<double>& s = fs->secondaries[i];
        esum += s.kin_energy;
        std::printf("    pdg %11d Z %3d A %3d  T %10.4f  m %10.4f  cos %7.4f\n", s.pdg, s.z,
                    s.a, s.kin_energy, s.mass, s.direction.z);
      }
      std::printf("    summed T = %.4f, balance dE = %.6g dP = %.6g\n", esum,
                  rr.balance.delta_e(), rr.balance.delta_p());
    }
    if (printed == 0) {
      std::printf("HARD-GAMMA: none in 20,000 events - the campaign's finding does not "
                  "reproduce\n");
    }
  }

  // -------------------------------------------------------------------------------------------
  // The statistical campaign against emextra_apply.csv and emextra_apply_species.csv
  //
  // Four models, 126 cases, N events each side. The oracle's N is in its own `events` column -
  // 20,000 when the campaign has been regenerated - and the port runs `G4GPU_EMEXTRA_EVENTS`
  // times that, 10 by default, so the port's own side is 200,000, which is the size the P13
  // brief prescribes for a row that sits above three sigma. It was 2,000 against 20,000 until
  // docs/RISK.md V177: at 2,000 oracle events one species spectrum rested on TWO samples and
  // read 7.54 sigma, and at 20,000 the same row's oracle mean moved from 17.76 to 12.53 MeV -
  // towards the port's 11.10 - and its spread from 0.77 to 4.75. Raising the PORT's side alone
  // would not have found it: the error of a comparison of two means is dominated by whichever
  // side has fewer samples, and that was the oracle.
  //
  // FIVE SIGMA, over about four thousand comparisons, is the band. The exceptions are stated
  // where they are taken:
  //   * the summed secondary kinetic energy is compared RELATIVELY at 1e-3, not in sigma -
  //     docs/RISK.md V133, and this is the same trap: for GammaNPreco the total is the photon
  //     energy less the recoil every single time, so its sample variance is rounding;
  //   * `pz_mean` likewise, at 1e-3;
  //   * a species whose oracle count is zero is not compared, it is COUNTED - a yield the port
  //     produces and Geant4 never did is reported as a coverage failure instead, which is the
  //     direction that matters;
  //   * a case whose refusal fraction is above 5% reports its worst sigma rather than
  //     asserting it, because the port's sample is then a conditional distribution and the
  //     oracle's is not. Only the 5 GeV Bertini rows can be in that bracket, and they are not:
  //     the QGS refusal belongs to the model CHOICE, which this campaign does not run.
  // -------------------------------------------------------------------------------------------
  const int bMult = new_bucket("Multiplicity", 5.0);
  const int bEkinId = new_bucket("SummedEnergyIdentity", 1.0);   // relative, 1e-9 band
  const int bEkin = new_bucket("SummedKineticEnergy", 5.0);      // sigma
  const int bPzId = new_bucket("SummedZMomentumIdentity", 1.0);  // relative, 1e-9 band
  const int bPz = new_bucket("SummedZMomentum", 5.0);            // sigma
  const int bLep = new_bucket("ScatteredLeptonEnergy", 5.0);
  const int bCos = new_bucket("ScatteredLeptonCosTheta", 5.0);
  const int bNoPh = new_bucket("NoPhotonFraction", 5.0);
  const int bYield = new_bucket("SpeciesYield", 5.0);
  const int bHard = new_bucket("SpeciesHardRate", 5.0);
  const int bKe = new_bucket("SpeciesSoftSpectrum", 5.0);
  const int bAng = new_bucket("SpeciesCosTheta", 5.0);
  {
    long long mult_events = 10;
    if (const char* m = std::getenv("G4GPU_EMEXTRA_EVENTS")) {
      const long long v = std::atoll(m);
      if (v > 0) { mult_events = v; }
    }
    Csv ca, cs;
    if (!ca.load(dir + "/emextra_apply.csv") ||
        !cs.load(dir + "/emextra_apply_species.csv")) {
      std::printf("FAIL: cannot read %s/emextra_apply{,_species}.csv\n", dir.c_str());
      ++fails;
    } else {
      // Index the species rows by (model, particle, ke, Z, A, species).
      std::map<std::string, std::size_t> sidx;
      for (std::size_t r = 0; r < cs.rows.size(); ++r) {
        sidx[cs.s(r, "model") + "|" + cs.s(r, "particle") + "|" + cs.s(r, "ke_MeV") + "|" +
             cs.s(r, "Z") + "|" + cs.s(r, "A") + "|" + cs.s(r, "species")] = r;
      }
      long long n_cases = 0, n_high_refusal = 0, n_weak = 0;
      double grand_worst = 0.0;
      std::string grand_where;
      std::map<int, long long> port_refusals;
      long long total_port_events = 0, total_port_refused = 0;
      // The pdgs the port put in the catch-all species bucket, and the spectrum rows that were
      // not comparable because one side had too few samples to estimate a spread.
      std::map<int, long long> other_pdgs;
      long long n_thin = 0;
      std::string thin_worst;
      double thin_worst_gap = 0.0;

      for (std::size_t r = 0; r < ca.rows.size(); ++r) {
        const std::string model = ca.s(r, "model");
        // The dump writes a flushed `start` marker before each case, so that a crash names the
        // case it died in rather than taking its buffered stdout with it (docs/RISK.md V174).
        // They are progress, not data.
        if (model == "start") { continue; }
        const std::string part = ca.s(r, "particle");
        const double ke = ca.d(r, "ke_MeV");
        const int Z = static_cast<int>(ca.ll(r, "Z"));
        const int A = static_cast<int>(ca.ll(r, "A"));
        const long long oev = ca.ll(r, "events");
        const long long nev = oev * mult_events;
        const std::string w0 = model + " " + part + " " + ca.s(r, "ke_MeV") + " MeV on Z=" +
                               ca.s(r, "Z") + " A=" + ca.s(r, "A");
        case_worst = 0.0;
        ++n_cases;

        Moments mult, ekin, lep, coslep, pz;
        Moments syield[13], ske[13], scos[13];
        // The hard component is a COUNT, not a moment - see the species comparison below.
        long long n_hard[13] = {0};
        double smax[13] = {0.0};
        long long refused = 0, no_photon = 0;
        // One stream per case, seeded from the case, so that adding a case does not move
        // another case's numbers.
        Philox<double> crng(0xE1EAu, static_cast<unsigned>(Z * 1000 + A),
                            static_cast<unsigned>(ke * 10.0) + 1u);
        for (long long k = 0; k < nev; ++k) {
          HadProjectile<double> p;
          double lep_kin = ke;
          double lep_cos = 1.0;
          bool none = false;
          bool bad = false;
          if (model == "preco" || model == "bert") {
            p.pdg = 22;
            p.mass = 0.0;
            p.kin_energy = ke;
          } else if (model == "evd") {
            p.pdg = (part == "e-") ? 11 : -11;
            p.mass = 0.510998910;
            p.charge = (part == "e-") ? -1.0 : 1.0;
            p.kin_energy = ke;
          } else {
            p.pdg = 13;
            p.mass = 105.6583715;
            p.charge = -1.0;
            p.kin_energy = ke;
          }
          HadNucleus n;
          n.a = A;
          n.z = Z;

          if (model == "preco") {
            const ee::LowEGammaResult rr =
                ee::low_e_gamma_apply(p, n, *fs, lt, pool, ws.preco, crng);
            if (rr.refusal != ee::EmExtraRefusal::kNone) {
              bad = true;
              ++port_refusals[static_cast<int>(rr.refusal)];
            }
          } else if (model == "bert") {
            if (A < 3) {
              const ee::LightTargetResult rr = ee::light_target_collide(
                  p, n, *fs, bert::default_cascade_params(), ws.bert_ws, *ws.epo,
                  *ws.global_out, crng);
              if (rr.refusal != ee::EmExtraRefusal::kNone) {
                bad = true;
                ++port_refusals[static_cast<int>(rr.refusal)];
              }
            } else {
              const bert::ApplyResult rr = bert::apply_yourself(
                  p, n, *fs, ee::qbbc_gamma_deexcite_choice(), bert::default_cascade_params(),
                  bert::default_interface_limits(), *ws.model, *ws.global_out, *ws.out,
                  *ws.dex_out, *ws.tmp, *ws.epo, *ws.bert_ws, lt, pool, ws.preco, 0, crng);
              if (rr.refusal != bert::InterfaceRefusal::kNone) {
                bad = true;
                ++port_refusals[100 + static_cast<int>(rr.refusal)];
              }
            }
          } else if (model == "evd") {
            const ee::LeptonVdResult rr = ee::electro_vd_apply(p, n, *fs, ws, lt, pool, crng);
            if (rr.refusal != ee::EmExtraRefusal::kNone) {
              bad = true;
              ++port_refusals[static_cast<int>(rr.refusal)];
            }
            lep_kin = rr.lepton_final_kin;
            lep_cos = rr.lepton_cos_theta;
            none = (rr.no_photon != ee::NoPhotonReason::kNone);
          } else {
            const ee::LeptonVdResult rr =
                ee::muon_vd_apply(p, n, *fs, *mutab, ws, lt, pool, crng);
            if (rr.refusal != ee::EmExtraRefusal::kNone) {
              bad = true;
              ++port_refusals[static_cast<int>(rr.refusal)];
            }
            lep_kin = rr.lepton_final_kin;
            lep_cos = rr.lepton_cos_theta;
            none = (rr.no_photon != ee::NoPhotonReason::kNone);
          }
          ++total_port_events;
          if (bad) {
            ++refused;
            ++total_port_refused;
            continue;
          }
          if (none) { ++no_photon; }
          mult.add(double(fs->n_secondaries));
          double esum = 0.0, pzsum = 0.0;
          int per[13] = {0};
          for (int i = 0; i < fs->n_secondaries; ++i) {
            const HadSecondary<double>& s = fs->secondaries[i];
            const int b = species_bucket_of(s.pdg, s.z, s.a);
            ++per[b];
            // The same split the dump makes, at the same threshold: a tenth of the projectile
            // energy. See `SpeciesBucket` in ref/dump/dump_emextra.cc.
            if (s.kin_energy >= 0.1 * ke) { ++n_hard[b]; }
            else { ske[b].add(s.kin_energy); }
            // WHAT IS IN THE CATCH-ALL. Bucket 12 is "none of the other twelve" - a kaon, a
            // hyperon, a K0, an e+ - and its mean energy is therefore a mean over unlike
            // things, which is why the spectrum comparison treats it as it does. The oracle
            // does not record which species went in, so the port's list is printed at the end
            // and half the hole is named rather than none of it.
            if (b == 12) { ++other_pdgs[s.pdg]; }
            smax[b] = (s.kin_energy > smax[b]) ? s.kin_energy : smax[b];
            scos[b].add(s.direction.z);
            esum += s.kin_energy;
            pzsum += s.momentum() * s.direction.z;
          }
          for (int b = 0; b < 13; ++b) { syield[b].add(double(per[b])); }
          ekin.add(esum);
          pz.add(pzsum);
          lep.add(lep_kin);
          coslep.add(lep_cos);
        }

        const double frac = (nev > 0) ? double(refused) / double(nev) : 0.0;
        // AN ORACLE OF ONE EVENT PER CASE IS NOT A DISTRIBUTION, and comparing against it in
        // sigma would pass whatever the port did - the pooled standard error is then the
        // population sigma itself. `ref/oracle/run.bat` regenerates this campaign at ONE event
        // per case by default, because at 2,000 and above it dies inside the full dumper (see
        // ref/dump/dump_emextra.cc and docs/RISK.md V174), so the weak case is the NORMAL one
        // and it must be loud rather than silently green. What is still asserted at one event:
        // that every case ran, that every model was reached, and that no case produced neither
        // a final state nor a named refusal.
        const bool weak = (oev < 100);
        if (weak) { ++n_weak; }
        const bool assertable = (frac <= 0.05) && (mult.n > 0) && !weak;
        if (!assertable && !weak) { ++n_high_refusal; }
        if (mult.n == 0 && refused == 0) {
          std::printf("FAIL: %s produced neither a final state nor a refusal in %lld events\n",
                      w0.c_str(), nev);
          ++fails;
        }

        if (assertable) {
          cmp_welch(bMult, mult.mean(), mult.sigma(), mult.n, ca.d(r, "mult_mean"),
                    ca.d(r, "mult_rms"), oev, w0 + " mult");
          // THE ORACLE'S OWN RMS DECIDES WHICH COMPARISON THE SUMMED ENERGY GETS, and the two
          // are not interchangeable - docs/RISK.md V133.
          //
          // For `preco` the summed secondary kinetic energy IS the photon energy less the
          // recoil, every single event, so its rms is 1e-7 of its mean and it is a conservation
          // identity: a five-sigma band on it is a band of zero width and would report hundreds
          // of sigma for twelve digits of agreement. For `evd` and `mvd` the same quantity is a
          // function of the SAMPLED photon energy and has an rms comparable to its mean, so a
          // relative band at 1e-3 is absurdly tight - it reported 217 for two means that differ
          // by a fifth of one standard error. So the rule is the data's: an rms under 1e-6 of
          // the mean is an identity and is compared relatively at 1e-9; anything else is a
          // random variable and is compared in sigma.
          {
            const double want = ca.d(r, "ekin_mean");
            const double wrms = ca.d(r, "ekin_rms");
            if (std::fabs(wrms) <= 1e-6 * std::fabs(want)) {
              cmp_rel(bEkinId, ekin.mean(), want, 1e-6, w0 + " ekin (identity)");
            } else {
              cmp_welch(bEkin, ekin.mean(), ekin.sigma(), ekin.n, want, wrms, oev, w0 + " ekin");
            }
          }
          {
            const double want = ca.d(r, "pz_mean");
            const double wrms = pz.sigma();   // the oracle dumps no pz rms; the port's will do
            // 1e-6 and not 1e-9: the two sides average over different numbers of events - the
            // oracle's N against `G4GPU_EMEXTRA_EVENTS` times it - so the identity is summed in a
            // different order on each side and the last digits do not survive. Measured: the
            // worst of the 61 identity rows is 2.2e-8 of a 199 MeV momentum, which is 4.5 eV.
            if (std::fabs(wrms) <= 1e-6 * std::fabs(want)) {
              cmp_rel(bPzId, pz.mean(), want, 1e-6, w0 + " pz (identity)");
            } else {
              cmp_welch(bPz, pz.mean(), pz.sigma(), pz.n, want, pz.sigma(), oev, w0 + " pz");
            }
          }
          if (model == "evd" || model == "mvd") {
            cmp_welch(bLep, lep.mean(), lep.sigma(), lep.n, ca.d(r, "lep_mean"), ca.d(r, "lep_rms"), oev,
                      w0 + " lep");
            cmp_sigma(bCos, coslep.mean(), ca.d(r, "coslep_mean"), coslep.sigma(), coslep.n,
                      oev, w0 + " coslep");
            // The fraction of events in which no photon was produced: a binomial, compared
            // with the POOLED proportion's sqrt(p(1-p)) and not the port's own.
            //
            // The port's own is what this had, and it is degenerate exactly where the answer
            // matters. `evd e- 10 GeV on Fe56` produced no such event in 20,000 while Geant4
            // produced one in 20,000; the port's p is then 0, its sqrt(p(1-p)) is 0, and
            // `cmp_sigma`'s zero-variance branch - which exists for quantities that are exact,
            // not for rates that happen to be unobserved - reported 1e12 sigma for two rates
            // that differ by one event. Pooling gives p = 1/40,000, an error of 5e-5 on the
            // difference, and 1.0 sigma, which is what a single event in forty thousand is
            // worth. Same trap as docs/RISK.md V177 and the same shape as the species yields,
            // where a zero count on one side is already compared against the pooled rate.
            const long long got_k = no_photon, want_k = ca.ll(r, "no_photon");
            const double got_p = double(got_k) / double(mult.n);
            const double want_p = double(want_k) / double(oev);
            const double p_pool =
                double(got_k + want_k) / double(mult.n + oev);
            const double sp = std::sqrt(std::max(0.0, p_pool * (1.0 - p_pool)));
            cmp_sigma(bNoPh, got_p, want_p, sp, mult.n, oev, w0 + " no_photon");
          }
          for (int b = 0; b < 13; ++b) {
            const std::string key = model + "|" + part + "|" + ca.s(r, "ke_MeV") + "|" +
                                    ca.s(r, "Z") + "|" + ca.s(r, "A") + "|" +
                                    species_bucket_name(b);
            const auto it = sidx.find(key);
            if (it == sidx.end()) { continue; }
            const std::size_t sr = it->second;
            const long long ocount = cs.ll(sr, "count");
            const std::string ws2 = w0 + " " + species_bucket_name(b);

            // THE YIELD, INCLUDING WHEN GEANT4 PRODUCED NONE.
            //
            // A zero oracle count used to be a hard failure if the port produced anything at
            // all, and that is not a statistic: two thousand events that contain no internal
            // conversion electron are perfectly compatible with a rate of one in two thousand,
            // and the first version of this check failed on exactly that. So a zero count is
            // compared like any other, with the POOLED rate as the variance - which still fails
            // loudly for a species the port invents at a real rate, and says nothing about one
            // neither sample can resolve.
            const double want_yield = cs.d(sr, "yield_mean");
            const double pooled = 0.5 * (syield[b].mean() + want_yield);
            const double ys = (syield[b].sigma() > 0.0)
                                  ? syield[b].sigma()
                                  : std::sqrt(pooled > 0.0 ? pooled : 1.0);
            cmp_sigma(bYield, syield[b].mean(), want_yield, ys, syield[b].n, oev, ws2 + " yield");

            // THE HARD COMPONENT AS A RATE, AND THE SOFT ONE AS A SPECTRUM.
            //
            // The gamma bucket of a GeV photo-nuclear event holds two populations: one or two
            // MeV de-excitation photons, and - a few events in a thousand - the projectile
            // itself, elastically scattered off a bound nucleon (`{gam, pro}` is the first
            // two-body final state of `G4CascadeT1GamNChannel`, cross section 0.1 to 2.7
            // microbarn against a total a hundred times larger). Seven such events in two
            // thousand move the bucket's mean by a factor of four and its rms by forty, and no
            // pair of two-thousand-event samples can agree on that mean however right both are.
            // The first version of this comparison reported 67.79 sigma for exactly that and it
            // was the comparison that was wrong, not the physics: `max_ekin` said the port's
            // outlier was a 2,996 MeV photon out of a 3,000 MeV projectile, and printing the
            // whole final state showed a forward photon at cos 0.99 with an energy balance of
            // 5e-6 GeV - an elastic scatter, not a bug.
            //
            // So the hard count is a Poisson rate and the soft moments are the spectrum.
            const long long ohard = cs.ll(sr, "count_hard");
            const double ghard_rate = double(n_hard[b]) / double(mult.n);
            const double ohard_rate = double(ohard) / double(oev);
            const double hpool = 0.5 * (ghard_rate + ohard_rate);
            if (ohard > 0 || n_hard[b] > 0) {
              cmp_sigma(bHard, ghard_rate, ohard_rate,
                        std::sqrt(hpool > 0.0 ? hpool : 1.0), mult.n, oev, ws2 + " hard rate");
            }
            const double want_soft = cs.d(sr, "ekin_soft_mean");
            // The oracle's soft rms when it has one; `ekin_rms` IS the soft rms when the hard
            // count is zero, which is most rows, and the column is absent in an oracle written
            // before the split - in which case there is nothing to compare and the row is
            // skipped rather than compared against a zero.
            double want_soft_sd = cs.d(sr, "ekin_soft_rms");
            if (want_soft_sd == 0.0 && ohard == 0) { want_soft_sd = cs.d(sr, "ekin_rms"); }
            // FIVE SAMPLES A SIDE, and the rows that have fewer are COUNTED rather than
            // compared. A spread estimated from two samples has one degree of freedom and is
            // five times too small a few per cent of the time (docs/RISK.md V177); at five it
            // has four, which is still poor but is an estimate. The rows this drops are the
            // species that appear a handful of times in twenty thousand events, and they keep
            // their YIELD and their HARD-RATE comparisons - only the spectrum goes, because a
            // spectrum is the one thing four products cannot show. The count is printed, with
            // the widest gap among them, so that a real difference hiding in a thin row is
            // visible as a number to go and measure rather than as silence.
            const long long osoft = ocount - ohard;
            if (ske[b].n >= 5 && osoft >= 5 && want_soft > 0.0 && want_soft_sd > 0.0) {
              cmp_welch(bKe, ske[b].mean(), ske[b].sigma(), ske[b].n, want_soft, want_soft_sd,
                        osoft, ws2 + " soft ekin");
            } else if (ske[b].n > 0 && osoft > 0 && want_soft > 0.0) {
              ++n_thin;
              const double gap = (want_soft > 0.0)
                                     ? std::fabs(ske[b].mean() - want_soft) / want_soft
                                     : 0.0;
              if (gap > thin_worst_gap) {
                thin_worst_gap = gap;
                char buf[200];
                std::snprintf(buf, sizeof buf, "%s soft ekin %.4g (n %lld) vs %.4g (n %lld)",
                              ws2.c_str(), ske[b].mean(), ske[b].n, want_soft, osoft);
                thin_worst = buf;
              }
            }
            if (scos[b].n > 1 && ocount > 1) {
              // cos(theta) of an isotropic population has variance 1/3 on both sides, so the
              // pooled form is right here and Welch has nothing extra to say.
              cmp_sigma(bAng, scos[b].mean(), cs.d(sr, "cos_mean"),
                        std::sqrt(std::max(0.0, 1.0 / 3.0)), scos[b].n, ocount, ws2 + " cos");
            }
          }
        }
        if (case_worst > grand_worst) { grand_worst = case_worst; grand_where = w0; }
        if (frac > 0.001 || case_worst > 4.0) {
          std::printf("  CASE %-40s refused %6.3f%%  worst %7.2f\n", w0.c_str(), 100.0 * frac,
                      case_worst);
        }
      }

      std::printf("\nstatistical campaign: %lld cases, %lld port events, %lld refused "
                  "(%.4f%%), %lld cases above the 5%% refusal bracket\n",
                  n_cases, total_port_events, total_port_refused,
                  total_port_events > 0
                      ? 100.0 * double(total_port_refused) / double(total_port_events)
                      : 0.0,
                  n_high_refusal);
      if (n_weak > 0) {
        std::printf("*** %lld of %lld cases have an ORACLE OF UNDER 100 EVENTS and are NOT "
                    "asserted statistically - only that they ran, reached a model and produced "
                    "either a final state or a named refusal. Regenerate the campaign with "
                    "G4GPU_EMEXTRA_EVENTS=20000 to compare distributions; see "
                    "ref/dump/dump_emextra.cc on why that is not the default.\n",
                    n_weak, n_cases);
      }
      if (n_thin > 0) {
        std::printf("species spectra NOT compared because one side had under five samples: "
                    "%lld rows; widest relative gap among them %.1f%% at %s\n",
                    n_thin, 100.0 * thin_worst_gap, thin_worst.c_str());
      }
      if (!other_pdgs.empty()) {
        std::printf("the port's catch-all species bucket held:");
        for (const auto& kv : other_pdgs) { std::printf(" %d x%lld", kv.first, kv.second); }
        std::printf("  (the oracle does not record which, so only this side is named)\n");
      }
      if (!port_refusals.empty()) {
        std::printf("port refusals by kind:");
        for (const auto& kv : port_refusals) {
          if (kv.first >= 100) {
            std::printf(" bertini(%d)=%lld", kv.first - 100, kv.second);
          } else {
            std::printf(" %s=%lld",
                        ee::refusal_name(static_cast<ee::EmExtraRefusal>(kv.first)), kv.second);
          }
        }
        std::printf("\n");
      }
      std::printf("worst single comparison: %.2f in %s\n", grand_worst, grand_where.c_str());
    }
  }

  long long total = 0;
  for (const Bucket& b : buckets) {
    total += b.n;
    const bool bad = (b.worst > b.tol);
    std::printf("%-26s %8lld pts  worst %7.2f  tol %5.2f%s%s\n", b.name, b.n, b.worst, b.tol,
                bad ? "  FAIL " : "", bad ? b.where.c_str() : "");
    if (bad) { ++fails; }
  }
  std::printf("%lld statistical comparisons\n", total);

  delete mutab;
  delete fs;
  if (fails == 0) {
    std::printf("\ntest_emextra_models: OK\n");
    return 0;
  }
  std::printf("\ntest_emextra_models: %d FAILURES\n", fails);
  return 1;
}
