// G4PreCompoundModel and G4GeneratorPrecompoundInterface against ref/oracle/preco_*.csv.
//
// Six of the eight comparisons here are EXACT, and the reason is that Geant4 lets a prescribed
// random engine be installed: `ref/dump/dump_precompound.cc` runs every sampler in this
// package under an eight-value uniform cycle, which turns SampleKineticEnergy,
// ChooseFragment, PerformTransition and PerformEmission into deterministic functions of
// (inputs, phase). So this test does not have to fall back on distributions for them - it
// compares kinetic energies, four-momenta and exciton counts to 1e-13 relative, AND compares
// the NUMBER of deviates each one consumed, which is the part a transcription can get wrong
// while still producing a plausible answer.
//
//   preco_params.csv        the thirteen defaults, plus fFermiEnergy, fR0, fTransitionsR0 and
//                           G4Pow::logfactorial(0..40) - the one table the angular generator
//                           depends on.
//   preco_channels.csv      per (Z, A, E*, P, Pc, H) and channel: the residual, binding
//                           energy, energy threshold, both masses, IsItPossible, and the
//                           emission-probability integral for all five OPTxs. Exact.
//   preco_sample.csv        SampleKineticEnergy under the cycle. Exact, with the draw count.
//   preco_transitions.csv   CalculateProbability's three rates for CEM, CEM+NGB, Gupta,
//                           Gupta+NGB and GNASH, and the exciton configuration
//                           PerformTransition leaves. Exact, with both draw counts.
//   preco_equilibrium.csv   the level density, the equilibrium exciton number and both gates'
//                           verdicts. Exact.
//   preco_gates.csv         the two gates at the nuclides and the excitation energies where the
//                           four tokens they differ in are decidable. Exact.
//   preco_emission.csv      PerformEmission under the cycle: channel, ejectile four-momentum,
//                           residual (Z, A, E*, P, Pc, H). Exact.
//   preco_deexcite*.csv     the whole model, 20,000 events on fifteen fragments, statistical.
//                           This one cannot be exact: the equilibrium tail is P3's cascade
//                           under HepJamesRandom and this port's engine is Philox.
//   preco_*_species.csv     the two statistical tallies again, folded onto (Z, A, is_preco) on
//                           the dump side, for the per-event multiplicity variance - the one
//                           column that cannot be folded here, because the second moment of a
//                           sum is not the sum of second moments.
//   preco_apply*.csv        ApplyYourself on nine (projectile, target, energy) cases - four
//                           matched neutron/proton pairs and one below the entry gate - 20,000
//                           events each. Statistical, and it is the only check the initial
//                           fragment can have - see section 9.
//   preco_propagate*.csv    Propagate on a real wounded nucleus, replayed from the dumped
//                           nucleon list. Exact on which tracks escaped and on their summed
//                           four-momentum.
//
// **What the statistical file can and cannot separate.** The oracle tags each product with
// `G4ReactionProduct::GetCreatorModelID()`, and G4PreCompoundEmission stamps model_PRECO on
// its ejectiles while every de-excitation channel stamps its own - so the pre-equilibrium
// stage and P3's cascade ARE distinguishable in the oracle, and the counts are compared
// separately for the two tags. The one place the tag over-counts is a fragment released
// untouched by the handler AFTER an emission, because PerformEmission stamps PRECO on the
// residual too; the dump's leading-run count excludes it by requiring the product to be one
// of the six ejectile species, and the dump gives the initial fragment creator id -1 so an
// untouched release of the original is tagged -1 rather than PRECO.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "core/rng.cuh"
#include "data/level_data.cuh"
#include "host/g4data.cuh"
#include "physics/hadronic/precompound/generator_interface.cuh"

using namespace g4gpu;

/// The deliverable is device-callable, and a `__host__ __device__` template that is only ever
/// instantiated from the host is never compiled for the device at all. These kernels are never
/// launched - this package is validated on the host, and the machine's GPU belongs to the
/// integration builds - but instantiating them is what proves the module is device code: no
/// host-only header, no `std::` call without a device overload, no local static nvcc rejects.
/// Deleting either would leave the whole module compiling for the host only.
__global__ void preco_device_probe(const deex::Fragment* in, preco::Excitons ex,
                                   data::LevelTable lt, deex::FermiPool pool,
                                   preco::PrecoWorkspace ws, preco::PrecoStatus* out) {
  Philox<double> rng(1u, 2u, 3u);
  out[0] = preco::deexcite(in[0], ex, lt, pool, ws, rng);
}

__global__ void preco_interface_probe(const preco::CascadeTrack* tr, int n,
                                      preco::WoundedNucleus nuc, preco::CascadeTrack* esc,
                                      int* nesc, preco::GeneratorRefusal* ref,
                                      preco::CascadeResidual* out) {
  Philox<double> rng(4u, 5u, 6u);
  out[0] = preco::propagate_residual(tr, n, nuc, deex::LorentzVector(), esc, 8, *nesc, *ref,
                                     rng);
}

namespace {

int fails = 0;

/// The eight uniforms dump_precompound.cc's CycleEngine cycles through, in its order. If these
/// two lists ever diverge every exact comparison below turns into noise, which is why the
/// values are written out on both sides rather than derived.
///
/// A function-local static, which is the one form of a constant array nvcc accepts in
/// `__host__ __device__` code - the same pattern P3 uses for G4UnstableFragmentBreakUp's
/// tables. A namespace-scope array cannot be read from device code, and `CycleRng` has to be
/// device-callable because the samplers it drives are `__host__ __device__` templates: with a
/// host-only Rng every instantiation raises nvcc warning 20014, and a warning in a header
/// every test includes is noise that hides a real one.
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

// ---------------------------------------------------------------------------------------------
// Comparison bookkeeping
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

std::vector<std::string> read_lines(const std::string& path) {
  std::vector<std::string> out;
  FILE* f = std::fopen(path.c_str(), "r");
  if (f == nullptr) { return out; }
  char line[8192];
  while (std::fgets(line, sizeof line, f) != nullptr) { out.push_back(line); }
  std::fclose(f);
  return out;
}

/// Splits on commas. Fields are numeric except preco_propagate.csv's `escaped_ids`, which is a
/// '|'-joined list, so the raw strings are kept as well as the doubles.
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

std::string key(int z, int a, double e, int p, int pc, int h) {
  char buf[128];
  std::snprintf(buf, sizeof buf, "Z=%d A=%d E*=%g (%d,%d,%d)", z, a, e, p, pc, h);
  return buf;
}

// ---------------------------------------------------------------------------------------------
// Statistics, as tests/test_deex_breakup.cu defines them
// ---------------------------------------------------------------------------------------------

long long n_poisson_fallback = 0;

/// Two total counts over N events each, with the per-event multiplicity variance MEASURED on
/// BOTH sides rather than assumed Poisson - a de-excitation cascade's multiplicity is narrower
/// than Poisson because energy is conserved, so a Poisson sigma would be too large and would
/// hide a real disagreement.
///
/// The oracle's variance is measured because the port's is not a substitute for it. The port
/// emits exactly one proton in every 20 MeV p + C12 event, so `var1` is zero; Geant4 emits one
/// in 19,996 of 20,000, because its Fermi break-up occasionally chooses n + N12 instead. With
/// `s2 = 2 N var1` that 4-count difference is infinitely significant, and this test reported it
/// as 1e9 sigma until `mean_mult2` was added to the dump. With both variances it is 2.0 sigma,
/// which is what it is: 0 against 2 on a channel of probability 1e-4.
///
/// Both variances zero is still infinite significance, and correctly so - two deterministic
/// multiplicities that differ do differ - but it now takes BOTH sides being deterministic.
///
/// `var2 < 0` means the oracle side has no measured variance at all, which happens only for a
/// species the port produced and the oracle never did. There the stand-in is the Poisson
/// variance of the larger mean, the maximum-entropy choice for a count whose per-event
/// distribution is unknown; assuming determinism instead reports 1 against 0 as a 1e9-sigma
/// failure, which is what this test did on its first run (Ca40 at 80 MeV, one Be10).
double multiplicity_z(long long n1, long long n2, long long N, double var1, double var2) {
  if (N < 2) { return 0.0; }
  double v1 = (var1 > 0.0) ? var1 : 0.0;
  double v2 = var2;
  if (v2 < 0.0) {
    const double m1 = static_cast<double>(n1) / static_cast<double>(N);
    const double m2 = static_cast<double>(n2) / static_cast<double>(N);
    v2 = (m1 > m2) ? m1 : m2;
    ++n_poisson_fallback;
  }
  if (v2 < 0.0) { v2 = 0.0; }
  const double s2 = static_cast<double>(N) * (v1 + v2);
  if (!(s2 > 0.0)) { return (n1 == n2) ? 0.0 : 1.e9; }
  return std::fabs(static_cast<double>(n1 - n2)) / std::sqrt(s2);
}

/// A per-event categorical count is Binomial(N, p), not Poisson: the residual (Z, A) and the
/// pre-equilibrium multiplicity histogram are one draw per event, and for a bin that holds
/// every event a Poisson sigma of sqrt(N) would accept a 1% disagreement.
double binomial_z(long long n1, long long n2, long long N) {
  if (N < 2) { return 0.0; }
  const double p1 = static_cast<double>(n1) / static_cast<double>(N);
  const double p2 = static_cast<double>(n2) / static_cast<double>(N);
  const double s2 = static_cast<double>(N) * (p1 * (1.0 - p1) + p2 * (1.0 - p2));
  if (!(s2 > 0.0)) { return (n1 == n2) ? 0.0 : 1.e9; }
  return std::fabs(static_cast<double>(n1 - n2)) / std::sqrt(s2);
}

double mean_z(double m1, double v1, long long n1, double m2, double v2, long long n2) {
  if (n1 < 2 || n2 < 2) { return 0.0; }
  const double s2 = v1 / static_cast<double>(n1) + v2 / static_cast<double>(n2);
  if (!(s2 > 0.0)) { return 0.0; }
  return std::fabs(m1 - m2) / std::sqrt(s2);
}

/// mean_z where the quantity can be deterministic - a discrete gamma line gives every event
/// the same energy, and mean_z's zero-variance answer of 0 would then accept anything.
double moment_z(double m1, double v1, long long n1, double m2, double v2, long long n2) {
  if (!(v1 > 0.0) && !(v2 > 0.0)) {
    return (std::fabs(m1 - m2) <= 1.e-9 * (1.0 + std::fabs(m2))) ? 0.0 : 1.e9;
  }
  return mean_z(m1, v1, n1, m2, v2, n2);
}

struct Tally {
  long long count = 0;
  double sum_e = 0.0;
  double sum_e2 = 0.0;
  double sum_k2 = 0.0;    ///< per-event multiplicity second moment, summed over events
  bool has_k2 = false;    ///< false on the oracle side when no *_species.csv row supplied one
};

/// The per-event multiplicity variance of a tally, or -1 when there is none to be had.
double tally_var(const Tally& t, long long N) {
  if (!t.has_k2 || N < 1) { return -1.0; }
  const double m = static_cast<double>(t.count) / static_cast<double>(N);
  const double v = t.sum_k2 / static_cast<double>(N) - m * m;
  return (v > 0.0) ? v : 0.0;
}

/// (Z, A) from an oracle PDG code, with the isomer digit folded away.
///
/// **Why the statistical comparison is keyed on (Z, A) and not on the PDG code.** For a heavy
/// ion the last digit of `10LZZZAAAI` is an isomer level G4IonTable assigns when it first
/// creates the ion, taken from G4ENSDFSTATE when the excitation matches and numbered in
/// creation order otherwise - a property of the RUN's ion table, not of the fragment. P3
/// refuses to invent it (`deex_fixed_pdg` returns 0 for an ion, see
/// deexcitation/excitation_handler.cuh's header), so the port's heavy products carry pdg = 0
/// and (Z, A). Folding the oracle onto (Z, A) is what makes the two comparable; comparing PDG
/// codes instead reports every heavy product as missing, which is what this test did on its
/// first run - 20,000 C12 residuals in the oracle against zero in the port, with the residual
/// histogram agreeing at 3 sigma the whole time.
bool pdg_to_za(int pdg, int& Z, int& A) {
  if (pdg == 22) { Z = 0; A = 0; return true; }        // gamma
  if (pdg == 11) { Z = -1; A = 0; return true; }       // e-, told apart by Z
  if (pdg == 2112) { Z = 0; A = 1; return true; }
  if (pdg == 2212) { Z = 1; A = 1; return true; }
  if (pdg > 1000000000) {
    A = (pdg / 10) % 1000;
    Z = (pdg / 10000) % 1000;
    return true;
  }
  return false;
}

/// The eight species whose rest mass is a constant, and therefore the eight whose kinetic
/// energy the port can reproduce at all: `G4ReactionProduct::GetKineticEnergy` is
/// `totalEnergy - mass` with the DEFINITION's mass, and for an isomer that mass is the
/// excitation snapped to G4ENSDFSTATE. The port does not snap. So counts are compared for
/// every species and energies only for these.
bool is_fixed_mass(int Z, int A) {
  if (A == 0) { return true; }                       // gamma or conversion electron
  if (A == 1 && (Z == 0 || Z == 1)) { return true; }  // n, p
  if (A == 2 && Z == 1) { return true; }              // d
  if (A == 3 && (Z == 1 || Z == 2)) { return true; }  // t, He3
  if (A == 4 && Z == 2) { return true; }              // alpha
  return false;
}

/// The comparison key: (Z, A) plus the pre-equilibrium tag, offset so it stays positive.
int za_tag_key(int Z, int A, int is_preco) {
  return ((Z + 500) * 1000 + A) * 2 + is_preco;
}

// ---------------------------------------------------------------------------------------------

/// The de-excitation module's buffers, sized for the heaviest campaign. Pb208 at 200 MeV
/// makes tens of products and pushes every hot fragment back onto the evaporation list.
struct Buffers {
  std::vector<deex::Fragment> evap, results, step;
  std::vector<deex::DeexProduct> deex_products, products;
  Buffers() : evap(4096), results(1024), step(512), deex_products(1024), products(1024) {}
  preco::PrecoWorkspace view() {
    preco::PrecoWorkspace ws;
    ws.deex.evap_list = evap.data();
    ws.deex.evap_capacity = static_cast<int>(evap.size());
    ws.deex.results = results.data();
    ws.deex.results_capacity = static_cast<int>(results.size());
    ws.deex.step = step.data();
    ws.deex.step_capacity = static_cast<int>(step.size());
    ws.deex.products = deex_products.data();
    ws.deex.products_capacity = static_cast<int>(deex_products.size());
    ws.products = products.data();
    ws.products_capacity = static_cast<int>(products.size());
    return ws;
  }
};

}  // namespace

int main() {
  const char* env = std::getenv("G4GPU_ORACLE");
  const std::string dir = (env != nullptr) ? env : "ref/oracle";
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
  std::printf("level table: %d managers; Fermi pool: %d fragments\n", lt.n_managers,
              pool.n_frag);

  // -------------------------------------------------------------------------------------------
  // 1. preco_params.csv - the thirteen defaults, and G4Pow::logfactorial
  // -------------------------------------------------------------------------------------------
  {
    const int b = new_bucket("Params", 0.0);
    const auto lines = read_lines(dir + "/preco_params.csv");
    if (lines.size() < 2) {
      std::printf("cannot read %s/preco_params.csv - run ref/oracle/run.bat tables\n",
                  dir.c_str());
      return 1;
    }
    const deex::DeexParameters& p = deex::deex_params();
    int n_seen = 0;
    for (std::size_t i = 1; i < lines.size(); ++i) {
      const auto f = split(lines[i]);
      if (f.size() < 2) { continue; }
      const std::string& n = f[0];
      const double v = dv(f, 1);
      ++n_seen;
      if (n == "fPrecoLowEnergy") { cmp(b, p.preco_low_energy, v, n); }
      else if (n == "fPrecoHighEnergy") { cmp(b, p.preco_high_energy, v, n); }
      else if (n == "fPhenoFactor") { cmp(b, p.pheno_factor, v, n); }
      else if (n == "fMinZForPreco") { cmp_int(b, p.min_z_for_preco, (long long)v, n); }
      else if (n == "fMinAForPreco") { cmp_int(b, p.min_a_for_preco, (long long)v, n); }
      else if (n == "fPrecoType") { cmp_int(b, p.preco_type, (long long)v, n); }
      else if (n == "fNeverGoBack") { cmp_int(b, p.never_go_back ? 1 : 0, (long long)v, n); }
      else if (n == "fUseSoftCutoff") { cmp_int(b, p.use_soft_cutoff ? 1 : 0, (long long)v, n); }
      else if (n == "fUseCEM") { cmp_int(b, p.use_cem ? 1 : 0, (long long)v, n); }
      else if (n == "fUseGNASH") { cmp_int(b, p.use_gnash ? 1 : 0, (long long)v, n); }
      else if (n == "fUseHETC") { cmp_int(b, p.use_hetc ? 1 : 0, (long long)v, n); }
      else if (n == "fUseAngularGen") { cmp_int(b, p.use_angular_gen ? 1 : 0, (long long)v, n); }
      else if (n == "fPrecoDummy") { cmp_int(b, p.preco_dummy ? 1 : 0, (long long)v, n); }
      else if (n == "fFermiEnergy") { cmp(b, p.fermi_energy, v, n); }
      else if (n == "fTransitionsR0_fm") { cmp(b, p.transitions_r0 / deex::fermi(), v, n); }
      else if (n == "fR0_fm") { cmp(b, p.r0 / deex::fermi(), v, n); }
      else if (n.rfind("logfactorial_", 0) == 0) {
        const int k = std::atoi(n.c_str() + std::strlen("logfactorial_"));
        cmp(b, preco::log_factorial(k), v, n);
      } else { --n_seen; }
    }
    if (n_seen < 13 + 3 + 41) {
      std::printf("FAIL: preco_params.csv had only %d of the 57 expected rows\n", n_seen);
      ++fails;
    }
  }

  // -------------------------------------------------------------------------------------------
  // 2. preco_channels.csv - the six channels' Initialize and the emission-probability integral
  // -------------------------------------------------------------------------------------------
  long long n_early_return = 0;
  {
    const int b_res = new_bucket("ChannelResidual", 0.0);
    const int b_geom = new_bucket("ChannelGeometry", 1e-13);
    const int b_poss = new_bucket("ChannelIsItPossible", 0.0);
    const int b_prob[5] = {
      new_bucket("EmissionProbOPT0", 1e-12), new_bucket("EmissionProbOPT1", 1e-12),
      new_bucket("EmissionProbOPT2", 1e-12), new_bucket("EmissionProbOPT3", 1e-12),
      new_bucket("EmissionProbOPT4", 1e-12),
    };
    const auto lines = read_lines(dir + "/preco_channels.csv");
    if (lines.size() < 2) {
      std::printf("cannot read %s/preco_channels.csv\n", dir.c_str());
      return 1;
    }
    for (std::size_t i = 1; i < lines.size(); ++i) {
      const auto f = split(lines[i]);
      if (f.size() < 21) { continue; }
      const int Z = iv(f, 0), A = iv(f, 1);
      const double e = dv(f, 2);
      const preco::Excitons ex{iv(f, 3), iv(f, 4), iv(f, 5)};
      const int chan = iv(f, 6);
      const std::string w = key(Z, A, e, ex.particles, ex.charged, ex.holes) + " chan " +
                            std::to_string(chan);
      const deex::Fragment frag = deex::make_excited_fragment(Z, A, e, 0.0);

      // OPTxs is read by Initialize (it chooses `elim`), so each option gets its own state -
      // which is what the dump does too, by calling SetOPTxs before Initialize.
      for (int opt = 0; opt < 5; ++opt) {
        preco::PreFragState st = preco::pre_frag_initialize(chan, frag, lt, opt);
        const bool ok = preco::pre_frag_is_possible(st, ex.particles, ex.charged);
        const double prob = ok ? preco::pre_frag_emission_probability(st, frag, ex) : 0.0;
        cmp(b_prob[opt], prob, dv(f, 16 + opt), w);
        if (opt != 3) { continue; }
        cmp_int(b_res, st.z, iv(f, 7), w + " ejZ");
        cmp_int(b_res, st.a, iv(f, 8), w + " ejA");
        cmp_int(b_poss, ok ? 1 : 0, iv(f, 11), w + " possible");
        cmp(b_geom, st.mass, dv(f, 14), w + " mass");
        // theMinKinEnergy, theMaxKinEnergy and theCoulombBarrier are zeroed before
        // Initialize's early return, so the threshold is comparable on every row; theResA,
        // theResZ, theBindingEnergy and theResMass are NOT - Geant4 leaves the previous
        // fragment's values in them, and this test must not compare against those.
        cmp(b_geom, preco::pre_frag_energy_threshold(st), dv(f, 13), w + " threshold");
        if (!st.possible) { ++n_early_return; continue; }
        cmp_int(b_res, st.res_z, iv(f, 9), w + " resZ");
        cmp_int(b_res, st.res_a, iv(f, 10), w + " resA");
        cmp(b_geom, st.binding_energy, dv(f, 12), w + " binding");
        cmp(b_geom, st.res_mass, dv(f, 15), w + " resmass");
      }
    }
  }

  // -------------------------------------------------------------------------------------------
  // 3. preco_sample.csv - SampleKineticEnergy under the cycle
  // -------------------------------------------------------------------------------------------
  {
    const int b_t = new_bucket("SampleKineticEnergy", 1e-12);
    const int b_d = new_bucket("SampleDraws", 0.0);
    const auto lines = read_lines(dir + "/preco_sample.csv");
    for (std::size_t i = 1; i < lines.size(); ++i) {
      const auto f = split(lines[i]);
      if (f.size() < 11) { continue; }
      const int Z = iv(f, 0), A = iv(f, 1);
      const double e = dv(f, 2);
      const preco::Excitons ex{iv(f, 3), iv(f, 4), iv(f, 5)};
      const int chan = iv(f, 6), phase = iv(f, 7);
      const std::string w = key(Z, A, e, ex.particles, ex.charged, ex.holes) + " chan " +
                            std::to_string(chan) + " ph " + std::to_string(phase);
      const deex::Fragment frag = deex::make_excited_fragment(Z, A, e, 0.0);
      preco::PreFragState st = preco::pre_frag_initialize(chan, frag, lt, 3);
      if (!preco::pre_frag_is_possible(st, ex.particles, ex.charged)) { continue; }
      const double prob = preco::pre_frag_emission_probability(st, frag, ex);
      if (!(prob > 0.0)) { continue; }
      CycleRng rng;
      rng.reset(phase);
      int tries = 0;
      const double t = preco::pre_frag_sample_kinetic_energy(st, frag, ex, rng, &tries);
      cmp(b_t, t, dv(f, 9), w);
      // Two deviates per rejection try, which is what `draws` counts on the Geant4 side.
      cmp_int(b_d, rng.n, iv(f, 10), w + " draws");
    }
  }

  // -------------------------------------------------------------------------------------------
  // 4. preco_transitions.csv - five configurations, both halves
  // -------------------------------------------------------------------------------------------
  {
    const int b_p = new_bucket("TransitionProbs", 1e-12);
    const int b_d = new_bucket("TransitionDraws", 0.0);
    const int b_x = new_bucket("PerformTransition", 0.0);
    const auto lines = read_lines(dir + "/preco_transitions.csv");
    long long gnash_rows = 0, gnash_zero = 0;
    for (std::size_t i = 1; i < lines.size(); ++i) {
      const auto f = split(lines[i]);
      if (f.size() < 17) { continue; }
      const int Z = iv(f, 0), A = iv(f, 1);
      const double e = dv(f, 2);
      const preco::Excitons ex{iv(f, 3), iv(f, 4), iv(f, 5)};
      const int config = iv(f, 6), phase = iv(f, 7);
      const std::string w = key(Z, A, e, ex.particles, ex.charged, ex.holes) + " cfg " +
                            std::to_string(config) + " ph " + std::to_string(phase);
      const deex::Fragment frag = deex::make_excited_fragment(Z, A, e, 0.0);

      // config: 0 = CEM, 1 = CEM+NGB, 2 = Gupta, 3 = Gupta+NGB, 4 = GNASH
      const int model = (config == 4) ? preco::kGNASH
                                      : ((config < 2) ? preco::kCEM : preco::kGupta);
      const bool ngb = (config == 1 || config == 3);
      CycleRng rng;
      rng.reset(phase);
      const preco::TransitionProbs tp =
          preco::transition_probability(model, frag, ex, ngb, lt, rng);
      cmp(b_p, tp.total, dv(f, 8), w + " total");
      cmp(b_p, tp.p1, dv(f, 9), w + " p1");
      cmp(b_p, tp.p2, dv(f, 10), w + " p2");
      cmp(b_p, tp.p3, dv(f, 11), w + " p3");
      cmp_int(b_d, rng.n, iv(f, 12), w + " draws_prob");
      if (config == 4) {
        ++gnash_rows;
        if (dv(f, 9) == 0.0 && dv(f, 10) == 0.0 && dv(f, 11) == 0.0) { ++gnash_zero; }
      }

      // PerformTransition. The dump reset the engine to `phase + 3` before this call, so the
      // exciton comparison is over a different slice of the cycle from the probability one -
      // which is deliberate: it means a transcription that happened to work at one phase
      // offset is still exercised at another.
      CycleRng rng2;
      rng2.reset(phase + 3);
      preco::TransitionResult tr;
      if (model == preco::kGNASH) {
        tr = preco::perform_transition_gnash(frag, ex, rng2);
      } else {
        tr = preco::perform_transition(tp, frag, ex, rng2);
      }
      cmp_int(b_x, tr.ex.particles, iv(f, 13), w + " newP");
      cmp_int(b_x, tr.ex.charged, iv(f, 14), w + " newPc");
      cmp_int(b_x, tr.ex.holes, iv(f, 15), w + " newH");
      cmp_int(b_d, rng2.n, iv(f, 16), w + " draws_perform");
    }
    // The finding this file exists for, asserted on the oracle rather than on the port: with
    // G4GNASHTransitions the three transition probabilities the model reads back are all
    // exactly zero, so `P1 <= P2+P3` is true and pre-equilibrium never runs. If a future
    // Geant4 fixes it, this fails loudly instead of the dead branch quietly coming alive.
    if (gnash_rows == 0 || gnash_zero != gnash_rows) {
      std::printf("FAIL: GNASH left a non-zero TransitionProb on %lld of %lld oracle rows - "
                  "the dead branch in precompound_transitions.cuh is no longer dead\n",
                  gnash_rows - gnash_zero, gnash_rows);
      ++fails;
    } else {
      std::printf("GNASH: P1 = P2 = P3 = 0 on all %lld oracle rows (the dead branch)\n",
                  gnash_rows);
    }
  }

  // -------------------------------------------------------------------------------------------
  // 5. preco_equilibrium.csv - the level density, n_eq and both gates
  //
  // Every line here calls a `preco::` function. The first version of this section computed
  // n_eq and both gates from the parameters instead, which made three buckets - 1,920 points at
  // a worst relative error of exactly zero - that stayed green when the corresponding
  // expressions inside precompound_model.cuh were perturbed, because the test was comparing its
  // own copy of the formula against the CSV column the formula had been dumped from. See
  // docs/RISK.md V52; the three functions exist so that this section can address them.
  // -------------------------------------------------------------------------------------------
  {
    const int b_ld = new_bucket("LevelDensity", 1e-14);
    const int b_neq = new_bucket("EquilibriumExcitonNumber", 0.0);
    const int b_gate = new_bucket("EntryGate", 0.0);
    const auto lines = read_lines(dir + "/preco_equilibrium.csv");
    for (std::size_t i = 1; i < lines.size(); ++i) {
      const auto f = split(lines[i]);
      if (f.size() < 10) { continue; }
      const int Z = iv(f, 0), A = iv(f, 1);
      const double e = dv(f, 2);
      const preco::Excitons ex{iv(f, 3), iv(f, 4), iv(f, 5)};
      const std::string w = key(Z, A, e, ex.particles, ex.charged, ex.holes);
      const bool has_levels = (data::find_manager(lt, Z, A) >= 0);
      cmp(b_ld, deex::level_density(Z, A, e, has_levels), dv(f, 6), w + " level_density");
      cmp_int(b_neq, preco::preco_equilibrium_exciton_number(Z, A, e, has_levels), iv(f, 7),
              w + " n_eq");
      cmp_int(b_gate, preco::preco_entry_gate(Z, A, e) ? 1 : 0, iv(f, 8), w + " entry_gate");
      cmp_int(b_gate, preco::preco_loop_gate(Z, A, e) ? 1 : 0, iv(f, 9), w + " loop_gate");
    }
  }

  // -------------------------------------------------------------------------------------------
  // 5b. preco_gates.csv - the four tokens the two gates differ in
  //
  // The main grid cannot separate the entry gate's `(Z < minZ && A < minA)` from the loop's
  // `(Z < minZ || A < minA)`, because every nuclide in it has either both of Z and A below the
  // limits (He4) or neither. Nor can any campaign, at any statistics: a fragment the two forms
  // disagree about passes the entry gate, fails the loop's OR on the same iteration, and reaches
  // the same handler having consumed one extra uniform - identical products. This file supplies
  // He6, Li4, H3 and Be7, and the excitation energies where `<` and `<=` differ, which is
  // exactly `U == fPrecoLowEnergy*A` and `U == fPrecoHighEnergy*A` as doubles. `%.17g`
  // round-trips, so both sides compare the same bits.
  // -------------------------------------------------------------------------------------------
  {
    const int b = new_bucket("GateBoundaries", 0.0);
    const auto lines = read_lines(dir + "/preco_gates.csv");
    if (lines.size() < 2) {
      std::printf("cannot read %s/preco_gates.csv - run ref/oracle/run.bat tables\n",
                  dir.c_str());
      return 1;
    }
    for (std::size_t i = 1; i < lines.size(); ++i) {
      const auto f = split(lines[i]);
      if (f.size() < 5) { continue; }
      const int Z = iv(f, 0), A = iv(f, 1);
      const double U = dv(f, 2);
      char buf[96];
      std::snprintf(buf, sizeof buf, "Z=%d A=%d U=%.17g", Z, A, U);
      const std::string w = buf;
      cmp_int(b, preco::preco_entry_gate(Z, A, U) ? 1 : 0, iv(f, 3), w + " entry");
      cmp_int(b, preco::preco_loop_gate(Z, A, U) ? 1 : 0, iv(f, 4), w + " loop");
    }
  }

  // -------------------------------------------------------------------------------------------
  // 6. preco_emission.csv - PerformEmission, exact under the cycle
  // -------------------------------------------------------------------------------------------
  {
    const int b_tot = new_bucket("EmissionTotalProb", 1e-12);
    const int b_id = new_bucket("EmissionChannel", 0.0);
    const int b_kin = new_bucket("EmissionKinematics", 1e-11);
    const int b_res = new_bucket("EmissionResidual", 1e-9);
    const int b_ex = new_bucket("EmissionResidualExcitons", 0.0);
    const int b_d = new_bucket("EmissionDraws", 0.0);
    const auto lines = read_lines(dir + "/preco_emission.csv");
    for (std::size_t i = 1; i < lines.size(); ++i) {
      const auto f = split(lines[i]);
      if (f.size() < 22) { continue; }
      const int Z = iv(f, 0), A = iv(f, 1);
      const double e = dv(f, 2), pz = dv(f, 3);
      preco::Excitons ex{iv(f, 4), iv(f, 5), iv(f, 6)};
      const int phase = iv(f, 7);
      const std::string w = key(Z, A, e, ex.particles, ex.charged, ex.holes) + " pz " +
                            std::to_string((int)pz) + " ph " + std::to_string(phase);
      deex::Fragment frag = deex::make_excited_fragment(Z, A, e, pz);

      CycleRng rng;
      rng.reset(phase);
      preco::EmissionChannels ch = preco::emission_probabilities(frag, ex, lt, 3);
      cmp(b_tot, ch.total, dv(f, 8), w + " total");
      if (!(ch.total > 0.0)) { continue; }
      preco::PrecoRefusal ref;
      const int kind = preco::choose_fragment(ch, rng, ref);
      const preco::PrecoProduct prod =
          preco::perform_emission(kind, ch, frag, ex, lt, rng, ref);

      cmp_int(b_id, prod.pdg, iv(f, 9), w + " pdg");
      cmp(b_kin, prod.kinetic_energy(), dv(f, 10), w + " ekin");
      cmp(b_kin, prod.momentum.v.x, dv(f, 11), w + " px");
      cmp(b_kin, prod.momentum.v.y, dv(f, 12), w + " py");
      cmp(b_kin, prod.momentum.v.z, dv(f, 13), w + " pz");
      cmp(b_kin, prod.momentum.e, dv(f, 14), w + " e");
      cmp_int(b_res, frag.z, iv(f, 15), w + " resZ");
      cmp_int(b_res, frag.a, iv(f, 16), w + " resA");
      // The residual's excitation is a difference of ~2e5 MeV numbers, so its relative
      // tolerance is looser than the four-momentum's by the ratio of the two scales.
      cmp(b_res, frag.excitation, dv(f, 17), w + " resE*");
      cmp_int(b_ex, ex.particles, iv(f, 18), w + " resP");
      cmp_int(b_ex, ex.charged, iv(f, 19), w + " resPc");
      cmp_int(b_ex, ex.holes, iv(f, 20), w + " resH");
      cmp_int(b_d, rng.n, iv(f, 21), w + " draws");
    }
  }

  // -------------------------------------------------------------------------------------------
  // 7. preco_propagate*.csv - the cascade hand-over, replayed from the dumped nucleon list
  // -------------------------------------------------------------------------------------------
  {
    const int b_esc = new_bucket("PropagateEscapedSet", 0.0);
    const int b_mom = new_bucket("PropagateEscapedMomentum", 1e-12);
    const int b_qgs = new_bucket("PropagateQGSM", 0.0);

    // The track list, verbatim from dump_precompound.cc's tspecs. It is an INPUT, so it is
    // spelled out on both sides rather than dumped; the escaped ids and the momentum sums are
    // what the oracle contributes.
    struct TSpec { int pdg; double r_fm; double ekin; };
    const TSpec tspecs[] = {
      {2212, 0.5, 5.0},  {2112, 0.5, 20.0}, {2212, 1.5, 60.0}, {2112, 1.5, 150.0},
      {2212, 8.0, 30.0}, {2112, 2.5, 10.0}, {211, 0.5, 80.0},  {2212, 0.5, 200.0},
    };
    const int kNTracks = 8;
    // 139.5701 and not the PDG 139.57018: G4PionPlus is constructed with that literal, and
    // src/core/particle.cuh already records why.
    const double m_pion = 139.5701;

    const auto nlines = read_lines(dir + "/preco_propagate_nucleons.csv");
    const auto plines = read_lines(dir + "/preco_propagate.csv");

    // case -> the nucleon list, in the dump's loop order.
    std::map<int, std::vector<preco::HitNucleon>> hit_by_case;
    std::map<int, int> caseZ, caseA;
    std::map<int, double> caseR;
    for (std::size_t i = 1; i < nlines.size(); ++i) {
      const auto f = split(nlines[i]);
      if (f.size() < 13) { continue; }
      const int c = iv(f, 0);
      caseZ[c] = iv(f, 1);
      caseA[c] = iv(f, 2);
      caseR[c] = dv(f, 3) * deex::fermi();
      if (iv(f, 6) == 0) { continue; }   // only the hit nucleons matter to Propagate
      preco::HitNucleon hn;
      hn.charge = iv(f, 5);
      hn.momentum = deex::LorentzVector(dv(f, 7), dv(f, 8), dv(f, 9), dv(f, 10));
      hn.pdg_mass = dv(f, 11);
      hn.binding_energy = dv(f, 12);
      hit_by_case[c].push_back(hn);
    }

    for (std::size_t i = 1; i < plines.size(); ++i) {
      const auto f = split(plines[i]);
      if (f.size() < 18) { continue; }
      const int c = iv(f, 0), qgsm = iv(f, 2), phase = iv(f, 3);
      const std::string w = "case " + std::to_string(c) + " ph " + std::to_string(phase);

      preco::WoundedNucleus nuc;
      nuc.a = caseA[c];
      nuc.z = caseZ[c];
      nuc.radius = caseR[c];
      nuc.hit = hit_by_case[c].data();
      nuc.n_hit = static_cast<int>(hit_by_case[c].size());

      // The finding the dump is built around: a nucleus whose hit nucleons were left as
      // G4Fancy3DNucleus::Init made them is off shell, and Propagate's QGSM test is exactly
      // that. The `qgsm` column is the dump's own evaluation of Propagate's loop on the same
      // nucleus - and it is not always `on_shell == 0`, because putting a nucleon on shell
      // through `e = sqrt(p^2 + m^2)` can still leave `mag()` a half-ulp under m. That is why
      // the flag is dumped rather than inferred; case 1 in this grid is exactly such a case.
      cmp_int(b_qgs, preco::wounded_nucleus_is_qgsm(nuc) ? 1 : 0, qgsm, w + " qgsm");

      std::vector<preco::CascadeTrack> tracks(kNTracks);
      for (int t = 0; t < kNTracks; ++t) {
        const TSpec& s = tspecs[t];
        const double m = (s.pdg == 2212) ? deex::pdg_mass_proton()
                                         : ((s.pdg == 2112) ? deex::pdg_mass_neutron()
                                                            : m_pion);
        const double pmag = std::sqrt(s.ekin * (s.ekin + 2.0 * m));
        const Vec3<double> dir = normalize(Vec3<double>{0.3, -0.5, 0.81});
        const Vec3<double> pos =
            (s.r_fm * deex::fermi()) * normalize(Vec3<double>{0.6, 0.48, -0.64});
        tracks[t].pdg = s.pdg;
        tracks[t].charge = (s.pdg == 2112) ? 0 : 1;
        tracks[t].momentum = deex::LorentzVector(pmag * dir, s.ekin + m);
        tracks[t].position = pos;
        tracks[t].creator_model_id = t;
      }

      std::vector<preco::CascadeTrack> escaped(kNTracks);
      int n_escaped = 0;
      preco::GeneratorRefusal ref;
      CycleRng rng;
      rng.reset(phase);
      // The primary projectile the dump set: a 1 GeV proton along +z. Only the QGS arm reads
      // it, and Propagate reads only its four-momentum.
      const deex::LorentzVector primary(0.0, 0.0, dv(f, 4), dv(f, 5));
      const preco::CascadeResidual res = preco::propagate_residual(
          tracks.data(), kNTracks, nuc, primary, escaped.data(), kNTracks, n_escaped, ref, rng);
      (void)res;

      cmp_int(b_esc, n_escaped, iv(f, 6), w + " n_escaped");
      std::string ids;
      deex::LorentzVector sum;
      for (int k = 0; k < n_escaped; ++k) {
        if (!ids.empty()) { ids += "|"; }
        ids += std::to_string(escaped[k].creator_model_id);
        sum += escaped[k].momentum;
      }
      cmp_int(b_esc, (ids == f[7]) ? 1 : 0, 1, w + " escaped_ids \"" + ids + "\" vs \"" + f[7] +
                                                  "\"");
      cmp(b_mom, sum.v.x, dv(f, 8), w + " sum_px");
      cmp(b_mom, sum.v.y, dv(f, 9), w + " sum_py");
      cmp(b_mom, sum.v.z, dv(f, 10), w + " sum_pz");
      cmp(b_mom, sum.e, dv(f, 11), w + " sum_e");
    }
  }

  // -------------------------------------------------------------------------------------------
  // 8. preco_deexcite*.csv - the whole model, statistically
  // -------------------------------------------------------------------------------------------
  {
    struct Campaign {
      int Z, A, p, pc, h;
      double eexc, pz;
      int N;
    };
    std::vector<Campaign> camps;
    // key -> (species key -> tally). The species key is pdg*2 + is_preco, as the dump writes.
    std::map<std::string, std::map<int, Tally>> g4;
    std::map<std::string, std::map<int, long long>> g4res, g4mult;

    auto ckey = [](int Z, int A, double e, double pz, int p, int pc, int h) {
      char buf[160];
      std::snprintf(buf, sizeof buf, "%d/%d/%g/%g/%d/%d/%d", Z, A, e, pz, p, pc, h);
      return std::string(buf);
    };

    for (const auto& line : read_lines(dir + "/preco_deexcite.csv")) {
      const auto f = split(line);
      if (f.size() < 13 || f[0] == "Z") { continue; }
      const Campaign c{iv(f, 0), iv(f, 1), iv(f, 4), iv(f, 5), iv(f, 6),
                       dv(f, 2), dv(f, 3), iv(f, 7)};
      const std::string k = ckey(c.Z, c.A, c.eexc, c.pz, c.p, c.pc, c.h);
      if (g4.find(k) == g4.end()) { camps.push_back(c); }
      int pz_ = 0, pa_ = 0;
      if (!pdg_to_za(iv(f, 8), pz_, pa_)) {
        std::printf("FAIL: preco_deexcite.csv has PDG code %d, which is not a nucleus, a "
                    "nucleon, a gamma or an electron\n", iv(f, 8));
        ++fails;
        continue;
      }
      // Folding onto (Z, A) merges the isomers of one nuclide - 877 rows of this campaign do -
      // so the counts and the energy sums accumulate rather than assign. They are additive;
      // the per-event multiplicity's second moment is not, which is why it comes from
      // preco_deexcite_species.csv below, already folded on the dump side.
      Tally& t = g4[k][za_tag_key(pz_, pa_, iv(f, 9))];
      const long long n = std::atoll(f[10].c_str());
      t.count += n;
      t.sum_e += dv(f, 11) * static_cast<double>(n);
      t.sum_e2 += dv(f, 12) * static_cast<double>(n);
    }
    // The folded tally. `count` is repeated there, so comparing it against the folding just done
    // checks the folding itself - a species key that disagreed would mean the test and the dump
    // disagree about what (Z, A) a PDG code is.
    long long n_fold_checked = 0;
    for (const auto& line : read_lines(dir + "/preco_deexcite_species.csv")) {
      const auto f = split(line);
      if (f.size() < 13 || f[0] == "Z") { continue; }
      const std::string k = ckey(iv(f, 0), iv(f, 1), dv(f, 2), dv(f, 3), iv(f, 4), iv(f, 5),
                                 iv(f, 6));
      const long long N = std::atoll(f[7].c_str());
      Tally& t = g4[k][za_tag_key(iv(f, 8), iv(f, 9), iv(f, 10))];
      const long long n = std::atoll(f[11].c_str());
      if (t.count != n) {
        std::printf("FAIL: preco_deexcite_species.csv says %lld of Z=%d A=%d preco %d in %s, "
                    "the PDG-keyed file folds to %lld\n", n, iv(f, 8), iv(f, 9), iv(f, 10),
                    k.c_str(), t.count);
        ++fails;
      }
      ++n_fold_checked;
      t.sum_k2 = dv(f, 12) * static_cast<double>(N);
      t.has_k2 = true;
    }
    if (n_fold_checked == 0) {
      std::printf("cannot read %s/preco_deexcite_species.csv - run ref/oracle/run.bat tables\n",
                  dir.c_str());
      return 1;
    }
    for (const auto& line : read_lines(dir + "/preco_deexcite_residual.csv")) {
      const auto f = split(line);
      if (f.size() < 11 || f[0] == "Z") { continue; }
      const std::string k = ckey(iv(f, 0), iv(f, 1), dv(f, 2), dv(f, 3), iv(f, 4), iv(f, 5),
                                 iv(f, 6));
      g4res[k][iv(f, 8) * 1000 + iv(f, 9)] = std::atoll(f[10].c_str());
    }
    for (const auto& line : read_lines(dir + "/preco_deexcite_mult.csv")) {
      const auto f = split(line);
      if (f.size() < 10 || f[0] == "Z") { continue; }
      const std::string k = ckey(iv(f, 0), iv(f, 1), dv(f, 2), dv(f, 3), iv(f, 4), iv(f, 5),
                                 iv(f, 6));
      g4mult[k][iv(f, 8)] = std::atoll(f[9].c_str());
    }
    if (camps.empty()) {
      std::printf("cannot read %s/preco_deexcite.csv\n", dir.c_str());
      return 1;
    }

    Buffers bufs;
    double worst_count = 0.0, worst_moment = 0.0, worst_res = 0.0, worst_mult = 0.0;
    std::string worst_count_at, worst_moment_at, worst_res_at, worst_mult_at;
    long long n_count = 0, n_moment = 0, n_res = 0, n_mult = 0;
    long long refusals = 0;

    for (const Campaign& c : camps) {
      const std::string k = ckey(c.Z, c.A, c.eexc, c.pz, c.p, c.pc, c.h);
      std::map<int, Tally> mine;                  // za_tag_key
      std::map<int, long long> myres, mymult;
      std::map<int, long long> per_event;          // species key -> count in this event
      Philox<double> rng(0x9e3779b9u, static_cast<unsigned>(c.Z * 1000 + c.A), 12345u);
      for (int n = 0; n < c.N; ++n) {
        const deex::Fragment frag =
            deex::make_excited_fragment(c.Z, c.A, c.eexc, c.pz);
        const preco::Excitons ex{c.p, c.pc, c.h};
        preco::PrecoWorkspace ws = bufs.view();
        const preco::PrecoStatus st = preco::deexcite(frag, ex, lt, pool, ws, rng);
        if (st.ref.any() || st.deex.any_refusal()) { ++refusals; }
        per_event.clear();
        int heaviestA = -1, hz = 0, ha = 0;
        for (int i = 0; i < st.n_products; ++i) {
          const deex::DeexProduct& pr = ws.products[i];
          // is_preco is positional here and by creator id in the oracle: the model appends
          // the handler's products after the pre-equilibrium ones, so the first
          // n_preco_products entries are the ejectiles.
          const int is_preco = (i < st.n_preco_products) ? 1 : 0;
          // A conversion electron is (Z = -1, A = 0), as pdg_to_za encodes it, so it is not
          // merged with the gammas.
          const int pz_ = (pr.a == 0 && pr.pdg == deex::kPdgElectron) ? -1 : pr.z;
          const int skey = za_tag_key(pz_, pr.a, is_preco);
          Tally& t = mine[skey];
          ++t.count;
          t.has_k2 = true;
          const double ek = deex::deex_kinetic_energy(pr);
          t.sum_e += ek;
          t.sum_e2 += ek * ek;
          ++per_event[skey];
          if (pr.a > heaviestA) { heaviestA = pr.a; hz = pr.z; ha = pr.a; }
        }
        for (const auto& kv : per_event) {
          mine[kv.first].sum_k2 += static_cast<double>(kv.second) * kv.second;
        }
        ++myres[1000 * hz + ha];
        ++mymult[st.n_preco_products];
      }

      // Counts and energy moments, per ((Z, A), tag).
      auto label = [](int skey) {
        const int tag = skey % 2;
        const int za = skey / 2;
        return "Z=" + std::to_string(za / 1000 - 500) + " A=" + std::to_string(za % 1000) +
               " preco " + std::to_string(tag);
      };
      for (const auto& kv : g4[k]) {
        const int skey = kv.first;
        const Tally& go = kv.second;
        const Tally& po = mine[skey];
        const double z = multiplicity_z(po.count, go.count, c.N, tally_var(po, c.N),
                                        tally_var(go, c.N));
        if (z > worst_count) {
          worst_count = z;
          worst_count_at = k + " " + label(skey) + " port " + std::to_string(po.count) +
                           " g4 " + std::to_string(go.count);
        }
        ++n_count;
        const int za = skey / 2;
        if (po.count > 1 && go.count > 1 && is_fixed_mass(za / 1000 - 500, za % 1000)) {
          const double pm = po.sum_e / po.count, pv = po.sum_e2 / po.count - pm * pm;
          const double gm = go.sum_e / go.count, gv = go.sum_e2 / go.count - gm * gm;
          const double zm = moment_z(pm, pv, po.count, gm, gv, go.count);
          if (zm > worst_moment) {
            worst_moment = zm;
            worst_moment_at = k + " " + label(skey);
          }
          ++n_moment;
        }
      }
      // Species the port makes and the oracle does not. `var = 1` is a deliberately generous
      // stand-in for a distribution with no oracle side at all - a species the oracle never
      // produced is a failure at any multiplicity above a handful.
      for (const auto& kv : mine) {
        if (g4[k].count(kv.first) != 0) { continue; }
        // var2 = -1: the oracle never made this species, so there is no oracle-side spread to
        // scale by and multiplicity_z falls back to the Poisson stand-in.
        const double z = multiplicity_z(kv.second.count, 0, c.N, tally_var(kv.second, c.N),
                                        -1.0);
        if (z > worst_count) {
          worst_count = z;
          worst_count_at = k + " EXTRA " + label(kv.first) + " port " +
                           std::to_string(kv.second.count);
        }
        ++n_count;
      }
      // The residual distribution, per event, Binomial.
      for (const auto& kv : g4res[k]) {
        const double z = binomial_z(myres[kv.first], kv.second, c.N);
        if (z > worst_res) {
          worst_res = z;
          worst_res_at = k + " res " + std::to_string(kv.first / 1000) + "/" +
                         std::to_string(kv.first % 1000);
        }
        ++n_res;
      }
      // The pre-equilibrium multiplicity histogram - the "fraction that reaches equilibrium
      // without emitting" is its zero bin, and it is the sharpest single number here.
      for (const auto& kv : g4mult[k]) {
        const double z = binomial_z(mymult[kv.first], kv.second, c.N);
        if (z > worst_mult) {
          worst_mult = z;
          worst_mult_at = k + " nej " + std::to_string(kv.first);
        }
        ++n_mult;
      }
      for (const auto& kv : mymult) {
        if (g4mult[k].count(kv.first) != 0) { continue; }
        const double z = binomial_z(kv.second, 0, c.N);
        if (z > worst_mult) {
          worst_mult = z;
          worst_mult_at = k + " EXTRA nej " + std::to_string(kv.first);
        }
        ++n_mult;
      }
    }

    const double kSigma = 5.0;
    std::printf("\nDeExcite, %d campaigns x 20,000 events:\n", (int)camps.size());
    std::printf("  species counts    %6lld comparisons, worst %5.2f sigma  %s\n", n_count,
                worst_count, worst_count_at.c_str());
    std::printf("  energy moments    %6lld comparisons, worst %5.2f sigma  %s\n", n_moment,
                worst_moment, worst_moment_at.c_str());
    std::printf("  residual (Z, A)   %6lld comparisons, worst %5.2f sigma  %s\n", n_res,
                worst_res, worst_res_at.c_str());
    std::printf("  preco multiplicity%6lld comparisons, worst %5.2f sigma  %s\n", n_mult,
                worst_mult, worst_mult_at.c_str());
    std::printf("  refusals reported %6lld; Poisson fallbacks %lld of %lld count "
                "comparisons\n", refusals, n_poisson_fallback, n_count);
    if (worst_count > kSigma) { ++fails; std::printf("FAIL: species counts\n"); }
    if (worst_moment > kSigma) { ++fails; std::printf("FAIL: energy moments\n"); }
    if (worst_res > kSigma) { ++fails; std::printf("FAIL: residual distribution\n"); }
    if (worst_mult > kSigma) { ++fails; std::printf("FAIL: preco multiplicity\n"); }
  }

  // -------------------------------------------------------------------------------------------
  // 9. preco_apply*.csv - ApplyYourself, i.e. the initial fragment, through its consequences
  //
  // `apply_yourself_initial_fragment` is the only deliverable in this package with no exact
  // oracle, and it cannot have one: G4PreCompoundModel::ApplyYourself builds
  // `G4Fragment(A + Ap, Z + Zp, thePrimary.Get4Momentum() + (0,0,0,M(A,Z)))` as a local and
  // hands back only the final state. Rebuilding that fragment in the dump would compare the port
  // against a second copy of the same two lines and would assume, on both sides, the two things
  // most worth measuring - that the mass added is the TARGET's and not the compound's, and which
  // four-momentum it is added to. So the whole final state is compared instead: a wrong target
  // mass moves E* and every spectrum with it, a wrong (A + Ap, Z + Zp) moves the residual, and a
  // wrong (2, 1, 1) exciton configuration moves the pre-equilibrium multiplicity.
  //
  // The last case, a 1 MeV proton on Pb208, is below the entry gate's low limit (U ~ 5 MeV
  // against fPrecoLowEnergy*209 = 20.9 MeV), so its whole `n_preco_ejectiles` histogram must sit
  // in the zero bin on both sides.
  // -------------------------------------------------------------------------------------------
  {
    struct ACase { int pdg, Z, A; double ekin; int N; };
    std::vector<ACase> camps;
    std::map<std::string, std::map<int, Tally>> g4;
    std::map<std::string, std::map<int, long long>> g4mult;

    auto akey = [](int pdg, int Z, int A, double ekin) {
      char buf[96];
      std::snprintf(buf, sizeof buf, "%d/%d/%d/%g", pdg, Z, A, ekin);
      return std::string(buf);
    };

    for (const auto& line : read_lines(dir + "/preco_apply.csv")) {
      const auto f = split(line);
      if (f.size() < 10 || f[0] == "proj_pdg") { continue; }
      const ACase c{iv(f, 0), iv(f, 1), iv(f, 2), dv(f, 3), iv(f, 4)};
      const std::string k = akey(c.pdg, c.Z, c.A, c.ekin);
      if (g4.find(k) == g4.end()) { camps.push_back(c); }
      int pz_ = 0, pa_ = 0;
      if (!pdg_to_za(iv(f, 5), pz_, pa_)) {
        std::printf("FAIL: preco_apply.csv has PDG code %d, which is not a nucleus, a nucleon, "
                    "a gamma or an electron\n", iv(f, 5));
        ++fails;
        continue;
      }
      Tally& t = g4[k][za_tag_key(pz_, pa_, iv(f, 6))];
      const long long n = std::atoll(f[7].c_str());
      t.count += n;
      t.sum_e += dv(f, 8) * static_cast<double>(n);
      t.sum_e2 += dv(f, 9) * static_cast<double>(n);
    }
    long long n_fold_checked = 0;
    for (const auto& line : read_lines(dir + "/preco_apply_species.csv")) {
      const auto f = split(line);
      if (f.size() < 10 || f[0] == "proj_pdg") { continue; }
      const std::string k = akey(iv(f, 0), iv(f, 1), iv(f, 2), dv(f, 3));
      const long long N = std::atoll(f[4].c_str());
      Tally& t = g4[k][za_tag_key(iv(f, 5), iv(f, 6), iv(f, 7))];
      const long long n = std::atoll(f[8].c_str());
      if (t.count != n) {
        std::printf("FAIL: preco_apply_species.csv says %lld of Z=%d A=%d preco %d in %s, the "
                    "PDG-keyed file folds to %lld\n", n, iv(f, 5), iv(f, 6), iv(f, 7),
                    k.c_str(), t.count);
        ++fails;
      }
      ++n_fold_checked;
      t.sum_k2 = dv(f, 9) * static_cast<double>(N);
      t.has_k2 = true;
    }
    if (n_fold_checked == 0) {
      std::printf("cannot read %s/preco_apply_species.csv - run ref/oracle/run.bat tables\n",
                  dir.c_str());
      return 1;
    }
    for (const auto& line : read_lines(dir + "/preco_apply_mult.csv")) {
      const auto f = split(line);
      if (f.size() < 7 || f[0] == "proj_pdg") { continue; }
      g4mult[akey(iv(f, 0), iv(f, 1), iv(f, 2), dv(f, 3))][iv(f, 5)] =
          std::atoll(f[6].c_str());
    }
    if (camps.empty()) {
      std::printf("cannot read %s/preco_apply.csv - run ref/oracle/run.bat tables\n",
                  dir.c_str());
      return 1;
    }

    Buffers bufs;
    double worst_count = 0.0, worst_moment = 0.0, worst_mult = 0.0;
    std::string worst_count_at, worst_moment_at, worst_mult_at;
    long long n_count = 0, n_moment = 0, n_mult = 0, refused_projectiles = 0;

    for (const ACase& c : camps) {
      const std::string k = akey(c.pdg, c.Z, c.A, c.ekin);
      std::map<int, Tally> mine;
      std::map<int, long long> mymult;
      std::map<int, long long> per_event;
      Philox<double> rng(0x85ebca6bu, static_cast<unsigned>(c.pdg + c.Z * 1000 + c.A), 999u);
      for (int n = 0; n < c.N; ++n) {
        preco::Excitons ex;
        bool refused = false;
        const deex::Fragment frag = preco::apply_yourself_initial_fragment(
            c.pdg, c.ekin, Vec3<double>{0.0, 0.0, 1.0}, c.Z, c.A, ex, refused);
        if (refused) { ++refused_projectiles; continue; }
        preco::PrecoWorkspace ws = bufs.view();
        const preco::PrecoStatus st = preco::deexcite(frag, ex, lt, pool, ws, rng);
        per_event.clear();
        for (int i = 0; i < st.n_products; ++i) {
          const deex::DeexProduct& pr = ws.products[i];
          const int is_preco = (i < st.n_preco_products) ? 1 : 0;
          const int pz_ = (pr.a == 0 && pr.pdg == deex::kPdgElectron) ? -1 : pr.z;
          const int skey = za_tag_key(pz_, pr.a, is_preco);
          Tally& t = mine[skey];
          ++t.count;
          t.has_k2 = true;
          const double ek = deex::deex_kinetic_energy(pr);
          t.sum_e += ek;
          t.sum_e2 += ek * ek;
          ++per_event[skey];
        }
        for (const auto& kv : per_event) {
          mine[kv.first].sum_k2 += static_cast<double>(kv.second) * kv.second;
        }
        ++mymult[st.n_preco_products];
      }

      auto label = [](int skey) {
        const int tag = skey % 2;
        const int za = skey / 2;
        return "Z=" + std::to_string(za / 1000 - 500) + " A=" + std::to_string(za % 1000) +
               " preco " + std::to_string(tag);
      };
      for (const auto& kv : g4[k]) {
        const Tally& go = kv.second;
        const Tally& po = mine[kv.first];
        const double z = multiplicity_z(po.count, go.count, c.N, tally_var(po, c.N),
                                        tally_var(go, c.N));
        if (z > worst_count) {
          worst_count = z;
          worst_count_at = k + " " + label(kv.first) + " port " + std::to_string(po.count) +
                           " g4 " + std::to_string(go.count);
        }
        ++n_count;
        const int za = kv.first / 2;
        if (po.count > 1 && go.count > 1 && is_fixed_mass(za / 1000 - 500, za % 1000)) {
          const double pm = po.sum_e / po.count, pv = po.sum_e2 / po.count - pm * pm;
          const double gm = go.sum_e / go.count, gv = go.sum_e2 / go.count - gm * gm;
          const double zm = moment_z(pm, pv, po.count, gm, gv, go.count);
          if (zm > worst_moment) {
            worst_moment = zm;
            worst_moment_at = k + " " + label(kv.first);
          }
          ++n_moment;
        }
      }
      for (const auto& kv : mine) {
        if (g4[k].count(kv.first) != 0) { continue; }
        // var2 = -1: the oracle never made this species, so there is no oracle-side spread to
        // scale by and multiplicity_z falls back to the Poisson stand-in.
        const double z = multiplicity_z(kv.second.count, 0, c.N, tally_var(kv.second, c.N),
                                        -1.0);
        if (z > worst_count) {
          worst_count = z;
          worst_count_at = k + " EXTRA " + label(kv.first) + " port " +
                           std::to_string(kv.second.count);
        }
        ++n_count;
      }
      for (const auto& kv : g4mult[k]) {
        const double z = binomial_z(mymult[kv.first], kv.second, c.N);
        if (z > worst_mult) {
          worst_mult = z;
          worst_mult_at = k + " nej " + std::to_string(kv.first);
        }
        ++n_mult;
      }
      for (const auto& kv : mymult) {
        if (g4mult[k].count(kv.first) != 0) { continue; }
        const double z = binomial_z(kv.second, 0, c.N);
        if (z > worst_mult) {
          worst_mult = z;
          worst_mult_at = k + " EXTRA nej " + std::to_string(kv.first);
        }
        ++n_mult;
      }
    }

    const double kSigma = 5.0;
    std::printf("\nApplyYourself, %d cases x 20,000 events:\n", (int)camps.size());
    std::printf("  species counts    %6lld comparisons, worst %5.2f sigma  %s\n", n_count,
                worst_count, worst_count_at.c_str());
    std::printf("  energy moments    %6lld comparisons, worst %5.2f sigma  %s\n", n_moment,
                worst_moment, worst_moment_at.c_str());
    std::printf("  preco multiplicity%6lld comparisons, worst %5.2f sigma  %s\n", n_mult,
                worst_mult, worst_mult_at.c_str());
    if (worst_count > kSigma) { ++fails; std::printf("FAIL: ApplyYourself species counts\n"); }
    if (worst_moment > kSigma) { ++fails; std::printf("FAIL: ApplyYourself energy moments\n"); }
    if (worst_mult > kSigma) { ++fails; std::printf("FAIL: ApplyYourself multiplicity\n"); }
    // Every case in the oracle is a nucleon, so a refusal here would mean the port rejected a
    // projectile Geant4 accepted.
    if (refused_projectiles != 0) {
      std::printf("FAIL: apply_yourself_initial_fragment refused %lld nucleon projectiles\n",
                  refused_projectiles);
      ++fails;
    }
    // The pion ApplyYourself raises a FatalException for, as a refusal the port reports rather
    // than a fragment it invents. Asserted, because a silent acceptance would build a compound
    // nucleus with the wrong charge.
    {
      preco::Excitons ex;
      bool refused = false;
      preco::apply_yourself_initial_fragment(211, 100.0, Vec3<double>{0.0, 0.0, 1.0}, 6, 12, ex,
                                             refused);
      if (!refused) {
        std::printf("FAIL: apply_yourself_initial_fragment accepted a pi+, which is a "
                    "FatalException in G4PreCompoundModel::ApplyYourself\n");
        ++fails;
      }
    }
  }

  // -------------------------------------------------------------------------------------------

  std::printf("\n%-28s %10s %12s\n", "bucket", "points", "worst rel");
  for (const Bucket& b : buckets) {
    const bool bad = b.worst > b.tol;
    std::printf("%-28s %10lld %12.3e %s%s\n", b.name, b.n, b.worst, bad ? "FAIL " : "",
                bad ? b.where.c_str() : "");
    if (bad) { ++fails; }
    if (b.n == 0) {
      std::printf("FAIL: bucket %s compared nothing\n", b.name);
      ++fails;
    }
  }
  std::printf("\nInitialize early-return rows (stale binding/resmass not compared): %lld\n",
              n_early_return);
  std::printf("%s\n", fails == 0 ? "PASS" : "FAIL");
  return fails == 0 ? 0 : 1;
}
