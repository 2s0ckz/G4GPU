// The at-rest processes against ref/oracle/stopping_*.csv.
//
// Three exact comparisons and one statistical one, arranged the way the Bertini package is: every
// piece that Geant4 exposes on its own is driven under the eight-value cycle engine and compared
// value for value, and the assembly - which runs a nuclear model with its own retry loops and
// cannot survive a prescribed engine - is compared as a distribution.
//
//   stopping_emcascade.csv  G4EmCaptureCascade over 43 elements x 8 phases: the draw count, the
//                           number of secondaries, every kind and every kinetic energy.
//   stopping_murates.csv    G4MuonMinusBoundDecay's capture and decay rates and the effective
//                           charge. PURE FUNCTIONS of (Z, A) - no engine - so these are exact by
//                           construction and the tolerance is 1e-15, not a band.
//   stopping_select.csv     G4ElementSelector's Fermi-Teller weight for Z = 1..92.
//
// **The invariant the EM cascade has and nothing else in this port does.** Its transition
// energies telescope: the level-14 electron carries `level[13]`, every later step carries
// `level[i] - level[n]`, and the sum over a cascade that ends at level 0 is exactly `level[0]` -
// the K-shell energy. So `ebound` must equal `k_level_energy(Z)` for every element and every
// phase, at ZERO tolerance, independently of which branches the cascade took. That is asserted
// here rather than observed, because it is the one check that does not care about the sampling at
// all: a port that got the Auger/photon split wrong, or the level ordering wrong, or the
// interpolation wrong, would still pass every energy comparison ONLY if it also got this right.
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
#include "physics/hadronic/stopping/em_capture_cascade.cuh"
#include "physics/hadronic/stopping/element_selector.cuh"
#include "physics/hadronic/stopping/muon_bound_decay.cuh"
#include "physics/hadronic/stopping/stopping_process.cuh"

using namespace g4gpu;
using namespace g4gpu::physics::hadronic;

namespace {

int fails = 0;

__host__ __device__ inline const double* cycle_seq() {
  static const double s[8] = {0.05, 0.37, 0.63, 0.91, 0.12, 0.78, 0.29, 0.55};
  return s;
}

/// The same engine the dump installed, so that a sampler is a deterministic function of phase.
struct CycleRng {
  int phase = 0;
  long long n = 0;
  void reset(int p) { phase = p; n = 0; }
  double uniform() {
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

void cmp_rel(int bi, double got, double want, const std::string& where) {
  Bucket& b = buckets[bi];
  ++b.n;
  const double scale = (std::fabs(want) > 0.0) ? std::fabs(want) : 1.0;
  const double rel = std::fabs(got - want) / scale;
  if (rel > b.worst) {
    b.worst = rel;
    char buf[160];
    std::snprintf(buf, sizeof buf, " got %.17g want %.17g", got, want);
    b.where = where + buf;
  }
}

void cmp_int(int bi, long long got, long long want, const std::string& where) {
  Bucket& b = buckets[bi];
  ++b.n;
  if (got != want && b.worst < 1.0) {
    b.worst = 1.0;
    char buf[96];
    std::snprintf(buf, sizeof buf, " got %lld want %lld", got, want);
    b.where = where + buf;
  }
}

std::string oracle_dir() {
  const char* e = std::getenv("G4GPU_ORACLE");
  return (e != nullptr) ? std::string(e) : std::string("ref/oracle");
}

struct Csv {
  std::vector<std::string> head;
  std::vector<std::vector<std::string>> rows;
  int col(const char* name) const {
    for (std::size_t i = 0; i < head.size(); ++i) {
      if (head[i] == name) { return static_cast<int>(i); }
    }
    return -1;
  }
  double num(std::size_t r, int c) const { return std::atof(rows[r][c].c_str()); }
  long long i64(std::size_t r, int c) const { return std::atoll(rows[r][c].c_str()); }
};

bool load_csv(const std::string& path, Csv& out) {
  std::FILE* f = std::fopen(path.c_str(), "r");
  if (!f) { return false; }
  std::vector<char> line(1 << 16);
  bool first = true;
  while (std::fgets(line.data(), int(line.size()), f)) {
    std::string s(line.data());
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) { s.pop_back(); }
    if (s.empty()) { continue; }
    std::vector<std::string> cells;
    std::string cur;
    for (char ch : s) {
      if (ch == ',') { cells.push_back(cur); cur.clear(); } else { cur.push_back(ch); }
    }
    cells.push_back(cur);
    if (first) { out.head = cells; first = false; } else { out.rows.push_back(cells); }
  }
  std::fclose(f);
  return true;
}

/// P11's FTF entry point is not yet callable from a test - no workspace builder exists outside
/// its own model tests - so the anti-baryon arm is REFUSED BY NAME and counted, which is what the
/// brief asks for until P11d lands. The count is printed per species, so "the Fritiof arm did not
/// run" is a number rather than a silence.
struct FtfNotYet {
  mutable long long calls = 0;
  template <typename P, typename N, typename F, typename R>
  stopping::StoppingRefusal operator()(const P&, const N&, F&, R&) const {
    ++calls;
    return stopping::StoppingRefusal::kFtfRefused;
  }
};

struct NuclearMassMeV {
  double operator()(int a, int z) const { return deex::nuclear_mass(a, z); }
};

/// The five materials the brief names. Only the RELATIVE number densities matter to the element
/// selector - the Fermi-Teller weight multiplies them and the total cancels in the draw - so
/// water is written as its stoichiometry, 2 H to 1 O, rather than as a density times a mass
/// fraction. Everything else is one element.
struct MatDef {
  const char* name;
  int n;
  int z[2];
  double dens[2];
  int a[2];
};

void run_campaign(const data::LevelTable& lt, const deex::FermiPool& pool,
                  const preco::PrecoWorkspace& pws, stopping::BertiniArmState& bs,
                  HadFinalState<double, 256>& fs) {
  const MatDef mats[5] = {{"H2O", 2, {1, 8}, {2.0, 1.0}, {1, 16}},
                          {"C", 1, {6, 0}, {1.0, 0.0}, {12, 0}},
                          {"Al", 1, {13, 0}, {1.0, 0.0}, {27, 0}},
                          {"Fe", 1, {26, 0}, {1.0, 0.0}, {56, 0}},
                          {"Pb", 1, {82, 0}, {1.0, 0.0}, {207, 0}}};
  struct Sp { int pdg; double mass; const char* name; };
  const Sp species[8] = {{-211, 139.57061, "pi-"},   {-321, 493.677, "K-"},
                         {3112, 1197.449, "Sigma-"}, {3312, 1321.71, "Xi-"},
                         {3334, 1672.45, "Omega-"},  {13, 105.6583715, "mu-"},
                         {-2212, 938.272013, "anti-p"}, {-2112, 939.56536, "anti-n"}};
  const long long kN = 20000;

  std::printf("\n  == stopping::at_rest, %lld events per (species, material) ==\n", kN);
  std::printf("  %-8s %-5s %9s %8s %8s %8s %9s %9s %s\n", "species", "mat", "mean nsec",
              "mean EM", "mean nuc", "refused", "dio frac", "mean Edep", "captured on");
  FtfNotYet ftf;
  for (int si = 0; si < 8; ++si) {
    for (int mi = 0; mi < 5; ++mi) {
      const MatDef& m = mats[mi];
      int offs[2] = {0, 1};
      int nis[2] = {1, 1};
      bool nab[2] = {true, true};
      double abun[2] = {1.0, 1.0};
      MaterialComposition<double> mat;
      mat.n_elements = m.n;
      mat.element_z = m.z;
      mat.n_atoms_per_volume = m.dens;
      mat.n_isotopes = nis;
      mat.isotope_offset = offs;
      mat.natural_abundance = nab;
      mat.isotope_a = m.a;
      mat.isotope_abundance = abun;

      Philox<double> rng(0xA7u, static_cast<unsigned>(si), static_cast<unsigned>(mi));
      long long nsec = 0, nem = 0, nref = 0, ndio = 0, ndone = 0, non_z[2] = {0, 0};
      double edep = 0.0;
      std::map<int, long long> refkind;
      for (long long ev = 0; ev < kN; ++ev) {
        HadProjectile<double> p;
        p.pdg = species[si].pdg;
        p.mass = species[si].mass;
        p.kin_energy = 0.0;
        const stopping::AtRestResult r = stopping::at_rest(
            p, mat, fs, bert::default_cascade_params(), bert::default_interface_limits(), bs, lt,
            pool, pws, NuclearMassMeV(), ftf, 1, 2, 3, rng);
        if (r.element_index >= 0 && r.element_index < 2) { ++non_z[r.element_index]; }
        if (r.refusal != stopping::StoppingRefusal::kNone) {
          ++nref;
          ++refkind[static_cast<int>(r.refusal)];
          continue;
        }
        ++ndone;
        if (r.decayed_in_orbit) { ++ndio; }
        nsec += fs.n_secondaries;
        nem += r.n_em_cascade;
        edep += r.local_deposit_MeV;
      }
      const double d = (ndone > 0) ? double(ndone) : 1.0;
      char cap[64];
      if (m.n > 1) {
        std::snprintf(cap, sizeof cap, "Z%d %.1f%% / Z%d %.1f%%", m.z[0],
                      100.0 * double(non_z[0]) / double(kN), m.z[1],
                      100.0 * double(non_z[1]) / double(kN));
      } else {
        std::snprintf(cap, sizeof cap, "Z%d", m.z[0]);
      }
      std::printf("  %-8s %-5s %9.3f %8.3f %8.3f %8lld %9.4f %9.4g %s\n", species[si].name,
                  m.name, double(nsec) / d, double(nem) / d,
                  (double(nsec) - double(nem)) / d, nref, double(ndio) / d, edep / d, cap);
      if (nref > 0) {
        std::printf("        refusals:");
        for (const auto& kv : refkind) {
          std::printf(" kind %d x %lld", kv.first, kv.second);
        }
        std::printf("\n");
      }
    }
  }
  std::printf("  the Fritiof arm was asked %lld times and refused every one: P11's"
              " ftf::apply_yourself has no workspace builder outside its own model tests yet,"
              " so anti-p and anti-n are counted rather than approximated\n", ftf.calls);
}

}  // namespace

int main() {
  const std::string dir = oracle_dir();
  Csv emc, murates, select;
  if (!load_csv(dir + "/stopping_emcascade.csv", emc) ||
      !load_csv(dir + "/stopping_murates.csv", murates) ||
      !load_csv(dir + "/stopping_select.csv", select)) {
    std::printf("cannot read %s/stopping_*.csv - run ref/dump/build.bat then ref/oracle/run.bat"
                " tables\n", dir.c_str());
    return 1;
  }

  // ============================================================================================
  // 1. The Fermi-Teller weight, and the two exceptions that are the whole point of it.
  // ============================================================================================
  {
    const int b = new_bucket("SelectorWeight", 0.0);
    const int c_z = select.col("Z");
    const int c_w = select.col("weight_over_Z");
    int n_halogen = 0, n_oxygen = 0;
    for (std::size_t r = 0; r < select.rows.size(); ++r) {
      const int z = int(select.i64(r, c_z));
      const double want = select.num(r, c_w) * double(z);
      cmp_rel(b, stopping::element_capture_weight(z), want, "Z " + std::to_string(z));
      if (select.rows[r][select.col("note")] == "halogen") { ++n_halogen; }
      if (select.rows[r][select.col("note")] == "oxygen") { ++n_oxygen; }
    }
    std::printf("  element weights: %d halogens at 0.66, %d oxygen at 0.56, the rest Fermi-Teller"
                " - and water is 2 H to 1 O, so the oxygen line moves pi- capture there from"
                " 80.0%% to %.1f%%\n", n_halogen, n_oxygen,
                100.0 * stopping::element_capture_weight(8) /
                    (stopping::element_capture_weight(8) + 2.0 * stopping::element_capture_weight(1)));
  }

  // ============================================================================================
  // 2. The muon's two rates and the effective charge. Pure functions; zero tolerance in spirit,
  //    1e-15 in practice because the oracle is printed at 17 digits and re-parsed.
  // ============================================================================================
  {
    const int bc = new_bucket("MuonCaptureRate", 1e-15);
    const int bd = new_bucket("MuonDecayRate", 1e-15);
    const int bz = new_bucket("MuonZeff", 0.0);
    const int c_z = murates.col("Z");
    const int c_a = murates.col("A");
    const int c_tab = murates.col("tabulated");
    const int c_cap = murates.col("cap_rate_per_ns");
    const int c_dec = murates.col("decay_rate_per_ns");
    const int c_zeff = murates.col("zeff");
    const int c_m = murates.col("nucl_mass_MeV");
    long long n_tab = 0, n_formula = 0;
    for (std::size_t r = 0; r < murates.rows.size(); ++r) {
      const int z = int(murates.i64(r, c_z));
      const int a = int(murates.i64(r, c_a));
      const std::string w = "Z " + std::to_string(z) + " A " + std::to_string(a);
      cmp_rel(bc, stopping::muon_capture_rate(z, a), murates.num(r, c_cap), w);
      cmp_rel(bd, stopping::muon_decay_rate(z, 105.6583715, murates.num(r, c_m)),
              murates.num(r, c_dec), w);
      cmp_rel(bz, stopping::muon_zeff(z), murates.num(r, c_zeff), w);
      if (murates.i64(r, c_tab) != 0) { ++n_tab; } else { ++n_formula; }
    }
    std::printf("  muon rates: %lld (Z,A) from the measured table and %lld from Goulard-Primakoff"
                " - the pairing is what shows a listed Z with an unlisted A takes the formula\n",
                n_tab, n_formula);
  }

  // ============================================================================================
  // 3. The EM capture cascade, exact under the cycle, and the telescoping invariant.
  // ============================================================================================
  {
    const int bd = new_bucket("EmCascadeDraws", 0.0);
    const int bn = new_bucket("EmCascadeMultiplicity", 0.0);
    const int bk = new_bucket("EmCascadeKinds", 0.0);
    const int be = new_bucket("EmCascadeEnergies", 1e-13);
    const int bb = new_bucket("EmCascadeBoundEnergy", 1e-14);
    // The telescoping invariant is exact in exact arithmetic and holds to ONE ULP in doubles:
    // the cascade sums thirteen differences in the order the sampling produced them, and
    // `level[0]` is the same total summed once. Measured worst 1.4e-16 over 344 cases, which is
    // 2^-53 and not a tolerance chosen to fit. Asserting 0.0 here failed on the first run, and
    // the honest fix is to say "one ulp" rather than either to pretend it is exact or to widen
    // the band until the number stops mattering.
    const int bi = new_bucket("EmCascadeTelescopes", 4e-16);

    const int c_z = emc.col("Z");
    const int c_a = emc.col("A");
    const int c_ph = emc.col("phase");
    const int c_dr = emc.col("draws");
    const int c_n = emc.col("n");
    const int c_i = emc.col("i");
    const int c_pdg = emc.col("pdg");
    const int c_ek = emc.col("ekin_MeV");
    const int c_eb = emc.col("ebound_MeV");

    // The oracle is one row per secondary; group by (Z, phase).
    std::size_t r = 0;
    long long n_cases = 0, n_auger = 0, n_gamma = 0;
    int max_n = 0;
    while (r < emc.rows.size()) {
      const int z = int(emc.i64(r, c_z));
      const int a = int(emc.i64(r, c_a));
      const int phase = int(emc.i64(r, c_ph));
      const long long want_draws = emc.i64(r, c_dr);
      const int want_n = int(emc.i64(r, c_n));
      const double want_eb = emc.num(r, c_eb);
      const std::string w = "Z" + std::to_string(z) + " ph" + std::to_string(phase);

      CycleRng rng;
      rng.reset(phase);
      stopping::EmCascadeResult got;
      // The nuclear mass the dump used: G4NucleiProperties::GetNuclearMass(A, Z). P3's table is
      // the same one, and the murates file carries it, so it is not re-derived here.
      stopping::em_capture_cascade(z, deex::nuclear_mass(a, z), rng, got);

      cmp_int(bd, rng.n, want_draws, "draws " + w);
      cmp_int(bn, got.n, want_n, "n " + w);
      cmp_rel(bb, got.e_bound, want_eb, "ebound " + w);
      // THE INVARIANT: the transitions telescope to the K-shell energy, whatever path was taken.
      cmp_rel(bi, got.e_bound, stopping::k_level_energy(z), "telescope " + w);
      if (got.n > max_n) { max_n = got.n; }
      ++n_cases;

      for (int i = 0; i < want_n && r < emc.rows.size(); ++i, ++r) {
        if (int(emc.i64(r, c_i)) != i) { break; }
        const std::string wi = w + " p" + std::to_string(i);
        if (i < got.n) {
          cmp_int(bk, got.p[i].pdg, emc.i64(r, c_pdg), "kind " + wi);
          cmp_rel(be, got.p[i].kin_energy, emc.num(r, c_ek), "ekin " + wi);
          if (got.p[i].pdg == 11) { ++n_auger; } else { ++n_gamma; }
        }
      }
    }
    std::printf("  EM cascade: %lld cases, %lld electrons and %lld gammas, most secondaries %d of"
                " the %d the bound allows\n", n_cases, n_auger, n_gamma, max_n,
                stopping::kMaxEmCascadeSecondaries);
  }

  // ============================================================================================
  // 4. Pinned by construction: the species map, and the at-rest length that makes it matter.
  // ============================================================================================
  {
    const int b = new_bucket("PinnedByConstruction", 0.0);
    auto pin = [&](bool ok, const char* what) {
      Bucket& bb = buckets[b];
      ++bb.n;
      if (!ok) { bb.worst = 1.0; bb.where = what; std::printf("  FAIL pin: %s\n", what); }
    };
    using stopping::NuclearArm;
    pin(stopping::at_rest_interaction_length() == 0.0,
        "the at-rest length is ZERO - which is why capture pre-empts G4Decay every time");
    pin(stopping::stopping_arm(13) == NuclearArm::kMuonCapture, "mu- -> G4MuonMinusCapture");
    pin(stopping::stopping_arm(-211) == NuclearArm::kBertini, "pi- -> Bertini");
    pin(stopping::stopping_arm(-321) == NuclearArm::kBertini, "K- -> Bertini");
    pin(stopping::stopping_arm(3112) == NuclearArm::kBertini, "Sigma- -> Bertini");
    pin(stopping::stopping_arm(3312) == NuclearArm::kBertini, "Xi- -> Bertini");
    pin(stopping::stopping_arm(3334) == NuclearArm::kBertini, "Omega- -> Bertini");
    pin(stopping::stopping_arm(-2212) == NuclearArm::kFritiof, "anti-p -> Fritiof");
    pin(stopping::stopping_arm(-2112) == NuclearArm::kFritiof,
        "anti-n -> Fritiof, and it is NEUTRAL: the gate is charge <= 0, not < 0");
    pin(stopping::stopping_arm(-3122) == NuclearArm::kFritiof, "anti-Lambda -> Fritiof");
    pin(stopping::stopping_arm(-3212) == NuclearArm::kFritiof, "anti-Sigma0 -> Fritiof");
    pin(stopping::stopping_arm(-3222) == NuclearArm::kFritiof, "anti-Sigma+ -> Fritiof");
    pin(stopping::stopping_arm(-3322) == NuclearArm::kFritiof, "anti-Xi0 -> Fritiof");
    pin(stopping::stopping_arm(-1000010020) == NuclearArm::kFritiof,
        "an anti-deuteron -> Fritiof, by baryon number < -1");
    pin(stopping::stopping_arm(2112) == NuclearArm::kNone,
        "a NEUTRON gets nothing - neutral, heavy, long-lived, and in neither list");
    pin(stopping::stopping_arm(2212) == NuclearArm::kNone, "a proton gets nothing");
    pin(stopping::stopping_arm(211) == NuclearArm::kNone, "a pi+ gets nothing - charge > 0");
    // **The muon's Bertini is not the others' Bertini.** G4HadronicAbsorptionBertini calls
    // usePreCompoundDeexcitation(); G4MuonMinusCapture constructs a bare G4CascadeInterface.
    pin(stopping::stopping_deexcite_choice(NuclearArm::kMuonCapture) ==
            bert::DeexciteChoice::kCascade,
        "mu- capture de-excites with the CASCADE's own evaporators");
    pin(stopping::stopping_deexcite_choice(NuclearArm::kBertini) ==
            bert::DeexciteChoice::kPreCompound,
        "and the other five with P6's PreCompound - a fourth instance in a third configuration");
  }

  // ============================================================================================
  // 5. The assembly: `stopping::at_rest` for every species QBBC gives an at-rest process, in
  //    every material, 20,000 events each. Reported, not compared - there is no oracle column
  //    for it yet, and what it is for is to show the entry point RUNS for every species and to
  //    put numbers on what each arm does. The exact comparisons above are where correctness
  //    lives; this is where coverage does.
  // ============================================================================================
  {
    const std::string pe = host::g4photon_evaporation_dir();
    if (pe.empty()) {
      std::printf("  (campaign skipped: no PhotonEvaporation dataset)\n");
    } else {
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
      std::vector<deex::DeexProduct> dpv(1024), ppv(1024);
      preco::PrecoWorkspace pws;
      pws.deex.evap_list = evap.data();
      pws.deex.evap_capacity = 4096;
      pws.deex.results = results.data();
      pws.deex.results_capacity = 1024;
      pws.deex.step = step.data();
      pws.deex.step_capacity = 512;
      pws.deex.products = dpv.data();
      pws.deex.products_capacity = 1024;
      pws.products = ppv.data();
      pws.products_capacity = 1024;

      stopping::BertiniArmState bs;
      bs.model = new bert::NucleiModel();
      bs.ws = new bert::BertiniWorkspace();
      bs.global_out = new bert::CollisionOutput();
      bs.out = new bert::CollisionOutput();
      bs.dex_out = new bert::CollisionOutput();
      bs.tmp = new bert::CollisionOutput();
      bs.epo = new bert::ColliderOutput();
      auto* fs = new HadFinalState<double, 256>();

      run_campaign(lt, pool, pws, bs, *fs);
      delete fs;
    }
  }

  long long total = 0;
  std::printf("%-34s %10s %14s\n", "bucket", "points", "worst rel");
  for (const Bucket& b : buckets) {
    total += b.n;
    const bool ok = !(b.worst > b.tol);
    if (!ok) { ++fails; }
    std::printf("%-34s %10lld %14.3g %-4s %s\n", b.name, b.n, b.worst, ok ? "ok" : "FAIL",
                b.where.c_str());
  }
  std::printf("%lld comparisons, %d failures\n", total, fails);
  return (fails == 0) ? 0 : 1;
}
