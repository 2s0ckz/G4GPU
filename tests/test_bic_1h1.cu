// P18: `G4BinaryCascade::Propagate1H1` - the Binary cascade on a HYDROGEN target - against
// ref/oracle/bic_1h1*.csv.
//
// `bic::apply_yourself` sends a target of A == 1 to `bic/propagate_1h1.cuh` exactly where
// `G4BinaryCascade::ApplyYourself` sends it to `Propagate1H1` (G4BinaryCascade.cc:329), inside
// the same two retry loops the cascade runs in. This file checks that arm three ways, and they
// are three different kinds of claim.
//
//   1. **The tape: every uniform and every product, against Geant4's own stream.**
//      `ref/dump/dump_bic.cc`'s `write_bic_1h1` wraps the PUBLIC `ApplyYourself` in a
//      HepJamesRandom that records every value it serves - from before the one-nucleon
//      `G4Fancy3DNucleus::Init` to the last resonance decay - for {p, n, pi+, pi-} at {100, 400,
//      800, 1400} MeV on H1, twenty events each. The port replays each tape and must consume
//      exactly as many uniforms, never read past the end, and produce the same secondaries in the
//      same order with the same definitions, creator id and parent resonance (definition and
//      keV-rounded id), and the same four-momenta to the tolerance P9d's cascade tape uses. The
//      dump also replays `Propagate1H1`'s body out of public pieces on a second engine, and
//      where that replay reproduces the real call (`probe_ok`) its counts - the uniforms the
//      nucleus took, the uniforms `GetSpherePoint` took, the number of `Scatter` calls, whether
//      the 200 tries ran out, the number of decays - are compared too, so a failure says WHERE.
//
//   2. **The campaign: the same sixteen cases, 20,000 Geant4 events against 200,000 of the
//      port's**, on its own Philox stream. Species yields, their kinetic-energy means and the
//      multiplicity, with the per-event variances and the five-sigma band `tests/test_bic_apply.cu`
//      uses; plus two rates that are this arm's own - the events whose 200 tries ran out and
//      returned the last ELASTIC scatter, which Geant4 cannot report but which are exactly its
//      events with no secondary carrying a parent resonance, and the events per parent
//      resonance, which is the channel selection seen through what decayed.
//
//   3. **Conservation, per event, with no oracle at all.** A free-nucleon reaction has no
//      nucleus to hide energy in and no de-excitation: the products must add up to the
//      projectile plus a nucleon at rest in energy, momentum, charge and baryon number, event by
//      event, to the rounding of the arithmetic. This is the sharpest statement in the file,
//      because it needs nothing to compare against.
//
//   4. **Where the resonances turn on.** A scan of the elastic-return fraction for protons and
//      neutrons from 300 to 700 MeV, Geant4's 2,000 events a point against the port's 20,000.
//
// **What Geant4 does where no resonance can form is part of the answer.** `done` is set only by
// a short-lived product, so a nucleon below the energy at which `G4CollisionNN`'s resonance
// partials open burns all 200 tries and returns the 200th elastic scatter - and that energy is
// not the physical pion threshold near 290 MeV: every 400 MeV nucleon event does it too, in
// Geant4 and here, and section 4 says where it stops.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

#include "core/rng.cuh"
#include "data/level_data.cuh"
#include "host/g4data.cuh"
#include "physics/hadronic/bic/binary_cascade.cuh"

using namespace g4gpu;

namespace {

int fails = 0;

// ---------------------------------------------------------------------------------------------
// Buckets, the shape tests/test_bic_imr.cu uses
// ---------------------------------------------------------------------------------------------

struct Bucket {
  const char* name;
  long long n = 0;
  double worst = 0.0;
  std::string where;
  double tol = 0.0;
};
std::vector<Bucket> buckets;

int new_bucket(const char* name, double tol) {
  Bucket b;
  b.name = name;
  b.tol = tol;
  buckets.push_back(b);
  return static_cast<int>(buckets.size()) - 1;
}

/// An exact comparison: the bucket's worst becomes 1 at the first mismatch, and its tolerance is
/// zero, so one mismatch fails it.
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

/// Relative to `scale` - a product's own total energy - and not to the component: a transverse
/// momentum of 0.3 MeV on a 1.4 GeV pion is a cancellation, and dividing by it would turn double
/// rounding into a factor of a thousand.
void cmp_scaled(int bi, double got, double want, double scale, const std::string& where) {
  Bucket& b = buckets[bi];
  ++b.n;
  const double d = std::fabs(got - want) / ((scale > 0.0) ? scale : 1.0);
  if (d > b.worst) {
    b.worst = d;
    b.where = where;
  }
}

/// A statistical comparison: `z` in sigma against the bucket's band.
void cmp_sigma(int bi, double z, const std::string& where) {
  Bucket& b = buckets[bi];
  ++b.n;
  if (z > b.worst) {
    b.worst = z;
    b.where = where;
  }
}

// ---------------------------------------------------------------------------------------------
// CSV reading, the same shape tests/test_bic_apply.cu uses
// ---------------------------------------------------------------------------------------------

std::string oracle_dir() {
  const char* env = std::getenv("G4GPU_ORACLE");
  return (env != nullptr) ? std::string(env) : std::string("ref/oracle");
}

std::vector<std::vector<std::string>> read_csv(const std::string& name) {
  std::vector<std::vector<std::string>> rows;
  const std::string path = oracle_dir() + "/" + name;
  FILE* f = std::fopen(path.c_str(), "r");
  if (f == nullptr) {
    std::printf("MISSING %s\n", path.c_str());
    ++fails;
    return rows;
  }
  char line[8192];
  bool header = true;
  while (std::fgets(line, sizeof line, f) != nullptr) {
    if (header) { header = false; continue; }
    std::vector<std::string> cols;
    std::string cur;
    for (const char* p = line; *p != '\0'; ++p) {
      if (*p == ',') { cols.push_back(cur); cur.clear(); }
      else if (*p != '\n' && *p != '\r') { cur.push_back(*p); }
    }
    cols.push_back(cur);
    if (!cols.empty()) { rows.push_back(cols); }
  }
  std::fclose(f);
  return rows;
}

double dv(const std::vector<std::string>& f, std::size_t i) {
  return (i < f.size()) ? std::atof(f[i].c_str()) : 0.0;
}
int iv(const std::vector<std::string>& f, std::size_t i) {
  return (i < f.size()) ? std::atoi(f[i].c_str()) : 0;
}
long long lv(const std::vector<std::string>& f, std::size_t i) {
  return (i < f.size()) ? std::atoll(f[i].c_str()) : 0;
}
std::string sv(const std::vector<std::string>& f, std::size_t i) {
  return (i < f.size()) ? f[i] : std::string();
}

/// The longest tape here is a 100 MeV nucleon's: 200 elastic scatters of four or five uniforms
/// each plus the nucleus. 32,768 is room for eight times that.
inline constexpr int kTapeMax = 32768;

/// A RECORDED random stream, replayed value by value - the same engine tests/test_bic_imr.cu and
/// tests/test_bic_apply.cu drive `Propagate` and `Interact` with, and for the same reason: the
/// number of uniforms `Propagate1H1` consumes depends on how many scatters came back elastic, so
/// a prescribed sequence cannot check it and a recorded one checks it by construction. Past the
/// end it returns a 64-value ladder rather than a constant, so a rejection sampler downstream
/// cannot spin on a fixed point, and `overrun` says the port asked for more than Geant4 did.
struct TapeRng {
  double value[kTapeMax] = {};
  int n_values = 0;
  int n = 0;
  int overrun = 0;
  __host__ __device__ double uniform() {
    if (n >= n_values || n >= kTapeMax) {
      const double v = (2.0 * static_cast<double>(overrun % 64) + 1.0) / 128.0;
      ++overrun;
      ++n;
      return v;
    }
    return value[n++];
  }
};

/// Charge and baryon number of every species a pion or a nucleon on a free nucleon can leave
/// behind below 1.5 GeV once `Propagate1H1` has decayed its resonances: the nucleons and the
/// three pions; the eta, which is not short-lived and so is left for the transport to decay; and
/// the strange pairs an N* reaches through `Lambda K` and `Sigma K`, the neutral kaon arriving as
/// K0S or K0L because the kinetic-track constructor substitutes it (docs/RISK.md V150). A code
/// outside this list is a failure of its own, reported by name, rather than a silent zero in the
/// balance.
bool species_qb(int pdg, int& q, int& b) {
  switch (pdg) {
    case 2212: q = 1;  b = 1; return true;
    case 2112: q = 0;  b = 1; return true;
    case 211:  q = 1;  b = 0; return true;
    case -211: q = -1; b = 0; return true;
    case 111:  q = 0;  b = 0; return true;
    case 221:  q = 0;  b = 0; return true;
    case 22:   q = 0;  b = 0; return true;
    case 321:  q = 1;  b = 0; return true;
    case -321: q = -1; b = 0; return true;
    case 310:  q = 0;  b = 0; return true;
    case 130:  q = 0;  b = 0; return true;
    case 3122: q = 0;  b = 1; return true;
    case 3222: q = 1;  b = 1; return true;
    case 3212: q = 0;  b = 1; return true;
    case 3112: q = -1; b = 1; return true;
    default:   q = 0;  b = 0; return false;
  }
}

double proj_mass(int pdg) {
  if (pdg == 2212) { return units::proton_mass_c2<double>(); }
  if (pdg == 2112) { return units::neutron_mass_c2<double>(); }
  return bic::pdg_mass_pion_charged();
}

/// Per-species tallies, as tests/test_bic_apply.cu keeps them.
struct Tally {
  long long count = 0;
  double sum_e = 0.0;
  double sum_e2 = 0.0;
  double sum_k2 = 0.0;
};

/// A per-event MEAN compared between two runs of different length, with each side's per-event
/// multiplicity variance - `tests/test_bic_apply.cu`'s `yield_z`.
double yield_z(double m1, double v1, long long ev1, double m2, double v2, long long ev2) {
  if (ev1 < 2 || ev2 < 2) { return 0.0; }
  const double s2 = ((v1 > 0.0) ? v1 : 0.0) / static_cast<double>(ev1) +
                    ((v2 > 0.0) ? v2 : 0.0) / static_cast<double>(ev2);
  if (!(s2 > 0.0)) { return (m1 == m2) ? 0.0 : 1.e9; }
  return std::fabs(m1 - m2) / std::sqrt(s2);
}

/// Two binomial proportions, k1/n1 against k2/n2, pooled. Zero against zero is exact agreement;
/// a rate one side never produced and the other did is compared against the pooled rate, which
/// is what makes "the port never exhausts where Geant4 always does" a large number and not 0/0.
double proportion_z(long long k1, long long n1, long long k2, long long n2) {
  if (n1 < 1 || n2 < 1) { return 0.0; }
  const double p1 = static_cast<double>(k1) / static_cast<double>(n1);
  const double p2 = static_cast<double>(k2) / static_cast<double>(n2);
  const double p = static_cast<double>(k1 + k2) / static_cast<double>(n1 + n2);
  const double s2 = p * (1.0 - p) * (1.0 / static_cast<double>(n1) + 1.0 / static_cast<double>(n2));
  if (!(s2 > 0.0)) { return (p1 == p2) ? 0.0 : 1.e9; }
  return std::fabs(p1 - p2) / std::sqrt(s2);
}

/// Caller-owned storage for `bic::apply_yourself`, as the transport's slot hands it over - the
/// A == 1 arm reads one nucleon, the scratch, the track pool as its working list, the product
/// list and the channel buffers, and none of the rest.
struct Storage {
  std::vector<bic::Nucleon> nucleons = std::vector<bic::Nucleon>(256);
  std::vector<deex::Vec3d> mom = std::vector<deex::Vec3d>(256);
  std::vector<double> fermi = std::vector<double>(256);
  std::vector<bic::NucleusSortEntry> sums = std::vector<bic::NucleusSortEntry>(256);
  std::vector<double> flat = std::vector<double>(bic::kFlatBlock);
  std::vector<double> pfield = std::vector<double>(bic::kMaxFieldTable);
  std::vector<double> nfield = std::vector<double>(bic::kMaxFieldTable);
  std::vector<bic::CascadeTrack> pool = std::vector<bic::CascadeTrack>(512);
  std::vector<bic::imr::CollisionInitialState> colls =
      std::vector<bic::imr::CollisionInitialState>(64);
  std::vector<bic::imr::ConcreteChannel> chans =
      std::vector<bic::imr::ConcreteChannel>(bic::imr::kConcreteChannelCount);
  bic::CascadeBuffers buffers;
  std::vector<bic::CascadeProduct> products = std::vector<bic::CascadeProduct>(256);
  std::vector<bic::CascadeProduct> preco = std::vector<bic::CascadeProduct>(64);
  int n_chan = 0;

  Storage() {
    n_chan = bic::imr::build_concrete_channels(chans.data(), bic::imr::kConcreteChannelCount);
  }
  bic::BicStorage view() {
    bic::BicStorage s;
    s.nucleons = nucleons.data();
    s.scratch.momentum = mom.data();
    s.scratch.fermi_p = fermi.data();
    s.scratch.test_sums = sums.data();
    s.scratch.flat_block = flat.data();
    s.scratch.capacity = 256;
    s.proton_field = pfield.data();
    s.neutron_field = nfield.data();
    s.field_capacity = bic::kMaxFieldTable;
    s.cascade.pool = pool.data();
    s.cascade.pool_capacity = static_cast<int>(pool.size());
    s.cascade.collisions = colls.data();
    s.cascade.collision_capacity = static_cast<int>(colls.size());
    s.cascade.channels = chans.data();
    s.cascade.n_channels = n_chan;
    s.cascade.buffers = &buffers;
    s.cascade.products = products.data();
    s.cascade.product_capacity = static_cast<int>(products.size());
    s.cascade.preco_products = preco.data();
    s.cascade.preco_capacity = static_cast<int>(preco.size());
    return s;
  }
};

/// P6's buffers. The A == 1 arm never de-excites anything, but `apply_yourself` takes them
/// because a nucleon below 45 MeV on hydrogen goes to the precompound model instead.
struct PrecoBuffers {
  std::vector<deex::Fragment> evap, results, step;
  std::vector<deex::DeexProduct> deex_products, products;
  PrecoBuffers() : evap(512), results(256), step(256), deex_products(256), products(256) {}
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

physics::hadronic::HadProjectile<double> make_projectile(int pdg, double ekin) {
  physics::hadronic::HadProjectile<double> p;
  p.pdg = pdg;
  p.mass = proj_mass(pdg);
  p.kin_energy = ekin;
  p.charge = (pdg == 2212 || pdg == 211) ? 1.0 : ((pdg == -211) ? -1.0 : 0.0);
  p.baryon_number = (pdg == 2212 || pdg == 2112) ? 1 : 0;
  return p;
}

}  // namespace

int main() {
  const std::string pe = host::g4photon_evaporation_dir();
  if (pe.empty()) {
    std::printf("PhotonEvaporation dataset not found - g4data.cuh could not resolve "
                "G4LEVELGAMMADATA\n");
    return 1;
  }
  data::LevelTableStorage lts;
  data::read_all_level_data(
      lts, pe, data::kLevelZMax,
      [](int z, int a) { return deex::shell_correction(a, z); },
      [](int z, int a) { return deex::level_manager_level_density(z, a); });
  const data::LevelTable lt = lts.view();
  deex::FermiPoolStorage ps;
  deex::build_fermi_pool(ps, lt);
  const deex::FermiPool fpool = ps.view();

  static Storage st;
  static PrecoBuffers pb;
  const double m_target = units::proton_mass_c2<double>();

  // ===========================================================================================
  // 0. `IsShortLived` for everything `Scatter` can hand `Propagate1H1`
  // ===========================================================================================
  //
  // The retry loop turns on this one flag and nothing else - a short-lived product ends it, an
  // elastic one does not - so the flag has to be Geant4's for EVERY species the scatterer can
  // produce on a nucleon target, and not merely for the ones a tape happened to meet. The set is
  // the port's own channel tables: both outgoing species of all 306 concrete nucleon-nucleon
  // channels, every charge state of the 25 meson-baryon resonance multiplets, and the elastic
  // outputs. The answer is `bic_imr_decaytable.csv`'s `shortlived` column, which the dump reads
  // off each `G4ParticleDefinition` - so this compares the port against Geant4 and not against
  // itself. A Delta and a rho are named explicitly, because the brief asked about them.
  const int b_sl = new_bucket("IsShortLived(species)", 0.0);
  {
    std::map<int, int> g4_short;
    for (const auto& r : read_csv("bic_imr_decaytable.csv")) { g4_short[iv(r, 0)] = iv(r, 4); }
    std::map<int, int> produced;  // pdg -> the short-lived flag a scatter product must carry
    for (int c = 0; c < st.n_chan; ++c) {
      produced[st.chans[c].out1] = -1;
      produced[st.chans[c].out2] = -1;
    }
    for (int c = 0; c < bic::imr::kMesonBaryonToResonanceCount; ++c) {
      const bic::imr::MesonBaryonChannelSpec& sp = bic::imr::meson_baryon_channels()[c];
      const int* codes = sp.is_delta ? bic::imr::delta_codes(sp.mass)
                                     : bic::imr::nstar_codes(sp.mass);
      for (int k = 0; k < (sp.is_delta ? 4 : 2); ++k) { produced[codes[k]] = -1; }
    }
    for (int p : {2212, 2112, 211, -211, 111}) { produced[p] = -1; }
    for (int p : {1114, 2114, 2214, 2224, 113, 213, -213}) {
      cmp_int(b_sl, g4_short.count(p) ? g4_short[p] : -1, 1,
              "Geant4 calls " + std::to_string(p) + " short-lived");
      produced[p] = -1;
    }
    for (const auto& kv : produced) {
      const int p = kv.first;
      if (!g4_short.count(p)) {
        cmp_int(b_sl, p, 0, "species " + std::to_string(p) + " is absent from the oracle table");
        continue;
      }
      cmp_int(b_sl, bic::kinetic_decay_is_short_lived(p) ? 1 : 0, g4_short[p],
              "IsShortLived(" + std::to_string(p) + ")");
    }
    std::printf("IsShortLived checked for %zu species the scatterer can produce\n",
                produced.size());
  }

  // ===========================================================================================
  // 1. The tape
  // ===========================================================================================
  const int b_tape = new_bucket("TapeStructure", 0.0);
  // P9d's tolerance for the cascade tape (tests/test_bic_imr.cu, `PropagateMomenta`), relative to
  // each product's own total energy. `Propagate1H1` is one scatter and one or two decays, far
  // less arithmetic than a cascade, so this is generous; the measured worst is printed.
  const int b_mom = new_bucket("TapeMomenta", 3e-13);
  const int b_diag = new_bucket("TapeProbeCounts", 0.0);
  long long tape_events = 0, tape_exhausted = 0, tape_probe_ok = 0;
  {
    const auto rows = read_csv("bic_1h1_tape.csv");
    const auto tvals = read_csv("bic_1h1_tapeval.csv");
    const auto tfs = read_csv("bic_1h1_tapefs.csv");
    if (rows.empty()) {
      std::printf("bic_1h1_tape.csv is empty\n");
      ++fails;
    }
    // Index the values and secondaries by (case, ev) once; the files are a few hundred thousand
    // rows and a scan per event would be quadratic.
    std::map<std::string, std::vector<const std::vector<std::string>*>> vals_of, fs_of;
    for (const auto& r : tvals) { vals_of[sv(r, 0) + "#" + sv(r, 1)].push_back(&r); }
    for (const auto& r : tfs) { fs_of[sv(r, 0) + "#" + sv(r, 1)].push_back(&r); }

    for (const auto& r : rows) {
      const std::string cname = sv(r, 0);
      const int ev = iv(r, 1);
      const int pdg = iv(r, 2);
      const double ekin = dv(r, 3);
      const int tz = iv(r, 4), ta = iv(r, 5);
      const int ndraws = iv(r, 6);
      const std::string want_status = sv(r, 7);
      const int want_nsec = iv(r, 8);
      const int bicid = iv(r, 9);
      const int draws_init = iv(r, 10), draws_sphere = iv(r, 11);
      const int want_scatters = iv(r, 12), want_exhausted = iv(r, 13), want_decays = iv(r, 14);
      const bool probe_ok = iv(r, 15) == 1;
      const std::string where = cname + " ev " + std::to_string(ev);
      const std::string key = cname + "#" + std::to_string(ev);
      ++tape_events;

      static TapeRng tape;
      tape.n_values = 0;
      for (const auto* tr : vals_of[key]) {
        const int i = iv(*tr, 2);
        if (i >= 0 && i < kTapeMax) {
          tape.value[i] = dv(*tr, 3);
          if (i + 1 > tape.n_values) { tape.n_values = i + 1; }
        }
      }
      cmp_int(b_tape, tape.n_values, ndraws, where + " tape length");

      // WHERE, before WHAT: the one-nucleon nucleus and the sphere point on a copy of the tape,
      // so a divergence in either is named rather than seen downstream as a wrong scatter.
      {
        static TapeRng t2;
        t2 = tape;
        t2.n = 0;
        t2.overrun = 0;
        bic::Nucleus3D nuc;
        bic::BicStorage sv2 = st.view();
        nuc.nucleons = sv2.nucleons;
        nuc.capacity = 256;
        const bic::NucleusReport nrep = bic::nucleus_init(nuc, sv2.scratch, ta, tz, t2);
        cmp_int(b_diag, nrep.fatal() ? 1 : 0, 0, where + " nucleus built");
        cmp_int(b_diag, t2.n, draws_init, where + " uniforms the one-nucleon Init took");
        const double m = proj_mass(pdg);
        const double pmag = std::sqrt(ekin * (ekin + 2.0 * m));
        const int before = t2.n;
        (void)bic::get_sphere_point(1.1 * (nuc.outer_radius() + 3.0 * deex::fermi()),
                                    Vec3<double>{0.0, 0.0, pmag}, t2);
        cmp_int(b_diag, t2.n - before, draws_sphere, where + " uniforms GetSpherePoint took");
      }

      tape.n = 0;
      tape.overrun = 0;
      const physics::hadronic::HadProjectile<double> proj = make_projectile(pdg, ekin);
      physics::hadronic::HadNucleus tgt;
      tgt.z = tz;
      tgt.a = ta;
      tgt.l = 0;
      bic::BicStorage store = st.view();
      st.buffers = bic::CascadeBuffers{};
      static bic::BicFinalState fs;
      bic::BicRefusal ref;
      bic::BicReport rep;
      preco::PrecoWorkspace pws = pb.view();
      bic::apply_yourself(proj, tgt, lt, fpool, pws, store, tape, fs, ref, rep);
      if (ref.any()) {
        std::printf("REFUSED %s: hydrogen=%d (decay_null=%d decay_refused=%d unknown=%d "
                    "charge=%d cap=%d pdg=%d) nucleus=%d capacity=%d\n",
                    where.c_str(), ref.hydrogen ? 1 : 0, ref.h1.decay_null ? 1 : 0,
                    ref.h1.decay_refused ? 1 : 0, ref.h1.unknown_species ? 1 : 0,
                    ref.h1.charge_imbalance ? 1 : 0, ref.h1.capacity ? 1 : 0,
                    ref.h1.refused_pdg, ref.nucleus ? 1 : 0, ref.capacity ? 1 : 0);
        ++fails;
        continue;
      }
      const std::string got_status =
          (fs.status == physics::hadronic::HadFinalStateStatus::kIsAlive) ? "isAlive"
                                                                         : "stopAndKill";
      cmp_int(b_tape, (got_status == want_status) ? 1 : 0, 1,
              where + " status " + got_status + " want " + want_status);
      cmp_int(b_tape, tape.n, ndraws, where + " uniforms consumed");
      cmp_int(b_tape, tape.overrun, 0, where + " tape not overrun");
      cmp_int(b_tape, fs.n_secondaries, want_nsec, where + " secondary count");
      if (probe_ok) {
        ++tape_probe_ok;
        cmp_int(b_diag, rep.h1.scatters, want_scatters, where + " Scatter calls");
        cmp_int(b_diag, rep.h1.exhausted ? 1 : 0, want_exhausted, where + " 200 tries ran out");
        cmp_int(b_diag, rep.h1.n_decays, want_decays, where + " resonances decayed");
        cmp_int(b_diag, rep.outer_tries, 1, where + " one outer turn");
      }
      if (rep.h1.exhausted) { ++tape_exhausted; }

      int seen = 0;
      for (const auto* srp : fs_of[key]) {
        const auto& sr = *srp;
        const int i = iv(sr, 2);
        if (i < 0 || i >= fs.n_secondaries) { continue; }
        const physics::hadronic::HadSecondary<double>& s = fs.secondaries[i];
        const bic::CascadeProduct& cp = st.products[i];
        const std::string sw = where + " sec " + std::to_string(i);
        cmp_int(b_tape, s.pdg, iv(sr, 3), sw + " pdg");
        const double e = s.total_energy();
        const double p = s.momentum();
        const double sc = std::fabs(dv(sr, 7));
        cmp_scaled(b_mom, p * s.direction.x, dv(sr, 4), sc, sw + " px");
        cmp_scaled(b_mom, p * s.direction.y, dv(sr, 5), sc, sw + " py");
        cmp_scaled(b_mom, p * s.direction.z, dv(sr, 6), sc, sw + " pz");
        cmp_scaled(b_mom, e, dv(sr, 7), sc, sw + " e");
        cmp_int(b_tape, (s.time == dv(sr, 8)) ? 1 : 0, 1, sw + " time");
        cmp_int(b_tape, s.creator_model_id, iv(sr, 9), sw + " creator model id");
        cmp_int(b_tape, s.creator_model_id, bicid, sw + " creator id is theBIC_ID");
        cmp_int(b_tape, cp.parent_resonance_pdg, iv(sr, 10), sw + " parent resonance");
        cmp_int(b_tape, cp.parent_resonance_id, iv(sr, 11), sw + " parent resonance id");
        ++seen;
      }
      cmp_int(b_tape, seen, fs.n_secondaries, where + " secondary rows read");
    }
  }

  // ===========================================================================================
  // 2 and 3. The campaign, and conservation per event
  // ===========================================================================================
  const int b_yield = new_bucket("SpeciesYield(sigma)", 5.0);
  const int b_ekin = new_bucket("SpeciesEkin(sigma)", 5.0);
  const int b_mult = new_bucket("Multiplicity(sigma)", 5.0);
  const int b_exh = new_bucket("ElasticReturnRate(sigma)", 5.0);
  const int b_alive = new_bucket("AliveRate(sigma)", 5.0);
  const int b_parent = new_bucket("ParentEvents(sigma)", 5.0);
  // Per event, no oracle: the products against projectile + nucleon at rest. MEASURED over the
  // 3,200,000 events, on two Philox streams (the second is what an anti-vacuity run that moved
  // the nucleus off the stream left behind): energy 3.5e-11 and 4.5e-10 MeV, momentum 9.8e-11
  // and 7.7e-10 MeV, both on 1.4 GeV pions - a 2.4 GeV system through a scatter, a decay chain
  // and the boosts between them, a few hundred ulps. The bound is the larger rounded up a
  // decade: a physics loss arrives at 1e-3 MeV and a wrong mass at 1 MeV.
  const int b_bal_e = new_bucket("BalanceE(MeV/event)", 1e-8);
  const int b_bal_p = new_bucket("BalanceP(MeV/event)", 1e-8);
  const int b_qb = new_bucket("ChargeBaryon(events)", 0.0);
  // The exhaustion flag the port reports and the definition the oracle uses must be the same
  // event set, event by event - see `n_exhausted` in the dump.
  const int b_exh_def = new_bucket("ElasticReturnIsParentless", 0.0);
  // Geant4's own mean total energy against projectile + nucleon: it conserves too, and a
  // projectile or target mass convention that differed between the two sides would show here.
  const int b_g4_bal = new_bucket("OracleMeanE(MeV)", 1e-6);

  const long long kPortEvents = 200000;
  const auto status_rows = read_csv("bic_1h1_status.csv");
  const auto species_rows = read_csv("bic_1h1.csv");
  const auto parent_rows = read_csv("bic_1h1_parent.csv");
  if (status_rows.empty()) {
    std::printf("bic_1h1_status.csv is empty\n");
    ++fails;
  }
  long long total_refused = 0, total_events = 0;
  long long r_decay_null = 0, r_decay_refused = 0, r_unknown = 0, r_charge = 0, r_cap = 0;
  long long r_nucleus = 0, r_capacity = 0;
  std::printf("\n%-12s %8s %9s %9s %11s %11s %9s %9s %8s\n", "case", "G4 N", "G4 alive",
              "port alive", "G4 elastic", "port elast", "G4 mult", "port mult", "refused");

  int icase = 0;
  for (const auto& row : status_rows) {
    const std::string cname = sv(row, 0);
    const int pz = iv(row, 1), pa = iv(row, 2);
    const double ekin = dv(row, 3);
    const int tz = iv(row, 4), ta = iv(row, 5);
    const long long o_n = lv(row, 6);
    const double o_mean_e = dv(row, 11);
    const double o_mean_mult = dv(row, 13);
    const long long o_alive = lv(row, 14);
    const long long o_exhausted = lv(row, 15);
    const double o_mean_mult2 = dv(row, 16);
    const long long o_kill = o_n - o_alive;
    const int pdg = (pa == 1) ? ((pz == 1) ? 2212 : 2112) : ((pz == 1) ? 211 : -211);
    const physics::hadronic::HadProjectile<double> proj = make_projectile(pdg, ekin);
    physics::hadronic::HadNucleus tgt;
    tgt.z = tz;
    tgt.a = ta;
    tgt.l = 0;
    const double e_in = ekin + proj.mass + m_target;
    const double p_in = std::sqrt(ekin * (ekin + 2.0 * proj.mass));
    const int q_in = pz + tz;
    const int b_in = pa + ta;

    // ---- the oracle's species and parents for this case
    std::map<int, Tally> go;
    for (const auto& s : species_rows) {
      if (sv(s, 0) != cname) { continue; }
      Tally& t = go[iv(s, 7)];
      t.count = lv(s, 8);
      t.sum_e = dv(s, 9) * static_cast<double>(t.count);
      t.sum_e2 = dv(s, 10) * static_cast<double>(t.count);
      t.sum_k2 = dv(s, 11) * static_cast<double>(o_kill);
    }
    std::map<int, long long> go_parent;
    for (const auto& s : parent_rows) {
      if (sv(s, 0) == cname) { go_parent[iv(s, 2)] = lv(s, 3); }
    }
    cmp_sigma(b_g4_bal, std::fabs(o_mean_e - e_in), cname + " Geant4 mean total energy");

    // ---- the port
    Philox<double> rng(0x1a11b0dau, static_cast<unsigned>(icase), 18u);
    std::map<int, Tally> mine;
    std::map<int, long long> my_parent;
    std::map<int, int> per_event, parent_seen;
    long long n_alive = 0, n_kill = 0, n_sec = 0, n_exh = 0, n_refused = 0;
    long long sum_scatters = 0;
    double sum_mult2 = 0.0;
    for (long long ev = 0; ev < kPortEvents; ++ev) {
      bic::BicStorage store = st.view();
      static bic::BicFinalState fs;
      bic::BicRefusal ref;
      bic::BicReport rep;
      preco::PrecoWorkspace pws = pb.view();
      bic::apply_yourself(proj, tgt, lt, fpool, pws, store, rng, fs, ref, rep);
      if (ref.any()) {
        ++n_refused;
        if (ref.h1.decay_null) { ++r_decay_null; }
        if (ref.h1.decay_refused) { ++r_decay_refused; }
        if (ref.h1.unknown_species) { ++r_unknown; }
        if (ref.h1.charge_imbalance) { ++r_charge; }
        if (ref.h1.capacity) { ++r_cap; }
        if (ref.nucleus) { ++r_nucleus; }
        if (ref.capacity) { ++r_capacity; }
        continue;
      }
      if (fs.status == physics::hadronic::HadFinalStateStatus::kIsAlive) {
        ++n_alive;
        continue;
      }
      ++n_kill;
      sum_scatters += rep.h1.scatters;
      per_event.clear();
      parent_seen.clear();
      double tot_e = 0.0;
      Vec3<double> tot_p{0.0, 0.0, 0.0};
      int q_out = 0, b_out = 0;
      bool any_parent = false;
      n_sec += fs.n_secondaries;
      sum_mult2 += static_cast<double>(fs.n_secondaries) * fs.n_secondaries;
      for (int i = 0; i < fs.n_secondaries; ++i) {
        const physics::hadronic::HadSecondary<double>& s = fs.secondaries[i];
        Tally& t = mine[s.pdg];
        ++t.count;
        t.sum_e += s.kin_energy;
        t.sum_e2 += s.kin_energy * s.kin_energy;
        ++per_event[s.pdg];
        tot_e += s.total_energy();
        tot_p = tot_p + s.momentum() * s.direction;
        int q = 0, b = 0;
        if (!species_qb(s.pdg, q, b)) {
          cmp_int(b_qb, s.pdg, 0, cname + " ev " + std::to_string(ev) + " unexpected species");
        }
        q_out += q;
        b_out += b;
        const int ppdg = st.products[i].parent_resonance_pdg;
        ++parent_seen[ppdg];
        if (ppdg != 0) { any_parent = true; }
      }
      for (const auto& kv : per_event) {
        mine[kv.first].sum_k2 += static_cast<double>(kv.second) * kv.second;
      }
      for (const auto& kv : parent_seen) { ++my_parent[kv.first]; }
      if (rep.h1.exhausted) { ++n_exh; }
      cmp_int(b_exh_def, rep.h1.exhausted ? 1 : 0, any_parent ? 0 : 1,
              cname + " ev " + std::to_string(ev));
      const std::string evw = cname + " ev " + std::to_string(ev);
      cmp_int(b_qb, q_out, q_in, evw + " charge");
      cmp_int(b_qb, b_out, b_in, evw + " baryon number");
      // THE BALANCE IS TO THE ROUNDING, and the reason it can be: `Scatter` conserves the pair's
      // four-momentum exactly in its own arithmetic (two-body kinematics in the CM frame and a
      // boost back), every decay does in the parent's rest frame, and each stable product is on
      // its definition's mass shell - so `G4DynamicParticle`'s (total energy, momentum)
      // constructor keeps the energy it is handed and the sum is the entrance energy up to the
      // ulps of a few boosts. Momentum is compared through `sqrt(T(T+2m))` times a unit vector,
      // which is what the port's `HadSecondary` stores, so it carries a few more.
      cmp_scaled(b_bal_e, tot_e, e_in, 1.0, evw + " energy");
      cmp_scaled(b_bal_p, tot_p.z, p_in, 1.0, evw + " pz");
      cmp_scaled(b_bal_p, tot_p.x, 0.0, 1.0, evw + " px");
      cmp_scaled(b_bal_p, tot_p.y, 0.0, 1.0, evw + " py");
    }
    total_refused += n_refused;
    total_events += kPortEvents;
    const double pk = (n_kill > 0) ? static_cast<double>(n_kill) : 1.0;
    const double ok_ = (o_kill > 0) ? static_cast<double>(o_kill) : 1.0;
    std::printf("%-12s %8lld %9lld %9lld %10.4f%% %10.4f%% %9.4f %9.4f %8lld  (%.2f scatters/"
                "event)\n",
                cname.c_str(), o_n, o_alive, n_alive, 100.0 * o_exhausted / ok_,
                100.0 * n_exh / pk, o_mean_mult, n_sec / pk, n_refused, sum_scatters / pk);

    // ---- the statistical comparisons
    const std::string cw = cname + " ";
    cmp_sigma(b_alive, proportion_z(n_alive, kPortEvents - n_refused, o_alive, o_n),
              cw + "alive port " + std::to_string(n_alive) + " g4 " + std::to_string(o_alive));
    cmp_sigma(b_exh, proportion_z(n_exh, n_kill, o_exhausted, o_kill),
              cw + "elastic returns port " + std::to_string(n_exh) + "/" +
                  std::to_string(n_kill) + " g4 " + std::to_string(o_exhausted) + "/" +
                  std::to_string(o_kill));
    // The multiplicity with BOTH variances - the dump writes the total's second moment, which
    // tests/test_bic_apply.cu's oracle does not have.
    {
      const double pm = n_sec / pk;
      const double pv = sum_mult2 / pk - pm * pm;
      const double ov = o_mean_mult2 - o_mean_mult * o_mean_mult;
      cmp_sigma(b_mult, yield_z(pm, pv, n_kill, o_mean_mult, ov, o_kill),
                cw + "port " + std::to_string(pm) + " g4 " + std::to_string(o_mean_mult));
    }
    std::map<int, Tally> keys = go;
    for (const auto& kv : mine) { keys[kv.first]; }
    for (const auto& kv : keys) {
      const int key = kv.first;
      const Tally& o = go[key];
      const Tally& p = mine[key];
      const double m1 = p.count / pk, m2 = o.count / ok_;
      const double v1 = p.sum_k2 / pk - m1 * m1;
      const double v2 = (o.count > 0) ? (o.sum_k2 / ok_ - m2 * m2) : m1;  // Poisson if absent
      cmp_sigma(b_yield, yield_z(m1, v1, n_kill, m2, v2, o_kill),
                cw + "pdg " + std::to_string(key) + " port " + std::to_string(p.count) + " g4 " +
                    std::to_string(o.count));
      if (p.count >= 25 && o.count >= 25) {
        const double pm = p.sum_e / p.count, om = o.sum_e / o.count;
        const double pv = p.sum_e2 / p.count - pm * pm;
        const double ov = o.sum_e2 / o.count - om * om;
        const double s2 = ((pv > 0.0) ? pv : 0.0) / p.count + ((ov > 0.0) ? ov : 0.0) / o.count;
        const double z = (s2 > 0.0) ? std::fabs(pm - om) / std::sqrt(s2)
                                    : ((std::fabs(pm - om) <= 1e-9 * (1.0 + std::fabs(om)))
                                           ? 0.0
                                           : 1.e9);
        cmp_sigma(b_ekin, z, cw + "pdg " + std::to_string(key) + " port " + std::to_string(pm) +
                                 " g4 " + std::to_string(om) + " MeV");
      }
    }
    std::map<int, long long> pkeys = go_parent;
    for (const auto& kv : my_parent) { pkeys[kv.first]; }
    for (const auto& kv : pkeys) {
      cmp_sigma(b_parent, proportion_z(my_parent[kv.first], n_kill, go_parent[kv.first], o_kill),
                cw + "parent " + std::to_string(kv.first) + " port " +
                    std::to_string(my_parent[kv.first]) + "/" + std::to_string(n_kill) + " g4 " +
                    std::to_string(go_parent[kv.first]) + "/" + std::to_string(o_kill));
    }
    ++icase;
  }

  // ===========================================================================================
  // 4. Where a nucleon on hydrogen stops returning an elastic scatter
  // ===========================================================================================
  //
  // The campaign's 400 MeV rows are 100% elastic returns and its 800 MeV rows are 0%, so the
  // energy at which `Propagate1H1` starts finding a resonance is somewhere in between - and it
  // is not the physical pion threshold, which `G4ParticleInelasticXS` puts near 260 MeV for
  // p + H. `bic_1h1_scan.csv` is Geant4's elastic-return fraction from 300 to 700 MeV, 2,000
  // events a point; the port runs ten times that at each point and the two binomial fractions
  // must agree. This is the measurement docs/RISK.md records the turn-on from.
  const int b_scan = new_bucket("ElasticReturnScan(sigma)", 5.0);
  {
    const auto rows = read_csv("bic_1h1_scan.csv");
    if (rows.empty()) {
      std::printf("bic_1h1_scan.csv is empty\n");
      ++fails;
    }
    std::printf("\n%6s %8s %14s %14s %8s\n", "pdg", "T (MeV)", "G4 elastic", "port elastic",
                "sigma");
    unsigned int iscan = 0;
    for (const auto& r : rows) {
      const int pdg = iv(r, 0);
      const double ekin = dv(r, 1);
      const long long o_n = lv(r, 2);
      const long long o_alive = lv(r, 3);
      const long long o_exh = lv(r, 4);
      const long long o_kill = o_n - o_alive;
      const physics::hadronic::HadProjectile<double> proj = make_projectile(pdg, ekin);
      physics::hadronic::HadNucleus tgt;
      tgt.z = 1;
      tgt.a = 1;
      tgt.l = 0;
      Philox<double> rng(0x5ca11b0du, iscan++, 18u);
      const long long n_port = 10 * o_n;
      long long n_kill = 0, n_exh = 0;
      for (long long ev = 0; ev < n_port; ++ev) {
        bic::BicStorage store = st.view();
        static bic::BicFinalState fs;
        bic::BicRefusal ref;
        bic::BicReport rep;
        preco::PrecoWorkspace pws = pb.view();
        bic::apply_yourself(proj, tgt, lt, fpool, pws, store, rng, fs, ref, rep);
        if (ref.any()) {
          ++total_refused;
          if (ref.h1.decay_null) { ++r_decay_null; }
          if (ref.h1.decay_refused) { ++r_decay_refused; }
          if (ref.h1.unknown_species) { ++r_unknown; }
          if (ref.h1.charge_imbalance) { ++r_charge; }
          if (ref.h1.capacity) { ++r_cap; }
          if (ref.nucleus) { ++r_nucleus; }
          if (ref.capacity) { ++r_capacity; }
          continue;
        }
        if (fs.status == physics::hadronic::HadFinalStateStatus::kIsAlive) { continue; }
        ++n_kill;
        if (rep.h1.exhausted) { ++n_exh; }
      }
      total_events += n_port;
      const double z = proportion_z(n_exh, n_kill, o_exh, o_kill);
      cmp_sigma(b_scan, z, std::to_string(pdg) + " at " + std::to_string(ekin) + " MeV port " +
                           std::to_string(n_exh) + "/" + std::to_string(n_kill) + " g4 " +
                           std::to_string(o_exh) + "/" + std::to_string(o_kill));
      std::printf("%6d %8.1f %13.2f%% %13.2f%% %8.2f\n", pdg, ekin,
                  (o_kill > 0) ? 100.0 * o_exh / o_kill : 0.0,
                  (n_kill > 0) ? 100.0 * n_exh / n_kill : 0.0, z);
    }
  }

  // ===========================================================================================
  std::printf("\n%-30s %10s %12s  %s\n", "bucket", "points", "worst", "where");
  for (const Bucket& b : buckets) {
    const bool bad = b.worst > b.tol;
    if (bad) { ++fails; }
    std::printf("%-30s %10lld %12.4g  %s%s\n", b.name, b.n, b.worst, bad ? "FAIL " : "",
                b.where.c_str());
  }
  std::printf("  tape: %lld events, %lld reproduced by the dump's public-piece probe, %lld of "
              "them returned the last elastic scatter after 200 tries\n",
              tape_events, tape_probe_ok, tape_exhausted);
  std::printf("  campaign: %lld port events, %lld refused (%.4g%%) - by name: Decay() returned "
              "null %lld, decay engine refused %lld, unknown species %lld, charge imbalance "
              "%lld, H1 capacity %lld, nucleus %lld, final-state capacity %lld\n",
              total_events, total_refused,
              (total_events > 0) ? 100.0 * total_refused / total_events : 0.0, r_decay_null,
              r_decay_refused, r_unknown, r_charge, r_cap, r_nucleus, r_capacity);
  std::printf("\ntest_bic_1h1: %s\n", (fails == 0) ? "PASS" : "FAIL");
  return (fails == 0) ? 0 : 1;
}
