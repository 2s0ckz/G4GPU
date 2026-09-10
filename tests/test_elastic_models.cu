// The four elastic final-state models, against Geant4 11.1.1 itself.
//
//   G4HadronElastic            (the Gheisha two-exponential sampler, and the ApplyYourself
//                               kinematics every other model inherits)
//   G4ChipsElasticModel        (proton and neutron, through G4Chips{Proton,Neutron}ElasticXS)
//   G4ElasticHadrNucleusHE     (pi+ and pi-, including the tables it builds at initialisation)
//   G4NuclNuclDiffuseElastic   (ions)
//
// Oracles: ref/oracle/elastic_sample_t.csv, elastic_chips_xs.csv, elastic_moments.csv,
// elastic_nuclear_mass.csv, elastic_he_nist_a.csv - see ref/dump/dump_elastic.cc.
//
// ---------------------------------------------------------------------------------------------
// A sampler is compared EXACTLY here, not statistically, and that is the point.
//
// `ref/dump/dump_elastic.cc` installs a CLHEP engine that returns a prescribed eight-value cycle
// instead of random numbers, so `SampleInvariantT` and `ApplyYourself` become deterministic
// functions of their inputs. This test feeds the same cycle to the port. So the comparison is
// the one a cross-section table gets - agreement to the last few digits - and not "the two
// histograms look alike", which cannot tell a right distribution from a right distribution with
// a coefficient 1e-6 out.
//
// The number of uniforms each call consumed is in the oracle too, and it is compared. A port
// that took a different branch - fell back to Gheisha when Geant4 did not, resampled when it did
// not, drew phi in a different place - reads a different uniform from the cycle and so gets a
// different answer; the draw count says which of the two went wrong.
//
// `elastic_moments.csv` is then the statistical check, with Geant4's own engine at a fixed seed:
// 20,000 samples per point, the first two moments of t/tmax and of cos(theta_cm), and a 20-bin
// histogram. Eight phases of a cycle visit eight points of the inverse CDF; 20,000 draws visit
// all of it, including channel weights that the eight happen to miss.
//
// ---------------------------------------------------------------------------------------------
// What comes from another package.
//
// `G4NucleiProperties::GetNuclearMass(A, Z)` is package P3's. Every elastic model needs it - for
// the target mass in the CMS boost, for (-t)max, and for the recoil's mass - so this package
// takes it as a functor and this test supplies it from `elastic_nuclear_mass.csv`, which is that
// function dumped for every (Z, A) used here. A port of the table would be P3's work; a formula
// for it here would make exact kinematics impossible.
// ---------------------------------------------------------------------------------------------
// This translation unit hands HOST-ONLY functors to __host__ __device__ templates on purpose.
//
// Every function under test is `__host__ __device__` so that the same code runs in a kernel, and
// nvcc instantiates the device side of each template even when nothing launches it. The functors
// this test supplies - the oracle-backed nuclear-mass table, the recoil species map, the CHIPS
// parameter lambdas - read std::vector and std::string and cannot be device code, and they do
// not need to be: what is being checked is the numbers, on the host, against a CSV. So nvcc's
// "calling a __host__ function from a __host__ __device__ function" family is expected here and
// is silenced by number rather than by making the whole build quieter or by adding
// --extended-lambda to the shared build_one_test.bat. A REAL caller of these templates from
// device code supplies device functors and gets the warning if it does not.
//
//   20011 - a __host__ function called from a __host__ __device__ one
//   20013 - the same for a constexpr __host__ function (every C++17 lambda here is one)
//   20015 - the same, reported without a name
#ifdef __CUDACC__
#pragma nv_diag_suppress 20011
#pragma nv_diag_suppress 20013
#pragma nv_diag_suppress 20015
#endif

#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "core/rng.cuh"
#include "data/atomic_masses.cuh"
#include "physics/hadronic/elastic/chips_elastic.cuh"
#include "physics/hadronic/elastic/elastic_hadr_nucleus_he.cuh"
#include "physics/hadronic/elastic/hadron_elastic.cuh"
#include "physics/hadronic/elastic/nucl_nucl_diffuse_elastic.cuh"
#include "physics/hadronic/process.cuh"

using namespace g4gpu;
using namespace g4gpu::physics::hadronic;
using namespace g4gpu::physics::hadronic::elastic;
using real_t = double;

namespace {

int fails = 0;

void fail(const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  std::printf("  FAIL: ");
  std::vprintf(fmt, ap);
  std::printf("\n");
  va_end(ap);
  ++fails;
}

// ------------------------------------------------------------------------------------------
// The prescribed uniform cycle, identical to ref/dump/dump_elastic.cc's CycleEngine.
// ------------------------------------------------------------------------------------------

/// A function-local static rather than a namespace-scope array: nvcc cannot take the address of
/// a namespace-scope constant from device code, and `hadron_elastic_apply_yourself` is
/// `__host__ __device__`, so the device instantiation of this Rng has to compile too.
__host__ __device__ inline const double* cycle_seq() {
  static const double v[8] = {0.05, 0.37, 0.63, 0.91, 0.12, 0.78, 0.29, 0.55};
  return v;
}

struct CycleRng {
  int phase = 0;
  int n = 0;
  __host__ __device__ void reset(int p) { phase = p; n = 0; }
  __host__ __device__ real_t uniform() {
    const real_t v = real_t(cycle_seq()[(n + phase) % 8]);
    ++n;
    return v;
  }
};

// ------------------------------------------------------------------------------------------
// Particle masses, from ref/oracle/hadron_tables.csv and CLHEP.
// ------------------------------------------------------------------------------------------

constexpr real_t kMassProton = real_t(938.272013);
constexpr real_t kMassNeutron = real_t(939.56536);
constexpr real_t kMassPion = real_t(139.5701);
constexpr real_t kMassAlpha = real_t(3727.379);

// ------------------------------------------------------------------------------------------
// CSV helpers
// ------------------------------------------------------------------------------------------

std::string oracle_dir() {
  const char* env = std::getenv("G4GPU_ORACLE");
  return (env != nullptr) ? std::string(env) : std::string("ref/oracle");
}

std::vector<std::string> split(const std::string& s, char sep) {
  std::vector<std::string> out;
  std::size_t a = 0;
  while (true) {
    const std::size_t b = s.find(sep, a);
    if (b == std::string::npos) { out.push_back(s.substr(a)); break; }
    out.push_back(s.substr(a, b - a));
    a = b + 1;
  }
  return out;
}

std::vector<std::vector<std::string>> load_rows(const std::string& path) {
  std::vector<std::vector<std::string>> out;
  FILE* f = std::fopen(path.c_str(), "r");
  if (f == nullptr) { return out; }
  char line[8192];
  if (std::fgets(line, sizeof line, f) == nullptr) { std::fclose(f); return out; }
  while (std::fgets(line, sizeof line, f) != nullptr) {
    std::string s(line);
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) { s.pop_back(); }
    if (s.empty()) { continue; }
    out.push_back(split(s, ','));
  }
  std::fclose(f);
  return out;
}

// ------------------------------------------------------------------------------------------
// P3's nuclear masses, as a functor.
// ------------------------------------------------------------------------------------------

struct NuclearMassTable {
  std::vector<int> z, a;
  std::vector<real_t> m;

  void add(int zz, int aa, real_t mm) {
    for (std::size_t i = 0; i < z.size(); ++i) {
      if (z[i] == zz && a[i] == aa) { return; }
    }
    z.push_back(zz);
    a.push_back(aa);
    m.push_back(mm);
  }
  /// Refused by name rather than approximated: an (Z,A) that is not in the dumped table has no
  /// mass here, and returning a formula's value would make every kinematic result downstream
  /// wrong by an unknown amount.
  __host__ real_t operator()(int zz, int aa) const {
    for (std::size_t i = 0; i < z.size(); ++i) {
      if (z[i] == zz && a[i] == aa) { return m[i]; }
    }
    std::printf("  FAIL: no nuclear mass for (Z=%d, A=%d) in elastic_nuclear_mass.csv\n", zz,
                aa);
    return real_t(0);
  }
};

NuclearMassTable g_masses;

/// G4HadronElastic::ApplyYourself's recoil species: the five light nuclei it names explicitly,
/// and G4IonTable::GetIon(Z,A,0) for everything else. The mass of every one of them is
/// G4NucleiProperties::GetNuclearMass(A,Z) - `elastic_nuclear_mass.csv` carries both columns and
/// they agree, which is why one table serves both.
struct RecoilSpecies {
  __host__ void operator()(int z, int a, int* pdg, real_t* mass) const {
    if (z == 1 && a == 1)       { *pdg = 2212; }
    else if (z == 1 && a == 2)  { *pdg = 1000010020; }
    else if (z == 1 && a == 3)  { *pdg = 1000010030; }
    else if (z == 2 && a == 3)  { *pdg = 1000020030; }
    else if (z == 2 && a == 4)  { *pdg = 1000020040; }
    else                        { *pdg = pdg_nuclear_code(z, a); }
    *mass = g_masses(z, a);
  }
};

// ------------------------------------------------------------------------------------------
// Deviation bookkeeping
// ------------------------------------------------------------------------------------------

struct Cell {
  int n = 0;
  double worst = 0;
  std::string where;
  void note(double dev, const std::string& w) {
    ++n;
    if (dev > worst) { worst = dev; where = w; }
  }
};

double reldev(double ours, double g4, double floor_abs) {
  const double d = std::fabs(ours - g4);
  const double s = std::fabs(g4);
  if (s > floor_abs) { return d / s; }
  return (d > floor_abs) ? d / floor_abs : 0.0;
}

// ------------------------------------------------------------------------------------------
// 1. The NIST A each G4ElasticHadrNucleusHE table is built at.
// ------------------------------------------------------------------------------------------

void check_he_nist_a(const std::string& dir) {
  std::printf("== the A G4ElasticHadrNucleusHE builds each element's table at ==\n");
  const auto rows = load_rows(dir + "/elastic_he_nist_a.csv");
  if (rows.empty()) { fail("cannot read elastic_he_nist_a.csv"); return; }
  int n = 0, worst_z = 0;
  double worst = 0;
  for (const auto& r : rows) {
    if (r.size() < 3) { continue; }
    const int z = std::atoi(r[0].c_str());
    const double amu = std::atof(r[1].c_str());
    const int la = std::atoi(r[2].c_str());
    const double ours_amu = data::nist_atomic_mass_table()[z];
    const int ours_la = int(std::lround(ours_amu));
    const double d = std::fabs(ours_amu / amu - 1.0);
    if (d > worst) { worst = d; worst_z = z; }
    if (ours_la != la) {
      fail("Z=%d: the rounded A is %d in the port and %d in Geant4 (amu %.17g vs %.17g)", z,
           ours_la, la, ours_amu, amu);
    }
    ++n;
  }
  std::printf("  Z = 1..%d: %d elements, worst relative deviation in the atomic mass %.3g "
              "(Z=%d)\n", n, n, worst, worst_z);
  if (n != 92) { fail("expected 92 elements, compared %d", n); }
  if (worst > 1e-15) { fail("the port's atomic masses differ from Geant4's by %.3g", worst); }
  std::printf("\n");
}

// ------------------------------------------------------------------------------------------
// 2. The CHIPS classes: cross section, (-t)max, and GetExchangeT.
// ------------------------------------------------------------------------------------------

void check_chips(const std::string& dir) {
  std::printf("== G4Chips{Proton,Neutron}ElasticXS: cross section, (-t)max and GetExchangeT ==\n");
  const auto rows = load_rows(dir + "/elastic_chips_xs.csv");
  if (rows.empty()) { fail("cannot read elastic_chips_xs.csv"); return; }

  Cell cs_p, cs_n, tm_p, tm_n, t_p, t_n;
  int draws_mismatch = 0;
  const real_t gev2 = units::GeV<real_t>() * units::GeV<real_t>();

  for (const auto& r : rows) {
    if (r.size() < 9) { continue; }
    const int pdg = std::atoi(r[0].c_str());
    const int z = std::atoi(r[1].c_str());
    const int n = std::atoi(r[2].c_str());
    const real_t p = std::atof(r[3].c_str());
    const double cs_g4 = std::atof(r[4].c_str());
    const double hmax_g4 = std::atof(r[5].c_str());
    const int phase = std::atoi(r[6].c_str());
    const int ndraws_g4 = std::atoi(r[7].c_str());
    const double t_g4 = std::atof(r[8].c_str());

    const bool is_proton = (pdg == 2212);
    const real_t m_gev = (is_proton ? kMassProton : kMassNeutron) * real_t(1e-3);
    // Two flags, not one: GetQ2max asks `tgZ==0 && tgN==1` in the neutron class where
    // GetExchangeT asks `tgZ==1 && tgN==0`. Geant4's inconsistency, reproduced - see the note in
    // chips_sample_invariant_t. One flag used for both put n+p through the four-channel nuclear
    // sampler and this comparison caught it at Z=1 N=0, p = 3 GeV/c.
    const bool hh_q2max = is_proton ? (z == 1 && n == 0) : (z == 0 && n == 1);
    const bool hh_exchange = (z == 1 && n == 0);
    const real_t tgt_mass_gev =
        (!is_proton && n == 0 && z <= 1) ? kMassProton * real_t(1e-3)
                                         : g_masses(z, z + n) * real_t(1e-3);
    const real_t q2max = chips_q2max_gev2<real_t>(p / units::GeV<real_t>(), m_gev,
                                                  tgt_mass_gev, hh_q2max);
    ChipsState<real_t> st;
    if (is_proton) {
      const ChipsProtonPars<real_t> pars = chips_proton_pars<real_t>(z, n);
      auto tab = [&](real_t lp, bool* ref) {
        return chips_proton_tab_values<real_t>(pars, lp, z, n, ref);
      };
      st = chips_calculate<real_t>(tab, p, q2max);
    } else {
      bool refused = false;
      const ChipsNeutronPars<real_t> pars = chips_neutron_pars<real_t>(z, n, &refused);
      auto tab = [&](real_t lp, bool* ref) {
        return chips_neutron_tab_values<real_t>(pars, lp, z, n, ref);
      };
      st = chips_calculate<real_t>(tab, p, q2max);
      if (refused) { fail("the neutron parameters were refused for Z=%d N=%d", z, n); }
    }

    char buf[192];
    std::snprintf(buf, sizeof buf, "pdg=%d Z=%d N=%d p=%.6g MeV/c", pdg, z, n, double(p));
    // The total cross section: the port's is in mb already (CHIPS works in mb internally).
    (is_proton ? cs_p : cs_n).note(reldev(st.v.cs, cs_g4, 1e-30), buf);
    // GetHMaxT is lastTM*GeV^2/2.
    (is_proton ? tm_p : tm_n)
        .note(reldev(double(q2max * gev2 * real_t(0.5)), hmax_g4, 1e-30), buf);

    CycleRng rng;
    rng.reset(phase);
    const real_t t_ours =
        chips_exchange_t<real_t>(st, z, n, hh_exchange, !is_proton, rng);
    if (rng.n != ndraws_g4) {
      ++draws_mismatch;
      if (draws_mismatch <= 5) {
        fail("GetExchangeT draw count: port %d, Geant4 %d (%s, phase %d)", rng.n, ndraws_g4,
             buf, phase);
      }
    }
    char buf2[224];
    std::snprintf(buf2, sizeof buf2, "%s phase=%d", buf, phase);
    // -t can legitimately be 0 (the sampler clamps), so the floor is absolute in MeV^2 and
    // scaled to the point's own (-t)max rather than being a bare epsilon.
    const double floor_t = 1e-14 * hmax_g4 + 1e-30;
    (is_proton ? t_p : t_n).note(reldev(double(t_ours), t_g4, floor_t), buf2);
  }

  // A few ulp. The stateless reformulation of the CHIPS "Associative Memory DB" reproduces the
  // interpolation exactly, so the cross section is within one ulp and (-t)max and GetExchangeT
  // are bitwise; 1e-15 is what that deserves rather than the 1e-11 this test started with, which
  // would have accepted a wrong coefficient in the sixth digit.
  struct Item { const char* what; Cell* c; double tol; };
  const Item items[] = {
      {"proton cross section", &cs_p, 1e-15},
      {"neutron cross section", &cs_n, 1e-15},
      {"proton (-t)max", &tm_p, 1e-15},
      {"neutron (-t)max", &tm_n, 1e-15},
      {"proton GetExchangeT", &t_p, 1e-15},
      {"neutron GetExchangeT", &t_n, 1e-15},
  };
  for (const Item& it : items) {
    std::printf("  %-22s %6d points  worst %10.3e  tol %8.1e  %s\n", it.what, it.c->n,
                it.c->worst, it.tol, it.c->where.c_str());
    if (it.c->n == 0) { fail("no %s points compared", it.what); }
    if (it.c->worst > it.tol) {
      fail("%s: worst relative deviation %.3e exceeds %.1e", it.what, it.c->worst, it.tol);
    }
  }
  if (draws_mismatch > 0) {
    fail("%d GetExchangeT calls consumed a different number of uniforms", draws_mismatch);
  }
  std::printf("\n");
}

// ------------------------------------------------------------------------------------------
// 3. The G4ElasticHadrNucleusHE tables, built here.
// ------------------------------------------------------------------------------------------

struct HeTables {
  HeEnergyGrid<real_t> grid;
  HeBoundary<real_t> bnd;
  std::vector<int> pdg, z;
  std::vector<HeElasticData<real_t>> data;

  const HeElasticData<real_t>* get(int p, int zz) {
    for (std::size_t i = 0; i < pdg.size(); ++i) {
      if (pdg[i] == p && z[i] == zz) { return &data[i]; }
    }
    // Build it, exactly as FillData does on demand: A is the ROUNDED NIST atomic mass for Z,
    // not the isotope's A.
    int idx = -1, ih = -1, ih1 = -1;
    const int* codes = he_hadron_code();
    for (int i = 0; i < HeDims::kNHadrons; ++i) {
      if (codes[i] == p) { idx = i; ih = he_hadron_type()[i]; ih1 = he_hadron_type1()[i]; break; }
    }
    if (idx < 0) { return nullptr; }
    const int a = int(std::lround(data::nist_atomic_mass_table()[zz]));
    const real_t mass_a = g_masses(zz, a);
    const real_t hm = (p == 211 || p == -211) ? kMassPion : real_t(0);
    data.push_back(he_fill_data<real_t>(zz, a, mass_a, hm, ih, ih1, p,
                                        /*is_proton_projectile=*/false, grid, bnd));
    pdg.push_back(p);
    z.push_back(zz);
    return &data.back();
  }
};

HeTables g_he;

// ------------------------------------------------------------------------------------------
// 4. Every model's ApplyYourself, deterministically.
// ------------------------------------------------------------------------------------------

struct SampleRow {
  std::string model;
  int pdg = 0, z = 0, a = 0, phase = 0, ndraws = 0, nsec = 0, recoil_pdg = 0;
  int t_draws = 0;
  double ekin = 0, tmax = 0, t = 0, cost = 0, efinal = 0, edep = 0, erec = 0;
  double dirx = 0, diry = 0, dirz = 0, recx = 0, recy = 0, recz = 0;
  /// -t reconstructed as `2*M*T_recoil` instead of read from the sampler. Kept in the oracle and
  /// checked here only for how far it is from the real thing: see the comment on `check_sample_t`.
  double t_from_erec = 0;
};

real_t mass_of(int pdg) {
  switch (pdg) {
    case 2212: return kMassProton;
    case 2112: return kMassNeutron;
    case 211: case -211: return kMassPion;
    case 1000020040: return kMassAlpha;
    case 1000060120: return g_masses(6, 12);
    default: return real_t(0);
  }
}

int baryon_of(int pdg) {
  switch (pdg) {
    case 2212: case 2112: return 1;
    case 211: case -211: return 0;
    case 1000020040: return 4;
    case 1000060120: return 12;
    default: return 0;
  }
}

real_t charge_of(int pdg) {
  switch (pdg) {
    case 2212: return real_t(1);
    case 2112: return real_t(0);
    case 211: return real_t(1);
    case -211: return real_t(-1);
    case 1000020040: return real_t(2);
    case 1000060120: return real_t(6);
    default: return real_t(0);
  }
}

void check_sample_t(const std::string& dir) {
  std::printf("== ApplyYourself under the prescribed uniform cycle ==\n");
  const auto raw = load_rows(dir + "/elastic_sample_t.csv");
  if (raw.empty()) { fail("cannot read elastic_sample_t.csv"); return; }

  const char* mnames[4] = {"Gheisha", "Chips", "HE", "NNDiffuse"};
  Cell ct[4], ccos[4], cef[4], cerec[4], cdir[4], crecon[4];
  int nrows[4] = {0, 0, 0, 0};
  int draws_bad[4] = {0, 0, 0, 0};
  int nsec_bad = 0, pdg_bad = 0, resampled = 0;
  const NuclNuclDiffuseParams<real_t> nnpar;
  const RecoilSpecies recoil;

  for (const auto& r : raw) {
    if (r.size() < 23) { continue; }
    SampleRow s;
    s.model = r[0];
    s.pdg = std::atoi(r[1].c_str());
    s.z = std::atoi(r[2].c_str());
    s.a = std::atoi(r[3].c_str());
    s.ekin = std::atof(r[4].c_str());
    s.phase = std::atoi(r[5].c_str());
    s.ndraws = std::atoi(r[6].c_str());
    s.tmax = std::atof(r[7].c_str());
    s.t = std::atof(r[8].c_str());
    s.cost = std::atof(r[9].c_str());
    s.efinal = std::atof(r[10].c_str());
    s.edep = std::atof(r[11].c_str());
    s.erec = std::atof(r[12].c_str());
    s.nsec = std::atoi(r[13].c_str());
    s.recoil_pdg = std::atoi(r[14].c_str());
    s.dirx = std::atof(r[15].c_str());
    s.diry = std::atof(r[16].c_str());
    s.dirz = std::atof(r[17].c_str());
    s.recx = std::atof(r[18].c_str());
    s.recy = std::atof(r[19].c_str());
    s.recz = std::atof(r[20].c_str());
    s.t_draws = std::atoi(r[21].c_str());
    s.t_from_erec = std::atof(r[22].c_str());

    int mi = -1;
    for (int i = 0; i < 4; ++i) { if (s.model == mnames[i]) { mi = i; break; } }
    if (mi < 0) { fail("unknown model %s in elastic_sample_t.csv", s.model.c_str()); continue; }

    HadProjectile<real_t> pro;
    pro.pdg = s.pdg;
    pro.mass = mass_of(s.pdg);
    pro.baryon_number = baryon_of(s.pdg);
    pro.charge = charge_of(s.pdg);
    pro.kin_energy = s.ekin;
    if (pro.mass <= real_t(0)) { fail("no mass for PDG %d", s.pdg); continue; }
    const HadNucleus tgt{s.z, s.a, 0};

    CycleRng rng;
    rng.reset(s.phase);
    HadFinalState<real_t, 8> fs;
    ElasticSample<real_t> info;

    // The recoil threshold is 0 in the dump (`SetRecoilEnergyThreshold(0.0)`), so the recoil is
    // always emitted and `erec` is observable rather than folded into the deposit.
    const real_t thr = real_t(0);

    if (mi == 0) {
      auto sampler = [&](int pdg, real_t plab, int z, int a, real_t tmax, CycleRng& g) {
        return hadron_elastic_sample_invariant_t<real_t>(pdg, plab, z, a, tmax, g);
      };
      info = hadron_elastic_apply_yourself<real_t, 8>(pro, tgt, sampler, g_masses, recoil, thr,
                                                      -1, rng, &fs);
    } else if (mi == 1) {
      auto sampler = [&](int pdg, real_t plab, int z, int a, real_t tmax, CycleRng& g) {
        bool fb = false, unsup = false;
        const real_t t = chips_sample_invariant_t<real_t>(pdg, plab, z, a, tmax, g_masses, g,
                                                          &fb, &unsup);
        if (unsup) { fail("Chips refused PDG %d", pdg); }
        return t;
      };
      info = hadron_elastic_apply_yourself<real_t, 8>(pro, tgt, sampler, g_masses, recoil, thr,
                                                      -1, rng, &fs);
    } else if (mi == 2) {
      const HeElasticData<real_t>* tab = (s.z == 1) ? nullptr : g_he.get(s.pdg, s.z);
      if (s.z != 1 && tab == nullptr) { fail("no HE table for pdg %d Z=%d", s.pdg, s.z); continue; }
      auto sampler = [&](int pdg, real_t plab, int z, int a, real_t tmax, CycleRng& g) {
        bool fb = false, unk = false;
        int hidx = -1;
        real_t t = he_sample_invariant_t<real_t>(pdg, plab, z, a, pro.mass, tmax, tab,
                                                 g_he.grid, g_he.bnd, g, &fb, &unk, &hidx);
        if (unk) { fail("the HE model does not know PDG %d", pdg); }
        // G4ElasticHadrNucleusHE::SampleInvariantT does the hand-over itself, so the Gheisha
        // draws come out of the same stream at the same point.
        if (fb) { t = hadron_elastic_sample_invariant_t<real_t>(pdg, plab, z, a, tmax, g); }
        return t;
      };
      info = hadron_elastic_apply_yourself<real_t, 8>(pro, tgt, sampler, g_masses, recoil, thr,
                                                      -1, rng, &fs);
    } else {
      auto sampler = [&](int, real_t, int z, int a, real_t, CycleRng& g) {
        NuclNuclDiffuseState<real_t> st;
        bool refused = false;
        const real_t t = nucl_nucl_diffuse_sample_invariant_t<real_t>(pro, z, a, g_masses,
                                                                      nnpar, g, &st, &refused);
        if (refused) { fail("NNDiffuse refused a neutral projectile"); }
        return t;
      };
      info = hadron_elastic_apply_yourself<real_t, 8>(pro, tgt, sampler, g_masses, recoil, thr,
                                                      -1, rng, &fs);
    }

    ++nrows[mi];
    if (rng.n != s.ndraws) {
      ++draws_bad[mi];
      if (draws_bad[mi] <= 3) {
        fail("%s pdg=%d Z=%d A=%d E=%.6g phase=%d: %d uniforms consumed, Geant4 used %d",
             s.model.c_str(), s.pdg, s.z, s.a, s.ekin, s.phase, rng.n, s.ndraws);
      }
    }

    // The ApplyYourself resample path (`t < 0 || t > tmax`, which falls back to
    // G4HadronElastic::SampleInvariantT and costs a second set of uniforms) is not reached
    // anywhere on this grid, and the oracle proves it rather than the test assuming it: with no
    // resample, ApplyYourself consumes exactly the sampler's draws plus one for phi. If a future
    // grid point does resample, this fires and the `t` column - which the dump samples on its
    // own, before any fallback - stops being the t ApplyYourself used.
    if (s.ndraws != s.t_draws + 1) { ++resampled; }

    char buf[224];
    std::snprintf(buf, sizeof buf, "%s pdg=%d Z=%d A=%d E=%.6g MeV phase=%d", s.model.c_str(),
                  s.pdg, s.z, s.a, s.ekin, s.phase);
    // -t is compared relative to the point's own (-t)max, because a t of exactly zero is a legal
    // sample and a bare relative deviation would divide by it.
    const double floor_t = 1e-13 * s.tmax + 1e-30;
    ct[mi].note(reldev(double(info.t), s.t, floor_t), buf);
    ccos[mi].note(reldev(double(info.cos_theta_cms), s.cost, 1e-13), buf);
    cef[mi].note(reldev(double(fs.energy_change > 0 ? fs.energy_change : 0), s.efinal,
                        1e-13 * s.ekin + 1e-30), buf);
    // How far Geant4's own `2*M*T_recoil` is from Geant4's own sampled -t. Not a check on the
    // port at all: it is the size of the cancellation in `(lv - nlv1).e() - M`, reported so that
    // the reason this test compares against the sampler and not against the recoil is a number
    // in the output and not a claim in a comment. See ref/dump/dump_elastic.cc's run_apply.
    crecon[mi].note(reldev(s.t_from_erec, s.t, floor_t), buf);
    const real_t erec_ours = (fs.n_secondaries > 0) ? fs.secondaries[0].kin_energy
                                                    : fs.local_energy_deposit;
    cerec[mi].note(reldev(double(erec_ours), s.erec, 1e-13 * s.ekin + 1e-30), buf);
    // The direction: compare the full unit vector, so phi is checked and not only the polar
    // angle. An absolute tolerance, because a component can legitimately be zero.
    const double dd = std::fabs(fs.momentum_change.x - s.dirx) +
                      std::fabs(fs.momentum_change.y - s.diry) +
                      std::fabs(fs.momentum_change.z - s.dirz);
    double dr = 0;
    if (fs.n_secondaries > 0) {
      dr = std::fabs(fs.secondaries[0].direction.x - s.recx) +
           std::fabs(fs.secondaries[0].direction.y - s.recy) +
           std::fabs(fs.secondaries[0].direction.z - s.recz);
    }
    cdir[mi].note(dd + dr, buf);
    if (fs.n_secondaries != s.nsec) { ++nsec_bad; }
    if (fs.n_secondaries > 0 && fs.secondaries[0].pdg != s.recoil_pdg) {
      ++pdg_bad;
      if (pdg_bad <= 3) {
        fail("%s: recoil PDG %d, Geant4 %d", buf, fs.secondaries[0].pdg, s.recoil_pdg);
      }
    }
  }

  // A few ulp, for every model and every column.
  //
  // These were 1e-13 to 1e-9 while the oracle's `t` column was reconstructed from the recoil
  // energy, and even that was not loose enough. With the sampler's own -t in the file, three of
  // the four models agree BITWISE on all five columns and G4ElasticHadrNucleusHE is one ulp out
  // on -t alone, so 1e-15 is thirteen orders tighter than what this test used to accept and is
  // still not tight enough to break on a compiler that reassociates one product. Anything that
  // is not the same arithmetic on the same inputs shows up far above it: the CLHEP `pp*(1./ee)`
  // boost vector and `p*=(1./|p|)` unit vector, both ulp-level on their own, put the recoil
  // direction 5.8e-13 out before they were reproduced.
  const double tol_t[4] = {1e-15, 1e-15, 1e-15, 1e-15};
  const double tol_dir[4] = {1e-15, 1e-15, 1e-15, 1e-15};
  for (int i = 0; i < 4; ++i) {
    std::printf("  %-10s %5d rows  t %9.2e  cos %9.2e  Efin %9.2e  Erec %9.2e  dir %9.2e "
                "(tol %.0e)\n", mnames[i], nrows[i], ct[i].worst, ccos[i].worst, cef[i].worst,
                cerec[i].worst, cdir[i].worst, tol_t[i]);
    if (nrows[i] == 0) { fail("no %s rows compared", mnames[i]); continue; }
    std::printf("      worst t at: %s\n", ct[i].where.c_str());
    std::printf("      Geant4's own 2*M*Trec vs Geant4's own -t: worst %9.2e (the cancellation, "
                "not the port)\n", crecon[i].worst);
    if (ct[i].worst > tol_t[i]) {
      fail("%s: -t differs by %.3e (tol %.0e)", mnames[i], ct[i].worst, tol_t[i]);
    }
    if (ccos[i].worst > tol_t[i]) {
      fail("%s: cos(theta_cm) differs by %.3e", mnames[i], ccos[i].worst);
    }
    if (cef[i].worst > tol_t[i]) {
      fail("%s: the primary's final energy differs by %.3e", mnames[i], cef[i].worst);
    }
    if (cerec[i].worst > tol_t[i]) {
      fail("%s: the recoil energy differs by %.3e", mnames[i], cerec[i].worst);
    }
    if (cdir[i].worst > tol_dir[i]) {
      fail("%s: the directions differ by %.3e (tol %.0e)", mnames[i], cdir[i].worst,
           tol_dir[i]);
    }
    if (draws_bad[i] > 0) {
      fail("%s: %d rows consumed a different number of uniforms", mnames[i], draws_bad[i]);
    }
  }
  if (nsec_bad > 0) { fail("%d rows produced a different number of secondaries", nsec_bad); }
  if (pdg_bad > 0) { fail("%d rows produced a different recoil species", pdg_bad); }
  if (resampled > 0) {
    fail("%d rows resampled t inside ApplyYourself; the oracle's t column is the FIRST sample "
         "and no longer the one ApplyYourself used", resampled);
  }
  std::printf("\n");
}

// ------------------------------------------------------------------------------------------
// 5. The statistical comparison.
// ------------------------------------------------------------------------------------------

void check_moments(const std::string& dir) {
  std::printf("== 20,000 samples per point: moments and histogram of t/tmax ==\n");
  const auto raw = load_rows(dir + "/elastic_moments.csv");
  if (raw.empty()) { fail("cannot read elastic_moments.csv"); return; }

  const char* mnames[4] = {"Gheisha", "Chips", "HE", "NNDiffuse"};
  double worst_sigma[4] = {0, 0, 0, 0};
  std::string worst_where[4];
  double worst_hist[4] = {0, 0, 0, 0};
  int nrows[4] = {0, 0, 0, 0};
  const NuclNuclDiffuseParams<real_t> nnpar;
  const RecoilSpecies recoil;

  for (const auto& r : raw) {
    if (r.size() < 31) { continue; }
    const std::string model = r[0];
    const int pdg = std::atoi(r[1].c_str());
    const int z = std::atoi(r[2].c_str());
    const int a = std::atoi(r[3].c_str());
    const double ekin = std::atof(r[4].c_str());
    const int nsample = std::atoi(r[5].c_str());
    const double mx_g4 = std::atof(r[6].c_str());
    const double vx_g4 = std::atof(r[7].c_str());
    const double mc_g4 = std::atof(r[8].c_str());
    const double vc_g4 = std::atof(r[9].c_str());
    int h_g4[20];
    for (int b = 0; b < 20; ++b) { h_g4[b] = std::atoi(r[11 + b].c_str()); }

    int mi = -1;
    for (int i = 0; i < 4; ++i) { if (model == mnames[i]) { mi = i; break; } }
    if (mi < 0) { continue; }

    HadProjectile<real_t> pro;
    pro.pdg = pdg;
    pro.mass = mass_of(pdg);
    pro.baryon_number = baryon_of(pdg);
    pro.charge = charge_of(pdg);
    pro.kin_energy = ekin;
    if (pro.mass <= real_t(0)) { continue; }
    const HadNucleus tgt{z, a, 0};
    const HeElasticData<real_t>* tab = (mi == 2 && z != 1) ? g_he.get(pdg, z) : nullptr;

    Philox<real_t> rng(static_cast<uint32_t>(1000 + mi), static_cast<uint32_t>(z * 100 + a),
                       static_cast<uint32_t>(ekin));
    HadFinalState<real_t, 8> fs;
    double sx = 0, sxx = 0, sc = 0, scc = 0;
    int hist[20] = {0};
    for (int i = 0; i < nsample; ++i) {
      ElasticSample<real_t> info;
      if (mi == 0) {
        auto sm = [&](int p, real_t pl, int zz, int aa, real_t tm, Philox<real_t>& g) {
          return hadron_elastic_sample_invariant_t<real_t>(p, pl, zz, aa, tm, g);
        };
        info = hadron_elastic_apply_yourself<real_t, 8>(pro, tgt, sm, g_masses, recoil,
                                                        real_t(0), -1, rng, &fs);
      } else if (mi == 1) {
        auto sm = [&](int p, real_t pl, int zz, int aa, real_t tm, Philox<real_t>& g) {
          bool fb = false, un = false;
          return chips_sample_invariant_t<real_t>(p, pl, zz, aa, tm, g_masses, g, &fb, &un);
        };
        info = hadron_elastic_apply_yourself<real_t, 8>(pro, tgt, sm, g_masses, recoil,
                                                        real_t(0), -1, rng, &fs);
      } else if (mi == 2) {
        auto sm = [&](int p, real_t pl, int zz, int aa, real_t tm, Philox<real_t>& g) {
          bool fb = false, un = false;
          int hi = -1;
          real_t t = he_sample_invariant_t<real_t>(p, pl, zz, aa, pro.mass, tm, tab,
                                                   g_he.grid, g_he.bnd, g, &fb, &un, &hi);
          if (fb) { t = hadron_elastic_sample_invariant_t<real_t>(p, pl, zz, aa, tm, g); }
          return t;
        };
        info = hadron_elastic_apply_yourself<real_t, 8>(pro, tgt, sm, g_masses, recoil,
                                                        real_t(0), -1, rng, &fs);
      } else {
        auto sm = [&](int, real_t, int zz, int aa, real_t, Philox<real_t>& g) {
          NuclNuclDiffuseState<real_t> st;
          bool rf = false;
          return nucl_nucl_diffuse_sample_invariant_t<real_t>(pro, zz, aa, g_masses, nnpar, g,
                                                              &st, &rf);
        };
        info = hadron_elastic_apply_yourself<real_t, 8>(pro, tgt, sm, g_masses, recoil,
                                                        real_t(0), -1, rng, &fs);
      }
      const double x = (info.p_local_tmax > 0) ? double(info.t / info.p_local_tmax) : 0.0;
      sx += x; sxx += x * x;
      sc += double(info.cos_theta_cms);
      scc += double(info.cos_theta_cms) * double(info.cos_theta_cms);
      int b = int(x * 20.0);
      if (b < 0) { b = 0; }
      if (b > 19) { b = 19; }
      ++hist[b];
    }
    const double mx = sx / nsample, mc = sc / nsample;
    const double vx = sxx / nsample - mx * mx, vc = scc / nsample - mc * mc;
    ++nrows[mi];

    char buf[224];
    std::snprintf(buf, sizeof buf, "%s pdg=%d Z=%d A=%d E=%.6g MeV", model.c_str(), pdg, z, a,
                  ekin);
    // The mean of N samples has standard error sqrt(var/N); comparing two independent estimates
    // gives sqrt(2) more. Report the deviation in units of that, so the number is a sigma count
    // and not a tolerance that has to be re-argued per point.
    auto sig = [&](double a1, double a2, double v1, double v2) {
      const double se = std::sqrt((std::fabs(v1) + std::fabs(v2)) / double(nsample));
      return (se > 0) ? std::fabs(a1 - a2) / se : ((std::fabs(a1 - a2) > 1e-12) ? 1e9 : 0.0);
    };
    const double s1 = sig(mx, mx_g4, vx, vx_g4);
    const double s2 = sig(mc, mc_g4, vc, vc_g4);
    const double s = (s1 > s2) ? s1 : s2;
    if (s > worst_sigma[mi]) { worst_sigma[mi] = s; worst_where[mi] = buf; }
    // The histogram, bin fraction by bin fraction, also in sigma.
    for (int b = 0; b < 20; ++b) {
      const double f1 = double(hist[b]) / double(nsample);
      const double f2 = double(h_g4[b]) / double(nsample);
      const double p = 0.5 * (f1 + f2);
      const double se = std::sqrt(2.0 * p * (1.0 - p) / double(nsample)) + 1.0 / double(nsample);
      const double d = std::fabs(f1 - f2) / se;
      if (d > worst_hist[mi]) { worst_hist[mi] = d; }
    }
  }
  for (int i = 0; i < 4; ++i) {
    std::printf("  %-10s %4d points  worst moment %6.2f sigma  worst bin %6.2f sigma\n",
                mnames[i], nrows[i], worst_sigma[i], worst_hist[i]);
    if (nrows[i] > 0) { std::printf("      at: %s\n", worst_where[i].c_str()); }
    if (nrows[i] == 0) { fail("no %s statistical points compared", mnames[i]); }
    if (worst_sigma[i] > 6.0) {
      fail("%s: a moment is %.2f sigma from Geant4's", mnames[i], worst_sigma[i]);
    }
    if (worst_hist[i] > 6.0) {
      fail("%s: a histogram bin is %.2f sigma from Geant4's", mnames[i], worst_hist[i]);
    }
  }
  std::printf("\n");
}

}  // namespace

int main() {
  const std::string dir = oracle_dir();
  {
    const auto rows = load_rows(dir + "/elastic_nuclear_mass.csv");
    if (rows.empty()) {
      std::printf("cannot read %s/elastic_nuclear_mass.csv - run ref/oracle/run.bat tables "
                  "first\n", dir.c_str());
      return 1;
    }
    for (const auto& r : rows) {
      if (r.size() < 3) { continue; }
      g_masses.add(std::atoi(r[0].c_str()), std::atoi(r[1].c_str()),
                   real_t(std::atof(r[2].c_str())));
    }
    std::printf("== P3's G4NucleiProperties::GetNuclearMass, %d (Z,A) pairs loaded ==\n\n",
                int(g_masses.z.size()));
  }
  check_he_nist_a(dir);
  check_chips(dir);
  check_sample_t(dir);
  check_moments(dir);

  if (fails == 0) {
    std::printf("test_elastic_models: PASS\n");
    return 0;
  }
  std::printf("test_elastic_models: %d FAILURES\n", fails);
  return 1;
}
