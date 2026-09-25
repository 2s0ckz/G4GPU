// G4RandGauss is CLHEP::RandGaussQ at every call site - the port's one transcription of it, and
// the draw count at every site that calls it.
//
// `Randomize.hh` line 47 is `#define G4RandGauss CLHEP::RandGaussQ`: one uniform per value,
// through `transformQuick`'s 1,250-entry inverse-CDF table, with no state. docs/RISK.md V180
// found the Binary cascade implementing `CLHEP::RandGauss` instead, and V185 the six other sites
// QBBC reaches, which had Box-Muller (two uniforms per value) or the polar method with a cached
// spare. They all call src/core/rand_gauss_q.cuh now, and this test holds them to it:
//
//   1. THE FUNCTION, BITWISE. ref/oracle/gaussq_transform.csv is `transformQuick(r)` and
//      `transformSmall(r)` from CLHEP itself over a stated grid - the series tail with each
//      engine lattice's smallest uniform, every node of both tables with the double on either
//      side and the midpoint, the median, the mirror and the two NaN arguments - and
//      ref/oracle/gaussq_shoot.csv is `shoot(engine, mean, sd)` and `shoot(mean, sd)` on a
//      recorded HepJamesRandom stream, 32,000 values with the uniform each one consumed. Both
//      are compared to the last bit, and so is the float conversion of the `real_t` overload.
//   2. THE DRAW COUNT AT EVERY SITE, through the port's own functions with a counting tape
//      engine: G4UniversalFluctuation's thick-absorber Gaussian and its SampleGauss, the
//      G4IonFluctuations Gaussian, Urban's Randomizetlimit on both step-limit branches,
//      WentzelVI's lateral displacement (two, x then y), and G4CompetitiveFission's charge and
//      kinetic energy. One uniform per Gaussian everywhere, two at the displacement; every
//      value is checked against the uniform it was drawn from, so a count that is right by
//      accident is not enough either.
//
// Host only. The sites are `__host__ __device__` and the transport compiles the same source for
// the device; which uniform goes where is the same on both, and the host is where it can be
// counted.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "core/particle.cuh"
#include "core/rand_gauss_q.cuh"
#include "core/rng.cuh"
#include "core/units.cuh"
#include "core/vec3.cuh"
#include "data/materials.cuh"
#include "physics/em/fluctuation.cuh"
#include "physics/em/hadron_ionisation.cuh"
#include "physics/em/ion_fluctuation.cuh"
#include "physics/em/urban_msc.cuh"
#include "physics/em/wentzel_msc.cuh"
#include "physics/hadronic/deexcitation/fission.cuh"

using namespace g4gpu;

namespace {

int fails = 0;

void fail(const std::string& what) {
  std::printf("  FAIL: %s\n", what.c_str());
  ++fails;
}

/// Bit-identical, except that any NaN equals any NaN: CLHEP's NaN is written "nan" and its
/// payload is not part of the answer.
bool bitwise(double a, double b) {
  if (std::isnan(a) || std::isnan(b)) { return std::isnan(a) && std::isnan(b); }
  std::uint64_t ua = 0, ub = 0;
  std::memcpy(&ua, &a, sizeof ua);
  std::memcpy(&ub, &b, sizeof ub);
  return ua == ub;
}

std::string fmt(double v) {
  char b[64];
  std::snprintf(b, sizeof b, "%.17g", v);
  return b;
}

double parse(const std::string& s) {
  if (s == "nan") { return std::nan(""); }
  return std::strtod(s.c_str(), nullptr);
}

std::vector<std::vector<std::string>> read_csv(const std::string& path) {
  std::vector<std::vector<std::string>> rows;
  std::ifstream in(path);
  if (!in) { return rows; }
  std::string line;
  std::getline(in, line);  // header
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') { line.pop_back(); }
    if (line.empty()) { continue; }
    std::vector<std::string> row;
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ',')) { row.push_back(cell); }
    if (!line.empty() && line.back() == ',') { row.push_back(""); }
    rows.push_back(row);
  }
  return rows;
}

/// A recorded stream: serves the tape in order and COUNTS. Past the end it serves 0.5 and
/// counts that too, so a sampler that wants more uniforms than the site should draw is caught
/// by the count rather than by reading off the end of an array.
struct TapeRng {
  const double* u = nullptr;
  int n = 0;
  int i = 0;
  __host__ __device__ double uniform() {
    const double v = (i < n) ? u[i] : 0.5;
    ++i;
    return v;
  }
};

/// The same in float, for the `real_t = float` overload.
struct TapeRngF {
  const float* u = nullptr;
  int n = 0;
  int i = 0;
  __host__ __device__ float uniform() {
    const float v = (i < n) ? u[i] : 0.5f;
    ++i;
    return v;
  }
};

double tq(double r) { return rand_gauss_q_transform(r); }

// ---------------------------------------------------------------------------------------------
// 1. the function, bitwise
// ---------------------------------------------------------------------------------------------

void check_transform(const std::string& dir) {
  std::printf("== 1a. transformQuick and transformSmall against CLHEP (gaussq_transform.csv) ==\n");
  const auto rows = read_csv(dir + "/gaussq_transform.csv");
  if (rows.empty()) {
    fail("cannot read " + dir + "/gaussq_transform.csv - run ref/oracle/run.bat tables");
    return;
  }
  int n = 0, bad_q = 0, bad_s = 0;
  int n_series = 0, n_fine = 0, n_coarse = 0, n_mirror = 0, n_zero = 0, n_nan = 0;
  int n_t0_node = 0, n_t1_node = 0, n_lattice = 0;
  std::string first_bad;
  for (const auto& row : rows) {
    if (row.size() < 4) {
      fail("short row in gaussq_transform.csv");
      continue;
    }
    ++n;
    const std::string& label = row[0];
    const double r = parse(row[1]);
    const double want_q = parse(row[2]);
    const double want_s = parse(row[3]);
    const double got_q = rand_gauss_q_transform(r);
    const double got_s = rand_gauss_q_small(r);
    if (!bitwise(got_q, want_q)) {
      if (bad_q++ == 0) {
        first_bad = label + " r=" + row[1] + ": quick " + fmt(got_q) + " want " + row[2];
      }
    }
    if (!bitwise(got_s, want_s)) {
      if (bad_s++ == 0 && first_bad.empty()) {
        first_bad = label + " r=" + row[1] + ": small " + fmt(got_s) + " want " + row[3];
      }
    }
    // the census, by the branch CLHEP takes
    const double rr = (r > 0.5) ? 1.0 - r : r;
    if (r > 0.5) { ++n_mirror; }
    if (std::isnan(want_q)) { ++n_nan; }
    else if (want_q == 0.0 && r == 0.5) { ++n_zero; }
    else if (rr <= kGaussQTable0Step) { ++n_series; }
    else if (rr < kGaussQTable1Step) { ++n_fine; }
    else { ++n_coarse; }
    if (label == "t0_node") { ++n_t0_node; }
    if (label == "t1_node") { ++n_t1_node; }
    if (label.rfind("lattice_", 0) == 0) { ++n_lattice; }
  }
  std::printf("  %d rows: %d series, %d fine table, %d coarse table, %d mirrored, %d median, "
              "%d NaN\n", n, n_series, n_fine, n_coarse, n_mirror, n_zero, n_nan);
  std::printf("  transformQuick %d of %d differ, transformSmall %d of %d differ\n", bad_q, n,
              bad_s, n);
  if (bad_q > 0 || bad_s > 0) { fail("not bitwise: " + first_bad); }
  // A thinned oracle must fail rather than pass on the branches it kept.
  if (n_series < 100) { fail("fewer than 100 series-tail rows"); }
  if (n_fine < 500) { fail("fewer than 500 fine-table rows"); }
  if (n_coarse < 2000) { fail("fewer than 2000 coarse-table rows"); }
  if (n_mirror < 100) { fail("fewer than 100 mirrored rows"); }
  if (n_zero < 1) { fail("the r = 0.5 exact-zero row is missing"); }
  if (n_nan != 2) { fail("the r = 0 and r = 1 rows are missing"); }
  if (n_t0_node != 250) { fail("not every node of table 0 is in the file"); }
  if (n_t1_node != 1000) { fail("not every node of table 1 is in the file"); }
  if (n_lattice != 5) { fail("the engine-lattice rows are missing"); }
}

void check_shoot(const std::string& dir) {
  std::printf("== 1b. shoot(engine, mean, sd) and shoot(mean, sd) on a recorded "
              "HepJamesRandom stream (gaussq_shoot.csv) ==\n");
  const auto rows = read_csv(dir + "/gaussq_shoot.csv");
  if (rows.empty()) {
    fail("cannot read " + dir + "/gaussq_shoot.csv - run ref/oracle/run.bat tables");
    return;
  }
  int n = 0, bad = 0, bad_draws = 0, bad_port_draws = 0, n_unit = 0, bad_unit = 0;
  int n_engine = 0, n_static = 0, bad_float = 0, n_float = 0, flag_rows = 0;
  std::string first_bad;
  for (const auto& row : rows) {
    if (row.size() >= 6 && row[0] == "randgauss_state_changed_by_cases") {
      ++flag_rows;
      // RandGauss's pair cache - the state RandGaussQ does not have - is exactly where each
      // case of 4,000 draws found it, flag and value.
      if (std::atoi(row[5].c_str()) != 0) {
        fail("RandGaussQ draws changed CLHEP::RandGauss's cached pair");
      }
      continue;
    }
    if (row.size() < 8) {
      fail("short row in gaussq_shoot.csv");
      continue;
    }
    ++n;
    (row[1] == "engine") ? ++n_engine : ++n_static;
    const double mean = parse(row[2]);
    const double sd = parse(row[3]);
    const int draws = std::atoi(row[5].c_str());
    const double u = parse(row[6]);
    const double want = parse(row[7]);
    // Geant4's own count: one flat per G4RandGauss::shoot, never a pair.
    if (draws != 1) { ++bad_draws; }
    TapeRng rng{&u, 1, 0};
    const double got = rand_gauss_q(rng, mean, sd);
    if (rng.i != 1) { ++bad_port_draws; }
    if (!bitwise(got, want)) {
      if (bad++ == 0) { first_bad = row[0] + " i=" + row[4] + ": " + fmt(got) + " want " + row[7]; }
    }
    // shoot() with no arguments, where the row is the unit Gaussian
    if (mean == 0.0 && sd == 1.0) {
      ++n_unit;
      TapeRng r1{&u, 1, 0};
      if (!bitwise(rand_gauss_q(r1) * 1.0 + 0.0, want) || r1.i != 1) { ++bad_unit; }
    }
    // The real_t = float overload: the Gaussian in double exactly as CLHEP computes it, ONE
    // conversion on the way out. HepJamesRandom's uniforms are multiples of 2^-24, so the float
    // tape holds them exactly and the double path is the reference.
    if ((n % 16) == 0) {
      ++n_float;
      const float uf = static_cast<float>(u);
      TapeRngF rf{&uf, 1, 0};
      const float mf = static_cast<float>(mean), sf = static_cast<float>(sd);
      const float gotf = rand_gauss_q(rf, mf, sf);
      const float wantf = static_cast<float>(rand_gauss_q_transform(static_cast<double>(uf)) *
                                                 static_cast<double>(sf) +
                                             static_cast<double>(mf));
      if (!(static_cast<double>(uf) == u) || !bitwise(gotf, wantf) || rf.i != 1) { ++bad_float; }
    }
  }
  std::printf("  %d values (%d through the engine overload, %d through the static one): %d "
              "differ, %d rows where CLHEP drew other than one uniform, %d where the port did\n",
              n, n_engine, n_static, bad, bad_draws, bad_port_draws);
  std::printf("  shoot(): %d unit rows, %d differ;  real_t = float: %d rows, %d differ\n",
              n_unit, bad_unit, n_float, bad_float);
  if (bad > 0) { fail("shoot not bitwise: " + first_bad); }
  if (bad_draws > 0) { fail("the oracle says CLHEP drew other than one uniform per value"); }
  if (bad_port_draws > 0) { fail("the port drew other than one uniform per value"); }
  if (bad_unit > 0) { fail("shoot() with no arguments differs"); }
  if (bad_float > 0) { fail("the real_t = float overload does not convert once, at the end"); }
  if (n != 32000 || n_engine == 0 || n_static == 0) {
    fail("gaussq_shoot.csv is not the 32,000 rows of both overloads");
  }
  if (n_unit == 0 || n_float == 0) { fail("no unit or float rows were compared"); }
  if (flag_rows != 1) { fail("the RandGauss flag row is missing"); }
}

/// The number core/rand_gauss_q.cuh quotes for the tail: `Philox<double>::uniform()` is
/// `(k + 0.5) * 2^-32` for a 32-bit k (core/rng.cuh), so the smallest value it can serve is
/// 2^-33 and the largest 1 - 2^-33, for which `1 - r` is exact - and RandGaussQ is +-6.338
/// there. This checks the transform at the formula's two ends; it does not drive Philox to them.
void check_lattice_bound() {
  std::printf("== 1c. the tail bound at the ends of Philox<double>'s lattice ==\n");
  const double umin = (0.0 + 0.5) * 2.3283064365386963e-10;
  const double umax = (4294967295.0 + 0.5) * 2.3283064365386963e-10;
  const double zlo = tq(umin), zhi = tq(umax);
  std::printf("  (k + 0.5)/2^32 runs from %.17g to 1 - %.17g: RandGaussQ %.9f and %.9f there\n",
              umin, 1.0 - umax, zlo, zhi);
  if (!(umin == std::ldexp(1.0, -33)) || !(1.0 - umax == std::ldexp(1.0, -33))) {
    fail("the lattice's ends are not 2^-33 and 1 - 2^-33");
  }
  if (!(zlo == -zhi) || !(std::fabs(zlo) > 6.3379) || !(std::fabs(zlo) < 6.3380)) {
    fail("RandGaussQ at the lattice's ends is not +-6.3380");
  }
}

// ---------------------------------------------------------------------------------------------
// 2. the draw count at every site
// ---------------------------------------------------------------------------------------------

/// One site, one tape: the count the site must consume, and the value it must return.
void expect(const char* site, const char* what, int drawn, int want_drawn, double got,
            double want) {
  const bool ok_n = (drawn == want_drawn);
  const bool ok_v = bitwise(got, want);
  std::printf("  %-44s %-26s drew %d (want %d)  %s\n", site, what, drawn, want_drawn,
              ok_v ? "value exact" : ("value " + fmt(got) + " want " + fmt(want)).c_str());
  if (!ok_n) { fail(std::string(site) + " " + what + ": drew the wrong number of uniforms"); }
  if (!ok_v) { fail(std::string(site) + " " + what + ": value is not the tape's RandGaussQ"); }
}

void check_sites() {
  std::printf("== 2. the draw count at every G4RandGauss site ==\n");
  data::Material<double> mats[data::kNumMaterials];
  data::build_b1_materials<double>(mats);
  const data::Material<double>& water = mats[data::kWater];

  // ---- G4UniversalFluctuation::SampleFluctuations, the thick-absorber Gaussian (.cc:138).
  // A 100 MeV proton over 5 mm of water with a 3.6 MeV mean loss: tmax 0.229 MeV is under the
  // 0.278 MeV cut, so tcut = tmax and the regime test passes with room.
  {
    const auto pd = particle_def<double>(ParticleType::kProton);
    const double e = 100.0, L = 5.0, mean = 3.6;
    const double tmax = em::hadron_max_secondary_energy(pd, e);
    const double tcut = std::fmin(water.cut_electron, tmax);
    const double etot = e + pd.mass;
    const double beta2 = e * (e + 2.0 * pd.mass) / (etot * etot);
    const double siga = std::sqrt((tmax / beta2 - 0.5 * tcut) * em::twopi_mc2_rcl2<double>() * L *
                                  1.0 * water.electron_density);
    const bool regime = pd.mass > units::electron_mass_c2<double>() &&
                        mean >= em::kFlucMinNBohr<double>() * tcut && tmax <= 2.0 * tcut &&
                        mean / siga >= 2.0;
    if (!regime) { fail("SampleFluctuations case is not in the Gaussian regime"); }
    const char* site = "UniversalFluctuation::SampleFluctuations";
    {
      const double tape[] = {0.7};
      TapeRng rng{tape, 1, 0};
      const double got = em::sample_fluctuation(water, pd, e, tcut, tmax, L, mean, 1.0, rng);
      expect(site, "accepted first", rng.i, 1, got, tq(0.7) * siga + mean);
    }
    {
      // 1e-300 is z = -37.05: a loss below zero, rejected, and the next uniform is taken.
      const double tape[] = {1e-300, 1e-300, 1e-300, 0.3};
      TapeRng rng{tape, 4, 0};
      const double got = em::sample_fluctuation(water, pd, e, tcut, tmax, L, mean, 1.0, rng);
      expect(site, "three rejected, one taken", rng.i, 4, got, tq(0.3) * siga + mean);
    }
  }

  // ---- G4UniversalFluctuation::SampleGauss (.hh:156), called from SampleGlandz.
  {
    const char* site = "UniversalFluctuation::SampleGauss";
    const double eav = 1.0, esig2 = 0.04;
    const double sig = std::sqrt(esig2);
    {
      const double tape[] = {0.7};
      TapeRng rng{tape, 1, 0};
      double eloss = 0.0;
      em::fluc_sample_gauss(eav, esig2, eloss, rng);
      expect(site, "accepted first", rng.i, 1, eloss, 0.0 + (tq(0.7) * sig + eav));
    }
    {
      const double tape[] = {1e-300, 0.2};
      TapeRng rng{tape, 2, 0};
      double eloss = 0.0;
      em::fluc_sample_gauss(eav, esig2, eloss, rng);
      expect(site, "one rejected, one taken", rng.i, 2, eloss, 0.0 + (tq(0.2) * sig + eav));
    }
  }

  // ---- G4IonFluctuations::SampleFluctuations (.cc:149). A 20 MeV alpha - under the 79.45 MeV
  // hand-over to the Universal model - over 50 um of water, losing 1.2 MeV: under a fifth of
  // its energy, so no widening, and far more than two sigma.
  {
    const char* site = "IonFluctuations::SampleFluctuations";
    const auto pd = particle_def<double>(ParticleType::kAlpha);
    const double e = 20.0, L = 0.05, mean = 1.2, q2 = 4.0;
    const double tmax = em::hadron_max_secondary_energy(pd, e);
    const double tcut = std::fmin(water.cut_electron, tmax);
    const double siga = std::sqrt(em::ion_dispersion<double>(water, pd, e, tcut, tmax, L, q2));
    const bool regime = e <= em::ion_vavilov_threshold<double>(pd) && mean <= 0.2 * e &&
                        mean / siga >= 2.0;
    if (!regime) { fail("IonFluctuations case is not in the Gaussian regime"); }
    {
      const double tape[] = {0.7};
      TapeRng rng{tape, 1, 0};
      const double got = em::sample_ion_fluctuation(water, pd, e, tcut, tmax, L, mean, q2, rng);
      expect(site, "accepted first", rng.i, 1, got, tq(0.7) * siga + mean);
    }
    {
      const double tape[] = {1e-300, 0.25};
      TapeRng rng{tape, 2, 0};
      const double got = em::sample_ion_fluctuation(water, pd, e, tcut, tmax, L, mean, q2, rng);
      expect(site, "one rejected, one taken", rng.i, 2, got, tq(0.25) * siga + mean);
    }
  }

  // ---- G4UrbanMscModel::Randomizetlimit (.hh:220), on both step-limit branches.
  {
    const auto c = em::urban_coeffs(water);
    {
      // fUseSafety (e-/e+): a carried tlimit of 1 mm, tlimitmin 0.01 mm, range 10 mm, on a
      // boundary (safety 0), so tlimit = 1 mm < range and the limit is randomised.
      const char* site = "UrbanMsc::Randomizetlimit, fUseSafety";
      const double tape[] = {0.7};
      TapeRng rng{tape, 1, 0};
      double tlimit_base = 1.0, tlimitmin = 0.01;
      const double got = em::urban_step_limit(c, 1.0, 1.0, 10.0, 0.0, false, rng, tlimit_base,
                                              tlimitmin);
      const double want = std::fmin(
          10.0, std::fmax(tq(0.7) * (0.1 * (1.0 - 0.01)) + 1.0, 0.01));
      expect(site, "tlimit > tlimitmin", rng.i, 1, got, want);
      // and tlimit == tlimitmin draws nothing at all: `if(tlimit > tlimitmin)` is Geant4's
      double base2 = 0.005, min2 = 0.01;
      TapeRng none{tape, 1, 0};
      const double got2 = em::urban_step_limit(c, 1.0, 1.0, 10.0, 0.0, false, none, base2, min2);
      expect(site, "tlimit == tlimitmin", none.i, 0, got2, 0.01);
    }
    {
      // fMinimal (a proton): tlimit 1 mm carried, t_path 5 mm, range 10 mm, safety 0.
      const char* site = "UrbanMsc::Randomizetlimit, fMinimal";
      const double tape[] = {0.3};
      TapeRng rng{tape, 1, 0};
      double tlimit = 1.0;
      const double mass = particle_def<double>(ParticleType::kProton).mass;
      const double got = em::urban_step_limit_heavy(c, 1.0, 0.2, 100.0, mass, 10.0, 0.0, 5.0,
                                                    rng, tlimit);
      const double tmin = em::kTlimitMinMinimal<double>();
      const double want = std::fmin(5.0, std::fmax(tq(0.3) * (0.1 * (1.0 - tmin)) + 1.0, tmin));
      expect(site, "tlimit > tlimitmin", rng.i, 1, got, want);
    }
  }

  // ---- G4WentzelVIModel::SampleScattering's lateral displacement (.cc:658-659): TWO draws, x
  // then y. One multiple-scattering sub-step and nothing else: xtsec = 0, so no single scatter
  // and no interval draw; z0 = 0.005 < zzmin, so the step is not split; prob2 = 0, so isFirst
  // draws nothing. The sub-step then takes one uniform for z and one for phi before the
  // displacement, and the difference between displacement on and off is the displacement's.
  {
    const char* site = "WentzelVI::SampleScattering displacement";
    const auto pd = particle_def<double>(ParticleType::kElectron);
    em::WentzelMscState<double> st{};
    st.t_path = 0.1;
    st.z_path = 0.09;
    st.lambda_eff = 10.0;
    st.cos_theta_min = 1.0;
    st.cos_tet_max_nuc = -1.0;
    st.xtsec = 0.0;
    st.range = 100.0;
    st.pre_kin_energy = 200.0;
    st.eff_kin_energy = 200.0;
    st.single_scattering_mode = false;
    em::WentzelElementXs<double> els{};
    els.n = 1;
    els.cumulative[0] = 1.0;
    els.electron_fraction[0] = 0.0;
    const Vec3<double> up{0.0, 0.0, 1.0};
    const double tape[] = {0.5, 0.3, 0.9, 0.2};
    TapeRng on{tape, 4, 0};
    const auto res = em::wv_sample_scattering(water, pd, ParticleType::kElectron, st, els, 1.0,
                                              -1.0, up, true, on);
    TapeRng off{tape, 2, 0};
    (void)em::wv_sample_scattering(water, pd, ParticleType::kElectron, st, els, 1.0, -1.0, up,
                                   false, off);
    // The displacement as G4WentzelVIModel.cc:655-659 builds it from this tape. With the
    // incident direction +z both rotations are the identity, so it comes out unrotated.
    const double z0 = st.t_path * (0.5 / st.lambda_eff);
    double z = -std::log(0.5);
    z *= z0;
    const double cost = 1.0 - 2.0 * z;
    const double sint = std::sqrt((1.0 - cost) * (1.0 + cost));
    const double phi = units::twopi<double>() * 0.3;
    const double vx1 = sint * std::cos(phi);
    const double vy1 = sint * std::sin(phi);
    const double invsqrt12 = 1.0 / 3.4641016151377544;
    const double rms = invsqrt12 * std::sqrt(2.0 * z0);
    const double r = st.t_path * (st.z_path / st.t_path);
    const double dx = r * (0.5 * vx1 + rms * tq(0.9));
    const double dy = r * (0.5 * vy1 + rms * tq(0.2));
    expect(site, "x from the 1st Gaussian", on.i - off.i, 2, res.displacement.x, dx);
    expect(site, "y from the 2nd Gaussian", on.i, 4, res.displacement.y, dy);
    // The order is a fact the comparison can see: the same tape with y drawn first would give
    // these, and they are different numbers.
    const double dx_swapped = r * (0.5 * vx1 + rms * tq(0.2));
    if (bitwise(dx_swapped, dx)) { fail("the displacement test cannot tell x from y"); }
  }

  // ---- G4CompetitiveFission::FissionCharge (.cc:292) and FissionKineticEnergy (.cc:363).
  {
    const char* site = "CompetitiveFission::FissionCharge";
    {
      // U238 into A = 100: Af <= A - 134, so DeltaZ = +0.45 and Zmean = 39.106
      const double tape[] = {0.7};
      TapeRng rng{tape, 1, 0};
      const int got = deex::fission_charge(238, 92, 100.0, rng);
      const double zmean = (100.0 / 238) * 92 + 0.45;
      const int want = static_cast<int>(std::nearbyint(tq(0.7) * 0.6 + zmean));
      expect(site, "accepted first", rng.i, 1, got, want);
    }
    {
      // A = 3: Zmean = 1.61, and u = 0.05 puts Z at 0.62, below the Z >= 1 window
      const double tape[] = {0.05, 0.5};
      TapeRng rng{tape, 2, 0};
      const int got = deex::fission_charge(238, 92, 3.0, rng);
      const double zmean = (3.0 / 238) * 92 + 0.45;
      const int want = static_cast<int>(std::nearbyint(tq(0.5) * 0.6 + zmean));
      expect(site, "one rejected, one taken", rng.i, 2, got, want);
    }
  }
  {
    const char* site = "CompetitiveFission::FissionKineticEnergy";
    const double U = 20.0;
    deex::FissionParameters p;
    p.define(238, 92, U, deex::fission_barrier(238, 92, U));
    // The mode draw first (0.999 against Psy picks the asymmetric mode, ESigma = 10 MeV), then
    // the Gaussian. u = 0.5 is z = 0 exactly, so that call returns TaverageAfMax itself.
    const double t0[] = {0.999, 0.5};
    TapeRng r0{t0, 2, 0};
    const double t_average = deex::fission_kinetic_energy(p, 238, 92, 140, 98, 1000.0, r0);
    // The value of this call IS the reference for the two below, so what it is checked against is
    // the window every accepted energy must fall in: Eaverage +- 3.72*ESigma.
    const double eaverage = 0.1071 * (92.0 * 92) / data::g4pow_z13<double>(238) + 22.2;
    const bool in_window = std::isfinite(t_average) && t_average >= eaverage - 3.72 * 10.0 &&
                           t_average <= eaverage + 3.72 * 10.0;
    expect(site, "mode, z = 0: in the window", r0.i, 2, in_window ? 1.0 : 0.0, 1.0);
    const double t1[] = {0.999, 0.7};
    TapeRng r1{t1, 2, 0};
    const double got1 = deex::fission_kinetic_energy(p, 238, 92, 140, 98, 1000.0, r1);
    expect(site, "mode, then accepted", r1.i, 2, got1, tq(0.7) * 10.0 + t_average);
    const double t2[] = {0.999, 1e-300, 0.7};
    TapeRng r2{t2, 3, 0};
    const double got2 = deex::fission_kinetic_energy(p, 238, 92, 140, 98, 1000.0, r2);
    expect(site, "mode, one rejected, one taken", r2.i, 3, got2, tq(0.7) * 10.0 + t_average);
  }
}

}  // namespace

int main() {
  const char* env = std::getenv("G4GPU_ORACLE");
  const std::string dir = (env != nullptr) ? env : "ref/oracle";
  check_transform(dir);
  check_shoot(dir);
  check_lattice_bound();
  check_sites();
  std::printf("\ntest_rand_gauss_q: %s (%d failure%s)\n", fails == 0 ? "PASS" : "FAIL", fails,
              fails == 1 ? "" : "s");
  return fails == 0 ? 0 : 1;
}
