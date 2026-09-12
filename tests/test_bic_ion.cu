// G4BinaryLightIonReaction's fusion arm - the model QBBC gives every ion below 6 GeV/nucleon -
// against ref/oracle/bic_blir*.csv.
//
// `tests/test_bic_nucleus.cu` covers the machinery UNDER the two binary-cascade models: the
// nucleus, the densities, the fields and the Runge-Kutta propagator, all of it exact against a
// replayed configuration. This file covers the one place a whole MODEL is called the way the
// framework calls it, and it is the piece that matters for a galactic-cosmic-ray shielding
// calculation: below 50 MeV per nucleon `G4BinaryLightIonReaction::ApplyYourself` runs no
// cascade at all - it fuses the two nuclei into one compound of (pZ+tZ, pA+tA) with pA particle
// excitons of which pZ are charged and no holes, and hands it to `G4PreCompoundModel::DeExcite`.
//
// **Why this file had to exist.** `light_ion_reaction.cuh`'s header said the fusion arm was
// "complete here and validated end to end" while nothing whatsoever included the header. It
// compiled in no translation unit, so the claim was not merely unchecked: the code had never
// been built. That is docs/RISK.md V52's failure in its strongest form - an assertion made in a
// comment - and it is why the port's entry point is now instantiated by a `__global__` probe as
// well as called here.
//
// **What is exact and what is statistical.** The port's engine is Philox and Geant4's is
// HepJamesRandom, so the product lists cannot be compared event by event. Three things can be
// compared exactly anyway, and they are the three that would break if the gate, the swap or the
// bookkeeping were wrong:
//
//   * the FUSION GATE's verdict. `m2Compound < mFused^2` returns null and `ApplyYourself` then
//     returns the primary ALIVE with its own energy and direction - the one branch of this
//     model that does not kill the primary. It is deterministic per case, so the oracle's
//     `status` column is an integer the port must reproduce. One of the twenty cases takes it:
//     an alpha at 1 MeV/nucleon on hydrogen, because Li5 is unbound and `GetIonMass(3,5)`
//     exceeds the invariant mass of alpha + p. That case is the whole reason hydrogen is in the
//     target list - water and polyethylene are the shields this project cares about.
//   * the COMPOUND's (Z, A). Every secondary of every event must add up to (pZ+tZ, pA+tA),
//     whether or not `SetLighterAsProjectile` swapped, and the oracle reports -1 if Geant4 ever
//     varied. Exact integers, 100,000 events.
//   * the ENERGY AND MOMENTUM BALANCE, ONCE THE CONVERSION ELECTRONS ARE PAID FOR. The compound
//     four-momentum is `(mom.vect(), mom.e() + mTarget)`, and this test was written expecting
//     the secondaries to add up to it exactly. They do not, and the amount by which they miss is
//     exactly one electron rest mass per conversion electron.
//
//     `G4PhotonEvaporation::GenerateGamma` computes the emitting system's invariant mass as
//     `ecm = lv.mag()` and then, for internal conversion, `ecm += (electron_mass_c2 -
//     bond_energy)` with `bond_energy` a local initialised to 0 and never assigned - the
//     atomic binding that should have paid for the electron's rest mass. So the emission
//     rescales the whole four-momentum by `(1 + m_e/M)` and CREATES 511 keV. Measured on P6's
//     `deexcite` directly, with the cascade taken out of the picture: a deuteron-on-Pb208
//     compound emits 0.26 conversion electrons per event and its product list comes out
//     +0.13286 MeV heavy, against `0.26 * m_e = 0.13286 MeV`; a C12-on-Pb208 compound, 0.4198
//     per event and +0.214539 against 0.214517; and the two cases with no conversion electron
//     at all balance to 1.1e-13 MeV. docs/RISK.md V76.
//
//     It is Geant4's arithmetic and the port reproduces it, so the comparison subtracts
//     `n_electrons * m_e` from BOTH sides - the oracle's count is the `pdg = 11` row of
//     bic_blir.csv - and what is left is exact again. That is the only form in which this
//     assertion says anything: comparing the raw sums instead compares two Poisson counts of
//     conversion electrons and fails at three sigma on the heavy targets for a reason that has
//     nothing to do with the ion model. It still catches a swap that boosted the wrong way,
//     which is what it is here for: the mirror `tmp.setVect(-tmp.vect())` after
//     `toBreit.inverse()` changes the sign of the momentum balance and nothing else.
//
// Everything else - the species yields and their kinetic-energy moments - is statistical at
// N = 5,000, with the 5 sigma band and the per-event multiplicity variance that
// tests/test_precompound.cu and tests/test_deex_breakup.cu define. The comparison is keyed on
// (Z, A) and not on the PDG code for the reason test_precompound.cu's `pdg_to_za` gives: the
// isomer digit of `10LZZZAAAI` is a property of the run's ion table and P3 refuses to invent it.
#include <algorithm>
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
#include "physics/hadronic/bic/light_ion_reaction.cuh"

using namespace g4gpu;

/// The entry point is device code and a `__host__ __device__` template that is only ever called
/// from the host is never compiled for the device. This probe is never launched; instantiating
/// it is what proves the model builds for sm_86, and `-Xptxas -v` on this translation unit is
/// what reports its frame. Nothing here is a local array: the products are the workspace's.
__global__ void bic_blir_probe(physics::hadronic::HadProjectile<double> proj,
                               physics::hadronic::HadNucleus tgt, const data::LevelTable* lt,
                               const deex::FermiPool* pool, preco::PrecoWorkspace ws,
                               bic::BlirFinalState* fs, bic::BlirRefusal* ref) {
  Philox<double> rng(1u, 2u, 3u);
  bic::blir_apply_yourself(proj, tgt, *lt, *pool, ws, rng, *fs, *ref);
}

namespace {

int fails = 0;

// ---------------------------------------------------------------------------------------------
// Statistics, exactly as tests/test_precompound.cu defines them
// ---------------------------------------------------------------------------------------------

long long n_poisson_fallback = 0;

struct Tally {
  long long count = 0;
  double sum_e = 0.0;
  double sum_e2 = 0.0;
  double sum_k2 = 0.0;   ///< per-event multiplicity second moment, summed over events
  bool has_k2 = false;
};

double tally_var(const Tally& t, long long n) {
  if (!t.has_k2 || n < 1) { return -1.0; }
  const double m = static_cast<double>(t.count) / static_cast<double>(n);
  const double v = t.sum_k2 / static_cast<double>(n) - m * m;
  return (v > 0.0) ? v : 0.0;
}

/// A species yield is a sum of per-event multiplicities, so its variance is the multiplicity's
/// and not the count's; `mean_mult2` is dumped for exactly this. The Poisson fallback is for a
/// species the other side never produced, where there is no variance to be had.
double multiplicity_z(long long n1, long long n2, long long n, double var1, double var2) {
  if (n < 2) { return 0.0; }
  double v1 = (var1 > 0.0) ? var1 : 0.0;
  double v2 = var2;
  if (v2 < 0.0) {
    const double m1 = static_cast<double>(n1) / static_cast<double>(n);
    const double m2 = static_cast<double>(n2) / static_cast<double>(n);
    v2 = (m1 > m2) ? m1 : m2;
    ++n_poisson_fallback;
  }
  if (v2 < 0.0) { v2 = 0.0; }
  const double s2 = static_cast<double>(n) * (v1 + v2);
  if (!(s2 > 0.0)) { return (n1 == n2) ? 0.0 : 1.e9; }
  return std::fabs(static_cast<double>(n1 - n2)) / std::sqrt(s2);
}

double mean_z(double m1, double v1, long long n1, double m2, double v2, long long n2) {
  if (n1 < 2 || n2 < 2) { return 0.0; }
  const double s2 = v1 / static_cast<double>(n1) + v2 / static_cast<double>(n2);
  if (!(s2 > 0.0)) { return 0.0; }
  return std::fabs(m1 - m2) / std::sqrt(s2);
}

/// mean_z where the quantity can be deterministic - Li5 breaks up into an alpha and a proton in
/// every event, so both variances are zero and mean_z's answer of 0 would accept anything.
double moment_z(double m1, double v1, long long n1, double m2, double v2, long long n2) {
  if (!(v1 > 0.0) && !(v2 > 0.0)) {
    return (std::fabs(m1 - m2) <= 1.e-9 * (1.0 + std::fabs(m2))) ? 0.0 : 1.e9;
  }
  return mean_z(m1, v1, n1, m2, v2, n2);
}

/// (Z, A) from an oracle PDG code, isomer digit folded away. See the file header.
bool pdg_to_za(int pdg, int& z, int& a) {
  if (pdg == 22) { z = 0; a = 0; return true; }
  if (pdg == 11) { z = -1; a = 0; return true; }
  if (pdg == 2112) { z = 0; a = 1; return true; }
  if (pdg == 2212) { z = 1; a = 1; return true; }
  if (pdg > 1000000000) {
    a = (pdg / 10) % 1000;
    z = (pdg / 10000) % 1000;
    return true;
  }
  return false;
}

int za_key(int z, int a) { return (z + 500) * 1000 + a; }
std::string za_label(int key) {
  return "Z=" + std::to_string(key / 1000 - 500) + " A=" + std::to_string(key % 1000);
}

// ---------------------------------------------------------------------------------------------
// CSV reading, the same shape test_bic_nucleus.cu uses
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

/// P6's buffers, sized for the heaviest case here: a C12 at 25 MeV/nucleon on Pb208 fuses into
/// a compound of A = 220 and the evaporation cascade on it emits sixteen products on average.
/// Same shape as tests/test_precompound.cu's, one size up on the evaporation list.
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

/// One oracle case, from bic_blir_status.csv.
struct OracleCase {
  std::string name;
  int pz = 0, pa = 0, tz = 0, ta = 0;
  double ekin_per_a = 0.0;
  long long n = 0;
  std::string status;
  long long n_secondaries = 0;
  long long sum_z = 0, sum_a = 0;
  double mean_e = 0.0, mean_pz = 0.0, mean_mult = 0.0;
};

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
  const deex::FermiPool pool = ps.view();

  // -------------------------------------------------------------------------------------------
  // The oracle
  // -------------------------------------------------------------------------------------------
  std::vector<OracleCase> cases;
  for (const auto& row : read_csv("bic_blir_status.csv")) {
    OracleCase c;
    c.name = sv(row, 0);
    c.pz = iv(row, 1);
    c.pa = iv(row, 2);
    c.ekin_per_a = dv(row, 3);
    c.tz = iv(row, 4);
    c.ta = iv(row, 5);
    c.n = lv(row, 6);
    c.status = sv(row, 7);
    c.n_secondaries = lv(row, 8);
    c.sum_z = lv(row, 9);
    c.sum_a = lv(row, 10);
    c.mean_e = dv(row, 11);
    c.mean_pz = dv(row, 12);
    c.mean_mult = dv(row, 13);
    cases.push_back(c);
  }
  if (cases.empty()) {
    std::printf("no cases in bic_blir_status.csv\n");
    return 1;
  }

  // (case, (Z, A)) -> tally, folded off the per-PDG oracle rows.
  std::map<std::string, std::map<int, Tally>> g4;
  for (const auto& row : read_csv("bic_blir.csv")) {
    const std::string name = sv(row, 0);
    int z = 0, a = 0;
    if (!pdg_to_za(iv(row, 7), z, a)) {
      std::printf("bic_blir.csv: unconvertible pdg %d in %s\n", iv(row, 7), name.c_str());
      ++fails;
      continue;
    }
    Tally& t = g4[name][za_key(z, a)];
    const long long cnt = lv(row, 8);
    const double me = dv(row, 9), me2 = dv(row, 10), mk2 = dv(row, 11);
    // Folding two PDG rows (an isomer and its ground state) onto one (Z, A) means adding counts
    // and re-weighting the energy moments. The multiplicity second moment CANNOT be added: the
    // per-event multiplicities of the two isomers are correlated and `E[(k1+k2)^2]` is not
    // `E[k1^2]+E[k2^2]`. So `has_k2` is set only while the key has seen exactly one PDG row,
    // and a folded key falls back to the Poisson variance - counted and printed, never silent.
    const double old = static_cast<double>(t.count);
    t.sum_e += me * static_cast<double>(cnt);
    t.sum_e2 += me2 * static_cast<double>(cnt);
    if (t.count == 0) { t.sum_k2 = mk2 * static_cast<double>(cases[0].n); t.has_k2 = true; }
    else { t.has_k2 = false; }
    t.count += cnt;
    (void)old;
  }

  // -------------------------------------------------------------------------------------------
  // The port, case by case
  // -------------------------------------------------------------------------------------------
  Buffers bufs;
  bic::BlirFinalState result;

  double worst_count = 0.0, worst_ekin = 0.0, worst_mult = 0.0;
  std::string worst_count_at, worst_ekin_at, worst_mult_at;
  /// The worst few, not just the worst one: a single outlier and a systematic tilt look the
  /// same in a one-line summary, and the difference decides whether the answer is "resample" or
  /// "read the code". Kept for both statistical buckets and printed below the table.
  std::vector<std::pair<double, std::string>> top_count, top_ekin;
  auto keep_top = [](std::vector<std::pair<double, std::string>>& v, double z,
                     const std::string& what) {
    v.emplace_back(z, what);
    if (v.size() > 64) {
      std::sort(v.begin(), v.end(),
                [](const std::pair<double, std::string>& a,
                   const std::pair<double, std::string>& b) { return a.first > b.first; });
      v.resize(16);
    }
  };
  long long n_count = 0, n_ekin = 0, n_mult = 0;
  int n_status_bad = 0, n_za_bad = 0, n_balance_bad = 0, n_refused = 0, n_overflow = 0;
  long long n_thin = 0;
  double worst_e = 0.0, worst_pz = 0.0, worst_ev_exact = 0.0, worst_ev_ic = 0.0;
  double worst_pe = 0.0;
  std::string worst_e_at, worst_pz_at, worst_ev_exact_at, worst_ev_ic_at;

  for (const OracleCase& c : cases) {
    physics::hadronic::HadProjectile<double> proj;
    proj.pdg = physics::hadronic::pdg_nuclear_code(c.pz, c.pa);
    proj.baryon_number = c.pa;
    proj.charge = static_cast<double>(c.pz);
    // `G4HadProjectile::Get4Momentum()` is built from the dynamic particle's mass, which for an
    // ion is `G4IonTable::GetIonMass(Z, A)`. light_ion_reaction.cuh's header argues that equals
    // `G4NucleiProperties::GetNuclearMass(A, Z)` for every (Z <= A, A >= 1); the energy balance
    // below is what tests that claim, because a wrong projectile mass moves mean_e by its error.
    proj.mass = deex::nuclear_mass(c.pa, c.pz);
    proj.kin_energy = c.ekin_per_a * c.pa;

    physics::hadronic::HadNucleus tgt;
    tgt.z = c.tz;
    tgt.a = c.ta;
    tgt.l = 0;

    std::map<int, Tally> mine;
    std::map<int, int> per_event;
    long long n_sec = 0, n_alive = 0, n_kill = 0, n_electrons = 0;
    long long sum_z = -1, sum_a = -1;
    bool za_varies = false;
    double sum_tot_e = 0.0, sum_tot_pz = 0.0;
    double worst_ev_e = 0.0, worst_ev_e_ic = 0.0, worst_per_electron = 0.0;
    /// The conversion-electron surplus, per event: `n_e * m_e` in energy and the same rest mass
    /// carried along by the boost in momentum. See the file header.
    const double kMe = units::electron_mass_c2<double>();

    Philox<double> rng(0x51ed270bu, static_cast<unsigned>(c.pa * 1000 + c.ta),
                       static_cast<unsigned>(c.ekin_per_a * 10.0) + 1u);
    for (long long ev = 0; ev < c.n; ++ev) {
      bic::BlirRefusal ref;
      preco::PrecoWorkspace ws = bufs.view();
      const preco::PrecoStatus st =
          bic::blir_apply_yourself(proj, tgt, lt, pool, ws, rng, result, ref);
      if (ref.cascade || ref.anti_or_hyper) { ++n_refused; continue; }
      if (ref.capacity || result.secondary_overflow > 0) { ++n_overflow; }
      if (st.ref.any()) { ++n_refused; }
      if (result.status == physics::hadronic::HadFinalStateStatus::kIsAlive) {
        ++n_alive;
        continue;
      }
      ++n_kill;
      per_event.clear();
      long long ez = 0, ea = 0;
      int n_ev_electrons = 0;
      double tot_e = 0.0, tot_pz = 0.0;
      n_sec += result.n_secondaries;
      for (int i = 0; i < result.n_secondaries; ++i) {
        const physics::hadronic::HadSecondary<double>& s = result.secondaries[i];
        // An electron is (Z = 0, A = 0) in P3's product, told apart from a gamma by its PDG.
        const int z = (s.a == 0 && s.pdg == 11) ? -1 : s.z;
        if (z == -1 && s.a == 0) { ++n_ev_electrons; }
        const int key = za_key(z, s.a);
        Tally& t = mine[key];
        ++t.count;
        t.has_k2 = true;
        t.sum_e += s.kin_energy;
        t.sum_e2 += s.kin_energy * s.kin_energy;
        ++per_event[key];
        tot_e += s.total_energy();
        tot_pz += s.momentum() * s.direction.z;
        ea += s.a;
        ez += s.z;
      }
      for (const auto& kv : per_event) {
        mine[kv.first].sum_k2 += static_cast<double>(kv.second) * kv.second;
      }
      if (sum_z < 0) { sum_z = ez; sum_a = ea; }
      else if (ez != sum_z || ea != sum_a) { za_varies = true; }
      n_electrons += n_ev_electrons;
      sum_tot_e += tot_e - n_ev_electrons * kMe;
      // The same surplus in momentum. The emission scales the whole four-vector by `1 + m_e/M`,
      // so the energy gains `gamma*m_e` and the momentum `gamma*beta*m_e`; `mean_pz/mean_e` is
      // `beta` to the precision this needs, and dropping the gamma is what leaves the 1e-7
      // residue the tolerance below is set from.
      sum_tot_pz += tot_pz - n_ev_electrons * kMe * (c.mean_pz / c.mean_e);
      // Per event, and against the port's OWN compound rather than the oracle's mean: this is
      // the strongest form of the statement and it needs no oracle at all. Everything the
      // de-excitation emitted, minus the rest mass it created for each conversion electron,
      // is the four-momentum `FuseNucleiAndPrompound` handed it.
      const double want_e = proj.kin_energy + proj.mass + deex::nuclear_mass(c.ta, c.tz);
      const double de = std::fabs(tot_e - n_ev_electrons * kMe - want_e);
      // Split on whether the event emitted a conversion electron at all, because the two are
      // different statements. With none, the balance is EXACT and the bound below is the ulp of
      // a 205 GeV sum. With one or more, the flat `n_e * m_e` is only the first term: the mass
      // is added to the EMITTING residual's invariant mass and the four-vector is then rescaled
      // and re-clamped by `GenerateGamma`'s two `if`s, so a residue of order 1e-4 MeV per
      // electron is left. Lumping them into one tolerance would hide the exact half behind the
      // approximate one - which is what the first version of this test did.
      if (n_ev_electrons == 0) {
        if (de > worst_ev_e) { worst_ev_e = de; }
      } else {
        if (de > worst_ev_e_ic) { worst_ev_e_ic = de; }
        if (de / n_ev_electrons > worst_per_electron) {
          worst_per_electron = de / n_ev_electrons;
        }
      }
    }

    // ---- exact: the fusion gate's verdict
    const std::string status = (n_alive == c.n) ? "isAlive"
                                                : ((n_kill == c.n) ? "stopAndKill" : "MIXED");
    if (status != c.status) {
      std::printf("GATE %s: port %s, Geant4 %s\n", c.name.c_str(), status.c_str(),
                  c.status.c_str());
      ++n_status_bad;
      ++fails;
    }
    if (c.status == "isAlive") {
      // The one branch that does not kill the primary: it must come back with the projectile's
      // own kinetic energy and direction, which is what G4BinaryLightIonReaction sets.
      if (std::fabs(result.energy_change - proj.kin_energy) > 1e-12 * proj.kin_energy ||
          result.momentum_change.z != 1.0) {
        std::printf("GATE %s: isAlive but E=%.17g (want %.17g) dir.z=%g\n", c.name.c_str(),
                    result.energy_change, proj.kin_energy, result.momentum_change.z);
        ++fails;
      }
      continue;
    }

    // ---- exact: the compound's (Z, A), in every event
    if (za_varies || sum_z != c.sum_z || sum_a != c.sum_a) {
      std::printf("COMPOUND %s: port (Z=%lld A=%lld%s), Geant4 (Z=%lld A=%lld)\n",
                  c.name.c_str(), sum_z, sum_a, za_varies ? ", VARIES" : "", c.sum_z, c.sum_a);
      ++n_za_bad;
      ++fails;
    }

    // ---- exact, once the conversion electrons are paid for on both sides. See the file
    // header: the emission creates `m_e` per electron, so the raw sums differ by the two sides'
    // Poisson counts and the corrected ones do not. The oracle's count is the `Z=-1, A=0` key.
    //
    // The tolerance is 1e-5 MeV. It is not the ulp of the total - 205 GeV summed over sixteen
    // products rounds at about 3e-8 MeV - but the residue of the correction: the electron's rest
    // mass is added to the emitting system's INVARIANT mass and the whole four-vector is then
    // scaled by `(1 + m_e/M)`, so subtracting a flat `m_e` leaves `m_e * (E/M - 1)` behind, a
    // few times 1e-7 MeV per electron here. Anything at 1e-3 is a physics difference and
    // anything at 0.5 is an electron.
    const double nk = (n_kill > 0) ? static_cast<double>(n_kill) : 1.0;
    const long long o_electrons = g4[c.name].count(za_key(-1, 0))
                                      ? g4[c.name][za_key(-1, 0)].count : 0;
    const double o_mean_e = c.mean_e - static_cast<double>(o_electrons) /
                                           static_cast<double>(c.n) * kMe;
    const double o_mean_pz = c.mean_pz - static_cast<double>(o_electrons) /
                                             static_cast<double>(c.n) * kMe *
                                             (c.mean_pz / c.mean_e);
    const double de = std::fabs(sum_tot_e / nk - o_mean_e);
    const double dp = std::fabs(sum_tot_pz / nk - o_mean_pz);
    if (de > worst_e) { worst_e = de; worst_e_at = c.name; }
    if (dp > worst_pz) { worst_pz = dp; worst_pz_at = c.name; }
    if (worst_ev_e > worst_ev_exact) { worst_ev_exact = worst_ev_e; worst_ev_exact_at = c.name; }
    if (worst_ev_e_ic > worst_ev_ic) { worst_ev_ic = worst_ev_e_ic; worst_ev_ic_at = c.name; }
    if (worst_per_electron > worst_pe) { worst_pe = worst_per_electron; }
    if (de > 1e-5 || worst_ev_e > 1e-8) {
      std::printf("BALANCE %s: mean dE %.3g, worst dE with no conversion electron %.3g MeV "
                  "(port %lld electrons, Geant4 %lld)\n",
                  c.name.c_str(), de, worst_ev_e, n_electrons, o_electrons);
      ++n_balance_bad;
      ++fails;
    }

    // ---- statistical: the species yields and their kinetic-energy moments
    std::map<int, Tally>& go = g4[c.name];
    for (const auto& kv : go) {
      const int key = kv.first;
      const Tally& o = kv.second;
      const Tally& p = mine[key];
      const double z = multiplicity_z(p.count, o.count, c.n, tally_var(p, c.n),
                                      tally_var(o, c.n));
      ++n_count;
      keep_top(top_count, z, c.name + " " + za_label(key) + " port " +
                             std::to_string(p.count) + " g4 " + std::to_string(o.count));
      if (z > worst_count) {
        worst_count = z;
        worst_count_at = c.name + " " + za_label(key) + " port " + std::to_string(p.count) +
                         " g4 " + std::to_string(o.count);
      }
      // **A minimum of 25 on both sides, and it is not a convenience.** A two-sample mean test
      // divides by an estimate of the variance, and at n = 2 or 3 that estimate is itself a
      // draw: the first run of this test reported 8.55 sigma on a Pb-201 residual the oracle
      // produced THREE times, and 7.12 on one it produced twice. Those are not disagreements,
      // they are the t-distribution's tail being read as a normal's. 25 is where the two are
      // within 10% at five sigma. What the rare species ARE checked by is the yield bucket
      // above, which is a count test and correct at any count. `n_thin` says how many were
      // skipped, so a port that produced nothing but rare species could not hide here.
      if (p.count < 25 || o.count < 25) { ++n_thin; }
      if (p.count >= 25 && o.count >= 25) {
        const double pm = p.sum_e / static_cast<double>(p.count);
        const double pv = p.sum_e2 / static_cast<double>(p.count) - pm * pm;
        const double om = o.sum_e / static_cast<double>(o.count);
        const double ov = o.sum_e2 / static_cast<double>(o.count) - om * om;
        const double ze = moment_z(pm, (pv > 0.0) ? pv : 0.0, p.count, om,
                                   (ov > 0.0) ? ov : 0.0, o.count);
        ++n_ekin;
        keep_top(top_ekin, ze, c.name + " " + za_label(key) + " n=" +
                               std::to_string(o.count) + " port " + std::to_string(pm) +
                               " g4 " + std::to_string(om) + " MeV");
        if (ze > worst_ekin) {
          worst_ekin = ze;
          worst_ekin_at = c.name + " " + za_label(key) + " port " + std::to_string(pm) +
                          " g4 " + std::to_string(om) + " MeV";
        }
      }
    }
    // A species the port makes and Geant4 never did is not covered by the loop above, and it is
    // the failure mode that matters most - a channel opened by mistake produces nothing on the
    // other side to compare against.
    for (const auto& kv : mine) {
      if (go.find(kv.first) == go.end()) {
        const double z = multiplicity_z(kv.second.count, 0, c.n, tally_var(kv.second, c.n),
                                        -1.0);
        ++n_count;
        if (z > worst_count) {
          worst_count = z;
          worst_count_at = c.name + " " + za_label(kv.first) + " port " +
                           std::to_string(kv.second.count) + " g4 0 (species absent from the "
                           "oracle)";
        }
      }
    }

    // ---- statistical: the mean secondary multiplicity per event
    const double pm = static_cast<double>(n_sec) / nk;
    ++n_mult;
    const double zm = std::fabs(pm - c.mean_mult) /
                      std::sqrt((pm + c.mean_mult) / nk + 1e-300);
    if (zm > worst_mult) {
      worst_mult = zm;
      worst_mult_at = c.name + " port " + std::to_string(pm) + " g4 " +
                      std::to_string(c.mean_mult);
    }
  }

  // -------------------------------------------------------------------------------------------
  std::printf("\n%-30s %8s %12s  %s\n", "bucket", "points", "worst", "where");
  struct Row { const char* name; long long n; double worst; double tol; std::string at; };
  const Row rows[] = {
    {"GateVerdict(cases)", static_cast<long long>(cases.size()),
     static_cast<double>(n_status_bad), 0.0, ""},
    {"CompoundZA(cases)", static_cast<long long>(cases.size()),
     static_cast<double>(n_za_bad), 0.0, ""},
    // The two means, against the oracle, both electron-corrected. The bounds are the residues
    // measured below divided by 5,000 events and multiplied by the electron rate, rounded up to
    // the next decade: a physics difference arrives at 1e-3 and an uncorrected electron at 0.5.
    {"EnergyBalance(MeV)", static_cast<long long>(cases.size()), worst_e, 1e-5, worst_e_at},
    {"MomentumBalance(MeV)", static_cast<long long>(cases.size()), worst_pz, 5e-5, worst_pz_at},
    // Per event, against the port's own compound - no oracle. The first is the sharp one.
    {"BalanceNoIC(MeV/event)", static_cast<long long>(cases.size()), worst_ev_exact, 1e-8,
     worst_ev_exact_at},
    {"BalanceWithIC(MeV/event)", static_cast<long long>(cases.size()), worst_ev_ic, 2e-3,
     worst_ev_ic_at},
    {"SpeciesYield(sigma)", n_count, worst_count, 5.0, worst_count_at},
    {"SpeciesEkin(sigma)", n_ekin, worst_ekin, 5.0, worst_ekin_at},
    {"Multiplicity(sigma)", n_mult, worst_mult, 5.0, worst_mult_at},
  };
  for (const Row& r : rows) {
    const bool bad = r.worst > r.tol;
    if (bad && r.tol > 0.0) { ++fails; }
    std::printf("%-30s %8lld %12.4g  %s%s\n", r.name, r.n, r.worst, bad ? "FAIL " : "",
                r.at.c_str());
  }
  std::printf("  worst residue left by the flat m_e correction: %.3g MeV per conversion "
              "electron\n", worst_pe);
  std::printf("  refused %d, overflowed %d, Poisson variance fallbacks %lld, %lld of %lld "
              "species too thin for a mean test (under 25 a side)\n",
              n_refused, n_overflow, n_poisson_fallback, n_thin, n_count);
  auto by_sigma = [](const std::pair<double, std::string>& a,
                     const std::pair<double, std::string>& b) { return a.first > b.first; };
  std::sort(top_count.begin(), top_count.end(), by_sigma);
  std::sort(top_ekin.begin(), top_ekin.end(), by_sigma);
  std::printf("  worst species yields:\n");
  for (std::size_t i = 0; i < top_count.size() && i < 5; ++i) {
    std::printf("    %6.2f  %s\n", top_count[i].first, top_count[i].second.c_str());
  }
  std::printf("  worst species kinetic energies:\n");
  for (std::size_t i = 0; i < top_ekin.size() && i < 5; ++i) {
    std::printf("    %6.2f  %s\n", top_ekin[i].first, top_ekin[i].second.c_str());
  }
  (void)n_balance_bad;
  std::printf("\ntest_bic_ion: %s\n", (fails == 0) ? "PASS" : "FAIL");
  return (fails == 0) ? 0 : 1;
}
