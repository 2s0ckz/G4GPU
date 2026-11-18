// P29: FTFP's ion arm re-measured with the de-excited residual - the port's side, against
// Geant4's (ref/dump/dump_ion_arm.cc -> ion_arm.csv).
//
// docs/RISK.md V236 (P24) measured the final-state size of the ion arm at the five heavy points
// of tests/test_ftf_model.cu's ion grid - Fe56 on Pb207 at 20 and 8 GeV per nucleon, Fe56 on Fe56,
// O16 and C12 on Pb207 at 20 - and found the port 25-39% short of QBBC's FTFP while the string
// products and the decayed list agreed. P28 (V264-V265) found that no residual of the FTF hand-over
// had ever been de-excited and wired `ftf_deexcite_residual` into it; the ion arm's two residuals
// (`PropagateNuclNucl`'s target and projectile) go through the same function. This test runs
// `ftf::entry::apply` - the call `run_one_model<kFtfp>` makes - WITH the de-excitation, on the same
// five points and event counts, and compares:
//
//   1. V236's table: the maxima of the string products, the decayed list and the final state,
//      and the mean final state, Geant4 against the port (the "after" of V236's last column);
//   2. the per-species rows: per event, the multiplicity and the kinetic energy (`E - m_PDG`, what
//      FillResult hands the transport) of n, p, d, t, He3, alpha, heavier fragments, pi+, pi-,
//      pi0, kaons, eta/eta', gammas, anti-baryons, hyperons and the rest, and the event's energy
//      balance - each with its z from the per-event variance on both sides.
//
// The eta and eta' are compared like any group: at the MODEL level the port makes them (the gap
// of V240 is the transport's - no ParticleType - and lies after this boundary).
//
// `G4GPU_ION_ARM_UNDECAYED=1` runs the hand-over stage alone (the residuals emitted whole, as before
// P28) and asserts nothing: the "before" column.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "core/rng.cuh"
#include "data/level_data.cuh"
#include "host/g4data.cuh"
#include "physics/hadronic/deexcitation/nuclear_masses.cuh"
#include "physics/hadronic/ftf/ftf_entry.cuh"

using namespace g4gpu;
namespace entry = g4gpu::hadronic::ftf::entry;
namespace hp = g4gpu::physics::hadronic;

namespace {

int fails = 0;

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

/// Per-event accumulators: `count2` and `ke2` are sums over events of the event's own count and
/// kinetic energy squared, so a mean's variance is the sample variance over N.
struct GroupAcc {
  long long count = 0, count2 = 0;
  double ke = 0.0, ke2 = 0.0;
};

struct Side {
  int n = 0;        ///< events with a final state
  int n_asked = 0;  ///< events asked for
  std::map<int, long long> h_strings, h_decayed, h_fs;
  std::map<int, GroupAcc> groups;  ///< 0 = every secondary
  GroupAcc balance, excitation;    ///< E_out - E_in; the A > 4 excitation
};

struct Point {
  const char* name;
  int pz, pa;
  double per_a;
  int tz, ta;
};

/// dump_ion_arm.cc's `ion_arm_key`, on the port's secondary: a nucleus is (a > 0) with a nuclear
/// code, no code, or a nucleon's.
int species_key(const hp::HadSecondary<double>& s) {
  const bool nuclear_code = (s.pdg >= 1000000000);
  const bool nucleus = (s.a > 0) && (nuclear_code || s.pdg == 0 || s.pdg == 2212 ||
                                     s.pdg == 2112);
  if (!nucleus) { return s.pdg; }
  if (s.a == 1) { return (s.z == 1) ? 2212 : 2112; }
  if (s.a <= 4) { return 1000000000 + s.z * 10000 + s.a * 10; }
  return 1;
}

/// dump_ion_arm.cc's `ion_arm_group`, the same function.
int ion_arm_group(int key) {
  switch (key) {
    case 2112: return 1;
    case 2212: return 2;
    case 1000010020: return 3;
    case 1000010030: return 4;
    case 1000020030: return 5;
    case 1000020040: return 6;
    case 211: return 8;
    case -211: return 9;
    case 111: return 10;
    case 321: case -321: case 311: case -311: case 310: case 130: return 11;
    case 221: case 331: return 12;
    case 22: return 13;
    case 3122: case 3222: case 3212: case 3112: case 3322: case 3312: case 3334: return 15;
    default: break;
  }
  if (key == 1 || key > 1000000000) { return 7; }
  if (key < -1000) { return 14; }
  return 16;
}

const char* group_name(int g) {
  switch (g) {
    case 0: return "all";
    case 1: return "n";
    case 2: return "p";
    case 3: return "d";
    case 4: return "t";
    case 5: return "He3";
    case 6: return "alpha";
    case 7: return "A>4 frag";
    case 8: return "pi+";
    case 9: return "pi-";
    case 10: return "pi0";
    case 11: return "kaons";
    case 12: return "eta/eta'";
    case 13: return "gamma";
    case 14: return "anti-B";
    case 15: return "hyperons";
    case 16: return "other";
    default: return "?";
  }
}

/// The definition mass FillResult snaps the secondary to (tests/test_ftfp_tape.cu's).
double definition_mass(const hp::HadSecondary<double>& s, int key) {
  if (key == 1 || key > 1000000000) { return deex::nuclear_mass(s.a, s.z); }
  if (key == 2212) { return 938.272013; }
  if (key == 2112) { return 939.56536; }
  const data::FtfHadron* d = data::ftf_find_hadron(s.pdg);
  return (d != nullptr) ? d->mass : s.mass;
}

double mean(const std::map<int, long long>& h, int* n_out = nullptr) {
  long long n = 0;
  double s = 0.0;
  for (const auto& kv : h) {
    if (kv.first < 0) { continue; }
    n += kv.second;
    s += double(kv.first) * kv.second;
  }
  if (n_out != nullptr) { *n_out = static_cast<int>(n); }
  return n ? s / n : 0.0;
}

int top(const std::map<int, long long>& h) { return h.empty() ? 0 : h.rbegin()->first; }

/// z of two per-event means from (sum, sum of squares) over n events each.
double zscore(double sg, double s2g, int ng, double sp, double s2p, int np) {
  const double mg = sg / ng, mp = sp / np;
  const double vg = std::max(s2g / ng - mg * mg, 0.0) / ng;
  const double vp = std::max(s2p / np - mp * mp, 0.0) / np;
  return (mp - mg) / std::sqrt(vg + vp + 1e-30);
}

}  // namespace

int main() {
  const bool undecayed = (std::getenv("G4GPU_ION_ARM_UNDECAYED") != nullptr);
  const Point points[] = {{"Fe56_20000_Pb207", 26, 56, 20000.0, 82, 207},
                          {"Fe56_20000_Fe56", 26, 56, 20000.0, 26, 56},
                          {"Fe56_8000_Pb207", 26, 56, 8000.0, 82, 207},
                          {"O16_20000_Pb207", 8, 16, 20000.0, 82, 207},
                          {"C12_20000_Pb207", 6, 12, 20000.0, 82, 207}};

  // ---- Geant4's side.
  std::map<std::string, Side> g4;
  {
    const std::string path = oracle_dir() + "/ion_arm.csv";
    std::ifstream in(path);
    if (!in) {
      std::printf("FAIL: cannot open %s (G4GPU_DUMP_ONLY=ion_arm ref/dump's g4dump)\n",
                  path.c_str());
      return 1;
    }
    std::string line;
    std::getline(in, line);
    while (std::getline(in, line)) {
      const std::vector<std::string> c = split_csv(line);
      if (c.size() < 13) { continue; }
      Side& s = g4[c[0]];
      s.n_asked = std::atoi(c[6].c_str());
      const std::string& row = c[7];
      const int key = std::atoi(c[8].c_str());
      const long long count = std::atoll(c[9].c_str());
      if (row == "strings") { s.h_strings[key] += count; }
      if (row == "decayed") { s.h_decayed[key] += count; }
      if (row == "fs") { s.h_fs[key] += count; }
      if (row == "group" || row == "event") {
        GroupAcc& a = (row == "group") ? s.groups[key]
                      : (key == 0)     ? s.balance : s.excitation;
        a.count = count;
        a.count2 = std::atoll(c[10].c_str());
        a.ke = std::atof(c[11].c_str());
        a.ke2 = std::atof(c[12].c_str());
      }
    }
    for (auto& kv : g4) { mean(kv.second.h_fs, &kv.second.n); }
  }
  for (const Point& p : points) {
    if (g4.find(p.name) == g4.end() || g4[p.name].n == 0) {
      std::printf("FAIL: ion_arm.csv has no events at %s\n", p.name);
      return 1;
    }
  }
  // The instrument is V236's: same points, order, seeds and counts as dump_capacity, and the
  // accounting draws nothing - so where ftf_capacity.csv is present its histograms must be these.
  {
    std::ifstream in(oracle_dir() + "/ftf_capacity.csv");
    if (in) {
      std::map<std::string, std::map<std::string, std::map<int, long long>>> cap;
      std::string line;
      std::getline(in, line);
      while (std::getline(in, line)) {
        const std::vector<std::string> c = split_csv(line);
        if (c.size() < 10) { continue; }
        cap[c[0]][c[7]][std::atoi(c[8].c_str())] += std::atoll(c[9].c_str());
      }
      long long differ = 0;
      for (const Point& p : points) {
        const Side& s = g4[p.name];
        if (cap[p.name]["strings"] != s.h_strings) { ++differ; }
        if (cap[p.name]["decayed"] != s.h_decayed) { ++differ; }
        if (cap[p.name]["fs"] != s.h_fs) { ++differ; }
      }
      std::printf("ion_arm.csv against ftf_capacity.csv (V236's run): %lld of 15 histograms "
                  "differ\n", differ);
      if (differ != 0) {
        std::printf("FAIL: the instrument is not V236's - its histograms moved\n");
        ++fails;
      }
    } else {
      std::printf("(ftf_capacity.csv not present: V236's histograms not cross-checked)\n");
    }
  }

  // ---- The port's side: `entry::apply`, with the residual de-excitation the transport's FTFP arm
  // hands it (P28), at the slot's sizes.
  const std::string pe = host::g4photon_evaporation_dir();
  if (pe.empty()) {
    std::printf("FAIL: PhotonEvaporation dataset not found (G4LEVELGAMMADATA)\n");
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
  const deex::FermiPool fpool = ps.view();
  std::vector<deex::Fragment> evap(4096), results(1024), step(512);
  std::vector<deex::DeexProduct> deex_products(1024), preco_products(1024);
  entry::ResidualDeexcitation dx;
  dx.levels = &lt;
  dx.fermi = &fpool;
  dx.preco.deex.evap_list = evap.data();
  dx.preco.deex.evap_capacity = static_cast<int>(evap.size());
  dx.preco.deex.results = results.data();
  dx.preco.deex.results_capacity = static_cast<int>(results.size());
  dx.preco.deex.step = step.data();
  dx.preco.deex.step_capacity = static_cast<int>(step.size());
  dx.preco.deex.products = deex_products.data();
  dx.preco.deex.products_capacity = static_cast<int>(deex_products.size());
  dx.preco.products = preco_products.data();
  dx.preco.products_capacity = static_cast<int>(preco_products.size());
  const entry::ResidualDeexcitation* dxp = undecayed ? nullptr : &dx;

  static entry::Workspace slots[1];
  static hadronic::ftf::LundTables<double> lund;
  const entry::Handle<entry::Workspace> h = entry::host_handle(slots, 1, &lund);
  static hp::HadFinalState<double, entry::kMaxSecondaries> out;

  std::map<std::string, Side> port;
  long long refused_total = 0;
  for (const Point& p : points) {
    const Side& gs = g4[p.name];
    Side& ps_ = port[p.name];
    hp::HadProjectile<double> pr;
    pr.pdg = 1000000000 + p.pz * 10000 + p.pa * 10;
    pr.mass = deex::nuclear_mass(p.pa, p.pz);
    pr.charge = p.pz;
    pr.baryon_number = p.pa;
    pr.kin_energy = p.per_a * p.pa;
    hp::HadNucleus nuc;
    nuc.a = p.ta;
    nuc.z = p.tz;
    const double e_in = pr.kin_energy + pr.mass + deex::nuclear_mass(p.ta, p.tz);
    ps_.n_asked = gs.n_asked;
    long long refused = 0;
    for (int ev = 0; ev < gs.n_asked; ++ev) {
      out = hp::HadFinalState<double, entry::kMaxSecondaries>();
      // test_ftf_model's ion-grid stream (Philox(ev, 53)), so the string products and the decayed
      // list of the first 500 events are the ones V236's port column measured.
      Philox<double> rng(static_cast<uint32_t>(ev), 53u);
      entry::Report rep;
      const entry::Status st = entry::apply(h, 0, pr, nuc, out, rep, rng, dxp);
      ++ps_.h_strings[slots[0].strings.n_out];
      ++ps_.h_decayed[slots[0].n_tracks];
      if (st != entry::Status::kRan) {
        ++refused;
        ++ps_.h_fs[-1];
        continue;
      }
      ++ps_.h_fs[out.n_secondaries];
      ++ps_.n;
      std::map<int, GroupAcc> evg;
      double e_out = 0.0, ex = 0.0;
      for (int i = 0; i < out.n_secondaries; ++i) {
        const hp::HadSecondary<double>& s = out.secondaries[i];
        const int key = species_key(s);
        const double m = definition_mass(s, key);
        const double e = s.kin_energy + s.mass;
        e_out += e;
        if (key == 1) { ex += s.mass - m; }
        GroupAcc& g = evg[ion_arm_group(key)];
        ++g.count;
        g.ke += e - m;
      }
      double ke_all = 0.0;
      for (const auto& kv : evg) {
        GroupAcc& g = ps_.groups[kv.first];
        g.count += kv.second.count;
        g.count2 += kv.second.count * kv.second.count;
        g.ke += kv.second.ke;
        g.ke2 += kv.second.ke * kv.second.ke;
        ke_all += kv.second.ke;
      }
      GroupAcc& all = ps_.groups[0];
      all.count += out.n_secondaries;
      all.count2 += static_cast<long long>(out.n_secondaries) * out.n_secondaries;
      all.ke += ke_all;
      all.ke2 += ke_all * ke_all;
      const double de = e_out - e_in;
      ps_.balance.ke += de;
      ps_.balance.ke2 += de * de;
      ps_.excitation.ke += ex;
      ps_.excitation.ke2 += ex * ex;
    }
    refused_total += refused;
    if (refused != 0 && !undecayed) {
      std::printf("FAIL: %s: %lld of %d events refused\n", p.name, refused, gs.n_asked);
      ++fails;
    }
  }

  // ---- 1. V236's table, after.
  std::printf("\nV236's table%s: maxima of the string products, the decayed list and the final "
              "state, and the mean final state\n", undecayed ? " [residuals UNDECAYED]" : "");
  std::printf("  %-18s  G4: N strings tracks   fs  <fs>  |  port: N strings tracks   fs  <fs>"
              "    z(<fs>)\n", "point");
  for (const Point& p : points) {
    const Side& g = g4[p.name];
    const Side& q = port[p.name];
    const GroupAcc& ga = g.groups.count(0) ? g.groups.at(0) : GroupAcc{};
    const GroupAcc& pa = q.groups.count(0) ? q.groups.at(0) : GroupAcc{};
    const double z = zscore(double(ga.count), double(ga.count2), g.n, double(pa.count),
                            double(pa.count2), std::max(q.n, 1));
    std::printf("  %-18s %6d %7d %6d %4d %5.1f | %7d %7d %6d %4d %5.1f   %+6.2f\n", p.name, g.n,
                top(g.h_strings), top(g.h_decayed), top(g.h_fs), mean(g.h_fs), q.n,
                top(q.h_strings), top(q.h_decayed), top(q.h_fs), mean(q.h_fs), z);
  }
  // The transport's capacities still cover both sides' largest event (V236's assertion).
  {
    int need_t = 0, need_fs = 0;
    for (const Point& p : points) {
      for (const Side* s : {&g4[p.name], &port[p.name]}) {
        need_t = std::max({need_t, top(s->h_strings), top(s->h_decayed)});
        need_fs = std::max(need_fs, top(s->h_fs));
      }
    }
    std::printf("  capacities: %d tracks, %d secondaries; needed %d and %d\n",
                entry::kIonMaxTracks, entry::kMaxSecondaries, need_t, need_fs);
    if (need_t > entry::kIonMaxTracks || need_fs > entry::kMaxSecondaries) {
      std::printf("FAIL: a capacity is under what an event needed\n");
      ++fails;
    }
  }

  // ---- 2. The per-species rows.
  long long n_compared = 0, n_over = 0;
  double worst_z = 0.0;
  std::string worst_where;
  std::vector<std::string> over;
  for (const Point& p : points) {
    const Side& g = g4[p.name];
    const Side& q = port[p.name];
    const int ng = g.n, np = std::max(q.n, 1);
    std::printf("\n%s   G4 %d events, port %d\n", p.name, g.n, q.n);
    std::printf("  %-9s %9s %9s %7s   %11s %11s %7s\n", "group", "G4 n/ev", "port n/ev", "z",
                "G4 KE/ev", "port KE/ev", "z");
    std::map<int, int> keys;
    for (const auto& kv : g.groups) { keys[kv.first] = 1; }
    for (const auto& kv : q.groups) { keys[kv.first] = 1; }
    auto row = [&](const char* nm, const GroupAcc& ga, const GroupAcc& pa, bool mult,
                   bool counted) {
      const double zm = mult ? zscore(double(ga.count), double(ga.count2), ng, double(pa.count),
                                      double(pa.count2), np)
                             : 0.0;
      const double ze = zscore(ga.ke, ga.ke2, ng, pa.ke, pa.ke2, np);
      const bool bad = counted && (std::fabs(zm) > 5.0 || std::fabs(ze) > 5.0);
      if (mult) {
        std::printf("  %-9s %9.3f %9.3f %7.2f   %11.2f %11.2f %7.2f%s\n", nm,
                    double(ga.count) / ng, double(pa.count) / np, zm, ga.ke / ng, pa.ke / np, ze,
                    bad ? "  <<<" : "");
      } else {
        std::printf("  %-9s %9s %9s %7s   %11.2f %11.2f %7.2f%s\n", nm, "", "", "", ga.ke / ng,
                    pa.ke / np, ze, bad ? "  <<<" : "");
      }
      if (!counted) { return; }
      const double zs[2] = {zm, ze};
      const char* what[2] = {"n/ev", "KE/ev"};
      for (int k = mult ? 0 : 1; k < 2; ++k) {
        const double z = zs[k];
        ++n_compared;
        if (std::fabs(z) > std::fabs(worst_z)) {
          worst_z = z;
          worst_where = std::string(p.name) + " " + nm + " " + what[k];
        }
        if (std::fabs(z) > 5.0) {
          ++n_over;
          char b[160];
          std::snprintf(b, sizeof b, "%s %s (%s z %.2f)", p.name, nm, what[k], z);
          over.push_back(b);
        }
      }
    };
    for (const auto& kk : keys) {
      const GroupAcc ga = g.groups.count(kk.first) ? g.groups.at(kk.first) : GroupAcc{};
      const GroupAcc pa = q.groups.count(kk.first) ? q.groups.at(kk.first) : GroupAcc{};
      // Every group with twenty or more particles on either side, as test_ftfp_tape gates.
      row(group_name(kk.first), ga, pa, true, ga.count + pa.count >= 20);
    }
    row("E balance", g.balance, q.balance, false, true);
    // The residual-class excitation is printed and not gated: it is zero in Geant4 up to the
    // long-lived levels the handler leaves (V265), so its error is no scale for a difference.
    row("A>4 E*", g.excitation, q.excitation, false, false);
  }
  std::printf("\n%lld comparisons (per event: multiplicity and kinetic energy of every group with "
              "20 or more particles, the final state's size, the energy balance), %lld past 5 "
              "sigma, worst %.2f at %s; %lld refused%s\n", n_compared, n_over, worst_z,
              worst_where.c_str(), refused_total, undecayed ? "  [residuals UNDECAYED]" : "");
  for (const std::string& s : over) { std::printf("  past 5 sigma: %s\n", s.c_str()); }
  if (!undecayed && n_over > 0) {
    std::printf("FAIL: %lld comparisons past 5 sigma\n", n_over);
    ++fails;
  }
  if (undecayed) {
    std::printf("(the hand-over stage alone: nothing asserted)\n");
    return 0;
  }
  std::printf("%s\n", fails == 0 ? "PASS test_ion_arm" : "FAIL test_ion_arm");
  return fails == 0 ? 0 : 1;
}
