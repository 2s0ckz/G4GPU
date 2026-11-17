// P28: the FTFP excess - the port's side of the per-interaction tape, against Geant4's.
//
// `ref/dump/dump_ftfp_tape.cc` runs QBBC's own G4TheoFSGenerator("FTFP") instances - in flight
// and `hFritiofCaptureAtRest`'s - on the projectiles and targets the three FTFP beams of the B1
// sweep reach, and writes per (case, species) the multiplicity and the summed kinetic energy
// `E - m_PDG` the transport receives. This file runs `ftf::entry::apply`, the function both of the
// port's FTFP arms call (the in-flight `run_arm_ftfp` and the at-rest `fritiof_at_rest_wide`), on
// the same rows and compares, species by species, with the event totals - the energy balance
// `E_out - E_in` - first.
//
// The comparison is statistical: per-event means with their errors from the sums of squares, a
// z-score for each, the two samples independent.
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

struct Acc {
  long long count = 0;
  double sum = 0.0, sum2 = 0.0;
};

struct Case {
  int id = 0, pdg = 0, tz = 0, ta = 0, n = 0;
  double t = 0.0;
  std::map<int, Acc> sp;  ///< species -> accumulators; 0 = event totals, -1 = residual E*
};

/// dump_ftfp_tape.cc's `species_key`, the same rule on the port's secondary.
int species_key(const hp::HadSecondary<double>& s) {
  const bool nuclear_code = (s.pdg >= 1000000000);
  const bool nucleus = (s.a > 0) && (nuclear_code || s.pdg == 0 || s.pdg == 2212 ||
                                     s.pdg == 2112);
  if (!nucleus) { return s.pdg; }
  if (s.a == 1) { return (s.z == 1) ? 2212 : 2112; }
  if (s.a <= 4) { return 1000000000 + s.z * 10000 + s.a * 10; }
  return 1;
}

/// The definition mass `G4HadronicProcess::FillResult` snaps the secondary to - the port's
/// `definition_mass_of` - so that `E - m` is the kinetic energy the transport receives.
double definition_mass(const hp::HadSecondary<double>& s, int key) {
  if (key == 1 || key > 1000000000) { return deex::nuclear_mass(s.a, s.z); }
  if (key == 2212) { return 938.272013; }
  if (key == 2112) { return 939.56536; }
  const data::FtfHadron* d = data::ftf_find_hadron(s.pdg);
  return (d != nullptr) ? d->mass : s.mass;
}

const char* species_name(int k) {
  switch (k) {
    case 0: return "TOTAL(dE)";
    case -1: return "resid E*";
    case 1: return "A>4 frag";
    case 22: return "gamma";
    case 11: return "e-";
    case 211: return "pi+";
    case -211: return "pi-";
    case 111: return "pi0";
    case 221: return "eta";
    case 331: return "eta'";
    case 2212: return "p";
    case 2112: return "n";
    case -2212: return "pbar";
    case -2112: return "nbar";
    case 321: return "K+";
    case -321: return "K-";
    case 311: return "K0";
    case -311: return "K0bar";
    case 310: return "K0S";
    case 130: return "K0L";
    case 3122: return "Lambda";
    case 1000010020: return "d";
    case 1000010030: return "t";
    case 1000020030: return "He3";
    case 1000020040: return "alpha";
    default: return nullptr;
  }
}

}  // namespace

int main(int argc, char** argv) {
  const char* env_n = std::getenv("G4GPU_FTFP_TAPE_N");
  const int n_events = (env_n != nullptr) ? std::atoi(env_n) : 5000;
  const bool verbose = (argc > 1 && std::string(argv[1]) == "-v");

  // ---- Geant4's side.
  std::map<int, Case> g4;
  {
    const std::string path = oracle_dir() + "/ftfp_tape.csv";
    std::ifstream in(path);
    if (!in) {
      std::printf("FAIL: cannot open %s\n", path.c_str());
      return 1;
    }
    std::string line;
    std::getline(in, line);
    while (std::getline(in, line)) {
      const std::vector<std::string> c = split_csv(line);
      if (c.size() < 10) { continue; }
      Case& k = g4[std::atoi(c[0].c_str())];
      k.id = std::atoi(c[0].c_str());
      k.pdg = std::atoi(c[1].c_str());
      k.t = std::atof(c[2].c_str());
      k.tz = std::atoi(c[3].c_str());
      k.ta = std::atoi(c[4].c_str());
      k.n = std::atoi(c[5].c_str());
      Acc& a = k.sp[std::atoi(c[6].c_str())];
      a.count = std::atoll(c[7].c_str());
      a.sum = std::atof(c[8].c_str());
      a.sum2 = std::atof(c[9].c_str());
    }
  }
  if (g4.empty()) {
    std::printf("FAIL: ftfp_tape.csv has no rows\n");
    return 1;
  }


  // ---- The port's side, through the contract header exactly as both FTFP arms call it.
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
  // The slot's PreCompound buffers, at tests/test_bertini_apply.cu's sizes.
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
  // `G4GPU_FTFP_TAPE_UNDECAYED=1` runs the hand-over as it was before P28 - the residual emitted
  // whole - and asserts nothing: it is the "before" column of docs/RISK.md V264.
  const bool undecayed = (std::getenv("G4GPU_FTFP_TAPE_UNDECAYED") != nullptr);
  const entry::ResidualDeexcitation* dxp = undecayed ? nullptr : &dx;

  const int kSlots = 1;
  static entry::Workspace slots[kSlots];
  static hadronic::ftf::LundTables<double> lund;
  const entry::Handle<entry::Workspace> h = entry::host_handle(slots, kSlots, &lund);
  static hp::HadFinalState<double, 512> out;

  long long n_compared = 0, n_over = 0;
  double worst_z = 0.0;
  std::string worst_where;
  long long refused_total = 0;
  for (auto& kv : g4) {
    const Case& gc = kv.second;
    const data::FtfHadron* d = data::ftf_find_hadron(gc.pdg);
    hp::HadProjectile<double> pr;
    pr.pdg = gc.pdg;
    pr.mass = d->mass;
    pr.charge = static_cast<int>(d->charge);
    pr.baryon_number = d->baryon;
    pr.kin_energy = gc.t;
    hp::HadNucleus nuc;
    nuc.a = gc.ta;
    nuc.z = gc.tz;
    const double e_in = gc.t + d->mass + deex::nuclear_mass(gc.ta, gc.tz);

    Case pc;
    long long refused = 0, unchanged = 0;
    for (int ev = 0; ev < n_events; ++ev) {
      out = hp::HadFinalState<double, 512>();
      Philox<double> rng(static_cast<uint32_t>(ev), 2800u + static_cast<uint32_t>(gc.id));
      entry::Report rep;
      const entry::Status st = entry::apply(h, 0, pr, nuc, out, rep, rng, dxp);
      if (st == entry::Status::kRefused || st == entry::Status::kNoWorkspaceSlot) {
        ++refused;
        continue;
      }
      if (st == entry::Status::kPrimaryUnchanged) { ++unchanged; }
      ++pc.n;
      double e_out = 0.0, ex = 0.0;
      for (int i = 0; i < out.n_secondaries; ++i) {
        const hp::HadSecondary<double>& s = out.secondaries[i];
        const int key = species_key(s);
        const double m = definition_mass(s, key);
        const double e = s.kin_energy + s.mass;
        const double ke = e - m;
        e_out += e;
        if (key == 1) { ex += s.mass - m; }
        Acc& a = pc.sp[key];
        ++a.count;
        a.sum += ke;
        a.sum2 += ke * ke;
      }
      Acc& t = pc.sp[0];
      t.count += out.n_secondaries;
      const double de = e_out - e_in;
      t.sum += de;
      t.sum2 += de * de;
      Acc& x = pc.sp[-1];
      x.sum += ex;
      x.sum2 += ex * ex;
    }
    refused_total += refused;

    std::printf("\ncase %2d: pdg %d T %.0f MeV on (Z=%d,A=%d)   G4 %d events, port %d "
                "(refused %lld, unchanged %lld)%s\n", gc.id, gc.pdg, gc.t, gc.tz, gc.ta, gc.n,
                pc.n, refused, unchanged, undecayed ? "  [residual UNDECAYED]" : "");
    std::printf("  %-10s %9s %9s %6s   %11s %11s %6s\n", "species", "G4 n/ev", "port n/ev",
                "z", "G4 KE/ev", "port KE/ev", "z");
    std::map<int, int> keys;
    for (const auto& s : gc.sp) { keys[s.first] = 1; }
    for (const auto& s : pc.sp) { keys[s.first] = 1; }
    double ke_g4_sum = 0.0, ke_port_sum = 0.0;
    for (const auto& kk : keys) {
      const int key = kk.first;
      const Acc ga = gc.sp.count(key) ? gc.sp.at(key) : Acc{};
      const Acc pa = pc.sp.count(key) ? pc.sp.at(key) : Acc{};
      const double ng = gc.n, np = (pc.n > 0) ? pc.n : 1;
      const double mg = ga.count / ng, mp = pa.count / np;
      const double eg = ga.sum / ng, ep = pa.sum / np;
      // Errors. The event-total and E* rows are one number an event: the variance of the mean is
      // the sample variance over N. A species row is a sum over a Poisson-like number of
      // particles an event, whose variance is E[n] E[ke^2] - the compound-Poisson estimate,
      // sum(ke^2)/N per event - and whose count has variance ~ its mean.
      //
      // The two per-event rows are keys 0 and -1; every other key is a species, pi- (-211) and
      // the anti-nucleons included.
      const bool event_row = (key == 0 || key == -1);
      double ze = 0.0, zm = 0.0;
      if (event_row) {
        const double vg = std::max(ga.sum2 / ng - eg * eg, 0.0) / ng;
        const double vp = std::max(pa.sum2 / np - ep * ep, 0.0) / np;
        ze = (ep - eg) / std::sqrt(vg + vp + 1e-30);
      } else {
        ze = (ep - eg) / std::sqrt(ga.sum2 / (ng * ng) + pa.sum2 / (np * np) + 1e-30);
        zm = (mp - mg) / std::sqrt(ga.count / (ng * ng) + pa.count / (np * np) + 1e-30);
      }
      if (!event_row) {
        ke_g4_sum += eg;
        ke_port_sum += ep;
      }
      // THE GATE: every species with twenty or more particles on either side, and the two
      // per-event rows, in multiplicity and kinetic energy. -1 (the residual-class E*) is asserted
      // as an absolute difference instead: it is zero in Geant4 up to the long-lived levels the
      // handler leaves, so its error is no scale for a difference.
      const bool counted = (!event_row && ga.count + pa.count >= 20) || key == 0;
      if (!undecayed && counted) {
        n_compared += 2;
        for (double z : {zm, ze}) {
          if (std::fabs(z) > std::fabs(worst_z)) {
            worst_z = z;
            worst_where = "case " + std::to_string(gc.id) + " species " + std::to_string(key);
          }
          if (std::fabs(z) > 5.0) { ++n_over; }
        }
      }
      if (!undecayed && key == -1 && std::fabs(ep - eg) > 0.05 + 0.1 * std::fabs(eg)) {
        std::printf("FAIL: case %d: the residual-class excitation per event is %.3f MeV in the "
                    "port and %.3f in Geant4\n", gc.id, ep, eg);
        ++fails;
      }
      const char* nm = species_name(key);
      if (!verbose && nm == nullptr && std::fabs(ze) < 3.0 && std::fabs(zm) < 3.0 && mg < 0.05 &&
          mp < 0.05) {
        continue;
      }
      char buf[32];
      if (nm == nullptr) {
        std::snprintf(buf, sizeof buf, "%d", key);
        nm = buf;
      }
      std::printf("  %-10s %9.4f %9.4f %6.1f   %11.3f %11.3f %6.1f%s\n", nm, mg, mp, zm, eg, ep,
                  ze, (std::fabs(ze) > 5.0 || std::fabs(zm) > 5.0) ? "  <<<" : "");
    }
    std::printf("  %-10s %9s %9s %6s   %11.3f %11.3f\n", "sum KE", "", "", "", ke_g4_sum,
                ke_port_sum);
    if (!undecayed && refused > 0) {
      std::printf("FAIL: case %d: %lld of %d events refused\n", gc.id, refused, n_events);
      ++fails;
    }
  }
  std::printf("\n%lld comparisons (multiplicity and kinetic energy of every species with 20 or "
              "more particles, and the event's energy balance), %lld past 5 sigma, worst %.2f "
              "at %s; %lld refused\n", n_compared, n_over, worst_z, worst_where.c_str(),
              refused_total);
  // One in a thousand past 5 sigma would be chance at ~1e-3 of the time for 1,000 comparisons;
  // the gate allows none past 5 and the run shows how close the worst one is.
  if (!undecayed && n_over > 0) {
    std::printf("FAIL: %lld comparisons past 5 sigma\n", n_over);
    ++fails;
  }
  std::printf("%s\n", fails == 0 ? "PASS test_ftfp_tape" : "FAIL test_ftfp_tape");
  return fails == 0 ? 0 : 1;
}
