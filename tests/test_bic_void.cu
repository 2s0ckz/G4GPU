// P20: `G4BinaryCascade::FillVoidNucleusProducts` - the destroyed-nucleus branch - against
// ref/oracle/bic_void*.csv.
//
// `bic/cascade_void.cuh` is reached from `propagate` exactly where `G4BinaryCascade::Propagate`
// reaches `FillVoidNucleusProducts` (G4BinaryCascade.cc:519): after the collision loop, when the
// target list has no proton left in it. Both entry points get there - `bic::apply_yourself` for a
// nucleon or a pion, `bic::blir_apply_yourself` for an ion - and P9d refused the branch on both,
// so this checks both.
//
// **The tape: every uniform and every product, against Geant4's own stream.** The branch is
// private and is reached only when a cascade has destroyed the nucleus, so there is no way to
// call it: `ref/dump/dump_bic.cc`'s `write_bic_void` runs whole events of the PUBLIC
// `ApplyYourself`s under a HepJamesRandom that records every value it serves, and keeps the
// first twenty per case whose successful `Propagate` went out through the branch - which two
// private members say, read through an explicit instantiation (the dump's `VoidPeek`). The port
// replays each kept tape from before the first `G4Fancy3DNucleus::Init` and must
//
//   * consume exactly the recorded number of uniforms and never read past the end - so every
//     retry of both loops, the whole cascade, the two decay passes of the branch, the decays its
//     late-particle loop draws and discards, and its one `(0.1 + 5U) MeV` draw are all in step;
//   * go out through the branch itself (`kPropagateVoidNucleus`), with the same list sizes the
//     dump read off Geant4's cascade afterwards, the same `theMomentumTransfer`, the same
//     projectile four-momentum and initial mass and the same (currentA, currentZ);
//   * produce the same secondaries in the same order, with the same definitions, creator ids and
//     parent resonances, and the same four-momenta, compared relative to each product's own
//     energy;
//   * on the ion path, return from `Propagate` the same products the branch returned there - its
//     OWN output, before `G4BinaryLightIonReaction` sorted, boosted and corrected it, which the
//     dump reads through a `G4BinaryCascade` whose virtual `Propagate` keeps a copy
//     (`VoidRecordingCascade`) - in the same order and with the same `NewlyAdded`;
//   * draw as many scheduled decays in the collision drain as Geant4 did, which the dump counts
//     with a `G4BCDecay` of its own in place of the cascade's (`VoidDrainCountingDecay`). None of
//     the 160 events of the eight cases does; two DRAIN cases keep the eleven that 345,457 ion
//     reactions on hydrogen hold;
//   * walk theCapturedList in CAPTURE order - which changes an answer only where it is not the
//     order the tracks were made in, on an event with two or more captured nucleons: two
//     CAPTURED cases keep twenty of those, and one of them is out of order.
//
// **And the energy, per event, against the branch's own arithmetic.** On the nucleon path the
// branch is the whole of what happens after the cascade, and it conserves energy EXACTLY unless
// it refuses its own correction (docs/RISK.md V213): the event is then short by
// `Ekinetic - Ekineticrdm`, which `VoidReport` carries. The assertion is that the deficit EQUALS
// that - zero on the other two branches - to the rounding of the sum, which says the port loses
// what Geant4 loses and nothing else.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "data/level_data.cuh"
#include "host/g4data.cuh"
#include "physics/hadronic/bic/binary_cascade.cuh"
#include "physics/hadronic/bic/light_ion_reaction.cuh"

using namespace g4gpu;

namespace {

int fails = 0;

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

/// Exact: one mismatch sets the bucket's worst to 1 against a tolerance of zero.
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

/// Relative to `scale`, a product's own total energy - see tests/test_bic_1h1.cu.
void cmp_scaled(int bi, double got, double want, double scale, const std::string& where) {
  Bucket& b = buckets[bi];
  ++b.n;
  const double d = std::fabs(got - want) / ((scale > 0.0) ? scale : 1.0);
  if (d > b.worst) {
    b.worst = d;
    b.where = where;
  }
}

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
std::string sv(const std::vector<std::string>& f, std::size_t i) {
  return (i < f.size()) ? f[i] : std::string();
}

/// A RECORDED random stream, replayed value by value, as in tests/test_bic_apply.cu. Sized for
/// the longest tape the branch's cases record - a C12 on C12 at 1 GeV per nucleon spends tens of
/// thousands of uniforms before the nucleus is gone.
inline constexpr int kTapeMax = 1 << 18;
struct TapeRng {
  double value[kTapeMax] = {};
  int n_values = 0;
  int n = 0;
  int overrun = 0;
  __host__ __device__ double uniform() {
    if (n >= n_values || n >= kTapeMax) {
      const double v = (2.0 * static_cast<double>(overrun % 64) + 1.0) / 128.0;
      ++overrun;
      return v;
    }
    return value[n++];
  }
};

/// P6's buffers, tests/test_bic_apply.cu's shape.
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
  Buffers bufs;

  const int b_sel = new_bucket("VoidTapeSelection", 0.0);
  const int b_str = new_bucket("VoidTapeStructure", 0.0);
  const int b_lst = new_bucket("VoidTapeLists", 0.0);
  // theMomentumTransfer, the projectile four-momentum and the initial mass, relative to the
  // projectile's energy: the transfer is a sum over every time step of the cascade, and a port
  // whose propagation is one ulp out shows it here first.
  const int b_lsm = new_bucket("VoidTapeListMomenta", 1e-12);
  // P9e's tolerance for the ion tape (tests/test_bic_apply.cu, `IonTapeMomenta`), relative to
  // each secondary's own total energy; the measured worst is printed.
  const int b_mom = new_bucket("VoidTapeMomenta(nucleon)", 1e-9);
  const int b_imom = new_bucket("VoidTapeMomenta(ion)", 1e-9);
  // The ion path's BRANCH OUTPUT - what `Propagate` returned to `Interact`, before the light-ion
  // reaction sorted, boosted and corrected it (the dump's `VoidRecordingCascade`) - relative to
  // each product's own energy, and its order, species, `NewlyAdded` and creator exactly.
  const int b_bout = new_bucket("VoidBranchOutput(ion)", 1e-9);
  const int b_bstr = new_bucket("VoidBranchOutputStructure", 0.0);
  // The nucleon path's energy: the event's deficit against the branch's own residue, MeV.
  const int b_nrg = new_bucket("VoidEnergyResidue(MeV)", 1e-6);

  // -------------------------------------------------------------------------------------------
  // The selection: twenty kept events per case, and the rate it took to find them.
  // -------------------------------------------------------------------------------------------
  const auto cases = read_csv("bic_void_tapecases.csv");
  if (cases.empty()) {
    std::printf("bic_void_tapecases.csv is empty\n");
    ++fails;
  }
  for (const auto& c : cases) {
    // Twenty per case. The two DRAIN cases keep only events whose branch drew a scheduled decay in
    // its collision drain - the dump's `VoidDrainCountingDecay` - and those are rare enough that
    // the dump keeps all it finds in a fixed number of events: what it wanted is in the file,
    // and it must have found at least one, or a broken counter would pass as a rare event.
    // The CAPTURED cases keep only events that left two or more nucleons in theCapturedList -
    // ten of each - and are asserted below to reach a list whose capture order is not the pool's.
    const bool drain_case = sv(c, 1).rfind("void_drain_", 0) == 0;
    const bool captured_case = sv(c, 1).rfind("void_captured_", 0) == 0;
    cmp_int(b_sel, iv(c, 3), iv(c, 4), sv(c, 1) + " events kept");
    if (drain_case) {
      cmp_int(b_sel, iv(c, 3) > 0 ? 1 : 0, 1, sv(c, 1) + " found a drained decay");
    } else if (captured_case) {
      cmp_int(b_sel, iv(c, 4), 10, sv(c, 1) + " events wanted");
    } else {
      cmp_int(b_sel, iv(c, 4), 20, sv(c, 1) + " events wanted");
    }
    std::printf("  %-8s %-24s %2d kept of %7d events run (%.3g%% %s)\n", sv(c, 0).c_str(),
                sv(c, 1).c_str(), iv(c, 3), iv(c, 2),
                (iv(c, 2) > 0) ? 100.0 * iv(c, 3) / iv(c, 2) : 0.0,
                drain_case      ? "through the branch AND its drain"
                : captured_case ? "through the branch with two captured"
                                : "through the branch");
  }

  const auto rows = read_csv("bic_void_tape.csv");
  const auto tfs = read_csv("bic_void_tapefs.csv");
  std::map<std::string, std::vector<const std::vector<std::string>*>> fs_of;
  for (const auto& fr : tfs) { fs_of[sv(fr, 0) + "#" + sv(fr, 1)].push_back(&fr); }
  const auto tpr = read_csv("bic_void_tapepr.csv");
  std::map<std::string, std::vector<const std::vector<std::string>*>> pr_of;
  for (const auto& pr : tpr) { pr_of[sv(pr, 0) + "#" + sv(pr, 1)].push_back(&pr); }
  if (tpr.empty()) {
    std::printf("bic_void_tapepr.csv is empty\n");
    ++fails;
  }
  // The tapes are millions of rows - a C12 on C12 at 1 GeV per nucleon spends tens of thousands
  // of uniforms an event - so they are read straight into one array of doubles per (case, ev)
  // rather than into rows of strings, which would cost a gigabyte for 40 MB of numbers.
  std::map<std::string, std::vector<double>> vals_of;
  {
    const std::string path = oracle_dir() + "/bic_void_tapeval.csv";
    FILE* f = std::fopen(path.c_str(), "r");
    if (f == nullptr) {
      std::printf("MISSING %s\n", path.c_str());
      ++fails;
    } else {
      char line[512];
      bool header = true;
      while (std::fgets(line, sizeof line, f) != nullptr) {
        if (header) { header = false; continue; }
        // case,ev,i,u
        char* c1 = std::strchr(line, ',');
        if (c1 == nullptr) { continue; }
        char* c2 = std::strchr(c1 + 1, ',');
        if (c2 == nullptr) { continue; }
        char* c3 = std::strchr(c2 + 1, ',');
        if (c3 == nullptr) { continue; }
        const std::string key = std::string(line, c1) + "#" + std::string(c1 + 1, c2);
        const long i = std::strtol(c2 + 1, nullptr, 10);
        std::vector<double>& v = vals_of[key];
        if (i < 0) { continue; }
        const std::size_t at = static_cast<std::size_t>(i);
        if (at >= v.size()) { v.resize(at + 1, 0.0); }
        v[at] = std::strtod(c3 + 1, nullptr);
      }
      std::fclose(f);
    }
  }

  // Caller-owned storage, tests/test_bic_apply.cu's envelopes.
  static bic::Nucleon nuc[256];
  static bic::Nucleon pnuc[256];
  static deex::Vec3d mom[256];
  static double fermi[256];
  static bic::NucleusSortEntry sums[256];
  static double flat[bic::kFlatBlock];
  static double pfield[bic::kMaxFieldTable];
  static double nfield[bic::kMaxFieldTable];
  static bic::CascadeTrack cpool[1024];
  static bic::imr::CollisionInitialState colls[8192];
  static bic::imr::ConcreteChannel chans[bic::imr::kConcreteChannelCount];
  static bic::CascadeBuffers buffers;
  static bic::CascadeProduct products[512];
  static bic::CascadeProduct preco[256];
  static bic::BlirProduct spec[512];
  static bic::BlirProduct casc[512];
  static bic::CascadeTrack initial[256];
  const int n_chan = bic::imr::build_concrete_channels(chans, bic::imr::kConcreteChannelCount);
  auto cascade_ws = [&]() {
    bic::CascadeWorkspace ws;
    ws.pool = cpool;
    ws.pool_capacity = 1024;
    ws.collisions = colls;
    ws.collision_capacity = 8192;
    ws.channels = chans;
    ws.n_channels = n_chan;
    ws.buffers = &buffers;
    ws.products = products;
    ws.product_capacity = 512;
    ws.preco_products = preco;
    ws.preco_capacity = 256;
    return ws;
  };

  long long n_events = 0, n_nucleon = 0, n_ion = 0;
  long long branch_count[3] = {0, 0, 0};
  long long n_empty_targets = 0, n_late_decays = 0, n_lates = 0;
  /// Events whose theCapturedList, walked in capture order, is not in pool order - the only ones
  /// on which that order changes an answer. See `VoidReport::captured_reordered`.
  long long n_reordered = 0;
  std::map<std::string, long long> reordered_of;
  int worst_loops = 0;
  double worst_left = 0.0;
  static TapeRng tape;
  for (const auto& r : rows) {
    const std::string path = sv(r, 0);
    const std::string cname = sv(r, 1);
    const int ev = iv(r, 2);
    const int pdg = iv(r, 4);
    const int pz = iv(r, 5), pa = iv(r, 6);
    const double ekin = dv(r, 7);
    const int tz = iv(r, 8), ta = iv(r, 9);
    const int ndraws = iv(r, 10);
    const int want_nsec = iv(r, 12);
    const std::string where = cname + " ev " + std::to_string(ev);
    const bool ion = (path == "ion");
    ++n_events;
    if (ion) { ++n_ion; } else { ++n_nucleon; }

    tape.n_values = 0;
    {
      const std::vector<double>& v = vals_of[cname + "#" + std::to_string(ev)];
      const int nv = static_cast<int>(std::min<std::size_t>(v.size(), kTapeMax));
      for (int i = 0; i < nv; ++i) { tape.value[i] = v[static_cast<std::size_t>(i)]; }
      tape.n_values = nv;
      if (v.size() > static_cast<std::size_t>(kTapeMax)) {
        std::printf("TAPE %s: %zu values, past this file's %d\n", where.c_str(), v.size(),
                    kTapeMax);
        ++fails;
      }
    }
    cmp_int(b_str, tape.n_values, ndraws, where + " tape length");
    tape.n = 0;
    tape.overrun = 0;
    buffers = bic::CascadeBuffers{};
    preco::PrecoWorkspace pws = bufs.view();

    physics::hadronic::HadNucleus tgt;
    tgt.z = tz;
    tgt.a = ta;
    tgt.l = 0;
    physics::hadronic::HadProjectile<double> proj;
    proj.pdg = pdg;
    proj.charge = static_cast<double>(pz);

    std::string diag;
    // The port's answer, in one shape for both entry points.
    const physics::hadronic::HadSecondary<double>* secs = nullptr;
    int n_secs = 0;
    bool killed = false;
    bool refused = false;
    bic::PropagateResult pr;
    static bic::BicFinalState nfs;
    static bic::BlirFinalState ifs;
    if (ion) {
      proj.baryon_number = pa;
      proj.mass = deex::nuclear_mass(pa, pz);
      proj.kin_energy = ekin * pa;
      bic::BlirStorage store;
      store.projectile_nucleons = pnuc;
      store.target_nucleons = nuc;
      store.scratch.momentum = mom;
      store.scratch.fermi_p = fermi;
      store.scratch.test_sums = sums;
      store.scratch.flat_block = flat;
      store.scratch.capacity = 256;
      store.proton_field = pfield;
      store.neutron_field = nfield;
      store.field_capacity = bic::kMaxFieldTable;
      store.cascade = cascade_ws();
      store.spectators = spec;
      store.spectator_capacity = 512;
      store.cascaders = casc;
      store.cascader_capacity = 512;
      store.initial = initial;
      store.initial_capacity = 256;
      bic::BlirRefusal bref;
      bic::BlirReport brep;
      bic::blir_apply_yourself(proj, tgt, lt, pool, pws, store, tape, ifs, bref, brep);
      refused = bref.any();
      if (refused) {
        std::printf("REFUSED %s: cascade=%d nucleus=%d nofs=%d mom=%d cap=%d decay_null=%d\n",
                    where.c_str(), bref.cascade ? 1 : 0, bref.nucleus ? 1 : 0,
                    bref.no_final_state ? 1 : 0, bref.momentum_not_conserved ? 1 : 0,
                    bref.capacity ? 1 : 0, bref.cascade_ref.void_decay_null ? 1 : 0);
      }
      pr = brep.propagate;
      diag = " loops " + std::to_string(brep.correction_loops) + " last_ran " +
             std::to_string(brep.last_correction_ran ? 1 : 0) + " conv " +
             std::to_string(brep.last_correction_converged ? 1 : 0) + " att " +
             std::to_string(brep.last_correction_attempts) + " spec " +
             std::to_string(brep.spectator_a) + "/" + std::to_string(brep.spectator_z) +
             " casc " + std::to_string(brep.n_cascaders);
      secs = ifs.secondaries;
      n_secs = ifs.n_secondaries;
      killed = (ifs.status == physics::hadronic::HadFinalStateStatus::kStopAndKill);
    } else {
      proj.baryon_number = (pdg == 2212 || pdg == 2112) ? 1 : 0;
      proj.mass = (pdg == 2212)   ? deex::pdg_mass_proton()
                : (pdg == 2112) ? deex::pdg_mass_neutron()
                                : bic::pdg_mass_pion_charged();
      proj.kin_energy = ekin;
      bic::BicStorage store;
      store.nucleons = nuc;
      store.scratch.momentum = mom;
      store.scratch.fermi_p = fermi;
      store.scratch.test_sums = sums;
      store.scratch.flat_block = flat;
      store.scratch.capacity = 256;
      store.proton_field = pfield;
      store.neutron_field = nfield;
      store.field_capacity = bic::kMaxFieldTable;
      store.cascade = cascade_ws();
      bic::BicRefusal nref;
      bic::BicReport nrep;
      bic::apply_yourself(proj, tgt, lt, pool, pws, store, tape, nfs, nref, nrep);
      refused = nref.any();
      if (refused) {
        std::printf("REFUSED %s: cascade=%d nucleus=%d cap=%d decay_null=%d\n", where.c_str(),
                    nref.cascade ? 1 : 0, nref.nucleus ? 1 : 0, nref.capacity ? 1 : 0,
                    nref.cascade_ref.void_decay_null ? 1 : 0);
      }
      pr.outcome = nrep.propagate_outcome;
      pr.void_report = nrep.void_report;
      secs = nfs.secondaries;
      n_secs = nfs.n_secondaries;
      killed = (nfs.status == physics::hadronic::HadFinalStateStatus::kStopAndKill);
    }
    cmp_int(b_str, refused ? 1 : 0, 0, where + " refused");
    if (refused) { continue; }
    cmp_int(b_str, killed ? 1 : 0, 1, where + " stopAndKill");
    cmp_int(b_str, tape.n, ndraws, where + " uniforms consumed");
    cmp_int(b_str, tape.overrun, 0, where + " tape not overrun");
    cmp_int(b_str, pr.outcome, bic::kPropagateVoidNucleus, where + " went out through the branch");
    cmp_int(b_str, n_secs, want_nsec, where + " secondary count");

    // What the lists held afterwards - Geant4's, read off its members by the dump.
    const bic::VoidReport& vr = pr.void_report;
    cmp_int(b_lst, vr.n_targets, iv(r, 13), where + " theTargetList");
    cmp_int(b_lst, vr.n_secondaries, iv(r, 14), where + " theSecondaryList");
    cmp_int(b_lst, vr.n_captured, iv(r, 15), where + " theCapturedList");
    cmp_int(b_lst, vr.n_final, iv(r, 16), where + " theFinalState");
    cmp_int(b_lst, vr.current_a, iv(r, 25), where + " currentA");
    cmp_int(b_lst, vr.current_z, iv(r, 26), where + " currentZ");
    // The collision drain, counted on Geant4's side by the dump's `VoidDrainCountingDecay`: the
    // decays it asked for a final state, and of those the ones with one track, which it kept.
    cmp_int(b_str, vr.n_late_decays, iv(r, 27), where + " decays drawn in the drain");
    cmp_int(b_str, vr.n_lates, iv(r, 28), where + " drained decays kept");
    const double escale = std::fabs(dv(r, 23));
    cmp_scaled(b_lsm, vr.momentum_transfer.x, dv(r, 17), escale, where + " transfer x");
    cmp_scaled(b_lsm, vr.momentum_transfer.y, dv(r, 18), escale, where + " transfer y");
    cmp_scaled(b_lsm, vr.momentum_transfer.z, dv(r, 19), escale, where + " transfer z");
    cmp_scaled(b_lsm, vr.projectile_4mom.v.x, dv(r, 20), escale, where + " projectile px");
    cmp_scaled(b_lsm, vr.projectile_4mom.v.y, dv(r, 21), escale, where + " projectile py");
    cmp_scaled(b_lsm, vr.projectile_4mom.v.z, dv(r, 22), escale, where + " projectile pz");
    cmp_scaled(b_lsm, vr.projectile_4mom.e, dv(r, 23), escale, where + " projectile e");
    cmp_scaled(b_lsm, vr.initial_nuclear_mass, dv(r, 24), escale, where + " initial mass");
    if (vr.branch >= 0 && vr.branch < 3) { ++branch_count[vr.branch]; }
    if (vr.n_targets == 0) { ++n_empty_targets; }
    n_late_decays += vr.n_late_decays;
    n_lates += vr.n_lates;
    if (vr.captured_reordered) {
      ++n_reordered;
      ++reordered_of[cname];
    }
    if (cname.rfind("void_captured_", 0) == 0) {
      cmp_int(b_str, vr.n_captured >= 2 ? 1 : 0, 1, where + " two or more captured");
    }
    worst_loops = std::max(worst_loops, vr.momentum_loops);
    worst_left = std::max(worst_left, vr.momentum_left);

    // The ion path's branch output, before the light-ion reaction touched it. `products` is the
    // cascade workspace `propagate` wrote it into, and nothing after `Propagate` writes there:
    // `SortResult` copies out of it.
    if (ion) {
      const auto& want = pr_of[cname + "#" + std::to_string(ev)];
      cmp_int(b_bstr, pr.n_products, static_cast<int>(want.size()), where + " branch products");
      double bworst = 0.0;
      std::string bworst_at;
      for (const auto* wr : want) {
        const int i = iv(*wr, 2);
        if (i < 0 || i >= pr.n_products) { continue; }
        const bic::CascadeProduct& cp = products[i];
        const std::string bw = where + " branch product " + std::to_string(i);
        cmp_int(b_bstr, cp.pdg, iv(*wr, 3), bw + " pdg");
        cmp_int(b_bstr, cp.newly_added ? 1 : 0, iv(*wr, 8), bw + " NewlyAdded");
        cmp_int(b_bstr, cp.creator_model_id, iv(*wr, 9), bw + " creator model id");
        const double sc = std::fabs(dv(*wr, 7));
        cmp_scaled(b_bout, cp.momentum.v.x, dv(*wr, 4), sc, bw + " px");
        cmp_scaled(b_bout, cp.momentum.v.y, dv(*wr, 5), sc, bw + " py");
        cmp_scaled(b_bout, cp.momentum.v.z, dv(*wr, 6), sc, bw + " pz");
        cmp_scaled(b_bout, cp.momentum.e, dv(*wr, 7), sc, bw + " e");
        const double d = std::max({std::fabs(cp.momentum.v.x - dv(*wr, 4)),
                                   std::fabs(cp.momentum.v.y - dv(*wr, 5)),
                                   std::fabs(cp.momentum.v.z - dv(*wr, 6)),
                                   std::fabs(cp.momentum.e - dv(*wr, 7))}) /
                         ((sc > 0.0) ? sc : 1.0);
        if (d > bworst) { bworst = d; bworst_at = bw; }
        if (std::getenv("P20_DUMP_EVENT") != nullptr && where == std::getenv("P20_DUMP_EVENT")) {
          std::printf("    %s pdg %d new %d  got (%.12g %.12g %.12g %.12g) want (%.12g %.12g "
                      "%.12g %.12g) d %.3g\n",
                      bw.c_str(), cp.pdg, cp.newly_added ? 1 : 0, cp.momentum.v.x,
                      cp.momentum.v.y, cp.momentum.v.z, cp.momentum.e, dv(*wr, 4), dv(*wr, 5),
                      dv(*wr, 6), dv(*wr, 7), d);
        }
      }
      if (bworst > 1e-9) {
        std::printf("  BRANCH DEVIATION %-26s %.3g at %s; branch %d, momentum loops %d\n",
                    where.c_str(), bworst, bworst_at.c_str(), vr.branch, vr.momentum_loops);
      }
    }

    // Every secondary, in order.
    int seen = 0;
    double ev_worst = 0.0;
    std::string ev_worst_at;
    double tot_e = 0.0;
    for (const auto* fr : fs_of[cname + "#" + std::to_string(ev)]) {
      const int i = iv(*fr, 2);
      if (i < 0 || i >= n_secs) { continue; }
      const physics::hadronic::HadSecondary<double>& s = secs[i];
      const std::string sw = where + " sec " + std::to_string(i);
      int got_pdg = (s.pdg != 0) ? s.pdg : physics::hadronic::pdg_nuclear_code(s.z, s.a);
      int want_pdg = iv(*fr, 3);
      // The isomer digit is not compared, for the reason tests/test_bic_apply.cu's ion tape
      // gives: it is the run's ion table's, and P3 does not invent it.
      if (got_pdg > 1000000000 && want_pdg > 1000000000) {
        got_pdg = (got_pdg / 10) * 10;
        want_pdg = (want_pdg / 10) * 10;
      }
      cmp_int(b_str, got_pdg, want_pdg, sw + " pdg");
      const double e = s.total_energy();
      const double p = s.momentum();
      const double sc = std::fabs(dv(*fr, 7));
      cmp_scaled(ion ? b_imom : b_mom, p * s.direction.x, dv(*fr, 4), sc, sw + " px");
      cmp_scaled(ion ? b_imom : b_mom, p * s.direction.y, dv(*fr, 5), sc, sw + " py");
      cmp_scaled(ion ? b_imom : b_mom, p * s.direction.z, dv(*fr, 6), sc, sw + " pz");
      cmp_scaled(ion ? b_imom : b_mom, e, dv(*fr, 7), sc, sw + " e");
      {
        const double ref_sc = (sc > 0.0) ? sc : 1.0;
        const double d = std::max({std::fabs(p * s.direction.x - dv(*fr, 4)),
                                   std::fabs(p * s.direction.y - dv(*fr, 5)),
                                   std::fabs(p * s.direction.z - dv(*fr, 6)),
                                   std::fabs(e - dv(*fr, 7))}) / ref_sc;
        if (d > ev_worst) { ev_worst = d; ev_worst_at = sw; }
        if (std::getenv("P20_DUMP_EVENT") != nullptr && where == std::getenv("P20_DUMP_EVENT")) {
          std::printf("    %s pdg %d z %d a %d  got (%.10g %.10g %.10g %.10g) want (%.10g %.10g "
                      "%.10g %.10g) d %.3g\n",
                      sw.c_str(), s.pdg, s.z, s.a, p * s.direction.x, p * s.direction.y,
                      p * s.direction.z, e, dv(*fr, 4), dv(*fr, 5), dv(*fr, 6), dv(*fr, 7), d);
        }
      }
      cmp_int(b_str, s.creator_model_id, iv(*fr, 9), sw + " creator model id");
      tot_e += e;
      ++seen;
    }
    cmp_int(b_str, seen, n_secs, where + " secondary rows read");
    if (ev_worst > 1e-9) {
      std::printf("  DEVIATION %-26s %.3g at %s; branch %d, momentum loops %d%s\n",
                  where.c_str(), ev_worst, ev_worst_at.c_str(), vr.branch, vr.momentum_loops,
                  diag.c_str());
    }

    // The energy, nucleon path only: the ion path's cascaders go on through
    // `EnergyAndMomentumCorrector`, which tests/test_bic_apply.cu asserts on its own terms.
    if (!ion) {
      const double want_e = proj.kin_energy + proj.mass + deex::nuclear_mass(ta, tz);
      const double deficit = want_e - tot_e;
      Bucket& b = buckets[b_nrg];
      ++b.n;
      const double miss = std::fabs(deficit - vr.energy_residue);
      if (miss > b.worst) {
        b.worst = miss;
        b.where = where + " branch " + std::to_string(vr.branch) + " deficit " +
                  std::to_string(deficit) + " residue " + std::to_string(vr.energy_residue);
      }
    }
  }
  if (rows.empty()) {
    std::printf("bic_void_tape.csv is empty\n");
    ++fails;
  }

  // The capture ORDER matters only where it is not the pool's, and a tape that never reaches
  // such an event cannot tell the two apart: the CAPTURED cases exist to reach them, and they
  // must.
  {
    long long in_captured_cases = 0;
    for (const auto& kv : reordered_of) {
      if (kv.first.rfind("void_captured_", 0) == 0) { in_captured_cases += kv.second; }
    }
    cmp_int(b_sel, in_captured_cases > 0 ? 1 : 0, 1,
            "the CAPTURED cases reach a capture order that is not the pool's");
    std::printf("  captured lists out of pool order: %lld events, %lld of them in the CAPTURED "
                "cases\n", n_reordered, in_captured_cases);
  }

  std::printf("\n%-30s %10s %14s  %s\n", "bucket", "points", "worst", "where");
  for (const Bucket& b : buckets) {
    const bool bad = b.worst > b.tol;
    if (bad) { ++fails; }
    std::printf("%-30s %10lld %14.4g  %s%s\n", b.name, b.n, b.worst, bad ? "FAIL " : "",
                b.where.c_str());
  }
  std::printf("  %lld taped events (%lld nucleon path, %lld ion path); the branch's energy "
              "branches: shared %lld, corrected %lld, correction refused %lld; %lld with no "
              "target nucleon left; %lld scheduled decays drawn in the drain, %lld products "
              "taken from the collision drain; momentum loop "
              "worst %d passes, %.3g MeV left\n",
              n_events, n_nucleon, n_ion, branch_count[0], branch_count[1], branch_count[2],
              n_empty_targets, n_late_decays, n_lates, worst_loops, worst_left);
  std::printf("\ntest_bic_void: %s\n", (fails == 0) ? "PASS" : "FAIL");
  return (fails == 0) ? 0 : 1;
}
