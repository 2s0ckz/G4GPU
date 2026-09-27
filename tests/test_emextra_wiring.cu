// P19: the photon's, electron's, positron's and muon's nuclear interactions, as the transport
// reaches them - host half.
//
// Eight sections, each a different kind of claim:
//
//  1. WHICH PROCESS, read off the constructed QBBC (P13's `emextra_config.csv`,
//     `emextra_windows.csv`, `species_processes.csv`): the species that carry one, where it
//     sits - `photonNuclear` in the process store and NOT on the gamma's manager, the three
//     lepton processes last or next-to-last on theirs - and nobody else.
//  2. WHERE THE PHOTON'S SITS IN G4GammaGeneralProcess: the zone edges, the slot per zone in both
//     configurations, and the 11.1.1 zone-3 branch REPLAYED with this port's cross sections - the
//     photonNuclear branch is never taken with sigM = 0 and is taken the moment sigM is not, so
//     the replay can see the thing it asserts is absent. The model each reachable energy gets.
//  3. THE THRESHOLDS ARE AN IDENTITY: for every element of every material, exactly zero AT the
//     threshold and positive just above it, and the per-volume function skipping below it
//     returns what the full element sum returns.
//  4. THE MACROSCOPIC CROSS SECTION against P13's element oracles summed over B1's materials at
//     the oracles' own energies - the plumbing (atom densities, units, the element walk) that
//     an element test cannot see.
//  5. THE TWO COPIES are bounded by tests: `select_gamma_process_at` against P1's
//     `select_gamma_process`, and `fill_result_into` against P5's `fill_result`, field by field.
//  6. `run_emextra` - G4HadronicProcess::PostStepDoIt for the four processes - runs, balances
//     baryon number, charge and energy, draws the target from the right partial sums (the
//     leptons' at the PRE-step energy: a sharp case where only one element is open), consumes
//     one attempt, and maps P13's refusals onto the right ledger names.
//  7. The ledger: every new name is in the right group.
//
// The device half - the steppers, the queue, bit-identity below the thresholds and the rates
// on the device - is `tests/test_emextra_transport.cu`.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "core/rng.cuh"
#include "data/level_data.cuh"
#include "data/materials.cuh"
#include "data/photoelectric_data.cuh"
#include "data/rayleigh_data.cuh"
#include "host/g4data.cuh"
#include "host/hadronic_upload.cuh"
#include "physics/em/gamma_processes.cuh"
#include "physics/hadronic/interaction_apply.cuh"

using namespace g4gpu;
using real_t = double;
namespace hp = g4gpu::physics::hadronic;
namespace ee = g4gpu::physics::hadronic::emextra;
namespace hxs = g4gpu::hadronic::xs;

namespace {

int g_fails = 0;
void fail(const std::string& msg) {
  std::printf("  FAIL: %s\n", msg.c_str());
  ++g_fails;
}

std::string oracle_dir() {
  const char* e = std::getenv("G4GPU_ORACLE");
  return (e != nullptr && e[0] != '\0') ? std::string(e) : std::string("ref/oracle");
}

std::vector<std::string> split(const std::string& s) {
  std::vector<std::string> out;
  std::string cur;
  for (const char c : s) {
    if (c == ',') {
      out.push_back(cur);
      cur.clear();
    } else if (c != '\n' && c != '\r') {
      cur.push_back(c);
    }
  }
  out.push_back(cur);
  return out;
}

/// A CSV addressed by column name - P13's shape, so a reordered dump cannot compare the wrong
/// column.
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
  int i(std::size_t r, const char* col) const { return std::atoi(s(r, col).c_str()); }
};

/// A uniform generator with a definite stream for the replays and the random final states.
struct Lcg {
  unsigned long long s = 88172645463325252ULL;
  __host__ __device__ double uniform() {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return double(s >> 11) * (1.0 / 9007199254740992.0);
  }
};

const char* kMatName[data::kNumMaterials] = {"G4_AIR", "G4_WATER", "G4_A-150_TISSUE",
                                            "G4_BONE_COMPACT_ICRU"};

}  // namespace

int main() {
  const std::string dir = oracle_dir();
  static data::Material<real_t> mats[data::kNumMaterials];
  data::build_b1_materials<real_t>(mats);

  host::EmExtraHostTables<real_t> ht;
  host::build_emextra_host_tables<real_t>(ht, mats, data::kNumMaterials, /*verbose=*/true);
  if (!ht.gamma_ok) {
    std::printf("G4PARTICLEXS gamma data could not be resolved - nothing here can be tested\n");
    return 1;
  }
  const had::EmExtraTables<real_t> tables = ht.view();

  // ============================================================================================
  // 1. Which process, against the constructed QBBC
  // ============================================================================================
  std::printf("== 1. which species carries which process, against emextra_config.csv ==\n");
  {
    Csv c;
    if (!c.load(dir + "/emextra_config.csv")) {
      fail("cannot read " + dir + "/emextra_config.csv");
    } else {
      struct Want { ParticleType t; const char* name; const char* process; const char* source; };
      // `source` is WHERE the run holds the process: the gamma's is in G4HadronicProcessStore
      // only, because G4GammaGeneralProcess owns it; the leptons' are on their managers.
      const Want wants[] = {
          {ParticleType::kGamma, "gamma", "photonNuclear", "store"},
          {ParticleType::kElectron, "e-", "electronNuclear", "manager"},
          {ParticleType::kPositron, "e+", "positronNuclear", "manager"},
          {ParticleType::kMuonMinus, "mu-", "muonNuclear", "manager"},
          {ParticleType::kMuonPlus, "mu+", "muonNuclear", "manager"},
      };
      for (const Want& w : wants) {
        const had::EmExtraProcess p = had::emextra_process_of(w.t);
        if (std::string(had::emextra_process_name(p)) != w.process) {
          fail(std::string(w.name) + ": the transport gives '" + had::emextra_process_name(p)
               + "', the run has '" + w.process + "'");
        }
        bool found = false, on_manager = false;
        for (std::size_t r = 0; r < c.rows.size(); ++r) {
          if (c.s(r, "particle") != w.name || c.s(r, "process") != w.process) { continue; }
          if (c.i(r, "process_subtype") != 121) {
            fail(std::string(w.name) + " " + w.process + " is not fHadronInelastic (121)");
          }
          found |= (c.s(r, "source") == w.source);
          on_manager |= (c.s(r, "source") == "manager");
        }
        if (!found) {
          fail(std::string(w.name) + ": the run holds no '" + w.process + "' in its " + w.source);
        }
        // THE PHOTON'S IS NOT ON THE MANAGER, which is the whole reason step_gamma folds it
        // into its own total instead of drawing a sixth length.
        if (w.t == ParticleType::kGamma && on_manager) {
          fail("photonNuclear is on the gamma's process manager - the general process is off "
               "in this run and emextra_wiring.cuh's premise is wrong");
        }
      }
      // And the premise of the zone-3 finding: no G4GammaConversionToMuons, so sigM == 0.
      for (std::size_t r = 0; r < c.rows.size(); ++r) {
        if (c.s(r, "process") == "GammaToMuPair") {
          fail("the run HAS GammaToMuPair: sigM is not zero, and RISK V207's zone-3 argument "
               "does not hold for it");
        }
      }
      // Nobody else: every ParticleType the transport steps and QBBC gives no G4EmExtraPhysics
      // process answers kNone. (The dump covers the five particles the class touches; for
      // everything else the absence is the class's own `ConstructProcess`.)
      int n_with = 0;
      for (int t = 0; t < static_cast<int>(ParticleType::kNumTypes); ++t) {
        if (had::emextra_process_of(static_cast<ParticleType>(t)) != had::EmExtraProcess::kNone) {
          ++n_with;
        }
      }
      if (n_with != 5) {
        fail("emextra_process_of names a process for " + std::to_string(n_with)
             + " species; G4EmExtraPhysics gives one to exactly five");
      }
    }
    // Where the lepton processes sit on their managers: LAST for e-, e+ (after CoulombScat) and
    // after CoulombScat, before Decay, for the muons - `step_lepton` draws it last and
    // `step_hadron` next-to-last, and their tie comments say why that is the order they keep.
    Csv sp;
    if (!sp.load(dir + "/species_processes.csv")) {
      fail("cannot read species_processes.csv");
    } else {
      struct Ord { const char* p; const char* process; const char* before; const char* after; };
      const Ord ords[] = {{"e-", "electronNuclear", "CoulombScat", ""},
                          {"e+", "positronNuclear", "CoulombScat", ""},
                          {"mu-", "muonNuclear", "CoulombScat", "Decay"},
                          {"mu+", "muonNuclear", "CoulombScat", "Decay"}};
      for (const Ord& o : ords) {
        int ip = -1, ib = -1, ia = -1, last = -1, k = 0;
        for (std::size_t r = 0; r < sp.rows.size(); ++r) {
          if (sp.rows[r][0] != o.p) { continue; }
          const std::string& pn = sp.rows[r][1];
          if (pn == o.process) { ip = k; }
          if (pn == o.before) { ib = k; }
          if (o.after[0] != '\0' && pn == o.after) { ia = k; }
          last = k;
          ++k;
        }
        const bool ok_order = (ip > ib && ib >= 0)
                              && ((o.after[0] == '\0') ? (ip == last) : (ia == ip + 1));
        std::printf("   %-4s %-16s at position %d of %d%s\n", o.p, o.process, ip, last + 1,
                    ok_order ? "" : "  <- UNEXPECTED");
        if (!ok_order) {
          fail(std::string(o.p) + ": " + o.process + " is not where the steppers' tie order "
               "assumes it is");
        }
      }
    }
  }

  // ============================================================================================
  // 2. The photon's place in G4GammaGeneralProcess
  // ============================================================================================
  std::printf("== 2. photonNuclear inside G4GammaGeneralProcess ==\n");
  {
    // (a) The zone edges, transcribed: 150 keV, 2 m_e, 100 MeV, strict `<` on each.
    const real_t me2 = real_t(2) * units::electron_mass_c2<real_t>();
    struct Z { real_t e; int zone; };
    const Z zs[] = {{0.1499999, 0}, {0.150, 1}, {me2 * (1 - 1e-12), 1}, {me2, 2},
                    {99.9999999, 2}, {100.0, 3}, {1e5, 3}};
    for (const Z& z : zs) {
      if (had::gamma_general_zone<real_t>(z.e) != z.zone) {
        fail("gamma_general_zone(" + std::to_string(z.e) + ") is not "
             + std::to_string(z.zone));
      }
    }
    // (b) The slot per zone, both configurations.
    using S = had::GammaNuclearSlot;
    const S on[4] = {S::kAbsent, S::kAbsent, S::kPhotoNuclear, S::kConversion};
    for (int zi = 0; zi < 7; ++zi) {
      const S got_on = had::gamma_nuclear_slot<real_t>(had::GammaGeneralProcess::kOn, zs[zi].e);
      const S got_off = had::gamma_nuclear_slot<real_t>(had::GammaGeneralProcess::kOff, zs[zi].e);
      if (got_on != on[zs[zi].zone]) { fail("general-process slot wrong at " + std::to_string(zs[zi].e)); }
      if (got_off != S::kPhotoNuclear) {
        fail("with the general process off photonNuclear must be a competitor at every energy");
      }
    }

    // (c) THE ZONE-3 BRANCH, REPLAYED. `G4GammaGeneralProcess::BuildPhysicsTable` fills tables
    //     10-14 from the four cross sections and `PostStepDoIt`'s case 3 walks them with one
    //     uniform. Built here from THIS PORT's cross sections at the node energies Geant4 would
    //     use, the replay is a statement about the branch structure, which is what V207 is: with
    //     sigM = 0.0 the photonNuclear arm is never taken and its share lands on `conv`; with a
    //     sigM of any size it is taken, so the zero is something this replay could have missed
    //     and did not.
    const std::string phot_dir = host::g4emlow_subdir("epics2017/phot", "pe-cs-1.dat");
    const std::string rayl_dir = host::g4emlow_subdir("epics2017/rayl", "re-cs-1.dat");
    int zlist[32];
    int nzl = 0;
    for (int m = 0; m < data::kNumMaterials; ++m) {
      for (int i = 0; i < mats[m].n_elements; ++i) {
        const int z = static_cast<int>(mats[m].z[i] + 0.5);
        bool seen = false;
        for (int k = 0; k < nzl; ++k) { seen |= (zlist[k] == z); }
        if (!seen) { zlist[nzl++] = z; }
      }
    }
    static data::PhotoElectricTable<real_t> pe{};
    static std::vector<real_t> pte, ptv;
    static data::RayleighTable<real_t> rt{};
    static std::vector<real_t> rte, rtv;
    const bool pe_ok = data::load_photoelectric<real_t>(phot_dir, zlist, nzl, pe, pte, ptv);
    pe.table_e = pte.data();
    pe.table_v = ptv.data();
    const bool ra_ok = data::load_rayleigh<real_t>(rayl_dir, zlist, nzl, rt, rte, rtv);
    rt.table_e = rte.data();
    rt.table_v = rtv.data();
    if (!pe_ok || !ra_ok) { fail("could not load the photoelectric / Rayleigh tables"); }

    long long gn_taken_m0 = 0, gn_taken_m1 = 0, conv_extra = 0, grid = 0;
    const real_t es[] = {100.0, 120.0, 150.0, 200.0, 500.0, 1000.0, 3000.0};
    for (int m = 0; m < data::kNumMaterials; ++m) {
      for (real_t e : es) {
        const auto xs = em::gamma_macroscopic_xs<real_t>(mats[m], e, &pe, &rt);
        const real_t sn =
            had::photon_nuclear_xs<real_t>(tables, had::GammaGeneralProcess::kOn, m, mats[m], e);
        for (int pass = 0; pass < 2; ++pass) {
          const real_t sm = (pass == 0) ? real_t(0) : real_t(0.1) * sn;  // hypothetical mu-pair
          const real_t sum = xs.compton + xs.pair + xs.photoelectric + sn + sm;
          const real_t p11 = (xs.compton + xs.photoelectric + sn + sm) / sum;
          const real_t p12 = (xs.photoelectric + sn + sm) / sum;
          const real_t p13 = (sn + sm) / sum;
          const real_t p14 = sn / sum;
          for (int k = 0; k < 20000; ++k) {
            const real_t q = (k + 0.5) / 20000.0;
            int branch;
            if (q + p11 <= 1.0) { branch = 0; }        // conv
            else if (q + p12 <= 1.0) { branch = 1; }   // compt
            else if (q + p13 <= 1.0) { branch = 2; }   // phot
            else if (q + p14 <= 1.0) { branch = 3; }   // photonNuclear
            else { branch = (sm > 0) ? 4 : 0; }        // theConversionMM, else conv
            if (branch == 3) { (pass == 0 ? gn_taken_m0 : gn_taken_m1) += 1; }
            if (pass == 0) {
              ++grid;
              // The share the port hands to conversion: q in the top slice.
              if (q * (xs.compton + xs.pair + xs.photoelectric + sn) >= xs.compton + xs.pair
                                                                           + xs.photoelectric) {
                if (branch != 0) {
                  fail("zone 3: a q in the photo-nuclear slice did not select conv in Geant4's "
                       "branch replay");
                }
                ++conv_extra;
              }
            }
          }
        }
      }
    }
    std::printf("   zone-3 replay, %lld (material, energy, q) points: photonNuclear taken %lld "
                "times with sigM = 0 (conv got %lld top-slice q), %lld times with sigM = 0.1 sigN\n",
                grid, gn_taken_m0, conv_extra, gn_taken_m1);
    if (gn_taken_m0 != 0) { fail("zone 3: photonNuclear was selected with sigM = 0"); }
    if (gn_taken_m1 == 0) {
      fail("zone 3: the replay never takes photonNuclear even with sigM > 0 - it cannot see "
           "what it asserts is absent");
    }
    if (conv_extra == 0) { fail("zone 3: no q fell in the photo-nuclear slice at all"); }

    // (d) The model each reachable photon energy gets, through the entry point's own range
    //     manager, off `emextra_windows.csv`. In QBBC (general process on) the photon reaches
    //     photonNuclear only below 100 MeV, so GammaNPreco is the ONLY model: 100 draws at each
    //     energy, every one of them it, and none of them drawing a uniform (one window only).
    Csv w;
    if (!w.load(dir + "/emextra_windows.csv")) {
      fail("cannot read emextra_windows.csv");
    } else {
      double pre_max = -1, bert_min = -1, qgs_min = -1;
      for (std::size_t r = 0; r < w.rows.size(); ++r) {
        if (w.s(r, "particle") != "gamma") { continue; }
        if (w.s(r, "model") == "GammaNPreco") { pre_max = w.d(r, "max_MeV"); }
        if (w.s(r, "model") == "BertiniCascade") { bert_min = w.d(r, "min_MeV"); }
        if (w.s(r, "model") == "TheoFSGenerator") { qgs_min = w.d(r, "min_MeV"); }
      }
      std::printf("   emextra_windows: GammaNPreco to %.0f, Bertini from %.0f, QGS from %.0f MeV "
                  "- the general process stops at %.0f\n",
                  pre_max, bert_min, qgs_min, double(had::gamma_general_min_mm<real_t>()));
      if (!(bert_min > double(had::gamma_general_min_mm<real_t>()))
          || !(qgs_min > double(had::gamma_general_min_mm<real_t>()))) {
        fail("a photon model other than GammaNPreco starts below 100 MeV - V207's consequence "
             "that Bertini and QGS are unreachable in QBBC would be false");
      }
      const ee::ProcessConfig cfg = ee::qbbc_config(22);
      hp::ModelRange<real_t> rr[3];
      const int nr = ee::model_ranges<real_t>(cfg, rr);
      for (real_t e : {me2, real_t(5), real_t(20), real_t(50), real_t(99.999)}) {
        Lcg g;
        const unsigned long long s0 = g.s;
        for (int k = 0; k < 100; ++k) {
          const hp::ModelSelection sel = hp::choose_hadronic_interaction<real_t>(rr, nr, e, 0, g);
          if (sel.status != hp::ModelChoice::kOk
              || cfg.models[sel.index].model != ee::Model::kGammaNPreco) {
            fail("a zone-2 photon at " + std::to_string(e) + " MeV got a model other than "
                 "GammaNPreco");
            break;
          }
        }
        if (g.s != s0) { fail("the model choice drew a uniform at " + std::to_string(e) + " MeV"); }
      }
    }
  }

  // ============================================================================================
  // 3. The thresholds are an identity
  // ============================================================================================
  std::printf("== 3. the thresholds: exactly zero at and below them ==\n");
  {
    for (int m = 0; m < data::kNumMaterials; ++m) {
      const real_t tg = had::emextra_threshold<real_t>(tables, had::kGammaNuclearThreshold, m);
      const real_t te = had::emextra_threshold<real_t>(tables, had::kElectroNuclearThreshold, m);
      std::printf("   %-22s photo-nuclear %9.5f MeV, electro-nuclear %9.5f MeV\n", kMatName[m], tg,
                  te);
      real_t min_g = 1e30, min_e = 1e30;
      for (int i = 0; i < mats[m].n_elements; ++i) {
        const int z = static_cast<int>(mats[m].z[i] + 0.5);
        const real_t g = host::emextra_gamma_element_threshold<real_t>(ht.gamma, z);
        const real_t e = host::emextra_electro_element_threshold<real_t>(z);
        min_g = std::fmin(min_g, g);
        min_e = std::fmin(min_e, e);
        // THE IDENTITY: exactly zero at the threshold and at 400 log-spaced energies from
        // 10 keV up to it. That, and only that, is what makes skipping the evaluation below it
        // the same number as evaluating it.
        long long nonzero_below = 0;
        for (int k = 0; k <= 400; ++k) {
          const real_t f = std::pow(10.0, -2.0 + k * (std::log10(1.0) + 2.0) / 400.0);  // 0.01..1
          const real_t eg = g * f, ee_ = e * f;
          if (g > 0 && hxs::pxs_element_xs<real_t>(ht.gamma, eg, std::log(eg), z).value != 0.0) {
            ++nonzero_below;
          }
          hxs::chips::ElnState st;
          if (hxs::chips::eln_element_xs<real_t>(st, ee_, z).value != 0.0) { ++nonzero_below; }
        }
        if (nonzero_below != 0) {
          fail(std::string(kMatName[m]) + " Z=" + std::to_string(z) + ": "
               + std::to_string(nonzero_below) + " energies at or below a threshold have a "
               "nonzero cross section - skipping there would not be an identity");
        }
        // TIGHTNESS, which is efficiency and not correctness, and the two data sets differ:
        // `G4GammaNuclearXS`'s vectors are positive a hair above their last leading zero, while
        // CHIPS's electro-nuclear J-functions stay zero for a while above `ThresholdEnergy`
        // (their first non-zero channel is `LL[i]`), so that threshold is safe and loose. The gap
        // is printed so the looseness is a measured number rather than an unexamined one.
        const real_t gu = g * (1 + 1e-9);
        if (g > 0 && !(hxs::pxs_element_xs<real_t>(ht.gamma, gu, std::log(gu), z).value > 0.0)) {
          fail("Z=" + std::to_string(z) + ": photo-nuclear xs just above its threshold is not "
               "positive - the threshold is not the last leading zero");
        }
        real_t e_open = e;
        for (int k = 1; k <= 20000; ++k) {
          const real_t ek = e * (1.0 + 1e-4 * k);
          hxs::chips::ElnState st;
          if (hxs::chips::eln_element_xs<real_t>(st, ek, z).value > 0.0) {
            e_open = ek;
            break;
          }
        }
        if (m == data::kBoneCompact) {
          std::printf("        Z=%2d  electro-nuclear zero to %8.4f MeV, first positive at %8.4f\n",
                      z, e, e_open);
        }
      }
      if (min_g != tg || min_e != te) {
        fail(std::string(kMatName[m]) + ": the uploaded threshold is not the minimum over its elements");
      }
      // The per-ELEMENT table `EmExtraXsFn::element` skips below: the same two functions, entry
      // for entry, for every element of the material.
      for (int i = 0; i < mats[m].n_elements; ++i) {
        const int z = static_cast<int>(mats[m].z[i] + 0.5);
        if (had::emextra_element_threshold<real_t>(tables, had::kGammaNuclearThreshold, z)
                != host::emextra_gamma_element_threshold<real_t>(ht.gamma, z)
            || had::emextra_element_threshold<real_t>(tables, had::kElectroNuclearThreshold, z)
                   != host::emextra_electro_element_threshold<real_t>(z)) {
          fail(std::string(kMatName[m]) + " Z=" + std::to_string(z)
               + ": the per-element threshold table is not the element functions'");
        }
      }
      // The per-volume function below, at and above the threshold against the full element sum
      // with BOTH skips switched off (null views of the material and the element tables):
      // identical doubles everywhere - at the material's threshold, around every element's own,
      // and across a grid above them, where the element skip is what is being tested.
      had::EmExtraTables<real_t> nothr = tables;
      nothr.threshold = nullptr;
      nothr.element_threshold = nullptr;
      for (const had::EmExtraProcess p : {had::EmExtraProcess::kPhotonNuclear, had::EmExtraProcess::kElectronNuclear}) {
        const int kind = (p == had::EmExtraProcess::kPhotonNuclear) ? had::kGammaNuclearThreshold
                                                                    : had::kElectroNuclearThreshold;
        const real_t t = (p == had::EmExtraProcess::kPhotonNuclear) ? tg : te;
        std::vector<real_t> es_chk;
        for (real_t f : {0.5, 0.999999, 1.0, 1.000001, 1.5, 3.0}) { es_chk.push_back(t * f); }
        for (int i = 0; i < mats[m].n_elements; ++i) {
          const int z = static_cast<int>(mats[m].z[i] + 0.5);
          const real_t tz = had::emextra_element_threshold<real_t>(tables, kind, z);
          for (real_t f : {0.999999, 1.0, 1.000001, 1.01}) { es_chk.push_back(tz * f); }
        }
        for (int k = 0; k <= 60; ++k) { es_chk.push_back(t * std::pow(10.0, k / 30.0)); }
        for (const real_t e : es_chk) {
          const real_t f = (t > 0) ? e / t : 0;
          if (!(e > 0)) { continue; }
          hxs::MaterialXs<real_t> a{}, b{};
          const real_t xa = had::emextra_xs_per_volume<real_t>(tables, p, m, mats[m], e, a);
          const real_t xb = had::emextra_xs_per_volume<real_t>(nothr, p, m, mats[m], e, b);
          // The partial sums the target is drawn from as well, wherever the skipped call built
          // them at all (below the material's threshold it builds none, and the drain then has
          // no target to draw - which the evaluated call's all-zero sums would not give it either).
          bool partials_differ = false;
          if (xa > 0) {
            partials_differ = (a.n_elements != b.n_elements);
            for (int i = 0; i < a.n_elements && !partials_differ; ++i) {
              partials_differ = (a.cumulative[i] != b.cumulative[i]);
            }
          }
          if (partials_differ) {
            fail(std::string(kMatName[m]) + " " + had::emextra_process_name(p)
                 + ": the element skip changed the partial sums the target is drawn from");
          }
          if (xa != xb) {
            // %.17g and not std::to_string, whose %f prints a 1e-8 /mm cross section as zero.
            char msg[256];
            std::snprintf(msg, sizeof(msg), "%s %s at %.7g x threshold: skipped %.17g against "
                          "evaluated %.17g", kMatName[m], had::emextra_process_name(p),
                          double(f), double(xa), double(xb));
            fail(msg);
          }
        }
      }
    }
    // The two numbers the file header and the gate's bit-identity rest on.
    if (had::emextra_threshold<real_t>(tables, had::kGammaNuclearThreshold, data::kWater) != 11.499) {
      fail("water's photo-nuclear threshold is not oxygen's 11.499 MeV");
    }
  }

  // ============================================================================================
  // 4. The macroscopic cross sections against P13's element oracles
  // ============================================================================================
  std::printf("== 4. per-volume cross sections against the element oracles ==\n");
  {
    const double mb = double(hxs::chips::millibarn<double>());
    // (Z, energy) -> element xs in mb, from each oracle.
    std::map<std::pair<int, double>, double> gam, eln, kok;
    Csv g, e, k;
    if (!g.load(dir + "/emextra_gammanuc.csv")) { fail("cannot read emextra_gammanuc.csv"); }
    if (!e.load(dir + "/emextra_electronuc.csv")) { fail("cannot read emextra_electronuc.csv"); }
    if (!k.load(dir + "/emextra_kokoulin.csv")) { fail("cannot read emextra_kokoulin.csv"); }
    for (std::size_t r = 0; r < g.rows.size(); ++r) {
      if (g.s(r, "kind") == "element") { gam[{g.i(r, "Z"), g.d(r, "energy_MeV")}] = g.d(r, "xs_mb"); }
    }
    for (std::size_t r = 0; r < e.rows.size(); ++r) {
      eln[{e.i(r, "Z"), e.d(r, "energy_MeV")}] = e.d(r, "xs_mb");
    }
    for (std::size_t r = 0; r < k.rows.size(); ++r) {
      if (k.s(r, "kind") == "elem") { kok[{k.i(r, "Z"), k.d(r, "T_MeV")}] = k.d(r, "value"); }
    }
    struct Set { const char* name; had::EmExtraProcess p; std::map<std::pair<int, double>, double>* m; double unit; };
    // All three oracles are in millibarn - P13's own test converts the port's mm^2 with
    // `to_mb` before comparing each, Kokoulin's element rows included.
    const Set sets[] = {{"photonNuclear", had::EmExtraProcess::kPhotonNuclear, &gam, mb},
                        {"electronNuclear", had::EmExtraProcess::kElectronNuclear, &eln, mb},
                        {"muonNuclear", had::EmExtraProcess::kMuonNuclear, &kok, mb}};
    for (const Set& s : sets) {
      // The energies every element of a material has an oracle row at.
      std::set<double> energies;
      for (const auto& kv : *s.m) { energies.insert(kv.first.second); }
      long long n = 0, n_zero = 0;
      double worst = 0;
      for (int m = 0; m < data::kNumMaterials; ++m) {
        for (double en : energies) {
          double want = 0;
          bool all = true;
          for (int i = 0; i < mats[m].n_elements; ++i) {
            const int z = static_cast<int>(mats[m].z[i] + 0.5);
            const auto it = s.m->find({z, en});
            if (it == s.m->end()) { all = false; break; }
            want += double(mats[m].n_atoms[i]) * it->second * s.unit;
          }
          if (!all) { continue; }
          hxs::MaterialXs<real_t> mx{};
          const double got = had::emextra_xs_per_volume<real_t>(tables, s.p, m, mats[m], en, mx);
          ++n;
          if (want == 0) {
            ++n_zero;
            if (got != 0) { fail(std::string(s.name) + " in " + kMatName[m] + " at " + std::to_string(en) + " MeV: the oracle sum is 0 and the port says " + std::to_string(got)); }
            continue;
          }
          const double rel = std::fabs(got - want) / want;
          if (rel > worst) { worst = rel; }
          if (rel > 1e-12) {
            fail(std::string(s.name) + " in " + kMatName[m] + " at " + std::to_string(en)
                 + " MeV: " + std::to_string(got) + " against " + std::to_string(want));
          }
        }
      }
      std::printf("   %-16s %5lld (material, energy) points, %4lld of them exactly zero on both "
                  "sides, worst relative %.2e\n", s.name, n, n_zero, worst);
      if (n < 50) { fail(std::string(s.name) + ": too few oracle points compared to mean anything"); }
    }
  }

  // ============================================================================================
  // 5. The two copies, against their originals
  // ============================================================================================
  std::printf("== 5. select_gamma_process_at and fill_result_into against P1's and P5's ==\n");
  {
    long long n = 0, bad = 0;
    for (int m = 0; m < data::kNumMaterials; ++m) {
      for (real_t e : {0.01, 0.1, 0.5, 1.0, 3.0, 10.0, 50.0}) {
        em::GammaXS<real_t> xs{};
        xs.compton = em::compton_xs_per_atom<real_t>(e, mats[m].z[0]) * mats[m].n_atoms[0];
        xs.pair = em::pair_xs_per_atom<real_t>(e, mats[m].z[0]) * mats[m].n_atoms[0];
        xs.photoelectric = 0.37 * xs.compton;
        xs.rayleigh = 0.11 * xs.compton;
        xs.total = xs.compton + xs.pair + xs.photoelectric + xs.rayleigh;
        for (int k = 0; k < 5000; ++k) {
          const real_t q = (k + 0.25) / 5000.0;
          ++n;
          if (em::select_gamma_process<real_t>(xs, q)
              != had::select_gamma_process_at<real_t>(xs, q * xs.total)) {
            ++bad;
          }
        }
      }
    }
    std::printf("   select_gamma_process_at: %lld of %lld selections differ\n", bad, n);
    if (bad != 0) { fail("select_gamma_process_at disagrees with P1's select_gamma_process"); }

    // fill_result_into against fill_result on random final states, every field.
    Lcg g;
    auto* fs = new hp::HadFinalState<real_t, had::kInteractionSecondaryCap>();
    auto* into = new hp::HadronicStepResult<real_t, had::kInteractionSecondaryCap>();
    std::vector<real_t> pm(had::kInteractionSecondaryCap);
    long long cases = 0, mismatches = 0;
    for (int c = 0; c < 4000; ++c) {
      fs->clear();
      const int mode = c % 4;
      fs->status = (mode == 0) ? hp::HadFinalStateStatus::kStopAndKill : hp::HadFinalStateStatus::kIsAlive;
      fs->energy_change = (mode == 1) ? real_t(0) : real_t(1000 * g.uniform());
      const real_t ct = 2 * g.uniform() - 1, ph = 6.283185307179586 * g.uniform();
      const real_t stt = std::sqrt(1 - ct * ct);
      fs->momentum_change = Vec3<real_t>{stt * std::cos(ph), stt * std::sin(ph), ct};
      fs->local_energy_deposit = real_t(g.uniform());
      const int ns = static_cast<int>(40 * g.uniform());
      for (int i = 0; i < ns; ++i) {
        hp::HadSecondary<real_t> s;
        s.pdg = (i % 3 == 0) ? 2212 : ((i % 3 == 1) ? 22 : 11);
        s.mass = (s.pdg == 2212) ? real_t(938.272013 + (g.uniform() - 0.5) * 0.01) : real_t(0);
        s.kin_energy = real_t(200 * g.uniform());
        const real_t c2 = 2 * g.uniform() - 1, p2 = 6.283185307179586 * g.uniform();
        const real_t s2 = std::sqrt(1 - c2 * c2);
        s.direction = Vec3<real_t>{s2 * std::cos(p2), s2 * std::sin(p2), c2};
        s.time = real_t(g.uniform() - 0.2);
        s.weight = real_t(0.5 + g.uniform());
        fs->add_secondary(s);
        pm[i] = (s.pdg == 2212) ? real_t(938.272013) : real_t(0);
      }
      const real_t td = 2 * g.uniform() - 1, tp = 6.283185307179586 * g.uniform();
      const real_t ts = std::sqrt(1 - td * td);
      const Vec3<real_t> dir{ts * std::cos(tp), ts * std::sin(tp), td};
      const bool at_rest = (c % 2 == 0);
      const auto ref = hp::fill_result<real_t, had::kInteractionSecondaryCap, had::kInteractionSecondaryCap>(
          *fs, dir, real_t(12.5), real_t(0.75), at_rest, pm.data());
      had::fill_result_into<real_t, had::kInteractionSecondaryCap, had::kInteractionSecondaryCap>(
          *fs, dir, real_t(12.5), real_t(0.75), at_rest, pm.data(), *into);
      ++cases;
      bool same = ref.status == into->status && ref.energy == into->energy
                  && ref.momentum_direction.x == into->momentum_direction.x
                  && ref.momentum_direction.y == into->momentum_direction.y
                  && ref.momentum_direction.z == into->momentum_direction.z
                  && ref.local_energy_deposit == into->local_energy_deposit
                  && ref.weight == into->weight && ref.n_secondaries == into->n_secondaries
                  && ref.secondary_overflow == into->secondary_overflow
                  && ref.n_ic_electrons == into->n_ic_electrons
                  && ref.off_shell_fixed == into->off_shell_fixed;
      for (int i = 0; same && i < ref.n_secondaries; ++i) {
        const auto& a = ref.secondaries[i];
        const auto& b = into->secondaries[i];
        same = a.pdg == b.pdg && a.mass == b.mass && a.kin_energy == b.kin_energy
               && a.direction.x == b.direction.x && a.direction.y == b.direction.y
               && a.direction.z == b.direction.z && a.time == b.time && a.weight == b.weight;
      }
      if (!same) { ++mismatches; }
    }
    std::printf("   fill_result_into: %lld of %lld random final states differ from fill_result\n",
                mismatches, cases);
    if (mismatches != 0) { fail("fill_result_into is not P5's fill_result"); }
    delete fs;
    delete into;
  }

  // ============================================================================================
  // 6. run_emextra - G4HadronicProcess::PostStepDoIt for the four processes
  // ============================================================================================
  std::printf("== 6. run_emextra: the target, the model, the balance, the refusals ==\n");
  {
    const std::string pe_dir = host::g4photon_evaporation_dir();
    if (pe_dir.empty()) {
      fail("PhotonEvaporation not found");
    } else {
      static data::LevelTableStorage lts;
      data::read_all_level_data(
          lts, pe_dir, data::kLevelZMax, [](int Z, int A) { return deex::shell_correction(A, Z); },
          [](int Z, int A) { return deex::level_manager_level_density(Z, A); });
      const data::LevelTable lt = lts.view();
      static deex::FermiPoolStorage fps;
      deex::build_fermi_pool(fps, lt);
      const deex::FermiPool fpool = fps.view();
      auto* slot = new had::InteractionSlot<real_t>();

      struct Case {
        ParticleType t;
        real_t e;       ///< the post-step (interaction) energy
        real_t e_pre;   ///< the energy the partial sums are taken at
        int mat;
        int n;
        const char* name;
      };
      const Case cases[] = {
          {ParticleType::kGamma, 20, 20, data::kWater, 400, "gamma 20 MeV, water"},
          {ParticleType::kGamma, 20, 20, data::kBoneCompact, 400, "gamma 20 MeV, bone"},
          {ParticleType::kGamma, 60, 60, data::kAir, 200, "gamma 60 MeV, air"},
          {ParticleType::kElectron, 1000, 1000, data::kWater, 300, "e- 1 GeV, water"},
          {ParticleType::kPositron, 300, 300, data::kBoneCompact, 200, "e+ 300 MeV, bone"},
          {ParticleType::kMuonMinus, 10000, 10000, data::kBoneCompact, 150, "mu- 10 GeV, bone"},
          {ParticleType::kMuonPlus, 300, 300, data::kWater, 100, "mu+ 300 MeV, water"},
          // The sharp case for "the partials are the PRE-step energy's", its pre-step energy
          // found below: the lowest energy at which bone's electro-nuclear cross section is
          // positive and carried by ONE element alone. Every target must then be that element,
          // although the lepton reaches the interaction at 50 MeV, where the others are open too.
          {ParticleType::kElectron, 50, -1, data::kBoneCompact, 200, "e- 50 MeV, pre-step 1-elem"},
      };
      // The one-element window: scan up from bone's threshold for the first energy with a
      // positive cross section and check that exactly one element carries it.
      real_t one_elem_e = -1;
      int one_elem_z = 0;
      {
        const real_t t0 =
            had::emextra_threshold<real_t>(tables, had::kElectroNuclearThreshold, data::kBoneCompact);
        for (int k = 1; k <= 200000 && one_elem_e < 0; ++k) {
          const real_t ek = t0 * (1.0 + 1e-5 * k);
          int nopen = 0, zopen = 0;
          for (int i = 0; i < mats[data::kBoneCompact].n_elements; ++i) {
            const int z = static_cast<int>(mats[data::kBoneCompact].z[i] + 0.5);
            hxs::chips::ElnState st;
            if (hxs::chips::eln_element_xs<real_t>(st, ek, z).value > 0.0) {
              ++nopen;
              zopen = z;
            }
          }
          if (nopen == 1) {
            one_elem_e = ek;
            one_elem_z = zopen;
          }
          if (nopen > 1) { break; }
        }
        std::printf("   bone's electro-nuclear opens at %.5f MeV on Z = %d alone\n", one_elem_e,
                    one_elem_z);
        if (one_elem_e < 0) { fail("no one-element electro-nuclear window in bone"); }
      }
      for (const Case& c : cases) {
        const real_t e_pre = (c.e_pre < 0) ? one_elem_e : c.e_pre;
        if (!(e_pre > 0)) { continue; }
        const had::EmExtraProcess proc = had::emextra_process_of(c.t);
        hp::HadProjectile<real_t> proj;
        proj.pdg = pdg_code(c.t);
        proj.charge = particle_def<real_t>(c.t).charge;
        proj.mass = particle_def<real_t>(c.t).mass;
        proj.kin_energy = c.e;
        proj.baryon_number = 0;
        // The partial sums the target should follow.
        hxs::MaterialXs<real_t> mx{};
        (void)had::emextra_xs_per_volume<real_t>(tables, proc, c.mat, mats[c.mat], e_pre, mx);
        std::vector<long long> by_elem(mats[c.mat].n_elements, 0);
        int ran = 0, refused = 0, multi = 0, bad_ab = 0, alive = 0, killed = 0;
        int no_sec = 0;
        double worst_de = 0;
        std::map<int, int> refusal_counts;
        for (int k = 0; k < c.n; ++k) {
          Philox<real_t> rng(9000u + static_cast<unsigned int>(k), 7u, had::kInteractionRngPurpose);
          had::EmExtraDiag dg;
          had::InteractionOutcome o;
          if (proc == had::EmExtraProcess::kPhotonNuclear) {
            o = had::run_emextra<real_t, had::InteractionBucket::kPhotoNuclear>(
                proj, c.t, mats[c.mat], c.mat, tables, e_pre, *slot, lt, fpool, rng, &dg);
          } else {
            o = had::run_emextra<real_t, had::InteractionBucket::kLeptoNuclear>(
                proj, c.t, mats[c.mat], c.mat, tables, e_pre, *slot, lt, fpool, rng, &dg);
          }
          if (!o.ran) {
            ++refused;
            ++refusal_counts[dg.emextra_refusal];
            continue;
          }
          ++ran;
          if (o.attempts != 0) {
            ++multi;
            std::printf("        a re-entry: attempt %d accepted after CheckResult verdict %d, "
                        "dE = %.4g MeV\n", o.attempts + 1, dg.rejected_verdict,
                        dg.rejected_delta_e);
          }
          for (int i = 0; i < mats[c.mat].n_elements; ++i) {
            if (static_cast<int>(mats[c.mat].z[i] + 0.5) == o.target_z) { ++by_elem[i]; break; }
          }
          // FillResult and the balance: baryon number and nuclear charge exact (the atomic
          // conversion electron excluded, as tests/test_inelastic_transport.cu's emitter does),
          // energy to the precision the de-excitation chain's mass tables allow.
          had::fill_result_into<real_t, had::kInteractionSecondaryCap, had::kInteractionSecondaryCap>(
              slot->fs, Vec3<real_t>{0, 0, 1}, real_t(0), real_t(1), had::emextra_has_at_rest(c.t),
              slot->pdg_mass, slot->filled);
          const auto& fr = slot->filled;
          int b = 0, q = 0;
          double eout = double(fr.local_energy_deposit);
          for (int i = 0; i < fr.n_secondaries; ++i) {
            const auto& s = fr.secondaries[i];
            eout += double(s.kin_energy) + double(s.mass);
            if (s.a > 0) {
              b += s.a;
              q += s.z;
            } else if (s.pdg != 11) {
              const ParticleType st = particle_type_of_pdg(s.pdg);
              if (st != ParticleType::kNumTypes) {
                b += had::baryon_number_of(st, 0);
                q += static_cast<int>(std::lrint(particle_def<real_t>(st).charge));
              }
            } else {
              eout -= double(units::electron_mass_c2<real_t>());  // atomic: the shell pays it
            }
          }
          const bool survives = fr.status != hp::TrackStatusChange::kStopAndKill;
          if (survives) {
            ++alive;
            eout += double(fr.energy) + double(proj.mass);
            q += static_cast<int>(std::lrint(double(proj.charge)));
          } else {
            ++killed;
          }
          if (fr.n_secondaries == 0) { ++no_sec; }
          const bool untouched = survives && fr.n_secondaries == 0;
          const int want_q = static_cast<int>(std::lrint(double(proj.charge))) + o.target_z;
          if (!untouched && (b != o.target_a || q != want_q)) { ++bad_ab; }
          const double ein = double(proj.total_energy())
                             + (untouched ? 0.0 : deex::nuclear_mass(o.target_a, o.target_z));
          const double de = std::fabs(ein - eout);
          if (de > worst_de) { worst_de = de; }
        }
        // The target draw against the partial sums: a chi-square over the elements.
        double chi2 = 0;
        int dof = 0;
        for (int i = 0; i < mats[c.mat].n_elements; ++i) {
          const double lo = (i == 0) ? 0.0 : double(mx.cumulative[i - 1]);
          const double pi = (double(mx.cumulative[i]) - lo) / double(mx.total);
          const double ex = pi * ran;
          if (ex > 5) {
            chi2 += (by_elem[i] - ex) * (by_elem[i] - ex) / ex;
            ++dof;
          } else if (pi == 0 && by_elem[i] != 0) {
            fail(std::string(c.name) + ": an element with zero partial cross section was drawn "
                 + std::to_string(by_elem[i]) + " times");
          }
        }
        std::printf("   %-26s ran %4d refused %3d | alive %4d killed %4d no-secondary %4d | "
                    "re-entries %d | A/Z unbalanced %d | worst |dE| %.3g MeV | target chi2 %.1f/%d\n",
                    c.name, ran, refused, alive, killed, no_sec, multi, bad_ab, worst_de, chi2,
                    dof > 1 ? dof - 1 : 0);
        for (const auto& kv : refusal_counts) {
          std::printf("        refused: %s x%d\n",
                      ee::refusal_name(static_cast<ee::EmExtraRefusal>(kv.first)), kv.second);
        }
        if (ran < c.n * 9 / 10) { fail(std::string(c.name) + ": fewer than 90% ran"); }
        // A re-entry is Geant4's own loop and harmless for a lepton, whose single model takes
        // the range manager's no-draw shortcut; for a PHOTON it re-runs a choice that draws in
        // the overlap windows (run_arm_emextra's note), so these photon cases - all outside
        // them - must show none for that note to rest on a measurement.
        if (c.t == ParticleType::kGamma && multi != 0) {
          fail(std::string(c.name) + ": CheckResult re-entered on a photon");
        }
        if (bad_ab != 0) { fail(std::string(c.name) + ": baryon number or charge not conserved"); }
        if (worst_de > 1.0) { fail(std::string(c.name) + ": energy not conserved to 1 MeV"); }
        if (dof > 1 && chi2 > 3.0 * (dof - 1) + 25.0) {
          fail(std::string(c.name) + ": the target draw does not follow the partial sums");
        }
        if (c.t == ParticleType::kGamma && alive != 0) {
          fail(std::string(c.name) + ": G4LowEGammaNuclearModel always kills the photon");
        }
        if (c.t != ParticleType::kGamma && killed != 0) {
          fail(std::string(c.name) + ": a lepto-nuclear model killed its lepton");
        }
        if (e_pre < c.e) {
          int nz = 0;
          for (int i = 0; i < mats[c.mat].n_elements; ++i) {
            if (by_elem[i] > 0) { ++nz; }
            if (by_elem[i] > 0 && static_cast<int>(mats[c.mat].z[i] + 0.5) != one_elem_z) {
              fail(std::string(c.name) + ": a target other than Z = " + std::to_string(one_elem_z)
                   + " was drawn - the partial sums are not the pre-step energy's");
            }
          }
          if (nz != 1) { fail(std::string(c.name) + ": expected one target element and only one"); }
        }
        if (c.t == ParticleType::kMuonPlus && c.e < 563.4) {
          // Below `epmax <= CutFixed` the muon model returns the muon untouched: no secondaries,
          // alive, same energy.
          if (no_sec != ran || alive != ran) {
            fail(std::string(c.name) + ": below T = 563.5 MeV the muon model must return the "
                 "muon unchanged");
          }
        }
      }

      // THE ISOTOPE DRAW, which the element chi-square above cannot see: `G4GammaNuclearXS::
      // SelectIsotope` weights the abundances by `IsoCrossSection` below 150 MeV and uses the
      // abundances alone above it, and the two lepton data sets always use the abundances alone
      // (they override neither `SelectIsotope` nor `GetIsoCrossSection`). 400,000 draws of
      // P2's `SampleZandA` through this package's functor per case, against those weights
      // computed here from the same isotope table, element by element.
      {
        struct IsoCase { had::EmExtraProcess p; real_t e; int mat; bool xs_weighted; const char* name; };
        const IsoCase ics[] = {
            {had::EmExtraProcess::kPhotonNuclear, 20, data::kBoneCompact, true, "gamma 20 MeV, bone"},
            {had::EmExtraProcess::kPhotonNuclear, 200, data::kBoneCompact, false, "gamma 200 MeV, bone"},
            {had::EmExtraProcess::kElectronNuclear, 300, data::kBoneCompact, false, "e- 300 MeV, bone"},
        };
        for (const IsoCase& ic : ics) {
          hxs::MaterialXs<real_t> mx{};
          (void)had::emextra_xs_per_volume<real_t>(tables, ic.p, ic.mat, mats[ic.mat], ic.e, mx);
          const had::EmExtraXsFn<real_t> fn = had::emextra_xs_fn<real_t>(tables, ic.p, ic.e);
          std::map<std::pair<int, int>, long long> count;
          std::map<int, long long> by_z;
          Lcg g;
          constexpr long long kDraws = 400000;
          for (long long k = 0; k < kDraws; ++k) {
            const hxs::TargetZA t = hxs::store_sample_za_rng<real_t>(
                fn, mats[ic.mat], hxs::nist_isotopes_of<real_t>(mats[ic.mat]), mx, g);
            ++count[{t.z, t.a}];
            ++by_z[t.z];
          }
          double worst = 0;
          std::string where;
          for (int i = 0; i < mats[ic.mat].n_elements; ++i) {
            const int z = static_cast<int>(mats[ic.mat].z[i] + 0.5);
            const hxs::ElementIsotopes<real_t> iso = hxs::nist_element_isotopes<real_t>(z);
            if (iso.n <= 1 || by_z[z] < 1000) { continue; }
            double wsum = 0;
            std::vector<double> w(iso.n);
            for (int j = 0; j < iso.n; ++j) {
              w[j] = double(iso.abundance[j])
                     * (ic.xs_weighted
                            ? double(hxs::pxs_iso_xs<real_t>(ht.gamma, ic.e, std::log(ic.e), z,
                                                              iso.a[j]).value)
                            : 1.0);
              wsum += w[j];
            }
            for (int j = 0; j < iso.n; ++j) {
              const double pj = w[j] / wsum;
              const double n = double(by_z[z]);
              const double se = std::sqrt(std::fmax(pj * (1 - pj), 1e-12) / n);
              const double dev = std::fabs(double(count[{z, iso.a[j]}]) / n - pj) / se;
              if (dev > worst) {
                worst = dev;
                where = "Z=" + std::to_string(z) + " A=" + std::to_string(iso.a[j]);
              }
            }
          }
          std::printf("   isotope draw, %-20s worst %.2f sigma (%s)\n", ic.name, worst,
                      where.c_str());
          if (worst > 5.0) {
            fail(std::string("isotope draw, ") + ic.name + ": " + where + " at "
                 + std::to_string(worst) + " sigma from the "
                 + (ic.xs_weighted ? "abundance x IsoCrossSection" : "abundance")
                 + " weights");
          }
        }
      }

      // The refusal names. A 10 GeV photon's range manager chooses QGS (E > 6 GeV: one window
      // only), and a 30 GeV electron's equivalent photon exceeds 10 GeV some of the time.
      {
        hp::HadProjectile<real_t> g;
        g.pdg = 22;
        g.kin_energy = 10000;
        int qgs = 0;
        for (int k = 0; k < 20; ++k) {
          Philox<real_t> rng(4242u + k, 1u, had::kInteractionRngPurpose);
          const had::InteractionOutcome o = had::run_emextra<real_t, had::InteractionBucket::kPhotoNuclear>(
              g, ParticleType::kGamma, mats[data::kWater], data::kWater, tables, g.kin_energy,
              *slot, lt, fpool, rng, nullptr);
          if (o.refusal == had::HadronicRefusal::kPhotoNuclearQgs && !o.ran) { ++qgs; }
        }
        hp::HadProjectile<real_t> e;
        e.pdg = 11;
        e.mass = particle_def<real_t>(ParticleType::kElectron).mass;
        e.charge = -1;
        e.kin_energy = 30000;
        int ftf = 0, eran = 0, other = 0;
        for (int k = 0; k < 200; ++k) {
          Philox<real_t> rng(5151u + k, 1u, had::kInteractionRngPurpose);
          const had::InteractionOutcome o = had::run_emextra<real_t, had::InteractionBucket::kLeptoNuclear>(
              e, ParticleType::kElectron, mats[data::kWater], data::kWater, tables, e.kin_energy,
              *slot, lt, fpool, rng, nullptr);
          if (o.ran) { ++eran; }
          else if (o.refusal == had::HadronicRefusal::kLeptoNuclearFtf) { ++ftf; }
          else { ++other; }
        }
        std::printf("   refusal names: 10 GeV photon -> kPhotoNuclearQgs %d/20; 30 GeV e- -> ran %d, "
                    "kLeptoNuclearFtf %d, other %d of 200\n", qgs, eran, ftf, other);
        if (qgs != 20) { fail("a 10 GeV photon did not refuse by name as kPhotoNuclearQgs"); }
        if (ftf == 0) { fail("a 30 GeV electron never reached the FTF arm's refusal"); }
        if (other != 0) { fail("a 30 GeV electron was refused under an unexpected name"); }
      }
      delete slot;
    }
  }

  // ============================================================================================
  // 7. The ledger
  // ============================================================================================
  std::printf("== 7. the ledger names ==\n");
  {
    using R = had::HadronicRefusal;
    const R size_rows[] = {R::kNeutronInelastic, R::kChargedHadronInelastic, R::kPhotoNuclear,
                           R::kLeptoNuclear};
    for (const R r : size_rows) {
      if (std::string(had::hadronic_refusal_name(r)).find("[SIZE") == std::string::npos) {
        fail(std::string("a SIZE row is not marked as one: ") + had::hadronic_refusal_name(r));
      }
    }
    for (const R r : {R::kPhotoNuclearQgs, R::kLeptoNuclearFtf, R::kEmExtraRefused}) {
      if (std::string(had::hadronic_refusal_name(r)).rfind("WHY:", 0) != 0) {
        fail(std::string("a WHY row is not marked as one: ") + had::hadronic_refusal_name(r));
      }
    }
    // Appended, so no P15 row moved: kRefusedEnergyScored is still the one before them.
    if (static_cast<int>(R::kPhotoNuclear) != static_cast<int>(R::kRefusedEnergyScored) + 1) {
      fail("the P19 rows were not appended after kRefusedEnergyScored");
    }
  }

  if (g_fails == 0) {
    std::printf("test_emextra_wiring: ALL OK\n");
    return 0;
  }
  std::printf("test_emextra_wiring: %d FAILURE(S)\n", g_fails);
  return 1;
}
