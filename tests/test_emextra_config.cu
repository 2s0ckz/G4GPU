// QBBC's G4EmExtraPhysics configuration, against what the running physics list actually holds.
//
// `src/physics/hadronic/emextra/config.cuh` is a transcription of a CONFIGURATION, not of a
// formula, and a transcription of a configuration fails in a way a numeric tolerance cannot
// see: a window off by a MeV, a model on the wrong particle, a flag read from a header rather
// than from the run. So every row is compared in BOTH directions against
// `ref/oracle/emextra_*.csv` - a process the port claims and the run does not have fails, and a
// process the run has and the port does not know about fails too. The second direction is the
// one that matters: it is what would catch a synchrotron process appearing because someone
// changed a default.
//
//   emextra_config.csv    every process on gamma, e-, e+, mu-, mu+, from the process manager
//                         AND from G4HadronicProcessStore (the gamma's photonNuclear is inside
//                         G4GammaGeneralProcess and is absent from the manager)
//   emextra_windows.csv   the G4EnergyRangeManager rows: model name, index, [emin, emax], and
//                         the high-energy generator inside a G4TheoFSGenerator
//   emextra_xs.csv        the cross-section data sets, in store order
//   emextra_params.csv    the G4HadronicParameters the photon's windows are built from
//
// And one thing no CSV can state, because it is a distribution: the model choice in the two
// overlaps. `choose_hadronic_interaction` is P5's and is already validated there; what is new
// here is that the photon has THREE models with TWO overlaps of very different widths, and that
// the upper one selects an unported model. Both are counted against the closed form.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "physics/hadronic/emextra/config.cuh"

using namespace g4gpu;
using namespace g4gpu::physics::hadronic;
namespace ee = g4gpu::physics::hadronic::emextra;

namespace {

int fails = 0;

void fail(const std::string& msg) {
  std::printf("FAIL: %s\n", msg.c_str());
  ++fails;
}

std::string oracle_dir() {
  const char* e = std::getenv("G4GPU_ORACLE");
  return (e != nullptr && e[0] != '\0') ? std::string(e) : std::string("ref/oracle");
}

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

/// A CSV with a header, addressed by column name so a reordered dump cannot silently compare
/// the wrong column against the right one.
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

/// The five particles G4EmExtraPhysics touches, by name and PDG code.
struct Species { const char* name; int pdg; };
const Species kSpecies[] = {{"gamma", 22}, {"e-", 11}, {"e+", -11}, {"mu-", 13}, {"mu+", -13}};

/// Every process name QBBC gives one of the five for a reason OTHER than G4EmExtraPhysics.
/// A name that is in neither this set nor `emextra`'s own four is a process nobody expected.
const char* kOtherProcesses[] = {
    "Transportation", "GammaGeneralProc", "msc",  "eIoni",   "eBrem", "CoulombScat",
    "annihil",        "muIoni",           "muBrems", "muPairProd", "Decay",
    "muMinusCaptureAtRest"};

/// Every process G4EmExtraPhysics WOULD create if one of its eight off-by-default flags were
/// true. Each name is the G4VProcess name the class gives it, so that its ABSENCE from the
/// run is the measurement of the flag - see config.cuh's header.
struct OffProcess { const char* name; const char* flag; };
const OffProcess kOffProcesses[] = {
    {"SynRad", "synActivated = false"},
    {"GammaToMuPair", "gmumuActivated = false"},
    {"AnnihiToMuPair", "pmumuActivated = false"},
    {"AnnihiToTauPair", "pmumuActivated = false (the second instance)"},
    {"ee2hadr", "phadActivated = false"},
    {"muToMuonPairProd", "mmumuActivated = false"},
    {"neutrino-electron", "fNuActivated = false"},
    {"muon-nuclear-nucleus", "fNuActivated = false"},
    {"tau-neutrino-nucleus", "fNuActivated = false"},
    {"electron-neutrino-nucleus", "fNuActivated = false"},
};

/// A uniform generator with a definite, reproducible stream, for the overlap counting. The same
/// shape P5's own test uses.
struct Lcg {
  unsigned long long s = 88172645463325252ULL;
  __host__ __device__ double uniform() {
    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
    return double(s >> 11) * (1.0 / 9007199254740992.0);
  }
};

}  // namespace

int main() {
  const std::string dir = oracle_dir();

  // -------------------------------------------------------------------------------------------
  // 1. emextra_config.csv - which processes exist, on which particle, from which list
  // -------------------------------------------------------------------------------------------
  {
    Csv c;
    if (!c.load(dir + "/emextra_config.csv")) {
      fail("cannot read " + dir + "/emextra_config.csv");
    } else {
      // The set of (particle, process) pairs the RUN has, from either list.
      std::set<std::string> run_pairs;
      std::map<std::string, int> subtype_of;
      for (std::size_t r = 0; r < c.rows.size(); ++r) {
        const std::string key = c.s(r, "particle") + "|" + c.s(r, "process");
        run_pairs.insert(key);
        subtype_of[key] = c.i(r, "process_subtype");
      }

      // (a) Every process the port's config claims must be in the run, with subtype 121
      //     (fHadronInelastic) - which is what makes all four G4HadronInelasticProcess and
      //     therefore P5's framework's business.
      for (const Species& sp : kSpecies) {
        const ee::ProcessConfig cfg = ee::qbbc_config(sp.pdg);
        if (!cfg.ok) {
          fail(std::string("qbbc_config has no process for ") + sp.name);
          continue;
        }
        const std::string key = std::string(sp.name) + "|" + ee::process_name(cfg.process);
        if (run_pairs.count(key) == 0) {
          fail("the port claims " + key + " and the run does not have it");
        } else if (subtype_of[key] != 121) {
          fail(key + " has subtype " + std::to_string(subtype_of[key]) + ", expected 121");
        }
      }

      // (b) Every process the run has must be one the port knows about, or one of the EM,
      //     transport and decay processes that belong to other packages. Anything else is a
      //     process that appeared without anyone noticing.
      std::set<std::string> known;
      for (const char* n : kOtherProcesses) { known.insert(n); }
      known.insert("photonNuclear");
      known.insert("electronNuclear");
      known.insert("positronNuclear");
      known.insert("muonNuclear");
      for (std::size_t r = 0; r < c.rows.size(); ++r) {
        const std::string pn = c.s(r, "process");
        if (known.count(pn) == 0) {
          fail("the run has process '" + pn + "' on " + c.s(r, "particle") +
               "', which no part of this port expects");
        }
      }

      // (c) The eight flags that are false. Each is measured by the absence of its process.
      for (const OffProcess& op : kOffProcesses) {
        for (std::size_t r = 0; r < c.rows.size(); ++r) {
          if (c.s(r, "process") == op.name) {
            fail(std::string("the run HAS ") + op.name + " - config.cuh says " + op.flag);
          }
        }
      }

      // (d) The gamma's photonNuclear is NOT on its process manager, and the two leptons' ARE.
      //     That asymmetry is G4GammaGeneralProcess existing and G4ElectronGeneralProcess not,
      //     and it is the single fact P15's wiring most needs to know.
      bool gamma_manager_has_photonuclear = false;
      bool gamma_manager_has_general = false;
      bool electron_manager_has_electronuclear = false;
      for (std::size_t r = 0; r < c.rows.size(); ++r) {
        if (c.s(r, "source") != "manager") { continue; }
        if (c.s(r, "particle") == "gamma" && c.s(r, "process") == "photonNuclear") {
          gamma_manager_has_photonuclear = true;
        }
        if (c.s(r, "particle") == "gamma" && c.s(r, "process") == "GammaGeneralProc") {
          gamma_manager_has_general = true;
        }
        if (c.s(r, "particle") == "e-" && c.s(r, "process") == "electronNuclear") {
          electron_manager_has_electronuclear = true;
        }
      }
      if (gamma_manager_has_photonuclear) {
        fail("photonNuclear is on the gamma's process manager - G4GammaGeneralProcess was "
             "expected to hold it");
      }
      if (!gamma_manager_has_general) {
        fail("the gamma has no GammaGeneralProc, so G4EmStandardPhysics did not set "
             "SetGeneralProcessActive(true) - every statement in config.cuh about where "
             "photonNuclear lives is then wrong");
      }
      if (!electron_manager_has_electronuclear) {
        fail("electronNuclear is NOT on the electron's process manager, so an electron general "
             "process exists after all");
      }
      std::printf("emextra_config.csv: %d rows, four processes, %d off-by-default processes "
                  "absent\n",
                  int(c.rows.size()), int(sizeof kOffProcesses / sizeof kOffProcesses[0]));
    }
  }

  // -------------------------------------------------------------------------------------------
  // 2. emextra_windows.csv - the model names, their order and their energy windows
  // -------------------------------------------------------------------------------------------
  {
    Csv c;
    if (!c.load(dir + "/emextra_windows.csv")) {
      fail("cannot read " + dir + "/emextra_windows.csv");
    } else {
      int n_rows = 0;
      std::set<std::string> seen;
      for (const Species& sp : kSpecies) {
        const ee::ProcessConfig cfg = ee::qbbc_config(sp.pdg);
        if (!cfg.ok) { continue; }
        int found = 0;
        for (std::size_t r = 0; r < c.rows.size(); ++r) {
          if (c.s(r, "particle") != sp.name) { continue; }
          if (c.s(r, "process") != ee::process_name(cfg.process)) { continue; }
          const int idx = c.i(r, "index");
          seen.insert(c.s(r, "particle") + "|" + std::to_string(idx));
          ++found;
          ++n_rows;
          if (idx < 0 || idx >= cfg.n_models) {
            fail(std::string(sp.name) + ": oracle has model index " + std::to_string(idx) +
                 " and the port has " + std::to_string(cfg.n_models) + " models");
            continue;
          }
          const ee::ModelWindow& w = cfg.models[idx];
          if (c.s(r, "model") != ee::model_name(w.model)) {
            fail(std::string(sp.name) + " model " + std::to_string(idx) + ": oracle '" +
                 c.s(r, "model") + "', port '" + ee::model_name(w.model) + "'");
          }
          if (c.d(r, "min_MeV") != w.emin_MeV) {
            std::printf("FAIL: %s model %d emin: oracle %.17g, port %.17g\n", sp.name, idx,
                        c.d(r, "min_MeV"), w.emin_MeV);
            ++fails;
          }
          if (c.d(r, "max_MeV") != w.emax_MeV) {
            std::printf("FAIL: %s model %d emax: oracle %.17g, port %.17g\n", sp.name, idx,
                        c.d(r, "max_MeV"), w.emax_MeV);
            ++fails;
          }
        }
        if (found != cfg.n_models) {
          fail(std::string(sp.name) + ": oracle has " + std::to_string(found) +
               " model rows, the port has " + std::to_string(cfg.n_models));
        }
      }

      // The finding this file exists for: the photon's high-energy generator is a QGS parton
      // string model and NOT FTF. The wrapper's own name is "TheoFSGenerator" either way, so
      // the only way to see it is the generator inside, and `G4QGSModel` takes
      // `G4VPartonStringModel`'s DEFAULT name because its constructor passes no argument.
      bool checked_heg = false;
      for (std::size_t r = 0; r < c.rows.size(); ++r) {
        if (c.s(r, "particle") != "gamma" || c.s(r, "model") != "TheoFSGenerator") { continue; }
        checked_heg = true;
        const std::string heg = c.s(r, "high_energy_generator");
        if (heg.empty()) {
          std::printf("NOTE: emextra_windows.csv has no high_energy_generator column - "
                      "regenerate the oracle to check QGS against FTF\n");
          checked_heg = false;
          break;
        }
        if (heg.find("FTF") != std::string::npos) {
          fail("the photon's high-energy generator reports '" + heg +
               "' - config.cuh refuses it as QGS and would be refusing the wrong model");
        }
        if (heg != "Parton String Model") {
          std::printf("NOTE: the photon's high-energy generator is '%s'; config.cuh expects "
                      "G4VPartonStringModel's default name for an unnamed G4QGSModel\n",
                      heg.c_str());
        }
      }
      if (!checked_heg) {
        std::printf("NOTE: the photon's TheoFSGenerator row carried no generator name\n");
      }
      std::printf("emextra_windows.csv: %d model rows compared exactly\n", n_rows);
    }
  }

  // -------------------------------------------------------------------------------------------
  // 3. emextra_xs.csv - one data set per process, by name
  // -------------------------------------------------------------------------------------------
  {
    Csv c;
    if (!c.load(dir + "/emextra_xs.csv")) {
      fail("cannot read " + dir + "/emextra_xs.csv");
    } else {
      for (const Species& sp : kSpecies) {
        const ee::ProcessConfig cfg = ee::qbbc_config(sp.pdg);
        if (!cfg.ok) { continue; }
        int found = 0;
        for (std::size_t r = 0; r < c.rows.size(); ++r) {
          if (c.s(r, "particle") != sp.name) { continue; }
          if (c.s(r, "process") != ee::process_name(cfg.process)) { continue; }
          ++found;
          if (c.s(r, "dataset") != ee::dataset_name(cfg.dataset)) {
            fail(std::string(sp.name) + ": oracle data set '" + c.s(r, "dataset") +
                 "', port '" + ee::dataset_name(cfg.dataset) + "'");
          }
          if (c.i(r, "index") != 0) {
            fail(std::string(sp.name) +
                 ": a second cross-section data set - G4CrossSectionDataStore walks the list "
                 "backwards and the port assumes there is only one");
          }
        }
        if (found != 1) {
          fail(std::string(sp.name) + ": " + std::to_string(found) +
               " data-set rows, expected exactly one");
        }
      }
      std::printf("emextra_xs.csv: one data set per process, all four names exact\n");
    }
  }

  // -------------------------------------------------------------------------------------------
  // 4. emextra_params.csv - the numbers the photon's windows are built from
  //
  // A window checked only against itself is a tautology: the port and the oracle would agree on
  // 6000 MeV whatever 6000 MeV meant. These rows check that the 6000 IS
  // GetMaxEnergyTransitionFTF_Cascade and the 1e8 IS GetMaxEnergy, so a Geant4 whose parameters
  // moved would fail here rather than pass everywhere.
  // -------------------------------------------------------------------------------------------
  {
    Csv c;
    if (!c.load(dir + "/emextra_params.csv")) {
      fail("cannot read " + dir + "/emextra_params.csv");
    } else {
      std::map<std::string, double> v;
      for (std::size_t r = 0; r < c.rows.size(); ++r) {
        v[c.rows[r][0]] = std::atof(c.rows[r][1].c_str());
      }
      auto chk = [&](const char* name, double want) {
        const auto it = v.find(name);
        if (it == v.end()) { fail(std::string(name) + " is not in emextra_params.csv"); return; }
        if (it->second != want) {
          std::printf("FAIL: %s: oracle %.17g, port %.17g\n", name, it->second, want);
          ++fails;
        }
      };
      chk("MinEnergyTransitionFTF_Cascade_MeV", ee::transition_min_MeV());
      chk("MaxEnergyTransitionFTF_Cascade_MeV", ee::transition_max_MeV());
      chk("MaxEnergy_MeV", ee::max_energy_MeV());
      // The photon's Bertini floor is the low-energy limit MINUS ONE MEV, and the low-energy
      // limit is a constructor literal with no getter. So it is checked the other way round:
      // the oracle's Bertini emin plus 1 MeV must be the port's fGNLowEnergyLimit, and the
      // GammaNPreco emax must be it exactly. Both come out of emextra_windows.csv above; what
      // is asserted here is that the two are one MeV apart, which is the seam itself.
      const double seam = ee::gn_low_energy_limit_MeV()
                          - ee::qbbc_config(22).models[1].emin_MeV;
      if (seam != 1.0) {
        std::printf("FAIL: the GammaNPreco/Bertini seam is %.17g MeV wide, expected 1\n", seam);
        ++fails;
      }
      // Two environment switches that must be OFF for the port's refusals to be the right ones.
      const auto pn = v.find("G4CASCADE_CHECK_PHOTONUCLEAR_set");
      if (pn != v.end() && pn->second != 0.0) {
        fail("G4CASCADE_CHECK_PHOTONUCLEAR is set on this install, so "
             "G4InuclCollider::photonuclearOkay RUNS and the port does not have it");
      }
      const auto ld = v.find("G4LENDDATA_set");
      if (ld != v.end() && ld->second != 0.0) {
        std::printf("NOTE: G4LENDDATA is set; LEND is still unreachable in QBBC because "
                    "gLENDActivated is false and the gamma general process exists\n");
      }
      // WHICH GAMMA CROSS SECTION THE TWO LEPTON MODELS FOUND.
      //
      // `G4ElectroVDNuclearModel`'s constructor asks the registry for "PhotoNuclearXS" and only
      // falls back to "GammaNuclearXS" if that is absent. It is NOT absent: `G4GammaNuclearXS`'s
      // own constructor asks for the same name, finds nothing, and does
      // `new G4PhotoNuclearCrossSection()` - whose base constructor registers it - and
      // `ConstructGammaElectroNuclear` builds the cross section before the model. So the
      // electron and positron acceptance test runs on the pure CHIPS parameterisation while the
      // photon process's own cross section is the IAEA-data one, and
      // `emextra/lepton_vd.cuh` calls `chips::photo_element_xs` for exactly that reason.
      //
      // Asserted rather than reasoned: if this row is ever 0, the model took the other branch
      // and that file is calling the wrong cross section.
      const auto pnx = v.find("registry_has_PhotoNuclearXS");
      if (pnx == v.end()) {
        std::printf("NOTE: emextra_params.csv has no registry rows - regenerate the oracle to "
                    "check which gamma cross section G4ElectroVDNuclearModel found\n");
      } else if (pnx->second == 0.0) {
        fail("G4PhotoNuclearCrossSection is NOT in the cross-section registry, so "
             "G4ElectroVDNuclearModel fell back to G4GammaNuclearXS and lepton_vd.cuh's "
             "acceptance test is calling the wrong class");
      }
      for (const char* n : {"registry_has_GammaNuclearXS", "registry_has_ElectroNuclearXS",
                            "registry_has_KokoulinMuonNuclearXS"}) {
        const auto it = v.find(n);
        if (it != v.end() && it->second == 0.0) {
          fail(std::string(n) + " is 0, so a data set emextra_xs.csv names is not registered");
        }
      }
      std::printf("emextra_params.csv: transition energies and both environment switches "
                  "checked\n");
    }
  }

  // -------------------------------------------------------------------------------------------
  // 5. The two overlaps, counted
  //
  // P5's `choose_hadronic_interaction` is validated in tests/test_hadronic_process.cu against
  // Geant4 itself; what is new here is the SHAPE of the photon's model list - three models, two
  // overlaps, one of them a single MeV wide and the other selecting a model this port refuses.
  // The counted frequency is compared against `overlap_probability_upper`, which is the closed
  // form written out separately, at 0.004 over 200,000 draws (about four standard errors of a
  // p = 0.5 binomial at that N).
  // -------------------------------------------------------------------------------------------
  {
    const ee::ProcessConfig cfg = ee::qbbc_config(22);
    ModelRange<double> ranges[3];
    const int n = ee::model_ranges<double>(cfg, ranges);
    if (n != 3) { fail("the photon does not have three models"); }

    struct Point { double e; int lower, upper; const char* what; };
    const Point pts[] = {
        {199.1,  0, 1, "GammaNPreco/Bertini seam, bottom"},
        {199.5,  0, 1, "GammaNPreco/Bertini seam, middle"},
        {199.9,  0, 1, "GammaNPreco/Bertini seam, top"},
        {3100.0, 1, 2, "Bertini/QGS overlap, bottom"},
        {4500.0, 1, 2, "Bertini/QGS overlap, middle"},
        {5000.0, 1, 2, "Bertini/QGS overlap, the campaign's top gamma energy"},
        {5900.0, 1, 2, "Bertini/QGS overlap, top"},
    };
    const long long N = 200000;
    long long compared = 0;
    double worst = 0.0;
    std::string worst_where;
    for (const Point& p : pts) {
      Lcg rng;
      long long upper = 0;
      for (long long k = 0; k < N; ++k) {
        const ModelSelection sel =
            choose_hadronic_interaction<double>(ranges, n, p.e, 0, rng);
        if (sel.status != ModelChoice::kOk) {
          fail(std::string("no model in range at ") + std::to_string(p.e) + " MeV");
          break;
        }
        if (sel.index == p.upper) { ++upper; }
        else if (sel.index != p.lower) {
          fail(std::string("a third model was chosen at ") + std::to_string(p.e) + " MeV");
          break;
        }
      }
      const double got = double(upper) / double(N);
      const double want = overlap_probability_upper<double>(
          cfg.models[p.lower].emax_MeV, cfg.models[p.upper].emin_MeV, p.e);
      const double diff = std::fabs(got - want);
      if (diff > worst) { worst = diff; worst_where = p.what; }
      ++compared;
      if (diff > 0.004) {
        std::printf("FAIL: %s: %.5f of draws chose the upper model, closed form %.5f\n", p.what,
                    got, want);
        ++fails;
      }
    }
    std::printf("model overlap: %lld points x %lld draws, worst |counted - closed form| = "
                "%.5f (%s)\n",
                compared, N, worst, worst_where.c_str());

    // Outside both overlaps exactly one model is in range, and which one is not a random
    // variable. Below 199 and above 200 the seam is closed; between 6000 and 100000 only the
    // unported generator remains, and a photon there is REFUSED - the rate is 1, not 2/3.
    struct Single { double e; int idx; };
    const Single singles[] = {{1.0, 0}, {150.0, 0}, {198.9, 0}, {201.0, 1}, {1000.0, 1},
                              {2999.0, 1}, {6001.0, 2}, {50000.0, 2}, {1.0e8, 2}};
    Lcg rng;
    for (const Single& s : singles) {
      const ModelSelection sel = choose_hadronic_interaction<double>(ranges, n, s.e, 0, rng);
      if (sel.status != ModelChoice::kOk || sel.index != s.idx) {
        std::printf("FAIL: at %.17g MeV the model is index %d (status %d), expected %d\n", s.e,
                    sel.index, int(sel.status), s.idx);
        ++fails;
      }
    }
    // And above 100 TeV there is NO model: G4EnergyRangeManager returns nullptr and the
    // process raises a FatalException. The port reports it rather than silently choosing the
    // nearest window.
    const ModelSelection over = choose_hadronic_interaction<double>(ranges, n, 2.0e8, 0, rng);
    if (over.status != ModelChoice::kNoModelInRange) {
      fail("a photon above G4HadronicParameters::GetMaxEnergy() found a model");
    }
    std::printf("single-model energies: %d points, and 200 TeV correctly has no model\n",
                int(sizeof singles / sizeof singles[0]));
  }

  if (fails == 0) {
    std::printf("\ntest_emextra_config: OK\n");
    return 0;
  }
  std::printf("\ntest_emextra_config: %d FAILURES\n", fails);
  return 1;
}
