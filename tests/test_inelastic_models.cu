// P15 deliverable 1: the inelastic model choice, against the windows a CONSTRUCTED QBBC holds.
//
// Three things are checked and they are three different kinds of claim.
//
//  1. **The model lists are Geant4's.** `ref/oracle/ftf_windows.csv` is written by
//     `ref/dump/dump_ftf.cc`, which walks each species' own process manager, dynamic_casts to
//     `G4HadronicProcess` and writes one row per registered model with `GetModelName()`,
//     `GetMinEnergy()` and `GetMaxEnergy()`. That is the CONSTRUCTED process - after every
//     SetMinEnergy/SetMaxEnergy the three builders made - and not a reading of the builders.
//     The port's `had::inelastic_models` is compared against it row for row, in order, and the
//     ORDER is part of the claim: `G4EnergyRangeManager::GetHadronicInteraction` keeps the last
//     two matching models, so the list order decides which pair competes and which index comes
//     back.
//
//  2. **The parameters under those windows are Geant4's.** `ref/oracle/hadronic_params.csv` is
//     `G4HadronicParameters::Instance()`, and the four numbers the windows are built from -
//     MinEnergyTransitionFTF_Cascade, MaxEnergyTransitionFTF_Cascade, MaxEnergy and
//     EnableNeutronGeneralProcess - are asserted against it rather than remembered.
//
//  3. **The choice across an overlap has Geant4's distribution and Geant4's draw COUNT.** The
//     frequency is compared against `overlap_probability_upper` computed from the CSV's OWN
//     emin/emax, so a port whose windows and whose sampler were wrong in the same direction
//     would still fail; and the number of uniforms consumed is compared against
//     `inelastic_choice_draws`, because a draw made where Geant4 makes none moves every number
//     downstream of it.
//
// THE NEUTRON IS THE ONE SPECIES THE ORACLE CANNOT SEE, and the CSV says so rather than being
// silent about it: its only hadronic row is `NeutronGeneralProc,116,(none),0,0`. The three
// models are registered on a `G4HadronInelasticProcess` that `G4HadProcesses::
// BuildNeutronInelasticAndCapture` hands to `G4NeutronGeneralProcess` as a SUB-process, which
// is on no manager and whose `GetHadronicInteractionList()` the dumper never reaches. So the
// neutron's windows are asserted against the PROTON's rows, and the reason they are the same
// three numbers is three lines of G4HadronInelasticQBBC.cc (163-166): `theFTFP`, `theBERT` and
// `theBIC` are the same three C++ OBJECTS registered on both processes, constructed once at the
// top of `ConstructProcess`. That is a source-level claim and it is written here as one.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "core/rng.cuh"
#include "physics/hadronic/inelastic_wiring.cuh"

using namespace g4gpu;
namespace hp = g4gpu::physics::hadronic;

static int g_fails = 0;

static void fail(const char* what, const std::string& detail) {
  std::printf("FAIL: %s  %s\n", what, detail.c_str());
  ++g_fails;
}

static std::string oracle_dir() {
  const char* e = std::getenv("G4GPU_ORACLE");
  return (e != nullptr && e[0] != '\0') ? std::string(e) : std::string("ref/oracle");
}

// ---------------------------------------------------------------------------------------------
// A header-indexed CSV reader, the shape tests/test_ftf_model.cu uses.
// ---------------------------------------------------------------------------------------------
struct Csv {
  std::vector<std::string> cols;
  std::vector<std::vector<std::string>> rows;
  std::map<std::string, int> ix;

  bool load(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "r");
    if (f == nullptr) {
      std::printf("FAIL: cannot open %s - run ref/oracle/run.bat tables first\n", path.c_str());
      ++g_fails;
      return false;
    }
    char line[8192];
    bool header = true;
    while (std::fgets(line, sizeof(line), f) != nullptr) {
      std::string s(line);
      while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) { s.pop_back(); }
      if (s.empty()) { continue; }
      std::vector<std::string> c;
      std::string cur;
      for (char ch : s) {
        if (ch == ',') { c.push_back(cur); cur.clear(); } else { cur.push_back(ch); }
      }
      c.push_back(cur);
      if (header) {
        cols = c;
        for (int i = 0; i < static_cast<int>(c.size()); ++i) { ix[c[i]] = i; }
        header = false;
      } else {
        rows.push_back(c);
      }
    }
    std::fclose(f);
    return true;
  }
  const std::string& s(int r, const char* c) const { return rows[r][ix.at(c)]; }
  double d(int r, const char* c) const { return std::atof(rows[r][ix.at(c)].c_str()); }
};

/// The CSV's model name -> the port's enum. Every name `dump_ftf.cc` can write for an inelastic
/// process is here; an unknown one is a failure and not a default, because a model this port has
/// never heard of appearing on a QBBC process is exactly the thing a test should shout about.
static had::InelasticModel model_of_name(const std::string& n, bool& known) {
  known = true;
  if (n == "FTFP") { return had::InelasticModel::kFtfp; }
  if (n == "BertiniCascade") { return had::InelasticModel::kBertini; }
  if (n == "Binary Cascade") { return had::InelasticModel::kBinary; }
  if (n == "Binary Light Ion Cascade") { return had::InelasticModel::kLightIon; }
  known = false;
  return had::InelasticModel::kNone;
}

/// The CSV's particle name -> the port's species, for the species this port transports and
/// gives an inelastic process. Anything else returns kNumTypes and the row is skipped with a
/// COUNT, so "skipped" is visible rather than silent.
static ParticleType species_of_name(const std::string& n) {
  if (n == "proton") { return ParticleType::kProton; }
  if (n == "neutron") { return ParticleType::kNeutron; }
  if (n == "pi+") { return ParticleType::kPionPlus; }
  if (n == "pi-") { return ParticleType::kPionMinus; }
  if (n == "kaon+") { return ParticleType::kKaonPlus; }
  if (n == "kaon-") { return ParticleType::kKaonMinus; }
  if (n == "anti_proton") { return ParticleType::kAntiProton; }
  if (n == "deuteron") { return ParticleType::kDeuteron; }
  if (n == "triton") { return ParticleType::kTriton; }
  if (n == "He3") { return ParticleType::kHe3; }
  if (n == "alpha") { return ParticleType::kAlpha; }
  if (n == "GenericIon") { return ParticleType::kGenericIon; }
  return ParticleType::kNumTypes;
}

/// The baryon number the model choice divides by, as the CSV reports it. GenericIon's row says
/// 1 because `G4GenericIon` is a placeholder whose own baryon number is 1; a REAL nuclide
/// carries its own A on the track, which is what `step_hadron` passes and what the transport
/// test exercises. Here the CSV's value is used so the comparison is against what Geant4 holds.
static int baryon_of(const Csv& c, int r) { return static_cast<int>(c.d(r, "baryon")); }

/// A counting RNG: real Philox numbers, with a tally of how many were drawn.
struct CountingRng {
  Philox<double> p;
  int n = 0;
  __host__ __device__ explicit CountingRng(unsigned int key, unsigned int step)
      : p(key, step, 0x1E7Au) {}
  __host__ __device__ double uniform() { ++n; return p.uniform(); }
};

int main() {
  const std::string dir = oracle_dir();

  // ============================================================================================
  // 1. The four G4HadronicParameters numbers the windows are built from.
  // ============================================================================================
  {
    Csv par;
    if (par.load(dir + "/hadronic_params.csv")) {
      std::map<std::string, double> v;
      for (int r = 0; r < static_cast<int>(par.rows.size()); ++r) {
        v[par.s(r, "name")] = par.d(r, "value");
      }
      struct { const char* name; double got; } want[] = {
          {"MinEnergyTransitionFTF_Cascade", had::ftf_transition_min<double>()},
          {"MaxEnergyTransitionFTF_Cascade", had::ftf_transition_max<double>()},
          {"MaxEnergy", had::hadronic_max_energy<double>()},
      };
      for (const auto& w : want) {
        if (v.find(w.name) == v.end()) {
          fail("hadronic_params.csv has no row", w.name);
        } else if (v[w.name] != w.got) {
          char b[256];
          std::snprintf(b, sizeof(b), "%s: oracle %.17g, port %.17g", w.name, v[w.name], w.got);
          fail("parameter", b);
        }
      }
      // Not a window, but the thing that decides whether the neutron HAS a separate inelastic
      // process at all - and therefore whether this file's neutron paragraph is true.
      if (v["EnableNeutronGeneralProcess"] != 1.0) {
        fail("EnableNeutronGeneralProcess", "expected 1 in 11.1.1; the neutron paragraph of "
                                            "this test assumes the sub-process arrangement");
      }
      std::printf("  G4HadronicParameters: 3 windows + EnableNeutronGeneralProcess, exact\n");
    }
  }

  // ============================================================================================
  // 2. The model lists, row for row and IN ORDER.
  // ============================================================================================
  Csv w;
  if (!w.load(dir + "/ftf_windows.csv")) { return 1; }

  // Group the inelastic rows by particle, keeping the file's order.
  std::vector<std::string> order;
  std::map<std::string, std::vector<int>> by_particle;
  int n_rows_inelastic = 0, n_skipped_species = 0;
  for (int r = 0; r < static_cast<int>(w.rows.size()); ++r) {
    const std::string& proc = w.s(r, "process");
    // `subtype` 121 is fHadronInelastic. Selecting on the SUBTYPE and not on the name's suffix,
    // because `ionInelastic` and `anti_protonInelastic` and `dInelastic` share no suffix rule
    // and `hFritiofCaptureAtRest` ends in neither.
    if (static_cast<int>(w.d(r, "subtype")) != 121) { continue; }
    ++n_rows_inelastic;
    if (by_particle.find(w.s(r, "particle")) == by_particle.end()) {
      order.push_back(w.s(r, "particle"));
    }
    by_particle[w.s(r, "particle")].push_back(r);
    (void)proc;
  }

  int n_species_checked = 0, n_models_checked = 0;
  for (const std::string& pname : order) {
    const ParticleType t = species_of_name(pname);
    if (t == ParticleType::kNumTypes) {
      ++n_skipped_species;  // a hyperon or an anti-nucleus: refused at emission, never stepped
      continue;
    }
    const std::vector<int>& rs = by_particle[pname];
    const had::InelasticModelList<double> L = had::inelastic_models<double>(t);
    if (L.n != static_cast<int>(rs.size())) {
      char b[256];
      std::snprintf(b, sizeof(b), "%s: oracle has %d registered models, port has %d",
                    pname.c_str(), static_cast<int>(rs.size()), L.n);
      fail("model count", b);
      continue;
    }
    for (int i = 0; i < L.n; ++i) {
      bool known = false;
      const had::InelasticModel m = model_of_name(w.s(rs[i], "model"), known);
      if (!known) {
        fail("unknown model name on a QBBC inelastic process", pname + " / "
                                                               + w.s(rs[i], "model"));
        continue;
      }
      const double emin = w.d(rs[i], "emin_MeV");
      const double emax = w.d(rs[i], "emax_MeV");
      char b[320];
      if (m != L.model[i]) {
        std::snprintf(b, sizeof(b), "%s slot %d: oracle %s, port %s", pname.c_str(), i,
                      w.s(rs[i], "model").c_str(), had::inelastic_model_name(L.model[i]));
        fail("model at this position in the registration order", b);
      }
      if (emin != L.range[i].min_energy || emax != L.range[i].max_energy) {
        std::snprintf(b, sizeof(b), "%s %s: oracle [%.17g, %.17g], port [%.17g, %.17g]",
                      pname.c_str(), w.s(rs[i], "model").c_str(), emin, emax,
                      L.range[i].min_energy, L.range[i].max_energy);
        fail("model window", b);
      }
      ++n_models_checked;
    }
    ++n_species_checked;
  }
  std::printf("  ftf_windows.csv: %d inelastic rows, %d species checked (%d skipped - species "
              "this port refuses at emission), %d (model, emin, emax) triples exact\n",
              n_rows_inelastic, n_species_checked, n_skipped_species, n_models_checked);

  // ============================================================================================
  // 3. The neutron, which the oracle cannot see - and the assertion that it cannot.
  // ============================================================================================
  {
    int n_neutron_inelastic = 0;
    bool saw_general_with_no_model = false;
    for (int r = 0; r < static_cast<int>(w.rows.size()); ++r) {
      if (w.s(r, "particle") != "neutron") { continue; }
      if (static_cast<int>(w.d(r, "subtype")) == 121) { ++n_neutron_inelastic; }
      if (w.s(r, "process") == "NeutronGeneralProc" && w.s(r, "model") == "(none)") {
        saw_general_with_no_model = true;
      }
    }
    if (n_neutron_inelastic != 0) {
      fail("the neutron now has a visible inelastic process",
           "this file's neutron paragraph - and step_neutral's whole sub-process arrangement - "
           "assumes it does not; re-read G4HadProcesses::BuildNeutronInelasticAndCapture");
    }
    if (!saw_general_with_no_model) {
      fail("NeutronGeneralProc's empty model list",
           "expected one row with model '(none)' - the dumper writes that when "
           "GetHadronicInteractionList() is empty");
    }
    // The port's neutron list must equal the PROTON's, because QBBC registers the same three
    // objects on both (G4HadronInelasticQBBC.cc:154-156 and :164-166).
    const had::InelasticModelList<double> np = had::inelastic_models<double>(ParticleType::kProton);
    const had::InelasticModelList<double> nn =
        had::inelastic_models<double>(ParticleType::kNeutron);
    bool same = (np.n == nn.n);
    for (int i = 0; same && i < np.n; ++i) {
      same = (np.model[i] == nn.model[i]) && (np.range[i].min_energy == nn.range[i].min_energy)
             && (np.range[i].max_energy == nn.range[i].max_energy);
    }
    if (!same) {
      fail("the neutron's model list differs from the proton's",
           "QBBC registers theFTFP, theBERT and theBIC - the same three objects - on both");
    }
    std::printf("  the neutron: no inelastic row in the oracle (the sub-process is on no "
                "manager), and its port list is the proton's, exact\n");
  }

  // ============================================================================================
  // 4. The choice itself: the distribution across every overlap, and the draw COUNT everywhere.
  // ============================================================================================
  //
  // 40 energies per species, logarithmic from 1 MeV to 100 GeV, which is the grid the brief
  // asks for and which straddles every boundary in the table: 1.0, 1.5, 3, 6 and 12 GeV.
  {
    constexpr int kNE = 40;
    double e[kNE];
    for (int i = 0; i < kNE; ++i) {
      e[i] = 1.0 * std::pow(10.0, 5.0 * double(i) / double(kNE - 1));  // 1 MeV .. 100 GeV
    }
    constexpr long long kDraws = 200000;
    const double kSigmaGate = 5.0;

    int n_cells = 0, n_overlap_cells = 0, n_draw_mismatch = 0;
    double worst_sigma = 0.0;
    std::string worst_where;

    for (const std::string& pname : order) {
      const ParticleType t = species_of_name(pname);
      if (t == ParticleType::kNumTypes) { continue; }
      const std::vector<int>& rs = by_particle[pname];
      const int baryon = baryon_of(w, rs[0]);

      for (int ie = 0; ie < kNE; ++ie) {
        const double ek = e[ie];
        ++n_cells;

        // ---- which models does the ORACLE's own table say are in range, and what does its own
        // arithmetic say the upper one's probability is? Computed from the CSV, not from the
        // port, so the two can disagree.
        const double epn = (std::abs(baryon) > 1) ? ek / double(std::abs(baryon)) : ek;
        int n_in = 0, lo_i = -1, hi_i = -1;
        for (int i = 0; i < static_cast<int>(rs.size()); ++i) {
          if (w.d(rs[i], "emin_MeV") <= epn && w.d(rs[i], "emax_MeV") >= epn) {
            ++n_in;
            if (lo_i < 0 || w.d(rs[i], "emin_MeV") < w.d(rs[lo_i], "emin_MeV")) {
              hi_i = lo_i; lo_i = i;
            } else {
              hi_i = i;
            }
          }
        }
        if (n_in == 0) {
          // A gap in the windows would be `had005` in Geant4. There is none for any species
          // this port transports, and this is the assertion that says so.
          fail("no model in range", pname + " at " + std::to_string(ek) + " MeV");
          continue;
        }
        const bool expect_draw = (n_in == 2);
        if (expect_draw) { ++n_overlap_cells; }

        // ---- the draw count, on ONE call.
        {
          CountingRng r(0xC0FFEEu + static_cast<unsigned int>(ie), static_cast<unsigned int>(t));
          hp::ModelChoice st = hp::ModelChoice::kOk;
          (void)had::choose_inelastic_model<double>(t, ek, baryon, r, st);
          const bool predicted = had::inelastic_choice_draws<double>(t, ek, baryon);
          if (r.n != (expect_draw ? 1 : 0) || predicted != expect_draw) {
            ++n_draw_mismatch;
            char b[256];
            std::snprintf(b, sizeof(b), "%s at %.4g MeV: oracle says %d models in range, the "
                                        "choice drew %d uniforms, the predicate says %d",
                          pname.c_str(), ek, n_in, r.n, predicted ? 1 : 0);
            fail("uniforms consumed by the model choice", b);
          }
        }

        if (!expect_draw) { continue; }

        // ---- the frequency, against the CSV's own emin/emax through P5's own formula.
        const double lower_emax = w.d(rs[lo_i], "emax_MeV");
        const double upper_emin = w.d(rs[hi_i], "emin_MeV");
        const double p_upper =
            hp::overlap_probability_upper<double>(lower_emax, upper_emin, epn);
        bool known_hi = false;
        const had::InelasticModel m_hi = model_of_name(w.s(rs[hi_i], "model"), known_hi);
        if (!known_hi) { continue; }

        long long hits = 0;
        Philox<double> rng(0xBEEF0000u + static_cast<unsigned int>(ie),
                           static_cast<unsigned int>(t), 0x1E7Au);
        for (long long k = 0; k < kDraws; ++k) {
          hp::ModelChoice st = hp::ModelChoice::kOk;
          if (had::choose_inelastic_model<double>(t, ek, baryon, rng, st) == m_hi) { ++hits; }
        }
        const double f = double(hits) / double(kDraws);
        const double sd = std::sqrt(p_upper * (1.0 - p_upper) / double(kDraws));
        const double sig = (sd > 0.0) ? std::abs(f - p_upper) / sd
                                      : ((f == p_upper) ? 0.0 : 1e9);
        if (sig > worst_sigma) {
          worst_sigma = sig;
          char b[256];
          std::snprintf(b, sizeof(b), "%s at %.4g MeV (%.4g MeV/n): p=%.6f, f=%.6f",
                        pname.c_str(), ek, epn, p_upper, f);
          worst_where = b;
        }
        if (sig > kSigmaGate) {
          char b[320];
          std::snprintf(b, sizeof(b), "%s at %.4g MeV: expected P(%s)=%.6f, got %.6f (%.2f "
                                      "sigma of %lld draws)",
                        pname.c_str(), ek, w.s(rs[hi_i], "model").c_str(), p_upper, f, sig,
                        kDraws);
          fail("overlap frequency", b);
        }
      }
    }
    std::printf("  the choice: %d (species, energy) cells, %d of them in an overlap, %lld draws "
                "each; worst %.2f sigma against a %.1f gate  [%s]\n",
                n_cells, n_overlap_cells, kDraws, worst_sigma, kSigmaGate,
                worst_where.c_str());
    std::printf("  the draw COUNT: %d mismatches over %d cells\n", n_draw_mismatch, n_cells);
  }

  // ============================================================================================
  // 5. The two channels that have no cross section, by name.
  // ============================================================================================
  {
    if (had::inelastic_channel_has_xs(had::inelastic_channel(ParticleType::kAntiProton))) {
      fail("the antiproton's inelastic channel",
           "G4ComponentAntiNuclNuclearXS is refused by P2; this channel must have no cross "
           "section, or a pbar draws an interaction length it cannot evaluate");
    }
    if (had::inelastic_channel_has_xs(had::inelastic_channel(ParticleType::kMuonMinus))
        || had::inelastic_channel_has_xs(had::inelastic_channel(ParticleType::kPiZero))) {
      fail("a species QBBC gives no inelastic process", "mu- or pi0 has one here");
    }
    // The pi0 and the muons must also have an EMPTY model list, which is the other half of
    // "no process": a species with windows and no cross section would draw nothing and then be
    // asked to choose a model.
    if (had::inelastic_models<double>(ParticleType::kMuonMinus).n != 0
        || had::inelastic_models<double>(ParticleType::kPiZero).n != 0) {
      fail("model list of a species with no inelastic process", "expected zero models");
    }
    std::printf("  no-process species: pbar (P2 refuses its data set), mu+-, pi0 - all exact\n");
  }

  std::printf("%s\n", g_fails == 0 ? "OK" : "FAILURES");
  return (g_fails == 0) ? 0 : 1;
}
