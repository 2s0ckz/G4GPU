// P23b - the strange projectiles and their data, against Geant4 11.1.1 as QBBC builds it.
//
// Three blocks, each against its own oracle file from ref/dump/dump_strange.cc. (The fourth
// file, strange_k0type.csv - the two neutral-kaon coin tosses of G4CascadeInterface - is read by
// tests/test_bertini_apply.cu, beside the interface it is about.)
//
//   1  the scope      strange_scope.csv, the processes and models QBBC really registered on
//                     every kaon, hyperon, anti-hyperon and b/c hadron. Printed as the table
//                     docs/RISK.md V226 is built on, and asserted where the port holds the same
//                     fact: Bertini's window per species (`qbbc_bertini_range`), Bertini's own
//                     IsApplicable (`interface_is_applicable`), and whether each species' decay
//                     table is transcribed or refused BY NAME.
//   2  the cross      strange_hnxsc.csv - G4HadronNucleonXsc's dispatcher, every branch these
//      sections       species reach, on both nucleons, 1 keV to 100 TeV;
//                     strange_ggxs.csv - G4ComponentGGHadronNucleusXsc, all five outputs;
//                     strange_scope_xs.csv - the REGISTERED processes' cross sections per atom
//                     beside the bare component's, which is what proves the processes carry
//                     the Glauber-Gribov component and nothing else, and the port against both.
//   3  heavy flavour  strange_lund.csv and ftf_modelstat_species.csv - that QBBC's own FTFP
//                     CAN make b/c hadrons, and at what rate it does in the oracle's events.
//
// The cross-section tolerance is test_hadronic_xs.cu's 1e-12 relative, for its reason: nothing
// stands between the port and Geant4 but double arithmetic in the same order.
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "data/ftf_hadrons.hh"
#include "data/materials.cuh"
#include "physics/decay/decay_tables.hh"
#include "physics/hadronic/bertini/cascade_interface.cuh"
#include "physics/hadronic/xs/gg_hadron_nucleus_xsc.cuh"
#include "physics/hadronic/xs/hadron_nucleon_xsc.cuh"

using namespace g4gpu;

namespace {

constexpr double kTol = 1e-12;
int g_fails = 0;

void fail(const char* fmt, ...) {
  ++g_fails;
  va_list ap;
  va_start(ap, fmt);
  std::printf("  FAIL: ");
  std::vprintf(fmt, ap);
  std::printf("\n");
  va_end(ap);
}

/// A CSV keyed by column NAME, as tests/test_decay.cu reads its files.
struct Csv {
  std::vector<std::string> header;
  std::vector<std::vector<std::string>> rows;
  int col(const char* name) const {
    for (std::size_t i = 0; i < header.size(); ++i) {
      if (header[i] == name) { return static_cast<int>(i); }
    }
    return -1;
  }
  const std::string& s(std::size_t r, const char* name) const {
    static const std::string empty;
    const int c = col(name);
    return (c < 0) ? empty : rows[r][static_cast<std::size_t>(c)];
  }
  double d(std::size_t r, const char* name) const { return std::atof(s(r, name).c_str()); }
  long long i(std::size_t r, const char* name) const { return std::atoll(s(r, name).c_str()); }
};

std::vector<std::string> split(const std::string& line) {
  std::vector<std::string> out;
  std::string cur;
  for (char c : line) {
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

bool load(const std::string& path, Csv& out) {
  FILE* f = std::fopen(path.c_str(), "r");
  if (f == nullptr) {
    fail("cannot read %s - run ref/dump/build.bat then ref/oracle/run.bat tables", path.c_str());
    return false;
  }
  std::string line;
  char buf[8192];
  bool first = true;
  while (std::fgets(buf, sizeof buf, f) != nullptr) {
    line = buf;
    if (first) {
      out.header = split(line);
      first = false;
      continue;
    }
    std::vector<std::string> r = split(line);
    if (r.size() == out.header.size()) { out.rows.push_back(r); }
  }
  std::fclose(f);
  if (out.rows.empty()) { fail("%s has no rows", path.c_str()); }
  return !out.rows.empty();
}

/// Relative deviation with an absolute floor for genuinely-zero columns, as P2's test does.
double deviation(double ours, double g4) {
  const double floor = 1e-300;
  if (std::fabs(g4) > floor) { return std::fabs(ours - g4) / std::fabs(g4); }
  return (std::fabs(ours) > floor) ? 1.0 : 0.0;
}

struct Cell {
  long long n = 0;
  double worst = 0.0;
  std::string where;
  void note(double dev, const std::string& w) {
    ++n;
    if (dev > worst) {
      worst = dev;
      where = w;
    }
  }
};

/// The branch of G4HadronNucleonXsc::HadronNucleonXsc a code reaches, for the bucket name - so
/// that a disagreement says WHICH transcription moved.
const char* branch_of(int pdg) {
  const int a = std::abs(pdg);
  if (a == 310 || a == 130) { return "KaonNucleonXscNS (K0S/K0L)"; }
  if (a > 3000) {
    if (a == 3334) { return "HadronNucleonXscPDG (Omega, not dispatched)"; }
    if (a >= 4000) { return "HyperonNucleonXscNS (c/b baryons)"; }
    return (pdg < 0) ? "HyperonNucleonXscNS (anti-hyperons)" : "HyperonNucleonXscNS (hyperons)";
  }
  if (a == 511 || a == 421 || a == 531 || a == 541 || a == 431 || a == 411 || a == 521 ||
      a == 221 || a == 331 || a == 441 || a == 443 || a == 543) {
    return "SCBMesonNucleonXscNS";
  }
  return "HadronNucleonXscPDG (default)";
}

/// A Projectile as the cross sections read one off a G4ParticleDefinition, from the FTF hadron
/// table - which is Geant4's own particle table dumped (data/ftf_hadrons.hh), and is what
/// tests/test_ftf_model.cu builds the hyperon cross-section projectiles from.
bool projectile_of(int pdg, hadronic::xs::Projectile<double>* p) {
  const data::FtfHadron* h = data::ftf_find_hadron(pdg);
  if (h == nullptr) { return false; }
  p->pdg = pdg;
  p->mass = h->mass;
  p->charge = h->charge;
  p->baryon_number = h->baryon;
  p->n_lambdas = 0;
  return true;
}

}  // namespace

int main() {
  const char* env = std::getenv("G4GPU_ORACLE");
  const std::string dir = (env != nullptr) ? env : "ref/oracle";
  namespace bert = physics::hadronic::bert;
  namespace hxs = hadronic::xs;

  // ===========================================================================================
  // 1. The scope, as the running physics list has it.
  // ===========================================================================================
  {
    std::printf("== 1. what QBBC registers on the strange and heavy species ==\n");
    Csv sc;
    if (load(dir + "/strange_scope.csv", sc)) {
      struct Species {
        std::string name;
        int pdg = 0;
        double lifetime = 0;
        int stable = 0, shortlived = 0, n_decay = 0, bert_ok = 0;
        std::vector<std::string> lines;   ///< one per hadronic (process, model) and at-rest row
        bool has_bert = false;
        double bert_lo = 0, bert_hi = 0;
        bool has_ftf = false;
        double ftf_lo = 0, ftf_hi = 0;
        bool has_inelastic = false, has_elastic = false;
        std::string atrest;
      };
      std::vector<Species> list;
      std::map<int, std::size_t> at;
      for (std::size_t r = 0; r < sc.rows.size(); ++r) {
        const int pdg = static_cast<int>(sc.i(r, "pdg"));
        if (at.find(pdg) == at.end()) {
          Species s;
          s.name = sc.s(r, "species");
          s.pdg = pdg;
          s.lifetime = sc.d(r, "lifetime_ns");
          s.stable = static_cast<int>(sc.i(r, "stable"));
          s.shortlived = static_cast<int>(sc.i(r, "shortlived"));
          s.n_decay = static_cast<int>(sc.i(r, "n_decay_channels"));
          s.bert_ok = static_cast<int>(sc.i(r, "bertini_applicable"));
          at[pdg] = list.size();
          list.push_back(s);
        }
        Species& s = list[at[pdg]];
        const std::string& proc = sc.s(r, "process");
        const std::string& model = sc.s(r, "model");
        const double lo = sc.d(r, "emin_MeV"), hi = sc.d(r, "emax_MeV");
        const int sub = static_cast<int>(sc.i(r, "process_subtype"));
        const int atrest = static_cast<int>(sc.i(r, "atrest_index"));
        // G4HadronicProcessType: fHadronElastic = 111, fHadronInelastic = 121.
        if (sub == 121) { s.has_inelastic = true; }
        if (sub == 111) { s.has_elastic = true; }
        if (model == "BertiniCascade" && sub == 121) {
          s.has_bert = true;
          s.bert_lo = lo;
          s.bert_hi = hi;
        }
        if (model == "FTFP" && sub == 121) {
          s.has_ftf = true;
          s.ftf_lo = lo;
          s.ftf_hi = hi;
        }
        if (atrest >= 0 && proc != "Decay") { s.atrest = proc; }
      }

      std::printf("  %-15s %6s  %-22s %-22s %-5s %-5s %-8s %s\n", "species", "pdg", "Bertini (MeV)",
                  "FTFP (MeV)", "inel", "elas", "decay", "at rest");
      int n_bert_checked = 0, n_appl_checked = 0, n_decay_checked = 0, n_decay_refused = 0;
      const char* generic = decay::decay_refusal_reason(987654321);
      for (const Species& s : list) {
        char bw[64] = "-", fw[64] = "-", dw[32];
        if (s.has_bert) { std::snprintf(bw, sizeof bw, "%g - %g", s.bert_lo, s.bert_hi); }
        if (s.has_ftf) { std::snprintf(fw, sizeof fw, "%g - %.3g", s.ftf_lo, s.ftf_hi); }
        std::snprintf(dw, sizeof dw, "%d ch", s.n_decay);
        std::printf("  %-15s %6d  %-22s %-22s %-5s %-5s %-8s %s\n", s.name.c_str(), s.pdg, bw,
                    fw, s.has_inelastic ? "yes" : "-", s.has_elastic ? "yes" : "-",
                    s.stable ? "stable" : dw, s.atrest.empty() ? "-" : s.atrest.c_str());

        // Bertini's window, and the de-excitation the instance that holds it uses. QBBC's third
        // instance (G4HadronicBuilder::BuildFTFP_BERT) never calls usePreCompoundDeexcitation.
        const bert::QbbcBertiniLookup w = bert::qbbc_bertini_range(s.pdg);
        ++n_bert_checked;
        if (w.ok != s.has_bert) {
          fail("%s: QBBC %s a Bertini model on it and the port's qbbc_bertini_range says %s",
               s.name.c_str(), s.has_bert ? "registers" : "does not register",
               w.ok ? "yes" : "no");
        } else if (w.ok) {
          if (w.range.emin_MeV != s.bert_lo || w.range.emax_MeV != s.bert_hi) {
            fail("%s: Bertini window %g-%g MeV in Geant4, %g-%g in the port", s.name.c_str(),
                 s.bert_lo, s.bert_hi, w.range.emin_MeV, w.range.emax_MeV);
          }
          if (w.range.deexcite != bert::DeexciteChoice::kCascade && s.pdg != 2212 &&
              s.pdg != 2112 && std::abs(s.pdg) != 211) {
            fail("%s: the kaon/hyperon instance de-excites through the cascade's own "
                 "evaporators", s.name.c_str());
          }
        }
        // Bertini's IsApplicable, asked of Geant4's own instance (for K0S/K0L it drew a
        // deviate there and the answer did not depend on it).
        ++n_appl_checked;
        if (bert::interface_is_applicable(s.pdg) != (s.bert_ok != 0)) {
          fail("%s: G4CascadeInterface::IsApplicable is %d, the port's is %d", s.name.c_str(),
               s.bert_ok, bert::interface_is_applicable(s.pdg) ? 1 : 0);
        }
        // The decay table: transcribed with the same number of channels, or refused BY NAME.
        // Stable species and species G4Decay declines (negative lifetime) have nothing to say.
        if (!s.stable && s.lifetime >= 0.0 && s.n_decay > 0) {
          const decay::DecayTableRow t = decay::decay_table_for(s.pdg);
          if (t.count > 0) {
            ++n_decay_checked;
            if (t.count != s.n_decay) {
              fail("%s: %d decay channels in Geant4, %d in the port", s.name.c_str(), s.n_decay,
                   t.count);
            }
          } else {
            ++n_decay_refused;
            if (std::strcmp(decay::decay_refusal_reason(s.pdg), generic) == 0) {
              std::printf("  (decay of %s, pdg %d: no table and no NAMED refusal)\n",
                          s.name.c_str(), s.pdg);
            }
          }
        }
      }
      std::printf("  Bertini windows %d, IsApplicable %d, decay tables transcribed %d, decay "
                  "tables not transcribed %d\n", n_bert_checked, n_appl_checked,
                  n_decay_checked, n_decay_refused);
      if (n_bert_checked < 50) { fail("only %d species in strange_scope.csv", n_bert_checked); }
    }
    std::printf("\n");
  }

  // ===========================================================================================
  // 2. The cross sections.
  // ===========================================================================================
  {
    std::printf("== 2a. G4HadronNucleonXsc::HadronNucleonXsc, every branch these species reach ==\n");
    Csv hn;
    if (load(dir + "/strange_hnxsc.csv", hn)) {
      std::map<std::string, Cell> cells;
      long long refused = 0, unknown = 0;
      for (std::size_t r = 0; r < hn.rows.size(); ++r) {
        const int pdg = static_cast<int>(hn.i(r, "pdg"));
        hxs::Projectile<double> p;
        if (!projectile_of(pdg, &p)) {
          if (++unknown <= 3) { fail("no particle row for pdg %d", pdg); }
          continue;
        }
        const bool on_p = hn.s(r, "nucleon") == "proton";
        const hxs::Projectile<double> n = on_p ? hxs::proton<double>() : hxs::neutron<double>();
        const double e = hn.d(r, "energy_MeV");
        const hxs::HadXs<double> got = hxs::hadron_nucleon_xsc<double>(p, n, e);
        if (!got.ok()) {
          if (++refused <= 3) {
            fail("%s on %s at %g MeV refused: %s", hn.s(r, "particle").c_str(),
                 hn.s(r, "nucleon").c_str(), e, hxs::xs_refusal_name(got.refused));
          }
          continue;
        }
        char w[160];
        std::snprintf(w, sizeof w, "%s on %s at %.6g MeV", hn.s(r, "particle").c_str(),
                      hn.s(r, "nucleon").c_str(), e);
        Cell& c = cells[branch_of(pdg)];
        c.note(deviation(got.total, hn.d(r, "total_mm2")), std::string(w) + " total");
        c.note(deviation(got.elastic, hn.d(r, "elastic_mm2")), std::string(w) + " elastic");
        c.note(deviation(got.inelastic, hn.d(r, "inelastic_mm2")), std::string(w) + " inelastic");
      }
      for (const auto& kv : cells) {
        std::printf("  %-44s %8lld values  worst %.3e  %s\n", kv.first.c_str(), kv.second.n,
                    kv.second.worst, kv.second.where.c_str());
        if (kv.second.worst > kTol) { fail("%s worst %.3e", kv.first.c_str(), kv.second.worst); }
      }
      std::printf("  refused %lld, species with no particle row %lld\n", refused, unknown);
      if (cells.find("SCBMesonNucleonXscNS") == cells.end()) {
        fail("no SCBMesonNucleonXscNS row was compared");
      }
    }
    std::printf("\n");

    std::printf("== 2b. G4ComponentGGHadronNucleusXsc, all five outputs ==\n");
    Csv gg;
    if (load(dir + "/strange_ggxs.csv", gg)) {
      const char* cols[5] = {"total_mm2", "inelastic_mm2", "elastic_mm2", "production_mm2",
                             "diffraction_mm2"};
      std::map<std::string, Cell> cells;
      long long refused = 0;
      std::set<int> species;
      for (std::size_t r = 0; r < gg.rows.size(); ++r) {
        const int pdg = static_cast<int>(gg.i(r, "pdg"));
        hxs::Projectile<double> p;
        if (!projectile_of(pdg, &p)) {
          fail("no particle row for pdg %d", pdg);
          continue;
        }
        species.insert(pdg);
        const int z = static_cast<int>(gg.i(r, "Z"));
        const int a = static_cast<int>(gg.i(r, "A"));
        const double e = gg.d(r, "energy_MeV");
        const hxs::HadXs<double> got = hxs::ggh_compute_cross_sections<double>(p, e, z, a);
        if (!got.ok()) {
          if (++refused <= 3) {
            fail("%s Z=%d at %g MeV refused: %s", gg.s(r, "particle").c_str(), z, e,
                 hxs::xs_refusal_name(got.refused));
          }
          continue;
        }
        const double ours[5] = {got.total, got.inelastic, got.elastic, got.production,
                                got.diffraction};
        char w[160];
        std::snprintf(w, sizeof w, "%s Z=%d A=%d at %.6g MeV", gg.s(r, "particle").c_str(), z,
                      a, e);
        Cell& c = cells[branch_of(pdg)];
        for (int k = 0; k < 5; ++k) {
          c.note(deviation(ours[k], gg.d(r, cols[k])), std::string(w) + " " + cols[k]);
        }
      }
      for (const auto& kv : cells) {
        std::printf("  %-44s %8lld values  worst %.3e  %s\n", kv.first.c_str(), kv.second.n,
                    kv.second.worst, kv.second.where.c_str());
        if (kv.second.worst > kTol) { fail("%s worst %.3e", kv.first.c_str(), kv.second.worst); }
      }
      std::printf("  %zu species, refused %lld\n", species.size(), refused);
      if (species.size() < 40) { fail("only %zu species in strange_ggxs.csv", species.size()); }
    }
    std::printf("\n");

    std::printf("== 2c. the REGISTERED processes against the component, and the port against "
                "both ==\n");
    Csv px;
    if (load(dir + "/strange_scope_xs.csv", px)) {
      Cell g4_in, g4_el, port_in, port_el;
      long long zero_process = 0;
      for (std::size_t r = 0; r < px.rows.size(); ++r) {
        const int pdg = static_cast<int>(px.i(r, "pdg"));
        const int z = static_cast<int>(px.i(r, "Z"));
        const double e = px.d(r, "energy_MeV");
        char w[160];
        std::snprintf(w, sizeof w, "%s Z=%d at %.6g MeV", px.s(r, "species").c_str(), z, e);
        const double pin = px.d(r, "process_inelastic_mm2");
        const double pel = px.d(r, "process_elastic_mm2");
        if (pin == 0.0 && px.d(r, "gg_inelastic_mm2") > 0.0) { ++zero_process; }
        // Geant4 against itself: the process QBBC built returns the component's number.
        g4_in.note(deviation(pin, px.d(r, "gg_inelastic_mm2")), w);
        g4_el.note(deviation(pel, px.d(r, "gg_elastic_mm2")), w);
        // The port: the G4double-A overloads, with the port's own NIST mean mass, rounded by
        // G4lrint inside - the whole path G4CrossSectionInelastic/Elastic takes.
        hxs::Projectile<double> p;
        if (!projectile_of(pdg, &p)) {
          fail("no particle row for pdg %d", pdg);
          continue;
        }
        const double a = data::atomic_mass<double>(z);
        const hxs::XsValue<double> in = hxs::ggh_inelastic_element<double>(p, e, z, a);
        const hxs::XsValue<double> el = hxs::ggh_elastic_element<double>(p, e, z, a);
        if (!in.ok() || !el.ok()) {
          fail("%s refused", w);
          continue;
        }
        port_in.note(deviation(in.value, pin), w);
        port_el.note(deviation(el.value, pel), w);
      }
      std::printf("  Geant4 process vs component, inelastic %6lld values  worst %.3e  %s\n",
                  g4_in.n, g4_in.worst, g4_in.where.c_str());
      std::printf("  Geant4 process vs component, elastic   %6lld values  worst %.3e  %s\n",
                  g4_el.n, g4_el.worst, g4_el.where.c_str());
      std::printf("  port vs process, inelastic             %6lld values  worst %.3e  %s\n",
                  port_in.n, port_in.worst, port_in.where.c_str());
      std::printf("  port vs process, elastic               %6lld values  worst %.3e  %s\n",
                  port_el.n, port_el.worst, port_el.where.c_str());
      // The first two are statements about QBBC, not about the port: every strange and heavy
      // inelastic and elastic process carries the bare Glauber-Gribov component - no second
      // data set, no scale factor. Equal to the last bit, because it is the same call.
      if (g4_in.worst > 0.0 || g4_el.worst > 0.0) {
        fail("a registered process does not return the bare component - the scope table's "
             "cross-section column is wrong");
      }
      if (zero_process > 0) { fail("%lld points where a process returned zero", zero_process); }
      if (port_in.worst > kTol || port_el.worst > kTol) {
        fail("port against the registered process: inelastic %.3e, elastic %.3e", port_in.worst,
             port_el.worst);
      }
    }
    std::printf("\n");
  }

  // ===========================================================================================
  // 3. Heavy flavour out of QBBC's own FTFP.
  // ===========================================================================================
  {
    std::printf("== 3. b/c hadrons out of QBBC's FTFP ==\n");
    Csv lu, sp;
    if (load(dir + "/strange_lund.csv", lu)) {
      const double cc = lu.d(0, "prob_ccbar");
      const double bb = lu.d(0, "prob_bbbar");
      std::printf("  EnableBCParticles %lld, ProbCCbar %.6g, ProbBBbar %.6g "
                  "(G4LundStringFragmentation.cc:69-71)\n",
                  lu.i(0, "enable_bc_particles"), cc, bb);
      if (lu.i(0, "enable_bc_particles") != 1 || !(cc > 0.0) || !(bb > 0.0)) {
        fail("QBBC's string fragmentation does not create c or b quarks - the b/c rows of the "
             "scope table would be unreachable from inside the model");
      }
    }
    // What the oracle's own FTFP events made, summed over its statistical cases: the rate is a
    // measurement of how often a QBBC transport meets a b/c hadron it did not shoot.
    if (load(dir + "/ftf_modelstat_species.csv", sp)) {
      long long bc = 0, events = 0;
      std::map<std::string, long long> seen;
      std::map<int, long long> by_pdg;
      for (std::size_t r = 0; r < sp.rows.size(); ++r) {
        const std::string& cs = sp.s(r, "case");
        if (seen.find(cs) == seen.end()) {
          seen[cs] = sp.i(r, "n_events");
          events += seen[cs];
        }
        const int a = std::abs(static_cast<int>(sp.i(r, "pdg")));
        const bool heavy = (a % 10000) / 100 == 4 || (a % 10000) / 100 == 5 ||
                           (a % 10000) / 1000 == 4 || (a % 10000) / 1000 == 5;
        if (heavy && a < 10000) {
          bc += sp.i(r, "count");
          by_pdg[static_cast<int>(sp.i(r, "pdg"))] += sp.i(r, "count");
        }
      }
      std::printf("  %lld charm/bottom hadrons in %lld FTFP events of the oracle (%.2e per "
                  "event):", bc, events, events > 0 ? double(bc) / double(events) : 0.0);
      for (const auto& kv : by_pdg) { std::printf(" %d:%lld", kv.first, kv.second); }
      std::printf("\n");
    }
    std::printf("\n");
  }

  std::printf("%s\n", g_fails == 0 ? "PASS test_strange" : "FAIL test_strange");
  return g_fails == 0 ? 0 : 1;
}
