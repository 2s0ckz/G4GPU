// G4FTFParameters and the Lund string-decay tables against ref/oracle/ftf_*.csv.
//
// Every comparison in this file is EXACT - not statistically exact, arithmetically exact -
// because nothing it checks samples anything. `G4FTFParameters::InitForInteraction` draws no
// random number at all: it is 700 lines of dispatch on (projectile PDG code, target Z and A,
// lab momentum) over a Glauber-Gribov hadron-nucleon cross section and about thirty tuned
// constants. `SetMinMasses` draws none either. So the oracle has no phase column and the
// tolerance is 1e-13 relative, which is where the G4Exp/G4Log-against-std difference would
// show if it mattered.
//
//   ftf_lund_params.csv   the ~30 scalars of G4VLongitudinalStringDecay and
//                         G4LundStringFragmentation, plus the three G4HadronicParameters
//                         flags this package dispatches on and all ten G4FTFTunings
//                         applicability states.
//   ftf_lund_tables.csv   minMassQQbarStr[5][5], minMassQDiQStr[5][5][5], Meson[5][5][7] and
//                         MesonWeight, Baryon[5][5][5][4] and BaryonWeight, Qcharge[5],
//                         Prob_QQbar[5] - 1,510 numbers, all computed by SetMinMasses rather
//                         than tabulated, so a wrong PDG mass or a wrong index shows here.
//   ftf_hadrons.csv       the particle table itself, in BOTH directions: every row of the CSV
//                         must be in data/ftf_hadrons.hh with the same mass, and every row of
//                         the header must be in the CSV. A one-directional check would pass a
//                         header that had invented a particle, which is a mass the port made
//                         up (the same argument docs/PORTED.md 2.1.1 makes for AME2012).
//   ftf_params.csv        1,716 rows x 66 columns: 22 projectiles x 6 targets x 13 momenta.
//   ftf_procprob.csv      GetProcProb(ProcN, y) on 21,780 (row, process, rapidity) points,
//                         including both sides of every Ymin in the table and the -100 and
//                         1000 sentinels the A > 10 branches write.
//   ftf_geom.csv          GammaElastic, GetInelasticProbability and
//                         GetProbabilityOfInteraction over impact parameter squared.
//   ftf_minmass.csv       SetMinimalStringMass over every legal parton pair at five string
//                         masses - the tables read through the function that reads them,
//                         including the DiQuark-AntiDiQuark re-arrangement arm, which no
//                         single table entry shows.
//
// WHAT IS COUNTED AS A REFUSAL RATHER THAN A PASS. A hyperon projectile (Lambda, Sigma+-,
// Xi-) needs G4HadronNucleonXsc::HyperonNucleonXscNS, which docs/PORTED.md 2.1.1 records as
// `P` - refused by name in xs/hadron_nucleon_xsc.cuh. So `ftf_init_for_interaction` refuses
// those rows and they are tallied separately, exactly as tests/test_particlexs.cu tallies the
// 1,128 photonuclear-gap points. An ANTI-hyperon is NOT refused, and the reason is worth
// stating: Geant4 computes the hadron-nucleon average for it and then overwrites both cross
// sections in the Arkhipov anti-baryon block, so the refused branch never reaches the answer.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "physics/hadronic/ftf/ftf_parameters.cuh"

using namespace g4gpu;
using namespace g4gpu::hadronic;

/// The deliverables are device-callable, and a `__host__ __device__` template that is only
/// ever instantiated from the host is never compiled for the device at all. This kernel is
/// never launched - the package is validated on the host and the machine's GPU belongs to the
/// integration builds - but instantiating it is what proves the module is device code, and
/// `-Xptxas -v` on it is what reports the register and stack cost the plan asks for. The
/// tables and the parameters are passed BY POINTER, never by value and never as locals: the
/// LundTables struct alone is about 12 kB.
__global__ void ftf_params_device_probe(ftf::LundTables<double>* lund,
                                        ftf::FtfParameters<double>* par,
                                        const xs::Projectile<double>* proj, int a, int z,
                                        double plab, double* out) {
  ftf::lund_init(lund, true);
  ftf::ftf_parameters_construct(par, false);
  ftf::ftf_init_for_interaction(par, proj[0], a, z, plab, lund);
  const ftf::MinimalStringMass<double> mm =
      ftf::ftf_minimal_string_mass(lund, 1, -1, 5000.0);
  out[0] = par->x_total + ftf::ftf_get_proc_prob(par, 0, 1.5) +
           ftf::ftf_gamma_elastic(par, 0.5) + ftf::ftf_get_inelastic_probability(par, 0.5) +
           ftf::ftf_get_probability_of_interaction(par, 0.5) + mm.mass +
           ftf::ftf_get_min_mass(lund, 2212);
}

namespace {

int fails = 0;

// ---------------------------------------------------------------------------------------------
// Comparison bookkeeping, as tests/test_precompound.cu defines it
// ---------------------------------------------------------------------------------------------

struct Bucket {
  const char* name;
  long long n = 0;
  double worst = 0.0;
  std::string where;
  double tol = 1e-13;
};

std::vector<Bucket> buckets;
long long n_refused = 0;

int new_bucket(const char* name, double tol) {
  Bucket b;
  b.name = name;
  b.tol = tol;
  buckets.push_back(b);
  return static_cast<int>(buckets.size()) - 1;
}

void cmp(int bi, double got, double want, const std::string& where) {
  Bucket& b = buckets[bi];
  ++b.n;
  const double scale = (std::fabs(want) > 0.0) ? std::fabs(want) : 1.0;
  const double rel = std::fabs(got - want) / scale;
  if (rel > b.worst) {
    b.worst = rel;
    b.where = where;
  }
}

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

// ---------------------------------------------------------------------------------------------
// CSV reading
// ---------------------------------------------------------------------------------------------

std::string oracle_dir() {
  const char* e = std::getenv("G4GPU_ORACLE");
  return (e != nullptr && e[0] != '\0') ? std::string(e) : std::string("ref/oracle");
}

std::vector<std::string> read_lines(const std::string& path) {
  std::vector<std::string> out;
  FILE* f = std::fopen(path.c_str(), "r");
  if (f == nullptr) { return out; }
  // 66 columns of %.17g plus a name: 2 kB is not enough, and a silently truncated line reads
  // as a row with fewer fields, which `dv` then answers 0.0 for. 16 kB, and the field count
  // is asserted per row below.
  static char line[65536];
  while (std::fgets(line, sizeof line, f) != nullptr) { out.push_back(line); }
  std::fclose(f);
  return out;
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

double dv(const std::vector<std::string>& f, std::size_t i) {
  return (i < f.size()) ? std::atof(f[i].c_str()) : 0.0;
}
int iv(const std::vector<std::string>& f, std::size_t i) {
  return (i < f.size()) ? std::atoi(f[i].c_str()) : 0;
}

/// Column index by header name, so a reordered dump does not silently compare the wrong
/// column against the right one.
struct Header {
  std::map<std::string, std::size_t> ix;
  std::size_t n = 0;
  void parse(const std::string& line) {
    const std::vector<std::string> f = split(line);
    n = f.size();
    for (std::size_t i = 0; i < f.size(); ++i) { ix[f[i]] = i; }
  }
  std::size_t at(const char* name) const {
    const auto it = ix.find(name);
    if (it == ix.end()) {
      std::printf("FAIL: oracle column '%s' is missing\n", name);
      ++fails;
      return 0;
    }
    return it->second;
  }
};

// ---------------------------------------------------------------------------------------------
// The projectiles the oracle grid uses, by PDG code
// ---------------------------------------------------------------------------------------------

/// An `xs::Projectile` for every PDG code in ftf_params.csv. The ion codes come back with the
/// baryon number and charge decoded out of 10LZZZAAAI, which is all InitForInteraction reads
/// of a nucleus: it replaces the mass with the proton's before it is used for anything.
bool projectile_for(int pdg, double mass, xs::Projectile<double>* out) {
  if (pdg > 1000000000 || pdg < -1000000000) {
    const int a = (std::abs(pdg) / 10) % 1000;
    const int z = (std::abs(pdg) / 10000) % 1000;
    out->pdg = pdg;
    out->mass = mass;
    out->charge = (pdg > 0) ? double(z) : -double(z);
    out->baryon_number = (pdg > 0) ? a : -a;
    out->n_lambdas = 0;
    return true;
  }
  const data::FtfHadron* h = data::ftf_find_hadron(pdg);
  if (h == nullptr) { return false; }
  out->pdg = pdg;
  out->mass = h->mass;
  out->charge = h->charge;
  out->baryon_number = h->baryon;
  out->n_lambdas = 0;
  return true;
}

}  // namespace

int main() {
  const std::string dir = oracle_dir();

  // A single set of tables and one parameter object, both heap-allocated: the plan's rule that
  // the workspace is a struct passed by pointer applies on the host too, and a 12 kB local
  // would be a 12 kB stack frame in the device probe above.
  ftf::LundTables<double>* lund = new ftf::LundTables<double>();
  ftf::lund_init(lund, true);  // EnableBCParticles is 1 in 11.1.1 - asserted below

  // -------------------------------------------------------------------------------------------
  // 1. ftf_lund_params.csv
  // -------------------------------------------------------------------------------------------
  const int b_lundpar = new_bucket("LundScalarParameters", 1e-14);
  {
    const std::vector<std::string> lines = read_lines(dir + "/ftf_lund_params.csv");
    if (lines.size() < 2) {
      std::printf("FAIL: cannot read %s/ftf_lund_params.csv\n", dir.c_str());
      ++fails;
    }
    std::map<std::string, double> v;
    for (std::size_t i = 1; i < lines.size(); ++i) {
      const std::vector<std::string> f = split(lines[i]);
      if (f.size() < 2) { continue; }
      v[f[0]] = std::atof(f[1].c_str());
    }
    auto chk = [&](const char* name, double got) {
      const auto it = v.find(name);
      if (it == v.end()) {
        std::printf("FAIL: %s is not in ftf_lund_params.csv\n", name);
        ++fails;
        return;
      }
      cmp(b_lundpar, got, it->second, name);
    };
    chk("MassCut", lund->mass_cut);
    chk("SigmaQT", lund->sigma_qt);
    chk("StrangeSuppress", lund->strange_suppress);
    chk("DiquarkSuppress", lund->diquark_suppress);
    chk("DiquarkBreakProb", lund->diquark_break_prob);
    chk("StringLoopInterrupt", lund->string_loop_interrupt);
    chk("ClusterLoopInterrupt", lund->cluster_loop_interrupt);
    chk("pspin_barion", lund->pspin_barion);
    for (int i = 0; i < 3; ++i) {
      chk(("pspin_meson_" + std::to_string(i)).c_str(), lund->pspin_meson[i]);
    }
    for (int i = 0; i < 6; ++i) {
      chk(("scalarMesonMix_" + std::to_string(i)).c_str(), lund->scalar_meson_mix[i]);
      chk(("vectorMesonMix_" + std::to_string(i)).c_str(), lund->vector_meson_mix[i]);
    }
    chk("ProbCCbar", lund->prob_ccbar);
    chk("ProbBBbar", lund->prob_bbbar);
    chk("ProbCB", lund->prob_cb);
    chk("ProbEta_c", lund->prob_eta_c);
    chk("ProbEta_b", lund->prob_eta_b);
    // Kappa is 1e30 MeV^2/mm^2, not 1e15 MeV/mm: SetStringTensionParameter multiplies by
    // GeV/fermi and G4LundStringFragmentation's constructor hands it an argument that is
    // already GeV/fermi. See docs/RISK.md V87. Nothing in QBBC reads it - its only consumer
    // is CalculateHadronTimePosition, which only G4QGSMFragmentation calls - and that is why
    // the value can be this wrong without a symptom.
    chk("Kappa", lund->kappa);
    chk("MaxMass", lund->max_mass);
    chk("Mass_of_light_quark", lund->mass_of_light_quark);
    chk("Mass_of_s_quark", lund->mass_of_s_quark);
    chk("Mass_of_c_quark", lund->mass_of_c_quark);
    chk("Mass_of_b_quark", lund->mass_of_b_quark);
    chk("Mass_of_string_junction", lund->mass_of_string_junction);

    // The flags this package's dispatch reads. Asserted, not assumed: EnableBCParticles is
    // what makes ProbCCbar 2e-4 rather than 0, i.e. what makes G4HadronBuilder's whole
    // charm/bottom substitution list reachable from an ordinary proton beam.
    if (v.count("EnableBCParticles") == 0 || v["EnableBCParticles"] != 1.0) {
      std::printf("FAIL: EnableBCParticles is %g, the port is built for 1\n",
                  v.count("EnableBCParticles") ? v["EnableBCParticles"] : -1.0);
      ++fails;
    }
    if (v.count("EnableDiffDissociationForBGreater10") == 0 ||
        v["EnableDiffDissociationForBGreater10"] != 0.0) {
      std::printf("FAIL: EnableDiffDissociationForBGreater10 is not 0\n");
      ++fails;
    }
    if (v.count("EnableCRCoalescence") == 0 || v["EnableCRCoalescence"] != 0.0) {
      std::printf("FAIL: EnableCRCoalescence is not 0 - G4CRCoalescence is refused by name\n");
      ++fails;
    }
    // All ten tune states, so that a run with a tune switched on fails here rather than
    // comparing two different parameter sets column by column.
    if (v.count("FTF_numberOfTunes") == 0 || v["FTF_numberOfTunes"] != 10.0) {
      std::printf("FAIL: G4FTFTunings::sNumberOfTunes is not 10\n");
      ++fails;
    }
    for (int i = 0; i < 10; ++i) {
      const std::string k = "FTF_tuneState_" + std::to_string(i);
      const double want = (i == 0) ? 1.0 : 0.0;
      if (v.count(k) == 0 || v[k] != want) {
        std::printf("FAIL: %s is %g, expected %g - FtfRefusal::kFtfTuneNonDefault\n", k.c_str(),
                    v.count(k) ? v[k] : -1.0, want);
        ++fails;
      }
    }
  }

  // -------------------------------------------------------------------------------------------
  // 2. ftf_lund_tables.csv
  // -------------------------------------------------------------------------------------------
  const int b_minq = new_bucket("MinMassQQbarStr", 1e-14);
  const int b_mindq = new_bucket("MinMassQDiQStr", 1e-14);
  const int b_meson = new_bucket("MesonTable", 0.0);
  const int b_mesonw = new_bucket("MesonWeight", 1e-14);
  const int b_baryon = new_bucket("BaryonTable", 0.0);
  const int b_baryonw = new_bucket("BaryonWeight", 1e-14);
  const int b_misc = new_bucket("QchargeAndProbQQbar", 0.0);
  {
    const std::vector<std::string> lines = read_lines(dir + "/ftf_lund_tables.csv");
    if (lines.size() < 2) {
      std::printf("FAIL: cannot read %s/ftf_lund_tables.csv\n", dir.c_str());
      ++fails;
    }
    long long seen = 0;
    for (std::size_t li = 1; li < lines.size(); ++li) {
      const std::vector<std::string> f = split(lines[li]);
      if (f.size() < 6) { continue; }
      const std::string& t = f[0];
      const int i = std::atoi(f[1].c_str());
      const int j = std::atoi(f[2].c_str());
      const int k = std::atoi(f[3].c_str());
      const int l = std::atoi(f[4].c_str());
      const double want = std::atof(f[5].c_str());
      char w[96];
      std::snprintf(w, sizeof w, "%s[%d][%d][%d][%d]", t.c_str(), i, j, k, l);
      ++seen;
      if (t == "minMassQQbarStr") {
        cmp(b_minq, lund->min_mass_qqbar[i][j], want, w);
      } else if (t == "minMassQDiQStr") {
        cmp(b_mindq, lund->min_mass_qdiq[i][j][k], want, w);
      } else if (t == "Meson") {
        cmp_int(b_meson, lund->meson[i][j][k], (long long)want, w);
      } else if (t == "MesonWeight") {
        cmp(b_mesonw, lund->meson_weight[i][j][k], want, w);
      } else if (t == "Baryon") {
        cmp_int(b_baryon, lund->baryon[i][j][k][l], (long long)want, w);
      } else if (t == "BaryonWeight") {
        cmp(b_baryonw, lund->baryon_weight[i][j][k][l], want, w);
      } else if (t == "Qcharge") {
        cmp_int(b_misc, lund->qcharge[i], (long long)want, w);
      } else if (t == "Prob_QQbar") {
        cmp(b_misc, lund->prob_qqbar[i], want, w);
      } else {
        std::printf("FAIL: unknown table '%s' in ftf_lund_tables.csv\n", t.c_str());
        ++fails;
      }
    }
    // 25 + 125 + 2*175 + 2*500 + 2*5 = 1510. Asserted so that a dump that stopped early -
    // which is exactly what a crash inside the dump looks like - fails here.
    if (seen != 1510) {
      std::printf("FAIL: ftf_lund_tables.csv has %lld rows, expected 1510\n", seen);
      ++fails;
    }
  }

  // The one derived assertion the tables alone do not make: SetMinMasses' d-dbar meson row
  // does not sum to one, and the u-ubar row does. docs/RISK.md V85. Stated as a test rather
  // than as a comment because a future "fix" to the index mismatch would silently change
  // every small-string two-body decay, and this line is what would object.
  {
    double sum_dd = 0.0, sum_uu = 0.0;
    for (int k = 0; k < 7; ++k) {
      sum_dd += lund->meson_weight[0][0][k];
      sum_uu += lund->meson_weight[1][1][k];
    }
    if (std::fabs(sum_uu - 1.0) > 1e-14) {
      std::printf("FAIL: MesonWeight[1][1][*] sums to %.17g, expected 1\n", sum_uu);
      ++fails;
    }
    if (std::fabs(sum_dd - 0.875) > 1e-14) {
      std::printf("FAIL: MesonWeight[0][0][*] sums to %.17g, expected 0.875 (V85)\n", sum_dd);
      ++fails;
    }
    if (lund->meson_weight[0][0][2] != 0.0) {
      std::printf("FAIL: MesonWeight[0][0][2] (the eta from d-dbar) is %.17g, expected 0\n",
                  lund->meson_weight[0][0][2]);
      ++fails;
    }
  }

  // -------------------------------------------------------------------------------------------
  // 3. ftf_hadrons.csv - the generated table, both directions
  // -------------------------------------------------------------------------------------------
  const int b_hadmass = new_bucket("HadronMasses", 0.0);
  {
    const std::vector<std::string> lines = read_lines(dir + "/ftf_hadrons.csv");
    if (lines.size() < 2) {
      std::printf("FAIL: cannot read %s/ftf_hadrons.csv\n", dir.c_str());
      ++fails;
    }
    Header h;
    h.parse(lines[0]);
    std::set<int> in_csv;
    long long dropped = 0;
    for (std::size_t li = 1; li < lines.size(); ++li) {
      const std::vector<std::string> f = split(lines[li]);
      if (f.size() < h.n) { continue; }
      const int pdg = iv(f, h.at("pdg"));
      if (pdg == 0 || pdg >= 1000000000 || pdg <= -1000000000) { ++dropped; continue; }
      in_csv.insert(pdg);
      const data::FtfHadron* hh = data::ftf_find_hadron(pdg);
      if (hh == nullptr) {
        std::printf("FAIL: PDG %d is in the oracle and not in data/ftf_hadrons.hh\n", pdg);
        ++fails;
        continue;
      }
      const std::string w = "pdg=" + std::to_string(pdg);
      cmp(b_hadmass, hh->mass, dv(f, h.at("mass")), w + " mass");
      cmp(b_hadmass, hh->width, dv(f, h.at("width")), w + " width");
      cmp(b_hadmass, hh->charge, dv(f, h.at("charge")), w + " charge");
      cmp_int(b_hadmass, hh->baryon, iv(f, h.at("baryon")), w + " baryon");
      cmp_int(b_hadmass, hh->shortlived ? 1 : 0, iv(f, h.at("shortlived")), w + " shortlived");
      cmp(b_hadmass, hh->minmass, dv(f, h.at("minmass")), w + " minmass");
      const std::string st = f[h.at("subtype")];
      const data::FtfSubType want = (st == "quark") ? data::FtfSubType::kQuark
                                    : (st == "di_quark") ? data::FtfSubType::kDiQuark
                                                         : data::FtfSubType::kOther;
      cmp_int(b_hadmass, (int)hh->subtype, (int)want, w + " subtype");
    }
    // The other direction: a header row with no oracle row is a particle the port invented.
    for (int i = 0; i < data::kFtfHadronCount; ++i) {
      const int pdg = data::ftf_hadrons()[i].pdg;
      if (in_csv.count(pdg) == 0) {
        std::printf("FAIL: PDG %d is in data/ftf_hadrons.hh and not in the oracle\n", pdg);
        ++fails;
      }
    }
    if ((int)in_csv.size() != data::kFtfHadronCount) {
      std::printf("FAIL: oracle has %d non-nucleus particles, the header has %d\n",
                  (int)in_csv.size(), data::kFtfHadronCount);
      ++fails;
    }
    // How many NUCLEUS rows the oracle carries is not asserted, and the measurement that
    // settled it is in tools/ftf_hadrons.pl: the same dump program wrote 31 on one run and
    // 1,783 on another, because G4ParticleTable holds whatever G4IonTable has been asked for
    // so far in that process and the ftf dump runs after three dumps that create ions. The 485
    // hadrons were identical to the last bit in both. The two directions above are the
    // assertion that matters; this line only reports what was skipped.
    std::printf("  (ftf_hadrons.csv: %lld nucleus/geantino rows skipped, %d hadrons compared)\n",
                dropped, (int)in_csv.size());
    // The table is binary-searched, so it must be strictly ascending.
    for (int i = 1; i < data::kFtfHadronCount; ++i) {
      if (data::ftf_hadrons()[i].pdg <= data::ftf_hadrons()[i - 1].pdg) {
        std::printf("FAIL: data/ftf_hadrons.hh is not sorted at index %d\n", i);
        ++fails;
        break;
      }
    }
  }

  // -------------------------------------------------------------------------------------------
  // 4. ftf_params.csv
  // -------------------------------------------------------------------------------------------
  const int b_xs = new_bucket("FtfCrossSections", 1e-13);
  const int b_geo = new_bucket("FtfGeometry", 1e-13);
  const int b_exc = new_bucket("FtfExcitationParams", 1e-13);
  const int b_pp = new_bucket("FtfProcParams", 1e-13);
  const int b_nd = new_bucket("FtfNuclearDestruction", 1e-13);
  const int b_tune = new_bucket("FtfIndexTune", 0.0);
  ftf::FtfParameters<double>* par = new ftf::FtfParameters<double>();
  ftf::ftf_parameters_construct(par, false);  // EnableDiffDissociationForBGreater10 = 0
  {
    const std::vector<std::string> lines = read_lines(dir + "/ftf_params.csv");
    if (lines.size() < 2) {
      std::printf("FAIL: cannot read %s/ftf_params.csv\n", dir.c_str());
      ++fails;
    }
    Header h;
    h.parse(lines[0]);
    long long rows = 0;
    for (std::size_t li = 1; li < lines.size(); ++li) {
      const std::vector<std::string> f = split(lines[li]);
      if (f.size() != h.n) {
        if (f.size() > 1) {
          std::printf("FAIL: ftf_params.csv row %d has %d fields, header has %d\n", (int)li,
                      (int)f.size(), (int)h.n);
          ++fails;
        }
        continue;
      }
      ++rows;
      const int pdg = iv(f, h.at("proj_pdg"));
      const int z = iv(f, h.at("tgt_z"));
      const int a = iv(f, h.at("tgt_a"));
      const double plab = dv(f, h.at("plab_mev"));
      char wbuf[160];
      std::snprintf(wbuf, sizeof wbuf, "%s(%d) on Z=%d A=%d at plab=%g",
                    f[h.at("proj_name")].c_str(), pdg, z, a, plab);
      const std::string w = wbuf;

      xs::Projectile<double> proj;
      if (!projectile_for(pdg, 0.0, &proj)) {
        std::printf("FAIL: no projectile for PDG %d (%s)\n", pdg, w.c_str());
        ++fails;
        continue;
      }
      ftf::ftf_init_for_interaction(par, proj, a, z, plab, lund);
      if (par->refused != ftf::FtfRefusal::kNone) {
        ++n_refused;
        continue;
      }
      cmp(b_xs, par->x_total, dv(f, h.at("xtotal")), w + " xtotal");
      cmp(b_xs, par->x_elastic, dv(f, h.at("xelastic")), w + " xelastic");
      cmp(b_xs, par->x_inelastic, dv(f, h.at("xinelastic")), w + " xinelastic");
      cmp(b_xs, par->prob_of_elastic_scatt, dv(f, h.at("prob_elastic")), w + " probel");
      cmp(b_xs, par->prob_of_annihilation, dv(f, h.at("prob_annih")), w + " probann");

      cmp(b_geo, par->radius_of_hn_interactions2, dv(f, h.at("radius2")), w + " radius2");
      cmp(b_geo, par->slope, dv(f, h.at("slope")), w + " slope");
      cmp(b_geo, par->gamma0, dv(f, h.at("gamma0")), w + " gamma0");
      cmp(b_geo, par->avarage_pt2_of_elastic_scattering, dv(f, h.at("avpt2_elastic")),
          w + " avpt2el");

      cmp(b_exc, par->delta_prob_at_quark_exchange, dv(f, h.at("delta_prob_qexchg")),
          w + " dpqe");
      cmp(b_exc, par->prob_of_same_quark_exchange, dv(f, h.at("prob_same_qexchg")),
          w + " psqe");
      cmp(b_exc, par->proj_min_diff_mass, dv(f, h.at("proj_min_diff_m")), w + " pmdm");
      cmp(b_exc, par->proj_min_non_diff_mass, dv(f, h.at("proj_min_nondiff_m")), w + " pmndm");
      cmp(b_exc, par->tar_min_diff_mass, dv(f, h.at("tar_min_diff_m")), w + " tmdm");
      cmp(b_exc, par->tar_min_non_diff_mass, dv(f, h.at("tar_min_nondiff_m")), w + " tmndm");
      cmp(b_exc, par->average_pt2, dv(f, h.at("average_pt2")), w + " avpt2");
      cmp(b_exc, par->prob_log_distr_prd, dv(f, h.at("prob_log_distr_prd")), w + " pldprd");
      cmp(b_exc, par->prob_log_distr, dv(f, h.at("prob_log_distr")), w + " pld");
      cmp(b_exc, par->pt2_kink, dv(f, h.at("pt2_kink")), w + " pt2kink");
      cmp(b_exc, par->quark_probabilities_at_gluon_split_up[0], dv(f, h.at("qprob0")),
          w + " qprob0");
      cmp(b_exc, par->quark_probabilities_at_gluon_split_up[1], dv(f, h.at("qprob1")),
          w + " qprob1");
      cmp(b_exc, par->quark_probabilities_at_gluon_split_up[2], dv(f, h.at("qprob2")),
          w + " qprob2");

      cmp(b_nd, par->max_number_of_collisions, dv(f, h.at("max_n_collisions")), w + " maxncol");
      cmp(b_nd, par->prob_of_inel_interaction, dv(f, h.at("prob_of_interaction")), w + " probint");
      cmp(b_nd, par->cof_nuclear_destruction_pr, dv(f, h.at("cof_nd_proj")), w + " cndpr");
      cmp(b_nd, par->cof_nuclear_destruction, dv(f, h.at("cof_nd_tgt")), w + " cnd");
      cmp(b_nd, par->r2_of_nuclear_destruction, dv(f, h.at("r2_nd")), w + " r2nd");
      cmp(b_nd, par->excitation_energy_per_wounded_nucleon,
          dv(f, h.at("exci_e_per_wounded")), w + " exci");
      cmp(b_nd, par->dof_nuclear_destruction, dv(f, h.at("d_nd")), w + " dnd");
      cmp(b_nd, par->pt2_of_nuclear_destruction, dv(f, h.at("pt2_nd")), w + " pt2nd");
      cmp(b_nd, par->max_pt2_of_nuclear_destruction, dv(f, h.at("max_pt2_nd")), w + " maxpt2nd");

      cmp_int(b_tune, par->index_tune, iv(f, h.at("index_tune")), w + " index_tune");

      for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 7; ++j) {
          char cn[16];
          std::snprintf(cn, sizeof cn, "pp_%d_%d", i, j);
          cmp(b_pp, par->proc_params[i][j], dv(f, h.at(cn)), w + " " + cn);
        }
      }
    }
    if (rows != 1716) {
      std::printf("FAIL: ftf_params.csv has %lld rows, expected 1716\n", rows);
      ++fails;
    }
  }

  // -------------------------------------------------------------------------------------------
  // 5. ftf_procprob.csv
  // -------------------------------------------------------------------------------------------
  const int b_procprob = new_bucket("GetProcProb", 1e-13);
  {
    const std::vector<std::string> lines = read_lines(dir + "/ftf_procprob.csv");
    if (lines.size() < 2) {
      std::printf("FAIL: cannot read %s/ftf_procprob.csv\n", dir.c_str());
      ++fails;
    }
    Header h;
    h.parse(lines[0]);
    int last_pdg = -1, last_z = -1, last_a = -1;
    double last_plab = -1.0;
    bool refused = false;
    long long rows = 0;
    for (std::size_t li = 1; li < lines.size(); ++li) {
      const std::vector<std::string> f = split(lines[li]);
      if (f.size() != h.n) { continue; }
      ++rows;
      const int pdg = iv(f, h.at("proj_pdg"));
      const int z = iv(f, h.at("tgt_z"));
      const int a = iv(f, h.at("tgt_a"));
      const double plab = dv(f, h.at("plab_mev"));
      if (pdg != last_pdg || z != last_z || a != last_a || plab != last_plab) {
        xs::Projectile<double> proj;
        refused = !projectile_for(pdg, 0.0, &proj);
        if (!refused) {
          ftf::ftf_init_for_interaction(par, proj, a, z, plab, lund);
          refused = (par->refused != ftf::FtfRefusal::kNone);
        }
        last_pdg = pdg; last_z = z; last_a = a; last_plab = plab;
      }
      if (refused) { ++n_refused; continue; }
      const int proc = iv(f, h.at("proc"));
      const double y = dv(f, h.at("y"));
      char wbuf[128];
      std::snprintf(wbuf, sizeof wbuf, "pdg=%d Z=%d A=%d plab=%g proc=%d y=%g", pdg, z, a,
                    plab, proc, y);
      cmp(b_procprob, ftf::ftf_get_proc_prob(par, proc, y), dv(f, h.at("prob")), wbuf);
    }
    if (rows != 21780) {
      std::printf("FAIL: ftf_procprob.csv has %lld rows, expected 21780\n", rows);
      ++fails;
    }
  }

  // -------------------------------------------------------------------------------------------
  // 6. ftf_geom.csv
  // -------------------------------------------------------------------------------------------
  const int b_gamma = new_bucket("GammaElasticAndProfiles", 1e-13);
  {
    const std::vector<std::string> lines = read_lines(dir + "/ftf_geom.csv");
    if (lines.size() < 2) {
      std::printf("FAIL: cannot read %s/ftf_geom.csv\n", dir.c_str());
      ++fails;
    }
    Header h;
    h.parse(lines[0]);
    int last_pdg = -1, last_z = -1, last_a = -1;
    double last_plab = -1.0;
    bool refused = false;
    for (std::size_t li = 1; li < lines.size(); ++li) {
      const std::vector<std::string> f = split(lines[li]);
      if (f.size() != h.n) { continue; }
      const int pdg = iv(f, h.at("proj_pdg"));
      const int z = iv(f, h.at("tgt_z"));
      const int a = iv(f, h.at("tgt_a"));
      const double plab = dv(f, h.at("plab_mev"));
      if (pdg != last_pdg || z != last_z || a != last_a || plab != last_plab) {
        xs::Projectile<double> proj;
        refused = !projectile_for(pdg, 0.0, &proj);
        if (!refused) {
          ftf::ftf_init_for_interaction(par, proj, a, z, plab, lund);
          refused = (par->refused != ftf::FtfRefusal::kNone);
        }
        last_pdg = pdg; last_z = z; last_a = a; last_plab = plab;
      }
      if (refused) { ++n_refused; continue; }
      const double b2 = dv(f, h.at("b2_fm2"));
      char wbuf[128];
      std::snprintf(wbuf, sizeof wbuf, "pdg=%d Z=%d A=%d plab=%g b2=%g", pdg, z, a, plab, b2);
      const std::string w = wbuf;
      cmp(b_gamma, ftf::ftf_gamma_elastic(par, b2), dv(f, h.at("gamma_elastic")), w + " gel");
      cmp(b_gamma, ftf::ftf_get_inelastic_probability(par, b2), dv(f, h.at("prob_inelastic")),
          w + " pinel");
      cmp(b_gamma, ftf::ftf_get_probability_of_interaction(par, b2), dv(f, h.at("prob_int")),
          w + " pint");
    }
  }

  // -------------------------------------------------------------------------------------------
  // 7. ftf_minmass.csv
  // -------------------------------------------------------------------------------------------
  const int b_mm = new_bucket("SetMinimalStringMass", 1e-14);
  const int b_frag = new_bucket("IsItFragmentablePredicate", 0.0);
  {
    const std::vector<std::string> lines = read_lines(dir + "/ftf_minmass.csv");
    if (lines.size() < 2) {
      std::printf("FAIL: cannot read %s/ftf_minmass.csv\n", dir.c_str());
      ++fails;
    }
    Header h;
    h.parse(lines[0]);
    long long rows = 0;
    for (std::size_t li = 1; li < lines.size(); ++li) {
      const std::vector<std::string> f = split(lines[li]);
      if (f.size() != h.n) { continue; }
      ++rows;
      const int lc = iv(f, h.at("left"));
      const int rc = iv(f, h.at("right"));
      const double m = dv(f, h.at("string_mass"));
      char wbuf[128];
      std::snprintf(wbuf, sizeof wbuf, "left=%d right=%d M=%g", lc, rc, m);
      const std::string w = wbuf;
      const ftf::MinimalStringMass<double> got = ftf::ftf_minimal_string_mass(lund, lc, rc, m);
      if (got.refused != ftf::FtfRefusal::kNone) {
        std::printf("FAIL: %s refused (%s) but the oracle answered\n", w.c_str(),
                    ftf::ftf_refusal_name(got.refused));
        ++fails;
        continue;
      }
      cmp(b_mm, got.mass, dv(f, h.at("minimal_mass")), w + " mass");
      cmp(b_mm, got.mass2, dv(f, h.at("minimal_mass2")), w + " mass2");
      // IsItFragmentable is `|MinimalStringMass| < Mass`, which the oracle dumps as a
      // separate column - so the predicate is compared, not re-derived from the mass here.
      // That is docs/RISK.md V52's lesson: a test line that mentions neither the module's
      // namespace nor one of its functions compares arithmetic against itself.
      const int want_frag = iv(f, h.at("fragmentable"));
      cmp_int(b_frag, (std::fabs(got.mass) < m) ? 1 : 0, want_frag, w + " fragmentable");
    }
    if (rows != 9000) {
      std::printf("FAIL: ftf_minmass.csv has %lld rows, expected 9000\n", rows);
      ++fails;
    }
  }

  // Every illegal parton pair must be refused, and the oracle skipped them - so this is the
  // one place the predicate's FALSE branch is exercised. 4 of the 16 quark pairs and 4 of the
  // 2500 diquark pairs are legal, so the count is not zero by construction.
  {
    long long n_illegal = 0, n_legal = 0;
    const int codes[] = {1, -1, 2, -2, 3, -3, 2101, -2101, 2103, -2103, 1103, -1103};
    for (int lc : codes) {
      for (int rc : codes) {
        const ftf::MinimalStringMass<double> got =
            ftf::ftf_minimal_string_mass(lund, lc, rc, 5000.0);
        if (got.refused == ftf::FtfRefusal::kIllegalPartonPair) { ++n_illegal; } else { ++n_legal; }
      }
    }
    if (n_illegal == 0 || n_legal == 0) {
      std::printf("FAIL: the parton-pair legality predicate answered one way for all %lld "
                  "pairs\n", n_illegal + n_legal);
      ++fails;
    }
  }

  // -------------------------------------------------------------------------------------------
  // Report
  // -------------------------------------------------------------------------------------------
  std::printf("\n%-32s %10s %14s  %s\n", "bucket", "points", "worst rel", "where");
  long long total = 0;
  for (const Bucket& b : buckets) {
    total += b.n;
    const bool bad = (b.worst > b.tol);
    if (bad) { ++fails; }
    std::printf("%-32s %10lld %14.3e%s %s\n", b.name, b.n, b.worst, bad ? " FAIL" : "",
                b.where.c_str());
    if (b.n == 0) {
      std::printf("FAIL: bucket %s compared nothing\n", b.name);
      ++fails;
    }
  }
  std::printf("\n%lld comparisons, %lld oracle rows refused by name "
              "(G4HadronNucleonXsc::HyperonNucleonXscNS)\n", total, n_refused);
  std::printf("%s\n", fails == 0 ? "PASS test_ftf_params" : "FAIL test_ftf_params");
  delete par;
  delete lund;
  return fails == 0 ? 0 : 1;
}
