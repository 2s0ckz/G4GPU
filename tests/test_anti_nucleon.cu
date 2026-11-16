// P23c: the anti-nucleon family's cross sections and elastic final state against the running
// Geant4 11.1.1 (`ref/dump/dump_anti_nucleon.cc`), and the anti-nucleus hand-over's rules.
//
//  1. G4ComponentAntiNuclNuclearXS: total, inelastic, elastic and the two anti-nucleon-nucleon
//     cross sections for pbar, nbar, anti-d, anti-t, anti-He3, anti-alpha on every element of
//     B1's four materials at the NIST mean mass, 1 keV-100 TeV (anti_xs.csv).
//  2. The wiring: hadElastic and the inelastic process per volume through `had::` for the same
//     six in the four materials, against G4HadronicProcessStore (anti_xs_volume.csv).
//  3. G4AntiNuclElastic::SampleInvariantT: its deterministic members (ptot, tmax, beta, n, Am,
//     Ra, Ref, ceff), GetcosTeta1, and 32 consecutive samples per point under the LCG both
//     sides run, with the draw count (anti_elastic.csv).
//  4. The angular-distribution campaign: 20,000 whole G4HadronElastic::ApplyYourself calls through
//     G4AntiNuclElastic per point, pbar on C12 and Pb208 at 0.3, 1 and 10 GeV and anti-d and
//     anti-alpha on C12 at 1 GeV/n - the lab cos(theta) histogram bin for bin, the draw count and
//     sum t (anti_elastic_hist.csv), plus the chi2 a statistical comparison would give.
//  5. Structure: the channels, model lists and integral-approach flag of the six; the
//     anti-nucleus residual re-typing of PropagateNuclNucl; the anti-neutron's decay table.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "data/materials.cuh"
#include "physics/decay/decay.cuh"
#include "physics/hadronic/elastic_wiring.cuh"
#include "physics/hadronic/inelastic_wiring.cuh"
#include "physics/hadronic/precompound/generator_interface.cuh"

using namespace g4gpu;
using real_t = double;
namespace hp = g4gpu::physics::hadronic;
namespace el = g4gpu::physics::hadronic::elastic;
namespace hxs = g4gpu::hadronic::xs;

static int g_fails = 0;
static void fail(const std::string& what) {
  if (g_fails < 40) { std::printf("  FAIL: %s\n", what.c_str()); }
  ++g_fails;
}

static std::string oracle_dir() {
  const char* env = std::getenv("G4GPU_ORACLE");
  return (env != nullptr) ? std::string(env) : std::string("ref/oracle");
}

static std::vector<std::vector<std::string>> load_csv(const std::string& path) {
  std::vector<std::vector<std::string>> rows;
  FILE* f = std::fopen(path.c_str(), "r");
  if (f == nullptr) { return rows; }
  static char line[1 << 16];
  std::fgets(line, sizeof line, f);
  while (std::fgets(line, sizeof line, f) != nullptr) {
    std::vector<std::string> out;
    std::string cur;
    for (char* c = line; *c; ++c) {
      if (*c == ',') { out.push_back(cur); cur.clear(); }
      else if (*c != '\r' && *c != '\n') { cur.push_back(*c); }
    }
    out.push_back(cur);
    rows.push_back(out);
  }
  std::fclose(f);
  return rows;
}

static double rel(double a, double b) {
  if (a == b) { return 0.0; }
  return std::fabs(a - b) / std::fmax(std::fabs(b), 1e-300);
}

/// dump_bertini.cc's LcgEngine, bit for bit.
struct Lcg {
  unsigned long long s = 0, n = 0;
  void reset(int phase) {
    s = 88172645463325252ull + 1442695040888963407ull * static_cast<unsigned long long>(phase + 1);
    n = 0;
  }
  double uniform() {
    s = 6364136223846793005ull * s + 1442695040888963407ull;
    ++n;
    return (static_cast<double>(s >> 11) + 0.5) * (1.0 / 9007199254740992.0);
  }
};

static ParticleType type_of_pdg_anti(int pdg) { return particle_type_of_pdg(pdg); }

static int material_of(const std::string& n) {
  if (n == "G4_AIR") { return data::kAir; }
  if (n == "G4_WATER") { return data::kWater; }
  if (n == "G4_A-150_TISSUE") { return data::kA150Tissue; }
  if (n == "G4_BONE_COMPACT_ICRU") { return data::kBoneCompact; }
  return -1;
}

int main() {
  const std::string dir = oracle_dir();
  // ------------------------------------------------------------------------- 1. the component
  std::printf("== 1. G4ComponentAntiNuclNuclearXS against Geant4 (anti_xs.csv) ==\n");
  {
    const auto rows = load_csv(dir + "/anti_xs.csv");
    if (rows.empty()) { fail("cannot read anti_xs.csv - run the anti_nucleon dump"); }
    double worst[5] = {0, 0, 0, 0, 0};
    std::string where[5];
    long long n = 0;
    const double mm2 = 1.0;  // the dump writes mm^2, the port's unit
    for (const auto& c : rows) {
      if (c.size() < 9) { continue; }
      const int pdg = std::atoi(c[0].c_str());
      const int Z = std::atoi(c[1].c_str());
      const double A = std::atof(c[2].c_str());
      const double e = std::atof(c[3].c_str());
      const hxs::Projectile<double> p = el::anuc_projectile<double>(pdg);
      const double o[5] = {hxs::anu_total_element<double>(p, e, Z, A).value / mm2,
                           hxs::anu_inelastic_element<double>(p, e, Z, A).value / mm2,
                           hxs::anu_elastic_element<double>(p, e, Z, A).value / mm2,
                           hxs::anu_nucleon_total<double>(p, e).total_mb,
                           hxs::anu_nucleon_elastic<double>(p, e)};
      for (int k = 0; k < 5; ++k) {
        const double g = std::atof(c[4 + k].c_str());
        // The elastic one is a difference of two numbers within a few per cent of each other
        // at low energy; its tolerance is the absolute one of its parents.
        const double d = (k == 2) ? std::fabs(o[k] - g) / std::fmax(std::atof(c[4].c_str()), 1e-300)
                                  : rel(o[k], g);
        if (d > worst[k]) {
          worst[k] = d;
          char b[160];
          std::snprintf(b, sizeof b, "pdg %d Z %d %g MeV: %.17g vs %.17g", pdg, Z, e, o[k], g);
          where[k] = b;
        }
      }
      ++n;
    }
    const char* nm[5] = {"total", "inelastic", "elastic (rel. to total)", "NN total", "NN elastic"};
    for (int k = 0; k < 5; ++k) {
      std::printf("  %-24s %lld points, worst %.3e  [%s]\n", nm[k], n, worst[k], where[k].c_str());
      if (worst[k] > 1e-12) { fail(std::string("component ") + nm[k]); }
    }
    if (n < 6LL * 10 * 111) { fail("anti_xs.csv row count " + std::to_string(n)); }
  }

  // ------------------------------------------------------------------------- 2. the wiring
  std::printf("== 2. hadElastic and Inelastic per volume through had:: (anti_xs_volume.csv) ==\n");
  {
    static data::Material<real_t> mats[data::kNumMaterials];
    data::build_b1_materials<real_t>(mats);
    const auto rows = load_csv(dir + "/anti_xs_volume.csv");
    if (rows.empty()) { fail("cannot read anti_xs_volume.csv"); }
    had::ElasticTables<real_t> elt{};
    had::InelasticTables<real_t> inl{};
    double w_el = 0, w_in = 0;
    std::string s_el, s_in;
    long long n = 0;
    for (const auto& c : rows) {
      if (c.size() < 5) { continue; }
      const int mi = material_of(c[0]);
      const ParticleType t = type_of_pdg_anti(std::atoi(c[1].c_str()));
      // The dump walks every material g4dump builds; B1's four are the ones compared.
      if (mi < 0) { continue; }
      if (t == ParticleType::kNumTypes) { fail("row " + c[0] + " " + c[1]); continue; }
      const double e = std::atof(c[2].c_str());
      hxs::MaterialXs<real_t> mxs{};
      const double o_el = had::elastic_xs_per_volume<real_t>(elt, mats[mi], t, e, mxs);
      const double o_in = had::inelastic_xs_per_volume<real_t>(inl, mats[mi], t, e, 0, 0, mxs);
      const double d_el = rel(o_el, std::atof(c[3].c_str()));
      const double d_in = rel(o_in, std::atof(c[4].c_str()));
      if (d_el > w_el) { w_el = d_el; s_el = c[0] + " " + particle_name(t) + " " + c[2]; }
      if (d_in > w_in) { w_in = d_in; s_in = c[0] + " " + particle_name(t) + " " + c[2]; }
      ++n;
    }
    std::printf("  hadElastic %lld points, worst %.3e  [%s]\n", n, w_el, s_el.c_str());
    std::printf("  Inelastic  %lld points, worst %.3e  [%s]\n", n, w_in, s_in.c_str());
    if (w_el > 1e-12) { fail("hadElastic per volume"); }
    if (w_in > 1e-12) { fail("Inelastic per volume"); }
    if (n != 4LL * 6 * 111) { fail("anti_xs_volume.csv row count " + std::to_string(n)); }
  }

  // ------------------------------------------------------------------- 3. SampleInvariantT
  std::printf("== 3. G4AntiNuclElastic::SampleInvariantT, pieces and 32 samples (anti_elastic.csv) ==\n");
  {
    const auto rows = load_csv(dir + "/anti_elastic.csv");
    if (rows.empty()) { fail("cannot read anti_elastic.csv"); }
    double w_piece = 0, w_t = 0;
    std::string s_piece, s_t;
    long long n_pts = 0, n_t = 0, draw_mism = 0, n_coul = 0, n_strong = 0;
    for (const auto& c : rows) {
      if (c.size() < 15 + 32) { continue; }
      const int pdg = std::atoi(c[0].c_str()), Z = std::atoi(c[1].c_str()),
                A = std::atoi(c[2].c_str());
      const double plab = std::atof(c[4].c_str());
      const el::AnucState<double> s = el::anuc_prepare<double>(pdg, plab, Z, A);
      char at[120];
      std::snprintf(at, sizeof at, "pdg %d Z %d A %d plab %g", pdg, Z, A, plab);
      // ctet1, ptot, tmax always; beta, n, am once the momentum is above the low branch; the
      // strong arm's ra, ref, ceff only if a strong sample was taken (they are members Geant4
      // sets inside that arm, zeroed by the dump before the first call).
      const double ours[9] = {el::anuc_cos_teta1<double>(plab, A), s.ptot, s.tmax, s.beta, s.n,
                              s.am, s.ra, s.ref, s.ceff};
      for (int k = 0; k < 9; ++k) {
        const double g = std::atof(c[5 + k].c_str());
        if (k >= 3 && s.low_momentum) { continue; }
        if (k >= 6 && g == 0.0) { continue; }
        const double d = (k == 4 && g == 0.0) ? std::fabs(ours[k]) : rel(ours[k], g);
        if (d > w_piece) { w_piece = d; s_piece = std::string(at) + " piece " + std::to_string(k); }
      }
      Lcg r;
      r.reset(0);
      for (int k = 0; k < 32; ++k) {
        const double g = std::atof(c[15 + k].c_str());
        const double before = double(r.n);
        const double t = el::anuc_sample_invariant_t<double>(pdg, plab, Z, A, r);
        if (!s.low_momentum) {
          if (double(r.n) - before >= 4) { ++n_strong; } else { ++n_coul; }
        }
        const double d = (g == 0.0) ? std::fabs(t) : rel(t, g);
        if (d > w_t) { w_t = d; s_t = std::string(at) + " sample " + std::to_string(k); }
        ++n_t;
      }
      if (static_cast<long long>(r.n) != std::atoll(c[14].c_str())) {
        ++draw_mism;
        if (draw_mism < 4) { fail(std::string("draw count ") + at); }
      }
      ++n_pts;
    }
    std::printf("  %lld points: deterministic pieces worst %.3e  [%s]\n", n_pts, w_piece,
                s_piece.c_str());
    std::printf("  %lld samples (%lld with a Coulomb-or-short draw pattern, %lld strong): worst "
                "%.3e  [%s]; draw-count mismatches %lld\n",
                n_t, n_coul, n_strong, w_t, s_t.c_str(), draw_mism);
    if (w_piece > 1e-12) { fail("SampleInvariantT pieces"); }
    if (w_t > 1e-9) { fail("SampleInvariantT samples"); }
    if (draw_mism > 0) { fail("SampleInvariantT draw count"); }
    if (n_pts != 6LL * 7 * 7) { fail("anti_elastic.csv row count " + std::to_string(n_pts)); }
  }

  // ----------------------------------------------------------------------- 4. the campaign
  std::printf("== 4. 20,000 ApplyYourself calls per point through G4AntiNuclElastic "
              "(anti_elastic_hist.csv) ==\n");
  {
    const auto rows = load_csv(dir + "/anti_elastic_hist.csv");
    if (rows.empty()) { fail("cannot read anti_elastic_hist.csv"); }
    for (const auto& c : rows) {
      if (c.size() < 8 + 50) { continue; }
      const int pdg = std::atoi(c[0].c_str()), Z = std::atoi(c[1].c_str()),
                A = std::atoi(c[2].c_str());
      const double ekin = std::atof(c[3].c_str());
      const int N = std::atoi(c[4].c_str());
      const hxs::Projectile<double> pj = el::anuc_projectile<double>(pdg);
      hp::HadProjectile<double> proj;
      proj.pdg = pdg;
      proj.baryon_number = pj.baryon_number;
      proj.charge = pj.charge;
      proj.mass = pj.mass;
      proj.kin_energy = ekin;
      hp::HadNucleus nuc;
      nuc.z = Z;
      nuc.a = A;
      auto nuclear_mass = [](int z, int a) { return data::nuclear_mass<double>(a, z); };
      auto recoil_species = [](int z, int a, int* p, double* m) {
        *p = hp::pdg_nuclear_code(z, a);
        *m = data::nuclear_mass<double>(a, z);
      };
      auto sampler = [&](int pd, double plab, int z, int a, double, Lcg& rr) -> double {
        return el::anuc_sample_invariant_t<double>(pd, plab, z, a, rr);
      };
      Lcg r;
      r.reset(7);
      long long hist[50] = {0};
      double sum_t = 0, sum_ef = 0;
      for (int i = 0; i < N; ++i) {
        hp::HadFinalState<double, 1> fs;
        (void)el::hadron_elastic_apply_yourself<double, 1>(proj, nuc, sampler, nuclear_mass,
                                                            recoil_species, 0.0, 0, r, &fs);
        int b = int((fs.momentum_change.z + 1.0) * 25.0);
        b = (b < 0) ? 0 : (b > 49 ? 49 : b);
        ++hist[b];
        sum_ef += fs.energy_change;
        const double er = (fs.n_secondaries > 0) ? fs.secondaries[0].kin_energy
                                                 : fs.local_energy_deposit;
        sum_t += 2.0 * data::nuclear_mass<double>(A, Z) * er;
      }
      long long mism = 0;
      double chi2 = 0;
      int ndf = 0;
      for (int b = 0; b < 50; ++b) {
        const long long g = std::atoll(c[8 + b].c_str());
        if (g != hist[b]) { ++mism; }
        if (g + hist[b] > 0) {
          chi2 += double(g - hist[b]) * double(g - hist[b]) / double(g + hist[b]);
          ++ndf;
        }
      }
      const long long gd = std::atoll(c[5].c_str());
      std::printf("  pdg %11d on Z=%2d A=%3d at %8.1f MeV: bins differing %lld of 50, chi2 %.2f / "
                  "%d, draws %llu vs %lld, sum t rel %.2e, sum Ef rel %.2e\n",
                  pdg, Z, A, ekin, mism, chi2, ndf, r.n, gd, rel(sum_t, std::atof(c[6].c_str())),
                  rel(sum_ef, std::atof(c[7].c_str())));
      if (mism > 0 || static_cast<long long>(r.n) != gd) { fail("campaign point not exact"); }
      if (rel(sum_t, std::atof(c[6].c_str())) > 1e-9) { fail("campaign sum t"); }
    }
  }

  // ----------------------------------------------------------------------- 5. structure
  std::printf("== 5. structure: channels, models, integral flag, hand-over re-typing, decay ==\n");
  {
    const ParticleType six[6] = {ParticleType::kAntiProton, ParticleType::kAntiNeutron,
                                 ParticleType::kAntiDeuteron, ParticleType::kAntiTriton,
                                 ParticleType::kAntiHe3, ParticleType::kAntiAlpha};
    for (ParticleType t : six) {
      if (had::elastic_channel(t) != had::ElasticChannel::kAntiNucleus) {
        fail(std::string("elastic channel of ") + particle_name(t));
      }
      if (had::inelastic_channel(t) != had::InelasticChannel::kAntiNucleus) {
        fail(std::string("inelastic channel of ") + particle_name(t));
      }
      const auto L = had::inelastic_models<double>(t);
      if (L.n != 1 || L.model[0] != had::InelasticModel::kFtfp || L.range[0].min_energy != 0.0
          || L.range[0].max_energy != 1e8) {
        fail(std::string("model list of ") + particle_name(t));
      }
      const hxs::Projectile<double> pj = had::inelastic_projectile<double>(t, 0, 0);
      const hp::HadXsType xt = had::inelastic_xs_type<double>(pj);
      // G4HadronicProcess::BuildPhysicsTable: charged and (atomic number != 0 or mass < 1 GeV),
      // and the anti-ions' atomic number is |Z| (G4Ions.cc:96) - so every charged one is
      // fHadDecreasing and the anti-neutron fHadNoIntegral.
      const hp::HadXsType want = (pj.charge == 0.0) ? hp::HadXsType::kNoIntegral
                                                    : hp::HadXsType::kDecreasing;
      if (xt != want) { fail(std::string("integral flag of ") + particle_name(t)); }
      if (pdg_code(t) != pj.pdg || std::fabs(particle_def<double>(t).mass - pj.mass) > 1e-9) {
        fail(std::string("row against projectile for ") + particle_name(t));
      }
    }
    // PropagateNuclNucl's re-typing: the six, and "anything else keeps its definition".
    struct R { int pdg, z, a, want; };
    const R rr[] = {{2212, 1, 1, -2212}, {2112, 0, 1, -2112}, {1000010020, 1, 2, -1000010020},
                    {1000010030, 1, 3, -1000010030}, {1000020030, 2, 3, -1000020030},
                    {1000020040, 2, 4, -1000020040}, {22, 0, 0, 22},
                    {1000000020, 0, 2, 1000000020}, {1000030060, 3, 6, 1000030060}};
    for (const R& x : rr) {
      if (preco::propagate_nucl_nucl_anti_species(x.pdg, x.z, x.a) != x.want) {
        fail("anti re-typing of " + std::to_string(x.pdg));
      }
    }
    // The anti-neutron's decay: G4NeutronBetaDecayChannel("anti_neutron") into e+ nu_e pbar.
    const decay::DecayTableRow row = decay::decay_table_for(-2112);
    if (row.count != 1) {
      fail("anti_neutron decay table");
    } else {
      const decay::ChannelRow& ch = decay::channel_rows()[row.first];
      if (ch.kind != decay::ChannelKind::kNeutronBeta || ch.daughter[0] != -11
          || ch.daughter[1] != 12 || ch.daughter[2] != -2212) {
        fail("anti_neutron decay channel");
      }
    }
    // hadElastic's two models and their windows, against the constructed QBBC (anti_scope.csv):
    // hElasticLHEP [0, 100.1] then AntiAElastic [100, 1e8], in registration order, for all six.
    // `elastic_apply` spells the same four numbers; this row is what keeps them Geant4's.
    {
      const auto rows = load_csv(dir + "/anti_scope.csv");
      if (rows.empty()) { fail("cannot read anti_scope.csv"); }
      int n_ok = 0;
      for (ParticleType t : six) {
        std::vector<std::vector<std::string>> el;
        for (const auto& c : rows) {
          if (c.size() >= 18 && c[0] == particle_name(t) && c[11] == "hadElastic") { el.push_back(c); }
        }
        const bool ok = el.size() == 2 && el[0][15] == "hElasticLHEP"
                        && std::atof(el[0][16].c_str()) == 0.0
                        && std::atof(el[0][17].c_str()) == 100.1
                        && el[1][15] == "AntiAElastic" && std::atof(el[1][16].c_str()) == 100.0
                        && std::atof(el[1][17].c_str()) == 1e8;
        if (!ok) { fail(std::string("hadElastic models of ") + particle_name(t)); } else { ++n_ok; }
      }
      std::printf("  hadElastic windows equal Geant4's for %d of 6 species\n", n_ok);
    }
    std::printf("  six species: elastic and inelastic AntiAGlauber, FTFP 0-100 TeV alone, the "
                "integral flag; re-typing 9 cases; anti_neutron -> e+ nu_e anti_proton\n");
  }

  std::printf("\n%s (%d failure%s)\n", g_fails == 0 ? "ALL OK" : "FAILED", g_fails,
              g_fails == 1 ? "" : "s");
  return g_fails == 0 ? 0 : 1;
}
