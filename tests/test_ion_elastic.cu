// P25: ionElastic - G4IonElasticPhysics' process on GenericIon - wired, against the running
// Geant4 11.1.1 (`ref/dump/dump_ion_elastic.cc`).
//
//  1. Scope: which species HAS ionElastic in the constructed QBBC (GenericIon and a real C12 ion;
//     d, t, He3 and alpha keep hadElastic), its one model and window, and the integral-approach
//     type the process runs with - against `had::elastic_channel` and the flag `elastic_apply`
//     evaluates (ion_elastic_scope.csv).
//  2. G4ComponentGGNuclNuclXsc's elastic and inelastic element cross sections for eighteen real
//     nuclides on every element of B1's four materials at the NIST mean mass, 0.01 MeV/u -
//     100 GeV/u (ion_elastic_xs.csv). The component is V already (PORTED.md); this is the
//     projectile `xs::generic_ion(Z, A)` the wiring hands it, at the real ions' charges.
//  3. The wiring: `had::elastic_xs_per_volume(kGenericIon, ..., Z, A)` against
//     G4HadronicProcessStore::GetElasticCrossSectionPerVolume for the same eighteen in B1's
//     four materials (ion_elastic_xs_volume.csv).
//  4. The REAL G4HadronElasticProcess::PostStepDoIt of ionElastic replayed under the LCG both
//     sides run: `had::elastic_apply(kGenericIon, ..., Z, A)` per row - the integral rejection
//     against the cross section at the step's start, SampleZandA, G4NuclNuclDiffuseElastic's
//     t, phi, the rotation, the recoil threshold, the recoil's species, energy and direction -
//     with the draw count (ion_elastic_apply.csv). 2,680 grid rows, of which none emits a
//     recoil above the 70 keV cut, and 48 rows the dump FOUND by scanning phases where one is
//     emitted - a proton, or a C12, N14 or O16 GenericIon.
//  5. The campaign: 20,000 PostStepDoIt calls per point on one stream, through elastic_apply -
//     draws, sums, recoil counts and two histograms bin for bin (ion_elastic_hist.csv).
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "data/materials.cuh"
#include "physics/hadronic/elastic_wiring.cuh"

using namespace g4gpu;
using real_t = double;
namespace hp = g4gpu::physics::hadronic;
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

static int material_of(const std::string& n) {
  if (n == "G4_AIR") { return data::kAir; }
  if (n == "G4_WATER") { return data::kWater; }
  if (n == "G4_A-150_TISSUE") { return data::kA150Tissue; }
  if (n == "G4_BONE_COMPACT_ICRU") { return data::kBoneCompact; }
  return -1;
}

/// A Geant4 PDG code as (Z, A): the proton's 2212 and the five light codes, or 100ZZZAAAI.
static void za_of_pdg(int pdg, int* z, int* a) {
  if (pdg == 2212) { *z = 1; *a = 1; return; }
  *z = (pdg / 10000) % 1000;
  *a = (pdg / 10) % 1000;
}

/// The dump's direction, (0.3, -0.4, sqrt(1 - 0.25)), with no renormalisation: G4ThreeVector
/// keeps what it is handed and G4DynamicParticle stores it as given.
static const Vec3<real_t> kDir{0.3, -0.4, std::sqrt(1.0 - 0.25)};
/// QBBC's default range cut, which `proton_recoil_cut_energy` turns into the 70 keV threshold
/// G4HadronElasticProcess reads from the proton's energy-cuts vector.
static const real_t kRangeCutMm = 0.7;

int main() {
  const std::string dir = oracle_dir();
  static data::Material<real_t> mats[data::kNumMaterials];
  data::build_b1_materials<real_t>(mats);
  const had::ElasticTables<real_t> elt{};

  // ------------------------------------------------------------------------------- 1. scope
  std::printf("== 1. which species has ionElastic, its model and its integral flag "
              "(ion_elastic_scope.csv) ==\n");
  {
    const auto rows = load_csv(dir + "/ion_elastic_scope.csv");
    if (rows.empty()) { fail("cannot read ion_elastic_scope.csv - run the ion_elastic dump"); }
    int n_ion_el = 0, n_light_el = 0, n_light_ion_el = 0;
    int ion_el_index = -1, ion_inel_index = -1, gi_row = 0;
    for (const auto& c : rows) {
      if (c.size() < 7) { continue; }
      const bool ion_row = (c[0] == "GenericIon" || c[0] == "C12");
      if (c[0] == "GenericIon") {
        if (c[1] == "ionInelastic" && ion_inel_index < 0) { ion_inel_index = gi_row; }
        if (c[1] == "ionElastic") { ion_el_index = gi_row; }
        ++gi_row;
      }
      if (c[1] == "ionElastic") {
        if (!ion_row) { ++n_light_ion_el; continue; }
        ++n_ion_el;
        // One model, G4NuclNuclDiffuseElastic ("NNDiffuseElastic"), SetMinEnergy(0) by the
        // constructor and the hadronic maximum, 100 TeV; fHadIncreasing (1) on the process.
        if (c[4] != "NNDiffuseElastic" || std::atof(c[5].c_str()) != 0.0
            || std::atof(c[6].c_str()) != 1e8) {
          fail("ionElastic model row " + c[0] + " " + c[4]);
        }
        const int want = static_cast<int>(hp::HadXsType::kIncreasing);
        if (std::atoi(c[3].c_str()) != want) { fail("ionElastic xs type " + c[3]); }
      }
      if (c[1] == "hadElastic" && !ion_row) { ++n_light_el; }
    }
    if (n_ion_el != 2) { fail("ionElastic rows for GenericIon and C12: " + std::to_string(n_ion_el)); }
    if (n_light_ion_el != 0) { fail("a light ion has ionElastic"); }
    if (n_light_el != 4) { fail("hadElastic rows for d, t, He3, alpha: " + std::to_string(n_light_el)); }
    // GenericIon's manager holds ionInelastic BEFORE ionElastic - the order the stepper's tie
    // note in step_hadron describes.
    if (!(ion_inel_index >= 0 && ion_el_index > ion_inel_index)) {
      fail("GenericIon's process order: ionInelastic then ionElastic");
    }
    // The port's side of the same statements.
    if (had::elastic_channel(ParticleType::kGenericIon) != had::ElasticChannel::kIonDiffuse) {
      fail("elastic_channel(kGenericIon)");
    }
    if (!had::elastic_channel_has_xs(had::elastic_channel(ParticleType::kGenericIon))) {
      fail("kIonDiffuse has no cross section");
    }
    const ParticleType light[4] = {ParticleType::kDeuteron, ParticleType::kTriton,
                                   ParticleType::kHe3, ParticleType::kAlpha};
    for (ParticleType t : light) {
      if (had::elastic_channel(t) != had::ElasticChannel::kGheishaLightIon) {
        fail(std::string("light-ion channel of ") + particle_name(t));
      }
    }
    // The flag elastic_apply evaluates for the row - G4GenericIon's own definition - and the one
    // the ion's own projectile would give: both fHadIncreasing, for every nuclide in the grid.
    const ParticleDef<real_t> gi = particle_def<real_t>(ParticleType::kGenericIon);
    if (hp::hadronic_xs_type<real_t>(pdg_code(ParticleType::kGenericIon), gi.charge, gi.mass, 1,
                                     false, true) != hp::HadXsType::kIncreasing) {
      fail("GenericIon's integral flag");
    }
    const int nz[6][2] = {{3, 7}, {6, 12}, {7, 14}, {8, 16}, {26, 56}, {82, 208}};
    for (const auto& q : nz) {
      const hxs::Projectile<real_t> pj = had::elastic_projectile<real_t>(ParticleType::kGenericIon,
                                                                         q[0], q[1]);
      if (hp::hadronic_xs_type<real_t>(pj.pdg, pj.charge, pj.mass,
                                       hp::xs_type_atomic_number(pj.baryon_number), false,
                                       true) != hp::HadXsType::kIncreasing) {
        fail("the ion's own flag at Z=" + std::to_string(q[0]));
      }
    }
    std::printf("  ionElastic on GenericIon and C12 only, NNDiffuseElastic [0, 1e8 MeV], "
                "fHadIncreasing; d/t/He3/alpha hadElastic; ionInelastic ahead of ionElastic\n");
  }

  // --------------------------------------------------------------------------- 2. component
  std::printf("== 2. G4ComponentGGNuclNuclXsc on the real ions (ion_elastic_xs.csv) ==\n");
  {
    const auto rows = load_csv(dir + "/ion_elastic_xs.csv");
    if (rows.empty()) { fail("cannot read ion_elastic_xs.csv"); }
    double w_el = 0, w_in = 0;
    std::string s_el, s_in;
    long long n = 0, n_nonzero = 0;
    for (const auto& c : rows) {
      if (c.size() < 7) { continue; }
      const int zp = std::atoi(c[0].c_str()), ap = std::atoi(c[1].c_str());
      const int zt = std::atoi(c[2].c_str());
      const double at = std::atof(c[3].c_str()), e = std::atof(c[4].c_str());
      const hxs::Projectile<real_t> pj = hxs::generic_ion<real_t>(zp, ap);
      const double o_el = hxs::ggnn_elastic_element<real_t>(pj, e, zt, at).value;
      const double o_in = hxs::ggnn_inelastic_element<real_t>(pj, e, zt, at).value;
      const double g_el = std::atof(c[5].c_str()), g_in = std::atof(c[6].c_str());
      // The elastic one is a difference (total - inelastic, floored at zero); its scale is the
      // inelastic one's, so it is compared relative to that where it is not itself the scale.
      const double d_el = std::fabs(o_el - g_el) / std::fmax(std::fmax(g_el, g_in), 1e-300);
      const double d_in = rel(o_in, g_in);
      char at_s[96];
      std::snprintf(at_s, sizeof at_s, "Z=%d A=%d on Z=%d at %g MeV", zp, ap, zt, e);
      if (d_el > w_el) { w_el = d_el; s_el = at_s; }
      if (d_in > w_in) { w_in = d_in; s_in = at_s; }
      if (g_el > 0) { ++n_nonzero; }
      ++n;
    }
    std::printf("  elastic   %lld points (%lld non-zero), worst %.3e  [%s]\n", n, n_nonzero, w_el,
                s_el.c_str());
    std::printf("  inelastic %lld points, worst %.3e  [%s]\n", n, w_in, s_in.c_str());
    if (w_el > 1e-12) { fail("component elastic"); }
    if (w_in > 1e-12) { fail("component inelastic"); }
    if (n < 18LL * 71 * 8) { fail("ion_elastic_xs.csv row count " + std::to_string(n)); }
  }

  // ------------------------------------------------------------------------------ 3. wiring
  std::printf("== 3. ionElastic per volume through had:: (ion_elastic_xs_volume.csv) ==\n");
  {
    const auto rows = load_csv(dir + "/ion_elastic_xs_volume.csv");
    if (rows.empty()) { fail("cannot read ion_elastic_xs_volume.csv"); }
    double w = 0;
    std::string s;
    long long n = 0, n_nonzero = 0;
    for (const auto& c : rows) {
      if (c.size() < 5) { continue; }
      const int mi = material_of(c[0]);
      if (mi < 0) { continue; }  // g4dump's custom materials are not B1's
      const int z = std::atoi(c[1].c_str()), a = std::atoi(c[2].c_str());
      const double e = std::atof(c[3].c_str()), g = std::atof(c[4].c_str());
      hxs::MaterialXs<real_t> mxs{};
      const double o = had::elastic_xs_per_volume<real_t>(elt, mats[mi], ParticleType::kGenericIon,
                                                          e, mxs, z, a);
      const double d = rel(o, g);
      if (d > w) {
        w = d;
        char b[128];
        std::snprintf(b, sizeof b, "%s Z=%d A=%d %g MeV: %.17g vs %.17g", c[0].c_str(), z, a, e,
                      o, g);
        s = b;
      }
      if (g > 0) { ++n_nonzero; }
      ++n;
    }
    std::printf("  %lld points in B1's four materials (%lld non-zero), worst %.3e  [%s]\n", n,
                n_nonzero, w, s.c_str());
    if (w > 1e-12) { fail("ionElastic per volume"); }
    if (n != 4LL * 18 * 71) { fail("ion_elastic_xs_volume.csv row count " + std::to_string(n)); }
    if (n_nonzero < n / 2) { fail("most of the grid is zero - the wiring is not reached"); }
  }

  // ------------------------------------------------------------- 4. PostStepDoIt, replayed
  std::printf("== 4. ionElastic's PostStepDoIt replayed under the LCG (ion_elastic_apply.csv) ==\n");
  {
    const auto rows = load_csv(dir + "/ion_elastic_apply.csv");
    if (rows.empty()) { fail("cannot read ion_elastic_apply.csv"); }
    long long n = 0, n_rej = 0, n_int = 0, n_rec = 0, n_rec_p = 0, n_rec_ion = 0;
    long long draw_mism = 0, fate_mism = 0, rec_mism = 0;
    double w_xs = 0, w_e = 0, w_dir = 0, w_dep = 0, w_rec_e = 0, w_rec_dir = 0;
    std::string s_e, s_dir, s_dep, s_rec;
    for (const auto& c : rows) {
      if (c.size() < 23) { continue; }
      const int mi = material_of(c[0]);
      if (mi < 0) { fail("material " + c[0]); continue; }
      const int z = std::atoi(c[1].c_str()), a = std::atoi(c[2].c_str());
      const double e_start = std::atof(c[3].c_str()), e_end = std::atof(c[4].c_str());
      const int phase = std::atoi(c[5].c_str());
      const long long g_draws = std::atoll(c[7].c_str());
      const double g_ef = std::atof(c[8].c_str());
      const Vec3<real_t> g_dir{std::atof(c[9].c_str()), std::atof(c[10].c_str()),
                               std::atof(c[11].c_str())};
      const double g_edep = std::atof(c[12].c_str()), g_niel = std::atof(c[13].c_str());
      const int g_status = std::atoi(c[14].c_str()), g_nsec = std::atoi(c[15].c_str());
      const int g_sec_pdg = std::atoi(c[18].c_str());
      const double g_sec_e = std::atof(c[19].c_str());
      const Vec3<real_t> g_sdir{std::atof(c[20].c_str()), std::atof(c[21].c_str()),
                                std::atof(c[22].c_str())};
      char at[128];
      std::snprintf(at, sizeof at, "%s Z=%d A=%d %g->%g MeV phase %d", c[0].c_str(), z, a,
                    e_start, e_end, phase);

      // The cross section the interaction length was drawn with, at the step's START energy,
      // with its partial sums - what step_hadron hands elastic_apply.
      hxs::MaterialXs<real_t> mxs{};
      const double xs_start = had::elastic_xs_per_volume<real_t>(
          elt, mats[mi], ParticleType::kGenericIon, e_start, mxs, z, a);
      w_xs = std::fmax(w_xs, rel(xs_start, std::atof(c[6].c_str())));

      Lcg r;
      r.reset(phase);
      const had::ElasticStepOutcome<real_t> o = had::elastic_apply<real_t>(
          elt, mats[mi], ParticleType::kGenericIon, e_end, kDir, kRangeCutMm, xs_start, mxs, r,
          z, a);
      ++n;
      if (static_cast<long long>(r.n) != g_draws) {
        ++draw_mism;
        if (draw_mism <= 3) {
          fail(std::string("draws ") + at + ": " + std::to_string(r.n) + " vs " +
               std::to_string(g_draws));
        }
      }
      // A rejection consumes exactly the one uniform and changes nothing.
      const bool g_rejected = (g_draws == 1);
      if (g_rejected) { ++n_rej; } else { ++n_int; }
      if (o.rejected_by_integral_xs != g_rejected || o.interacted == g_rejected) {
        ++fate_mism;
        if (fate_mism <= 3) { fail(std::string("rejected/interacted ") + at); }
        continue;
      }
      if (g_rejected) { continue; }
      const double de = rel(o.energy, g_ef);
      if (de > w_e) { w_e = de; s_e = at; }
      const double dd = std::fmax(std::fabs(o.dir.x - g_dir.x),
                                  std::fmax(std::fabs(o.dir.y - g_dir.y),
                                            std::fabs(o.dir.z - g_dir.z)));
      if (dd > w_dir) { w_dir = dd; s_dir = at; }
      // Both deposits carry the same number in Geant4 (local and non-ionizing); the port keeps
      // one and the stepper books it as both.
      if (g_edep != g_niel) { fail(std::string("Geant4's two deposits differ ") + at); }
      const double ddep = std::fabs(o.edep - g_edep) / std::fmax(g_edep, 1e-300);
      if (ddep > w_dep) { w_dep = ddep; s_dep = at; }
      if (o.primary_survives != (g_status == 0)) { fail(std::string("status ") + at); }
      if (o.emit_recoil != (g_nsec == 1)) {
        ++rec_mism;
        if (rec_mism <= 3) { fail(std::string("recoil emitted ") + at); }
        continue;
      }
      if (!o.emit_recoil) { continue; }
      ++n_rec;
      int gz = 0, ga = 0;
      za_of_pdg(g_sec_pdg, &gz, &ga);
      if (gz == 1 && ga == 1) { ++n_rec_p; } else { ++n_rec_ion; }
      if (o.recoil_z != gz || o.recoil_a != ga) {
        fail(std::string("recoil species ") + at + ": " + std::to_string(o.recoil_z) + "," +
             std::to_string(o.recoil_a) + " vs pdg " + std::to_string(g_sec_pdg));
      }
      // The species the stepper's push_nucleus gives it: G4HadronElastic::ApplyYourself names the
      // proton (and d, t, He3, alpha) and takes every heavier nuclide from GetIon(Z, A, 0).
      const ParticleType want = (gz == 1 && ga == 1) ? ParticleType::kProton
                                                     : ParticleType::kGenericIon;
      if (particle_type_of_nucleus(o.recoil_z, o.recoil_a) != want) {
        fail(std::string("recoil track species ") + at);
      }
      const double dre = rel(o.recoil_ekin, g_sec_e);
      if (dre > w_rec_e) { w_rec_e = dre; s_rec = at; }
      const double drd = std::fmax(std::fabs(o.recoil_dir.x - g_sdir.x),
                                   std::fmax(std::fabs(o.recoil_dir.y - g_sdir.y),
                                             std::fabs(o.recoil_dir.z - g_sdir.z)));
      w_rec_dir = std::fmax(w_rec_dir, drd);
    }
    std::printf("  %lld calls: %lld rejected by the integral approach, %lld scattered, %lld "
                "emitted a recoil above 70 keV (%lld protons, %lld GenericIons)\n", n, n_rej,
                n_int, n_rec, n_rec_p, n_rec_ion);
    std::printf("  draw-count mismatches %lld, fate mismatches %lld, recoil mismatches %lld\n",
                draw_mism, fate_mism, rec_mism);
    std::printf("  xs at the step's start worst %.3e; Ef worst %.3e [%s]; direction worst %.3e "
                "[%s]\n", w_xs, w_e, s_e.c_str(), w_dir, s_dir.c_str());
    std::printf("  deposit (the sub-threshold recoil) worst %.3e [%s]; recoil energy worst %.3e "
                "[%s], direction %.3e\n", w_dep, s_dep.c_str(), w_rec_e, s_rec.c_str(),
                w_rec_dir);
    if (draw_mism + fate_mism + rec_mism > 0) { fail("PostStepDoIt replay"); }
    if (w_xs > 1e-12 || w_e > 1e-12 || w_dir > 1e-12 || w_dep > 1e-9 || w_rec_e > 1e-9
        || w_rec_dir > 1e-9) {
      fail("PostStepDoIt replay tolerance");
    }
    if (n_rej == 0) { fail("the integral rejection never fired"); }
    if (n_rec_p == 0 || n_rec_ion == 0) { fail("a recoil species never emitted"); }
    if (n < 2600) { fail("ion_elastic_apply.csv row count " + std::to_string(n)); }
  }

  // ------------------------------------------------------------------------ 5. the campaign
  std::printf("== 5. 20,000 PostStepDoIt calls per point (ion_elastic_hist.csv) ==\n");
  {
    const auto rows = load_csv(dir + "/ion_elastic_hist.csv");
    if (rows.empty()) { fail("cannot read ion_elastic_hist.csv"); }
    for (const auto& c : rows) {
      if (c.size() < 13 + 80) { continue; }
      const int mi = material_of(c[0]);
      const int z = std::atoi(c[1].c_str()), a = std::atoi(c[2].c_str());
      const double e = std::atof(c[3].c_str());
      const int N = std::atoi(c[4].c_str());
      hxs::MaterialXs<real_t> mxs{};
      const double xs = had::elastic_xs_per_volume<real_t>(elt, mats[mi], ParticleType::kGenericIon,
                                                           e, mxs, z, a);
      Lcg r;
      r.reset(11);
      long long er[40] = {0}, th[40] = {0};
      long long n_sec = 0, n_p = 0, n_ion = 0, n_below = 0;
      double sum_ef = 0, sum_er = 0, sum_ed = 0;
      for (int i = 0; i < N; ++i) {
        const had::ElasticStepOutcome<real_t> o = had::elastic_apply<real_t>(
            elt, mats[mi], ParticleType::kGenericIon, e, kDir, kRangeCutMm, xs, mxs, r, z, a);
        // A rejection proposes nothing and the dump's particle change then reports the track's
        // own energy and direction - which is what `o` carries on that path too.
        sum_ef += o.energy;
        sum_ed += o.edep;
        const double cth = o.dir.x * kDir.x + o.dir.y * kDir.y + o.dir.z * kDir.z;
        const double omc = 1.0 - cth;
        int bt = (omc > 0.0) ? int((std::log10(omc) + 16.0) * 2.5) : 0;
        bt = bt < 0 ? 0 : (bt > 39 ? 39 : bt);
        ++th[bt];
        if (o.emit_recoil) {
          ++n_sec;
          if (o.recoil_z == 1 && o.recoil_a == 1) { ++n_p; } else { ++n_ion; }
          sum_er += o.recoil_ekin;
          int b = int((std::log10(o.recoil_ekin) + 2.0) * (40.0 / 6.0));
          b = b < 0 ? 0 : (b > 39 ? 39 : b);
          ++er[b];
        } else if (o.edep > 0.0) {
          ++n_below;
        }
      }
      long long mism = 0;
      for (int b = 0; b < 40; ++b) {
        if (er[b] != std::atoll(c[13 + b].c_str())) { ++mism; }
        if (th[b] != std::atoll(c[53 + b].c_str())) { ++mism; }
      }
      const long long gd = std::atoll(c[5].c_str());
      const bool counts_ok = n_sec == std::atoll(c[6].c_str()) && n_p == std::atoll(c[7].c_str())
                             && n_ion == std::atoll(c[8].c_str())
                             && n_below == std::atoll(c[9].c_str());
      const double d_ef = rel(sum_ef, std::atof(c[10].c_str()));
      const double d_er = rel(sum_er, std::atof(c[11].c_str()));
      const double d_ed = rel(sum_ed, std::atof(c[12].c_str()));
      std::printf("  %-21s Z=%d A=%3d %8.0f MeV: bins differing %lld of 80, draws %llu vs %lld, "
                  "recoils %lld (%lld below the cut), sum Ef %.1e, sum edep %.1e\n",
                  c[0].c_str(), z, a, e, mism, r.n, gd, n_sec, n_below, d_ef, d_ed);
      if (mism > 0 || static_cast<long long>(r.n) != gd || !counts_ok) {
        fail("campaign point not exact: " + c[0]);
      }
      if (d_ef > 1e-12 || d_er > 1e-9 || d_ed > 1e-9) { fail("campaign sums: " + c[0]); }
    }
    if (rows.size() != 6) { fail("ion_elastic_hist.csv row count"); }
  }

  std::printf("\n%s (%d failure%s)\n", g_fails == 0 ? "ALL OK" : "FAILED", g_fails,
              g_fails == 1 ? "" : "s");
  return g_fails == 0 ? 0 : 1;
}
