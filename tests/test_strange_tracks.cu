// P23a: the sixteen strange species QBBC transports, as TRACKS.
//
// What a stepping kernel needs of a species, held against what the constructed QBBC gives it.
// Eight sections, and each is a different kind of claim:
//
//  1. STRUCTURE. Which discrete processes a species carries - Decay, hadElastic,
//     <name>Inelastic, an at-rest capture - against `ref/oracle/species_processes.csv`, the
//     process managers of the running QBBC. A process the port draws where Geant4 has none, or
//     the reverse, moves every random number after it.
//  2. MODEL WINDOWS. `had::inelastic_models` against `strange_scope.csv`'s rows: the registered
//     models, in registration order, with the energy windows the constructed processes ask for.
//  3. CROSS SECTIONS. `had::elastic_xs_per_volume` and `inelastic_xs_per_volume` in B1's four
//     materials against `G4HadronicProcessStore`'s per-volume answers (strange_tracks_xs.csv),
//     1 MeV to 100 TeV - the two interaction lengths, through the wiring this package wrote.
//  4. DECAY LENGTH. `decay::in_flight_mean_free_path` against `G4Decay::GetMeanFreePath` on a
//     real G4Track, same file.
//  5. THE STREAM. The uniforms one step of each species consumes, counted: V226's first note
//     says a K0S or K0L inelastic interaction costs one more than a lambda's, at every energy.
//  6. CONSERVATION. Every species stepped on the host through the function its group kernel
//     calls, in water and bone, decays and elastic scatters live: charge and baryon number of
//     every decay's products against the parent, and its total energy.
//  7. EMISSION. A cascade hyperon arrives as a hyperon (V241's fix), and a charm or bottom
//     secondary is booked under its own name.
//  8. THE b/c PREDICATE against every b/c code QBBC knows (strange_scope.csv).
//
// Host-only. The kernels themselves are the engine's; `tests/test_inelastic_transport.cu` runs
// the queued interactions of the same stepper and balances them.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

#include "core/rng.cuh"
#include "data/materials.cuh"
#include "physics/em/hadron_range.cuh"
#include "physics/hadronic/interaction_apply.cuh"
#include "physics/stepper.cuh"

using namespace g4gpu;
using real_t = double;
namespace hp = g4gpu::physics::hadronic;

static int g_fails = 0;
static int g_compared = 0;
static void fail(const std::string& what) {
  std::printf("  FAIL: %s\n", what.c_str());
  ++g_fails;
}

static std::string oracle_dir() {
  const char* env = std::getenv("G4GPU_ORACLE");
  return (env != nullptr) ? std::string(env) : std::string("ref/oracle");
}

static std::vector<std::string> split(const std::string& s, char sep) {
  std::vector<std::string> out;
  std::string cur;
  for (char c : s) {
    if (c == sep) {
      out.push_back(cur);
      cur.clear();
    } else if (c != '\r' && c != '\n') {
      cur.push_back(c);
    }
  }
  out.push_back(cur);
  return out;
}

static std::vector<std::vector<std::string>> load_csv(const std::string& path) {
  std::vector<std::vector<std::string>> rows;
  FILE* f = std::fopen(path.c_str(), "r");
  if (f == nullptr) { return rows; }
  char line[4096];
  std::fgets(line, sizeof line, f);  // header
  while (std::fgets(line, sizeof line, f) != nullptr) { rows.push_back(split(line, ',')); }
  std::fclose(f);
  return rows;
}

static bool is_strange(ParticleType t) {
  return is_strange_charged_track(t) || is_strange_neutral_track(t);
}

static ParticleType type_of_name(const std::string& n) {
  for (int t = 0; t < static_cast<int>(ParticleType::kNumTypes); ++t) {
    if (n == particle_name(static_cast<ParticleType>(t))) { return static_cast<ParticleType>(t); }
  }
  return ParticleType::kNumTypes;
}

static int material_of(const std::string& n) {
  if (n == "G4_AIR") { return data::kAir; }
  if (n == "G4_WATER") { return data::kWater; }
  if (n == "G4_A-150_TISSUE") { return data::kA150Tissue; }
  if (n == "G4_BONE_COMPACT_ICRU") { return data::kBoneCompact; }
  return -1;
}

// A Philox that counts what it hands out.
struct CountingRng {
  Philox<real_t> r;
  long long n = 0;
  CountingRng(unsigned k, unsigned s, unsigned p) : r(k, s, p) {}
  __host__ __device__ real_t uniform() {
    ++n;
    return r.uniform();
  }
};

// An emitter that records what came out: charge, baryon number, total energy.
struct BalanceEmitter {
  Vec3<real_t> pos{0, 0, 0};
  int volume = 0;
  int event = 0;
  unsigned int parent_key = 0, parent_step = 0, child_count = 0;
  real_t parent_time = 0, parent_weight = 1, parent_velocity = 0;
  SecondaryArena arena{};
  int last_secondary = -1;
  const StepReport<real_t>* report = nullptr;
  had::HadronicRefusalBooks books{};
  int charge = 0, baryon = 0, n = 0;
  double etot = 0;
  ParticleType last = ParticleType::kNumTypes;
  __host__ __device__ int push(ParticleType t, const Vec3<real_t>&, real_t ekin, int,
                               unsigned short za = 0) {
    ++child_count;
    ++n;
    last = t;
    charge += static_cast<int>(std::lrint(particle_def<real_t>(t).charge));
    baryon += had::baryon_number_of(t, ion_a_of(za));
    etot += static_cast<double>(ekin) + (particle_def<real_t>(t).mass > 0
                                             ? static_cast<double>(particle_def<real_t>(t).mass)
                                             : 0.0);
    return 0;
  }
  __host__ __device__ int push_nucleus(int z, int a, const Vec3<real_t>&, real_t ekin, int) {
    ++child_count;
    ++n;
    charge += z;
    last = particle_type_of_nucleus(z, a);
    baryon += a;
    etot += static_cast<double>(ekin);
    return 0;
  }
};

int main() {
  const std::string dir = oracle_dir();
  static data::Material<real_t> mats[data::kNumMaterials];
  data::build_b1_materials<real_t>(mats);

  // ============================================================================ 1. structure
  std::printf("== 1. the discrete processes of each strange species, against QBBC's managers ==\n");
  {
    const auto rows = load_csv(dir + "/species_processes.csv");
    if (rows.empty()) {
      fail("cannot read species_processes.csv");
    }
    std::map<std::string, std::vector<std::string>> procs;
    for (const auto& c : rows) {
      if (c.size() >= 2) { procs[c[0]].push_back(c[1]); }
    }
    int n_species = 0;
    for (int t = 0; t < static_cast<int>(ParticleType::kNumTypes); ++t) {
      const ParticleType pt = static_cast<ParticleType>(t);
      if (!is_strange(pt)) { continue; }
      const std::string who = particle_name(pt);
      auto it = procs.find(who);
      if (it == procs.end()) {
        fail(who + " has no rows in species_processes.csv");
        continue;
      }
      ++n_species;
      bool decay = false, el = false, inel = false, bert_rest = false, ftf_rest = false;
      for (const std::string& p : it->second) {
        decay |= (p == "Decay");
        el |= (p == "hadElastic");
        inel |= (p == who + "Inelastic");
        bert_rest |= (p == "hBertiniCaptureAtRest");
        ftf_rest |= (p == "hFritiofCaptureAtRest");
      }
      auto check = [&](const char* what, bool ours, bool g4) {
        ++g_compared;
        if (ours != g4) {
          fail(who + ": " + what + " ours " + (ours ? "yes" : "no") + ", Geant4 "
               + (g4 ? "yes" : "no"));
        }
      };
      check("Decay", had::decays_in_flight(pt), decay);
      check("hadElastic", had::elastic_channel_has_xs(had::elastic_channel(pt)), el);
      check("Inelastic", had::inelastic_channel_has_xs(had::inelastic_channel(pt)), inel);
      check("an at-rest capture", had::has_at_rest_arm(pt), bert_rest || ftf_rest);
      std::printf("  %-12s Decay %d hadElastic %d Inelastic %d at-rest %s\n", who.c_str(),
                  decay ? 1 : 0, el ? 1 : 0, inel ? 1 : 0,
                  bert_rest ? "Bertini" : (ftf_rest ? "Fritiof" : "-"));
    }
    ++g_compared;
    if (n_species != 16) { fail("expected 16 strange species, found " + std::to_string(n_species)); }
  }

  // ======================================================================== 2. model windows
  std::printf("\n== 2. the inelastic models and their windows, against strange_scope.csv ==\n");
  {
    const auto rows = load_csv(dir + "/strange_scope.csv");
    if (rows.empty()) { fail("cannot read strange_scope.csv"); }
    // species -> the (model, emin, emax) rows of its <name>Inelastic process, in file order.
    struct M { std::string model; double lo, hi; };
    std::map<std::string, std::vector<M>> g4;
    for (const auto& c : rows) {
      if (c.size() < 18) { continue; }
      if (c[11] != c[0] + "Inelastic") { continue; }
      g4[c[0]].push_back({c[15], std::atof(c[16].c_str()), std::atof(c[17].c_str())});
    }
    int checked = 0;
    for (int t = 0; t < static_cast<int>(ParticleType::kNumTypes); ++t) {
      const ParticleType pt = static_cast<ParticleType>(t);
      if (!is_strange(pt)) { continue; }
      const std::string who = particle_name(pt);
      const had::InelasticModelList<real_t> L = had::inelastic_models<real_t>(pt);
      const auto& want = g4[who];
      ++g_compared;
      if (static_cast<int>(want.size()) != L.n) {
        fail(who + ": " + std::to_string(L.n) + " models registered, Geant4 has "
             + std::to_string(want.size()));
        continue;
      }
      for (int i = 0; i < L.n; ++i) {
        const std::string name = (L.model[i] == had::InelasticModel::kFtfp) ? "FTFP"
                                 : (L.model[i] == had::InelasticModel::kBertini)
                                     ? "BertiniCascade"
                                     : "?";
        ++g_compared;
        if (name != want[i].model || L.range[i].min_energy != want[i].lo
            || L.range[i].max_energy != want[i].hi) {
          char b[256];
          std::snprintf(b, sizeof b, "%s model %d: ours %s [%g, %g], Geant4 %s [%g, %g]",
                        who.c_str(), i, name.c_str(), L.range[i].min_energy,
                        L.range[i].max_energy, want[i].model.c_str(), want[i].lo, want[i].hi);
          fail(b);
        }
      }
      ++checked;
    }
    std::printf("  %d species' model lists equal Geant4's, model for model and window for "
                "window\n", checked);
  }

  // ============================================================== 3, 4. cross sections, decay
  std::printf("\n== 3. hadElastic and Inelastic per volume, 4. G4Decay's mean free path ==\n");
  {
    const auto rows = load_csv(dir + "/strange_tracks_xs.csv");
    if (rows.empty()) { fail("cannot read strange_tracks_xs.csv - run the strange_tracks dump"); }
    had::ElasticTables<real_t> el{};
    had::InelasticTables<real_t> in{};
    double worst_el = 0, worst_in = 0, worst_mfp = 0;
    std::string w_el, w_in, w_mfp;
    long long n_el = 0, n_in = 0, n_mfp = 0;
    for (const auto& c : rows) {
      if (c.size() < 7) { continue; }
      const int mi = material_of(c[0]);
      if (mi < 0) { continue; }
      const ParticleType pt = type_of_name(c[1]);
      if (!is_strange(pt)) {
        fail("unknown species in strange_tracks_xs.csv: " + c[1]);
        continue;
      }
      const double e = std::atof(c[3].c_str());
      const double g_el = std::atof(c[4].c_str());
      const double g_in = std::atof(c[5].c_str());
      const double g_mfp = std::atof(c[6].c_str());
      hadronic::xs::MaterialXs<real_t> mxs{};
      const double o_el = had::elastic_xs_per_volume<real_t>(el, mats[mi], pt, e, mxs);
      const double o_in = had::inelastic_xs_per_volume<real_t>(in, mats[mi], pt, e, 0, 0, mxs);
      auto rel = [](double a, double b) {
        if (a == b) { return 0.0; }
        return std::fabs(a - b) / std::fmax(std::fabs(b), 1e-300);
      };
      char where[200];
      std::snprintf(where, sizeof where, "%s in %s at %g MeV", c[1].c_str(), c[0].c_str(), e);
      const double d_el = rel(o_el, g_el), d_in = rel(o_in, g_in);
      ++n_el;
      ++n_in;
      if (d_el > worst_el) { worst_el = d_el; w_el = std::string(where) + " ours "
                                                   + std::to_string(o_el) + " G4 "
                                                   + std::to_string(g_el); }
      if (d_in > worst_in) { worst_in = d_in; w_in = std::string(where) + " ours "
                                                   + std::to_string(o_in) + " G4 "
                                                   + std::to_string(g_in); }
      if (g_mfp > 0 && g_mfp < 1e300) {
        const double o_mfp = decay::in_flight_mean_free_path<real_t>(
            pdg_code(pt), particle_def<real_t>(pt).mass, e);
        const double d = rel(o_mfp, g_mfp);
        ++n_mfp;
        if (d > worst_mfp) { worst_mfp = d; w_mfp = where; }
      }
    }
    std::printf("  hadElastic  %lld points, worst %.3e  [%s]\n", n_el, worst_el, w_el.c_str());
    std::printf("  Inelastic   %lld points, worst %.3e  [%s]\n", n_in, worst_in, w_in.c_str());
    std::printf("  decay mfp   %lld points, worst %.3e  [%s]\n", n_mfp, worst_mfp, w_mfp.c_str());
    // The tolerance is the material's: the cross sections per atom are P23b's and exact
    // (V226-V228), and what a per-volume number adds is the atom densities of the four
    // materials, which data/materials.cuh builds from the same NIST compositions.
    const double kTol = 1e-9;
    g_compared += 3;
    if (n_el == 0 || worst_el > kTol) { fail("hadElastic per volume: worst " + std::to_string(worst_el)); }
    if (n_in == 0 || worst_in > kTol) { fail("Inelastic per volume: worst " + std::to_string(worst_in)); }
    if (n_mfp == 0 || worst_mfp > 1e-12) { fail("decay mfp: worst " + std::to_string(worst_mfp)); }
  }

  // ================================================================== 5. the stream, counted
  std::printf("\n== 5. the uniforms an inelastic step consumes: K0S/K0L against lambda ==\n");
  {
    // One material and one energy where the inelastic length always wins: no boundary within
    // reach (a 1 km box), no decay drawn (`decay` off), no elastic (`hadron_elastic` off). What
    // is left is the inelastic draw, then - for a K0 - Bertini's IsApplicable, then the model
    // choice, which draws only in the FTFP/Bertini overlap. At 4.5 GeV both models match, the
    // windows are not nested, and the choice draws once.
    static em::HadronRangeTable<real_t> hrt;
    {
      std::vector<real_t> cuts(data::kNumMaterials);
      for (int i = 0; i < data::kNumMaterials; ++i) { cuts[i] = mats[i].cut_electron; }
      em::build_hadron_range_table<real_t>(mats, cuts.data(), hrt, nullptr, data::kNumMaterials);
    }
    geom::Volume<real_t> vols[1] = {{{geom::SolidType::kBox, {1e6, 1e6, 1e6}},
                                     geom::make_translation<real_t>({0, 0, 0}), 0, data::kWater,
                                     0}};
    Scene<real_t> scene{};
    scene.geometry = geom::Geometry<real_t>{vols, 1, 0};
    scene.materials = mats;
    scene.hadron_range = &hrt;
    scene.range_cut = real_t(0.7);
    static std::vector<had::PendingInteraction<real_t>> qbuf(4);
    static int qcursor = 0;
    had::HadronicWiring<real_t> had{};
    had.stage = had::HadronicStage::kFinal;
    had.decay = false;
    had.hadron_elastic = false;
    had.queue.items = qbuf.data();
    had.queue.cursor = &qcursor;
    had.queue.capacity = 4;
    const ParticleType who[] = {ParticleType::kLambda, ParticleType::kKaonZeroLong,
                                ParticleType::kKaonZeroShort, ParticleType::kAntiLambda};
    for (double e : {1000.0, 4500.0, 50000.0}) {
      long long used[4] = {0, 0, 0, 0};
      for (int k = 0; k < 4; ++k) {
        TrackState<real_t> p{};
        p.species = who[k];
        p.pos = Vec3<real_t>{0, 0, 0};
        p.dir = Vec3<real_t>{0, 0, 1};
        p.ekin = e;
        p.volume = 0;
        p.rng_key = 99u;
        CountingRng rng(p.rng_key, 0u, 0x4E7Au);
        BalanceEmitter em;
        StepReport<real_t> rep;
        real_t edep = 0;
        bool queued = false;
        qcursor = 0;
        step_neutral_strange(scene, p, who[k], had, rng, em, edep, rep, vis::no_capture(),
                             &queued);
        ++g_compared;
        if (!queued) { fail(std::string(particle_name(who[k])) + " did not queue"); }
        used[k] = rng.n;
      }
      // The expectation, built from the predicates the stepper does NOT use for the count: one
      // inelastic draw, Bertini's IsApplicable for a K0, the choice's own draw.
      for (int k = 0; k < 4; ++k) {
        long long want = 1 + (had::range_manager_draws_is_applicable(who[k]) ? 1 : 0)
                         + (had::inelastic_choice_draws<real_t>(who[k], e, 1) ? 1 : 0);
        ++g_compared;
        if (used[k] != want) {
          fail(std::string(particle_name(who[k])) + " at " + std::to_string(e) + " MeV used "
               + std::to_string(used[k]) + " uniforms, expected " + std::to_string(want));
        }
      }
      std::printf("  %8.0f MeV  lambda %lld  kaon0L %lld  kaon0S %lld  anti_lambda %lld\n", e,
                  used[0], used[1], used[2], used[3]);
      ++g_compared;
      if (used[1] != used[0] + 1 || used[2] != used[0] + 1) {
        fail("a K0 inelastic step does not cost exactly one uniform more than a lambda's");
      }
    }
  }

  // ======================================================================= 6. conservation
  std::printf("\n== 6. every strange species stepped: charge, baryon number and energy of "
              "every decay ==\n");
  {
    static em::HadronRangeTable<real_t> hrt;
    static em::ShellTables<real_t> shell;
    em::build_shell_tables(shell);
    {
      std::vector<real_t> cuts(data::kNumMaterials);
      for (int i = 0; i < data::kNumMaterials; ++i) { cuts[i] = mats[i].cut_electron; }
      em::build_hadron_range_table<real_t>(mats, cuts.data(), hrt, &shell, data::kNumMaterials);
    }
    static std::vector<had::PendingInteraction<real_t>> qbuf(4);
    static int qcursor = 0;
    std::vector<int> ref_n(static_cast<int>(had::HadronicRefusal::kNumHadronicRefusals), 0);
    std::vector<double> ref_e(ref_n.size(), 0.0);
    long long decays = 0, rest_decays = 0, elastic = 0, queued_n = 0, bad = 0, steps = 0;
    long long an_seen = 0;
    double worst_e = 0;
    for (int mi : {data::kWater, data::kBoneCompact}) {
      geom::Volume<real_t> vols[1] = {{{geom::SolidType::kBox, {500, 500, 500}},
                                       geom::make_translation<real_t>({0, 0, 0}), 0, mi, 0}};
      Scene<real_t> scene{};
      scene.geometry = geom::Geometry<real_t>{vols, 1, 0};
      scene.materials = mats;
      scene.hadron_range = &hrt;
      scene.range_cut = real_t(0.7);
      had::HadronicWiring<real_t> had{};
      had.stage = had::HadronicStage::kFinal;
      had.queue.items = qbuf.data();
      had.queue.cursor = &qcursor;
      had.queue.capacity = 4;
      had.books.count = ref_n.data();
      had.books.energy = ref_e.data();
      for (int t = 0; t < static_cast<int>(ParticleType::kNumTypes); ++t) {
        const ParticleType pt = static_cast<ParticleType>(t);
        if (!is_strange(pt)) { continue; }
        const ParticleDef<real_t> pd = particle_def<real_t>(pt);
        for (int k = 0; k < 300; ++k) {
          TrackState<real_t> p{};
          p.species = pt;
          p.pos = Vec3<real_t>{0, 0, -490};
          p.dir = Vec3<real_t>{0, 0, 1};
          // A spread of energies including zero: a neutral strange species born at rest
          // decays or is annihilated at rest, which is a branch of its own.
          p.ekin = (k % 10 == 0) ? real_t(0) : real_t(20.0 * (1 + k % 50));
          p.volume = 0;
          p.rng_key = 31337u + static_cast<unsigned int>(k) * 2654435761u + 7u * t;
          p.begin(p.pos, p.dir, p.ekin, 0, 0u, ProcessId::fNotDefined, real_t(0), real_t(1));
          bool alive = true;
          for (int st = 0; alive && st < 20000; ++st) {
            Philox<real_t> rng(p.rng_key, p.step, 0xB19Du);
            BalanceEmitter em;
            StepReport<real_t> rep;
            real_t edep = 0;
            bool queued = false;
            qcursor = 0;
            const double e_parent = static_cast<double>(p.ekin) + pd.mass;
            const int q_parent = static_cast<int>(std::lrint(pd.charge));
            const int b_parent = had::baryon_number_of(pt, 0);
            const int kAn = static_cast<int>(had::HadronicRefusal::kAntiNeutronSecondary);
            const int an_n0 = ref_n[kAn];
            const double an_e0 = ref_e[kAn];
            alive = is_strange_charged_track(pt)
                        ? step_hadron(scene, p, pt, had, rng, em, edep, rep, vis::no_capture(),
                                      &queued, ParticleType::kSigmaPlus)
                        : step_neutral_strange(scene, p, pt, had, rng, em, edep, rep,
                                               vis::no_capture(), &queued);
            ++steps;
            ++p.step;
            if (queued) {
              ++queued_n;
              break;
            }
            if (rep.process == ProcessId::fDecay) {
              // In flight the products carry the parent's total energy at the decay point,
              // which for a charged parent is AFTER the step's continuous loss; at rest, its
              // mass. The charged parent's post-step energy is `p.ekin` at return.
              const bool at_rest = (rep.status == StepStatus::fStopAndKill);
              const double e_in = is_strange_charged_track(pt)
                                      ? (at_rest ? pd.mass : static_cast<double>(p.ekin) + pd.mass)
                                      : e_parent;
              if (at_rest) { ++rest_decays; } else { ++decays; }
              // An anti-neutron daughter is BOOKED, by name, with its kinetic energy - the
              // one product the ledger carries instead of a track (P23c's species). Its baryon
              // number and rest mass go back into the sum here, exactly.
              const int an = ref_n[kAn] - an_n0;
              an_seen += an;
              em.baryon -= an;
              em.etot += (ref_e[kAn] - an_e0) + an * units::neutron_mass_c2<double>();
              const bool ok_q = (em.charge == q_parent), ok_b = (em.baryon == b_parent);
              const double de = std::fabs(em.etot - e_in) / e_in;
              if (de > worst_e) { worst_e = de; }
              // A neutrino's energy is booked by `push` here like any other product, so the
              // sum is complete. 1e-9: the decay kinematics are double precision throughout.
              if (!ok_q || !ok_b || de > 1e-9) {
                ++bad;
                if (bad <= 5) {
                  std::printf("  bad decay of %s: charge %d/%d baryon %d/%d energy %.9g/%.9g\n",
                              particle_name(pt), em.charge, q_parent, em.baryon, b_parent,
                              em.etot, e_in);
                }
              }
            }
            if (rep.process == ProcessId::fHadronElastic) { ++elastic; }
          }
        }
      }
    }
    std::printf("  %lld steps: %lld decays in flight, %lld at rest, %lld elastic scatters, %lld "
                "interactions queued; %lld decays out of balance, worst energy %.2e\n",
                steps, decays, rest_decays, elastic, queued_n, bad, worst_e);
    std::printf("  %lld anti-neutron daughters booked by name (kAntiNeutronSecondary) and "
                "balanced back in\n", an_seen);
    g_compared += 4;
    if (an_seen == 0) { fail("no anti-neutron daughter was met - the named row is untested"); }
    if (bad > 0) { fail(std::to_string(bad) + " decays do not conserve charge, baryon number or energy"); }
    if (decays == 0 || rest_decays == 0 || elastic == 0 || queued_n == 0) {
      fail("a branch was never reached - the conservation check is vacuous for it");
    }
    if (ref_n[static_cast<int>(had::HadronicRefusal::kDecayChannel)] > 0) {
      fail("a strange decay booked kDecayChannel - a daughter with no row");
    }
  }

  // ============================================================================ 7. emission
  std::printf("\n== 7. what a model hands back: a cascade hyperon, and a b/c hadron ==\n");
  {
    std::vector<int> ref_n(static_cast<int>(had::HadronicRefusal::kNumHadronicRefusals), 0);
    std::vector<double> ref_e(ref_n.size(), 0.0);
    had::HadronicRefusalBooks books{ref_n.data(), ref_e.data()};
    // Bertini's own fields for a lambda, a sigma-, a proton and a D+ (411), as
    // `G4CascadeInterface` fills them: z = charge, a = baryon number - which is what used to
    // send the lambda down the nucleus path as a neutron.
    struct S { int pdg, z, a; ParticleType want; };
    const S cases[] = {{3122, 0, 1, ParticleType::kLambda},
                       {3112, -1, 1, ParticleType::kSigmaMinus},
                       {3322, 0, 1, ParticleType::kXiZero},
                       {3334, -1, 1, ParticleType::kOmegaMinus},
                       {2212, 1, 1, ParticleType::kProton},
                       {2112, 0, 1, ParticleType::kNeutron}};
    for (const S& c : cases) {
      hp::HadronicStepResult<real_t, had::kInteractionSecondaryCap> r{};
      r.n_secondaries = 1;
      r.secondaries[0].pdg = c.pdg;
      r.secondaries[0].z = c.z;
      r.secondaries[0].a = c.a;
      r.secondaries[0].kin_energy = 100;
      BalanceEmitter em;
      had::emit_interaction_result<real_t>(r, em, books);
      ++g_compared;
      if (em.last != c.want) {
        fail(std::string("a cascade secondary of pdg ") + std::to_string(c.pdg) + " became "
             + particle_name(em.last) + ", not " + particle_name(c.want));
      }
    }
    {
      hp::HadronicStepResult<real_t, had::kInteractionSecondaryCap> r{};
      r.n_secondaries = 1;
      r.secondaries[0].pdg = 411;
      r.secondaries[0].kin_energy = 250;
      BalanceEmitter em;
      had::emit_interaction_result<real_t>(r, em, books);
      ++g_compared;
      if (em.n != 0 || ref_n[static_cast<int>(had::HadronicRefusal::kBcHadronSecondary)] != 1) {
        fail("a D+ was not booked under kBcHadronSecondary");
      }
    }
    std::printf("  six cascade baryons emitted as themselves; a D+ booked by name\n");
  }

  // ======================================================================= 8. b/c predicate
  std::printf("\n== 8. is_bc_hadron_pdg against every species strange_scope.csv lists ==\n");
  {
    const auto rows = load_csv(dir + "/strange_scope.csv");
    std::map<int, bool> seen;
    for (const auto& c : rows) {
      if (c.size() > 1) { seen[std::atoi(c[1].c_str())] = true; }
    }
    int n_bc = 0, n_other = 0;
    for (const auto& kv : seen) {
      const int pdg = kv.first;
      const int a = std::abs(pdg);
      // The truth, independently: the scope file's charm/bottom block is every code from 400 up
      // to 600 (mesons) and from 4000 up (baryons); the kaons and hyperons are below both.
      const bool want = (a >= 400 && a < 600) || a >= 4000;
      ++g_compared;
      if (is_bc_hadron_pdg(pdg) != want) {
        fail("is_bc_hadron_pdg(" + std::to_string(pdg) + ") is wrong");
      }
      if (want) { ++n_bc; } else { ++n_other; }
    }
    for (int t = 0; t < static_cast<int>(ParticleType::kNumTypes); ++t) {
      ++g_compared;
      if (is_bc_hadron_pdg(pdg_code(static_cast<ParticleType>(t)))) {
        fail(std::string("is_bc_hadron_pdg says ") + particle_name(static_cast<ParticleType>(t))
             + " is a b/c hadron");
      }
    }
    std::printf("  %d b/c codes and %d strange codes from the scope file, and all %d "
                "ParticleTypes\n", n_bc, n_other, static_cast<int>(ParticleType::kNumTypes));
    ++g_compared;
    if (n_bc < 39) { fail("fewer than 39 b/c codes in strange_scope.csv: the predicate was not exercised"); }
  }

  std::printf("\n%d comparisons, %d failures\n", g_compared, g_fails);
  std::printf("%s\n", g_fails == 0 ? "OK" : "FAILURES");
  return (g_fails == 0) ? 0 : 1;
}
