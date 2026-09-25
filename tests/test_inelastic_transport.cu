// P15: the inelastic interaction and the at-rest capture, end to end through the wiring.
//
// Five sections, and they check five different kinds of claim.
//
//  1. **The slot pool costs what the headers say.** Every byte count P15's commit messages and
//     docs/RISK.md V189 quote is `sizeof`, printed here from the types themselves and pinned by
//     `static_assert`. docs/RISK.md V167 is what a sizing table in a comment does when it is
//     only a comment: it drifted 24% below `sizeof` and a caller budgeted from it.
//
//  2. **The two tables P15 had to COPY agree with their originals.** `physics/stepper.cuh`
//     cannot include `stopping/stopping_process.cuh` - that header pulls in Bertini and FTFP
//     and would put both inside every stepping kernel (docs/RISK.md V188) - so
//     `had::at_rest_bucket` is a second copy of `stopping::stopping_arm`. A copy bounded by a
//     comment is a copy that drifts; this is the assertion that bounds it, over every
//     `ParticleType` in the enum rather than over the four the port happens to transport.
//
//  3. **The interaction happens at the rate the cross section says.** Not "an interaction
//     happened", which is what a wiring test usually settles for: the stepper draws
//     `-log(u)*lambda` per step at the PRE-step energy, so the number of interactions along a
//     realised path is Poisson with mean `sum(L_i / lambda_i)` over the steps actually taken,
//     whatever the energy loss did in between. That sum is accumulated here from the same
//     `inelastic_xs_per_volume` the stepper called, and the count is compared against it with a
//     stated band. A port that drew its length from the wrong energy, or from the wrong
//     material, or forgot the `1/xs`, fails this and passes "an interaction happened".
//
//  4. **Energy, baryon number and charge balance across the interaction**, per event, within
//     the refusal ledger - which is the qualifier that makes it checkable at all, because a
//     secondary this transport cannot step (a hyperon, a K0) leaves with its baryon number and
//     is booked rather than emitted.
//
//  5. **The at-rest hook**: a stopped pi- and mu- go to `stopping::at_rest` and not to
//     `G4Decay`, which is `AtRestGetPhysicalInteractionLength` returning 0.0 - a pre-emption
//     and not a race.
//
// WHY THE HEAVY HALF IS ON THE HOST. `step_hadron`, `had::enqueue_interaction`,
// `had::run_inelastic` and `had::run_at_rest` are all `__host__ __device__` and this file runs
// the same source the kernels run, which is the arrangement `tests/test_step_hadron.cu` and
// `tests/test_neutron_general.cu` already use. What it does NOT do is instantiate a model in a
// `__global__`: one interaction kernel costs ptxas 6 to 20 GB and 40 to 660 seconds
// (docs/RISK.md V189), so a test that carried one would cost more to build than the engine
// does. The DEVICE half below is the machinery P15 adds that is not a model - the queue and its
// bucket sort - run as a kernel and compared against the host. The models' own device path is
// exercised by the engine build and by every beam of `tools/b1_sweep.ps1`.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "core/rng.cuh"
#include "data/materials.cuh"
#include "host/hadronic_upload.cuh"
#include "physics/em/hadron_range.cuh"
#include "physics/hadronic/interaction_apply.cuh"
#include "physics/stepper.cuh"

using namespace g4gpu;
using real_t = double;
#ifndef G4GPU_PENDING_BYTES
#define G4GPU_PENDING_BYTES 392
#endif
namespace hp = g4gpu::physics::hadronic;
namespace stop = g4gpu::physics::hadronic::stopping;
namespace ftfe = g4gpu::hadronic::ftf::entry;

static int g_fails = 0;
static int g_max_secondaries = 0;
static void fail(const std::string& what) {
  std::printf("  FAIL: %s\n", what.c_str());
  ++g_fails;
}

// =============================================================================================
// An emitter that counts rather than transports.
// =============================================================================================
struct CountingEmitter {
  Vec3<real_t> pos{0, 0, 0};
  int volume = 0;
  int event = 0;
  unsigned int parent_key = 0, parent_step = 0, child_count = 0;
  real_t parent_time = 0, parent_weight = 1, parent_velocity = 0;
  SecondaryArena arena{};
  int last_secondary = -1;
  const StepReport<real_t>* report = nullptr;
  had::HadronicRefusalBooks books{};

  // What came out, for the balance.
  double secondary_energy = 0;
  int n_secondaries = 0;
  int baryon = 0;
  int charge = 0;

  __host__ __device__ int push(ParticleType t, const Vec3<real_t>& /*dir*/, real_t ekin,
                               int /*event_id*/, unsigned short za = 0) {
    ++child_count;
    ++n_secondaries;
    secondary_energy += static_cast<double>(ekin);
    baryon += had::baryon_number_of(t, ion_a_of(za));
    // THE BALANCE IS OF NUCLEAR CHARGE, AND AN ELECTRON IN A HADRONIC FINAL STATE IS ATOMIC.
    // The only electron any of these models emits is P3's internal conversion:
    // `G4GammaTransition` hands the transition energy to a SHELL electron instead of a photon,
    // which leaves the atom ionised and the nucleus's Z untouched. Counting its -1 against
    // projectile + target Z is comparing two different charges. Found when P9e's corrected
    // random stream (V180/V181) reshuffled the events and the grid sampled its first conversion
    // electrons - a 1 GeV proton and a 100 MeV neutron on Ca40, each leaving Ar37 and a 15 keV
    // e-, each one short by exactly that electron. `conversion_electrons` keeps the count, so a
    // stray electron from anywhere else is still visible rather than silently excused.
    if (t == ParticleType::kElectron) {
      ++conversion_electrons;
    } else {
      charge += static_cast<int>(std::lrint(particle_def<real_t>(t).charge));
    }
    return 0;
  }
  int conversion_electrons = 0;
  __host__ __device__ int push_nucleus(int z, int a, const Vec3<real_t>& dir, real_t ekin,
                                       int event_id) {
    ++child_count;
    ++n_secondaries;
    secondary_energy += static_cast<double>(ekin);
    baryon += a;
    charge += z;
    (void)dir; (void)event_id;
    return 0;
  }
};

// =============================================================================================
// The pool, on the host.
// =============================================================================================
namespace {

struct HostPool {
  std::vector<had::InteractionSlot<real_t>> slots;
  std::vector<ftfe::Workspace> ftf_slots;
  g4gpu::hadronic::ftf::LundTables<double> lund{};
  std::vector<bic::imr::ConcreteChannel> channels;
  deex::FermiPoolStorage fermi_storage;
  had::InteractionPool<real_t> view{};

  void build(int n, const data::LevelTable& lt) {
    slots.resize(static_cast<std::size_t>(n));
    ftf_slots.resize(static_cast<std::size_t>(n));
    channels.resize(bic::imr::kConcreteChannelCount);
    const int n_ch = bic::imr::build_concrete_channels(channels.data(),
                                                       bic::imr::kConcreteChannelCount);
    deex::build_fermi_pool(fermi_storage, lt);
    view.slots = slots.data();
    view.n_slots = n;
    view.bic_channels = channels.data();
    view.n_bic_channels = n_ch;
    view.ftf = ftfe::host_handle(ftf_slots.data(), n, &lund);
    view.fermi = fermi_storage.view();
  }
};

}  // namespace

// =============================================================================================
// 1. What a slot costs
// =============================================================================================
static void section_sizes() {
  std::printf("== 1. the pool, by sizeof ==\n");
  const std::size_t slot = sizeof(had::InteractionSlot<real_t>);
  const std::size_t ftf = ftfe::kWorkspaceBytes;
  const std::size_t per = slot + ftf;
  std::printf("  InteractionSlot<double>     %10zu B\n", slot);
  std::printf("  ftf::entry::Workspace       %10zu B\n", ftf);
  std::printf("  one slot, both pools        %10zu B  (%.2f kB)\n", per, double(per) / 1024.0);
  std::printf("  PendingInteraction<double>  %10zu B\n", sizeof(had::PendingInteraction<real_t>));
  for (int n : {64, 128, 256, 512, 1024}) {
    std::printf("    %5d slots: %8.1f MB\n", n, double(n) * double(per) / 1048576.0);
  }
  // PINNED, for the reason docs/RISK.md V167 gives: the numbers above appear in P15's commit
  // messages, in docs/RISK.md V189 and in `interaction_apply.cuh`'s own header, and a table
  // that only a comment carries drifts. A change to any array in `InteractionSlot` fails the
  // build here with the new figure in the message. Update the assertion and the prose together,
  // or not at all.
  static_assert(sizeof(had::InteractionSlot<double>) == 1481056,
                "had::InteractionSlot changed size. interaction_apply.cuh's header table, "
                "docs/RISK.md V189 and this assertion all carry the number - update all three.");
  static_assert(sizeof(had::PendingInteraction<double>) == G4GPU_PENDING_BYTES,
                "had::PendingInteraction changed size. The queue's byte cost is quoted in "
                "interaction_queue.cuh and in the engine's own printout.");
  std::printf("\n");
}

// =============================================================================================
// 2. The two copies, against their originals
// =============================================================================================
static void section_copies() {
  std::printf("== 2. the tables P15 had to copy, against the originals ==\n");
  int n_checked = 0, n_arms = 0;
  for (int i = 0; i < static_cast<int>(ParticleType::kNumTypes); ++i) {
    const ParticleType t = static_cast<ParticleType>(i);
    const int pdg = pdg_code(t);
    // `stopping::stopping_arm` is the original; `had::at_rest_bucket` is the copy the stepper
    // can reach. They must agree about WHETHER there is an arm for every species in the enum.
    const stop::NuclearArm arm = stop::stopping_arm(pdg);
    const had::InteractionBucket b = had::at_rest_bucket(t);
    const bool orig_has = (arm != stop::NuclearArm::kNone);
    const bool copy_has = (b != had::InteractionBucket::kNone);
    if (orig_has != copy_has) {
      fail("at_rest_bucket disagrees with stopping::stopping_arm for "
           + std::string(particle_name(t)) + " (pdg " + std::to_string(pdg) + "): original "
           + (orig_has ? "has" : "has no") + " arm, copy " + (copy_has ? "has" : "has no"));
    }
    if (copy_has != had::has_at_rest_arm(t)) {
      fail("has_at_rest_arm disagrees with at_rest_bucket for "
           + std::string(particle_name(t)));
    }
    if (orig_has) { ++n_arms; }
    ++n_checked;
  }
  std::printf("  at_rest_bucket vs stopping::stopping_arm: %d species, %d with an at-rest "
              "nuclear arm, 0 disagreements\n", n_checked, n_arms);

  // `baryon_number_of`, against the values `G4ParticleDefinition::GetBaryonNumber()` holds.
  struct BRow { ParticleType t; int a; int want; };
  const BRow rows[] = {
      {ParticleType::kProton, 0, 1},      {ParticleType::kNeutron, 0, 1},
      {ParticleType::kAntiProton, 0, -1}, {ParticleType::kPionPlus, 0, 0},
      {ParticleType::kPionMinus, 0, 0},   {ParticleType::kKaonPlus, 0, 0},
      {ParticleType::kMuonMinus, 0, 0},   {ParticleType::kElectron, 0, 0},
      {ParticleType::kGamma, 0, 0},       {ParticleType::kDeuteron, 0, 2},
      {ParticleType::kTriton, 0, 3},      {ParticleType::kHe3, 0, 3},
      {ParticleType::kAlpha, 0, 4},       {ParticleType::kGenericIon, 12, 12},
      {ParticleType::kGenericIon, 208, 208},
  };
  int n_b = 0;
  for (const BRow& r : rows) {
    const int got = had::baryon_number_of(r.t, r.a);
    if (got != r.want) {
      fail("baryon_number_of(" + std::string(particle_name(r.t)) + ", A=" + std::to_string(r.a)
           + ") = " + std::to_string(got) + ", want " + std::to_string(r.want));
    }
    ++n_b;
  }
  std::printf("  baryon_number_of: %d rows exact, including a GenericIon's own A (12 and 208, "
              "which is what the per-nucleon model choice divides by)\n", n_b);
  std::printf("\n");
}

// =============================================================================================
// 3 and 4. The rate, and the balance
// =============================================================================================
namespace {

/// Everything one (species, energy, material) cell produced.
struct Cell {
  double expected = 0;     ///< sum(L_i / lambda_i) over the realised path - the Poisson mean
  long long queued = 0;    ///< interaction lengths that WON the competition
  long long ran = 0;       ///< and produced a final state
  long long rejected = 0;  ///< G4HadronicProcess's integral-approach rejection
  long long refused[static_cast<int>(had::HadronicRefusal::kNumHadronicRefusals)] = {};
  long long bad_baryon = 0, bad_charge = 0;
  double worst_energy = 0;
  long long n_balance = 0;
  long long secondaries = 0;
  /// The most secondaries any ONE interaction of this cell emitted into the pool.
  ///
  /// THIS IS THE NUMBER `max_secondaries_per_step` IS SET FROM, and it is measured here rather
  /// than guessed because the engine's output-pool budget is `(capacity - live) / reservation`:
  /// too low and a launch overruns the pool and the batch halves, too high and a hadron run
  /// steps a fraction of its tracks per iteration. `core/track_buffer.cuh`'s own comment says
  /// the hadronic rows are the wiring package's to raise and does the arithmetic for what
  /// raising the SHARED constant would cost a gamma; this is the other half of it.
  int max_secondaries = 0;
};

}  // namespace

int main() {
  std::printf("== P15: the inelastic interaction and the at-rest capture, end to end ==\n\n");
  section_sizes();
  section_copies();

  // ---- the data every arm's de-excitation tail needs.
  const std::string pe = host::g4photon_evaporation_dir();
  if (pe.empty()) {
    std::printf("  (sections 3-5 skipped: no PhotonEvaporation dataset)\n");
    std::printf("%s\n", g_fails == 0 ? "OK" : "FAILURES");
    return (g_fails == 0) ? 0 : 1;
  }
  static data::LevelTableStorage lts;
  data::read_all_level_data(
      lts, pe, data::kLevelZMax, [](int Z, int A) { return deex::shell_correction(A, Z); },
      [](int Z, int A) { return deex::level_manager_level_density(Z, A); });
  const data::LevelTable lt = lts.view();

  static HostPool pool;
  pool.build(1, lt);

  // ---- the inelastic cross sections, on the host.
  const std::string dir = host::g4particlexs_subdir("proton");
  if (dir.empty()) {
    std::printf("  (sections 3-5 skipped: G4PARTICLEXSDATA could not be resolved)\n");
    std::printf("%s\n", g_fails == 0 ? "OK" : "FAILURES");
    return (g_fails == 0) ? 0 : 1;
  }
  static data::ParticleXsTable<real_t> t_p, t_d, t_t, t_h3, t_a, t_n;
  static hadronic::xs::PxsDataSet<real_t> ds_p, ds_d, ds_t, ds_h3, ds_a, ds_n;
  namespace hxs = g4gpu::hadronic::xs;
  const bool got =
      hxs::pxs_load<real_t>(hxs::PxsKind::kParticleInelastic, hxs::proton<real_t>(),
                            host::g4particlexs_subdir("proton"), t_p, ds_p)
      && hxs::pxs_load<real_t>(hxs::PxsKind::kParticleInelastic, hxs::deuteron<real_t>(),
                               host::g4particlexs_subdir("deuteron"), t_d, ds_d)
      && hxs::pxs_load<real_t>(hxs::PxsKind::kParticleInelastic, hxs::triton<real_t>(),
                               host::g4particlexs_subdir("triton"), t_t, ds_t)
      && hxs::pxs_load<real_t>(hxs::PxsKind::kParticleInelastic, hxs::he3<real_t>(),
                               host::g4particlexs_subdir("He3"), t_h3, ds_h3)
      && hxs::pxs_load<real_t>(hxs::PxsKind::kParticleInelastic, hxs::alpha<real_t>(),
                               host::g4particlexs_subdir("alpha"), t_a, ds_a)
      && hxs::pxs_load<real_t>(hxs::PxsKind::kNeutronInelastic, hxs::neutron<real_t>(),
                               host::g4particlexs_subdir("neutron"), t_n, ds_n);
  if (!got) {
    fail("G4PARTICLEXS4.0 is incomplete - an element file is missing");
    std::printf("FAILURES\n");
    return 1;
  }
  static hxs::BggPionTable<real_t> bgg_pion;
  hxs::bgg_build_pion_table<real_t>(/*is_elastic=*/false, bgg_pion);

  had::InelasticTables<real_t> xs{};
  xs.particle[0] = &ds_p;
  xs.particle[1] = &ds_d;
  xs.particle[2] = &ds_t;
  xs.particle[3] = &ds_h3;
  xs.particle[4] = &ds_a;
  xs.neutron = &ds_n;
  xs.bgg_pion = &bgg_pion;

  // ---- the scene: one slab of one material, 400 mm of half-width so nothing escapes.
  static data::Material<real_t> mats[data::kNumMaterials];
  data::build_b1_materials<real_t>(mats);
  static auto* hrt = new em::HadronRangeTable<real_t>();
  {
    std::vector<real_t> cuts(data::kNumMaterials);
    for (int i = 0; i < data::kNumMaterials; ++i) { cuts[i] = mats[i].cut_electron; }
    static em::ShellTables<real_t> shell;
    em::build_shell_tables(shell);
    em::build_hadron_range_table<real_t>(mats, cuts.data(), *hrt, &shell, data::kNumMaterials);
  }

  // The queue, one entry deep per step - the track is followed one step at a time and the
  // queue drained after each, which is what the engine does per launch.
  static std::vector<had::PendingInteraction<real_t>> qbuf(64);
  static int qcursor = 0;

  std::vector<int> ref_n(static_cast<int>(had::HadronicRefusal::kNumHadronicRefusals), 0);
  std::vector<double> ref_e(ref_n.size(), 0.0);

  struct Beam { ParticleType t; real_t e; const char* name; };
  const Beam beams[] = {
      {ParticleType::kProton, 210, "proton 210 MeV"},
      {ParticleType::kProton, 1000, "proton 1 GeV"},
      {ParticleType::kProton, 4000, "proton 4 GeV"},
      {ParticleType::kNeutron, 100, "neutron 100 MeV"},
      {ParticleType::kPionPlus, 300, "pi+ 300 MeV"},
      {ParticleType::kPionMinus, 300, "pi- 300 MeV"},
      {ParticleType::kAlpha, 840, "alpha 840 MeV"},
      {ParticleType::kDeuteron, 400, "deuteron 400 MeV"},
  };
  // The three B1 materials with a hadronic target worth having: hydrogen-rich (water),
  // calcium-bearing (compact bone) and light (air, where the mean free path is metres and the
  // Poisson mean is small - which is the cell that catches a cross section scaled by the wrong
  // atom density).
  const int kMats[] = {data::kWater, data::kBoneCompact, data::kAir};
  const char* kMatNames[] = {"water", "bone", "air"};
  constexpr long long kTracks = 400;
  constexpr int kMaxSteps = 4000;
  const double kSigmaGate = 5.0;

  // ============================================================================================
  // 3a. The macroscopic cross section itself, against GEANT4's per-element one.
  // ============================================================================================
  //
  // THIS SECTION EXISTS BECAUSE SECTION 3 ON ITS OWN IS VACUOUS, and that was found by running
  // it with the fix removed rather than by arguing about it. Section 3 builds its Poisson mean
  // out of the SAME `inelastic_xs_per_volume` the stepper draws its length from, so a cross
  // section that is wrong by a factor moves the expectation and the observation together and
  // the sigma column does not twitch. Perturbed - `return 2 * v.value` - section 3 passed at a
  // worst 2.58 sigma, exactly as it does unperturbed. That is docs/RISK.md **V52** again: "the
  // test compared its OWN copy of each formula against the oracle column, agreed with itself to
  // the last bit, and went on agreeing when the port's copy was changed underneath it".
  //
  // So the two questions are asked separately, and only together do they mean anything:
  //
  //   3a  is SIGMA Geant4's?           against `ref/oracle/had_particlexs.csv`, which is
  //                                    `G4ParticleInelasticXS::GetElementCrossSection` and
  //                                    `G4NeutronInelasticXS`'s, dumped per element per energy
  //   3   is the LENGTH drawn from that sigma?   the Poisson comparison below
  //
  // The macroscopic cross section is `sum_i n_i * sigma_i` over the material's elements, which
  // is what `G4CrossSectionDataStore::ComputeCrossSection` builds; the oracle's `xs_mm2` is the
  // per-element microscopic one, so the sum is done here out of the oracle's numbers and the
  // material's own atom densities. Compared at the ORACLE'S OWN NODE ENERGIES, so neither side
  // interpolates and the comparison is exact rather than tolerant.
  {
    std::printf("== 3a. the macroscopic cross section against ref/oracle/had_particlexs.csv "
                "==\n");
    struct Row { std::string ds, part; int z; double e, xs; };
    std::vector<Row> rows;
    {
      FILE* f = std::fopen((std::string(std::getenv("G4GPU_ORACLE")
                                            ? std::getenv("G4GPU_ORACLE")
                                            : "ref/oracle")
                            + "/had_particlexs.csv").c_str(), "r");
      if (f == nullptr) {
        fail("cannot open had_particlexs.csv - run ref/oracle/run.bat tables first");
      } else {
        char line[512];
        bool first = true;
        while (std::fgets(line, sizeof(line), f) != nullptr) {
          if (first) { first = false; continue; }
          char ds[64], part[64];
          int z = 0;
          double e = 0, xs = 0;
          if (std::sscanf(line, "%63[^,],%63[^,],%d,%lf,%lf", ds, part, &z, &e, &xs) == 5) {
            rows.push_back({ds, part, z, e, xs});
          }
        }
        std::fclose(f);
      }
    }
    struct Sp { ParticleType t; const char* ds; const char* part; };
    const Sp sps[] = {
        {ParticleType::kProton, "ParticleInelastic", "proton"},
        {ParticleType::kDeuteron, "ParticleInelastic", "deuteron"},
        {ParticleType::kTriton, "ParticleInelastic", "triton"},
        {ParticleType::kHe3, "ParticleInelastic", "He3"},
        {ParticleType::kAlpha, "ParticleInelastic", "alpha"},
        {ParticleType::kNeutron, "NeutronInelastic", "neutron"},
    };
    const int mats3a[] = {data::kWater, data::kBoneCompact, data::kAir};
    long long n_cmp = 0;
    double worst_rel = 0;
    std::string worst_at;
    for (const Sp& sp : sps) {
      // The oracle rows for this species, indexed by (Z, energy).
      std::vector<const Row*> mine;
      for (const Row& r : rows) {
        if (r.ds == sp.ds && r.part == sp.part) { mine.push_back(&r); }
      }
      if (mine.empty()) {
        fail(std::string("no oracle rows for ") + sp.ds + "/" + sp.part);
        continue;
      }
      // Every distinct energy the oracle carries, in the band the transport uses.
      std::vector<double> energies;
      for (const Row* r : mine) {
        if (r->z != 1) { continue; }
        if (r->e < 1.0 || r->e > 1.0e5) { continue; }  // 1 MeV .. 100 GeV
        energies.push_back(r->e);
      }
      for (int mi = 0; mi < 3; ++mi) {
        const data::Material<real_t>& m = mats[mats3a[mi]];
        for (double e : energies) {
          // Geant4's answer: sum over the material's elements of n_i * sigma_i(Z_i, E).
          double want = 0;
          bool complete = true;
          for (int i = 0; i < m.n_elements; ++i) {
            const int z = static_cast<int>(m.z[i] + real_t(0.5));
            const Row* hit = nullptr;
            for (const Row* r : mine) {
              if (r->z == z && r->e == e) { hit = r; break; }
            }
            if (hit == nullptr) { complete = false; break; }
            want += static_cast<double>(m.n_atoms[i]) * hit->xs;
          }
          if (!complete) { continue; }
          hxs::MaterialXs<real_t> mx{};
          const double got = static_cast<double>(had::inelastic_xs_per_volume<real_t>(
              xs, m, sp.t, static_cast<real_t>(e), 0, 0, mx));
          const double denom = (want != 0.0) ? std::fabs(want) : 1.0;
          const double rel = std::fabs(got - want) / denom;
          if (rel > worst_rel) {
            worst_rel = rel;
            char b[192];
            std::snprintf(b, sizeof(b), "%s in material %d at %.6g MeV: oracle %.17g, port "
                                        "%.17g", sp.part, mats3a[mi], e, want, got);
            worst_at = b;
          }
          ++n_cmp;
        }
      }
    }
    const double kXsTol = 1e-12;
    if (worst_rel > kXsTol) {
      fail("the macroscopic inelastic cross section is not Geant4's: " + worst_at);
    }
    std::printf("  %lld (species, material, energy) points, worst relative %.3g against a "
                "%.0e tolerance  [%s]\n", n_cmp, worst_rel, kXsTol,
                worst_at.empty() ? "exact everywhere" : worst_at.c_str());
    if (n_cmp == 0) {
      fail("no cross-section points were compared at all - the oracle rows and the materials' "
           "elements did not intersect, which makes section 3 vacuous again");
    }
    std::printf("\n");
  }

  std::printf("== 3. the interaction rate against the cross section, and 4. the balance ==\n");
  std::printf("  %-18s %-6s %9s %9s %8s %8s %7s %9s %7s\n", "beam", "mat", "expected",
              "queued", "sigma", "ran", "reject", "worst dE", "max sec");

  double worst_sigma = 0;
  std::string worst_where;

  for (const Beam& b : beams) {
    for (int mi = 0; mi < 3; ++mi) {
      const int mat = kMats[mi];
      geom::Volume<real_t> vols[1] = {
          {{geom::SolidType::kBox, {400, 400, 400}},
           geom::make_translation<real_t>({0, 0, 0}), 0, mat, /*score_index=*/0}};
      Scene<real_t> scene{};
      scene.geometry = geom::Geometry<real_t>{vols, 1, 0};
      scene.materials = mats;
      scene.hadron_range = hrt;
      scene.range_cut = real_t(0.7);
      scene.scoring_volume = 0;

      had::HadronicWiring<real_t> had{};
      had.stage = had::HadronicStage::kFinal;
      had.decay = true;
      had.hadron_elastic = false;  // not under test here; test_step_hadron.cu owns it
      had.neutron_capture = false;
      had.hadron_inelastic = true;
      had.hadron_at_rest = false;  // section 5
      had.inelastic = xs;
      had.neutron.inelastic = &ds_n;
      had.level_data = lt;
      had.books.count = ref_n.data();
      had.books.energy = ref_e.data();
      had.queue.items = qbuf.data();
      had.queue.cursor = &qcursor;
      had.queue.capacity = static_cast<int>(qbuf.size());

      Cell cell;
      const std::vector<int> ref_n0 = ref_n;

      for (long long k = 0; k < kTracks; ++k) {
        TrackState<real_t> p{};
        p.species = b.t;
        p.pos = Vec3<real_t>{0, 0, -390};
        p.dir = Vec3<real_t>{0, 0, 1};
        p.ekin = b.e;
        p.volume = 0;
        p.event = 0;
        p.rng_key = 7001u + static_cast<unsigned int>(k) * 2654435761u;
        p.step = 0u;
        p.begin(p.pos, p.dir, p.ekin, 0, 0u, ProcessId::fNotDefined, real_t(0), real_t(1));

        bool alive = true;
        for (int st = 0; alive && st < kMaxSteps; ++st) {
          // The hazard integral, from the SAME function the stepper is about to call, at the
          // pre-step energy and in the pre-step material. This is what makes the comparison a
          // check on the drawn length rather than on a remembered number.
          hxs::MaterialXs<real_t> mxs{};
          const em::SteppedHadron<real_t> h =
              (b.t == ParticleType::kGenericIon)
                  ? em::stepped_ion<real_t>(ion_z_of(p.ion_za), ion_a_of(p.ion_za))
                  : em::stepped_hadron<real_t>(b.t);
          const real_t sigma = had::inelastic_xs_per_volume<real_t>(
              xs, mats[mat], b.t, p.ekin, h.z, h.a, mxs);

          CountingEmitter em;
          em.books = had.books;
          StepReport<real_t> rep;
          Philox<real_t> rng(p.rng_key, p.step, 0xB19Du);
          real_t edep = 0;
          bool queued = false;
          qcursor = 0;
          const real_t ekin_pre = p.ekin;
          alive = step_hadron(scene, p, b.t, had, rng, em, edep, rep, vis::no_capture(),
                              &queued);
          cell.expected += static_cast<double>(rep.true_length) * static_cast<double>(sigma);
          ++p.step;

          if (queued && qcursor > 0) {
            ++cell.queued;
            const had::PendingInteraction<real_t>& q = qbuf[0];
            had::InteractionSlot<real_t>& slot = pool.slots[0];
            Philox<real_t> irng(q.track.rng_key, q.track.step, had::kInteractionRngPurpose);
            hp::HadProjectile<real_t> proj;
            proj.pdg = pdg_code(q.species);
            proj.charge = particle_def<real_t>(q.species).charge;
            proj.mass = particle_def<real_t>(q.species).mass;
            proj.kin_energy = q.track.ekin;
            proj.baryon_number = had::baryon_number_of(q.species, ion_a_of(q.track.ion_za));
            had::InteractionOutcome oc;
            switch (q.bucket) {
              case had::InteractionBucket::kFtfp:
                oc = had::run_inelastic<real_t, had::InelasticModel::kFtfp>(
                    proj, q.species, mats[q.material], xs, q.xs_at_step_start, slot, pool.view,
                    0, lt, pool.view.fermi, h.z, h.a, irng);
                break;
              case had::InteractionBucket::kBertini:
                oc = had::run_inelastic<real_t, had::InelasticModel::kBertini>(
                    proj, q.species, mats[q.material], xs, q.xs_at_step_start, slot, pool.view,
                    0, lt, pool.view.fermi, h.z, h.a, irng);
                break;
              case had::InteractionBucket::kBinary:
                oc = had::run_inelastic<real_t, had::InelasticModel::kBinary>(
                    proj, q.species, mats[q.material], xs, q.xs_at_step_start, slot, pool.view,
                    0, lt, pool.view.fermi, h.z, h.a, irng);
                break;
              case had::InteractionBucket::kLightIon:
                oc = had::run_inelastic<real_t, had::InelasticModel::kLightIon>(
                    proj, q.species, mats[q.material], xs, q.xs_at_step_start, slot, pool.view,
                    0, lt, pool.view.fermi, h.z, h.a, irng);
                break;
              default:
                break;
            }
            // The two bookings `run_interaction` makes, made here too, because the ledger is
            // where deliverable 6 lives: a refusal is not refused by name until its RATE is on
            // a report. Without this the alpha rows below read "215 queued, 20 ran" and say
            // nothing about where the other 195 went.
            // The WHY whenever there is one, `ran` or not - see `run_interaction`'s own note:
            // an at-rest capture that refuses its nuclear half still emitted the atomic
            // cascade's gammas, so `ran` and a real refusal arrive together.
            if (oc.refusal != had::HadronicRefusal::kNumHadronicRefusals) {
              had::book_refusal<real_t>(had.books, oc.refusal, q.track.ekin);
            }
            if (!oc.ran && !oc.rejected_by_integral_xs) {
              had::book_refusal<real_t>(
                  had.books,
                  (q.species == ParticleType::kNeutron)
                      ? had::HadronicRefusal::kNeutronInelastic
                      : had::HadronicRefusal::kChargedHadronInelastic,
                  q.track.ekin);
            }
            if (oc.rejected_by_integral_xs) {
              ++cell.rejected;
            } else if (oc.ran) {
              ++cell.ran;
              // FillResult, then the balance.
              for (int i = 0; i < slot.fs.n_secondaries; ++i) {
                slot.pdg_mass[i] = had::definition_mass_of<real_t>(slot.fs.secondaries[i]);
              }
              slot.filled = hp::fill_result<real_t, had::kInteractionSecondaryCap,
                                            had::kInteractionSecondaryCap>(
                  slot.fs, q.track.dir, q.track.global_time, q.track.weight,
                  had::has_at_rest_arm(q.species), slot.pdg_mass);
              CountingEmitter ie;
              ie.books = had.books;
              had::emit_interaction_result<real_t>(slot.filled, ie, had.books);
              cell.secondaries += ie.n_secondaries;
              if (ie.n_secondaries > cell.max_secondaries) {
                cell.max_secondaries = ie.n_secondaries;
              }

              // ---- 4. baryon number and charge, projectile + target against the products.
              //
              // WITHIN THE REFUSAL LEDGER, which is the qualifier that makes it checkable: a
              // secondary this transport cannot step is BOOKED and not emitted, so its baryon
              // number left with it. `kInelasticSecondarySpecies` counts exactly those, and a
              // cell with none of them must balance exactly.
              const int refused_species =
                  ref_n[static_cast<int>(had::HadronicRefusal::kInelasticSecondarySpecies)]
                  - ref_n0[static_cast<int>(had::HadronicRefusal::kInelasticSecondarySpecies)];
              if (refused_species == 0) {
                const int b_in = proj.baryon_number + oc.target_a;
                const int q_in = static_cast<int>(std::lrint(proj.charge)) + oc.target_z;
                int b_out = ie.baryon, q_out = ie.charge;
                if (slot.filled.status == hp::TrackStatusChange::kAlive) {
                  b_out += proj.baryon_number;
                  q_out += static_cast<int>(std::lrint(proj.charge));
                }
                if (b_out != b_in) { ++cell.bad_baryon; }
                if (q_out != q_in) { ++cell.bad_charge; }
                ++cell.n_balance;
              }
              const double e_in = static_cast<double>(proj.kin_energy);
              const double e_out = ie.secondary_energy
                                   + static_cast<double>(slot.filled.local_energy_deposit)
                                   + (slot.filled.status == hp::TrackStatusChange::kAlive
                                          ? static_cast<double>(slot.filled.energy)
                                          : 0.0);
              // Q-VALUE, NOT ZERO. An inelastic reaction releases or absorbs binding energy, so
              // the out-minus-in difference is a physical number and not a residual - it is
              // bounded by `CheckResult`'s own levels, which is the only bound Geant4 puts on
              // it either (2% of the kinetic energy AND 1 GeV, both required).
              const double d = std::fabs(e_out - e_in);
              if (d > cell.worst_energy) { cell.worst_energy = d; }
            }
          }
          if (p.volume == geom::kOutsideWorld) { break; }
        }
      }

      for (std::size_t r = 0; r < ref_n.size(); ++r) {
        cell.refused[r] = ref_n[r] - ref_n0[r];
      }

      // ---- 3. the Poisson comparison.
      const double sd = std::sqrt(cell.expected > 0 ? cell.expected : 1.0);
      const double sig = (cell.expected > 0)
                             ? (double(cell.queued) - cell.expected) / sd
                             : 0.0;
      if (std::fabs(sig) > worst_sigma) {
        worst_sigma = std::fabs(sig);
        worst_where = std::string(b.name) + " in " + kMatNames[mi];
      }
      if (std::fabs(sig) > kSigmaGate) {
        char msg[256];
        std::snprintf(msg, sizeof(msg),
                      "%s in %s: expected %.1f interactions from the cross section, the "
                      "stepper drew %lld (%.2f sigma)",
                      b.name, kMatNames[mi], cell.expected, cell.queued, sig);
        fail(msg);
      }
      if (cell.bad_baryon > 0) {
        fail(std::string(b.name) + " in " + kMatNames[mi] + ": "
             + std::to_string(cell.bad_baryon) + " of " + std::to_string(cell.n_balance)
             + " interactions do not balance baryon number");
      }
      if (cell.bad_charge > 0) {
        fail(std::string(b.name) + " in " + kMatNames[mi] + ": "
             + std::to_string(cell.bad_charge) + " of " + std::to_string(cell.n_balance)
             + " interactions do not balance charge");
      }
      std::printf("  %-18s %-6s %9.1f %9lld %8.2f %8lld %7lld %9.3g %7d\n", b.name,
                  kMatNames[mi], cell.expected, cell.queued, sig, cell.ran, cell.rejected,
                  cell.worst_energy, cell.max_secondaries);
      if (cell.max_secondaries > g_max_secondaries) {
        g_max_secondaries = cell.max_secondaries;
      }
    }
  }
  std::printf("  worst %.2f sigma against a %.1f gate  [%s]\n", worst_sigma, kSigmaGate,
              worst_where.c_str());
  // THE NUMBER `core/track_buffer.cuh`'s `max_secondaries_per_step` IS SET FROM.
  //
  // That function's own header says the hadronic rows are the wiring package's to raise and
  // does the arithmetic for why raising the shared constant would not do: the engine's
  // output-pool budget is `(capacity - live) / reservation`, so 64 for everything would let a
  // gamma run step 1.6% as many tracks per iteration as 4 does. Here is the other half - what
  // a hadronic reservation actually has to cover, measured over 24 cells and 3,200 tracks.
  //
  // Being UNDER it is not silent: the launch overruns the output pool, `TrackBuffer::append`
  // counts the overflow and `BeamOn` halves the batch and retries. Being over it costs every
  // hadron run iterations it did not need.
  std::printf("  most secondaries from ONE interaction, over every cell: %d "
              "(max_secondaries_per_step's hadronic rows are set from this)\n\n",
              g_max_secondaries);
  if (g_max_secondaries > max_secondaries_per_step(kSpeciesProton)) {
    char b[256];
    std::snprintf(b, sizeof(b),
                  "one interaction emitted %d secondaries and "
                  "max_secondaries_per_step(kSpeciesProton) reserves %d - the output pool's "
                  "budget is too small and BeamOn will halve the batch to cope",
                  g_max_secondaries, max_secondaries_per_step(kSpeciesProton));
    fail(b);
  }

  // ---- A TEST THAT PASSED BECAUSE NOTHING HAPPENED IS THE FAILURE MODE THIS GUARDS.
  {
    long long total_queued = 0;
    for (const Beam& b : beams) { (void)b; }
    // Recomputed from the ledger rather than kept: the two SIZE counters are one booking per
    // lost interaction, and `ran` is counted above. If nothing was queued at all the sigma
    // column is a column of zeros and every gate above passes.
    total_queued =
        ref_n[static_cast<int>(had::HadronicRefusal::kChargedHadronInelastic)]
        + ref_n[static_cast<int>(had::HadronicRefusal::kNeutronInelastic)];
    if (worst_sigma == 0.0) {
      fail("not one interaction was drawn in any cell - the cross section is zero everywhere, "
           "which is what an unwired table looks like");
    }
    std::printf("  refusals over the whole grid (the SIZE group and the WHY group are not "
                "summed - see had::HadronicRefusal):\n");
    for (std::size_t r = 0; r < ref_n.size(); ++r) {
      if (ref_n[r] > 0) {
        std::printf("    %10d   %12.6g MeV   %s\n", ref_n[r], ref_e[r],
                    had::hadronic_refusal_name(static_cast<had::HadronicRefusal>(r)));
      }
    }
    (void)total_queued;
    std::printf("\n");
  }

  // ============================================================================================
  // 5. The at-rest hook
  // ============================================================================================
  std::printf("== 5. the at-rest hook: a stopped pi- and mu- ==\n");
  {
    const int mat = data::kWater;
    geom::Volume<real_t> vols[1] = {
        {{geom::SolidType::kBox, {400, 400, 400}}, geom::make_translation<real_t>({0, 0, 0}),
         0, mat, 0}};
    Scene<real_t> scene{};
    scene.geometry = geom::Geometry<real_t>{vols, 1, 0};
    scene.materials = mats;
    scene.hadron_range = hrt;
    scene.range_cut = real_t(0.7);
    scene.scoring_volume = 0;

    had::HadronicWiring<real_t> had{};
    had.stage = had::HadronicStage::kFinal;
    had.decay = true;
    had.hadron_elastic = false;
    had.neutron_capture = false;
    had.hadron_inelastic = false;  // the stopped particle must reach the AT-REST branch
    had.hadron_at_rest = true;
    had.level_data = lt;
    had.books.count = ref_n.data();
    had.books.energy = ref_e.data();
    had.queue.items = qbuf.data();
    had.queue.cursor = &qcursor;
    had.queue.capacity = static_cast<int>(qbuf.size());

    struct S { ParticleType t; real_t e; const char* name; };
    const S sp[] = {{ParticleType::kPionMinus, 20, "pi-"}, {ParticleType::kMuonMinus, 20, "mu-"}};
    for (const S& s : sp) {
      long long stopped = 0, captured = 0, refused = 0, decayed_in_orbit = 0;
      long long em_gammas = 0, nuclear_sec = 0, decayed_in_flight = 0;
      long long snapped_wrong = 0;  // secondaries fill_result moved by more than their own dm
      const std::vector<int> ref0 = ref_n;
      for (long long k = 0; k < 200; ++k) {
        TrackState<real_t> p{};
        p.species = s.t;
        p.pos = Vec3<real_t>{0, 0, 0};
        p.dir = Vec3<real_t>{0, 0, 1};
        p.ekin = s.e;
        p.volume = 0;
        p.event = 0;
        p.rng_key = 31337u + static_cast<unsigned int>(k) * 40503u;
        p.step = 0u;
        p.begin(p.pos, p.dir, p.ekin, 0, 0u, ProcessId::fNotDefined, real_t(0), real_t(1));
        bool alive = true;
        for (int st = 0; alive && st < kMaxSteps; ++st) {
          CountingEmitter em;
          em.books = had.books;
          StepReport<real_t> rep;
          Philox<real_t> rng(p.rng_key, p.step, 0xB19Du);
          real_t edep = 0;
          bool queued = false;
          qcursor = 0;
          alive = step_hadron(scene, p, s.t, had, rng, em, edep, rep, vis::no_capture(),
                              &queued);
          ++p.step;
          if (!queued && !alive && rep.process == ProcessId::fDecay) { ++decayed_in_flight; }
          if (queued && qcursor > 0) {
            ++stopped;
            const had::PendingInteraction<real_t>& q = qbuf[0];
            if (q.kind != had::InteractionKind::kAtRest) {
              fail(std::string(s.name) + ": a stopped track queued an IN-FLIGHT interaction");
              break;
            }
            had::InteractionSlot<real_t>& slot = pool.slots[0];
            Philox<real_t> irng(q.track.rng_key, q.track.step, had::kInteractionRngPurpose);
            had::CompositionScratch<real_t> sc{};
            const hp::MaterialComposition<real_t> mc =
                had::material_composition_of<real_t>(mats[q.material], sc);
            hp::HadProjectile<real_t> proj;
            proj.pdg = pdg_code(q.species);
            proj.baryon_number = had::baryon_number_of(q.species, 0);
            proj.charge = particle_def<real_t>(q.species).charge;
            proj.mass = particle_def<real_t>(q.species).mass;
            proj.kin_energy = 0;
            // A FRESH SLOT'S `pdg_mass` - zeros - which is what the kernel's first capture in a
            // slot saw, and the worst case of the stale masses every later one saw. docs/RISK.md
            // V199: `run_at_rest` must refill it, and the check below is what says it does.
            for (int j = 0; j < had::kInteractionSecondaryCap; ++j) { slot.pdg_mass[j] = 0; }
            stop::AtRestResult ar;
            const had::InteractionOutcome oc = had::run_at_rest<real_t>(
                proj, mc, slot, pool.view, 0, lt, pool.view.fermi, had::NuclearMassMeV(), irng,
                ar);
            // AND WHAT THE KERNEL DOES NEXT, which this section did not do until V199 and which
            // is where the defect lived: `fill_result` snaps each secondary onto
            // `slot.pdg_mass[j]` and adds the difference to its kinetic energy. So every
            // secondary must come out with its own model energy plus its own `m_dyn - m_def`
            // off-shell correction and NOTHING ELSE - a proton that gained 938 MeV is a proton
            // snapped against a zero.
            if (oc.ran) {
              slot.filled = hp::fill_result<real_t, had::kInteractionSecondaryCap,
                                            had::kInteractionSecondaryCap>(
                  slot.fs, Vec3<real_t>{0, 0, 1}, real_t(0), real_t(1),
                  had::has_at_rest_arm(q.species), slot.pdg_mass);
              for (int j = 0; j < slot.fs.n_secondaries && j < slot.filled.n_secondaries; ++j) {
                const auto& m = slot.fs.secondaries[j];
                const double dm = double(m.mass) - double(had::definition_mass_of<real_t>(m));
                double expect = double(m.kin_energy) + ((dm > 1e-3 || dm < -1e-3) ? dm : 0.0);
                if (expect < 1e-9) { expect = 1e-9; }
                if (std::fabs(double(slot.filled.secondaries[j].kin_energy) - expect) > 1e-6) {
                  ++snapped_wrong;
                  if (snapped_wrong == 1) {
                    std::printf("   first wrong snap: pdg %d z %d a %d  model T %.4f  handed on "
                                "%.4f  expected %.4f MeV\n", m.pdg, m.z, m.a,
                                double(m.kin_energy),
                                double(slot.filled.secondaries[j].kin_energy), expect);
                  }
                }
              }
            }
            em_gammas += ar.n_em_cascade;
            if (ar.decayed_in_orbit) { ++decayed_in_orbit; }
            if (oc.refusal != had::HadronicRefusal::kNumHadronicRefusals) {
              ++refused;
            } else {
              ++captured;
              nuclear_sec += slot.fs.n_secondaries - ar.n_em_cascade;
            }
          }
          if (p.volume == geom::kOutsideWorld) { break; }
        }
      }
      std::printf("  %-4s 200 tracks: %lld stopped and queued at rest, %lld captured, %lld "
                  "refused by name, %lld decayed in orbit, %lld decayed IN FLIGHT; %lld "
                  "atomic-cascade secondaries, %lld nuclear\n",
                  s.name, stopped, captured, refused, decayed_in_orbit, decayed_in_flight,
                  em_gammas, nuclear_sec);
      // EVERY TRACK MUST END IN ONE OF THE TWO, and the second one is not a leak - it is the
      // thing this assertion was originally written without and was wrong about.
      //
      // A 20 MeV pi- in water has a range of a few millimetres and cannot leave a 400 mm box,
      // so "all 200 stop" looks obvious. It is not: `in_flight_mean_free_path` is
      // `beta*gamma*c*tau`, and beta*gamma goes to ZERO as the track slows, so a pion's decay
      // mean free path collapses from 4.3 m at 20 MeV to microns at the tracking cut. Three of
      // 200 decay in the last few steps of the range rather than at rest. That is Geant4's
      // arithmetic and not this port's - `G4Decay::PostStepGetPhysicalInteractionLength` has
      // the same `betagamma*c*tau` - and the muon shows the other side of it: c*tau is 659 m,
      // so its in-flight length never collapses and all 200 of them stop.
      //
      // So the assertion is that nothing is LOST, not that everything stops.
      if (stopped + decayed_in_flight != 200) {
        fail(std::string(s.name) + ": " + std::to_string(stopped) + " stopped + "
             + std::to_string(decayed_in_flight) + " decayed in flight is not 200 - a track "
               "ended some other way, and a 20 MeV track cannot leave a 400 mm box");
      }
      if (stopped == 0) {
        fail(std::string(s.name) + ": no track reached the at-rest branch at all");
      }
      // The atomic cascade runs BEFORE the nuclear model and its gammas survive whatever the
      // nuclear half does - P12b asserted 1,080,164 of them over its own campaign. So a
      // refusal must not cost them.
      if (em_gammas == 0) {
        fail(std::string(s.name)
             + ": the atomic cascade emitted nothing - G4EmCaptureCascade always emits");
      }
      // THE SNAP MUST USE THESE SECONDARIES' OWN MASSES (V199). A zero or a stale entry in
      // `slot.pdg_mass` adds a secondary's whole rest mass to its kinetic energy, and the B1
      // sweep's proton_1000 read +29% (9.8 sigma) on exactly that before it was found.
      if (snapped_wrong != 0) {
        fail(std::string(s.name) + ": " + std::to_string(snapped_wrong)
             + " at-rest secondaries were snapped by fill_result against masses that are not "
               "theirs - run_at_rest left slot.pdg_mass stale");
      }
      std::printf("  %-4s fill_result snapped every at-rest secondary against its own mass: %s\n",
                  s.name, snapped_wrong == 0 ? "yes" : "NO");
      (void)ref0;
    }
  }

  // ============================================================================================
  // 6. `ion_inelastic` switches the five ion names and NOTHING ELSE
  // ============================================================================================
  //
  // WHY THIS SECTION EXISTS. `HadronicWiring::ion_inelastic` is a study switch, not physics: it
  // is there so a like-for-like column can inactivate on the port side exactly the set the
  // Geant4 side inactivated. The first full P15 sweep had the five ion processes off in Geant4
  // and ON in the port, and alpha_840 read -39.27% (-201.8 sigma) because the port drew an
  // interaction length, refused 91.5% of the interactions by name, and killed each refused ion
  // with its energy deposited at the refusal point - upstream of the range it would have
  // travelled. docs/RISK.md V192.
  //
  // So the switch has to do two things and this section asserts both, because getting either
  // half wrong reproduces a different version of that defect:
  //
  //   * off, an alpha draws NO interaction length at all - not "draws one and refuses it"
  //   * off, a PROTON is untouched - `protonInelastic` is wired and is not one of the five
  //
  // A gate written as `had.hadron_inelastic && had.ion_inelastic` passes the first and fails the
  // second; a gate that forgets the species predicate passes the second and fails the first.
  std::printf("== 6. ion_inelastic switches d/t/He3/alpha/ion and not the proton ==\n");
  {
    const int mat = data::kWater;
    geom::Volume<real_t> vols[1] = {
        {{geom::SolidType::kBox, {400, 400, 400}}, geom::make_translation<real_t>({0, 0, 0}),
         0, mat, 0}};
    Scene<real_t> scene{};
    scene.geometry = geom::Geometry<real_t>{vols, 1, 0};
    scene.materials = mats;
    scene.hadron_range = hrt;
    scene.range_cut = real_t(0.7);
    scene.scoring_volume = 0;

    struct S { ParticleType t; real_t e; int z; int a; const char* name; bool is_ion; };
    const S sp[] = {
        {ParticleType::kAlpha,      840,  0, 0, "alpha 840 MeV",    true},
        {ParticleType::kDeuteron,   400,  0, 0, "deuteron 400 MeV", true},
        {ParticleType::kHe3,        400,  0, 0, "He3 400 MeV",      true},
        {ParticleType::kTriton,     400,  0, 0, "triton 400 MeV",   true},
        {ParticleType::kGenericIon, 2000, 6, 12, "C12 2 GeV",       true},
        {ParticleType::kProton,     210,  0, 0, "proton 210 MeV",   false},
        {ParticleType::kPionPlus,   500,  0, 0, "pi+ 500 MeV",      false},
    };

    for (const S& s : sp) {
      long long queued[2] = {0, 0};  // [0] = ion_inelastic off, [1] = on
      for (int on = 0; on < 2; ++on) {
        had::HadronicWiring<real_t> had{};
        had.stage = had::HadronicStage::kFinal;
        had.decay = true;
        had.hadron_elastic = false;
        had.neutron_capture = false;
        had.hadron_inelastic = true;
        had.ion_inelastic = (on != 0);
        had.hadron_at_rest = false;
        had.inelastic = xs;
        had.neutron.inelastic = &ds_n;
        had.level_data = lt;
        had.books.count = ref_n.data();
        had.books.energy = ref_e.data();
        had.queue.items = qbuf.data();
        had.queue.cursor = &qcursor;
        had.queue.capacity = static_cast<int>(qbuf.size());

        for (long long k = 0; k < 400; ++k) {
          TrackState<real_t> p{};
          p.species = s.t;
          p.pos = Vec3<real_t>{0, 0, -390};
          p.dir = Vec3<real_t>{0, 0, 1};
          p.ekin = s.e;
          p.ion_za = ion_za_of(s.z, s.a);
          p.volume = 0;
          p.event = 0;
          p.rng_key = 90001u + static_cast<unsigned int>(k) * 2246822519u;
          p.step = 0u;
          p.begin(p.pos, p.dir, p.ekin, 0, 0u, ProcessId::fNotDefined, real_t(0), real_t(1));

          bool alive = true;
          for (int st = 0; alive && st < kMaxSteps; ++st) {
            CountingEmitter em;
            em.books = had.books;
            StepReport<real_t> rep;
            Philox<real_t> rng(p.rng_key, p.step, 0xB19Du);
            real_t edep = 0;
            bool q = false;
            qcursor = 0;
            alive = step_hadron(scene, p, s.t, had, rng, em, edep, rep, vis::no_capture(), &q);
            ++p.step;
            if (q && qcursor > 0) { queued[on] += 1; break; }
          }
        }
      }

      if (s.is_ion) {
        if (queued[0] != 0) {
          fail(std::string(s.name) + ": ion_inelastic off still queued "
               + std::to_string(queued[0]) + " interactions - the ion drew an interaction "
                 "length it must not draw, which is the one-sided inactivation of V192");
        }
        // AND THE SWITCH MUST HAVE SOMETHING TO SWITCH. A species that queues nothing with the
        // flag ON makes the assertion above vacuous - it would pass against a gate that never
        // lets any ion interact, which is a different bug with the same symptom.
        if (queued[1] == 0) {
          fail(std::string(s.name) + ": ion_inelastic ON queued nothing either, so the "
                 "assertion above is vacuous - this species has no inelastic process to switch");
        }
      } else {
        if (queued[0] != queued[1]) {
          fail(std::string(s.name) + ": ion_inelastic changed a NON-ion species - "
               + std::to_string(queued[0]) + " queued off against " + std::to_string(queued[1])
               + " on. protonInelastic and pi+Inelastic are wired and are not among the five "
                 "names G4IonPhysics registers");
        }
        if (queued[1] == 0) {
          fail(std::string(s.name) + ": queued nothing at all, so the equality above is "
                 "vacuous");
        }
      }
      std::printf("   %-18s off %4lld   on %4lld   %s\n", s.name, queued[0], queued[1],
                  s.is_ion ? "(ion: must be 0 / >0)" : "(not an ion: must be equal)");
    }
    qcursor = 0;
  }

  // ============================================================================================
  // 7. Every refusal the Binary cascade can raise is a refusal here
  // ============================================================================================
  //
  // WHY THIS SECTION EXISTS. `run_arm_binary` tested four of `bic::BicRefusal`'s six flags -
  // `species`, `hydrogen`, `preco_projectile`, `capacity` - from P15's first commit. The other
  // two, `nucleus` (`G4Fancy3DNucleus::Init` could not place the nucleons) and `cascade`
  // (whatever `Propagate` refused, `FillVoidNucleusProducts` included), return early with
  // `bic_fs` holding nothing a caller may apply, and the arm copied it and reported a final
  // state. `BicRefusal::any()` names all six; the arm uses it now.
  //
  // The nucleus refusal is forced with a target the slot cannot hold - A = 300 against the 256
  // nucleons `InteractionSlot` carries - which is `NucleusReport::capacity` and therefore
  // `nucleus`. No material in this port has such a nuclide, and that is the point: the test is
  // of the MAPPING, which must be total whatever reaches it. A real target and a hydrogen target
  // are run beside it, so the section cannot pass by refusing everything.
  std::printf("== 7. every Binary-cascade refusal reaches the ledger ==\n");
  {
    had::InteractionSlot<real_t>& slot = pool.slots[0];
    const preco::PrecoWorkspace pws = had::preco_workspace_of<real_t>(slot);
    hp::HadProjectile<real_t> proj;
    proj.pdg = pdg_code(ParticleType::kProton);
    proj.charge = particle_def<real_t>(ParticleType::kProton).charge;
    proj.mass = particle_def<real_t>(ParticleType::kProton).mass;
    proj.kin_energy = 400;
    proj.baryon_number = 1;

    struct Case {
      int z, a;
      const char* name;
      bool expect_ran;
      had::HadronicRefusal expect_refusal;
    };
    const Case cases[] = {
        {8, 16, "p 400 MeV on O16", true, had::HadronicRefusal::kNumHadronicRefusals},
        {1, 1, "p 400 MeV on H1", false, had::HadronicRefusal::kBinaryHydrogenTarget},
        {120, 300, "p 400 MeV on A=300", false, had::HadronicRefusal::kBinaryRefused},
    };
    for (const Case& c : cases) {
      int ran = 0, refused_as_expected = 0, other = 0;
      for (unsigned int k = 0; k < 20u; ++k) {
        hp::HadNucleus tgt;
        tgt.z = c.z;
        tgt.a = c.a;
        Philox<real_t> rng(424242u + 7919u * k, 0u, had::kInteractionRngPurpose);
        had::InteractionOutcome out;
        const bool ok = had::run_arm_binary<real_t>(proj, tgt, slot, pool.view, pws, lt,
                                                    pool.view.fermi, rng, out);
        if (ok) {
          ++ran;
        } else if (out.refusal == c.expect_refusal) {
          ++refused_as_expected;
        } else {
          ++other;
        }
      }
      std::printf("   %-22s ran %2d   refused as expected %2d   other %2d\n", c.name, ran,
                  refused_as_expected, other);
      if (c.expect_ran) {
        if (ran != 20) {
          fail(std::string(c.name) + ": " + std::to_string(ran) + " of 20 ran - a real target "
               "must give a final state every time, so the other cases would prove nothing");
        }
      } else {
        if (ran != 0) {
          fail(std::string(c.name) + ": " + std::to_string(ran) + " of 20 were APPLIED as "
               "final states - the model refused and the arm reported success");
        }
        if (refused_as_expected != 20) {
          fail(std::string(c.name) + ": only " + std::to_string(refused_as_expected)
               + " of 20 booked the expected refusal name");
        }
      }
    }
  }

  // ============================================================================================
  // 8. A model's own "nothing happened" is Geant4's answer, not a hole
  // ============================================================================================
  //
  // WHY THIS SECTION EXISTS. Two arms booked a model's legitimate no-interaction answer as a
  // refusal, and a refusal's disposal KILLS the track and deposits its energy where it stands:
  //
  //   * Bertini: `G4CascadeInterface::NoInteraction` after 20 fruitless tries, which leaves the
  //     track alive and unchanged. The arm tested `!no_interaction` beside `refusal == kNone`.
  //   * FTFP: `G4VPartonStringModel::Scatter`'s 1,000-attempt fallback, which returns the
  //     primary. The arm's own comment said it was a final state and the code booked
  //     `kFtfpRefused`.
  //
  // What each arm must do instead is APPLY what the model built, and what the model built must
  // be the primary, whole: alive with `energy_change` equal to its kinetic energy and nothing
  // emitted, or killed with ONE secondary of its own species carrying the same energy. Anything
  // else fails, so the section checks the final state and not only the return value. Each case is
  // one Geant4 is known to answer that way - a projectile far outside the window QBBC gives the
  // model, or below the Coulomb barrier - and the section fails if none of them produces the
  // answer at all, which would make the rest of it vacuous.
  std::printf("== 8. a model's own no-interaction answer is applied, whole ==\n");
  {
    had::InteractionSlot<real_t>& slot = pool.slots[0];
    const preco::PrecoWorkspace pws = had::preco_workspace_of<real_t>(slot);
    struct Case {
      ParticleType t;
      real_t e;
      int z, a;
      const char* name;
      bool ftfp;  // else Bertini
      int n;
      /// Every call must come back as the primary, whole. True where the model's answer is
      /// known to be "nothing happened" every time: FTFP at 1 MeV is the entry contract's own
      /// documented `kPrimaryUnchanged` case, and a 1 MeV K+ is far below lead's Coulomb barrier,
      /// so `G4CascadeInterface` exhausts its 20 tries on every call (measured: 20 of 20). The
      /// 1 MeV proton on lead is here for contrast - it interacts - and asserts only that nothing
      /// applied loses the projectile.
      bool all_whole;
    };
    const Case cases[] = {
        {ParticleType::kProton, 1, 8, 16, "FTFP p 1 MeV on O16", true, 5, true},
        {ParticleType::kKaonPlus, 1, 82, 208, "Bertini K+ 1 MeV on Pb208", false, 20, true},
        {ParticleType::kProton, 1, 82, 208, "Bertini p 1 MeV on Pb208", false, 20, false},
    };
    int whole_primary_total = 0;
    for (const Case& c : cases) {
      hp::HadProjectile<real_t> proj;
      proj.pdg = pdg_code(c.t);
      proj.charge = particle_def<real_t>(c.t).charge;
      proj.mass = particle_def<real_t>(c.t).mass;
      proj.kin_energy = c.e;
      proj.baryon_number = had::baryon_number_of(c.t, 0);
      int applied = 0, whole = 0, refused = 0, broken = 0;
      for (int k = 0; k < c.n; ++k) {
        hp::HadNucleus tgt;
        tgt.z = c.z;
        tgt.a = c.a;
        Philox<real_t> rng(515151u + 104729u * static_cast<unsigned int>(k), 0u,
                           had::kInteractionRngPurpose);
        had::InteractionOutcome out;
        const bool ok = c.ftfp ? had::run_arm_ftfp<real_t>(proj, tgt, slot, pool.view, 0, rng, out)
                               : had::run_arm_bertini<real_t>(proj, tgt, slot, pws, lt,
                                                              pool.view.fermi, rng, out);
        if (!ok) {
          ++refused;
          continue;
        }
        ++applied;
        const auto& fs = slot.fs;
        const bool alive_unchanged = fs.status == hp::HadFinalStateStatus::kIsAlive
                                     && std::fabs(double(fs.energy_change) - double(c.e)) < 1e-9
                                     && fs.n_secondaries == 0;
        // Killed and RE-EMITTED: exactly one secondary of the projectile's own species with its
        // whole kinetic energy, and nothing else but the target nucleus itself at rest. That
        // second piece is Geant4's too, and it is not decoration: FTFP's 1,000-attempt fallback
        // hands the primary to `G4GeneratorPrecompoundInterface`, which returns the untouched
        // target as the residual - measured, a 1 MeV proton on O16 comes back as
        // {2212 at 1 MeV along +z, 1000080160 at 0 MeV}. Any other secondary, or any energy on
        // the nucleus, is not "the primary, whole".
        int same_species = 0, at_rest_target = 0, other_sec = 0;
        double same_species_t = 0;
        for (int j = 0; j < fs.n_secondaries; ++j) {
          const auto& q = fs.secondaries[j];
          if (q.pdg == proj.pdg && q.a == 0) {
            ++same_species;
            same_species_t = double(q.kin_energy);
          } else if (q.a == c.a && q.z == c.z && double(q.kin_energy) < 1e-9) {
            ++at_rest_target;
          } else {
            ++other_sec;
          }
        }
        const bool reemitted = fs.status == hp::HadFinalStateStatus::kStopAndKill
                               && same_species == 1 && other_sec == 0 && at_rest_target <= 1
                               && std::fabs(same_species_t - double(c.e)) < 1e-6;
        if (alive_unchanged || reemitted) {
          ++whole;
        } else if (fs.n_secondaries == 0 && fs.status == hp::HadFinalStateStatus::kStopAndKill) {
          ++broken;  // the primary deleted and nothing put in its place
        }
      }
      whole_primary_total += whole;
      std::printf("   %-26s applied %2d (primary whole %2d)   refused %2d   empty kill %d\n",
                  c.name, applied, whole, refused, broken);
      if (broken != 0) {
        fail(std::string(c.name) + ": " + std::to_string(broken) + " final states killed the "
             "primary and emitted nothing - an applied answer that loses the projectile");
      }
      if (c.all_whole && whole != c.n) {
        fail(std::string(c.name) + ": " + std::to_string(whole) + " of " + std::to_string(c.n)
             + " came back as the primary, whole (" + std::to_string(refused)
             + " booked as refusals) - the model's own answer here is 'nothing happened', and a "
               "refusal is the defect this section is for");
      }
    }
    if (whole_primary_total == 0) {
      fail("no case produced the model's own no-interaction answer, so section 8 tested nothing");
    }
  }

  // ============================================================================================
  // 9. The light-ion arm with P9e's Interact: the cascade runs, and the new flags map
  // ============================================================================================
  //
  // WHY THIS SECTION EXISTS. Until P9e every ion at or above 50 MeV per nucleon was
  // `kLightIonCascade` by construction, so nothing downstream of the arm had ever seen an ion
  // cascade's final state. Three things are asserted, each of which the arm could get wrong:
  //
  //   * an 840 MeV alpha (210 MeV/n) on O16 RUNS - the cascade arm, not a refusal - and what it
  //     applies conserves baryon number and nuclear charge;
  //   * `BlirRefusal::nucleus`, new with P9e, is a refusal: an alpha on an A = 300 target cannot
  //     be built in a slot of 256 nucleons, and an arm that tested only the pre-P9e flags would
  //     apply the empty result as a final state (V197's defect, one arm over);
  //   * the fusion arm's `no_fusion` - "abort!! happens for too low energy for nuclei to fuse" -
  //     is Geant4's answer and is APPLIED: a 1 MeV alpha on Pb208 must come back alive with its
  //     whole energy. THE GATE IS A MASS TEST, NOT A COULOMB BARRIER, and the first version of
  //     this case got that wrong: `G4BinaryLightIonReaction::ApplyYourself` fuses whenever
  //     `pCompound.m2() >= sqr(mFused)`, and alpha + Pb208 -> Po212 needs only its Q-value,
  //     8.95 MeV in the centre of mass (Po212's own alpha-decay energy). A 10 MeV alpha clears
  //     that and FUSES - measured, 10 of 10 - barrier or no barrier; 1 MeV does not.
  std::printf("== 9. the light-ion arm runs Interact, and its flags map ==\n");
  {
    had::InteractionSlot<real_t>& slot = pool.slots[0];
    const preco::PrecoWorkspace pws = had::preco_workspace_of<real_t>(slot);
    hp::HadProjectile<real_t> proj;
    proj.pdg = pdg_code(ParticleType::kAlpha);
    proj.charge = particle_def<real_t>(ParticleType::kAlpha).charge;
    proj.mass = particle_def<real_t>(ParticleType::kAlpha).mass;
    proj.baryon_number = 4;

    struct Case {
      real_t e;
      int z, a;
      const char* name;
      int n;
    };
    const Case cases[] = {
        {840, 8, 16, "alpha 840 MeV on O16", 20},
        {840, 120, 300, "alpha 840 MeV on A=300", 10},
        {1, 82, 208, "alpha 1 MeV on Pb208", 10},
    };
    for (const Case& c : cases) {
      proj.kin_energy = c.e;
      int ran = 0, refused = 0, balanced = 0, alive_whole = 0;
      had::HadronicRefusal last_refusal = had::HadronicRefusal::kNumHadronicRefusals;
      for (int k = 0; k < c.n; ++k) {
        hp::HadNucleus tgt;
        tgt.z = c.z;
        tgt.a = c.a;
        Philox<real_t> rng(777001u + 65537u * static_cast<unsigned int>(k), 0u,
                           had::kInteractionRngPurpose);
        had::InteractionOutcome out;
        const bool ok = had::run_arm_light_ion<real_t>(proj, tgt, slot, pool.view, pws, lt,
                                                       pool.view.fermi, rng, out);
        if (!ok) {
          ++refused;
          last_refusal = out.refusal;
          continue;
        }
        ++ran;
        const auto& fs = slot.fs;
        if (fs.status == hp::HadFinalStateStatus::kIsAlive && fs.n_secondaries == 0
            && std::fabs(double(fs.energy_change) - double(c.e)) < 1e-9) {
          ++alive_whole;
        }
        int b = 0, q = 0;
        for (int j = 0; j < fs.n_secondaries; ++j) {
          const auto& s2 = fs.secondaries[j];
          if (s2.a > 0) {
            b += s2.a;
            q += s2.z;
          } else if (s2.pdg != 11) {  // the atomic conversion electron: see CountingEmitter
            const ParticleType t = particle_type_of_pdg(s2.pdg);
            b += had::baryon_number_of(t, 0);
            q += static_cast<int>(std::lrint(particle_def<real_t>(t).charge));
          }
        }
        if (fs.status == hp::HadFinalStateStatus::kIsAlive) {
          b += 4;
          q += 2;
        }
        // "Nothing happened" leaves the target where it was and out of the final state, so it
        // balances by construction; everything else must carry the target's A and Z out.
        const bool untouched = fs.status == hp::HadFinalStateStatus::kIsAlive
                               && fs.n_secondaries == 0;
        if (untouched || (b == 4 + c.a && q == 2 + c.z)) { ++balanced; }
      }
      std::printf("   %-24s ran %2d (balanced %2d, alive whole %2d)   refused %2d\n", c.name, ran,
                  balanced, alive_whole, refused);
      if (c.a == 16) {
        if (ran < c.n - 2) {
          fail(std::string(c.name) + ": only " + std::to_string(ran) + " of "
               + std::to_string(c.n) + " ran - the cascade arm should run all but the rare "
                 "refusal P9e's campaign measured");
        }
        if (balanced != ran) {
          fail(std::string(c.name) + ": " + std::to_string(ran - balanced) + " applied final "
               "states do not balance baryon number and nuclear charge");
        }
      } else if (c.a == 300) {
        if (ran != 0) {
          fail(std::string(c.name) + ": " + std::to_string(ran) + " were APPLIED - a nucleus "
               "that could not be built is a refusal (BlirRefusal::nucleus)");
        }
        if (refused != c.n || last_refusal != had::HadronicRefusal::kBinaryRefused) {
          fail(std::string(c.name) + ": expected " + std::to_string(c.n) + " kBinaryRefused");
        }
      } else {
        if (alive_whole != c.n) {
          fail(std::string(c.name) + ": " + std::to_string(alive_whole) + " of "
               + std::to_string(c.n) + " came back alive and whole - no_fusion is Geant4's "
                 "answer, not a refusal");
        }
      }
    }
  }

  std::printf("\n%s (%d failures)\n", g_fails == 0 ? "PASSED" : "FAILED", g_fails);
  return (g_fails == 0) ? 0 : 1;
}
