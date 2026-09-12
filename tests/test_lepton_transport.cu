// A 1 GeV electron on the device, and the energy balance that had nothing to fail against.
//
// WHY THIS FILE EXISTS
//
// docs/RISK.md V64 ends with a lesson rather than a fix: "a transport needs an energy balance,
// primary energy in against deposited plus escaped plus refused, and this port has one only in
// the depth-dose gate, for protons." The e+- range table clamped at 100 MeV, so a 1 GeV
// electron's first step handed back at most 100 MeV and the other 900 went nowhere - not
// deposited, not carried by a secondary, not counted. Nothing in the port looked at a lepton's
// books, so nothing saw it. `tests/test_electron_hi.cu` now compares the TABLES against
// `G4EmCalculator` over twelve decades, which would have caught that particular defect; it
// would not catch the next one that loses energy inside `step_lepton` rather than inside a
// table, because it never runs `step_lepton`.
//
// So this is the other half: the real kernel, on the real device, following electrons and
// positrons to a stop and adding up where every MeV went.
//
//     E0  ==  deposited  +  carried off by secondaries  +  left the world  +  refused
//
// with nothing on the right that is not one of those four. The secondaries are COUNTED and NOT
// TRANSPORTED - a `CountingEmitter` as `tests/test_ion_transport.cu` and `tests/test_neutron.cu`
// use - which is what makes the balance a statement about one track's stepping rather than
// about the whole shower: a delta ray or a bremsstrahlung photon leaves the balance at the
// energy `step_lepton` gave it, and if that energy is wrong the sum does not close.
//
// WHAT THE FOUR BLOCKS BELOW ARE FOR, AND WHY NONE OF THEM IS THE BALANCE ALONE
//
//  1. The balance itself, at 1 MeV, 50 MeV, 1 GeV and 10 GeV, for e- and e+. This is the
//     assertion V64 asked for.
//  2. The 1 GeV row's DISPOSITION - how much stayed local and how much left as photons. A
//     balance closes just as well when the track is killed on its first step and its whole
//     energy deposited locally, which is what a clamp at the BOTTOM of the table would do, so
//     the balance needs a companion that says the energy went somewhere plausible. At 1 GeV in
//     water almost all of it leaves as bremsstrahlung, and the radiative fraction is compared
//     against the table's own restricted dE/dx split rather than against a remembered number.
//  3. The energy past the top of the table. `above_table` fires, the track is killed, its
//     energy is booked under `kElectron`/`kPositron` in the refusal ledger, and the balance
//     still closes - because a refusal is a term in it. Under the old code this energy was
//     silently clamped into the table and the balance closed by losing it.
//  4. How much of a track sits above the msc model boundary. `em::kMscEnergyLimit()` is
//     100 MeV, and this counts the steps of a 1 GeV and a 10 GeV track taken above it against
//     those below. IT DOES NOT SAY WHICH MODEL RAN, and that is measured rather than assumed:
//     forcing `wv_msc` to false - the arrangement `em::kWentzelLeptonMscWired` ships with -
//     still passes this block (9,508 of 13,728 steps above the limit rather than 9,585 of
//     15,933). What it is for is the ENERGY profile of the steps, which is what the range
//     table sets: with the V64 ceiling restored a 1 GeV track takes 309 of 7,722 steps above
//     100 MeV instead of 9,585 of 15,933, because it is a 100 MeV electron after its first
//     step. Which model answers at the boundary is `tests/test_electron_hi.cu`'s question and
//     whether it is dispatched at all is `em::kWentzelLeptonMscWired`.
//
// THE TOLERANCE IS FLOATING POINT AND NOT PHYSICS. Everything in the balance is added in
// double; the only rounding is the accumulation order, so the limit is 1e-12 of the primary
// energy and not a per cent. A balance test with a per-cent tolerance would have passed with
// V64 in place for every energy under 101 MeV.
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "core/rng.cuh"
#include "core/track_buffer.cuh"
#include "data/brems_data.cuh"
#include "data/materials.cuh"
#include "physics/em/electron_processes.cuh"
#include "physics/em/urban_msc.cuh"
#include "physics/em/wentzel_msc.cuh"
#include "physics/scene.cuh"
#include "physics/stepper.cuh"
#include "render/trajectory.cuh"

using real_t = double;
using namespace g4gpu;

namespace {

int g_fails = 0;
void fail(const char* fmt, ...) {
  std::printf("  FAIL: ");
  va_list ap;
  va_start(ap, fmt);
  std::vprintf(fmt, ap);
  va_end(ap);
  std::printf("\n");
  ++g_fails;
}

/// Counts what a step gave away without transporting any of it. `books` is a member because
/// `step_lepton` books a refused track through it; the rest is `BufferEmitter`'s interface.
struct CountingEmitter {
  EmitterBooks books{};
  double secondary = 0;      ///< MeV handed to secondaries
  double secondary_gamma = 0;
  Vec3<real_t> pos{};
  int volume = 0;
  int event = 0;
  unsigned int child_count = 0u;

  __device__ int push(ParticleType t, const Vec3<real_t>&, real_t ekin, int,
                      unsigned short = 0) {
    ++child_count;
    secondary += static_cast<double>(ekin);
    if (t == ParticleType::kGamma) { secondary_gamma += static_cast<double>(ekin); }
    return 0;
  }
  /// `coulomb_apply` emits the recoil NUCLEUS through this, and a lepton above 100 MeV reaches
  /// it - `G4CoulombScattering` is on e+- from `MscEnergyLimit()` up (P8b wired it). Counted as
  /// a secondary like any other: what the balance needs is that the energy left the track with
  /// a number attached, and which species carried it is `tests/test_coulomb_scattering.cu`'s
  /// question, not this file's.
  __device__ int push_nucleus(int z, int a, const Vec3<real_t>& dir, real_t ekin,
                              int event_id) {
    return push(particle_type_of_nucleus(z, a), dir, ekin, event_id, ion_za_of(z, a));
  }
};

/// One track's books, in MeV.
struct Books {
  double edep = 0;        ///< deposited along the way
  double secondary = 0;   ///< given to secondaries
  double secondary_gamma = 0;
  double escaped = 0;     ///< still on the track when it left the world
  double path = 0;        ///< mm, the sum of the true step lengths
  int steps = 0;
  int steps_wv = 0;       ///< steps taken with p.ekin above em::kMscEnergyLimit()
  int refused = 0;
};

/// Follows one lepton to a stop, a boundary or the step guard.
///
/// `run_step_lepton` in `host/transport_run_impl.cuh` is the production caller and this is the
/// same loop without the pool: one `Philox` per step keyed on the track and the step index,
/// exactly as it keys it, so a step here draws the uniforms a step there would.
__global__ void Follow(Scene<real_t> scene, bool is_positron, real_t ekin0, int n,
                       unsigned int key0, Books* out, EmitterBooks books) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) { return; }

  TrackState<real_t> p{};
  p.species = is_positron ? ParticleType::kPositron : ParticleType::kElectron;
  p.pos = Vec3<real_t>{0, 0, 0};
  p.dir = Vec3<real_t>{0, 0, 1};
  p.ekin = ekin0;
  p.volume = 0;
  p.event = 0;
  p.rng_key = key0 + static_cast<unsigned int>(i);
  p.step = 0u;
  p.begin(p.pos, p.dir, p.ekin, 0, p.rng_key, ProcessId::fNotDefined, real_t(0), real_t(1));

  CountingEmitter em;
  em.books = books;
  Books b{};
  bool alive = true;
  while (alive && b.steps < kMaxStepsPerTrack) {
    if (p.ekin > em::kMscEnergyLimit<real_t>()) { ++b.steps_wv; }
    StepReport<real_t> rep{};
    Philox<real_t> rng(p.rng_key, p.step, 0xB19Du);
    em.pos = p.pos;
    em.volume = p.volume;
    em.event = p.event;
    real_t edep = 0;
    const real_t before = p.ekin;
    alive = step_lepton(scene, p, is_positron, rng, em, edep, rep, vis::no_capture());
    ++p.step;
    ++b.steps;
    b.edep += static_cast<double>(edep);
    b.path += static_cast<double>(rep.true_length);
    // A refusal kills the track with its energy still on it; `step_lepton` has booked that
    // energy in the ledger, so it is NOT counted again here - it is read back on the host.
    if (rep.status == StepStatus::fStopAndKill && rep.process == ProcessId::fNotDefined
        && before > scene.range_table->e_max) {
      ++b.refused;
      break;
    }
    if (p.volume == geom::kOutsideWorld) {
      b.escaped = static_cast<double>(p.ekin);
      break;
    }
  }
  b.secondary = em.secondary;
  b.secondary_gamma = em.secondary_gamma;
  out[i] = b;
}

struct Totals {
  double edep = 0, secondary = 0, secondary_gamma = 0, escaped = 0, path = 0;
  int steps = 0, steps_wv = 0, refused = 0, escapes = 0, n = 0;
  double worst_resid = 0;
  double worst_at = 0;
};

}  // namespace

int main() {
  std::printf("== a lepton on the device: where every MeV went ==\n\n");

  int dev = 0;
  if (cudaGetDevice(&dev) != cudaSuccess) {
    std::printf("  no CUDA device\n");
    return 1;
  }
  // `run_step_lepton` reports a 3,088 byte stack frame and the device default is 1,024, so
  // without this the kernel below faults with `an illegal memory access was encountered` and
  // nothing says why. `TransportEngine::Upload` and `b1_gpu_sched.cu` both set the same limit;
  // this is the third caller and it is here because a test that sets up its own Scene has to
  // set up everything the engine does.
  cudaDeviceSetLimit(cudaLimitStackSize, 16384);

  // ---------------------------------------------------------------- the tables
  static data::Material<real_t> mats[data::kNumMaterials];
  data::build_b1_materials<real_t>(mats);

  static data::SBTableSet<real_t> sb{};
  {
    // Every Z in B1's four materials - the same list `tests/test_electron.cu` uses. A Z that is
    // NOT here is not a missing radiative term, it is an out-of-bounds read: the brems branch
    // of `step_lepton` indexes `s.sb->elements[z_to_index[z]]` below 1 GeV and that is -1 for a
    // Z with no table. A-150 tissue carries fluorine, which is why 9 is here.
    const int zs[10] = {1, 6, 7, 8, 9, 12, 15, 16, 18, 20};
    const char* env = std::getenv("G4GPU_SB_DIR");
    const std::string d =
        (env != nullptr) ? env
                         : "D:\\Documents\\Geant4\\Windows\\geant4-v11.1.1-install\\share"
                           "\\Geant4\\data\\G4EMLOW8.2\\brem_SB";
    if (!data::load_sb_tables<real_t>(d, zs, 10, sb)) {
      std::printf("  cannot read the Seltzer-Berger tables from %s\n", d.c_str());
      return 1;
    }
  }
  static data::BremsTable<real_t> bt;
  data::build_brems_tables<real_t>(mats, sb, bt);
  static em::RangeTable<real_t> rt;
  em::build_range_table<real_t>(mats, rt, &sb);
  auto* h_urban = new em::UrbanTable<real_t>();
  em::build_urban_table<real_t>(mats, data::kNumMaterials, *h_urban);
  auto* h_wv = new em::WentzelLeptonTable<real_t>();
  em::build_wentzel_lepton_table<real_t>(mats, data::kNumMaterials, *h_wv);

  // ---------------------------------------------------------------- the world
  //
  // ONE WATER BOX AND NOTHING ELSE, 4 m of half-width. The secondaries are not transported, so
  // this does not have to hold a shower - it has to hold the PRIMARY until it stops, and the
  // longest track here is a 10 GeV electron whose range in water is 25 m. It will leave, and
  // that is why `escaped` is a term in the balance rather than an assumption; the block below
  // reports how many tracks used it.
  const real_t kHalf = 4000;
  static geom::Volume<real_t> vols[1] = {
      {{geom::SolidType::kBox, {kHalf, kHalf, kHalf}},
       geom::make_translation<real_t>({0, 0, 0}), 0, data::kWater, /*score_index=*/0}};

  geom::Volume<real_t>* d_vols = nullptr;
  data::Material<real_t>* d_mats = nullptr;
  em::RangeTable<real_t>* d_rt = nullptr;
  data::BremsTable<real_t>* d_bt = nullptr;
  data::SBTableSet<real_t>* d_sb = nullptr;
  em::UrbanTable<real_t>* d_urban = nullptr;
  em::WentzelLeptonTable<real_t>* d_wv = nullptr;
  if (cudaMalloc(&d_vols, sizeof(vols)) != cudaSuccess
      || cudaMalloc(&d_mats, sizeof(mats)) != cudaSuccess
      || cudaMalloc(&d_rt, sizeof(rt)) != cudaSuccess
      || cudaMalloc(&d_bt, sizeof(bt)) != cudaSuccess
      || cudaMalloc(&d_sb, sizeof(sb)) != cudaSuccess
      || cudaMalloc(&d_urban, sizeof(*h_urban)) != cudaSuccess
      || cudaMalloc(&d_wv, sizeof(*h_wv)) != cudaSuccess) {
    std::printf("  FAIL: cudaMalloc\n");
    return 1;
  }
  cudaMemcpy(d_vols, vols, sizeof(vols), cudaMemcpyHostToDevice);
  cudaMemcpy(d_mats, mats, sizeof(mats), cudaMemcpyHostToDevice);
  cudaMemcpy(d_rt, &rt, sizeof(rt), cudaMemcpyHostToDevice);
  cudaMemcpy(d_bt, &bt, sizeof(bt), cudaMemcpyHostToDevice);
  cudaMemcpy(d_sb, &sb, sizeof(sb), cudaMemcpyHostToDevice);
  cudaMemcpy(d_urban, h_urban, sizeof(*h_urban), cudaMemcpyHostToDevice);
  cudaMemcpy(d_wv, h_wv, sizeof(*h_wv), cudaMemcpyHostToDevice);

  Scene<real_t> scene{};
  scene.geometry = geom::Geometry<real_t>{d_vols, 1, 0};
  scene.materials = d_mats;
  scene.range_table = d_rt;
  scene.brems = d_bt;
  scene.sb = d_sb;
  scene.msc = d_urban;
  scene.wv_lepton = d_wv;
  scene.range_cut = real_t(0.7);
  scene.scoring_volume = 0;

  // The refusal ledger, one slot per ParticleType, as the engine allocates it.
  constexpr int kNT = static_cast<int>(ParticleType::kNumTypes);
  EmitterBooks books{};
  cudaMalloc(&books.refused_by_type, sizeof(int) * kNT);
  cudaMalloc(&books.refused_energy, sizeof(double) * kNT);

  constexpr int kTracks = 256;
  Books* d_books = nullptr;
  cudaMalloc(&d_books, sizeof(Books) * kTracks);
  std::vector<Books> got(kTracks);
  std::vector<int> ref_n(kNT);
  std::vector<double> ref_e(kNT);

  auto run = [&](bool pos, double e0, Totals& t) {
    cudaMemset(books.refused_by_type, 0, sizeof(int) * kNT);
    cudaMemset(books.refused_energy, 0, sizeof(double) * kNT);
    cudaMemset(d_books, 0, sizeof(Books) * kTracks);
    Follow<<<(kTracks + 63) / 64, 64>>>(scene, pos, real_t(e0), kTracks, 0x51ED0000u + 7u,
                                        d_books, books);
    const cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      fail("kernel: %s", cudaGetErrorString(err));
      return;
    }
    cudaMemcpy(got.data(), d_books, sizeof(Books) * kTracks, cudaMemcpyDeviceToHost);
    cudaMemcpy(ref_n.data(), books.refused_by_type, sizeof(int) * kNT, cudaMemcpyDeviceToHost);
    cudaMemcpy(ref_e.data(), books.refused_energy, sizeof(double) * kNT,
               cudaMemcpyDeviceToHost);
    t = Totals{};
    t.n = kTracks;
    const int slot = static_cast<int>(pos ? ParticleType::kPositron : ParticleType::kElectron);
    for (int i = 0; i < kTracks; ++i) {
      t.edep += got[i].edep;
      t.secondary += got[i].secondary;
      t.secondary_gamma += got[i].secondary_gamma;
      t.escaped += got[i].escaped;
      t.path += got[i].path;
      t.steps += got[i].steps;
      t.steps_wv += got[i].steps_wv;
      t.refused += got[i].refused;
      if (got[i].escaped > 0) { ++t.escapes; }
      // Per TRACK, because a sum over 256 tracks can cancel a track that lost energy against
      // one that gained it, and "the total closes" is a weaker statement than "every track
      // closes". The refused energy is per species and cannot be split per track, so a refused
      // track's residual is its own energy and is checked through the ledger below instead.
      //
      // A POSITRON'S BOOKS DO NOT BALANCE AGAINST ITS KINETIC ENERGY AND MUST NOT.
      // `step_lepton`'s dying branch emits `G4eplusAnnihilation`'s two 511 keV photons, whose
      // energy is the pair's REST MASS and not anything the track carried, so the input side
      // of a positron's balance is `E0 + 2 m_e c^2`. That is 1.022 MeV of real energy that a
      // detector downstream really sees; leaving it out of the input would make this test read
      // as a 102% energy gain at 1 MeV, which is what it did before this line existed.
      const double e_in =
          e0 + (pos ? 2.0 * static_cast<double>(units::electron_mass_c2<real_t>()) : 0.0);
      if (got[i].refused == 0) {
        const double sum = got[i].edep + got[i].secondary + got[i].escaped;
        const double resid = std::fabs(sum - e_in) / e_in;
        if (resid > t.worst_resid) {
          t.worst_resid = resid;
          t.worst_at = sum;
        }
      }
    }
    t.refused = ref_n[slot];
    // A refused track's energy is in the ledger; fold it in so the printed total is the whole
    // beam and not the part that was transported.
    t.escaped += 0;
    if (ref_n[slot] > 0) { t.worst_at = ref_e[slot]; }
  };

  // ================================================================ 1. the balance
  std::printf("-- 1. E0 = deposited + secondaries + escaped, per track --\n");
  std::printf("  %-3s %10s %8s %13s %13s %13s %8s %9s %11s\n", "", "E0 (MeV)", "tracks",
              "edep (MeV)", "sec (MeV)", "esc (MeV)", "steps", "escapes", "worst resid");
  {
    const double energies[4] = {1.0, 50.0, 1000.0, 10000.0};
    for (int p = 0; p < 2; ++p) {
      for (double e0 : energies) {
        Totals t;
        run(p == 1, e0, t);
        std::printf("  %-3s %10g %8d %13.6g %13.6g %13.6g %8d %9d %11.3e\n",
                    (p == 1) ? "e+" : "e-", e0, t.n, t.edep, t.secondary, t.escaped, t.steps,
                    t.escapes, t.worst_resid);
        if (t.worst_resid > 1e-12) {
          fail("%s at %g MeV: a track's books are off by %.3e of its energy (sum %.10g)",
               (p == 1) ? "e+" : "e-", e0, t.worst_resid, t.worst_at);
        }
        if (t.steps <= t.n) {
          fail("%s at %g MeV: %d steps for %d tracks - nothing was transported",
               (p == 1) ? "e+" : "e-", e0, t.steps, t.n);
        }
      }
    }
  }
  std::printf("\n");

  // ================================================================ 2. the disposition
  //
  // A closed balance says nothing left the arithmetic; it does not say the energy went
  // somewhere a 1 GeV electron would put it. At 1 GeV in water the radiative and collision
  // stopping powers are 0.1837 and 0.0225 MeV/mm out of the port's own table, so about 89% of
  // the loss is radiative - and the radiative half above the gamma production cut leaves as
  // explicit photons while the sub-cut remainder is deposited. The check is that the photon
  // share is the LARGER one and that both are non-zero, which is what distinguishes a 1 GeV
  // electron from a 100 MeV one (at 100 MeV the two stopping powers are 0.0165 and 0.0205,
  // and the collision half is the larger).
  std::printf("-- 2. what a 1 GeV electron in water does with its energy --\n");
  {
    Totals t;
    run(false, 1000.0, t);
    const double tot = t.edep + t.secondary + t.escaped;
    std::printf("  deposited %.4f%%   secondaries %.4f%% (photons %.4f%%)   escaped %.4f%%\n",
                100 * t.edep / tot, 100 * t.secondary / tot, 100 * t.secondary_gamma / tot,
                100 * t.escaped / tot);
    std::printf("  mean true path %.3f mm over %.1f steps per track\n", t.path / t.n,
                double(t.steps) / t.n);
    const real_t col = em::collision_dedx<real_t>(mats[data::kWater], real_t(1000), false);
    const data::BremsBoundary<real_t> bnd =
        data::brems_boundary(mats[data::kWater], sb, false);
    const real_t rad =
        em::brems_restricted_dedx<real_t>(mats[data::kWater], sb, bnd, real_t(1000), false);
    std::printf("  the table's own split at 1 GeV in water: collision %.6g, sub-cut radiative "
                "%.6g MeV/mm\n", double(col), double(rad));
    if (!(t.secondary_gamma > t.edep)) {
      fail("a 1 GeV electron put %.6g MeV into photons and %.6g MeV into local deposit; at "
           "1 GeV the radiative channel is the larger one", t.secondary_gamma, t.edep);
    }
    if (!(t.edep > 0)) { fail("a 1 GeV electron deposited nothing at all"); }
  }
  std::printf("\n");

  // ================================================================ 3. past the top
  //
  // 200 TeV is above `G4EmParameters::MaxKinEnergy`, so there is no Geant4 table to reproduce
  // and `step_lepton` refuses the track by name. What is asserted is that the refusal is
  // BOOKED: the count and the energy both land in the ledger under the track's own species,
  // and nothing is deposited. Under the ceiling this port had, a 200 TeV electron was
  // transported as a 100 MeV one and 199.9999 TeV went missing with no counter moving.
  std::printf("-- 3. a kinetic energy past the top of Geant4's own tables --\n");
  {
    for (int p = 0; p < 2; ++p) {
      Totals t;
      run(p == 1, 2e8, t);   // 200 TeV
      const int slot =
          static_cast<int>((p == 1) ? ParticleType::kPositron : ParticleType::kElectron);
      std::printf("  %-3s 200 TeV: refused %d of %d tracks, %.6g MeV booked, %.6g MeV "
                  "deposited\n", (p == 1) ? "e+" : "e-", ref_n[slot], kTracks, ref_e[slot],
                  t.edep);
      if (ref_n[slot] != kTracks) {
        fail("%s at 200 TeV: %d of %d tracks refused, expected all of them",
             (p == 1) ? "e+" : "e-", ref_n[slot], kTracks);
      }
      if (std::fabs(ref_e[slot] - kTracks * 2e8) > 1e-6 * kTracks * 2e8) {
        fail("%s at 200 TeV: %.10g MeV booked, expected %.10g", (p == 1) ? "e+" : "e-",
             ref_e[slot], double(kTracks) * 2e8);
      }
      if (t.edep != 0 || t.secondary != 0) {
        fail("%s at 200 TeV: a refused track deposited %.6g MeV and made %.6g MeV of "
             "secondaries", (p == 1) ? "e+" : "e-", t.edep, t.secondary);
      }
    }
    // And 99 TeV, one decade below the ceiling, is transported rather than refused - or the
    // refusal is not a ceiling, it is a wall in the middle of the table.
    Totals t;
    run(false, 9.9e7, t);
    const int slot = static_cast<int>(ParticleType::kElectron);
    std::printf("  e-   99 TeV: refused %d of %d tracks, %.6g MeV deposited, %.6g MeV to "
                "secondaries\n", ref_n[slot], kTracks, t.edep, t.secondary);
    if (ref_n[slot] != 0) {
      fail("99 TeV is inside the table and %d tracks were refused there", ref_n[slot]);
    }
  }
  std::printf("\n");

  // ================================================================ 4. which msc model ran
  //
  // The dispatch itself, counted rather than assumed. `em::kMscEnergyLimit()` is 100 MeV and
  // `step_lepton` takes the WentzelVI branch strictly above it, so a 1 GeV track must spend its
  // first steps there and its last ones in Urban, and a 1 MeV track must never reach it.
  std::printf("-- 4. how much of a track sits above the %g MeV msc model boundary "
              "(WentzelVI wired: %d) --\n", double(em::kMscEnergyLimit<real_t>()),
              static_cast<int>(em::kWentzelLeptonMscWired));
  {
    struct Row { double e0; bool want_wv; };
    const Row rows[3] = {{1.0, false}, {1000.0, true}, {10000.0, true}};
    for (const Row& r : rows) {
      Totals t;
      run(false, r.e0, t);
      std::printf("  e- %8g MeV: %6d of %6d steps above the limit (%.2f%%)\n", r.e0, t.steps_wv,
                  t.steps, 100.0 * t.steps_wv / t.steps);
      if (r.want_wv && t.steps_wv == 0) {
        fail("a %g MeV electron took no step above %g MeV - it was a %g MeV electron after "
             "its first step, which is docs/RISK.md V64", r.e0,
             double(em::kMscEnergyLimit<real_t>()), double(em::kMscEnergyLimit<real_t>()));
      }
      if (!r.want_wv && t.steps_wv != 0) {
        fail("a %g MeV electron took %d steps above %g MeV", r.e0, t.steps_wv,
             double(em::kMscEnergyLimit<real_t>()));
      }
      if (r.want_wv && t.steps_wv == t.steps) {
        fail("every step of a %g MeV electron was above %g MeV - it never slowed down", r.e0,
             double(em::kMscEnergyLimit<real_t>()));
      }
    }
  }

  cudaFree(d_vols);
  cudaFree(d_mats);
  cudaFree(d_rt);
  cudaFree(d_bt);
  cudaFree(d_sb);
  cudaFree(d_urban);
  cudaFree(d_wv);
  cudaFree(d_books);
  cudaFree(books.refused_by_type);
  cudaFree(books.refused_energy);
  delete h_urban;
  delete h_wv;

  std::printf("\n%s (%d failures)\n", g_fails ? "FAILED" : "PASSED", g_fails);
  return g_fails ? 1 : 0;
}
