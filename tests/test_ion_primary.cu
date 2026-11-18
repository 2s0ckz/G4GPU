// A nucleus seeded as a PRIMARY is the same track as the same nucleus emitted as a SECONDARY.
//
// WHY THIS FILE EXISTS (P30)
//
// Every real nuclide has been stepped since P8c, and every one of them arrived as a secondary:
// `BufferEmitter::push_nucleus` writes the nuclide into `TrackState::ion_za`, and from that one
// field `step_hadron` derives the mass (G4NucleiProperties), the charge, the spin and the scaled
// tables (`em::stepped_ion`), and `run_step_hadron` the velocity. P30 gives the gun a nuclide
// (`/gun/ion Z A`, tests/test_ion_gun.cu) and the seeding a place to put it
// (`Primary::ion_za` -> `seed_from_primaries` -> `seed_track_slot`). The claim this file holds
// the port to is that nothing else is needed: a C12 the gun fires is, to the bit, the C12 a
// fragmentation would have made.
//
// WHAT IS CHECKED, for C12 at 12 GeV (1 GeV per nucleon) in G4_WATER, G4_BONE_COMPACT_ICRU,
// G4_AIR and G4_Al - each a 400 mm box, 150 tracks from its centre along +z:
//
//  1. THE SEEDED SLOT. `host::seed_from_primaries` - the engine's own seeding kernel - against
//     `BufferEmitter::push_nucleus` for the same (Z, A, E, position, direction): every field a
//     step reads (species, ion_za, position, direction, energy, volume, step, the msc and
//     neutron-general state, status, weight) is EQUAL. The fields that differ are the lineage
//     Geant4 also gives a primary and a secondary differently - the RNG key, the parent, the
//     creator process, the first-step flag - and are printed, not compared.
//  2. THE FIRST STEP, EXACT. Both tracks stepped by `step_hadron` on the device with the
//     PRIMARY's key: every field of the outcome identical to the bit - the loss, the deposit,
//     the true path, the process that won, the secondaries. Plus the range (`range_for`), the
//     dE/dx (`dedx_for`), the ionInelastic and ionElastic cross sections and the track's mass,
//     each computed from the slot's own `ion_za`, identical to the bit.
//  3. HOST AGAINST DEVICE for the seeded primary, in the shape of tests/test_step_hadron.cu:
//     discrete fields exact, numeric ones to its 1e-11 (nvcc's FMA contraction, which that
//     file measures).
//  4. THE LEDGER. Every `had::HadronicRefusal` row is zero after the primaries' steps - and,
//     so that the zero means something, a CONTROL seeds the same primaries with `ion_za = 0`
//     (what `seed_track_slot` wrote before P30) and the `kIonWithoutNuclide` row counts every
//     one of them.
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "core/rng.cuh"
#include "g4/G4NistManager.hh"
#include "host/hadronic_upload.cuh"
#include "host/transport_run_impl.cuh"  // host::seed_from_primaries, the engine's seeding kernel
#include "physics/em/hadron_range.cuh"
#include "physics/scene.cuh"
#include "physics/stepper.cuh"

using real_t = double;
using namespace g4gpu;

namespace {

int g_fails = 0;
void fail(const char* what) {
  std::printf("  FAIL: %s\n", what);
  ++g_fails;
}

constexpr int kMaxSec = 8;

/// What one step produced. tests/test_step_hadron.cu's record.
struct Outcome {
  real_t ekin = 0, edep = 0, length = 0, non_ionizing = 0;
  real_t px = 0, py = 0, pz = 0;
  real_t dx = 0, dy = 0, dz = 0;
  int status = -1, process = -1, volume = -1;
  int alive = 0;
  int n_sec = 0;
  int sec_type[kMaxSec] = {};
  int sec_za[kMaxSec] = {};
  real_t sec_ekin[kMaxSec] = {};
  real_t sec_dx[kMaxSec] = {}, sec_dy[kMaxSec] = {}, sec_dz[kMaxSec] = {};
  /// What the step's physics sees of the nucleus, from the track's own `ion_za`.
  real_t range = 0, dedx = 0, inel_xs = 0, el_xs = 0, mass = 0;
};

struct RecordingEmitter {
  Outcome* out = nullptr;
  Vec3<real_t> pos{};
  int volume = 0;
  int event = 0;
  unsigned int child_count = 0u;
  int last_secondary = -1;

  __host__ __device__ int push(ParticleType type, const Vec3<real_t>& dir, real_t ekin, int,
                               unsigned short za = 0) {
    ++child_count;
    if (out->n_sec < kMaxSec) {
      const int i = out->n_sec++;
      out->sec_type[i] = static_cast<int>(type);
      out->sec_za[i] = za;
      out->sec_ekin[i] = ekin;
      out->sec_dx[i] = dir.x;
      out->sec_dy[i] = dir.y;
      out->sec_dz[i] = dir.z;
    }
    return 0;
  }
  __host__ __device__ int push_nucleus(int z, int a, const Vec3<real_t>& dir, real_t ekin,
                                       int event_id) {
    return push(particle_type_of_nucleus(z, a), dir, ekin, event_id, ion_za_of(z, a));
  }
};

/// One step of one track, the same source on both sides, plus what the step reads.
__host__ __device__ void one_step(const Scene<real_t>& scene, TrackState<real_t> p,
                                  had::HadronicWiring<real_t> had, Outcome* out) {
  // What `step_hadron` and `run_step_hadron` derive from the slot, before the step moves it.
  // Skipped for a slot with no nuclide (the control), which has no mass to read tables at.
  if (p.ion_za != 0) {
    const em::SteppedHadron<real_t> h =
        em::stepped_ion<real_t>(ion_z_of(p.ion_za), ion_a_of(p.ion_za));
    const int mat = geom::material_at(scene.geometry, p.volume, p.pos);
    const data::Material<real_t>& mm = scene.materials[mat];
    out->range = scene.hadron_range->range_for(mm, h, mat, p.ekin);
    out->dedx = scene.hadron_range->dedx_for(mm, h, mat, p.ekin);
    hadronic::xs::MaterialXs<real_t> mi{};
    out->inel_xs = had::inelastic_xs_per_volume<real_t>(had.inelastic, mm, p.species, p.ekin,
                                                        h.z, h.a, mi);
    hadronic::xs::MaterialXs<real_t> me{};
    out->el_xs = had::elastic_xs_per_volume<real_t>(had.elastic, mm, p.species, p.ekin, me,
                                                    h.z, h.a);
    out->mass = em::ion_particle_def<real_t>(ion_z_of(p.ion_za), ion_a_of(p.ion_za)).mass;
  }
  StepReport<real_t> rep;
  Philox<real_t> rng(p.rng_key, p.step, 0xB19Du);  // run_step_hadron's purpose value
  RecordingEmitter em{out, p.pos, p.volume, p.event, 0u, -1};
  real_t edep = 0;
  const bool alive = step_hadron(scene, p, ParticleType::kGenericIon, had, rng, em, edep, rep);
  out->ekin = p.ekin;
  out->edep = edep;
  out->length = rep.true_length;
  out->non_ionizing = rep.non_ionizing;
  out->px = p.pos.x;
  out->py = p.pos.y;
  out->pz = p.pos.z;
  out->dx = p.dir.x;
  out->dy = p.dir.y;
  out->dz = p.dir.z;
  out->status = static_cast<int>(rep.status);
  out->process = static_cast<int>(rep.process);
  out->volume = p.volume;
  out->alive = alive ? 1 : 0;
}

__global__ void RunHadron(Scene<real_t> scene, const TrackState<real_t>* in, int n,
                          had::HadronicWiring<real_t> had, Outcome* out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) { return; }
  out[i] = Outcome{};
  one_step(scene, in[i], had, &out[i]);
}

/// The same nucleus as a fragment would make it: `BufferEmitter::push_nucleus`, the only thing
/// that put a nuclide on a track before P30.
__global__ void EmitAsSecondary(TrackBuffer<real_t> out, Vec3<real_t> pos, int volume,
                                Vec3<real_t> dir, real_t ekin, int z, int a, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) { return; }
  BufferEmitter<real_t> em{out, pos, volume, i,
                           0x5EC0u + static_cast<unsigned int>(i), 3u, 0u,
                           real_t(0), real_t(1), real_t(0),
                           SecondaryArena{}, -1, nullptr, EmitterBooks{}};
  em.push_nucleus(z, a, dir, ekin, i);
}

/// Every slot of @p buf back as a TrackState, indexed by its event.
__global__ void Gather(TrackBuffer<real_t> buf, int n, TrackState<real_t>* out) {
  const int s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= n) { return; }
  TrackState<real_t> t{};
  buf.load(s, t);
  if (t.event >= 0 && t.event < n) { out[t.event] = t; }
}

constexpr double kNumTol = 1e-11;  // tests/test_step_hadron.cu's, and its reason

/// @param tol 0 for device against device, kNumTol for host against device.
int compare(const Outcome& h, const Outcome& d, const char* tag, int idx, double tol,
            double* worst, int* printed) {
  int bad = 0;
  auto chk = [&](real_t a, real_t b, const char* field) {
    if (a == b) { return; }
    const double rel = (b != 0.0) ? std::fabs(double(a - b) / double(b))
                                  : std::fabs(double(a - b));
    if (rel > *worst) { *worst = rel; }
    if (rel <= tol) { return; }
    if (*printed < 12) {
      std::printf("  FAIL: %s track %d: %s %.17g vs %.17g (rel %.3e > %.0e)\n", tag, idx,
                  field, double(a), double(b), rel, tol);
      ++(*printed);
    }
    ++bad;
  };
  chk(h.ekin, d.ekin, "ekin");
  chk(h.edep, d.edep, "edep");
  chk(h.length, d.length, "true_length");
  // THE RECOIL'S ENERGY HAS AN ABSOLUTE GATE AS WELL, for docs/RISK.md V270's reason: an
  // ionElastic recoil of 1e-8 MeV is `lv.e() - mass2` on a total energy of tens of GeV, so its
  // last bits are the projectile's and the target's, and nvcc's FMA moves a few ulps of THAT -
  // measured here 3.6e-12 MeV on an Al27 recoil of 1.4e-8 (rel 2.7e-4), the cancellation V270
  // met on O16. 16 ulps of (T + m_C12 + m_Ca40), the heaviest target in the four materials.
  const double kRecoilAbs =
      (tol > 0) ? 16 * 2.220446049250313e-16
                      * (h.ekin + data::get_nuclear_mass(12, 6) + data::get_nuclear_mass(40, 20))
                : 0.0;
  auto chk_recoil = [&](real_t a, real_t b, const char* field) {
    if (std::fabs(double(a - b)) <= kRecoilAbs) { return; }
    chk(a, b, field);
  };
  chk_recoil(h.non_ionizing, d.non_ionizing, "non_ionizing");
  chk(h.px, d.px, "pos.x");
  chk(h.py, d.py, "pos.y");
  chk(h.pz, d.pz, "pos.z");
  chk(h.dx, d.dx, "dir.x");
  chk(h.dy, d.dy, "dir.y");
  chk(h.dz, d.dz, "dir.z");
  chk(h.range, d.range, "range");
  chk(h.dedx, d.dedx, "dedx");
  chk(h.inel_xs, d.inel_xs, "ionInelastic xs");
  chk(h.el_xs, d.el_xs, "ionElastic xs");
  chk(h.mass, d.mass, "mass");
  if (h.status != d.status || h.process != d.process || h.alive != d.alive
      || h.volume != d.volume || h.n_sec != d.n_sec) {
    std::printf("  FAIL: %s track %d: status/process/alive/volume/n_sec %d/%d/%d/%d/%d vs "
                "%d/%d/%d/%d/%d\n", tag, idx, h.status, h.process, h.alive, h.volume, h.n_sec,
                d.status, d.process, d.alive, d.volume, d.n_sec);
    ++bad;
  } else {
    for (int j = 0; j < h.n_sec && j < kMaxSec; ++j) {
      if (h.sec_type[j] != d.sec_type[j] || h.sec_za[j] != d.sec_za[j]) {
        std::printf("  FAIL: %s track %d secondary %d: species %d (za %d) vs %d (%d)\n", tag,
                    idx, j, h.sec_type[j], h.sec_za[j], d.sec_type[j], d.sec_za[j]);
        ++bad;
      }
      chk_recoil(h.sec_ekin[j], d.sec_ekin[j], "secondary ekin");
      chk(h.sec_dx[j], d.sec_dx[j], "secondary dir.x");
      chk(h.sec_dy[j], d.sec_dy[j], "secondary dir.y");
      chk(h.sec_dz[j], d.sec_dz[j], "secondary dir.z");
    }
  }
  return bad;
}

/// The fields a step reads, equal between the seeded primary and the emitted secondary.
int compare_slots(const TrackState<real_t>& p, const TrackState<real_t>& s, int idx) {
  int bad = 0;
  auto eq = [&](bool ok, const char* field) {
    if (ok) { return; }
    static int printed = 0;
    if (bad == 0 && printed < 4) {
      std::printf("  FAIL: track %d: the seeded slot's %s differs\n", idx, field);
      ++printed;
    }
    ++bad;
  };
  eq(p.species == s.species, "species");
  eq(p.ion_za == s.ion_za, "ion_za");
  eq(p.pos.x == s.pos.x && p.pos.y == s.pos.y && p.pos.z == s.pos.z, "position");
  eq(p.dir.x == s.dir.x && p.dir.y == s.dir.y && p.dir.z == s.dir.z, "direction");
  eq(p.ekin == s.ekin, "kinetic energy");
  eq(p.volume == s.volume, "volume");
  eq(p.event == s.event, "event");
  eq(p.step == s.step, "step count");
  eq(p.msc_tlimit == s.msc_tlimit && p.msc_tlimitmin == s.msc_tlimitmin, "msc state");
  eq(p.ngp_left == s.ngp_left && p.ngp_prev_step == s.ngp_prev_step, "neutron-general state");
  eq(p.status == s.status, "status");
  eq(p.weight == s.weight, "weight");
  return bad;
}

}  // namespace

int main() {
  const cudaError_t lim = cudaDeviceSetLimit(cudaLimitStackSize, 16384);
  if (lim != cudaSuccess) {
    std::printf("FATAL: cudaDeviceSetLimit: %s\n", cudaGetErrorString(lim));
    return 2;
  }
  std::printf("== a C12 primary, seeded, against the same C12 emitted as a secondary ==\n\n");

  // ---------------------------------------------------------------- the materials
  //
  // Through G4NistManager and G4Material::Build, the path every geometry's materials take.
  const char* const kMatNames[] = {"G4_WATER", "G4_BONE_COMPACT_ICRU", "G4_AIR", "G4_Al"};
  constexpr int kNMat = 4;
  static data::MaterialTable<real_t> table{};
  for (const char* nm : kMatNames) {
    const int idx = G4NistManager::Instance()->FindOrBuildMaterial(nm)->Build(table, 0.7);
    if (idx < 0) {
      std::printf("FATAL: %s did not build\n", nm);
      return 2;
    }
  }
  data::Material<real_t>* mats = table.m;
  data::Material<real_t>* d_mats = nullptr;
  cudaMalloc(&d_mats, sizeof(data::Material<real_t>) * kNMat);
  cudaMemcpy(d_mats, mats, sizeof(data::Material<real_t>) * kNMat, cudaMemcpyHostToDevice);

  auto* h_hrt = new em::HadronRangeTable<real_t>();
  {
    std::vector<real_t> cuts(kNMat);
    for (int i = 0; i < kNMat; ++i) { cuts[i] = mats[i].cut_electron; }
    static em::ShellTables<real_t> shell;
    em::build_shell_tables(shell);
    em::build_hadron_range_table<real_t>(mats, cuts.data(), *h_hrt, &shell, kNMat);
  }
  em::HadronRangeTable<real_t>* d_hrt = nullptr;
  cudaMalloc(&d_hrt, sizeof(*h_hrt));
  cudaMemcpy(d_hrt, h_hrt, sizeof(*h_hrt), cudaMemcpyHostToDevice);

  // hadElastic's tables (ionElastic reads none of them; the pion and nucleon rows are there
  // because `elastic_xs_per_volume` takes the whole view) for every element of the four, and
  // their host twin built the same way - tests/test_step_hadron.cu's block.
  const std::vector<int> zs = {1, 6, 7, 8, 12, 13, 15, 16, 18, 20};
  auto owner = host::upload_elastic_tables<real_t>(zs, /*verbose=*/false);
  host::ElasticTableOwner<real_t> host_owner;
  {
    namespace el = g4gpu::physics::hadronic::elastic;
    namespace hxs = g4gpu::hadronic::xs;
    auto* bn = new hxs::BggNucleonTable<real_t>();
    hxs::bgg_build_nucleon_table<real_t>(true, *bn, true);
    auto* bp = new hxs::BggPionTable<real_t>();
    hxs::bgg_build_pion_table<real_t>(true, *bp);
    auto* gr = new el::HeEnergyGrid<real_t>();
    auto* bd = new el::HeBoundary<real_t>();
    int max_z = 1;
    for (int z : zs) {
      if (z > max_z) { max_z = z; }
    }
    auto* slot = new std::vector<short>(static_cast<std::size_t>(max_z) + 1, -1);
    auto* tabs = new std::vector<el::HeElasticData<real_t>>();
    int n = 0;
    for (int z : zs) {
      if ((*slot)[z] >= 0) { continue; }
      (*slot)[z] = static_cast<short>(n++);
    }
    tabs->resize(static_cast<std::size_t>(n) * had::kHeNumHadrons);
    for (int z = 1; z <= max_z; ++z) {
      if ((*slot)[z] < 0) { continue; }
      const int a = static_cast<int>(data::atomic_mass<real_t>(z) + real_t(0.5));
      const real_t mass_a = data::nuclear_mass<real_t>(a, z);
      for (int h = 0; h < had::kHeNumHadrons; ++h) {
        const real_t hm = particle_def<real_t>(h == 0 ? ParticleType::kPionPlus
                                                      : ParticleType::kPionMinus).mass;
        (*tabs)[static_cast<std::size_t>((*slot)[z]) * had::kHeNumHadrons + h] =
            el::he_fill_data<real_t>(z, a, mass_a, hm, el::he_hadron_type()[h],
                                     el::he_hadron_type1()[h], el::he_hadron_code()[h], false,
                                     *gr, *bd);
      }
    }
    host_owner.view.bgg_nucleon = bn;
    host_owner.view.bgg_pion = bp;
    host_owner.view.he = tabs->data();
    host_owner.view.he_slot_of_z = slot->data();
    host_owner.view.he_max_z = max_z;
    host_owner.view.he_grid = gr;
    host_owner.view.he_bnd = bd;
  }

  const int kNRef = static_cast<int>(had::HadronicRefusal::kNumHadronicRefusals);
  std::vector<int> h_ref_n(kNRef, 0);
  std::vector<double> h_ref_e(kNRef, 0.0);
  had::HadronicWiring<real_t> hhad{};
  hhad.elastic = host_owner.view;
  hhad.books.count = h_ref_n.data();
  hhad.books.energy = h_ref_e.data();
  int* d_ref_n = nullptr;
  double* d_ref_e = nullptr;
  cudaMalloc(&d_ref_n, sizeof(int) * kNRef);
  cudaMalloc(&d_ref_e, sizeof(double) * kNRef);
  cudaMemset(d_ref_n, 0, sizeof(int) * kNRef);
  cudaMemset(d_ref_e, 0, sizeof(double) * kNRef);
  had::HadronicWiring<real_t> dhad{};
  dhad.elastic = owner.view;
  dhad.books.count = d_ref_n;
  dhad.books.energy = d_ref_e;

  // ---------------------------------------------------------------- the beam
  constexpr int kZ = 6, kA = 12;
  const real_t kEkin = 12000;  // MeV: /gun/energy is the TOTAL kinetic energy, 1 GeV/n
  constexpr int kPer = 150;
  const Vec3<real_t> kPos{0, 0, 0}, kDir{0, 0, 1};
  const unsigned int kSeed = 0xF00Du;

  TrackBuffer<real_t> pool_p{}, pool_s{};
  if (allocate_track_buffer(pool_p, kPer) != cudaSuccess
      || allocate_track_buffer(pool_s, kPer) != cudaSuccess) {
    std::printf("FATAL: track buffers\n");
    return 2;
  }
  Primary<real_t>* d_prim = nullptr;
  cudaMalloc(&d_prim, sizeof(Primary<real_t>) * kPer);
  TrackState<real_t>* d_ts = nullptr;
  cudaMalloc(&d_ts, sizeof(TrackState<real_t>) * kPer * 2);
  Outcome* d_out = nullptr;
  cudaMalloc(&d_out, sizeof(Outcome) * kPer * 2);

  /// Seeds @p prims through the engine's kernel and reads the slots back by event.
  auto seed = [&](const std::vector<Primary<real_t>>& prims, const geom::Geometry<real_t>& g,
                  std::vector<TrackState<real_t>>& out) {
    cudaMemset(pool_p.count, 0, sizeof(int));
    cudaMemset(pool_p.overflow, 0, sizeof(int));
    cudaMemcpy(d_prim, prims.data(), sizeof(Primary<real_t>) * kPer, cudaMemcpyHostToDevice);
    host::seed_from_primaries<real_t><<<(kPer + 63) / 64, 64>>>(pool_p, g, d_prim, kPer, 0u,
                                                                kSeed);
    Gather<<<(kPer + 63) / 64, 64>>>(pool_p, kPer, d_ts);
    out.assign(kPer, TrackState<real_t>{});
    cudaMemcpy(out.data(), d_ts, sizeof(TrackState<real_t>) * kPer, cudaMemcpyDeviceToHost);
  };
  auto run = [&](const Scene<real_t>& sc, const std::vector<TrackState<real_t>>& in,
                 std::vector<Outcome>& out) {
    cudaMemcpy(d_ts, in.data(), sizeof(TrackState<real_t>) * in.size(), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, sizeof(Outcome) * in.size());
    RunHadron<<<(static_cast<int>(in.size()) + 63) / 64, 64>>>(
        sc, d_ts, static_cast<int>(in.size()), dhad, d_out);
    const cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
      std::printf("FATAL: RunHadron: %s\n", cudaGetErrorString(e));
      std::exit(2);
    }
    out.assign(in.size(), Outcome{});
    cudaMemcpy(out.data(), d_out, sizeof(Outcome) * in.size(), cudaMemcpyDeviceToHost);
  };

  std::printf("  %-22s %11s %11s %11s %11s %9s %9s %7s %10s\n", "material", "range mm",
              "dE/dx", "inel mfp", "el mfp", "<loss>", "<step>", "slots", "worst h/d");
  long long n_step_compared = 0, n_interacted = 0;
  double worst_hd = 0;
  for (int m = 0; m < kNMat; ++m) {
    geom::Volume<real_t> vols[1] = {
        {{geom::SolidType::kBox, {200, 200, 200}},
         geom::make_translation<real_t>({0, 0, 0}), 0, m, /*score_index=*/0},
    };
    geom::Volume<real_t>* d_vols = nullptr;
    cudaMalloc(&d_vols, sizeof(vols));
    cudaMemcpy(d_vols, vols, sizeof(vols), cudaMemcpyHostToDevice);
    Scene<real_t> dscene{};
    dscene.geometry = geom::Geometry<real_t>{d_vols, 1, 0};
    dscene.materials = d_mats;
    dscene.hadron_range = d_hrt;
    dscene.range_cut = real_t(0.7);
    dscene.scoring_volume = 0;
    Scene<real_t> hscene = dscene;
    hscene.geometry = geom::Geometry<real_t>{vols, 1, 0};
    hscene.materials = mats;
    hscene.hadron_range = h_hrt;

    // 1. the seeded primary, through the engine's kernel
    std::vector<Primary<real_t>> prims(kPer);
    for (auto& p : prims) {
      p.pos = kPos;
      p.dir = kDir;
      p.ekin = kEkin;
      p.particle = ParticleType::kGenericIon;
      p.ion_za = ion_za_of(kZ, kA);
      p.t0 = 0;
    }
    std::vector<TrackState<real_t>> prim_ts;
    seed(prims, dscene.geometry, prim_ts);

    // ... and the secondary, through push_nucleus
    cudaMemset(pool_s.count, 0, sizeof(int));
    cudaMemset(pool_s.overflow, 0, sizeof(int));
    EmitAsSecondary<<<(kPer + 63) / 64, 64>>>(pool_s, kPos, 0, kDir, kEkin, kZ, kA, kPer);
    Gather<<<(kPer + 63) / 64, 64>>>(pool_s, kPer, d_ts + kPer);
    std::vector<TrackState<real_t>> sec_ts(kPer);
    cudaMemcpy(sec_ts.data(), d_ts + kPer, sizeof(TrackState<real_t>) * kPer,
               cudaMemcpyDeviceToHost);

    int slot_bad = 0;
    for (int i = 0; i < kPer; ++i) { slot_bad += compare_slots(prim_ts[i], sec_ts[i], i); }
    if (slot_bad != 0) { fail("a seeded slot differs from the emitted one in a field a step reads"); }
    if (prim_ts[0].ion_za != ion_za_of(kZ, kA)) {
      fail("the seeded primary does not carry ion_za_of(6, 12)");
    }
    if (m == 0) {
      std::printf("  lineage (not compared): primary key %u parent %u creator %d flags %u; "
                  "secondary key %u parent %u creator %d flags %u\n",
                  prim_ts[0].rng_key, prim_ts[0].parent_key,
                  static_cast<int>(prim_ts[0].creator_process), prim_ts[0].flags,
                  sec_ts[0].rng_key, sec_ts[0].parent_key,
                  static_cast<int>(sec_ts[0].creator_process), sec_ts[0].flags);
    }

    // 2. both stepped on the device with the PRIMARY's key: identical to the bit
    std::vector<TrackState<real_t>> sec_keyed = sec_ts;
    for (int i = 0; i < kPer; ++i) { sec_keyed[i].rng_key = prim_ts[i].rng_key; }
    std::vector<Outcome> dp, ds;
    run(dscene, prim_ts, dp);
    run(dscene, sec_keyed, ds);
    double worst_pp = 0;
    int printed = 0, bad = 0;
    for (int i = 0; i < kPer; ++i) {
      bad += compare(dp[i], ds[i], "primary vs secondary", i, 0.0, &worst_pp, &printed);
    }
    if (bad != 0) { fail("the seeded primary's first step is not the emitted secondary's"); }

    // 3. host against device for the primary
    double worst = 0;
    printed = 0;
    bad = 0;
    double loss = 0, step = 0;
    for (int i = 0; i < kPer; ++i) {
      Outcome h{};
      one_step(hscene, prim_ts[i], hhad, &h);
      bad += compare(h, dp[i], "host vs device", i, kNumTol, &worst, &printed);
      loss += dp[i].edep;
      step += dp[i].length;
      if (dp[i].process != static_cast<int>(ProcessId::fTransportation)
          && dp[i].process != static_cast<int>(ProcessId::fIonisation)
          && dp[i].process != static_cast<int>(ProcessId::fMultipleScattering)) {
        ++n_interacted;
      }
      ++n_step_compared;
    }
    if (bad != 0) { fail("host and device disagree on the seeded primary's step"); }
    if (worst > worst_hd) { worst_hd = worst; }
    if (!(dp[0].range > 0) || !(dp[0].dedx > 0) || !(dp[0].inel_xs > 0) || !(dp[0].el_xs > 0)) {
      fail("a range, dE/dx or cross section is zero - the comparison would be vacuous");
    }
    if (dp[0].mass != data::get_nuclear_mass(kA, kZ)) {
      fail("the stepped mass is not G4NucleiProperties::GetNuclearMass(12, 6)");
    }
    std::printf("  %-22s %11.5g %11.5g %11.5g %11.5g %9.4g %9.4g %7s %10.2e\n", kMatNames[m],
                dp[0].range, dp[0].dedx, 1.0 / dp[0].inel_xs, 1.0 / dp[0].el_xs, loss / kPer,
                step / kPer, slot_bad == 0 ? "equal" : "DIFFER", worst);
    cudaFree(d_vols);
  }
  std::printf("  %lld first steps compared host/device and primary/secondary; %lld ended in a "
              "discrete process\n", n_step_compared, n_interacted);
  std::printf("  mass C12 %.6f MeV (GetNuclearMass(12, 6)); worst host/device rel %.2e\n",
              data::get_nuclear_mass(kA, kZ), worst_hd);

  // ---------------------------------------------------------------- 4. the ledger
  std::vector<int> d_counts(kNRef, 0);
  cudaMemcpy(d_counts.data(), d_ref_n, sizeof(int) * kNRef, cudaMemcpyDeviceToHost);
  int any = 0;
  for (int r = 0; r < kNRef; ++r) {
    if (d_counts[r] != 0 || h_ref_n[r] != 0) {
      std::printf("  ledger: %s - device %d, host %d\n",
                  had::hadronic_refusal_name(static_cast<had::HadronicRefusal>(r)), d_counts[r],
                  h_ref_n[r]);
      ++any;
    }
  }
  if (any != 0) {
    fail("the primaries' steps booked a refusal");
  } else {
    std::printf("  ledger after %lld primary steps: every row zero\n", n_step_compared);
  }
  const int nuc_before = d_counts[static_cast<int>(had::HadronicRefusal::kIonWithoutNuclide)];

  // The control: the same primaries with no nuclide, as seed_track_slot wrote before P30.
  {
    geom::Volume<real_t> vols[1] = {
        {{geom::SolidType::kBox, {200, 200, 200}},
         geom::make_translation<real_t>({0, 0, 0}), 0, 0, 0},
    };
    geom::Volume<real_t>* d_vols = nullptr;
    cudaMalloc(&d_vols, sizeof(vols));
    cudaMemcpy(d_vols, vols, sizeof(vols), cudaMemcpyHostToDevice);
    Scene<real_t> dscene{};
    dscene.geometry = geom::Geometry<real_t>{d_vols, 1, 0};
    dscene.materials = d_mats;
    dscene.hadron_range = d_hrt;
    dscene.range_cut = real_t(0.7);
    dscene.scoring_volume = 0;
    std::vector<Primary<real_t>> prims(kPer);
    for (auto& p : prims) {
      p.pos = kPos;
      p.dir = kDir;
      p.ekin = kEkin;
      p.particle = ParticleType::kGenericIon;
      p.ion_za = 0;
    }
    std::vector<TrackState<real_t>> ts;
    seed(prims, dscene.geometry, ts);
    // The probe in one_step needs a nuclide; the control asks only what the step books.
    std::vector<TrackState<real_t>> keep = ts;
    for (auto& t : keep) { t.ion_za = 0; }
    std::vector<Outcome> o;
    run(dscene, keep, o);
    cudaMemcpy(d_counts.data(), d_ref_n, sizeof(int) * kNRef, cudaMemcpyDeviceToHost);
    const int nuc =
        d_counts[static_cast<int>(had::HadronicRefusal::kIonWithoutNuclide)] - nuc_before;
    std::printf("  control, %d primaries seeded with ion_za = 0: %s = %d\n", kPer,
                had::hadronic_refusal_name(had::HadronicRefusal::kIonWithoutNuclide), nuc);
    if (nuc != kPer) { fail("the control did not book every nuclide-less primary"); }
    cudaFree(d_vols);
  }

  std::printf("\n%s (%d failure%s)\n", g_fails == 0 ? "PASS" : "FAIL", g_fails,
              g_fails == 1 ? "" : "s");
  return g_fails == 0 ? 0 : 1;
}
