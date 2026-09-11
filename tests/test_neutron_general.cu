// The neutron general process in the neutral stepper: the tables on the device, the sub-process
// choice in both configurations, and `step_neutral` host against device.
//
// P8d. The three things this file has to establish, in the order they can go wrong:
//
//  1. THE TABLES SURVIVED THE UPLOAD, EXACTLY. Five combined tables per material on
//     G4NeutronGeneralProcess's two grids, plus the two per-process G4PARTICLEXS data sets, read
//     back through the SAME accessors the transport reads them through - `NeutronGeneralXs::
//     total`, `::select` and `pxs_element_xs` - at every node of both grids and every bin
//     midpoint. Compared bit for bit, because an upload that changed a number is the failure
//     this section exists for and there is no arithmetic between the two sides to excuse a
//     difference. docs/RISK.md V53 is what reading the wrong transcription of this grid cost.
//
//  2. THE SUB-PROCESS FREQUENCIES MATCH THE PARTIALS, IN BOTH CONFIGURATIONS. This is the check
//     that the wiring picks the process the cross sections say it should, and the two
//     configurations predict it in two different ways:
//
//       kFinal    the general table's cumulative partials ARE the probabilities. P(elastic) is
//                 `table1`, P(inelastic) is `table2 - table1` and P(capture) is `1 - table2`
//                 below 20 MeV; above it P(inelastic) is `table4`. So the prediction is a number
//                 read off the device.
//       kStage1   two independent exponentials, so P(elastic first) is
//                 `sigma_el/(sigma_el + sigma_cap)` - the memorylessness this port already
//                 relies on for every discrete process (physics/hadronic/wiring.cuh's note on
//                 `decay_in_flight_length`), used here as a prediction rather than as an
//                 argument.
//
//     Both are measured by running `step_neutral` ITSELF, 200,000 times per (material, energy)
//     with a fixed seed, inside a box big enough that geometry never wins - so what is counted
//     is the process that defined the step, and the competition, the draw order and the stage
//     switch are all in the measurement. A test that re-implemented the competition would be a
//     tautology.
//
//  3. `step_neutral` ON THE DEVICE AGAINST THE SAME FUNCTION ON THE HOST, field by field, as
//     `tests/test_step_hadron.cu` does for the charged stepper - including each secondary's
//     species, NUCLIDE and energy, and the energy balance of every step.
//
// AND THE CAPTURE CAPACITY IS MEASURED HERE. `had::kNeutronCaptureSecondaryCap` is 16 because
// each unit costs 353 bytes of `run_step_neutral`'s frame; section 4 runs the cascade over every
// element of B1's four materials and reports the longest one, so the capacity is a claim with a
// number under it rather than a guess with a tripwire over it.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "core/rng.cuh"
#include "data/isotope_abundance.hh"
#include "data/materials.cuh"
#include "data/nist_materials.hh"
#include "host/level_upload.cuh"
#include "host/neutron_upload.cuh"
#include "physics/em/hadron_range.cuh"
#include "physics/scene.cuh"
#include "physics/stepper.cuh"

using real_t = double;
using namespace g4gpu;
namespace hxs = g4gpu::hadronic::xs;

namespace {

int g_fails = 0;
void fail(const char* what) {
  std::printf("  FAIL: %s\n", what);
  ++g_fails;
}

// =============================================================================================
// 1. the tables, read back through the transport's own accessors
// =============================================================================================

/// One (material, energy) probe of everything the transport can ask the tables.
struct Probe {
  real_t total = 0;        ///< NeutronGeneralXs::total
  real_t p_el = 0;         ///< table 1 (low) - read through select() at a swept q
  real_t p_el_inel = 0;    ///< table 2 (low)
  real_t p_inel = 0;       ///< table 4 (high)
  real_t xs_el = 0;        ///< the per-process elastic macroscopic cross section, 1/mm
  real_t xs_cap = 0;       ///< and the capture one
  int sub_at_q[9] = {};    ///< select() at nine fixed q values, as an int
};

/// The probe, written once so the host and the device run the same source.
__host__ __device__ void one_probe(const had::NeutronGeneralXs<real_t>& xs,
                                   const hxs::PxsDataSet<real_t>& el,
                                   const hxs::PxsDataSet<real_t>& cap,
                                   const data::Material<real_t>& mat, int imat, real_t e,
                                   Probe* out) {
  const real_t loge = log(e);
  out->total = xs.total(imat, e, loge);
  // The partials, read where `select` reads them. `phys_vec_log_value` on the same PhysVec is
  // what the socket's own `select` calls, so this is the number the choice is made from and not
  // a second way of computing it.
  const bool low = (e <= had::kNeutronXsEMiddle<real_t>());
  out->p_el = low ? hxs::phys_vec_log_value(xs.t1[imat], e, loge) : real_t(0);
  out->p_el_inel = low ? hxs::phys_vec_log_value(xs.t2[imat], e, loge) : real_t(0);
  out->p_inel = low ? real_t(0) : hxs::phys_vec_log_value(xs.t4[imat], e, loge);
  hxs::MaterialXs<real_t> m1{}, m2{};
  out->xs_el = had::neutron_sub_xs_per_volume<real_t>(&el, mat, e, loge, m1);
  out->xs_cap = had::neutron_sub_xs_per_volume<real_t>(&cap, mat, e, loge, m2);
  for (int i = 0; i < 9; ++i) {
    const real_t q = real_t(i) / real_t(8);
    out->sub_at_q[i] = static_cast<int>(xs.select(imat, e, loge, q));
  }
}

__global__ void RunProbes(had::NeutronGeneralXs<real_t> xs, const hxs::PxsDataSet<real_t>* el,
                          const hxs::PxsDataSet<real_t>* cap,
                          const data::Material<real_t>* mats, const int* imats,
                          const real_t* es, int n, Probe* out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) { return; }
  out[i] = Probe{};
  one_probe(xs, *el, *cap, mats[imats[i]], imats[i], es[i], &out[i]);
}

// =============================================================================================
// 2, 3. step_neutral, on both sides
// =============================================================================================

constexpr int kMaxSec = 20;

/// What one step produced, in a form both sides can write and a host can compare.
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
};

/// An emitter that records instead of appending, and works on both sides. The three positioning
/// fields are part of the interface `BufferEmitter` offers and a test emitter without them
/// compiles until a stepper starts assigning one.
struct RecordingEmitter {
  Outcome* out = nullptr;
  Vec3<real_t> pos{};
  int volume = 0;
  int event = 0;
  unsigned int child_count = 0u;

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

/// One step of one neutron, host and device from one source. The purpose value is
/// `run_step_neutral`'s, so the stream is the transport's.
__host__ __device__ void one_step(const Scene<real_t>& scene, TrackState<real_t> p,
                                  const had::NeutronGeneralXs<real_t>* xs,
                                  had::HadronicWiring<real_t> had, Outcome* out) {
  StepReport<real_t> rep;
  Philox<real_t> rng(p.rng_key, p.step, 0x4E7Au);
  RecordingEmitter em{out, p.pos, p.volume, p.event, 0u};
  real_t edep = 0;
  const bool alive =
      step_neutral(scene, p, ParticleType::kNeutron, xs, had, rng, em, edep, rep);
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

__global__ void RunNeutral(Scene<real_t> scene, const TrackState<real_t>* in, int n,
                           const had::NeutronGeneralXs<real_t>* xs,
                           had::HadronicWiring<real_t> had, Outcome* out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) { return; }
  out[i] = Outcome{};
  one_step(scene, in[i], xs, had, &out[i]);
}

/// The frequency run: which process defined the step, and how many secondaries it made. Written
/// to a compact record so 200,000 tracks per cell cost 16 bytes each instead of an `Outcome`.
struct Tally {
  int process = -1;
  int n_sec = 0;
  int max_sec_in_one_step = 0;
  float edep = 0;
};

__global__ void RunTally(Scene<real_t> scene, const had::NeutronGeneralXs<real_t>* xs,
                         had::HadronicWiring<real_t> had, real_t ekin, int volume, int n,
                         unsigned int seed, Tally* out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) { return; }
  Outcome o{};
  TrackState<real_t> p{};
  p.species = ParticleType::kNeutron;
  p.pos = Vec3<real_t>{0, 0, 0};
  p.dir = Vec3<real_t>{0, 0, 1};
  p.ekin = ekin;
  p.volume = volume;
  p.event = 0;
  p.weight = real_t(1);
  p.global_time = real_t(0);
  p.rng_key = seed + static_cast<unsigned int>(i);
  p.step = 0u;
  one_step(scene, p, xs, had, &o);
  Tally t{};
  t.process = o.process;
  t.n_sec = o.n_sec;
  t.edep = static_cast<float>(o.edep);
  out[i] = t;
}

// ---------------------------------------------------------------------------------------------

/// See `tests/test_step_hadron.cu`: nvcc contracts `a*b + c` into an FMA by default and MSVC on
/// the host does not, so a whole step differs in the last place or two and the difference
/// accumulates through the elastic boost and the capture cascade. The numeric fields are
/// compared to a relative tolerance three orders above what is measured; the DISCRETE ones -
/// status, process, secondary count, species and nuclide - are compared EXACTLY, because a
/// wiring defect shows up there and never as a rounding difference.
constexpr double kNumTol = 1e-9;

int compare(const Outcome& h, const Outcome& d, const char* tag, int idx, double tol,
            double* worst, int* printed) {
  int bad = 0;
  auto chk = [&](real_t a, real_t b, const char* field) {
    if (a == b) { return; }
    const double rel = (b != 0.0) ? std::fabs(double(a - b) / double(b))
                                  : std::fabs(double(a - b));
    if (rel > *worst) { *worst = rel; }
    if (rel <= tol) { return; }
    // AND AN ABSOLUTE FLOOR, WHICH IS THE RIGHT FORM FOR THE ONE FIELD THAT NEEDS IT.
    //
    // An elastic recoil's energy is `lv.e() - mass2` with both terms of order the TARGET's
    // mass - 1.7e4 MeV for oxygen and 1.9e5 for lead - so one ulp of the subtraction is 2e-12
    // to 3e-11 MeV however small the answer is. A sub-threshold recoil of 6e-5 MeV therefore
    // differs between the two compilers by 3e-8 RELATIVE and 2e-12 absolute, and it is the
    // second number that says whether anything is wrong. docs/PORTED.md 2.1.6 records the same
    // cancellation at 5.80e-11 for `elastic_apply` on a charged hadron. 1e-9 MeV is thirty
    // times the worst ulp and is a millionth of the smallest production cut in this port.
    if (std::fabs(double(a - b)) <= 1e-9) { return; }
    if (*printed < 12) {
      std::printf("  FAIL: %s track %d: %s host %.17g device %.17g (rel %.3e > %.0e)\n", tag,
                  idx, field, double(a), double(b), rel, tol);
      ++(*printed);
    }
    ++bad;
  };
  chk(h.ekin, d.ekin, "ekin");
  chk(h.edep, d.edep, "edep");
  chk(h.length, d.length, "true_length");
  chk(h.non_ionizing, d.non_ionizing, "non_ionizing");
  chk(h.px, d.px, "pos.x");
  chk(h.py, d.py, "pos.y");
  chk(h.pz, d.pz, "pos.z");
  chk(h.dx, d.dx, "dir.x");
  chk(h.dy, d.dy, "dir.y");
  chk(h.dz, d.dz, "dir.z");
  if (h.status != d.status || h.process != d.process) {
    if (*printed < 12) {
      std::printf("  FAIL: %s track %d: (status, process) host (%d, %d) device (%d, %d)\n", tag,
                  idx, h.status, h.process, d.status, d.process);
      ++(*printed);
    }
    ++bad;
  }
  if (h.alive != d.alive || h.volume != d.volume) {
    std::printf("  FAIL: %s track %d: alive/volume host %d/%d device %d/%d\n", tag, idx,
                h.alive, h.volume, d.alive, d.volume);
    ++bad;
  }
  if (h.n_sec != d.n_sec) {
    std::printf("  FAIL: %s track %d: %d secondaries on the host, %d on the device\n", tag, idx,
                h.n_sec, d.n_sec);
    ++bad;
  } else {
    for (int j = 0; j < h.n_sec; ++j) {
      if (h.sec_type[j] != d.sec_type[j] || h.sec_za[j] != d.sec_za[j]) {
        std::printf("  FAIL: %s track %d secondary %d: species %d (ZA %d) vs %d (%d)\n", tag,
                    idx, j, h.sec_type[j], h.sec_za[j], d.sec_type[j], d.sec_za[j]);
        ++bad;
      }
      chk(h.sec_ekin[j], d.sec_ekin[j], "secondary ekin");
      chk(h.sec_dx[j], d.sec_dx[j], "secondary dir.x");
      chk(h.sec_dy[j], d.sec_dy[j], "secondary dir.y");
      chk(h.sec_dz[j], d.sec_dz[j], "secondary dir.z");
    }
  }
  return bad;
}

/// A binomial z-score: |observed - expected| / sqrt(N p (1-p)). The prediction is exact (it is
/// read off the table the choice is made from), so the only uncertainty is the sampling.
double zscore(long long observed, double p, long long n) {
  if (p <= 0.0 || p >= 1.0) {
    return (observed == static_cast<long long>(p * double(n))) ? 0.0 : 1e9;
  }
  const double mean = p * double(n);
  const double sd = std::sqrt(double(n) * p * (1.0 - p));
  return std::fabs(double(observed) - mean) / sd;
}

}  // namespace

int main() {
  const cudaError_t lim = cudaDeviceSetLimit(cudaLimitStackSize, 16384);
  if (lim != cudaSuccess) {
    std::printf("FATAL: cudaDeviceSetLimit: %s\n", cudaGetErrorString(lim));
    return 2;
  }
  std::printf("== the neutron general process in step_neutral ==\n\n");

  // ---------------------------------------------------------------- the scene
  //
  // ONE BOX PER MATERIAL, and it is 1e6 mm across for the reason section 2 gives: the frequency
  // measurement counts which process defined the step, so geometry must never win. A 10 MeV
  // neutron's total mean free path in water is about 40 mm, and at thermal energies it is
  // millimetres, so 1e6 mm of half-width is many thousands of mean free paths in every material.
  // LEAD IS NOT ONE OF B1's FOUR, so it is added from the NIST database rather than spelled out
  // here - one Z, but also a density and a mean excitation energy, and a test that invented
  // those would be measuring its own numbers. `tests/test_all_materials.cu` builds all 300 the
  // same way. It is in the set because the neutron's cross sections are asked to be general
  // across material and Z, and B1's four are hydrogen, carbon, nitrogen, oxygen, magnesium,
  // phosphorus, sulphur, argon and calcium - nothing above Z = 20 and nothing with a large
  // capture cross section.
  // 1e9 mm, AND THE FIRST NUMBER WAS 1e6 AND WAS NOT ENOUGH. The frequency measurement counts
  // which process defined the step, so geometry must never win - and a 10 MeV neutron's total
  // mean free path in AIR is about 62 m, because air is a gas: at a half-width of 1 km, 3249 of
  // 200,000 tracks left the box. Escapees do not bias the ratio (given both exponentials exceed
  // L, the remainders are again exponentials with the same rates, so P(elastic first) is
  // unchanged) but they do make the denominator something the test has to explain, and a box
  // that is large enough explains nothing. At 1e9 mm the escape probability in air at 10 MeV is
  // exp(-16), i.e. under one track in 200,000.
  const real_t kHalf = 1e9;
  const int kNMat = 4;
  data::MaterialTable<real_t> mtab{};
  data::build_b1_materials<real_t>(mtab.m);
  mtab.count = data::kNumMaterials;
  int lead_index = -1;
  for (int i = 0; i < g4::nist::kNumNistMaterials; ++i) {
    const g4::nist::NistMaterial& e = g4::nist::kNistMaterials[i];
    if (std::strcmp(e.name, "G4_Pb") != 0) { continue; }
    int zs[data::kMaxElements];
    real_t w[data::kMaxElements];
    for (int k = 0; k < e.n_components; ++k) {
      zs[k] = e.components[k].z;
      w[k] = static_cast<real_t>(e.components[k].fraction);
    }
    lead_index = data::add_material<real_t>(mtab, static_cast<real_t>(e.density_g_cm3),
                                            e.n_components, zs, w,
                                            static_cast<real_t>(e.mean_excitation_eV),
                                            real_t(0.7), data::MaterialState::kSolid);
    if (lead_index >= 0 && e.has_sternheimer) {
      data::set_sternheimer<real_t>(mtab.m[lead_index], e.cbar, e.x0, e.x1, e.a, e.m,
                                    e.delta0);
    }
    if (lead_index >= 0) {
      data::set_nist_stopping<real_t>(mtab.m[lead_index], e.name, nullptr);
    }
    break;
  }
  if (lead_index < 0) {
    fail("G4_Pb could not be built - the fourth material of this test is missing");
    std::printf("\nFAILED (%d failures)\n", g_fails);
    return 1;
  }
  const int n_mats = mtab.count;
  const int mat_index[kNMat] = {data::kWater, data::kAir, data::kBoneCompact, lead_index};
  const char* mat_name[kNMat] = {"water", "air", "bone", "lead"};
  geom::Volume<real_t> vols[kNMat];
  for (int i = 0; i < kNMat; ++i) {
    vols[i] = geom::Volume<real_t>{{geom::SolidType::kBox, {kHalf, kHalf, kHalf}},
                                   geom::make_translation<real_t>({0, 0, 0}), 0,
                                   mat_index[i], /*score_index=*/0};
  }
  geom::Volume<real_t>* d_vols = nullptr;
  cudaMalloc(&d_vols, sizeof(vols));
  cudaMemcpy(d_vols, vols, sizeof(vols), cudaMemcpyHostToDevice);

  data::Material<real_t>* mats = mtab.m;
  data::Material<real_t>* d_mats = nullptr;
  cudaMalloc(&d_mats, sizeof(data::Material<real_t>) * n_mats);
  cudaMemcpy(d_mats, mats, sizeof(data::Material<real_t>) * n_mats,
             cudaMemcpyHostToDevice);

  // ---------------------------------------------------------------- the tables
  //
  // The device copies through the uploader the engine uses, and a HOST twin built by the same
  // builders - `pxs_load` and `ngp_build_table`, which are the only ones there are - so the
  // host/device comparison below is of the STEPPER and not of two table builders. This is the
  // discipline `tests/test_step_hadron.cu` states for the elastic tables.
  auto nown = host::upload_neutron_tables<real_t>(mats, n_mats, /*verbose=*/true);
  if (!nown.ok) {
    fail("the neutron tables did not upload - G4PARTICLEXSDATA could not be resolved");
    std::printf("\nFAILED (%d failures)\n", g_fails);
    return 1;
  }

  auto* h_t_el = new data::ParticleXsTable<real_t>();
  auto* h_t_inel = new data::ParticleXsTable<real_t>();
  auto* h_t_cap = new data::ParticleXsTable<real_t>();
  auto* h_ds_el = new hxs::PxsDataSet<real_t>();
  auto* h_ds_inel = new hxs::PxsDataSet<real_t>();
  auto* h_ds_cap = new hxs::PxsDataSet<real_t>();
  {
    const std::string dir = host::g4particlexs_subdir("neutron");
    const bool ok =
        hxs::pxs_load<real_t>(hxs::PxsKind::kNeutronElastic, hxs::neutron<real_t>(), dir,
                              *h_t_el, *h_ds_el)
        && hxs::pxs_load<real_t>(hxs::PxsKind::kNeutronInelastic, hxs::neutron<real_t>(), dir,
                                 *h_t_inel, *h_ds_inel)
        && hxs::pxs_load<real_t>(hxs::PxsKind::kNeutronCapture, hxs::neutron<real_t>(), dir,
                                 *h_t_cap, *h_ds_cap);
    if (!ok) {
      fail("the host copy of G4PARTICLEXS4.0/neutron would not load");
      std::printf("\nFAILED (%d failures)\n", g_fails);
      return 1;
    }
  }
  auto* h_gt = new hxs::NeutronGeneralTable<real_t>();
  hxs::ngp_build_table<real_t>(*h_ds_el, *h_ds_inel, *h_ds_cap, mats, n_mats, *h_gt);
  const had::NeutronGeneralXs<real_t> h_xs = had::neutron_general_view<real_t>(*h_gt);

  // P3's level scheme, on the device and in host memory, for the capture cascade.
  data::LevelTableStorage storage;
  auto lown = host::upload_level_data(storage, data::kLevelZMax, /*verbose=*/true);
  if (lown.view.managers == nullptr) {
    fail("the level table did not upload - G4LEVELGAMMADATA could not be resolved");
    std::printf("\nFAILED (%d failures)\n", g_fails);
    return 1;
  }
  const data::LevelTable h_levels = storage.view();

  // ONE VOLUME PER SCENE, and the four scenes differ only in which of `vols` they point at.
  //
  // Four coincident boxes in one geometry was the first version of this and it measured nothing:
  // `step_to_boundary` from a volume that another volume occupies exactly returns a distance of
  // zero, so every one of 200,000 steps ended on the boundary and every sub-process frequency
  // came out 0.000000 against a prediction of 0.985. A test whose geometry is wrong reports the
  // physics as broken, which is why this note is here rather than only the fix.
  Scene<real_t> dscene[kNMat], hscene[kNMat];
  for (int i = 0; i < kNMat; ++i) {
    dscene[i] = Scene<real_t>{};
    dscene[i].geometry = geom::Geometry<real_t>{d_vols + i, 1, 0};
    dscene[i].materials = d_mats;
    dscene[i].range_cut = real_t(0.7);
    dscene[i].scoring_volume = 0;
    hscene[i] = dscene[i];
    hscene[i].geometry = geom::Geometry<real_t>{vols + i, 1, 0};
    hscene[i].materials = mats;
  }

  had::HadronicWiring<real_t> dhad{};
  dhad.neutron = nown.sub;
  dhad.level_data = lown.view;
  had::HadronicWiring<real_t> hhad{};
  hhad.neutron.elastic = h_ds_el;
  hhad.neutron.capture = h_ds_cap;
  hhad.level_data = h_levels;
  std::vector<int> h_ref_n(static_cast<int>(had::HadronicRefusal::kNumHadronicRefusals), 0);
  std::vector<double> h_ref_e(h_ref_n.size(), 0.0);
  hhad.books.count = h_ref_n.data();
  hhad.books.energy = h_ref_e.data();
  int* d_ref_n = nullptr;
  double* d_ref_e = nullptr;
  cudaMalloc(&d_ref_n, sizeof(int) * h_ref_n.size());
  cudaMalloc(&d_ref_e, sizeof(double) * h_ref_n.size());
  cudaMemset(d_ref_n, 0, sizeof(int) * h_ref_n.size());
  cudaMemset(d_ref_e, 0, sizeof(double) * h_ref_n.size());
  dhad.books.count = d_ref_n;
  dhad.books.energy = d_ref_e;

  // =========================================================================== 1. the upload
  std::printf("\n-- 1. the tables on the device, read through the transport's own accessors\n");
  {
    // Every node of both grids and every bin midpoint, for every material. The midpoints are
    // what make this a test of the GRID and not only of the values: an interpolation divides by
    // a bin width, so a node energy that moved by an ulp shows up between the nodes and nowhere
    // else. That is docs/RISK.md V53's measurement, here against the uploaded copy.
    std::vector<int> imats;
    std::vector<real_t> es;
    std::vector<char> is_node;
    for (int mi = 0; mi < n_mats; ++mi) {
      for (int j = 0; j < h_gt->n_low; ++j) {
        imats.push_back(mi);
        es.push_back(h_gt->e_low[static_cast<std::size_t>(j)]);
        is_node.push_back(1);
        if (j + 1 < h_gt->n_low) {
          imats.push_back(mi);
          es.push_back(real_t(0.5) * (h_gt->e_low[static_cast<std::size_t>(j)]
                                      + h_gt->e_low[static_cast<std::size_t>(j + 1)]));
          is_node.push_back(0);
        }
      }
      for (int j = 0; j < h_gt->n_high; ++j) {
        imats.push_back(mi);
        es.push_back(h_gt->e_high[static_cast<std::size_t>(j)]);
        is_node.push_back(1);
        if (j + 1 < h_gt->n_high) {
          imats.push_back(mi);
          es.push_back(real_t(0.5) * (h_gt->e_high[static_cast<std::size_t>(j)]
                                      + h_gt->e_high[static_cast<std::size_t>(j + 1)]));
          is_node.push_back(0);
        }
      }
    }
    const int n = static_cast<int>(es.size());
    int* d_im = nullptr;
    real_t* d_es = nullptr;
    Probe* d_pr = nullptr;
    cudaMalloc(&d_im, sizeof(int) * n);
    cudaMalloc(&d_es, sizeof(real_t) * n);
    cudaMalloc(&d_pr, sizeof(Probe) * n);
    cudaMemcpy(d_im, imats.data(), sizeof(int) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_es, es.data(), sizeof(real_t) * n, cudaMemcpyHostToDevice);
    had::NeutronGeneralXs<real_t> d_view{};
    cudaMemcpy(&d_view, nown.d_general, sizeof(d_view), cudaMemcpyDeviceToHost);
    RunProbes<<<(n + 127) / 128, 128>>>(d_view, nown.sub.elastic, nown.sub.capture, d_mats,
                                        d_im, d_es, n, d_pr);
    const cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
      std::printf("  FAIL: probe launch: %s\n", cudaGetErrorString(e));
      ++g_fails;
    } else {
      std::vector<Probe> dev(n);
      cudaMemcpy(dev.data(), d_pr, sizeof(Probe) * n, cudaMemcpyDeviceToHost);
      long long bad = 0, compared = 0, summed = 0;
      double worst = 0, worst_sum = 0;
      int printed = 0;
      for (int i = 0; i < n; ++i) {
        Probe h{};
        one_probe(h_xs, *h_ds_el, *h_ds_cap, mats[imats[i]], imats[i], es[i], &h);
        const Probe& g = dev[i];
        // AT A NODE, EXACTLY. `phys_vec_log_value` at a node energy returns
        // `y1 + 0*(y2-y1)`, so the number that comes out is the uploaded double itself and any
        // difference means the upload changed it or the node energies moved - which is the
        // failure this section is for, and docs/RISK.md V53's own measurement.
        //
        // BETWEEN NODES, TO A TOLERANCE, and for one compiler flag rather than for comfort:
        // the interpolation is `y1 + (e-x1)/(x2-x1) * (y2-y1)`, a multiply-add that nvcc
        // contracts into an FMA (`-fmad=true`, which build_all.bat does not turn off) and MSVC
        // on the host does not. Measured over these probes, 5 of 18,840 midpoint values differ
        // and the worst is 1.9e-16 relative - one ulp. Comparing a midpoint exactly would be
        // asserting that two compilers round the same way.
        const bool node = (is_node[static_cast<std::size_t>(i)] != 0);
        auto ex = [&](real_t a, real_t b, const char* what) {
          ++compared;
          if (a == b) { return; }
          const double rel = (b != 0.0) ? std::fabs(double(a - b) / double(b))
                                        : std::fabs(double(a - b));
          if (rel > worst) { worst = rel; }
          if (!node && rel <= 1e-14) { return; }
          if (printed < 8) {
            std::printf("  FAIL: %s at material %d, %.17g MeV (%s): host %.17g device %.17g\n",
                        what, imats[i], double(es[i]), node ? "node" : "midpoint", double(a),
                        double(b));
            ++printed;
          }
          ++bad;
        };
        // THE PER-PROCESS SUMS ARE NOT, AND THE REASON IS ONE COMPILER FLAG.
        // `store_compute_cross_section` accumulates `total += n_atoms[i] * sigma_i` over the
        // material's elements, and nvcc contracts that multiply-add into an FMA by default
        // (`-fmad=true`, which build_all.bat does not turn off) where MSVC on the host does not.
        // So a two- to nine-element sum differs in the last place and nothing about the upload
        // is involved. Measured at 1.9e-15 relative over these 4710 probes, which is what the
        // tolerance is set three orders above; `tests/test_step_hadron.cu` records the same
        // mechanism for the charged stepper. The TABLES above have no such sum in them, which
        // is why only these two rows get a tolerance.
        auto approx = [&](real_t a, real_t b, const char* what) {
          ++summed;
          if (a == b) { return; }
          const double rel = (b != 0.0) ? std::fabs(double(a - b) / double(b))
                                        : std::fabs(double(a - b));
          if (rel > worst_sum) { worst_sum = rel; }
          if (rel <= 1e-12) { return; }
          if (printed < 8) {
            std::printf("  FAIL: %s at material %d, %.17g MeV: host %.17g device %.17g "
                        "(rel %.3e)\n", what, imats[i], double(es[i]), double(a), double(b),
                        rel);
            ++printed;
          }
          ++bad;
        };
        ex(h.total, g.total, "the combined cross section");
        ex(h.p_el, g.p_el, "table 1 (sigma_el/total)");
        ex(h.p_el_inel, g.p_el_inel, "table 2 ((el+inel)/total)");
        ex(h.p_inel, g.p_inel, "table 4 (sigma_inel/total)");
        approx(h.xs_el, g.xs_el, "G4NeutronElasticXS per volume");
        approx(h.xs_cap, g.xs_cap, "G4NeutronCaptureXS per volume");
        for (int k = 0; k < 9; ++k) {
          ++compared;
          if (h.sub_at_q[k] != g.sub_at_q[k]) {
            ++bad;
            if (printed < 8) {
              std::printf("  FAIL: select() at q=%.3f, material %d, %.17g MeV: host %d "
                          "device %d\n", double(k) / 8.0, imats[i], double(es[i]),
                          h.sub_at_q[k], g.sub_at_q[k]);
              ++printed;
            }
          }
        }
      }
      std::printf("  %lld exact comparisons and %lld tolerant ones over %d (material, energy) "
                  "probes - %d nodes and %d bin midpoints per material, both grids\n",
                  compared, summed, n, h_gt->n_low + h_gt->n_high,
                  h_gt->n_low + h_gt->n_high - 2);
      std::printf("  the five combined tables and select(): worst relative deviation %.3e\n",
                  worst);
      std::printf("  the two per-process sums: worst relative deviation %.3e against 1e-12\n",
                  worst_sum);
      std::printf("  %lld disagreements\n", bad);
      if (bad != 0) { ++g_fails; }
      // ANTI-VACUITY: the probe has to be able to see a difference at all. Nothing here would
      // fail if `total` returned zero on both sides, so the magnitudes are asserted too.
      long long nonzero = 0;
      for (int i = 0; i < n; ++i) {
        if (dev[i].total > 0 && dev[i].xs_el > 0) { ++nonzero; }
      }
      std::printf("  %lld of %d probes have a positive combined and elastic cross section\n",
                  nonzero, n);
      if (nonzero < n / 2) {
        fail("more than half the probes read a zero cross section - the tables are empty");
      }
    }
    cudaFree(d_im);
    cudaFree(d_es);
    cudaFree(d_pr);
  }

  // ================================================== 2. the sub-process frequencies
  std::printf("\n-- 2. the sub-process frequencies against the partials, both configurations\n");
  {
    // Thermal, and then four decades. `kNeutronXsEMin` is 1 keV, so 2.53e-8 MeV is below the
    // table's first node and reads it - which is a real case and the one a thermal neutron in a
    // shield is in, so it is measured rather than avoided.
    const real_t kEnergies[] = {real_t(2.53e-8), real_t(1e-3), real_t(0.1), real_t(1),
                                real_t(10)};
    const char* ename[] = {"thermal", "1 keV", "100 keV", "1 MeV", "10 MeV"};
    const int kN = 200000;
    Tally* d_tal = nullptr;
    cudaMalloc(&d_tal, sizeof(Tally) * kN);
    std::vector<Tally> tal(kN);

    for (int stage = 0; stage < 2; ++stage) {
      had::HadronicWiring<real_t> w = dhad;
      w.stage = (stage == 0) ? had::HadronicStage::kStage1 : had::HadronicStage::kFinal;
      std::printf("  %s\n", had::hadronic_stage_name(w.stage));
      std::printf("  %-6s %-9s %12s %12s %12s %10s %10s %8s %6s\n", "mat", "E", "P(el) pred",
                  "P(el) meas", "P(cap) pred", "P(cap) meas", "P(inel)", "worst z", "esc");
      for (int mi = 0; mi < kNMat; ++mi) {
        for (int ei = 0; ei < 5; ++ei) {
          const real_t e = kEnergies[ei];
          const real_t loge = std::log(e);
          const int imat = mat_index[mi];
          // THE PREDICTION, and it is a different formula per stage - see the file header.
          double p_el = 0, p_cap = 0, p_inel = 0;
          if (stage == 0) {
            hxs::MaterialXs<real_t> m1{}, m2{};
            const double xe = had::neutron_sub_xs_per_volume<real_t>(h_ds_el, mats[imat], e,
                                                                      loge, m1);
            const double xc = had::neutron_sub_xs_per_volume<real_t>(h_ds_cap, mats[imat], e,
                                                                      loge, m2);
            if (xe + xc > 0) {
              p_el = xe / (xe + xc);
              p_cap = xc / (xe + xc);
            }
          } else if (e <= had::kNeutronXsEMiddle<real_t>()) {
            p_el = double(hxs::phys_vec_log_value(h_xs.t1[imat], e, loge));
            p_inel = double(hxs::phys_vec_log_value(h_xs.t2[imat], e, loge)) - p_el;
            p_cap = 1.0 - p_el - p_inel;
          } else {
            p_inel = double(hxs::phys_vec_log_value(h_xs.t4[imat], e, loge));
            p_el = 1.0 - p_inel;
          }

          // The seed is fixed per (stage, material, energy) so the run is reproducible and the
          // twenty cells are independent draws rather than one stream reused.
          const unsigned int seed = 0x9E37u + 1000u * static_cast<unsigned int>(stage)
                                    + 100u * static_cast<unsigned int>(mi)
                                    + 7u * static_cast<unsigned int>(ei);
          RunTally<<<(kN + 255) / 256, 256>>>(dscene[mi], nown.d_general, w, e, 0, kN, seed,
                                              d_tal);
          const cudaError_t err = cudaDeviceSynchronize();
          if (err != cudaSuccess) {
            std::printf("  FAIL: tally launch (%s, %s): %s\n", mat_name[mi], ename[ei],
                        cudaGetErrorString(err));
            ++g_fails;
            continue;
          }
          cudaMemcpy(tal.data(), d_tal, sizeof(Tally) * kN, cudaMemcpyDeviceToHost);
          long long n_el = 0, n_cap = 0, n_inel = 0, n_other = 0;
          for (int i = 0; i < kN; ++i) {
            switch (tal[i].process) {
              case static_cast<int>(ProcessId::fHadronElastic): ++n_el; break;
              case static_cast<int>(ProcessId::fNeutronCapture): ++n_cap; break;
              case static_cast<int>(ProcessId::fHadronInelastic): ++n_inel; break;
              default: ++n_other; break;
            }
          }
          // The denominator is the number of steps that INTERACTED. See the note on `kHalf`:
          // conditioning on "the box did not win" leaves the ratio unchanged, because two
          // exponentials conditioned on both exceeding L are again exponentials with the same
          // rates - but the count has to be the one the prediction is about, and a handful of
          // escapees at 1e9 mm is what is left of the effect.
          const long long n_int = n_el + n_cap + n_inel;
          const double z_el = zscore(n_el, p_el, n_int);
          const double z_cap = zscore(n_cap, p_cap, n_int);
          const double z_inel = (p_inel > 0) ? zscore(n_inel, p_inel, n_int)
                                             : ((n_inel == 0) ? 0.0 : 1e9);
          const double worst = std::fmax(z_el, std::fmax(z_cap, z_inel));
          std::printf("  %-6s %-9s %12.6f %12.6f %12.6f %10.6f %10.6f %8.2f %6lld\n",
                      mat_name[mi], ename[ei], p_el, double(n_el) / double(n_int), p_cap,
                      double(n_cap) / double(n_int), double(n_inel) / double(n_int), worst,
                      n_other);
          // 5 sigma, which is the gate P2's own SampleZandA frequencies are measured against.
          if (worst > 5.0) {
            std::printf("    FAIL: %s at %s, %s: worst z = %.2f over %lld interactions\n",
                        (stage == 0) ? "stage1" : "final", mat_name[mi], ename[ei], worst,
                        n_int);
            ++g_fails;
          }
          if (n_other > kN / 1000) {
            std::printf("    FAIL: %lld of %d steps ended on the boundary or the time cut - "
                        "the box is too small for this mean free path\n",
                        n_other, kN);
            ++g_fails;
          }
          // The stage's own claim: no inelastic in stage 1, some in the final stage wherever
          // the partial says so. Without this the two stages could be the same code path.
          if (stage == 0 && n_inel != 0) {
            fail("stage 1 selected the inelastic sub-process - it is not a process there");
            ++g_fails;
          }
        }
      }
    }
    cudaFree(d_tal);
  }

  // ============================ 2b. the same frequencies, on the HOST and on the host tables
  std::printf("\n-- 2b. the choice functions on the host, against the host-built tables\n");
  {
    // WHY THIS IS NOT THE SAME TEST AS 2. Section 2 ran `step_neutral` on the DEVICE, because
    // 8 million steps with a capture cascade in them is a GPU's work and minutes of a host's.
    // What that leaves unchecked is the choice read off the tables the HOST built, which is a
    // different object from the uploaded copy - and section 1 compared the two only through the
    // accessors, at fixed q values. So this draws the uniforms and counts, on the host, with no
    // transport in it at all: `select()` for the final configuration and the two exponentials
    // for stage 1, 200,000 draws per cell.
    //
    // It is cheap because there is no final state in it - four million log-vector lookups - and
    // it is the section the brief asked for in as many words. Section 3's field-by-field
    // host-against-device comparison is what ties the two together.
    const real_t kEnergies[] = {real_t(2.53e-8), real_t(1e-3), real_t(0.1), real_t(1),
                                real_t(10)};
    const char* ename[] = {"thermal", "1 keV", "100 keV", "1 MeV", "10 MeV"};
    const int kN = 200000;
    double worst_overall = 0;
    for (int stage = 0; stage < 2; ++stage) {
      for (int mi = 0; mi < kNMat; ++mi) {
        for (int ei = 0; ei < 5; ++ei) {
          const real_t e = kEnergies[ei];
          const real_t loge = std::log(e);
          const int imat = mat_index[mi];
          Philox<real_t> rng(0x5EEDu + 97u * static_cast<unsigned int>(stage * 100 + mi * 5
                                                                       + ei),
                             0u, 0x2B1Cu);
          long long n_el = 0, n_cap = 0, n_inel = 0;
          double p_el = 0, p_cap = 0, p_inel = 0;
          if (stage == 1) {
            for (int i = 0; i < kN; ++i) {
              switch (h_xs.select(imat, e, loge, rng.uniform())) {
                case had::NeutronSubProcess::kElastic: ++n_el; break;
                case had::NeutronSubProcess::kInelastic: ++n_inel; break;
                default: ++n_cap; break;
              }
            }
            if (e <= had::kNeutronXsEMiddle<real_t>()) {
              p_el = double(hxs::phys_vec_log_value(h_xs.t1[imat], e, loge));
              p_inel = double(hxs::phys_vec_log_value(h_xs.t2[imat], e, loge)) - p_el;
              p_cap = 1.0 - p_el - p_inel;
            } else {
              p_inel = double(hxs::phys_vec_log_value(h_xs.t4[imat], e, loge));
              p_el = 1.0 - p_inel;
            }
          } else {
            hxs::MaterialXs<real_t> m1{}, m2{};
            const real_t xe = had::neutron_sub_xs_per_volume<real_t>(h_ds_el, mats[imat], e,
                                                                     loge, m1);
            const real_t xc = had::neutron_sub_xs_per_volume<real_t>(h_ds_cap, mats[imat], e,
                                                                     loge, m2);
            // The two exponentials `step_neutral` draws in stage 1, in its order, so this
            // counts the same competition and not a formula for it.
            for (int i = 0; i < kN; ++i) {
              const real_t s_el = (xe > real_t(0)) ? -std::log(rng.uniform()) / xe
                                                   : geom::kInfinity<real_t>();
              const real_t s_cap = (xc > real_t(0)) ? -std::log(rng.uniform()) / xc
                                                    : geom::kInfinity<real_t>();
              if (s_cap < s_el) { ++n_cap; } else { ++n_el; }
            }
            if (double(xe) + double(xc) > 0) {
              p_el = double(xe) / (double(xe) + double(xc));
              p_cap = double(xc) / (double(xe) + double(xc));
            }
          }
          const double z = std::fmax(zscore(n_el, p_el, kN),
                                     std::fmax(zscore(n_cap, p_cap, kN),
                                               (p_inel > 0) ? zscore(n_inel, p_inel, kN)
                                                            : ((n_inel == 0) ? 0.0 : 1e9)));
          if (z > worst_overall) { worst_overall = z; }
          if (z > 5.0) {
            std::printf("  FAIL: %s %s %s: worst z = %.2f (P(el) %.6f measured %.6f)\n",
                        (stage == 0) ? "stage1" : "final", mat_name[mi], ename[ei], z, p_el,
                        double(n_el) / kN);
            ++g_fails;
          }
        }
      }
    }
    std::printf("  40 cells x %d draws on the host tables: worst z = %.2f against a 5-sigma "
                "gate\n", kN, worst_overall);
  }

  // ============================================ 3. step_neutral, host against device
  std::printf("\n-- 3. step_neutral on the device against the same function on the host\n");
  {
    // 400 tracks per material per configuration, at four energies, and the box is the huge one
    // so every one of them interacts. The RNG keys are distinct, so this covers 1600 independent
    // draws of the whole chain per column - the target draw, the Chips scatter, the cascade.
    //
    // THREE COLUMNS AND NOT TWO, because P(capture) is about 1e-5 at these energies in every one
    // of these materials - so the first two columns are 1600 ELASTIC steps and would have left
    // the capture branch, the biggest single piece of code this package wires, uncompared
    // between host and device. The third switches the elastic sub-process off, which is a real
    // Geant4 configuration in stage 1 (`/process/inactivate hadElastic` reaches a neutron
    // exactly when the general process is off - docs/RISK.md V53) and makes every interaction a
    // capture.
    const real_t kEs[] = {real_t(1e-3), real_t(0.1), real_t(2), real_t(50)};
    const int kPer = 100;
    const int n = kNMat * 4 * kPer;
    const char* colname[3] = {"stage1", "final", "stage1, capture only"};
    TrackState<real_t>* d_in = nullptr;
    Outcome* d_out = nullptr;
    cudaMalloc(&d_in, sizeof(TrackState<real_t>) * n);
    cudaMalloc(&d_out, sizeof(Outcome) * n);

    for (int stage = 0; stage < 3; ++stage) {
      had::HadronicWiring<real_t> dw = dhad, hw = hhad;
      dw.stage = hw.stage =
          (stage == 1) ? had::HadronicStage::kFinal : had::HadronicStage::kStage1;
      if (stage == 2) { dw.hadron_elastic = hw.hadron_elastic = false; }
      double worst = 0;
      int printed = 0, bad = 0;
      int n_el = 0, n_cap = 0, n_inel = 0, n_sec_total = 0, max_sec = 0, det_bad = 0;
      double balance_worst = 0;
      // ONE LAUNCH PER MATERIAL, because each scene holds one volume - see the note above.
      for (int mi = 0; mi < kNMat; ++mi) {
        std::vector<TrackState<real_t>> tracks;
        for (int ei = 0; ei < 4; ++ei) {
          for (int k = 0; k < kPer; ++k) {
            TrackState<real_t> p{};
            p.species = ParticleType::kNeutron;
            p.pos = Vec3<real_t>{0, 0, 0};
            p.dir = Vec3<real_t>{0, 0, 1};
            p.ekin = kEs[ei];
            p.volume = 0;
            p.event = 0;
            p.weight = real_t(1);
            p.global_time = real_t(0);
            p.rng_key = 0x51A7u + 977u * static_cast<unsigned int>(tracks.size())
                        + 104729u * static_cast<unsigned int>(mi);
            p.step = 0u;
            tracks.push_back(p);
          }
        }
        const int nm = static_cast<int>(tracks.size());
        cudaMemcpy(d_in, tracks.data(), sizeof(TrackState<real_t>) * nm,
                   cudaMemcpyHostToDevice);
        cudaMemset(d_out, 0, sizeof(Outcome) * nm);
        RunNeutral<<<(nm + 63) / 64, 64>>>(dscene[mi], d_in, nm, nown.d_general, dw, d_out);
        const cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) {
          std::printf("  FAIL: step launch (%s): %s\n", mat_name[mi], cudaGetErrorString(e));
          ++g_fails;
          continue;
        }
        std::vector<Outcome> dev(nm);
        cudaMemcpy(dev.data(), d_out, sizeof(Outcome) * nm, cudaMemcpyDeviceToHost);

        // DETERMINISM ACROSS THE LAUNCH GEOMETRY, which is the property every dose in this
        // repository rests on and which `tests/test_step_hadron.cu` section 5 asserts for the
        // charged stepper. The same tracks at a different block size must give IDENTICAL
        // results - tolerance zero - because nothing in a step may depend on which thread ran
        // it. It is worth asserting here and not only there because P8d is the first thing to
        // emit a secondary out of `step_neutral`, and a secondary's RNG key is derived from its
        // parent's `child_count` rather than from a slot index for exactly this reason.
        cudaMemset(d_out, 0, sizeof(Outcome) * nm);
        RunNeutral<<<(nm + 255) / 256, 256>>>(dscene[mi], d_in, nm, nown.d_general, dw, d_out);
        if (cudaDeviceSynchronize() != cudaSuccess) {
          fail("the 256-thread relaunch failed");
        } else {
          std::vector<Outcome> dev2(nm);
          cudaMemcpy(dev2.data(), d_out, sizeof(Outcome) * nm, cudaMemcpyDeviceToHost);
          double wd = 0;
          int pd = 0;
          for (int i = 0; i < nm; ++i) {
            det_bad += compare(dev[i], dev2[i], "block size", i, 0.0, &wd, &pd);
          }
        }

        for (int i = 0; i < nm; ++i) {
          Outcome h{};
          one_step(hscene[mi], tracks[i], &h_xs, hw, &h);
          bad += compare(h, dev[i], colname[stage], i, kNumTol, &worst, &printed);
          if (h.process == static_cast<int>(ProcessId::fHadronElastic)) { ++n_el; }
          if (h.process == static_cast<int>(ProcessId::fNeutronCapture)) { ++n_cap; }
          if (h.process == static_cast<int>(ProcessId::fHadronInelastic)) { ++n_inel; }
          n_sec_total += h.n_sec;
          if (h.n_sec > max_sec) { max_sec = h.n_sec; }
          // THE ENERGY BALANCE, which is independent of what Geant4 would have done: what came
          // in must be what is left plus what was deposited plus what the secondaries carry.
          //
          // The capture branch is EXCLUDED from it, and the reason is the physics rather than
          // an exemption: a radiative capture releases the neutron's BINDING energy, 2 to
          // 9 MeV that was not on the track, so `ekin_pre` is not an upper bound on the
          // products. The mass balance of a capture is what `tests/test_capture.cu` checks
          // against the oracle over 600 calls, with the nuclear masses in it.
          if (h.process == static_cast<int>(ProcessId::fHadronElastic)) {
            double out_e = double(h.ekin) + double(h.edep);
            for (int j = 0; j < h.n_sec; ++j) { out_e += double(h.sec_ekin[j]); }
            const double err = std::fabs(out_e - double(tracks[i].ekin));
            if (err > balance_worst) { balance_worst = err; }
          }
        }
      }
      std::printf("  %-22s %d tracks, %d disagreements, worst relative deviation %.3e\n",
                  colname[stage], n, bad, worst);
      std::printf("    processes: %d elastic, %d capture, %d inelastic (refused); %d "
                  "secondaries, at most %d in one step\n",
                  n_el, n_cap, n_inel, n_sec_total, max_sec);
      std::printf("    elastic energy balance, worst ABSOLUTE: %.3e MeV\n", balance_worst);
      std::printf("    64 threads against 256, bit for bit: %d disagreements\n", det_bad);
      if (bad != 0) { ++g_fails; }
      if (det_bad != 0) {
        fail("the same tracks gave different answers at a different block size");
      }
      // ABSOLUTE, for the reason `compare` gives at length: the recoil term is a difference of
      // two numbers of order the target mass, so a relative limit on a 6e-5 MeV deposit is a
      // limit on an ulp of 1.9e5.
      if (balance_worst > 1e-9) {
        fail("an elastic step did not conserve energy to 1e-9 MeV");
      }
      // A test that passed because nothing happened is the failure mode docs/RISK.md V29 and
      // V32 are about, so a zero is a failure in the column where the process should fire.
      if (stage != 2 && n_el == 0) {
        fail("the elastic sub-process never fired in 1600 steps");
      }
      if (stage == 2 && n_cap == 0) {
        fail("the capture sub-process never fired in 1600 steps with elastic switched off");
      }
      if (stage == 2 && n_el != 0) {
        fail("the elastic sub-process fired with hadron_elastic false");
      }
      if (stage == 1 && n_inel == 0) {
        fail("the inelastic sub-process was never selected in the final stage - the refusal "
             "counter cannot move");
      }
      if (stage != 1 && n_inel != 0) {
        fail("stage 1 selected the inelastic sub-process");
      }
      if (max_sec >= kMaxSec) {
        fail("a step filled this test's secondary buffer - the count is not trustworthy");
      }
    }
    cudaFree(d_in);
    cudaFree(d_out);
  }

  // ================================ 4. how long the longest capture cascade actually is
  std::printf("\n-- 4. the capture cascade's length, against the capacity it is given\n");
  {
    // `had::kNeutronCaptureSecondaryCap` is 16 and it costs 353 bytes of kernel frame per unit,
    // so the number has to be justified by what the cascade does rather than by comfort. This
    // runs capture on every element of B1's four materials at five energies and reports the
    // longest cascade seen - through `step_neutral` itself, so it is the transport's own path.
    //
    // 2000 captures per (material, energy) is enough to see a tail: P7's oracle measured 3.05
    // secondaries per capture over 600 calls, so a cascade of 16 is five times the mean and this
    // is looking for it rather than assuming it is absent.
    const real_t kEs[] = {real_t(2.53e-8), real_t(1e-4), real_t(1e-2), real_t(1), real_t(15)};
    const int kN = 2000;
    TrackState<real_t>* d_in = nullptr;
    Outcome* d_out = nullptr;
    cudaMalloc(&d_in, sizeof(TrackState<real_t>) * kN);
    cudaMalloc(&d_out, sizeof(Outcome) * kN);
    int overall_max = 0;
    long long captures = 0, secondaries = 0;
    had::HadronicWiring<real_t> w = dhad;
    w.stage = had::HadronicStage::kStage1;
    // Capture only: the elastic sub-process is switched off so every interaction IS a capture,
    // which is what makes 2000 tracks 2000 cascades instead of a few dozen. It is a legal
    // configuration in stage 1 and only in stage 1 - `/process/inactivate hadElastic` reaches a
    // neutron exactly when the general process is off (docs/RISK.md V53).
    w.hadron_elastic = false;
    std::vector<Outcome> host_out(kN);
    for (int mi = 0; mi < kNMat; ++mi) {
      for (int ei = 0; ei < 5; ++ei) {
        std::vector<TrackState<real_t>> tracks(kN);
        for (int k = 0; k < kN; ++k) {
          TrackState<real_t>& p = tracks[static_cast<std::size_t>(k)];
          p = TrackState<real_t>{};
          p.species = ParticleType::kNeutron;
          p.pos = Vec3<real_t>{0, 0, 0};
          p.dir = Vec3<real_t>{0, 0, 1};
          p.ekin = kEs[ei];
          p.volume = 0;
          p.weight = real_t(1);
          p.rng_key = 0xC0FFEEu + 31u * static_cast<unsigned int>(mi * 5 + ei)
                      + 1009u * static_cast<unsigned int>(k);
          p.step = 0u;
        }
        cudaMemcpy(d_in, tracks.data(), sizeof(TrackState<real_t>) * kN,
                   cudaMemcpyHostToDevice);
        cudaMemset(d_out, 0, sizeof(Outcome) * kN);
        RunNeutral<<<(kN + 63) / 64, 64>>>(dscene[mi], d_in, kN, nown.d_general, w, d_out);
        if (cudaDeviceSynchronize() != cudaSuccess) {
          fail("the cascade-length launch failed");
          continue;
        }
        cudaMemcpy(host_out.data(), d_out, sizeof(Outcome) * kN, cudaMemcpyDeviceToHost);
        for (int k = 0; k < kN; ++k) {
          if (host_out[k].process != static_cast<int>(ProcessId::fNeutronCapture)) { continue; }
          ++captures;
          secondaries += host_out[k].n_sec;
          if (host_out[k].n_sec > overall_max) { overall_max = host_out[k].n_sec; }
        }
      }
    }
    std::printf("  %lld captures, %lld secondaries (%.2f per capture), longest cascade %d "
                "against a capacity of %d\n",
                captures, secondaries,
                (captures > 0) ? double(secondaries) / double(captures) : 0.0, overall_max,
                had::kNeutronCaptureSecondaryCap);

    // AND WHAT A NULL LEVEL SCHEME DOES, which is the claim `Upload`'s new refusal rests on.
    //
    // The refusal says a capture with no levels to walk "would kill the neutron and emit no
    // gamma - its binding energy, 2 to 9 MeV per capture, would vanish". That is a statement
    // about `G4PhotonEvaporation::BreakUpChain` with an empty table and it is measured here
    // rather than asserted: the same tracks, with `level_data` replaced by a null view. If the
    // two columns agreed, the refusal would be protecting nothing and the 9.52 MB upload would
    // be free to skip.
    //
    // IN LEAD, AND WATER WAS THE WRONG CHOICE FOR A REASON WORTH KEEPING. Run in water at
    // thermal energy the two columns are IDENTICAL - 2000 captures, exactly 2 secondaries each,
    // 4448.7461 MeV emitted either way - because a thermal neutron in water captures on
    // HYDROGEN, and `G4NeutronRadCapture::ApplyYourself`'s `A <= 1` branch is a closed-form
    // two-body decay (n + p -> d + gamma) that never opens the level scheme. So the one capture
    // a water phantom mostly makes is the one that does not read the table this test is about.
    // Lead is A = 204..208, which is the `A >= 5` compound branch through `BreakUpChain`.
    {
      had::HadronicWiring<real_t> nolevels = w;
      nolevels.level_data = data::LevelTable{};
      std::vector<TrackState<real_t>> tracks(kN);
      for (int k = 0; k < kN; ++k) {
        TrackState<real_t>& p = tracks[static_cast<std::size_t>(k)];
        p = TrackState<real_t>{};
        p.species = ParticleType::kNeutron;
        p.pos = Vec3<real_t>{0, 0, 0};
        p.dir = Vec3<real_t>{0, 0, 1};
        p.ekin = real_t(2.53e-8);
        p.volume = 0;
        p.weight = real_t(1);
        p.rng_key = 0xC0FFEEu + 1009u * static_cast<unsigned int>(k);
        p.step = 0u;
      }
      cudaMemcpy(d_in, tracks.data(), sizeof(TrackState<real_t>) * kN, cudaMemcpyHostToDevice);
      long long with_n = 0, with_sec = 0, without_n = 0, without_sec = 0;
      double with_e = 0, without_e = 0;
      for (int pass = 0; pass < 2; ++pass) {
        cudaMemset(d_out, 0, sizeof(Outcome) * kN);
        RunNeutral<<<(kN + 63) / 64, 64>>>(dscene[3], d_in, kN, nown.d_general,
                                           (pass == 0) ? w : nolevels, d_out);
        if (cudaDeviceSynchronize() != cudaSuccess) {
          fail("the null-level-scheme launch failed");
          break;
        }
        cudaMemcpy(host_out.data(), d_out, sizeof(Outcome) * kN, cudaMemcpyDeviceToHost);
        for (int k = 0; k < kN; ++k) {
          if (host_out[k].process != static_cast<int>(ProcessId::fNeutronCapture)) { continue; }
          double se = 0;
          for (int j = 0; j < host_out[k].n_sec; ++j) { se += double(host_out[k].sec_ekin[j]); }
          if (pass == 0) {
            ++with_n;
            with_sec += host_out[k].n_sec;
            with_e += se;
          } else {
            ++without_n;
            without_sec += host_out[k].n_sec;
            without_e += se;
          }
        }
      }
      std::printf("  thermal captures in lead, with the level scheme:     %lld captures, "
                  "%lld secondaries, %.4f MeV emitted\n", with_n, with_sec, with_e);
      std::printf("                                 with a null table:    %lld captures, "
                  "%lld secondaries, %.4f MeV emitted\n", without_n, without_sec, without_e);
      // NOT "FEWER", AND THAT IS THE FINDING. A null level scheme does not emit nothing: the
      // cascade takes the continuum arm of `generate_gamma` instead of walking a discrete level
      // scheme, and in lead at thermal energy it emits 26,485 secondaries and 14,609.9 MeV where
      // the real table gives 9,021 and 13,694.7 - nearly three times the multiplicity and 6.7%
      // more energy. So the failure mode a missing G4LEVELGAMMADATA produces is a plausible
      // capture with a wrong spectrum, which is worse than a zero and is why `Upload` refuses
      // the combination rather than warning about it. docs/RISK.md V60.
      if (with_n == 0 || without_n == 0) {
        fail("no thermal captures in lead - the comparison has nothing in it");
      } else if (without_sec == with_sec && without_e == with_e) {
        fail("a null level scheme produced the identical cascade - Upload's refusal is "
             "protecting nothing");
      }
    }
    if (captures < 1000) {
      fail("fewer than 1000 captures in 40,000 tracks - the measurement has no tail to see");
    }
    if (overall_max >= had::kNeutronCaptureSecondaryCap) {
      fail("a cascade reached the capacity - raise kNeutronCaptureSecondaryCap and re-measure "
           "the kernel frame");
    }
    cudaFree(d_in);
    cudaFree(d_out);
  }

  // ============================================ 5. the refusal ledgers, on both sides
  std::printf("\n-- 5. the ledgers\n");
  {
    std::vector<int> d_n(h_ref_n.size(), 0);
    std::vector<double> d_e(h_ref_n.size(), 0.0);
    cudaMemcpy(d_n.data(), d_ref_n, sizeof(int) * d_n.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(d_e.data(), d_ref_e, sizeof(double) * d_e.size(), cudaMemcpyDeviceToHost);
    for (std::size_t i = 0; i < d_n.size(); ++i) {
      if (d_n[i] == 0 && h_ref_n[i] == 0) { continue; }
      std::printf("  %-28s device %6d (%10.3f MeV)   host %6d (%10.3f MeV)\n",
                  had::hadronic_refusal_name(static_cast<had::HadronicRefusal>(i)), d_n[i],
                  d_e[i], h_ref_n[i], h_ref_e[i]);
    }
    const int k = static_cast<int>(had::HadronicRefusal::kCaptureOverflow);
    if (d_n[k] != 0) {
      fail("a capture cascade overflowed the final state on the device");
    }
    const int ks = static_cast<int>(had::HadronicRefusal::kCaptureSecondarySpecies);
    if (d_n[ks] != 0) {
      fail("a capture secondary had a PDG code core/particle.cuh has no row for");
    }
    const int kr = static_cast<int>(had::HadronicRefusal::kCaptureRefused);
    if (d_n[kr] != 0) {
      fail("G4NeutronRadCapture refused a capture - an unphysical target or 100 rejected "
           "attempts");
    }
  }

  host::free_neutron_tables<real_t>(nown);
  host::free_level_data(lown);
  if (g_fails == 0) {
    std::printf("\nPASSED\n");
    return 0;
  }
  std::printf("\nFAILED (%d failures)\n", g_fails);
  return 1;
}
