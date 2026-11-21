// P36: the flat Binary cascade's unit (docs/RISK.md V308, V309) - `G4BinaryCascade::
// ApplyYourself` as data, its kernels and its drain loop from host/flat_drain_kernels.cuh with
// the nucleon arm's set-up and packing. The monolithic `run_interaction<kBinary>` and its unit
// are untouched; this unit compiles no stepping kernel and no other model.
#define G4GPU_ENGINE_KERNELS_ONLY
#include "host/flat_drain_kernels.cuh"

namespace g4gpu::host {

namespace {

/// The nucleon arm. 64 nucleons cover every target of the B1 materials (A <= 48 measured) and
/// Fe56; a heavier target is handed to the thread path at `stage_model`. The pool and the
/// collision list are V307's high-water marks (89 and 95) with margin.
struct BinaryPolicy {
  static constexpr int kN = 64;
  static constexpr int kP = 128;
  static constexpr int kC = 256;
  static constexpr int kItemsPerLane = 3;
  static constexpr int kNucleiPerTry = 1;  ///< P39: the target
  using Rec = bic::flat::FlatCascade<kN, kP, kC>;
  static constexpr had::InelasticModel kModel = had::InelasticModel::kBinary;
  __device__ static Rec& core(Rec& r) { return r; }
  __device__ static void start(Rec& r, int e, const physics::hadronic::HadProjectile<double>& p,
                               const Philox<double>& rng) {
    bic::flat::flat_start(r, e, p.pdg, p.kin_energy, 0, 0, rng);
  }
  __device__ static void model(Rec& r) { bic::flat::stage_model(r); }
  /// P39: the turn's target, from the nucleus factory (docs/RISK.md V321).
  __device__ static void request(Rec& r, const npool::FactoryLists& lists) {
    bic::flat::flat_request_target(r, lists);
  }
  __device__ static void nucleus(Rec& r, const bic::flat::FlatEnv& env,
                                 bic::Nucleus3DScratch& nsc,
                                 bic::flat::FlatScratch<kN, kP>& scr) {
    bic::flat::stage_nucleus(r, env, nsc, scr);
  }
  /// `apply_yourself`'s packing into `bic_fs`, then `run_arm_binary`'s `copy_final_state`.
  __device__ static bool pack(Rec& r, had::InteractionSlot<double>& s,
                              const had::InteractionPool<double>& pool,
                              const data::LevelTable& lt, const preco::PrecoWorkspace& pws) {
    (void)lt;
    const bic::BicStorage store = had::bic_storage_of<double>(s, pool, pws);
    if (!bic::flat::flat_pack_nucleon(r, store.cascade, s.bic_fs)) { return false; }
    had::copy_final_state<double>(s.bic_fs, s.fs);
    return true;
  }

  // ---- P41 (docs/RISK.md V325, V326): the tail pool. Capacities over V325's measured maxima
  // (the handler's lists 5 / 27 / 26 / 27 / 27, the products 26, the final state 128 = the
  // model's own `BicFinalState`, so `copy_final_state` never counts an overflow the slot's 856
  // would not); width, gate and block are V326's measurements.
  using TailCaps = deex::flat::BinaryTailCaps;
  using Tail = deex::flat::TailEntry<TailCaps, bic::BicFinalState>;
  static constexpr int kTailWidth = 4096;
  static constexpr int kTailEmitWidth = 256;
  static constexpr int kTailPoolMin = kTailWidth;
  static constexpr int kTailPoolBlock = 1;
  /// The cost bin a waiting tail is sorted into: V325's measurement says what predicts a
  /// residual's de-excitation time before it runs.
  __device__ static int tail_bin(const bic::flat::FlatCascade<kN, kP, kC>& c) {
    return deex::flat::tail_cost_bin(c);
  }
  /// `apply_yourself`'s packing into the entry's `BicFinalState`.
  __device__ static bool pack_entry(Rec& r, Tail& t, const had::InteractionPool<double>& pool,
                                    const data::LevelTable& lt,
                                    const preco::PrecoWorkspace& pws,
                                    const bic::flat::TailFallback* fb) {
    (void)fb;
    (void)pool;
    (void)lt;
    (void)pws;
    const bic::CascadeWorkspace ws = deex::flat::tail_cascade_view(t);
    return bic::flat::flat_pack_nucleon(r, ws, t.model_fs);
  }
};

}  // namespace

std::size_t flat_binary_lane_bytes() { return flatk::lane_bytes<BinaryPolicy>(); }

FlatPool flat_binary_pool_build(int lanes, int k_steps, int strag_capacity) {
  return flatk::pool_build<BinaryPolicy>(lanes, k_steps, strag_capacity);
}

void flat_binary_pool_free(FlatPool& p) { flatk::pool_free(p); }

// P41 (docs/RISK.md V328): a view of the pool with its slots moved. The kernels index a slot -
// and FTF's workspace - by their thread index from 0 (`pool.slot(i)`, `pool.ftf.slot(i)`), so a
// launch handed this view uses slots [first, first + count) and FTF's [0, count) and nothing else.
const had::InteractionPool<double>* flat_pool_range(const had::InteractionPool<double>* d_pool,
                                                     int first, int count, bool with_ftf) {
  had::InteractionPool<double> v;
  if (cudaMemcpy(&v, d_pool, sizeof(v), cudaMemcpyDeviceToHost) != cudaSuccess) {
    cudaGetLastError();
    return nullptr;
  }
  if (first < 0 || count <= 0 || first + count > v.n_slots) { return nullptr; }
  v.slots += first;
  v.n_slots = count;
  v.ftf.n_slots = with_ftf ? std::min(v.ftf.n_slots, count) : 0;
  had::InteractionPool<double>* d = nullptr;
  if (cudaMalloc(&d, sizeof(v)) != cudaSuccess ||
      cudaMemcpy(d, &v, sizeof(v), cudaMemcpyHostToDevice) != cudaSuccess) {
    cudaGetLastError();
    cudaFree(d);
    return nullptr;
  }
  return d;
}

int flat_binary_drain(FlatPool& pool, const Scene<double>& scene,
                      const had::PendingInteraction<double>* items, const int* idx, int n,
                      int base, TrackBuffer<double> out, int batch, double* score,
                      double* voxel_score, const had::HadronicWiring<double>& had,
                      const had::InteractionPool<double>* pool_ptr, int n_tail_slots,
                      vis::TrajectoryBuffer traj, int* status_warn, SecondaryArena sec,
                      EmitterBooks books, StepTap<double> hook, FlatStats& stats) {
  return flatk::drain<BinaryPolicy>(pool, scene, items, idx, n, base, out, batch, score,
                                    voxel_score, had, pool_ptr, n_tail_slots, traj, status_warn,
                                    sec, books, hook, stats);
}

}  // namespace g4gpu::host
