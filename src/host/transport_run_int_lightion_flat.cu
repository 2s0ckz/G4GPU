// P36: the flat light-ion reaction's unit (docs/RISK.md V310) - `G4BinaryLightIonReaction::
// ApplyYourself`'s cascade arm as data (bic/flat/ion_flat.cuh), on the kernels and the drain loop
// of host/flat_drain_kernels.cuh. The monolithic `run_interaction<kLightIon>` and its unit are
// untouched.
#define G4GPU_ENGINE_KERNELS_ONLY
#include "host/flat_drain_kernels.cuh"

namespace g4gpu::host {

namespace {

/// The light-ion reaction. 64 nucleons a nucleus (V307: the heavier nucleus is at most Fe56);
/// 256 tracks (V307's high-water 259 - the one call past it goes to the thread path) and 512
/// collisions (303 measured).
struct IonPolicy {
  static constexpr int kN = 64;
  static constexpr int kP = 256;
  static constexpr int kC = 512;
  static constexpr int kItemsPerLane = 16;
  using Rec = bic::flat::FlatIon<kN, kP, kC>;
  static constexpr had::InelasticModel kModel = had::InelasticModel::kLightIon;
  __device__ static bic::flat::FlatCascade<kN, kP, kC>& core(Rec& r) { return r.c; }
  __device__ static void start(Rec& r, int e, const physics::hadronic::HadProjectile<double>& p,
                               const Philox<double>& rng) {
    bic::flat::ion_start(r, e, p.pdg, p.kin_energy, p.mass, p.charge, p.baryon_number, 0, 0,
                         rng);
  }
  /// `blir_apply_yourself` is handed `HadNucleus{z, a, 0}`: no lambdas.
  __device__ static void model(Rec& r) { bic::flat::stage_ion_model(r, 0); }
  __device__ static void nucleus(Rec& r, const bic::flat::FlatEnv& env,
                                 bic::Nucleus3DScratch& nsc,
                                 bic::flat::FlatScratch<kN, kP>& scr) {
    bic::flat::stage_ion_nucleus(r, env, nsc, scr);
  }
  /// `blir_cascade_arm` after `Interact` into `blir_fs`, then `run_arm_light_ion`'s
  /// `copy_final_state`.
  __device__ static bool pack(Rec& r, had::InteractionSlot<double>& s,
                              const had::InteractionPool<double>& pool,
                              const data::LevelTable& lt, const preco::PrecoWorkspace& pws) {
    bic::BlirStorage store = had::blir_storage_of<double>(s, pool);
    if (bic::flat::ion_pack(r, store, lt, pool.fermi, pws, s.blir_fs) !=
        bic::flat::kIonFinalState) {
      return false;
    }
    had::copy_final_state<double>(s.blir_fs, s.fs);
    return true;
  }
};

}  // namespace

std::size_t flat_ion_lane_bytes() { return flatk::lane_bytes<IonPolicy>(); }

FlatPool flat_ion_pool_build(int lanes, int k_steps, int strag_capacity) {
  return flatk::pool_build<IonPolicy>(lanes, k_steps, strag_capacity);
}

void flat_ion_pool_free(FlatPool& p) { flatk::pool_free(p); }

int flat_ion_drain(FlatPool& pool, const Scene<double>& scene,
                   const had::PendingInteraction<double>* items, const int* idx, int n, int base,
                   TrackBuffer<double> out, int batch, double* score, double* voxel_score,
                   const had::HadronicWiring<double>& had,
                   const had::InteractionPool<double>* pool_ptr, int n_tail_slots,
                   vis::TrajectoryBuffer traj, int* status_warn, SecondaryArena sec,
                   EmitterBooks books, StepTap<double> hook, FlatStats& stats) {
  return flatk::drain<IonPolicy>(pool, scene, items, idx, n, base, out, batch, score,
                                 voxel_score, had, pool_ptr, n_tail_slots, traj, status_warn, sec,
                                 books, hook, stats);
}

}  // namespace g4gpu::host
