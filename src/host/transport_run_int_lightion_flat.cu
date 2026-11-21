// P36: the flat light-ion reaction's unit (docs/RISK.md V310) - `G4BinaryLightIonReaction::
// ApplyYourself`'s cascade arm as data (bic/flat/ion_flat.cuh), on the kernels and the drain loop
// of host/flat_drain_kernels.cuh. The monolithic `run_interaction<kLightIon>` and its unit are
// untouched.
#define G4GPU_ENGINE_KERNELS_ONLY
#include "host/flat_drain_kernels.cuh"

namespace g4gpu::host {

namespace {

/// The light-ion reaction. 64 nucleons a nucleus (V307: the heavier nucleus is at most Fe56);
/// 256 tracks (V307's high-water 259) and, since P42, 1,024 collisions (V331: 381 at most in
/// the campaign's Fe56 on Al27, 160 after a set-up; the room `flat_room_for_step` asks before an
/// iteration is up to 4 x (1 + 2 x 56) = 452 collisions, so at 512 nearly every Fe56 reaction
/// would have moved to a larger record, at 1,024 only one past 572). A reaction that would
/// outgrow either moves to `Big` (P42, V331).
struct IonPolicy {
  static constexpr int kN = 64;
  static constexpr int kP = 256;
  static constexpr int kC = 1024;
  static constexpr int kItemsPerLane = 16;
  static constexpr int kNucleiPerTry = 2;  ///< P39: the projectile and the target
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
  /// P39: the try's projectile and target, from the nucleus factory (docs/RISK.md V322).
  __device__ static void request(Rec& r, const npool::FactoryLists& lists) {
    bic::flat::ion_request(r, lists);
  }
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

  // ---- P42 (docs/RISK.md V331): the larger record a reaction that would outgrow its own is
  // continued on - the thread path's slot capacities (`kSlotPoolCap`, `kSlotCollisionCap`), so
  // it refuses for capacity exactly where the thread path would. 451,616 B each; 64 of them.
  static constexpr int kBigP = bic::kSlotPoolCap;
  static constexpr int kBigC = bic::kSlotCollisionCap;
  static constexpr int kBigRecords = 64;
  using Big = bic::flat::FlatIon<kN, kBigP, kBigC>;
  __device__ static bool grow_copy(Big& b, const Rec& r) { return bic::flat::ion_grow_copy(b, r); }
  __device__ static void big_model(Big& b) { bic::flat::stage_ion_model(b, 0); }
  __device__ static void big_nucleus(Big& b, const bic::flat::FlatEnv& env,
                                     bic::Nucleus3DScratch& nsc,
                                     bic::flat::FlatScratch<kN, kBigP>& scr) {
    bic::flat::stage_ion_nucleus(b, env, nsc, scr);
  }
  __device__ static bool big_pack(Big& b, had::InteractionSlot<double>& s,
                                  const had::InteractionPool<double>& pool,
                                  const data::LevelTable& lt, const preco::PrecoWorkspace& pws) {
    bic::BlirStorage store = had::blir_storage_of<double>(s, pool);
    if (bic::flat::ion_pack(b, store, lt, pool.fermi, pws, s.blir_fs) !=
        bic::flat::kIonFinalState) {
      return false;
    }
    had::copy_final_state<double>(s.blir_fs, s.fs);
    return true;
  }

  // ---- P41 (docs/RISK.md V325, V327): the tail pool. The light-ion arm's products (85 at most,
  // V680/V325) and `SortResult`'s two lists at 256; the final state 168 = the model's own
  // `BlirFinalState`, so `copy_final_state` never counts an overflow the slot would not.
  using TailCaps = deex::flat::IonTailCaps;
  using Tail = deex::flat::TailEntry<TailCaps, bic::BlirFinalState>;
  static constexpr int kTailWidth = 1024;
  static constexpr int kTailEmitWidth = 128;
  static constexpr int kTailPoolMin = kTailWidth;
  static constexpr int kTailPoolBlock = 1;
  __device__ static int tail_bin(const bic::flat::FlatCascade<kN, kP, kC>& c) {
    return deex::flat::tail_cost_bin(c);
  }
  /// `blir_cascade_arm` after `Interact` into the entry's `BlirFinalState`; a compact handler
  /// workspace that binds is continued on a slot-sized one (`fb`).
  __device__ static bool pack_entry(Rec& r, Tail& t, const had::InteractionPool<double>& pool,
                                    const data::LevelTable& lt,
                                    const preco::PrecoWorkspace& pws,
                                    const bic::flat::TailFallback* fb) {
    bic::BlirStorage store = deex::flat::tail_blir_view(t);
    return bic::flat::ion_pack(r, store, lt, pool.fermi, pws, t.model_fs, fb,
                               t.cascaders_save) == bic::flat::kIonFinalState;
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
