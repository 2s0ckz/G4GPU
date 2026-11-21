// P36: the engine's view of the flat Binary cascade (docs/RISK.md V308-V311) - a pool of
// per-cascade records the phase kernels advance together, and the one call that drains a
// kBinary bucket through them and hands back the entries the one-thread path must still run.
//
// THIS HEADER NAMES NO MODEL TYPE, for the reason `interaction_pool.cuh` names none (V233): the
// engine unit includes it, and the engine unit must not compile the cascade. The records, the
// kernels and the drain loop are `src/host/transport_run_int_binary_flat.cu`'s.
//
// WHO RUNS WHAT. The flat path is the default for `run_interaction<kBinary>`'s bucket, and only
// for the engine's own hook (`StepTap<double>`): a hook project's units are its own and keep the
// thread path. `G4GPU_CASCADE_FLAT=0` selects the thread path for every interaction - the
// one-thread transcription the exact oracle tests hold, untouched. An entry the flat path does
// not finish - more than K time steps of its own, a flat list too short, any refusal, a hydrogen
// target, a nucleon under `theBCminP`, `CheckResult` exhausted - is handed back by queue index
// and rerun on the thread path from its key, so what it computes is exactly what the thread path
// computes for it, and whether an entry is handed back depends on that entry alone.
#pragma once

#include <cstddef>

#include "core/step_hook.cuh"
#include "core/track_buffer.cuh"
#include "physics/hadronic/interaction_queue.cuh"
#include "physics/hadronic/wiring.cuh"
#include "physics/scene.cuh"
#include "render/trajectory.cuh"

namespace g4gpu::had {
template <typename real_t>
struct InteractionPool;
}

namespace g4gpu::host {

/// The flat pool as the engine owns it: opaque device memory and its sizes.
struct FlatPool {
  void* records = nullptr;      ///< `lanes` per-cascade records
  void* buffers = nullptr;      ///< ONE prebuilt `bic::CascadeBuffers`, read by every lane
  int* counters = nullptr;      ///< the drain's device counters
  int* items = nullptr;         ///< TRANSPORT's (cascade, secondary) items
  int* strag = nullptr;         ///< the entries handed to the thread path, by queue index
  /// P39 (docs/RISK.md V321): the nucleus factory's two request lists (A <= 32, A <= 64) of
  /// `factory_cap` requests each and their two counts - the set-up's nuclei, built in the
  /// records by `npool::nucleus_factory_list_kernel` before the set-up reads them.
  void* factory_req = nullptr;
  int* factory_count = nullptr;
  int factory_cap = 0;
  int lanes = 0;
  int item_capacity = 0;
  int k_steps = 0;              ///< the straggler bound K
  std::size_t record_bytes = 0;
  std::size_t bytes = 0;        ///< everything above
  bool buffers_built = false;   ///< built at the first drain, from the interaction pool's table
  // ---- P41 (docs/RISK.md V326): the tail pool - one entry per tail in flight, off the
  // interaction slots (physics/hadronic/deexcitation/flat/tail_pool.cuh) - built with the
  // records and counted in `bytes`. Width 0 (`G4GPU_FLAT_TAIL_WIDTH=0`) builds none, and every
  // tail then takes a thread-path slot as P36's did.
  void* tail_entries = nullptr;
  int* tail_bins = nullptr;     ///< the waiting tails by cost bin: 16 lists of `lanes`
  int* tail_order = nullptr;    ///< one launch's lanes in cost order, `tail_width`
  void* tail_filled = nullptr;  ///< the emission's 856-wide result buffers, `tail_emit_width`
  int tail_width = 0;
  int tail_emit_width = 0;
  void* tail_big = nullptr;      ///< slot-sized de-excitation workspaces a lane continues on
  void* tail_big_ws = nullptr;   ///< their `preco::PrecoWorkspace` views
  int* tail_big_busy = nullptr;  ///< a flag each
  int tail_big_n = 0;
  std::size_t tail_entry_bytes = 0;
  std::size_t tail_bytes = 0;   ///< all of the above, inside `bytes`
  // ---- end P41
  // ---- P42 (docs/RISK.md V331): the larger records a reaction that would outgrow its own is
  // continued on (the light-ion reaction with its factory on), `n_big` of them, in `bytes`.
  void* big = nullptr;
  int n_big = 0;
  bool ok() const { return records != nullptr && lanes > 0; }
};

/// The per-lane bytes the pool costs, so the engine can size it before building it.
std::size_t flat_binary_lane_bytes();

/// Builds the pool at `lanes` lanes (the buffers prebuilt on the device), or returns an empty
/// one if the card cannot hold it. `k_steps` is the straggler bound.
FlatPool flat_binary_pool_build(int lanes, int k_steps, int strag_capacity);
void flat_binary_pool_free(FlatPool& p);

/// What the drain reports, summed over a run.
struct FlatStats {
  long long entries = 0;        ///< kBinary entries the flat path took
  long long finished = 0;       ///< finished on it
  long long handed[8] = {};     ///< handed back, by `bic::flat::HandOff`
  long long rounds = 0;         ///< lockstep rounds
  long long launches = 0;
  long long tail_waits = 0;     ///< rounds a record waited for a tail slot
};

/// Drains `n` kBinary entries (`items[idx[base + e]]`) through the flat path into `out`.
/// Returns how many entries it handed back; their queue indices are in `pool.strag[0..)`, ready
/// for `run_interaction<kBinary>` with `base = 0`.
int flat_binary_drain(FlatPool& pool, const Scene<double>& scene,
                      const had::PendingInteraction<double>* items, const int* idx, int n,
                      int base, TrackBuffer<double> out, int batch, double* score,
                      double* voxel_score, const had::HadronicWiring<double>& had,
                      const had::InteractionPool<double>* pool_ptr, int n_tail_slots,
                      vis::TrajectoryBuffer traj, int* status_warn, SecondaryArena sec,
                      EmitterBooks books, StepTap<double> hook, FlatStats& stats);

/// P41 (docs/RISK.md V328): a device copy of the interaction pool's view whose slots are
/// `count` of the pool's from `first` - and FTF's workspaces the first `count` of FTF's when
/// `with_ftf`, none otherwise - so that drains running at once never share a slot. Defined in
/// transport_run_int_binary_flat.cu (it needs the view's type); null when it cannot be made.
const had::InteractionPool<double>* flat_pool_range(const had::InteractionPool<double>* d_pool,
                                                     int first, int count, bool with_ftf);

/// The light-ion reaction's flat path (docs/RISK.md V310): the same contract for the kLightIon
/// bucket, from transport_run_int_lightion_flat.cu.
std::size_t flat_ion_lane_bytes();
FlatPool flat_ion_pool_build(int lanes, int k_steps, int strag_capacity);
void flat_ion_pool_free(FlatPool& p);
int flat_ion_drain(FlatPool& pool, const Scene<double>& scene,
                   const had::PendingInteraction<double>* items, const int* idx, int n, int base,
                   TrackBuffer<double> out, int batch, double* score, double* voxel_score,
                   const had::HadronicWiring<double>& had,
                   const had::InteractionPool<double>* pool_ptr, int n_tail_slots,
                   vis::TrajectoryBuffer traj, int* status_warn, SecondaryArena sec,
                   EmitterBooks books, StepTap<double> hook, FlatStats& stats);

}  // namespace g4gpu::host
