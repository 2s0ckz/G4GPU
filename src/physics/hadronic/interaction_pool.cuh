// The interaction pool as the HOST sees it: an opaque device view, the sizes it was built at,
// and the two functions that build and free it - and not one model header.
//
// WHY THIS HEADER EXISTS. docs/RISK.md V221 found, with nvcc's own dependency files, that every
// one of the engine's translation units - the gamma stepping unit included - read every hadronic
// model header, through one chain: `host/transport_run.cuh` included `interaction_apply.cuh` so
// that `Upload` could `sizeof` an `InteractionSlot`, construct one as the image uploaded into
// every slot, build the Binary cascade's channel table and FTFP's pool, and hand the kernels a
// `had::InteractionPool` BY VALUE. So a change to one cascade header rebuilt twenty-three units,
// 78 minutes, when one of them reaches it. Before the dependency files nothing could have shown
// it; V65 and V189 had split the kernels one per unit and left the headers whole.
//
// WHAT THE HOST READS INSTEAD. The two pool types are DECLARED here and never defined: the
// definitions stay in `interaction_apply.cuh`, which only the seven interaction and drain units
// include (through `host/transport_run_int_impl.cuh`). The kernels take a device POINTER to the
// view - one copy, written by the builder - so no launching translation unit needs the view's
// size. The builder is defined beside the models (`host/interaction_pool_build.cuh`) and
// instantiated for double in exactly ONE translation unit of the engine
// (`src/host/transport_run_pool.cu`, archived into out/transport_run.lib) and in one of every
// hook project's archive (`tools/gen_hook_units.ps1` writes `hook_pool.cu`), because a hook
// project links its own archive and not the engine's. The `extern template` lines below are what
// keep `Upload` from instantiating the builder - and therefore the models' host code - in the
// object that calls it. docs/RISK.md V233 has the measurement: what a stepping unit reads
// before and after, and what one touched cascade header rebuilds.
#pragma once

#include <cstddef>

#include "data/level_data.cuh"

namespace g4gpu::had {

/// Everything ONE thread inside the interaction kernel needs, by value. Defined in
/// `physics/hadronic/interaction_apply.cuh`; the host sees its size through the owner below.
template <typename real_t>
struct InteractionSlot;

/// What an interaction or drain launch reads: the slots, the Binary cascade's channel table,
/// FTFP's pool and the Fermi break-up pool. Defined in `interaction_apply.cuh`; the host holds
/// ONE device copy of it and hands the kernels its address.
template <typename real_t>
struct InteractionPool;

/// What the host owns once `build_interaction_pool` has run: the device view's address, the
/// sizes the run reports, and - opaque, in the forms the builder needs to free them - what it
/// allocated. A default-constructed owner is "no pool", and `free_interaction_pool` of it is a
/// no-op, which is what `Free` on an engine that never uploaded relies on.
template <typename real_t>
struct InteractionPoolOwner {
  /// ONE device copy of the view, which every interaction and drain launch is handed.
  const InteractionPool<real_t>* d_pool = nullptr;
  int n_slots = 0;
  std::size_t slot_bytes = 0;           ///< `sizeof(InteractionSlot<real_t>)`
  std::size_t ftf_workspace_bytes = 0;  ///< `ftf::entry::kWorkspaceBytes`
  std::size_t slots_bytes = 0;          ///< `n_slots * slot_bytes`
  std::size_t ftf_bytes = 0;            ///< FTFP's pool: its slots and the one Lund table
  std::size_t channel_bytes = 0;        ///< the Binary cascade's concrete channels, one copy
  std::size_t fermi_bytes = 0;          ///< P3's Fermi break-up pool, one copy
  std::size_t bytes = 0;                ///< everything above plus the view's own copy
  bool ok = false;
  /// The builder's own. `ftf_owner` is a heap `ftf::entry::Owner<ftf::entry::Workspace>` and
  /// `fermi_owner` a heap `host::FermiPoolOwner`; both are types this header must not name.
  void* slots = nullptr;
  void* bic_channels = nullptr;
  void* ftf_owner = nullptr;
  void* fermi_owner = nullptr;
};

/// Allocates and uploads @p n_slots slots - ONE constructed host image per slot, never a memset
/// (docs/RISK.md V104) - the channel table, FTFP's pool through its own contract and the Fermi
/// break-up pool built from @p host_levels, and writes the one device view. Fatal, with the
/// reason printed, when any of it cannot be allocated or FTFP's pool cannot be built: there is
/// no run without them, and `Upload` exited on the same conditions before this contract.
template <typename real_t>
InteractionPoolOwner<real_t> build_interaction_pool(int n_slots,
                                                    const data::LevelTable& host_levels,
                                                    bool verbose);

/// Gives everything back and leaves @p own default-constructed.
template <typename real_t>
void free_interaction_pool(InteractionPoolOwner<real_t>& own);

// Instantiated for double in src/host/transport_run_pool.cu (the engine) and in the generated
// hook_pool.cu of every hook project's archive; these keep every other translation unit that
// calls them - every engine instantiation's `Upload` and `Free` - from instantiating the
// models' host code itself.
extern template InteractionPoolOwner<double> build_interaction_pool<double>(
    int, const data::LevelTable&, bool);
extern template void free_interaction_pool<double>(InteractionPoolOwner<double>&);

}  // namespace g4gpu::had
