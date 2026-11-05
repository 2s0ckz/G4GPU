// The interaction pool's builder: everything `TransportEngine::Upload` did with the models' host
// code before docs/RISK.md V233, behind the contract physics/hadronic/interaction_pool.cuh
// declares - and P3's Fermi break-up pool's upload, which moved here from
// host/hadronic_upload.cuh because only the interaction kernels read it and its header is a
// model's.
//
// INCLUDED BY src/host/transport_run_pool.cu AND BY THE GENERATED hook_pool.cu OF EVERY HOOK
// PROJECT'S ARCHIVE, which instantiate the two functions for double. Nothing that launches a
// kernel includes it: that is the point. A test that wants the pool on the host includes it and
// instantiates what it uses.
#pragma once

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "data/level_data.cuh"
#include "physics/hadronic/deexcitation/fermi_breakup.cuh"
#include "physics/hadronic/interaction_apply.cuh"
#include "physics/hadronic/interaction_pool.cuh"

namespace g4gpu::host {

// =============================================================================================
// P3's Fermi break-up pool, on the device
// =============================================================================================
//
// `G4FermiFragmentsPoolVI::Initialise`'s output: the fragment list, the per-A index, the
// channel sets and their cumulative probabilities. `deex::build_fermi_pool` builds it on the
// host from the level scheme, and every arm of the inelastic chain walks it - the Binary
// cascade's de-excitation, Bertini's, FTFP's and the at-rest captures' all end in P3's
// `G4ExcitationHandler`, whose Fermi break-up branch reads this.
//
// Nine arrays and no pointers inside them, so uploading it is nine memcpys and a view whose
// pointers are repointed - the same shape `detail::upload_pxs` has for a cross-section table.
//
// BUILT FROM THE HOST LEVEL SCHEME, which means a run with `SetNuclearLevelData(false)` gets an
// EMPTY pool. That is not silent: `deex::fermi_break_up` with no fragments reports its own
// refusal, and `Upload` already refuses the combination of a neutron cross section with no
// level scheme (docs/RISK.md V60). A run that can make no hadron pays nothing for this either
// way, because nothing reads it.

struct FermiPoolOwner {
  deex::FermiPool view{};
  std::vector<void*> allocs;
  std::size_t bytes = 0;
};

inline FermiPoolOwner upload_fermi_pool(const data::LevelTable& host_levels,
                                        bool verbose = true) {
  FermiPoolOwner own;
  auto* st = new deex::FermiPoolStorage();
  deex::build_fermi_pool(*st, host_levels);

  auto up = [&own](const void* src, std::size_t n) -> void* {
    if (n == 0) { return nullptr; }
    void* p = nullptr;
    if (cudaMalloc(&p, n) != cudaSuccess
        || cudaMemcpy(p, src, n, cudaMemcpyHostToDevice) != cudaSuccess) {
      std::printf("\nFATAL: could not upload %zu bytes of the Fermi break-up pool\n", n);
      std::exit(1);
    }
    own.allocs.push_back(p);
    own.bytes += n;
    return p;
  };

  deex::FermiPool v = st->view();
  v.frag = static_cast<const deex::FermiFragment*>(
      up(st->frag.data(), st->frag.size() * sizeof(deex::FermiFragment)));
  v.by_a = static_cast<const int*>(up(st->by_a.data(), st->by_a.size() * sizeof(int)));
  v.by_a_offset =
      static_cast<const int*>(up(st->by_a_offset.data(), st->by_a_offset.size() * sizeof(int)));
  v.by_a_count =
      static_cast<const int*>(up(st->by_a_count.data(), st->by_a_count.size() * sizeof(int)));
  v.ch_offset =
      static_cast<const int*>(up(st->ch_offset.data(), st->ch_offset.size() * sizeof(int)));
  v.ch_count =
      static_cast<const int*>(up(st->ch_count.data(), st->ch_count.size() * sizeof(int)));
  v.pairs = static_cast<const deex::FermiPair*>(
      up(st->pairs.data(), st->pairs.size() * sizeof(deex::FermiPair)));
  v.pair_of_channel = static_cast<const int*>(
      up(st->pair_of_channel.data(), st->pair_of_channel.size() * sizeof(int)));
  v.cum_prob = static_cast<const double*>(
      up(st->cum_prob.data(), st->cum_prob.size() * sizeof(double)));
  own.view = v;

  if (verbose) {
    std::printf("Fermi break-up pool: %d fragments, %d channels, %d pairs, %.2f MB\n",
                v.n_frag, v.n_channels, v.n_pairs, double(own.bytes) / 1048576.0);
  }
  delete st;
  return own;
}

inline void free_fermi_pool(FermiPoolOwner& own) {
  for (void* p : own.allocs) { cudaFree(p); }
  own = FermiPoolOwner{};
}

}  // namespace g4gpu::host

namespace g4gpu::had {

/// A device allocation or copy the pool cannot do without. Printed and fatal, as `Upload`'s
/// G4GPU_CUDA_CHECK was for the same calls before V233: a pool with a slot missing is a kernel
/// reading unmapped memory, not a smaller pool.
inline void interaction_pool_fatal(const char* what, std::size_t bytes, cudaError_t e) {
  std::printf("\nFATAL: the interaction pool could not %s (%zu bytes): %s\n", what, bytes,
              cudaGetErrorString(e));
  std::exit(1);
}

template <typename real_t>
InteractionPoolOwner<real_t> build_interaction_pool(int n_slots,
                                                    const data::LevelTable& host_levels,
                                                    bool verbose) {
  InteractionPoolOwner<real_t> own;
  own.n_slots = n_slots;
  own.slot_bytes = sizeof(InteractionSlot<real_t>);
  own.ftf_workspace_bytes = ftfe::kWorkspaceBytes;

  // The slots. ONE CONSTRUCTED HOST IMAGE, uploaded into every slot, and not a memset:
  // docs/RISK.md V104 is a member that only a constructor sets and that decides physics when it
  // is zero, and `bert::NucleiModel` and `bic::CascadeBuffers` are the same shape of risk.
  // `ftf::entry::build` does exactly this for its own pool and says so.
  own.slots_bytes = own.slot_bytes * static_cast<std::size_t>(n_slots);
  if (cudaError_t e = cudaMalloc(&own.slots, own.slots_bytes); e != cudaSuccess) {
    interaction_pool_fatal("allocate its slots", own.slots_bytes, e);
  }
  {
    auto* image = new InteractionSlot<real_t>();
    for (int i = 0; i < n_slots; ++i) {
      if (cudaError_t e = cudaMemcpy(static_cast<InteractionSlot<real_t>*>(own.slots) + i, image,
                                     sizeof(*image), cudaMemcpyHostToDevice);
          e != cudaSuccess) {
        interaction_pool_fatal("upload a slot's image", sizeof(*image), e);
      }
    }
    delete image;
  }

  // The Binary cascade's 306 concrete channels: read-only, identical for every thread, one copy
  // for the whole run - the same relationship `LundTables` has to FTFP's pool.
  int n_channels = 0;
  {
    auto* ch = new bic::imr::ConcreteChannel[bic::imr::kConcreteChannelCount];
    n_channels = bic::imr::build_concrete_channels(ch, bic::imr::kConcreteChannelCount);
    own.channel_bytes = sizeof(bic::imr::ConcreteChannel) * static_cast<std::size_t>(n_channels);
    if (cudaError_t e = cudaMalloc(&own.bic_channels, own.channel_bytes); e != cudaSuccess) {
      interaction_pool_fatal("allocate the cascade's channel table", own.channel_bytes, e);
    }
    if (cudaError_t e =
            cudaMemcpy(own.bic_channels, ch, own.channel_bytes, cudaMemcpyHostToDevice);
        e != cudaSuccess) {
      interaction_pool_fatal("upload the cascade's channel table", own.channel_bytes, e);
    }
    delete[] ch;
  }

  // FTFP's pool, through its own contract. Same slot count, same thread index - and
  // `entry::Workspace` rather than `entry::HadronWorkspace` because a GenericIon above
  // 3 GeV per nucleon goes to FTFP and the hadron-only type refuses an ion by capacity.
  auto* ftf = new ftfe::Owner<ftfe::Workspace>(
      ftfe::build<ftfe::Workspace>(n_slots, /*enable_bc_particles=*/true, /*verbose=*/false));
  if (!ftf->view.ok()) {
    std::printf("\nFATAL: FTFP's workspace pool (%d slots) could not be built. The "
                "inelastic\n       process cannot run without it; see "
                "ftf/ftf_entry.cuh.\n", n_slots);
    std::exit(1);
  }
  own.ftf_owner = ftf;
  own.ftf_bytes = ftf->bytes;

  // P3's Fermi break-up table, which every arm's de-excitation tail walks. Built on the host
  // from the level scheme that is already there, and uploaded once.
  auto* fermi =
      new host::FermiPoolOwner(host::upload_fermi_pool(host_levels, /*verbose=*/false));
  own.fermi_owner = fermi;
  own.fermi_bytes = fermi->bytes;

  // The view the kernels read: ONE device copy, so that a launch carries its address and no
  // launching translation unit needs the view's definition.
  InteractionPool<real_t> view;
  view.slots = static_cast<InteractionSlot<real_t>*>(own.slots);
  view.n_slots = n_slots;
  view.bic_channels = static_cast<const bic::imr::ConcreteChannel*>(own.bic_channels);
  view.n_bic_channels = n_channels;
  view.ftf = ftf->view;
  view.fermi = fermi->view;
  InteractionPool<real_t>* d_view = nullptr;
  if (cudaError_t e = cudaMalloc(&d_view, sizeof(view)); e != cudaSuccess) {
    interaction_pool_fatal("allocate its view", sizeof(view), e);
  }
  if (cudaError_t e = cudaMemcpy(d_view, &view, sizeof(view), cudaMemcpyHostToDevice);
      e != cudaSuccess) {
    interaction_pool_fatal("upload its view", sizeof(view), e);
  }
  own.d_pool = d_view;
  own.bytes =
      own.slots_bytes + own.channel_bytes + own.ftf_bytes + own.fermi_bytes + sizeof(view);
  own.ok = true;
  if (verbose) {
    std::printf("interaction pool: %d slots of %zu B, FTFP's %zu B, %d cascade channels %zu B, "
                "Fermi pool %zu B, view %zu B: %.1f MB\n",
                n_slots, own.slot_bytes, own.ftf_bytes, n_channels, own.channel_bytes,
                own.fermi_bytes, sizeof(view), double(own.bytes) / 1048576.0);
  }
  return own;
}

template <typename real_t>
void free_interaction_pool(InteractionPoolOwner<real_t>& own) {
  // `ftf::entry::free` is FTFP's own, for the allocations its contract made, and
  // `free_fermi_pool` the Fermi pool's; the rest is this file's. A default-constructed owner has
  // nothing to give back and every call below is a no-op on null.
  if (own.ftf_owner != nullptr) {
    auto* ftf = static_cast<ftfe::Owner<ftfe::Workspace>*>(own.ftf_owner);
    ftfe::free<ftfe::Workspace>(*ftf);
    delete ftf;
  }
  if (own.fermi_owner != nullptr) {
    auto* fermi = static_cast<host::FermiPoolOwner*>(own.fermi_owner);
    host::free_fermi_pool(*fermi);
    delete fermi;
  }
  cudaFree(own.slots);
  cudaFree(own.bic_channels);
  cudaFree(const_cast<InteractionPool<real_t>*>(own.d_pool));
  own = InteractionPoolOwner<real_t>{};
}

}  // namespace g4gpu::had
