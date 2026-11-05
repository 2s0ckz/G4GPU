// The interaction pool's builder, instantiated for double.
//
// THE ONE OBJECT IN THE ENGINE THAT CARRIES THE MODELS' HOST CODE: the constructed slot image
// (`had::InteractionSlot`'s constructor, which is Bertini's, the Binary cascade's and the
// de-excitation buffers' constructors), the Binary cascade's concrete channel table, FTFP's pool
// through `ftf::entry::build`, and P3's Fermi break-up pool. Until docs/RISK.md V233 all of it
// was in `TransportEngine::Upload`, so every translation unit that instantiated the engine or a
// kernel of it read every model header. Now `physics/hadronic/interaction_pool.cuh` declares the
// builder, this unit defines it, and `extern template` keeps it out of everything else.
//
// It carries NO KERNEL. build_engine.bat compiles it like any stepping unit - the glob picks it
// up, six at a time - and its ptxas has nothing to do; what it costs is the host compile of the
// model headers, once, when one of them changes. A hook project gets the same unit from
// tools/gen_hook_units.ps1 (`hook_pool.cu`), because it links its own archive and not this one.
#include "host/interaction_pool_build.cuh"

namespace g4gpu::had {

template InteractionPoolOwner<double> build_interaction_pool<double>(int, const data::LevelTable&,
                                                                     bool);
template void free_interaction_pool<double>(InteractionPoolOwner<double>&);

}  // namespace g4gpu::had
