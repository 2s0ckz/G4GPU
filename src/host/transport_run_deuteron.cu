// The stock deuteron kernel.
//
// One of the transport engine's translation units. Each holds the explicit instantiation of one
// stepping kernel and nothing else; the block under the kernels in transport_run_impl.cuh is
// what keeps that kernel out of every other object, and says why. docs/RISK.md V65.
#include "host/transport_run_impl.cuh"

namespace g4gpu::host {

template G4GPU_STEP_HADRON(ParticleType::kDeuteron, StepTap<double>);

}  // namespace g4gpu::host
