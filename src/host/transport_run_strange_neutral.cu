// P23a's neutral strange group kernel: K0S, K0L, lambda, sigma0, xi0, anti-lambda, anti-sigma0
// and anti-xi0, through `step_neutral_strange`.
//
// One of the transport engine's translation units, holding one stepping kernel and nothing
// else; the block under the kernels in transport_run_impl.cuh is what keeps it out of every
// other object (docs/RISK.md V65). A unit of its own rather than a third instantiation in
// transport_run_neutral.cu, whose neutron kernel carries the general process and the capture
// cascade this one has no use for - docs/RISK.md V240.
#include "host/transport_run_impl.cuh"

namespace g4gpu::host {

template G4GPU_STEP_STRANGE(false, StepTap<double>);

}  // namespace g4gpu::host
