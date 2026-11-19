// P23a's charged strange group kernel: sigma+, sigma-, xi-, omega- and their four antiparticles.
//
// One of the transport engine's translation units, holding one stepping kernel and nothing
// else; the block under the kernels in transport_run_impl.cuh is what keeps it out of every
// other object (docs/RISK.md V65). It is ONE unit for eight species because nothing about the
// eight differs at compile time - docs/RISK.md V240 has the measurement that chose it over
// eight units, and the frame it compiles to.
#define G4GPU_ENGINE_KERNELS_ONLY  // a kernel unit: no engine host code here (P32)
#include "host/transport_run_impl.cuh"

namespace g4gpu::host {

template G4GPU_STEP_STRANGE(true, StepTap<double>);

}  // namespace g4gpu::host
