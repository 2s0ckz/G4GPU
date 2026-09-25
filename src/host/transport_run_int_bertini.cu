// P15's interaction kernel for the Bertini cascade, 9,319 MB and 210 s.
//
// One of the transport engine's translation units. Each holds the explicit instantiation of one
// kernel and nothing else; the block under the kernels in transport_run_impl.cuh is what keeps
// that kernel out of every other object, and says why. docs/RISK.md V65.
//
// THESE FIVE ARE THE ONLY OBJECTS IN THE BUILD THAT CARRY A HADRONIC MODEL, and there are five
// of them rather than one because ptxas cannot compile four models into one module: the
// single-kernel version went past 22 GB of working set in 200 seconds and was still climbing.
// One at a time they compile, and the figure in this file's first line is what this one costs.
// docs/RISK.md V189 has the whole table and the reason nothing had ever found this before -
// every model test in this project is host-only.
#include "host/transport_run_impl.cuh"

namespace g4gpu::host {

template G4GPU_INTERACTION(had::InteractionBucket::kBertini, StepTap<double>);

}  // namespace g4gpu::host
