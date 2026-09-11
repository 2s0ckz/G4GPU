// The stock e- and e+ kernels.
//
// One of the transport engine's translation units; the block under the kernels in
// transport_run_impl.cuh is what keeps them out of every other object, and says why.
//
// TWO INSTANTIATIONS IN ONE UNIT, WHERE THE HADRON KERNELS GET ONE EACH, and the reason is
// measurement rather than taste: `run_step_lepton<real_t, kIsPositron, StepHook>` is one
// template switched by a compile-time bool, the pair compiles in 269 s, and it is not the
// build's long pole. Four instantiations of `run_step_hadron` in one unit is the arrangement
// ptxas refuses outright, which is why those are thirteen units. docs/RISK.md V65.
#include "host/transport_run_impl.cuh"

namespace g4gpu::host {

template G4GPU_STEP_LEPTON(false, StepTap<double>);
template G4GPU_STEP_LEPTON(true, StepTap<double>);

}  // namespace g4gpu::host
