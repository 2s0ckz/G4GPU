// The stock neutral-hadron kernels: the neutron and the pi0.
//
// One of the transport engine's translation units; the block under the kernels in
// transport_run_impl.cuh is what keeps them out of every other object, and says why.
//
// Two instantiations in one unit, for the reason transport_run_lepton.cu gives: they are one
// template switched by a compile-time `ParticleType`, and the pair compiles in 260 s. This is
// also the unit docs/RISK.md V55 warned about when it said the neutron general process "drags
// the whole de-excitation chain into this file" - G4NeutronGeneralProcess's five combined
// tables, the capture cascade and PhotonEvaporation5.7's level scheme, through
// physics/hadronic/neutron_wiring.cuh. That chain is now compiled for two kernels rather than
// alongside sixteen others, and the neutron's is the largest stack frame in the engine at
// 7472 bytes against the 16384 `Upload` sets. docs/RISK.md V65.
#include "host/transport_run_impl.cuh"

namespace g4gpu::host {

template G4GPU_STEP_NEUTRAL(ParticleType::kNeutron, StepTap<double>);
template G4GPU_STEP_NEUTRAL(ParticleType::kPiZero, StepTap<double>);

}  // namespace g4gpu::host
