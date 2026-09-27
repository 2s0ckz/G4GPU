// P19's drain for photonNuclear: `emextra::photon_nuclear`, which is G4LowEGammaNuclearModel
// into PreCompound and P3's evaporation below 200 MeV, Bertini's photon arm with
// G4LightTargetCollider from 199 MeV, and the QGS refusal from 3 GeV, behind one range manager.
//
// One of the transport engine's translation units, and in build_engine.bat's SERIAL pass with
// P15's five because it carries a hadronic model (the `transport_run_int_*` glob is what puts it
// there). One model family to a unit is docs/RISK.md V189's rule, and this is the photon's.
// Its registers, frame and ptxas cost are in docs/RISK.md V210 - measured, not estimated, and
// the frame is the number that matters most here: this kernel launches on every iteration of a
// photon run, so a frame past the stepping kernels' 16,384 bytes would make the driver raise
// the device stack for the whole card (V196) at the gamma gate's expense (V190).
#include "host/transport_run_impl.cuh"

namespace g4gpu::host {

template G4GPU_EMX_DRAIN(had::InteractionBucket::kPhotoNuclear, StepTap<double>);

}  // namespace g4gpu::host
