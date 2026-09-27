// P19's drain for electronNuclear, positronNuclear and muonNuclear: `emextra::electron_nuclear`
// and `emextra::muon_nuclear` - G4ElectroVDNuclearModel and G4MuonVDNuclearModel, each handing
// its equivalent photon to its own G4CascadeInterface (the cascade's own de-excitation, so no
// PreCompound here at all) below 10 GeV and refusing the pi0-into-FTF arm above it by name.
//
// One of the transport engine's translation units, in build_engine.bat's serial pass beside
// P15's five for the reason photonuclear's gives. The three processes share one unit because
// the two entry points share the gamma chain: P13's lepton probe instantiated both together at
// 255 registers and a 2,544-byte frame. docs/RISK.md V210 has what this kernel measures.
#include "host/transport_run_impl.cuh"

namespace g4gpu::host {

template G4GPU_EMX_DRAIN(had::InteractionBucket::kLeptoNuclear, StepTap<double>);

}  // namespace g4gpu::host
