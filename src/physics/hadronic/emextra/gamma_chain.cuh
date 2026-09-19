// The buffers a photo-nuclear event needs, and the one configuration fact every caller of the
// gamma chain has to agree on.
//
// Three entry points in this package hand a photon to Bertini - `photon_nuclear` for the photon
// process, and `electro_vd_apply` and `muon_vd_apply` for the two lepton models - and all three
// need the same workspace and the same de-excitation choice. They are here rather than in one
// of the three so that `lepton_vd.cuh` does not have to include `photon_nuclear.cuh` to reach
// them, which would be a cycle: the photon process's model dispatch is not the lepton models'
// business and the lepton models are not the photon process's.
#ifndef G4GPU_HADRONIC_EMEXTRA_GAMMA_CHAIN_CUH
#define G4GPU_HADRONIC_EMEXTRA_GAMMA_CHAIN_CUH

#include "physics/hadronic/bertini/cascade_interface.cuh"
#include "physics/hadronic/emextra/config.cuh"
#include "physics/hadronic/precompound/precompound_model.cuh"

namespace g4gpu::physics::hadronic::emextra {

/// `G4CascadeInterface`'s de-excitation choice for every instance `G4EmExtraPhysics` builds -
/// the photon process's, the electron model's and the muon model's.
///
/// All three are plain `new G4CascadeInterface`, so the constructor's
/// `G4CascadeParameters::usePreCompound()` test - false on this install, see
/// `ref/oracle/bertini_params.csv` - leaves them on `useCascadeDeexcitation()`. Nothing in
/// `G4EmExtraPhysics` calls `usePreCompoundDeexcitation()`, which is what
/// `G4HadronInelasticQBBC::ConstructProcess` does for the nucleon and pion instances. So a
/// photo-nuclear cascade de-excites through `bertini/deexcite.cuh` and a proton-induced one
/// through P6, on the SAME residual - docs/PORTED.md 2.1.12 and docs/RISK.md V118 list three
/// Bertini instances in QBBC and there are six, of which three are this package's.
__host__ __device__ inline bert::DeexciteChoice qbbc_gamma_deexcite_choice() {
  return bert::DeexciteChoice::kCascade;
}

/// The buffers one photo-nuclear event needs. Every one of them is the caller's, allocated
/// once and passed by pointer: `BertiniWorkspace` alone is tens of kilobytes and none of this
/// can live on a kernel's stack.
///
/// `preco` is carried even though `qbbc_gamma_deexcite_choice()` is the cascade's own arm,
/// because `bert::apply_yourself` takes it unconditionally and because the low-energy gamma
/// model needs it: `G4LowEGammaNuclearModel` goes straight to `G4PreCompoundModel::DeExcite`
/// and is the one place in this package where P6 IS the de-excitation.
struct GammaWorkspace {
  bert::NucleiModel* model = nullptr;
  bert::CollisionOutput* global_out = nullptr;
  bert::CollisionOutput* out = nullptr;
  bert::CollisionOutput* dex_out = nullptr;
  bert::CollisionOutput* tmp = nullptr;
  bert::ColliderOutput* epo = nullptr;
  bert::BertiniWorkspace* bert_ws = nullptr;
  preco::PrecoWorkspace preco;

  __host__ __device__ bool complete() const {
    return model != nullptr && global_out != nullptr && out != nullptr && dex_out != nullptr
           && tmp != nullptr && epo != nullptr && bert_ws != nullptr
           && preco.products != nullptr;
  }
};

}  // namespace g4gpu::physics::hadronic::emextra

#endif
