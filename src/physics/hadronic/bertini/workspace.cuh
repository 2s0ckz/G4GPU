// The Bertini cascade's workspace: every buffer whose size a thread cannot afford on its stack.
//
// There is no Geant4 class behind this file. In Geant4 these are `std::vector` data members of
// G4NucleiModel, G4IntraNucleiCascader, G4ElementaryParticleCollider and G4CollisionOutput -
// `thePartners`, `theCascade`, `output`, `sigmaBuf`, `qdeutrons`, `acsecs`, `collisionPts` - all
// of them reused across calls rather than reallocated, and all of them unbounded. A device port
// has to give each one a capacity, and every capacity is a REFUSAL: the cascade reports that it
// ran out of room rather than dropping a particle.
//
// **Why it is a struct passed by pointer and not locals.** Measured, not assumed. Moving one
// 96-element `double` buffer out of `outgoing_particle_types` took a device probe from a
// 992-byte stack frame to 224 bytes under `-Xptxas -v`; the whole cascade's buffers together are
// tens of kilobytes, which no thread can hold. So the caller allocates one of these per thread
// (or per warp slot) in global memory and passes the pointer, exactly as `precompound.cuh`'s
// `PrecoWorkspace` is passed.
//
// The capacities below are the ones this package can justify; each says what bounds it and what
// happens when it is exceeded. Nothing is silently truncated - `CascadeOverflow` records which
// buffer ran out, and the interface turns that into a refused event rather than a short final
// state.
#ifndef G4GPU_BERTINI_WORKSPACE_CUH
#define G4GPU_BERTINI_WORKSPACE_CUH

#include "physics/hadronic/bertini/channel_tables.cuh"
#include "physics/hadronic/bertini/inucl_particle.cuh"
#include "physics/hadronic/bertini/lorentz_convertor.cuh"
#include "physics/hadronic/bertini/nuclei_model.cuh"

namespace g4gpu::physics::hadronic::bert {

/// One cascade particle: G4CascadParticle reduced to its data.
struct CascadeParticle {
  int type = 0;
  LV momentum;
  Vec3d position{0.0, 0.0, 0.0};
  int current_zone = -1;
  double current_path = -1.0;
  bool moving_in = false;
  int reflection_counter = 0;
  bool reflected = false;
  int generation = -1;
  int history_id = -1;
};

/// Which buffer ran out. Every one of these is reported by name, never absorbed.
enum class CascadeOverflow : int {
  kNone = 0,
  kCascadeStack,        ///< particles in flight
  kOutgoingParticles,   ///< G4CollisionOutput's elementary particles
  kOutgoingNuclei,      ///< G4CollisionOutput's fragments
  kCollisionPoints,     ///< the trailing-effect hit list
  kPartners,            ///< G4NucleiModel::thePartners
  kSigmaBuffer          ///< G4CascadeSampler::sigmaBuf, i.e. kMaxChannelsPerMult
};

/// Capacities.
///
/// `kMaxCascadeParticles` - the cascade stack. A 5 GeV proton on lead produces of order 100
/// cascade particles before de-excitation, and each collision replaces one particle with up to
/// nine; 512 is four times the largest count seen in the statistical campaign and is REPORTED
/// when reached rather than silently dropping the rest of the cascade. It is a refusal this
/// package names in PORTED.md because there is no bound in Geant4 to transcribe: `theCascade`
/// is a std::vector.
constexpr int kMaxCascadeParticles = 512;
/// G4CollisionOutput's two lists. The outgoing-particle list holds everything the cascade
/// emits plus everything de-excitation emits, and a heavy target evaporating from a few hundred
/// MeV of excitation contributes tens of nucleons and alphas.
constexpr int kMaxOutgoingParticles = 512;
constexpr int kMaxOutgoingNuclei = 64;
/// The trailing-effect hit list. With `radiusTrailing = 0` - the dumped default - nothing is
/// ever rejected by it, so the list is written and never read; the capacity therefore costs
/// nothing today and is here so that a run with the envvar set is reproducible.
constexpr int kMaxCollisionPoints = 512;
/// G4NucleiModel::thePartners: at most one per nucleon type (2) plus one quasi-deuteron plus
/// the total-path placeholder the generator pushes last. Four is the exact bound, not an
/// estimate - `generateInteractionPartners` pushes at most 2 + 1 + 1 entries.
constexpr int kMaxPartners = 4;

/// One partner: the target particle and the path at which it would be met.
struct InteractionPartner {
  int type = 0;
  LV momentum;
  double path = 0.0;
};

/// Everything a cascade needs that will not fit on a thread's stack.
struct BertiniWorkspace {
  /// G4CascadeSampler::sigmaBuf - the per-multiplicity cross-section buffer
  /// `outgoing_particle_types` fills. 96 doubles, which is the widest multiplicity block in the
  /// 34 tables (G4CascadeKminusPChannel's multiplicity 7); asserted against the data in
  /// tests/test_bertini_data.cu rather than taken on faith.
  double sigma_buf[kMaxChannelsPerMult];

  /// G4NucleiModel::thePartners, and the two parallel buffers the quasi-deuteron selection
  /// uses (`qdeutrons` and `acsecs`).
  InteractionPartner partners[kMaxPartners];
  int n_partners = 0;
  LV qdeutrons[3];
  int qdeutron_types[3];
  double acsecs[3];
  int n_qdeutrons = 0;

  /// G4IntraNucleiCascader::theCascade plus the next generation it builds.
  CascadeParticle cascade[kMaxCascadeParticles];
  int n_cascade = 0;

  /// G4NucleiModel::collisionPts.
  Vec3d collision_points[kMaxCollisionPoints];
  int n_collision_points = 0;

  /// Which capacity was exceeded, if any.
  CascadeOverflow overflow = CascadeOverflow::kNone;
};

__host__ __device__ inline void ws_reset(BertiniWorkspace& ws) {
  ws.n_partners = 0;
  ws.n_qdeutrons = 0;
  ws.n_cascade = 0;
  ws.n_collision_points = 0;
  ws.overflow = CascadeOverflow::kNone;
}

/// Push onto the cascade stack, reporting rather than dropping.
__host__ __device__ inline bool ws_push_cascade(BertiniWorkspace& ws,
                                                const CascadeParticle& p) {
  if (ws.n_cascade >= kMaxCascadeParticles) {
    ws.overflow = CascadeOverflow::kCascadeStack;
    return false;
  }
  ws.cascade[ws.n_cascade++] = p;
  return true;
}

__host__ __device__ inline bool ws_push_collision_point(BertiniWorkspace& ws,
                                                        const Vec3d& p) {
  if (ws.n_collision_points >= kMaxCollisionPoints) {
    ws.overflow = CascadeOverflow::kCollisionPoints;
    return false;
  }
  ws.collision_points[ws.n_collision_points++] = p;
  return true;
}

}  // namespace g4gpu::physics::hadronic::bert

#endif  // G4GPU_BERTINI_WORKSPACE_CUH
