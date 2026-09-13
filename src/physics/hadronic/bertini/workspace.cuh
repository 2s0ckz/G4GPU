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

#include "physics/hadronic/bertini/angular_dist.cuh"
#include "physics/hadronic/bertini/channel_tables.cuh"
#include "physics/hadronic/bertini/inucl_particle.cuh"
#include "physics/hadronic/bertini/lorentz_convertor.cuh"
#include "physics/hadronic/bertini/nuclei_model.cuh"
#include "physics/hadronic/deexcitation/fragment.cuh"

namespace g4gpu::physics::hadronic::bert {

/// The state G4CascadeFinalStateAlgorithm::Configure builds and its generators then read.
///
/// It is here rather than beside the code that uses it because it is 264 bytes of arrays, and
/// because in Geant4 it is not a per-call object at all: `kinds`, `masses`, `masses2`,
/// `modules`, `toSCM` and `bullet_ekin` are data members of G4CascadeFinalStateAlgorithm, and
/// `particle_kinds` and `masses` of G4ElementaryParticleCollider, allocated once per thread and
/// reused for every collision of the run.
struct FinalStateConfig {
  int multiplicity = 0;
  int kinds[kMaxFinalStateSize];
  double masses[kMaxFinalStateSize];
  double masses2[kMaxFinalStateSize];
  double modules[kMaxFinalStateSize];
  LorentzConvertor to_scm;
  double bullet_ekin = 0.0;
  AngDstChoice ang;          ///< kNone when ChooseGenerators leaves angDist null
  int mom_index = -1;        ///< -1 when usePhaseSpace switches momDist off
  bool use_phase_space = false;
};

/// One cascade particle: G4CascadParticle reduced to its data.
///
/// The defaults are G4CascadParticle's DEFAULT constructor, which its own comment calls
/// "non-functional": zone -1, path -1, `movingIn` false, generation -1. The five-argument
/// constructor is a different set of defaults - `movingIn(true)`, `reflectionCounter(0)`,
/// `reflected(false)`, `historyId(-1)` - and that one is what every real cascade particle is
/// built with, so it is `cp_fill` below and not this struct's initialisers. Getting `moving_in`
/// wrong here is invisible until the first boundary: `boundaryTransition` reads it to choose
/// `zone-1` or `zone+1`, so a particle created with the struct default would transmit OUTWARD
/// on its first crossing instead of inward.
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

/// G4CascadParticle's five-argument constructor, and `fill()`, which are the same assignment.
__host__ __device__ inline CascadeParticle cp_fill(int type, const LV& mom, const Vec3d& pos,
                                                   int izone, double cpath, int gen) {
  CascadeParticle c;
  c.type = type;
  c.momentum = mom;
  c.position = pos;
  c.current_zone = izone;
  c.current_path = cpath;
  c.moving_in = true;        // movingIn(true) in the constructor, NOT the struct default
  c.reflection_counter = 0;
  c.reflected = false;
  c.generation = gen;
  c.history_id = -1;
  return c;
}

/// Which buffer ran out. Every one of these is reported by name, never absorbed.
enum class CascadeOverflow : int {
  kNone = 0,
  kCascadeStack,        ///< particles in flight
  kOutgoingParticles,   ///< G4CollisionOutput's elementary particles
  kOutgoingNuclei,      ///< G4CollisionOutput's fragments
  kCollisionPoints,     ///< the trailing-effect hit list
  kPartners,            ///< G4NucleiModel::thePartners
  kSigmaBuffer,         ///< G4CascadeSampler::sigmaBuf, i.e. kMaxChannelsPerMult
  kBigBangNucleons,     ///< G4BigBanger::momModules - a fragment with more nucleons than
                        ///< kMaxBangA, which cannot be exploded without dropping some
  kDeexStack            ///< the fission work list: more than kMaxDeexStack levels
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

  /// G4CascadeFinalStateAlgorithm's and G4ElementaryParticleCollider's per-collision state.
  /// Worth 264 bytes of a thread's stack frame if it is a local instead - measured, see the
  /// probe in tests/test_bertini_collide.cu.
  FinalStateConfig fs_cfg;

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

  /// G4IntraNucleiCascader::new_cascad_particles - the buffer `generateParticleFate` fills.
  /// Bounded exactly: a fate is either one propagated particle or the products of ONE
  /// two-body collision, which a channel table caps at `kMaxFinalStateSize` (9).
  CascadeParticle new_cascade[kMaxFinalStateSize];
  int n_new_cascade = 0;

  /// G4NucleiModel::collisionPts.
  Vec3d collision_points[kMaxCollisionPoints];
  int n_collision_points = 0;

  /// G4NucleiModel::coordinates, momentums and raw_particles - the three buffers the
  /// NUCLEUS-projectile `initializeCascad` builds its nucleons in. Four entries exactly:
  /// the routine's own `max_a_for_cascad` is 5 and the test is `ab < max_a_for_cascad`, so a
  /// bullet of A = 4 is the largest that ever gets here and A >= 5 produces a compound nucleus
  /// with no cascade particles at all.
  static constexpr int kMaxIonBulletA = 4;
  Vec3d ion_coordinates[kMaxIonBulletA];
  LV ion_momenta[kMaxIonBulletA];
  int n_ion_coordinates = 0;
  int n_ion_momenta = 0;

  /// G4BigBanger::momModules and ::scm_momentums - one entry per nucleon of the exploding
  /// fragment. `G4CascadeDeexciteBase::explosion` lets a fragment through when `A <= 20` OR
  /// `Z == 0`, so the bound is not 20: a pure neutron ball of any mass number qualifies, and
  /// `G4BigBanger::deExcite` is a public entry point that will explode ANY fragment it is
  /// handed - `ref/oracle/bertini_deexcite.csv` asks it for lead. 256 covers every nuclide in
  /// the mass table, so the refusal below is unreachable rather than merely unlikely; the cost
  /// is 10 kB of a per-thread workspace that already holds 512 cascade particles.
  static constexpr int kMaxBangA = 256;
  double bang_modules[kMaxBangA];
  LV bang_momenta[kMaxBangA];
  /// The parallel arrays the big bang's descending-kinetic-energy sort needs: the type is a
  /// function of the INDEX (the first Z are protons), so it has to move with the momentum.
  /// Measured: as locals these two cost 3 kB of a thread's stack frame on their own.
  int bang_kinds[kMaxBangA];
  double bang_ekin[kMaxBangA];

  /// G4FissionStore's candidate list and its cumulative probabilities. The bound is exact, not
  /// an estimate: `G4Fissioner::deExcite`'s loop is `for (i = 0; i < 50 && A1 > 30; i++)`.
  static constexpr int kMaxFissionConfigs = 50;
  double fission_afirst[kMaxFissionConfigs];
  double fission_zfirst[kMaxFissionConfigs];
  double fission_ezet[kMaxFissionConfigs];
  double fission_ekin[kMaxFissionConfigs];
  double fission_probs[kMaxFissionConfigs];

  /// The de-excitation work list. `G4EquilibriumEvaporator::deExcite` calls ITSELF on each of
  /// the two fission fragments; a device kernel does not recurse, so the fragments go on this
  /// stack instead. The depth is bounded by the fission threshold: a fragment must have A >= 100
  /// to fission at all, so a uranium residual splits into two ~120 halves, each of which can
  /// split once more into ~60 - and 60 cannot. Eight slots is far more than the three levels
  /// that arithmetic allows.
  static constexpr int kMaxDeexStack = 8;
  deex::Fragment deex_stack[kMaxDeexStack];
  int n_deex_stack = 0;

  /// G4CascadeCoalescence::clusters / usedNucleons - the candidate list the four nested index
  /// loops build and the mark set that strikes a nucleon out of every later combination. The
  /// bound is one cluster per two nucleons because the smallest cluster is a deuteron.
  ///
  /// MEASURED, and the two numbers disagree for a reason worth writing down. A probe over
  /// `cascader_collide` alone - the cascade arm, which reaches coalescence and never reaches
  /// precompound - falls from a 6,672-byte stack frame to 1,232 when these three move here, the
  /// 5,440 bytes they declare (256 x 4 ints + 256 ints + 512 bools, rounded up). In the FULL
  /// apply probe the same move is worth 16 bytes: ptxas overlaps disjoint live ranges, and the
  /// precompound arm's own 10,160-byte frame (measured on a probe that calls nothing else) is
  /// the larger of the two, so the union does not move. The cascade-only number is the honest
  /// one, because a kernel that runs Bertini's own de-excitation never allocates precompound's.
  static constexpr int kMaxCoalClusters = kMaxOutgoingParticles / 2;
  int coal_cand_idx[kMaxCoalClusters][4];
  int coal_cand_n[kMaxCoalClusters];
  bool coal_used[kMaxOutgoingParticles];

  /// Which capacity was exceeded, if any.
  CascadeOverflow overflow = CascadeOverflow::kNone;
};

__host__ __device__ inline void ws_reset(BertiniWorkspace& ws) {
  ws.n_partners = 0;
  ws.n_qdeutrons = 0;
  ws.n_cascade = 0;
  ws.n_new_cascade = 0;
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
