// G4BinaryCascade's collision finding and G4BinaryCascade::ApplyCollision.
//
// Transcribed from G4BinaryCascade.cc, G4Scatterer::GetCollisions and the three G4BCAction
// implementations (11.1.1). This is the other half of the cascade loop: the scheduler that asks
// every action what could happen to a track, and the bookkeeping that runs one of those and puts
// the results back into the lists.
//
// ## THE ORDER OF THE THREE ACTIONS IS THE ORDER THE CONSTRUCTOR PUSHES THEM
//
//     theDecay = new G4BCDecay;            theImR.push_back(theDecay);     // 0
//     G4MesonAbsorption * aAb = ...;       theImR.push_back(aAb);          // 1
//     G4Scatterer * aSc = new G4Scatterer; theImR.push_back(aSc);          // 2
//
// and `theLateParticle` is NOT in that vector - only `FindLateParticleCollision` uses it. The
// order matters for the random stream and not for the physics: `G4BCDecay::GetCollisions` draws
// one uniform for `SampleResidualLifetime` and the other two draw none, so a port that asked the
// scatterer first would put every later random in the event one place out of step.
//
// ## A RESONANCE CANNOT SCATTER, AND THAT FALLS OUT OF TWO `IsInCharge`s
//
// `G4Scatterer::FindCollision` tries `G4CollisionNN` then `G4CollisionMesonBaryon`. The first
// needs TWO NUCLEONS (`G4GeneralNNCollision::IsInCharge`); the second needs a two-parton track
// and a three-parton one, and a resonance has three. So a Delta or an N* with a nucleon target
// is in charge of NEITHER, `FindCollision` returns null, and the only thing that can happen to a
// resonance inside the nucleus is its own decay. That is what bounds the buffer cache below to
// the four nucleon pairs and the pion-nucleon ones.
//
// ## REFUSED, by name
//
//   * nothing new. `G4MesonAbsorption`'s final state is in `im_r/absorption.cuh` and its
//     `G4Absorber` half is refused there, with the constant that makes it unreachable measured.
#ifndef G4GPU_BIC_CASCADE_FIND_CUH
#define G4GPU_BIC_CASCADE_FIND_CUH

#include <cfloat>
#include <cmath>

#include "physics/hadronic/bic/cascade_step.cuh"
#include "physics/hadronic/bic/im_r/absorption.cuh"

namespace g4gpu::bic {

/// Which action a scheduled collision came from - `G4CollisionInitialState::GetGenerator()`, as
/// a tag rather than a pointer. The three values 0..2 are `theImR`'s order.
enum CollisionGenerator : int {
  kGenDecay = 0,
  kGenAbsorption = 1,
  kGenScatterer = 2,
  kGenLateParticle = 3
};

/// The cross-section buffers `G4CollisionComposite` caches, one entry per pair of particle
/// DEFINITIONS, owned by the caller because a device kernel cannot allocate one inside the call.
///
/// The set is bounded by the two `IsInCharge`s named in the file header: the only pairs that
/// reach a composite are (nucleon, nucleon) - four of them - and (pion, nucleon) - six. Nothing
/// else in the cascade is in charge of anything, so nothing else needs a buffer.
inline constexpr int kNNPairs = 4;
inline constexpr int kMesonPairs = 6;

struct CascadeBuffers {
  imr::NNChannelBuffers nn[kNNPairs];
  int nn_pdg1[kNNPairs] = {};
  int nn_pdg2[kNNPairs] = {};
  bool nn_built[kNNPairs] = {};
  imr::MesonBaryonBuffers meson[kMesonPairs];
  int meson_pion[kMesonPairs] = {};
  int meson_baryon[kMesonPairs] = {};
  bool meson_built[kMesonPairs] = {};

  __host__ __device__ int find_nn(int pdg1, int pdg2) const {
    for (int i = 0; i < kNNPairs; ++i) {
      if (nn_built[i] && nn_pdg1[i] == pdg1 && nn_pdg2[i] == pdg2) { return i; }
    }
    return -1;
  }
  __host__ __device__ int find_meson(int pion, int baryon) const {
    for (int i = 0; i < kMesonPairs; ++i) {
      if (meson_built[i] && meson_pion[i] == pion && meson_baryon[i] == baryon) { return i; }
    }
    return -1;
  }
};

/// `G4CollisionComposite::BufferCrossSection`'s lazy build, for whichever pair is asked about.
///
/// Geant4 builds the buffer inside `CrossSection` under `bufferMutex` the first time it sees a
/// pair and keeps it for the life of the composite - which, because the composites are owned by
/// a static `G4Scatterer::collisions`, is the life of the PROCESS. The port builds it on first
/// use here and keeps it for the event, which is the same numbers and a lifetime a kernel can
/// have (docs/RISK.md V155 is what the process-wide version costs).
__host__ __device__ inline int ensure_nn_buffer(CascadeBuffers& buf,
                                                const imr::ConcreteChannel* chans, int n_chan,
                                                int pdg1, int pdg2, double m1, double m2,
                                                CascadeRefusal& ref) {
  const int have = buf.find_nn(pdg1, pdg2);
  if (have >= 0) { return have; }
  for (int i = 0; i < kNNPairs; ++i) {
    if (buf.nn_built[i]) { continue; }
    imr::ResonanceTableRefusal tref;
    imr::build_nn_channel_buffers(chans, n_chan, pdg1, pdg2, m1, m2, buf.nn[i], tref);
    buf.nn_pdg1[i] = pdg1;
    buf.nn_pdg2[i] = pdg2;
    buf.nn_built[i] = true;
    return i;
  }
  ref.capacity = true;
  return -1;
}

__host__ __device__ inline int ensure_meson_buffer(CascadeBuffers& buf, int pion, int baryon,
                                                   double m_pion, double m_baryon, int iso3_pion,
                                                   int iso3_baryon, double m_pi_plus,
                                                   double proton_mass, CascadeRefusal& ref) {
  const int have = buf.find_meson(pion, baryon);
  if (have >= 0) { return have; }
  for (int i = 0; i < kMesonPairs; ++i) {
    if (buf.meson_built[i]) { continue; }
    imr::AnnihRefusal aref;
    imr::build_meson_baryon_buffers(pion, baryon, m_pion, m_baryon, iso3_pion, iso3_baryon,
                                    m_pi_plus, proton_mass, buf.meson[i], aref);
    buf.meson_pion[i] = pion;
    buf.meson_baryon[i] = baryon;
    buf.meson_built[i] = true;
    if (aref.unknown_resonance) {
      ref.unknown_species = true;
      ref.refused_pdg = aref.refused_pdg;
    }
    return i;
  }
  ref.capacity = true;
  return -1;
}

/// The masses and isospins the cascade needs for a species, in one place so that the three
/// callers below cannot disagree about them.
struct CascadeSpecies {
  double proton_mass = 0.0;
  double neutron_mass = 0.0;
  double pi_plus_mass = 0.0;
  double pi_zero_mass = 0.0;

  __host__ __device__ double mass_of(int pdg) const {
    if (pdg == imr::kPdgProton) { return proton_mass; }
    if (pdg == imr::kPdgNeutron) { return neutron_mass; }
    if (pdg == 111) { return pi_zero_mass; }
    return pi_plus_mass;
  }
  __host__ __device__ static int iso3_of(int pdg) {
    if (pdg == imr::kPdgProton) { return 1; }
    if (pdg == imr::kPdgNeutron) { return -1; }
    if (pdg == imr::kPdgPiPlus) { return 2; }
    if (pdg == imr::kPdgPiMinus) { return -2; }
    return 0;
  }
  __host__ __device__ static bool is_pion(int pdg) {
    return pdg == imr::kPdgPiPlus || pdg == imr::kPdgPiMinus || pdg == 111;
  }
};

/// `G4BCDecay::GetCollisions` for one track - the only one of the three that draws a uniform.
///
/// Adds at most one collision, at `theCurrentTime + SampleResidualLifetime()`, with NO target.
template <typename Rng>
__host__ __device__ inline void schedule_decay(BicCascadeState& st, imr::CollisionList& colls,
                                               int index, Rng& rng, KineticDecayRefusal& dref) {
  const CascadeTrack& t = st.lists.pool[index];
  if (!kinetic_decay_is_short_lived(t.pdg)) { return; }
  const double life = kinetic_decay_residual_lifetime(t.pdg, t.momentum, rng, dref);
  if (life < 0.0) { return; }
  imr::ScatterRefusal cref;
  colls.add(st.current_time + life, index, -1, kGenDecay, cref);
}

/// `G4MesonAbsorption::GetCollisions` for one track over the target list.
///
/// `G4MesonAbsorption` needs at least two candidates and a cluster partner; the partner search
/// is `im_r/absorption.cuh`'s, with its `|r1 + r2|` minimum (docs/RISK.md V154).
template <typename Prop>
__host__ __device__ inline void schedule_absorption(BicCascadeState& st,
                                                    imr::CollisionList& colls, int index,
                                                    const Prop& propagator,
                                                    imr::AbsorptionRefusal& aref) {
  const CascadeTrack& pro = st.lists.pool[index];
  // The candidate list is theTargetList, so a pion is always the PROJECTILE here - which is why
  // the `!=` of docs/RISK.md V153 is latent and not live.
  int n_cand = 0;
  for (int i = 0; i < st.lists.n_pool; ++i) {
    if (st.lists.pool[i].list == kListTarget) { ++n_cand; }
  }
  if (n_cand <= 1) { return; }
  for (int i = 0; i < st.lists.n_pool; ++i) {
    const CascadeTrack& tgt = st.lists.pool[i];
    if (tgt.list != kListTarget) { continue; }
    const double time = imr::time_to_absorption(pro.pdg, tgt.pdg, pro.charge, tgt.charge,
                                                pro.position, tgt.position, pro.momentum,
                                                pro.momentum, tgt.momentum, pro.actual_mass(),
                                                tgt.actual_mass(), aref);
    if (time == DBL_MAX) { continue; }
    // FindAndFillCluster over the same candidates, by pool index.
    int charges[256];
    deex::Vec3d positions[256];
    int ids[256];
    int n = 0;
    int first_slot = -1;
    for (int j = 0; j < st.lists.n_pool && n < 256; ++j) {
      if (st.lists.pool[j].list != kListTarget) { continue; }
      charges[n] = st.lists.pool[j].charge;
      positions[n] = st.lists.pool[j].position;
      ids[n] = j;
      if (j == i) { first_slot = n; }
      ++n;
    }
    if (first_slot < 0) { continue; }
    const int partner = imr::find_and_fill_cluster(pro.charge, tgt.charge, tgt.position, charges,
                                                   positions, n, first_slot);
    if (partner < 0) { continue; }
    imr::ScatterRefusal cref;
    colls.add(time + st.current_time, index, i, kGenAbsorption, cref, ids[partner]);
  }
  (void)propagator;
}

/// `G4Scatterer::GetCollisions` for one track over the target list.
template <typename Prop>
__host__ __device__ inline void schedule_scattering(BicCascadeState& st,
                                                    imr::CollisionList& colls, int index,
                                                    CascadeBuffers& buf, const CascadeSpecies& sp,
                                                    const Prop& propagator,
                                                    imr::ScatterRefusal& sref,
                                                    CascadeRefusal& ref) {
  const CascadeTrack& pro = st.lists.pool[index];
  for (int i = 0; i < st.lists.n_pool; ++i) {
    const CascadeTrack& tgt = st.lists.pool[i];
    if (tgt.list != kListTarget) { continue; }
    const imr::MesonBaryonBuffers* mb = nullptr;
    if (CascadeSpecies::is_pion(pro.pdg)) {
      const int slot = ensure_meson_buffer(buf, pro.pdg, tgt.pdg, sp.mass_of(pro.pdg),
                                           sp.mass_of(tgt.pdg),
                                           CascadeSpecies::iso3_of(pro.pdg),
                                           CascadeSpecies::iso3_of(tgt.pdg), sp.pi_plus_mass,
                                           sp.proton_mass, ref);
      if (slot >= 0) { mb = &buf.meson[slot]; }
    }
    const imr::TimeToInteraction tt = imr::scatterer_time_to_interaction(
        pro.pdg, tgt.pdg, pro.charge, tgt.charge, pro.position, tgt.position, pro.momentum,
        pro.momentum, tgt.momentum, pro.actual_mass(), tgt.actual_mass(), pro.pdg_mass,
        tgt.pdg_mass, sref, mb);
    if (tt.time == DBL_MAX) { continue; }
    colls.add(tt.time + st.current_time, index, i, kGenScatterer, sref);
  }
  (void)propagator;
}

/// `G4BinaryCascade::FindCollisions(secondaries)` - every action, in `theImR`'s order, for every
/// track in the list it is given.
template <typename Prop, typename Rng>
__host__ __device__ inline void find_collisions(BicCascadeState& st, imr::CollisionList& colls,
                                                const int* indices, int n,
                                                const imr::ConcreteChannel* chans, int n_chan,
                                                CascadeBuffers& buf, const CascadeSpecies& sp,
                                                const Prop& propagator, Rng& rng,
                                                CascadeRefusal& ref) {
  (void)chans;
  (void)n_chan;
  for (int k = 0; k < n; ++k) {
    KineticDecayRefusal dref;
    imr::AbsorptionRefusal aref;
    imr::ScatterRefusal sref;
    schedule_decay(st, colls, indices[k], rng, dref);
    schedule_absorption(st, colls, indices[k], propagator, aref);
    schedule_scattering(st, colls, indices[k], buf, sp, propagator, sref, ref);
    if (dref.unknown_species) {
      ref.unknown_species = true;
      ref.refused_pdg = dref.refused_pdg;
    }
  }
}

/// `G4BinaryCascade::FindLateParticleCollision` - the state is set from the sphere intersection
/// times BEFORE the collision is scheduled, and then `G4BCLateParticle` schedules one for every
/// track whatever its state.
template <typename Prop>
__host__ __device__ inline void find_late_particle_collision(BicCascadeState& st,
                                                             imr::CollisionList& colls, int index,
                                                             const Prop& propagator) {
  CascadeTrack& t = st.lists.pool[index];
  const KineticTrack kt = as_kinetic_track(t);
  double tin = 0.0;
  double tout = 0.0;
  if (propagator.sphere_intersection_times(kt, tin, tout)) {
    if (tin > 0.0) {
      t.state = kOutside;
    } else if (tout > 0.0) {
      t.state = kInside;
    } else {
      t.state = kMissNucleus;
    }
  } else {
    t.state = kMissNucleus;
  }
  imr::ScatterRefusal cref;
  colls.add(bc_late_particle_time(t.formation_time, st.current_time), index, -1,
            kGenLateParticle, cref);
}

/// The products of one scheduled collision, before they are put into the lists.
struct CollisionProducts {
  CascadeTrack track[4];
  int n = 0;
  bool empty_final_state = false;  ///< the generator returned nothing, which vetoes the collision
};

/// `G4CollisionInitialState::GetFinalState()` - which generator made the collision decides which
/// final state runs, and the four differ in what the products INHERIT.
///
///   * `G4VElasticCollision::FinalState` makes `new G4KineticTrack(trk1)` and
///     `new G4KineticTrack(trk2)` - COPIES of the entrance tracks - and only replaces their
///     momenta. So an elastic product keeps the entrance track's position, formation time, state
///     AND its `theNucleon` pointer, which is what makes `Hit()` on it mark the nucleon.
///   * `G4VScatteringCollision::FinalState` makes NEW tracks at `trk1.GetPosition()` and
///     `trk2.GetPosition()` with a formation time of zero and no nucleon. A resonance-production
///     product therefore has no nucleon to mark.
///   * `G4VAnnihilationCollision::FinalState` makes ONE new track at `trk1.GetPosition()`.
///   * `G4BCDecay` hands back `aProjectile->Decay()`, and `G4BCLateParticle` a copy of the
///     projectile.
template <typename Rng>
__host__ __device__ inline CollisionProducts collision_final_state(
    BicCascadeState& st, const imr::CollisionInitialState& coll,
    const imr::ConcreteChannel* chans, int n_chan, CascadeBuffers& buf, const CascadeSpecies& sp,
    Rng& rng, CascadeRefusal& ref) {
  CollisionProducts out;
  const CascadeTrack& pro = st.lists.pool[coll.primary];
  if (coll.generator == kGenLateParticle) {
    out.track[0] = pro;
    out.n = 1;
    return out;
  }
  if (coll.generator == kGenDecay) {
    DecayTrack in;
    in.pdg = pro.pdg;
    in.momentum = pro.momentum;
    in.position = pro.position;
    in.formation_time = pro.formation_time;
    in.creator_model_id = pro.creator_model_id;
    DecayTrack products[imr::kDecayMaxDaughters];
    KineticDecayRefusal dref;
    const int n = kinetic_decay_one(in, products, rng, dref);
    if (dref.any()) {
      ref.unknown_species = ref.unknown_species || dref.unknown_species;
      ref.refused_pdg = dref.refused_pdg;
    }
    if (n == 0) {
      out.empty_final_state = true;
      return out;
    }
    for (int i = 0; i < n && i < 4; ++i) {
      CascadeTrack& t = out.track[i];
      t = CascadeTrack{};
      t.pdg = products[i].pdg;
      const int si = imr::decay_species_index(t.pdg);
      t.pdg_mass = (si >= 0) ? imr::decay_species_mass()[si] : 0.0;
      t.charge = (si >= 0) ? imr::decay_species_charge()[si] : 0;
      t.baryon = (si >= 0) ? imr::decay_species_baryon()[si] : 0;
      t.momentum = products[i].momentum;
      t.position = products[i].position;
      t.formation_time = products[i].formation_time;
      t.creator_model_id = products[i].creator_model_id;
      t.parent_resonance_pdg = products[i].parent_resonance_pdg;
      t.parent_resonance_id = products[i].parent_resonance_id;
      ++out.n;
    }
    return out;
  }
  if (coll.target < 0) {
    out.empty_final_state = true;
    return out;
  }
  const CascadeTrack& tgt = st.lists.pool[coll.target];
  if (coll.generator == kGenAbsorption) {
    if (coll.target2 < 0) {
      out.empty_final_state = true;
      return out;
    }
    const CascadeTrack& tgt2 = st.lists.pool[coll.target2];
    const imr::AbsorptionFinalState fs = imr::meson_absorption_final_state(
        pro.pdg, static_cast<double>(pro.charge), pro.momentum, tgt.momentum, tgt2.momentum,
        tgt.pdg, tgt2.pdg, sp.proton_mass, sp.neutron_mass, rng);
    // `new G4KineticTrack(d1, 0., targets[0]->GetPosition(), final1)` - the two TARGETS'
    // positions, not the projectile's.
    out.track[0] = CascadeTrack{};
    out.track[0].pdg = fs.pdg1;
    out.track[0].pdg_mass = sp.mass_of(fs.pdg1);
    out.track[0].charge = (fs.pdg1 == imr::kPdgProton) ? 1 : 0;
    out.track[0].baryon = 1;
    out.track[0].momentum = fs.p1;
    out.track[0].position = tgt.position;
    out.track[1] = CascadeTrack{};
    out.track[1].pdg = fs.pdg2;
    out.track[1].pdg_mass = sp.mass_of(fs.pdg2);
    out.track[1].charge = (fs.pdg2 == imr::kPdgProton) ? 1 : 0;
    out.track[1].baryon = 1;
    out.track[1].momentum = fs.p2;
    out.track[1].position = tgt2.position;
    out.n = 2;
    return out;
  }
  // kGenScatterer. `G4Scatterer::GetFinalState` copies the target - `G4KineticTrack
  // target_reloc(*(theTargets[0]))` - and calls Scatter on the copy, so nothing it does to the
  // target survives.
  imr::ScatterRefusal sref;
  imr::ScatterFinalState fs;
  if (CascadeSpecies::is_pion(pro.pdg)) {
    const int slot = ensure_meson_buffer(buf, pro.pdg, tgt.pdg, sp.mass_of(pro.pdg),
                                         sp.mass_of(tgt.pdg), CascadeSpecies::iso3_of(pro.pdg),
                                         CascadeSpecies::iso3_of(tgt.pdg), sp.pi_plus_mass,
                                         sp.proton_mass, ref);
    if (slot < 0) {
      out.empty_final_state = true;
      return out;
    }
    fs = imr::meson_scatter_final_state(pro.pdg, tgt.pdg, pro.momentum, tgt.momentum,
                                        pro.actual_mass(), tgt.actual_mass(), pro.pdg_mass,
                                        tgt.pdg_mass, sp.mass_of(pro.pdg), sp.mass_of(tgt.pdg),
                                        sp.pi_plus_mass, sp.proton_mass,
                                        CascadeSpecies::iso3_of(pro.pdg),
                                        CascadeSpecies::iso3_of(tgt.pdg), buf.meson[slot], rng,
                                        sref);
  } else {
    const int slot = ensure_nn_buffer(buf, chans, n_chan, pro.pdg, tgt.pdg, pro.pdg_mass,
                                      tgt.pdg_mass, ref);
    if (slot < 0) {
      out.empty_final_state = true;
      return out;
    }
    fs = imr::nn_scatter_final_state(chans, n_chan, pro.pdg, tgt.pdg, pro.momentum, tgt.momentum,
                                     pro.actual_mass(), tgt.actual_mass(), pro.pdg_mass,
                                     tgt.pdg_mass, sp.proton_mass, sp.neutron_mass,
                                     sp.pi_plus_mass, buf.nn[slot], rng, sref);
  }
  if (fs.n == 0) {
    out.empty_final_state = true;
    return out;
  }
  const bool elastic = (fs.channel == 0 && (fs.component == imr::kNpElastic ||
                                            fs.component == imr::kNNElastic)) ||
                       (fs.channel == 1 && fs.component == imr::kMesonBaryonElastic);
  for (int i = 0; i < fs.n; ++i) {
    CascadeTrack& t = out.track[i];
    if (elastic) {
      // A COPY of the entrance track, momentum replaced. Everything else is inherited.
      t = (i == 0) ? pro : tgt;
      t.momentum = fs.p[i];
    } else {
      t = CascadeTrack{};
      t.pdg = fs.pdg[i];
      t.momentum = fs.p[i];
      t.position = (i == 0) ? pro.position : tgt.position;
      const int si = imr::decay_species_index(fs.pdg[i]);
      t.pdg_mass = (si >= 0) ? imr::decay_species_mass()[si] : sp.mass_of(fs.pdg[i]);
      t.charge = (si >= 0) ? imr::decay_species_charge()[si] : 0;
      t.baryon = (si >= 0) ? imr::decay_species_baryon()[si] : 0;
      if (si < 0) {
        ref.unknown_species = true;
        ref.refused_pdg = fs.pdg[i];
      }
    }
    ++out.n;
  }
  return out;
}

/// `G4BinaryCascade::ApplyCollision`.
///
/// Returns false for a collision that was vetoed - by an empty final state, by the Pauli block or
/// by the Fermi correction - in which case `Propagate` removes it from the manager. A DECAY that
/// is vetoed is re-scheduled first, which is `FindDecayCollision(primary)`: the same track gets a
/// new residual lifetime and another chance, and that draws another uniform.
template <typename Prop, typename Rng, typename FindColl>
__host__ __device__ inline bool apply_collision(BicCascadeState& st, imr::CollisionList& colls,
                                                int collision_index,
                                                const imr::ConcreteChannel* chans, int n_chan,
                                                CascadeBuffers& buf, const CascadeSpecies& sp,
                                                const Prop& propagator,
                                                const NuclearDensity& density,
                                                double coulomb_barrier, Rng& rng,
                                                FindColl&& find_collisions,
                                                CascadeRefusal& ref) {
  const imr::CollisionInitialState coll = colls.items[collision_index];
  CascadeTrack& primary = st.lists.pool[coll.primary];
  const int n_targets = coll.n_targets();
  const bool have_target = n_targets > 0;
  if (have_target && primary.state != kInside) { return false; }

  const imr::LorentzVector mom4_primary = primary.momentum;
  int initial_baryon = 0;
  int initial_charge = 0;
  if (primary.state == kInside) {
    initial_baryon = primary.baryon;
    initial_charge = primary.charge;
  }
  CascadeTrack targets[2];
  int n_tgt = 0;
  if (coll.target >= 0) { targets[n_tgt++] = st.lists.pool[coll.target]; }
  if (coll.target2 >= 0) { targets[n_tgt++] = st.lists.pool[coll.target2]; }
  const double initial_e_fermi =
      correct_shortlived_primary_for_fermi(primary, targets, n_tgt, propagator);

  CollisionProducts products = collision_final_state(st, coll, chans, n_chan, buf, sp, rng, ref);
  // "reset primary to initial state, in case there is a veto..."
  primary.momentum = mom4_primary;

  const bool late_particle = !have_target && products.n == 1;
  const bool decay_collision = !have_target && products.n > 1;
  bool success = true;
  if (late_particle) {
    initial_baryon = 0;
    initial_charge = 0;
    st.late_a -= primary.baryon;
    st.late_z -= primary.charge;
  }
  for (int i = 0; i < n_tgt; ++i) {
    initial_baryon += targets[i].baryon;
    initial_charge += targets[i].charge;
  }
  if (!late_particle) {
    if (products.n == 0 || products.empty_final_state ||
        !check_pauli_principle(products.track, products.n, st.initial_a, st.initial_z, density,
                              coulomb_barrier)) {
      success = false;
    }
    if (success && primary.state == kInside) {
      if (!correct_shortlived_finals_for_fermi(products.track, products.n, initial_e_fermi,
                                               propagator)) {
        success = false;
      }
    }
  }
  if (!success) {
    if (decay_collision) {
      KineticDecayRefusal dref;
      schedule_decay(st, colls, coll.primary, rng, dref);
    }
    return false;
  }

  int final_baryon = 0;
  int final_charge = 0;
  int new_index[4];
  int n_new = 0;
  int to_final[4];
  int n_to_final = 0;
  for (int i = 0; i < products.n; ++i) {
    CascadeTrack t = products.track[i];
    if (!late_particle) {
      t.state = primary.state;  // "decay may be anywhere!"
      if (t.state == kInside) {
        final_baryon += t.baryon;
        final_charge += t.charge;
      }
    } else {
      const KineticTrack kt = as_kinetic_track(t);
      double tin = 0.0;
      double tout = 0.0;
      if (propagator.sphere_intersection_times(kt, tin, tout)) {
        if (tin > 0.0) {
          t.state = kOutside;
        } else if (tout > 0.0) {
          t.state = kInside;
          final_baryon += t.baryon;
          final_charge += t.charge;
        } else {
          t.state = kGoneOut;
        }
      } else {
        t.state = kMissNucleus;
      }
    }
    const int idx = st.lists.add(t, kListNone);
    if (idx < 0) {
      ref.capacity = true;
      return false;
    }
    if (late_particle && (t.state == kGoneOut || t.state == kMissNucleus)) {
      st.lists.pool[idx].list = kListFinal;
      to_final[n_to_final++] = idx;
    } else {
      new_index[n_new++] = idx;
    }
  }
  if (n_to_final > 0) { colls.remove_tracks(to_final, n_to_final); }

  st.current_a += final_baryon - initial_baryon;
  st.current_z += final_charge - initial_charge;

  const int old_secondary[1] = {coll.primary};
  primary.hit = true;
  int old_target[2];
  int n_old_tgt = 0;
  if (coll.target >= 0) {
    st.lists.pool[coll.target].hit = true;
    old_target[n_old_tgt++] = coll.target;
  }
  if (coll.target2 >= 0) {
    st.lists.pool[coll.target2].hit = true;
    old_target[n_old_tgt++] = coll.target2;
  }
  update_tracks_and_collisions(st, colls, old_secondary, 1, old_target, n_old_tgt, new_index,
                               n_new, find_collisions);
  return true;
}

}  // namespace g4gpu::bic

#endif
