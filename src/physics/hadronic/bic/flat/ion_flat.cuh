// P36: the light-ion reaction as data - `G4BinaryLightIonReaction::ApplyYourself`'s cascade arm
// cut into the same stages as the nucleon arm (cascade_flat.cuh, docs/RISK.md V308, V310).
//
// The reaction's `Interact` is the nucleon arm's `Propagate` on a different set-up: TWO
// `G4Fancy3DNucleus`es (the lighter nucleus is the projectile, `SetLighterAsProjectile`), one
// `G4KineticTrack` per projectile nucleon as the secondaries, and an outer loop of at most 150
// tries in which a NULL and an empty product list both try again. So the record is the nucleon
// arm's (`FlatCascade`, with `model = kModelIon` selecting `Interact`'s outer test) plus the
// projectile nucleus, its secondaries and the frame; the step, the finish and the tail up to the
// product list are the nucleon arm's stages unchanged; and the end - `SortResult`, the energy
// corrector, the spectators' de-excitation and the packing - is `blir_cascade_arm`'s lines after
// `Interact`, on the tail slot. light_ion_reaction.cuh is not edited: it is the thread path.
//
// The fusion arm (below 50 MeV per nucleon: one PreCompound call, no cascade) is handed to the
// thread path, like the nucleon arm's `theBCminP` arm.
#ifndef G4GPU_BIC_FLAT_ION_FLAT_CUH
#define G4GPU_BIC_FLAT_ION_FLAT_CUH

#include <cstddef>
#include <cstring>

#include "physics/hadronic/bic/flat/cascade_flat.cuh"

namespace g4gpu::bic::flat {

template <int N, int P, int C>
struct FlatIon {
  FlatCascade<N, P, C> c;
  Nucleus3D proj;
  BlirFrame f;
  double projectile_excitation = 0.0;
  int baryon = 0;          ///< the projectile's A, as `G4HadProjectile` gives it
  double charge = 0.0;
  double mass = 0.0;
  Nucleon proj_nucleons[N];
  CascadeTrack initial[N];  ///< `Interact`'s secondaries, one a projectile nucleon
  NucleusReport proj_rep;   ///< P39: the factory's report for the projectile
};

/// A lane takes a light-ion reaction.
template <int N, int P, int C>
__host__ __device__ inline void ion_start(FlatIon<N, P, C>& r, int entry, int pdg, double kin,
                                          double mass, double charge, int baryon, int tz, int ta,
                                          const Philox<double>& rng) {
  flat_start(r.c, entry, pdg, kin, tz, ta, rng);
  r.c.model = kModelIon;
  r.mass = mass;
  r.charge = charge;
  r.baryon = baryon;
  r.proj = Nucleus3D{};
  r.proj.nucleons = r.proj_nucleons;
  r.proj.capacity = N;
  r.projectile_excitation = 0.0;
}

/// P39: the light-ion set-up's requests - this try's projectile and target, keyed on the
/// reaction and the try: tags 2t and 2t + 1 (cascade_flat.cuh's note). Both or neither: a try
/// whose second request could not be placed samples both on its lane.
template <int N, int P, int C>
__device__ inline void ion_request(FlatIon<N, P, C>& r, const npool::FactoryLists& lists) {
  FlatCascade<N, P, C>& c = r.c;
  const int t = c.n_nuclei++;
  const bool p_ok = flat_push_nucleus(lists, r.proj_nucleons, &r.proj_rep, r.f.pa, r.f.pz,
                                      c.nkey_lo, c.nkey_hi, 2 * t);
  const bool t_ok = p_ok && flat_push_nucleus(lists, c.nucleons, &c.nuc_rep, r.f.ta, r.f.tz,
                                              c.nkey_lo, c.nkey_hi, 2 * t + 1);
  c.nuc_ready = t_ok ? 1 : 0;
}

/// `blir_apply_yourself` above `Interact`: the anti/hyper test, `SetLighterAsProjectile`, the
/// fusion threshold.
template <int N, int P, int C>
__host__ __device__ inline void stage_ion_model(FlatIon<N, P, C>& r, int target_l) {
  FlatCascade<N, P, C>& c = r.c;
  if (r.baryon < 1 || target_l != 0) { return flat_hand_off(c, kHandRefused); }
  const double m = r.mass;
  const double t = c.kin;
  const LorentzVector p4(Vec3d{0.0, 0.0, std::sqrt(t * (t + 2.0 * m))}, t + m);
  r.f = blir_set_lighter_as_projectile(r.baryon, static_cast<int>(std::lrint(r.charge)), c.ta,
                                       c.tz, p4);
  const double kin_per_nucleon = blir_kin_per_nucleon(r.f);
  if (kin_per_nucleon < blir_fusion_threshold_per_nucleon()) {
    return flat_hand_off(c, kHandPrecompound);
  }
  if (r.f.pa > N || r.f.ta > N) { return flat_hand_off(c, kHandCapacity); }
  // From here the record's target is the frame's: the heavier nucleus.
  c.ta = r.f.ta;
  c.tz = r.f.tz;
  c.try_count = 0;
  c.have = false;
  c.n_products = 0;
  c.outcome = kPropagateOk;
  c.stage = kStNucleus;
}

/// `Interact`'s tries, from the top of one to the first `Entries()` of its `Propagate`: both
/// nuclei, the impact parameter, the projectile nucleons as secondaries, the field maps and the
/// prologue. A try whose `Propagate` finds no collision is the loop's next try, in this call.
template <int N, int P, int C>
__host__ __device__ inline void stage_ion_nucleus(FlatIon<N, P, C>& r, const FlatEnv& env,
                                                  Nucleus3DScratch& nsc,
                                                  FlatScratch<N, P>& scr) {
  FlatCascade<N, P, C>& c = r.c;
  const BlirFrame& f = r.f;
  for (;;) {
    if (++c.steps > env.k_steps) { return flat_straggle(c, env); }
    ++c.try_count;
    // P39: this try's two nuclei from the factory (requested by `ion_request` before this
    // launch), or - the fall-back - sampled here in `Interact`'s order.
    const bool from_factory = (c.nuc_ready != 0);
    c.nuc_ready = 0;
    r.proj.nucleons = r.proj_nucleons;
    r.proj.capacity = N;
    const NucleusReport prep =
        from_factory ? flat_take_factory_nucleus(r.proj, r.proj_nucleons, N, f.pa, f.pz,
                                                 r.proj_rep)
                     : nucleus_init(r.proj, nsc, f.pa, f.pz, c.rng);
    if (prep.fatal()) { return flat_hand_off(c, prep.capacity ? kHandCapacity : kHandRefused); }
    r.proj.center_nucleons();
    c.nucleus.nucleons = c.nucleons;
    c.nucleus.capacity = N;
    const NucleusReport trep =
        from_factory ? flat_take_factory_nucleus(c.nucleus, c.nucleons, N, f.ta, f.tz, c.nuc_rep)
                     : nucleus_init(c.nucleus, nsc, f.ta, f.tz, c.rng);
    if (trep.fatal()) { return flat_hand_off(c, trep.capacity ? kHandCapacity : kHandRefused); }

    const double impact_max = c.nucleus.outer_radius() + r.proj.outer_radius();
    const double ax = (2.0 * c.rng.uniform() - 1.0) * impact_max;
    const double ay = (2.0 * c.rng.uniform() - 1.0) * impact_max;
    const deex::Vec3d pos{ax, ay, -2.0 * impact_max - 5.0 * deex::fermi()};
    const double inv_pa = 1.0 / static_cast<double>(f.pa);
    const deex::Vec3d scaled = inv_pa * f.mom.v;
    const imr::LorentzVector nucleon_mom(deex::Vec3d{0.0, 0.0, std::sqrt(g4gpu::mag2(scaled))},
                                         inv_pa * f.mom.e);
    int n_initial = 0;
    for (int i = 0; i < r.proj.my_a && n_initial < N; ++i) {
      const Nucleon& n = r.proj.nucleons[i];
      CascadeTrack t;
      t.pdg = n.pdg();
      t.pdg_mass = n.pdg_mass();
      t.charge = n.charge();
      t.baryon = 1;
      t.momentum = nucleon_mom;
      t.position = n.position + pos;
      t.formation_time = 0.0;
      t.state = kOutside;
      t.nucleon_index = i;
      t.nucleon_owner = 1;
      t.creator_model_id = blir_model_id();
      const double density = r.proj.density.density(n.position);
      const double p_fermi = r.proj.fermi.fermi_momentum(density);
      const double e_fermi = std::sqrt(t.pdg_mass * t.pdg_mass + p_fermi * p_fermi) - t.pdg_mass;
      t.projectile_potential = -e_fermi;
      r.initial[n_initial++] = t;
    }
    if (n_initial != r.proj.my_a) { return flat_hand_off(c, kHandCapacity); }
    NucleusReport frep;
    c.prop = make_rk_propagation(c.nucleus, frep, c.proton_field, c.neutron_field,
                                 kMaxFieldTable);
    c.st.projectile_nucleons = r.proj.nucleons;
    if (!flat_prologue(c, r.initial, n_initial, env, scr)) { return; }
    if (c.colls.size() != 0) { break; }
    // `Propagate`'s NULL: `Interact` tries again, at most 150 times.
    c.outcome = kPropagateNoCollision;
    c.n_products = 0;
    if (c.try_count >= 150) {
      c.have = false;
      c.stage = kStTail;
      c.tail_work = kTailResult;
      return;
    }
    // P39: a try on the factory's nuclei ends the call - the record stays in `kStNucleus`, and
    // the next set-up asks the factory for the next try's pair (tags 2t + 2, 2t + 3) - so a
    // miss costs a set-up round where it cost two `G4Fancy3DNucleus::Init`s on this lane.
    if (from_factory) { return; }
  }
  c.loop_guard = 1000000;
  c.have_products = false;
  c.mode = kModeMain;
  c.stepping = false;
  c.stage = kStStep;
}

/// P42 (docs/RISK.md V331): a reaction's state moved, at a stage boundary, from a record of `P`
/// tracks and `C` collisions to one of `P2` and `C2` (the thread path's slot: 1,024 and 8,192),
/// where it is CONTINUED - the generator, the counters, the nuclei, every track and every
/// collision as they are, so the larger record computes exactly what the smaller would have
/// computed had it had the room. Never a rerun from the key (the hand-back rule, V322).
///
/// `FlatCascade`'s members before `nucleons` depend on neither capacity, so they are copied as
/// bytes; the arrays are copied up to their fill; then every pointer that pointed into the
/// source record is pointed at the same member of the destination, and the two capacities are
/// the destination's. Returns false (and copies nothing) if the layouts ever disagreed.
template <int N, int P, int C, int P2, int C2>
__host__ __device__ inline bool ion_grow_copy(FlatIon<N, P2, C2>& d, const FlatIon<N, P, C>& s) {
  static_assert(P2 >= P && C2 >= C, "a reaction only ever moves to a larger record");
  const FlatCascade<N, P, C>& a = s.c;
  FlatCascade<N, P2, C2>& b = d.c;
  const std::size_t head = static_cast<std::size_t>(reinterpret_cast<const char*>(a.nucleons) -
                                                    reinterpret_cast<const char*>(&a));
  const std::size_t head2 = static_cast<std::size_t>(reinterpret_cast<const char*>(b.nucleons) -
                                                     reinterpret_cast<const char*>(&b));
  if (head != head2) { return false; }
  memcpy(static_cast<void*>(&b), static_cast<const void*>(&a), head);
  for (int i = 0; i < N; ++i) { b.nucleons[i] = a.nucleons[i]; }
  for (int i = 0; i < kMaxFieldTable; ++i) {
    b.proton_field[i] = a.proton_field[i];
    b.neutron_field[i] = a.neutron_field[i];
  }
  const int np = a.st.lists.n_pool;
  for (int i = 0; i < np; ++i) {
    b.pool[i] = a.pool[i];
    b.prestate[i] = a.prestate[i];
    b.transfer[i] = a.transfer[i];
  }
  for (int i = 0; i < a.colls.n; ++i) { b.coll_items[i] = a.coll_items[i]; }
  d.proj = s.proj;
  d.f = s.f;
  d.projectile_excitation = s.projectile_excitation;
  d.baryon = s.baryon;
  d.charge = s.charge;
  d.mass = s.mass;
  for (int i = 0; i < N; ++i) {
    d.proj_nucleons[i] = s.proj_nucleons[i];
    d.initial[i] = s.initial[i];
  }
  d.proj_rep = s.proj_rep;
  // The pointers into the record.
  b.st.lists.pool = b.pool;
  b.st.lists.capacity = P2;
  b.colls.items = b.coll_items;
  b.colls.capacity = C2;
  b.nucleus.nucleons = b.nucleons;
  if (a.st.nucleons == a.nucleons) { b.st.nucleons = b.nucleons; }
  if (a.st.projectile_nucleons == s.proj_nucleons) { b.st.projectile_nucleons = d.proj_nucleons; }
  if (s.proj.nucleons == s.proj_nucleons) { d.proj.nucleons = d.proj_nucleons; }
  if (a.prop.proton.nucleon.table == a.proton_field) { b.prop.proton.nucleon.table = b.proton_field; }
  if (a.prop.neutron.nucleon.table == a.neutron_field) {
    b.prop.neutron.nucleon.table = b.neutron_field;
  }
  return true;
}

/// What the light-ion arm's end decided, for the caller's `run_arm_light_ion` mapping.
enum IonEnd : int {
  kIonFinalState = 0,  ///< `result` is the final state (the cascaders, or the primary alive)
  kIonHandBack = 1     ///< a refusal the thread path must book: the record is handed back
};

/// `blir_cascade_arm` after `Interact`, on the tail slot: `SortResult`, the corrector loop, the
/// spectators' de-excitation and the packing - line for line, the products being the ones
/// `stage_tail` left in `store.cascade.products`. `run_arm_light_ion`'s mapping is applied: a
/// refusal hands the record back, Geant4's own "primary alive" answers are final states.
template <int N, int P, int C>
__host__ __device__ inline int ion_pack(FlatIon<N, P, C>& r, BlirStorage& store,
                                        const data::LevelTable& lt, const deex::FermiPool& pool,
                                        const preco::PrecoWorkspace& ws, BlirFinalState& result,
                                        const TailFallback* fb = nullptr,
                                        BlirProduct* cascaders_save = nullptr) {
  FlatCascade<N, P, C>& c = r.c;
  const BlirFrame& f = r.f;
  result.clear();
  if (!c.have) {
    // `ref.no_final_state`: the primary alive (not a refusal in `run_arm_light_ion`).
    result.status = physics::hadronic::HadFinalStateStatus::kIsAlive;
    result.energy_change = c.kin;
    result.momentum_change = Vec3<double>{0.0, 0.0, 1.0};
    return kIonFinalState;
  }
  r.projectile_excitation = blir_projectile_excitation(r.proj);
  BlirRefusal ref;
  imr::LorentzVector p_initial_state = f.mom;
  p_initial_state.e += deex::nuclear_mass(f.ta, f.tz);
  imr::LorentzVector p_final_state;
  int n_spectators = 0, n_cascaders = 0, spectator_a = 0, spectator_z = 0;
  const imr::LorentzVector p_spectators = blir_sort_result(
      store.cascade.products, c.n_products, store.spectators, store.spectator_capacity,
      store.cascaders, store.cascader_capacity, n_spectators, n_cascaders, spectator_a,
      spectator_z, p_final_state, ref);
  if (ref.any()) {
    flat_hand_off(c, kHandCapacity);
    return kIonHandBack;
  }
  imr::LorentzVector momentum = p_initial_state - p_final_state;
  int loopcount = 0;
  while (std::fabs(momentum.e - p_spectators.e) > 10.0 * u::MeV<double>()) {
    const imr::LorentzVector p_correct = p_initial_state - p_spectators;
    blir_energy_and_momentum_corrector(store.cascaders, n_cascaders, p_correct);
    p_final_state = imr::LorentzVector();
    for (int i = 0; i < n_cascaders; ++i) {
      p_final_state = p_final_state + store.cascaders[i].momentum;
    }
    momentum = p_initial_state - p_final_state;
    if (++loopcount > 10) {
      if (std::sqrt(g4gpu::mag2(momentum.v)) - momentum.e > 10.0 * u::keV<double>()) {
        // `throw G4HadronicException` - `run_arm_light_ion` books it: the thread path's.
        flat_hand_off(c, kHandRefused);
        return kIonHandBack;
      }
      break;
    }
  }
  if (spectator_a > 0) {
    if (std::sqrt(g4gpu::mag2(momentum.v)) - momentum.e < 10.0 * u::keV<double>()) {
      deex::DeexStatus dstatus;
      BlirCorrectorReport last;
      // P41: what the spectators' de-excitation changes - the cascaders, their count and the
      // generator - kept, so that a compact workspace that binds is CONTINUED from here on a
      // slot-sized one (cascade_flat.cuh's `TailFallback`), never rerun from the key.
      const Philox<double> rng_before = c.rng;
      const int n_cascaders_before = n_cascaders;
      if (fb != nullptr) {
        for (int i = 0; i < n_cascaders; ++i) { cascaders_save[i] = store.cascaders[i]; }
      }
      blir_deexcite_spectator(store.spectators, n_spectators, store.cascaders, n_cascaders,
                              store.cascader_capacity, spectator_a, spectator_z, f.pa,
                              r.projectile_excitation, momentum, p_initial_state,
                              p_final_state, lt, pool, ws.deex, c.rng, dstatus, ref, last);
      if (fb != nullptr && dstatus.refused_capacity) {
        const int k = fb->acquire();
        if (k < 0) {
          flat_hand_off(c, kHandTailScratch);
          return kIonHandBack;
        }
        c.rng = rng_before;
        n_cascaders = n_cascaders_before;
        for (int i = 0; i < n_cascaders; ++i) { store.cascaders[i] = cascaders_save[i]; }
        ref = BlirRefusal{};
        dstatus = deex::DeexStatus{};
        blir_deexcite_spectator(store.spectators, n_spectators, store.cascaders, n_cascaders,
                                store.cascader_capacity, spectator_a, spectator_z, f.pa,
                                r.projectile_excitation, momentum, p_initial_state,
                                p_final_state, lt, pool, fb->big[k].deex, c.rng, dstatus, ref,
                                last);
        fb->release(k);
      }
      if (ref.any()) {
        flat_hand_off(c, ref.capacity ? kHandCapacity : kHandRefused);
        return kIonHandBack;
      }
    } else {
      // "invalid final state": the primary alive, Geant4's own answer.
      result.status = physics::hadronic::HadFinalStateStatus::kIsAlive;
      result.energy_change = c.kin;
      result.momentum_change = Vec3<double>{0.0, 0.0, 1.0};
      return kIonFinalState;
    }
  }
  result.status = physics::hadronic::HadFinalStateStatus::kStopAndKill;
  const deex::Vec3d inv_boost{-f.breit_boost.x, -f.breit_boost.y, -f.breit_boost.z};
  for (int i = 0; i < n_cascaders; ++i) {
    if (!store.cascaders[i].newly_added) { continue; }
    imr::LorentzVector q = store.cascaders[i].momentum;
    if (f.swapped) {
      q = boost_by_rotation(q, inv_boost);
      q.v = deex::Vec3d{-q.v.x, -q.v.y, -q.v.z};
    }
    physics::hadronic::HadSecondary<double> s;
    s.pdg = store.cascaders[i].pdg;
    s.z = store.cascaders[i].z;
    s.a = store.cascaders[i].a;
    s.time = 0.0;
    s.weight = 1.0;
    s.creator_model_id = blir_model_id();
    capture::set_four_momentum(s, q, store.cascaders[i].pdg_mass);
    if (!result.add_secondary(s)) {
      flat_hand_off(c, kHandCapacity);
      return kIonHandBack;
    }
  }
  return kIonFinalState;
}

}  // namespace g4gpu::bic::flat

#endif
