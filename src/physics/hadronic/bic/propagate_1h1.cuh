// G4BinaryCascade::Propagate1H1 - a nucleon or a charged pion on a single free nucleon.
//
// Transcribed from G4BinaryCascade.cc (models/binary_cascade/src, 11.1.1), `Propagate1H1` at
// line 2649, which `ApplyYourself` calls at line 329 instead of `Propagate` when the target
// nucleus has `massNumber > 1` false. The two loops around it are `ApplyYourself`'s own and live
// in `binary_cascade.cuh`, beside the ones that surround `Propagate`; this file is the function
// body and nothing else.
//
// ## WHAT THE FUNCTION IS: A RETRY LOOP, ONE SCATTER, ONE DECAY PASS AND THE PACKAGING
//
//     aHTarg = proton, or neutron if nucleus->GetCharge() == 0;   mom = G4LorentzVector(mass)
//     G4KineticTrack aTarget(aHTarg, 0., pos, mom);                // at rest, on shell
//     while (!done && tryCount++ < 200) {
//        secs = theH1Scatterer->Scatter(*(*secondaries).front(), aTarget);
//        for (ss ...) if ((*secs)[ss]->GetDefinition()->IsShortLived()) done = true;
//     }
//     for (current = 0; secs && current < secs->size(); current++)   // secs GROWS in the loop
//        if (IsShortLived) { dec = (*secs)[current]->Decay(); append dec to secs; }
//        else theFinalState.push_back((*secs)[current]);
//     every theFinalState track -> G4ReactionProduct(definition), momentum, total energy,
//        SetCreatorModelID(theBIC_ID), parent-resonance definition and id copied from the track
//
// Every piece under it is already ported and validated to the last bit: `G4Scatterer::Scatter`
// is `imr::nn_scatter_final_state` and `imr::meson_scatter_final_state` (P9b, P9d,
// docs/PORTED.md 2.1.10), `G4KineticTrack::Decay` is `kinetic_decay_one` over the shared decay
// engine (P9d), and the short-lived test is the generated table's copy of Geant4's own
// `IsShortLived()` flag. What this file adds is the order they are called in.
//
// ## THE LOOP REJECTS ELASTIC OUTCOMES, AND BELOW THE PION THRESHOLD IT CANNOT DO ANYTHING ELSE
//
// "must have one resonance in final state, or it was elastic, not allowed here." `done` is set
// by a SHORT-LIVED product and by nothing else, so every `Scatter` that came back elastic - two
// nucleons, or a pion and a nucleon - is thrown away and drawn again, up to 200 times. When the
// 200 run out, `secs` still holds the LAST scatter's products and the decay pass and the
// packaging run on them as if nothing had happened: Geant4 returns that elastic final state.
// Transcribed, not improved, and counted - `H1Report::exhausted`.
//
// It is not a corner. `G4CollisionNN`'s six resonance-production components are all zero below
// the Delta threshold, so a nucleon on hydrogen below about 290 MeV can only scatter elastically
// and EVERY call burns all 200 tries and returns the 200th elastic scatter. The uniforms spent on
// the 199 discarded ones are part of the stream, which is why the tape counts them.
//
// ## THE DECAY PASS IS NOT `G4DecayKineticTracks`
//
// It is its own loop, written inline in `Propagate1H1`, and it differs from the shared engine's
// `decay_kinetic_tracks` in one place: `G4KineticTrack::Decay()` returns 0 for a short-lived
// track whose total actual width is zero (and after a failed phase-space decay), and
// `G4DecayKineticTracks` leaves such a track in the list, while `Propagate1H1` writes
// `for(jter=dec->begin(); ...)` on the returned pointer without testing it - a null dereference.
// There is no Geant4 answer to port there, so it is `H1Refusal::decay_null`, named at the point
// Geant4 would crash. The ORDER of what survives is the same in both: the list is walked
// forwards while it grows, a decayed parent is skipped (Geant4 deletes it and leaves the dangling
// pointer in the vector, never to be read again), and the stable tracks reach `theFinalState` in
// index order - the Scatter's own products first, then each generation of daughters.
//
// ## `theH1Scatterer` IS A SECOND OBJECT OVER THE SAME CLASS-STATIC CHANNELS
//
// The constructor builds it with `theH1Scatterer = new G4Scatterer;` beside the cascade's own
// `aSc`, and both read `G4Scatterer::collisions`, the static `GROUP2(G4CollisionNN,
// G4CollisionMesonBaryon)` vector - so the two share the composites, their lazily built 32-point
// cross-section buffers included. The port has no static registry (docs/RISK.md V155): the
// buffers are `CascadeBuffers`, owned by the caller and built on first use by the same
// `ensure_nn_buffer`/`ensure_meson_buffer` the cascade calls, which is the state P9b ported. A
// buffer is a function of the pair of definitions and nothing else, so a buffer built by the
// cascade and one built here are the same doubles.
//
// ## WHAT IS NOT CARRIED, AND WHY IT IS NOT AN APPROXIMATION
//
//   * the positions. `Scatter` never calls `GetTimeToInteraction`, so neither track's position
//     reaches a number this function returns; the products inherit them and the packaging drops
//     them. They are carried anyway, as Geant4 carries them.
//   * `G4Scatterer::Scatter`'s energy and momentum balance, which is computed and printed only
//     under the environment variable `ScattererEnergyBalanceCheck`. The CHARGE balance is not
//     optional - `G4Exception("G4Scatterer", "im_r_matrix001", FatalException, ...)` - and it is
//     `H1Refusal::charge_imbalance` here.
#ifndef G4GPU_BIC_PROPAGATE_1H1_CUH
#define G4GPU_BIC_PROPAGATE_1H1_CUH

#include <cmath>

#include "physics/hadronic/bic/cascade_propagate.cuh"

namespace g4gpu::bic {

/// What `Propagate1H1` could not do. None of these is a Geant4 answer: each is a point where
/// Geant4 crashes, throws or reads storage this port has to be handed.
struct H1Refusal {
  /// `(*secs)[current]->Decay()` returned 0 for a short-lived track - a zero total actual width
  /// or a failed phase-space decay - and `Propagate1H1` dereferences the null it gets back.
  bool decay_null = false;
  /// The decay engine refused the track's decay by one of its own names (`KineticDecayRefusal`:
  /// a channel with more than four daughters, the 10,000-try threshold loop Geant4 ends in a
  /// throw, a phase-space rejection loop that gave up).
  bool decay_refused = false;
  /// A product whose PDG code the generated decay table does not carry, so its `IsShortLived()`
  /// flag is not known. The table is the closure of everything `G4Scatterer` makes; reaching
  /// this means the closure and the channel tables have drifted apart.
  bool unknown_species = false;
  /// `G4Scatterer::Scatter`'s `im_r_matrix001` FatalException: the products do not carry the
  /// entrance charge. Nothing in the tree should reach it.
  bool charge_imbalance = false;
  /// The working list (`CascadeWorkspace::pool`), the product list or the buffer table is full,
  /// or the caller handed a null one.
  bool capacity = false;
  int refused_pdg = 0;
  __host__ __device__ bool any() const {
    return decay_null || decay_refused || unknown_species || charge_imbalance || capacity;
  }
};

/// What `Propagate1H1` did, beyond its products.
struct H1Report {
  /// How many times `Scatter` was called: 1 to 200. `tryCount` itself ends one higher when the
  /// loop runs out, because the 201st evaluation of `tryCount++ < 200` increments it and fails.
  int scatters = 0;
  /// The 200 tries ran out with no short-lived product, and `secs` - the LAST scatter's products,
  /// elastic, or nothing at all - was used as it stood. Geant4's answer; see the file header.
  bool exhausted = false;
  /// How many tracks the last `Scatter` returned: 0 for NULL, 1 for a meson-baryon resonance, 2
  /// for everything else.
  int n_secs = 0;
  /// How many short-lived tracks the decay pass decayed.
  int n_decays = 0;
  /// Which way the last `Scatter` went - `imr::ScatterFinalState`'s path.
  int channel = -1;
  int component = -1;
  int middle = -1;
  int concrete = -1;
};

/// The track `G4Scatterer::Scatter` builds for product `i` of a final state.
///
/// An ELASTIC product is a COPY of the entrance track with its momentum replaced -
/// `G4VElasticCollision::FinalState` makes `new G4KineticTrack(trk1)` and `(trk2)` - so it keeps
/// the entrance track's creator id and parent-resonance fields (all -1/0 here: neither the
/// projectile `ApplyYourself` builds nor `aTarget` has any). Every other product is a NEW
/// `G4KineticTrack` of the channel's outgoing definition at the entrance track's position, which
/// is what `G4VScatteringCollision::FinalState` and `G4VAnnihilationCollision::FinalState` make.
/// The same construction `collision_final_state` in cascade_find.cuh uses for the cascade.
__host__ __device__ inline bool h1_product_track(const imr::ScatterFinalState& fs, int i,
                                                 const CascadeTrack& pro, const CascadeTrack& tgt,
                                                 const CascadeSpecies& sp, CascadeTrack& t,
                                                 H1Refusal& ref) {
  const bool elastic = (fs.channel == 0 && (fs.component == imr::kNpElastic ||
                                            fs.component == imr::kNNElastic)) ||
                       (fs.channel == 1 && fs.component == imr::kMesonBaryonElastic);
  if (elastic) {
    t = (i == 0) ? pro : tgt;
    t.momentum = fs.p[i];
    return true;
  }
  t = CascadeTrack{};
  t.pdg = fs.pdg[i];
  t.momentum = fs.p[i];
  t.position = (i == 0) ? pro.position : tgt.position;
  const int si = imr::decay_species_index(fs.pdg[i]);
  if (si < 0) {
    ref.unknown_species = true;
    ref.refused_pdg = fs.pdg[i];
    return false;
  }
  t.pdg_mass = imr::decay_species_mass()[si];
  t.charge = imr::decay_species_charge()[si];
  t.baryon = imr::decay_species_baryon()[si];
  (void)sp;
  return true;
}

/// `G4Scatterer::Scatter(trk1, trk2)`, for the one pair `Propagate1H1` ever hands it: the
/// projectile and a free nucleon at rest.
///
///     collision = FindCollision(trk1, trk2);             // G4CollisionNN, then MesonBaryon
///     if (collision) { xs = collision->CrossSection(trk1, trk2);
///        if (xs > 0) { products = collision->FinalState(trk1, trk2); ... charge balance ...
///                      return products; } }
///     return NULL;
///
/// Writes the products into `out` and returns how many - 0 for Geant4's NULL and for an empty
/// vector alike, because `Propagate1H1` treats the two the same way (its `secs &&` tests guard
/// both) - or -1 for a refusal.
///
/// The cross section is evaluated here even though the cascade's `collision_final_state` never
/// evaluates it: inside the cascade the collision was only SCHEDULED because
/// `GetTimeToInteraction` had already found it positive, and `GetFinalState` goes straight to
/// `Scatter`. Here nothing has asked, so the `aCrossSection > 0.0` gate is live - and it draws no
/// uniform, so asking it cannot move the stream.
template <typename Rng>
__host__ __device__ inline int h1_scatter(const CascadeTrack& pro, const CascadeTrack& tgt,
                                          const CascadeWorkspace& ws, const CascadeSpecies& sp,
                                          Rng& rng, CascadeTrack* out, H1Report& rep,
                                          H1Refusal& ref) {
  imr::ScatterRefusal sref;
  const int channel = imr::scatterer_find_collision(pro.pdg, tgt.pdg, sref);
  if (channel < 0) { return 0; }

  // `collision->CrossSection(trk1, trk2)`. `G4CollisionNN` overrides it with the total on the
  // recast on-shell pair; `G4CollisionMesonBaryon` is a composite with no source and answers
  // from its 32-point buffer, which has to exist first.
  CascadeRefusal cref;
  int meson_slot = -1;
  const imr::MesonBaryonBuffers* mb = nullptr;
  if (channel == 1) {
    meson_slot = ensure_meson_buffer(*ws.buffers, pro.pdg, tgt.pdg, sp.mass_of(pro.pdg),
                                     sp.mass_of(tgt.pdg), CascadeSpecies::iso3_of(pro.pdg),
                                     CascadeSpecies::iso3_of(tgt.pdg), sp.pi_plus_mass,
                                     sp.proton_mass, cref);
    if (meson_slot < 0) {
      ref.capacity = true;
      return -1;
    }
    mb = &ws.buffers->meson[meson_slot];
  }
  const double xs = imr::scatterer_cross_section(pro.pdg, tgt.pdg, pro.momentum, tgt.momentum,
                                                 pro.actual_mass(), tgt.actual_mass(),
                                                 pro.pdg_mass, tgt.pdg_mass, sref, mb);
  if (!(xs > 0.0)) { return 0; }

  // `collision->FinalState(trk1, trk2)`, the same two dispatches the cascade's
  // `collision_final_state` makes, with the same arguments in the same order.
  imr::ScatterFinalState fs;
  if (channel == 1) {
    fs = imr::meson_scatter_final_state(pro.pdg, tgt.pdg, pro.momentum, tgt.momentum,
                                        pro.actual_mass(), tgt.actual_mass(), pro.pdg_mass,
                                        tgt.pdg_mass, sp.mass_of(pro.pdg), sp.mass_of(tgt.pdg),
                                        sp.pi_plus_mass, sp.proton_mass,
                                        CascadeSpecies::iso3_of(pro.pdg),
                                        CascadeSpecies::iso3_of(tgt.pdg),
                                        ws.buffers->meson[meson_slot], rng, sref);
  } else {
    const int nn_slot = ensure_nn_buffer(*ws.buffers, ws.channels, ws.n_channels, pro.pdg,
                                         tgt.pdg, pro.pdg_mass, tgt.pdg_mass, cref);
    if (nn_slot < 0) {
      ref.capacity = true;
      return -1;
    }
    fs = imr::nn_scatter_final_state(ws.channels, ws.n_channels, pro.pdg, tgt.pdg, pro.momentum,
                                     tgt.momentum, pro.actual_mass(), tgt.actual_mass(),
                                     pro.pdg_mass, tgt.pdg_mass, sp.proton_mass,
                                     sp.neutron_mass, sp.pi_plus_mass, ws.buffers->nn[nn_slot],
                                     rng, sref);
  }
  rep.channel = fs.channel;
  rep.component = fs.component;
  rep.middle = fs.middle;
  rep.concrete = fs.concrete;
  if (sref.final_state) {
    // A concrete channel whose outgoing species the port's tables do not carry.
    ref.unknown_species = true;
    ref.refused_pdg = sref.pdg1;
    return -1;
  }
  // `if(!products || products->size() == 0) return products;`
  if (fs.n == 0) { return 0; }

  int q_out[2] = {0, 0};
  for (int i = 0; i < fs.n; ++i) {
    if (!h1_product_track(fs, i, pro, tgt, sp, out[i], ref)) { return -1; }
    q_out[i] = out[i].charge;
  }
  // `if(products->size() == 1) return products;` comes BEFORE the balance, so a meson-baryon
  // pair that formed one resonance is never charge-checked. It cannot fail - the resonance's
  // charge state is picked from the summed iso3 - but the check is Geant4's to skip, not ours.
  if (fs.n == 1) { return 1; }
  if (!imr::scatter_charge_balanced(pro.charge, tgt.charge, fs, q_out)) {
    ref.charge_imbalance = true;
    ref.refused_pdg = fs.pdg[0];
    return -1;
  }
  return fs.n;
}

/// `G4BinaryCascade::Propagate1H1(secondaries, nucleus)`.
///
/// `projectile` is `(*secondaries).front()`, the track `ApplyYourself` built; `target_z` is
/// `nucleus->GetCharge()`. The products go into `ws.products` in `theFinalState`'s order, each
/// with the creator id and parent-resonance fields `Propagate1H1` writes; the return is their
/// count, which is 0 exactly when `secs` ended NULL or empty (`ApplyYourself`'s outer loop then
/// tries again), or -1 for a refusal.
///
/// `ws.pool` is the working list - `secs`, which the decay pass appends to - so the function
/// puts nothing larger than one decay's four daughters on its own frame.
template <typename Rng>
__host__ __device__ inline int propagate_1h1(const CascadeTrack& projectile, int target_z,
                                             CascadeWorkspace& ws, const CascadeSpecies& sp,
                                             Rng& rng, H1Report& rep, H1Refusal& ref) {
  rep = H1Report{};
  if (ws.pool == nullptr || ws.pool_capacity < 2 || ws.products == nullptr ||
      ws.buffers == nullptr || ws.channels == nullptr) {
    ref.capacity = true;
    return -1;
  }

  // `const G4ParticleDefinition * aHTarg = G4Proton::ProtonDefinition();
  //  if (nucleus->GetCharge() == 0) aHTarg = G4Neutron::NeutronDefinition();`
  // and `G4LorentzVector mom(mass)`, which is (0, 0, 0, mass) with the PDG mass.
  CascadeTrack target;
  target.pdg = (target_z == 0) ? imr::kPdgNeutron : imr::kPdgProton;
  target.pdg_mass = sp.mass_of(target.pdg);
  target.charge = (target_z == 0) ? 0 : 1;
  target.baryon = 1;
  target.momentum = imr::LorentzVector(deex::Vec3d{0.0, 0.0, 0.0}, target.pdg_mass);
  target.position = deex::Vec3d{0.0, 0.0, 0.0};
  target.formation_time = 0.0;

  // `while(!done && tryCount++ <200)`. Each turn REPLACES `secs`: Geant4 deletes the previous
  // vector and its tracks at the top of the body, so only the last scatter survives the loop.
  bool done = false;
  int try_count = 0;
  int n_secs = 0;
  while (!done && try_count++ < 200) {
    n_secs = h1_scatter(projectile, target, ws, sp, rng, ws.pool, rep, ref);
    if (n_secs < 0) { return -1; }
    ++rep.scatters;
    for (int ss = 0; ss < n_secs; ++ss) {
      if (!kinetic_decay_knows(ws.pool[ss].pdg)) {
        ref.unknown_species = true;
        ref.refused_pdg = ws.pool[ss].pdg;
        return -1;
      }
      // "must have one resonance in final state, or it was elastic, not allowed here."
      if (kinetic_decay_is_short_lived(ws.pool[ss].pdg)) { done = true; }
    }
  }
  rep.exhausted = !done;
  rep.n_secs = n_secs;

  // The decay pass, over a list that grows while it is walked.
  int n = n_secs;
  int n_final = 0;
  for (int current = 0; current < n; ++current) {
    const CascadeTrack t = ws.pool[current];
    if (kinetic_decay_is_short_lived(t.pdg)) {
      DecayTrack in;
      in.pdg = t.pdg;
      in.momentum = t.momentum;
      in.position = t.position;
      in.formation_time = t.formation_time;
      in.creator_model_id = t.creator_model_id;
      in.parent_resonance_pdg = t.parent_resonance_pdg;
      in.parent_resonance_id = t.parent_resonance_id;
      DecayTrack dec[imr::kDecayMaxDaughters];
      KineticDecayRefusal dref;
      const int nd = kinetic_decay_one(in, dec, rng, dref);
      if (dref.any()) {
        ref.decay_refused = true;
        ref.unknown_species = ref.unknown_species || dref.unknown_species;
        ref.refused_pdg = (dref.refused_pdg != 0) ? dref.refused_pdg : t.pdg;
        return -1;
      }
      if (nd == 0) {
        // `G4KineticTrackVector * dec = (*secs)[current]->Decay(); for(jter=dec->begin(); ...`
        // on a null - see the file header. Geant4 has no answer here to port.
        ref.decay_null = true;
        ref.refused_pdg = t.pdg;
        return -1;
      }
      if (n + nd > ws.pool_capacity) {
        ref.capacity = true;
        ref.refused_pdg = t.pdg;
        return -1;
      }
      // `secs->push_back(*jter)` for each daughter, in the order `Decay()` returned them - which
      // is the REVERSE of the order the phase-space decay made them (docs/RISK.md V151), and
      // `kinetic_decay_one` already hands them back that way.
      for (int k = 0; k < nd; ++k) {
        CascadeTrack d;
        d.pdg = dec[k].pdg;
        const int si = imr::decay_species_index(d.pdg);
        if (si < 0) {
          ref.unknown_species = true;
          ref.refused_pdg = d.pdg;
          return -1;
        }
        d.pdg_mass = imr::decay_species_mass()[si];
        d.charge = imr::decay_species_charge()[si];
        d.baryon = imr::decay_species_baryon()[si];
        d.momentum = dec[k].momentum;
        d.position = dec[k].position;
        d.formation_time = dec[k].formation_time;
        d.creator_model_id = dec[k].creator_model_id;
        d.parent_resonance_pdg = dec[k].parent_resonance_pdg;
        d.parent_resonance_id = dec[k].parent_resonance_id;
        ws.pool[n++] = d;
      }
      ++rep.n_decays;
      continue;
    }
    // `theFinalState.push_back((*secs)[current])`, and then the packaging loop - done here in
    // one pass, because nothing reads theFinalState between the two.
    if (n_final >= ws.product_capacity) {
      ref.capacity = true;
      ref.refused_pdg = t.pdg;
      return -1;
    }
    CascadeProduct& p = ws.products[n_final++];
    p = CascadeProduct{};
    p.pdg = t.pdg;
    // `new G4ReactionProduct(kt->GetDefinition())` - the DEFINITION's mass, which is what
    // `G4DynamicParticle`'s (total energy, momentum) constructor compares the invariant mass
    // against in `ApplyYourself`.
    p.pdg_mass = t.pdg_mass;
    // `aNew->SetMomentum(kt->Get4Momentum().vect()); aNew->SetTotalEnergy(...e())`.
    p.momentum = t.momentum;
    // `G4ReactionProduct`'s constructor leaves `NewlyAdded` false and `Propagate1H1` never sets
    // it; nothing downstream of `ApplyYourself` reads it.
    p.newly_added = false;
    p.creator_model_id = bic_model_id();
    p.parent_resonance_pdg = t.parent_resonance_pdg;
    p.parent_resonance_id = t.parent_resonance_id;
    // The framework's (Z, A) for a nucleon, by the rule `products_add_final_state` gives: the
    // NUCLEON test, not the baryon-number one.
    p.nucleus_z = (t.pdg == imr::kPdgProton) ? 1 : 0;
    p.nucleus_a = (t.pdg == imr::kPdgProton || t.pdg == imr::kPdgNeutron) ? 1 : 0;
  }
  return n_final;
}

}  // namespace g4gpu::bic

#endif
