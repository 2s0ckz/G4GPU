// G4BinaryCascade::FillVoidNucleusProducts - what the cascade hands back when it has destroyed
// the nucleus.
//
// Transcribed from G4BinaryCascade.cc (models/binary_cascade, 11.1.1), lines 2885-3068, and
// reached from `Propagate` at line 519:
//
//     if ( ! theTargetList.size() || ! nProtons ){
//         // nucleus completely destroyed, fill in ReactionProductVector
//        products = FillVoidNucleusProducts(products);
//        return products;
//     }
//
// right after the collision loop, where `nProtons` counts the protons in theTargetList ONLY - a
// nucleus whose protons were all knocked out is "void" even if the captured list holds some.
// There is no precompound model on this branch, no excitation energy and no fragment: everything
// that is left becomes a product, and the energy is balanced by hand.
//
// It was REFUSED by name from P9d to P20 (`CascadeRefusal::void_nucleus`), and measured at the
// fork point it was every refusal the cascade still made: 103 of P9d's 1.96 million nucleon and
// pion cascades, 2,416 of P9e's 800,000 ion reactions - 5.87% of C12 on C12 at 1 GeV per nucleon
// - and all 344 of the 4 GeV alpha beam's refused interactions in the B1 sweep, which read 12%
// above Geant4 with them (docs/RISK.md V212).
//
// ## WHAT IT DOES, IN ITS OWN ORDER
//
//   1. `decayKTV.Decay(&theFinalState)`, and every track in theFinalState becomes a product with
//      `SetNewlyAdded(true)` - NOT `IsParticipant()`, which is what `ProductsAddFinalState` sets
//      on the normal path. So a projectile nucleon of an ion that flew straight through and left
//      before the nucleus was destroyed comes back a CASCADER here, where the same nucleon on the
//      normal path is a spectator of `SortResult`.
//   2. every collision still scheduled is taken off the manager in time order, and one with no
//      target is asked for its final state - a DECAY runs `G4KineticTrack::Decay()` on its
//      resonance and the products are kept only `if ( lates->size() == 1 )`. A decay has two or
//      more daughters, so what this does is DRAW the decay's random numbers and throw the result
//      away; the same resonance is decayed again, for real, in step 3. Reproduced, because the
//      draws are on the stream.
//   3. `decayKTV.Decay(&theSecondaryList)` - after the collisions, because the decay deletes the
//      tracks the collisions pointed at.
//   4. `transferCorrection = theMomentumTransfer / (theSecondaryList.size() +
//      theCapturedList.size())`, added to every secondary and every captured nucleon through
//      `Update4Momentum`, which keeps each track's own invariant; those become products, the
//      secondaries with `SetNewlyAdded(IsParticipant())` and the captured with `true`.
//   5. the energy left over, `Ekinetic = theProjectile4Momentum.e() + initial_nuclear_mass -
//      Esecondaries - SumMassNucleons`, is shared equally among the target nucleons that are left
//      - neutrons only, by construction - each put on its PDG mass with that kinetic energy along
//      its own Fermi momentum's direction, and `Hit()`.
//
//      IF IT IS NOT POSITIVE, OR THERE IS NO TARGET NUCLEON LEFT, the branch invents energy: the
//      nucleons are given `(0.1 + 5*G4UniformRand()) MeV` between them (the source's own comment
//      is "leave some Energy for Nucleons") and every product's KINETIC energy is scaled by
//      `1 + (Ekinetic - Ekineticrdm)/TotalEkin` to pay for it - but only if that is a correction
//      of less than twenty per cent. Past twenty per cent nothing is scaled and the event keeps
//      the imbalance: short by `Ekinetic - Ekineticrdm`, which is an EXCESS whenever Ekinetic is
//      negative. `VoidReport` carries the branch and the numbers so that a test can assert the
//      imbalance equals exactly that.
//   6. ten passes at most over the products in REVERSE order, each rotating one product's
//      momentum towards the missing momentum at fixed magnitude, until what is missing is below
//      0.1 MeV. `SumMom` is recomputed from scratch before every single product - the loop is
//      quadratic, and the order it subtracts in is part of the answer.
//
// ## THE CAPTURED LIST IS IN CAPTURE ORDER
//
// Step 4 walks theCapturedList as a vector, and it is filled in the order nucleons were caught -
// by `Capture()` and by `DoTimeStep`'s boundary corrections, one step after another - which is
// not the order of the port's track pool. `CascadeTrack::captured_seq` is that order, handed out
// by `push_captured`, for the same reason `final_seq` exists for theFinalState. It changes which
// product comes out where AND the momentum correction of step 6, whose result depends on the
// order it walks the products in.
//
// ## REFUSED, by name
//
//   * a scheduled decay whose `G4KineticTrack::Decay()` returns 0 in step 2: the source writes
//     `lates->size()` on the null, so Geant4 has no answer to port - `void_decay_null`.
//   * the decay engine's own refusals and the caller's capacities, as everywhere in this package.
#ifndef G4GPU_BIC_CASCADE_VOID_CUH
#define G4GPU_BIC_CASCADE_VOID_CUH

#include <cmath>

#include "physics/hadronic/bic/cascade_deexcite.cuh"

namespace g4gpu::bic {

/// CLHEP's `Hep3Vector::unit()`: `p *= (1.0/std::sqrt(tot))` for a non-zero vector and the ZERO
/// vector returned unchanged for a zero one - not `g4gpu::normalize`'s (0, 0, 1). Written here and
/// not borrowed from P3's `deex::clhep_unit`, so that this file keeps the cascade independent of
/// the de-excitation package, which is the rule `propagate`'s `de_excite` parameter exists for.
__host__ __device__ inline deex::Vec3d void_unit(const deex::Vec3d& v) {
  const double tot = g4gpu::mag2(v);
  if (!(tot > 0.0)) { return v; }
  const double inv = 1.0 / std::sqrt(tot);
  return deex::Vec3d{v.x * inv, v.y * inv, v.z * inv};
}

/// What `FillVoidNucleusProducts` did, beyond the products themselves. Every field is what a
/// test needs to ASSERT this branch rather than excuse it.
struct VoidReport {
  bool ran = false;
  int n_final = 0;          ///< products from theFinalState, after its decay pass
  /// theCapturedList walked in capture order visited the pool OUT of its order at least once.
  bool captured_reordered = false;
  int n_late_decays = 0;    ///< scheduled decays whose `Decay()` step 2 ran and discarded
  int n_lates = 0;          ///< products from step 2 - a decay with exactly one daughter
  int n_secondaries = 0;    ///< products from theSecondaryList, after its decay pass
  int n_captured = 0;
  int n_targets = 0;        ///< the nucleons left in theTargetList - all neutrons
  /// The cascade's state as the branch found it - `theMomentumTransfer`,
  /// `theProjectile4Momentum`, `initial_nuclear_mass` and (currentA, currentZ) - which the dump
  /// reads off Geant4's own members after the event, so that a divergence can be placed before
  /// the branch or inside it.
  deex::Vec3d momentum_transfer{0.0, 0.0, 0.0};
  imr::LorentzVector projectile_4mom;
  double initial_nuclear_mass = 0.0;
  int current_a = 0;
  int current_z = 0;
  deex::Vec3d transfer{0.0, 0.0, 0.0};  ///< `transferCorrection`
  /// 0 - `Ekinetic` was positive and there were target nucleons to share it among;
  /// 1 - the scaling branch, with the correction applied (under 20%);
  /// 2 - the scaling branch, with the correction REFUSED (20% or more): the event keeps
  ///     `energy_residue` of imbalance.
  int branch = -1;
  double e_initial = 0.0;           ///< `theProjectile4Momentum.e() + initial_nuclear_mass`
  double e_secondaries = 0.0;       ///< `Esecondaries`, before the target nucleons are added
  double ekinetic_available = 0.0;  ///< `Ekinetic` as first computed, before either branch
  double ekinetic_rdm = 0.0;        ///< `(0.1 + 5*U) MeV`, drawn only with target nucleons left
  double total_ekin = 0.0;          ///< `TotalEkin`
  double correction = 1.0;
  double ekinetic_per_target = 0.0;
  /// What FillVoidNucleusProducts' own arithmetic leaves the event short by: exactly zero on
  /// branches 0 and 1 (to rounding), and `ekinetic_available - ekinetic_rdm` on branch 2.
  double energy_residue = 0.0;
  int momentum_loops = 0;           ///< `loopcount`, at most 10
  double momentum_left = 0.0;       ///< `SumMom.mag()` at the loop's last test
};

/// `G4BinaryCascade::FillVoidNucleusProducts`'s product, as `G4ReactionProduct` holds it: the
/// definition's mass travels with it because steps 5 and 6 read the KINETIC energy, which
/// `SetTotalEnergy` stores as `totalEnergy - mass`.
__host__ __device__ inline CascadeProduct void_product_of(const CascadeTrack& t, int creator_id,
                                                          bool newly_added) {
  CascadeProduct p;
  p.pdg = t.pdg;
  p.pdg_mass = t.pdg_mass;
  // The nucleon rule `products_add_final_state` gives, for the reason it gives: a proton is
  // (1, 1) and a neutron (0, 1), and nothing else the cascade emits is a nucleus.
  p.nucleus_z = (t.pdg == imr::kPdgProton) ? 1 : 0;
  p.nucleus_a = (t.pdg == imr::kPdgProton || t.pdg == imr::kPdgNeutron) ? 1 : 0;
  // `SetMomentum(Get4Momentum().vect()); SetTotalEnergy(Get4Momentum().e());`
  p.momentum = t.momentum;
  p.newly_added = newly_added;
  p.creator_model_id = creator_id;
  p.parent_resonance_pdg = t.parent_resonance_pdg;
  p.parent_resonance_id = t.parent_resonance_id;
  return p;
}

/// A cascade track built from one of the decay engine's daughters: `new G4KineticTrack(aProduct,
/// formationTime, position, momentum)` in `G4KineticTrack::Decay`, with a null `theNucleon` - so
/// `IsParticipant()` is true on it - and the creator and parent resonance `Decay` sets.
__host__ __device__ inline bool void_daughter_track(const DecayTrack& d, CascadeTrack& t,
                                                    CascadeRefusal& ref) {
  const int si = imr::decay_species_index(d.pdg);
  if (si < 0) {
    ref.unknown_species = true;
    ref.refused_pdg = d.pdg;
    return false;
  }
  t = CascadeTrack{};
  t.pdg = d.pdg;
  t.pdg_mass = imr::decay_species_mass()[si];
  t.charge = imr::decay_species_charge()[si];
  t.baryon = imr::decay_species_baryon()[si];
  t.momentum = d.momentum;
  t.position = d.position;
  t.formation_time = d.formation_time;
  t.creator_model_id = d.creator_model_id;
  t.parent_resonance_pdg = d.parent_resonance_pdg;
  t.parent_resonance_id = d.parent_resonance_id;
  t.nucleon_index = -1;
  t.hit = false;
  return true;
}

__host__ __device__ inline DecayTrack void_decay_input(const CascadeTrack& t) {
  DecayTrack in;
  in.pdg = t.pdg;
  in.momentum = t.momentum;
  in.position = t.position;
  in.formation_time = t.formation_time;
  in.creator_model_id = t.creator_model_id;
  in.parent_resonance_pdg = t.parent_resonance_pdg;
  in.parent_resonance_id = t.parent_resonance_id;
  return in;
}

/// Folds the decay engine's refusal into the cascade's exactly as the normal path's decay pass
/// in `propagate` does - `unknown_species` and `list_full`, and the code - so that the two
/// branches of `Propagate` name the same thing the same way.
__host__ __device__ inline void void_fold_decay_refusal(const KineticDecayRefusal& kref,
                                                        CascadeRefusal& ref) {
  if (!kref.any()) { return; }
  ref.unknown_species = ref.unknown_species || kref.unknown_species;
  ref.capacity = ref.capacity || kref.list_full;
  ref.refused_pdg = kref.refused_pdg;
}

/// `decayKTV.Decay(&theFinalState)` or `decayKTV.Decay(&theSecondaryList)`, done IN THE POOL.
///
/// `G4DecayKineticTracks::Decay` walks the vector forwards while it grows, appends every decay's
/// daughters at the end, nulls the parent, and erases the nulls afterwards - so a daughter that
/// is itself short-lived is decayed when the walk reaches it, and what is left is the survivors
/// in their order and then the daughters, generation by generation. The port's `list` is either
/// the final state, whose order is `final_seq`, or the secondary list, whose order is the pool's;
/// a daughter appended to the POOL, and given the next `final_seq` when the list is the final
/// state, is at the end of that order either way, so the same walk over the pool is the same
/// pass. It is done in place, and not through `decay_kinetic_tracks` on a copy, so that no
/// 256-entry `DecayTrack` array - 25 kB - joins the cascade's kernel frame (docs/RISK.md V183).
///
/// Returns false on a refusal that leaves the list unusable; `ref` says which.
template <typename Rng>
__host__ __device__ inline bool void_decay_list(BicCascadeState& st, int list, Rng& rng,
                                                CascadeRefusal& ref) {
  DecayTrack daughters[imr::kDecayMaxDaughters];
  const bool final_list = (list == kListFinal);
  // The walk's position is an index into the list's ORDER: `final_seq` for the final state, the
  // pool index for the secondary list. Both grow as daughters are appended.
  for (int pos = 0; pos < (final_list ? st.n_final_pushed : st.lists.n_pool); ++pos) {
    int i = -1;
    if (final_list) {
      for (int k = 0; k < st.lists.n_pool; ++k) {
        if (st.lists.pool[k].list == kListFinal && st.lists.pool[k].final_seq == pos) {
          i = k;
          break;
        }
      }
    } else if (st.lists.pool[pos].list == kListSecondary) {
      i = pos;
    }
    if (i < 0) { continue; }
    const int pdg = st.lists.pool[i].pdg;
    if (!kinetic_decay_is_short_lived(pdg)) {
      if (!kinetic_decay_knows(pdg)) {
        ref.unknown_species = true;
        ref.refused_pdg = pdg;
      }
      continue;
    }
    KineticDecayRefusal kref;
    const int nd = kinetic_decay_one(void_decay_input(st.lists.pool[i]), daughters, rng, kref);
    void_fold_decay_refusal(kref, ref);
    if (nd == 0) { continue; }   // Geant4's NULL from `Decay()`: the parent stays in the list
    if (st.lists.n_pool + nd > st.lists.capacity) {
      ref.capacity = true;
      ref.refused_pdg = pdg;
      return false;
    }
    for (int k = 0; k < nd; ++k) {
      CascadeTrack t;
      if (!void_daughter_track(daughters[k], t, ref)) { return false; }
      // A daughter keeps its parent's state: the parent was in this list, and the state is
      // read by nothing downstream of this branch.
      t.state = st.lists.pool[i].state;
      const int added = st.lists.add(t, final_list ? kListNone : kListSecondary);
      if (added < 0) {
        ref.capacity = true;
        return false;
      }
      if (final_list) { push_final(st, added); }
    }
    st.lists.pool[i].list = kListNone;   // `delete track; (*tracks)[i] = nullptr;`
  }
  return true;
}

/// `G4BinaryCascade::FillVoidNucleusProducts(products)`.
///
/// `products` receives the list; returns how many, or -1 on a refusal (`ref` says which). `rng`
/// is the cascade's stream: steps 1-3 draw through the decay engine and step 5 draws one uniform
/// when the scaling branch runs with target nucleons left.
///
/// `__noinline__`, so that its locals are a frame of their own on top of `propagate`'s at the one
/// point it is called, rather than a permanent part of `propagate`'s frame for the whole cascade
/// loop - the same reason `bic_deexcite_fragment` is, and a smaller one: the deepest call chain
/// under `propagate` is still the de-excitation's (docs/RISK.md V183, V196).
template <typename Rng>
__host__ __device__ __noinline__ int fill_void_nucleus_products(BicCascadeState& st,
                                                                imr::CollisionList& colls,
                                                                CascadeProduct* products,
                                                                int capacity, int bic_id,
                                                                Rng& rng, CascadeRefusal& ref,
                                                                VoidReport& vr) {
  vr = VoidReport{};
  vr.ran = true;
  vr.momentum_transfer = st.momentum_transfer;
  vr.projectile_4mom = st.projectile_4mom;
  vr.initial_nuclear_mass = st.initial_nuclear_mass;
  vr.current_a = st.current_a;
  vr.current_z = st.current_z;
  int n = 0;
  auto push = [&](const CascadeProduct& p) {
    if (n >= capacity) {
      ref.capacity = true;
      return false;
    }
    products[n++] = p;
    return true;
  };
  double e_secondaries = 0.0;   // `Esecondaries`

  // ---- 1. `decayKTV.Decay(&theFinalState)`, then theFinalState in its order, all
  // `SetNewlyAdded(true)` and all `theBIC_ID`.
  if (!void_decay_list(st, kListFinal, rng, ref)) { return -1; }
  for (int seq = 0; seq < st.n_final_pushed; ++seq) {
    int i = -1;
    for (int k = 0; k < st.lists.n_pool; ++k) {
      if (st.lists.pool[k].list == kListFinal && st.lists.pool[k].final_seq == seq) {
        i = k;
        break;
      }
    }
    if (i < 0) { continue; }
    const CascadeTrack& t = st.lists.pool[i];
    if (!push(void_product_of(t, bic_id, true))) { return -1; }
    e_secondaries += t.momentum.e;
    ++vr.n_final;
  }

  // ---- 2. "pull out late particles from collisions", earliest first, every one removed.
  while (colls.size() > 0) {
    const int next = colls.next_collision();
    if (next < 0) { break; }
    const imr::CollisionInitialState c = colls.items[next];
    if (c.n_targets() == 0) {
      // `G4KineticTrackVector * lates = collision->GetFinalState();` - `G4BCDecay` hands back
      // `aProjectile->Decay()` and `G4BCLateParticle` a copy of the projectile.
      const CascadeTrack& pro = st.lists.pool[c.primary];
      if (c.generator == kGenLateParticle) {
        // "late particle @ void Nucl": one track, so it is kept, with its OWN creator id.
        if (!push(void_product_of(pro, pro.creator_model_id, true))) { return -1; }
        e_secondaries += pro.momentum.e;
        ++vr.n_lates;
      } else {
        DecayTrack out[imr::kDecayMaxDaughters];
        KineticDecayRefusal kref;
        const int nd = kinetic_decay_one(void_decay_input(pro), out, rng, kref);
        ++vr.n_late_decays;
        if (nd == 0) {
          // `lates->size()` on a null. See the file header.
          ref.void_decay_null = true;
          ref.refused_pdg = pro.pdg;
          return -1;
        }
        void_fold_decay_refusal(kref, ref);
        if (nd == 1) {
          // `SetCreatorModelID(atrack->GetCreatorModelID())` and the parent resonance, both of
          // which `Decay()` set on its daughter from the parent.
          CascadeTrack d;
          if (!void_daughter_track(out[0], d, ref)) { return -1; }
          if (!push(void_product_of(d, d.creator_model_id, true))) { return -1; }
          e_secondaries += d.momentum.e;
          ++vr.n_lates;
        }
        // Two or more daughters: dropped, and the draws that made them stay drawn.
      }
    }
    colls.remove(next);
  }

  // ---- 3. `decayKTV.Decay(&theSecondaryList)`, after the collisions are gone.
  if (!void_decay_list(st, kListSecondary, rng, ref)) { return -1; }

  // ---- 4. the momentum the field took up, shared over the secondaries and the captured.
  int n_sec = 0;
  int n_cap = 0;
  for (int i = 0; i < st.lists.n_pool; ++i) {
    if (st.lists.pool[i].list == kListSecondary) { ++n_sec; }
    if (st.lists.pool[i].list == kListCaptured) { ++n_cap; }
  }
  deex::Vec3d transfer{0.0, 0.0, 0.0};
  if (n_sec + n_cap > 0) {
    // CLHEP's `operator/(Hep3Vector, double)` is `v1 * (1.0/c)`.
    const double inv = 1.0 / static_cast<double>(n_sec + n_cap);
    transfer = inv * st.momentum_transfer;
  }
  vr.transfer = transfer;
  // `(*iter)->Update4Momentum((*iter)->Get4Momentum().vect()+transferCorrection)` - the energy
  // rebuilt from the track's OWN invariant, `sqrt(theTotal4Momentum.mag2() + p'.mag2())`, so an
  // off-shell track keeps exactly the amount by which it was off.
  auto update4 = [&](CascadeTrack& t) {
    const double m2 = t.momentum.e * t.momentum.e - g4gpu::mag2(t.momentum.v);
    const deex::Vec3d p = t.momentum.v + transfer;
    t.momentum = imr::LorentzVector(p, std::sqrt(m2 + g4gpu::mag2(p)));
  };
  for (int i = 0; i < st.lists.n_pool; ++i) {
    CascadeTrack& t = st.lists.pool[i];
    if (t.list != kListSecondary) { continue; }
    update4(t);
    // `if ( (*iter)->IsParticipant() ) aNew->SetNewlyAdded(true);` - and false otherwise, which
    // is a projectile nucleon of an ion that nothing touched: a spectator for `SortResult`.
    if (!push(void_product_of(t, bic_id, is_participant(st, t)))) { return -1; }
    e_secondaries += t.momentum.e;
    ++vr.n_secondaries;
  }
  int last_captured_index = -1;
  for (int seq = 0; seq < st.n_captured_pushed; ++seq) {
    const int i = captured_at(st, seq);
    if (i < 0) { continue; }
    // Whether theCapturedList's order is NOT the pool's - the only events on which walking it in
    // capture order changes an answer, and the ones a tape has to reach to say it does.
    if (i < last_captured_index) { vr.captured_reordered = true; }
    last_captured_index = i;
    CascadeTrack& t = st.lists.pool[i];
    update4(t);
    if (!push(void_product_of(t, bic_id, true))) { return -1; }
    e_secondaries += t.momentum.e;
    ++vr.n_captured;
  }
  vr.e_secondaries = e_secondaries;

  // ---- 5. what is left of the energy, to the nucleons that are left.
  double sum_mass_nucleons = 0.0;
  int n_tgt = 0;
  for (int i = 0; i < st.lists.n_pool; ++i) {
    const CascadeTrack& t = st.lists.pool[i];
    if (t.list != kListTarget) { continue; }
    sum_mass_nucleons += t.pdg_mass;
    ++n_tgt;
  }
  vr.n_targets = n_tgt;
  vr.e_initial = st.projectile_4mom.e + st.initial_nuclear_mass;
  // `theProjectile4Momentum.e() + initial_nuclear_mass - Esecondaries - SumMassNucleons`, left to
  // right.
  double ekinetic = st.projectile_4mom.e + st.initial_nuclear_mass - e_secondaries -
                    sum_mass_nucleons;
  vr.ekinetic_available = ekinetic;
  if (ekinetic > 0.0 && n_tgt > 0) {
    vr.branch = 0;
    ekinetic /= static_cast<double>(n_tgt);
  } else {
    double ekinetic_rdm = 0.0;
    // "leave some  Energy for Nucleons" - drawn only when there is a nucleon to give it to.
    if (n_tgt > 0) { ekinetic_rdm = (0.1 + rng.uniform() * 5.0) * u::MeV<double>(); }
    double total_ekin = ekinetic_rdm;
    for (int k = 0; k < n; ++k) {
      // `GetKineticEnergy()`, which `SetTotalEnergy` stored as `totalEnergy - mass`.
      total_ekin += products[k].momentum.e - products[k].pdg_mass;
    }
    double correction = 1.0;
    // `20*perCent` is `20 * 0.01` in double, evaluated before the multiplication.
    if (std::fabs(ekinetic) < (20.0 * 0.01) * total_ekin) {
      // "Ekinetic < 0 == IS < FS, need to reduce energies"
      correction = 1.0 + (ekinetic - ekinetic_rdm) / total_ekin;
      vr.branch = 1;
    } else {
      vr.branch = 2;
    }
    // EVERY product is rebuilt, including when the correction is exactly 1: `SetKineticEnergy`
    // puts it back on its PDG mass with `T = E - m`, and `GetTotalMomentum()` is
    // `sqrt(|T*(E+m)|)`, not the momentum it had.
    for (int k = 0; k < n; ++k) {
      CascadeProduct& p = products[k];
      const double m = p.pdg_mass;
      const double t_new = (p.momentum.e - m) * correction;
      const double e_new = t_new + m;
      const double p_mag = std::sqrt(std::fabs(t_new * (e_new + m)));
      p.momentum = imr::LorentzVector(p_mag * void_unit(p.momentum.v), e_new);
    }
    ekinetic = ekinetic_rdm * correction;
    if (n_tgt > 0) { ekinetic /= static_cast<double>(n_tgt); }
    vr.ekinetic_rdm = ekinetic_rdm;
    vr.total_ekin = total_ekin;
    vr.correction = correction;
    if (vr.branch == 2) { vr.energy_residue = vr.ekinetic_available - ekinetic_rdm; }
  }
  vr.ekinetic_per_target = ekinetic;
  for (int i = 0; i < st.lists.n_pool; ++i) {
    CascadeTrack& t = st.lists.pool[i];
    if (t.list != kListTarget) { continue; }
    // "set Nucleon it to be hit - as it is in fact"
    mark_hit(st, i);
    CascadeProduct p = void_product_of(t, bic_id, true);
    const double m = t.pdg_mass;
    // `aNew->SetKineticEnergy(Ekinetic)` on a fresh product, then
    // `SetMomentum(aNew->GetTotalMomentum() * ((*iter)->Get4Momentum().vect().unit()))`.
    const double e_new = ekinetic + m;
    const double p_mag = std::sqrt(std::fabs(ekinetic * (e_new + m)));
    p.momentum = imr::LorentzVector(p_mag * void_unit(t.momentum.v), e_new);
    if (!push(p)) { return -1; }
  }

  // ---- 6. the momentum, at fixed magnitudes, towards what is missing.
  deex::Vec3d p_sum{0.0, 0.0, 0.0};
  for (int k = 0; k < n; ++k) { p_sum = p_sum + products[k].momentum.v; }
  // `initial4Mom = theProjectile4Momentum + G4LorentzVector(initial_nuclear_mass)`: the nucleus
  // adds energy and a ZERO three-vector, and the zero is added as CLHEP adds it, so that a
  // component that was -0.0 comes out +0.0 as it does there.
  const deex::Vec3d p_initial = st.projectile_4mom.v + deex::Vec3d{0.0, 0.0, 0.0};
  deex::Vec3d sum_mom = p_initial - p_sum;
  int loopcount = 0;
  while (std::sqrt(g4gpu::mag2(sum_mom)) > 0.1 * u::MeV<double>() && loopcount++ < 10) {
    // "reverse_iterator reverse - start to correct last added first"
    for (int k = n - 1; k >= 0; --k) {
      sum_mom = p_initial;
      for (int j = 0; j < n; ++j) { sum_mom = sum_mom - products[j].momentum.v; }
      const double p_mag = std::sqrt(g4gpu::mag2(products[k].momentum.v));
      products[k].momentum.v = p_mag * void_unit(products[k].momentum.v + sum_mom);
    }
  }
  vr.momentum_loops = loopcount;
  vr.momentum_left = std::sqrt(g4gpu::mag2(sum_mom));
  return n;
}

}  // namespace g4gpu::bic

#endif
