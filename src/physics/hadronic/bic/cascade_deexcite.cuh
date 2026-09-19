// G4BinaryCascade's exit into the precompound model, and the all-neutron remnant that bypasses it.
//
// Transcribed from G4BinaryCascade.cc's FindFragments, DeExcite, DecayVoidNucleus,
// ProductsAddFinalState and ProductsAddPrecompound, and from G4FermiPhaseSpaceDecay
// (de_excitation/fermi_breakup) - the class P3's `fermi_breakup.cuh` records as "not used by this
// model at all; only G4BinaryCascade instantiates it".
//
// ## THE EXCITON COUNT IS NOT THE PARTICLE COUNT
//
//     G4int holes    = the3DNucleus->GetMassNumber() - theTargetList.size();
//     G4int excitons = theCapturedList.size();
//     fragment->SetNumberOfHoles(holes);
//     fragment->SetNumberOfParticles(excitons);       // GF: not excitons-holes
//     fragment->SetNumberOfCharged(zCaptured);
//
// A hole is a nucleon that left the target list - knocked out, absorbed, or turned into
// something else - and a particle is a nucleon the cascade pushed INTO the nucleus and captured.
// The commented-out `excitons-holes` beside the live line is the definition the precompound model
// itself uses elsewhere; BIC hands it the raw count. And `SetNumberOfCharged` is the number of
// captured PROTONS, not the fragment's charge.
//
// ## THE PRECOMPOUND PRODUCTS COME BACK IN THE NUCLEUS'S REST FRAME AND ARE BOOSTED HERE
//
// `GetFinalNucleusMomentum` boosts the remnant into the frame where its momentum equals the
// captured nucleons' and STORES that boost in `precompoundLorentzboost`; the fragment handed to
// the precompound model is in that frame, and `ProductsAddPrecompound` multiplies every product
// by the stored boost on the way out. So a port that forgot the boost would put every evaporated
// neutron in the wrong frame and still conserve energy in that frame.
//
// ## AN ALL-NEUTRON REMNANT NEVER REACHES THE PRECOMPOUND MODEL
//
// `FindFragments` returns null for `z < 1`, and `DeExcite` then calls `DecayVoidNucleus`, which
// shares the remnant's four-momentum over the remaining nucleons with an n-body Kopylov phase
// space and returns them as products directly. It also CHEATS when there is not enough energy:
//
//     if ( eCMS < sumMass )                    // @@GF --- Cheat!!
//     { eCMS = sumMass + 2*MeV*masses.size();
//       finalP.setE(std::sqrt(finalP.vect().mag2() + sqr(eCMS))); }
//
// - two MeV per nucleon is invented, and the comment says so. Reproduced.
//
// ## REFUSED, by name
//
//   * the `A == 1` arm of `DeExcite`, which builds a single `G4ReactionProduct` from the one
//     remaining nucleon AT REST - `SetTotalEnergy(PDGMass)`, `SetMomentum(0)` - and relies on
//     `ProductsAddPrecompound`'s boost to give it its momentum. It is transcribed; what is
//     refused is the `G4HadronicException` beside it for a target list of more than one, which a
//     kernel cannot throw: `CascadeRefusal::invalid_nucleus` is set instead.
#ifndef G4GPU_BIC_CASCADE_DEEXCITE_CUH
#define G4GPU_BIC_CASCADE_DEEXCITE_CUH

#include <cmath>

#include "physics/hadronic/bic/cascade_finish.cuh"

namespace g4gpu::bic {

/// `G4FermiPhaseSpaceDecay::PtwoBody` - the same four-factor product `G4GeneralPhaseSpaceDecay::
/// Pmx` uses, but returning ZERO rather than -1 for a non-positive radicand.
__host__ __device__ inline double fps_ptwo_body(double e, double p1, double p2) {
  const double p = (e + p1 + p2) * (e + p1 - p2) * (e - p1 + p2) * (e - p1 - p2) / (4.0 * e * e);
  return (p > 0.0) ? std::sqrt(p) : 0.0;
}

/// `G4FermiPhaseSpaceDecay::BetaKopylov(K)` - a rejection sample of `chi` from
/// `sqrt(chi^N (1-chi))` with `N = 3K-5`, against `Fmax = sqrt((N/(N+1))^N/(N+1))`.
///
/// `powN` is `G4Pow::powN`, a repeated multiplication and not `std::pow`, so the port uses
/// `g4pow_pow_n` for both. Two uniforms per attempt.
///
/// MEASURED: putting `std::pow` in the `Fmax` line alone changes none of the 896 draw counts or
/// 3,584 components, because `Fmax` enters only as a multiplier on the test value and a last-bit
/// difference in it does not flip any of the ladder's acceptance decisions. It is kept as `powN`
/// anyway - a different uniform sequence would sit on one of those decisions, and the `F` line
/// beside it, where the same `powN` raises chi to N, is not inert at all.
///
/// **The loop has no iteration guard in Geant4.** `while (Fmax*flat() > F)` is all there is, so a
/// prescribed uniform sequence that never satisfies it spins forever - the port's `guard` counter
/// has no counterpart upstream. docs/RISK.md V157, and the dump's 64-value ladder exists for it.
template <typename Rng>
__host__ __device__ inline double beta_kopylov(int k, Rng& rng) {
  const int n = 3 * k - 5;
  const double xn = static_cast<double>(n);
  const double xn1 = static_cast<double>(n + 1);
  const double fmax = std::sqrt(imr::g4pow_pow_n(xn / xn1, n) / xn1);
  double chi = 0.0;
  double f = 0.0;
  int guard = 0;
  do {
    chi = rng.uniform();
    f = std::sqrt(imr::g4pow_pow_n(chi, n) * (1.0 - chi));
  } while (fmax * rng.uniform() > f && ++guard < 10000);
  return chi;
}

/// `G4FermiPhaseSpaceDecay::Decay(M, masses)` - Kopylov's method, walking the fragments from the
/// LAST to the first and boosting the remainder each time.
///
/// The parent mass is raised to `max(M, mtot + eV)` first, so a decay asked for below threshold
/// is given an electronvolt of slack rather than refused. `out` must hold `n` four-vectors.
template <typename Rng>
__host__ __device__ inline void fermi_phase_space_decay(double parent_mass, const double* masses,
                                                        int n, imr::LorentzVector* out,
                                                        Rng& rng) {
  double mtot = 0.0;
  for (int k = 0; k < n; ++k) { mtot += masses[k]; }
  double mu = mtot;
  const double ev = 1.0e-6;  // CLHEP::eV in MeV
  double mass = (parent_mass > mtot + ev) ? parent_mass : (mtot + ev);
  double t = mass - mtot;
  imr::LorentzVector p_rest_lab(deex::Vec3d{0.0, 0.0, 0.0}, mass);
  for (int k = n - 1; k > 0; --k) {
    mu -= masses[k];
    if (k > 1) {
      t *= beta_kopylov(k, rng);
    } else {
      t = 0.0;
    }
    const double rest_mass = mu + t;
    const double p_mag = fps_ptwo_body(mass, masses[k], rest_mass);
    const deex::Vec3d rand_vector = p_mag * deex::random_direction(rng);
    imr::LorentzVector p_frag_cm(rand_vector,
                                 std::sqrt(p_mag * p_mag + masses[k] * masses[k]));
    imr::LorentzVector p_rest_cm(-1.0 * rand_vector,
                                 std::sqrt(p_mag * p_mag + rest_mass * rest_mass));
    const deex::Vec3d boost = p_rest_lab.boost_vector();
    p_frag_cm.boost(boost);
    out[k] = p_frag_cm;
    p_rest_cm.boost(boost);
    p_rest_lab = p_rest_cm;
    mass = rest_mass;
  }
  out[0] = p_rest_lab;
}

/// `G4BinaryCascade::FindFragments`' answer: the residual fragment the precompound model gets, or
/// `a == 0` for "no fragment, can be neutrons only".
struct CascadeFragment {
  int a = 0;
  int z = 0;
  int holes = 0;
  int particles = 0;   ///< theCapturedList.size(), NOT particles-minus-holes
  int charged = 0;     ///< the number of captured PROTONS
  imr::LorentzVector momentum;
};

/// `G4BinaryCascade::FindFragments`.
///
/// `a` counts the target list AND the captured list; `z` counts the protons in both. A fragment
/// with `z < 1` is not made at all, which is what sends `DeExcite` to `DecayVoidNucleus`.
__host__ __device__ inline CascadeFragment find_fragments(BicCascadeState& st) {
  CascadeFragment f;
  int n_target = 0;
  int z_target = 0;
  int z_captured = 0;
  int n_captured = 0;
  for (int i = 0; i < st.lists.n_pool; ++i) {
    const CascadeTrack& t = st.lists.pool[i];
    if (t.list == kListTarget) {
      ++n_target;
      if (t.charge == 1) { ++z_target; }
    } else if (t.list == kListCaptured) {
      ++n_captured;
      if (t.charge == 1) { ++z_captured; }
    }
  }
  const int z = z_target + z_captured;
  if (z < 1) { return f; }
  f.a = n_target + n_captured;
  f.z = z;
  f.holes = st.initial_a - n_target;
  f.particles = n_captured;
  f.charged = z_captured;
  f.momentum = get_final_nucleus_momentum(st);
  return f;
}

/// One product on its way out of the cascade - `G4ReactionProduct`, only the fields BIC sets.
struct CascadeProduct {
  int pdg = 0;
  imr::LorentzVector momentum;
  bool newly_added = false;
  int creator_model_id = -1;
  int parent_resonance_pdg = 0;
  int parent_resonance_id = 0;
  /// The (Z, A) of a NUCLEUS product. `G4ReactionProduct` carries a `G4ParticleDefinition*` and
  /// an ion is a definition like any other, so Geant4 needs no extra field; the port names the
  /// species by PDG code, and a PDG ion code would have to be decoded again at the far end. Both
  /// are zero for every product the cascade itself makes - it emits only nucleons, pions and
  /// what they decay to - and are set only by the precompound exit, which emits fragments.
  int nucleus_z = 0;
  int nucleus_a = 0;
  /// `G4ReactionProduct`'s mass, which is the DEFINITION's and not `momentum.mag()`. It is what
  /// `new G4DynamicParticle(def, GetTotalEnergy(), GetMomentum())` compares the invariant mass
  /// against, and an off-shell product keeps its own mass only when the two differ by more than
  /// the constructor's tolerance - so it has to travel with the product.
  double pdg_mass = 0.0;
};

/// `G4BinaryCascade::DecayVoidNucleus` - the all-neutron remnant shared out by Kopylov phase
/// space, with the two-MeV-per-nucleon cheat its own comment names.
template <typename Rng>
__host__ __device__ inline int decay_void_nucleus(BicCascadeState& st, CascadeProduct* out,
                                                  int capacity, int bic_model_id, Rng& rng,
                                                  CascadeRefusal& ref) {
  int idx[256];
  double masses[256];
  int n = 0;
  double sum_mass = 0.0;
  // theTargetList first, then theCapturedList - the order the momenta are handed back in.
  for (int pass = 0; pass < 2; ++pass) {
    const int want = (pass == 0) ? kListTarget : kListCaptured;
    for (int i = 0; i < st.lists.n_pool && n < 256; ++i) {
      if (st.lists.pool[i].list != want) { continue; }
      idx[n] = i;
      masses[n] = st.lists.pool[i].pdg_mass;
      sum_mass += masses[n];
      ++n;
    }
  }
  if (n == 0) { return 0; }
  if (n > capacity) {
    ref.capacity = true;
    return 0;
  }
  imr::LorentzVector final_p = get_final_4momentum(st);
  double e_cms = final_p.mag();
  if (e_cms < sum_mass) {
    // "@@GF --- Cheat!!" - two MeV per nucleon, invented.
    e_cms = sum_mass + 2.0 * u::MeV<double>() * static_cast<double>(n);
    final_p.e = std::sqrt(g4gpu::mag2(final_p.v) + e_cms * e_cms);
  }
  st.precompound_boost = final_p.boost_vector();
  imr::LorentzVector momenta[256];
  fermi_phase_space_decay(e_cms, masses, n, momenta, rng);
  for (int k = 0; k < n; ++k) {
    const CascadeTrack& t = st.lists.pool[idx[k]];
    out[k].pdg = t.pdg;
    out[k].pdg_mass = t.pdg_mass;
    out[k].nucleus_z = t.charge;
    out[k].nucleus_a = t.baryon;
    out[k].momentum = momenta[k];
    out[k].creator_model_id = bic_model_id;
    out[k].parent_resonance_pdg = t.parent_resonance_pdg;
    out[k].parent_resonance_id = t.parent_resonance_id;
  }
  return n;
}

/// `G4BinaryCascade::ProductsAddFinalState` - every track in theFinalState becomes a product,
/// with `SetNewlyAdded(kt->IsParticipant())`.
///
/// `IsParticipant()` is `theNucleon && theNucleon->AreYouHit()`, so it is true only for a track
/// that IS one of the nucleus's nucleons and has been hit - which for an elastic product is
/// inherited from the entrance track (see `cascade_find.cuh`) and for a resonance product is
/// false, because that product has no nucleon.
__host__ __device__ inline int products_add_final_state(const BicCascadeState& st,
                                                        CascadeProduct* out, int n_out,
                                                        int capacity, int bic_model_id,
                                                        CascadeRefusal& ref) {
  int n = n_out;
  // `for(i = 0; i < fs.size(); i++)` - theFinalState IN ORDER, which for the port is ascending
  // `final_seq` and not pool order; see `push_final`.
  for (int seq = 0; seq < st.n_final_pushed; ++seq) {
    int idx = -1;
    for (int k = 0; k < st.lists.n_pool; ++k) {
      if (st.lists.pool[k].list == kListFinal && st.lists.pool[k].final_seq == seq) {
        idx = k;
        break;
      }
    }
    if (idx < 0) { continue; }
    const CascadeTrack& t = st.lists.pool[idx];
    if (n >= capacity) {
      ref.capacity = true;
      return n;
    }
    out[n].pdg = t.pdg;
    out[n].pdg_mass = t.pdg_mass;
    // A cascade product is a PARTICLE and Geant4 names it with a `G4ParticleDefinition*`; the
    // framework this port feeds asks for (Z, A) as well, and for a nucleon that is (charge, 1)
    // and not (0, 0). MEASURED with them left at zero: every proton and neutron the cascade
    // emitted was filed under (Z=0, A=0) - the gamma bucket - so n46_C12 came out 1,726
    // neutrons and 1,112 protons short of the oracle, at 30 and 22 sigma, with the missing
    // ones piled up somewhere the comparison could not see them.
    //
    // It is the NUCLEON test and not the baryon-number test, and that is the second measurement.
    // A Lambda has baryon number 1 and is not a nucleus; with `nucleus_a = t.baryon` it counted
    // as one nucleon in the summed (Z, A) of the event, where `G4BinaryCascade`'s own accounting
    // - and `deex_fixed_pdg`, and the dump - count only a proton, a neutron or an ion. Thirty of
    // the ninety-eight campaign cases then reported a summed (Z, A) that VARIED event to event
    // against a Geant4 answer that did not.
    out[n].nucleus_z = (t.pdg == imr::kPdgProton) ? 1 : 0;
    out[n].nucleus_a = (t.pdg == imr::kPdgProton || t.pdg == imr::kPdgNeutron) ? 1 : 0;
    out[n].momentum = t.momentum;
    out[n].newly_added = is_participant(st, t);
    out[n].creator_model_id = bic_model_id;
    out[n].parent_resonance_pdg = t.parent_resonance_pdg;
    out[n].parent_resonance_id = t.parent_resonance_id;
    ++n;
  }
  return n;
}

/// `G4BinaryCascade::ProductsAddPrecompound` - the stored boost applied to every precompound
/// product, and `SetNewlyAdded(true)` on all of them.
__host__ __device__ inline int products_add_precompound(const BicCascadeState& st,
                                                        const CascadeProduct* preco, int n_preco,
                                                        CascadeProduct* out, int n_out,
                                                        int capacity, CascadeRefusal& ref) {
  int n = n_out;
  const imr::LorentzRotation boost = imr::LorentzRotation::from_boost(st.precompound_boost);
  for (int i = 0; i < n_preco; ++i) {
    if (n >= capacity) {
      ref.capacity = true;
      return n;
    }
    out[n] = preco[i];
    out[n].momentum = boost * preco[i].momentum;
    out[n].newly_added = true;
    ++n;
  }
  return n;
}

}  // namespace g4gpu::bic

#endif
