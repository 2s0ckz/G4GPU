// G4BinaryCascade - the entry point, and the two loops that surround the cascade.
//
// Transcribed from G4BinaryCascade.{hh,cc} (models/binary_cascade) in 11.1.1.
//
// QBBC registers `G4BinaryCascade` for protons, neutrons, pi+ and pi- from 0 to 1.5 GeV
// (`G4HadronInelasticQBBC::ConstructProcess`, four `RegisterMe(theBIC)` calls, `emaxBic =
// 1.5*CLHEP::GeV`, no `SetMinEnergy`) and for nothing else - kaons and hyperons go to
// `G4HadronicBuilder::BuildKaonsFTFP_BERT` and `BuildHyperonsFTFP_BERT`. `bic_params.cuh`
// carries the numbers; `tools/extract_bic_constants.pl` pins them in the physics list's source.
//
// ## The first statement of ApplyYourself decides which of the two models runs
//
//     if (initial4Momentum.e()-initial4Momentum.m() < theBCminP &&
//         (definition==G4Neutron::NeutronDefinition() || definition==G4Proton::ProtonDefinition()))
//         return theDeExcitation->ApplyYourself(aTrack, aNucleus);
//
// A NUCLEON below `theBCminP` = 45 MeV never enters the cascade. The whole reaction is
// `G4PreCompoundModel::ApplyYourself` - an (A+1, Z+Zp) fragment with two particle excitons of
// which exactly one is charged and one hole, then `DeExcite` - which is P6's
// `preco::apply_yourself_initial_fragment` plus `preco::deexcite`.
//
// A PION below 45 MeV does enter the cascade, because the species test is an `&&` and not an
// `||`. That asymmetry is the whole content of the constant.
//
// ## THE TWO RETRY LOOPS, AND WHY THEY ARE NOT ONE
//
//     do {                                          // OUTER: interactionCounter, 100 tries
//        theCollisionMgr->ClearAndDestroy();
//        the3DNucleus->Init(A, Z);                  // a NEW nucleus every outer turn
//        thePropagator->Init(the3DNucleus);
//        collisionLoopMaxCount = 200;
//        do {                                       // INNER: impact parameter, 200 tries
//           theCurrentTime=0;
//           radius = the3DNucleus->GetOuterRadius()+3*fermi;
//           initialPosition = GetSpherePoint(1.1*radius, initial4Momentum);
//           kt = new G4KineticTrack(definition, 0., initialPosition, initial4Momentum);
//           kt->SetState(G4KineticTrack::outside);
//           products = (A>1) ? Propagate(secondaries, the3DNucleus)
//                            : Propagate1H1(secondaries, the3DNucleus);
//        } while(! products && --collisionLoopMaxCount>0);
//        if(++interactionCounter>99) break;
//     } while(products && products->size() == 0);
//
// The inner loop turns on `Propagate` returning NULL - no collision was scheduled at all, so the
// sampled impact parameter missed - and it resamples ONLY the position: the same nucleus, the
// same field map. The outer loop turns on an EMPTY product vector - collisions were scheduled and
// every one was refused, or the excitation came out negative - and it throws the nucleus away and
// builds a new one. Merging them would rebuild a 208-nucleon nucleus for a geometric miss, which
// is both a different event and a different amount of work.
//
// `interactionCounter` is tested with `++interactionCounter>99` AFTER the inner loop and BEFORE
// the outer condition, so the outer body runs at most 100 times whatever `products` says.
//
// ## THE SAMPLED POSITION IS ONE AND A HALF RADII BACK, NOT ONE
//
// `get_sphere_point` lives in `cascade_propagate.cuh` beside the loop that calls it; the note
// is here because this is the file whose loop depends on it.
//
// `GetSpherePoint(r, mom4)` says in its own comment that it returns "a point random in the plane
// orthogonal to mom, plus -1*r*mom->vect()->unit()", and then writes
//
//     return G4ThreeVector(r*(x1*o1.unit() + x2*o2.unit() - 1.5* mom.unit()));
//
// with 1.5 and not 1. With `r = 1.1*(outerRadius + 3 fermi)` the track therefore starts about
// 1.65 outer radii upstream, which is far enough that the `outside` state is never in doubt. The
// comment is wrong and the code is what runs; ported as written.
//
// ## REFUSED, by name
//
//   * **G4BinaryCascade::Propagate1H1**, the A == 1 arm - a 230-line hand-written single-nucleon
//     reaction with its own elastic/inelastic split and its own `G4Scatterer` instance
//     (`theH1Scatterer`), not the cascade at all. `BicRefusal::hydrogen` is set at the point it
//     would have been needed. That is what makes water incomplete: `Propagate` answers for the
//     oxygen and nothing answers for the hydrogen.
//   * **every projectile that is not a nucleon or a charged pion.** Geant4 throws
//     `G4HadronicException` for one unless the environment variable
//     `I_Am_G4BinaryCascade_Developer` is set; a kernel cannot throw, so it is
//     `BicRefusal::species`.
//   * **the secondaries' creator model id ON THE PRECOMPOUND PATH.**
//     `G4PreCompoundModel::ApplyYourself` writes `aNew.SetCreatorModelID(prod->GetCreatorModelID())`
//     - the id of whichever model emitted THAT product - and P3's `deex::DeexProduct` does not
//     carry a per-product creator id. So `creator_model_id` is -1 on that path rather than the
//     24000 of `model_PRECO`, and `BicReport::creator_ids_unavailable` says so. On the CASCADE
//     path the id is known exactly - `theBIC_ID` for everything `Propagate` emits - and is set.
//   * **the secondaries' global time.** Geant4 sets `timePrimary + max(GetFormationTime(), 0)`.
//     P5's `HadProjectile` carries no global time, so `timePrimary` is zero here. For a primary
//     at global time zero - which is every call the framework makes today - that is the same
//     number, and the formation time IS carried on the cascade path.
#ifndef G4GPU_BIC_BINARY_CASCADE_CUH
#define G4GPU_BIC_BINARY_CASCADE_CUH

#include <cmath>

#include "core/units.cuh"
#include "core/vec3.cuh"
#include "physics/hadronic/bic/bic_params.cuh"
#include "physics/hadronic/bic/cascade_propagate.cuh"
#include "physics/hadronic/bic/light_ion_reaction.cuh"
#include "physics/hadronic/precompound/precompound_model.cuh"
#include "physics/hadronic/process.cuh"

namespace g4gpu::bic {

/// What `apply_yourself` could not do. `any()` is true only for the ones that make the final
/// state meaningless; the two reports below are true on every successful precompound call.
struct BicRefusal {
  /// A == 1: `Propagate1H1`, which this package does not have. Named at the point it would have
  /// been needed.
  bool hydrogen = false;
  /// Not a proton, a neutron, a pi+ or a pi-: Geant4's `G4HadronicException`.
  bool species = false;
  /// The caller's secondary buffer, or one of the cascade's own caller-owned arrays.
  bool capacity = false;
  /// P6 refused the projectile - it accepts only nucleons, which this file has already checked,
  /// so it can only mean P6's own set changed.
  bool preco_projectile = false;
  /// `G4Fancy3DNucleus::Init` could not place the nucleons.
  bool nucleus = false;
  /// Whatever `Propagate` refused - see `CascadeRefusal`, which is carried whole in `cascade`.
  bool cascade = false;
  int refused_pdg = 0;
  double refused_kin = 0.0;
  CascadeRefusal cascade_ref;
  NucleusReport nucleus_rep;

  __host__ __device__ bool any() const {
    return hydrogen || species || capacity || preco_projectile || nucleus || cascade;
  }
};

/// The two things a precompound-path call cannot supply, reported rather than invented, plus
/// what the two retry loops did.
struct BicReport {
  bool creator_ids_unavailable = false;  ///< see the file header; precompound path only
  bool times_unavailable = false;        ///< likewise
  bool no_interaction = false;           ///< both loops gave up: Geant4 returns the primary alive
  int outer_tries = 0;                   ///< `interactionCounter`
  int inner_tries = 0;                   ///< 200 - collisionLoopMaxCount, on the last outer turn
  int propagate_outcome = kPropagateOk;  ///< the last `Propagate`'s verdict
  double excitation_energy = 0.0;
  int fragment_a = 0;
  int fragment_z = 0;
};

/// The `HadFinalState` width this model instantiates. A 1.4 GeV proton on lead makes a cascade of
/// a few tens of products plus an evaporation chain; 128 is generous and the overflow is
/// reported, never silent.
inline constexpr int kBicMaxSecondaries = 128;
using BicFinalState = physics::hadronic::HadFinalState<double, kBicMaxSecondaries>;

/// Every array the cascade needs and cannot allocate. Sized by the caller for the heaviest target
/// it will see: `nucleons` and the three scratch arrays need A entries, the two field tables need
/// `kMaxFieldTable` doubles each, and the pool needs A plus whatever the cascade makes.
struct BicStorage {
  Nucleon* nucleons = nullptr;
  Nucleus3DScratch scratch;          ///< its four pointers are the caller's too
  double* proton_field = nullptr;
  double* neutron_field = nullptr;
  int field_capacity = 0;
  CascadeWorkspace cascade;
  const preco::PrecoWorkspace* preco = nullptr;
};

/// `G4BinaryCascade::ApplyYourself`.
///
/// `projectile` and `target` are P5's shapes. Geant4's test is on
/// `initial4Momentum.e() - initial4Momentum.m()`, and for a `G4HadProjectile` that is the kinetic
/// energy exactly: `InitialiseLocal` stores `(0, 0, sqrt(T(T+2m)), T+m)`, whose invariant mass is
/// `m` to the last bit, so `e - m` is `T`. Written as `kin_energy` with this note rather than as
/// a subtraction that reconstructs the same number less accurately.
///
/// The initial four-momentum is along +z, which is what P5 hands over and what makes
/// `GetSpherePoint`'s disc the x-y plane.
template <typename Rng>
__host__ __device__ inline preco::PrecoStatus apply_yourself(
    const physics::hadronic::HadProjectile<double>& projectile,
    const physics::hadronic::HadNucleus& target, const data::LevelTable& lt,
    const deex::FermiPool& pool, const preco::PrecoWorkspace& pws, BicStorage& store,
    Rng& rng, BicFinalState& result, BicRefusal& ref, BicReport& rep) {
  preco::PrecoStatus status;
  result.clear();

  const int pdg = projectile.pdg;
  const bool is_nucleon = (pdg == 2212 || pdg == 2112);
  const bool is_charged_pion = (pdg == 211 || pdg == -211);
  if (!is_nucleon && !is_charged_pion) {
    ref.species = true;
    ref.refused_pdg = pdg;
    ref.refused_kin = projectile.kin_energy;
    return status;
  }

  // `initial4Momentum.e()-initial4Momentum.m() < theBCminP && (neutron || proton)`.
  if (projectile.kin_energy < bc_min_p() && is_nucleon) {
    // `return theDeExcitation->ApplyYourself(aTrack, aNucleus)` - P6's, in two calls.
    preco::Excitons ex;
    bool preco_refused = false;
    const deex::Fragment frag = preco::apply_yourself_initial_fragment(
        pdg, projectile.kin_energy, Vec3<double>{0.0, 0.0, 1.0}, target.z, target.a, ex,
        preco_refused);
    if (preco_refused) {
      ref.preco_projectile = true;
      ref.refused_pdg = pdg;
      return status;
    }
    status = preco::deexcite(frag, ex, lt, pool, pws, rng);
    result.status = physics::hadronic::HadFinalStateStatus::kStopAndKill;
    rep.creator_ids_unavailable = true;
    rep.times_unavailable = true;
    for (int i = 0; i < status.n_products; ++i) {
      const deex::DeexProduct& p = pws.products[i];
      physics::hadronic::HadSecondary<double> s;
      s.pdg = p.pdg;
      s.z = p.z;
      s.a = p.a;
      s.time = 0.0;
      s.weight = 1.0;
      s.creator_model_id = -1;   // see the file header
      const double pdg_mass =
          (p.a > 0) ? (deex::nuclear_mass(p.a, p.z) + p.excitation)
                    : ((p.pdg == deex::kPdgElectron) ? u::electron_mass_c2<double>() : 0.0);
      capture::set_four_momentum(s, p.momentum, pdg_mass);
      if (!result.add_secondary(s)) {
        ref.capacity = true;
        break;
      }
    }
    return status;
  }

  // ------------------------------------------------------------------------------------------
  // The cascade.
  // ------------------------------------------------------------------------------------------
  if (target.a <= 1) {
    // `products = Propagate1H1(secondaries, the3DNucleus)` - refused by name; see the header.
    ref.hydrogen = true;
    ref.refused_pdg = pdg;
    ref.refused_kin = projectile.kin_energy;
    return status;
  }

  const double mass = (pdg == 2212)   ? u::proton_mass_c2<double>()
                    : (pdg == 2112) ? u::neutron_mass_c2<double>()
                                    : pdg_mass_pion_charged();
  const double e = projectile.kin_energy + mass;
  const double p_mag = std::sqrt(projectile.kin_energy *
                                 (projectile.kin_energy + 2.0 * mass));
  const imr::LorentzVector initial4(Vec3<double>{0.0, 0.0, p_mag}, e);

  CascadeSpecies sp;
  sp.proton_mass = u::proton_mass_c2<double>();
  sp.neutron_mass = u::neutron_mass_c2<double>();
  sp.pi_plus_mass = pdg_mass_pion_charged();
  sp.pi_zero_mass = pdg_mass_pion_zero();

  Nucleus3D nucleus;
  nucleus.nucleons = store.nucleons;
  nucleus.capacity = store.scratch.capacity;
  RkPropagation propagator;
  BicCascadeState st;
  PropagateResult pr;
  bool have = false;

  // The OUTER loop: a new nucleus every turn, at most 100 turns.
  int interaction_counter = 0;
  for (;;) {
    NucleusReport nrep = nucleus_init(nucleus, store.scratch, target.a, target.z, rng);
    if (nrep.fatal()) {
      ref.nucleus = true;
      ref.nucleus_rep = nrep;
      return status;
    }
    ref.nucleus_rep = nrep;
    // `thePropagator->Init(the3DNucleus)`; `Propagate` calls it a second time on the same
    // nucleus, which is the same state, so it is done once here.
    propagator = make_rk_propagation(nucleus, nrep, store.proton_field, store.neutron_field,
                                     store.field_capacity);

    // The INNER loop: a new impact parameter every turn, at most 200 turns.
    int collision_loop_max_count = 200;
    do {
      const double radius = nucleus.outer_radius() + 3.0 * deex::fermi();
      const Vec3<double> pos = get_sphere_point(1.1 * radius, initial4.v, rng);
      CascadeTrack kt;
      kt.pdg = pdg;
      kt.pdg_mass = mass;
      kt.charge = (pdg == 2212 || pdg == 211) ? 1 : ((pdg == -211) ? -1 : 0);
      kt.baryon = is_nucleon ? 1 : 0;
      kt.momentum = initial4;
      kt.position = pos;
      kt.formation_time = 0.0;
      kt.state = kOutside;          ///< `kt->SetState(G4KineticTrack::outside)`
      kt.creator_model_id = bic_model_id();
      CascadeRefusal cref;
      auto de = [&](const CascadeFragment& frag, CascadeProduct* out, int capacity) {
        bool overflow = false;
        const int n = bic_deexcite_fragment(frag, out, capacity, lt, pool, pws, rng, status,
                                            overflow);
        if (overflow) { ref.capacity = true; }
        return n;
      };
      pr = propagate(st, nucleus, propagator, sp, store.cascade, &kt, 1, nucleus.density,
                     coulomb_barrier_mev(target.a, target.z), de, rng, cref);
      ref.cascade_ref = cref;
      ++rep.inner_tries;
      // A VOID NUCLEUS is a refusal too, and not an empty result. `Propagate` reaches it when
      // the cascade has destroyed the nucleus and `FillVoidNucleusProducts` would have had to
      // run; that branch is refused by name, so the event cannot be completed. Letting it fall
      // through as an empty product vector would send the OUTER loop off to rebuild the
      // nucleus a hundred times and then return the primary alive - an event that looks like a
      // miss and is really a hole in the port.
      if (pr.outcome == kPropagateRefused || pr.outcome == kPropagateVoidNucleus) {
        ref.cascade = true;
        ref.refused_pdg = cref.refused_pdg;
        return status;
      }
      // `! products` is Geant4's NULL, which is the no-collision case and nothing else.
      have = (pr.outcome != kPropagateNoCollision);
    } while (!have && --collision_loop_max_count > 0);

    // `if(++interactionCounter>99) break;` - tested here, before the outer condition.
    if (++interaction_counter > 99) { break; }
    // `while(products && products->size() == 0)`: keep going only if we HAVE a vector and it is
    // empty. A NULL that survived 200 inner turns ends the outer loop too.
    if (!have) { break; }
    if (pr.n_products > 0) { break; }
  }
  rep.outer_tries = interaction_counter;
  rep.propagate_outcome = pr.outcome;
  rep.excitation_energy = pr.excitation_energy;
  rep.fragment_a = pr.fragment_a;
  rep.fragment_z = pr.fragment_z;

  if (!have || pr.n_products == 0) {
    // "no interaction, return primary" - `isAlive`, with the primary's own energy and direction.
    rep.no_interaction = true;
    result.status = physics::hadronic::HadFinalStateStatus::kIsAlive;
    result.energy_change = projectile.kin_energy;
    result.momentum_change = Vec3<double>{0.0, 0.0, 1.0};
    return status;
  }

  result.status = physics::hadronic::HadFinalStateStatus::kStopAndKill;
  for (int i = 0; i < pr.n_products; ++i) {
    const CascadeProduct& p = store.cascade.products[i];
    physics::hadronic::HadSecondary<double> s;
    s.pdg = p.pdg;
    s.z = p.nucleus_z;
    s.a = p.nucleus_a;
    // `G4double time=(*iter)->GetFormationTime(); if(time < 0.0) time = 0.0;` and then
    // `timePrimary + time`, with `timePrimary` zero here - see the file header.
    s.time = 0.0;
    s.weight = 1.0;
    s.creator_model_id = p.creator_model_id;
    // `new G4DynamicParticle(definition, GetTotalEnergy(), GetMomentum())`, whose mass is the
    // DEFINITION.s - so a product that left the cascade off shell is put back on it here, and
    // the kinetic energy that comes out is `e - m_PDG`, not `e - |p4|`.
    capture::set_four_momentum(s, p.momentum, p.pdg_mass);
    if (!result.add_secondary(s)) {
      ref.capacity = true;
      break;
    }
  }
  return status;
}

}  // namespace g4gpu::bic

#endif
