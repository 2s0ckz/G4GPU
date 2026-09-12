// G4VPartonStringModel::Scatter and G4TheoFSGenerator::ApplyYourself - the FTFP entry point.
//
// Transcribed from
//   source/processes/hadronic/models/parton_string/management/src/G4VPartonStringModel.cc
//     Scatter - the rotation to z, the 1000-attempt loop, the string-vector assembly, the
//     wounded-nucleus rotation back to the lab, the unphysical-residual rejection table, and
//     the `SumMass > InvMass` check
//   source/processes/hadronic/models/theo_high_energy/src/G4TheoFSGenerator.cc
//     ApplyYourself - the two dummy low-energy branches, the Scatter call, the hitCount test
//     that selects Propagate versus the decay-only path, PropagateNuclNucl, and the
//     G4HadFinalState assembly
//   source/processes/hadronic/models/parton_string/hadronization/src/G4ExcitedStringDecay.cc
//     FragmentStrings - P11's ftf_fragment_strings, called from here
//   source/processes/hadronic/models/de_excitation/.../G4GeneratorPrecompoundInterface.cc
//     Propagate - P6's preco::propagate_residual, called from here
//
// ## The retry loop is the model, not a safety net
//
// `Scatter` runs `Init` + `GetStrings` + `FragmentStrings` up to 1000 times and keeps the first
// attempt that produces a non-empty string list, a PHYSICAL nuclear residual, a non-null
// fragmentation result and `SumMass <= InvMass`. Every one of those four can fail on a
// perfectly ordinary event - the residual test alone rejects any configuration that would leave
// 4 protons and no neutrons - so a typical interaction takes more than one attempt and each
// attempt REBUILDS BOTH NUCLEI from scratch. That is why `Init` is inside the loop and why the
// deviate count of one `ftf::apply_yourself` call is not a function of the physics alone.
//
// After 1000 attempts Geant4 raises a JustWarning, resets the nuclei to their ground states,
// erases every `AreYouHit` mark, and returns THE PRIMARY UNCHANGED as a single kinetic track
// at `z = 2 * OuterRadius`. That is an elastic-looking final state from an inelastic process,
// and `kScatterAttemptsExhausted` is set so it cannot be mistaken for one.
//
// ## Where the projectile's direction goes
//
// `Scatter` rotates the primary onto +z with `toZ` and does all of FTF in that frame, then
// rotates the strings and the wounded nucleons back with `toLab`. But `G4HadProjectile`'s own
// constructor has ALREADY put the projectile along +z (`G4HadProjectile::InitialiseLocal`
// stores `(0, 0, sqrt(T(T+2m)), T+m)` - the same observation P9 records for
// `G4BinaryLightIonReaction`), so `toZ` is the identity for every call that comes through
// `G4HadronicProcess`. It is transcribed because `Scatter` is also callable directly and
// because an identity rotation that is assumed is a bug waiting for a caller.
//
// ## The two exits, and QBBC takes the first
//
// `ApplyYourself` chooses between `theTransport->Propagate` (P6's interface) and
// `theDecay.Propagate` by counting hit nucleons: if EVERY nucleon of the target was hit there
// is no residual nucleus and the secondaries are simply decayed. The second path is
// `G4DecayStrongResonances`, which QBBC reaches only for a hydrogen or very light target, and
// it is REFUSED BY NAME (`kDecayStrongResonances`) rather than approximated by the first.
#pragma once
#include <cmath>

#include "core/units.cuh"
#include "physics/hadronic/ftf/ftf_model.cuh"
#include "physics/hadronic/ftf/string_fragmentation.cuh"
#include "physics/hadronic/precompound/generator_interface.cuh"
#include "physics/hadronic/process.cuh"

namespace g4gpu::hadronic::ftf {

// P5's process framework lives in `g4gpu::physics::hadronic` and this package in
// `g4gpu::hadronic::ftf` - two different trees, because P5's header was written before the
// hadronic namespace settled. The aliases are here rather than a `using namespace` so that the
// entry point's signature reads as P5 declared it.
template <typename real_t>
using HadProjectile = g4gpu::physics::hadronic::HadProjectile<real_t>;
using HadNucleus = g4gpu::physics::hadronic::HadNucleus;
template <typename real_t, int N>
using HadFinalState = g4gpu::physics::hadronic::HadFinalState<real_t, N>;
template <typename real_t>
using HadSecondary = g4gpu::physics::hadronic::HadSecondary<real_t>;
using HadFinalStateStatus = g4gpu::physics::hadronic::HadFinalStateStatus;
using g4gpu::physics::hadronic::pdg_nuclear_code;

/// What `ftf::apply_yourself` reports about one interaction.
struct FtfApplyReport {
  FtfRefusal refused = FtfRefusal::kNone;
  FtfModelReport model;                 ///< the last attempt's model report
  preco::GeneratorRefusal generator;    ///< P6's
  int attempts = 0;                     ///< G4VPartonStringModel::Scatter's `attempts`
  int fragment_retries = 0;             ///< attempts FragmentStrings answered with a null list
  /// WHY the retry loop turned round, one counter per `Success = false`. These are here for the
  /// same reason G4FTFModel's three public counters are (docs/RISK.md V99): `attempts` says
  /// that a case retries and says nothing about which of the four conditions did it, and the
  /// four are in completely different parts of the model.
  int retries_no_strings = 0;           ///< GetStrings produced none
  int retries_residual = 0;             ///< the unphysical-residual table rejected the collision
  int retries_residual_target = 0;      ///< ...because of the TARGET residual
  int retries_residual_projectile = 0;  ///< ...because of the PROJECTILE residual (ion beams)
  int retries_summass = 0;              ///< SumMass > InvMass, or SumMass == 0
  /// A refusal the SUCCESSFUL fragmentation reported and that Geant4 keeps the result for -
  /// `kFragmentLoopExhausted` or `kEnergyCorrectorFailed` on a string that still produced
  /// hadrons. Carried separately from `refused`, which aborts.
  FtfRefusal fragment_report = FtfRefusal::kNone;
  bool attempts_exhausted = false;
  bool primary_returned_unchanged = false;  ///< the 1000-attempt fallback fired
  bool track_capacity = false;
  bool secondary_overflow = false;
  bool low_energy_dummy = false;        ///< the charm/bottom or hypernucleus <100 MeV branch

  __host__ __device__ bool any() const {
    return refused != FtfRefusal::kNone || model.any() || generator.any() ||
           attempts_exhausted || track_capacity || secondary_overflow;
  }
};

/// The state one `ftf::apply_yourself` call needs, beyond the model's own.
///
/// `kMaxTracks` bounds the hadrons one interaction can produce: P11's string-decay workspace
/// caps a single FragmentStrings call at `kMaxOut` hadrons, and this list holds the same set
/// converted to P6's `CascadeTrack` plus whatever `Propagate` lets escape.
template <int kMaxTargetA = bic::kMaxNucleons, int kMaxProjA = bic::kMaxNucleons,
          int kMaxInteractions = 1024, int kMaxStrings = 2 * bic::kMaxNucleons + 2,
          int kMaxTracks = 256, int kPerString = 96>
struct FtfWorkspace {
  FtfModelWorkspace<kMaxTargetA, kMaxProjA, kMaxInteractions, kMaxStrings> model;
  StringsWorkspace<double, kMaxTracks, kPerString> strings;

  preco::CascadeTrack tracks[kMaxTracks];
  int n_tracks = 0;
  preco::CascadeTrack escaped[kMaxTracks];
  int n_escaped = 0;
  preco::HitNucleon hit[kMaxTargetA];
  int n_hit = 0;

  FtfApplyReport report;
};

/// `G4KineticTrack` -> `preco::CascadeTrack`, which is the whole of what
/// `G4GeneratorPrecompoundInterface::Propagate` reads off a secondary.
__host__ __device__ inline preco::CascadeTrack ftf_track_from_hadron(const FragHadron& h,
                                                                     int model_id,
                                                                     FtfRefusal* refused) {
  preco::CascadeTrack t;
  const data::FtfHadron* d = data::ftf_find_hadron(h.pdg);
  if (d == nullptr) {
    if (refused != nullptr) { *refused = FtfRefusal::kUnknownHadronCode; }
    return t;
  }
  t.pdg = h.pdg;
  // `G4int(GetPDGCharge()/eplus + 0.1)` - the form every consumer of a charge uses, and it is a
  // truncation of charge+0.1 and not a round.
  t.charge = static_cast<int>(d->charge + ((d->charge < 0.0) ? -0.1 : 0.1));
  t.momentum = h.momentum;
  t.position = h.position;
  t.formation_time = h.formation_time;
  t.creator_model_id = model_id;
  t.is_short_lived = d->shortlived;
  return t;
}

/// G4VPartonStringModel::Scatter's unphysical-residual table.
///
/// "1 (H), 2 (2He) and 3 (3Li) protons alone without neutrons can exist, but not more; no bound
/// states of 2 or more neutrons without protons can exist." The projectile half of the table
/// adds six hypernuclear combinations; with an anti-nucleus projectile refused and a
/// hypernucleus refused, `n_lambda` is always 0 here and only the first two of its seven
/// clauses can fire - but all seven are transcribed, because the clause that fires depends on a
/// count this function is given rather than on what the caller happens to support.
__host__ __device__ inline bool ftf_unphysical_residual(int n_proton_t, int n_neutron_t,
                                                        int n_proton_p, int n_neutron_p,
                                                        int n_lambda_p, bool has_projectile,
                                                        bool* by_target = nullptr,
                                                        bool* by_projectile = nullptr) {
  bool unphysical = false;
  if ((n_proton_t > 3 && n_neutron_t == 0) || (n_proton_t == 0 && n_neutron_t > 1)) {
    unphysical = true;
    if (by_target != nullptr) { *by_target = true; }
  }
  if (!has_projectile) { return unphysical; }
  if ((n_proton_p > 3 && n_neutron_p == 0) ||
      (n_proton_p == 0 && n_neutron_p > 1 && n_lambda_p == 0) ||
      (n_proton_p == 0 && n_neutron_p <= 1 && n_lambda_p > 0) ||
      (n_proton_p == 0 && n_neutron_p > 2 && n_lambda_p > 0) || (n_lambda_p > 2) ||
      (n_proton_p > 0 && n_neutron_p == 0 && n_lambda_p > 0) ||
      (n_proton_p > 1 && n_neutron_p > 1 && n_lambda_p > 1)) {
    unphysical = true;
    if (by_projectile != nullptr) { *by_projectile = true; }
  }
  return unphysical;
}

/// G4VPartonStringModel::Scatter.
///
/// On success the hadrons are in `ws->strings.out[0 .. n_out)` in the LAB frame, the target
/// nucleus carries the wounded nucleons with their lab momenta, and `ws->report.attempts` says
/// how many rebuilds it took.
template <int kA, int kP, int kI, int kS, int kT, int kPS, typename Rng>
__host__ __device__ inline bool ftf_scatter(FtfWorkspace<kA, kP, kI, kS, kT, kPS>* ws,
                                            const xs::Projectile<double>& proj,
                                            const Vec4& primary_p4, int target_a, int target_z,
                                            const LundTables<double>* lund, Rng& rng) {
  // Rotate the primary onto +z. Identity for a G4HadProjectile, and transcribed anyway.
  LorentzRot to_z;
  lorentz_rotate_z(&to_z, -lv_phi(primary_p4));
  lorentz_rotate_y(&to_z, -lv_theta(primary_p4));
  const Vec4 p_primary_z = lorentz_apply(to_z, primary_p4);
  const LorentzRot to_lab = lorentz_inverse(to_z);

  bool success = true;
  int attempts = 0;
  const int max_attempts = 1000;

  do {
    if (attempts++ > max_attempts) {
      ws->report.attempts_exhausted = true;
      ws->report.primary_returned_unchanged = true;
      ws->report.attempts = attempts;
      // Geant4 rebuilds the nucleus in its ground state and erases every hit mark, then returns
      // the primary as one kinetic track at z = 2*OuterRadius. The nucleus rebuild is
      // reproduced so that a caller reading the wounded nucleus afterwards sees no holes.
      ftf_model_init(&ws->model, proj, p_primary_z, target_a, target_z, lund, rng);
      for (int i = 0; i < ws->model.target.my_a; ++i) {
        ws->model.target.nucleons[i].hit = false;
        ws->model.target.nucleons[i].hit_by = kNullSplitable;
      }
      if (ws->model.has_projectile_nucleus) {
        for (int i = 0; i < ws->model.projectile.my_a; ++i) {
          ws->model.projectile.nucleons[i].hit = false;
          ws->model.projectile.nucleons[i].hit_by = kNullSplitable;
        }
      }
      ws->strings.n_out = 1;
      ws->strings.out[0] = FragHadron();
      ws->strings.out[0].pdg = proj.pdg;
      ws->strings.out[0].momentum = primary_p4;
      ws->strings.out[0].position = Vec3d{0.0, 0.0, 2.0 * ws->model.target.outer_radius()};
      ws->strings.out[0].formation_time = 0.0;
      return true;
    }

    success = true;

    ftf_model_init(&ws->model, proj, p_primary_z, target_a, target_z, lund, rng);
    if (ws->model.report.refused != FtfRefusal::kNone || ws->model.report.nucleus_failed) {
      ws->report.model = ws->model.report;
      ws->report.refused = ws->model.report.refused;
      ws->report.attempts = attempts;
      return false;
    }

    ftf_get_strings(&ws->model, rng);
    if (ws->model.report.refused != FtfRefusal::kNone) {
      ws->report.model = ws->model.report;
      ws->report.refused = ws->model.report.refused;
      ws->report.attempts = attempts;
      return false;
    }
    if (ws->model.n_strings == 0) {
      ++ws->report.retries_no_strings;
      success = false;
      continue;
    }

    // Rotate the strings back to the lab and accumulate their four-momentum. Geant4 rotates
    // EVERY string, excited or not, and sums the excited ones from their partons and the
    // non-excited ones from their kinetic tracks - the same four-vector by a different route.
    Vec4 sum_string_mom(0.0, 0.0, 0.0, 0.0);
    for (int i = 0; i < ws->model.n_strings; ++i) {
      ExcitedString& s = ws->model.strings[i];
      if (s.excited) {
        s.pleft = lorentz_apply(to_lab, s.pleft);
        s.pright = lorentz_apply(to_lab, s.pright);
        sum_string_mom = sum_string_mom + (s.pleft + s.pright);
      } else {
        s.track_mom = lorentz_apply(to_lab, s.track_mom);
        sum_string_mom = sum_string_mom + s.track_mom;
      }
    }

    // The wounded nucleons, rotated to the lab and counted by species.
    int n_proton_projectile_hits = 0, n_neutron_projectile_hits = 0, n_lambda_projectile_hits = 0;
    if (ws->model.has_projectile_nucleus) {
      for (int i = 0; i < ws->model.projectile.my_a; ++i) {
        bic::Nucleon* n = &ws->model.projectile.nucleons[i];
        if (n->hit) {
          n->momentum = lorentz_apply(to_lab, n->momentum);
          if (n->type == bic::kProton) { ++n_proton_projectile_hits; }
          if (n->type == bic::kNeutron) { ++n_neutron_projectile_hits; }
          if (n->type == bic::kLambda) { ++n_lambda_projectile_hits; }
        }
      }
    }
    int n_proton_target_hits = 0, n_neutron_target_hits = 0;
    for (int i = 0; i < ws->model.target.my_a; ++i) {
      bic::Nucleon* n = &ws->model.target.nucleons[i];
      if (n->hit) {
        n->momentum = lorentz_apply(to_lab, n->momentum);
        if (n->type == bic::kProton) { ++n_proton_target_hits; }
        if (n->type == bic::kNeutron) { ++n_neutron_target_hits; }
      }
    }

    const int abs_proj_charge =
        (ws->model.projectile_charge < 0) ? -ws->model.projectile_charge
                                          : ws->model.projectile_charge;
    const int abs_proj_baryon = (ws->model.projectile_baryon < 0) ? -ws->model.projectile_baryon
                                                                  : ws->model.projectile_baryon;
    int n_proton_p = 0, n_neutron_p = 0, n_lambda_p = 0;
    if (ws->model.has_projectile_nucleus) {
      const int lam = proj.n_lambdas;
      n_proton_p = abs_proj_charge - n_proton_projectile_hits;
      if (n_proton_p < 0) { n_proton_p = 0; }
      n_lambda_p = lam - n_lambda_projectile_hits;
      if (n_lambda_p < 0) { n_lambda_p = 0; }
      n_neutron_p = abs_proj_baryon - abs_proj_charge - lam - n_neutron_projectile_hits;
      if (n_neutron_p < 0) { n_neutron_p = 0; }
    }
    const int n_proton_t = target_z - n_proton_target_hits;
    const int n_neutron_t = target_a - target_z - n_neutron_target_hits;

    bool residual_by_target = false, residual_by_projectile = false;
    if (ftf_unphysical_residual(n_proton_t, n_neutron_t, n_proton_p, n_neutron_p, n_lambda_p,
                                ws->model.has_projectile_nucleus, &residual_by_target,
                                &residual_by_projectile)) {
      ++ws->report.retries_residual;
      if (residual_by_target) { ++ws->report.retries_residual_target; }
      if (residual_by_projectile) { ++ws->report.retries_residual_projectile; }
      success = false;
      continue;
    }

    const double inv_mass = sum_string_mom.mag();

    ftf_fragment_strings(lund, &ws->strings, ws->model.strings, ws->model.n_strings, rng);
    if (!ws->strings.success) {
      // FragmentStrings returned a null vector: Geant4's `if (theResult == 0) { Success=false;
      // continue; }`. NOT a refusal even though `ftf_fragment_strings` leaves
      // `kEnergyCorrectorFailed` behind - that value is how P11's layer says "null vector", and
      // a null vector is what Scatter's retry loop exists for. Counted so that an event which
      // needed several of them is visible.
      ++ws->report.fragment_retries;
      success = false;
      continue;
    }
    // A refusal that survives a SUCCESSFUL fragmentation means the hadron list is incomplete -
    // a capacity, or a PDG code the table does not carry - and retrying cannot fix it. The two
    // loop-limit values are reports and are carried without aborting, because Geant4 keeps the
    // result in both cases.
    if (ws->strings.refused == FtfRefusal::kStringHadronCapacity ||
        ws->strings.refused == FtfRefusal::kUnknownHadronCode ||
        ws->strings.refused == FtfRefusal::kResonanceMinimumMassMissing ||
        ws->strings.refused == FtfRefusal::kIllegalPartonPair ||
        ws->strings.refused == FtfRefusal::kHadronBuilderIllegalContent) {
      ws->report.refused = ws->strings.refused;
      ws->report.attempts = attempts;
      return false;
    }
    if (ws->strings.refused != FtfRefusal::kNone) {
      ws->report.fragment_report = ws->strings.refused;
    }

    double sum_mass = 0.0;
    for (int i = 0; i < ws->strings.n_out; ++i) {
      sum_mass += ws->strings.out[i].momentum.mag();
    }
    if ((sum_mass > inv_mass) || (sum_mass == 0.0)) {
      ++ws->report.retries_summass;
      success = false;
    }

  } while (!success);

  ws->report.attempts = attempts;
  ws->report.model = ws->model.report;
  return true;
}

/// The wounded nucleus, as P6's `Propagate` wants it.
///
/// `radius` is `GetNuclearRadius()`, the HALF-density radius - not `GetOuterRadius`, which is
/// what the impact-parameter sampling used. The interface compares a track's formation position
/// against it to decide whether the track escaped.
template <int kA, int kP, int kI, int kS, int kT, int kPS>
__host__ __device__ inline preco::WoundedNucleus ftf_wounded_nucleus(
    FtfWorkspace<kA, kP, kI, kS, kT, kPS>* ws, int target_a, int target_z) {
  ws->n_hit = 0;
  for (int i = 0; i < ws->model.target.my_a; ++i) {
    const bic::Nucleon& n = ws->model.target.nucleons[i];
    if (!n.hit) { continue; }
    if (ws->n_hit >= kA) {
      ws->report.refused = FtfRefusal::kWoundedNucleonCapacity;
      break;
    }
    preco::HitNucleon h;
    h.charge = n.charge();
    h.pdg_mass = n.pdg_mass();
    h.binding_energy = n.binding_energy;
    h.momentum = n.momentum;
    h.is_lambda = (n.type == bic::kLambda);
    ws->hit[ws->n_hit++] = h;
  }
  preco::WoundedNucleus nuc;
  nuc.a = target_a;
  nuc.z = target_z;
  nuc.lambdas = 0;
  nuc.radius = ws->model.target.nuclear_radius();
  nuc.hit = ws->hit;
  nuc.n_hit = ws->n_hit;
  return nuc;
}

/// The hand-over into P6, behind a `__noinline__` so that its 10 kB of stack
/// (docs/PORTED.md 2.1.10: `preco::deexcite` inlines whole at about 10,080 bytes) is not added
/// to every frame that merely calls FTF.
///
/// It stops at `propagate_residual`, i.e. at the point where Geant4 hands the excited residual
/// to `DeExcite`. The de-excitation itself is P3's and is invoked by the caller, so that the
/// three stages - strings, cascade hand-over, de-excitation - can be dumped separately, which
/// is how P6 tagged its own oracle.
template <int kA, int kP, int kI, int kS, int kT, int kPS, typename Rng>
__host__ __device__ __noinline__ preco::CascadeResidual ftf_propagate(
    FtfWorkspace<kA, kP, kI, kS, kT, kPS>* ws, const preco::WoundedNucleus& nuc,
    const Vec4& primary_p4, Rng& rng) {
  return preco::propagate_residual(ws->tracks, ws->n_tracks, nuc, primary_p4, ws->escaped, kT,
                                   ws->n_escaped, ws->report.generator, rng);
}

/// G4TheoFSGenerator::ApplyYourself, as the package's entry point.
///
/// @param proj       the projectile definition (xs/projectile.cuh)
/// @param kin_energy the projectile's kinetic energy, MeV
/// @param target     (Z, A) of the target nucleus
/// @param out        the final state; `theParticleChange`
/// @param ws         the workspace, one per track in flight, never a local
/// @param lund       P11's fragmentation tables
///
/// `theParticleChange->SetStatusChange(stopAndKill)` is the FIRST statement: FTF always kills
/// the primary, and the two dummy branches below it set `isAlive` back. Those branches - a
/// charm or bottom hadron, or a hypernucleus, below 100 MeV kinetic energy - return the primary
/// untouched, and this port reproduces them rather than refusing, because returning the primary
/// IS what Geant4 does and refusing would delete an interaction that Geant4 performs.
template <typename real_t, int kA, int kP, int kI, int kS, int kT, int kPS, int kMaxSec,
          typename Rng>
__host__ __device__ inline void apply_yourself(const HadProjectile<real_t>& proj_in,
                                               const HadNucleus& target,
                                               HadFinalState<real_t, kMaxSec>& out,
                                               FtfWorkspace<kA, kP, kI, kS, kT, kPS>* ws,
                                               const LundTables<double>* lund, Rng& rng) {
  out.clear();
  out.status = HadFinalStateStatus::kStopAndKill;
  ws->report = FtfApplyReport();
  ws->n_tracks = 0;
  ws->n_escaped = 0;

  // AN ION IS NOT IN THE HADRON TABLE and cannot be: `data/ftf_hadrons.hh` drops every 10LZZZAAAI
  // code because which ions exist is a property of the run rather than of Geant4 (the generator
  // script's header says why, and docs/RISK.md V89 is what happened when a count of them was
  // asserted). An ion is identified by its baryon number instead, and it carries no charm or
  // bottom quark, so the first dummy branch below cannot apply to one.
  const bool projectile_is_ion =
      (proj_in.baryon_number > 1) || (proj_in.baryon_number < -1);
  const data::FtfHadron* pdef =
      projectile_is_ion ? nullptr : data::ftf_find_hadron(proj_in.pdg);
  if (pdef == nullptr && !projectile_is_ion) {
    ws->report.refused = FtfRefusal::kUnknownHadronCode;
    return;
  }

  // The two dummy low-energy branches, in Geant4's order.
  const double energy_threshold_heavy = 100.0 * units::MeV<double>();
  if (!projectile_is_ion && static_cast<double>(proj_in.kin_energy) < energy_threshold_heavy &&
      (pdef->nq4 != 0 || pdef->naq4 != 0 || pdef->nq5 != 0 || pdef->naq5 != 0)) {
    out.status = HadFinalStateStatus::kIsAlive;
    out.energy_change = proj_in.kin_energy;
    out.momentum_change = Vec3<real_t>{real_t(0), real_t(0), real_t(1)};
    ws->report.low_energy_dummy = true;
    return;
  }
  // The second dummy branch is `IsHypernucleus()` on the PROJECTILE below 100 MeV. P5's
  // `HadProjectile` carries no lambda count - there is no field for it - so a hypernucleus
  // projectile is not representable at this interface at all, and `proj.n_lambdas` is set to 0
  // below. A hyper-TARGET is representable (`HadNucleus::l`) and is refused by name, as P3 and
  // P9 both refuse it; the projectile side is the one to remember when P1 grows the species.
  if (target.l != 0) {
    ws->report.refused = FtfRefusal::kHyperNucleus;
    return;
  }

  xs::Projectile<double> proj;
  proj.pdg = proj_in.pdg;
  proj.mass = static_cast<double>(proj_in.mass);
  proj.charge = static_cast<double>(proj_in.charge);
  proj.baryon_number = proj_in.baryon_number;
  proj.n_lambdas = 0;

  // `G4DynamicParticle aPart(def, thePrimary.Get4Momentum().vect())` - the momentum is the
  // projectile's and the energy is rebuilt from the DEFINITION's mass, which is what
  // G4HadProjectile already stores.
  const double plab = static_cast<double>(proj_in.momentum());
  const Vec4 primary_p4(0.0, 0.0, plab, static_cast<double>(proj_in.total_energy()));

  if (!ftf_scatter(ws, proj, primary_p4, target.a, target.z, lund, rng)) {
    ws->report.model = ws->model.report;
    return;
  }

  // The string model's secondaries become the cascade tracks P6's interface propagates.
  ws->n_tracks = 0;
  for (int i = 0; i < ws->strings.n_out; ++i) {
    if (ws->n_tracks >= kT) {
      ws->report.track_capacity = true;
      break;
    }
    ws->tracks[ws->n_tracks++] =
        ftf_track_from_hadron(ws->strings.out[i], /*model_id=*/0, &ws->report.refused);
  }
  if (ws->report.refused != FtfRefusal::kNone) { return; }

  // `hitCount != GetMassNumber()` selects the residual path; the equality selects
  // G4DecayStrongResonances, which is not written.
  int hit_count = 0;
  for (int i = 0; i < ws->model.target.my_a; ++i) {
    if (ws->model.target.nucleons[i].hit) { ++hit_count; }
  }
  if (ws->model.has_projectile_nucleus) {
    // PropagateNuclNucl - a second residual for the projectile, plus MakeCoalescence. Not
    // written: preco::propagate_residual is the hadron-nucleus arm only.
    ws->report.refused = FtfRefusal::kPropagateNuclNucl;
    return;
  }
  if (hit_count == ws->model.target.my_a) {
    ws->report.refused = FtfRefusal::kDecayStrongResonances;
    return;
  }

  const preco::WoundedNucleus nuc = ftf_wounded_nucleus(ws, target.a, target.z);
  if (ws->report.refused != FtfRefusal::kNone) { return; }

  const preco::CascadeResidual residual = ftf_propagate(ws, nuc, primary_p4, rng);
  if (ws->report.generator.any()) { return; }

  // G4HadFinalState assembly. `time = max(GetFormationTime(), 0)` and the primary's global time
  // is added by the process, not here.
  for (int i = 0; i < ws->n_escaped; ++i) {
    const preco::CascadeTrack& t = ws->escaped[i];
    const data::FtfHadron* d = data::ftf_find_hadron(t.pdg);
    HadSecondary<real_t> s;
    s.pdg = t.pdg;
    s.z = 0;
    s.a = 0;
    s.mass = static_cast<real_t>((d != nullptr) ? d->mass : t.momentum.mag());
    s.kin_energy = static_cast<real_t>(t.momentum.e) - s.mass;
    const double p = std::sqrt(g4gpu::mag2(t.momentum.v));
    s.direction = (p > 0.0)
                      ? Vec3<real_t>{static_cast<real_t>(t.momentum.v.x / p),
                                     static_cast<real_t>(t.momentum.v.y / p),
                                     static_cast<real_t>(t.momentum.v.z / p)}
                      : Vec3<real_t>{real_t(0), real_t(0), real_t(1)};
    s.time = static_cast<real_t>((t.formation_time > 0.0) ? t.formation_time : 0.0);
    s.creator_model_id = t.creator_model_id;
    if (!out.add_secondary(s)) { ws->report.secondary_overflow = true; }
  }

  // The excited residual is handed on as a (Z, A, E*) secondary; P3's de-excitation is the
  // caller's next call, and `CascadeResidual` carries the fragment it needs.
  if (residual.exists) {
    HadSecondary<real_t> s;
    s.z = residual.fragment.z;
    s.a = residual.fragment.a;
    s.pdg = pdg_nuclear_code(s.z, s.a);
    s.mass = static_cast<real_t>(residual.fragment.ground_state_mass +
                                 residual.fragment.excitation);
    s.kin_energy = static_cast<real_t>(residual.fragment.momentum.e) - s.mass;
    const double p = std::sqrt(g4gpu::mag2(residual.fragment.momentum.v));
    s.direction = (p > 0.0) ? Vec3<real_t>{
                                  static_cast<real_t>(residual.fragment.momentum.v.x / p),
                                  static_cast<real_t>(residual.fragment.momentum.v.y / p),
                                  static_cast<real_t>(residual.fragment.momentum.v.z / p)}
                            : Vec3<real_t>{real_t(0), real_t(0), real_t(1)};
    if (!out.add_secondary(s)) { ws->report.secondary_overflow = true; }
  }
}

}  // namespace g4gpu::hadronic::ftf
