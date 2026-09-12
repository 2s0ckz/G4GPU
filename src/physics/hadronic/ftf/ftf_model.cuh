// G4FTFModel: a projectile and a nucleus in, a vector of excited strings and a wounded nucleus
// out.
//
// Transcribed from
//   source/processes/hadronic/models/parton_string/diffraction/src/G4FTFModel.cc
//     Init, GetStrings, StoreInvolvedNucleon, ReggeonCascade, PutOnMassShell,
//     ComputeNucleusProperties, GenerateDeltaIsobar, SamplingNucleonKinematics,
//     CheckKinematics, FinalizeKinematics, ExciteParticipants, BuildStrings, GetResiduals,
//     GaussianPt
//   .../management/src/G4VParticipants.cc   Init, InitProjectileNucleus
//
// ## The three stages, and which one each number comes from
//
// 1. `GetList` (participants.cuh) samples an impact parameter and marks the nucleons the
//    projectile passes close to. Those are the GLAUBER participants.
// 2. `ReggeonCascade` marks MORE nucleons - neighbours of the participants, with probability
//    `Cnd * exp(-b^2/R2nd)` - and gives each of them `status = 3`. They never collide with
//    anything; they exist so that the residual nucleus is left with a hole and an excitation.
// 3. `PutOnMassShell` takes every marked nucleon, gives it a transverse momentum and a
//    light-cone fraction, and rebuilds the whole system's kinematics in the c.m.s. so that the
//    marked nucleons plus a residual nucleus carry exactly the collision's four-momentum. This
//    is what puts the nucleons back on their PDG mass shells (docs/RISK.md V50).
//
// Only then does `ExciteParticipants` run the collisions, `BuildStrings` turn each participant
// into a string, and `GetResiduals` distribute the residual nucleus's four-momentum and
// excitation over the wounded nucleons - which is the form P6's `Propagate` reads.
//
// ## What HighEnergyInter decides, and why the anti-nucleon arm needs it
//
// `HighEnergyInter` is `PlabPerParticle >= 1 GeV/c`. Above it, stages 2 and 3 run and the
// collisions are pure FTF. Below it, `ReggeonCascade` and `PutOnMassShell` are SKIPPED and
// every collision instead calls `AdjustNucleons`, which rebuilds the two-body kinematics
// against the residual nucleus one collision at a time. QBBC reaches the low arm only through
// an anti-baryon projectile, which it gives FTFP at every energy
// (`G4HadronicBuilder::BuildFTFP_BERT(..., bert=false)`); every other projectile enters FTFP at
// 3 GeV kinetic energy and is far above 1 GeV/c. `AdjustNucleons` and its three algorithm
// methods are REFUSED BY NAME (`kAdjustNucleons`), so an anti-nucleon below 1 GeV/c is reported
// rather than approximated.
//
// ## Two masses of the same residual nucleus
//
// `Init` sets `TargetResidual4Momentum.setE( GetIonMass(Z, A) )` - the AME table mass - while
// `ComputeNucleusProperties` rebuilds the residual mass from `GetIonMass` again but with the
// residual (Z, A), and `G4Fancy3DNucleus::GetMass()` (the sum of nucleon masses minus the
// binding energy) is a THIRD answer that this file never uses. P9's contract header
// (bic/nucleus/nucleus_model.cuh) lists all three; `deex::nuclear_mass(A, Z)` is the one
// `G4IonTable::GetIonMass` resolves to, and it is what is used here.
//
// ## The residual excitation is SAMPLED, not accumulated
//
// `residualExcitationEnergy += -ExcitationEnergyPerWoundedNucleon * G4Log( G4UniformRand() )`
// - one exponential deviate per wounded nucleon, mean `ExcitationEnergyPerWoundedNucleon`. The
// commented-out line above it (`+= ExcitationEnergyPerWoundedNucleon`, "In G4 10.1") is the
// deterministic version this replaced. So `ComputeNucleusProperties` consumes one deviate per
// HIT nucleon and none per spectator, which is a draw count a test can check.
#pragma once
#include <cmath>

#include "core/units.cuh"
#include "data/ftf_hadrons.hh"
#include "physics/hadronic/bic/nucleus/nucleus_model.cuh"
#include "physics/hadronic/deexcitation/nuclear_masses.cuh"
#include "physics/hadronic/ftf/diffractive_excitation.cuh"
#include "physics/hadronic/ftf/elastic_hn.cuh"
#include "physics/hadronic/ftf/ftf_parameters.cuh"
#include "physics/hadronic/ftf/lorentz.cuh"
#include "physics/hadronic/ftf/participants.cuh"
#include "physics/hadronic/ftf/refusal.cuh"
#include "physics/hadronic/ftf/splitable_hadron.cuh"
#include "physics/hadronic/ftf/string_fragmentation.cuh"

namespace g4gpu::hadronic::ftf {

/// G4FTFModel::LowEnergyLimit - 1000 MeV/c, and the commented-out 2000 above it.
__host__ __device__ inline constexpr double ftf_low_energy_limit() {
  return 1000.0 * units::MeV<double>();
}

/// What `ftf_get_strings` reports back about a failed or partial event. Every field is a
/// Geant4 behaviour that a null pointer or an empty vector would have hidden.
struct FtfModelReport {
  FtfRefusal refused = FtfRefusal::kNone;
  bool put_on_mass_shell_failed = false;   ///< PutOnMassShell returned false
  bool excite_failed = false;              ///< ExciteParticipants returned false
  bool participants_empty = false;         ///< GetList produced nothing after 1000 tries
  bool nucleus_failed = false;             ///< bic::nucleus_init could not build the nucleus
  bool string_capacity = false;
  bool involved_capacity = false;
  int refused_z = 0, refused_a = 0;

  __host__ __device__ bool any() const {
    return refused != FtfRefusal::kNone || put_on_mass_shell_failed || excite_failed ||
           participants_empty || nucleus_failed || string_capacity || involved_capacity;
  }
};

/// Everything one FTF interaction's state lives in, behind one pointer.
///
/// `kMaxTargetA` and `kMaxProjA` size the two nuclei independently because the target can be
/// U238 and the projectile is an ion only when the beam is one. Both default to P9's
/// `bic::kMaxNucleons`, so nothing a QBBC run can ask for is refused by capacity; a caller that
/// wants a smaller frame instantiates smaller and gets `FtfModelReport::involved_capacity` for
/// anything that does not fit.
template <int kMaxTargetA = bic::kMaxNucleons, int kMaxProjA = bic::kMaxNucleons,
          int kMaxInteractions = 1024, int kMaxStrings = 2 * bic::kMaxNucleons + 2>
struct FtfModelWorkspace {
  static constexpr int kScratchA = (kMaxTargetA > kMaxProjA) ? kMaxTargetA : kMaxProjA;

  // ---- the two nuclei, P9's model ----
  bic::Nucleon target_nucleons[kMaxTargetA];
  bic::Nucleon proj_nucleons[kMaxProjA];
  // ONE scratch, shared between the two nucleus_init calls - which is what
  // bic/nucleus/nucleus_model.cuh's contract asks for: it carries the Gaussian latch that
  // CLHEP keeps in a thread-local static, and a second scratch would restart the pair cache and
  // give a different stream from Geant4's.
  Vec3d scratch_momentum[kScratchA];
  double scratch_fermi_p[kScratchA];
  bic::NucleusSortEntry scratch_sums[kScratchA];
  double scratch_flat[bic::kFlatBlock];
  bic::Nucleus3D target;
  bic::Nucleus3D projectile;
  bic::Nucleus3DScratch scratch;
  bool has_projectile_nucleus = false;

  // ---- participants and the splitable-hadron pool ----
  FtfParticipants<kScratchA, kMaxInteractions> participants;

  // ---- G4FTFModel's own members ----
  FtfParameters<double> params;
  int projectile_pdg = 0;
  Vec4 projectile_p4;            ///< theProjectile, MUTATED by PutOnMassShell
  int projectile_baryon = 0;
  int projectile_charge = 0;
  bool high_energy_inter = true;

  int involved_target[kMaxTargetA];      ///< TheInvolvedNucleonsOfTarget, as nucleon indices
  int n_involved_target = 0;
  int involved_projectile[kMaxProjA];    ///< TheInvolvedNucleonsOfProjectile
  int n_involved_projectile = 0;

  Vec4 projectile_residual_p4;
  int projectile_residual_a = 0;
  int projectile_residual_z = 0;
  int projectile_residual_lambda = 0;
  double projectile_residual_exc = 0.0;

  Vec4 target_residual_p4;
  int target_residual_a = 0;
  int target_residual_z = 0;
  double target_residual_exc = 0.0;

  int n_projectile_spectators = 0;
  int n_target_spectators = 0;
  int n_nn_collisions = 0;

  // ---- the strings this model produces ----
  ExcitedString strings[kMaxStrings];
  int n_strings = 0;

  ExciteCommon excite;
  FtfModelReport report;

  __host__ __device__ SplitableHadron* splitable(int slot) {
    return (slot == kNullSplitable) ? nullptr : &participants.pool[slot];
  }
};

/// G4FTFModel::GaussianPt - and it is NOT G4DiffractiveExcitation's or
/// G4ElasticHNScattering's. The difference is the `ymax < 200` split: when `maxPtSquare` is
/// more than 200 average-pt2's away, `exp(-ymax)` underflows the other two functions' `1 +
/// u*(exp(-ymax)-1)` to `1 - u` and this one writes `-<pt2> log(1-u)` directly. The two agree
/// analytically and not bitwise, and PutOnMassShell calls THIS one.
template <typename Rng>
__host__ __device__ inline Vec3d ftf_gaussian_pt_model(double average_pt2, double max_pt_square,
                                                       Rng& rng) {
  double pt2 = 0.0, pt = 0.0;
  if (average_pt2 > 0.0) {
    const double ymax = max_pt_square / average_pt2;
    if (ymax < 200.0) {
      pt2 = -average_pt2 * std::log(1.0 + rng.uniform() * (std::exp(-ymax) - 1.0));
    } else {
      pt2 = -average_pt2 * std::log(1.0 - rng.uniform());
    }
    pt = std::sqrt(pt2);
  }
  const double phi = rng.uniform() * units::twopi<double>();
  return Vec3d{pt * std::cos(phi), pt * std::sin(phi), 0.0};
}

/// `G4IonTable::GetIonMass(Z, A)` for a residual with a given number of lambdas, as
/// ComputeNucleusProperties and AdjustNucleons both spell it out.
///
/// A == 1 resolves to a proton, a lambda or a neutron by charge; A == 2 with lambdas is the SUM
/// of two hadron masses and not a nuclear mass at all; A >= 3 with lambdas is
/// `G4HyperNucleiProperties::GetNuclearMass`, which this port refuses by name
/// (`kHyperNucleus`); everything else is `deex::nuclear_mass(A, |Z|)`.
__host__ __device__ inline double ftf_residual_mass(int a, int z, int n_lambdas,
                                                    FtfRefusal* refused) {
  const int abs_z = (z < 0) ? -z : z;
  if (a == 1) {
    if (abs_z == 1) { return data::ftf_find_hadron(2212)->mass; }
    if (n_lambdas == 1) { return data::ftf_find_hadron(3122)->mass; }
    return data::ftf_find_hadron(2112)->mass;
  }
  if (n_lambdas > 0) {
    if (a == 2) {
      double m = data::ftf_find_hadron(3122)->mass;
      if (abs_z == 1) {
        m += data::ftf_find_hadron(2212)->mass;      // lambda + proton
      } else if (n_lambdas == 1) {
        m += data::ftf_find_hadron(2112)->mass;      // lambda + neutron
      } else {
        m += data::ftf_find_hadron(3122)->mass;      // lambda + lambda
      }
      return m;
    }
    if (refused != nullptr) { *refused = FtfRefusal::kHyperNucleus; }
    return 0.0;
  }
  return deex::nuclear_mass(a, abs_z);
}

/// G4FTFModel::StoreInvolvedNucleon.
template <int kA, int kP, int kI, int kS>
__host__ __device__ inline void ftf_store_involved_nucleon(FtfModelWorkspace<kA, kP, kI, kS>* w) {
  w->n_involved_target = 0;
  for (int i = 0; i < w->target.my_a; ++i) {
    if (w->target.nucleons[i].hit) {
      if (w->n_involved_target >= kA) {
        w->report.involved_capacity = true;
        return;
      }
      w->involved_target[w->n_involved_target++] = i;
    }
  }
  if (!w->has_projectile_nucleus) { return; }

  w->n_involved_projectile = 0;
  for (int i = 0; i < w->projectile.my_a; ++i) {
    if (w->projectile.nucleons[i].hit) {
      if (w->n_involved_projectile >= kP) {
        w->report.involved_capacity = true;
        return;
      }
      w->involved_projectile[w->n_involved_projectile++] = i;
    }
  }
}

/// G4FTFModel::ReggeonCascade - the reggeon-theory-inspired destruction of the two nuclei.
///
/// `InitNINt` is latched BEFORE the loop, so a nucleon dragged in by the cascade does not
/// itself seed more cascading: the commented-out `InvPN < NumberOfInvolvedNucleonsOfProjectile`
/// above the projectile loop is the version that would have. One deviate per untouched nucleon
/// per participant, so the cost is `n_participants * (A - n_participants)` deviates and the
/// count is a strong check on the participant list.
///
/// The new nucleon gets the SEEDING participant's creation time, not its own.
template <int kA, int kP, int kI, int kS, typename Rng>
__host__ __device__ inline void ftf_reggeon_cascade(FtfModelWorkspace<kA, kP, kI, kS>* w,
                                                    Rng& rng) {
  const int init_nint = w->n_involved_target;

  for (int inv_tn = 0; inv_tn < init_nint; ++inv_tn) {
    bic::Nucleon* a_target_nucleon = &w->target.nucleons[w->involved_target[inv_tn]];
    const int seed_slot = a_target_nucleon->hit_by;
    if (seed_slot == kNullSplitable) {
      w->participants.null_target = true;
      continue;
    }
    const double creation_time = w->participants.pool[seed_slot].time_of_creation;
    const double x_of_wounded = a_target_nucleon->position.x;
    const double y_of_wounded = a_target_nucleon->position.y;

    for (int i = 0; i < w->target.my_a; ++i) {
      bic::Nucleon* neighbour = &w->target.nucleons[i];
      if (!neighbour->hit) {
        const double dx = x_of_wounded - neighbour->position.x;
        const double dy = y_of_wounded - neighbour->position.y;
        const double impact2 = dx * dx + dy * dy;
        if (rng.uniform() < w->params.cof_nuclear_destruction *
                                std::exp(-impact2 / w->params.r2_of_nuclear_destruction)) {
          if (w->n_involved_target >= kA) {
            w->report.involved_capacity = true;
            return;
          }
          w->involved_target[w->n_involved_target++] = i;
          const int slot = FtfParticipants<
              FtfModelWorkspace<kA, kP, kI, kS>::kScratchA, kI>::kTargetBase + i;
          w->participants.pool[slot] =
              splitable_from_nucleon(neighbour->pdg(), neighbour->momentum, neighbour->position);
          neighbour->hit = true;
          neighbour->hit_by = slot;
          w->participants.pool[slot].time_of_creation = creation_time;
          w->participants.pool[slot].status = 3;  // the comment says "2->3"
        }
      }
    }
  }

  if (!w->has_projectile_nucleus) { return; }

  const int init_ninp = w->n_involved_projectile;
  for (int inv_pn = 0; inv_pn < init_ninp; ++inv_pn) {
    bic::Nucleon* a_proj_nucleon = &w->projectile.nucleons[w->involved_projectile[inv_pn]];
    const int seed_slot = a_proj_nucleon->hit_by;
    if (seed_slot == kNullSplitable) {
      w->participants.null_target = true;
      continue;
    }
    const double creation_time = w->participants.pool[seed_slot].time_of_creation;
    const double x_of_wounded = a_proj_nucleon->position.x;
    const double y_of_wounded = a_proj_nucleon->position.y;

    for (int i = 0; i < w->projectile.my_a; ++i) {
      bic::Nucleon* neighbour = &w->projectile.nucleons[i];
      if (!neighbour->hit) {
        const double dx = x_of_wounded - neighbour->position.x;
        const double dy = y_of_wounded - neighbour->position.y;
        const double impact2 = dx * dx + dy * dy;
        if (rng.uniform() < w->params.cof_nuclear_destruction_pr *
                                std::exp(-impact2 / w->params.r2_of_nuclear_destruction)) {
          if (w->n_involved_projectile >= kP) {
            w->report.involved_capacity = true;
            return;
          }
          w->involved_projectile[w->n_involved_projectile++] = i;
          const int slot = FtfParticipants<
              FtfModelWorkspace<kA, kP, kI, kS>::kScratchA, kI>::kProjectileBase + i;
          w->participants.pool[slot] =
              splitable_from_nucleon(neighbour->pdg(), neighbour->momentum, neighbour->position);
          neighbour->hit = true;
          neighbour->hit_by = slot;
          w->participants.pool[slot].time_of_creation = creation_time;
          w->participants.pool[slot].status = 3;
        }
      }
    }
  }
}

/// G4FTFModel::ComputeNucleusProperties.
///
/// `sumMasses` accumulates, per INVOLVED nucleon, the nucleon's TRANSVERSE mass built from its
/// PDG mass (not its off-shell current mass) plus 20 MeV of separation energy, and finally the
/// residual's transverse mass. `residualMomentum` is the vector sum of the SPECTATORS' momenta
/// with pz and E then zeroed - so only its transverse part survives, and that is what the
/// residual recoils with.
template <typename Rng>
__host__ __device__ inline bool ftf_compute_nucleus_properties(
    bic::Nucleus3D* nucleus, const SplitableHadron* pool, Vec4* nucleus_momentum,
    Vec4* residual_momentum, double* sum_masses, double* residual_excitation,
    double* residual_mass, int* residual_mass_number, int* residual_charge,
    double excitation_per_wounded, FtfRefusal* refused, Rng& rng) {
  if (nucleus == nullptr) { return false; }
  (void)pool;

  int residual_lambdas = 0;
  for (int i = 0; i < nucleus->my_a; ++i) {
    bic::Nucleon* a = &nucleus->nucleons[i];
    *nucleus_momentum = *nucleus_momentum + a->momentum;
    if (a->hit) {
      const double pdg_mass = a->pdg_mass();
      const double perp2 = a->momentum.v.x * a->momentum.v.x + a->momentum.v.y * a->momentum.v.y;
      *sum_masses += std::sqrt(pdg_mass * pdg_mass + perp2);
      *sum_masses += 20.0 * units::MeV<double>();  // Separation energy for a nucleon
      *residual_excitation += -excitation_per_wounded * std::log(rng.uniform());
      (*residual_mass_number)--;
      // The absolute value is needed only for an anti-nucleus.
      const int q = a->charge();
      *residual_charge -= (q < 0) ? -q : q;
    } else {
      *residual_momentum = *residual_momentum + a->momentum;
      if (a->type == bic::kLambda) { ++residual_lambdas; }
    }
  }
  residual_momentum->v.z = 0.0;
  residual_momentum->e = 0.0;

  if (*residual_mass_number == 0) {
    *residual_mass = 0.0;
    *residual_excitation = 0.0;
  } else {
    if (*residual_mass_number == 1) {
      *residual_mass = ftf_residual_mass(1, *residual_charge, residual_lambdas, refused);
      *residual_excitation = 0.0;
    } else {
      *residual_mass =
          ftf_residual_mass(*residual_mass_number, *residual_charge, residual_lambdas, refused);
    }
    *residual_mass += *residual_excitation;
  }
  const double perp2 = residual_momentum->v.x * residual_momentum->v.x +
                       residual_momentum->v.y * residual_momentum->v.y;
  *sum_masses += std::sqrt((*residual_mass) * (*residual_mass) + perp2);
  return true;
}

/// G4FTFModel::GenerateDeltaIsobar.
///
/// One deviate per involved nucleon, ALWAYS - `G4UniformRand() < probDeltaIsobar` is the first
/// operand of the `&&`, so the deviate is spent even when `numberOfDeltas` has already reached
/// the maximum. `numberOfDeltas` is incremented BEFORE the lambda skip and before the
/// energy test, so a rejected conversion still uses up one of the allowed deltas.
///
/// `maxNumberOfDeltas = (sqrtS - sumMasses)/400 MeV` uses 400 where the actual nucleon-to-delta
/// mass step is 292-294 MeV; Geant4's own comment says so.
///
/// The conversion is `pdg/10*10 + 4`: 2212 -> 2214 (Delta+), 2112 -> 2114 (Delta0), with the
/// sign restored for an antinucleon.
template <typename Rng>
__host__ __device__ inline bool ftf_generate_delta_isobar(double sqrt_s, const int* involved,
                                                          int n_involved, bic::Nucleus3D* nucleus,
                                                          SplitableHadron* pool,
                                                          double* sum_masses, Rng& rng) {
  if (sqrt_s < 0.0 || n_involved <= 0 || *sum_masses < 0.0) { return false; }

  const double prob_delta_isobar = 0.05;
  const int max_number_of_deltas =
      static_cast<int>((sqrt_s - *sum_masses) / (400.0 * units::MeV<double>()));
  int number_of_deltas = 0;

  for (int i = 0; i < n_involved; ++i) {
    if (rng.uniform() < prob_delta_isobar && number_of_deltas < max_number_of_deltas) {
      number_of_deltas++;
      bic::Nucleon* n = &nucleus->nucleons[involved[i]];
      if (n->type == bic::kLambda) { continue; }
      const int slot = n->hit_by;
      if (slot == kNullSplitable) { continue; }
      SplitableHadron* sh = &pool[slot];
      const data::FtfHadron* old_def = data::ftf_find_hadron(sh->pdg);
      if (old_def == nullptr) { continue; }
      const double perp2 =
          sh->momentum.v.x * sh->momentum.v.x + sh->momentum.v.y * sh->momentum.v.y;
      const double mass_nuc = std::sqrt(old_def->mass * old_def->mass + perp2);
      const int pdg_code = (sh->pdg < 0) ? -sh->pdg : sh->pdg;
      int new_pdg_code = pdg_code / 10;
      new_pdg_code = new_pdg_code * 10 + 4;  // Delta
      if (sh->pdg < 0) { new_pdg_code *= -1; }
      const data::FtfHadron* new_def = data::ftf_find_hadron(new_pdg_code);
      if (new_def == nullptr) { continue; }
      sh->pdg = new_pdg_code;
      const double mass_delta = std::sqrt(new_def->mass * new_def->mass + perp2);
      if (sqrt_s < *sum_masses + mass_delta - mass_nuc) {  // Change cannot be accepted
        sh->pdg = old_def->pdg;
        break;
      } else {
        *sum_masses += (mass_delta - mass_nuc);
      }
    }
  }
  return true;
}

/// G4FTFModel::SamplingNucleonKinematics.
///
/// The nucleon's four-momentum is used as a SCRATCH of four unrelated numbers here, and that is
/// why the code looks wrong: after the first block it holds `(px, py, 0, Mt)`, after the second
/// `(px, py, x, Mt)` with the light-cone fraction x IN THE z SLOT, and it stays that way until
/// FinalizeKinematics turns it back into a momentum. Geant4's own comment - "The energy is in
/// the lab (instead of cms) frame but it will not be used" - marks the one field that is not
/// what its name says.
///
/// `deltaPx/deltaPy` subtract the mean of (sampled pt sum - the residual's pt) from every
/// nucleon, which is what conserves transverse momentum by construction.
///
/// The `eps = 1e-10` tolerance is applied to a DIMENSIONLESS x, and `x = min(1, max(x, eps))`
/// uses eps as the floor rather than 0 - because `mass2 += E^2/x` divides by it.
template <typename Rng>
__host__ __device__ inline bool ftf_sampling_nucleon_kinematics(
    double average_pt2, double max_pt2, double d_cor, bic::Nucleus3D* nucleus,
    const SplitableHadron* pool, const Vec4& p_residual, double residual_mass,
    int residual_mass_number, const int* involved, int n_involved, double* mass2, Rng& rng) {
  if (nucleus == nullptr || n_involved < 1) { return false; }

  if (residual_mass_number == 0 && n_involved == 1) {
    d_cor = 0.0;
    average_pt2 = 0.0;
  }

  bool success = true;
  const double inv_n = 1.0 / static_cast<double>(n_involved);
  const double eps = 1.0e-10;
  const int max_number_of_loops = 1000;
  int loop_counter = 0;
  do {
    success = true;

    Vec3d pt_sum{0.0, 0.0, 0.0};
    if (average_pt2 > 0.0) {
      for (int i = 0; i < n_involved; ++i) {
        bic::Nucleon* a = &nucleus->nucleons[involved[i]];
        const Vec3d tmp_pt = ftf_gaussian_pt_model(average_pt2, max_pt2, rng);
        pt_sum = pt_sum + tmp_pt;
        a->momentum = Vec4(tmp_pt.x, tmp_pt.y, 0.0, 0.0);
      }
    }

    const double delta_px = (pt_sum.x - p_residual.v.x) * inv_n;
    const double delta_py = (pt_sum.y - p_residual.v.y) * inv_n;

    double sum_masses = residual_mass;
    for (int i = 0; i < n_involved; ++i) {
      bic::Nucleon* a = &nucleus->nucleons[involved[i]];
      const double px = a->momentum.v.x - delta_px;
      const double py = a->momentum.v.y - delta_py;
      // The mass is the SPLITABLE HADRON's, not the nucleon's: GenerateDeltaIsobar has already
      // turned some of them into Delta isobars and only the splitable hadron knows.
      const int slot = a->hit_by;
      double pdg_mass = a->pdg_mass();
      if (slot != kNullSplitable) {
        const data::FtfHadron* d = data::ftf_find_hadron(pool[slot].pdg);
        if (d != nullptr) { pdg_mass = d->mass; }
      }
      const double mt_n = std::sqrt(pdg_mass * pdg_mass + px * px + py * py);
      sum_masses += mt_n;
      a->momentum = Vec4(px, py, 0.0, mt_n);
    }

    double x_sum = 0.0;
    for (int i = 0; i < n_involved; ++i) {
      bic::Nucleon* a = &nucleus->nucleons[involved[i]];
      double x = 0.0;
      if (0.0 != d_cor) {
        // GaussianPt is reused as a one-dimensional Gaussian-ish sampler: `<pt2> = dCor^2`,
        // `maxPt2 = 1`, and only the x component of the two-dimensional answer is kept - so the
        // y component and the phi deviate are sampled and thrown away.
        const Vec3d tmp_x = ftf_gaussian_pt_model(d_cor * d_cor, 1.0, rng);
        x = tmp_x.x;
      }
      x += a->momentum.e / sum_masses;
      if (x < -eps || x > 1.0 + eps) {
        success = false;
        break;
      }
      x = (x < 0.0) ? 0.0 : ((x > 1.0) ? 1.0 : x);
      x_sum += x;
      a->momentum = Vec4(a->momentum.v.x, a->momentum.v.y, x, a->momentum.e);
    }

    if (x_sum < -eps || x_sum > 1.0 + eps) { success = false; }
    if (!success) { continue; }

    const double delta =
        (residual_mass_number == 0) ? (((x_sum - 1.0) < 0.0) ? (x_sum - 1.0) : 0.0) * inv_n : 0.0;

    x_sum = 1.0;
    *mass2 = 0.0;
    for (int i = 0; i < n_involved; ++i) {
      bic::Nucleon* a = &nucleus->nucleons[involved[i]];
      double x = a->momentum.v.z - delta;
      x_sum -= x;

      if (residual_mass_number == 0) {
        if (x <= -eps || x > 1.0 + eps) {
          success = false;
          break;
        }
      } else {
        if (x <= -eps || x > 1.0 + eps || x_sum <= -eps || x_sum > 1.0 + eps) {
          success = false;
          break;
        }
      }
      x = (x < eps) ? eps : ((x > 1.0) ? 1.0 : x);

      *mass2 += a->momentum.e * a->momentum.e / x;
      a->momentum = Vec4(a->momentum.v.x, a->momentum.v.y, x, a->momentum.e);
    }
    if (!success) { continue; }
    x_sum = (x_sum < eps) ? eps : ((x_sum > 1.0) ? 1.0 : x_sum);

    if (residual_mass_number > 0) {
      const double perp2 = p_residual.v.x * p_residual.v.x + p_residual.v.y * p_residual.v.y;
      *mass2 += (residual_mass * residual_mass + perp2) / x_sum;
    }
  } while ((!success) && ++loop_counter < max_number_of_loops);
  return (loop_counter < max_number_of_loops);
}

/// G4FTFModel::CheckKinematics. Draws no random number; it computes the two light-cone scales
/// and then rejects the configuration if any nucleon's rapidity is more than 2 units from its
/// nucleus's, or if it has overtaken the opposite side.
///
/// The return value is ALWAYS true; the verdict is `success`, which is an in-out parameter the
/// caller has already set. That asymmetry is Geant4's and is why the caller writes
/// `isOk = isOk && CheckKinematics(...)` around a variable that cannot become false.
__host__ __device__ inline bool ftf_check_kinematics(double s_value, double sqrt_s,
                                                     double projectile_mass2,
                                                     double target_mass2, double nucleus_y,
                                                     bool is_projectile_nucleus,
                                                     const int* involved, int n_involved,
                                                     bic::Nucleus3D* nucleus,
                                                     const SplitableHadron* pool,
                                                     double* target_wminus,
                                                     double* projectile_wplus, bool* success) {
  const double decay_momentum2 =
      s_value * s_value + projectile_mass2 * projectile_mass2 + target_mass2 * target_mass2 -
      2.0 * (s_value * (projectile_mass2 + target_mass2) + projectile_mass2 * target_mass2);
  *target_wminus =
      (s_value - projectile_mass2 + target_mass2 + std::sqrt(decay_momentum2)) / 2.0 / sqrt_s;
  *projectile_wplus = sqrt_s - target_mass2 / (*target_wminus);
  const double projectile_pz =
      *projectile_wplus / 2.0 - projectile_mass2 / 2.0 / (*projectile_wplus);
  const double projectile_e =
      *projectile_wplus / 2.0 + projectile_mass2 / 2.0 / (*projectile_wplus);
  const double projectile_y =
      0.5 * std::log((projectile_e + projectile_pz) / (projectile_e - projectile_pz));
  const double target_pz = -(*target_wminus) / 2.0 + target_mass2 / 2.0 / (*target_wminus);
  const double target_e = (*target_wminus) / 2.0 + target_mass2 / 2.0 / (*target_wminus);
  const double target_y = 0.5 * std::log((target_e + target_pz) / (target_e - target_pz));

  for (int i = 0; i < n_involved; ++i) {
    bic::Nucleon* a = &nucleus->nucleons[involved[i]];
    const Vec4 tmp = a->momentum;
    const int slot = a->hit_by;
    double pdg_mass = a->pdg_mass();
    if (slot != kNullSplitable) {
      const data::FtfHadron* d = data::ftf_find_hadron(pool[slot].pdg);
      if (d != nullptr) { pdg_mass = d->mass; }
    }
    const double mt2 = tmp.v.x * tmp.v.x + tmp.v.y * tmp.v.y + pdg_mass * pdg_mass;
    const double x = tmp.v.z;
    double pz = -(*target_wminus) * x / 2.0 + mt2 / (2.0 * (*target_wminus) * x);
    double e = (*target_wminus) * x / 2.0 + mt2 / (2.0 * (*target_wminus) * x);
    if (is_projectile_nucleus) {
      pz = (*projectile_wplus) * x / 2.0 - mt2 / (2.0 * (*projectile_wplus) * x);
      e = (*projectile_wplus) * x / 2.0 + mt2 / (2.0 * (*projectile_wplus) * x);
    }
    const double nucleon_y = 0.5 * std::log((e + pz) / (e - pz));

    if (std::fabs(nucleon_y - nucleus_y) > 2 ||
        (is_projectile_nucleus && target_y > nucleon_y) ||
        (!is_projectile_nucleus && projectile_y < nucleon_y)) {
      *success = false;
      break;
    }
  }
  return true;
}

/// G4FTFModel::FinalizeKinematics.
///
/// `residual3Momentum` starts at `(0, 0, 1)` - the z component is ONE, not zero, because the
/// nucleons' `x` fractions are subtracted from it and the residual gets what is left of the
/// unit light-cone. The x and y components start at zero and collect the recoil.
__host__ __device__ inline bool ftf_finalize_kinematics(double w, bool is_projectile_nucleus,
                                                        const LorentzRot& boost_cms_to_lab,
                                                        double residual_mass,
                                                        int residual_mass_number,
                                                        const int* involved, int n_involved,
                                                        bic::Nucleus3D* nucleus,
                                                        SplitableHadron* pool,
                                                        Vec4* residual_4momentum) {
  Vec3d residual3{0.0, 0.0, 1.0};

  for (int i = 0; i < n_involved; ++i) {
    bic::Nucleon* a = &nucleus->nucleons[involved[i]];
    Vec4 tmp = a->momentum;
    residual3 = residual3 - tmp.v;
    const int slot = a->hit_by;
    double pdg_mass = a->pdg_mass();
    if (slot != kNullSplitable) {
      const data::FtfHadron* d = data::ftf_find_hadron(pool[slot].pdg);
      if (d != nullptr) { pdg_mass = d->mass; }
    }
    const double mt2 = tmp.v.x * tmp.v.x + tmp.v.y * tmp.v.y + pdg_mass * pdg_mass;
    const double x = tmp.v.z;
    double pz = -w * x / 2.0 + mt2 / (2.0 * w * x);
    const double e = w * x / 2.0 + mt2 / (2.0 * w * x);
    if (is_projectile_nucleus) { pz *= -1.0; }
    tmp.v.z = pz;
    tmp.e = e;
    tmp = lorentz_apply(boost_cms_to_lab, tmp);
    a->momentum = tmp;
    if (slot != kNullSplitable) { pool[slot].momentum = tmp; }
  }

  const double residual_mt2 =
      residual_mass * residual_mass + residual3.x * residual3.x + residual3.y * residual3.y;

  double residual_pz = 0.0, residual_e = 0.0;
  if (residual_mass_number != 0) {
    residual_pz = -w * residual3.z / 2.0 + residual_mt2 / (2.0 * w * residual3.z);
    residual_e = w * residual3.z / 2.0 + residual_mt2 / (2.0 * w * residual3.z);
    if (is_projectile_nucleus) { residual_pz *= -1.0; }
  }

  residual_4momentum->v.x = residual3.x;
  residual_4momentum->v.y = residual3.y;
  residual_4momentum->v.z = residual_pz;
  residual_4momentum->e = residual_e;
  return true;
}

/// G4FTFModel::PutOnMassShell.
template <int kA, int kP, int kI, int kS, typename Rng>
__host__ __device__ inline bool ftf_put_on_mass_shell(FtfModelWorkspace<kA, kP, kI, kS>* w,
                                                      Rng& rng) {
  const bool is_projectile_nucleus = w->has_projectile_nucleus;

  Vec4 p_projectile = w->projectile_p4;
  if (p_projectile.v.z < 0.0) { return false; }

  bool is_ok = true;
  Vec4 p_target(0.0, 0.0, 0.0, 0.0);
  Vec4 p_target_residual(0.0, 0.0, 0.0, 0.0);
  double sum_masses = 0.0;
  double target_residual_mass = 0.0;

  is_ok = ftf_compute_nucleus_properties(
      &w->target, w->participants.pool, &p_target, &p_target_residual, &sum_masses,
      &w->target_residual_exc, &target_residual_mass, &w->target_residual_a,
      &w->target_residual_z, w->params.excitation_energy_per_wounded_nucleon,
      &w->report.refused, rng);
  if (!is_ok) { return false; }

  double m_projectile = 0.0, m2_projectile = 0.0;
  Vec4 p_proj(0.0, 0.0, 0.0, 0.0);
  Vec4 p_proj_residual(0.0, 0.0, 0.0, 0.0);
  double pr_residual_mass = 0.0;

  if (!is_projectile_nucleus) {  // hadron-nucleus collision
    m_projectile = p_projectile.mag();
    m2_projectile = p_projectile.e * p_projectile.e - g4gpu::mag2(p_projectile.v);
    sum_masses += m_projectile + 20.0 * units::MeV<double>();
  } else {
    is_ok = ftf_compute_nucleus_properties(
        &w->projectile, w->participants.pool, &p_proj, &p_proj_residual, &sum_masses,
        &w->projectile_residual_exc, &pr_residual_mass, &w->projectile_residual_a,
        &w->projectile_residual_z, w->params.excitation_energy_per_wounded_nucleon,
        &w->report.refused, rng);
    if (!is_ok) { return false; }
  }

  const Vec4 psum = p_projectile + p_target;
  const double sqrt_s = psum.mag();
  const double s = psum.e * psum.e - g4gpu::mag2(psum.v);

  if (sqrt_s < sum_masses) { return false; }  // impossible after putting nucleons on shell

  // Try to fit the residual excitation into the available energy; if it does not fit, drop the
  // excitation entirely (both nuclei's) and go back to the saved sum.
  const double saved_sum_masses = sum_masses;
  if (is_projectile_nucleus) {
    const double perp2 =
        p_proj_residual.v.x * p_proj_residual.v.x + p_proj_residual.v.y * p_proj_residual.v.y;
    sum_masses -= std::sqrt(pr_residual_mass * pr_residual_mass + perp2);
    sum_masses += std::sqrt((pr_residual_mass + w->projectile_residual_exc) *
                                (pr_residual_mass + w->projectile_residual_exc) + perp2);
  }
  {
    const double perp2 = p_target_residual.v.x * p_target_residual.v.x +
                         p_target_residual.v.y * p_target_residual.v.y;
    sum_masses -= std::sqrt(target_residual_mass * target_residual_mass + perp2);
    sum_masses += std::sqrt((target_residual_mass + w->target_residual_exc) *
                                (target_residual_mass + w->target_residual_exc) + perp2);
  }

  if (sqrt_s < sum_masses) {
    sum_masses = saved_sum_masses;
    if (is_projectile_nucleus) { w->projectile_residual_exc = 0.0; }
    w->target_residual_exc = 0.0;
  }

  target_residual_mass += w->target_residual_exc;
  if (is_projectile_nucleus) { pr_residual_mass += w->projectile_residual_exc; }

  // Sampling of nucleons that can turn into delta-isobars
  if (is_projectile_nucleus && w->projectile.my_a != 1) {
    is_ok = ftf_generate_delta_isobar(sqrt_s, w->involved_projectile, w->n_involved_projectile,
                                      &w->projectile, w->participants.pool, &sum_masses, rng);
  }
  if (w->target.my_a != 1) {
    is_ok = is_ok && ftf_generate_delta_isobar(sqrt_s, w->involved_target, w->n_involved_target,
                                               &w->target, w->participants.pool, &sum_masses,
                                               rng);
  }
  if (!is_ok) { return false; }

  const Vec3d bv = psum.boost_vector();
  const LorentzRot to_cms = lorentz_boost(Vec3d{-bv.x, -bv.y, -bv.z});
  Vec4 ptmp = lorentz_apply(to_cms, p_projectile);
  if (ptmp.v.z <= 0.0) { return false; }  // "String" moving backwards in c.m.s.

  const LorentzRot to_lab = lorentz_inverse(to_cms);

  double y_projectile_nucleus = 0.0;
  if (is_projectile_nucleus) {
    ptmp = lorentz_apply(to_cms, p_proj);
    y_projectile_nucleus = lv_rapidity(ptmp);
  }
  ptmp = lorentz_apply(to_cms, p_target);
  const double y_target_nucleus = lv_rapidity(ptmp);

  double dcor_p = 0.0;
  if (is_projectile_nucleus) {
    dcor_p = w->params.dof_nuclear_destruction / w->projectile.my_a;
  }
  double dcor_t = w->params.dof_nuclear_destruction / w->target.my_a;
  double average_pt2 = w->params.pt2_of_nuclear_destruction;
  const double max_pt_square = w->params.max_pt2_of_nuclear_destruction;

  double m2_proj = m2_projectile;  // only used for hadron-nucleus
  double wplus_projectile = 0.0;
  double m2_target = 0.0;
  double wminus_target = 0.0;
  int number_of_tries = 0;
  double scale_factor = 2.0;
  bool outer_success = true;

  const int max_number_of_loops = 1000;
  int loop_counter = 0;
  do {
    outer_success = true;
    const int max_number_of_inner_loops = 10000;
    do {
      number_of_tries++;
      if (number_of_tries == 100 * (number_of_tries / 100)) {
        // Every 100 tries the sampled momenta are halved again, so that momentum conservation
        // becomes easier to satisfy. ScaleFactor is not reset between outer iterations, so the
        // shrinking is cumulative over the whole call.
        scale_factor /= 2.0;
        dcor_p *= scale_factor;
        dcor_t *= scale_factor;
        average_pt2 *= scale_factor;
      }
      if (is_projectile_nucleus) {
        is_ok = ftf_sampling_nucleon_kinematics(
            average_pt2, max_pt_square, dcor_p, &w->projectile, w->participants.pool,
            p_proj_residual, pr_residual_mass, w->projectile_residual_a,
            w->involved_projectile, w->n_involved_projectile, &m2_proj, rng);
      }
      is_ok = is_ok && ftf_sampling_nucleon_kinematics(
                           average_pt2, max_pt_square, dcor_t, &w->target, w->participants.pool,
                           p_target_residual, target_residual_mass, w->target_residual_a,
                           w->involved_target, w->n_involved_target, &m2_target, rng);
      if (!is_ok) { return false; }
    } while ((sqrt_s < std::sqrt(m2_proj) + std::sqrt(m2_target)) &&
             number_of_tries < max_number_of_inner_loops);
    if (number_of_tries >= max_number_of_inner_loops) { return false; }

    if (is_projectile_nucleus) {
      is_ok = ftf_check_kinematics(s, sqrt_s, m2_proj, m2_target, y_projectile_nucleus, true,
                                   w->involved_projectile, w->n_involved_projectile,
                                   &w->projectile, w->participants.pool, &wminus_target,
                                   &wplus_projectile, &outer_success);
    }
    is_ok = is_ok && ftf_check_kinematics(s, sqrt_s, m2_proj, m2_target, y_target_nucleus, false,
                                          w->involved_target, w->n_involved_target, &w->target,
                                          w->participants.pool, &wminus_target,
                                          &wplus_projectile, &outer_success);
    if (!is_ok) { return false; }
  } while ((!outer_success) && ++loop_counter < max_number_of_loops);
  if (loop_counter >= max_number_of_loops) { return false; }

  if (!is_projectile_nucleus) {  // hadron-nucleus collision
    const double pz_projectile = wplus_projectile / 2.0 - m2_projectile / 2.0 / wplus_projectile;
    const double e_projectile = wplus_projectile / 2.0 + m2_projectile / 2.0 / wplus_projectile;
    p_projectile.v.z = pz_projectile;
    p_projectile.e = e_projectile;
    p_projectile = lorentz_apply(to_lab, p_projectile);
    w->projectile_p4 = p_projectile;

    // `theParticipants.StartLoop(); theParticipants.Next();` - the FIRST interaction's
    // projectile is the primary, and every interaction in the hadron arm shares it.
    w->participants.start_loop();
    if (w->participants.next()) {
      const int slot = w->participants.interaction().projectile;
      if (slot != kNullSplitable) { w->participants.pool[slot].momentum = p_projectile; }
    }
  } else {
    is_ok = ftf_finalize_kinematics(
        wplus_projectile, true, to_lab, pr_residual_mass, w->projectile_residual_a,
        w->involved_projectile, w->n_involved_projectile, &w->projectile, w->participants.pool,
        &w->projectile_residual_p4);
    if (!is_ok) { return false; }
    w->projectile_residual_p4 = lorentz_apply(to_lab, w->projectile_residual_p4);
  }

  is_ok = ftf_finalize_kinematics(wminus_target, false, to_lab, target_residual_mass,
                                  w->target_residual_a, w->involved_target,
                                  w->n_involved_target, &w->target, w->participants.pool,
                                  &w->target_residual_p4);
  if (!is_ok) { return false; }
  w->target_residual_p4 = lorentz_apply(to_lab, w->target_residual_p4);
  return true;
}

/// G4FTFModel::ExciteParticipants - the loop over collisions.
///
/// `MaxNumOfInelCollisions` is `GetMaxNumberOfCollisions()` rounded DOWN plus one more with
/// probability equal to the fractional part; when the parameter is <= 0 (Plab below the bound)
/// it is forced to 1 and the fractional deviate is NOT spent - so the draw count says which
/// branch ran.
///
/// The inelastic-versus-elastic decision inside a collision is
/// `u < (1 - n_t/Nmax)(1 - n_p/Nmax)`, with n the soft-collision counts so far: a nucleon that
/// has already collided Nmax times can no longer scatter inelastically.
///
/// ANNIHILATION IS REFUSED BY NAME. `GetProbabilityOfAnnihilation()` is non-zero only for an
/// anti-baryon projectile (G4FTFParameters::InitForInteraction's anti-baryon branch), so every
/// other beam takes the `u > ProbabilityOfAnnihilation` arm with probability 1 and never
/// reaches it. For an anti-nucleon it fires, and `kFtfAnnihilation` is reported rather than
/// approximated - G4FTFAnnihilation's five channels are not written.
template <int kA, int kP, int kI, int kS, typename Rng>
__host__ __device__ inline bool ftf_model_excite_participants(
    FtfModelWorkspace<kA, kP, kI, kS>* w, Rng& rng) {
  bool success = false;
  int max_num_of_inel_collisions = static_cast<int>(w->params.max_number_of_collisions);
  if (max_num_of_inel_collisions > 0) {
    const double prob_max_number = w->params.max_number_of_collisions - max_num_of_inel_collisions;
    if (rng.uniform() < prob_max_number) { max_num_of_inel_collisions++; }
  } else {
    max_num_of_inel_collisions = 1;
  }

  int current_interaction = 0;
  w->participants.start_loop();

  bool inner_success = true;
  while (w->participants.next()) {
    current_interaction++;
    Interaction& collision = w->participants.interaction();
    SplitableHadron* projectile = w->splitable(collision.projectile);
    SplitableHadron* target = w->splitable(collision.target);
    if (projectile == nullptr || target == nullptr) {
      w->participants.null_target = true;
      continue;
    }

    if (collision.status) {
      if (rng.uniform() < w->params.prob_of_elastic_scatt) {
        if (!w->high_energy_inter) {
          w->report.refused = FtfRefusal::kAdjustNucleons;
          return false;
        }
        inner_success = ftf_elastic_scattering(projectile, target, &w->params, rng);
      } else if (rng.uniform() > w->params.prob_of_annihilation) {
        if (!w->high_energy_inter) {
          w->report.refused = FtfRefusal::kAdjustNucleons;
          return false;
        }
        // THE DIVISION IS INTEGER. `GetSoftCollisionCount()` returns `G4int` and
        // `MaxNumOfInelCollisions` is a `G4int`, so `n / Nmax` truncates: the factor is EXACTLY
        // 1 until a hadron has had Nmax soft collisions and EXACTLY 0 at Nmax, rather than
        // falling off linearly. (And at 2*Nmax it is -1, so two saturated hadrons give a
        // product of +1 and the collision is accepted again - which is why the count is
        // compared and not clamped.)
        //
        // Written as a floating-point division this reads as a smooth suppression and it is
        // not, and the difference is most of the model's inelasticity: MEASURED on
        // ref/oracle/ftf_modelstat_*.csv for a 10 GeV proton on Pb, the float version rejected
        // enough inelastic collisions to turn them into elastic ones and gave 12% fewer pions,
        // 0.33 fewer quark-exchange tracks per event and an excited-string mass spectrum
        // 27% short in the 1.26-1.58 GeV bin, while the hole count still agreed to 0.2%.
        const int t_ratio = target->collision_count / max_num_of_inel_collisions;
        const int p_ratio = projectile->collision_count / max_num_of_inel_collisions;
        if (rng.uniform() <
            (1.0 - static_cast<double>(t_ratio)) * (1.0 - static_cast<double>(p_ratio))) {
          if (ftf_excite_participants(projectile, target, &w->params, &w->excite, rng)) {
            inner_success = true;
            w->n_nn_collisions++;
          } else {
            if (w->excite.refused != FtfRefusal::kNone) { w->report.refused = w->excite.refused; }
            inner_success = ftf_elastic_scattering(projectile, target, &w->params, rng);
          }
        } else {
          // The inelastic interaction was rejected -> elastic scattering
          inner_success = ftf_elastic_scattering(projectile, target, &w->params, rng);
        }
      } else {
        // Annihilation. Not written: G4FTFAnnihilation's four channel builders and the
        // `theAdditionalString` bookkeeping under them.
        w->report.refused = FtfRefusal::kFtfAnnihilation;
        return false;
      }
    }

    if (inner_success) { success = true; }
  }
  return success;
}

/// G4FTFModel::BuildStrings.
///
/// The FIVE cases per hadron, and each of them is a different physical object:
///   status 0                       -> a string from its two partons (non-diffractive)
///   status 1, SoftCollisionCount!=0 -> a string from its two partons (diffractive)
///   status 1, SoftCollisionCount==0, high energy -> a KINETIC TRACK, not a string: the nucleon
///                                   was marked but its interaction was skipped
///   status 1, SoftCollisionCount==0, low energy  -> nothing; the nucleon goes back into the
///                                   nucleus with status 5
///   status 2 or 3                  -> a kinetic track (quark exchange, or reggeon cascade)
///
/// `NumberOfProjectileSpectatorNucleons` and `NumberOfTargetSpectatorNucleons` are decremented
/// case by case, and NOT for the status-3 reggeon nucleons - a reggeon-cascaded nucleon is
/// counted as a spectator even though a kinetic track is built for it.
///
/// The hadron-projectile arm de-duplicates the primary: every interaction in a hadron-nucleus
/// event shares one `G4VSplitableHadron*`, so `primaries` ends up with exactly one entry.
template <int kA, int kP, int kI, int kS, typename Rng>
__host__ __device__ inline void ftf_build_strings(FtfModelWorkspace<kA, kP, kI, kS>* w,
                                                  Rng& rng) {
  ExcitedString made[2];
  int n_made = 0;

  if (!w->has_projectile_nucleus) {
    // `primaries` without duplicates. In the hadron arm every live interaction names the same
    // slot, so the list is one long; the de-duplication is transcribed rather than assumed.
    int primaries[2] = {kNullSplitable, kNullSplitable};
    int n_primaries = 0;
    w->participants.start_loop();
    while (w->participants.next()) {
      const Interaction& in = w->participants.interaction();
      if (in.status) {
        bool found = false;
        for (int k = 0; k < n_primaries; ++k) {
          if (primaries[k] == in.projectile) { found = true; }
        }
        if (!found) {
          if (n_primaries >= 2) {
            w->report.string_capacity = true;
            return;
          }
          primaries[n_primaries++] = in.projectile;
        }
      }
    }

    for (int ah = 0; ah < n_primaries; ++ah) {
      SplitableHadron* p = w->splitable(primaries[ah]);
      if (p == nullptr) { continue; }
      n_made = 0;
      if (p->status == 0) {
        ftf_create_strings(p, true, &w->params, made, &n_made, &w->report.refused, rng);
        w->n_projectile_spectators--;
      } else if (p->status == 1 && p->collision_count != 0) {
        ftf_create_strings(p, true, &w->params, made, &n_made, &w->report.refused, rng);
        w->n_projectile_spectators--;
      } else if (p->status == 1 && p->collision_count == 0) {
        made[0] = ExcitedString();
        made[0].excited = false;
        made[0].track_pdg = ftf_kinetic_track_pdg<double>(p->pdg, rng);
        made[0].track_mom = p->momentum;
        made[0].track_time = p->time_of_creation;
        made[0].track_position = p->position;
        made[0].time_of_creation = p->time_of_creation;
        made[0].position = p->position;
        made[0].direction = 0;  // G4ExcitedString(G4KineticTrack*) sets theDirection = 0
        n_made = 1;
      } else if (p->status == 2) {
        made[0] = ExcitedString();
        made[0].excited = false;
        made[0].track_pdg = ftf_kinetic_track_pdg<double>(p->pdg, rng);
        made[0].track_mom = p->momentum;
        made[0].track_time = p->time_of_creation;
        made[0].track_position = p->position;
        made[0].time_of_creation = p->time_of_creation;
        made[0].position = p->position;
        made[0].direction = 0;
        n_made = 1;
        w->n_projectile_spectators--;
      } else {
        // Geant4 prints "Something wrong in FTF Model Build String" and builds nothing.
        n_made = 0;
      }
      for (int k = 0; k < n_made; ++k) {
        if (w->n_strings >= kS) {
          w->report.string_capacity = true;
          return;
        }
        w->strings[w->n_strings++] = made[k];
      }
    }
  } else {
    for (int ah = 0; ah < w->n_involved_projectile; ++ah) {
      bic::Nucleon* n = &w->projectile.nucleons[w->involved_projectile[ah]];
      SplitableHadron* p = w->splitable(n->hit_by);
      if (p == nullptr) { continue; }
      n_made = 0;
      if (p->status == 0) {
        ftf_create_strings(p, true, &w->params, made, &n_made, &w->report.refused, rng);
        w->n_projectile_spectators--;
      } else if (p->status == 1 && p->collision_count != 0) {
        ftf_create_strings(p, true, &w->params, made, &n_made, &w->report.refused, rng);
        w->n_projectile_spectators--;
      } else if (p->status == 1 && p->collision_count == 0 && w->high_energy_inter) {
        made[0] = ExcitedString();
        made[0].excited = false;
        made[0].track_pdg = ftf_kinetic_track_pdg<double>(p->pdg, rng);
        made[0].track_mom = p->momentum;
        made[0].track_time = p->time_of_creation;
        made[0].track_position = p->position;
        made[0].time_of_creation = p->time_of_creation;
        made[0].position = p->position;
        made[0].direction = 0;
        n_made = 1;
      } else if (p->status == 2 || p->status == 3) {
        made[0] = ExcitedString();
        made[0].excited = false;
        made[0].track_pdg = ftf_kinetic_track_pdg<double>(p->pdg, rng);
        made[0].track_mom = p->momentum;
        made[0].track_time = p->time_of_creation;
        made[0].track_position = p->position;
        made[0].time_of_creation = p->time_of_creation;
        made[0].position = p->position;
        made[0].direction = 0;
        n_made = 1;
        if (p->status == 2) { w->n_projectile_spectators--; }
      } else {
        n_made = 0;
      }
      for (int k = 0; k < n_made; ++k) {
        if (w->n_strings >= kS) {
          w->report.string_capacity = true;
          return;
        }
        w->strings[w->n_strings++] = made[k];
      }
    }
  }

  // Target-like strings
  for (int ah = 0; ah < w->n_involved_target; ++ah) {
    bic::Nucleon* nn = &w->target.nucleons[w->involved_target[ah]];
    SplitableHadron* a = w->splitable(nn->hit_by);
    if (a == nullptr) { continue; }
    n_made = 0;

    if (a->status == 0) {
      ftf_create_strings(a, false, &w->params, made, &n_made, &w->report.refused, rng);
      w->n_target_spectators--;
    } else if (a->status == 1 && a->collision_count != 0) {
      ftf_create_strings(a, false, &w->params, made, &n_made, &w->report.refused, rng);
      w->n_target_spectators--;
    } else if (a->status == 1 && a->collision_count == 0 && w->high_energy_inter) {
      made[0] = ExcitedString();
      made[0].excited = false;
      made[0].track_pdg = ftf_kinetic_track_pdg<double>(a->pdg, rng);
      made[0].track_mom = a->momentum;
      made[0].track_time = a->time_of_creation;
      made[0].track_position = a->position;
      made[0].time_of_creation = a->time_of_creation;
      made[0].position = a->position;
      made[0].direction = 0;
      n_made = 1;
    } else if (a->status == 1 && a->collision_count == 0 && !w->high_energy_inter) {
      a->status = 5;  // the comment says "4->5": back into the nucleus, no string
      n_made = 0;
    } else if (a->status == 2 || a->status == 3) {
      made[0] = ExcitedString();
      made[0].excited = false;
      made[0].track_pdg = ftf_kinetic_track_pdg<double>(a->pdg, rng);
      made[0].track_mom = a->momentum;
      made[0].track_time = a->time_of_creation;
      made[0].track_position = a->position;
      made[0].time_of_creation = a->time_of_creation;
      made[0].position = a->position;
      made[0].direction = 0;
      n_made = 1;
      if (a->status == 2) { w->n_target_spectators--; }
    } else {
      n_made = 0;
    }

    for (int k = 0; k < n_made; ++k) {
      if (w->n_strings >= kS) {
        w->report.string_capacity = true;
        return;
      }
      w->strings[w->n_strings++] = made[k];
    }
  }

  // `theAdditionalString` is filled only by G4FTFAnnihilation, which this port refuses; there
  // is nothing to loop over and the loop is not written as an empty one.
}

/// G4FTFModel::GetResiduals, the HighEnergyInter arm.
///
/// This is the method that makes the hand-over to P6 work: every involved nucleon is given
/// `-TargetResidual4Momentum / N` as its momentum and `TargetResidualExcitationEnergy / N` as
/// its BINDING ENERGY - so `G4GeneratorPrecompoundInterface::Propagate`'s loop over the hit
/// nucleons reads exactly the residual's four-momentum back out, spread over the holes.
///
/// The SPECTATORS are then rescaled: they are boosted into the residual's rest frame, their
/// mean momentum is subtracted, each is put on the shell `m - BE`, and a bisection on a common
/// scale factor C makes their energies sum to the residual's invariant mass. The bisection runs
/// until `Chigh - Clow <= 0.01` - an ABSOLUTE tolerance on a dimensionless scale, so about 7
/// iterations, and the answer is only two digits. That is Geant4's; the sum after it is not
/// exactly the residual mass and nothing checks that it is.
template <int kA, int kP, int kI, int kS>
__host__ __device__ inline void ftf_get_residuals_one(bic::Nucleus3D* nucleus,
                                                      const int* involved, int n_involved,
                                                      const Vec4& residual_p4,
                                                      double residual_exc, int residual_a) {
  const double delta_exc = residual_exc / static_cast<double>(n_involved);
  // `HepLorentzVector operator/(v, c)` is `oneOverC = 1.0/c` and then FOUR multiplications
  // (CLHEP LorentzVector.cc:161), not four divisions. The two differ by an ulp, and this
  // quantity is what every wounded nucleon's momentum is set to - i.e. it is exactly the
  // number `preco::propagate_residual` sums back up to build the residual, where a
  // cancellation amplifies it. The same reciprocal-then-multiply as `boost_vector()`
  // (deexcitation/fragment.cuh's note) and for the same reason.
  const double inv_n = 1.0 / static_cast<double>(n_involved);
  const Vec4 delta_p(residual_p4.v.x * inv_n, residual_p4.v.y * inv_n, residual_p4.v.z * inv_n,
                     residual_p4.e * inv_n);

  for (int i = 0; i < n_involved; ++i) {
    bic::Nucleon* a = &nucleus->nucleons[involved[i]];
    a->momentum = Vec4(-delta_p.v.x, -delta_p.v.y, -delta_p.v.z, -delta_p.e);
    a->binding_energy = delta_exc;
  }

  if (residual_a == 0) { return; }

  // findBoostToCM() is `-boostVector()`.
  const Vec3d bv = residual_p4.boost_vector();
  const Vec3d bst_to_cm{-bv.x, -bv.y, -bv.z};

  Vec4 residual_momentum(0.0, 0.0, 0.0, 0.0);
  for (int i = 0; i < nucleus->my_a; ++i) {
    bic::Nucleon* a = &nucleus->nucleons[i];
    if (!a->hit) {
      Vec4 tmp = a->momentum;
      tmp.boost(bst_to_cm);
      a->momentum = tmp;
      residual_momentum = residual_momentum + tmp;
    }
  }
  // `operator/=` is the same reciprocal-then-multiply (CLHEP LorentzVector.cc:148).
  const double inv_a = 1.0 / static_cast<double>(residual_a);
  residual_momentum = Vec4(residual_momentum.v.x * inv_a, residual_momentum.v.y * inv_a,
                           residual_momentum.v.z * inv_a, residual_momentum.e * inv_a);

  const double mass = residual_p4.mag();
  double sum_masses = 0.0;

  for (int i = 0; i < nucleus->my_a; ++i) {
    bic::Nucleon* a = &nucleus->nucleons[i];
    if (!a->hit) {
      Vec4 tmp = a->momentum - residual_momentum;
      const double m = a->pdg_mass() - a->binding_energy;
      const double e = std::sqrt(g4gpu::mag2(tmp.v) + m * m);
      tmp.e = e;
      a->momentum = tmp;
      sum_masses += e;
    }
  }

  double chigh = mass / sum_masses;
  double clow = 0.0;
  double c = 0.0;
  const int max_number_of_loops = 1000;
  int loop_counter = 0;
  do {
    c = (chigh + clow) / 2.0;
    sum_masses = 0.0;
    for (int i = 0; i < nucleus->my_a; ++i) {
      bic::Nucleon* a = &nucleus->nucleons[i];
      if (!a->hit) {
        const Vec4 tmp = a->momentum;
        const double m = a->pdg_mass() - a->binding_energy;
        sum_masses += std::sqrt(g4gpu::mag2(tmp.v) * c * c + m * m);
      }
    }
    if (sum_masses > mass) { chigh = c; } else { clow = c; }
  } while (chigh - clow > 0.01 && ++loop_counter < max_number_of_loops);
  if (loop_counter >= max_number_of_loops) { return; }

  for (int i = 0; i < nucleus->my_a; ++i) {
    bic::Nucleon* a = &nucleus->nucleons[i];
    if (!a->hit) {
      Vec4 tmp = a->momentum;
      tmp = Vec4(tmp.v.x * c, tmp.v.y * c, tmp.v.z * c, tmp.e * c);
      const double m = a->pdg_mass() - a->binding_energy;
      const double e = std::sqrt(g4gpu::mag2(tmp.v) + m * m);
      tmp.e = e;
      tmp.boost(Vec3d{-bst_to_cm.x, -bst_to_cm.y, -bst_to_cm.z});
      a->momentum = tmp;
    }
  }
}

/// G4FTFModel::GetResiduals.
template <int kA, int kP, int kI, int kS>
__host__ __device__ inline void ftf_get_residuals(FtfModelWorkspace<kA, kP, kI, kS>* w) {
  if (w->high_energy_inter) {
    if (w->n_involved_target > 0) {
      ftf_get_residuals_one<kA, kP, kI, kS>(&w->target, w->involved_target, w->n_involved_target,
                                            w->target_residual_p4, w->target_residual_exc,
                                            w->target_residual_a);
    }
    if (!w->has_projectile_nucleus) { return; }
    if (w->n_involved_projectile > 0) {
      ftf_get_residuals_one<kA, kP, kI, kS>(&w->projectile, w->involved_projectile,
                                            w->n_involved_projectile, w->projectile_residual_p4,
                                            w->projectile_residual_exc,
                                            w->projectile_residual_a);
    }
    return;
  }

  // The low-energy arm is reached only through AdjustNucleons, which is refused before any
  // collision runs, so nothing can arrive here. Reported rather than silently skipped.
  w->report.refused = FtfRefusal::kAdjustNucleons;
}

/// G4FTFModel::Init.
///
/// `PlabPerParticle` is the projectile's **z momentum**, per nucleon for an ion, and it is what
/// `G4FTFParameters::InitForInteraction`'s fourth argument wants (whose declaration calls it
/// `s` - the header's name is wrong; ftf_parameters.cuh records that).
///
/// The ANTI-NUCLEUS branch re-types every nucleon of the projectile nucleus after building it:
/// `G4Fancy3DNucleus` has no anti-nucleon mode (bic/nucleus/nucleus_model.cuh refuses one), so
/// Geant4 builds an ordinary nucleus and then calls `SetParticleType(G4AntiProton...)` on each
/// nucleon. P9's `bic::Nucleon` has no anti types, so this port refuses an ANTI-NUCLEUS
/// projectile by name and accepts an anti-NUCLEON, which needs no nucleus at all.
template <int kA, int kP, int kI, int kS, typename Rng>
__host__ __device__ inline void ftf_model_init(FtfModelWorkspace<kA, kP, kI, kS>* w,
                                               const xs::Projectile<double>& proj,
                                               const Vec4& proj_p4, int target_a, int target_z,
                                               const LundTables<double>* lund, Rng& rng) {
  // Wire P9's scratch to the workspace's own storage. POINTERS AND CAPACITY ONLY: the two
  // Gaussian fields in `Nucleus3DScratch` mirror CLHEP's `RandGauss` thread-local cache and
  // MUST survive from one `nucleus_init` to the next, which is what
  // bic/nucleus/nucleus_model.cuh's note 5 asks for. Re-seeding them here would restart the
  // polar-method pair cache on every Scatter attempt and give a different stream from Geant4's.
  w->scratch.momentum = w->scratch_momentum;
  w->scratch.fermi_p = w->scratch_fermi_p;
  w->scratch.test_sums = w->scratch_sums;
  w->scratch.flat_block = w->scratch_flat;
  w->scratch.capacity = FtfModelWorkspace<kA, kP, kI, kS>::kScratchA;

  w->report = FtfModelReport();
  w->participants.clean();
  w->participants.null_target = false;
  w->participants.loops_exhausted = false;
  w->participants.interaction_capacity = false;
  w->has_projectile_nucleus = false;
  w->n_strings = 0;
  w->n_involved_target = 0;
  w->n_involved_projectile = 0;
  w->n_nn_collisions = 0;
  w->projectile_pdg = proj.pdg;
  w->projectile_p4 = proj_p4;
  w->projectile_baryon = proj.baryon_number;
  w->projectile_charge = static_cast<int>(proj.charge);

  double plab_per_particle = 0.0;

  w->projectile_residual_a = 0;
  w->projectile_residual_z = 0;
  w->projectile_residual_lambda = 0;
  w->projectile_residual_exc = 0.0;
  w->projectile_residual_p4 = Vec4(0.0, 0.0, 0.0, 0.0);

  w->target_residual_a = target_a;
  w->target_residual_z = target_z;
  w->target_residual_exc = 0.0;
  w->target_residual_p4 = Vec4(0.0, 0.0, 0.0, 0.0);
  w->target_residual_p4.e = deex::nuclear_mass(target_a, target_z);

  const int abs_b = (proj.baryon_number < 0) ? -proj.baryon_number : proj.baryon_number;
  if (abs_b <= 1) {
    w->projectile_residual_a = abs_b;
    w->projectile_residual_z = static_cast<int>(proj.charge);
    plab_per_particle = proj_p4.v.z;
    w->projectile_residual_exc = 0.0;
    w->projectile_residual_p4 = proj_p4;
    w->high_energy_inter = (plab_per_particle >= ftf_low_energy_limit());
  } else {
    if (proj.baryon_number < -1) {
      // An anti-nucleus projectile: G4Fancy3DNucleus has no anti-nucleon type and Geant4
      // re-types the nucleons after building the nucleus. P9's model refuses anti-nuclei by
      // name; so does this.
      w->report.refused = FtfRefusal::kAntiNucleusProjectile;
      return;
    }
    if (proj.n_lambdas > 0) {
      w->report.refused = FtfRefusal::kHyperNucleus;
      return;
    }
    if (proj.baryon_number > kP) {
      w->report.involved_capacity = true;
      w->report.refused_a = proj.baryon_number;
      return;
    }
    w->projectile_residual_a = proj.baryon_number;
    w->projectile_residual_z = static_cast<int>(proj.charge);
    w->projectile_residual_lambda = proj.n_lambdas;
    plab_per_particle = proj_p4.v.z / w->projectile_residual_a;
    w->high_energy_inter = (plab_per_particle >= ftf_low_energy_limit());

    w->projectile.nucleons = w->proj_nucleons;
    w->projectile.capacity = kP;
    bic::NucleusReport prep = bic::nucleus_init(w->projectile, w->scratch,
                                                w->projectile_residual_a,
                                                w->projectile_residual_z, rng,
                                                w->projectile_residual_lambda);
    if (prep.fatal()) {
      w->report.nucleus_failed = true;
      return;
    }
    w->projectile.sort_nucleons_dec_z();
    w->has_projectile_nucleus = true;

    const Vec3d boost{proj_p4.v.x / proj_p4.e, proj_p4.v.y / proj_p4.e, proj_p4.v.z / proj_p4.e};
    w->projectile.do_lorentz_boost(boost);
    w->projectile.do_lorentz_contraction(boost);
    w->projectile_residual_exc = 0.0;
    w->projectile_residual_p4 = proj_p4;
  }

  if (target_a > kA) {
    w->report.involved_capacity = true;
    w->report.refused_a = target_a;
    return;
  }
  w->target.nucleons = w->target_nucleons;
  w->target.capacity = kA;
  bic::NucleusReport trep = bic::nucleus_init(w->target, w->scratch, target_a, target_z, rng);
  if (trep.fatal()) {
    w->report.nucleus_failed = true;
    return;
  }
  w->target.sort_nucleons_inc_z();

  w->n_projectile_spectators = abs_b;
  w->n_target_spectators = target_a;
  w->n_nn_collisions = 0;

  ftf_init_for_interaction(&w->params, proj, target_a, target_z, plab_per_particle, lund);
  if (w->params.refused != FtfRefusal::kNone) {
    w->report.refused = w->params.refused;
    return;
  }

  // Hydrogen target, non-ion projectile: the quasi-elastic channel would be identical to the
  // elastic process QBBC already runs, so the elastic probability is forced to zero.
  if (abs_b <= 1 && target_a < 2) { w->params.prob_of_elastic_scatt = 0.0; }
}

/// G4FTFModel::GetStrings - the whole model, in the order the original calls it.
///
/// Geant4 deletes the splitable hadrons and cleans the participants at the end whether or not
/// the event succeeded; the only state that survives into `Scatter` is the STRINGS and the two
/// nuclei's `AreYouHit` marks and momenta. Here the workspace is reused rather than freed, and
/// `clean()` marks the pool dead so a stale slot cannot be read.
template <int kA, int kP, int kI, int kS, typename Rng>
__host__ __device__ inline void ftf_get_strings(FtfModelWorkspace<kA, kP, kI, kS>* w, Rng& rng) {
  w->n_strings = 0;
  if (w->report.any()) { return; }

  if (w->has_projectile_nucleus) {
    ftf_participants_get_list_nucleus(&w->participants, &w->target, &w->projectile, &w->params,
                                      w->projectile_p4, rng);
  } else {
    ftf_participants_get_list_hadron(&w->participants, &w->target, &w->params, w->projectile_pdg,
                                     w->projectile_p4, rng);
  }
  if (w->participants.interaction_capacity) {
    w->report.involved_capacity = true;
    return;
  }
  if (w->participants.n_interactions == 0) {
    w->report.participants_empty = true;
    return;
  }

  ftf_store_involved_nucleon(w);
  if (w->report.involved_capacity) { return; }

  bool success = true;
  if (w->high_energy_inter) {
    ftf_reggeon_cascade(w, rng);
    if (w->report.involved_capacity) { return; }
    success = ftf_put_on_mass_shell(w, rng);
    if (!success) { w->report.put_on_mass_shell_failed = true; }
  }

  if (success) { success = ftf_model_excite_participants(w, rng); }
  if (!success && !w->report.put_on_mass_shell_failed) { w->report.excite_failed = true; }
  if (w->report.refused != FtfRefusal::kNone) { return; }

  if (success) {
    ftf_build_strings(w, rng);
    if (w->report.refused != FtfRefusal::kNone || w->report.string_capacity) { return; }
    ftf_get_residuals(w);
  }
}

}  // namespace g4gpu::hadronic::ftf
