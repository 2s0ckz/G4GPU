// G4FTFModel::AdjustNucleons and its three algorithm methods - the sub-GeV arm of FTFP.
//
// Transcribed from
//   source/processes/hadronic/models/parton_string/diffraction/src/G4FTFModel.cc
//     AdjustNucleons, AdjustNucleonsAlgorithm_beforeSampling,
//     AdjustNucleonsAlgorithm_Sampling, AdjustNucleonsAlgorithm_afterSampling
//
// ## What it is for, and who reaches it
//
// `G4FTFModel::ExciteParticipants` calls this before EVERY collision when `HighEnergyInter` is
// false, i.e. when the projectile carries less than 1 GeV/c per nucleon. `PutOnMassShell` has
// not run in that case - `GetStrings` skips it - so the two colliding hadrons are still off
// their mass shells and the residual nuclei have no kinematics. AdjustNucleons does for one
// collision what PutOnMassShell does for the whole event: it pulls a nucleon out of a residual
// nucleus, gives it and the remaining residual a transverse momentum and a light-cone fraction,
// and puts the pair on shell in their common c.m.s.
//
// QBBC reaches it through exactly one door. `G4HadronicBuilder::BuildFTFP_BERT(..., bert=false)`
// gives FTFP **all** energies for anti-nucleons, anti-light-ions and anti-hyperons, so an
// anti-proton at rest is an FTFP event; every other beam is handed to FTFP only above 3 GeV and
// never gets here. That is why the parameter is called `SelectedAntiBaryon` and why the
// `Annihilation` flag exists - at rest, annihilation is the whole cross section.
//
// ## The three interaction cases, and they are not symmetric
//
//   1. hadron-nucleus, or a projectile nucleon that already collided meeting a fresh target
//      nucleon. The projectile keeps its own mass; the TARGET residual gives up a nucleon.
//   2. a fresh projectile nucleon meeting a target nucleon that already collided. The mirror
//      image - and Geant4 writes it by REUSING the `TResidual*` variables for the PROJECTILE
//      residual. `common.t_residual_mass_number` in case 2 is the projectile's. That is not a
//      transcription slip here; it is what the source does, and `_afterSampling` reads them
//      back the same way.
//   3. nucleus-nucleus, both fresh. This is the only case with a separate `PResidual*` set.
//
// ## Where the deviates go
//
// `- ExcitationEnergyPerWoundedNucleon * log(G4UniformRand())` is drawn once per residual that
// gives up a nucleon - once in cases 1 and 2, TWICE in case 3, and the projectile's is drawn
// FIRST there. Then the sampling loop draws a `GaussianPt` (two deviates) per nucleon and per
// residual x-fraction. A failed inner loop redraws; a failed outer loop redraws everything.
//
// ## Refused by name
//
// The hypernucleus arm: `common.PResidualLambdaNumber > 0` with `PResidualMassNumber > 2` needs
// `G4HyperNucleiProperties::GetNuclearMass`, which P3 refuses (`kHyperNucleus`). The two- and
// one-nucleon hyper cases above it are transcribed because they are sums of PDG masses and need
// no table. An ANTI-nucleus projectile is refused before this is reached.
#pragma once
#include <cmath>

#include "core/units.cuh"
#include "physics/hadronic/deexcitation/nuclear_masses.cuh"
#include "physics/hadronic/ftf/ftf_parameters.cuh"
#include "physics/hadronic/ftf/lorentz.cuh"
#include "physics/hadronic/ftf/refusal.cuh"
#include "physics/hadronic/ftf/splitable_hadron.cuh"

namespace g4gpu::hadronic::ftf {

/// The residual nucleus AdjustNucleons reads and writes, one per side.
///
/// It is a view onto `FtfModelWorkspace`'s `*_residual_*` fields rather than a copy, because
/// every branch of the algorithm both reads them and writes them back.
struct AdjustResidual {
  int* mass_number = nullptr;
  int* charge = nullptr;
  int* lambda_number = nullptr;   ///< nullptr for the target, which is never a hypernucleus
  double* excitation = nullptr;
  Vec4* momentum = nullptr;
};

/// G4FTFModel::CommonVariables, the subset AdjustNucleons uses.
struct AdjustCommon {
  Vec4 psum, pprojectile, ptarget, ptmp;
  LorentzRot to_cms, to_lab;
  double sqrt_s = 0.0, s = 0.0;
  double sum_masses = 0.0;
  double mprojectile = 0.0, m2projectile = 0.0, mtarget = 0.0, m2target = 0.0;

  // The "T" set is the TARGET residual in cases 1 and 3 and the PROJECTILE residual in case 2.
  int t_residual_mass_number = 0, t_residual_charge = 0;
  double t_residual_excitation = 0.0, t_residual_mass = 0.0, t_nucleon_mass = 0.0;
  // The "P" set exists only in case 3.
  int p_residual_mass_number = 0, p_residual_charge = 0, p_residual_lambda = 0;
  double p_residual_excitation = 0.0, p_residual_mass = 0.0, p_nucleon_mass = 0.0;

  Vec4 t_residual_p4, p_residual_p4;
  double y_target_nucleus = 0.0, y_projectile_nucleus = 0.0;

  Vec3d pt_nucleon{0.0, 0.0, 0.0}, pt_residual{0.0, 0.0, 0.0};
  Vec3d pt_nucleon_p{0.0, 0.0, 0.0}, pt_residual_p{0.0, 0.0, 0.0};
  Vec3d pt_nucleon_t{0.0, 0.0, 0.0}, pt_residual_t{0.0, 0.0, 0.0};
  double xplus_nucleon = 0.0, xplus_residual = 0.0;
  double xminus_nucleon = 0.0, xminus_residual = 0.0;
  double wplus_projectile = 0.0, wminus_target = 0.0;
  double pzprojectile = 0.0, eprojectile = 0.0, pztarget = 0.0, etarget = 0.0;
  double mt2_projectile_nucleon = 0.0, mt2_target_nucleon = 0.0;
  double pz_projectile_nucleon = 0.0, e_projectile_nucleon = 0.0;
  double pz_target_nucleon = 0.0, e_target_nucleon = 0.0;

  FtfRefusal refused = FtfRefusal::kNone;
};

/// `G4ParticleTable::GetIonTable()->GetIonMass(Z, A)`, which is
/// `G4NucleiProperties::GetNuclearMass(A, Z)` - P3's `deex::nuclear_mass`, and NOT the sum of
/// nucleon masses minus a binding energy.
__host__ __device__ inline double adjust_ion_mass(int a, int z) {
  return deex::nuclear_mass(a, z);
}

/// The projectile residual's mass in case 3, which is the only place the lambda count matters.
///
/// Geant4 spells the A = 1 and A = 2 cases out as sums of PDG masses and only calls
/// `G4HyperNucleiProperties` for A > 2 with a lambda in it. That last branch is REFUSED
/// (`kHyperNucleus`) rather than approximated by the ordinary ion mass, which would be wrong by
/// the lambda-nucleon mass difference times the lambda count.
__host__ __device__ inline double adjust_projectile_residual_mass(int a, int z, int n_lambda,
                                                                  FtfRefusal* refused) {
  const int az = (z < 0) ? -z : z;
  if (a == 0) { return 0.0; }
  if (a == 1) {
    if (az == 1) { return bic::nucleon_pdg_mass(bic::kProton); }
    if (n_lambda == 1) { return bic::nucleon_pdg_mass(bic::kLambda); }
    return bic::nucleon_pdg_mass(bic::kNeutron);
  }
  if (n_lambda > 0) {
    if (a == 2) {
      double m = bic::nucleon_pdg_mass(bic::kLambda);
      if (az == 1) {
        m += bic::nucleon_pdg_mass(bic::kProton);       // lambda + proton
      } else if (n_lambda == 1) {
        m += bic::nucleon_pdg_mass(bic::kNeutron);      // lambda + neutron
      } else {
        m += bic::nucleon_pdg_mass(bic::kLambda);       // lambda + lambda
      }
      return m;
    }
    if (refused != nullptr) { *refused = FtfRefusal::kHyperNucleus; }
    return 0.0;
  }
  return adjust_ion_mass(a, az);
}

/// G4FTFModel::AdjustNucleonsAlgorithm_beforeSampling.
///
/// Returns 0 (done - the "Stopping" branch put both hadrons at rest in the c.m.s. and wrote both
/// residuals back), 1 (continue into the sampling) or 99 (failed; the caller skips the
/// collision).
template <typename Rng>
__host__ __device__ inline int ftf_adjust_before_sampling(
    int interaction_case, SplitableHadron* selected_antibaryon, const bic::Nucleon* proj_nucleon,
    SplitableHadron* selected_target_nucleon, const bic::Nucleon* targ_nucleon,
    bool annihilation, const FtfParameters<double>* params, const AdjustResidual& proj_res,
    const AdjustResidual& targ_res, AdjustCommon* c, Rng& rng) {
  const int failed = 99;
  const double exc_per_wounded = params->excitation_energy_per_wounded_nucleon;

  if (interaction_case == 1) {
    c->psum = selected_antibaryon->momentum + *targ_res.momentum;
    c->pprojectile = selected_antibaryon->momentum;
  } else if (interaction_case == 2) {
    c->psum = *proj_res.momentum + selected_target_nucleon->momentum;
    c->pprojectile = *proj_res.momentum;
  } else {
    c->psum = *proj_res.momentum + *targ_res.momentum;
    c->pprojectile = *proj_res.momentum;
  }

  const Vec3d bv = c->psum.boost_vector();
  c->to_cms = lorentz_boost(Vec3d{-bv.x, -bv.y, -bv.z});
  c->ptmp = lorentz_apply(c->to_cms, c->pprojectile);
  lorentz_rotate_z(&c->to_cms, -lv_phi(c->ptmp));
  lorentz_rotate_y(&c->to_cms, -lv_theta(c->ptmp));
  c->pprojectile = lorentz_apply(c->to_cms, c->pprojectile);
  c->to_lab = lorentz_inverse(c->to_cms);
  c->sqrt_s = c->psum.mag();
  c->s = c->sqrt_s * c->sqrt_s;

  bool stopping = false;
  if (interaction_case == 1) {
    c->t_residual_mass_number = *targ_res.mass_number - 1;
    c->t_residual_charge = *targ_res.charge - targ_nucleon->charge();
    c->t_residual_excitation = *targ_res.excitation - exc_per_wounded * std::log(rng.uniform());
    if (c->t_residual_mass_number <= 1) { c->t_residual_excitation = 0.0; }
    if (c->t_residual_mass_number != 0) {
      c->t_residual_mass = adjust_ion_mass(c->t_residual_mass_number, c->t_residual_charge);
    }
    c->t_nucleon_mass = targ_nucleon->pdg_mass();
    c->sum_masses =
        selected_antibaryon->momentum.mag() + c->t_nucleon_mass + c->t_residual_mass;
  } else if (interaction_case == 2) {
    c->ptarget = lorentz_apply(c->to_cms, selected_target_nucleon->momentum);
    // THE "T" SET IS THE PROJECTILE'S RESIDUAL HERE. Geant4 reuses the variables.
    c->t_residual_mass_number = *proj_res.mass_number - 1;
    const int q = proj_nucleon->charge();
    c->t_residual_charge = *proj_res.charge - ((q < 0) ? -q : q);
    c->t_residual_excitation = *proj_res.excitation - exc_per_wounded * std::log(rng.uniform());
    if (c->t_residual_mass_number <= 1) { c->t_residual_excitation = 0.0; }
    if (c->t_residual_mass_number != 0) {
      c->t_residual_mass = adjust_ion_mass(c->t_residual_mass_number, c->t_residual_charge);
    }
    c->t_nucleon_mass = proj_nucleon->pdg_mass();
    c->sum_masses =
        selected_target_nucleon->momentum.mag() + c->t_nucleon_mass + c->t_residual_mass;
  } else {
    c->ptarget = lorentz_apply(c->to_cms, *targ_res.momentum);
    c->p_residual_mass_number = *proj_res.mass_number - 1;
    const int q = proj_nucleon->charge();
    c->p_residual_charge = *proj_res.charge - ((q < 0) ? -q : q);
    c->p_residual_lambda = (proj_res.lambda_number != nullptr) ? *proj_res.lambda_number : 0;
    if (proj_nucleon->type == bic::kLambda) { --c->p_residual_lambda; }
    c->p_residual_excitation = *proj_res.excitation - exc_per_wounded * std::log(rng.uniform());
    if (c->p_residual_mass_number <= 1) { c->p_residual_excitation = 0.0; }
    if (c->p_residual_mass_number != 0) {
      c->p_residual_mass = adjust_projectile_residual_mass(
          c->p_residual_mass_number, c->p_residual_charge, c->p_residual_lambda, &c->refused);
      if (c->refused != FtfRefusal::kNone) { return failed; }
    }
    c->p_nucleon_mass = proj_nucleon->pdg_mass();
    c->t_residual_mass_number = *targ_res.mass_number - 1;
    c->t_residual_charge = *targ_res.charge - targ_nucleon->charge();
    c->t_residual_excitation = *targ_res.excitation - exc_per_wounded * std::log(rng.uniform());
    if (c->t_residual_mass_number <= 1) { c->t_residual_excitation = 0.0; }
    if (c->t_residual_mass_number != 0) {
      c->t_residual_mass = adjust_ion_mass(c->t_residual_mass_number, c->t_residual_charge);
    }
    c->t_nucleon_mass = targ_nucleon->pdg_mass();
    c->sum_masses =
        c->p_nucleon_mass + c->p_residual_mass + c->t_nucleon_mass + c->t_residual_mass;
  }

  if (!annihilation) {
    if (c->sqrt_s < c->sum_masses) { return failed; }
    if (interaction_case == 1 || interaction_case == 2) {
      if (c->sqrt_s < c->sum_masses + c->t_residual_excitation) {
        // Geant4 sets `Stopping = true` here and RETURNS the failure code anyway, so the
        // assignment is dead. Transcribed as the return it is.
        c->t_residual_excitation = c->sqrt_s - c->sum_masses;
        return failed;
      }
    } else {
      if (c->sqrt_s < c->sum_masses + c->p_residual_excitation + c->t_residual_excitation) {
        stopping = true;
        if (c->p_residual_excitation <= 0.0) {
          c->t_residual_excitation = c->sqrt_s - c->sum_masses;
        } else if (c->t_residual_excitation <= 0.0) {
          c->p_residual_excitation = c->sqrt_s - c->sum_masses;
        } else {
          const double fraction = (c->sqrt_s - c->sum_masses) /
                                  (c->p_residual_excitation + c->t_residual_excitation);
          c->p_residual_excitation *= fraction;
          c->t_residual_excitation *= fraction;
        }
      }
    }
  } else {
    if (c->sqrt_s < c->sum_masses - c->t_nucleon_mass) { return failed; }
    if (c->sqrt_s < c->sum_masses) {
      if (interaction_case == 2 || interaction_case == 3) { c->t_residual_excitation = 0.0; }
      // The nucleon is taken OFF its mass shell to make room: this is the one place in FTF
      // where a nucleon's mass is set from the energy budget rather than from the table.
      c->t_nucleon_mass = c->sqrt_s - (c->sum_masses - c->t_nucleon_mass) -
                          c->t_residual_excitation;
      c->sum_masses = c->sqrt_s - c->t_residual_excitation;
      stopping = true;
    }
    if (interaction_case == 1 || interaction_case == 2) {
      if (c->sqrt_s < c->sum_masses + c->t_residual_excitation) {
        c->t_residual_excitation = c->sqrt_s - c->sum_masses;
        stopping = true;
      }
    } else {
      if (c->sqrt_s < c->sum_masses + c->p_residual_excitation + c->t_residual_excitation) {
        stopping = true;
        if (c->p_residual_excitation <= 0.0) {
          c->t_residual_excitation = c->sqrt_s - c->sum_masses;
        } else if (c->t_residual_excitation <= 0.0) {
          c->p_residual_excitation = c->sqrt_s - c->sum_masses;
        } else {
          const double fraction = (c->sqrt_s - c->sum_masses) /
                                  (c->p_residual_excitation + c->t_residual_excitation);
          c->p_residual_excitation *= fraction;
          c->t_residual_excitation *= fraction;
        }
      }
    }
  }

  if (stopping) {
    // Both hadrons are put AT REST in the c.m.s. and the residuals take the recoil. Note that
    // the two `save*4Momentum` vectors are the hadrons' ORIGINAL momenta transformed to the
    // c.m.s., read after the hadron has already been overwritten - so the order of the four
    // statements below matters and is preserved.
    c->ptmp = Vec4(0.0, 0.0, 0.0, 0.0);
    if (interaction_case == 1) {
      c->ptmp.e = selected_antibaryon->momentum.mag();
    } else if (interaction_case == 2) {
      c->ptmp.e = c->t_nucleon_mass;
    } else {
      c->ptmp.e = c->p_nucleon_mass;
    }
    c->pprojectile = lorentz_apply(c->to_lab, c->ptmp);
    const Vec4 save_antibaryon = lorentz_apply(c->to_cms, selected_antibaryon->momentum);
    selected_antibaryon->momentum = c->pprojectile;

    if (interaction_case == 1 || interaction_case == 3) {
      c->ptmp.e = c->t_nucleon_mass;
    } else {
      c->ptmp.e = selected_target_nucleon->momentum.mag();
    }
    c->ptarget = lorentz_apply(c->to_lab, c->ptmp);
    const Vec4 save_target = lorentz_apply(c->to_cms, selected_target_nucleon->momentum);
    selected_target_nucleon->momentum = c->ptarget;

    if (interaction_case == 1 || interaction_case == 3) {
      *targ_res.mass_number = c->t_residual_mass_number;
      *targ_res.charge = c->t_residual_charge;
      *targ_res.excitation = c->t_residual_excitation;
      Vec3d v{-save_target.v.x, -save_target.v.y, -save_target.v.z};
      const double m = c->t_residual_mass + *targ_res.excitation;
      Vec4 p(v.x, v.y, v.z, std::sqrt(m * m + g4gpu::mag2(v)));
      *targ_res.momentum = lorentz_apply(c->to_lab, p);
    }
    if (interaction_case == 2 || interaction_case == 3) {
      Vec4 p;
      if (interaction_case == 2) {
        *proj_res.mass_number = c->t_residual_mass_number;
        *proj_res.charge = c->t_residual_charge;
        // "The target nucleus and its residual are never hypernuclei."
        if (proj_res.lambda_number != nullptr) { *proj_res.lambda_number = 0; }
        *proj_res.excitation = c->t_residual_excitation;
        p = Vec4(0.0, 0.0, 0.0, c->t_residual_mass + *proj_res.excitation);
      } else {
        *proj_res.mass_number = c->p_residual_mass_number;
        *proj_res.charge = c->p_residual_charge;
        if (proj_res.lambda_number != nullptr) {
          *proj_res.lambda_number = c->p_residual_lambda;
        }
        *proj_res.excitation = c->p_residual_excitation;
        Vec3d v{-save_antibaryon.v.x, -save_antibaryon.v.y, -save_antibaryon.v.z};
        const double m = c->p_residual_mass + *proj_res.excitation;
        p = Vec4(v.x, v.y, v.z, std::sqrt(m * m + g4gpu::mag2(v)));
      }
      *proj_res.momentum = lorentz_apply(c->to_lab, p);
    }
    return 0;
  }

  if (interaction_case == 1) {
    c->mprojectile = c->pprojectile.mag();
    // `HepLorentzVector::mag2()` is `t*t - v.mag2()` and keeps its SIGN, which `mag()*mag()`
    // would lose; both are timelike here, and it is written the way CLHEP computes it.
    c->m2projectile = c->pprojectile.e * c->pprojectile.e - g4gpu::mag2(c->pprojectile.v);
    c->t_residual_p4 = lorentz_apply(c->to_cms, *targ_res.momentum);
    c->y_target_nucleus = lv_rapidity(c->t_residual_p4);
    c->t_residual_mass += c->t_residual_excitation;
  } else if (interaction_case == 2) {
    c->mtarget = c->ptarget.mag();
    c->m2target = c->ptarget.e * c->ptarget.e - g4gpu::mag2(c->ptarget.v);
    c->t_residual_p4 = lorentz_apply(c->to_cms, *proj_res.momentum);
    c->y_projectile_nucleus = lv_rapidity(c->t_residual_p4);
    c->t_residual_mass += c->t_residual_excitation;
  } else {
    c->p_residual_p4 = lorentz_apply(c->to_cms, *proj_res.momentum);
    c->y_projectile_nucleus = lv_rapidity(c->p_residual_p4);
    c->t_residual_p4 = lorentz_apply(c->to_cms, *targ_res.momentum);
    c->y_target_nucleus = lv_rapidity(c->t_residual_p4);
    c->p_residual_mass += c->p_residual_excitation;
    c->t_residual_mass += c->t_residual_excitation;
  }
  return 1;
}

/// G4FTFModel::AdjustNucleonsAlgorithm_Sampling. Returns false if it could not sample.
template <typename Rng>
__host__ __device__ inline bool ftf_adjust_sampling(int interaction_case,
                                                    const FtfParameters<double>* params,
                                                    int projectile_residual_a,
                                                    int target_residual_a, AdjustCommon* c,
                                                    Rng& rng) {
  const double dcor = params->dof_nuclear_destruction;
  double dcor_p = 0.0, dcor_t = 0.0;
  if (projectile_residual_a != 0) { dcor_p = dcor / static_cast<double>(projectile_residual_a); }
  if (target_residual_a != 0) { dcor_t = dcor / static_cast<double>(target_residual_a); }
  double average_pt2 = params->pt2_of_nuclear_destruction;
  const double max_pt_square = params->max_pt2_of_nuclear_destruction;

  double scale_factor = 1.0;
  bool outer_success = true;
  const int max_number_of_loops = 1000;
  const int max_number_of_tries = 10000;
  int loop_counter = 0;
  int number_of_tries = 0;
  do {
    outer_success = true;
    bool loop_condition = false;
    do {
      // `NumberOfTries == 100*(NumberOfTries/100)` is an integer-division test for a multiple of
      // 100, and it is TRUE at zero - so the halving fires on the very first pass and
      // `AveragePt2` is already half the parameter before anything is sampled.
      if (number_of_tries == 100 * (number_of_tries / 100)) {
        scale_factor /= 2.0;
        dcor_p *= scale_factor;
        dcor_t *= scale_factor;
        average_pt2 *= scale_factor;
      }

      if (interaction_case == 2) {
        if (projectile_residual_a > 1) {
          c->pt_nucleon = ftf_gaussian_pt_model(average_pt2, max_pt_square, rng);
        } else {
          c->pt_nucleon = Vec3d{0.0, 0.0, 0.0};
        }
        c->pt_residual = Vec3d{-c->pt_nucleon.x, -c->pt_nucleon.y, -c->pt_nucleon.z};
        c->mprojectile =
            std::sqrt(c->t_nucleon_mass * c->t_nucleon_mass + g4gpu::mag2(c->pt_nucleon)) +
            std::sqrt(c->t_residual_mass * c->t_residual_mass + g4gpu::mag2(c->pt_residual));
        c->m2projectile = c->mprojectile * c->mprojectile;
        if (c->sqrt_s < c->mtarget + c->mprojectile) {
          outer_success = false;
          loop_condition = true;
          continue;
        }
      } else if (interaction_case == 3) {
        if (projectile_residual_a > 1) {
          c->pt_nucleon_p = ftf_gaussian_pt_model(average_pt2, max_pt_square, rng);
        } else {
          c->pt_nucleon_p = Vec3d{0.0, 0.0, 0.0};
        }
        c->pt_residual_p = Vec3d{-c->pt_nucleon_p.x, -c->pt_nucleon_p.y, -c->pt_nucleon_p.z};
        if (target_residual_a > 1) {
          c->pt_nucleon_t = ftf_gaussian_pt_model(average_pt2, max_pt_square, rng);
        } else {
          c->pt_nucleon_t = Vec3d{0.0, 0.0, 0.0};
        }
        c->pt_residual_t = Vec3d{-c->pt_nucleon_t.x, -c->pt_nucleon_t.y, -c->pt_nucleon_t.z};
        c->mprojectile =
            std::sqrt(c->p_nucleon_mass * c->p_nucleon_mass + g4gpu::mag2(c->pt_nucleon_p)) +
            std::sqrt(c->p_residual_mass * c->p_residual_mass + g4gpu::mag2(c->pt_residual_p));
        c->m2projectile = c->mprojectile * c->mprojectile;
        c->mtarget =
            std::sqrt(c->t_nucleon_mass * c->t_nucleon_mass + g4gpu::mag2(c->pt_nucleon_t)) +
            std::sqrt(c->t_residual_mass * c->t_residual_mass + g4gpu::mag2(c->pt_residual_t));
        c->m2target = c->mtarget * c->mtarget;
        if (c->sqrt_s < c->mprojectile + c->mtarget) {
          outer_success = false;
          loop_condition = true;
          continue;
        }
      }

      const int n_execute = (interaction_case == 3) ? 2 : 1;
      for (int i_execute = 0; i_execute < n_execute; ++i_execute) {
        bool inner_success = true;
        const bool target_side =
            (interaction_case == 1) || (interaction_case == 3 && i_execute == 1);
        const bool condition =
            target_side ? (target_residual_a > 1) : (projectile_residual_a > 1);
        if (condition) {
          const int max_inner = 1000;
          int inner_counter = 0;
          do {
            inner_success = true;
            if (target_side) {
              double xcenter = 0.0;
              if (interaction_case == 1) {
                c->pt_nucleon = ftf_gaussian_pt_model(average_pt2, max_pt_square, rng);
                c->pt_residual = Vec3d{-c->pt_nucleon.x, -c->pt_nucleon.y, -c->pt_nucleon.z};
                c->mtarget = std::sqrt(c->t_nucleon_mass * c->t_nucleon_mass +
                                       g4gpu::mag2(c->pt_nucleon)) +
                             std::sqrt(c->t_residual_mass * c->t_residual_mass +
                                       g4gpu::mag2(c->pt_residual));
                if (c->sqrt_s < c->mprojectile + c->mtarget) {
                  inner_success = false;
                  continue;
                }
                xcenter = std::sqrt(c->t_nucleon_mass * c->t_nucleon_mass +
                                    g4gpu::mag2(c->pt_nucleon)) /
                          c->mtarget;
              } else {
                xcenter = std::sqrt(c->t_nucleon_mass * c->t_nucleon_mass +
                                    g4gpu::mag2(c->pt_nucleon_t)) /
                          c->mtarget;
              }
              // `GaussianPt(Dcor*Dcor, 1.0)` is the light-cone fraction's spread: a Gaussian pt
              // sampler used as a one-dimensional smear, and only its x component is read.
              const Vec3d tmp_x = ftf_gaussian_pt_model(dcor_t * dcor_t, 1.0, rng);
              c->xminus_nucleon = xcenter + tmp_x.x;
              if (c->xminus_nucleon <= 0.0 || c->xminus_nucleon >= 1.0) {
                inner_success = false;
                continue;
              }
              c->xminus_residual = 1.0 - c->xminus_nucleon;
            } else {
              const Vec3d tmp_x = ftf_gaussian_pt_model(dcor_p * dcor_p, 1.0, rng);
              double xcenter = 0.0;
              if (interaction_case == 2) {
                xcenter = std::sqrt(c->t_nucleon_mass * c->t_nucleon_mass +
                                    g4gpu::mag2(c->pt_nucleon)) /
                          c->mprojectile;
              } else {
                xcenter = std::sqrt(c->p_nucleon_mass * c->p_nucleon_mass +
                                    g4gpu::mag2(c->pt_nucleon_p)) /
                          c->mprojectile;
              }
              c->xplus_nucleon = xcenter + tmp_x.x;
              if (c->xplus_nucleon <= 0.0 || c->xplus_nucleon >= 1.0) {
                inner_success = false;
                continue;
              }
              c->xplus_residual = 1.0 - c->xplus_nucleon;
            }
          } while ((!inner_success) && ++inner_counter < max_inner);
          if (inner_counter >= max_inner) { return false; }
        } else {
          // "It must be 0, but in the calculation of Pz, E is problematic" - Geant4's comment,
          // and the 1.0 is what makes the residual's light-cone term finite when there is no
          // residual left.
          if (target_side) {
            c->xminus_nucleon = 1.0;
            c->xminus_residual = 1.0;
          } else {
            c->xplus_nucleon = 1.0;
            c->xplus_residual = 1.0;
          }
        }
      }

      if (interaction_case == 1) {
        c->m2target =
            (c->t_nucleon_mass * c->t_nucleon_mass + g4gpu::mag2(c->pt_nucleon)) /
                c->xminus_nucleon +
            (c->t_residual_mass * c->t_residual_mass + g4gpu::mag2(c->pt_residual)) /
                c->xminus_residual;
        loop_condition = (c->sqrt_s < c->mprojectile + std::sqrt(c->m2target));
      } else if (interaction_case == 2) {
        c->m2projectile =
            (c->t_nucleon_mass * c->t_nucleon_mass + g4gpu::mag2(c->pt_nucleon)) /
                c->xplus_nucleon +
            (c->t_residual_mass * c->t_residual_mass + g4gpu::mag2(c->pt_residual)) /
                c->xplus_residual;
        loop_condition = (c->sqrt_s < c->mtarget + std::sqrt(c->m2projectile));
      } else {
        c->m2projectile =
            (c->p_nucleon_mass * c->p_nucleon_mass + g4gpu::mag2(c->pt_nucleon_p)) /
                c->xplus_nucleon +
            (c->p_residual_mass * c->p_residual_mass + g4gpu::mag2(c->pt_residual_p)) /
                c->xplus_residual;
        c->m2target =
            (c->t_nucleon_mass * c->t_nucleon_mass + g4gpu::mag2(c->pt_nucleon_t)) /
                c->xminus_nucleon +
            (c->t_residual_mass * c->t_residual_mass + g4gpu::mag2(c->pt_residual_t)) /
                c->xminus_residual;
        loop_condition = (c->sqrt_s < (std::sqrt(c->m2projectile) + std::sqrt(c->m2target)));
      }
    } while (loop_condition && ++number_of_tries < max_number_of_tries);
    if (number_of_tries >= max_number_of_tries) { return false; }

    double y_projectile = 0.0, y_projectile_nucleon = 0.0;
    double y_target = 0.0, y_target_nucleon = 0.0;
    const double decay_momentum2 =
        c->s * c->s + c->m2projectile * c->m2projectile + c->m2target * c->m2target -
        2.0 * (c->s * (c->m2projectile + c->m2target) + c->m2projectile * c->m2target);
    if (interaction_case == 1) {
      c->wminus_target = (c->s - c->m2projectile + c->m2target + std::sqrt(decay_momentum2)) /
                         2.0 / c->sqrt_s;
      c->wplus_projectile = c->sqrt_s - c->m2target / c->wminus_target;
      c->pzprojectile = c->wplus_projectile / 2.0 - c->m2projectile / 2.0 / c->wplus_projectile;
      c->eprojectile = c->wplus_projectile / 2.0 + c->m2projectile / 2.0 / c->wplus_projectile;
      y_projectile =
          0.5 * std::log((c->eprojectile + c->pzprojectile) / (c->eprojectile - c->pzprojectile));
      c->mt2_target_nucleon =
          c->t_nucleon_mass * c->t_nucleon_mass + g4gpu::mag2(c->pt_nucleon);
      c->pz_target_nucleon = -c->wminus_target * c->xminus_nucleon / 2.0 +
                             c->mt2_target_nucleon /
                                 (2.0 * c->wminus_target * c->xminus_nucleon);
      c->e_target_nucleon = c->wminus_target * c->xminus_nucleon / 2.0 +
                            c->mt2_target_nucleon /
                                (2.0 * c->wminus_target * c->xminus_nucleon);
      y_target_nucleon = 0.5 * std::log((c->e_target_nucleon + c->pz_target_nucleon) /
                                        (c->e_target_nucleon - c->pz_target_nucleon));
      const double dy = y_target_nucleon - c->y_target_nucleus;
      if (((dy < 0.0) ? -dy : dy) > 2 || y_projectile < y_target_nucleon) {
        outer_success = false;
        continue;
      }
    } else if (interaction_case == 2) {
      c->wplus_projectile = (c->s + c->m2projectile - c->m2target + std::sqrt(decay_momentum2)) /
                            2.0 / c->sqrt_s;
      c->wminus_target = c->sqrt_s - c->m2projectile / c->wplus_projectile;
      c->pztarget = -c->wminus_target / 2.0 + c->m2target / 2.0 / c->wminus_target;
      c->etarget = c->wminus_target / 2.0 + c->m2target / 2.0 / c->wminus_target;
      y_target = 0.5 * std::log((c->etarget + c->pztarget) / (c->etarget - c->pztarget));
      c->mt2_projectile_nucleon =
          c->t_nucleon_mass * c->t_nucleon_mass + g4gpu::mag2(c->pt_nucleon);
      c->pz_projectile_nucleon = c->wplus_projectile * c->xplus_nucleon / 2.0 -
                                 c->mt2_projectile_nucleon /
                                     (2.0 * c->wplus_projectile * c->xplus_nucleon);
      c->e_projectile_nucleon = c->wplus_projectile * c->xplus_nucleon / 2.0 +
                                c->mt2_projectile_nucleon /
                                    (2.0 * c->wplus_projectile * c->xplus_nucleon);
      y_projectile_nucleon =
          0.5 * std::log((c->e_projectile_nucleon + c->pz_projectile_nucleon) /
                         (c->e_projectile_nucleon - c->pz_projectile_nucleon));
      const double dy = y_projectile_nucleon - c->y_projectile_nucleus;
      if (((dy < 0.0) ? -dy : dy) > 2 || y_target > y_projectile_nucleon) {
        outer_success = false;
        continue;
      }
    } else {
      c->wplus_projectile = (c->s + c->m2projectile - c->m2target + std::sqrt(decay_momentum2)) /
                            2.0 / c->sqrt_s;
      c->wminus_target = c->sqrt_s - c->m2projectile / c->wplus_projectile;
      c->mt2_projectile_nucleon =
          c->p_nucleon_mass * c->p_nucleon_mass + g4gpu::mag2(c->pt_nucleon_p);
      c->pz_projectile_nucleon = c->wplus_projectile * c->xplus_nucleon / 2.0 -
                                 c->mt2_projectile_nucleon /
                                     (2.0 * c->wplus_projectile * c->xplus_nucleon);
      c->e_projectile_nucleon = c->wplus_projectile * c->xplus_nucleon / 2.0 +
                                c->mt2_projectile_nucleon /
                                    (2.0 * c->wplus_projectile * c->xplus_nucleon);
      y_projectile_nucleon =
          0.5 * std::log((c->e_projectile_nucleon + c->pz_projectile_nucleon) /
                         (c->e_projectile_nucleon - c->pz_projectile_nucleon));
      c->mt2_target_nucleon =
          c->t_nucleon_mass * c->t_nucleon_mass + g4gpu::mag2(c->pt_nucleon_t);
      c->pz_target_nucleon = -c->wminus_target * c->xminus_nucleon / 2.0 +
                             c->mt2_target_nucleon /
                                 (2.0 * c->wminus_target * c->xminus_nucleon);
      c->e_target_nucleon = c->wminus_target * c->xminus_nucleon / 2.0 +
                            c->mt2_target_nucleon /
                                (2.0 * c->wminus_target * c->xminus_nucleon);
      y_target_nucleon = 0.5 * std::log((c->e_target_nucleon + c->pz_target_nucleon) /
                                        (c->e_target_nucleon - c->pz_target_nucleon));
      const double dyt = y_target_nucleon - c->y_target_nucleus;
      const double dyp = y_projectile_nucleon - c->y_projectile_nucleus;
      if (((dyt < 0.0) ? -dyt : dyt) > 2 || ((dyp < 0.0) ? -dyp : dyp) > 2 ||
          y_projectile_nucleon < y_target_nucleon) {
        outer_success = false;
        continue;
      }
    }
  } while ((!outer_success) && ++loop_counter < max_number_of_loops);
  if (loop_counter >= max_number_of_loops) { return false; }
  return true;
}

/// G4FTFModel::AdjustNucleonsAlgorithm_afterSampling - the final kinematics.
__host__ __device__ inline void ftf_adjust_after_sampling(int interaction_case,
                                                          SplitableHadron* selected_antibaryon,
                                                          SplitableHadron* selected_target,
                                                          const AdjustResidual& proj_res,
                                                          const AdjustResidual& targ_res,
                                                          AdjustCommon* c) {
  if (interaction_case == 1) {
    c->pprojectile.v.z = c->pzprojectile;
    c->pprojectile.e = c->eprojectile;
  } else if (interaction_case == 2) {
    c->pprojectile.v.x = c->pt_nucleon.x;
    c->pprojectile.v.y = c->pt_nucleon.y;
    c->pprojectile.v.z = c->pz_projectile_nucleon;
    c->pprojectile.e = c->e_projectile_nucleon;
  } else {
    c->pprojectile.v.x = c->pt_nucleon_p.x;
    c->pprojectile.v.y = c->pt_nucleon_p.y;
    c->pprojectile.v.z = c->pz_projectile_nucleon;
    c->pprojectile.e = c->e_projectile_nucleon;
  }
  c->pprojectile = lorentz_apply(c->to_lab, c->pprojectile);
  selected_antibaryon->momentum = c->pprojectile;

  if (interaction_case == 1) {
    c->ptarget.v.x = c->pt_nucleon.x;
    c->ptarget.v.y = c->pt_nucleon.y;
    c->ptarget.v.z = c->pz_target_nucleon;
    c->ptarget.e = c->e_target_nucleon;
  } else if (interaction_case == 2) {
    c->ptarget.v.z = c->pztarget;
    c->ptarget.e = c->etarget;
  } else {
    c->ptarget.v.x = c->pt_nucleon_t.x;
    c->ptarget.v.y = c->pt_nucleon_t.y;
    c->ptarget.v.z = c->pz_target_nucleon;
    c->ptarget.e = c->e_target_nucleon;
  }
  c->ptarget = lorentz_apply(c->to_lab, c->ptarget);
  selected_target->momentum = c->ptarget;

  if (interaction_case == 1 || interaction_case == 3) {
    *targ_res.mass_number = c->t_residual_mass_number;
    *targ_res.charge = c->t_residual_charge;
    *targ_res.excitation = c->t_residual_excitation;
    if (*targ_res.mass_number != 0) {
      double mt2 = 0.0;
      Vec4 p;
      if (interaction_case == 1) {
        mt2 = c->t_residual_mass * c->t_residual_mass + g4gpu::mag2(c->pt_residual);
        p.v.x = c->pt_residual.x;
        p.v.y = c->pt_residual.y;
      } else {
        mt2 = c->t_residual_mass * c->t_residual_mass + g4gpu::mag2(c->pt_residual_t);
        p.v.x = c->pt_residual_t.x;
        p.v.y = c->pt_residual_t.y;
      }
      p.v.z = -c->wminus_target * c->xminus_residual / 2.0 +
              mt2 / (2.0 * c->wminus_target * c->xminus_residual);
      p.e = c->wminus_target * c->xminus_residual / 2.0 +
            mt2 / (2.0 * c->wminus_target * c->xminus_residual);
      *targ_res.momentum = lorentz_apply(c->to_lab, p);
    } else {
      *targ_res.momentum = Vec4(0.0, 0.0, 0.0, 0.0);
    }
  }

  if (interaction_case == 2 || interaction_case == 3) {
    if (interaction_case == 2) {
      *proj_res.mass_number = c->t_residual_mass_number;
      *proj_res.charge = c->t_residual_charge;
      *proj_res.excitation = c->t_residual_excitation;
      if (proj_res.lambda_number != nullptr) { *proj_res.lambda_number = c->p_residual_lambda; }
    } else {
      *proj_res.mass_number = c->p_residual_mass_number;
      *proj_res.charge = c->p_residual_charge;
      *proj_res.excitation = c->p_residual_excitation;
      if (proj_res.lambda_number != nullptr) { *proj_res.lambda_number = c->p_residual_lambda; }
    }
    if (*proj_res.mass_number != 0) {
      double mt2 = 0.0;
      Vec4 p;
      if (interaction_case == 2) {
        mt2 = c->t_residual_mass * c->t_residual_mass + g4gpu::mag2(c->pt_residual);
        p.v.x = c->pt_residual.x;
        p.v.y = c->pt_residual.y;
      } else {
        mt2 = c->p_residual_mass * c->p_residual_mass + g4gpu::mag2(c->pt_residual_p);
        p.v.x = c->pt_residual_p.x;
        p.v.y = c->pt_residual_p.y;
      }
      p.v.z = c->wplus_projectile * c->xplus_residual / 2.0 -
              mt2 / (2.0 * c->wplus_projectile * c->xplus_residual);
      p.e = c->wplus_projectile * c->xplus_residual / 2.0 +
            mt2 / (2.0 * c->wplus_projectile * c->xplus_residual);
      *proj_res.momentum = lorentz_apply(c->to_lab, p);
    } else {
      *proj_res.momentum = Vec4(0.0, 0.0, 0.0, 0.0);
    }
  }
}

/// G4FTFModel::AdjustNucleons - the entry point, including the three-way case selection.
///
/// `has_projectile_nucleus` is `GetProjectileNucleus() != nullptr`. The two "residual mass
/// number == 1" short-circuits below hand the WHOLE residual to the selected hadron and zero the
/// residual, which is how the last nucleon of a nucleus leaves it.
template <typename Rng>
__host__ __device__ inline bool ftf_adjust_nucleons(
    SplitableHadron* selected_antibaryon, const bic::Nucleon* proj_nucleon,
    SplitableHadron* selected_target_nucleon, const bic::Nucleon* targ_nucleon,
    bool annihilation, bool has_projectile_nucleus, const FtfParameters<double>* params,
    const AdjustResidual& proj_res, const AdjustResidual& targ_res, AdjustCommon* c, Rng& rng) {
  *c = AdjustCommon();

  if (selected_antibaryon->collision_count != 0 &&
      selected_target_nucleon->collision_count != 0) {
    return true;  // Selected hadrons were adjusted before.
  }

  int interaction_case = 0;
  if ((!has_projectile_nucleus && selected_antibaryon->collision_count == 0 &&
       selected_target_nucleon->collision_count == 0) ||
      (selected_antibaryon->collision_count != 0 &&
       selected_target_nucleon->collision_count == 0)) {
    interaction_case = 1;
    if (*targ_res.mass_number < 1) { return false; }
    if (lv_rapidity(selected_antibaryon->momentum) < lv_rapidity(*targ_res.momentum)) {
      return false;
    }
    if (*targ_res.mass_number == 1) {
      *targ_res.mass_number = 0;
      *targ_res.charge = 0;
      *targ_res.excitation = 0.0;
      selected_target_nucleon->momentum = *targ_res.momentum;
      *targ_res.momentum = Vec4(0.0, 0.0, 0.0, 0.0);
      return true;
    }
  } else if (selected_antibaryon->collision_count == 0 &&
             selected_target_nucleon->collision_count != 0) {
    interaction_case = 2;
    if (*proj_res.mass_number < 1) { return false; }
    if (lv_rapidity(*proj_res.momentum) <=
        lv_rapidity(selected_target_nucleon->momentum)) {
      return false;
    }
    if (*proj_res.mass_number == 1) {
      *proj_res.mass_number = 0;
      *proj_res.charge = 0;
      *proj_res.excitation = 0.0;
      selected_antibaryon->momentum = *proj_res.momentum;
      *proj_res.momentum = Vec4(0.0, 0.0, 0.0, 0.0);
      return true;
    }
  } else {
    interaction_case = 3;
    if (!has_projectile_nucleus) { return false; }
  }

  const int code = ftf_adjust_before_sampling(
      interaction_case, selected_antibaryon, proj_nucleon, selected_target_nucleon,
      targ_nucleon, annihilation, params, proj_res, targ_res, c, rng);
  if (code == 0) { return true; }
  if (code != 1) { return false; }
  if (!ftf_adjust_sampling(interaction_case, params, *proj_res.mass_number,
                           *targ_res.mass_number, c, rng)) {
    return false;
  }
  ftf_adjust_after_sampling(interaction_case, selected_antibaryon, selected_target_nucleon,
                            proj_res, targ_res, c);
  return true;
}

}  // namespace g4gpu::hadronic::ftf
