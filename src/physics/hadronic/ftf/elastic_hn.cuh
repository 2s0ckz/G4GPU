// G4ElasticHNScattering::ElasticScattering - the hadron-nucleon elastic channel inside FTF.
//
// Transcribed from
//   source/processes/hadronic/models/parton_string/diffraction/src/G4ElasticHNScattering.cc
//     ElasticScattering, GaussianPt
//
// THIS IS NOT `G4HadronElastic`. It is the elastic branch of one nucleon-nucleon collision
// INSIDE a string model: it transfers a Gaussian-sampled transverse momentum between two
// splitable hadrons that are already off their PDG mass shells, keeps each one's INVARIANT mass
// (`M0projectile = Pprojectile.mag()`, the current mass, not the definition's), and leaves both
// participants with `GetSoftCollisionCount()` incremented so that `BuildStrings` builds a
// string for each. QBBC's elastic process is a different model on a different track.
//
// THE COLLISION COUNTS ARE INCREMENTED FIRST, BEFORE ANY TEST THAT CAN RETURN FALSE.
// `projectile->IncrementCollisionCount(1); target->IncrementCollisionCount(1);` are the first
// two statements, and there are four `return false` paths after them - a backward projectile, a
// sub-threshold sqrt(s), a Pt loop that exhausted 1000 attempts. So a FAILED elastic scattering
// still counts as a soft collision on both hadrons, which is what makes BuildStrings treat them
// as "took part in a diffractive interaction" (status 1, SoftCollisionCount != 0) rather than
// as spectators. The two commented-out increments at the end of the method are where they used
// to be. Transcribed in the order the source has them, because the order is the behaviour.
//
// THE `PZcms2 < 0` CLAMP IS NOT THE SAME AS THE EXCITATION'S. Here a negative PZcms2 after the
// Pt sampling is set to zero "to avoid the exactness problem" and the scattering proceeds; in
// G4DiffractiveExcitation the same quantity going negative aborts the interaction. The
// difference is that this one has already passed `SqrtS >= ProjMassT + TargMassT`, so a
// negative value can only be rounding.
#pragma once
#include <cmath>

#include "core/units.cuh"
#include "physics/hadronic/ftf/ftf_parameters.cuh"
#include "physics/hadronic/ftf/lorentz.cuh"
#include "physics/hadronic/ftf/splitable_hadron.cuh"

namespace g4gpu::hadronic::ftf {

/// HepLorentzVector::rapidity() - `0.5*log((E+pz)/(E-pz))`, with CLHEP's spacelike guard, which
/// prints to cerr and returns 0 rather than producing a NaN. A kernel cannot print; the zero is
/// what the callers see either way, and `spacelike` records that it happened.
__host__ __device__ inline double lv_rapidity(const Vec4& p, bool* spacelike = nullptr) {
  const double z1 = p.v.z;
  if (std::fabs(p.e) < std::fabs(z1)) {
    if (spacelike != nullptr) { *spacelike = true; }
    return 0.0;
  }
  const double q = (p.e + z1) / (p.e - z1);
  return 0.5 * std::log(q);
}

/// HepLorentzVector::mt2() = `E^2 - pz^2`. Used by CreateStrings' no-kink arm.
__host__ __device__ inline double lv_mt2(const Vec4& p) { return p.e * p.e - p.v.z * p.v.z; }

/// G4ElasticHNScattering::GaussianPt and G4DiffractiveExcitation::GaussianPt - the same eight
/// lines in two classes, both differing from G4FTFModel::GaussianPt (which has the `ymax < 200`
/// split and the `AveragePt2 > 0` test the other way round).
///
/// `-<pt2> log(1 + u (exp(-maxPt2/<pt2>) - 1))` is the exponential in pt^2 truncated at
/// maxPtSquare. The `AveragePt2 <= 0` arm gives Pt = 0 and STILL spends the phi deviate, which
/// is why the draw count distinguishes it from not being called at all.
///
/// G4ElasticHNScattering's version guards the square root with `Pt2 > 0.0 ? sqrt : 0.0` and
/// G4DiffractiveExcitation's does not; for Pt2 exactly 0 the two agree, and Pt2 cannot be
/// negative here. Both spellings are kept, one per function, rather than shared.
template <typename Rng>
__host__ __device__ inline Vec3d ftf_gaussian_pt_elastic(double average_pt2, double max_pt_square,
                                                         Rng& rng) {
  double pt2 = 0.0;
  if (average_pt2 <= 0.0) {
    pt2 = 0.0;
  } else {
    pt2 = -average_pt2 *
          std::log(1.0 + rng.uniform() * (std::exp(-max_pt_square / average_pt2) - 1.0));
  }
  const double pt = (pt2 > 0.0) ? std::sqrt(pt2) : 0.0;
  const double phi = rng.uniform() * units::twopi<double>();
  return Vec3d{pt * std::cos(phi), pt * std::sin(phi), 0.0};
}

/// G4DiffractiveExcitation::GaussianPt - the same formula without the `Pt2 > 0` guard on the
/// square root.
template <typename Rng>
__host__ __device__ inline Vec3d ftf_gaussian_pt_excitation(double average_pt2,
                                                            double max_pt_square, Rng& rng) {
  double pt2 = 0.0;
  if (average_pt2 <= 0.0) {
    pt2 = 0.0;
  } else {
    pt2 = -average_pt2 *
          std::log(1.0 + rng.uniform() * (std::exp(-max_pt_square / average_pt2) - 1.0));
  }
  const double pt = std::sqrt(pt2);
  const double phi = rng.uniform() * units::twopi<double>();
  return Vec3d{pt * std::cos(phi), pt * std::sin(phi), 0.0};
}

/// G4ElasticHNScattering::ElasticScattering.
template <typename Rng>
__host__ __device__ inline bool ftf_elastic_scattering(SplitableHadron* projectile,
                                                       SplitableHadron* target,
                                                       const FtfParameters<double>* params,
                                                       Rng& rng) {
  projectile->collision_count += 1;
  target->collision_count += 1;

  if (projectile->momentum.v.z < 0.0) { return false; }

  Vec4 p_projectile = projectile->momentum;
  const double m0_projectile = p_projectile.mag();
  const double m0_projectile2 = m0_projectile * m0_projectile;

  Vec4 p_target = target->momentum;
  const double m0_target = p_target.mag();
  const double m0_target2 = m0_target * m0_target;

  const double average_pt2 = params->avarage_pt2_of_elastic_scattering;

  const Vec4 psum = p_projectile + p_target;
  const Vec3d bv = psum.boost_vector();
  LorentzRot to_cms = lorentz_boost(Vec3d{-bv.x, -bv.y, -bv.z});
  Vec4 ptmp = lorentz_apply(to_cms, p_projectile);
  if (ptmp.v.z <= 0.0) { return false; }  // "String" moving backwards in CMS, abort collision
  lorentz_rotate_z(&to_cms, -lv_phi(ptmp));
  lorentz_rotate_y(&to_cms, -lv_theta(ptmp));
  const LorentzRot to_lab = lorentz_inverse(to_cms);
  p_projectile = lorentz_apply(to_cms, p_projectile);
  p_target = lorentz_apply(to_cms, p_target);

  const double s = psum.e * psum.e - g4gpu::mag2(psum.v);
  const double sqrt_s = std::sqrt(s);
  if (sqrt_s < m0_projectile + m0_target) { return false; }

  double pz_cms2 = (s * s + m0_projectile2 * m0_projectile2 + m0_target2 * m0_target2 -
                    2.0 * s * m0_projectile2 - 2.0 * s * m0_target2 -
                    2.0 * m0_projectile2 * m0_target2) /
                   4.0 / s;
  double pz_cms = (pz_cms2 > 0.0) ? std::sqrt(pz_cms2) : 0.0;
  const double max_pt_square = pz_cms2;

  double pt2 = 0.0, proj_mass_t2 = 0.0, proj_mass_t = 0.0, targ_mass_t2 = 0.0, targ_mass_t = 0.0;
  Vec4 q_momentum;

  const int max_number_of_loops = 1000;
  int loop_counter = 0;
  do {
    q_momentum = Vec4(ftf_gaussian_pt_elastic(average_pt2, max_pt_square, rng), 0.0);
    pt2 = g4gpu::mag2(q_momentum.v);
    proj_mass_t2 = m0_projectile2 + pt2;
    proj_mass_t = std::sqrt(proj_mass_t2);
    targ_mass_t2 = m0_target2 + pt2;
    targ_mass_t = std::sqrt(targ_mass_t2);
  } while ((sqrt_s < proj_mass_t + targ_mass_t) && ++loop_counter < max_number_of_loops);
  if (loop_counter >= max_number_of_loops) { return false; }

  pz_cms2 = (s * s + proj_mass_t2 * proj_mass_t2 + targ_mass_t2 * targ_mass_t2 -
             2.0 * s * proj_mass_t2 - 2.0 * s * targ_mass_t2 -
             2.0 * proj_mass_t2 * targ_mass_t2) /
            4.0 / s;
  if (pz_cms2 < 0.0) { pz_cms2 = 0.0; }  // to avoid the exactness problem
  pz_cms = std::sqrt(pz_cms2);
  p_projectile.v.z = pz_cms;
  p_target.v.z = -pz_cms;
  p_projectile = p_projectile + q_momentum;
  p_target = p_target - q_momentum;

  p_projectile = lorentz_apply(to_lab, p_projectile);
  p_target = lorentz_apply(to_lab, p_target);

  // The projectile inherits the target's creation time and position: the target nucleon's were
  // set by ReggeonCascade / ShiftInteractionTime and are the point the collision happened at.
  projectile->time_of_creation = target->time_of_creation;
  projectile->position = target->position;

  projectile->momentum = p_projectile;
  target->momentum = p_target;
  return true;
}

}  // namespace g4gpu::hadronic::ftf
