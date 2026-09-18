// G4MesonAbsorption and the three G4BCAction implementations the binary cascade schedules on.
//
// Transcribed from G4MesonAbsorption.{hh,cc}, G4BCAction.hh and G4BCDecay.hh (im_r_matrix) and
// G4BCLateParticle.hh (binary_cascade), 11.1.1.
//
// `G4BinaryCascade` holds three actions in `theImRActions` plus the scatterer, and asks each of
// them for candidate collisions on every track it propagates:
//
//     theImR.push_back(new G4MesonAbsorption);     // a pi+- swallowed by two nucleons
//     theImR.push_back(new G4BCDecay);             // a short-lived track decaying on its own
//     theImR.push_back(new G4BCLateParticle);      // a track waiting for its formation time
//
// The last two are eight lines each and are here in full. The first is the one with the physics
// in it, and it has three defects that the oracle is the only way to be sure of.
//
// ## THE CROSS-SECTION TABLE IS READ BEFORE ITS FIRST ELEMENT
//
// `AbsorptionCrossSection` decides whose kinetic energy to look the cross section up at:
//
//     if (aT is pi+ || aT is pi-)                  t = aT.E - aT.mag()/MeV;
//     else if (bT is pi+ || bT is NOT pi-)         t = bT.E - bT.mag()/MeV;
//
// The second comparison in the `else if` is `!=` where every reading of the intent says `==`.
// The consequence is not that it fires too often - it is that it does not fire at all for the one
// case it was written for. `GetTimeToAbsorption` has already established that ONE of the two
// tracks is a charged pion, so if `aT` is not one then `bT` is; and if that `bT` is a **pi-**,
// the first disjunct is false and the second is `pi- != pi-`, also false. **`t` keeps its
// initialiser of zero.**
//
// Then:
//
//     if (t<=it[24]) { G4int count = 0; while (t>it[count]) count+=2;
//                      x1 = it[count-2]; y1 = it[count-1]; x2 = it[count]; y2 = it[count+1]; ... }
//
// `it[0]` is 0, so `t > it[0]` is false, `count` stays 0, and **`it[-2]` and `it[-1]` are read**:
// two doubles from whatever precedes the static array. The arithmetic that follows is
// `y1 + (y2-y1)/(x2-x1)*(t-x1)` with `t == x2 == 0`, which is `y1 + (y2-y1)*(-x1)/(-x1)` = `y2`
// exactly - **the out-of-bounds values cancel** - for any finite `x1 != 0`. The port returns `y2`,
// which is `it[1] = 4` mb before the 0.5, and `absorption_read_before_table` says it happened.
// docs/RISK.md V153. An `x1` of zero, or a byte pattern that is a NaN, would not cancel, and the
// port cannot reproduce that because there is nothing before the array to read.
//
// The same `count == 0` exit is reached honestly by a charged pion AT REST, where `t` really is
// zero. It is the same read.
//
// ## `/MeV` ON A MASS INSIDE A SUBTRACTION
//
// `t = aT.Get4Momentum().t() - aT.Get4Momentum().mag()/MeV` divides one of the two terms by a
// unit and not the other. It is only right because Geant4's internal energy unit IS the MeV, so
// the division is by 1.0; in a build with a different unit basis the kinetic energy would be a
// nonsense. Transcribed as `e - mag`, with this note, because that is the value it computes.
//
// ## THE CLUSTER PARTNER IS THE ONE NEAREST THE ORIGIN, NOT THE ONE NEAREST THE TARGET
//
// `FindAndFillCluster` looks for the second nucleon of the absorbing pair by
//
//     G4ThreeVector firstBase = aTarget->GetPosition();
//     ...
//     G4ThreeVector secodeBase = (*j)->GetPosition();
//     if((firstBase+secodeBase).mag()<min) { min=(firstBase+secodeBase).mag(); partner = *j; }
//
// - a SUM of two position vectors, where a distance is a difference. What it minimises is
// `|r1 + r2|`, which is smallest for a candidate diametrically opposite the first target through
// the nucleus's centre, i.e. **the farthest one in the geometry it was meant to pick the nearest
// in**. The misspelt `secodeBase` suggests how long this has gone unread. docs/RISK.md V154.
// Reproduced.
//
// ## REFUSED, by name
//
//   * `G4MesonAbsorption::GetCollision(projectile, targets)`. It is declared public in the header
//     and **never defined and never called** anywhere in 11.1.1; there is no body to transcribe.
#ifndef G4GPU_BIC_IMR_ABSORPTION_CUH
#define G4GPU_BIC_IMR_ABSORPTION_CUH

#include <cfloat>
#include <cmath>

#include "physics/hadronic/bic/im_r/decay.cuh"
#include "physics/hadronic/bic/im_r/scatterer.cuh"

namespace g4gpu::bic::imr {

/// What the absorption could not do, and what it noticed.
struct AbsorptionRefusal {
  bool read_before_table = false;  ///< the `it[-2]` branch; see the file header
  bool negative_mass2 = false;     ///< `mom1.mag2() < -1*eV`, which Geant4 prints and continues
  bool no_partner = false;         ///< FindAndFillCluster found no second nucleon
  __host__ __device__ bool any() const { return false; }  // none of these stops the cascade
};

/// `G4MesonAbsorption::AbsorptionCrossSection`'s table: thirteen (kinetic energy, cross section)
/// pairs interleaved, the energies in MeV and the cross sections in millibarn BEFORE the 0.5.
__host__ __device__ inline const double* absorption_table() {
  static const double it[26] = {0,   4,  50,  5.5, 75,  8,   95,  10,  120, 11.5, 140, 12,  160,
                                11.5, 180, 10, 190, 8,   210, 6,   235, 4,   260, 3,    300, 2};
  return it;
}

/// `G4MesonAbsorption::AbsorptionCrossSection(aT, bT)`.
///
/// `is_pi_charged` is `definition == pi+ || definition == pi-`; `is_pi_minus` is the second
/// comparison on its own, because the `!=` in the source turns it into its own branch. `e` and
/// `mag` are `Get4Momentum().t()` and `Get4Momentum().mag()` - the ACTUAL mass, not the PDG one.
__host__ __device__ inline double absorption_cross_section(bool a_is_pi_plus, bool a_is_pi_minus,
                                                           bool b_is_pi_plus, bool b_is_pi_minus,
                                                           double e_a, double mag_a, double e_b,
                                                           double mag_b,
                                                           AbsorptionRefusal& ref) {
  double t = 0.0;
  if (a_is_pi_plus || a_is_pi_minus) {
    t = e_a - mag_a;
  } else if (b_is_pi_plus || !b_is_pi_minus) {
    // The `!=` of the source. For a pi- in slot b this is false and `t` stays zero.
    t = e_b - mag_b;
  }
  const double* it = absorption_table();
  double a_cross = 0.0;
  if (t <= it[24]) {
    int count = 0;
    while (t > it[count]) { count += 2; }
    if (count == 0) {
      // `it[-2]` and `it[-1]`. The interpolation below is `y1 + (y2-y1)*(t-x1)/(x2-x1)` with
      // `t == x2 == 0`, so it collapses to `y2` for any finite `x1 != 0` - which is what this
      // returns. See the file header and docs/RISK.md V153.
      ref.read_before_table = true;
      a_cross = it[1];
    } else {
      const double x1 = it[count - 2];
      const double x2 = it[count];
      const double y1 = it[count - 1];
      const double y2 = it[count + 1];
      a_cross = y1 + (y2 - y1) / (x2 - x1) * (t - x1);
    }
  }
  return 0.5 * a_cross * millibarn();
}

/// `G4MesonAbsorption::GetTimeToAbsorption`.
///
/// The same three optimisation gates as `G4Scatterer::GetTimeToInteraction` - 500 mb, 200 mb for
/// two charged particles, 200 mb for a neutron above sqrt(s) = 1.91 GeV - but NOT the same
/// geometry: this one has no z-aligned fast path and no `0.7*pi*distanceFast` pre-gate, so it
/// computes the CM impact parameter for every pair it is asked about.
///
/// The entrance condition is that ONE of the two tracks is a charged pion. A pi0 is not one, so a
/// neutral pion is never absorbed.
__host__ __device__ inline double time_to_absorption(
    int pdg1, int pdg2, int charge1, int charge2, const Vec3d& pos1, const Vec3d& pos2,
    const LorentzVector& tracking1, const LorentzVector& p1, const LorentzVector& p2,
    double actual1, double actual2, AbsorptionRefusal& ref) {
  const bool a_pip = (pdg1 == kPdgPiPlus);
  const bool a_pim = (pdg1 == kPdgPiMinus);
  const bool b_pip = (pdg2 == kPdgPiPlus);
  const bool b_pim = (pdg2 == kPdgPiMinus);
  if (!a_pip && !a_pim && !b_pip && !b_pim) { return DBL_MAX; }
  double time = DBL_MAX;
  const double sqrt_s = (p1 + p2).mag();
  if (!(actual1 + actual2 < sqrt_s)) { return time; }

  LorentzVector mom1 = tracking1;
  // `mom1.mag2() < -1.*eV` compares a squared mass against an ENERGY, so the threshold is
  // 1e-6 MeV^2 and not 1e-12; Geant4 prints and carries on, and so does this.
  const double mom1_mag2 = mom1.e * mom1.e - g4gpu::mag2(mom1.v);
  if (mom1_mag2 < -1.0e-6) { ref.negative_mass2 = true; }
  const Vec3d position = pos1 - pos2;
  const Vec3d velocity = (1.0 / mom1.e) * mom1.v * u::c_light<double>();
  const double collision_time = -g4gpu::dot(position, velocity) / g4gpu::mag2(velocity);
  if (!(collision_time > 0.0)) { return time; }

  LorentzVector mom2(Vec3d{0.0, 0.0, 0.0}, p2.mag());
  const LorentzRotation to_cms = LorentzRotation::from_boost(-1.0 * (mom1 + mom2).boost_vector());
  mom1 = to_cms * mom1;
  mom2 = to_cms * mom2;
  const LorentzVector coordinate1(pos1, 100.0);
  const LorentzVector coordinate2(pos2, 100.0);
  const Vec3d pos = (to_cms * coordinate1).v - (to_cms * coordinate2).v;
  const Vec3d mom = mom1.v - mom2.v;
  const double dot_pm = g4gpu::dot(pos, mom);
  const double distance = g4gpu::dot(pos, pos) - dot_pm * dot_pm / g4gpu::mag2(mom);

  if (u::pi<double>() * distance > max_cross_section()) { return time; }
  if (charge1 != 0 && charge2 != 0 &&
      u::pi<double>() * distance > max_charged_cross_section()) {
    return time;
  }
  if ((pdg1 == kPdgNeutron || pdg2 == kPdgNeutron) && sqrt_s > neutron_special_sqrt_s() &&
      u::pi<double>() * distance > max_charged_cross_section()) {
    return time;
  }
  const double total = absorption_cross_section(a_pip, a_pim, b_pip, b_pim, p1.e, p1.mag(), p2.e,
                                                p2.mag(), ref);
  if (total > 0.0 && distance <= total / u::pi<double>()) { time = collision_time; }
  return time;
}

/// `G4MesonAbsorption::FindAndFillCluster` - the second nucleon of the absorbing pair.
///
/// `first` is the candidate already chosen; the return is the index into `candidates` of the
/// partner, or -1. The charge filter is on the SUM of the projectile's, the first target's and
/// the candidate's charges, which must land in [0, 2]; the selection minimises `|r1 + r2|`, which
/// is a sum where a distance would be a difference. See the file header.
__host__ __device__ inline int find_and_fill_cluster(int projectile_charge, int first_charge,
                                                     const Vec3d& first_pos, const int* charges,
                                                     const Vec3d* positions, int n_candidates,
                                                     int first_index) {
  const int charge_sum = first_charge + projectile_charge;
  double min = DBL_MAX;
  int partner = -1;
  for (int j = 0; j < n_candidates; ++j) {
    if (j == first_index) { continue; }
    const int c = charges[j];
    if (charge_sum + c > 2) { continue; }
    if (charge_sum + c < 0) { continue; }
    const Vec3d sum = first_pos + positions[j];
    const double d = std::sqrt(g4gpu::mag2(sum));
    if (d < min) {
      min = d;
      partner = j;
    }
  }
  return partner;
}

/// The two outgoing nucleons of an absorption.
struct AbsorptionFinalState {
  int pdg1 = 0;
  int pdg2 = 0;
  LorentzVector p1;
  LorentzVector p2;
};

/// `G4MesonAbsorption::GetFinalState`.
///
/// Two boosts and two rotations, and FOUR possible uniforms before the angles: the identity swap
/// always draws one, and the charge fixing draws a second only when both nucleons are on the same
/// side of the pion's charge. The angles are `costh = 2u-1` and `phi = 2 pi u`, and the
/// polar angle is taken as `sin(acos(costh))` and not as `sqrt(1-costh^2)`. The two are the same
/// function and differ only in their rounding; MEASURED, over the eight prescribed cosines and the
/// 1,728 compared components, they agree to within 5e-15 and the substitution passes. It is
/// transcribed as written because the agreement is a property of these eight values and not of the
/// two expressions - `acos` near +-1 loses half the significant digits, and a cosine drawn from a
/// real engine will land there.
///
/// The rotation `toZ` is built from the projectile's LAB four-momentum, not from the boosted one.
/// That is what the source does - `G4LorentzVector Ptmp=projectile->Get4Momentum()` is read again
/// after `thePro` has already been boosted - so the "rotate to z" is a rotation to the lab z of
/// the pion, applied to vectors that are in the centre-of-momentum frame.
template <typename Rng>
__host__ __device__ inline AbsorptionFinalState meson_absorption_final_state(
    int pdg_pro, double charge_pro, const LorentzVector& pro, const LorentzVector& t1,
    const LorentzVector& t2, int pdg_t1, int pdg_t2, double m_proton, double m_neutron,
    Rng& rng) {
  AbsorptionFinalState out;
  LorentzVector the_pro = pro;
  LorentzVector the_t1 = t1;
  LorentzVector the_t2 = t2;
  const LorentzRotation to_sps =
      LorentzRotation::from_boost(-1.0 * (the_pro + the_t1 + the_t2).boost_vector());
  the_t1 = to_sps * the_t1;
  the_t2 = to_sps * the_t2;
  the_pro = to_sps * the_pro;
  const LorentzRotation from_sps = to_sps.inverse();

  LorentzRotation to_z;
  to_z.rotate_z(-1.0 * lv_phi(pro));
  to_z.rotate_y(-1.0 * lv_theta(pro));
  the_t1 = to_z * the_t1;
  the_t2 = to_z * the_t2;
  the_pro = to_z * the_pro;
  const LorentzRotation to_lab = to_z.inverse();

  int d1 = pdg_t1;
  int d2 = pdg_t2;
  if (0.5 > rng.uniform()) {
    const int tmp = d1;
    d1 = d2;
    d2 = tmp;
  }
  const double q1 = (d1 == kPdgProton) ? 1.0 : 0.0;
  const double q2 = (d2 == kPdgProton) ? 1.0 : 0.0;
  if (charge_pro < -0.5) {
    if (q1 > 0.5) {
      if (q2 > 0.5 && 0.5 > rng.uniform()) {
        d2 = kPdgNeutron;
      } else {
        d1 = kPdgNeutron;
      }
    } else if (q2 > 0.5) {
      d2 = kPdgNeutron;
    }
  } else if (charge_pro > 0.5) {
    if (q1 < 0.5) {
      if (q2 < 0.5 && 0.5 > rng.uniform()) {
        d2 = kPdgProton;
      } else {
        d1 = kPdgProton;
      }
    } else if (q2 < 0.5) {
      d2 = kPdgProton;
    }
  }

  const LorentzVector total = the_pro + the_t1 + the_t2;
  const double m_sq_total = total.e * total.e - g4gpu::mag2(total.v);
  const double m1 = (d1 == kPdgProton) ? m_proton : m_neutron;
  const double m2 = (d2 == kPdgProton) ? m_proton : m_neutron;
  const double m1_sq = m1 * m1;
  const double m2_sq = m2 * m2;
  const double m_sq = m_sq_total - m1_sq - m2_sq;
  const double p = std::sqrt((m_sq * m_sq - 4.0 * m1_sq * m2_sq) / (4.0 * m_sq_total));
  const double costh = 2.0 * rng.uniform() - 1.0;
  const double phi = 2.0 * u::pi<double>() * rng.uniform();
  const double sinth = std::sin(std::acos(costh));
  const Vec3d p_final{p * sinth * std::cos(phi), p * sinth * std::sin(phi), p * costh};
  const double p2mag = g4gpu::mag2(p_final);
  LorentzVector final1(p_final, std::sqrt(m1_sq + p2mag));
  LorentzVector final2(-1.0 * p_final, std::sqrt(m2_sq + p2mag));
  final1 = to_lab * final1;
  final2 = to_lab * final2;
  final1 = from_sps * final1;
  final2 = from_sps * final2;
  (void)pdg_pro;
  out.pdg1 = d1;
  out.pdg2 = d2;
  out.p1 = final1;
  out.p2 = final2;
  return out;
}

/// `G4BCDecay::GetCollisions` - a short-lived track schedules its own decay at
/// `theCurrentTime + SampleResidualLifetime()`, with no target. Anything else schedules nothing.
///
/// Returns the time, or DBL_MAX when the track is not short-lived. One uniform when it is.
template <typename Rng>
__host__ __device__ inline double bc_decay_collision_time(bool is_short_lived,
                                                          double current_time,
                                                          double total_actual_width,
                                                          double lorentz_gamma, Rng& rng) {
  if (!is_short_lived) { return DBL_MAX; }
  return current_time + sample_residual_lifetime(total_actual_width, lorentz_gamma, rng);
}

/// `G4BCLateParticle::GetCollisions` - EVERY track schedules one, at
/// `max(0, formationTime) + theCurrentTime`, and its final state is a copy of the projectile.
///
/// The `std::max` is against zero and not against the current time, so a track whose formation
/// time has already passed is scheduled at the current time and comes straight back out.
__host__ __device__ inline double bc_late_particle_time(double formation_time,
                                                        double current_time) {
  const double f = (formation_time > 0.0) ? formation_time : 0.0;
  return f + current_time;
}

}  // namespace g4gpu::bic::imr

#endif
