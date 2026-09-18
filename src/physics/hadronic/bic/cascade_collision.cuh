// The three tests G4BinaryCascade::ApplyCollision puts a candidate final state through.
//
// Transcribed from G4BinaryCascade.cc (models/binary_cascade, 11.1.1): `CheckPauliPrinciple`,
// `CorrectShortlivedPrimaryForFermi` and `CorrectShortlivedFinalsForFermi`. A collision that
// fails any of them is thrown away and its `G4CollisionInitialState` removed, so between them
// they decide which of the collisions the scatterer proposes actually happen.
//
// ## THE PAULI BLOCK IS AGAINST THE ORIGINAL NUCLEUS, NOT THE CURRENT ONE
//
//     G4int A = the3DNucleus->GetMassNumber();
//     G4int Z = the3DNucleus->GetCharge();
//     G4FermiMomentum fermiMom;  fermiMom.Init(A, Z);
//
// `A` and `Z` here are the nucleus as it was BUILT, not `currentA` and `currentZ`. A cascade that
// has already knocked twenty nucleons out of a lead nucleus still blocks against lead's Fermi
// momentum. Reproduced; the caller passes the initial pair and this file does not see the current
// one.
//
// The energy compared against is `sqrt(pFermi(rho(x))^2 + p4.mag2())` - the Fermi momentum at the
// product's own position combined with the product's ACTUAL mass, not its PDG mass - less the
// nucleus's Coulomb barrier for a proton. And the comparison is on the total energy `mom.e()`,
// which for a track inside the nucleus already carries the field.
//
// **It does not return early.** Geant4's `return false` inside the loop is commented out and
// replaced by a flag, so every product is examined even after one has already blocked the
// collision. Nothing observable depends on it - the flag is the same either way - but the port
// keeps the shape, because the commented-out line is exactly the sort of thing a release restores.
//
// ## A RESONANCE IS GIVEN THE NEUTRON'S FIELD, AND THE ENERGY IS SHARED BACK OUT AFTERWARDS
//
// `CorrectShortlivedPrimaryForFermi` computes the total nuclear field the collision starts in:
// the primary's own field plus every target's. For a primary that is a RESONANCE - baryon number
// non-zero, PDG code above 1000 in absolute value and not a nucleon - it uses the NEUTRON's field
// instead of the resonance's own (the comment says "subtract neutron ( = proton) field") and
// takes that field back off the primary's energy before the collision is generated.
//
// `CorrectShortlivedFinalsForFermi` then sums the field over the products, splits the difference
// equally between the products that are themselves resonances, and moves each one's energy by its
// share at fixed invariant mass. If any of them would go below its own mass the whole collision
// is vetoed. With no resonance among the products the difference is simply DISCARDED - there is
// nothing to share it with - so energy conservation across a collision is only as good as the
// field difference between the entrance and exit species.
//
// ## REFUSED, by name
//
//   * nothing. All three are here in full.
#ifndef G4GPU_BIC_CASCADE_COLLISION_CUH
#define G4GPU_BIC_CASCADE_COLLISION_CUH

#include <cmath>

#include "physics/hadronic/bic/cascade_state.cuh"
#include "physics/hadronic/bic/rk_propagation.cuh"

namespace g4gpu::bic {

/// Whether a PDG code is a short-lived BARYON for the two Fermi corrections:
/// `std::abs(PDGcode) > 1000 && PDGcode != 2112 && PDGcode != 2212`.
///
/// Note what that test does and does not catch. It is on the CODE and not on `IsShortLived`, so
/// an anti-nucleon (-2112, -2212) IS one of these - `abs` puts it above 1000 and the two
/// exclusions are the positive codes only - and a pion (211) is not, which is right. A lambda
/// (3122) is also one, and would be given the neutron's field.
__host__ __device__ inline bool is_shortlived_for_fermi(int pdg) {
  const int a = (pdg < 0) ? -pdg : pdg;
  return a > 1000 && pdg != 2112 && pdg != 2212;
}

/// `G4BinaryCascade::CheckPauliPrinciple`.
///
/// `initial_a` and `initial_z` are the nucleus as built - see the file header. Returns true when
/// the collision is allowed.
__host__ __device__ inline bool check_pauli_principle(const CascadeTrack* products, int n,
                                                      int initial_a, int initial_z,
                                                      const NuclearDensity& density,
                                                      double coulomb_barrier) {
  FermiMomentum fermi;
  fermi.init(initial_a, initial_z);
  bool allowed = true;
  for (int i = 0; i < n; ++i) {
    const CascadeTrack& p = products[i];
    if (p.pdg != imr::kPdgProton && p.pdg != imr::kPdgNeutron) { continue; }
    const double d = density.density(p.position);
    // `Get4Momentum().mag2()` is the ACTUAL mass squared and it is SIGNED - a product that is
    // spacelike by rounding contributes a negative term here rather than a NaN.
    const double m2 = p.momentum.e * p.momentum.e - g4gpu::mag2(p.momentum.v);
    const double pf = fermi.fermi_momentum(d);
    double e_fermi = std::sqrt(pf * pf + m2);
    if (p.pdg == imr::kPdgProton) { e_fermi -= coulomb_barrier; }
    if (p.momentum.e < e_fermi) {
      // Geant4's `return false` is commented out here and replaced by this flag, so the loop
      // runs to the end. See the file header.
      allowed = false;
    }
  }
  return allowed;
}

/// `G4BinaryCascade::CorrectShortlivedPrimaryForFermi`.
///
/// Returns the total entrance field, and moves the primary's energy down by the neutron field
/// when the primary is a short-lived baryon. `primary` is updated in place, as Geant4 updates the
/// track; `Update4Momentum(E)` keeps the mass and rebuilds the momentum, which is why the caller
/// has to save the four-momentum first and put it back if the collision is vetoed.
template <typename Prop>
__host__ __device__ inline double correct_shortlived_primary_for_fermi(
    CascadeTrack& primary, const CascadeTrack* targets, int n_targets, const Prop& propagator) {
  if (primary.state != kInside) { return 0.0; }
  double e_fermi = propagator.field(primary.pdg, primary.position);
  if (is_shortlived_for_fermi(primary.pdg)) {
    e_fermi = propagator.field(imr::kPdgNeutron, primary.position);
    // `Update4Momentum(e)`: the MASS is kept and the momentum rebuilt from the new energy.
    const double mass = primary.actual_mass();
    const double new_e = primary.momentum.e - e_fermi;
    const double p2 = new_e * new_e - mass * mass;
    const deex::Vec3d dir = g4gpu::normalize(primary.momentum.v);
    primary.momentum = imr::LorentzVector(dir * ((p2 > 0.0) ? std::sqrt(p2) : 0.0), new_e);
  }
  for (int i = 0; i < n_targets; ++i) {
    e_fermi += propagator.field(targets[i].pdg, targets[i].position);
  }
  return e_fermi;
}

/// `G4BinaryCascade::CorrectShortlivedFinalsForFermi`.
///
/// Returns false to veto the collision. The share-out is by COUNT of resonances among the
/// products and not by energy, and a collision with no resonance among its products simply loses
/// the difference.
template <typename Prop>
__host__ __device__ inline bool correct_shortlived_finals_for_fermi(CascadeTrack* products, int n,
                                                                    double initial_e_fermi,
                                                                    const Prop& propagator) {
  double final_e_fermi = 0.0;
  int n_res = 0;
  for (int i = 0; i < n; ++i) {
    final_e_fermi += propagator.field(products[i].pdg, products[i].position);
    if (is_shortlived_for_fermi(products[i].pdg)) { ++n_res; }
  }
  if (n_res == 0) { return true; }
  const double delta = (initial_e_fermi - final_e_fermi) / static_cast<double>(n_res);
  for (int i = 0; i < n; ++i) {
    if (!is_shortlived_for_fermi(products[i].pdg)) { continue; }
    const imr::LorentzVector& mom = products[i].momentum;
    const double mass2 = mom.e * mom.e - g4gpu::mag2(mom.v);
    const double new_e = mom.e + delta;
    const double new_e2 = new_e * new_e;
    if (new_e2 < mass2) { return false; }
    const deex::Vec3d dir = g4gpu::normalize(mom.v);
    products[i].momentum = imr::LorentzVector(std::sqrt(new_e2 - mass2) * dir, new_e);
  }
  return true;
}

}  // namespace g4gpu::bic

#endif
