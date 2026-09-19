// G4BinaryCascade::Absorb and Capture - the two ways a cascade particle stops being one.
//
// Transcribed from G4BinaryCascade.cc and G4Absorber.cc (models/binary_cascade, 11.1.1).
// `Propagate` calls both at the top of every turn of its collision loop and again after it.
//
// ## `Absorb()` CANNOT ABSORB ANYTHING, AND THAT IS THE CONSTANT'S DOING
//
// `G4Absorber::WillBeAbsorbed` is
//
//     if (kt.Get4Momentum().e() - kt.GetActualMass() < theCutOnP)
//     { if (pi+ || pi0 || pi-) return true; }
//     return false;
//
// - a pion whose KINETIC ENERGY is below the threshold. `G4BinaryCascade`'s constructor sets that
// threshold to `theCutOnPAbsorb = 0*MeV`, with the comment "No Absorption of slow Mesons, other
// than above" beside it. A kinetic energy is never below zero, so the predicate is false for
// every track that has ever existed: `absorbList` is always empty, `Absorb()` always returns
// false, and `G4Absorber::FindAbsorbers`, `FindProducts`, `Absorb` and `GetProducts` are
// unreachable. The whole class is dead code in 11.1.1.
//
// `tools/extract_bic_constants.pl` pins the zero in the physics source, `bic_params.cuh` carries
// it, and `tests/test_bic_imr.cu` asserts that no pion between 0 and 2 GeV passes the predicate.
// **REFUSED, by name: `G4Absorber`** - all four of its functions - because reaching them needs a
// release that changes the constant, and a port that guessed at what they do would be inventing
// physics no run can ask for.
//
// ## `Capture()` IS ALL OR NOTHING, AND IT IS DECIDED BY A MEAN
//
// The gate Geant4 actually evaluates is
//
//     if(particlesBelowCut>0 && capturedEnergy/particlesBelowCut<0.2*theCutOnP)
//
// and the line above it, `if(particlesAboveCut==0 && ...)`, is commented out - as is the
// `if(energy < theCutOnP)` that would have made `particlesAboveCut` non-zero. So:
//
//   * `particlesBelowCut` counts EVERY proton and neutron inside the nucleus, whatever its
//     energy, and `capturedEnergy` sums their energies whatever their energy;
//   * the test is on the MEAN energy of those nucleons against `0.2*theCutOnP` = 9 MeV, since
//     `theCutOnP` is always 45 (docs/RISK.md V72);
//   * and when it passes, EVERY nucleon inside is captured at once - not the slow ones, all of
//     them.
//
// The energy each contributes is `e() - actualMass + field - barrier`, i.e. a kinetic energy with
// the nuclear field added and the barrier taken off, so it can be NEGATIVE - a nucleon sitting
// deep in the well pulls the mean down and can carry a fast one into the nucleus with it.
//
// ## REFUSED, by name
//
//   * `G4Absorber`, above, with its threshold measured.
#ifndef G4GPU_BIC_CASCADE_CAPTURE_CUH
#define G4GPU_BIC_CASCADE_CAPTURE_CUH

#include <cmath>

#include "physics/hadronic/bic/cascade_collision.cuh"

namespace g4gpu::bic {

/// `G4Absorber::WillBeAbsorbed` - a pion below `theCutOnPAbsorb` in KINETIC energy.
///
/// `cut_on_p_absorb` is zero, so this is false for everything; see the file header. It is written
/// out rather than replaced by `return false` so that a release which changes the constant gets
/// the predicate and not a stub.
__host__ __device__ inline bool will_be_absorbed(int pdg, double energy, double actual_mass,
                                                 double cut_on_p_absorb) {
  if (energy - actual_mass < cut_on_p_absorb) {
    if (pdg == imr::kPdgPiPlus || pdg == 111 || pdg == imr::kPdgPiMinus) { return true; }
  }
  return false;
}

/// `G4BinaryCascade::Absorb`'s first loop, which is all of it that ever runs.
///
/// Returns the number of secondaries that would be absorbed. It is zero for every input, and
/// `AbsorbRefusal::would_absorb` is the flag a caller checks before trusting that: reaching one
/// means the threshold changed and `G4Absorber` is needed after all.
struct AbsorbRefusal {
  bool would_absorb = false;   ///< G4Absorber is refused by name; see the file header
  int refused_pdg = 0;
};

/// `pool` is the whole track pool and `n` its length; the loop takes only the tracks tagged
/// `kListSecondary`, because Geant4 iterates `theSecondaryList`. **That filter is the whole
/// correctness of this function.** Without it the loop walks the nucleus's own target nucleons,
/// which are all `inside` - see `capture_decision` below, where leaving it out was measured.
__host__ __device__ inline bool absorb(const CascadeTrack* pool, int n,
                                       double cut_on_p_absorb, AbsorbRefusal& ref) {
  for (int i = 0; i < n; ++i) {
    const CascadeTrack& kt = pool[i];
    if (kt.list != kListSecondary) { continue; }
    if (kt.state != kInside) { continue; }
    if (will_be_absorbed(kt.pdg, kt.momentum.e, kt.actual_mass(), cut_on_p_absorb)) {
      ref.would_absorb = true;
      ref.refused_pdg = kt.pdg;
    }
  }
  return false;
}

/// What `Capture`'s gate looked at, kept so that a caller - and a test - can see the mean rather
/// than only the verdict.
struct CaptureDecision {
  int particles_below_cut = 0;   ///< every nucleon INSIDE, whatever its energy
  int particles_above_cut = 0;   ///< always zero: the increment is commented out upstream
  double captured_energy = 0.0;
  bool capture = false;
};

/// `G4BinaryCascade::Capture`'s decision, without moving any track.
///
/// `field_minus_barrier` is `GetField(pdg, pos) - GetBarrier(pdg)`, which the caller has the
/// propagator for. Splitting the decision from the move is the port's shape, not Geant4's: the
/// move is three lines and the decision is the part worth checking on its own.
///
/// ## THE LOOP IS OVER theSecondaryList AND NOTHING ELSE
///
/// `pool` is the whole track pool, so the `kListSecondary` filter is what makes this Geant4's
/// loop. Leaving it out does not look like a bug - every skipped track is still `inside` and
/// still a nucleon - and it is catastrophic, because the gate is a MEAN and a BOUND nucleon
/// contributes a NEGATIVE term: `e - m` is a few tens of MeV and the field is about -40.
/// MEASURED, on the C12 cases of `bic_imr_prop.csv` with the filter absent: the gate saw
/// `count 12` - the whole nucleus - with `capturedEnergy = -240.7 MeV`, a mean of -20.1 against
/// a threshold of 9, so it said CAPTURE on the first turn of the collision loop of every single
/// case. The projectile went straight into theCapturedList, the cascade ended with nothing in
/// theFinalState, and the excitation energy came out equal to the whole beam energy. Twelve of
/// the forty cases produced zero products where Geant4 produced one to five. Nothing threw,
/// nothing was refused, and the event looked like a projectile that had simply been absorbed.
template <typename Prop>
__host__ __device__ inline CaptureDecision capture_decision(const CascadeTrack* pool,
                                                            int n, double cut_on_p,
                                                            const Prop& propagator) {
  CaptureDecision d;
  for (int i = 0; i < n; ++i) {
    const CascadeTrack& kt = pool[i];
    if (kt.list != kListSecondary) { continue; }
    if (kt.state != kInside) { continue; }
    if (kt.pdg != imr::kPdgProton && kt.pdg != imr::kPdgNeutron) { continue; }
    const double field = propagator.field(kt.pdg, kt.position) - propagator.barrier(kt.pdg);
    d.captured_energy += kt.momentum.e - kt.actual_mass() + field;
    ++d.particles_below_cut;
  }
  // `particlesAboveCut == 0 &&` is commented out upstream, and so is the `if (energy < theCutOnP)`
  // that would have made the counter non-zero. What is left is the mean against 0.2*theCutOnP.
  if (d.particles_below_cut > 0 &&
      d.captured_energy / static_cast<double>(d.particles_below_cut) < 0.2 * cut_on_p) {
    d.capture = true;
  }
  return d;
}

}  // namespace g4gpu::bic

#endif
