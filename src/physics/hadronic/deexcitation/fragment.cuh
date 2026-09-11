// G4Fragment and the four-vector arithmetic the de-excitation module does on it.
//
// Transcribed from G4Fragment (hadronic/util) and the CLHEP HepLorentzVector operations it
// uses - mag(), boost(), boostVector() (11.1.1).
//
// A `Fragment` is a nucleus in flight with an excitation energy, and the invariant that makes
// the module work is that the excitation is DERIVED, not stored independently:
//
//     theExcitationEnergy = theMomentum.mag() - theGroundStateMass
//
// recomputed on every SetMomentum. So an evaporation step that subtracts the ejectile's
// 4-momentum from the parent's automatically leaves the residual with the right excitation,
// and there is nowhere for the two to disagree. The one exception is
// SetExcEnergyAndMomentum, which sets the excitation and then rebuilds the 4-momentum from it
// - that is what G4GammaTransition uses, and it is the other direction of the same identity.
//
// The 10 eV floor on the excitation is G4Fragment's, not this module's: `minFragExcitation`
// in CalculateMassAndExcitationEnergy. Below it the excitation is set to exactly zero, and
// below MINUS it Geant4 prints a warning. The warning is kept as a flag on the fragment
// rather than a printout, because a device kernel cannot print and because a caller that
// produces a negative-mass fragment wants to know.
//
// Doubles throughout, for the reason nuclear_masses.cuh gives: these are ~2e5 MeV numbers
// whose differences matter at 1e-5 MeV.
#ifndef G4GPU_DEEX_FRAGMENT_CUH
#define G4GPU_DEEX_FRAGMENT_CUH

#include <cmath>

#include "core/units.cuh"
#include "core/vec3.cuh"
#include "physics/hadronic/deexcitation/nuclear_masses.cuh"

namespace g4gpu::deex {

using Vec3d = Vec3<double>;

/// CLHEP::HepLorentzVector, only the parts this module uses.
struct LorentzVector {
  Vec3d v{0.0, 0.0, 0.0};
  double e = 0.0;

  // No __host__ __device__ on the defaulted constructor: nvcc warns 20012 that the annotation
  // is ignored there, and a warning in a header every test includes is noise that hides a real
  // one. An implicitly-defaulted constructor is callable from device code anyway.
  LorentzVector() = default;
  __host__ __device__ LorentzVector(const Vec3d& p, double energy) : v(p), e(energy) {}
  __host__ __device__ LorentzVector(double x, double y, double z, double energy)
      : v{x, y, z}, e(energy) {}

  /// HepLorentzVector::mag() - the SIGNED square root of the invariant, i.e.
  /// `m2 < 0 ? -sqrt(-m2) : sqrt(m2)`. The sign matters: G4Fragment computes its excitation
  /// from this, and a fragment whose 4-momentum is spacelike by rounding gets a small negative
  /// excitation rather than a NaN.
  __host__ __device__ double mag() const {
    const double m2 = e * e - g4gpu::mag2(v);
    return (m2 < 0.0) ? -std::sqrt(-m2) : std::sqrt(m2);
  }

  /// `HepLorentzVector::boostVector()`, which is `pp * (1./ee)` and NOT `pp/ee`
  /// (LorentzVector.cc:189).
  ///
  /// This divided, and the two differ by an ulp. docs/RISK.md V37 already measured what that
  /// ulp costs in `elastic/hadron_elastic.cuh` - "the recoil direction 5.8e-13 out and the
  /// alpha-on-Pb recoil energy 2.3e-10 out" - because a recoil is a cancellation and a
  /// cancellation amplifies an ulp by the ratio of the operands to the result. It costs the
  /// same thing here, and P7's oracle is what found it: MEASURED, by putting the division back
  /// and rerunning `tests/test_capture.cu`, the 5493 secondary direction components of a
  /// capture cascade go from exactly zero to 1.4e-13 and their kinetic energies from exactly
  /// zero to 2.0e-16. Nothing in P3's own tests could see it: their fragments are at rest or
  /// nearly so, where the boost vector is zero and the arithmetic never runs.
  ///
  /// The `e == 0` guard is not CLHEP's shape - CLHEP prints to cerr and returns `pp/ee` - but
  /// it is the same answer for `pp == 0` and a kernel cannot print. A non-zero momentum with
  /// zero energy is unphysical and unreachable from any fragment this module builds.
  __host__ __device__ Vec3d boost_vector() const {
    if (e == 0.0) { return Vec3d{0.0, 0.0, 0.0}; }
    const double inv = 1.0 / e;
    return Vec3d{v.x * inv, v.y * inv, v.z * inv};
  }

  /// HepLorentzVector::boost(b). Written out rather than assembled from a matrix because
  /// CLHEP's own form is what the module's kinematics were tuned against, and the `gamma2`
  /// factor below is singular-free only in this arrangement.
  __host__ __device__ void boost(const Vec3d& b) {
    const double b2 = g4gpu::mag2(b);
    if (b2 <= 0.0) { return; }
    const double ggamma = 1.0 / std::sqrt(1.0 - b2);
    const double bp = g4gpu::dot(b, v);
    const double gamma2 = (b2 > 0.0) ? (ggamma - 1.0) / b2 : 0.0;
    const double nx = v.x + gamma2 * bp * b.x + ggamma * b.x * e;
    const double ny = v.y + gamma2 * bp * b.y + ggamma * b.y * e;
    const double nz = v.z + gamma2 * bp * b.z + ggamma * b.z * e;
    e = ggamma * (e + bp);
    v = Vec3d{nx, ny, nz};
  }

  __host__ __device__ LorentzVector& operator-=(const LorentzVector& o) {
    v = v - o.v;
    e -= o.e;
    return *this;
  }
  __host__ __device__ LorentzVector& operator+=(const LorentzVector& o) {
    v = v + o.v;
    e += o.e;
    return *this;
  }
};

__host__ __device__ inline LorentzVector operator-(LorentzVector a, const LorentzVector& b) {
  a -= b;
  return a;
}
__host__ __device__ inline LorentzVector operator+(LorentzVector a, const LorentzVector& b) {
  a += b;
  return a;
}

/// G4Fragment, restricted to the non-hyper case. `lambdas` is carried so that the handler can
/// refuse a hyper-fragment by name instead of silently treating it as an ordinary one.
struct Fragment {
  LorentzVector momentum;
  double ground_state_mass = 0.0;
  double excitation = 0.0;
  int z = 0;
  int a = 0;
  int lambdas = 0;
  int floating_level = 0;   ///< G4Fragment::GetFloatingLevelNumber
  bool long_lived = false;  ///< G4Fragment::IsLongLived
  /// Set when the excitation came out below -10 eV. G4Fragment prints a warning here
  /// (ExcitationEnergyWarning) and clamps to zero; a device kernel cannot print, so the fact
  /// travels with the fragment.
  bool negative_excitation = false;
  /// 0 for a nucleus. A gamma or an electron produced by photon evaporation is carried as a
  /// Fragment with A = 0 and this set, exactly as G4Fragment does with its particle-definition
  /// constructor - so one output list holds both.
  int pdg_if_not_nucleus = 0;

  /// G4Fragment::CalculateMassAndExcitationEnergy.
  __host__ __device__ void recalculate() {
    ground_state_mass = deex::ground_state_mass(z, a);
    const double min_exc = 10.0 * g4gpu::units::eV<double>();
    excitation = momentum.mag() - ground_state_mass;
    if (excitation < min_exc) {
      if (excitation < -min_exc) { negative_excitation = true; }
      excitation = 0.0;
    }
  }

  __host__ __device__ void set_momentum(const LorentzVector& lv) {
    momentum = lv;
    recalculate();
  }

  /// G4Fragment::SetZandA_asInt followed by SetMomentum, which is SetZAandMomentum.
  __host__ __device__ void set_za_and_momentum(const LorentzVector& lv, int Z, int A) {
    z = Z;
    a = A;
    momentum = lv;
    recalculate();
  }

  /// G4Fragment::SetExcEnergyAndMomentum - the other direction: the excitation is given and
  /// the 4-momentum is rebuilt as a mass at rest boosted by `v`'s velocity.
  __host__ __device__ void set_exc_energy_and_momentum(double eexc, const LorentzVector& v) {
    excitation = eexc;
    momentum = LorentzVector(0.0, 0.0, 0.0, ground_state_mass + eexc);
    momentum.boost(v.boost_vector());
  }
};

/// A fragment built from (A, Z) and a 4-momentum: `G4Fragment(G4int A, G4int Z,
/// const G4LorentzVector&)`, and the argument order is Geant4's, **A first**.
///
/// It was (Z, A) here for a day and every one of the five emitters in this module called it as
/// (A, Z), because that is the order the Geant4 line beside them has. The result was that
/// every ejectile came out with its charge and its mass number exchanged: a neutron became
/// (Z = 1, A = 0), which the handler's final loop reads as `A == 0` and reports as a gamma; an
/// alpha became (Z = 4, A = 2), for which G4UnstableFragmentBreakUp has no channel at all.
/// tests/test_deex_breakup.cu found it in one run - 60,000 gammas where Geant4 makes 64,000
/// and no neutrons at all where it makes 59,000 - and the refusal machinery named the
/// unphysical fragment rather than reading its tables out of bounds. Matching Geant4's order
/// is what stops it coming back.
__host__ __device__ inline Fragment make_fragment(int A, int Z, const LorentzVector& lv) {
  Fragment f;
  f.z = Z;
  f.a = A;
  f.momentum = lv;
  f.recalculate();
  return f;
}

/// A fragment at rest with a given excitation, moving with momentum `pz` along z. The
/// convenience the oracle's dump uses, reproduced here so a test builds the same input - and
/// with the dump's argument order, `(Z, A)`, which is NOT make_fragment's.
__host__ __device__ inline Fragment make_excited_fragment(int Z, int A, double eexc,
                                                          double pz) {
  const double m = deex::ground_state_mass(Z, A) + eexc;
  return make_fragment(A, Z, LorentzVector(0.0, 0.0, pz, std::sqrt(m * m + pz * pz)));
}

/// An isotropic direction, from G4RandomDirection() (global/HEPRandom).
///
/// It is Marsaglia's rejection pair, NOT a (cos theta, phi) sample: two uniforms in [-1, 1],
/// rejected outside the unit disc, then `(2u sqrt(1-b), 2v sqrt(1-b), 2b - 1)`. Both give an
/// isotropic direction, so the distribution is the same either way - but the number of random
/// draws per direction is 2 on average 4/pi times, not 2 exactly, and no trigonometry is
/// evaluated. Transcribed as written because "isotropic" is the specification and this is the
/// implementation, and because a sampler that consumes a different count of random numbers
/// cannot be compared against Geant4 stream for stream even in principle.
template <typename Rng>
__host__ __device__ inline Vec3d random_direction(Rng& rng) {
  double u, v, b;
  do {
    u = 2.0 * rng.uniform() - 1.0;
    v = 2.0 * rng.uniform() - 1.0;
    b = u * u + v * v;
  } while (b > 1.0);
  const double a = 2.0 * std::sqrt(1.0 - b);
  return Vec3d{a * u, a * v, 2.0 * b - 1.0};
}

}  // namespace g4gpu::deex

#endif
