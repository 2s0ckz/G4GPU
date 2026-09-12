// G4FermiMomentum - the local Fermi momentum of a nucleus, and a sample from inside that ball.
//
// Transcribed from G4FermiMomentum.{hh,cc} (hadronic/util) in 11.1.1. The whole class is four
// lines of arithmetic; what is worth writing down is what each of them is NOT.
//
// **`GetFermiMomentum(density)` multiplies the density by A.** The formula is
// `pmax = hbar c (3 pi^2 rho)^(1/3)` and the code is `constofpmax * cbrt(density * theA)` with
// `constofpmax = hbarc * (3 pi^2)^(1/3)`. So the `rho` in the formula is the number density of
// ONE nucleon species times A, i.e. `theA * theDensity->GetDensity(pos)` - because
// G4VNuclearDensity's rho0 is normalised so that the integral of `GetDensity` is 1, not A.
// (G4NuclearFermiDensity's rho0 carries a `1/theA`; the shell model's does not carry an A at
// all, and its normalisation `(pi R^2)^(-3/2)` integrates a Gaussian to 1.) So the `theA` here
// is the factor that turns a probability density into a number density, and `theZ` - set by
// `Init` and stored - is read nowhere in the class.
//
// **`cbrt` is `G4Pow::A13`, not `std::cbrt`.** The private helper is
// `G4double cbrt(G4double x) { return G4Pow::GetInstance()->A13(x); }`, and A13 is the
// third-order expansion about a quarter-integer that docs/HADRONIC_PLAN.md section 8 names.
// Its argument here is `density * theA`, a number of order 1e-38 in mm^-3, so it is folded
// through A13's `x < 1 -> 1/x` inversion branch and the expansion is evaluated at a huge
// argument. That is Geant4's answer and it is what this reproduces; `std::cbrt` differs from it
// at 1e-7 relative, which is 1e8 times the tolerance this is compared at.
//
// **`GetMomentum` samples a uniform ball, and rejects on `mag() > 1` and not `mag2() > 1`.**
// Same acceptance set, one more square root per trial, and - the part that matters - the same
// number of uniform deviates consumed per trial, which is three.
#ifndef G4GPU_BIC_FERMI_MOMENTUM_CUH
#define G4GPU_BIC_FERMI_MOMENTUM_CUH

#include <cmath>

#include "core/units.cuh"
#include "core/vec3.cuh"
#include "data/g4pow.hh"

namespace g4gpu::bic {

namespace u = g4gpu::units;

/// `constofpmax = hbarc * cbrt(3 pi^2)` from G4FermiMomentum's constructor, with `cbrt` the
/// A13 expansion. `3 pi^2 = 29.608813203268074` and `pi2` is CLHEP's `pi*pi`.
__host__ __device__ inline double fermi_const_of_pmax() {
  const double pi2 = u::pi<double>() * u::pi<double>();
  return u::hbarc<double>() * data::g4pow_a13<double>(3.0 * pi2);
}

/// G4FermiMomentum. `the_z` is stored because `Init` stores it; nothing reads it.
struct FermiMomentum {
  int the_a = 0;
  int the_z = 0;

  /// G4FermiMomentum::Init.
  __host__ __device__ void init(int an_a, int a_z) {
    the_a = an_a;
    the_z = a_z;
  }

  /// G4FermiMomentum::GetFermiMomentum(density).
  __host__ __device__ double fermi_momentum(double density) const {
    return fermi_const_of_pmax() *
           data::g4pow_a13<double>(density * static_cast<double>(the_a));
  }

  /// G4FermiMomentum::GetMomentum(density, maxMomentum = -1).
  ///
  /// Three uniforms per trial, rejecting outside the unit ball, then scaled by `maxMomentum` -
  /// so the sample is uniform in the ball of radius maxMomentum and NOT uniform in |p| or in
  /// the Fermi-gas sense. A negative `maxMomentum` (the default) means "use the local Fermi
  /// momentum", which is the only call G4Fancy3DNucleus makes on the first attempt.
  template <typename Rng>
  __host__ __device__ Vec3<double> momentum(double density, Rng& rng,
                                            double max_momentum = -1.0) const {
    if (max_momentum < 0.0) { max_momentum = fermi_momentum(density); }
    Vec3<double> p{0.0, 0.0, 0.0};
    do {
      p = Vec3<double>{2.0 * rng.uniform() - 1.0, 2.0 * rng.uniform() - 1.0,
                       2.0 * rng.uniform() - 1.0};
    } while (g4gpu::mag(p) > 1.0);
    return p * max_momentum;
  }
};

}  // namespace g4gpu::bic

#endif
