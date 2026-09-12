// The two nuclear densities G4Fancy3DNucleus chooses between, and the base class's one method.
//
// Transcribed from G4VNuclearDensity.hh, G4NuclearFermiDensity.{hh,cc} and
// G4NuclearShellModelDensity.{hh,cc} (hadronic/util) in 11.1.1.
//
// Geant4 makes these three classes and dispatches through a virtual `GetRelativeDensity`. A
// device kernel cannot follow a vtable across a launch boundary, so the two concretes are one
// tagged struct here and the dispatch is a switch on `kind`. That is not a simplification of
// the physics: the CHOICE between them is `myA < 17` in `G4Fancy3DNucleus::Init` and it is
// reproduced there, so both shapes are reachable exactly where Geant4 reaches them.
//
// **std::exp and std::log are the right transcription on this machine, and that is a platform
// fact, not a convenience.** Both classes call `G4Exp`/`G4Log`, which are Geant4's own
// third-order expansions on Linux - the thing docs/HADRONIC_PLAN.md section 8 warns about for
// `G4Pow`. But `G4Exp.hh:60` and `G4Log.hh:60` open with `#ifdef WIN32 / #define G4Exp std::exp`,
// and the oracle this port is measured against is a Windows install. So on this oracle G4Exp IS
// std::exp, bit for bit, and a transcription that faithfully reproduced Geant4's expansion
// would be the one that failed. docs/RISK.md V67.
//
// **rho0 is a normalisation and it is NOT the central density.** `GetDensity` is
// `rho0 * GetRelativeDensity`, and the relative density is 1 at the centre for the shell model
// but `1/(1+exp(-R/a))` for the Fermi shape - about 0.9999 for iron, so the two conventions
// agree to four digits and disagree in principle. The Fermi rho0 is fixed by requiring the
// integral over the whole (untruncated) Fermi distribution to be A, using the standard
// `1 + pi^2 (a/R)^2` correction; the shell-model rho0 is the Gaussian normalisation
// `(pi R^2)^(-3/2)`. Neither is a measured central density and nothing should be compared
// against one.
#ifndef G4GPU_BIC_NUCLEAR_DENSITY_CUH
#define G4GPU_BIC_NUCLEAR_DENSITY_CUH

#include <cfloat>
#include <cmath>

#include "core/units.cuh"
#include "core/vec3.cuh"
#include "data/g4pow.hh"
#include "physics/hadronic/deexcitation/deex_params.cuh"

namespace g4gpu::bic {

namespace u = g4gpu::units;
using Vec3d = Vec3<double>;

/// Which of the two G4VNuclearDensity subclasses this is.
enum NuclearDensityKind : int { kShellModelDensity = 0, kFermiDensity = 1 };

/// G4VNuclearDensity plus the data members of both subclasses.
///
/// `the_r` and `a` are the Fermi shape's radius and surface thickness; `the_r_square` is the
/// shell model's R^2. Only one pair is meaningful for a given `kind`, and the unused one is
/// left at zero rather than aliased, so a misuse reads a zero and not another shape's number.
struct NuclearDensity {
  int kind = kFermiDensity;
  int the_a = 0;
  double rho0 = 0.0;
  double the_r = 0.0;         ///< G4NuclearFermiDensity::theR
  double a = 0.0;             ///< G4NuclearFermiDensity::a, 0.545 fermi
  double the_r_square = 0.0;  ///< G4NuclearShellModelDensity::theRsquare

  /// G4VNuclearDensity::GetDensity - `rho0 * GetRelativeDensity(pos)`.
  __host__ __device__ double density(const Vec3d& pos) const {
    return rho0 * relative_density(pos);
  }

  /// The two `GetRelativeDensity` overrides.
  __host__ __device__ double relative_density(const Vec3d& pos) const {
    if (kind == kFermiDensity) {
      // G4NuclearFermiDensity.hh:47
      return 1.0 / (1.0 + std::exp((g4gpu::mag(pos) - the_r) / a));
    }
    // G4NuclearShellModelDensity.cc:46
    return std::exp(-1.0 * g4gpu::mag2(pos) / the_r_square);
  }

  /// The two `GetRadius(maxRelativeDensity)` overrides - the radius at which the RELATIVE
  /// density falls to the given value. Both return DBL_MAX outside (0, 1]; that sentinel is
  /// Geant4's and it is load-bearing, because `G4Fancy3DNucleus::ChoosePositions` would
  /// otherwise sample positions in a ball of radius DBL_MAX and never place a nucleon.
  __host__ __device__ double radius(double max_relative_density) const {
    if (!(max_relative_density > 0.0 && max_relative_density <= 1.0)) { return DBL_MAX; }
    if (kind == kFermiDensity) {
      // G4NuclearFermiDensity.hh:52. The `exp(-theR/a)` term is the correction for the
      // distribution not being exactly 1 at the origin; it is ~1e-6 for iron and not dropped.
      return the_r +
             a * std::log((1.0 - max_relative_density + std::exp(-1.0 * the_r / a)) /
                          max_relative_density);
    }
    // G4NuclearShellModelDensity.cc:52
    return std::sqrt(the_r_square * std::log(1.0 / max_relative_density));
  }

  /// The two `GetDeriv` overrides - d(absolute density)/dr, used by G4RKPropagation's field.
  ///
  /// The Fermi one carries a cut-off, `r > 40*theR -> 0`, which the shell-model one does not
  /// have; at 40 R the Fermi exponential has already overflowed a double, so the cut is a guard
  /// against inf*0 and not a physics choice. Note that it is the only place in either class
  /// where the two shapes differ structurally rather than in their formula.
  __host__ __device__ double deriv(const Vec3d& pos) const {
    if (kind == kFermiDensity) {
      const double current_r = g4gpu::mag(pos);
      if (current_r > 40.0 * the_r) { return 0.0; }
      const double d = density(pos);
      return -std::exp((current_r - the_r) / a) * d * d / (a * rho0);
    }
    return -2.0 * g4gpu::mag(pos) / the_r_square * density(pos);
  }
};

/// G4NuclearFermiDensity::G4NuclearFermiDensity(A, Z). Z is taken and ignored, as in Geant4.
///
/// `r0 = 1.16*(1 - 1.16/A^(2/3)) fermi` is an A-dependent radius parameter, so `theR` is NOT
/// `r0 A^(1/3)` with a constant r0 - it falls below 1.16 A^(1/3) fm and does so most for light
/// nuclei. At A = 17, the lightest nucleus that reaches this class, `1 - 1.16/A^(2/3)` is 0.824.
///
/// `A^(1/3)` is `G4Pow::Z13(A)`, the tabulated integer cube root, not `G4Pow::A13` and not
/// `std::cbrt` - and `a13*a13` is used for A^(2/3) rather than `Z23`, which is the same number
/// only because Z23 is defined as Z13 squared. Kept as written.
__host__ __device__ inline NuclearDensity make_fermi_density(int an_a, int /*a_z*/) {
  NuclearDensity d;
  d.kind = kFermiDensity;
  d.the_a = an_a;
  d.a = 0.545 * deex::fermi();
  const double a13 = data::g4pow_z13<double>(an_a);
  const double r0 = 1.16 * (1.0 - 1.16 / (a13 * a13)) * deex::fermi();
  d.the_r = r0 * a13;
  const double pi2 = u::pi<double>() * u::pi<double>();  // CLHEP pi2
  const double as = d.a / d.the_r;
  d.rho0 = 3.0 / (4.0 * u::pi<double>() * r0 * r0 * r0 * static_cast<double>(d.the_a) *
                  (1.0 + as * as * pi2));
  return d;
}

/// G4NuclearShellModelDensity::G4NuclearShellModelDensity(A, Z). Z is taken and ignored.
///
/// `r0sq = 0.8133 fermi^2` is a squared length written as one constant, so the Gaussian width
/// is `sqrt(0.8133) = 0.9018 fm` times `A^(1/3)`. `Z23(A)` here IS `Z13(A)` squared, by
/// G4Pow's definition, so R^2 goes as A^(2/3) and R as A^(1/3) - the same scaling the Fermi
/// class has, with a Gaussian instead of a Woods-Saxon profile.
__host__ __device__ inline NuclearDensity make_shell_model_density(int an_a, int /*a_z*/) {
  NuclearDensity d;
  d.kind = kShellModelDensity;
  d.the_a = an_a;
  const double r0sq = 0.8133 * deex::fermi() * deex::fermi();
  d.the_r_square = r0sq * data::g4pow_z13<double>(an_a) * data::g4pow_z13<double>(an_a);
  const double x = 1.0 / (u::pi<double>() * d.the_r_square);
  d.rho0 = x * std::sqrt(x);
  return d;
}

}  // namespace g4gpu::bic

#endif
