// Nuclear radii and Coulomb barriers for evaporation and Fermi break-up.
//
// Transcribed from G4NuclearRadii (hadronic/util), G4VCoulombBarrier, G4CoulombBarrier and its
// six ejectile subclasses, and G4FermiCoulombBarrier (11.1.1).
//
// Two barrier families, and they are not the same function:
//
//   G4CoulombBarrier      cb = e^2 Z_ej Z_res / (R_CB(Z_res, A_res) + rho), rho = 0.4 R_CB of
//                         the EJECTILE. Used by every evaporation channel.
//   G4FermiCoulombBarrier a difference of three Z^2/A^(1/3) terms with rho = R_CB of the
//                         emitter and a 1.3 fm r0. Used only inside the Fermi break-up pool,
//                         where the "ejectile" can be as heavy as the residual.
//
// The 1/(1 + sqrt(U/2A)) softening in the first is applied only when U > 0, and every caller in
// the default configuration passes U = 0 - so the barrier an evaporation channel uses is the
// unsoftened one. Ported anyway, since the argument exists and `G4Fragment`'s excitation is
// what a future caller would pass.
//
// G4Pow::powZ and G4Pow::powN are here rather than in src/data/g4pow.hh so that this package
// appends to no shared header while several packages are in flight. They are the same
// expansions - powZ(Z, y) = expA(y * log Z) - and if two copies ever exist,
// tests/test_deex_nuclear.cu compares this one against Geant4 directly, so the copy that is
// checked is the one that runs.
#ifndef G4GPU_DEEX_COULOMB_BARRIER_CUH
#define G4GPU_DEEX_COULOMB_BARRIER_CUH

#include <cmath>

#include "core/units.cuh"
#include "data/g4pow.hh"
#include "physics/hadronic/deexcitation/deex_params.cuh"

namespace g4gpu::deex {

namespace u = g4gpu::units;

/// G4Pow::powZ(Z, y) = expA(y * lz[Z]) with lz[Z] = log(Z). Z = 0 would read lz[0] = 0 in
/// Geant4 and give expA(0) = 1; reproduced rather than guarded, because a caller reaching
/// powZ(0, y) has a Z it should not have and returning 1 keeps that visible.
__host__ __device__ inline double g4pow_pow_z(int Z, double y) {
  const double lz = (Z <= 0) ? 0.0 : log(static_cast<double>(Z));
  return data::g4pow_exp_a<double>(y * lz);
}

/// G4Pow::powN(x, n) - repeated multiplication up to |n| = 8, std::pow beyond.
__host__ __device__ inline double g4pow_pow_n(double x, int n) {
  if (x == 0.0) { return 0.0; }
  if (n > 8 || n < -8) { return std::pow(x, static_cast<double>(n)); }
  double res = 1.0;
  if (n >= 0) {
    for (int i = 0; i < n; ++i) { res *= x; }
  } else {
    const double y = 1.0 / x;
    const int nn = -n;
    for (int i = 0; i < nn; ++i) { res *= y; }
  }
  return res;
}

// ---------------------------------------------------------------------------------------------
// G4NuclearRadii
// ---------------------------------------------------------------------------------------------

/// G4NuclearRadii::r0[] - the Coulomb-barrier radius parameter per Z, 1.2 fm at index 0 and
/// then Z = 1..92. Everything above Z = 92 is clamped to 92 by RadiusCB.
__host__ __device__ inline const double* nuclear_radii_r0() {
  static const double v[93] = {
    1.2,
    1.3,  1.3,  1.3,  1.3,  1.17, 1.54, 1.65, 1.71, 1.7,  1.75,  //  1-10
    1.7,  1.57, 1.53, 1.4,  1.3,  1.30, 1.44, 1.4,  1.4,  1.4,   // 11-20
    1.4,  1.4,  1.46, 1.4,  1.4,  1.46, 1.55, 1.5,  1.38, 1.48,  // 21-30
    1.4,  1.4,  1.4,  1.46, 1.4,  1.4,  1.4,  1.4,  1.4,  1.45,  // 31-40
    1.4,  1.4,  1.4,  1.4,  1.4,  1.4,  1.45, 1.48, 1.4,  1.52,  // 41-50
    1.46, 1.4,  1.4,  1.4,  1.4,  1.4,  1.4,  1.4,  1.4,  1.5,   // 51-60
    1.4,  1.4,  1.4,  1.3,  1.3,  1.3,  1.3,  1.3,  1.3,  1.4,   // 61-70
    1.3,  1.3,  1.3,  1.3,  1.3,  1.3,  1.3,  1.3,  1.33, 1.43,  // 71-80
    1.3,  1.32, 1.34, 1.3,  1.3,  1.3,  1.3,  1.3,  1.3,  1.3,   // 81-90
    1.3,  1.3,
  };
  return v;
}

/// G4NuclearRadii::ExplicitRadius - measured rms radii for the seven lightest species, which
/// override every parameterisation below. Note it tests Z <= 4 and then A, so a Z = 3 nuclide
/// of ANY mass number gets Li7's 2.40 fm and any Z = 4 gets Be9's 2.51 fm.
__host__ __device__ inline double explicit_radius(int Z, int A) {
  double R = 0.0;
  if (Z <= 4) {
    if (A == 1) { R = 0.895 * fermi(); }                 // p
    else if (A == 2) { R = 2.13 * fermi(); }             // d
    else if (Z == 1 && A == 3) { R = 1.80 * fermi(); }   // t
    else if (Z == 2 && A == 3) { R = 1.96 * fermi(); }   // He3
    else if (Z == 2 && A == 4) { R = 1.68 * fermi(); }   // He4
    else if (Z == 3) { R = 2.40 * fermi(); }             // Li7
    else if (Z == 4) { R = 2.51 * fermi(); }             // Be9
  }
  return R;
}

/// G4NuclearRadii::RadiusCB - the radius the Coulomb barrier is built from.
__host__ __device__ inline double radius_cb(int Z, int A) {
  double R = explicit_radius(Z, A);
  if (R == 0.0) {
    const int z = (Z < 92) ? Z : 92;
    R = nuclear_radii_r0()[z] * data::g4pow_z13<double>(A) * fermi();
  }
  return R;
}

/// G4NuclearRadii::Radius - the general nuclear radius. Not used by the barriers; here because
/// the four radius definitions in that class differ by up to 30% and naming only the one this
/// module uses would invite the next reader to reach for the wrong one.
__host__ __device__ inline double nuclear_radius(int Z, int A) {
  double R = explicit_radius(Z, A);
  if (R == 0.0) {
    if (A <= 50) {
      double y = 1.1;
      if (A <= 15) { y = 1.26; }
      else if (A <= 20) { y = 1.19; }
      else if (A <= 30) { y = 1.12; }
      const double x = data::g4pow_z13<double>(A);
      R = y * (x - 1.0 / x);
    } else {
      R = g4pow_pow_z(A, 0.27);
    }
    R *= fermi();
  }
  return R;
}

// ---------------------------------------------------------------------------------------------
// G4CoulombBarrier - the evaporation barrier, one instance per ejectile (A, Z).
// ---------------------------------------------------------------------------------------------

/// G4CoulombBarrier::GetCoulombBarrier(ARes, ZRes, U) for an ejectile of (theA, theZ).
///
/// `factor = elm_coupling * theZ` and `theRho = 0.4 * RadiusCB(theZ, theA)` are set in the
/// constructor; theR0 = 1.5 fm is set too and never read by this class. A neutral ejectile
/// returns exactly zero.
__host__ __device__ inline double coulomb_barrier(int ejA, int ejZ, int ARes, int ZRes,
                                                  double U) {
  if (ejZ == 0) { return 0.0; }
  const double factor = elm_coupling() * static_cast<double>(ejZ);
  const double rho = 0.4 * radius_cb(ejZ, ejA);
  double cb = factor * ZRes / (radius_cb(ZRes, ARes) + rho);
  if (U > 0.0) {
    cb /= (1.0 + std::sqrt(U / ((2 * ARes) * u::MeV<double>())));
  }
  return cb;
}

/// The barrier penetration factor K of Dostrovsky, Fraenkel and Friedlander, Phys. Rev. 116
/// (1959) 683, as the six ejectile subclasses of G4CoulombBarrier define it.
///
/// The base class G4CoulombBarrier has a combined version of this that branches on theZ, and
/// each of G4Proton/Deuteron/Triton/He3/AlphaCoulombBarrier overrides it with the single case.
/// The two agree: the base class's `res += 0.06*(theA - 1)` reproduces the deuteron's +0.06 and
/// the triton's +0.12, and `res += 0.12*(4 - theA)` reproduces He3's +0.12 and the alpha's +0.
/// Written as the override set, because that is what a channel actually holds.
///
/// Nothing in the default configuration calls it: it is used by the OPTxs == 0 emission-width
/// branch, and OPTxs is 3. Ported so that a dumped table of it can show that.
__host__ __device__ inline double barrier_penetration_factor(int ejA, int ejZ, int aZ) {
  if (ejZ == 1) {
    double res = (aZ >= 70) ? 0.80
                            : (((0.2357e-5 * aZ) - 0.42679e-3) * aZ + 0.27035e-1) * aZ + 0.19025;
    res += 0.06 * (ejA - 1);
    return res;
  }
  if (ejZ == 2 && ejA <= 4) {
    double res = (aZ >= 70) ? 0.98
                            : (((0.23684e-5 * aZ) - 0.42143e-3) * aZ + 0.25222e-1) * aZ + 0.46699;
    res += 0.12 * (4 - ejA);
    return res;
  }
  return 1.0;  // G4VCoulombBarrier::BarrierPenetrationFactor - neutrons and everything else
}

// ---------------------------------------------------------------------------------------------
// G4FermiCoulombBarrier
// ---------------------------------------------------------------------------------------------

/// G4FermiCoulombBarrier::GetCoulombBarrier(ARes, ZRes, -) for an emitter of (theA, theZ).
/// The excitation argument is accepted and ignored, by Geant4 too.
///
/// `factor = elm_coupling * 0.6 * Z13(7) / theRho` with `theRho = RadiusCB(theZ, theA)` - the
/// Z13(7) is nitrogen's cube root as a fixed number, not a property of either fragment, and
/// the 1.3 fm r0 passed to SetParameters is never read. Both are Geant4's.
__host__ __device__ inline double fermi_coulomb_barrier(int emA, int emZ, int ARes, int ZRes) {
  if (emZ == 0) { return 0.0; }
  const double rho = radius_cb(emZ, emA);
  const double factor = elm_coupling() * 0.6 * data::g4pow_z13<double>(7) / rho;
  const int A = emA + ARes;
  const int Z = emZ + ZRes;
  return factor * ((static_cast<double>(Z) * Z) / data::g4pow_z13<double>(A) -
                   (static_cast<double>(emZ) * emZ) / data::g4pow_z13<double>(emA) -
                   (static_cast<double>(ZRes) * ZRes) / data::g4pow_z13<double>(ARes));
}

}  // namespace g4gpu::deex

#endif
