// G4RKPropagation - moving a cascade particle through the nuclear potential.
//
// Transcribed from G4RKPropagation.{hh,cc}, G4KM_NucleonEqRhs.{hh,cc},
// G4KM_OpticalEqRhs.{hh,cc} (models/binary_cascade) and, because the propagation is done by
// Geant4's magnetic-field machinery, G4ClassicalRK4.cc, G4MagErrorStepper.cc,
// G4MagIntegratorDriver.{cc,icc,hh}, G4VIntegrationDriver.hh and G4FieldTrack.{cc,icc}
// (geometry/magneticfield) in 11.1.1.
//
// The cascade steps in TIME, and the field integrator steps in curve length, and BIC gets away
// with it because the right-hand side is written so that the two coincide:
// `dydx[0..2] = c p/E` is a velocity, so integrating it over the driver's "curve length"
// advances the position by a distance while the parameter advances by a time. Every place the
// driver compares a length to its step therefore compares a length to a time. Two of those
// comparisons matter:
//
//   * `eps_pos = eps_rel_max * max(h, fMinimumStep)` in `OneGoodStep` is the position-error
//     tolerance, and `h` is a time in ns. With `eps = 0.01` and a cascade step of 0.01 ns that
//     is 1e-4 - read as mm, 1e8 fm. So the POSITION error never fails a trial step and the
//     adaptive step size is controlled entirely by the momentum error `|dp|^2/|p|^2/eps^2`.
//     docs/RISK.md V71.
//   * `endPointDist >= hdid*(1.+perMillion)` in `AccurateAdvance` compares the chord length in
//     mm against the step in ns, and c = 299.79 mm/ns, so it is true on essentially every step
//     and `fNoBadSteps` counts every step. The warning it guards is inside `#ifdef
//     G4DEBUG_FIELD`, so nothing is printed and nothing changes; the statistic is meaningless.
//
// **`QuickAdvance` is reachable, and finding that out cost a draft.** The driver takes it when
// `h <= fMinimumStep`, and `G4RKPropagation::FieldTransport` builds the driver with
// `hMin = 1.0e-25*second` = 1e-16 ns against a cascade time step of 1e-5 to 1 ns - twelve
// orders of magnitude of headroom, which is why this header first said the arm could not be
// entered. It can: `AccurateAdvance`'s overshoot clamp `if (x+h > x2) h = x2 - x` can leave a
// step of a few times 1e-17 ns when the previous step landed just short of the end of the
// interval, and 17 of the 55 (nucleus, case) pairs in `bic_rk.csv` do exactly that, for 55
// sub-steps in all. Those two 55s are a coincidence - a draft of this header quoted them as one
// number - so `tests/test_bic_nucleus.cu` now prints both. It is transcribed, and
// `RkAdvanceReport::n_quick_advance` counts it.
//
// **The driver's parameters, from the constructor and `ReSetParameters`:** `safety = 0.9`;
// `IntegratorOrder() = 4` for G4ClassicalRK4, so `pshrnk = -0.25` and `pgrow = -0.2`;
// `errcon = pow(max_stepping_increase/safety, 1/pgrow) = pow(5/0.9, -5) = 1.88958e-4`;
// `max_stepping_increase = 5`; `fMaxNoSteps = fMaxStepBase/IntegratorOrder() = 250/4 = 62`;
// `fSmallestFraction = 1e-12`; `eps = 0.01`, passed by FieldTransport. `fNoVars =
// max(6, fMinNoVars=12) = 12` in the driver and `GetNumberOfVariables() = 6` in the stepper,
// which is why the stepper integrates six components and copies six through.
//
// **Richardson extrapolation is where the error estimate comes from.**
// `G4MagErrorStepper::Stepper` takes two half steps and one full step of the same RK4
// `DumbStepper`, calls the difference the error, and adds `error/15` to the two-half-step
// answer. So one accepted step is THREE RK4 steps and eleven right-hand-side evaluations, and
// the answer is fifth-order.
//
// **REFUSED, by name:**
//   * `G4KM_OpticalEqRhs` is transcribed (it is 20 lines and the pion fields need it) but the
//     nine species that use it other than the three pions - anti-proton, three kaons, three
//     sigmas - are refused in `nuclear_field.cuh`, so only the pion equations are reachable.
//   * `G4RKFieldIntegrator` and `G4Absorber`: not on this path. `G4RKPropagation` never
//     constructs G4RKFieldIntegrator, and nothing else in 11.1.1 does either.
//   * spin: `hasSpin` is `y[9..11].mag2() > 0` and `G4FieldTrack`'s polarization is (0,0,0)
//     here, so the spin error term is never evaluated. `NormalisePolarizationVector` is
//     likewise dead - it runs only for `nvar == 12` and nvar is 6.
#ifndef G4GPU_BIC_RK_PROPAGATION_CUH
#define G4GPU_BIC_RK_PROPAGATION_CUH

#include <cfloat>
#include <cmath>

#include "core/units.cuh"
#include "core/vec3.cuh"
#include "data/g4pow.hh"
#include "physics/hadronic/bic/kinetic_track.cuh"
#include "physics/hadronic/bic/nuclear_field.cuh"
#include "physics/hadronic/bic/nucleus/nucleus_model.cuh"

namespace g4gpu::bic {

// =============================================================================================
// CLHEP's pure-boost LorentzRotation, because G4RKPropagation applies one four times
// =============================================================================================

/// `p *= G4LorentzRotation(boost)`, i.e. `HepLorentzRotation::set(bx,by,bz)` followed by
/// `vectorMultiplication`.
///
/// Mathematically this is `HepLorentzVector::boost(boost)` - the matrix's `bgamma =
/// gamma^2/(1+gamma)` and boost()'s `gamma2 = (gamma-1)/b^2` are the same number - but they are
/// not the same arithmetic and they do not agree in the last bit. G4RKPropagation writes the
/// rotation form in all four of its momentum corrections, so that is what is written here;
/// fragment.cuh's header records what one ulp costs a recoil.
__host__ __device__ inline LorentzVector boost_by_rotation(const LorentzVector& p,
                                                           const Vec3d& b) {
  const double bp2 = g4gpu::mag2(b);
  if (bp2 <= 0.0) { return p; }
  const double gamma = 1.0 / std::sqrt(1.0 - bp2);
  const double bgamma = gamma * gamma / (1.0 + gamma);
  const double br = g4gpu::dot(b, p.v);
  LorentzVector out;
  out.v.x = p.v.x + bgamma * b.x * br + gamma * b.x * p.e;
  out.v.y = p.v.y + bgamma * b.y * br + gamma * b.y * p.e;
  out.v.z = p.v.z + bgamma * b.z * br + gamma * b.z * p.e;
  out.e = gamma * (br + p.e);
  return out;
}

// =============================================================================================
// The two equations of motion
// =============================================================================================

/// Which of the two `G4Mag_EqRhs` subclasses a species uses. `G4RKPropagation::Init` gives the
/// proton and the neutron a `G4KM_NucleonEqRhs` and everything else a `G4KM_OpticalEqRhs`.
enum EqRhsKind : int { kNucleonEq = 0, kOpticalEq = 1 };

/// G4KM_NucleonEqRhs and G4KM_OpticalEqRhs, as one struct with a tag.
///
/// `G4KM_NucleonEqRhs`'s constructor sets `factor = hbarc^2 * A23(3 pi^2 A)/3` and its
/// right-hand side divides that by `theMass` and by `A13(density)`. `G4KM_OpticalEqRhs`'s
/// `SetFactor` folds everything into one number. The two differ in three ways that are not
/// cosmetic:
///
///   1. the nucleon equation's force term is `+deriv*y[i]/yMod*c_light`; the optical one's is
///      `-deriv*...`. The sign is opposite, and the nucleon version has the minus sign
///      commented out three lines above it in Geant4.
///   2. the nucleon equation's `deriv` carries a `1/A13(density)` factor, i.e. the potential is
///      `~rho^(2/3)` (a Fermi energy), while the optical one's potential is linear in rho.
///   3. `SetFactor` builds the nucleus mass with `+bindingEnergy`, the same sign error as the
///      pion fields - see `nuclear_field.cuh`. It enters only through the reduced mass.
struct EqRhs {
  int kind = kNucleonEq;
  double the_mass = 0.0;
  double factor = 0.0;     ///< G4KM_NucleonEqRhs::factor or G4KM_OpticalEqRhs::theFactor

  /// `G4KM_NucleonEqRhs::EvaluateRhsGivenB` / `G4KM_OpticalEqRhs::EvaluateRhsGivenB`. `y` is
  /// (position, momentum) and `dydx` their time derivatives. The magnetic field argument is
  /// ignored by both, which is why `G4KM_DummyField` exists at all.
  __host__ __device__ void evaluate(const double y[], double dydx[],
                                    const NuclearDensity& density) const {
    const double y_mod = std::sqrt(y[0] * y[0] + y[1] * y[1] + y[2] * y[2]);
    const double e =
        std::sqrt(the_mass * the_mass + y[3] * y[3] + y[4] * y[4] + y[5] * y[5]);
    const double c = u::c_light<double>();
    dydx[0] = c * y[3] / e;
    dydx[1] = c * y[4] / e;
    dydx[2] = c * y[5] / e;

    const Vec3d pos{y[0], y[1], y[2]};
    double deriv = 0.0;
    double sign = 1.0;
    if (kind == kNucleonEq) {
      const double dens = density.density(pos);
      // `if (density > 0)` - a position outside the density's support gives deriv = 0 and no
      // force, and the guard is there because A13(0) is 0 and the division would be infinite.
      if (dens > 0.0) {
        deriv = (factor / the_mass) / data::g4pow_a13<double>(dens) * density.deriv(pos);
      }
      sign = 1.0;    // dydx[3] = +deriv*y[0]/yMod*c_light
    } else {
      deriv = factor * density.deriv(pos);
      sign = -1.0;   // dydx[3] = -deriv*y[0]/yMod*c_light
    }
    dydx[3] = (y_mod == 0.0) ? 0.0 : sign * deriv * y[0] / y_mod * c;
    dydx[4] = (y_mod == 0.0) ? 0.0 : sign * deriv * y[1] / y_mod * c;
    dydx[5] = (y_mod == 0.0) ? 0.0 : sign * deriv * y[2] / y_mod * c;
  }
};

/// `G4KM_NucleonEqRhs`'s constructor plus `SetMass`, which `G4RKPropagation::Init` calls
/// immediately after it. `A23(x)` is `A13(x)` squared, G4Pow's definition.
__host__ __device__ inline EqRhs make_nucleon_eq(int a, double mass) {
  EqRhs eq;
  eq.kind = kNucleonEq;
  eq.the_mass = mass;
  const double pi2 = u::pi<double>() * u::pi<double>();
  eq.factor = u::hbarc<double>() * u::hbarc<double>() *
              data::g4pow_a23<double>(3.0 * pi2 * static_cast<double>(a)) / 3.0;
  return eq;
}

/// `G4KM_OpticalEqRhs::SetFactor(mass, opticalParameter)`. `A` is the nucleus's mass number and
/// the `* A` at the end is the same "turn a density normalised to 1 into a number density"
/// factor `G4FermiMomentum::GetFermiMomentum` applies - see `fermi_momentum.cuh`.
__host__ __device__ inline EqRhs make_optical_eq(int a, int z, double mass,
                                                 double optical_parameter) {
  EqRhs eq;
  eq.kind = kOpticalEq;
  eq.the_mass = mass;
  const double binding_energy = deex::binding_energy(a, z);
  const double nucleus_mass = static_cast<double>(z) * u::proton_mass_c2<double>() +
                              static_cast<double>(a - z) * u::neutron_mass_c2<double>() +
                              binding_energy;
  const double reduced_mass = mass * nucleus_mass / (mass + nucleus_mass);
  const double nucleon_mass =
      (u::proton_mass_c2<double>() + u::neutron_mass_c2<double>()) / 2.0;
  eq.factor = 2.0 * u::pi<double>() * u::hbarc<double>() * u::hbarc<double>() *
              (1.0 + mass / nucleon_mass) * optical_parameter / reduced_mass *
              static_cast<double>(a);
  return eq;
}

// =============================================================================================
// G4ClassicalRK4 and G4MagErrorStepper
// =============================================================================================

/// `G4FieldTrack::ncompSVEC`. The driver copies twelve components; the stepper integrates six.
inline constexpr int kNcompSVEC = 12;
inline constexpr int kNIntegrationVars = 6;

/// G4ClassicalRK4::DumbStepper - the textbook fourth-order Runge-Kutta step, NRC p.712.
///
/// Note the two lines Geant4 fuses: `yt[i] = yIn[i] + h*dydxm[i]` and `dydxm[i] += dydxt[i]`
/// in the same loop, so `dydxm` becomes (K2+K3)/h AFTER it has been used to build `yt`. Order
/// matters and is preserved. `yt[7]` and `yOut[7]` are copied before the loop because the time
/// component is not integrated.
__host__ __device__ inline void rk4_dumb_stepper(const double y_in[], const double dydx[],
                                                 double h, double y_out[],
                                                 const EqRhs& eq,
                                                 const NuclearDensity& density) {
  const int nvar = kNIntegrationVars;
  double dydxm[kNcompSVEC] = {0.0};
  double dydxt[kNcompSVEC] = {0.0};
  double yt[kNcompSVEC] = {0.0};
  const double hh = h * 0.5;
  const double h6 = h / 6.0;

  yt[7] = y_in[7];
  y_out[7] = y_in[7];

  for (int i = 0; i < nvar; ++i) { yt[i] = y_in[i] + hh * dydx[i]; }
  eq.evaluate(yt, dydxt, density);

  for (int i = 0; i < nvar; ++i) { yt[i] = y_in[i] + hh * dydxt[i]; }
  eq.evaluate(yt, dydxm, density);

  for (int i = 0; i < nvar; ++i) {
    yt[i] = y_in[i] + h * dydxm[i];
    dydxm[i] += dydxt[i];
  }
  eq.evaluate(yt, dydxt, density);

  for (int i = 0; i < nvar; ++i) {
    y_out[i] = y_in[i] + h6 * (dydx[i] + dydxt[i] + 2.0 * dydxm[i]);
  }
}

/// G4MagErrorStepper::Stepper - two half steps, one full step, Richardson extrapolation.
///
/// `correction = 1/((1 << IntegratorOrder()) - 1) = 1/15`. The extrapolated answer is written
/// back into `y_output`, so the accepted step is the two-half-step result PLUS a fifteenth of
/// its difference from the full step, and `y_error` is that difference.
__host__ __device__ inline void rk4_stepper(const double y_input[], const double dydx[],
                                            double hstep, double y_output[], double y_error[],
                                            const EqRhs& eq, const NuclearDensity& density) {
  const int nvar = kNIntegrationVars;
  const int maxvar = kNcompSVEC;
  const double correction = 1.0 / static_cast<double>((1 << 4) - 1);

  double y_initial[kNcompSVEC] = {0.0};
  double y_middle[kNcompSVEC] = {0.0};
  double y_one_step[kNcompSVEC] = {0.0};
  double dydx_mid[kNcompSVEC] = {0.0};

  for (int i = 0; i < nvar; ++i) { y_initial[i] = y_input[i]; }
  y_initial[7] = y_input[7];
  y_middle[7] = y_input[7];
  y_one_step[7] = y_input[7];
  for (int i = nvar; i < maxvar; ++i) { y_output[i] = y_input[i]; }

  const double half_step = hstep * 0.5;

  rk4_dumb_stepper(y_initial, dydx, half_step, y_middle, eq, density);
  eq.evaluate(y_middle, dydx_mid, density);
  rk4_dumb_stepper(y_middle, dydx_mid, half_step, y_output, eq, density);

  rk4_dumb_stepper(y_initial, dydx, hstep, y_one_step, eq, density);
  for (int i = 0; i < nvar; ++i) {
    y_error[i] = y_output[i] - y_one_step[i];
    y_output[i] += y_error[i] * correction;
  }
}

// =============================================================================================
// G4MagInt_Driver
// =============================================================================================

/// The driver's fixed parameters for a G4ClassicalRK4 of order 4. All five are derived in
/// `ReSetParameters`/`ComputeAndSetErrcon` and none is ever changed on this path.
struct RkDriverParams {
  double safety = 0.9;
  double pshrnk = -1.0 / 4.0;
  double pgrow = -1.0 / (1.0 + 4.0);
  double max_stepping_increase = 5.0;
  double h_min = 0.0;    ///< fMinimumStep, set by FieldTransport to 1e-25 s
  int max_no_steps = 250 / 4;
  double smallest_fraction = 1.0e-12;

  __host__ __device__ double errcon() const {
    return std::pow(max_stepping_increase / safety, 1.0 / pgrow);
  }
};

/// G4MagInt_Driver::OneGoodStep, NRC's rkqs.
///
/// The spin branch is omitted because `hasSpin` is false here (the file header says why), and
/// the `magvel_sq == 0` JustWarning branch is kept as `errvel_sq = sumerr_sq` - a momentum of
/// exactly zero is reachable for a nucleon whose Fermi momentum was zeroed.
__host__ __device__ inline void rk_one_good_step(double y[], const double dydx[], double& x,
                                                 double htry, double eps_rel_max, double& hdid,
                                                 double& hnext, const EqRhs& eq,
                                                 const NuclearDensity& density,
                                                 const RkDriverParams& par) {
  double errmax_sq = 0.0;
  double h = htry;
  const double inv_eps_vel_sq = 1.0 / (eps_rel_max * eps_rel_max);
  double yerr[kNcompSVEC] = {0.0};
  double ytemp[kNcompSVEC] = {0.0};
  const int max_trials = 100;

  for (int iter = 0; iter < max_trials; ++iter) {
    rk4_stepper(y, dydx, h, ytemp, yerr, eq, density);
    const double eps_pos = eps_rel_max * ((h > par.h_min) ? h : par.h_min);
    const double inv_eps_pos_sq = 1.0 / (eps_pos * eps_pos);

    double errpos_sq = yerr[0] * yerr[0] + yerr[1] * yerr[1] + yerr[2] * yerr[2];
    errpos_sq *= inv_eps_pos_sq;

    const double magvel_sq = y[3] * y[3] + y[4] * y[4] + y[5] * y[5];
    const double sumerr_sq = yerr[3] * yerr[3] + yerr[4] * yerr[4] + yerr[5] * yerr[5];
    double errvel_sq = (magvel_sq > 0.0) ? (sumerr_sq / magvel_sq) : sumerr_sq;
    errvel_sq *= inv_eps_vel_sq;
    errmax_sq = (errpos_sq > errvel_sq) ? errpos_sq : errvel_sq;

    if (errmax_sq <= 1.0) { break; }

    const double htemp = par.safety * h * std::pow(errmax_sq, 0.5 * par.pshrnk);
    h = (htemp >= 0.1 * h) ? htemp : (0.1 * h);
    const double xnew = x + h;
    if (xnew == x) { break; }   // "Stepsize underflow in Stepper", a JustWarning
  }

  const double ec = par.errcon();
  if (errmax_sq > ec * ec) {
    hnext = par.safety * h * std::pow(errmax_sq, 0.5 * par.pgrow);
  } else {
    hnext = par.max_stepping_increase * h;
  }
  hdid = h;
  x += hdid;

  for (int k = 0; k < kNIntegrationVars; ++k) { y[k] = ytemp[k]; }
}

/// G4MagInt_Driver::ComputeNewStepSize, which forwards to
/// `ComputeNewStepSize_WithoutReductionLimit` - the "legacy behaviour" the source labels, with
/// the limited version written out beside it and returned by nobody.
__host__ __device__ inline double rk_compute_new_step_size(double err_max_norm,
                                                           double hstep_current,
                                                           const RkDriverParams& par) {
  if (err_max_norm > 1.0) {
    return par.safety * hstep_current * std::pow(err_max_norm, par.pshrnk);
  }
  if (err_max_norm > 0.0) {
    return par.safety * hstep_current * std::pow(err_max_norm, par.pgrow);
  }
  return par.max_stepping_increase * hstep_current;
}

/// G4MagInt_Driver::QuickAdvance - one Stepper call with no error control.
///
/// It is NOT unreachable. `AccurateAdvance` takes it when `h <= fMinimumStep`, and although the
/// cascade's time steps are twelve orders of magnitude above `fMinimumStep`, the clamp
/// `if (x + h > x2) h = x2 - x` can leave a step of a few times 1e-17 ns when the previous step
/// landed just short of the interval's end. Measured: eight of the 55 (nucleus, case) pairs in
/// `bic_rk.csv` reach it, and refusing it there made the port free-transport where Geant4
/// integrated. The file header used to claim it could not happen; the oracle said otherwise.
///
/// The error it returns is not used by `AccurateAdvance` for anything except the next step
/// size, and the next step is the last one. `DistChord` is computed by Geant4 and discarded.
///
/// The G4FieldTrack it round-trips through is built with a rest mass of ZERO
/// (`G4FieldTrack(G4ThreeVector(0,0,0), G4ThreeVector(0,0,0), 0., 0., 0., 0.)`), so the
/// kinetic energy `LoadFromArray` recomputes is `|p|` rather than `sqrt(p^2+m^2)-m`. That
/// component is not integrated and not read, which is why it does not matter - but it is why
/// `y[6]` comes back as a momentum here and as a kinetic energy everywhere else.
__host__ __device__ inline void rk_quick_advance(double y[], double& x, double h,
                                                 const double dydx[], double& dyerr,
                                                 const EqRhs& eq,
                                                 const NuclearDensity& density) {
  double yerr[kNcompSVEC] = {0.0};
  double yout[kNcompSVEC] = {0.0};
  rk4_stepper(y, dydx, h, yout, yerr, eq, density);

  const double vel_mag_sq = yout[3] * yout[3] + yout[4] * yout[4] + yout[5] * yout[5];
  const double inv_vel_mag_sq = 1.0 / vel_mag_sq;
  const double dyerr_pos_sq = yerr[0] * yerr[0] + yerr[1] * yerr[1] + yerr[2] * yerr[2];
  const double dyerr_mom_sq = yerr[3] * yerr[3] + yerr[4] * yerr[4] + yerr[5] * yerr[5];
  const double dyerr_mom_rel_sq = dyerr_mom_sq * inv_vel_mag_sq;

  if (dyerr_pos_sq > (dyerr_mom_rel_sq * h * h)) {
    dyerr = std::sqrt(dyerr_pos_sq);
  } else {
    dyerr = std::sqrt(dyerr_mom_rel_sq) * h;
  }

  // `LoadFromArray(yarrout, 6)` then `DumpToArray(y)`: the six integrated components survive
  // and 6..11 are zeroed, except that DumpToArray writes the recomputed kinetic energy into
  // y[6].
  for (int i = 0; i < kNIntegrationVars; ++i) { y[i] = yout[i]; }
  const double p_mag_sq = y[3] * y[3] + y[4] * y[4] + y[5] * y[5];
  y[6] = (p_mag_sq > 0.0) ? (p_mag_sq / std::sqrt(p_mag_sq)) : 0.0;  // rest mass is 0 here
  for (int i = 7; i < kNcompSVEC; ++i) { y[i] = 0.0; }
  x += h;
}

/// G4MagInt_Driver::AccurateAdvance, NRC's odeint.
///
/// `y` is the twelve-component state in and out, `x` the curve-length parameter (a TIME here).
/// Returns false when the integration did not reach `x1 + hstep`, which is what
/// `FieldTransport` turns into "cannot track this particle" and BIC into a free transport.
///
/// The `h <= fMinimumStep` arm - `QuickAdvance` - IS reached and is transcribed, not refused;
/// `rk_quick_advance`'s own comment says how often and why the first draft had it the other way.
struct RkAdvanceReport {
  /// How many sub-steps took the `h <= fMinimumStep` arm. Not a refusal - it is transcribed -
  /// but it is counted, because the file header claimed for one draft that it could not happen
  /// and the oracle disagreed on eight of 55 cases.
  int n_quick_advance = 0;
  /// `h == 0.0` inside that arm, which is a FatalException in Geant4.
  bool step_became_zero = false;
  /// The track's species has a field in Geant4's map and not in this one - see
  /// `RkPropagation::is_refused_field_species`. Free-streaming it would be an approximation.
  bool refused_field_species = false;
  int refused_pdg = 0;
  int n_steps = 0;
};

__host__ __device__ inline bool rk_accurate_advance(double y[], double& curve_length,
                                                    double hstep, double eps, const EqRhs& eq,
                                                    const NuclearDensity& density,
                                                    const RkDriverParams& par,
                                                    RkAdvanceReport& rep) {
  if (hstep <= 0.0) {
    // hstep == 0 is a JustWarning and returns success without moving; hstep < 0 is
    // EventMustBeAborted and returns false.
    return hstep == 0.0;
  }

  double ystart[kNcompSVEC];
  for (int i = 0; i < kNcompSVEC; ++i) { ystart[i] = y[i]; }

  const double start_curve_length = curve_length;
  const double x1 = start_curve_length;
  const double x2 = x1 + hstep;

  // `hinitial` is 0 from FieldTransport's three-argument call, so h defaults to the full
  // interval.
  double h = hstep;
  double x = x1;
  double yw[kNcompSVEC];
  for (int i = 0; i < kNcompSVEC; ++i) { yw[i] = ystart[i]; }

  bool last_step = false;
  int nstp = 1;
  double hdid = 0.0, hnext = 0.0;

  do {
    double dydx[kNcompSVEC] = {0.0};
    eq.evaluate(yw, dydx, density);

    if (h > par.h_min) {
      rk_one_good_step(yw, dydx, x, h, eps, hdid, hnext, eq, density, par);
    } else {
      // The QuickAdvance arm. `h == 0.0` here is a FatalException in Geant4; a kernel cannot
      // throw, so it is reported and the integration stops - which is what the exception does.
      if (h == 0.0) {
        rep.step_became_zero = true;
        return false;
      }
      ++rep.n_quick_advance;
      double dyerr_len = 0.0;
      rk_quick_advance(yw, x, h, dydx, dyerr_len, eq, density);
      const double dyerr = dyerr_len / h;
      hdid = h;
      hnext = rk_compute_new_step_size(dyerr / eps, h, par);
    }

    // `endPointDist >= hdid*(1+perMillion)` and its perThousand sibling count bad steps and
    // print nothing. Both are omitted: they change no value. See the file header.

    if ((h < eps * hstep) || (h < par.smallest_fraction * start_curve_length)) {
      last_step = true;
    } else {
      h = (std::abs(hnext) <= par.h_min) ? par.h_min : hnext;
      if (x + h > x2) { h = x2 - x; }
      if (h == 0.0) { last_step = true; }
    }
  } while (((++nstp) <= par.max_no_steps) && (x < x2) && (!last_step));

  rep.n_steps = nstp;
  bool succeeded = (x >= x2);

  // `LoadFromArray(yEnd, fNoIntegrationVariables)` zeroes components 6..11, which is why the
  // kinetic energy, both times and the polarization come back as zero.
  for (int i = 0; i < kNIntegrationVars; ++i) { y[i] = yw[i]; }
  for (int i = kNIntegrationVars; i < kNcompSVEC; ++i) { y[i] = 0.0; }
  curve_length = x;

  if (nstp > par.max_no_steps) { succeeded = false; }
  return succeeded;
}

// =============================================================================================
// G4RKPropagation
// =============================================================================================

/// The field and equation for one species, as `G4RKPropagation::Init`'s two maps hold them.
/// `has_field` is the `fieldIter != theFieldMap->end()` test that decides whether a particle is
/// propagated in the field at all or streamed in a straight line.
struct SpeciesField {
  bool has_field = false;
  bool is_nucleon_field = false;
  NucleonField nucleon;
  PionField pion;
  EqRhs eq;

  __host__ __device__ double field(const Vec3d& pos, const NuclearDensity& d) const {
    return is_nucleon_field ? nucleon.field(pos) : pion.field(pos, d);
  }
  __host__ __device__ double barrier() const {
    return is_nucleon_field ? nucleon.get_barrier() : pion.get_barrier();
  }
};

/// The propagator: the nucleus it was `Init`-ed on, and the four species fields QBBC's BIC can
/// reach. `theOuterRadius` is `GetOuterRadius()` at Init time and does not follow the nucleus.
struct RkPropagation {
  double outer_radius = 0.0;
  double nucleus_mass = 0.0;   ///< theNucleus->GetMass(), read by every momentum correction
  NuclearDensity density;
  Vec3d momentum_transfer{0.0, 0.0, 0.0};  ///< theMomentumTranfer (sic), summed over Transport

  SpeciesField proton, neutron, pion_plus, pion_minus, pion_zero;

  /// The `theFieldMap->find(encoding)` lookup, restricted to the species this package reaches.
  ///
  /// A species Geant4 has NO field for is free-streamed, and returning null here reproduces
  /// that. But Geant4's map holds twelve species and this holds five, so the nine it does not
  /// hold split into two kinds, and they must not be confused:
  ///
  ///   * a resonance, a hyperon Geant4 also has no field for, a deuteron - free-streamed by
  ///     both, and `refused_field` stays false.
  ///   * an anti-proton, a kaon or a sigma - fielded by Geant4 and refused here. Free-streaming
  ///     one would be a silent approximation, so `is_refused_field_species` says so and the
  ///     caller sets its refusal. `nuclear_field.cuh`'s header says why they are out of scope.
  __host__ __device__ static bool is_refused_field_species(int pdg) {
    switch (pdg) {
      case -2212:  // anti-proton
      case 321: case -321: case 311:  // K+, K-, K0
      case 3222: case 3112: case 3212:  // Sigma+, Sigma-, Sigma0
        return true;
      default:
        return false;
    }
  }

  __host__ __device__ const SpeciesField* find_field(int pdg) const {
    switch (pdg) {
      case 2212: return &proton;
      case 2112: return &neutron;
      case 211: return &pion_plus;
      case -211: return &pion_minus;
      case 111: return &pion_zero;
      default: return nullptr;
    }
  }

  /// `G4RKPropagation::GetSphereIntersectionTimes(const G4KineticTrack*, t1, t2)`.
  ///
  /// The sphere is `theOuterRadius + 3 fermi` - the "safety of 3 fermi" the source names - and
  /// NOT the outer radius, so a particle is "inside" out to three femtometres beyond the
  /// outermost nucleon. `speed = p/E` is dimensionless and the division by `c_light` at the end
  /// turns a length over a speed into a time.
  ///
  /// `sqrtArg <= 0` returns false, which callers read as "misses the nucleus". Note that a
  /// particle exactly on the surface gives sqrtArg == 0 and is reported as a MISS.
  __host__ __device__ bool sphere_intersection_times(const KineticTrack& kt, double& t1,
                                                     double& t2) const {
    const double radius = outer_radius + 3.0 * deex::fermi();
    const LorentzVector& p = kt.tracking_momentum();
    const Vec3d speed{p.v.x / p.e, p.v.y / p.e, p.v.z / p.e};
    const double scalar_prod = g4gpu::dot(kt.position, speed);
    const double speed_mag2 = g4gpu::mag2(speed);
    const double sqrt_arg = scalar_prod * scalar_prod -
                            speed_mag2 * (g4gpu::mag2(kt.position) - radius * radius);
    if (sqrt_arg <= 0.0) { return false; }
    const double c = u::c_light<double>();
    t1 = (-scalar_prod - std::sqrt(sqrt_arg)) / speed_mag2 / c;
    t2 = (-scalar_prod + std::sqrt(sqrt_arg)) / speed_mag2 / c;
    return true;
  }

  /// The `(radius, position, momentum)` overload, which takes the radius rather than adding the
  /// 3 fermi. Nothing in 11.1.1 calls it; transcribed because the two are easy to confuse and
  /// the difference is three femtometres of nucleus.
  __host__ __device__ static bool sphere_intersection_times(double radius, const Vec3d& pos,
                                                            const LorentzVector& momentum,
                                                            double& t1, double& t2) {
    const Vec3d speed{momentum.v.x / momentum.e, momentum.v.y / momentum.e,
                      momentum.v.z / momentum.e};
    const double scalar_prod = g4gpu::dot(pos, speed);
    const double speed_mag2 = g4gpu::mag2(speed);
    const double sqrt_arg =
        scalar_prod * scalar_prod - speed_mag2 * (g4gpu::mag2(pos) - radius * radius);
    if (sqrt_arg <= 0.0) { return false; }
    const double c = u::c_light<double>();
    t1 = (-scalar_prod - std::sqrt(sqrt_arg)) / speed_mag2 / c;
    t2 = (-scalar_prod + std::sqrt(sqrt_arg)) / speed_mag2 / c;
    return true;
  }

  /// `G4RKPropagation::FreeTransport` - a straight line at the track's own velocity.
  /// `pos += timeStep*c_light/E * p`, in that grouping.
  __host__ __device__ static void free_transport(KineticTrack& kt, double time_step) {
    const LorentzVector& p = kt.tracking_momentum();
    const double s = time_step * u::c_light<double>() / p.e;
    kt.position = kt.position + s * p.v;
  }

  /// `G4RKPropagation::FieldTransport`.
  ///
  /// Builds the twelve-component state from the track, integrates for `timeStep`, and then
  /// applies the momentum correction: the momentum the integration took from the particle is
  /// given to the nucleus, and the particle is boosted into the moving nucleus's frame by
  /// `transfer/sqrt(transfer^2 + M^2)`. `theMomentumTranfer` accumulates the difference
  /// BEFORE and AFTER that boost, not the integration's own change - so it is the transfer the
  /// boost did not absorb.
  ///
  /// Note what is NOT carried across: `G4FieldTrack` is built with the track's kinetic energy
  /// and REST MASS `GetActualMass()`, and the energy that comes back is recomputed as
  /// `sqrt(p^2 + GetActualMass()^2)` - so the particle is put back exactly on its own mass
  /// shell and the potential energy it gained or lost inside the nucleus is discarded here and
  /// re-applied by `Transport` at the surface. The field appears in the trajectory and not in
  /// the energy.
  __host__ __device__ bool field_transport(KineticTrack& kt, double time_step,
                                           const EqRhs& eq, RkAdvanceReport& rep) {
    RkDriverParams par;
    // hMin = 1.0e-25*second. CLHEP's second is 1e9 ns, so this is 1e-16 ns.
    par.h_min = 1.0e-25 * u::s<double>();

    const double mass = kt.actual_mass();
    const LorentzVector& tm = kt.tracking_momentum();
    const double kin = tm.e - mass;
    const Vec3d dir = g4gpu::normalize(tm.v);
    const double p_mag = std::sqrt(kin * kin + 2.0 * mass * kin);

    double y[kNcompSVEC] = {0.0};
    y[0] = kt.position.x;
    y[1] = kt.position.y;
    y[2] = kt.position.z;
    y[3] = p_mag * dir.x;
    y[4] = p_mag * dir.y;
    y[5] = p_mag * dir.z;
    y[6] = kin;
    double curve_length = 0.0;

    const double eps = 0.01;
    if (!rk_accurate_advance(y, curve_length, time_step, eps, eq, density, par, rep)) {
      return false;
    }

    const Vec3d new_p{y[3], y[4], y[5]};
    const Vec3d transfer = tm.v - new_p;
    const Vec3d boost = (1.0 / std::sqrt(g4gpu::mag2(transfer) +
                                         nucleus_mass * nucleus_mass)) *
                        transfer;

    kt.position = Vec3d{y[0], y[1], y[2]};
    LorentzVector mom(new_p, std::sqrt(g4gpu::mag2(new_p) + mass * mass));
    mom = boost_by_rotation(mom, boost);
    momentum_transfer = momentum_transfer + (kt.tracking_momentum().v - mom.v);
    kt.set_tracking_momentum(mom);
    return true;
  }

  /// The four identical momentum corrections `Transport` applies at the nuclear surface: put
  /// the particle at energy `newE` on its own mass shell, then boost into the frame of the
  /// nucleus that took up the difference. Factored out because Geant4 writes it out four times
  /// and a transcription that got one of the four wrong would be very hard to see.
  __host__ __device__ void surface_correction(KineticTrack& kt, double new_e) const {
    const double mass = kt.actual_mass();
    const double new_p = std::sqrt(new_e * new_e - mass * mass);
    LorentzVector n4(new_p * g4gpu::normalize(kt.tracking_momentum().v), new_e);
    const Vec3d transfer = kt.tracking_momentum().v - n4.v;
    const Vec3d boost =
        (1.0 / std::sqrt(g4gpu::mag2(transfer) + nucleus_mass * nucleus_mass)) * transfer;
    n4 = boost_by_rotation(n4, boost);
    kt.set_tracking_momentum(n4);
  }

  /// `G4RKPropagation::Transport(active, dummy, timeStep)` for ONE track.
  ///
  /// Geant4 loops over the active list and resets `theMomentumTranfer` once at the top; the
  /// reset is the caller's job here so that one track can be stepped for a test. The control
  /// flow is the whole content of the method and it is easier to read as the six outcomes it
  /// has than as the nest it is written as:
  ///
  ///   * no sphere intersection            -> miss_nucleus, no move
  ///   * no field for this species         -> free transport, then inside/gone_out/miss by
  ///                                          comparing the step to t_enter and t_leave
  ///   * outside and the step is too short -> free transport, still outside
  ///   * outside and the step reaches in   -> free transport to the surface, pay the field,
  ///                                          and if that costs more than its kinetic energy,
  ///                                          free-transport 1.1*t_leave past the nucleus and
  ///                                          call it a miss
  ///   * inside                            -> field transport (falling back to free transport
  ///                                          if the integration failed)
  ///   * inside and leaving                -> transport to the boundary, correct for the field
  ///                                          difference across the step, add the barrier, and
  ///                                          go out - or be CAPTURED if a nucleon cannot pay,
  ///                                          or gone_out anyway if it is not a nucleon
  ///
  /// The last of those is where BIC's `captured` state comes from, and it is why a nucleon that
  /// cannot climb out of the well ends up in `theCapturedList` and in the residual's exciton
  /// count rather than in the final state.
  template <typename Rep>
  __host__ __device__ void transport_one(KineticTrack& kt, double time_step, Rep& rep) {
    double curr = time_step;
    const SpeciesField* sf = find_field(kt.pdg);

    if (is_refused_field_species(kt.pdg)) {
      rep.refused_field_species = true;
      rep.refused_pdg = kt.pdg;
      return;
    }

    double t_enter = 0.0, t_leave = 0.0;
    if (!sphere_intersection_times(kt, t_enter, t_leave)) {
      kt.state = kMissNucleus;
      return;
    }

    if (sf == nullptr || !sf->has_field) {
      if (curr == DBL_MAX) { curr = t_leave * 1.05; }
      free_transport(kt, curr);
      if (curr >= t_leave) {
        kt.state = (kt.state == kInside) ? kGoneOut : kMissNucleus;
      } else if (kt.state == kOutside && curr >= t_enter) {
        kt.state = kInside;
      }
      return;
    }

    if (t_enter > 0.0) {
      if (t_enter > curr) {
        free_transport(kt, curr);
        return;
      }
      free_transport(kt, t_enter);
      curr -= t_enter;
      t_leave -= t_enter;
      const double new_e = kt.tracking_momentum().e - sf->field(kt.position, density);
      if (new_e <= kt.actual_mass()) {
        free_transport(kt, 1.1 * t_leave);
        kt.state = kMissNucleus;
        return;
      }
      surface_correction(kt, new_e);
      kt.state = kInside;
    }

    bool is_exiting = false;
    if (curr > t_leave) {
      curr = t_leave;
      is_exiting = true;
    }

    if (curr > 0.0 && !field_transport(kt, curr, sf->eq, rep)) { free_transport(kt, curr); }

    // `G4double t_in=-1, t_out=0;  // set onto boundary.` and then
    //
    //     if (is_exiting || (GetSphereIntersectionTimes(kt, t_in, t_out) && t_in<0 && t_out<=0))
    //
    // **The short circuit is load-bearing.** When `is_exiting` is true the intersection is
    // never computed, so `t_in` and `t_out` keep the -1 and 0 the declaration gave them - which
    // is what the comment "set onto boundary" means - and the block below then takes its
    // `t_in < 0 && t_out >= 0` branch with `t_out == 0`, i.e. `FreeTransport(kt, 0)`: no move,
    // and a field difference evaluated at one point against itself, which is zero. Calling the
    // intersection unconditionally instead overwrites both, moves the track by the real `t_out`
    // and applies a real field difference. Measured: doing that put the C12 neutron trajectory
    // 2.8 times its own x off the oracle.
    double t_in = -1.0, t_out = 0.0;
    if (is_exiting || (sphere_intersection_times(kt, t_in, t_out) && t_in < 0.0 &&
                       t_out <= 0.0)) {
      if (t_in < 0.0 && t_out >= 0.0) {
        const Vec3d save_pos = kt.position;
        free_transport(kt, t_out);
        double new_e = kt.tracking_momentum().e;
        // The guard is Geant4's own FixMe: outside the nucleus GetField is 0, and adding a
        // zero here would subtract the barrier twice. Both fields have to be non-zero for the
        // difference to be applied at all.
        if (std::abs(sf->field(save_pos, density)) > 0.0 &&
            std::abs(sf->field(kt.position, density)) > 0.0) {
          new_e += sf->field(save_pos, density) - sf->field(kt.position, density);
        }
        if (new_e < kt.actual_mass()) {
          kt.state = (kt.pdg == 2212 || kt.pdg == 2112) ? kCaptured : kGoneOut;
          return;
        }
        surface_correction(kt, new_e);
      }
      const double new_e = kt.tracking_momentum().e + sf->field(kt.position, density);
      if (new_e < kt.actual_mass()) {
        kt.state = (kt.pdg == 2212 || kt.pdg == 2112) ? kCaptured : kGoneOut;
        return;
      }
      surface_correction(kt, new_e);
      kt.state = kGoneOut;
    }
  }
};

/// `G4RKPropagation::Init(nucleus)`, restricted to the five species QBBC's BIC reaches.
///
/// `theOuterRadius = GetOuterRadius()` is read once, so a nucleus that loses nucleons later
/// keeps the radius it was Init-ed with. BIC calls `Init` twice - once in `ApplyYourself` and
/// again at the top of `Propagate` - on the same nucleus, so the two agree; a caller that
/// wounds the nucleus in between would not notice.
///
/// The two field tables are the caller's storage. `proton_table` and `neutron_table` must each
/// hold `kMaxFieldTable` doubles; `NucleonField::overflow` reports a nucleus whose outer radius
/// needs more.
__host__ __device__ inline RkPropagation make_rk_propagation(const Nucleus3D& nuc,
                                                             NucleusReport& nrep,
                                                             double* proton_table,
                                                             double* neutron_table,
                                                             int table_capacity) {
  RkPropagation p;
  p.outer_radius = nuc.outer_radius();
  p.nucleus_mass = nuc.mass(nrep);
  p.density = nuc.density;
  const int a = nuc.my_a;
  const int z = nuc.my_z;

  p.proton.has_field = true;
  p.proton.is_nucleon_field = true;
  p.proton.nucleon = make_nucleon_field(true, nuc.density, nuc.fermi, a, z, p.outer_radius,
                                        proton_table, table_capacity);
  p.proton.eq = make_nucleon_eq(a, u::proton_mass_c2<double>());

  p.neutron.has_field = true;
  p.neutron.is_nucleon_field = true;
  p.neutron.nucleon = make_nucleon_field(false, nuc.density, nuc.fermi, a, z, p.outer_radius,
                                         neutron_table, table_capacity);
  p.neutron.eq = make_nucleon_eq(a, u::neutron_mass_c2<double>());

  // The three pion fields, and the optical equations whose `opticalCoeff` argument is
  // `theFieldMap[pdg]->GetCoeff()` - the coefficient the field was constructed with.
  p.pion_plus.has_field = true;
  p.pion_plus.pion = make_pion_field(kPionPlusField, a, z, p.outer_radius);
  p.pion_plus.eq = make_optical_eq(a, z, pdg_mass_pion_charged(), field_coeff_pion());

  p.pion_minus.has_field = true;
  p.pion_minus.pion = make_pion_field(kPionMinusField, a, z, p.outer_radius);
  p.pion_minus.eq = make_optical_eq(a, z, pdg_mass_pion_charged(), field_coeff_pion());

  p.pion_zero.has_field = true;
  p.pion_zero.pion = make_pion_field(kPionZeroField, a, z, p.outer_radius);
  p.pion_zero.eq = make_optical_eq(a, z, pdg_mass_pion_zero(), field_coeff_pion());

  return p;
}

}  // namespace g4gpu::bic

#endif
