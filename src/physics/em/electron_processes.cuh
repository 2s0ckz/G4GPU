// Electron and positron transport by condensed history.
//
// dE/dx     : Berger-Seltzer, transcribed from G4MollerBhabhaModel::ComputeDEDXPerVolume.
//             RESTRICTED to the material production cut - transfers above it produce an
//             explicit delta ray instead of depositing locally.
// Delta rays: G4MollerBhabhaModel cross section and sampler, transcribed verbatim.
// Range     : the dE/dx, range and inverse-range tables Geant4 transports on, built by
//             G4LossTableBuilder's own algorithm on G4EmParameters' own grid - 100 eV to
//             100 TeV at 7 bins per decade, cubic spline, 100 midpoint sub-steps per bin.
//             See `RangeTable` below and docs/RISK.md V64 for the ceiling this replaced.
// MSC       : `em/urban_msc.cuh` below 100 MeV and `em/wentzel_msc.cuh` above it, which is
//             where G4EmStandardPhysics switches models. The Highland form at the bottom of
//             this file is used by nothing in the transport.
// Brems     : G4SeltzerBergerModel below 1 GeV and G4eBremsstrahlungRelModel above, both
//             through `data/brems_data.cuh`; explicit photons, plus the sub-cut radiative
//             loss which is part of the dE/dx table below.
// Annihil.  : positrons at rest emit two back-to-back 511 keV photons.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "core/particle.cuh"
#include "core/rng.cuh"
#include "core/units.cuh"
#include "core/vec3.cuh"
#include "data/brems_data.cuh"
#include "data/g4spline.hh"
#include "data/materials.cuh"
#include "physics/em/brems_rel.cuh"

namespace g4gpu::em {

/// Electron rest mass, re-exported so the stepper need not include the units header.
template <typename real_t> __host__ __device__ constexpr real_t units_me() {
  return units::electron_mass_c2<real_t>();
}

/// Electrons below this deposit their remaining energy on the spot and stop.
///
/// This is `G4VEnergyLossProcess::lowestKinEnergy`, which for an e+- is
/// `G4EmParameters::LowestElectronEnergy()` = 1 keV (G4VEnergyLossProcess.cc:101). It is read in
/// TWO places of `AlongStepDoIt` and `step_lepton` now uses it in both: the "stopping" test at
/// :812 that takes the whole kinetic energy, and the energy balance at :914 that does the same to
/// whatever a fluctuated step left behind.
template <typename real_t> __host__ __device__ constexpr real_t kElectronTrackingCut() {
  return real_t(1e-3);  // 1 keV
}

/// `G4EmParameters::LinearLossLimit`, the fraction of the kinetic energy above which
/// `G4VEnergyLossProcess::AlongStepDoIt` stops trusting `length * dE/dx` and inverts the range
/// table instead (G4VEnergyLossProcess.cc:830).
///
/// 0.01 for a lepton, and NOT a global: `G4ionIonisation`'s constructor calls
/// `SetLinearLossLimit(0.02)`, so every species that process is registered for uses twice this -
/// which is why `step_hadron` computes its own per-species constant rather than calling here.
/// `G4eIonisation` does not override the parameter, so an e+- gets `G4EmParameters`' own value.
template <typename real_t> __host__ __device__ constexpr real_t kLinearLossLimit() {
  return real_t(0.01);
}

/// 2*pi*m_e*c^2*r_e^2 in MeV*mm^2, the Berger-Seltzer and Bethe-Bloch ionisation prefactor.
///
/// `units::twopi_mc2_rcl2` and not the product of the three factors. CLHEP derives this
/// quantity itself and pins it, and recomputing it from `twopi * m_e * r_e * r_e` gives a
/// different last bit.
///
/// This used to be defined *twice* - here as `constexpr` and again in hadron_ionisation.cuh as
/// `inline`, each computing the product its own way. Two definitions of the same symbol in the
/// same namespace is an ODR violation that only failed to compile because no translation unit
/// had ever included both headers; a test that needed an electron dE/dx and a hadron dE/dx in
/// one file found it immediately. Worse than the violation is what it invited: two prefactors
/// that could drift apart, with electron and hadron ionisation silently using different ones.
template <typename real_t> __host__ __device__ inline real_t twopi_mc2_rcl2() {
  return units::twopi_mc2_rcl2<real_t>();
}

/// Largest energy transferable to a delta ray, MeV. Transcribed from
/// G4MollerBhabhaModel::MaxSecondaryEnergy: half the kinetic energy for Moller (identical
/// particles), all of it for Bhabha.
template <typename real_t>
__host__ __device__ inline real_t max_secondary_energy(real_t kinetic, bool is_positron) {
  return is_positron ? kinetic : real_t(0.5) * kinetic;
}

/// Restricted collision stopping power, MeV/mm. @p is_positron selects the Bhabha branch.
///
/// "Restricted" means only energy transfers below @p cut are continuous; above it Geant4
/// emits an explicit delta ray instead, so that energy leaves the local deposit. Pass a
/// negative @p cut to use the material's own production threshold, or a huge value for the
/// unrestricted (total) stopping power.
template <typename real_t>
__host__ __device__ inline real_t collision_dedx(const data::Material<real_t>& m, real_t kinetic,
                                                 bool is_positron, real_t cut = real_t(-1)) {
  const real_t me = units::electron_mass_c2<real_t>();
  // Geant4 clamps below 0.25*sqrt(Zeff) keV and extrapolates underneath.
  const real_t th = real_t(0.25) * sqrt(m.z_eff) * real_t(1e-3);
  const real_t tkin = fmax(kinetic, th);

  const real_t tau = tkin / me;
  const real_t gam = tau + real_t(1);
  const real_t gamma2 = gam * gam;
  const real_t bg2 = tau * (tau + real_t(2));
  const real_t beta2 = bg2 / gamma2;

  const real_t eexc = m.mean_excitation / me;  // mean excitation energy, MeV -> units of m_e
  const real_t eexc2 = eexc * eexc;

  // G4MollerBhabhaModel: d = min(cut, MaxSecondaryEnergy) / m_e.
  const real_t tmax = max_secondary_energy(tkin, is_positron);
  const real_t use_cut = (cut < real_t(0)) ? m.cut_electron : cut;
  const real_t d = fmin(use_cut, tmax) / me;

  real_t dedx;
  if (!is_positron) {
    dedx = log(real_t(2) * (tau + real_t(2)) / eexc2) - real_t(1) - beta2 + log((tau - d) * d)
           + tau / (tau - d)
           + (real_t(0.5) * d * d + (real_t(2) * tau + real_t(1)) * log1p(-d / tau)) / gamma2;
  } else {
    const real_t d2 = d * d * real_t(0.5);
    const real_t d3 = d2 * d / real_t(1.5);
    const real_t d4 = d3 * d * real_t(0.75);
    const real_t y = real_t(1) / (real_t(1) + gam);
    dedx = log(real_t(2) * (tau + real_t(2)) / eexc2) + log(tau * d)
           - beta2
                 * (tau + real_t(2) * d
                    - y * (real_t(3) * d2 + y * (d - d3 + y * (d2 - tau * d3 + d4))))
                 / tau;
  }

  // Sternheimer density-effect correction, x = log10(beta*gamma) = log(bg2)/(2 ln 10).
  dedx -= data::density_correction(m, log(bg2) / data::twoln10<real_t>());

  dedx *= twopi_mc2_rcl2<real_t>() * m.electron_density / beta2;
  if (dedx < real_t(0)) { dedx = real_t(0); }

  // Geant4 low-energy extrapolation below the threshold, verbatim from
  // G4MollerBhabhaModel::ComputeDEDXPerVolume's last four lines:
  //
  //     if (kineticEnergy < th) {
  //       x = kineticEnergy/th;
  //       if(x > 0.25) { dedx /= sqrt(x); }
  //       else         { dedx *= 1.4*sqrt(x)/(0.1 + x); }
  //     }
  //
  // THE SECOND BRANCH WAS A CONSTANT 2 AND IS NOT ONE. It read
  // `real_t(1)/sqrt(real_t(0.25))`, which is the first branch frozen at the breakpoint - the
  // two forms do agree there (1.4*0.5/0.35 = 2.0 exactly, which is why the substitution looks
  // harmless) and they diverge below it: the correct form falls off as sqrt(x)/0.1 towards
  // zero while a constant 2 keeps the full stopping power. At 100 eV in water x is 0.149 and
  // the two differ by 8.5%; at 100 eV in `CustomSiGe` by 9.6%.
  //
  // IT COULD NOT BE SEEN UNTIL THE TABLE REACHED 100 eV. `x <= 0.25` means
  // `E <= 0.0625*sqrt(Zeff) keV`, which is 168 eV in water and 306 eV in lead - below the
  // 1 keV floor the old e+- table started at, below the lowest row of
  // `ref/oracle/electron_tables.csv`, and below `G4EmParameters::LowestElectronEnergy`, so no
  // electron is ever TRACKED there. What is there is the first two or three nodes of the range
  // table, and the range at 1 keV is the integral from 100 eV upwards: the error showed up
  // four decades higher as a 1.95% range residual with no visible cause, and
  // `ref/dump/dump_electron_hi.cc`'s node grid is what made it visible.
  if (kinetic < th) {
    const real_t x = kinetic / th;
    if (x > real_t(0.25)) {
      dedx /= sqrt(x);
    } else {
      dedx *= real_t(1.4) * sqrt(x) / (real_t(0.1) + x);
    }
  }
  return dedx;
}

/// Radiative stopping power, MeV/mm, from an approximate radiation-yield scaling.
///
/// NOT REACHED BY THE TRANSPORT AND NOT REACHED BY THE TABLES. It survives because
/// `step_lepton`'s collision/radiative split falls back to it when a Scene carries no
/// bremsstrahlung table at all (a study configuration), and because two old tests call it.
/// The dE/dx table below uses `brems_restricted_dedx`, which is the real Seltzer-Berger and
/// relativistic integrals. See docs/RISK.md entry E2.
template <typename real_t>
__host__ __device__ inline real_t radiative_dedx(const data::Material<real_t>& m, real_t kinetic) {
  // (dE/dx)_rad / (dE/dx)_col ~ E * Z_eff / 800 MeV  (Evans, order-of-magnitude form)
  const real_t ratio = kinetic * m.z_eff / real_t(800);
  return collision_dedx(m, kinetic, false) * ratio;
}

/// Restricted total stopping power: restricted collision loss plus radiative loss. This is
/// what Geant4 integrates to build its range table, which is why an unrestricted integral
/// came out 4-8% short.
template <typename real_t>
__host__ __device__ inline real_t total_dedx(const data::Material<real_t>& m, real_t kinetic,
                                             bool is_positron) {
  return collision_dedx(m, kinetic, is_positron) + radiative_dedx(m, kinetic);
}

// ---------------------------------------------------------------- delta rays

/// Macroscopic cross section for producing a delta ray above @p cut, 1/mm.
/// Transcribed from G4MollerBhabhaModel::ComputeCrossSectionPerElectron, scaled by the
/// material electron density as G4VEmModel::CrossSectionPerVolume does.
template <typename real_t>
__host__ __device__ inline real_t delta_ray_xs(const data::Material<real_t>& m, real_t kinetic,
                                               bool is_positron, real_t cut = real_t(-1)) {
  const real_t me = units::electron_mass_c2<real_t>();
  const real_t cut_energy = (cut < real_t(0)) ? m.cut_electron : cut;
  const real_t tmax = max_secondary_energy(kinetic, is_positron);
  if (cut_energy >= tmax || kinetic <= real_t(0)) { return real_t(0); }

  const real_t xmin = cut_energy / kinetic;
  const real_t xmax = tmax / kinetic;
  const real_t tau = kinetic / me;
  const real_t gam = tau + real_t(1);
  const real_t gamma2 = gam * gam;
  const real_t beta2 = tau * (tau + real_t(2)) / gamma2;

  real_t cross;
  if (!is_positron) {  // Moller
    const real_t gg = (real_t(2) * gam - real_t(1)) / gamma2;
    cross = ((xmax - xmin) * (real_t(1) - gg + real_t(1) / (xmin * xmax)
                              + real_t(1) / ((real_t(1) - xmin) * (real_t(1) - xmax)))
             - gg * log(xmax * (real_t(1) - xmin) / (xmin * (real_t(1) - xmax))))
            / beta2;
  } else {  // Bhabha
    const real_t y = real_t(1) / (real_t(1) + gam);
    const real_t y2 = y * y;
    const real_t y12 = real_t(1) - real_t(2) * y;
    const real_t b1 = real_t(2) - y2;
    const real_t b2 = y12 * (real_t(3) + y2);
    const real_t y122 = y12 * y12;
    const real_t b4 = y122 * y12;
    const real_t b3 = b4 + y122;
    cross = (xmax - xmin) * (real_t(1) / (beta2 * xmin * xmax) + b2
                             - real_t(0.5) * b3 * (xmin + xmax)
                             + b4 * (xmin * xmin + xmin * xmax + xmax * xmax) / real_t(3))
            - b1 * log(xmax / xmin);
  }
  // G4: cross *= twopi_mc2_rcl2/kineticEnergy. Our constant is the same CLHEP quantity.
  cross *= twopi_mc2_rcl2<real_t>() / kinetic;
  return fmax(cross, real_t(0)) * m.electron_density;
}

template <typename real_t>
struct DeltaRay {
  Vec3<real_t> delta_dir;
  real_t delta_ekin;
  Vec3<real_t> primary_dir;  ///< recoil direction of the primary after the collision
  real_t primary_ekin;
  bool produced;
};

/// Samples one delta ray. Transcribed from G4MollerBhabhaModel::SampleSecondaries,
/// including the rejection majorants and the exact recoil kinematics (the primary
/// direction comes from momentum conservation, not from an angle formula).
template <typename real_t, typename Rng>
__host__ __device__ inline DeltaRay<real_t> sample_delta_ray(
    const data::Material<real_t>& m, real_t kinetic, const Vec3<real_t>& dir, bool is_positron,
    Rng& rng, real_t cut = real_t(-1)) {
  DeltaRay<real_t> out{dir, real_t(0), dir, kinetic, false};
  const real_t me = units::electron_mass_c2<real_t>();
  const real_t tmin = (cut < real_t(0)) ? m.cut_electron : cut;
  const real_t tmax = max_secondary_energy(kinetic, is_positron);
  if (tmin >= tmax) { return out; }

  const real_t energy = kinetic + me;          // total energy
  const real_t xmin = tmin / kinetic;
  const real_t xmax = tmax / kinetic;
  const real_t gam = energy / me;
  const real_t gamma2 = gam * gam;
  const real_t beta2 = real_t(1) - real_t(1) / gamma2;

  real_t x, z, grej;
  if (!is_positron) {  // Moller
    const real_t gg = (real_t(2) * gam - real_t(1)) / gamma2;
    real_t y = real_t(1) - xmax;
    grej = real_t(1) - gg * xmax
           + xmax * xmax * (real_t(1) - gg + (real_t(1) - gg * y) / (y * y));
    do {
      const real_t r0 = rng.uniform(), r1 = rng.uniform();
      x = xmin * xmax / (xmin * (real_t(1) - r0) + xmax * r0);
      y = real_t(1) - x;
      z = real_t(1) - gg * x + x * x * (real_t(1) - gg + (real_t(1) - gg * y) / (y * y));
      if (grej * r1 <= z) { break; }
    } while (true);
  } else {  // Bhabha
    const real_t yy = real_t(1) / (real_t(1) + gam);
    const real_t y2 = yy * yy;
    const real_t y12 = real_t(1) - real_t(2) * yy;
    const real_t b1 = real_t(2) - y2;
    const real_t b2 = y12 * (real_t(3) + y2);
    const real_t y122 = y12 * y12;
    const real_t b4 = y122 * y12;
    const real_t b3 = b4 + y122;
    real_t y = xmax * xmax;
    grej = real_t(1) + (y * y * b4 - xmin * xmin * xmin * b3 + y * b2 - xmin * b1) * beta2;
    do {
      const real_t r0 = rng.uniform(), r1 = rng.uniform();
      x = xmin * xmax / (xmin * (real_t(1) - r0) + xmax * r0);
      y = x * x;
      z = real_t(1) + (y * y * b4 - x * y * b3 + y * b2 - x * b1) * beta2;
      if (grej * r1 <= z) { break; }
    } while (true);
  }

  const real_t delta_kin = x * kinetic;
  const real_t delta_mom = sqrt(delta_kin * (delta_kin + real_t(2) * me));
  const real_t primary_mom = sqrt(kinetic * (kinetic + real_t(2) * me));
  real_t cost = delta_kin * (energy + me) / (delta_mom * primary_mom);
  if (cost > real_t(1)) { cost = real_t(1); }
  const real_t sint = sqrt((real_t(1) - cost) * (real_t(1) + cost));
  const real_t phi = units::twopi<real_t>() * rng.uniform();
  const Vec3<real_t> local{sint * cos(phi), sint * sin(phi), cost};
  out.delta_dir = normalize(rotate_uz(local, dir));
  out.delta_ekin = delta_kin;

  // Primary recoil from momentum conservation, exactly as Geant4 does it.
  const Vec3<real_t> p_before = primary_mom * dir;
  const Vec3<real_t> p_delta = delta_mom * out.delta_dir;
  out.primary_dir = normalize(p_before - p_delta);
  out.primary_ekin = kinetic - delta_kin;
  out.produced = true;
  return out;
}

// ------------------------------------------------------- the e+- energy-loss tables
//
// EVERY ELECTRON ABOVE 100 MeV WAS A 100 MeV ELECTRON, AND THIS IS THE TABLE THAT MADE IT SO.
// Read docs/RISK.md V64 before changing anything here.
//
// What was here: a flat log grid of 128 points from 1 keV to **100 MeV**, linearly
// interpolated, ONE species, `lookup` returning the last bin for any energy at or above its
// ceiling and `energy_from_range` returning that ceiling for any range past the table. A 1 GeV
// electron's first step read a 100 MeV range, took its step, inverted the remaining range and
// came out at or below 100 MeV; the other 900 MeV was not deposited, not carried off by a
// secondary and not counted. The twelve-beam B1 sweep of 2026-09-11 found it at 46% of Geant4's
// dose (docs/B1_SWEEP.md), and nothing before that had ever run an electron above 100 MeV.
//
// Three things are different now, and each of them is Geant4's and not a choice made here.
//
//   1. **The grid is G4EmParameters'.** `MinKinEnergy` 100 eV, `MaxKinEnergy` 100 TeV,
//      `NumberOfBinsPerDecade` 7 - so `nBins = 84` and 85 points over twelve decades, cubic
//      spline (`G4LossTableBuilder::splineFlag` is true by default). This is the same grid and
//      the same construction `em/hadron_range.cuh` already uses, for the reason
//      docs/PORTED.md 4.3 gives at length: `G4VEnergyLossProcess` never evaluates a model
//      during transport, so reproducing the transport means reproducing the VECTOR and not
//      the model it was sampled from.
//
//   2. **e- and e+ have separate tables**, because Geant4 builds one per particle and the
//      restricted collision stopping power is Moller for one and Bhabha for the other. The
//      old table was built with `is_positron = false` throughout and every positron in this
//      port read the electron's range - a second defect the ceiling was hiding.
//
//   3. **Nothing clamps at the top.** `data::spline_value` returns the end value outside the
//      table, which is `G4PhysicsVector::Interpolation`'s own behaviour at 100 TeV and
//      therefore right; above 100 TeV there is no Geant4 answer to reproduce, so
//      `step_lepton` refuses such a track by name and books its energy rather than
//      transporting it as a 100 TeV electron. `above_table` is that test.
//
// The restricted dE/dx summed here is `G4LossTableManager::BuildTables`' sum over the energy
// loss processes on the particle: `G4eIonisation` (`collision_dedx`, restricted to the
// electron production cut) plus `G4eBremsstrahlung` (`brems_restricted_dedx`, restricted to
// the GAMMA production cut). `ref/oracle/electron_tables.csv`'s `dedx_total_MeV_per_mm` is
// `G4EmCalculator::GetDEDX`, which is exactly that sum read off Geant4's own table.

constexpr int kRangeBins = 85;
/// G4EmParameters::MinKinEnergy / MaxKinEnergy, MeV.
constexpr double kRangeEMin = 1e-4;
constexpr double kRangeEMax = 1e8;
constexpr int kRangeBinsPerDecade = 7;
/// The four above are not independent, and `build_range_table` builds the grid from THREE of
/// them - `kRangeEMax` only ever reaches `t.e_max`, which is what `above_table` and the sqrt
/// taper read. So lowering `kRangeEMax` alone moves the refusal threshold and leaves the table
/// where it was, and raising `kRangeBins` alone extends the table past the energy the refusal
/// names. Both were tried while inverting the ceiling check in `tests/test_electron_hi.cu` and
/// both produced a table that passed some of it; this is the assertion that stops either.
static_assert(kRangeBins == kRangeBinsPerDecade * 12 + 1,
              "kRangeBins must be kRangeBinsPerDecade per decade over the twelve decades from "
              "kRangeEMin = 100 eV to kRangeEMax = 100 TeV, plus one for the closing node");

/// Restricted radiative stopping power, MeV/mm - the `G4eBremsstrahlung` half of the e+-
/// dE/dx table.
///
/// EVALUATED RATHER THAN READ OUT OF `data::BremsTable`, and the reason is arithmetic: that
/// table is 40 bins per decade and this one is 7, and `10^(b/7)` coincides with `10^(c/40)`
/// only when b is a multiple of 7 - so 6 of every 7 nodes of the dE/dx table would carry the
/// brems table's interpolation error instead of the model's value. Geant4 samples the models at
/// its own 85 nodes; so does this.
///
/// `data::brems_dedx_xs` is the composition - the two models, the 1 GeV split and
/// `G4EmModelManager`'s boundary factor - and it is called rather than repeated, because the
/// other caller is `data::build_brems_tables` and two copies of one composition is the
/// arrangement `twopi_mc2_rcl2`'s header above describes going wrong. `tests/test_electron_hi.cu`
/// still checks the two agree at the brems table's own nodes, because that is cheap and because
/// the next person to need a third grid should find the check already there.
template <typename real_t>
__host__ inline real_t brems_restricted_dedx(const data::Material<real_t>& mat,
                                             const data::SBTableSet<real_t>& sb,
                                             const data::BremsBoundary<real_t>& bnd,
                                             real_t kinetic, bool is_positron) {
  real_t dedx = real_t(0), xs = real_t(0);
  data::brems_dedx_xs(mat, sb, bnd, kinetic, is_positron, dedx, xs);
  return dedx;
}

/// The restricted dE/dx, range and inverse-range tables for e- and e+, on Geant4's grid.
///
/// Indexed `[is_positron][material][bin]`. One row per (species, material) because that is one
/// `G4PhysicsVector` per material cuts couple per particle, which is what Geant4 holds.
template <typename real_t>
struct RangeTable {
  real_t e_min, e_max;  ///< MeV, G4EmParameters' MinKinEnergy and MaxKinEnergy
  int n_materials = data::kNumMaterials;

  /// The shared log grid. Held rather than recomputed because the spline needs the abscissae
  /// as an array, and because the inverse lookup uses it as the *ordinate*.
  real_t energy[kRangeBins];

  real_t dedx[2][data::kMaxMaterials][kRangeBins];     ///< MeV/mm, restricted
  real_t dedx_d2[2][data::kMaxMaterials][kRangeBins];
  real_t range[2][data::kMaxMaterials][kRangeBins];    ///< mm
  real_t range_d2[2][data::kMaxMaterials][kRangeBins];
  /// Second derivatives of the *inverse* range table. `G4LossTableBuilder::
  /// BuildInverseRangeTable` stores the same points with range as the abscissa and energy as
  /// the ordinate and splines that; the abscissae are the range row itself, so only the
  /// derivatives need their own array.
  real_t inv_d2[2][data::kMaxMaterials][kRangeBins];

  /// True for a kinetic energy Geant4 has no table for. `step_lepton` refuses such a track by
  /// name; see the block at the top of this section.
  __host__ __device__ bool above_table(real_t kinetic) const { return kinetic > e_max; }

  /// Restricted dE/dx, MeV/mm.
  ///
  /// `G4VEnergyLossProcess::GetDEDXForScaledEnergy`: the spline, then a sqrt taper below
  /// MinKinEnergy. The taper is Geant4's and is why nothing clamps to the first bin.
  __host__ __device__ real_t dedx_at(int material, bool pos, real_t kinetic) const {
    const int p = pos ? 1 : 0;
    real_t x = data::spline_value<real_t>(energy, dedx[p][material], dedx_d2[p][material],
                                          kRangeBins, kinetic);
    if (kinetic < e_min) { x *= sqrt(kinetic / e_min); }
    return fmax(x, real_t(0));
  }

  /// Range, mm. `G4VEnergyLossProcess::GetScaledRangeForScaledEnergy`, same taper.
  __host__ __device__ real_t lookup(int material, bool pos, real_t kinetic) const {
    const int p = pos ? 1 : 0;
    real_t r = data::spline_value<real_t>(energy, range[p][material], range_d2[p][material],
                                          kRangeBins, kinetic);
    if (kinetic < e_min) { r *= sqrt(kinetic / e_min); }
    return fmax(r, real_t(0));
  }

  /// The kinetic energy whose range is @p r, MeV.
  ///
  /// `G4VEnergyLossProcess::ScaledKinEnergyForLoss`: the inverse table's spline above its
  /// first point, and `minKinEnergy * (r/rmin)^2` below it - the exact inverse of the sqrt
  /// taper the two lookups above apply, which is why they have to be the same taper.
  __host__ __device__ real_t energy_from_range(int material, bool pos, real_t r) const {
    const int p = pos ? 1 : 0;
    const real_t* row = range[p][material];
    const real_t rmin = row[0];
    if (r < rmin) {
      if (r <= real_t(0)) { return real_t(0); }
      const real_t x = r / rmin;
      return e_min * x * x;
    }
    return data::spline_value<real_t>(row, energy, inv_d2[p][material], kRangeBins, r);
  }
};

/// Builds the e+- dE/dx, range and inverse-range tables by G4LossTableBuilder's algorithm.
///
/// Not "integrate 1/(dE/dx)". Geant4 integrates its *own interpolated table*, and the
/// difference is not academic - 7 points per decade across a Seltzer-Berger radiative term
/// that turns on at the gamma cut is a function the models do not describe between nodes.
///
///   G4LossTableBuilder::BuildRangeTable, verbatim:
///     range(0) = 2 * E(0) / dedx(0)
///     range(j) = range(j-1) + sum over n=100 midpoint sub-steps of de / dedx_spline(e)
///
/// The seed's factor of two is the boundary condition for a stopping power that goes as
/// sqrt(E) below the table's first node: the integral of 1/sqrt from zero to E is 2E/dedx(E).
/// Getting it wrong made the proton range 50.008% short at 1 keV in every material
/// (em/hadron_range.cuh says so at length) and the same arithmetic applies here.
///
/// @param sb the Seltzer-Berger differential tables. REQUIRED: Geant4's e+- dE/dx table is
///           the sum over `G4eIonisation` AND `G4eBremsstrahlung`, so a table built without
///           the radiative term is a different quantity from the one the transport reads, and
///           building one silently would be exactly the defect V64 records. Refused loudly.
template <typename real_t>
__host__ inline void build_range_table(const data::Material<real_t>* mats, RangeTable<real_t>& t,
                                       const data::SBTableSet<real_t>* sb = nullptr,
                                       int n_materials = data::kNumMaterials) {
  if (sb == nullptr) {
    std::printf("\nFATAL: build_range_table was called with no Seltzer-Berger tables.\n"
                "  Geant4's e+- dE/dx table is the sum over G4eIonisation and\n"
                "  G4eBremsstrahlung (G4LossTableManager::BuildTables); without the second\n"
                "  term this is not the table the transport reads. See docs/RISK.md V64.\n");
    std::exit(2);
  }
  t.e_min = real_t(kRangeEMin);
  t.e_max = real_t(kRangeEMax);
  t.n_materials = n_materials;

  for (int b = 0; b < kRangeBins; ++b) {
    t.energy[b] =
        static_cast<real_t>(kRangeEMin * std::pow(10.0, double(b) / kRangeBinsPerDecade));
  }
  // The grid's top node IS `e_max`, and `above_table` promises that nothing past it is
  // transported. See the static_assert on the constants: this is its run-time half, for the
  // case where `std::pow` and the exponent arithmetic do not land where the integers say.
  if (!(std::fabs(static_cast<double>(t.energy[kRangeBins - 1]) / kRangeEMax - 1.0) < 1e-9)) {
    std::printf("\nFATAL: the e+- table's top node is %g MeV and e_max is %g MeV.\n"
                "  `above_table` refuses a track past e_max, so the two must be the same\n"
                "  energy or there is a band the table clamps in and nothing refuses.\n"
                "  See build_range_table.\n",
                static_cast<double>(t.energy[kRangeBins - 1]), kRangeEMax);
    std::exit(2);
  }

  std::vector<double> x(kRangeBins), y(kRangeBins), d2(kRangeBins), r(kRangeBins);
  std::vector<double> rd2(kRangeBins), id2(kRangeBins);
  for (int b = 0; b < kRangeBins; ++b) { x[b] = static_cast<double>(t.energy[b]); }

  for (int p = 0; p < 2; ++p) {
    const bool pos = (p == 1);
    for (int m = 0; m < n_materials; ++m) {
      // `G4EmModelManager`'s continuity factor across G4eBremsstrahlung's 1 GeV model
      // boundary, fixed once per (material, species) exactly as it is there.
      const data::BremsBoundary<real_t> bnd = data::brems_boundary(mats[m], *sb, pos);
      // ---- restricted dE/dx on Geant4's grid, and its spline.
      for (int b = 0; b < kRangeBins; ++b) {
        const real_t e = t.energy[b];
        y[b] = static_cast<double>(collision_dedx(mats[m], e, pos)
                                   + brems_restricted_dedx(mats[m], *sb, bnd, e, pos));
      }
      // Geant4 skips leading zero bins and rebuilds the vector on a shorter grid, which would
      // be a different table from this one - so a zero is refused rather than worked around.
      // The same refusal, for the same reason, as build_hadron_range_table's.
      if (!(y[0] > 0.0)) {
        std::printf("\nFATAL: e+- dE/dx is zero at %g MeV for %s in material %d.\n"
                    "  G4LossTableBuilder::BuildRangeTable drops leading zero bins and builds\n"
                    "  the range on a shorter grid; this table has a fixed grid and would\n"
                    "  silently be a different table. See build_range_table.\n",
                    kRangeEMin, pos ? "e+" : "e-", m);
        std::exit(2);
      }
      // SPLINED, for both species. `G4LossTableBuilder::BuildDEDXTable` copies
      // `t_list[0]`'s vector and inherits its `useSpline`; `t_list[0]` is `G4eIonisation` for
      // both e- and e+ because `G4EmStandardPhysics::ConstructProcess` registers eIoni before
      // eBrem for each of them, and `G4eBremsstrahlung`'s constructor does NOT call
      // `SetSpline(false)` the way `G4MuBremsstrahlung`'s does. So the charge-odd
      // interpolation split that costs the negative hadrons 4-9% of their dose
      // (docs/RISK.md V44/V46, `em::hadron_table_uses_spline`) has no counterpart here: e+
      // and e- are separate process objects, not two members of one shared-process pair.
      data::fill_second_derivatives(x.data(), y.data(), kRangeBins, d2.data());
      for (int b = 0; b < kRangeBins; ++b) {
        t.dedx[p][m][b] = static_cast<real_t>(y[b]);
        t.dedx_d2[p][m][b] = static_cast<real_t>(d2[b]);
      }

      // ---- range, by integrating that table as it will be read.
      constexpr int kSub = 100;  // G4LossTableBuilder's n
      const double del = 1.0 / kSub;
      double e1 = x[0];
      double range = 2.0 * e1 / y[0];
      r[0] = range;
      for (int j = 1; j < kRangeBins; ++j) {
        const double e2 = x[j];
        const double de = (e2 - e1) * del;
        double e = e2 + de * 0.5;
        double sum = 0.0;
        for (int k = 0; k < kSub; ++k) {
          e -= de;
          const double d =
              data::spline_value<double>(x.data(), y.data(), d2.data(), kRangeBins, e);
          if (d > 0.0) { sum += de / d; }
        }
        range += sum;
        r[j] = range;
        e1 = e2;
      }

      data::fill_second_derivatives(x.data(), r.data(), kRangeBins, rd2.data());
      // The inverse table: the same points, range as abscissa, energy as ordinate, and a
      // fresh spline. `G4LossTableBuilder::BuildInverseRangeTable` makes a new
      // `G4PhysicsFreeVector(npoints, splineFlag)` rather than copying the range vector.
      data::fill_second_derivatives(r.data(), x.data(), kRangeBins, id2.data());

      for (int b = 0; b < kRangeBins; ++b) {
        t.range[p][m][b] = static_cast<real_t>(r[b]);
        t.range_d2[p][m][b] = static_cast<real_t>(rd2[b]);
        t.inv_d2[p][m][b] = static_cast<real_t>(id2[b]);
      }
    }
  }
}

// ---------------------------------------------------------------- MSC (Highland)

/// RMS multiple-scattering deflection over a step, radians. Highland formula.
/// Radiation length is approximated from Z_eff and A_eff; see docs/RISK.md entry E3.
template <typename real_t>
__host__ __device__ inline real_t msc_theta0(const data::Material<real_t>& m, real_t kinetic,
                                             real_t step_mm) {
  if (step_mm <= real_t(0)) { return real_t(0); }
  const real_t me = units::electron_mass_c2<real_t>();
  const real_t p = sqrt(kinetic * (kinetic + real_t(2) * me));  // MeV/c
  const real_t e_total = kinetic + me;
  const real_t beta = p / e_total;
  if (p <= real_t(0) || beta <= real_t(0)) { return real_t(0); }
  const real_t x_over_x0 = step_mm / m.radiation_length;
  if (x_over_x0 <= real_t(0)) { return real_t(0); }
  return real_t(13.6) / (beta * p) * sqrt(x_over_x0)
         * (real_t(1) + real_t(0.038) * log(x_over_x0));
}

/// Applies an MSC deflection to @p dir, sampling the polar angle from a Gaussian of
/// width theta0 and the azimuth uniformly.
template <typename real_t, typename Rng>
__host__ __device__ inline Vec3<real_t> msc_scatter(const Vec3<real_t>& dir, real_t theta0,
                                                    Rng& rng) {
  if (theta0 <= real_t(0)) { return dir; }
  // Box-Muller for one Gaussian deviate.
  const real_t u1 = rng.uniform(), u2 = rng.uniform();
  const real_t theta = theta0 * sqrt(real_t(-2) * log(u1)) * cos(units::twopi<real_t>() * u2);
  const real_t phi = units::twopi<real_t>() * rng.uniform();
  const real_t st = sin(theta), ct = cos(theta);
  const Vec3<real_t> local{st * cos(phi), st * sin(phi), ct};
  return normalize(rotate_uz(local, dir));
}

}  // namespace g4gpu::em
