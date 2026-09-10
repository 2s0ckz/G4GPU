// Total stopping power and range for a heavy charged particle.
//
// Everything here is assembly rather than new physics: the stopping powers themselves are in
// bragg.cuh, hadron_ionisation.cuh, icru73qo.cuh and nuclear_stopping.cuh, each already checked
// against Geant4's own tables to 1e-6 or better. What was missing was the two things a
// *transport* needs and a table comparison does not:
//
//   * one function that picks the right model for an energy, as G4hIonisation and
//     G4ionIonisation do when they hand a step to a model, and
//   * the range integral over it, which is what limits a step and what converts a step length
//     back into an energy.
//
// Both are checked against Geant4 directly: `dedx_total` and `range_mm` in
// ref/oracle/hadron_tables.csv are G4EmCalculator::GetDEDX and ::GetRange for the same
// particle, material and energy. See tests/test_hadron_range.cu.
#pragma once
#include <cmath>

#include "core/particle.cuh"
#include "core/units.cuh"
#include "data/materials.cuh"
#include "physics/em/bragg.cuh"
#include "physics/em/em_corrections.cuh"
#include "physics/em/hadron_ionisation.cuh"
#include "physics/em/icru73qo.cuh"
#include "physics/em/nuclear_stopping.cuh"
#include "data/g4spline.hh"

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace g4gpu::em {

// ------------------------------------------------------------------ transport parameters
//
// Every one of these is a G4EmParameters default that differs from the electron's, and each
// was read out of the source rather than assumed to be shared. A stepper written by copying
// step_lepton and swapping the physics would inherit four wrong numbers.

/// G4EmExtraParameters::finalRangeMuHad, and finalRangeLIons which happens to equal it.
/// The electron's is 1 mm; a hadron's is a tenth of that, which is what keeps the last
/// millimetre of a Bragg peak - the part that matters - from being one step.
template <typename real_t> __host__ __device__ constexpr real_t kHadronFinalRange() {
  return real_t(0.1);  // mm
}

/// dRoverRangeMuHad. The same 0.2 as the electron's, stated separately because it is a
/// separate parameter that merely happens to agree.
template <typename real_t> __host__ __device__ constexpr real_t kHadronDRoverRange() {
  return real_t(0.2);
}

/// G4EmParameters::LowestMuHadEnergy. Below it G4VEnergyLossProcess stops tracking and
/// deposits the remainder locally.
template <typename real_t> __host__ __device__ constexpr real_t kHadronTrackingCut() {
  return real_t(1e-3);  // 1 keV
}

/// G4EmParameters::MscMuHadRangeFactor, the `facrange` in G4WentzelVIModel's step limit.
/// **0.2, not the electron's 0.04** - a factor of five, straight into the step length.
template <typename real_t> __host__ __device__ constexpr real_t kHadronFacRange() {
  return real_t(0.2);
}

/// G4EmParameters::MuHadLateralDisplacement, which is *false* where the electron's
/// LateralDisplacement is true. A proton's MSC deflects it but does not shift it sideways.
template <typename real_t> __host__ __device__ constexpr bool kHadronLateralDisplacement() {
  return false;
}

/// Which of the three ionisation models Geant4 would use here.
///
/// G4hIonisation gives a proton G4BraggModel below 2 MeV and G4BetheBlochModel above.
/// Which ionisation model Geant4 uses, for any charged hadron, muon or ion at any energy.
///
/// Three processes, five models, and the choice is by *process* rather than by any property of
/// the particle - see uses_ion_ionisation and is_muon in core/particle.cuh for why that
/// distinction is not cosmetic.
///
///                       below the boundary            boundary          above
///   G4hIonisation       q>0 Bragg / q<0 ICRU73QO      2 MeV * m/m_p     BetheBloch
///   G4ionIonisation     BraggIon                      2 MeV * m/m_p     BetheBloch
///   G4MuIonisation      q>0 Bragg / q<0 ICRU73QO      200 keV, flat     MuBetheBloch
///
/// The muon's boundary is the one place a rule written on mass gets it wrong: 2 MeV scaled by
/// the muon mass is 225 keV, and G4MuIonisation uses a literal `elow = 0.2*CLHEP::MeV`. The
/// two differ by 25 keV, a band a stopping muon spends real time in.
///
/// ICRU73QO covers every *negative* hadron below the boundary - pi-, K-, anti-proton, mu- -
/// and not just antiprotons. It exists because the Barkas term in the stopping power is odd in
/// the charge, so a negative hadron does not stop like a positive one of the same mass.
enum class HadronIoniModel : int { kBragg, kBraggIon, kICRU73QO, kBetheBloch, kMuBetheBloch };

/// The energy at which the low-energy model hands over.
template <typename real_t>
__host__ __device__ inline real_t ioni_model_boundary(ParticleType type,
                                                      const ParticleDef<real_t>& pd) {
  return is_muon(type) ? kMuBetheBlochLow<real_t>() : bragg_bethe_boundary(pd);
}

template <typename real_t>
__host__ __device__ inline HadronIoniModel hadron_ioni_model(ParticleType type,
                                                             const ParticleDef<real_t>& pd,
                                                             real_t kinetic) {
  if (kinetic >= ioni_model_boundary(type, pd)) {
    return is_muon(type) ? HadronIoniModel::kMuBetheBloch : HadronIoniModel::kBetheBloch;
  }
  if (uses_ion_ionisation(type)) { return HadronIoniModel::kBraggIon; }
  return (pd.charge > real_t(0)) ? HadronIoniModel::kBragg : HadronIoniModel::kICRU73QO;
}

/// True for the two Bragg-family models, which share a 0.25 keV floor on the delta-ray window
/// that Bethe-Bloch does not have. ICRU73QO has a floor too, but a different one (5 keV), so
/// it is deliberately not lumped in here.
__host__ __device__ inline bool is_bragg_family(HadronIoniModel m) {
  return m == HadronIoniModel::kBragg || m == HadronIoniModel::kBraggIon;
}

/// Restricted ionisation dE/dx, MeV/mm, from whichever model applies.
///
/// @param cut the electron production threshold for this material, MeV. Energy transferred
///            above it leaves as a delta ray and is not part of this number.
template <typename real_t>
__host__ __device__ inline real_t hadron_ioni_dedx(const data::Material<real_t>& m,
                                                   ParticleType type, real_t kinetic,
                                                   real_t cut,
                                                   const ShellTables<real_t>* shell) {
  const ParticleDef<real_t> pd = particle_def<real_t>(type);
  if (pd.mass <= real_t(0) || kinetic <= real_t(0)) { return real_t(0); }
  switch (hadron_ioni_model(type, pd, kinetic)) {
    case HadronIoniModel::kMuBetheBloch:
      // `shell`, WHICH THIS DID NOT PASS. mu_bethe_bloch_dedx defaults it to null, and a null
      // shell pointer switches off two things inside it: the shell correction, which
      // G4MuBetheBlochModel subtracts as `dedx -= 2.0*corr->ShellCorrection(...)`, and
      // G4EmCorrections::HighOrderCorrections, which the same function adds as its last line.
      // So the muon's range table, and every dE/dx the muon was transported with, was the
      // Bethe-Bloch bracket with the radiative correction and nothing else.
      //
      // The half of that which announced itself is the HIGH-ORDER term, because it is
      // CHARGE-ODD: high_order_bracket is 2*(Barkas + Bloch) + Mott, and Barkas goes as z^3.
      // Without it mu+ and mu- have identical stopping power, and the port transported them
      // to a bit-identical B1 dose - 23.8377 nGy for both - where Geant4 gives 23.11 for mu-
      // and 24.1721 for mu+, a 4.6% asymmetry at 8 sigma. A single number for two particles
      // that the reference separates is the shape of a missing odd term, and it is why this
      // was found by a like-for-like run rather than by tests/test_muon.cu: that file calls
      // mu_bethe_bloch_dedx directly and passes the tables, so it agreed to 0.0000% on 1392
      // points while the transport was reading a different function.
      //
      // A defaulted parameter is what made it possible. Every other branch here passes what
      // it was given; this one silently took a different meaning for the same call. The
      // default is left in place because tests call the model with no tables on purpose, but
      // no transport path may rely on it - see the measurement in docs/RISK.md.
      return mu_bethe_bloch_dedx(m, type, kinetic, cut, shell);
    case HadronIoniModel::kBetheBloch:
      return bethe_bloch_dedx(m, type, kinetic, cut, shell);
    case HadronIoniModel::kBraggIon:
      return bragg_ion_dedx(m, type, kinetic, cut);
    case HadronIoniModel::kICRU73QO:
      return qo_dedx(m, type, kinetic, cut);
    case HadronIoniModel::kBragg:
    default:
      return bragg_dedx(m, type, kinetic, cut);
  }
}

/// Total dE/dx for the *range table*: ionisation only.
///
/// Nuclear stopping is deliberately not here, and that is a fidelity point rather than an
/// omission. In Geant4 `G4NuclearStopping` is a G4VEmProcess that applies its loss along the
/// step; it does not build a DEDX table and does not contribute to the one
/// G4VEnergyLossProcess integrates into a range. So Geant4's own range for a proton is the
/// range against ionisation alone, and a table that added nuclear stopping would be a
/// different quantity from the one the stepper's limits are expressed in.
///
/// The oracle says so directly: in ref/oracle/hadron_tables.csv the `nuclear_dedx` column is
/// zero for every proton row and `dedx_total` equals `dedx_ioni`. Adding nuclear stopping here
/// put the proton range 25% short at 10 keV, which is the kind of error that moves a Bragg
/// peak rather than the kind that rounds away.
///
/// Nuclear stopping still happens - the stepper applies it to the energy after the continuous
/// loss, as G4NuclearStopping::AlongStepDoIt does. See step_hadron.
template <typename real_t>
__host__ __device__ inline real_t hadron_total_dedx(const data::Material<real_t>& m,
                                                    ParticleType type, real_t kinetic,
                                                    real_t cut,
                                                    const ShellTables<real_t>* shell) {
  return hadron_ioni_dedx(m, type, kinetic, cut, shell);
}

// ---------------------------------------------------------------- the range table

/// The species that get a table of their own.
///
/// Exactly Geant4's set. G4VEnergyLossProcess builds one dE/dx and one range table per *base
/// particle*, and G4hIonisation::InitialiseEnergyLossProcess names the base particles by hand:
///
///     proton, anti_proton, pi+, pi-, kaon+, kaon-, GenericIon, alpha   -> no base particle
///     everything else                                                 -> proton / anti_proton
///                                                                        / kaon+ / kaon-
///
/// plus mu+ and mu-, which G4MuIonisation never gives a base to. Ten in all. Everything else -
/// He3, deuteron, triton, hyperons, anti-nuclei - looks one of these up at a scaled energy.
///
/// It was two before (proton and alpha), with He3 and GenericIon mapped onto the alpha's
/// table. That is not an approximation of the scaling, it is a different particle: He3's dE/dx
/// came out wrong by a factor of twelve, which tests/test_hadron_range.cu reported at 91% and
/// nothing acted on. A separate table per base particle is what makes the scaling meaningful.
enum class HadronSpecies : int {
  kProton = 0,
  kAntiProton = 1,
  kPionPlus = 2,
  kPionMinus = 3,
  kKaonPlus = 4,
  kKaonMinus = 5,
  kMuonPlus = 6,
  kMuonMinus = 7,
  kAlpha = 8,
  kGenericIon = 9,
  kCount = 10
};

// ---------------------------------------------------------------- Geant4's grid
//
// Not a grid chosen here. G4VEnergyLossProcess never evaluates a model during transport: at
// initialisation it builds a dE/dx vector on a log grid, splines it, and from then on
// interpolates. Everything downstream - the range, its inverse, the step limit - is that
// vector. So reproducing Geant4's transport means reproducing the vector, not the model it was
// sampled from.
//
//   G4EmParameters::MinKinEnergy      100 eV
//   G4EmParameters::MaxKinEnergy      100 TeV
//   G4EmParameters::NumberOfBinsPerDecade   7
//
// -> 85 points over twelve decades, and a cubic spline between them (G4LossTableBuilder's
// splineFlag is true by default).
//
// This was a 256-point grid from 1 keV to 10 GeV, with linear interpolation, evaluating the
// models exactly. That is a *finer* description of the physics and a *worse* description of
// Geant4: at a model boundary Geant4's coarse spline rings across the discontinuity, and at
// the bottom a 1 keV floor throws away range a light particle actually has. Measured before
// the change - a mu- range 52% wrong here, 2% on Geant4's grid; the proton's long-standing
// 1.3% worst point at 2.24 MeV mostly this rather than the physics. See docs/RISK.md V7.
// One consequence worth knowing before anyone builds with G4GPU_FP32: this grid spans twelve
// decades, so the range row runs from about 1e-6 mm to 1e10 mm. In float that is seven digits
// across a range that needs more, and the spline's differences of neighbouring values are
// where it would show. The default build is double throughout and nothing measures the float
// one today.
constexpr int kHadronRangeBins = 85;
/// G4EmParameters::MinKinEnergy / MaxKinEnergy, MeV.
constexpr double kHadronRangeEMin = 1e-4;
constexpr double kHadronRangeEMax = 1e8;
constexpr int kHadronRangeBinsPerDecade = 7;

/// The ParticleType each tabulated species is, in HadronSpecies order. The one place the two
/// enumerations are tied together; everything else goes through it.
__host__ __device__ inline ParticleType hadron_species_particle(HadronSpecies s) {
  switch (s) {
    case HadronSpecies::kProton: return ParticleType::kProton;
    case HadronSpecies::kAntiProton: return ParticleType::kAntiProton;
    case HadronSpecies::kPionPlus: return ParticleType::kPionPlus;
    case HadronSpecies::kPionMinus: return ParticleType::kPionMinus;
    case HadronSpecies::kKaonPlus: return ParticleType::kKaonPlus;
    case HadronSpecies::kKaonMinus: return ParticleType::kKaonMinus;
    case HadronSpecies::kMuonPlus: return ParticleType::kMuonPlus;
    case HadronSpecies::kMuonMinus: return ParticleType::kMuonMinus;
    case HadronSpecies::kAlpha: return ParticleType::kAlpha;
    default: return ParticleType::kGenericIon;
  }
}

/// The table a species reads: its own where it has one, its base particle's otherwise.
template <typename real_t>
__host__ __device__ inline HadronSpecies hadron_species_of(ParticleType t) {
  switch (hadron_base_particle(t)) {
    case ParticleType::kAntiProton: return HadronSpecies::kAntiProton;
    case ParticleType::kPionPlus: return HadronSpecies::kPionPlus;
    case ParticleType::kPionMinus: return HadronSpecies::kPionMinus;
    case ParticleType::kKaonPlus: return HadronSpecies::kKaonPlus;
    case ParticleType::kKaonMinus: return HadronSpecies::kKaonMinus;
    case ParticleType::kMuonPlus: return HadronSpecies::kMuonPlus;
    case ParticleType::kMuonMinus: return HadronSpecies::kMuonMinus;
    case ParticleType::kAlpha: return HadronSpecies::kAlpha;
    case ParticleType::kGenericIon: return HadronSpecies::kGenericIon;
    default: return HadronSpecies::kProton;
  }
}

/// G4VEnergyLossProcess::massRatio - `baseParticleMass / particleMass`, or 1 for a species
/// with its own table. A lookup is done at `E * massRatio`.
template <typename real_t>
__host__ __device__ inline real_t hadron_mass_ratio(ParticleType t) {
  const ParticleType base = hadron_base_particle(t);
  if (base == t) { return real_t(1); }
  return particle_def<real_t>(base).mass / particle_def<real_t>(t).mass;
}

/// Does G4VEnergyLossProcess replace this species' charge-square ratio with an effective
/// charge on every step?
///
/// `G4VEnergyLossProcess::AlongStepGetPhysicalInteractionLength` does it under `if(isIon)`, and
/// `isIon` comes from G4EmTableUtil::CheckIon: true for a particle whose type is "nucleus"
/// *except* deuteron, triton, alpha+ and alpha, which are excluded by name. So an alpha keeps
/// its bare charge and a He3 - same charge, one nucleon lighter - does not.
///
/// That exclusion list is why this is a predicate and not `charge > 1.5`, and why the alpha's
/// fluctuation model needs no effective charge while He3's dE/dx is wrong by a factor of three
/// at 1 keV without one.
__host__ __device__ inline bool uses_dynamic_effective_charge(ParticleType t) {
  return t == ParticleType::kHe3;
}

/// G4VEnergyLossProcess::chargeSqRatio - what the base particle's dE/dx is multiplied by, and
/// what its range is divided by (along with the mass ratio).
///
/// Two forms, and Geant4 uses both for different species:
///
///   static   `(Q_this / Q_base)^2`, from the PDG charges, set once in PreparePhysicsTable.
///   dynamic  `G4EmCorrections::EffectiveChargeSquareRatio` - which despite the name is an
///            absolute `(q_eff * chargeCorrection)^2` and not a ratio - recomputed at the
///            pre-step energy for every species uses_dynamic_effective_charge names.
///
/// The two coincide at high energy, where a light ion is fully stripped, and diverge as it
/// slows and starts picking up electrons. They also coincide *dimensionally* only because
/// every ion's base particle is GenericIon, whose charge is 1: were the base charge anything
/// else the absolute form would need dividing by it, and Geant4 does not, because it never has
/// to.
template <typename real_t>
__host__ __device__ inline real_t hadron_charge_sq_ratio(const data::Material<real_t>& m,
                                                         ParticleType t, real_t kinetic) {
  const ParticleType base = hadron_base_particle(t);
  if (base == t) { return real_t(1); }
  if (uses_dynamic_effective_charge(t)) {
    const ParticleDef<real_t> pd = particle_def<real_t>(t);
    real_t corr = real_t(1);
    const real_t qc = ion_effective_charge(m, pd, kinetic, corr) * corr;
    return qc * qc;
  }
  const real_t q = particle_def<real_t>(t).charge / particle_def<real_t>(base).charge;
  return q * q;
}

/// Range as a function of energy, per species and material.
///
/// The same shape as em::RangeTable for electrons, and for the same two reasons: a step is
/// limited by a fraction of the remaining range, and the energy after a step of known length
/// is the inverse lookup. Doing either from dE/dx directly would mean integrating inside the
/// stepper, per track, per step.
template <typename real_t>
struct HadronRangeTable {
  real_t e_min, e_max;  ///< MeV, Geant4's MinKinEnergy and MaxKinEnergy
  int n_materials = data::kNumMaterials;

  /// The shared log grid. Held rather than recomputed because the spline needs the abscissae
  /// as an array, and because the inverse lookup uses it as the *ordinate*.
  real_t energy[kHadronRangeBins];

  /// Restricted ionisation dE/dx, MeV/mm, and its spline second derivatives.
  real_t dedx[static_cast<int>(HadronSpecies::kCount)][data::kMaxMaterials][kHadronRangeBins];
  real_t dedx_d2[static_cast<int>(HadronSpecies::kCount)][data::kMaxMaterials][kHadronRangeBins];

  /// Range, mm, and its spline second derivatives.
  real_t range[static_cast<int>(HadronSpecies::kCount)][data::kMaxMaterials][kHadronRangeBins];
  real_t range_d2[static_cast<int>(HadronSpecies::kCount)][data::kMaxMaterials][kHadronRangeBins];

  /// Second derivatives of the *inverse* range table: G4LossTableBuilder::BuildInverseRangeTable
  /// stores the same points with range as the abscissa and energy as the ordinate, and splines
  /// that. The abscissae are the range row itself, so only the derivatives need their own array.
  real_t inv_d2[static_cast<int>(HadronSpecies::kCount)][data::kMaxMaterials][kHadronRangeBins];

  /// Restricted ionisation dE/dx at this energy, MeV/mm.
  ///
  /// G4VEnergyLossProcess::GetDEDXForScaledEnergy: the spline, then a sqrt taper below
  /// MinKinEnergy. The taper is Geant4's and is why nothing clamps to the first bin.
  __host__ __device__ real_t dedx_at(HadronSpecies sp, int material, real_t kinetic) const {
    const real_t* row = dedx[static_cast<int>(sp)][material];
    const real_t* d2 = dedx_d2[static_cast<int>(sp)][material];
    real_t x = data::spline_value<real_t>(energy, row, d2, kHadronRangeBins, kinetic);
    if (kinetic < e_min) { x *= sqrt(kinetic / e_min); }
    return fmax(x, real_t(0));
  }

  /// Range, mm. G4VEnergyLossProcess::GetScaledRangeForScaledEnergy, same taper.
  __host__ __device__ real_t lookup(HadronSpecies sp, int material, real_t kinetic) const {
    const real_t* row = range[static_cast<int>(sp)][material];
    const real_t* d2 = range_d2[static_cast<int>(sp)][material];
    real_t r = data::spline_value<real_t>(energy, row, d2, kHadronRangeBins, kinetic);
    if (kinetic < e_min) { r *= sqrt(kinetic / e_min); }
    return fmax(r, real_t(0));
  }

  /// The energy whose range is @p r, MeV.
  ///
  /// G4VEnergyLossProcess::ScaledKinEnergyForLoss: the inverse table's spline above its first
  /// point, and `minKinEnergy * (r/rmin)^2` below it - the exact inverse of the sqrt taper the
  /// two lookups above apply, which is why they have to be the same taper.
  __host__ __device__ real_t energy_from_range(HadronSpecies sp, int material, real_t r) const {
    const real_t* row = range[static_cast<int>(sp)][material];
    const real_t* d2 = inv_d2[static_cast<int>(sp)][material];
    const real_t rmin = row[0];
    if (r < rmin) {
      if (r <= real_t(0)) { return real_t(0); }
      const real_t x = r / rmin;
      return e_min * x * x;
    }
    return data::spline_value<real_t>(row, energy, d2, kHadronRangeBins, r);
  }

  // ------------------------------------------------------------------ by particle
  //
  // The three above take a tabulated species and an energy in that species' own terms. These
  // take a *particle* and do what G4VEnergyLossProcess does for one with no table of its own:
  // look the base particle's table up at a scaled energy and rescale the answer.
  //
  //     scaledE = E * massRatio                massRatio = m_base / m_this
  //     dE/dx   = chargeSqRatio * table_dedx(scaledE)
  //     range   = table_range(scaledE) / (chargeSqRatio * massRatio)
  //
  // For the ten species that have their own table both ratios are exactly 1 and these are the
  // functions above with one extra multiply. Callers should prefer these anyway: a species
  // gaining or losing a table of its own then changes nothing at the call site.

  /// Restricted ionisation dE/dx for this particle, MeV/mm.
  __host__ __device__ real_t dedx_for(const data::Material<real_t>& m, ParticleType t,
                                      int material, real_t kinetic) const {
    const real_t mr = hadron_mass_ratio<real_t>(t);
    return hadron_charge_sq_ratio<real_t>(m, t, kinetic)
           * dedx_at(hadron_species_of<real_t>(t), material, kinetic * mr);
  }

  /// Range for this particle, mm.
  __host__ __device__ real_t range_for(const data::Material<real_t>& m, ParticleType t,
                                       int material, real_t kinetic) const {
    const real_t mr = hadron_mass_ratio<real_t>(t);
    const real_t reduce = real_t(1) / (hadron_charge_sq_ratio<real_t>(m, t, kinetic) * mr);
    return reduce * lookup(hadron_species_of<real_t>(t), material, kinetic * mr);
  }

  /// The kinetic energy at which this particle has range @p r, MeV.
  ///
  /// The inverse of range_for, and it has to undo both scalings in the right order: the table
  /// is inverted on the *scaled* range, and what comes back is a *scaled* energy.
  ///
  /// The charge-square ratio is a function of energy for an ion, and the energy is what is
  /// being solved for - so this needs one it does not have. Geant4 has the same problem and
  /// answers it the same way: `chargeSqRatio` is whatever
  /// AlongStepGetPhysicalInteractionLength last set from the *pre-step* energy, and stays that
  /// for the whole step. @p at_energy is that pre-step energy; pass the energy the step
  /// started at.
  __host__ __device__ real_t energy_from_range_for(const data::Material<real_t>& m,
                                                   ParticleType t, int material, real_t r,
                                                   real_t at_energy) const {
    const real_t mr = hadron_mass_ratio<real_t>(t);
    const real_t reduce = real_t(1) / (hadron_charge_sq_ratio<real_t>(m, t, at_energy) * mr);
    const real_t scaled = energy_from_range(hadron_species_of<real_t>(t), material, r / reduce);
    return scaled / mr;
  }
};

/// Builds the dE/dx, range and inverse-range tables, by G4LossTableBuilder's algorithm.
///
/// Not "integrate 1/(dE/dx)". Geant4 integrates its *own interpolated table*, and the
/// difference is not academic - the table is 7 points per decade and the models it samples are
/// discontinuous at their boundaries, so the function being integrated is not the model.
///
///   G4LossTableBuilder::BuildRangeTable, verbatim:
///     range(0) = 2 * E(0) / dedx(0)
///     range(j) = range(j-1) + sum over n=100 midpoint sub-steps of de / dedx_spline(e)
///
/// The seed's factor of two is not a fudge: below its lowest tabulated energy the stopping
/// power goes as sqrt(E), and the integral of 1/sqrt from zero to E is 2E/dedx(E) rather than
/// E/dedx(E). Getting it wrong made the proton range 50.008% short at 1 keV in every material -
/// a factor of two, to four digits, constant, which is what a boundary condition looks like and
/// not what a physics error looks like.
///
/// @param cuts per-material electron production threshold, MeV. The range is *restricted*:
///             Geant4's range table integrates the restricted dE/dx, so energy that leaves as
///             a delta ray is not counted as stopping the primary.
template <typename real_t>
__host__ inline void build_hadron_range_table(const data::Material<real_t>* mats,
                                              const real_t* cuts, HadronRangeTable<real_t>& t,
                                              const ShellTables<real_t>* shell = nullptr,
                                              int n_materials = data::kNumMaterials) {
  t.e_min = real_t(kHadronRangeEMin);
  t.e_max = real_t(kHadronRangeEMax);
  t.n_materials = n_materials;

  for (int b = 0; b < kHadronRangeBins; ++b) {
    t.energy[b] = static_cast<real_t>(
        kHadronRangeEMin * std::pow(10.0, double(b) / kHadronRangeBinsPerDecade));
  }

  std::vector<double> x(kHadronRangeBins), y(kHadronRangeBins), d2(kHadronRangeBins);
  for (int b = 0; b < kHadronRangeBins; ++b) { x[b] = static_cast<double>(t.energy[b]); }

  for (int s = 0; s < static_cast<int>(HadronSpecies::kCount); ++s) {
    const ParticleType pt = hadron_species_particle(static_cast<HadronSpecies>(s));
    for (int m = 0; m < n_materials; ++m) {
      const real_t cut = (cuts != nullptr) ? cuts[m] : real_t(0.99e-3);

      // ---- dE/dx on Geant4's grid, and its spline.
      for (int b = 0; b < kHadronRangeBins; ++b) {
        y[b] = static_cast<double>(hadron_total_dedx(mats[m], pt, t.energy[b], cut, shell));
      }
      // Geant4 skips leading zero bins and rebuilds the vector on a shorter grid. That would be
      // a different table from this one, so a zero is refused rather than worked around - if a
      // model ever returns zero at 100 eV this needs the shrinking grid, and silently building
      // something else would hide it.
      if (!(y[0] > 0.0)) {
        std::printf("\nFATAL: dE/dx is zero at %g MeV for species %d in material %d.\n"
                    "  G4LossTableBuilder::BuildRangeTable drops leading zero bins and builds\n"
                    "  the range on a shorter grid; this table has a fixed grid and would\n"
                    "  silently be a different table. See build_hadron_range_table.\n",
                    kHadronRangeEMin, s, m);
        std::exit(2);
      }
      data::fill_second_derivatives(x.data(), y.data(), kHadronRangeBins, d2.data());
      for (int b = 0; b < kHadronRangeBins; ++b) {
        t.dedx[s][m][b] = static_cast<real_t>(y[b]);
        t.dedx_d2[s][m][b] = static_cast<real_t>(d2[b]);
      }

      // ---- range, by integrating that spline.
      constexpr int kSub = 100;  // G4LossTableBuilder's n
      const double del = 1.0 / kSub;
      std::vector<double> r(kHadronRangeBins);
      double e1 = x[0];
      double range = 2.0 * e1 / y[0];
      r[0] = range;
      for (int j = 1; j < kHadronRangeBins; ++j) {
        const double e2 = x[j];
        const double de = (e2 - e1) * del;
        double e = e2 + de * 0.5;
        double sum = 0.0;
        for (int k = 0; k < kSub; ++k) {
          e -= de;
          const double d =
              data::spline_value<double>(x.data(), y.data(), d2.data(), kHadronRangeBins, e);
          if (d > 0.0) { sum += de / d; }
        }
        range += sum;
        r[j] = range;
        e1 = e2;
      }

      std::vector<double> rd2(kHadronRangeBins);
      data::fill_second_derivatives(x.data(), r.data(), kHadronRangeBins, rd2.data());
      // The inverse table: the same points, range as abscissa, energy as ordinate.
      std::vector<double> id2(kHadronRangeBins);
      data::fill_second_derivatives(r.data(), x.data(), kHadronRangeBins, id2.data());

      for (int b = 0; b < kHadronRangeBins; ++b) {
        t.range[s][m][b] = static_cast<real_t>(r[b]);
        t.range_d2[s][m][b] = static_cast<real_t>(rd2[b]);
        t.inv_d2[s][m][b] = static_cast<real_t>(id2[b]);
      }
    }
  }
}

}  // namespace g4gpu::em
