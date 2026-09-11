// Delta rays from a heavy charged particle: the final state, not the cross section.
//
// The cross section is in hadron_ionisation.cuh and is already checked against Geant4 to a
// fraction of a per cent. What was missing is what happens when one fires: how much energy the
// knocked-out electron takes and in which direction. Without it a proton could be stepped but
// could not produce a secondary, so every transfer above the production cut - for a fast proton,
// a large share of the ionisation - would simply vanish.
//
// Transcribed from G4BetheBlochModel::SampleSecondaries and G4BraggModel::SampleSecondaries
// (11.1.1; G4BraggIonModel's is line-for-line identical to G4BraggModel's). All three share the
// 1/T^2 inversion and the (1 - beta^2 T/Tmax) rejection. Bethe-Bloch differs in three ways that
// matter and are easy to lose:
//
//   - it adds a spin term to the rejection function *and* to the majorant;
//   - it applies a projectile form factor afterwards, whose failure produces no secondary at
//     all rather than a resample;
//   - its lower bound is the production cut alone, with no 0.25 keV * massRate floor.
#pragma once
#include <cmath>

#include "core/particle.cuh"
#include "core/units.cuh"
#include "core/vec3.cuh"
#include "data/materials.cuh"
#include "physics/em/bragg.cuh"
#include "physics/em/em_corrections.cuh"
#include "physics/em/electron_processes.cuh"  // em::DeltaRay
#include "physics/em/hadron_ionisation.cuh"
#include "physics/em/icru73qo.cuh"
#include "physics/em/hadron_range.cuh"

namespace g4gpu::em {

/// G4BetheBlochModel::SetupParameters' `formfact`, in 1/MeV.
///
/// The projectile is not a point charge: above a transfer of 2/formfact its form factor cuts
/// the cross section off. For a proton at therapy energies this is nil - the suppression only
/// bites as the transfer approaches a GeV - but it is a few flops and it is what the model does.
///
/// Zero for a lepton, which is Geant4's `GetLeptonNumber() == 0` guard. That branch is dead in
/// this transport (leptons go through Moller-Bhabha, not this file) but the guard is cheap and
/// leaving it out would make the function wrong if it were ever reused.
///
/// @param mass_number `G4NistManager::GetA27`'s argument. Zero means "take it as 2Z", which is
///        the alpha's own A and was the only answer this function needed while the alpha was
///        the only species reaching the heavy branch. A real nucleus has to pass its own A: the
///        term is `A^0.27` and for O16 the two differ by (16/16)^0.27 = 1 only by luck - for
///        Ca40 it is (40/40)^0.27 = 1 as well, and for Li7 it is (7/6)^0.27, 4%. The ratio is
///        A/2Z, which is 1 for every N = Z nuclide and rises to about 1.3 at the top of the
///        chart, so a rule written on 2Z is right for the light even-even nuclides and wrong
///        elsewhere. `em::SteppedHadron` carries A for this.
template <typename real_t>
__host__ __device__ inline real_t hadron_formfactor(const ParticleDef<real_t>& pd,
                                                    int mass_number = 0) {
  if (pd.is_lepton) { return real_t(0); }
  constexpr real_t kGeV = real_t(1000);
  real_t x = real_t(0.8426) * kGeV;
  if (pd.spin == real_t(0) && pd.mass < kGeV) {
    x = real_t(0.736) * kGeV;
  } else if (pd.mass > kGeV) {
    // G4NistManager::GetA27(Z) is A^0.27 of the projectile's own mass number.
    const int iz = static_cast<int>(fabs(pd.charge) + real_t(0.5));
    const int ia = (mass_number > 0) ? mass_number : 2 * iz;
    if (iz > 1) { x /= pow(static_cast<real_t>(ia), real_t(0.27)); }
  }
  return real_t(2) * units::electron_mass_c2<real_t>() / (x * x);
}

/// Delta-ray production rate, 1/mm, from whichever model applies at this energy.
///
/// This exists next to the sampler rather than in hadron_ionisation.cuh because the two must
/// agree on the lower bound of the integral. bethe_bloch_delta_xs integrates from the cut;
/// both Bragg models integrate from max(cut, 0.25 keV * massRate) and sample from the same
/// place. Take the rate from one and the spectrum from the other and the deltas come out at
/// the wrong rate below the floor - silently, because each half is individually right.
///
/// The one thing the two halves do *not* share is the spin term: G4BraggModel's cross section
/// carries it and G4BraggModel::SampleSecondaries does not. That is Geant4's own inconsistency,
/// it is transcribed rather than tidied, and it is numerically nothing - at the 2 MeV top of
/// the Bragg range the term is ~1e-9 of a cross section whose leading term is ~1e3.
///
/// The alpha carries one more factor, and it is a different one in each regime. Both models'
/// CrossSectionPerVolume scale the rate by an effective helium charge squared over the nominal
/// 4 - the ion is not fully stripped at these energies - but they do not use the same function:
///
///   G4BraggIonModel   HeEffChargeSquare(zeff, E_alpha) / chargeSquare
///   G4BetheBlochModel G4EmCorrections::EffectiveChargeSquareRatio(...) / chargeSquare, i.e.
///                     G4ionEffectiveCharge on the *reduced* energy E*m_p/m
///
/// They are the same Ziegler polynomial evaluated on differently scaled arguments, so they are
/// close but not equal, and using either one everywhere is wrong by about a per cent in the
/// half of the range it does not belong to. Both were caught by the test - the Bragg factor at
/// 2 MeV (2.8%) and, once that was in, the Bethe-Bloch factor at 8.9 MeV (1.4%), which is how
/// the second one was found at all.
///
/// The dE/dx carries the same scaling, so omitting it would put the delta rate out of step with
/// the stopping power the range table is built from. The *sampler* does not scale: an effective
/// charge changes how often a delta is made, not the spectrum of one that is, and neither
/// model's SampleSecondaries touches it.
template <typename real_t>
__host__ __device__ inline real_t hadron_delta_xs(const data::Material<real_t>& m,
                                                  ParticleType type, real_t kinetic, real_t cut,
                                                  real_t max_energy) {
  const ParticleDef<real_t> pd = particle_def<real_t>(type);
  if (pd.mass <= real_t(0) || kinetic <= real_t(0)) { return real_t(0); }
  const HadronIoniModel model = hadron_ioni_model(type, pd, kinetic);
  // The lower bound of the transfer window, and the low-energy models do not agree on it.
  // Both Bragg models floor the cut at 0.25 keV * massRate, G4ICRU73QOModel at 5 keV *
  // massRate, and Bethe-Bloch takes the cut as given. The floor is where each parameterisation
  // stops having anything to say, so it belongs to the model and not to the particle: two
  // negative hadrons of the same mass get different floors either side of the boundary.
  const real_t mass_rate = pd.mass / units::proton_mass_c2<real_t>();
  real_t lo = cut;
  if (is_bragg_family(model)) {
    lo = fmax(cut, bragg_lowest_energy<real_t>() * mass_rate);
  } else if (model == HadronIoniModel::kICRU73QO) {
    lo = fmax(cut, qo_lowest_energy<real_t>() * mass_rate);
  }
  real_t xs = bethe_bloch_delta_xs(m, type, kinetic, lo, max_energy);
  // Effective-charge scaling, for every species G4ionIonisation is registered for rather than
  // for the alpha alone - He3 and a generic ion need it too, and needed it before they were
  // transported. The two models scale differently: G4BraggIonModel divides out the bare charge
  // and multiplies by HeEffChargeSquare at the helium-equivalent energy, G4BetheBlochModel by
  // G4ionEffectiveCharge on the reduced one.
  if (uses_ion_ionisation(type)) {
    const real_t q2 = pd.charge * pd.charge;
    if (model == HadronIoniModel::kBraggIon) {
      xs *= he_eff_charge_square(m.z_eff, kinetic) / q2;
    } else if (model == HadronIoniModel::kBetheBloch) {
      const real_t q = ion_effective_charge(m, pd, kinetic);
      xs *= q * q / q2;
    }
  }
  return xs;
}

// The result type is em::DeltaRay from electron_processes.cuh, shared with the Moller-Bhabha
// sampler rather than duplicated. A stepper handling both leptons and hadrons reads the same
// five fields either way, and two structurally identical "delta ray" types with different field
// names is the sort of thing that compiles for a year and then swaps two of them.
//
// `produced == false` means no secondary at all: an empty kinematic window, or a form-factor
// rejection. In both cases the primary comes back untouched - Geant4 returns from
// SampleSecondaries without proposing a change, and the difference between suppressing a
// transfer and redistributing it is exactly the thing a rewrite gets wrong.

/// Samples the knock-on electron with whichever model applies at this energy.
///
/// @param cut  the electron production threshold in MeV: transfers below it are already in the
///             continuous loss and must not be double-counted as a secondary here.
template <typename real_t, typename Rng>
__host__ __device__ inline DeltaRay<real_t> sample_hadron_delta(ParticleType type, real_t kinetic,
                                                               real_t cut,
                                                               const Vec3<real_t>& dir,
                                                               Rng& rng) {
  DeltaRay<real_t> out{dir, real_t(0), dir, kinetic, false};

  const ParticleDef<real_t> pd = particle_def<real_t>(type);
  if (pd.mass <= real_t(0) || kinetic <= real_t(0)) { return out; }

  const real_t me = units::electron_mass_c2<real_t>();
  const real_t tmax = hadron_max_secondary_energy(pd, kinetic);
  const HadronIoniModel model = hadron_ioni_model(type, pd, kinetic);
  // "Low energy" here means one of the three parameterised models, all of which sample the
  // same spin-0 form: a 1/T^2 spectrum accepted against (1 - beta^2 T/Tmax), majorant one.
  // G4BraggModel, G4BraggIonModel and G4ICRU73QOModel have identical SampleSecondaries bodies
  // apart from the floor.
  //
  // NOT included: G4MuBetheBlochModel. A muon above 200 keV samples with a radiative
  // correction term this does not carry, so it would fall through to the plain Bethe-Bloch
  // branch below. That is a real gap and it is inert only because muons are not transported;
  // the delta *rate* for one is already right (em::mu_bethe_bloch_delta_xs, tests/test_muon).
  const bool low_energy = (model != HadronIoniModel::kBetheBloch
                           && model != HadronIoniModel::kMuBetheBloch);
  const bool bragg = low_energy;

  // The lower bound. 0.25 keV * massRate for the Bragg models, 5 keV * massRate for ICRU73QO,
  // the cut as given for Bethe-Bloch - each parameterisation's own floor.
  real_t xmin = cut;
  if (low_energy) {
    const real_t mass_rate = pd.mass / units::proton_mass_c2<real_t>();
    const real_t floor = is_bragg_family(model) ? bragg_lowest_energy<real_t>()
                                                : qo_lowest_energy<real_t>();
    xmin = fmax(cut, floor * mass_rate);
  }
  // xmax is min(tmax, maxEnergy); maxEnergy is DBL_MAX from G4VEnergyLossProcess::PostStepDoIt,
  // so it is tmax. Note `f` below divides by tmax, not by xmax - they coincide here, but the
  // two are different quantities and writing xmax there would be wrong the day they diverge.
  const real_t xmax = tmax;
  if (xmin >= xmax) { return out; }

  const real_t etot = kinetic + pd.mass;
  const real_t etot2 = etot * etot;
  const real_t beta2 = kinetic * (kinetic + real_t(2) * pd.mass) / etot2;
  const bool spin = (!bragg && pd.spin > real_t(0));

  // The majorant. With the spin term the rejection function exceeds one, so the majorant has
  // to carry that term at its largest; Bragg's is exactly one.
  real_t fmaj = real_t(1);
  if (spin) { fmaj += real_t(0.5) * xmax * xmax / etot2; }

  real_t delta = 0, f = 0, f1 = 0;
  // 1/T^2 between xmin and xmax by inversion, then accept against (1 - beta^2 T/Tmax [+ spin]).
  // Bounded rather than Geant4's do/while: acceptance is high - the rejection function is close
  // to one over nearly the whole range - so a hundred tries is many orders of magnitude of
  // headroom, and a device kernel that can spin forever is worse than one that gives up.
  constexpr int kMaxTries = 100;
  bool accepted = false;
  for (int i = 0; i < kMaxTries && !accepted; ++i) {
    const real_t r0 = rng.uniform();
    const real_t r1 = rng.uniform();
    delta = xmin * xmax / (xmin * (real_t(1) - r0) + xmax * r0);
    f = real_t(1) - beta2 * delta / tmax;
    f1 = real_t(0);
    if (spin) {
      f1 = real_t(0.5) * delta * delta / etot2;
      f += f1;
    }
    accepted = (fmaj * r1 <= f);
  }
  if (!accepted) { return out; }

  // The projectile form factor, Bethe-Bloch only. A rejection here yields no secondary and the
  // primary keeps its energy; it is not a resample.
  if (!bragg) {
    const real_t x = hadron_formfactor(pd) * delta;
    if (x > real_t(1e-6)) {
      const real_t x1 = real_t(1) + x;
      real_t grej = real_t(1) / (x1 * x1);
      if (spin) {
        const real_t x2 = real_t(0.5) * me * delta / (pd.mass * pd.mass);
        grej *= (real_t(1) + pd.mag_moment2 * (x2 - f1 / f) / (real_t(1) + x2));
      }
      if (rng.uniform() > grej) { return out; }
    }
  }

  // Two-body kinematics fix the electron's angle from its energy alone.
  const real_t dmom = sqrt(delta * (delta + real_t(2) * me));
  const real_t pmom = sqrt(kinetic * (kinetic + real_t(2) * pd.mass));
  real_t cost = (dmom > real_t(0) && pmom > real_t(0)) ? delta * (etot + me) / (dmom * pmom)
                                                      : real_t(1);
  cost = fmin(cost, real_t(1));
  const real_t sint = sqrt(fmax(real_t(0), (real_t(1) - cost) * (real_t(1) + cost)));
  const real_t phi = units::twopi<real_t>() * rng.uniform();

  out.delta_dir = rotate_uz(Vec3<real_t>{sint * cos(phi), sint * sin(phi), cost}, dir);
  out.delta_ekin = delta;
  out.produced = true;

  // The primary recoils: momentum conservation, not just an energy subtraction. For a proton
  // the deflection is minute, but it is the model's own line (finalP = p - p_delta).
  out.primary_ekin = kinetic - delta;
  const Vec3<real_t> pfin = pmom * dir - dmom * out.delta_dir;
  out.primary_dir = (mag(pfin) > real_t(0)) ? normalize(pfin) : dir;
  return out;
}

}  // namespace g4gpu::em
