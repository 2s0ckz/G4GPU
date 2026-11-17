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
/// P26: `G4NistManager::GetA27(iz)` is the ELEMENT's mean atomic mass to the 0.27
/// (`em::nist_a27`), indexed by the projectile's charge - not the projectile's own mass number,
/// which this took (as 2Z when not given) until P26. For the alpha that was (4/4.0026)^0.27, a
/// part in 6,000 of the form-factor scale; for He3, which passed nothing, the same; for a real
/// nucleus it would have been A/A_mean of its element (`em::nist_a27` is muon_radiative.cuh's,
/// which `hadron_ionisation.cuh` includes). Geant4 knows only Z here
/// (`G4lrint(std::abs(q))`), so a C13 projectile gets carbon's 12.0107^0.27 like a C12.
template <typename real_t>
__host__ __device__ inline real_t hadron_formfactor(const ParticleDef<real_t>& pd) {
  if (pd.is_lepton) { return real_t(0); }
  constexpr real_t kGeV = real_t(1000);
  real_t x = real_t(0.8426) * kGeV;
  if (pd.spin == real_t(0) && pd.mass < kGeV) {
    x = real_t(0.736) * kGeV;
  } else if (pd.mass > kGeV) {
    const int iz = static_cast<int>(fabs(pd.charge) + real_t(0.5));
    if (iz > 1) { x /= nist_a27<real_t>(iz); }
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

/// Samples the knock-on electron for the projectile DEFINITION `pd` with the model `model`.
///
/// P26b split this out of `sample_hadron_delta` so that a real nucleus can be sampled on its own
/// definition - G4BetheBlochModel::SampleSecondaries calls `MaxSecondaryEnergy(
/// dp->GetDefinition(), ...)`, whose `SetupParameters(pd)` sets `mass`, `spin`, `magMoment2` and
/// `formfact` from the ION (an O16, not G4GenericIon), and G4BraggIonModel's `SetParticle` does the
/// same for its mass and `massRate` - while the model is chosen by the process the ion shares with
/// G4GenericIon, at the SCALED energy. For every species with a definition of its own the two
/// arguments are `particle_def(type)` and `hadron_ioni_model(type, pd, kinetic)`, which is
/// exactly what `sample_hadron_delta` passes, so its uniforms and answers are the ones it drew
/// before the split.
///
/// @param cut  the electron production threshold in MeV: transfers below it are already in the
///             continuous loss and must not be double-counted as a secondary here.
template <typename real_t, typename Rng>
__host__ __device__ inline DeltaRay<real_t> sample_delta_for_definition(
    const ParticleDef<real_t>& pd, HadronIoniModel model, real_t kinetic, real_t cut,
    const Vec3<real_t>& dir, Rng& rng) {
  DeltaRay<real_t> out{dir, real_t(0), dir, kinetic, false};

  if (pd.mass <= real_t(0) || kinetic <= real_t(0)) { return out; }

  const real_t me = units::electron_mass_c2<real_t>();
  const real_t tmax = hadron_max_secondary_energy(pd, kinetic);
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
    // Bethe-Bloch loops `while (fmax*rndm[1] > f)`; the three parameterised models loop
    // `while (grej*rndm[1] >= f)` with grej = 1. The two differ only at equality, a set of
    // measure zero, and are transcribed each as written (P26b).
    accepted = bragg ? (fmaj * r1 < f) : (fmaj * r1 <= f);
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

/// Samples the knock-on electron with whichever model applies at this energy, for a species
/// that is its own definition (everything but a real nucleus: see `sample_ion_delta`).
template <typename real_t, typename Rng>
__host__ __device__ inline DeltaRay<real_t> sample_hadron_delta(ParticleType type, real_t kinetic,
                                                               real_t cut,
                                                               const Vec3<real_t>& dir,
                                                               Rng& rng) {
  const ParticleDef<real_t> pd = particle_def<real_t>(type);
  return sample_delta_for_definition<real_t>(pd, hadron_ioni_model(type, pd, kinetic), kinetic,
                                             cut, dir, rng);
}

// ------------------------------------------------------------------ a real nucleus's delta rays
//
// P26b. A real nucleus (an O16, a Li7 - anything stepped as `kGenericIon`) shares G4GenericIon's
// `ionIoni`, so its delta rays are two different particles' business, and this pair keeps them
// apart the way G4VEnergyLossProcess does:
//
//   * the RATE is GenericIon's. `PostStepGetPhysicalInteractionLength` reads GenericIon's lambda
//     table at `preStepScaledEnergy = E*massRatio` (massRatio = 938.2723 MeV / the ion's mass) and
//     multiplies it by `fFactor`, which for an `isIon` particle tracks
//     `currentModel->ChargeSquareRatio(track)` - G4EmCorrections::EffectiveChargeSquareRatio at
//     the ion's own energy. The table's nodes are G4BraggIonModel's (below 2 MeV scaled) or
//     G4BetheBlochModel's CrossSectionPerVolume for G4GenericIon itself: mass 938.2723 MeV, charge
//     1, SPIN 1/2, no effective-charge factor (both models apply one for `isAlpha` alone), and
//     Bragg's 0.25 keV floor at GenericIon's massRate. So the ion's own spin is not in its rate.
//   * the SPECTRUM is the ion's. PostStepDoIt hands the model the dynamic particle, and both
//     models re-derive mass, spin, magMoment2, formfact and massRate from its definition.
//
// The rate is the model at the pre-step energy, not GenericIon's lambda table and not the
// integral approach's rejection against the post-step lambda: the convention every hadron and
// lepton in this stepper is drawn with (docs/PORTED.md, G4hIonisation's row; docs/RISK.md V78),
// applied to the ion as to the alpha. ref/oracle/ion_delta.csv carries both of Geant4's numbers
// (the model, `ComputeCrossSectionPerVolume`, and the table, `GetCrossSectionPerVolume`) so the
// test can say how far apart they are at the nuclides and energies the beams reach.

/// An ion's delta-ray production rate, 1/mm: GenericIon's cross section at the scaled energy
/// times the dynamic charge-square ratio. Zero below the window, i.e. while GenericIon's tmax
/// at the scaled energy is at or below the (floored) cut.
template <typename real_t>
__host__ __device__ inline real_t ion_delta_xs(const data::Material<real_t>& m,
                                               const SteppedHadron<real_t>& h, real_t kinetic,
                                               real_t cut, real_t max_energy) {
  if (!h.is_real_ion() || !(h.def.mass > real_t(0)) || kinetic <= real_t(0)) { return real_t(0); }
  const ParticleDef<real_t> gi = particle_def<real_t>(ParticleType::kGenericIon);
  const real_t es = kinetic * hadron_mass_ratio(h);
  const HadronIoniModel model = hadron_ioni_model(ParticleType::kGenericIon, gi, es);
  real_t lo = cut;
  if (is_bragg_family(model)) {
    lo = fmax(cut, bragg_lowest_energy<real_t>() * (gi.mass / units::proton_mass_c2<real_t>()));
  }
  const real_t xs = bethe_bloch_delta_xs(m, ParticleType::kGenericIon, es, lo, max_energy);
  if (!(xs > real_t(0))) { return real_t(0); }
  return xs * hadron_charge_sq_ratio(m, h, kinetic);
}

/// Samples an ion's knock-on electron: the model GenericIon's process selects at the scaled
/// energy, run on the ion's own definition (`h.def`: ENSDFSTATE's spin and moment since P26b).
template <typename real_t, typename Rng>
__host__ __device__ inline DeltaRay<real_t> sample_ion_delta(const SteppedHadron<real_t>& h,
                                                            real_t kinetic, real_t cut,
                                                            const Vec3<real_t>& dir, Rng& rng) {
  const ParticleDef<real_t> gi = particle_def<real_t>(ParticleType::kGenericIon);
  const HadronIoniModel model =
      hadron_ioni_model(ParticleType::kGenericIon, gi, kinetic * hadron_mass_ratio(h));
  return sample_delta_for_definition<real_t>(h.def, model, kinetic, cut, dir, rng);
}

}  // namespace g4gpu::em
