// G4KineticTrack's resonance machinery: the actual widths, the total, and the residual lifetime.
//
// Transcribed from G4KineticTrack.{hh,cc} (hadronic/util, 11.1.1), G4SampleResonance.cc and
// G4Integrator.icc, over the decay tables in `decay_tables.hh`.
//
// This is what decides WHEN a resonance in the cascade decays and WHICH channel it takes.
// `G4BinaryCascade` asks `SampleResidualLifetime` once per resonance it creates and schedules a
// `G4BCDecay` at that time; `Decay()` then picks a channel in proportion to the same array.
//
// ## The width of a channel is a ratio of two phase-space integrals, and one of them is wrong
//
// For each decay channel the constructor computes
//
//     theActualWidth[index] = thePoleWidth * (thePoleMass/theActualMass) * (theActualMom/thePoleMom)
//
// where `thePoleWidth` is the channel's branching ratio times the parent's PDG width, and the
// two momenta are the two-body CM momentum of the channel's daughters - evaluated at the track's
// ACTUAL mass and at the parent's POLE mass. When a daughter is itself short-lived the momentum
// is not a closed form but an integral of that momentum over the daughter's Breit-Wigner, done
// with `G4Integrator::Simpson` at 100 iterations, i.e. 201 evaluations.
//
// ## TWO OF THE FOUR BRANCHES ARE DEAD, and that is a measurement and not a reading
//
// The constructor splits on how many of a channel's daughters are short-lived. MEASURED over
// every one of the 563 channels in `decay_tables.hh`: of the 491 two-body channels, 159 have no
// short-lived daughter and 332 have exactly one - and **none has two**. Of the 70 three-body
// channels, **all 70 have none**. So:
//
//   * `IntegrateCMMomentum2`, `IntegrandFunction3` and `IntegrandFunction4` - the two-resonance
//     branch - are never entered by the binary cascade, and
//   * the three-body `nShortLived >= 1` branch, with its second swap and its `theDaughterMass[0]
//     += theDaughterMass[2]`, is never entered either.
//
// The reason is `IsShortLived`: it is true for the baryon resonances and for the rho and the
// omega, and false for the pion, the nucleon, the eta, the kaons and the lambda. Every channel
// in the closure pairs at most one short-lived daughter with long-lived ones. The assertion is
// in tests/test_bic_imr.cu over the port's own table, so a Geant4 release that adds a
// resonance-to-two-resonances channel fails there rather than running code nothing has checked.
//
// Both branches are transcribed anyway, and both carry a bug that would show the moment they
// woke up. `IntegrateCMMomentum2` reads `theActualMass` for its upper limit and never
// `G4KineticTrack_Gmass`, so its "pole" call integrates over the actual mass's range with the
// pole mass inside the integrand - unlike the one-resonance branch, whose second call takes
// `poleMass` and uses it for both. And `IntegrandFunction3` has no `std::max(...,0.0)` under its
// square root where `IntegrandFunction1` and `2` do, so with `Gmass` at the pole and the outer
// variable past it the inner limit `mass - xmass` goes negative and Simpson integrates backwards
// over a possibly negative radicand. MEASURED: replacing the upper limit with `gmass` - the
// obvious fix - changes none of the 15,147 compared widths, which is the same statement as
// "never entered". docs/RISK.md V149.
//
// ## Four things in here are inert, and each is recorded rather than simplified away
//
// MEASURED, each by removing it and running the 15,147 compared widths and 2,160 lifetimes:
//
//   * **`BrWig`'s `1/twopi`.** It multiplies both the actual-mass and the pole-mass integral, and
//     `theActualWidth` uses only their RATIO, so the normalisation cancels exactly. Replacing it
//     with `1/pi` changes nothing. It is kept because `G4SampleResonance` uses a differently
//     normalised Breit-Wigner for the same resonance and a reader comparing them needs to see
//     which is which.
//   * **`EvaluateTotalActualWidth`'s backwards loop.** Summing forwards changes nothing: no
//     species has enough channels of sufficiently different size for the order to move a bit.
//   * **`GetMinimumMass`'s `std::min(highestChannelProbability, thresholdChannelProbability)`.**
//     Comparing against the flat 0.10 instead changes nothing, because every species in the
//     closure has a channel above 0.10 and the fallback path is never taken.
//   * **the gamma exclusion, `if (!minMass) minMass = DBL_MAX`.** The only channels with a photon
//     in them have branching ratios of 0.01, which the 0.10 threshold rejects first. Removing the
//     exclusion changes nothing here - but it would be the delta0 and delta+ floors that moved,
//     from 1074.5 and 1073.2 MeV down to the nucleon mass, if a release raised that ratio.
//
// ## REFUSED, by name

//
//   * the **G4Integrator adaptive methods**. Only `Simpson(ptrT, f, a, b, n)` is reachable from
//     G4KineticTrack and only that one is here.
//   * **four-daughter channels**. `G4KineticTrack`'s constructor sends anything that is not two
//     or three daughters down the `theActualWidth = BR * PDGWidth` branch, which this file
//     reproduces, but `Decay`'s `ManyBodyDecayIt` is not here. Nothing in the transitive closure
//     of `decay_tables.hh` has four, and `tools/extract_bic_decay.pl` asserts that.
#ifndef G4GPU_BIC_IMR_DECAY_CUH
#define G4GPU_BIC_IMR_DECAY_CUH

#include <cmath>

#include "core/units.cuh"
#include "physics/hadronic/bic/im_r/decay_tables.hh"
#include "physics/hadronic/bic/im_r/resonance_fs.cuh"

namespace g4gpu::bic::imr {

/// What the decay machinery could not do.
struct DecayRefusal {
  bool unknown_species = false;   ///< a PDG code that is not in the closure
  bool too_many_daughters = false;///< a four-body channel; see the file header
  bool too_many_channels = false; ///< more channels than the caller's array can hold
  int refused_pdg = 0;
  __host__ __device__ bool any() const {
    return unknown_species || too_many_daughters || too_many_channels;
  }
};

/// The index of a PDG code in `decay_tables.hh`, or -1.
///
/// A linear scan over 94 entries. The table is in the order the closure found the species, which
/// is not sorted, and sorting it would lose the property that the seeds come first - which is
/// what makes the generated file readable next to the dump it came from.
__host__ __device__ inline int decay_species_index(int pdg) {
  const int* codes = decay_species_pdg();
  for (int i = 0; i < kDecaySpeciesCount; ++i) {
    if (codes[i] == pdg) { return i; }
  }
  return -1;
}

/// The FIRST thing `G4KineticTrack`'s constructor does, before it looks at a decay table:
///
///     if (G4KaonZero::KaonZero() == theDefinition || G4AntiKaonZero::AntiKaonZero() == ...)
///     { if (G4UniformRand()<0.5) theDefinition = KaonZeroShort; else KaonZeroLong; }
///
/// It consumes one uniform and it CHANGES THE PARTICLE'S IDENTITY - a K0 track is a K0S or a K0L
/// track from the moment it is built, with two decay channels or six. The cascade reaches it:
/// N(1650)0, N(1710)0, N(1720)0 and N(1990)0 all have a Lambda K0 channel, so the substitution
/// runs the first time one of those decays. docs/RISK.md V150.
///
/// Returns the substituted PDG code; everything else passes through untouched.
template <typename Rng>
__host__ __device__ inline int kinetic_track_substitute_k0(int pdg, Rng& rng) {
  if (pdg == 311 || pdg == -311) { return (rng.uniform() < 0.5) ? 310 : 130; }
  return pdg;
}

/// `G4SampleResonance::GetMinimumMass(p)`, from the cached column.
///
/// Geant4 caches this in a thread-local map keyed by particle definition and computes it once;
/// the column is that cached value, dumped. `compute_min_mass` below is the recursion itself,
/// and tests/test_bic_imr.cu checks it against all 94 columns rather than assuming they agree.
__host__ __device__ inline double species_min_mass(int pdg, DecayRefusal& ref) {
  const int i = decay_species_index(pdg);
  if (i < 0) {
    ref.unknown_species = true;
    ref.refused_pdg = pdg;
    return 0.0;
  }
  return decay_species_min_mass()[i];
}

/// `G4SampleResonance::GetMinimumMass`, the recursion.
///
/// A stable particle's minimum mass is its PDG mass. A short-lived one's is the smallest sum of
/// daughter minimum masses over the channels whose branching ratio exceeds 0.10 - and if no
/// channel does, the sum for the single most probable channel whatever its ratio. A daughter
/// whose own minimum mass is ZERO - the photon - is replaced by DBL_MAX, which is how the gamma
/// channel is kept from setting a resonance's floor at the nucleon mass.
///
/// `depth` is the port's own recursion bound and has no counterpart in Geant4, which relies on
/// the decay graph being acyclic. It is set high enough that the closure cannot reach it.
__host__ __device__ inline double compute_min_mass(int pdg, DecayRefusal& ref, int depth = 0) {
  const int i = decay_species_index(pdg);
  if (i < 0) {
    ref.unknown_species = true;
    ref.refused_pdg = pdg;
    return 0.0;
  }
  if (decay_species_shortlived()[i] == 0 || depth > 12) { return decay_species_mass()[i]; }
  const int n = decay_species_n_channels()[i];
  const int first = decay_species_first_channel()[i];
  const double kThreshold = 0.10;
  bool found_above = false;
  double min_most_probable = 0.0;
  double highest = 0.0;
  double min_resonance = 1.7976931348623157e308;  // DBL_MAX
  for (int c = 0; c < n; ++c) {
    const double br = decay_channel_br()[first + c];
    // The guard is `decayBr > std::min(highestChannelProbability, thresholdChannelProbability)`,
    // so a channel is examined when it beats EITHER the running best or the 0.10 floor,
    // whichever is smaller - not both.
    const double bar = (highest < kThreshold) ? highest : kThreshold;
    if (!(br > bar)) { continue; }
    const int nd = decay_channel_n_daughters()[first + c];
    double channel_mass = 0.0;
    for (int j = 0; j < nd; ++j) {
      const int d = decay_channel_daughters()[(first + c) * kDecayMaxDaughters + j];
      double dm = compute_min_mass(d, ref, depth + 1);
      if (dm == 0.0) { dm = 1.7976931348623157e308; }  // exclude the gamma channel
      channel_mass += dm;
    }
    if (br > highest) {
      highest = br;
      min_most_probable = channel_mass;
    }
    if (br > kThreshold) {
      found_above = true;
      if (channel_mass < min_resonance) { min_resonance = channel_mass; }
    }
  }
  if (!found_above) { min_resonance = min_most_probable; }
  return min_resonance;
}

/// `G4KineticTrack::BrWig` - a Breit-Wigner normalised by twopi, NOT by its own integral.
__host__ __device__ inline double kt_brwig(double gamma, double rmass, double mass) {
  return (gamma / ((mass - rmass) * (mass - rmass) + gamma * gamma / 4.0)) / u::twopi<double>();
}

/// `G4KineticTrack::EvaluateCMMomentum` - zero below threshold rather than imaginary.
__host__ __device__ inline double evaluate_cm_momentum(double mass, double m0, double m1) {
  if ((m0 + m1) < mass) {
    return 1.0 / (2.0 * mass) *
           std::sqrt(((mass * mass) - (m0 + m1) * (m0 + m1)) *
                     ((mass * mass) - (m0 - m1) * (m0 - m1)));
  }
  return 0.0;
}

/// `G4Integrator::Simpson(T* ptrT, F f, xInitial, xFinal, iterationNumber)`.
///
/// Written out because the order of the additions is the answer's last three digits: the
/// endpoints are halved into `mean` first, the half-step points accumulate into `sum`, and `sum`
/// is doubled into `mean` once at the end rather than per term.
template <typename F>
__host__ __device__ inline double simpson(const F& f, double x_initial, double x_final, int n) {
  const double step = (x_final - x_initial) / n;
  double x = x_initial;
  double x_plus = x_initial + 0.5 * step;
  double mean = (f(x_initial) + f(x_final)) * 0.5;
  double sum = f(x_plus);
  for (int i = 1; i < n; ++i) {
    x += step;
    x_plus += step;
    mean += f(x);
    sum += f(x_plus);
  }
  mean += 2.0 * sum;
  return mean * step / 3.0;
}

/// The members the four integrand functions read: two daughter masses and widths after the
/// constructor's swaps, the track's actual mass, the definition's pole mass, and the two
/// file-scope globals `G4KineticTrack_Gmass` and `G4KineticTrack_xmass1` that the two-resonance
/// branch communicates through.
struct WidthCtx {
  double dmass[3] = {0.0, 0.0, 0.0};
  double dwidth[3] = {0.0, 0.0, 0.0};
  double actual_mass = 0.0;
  double pole_mass = 0.0;
  double gmass = 0.0;
  double xmass1 = 0.0;
};

/// `G4KineticTrack::IntegrandFunction1` - the CM momentum at the ACTUAL mass, weighted by the
/// second daughter's Breit-Wigner. The `std::max(..., 0.0)` is Geant4's.
struct Integrand1 {
  const WidthCtx* c;
  __host__ __device__ double operator()(double xmass) const {
    const double mass = c->actual_mass;
    const double mass1 = c->dmass[0];
    const double prod = ((mass * mass) - (mass1 + xmass) * (mass1 + xmass)) *
                        ((mass * mass) - (mass1 - xmass) * (mass1 - xmass));
    return (1.0 / (2 * mass)) * std::sqrt(prod > 0.0 ? prod : 0.0) *
           kt_brwig(c->dwidth[1], c->dmass[1], xmass);
  }
};

/// `G4KineticTrack::IntegrandFunction2` - the same at the POLE mass.
struct Integrand2 {
  const WidthCtx* c;
  __host__ __device__ double operator()(double xmass) const {
    const double mass = c->pole_mass;
    const double mass1 = c->dmass[0];
    const double prod = ((mass * mass) - (mass1 + xmass) * (mass1 + xmass)) *
                        ((mass * mass) - (mass1 - xmass) * (mass1 - xmass));
    return (1.0 / (2 * mass)) * std::sqrt(prod > 0.0 ? prod : 0.0) *
           kt_brwig(c->dwidth[1], c->dmass[1], xmass);
  }
};

/// `G4KineticTrack::IntegrandFunction3` - the inner integrand of the two-resonance branch. NOTE
/// the absent `std::max`: this one takes the square root of whatever the product is.
struct Integrand3 {
  const WidthCtx* c;
  __host__ __device__ double operator()(double xmass) const {
    const double mass = c->gmass;
    return (1.0 / (2 * mass)) *
           std::sqrt(((mass * mass) - (c->xmass1 + xmass) * (c->xmass1 + xmass)) *
                     ((mass * mass) - (c->xmass1 - xmass) * (c->xmass1 - xmass))) *
           kt_brwig(c->dwidth[1], c->dmass[1], xmass);
  }
};

/// `G4KineticTrack::IntegrandFunction4` - the outer one, which writes the shared `xmass1` and
/// then runs its own 100-iteration Simpson over `IntegrandFunction3`. 201 x 201 evaluations.
struct Integrand4 {
  WidthCtx* c;
  __host__ __device__ double operator()(double xmass) const {
    const double mass = c->gmass;
    const double mass1 = c->dmass[0];
    const double gamma1 = c->dwidth[0];
    c->xmass1 = xmass;
    const Integrand3 inner{c};
    return kt_brwig(gamma1, mass1, xmass) * simpson(inner, 0.0, mass - xmass, 100);
  }
};

/// `G4KineticTrack::IntegrateCMMomentum(theLowerLimit)`.
__host__ __device__ inline double integrate_cm_momentum(const WidthCtx& c, double lower) {
  const double upper = c.actual_mass - c.dmass[0];
  if (lower >= upper) { return 0.0; }
  const Integrand1 f{&c};
  return simpson(f, lower, upper, 100);
}

/// `G4KineticTrack::IntegrateCMMomentum(theLowerLimit, poleMass)`.
__host__ __device__ inline double integrate_cm_momentum_pole(const WidthCtx& c, double lower,
                                                             double pole_mass) {
  const double upper = pole_mass - c.dmass[0];
  if (lower >= upper) { return 0.0; }
  const Integrand2 f{&c};
  return simpson(f, lower, upper, 100);
}

/// `G4KineticTrack::IntegrateCMMomentum2` - the two-resonance branch. Its upper limit is
/// `theActualMass` whichever mass `gmass` currently holds; see the file header.
__host__ __device__ inline double integrate_cm_momentum2(WidthCtx& c) {
  const double upper = c.actual_mass;
  if (0.0 >= upper) { return 0.0; }
  const Integrand4 f{&c};
  return simpson(f, 0.0, upper, 100);
}

/// `G4KineticTrack`'s `(definition, time, position, momentum)` constructor, the part that fills
/// `theActualWidth[nChannels]`.
///
/// `actual_mass` is `GetActualMass()` = `sqrt(|the4Momentum.mag2()|)`. Returns the number of
/// channels written, or -1 on a refusal.
__host__ __device__ inline int kinetic_track_actual_widths(int pdg, double actual_mass,
                                                           double* out, int capacity,
                                                           DecayRefusal& ref) {
  // The caller passes the SUBSTITUTED code: for a K0 the identity has already changed and the
  // table this reads is K0S's or K0L's, never K0's own two one-daughter channels.
  const int i = decay_species_index(pdg);
  if (i < 0) {
    ref.unknown_species = true;
    ref.refused_pdg = pdg;
    return -1;
  }
  const int n = decay_species_n_channels()[i];
  if (n > capacity) {
    ref.too_many_channels = true;
    ref.refused_pdg = pdg;
    return -1;
  }
  const int first = decay_species_first_channel()[i];
  const double pole_mass = decay_species_mass()[i];
  const double mother_width = decay_species_width()[i];
  // Geant4 walks the channels BACKWARDS. The order matters only through the arithmetic each
  // channel does on its own, which is independent - but it is kept because a reader comparing
  // the two files should not have to check that.
  for (int index = n - 1; index >= 0; --index) {
    const int ch = first + index;
    const int nd = decay_channel_n_daughters()[ch];
    const double br = decay_channel_br()[ch];
    if (nd != 2 && nd != 3) {
      // nDaughters 1 - and 0, and 4 and more, which the closure does not contain.
      out[index] = br * mother_width;
      continue;
    }
    const double pole_width = br * mother_width;
    WidthCtx c;
    c.actual_mass = actual_mass;
    c.pole_mass = pole_mass;
    bool sl[3] = {false, false, false};
    int dcode[3] = {0, 0, 0};
    for (int j = 0; j < nd; ++j) {
      dcode[j] = decay_channel_daughters()[ch * kDecayMaxDaughters + j];
      const int dj = decay_species_index(dcode[j]);
      if (dj < 0) {
        ref.unknown_species = true;
        ref.refused_pdg = dcode[j];
        return -1;
      }
      c.dmass[j] = decay_species_mass()[dj];
      c.dwidth[j] = decay_species_width()[dj];
      sl[j] = decay_species_shortlived()[dj] != 0;
    }
    double actual_mom = 0.0;
    double pole_mom = 0.0;
    if (nd == 2) {
      if (!sl[0] && !sl[1]) {
        actual_mom = evaluate_cm_momentum(actual_mass, c.dmass[0], c.dmass[1]);
        pole_mom = evaluate_cm_momentum(pole_mass, c.dmass[0], c.dmass[1]);
      } else if (!sl[0] && sl[1]) {
        const double lower = species_min_mass(dcode[1], ref);
        actual_mom = integrate_cm_momentum(c, lower);
        pole_mom = integrate_cm_momentum_pole(c, lower, pole_mass);
      } else if (sl[0] && !sl[1]) {
        // G4SwapObj on both arrays, so the short-lived daughter ends up in slot 1 - but the
        // lower limit is still taken from DAUGHTER 0 of the channel, which is the one that just
        // moved. Both statements are Geant4's and they agree only because of the swap.
        const double tm = c.dmass[0]; c.dmass[0] = c.dmass[1]; c.dmass[1] = tm;
        const double tw = c.dwidth[0]; c.dwidth[0] = c.dwidth[1]; c.dwidth[1] = tw;
        const double lower = species_min_mass(dcode[0], ref);
        actual_mom = integrate_cm_momentum(c, lower);
        pole_mom = integrate_cm_momentum_pole(c, lower, pole_mass);
      } else {
        c.gmass = actual_mass;
        actual_mom = integrate_cm_momentum2(c);
        c.gmass = pole_mass;
        pole_mom = integrate_cm_momentum2(c);
      }
    } else {
      int n_short = 0;
      if (sl[0]) { ++n_short; }
      if (sl[1]) {
        ++n_short;
        const double tm = c.dmass[0]; c.dmass[0] = c.dmass[1]; c.dmass[1] = tm;
        const double tw = c.dwidth[0]; c.dwidth[0] = c.dwidth[1]; c.dwidth[1] = tw;
      }
      if (sl[2]) {
        ++n_short;
        const double tm = c.dmass[0]; c.dmass[0] = c.dmass[2]; c.dmass[2] = tm;
        const double tw = c.dwidth[0]; c.dwidth[0] = c.dwidth[2]; c.dwidth[2] = tw;
      }
      if (n_short == 0) {
        c.dmass[1] += c.dmass[2];
        actual_mom = evaluate_cm_momentum(actual_mass, c.dmass[0], c.dmass[1]);
        pole_mom = evaluate_cm_momentum(pole_mass, c.dmass[0], c.dmass[1]);
      } else {
        // "need the shortlived particle in slot 1! (very bad style...)" - Geant4's own comment.
        const double tm = c.dmass[0]; c.dmass[0] = c.dmass[1]; c.dmass[1] = tm;
        const double tw = c.dwidth[0]; c.dwidth[0] = c.dwidth[1]; c.dwidth[1] = tw;
        c.dmass[0] += c.dmass[2];
        actual_mom = integrate_cm_momentum(c, 0.0);
        pole_mom = integrate_cm_momentum_pole(c, 0.0, pole_mass);
      }
    }
    const double mass_ratio = pole_mass / actual_mass;
    const double mom_ratio = actual_mom / pole_mom;
    out[index] = pole_width * mass_ratio * mom_ratio;
  }
  return n;
}

/// `G4KineticTrack::EvaluateTotalActualWidth` - summed BACKWARDS, which is the order the digits
/// come out in.
__host__ __device__ inline double evaluate_total_actual_width(const double* width, int n) {
  double total = 0.0;
  for (int index = n - 1; index >= 0; --index) { total += width[index]; }
  return total;
}

/// `G4KineticTrack::SampleResidualLifetime`.
///
/// `tau = hbar_Planck * (-1/totalWidth)` - a NEGATIVE tau multiplied by the log of a uniform,
/// which is also negative, so the product is positive. Dilated by the track's Lorentz gamma,
/// `the4Momentum.gamma()`, which is `e/mag()`.
template <typename Rng>
__host__ __device__ inline double sample_residual_lifetime(double total_actual_width,
                                                           double lorentz_gamma, Rng& rng) {
  // CLHEP DERIVES hbar_Planck as h_Planck/twopi and hbarc as hbar_Planck*c_light, so the
  // constant this needs is hbarc/c_light and not a pasted decimal - the same rule
  // `xsec_annihilation.cuh` records for hbarc itself, where the PDG's own value put every
  // cross section 1.25e-7 out. In MeV ns it is 6.582e-13.
  const double hbar = u::hbarc<double>() / u::c_light<double>();
  const double tau = hbar * (-1.0 / total_actual_width);
  return tau * std::log(rng.uniform()) * lorentz_gamma;
}

}  // namespace g4gpu::bic::imr

#endif
