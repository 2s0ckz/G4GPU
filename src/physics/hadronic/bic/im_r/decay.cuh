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
  bool below_threshold = false;   ///< Pmx would have thrown: the daughters do not fit
  bool phase_space_failed = false;///< ThreeBodyDecayIt's rejection loop hit 10,000
  bool no_channel_chosen = false; ///< the cumulative search fell off the end
  int refused_pdg = 0;
  __host__ __device__ bool any() const {
    return unknown_species || too_many_daughters || too_many_channels || below_threshold ||
           phase_space_failed || no_channel_chosen;
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

// ==============================================================================================
// G4KineticTrack::Decay - the channel draw, the daughter masses and the phase-space kinematics.
// ==============================================================================================

/// `G4SampleResonance::SampleMass(poleMass, gamma, minMass, maxMass)`.
///
/// NOT the same function as `G4VScatteringCollision::SampleResonanceMass` in `resonance_fs.cuh`,
/// though both invert the same Breit-Wigner integral with the same `BrWigInt0` and `BrWigInv`.
/// Three differences, and they are the reason both are carried:
///
///   * an empty window is CLAMPED here (`protectedMinMass = maxMass`) where the other subtracts a
///     pion mass, tries again, and then allows zero;
///   * the zero-width shortcut fires at `gamma < DBL_EPSILON`, 2.2e-16, where the other's fires at
///     `1e-10*GeV` = 1e-7 MeV - twenty-five orders of magnitude larger;
///   * the shortcut uses the UNPROTECTED `minMass`, so it can return a mass above `maxMass` on
///     exactly the input the line above it was written to protect against.
template <typename Rng>
__host__ __device__ inline double g4sample_resonance_mass(double pole_mass, double gamma,
                                                          double min_mass, double max_mass,
                                                          Rng& rng) {
  double protected_min = min_mass;
  if (min_mass > max_mass) { protected_min = max_mass; }
  if (gamma < 2.2204460492503131e-16) {  // DBL_EPSILON
    const double hi = (max_mass < pole_mass) ? max_mass : pole_mass;
    return (min_mass > hi) ? min_mass : hi;
  }
  const double fmin = brwig_int0(protected_min, gamma, pole_mass);
  const double fmax = brwig_int0(max_mass, gamma, pole_mass);
  const double f = fmin + (fmax - fmin) * rng.uniform();
  return brwig_inv(f, gamma, pole_mass);
}

/// `G4GeneralPhaseSpaceDecay::Pmx` - the two-body CM momentum, written as a product of four
/// factors rather than as the usual difference of squares.
///
/// It THROWS when `e - p1 - p2 < 0`, which is why `Decay`'s retry loop exists at all: the loop
/// resamples the daughter masses until their sum is below the parent's, and only then builds the
/// phase-space decay. `pmx_below_threshold` reports what the throw would have been.
__host__ __device__ inline double pmx(double e, double p1, double p2, bool& below_threshold) {
  if (e - p1 - p2 < 0.0) {
    below_threshold = true;
    return 0.0;
  }
  const double ppp = (e + p1 + p2) * (e + p1 - p2) * (e - p1 + p2) * (e - p1 - p2) / (4.0 * e * e);
  return (ppp > 0.0) ? std::sqrt(ppp) : -1.0;
}

/// `G4DynamicParticle(definition, totalEnergy, momentum)` followed by `Get4Momentum()`.
///
/// This is not the identity, and the difference is not always in the last bits. The constructor
/// keeps the DIRECTION and the total energy, decides on a dynamical mass, and stores a kinetic
/// energy; `Get4Momentum` then rebuilds the momentum as `sqrt(T^2 + 2 m T)`. Which mass it uses
/// is a three-way test against `EnergyMomentumRelationAllowance = 1e-2 keV`:
///
///   * `E^2 - p^2 < (1e-5 MeV)^2`   -> the particle is treated as MASSLESS, T = E;
///   * `|m_PDG^2 - (E^2-p^2)| > (1e-5 MeV)^2` -> the dynamical mass is `sqrt(E^2-p^2)`, which is
///     the sampled resonance mass, and the momentum comes back unchanged to rounding;
///   * otherwise the PDG mass is kept and **the momentum is rebuilt from it**, so a daughter
///     whose sampled mass is within 1e-5 MeV of its pole has its momentum quietly moved onto the
///     pole-mass shell.
///
/// Every decay product in the cascade goes through it, so the port does too - and MEASURED: for
/// the 11,316 compared components it is a no-op to within 5e-15, because the energy and momentum
/// handed to it were built consistently from the same mass in the first place. It is kept because
/// it is not a no-op in general: the third branch REBUILDS the momentum from the PDG mass, so a
/// daughter whose sampled mass is within 1e-5 MeV of its pole is quietly moved onto the pole
/// shell, and a release that widened the allowance would move every resonance daughter with it.
__host__ __device__ inline LorentzVector dynamic_particle_4momentum(double pdg_mass,
                                                                    double total_energy,
                                                                    const Vec3d& p) {
  const double allow = 1.0e-2 * u::keV<double>();
  const double allow2 = allow * allow;
  const double p2 = g4gpu::mag2(p);
  double m = pdg_mass;
  double ke = 0.0;
  Vec3d dir{1.0, 0.0, 0.0};
  if (p2 > 0.0) {
    const double mass2 = total_energy * total_energy - p2;
    dir = g4gpu::normalize(p);
    if (mass2 < allow2) {
      m = 0.0;
      ke = total_energy;
    } else if (std::fabs(pdg_mass * pdg_mass - mass2) > allow2) {
      m = std::sqrt(mass2);
      ke = total_energy - m;
    } else {
      ke = total_energy - m;
    }
  }
  const double mom = std::sqrt(ke * ke + 2.0 * m * ke);
  return LorentzVector(dir * mom, ke + m);
}

/// One outgoing track of `G4KineticTrack::Decay`.
struct DecayProduct {
  int pdg = 0;
  LorentzVector p;
};

/// What `Decay()` produced. `n == 0` is Geant4's NULL return - a track with no decay table or a
/// total actual width of zero.
struct DecayResult {
  int n = 0;
  int channel = -1;
  double parent_mass = 0.0;
  int loops = 0;          ///< how many times the below-threshold retry went round
  DecayProduct prod[3];
};

/// `G4GeneralPhaseSpaceDecay::TwoBodyDecayIt`, in the parent's rest frame.
///
/// Two uniforms: `costheta = 2u-1` and `phi = twopi*u`. The second daughter takes the opposite
/// momentum, and both go through the `G4DynamicParticle` round trip above.
template <typename Rng>
__host__ __device__ inline void two_body_decay(double parent_mass, const double* masses,
                                               const int* dpdg, Rng& rng, DecayResult& out,
                                               DecayRefusal& ref) {
  bool below = false;
  const double q = pmx(parent_mass, masses[0], masses[1], below);
  if (below) {
    ref.below_threshold = true;
    return;
  }
  const double costheta = 2.0 * rng.uniform() - 1.0;
  const double sintheta = std::sqrt((1.0 - costheta) * (1.0 + costheta));
  const double phi = u::twopi<double>() * rng.uniform();
  const Vec3d dir{sintheta * std::cos(phi), sintheta * std::sin(phi), costheta};
  double etot = std::sqrt(masses[0] * masses[0] + q * q);
  out.prod[0].pdg = dpdg[0];
  out.prod[0].p = dynamic_particle_4momentum(masses[0], etot, dir * q);
  etot = std::sqrt(masses[1] * masses[1] + q * q);
  out.prod[1].pdg = dpdg[1];
  out.prod[1].p = dynamic_particle_4momentum(masses[1], etot, dir * (-1.0 * q));
  out.n = 2;
}

/// `G4GeneralPhaseSpaceDecay::ThreeBodyDecayIt` - "originally written in GDECA3 of GEANT3".
///
/// The rejection loop draws TWO uniforms per attempt and orders them, so the number of randoms it
/// consumes is data-dependent; the oracle dumps the count for that reason. The products are
/// pushed in the order 0, 2, 1, which is not the order they are computed in and not the order
/// they come back out in either - see `kinetic_track_decay`.
template <typename Rng>
__host__ __device__ inline void three_body_decay(double parent_mass, const double* masses,
                                                 const int* dpdg, Rng& rng, DecayResult& out,
                                                 DecayRefusal& ref) {
  const double sum_mass = masses[0] + masses[1] + masses[2];
  double q[3] = {0.0, 0.0, 0.0};
  double momentum_max = 0.0;
  double momentum_sum = 0.0;
  int loop = 0;
  do {
    double rd1 = rng.uniform();
    double rd2 = rng.uniform();
    if (rd2 > rd1) {
      const double t = rd1;
      rd1 = rd2;
      rd2 = t;
    }
    momentum_max = 0.0;
    momentum_sum = 0.0;
    double energy = rd2 * (parent_mass - sum_mass);
    q[0] = std::sqrt(energy * energy + 2.0 * energy * masses[0]);
    if (q[0] > momentum_max) { momentum_max = q[0]; }
    momentum_sum += q[0];
    energy = (1.0 - rd1) * (parent_mass - sum_mass);
    q[1] = std::sqrt(energy * energy + 2.0 * energy * masses[1]);
    if (q[1] > momentum_max) { momentum_max = q[1]; }
    momentum_sum += q[1];
    energy = (rd1 - rd2) * (parent_mass - sum_mass);
    q[2] = std::sqrt(energy * energy + 2.0 * energy * masses[2]);
    if (q[2] > momentum_max) { momentum_max = q[2]; }
    momentum_sum += q[2];
  } while ((momentum_max > momentum_sum - momentum_max) && ++loop < 10000);
  if (loop >= 10000) {
    ref.phase_space_failed = true;
    return;
  }
  const double costheta = 2.0 * rng.uniform() - 1.0;
  const double sintheta = std::sqrt((1.0 - costheta) * (1.0 + costheta));
  const double phi = u::twopi<double>() * rng.uniform();
  const double sinphi = std::sin(phi);
  const double cosphi = std::cos(phi);
  const Vec3d dir0{sintheta * cosphi, sintheta * sinphi, costheta};
  double etot = std::sqrt(masses[0] * masses[0] + q[0] * q[0]);
  // Push order 0, then 2, then 1.
  out.prod[0].pdg = dpdg[0];
  out.prod[0].p = dynamic_particle_4momentum(masses[0], etot, dir0 * q[0]);

  const double costhetan = (q[1] * q[1] - q[2] * q[2] - q[0] * q[0]) / (2.0 * q[2] * q[0]);
  const double sinthetan = std::sqrt((1.0 - costhetan) * (1.0 + costhetan));
  const double phin = u::twopi<double>() * rng.uniform();
  const double sinphin = std::sin(phin);
  const double cosphin = std::cos(phin);
  Vec3d dir2{
      sinthetan * cosphin * costheta * cosphi - sinthetan * sinphin * sinphi +
          costhetan * sintheta * cosphi,
      sinthetan * cosphin * costheta * sinphi + sinthetan * sinphin * cosphi +
          costhetan * sintheta * sinphi,
      -sinthetan * cosphin * sintheta + costhetan * costheta};
  // Geant4 divides by `direction2`'s magnitude twice - once inside the energy under a SQUARE,
  // `q2^2/direction2.mag2()`, and once outside it, `direction2*(q2/direction2.mag())` - as though
  // the vector were not normalised. It is: the three components above are a rotation applied to
  // (sin(thetan)cos(phin), sin(thetan)sin(phin), cos(thetan)), which is a unit vector, so the
  // magnitude is 1 to within an ulp. MEASURED: dropping the division from the energy changes none
  // of the 11,316 compared components. Both divisions are transcribed, because the rounding they
  // carry is real and because a release that changed the construction of `direction2` would make
  // them matter.
  const double dir2_mag2 = g4gpu::mag2(dir2);
  const double dir2_mag = std::sqrt(dir2_mag2);
  etot = std::sqrt(masses[2] * masses[2] + q[2] * q[2] / dir2_mag2);
  out.prod[1].pdg = dpdg[2];
  out.prod[1].p = dynamic_particle_4momentum(masses[2], etot, dir2 * (q[2] / dir2_mag));

  const Vec3d mom = -1.0 * (dir0 * q[0] + dir2 * (q[2] / dir2_mag));
  etot = std::sqrt(masses[1] * masses[1] + g4gpu::mag2(mom));
  out.prod[2].pdg = dpdg[1];
  out.prod[2].p = dynamic_particle_4momentum(masses[1], etot, mom);
  out.n = 3;
}

/// `G4KineticTrack::Decay`.
///
/// `widths` is `theActualWidth[]` from `kinetic_track_actual_widths` and `n_channels` its length;
/// `parent` is the track's four-momentum in the lab. Returns the products already boosted into
/// the lab, in the order Geant4 hands them back.
///
/// **The product order is the phase-space push order REVERSED.** `G4DecayProducts::PopProducts`
/// pops from the BACK, and `Decay`'s loop pops `dEntries` times - so a two-body decay comes back
/// as (daughter 1, daughter 0) and a three-body one, pushed 0, 2, 1, comes back as (1, 2, 0).
/// Nothing in the physics depends on it and everything in the random stream downstream does,
/// because the cascade pushes these onto its track list in this order.
///
/// The retry loop around the channel draw is A.R.'s of 16 Aug 2016: it redraws the channel AND
/// the daughter masses until the daughters fit inside the parent, up to 10,000 times, and then
/// goes ahead anyway - at which point `Pmx` throws. `DecayRefusal::below_threshold` is that exit.
template <typename Rng>
__host__ __device__ inline DecayResult kinetic_track_decay(int pdg, double actual_mass,
                                                           const LorentzVector& parent,
                                                           const double* widths, int n_channels,
                                                           Rng& rng, DecayRefusal& ref) {
  DecayResult out;
  out.parent_mass = actual_mass;
  const int si = decay_species_index(pdg);
  if (si < 0) {
    ref.unknown_species = true;
    ref.refused_pdg = pdg;
    return out;
  }
  if (n_channels == 0) { return out; }
  const double total = evaluate_total_actual_width(widths, n_channels);
  if (total == 0.0) { return out; }
  const int first = decay_species_first_channel()[si];

  int chosen = -1;
  double masses[4] = {0.0, 0.0, 0.0, 0.0};
  int dpdg[3] = {0, 0, 0};
  int nd = 0;
  bool below_threshold = true;
  int loop = 0;
  do {
    // The cumulative array is built BACKWARDS, so `cum[index]` is the sum from `index` to the
    // end and `cum[0]` is the total. The search then walks from the last channel down and takes
    // the first index whose cumulative exceeds the draw.
    double cum[kDecayMaxChannels];
    double running = 0.0;
    for (int index = n_channels - 1; index >= 0; --index) {
      running += widths[index];
      cum[index] = running;
    }
    const double r = total * rng.uniform();
    chosen = -1;
    for (int index = n_channels - 1; index >= 0; --index) {
      if (r < cum[index]) {
        chosen = index;
        break;
      }
    }
    if (chosen < 0) {
      ref.no_channel_chosen = true;
      return out;
    }
    const int ch = first + chosen;
    nd = decay_channel_n_daughters()[ch];
    for (int i = 0; i < 4; ++i) { masses[i] = 0.0; }
    int shortlived[4];
    int n_short = 0;
    double sum_long = 0.0;
    for (int j = 0; j < nd && j < 3; ++j) {
      dpdg[j] = decay_channel_daughters()[ch * kDecayMaxDaughters + j];
      const int dj = decay_species_index(dpdg[j]);
      if (dj < 0) {
        ref.unknown_species = true;
        ref.refused_pdg = dpdg[j];
        return out;
      }
      masses[j] = decay_species_mass()[dj];
      if (decay_species_shortlived()[dj] != 0) {
        shortlived[n_short] = j;
        ++n_short;
      } else {
        sum_long += decay_species_mass()[dj];
      }
    }
    if (nd == 2 || nd == 3) {
      if (n_short == 1) {
        const int j = shortlived[0];
        const int dj = decay_species_index(dpdg[j]);
        const double massmax = actual_mass - sum_long;
        masses[j] = g4sample_resonance_mass(decay_species_mass()[dj], decay_species_width()[dj],
                                            decay_species_min_mass()[dj], massmax, rng);
      } else if (n_short == 2 && nd == 2) {
        // Dead in 11.1.1 - no channel in the closure has two short-lived daughters, docs/RISK.md
        // V149 - but it draws an extra uniform before either mass, so a release that woke it up
        // would move the whole stream and not just this decay.
        const int zero = (rng.uniform() > 0.5) ? 0 : 1;
        const int one = 1 - zero;
        const int jz = shortlived[zero];
        const int jo = shortlived[one];
        const int dz = decay_species_index(dpdg[jz]);
        const int do_ = decay_species_index(dpdg[jo]);
        double massmax = actual_mass - decay_species_min_mass()[do_];
        masses[jz] = g4sample_resonance_mass(decay_species_mass()[dz], decay_species_width()[dz],
                                             decay_species_min_mass()[dz], massmax, rng);
        massmax = actual_mass - masses[jz];
        masses[jo] = g4sample_resonance_mass(decay_species_mass()[do_],
                                             decay_species_width()[do_],
                                             decay_species_min_mass()[do_], massmax, rng);
      }
    }
    double sum_daughters = 0.0;
    for (int i = 0; i < 4; ++i) { sum_daughters += masses[i]; }
    below_threshold = !(actual_mass - sum_daughters > 0.0);
  } while (below_threshold && ++loop < 10000);
  out.loops = loop;
  out.channel = chosen;

  if (nd == 1) {
    // OneBodyDecayIt: the daughter is created AT REST and draws nothing.
    out.prod[0].pdg = dpdg[0];
    out.prod[0].p = LorentzVector(Vec3d{0.0, 0.0, 0.0}, masses[0]);
    out.n = 1;
  } else if (nd == 2) {
    two_body_decay(actual_mass, masses, dpdg, rng, out, ref);
  } else if (nd == 3) {
    three_body_decay(actual_mass, masses, dpdg, rng, out, ref);
  } else {
    ref.too_many_daughters = true;
    return out;
  }
  if (out.n == 0) { return out; }

  // PopProducts pops from the BACK, so the list comes back reversed.
  DecayProduct rev[3];
  for (int i = 0; i < out.n; ++i) { rev[i] = out.prod[out.n - 1 - i]; }
  // `G4LorentzRotation toMoving(Get4Momentum().boostVector())` - a pure boost out of the rest
  // frame, applied to each product as a 4x4 matrix and not as HepLorentzVector::boost.
  const LorentzRotation to_moving = LorentzRotation::from_boost(parent.boost_vector());
  for (int i = 0; i < out.n; ++i) {
    out.prod[i].p = to_moving * rev[i].p;
    // Each product is then made into a G4KineticTrack, and THAT constructor is where the K0
    // substitution lives - so a Lambda K0 channel draws one more uniform here, after the
    // kinematics, and hands back a K0S or a K0L. docs/RISK.md V150. MEASURED: without this the
    // N(1650)0 rows report product 310 against 311 and the draw count 3 against 4.
    out.prod[i].pdg = kinetic_track_substitute_k0(rev[i].pdg, rng);
  }
  return out;
}

}  // namespace g4gpu::bic::imr

#endif
