// Energy-loss fluctuations: G4UniversalFluctuation.
//
// The stopping power is a mean. A real particle crossing a real step loses a *sample* from a
// distribution whose mean is that number, and the width of that distribution is what gives a
// Bragg peak its distal edge. Without this a proton beam's protons all stop at the same depth
// to within the step size: the first comparison of this port's depth-dose curve against
// Geant4's had the peak in the right place, the plateau right to a per cent, and a falloff
// three bins wide against Geant4's twelve.
//
// Transcribed from G4UniversalFluctuation::SampleFluctuations and ::SampleGlandz (11.1.1),
// which is the Urban model - a two-component picture where the loss is a sum over excitations
// at fixed energies plus an ionisation term sampled from a 1/E^2 spectrum between e0 and the
// production cut. G4EmParameters' default FluctuationType is fUniversalFluctuation, so this is
// what a stock physics list runs; G4UrbanFluctuation is the opt-in variant and is not here.
//
// Two things this file needs that the transport did not already have: a Poisson deviate and a
// Gamma deviate. Both are below, and both are transcriptions rather than inventions - G4Poisson
// exactly, G4RandGamma by algorithm rather than by line, since its result only has to have the
// right distribution and its internal rejection sequence is not observable.
#pragma once
#include <cmath>

/// __noinline__ where the device compiler understands it, nothing on the host.
#if defined(__CUDACC__)
#define G4GPU_NOINLINE __noinline__
#else
#define G4GPU_NOINLINE
#endif

#include "core/particle.cuh"
#include "core/units.cuh"
#include "data/materials.cuh"
#include "physics/em/electron_processes.cuh"  // twopi_mc2_rcl2

namespace g4gpu::em {

/// Verbatim from G4Poisson (G4Poisson.hh): a direct inversion below a mean of 16 and a
/// Gaussian approximation above it. The cutover and the +0.5 rounding are Geant4's.
template <typename real_t, typename Rng>
__host__ __device__ inline int g4_poisson(real_t mean, Rng& rng) {
  constexpr real_t kBorder = real_t(16);
  constexpr real_t kLimit = real_t(2e9);
  if (mean <= kBorder) {
    const real_t position = rng.uniform();
    real_t value = exp(-mean);
    real_t sum = value;
    int n = 0;
    // Bounded where Geant4's while is not. The inversion terminates with probability one, but
    // a device kernel that can spin forever on a denormal is a hang with no diagnostic; at a
    // mean of 16 the chance of needing more than 200 terms is far below any floating-point
    // resolution.
    while (sum <= position && n < 200) {
      ++n;
      value *= mean / real_t(n);
      sum += value;
    }
    return n;
  }
  real_t t = sqrt(real_t(-2) * log(rng.uniform()));
  const real_t y = units::twopi<real_t>() * rng.uniform();
  t *= cos(y);
  const real_t value = mean + t * sqrt(mean) + real_t(0.5);
  if (value <= real_t(0)) { return 0; }
  return static_cast<int>(fmin(value, kLimit));
}

/// Gaussian deviate, Box-Muller. Stands in for G4RandGauss::shoot, which is the same
/// distribution by a different route (CLHEP caches the second deviate; nothing here can
/// observe that, and caching across a device thread's step would be worse than not).
template <typename real_t, typename Rng>
__host__ __device__ inline real_t g4_gauss(real_t mean, real_t sigma, Rng& rng) {
  const real_t u1 = rng.uniform(), u2 = rng.uniform();
  return mean + sigma * sqrt(real_t(-2) * log(u1)) * cos(units::twopi<real_t>() * u2);
}

/// Gamma deviate with shape @p a and unit scale, standing in for G4RandGamma::shoot(a, 1).
///
/// Marsaglia and Tsang (2000) for a >= 1, with the standard a < 1 boost. Chosen over
/// transcribing CLHEP's own rejection because only the distribution is observable and this one
/// has a bounded expected iteration count - CLHEP's RandGamma uses a loop whose worst case is
/// not obviously bounded, which on a device is the difference between slow and hung.
template <typename real_t, typename Rng>
__host__ __device__ inline real_t g4_gamma(real_t a, Rng& rng) {
  if (!(a > real_t(0))) { return real_t(0); }
  real_t boost = real_t(1);
  if (a < real_t(1)) {
    boost = pow(rng.uniform(), real_t(1) / a);
    a += real_t(1);
  }
  const real_t d = a - real_t(1) / real_t(3);
  const real_t c = real_t(1) / sqrt(real_t(9) * d);
  for (int i = 0; i < 100; ++i) {
    real_t x, v;
    do {
      x = g4_gauss(real_t(0), real_t(1), rng);
      v = real_t(1) + c * x;
    } while (v <= real_t(0));
    v = v * v * v;
    const real_t u = rng.uniform();
    if (u < real_t(1) - real_t(0.0331) * x * x * x * x) { return boost * d * v; }
    if (log(u) < real_t(0.5) * x * x + d * (real_t(1) - v + log(v))) { return boost * d * v; }
  }
  return boost * d;  // the mean, if the rejection somehow never accepts
}

// ---------------------------------------------------------------- the model's constants
//
// G4UniversalFluctuation's members, none of which a stock physics list changes.
template <typename real_t> __host__ __device__ constexpr real_t kFlucMinLoss() {
  return real_t(1e-5);  // 10 eV
}
template <typename real_t> __host__ __device__ constexpr real_t kFlucE0() {
  return real_t(1e-5);  // G4IonisParamMat::GetEnergy0fluct(), 10 eV for every material
}
template <typename real_t> __host__ __device__ constexpr real_t kFlucMinNBohr() {
  return real_t(10);  // minNumberInteractionsBohr
}
template <typename real_t> __host__ __device__ constexpr real_t kFlucNmaxCont() {
  return real_t(8);
}
template <typename real_t> __host__ __device__ constexpr real_t kFlucRate() {
  return real_t(0.56);
}
template <typename real_t> __host__ __device__ constexpr real_t kFlucFw() { return real_t(4); }
template <typename real_t> __host__ __device__ constexpr real_t kFlucA0() { return real_t(42); }

/// G4UniversalFluctuation::AddExcitation.
template <typename real_t, typename Rng>
__host__ __device__ inline void fluc_add_excitation(real_t ax, real_t ex, real_t& eav,
                                                    real_t& eloss, real_t& esig2, Rng& rng) {
  if (ax > kFlucNmaxCont<real_t>()) {
    eav += ax * ex;
    esig2 += ax * ex * ex;
  } else {
    const int p = g4_poisson(ax, rng);
    if (p > 0) { eloss += (real_t(p + 1) - real_t(2) * rng.uniform()) * ex; }
  }
}

/// G4UniversalFluctuation::SampleGauss. Note the small-mean branch, which replaces the
/// Gaussian by a uniform: a truncated Gaussian whose mean is under a quarter of its width is
/// not a Gaussian, and sampling one by rejection would spin.
template <typename real_t, typename Rng>
__host__ __device__ inline void fluc_sample_gauss(real_t eav, real_t esig2, real_t& eloss,
                                                  Rng& rng) {
  const real_t sig = sqrt(esig2);
  real_t x = eav;
  if (eav < real_t(0.25) * sig) {
    x += (real_t(2) * rng.uniform() - real_t(1)) * eav;
  } else {
    bool ok = false;
    for (int i = 0; i < 100 && !ok; ++i) {
      x = g4_gauss(eav, sig, rng);
      ok = (x >= real_t(0) && x <= real_t(2) * eav);
    }
    if (!ok) { x = eav; }
  }
  eloss += x;
}

/// G4UniversalFluctuation::SampleGlandz.
///
/// @param mean_loss  already divided by `scaling` by the caller, as the model does.
template <typename real_t, typename Rng>
__host__ __device__ inline real_t fluc_sample_glandz(real_t mean_loss, real_t tcut,
                                                     real_t ipot, Rng& rng) {
  const real_t e0 = kFlucE0<real_t>();
  const real_t rate = kFlucRate<real_t>();
  const real_t fw = kFlucFw<real_t>();
  const real_t a0 = kFlucA0<real_t>();

  real_t a1 = real_t(0), a3 = real_t(0);
  real_t loss = real_t(0);
  real_t e1 = ipot;

  if (tcut > e1) {
    a1 = mean_loss * (real_t(1) - rate) / e1;
    if (a1 < a0) {
      const real_t fwnow = real_t(0.1) + (fw - real_t(0.1)) * sqrt(a1 / a0);
      a1 /= fwnow;
      e1 *= fwnow;
    } else {
      a1 /= fw;
      e1 *= fw;
    }
  }

  const real_t w1 = tcut / e0;
  a3 = rate * mean_loss * (tcut - e0) / (e0 * tcut * log(w1));
  if (a1 <= real_t(0)) { a3 /= rate; }

  real_t emean = real_t(0), sig2e = real_t(0);

  // Excitation.
  if (a1 > real_t(0)) { fluc_add_excitation(a1, e1, emean, loss, sig2e, rng); }
  if (sig2e > real_t(0)) { fluc_sample_gauss(emean, sig2e, loss, rng); }

  // Ionisation: a Poisson number of transfers drawn from 1/E^2 between w3 and tcut, with the
  // bulk of them replaced by a Gaussian once there are more than nmaxCont.
  if (a3 > real_t(0)) {
    emean = real_t(0);
    sig2e = real_t(0);
    real_t p3 = a3;
    real_t alfa = real_t(1);
    if (a3 > kFlucNmaxCont<real_t>()) {
      const real_t nmax = kFlucNmaxCont<real_t>();
      alfa = w1 * (nmax + a3) / (w1 * nmax + a3);
      const real_t alfa1 = alfa * log(alfa) / (alfa - real_t(1));
      const real_t namean = a3 * w1 * (alfa - real_t(1)) / ((w1 - real_t(1)) * alfa);
      emean += namean * e0 * alfa1;
      sig2e += e0 * e0 * namean * (alfa - alfa1 * alfa1);
      p3 = a3 - namean;
    }

    const real_t w3 = alfa * e0;
    if (tcut > w3) {
      const real_t w = (tcut - w3) / tcut;
      // Geant4 draws the whole Poisson count; here it is capped. The cap is not a physics
      // choice - it bounds a device thread's work, and p3 is at most nmaxCont plus the
      // Poisson tail, so 512 is far beyond anything a step produces.
      int nnb = g4_poisson(p3, rng);
      if (nnb > 512) { nnb = 512; }
      for (int k = 0; k < nnb; ++k) { loss += w3 / (real_t(1) - w * rng.uniform()); }
    }
    if (sig2e > real_t(0)) { fluc_sample_gauss(emean, sig2e, loss, rng); }
  }
  return loss;
}

/// The sampled loss for one step, given the mean the stopping power predicted.
///
/// Verbatim from G4UniversalFluctuation::SampleFluctuations, including the shortcut for a loss
/// under 10 eV and the two regimes: a Gaussian (or Gamma) for a heavy particle whose mean loss
/// is large compared with the production cut, and the Glandz sampling otherwise.
///
/// @param tcut  min(production cut, tmax), as G4VEnergyLossProcess::AlongStepDoIt passes it
/// @param tmax  the model's own MaxSecondaryKinEnergy at this energy
/// `__noinline__` on the device, and it is a workaround for a compiler crash rather than a
/// performance choice. Inlined into step_lepton - which already carries Urban MSC, the
/// Seltzer-Berger sampler and the recursive solid engine - this function's two rejection loops
/// and three nested samplers took ptxas past whatever internal limit it has, and CUDA 11.6's
/// ptxas died with an access violation rather than diagnosing anything. Not inlining it is the
/// smallest change that compiles; it costs a call per step, against a kernel whose limit is
/// already register pressure.
template <typename real_t, typename Rng>
__host__ __device__ G4GPU_NOINLINE inline real_t sample_fluctuation(
    const data::Material<real_t>& m, const ParticleDef<real_t>& pd, real_t kinetic, real_t tcut,
    real_t tmax, real_t length, real_t mean_loss, real_t charge_square, Rng& rng) {
  if (mean_loss < kFlucMinLoss<real_t>()) { return mean_loss; }
  if (length <= real_t(0) || pd.mass <= real_t(0)) { return mean_loss; }

  const real_t etot = kinetic + pd.mass;
  const real_t beta2 = kinetic * (kinetic + real_t(2) * pd.mass) / (etot * etot);
  if (!(beta2 > real_t(0))) { return mean_loss; }

  // The Gaussian regime, for a heavy particle over a step long enough that many collisions
  // happened and short enough that none of them could have been a delta ray.
  if (pd.mass > units::electron_mass_c2<real_t>()
      && mean_loss >= kFlucMinNBohr<real_t>() * tcut && tmax <= real_t(2) * tcut) {
    const real_t siga = sqrt((tmax / beta2 - real_t(0.5) * tcut) * twopi_mc2_rcl2<real_t>()
                             * length * charge_square * m.electron_density);
    if (!(siga > real_t(0))) { return mean_loss; }
    const real_t sn = mean_loss / siga;
    if (sn >= real_t(2)) {
      // Truncated Gaussian on (0, 2*mean).
      const real_t two_mean = mean_loss + mean_loss;
      for (int i = 0; i < 100; ++i) {
        const real_t loss = g4_gauss(mean_loss, siga, rng);
        if (loss >= real_t(0) && loss <= two_mean) { return loss; }
      }
      return mean_loss;
    }
    // Too few effective collisions for a Gaussian: a Gamma with the same relative width.
    const real_t neff = sn * sn;
    return mean_loss * g4_gamma(neff, rng) / neff;
  }

  const real_t e0 = kFlucE0<real_t>();
  if (tcut <= e0) { return mean_loss; }

  // The small-cut width correction, and it is applied to the *mean* before sampling and undone
  // on the result - the model widens the distribution by narrowing what it samples around.
  const real_t scaling = fmin(real_t(1) + real_t(0.5) * real_t(1e-3) / tcut, real_t(1.5));
  return fluc_sample_glandz(mean_loss / scaling, tcut, m.mean_excitation, rng) * scaling;
}

}  // namespace g4gpu::em
