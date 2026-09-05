// Primary particle sources, sampled on the device.
//
// Geant4 calls GeneratePrimaries(G4Event*) once per event, on the CPU. At 2.7 million events
// per second that callback is the whole budget several times over, so it cannot be what
// actually produces primaries here. Instead a source is *described* - shape, direction
// distribution, energy distribution, particle - and every event samples that description
// inside the seeding kernel.
//
// The consequence for a user is one real difference from Geant4, and it is worth stating
// plainly: a PrimaryGeneratorAction that randomises something itself, in C++, per event, gets
// called once rather than a million times. What replaces it is a distribution on the gun -
// `SetBeamCrossSectionRectangular` instead of two calls to G4UniformRand. Every distribution
// Geant4's own G4GeneralParticleSource offers is available that way, and the sampling is
// bit-reproducible because it is keyed on the event index rather than on a stream position.
#pragma once
#include <cmath>
#include "core/particle.cuh"
#include "core/rng.cuh"
#include "core/units.cuh"
#include "core/vec3.cuh"

namespace g4gpu {

/// Where primaries start.
enum class SourceShape : int {
  kPoint = 0,           ///< all primaries from one position
  kBeamRectangle = 1,   ///< uniform over a rectangle normal to `dir`, half-widths hx, hy
  kBeamEllipse = 2,     ///< uniform over an ellipse normal to `dir`, semi-axes hx, hy
  kIsotropicShell = 3,  ///< uniform over a sphere of radius `radius`, centred on `pos`
  kVolumeBox = 4,       ///< uniform inside a box of half-widths hx, hy, hz about `pos`
};

/// How the initial direction is distributed.
enum class AngularShape : int {
  kUnidirectional = 0,  ///< exactly `dir`
  kCone = 1,            ///< within `cone_half_angle` of `dir`, uniform in solid angle
  kIsotropic = 2,       ///< 4pi
  kInward = 3,          ///< toward the centre, for a shell source; cosine-weighted
};

/// How the initial energy is distributed.
enum class EnergyShape : int {
  kMonoenergetic = 0,
  kSpectrum = 1,  ///< sampled from a tabulated cumulative distribution
};

/// A tabulated energy spectrum, as a cumulative distribution. Held separately from Source so
/// that a Source stays trivially copyable into a kernel argument.
template <typename real_t>
struct EnergySpectrum {
  const real_t* energy;  ///< MeV, ascending
  const real_t* cdf;     ///< 0 at the first point, 1 at the last
  int n;
};

/// One primary, as the device receives it.
///
/// Generated on the host - G4VUserPrimaryGeneratorAction::GeneratePrimaries runs once per
/// event and builds vertices - then uploaded a batch at a time and read by
/// `seed_from_primaries`. The device no longer samples a Source record of its own, because
/// sampling on the device made the set of possible primaries closed: whatever distributions
/// the gun happened to offer, and nothing else. A Gaussian beam spot, a phase-space file, a
/// different species on alternate events - none of them could be written.
///
/// Sixteen bytes of position, sixteen of direction, one energy and one species: 56 bytes at
/// double precision. Two million events is 112 MB uploaded over a run, in batch-sized pieces,
/// against about a second of transport. That is the price of the generality and it is worth it.
template <typename real_t>
struct Primary {
  Vec3<real_t> pos;
  Vec3<real_t> dir;
  real_t ekin;
  ParticleType particle;
};

template <typename real_t>
struct Source {
  SourceShape shape = SourceShape::kPoint;
  Vec3<real_t> pos{0, 0, 0};
  real_t hx = 0, hy = 0, hz = 0;
  real_t radius = 0;

  Vec3<real_t> dir{0, 0, 1};
  AngularShape angular = AngularShape::kUnidirectional;
  real_t cone_half_angle = 0;

  EnergyShape energy_shape = EnergyShape::kMonoenergetic;
  real_t energy = 1;
  EnergySpectrum<real_t> spectrum{nullptr, nullptr, 0};

  ParticleType particle = ParticleType::kGamma;

  /// The counter-based RNG key for the run. Every track's stream is Philox(seed, event) and
  /// its step counter, so this is the only thing that changes a run's random numbers - the
  /// host engine in g4/Randomize.hh does not. G4RunManager::SetRandomSeed sets it.
  unsigned int seed = 0xF00Du;
};

/// Two unit vectors orthogonal to @p n and to each other.
template <typename real_t>
__host__ __device__ inline void basis_from(const Vec3<real_t>& n, Vec3<real_t>& u,
                                           Vec3<real_t>& v) {
  // Pick the axis least aligned with n, so the cross product never degenerates.
  const Vec3<real_t> a = (fabs(n.z) < real_t(0.9)) ? Vec3<real_t>{0, 0, real_t(1)}
                                                   : Vec3<real_t>{real_t(1), 0, 0};
  u = normalize(cross(a, n));
  v = cross(n, u);
}

/// Inverts a tabulated cumulative distribution by binary search plus linear interpolation.
template <typename real_t>
__host__ __device__ inline real_t sample_spectrum(const EnergySpectrum<real_t>& s, real_t r) {
  if (s.n <= 1 || s.cdf == nullptr) { return real_t(0); }
  int lo = 0, hi = s.n - 1;
  while (hi - lo > 1) {
    const int mid = (lo + hi) / 2;
    if (s.cdf[mid] <= r) { lo = mid; } else { hi = mid; }
  }
  const real_t c0 = s.cdf[lo], c1 = s.cdf[hi];
  const real_t f = (c1 > c0) ? (r - c0) / (c1 - c0) : real_t(0);
  return s.energy[lo] + f * (s.energy[hi] - s.energy[lo]);
}

/// Samples one primary. @p rng must be seeded from the event index, so that the same event
/// always yields the same primary regardless of batch size or thread count.
template <typename real_t, typename Rng>
__host__ __device__ inline void sample_primary(const Source<real_t>& src, Rng& rng,
                                               Vec3<real_t>& pos, Vec3<real_t>& dir,
                                               real_t& ekin) {
  const real_t twopi = units::twopi<real_t>();
  Vec3<real_t> u, v;
  basis_from(src.dir, u, v);

  switch (src.shape) {
    case SourceShape::kPoint:
      pos = src.pos;
      break;
    case SourceShape::kBeamRectangle: {
      const real_t a = src.hx * (real_t(2) * rng.uniform() - real_t(1));
      const real_t b = src.hy * (real_t(2) * rng.uniform() - real_t(1));
      pos = src.pos + a * u + b * v;
      break;
    }
    case SourceShape::kBeamEllipse: {
      // sqrt of a uniform gives a uniform areal density; sampling the radius directly would
      // pile primaries up on the axis.
      const real_t r = sqrt(rng.uniform());
      const real_t ph = twopi * rng.uniform();
      pos = src.pos + (src.hx * r * cos(ph)) * u + (src.hy * r * sin(ph)) * v;
      break;
    }
    case SourceShape::kIsotropicShell: {
      const real_t ct = real_t(2) * rng.uniform() - real_t(1);
      const real_t st = sqrt(fmax(real_t(0), real_t(1) - ct * ct));
      const real_t ph = twopi * rng.uniform();
      pos = src.pos
            + src.radius * Vec3<real_t>{st * cos(ph), st * sin(ph), ct};
      break;
    }
    case SourceShape::kVolumeBox: {
      pos = src.pos
            + Vec3<real_t>{src.hx * (real_t(2) * rng.uniform() - real_t(1)),
                           src.hy * (real_t(2) * rng.uniform() - real_t(1)),
                           src.hz * (real_t(2) * rng.uniform() - real_t(1))};
      break;
    }
  }

  switch (src.angular) {
    case AngularShape::kUnidirectional:
      dir = src.dir;
      break;
    case AngularShape::kCone: {
      // Uniform in solid angle within the cone: cos(theta) uniform in [cos(a), 1].
      const real_t ca = cos(src.cone_half_angle);
      const real_t ct = ca + (real_t(1) - ca) * rng.uniform();
      const real_t st = sqrt(fmax(real_t(0), real_t(1) - ct * ct));
      const real_t ph = twopi * rng.uniform();
      dir = normalize(ct * src.dir + (st * cos(ph)) * u + (st * sin(ph)) * v);
      break;
    }
    case AngularShape::kIsotropic: {
      const real_t ct = real_t(2) * rng.uniform() - real_t(1);
      const real_t st = sqrt(fmax(real_t(0), real_t(1) - ct * ct));
      const real_t ph = twopi * rng.uniform();
      dir = Vec3<real_t>{st * cos(ph), st * sin(ph), ct};
      break;
    }
    case AngularShape::kInward: {
      // A shell that irradiates its interior uniformly needs a cosine-weighted inward
      // distribution, not an isotropic one: isotropic emission from a surface over-weights
      // grazing directions and leaves the centre cold.
      Vec3<real_t> n = pos - src.pos;
      const real_t len = sqrt(dot(n, n));
      n = (len > real_t(0)) ? (real_t(1) / len) * n : Vec3<real_t>{0, 0, real_t(1)};
      Vec3<real_t> su, sv;
      basis_from(n, su, sv);
      const real_t r = sqrt(rng.uniform());
      const real_t ph = twopi * rng.uniform();
      const real_t ct = sqrt(fmax(real_t(0), real_t(1) - r * r));
      dir = normalize(-ct * n + (r * cos(ph)) * su + (r * sin(ph)) * sv);
      break;
    }
  }

  ekin = (src.energy_shape == EnergyShape::kMonoenergetic)
             ? src.energy
             : sample_spectrum(src.spectrum, rng.uniform());
}

}  // namespace g4gpu
