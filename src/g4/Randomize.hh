// Randomize.hh: the host random number generator, and what it does and does not reach.
//
// Geant4 examples include this for G4UniformRand() in a PrimaryGeneratorAction, and some set a
// different engine or a seed in main(). Both work, and both matter.
//
// **This engine generates the primaries.** GeneratePrimaries runs on the host, once per event,
// and every random number it draws - a beam spot, a Gaussian, a source chosen by weight - comes
// from here. Reseeding it changes which primaries a run fires, and therefore the run's result.
// A run is not reproducible from G4RunManager::SetRandomSeed alone.
//
// **It does not generate the shower.** Once a primary is on the device, every random number the
// transport consumes is drawn by a counter-based Philox generator keyed on the track's own
// (rng_key, step) - see src/core/rng.cuh - which is what makes the transport of a given set of
// primaries bit-reproducible regardless of batch size or thread count. That key is seeded from
// G4RunManager::SetRandomSeed.
//
// So a fully reproducible run needs both: this engine seeded for the primaries, and the run's
// seed for the showers. That is what Geant4 users expect from /random/setSeeds, and it is why
// two successive /run/beamOn calls in one session give different answers - the host stream
// carries on where the last run left it, exactly as it does in Geant4. The viewer's "accumulate
// across runs" depends on that: when the primaries came from a device stream keyed only on the
// run's seed, two runs of the same size were *identical*, and accumulating them multiplied one
// run's result instead of reducing its variance.
#pragma once
#include <cstdio>
#include <random>
#include "g4/G4Types.hh"

namespace CLHEP {

class HepRandomEngine {
 public:
  virtual ~HepRandomEngine() = default;
  virtual G4double flat() = 0;
  virtual const char* name() const = 0;
};

/// The default: a 64-bit Mersenne twister. Host side only.
class MixMaxRng : public HepRandomEngine {
 public:
  G4double flat() override { return dist_(gen_); }
  const char* name() const override { return "MixMaxRng"; }
  void setSeed(long s, int = 0) { gen_.seed(static_cast<unsigned long long>(s)); }

 private:
  std::mt19937_64 gen_{5489ULL};
  std::uniform_real_distribution<G4double> dist_{0.0, 1.0};
};

class MTwistEngine : public MixMaxRng {
 public:
  const char* name() const override { return "MTwistEngine"; }
};
class RanecuEngine : public MixMaxRng {
 public:
  const char* name() const override { return "RanecuEngine"; }
};

class HepRandom {
 public:
  static HepRandomEngine* getTheEngine() { return &Default(); }

  static void setTheEngine(HepRandomEngine* e) {
    Current() = (e != nullptr) ? e : &Default();
    std::printf(
        "note: G4Random::setTheEngine(%s) applies to host-side code only. The transport\n"
        "      draws from a device Philox stream keyed on each track; see g4/Randomize.hh.\n",
        Current()->name());
  }

  static void setTheSeed(long s, int lux = 0) {
    Default().setSeed(s, lux);
  }

  static G4double flat() { return Current()->flat(); }

 private:
  static MixMaxRng& Default() {
    static MixMaxRng e;
    return e;
  }
  static HepRandomEngine*& Current() {
    static HepRandomEngine* c = &Default();
    return c;
  }
};

}  // namespace CLHEP

using G4Random = CLHEP::HepRandom;

inline G4double G4UniformRand() { return CLHEP::HepRandom::flat(); }
