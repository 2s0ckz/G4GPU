// P17 - CLHEP::RandGaussQ, the function every `G4RandGauss::shoot` in Geant4 calls.
//
// `Randomize.hh` line 47 is `#define G4RandGauss CLHEP::RandGaussQ`. The port has ONE
// transcription of it, src/core/rand_gauss_q.cuh, shared since docs/RISK.md V185 by the Binary
// cascade's nucleus, both energy-loss fluctuation models, both multiple-scattering models and
// competitive fission. This file is its oracle, in two parts:
//
//   gaussq_transform.csv  `transformQuick(r)` and `transformSmall(r)` - both protected statics,
//                         reached through a derived struct's `using` declarations - over a
//                         STATED grid, every row labelled with why it is there:
//                           * the series tail, r <= 2e-6: a log sweep from 1e-300, and the
//                             smallest uniform of each engine lattice that matters here -
//                             2^-24 (HepJamesRandom), 2^-33 (the port's Philox<double>), and
//                             2^-52 and 2^-61 (the two conversions MixMaxRng's flat() uses as
//                             MSVC compiles it), with 2^-53 beside them;
//                           * EVERY node of both tables, k*Table0step for k = 1..250 and
//                             k*Table1step for k = 1..1000, with the double on either side of
//                             each and the bin's midpoint. `index = int(Table0size*rr)` and
//                             `int((Table1size<<1)*r)` land a node in the bin below or the bin
//                             above by the last bit of a product, and which one is the oracle's
//                             to say;
//                           * r near 0.5, where `index == Table1size` returns exactly zero, and
//                             the mirror `r > 0.5`, including the largest double below 1;
//                           * r = 0 and r = 1, which are NaN in CLHEP - no CLHEP engine returns
//                             either, and a port whose uniform can must know what it gets.
//   gaussq_shoot.csv      `shoot(engine, mean, stdDev)` and `shoot(mean, stdDev)` - the two
//                         overloads the sites use (the EM models pass their engine, fission
//                         does not) - on a recorded HepJamesRandom stream: every `flat()` the
//                         engine serves is written beside the value it became, and the number
//                         of flats each call consumed. The port's test feeds the same uniforms to
//                         its own function and compares the values to the last bit, and the
//                         draw count is Geant4's own: one per value, never a pair.
//
// Fixed seeds, so the file is the same every time it is written.
#include "dump_registry.hh"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "CLHEP/Random/JamesRandom.h"
#include "CLHEP/Random/RandGauss.h"
#include "CLHEP/Random/RandGaussQ.h"
#include "Randomize.hh"

namespace {

/// `transformQuick` and `transformSmall` are protected statics of `CLHEP::RandGaussQ`; a
/// `using` declaration in a derived class makes them public there. Never instantiated.
struct GaussQStatics : public CLHEP::RandGaussQ {
  using CLHEP::RandGaussQ::transformQuick;
  using CLHEP::RandGaussQ::transformSmall;
};

/// The same for `CLHEP::RandGauss`'s cached second value, so the dump can say that the draws
/// below leave it exactly as they found it - whatever an earlier dump left there - without
/// touching it.
struct GaussStatics : public CLHEP::RandGauss {
  using CLHEP::RandGauss::getVal;
};

/// HepJamesRandom with every value it serves recorded, and a counter per call.
class GaussQTapeEngine : public CLHEP::HepRandomEngine {
 public:
  explicit GaussQTapeEngine(long seed) { base_.setSeed(seed, 0); }
  double flat() override {
    const double v = base_.flat();
    tape_.push_back(v);
    return v;
  }
  void flatArray(const int size, double* vect) override {
    for (int i = 0; i < size; ++i) { vect[i] = flat(); }
  }
  void setSeed(long s, int i) override { base_.setSeed(s, i); }
  void setSeeds(const long* s, int i) override { base_.setSeeds(s, i); }
  void saveStatus(const char[]) const override {}
  void restoreStatus(const char[]) override {}
  void showStatus() const override {}
  std::string name() const override { return "GaussQTapeEngine"; }
  const std::vector<double>& tape() const { return tape_; }

 private:
  CLHEP::HepJamesRandom base_;
  std::vector<double> tape_;
};

/// %.17g round-trips every double; NaN is spelled "nan" so a reader need not parse MSVC's
/// "-nan(ind)".
void put(FILE* f, double v) {
  if (std::isnan(v)) {
    std::fprintf(f, "nan");
  } else {
    std::fprintf(f, "%.17g", v);
  }
}

void transform_row(FILE* f, const char* label, double r) {
  std::fprintf(f, "%s,", label);
  put(f, r);
  std::fprintf(f, ",");
  put(f, GaussQStatics::transformQuick(r));
  std::fprintf(f, ",");
  put(f, GaussQStatics::transformSmall(r));
  std::fprintf(f, "\n");
}

void write_transform() {
  FILE* f = std::fopen("gaussq_transform.csv", "w");
  std::fprintf(f, "label,r,quick,small\n");

  // ---- the series tail, r <= Table0step = 2e-6
  for (int i = 0; i <= 120; ++i) {
    // 1e-300 to 2e-6 in 121 log steps, the last one exactly on 2e-6's decade point
    const double lg = -300.0 + (std::log10(2e-6) + 300.0) * i / 120.0;
    transform_row(f, "tail_sweep", std::pow(10.0, lg));
  }
  transform_row(f, "lattice_2^-24_HepJamesRandom", std::ldexp(1.0, -24));
  transform_row(f, "lattice_2^-33_Philox_double", std::ldexp(1.0, -33));
  transform_row(f, "lattice_2^-52_MixMax_convert1double", std::ldexp(1.0, -52));
  transform_row(f, "lattice_2^-53", std::ldexp(1.0, -53));
  transform_row(f, "lattice_2^-61_MixMax_INV_M61", std::ldexp(1.0, -61));
  transform_row(f, "tail_1e-8", 1e-8);
  transform_row(f, "tail_1e-7", 1e-7);
  transform_row(f, "tail_1e-6", 1e-6);
  transform_row(f, "Table0step_below", std::nextafter(2e-6, 0.0));
  transform_row(f, "Table0step", 2e-6);   // `r > Table0step` is false: the series
  transform_row(f, "Table0step_above", std::nextafter(2e-6, 1.0));

  // ---- the fine table, nodes k*Table0step, k = 1..250 (k = 250 is Table1step)
  for (int k = 1; k <= 250; ++k) {
    const double node = k * 2.0e-6;
    transform_row(f, "t0_node_below", std::nextafter(node, 0.0));
    transform_row(f, "t0_node", node);
    transform_row(f, "t0_node_above", std::nextafter(node, 1.0));
    if (k < 250) { transform_row(f, "t0_mid", (k + 0.5) * 2.0e-6); }
  }
  transform_row(f, "Table1step", 5.0e-4);   // `r >= Table1step`: the coarse table

  // ---- the coarse table, nodes k*Table1step, k = 1..1000 (k = 1000 is 0.5)
  for (int k = 1; k <= 1000; ++k) {
    const double node = k * 5.0e-4;
    transform_row(f, "t1_node_below", std::nextafter(node, 0.0));
    transform_row(f, "t1_node", node);
    transform_row(f, "t1_node_above", std::nextafter(node, 1.0));
    if (k < 1000) { transform_row(f, "t1_mid", (k + 0.5) * 5.0e-4); }
  }

  // ---- the median and the mirror
  transform_row(f, "half", 0.5);   // index == Table1size: exactly 0.0
  transform_row(f, "half_below", std::nextafter(0.5, 0.0));
  transform_row(f, "half_above", std::nextafter(0.5, 1.0));
  for (int k = 1; k < 1000; k += 7) {   // 1 - node, and 1 - (node +- one ulp) through the mirror
    const double u = 1.0 - k * 5.0e-4;
    transform_row(f, "mirror_t1", u);
    transform_row(f, "mirror_t1_below", std::nextafter(u, 0.0));
    transform_row(f, "mirror_t1_above", std::nextafter(u, 1.0));
  }
  for (int k = 1; k < 250; k += 3) {
    const double u = 1.0 - k * 2.0e-6;
    transform_row(f, "mirror_t0", u);
    transform_row(f, "mirror_t0_above", std::nextafter(u, 1.0));
  }
  transform_row(f, "mirror_1-2^-24", 1.0 - std::ldexp(1.0, -24));
  transform_row(f, "mirror_1-2^-33", 1.0 - std::ldexp(1.0, -33));
  transform_row(f, "mirror_1-1e-9", 1.0 - 1e-9);
  transform_row(f, "mirror_below_one", std::nextafter(1.0, 0.0));   // 1 - 2^-53

  // ---- what CLHEP gives for an argument no CLHEP engine produces
  transform_row(f, "zero", 0.0);
  transform_row(f, "one", 1.0);
  std::fclose(f);
}

struct ShootCase {
  const char* name;
  double mean, std_dev;
  bool engine_overload;   ///< shoot(engine, mean, sd) if true, shoot(mean, sd) if false
  long seed;
};

void write_shoot() {
  // The (mean, stdDev) pairs are the shapes the sites pass: the unit Gaussian WentzelVI draws
  // twice per sub-step, C12's cluster spread, a thick-absorber energy loss, Urban's tlimit,
  // fission's charge and kinetic energy, and one large mean with a small sigma so the sum's
  // rounding is exercised as well as the product's.
  const ShootCase cases[] = {
      {"unit_engine", 0.0, 1.0, true, 170001L},
      {"unit_static", 0.0, 1.0, false, 170002L},
      {"c12_disp", 0.0, 0.552, false, 170003L},
      {"eloss", 3.6525, 0.2481, true, 170004L},
      {"urban_tlimit", 1.0, 0.099, true, 170005L},
      {"fission_charge", 39.2307692307692, 0.6, false, 170006L},
      {"fission_ke", 171.25, 8.0, false, 170007L},
      {"big_mean", 12345.678, 1.0e-3, true, 170008L},
  };
  const int kPerCase = 4000;
  FILE* f = std::fopen("gaussq_shoot.csv", "w");
  std::fprintf(f, "case,overload,mean,std_dev,i,draws,u,value\n");
  CLHEP::HepRandomEngine* saved = CLHEP::HepRandom::getTheEngine();
  int state_changed = 0;
  for (const ShootCase& c : cases) {
    auto* eng = new GaussQTapeEngine(c.seed);
    CLHEP::HepRandom::setTheEngine(eng);
    const bool flag_before = CLHEP::RandGauss::getFlag();
    const double val_before = GaussStatics::getVal();
    for (int i = 0; i < kPerCase; ++i) {
      const size_t before = eng->tape().size();
      const double v = c.engine_overload ? G4RandGauss::shoot(eng, c.mean, c.std_dev)
                                         : G4RandGauss::shoot(c.mean, c.std_dev);
      const size_t draws = eng->tape().size() - before;
      // One uniform per value is the claim; if CLHEP ever drew more, the row carries the
      // first and the count says so.
      const double u = (draws > 0) ? eng->tape()[before] : -1.0;
      std::fprintf(f, "%s,%s,%.17g,%.17g,%d,%d,", c.name,
                   c.engine_overload ? "engine" : "static", c.mean, c.std_dev, i, int(draws));
      put(f, u);
      std::fprintf(f, ",");
      put(f, v);
      std::fprintf(f, "\n");
    }
    // `RandGauss`'s pair cache is the state `RandGaussQ` does not have, and 4,000 draws must
    // leave it bit for bit as they found it. Compared before and after rather than tested for
    // "unset", because another dump may legitimately have left a value there first (docs/RISK.md
    // V180 measured it unset after the cascade's nuclei).
    const double val_after = GaussStatics::getVal();
    if (CLHEP::RandGauss::getFlag() != flag_before ||
        std::memcmp(&val_before, &val_after, sizeof val_after) != 0) {
      ++state_changed;
    }
    CLHEP::HepRandom::setTheEngine(saved);
    delete eng;
  }
  std::fprintf(f, "randgauss_state_changed_by_cases,,,,,%d,,\n", state_changed);
  std::fclose(f);
}

void dump_gaussq(const DumpContext&) {
  write_transform();
  write_shoot();
}

}  // namespace

G4GPU_REGISTER_DUMP("gaussq", "gaussq_transform.csv gaussq_shoot.csv", dump_gaussq);
