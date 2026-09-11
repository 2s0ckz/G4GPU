// The transport of a nucleus: the base-particle scaling `step_hadron` reads, end to end.
//
// WHY THIS FILE EXISTS
//
// `em::HadronRangeTable` has two sets of entry points and only one of them is the transport's.
//
//     dedx_at(species, mat, E)          the table in its OWN terms - a species that owns a table
//     lookup(species, mat, E)           reads this, and for it the two sets are identical
//     energy_from_range(species, mat, R)
//
//     dedx_for(mat, type, imat, E)      G4VEnergyLossProcess's scaling on top of it:
//     range_for(mat, type, imat, E)         scaledE = E * massRatio
//     energy_from_range_for(...)            dE/dx   = chargeSqRatio * table(scaledE)
//                                           range   = table(scaledE) / (chargeSqRatio*massRatio)
//
// `tests/test_hadron_range.cu` compares the second set against `G4EmCalculator::GetDEDX` and
// `::GetRange` for every species, so the SCALING is measured against Geant4. `step_hadron`
// called the first set. For the ten species that own a table both ratios are exactly one and the
// two are the same function, which is why it went unnoticed for as long as the port transported
// only those ten; for the deuteron and the triton, which P1 gave kernels, massRatio is 0.500246
// and 0.333866 and the transport was reading a PROTON's range and stopping power at the
// deuteron's own kinetic energy.
//
// So this test asserts the thing the two sets differ by, in both directions:
//
//  1. THE SCALING IS THE IDENTITY FOR THE TEN, AND IS NOT FOR THE REST. Arithmetic, per
//     species: massRatio and chargeSqRatio are exactly 1.0 for p, pbar, pi+-, K+-, mu+-, alpha
//     and GenericIon, and they are not for the deuteron, the triton and He3. Without this the
//     rest of the file could pass with the bug in place for eight of its eleven rows.
//  2. THE TRANSPORT AGREES WITH THE TABLE IT READS. A track is stepped to a stop and its total
//     TRUE PATH LENGTH is compared with `range_for` at its starting energy - which is what a
//     range IS - and its total deposit plus what left as delta rays with its starting energy.
//     A transport reading the wrong row of the table fails the first by the factor the row is
//     wrong by, and passes the second, because energy is conserved either way.
//
// Section 3 onwards is the ion itself; see the header above it.
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "core/rng.cuh"
#include "data/materials.cuh"
#include "physics/em/hadron_range.cuh"
#include "physics/scene.cuh"
#include "physics/stepper.cuh"

using real_t = double;
using namespace g4gpu;

namespace {

int g_fails = 0;
void fail(const char* fmt, ...) {
  std::printf("  FAIL: ");
  va_list ap;
  va_start(ap, fmt);
  std::vprintf(fmt, ap);
  va_end(ap);
  std::printf("\n");
  ++g_fails;
}

/// An emitter that only counts what it was given, so a stopping track's energy balance can be
/// closed without a track pool. The three fields every stepper assigns before pushing are here
/// because they are part of the interface `BufferEmitter` offers - see tests/test_neutron.cu.
struct CountingEmitter {
  double secondary_energy = 0;
  int n = 0;
  Vec3<real_t> pos{};
  int volume = 0;
  int event = 0;
  unsigned int child_count = 0u;
  int last_secondary = -1;

  __host__ __device__ int push(ParticleType, const Vec3<real_t>& , real_t ekin, int) {
    ++child_count;
    ++n;
    secondary_energy += static_cast<double>(ekin);
    return 0;
  }
};

/// What following one track to a stop produced.
struct Stopped {
  double path = 0;        ///< mm, the sum of every step's TRUE path length
  double edep = 0;        ///< MeV deposited in the scoring volume
  double secondary = 0;   ///< MeV that left as secondaries
  int steps = 0;
  bool escaped = false;
};

Stopped follow(const Scene<real_t>& s, ParticleType type, real_t ekin,
               const had::HadronicWiring<real_t>& had, unsigned int key) {
  Stopped out;
  TrackState<real_t> p{};
  p.species = type;
  p.pos = Vec3<real_t>{0, 0, 0};
  p.dir = Vec3<real_t>{0, 0, 1};
  p.ekin = ekin;
  p.volume = 0;
  p.event = 0;
  p.rng_key = key;
  p.step = 0u;
  p.begin(p.pos, p.dir, p.ekin, 0, 0u, ProcessId::fNotDefined, real_t(0), real_t(1));
  CountingEmitter em;
  bool alive = true;
  while (alive && out.steps < kMaxStepsPerTrack) {
    StepReport<real_t> rep;
    Philox<real_t> rng(p.rng_key, p.step, 0xB19Du);
    em.pos = p.pos;
    em.volume = p.volume;
    em.event = p.event;
    real_t edep = 0;
    alive = step_hadron(s, p, type, had, rng, em, edep, rep);
    ++p.step;
    ++out.steps;
    out.path += static_cast<double>(rep.true_length);
    out.edep += static_cast<double>(edep);
    if (p.volume == geom::kOutsideWorld) {
      out.escaped = true;
      break;
    }
  }
  out.secondary = em.secondary_energy;
  return out;
}

}  // namespace

int main() {
  std::printf("== the transport of a nucleus: the scaling step_hadron reads ==\n\n");

  data::Material<real_t> mats[data::kNumMaterials];
  data::build_b1_materials<real_t>(mats);

  auto* hrt = new em::HadronRangeTable<real_t>();
  {
    std::vector<real_t> cuts(data::kNumMaterials);
    for (int i = 0; i < data::kNumMaterials; ++i) { cuts[i] = mats[i].cut_electron; }
    static em::ShellTables<real_t> shell;
    em::build_shell_tables(shell);
    em::build_hadron_range_table<real_t>(mats, cuts.data(), *hrt, &shell, data::kNumMaterials);
  }

  // One water cube as the world: what is under test is the step, and a nested geometry would
  // put `step_to_boundary` between the input and the assertion. 200 mm of half-width is more
  // than any range below.
  const real_t kHalf = 200;
  geom::Volume<real_t> vols[1] = {
      {{geom::SolidType::kBox, {kHalf, kHalf, kHalf}},
       geom::make_translation<real_t>({0, 0, 0}), 0, data::kWater, /*score_index=*/0},
  };
  Scene<real_t> scene{};
  scene.geometry = geom::Geometry<real_t>{vols, 1, 0};
  scene.materials = mats;
  scene.hadron_range = hrt;
  scene.range_cut = real_t(0.7);
  scene.scoring_volume = 0;

  // hadElastic OFF, and said rather than assumed: this file has no elastic tables to hand
  // `HadronicWiring`, and a null table would make `elastic_xs_per_volume` return zero anyway -
  // but a zero that came from an absent table and a zero that came from a switch are different
  // statements, and `tests/test_step_hadron.cu` is where the elastic branch is exercised.
  had::HadronicWiring<real_t> had{};
  had.stage = had::HadronicStage::kStage1;
  had.decay = true;
  had.hadron_elastic = false;
  had.neutron_capture = false;

  // ---------------------------------------------------------------------------- 1. the scaling
  //
  // Which species the two sets of entry points agree for, as arithmetic on the ratios rather
  // than as a claim. This is the anti-vacuity clause of section 2: eight of the eleven rows
  // below would pass section 2 with `lookup` in place of `range_for`.
  std::printf("-- 1. massRatio and chargeSqRatio per species, in water --\n");
  {
    struct Row { ParticleType t; const char* name; bool identity; };
    const Row rows[] = {
        {ParticleType::kProton, "proton", true},
        {ParticleType::kAntiProton, "anti_proton", true},
        {ParticleType::kPionPlus, "pi+", true},
        {ParticleType::kPionMinus, "pi-", true},
        {ParticleType::kKaonPlus, "kaon+", true},
        {ParticleType::kKaonMinus, "kaon-", true},
        {ParticleType::kMuonPlus, "mu+", true},
        {ParticleType::kMuonMinus, "mu-", true},
        {ParticleType::kAlpha, "alpha", true},
        {ParticleType::kGenericIon, "GenericIon", true},
        {ParticleType::kDeuteron, "deuteron", false},
        {ParticleType::kTriton, "triton", false},
        {ParticleType::kHe3, "He3", false},
    };
    const real_t e = real_t(50);
    for (const Row& r : rows) {
      const real_t mr = em::hadron_mass_ratio<real_t>(r.t);
      const real_t qr = em::hadron_charge_sq_ratio<real_t>(mats[data::kWater], r.t, e);
      const real_t scaled = hrt->range_for(mats[data::kWater], r.t, data::kWater, e);
      const real_t raw = hrt->lookup(em::hadron_species_of<real_t>(r.t), data::kWater, e);
      const double ratio = (raw > 0) ? double(scaled) / double(raw) : 0.0;
      std::printf("  %-12s massRatio %.9g  chargeSqRatio %.9g   range_for/lookup %.6f\n",
                  r.name, double(mr), double(qr), ratio);
      if (r.identity) {
        // EXACTLY one, not nearly one: G4VEnergyLossProcess sets both to 1.0 for a particle
        // with no base particle, so a scaled lookup and a raw one are the same double and every
        // number these ten species have ever produced is unchanged by this commit.
        if (mr != real_t(1) || qr != real_t(1) || scaled != raw) {
          fail("%s should be its own base particle: massRatio %.17g chargeSqRatio %.17g, "
               "range_for %.17g lookup %.17g", r.name, double(mr), double(qr), double(scaled),
               double(raw));
        }
      } else if (std::fabs(ratio - 1.0) < 0.2) {
        fail("%s reads a base particle's table and the scaling moved its range by only %.2f%% - "
             "section 2 could then pass with the unscaled lookup in place",
             r.name, 100.0 * (ratio - 1.0));
      }
    }
  }

  // ------------------------------------------------------- 2. the transport against its table
  //
  // A range IS the path length of a track that stops, so the two have to agree. Averaged over
  // 200 tracks because a single one straggles: `G4UniversalFluctuation` puts a few per cent of
  // width on the path length of one proton and none on the mean.
  //
  // The tolerance is 1.5%, and what sets it is not the statistics. `step_hadron` stops a track
  // at `kHadronTrackingCut` (1 keV) or 10 um of residual range, whichever comes first, so the
  // path is short of the full range by that much - 10 um out of 14 mm for the deuteron below is
  // 0.07% - and the step function integrates dE/dx over finite steps, which for a curve this
  // convex biases the path a little long near the peak.
  std::printf("\n-- 2. total true path length of a stopping track, against range_for --\n");
  {
    struct Beam { ParticleType t; const char* name; real_t ekin; };
    const Beam beams[] = {
        {ParticleType::kProton, "proton", real_t(100)},
        {ParticleType::kAlpha, "alpha", real_t(200)},
        {ParticleType::kDeuteron, "deuteron", real_t(50)},
        {ParticleType::kTriton, "triton", real_t(50)},
    };
    constexpr int kN = 200;
    for (const Beam& b : beams) {
      const double want = double(hrt->range_for(mats[data::kWater], b.t, data::kWater, b.ekin));
      const double raw =
          double(hrt->lookup(em::hadron_species_of<real_t>(b.t), data::kWater, b.ekin));
      double path = 0, edep = 0, sec = 0;
      int steps = 0, escaped = 0;
      for (int i = 0; i < kN; ++i) {
        const Stopped st = follow(scene, b.t, b.ekin, had, 0x51A0u + 7919u * unsigned(i));
        path += st.path;
        edep += st.edep;
        sec += st.secondary;
        steps += st.steps;
        if (st.escaped) { ++escaped; }
      }
      const double mean_path = path / kN;
      const double dev = mean_path / want - 1.0;
      // Energy in equals energy deposited plus energy that left as a transportable secondary.
      // Exact to rounding: nothing in this configuration carries energy out of the event.
      const double bal = (edep + sec) / (kN * double(b.ekin)) - 1.0;
      std::printf("  %-9s %6.6g MeV  path %10.6f mm  range_for %10.6f mm  (%+.3f%%)  "
                  "unscaled %10.6f mm  balance %+.2e  %.1f steps\n",
                  b.name, double(b.ekin), mean_path, want, 100.0 * dev, raw, bal,
                  double(steps) / kN);
      if (escaped > 0) { fail("%s: %d of %d tracks left the world", b.name, escaped, kN); }
      if (std::fabs(dev) > 0.015) {
        fail("%s: the transport ran %.3f mm and the table it reads says %.3f mm (%+.2f%%)",
             b.name, mean_path, want, 100.0 * dev);
      }
      if (std::fabs(bal) > 1e-12) {
        fail("%s: %.3e of the beam energy is unaccounted for", b.name, bal);
      }
    }
  }

  std::printf("\n%s (%d failures)\n", g_fails == 0 ? "PASS" : "FAIL", g_fails);
  return g_fails == 0 ? 0 : 1;
}
