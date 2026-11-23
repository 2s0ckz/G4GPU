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
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "core/rng.cuh"
#include "data/isotope_abundance.hh"
#include "data/materials.cuh"
#include "data/natural_isotopes.hh"
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

  __host__ __device__ int push(ParticleType, const Vec3<real_t>&, real_t ekin, int,
                               unsigned short = 0) {
    ++child_count;
    ++n;
    secondary_energy += static_cast<double>(ekin);
    return 0;
  }
  /// The nuclide is discarded here on purpose: this emitter exists to close an energy balance,
  /// and the elastic branch it would come from is switched off below. `push_nucleus` has to
  /// EXIST because `step_hadron` calls it, which is the same reason tests/test_neutron.cu gives
  /// for carrying `pos`, `volume` and `event`.
  __host__ __device__ int push_nucleus(int z, int a, const Vec3<real_t>& dir, real_t ekin,
                                       int event_id) {
    return push(particle_type_of_nucleus(z, a), dir, ekin, event_id, ion_za_of(z, a));
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

  // -------------------------------------------------- 3. the nuclide, and what carrying it cost
  //
  // docs/RISK.md V22 names a wider `TrackState` as the one thing the register budget cannot
  // afford, so the claim that this field is free is arithmetic here rather than a comment in
  // the header. `species` is an int at offset 0 and `pos` is a `Vec3<double>` that needs
  // 8-byte alignment, so four bytes of padding already existed at offset 4.
  std::printf("\n-- 3. TrackState::ion_za: the encoding, and its cost --\n");
  {
    using TS = TrackState<real_t>;
    std::printf("  sizeof(TrackState<double>) = %zu, offsetof(ion_za) = %zu, "
                "offsetof(pos) = %zu\n",
                sizeof(TS), offsetof(TS, ion_za), offsetof(TS, pos));
    if (offsetof(TS, ion_za) != sizeof(int)) {
      fail("ion_za is at offset %zu, not immediately after `species` - it is only free where "
           "the padding is", offsetof(TS, ion_za));
    }
    if (offsetof(TS, pos) != 8) {
      fail("`pos` is at offset %zu, so ion_za did not land in pre-existing padding and the "
           "struct grew", offsetof(TS, pos));
    }
    // The round trip, over every nuclide the transport can meet: every natural isotope of
    // every element. `ion_za_of` has to be injective over that set or two nuclides share a
    // track field.
    int checked = 0;
    for (int z = 1; z <= data::kNistBuildableMaxZ; ++z) {
      for (int a = z; a <= 4 * z + 8; ++a) {
        if (!data::is_natural_isotope(z, a)) { continue; }
        const unsigned short za = ion_za_of(z, a);
        if (za == 0 || ion_z_of(za) != z || ion_a_of(za) != a) {
          fail("(Z=%d, A=%d) does not round-trip: encoded %u decodes to (%d, %d)", z, a,
               unsigned(za), ion_z_of(za), ion_a_of(za));
        }
        ++checked;
      }
    }
    std::printf("  %d natural isotopes round-trip through a 16-bit field\n", checked);
    if (checked < 250) { fail("only %d isotopes were checked - the loop missed most", checked); }
    // And the four refusals, because a zero has to mean exactly one thing.
    if (ion_za_of(0, 16) != 0 || ion_za_of(8, 0) != 0 || ion_za_of(8, 512) != 0
        || ion_za_of(128, 16) != 0) {
      fail("ion_za_of does not refuse an out-of-range nuclide with zero");
    }
  }

  // ------------------------------------------------- 4. the ion's definition, from (Z, A)
  std::printf("\n-- 4. ion_particle_def against G4IonTable::CreateIon's own inputs --\n");
  {
    struct N { int z, a; const char* name; };
    const N nuclides[] = {{6, 12, "C12"},  {7, 14, "N14"},  {8, 16, "O16"},
                          {20, 40, "Ca40"}, {8, 18, "O18"}, {82, 208, "Pb208"}};
    for (const N& n : nuclides) {
      const ParticleDef<real_t> pd = em::ion_particle_def<real_t>(n.z, n.a);
      const real_t want = data::nuclear_mass<real_t>(n.a, n.z);
      std::printf("  %-6s Z=%2d A=%3d  mass %14.7f MeV  charge %5.1f  is_ion %d  "
                  "massRatio %.9g\n",
                  n.name, n.z, n.a, double(pd.mass), double(pd.charge), int(pd.is_ion),
                  double(em::hadron_mass_ratio<real_t>(em::stepped_ion<real_t>(n.z, n.a))));
      // G4IonTable::CreateIon: `mass = GetNucleusMass(Z, A) + Eex` with Eex = 0, and
      // `GetNucleusMass` is `G4NucleiProperties::GetNuclearMass(A, Z)` for Z > 2. Bit for bit,
      // because the elastic recoil's kinematics were computed with the same function and a
      // transported recoil must be the particle that was emitted.
      if (pd.mass != want) {
        fail("%s mass %.17g, G4NucleiProperties::GetNuclearMass %.17g", n.name, double(pd.mass),
             double(want));
      }
      if (pd.charge != static_cast<real_t>(n.z) || !pd.is_ion || pd.is_alpha) {
        fail("%s: charge %.17g is_ion %d is_alpha %d", n.name, double(pd.charge),
             int(pd.is_ion), int(pd.is_alpha));
      }
      // The mass ratio is G4GenericIon's own literal over the nuclide's - NOT
      // units::proton_mass_c2 over it. The two differ by 3e-7 and core/particle.cuh keeps them
      // apart on purpose.
      const real_t mr = em::hadron_mass_ratio<real_t>(em::stepped_ion<real_t>(n.z, n.a));
      if (mr != particle_def<real_t>(ParticleType::kGenericIon).mass / want) {
        fail("%s massRatio is not m(G4GenericIon)/m(ion)", n.name);
      }
    }
    // The five nuclides that are NOT GenericIon, so that a recoil of one of them is stepped as
    // itself with its own definition rather than through the ion path.
    struct L { int z, a; ParticleType t; const char* name; };
    const L light[] = {{1, 1, ParticleType::kProton, "proton"},
                       {1, 2, ParticleType::kDeuteron, "deuteron"},
                       {1, 3, ParticleType::kTriton, "triton"},
                       {2, 3, ParticleType::kHe3, "He3"},
                       {2, 4, ParticleType::kAlpha, "alpha"}};
    for (const L& l : light) {
      if (particle_type_of_nucleus(l.z, l.a) != l.t) {
        fail("(%d, %d) should map to %s", l.z, l.a, l.name);
      }
    }
    // A nuclide outside AME2012 has the mass Geant4 gives it. Until P24 it had none, and
    // `step_hadron` refused it by name (`kIonWithoutNuclide`); data/nuclei_mass_ame12.hh now
    // carries G4NucleiProperties' theoretical table and Weizsaecker's formula, and (Z=8, A=40)
    // is the formula's 37,493.505693394261 MeV in Geant4 and here (tests/test_hadronic_xs.cu
    // holds every 1 <= A <= 400 to the oracle exactly). What is still refused is what Geant4
    // itself gives zero, a "nuclide" with Z > A.
    if (!data::nuclear_mass_known(40, 8) ||
        em::ion_particle_def<real_t>(8, 40).mass != data::nuclear_mass<real_t>(40, 8) ||
        em::ion_particle_def<real_t>(8, 40).mass != real_t(37493.505693394261)) {
      fail("(Z=8, A=40) is past AME2012 and must come back with Geant4's formula mass");
    }
    if (data::nuclear_mass_known(8, 9) || em::ion_particle_def<real_t>(9, 8).mass != real_t(0)) {
      fail("(Z=9, A=8) has Z > A, Geant4's mass is zero, and it must come back with none");
    }
  }

  // ----------------------------------------------- 5. the ion, stepped to a stop in water
  //
  // The same two assertions as section 2 - the path length against the range, and the energy
  // balance - now for a species whose definition is not its species. The energies are the
  // recoil band the elastic process actually produces (the recoil threshold is 70 keV, and a
  // 100 MeV proton's oxygen recoils run to a few MeV) plus one above it.
  //
  // WHAT THE RANGE IS AT THESE ENERGIES IS THE POINT of the whole exercise: a 0.55 MeV oxygen
  // ion goes about a micrometre. Until P45 that was three decades under `step_hadron`'s 10 um
  // `kMinUsefulRange` guard, so the ion died on its first step with a zero path and its energy
  // was deposited where the scatter happened. Inside one volume that is the same deposit, which
  // is why this section passed with the guard in place; at a boundary it is not - P27's C12 and
  // Fe56 at 1 MeV/n fired 1 um before the trapezoid (ranges 9.3 and 6.4 um) left all of their
  // energy in the envelope, where Geant4 crosses (docs/RISK.md V342). `G4VEnergyLossProcess` has
  // no range floor, only the 1 keV `lowestKinEnergy`, so a real nucleus is now STEPPED to its
  // range at any range, and the assertion below says so; He3 keeps the floor with the other
  // species that have kernels of their own.
  std::printf("\n-- 5. a nucleus stepped to a stop in water --\n");
  {
    struct Ion { int z, a; const char* name; real_t ekin; };
    const Ion ions[] = {
        {8, 16, "O16", real_t(0.55)},   {8, 16, "O16", real_t(20)},
        {6, 12, "C12", real_t(5)},      {7, 14, "N14", real_t(5)},
        {20, 40, "Ca40", real_t(20)},   {2, 3, "He3", real_t(20)},
    };
    constexpr int kN = 100;
    for (const Ion& io : ions) {
      const bool is_he3 = (io.z == 2 && io.a == 3);
      const ParticleType t = is_he3 ? ParticleType::kHe3 : ParticleType::kGenericIon;
      const em::SteppedHadron<real_t> h =
          is_he3 ? em::stepped_hadron<real_t>(ParticleType::kHe3)
                 : em::stepped_ion<real_t>(io.z, io.a);
      const double want = double(hrt->range_for(mats[data::kWater], h, data::kWater, io.ekin));
      const double dedx = double(hrt->dedx_for(mats[data::kWater], h, data::kWater, io.ekin));
      const double q2 = double(em::hadron_charge_sq_ratio<real_t>(mats[data::kWater], h,
                                                                  io.ekin));
      double path = 0, edep = 0, sec = 0;
      int steps = 0, escaped = 0;
      for (int i = 0; i < kN; ++i) {
        TrackState<real_t> p{};
        p.species = t;
        p.ion_za = is_he3 ? 0 : ion_za_of(io.z, io.a);
        p.pos = Vec3<real_t>{0, 0, 0};
        p.dir = Vec3<real_t>{0, 0, 1};
        p.ekin = io.ekin;
        p.volume = 0;
        p.rng_key = 0x100Du + 7919u * unsigned(i);
        p.begin(p.pos, p.dir, p.ekin, 0, 0u, ProcessId::fNotDefined, real_t(0), real_t(1));
        CountingEmitter em;
        bool alive = true;
        int n = 0;
        while (alive && n < kMaxStepsPerTrack) {
          StepReport<real_t> rep;
          Philox<real_t> rng(p.rng_key, p.step, 0xB19Du);
          em.pos = p.pos;
          em.volume = p.volume;
          em.event = p.event;
          real_t ed = 0;
          alive = step_hadron(scene, p, t, had, rng, em, ed, rep);
          ++p.step;
          ++n;
          path += static_cast<double>(rep.true_length);
          edep += static_cast<double>(ed);
          if (p.volume == geom::kOutsideWorld) { ++escaped; break; }
        }
        sec += em.secondary_energy;
        steps += n;
      }
      const double mean_path = path / kN;
      const double bal = (edep + sec) / (kN * double(io.ekin)) - 1.0;
      std::printf("  %-5s %6.6g MeV  q_eff^2 %7.3f  dE/dx %11.4f MeV/mm  range %10.3e mm  "
                  "path %10.3e mm  balance %+.2e  %.1f steps\n",
                  io.name, double(io.ekin), q2, dedx, want, mean_path, bal,
                  double(steps) / kN);
      if (escaped > 0) { fail("%s: %d tracks left the world", io.name, escaped); }
      if (std::fabs(bal) > 1e-12) {
        fail("%s at %g MeV: %.3e of the energy is unaccounted for", io.name, double(io.ekin),
             bal);
      }
      // ---- P44 (docs/RISK.md V351): THERE IS NO STOPPING GUARD, AND THAT IS THE ASSERTION.
      // Until P44 `step_hadron` killed a track with less than 10 um of residual range where it
      // stood, and this section asserted it: a zero path on the first step below 10 um, the range
      // less the guard above. Geant4 stops a track at 1 keV or at the end of its range and
      // nowhere else, and so does the port now - an O16 recoil at 0.55 MeV (1.3 um) runs its
      // range in one step (the step function's `x = range` below finalRange) and stops at its
      // end. So every regime is the same statement: the path is the range, to section 2's 2%.
      {
        const double dev = mean_path / want - 1.0;
        const double bound = 0.02;
        if (std::fabs(dev) > bound) {
          fail("%s at %g MeV: the transport ran %.4e mm and the table says %.4e mm (%+.2f%%, "
               "bound %.2f%%)",
               io.name, double(io.ekin), mean_path, want, 100.0 * dev, 100.0 * bound);
        }
      }
      // The effective charge has to be doing something, or section 5 is a test of a bare
      // charge. For O16 at 0.55 MeV (34 keV/u) the ion is far from stripped.
      if (!is_he3 && q2 >= double(io.z) * double(io.z)) {
        fail("%s at %g MeV: q_eff^2 = %.3f is not below the bare Z^2 = %d", io.name,
             double(io.ekin), q2, io.z * io.z);
      }
    }
  }

  // ------------------------------------ 6. the ion's dE/dx, range and effective charge, exactly
  //
  // `ref/oracle/ion_tables.csv` is `G4EmCalculator::GetDEDX`, `::GetRange` and
  // `G4EmCorrections::EffectiveChargeSquareRatio` for eleven real nuclides in each of B1's four
  // materials, 61 energies per row from 10 keV to 1 GeV. `ref/dump/dump_ions.cc` says what
  // each column is and why `GetDEDX`'s own `isIon` correction is inert over the 1 nm step it
  // uses - which is what makes the column a statement about the TABLE and the effective charge
  // and nothing else.
  //
  // THE THIRD COLUMN IS THE POINT. dE/dx and range both contain `q2_eff` and the mass ratio, so
  // agreeing on them and disagreeing on the charge is possible only by a compensating error -
  // but `q2_eff` dumped on its own pins `ion_effective_charge` times its `chargeCorrection`
  // against Geant4's own number, which is the piece no other test in this tree reaches for a
  // heavy ion (`tests/test_ion_charge.cu` covers the model; this is the model as the transport
  // composes it).
  std::printf("\n-- 6. against ref/oracle/ion_tables.csv --\n");
  {
    const char* env = std::getenv("G4GPU_ORACLE");
    const std::string dir = (env != nullptr) ? std::string(env) : std::string("ref/oracle");
    const std::string path = dir + "/ion_tables.csv";
    FILE* f = std::fopen(path.c_str(), "r");
    if (f == nullptr) {
      std::printf("  cannot read %s - run ref/oracle/run.bat tables first\n", path.c_str());
      fail("no ion oracle: this section is the only place the ion's effective charge is "
           "compared against Geant4's own number");
    } else {
      char line[512];
      // Worst deviation per quantity, and the row it was on.
      double worst[3] = {0, 0, 0};
      char where[3][128] = {{0}, {0}, {0}};
      int compared = 0, skipped = 0;
      // The oracle's first data line is a `#` row carrying G4GenericIon's own mass, so that
      // the mass ratio is checked against the literal Geant4 uses rather than against
      // proton_mass_c2. Read and asserted, not skipped.
      bool saw_gion = false;
      while (std::fgets(line, sizeof line, f) != nullptr) {
        char mat[64] = {0}, ion[32] = {0};
        int z = 0, a = 0;
        double mass = 0, charge = 0, e = 0, dedx = 0, range = 0, q2 = 0, mr = 0;
        if (line[0] == '#') {
          // `#,G4GenericIon,0,0,<mass>,<charge>,...`
          if (std::sscanf(line, "#,%31[^,],%d,%d,%lf,%lf", ion, &z, &a, &mass, &charge) == 5) {
            saw_gion = true;
            const real_t ours = particle_def<real_t>(ParticleType::kGenericIon).mass;
            if (ours != static_cast<real_t>(mass)) {
              fail("particle_def(kGenericIon).mass is %.17g, G4GenericIon's is %.17g - the mass "
                   "ratio of every ion is against this number", double(ours), mass);
            }
          }
          continue;
        }
        if (std::sscanf(line, "%63[^,],%31[^,],%d,%d,%lf,%lf,%lf,%lf,%lf,%lf,%lf", mat, ion, &z,
                        &a, &mass, &charge, &e, &dedx, &range, &q2, &mr) != 11) {
          continue;  // the header
        }
        int mi = -1;
        if (std::string(mat) == "G4_AIR") { mi = data::kAir; }
        else if (std::string(mat) == "G4_WATER") { mi = data::kWater; }
        else if (std::string(mat) == "G4_A-150_TISSUE") { mi = data::kA150Tissue; }
        else if (std::string(mat) == "G4_BONE_COMPACT_ICRU") { mi = data::kBoneCompact; }
        if (mi < 0 || !(dedx > 0) || !(range > 0) || !(q2 > 0)) { ++skipped; continue; }
        if (!data::nuclear_mass_known(a, z)) { ++skipped; continue; }
        const em::SteppedHadron<real_t> h = em::stepped_ion<real_t>(z, a);
        const double got[3] = {
            double(hrt->dedx_for(mats[mi], h, mi, real_t(e))),
            double(hrt->range_for(mats[mi], h, mi, real_t(e))),
            double(em::hadron_charge_sq_ratio<real_t>(mats[mi], h, real_t(e))),
        };
        const double want[3] = {dedx, range, q2};
        for (int k = 0; k < 3; ++k) {
          const double dev = std::fabs(got[k] - want[k]) / want[k];
          if (dev > worst[k]) {
            worst[k] = dev;
            std::snprintf(where[k], sizeof where[k], "%s %s %g MeV: %.10g vs %.10g", ion, mat,
                          e, got[k], want[k]);
          }
        }
        ++compared;
      }
      std::fclose(f);
      if (!saw_gion) { fail("the oracle has no G4GenericIon mass row"); }
      // The skips are all materials: the dump writes every material its own detector builds
      // (seven) and this port carries B1's four, so 3 x 11 x 61 rows have no index here.
      std::printf("  %d rows compared, %d skipped for a material this port does not build\n",
                  compared, skipped);
      if (compared < 1000) {
        fail("only %d rows compared - the oracle is not the one dump_ions.cc writes", compared);
      }
      // dE/dx and range are the table's own accuracy times the effective charge's, and
      // `tests/test_hadron_range.cu` already bounds GenericIon's own row against Geant4 (0.001%
      // at 1-10 keV, worse in the bands its own table reports). The limit here is what the
      // SCALING adds on top of that, so it is the same order as that file's cells rather than
      // machine precision. `q2_eff` IS machine precision: it is one closed-form function of
      // (Z, A, material, E) with no table under it.
      const double limit[3] = {0.05, 0.05, 1e-12};
      static const char* qn[3] = {"dE/dx", "range", "q2_eff"};
      for (int k = 0; k < 3; ++k) {
        std::printf("  worst %-7s %11.4e   %s\n", qn[k], worst[k], where[k]);
        if (worst[k] > limit[k]) {
          fail("%s worst deviation %.4e exceeds %.0e - %s", qn[k], worst[k], limit[k], where[k]);
        }
      }
    }
  }

  // ----------------------------------- 7. a nucleus 1 um before a boundary crosses it (P45)
  //
  // docs/RISK.md V342: P27's C12 and Fe56 at 1 MeV/n, fired 1 um before B1's trapezoid, were
  // deposited where they started - `step_hadron`'s 10 um range floor stopped them in place, ranges
  // 9.32 and 6.40 um - and the trapezoid read -100% against Geant4. Geant4's own step tape of the
  // same primaries (ref/b1mat, /tracking/verbose 1, Geant4 11.1.1 QBBC, three tracks each):
  //
  //     Fe56 56 MeV -> water:  step 1  1 um   Transportation  dE 5.097 / 4.972 / 5.064 MeV
  //                            step 2  5.787-5.802 um  ionIoni in Shape2, the rest deposited
  //     Fe56 56 MeV -> bone:   step 2  3.425-3.434 um in Shape2
  //     C12  12 MeV -> water:  step 1  1 um   dE 752.1 / 713.2 / 741.7 keV
  //                            step 2  8.525-8.565 um in Shape2
  //
  // So the geometry here is that: a water world, and a box of water or bone whose face is 1 um
  // ahead. The assertions are the tape's: every track takes a 1 um Transportation step first, the
  // envelope keeps the 1 um's loss (Geant4's three tracks are 4.97-5.10 MeV and 0.71-0.75 MeV; the
  // mean here must be within 5% of their mean), and the box receives the rest. With the floor
  // back the box receives NOTHING and every row fails by 100%.
  std::printf("\n-- 7. a nucleus fired 1 um before a boundary crosses it --\n");
  {
    geom::Volume<real_t> fv[2] = {
        {{geom::SolidType::kBox, {kHalf, kHalf, kHalf}},
         geom::make_translation<real_t>({0, 0, 0}), 0, data::kWater, /*score_index=*/-1},
        {{geom::SolidType::kBox, {50, 50, 30}}, geom::make_translation<real_t>({0, 0, 30.001}), 1,
         data::kWater, /*score_index=*/0},
    };
    struct Row { int z, a; const char* name; real_t ekin; int mat; double g4_env; };
    // Geant4's envelope loss, the mean of the tape's three tracks.
    const Row rows[] = {
        {26, 56, "Fe56 water", real_t(56), data::kWater, (5.097 + 4.972 + 5.064) / 3.0},
        {26, 56, "Fe56 bone", real_t(56), data::kBoneCompact, (5.097 + 4.972 + 5.064) / 3.0},
        {6, 12, "C12 water", real_t(12), data::kWater, (0.7521 + 0.7132 + 0.7417) / 3.0},
    };
    constexpr int kN = 200;
    for (const Row& r : rows) {
      fv[1].material = r.mat;
      Scene<real_t> fs = scene;
      fs.geometry = geom::Geometry<real_t>{fv, 2, 0};
      fs.scoring_volume = 1;
      double box = 0, env = 0;
      int first_not_boundary = 0;
      for (int i = 0; i < kN; ++i) {
        TrackState<real_t> p{};
        p.species = ParticleType::kGenericIon;
        p.ion_za = ion_za_of(r.z, r.a);
        p.pos = Vec3<real_t>{0, 0, 0};
        p.dir = Vec3<real_t>{0, 0, 1};
        p.ekin = r.ekin;
        p.volume = 0;
        p.rng_key = 0x45F0u + 7919u * unsigned(i);
        p.begin(p.pos, p.dir, p.ekin, 0, 0u, ProcessId::fNotDefined, real_t(0), real_t(1));
        CountingEmitter em;
        bool alive = true;
        int n = 0;
        while (alive && n < kMaxStepsPerTrack) {
          StepReport<real_t> rep;
          Philox<real_t> rng(p.rng_key, p.step, 0xB19Du);
          em.pos = p.pos;
          em.volume = p.volume;
          em.event = p.event;
          const real_t e0 = p.ekin;
          const int v0 = p.volume;
          real_t ed = 0;
          alive = step_hadron(fs, p, ParticleType::kGenericIon, had, rng, em, ed, rep);
          ++p.step;
          if (n == 0 && rep.status != StepStatus::fGeomBoundary) { ++first_not_boundary; }
          ++n;
          box += static_cast<double>(ed);
          // The envelope is not scored, so its share is what the ion lost while in it.
          if (v0 == 0) { env += static_cast<double>(e0 - (alive ? p.ekin : real_t(0))); }
          if (p.volume == geom::kOutsideWorld) { break; }
        }
      }
      box /= kN;
      env /= kN;
      std::printf("  %-11s %5.4g MeV  envelope %8.5f MeV (Geant4 %8.5f)  box %9.5f MeV\n",
                  r.name, double(r.ekin), env, r.g4_env, box);
      if (first_not_boundary > 0) {
        fail("%s: %d of %d tracks did not end their first step on the boundary 1 um ahead",
             r.name, first_not_boundary, kN);
      }
      if (std::fabs(env / r.g4_env - 1.0) > 0.05) {
        fail("%s: the 1 um before the face kept %.4f MeV and Geant4's tape %.4f MeV", r.name,
             env, r.g4_env);
      }
      if (std::fabs(box + env - double(r.ekin)) > 1e-9 * double(r.ekin)) {
        fail("%s: box %.6f + envelope %.6f MeV is not the %g MeV fired", r.name, box, env,
             double(r.ekin));
      }
    }
  }

  std::printf("\n%s (%d failures)\n", g_fails == 0 ? "PASS" : "FAIL", g_fails);
  return g_fails == 0 ? 0 : 1;
}
