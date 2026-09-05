// Delta rays from a heavy charged particle: the rate and the spectrum.
//
// A sampler is harder to check than a formula. The formula has a number in ref/oracle to
// compare against; the sampler has a distribution, and "it ran and produced electrons" is not
// evidence of anything. So this compares moments, and it compares them against two independent
// things rather than one:
//
//   1. The *rate*, hadron_delta_xs, against Geant4's own delta_xs_per_mm column. That is the
//      zeroth moment and it is a direct oracle comparison - it is what catches the Bragg
//      models' 0.25 keV * massRate floor, which G4BetheBlochModel does not have and which the
//      first version of the cross section silently dropped.
//
//   2. The *spectrum*, by Monte Carlo, against the closed-form first moment of the same
//      differential cross section. The sampler inverts 1/T^2 and rejects; the integral
//      integral_a^b T dsigma/dT is elementary. If the sampler drew from a subtly different
//      density - a wrong bound, a missing rejection - the mean transfer moves and the closed
//      form does not.
//
//   3. The two together: rate * mean transfer is the energy per mm leaving as delta rays, which
//      must be the difference between unrestricted and restricted dE/dx. This is the check that
//      would fail if the *cross section* and the *sampler* were each self-consistent but
//      disagreed with the stopping power the range table is built from - i.e. if a proton
//      stepped through this transport lost a different amount of energy than its range said it
//      should. It is the loosest of the three and the most physically meaningful.
//
// Plus the bookkeeping that a stepper depends on: the primary's energy after the transfer, and
// that both directions come back as unit vectors.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "core/particle.cuh"
#include "core/rng.cuh"
#include "data/materials.cuh"
#include "physics/em/hadron_delta.cuh"

using namespace g4gpu;
using real_t = double;

namespace {

struct Row {
  std::string mat, part;
  double e = 0, xs = 0;
};

std::vector<Row> load(const std::string& path) {
  std::vector<Row> out;
  FILE* f = std::fopen(path.c_str(), "r");
  if (f == nullptr) { return out; }
  char line[1024];
  if (std::fgets(line, sizeof line, f) == nullptr) {
    std::fclose(f);
    return out;
  }
  while (std::fgets(line, sizeof line, f) != nullptr) {
    char mat[128], part[32];
    double mass, q, e, di, dt, r, dx, db, dp, bx, px, nd;
    const int n = std::sscanf(line,
                              "%127[^,],%31[^,],%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf",
                              mat, part, &mass, &q, &e, &di, &dt, &r, &dx, &db, &dp, &bx, &px,
                              &nd);
    if (n != 14) { continue; }
    out.push_back({mat, part, e, dx});
  }
  std::fclose(f);
  return out;
}

ParticleType type_of(const std::string& n) {
  if (n == "proton") { return ParticleType::kProton; }
  if (n == "alpha") { return ParticleType::kAlpha; }
  return ParticleType::kNumTypes;
}

int material_of(const std::string& n) {
  if (n == "G4_AIR") { return data::kAir; }
  if (n == "G4_WATER") { return data::kWater; }
  if (n == "G4_A-150_TISSUE") { return data::kA150Tissue; }
  if (n == "G4_BONE_COMPACT_ICRU") { return data::kBoneCompact; }
  return -1;
}

/// The closed-form first moment of the density the sampler draws from.
///
/// dsigma/dT is proportional to 1/T^2 - beta^2/(T*Tmax) [+ 0.5/E^2 with spin], so
///
///   integral_a^b T dsigma/dT dT  =  K * [ ln(b/a) - beta^2 (b-a)/Tmax + 0.25 (b^2-a^2)/E^2 ]
///
/// with the same K = twopi_mc2_rcl2 * z^2 / beta^2 * n_e that the zeroth moment carries. The
/// spin term is included only when the sampler includes it, which is Bethe-Bloch and not Bragg
/// - see the note in hadron_delta.cuh about Geant4 carrying it in G4BraggModel's cross section
/// but not in its sampler. Getting that condition wrong here rather than there would produce a
/// test that agrees with a sampler nobody wrote.
double mean_transfer_analytic(const data::Material<real_t>& m, ParticleType type, double kinetic,
                              double cut, double* rate_out) {
  const ParticleDef<real_t> pd = particle_def<real_t>(type);
  const double tmax = em::hadron_max_secondary_energy(pd, kinetic);
  const bool bragg = em::is_bragg_family(em::hadron_ioni_model(type, pd, kinetic));
  double a = cut;
  if (bragg) {
    a = std::fmax(cut, em::bragg_lowest_energy<real_t>() * pd.mass / units::proton_mass_c2<real_t>());
  }
  const double b = tmax;
  if (a >= b) {
    *rate_out = 0;
    return 0;
  }
  const double etot = kinetic + pd.mass;
  const double beta2 = kinetic * (kinetic + 2 * pd.mass) / (etot * etot);
  const bool spin = (!bragg && pd.spin > 0);

  // The K factor cancels in the ratio, so it need not be right for the *mean*; it is written
  // out anyway because the zeroth moment computed here is cross-checked against
  // hadron_delta_xs, and a K that cancelled everywhere would make that check vacuous.
  const double K = units::twopi<real_t>() * units::classic_electron_radius<real_t>()
                   * units::classic_electron_radius<real_t>() * units::electron_mass_c2<real_t>()
                   * pd.charge * pd.charge / beta2 * m.electron_density;

  double zeroth = 1.0 / a - 1.0 / b - beta2 * std::log(b / a) / tmax;
  double first = std::log(b / a) - beta2 * (b - a) / tmax;
  if (spin) {
    zeroth += 0.5 * (b - a) / (etot * etot);
    first += 0.25 * (b * b - a * a) / (etot * etot);
  }
  *rate_out = K * zeroth;
  return first / zeroth;
}

struct Cell {
  int n = 0;
  double worst = 0;
  std::string where;
};

}  // namespace

int main() {
  data::Material<real_t> mats[data::kNumMaterials];
  data::build_b1_materials<real_t>(mats);
  static em::ShellTables<real_t> shell;
  em::build_shell_tables(shell);

  const char* env = std::getenv("G4GPU_ORACLE");
  const std::string dir = (env != nullptr) ? env : "ref/oracle";
  const auto rows = load(dir + "/hadron_tables.csv");
  if (rows.empty()) {
    std::printf("cannot read %s/hadron_tables.csv - run ref/oracle/run.bat first\n", dir.c_str());
    return 1;
  }

  int fails = 0;

  // ------------------------------------------------------------ 1. the rate, against Geant4
  {
    Cell c[2];
    for (const Row& r : rows) {
      const ParticleType t = type_of(r.part);
      if (t == ParticleType::kNumTypes) { continue; }
      const int mi = material_of(r.mat);
      if (mi < 0 || r.xs <= 0 || r.e > 1e4) { continue; }
      const int pi = (t == ParticleType::kProton) ? 0 : 1;
      // The oracle's arguments, not the transport's, and the difference is not cosmetic.
      // G4EmCalculator::ComputeCrossSectionPerVolume floors the cut at
      // G4EmParameters::LowestElectronEnergy (1 keV) and passes maxEnergy = kinEnergy;
      // G4VEnergyLossProcess passes the raw production cut and lets maxEnergy default to
      // DBL_MAX, which is what hadron_delta_xs is called with in transport.
      //
      // Where the cut is well below tmax the two agree to nothing worth measuring. Where it is
      // not - a 0.5 MeV proton in air, tmax 1.09 keV against a 0.99 keV cut - the cross section
      // is the difference of two nearly equal numbers and the 1% shift in the cut moves it by
      // 12%. That is what the first run of this test reported, and it was the comparison that
      // was wrong, not the function: reproducing the oracle's convention here is the only way
      // the number means anything.
      const real_t cut = fmax(mats[mi].cut_electron, real_t(1e-3));
      const double ours = em::hadron_delta_xs(mats[mi], t, real_t(r.e), cut, real_t(r.e));
      const double dev = std::fabs(ours - r.xs) / r.xs;
      ++c[pi].n;
      if (dev > c[pi].worst) {
        c[pi].worst = dev;
        char buf[200];
        std::snprintf(buf, sizeof buf, "in %s at %.4g MeV (ours %.6g, G4 %.6g /mm)",
                      r.mat.c_str(), r.e, ours, r.xs);
        c[pi].where = buf;
      }
    }
    std::printf("== delta-ray rate vs G4EmCalculator::ComputeCrossSectionPerVolume ==\n");
    for (int p = 0; p < 2; ++p) {
      std::printf("  %-8s %6d points  worst %8.4f%%  %s\n", p == 0 ? "proton" : "alpha", c[p].n,
                  100 * c[p].worst, c[p].where.c_str());
      // 0.1%: the formula is the same handful of terms as Geant4's, so the only differences
      // are the electron density and the constants. Anything above this is a transcription
      // error, not a rounding one.
      if (c[p].n > 0 && c[p].worst > 1e-3) {
        std::printf("  FAIL: rate off by %.4f%%, limit 0.1%%\n", 100 * c[p].worst);
        ++fails;
      }
      if (c[p].n == 0) {
        std::printf("  FAIL: no points compared\n");
        ++fails;
      }
    }
    std::printf("\n");
  }

  // ------------------------------------------------------------ 2. the spectrum, by sampling
  //
  // 200k samples per point gives a standard error on the mean of roughly 0.3% for the 1/T^2
  // density - it is heavy-tailed, so the error is dominated by the rare large transfers and
  // falls slowly. The tolerance below is set from that, not from the physics.
  {
    const ParticleType parts[2] = {ParticleType::kProton, ParticleType::kAlpha};
    const char* pname[2] = {"proton", "alpha"};
    const double energies[] = {0.5, 1.5, 5.0, 20.0, 100.0, 500.0};
    constexpr int kN = 200000;

    double worst = 0, worst_frac = 0;
    std::string where, where_frac;
    int npoints = 0;

    for (int p = 0; p < 2; ++p) {
      for (int mi = 0; mi < data::kNumMaterials; ++mi) {
        for (double e : energies) {
          const real_t cut = mats[mi].cut_electron;
          double rate = 0;
          const double want = mean_transfer_analytic(mats[mi], parts[p], e, cut, &rate);
          if (!(want > 0)) { continue; }

          Philox<real_t> rng(static_cast<uint32_t>(mi * 97 + p * 13),
                             static_cast<uint32_t>(e * 1000), 7u);
          double sum = 0;
          int fired = 0;
          for (int i = 0; i < kN; ++i) {
            const auto d = em::sample_hadron_delta<real_t>(parts[p], real_t(e), cut,
                                                           Vec3<real_t>{0, 0, 1}, rng);
            if (!d.produced) { continue; }
            ++fired;
            sum += d.delta_ekin;

            // Bookkeeping, on every sample rather than a spot check: the stepper reads all
            // four of these fields and a NaN or a non-unit direction here becomes a lost track
            // a long way downstream.
            const double etot_check = d.delta_ekin + d.primary_ekin;
            if (std::fabs(etot_check - e) > 1e-9 * e) {
              std::printf("  FAIL: energy not conserved: %.17g + %.17g != %.17g\n", d.delta_ekin,
                          d.primary_ekin, e);
              ++fails;
              i = kN;
            }
            if (std::fabs(mag(d.delta_dir) - 1) > 1e-9 || std::fabs(mag(d.primary_dir) - 1) > 1e-9) {
              std::printf("  FAIL: direction not a unit vector (|delta| %.17g, |primary| %.17g)\n",
                          mag(d.delta_dir), mag(d.primary_dir));
              ++fails;
              i = kN;
            }
            if (d.delta_ekin < cut * 0.999 || d.delta_ekin > e) {
              std::printf("  FAIL: delta energy %.6g outside [cut %.6g, E %.6g]\n", d.delta_ekin, cut, e);
              ++fails;
              i = kN;
            }
          }
          if (fired == 0) { continue; }
          ++npoints;

          const double got = sum / fired;
          const double dev = std::fabs(got / want - 1);
          if (dev > worst) {
            worst = dev;
            char buf[220];
            std::snprintf(buf, sizeof buf, "%s in material %d at %g MeV (%.6g vs %.6g MeV)",
                          pname[p], mi, e, got, want);
            where = buf;
          }
          // The form factor is the only thing that can suppress a sample, and below a GeV it
          // does nothing at all - so every draw should fire. A shortfall here means the
          // rejection loop is running out of tries, which would bias the spectrum silently.
          const double frac = 1.0 - double(fired) / kN;
          if (frac > worst_frac) {
            worst_frac = frac;
            char buf[160];
            std::snprintf(buf, sizeof buf, "%s in material %d at %g MeV", pname[p], mi, e);
            where_frac = buf;
          }
        }
      }
    }
    std::printf("== sampled mean transfer vs the closed-form first moment ==\n");
    std::printf("  %d points, %d samples each\n", npoints, kN);
    std::printf("  worst: %.3f%%  (%s)\n", 100 * worst, where.c_str());
    std::printf("  worst suppressed fraction: %.4f%%  (%s)\n\n", 100 * worst_frac,
                where_frac.c_str());
    if (npoints == 0) {
      std::printf("  FAIL: nothing sampled\n");
      ++fails;
    }
    if (worst > 0.02) {
      std::printf("  FAIL: sampled mean off by %.3f%%, limit 2%% (2e5 samples of a 1/T^2 tail)\n",
                  100 * worst);
      ++fails;
    }
    if (worst_frac > 1e-3) {
      std::printf("  FAIL: %.4f%% of draws produced nothing; the form factor cannot do that "
                  "below a GeV, so the rejection loop is giving up\n", 100 * worst_frac);
      ++fails;
    }
  }

  // ------------------------------------------------ 3. rate * mean vs the dE/dx restriction
  //
  // The closure that ties the sampler to the range table. Energy leaving as delta rays per mm
  // is what the restricted stopping power leaves out, so
  //
  //     xs * <T>  ==  dEdx(unrestricted) - dEdx(cut)
  //
  // must hold, and it is computed here from three functions that share no code: the cross
  // section, the analytic first moment, and the stopping power. A proton whose steps lost a
  // different amount of energy than its own range table predicted would show up here and
  // nowhere else - the range table is built from the restricted dE/dx, and the deltas are
  // sampled from the cross section, and nothing else compares the two.
  //
  // Analytic on both sides, so the residual is a real discrepancy and not Monte Carlo noise.
  // It is split by particle because the two have different reasons to be non-zero and one
  // number over both would hide the proton behind the alpha:
  //
  //   proton - closes to rounding. Geant4's restriction term is the exact integral of its own
  //     delta cross section: d/dTup of [ln(2 me bg^2 Tup/I^2) - (1+Tup/tmax)beta^2 +
  //     0.25 Tup^2/E^2] is [1/Tup - beta^2/tmax + 0.5 Tup/E^2], which is T dsigma/dT term for
  //     term. What is left is HighOrderCorrections(..., cutEnergy), the one correction that
  //     takes the cut as an argument.
  //
  //   alpha - about a per cent above 8 MeV, and it is Geant4's own inconsistency rather than
  //     this port's. G4BetheBlochModel::CrossSectionPerVolume scales an alpha's delta rate by
  //     EffectiveChargeSquareRatio/chargeSquare; its ComputeDEDXPerVolume does not, and the
  //     process does not put it back either - G4BetheBlochModel::GetChargeSquareRatio returns
  //     exactly 1.0 for an alpha. So Geant4 makes deltas at an effective-charge rate and
  //     accounts for them at a nominal-charge rate, and the gap is that ratio: 1.5% at 8.9 MeV,
  //     falling as the ion strips. Below 8 MeV, where G4BraggIonModel applies the same
  //     heChargeSquare to both, it closes to rounding. The limit is sized to the larger.
  {
    const ParticleType parts[2] = {ParticleType::kProton, ParticleType::kAlpha};
    const char* pname[2] = {"proton", "alpha"};
    double worst[2] = {0, 0};
    std::string where[2];
    int n[2] = {0, 0};
    for (int p = 0; p < 2; ++p) {
      for (int mi = 0; mi < data::kNumMaterials; ++mi) {
        for (double e = 2.5; e < 1e3; e *= 1.6) {
          const real_t cut = mats[mi].cut_electron;
          double unused = 0;
          const double meanT = mean_transfer_analytic(mats[mi], parts[p], e, cut, &unused);
          // The rate comes from the shipped function, not from the analytic zeroth moment
          // above: it is the one the stepper will call, and for an alpha it carries an
          // effective-charge factor the bare formula does not.
          const double rate = em::hadron_delta_xs(mats[mi], parts[p], real_t(e), cut,
                                                  real_t(1e30));
          if (!(rate > 0) || !(meanT > 0)) { continue; }
          const double to_deltas = rate * meanT;

          const double restricted =
              em::hadron_ioni_dedx(mats[mi], parts[p], real_t(e), cut, &shell);
          const double unrestricted =
              em::hadron_ioni_dedx(mats[mi], parts[p], real_t(e), real_t(1e30), &shell);
          const double want = unrestricted - restricted;
          if (!(want > 0)) { continue; }
          ++n[p];
          const double dev = std::fabs(to_deltas / want - 1);
          if (dev > worst[p]) {
            worst[p] = dev;
            char buf[220];
            std::snprintf(buf, sizeof buf,
                          "in material %d at %.4g MeV (xs*<T> %.6g, dEdx diff %.6g MeV/mm)", mi,
                          e, to_deltas, want);
            where[p] = buf;
          }
        }
      }
    }
    std::printf("== rate * mean transfer vs (unrestricted - restricted) dE/dx ==\n");
    // Limits are per particle and both are measured, not guessed. A sampler drawing from the
    // wrong density - a missing rejection, the wrong lower bound - moves these by tens of per
    // cent, which is the size of error they exist to catch.
    const double limit[2] = {0.002, 0.02};
    for (int p = 0; p < 2; ++p) {
      std::printf("  %-8s %4d points  worst %6.3f%%  (limit %.1f%%)  %s\n", pname[p], n[p],
                  100 * worst[p], 100 * limit[p], where[p].c_str());
      if (n[p] == 0) {
        std::printf("  FAIL: nothing compared for %s\n", pname[p]);
        ++fails;
      }
      if (worst[p] > limit[p]) {
        std::printf("  FAIL: %s delta energy budget off by %.3f%%, limit %.1f%%\n", pname[p],
                    100 * worst[p], 100 * limit[p]);
        ++fails;
      }
    }
    std::printf("\n");
  }

  std::printf("%s (%d failures)\n", fails ? "FAILED" : "PASSED", fails);
  return fails ? 1 : 0;
}
