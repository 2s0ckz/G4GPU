// src/physics/em/coulomb_scattering.cuh against the real G4CoulombScattering.
//
// Three blocks, against the three CSVs ref/dump/dump_coulomb.cc writes, and they are three
// different kinds of check.
//
//   1. WHO GETS IT.  coulomb_limits.csv is read off the CONSTRUCTED QBBC - one row per model
//      on each species' own process manager - and compared against the species x (model,
//      limits) table in coulomb_scattering.cuh's file header. A table in a comment is a
//      measurement and decays like one (docs/RISK.md V44's rule about the three macro headers
//      that said a pion stops in B1); this is what stops it decaying. It also checks the
//      absences: alpha, He3, deuteron, triton and GenericIon must have NO CoulombScat, which
//      is what G4EmBuilder::ConstructIonEmPhysics's lack of an isWVI argument means.
//   2. THE CROSS SECTION, exactly. Ten species x six Z x 73 energies per material, against
//      G4eCoulombScatteringModel::ComputeCrossSectionPerAtom. Deterministic, so the tolerance
//      is near machine precision - and the angular interval it was integrated over is
//      compared too, because the same cross section over the wrong interval is the one error
//      this quantity can make while looking right.
//   3. THE ANGLE SAMPLER, statistically. 20,000 SampleSingleScattering draws per cell under a
//      fixed seed, with (Z, A) and the target mass FIXED so the isotope draw this port refuses
//      is out of the comparison. Moments and a log-spaced histogram of 1 - cos(theta).
//
// WHY n_scattered IS A COLUMN AND NOT A DETAIL. `SampleSingleScattering` returns (0,0,1) - no
// deflection at all - whenever its rejection loop fails, and for a 200 MeV proton on oxygen it
// fails 90% of the time. So the mean of cos(theta) over all draws is dominated by the
// un-scattered ones and is a weak check; the scattered FRACTION is what tests the rejection
// function, which is where factD and the form factor live. Both are compared.
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "core/particle.cuh"
#include "core/rng.cuh"
#include "data/materials.cuh"
#include "physics/em/coulomb_scattering.cuh"

using namespace g4gpu;
using real_t = double;

namespace {

int material_of(const std::string& n) {
  if (n == "G4_AIR") { return data::kAir; }
  if (n == "G4_WATER") { return data::kWater; }
  if (n == "G4_A-150_TISSUE") { return data::kA150Tissue; }
  if (n == "G4_BONE_COMPACT_ICRU") { return data::kBoneCompact; }
  return -1;
}

/// The ten species with the process and the five without, with what the file-header table in
/// coulomb_scattering.cuh claims for each. `low`/`high`/`act` are the SS model's own limits in
/// MeV; -1 means "no CoulombScat at all".
struct Claim {
  const char* name;
  ParticleType type;
  double low, high, act;
};
const Claim kClaims[] = {
    // e+- : SetMinKinEnergy / SetLowEnergyLimit / SetActivationLowEnergyLimit, all
    // G4EmParameters::MscEnergyLimit() = 100 MeV.
    {"e-", ParticleType::kElectron, 100.0, 1e8, 100.0},
    {"e+", ParticleType::kPositron, 100.0, 1e8, 100.0},
    // The isWVI branch: no limits set at all, so the model keeps G4VEmModel's defaults of
    // 100 eV and 100 TeV and an activation limit of zero.
    {"mu+", ParticleType::kMuonPlus, 1e-4, 1e8, 0.0},
    {"mu-", ParticleType::kMuonMinus, 1e-4, 1e8, 0.0},
    {"pi+", ParticleType::kPionPlus, 1e-4, 1e8, 0.0},
    {"pi-", ParticleType::kPionMinus, 1e-4, 1e8, 0.0},
    {"kaon+", ParticleType::kKaonPlus, 1e-4, 1e8, 0.0},
    {"kaon-", ParticleType::kKaonMinus, 1e-4, 1e8, 0.0},
    {"proton", ParticleType::kProton, 1e-4, 1e8, 0.0},
    {"anti_proton", ParticleType::kAntiProton, 1e-4, 1e8, 0.0},
    // No process. G4EmBuilder::ConstructIonEmPhysics takes no isWVI argument.
    {"alpha", ParticleType::kAlpha, -1, -1, -1},
    {"He3", ParticleType::kHe3, -1, -1, -1},
    {"deuteron", ParticleType::kDeuteron, -1, -1, -1},
    {"triton", ParticleType::kTriton, -1, -1, -1},
    {"GenericIon", ParticleType::kGenericIon, -1, -1, -1},
};
constexpr int kNClaims = int(sizeof kClaims / sizeof kClaims[0]);

int claim_of(const std::string& n) {
  for (int i = 0; i < kNClaims; ++i) {
    if (n == kClaims[i].name) { return i; }
  }
  return -1;
}

struct Worst {
  double dev = 0;
  int n = 0;
  char where[240] = {0};
};
void note(Worst& w, double dev, const char* fmt, ...) {
  ++w.n;
  if (dev <= w.dev) { return; }
  w.dev = dev;
  va_list ap;
  va_start(ap, fmt);
  std::vsnprintf(w.where, sizeof w.where, fmt, ap);
  va_end(ap);
}

/// The one number every angular quantity in this file is measured against: option0's
/// G4EmParameters::MscThetaLimit() is pi, so both cosThetaMin and cosThetaMax are -1 and the
/// single-scattering process covers the whole angular range. Checked against the dump's
/// msc_theta_limit column rather than assumed.
constexpr double kCosThetaMin = -1.0;
constexpr double kCosThetaMax = -1.0;

}  // namespace

int main() {
  data::Material<real_t> mats[data::kNumMaterials];
  data::build_b1_materials<real_t>(mats);

  const char* env = std::getenv("G4GPU_ORACLE");
  const std::string dir = (env != nullptr) ? env : "ref/oracle";
  int fails = 0;

  // ---------------------------------------------------------------- 1. who gets it
  {
    FILE* f = std::fopen((dir + "/coulomb_limits.csv").c_str(), "r");
    if (f == nullptr) {
      std::printf("cannot read %s/coulomb_limits.csv - run ref/oracle/run.bat tables first\n",
                  dir.c_str());
      return 1;
    }
    char line[1024];
    std::fgets(line, sizeof line, f);
    bool seen[kNClaims] = {false};
    int rows = 0;
    double theta_limit = 0, factor_angle = 0, q2max = 0, msc_limit = 0;
    while (std::fgets(line, sizeof line, f) != nullptr) {
      char part[64], proc[64], model[64];
      int subtype, build;
      double low, high, act, tl, fa, q2, ml;
      const int n = std::sscanf(line,
                                "%63[^,],%63[^,],%d,%d,%63[^,],%lf,%lf,%lf,%lf,%lf,%lf,%lf",
                                part, proc, &subtype, &build, model, &low, &high, &act, &tl,
                                &fa, &q2, &ml);
      if (n != 12) { continue; }
      ++rows;
      theta_limit = tl;
      factor_angle = fa;
      q2max = q2;
      msc_limit = ml;
      const int ci = claim_of(part);
      if (ci < 0) { continue; }
      const bool is_ss = (std::strcmp(proc, "CoulombScat") == 0);
      if (kClaims[ci].low < 0) {
        // The five that must not have it.
        if (is_ss) {
          std::printf("  FAIL: %s has CoulombScat and the header table says it does not\n",
                      kClaims[ci].name);
          ++fails;
        }
        seen[ci] = true;
        continue;
      }
      if (!is_ss) { continue; }
      seen[ci] = true;
      if (std::strcmp(model, "eCoulombScattering") != 0) {
        std::printf("  FAIL: %s CoulombScat model is %s, not eCoulombScattering\n",
                    kClaims[ci].name, model);
        ++fails;
      }
      if (std::fabs(low - kClaims[ci].low) > 1e-9 * std::fabs(kClaims[ci].low) + 1e-12
          || std::fabs(high - kClaims[ci].high) > 1e-9 * kClaims[ci].high
          || std::fabs(act - kClaims[ci].act) > 1e-9 * kClaims[ci].act + 1e-12) {
        std::printf("  FAIL: %s limits are (%g, %g, %g), the header table says (%g, %g, %g)\n",
                    kClaims[ci].name, low, high, act, kClaims[ci].low, kClaims[ci].high,
                    kClaims[ci].act);
        ++fails;
      }
    }
    std::fclose(f);
    for (int i = 0; i < kNClaims; ++i) {
      if (!seen[i]) {
        std::printf("  FAIL: %s has no row in coulomb_limits.csv - the header table claims "
                    "something about a species the dump never saw\n", kClaims[i].name);
        ++fails;
      }
    }
    // The angular parameters the whole file turns on, against the port's own constants.
    if (std::fabs(theta_limit - units::pi<real_t>()) > 1e-9) {
      std::printf("  FAIL: MscThetaLimit is %.12g, not pi - cosThetaMin/-Max of %g/%g in this "
                  "file and in coulomb_scattering.cuh are wrong\n", theta_limit, kCosThetaMin,
                  kCosThetaMax);
      ++fails;
    }
    if (std::fabs(factor_angle - 1.0) > 1e-12) {
      std::printf("  FAIL: FactorForAngleLimit is %.12g, not 1\n", factor_angle);
      ++fails;
    }
    const double ours_q2 = em::coulomb_q2_max<real_t>();
    if (std::fabs(ours_q2 - q2max) > 1e-13 * q2max) {
      std::printf("  FAIL: q2Max is %.17g, Geant4 says %.17g\n", ours_q2, q2max);
      ++fails;
    }
    std::printf("== who gets G4CoulombScattering ==\n  %d model rows, MscThetaLimit=%.12g, "
                "FactorForAngleLimit=%g, q2Max=%.10g MeV^2, MscEnergyLimit=%g MeV\n",
                rows, theta_limit, factor_angle, ours_q2, msc_limit);
  }

  // ---------------------------------------------------------------- 2. cross section per atom
  //
  // Deterministic, so the limit is near machine precision. The interval is compared as an
  // absolute difference in cos(theta), not relative: cos_t_max is -1 exactly for every species
  // but a proton on hydrogen, and a relative comparison against -1 hides nothing but is
  // meaningless where the value is 0.
  {
    FILE* f = std::fopen((dir + "/coulomb_xs.csv").c_str(), "r");
    if (f == nullptr) {
      std::printf("cannot read %s/coulomb_xs.csv\n", dir.c_str());
      return 1;
    }
    char line[1024];
    std::fgets(line, sizeof line, f);
    Worst xs[kNClaims], interval[kNClaims], pmin[kNClaims], mmin[kNClaims];
    long zeros_both = 0, zeros_us_only = 0, zeros_g4_only = 0;
    while (std::fgets(line, sizeof line, f) != nullptr) {
      char mat[64], part[64];
      int Z, A;
      double e, ecut, pcut, x, xn, xe, ctmin, ctmax, ctnuc, ctelec, inva23, tmass, pm, mm;
      const int n = std::sscanf(line,
                                "%63[^,],%63[^,],%d,%d,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,"
                                "%lf,%lf,%lf,%lf,%lf",
                                mat, part, &Z, &A, &e, &ecut, &pcut, &x, &xn, &xe, &ctmin,
                                &ctmax, &ctnuc, &ctelec, &inva23, &tmass, &pm, &mm);
      if (n != 18) { continue; }
      const int mi = material_of(mat);
      const int ci = claim_of(part);
      if (mi < 0 || ci < 0 || kClaims[ci].low < 0) { continue; }
      const ParticleDef<real_t> pd = particle_def<real_t>(kClaims[ci].type);
      const em::CoulombAtomXs<real_t> a =
          em::coulomb_xs_per_atom(pd, kClaims[ci].type, e, real_t(inva23), Z, real_t(ecut),
                                  real_t(kCosThetaMin), real_t(kCosThetaMax));
      // Both zero is the process being off below its threshold, and it must be off at the
      // same energies on both sides - a port that never returns zero would pass a relative
      // comparison on every nonzero point and be wrong about where the process exists.
      const bool g4z = !(x > 0), usz = !(a.total > 0);
      if (g4z && usz) { ++zeros_both; ++xs[ci].n; continue; }
      if (usz) {
        ++zeros_us_only;
        std::printf("  FAIL: zero here and %.6g in Geant4: %s %s Z=%d at %.6g MeV\n", x, mat,
                    part, Z, e);
        ++fails;
        continue;
      }
      if (g4z) {
        ++zeros_g4_only;
        std::printf("  FAIL: %.6g here and zero in Geant4: %s %s Z=%d at %.6g MeV\n",
                    double(a.total), mat, part, Z, e);
        ++fails;
        continue;
      }
      note(xs[ci], std::fabs(a.total - x) / x, "%s Z=%d at %.6g MeV (%.10g vs %.10g mm^2)",
           mat, Z, e, double(a.total), x);
      const double di = std::max(std::fabs(double(a.cos_t_min) - ctmin),
                                 std::fabs(double(a.cos_t_max) - ctmax));
      note(interval[ci], di, "%s Z=%d at %.6g MeV (min %.17g vs %.17g, max %.17g vs %.17g)",
           mat, Z, e, double(a.cos_t_min), ctmin, double(a.cos_t_max), ctmax);
      // The process's own table threshold, and the model's. Both are per (material, species)
      // and repeat down the file; comparing them on every row costs nothing and means a wrong
      // one cannot hide in a material nothing else exercises.
      const double ourpm = em::coulomb_process_min_primary_energy<real_t>(pd.mass,
                                                                          real_t(inva23));
      note(pmin[ci], std::fabs(ourpm - pm) / pm, "%s at %.6g MeV (%.10g vs %.10g MeV)", mat, e,
           ourpm, pm);
      if (mm > 0) {
        // The lightest element of the material, which is what the model's MinPrimaryEnergy
        // uses. Every B1 material contains hydrogen except none of them, so this is Z=1
        // throughout - but it is derived and not assumed.
        int zl = 300;
        for (int k = 0; k < mats[mi].n_elements; ++k) {
          const int zz = int(mats[mi].z[k] + 0.5);
          if (zz < zl) { zl = zz; }
        }
        const double ourmm =
            em::coulomb_model_min_primary_energy<real_t>(real_t(pcut), zl);
        note(mmin[ci], std::fabs(ourmm - mm) / mm, "%s (Zmin=%d, %.10g vs %.10g MeV)", mat, zl,
             ourmm, mm);
      }
    }
    std::fclose(f);
    std::printf("\n== cross section per atom vs ComputeCrossSectionPerAtom ==\n");
    std::printf("  %-12s %12s %12s %12s %12s %7s\n", "", "xs", "interval", "proc Emin",
                "model Emin", "n");
    for (int i = 0; i < kNClaims; ++i) {
      if (kClaims[i].low < 0 || xs[i].n == 0) { continue; }
      std::printf("  %-12s %11.3e %11.3e %11.3e %11.3e %7d\n", kClaims[i].name, xs[i].dev,
                  interval[i].dev, pmin[i].dev, mmin[i].dev, xs[i].n);
      // 1e-12 relative: the cross section is four multiplies and two divides on numbers the
      // port and Geant4 both compute in double from the same constants, and the measured
      // worst case is below 1e-14. Not 1e-15, because kin_factor carries `coeff` built from
      // CLHEP's classic_electr_radius and the last bit of that product is not reproducible
      // across a different order of operations.
      if (xs[i].dev > 1e-12) {
        std::printf("    FAIL: %s xs, %s\n", kClaims[i].name, xs[i].where);
        ++fails;
      }
      if (interval[i].dev > 1e-14) {
        std::printf("    FAIL: %s interval, %s\n", kClaims[i].name, interval[i].where);
        ++fails;
      }
      if (pmin[i].dev > 1e-12) {
        std::printf("    FAIL: %s process MinPrimaryEnergy, %s\n", kClaims[i].name,
                    pmin[i].where);
        ++fails;
      }
      // The model's MinPrimaryEnergy goes through G4NucleiProperties::GetNuclearMass, which
      // this port answers from data/nuclei_mass_ame12.hh - a table with its own two levels
      // and its own refusal - so it is held to 1e-9 rather than to the cross section's limit.
      if (mmin[i].dev > 1e-9) {
        std::printf("    FAIL: %s model MinPrimaryEnergy, %s\n", kClaims[i].name,
                    mmin[i].where);
        ++fails;
      }
    }
    std::printf("  zeros: %ld agreed, %ld ours only, %ld Geant4 only\n", zeros_both,
                zeros_us_only, zeros_g4_only);
    if (zeros_both == 0) {
      std::printf("  FAIL: no row had a zero cross section on either side, so the threshold "
                  "the process turns on at was never tested\n");
      ++fails;
    }
  }

  // ---------------------------------------------------------------- 3. the angle sampler
  {
    FILE* f = std::fopen((dir + "/coulomb_sample.csv").c_str(), "r");
    if (f == nullptr) {
      std::printf("cannot read %s/coulomb_sample.csv\n", dir.c_str());
      return 1;
    }
    char line[4096];
    std::fgets(line, sizeof line, f);
    std::printf("\n== SampleSingleScattering, 20,000 draws per cell ==\n");
    std::printf("  %-12s %4s %10s %12s %12s %12s\n", "particle", "Z", "E(MeV)", "P(scatter)",
                "<1-cos>", "chi2/bin");
    int cells = 0;
    while (std::fgets(line, sizeof line, f) != nullptr) {
      char mat[64], part[64];
      int Z, A, N;
      long nsc;
      double e, m1, m2, m3, mt, tmax, ratio, ctmin, ctmax, tmass;
      long h[20];
      char* p = line;
      // Fixed-width sscanf up to the histogram, then the twenty bins by hand.
      const int n = std::sscanf(p,
                                "%63[^,],%63[^,],%d,%d,%lf,%d,%ld,%lf,%lf,%lf,%lf,%lf,%lf,"
                                "%lf,%lf,%lf",
                                mat, part, &Z, &A, &e, &N, &nsc, &m1, &m2, &m3, &mt, &tmax,
                                &ratio, &ctmin, &ctmax, &tmass);
      if (n != 16) { continue; }
      int commas = 0;
      while (*p != '\0' && commas < 16) {
        if (*p == ',') { ++commas; }
        ++p;
      }
      bool ok = true;
      for (int b = 0; b < 20; ++b) {
        char* end = nullptr;
        h[b] = std::strtol(p, &end, 10);
        if (end == p) { ok = false; break; }
        p = (*end == ',') ? end + 1 : end;
      }
      if (!ok) { continue; }
      const int mi = material_of(mat);
      const int ci = claim_of(part);
      if (mi < 0 || ci < 0 || kClaims[ci].low < 0) { continue; }

      // The port's sampler, same N, same (Z, A, target mass), same angular interval. The
      // stream cannot be the same - Geant4's is CLHEP's HepJamesRandom and this is the port's
      // own generator - so this is a distribution comparison and nothing else. Its seed is
      // derived from the cell so that a cell reproduces on its own.
      const ParticleDef<real_t> pd = particle_def<real_t>(kClaims[ci].type);
      const em::CoulombAtomXs<real_t> a =
          em::coulomb_xs_per_atom(pd, kClaims[ci].type, real_t(e), mats[mi].inv_a23, Z,
                                  mats[mi].cut_electron, real_t(kCosThetaMin),
                                  real_t(kCosThetaMax));
      const em::WentzelState<real_t> st =
          em::wentzel_setup(pd, kClaims[ci].type, real_t(e), mats[mi].inv_a23, Z,
                            mats[mi].cut_electron, real_t(kCosThetaMin));
      // Philox, keyed by the cell, so a cell reproduces on its own the way the oracle side
      // does with CLHEP::HepRandom::setTheSeed.
      Philox<real_t> rng(0x5cabu + unsigned(ci) * 977u, unsigned(Z) * 31u,
                         unsigned(std::log10(e) * 4));
      long oh[20] = {0};
      long onsc = 0;
      double o1 = 0, o3 = 0;
      for (int k = 0; k < N; ++k) {
        const Vec3<real_t> v =
            em::coulomb_sample_single(st, Z, real_t(tmass), a.cos_t_min, a.cos_t_max,
                                      a.elec_ratio, rng);
        const double cost = double(v.z);
        o1 += cost;
        o3 += 1.0 - cost;
        if (cost < 1.0) { ++onsc; }
        int b = 0;
        if (1.0 - cost > 0.0) {
          b = int((std::log10(1.0 - cost) + 10.0) * 0.1 * 20);
          if (b < 0) { b = 0; }
          if (b > 19) { b = 19; }
        }
        ++oh[b];
      }
      // The scattered fraction, as a binomial: the two sides are independent draws of the same
      // Bernoulli, so the difference has sigma sqrt(2 p (1-p) / N).
      const double p1 = double(nsc) / N, p2 = double(onsc) / N;
      const double sig = std::sqrt(std::max(2.0 * p1 * (1.0 - p1) / N, 1.0 / (N * double(N))));
      const double zsc = std::fabs(p1 - p2) / sig;
      // The histogram, as a Pearson chi-square over the bins either side has entries in.
      double chi2 = 0;
      int nb = 0;
      for (int b = 0; b < 20; ++b) {
        const double s = double(h[b] + oh[b]);
        if (s < 20) { continue; }   // too few to be chi-square distributed
        const double d = double(h[b] - oh[b]);
        chi2 += d * d / s;
        ++nb;
      }
      const double chi2n = (nb > 0) ? chi2 / nb : 0.0;
      ++cells;
      // 5 sigma on the scattered fraction and 12 per bin on the chi-square. Both are wide on
      // purpose: two different generators drawing 20,000 samples of a distribution that spans
      // ten decades in 1 - cos will not agree bin for bin, and the alternative to a wide limit
      // here is a limit that fails at random. What makes this a check rather than a formality
      // is the anti-vacuity run in the commit message: dropping the 1/(1 + z1*factD) factor
      // from the rejection function - which is exactly what em/wentzel_msc.cuh's copy of this
      // sampler does - moves the proton's scattered fraction by 30 sigma.
      const bool bad = (zsc > 5.0) || (chi2n > 12.0);
      if (bad || Z == 8 || Z == 1) {
        std::printf("  %-12s %4d %10.4g %6.4f/%6.4f %11.4g %11.3f%s\n", part, Z, e, p1, p2, m3,
                    chi2n, bad ? "   <-- FAIL" : "");
      }
      if (bad) {
        std::printf("    nsc %ld vs %ld (%.1f sigma), <1-cos> %.6g vs %.6g, chi2/bin %.2f "
                    "over %d bins\n", nsc, onsc, zsc, m3, o3 / N, chi2n, nb);
        ++fails;
      }
    }
    std::fclose(f);
    std::printf("  %d cells compared\n", cells);
    if (cells == 0) {
      std::printf("  FAIL: no sampler cells compared\n");
      ++fails;
    }
  }

  std::printf("\n%s (%d failures)\n", fails ? "FAILED" : "PASSED", fails);
  return fails ? 1 : 0;
}
