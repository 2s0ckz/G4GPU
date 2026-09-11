// src/physics/em/coulomb_scattering.cuh against the real G4CoulombScattering.
//
// Four blocks, against the four CSVs ref/dump/dump_coulomb.cc writes, and they are four
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
//   4. THE RECOIL, exactly. One row per real SampleSecondaries call, carrying the angle Geant4
//      drew and the target it drew, so the recoil energy, the primary's final energy, the
//      local deposit, the threshold branch and the recoil ion's direction are compared point by
//      point rather than through the statistics of a second random stream. This is the block
//      the `coulomb_recoil` / `coulomb_sample_secondaries` split exists for.
//
// The cut in blocks 2 to 4 is the PROTON production cut, not the electron one, because
// G4CoulombScattering's secondary particle is the proton and so `theCuts` is cuts index 3 - see
// coulomb_scattering.cuh's header. Block 2 compares the cross section at BOTH cuts and finds
// them equal to the last bit, which is the finding rather than a formality: the only quantity
// the cut moves is `cosTetMaxElec`, and the electron channel that gates is closed over this
// process's whole interval. Block 2 prints how close it comes to opening.
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
      // `build_table`, which is the observable half of InitialiseProcess's
      // `mass > GeV || GetParticleType() == "nucleus"` branch: that branch calls
      // SetBuildTableFlag(false) AND swaps in G4IonCoulombScatteringModel. The header section
      // on the four switches claims it is never taken in option0 because the heaviest species
      // with this process is the proton at 938.272 MeV; a 1 here on every row is what says so,
      // and the model-name check above is the other half of the same claim.
      if (build != 1) {
        std::printf("  FAIL: %s CoulombScat has build_table=%d, so InitialiseProcess took its "
                    "mass > GeV branch and the model is not the electron one\n",
                    kClaims[ci].name, build);
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
    Worst xsp[kNClaims], elec[kNClaims];
    long zeros_both = 0, zeros_us_only = 0, zeros_g4_only = 0;
    long n_active = 0, n_g4_elec = 0, n_our_elec = 0;
    double min_elec_gap = 1e300;
    while (std::fgets(line, sizeof line, f) != nullptr) {
      char mat[64], part[64];
      int Z, A;
      double e, ecut, pcut, x, xn, xe, ctmin, ctmax, ctnuc, ctelec, inva23, tmass, pm, mm;
      double xp, xep, ctelecp;
      const int n = std::sscanf(line,
                                "%63[^,],%63[^,],%d,%d,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,"
                                "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf",
                                mat, part, &Z, &A, &e, &ecut, &pcut, &x, &xn, &xe, &ctmin,
                                &ctmax, &ctnuc, &ctelec, &inva23, &tmass, &pm, &mm, &xp, &xep,
                                &ctelecp);
      if (n != 21) { continue; }
      const int mi = material_of(mat);
      const int ci = claim_of(part);
      if (mi < 0 || ci < 0 || kClaims[ci].low < 0) { continue; }
      const ParticleDef<real_t> pd = particle_def<real_t>(kClaims[ci].type);
      const em::CoulombAtomXs<real_t> a =
          em::coulomb_xs_per_atom(pd, kClaims[ci].type, e, real_t(inva23), Z, real_t(ecut),
                                  real_t(kCosThetaMin), real_t(kCosThetaMax));
      // THE SAME FUNCTION AT THE OTHER CUT. The cut the TRANSPORT passes is the proton's, not
      // the electron's (coulomb_scattering.cuh's header has the four source lines), and it
      // reaches the cross section through exactly one quantity: ComputeMaxElectronScattering's
      // cosTetMaxElec, and through that the electron cross section. Both cuts are compared, and
      // what the pair MEASURES is that the cross sections are identical while cosTetMaxElec is
      // not - the electron channel is closed over this process's whole angular interval, so
      // the cut is a plumbing fact with no numerical consequence in option0. The counters below
      // are what would notice if that stopped being true.
      const em::CoulombAtomXs<real_t> ap =
          em::coulomb_xs_per_atom(pd, kClaims[ci].type, e, real_t(inva23), Z, real_t(pcut),
                                  real_t(kCosThetaMin), real_t(kCosThetaMax));
      if (xp > 0 && ap.total > 0) {
        note(xsp[ci], std::fabs(double(ap.total) - xp) / xp,
             "%s Z=%d at %.6g MeV, pcut %.4g (%.10g vs %.10g mm^2)", mat, Z, e, pcut,
             double(ap.total), xp);
        // How closed the electron channel is, and on both sides. `xe`/`xep` are Geant4's own
        // electron cross sections and the two `electron` fields are the port's; if either ever
        // became nonzero the two cut columns would stop being equal and this says so first.
        ++n_active;
        if (xe > 0 || xep > 0) { ++n_g4_elec; }
        if (a.electron > 0 || ap.electron > 0) { ++n_our_elec; }
        const double gap = ctelec - ctnuc;
        if (gap < min_elec_gap) { min_elec_gap = gap; }
      }
      // cosTetMaxElec itself, at both cuts, which is the one number the cut moves. Held to an
      // absolute tolerance in cos: it approaches 1 from below and a relative comparison there
      // measures nothing.
      {
        const em::WentzelState<real_t> se =
            em::wentzel_setup(pd, kClaims[ci].type, e, real_t(inva23), Z, real_t(ecut),
                              real_t(kCosThetaMin));
        const em::WentzelState<real_t> sp =
            em::wentzel_setup(pd, kClaims[ci].type, e, real_t(inva23), Z, real_t(pcut),
                              real_t(kCosThetaMin));
        const double de = std::max(std::fabs(double(se.cos_tet_max_elec) - ctelec),
                                   std::fabs(double(sp.cos_tet_max_elec) - ctelecp));
        note(elec[ci], de, "%s Z=%d at %.6g MeV (ecut %.17g vs %.17g, pcut %.17g vs %.17g)",
             mat, Z, e, double(se.cos_tet_max_elec), ctelec, double(sp.cos_tet_max_elec),
             ctelecp);
      }
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
    std::printf("  %-12s %12s %12s %12s %12s %12s %12s %7s\n", "", "xs(ecut)", "xs(pcut)",
                "cosTetElec", "interval", "proc Emin", "model Emin", "n");
    for (int i = 0; i < kNClaims; ++i) {
      if (kClaims[i].low < 0 || xs[i].n == 0) { continue; }
      std::printf("  %-12s %11.3e %11.3e %11.3e %11.3e %11.3e %11.3e %7d\n", kClaims[i].name,
                  xs[i].dev, xsp[i].dev, elec[i].dev, interval[i].dev, pmin[i].dev,
                  mmin[i].dev, xs[i].n);
      if (xsp[i].dev > 1e-12) {
        std::printf("    FAIL: %s xs at the proton cut, %s\n", kClaims[i].name, xsp[i].where);
        ++fails;
      }
      if (elec[i].dev > 1e-14) {
        std::printf("    FAIL: %s cosTetMaxElec, %s\n", kClaims[i].name, elec[i].where);
        ++fails;
      }
      if (xsp[i].n == 0) {
        std::printf("    FAIL: %s never compared at the proton cut, so the cut argument was "
                    "not tested\n", kClaims[i].name);
        ++fails;
      }
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
    // The electron channel, measured rather than asserted away. It is closed over this
    // process's interval at every active point, which is why the two cut columns agree; the gap
    // is how close cosTetMaxElec ever comes to cosTetMaxNuc, and it would have to go NEGATIVE
    // for the channel to open. So `wentzel_electron_xs` and the sampler's electron branch are
    // transcribed and not exercised THROUGH THIS PROCESS - the msc model is where they are.
    std::printf("  electron channel: %ld active rows, Geant4 nonzero in %ld, ours in %ld, "
                "smallest (cosTetMaxElec - cosTetMaxNuc) %+.3g\n", n_active, n_g4_elec,
                n_our_elec, min_elec_gap);
    if (n_g4_elec != n_our_elec) {
      std::printf("  FAIL: the two sides disagree about whether the electron channel is open\n");
      ++fails;
    }
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
    bool cut_reported = false;
    while (std::fgets(line, sizeof line, f) != nullptr) {
      char mat[64], part[64];
      int Z, A, N;
      long nsc;
      double e, m1, m2, m3, mt, tmax, ratio, ctmin, ctmax, tmass, cut;
      long h[20];
      char* p = line;
      // Fixed-width sscanf up to the histogram, then the twenty bins by hand.
      const int n = std::sscanf(p,
                                "%63[^,],%63[^,],%d,%d,%lf,%d,%ld,%lf,%lf,%lf,%lf,%lf,%lf,"
                                "%lf,%lf,%lf,%lf",
                                mat, part, &Z, &A, &e, &N, &nsc, &m1, &m2, &m3, &mt, &tmax,
                                &ratio, &ctmin, &ctmax, &tmass, &cut);
      if (n != 17) { continue; }
      int commas = 0;
      while (*p != '\0' && commas < 17) {
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
      // `cut` off the row, not `mats[mi].cut_electron`: the cut the process hands this model is
      // the PROTON production cut and it is the only thing that moves the electron/nucleus
      // split this sampler draws from. Reading it out of the oracle rather than recomputing it
      // keeps the two sides on the same number by construction; that the port can PRODUCE it
      // is checked separately, against `coulomb_secondary_cut`, below.
      const em::CoulombAtomXs<real_t> a =
          em::coulomb_xs_per_atom(pd, kClaims[ci].type, real_t(e), mats[mi].inv_a23, Z,
                                  real_t(cut), real_t(kCosThetaMin), real_t(kCosThetaMax));
      const em::WentzelState<real_t> st =
          em::wentzel_setup(pd, kClaims[ci].type, real_t(e), mats[mi].inv_a23, Z, real_t(cut),
                            real_t(kCosThetaMin));
      // QBBC's range cut is 0.7 mm and G4RToEConvForProton::Convert is linear in it with no
      // material argument, so one line reproduces every proton cut in the oracle. Reported
      // once, not 240 times.
      if (std::fabs(double(em::coulomb_secondary_cut<real_t>(real_t(0.7))) - cut) > 1e-15
          && !cut_reported) {
        std::printf("  FAIL: coulomb_secondary_cut(0.7 mm) = %.17g, oracle cut = %.17g\n",
                    double(em::coulomb_secondary_cut<real_t>(real_t(0.7))), cut);
        cut_reported = true;
        ++fails;
      }
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
        // mean_trec and max_trec are Geant4's own recoil energies for this cell, printed as
        // context and not asserted here: they are a function of the angles above, so their
        // agreement would be a consequence of it. The recoil is asserted EXACTLY in block 4,
        // against Geant4's per-call rows, which is where an arithmetic error in it belongs.
        std::printf("    nsc %ld vs %ld (%.1f sigma), <1-cos> %.6g vs %.6g, chi2/bin %.2f "
                    "over %d bins; G4 <trec> %.4g MeV, max %.4g MeV\n", nsc, onsc, zsc, m3,
                    o3 / N, chi2n, nb, mt, tmax);
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

  // ---------------------------------------------------------------- 4. the recoil, exactly
  //
  // One row per SampleSecondaries call, so this is not a distribution comparison: the row
  // carries the cos(theta) Geant4 drew and the (Z, A) it drew, and everything after those two
  // is arithmetic that must agree to machine precision. `coulomb_recoil` exists as a separate
  // function for exactly this - to be drivable by an angle rather than by an RNG.
  //
  // WHICH ROWS TEST WHICH THING, because the two passes are not interchangeable:
  //
  //  * A row where Geant4 EMITTED AN ION carries (Z, A), so the target mass comes from the
  //    port's own data/nuclei_mass_ame12.hh and every output is checked: trec, finalT, edep
  //    (which must be zero), the branch, and the ion's direction from momentum balance.
  //  * A row where it did NOT has no (Z, A) - below threshold the recoil becomes a deposit and
  //    the target is not observable from outside the model. Such a row cannot test trec: the
  //    only way to a target mass is to invert Geant4's own trec, and then recomputing trec
  //    from it would compare a number with itself. So those rows test the BRANCH, `edep = trec`
  //    and `finalT = T - trec`, which is the half the ion rows do not reach, and the inverted
  //    mass is used for nothing else.
  //
  // AND WHY THE ORACLE HAS THREE PASSES. With QBBC's real cuts the `edep = trec` arm is
  // UNREACHABLE: this process samples only angles beyond cosTetMaxNuc, so its smallest momentum
  // transfer is set by q2Max and <A^-2/3> rather than by the energy, and the smallest recoil it
  // can produce in these four materials is 0.285 MeV (calcium in bone) against a 0.07 MeV
  // proton cut - measured and printed below, not assumed. So `zerocut` (pCuts zeroed, every
  // draw emits its ion) carries the trec comparison and `highcut` (pCuts at 1e6 MeV, every draw
  // deposits) carries the other arm. The `pcut` pass is QBBC's own behaviour and is what says
  // which of the two the transport will actually see.
  {
    FILE* f = std::fopen((dir + "/coulomb_recoil.csv").c_str(), "r");
    if (f == nullptr) {
      std::printf("cannot read %s/coulomb_recoil.csv\n", dir.c_str());
      return 1;
    }
    char line[1024];
    std::fgets(line, sizeof line, f);
    Worst trec_w[kNClaims], ft_w[kNClaims], dep_w[kNClaims], dir_w[kNClaims], mass_w[kNClaims];
    long n_ion = 0, n_dep = 0, n_dep_nonzero = 0, n_branch_bad = 0, n_g4_bad = 0, n_clamp = 0;
    long n_composed = 0, n_compose_bad = 0, n_pcut_deflected = 0, n_clamp_tested = 0;
    double min_trec_pcut = 1e300, tcut_pcut = 0;
    while (std::fgets(line, sizeof line, f) != nullptr) {
      char mat[64], part[64], pass[16];
      int nsec, ionz, iona, call;
      double e, cut, tcut, mom2, cost, dx, dy, dz, ft, trec, edep, nonion, idx, idy, idz;
      const int n = std::sscanf(line,
                                "%63[^,],%63[^,],%15[^,],%lf,%d,%lf,%lf,%lf,%lf,%lf,%lf,%lf,"
                                "%lf,%lf,%lf,%lf,%d,%d,%d,%lf,%lf,%lf",
                                mat, part, pass, &e, &call, &cut, &tcut, &mom2, &cost, &dx,
                                &dy, &dz, &ft, &trec, &edep, &nonion, &nsec, &ionz, &iona,
                                &idx, &idy, &idz);
      if (n != 22) { continue; }
      const int ci = claim_of(part);
      const int mi = material_of(mat);
      if (ci < 0 || mi < 0 || kClaims[ci].low < 0) { continue; }
      const ParticleDef<real_t> pd = particle_def<real_t>(kClaims[ci].type);

      // Geant4's own row, checked for internal consistency first. If the reference's finalT is
      // not T - trec then this block is comparing against something it has mis-read, and that
      // has to fail loudly rather than be absorbed into a tolerance.
      if (std::fabs(ft + trec - e) > 1e-9 * e) { ++n_g4_bad; }
      if (nsec == 0 && trec > tcut && trec > 0) { ++n_g4_bad; }
      if (nsec > 0 && !(trec > tcut)) { ++n_g4_bad; }
      if (trec >= e) { ++n_clamp; }

      real_t tmass = 0;
      const bool have_target = (ionz > 0 && iona > 0);
      if (have_target) {
        tmass = data::nuclear_mass<real_t>(iona, ionz);
        // The mass Geant4 used, recovered from its own recoil expression - a diagnostic, so
        // that a trec failure says whether the mass table or the formula moved. Asserting on
        // it would be asserting on the formula, which is what trec already does.
        const double omc = 1.0 - cost;
        if (trec > 0 && omc > 0) {
          const double g4m = mom2 * omc / trec - (pd.mass + e) * omc;
          note(mass_w[ci], std::fabs(double(tmass) - g4m) / g4m, "Z=%d A=%d (%.10g vs %.10g)",
               ionz, iona, double(tmass), g4m);
        }
      } else {
        const double omc = 1.0 - cost;
        tmass = (trec > 0 && omc > 0)
                    ? real_t(mom2 * omc / trec - (pd.mass + e) * omc)
                    : real_t(1);   // cost == 1: trec is zero for any target
      }

      const em::CoulombFinalState<real_t> r =
          em::coulomb_recoil<real_t>(pd.mass, real_t(e), real_t(mom2), tmass, real_t(cost),
                                     real_t(tcut));
      if (r.emit_ion != (nsec > 0)) {
        ++n_branch_bad;
        if (n_branch_bad < 5) {
          std::printf("    FAIL branch: %s %s %s %.4g MeV call %d: G4 n_sec=%d, ours "
                      "emit_ion=%d (trec %.10g vs %.10g, tcut %.10g)\n", mat, part, pass, e,
                      call, nsec, int(r.emit_ion), double(r.trec), trec, tcut);
        }
      }
      note(ft_w[ci], (e > 0) ? std::fabs(double(r.final_t) - ft) / e : 0.0,
           "%s %s %s %.4g MeV call %d (%.12g vs %.12g MeV)", mat, part, pass, e, call,
           double(r.final_t), ft);
      note(dep_w[ci], (e > 0) ? std::fabs(double(r.edep) - edep) / e : 0.0,
           "%s %s %s %.4g MeV call %d (%.12g vs %.12g MeV)", mat, part, pass, e, call,
           double(r.edep), edep);
      if (have_target) {
        ++n_ion;
        if (trec > 0) {
          note(trec_w[ci], std::fabs(double(r.trec) - trec) / trec,
               "%s %s %s %.4g MeV call %d, Z=%d A=%d (%.12g vs %.12g MeV)", mat, part, pass, e,
               call, ionz, iona, double(r.trec), trec);
        }
        // The full `coulomb_sample_secondaries` on the same inputs, once per ion row. It draws
        // its own angle, so nothing here can be compared against the row - what is checked is
        // the COMPOSITION, which no other block reaches: that the ion identity is the caller's
        // (iz, ia) and not something invented, that the three energies still balance after the
        // parts are put together, and that the deposit goes to whichever of the two places the
        // branch chose. The parts themselves are what the row-by-row comparison above tests.
        {
          Philox<real_t> srng(0x51e3u + unsigned(ci) * 131u, unsigned(ionz) * 17u,
                              unsigned(call));
          const em::CoulombFinalState<real_t> fs = em::coulomb_sample_secondaries<real_t>(
              pd, kClaims[ci].type, real_t(e), mats[mi].inv_a23, ionz, iona, tmass,
              real_t(cut), real_t(kCosThetaMin), real_t(kCosThetaMax),
              Vec3<real_t>{0, 0, 1}, srng);
          ++n_composed;
          const bool ident = !fs.emit_ion || (fs.ion_z == ionz && fs.ion_a == iona);
          const double bal = std::fabs(double(fs.final_t + fs.trec) - e);
          const double dep = fs.emit_ion ? double(fs.edep)
                                         : std::fabs(double(fs.edep - fs.trec));
          const double unitv = std::fabs(double(mag(fs.ion_dir)) - 1.0);
          if (!ident || bal > 1e-12 * e || dep > 1e-12 * e || unitv > 1e-12) {
            if (n_compose_bad < 5) {
              std::printf("    FAIL composition: %s %s %.4g MeV: ion (%d,%d) vs (%d,%d), "
                          "balance %.3g, deposit %.3g, |ion_dir|-1 %.3g\n", part, mat, e,
                          fs.ion_z, fs.ion_a, ionz, iona, bal, dep, unitv);
            }
            ++n_compose_bad;
          }
        }
        // The recoil direction, from the primary's own before/after momenta. Compared as the
        // angle between the two unit vectors, so all three components count once.
        const Vec3<real_t> ours =
            em::coulomb_recoil_direction<real_t>(Vec3<real_t>{0, 0, 1},
                                                 Vec3<real_t>{real_t(dx), real_t(dy),
                                                              real_t(dz)},
                                                 real_t(mom2), pd.mass, real_t(ft));
        const double d = std::fabs(double(ours.x) - idx) + std::fabs(double(ours.y) - idy)
                         + std::fabs(double(ours.z) - idz);
        note(dir_w[ci], d, "%s %s %s %.4g MeV call %d ((%.9g,%.9g,%.9g) vs (%.9g,%.9g,%.9g))",
             mat, part, pass, e, call, double(ours.x), double(ours.y), double(ours.z), idx,
             idy, idz);
      } else {
        ++n_dep;
        if (trec > 0) { ++n_dep_nonzero; }
        // The non-ionizing route, which only the deposit arm uses: Geant4 proposes the recoil
        // as non-ionizing AND as local, so both columns must equal trec. The port carries one
        // `edep` and a comment saying which; if the two Geant4 columns ever disagreed, that
        // comment would be wrong and this is where it shows.
        if (std::fabs(nonion - trec) > 1e-12 * (trec + 1e-30)
            || std::fabs(edep - trec) > 1e-12 * (trec + 1e-30)) {
          ++n_g4_bad;
        }
      }
      // `trec = std::min(trec, kinEnergy)`, on the rows where GEANT4 took it. Those rows are
      // deposit rows, so the target mass this block inverts out of Geant4's own trec is the
      // mass that reproduces trec = T exactly - which would pass whether the port clamps or
      // not. The target can be identified without the inversion, though: only the LIGHTEST
      // element in the material can clamp, because trec exceeds T only when the target mass is
      // below the projectile's (omc*mass > targetMass at omc <= 2), and in these materials that
      // is hydrogen and nothing else. So the port is re-run on the lightest element's own mass
      // out of its own table, and it has to overshoot and then clamp: if its unclamped trec did
      // not exceed T, Geant4's clamp could not have fired either and the row is being
      // misattributed.
      if (trec >= e && !have_target) {
        int zl = 300;
        for (int k = 0; k < mats[mi].n_elements; ++k) {
          const int zz = int(mats[mi].z[k] + 0.5);
          if (zz < zl) { zl = zz; }
        }
        const int al = int(data::atomic_mass<real_t>(zl) + 0.5);
        const real_t lm = data::nuclear_mass<real_t>(al, zl);
        const double omc = 1.0 - cost;
        const double raw = mom2 * omc / (double(lm) + (pd.mass + e) * omc);
        const em::CoulombFinalState<real_t> rc = em::coulomb_recoil<real_t>(
            pd.mass, real_t(e), real_t(mom2), lm, real_t(cost), real_t(tcut));
        if (!(raw > e) || std::fabs(double(rc.trec) - e) > 1e-12 * e
            || double(rc.final_t) != 0.0 || std::fabs(double(rc.edep) - e) > 1e-12 * e) {
          std::printf("    FAIL clamp: %s %s %s %.4g MeV cost %.6g on Z=%d: unclamped %.6g, "
                      "clamped %.10g, finalT %.3g, edep %.10g\n", mat, part, pass, e, cost, zl,
                      raw, double(rc.trec), double(rc.final_t), double(rc.edep));
          ++fails;
        }
        ++n_clamp_tested;
      }
      // QBBC's own arm, measured: the smallest recoil this process can produce against the cut
      // it is compared with. This is the number behind "the local-deposit arm is unreachable".
      if (std::strcmp(pass, "pcut") == 0 && trec > 0) {
        ++n_pcut_deflected;
        if (trec < min_trec_pcut) { min_trec_pcut = trec; }
        tcut_pcut = tcut;
      }
    }
    std::fclose(f);
    std::printf("\n== SampleSecondaries, one row per call ==\n");
    std::printf("  %-12s %12s %12s %12s %12s %12s %7s\n", "particle", "trec", "finalT", "edep",
                "ion dir", "mass table", "n");
    for (int i = 0; i < kNClaims; ++i) {
      if (kClaims[i].low < 0 || ft_w[i].n == 0) { continue; }
      std::printf("  %-12s %11.3e %11.3e %11.3e %11.3e %11.3e %7d\n", kClaims[i].name,
                  trec_w[i].dev, ft_w[i].dev, dep_w[i].dev, dir_w[i].dev, mass_w[i].dev,
                  ft_w[i].n);
      // 1e-9 on trec, and the limit is the nuclear mass and not the arithmetic: the recoil
      // divides by `targetMass + (mass + T)*(1 - cost)`, and targetMass comes from this port's
      // data/nuclei_mass_ame12.hh against Geant4's G4NucleiProperties - the same 1e-9 the
      // model's MinPrimaryEnergy is held to in block 2, for the same reason. finalT, edep and
      // the direction are pure arithmetic on the row and are held to 1e-12.
      if (trec_w[i].dev > 1e-9) {
        std::printf("    FAIL: %s trec, %s\n", kClaims[i].name, trec_w[i].where);
        ++fails;
      }
      if (ft_w[i].dev > 1e-12) {
        std::printf("    FAIL: %s finalT, %s\n", kClaims[i].name, ft_w[i].where);
        ++fails;
      }
      if (dep_w[i].dev > 1e-12) {
        std::printf("    FAIL: %s edep, %s\n", kClaims[i].name, dep_w[i].where);
        ++fails;
      }
      if (dir_w[i].dev > 1e-9) {
        std::printf("    FAIL: %s recoil direction, %s\n", kClaims[i].name, dir_w[i].where);
        ++fails;
      }
    }
    std::printf("  %ld rows emitted an ion, %ld deposited locally (%ld of them a nonzero "
                "recoil), %ld branch disagreements, %ld reference inconsistencies, %ld "
                "trec clamped to T\n", n_ion, n_dep, n_dep_nonzero, n_branch_bad, n_g4_bad,
                n_clamp);
    std::printf("  %ld coulomb_sample_secondaries compositions, %ld bad\n", n_composed,
                n_compose_bad);
    fails += int(n_branch_bad > 0) + int(n_g4_bad > 0) + int(n_compose_bad > 0);
    if (n_composed == 0) {
      std::printf("  FAIL: coulomb_sample_secondaries was never called\n");
      ++fails;
    }
    // Both branches have to have run, or the `if(trec > tcut)` this block is about was never
    // decided: with only ion rows the local-deposit arm is dead code, and with only deposit
    // rows nothing tested the trec formula at all.
    if (n_ion == 0 || n_dep_nonzero == 0) {
      std::printf("  FAIL: the recoil threshold branch was not exercised both ways (%ld ion, "
                  "%ld nonzero deposit)\n", n_ion, n_dep_nonzero);
      ++fails;
    }
    // The finding, as a number rather than as a claim in a comment: with QBBC's real cuts every
    // deflected draw is above threshold, because the smallest recoil this process can produce
    // is several times the proton cut. If a future release, a heavier material or a larger
    // range cut ever brought the two together, this line is where it would show.
    if (n_pcut_deflected > 0) {
      std::printf("  QBBC cuts: %ld deflected draws, smallest recoil %.4g MeV against a %.4g "
                  "MeV threshold (ratio %.1f),\n             so the local-deposit arm is "
                  "unreachable in option0 and the highcut pass is what measures it\n",
                  n_pcut_deflected, min_trec_pcut, tcut_pcut, min_trec_pcut / tcut_pcut);
    } else {
      std::printf("  FAIL: the pcut pass produced no deflected draw at all\n");
      ++fails;
    }
    // Geant4's own "the check likely not needed" does fire, and only for an ANTIPROTON: the
    // clamp needs a target lighter than the projectile, and the proton is kept from
    // backscattering off hydrogen by SampleSecondaries' own `1 == iz && particle == theProton`
    // exception, which forces its cosThetaMax to 0. The pbar is not in that test, so it reaches
    // 180 degrees off a target of its own mass and takes the whole of its kinetic energy.
    std::printf("  min(trec, T) clamp: fired in Geant4 on %ld rows, %ld of them checked "
                "against the port\n", n_clamp, n_clamp_tested);
    if (n_clamp > 0 && n_clamp_tested == 0) {
      std::printf("  FAIL: the clamp fired in the reference and no row tested it\n");
      ++fails;
    }
    if (n_clamp == 0) {
      std::printf("  note: no row reached trec >= T, so the min(trec, kinEnergy) clamp is "
                  "transcribed but not exercised by this oracle\n");
    }
  }

  std::printf("\n%s (%d failures)\n", fails ? "FAILED" : "PASSED", fails);
  return fails ? 1 : 0;
}
