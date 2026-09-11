// G4UrbanMscModel's stepping half for an ion, against ref/oracle/ion_msc_*.csv (P14b).
//
// WHAT IS BEING TESTED AND WHY IT NEEDED THREE FILES. `urban_msc.csv` and
// tests/test_urban_general.cu already pin `ComputeCrossSectionPerAtom` for eight species over
// Z 1..92 to 6.7e-16. What had no oracle at all is everything the model does with that cross
// section once a track has it, and for the five species QBBC scatters by Urban - alpha, He3,
// deuteron, triton and GenericIon - the port did not do it: `em/urban_msc.cuh`'s stepping half
// was the electron's and `step_hadron` ran WentzelVI in its place.
//
// The three files split the comparison the way the failures split:
//
//   ion_msc_step.csv    deterministic. The transport mean free path, the step limit where it
//                       does not randomise, and both directions of the true<->geometric path
//                       conversion. Exact.
//   ion_msc_limit.csv   the randomised limit, as 20,000 draws per cell, plus two rows per cell
//                       either side of the doverrb early-return threshold.
//   ion_msc_sample.csv  the angular distribution, 400,000 SampleScattering draws per cell.
//
// RANGE AND LAMBDA ARE HANDED OVER, NOT RECOMPUTED, in sections 3 onward. Both are columns, so
// a disagreement in the path conversion cannot be blamed on the port's range table - which
// `hadron_tables.csv` and tests/test_hadron_range.cu already check - and section 1 checks the
// mean free path against the same column separately. That is also what lets the high-Z cell
// participate: `CustomDerivedI` (30% lead by mass, Zeff 50.5) exists in the oracle dumper's
// geometry and not in `data::build_b1_materials`, and everything from section 3 on needs only
// its Zeff and its radiation length, both columns. Section 1 skips it and says so.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "core/particle.cuh"
#include "core/rng.cuh"
#include "data/materials.cuh"
#include "physics/em/hadron_range.cuh"
#include "physics/em/urban_msc.cuh"

using namespace g4gpu;
using real_t = double;

namespace {

/// One row of any of the three files, as a name -> value map. The files have 24 to 44 columns
/// and three different layouts; parsing by header name rather than by position means a column
/// added to the dumper does not silently shift what this test reads.
struct Table {
  std::vector<std::string> cols;
  std::vector<std::vector<std::string>> rows;
  int index_of(const char* name) const {
    for (std::size_t i = 0; i < cols.size(); ++i) {
      if (cols[i] == name) { return static_cast<int>(i); }
    }
    return -1;
  }
  double num(std::size_t r, const char* name) const {
    const int i = index_of(name);
    return (i >= 0) ? std::atof(rows[r][i].c_str()) : 0.0;
  }
  const std::string& str(std::size_t r, const char* name) const {
    static const std::string empty;
    const int i = index_of(name);
    return (i >= 0) ? rows[r][i] : empty;
  }
};

std::vector<std::string> split(const std::string& s) {
  std::vector<std::string> out;
  std::size_t a = 0;
  while (true) {
    const std::size_t b = s.find(',', a);
    out.push_back(s.substr(a, (b == std::string::npos) ? std::string::npos : b - a));
    if (b == std::string::npos) { break; }
    a = b + 1;
  }
  return out;
}

bool load(const std::string& path, Table& t) {
  FILE* f = std::fopen(path.c_str(), "r");
  if (f == nullptr) { return false; }
  std::vector<char> buf(1 << 16);
  if (std::fgets(buf.data(), static_cast<int>(buf.size()), f) == nullptr) {
    std::fclose(f);
    return false;
  }
  {
    std::string line(buf.data());
    while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) { line.pop_back(); }
    t.cols = split(line);
  }
  while (std::fgets(buf.data(), static_cast<int>(buf.size()), f) != nullptr) {
    std::string line(buf.data());
    while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) { line.pop_back(); }
    if (line.empty()) { continue; }
    t.rows.push_back(split(line));
  }
  std::fclose(f);
  return !t.rows.empty();
}

/// The oracle's material names against the port's four. `CustomDerivedI` deliberately has no
/// entry: see the header.
int port_material(const std::string& name) {
  if (name == "G4_AIR") { return data::kAir; }
  if (name == "G4_WATER") { return data::kWater; }
  if (name == "G4_A-150_TISSUE") { return data::kA150Tissue; }
  if (name == "G4_BONE_COMPACT_ICRU") { return data::kBoneCompact; }
  return -1;
}

/// A material carrying only what Urban's stepping half reads: the radiation length. The
/// coefficients come from `urban_coeffs(zeff)` with the oracle's own Zeff, and nothing in
/// sections 3 onward touches the element list, so this is the whole of the material as far as
/// those sections are concerned - and it makes them independent of `data::build_b1_materials`.
data::Material<real_t> radlen_only(real_t radlen) {
  data::Material<real_t> m{};
  m.radiation_length = radlen;
  m.n_elements = 0;
  return m;
}

data::Material<real_t> coeff_material(real_t zeff, real_t radlen) {
  data::Material<real_t> m = radlen_only(radlen);
  m.z_eff = zeff;
  return m;
}

struct Worst {
  double v = 0;
  std::string where;
  int n = 0;
  void add(double err, const std::string& w) {
    ++n;
    if (err > v) {
      v = err;
      where = w;
    }
  }
};

double rel(double got, double want) {
  const double d = std::fabs(got - want);
  const double s = std::fabs(want);
  return (s > 0) ? d / s : d;
}

/// The species' mass and bare charge, from the port's own two constructors - which is what
/// `step_hadron` passes. `charge` is the PDG charge and NOT the effective charge:
/// G4UrbanMscModel::SetParticle reads `p->GetPDGCharge()/eplus` once and never refreshes it.
bool port_particle(const std::string& name, int z, int a, real_t& mass, real_t& charge) {
  if (name == "alpha") {
    const auto d = particle_def<real_t>(ParticleType::kAlpha);
    mass = d.mass;
    charge = d.charge;
    return true;
  }
  if (name == "He3") {
    const auto d = particle_def<real_t>(ParticleType::kHe3);
    mass = d.mass;
    charge = d.charge;
    return true;
  }
  const auto h = em::stepped_ion<real_t>(z, a);
  mass = h.def.mass;
  charge = h.def.charge;
  return true;
}

constexpr int kBins = 24;

}  // namespace

int main() {
  const char* env = std::getenv("G4GPU_ORACLE");
  const std::string dir = (env != nullptr) ? env : "ref/oracle";
  Table step, limit, samp;
  if (!load(dir + "/ion_msc_step.csv", step) || !load(dir + "/ion_msc_limit.csv", limit)
      || !load(dir + "/ion_msc_sample.csv", samp)) {
    std::printf("cannot read %s/ion_msc_*.csv - run ref/oracle/run.bat tables first\n",
                dir.c_str());
    return 1;
  }
  data::Material<real_t> mats[data::kNumMaterials];
  data::build_b1_materials<real_t>(mats);
  int fails = 0;

  // Zeff and the radiation length per material name, from the deterministic table, so the
  // other two files can be read without their own copies of them.
  std::map<std::string, std::pair<double, double>> matinfo;
  for (std::size_t r = 0; r < step.rows.size(); ++r) {
    matinfo[step.str(r, "material")] = {step.num(r, "zeff"), step.num(r, "radlen_mm")};
  }

  // ---------------------------------------------------------------- 1. the configuration
  //
  // Three numbers, on every row, and getting any of them from the model instead of from the
  // particle is the mistake this package exists to undo. `step_limit_type` 0 is fMinimal
  // (G4MscStepLimitType.hh), `facrange` is MscMuHadRangeFactor and `lat_disp` is
  // MuHadLateralDisplacement.
  {
    int bad = 0;
    for (std::size_t r = 0; r < step.rows.size(); ++r) {
      if (step.num(r, "step_limit_type") != 0) { ++bad; }
      if (step.num(r, "facrange") != 0.2) { ++bad; }
      if (step.num(r, "lat_disp") != 0) { ++bad; }
    }
    std::printf("== 1. the ion's msc configuration, on all %d rows ==\n",
                static_cast<int>(step.rows.size()));
    std::printf("  step limit type fMinimal(0), facrange 0.2, lateral displacement off: %s\n",
                bad ? "ORACLE DISAGREES" : "yes");
    if (bad) { ++fails; }
    // And the port's own constants say the same, in one place each.
    if (em::kFacRangeMuHad<real_t>() != real_t(0.2)) {
      std::printf("  FAIL: em::kFacRangeMuHad is %g, not 0.2\n",
                  double(em::kFacRangeMuHad<real_t>()));
      ++fails;
    }
    if (em::kHadronLateralDisplacement<real_t>()) {
      std::printf("  FAIL: em::kHadronLateralDisplacement is true\n");
      ++fails;
    }
  }

  // ---------------------------------------------------------------- 2. the transport mfp
  //
  // `urban_heavy_lambda` against `1/CrossSectionPerVolume`, which is what
  // G4VMscModel::GetTransportMeanFreePath returns for a particle with no cross-section table -
  // and no ion has one (G4VMscModel.cc:94). Two energies per row: the pre-step one and the one
  // at the residual range the path conversion needs.
  {
    Worst w;
    int skipped = 0;
    for (std::size_t r = 0; r < step.rows.size(); ++r) {
      const int mi = port_material(step.str(r, "material"));
      if (mi < 0) { ++skipped; continue; }
      real_t mass = 0, charge = 0;
      port_particle(step.str(r, "particle"), int(step.num(r, "Z")), int(step.num(r, "A")), mass,
                    charge);
      const std::string where = step.str(r, "particle") + " " + step.str(r, "material");
      const double l0 = em::urban_heavy_lambda<real_t>(mats[mi], step.num(r, "energy_MeV"), mass,
                                                       charge);
      w.add(rel(l0, step.num(r, "lambda0_mm")), where + " lambda0");
      const double lr = em::urban_heavy_lambda<real_t>(mats[mi], step.num(r, "e_rfin_MeV"), mass,
                                                       charge);
      w.add(rel(lr, step.num(r, "lambda_rfin_mm")), where + " lambda_rfin");
    }
    std::printf("\n== 2. transport mean free path, %d comparisons ==\n", w.n);
    std::printf("  worst relative error %.3g  (%s)\n", w.v, w.where.c_str());
    std::printf("  %d rows skipped: CustomDerivedI is in the dumper's geometry and not in "
                "data::build_b1_materials\n", skipped);
    if (!(w.v < 1e-13)) { std::printf("  FAIL: above 1e-13\n"); ++fails; }
  }

  // ---------------------------------------------------------------- 3. doverra / doverrb
  //
  // TWO TRANSCRIPTIONS OF ONE FORMULA, and that is why this is not the test that matters for
  // doverrb. `InitialiseModelCache` has no accessor, so the dumper computes the pair from Zeff
  // and so does `urban_coeffs`; agreement here says the two agree and not that either is right
  // (docs/RISK.md V37). What measures doverrb is section 5, where the oracle brackets the
  // `distance < presafety` threshold at 0.99 and 1.01 and the branch has to flip in the same
  // place. This section is here so that a failure there localises.
  {
    Worst w;
    for (const auto& kv : matinfo) {
      const auto c = em::urban_coeffs(coeff_material(kv.second.first, kv.second.second));
      for (std::size_t r = 0; r < step.rows.size(); ++r) {
        if (step.str(r, "material") != kv.first) { continue; }
        w.add(rel(c.doverra, step.num(r, "doverra")), kv.first + " doverra");
        w.add(rel(c.doverrb, step.num(r, "doverrb")), kv.first + " doverrb");
        break;
      }
    }
    std::printf("\n== 3. the Zeff distance coefficients, %d comparisons ==\n", w.n);
    std::printf("  worst relative error %.3g  (%s)\n", w.v, w.where.c_str());
    if (!(w.v < 1e-14)) { std::printf("  FAIL: above 1e-14\n"); ++fails; }
  }

  // ---------------------------------------------------------------- 4. the path conversion
  //
  // ComputeGeomPathLength forward and ComputeTrueStepLength back, on the deterministic rows -
  // the safety there is twice the doverrb distance, so the early return fires, the limit draws
  // no random number and `t_out` is `min(t_in, range)` exactly. The step fractions walk the
  // four branches: below tlimitminfix2, below dtrl, the `kinetic < mass` branch (every ion at
  // every energy here) and `t == range`.
  {
    Worst wg, wt, wl;
    for (std::size_t r = 0; r < step.rows.size(); ++r) {
      const auto& mn = step.str(r, "material");
      const auto info = matinfo[mn];
      const auto c = em::urban_coeffs(coeff_material(info.first, info.second));
      const data::Material<real_t> m = radlen_only(info.second);
      real_t mass = 0, charge = 0;
      port_particle(step.str(r, "particle"), int(step.num(r, "Z")), int(step.num(r, "A")), mass,
                    charge);
      const double range = step.num(r, "range_mm");
      const double lam0 = step.num(r, "lambda0_mm");
      const double lamr = step.num(r, "lambda_rfin_mm");
      const double e = step.num(r, "energy_MeV");
      const double t_in = step.num(r, "t_in_mm");
      const double safety = step.num(r, "safety_mm");
      const std::string where = step.str(r, "particle") + " " + mn + " E="
                                + std::to_string(e) + " t/R="
                                + std::to_string(t_in / (range > 0 ? range : 1));

      real_t tlimit = em::kMscAtBoundary<real_t>();
      Philox<real_t> rng(1u, 1u);
      const double t_out = em::urban_step_limit_heavy<real_t, Philox<real_t>>(
          c, lam0, em::kFacRangeMuHad<real_t>(), e, mass, range, safety, t_in, rng, tlimit);
      wl.add(rel(t_out, step.num(r, "t_out_mm")), where + " t_out");

      em::MscStep<real_t> st{};
      const double g = em::urban_geom_path<real_t>(t_out, lam0, range, lamr, e, mass, st);
      wg.add(rel(g, step.num(r, "g_out_mm")), where + " g_out");
      const double gf = step.num(r, "g_frac");
      const double tb = em::urban_true_path<real_t>(gf * step.num(r, "g_out_mm"), t_out, st);
      wt.add(rel(tb, step.num(r, "t_back_mm")), where + " t_back");
    }
    std::printf("\n== 4. step limit (early-return rows) and the path conversion, %d rows ==\n",
                static_cast<int>(step.rows.size()));
    std::printf("  t_out  worst %.3g  (%s)\n", wl.v, wl.where.c_str());
    std::printf("  g(t)   worst %.3g  (%s)\n", wg.v, wg.where.c_str());
    std::printf("  t(g')  worst %.3g  (%s)\n", wt.v, wt.where.c_str());
    if (!(wl.v < 1e-15)) { std::printf("  FAIL: t_out above 1e-15\n"); ++fails; }
    if (!(wg.v < 1e-12)) { std::printf("  FAIL: g(t) above 1e-12\n"); ++fails; }
    if (!(wt.v < 1e-12)) { std::printf("  FAIL: t(g') above 1e-12\n"); ++fails; }
  }

  // ---------------------------------------------------------------- 5. the randomised limit
  //
  // Three safeties per cell. At 1.01 of `range*doverrb` the `distance < presafety` test is
  // true, the function returns before the branch, and the returned length is `min(t_in, range)`
  // with no random number drawn - so `n_limited` must be 0. At 0.99 it is false and the branch
  // runs. Whether the branch then LIMITS anything is a second question with its own answer:
  // `tlimit = facrange*max(range, lambda0)` beats the range in 436 of 450 rows, because Urban's
  // transport mfp for an ion runs from 0.64 to 40,000 times its range, so multiple scattering
  // does not shorten an ion's step at all except where that ratio is under five.
  //
  // The 14 rows where it does are compared as moments. `Randomizetlimit` is
  // `max(Gauss(tlimit, 0.1*(tlimit - tlimitmin)), tlimitmin)` clamped to the proposed step, so
  // the mean is tlimit and the sd a tenth of it, and 20,000 draws pin the mean to 7e-4 of the
  // sd. The port draws its Gaussian from Box-Muller where Geant4 uses G4RandGauss, so these
  // two samples are not the same numbers and this is a distribution comparison by construction.
  {
    int bad_branch = 0, compared = 0, flat = 0;
    double worst_mean = 0, worst_sd = 0;
    std::string wm, ws;
    std::printf("\n== 5. the randomised step limit, %d rows ==\n",
                static_cast<int>(limit.rows.size()));
    for (std::size_t r = 0; r < limit.rows.size(); ++r) {
      const auto& mn = limit.str(r, "material");
      const auto info = matinfo[mn];
      const auto c = em::urban_coeffs(coeff_material(info.first, info.second));
      real_t mass = 0, charge = 0;
      port_particle(limit.str(r, "particle"), int(limit.num(r, "Z")), int(limit.num(r, "A")),
                    mass, charge);
      const double range = limit.num(r, "range_mm");
      const double lam0 = limit.num(r, "lambda0_mm");
      const double e = limit.num(r, "energy_MeV");
      const double safety = limit.num(r, "safety_mm");
      const long nlim = long(limit.num(r, "n_limited"));
      const std::string where = limit.str(r, "particle") + " " + mn + " E="
                                + std::to_string(e) + " sf="
                                + std::to_string(limit.num(r, "safety_over_distance"));

      const int n_oracle = int(limit.num(r, "n"));
      const bool moments_row = (limit.num(r, "safety_over_distance") == 0.0);
      const int n = (nlim > 0 && moments_row) ? 20000 : 64;
      Philox<real_t> rng(0x105c1f4u, unsigned(r) * 7919u + 13u);
      double mean = 0, m2 = 0;
      long n_limited = 0;
      for (int k = 0; k < n; ++k) {
        real_t tlimit = em::kMscAtBoundary<real_t>();
        const double t = em::urban_step_limit_heavy<real_t, Philox<real_t>>(
            c, lam0, em::kFacRangeMuHad<real_t>(), e, mass, range, safety, range, rng, tlimit);
        if (t != range) { ++n_limited; }
        const double d = t - mean;
        mean += d / (k + 1);
        m2 += d * (t - mean);
      }
      // The BRANCH, on every row: limited or not must agree, and "not" must be exactly the
      // range. The oracle's n_limited is out of its own n, so compare the predicates.
      const bool g4_limited = (nlim > 0);
      const bool pt_limited = (n_limited > 0);
      if (g4_limited != pt_limited) {
        ++bad_branch;
        std::printf("  FAIL branch: %s  Geant4 %s, port %s\n", where.c_str(),
                    g4_limited ? "limited" : "unlimited", pt_limited ? "limited" : "unlimited");
      }
      if (!g4_limited) {
        ++flat;
        if (mean != range) {
          ++bad_branch;
          std::printf("  FAIL: %s unlimited row returned %.17g, range %.17g\n", where.c_str(),
                      mean, range);
        }
        continue;
      }
      // The moments are compared only on the safety-zero rows, which the dumper runs at 20,000
      // draws. The two threshold rows are 200 draws each because what they measure is which
      // BRANCH ran - checked above, on every row - and comparing a 200-draw standard deviation
      // at a few per cent would fail on its own sampling error.
      if (!moments_row) { continue; }
      ++compared;
      const double sd = std::sqrt(m2 / n);
      const double g4mean = limit.num(r, "mean_t_mm"), g4sd = limit.num(r, "sd_t_mm");
      // Standard error on the difference of two independent means, EACH WITH ITS OWN n.
      const double se = std::sqrt(sd * sd / n + g4sd * g4sd / n_oracle);
      const double nsig = (se > 0) ? std::fabs(mean - g4mean) / se : 0.0;
      // And the standard deviations in units of their own standard error, 1/sqrt(2n) apiece.
      const double sdse = g4sd * std::sqrt(0.5 / n + 0.5 / n_oracle);
      const double sdsig = (sdse > 0) ? std::fabs(sd - g4sd) / sdse : 0.0;
      if (nsig > worst_mean) { worst_mean = nsig; wm = where; }
      if (sdsig > worst_sd) { worst_sd = sdsig; ws = where; }
    }
    std::printf("  %d rows the limit does not touch (exactly the range on both sides), "
                "%d compared as moments\n", flat, compared);
    std::printf("  worst mean %.2f sigma (%s)\n", worst_mean, wm.c_str());
    std::printf("  worst sd   %.2f sigma (%s)\n", worst_sd, ws.c_str());
    if (bad_branch) { ++fails; }
    if (worst_mean > 5.0) { std::printf("  FAIL: mean above 5 sigma\n"); ++fails; }
    if (worst_sd > 5.0) { std::printf("  FAIL: sd above 5 sigma\n"); ++fails; }
  }

  // ---------------------------------------------------------------- 6. the angular distribution
  //
  // `urban_sample_scattering` with the ion's mass and charge, against 400,000 of Geant4's own
  // SampleScattering draws per cell. The scatter energy and the mean free path at it are
  // columns, so the two sides sample the same distribution rather than two nearby ones - see
  // the dumper's header on what that does and does not measure.
  //
  // THE DISPLACEMENT IS PART OF THE COMPARISON. Every oracle row has `mean_disp_mm` exactly
  // zero, which is `MuHadLateralDisplacement = false` as a measurement. The port must produce
  // zero too, and for the same reason rather than by cancellation: it is asked for the
  // displacement with `lat_displacement` false, which is also what keeps it from drawing the
  // three extra uniforms the lepton path draws there.
  {
    int compared = 0, bad = 0;
    double worst_sig = 0, worst_chi2 = 0;
    std::string wsig, wchi;
    for (std::size_t r = 0; r < samp.rows.size(); ++r) {
      const auto& mn = samp.str(r, "material");
      const auto info = matinfo[mn];
      const auto c = em::urban_coeffs(coeff_material(info.first, info.second));
      const data::Material<real_t> m = radlen_only(info.second);
      real_t mass = 0, charge = 0;
      port_particle(samp.str(r, "particle"), int(samp.num(r, "Z")), int(samp.num(r, "A")), mass,
                    charge);
      const double lam0 = samp.num(r, "lambda0_mm");
      const double t = samp.num(r, "t_mm"), g = samp.num(r, "g_mm");
      const double e_pre = samp.num(r, "energy_MeV");
      const double e_scat = samp.num(r, "scatter_energy_MeV");
      const double lam_scat = samp.num(r, "lambda_scat_mm");
      const int n = int(samp.num(r, "n"));
      const std::string where = samp.str(r, "particle") + " " + mn + " E="
                                + std::to_string(e_pre) + " t/R="
                                + std::to_string(samp.num(r, "t_frac"));

      Philox<real_t> rng(0x105a3b9u, unsigned(r) * 6151u + 3u);
      double ad = 0;
      double om = 0, om2 = 0;  // Welford on (1 - cos), for the reason the oracle uses it
      long h[kBins] = {0};
      long nsc = 0;
      const double lo = samp.num(r, "hist_lo"), hi = samp.num(r, "hist_hi");
      for (int k = 0; k < n; ++k) {
        const auto out = em::urban_sample_scattering<real_t, Philox<real_t>>(
            m, c, lam0, Vec3<real_t>{0, 0, 1}, t, g, e_scat, e_pre,
            em::kHadronLateralDisplacement<real_t>(), mass, charge, false,
            em::kTlimitMinMinimal<real_t>(), rng, lam_scat);
        const double cost = out.dir.z;
        const double d = (1.0 - cost) - om;
        om += d / (k + 1);
        om2 += d * ((1.0 - cost) - om);
        ad += std::sqrt(out.displacement.x * out.displacement.x
                        + out.displacement.y * out.displacement.y
                        + out.displacement.z * out.displacement.z);
        if (cost < 1.0) {
          ++nsc;
          int b = int(kBins * (std::log10(1.0 - cost) - lo) / (hi - lo));
          if (b < 0) { b = 0; }
          if (b >= kBins) { b = kBins - 1; }
          ++h[b];
        }
      }
      if (ad != 0.0) {
        std::printf("  FAIL: %s displaced by %.3g mm; the oracle's is exactly 0\n",
                    where.c_str(), ad / n);
        ++bad;
      }
      // <1 - cos> and its standard error. Using 1-cos rather than cos is not cosmetic: at
      // small tau cos sits at 1 - 1e-6 and the mean of cos carries no significant digits of
      // the scattering at all, which is exactly the regime an ion's steps live in.
      const double sd = std::sqrt(om2 / n);
      const double got = om, want = samp.num(r, "mean_one_minus_cost");
      // `sd_one_minus_cost` is a column and is NOT reconstructed from mean_cost2. At the tau an
      // ion steps at, <cos^2> is 1 - 2e-6 and var(1-cos) is 1e-14 of it, so the subtraction
      // keeps no significant digits and the standard error comes out a hundred times too
      // small - which reports an agreeing distribution at 145 sigma. Welford on both sides.
      const double g4sd = samp.num(r, "sd_one_minus_cost");
      const double se = std::sqrt(sd * sd / n + g4sd * g4sd / n);
      const double nsig = (se > 0) ? std::fabs(got - want) / se : 0.0;
      if (nsig > worst_sig) { worst_sig = nsig; wsig = where; }
      // chi^2 per populated bin, over bins with at least 20 counts on both sides.
      double chi2 = 0;
      int nb = 0;
      const int i0 = samp.index_of("h0");
      for (int b = 0; b < kBins; ++b) {
        const double o = std::atof(samp.rows[r][i0 + b].c_str());
        const double p = double(h[b]);
        if (o < 20 || p < 20) { continue; }
        chi2 += (o - p) * (o - p) / (o + p);
        ++nb;
      }
      if (nb > 0) {
        chi2 /= nb;
        if (chi2 > worst_chi2) { worst_chi2 = chi2; wchi = where; }
      }
      ++compared;
      (void)nsc;
    }
    std::printf("\n== 6. the angular distribution, %d cells x 400,000 draws ==\n", compared);
    std::printf("  worst <1-cos> deviation %.2f sigma (%s)\n", worst_sig, wsig.c_str());
    std::printf("  worst chi2/bin %.3g (%s)\n", worst_chi2, wchi.c_str());
    std::printf("  lateral displacement zero on every row, both sides\n");
    if (bad) { ++fails; }
    // 300 cells, so the largest of 300 standard normals sits near 3.2 sigma by itself.
    if (worst_sig > 5.0) { std::printf("  FAIL: above 5 sigma\n"); ++fails; }
    if (worst_chi2 > 4.0) { std::printf("  FAIL: chi2/bin above 4\n"); ++fails; }
  }

  // ---------------------------------------------------------------- 7. what is NOT oracled
  //
  // Two states of the fMinimal branch cannot be reached from a geometry boundary, and every
  // oracle row is at one (the dumper's header says why: the non-boundary safety goes through
  // G4Navigator::ComputeSafety, which relocates within whatever volume the navigator was last
  // left in). They are asserted here against the six lines of G4UrbanMscModel.cc:655-666:
  //
  //     else {
  //       if (stepStatus == fGeomBoundary) {
  //         tlimit = (currentRange > lambda0) ? facrange*currentRange : facrange*lambda0;
  //         tlimit = std::max(tlimit, tlimitmin);
  //       }
  //       tPathLength = (tlimit < tPathLength) ? std::min(tPathLength, Randomizetlimit())
  //                                            : tPathLength;
  //     }
  //
  // There is no `firstStep` in it, and `StartTracking` sets `tlimit = geombig` (1e50 mm). So
  // the first step of a track is never limited by msc, and after one boundary the tlimit
  // computed there is held for every step until the next one - at the energy of that boundary,
  // not of the step it limits.
  {
    const auto info = matinfo["G4_WATER"];
    const auto c = em::urban_coeffs(coeff_material(info.first, info.second));
    real_t mass = 0, charge = 0;
    port_particle("Ca40", 20, 40, mass, charge);
    // The one cell where the limit bites, so "unlimited" is a distinguishable answer.
    double range = 0, lam0 = 0, e = 0;
    for (std::size_t r = 0; r < limit.rows.size(); ++r) {
      if (limit.str(r, "particle") == "Ca40" && limit.str(r, "material") == "G4_WATER"
          && limit.num(r, "n_limited") > 0) {
        range = limit.num(r, "range_mm");
        lam0 = limit.num(r, "lambda0_mm");
        e = limit.num(r, "energy_MeV");
        break;
      }
    }
    std::printf("\n== 7. the two fMinimal states no boundary row can reach ==\n");
    int bad = 0;
    if (!(range > 0)) {
      std::printf("  FAIL: no limited Ca40/water row in the oracle to test against\n");
      ++fails;
    } else {
      Philox<real_t> rng(7u, 7u);
      // (a) first step of a track: the seeded state, which is geombig.
      real_t tl = real_t(0);
      const double t1 = em::urban_step_limit_heavy<real_t, Philox<real_t>>(
          c, lam0, em::kFacRangeMuHad<real_t>(), e, mass, range, real_t(0), range, rng, tl);
      if (t1 != range) {
        std::printf("  FAIL: first step limited to %.17g of a %.17g mm range\n", t1, range);
        ++bad;
      }
      if (tl != real_t(0)) {
        std::printf("  FAIL: the first step wrote the tlimit state (%g)\n", double(tl));
        ++bad;
      }
      // (b) a boundary refreshes it, and the value is facrange*max(range, lambda0).
      tl = em::kMscAtBoundary<real_t>();
      const double t2 = em::urban_step_limit_heavy<real_t, Philox<real_t>>(
          c, lam0, em::kFacRangeMuHad<real_t>(), e, mass, range, real_t(0), range, rng, tl);
      const double want = 0.2 * ((range > lam0) ? range : lam0);
      if (rel(double(tl), want) > 1e-15) {
        std::printf("  FAIL: boundary tlimit %.17g, expected facrange*max(R,lambda) %.17g\n",
                    double(tl), want);
        ++bad;
      }
      if (!(t2 < range)) {
        std::printf("  FAIL: the boundary step was not limited (%.17g of %.17g)\n", t2, range);
        ++bad;
      }
      // (c) and it is FROZEN: a later step neither recomputes nor forgets it, even with a
      // range and a mean free path from a different energy.
      const real_t held = tl;
      const double t3 = em::urban_step_limit_heavy<real_t, Philox<real_t>>(
          c, lam0 * real_t(1000), em::kFacRangeMuHad<real_t>(), e, mass, range * real_t(0.5),
          real_t(0), range * real_t(0.5), rng, tl);
      if (tl != held) {
        std::printf("  FAIL: a non-boundary step changed tlimit from %.17g to %.17g\n",
                    double(held), double(tl));
        ++bad;
      }
      (void)t3;
      std::printf("  first step unlimited, boundary sets facrange*max(range, lambda0), "
                  "later steps hold it: %s\n", bad ? "NO" : "yes");
    }
    // (d) the extremesmallstep boundary is continuous, which is the one falsifiable statement
    // available about a branch no oracle row reaches: at t == tsmall the scale factor
    // sqrt(t/tsmall) is 1 and log(tsmall/lambda0) is log(tau), so the two paths must agree
    // BIT FOR BIT there, and just below it they must not.
    {
      const data::Material<real_t> m = radlen_only(info.second);
      const real_t ts = em::kTlimitMinMinimal<real_t>();
      Philox<real_t> r1(11u, 11u), r2(11u, 11u), r3(11u, 11u), r4(11u, 11u);
      const real_t at_on = em::urban_sample_cos_theta<real_t, Philox<real_t>>(
          m, c, ts, e, e, lam0, mass, charge, false, ts, r1);
      const real_t at_off = em::urban_sample_cos_theta<real_t, Philox<real_t>>(
          m, c, ts, e, e, lam0, mass, charge, false, real_t(0), r2);
      const real_t below_on = em::urban_sample_cos_theta<real_t, Philox<real_t>>(
          m, c, real_t(0.3) * ts, e, e, lam0, mass, charge, false, ts, r3);
      const real_t below_off = em::urban_sample_cos_theta<real_t, Philox<real_t>>(
          m, c, real_t(0.3) * ts, e, e, lam0, mass, charge, false, real_t(0), r4);
      const bool ok = (at_on == at_off) && (below_on != below_off);
      std::printf("  extremesmallstep is continuous at t == tsmall and active below it: %s\n",
                  ok ? "yes" : "NO");
      if (!ok) {
        std::printf("    at tsmall %.17g vs %.17g; below %.17g vs %.17g\n", double(at_on),
                    double(at_off), double(below_on), double(below_off));
        ++bad;
      }
      // AND THE SECOND HALF OF THAT BRANCH, which the continuity check above does NOT reach.
      //
      // `u` is that half: below tsmall it comes from log(tsmall/lambda0) rather than log(tau),
      // and it feeds only the tail parameter
      //
      //     xsi = coeffc1 + u*(coeffc2 + coeffc3*u) + coeffc4*log(lambdaeff/radlen)
      //     xsi = max(xsi, 1.9)
      //
      // Since `lambdaeff` is `trueStepLength/tau` and tau is `trueStepLength/lambda0`,
      // lambdaeff is lambda0 identically - so with the branch on, EVERY term of xsi is
      // independent of the step length and two sub-tsmall steps must give a bit-identical xsi.
      // With it off, u varies as t^(1/6) and they must differ. The clamp erases the difference
      // wherever xsi lands under 1.9, which it does at low lambda0, so only the cells where it
      // is off the floor can say anything - and the count of those is printed, because a test
      // that swept only clamped cells would pass with this half deleted.
      int off_floor = 0, u_bad = 0, u_live = 0;
      for (std::size_t r = 0; r < step.rows.size(); ++r) {
        const auto ci = matinfo[step.str(r, "material")];
        const auto cc = em::urban_coeffs(coeff_material(ci.first, ci.second));
        const data::Material<real_t> mm2 = radlen_only(ci.second);
        real_t m2 = 0, q2 = 0;
        port_particle(step.str(r, "particle"), int(step.num(r, "Z")), int(step.num(r, "A")), m2,
                      q2);
        const real_t l0 = step.num(r, "lambda0_mm");
        const real_t ee = step.num(r, "energy_MeV");
        em::UrbanDebug<real_t> d1{}, d2{}, d3{}, d4{};
        Philox<real_t> q1(17u, unsigned(r)), q2a(17u, unsigned(r));
        Philox<real_t> q3(17u, unsigned(r)), q4(17u, unsigned(r));
        em::urban_sample_cos_theta<real_t, Philox<real_t>>(
            mm2, cc, real_t(0.5) * ts, ee, ee, l0, m2, q2, false, ts, q1, real_t(-1), &d1);
        em::urban_sample_cos_theta<real_t, Philox<real_t>>(
            mm2, cc, real_t(0.02) * ts, ee, ee, l0, m2, q2, false, ts, q2a, real_t(-1), &d2);
        em::urban_sample_cos_theta<real_t, Philox<real_t>>(
            mm2, cc, real_t(0.5) * ts, ee, ee, l0, m2, q2, false, real_t(0), q3, real_t(-1),
            &d3);
        em::urban_sample_cos_theta<real_t, Philox<real_t>>(
            mm2, cc, real_t(0.02) * ts, ee, ee, l0, m2, q2, false, real_t(0), q4, real_t(-1),
            &d4);
        // `xsi >= 1.9` is how "the main branch ran" is recognised, and `branch` is not:
        // UrbanDebug is value-initialised, `branch` is set to 2 only once the fallbacks are
        // live, and every exit before that (tau under tausmall, which a 2 nm step in air's
        // 1e10 mm mean free path takes) leaves it 0 with xsi 0 - indistinguishable from the
        // main branch by that field alone. xsi is clamped at 1.9 whenever it is computed.
        if (!(d1.xsi >= real_t(1.9)) || !(d2.xsi >= real_t(1.9))) { continue; }
        if (d1.xsi == real_t(1.9)) { continue; }  // clamped: the u half is invisible here
        ++off_floor;
        // To 1e-13 rather than bit for bit, and the residual is not the branch: `lambdaeff` is
        // `true_step/tau` with `tau = true_step/lambda0`, so it is lambda0 through a divide and
        // a multiply and lands one ulp away from it at a different step length. That ulp
        // reaches xsi through coeffc4*log(lambdaeff/radlen). The unbranched difference below is
        // a factor of 25^(1/6) on u, three orders of magnitude larger.
        if (rel(d1.xsi, d2.xsi) > 1e-13) { ++u_bad; }
        // Cells where the UNBRANCHED xsi still differs between the two steps. They are what
        // makes the line above falsifiable, and they are not all of them: with the branch off,
        // u shrinks as t^(1/6) and a cell near the floor is clamped at the smaller step and so
        // agrees for the wrong reason.
        if (rel(d3.xsi, d4.xsi) > 1e-13) { ++u_live; }
      }
      std::printf("  ...and below tsmall its tail parameter is step-independent on all %d "
                  "cells the 1.9 clamp does not hide (%d of them would differ without it): "
                  "%s\n", off_floor, u_live, u_bad ? "NO" : "yes");
      if (u_bad != 0 || off_floor == 0 || u_live == 0) {
        std::printf("    %d disagree, %d unclamped, %d sensitive\n", u_bad, off_floor, u_live);
        ++bad;
      }
    }
    if (bad) { ++fails; }
  }

  // ---------------------------------------------------------------- 8. computed == tabulated
  //
  // NOT PHYSICS, AND IT COST A DEVICE RUN AND FIVE ENGINE BUILDS. `step_hadron` COMPUTES the
  // per-material coefficients rather than reading `s.msc->coeffs[mat]`, which holds exactly
  // these numbers for exactly these materials. The reason is in `em::UrbanCoeffs`: the struct
  // is seventeen doubles, so in that array every odd entry begins 8 bytes off a 16-byte
  // boundary, nvcc reads a copy of one with `ld.global.v2.f64`, and the run ends in
  //
  //     CUDA error misaligned address at src/host/transport_run.cuh:89
  //
  // on the alpha's first step - while the 2,000,000-event gamma run is fine, because
  // `step_lepton` binds a reference to the same field and always has. Both ways of fixing THAT
  // (padding the struct, binding a reference) take ptxas down on `transport_run.cu`.
  //
  // So the invariant that has to hold is the one asserted here: the computed coefficients and
  // the tabulated ones are the same numbers, bit for bit, on every material. If they ever stop
  // being, the ion and the electron are scattering in two different materials of the same name.
  {
    em::UrbanTable<real_t> tab{};
    em::build_urban_table<real_t>(mats, data::kNumMaterials, tab);
    int bad = 0;
    for (int m = 0; m < data::kNumMaterials; ++m) {
      const auto a = em::urban_coeffs(mats[m]);
      const auto& b = tab.coeffs[m];
      const real_t* pa = reinterpret_cast<const real_t*>(&a);
      const real_t* pb = reinterpret_cast<const real_t*>(&b);
      for (std::size_t k = 0; k < sizeof(a) / sizeof(real_t); ++k) {
        if (pa[k] != pb[k]) { ++bad; }
      }
    }
    std::printf("\n== 8. the coefficients step_hadron computes against the ones the electron's "
                "table holds ==\n");
    std::printf("  %d materials x %zu fields, %d differ; sizeof(UrbanCoeffs) %zu bytes "
                "(%zu mod 16, which is why they are computed)\n",
                data::kNumMaterials, sizeof(em::UrbanCoeffs<real_t>) / sizeof(real_t), bad,
                sizeof(em::UrbanCoeffs<real_t>), sizeof(em::UrbanCoeffs<real_t>) % 16);
    if (bad != 0) { ++fails; }
  }

  std::printf("\n%s (%d failures)\n", fails ? "FAILED" : "PASSED", fails);
  return fails ? 1 : 0;
}
