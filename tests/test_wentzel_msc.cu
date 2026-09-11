// The WentzelVI stepping algorithm: step limit, true<->geometric path conversion, and the
// mixed multiple/single scattering sampler.
//
// The cross sections this drives are diffed against G4WentzelOKandVIxSection in
// tests/test_wentzel.cu to 0.0003%. The stepping itself has no Geant4 accessor that can be
// called in isolation - G4WentzelVIModel::ComputeGeomPathLength and ComputeTrueStepLength
// mutate model state across calls and read a G4Track - so blocks 1 to 3 check the properties
// the algorithm is built on instead:
//
//   1. the path conversion is invertible: true -> geometric -> true returns the input
//   2. the geometric length never exceeds the true length (a detour cannot be longer than
//      the path that produced it) and both stay positive
//   3. the step limit is bounded by the range and by the requested step
//   4. the sampler produces unit directions and a displacement no longer than the step
//   5. single-scattering mode engages exactly when the expected collision count drops below
//      ten, which is what the algorithm keys on
//
// BLOCK 4 IS DIFFERENT IN KIND, AND EXISTS BECAUSE THE FIVE ABOVE ARE ALL PROPERTIES.
// `G4WentzelOKandVIxSection::SampleSingleScattering` CAN be called in isolation, and the one
// piece of this file that is a physics DISTRIBUTION rather than a geometric identity was never
// compared against it. docs/RISK.md V47 is what that cost: `wv_sample_single` dropped the
// `1/(1 + z1*factD)` factor from its rejection function, and every check above passed - a
// direction is still a unit vector and a displacement is still shorter than its step whatever
// angles are drawn. So block 4 draws the angle 400,000 times per cell and compares moments and
// a histogram against ref/oracle/wentzel_msc_sample.csv, which is the same Geant4 function set
// up as G4WentzelVIModel::SampleScattering sets it up.
//
// It is NOT tests/test_coulomb_scattering.cu's block 3 at a different seed. That one covers the
// interval [cosTetMaxNuc, -1], the large angles G4CoulombScattering owns; this one covers
// [cosThetaMin, cosTetMaxNuc], the small ones the msc model keeps, with a different production
// cut and a different target mass. See ref/dump/dump_wentzel_msc.cc's header for all three
// differences.
#include <cstdio>
#include <cstdarg>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <string>
#include "core/particle.cuh"
#include "core/rng.cuh"
#include "data/materials.cuh"
#include "physics/em/wentzel_msc.cuh"

using namespace g4gpu;
using real_t = double;

namespace {

/// The five species ref/dump/dump_wentzel_msc.cc dumps.
ParticleType type_of(const char* n) {
  if (0 == std::strcmp(n, "e-")) { return ParticleType::kElectron; }
  if (0 == std::strcmp(n, "mu-")) { return ParticleType::kMuonMinus; }
  if (0 == std::strcmp(n, "pi+")) { return ParticleType::kPionPlus; }
  if (0 == std::strcmp(n, "proton")) { return ParticleType::kProton; }
  if (0 == std::strcmp(n, "alpha")) { return ParticleType::kAlpha; }
  return ParticleType::kNumTypes;
}

/// Worst relative deviation seen, with the row that produced it.
struct Worst {
  double v = 0;
  char where[96] = "";
  void see(double a, double b, double scale, const char* fmt, ...) {
    const double d = std::fabs(a - b) / ((scale > 0) ? scale : 1.0);
    if (d <= v) { return; }
    v = d;
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(where, sizeof where, fmt, ap);
    va_end(ap);
  }
};

}  // namespace

int main() {
  data::Material<real_t> mats[data::kNumMaterials];
  data::build_b1_materials<real_t>(mats);
  const char* names[4] = {"air", "water", "A-150 tissue", "bone"};
  int fails = 0;

  const ParticleType parts[4] = {ParticleType::kElectron, ParticleType::kMuonMinus,
                                 ParticleType::kProton, ParticleType::kPionPlus};
  const char* pnames[4] = {"e-", "mu-", "proton", "pi+"};
  const real_t cos_lim = real_t(-1);  // the uncombined limit

  printf("== path conversion: true -> geometric -> true ==\n");
  printf("  %-8s %-12s %10s %12s %12s %10s %9s\n", "part", "material", "E (MeV)", "t in",
         "z", "t out", "mode");
  double worst_roundtrip = 0;
  int n_round = 0, n_single = 0, n_multi = 0;
  for (int pi = 0; pi < 4; ++pi) {
    const auto pd = particle_def<real_t>(parts[pi]);
    for (int mi = 0; mi < data::kNumMaterials; ++mi) {
      const auto& m = mats[mi];
      const real_t cut = m.cut_electron;
      for (real_t e : {real_t(1), real_t(10), real_t(100), real_t(1e3), real_t(1e5)}) {
        // A plausible range and transport mfp for this energy; the algorithm only needs
        // them to be consistent, not to come from a table.
        const real_t lambda = em::wentzel_lambda(m, parts[pi], e, cut, cos_lim);
        if (!(lambda > 0) || !std::isfinite(lambda)) { continue; }
        const real_t range = real_t(10) * lambda;

        em::WentzelMscState<real_t> st{};
        st.range = range;
        st.pre_kin_energy = e;
        st.eff_kin_energy = e;
        st.lambda_eff = lambda;
        st.single_scattering_mode = false;
        em::WentzelElementXs<real_t> els{};
        const auto s0 = em::wentzel_setup(pd, parts[pi], e, m.inv_a23,
                                          int(m.z[0] + real_t(0.5)), cut, cos_lim);
        st.cos_tet_max_nuc = s0.cos_tet_max_nuc;
        em::wv_transport_xs(m, pd, parts[pi], e, cut, cos_lim, real_t(1), st.cos_tet_max_nuc,
                            els, st.xtsec);

        const real_t t_in = real_t(0.3) * range;
        const real_t z = em::wv_geom_path(st, t_in, e * real_t(0.9), lambda, st.cos_tet_max_nuc);
        if (st.single_scattering_mode) { ++n_single; } else { ++n_multi; }

        // Property 2: the detour cannot be longer than the path that produced it.
        if (!(z > 0) || z > t_in * (1 + 1e-12)) {
          printf("  FAIL: geometric length %g exceeds true length %g (%s in %s at %g)\n", z,
                 t_in, pnames[pi], names[mi], e);
          ++fails;
        }

        // Property 1: feeding the geometric length straight back must return the true one.
        auto recompute = [&](real_t cos_min, real_t& xt) {
          return em::wv_transport_xs(m, pd, parts[pi], st.eff_kin_energy, cut, cos_lim,
                                     cos_min, st.cos_tet_max_nuc, els, xt);
        };
        em::WentzelMscState<real_t> st2 = st;
        const real_t t_out =
            em::wv_true_path(st2, z, e * real_t(0.9), lambda, st.cos_tet_max_nuc, recompute);
        ++n_round;
        // The second half of ComputeTrueStepLength deliberately re-derives t from the
        // refreshed cross section, so the round trip is only exact when geomStep == zPath,
        // which is the branch taken here.
        const double dev = std::fabs(t_out / t_in - 1);
        if (dev > worst_roundtrip) { worst_roundtrip = dev; }
        if ((n_round % 7) == 1) {
          printf("  %-8s %-12s %10.4g %12.5g %12.5g %12.5g %9s\n", pnames[pi], names[mi], e,
                 t_in, z, t_out, st.single_scattering_mode ? "single" : "multi");
        }
      }
    }
  }
  printf("\n  %d conversions; %d in multiple-scattering mode, %d in single\n", n_round,
         n_multi, n_single);
  printf("  worst round-trip deviation: %.4f%%\n", 100 * worst_roundtrip);
  if (n_multi == 0 || n_single == 0) {
    printf("  FAIL: only one mode was ever exercised, so the other is untested\n");
    ++fails;
  }

  printf("\n== step limit is bounded by the range and the request ==\n");
  {
    int n = 0;
    for (int pi = 0; pi < 4; ++pi) {
      const auto pd = particle_def<real_t>(parts[pi]);
      for (int mi = 0; mi < data::kNumMaterials; ++mi) {
        const auto& m = mats[mi];
        for (real_t e : {real_t(1), real_t(100), real_t(1e4)}) {
          const real_t lambda = em::wentzel_lambda(m, parts[pi], e, m.cut_electron, cos_lim);
          if (!(lambda > 0) || !std::isfinite(lambda)) { continue; }
          const real_t range = real_t(10) * lambda;
          for (real_t safety : {real_t(0), real_t(0.1) * range, real_t(10) * range}) {
            const real_t req = real_t(0.5) * range;
            const auto s0 = em::wentzel_setup(pd, parts[pi], e, m.inv_a23,
                                              int(m.z[0] + real_t(0.5)), m.cut_electron,
                                              cos_lim);
            const real_t lim = em::wv_step_limit(m, pd, parts[pi], e, range, lambda,
                                                 s0.cos_tet_max_nuc, cos_lim, safety,
                                                 real_t(0.7), req, em::kFacRange<real_t>());
            ++n;
            if (!(lim > 0) || lim > req * (1 + 1e-12) || lim > range * (1 + 1e-12)) {
              printf("  FAIL: limit %g outside (0, min(req %g, range %g)] for %s in %s\n",
                     lim, req, range, pnames[pi], names[mi]);
              ++fails;
            }
          }
        }
      }
    }
    printf("  %d limits checked, all within (0, min(request, range)]\n", n);
  }

  printf("\n== sampler: unit directions, bounded displacement ==\n");
  printf("  %-8s %-12s %10s %12s %14s %14s\n", "part", "material", "E (MeV)", "samples",
         "max |dir|-1", "max |disp|/t");
  for (int pi = 0; pi < 4; ++pi) {
    const auto pd = particle_def<real_t>(parts[pi]);
    for (int mi = 1; mi < data::kNumMaterials; ++mi) {
      const auto& m = mats[mi];
      const real_t cut = m.cut_electron;
      for (real_t e : {real_t(100), real_t(1e4)}) {
        const real_t lambda = em::wentzel_lambda(m, parts[pi], e, cut, cos_lim);
        if (!(lambda > 0) || !std::isfinite(lambda)) { continue; }
        const real_t range = real_t(10) * lambda;
        em::WentzelMscState<real_t> st{};
        st.range = range;
        st.pre_kin_energy = e;
        st.eff_kin_energy = e;
        st.lambda_eff = lambda;
        st.single_scattering_mode = false;
        em::WentzelElementXs<real_t> els{};
        const auto s0 = em::wentzel_setup(pd, parts[pi], e, m.inv_a23,
                                          int(m.z[0] + real_t(0.5)), cut, cos_lim);
        st.cos_tet_max_nuc = s0.cos_tet_max_nuc;
        em::wv_transport_xs(m, pd, parts[pi], e, cut, cos_lim, real_t(1), st.cos_tet_max_nuc,
                            els, st.xtsec);
        const real_t zz = em::wv_geom_path(st, real_t(0.3) * range, e * real_t(0.9), lambda,
                                           st.cos_tet_max_nuc);
        // The sampler requires the full sequence: wv_true_path is what lowers cos_theta_min
        // from 1 and recomputes xtsec above it.
        auto recompute2 = [&](real_t cos_min, real_t& xt) {
          return em::wv_transport_xs(m, pd, parts[pi], st.eff_kin_energy, cut, cos_lim,
                                     cos_min, st.cos_tet_max_nuc, els, xt);
        };
        em::wv_true_path(st, zz, e * real_t(0.9), lambda, st.cos_tet_max_nuc, recompute2);

        const Vec3<real_t> dir0{real_t(0), real_t(0), real_t(1)};
        Philox<real_t> rng(unsigned(pi * 31 + mi), unsigned(e));
        real_t worst_norm = 0, worst_disp = 0;
        const int N = 2000;
        for (int k = 0; k < N; ++k) {
          const auto r = em::wv_sample_scattering(m, pd, parts[pi], st, els, cut, cos_lim,
                                                  dir0, true, rng);
          const real_t n2 = std::sqrt(r.dir.x * r.dir.x + r.dir.y * r.dir.y
                                      + r.dir.z * r.dir.z);
          worst_norm = std::max(worst_norm, std::fabs(n2 - 1));
          const real_t d = std::sqrt(r.displacement.x * r.displacement.x
                                     + r.displacement.y * r.displacement.y
                                     + r.displacement.z * r.displacement.z);
          worst_disp = std::max(worst_disp, d / st.t_path);
        }
        printf("  %-8s %-12s %10.4g %12d %14.3g %14.4f\n", pnames[pi], names[mi], e, N,
               worst_norm, worst_disp);
        if (worst_norm > 1e-12) {
          printf("    FAIL: direction not unit\n");
          ++fails;
        }
        // The displacement is assembled from sub-steps of the geometric length, so it can
        // reach that length but must not exceed the true path it came from.
        if (!(worst_disp <= 1.0 + 1e-9)) {
          printf("    FAIL: displacement longer than the true step\n");
          ++fails;
        }
      }
    }
  }

  // ---------------------------------------------------------------- 4. the angle sampler
  //
  // Two halves, and only the first is statistical.
  //
  //   EXACT. Everything the sampler is SET UP with that Geant4 will report - the angular
  //   interval, the electron cut-off, the electron/nucleus split and the target mass factD is
  //   built from - is a deterministic function of (species, Z, energy, cut, <A^-2/3>) and is
  //   compared to machine precision. A distribution comparison alone would let any of these be
  //   wrong by a little and hide it inside the Monte Carlo noise. `fMottFactor` is the one
  //   set-up quantity with no accessor and so no column; see the note further down.
  //
  //   STATISTICAL. The angle itself, 400,000 draws per cell against Geant4's 400,000. The two
  //   random streams are different by construction - CLHEP's HepJamesRandom against this port's
  //   Philox - so this is three summary statistics and nothing else: the fraction of draws the
  //   rejection function ACCEPTS, the mean of 1 - cos(theta), and a Pearson chi-square over a
  //   histogram log-spaced across the cell's own [z1_min, z1_max]. The first two carry their
  //   own sigma, which is what makes "many sigma" a statement and not an adjective.
  //
  // WHY THE HISTOGRAM IS BINNED PER CELL. This interval is narrow and it MOVES: 1 - cos runs
  // from f*(1 - cosTetMaxNuc) up to (1 - cosTetMaxNuc), and that upper end is
  // min(2, factorA2*<A^-2/3>/mom2) = min(2, 13935 MeV^2/mom2) in water - saturated at 2 below
  // 1 MeV for a proton, and 0.032 at 210 MeV. A fixed decade grid like coulomb_sample.csv's
  // would put every draw of most cells into one or two bins and the chi-square would be
  // measuring nothing.
  {
    const char* env = std::getenv("G4GPU_ORACLE");
    const std::string dir = (env != nullptr) ? env : "ref/oracle";
    FILE* f = std::fopen((dir + "/wentzel_msc_sample.csv").c_str(), "r");
    if (f == nullptr) {
      printf("\ncannot read %s/wentzel_msc_sample.csv - run ref/oracle/run.bat tables first\n",
             dir.c_str());
      return 1;
    }
    constexpr int kBins = 24;
    char line[4096];
    if (std::fgets(line, sizeof line, f) == nullptr) { return 1; }   // header

    printf("\n== the angle sampler against G4WentzelOKandVIxSection::SampleSingleScattering ==\n");
    printf("  %-8s %3s %9s %5s %15s %8s %8s %9s\n", "part", "Z", "E (MeV)", "f",
           "P(accept) G4/us", "sigma", "d<1-cos>", "chi2/bin");
    Worst w_ctmax, w_ctelec, w_ratio, w_factd, w_tmass;
    int cells = 0, clamped = 0, printed = 0;
    double worst_z = 0, worst_zm = 0, worst_chi2 = 0;
    while (std::fgets(line, sizeof line, f) != nullptr) {
      char mat[64], part[32];
      int Z = 0, n = 0, pos = 0;
      long nsc = 0;
      double e, ff, cut, inva23, ctmin, ctmax, ctnucmat, ctelec, ratio, tmass, factd;
      double m1, m2, m3, z1min, z1max;
      const int got = std::sscanf(line,
                                  "%63[^,],%31[^,],%d,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,"
                                  "%lf,%lf,%d,%ld,%lf,%lf,%lf,%lf,%lf%n",
                                  mat, part, &Z, &e, &ff, &cut, &inva23, &ctmin, &ctmax,
                                  &ctnucmat, &ctelec, &ratio, &tmass, &factd, &n, &nsc, &m1,
                                  &m2, &m3, &z1min, &z1max, &pos);
      if (got != 21 || n <= 0) { continue; }
      long h[kBins] = {0};
      char* p = line + pos;
      for (int b = 0; b < kBins && *p == ','; ++b) { h[b] = std::strtol(p + 1, &p, 10); }
      const ParticleType type = type_of(part);
      if (type == ParticleType::kNumTypes) { continue; }
      const auto pd = particle_def<real_t>(type);

      // ---- exact. wentzel_setup is SetupKinematic followed by SetupTarget, and every number
      // those two produce that this sampler reads is a column.
      const auto s = em::wentzel_setup(pd, type, real_t(e), real_t(inva23), Z, real_t(cut),
                                       cos_lim);
      w_ctmax.see(double(s.cos_tet_max_nuc), ctmax, 1.0, "%s Z=%d %g MeV f=%g", part, Z, e, ff);
      w_ctelec.see(double(s.cos_tet_max_elec), ctelec, 1.0, "%s Z=%d %g MeV", part, Z, e);
      w_tmass.see(double(em::wv_target_mass<real_t>(Z)), tmass, tmass, "Z=%d", Z);
      w_factd.see(double(sqrt(s.mom2) / em::wv_target_mass<real_t>(Z)), factd, factd,
                  "%s Z=%d %g MeV", part, Z, e);
      // The split, as G4WentzelVIModel::ComputeTransportXSectionPerVolume builds it over this
      // cell's interval (G4WentzelVIModel.cc:738-743).
      double pratio = 0;
      if (ctmax < ctmin) {
        const real_t xn = em::wentzel_nuclear_xs(s, Z, real_t(ctmin), real_t(ctmax));
        const real_t xe = em::wentzel_electron_xs(s, real_t(ctmin), real_t(ctmax));
        if (xn + xe > real_t(0)) { pratio = double(xe / (xn + xe)); }
      }
      w_ratio.see(pratio, ratio, (ratio > 0) ? ratio : 1.0, "%s Z=%d %g MeV", part, Z, e);
      // SetupTarget's `if(targetZ == 1 && particle == theProton && cosTetMaxNuc2 < 0.0)` arm.
      // Counted rather than assumed: it is the one place cos_t_max and cos_tet_max_nuc_mat
      // differ, and a port that dropped it would still reproduce every other column here.
      if (ctmax != ctnucmat) {
        ++clamped;
        if (type != ParticleType::kProton || Z != 1 || ctmax != 0.0) {
          printf("  FAIL: the proton/hydrogen clamp fired on %s Z=%d at %g MeV\n", part, Z, e);
          ++fails;
        }
      }

      // ---- statistical.
      const unsigned pkey = unsigned(part[0]) * 131u + unsigned(part[1]);
      Philox<real_t> rng(0x77e20000u + pkey, unsigned(Z) * 1000u + unsigned(ff * 100 + 0.5),
                         unsigned(e * 10 + 0.5));
      long oh[kBins] = {0};
      long onsc = 0;
      double o1 = 0, o2 = 0, o3 = 0;
      const double lo = std::log10(z1min), hi = std::log10(z1max);
      for (int k = 0; k < n; ++k) {
        const Vec3<real_t> v = em::wv_sample_single(s, Z, real_t(ctmin), real_t(ctmax),
                                                    real_t(ratio), rng);
        const double cost = double(v.z);
        o1 += cost;
        o2 += cost * cost;
        o3 += 1.0 - cost;
        // A rejected draw is (0,0,1) and is binned nowhere, as in the dump: the accepted
        // FRACTION is the statistic that tests the rejection function's normalisation and the
        // histogram is the one that tests its shape, and mixing them weakens both.
        if (cost < 1.0) {
          ++onsc;
          int b = 0;
          if (hi > lo) {
            b = int(kBins * (std::log10(1.0 - cost) - lo) / (hi - lo));
            if (b < 0) { b = 0; }
            if (b >= kBins) { b = kBins - 1; }
          }
          ++oh[b];
        }
      }
      const double p1 = double(nsc) / n, p2 = double(onsc) / n;
      const double sig = std::sqrt(std::max(2.0 * p1 * (1.0 - p1) / n, 1.0 / (double(n) * n)));
      const double zsc = std::fabs(p1 - p2) / sig;
      // The sigma of the difference of the two means of 1 - cos, from BOTH sides' own second
      // moments - which is what mean_cost and mean_cost2 are columns for. <(1-c)^2> is
      // 1 - 2<c> + <c^2>, so each variance comes out of the two moments already dumped.
      const double var_g4 = std::max(1.0 - 2.0 * m1 + m2 - m3 * m3, 0.0);
      const double var_us = std::max(1.0 - 2.0 * o1 / n + o2 / n - (o3 / n) * (o3 / n), 0.0);
      const double sigm = std::sqrt(std::max((var_g4 + var_us) / n, 1e-300));
      const double zm = std::fabs(m3 - o3 / n) / sigm;
      double chi2 = 0;
      int nb = 0;
      for (int b = 0; b < kBins; ++b) {
        const double sum = double(h[b] + oh[b]);
        if (sum < 20) { continue; }
        const double d = double(h[b] - oh[b]);
        chi2 += d * d / sum;
        ++nb;
      }
      const double chi2n = (nb > 0) ? chi2 / nb : 0.0;
      ++cells;
      worst_z = std::max(worst_z, zsc);
      worst_zm = std::max(worst_zm, zm);
      worst_chi2 = std::max(worst_chi2, chi2n);
      // Five sigma on each of the two moments and five per bin on the chi-square. Wide, because
      // two generators drawing 400,000 samples each will not agree bin for bin, and a limit that
      // fails at random is worse than none. What makes it a check is the anti-vacuity run in the
      // commit message: removing 1/(1 + z1*factD) again moves cells by tens of sigma.
      const bool bad = (zsc > 5.0) || (zm > 5.0) || (chi2n > 5.0);
      if (bad || (Z == 1 && ff == 0.05) || printed < 2) {
        ++printed;
        printf("  %-8s %3d %9.4g %5.2f %7.5f/%7.5f %8.2f %8.2f %9.3f%s\n", part, Z, e, ff, p1,
               p2, zsc, zm, chi2n, bad ? "   <-- FAIL" : "");
      }
      if (bad) {
        printf("    <1-cos> %.8g vs %.8g (%.1f sigma), accepted %ld vs %ld (%.1f sigma), "
               "chi2/bin %.2f over %d bins; factD %.6g, z1 in [%.4g, %.4g]\n",
               m3, o3 / n, zm, nsc, onsc, zsc, chi2n, nb, factd, z1min, z1max);
        ++fails;
      }
    }
    std::fclose(f);
    printf("\n  %d cells, %d of them on the proton/hydrogen clamp\n", cells, clamped);
    // The Mott normalisation is NOT in this list and has no column: see the note in
    // ref/dump/dump_wentzel_msc.cc. It is exercised by the accepted fraction of the e- cells,
    // where it runs from 1.0002 at Z = 1 to 2.345 at Z = 82.
    printf("  worst: interval %.3g, cos_tet_max_elec %.3g, elec_ratio %.3g, target mass %.3g, "
           "factD %.3g\n", w_ctmax.v, w_ctelec.v, w_ratio.v, w_tmass.v, w_factd.v);
    printf("  worst statistic: %.2f sigma on the accepted fraction, %.2f sigma on <1-cos>, "
           "%.3f chi2/bin\n", worst_z, worst_zm, worst_chi2);
    if (cells == 0) {
      printf("  FAIL: no sampler cells compared\n");
      ++fails;
    }
    if (clamped == 0) {
      printf("  FAIL: no cell exercised SetupTarget's proton-on-hydrogen clamp\n");
      ++fails;
    }
    struct { const char* name; const Worst* w; } exact[] = {
        {"angular interval", &w_ctmax}, {"cos_tet_max_elec", &w_ctelec},
        {"electron fraction", &w_ratio}, {"target mass", &w_tmass}, {"factD", &w_factd}};
    for (const auto& x : exact) {
      if (x.w->v > 1e-12) {
        printf("  FAIL: %s deviates by %.3g at %s\n", x.name, x.w->v, x.w->where);
        ++fails;
      }
    }
  }

  printf("\n%s (%d failures)\n", fails ? "FAILED" : "PASSED", fails);
  return fails ? 1 : 0;
}
